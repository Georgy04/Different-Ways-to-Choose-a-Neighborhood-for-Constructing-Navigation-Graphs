import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import struct
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.special import gamma
from typing import List, Dict, Tuple, Set, Optional
import hashlib
import math
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D

# ================== КОНФИГУРАЦИЯ ==================
DATASET_TYPE = "siftsmall"  # "siftsmall" или "synthetic"
FIXED_DIMENSION = 8  # Фиксированная размерность для экспериментов
LOCAL_VIEW_SIZES = [25]  # Вариации локального представления
SW_VIEW_SIZES = [20]  # Вариации small-world представления
N_PEERS = 10000  # Уменьшено для ускорения экспериментов
N_QUERIES = 100
CYCLE_LIMIT = 30  # Уменьшено для ускорения экспериментов
MAX_THREADS = os.cpu_count()
R_RAYS = 1000
MAX_LAMBDA = 100
APPROX_KNN = False


# ==================================================

def get_data_path(filename):
    dataset_dir = 'datasets/siftsmall' if DATASET_TYPE == "siftsmall" else 'datasets/synthetic'
    return os.path.join(os.path.dirname(__file__), dataset_dir, filename)


def read_fvecs(filename):
    filepath = get_data_path(filename)
    with open(filepath, 'rb') as f:
        vecs = []
        while True:
            dim_b = f.read(4)
            if not dim_b: break
            dim = struct.unpack('i', dim_b)[0]
            vec = struct.unpack('f' * dim, f.read(4 * dim))
            vecs.append(np.array(vec, dtype=np.float32))
        return np.array(vecs)


def read_ivecs(filename):
    filepath = get_data_path(filename)
    with open(filepath, 'rb') as f:
        vecs = []
        while True:
            dim_b = f.read(4)
            if not dim_b: break
            dim = struct.unpack('i', dim_b)[0]
            vec = struct.unpack('i' * dim, f.read(4 * dim))
            vecs.append(np.array(vec, dtype=np.int32))
        return np.array(vecs)


def generate_synthetic_data(n_points: int, dim: int = 3) -> np.ndarray:
    """Генерирует синтетические данные в гиперкубе [0, 1]^dim"""
    data = np.random.rand(n_points, dim).astype(np.float32)
    return data / np.linalg.norm(data, axis=1, keepdims=True)


def unit_ball_volume(dim: int) -> float:
    """Объем единичного шара в d-мерном пространстве"""
    return np.pi ** (dim / 2) / gamma(dim / 2 + 1)


class RayNetNode:
    def __init__(self, node_id: int, point: np.ndarray, dim: int,
                 local_size: int, sw_size: int):
        self.id = node_id
        self.point = point
        self.dim = dim
        self.c = local_size
        self.sw_view_max = sw_size
        self.R = min(R_RAYS, 300)
        self.rays = self.generate_rays()
        self.local_view: List['RayNetNode'] = []
        self.sw_view: List[Tuple['RayNetNode', float]] = []
        self.voronoi_volume = float('inf')
        self.known_peers: Set[int] = set()
        self.cache: Dict[str, float] = {}
        self.distance_cache: Dict[Tuple[int, int], float] = {}
        self.lambda_cache: Dict[Tuple[tuple, int], float] = {}

    def distance_to(self, other: 'RayNetNode') -> float:
        key = (min(self.id, other.id), max(self.id, other.id))
        if key not in self.distance_cache:
            self.distance_cache[key] = np.linalg.norm(self.point - other.point)
        return self.distance_cache[key]

    def generate_rays(self) -> np.ndarray:
        rays = np.random.normal(size=(self.R, self.dim))
        norms = np.linalg.norm(rays, axis=1, keepdims=True)
        return rays / np.where(norms > 1e-10, norms, 1)

    def compute_lambda(self, ray: np.ndarray, peer: 'RayNetNode') -> float:
        cache_key = (tuple(ray), peer.id)
        if cache_key in self.lambda_cache:
            return self.lambda_cache[cache_key]

        diff = peer.point - self.point
        projection = np.dot(ray, diff)

        if projection <= 1e-10:
            self.lambda_cache[cache_key] = float('inf')
            return float('inf')

        lambda_val = np.dot(diff, diff) / (2 * projection)
        lambda_val = min(lambda_val, MAX_LAMBDA)
        self.lambda_cache[cache_key] = lambda_val
        return lambda_val

    def estimate_voronoi_volume(self, config: List['RayNetNode']) -> float:
        if not config:
            return float('inf')

        config_ids = tuple(sorted(n.id for n in config))
        cache_key = hashlib.md5(str(config_ids).encode()).hexdigest()

        if cache_key in self.cache:
            return self.cache[cache_key]

        # Используем 30% лучей для баланса скорости/точности
        sample_size = max(20, int(self.R * 0.3))
        indices = np.random.choice(self.R, sample_size, replace=False)
        sampled_rays = self.rays[indices]

        points = np.array([n.point for n in config])
        diff = points - self.point
        norms_sq = np.sum(diff ** 2, axis=1)

        projections = sampled_rays @ diff.T
        valid_mask = projections > 1e-10

        with np.errstate(divide='ignore', invalid='ignore'):
            lambda_vals = norms_sq[None, :] / (2 * projections)
            lambda_vals[~valid_mask] = np.inf

        min_lambda = np.min(np.where(lambda_vals > MAX_LAMBDA, MAX_LAMBDA, lambda_vals), axis=1)
        valid_lambda = min_lambda[np.isfinite(min_lambda) & (min_lambda > 1e-10)]

        if len(valid_lambda) == 0:
            volume = float('inf')
        else:
            # Корректная оценка объема согласно статье
            sum_ld = np.sum(valid_lambda ** self.dim)
            volume = unit_ball_volume(self.dim) * (sum_ld / len(valid_lambda))

        self.cache[cache_key] = volume
        return volume

    def update_improved(self, candidate_view: List['RayNetNode']) -> bool:
        """Улучшенный алгоритм с добавлением и заменой соседей"""
        candidates = list(set(candidate_view) - set(self.local_view))
        if not candidates:
            return False

        current_volume = self.voronoi_volume
        improved = False

        # 1. Попробуем добавить новых кандидатов
        if len(self.local_view) < self.c:
            for candidate in candidates:
                new_view = self.local_view + [candidate]
                new_volume = self.estimate_voronoi_volume(new_view)
                if new_volume < current_volume:
                    self.local_view = new_view
                    self.voronoi_volume = new_volume
                    self.known_peers.add(candidate.id)
                    improved = True
                    current_volume = new_volume
                    break  # Добавляем только одного кандидата за раз

        # 2. Попробуем заменить существующих соседей
        else:
            # Выберем 5 случайных кандидатов и 5 случайных текущих соседей
            num_samples = min(5, len(candidates), len(self.local_view))
            if num_samples > 0:
                candidates_sample = random.sample(candidates, num_samples)
                neighbors_sample = random.sample(self.local_view, num_samples)

                for candidate in candidates_sample:
                    for old_neighbor in neighbors_sample:
                        # Создаем копию текущего представления
                        new_view = self.local_view.copy()

                        # Найдем индекс старого соседа по ID
                        idx = next((i for i, n in enumerate(new_view) if n.id == old_neighbor.id), None)
                        if idx is None:
                            continue  # Пропускаем, если не нашли

                        # Заменяем старого соседа на кандидата
                        new_view[idx] = candidate
                        new_volume = self.estimate_voronoi_volume(new_view)

                        if new_volume < current_volume:
                            self.local_view = new_view
                            self.voronoi_volume = new_volume
                            self.known_peers.add(candidate.id)
                            improved = True
                            current_volume = new_volume
                            break  # Выходим из внутреннего цикла после первой успешной замены
                    if improved:
                        break  # Выходим из внешнего цикла после успешной замены

        return improved

    def add_sw_neighbor(self, neighbor: 'RayNetNode'):
        if neighbor.id == self.id or neighbor.id in self.known_peers:
            return

        if len(self.sw_view) >= self.sw_view_max:
            self.sw_view.sort(key=lambda x: x[1], reverse=True)
            self.sw_view.pop()

        dist = self.distance_to(neighbor)
        self.sw_view.append((neighbor, dist))
        self.known_peers.add(neighbor.id)

    def greedy_routing(self, target_point: np.ndarray, max_hops: int = 50) -> Tuple[List['RayNetNode'], 'RayNetNode']:
        current = self
        path = [current]
        visited = {self.id}

        for _ in range(max_hops):
            candidates = set(current.local_view)
            candidates.update(n for n, _ in current.sw_view)
            candidates = [n for n in candidates if n.id not in visited]

            if not candidates:
                break

            best_neighbor = None
            best_dist = float('inf')

            for neighbor in candidates:
                dist = np.linalg.norm(neighbor.point - target_point)
                if dist < best_dist:
                    best_dist = dist
                    best_neighbor = neighbor

            if best_dist >= np.linalg.norm(current.point - target_point):
                break

            current = best_neighbor
            visited.add(current.id)
            path.append(current)

        return path, path[-1]


def reduce_dimension(data: np.ndarray, target_dim: int) -> np.ndarray:
    if data.shape[1] <= target_dim:
        return data

    pca = PCA(n_components=target_dim)
    return pca.fit_transform(data)


def initialize_network(data: np.ndarray, dim: int, local_size: int, sw_size: int) -> List[RayNetNode]:
    print(f"Initializing network with {len(data)} nodes in {dim}D | Local: {local_size} | SW: {sw_size}...")
    nodes = [RayNetNode(i, point, dim, local_size, sw_size) for i, point in enumerate(data)]
    data_points = np.array([node.point for node in nodes])

    # 1. Инициализация локальных связей
    k_local = min(local_size, len(data) - 1)
    if k_local > 0:
        knn = NearestNeighbors(algorithm='brute', n_neighbors=k_local + 1)
        knn.fit(data_points)
        _, indices = knn.kneighbors(data_points)

        for i, node in enumerate(tqdm(nodes, desc="Initializing local views")):
            neighbors = [nodes[idx] for idx in indices[i][1:1 + k_local]]
            node.local_view = neighbors
            node.known_peers.update(n.id for n in neighbors)
            node.voronoi_volume = node.estimate_voronoi_volume(neighbors)

    # 2. Инициализация small-world связей
    for i, node in enumerate(tqdm(nodes, desc="Adding small-world neighbors")):
        candidates = [n for n in nodes if n.id != node.id and n.id not in node.known_peers]

        if not candidates:
            continue

        distances = np.array([node.distance_to(n) for n in candidates])
        distances = np.clip(distances, 1e-9, None)

        log_probs = -dim * np.log(distances)
        log_probs -= np.max(log_probs)
        probs = np.exp(log_probs)
        probs /= probs.sum()

        try:
            selected_indices = np.random.choice(
                len(candidates),
                size=min(sw_size, len(candidates)),
                replace=False,
                p=probs
            )
        except:
            selected_indices = np.random.choice(
                len(candidates),
                size=min(sw_size, len(candidates)),
                replace=False
            )

        for idx in selected_indices:
            node.add_sw_neighbor(candidates[idx])

    print("Network initialization complete")
    return nodes


def gossip_cycle(nodes: List[RayNetNode], cycle: int) -> int:
    updated_count = 0
    random.shuffle(nodes)

    def process_node(node):
        nonlocal updated_count
        candidates = []
        if node.sw_view:
            candidates.extend([peer for peer, _ in node.sw_view])
        if node.local_view and not candidates:
            candidates.extend(node.local_view)

        if not candidates:
            return 0

        partner = random.choice(candidates)
        candidate_view = partner.local_view

        if node.update_improved(candidate_view):
            updated_count += 1
        return 1

    if MAX_THREADS > 1:
        with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
            futures = [executor.submit(process_node, node) for node in nodes]
            for future in as_completed(futures):
                future.result()
    else:
        for node in tqdm(nodes, desc=f"Cycle {cycle}"):
            process_node(node)

    return updated_count


def evaluate_network(nodes: List[RayNetNode], queries: np.ndarray, true_neighbors: np.ndarray) -> dict:
    correct_count = 0
    total_hops = 0
    results = []

    # Собираем статистику по связям
    total_local_edges = 0
    total_sw_edges = 0

    for node in nodes:
        total_local_edges += len(node.local_view)
        total_sw_edges += len(node.sw_view)

    avg_local_edges = total_local_edges / len(nodes)
    avg_sw_edges = total_sw_edges / len(nodes)
    avg_total_edges = avg_local_edges + avg_sw_edges  # Средняя степень вершины

    def process_query(i):
        query = queries[i]
        start_node = random.choice(nodes)
        path, found_node = start_node.greedy_routing(query)
        success = 1 if found_node.id == true_neighbors[i] else 0
        hops = len(path) - 1
        return success, hops

    if MAX_THREADS > 1:
        with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
            futures = [executor.submit(process_query, i) for i in range(len(queries))]
            for future in as_completed(futures):
                success, hops = future.result()
                correct_count += success
                total_hops += hops
                results.append((success, hops))
    else:
        for i in tqdm(range(len(queries)), desc="Evaluating queries"):
            success, hops = process_query(i)
            correct_count += success
            total_hops += hops
            results.append((success, hops))

    recall_at_1 = correct_count / len(queries)
    avg_hops = total_hops / len(queries)

    return {
        'recall_at_1': recall_at_1,
        'avg_hops': avg_hops,
        'avg_local_edges': avg_local_edges,
        'avg_sw_edges': avg_sw_edges,
        'avg_total_edges': avg_total_edges,
        'results': results
    }


def run_experiment(data_dim: int, base_data: np.ndarray, query_data: np.ndarray, true_neighbors: np.ndarray,
                   local_size: int, sw_size: int) -> dict:
    # Уменьшение размерности
    if base_data.shape[1] > data_dim:
        combined = np.vstack((base_data, query_data))
        reduced = reduce_dimension(combined, data_dim)
        base_reduced = reduced[:len(base_data)]
        query_reduced = reduced[len(base_data):]

        print("Computing true neighbors...")
        knn = NearestNeighbors(n_neighbors=1, algorithm='brute')
        knn.fit(base_reduced)
        _, true_indices = knn.kneighbors(query_reduced)
        true_neighbors_reduced = true_indices.flatten()
    else:
        base_reduced = base_data
        query_reduced = query_data
        true_neighbors_reduced = true_neighbors

    # Инициализация сети
    nodes = initialize_network(base_reduced, data_dim, local_size, sw_size)

    # Начальная оценка
    print("Initial evaluation...")
    eval_results = evaluate_network(nodes, query_reduced, true_neighbors_reduced)
    print(f"Initial | Recall@1: {eval_results['recall_at_1']:.4f} | Avg Hops: {eval_results['avg_hops']:.2f}")
    print(f"Local edges: {eval_results['avg_local_edges']:.2f} | SW edges: {eval_results['avg_sw_edges']:.2f}")

    # Тренировочные циклы
    recall_history = [eval_results['recall_at_1']]
    hop_history = [eval_results['avg_hops']]
    local_edges_history = [eval_results['avg_local_edges']]
    sw_edges_history = [eval_results['avg_sw_edges']]
    total_edges_history = [eval_results['avg_total_edges']]

    for cycle in range(1, CYCLE_LIMIT + 1):
        print(f"\nStarting gossip cycle {cycle}...")
        start_time = time.time()
        updated = gossip_cycle(nodes, cycle)
        duration = time.time() - start_time

        print(f"Evaluating network after cycle {cycle}...")
        eval_results = evaluate_network(nodes, query_reduced, true_neighbors_reduced)
        recall_history.append(eval_results['recall_at_1'])
        hop_history.append(eval_results['avg_hops'])
        local_edges_history.append(eval_results['avg_local_edges'])
        sw_edges_history.append(eval_results['avg_sw_edges'])
        total_edges_history.append(eval_results['avg_total_edges'])

        print(f"Cycle {cycle:02d} | Updated: {updated} | Time: {duration:.2f}s | "
              f"Recall@1: {eval_results['recall_at_1']:.4f} | "
              f"Avg Hops: {eval_results['avg_hops']:.2f}")
        print(f"Local edges: {eval_results['avg_local_edges']:.2f} | SW edges: {eval_results['avg_sw_edges']:.2f}")

        # Ранний выход при сходимости
        if cycle > 2 and updated / len(nodes) < 0.05:
            print(f"Early stopping at cycle {cycle} (convergence reached)")
            break

    return {
        'dim': data_dim,
        'local_size': local_size,
        'sw_size': sw_size,
        'total_edges': local_size + sw_size,
        'final_recall_at_1': recall_history[-1],
        'final_hops': hop_history[-1],
        'final_avg_total_edges': total_edges_history[-1],
        'recall_history': recall_history,
        'hop_history': hop_history
    }


def run_dependency_analysis():
    # Загрузка данных
    if DATASET_TYPE == "siftsmall":
        print("Loading SIFTSMALL dataset...")
        base_data = read_fvecs("siftsmall_base.fvecs")[:N_PEERS]
        query_data = read_fvecs("siftsmall_query.fvecs")[:N_QUERIES]
        true_neighbors = read_ivecs("siftsmall_groundtruth.ivecs")[:N_QUERIES, 0]
    else:
        print("Generating synthetic data...")
        base_data = generate_synthetic_data(N_PEERS + 100, dim=FIXED_DIMENSION)[:N_PEERS]
        query_data = generate_synthetic_data(N_QUERIES, dim=FIXED_DIMENSION)

        print("Computing true neighbors...")
        knn = NearestNeighbors(n_neighbors=1, algorithm='brute')
        knn.fit(base_data)
        _, true_indices = knn.kneighbors(query_data)
        true_neighbors = true_indices.flatten()

    results = []
    total_experiments = len(LOCAL_VIEW_SIZES) * len(SW_VIEW_SIZES)
    completed = 0

    # Перебор всех комбинаций параметров
    for local_size in LOCAL_VIEW_SIZES:
        for sw_size in SW_VIEW_SIZES:
            completed += 1
            print(f"\n{'=' * 60}")
            print(f"Experiment {completed}/{total_experiments}: Local={local_size}, SW={sw_size}")
            print(f"{'=' * 60}")

            start_time = time.time()
            result = run_experiment(
                data_dim=FIXED_DIMENSION,
                base_data=base_data,
                query_data=query_data,
                true_neighbors=true_neighbors,
                local_size=local_size,
                sw_size=sw_size
            )
            duration = time.time() - start_time

            print(f"\nExperiment completed in {duration:.2f}s | "
                  f"Final Recall@1: {result['final_recall_at_1']:.4f} | "
                  f"Avg Total Edges: {result['final_avg_total_edges']:.2f}")

            results.append(result)

    # Подготовка данных для графика recall@1 vs avg_total_edges
    avg_total_edges = [res['final_avg_total_edges'] for res in results]
    recalls = [res['final_recall_at_1'] for res in results]
    local_sizes = [res['local_size'] for res in results]

    # График recall@1 vs avg_total_edges
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        avg_total_edges,
        recalls,
        c=local_sizes,
        cmap='viridis',
        s=100,
        alpha=0.7
    )

    plt.colorbar(scatter, label='Local View Size')
    plt.title(f"Recall@1 vs Average Node Degree ({FIXED_DIMENSION}D)")
    plt.xlabel("Average Node Degree (Total Edges)")
    plt.ylabel("Recall@1")
    plt.grid(True, linestyle='--', alpha=0.5)

    # Линия тренда
    z = np.polyfit(avg_total_edges, recalls, 1)
    p = np.poly1d(z)
    plt.plot(
        avg_total_edges,
        p(avg_total_edges),
        "r--",
        linewidth=2,
        label=f'Trend: y={z[0]:.4f}x + {z[1]:.2f}'
    )

    plt.legend()
    plt.tight_layout()
    plt.savefig("recall_vs_degree.png", dpi=150)
    plt.show()

    # 3D-график: Local Size, SW Size, Recall@1
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    x = [res['local_size'] for res in results]
    y = [res['sw_size'] for res in results]
    z = [res['final_recall_at_1'] for res in results]

    # Триангулированная поверхность
    surf = ax.plot_trisurf(x, y, z, cmap='viridis', edgecolor='none', alpha=0.8)

    # Точечный график
    ax.scatter(x, y, z, c='r', s=50, alpha=1, label='Experiments')

    ax.set_xlabel('Local Edges')
    ax.set_ylabel('SW Edges')
    ax.set_zlabel('Recall@1')
    ax.set_title(f"3D Recall@1 Analysis ({FIXED_DIMENSION}D)")
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Recall@1')
    plt.legend()
    plt.savefig("3d_recall_analysis.png", dpi=150)
    plt.show()

    # Поиск лучших параметров
    best_idx = np.argmax(recalls)
    best_params = results[best_idx]

    print("\nBest Parameters:")
    print(f"Local Size: {best_params['local_size']}")
    print(f"SW Size: {best_params['sw_size']}")
    print(f"Recall@1: {best_params['final_recall_at_1']:.4f}")
    print(f"Avg Total Edges: {best_params['final_avg_total_edges']:.2f}")

    return results


def main():
    # Запуск анализа зависимости
    print(f"\n{'#' * 60}")
    print(f"Starting Dependency Analysis for {FIXED_DIMENSION}D")
    print(f"Local sizes: {LOCAL_VIEW_SIZES}")
    print(f"SW sizes: {SW_VIEW_SIZES}")
    print(f"Total experiments: {len(LOCAL_VIEW_SIZES) * len(SW_VIEW_SIZES)}")
    print(f"{'#' * 60}\n")

    results = run_dependency_analysis()

    # Вывод результатов
    print("\nResults Summary:")
    print("Local\tSW\tRecall@1\tAvgEdges")
    for res in results:
        print(
            f"{res['local_size']}\t{res['sw_size']}\t{res['final_recall_at_1']:.4f}\t\t{res['final_avg_total_edges']:.2f}")


if __name__ == "__main__":
    main()