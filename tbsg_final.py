import numpy as np
import os
import time
import heapq
import math
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree, BallTree
import struct
from collections import defaultdict
from tqdm import tqdm
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def read_fvecs(filename):
    """Чтение файлов в формате fvecs"""
    with open(filename, 'rb') as f:
        vecs = []
        while True:
            dim_bin = f.read(4)
            if not dim_bin:
                break
            dim = struct.unpack('i', dim_bin)[0]
            vec = struct.unpack('f' * dim, f.read(4 * dim))
            vecs.append(np.array(vec))
    return np.vstack(vecs)


def read_ivecs(filename):
    """Чтение файлов в формате ivecs"""
    with open(filename, 'rb') as f:
        vecs = []
        while True:
            dim_bin = f.read(4)
            if not dim_bin:
                break
            dim = struct.unpack('i', dim_bin)[0]
            vec = struct.unpack('i' * dim, f.read(4 * dim))
            vecs.append(np.array(vec))
    return np.vstack(vecs)


class CoverTree:
    """Улучшенная реализация CoverTree с кэшированием расстояний и ограничением глубины"""

    def __init__(self, base=1.5, max_points=20, max_depth=10):
        self.base = base
        self.root = None
        self.nodes = []
        self.max_points = max_points
        self.max_depth = max_depth
        self.data = None
        self.point_to_node = {}
        self.distance_cache = {}
        self.level_limits = {}

    def _distance(self, i, j):
        """Вычисление расстояния с кэшированием"""
        if i == j:
            return 0.0
        key = (min(i, j), max(i, j))
        if key not in self.distance_cache:
            self.distance_cache[key] = np.linalg.norm(self.data[i] - self.data[j])
        return self.distance_cache[key]

    def build(self, data):
        """Построение CoverTree с ограничением глубины"""
        self.data = data  # Инициализация данных
        self.root = CoverTreeNode(0, 0)
        self.nodes = [self.root]
        self.point_to_node = {0: 0}
        self.level_limits = {0: 0.0}

        print("Building CoverTree...")
        start_ct = time.time()

        for i in tqdm(range(1, len(data)), desc="CoverTree Points"):
            current = self.root
            depth = 0

            while depth < self.max_depth:
                found_child = False
                best_child = None
                min_dist = float('inf')

                # Ищем ближайшего потомка
                for child_idx in current.children:
                    child = self.nodes[child_idx]
                    dist = self._distance(i, child.point_idx)
                    if dist < min_dist:
                        min_dist = dist
                        best_child = child

                # Проверяем, попадает ли точка в покрытие потомка
                if best_child and min_dist <= self.base ** best_child.level:
                    current = best_child
                    found_child = True
                    depth += 1
                else:
                    break

                # Проверяем ограничение по глубине
                if depth >= self.max_depth:
                    break

            # Создаем новый узел
            new_level = current.level - 1
            new_node = CoverTreeNode(i, new_level)
            new_node_idx = len(self.nodes)
            self.nodes.append(new_node)
            self.point_to_node[i] = new_node_idx

            # Добавляем к текущему узлу, если не превышен лимит
            if len(current.children) < self.max_points:
                current.children.append(new_node_idx)
            else:
                # Добавляем к корню, если у текущего узла переполнение
                self.root.children.append(new_node_idx)

        print(f"CoverTree built in {time.time() - start_ct:.2f} seconds")
        print(f"Total nodes: {len(self.nodes)}, Max depth: {self.max_depth}")


class CoverTreeNode:
    """Узел CoverTree"""

    def __init__(self, point_idx, level):
        self.point_idx = point_idx
        self.level = level
        self.children = []


class TBSG:
    """Улучшенная реализация Tree-Based Search Graph"""

    def __init__(self, K=100, mp=0.5):
        self.K = K
        self.mp = mp
        self.graph = {}
        self.cover_tree = None
        self.KNNG = {}
        self.BKNNG = {}
        self.data = None
        self.r_values = []
        self.enter_point = 0
        self.avg_degree = 0
        self.distance_cache = {}

    def set_data(self, data):
        """Установка данных перед построением структур"""
        self.data = data
        self.distance_cache = {}

    def _distance(self, i, j):
        """Вычисление расстояния с кэшированием"""
        if i == j:
            return 0.0
        key = (min(i, j), max(i, j))
        if key not in self.distance_cache:
            if self.data is None:
                raise ValueError("Data not initialized before distance calculation")
            self.distance_cache[key] = np.linalg.norm(self.data[i] - self.data[j])
        return self.distance_cache[key]

    def build_knng(self):
        """Построение K-Nearest Neighbor Graph с использованием BallTree"""
        if self.data is None:
            raise ValueError("Data not initialized before building KNNG")

        print("Building KNNG with BallTree...")
        start = time.time()

        # Используем BallTree для большей эффективности
        tree = BallTree(self.data)
        _, indices = tree.query(self.data, k=self.K + 1)  # +1 чтобы исключить саму точку

        for i in range(len(self.data)):
            self.KNNG[i] = indices[i, 1:].tolist()  # Исключаем саму точку

        print(f"KNNG built in {time.time() - start:.2f} seconds")

    def build_bknng(self):
        """Построение Bi-directional KNN Graph"""
        print("Building BKNNG...")
        start = time.time()
        self.BKNNG = defaultdict(set)

        for i, neighbors in self.KNNG.items():
            for neighbor in neighbors:
                self.BKNNG[neighbor].add(i)

        print(f"BKNNG built in {time.time() - start:.2f} seconds")

    def compute_r_values(self):
        """Вычисление r для каждой точки (расстояние до ближайшего соседа)"""
        if self.data is None:
            raise ValueError("Data not initialized before computing r-values")

        print("Computing r values for each point...")
        start = time.time()
        self.r_values = np.zeros(len(self.data))

        # Используем BallTree для эффективного поиска
        tree = BallTree(self.data)
        dists, _ = tree.query(self.data, k=2)  # Берем 2 соседей (сама точка и ближайший)

        for i in range(len(self.data)):
            self.r_values[i] = dists[i, 1]  # Расстояние до ближайшего соседа

        print(f"Computed r values in {time.time() - start:.2f} seconds")

        # Логируем статистику
        print(f"r-values stats: min={np.min(self.r_values):.4f}, "
              f"max={np.max(self.r_values):.4f}, mean={np.mean(self.r_values):.4f}")

    def calculate_min_prob(self, s_idx, e_idx, v_idx):
        """Корректное вычисление min_prob с обработкой граничных случаев"""
        try:
            AB = self._distance(s_idx, e_idx)
            AC = self._distance(s_idx, v_idx)
            BC = self._distance(e_idx, v_idx)

            # Вычисление углов
            # Угол α в точке A (между AB и AC)
            cos_alpha = (AB ** 2 + AC ** 2 - BC ** 2) / (2 * AB * AC + 1e-10)
            cos_alpha = max(min(cos_alpha, 1.0), -1.0)
            alpha = math.acos(cos_alpha)

            # Угол θ в точке B (между BA и BC)
            cos_theta = (AB ** 2 + BC ** 2 - AC ** 2) / (2 * AB * BC + 1e-10)
            cos_theta = max(min(cos_theta, 1.0), -1.0)
            theta = math.acos(cos_theta)

            # Вычисление sin(2α + θ)
            sin_val = math.sin(2 * alpha + theta)

            # Вычисление знаменателя
            denominator = 2 * math.sin(alpha + theta)

            # Избегаем деления на ноль
            if abs(denominator) < 1e-6:
                return 0.0

            # Вычисление φ
            ratio = sin_val / denominator

            # Ограничиваем значение для арккосинуса
            ratio = max(min(ratio, 1.0), -1.0)
            phi = math.acos(ratio)

            # Финальная формула min_prob
            min_prob = 1 - phi / math.pi

            # Гарантируем, что вероятность в [0, 1]
            return max(0.0, min(1.0, min_prob))

        except Exception as e:
            # В случае ошибок возвращаем консервативное значение
            print(f"Error in min_prob calculation: {e}")
            return 0.5  # Возвращаем среднее значение при ошибке

    def neighbor_selection_for_node(self, s_idx, candidates, m):
        """Улучшенная стратегия выбора соседей"""
        V = []
        s = self.data[s_idx]

        # Сортировка кандидатов по расстоянию
        candidate_dists = [self._distance(s_idx, c_idx) for c_idx in candidates]
        sorted_indices = np.argsort(candidate_dists)
        sorted_candidates = [candidates[i] for i in sorted_indices]

        for e_idx in sorted_candidates:
            if len(V) >= m:
                break

            # Для первых 5 соседей добавляем без проверки
            if len(V) < 5:
                V.append(e_idx)
                continue

            exclude = False
            for v_idx in V:
                # Упрощенная проверка без min_prob для первых 10 точек
                if s_idx < 10:
                    dist_ve = self._distance(v_idx, e_idx)
                    dist_se = self._distance(s_idx, e_idx)

                    if dist_ve < dist_se:
                        exclude = True
                        break
                else:
                    # Полная проверка с min_prob
                    dist_ve = self._distance(v_idx, e_idx)
                    dist_se = self._distance(s_idx, e_idx)

                    if dist_ve < dist_se:
                        min_prob = self.calculate_min_prob(s_idx, e_idx, v_idx)
                        if min_prob >= self.mp:
                            exclude = True
                            break

            if not exclude:
                V.append(e_idx)

        return V

    def build_tbsg(self, data, m):
        """Построение TBSG индекса с параметром m"""
        # Устанавливаем данные, если еще не установлены
        if self.data is None:
            self.set_data(data)

        # Построение структур
        if self.cover_tree is None:
            self.cover_tree = CoverTree(max_points=20, max_depth=10)
            self.cover_tree.build(data)
            self.enter_point = self.cover_tree.root.point_idx

        if not self.KNNG:
            self.build_knng()

        if not self.BKNNG:
            self.build_bknng()

        # Построение графа TBSG
        print(f"Building TBSG graph with m={m}...")
        start_graph = time.time()
        self.graph = {}
        total_edges = 0

        for i in tqdm(range(len(data)), desc="TBSG Nodes"):
            candidates = set()

            # Добавляем соседей из BKNNG
            if i in self.BKNNG:
                candidates.update(self.BKNNG[i])

            # Добавляем дочерние узлы из CoverTree
            if i in self.cover_tree.point_to_node:
                node_idx = self.cover_tree.point_to_node[i]
                node = self.cover_tree.nodes[node_idx]
                for child_idx in node.children:
                    child_node = self.cover_tree.nodes[child_idx]
                    candidates.add(child_node.point_idx)

            # Добавляем KNN, если нет кандидатов
            if not candidates and i in self.KNNG:
                candidates.update(self.KNNG[i])

            # Выбираем соседей
            self.graph[i] = self.neighbor_selection_for_node(i, list(candidates), m)
            total_edges += len(self.graph[i])

        # Вычисление средней степени
        self.avg_degree = total_edges / len(data)
        print(f"TBSG graph built in {time.time() - start_graph:.2f} seconds")
        print(f"Average degree: {self.avg_degree:.2f}")

        # Визуализируем граф для первых 50 точек
        self.visualize_graph(min(50, len(data)))

    def visualize_graph(self, num_nodes=50):
        """Визуализация подграфа для отладки"""
        if num_nodes <= 0:
            return

        print("Visualizing graph structure...")
        G = nx.DiGraph()

        # Добавляем узлы и ребра
        for i in range(num_nodes):
            G.add_node(i)
            if i in self.graph:
                for neighbor in self.graph[i]:
                    if neighbor < num_nodes:
                        G.add_edge(i, neighbor)

        # Рисуем граф
        plt.figure(figsize=(12, 10))
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, with_labels=True, node_size=300,
                node_color='skyblue', font_size=8,
                edge_color='gray', arrows=True)
        plt.title(f"TBSG Graph Structure (First {num_nodes} Nodes)")
        plt.savefig('tbsg_graph.png', dpi=300)
        plt.close()
        print("Graph visualization saved to tbsg_graph.png")

    def search(self, query, L=200, k=1):
        """Улучшенный алгоритм поиска с полноценным backtracking"""
        if self.data is None:
            raise ValueError("Data not initialized for search")

        visited = set()
        candidates = []  # Min-heap: (distance, node_idx)
        results = []  # Max-heap: (-distance, node_idx)

        # Начинаем с точки входа
        start_dist = np.linalg.norm(query - self.data[self.enter_point])
        heapq.heappush(candidates, (start_dist, self.enter_point))

        # Поиск ближайших соседей
        while candidates and len(visited) < L:
            dist, node_idx = heapq.heappop(candidates)

            if node_idx in visited:
                continue

            visited.add(node_idx)

            # Обновляем результаты
            if len(results) < k:
                heapq.heappush(results, (-dist, node_idx))
            elif dist < -results[0][0]:
                heapq.heapreplace(results, (-dist, node_idx))

            # Исследуем соседей
            if node_idx in self.graph:
                for neighbor_idx in self.graph[node_idx]:
                    if neighbor_idx in visited:
                        continue

                    n_dist = np.linalg.norm(query - self.data[neighbor_idx])
                    heapq.heappush(candidates, (n_dist, neighbor_idx))

        # Сортируем результаты
        sorted_results = sorted([(-dist, idx) for dist, idx in results])
        return [idx for dist, idx in sorted_results[:k]]


def recall(groundtruth, results, k=1):
    """Recall@k: доля запросов, где истинный 1-NN найден в топ-k результатов"""
    correct = 0
    for i in range(len(groundtruth)):
        true_nn = groundtruth[i, 0]
        if true_nn in results[i][:k]:
            correct += 1
    return correct / len(groundtruth)


def run_experiment(data, queries, groundtruth, m_values, K=100, mp=0.5, L_search=200, k_search=1):
    """Запуск эксперимента с оптимизированным построением структур"""
    # Инициализация TBSG
    tbsg = TBSG(K=K, mp=mp)
    tbsg.set_data(data)  # Устанавливаем данные перед построением структур

    # Предварительное построение общих структур
    print("Building common structures...")

    # Важно: сначала вычисляем r-значения
    tbsg.compute_r_values()

    # Затем строим остальные структуры
    tbsg.cover_tree = CoverTree(max_points=20, max_depth=10)
    tbsg.cover_tree.build(data)
    tbsg.build_knng()
    tbsg.build_bknng()

    # Результаты эксперимента
    results = {
        'm_values': m_values,
        'avg_degrees': [],
        'recalls': [],
        'query_times': [],
        'build_times': []
    }

    for m in m_values:
        print(f"\n{'=' * 50}")
        print(f"Running experiment for m={m}")
        print(f"{'=' * 50}")

        # Построение графа с текущим m
        start_build = time.time()
        tbsg.build_tbsg(data, m)
        build_time = time.time() - start_build

        # Выполнение поиска
        start_search = time.time()
        search_results = []

        for i, query in enumerate(tqdm(queries, desc=f"Searching (m={m})")):
            results_list = tbsg.search(query, L=L_search, k=k_search)
            search_results.append(results_list)

        search_time = time.time() - start_search

        # Расчет метрик
        recall_score = recall(groundtruth, search_results, k=k_search)
        avg_query_time = search_time / len(queries)

        # Сохранение результатов
        results['avg_degrees'].append(tbsg.avg_degree)
        results['recalls'].append(recall_score)
        results['query_times'].append(avg_query_time)
        results['build_times'].append(build_time)

        print(f"\nResults for m={m}:")
        print(f"  Avg degree: {tbsg.avg_degree:.2f}")
        print(f"  Recall@1: {recall_score:.4f}")
        print(f"  Avg query time: {avg_query_time:.6f} sec")
        print(f"  Build time: {build_time:.2f} sec")

    return results


def plot_results(results):
    """Визуализация результатов эксперимента"""
    plt.figure(figsize=(14, 10))

    # График точности
    plt.subplot(2, 2, 1)
    plt.plot(results['m_values'], results['recalls'], 'bo-', markersize=8)
    plt.title('Recall@1 vs Max Degree (m)', fontsize=14)
    plt.xlabel('Max Degree (m)', fontsize=12)
    plt.ylabel('Recall@1', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    for i, (x, y) in enumerate(zip(results['m_values'], results['recalls'])):
        plt.annotate(f"{y:.4f}", (x, y), textcoords="offset points",
                     xytext=(0, 10), ha='center', fontsize=9)

    # График времени запроса
    plt.subplot(2, 2, 2)
    plt.plot(results['m_values'], results['query_times'], 'ro-', markersize=8)
    plt.title('Query Time vs Max Degree (m)', fontsize=14)
    plt.xlabel('Max Degree (m)', fontsize=12)
    plt.ylabel('Avg Query Time (sec)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    for i, (x, y) in enumerate(zip(results['m_values'], results['query_times'])):
        plt.annotate(f"{y:.6f}", (x, y), textcoords="offset points",
                     xytext=(0, 10), ha='center', fontsize=7)

    # График средней степени
    plt.subplot(2, 2, 3)
    plt.plot(results['m_values'], results['avg_degrees'], 'go-', markersize=8)
    plt.title('Actual Degree vs Max Degree (m)', fontsize=14)
    plt.xlabel('Max Degree (m)', fontsize=12)
    plt.ylabel('Actual Average Degree', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    for i, (x, y) in enumerate(zip(results['m_values'], results['avg_degrees'])):
        plt.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                     xytext=(0, 10), ha='center', fontsize=9)

    # График зависимости времени построения
    plt.subplot(2, 2, 4)
    plt.plot(results['m_values'], results['build_times'], 'mo-', markersize=8)
    plt.title('Build Time vs Max Degree (m)', fontsize=14)
    plt.xlabel('Max Degree (m)', fontsize=12)
    plt.ylabel('Build Time (sec)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    for i, (x, y) in enumerate(zip(results['m_values'], results['build_times'])):
        plt.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                     xytext=(0, 10), ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig('tbsg_results.png', dpi=300)
    plt.show()


def apply_pca(data, queries, target_dim):
    """Применение PCA для снижения размерности данных"""
    print(f"Applying PCA to reduce dimensions from {data.shape[1]} to {target_dim}...")

    # Нормализация данных перед PCA
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data)
    queries_normalized = scaler.transform(queries)

    # Применение PCA
    pca = PCA(n_components=target_dim, random_state=42)
    data_reduced = pca.fit_transform(data_normalized)
    queries_reduced = pca.transform(queries_normalized)

    # Анализ сохраненной дисперсии
    explained_variance = np.sum(pca.explained_variance_ratio_)
    print(f"Explained variance: {explained_variance:.4f}")

    return data_reduced, queries_reduced


def main():
    """Основная функция с добавлением PCA эксперимента"""
    # Загрузка данных
    base_path = os.path.join("datasets", "siftsmall")
    print(f"Loading data from {base_path}...")

    base_data = read_fvecs(os.path.join(base_path, "siftsmall_base.fvecs"))
    query_data = read_fvecs(os.path.join(base_path, "siftsmall_query.fvecs"))
    groundtruth = read_ivecs(os.path.join(base_path, "siftsmall_groundtruth.ivecs"))

    print(f"Base data shape: {base_data.shape}")
    print(f"Query data shape: {query_data.shape}")
    print(f"Groundtruth shape: {groundtruth.shape}")

    # Параметры эксперимента
    m_values = [2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30]  # Значения максимальной степени
    K = 100
    mp = 0.5
    L_search = 200
    k_search = 1
    pca_dim = 8  # Целевая размерность для PCA

    # Результаты для сравнения
    results_original = []
    results_pca = []

    # Эксперимент с оригинальными данными
    print("\n" + "=" * 60)
    print("Running experiment with ORIGINAL data")
    print("=" * 60)
    res_orig = run_experiment(
        data=base_data.copy(),
        queries=query_data.copy(),
        groundtruth=groundtruth,
        m_values=m_values,
        K=K,
        mp=mp,
        L_search=L_search,
        k_search=k_search
    )
    results_original = list(zip(res_orig['avg_degrees'], res_orig['recalls']))

    # Эксперимент с данными после PCA (8D)
    print("\n" + "=" * 60)
    print("Running experiment with PCA (8D) data")
    print("=" * 60)
    base_pca, query_pca = apply_pca(base_data, query_data, pca_dim)
    res_pca = run_experiment(
        data=base_pca,
        queries=query_pca,
        groundtruth=groundtruth,
        m_values=m_values,
        K=K,
        mp=mp,
        L_search=L_search,
        k_search=k_search
    )
    results_pca = list(zip(res_pca['avg_degrees'], res_pca['recalls']))

    # Визуализация сравнения recall@1 vs. средняя степень
    plt.figure(figsize=(10, 6))

    # График для оригинальных данных
    deg_orig, recall_orig = zip(*results_original)
    plt.plot(deg_orig, recall_orig, 'bo-', markersize=8, label='Original Data')
    for i, (deg, rec) in enumerate(results_original):
        plt.annotate(f"m={m_values[i]}", (deg, rec), textcoords="offset points",
                     xytext=(0, 10), ha='center', fontsize=9)

    # График для данных после PCA
    deg_pca, recall_pca = zip(*results_pca)
    plt.plot(deg_pca, recall_pca, 'rs-', markersize=8, label=f'PCA ({pca_dim}D)')
    for i, (deg, rec) in enumerate(results_pca):
        plt.annotate(f"m={m_values[i]}", (deg, rec), textcoords="offset points",
                     xytext=(0, -15), ha='center', fontsize=9)

    # Настройка графика
    plt.title('Recall@1 vs. Average Degree Comparison', fontsize=14)
    plt.xlabel('Average Degree', fontsize=12)
    plt.ylabel('Recall@1', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()

    # Сохранение и отображение
    plt.savefig('recall_vs_degree_pca_comparison.png', dpi=300)
    plt.show()

    print("\nExperiment completed. Results saved to recall_vs_degree_pca_comparison.png")


if __name__ == "__main__":
    main()