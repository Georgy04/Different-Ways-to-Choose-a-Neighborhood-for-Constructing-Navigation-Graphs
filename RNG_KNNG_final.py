import numpy as np
import heapq
import time
import matplotlib.pyplot as plt
import struct
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA


def read_fvecs(filename):
    with open(filename, 'rb') as f:
        vecs = []
        while True:
            dim_b = f.read(4)
            if len(dim_b) < 4: break
            dim = struct.unpack('I', dim_b)[0]
            vec = struct.unpack(f'{dim}f', f.read(4 * dim))
            vecs.append(vec)
        return np.array(vecs, dtype=np.float32)


def read_ivecs(filename):
    with open(filename, 'rb') as f:
        vecs = []
        while True:
            k_b = f.read(4)
            if len(k_b) < 4: break
            k = struct.unpack('I', k_b)[0]
            idx = struct.unpack(f'{k}I', f.read(4 * k))
            vecs.append(idx)
        return np.array(vecs, dtype=np.int32)


class SparseRNG_ANNS:
    def __init__(self, metric='euclidean', n_components=None):
        self.metric = metric
        self.graph = None
        self.X_base = None
        self.avg_degree = None
        self.build_time = 0
        self.n_components = n_components
        self.pca = None

    def build_rng(self, X, K=5):
        start_time = time.time()

        # Применение PCA если задано количество компонент
        if self.n_components is not None:
            self.pca = PCA(n_components=self.n_components)
            self.X_base = self.pca.fit_transform(X)
        else:
            self.X_base = X.copy()

        n = self.X_base.shape[0]
        self.graph = [set() for _ in range(n)]

        # Построение KNNG с малым K
        knn = NearestNeighbors(n_neighbors=K + 1, metric=self.metric)
        knn.fit(self.X_base)
        knn_dist, knn_idx = knn.kneighbors(self.X_base)

        # Построение RNG с строгой проверкой
        for i in tqdm(range(n), desc="Построение RNG"):
            for j_idx in range(1, K + 1):
                j = knn_idx[i, j_idx]
                d_ij = knn_dist[i, j_idx]

                valid_edge = True
                for k in knn_idx[i, 1:K + 1]:
                    if k == j: continue

                    d_ik = knn_dist[i, np.where(knn_idx[i] == k)[0][0]]
                    d_jk = np.linalg.norm(self.X_base[j] - self.X_base[k])

                    # Строгое условие RNG
                    if d_ik < d_ij and d_jk < d_ij:
                        valid_edge = False
                        break

                if valid_edge:
                    self.graph[i].add(j)
                    self.graph[j].add(i)

        # Вычисление средней степени
        degrees = [len(neighbors) for neighbors in self.graph]
        self.avg_degree = np.mean(degrees)
        self.build_time = time.time() - start_time
        return self.graph

    def search(self, query, k=1, ef=100, start_index=0):
        # Преобразование запроса если использовался PCA
        if self.pca is not None:
            query = self.pca.transform(query.reshape(1, -1))[0]

        candidates = []
        visited = set()
        dist = np.linalg.norm(query - self.X_base[start_index])
        heapq.heappush(candidates, (dist, start_index))
        visited.add(start_index)
        results = []

        while candidates:
            dist, node = heapq.heappop(candidates)
            heapq.heappush(results, (-dist, node))
            if len(results) > ef: heapq.heappop(results)

            for neighbor in self.graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    dist_n = np.linalg.norm(query - self.X_base[neighbor])
                    heapq.heappush(candidates, (dist_n, neighbor))

        return [idx for _, idx in sorted(results, reverse=True)[:k]]

    def evaluate(self, queries, true_neighbors, k=1, ef=100):
        correct = 0
        for i, query in enumerate(queries):
            found = self.search(query, k=k, ef=ef)
            if true_neighbors[i, 0] in found:
                correct += 1
        return correct / len(queries)


def test_sparse_rng():
    # Загрузка данных
    X_base = read_fvecs("datasets/siftsmall/siftsmall_base.fvecs")
    X_query = read_fvecs("datasets/siftsmall/siftsmall_query.fvecs")
    gt = read_ivecs("datasets/siftsmall/siftsmall_groundtruth.ivecs")

    print(f"Исходная размерность данных: {X_base.shape[1]}D")
    print(f"Размер базы: {X_base.shape[0]} векторов")
    print(f"Количество запросов: {X_query.shape[0]}")

    # Параметры PCA для тестирования
    pca_components = [None, 8]  # None - без сжатия, 8 - сжатие до 8D
    results = {}

    plt.figure(figsize=(10, 6))

    for n_comp in pca_components:
        label = 'Без PCA' if n_comp is None else f'PCA {n_comp}D'
        print(f"\n--- Тестирование с {label} ---")

        K_values = list(range(1, 11))  # K от 1 до 10
        avg_degrees = []
        recalls = []
        build_times = []

        for K in K_values:
            model = SparseRNG_ANNS(n_components=n_comp)
            model.build_rng(X_base, K=K)
            recall = model.evaluate(X_query, gt, k=1, ef=200)

            avg_degrees.append(model.avg_degree)
            recalls.append(recall)
            build_times.append(model.build_time)

            print(
                f"K={K}, Средняя степень: {model.avg_degree:.2f}, Recall@1={recall:.4f}, Время построения: {model.build_time:.2f} сек")

        results[label] = (avg_degrees, recalls, build_times)
        plt.plot(avg_degrees, recalls, 'o-', markersize=8, label=label)

    # Визуализация сравнения recall
    plt.xlabel('Средняя степень вершины')
    plt.ylabel('Recall@1')
    plt.title('Сравнение точности поиска: оригинальные данные vs сжатие до 8D')
    plt.legend()
    plt.grid(True)
    plt.savefig('sparse_rng_8d_comparison.png')
    plt.show()

    # Визуализация времени построения
    plt.figure(figsize=(10, 6))
    for label, data in results.items():
        _, _, build_times = data
        plt.plot(K_values, build_times, 'o-', markersize=8, label=label)

    plt.xlabel('Параметр K')
    plt.ylabel('Время построения графа (сек)')
    plt.title('Сравнение времени построения графа: оригинальные данные vs сжатие до 8D')
    plt.legend()
    plt.grid(True)
    plt.savefig('build_time_comparison.png')
    plt.show()

    return results


if __name__ == "__main__":
    results = test_sparse_rng()