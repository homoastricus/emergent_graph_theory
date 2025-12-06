import numpy as np

# Параметры графа малого мира
N = 9000       # количество узлов
K = 8          # среднее количество соседей
p = 0.053       # вероятность перестановки
lambda_em = 0.00001  # эмерджентный спектральный масштаб

# Для примера создаём Лапласиан графа малого мира
import networkx as nx

G = nx.watts_strogatz_graph(n=N, k=K, p=p)
L = nx.laplacian_matrix(G).toarray()

# Собственные значения Лапласиана
eigvals = np.linalg.eigvalsh(L)

# Исключаем нулевое собственное значение для связного графа
eigvals_nonzero = eigvals[eigvals > 1e-12]

# Спектральная энтропия
p_i = eigvals_nonzero / np.sum(eigvals_nonzero)
entropy = -np.sum(p_i * np.log(p_i))

print("Спектральная энтропия графа малого мира:", entropy)
