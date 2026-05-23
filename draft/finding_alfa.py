import numpy as np
import networkx as nx
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh
import math

# =========================
# ПАРАМЕТРЫ
# =========================
N = 6000          # число узлов (начни с 1000–5000)
K = 6             # степень
p = 1.0 / (K * N**(1/3))  # твоя модель

# =========================
# ГЕНЕРАЦИЯ ГРАФА
# =========================
G = nx.watts_strogatz_graph(N, K, p)

# =========================
# ЛАПЛАСИАН
# =========================
L = csgraph.laplacian(nx.adjacency_matrix(G), normed=False)

# =========================
# СОБСТВЕННЫЕ ЗНАЧЕНИЯ
# =========================
# Берём часть спектра (важно!)
k_eigs = 200   # сколько брать (увеличивай для точности)

eigs = eigsh(L, k=k_eigs, which='SM', return_eigenvectors=False)

# убираем нулевое
eigs = eigs[eigs > 1e-10]

# =========================
# СПЕКТРАЛЬНАЯ СУММА
# =========================
S = np.sum(1.0 / eigs)

# нормировка
alpha_eff = S / N

# =========================
# СРАВНЕНИЕ С 1/ln N
# =========================
alpha_theory = 1.0 / math.log(N)

print(f"N = {N}")
print(f"p = {p:.3e}")
print(f"S/N = {alpha_eff:.6e}")
print(f"1/ln N = {alpha_theory:.6e}")

# коэффициент A
A_est = alpha_eff * math.log(N)

print(f"A (из спектра) = {A_est:.6f}")