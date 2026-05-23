import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# ============================================================
# ПАРАМЕТРЫ
# ============================================================
N = 1000
K = 6
p = 0.05

# ============================================================
# 1. СТРОИМ SMALL-WORLD ГРАФ
# ============================================================
G = nx.watts_strogatz_graph(N, K, p)

# ============================================================
# 2. ЛАПЛАСИАН
# ============================================================
L = nx.laplacian_matrix(G).toarray()

# ============================================================
# 3. СПЕКТР
# ============================================================
eigenvalues = np.linalg.eigvalsh(L)

# убираем нулевое собственное значение
eigenvalues = eigenvalues[1:]

# защита от log(0)
eigenvalues = np.clip(eigenvalues, 1e-12, None)

# ============================================================
# 4. СТРОИМ A(n) = Σ log λ_k
# ============================================================
log_lambda = np.log(eigenvalues)

A_n = np.cumsum(log_lambda)

n = np.arange(1, len(A_n) + 1)

# ============================================================
# 5. СРАВНЕНИЕ С n log n
# ============================================================
model_nlogn = n * np.log(n)

# нормируем для сравнения формы
A_n_norm = (A_n - A_n[0]) / (A_n[-1] - A_n[0])
model_norm = (model_nlogn - model_nlogn[0]) / (model_nlogn[-1] - model_nlogn[0])

# ============================================================
# 6. ГРАФИК
# ============================================================
plt.figure()
plt.plot(n, A_n_norm, label='A(n) from spectrum')
plt.plot(n, model_norm, '--', label='n log n')
plt.legend()
plt.xlabel('n')
plt.ylabel('normalized')
plt.title('Spectral test: does A(n) ~ n log n?')
plt.show()

# ============================================================
# 7. ЧИСЛЕННАЯ ПРОВЕРКА
# ============================================================
# линейная регрессия A(n) vs n log n
X = np.column_stack([n * np.log(n), n, np.ones(len(n))])
coeffs, *_ = np.linalg.lstsq(X, A_n, rcond=None)

pred = X @ coeffs

R2 = 1 - np.sum((A_n - pred)**2) / np.sum((A_n - np.mean(A_n))**2)

print("Fit: A(n) = a*n log n + b*n + c")
print(f"a = {coeffs[0]:.4f}")
print(f"b = {coeffs[1]:.4f}")
print(f"c = {coeffs[2]:.4f}")
print(f"R^2 = {R2:.4f}")