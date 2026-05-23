import numpy as np
import networkx as nx
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix, diags
from collections import defaultdict
import math

# ============================================================
# ПАРАМЕТРЫ ГРАФА
# ============================================================
K = 6
N_GRAPH = 17000  # Размер графа для вычислений
NUM_EIGS = 50  # Число собственных значений
NUM_RUNS = 10  # Усреднение по запускам

# ============================================================
# ПАРАМЕТРЫ ТЕОРИИ
# ============================================================
pi = math.pi
gamma_E = 0.5772156649015329

# ============================================================
# ЭКСПЕРИМЕНТАЛЬНЫЕ ЗНАЧЕНИЯ (ε_i для разных n)
# ============================================================
epsilon_exp = {
    10: [-5.961441e-05, -4.481392e-04, -3.672270e-05, 8.913598e-04,
         -2.265772e-04, 2.229981e-04, -1.422701e-04, -1.149169e-03, -1.262927e-03],
    11: [1.210081e-03, 1.422633e-03, 7.017339e-04, 8.578545e-04],
    12: [2.788355e-03, 2.511624e-03],
}

# ============================================================
# ГЕНЕРАЦИЯ ГРАФА
# ============================================================
def generate_small_world(N, K, p, seed=None):
    """Генерация small-world графа"""
    if seed is not None:
        np.random.seed(seed)

    G = nx.watts_strogatz_graph(N, K, p)
    A = nx.to_numpy_array(G)
    return A

# ============================================================
# ВЫЧИСЛЕНИЕ ЛАПЛАСИАНА И СПЕКТРА
# ============================================================
def compute_spectrum(A, num_eigs=NUM_EIGS):
    """Вычисляет собственные значения и функции лапласиана"""
    N = A.shape[0]
    degrees = np.sum(A, axis=1)
    D = np.diag(degrees)
    L = D - A

    # Используем eigsh для разреженных матриц
    L_sparse = csr_matrix(L)
    eigenvalues, eigenvectors = eigsh(L_sparse, k=num_eigs, which='SM')

    # Сортируем
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    return eigenvalues, eigenvectors, L

# ============================================================
# ВЫЧИСЛЕНИЕ C_i (аналог вашей формулы)
# ============================================================
def compute_C_from_spectrum(eigenvalues, n):
    """
    Вычисляет C_n = π(n + γ_E) из спектра лапласиана.

    Для лапласиана графа ожидается:
    λ_k ~ c·k^{2/d} для регулярной решётки
    λ_k ~ c·k для small-world (из-за shortcuts)

    Следовательно:
    C_n = π · (среднее геометрическое λ_k для k=1..n)
    """
    if n >= len(eigenvalues):
        return None

    # Берём первые n ненулевых собственных значений
    lambdas = eigenvalues[1:n+1]  # пропускаем λ_0 = 0

    if len(lambdas) == 0:
        return None

    # Среднее геометрическое
    log_mean = np.mean(np.log(np.maximum(lambdas, 1e-15)))
    geo_mean = np.exp(log_mean)

    # Нормируем на характерный масштаб
    # Для small-world графа: λ_k ~ K·k/N
    # C_n = π · geo_mean / (K/N)
    scale = K / N_GRAPH
    C_n = pi * geo_mean / scale

    return C_n

# ============================================================
# ВЫЧИСЛЕНИЕ ОТКЛОНЕНИЙ ОТ ИДЕАЛЬНОГО СПЕКТРА
# ============================================================
def compute_epsilon_from_spectrum(eigenvalues, n, C_n):
    """
    Вычисляет ε = C/(π) - (n + γ_E)
    """
    if C_n is None:
        return None
    return C_n / pi - (n + gamma_E)

# ============================================================
# ВЫЧИСЛЕНИЕ МАТРИЧНЫХ ЭЛЕМЕНТОВ V
# ============================================================
def compute_V_matrix(eigenvectors, n_particles, mode_indices):
    """
    Вычисляет матричные элементы оператора возмущения V.

    V_{ij} = ⟨ψ_i|V|ψ_j⟩ где V — некоторый оператор на графе.

    В качестве V используем:
    V(x,y) = 1 если x и y соединены shortcut
    V(x,y) = 0 иначе

    Это оператор нелокальности — ключевой объект ЕТИ.
    """
    N = eigenvectors.shape[0]
    k = len(mode_indices)
    V = np.zeros((k, k))

    # Вычисляем матричные элементы
    for i, idx_i in enumerate(mode_indices[:k]):
        psi_i = eigenvectors[:, idx_i]
        for j, idx_j in enumerate(mode_indices[:k]):
            psi_j = eigenvectors[:, idx_j]

            # Перекрытие собственных функций
            # V_{ij} = ⟨ψ_i|ψ_j⟩ = δ_{ij} для ортонормированных
            # Но нас интересует ВЗАИМОДЕЙСТВИЕ мод

            # Используем градиентное взаимодействие:
            # V_{ij} = ∫ |∇ψ_i|² · |∇ψ_j|² dx
            # Дискретный аналог: сумма по рёбрам разностей

            # Упрощение: V_{ij} ~ overlap производных
            overlap = np.sum(np.abs(psi_i) * np.abs(psi_j))
            V[i, j] = overlap / N  # Нормировка

    # Делаем матрицу эрмитовой
    V = (V + V.T) / 2

    # Вычитаем диагональ (нас интересует смешивание)
    np.fill_diagonal(V, 0)

    return V

# ============================================================
# ОСНОВНОЙ ЭКСПЕРИМЕНТ
# ============================================================
print("=" * 100)
print("ВЫВОД ε_i ИЗ СТРУКТУРЫ ГРАФА")
print("=" * 100)
print(f"\n  Параметры графа: N={N_GRAPH}, K={K}")
print(f"  Усреднение по {NUM_RUNS} запускам")
print()

# Накопители для усреднения
all_epsilons = defaultdict(list)
all_C_values = defaultdict(list)

for run in range(NUM_RUNS):
    seed = 42 + run * 137
    p = 1.0 / (K * N_GRAPH**(1/3))
    A = generate_small_world(N_GRAPH, K, p, seed=seed)
    eigenvalues, eigenvectors, L = compute_spectrum(A)

    # Для каждого n вычисляем C_n и ε_n
    for n in [10, 11, 12]:
        C_n = compute_C_from_spectrum(eigenvalues, n)
        if C_n is not None:
            eps = compute_epsilon_from_spectrum(eigenvalues, n, C_n)
            if eps is not None:
                all_epsilons[n].append(eps)
                all_C_values[n].append(C_n)

# ============================================================
# УСРЕДНЕНИЕ И ВЫВОД
# ============================================================
print(f"\n{'─' * 100}")
print(f"РЕЗУЛЬТАТЫ (усреднение по {NUM_RUNS} запускам)")
print(f"{'─' * 100}")

for n in sorted(all_epsilons.keys()):
    eps_list = all_epsilons[n]
    C_list = all_C_values[n]

    mean_eps = np.mean(eps_list)
    std_eps = np.std(eps_list)
    mean_C = np.mean(C_list)

    # Экспериментальные значения для сравнения
    exp_eps = epsilon_exp.get(n, [])
    mean_exp_eps = np.mean(exp_eps) if exp_eps else 0

    print(f"\n  n = {n}:")
    print(f"    C (из спектра):     {mean_C:.6f}")
    print(f"    Ожидаемое C = π(n+γ_E): {pi*(n+gamma_E):.6f}")
    print(f"    ε (из спектра):      {mean_eps:+.6e} ± {std_eps:.2e}")
    print(f"    ε (эксперимент, среднее): {mean_exp_eps:+.6e}")

    if exp_eps:
        diff = abs(mean_eps - mean_exp_eps)
        print(f"    Разница:             {diff:.2e}")
        if diff < std_eps:
            print(f"    ✅ СОВПАДЕНИЕ в пределах ошибки!")
        elif diff < 3*std_eps:
            print(f"    🟡 ЧАСТИЧНОЕ совпадение")
        else:
            print(f"    ❌ РАСХОЖДЕНИЕ")

# ============================================================
# СВОДНЫЙ АНАЛИЗ
# ============================================================
print(f"\n{'─' * 100}")
print("СВОДНЫЙ АНАЛИЗ")
print(f"{'─' * 100}")

all_theory = []
all_exp = []

for n in sorted(all_epsilons.keys()):
    if n in epsilon_exp:
        mean_theory = np.mean(all_epsilons[n])
        mean_experiment = np.mean(epsilon_exp[n])
        all_theory.append(mean_theory)
        all_exp.append(mean_experiment)

if len(all_theory) >= 3:
    correlation = np.corrcoef(all_theory, all_exp)[0, 1]
    print(f"\n  Корреляция ε_theory vs ε_experiment: {correlation:.4f}")

    if abs(correlation) > 0.7:
        print(f"  ✅ СИЛЬНАЯ корреляция — ε_i выводятся из спектра графа!")
    elif abs(correlation) > 0.4:
        print(f"  🟡 УМЕРЕННАЯ корреляция — частичное объяснение")
    else:
        print(f"  ❌ СЛАБАЯ корреляция — требуется другая модель")

# ============================================================
# ФИНАЛЬНЫЙ ВЕРДИКТ
# ============================================================
print(f"\n{'=' * 100}")
print("ФИНАЛЬНЫЙ ВЕРДИКТ")
print(f"{'=' * 100}")

print(f"""
  ЧТО ПРОВЕРЕНО:
  ✓ Построен small-world граф с K={K}, p=1/(K·N^(1/3))
  ✓ Вычислен лапласиан и его спектр
  ✓ Из спектра вычислены C_n = π(n + γ_E + ε_n)
  ✓ Результат усреднён по {NUM_RUNS} случайным реализациям графа
  
  ЕСЛИ ε_theory ≈ ε_experiment:
    → Фундаментальные константы = спектр оператора на графе
    → ε_i выводятся из первых принципов
    → ЕТИ становится строгой математической теорией
  
  ЕСЛИ ε_theory НЕ совпадает с ε_experiment:
    → Либо нужен другой оператор (не чистый лапласиан)
    → Либо нужен другой граф (не Watts-Strogatz)
    → Либо ε_i зависят от неизвестных параметров
""")