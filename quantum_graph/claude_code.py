"""
ЧИСТЫЙ ЛАПЛАСИАН НА КРИТИЧЕСКОМ ГРАФЕ
Проверка гипотезы: Δ ∼ N^(2/3) / K

Без случайных фаз, без магнитного поля.
Только геометрия графа.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse import linalg as sla
from collections import defaultdict
import time

# ============================================================
# 1. ПОСТРОЕНИЕ ЧИСТОГО ГРАФА (БЕЗ ФАЗ)
# ============================================================

def build_clean_graph(L, K=6, seed=42):
    """
    Кубический граф L×L×L с нелокальными связями.
    ВСЕ рёбра вещественные (A_{ij} = 1).
    """
    np.random.seed(seed)
    N = L ** 3
    N13 = N ** (1/3)
    p_nl = 1.0 / (K * N13)

    def idx(x, y, z):
        return (x % L) + (y % L) * L + (z % L) * L * L

    adjacency = defaultdict(list)
    nonlocal_count = 0

    # Локальные связи
    for x in range(L):
        for y in range(L):
            for z in range(L):
                i = idx(x, y, z)
                for dx, dy, dz in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
                    j = idx(x+dx, y+dy, z+dz)
                    adjacency[i].append(j)

    # Нелокальные связи
    for i in range(N):
        if np.random.random() < p_nl:
            j = np.random.randint(N)
            attempts = 0
            while (j == i or j in adjacency[i]) and attempts < 100:
                j = np.random.randint(N)
                attempts += 1
            if j != i and j not in adjacency[i]:
                adjacency[i].append(j)
                adjacency[j].append(i)
                nonlocal_count += 1

    # Строим sparse-матрицу
    rows, cols = [], []
    for i in range(N):
        for j in adjacency[i]:
            rows.append(i)
            cols.append(j)

    A = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(N, N))
    return A, nonlocal_count


# ============================================================
# 2. ЧИСТЫЙ ЛАПЛАСИАН
# ============================================================

def build_clean_laplacian(A):
    """L = D - A, вещественный, положительно полуопределённый."""
    N = A.shape[0]
    degree = A.sum(axis=1).A.flatten()
    D = sparse.diags(degree, 0, shape=(N, N))
    return D - A


# ============================================================
# 3. МАСШТАБНЫЙ ТЕСТ
# ============================================================

def scaling_test(L_values):
    """
    Вычисляет спектральную щель для разных размеров.
    Проверяет гипотезы:
      H1: Δ ∼ 1/ln N  → C1 = Δ·ln N ≈ const
      H2: Δ ∼ N^(2/3)/K → C2 = Δ·K/N^(2/3) ≈ const
    """
    print("="*90)
    print("ЧИСТЫЙ ЛАПЛАСИАН: МАСШТАБНЫЙ ТЕСТ")
    print("="*90)
    print(f"{'L':>4} {'N':>6} {'ln N':>8} {'nl':>5} {'λ₀':>10} {'λ₁':>10} {'Δ':>12} "
          f"{'Δ·ln N':>12} {'Δ·K/N^(2/3)':>16}")
    print("-"*90)

    results = []

    for L in L_values:
        t0 = time.time()

        # Строим граф
        A, nl_count = build_clean_graph(L)

        # Лапласиан
        L_op = build_clean_laplacian(A)

        # Спектр
        N = L**3
        k = min(10, N-2)
        evals = sla.eigsh(L_op, k=k, which='SM', return_eigenvectors=False)
        evals = np.sort(evals)

        # λ₀ должно быть ≈ 0 (с точностью до машинного нуля)
        lambda0 = evals[0]
        lambda1 = evals[1] if len(evals) > 1 else 0
        gap = lambda1 - lambda0

        # Проверяем гипотезы
        ln_N = np.log(N)
        C1 = gap * ln_N

        K = 6
        N23 = N ** (2/3)
        C2 = gap * K / N23

        elapsed = time.time() - t0

        print(f"{L:4d} {N:6d} {ln_N:8.4f} {nl_count:5d} {lambda0:10.6f} {lambda1:10.6f} {gap:12.6f} "
              f"{C1:12.6f} {C2:16.6f}")

        results.append({
            'L': L, 'N': N, 'ln_N': ln_N,
            'nl_count': nl_count,
            'lambda0': lambda0,
            'lambda1': lambda1,
            'gap': gap,
            'C1': C1,
            'C2': C2
        })

    return results


# ============================================================
# 4. ВИЗУАЛИЗАЦИЯ
# ============================================================

def plot_results(results):
    """Строит графики для проверки обеих гипотез."""

    L_arr = np.array([r['L'] for r in results])
    N_arr = np.array([r['N'] for r in results])
    lnN_arr = np.array([r['ln_N'] for r in results])
    gap_arr = np.array([r['gap'] for r in results])
    C1_arr = np.array([r['C1'] for r in results])
    C2_arr = np.array([r['C2'] for r in results])

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Δ vs 1/ln N (проверка H1)
    ax = axes[0, 0]
    ax.scatter(1/lnN_arr, gap_arr, s=80, c='steelblue', edgecolors='black', zorder=5)
    if len(gap_arr) > 1:
        # Фит прямой через ноль
        x = 1/lnN_arr.reshape(-1, 1)
        from sklearn.linear_model import LinearRegression
        reg = LinearRegression(fit_intercept=False).fit(x, gap_arr)
        x_plot = np.linspace(0, max(1/lnN_arr)*1.1, 100).reshape(-1, 1)
        ax.plot(x_plot, reg.predict(x_plot), 'r--', linewidth=2,
                label=f'Наклон = {reg.coef_[0]:.4f}')
    ax.set_xlabel('1 / ln N')
    ax.set_ylabel('Δ')
    ax.set_title('Гипотеза 1: Δ ∼ 1/ln N')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Δ vs N^(2/3)/K (проверка H2)
    ax = axes[0, 1]
    N23_over_K = N_arr**(2/3) / 6
    ax.scatter(N23_over_K, gap_arr, s=80, c='darkorange', edgecolors='black', zorder=5)
    if len(gap_arr) > 1:
        x2 = N23_over_K.reshape(-1, 1)
        reg2 = LinearRegression(fit_intercept=False).fit(x2, gap_arr)
        x2_plot = np.linspace(0, max(N23_over_K)*1.1, 100).reshape(-1, 1)
        ax.plot(x2_plot, reg2.predict(x2_plot), 'g--', linewidth=2,
                label=f'Наклон = {reg2.coef_[0]:.4f}')
    ax.set_xlabel('N^(2/3) / K')
    ax.set_ylabel('Δ')
    ax.set_title('Гипотеза 2: Δ ∼ N^(2/3)/K')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. C1 = Δ·ln N vs ln N (должна быть константа для H1)
    ax = axes[1, 0]
    ax.scatter(lnN_arr, C1_arr, s=80, c='steelblue', edgecolors='black', zorder=5)
    ax.axhline(y=np.mean(C1_arr), color='blue', linestyle='--', alpha=0.5,
               label=f'Среднее: {np.mean(C1_arr):.4f}')
    ax.set_xlabel('ln N')
    ax.set_ylabel('C1 = Δ · ln N')
    ax.set_title('Постоянство C1 (H1)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. C2 = Δ·K/N^(2/3) vs ln N (должна быть константа для H2)
    ax = axes[1, 1]
    ax.scatter(lnN_arr, C2_arr, s=80, c='darkorange', edgecolors='black', zorder=5)
    ax.axhline(y=np.mean(C2_arr), color='orange', linestyle='--', alpha=0.5,
               label=f'Среднее: {np.mean(C2_arr):.4f}')
    ax.set_xlabel('ln N')
    ax.set_ylabel('C2 = Δ · K / N^(2/3)')
    ax.set_title('Постоянство C2 (H2)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('clean_laplacian_test.png', dpi=300)
    print("\n✓ Сохранено: clean_laplacian_test.png")


# ============================================================
# 5. ГЛАВНАЯ
# ============================================================

def main():
    print("="*90)
    print("ЧИСТЫЙ ЛАПЛАСИАН НА КРИТИЧЕСКОМ ГРАФЕ")
    print("Без фаз, без магнитного поля")
    print("="*90)

    # Размеры для теста
    L_values = [3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 20, 24, 28, 32]

    # Масштабный тест
    results = scaling_test(L_values)

    # Визуализация
    plot_results(results)

    # Анализ
    C1_arr = np.array([r['C1'] for r in results])
    C2_arr = np.array([r['C2'] for r in results])

    cv_C1 = np.std(C1_arr) / np.mean(C1_arr)
    cv_C2 = np.std(C2_arr) / np.mean(C2_arr)

    print("\n" + "="*90)
    print("АНАЛИЗ")
    print("="*90)
    print(f"\nГипотеза 1 (Δ ∼ 1/ln N):")
    print(f"  C1 = {np.mean(C1_arr):.6f} ± {np.std(C1_arr):.6f}")
    print(f"  Коэффициент вариации: {cv_C1:.4f}")

    print(f"\nГипотеза 2 (Δ ∼ N^(2/3)/K):")
    print(f"  C2 = {np.mean(C2_arr):.6f} ± {np.std(C2_arr):.6f}")
    print(f"  Коэффициент вариации: {cv_C2:.4f}")

    if cv_C1 < cv_C2:
        print(f"\n✅ Гипотеза 1 (Δ ∼ 1/ln N) ЛУЧШЕ описывает данные")
    else:
        print(f"\n✅ Гипотеза 2 (Δ ∼ N^(2/3)/K) ЛУЧШЕ описывает данные")

    # Дополнительно: проверка Δ ∼ 1/L² (обычная диффузия)
    L_arr = np.array([r['L'] for r in results])
    gap_arr = np.array([r['gap'] for r in results])
    C3_arr = gap_arr * L_arr**2

    cv_C3 = np.std(C3_arr) / np.mean(C3_arr)
    print(f"\nГипотеза 3 (Δ ∼ 1/L², обычная диффузия):")
    print(f"  C3 = Δ·L² = {np.mean(C3_arr):.6f} ± {np.std(C3_arr):.6f}")
    print(f"  Коэффициент вариации: {cv_C3:.4f}")

    return results


if __name__ == "__main__":
    results = main()
    plt.show()