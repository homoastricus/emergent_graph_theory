import numpy as np
from scipy.optimize import root_scalar, fsolve
import matplotlib.pyplot as plt

# ПАРАМЕТРЫ МОДЕЛИ (фиксированные)

d = 3  # размерность пространства
K0 = 2 * d  # связность идеальной решётки (6)
alpha = 1.0  # стоимость локальной связи
beta = 1.0 / (2 * d)  # удорожание дальних связей
gamma = 1.0  # стоимость транспорта
eta = 0.1  # нелинейное насыщение
kappa = 1.23  # жёсткость геометрии
T = 0.5  # температура (энтропия)
lambda_spec = 0.01  # спектральный коэффициент


# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
def entropy(p):
    """Энтропия Шеннона для вероятности p"""
    p = np.clip(p, 1e-300, 1 - 1e-12)
    return -p * np.log(p) - (1 - p) * np.log(1 - p)


def solve_K(x, N):
    """
    Решает нелинейное уравнение для K при фиксированных x и N
    """
    p = x / N ** (1.0 / d)

    def equation(K):
        arg = x * K * K / N ** (1.0 / d)
        num = arg - 1.0
        den = (arg + 1.0) ** 2
        spec_term = lambda_spec * num / den

        return (2.0 * kappa * (K - K0) +
                alpha * (1.0 + beta * x) -
                T * entropy(p) -
                spec_term)

    try:
        sol = root_scalar(equation, bracket=[K0 - 3.0, K0 + 3.0], method='bisect')
        return sol.root
    except:
        return fsolve(equation, K0)[0]


def compute_c(N, max_iter=30, tol=1e-8, verbose=True):
    """
    Вычисляет фиксированную точку c(N) для данного N
    """
    lnN = np.log(N)
    # Начальное приближение (из упрощённой модели)
    c = (gamma / (2.0 * eta)) ** (1.0 / 3.0)
    K = None

    for i in range(max_iter):
        x = c * (lnN) ** (1.0 / 3.0)
        K = solve_K(x, N)
        c_new = np.sqrt(gamma * (lnN) ** (1.0 / 3.0) / (K + 2.0 * eta * x))

        if verbose and i % 5 == 0:
            print(f"  iter {i:2d}: c = {c:.6f}, K = {K:.6f}")

        if abs(c_new - c) < tol:
            return c_new, K
        c = c_new

    return c, K


def find_N_for_c_target(c_target=1.0, tol=1e-6, max_iter=50):
    """
    Находит N, при котором c(N) = c_target (например, 1.0)
    Метод: бисекция по ln(N)
    """

    def c_diff(logN):
        N = np.exp(logN)
        c, _ = compute_c(N, max_iter=25, verbose=False)
        return c - c_target

    # Диапазон поиска (по lnN)
    lnN_min = np.log(1e20)  # c < 1
    lnN_max = np.log(1e300)  # c > 1

    # Проверка знаков на концах
    c_min, _ = compute_c(np.exp(lnN_min), max_iter=25, verbose=False)
    c_max, _ = compute_c(np.exp(lnN_max), max_iter=25, verbose=False)

    print(f"c({np.exp(lnN_min):.1e}) = {c_min:.6f}")
    print(f"c({np.exp(lnN_max):.1e}) = {c_max:.6f}")

    if (c_min - c_target) * (c_max - c_target) > 0:
        print("Ошибка: c(N) не пересекает целевое значение в заданном диапазоне!")
        return None, None, None

    # Бисекция
    for i in range(max_iter):
        lnN_mid = (lnN_min + lnN_max) / 2
        N_mid = np.exp(lnN_mid)
        c_mid, K_mid = compute_c(N_mid, max_iter=25, verbose=False)

        if abs(c_mid - c_target) < tol:
            return N_mid, c_mid, K_mid

        if (c_min - c_target) * (c_mid - c_target) < 0:
            lnN_max = lnN_mid
            c_max = c_mid
        else:
            lnN_min = lnN_mid
            c_min = c_mid

    N_mid = np.exp((lnN_min + lnN_max) / 2)
    c_mid, K_mid = compute_c(N_mid, max_iter=25, verbose=False)
    return N_mid, c_mid, K_mid


# ОСНОВНАЯ ПРОГРАММА
def main():
    print("ПРОВЕРКА ГИПОТЕЗЫ: c* = 1 КАК ОСОБАЯ ТОЧКА, K* = 5")

    # 1. Находим N, при котором c(N) = 1
    N1, c1, K1 = find_N_for_c_target(c_target=1.0, tol=1e-6)

    if N1 is not None:
        print("РЕЗУЛЬТАТЫ ПОИСКА N₁ (c=1)")
        print(f"  N₁ = {N1:.6e}")
        print(f"  c(N₁) = {c1:.6f}")
        print(f"  K(N₁) = {K1:.6f}")

        # Проверка близости K к целому числу
        K_rounded = round(K1)
        diff_K = abs(K1 - K_rounded)
        print(f"\n  Ближайшее целое K = {K_rounded}")
        print(f"  Отклонение = {diff_K:.6f}")

        if diff_K < 0.01:
            print("  ✅ K близко к целому числу! Гипотеза подтверждается.")
        else:
            print("  ⚠️ K не близко к целому числу. Гипотеза требует проверки.")

        # Проверка, является ли K_rounded = 5
        if K_rounded == 5:
            print("  ✅ K = 5! Это целое число, что подтверждает гипотезу о фазовом переходе.")
        else:
            print(f"  ❌ K = {K_rounded}, а ожидалось 5.")

    # 2. Построение графиков для визуализации
    print("ПОСТРОЕНИЕ ГРАФИКОВ ЗАВИСИМОСТЕЙ")

    # Диапазон N для графиков
    N_range = np.logspace(20, 300, 15)
    c_vals = []
    K_vals = []
    x_vals = []

    print("\nВычисление c(N) для разных N:")
    for N in N_range:
        c, K = compute_c(N, max_iter=25, verbose=False)
        x = c * (np.log(N)) ** (1.0 / 3.0)
        c_vals.append(c)
        K_vals.append(K)
        x_vals.append(x)
        print(f"N = {N:.2e}, c = {c:.6f}, K = {K:.6f}, x = {x:.6f}")

    # График 1: c(N)
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.semilogx(N_range, c_vals, 'o-', color='blue', linewidth=2)
    plt.axhline(y=1.0, color='red', linestyle='--', label='c = 1')
    if N1 is not None:
        plt.axvline(x=N1, color='green', linestyle='--', alpha=0.7, label=f'N₁ = {N1:.2e}')
    plt.xlabel('N', fontsize=12)
    plt.ylabel('c(N)', fontsize=12)
    plt.title('Зависимость c(N)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()

    # График 2: K(N)
    plt.subplot(1, 3, 2)
    plt.semilogx(N_range, K_vals, 'o-', color='green', linewidth=2)
    plt.axhline(y=5.0, color='red', linestyle='--', label='K = 5')
    if N1 is not None:
        plt.axvline(x=N1, color='green', linestyle='--', alpha=0.7)
    plt.xlabel('N', fontsize=12)
    plt.ylabel('K(N)', fontsize=12)
    plt.title('Зависимость K(N)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()

    # График 3: K(c)
    plt.subplot(1, 3, 3)
    plt.plot(c_vals, K_vals, 'o-', color='purple', linewidth=2)
    plt.axhline(y=5.0, color='red', linestyle='--', label='K = 5')
    plt.axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label='c = 1')
    plt.xlabel('c', fontsize=12)
    plt.ylabel('K', fontsize=12)
    plt.title('Зависимость K(c)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.show()

    # 3. Вывод интерполяционной формулы для c(N)
    print("ИНТЕРПОЛЯЦИЯ ЗАВИСИМОСТИ c(N)")

    # Преобразуем для линейной регрессии: 1/c^3 vs 1/lnN
    x_fit = 1.0 / np.log(N_range)
    y_fit = 1.0 / np.array(c_vals) ** 3

    # Линейная регрессия
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_fit, y_fit)

    print(f"\n  Аппроксимация: 1/c³ = {intercept:.6f} + {slope:.6f} * (1/lnN)")
    print(f"  R² = {r_value ** 2:.6f}")

    # Экстраполяция к N -> infinity (1/lnN -> 0)
    c_inf = (1.0 / intercept) ** (1.0 / 3.0)
    print(f"\n  Экстраполяция при N → ∞: c* ≈ {c_inf:.6f}")

    # 4. Проверка гипотезы о K=5 при c=1
    if N1 is not None:
        print("ВЫВОДЫ ПО ГИПОТЕЗЕ")
        if abs(K1 - 5.0) < 0.01:
            print("  ✅ ГИПОТЕЗА ПОДТВЕРЖДАЕТСЯ: при c=1, K=5!")
            print(f"  N₁ = {N1:.6e} — количество узлов глобального информационного поля.")
        else:
            print(f"  ⚠️ Гипотеза не подтверждается: K = {K1:.4f}, а ожидалось 5.")
            print("  Возможно, K стремится к 5 при другом значении c.")


if __name__ == "__main__":
    main()