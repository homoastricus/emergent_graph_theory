import numpy as np
from scipy.optimize import root_scalar, fsolve
import matplotlib.pyplot as plt

# ============================================================
# ПАРАМЕТРЫ МОДЕЛИ (фиксированные)
# ============================================================
d = 3                      # размерность пространства
K0 = 2 * d                 # связность идеальной решётки (6)
alpha = 1.0                # стоимость локальной связи
beta = 1.0 / (2 * d)       # удорожание дальних связей
gamma = 1.0                # стоимость транспорта
eta = 0.1                  # нелинейное насыщение
T = 0.5                    # температура (энтропия)
lambda_spec = 0.01         # спектральный коэффициент

N = 1e300                  # достаточно большое N для асимптотики
lnN = np.log(N)

# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================
def entropy(p):
    """Энтропия Шеннона для вероятности p"""
    p = np.clip(p, 1e-300, 1 - 1e-12)
    return -p * np.log(p) - (1 - p) * np.log(1 - p)

def solve_K(x, kappa):
    """
    Решает нелинейное уравнение для K при фиксированных x и kappa
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
        sol = root_scalar(equation, bracket=[K0 - 4.0, K0 + 4.0], method='bisect')
        return sol.root
    except:
        return fsolve(equation, K0)[0]

def compute_c(kappa, max_iter=100, tol=1e-8, verbose=False):
    """
    Вычисляет фиксированную точку c(kappa) для данного kappa
    """
    # Начальное приближение (из упрощённой модели)
    c = (gamma / (2.0 * eta)) ** (1.0 / 3.0)

    for i in range(max_iter):
        x = c * (lnN) ** (1.0 / 3.0)
        K = solve_K(x, kappa)
        c_new = np.sqrt(gamma * (lnN) ** (1.0 / 3.0) / (K + 2.0 * eta * x))

        if verbose and i % 5 == 0:
            print(f"  iter {i:2d}: c = {c:.6f}, K = {K:.6f}")

        if abs(c_new - c) < tol:
            return c_new, K
        c = c_new

    return c, K

def find_optimal_kappa(kappa_range, criterion='c_max'):
    """
    Находит оптимальное kappa по заданному критерию.
    criterion: 'c_max', 'c_min', 'K_nearest_integer', 'dK_dkappa_max'
    """
    results = []
    c_vals = []
    K_vals = []

    print(f"\nСканирование kappa в диапазоне {kappa_range[0]:.2f} - {kappa_range[-1]:.2f}")
    print("-" * 60)

    for kappa in kappa_range:
        try:
            c, K = compute_c(kappa, max_iter=50, verbose=False)
            results.append((kappa, c, K))
            c_vals.append(c)
            K_vals.append(K)
            print(f"  κ = {kappa:.4f} -> c* = {c:.6f}, K* = {K:.6f}")
        except Exception as e:
            print(f"  κ = {kappa:.4f} -> ОШИБКА: {e}")
            results.append((kappa, np.nan, np.nan))
            c_vals.append(np.nan)
            K_vals.append(np.nan)

    # Поиск оптимального κ по критерию
    valid_results = [(k, c, K) for k, c, K in results if not np.isnan(c)]

    if not valid_results:
        print("Нет valid результатов!")
        return None, None, None, results

    if criterion == 'c_max':
        # Максимальное c (точка фазового перехода)
        best = max(valid_results, key=lambda x: x[1])
        print(f"\n✅ Оптимальное κ по критерию 'c_max': κ = {best[0]:.4f}, c* = {best[1]:.6f}, K* = {best[2]:.6f}")
    elif criterion == 'c_min':
        # Минимальное c
        best = min(valid_results, key=lambda x: x[1])
        print(f"\n✅ Оптимальное κ по критерию 'c_min': κ = {best[0]:.4f}, c* = {best[1]:.6f}, K* = {best[2]:.6f}")
    elif criterion == 'K_nearest_integer':
        # K ближайшее к целому (4,5,6)
        def dist_to_integer(K):
            return min(abs(K - 4), abs(K - 5), abs(K - 6))
        best = min(valid_results, key=lambda x: dist_to_integer(x[2]))
        print(f"\n✅ Оптимальное κ по критерию 'K_nearest_integer': κ = {best[0]:.4f}, c* = {best[1]:.6f}, K* = {best[2]:.6f}")
        print(f"   Ближайшее целое: {round(best[2])}")
    elif criterion == 'dK_dkappa_max':
        # Максимум производной dK/dκ (точка наибольшей чувствительности)
        kappa_vals = [r[0] for r in valid_results]
        K_vals_valid = [r[2] for r in valid_results]
        if len(kappa_vals) > 2:
            dK = np.gradient(K_vals_valid, kappa_vals)
            idx_max = np.argmax(np.abs(dK))
            best = valid_results[idx_max]
            print(f"\n✅ Оптимальное κ по критерию 'dK_dkappa_max': κ = {best[0]:.4f}, c* = {best[1]:.6f}, K* = {best[2]:.6f}")
        else:
            best = valid_results[-1]
            print(f"\n⚠️ Недостаточно точек для градиента. Использовано последнее значение: κ = {best[0]:.4f}")
    else:
        best = valid_results[-1]
        print(f"\n⚠️ Неизвестный критерий. Использовано последнее значение: κ = {best[0]:.4f}")

    return best[0], best[1], best[2], results

# ОСНОВНАЯ ПРОГРАММА
def main():
    print("ОПТИМИЗАЦИЯ ПАРАМЕТРА κ (ЖЁСТКОСТЬ ГЕОМЕТРИИ)")

    # Диапазон сканирования κ
    kappa_range = np.linspace(0.2, 10.0, 40)

    # Поиск оптимального κ по разным критериям
    criteria = ['c_max', 'c_min', 'K_nearest_integer', 'dK_dkappa_max']

    all_best = {}

    for crit in criteria:
        best_kappa, best_c, best_K, results = find_optimal_kappa(kappa_range, criterion=crit)
        all_best[crit] = (best_kappa, best_c, best_K)

    # Визуализация
    # Извлекаем результаты для последнего критерия (они одинаковы для всех, т.к. сканирование одно)
    _, _, _, results = find_optimal_kappa(kappa_range, criterion='c_max')
    kappa_vals = [r[0] for r in results]
    c_vals = [r[1] for r in results]
    K_vals = [r[2] for r in results]

    # Графики
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(kappa_vals, c_vals, 'o-', color='blue', linewidth=2)
    ax1.set_xlabel('κ (жёсткость геометрии)', fontsize=12)
    ax1.set_ylabel('c* (асимптотическое)', fontsize=12)
    ax1.set_title('Зависимость c* от κ', fontsize=14)
    ax1.grid(True, alpha=0.3)

    ax2.plot(kappa_vals, K_vals, 'o-', color='green', linewidth=2)
    ax2.axhline(y=5.0, color='red', linestyle='--', label='K = 5')
    ax2.axhline(y=4.0, color='orange', linestyle='--', label='K = 4')
    ax2.axhline(y=6.0, color='purple', linestyle='--', label='K = 6')
    ax2.set_xlabel('κ (жёсткость геометрии)', fontsize=12)
    ax2.set_ylabel('K* (асимптотическое)', fontsize=12)
    ax2.set_title('Зависимость K* от κ', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Вывод результатов
    print("СВОДКА ОПТИМАЛЬНЫХ ЗНАЧЕНИЙ κ")
    print(f"{'Критерий':<25} {'κ_opt':<12} {'c*':<12} {'K*':<12}")
    for crit, (k, c, K) in all_best.items():
        if k is not None:
            print(f"{crit:<25} {k:<12.4f} {c:<12.6f} {K:<12.6f}")

    # Анализ: к какому K стремится система при оптимальном κ?
    print("АНАЛИЗ: КАКОМУ K ТЯГОТЕЕТ СИСТЕМА?")

    # Находим κ, при котором K ближе всего к целым числам
    kappa_for_K4 = None
    kappa_for_K5 = None
    kappa_for_K6 = None
    min_dist4 = float('inf')
    min_dist5 = float('inf')
    min_dist6 = float('inf')

    for k, c, K in results:
        if not np.isnan(K):
            if abs(K - 4) < min_dist4:
                min_dist4 = abs(K - 4)
                kappa_for_K4 = k
            if abs(K - 5) < min_dist5:
                min_dist5 = abs(K - 5)
                kappa_for_K5 = k
            if abs(K - 6) < min_dist6:
                min_dist6 = abs(K - 6)
                kappa_for_K6 = k

    print(f"\n  K → 4 достигается при κ ≈ {kappa_for_K4:.4f} (ошибка {min_dist4:.6f})")
    print(f"  K → 5 достигается при κ ≈ {kappa_for_K5:.4f} (ошибка {min_dist5:.6f})")
    print(f"  K → 6 достигается при κ ≈ {kappa_for_K6:.4f} (ошибка {min_dist6:.6f})")

    # Определяем оптимальный κ как среднее между K=5 и K=4?
    if kappa_for_K4 is not None and kappa_for_K5 is not None:
        kappa_opt_guess = (kappa_for_K4 + kappa_for_K5) / 2
        print(f"\n  Предполагаемое оптимальное κ (между K=4 и K=5): {kappa_opt_guess:.4f}")

if __name__ == "__main__":
    main()