import numpy as np
from scipy.optimize import root_scalar

# ============================================================
# ПАРАМЕТРЫ МОДЕЛИ (калибруются из первых принципов)
# ============================================================
d = 3                      # размерность пространства
K0 = 2 * d                 # связность идеальной решётки (6)
alpha = 1.0                # стоимость локальной связи
beta = 1.0 / (2 * d)       # удорожание дальних связей
gamma = 1                  # стоимость транспорта
eta = 0.1                  # нелинейное насыщение
kappa = 1.0                # жёсткость геометрии
T = 0.5                    # температура (энтропия)
lambda_spec = 0.01         # спектральный коэффициент

N = 1e300                  # число узлов (голографическая энтропия)
lnN = np.log(N)

# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================
def entropy(p):
    """Энтропия Шеннона для вероятности p"""
    p = np.clip(p, 1e-300, 1 - 1e-12)
    return -p * np.log(p) - (1 - p) * np.log(1 - p)

# ============================================================
# РЕШЕНИЕ УРАВНЕНИЯ ДЛЯ K ПРИ ФИКСИРОВАННОМ x
# ============================================================
def solve_K(x):
    """
    Решает нелинейное уравнение для K:
    2κ(K-K0) + α(1+βx) - T·H(p) - λ_spec · (xK²/N^{1/d} - 1)/(xK²/N^{1/d} + 1)² = 0
    """
    p = x / N ** (1.0 / d)

    def equation(K):
        # Спектральный член (без упрощений)
        arg = x * K * K / N ** (1.0 / d)
        num = arg - 1.0
        den = (arg + 1.0) ** 2
        spec_term = lambda_spec * num / den

        return (2.0 * kappa * (K - K0) +
                alpha * (1.0 + beta * x) -
                T * entropy(p) -
                spec_term)

    # Поиск корня на интервале, где K > 0
    try:
        sol = root_scalar(equation, bracket=[K0 - 2.0, K0 + 4.0], method='bisect')
        return sol.root
    except:
        # Fallback: численное решение методом Ньютона (если брекетинг не сработал)
        from scipy.optimize import fsolve
        return fsolve(equation, K0)[0]

# ============================================================
# ИТЕРАЦИОННОЕ РЕШЕНИЕ ДЛЯ c
# ============================================================
def solve_c(max_iter=30, tol=1e-8, verbose=True):
    """
    Ищет фиксированную точку c = x / (ln N)^{1/3}
    с самосогласованием K = K(x)
    """
    # Начальное приближение (из упрощённой модели)
    c = (gamma / (2.0 * eta)) ** (1.0 / 3.0)
    if verbose:
        print(f"Начальное приближение c0 = {c:.6f}")

    for i in range(max_iter):
        x = c * (lnN) ** (1.0 / 3.0)

        # Самосогласование K
        K = solve_K(x)

        # Новое c из уравнения (1): K + 2ηx = γ lnN / x²
        # => γ lnN / x² = K + 2ηx
        # => x² = γ lnN / (K + 2ηx)
        # => c² (lnN)^{2/3} = γ lnN / (K + 2ηx)
        # => c² = γ (lnN)^{1/3} / (K + 2ηx)
        # => c = sqrt( γ (lnN)^{1/3} / (K + 2ηx) )
        c_new = np.sqrt(gamma * (lnN) ** (1.0 / 3.0) / (K + 2.0 * eta * x))

        if verbose:
            print(f"iter {i:2d}: c = {c:.6f}, K = {K:.6f}, Δc = {c_new - c:.2e}")

        if abs(c_new - c) < tol:
            c = c_new
            break
        c = c_new

    return c

# ============================================================
# ТЕСТЫ УСТОЙЧИВОСТИ И АНАЛИЗ
# ============================================================
def robustness():
    """Проверяет, что фиксированная точка не зависит от начальных условий и N"""
    print("ТЕСТ УСТОЙЧИВОСТИ")

    # Разные начальные c
    c_starts = [1.0, 2.0, 5.0, 10.0]
    results = []
    for c0 in c_starts:
        # Переопределяем solve_c с заданным начальным c
        def solve_c_from(c0):
            c = c0
            for i in range(30):
                x = c * (lnN) ** (1.0 / 3.0)
                K = solve_K(x)
                c_new = np.sqrt(gamma * (lnN) ** (1.0 / 3.0) / (K + 2.0 * eta * x))
                if abs(c_new - c) < 1e-8:
                    return c_new
                c = c_new
            return c
        results.append(solve_c_from(c0))

    print(f"\nФиксированные точки при разных c0:")
    for c0, c_fin in zip(c_starts, results):
        print(f"  c0 = {c0:.1f} -> c* = {c_fin:.6f}")

    # Разные N (степени 10)
    global N, lnN
    N_values = [1e20, 1e30, 1e60, 1e80, 1e100, 1e122, 1e184, 1e300]
    c_vs_N = []
    for Ni in N_values:
        N = Ni
        lnN = np.log(N)
        c_here = solve_c(max_iter=25, verbose=False)
        c_vs_N.append((Ni, c_here))
    # Восстанавливаем исходное N
    N = 1e300
    lnN = np.log(N)

    print(f"\nЗависимость c* от N:")
    for Ni, ci in c_vs_N:
        print(f"  N = {Ni:.1e} -> c* = {ci:.6f}")

    return results, c_vs_N

# ОСНОВНАЯ ПРОГРАММА
def main():
    print("САМОСОГЛАСОВАННАЯ RG-МОДЕЛЬ ВСЕЛЕННОЙ")

    # 1. Решение для c
    c_star = solve_c()
    x_star = c_star * (lnN) ** (1.0 / 3.0)
    K_star = solve_K(x_star)

    print("РЕЗУЛЬТАТЫ")
    print(f"  N                 = {N:.1e}")
    print(f"  c*                = {c_star:.6f}")
    print(f"  x* = p·N^(1/d)    = {x_star:.6f}")
    print(f"  K*                = {K_star:.6f}")

    # 2. Сравнение с известными константами
    print("СРАВНЕНИЕ С ИЗВЕСТНЫМИ КОНСТАНТАМИ")
    constants = {
        'π': np.pi,
        'e': np.e,
        '√(2π)': np.sqrt(2 * np.pi),
        'π/√2': np.pi / np.sqrt(2),
        'e/√2': np.e / np.sqrt(2),
        'plastic (ρ)': 1.324717957244746,
        'supergolden': 1.465571231876768,
        'Feigenbaum δ': 4.669201609102990,
    }
    print(f"\n  c* = {c_star:.6f}")
    for name, val in constants.items():
        diff = abs(c_star - val)
        rel = diff / val * 100
        print(f"  {name:15} = {val:.6f}  | diff = {diff:.6f} ({rel:.4f}%)")

    # 3. Тесты устойчивости
    robustness()

    # 4. Проверка на трансцендентность
    print("ПРОВЕРКА НА ТРАНСЦЕНДЕНТНОСТЬ")
    print(f"  c³ = {c_star ** 3:.6f}")
    print(f"  γ/(2η) = {gamma / (2 * eta):.6f}")
    # Вычисляем эффективное γ/(2η) из самосогласования
    eff_ratio = c_star ** 3
    print(f"  Отношение γ/(2η) должно быть {eff_ratio:.6f} для данной c*")

    # Если есть значительное отклонение от простого алгебраического соотношения,
    # это указывает на вклад поддоминантных членов (спектр, энтропия).
    if abs(eff_ratio - gamma / (2 * eta)) > 0.01:
        print("  ⚠️ Наблюдается значительный вклад поддоминантных членов!")
    else:
        print("  ✅ Основной вклад даёт баланс транспорта и нелинейности.")

if __name__ == "__main__":
    main()