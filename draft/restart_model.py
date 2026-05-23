"""
ФИНАЛЬНАЯ МОДЕЛЬ ПНИД — ИСПРАВЛЕННОЕ АНАЛИТИЧЕСКОЕ ПРИБЛИЖЕНИЕ
Решается правильное уравнение стационарности: K + 2η·x = γ·ln N / x²
"""

import math
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ПАРАМЕТРЫ МОДЕЛИ

class UniverseParameters:
    def __init__(self, d=3):
        self.d = d
        self.alpha = 1.0          # стоимость локальной связи
        self.beta = 1.0 / (2 * d) # удорожание дальних связей
        self.gamma = 2.0          # стоимость транспорта
        self.T = 0.5              # температура
        self.delta = 1.0          # вес энтропии
        self.lambda_spec = 0.01   # спектральная энергия
        self.kappa = 59.0          # геометрическая жесткость
        self.eta = 0.1            # коэффициент нелинейности (crowding)

# БАЗОВЫЕ ФУНКЦИИ

def entropy_density(p):
    """Локальная энтропия Шеннона."""
    if p < 1e-100:
        return -p * np.log(p + 1e-300)
    p_safe = np.clip(p, 1e-100, 1 - 1e-100)
    return -p_safe * np.log(p_safe) - (1 - p_safe) * np.log(1 - p_safe)

def L_NW(x, d):
    """Точная формула Ньюмана-Ваттса."""
    if x < 1e-6:
        return (2 / d) * (0.5 + x / 12)
    argument = np.clip(x / (x + 2), 0, 1 - 1e-12)
    return (2 / d) * (1 / x) * np.arctanh(argument)

def structural_energy_per_node(K, p, N, params):
    """Структурная энергия на узел (с насыщением дальних связей)."""
    x = p * N ** (1 / params.d)
    E_local = params.alpha * K
    E_long = params.alpha * params.beta * K * x + params.eta * x**2
    K_natural = 2 * params.d
    E_geom = params.kappa * (K - K_natural) ** 2
    return E_local + E_long + E_geom

def transport_energy_per_node(p, N, params):
    """Транспортная энергия с логарифмическим масштабированием."""
    x = p * N ** (1 / params.d)
    return params.gamma * np.log(float(N)) / (x + 1e-12)

def spectral_energy_per_node(K, p, params):
    """Спектральная энергия."""
    lambda_gap = p * K + 1.0 / (K + 1e-6) + 1e-12
    return params.lambda_spec / lambda_gap

def total_action_per_node(K, p, N, params):
    """Полное информационное действие на узел."""
    S_struct = structural_energy_per_node(K, p, N, params)
    S_trans = transport_energy_per_node(p, N, params)
    S_entropy = -params.T * params.delta * K * entropy_density(p)
    S_spec = spectral_energy_per_node(K, p, params)

    if K < 3.0:
        penalty = 100.0 * (3.0 - K) ** 2
    else:
        penalty = 0.0

    return S_struct + S_trans + S_entropy + S_spec + penalty

# ============================================================================
# РЕШЕНИЕ УРАВНЕНИЯ СТАЦИОНАРНОСТИ ДЛЯ x
# ============================================================================

def solve_x_from_stationarity(log_N, K, params):
    """
    Решает уравнение стационарности для x:
    K + 2η·x = γ·log N / x²
    """
    eta = params.eta
    gamma = params.gamma

    def f(x):
        return K + 2 * eta * x - gamma * log_N / (x**2 + 1e-12)

    def df(x):
        return 2 * eta + 2 * gamma * log_N / (x**3 + 1e-12)

    # Начальное приближение
    x = (gamma * log_N / (K + 2 * eta)) ** (1/3)

    # Метод Ньютона
    for _ in range(30):
        fx = f(x)
        dfx = df(x)
        if abs(dfx) > 1e-12:
            x_new = x - fx / dfx
        else:
            x_new = x * 0.9 + 0.1 * (gamma * log_N / K) ** (1/3)
        x = max(x_new, 1e-6)

    return x

# ============================================================================
# АНАЛИТИЧЕСКОЕ ПРИБЛИЖЕНИЕ (ИСПРАВЛЕННОЕ)
# ============================================================================

def analytical_approximation(N, d=4):
    N = float(N)
    log_N = np.log(N)
    params = UniverseParameters(d)

    # правильный scaling
    x = np.sqrt(log_N)

    for _ in range(30):
        # K из геометрического баланса
        K = 2*d - params.alpha*(1 + params.beta*x)/(2*params.kappa)
        K = np.clip(K, 3.0, 8.0)

        # уточнение через логарифм
        log_x = np.log(x + 1e-12)

        rhs = params.gamma * log_N * log_x / (x**2 + 1e-12)
        lhs = params.alpha * params.beta * K + 2 * params.eta * x

        # корректировка x
        x = x * (rhs / (lhs + 1e-12))**0.5

    p = x / N**(1/d)
    return K, p, x

# ОПТИМИЗАЦИЯ

def find_optimal_parameters(N, params):
    """Поиск оптимальных K и p."""
    N_float = float(N)
    K_init, p_init, _ = analytical_approximation(N_float, params.d)

    if K_init < 3.0:
        K_init = 3.0
    if p_init < 1e-100:
        p_init = np.sqrt(np.log(N_float)) / N_float ** (1 / params.d)

    log_p_init = np.log(np.clip(p_init, 1e-100, 0.5))
    bounds = [(3.0, 10.0), (np.log(1e-100), np.log(0.5))]

    def objective(vars):
        K, log_p = vars
        return total_action_per_node(K, np.exp(log_p), N_float, params)

    best_result = None
    best_score = np.inf

    for scale in [1.0, 0.9, 1.1]:
        result = minimize(
            objective,
            [K_init * scale, log_p_init],
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 10000, 'ftol': 1e-14, 'gtol': 1e-14}
        )
        if result.success and result.fun < best_score:
            best_result = result
            best_score = result.fun

    if best_result is not None:
        return best_result.x[0], np.exp(best_result.x[1])
    else:
        print(f"  ⚠️ Оптимизация не сошлась для N={N:.2e}")
        return K_init, p_init

# СКАНИРОВАНИЕ МАСШТАБОВ

def scan_all_scales():
    """Полное сканирование масштабов N."""
    params = UniverseParameters(d=3)
    N_values = np.logspace(2, 184, 30)
    results = {'N': [], 'K': [], 'p': [], 'x': [], 'L_ratio': []}

    print("СКАНИРОВАНИЕ МАСШТАБОВ — ИСПРАВЛЕННЫЙ ПНИД")
    print(f"{'N':<15} {'K_opt':<10} {'p_opt':<15} {'x = p·N^(1/d)':<18} {'L/L_0':<12} {'Тип сети':<20}")

    for N in N_values:
        try:
            K_opt, p_opt = find_optimal_parameters(N, params)
            x_opt = p_opt * N ** (1 / params.d)

            if x_opt > 0.01:
                argument = np.clip(x_opt / (x_opt + 2), 0, 0.999999)
                L_NW_val = (2 / params.d) * (1 / x_opt) * np.arctanh(argument)
            else:
                L_NW_val = (2 / params.d) * 0.5
            L_ratio = L_NW_val / N ** (1 / params.d)

            results['N'].append(N)
            results['K'].append(K_opt)
            results['p'].append(p_opt)
            results['x'].append(x_opt)
            results['L_ratio'].append(L_ratio)

            if x_opt < 0.5:
                net_type = "Решетка"
            elif x_opt < 2.0:
                net_type = "Докритический SW"
            elif x_opt < 5.0:
                net_type = "КРИТИЧЕСКИЙ SW"
            elif x_opt < 20.0:
                net_type = "Сверхкритический SW"
            else:
                net_type = "Ультра SW"

            if N <= 1e6 or int(np.log10(N)) % 20 == 0:
                print(f"{N:<15.2e} {K_opt:<10.4f} {p_opt:<15.4e} {x_opt:<18.6f} {L_ratio:<12.6f} {net_type:<20}")

        except Exception as e:
            print(f"Ошибка для N={N:.2e}: {e}")

    return results, params

# ЗАПУСК

def main():
    print("ФИНАЛЬНАЯ МОДЕЛЬ ПНИД — ИСПРАВЛЕННАЯ АНАЛИТИКА")

    results, params = scan_all_scales()

    N_horizon = 0.3576e122
    K_opt, p_opt = find_optimal_parameters(N_horizon, params)
    x_opt = p_opt * N_horizon ** (1 / params.d)

    K_an, p_an, x_an = analytical_approximation(N_horizon, params.d)

    print(f"РЕЗУЛЬТАТЫ ДЛЯ N = 10^184")

    print(f"\n▶ АНАЛИТИЧЕСКОЕ ПРИБЛИЖЕНИЕ:")
    print(f"  • K ≈ {K_an:.6f}")
    print(f"  • p ≈ {p_an:.6e}")
    print(f"  • x ≈ {x_an:.6f}")

    print(f"\n▶ ЧИСЛЕННОЕ РЕШЕНИЕ:")
    print(f"  • K = {K_opt:.6f}")
    print(f"  • p = {p_opt:.6e}")
    print(f"  • x = {x_opt:.6f}")

    theory_x = np.sqrt(np.log(N_horizon))
    print(f"\n▶ ПРОВЕРКА ТЕОРИИ:")
    print(f"  • Теория: x = √(ln N) = {theory_x:.6f}")
    print(f"  • Численно: x = {x_opt:.6f}")
    if x_opt > 0:
        print(f"  • Совпадение: {min(x_opt, theory_x) / max(x_opt, theory_x) * 100:.1f}%")

    L_val = L_NW(x_opt, params.d)
    L_ratio = L_val / N_horizon ** (1 / params.d)
    H = entropy_density(p_opt)
    S_total = N_horizon * K_opt * H

    print(f"\n▶ ФИЗИЧЕСКИЕ СЛЕДСТВИЯ:")
    print(f"  • L/L₀ = {L_ratio:.6e}")

    R_universe_cm = 4.4e28
    L_eff_cm = L_ratio * R_universe_cm
    ly_cm = 9.46e17
    L_eff_ly = L_eff_cm / ly_cm

    print(f"  • Эффективный радиус связности: {L_eff_ly:.2e} св. лет")
    print(f"  • Радиус наблюдаемой Вселенной: 4.6e10 св. лет")

    if L_eff_ly < 4.6e10:
        print(f"  ✓ Вселенная ПРИЧИННО СВЯЗНА через small-world механизм!")
    else:
        print(f"  ✗ Требуется больше связей")

    print(f"  • Энтропия на связь: H(p) = {H:.6e} нат")
    print(f"  • Полная энтропия: S_total = {S_total:.2e} нат")
    print(f"  • S_total (биты) = {S_total / np.log(2):.2e} бит")

    if x_opt < 0.5:
        net_type = "Решеточный граф"
    elif x_opt < 2.0:
        net_type = "Докритический Small-World"
    elif x_opt < 5.0:
        net_type = "КРИТИЧЕСКИЙ Small-World"
    elif x_opt < 20.0:
        net_type = "Сверхкритический Small-World"
    else:
        net_type = "Ультра Small-World"

    print(f"\n▶ КЛАССИФИКАЦИЯ СЕТИ: {net_type}")

    print("ИТОГОВАЯ ТАБЛИЦА ПАРАМЕТРОВ МОДЕЛИ ВСЕЛЕННОЙ")
    print(f"""
    │ ПАРАМЕТР                         │ ЗНАЧЕНИЕ                     │
    │ Размерность d                    │ {params.d}                            │
    │ Число узлов N                    │ 0.3576e122                      │
    │ Оптимальная степень K            │ {K_opt:.4f}                       │
    │ Вероятность нелокальной связи p  │ {p_opt:.4e}                 │
    │ Критический параметр x = p·N^1/d │ {x_opt:.4f}                       │
    │ Относительная длина пути L/L₀    │ {L_ratio:.6e}                   │
    │ Полная энтропия графа (нат)      │ {S_total:.2e}                 │
    │ Тип сети                         │ {net_type}""")

    return results, K_opt, p_opt, x_opt

if __name__ == "__main__":
    results, K, p, x = main()