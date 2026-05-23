import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
from itertools import product

# БАЗОВЫЕ ПАРАМЕТРЫ МОДЕЛИ
d = 3                      # размерность пространства
K0 = 2 * d                 # связность идеальной решётки (6)
alpha = 1.0                # стоимость локальной связи
beta = 1.0 / (2 * d)       # удорожание дальних связей
T = 0.5                    # температура (энтропия)

N = 1e184                  # число узлов (голографическая энтропия)
lnN = np.log(N)

# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
def entropy(p):
    """Энтропия Шеннона для вероятности p"""
    p = np.clip(p, 1e-300, 1 - 1e-12)
    return -p * np.log(p) - (1 - p) * np.log(1 - p)

def solve_K(x, gamma, eta, kappa, lambda_spec):
    """
    Решает нелинейное уравнение для K при фиксированных параметрах
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
        sol = root_scalar(equation, bracket=[K0 - 2.0, K0 + 4.0], method='bisect')
        return sol.root
    except:
        from scipy.optimize import fsolve
        return fsolve(equation, K0)[0]

def solve_c(gamma, eta, kappa, lambda_spec, max_iter=30, tol=1e-8, verbose=False):
    """
    Ищет фиксированную точку c = x / (ln N)^{1/3}
    """
    # Начальное приближение (из упрощённой модели)
    c = (gamma / (2.0 * eta)) ** (1.0 / 3.0)

    for i in range(max_iter):
        x = c * (lnN) ** (1.0 / 3.0)
        K = solve_K(x, gamma, eta, kappa, lambda_spec)
        c_new = np.sqrt(gamma * (lnN) ** (1.0 / 3.0) / (K + 2.0 * eta * x))

        if abs(c_new - c) < tol:
            return c_new, K
        c = c_new

    return c, K

# ИССЛЕДОВАНИЕ УСТОЙЧИВОСТИ
def scan_parameter(param_name, base_value, variations,
                   gamma_base=2.0, eta_base=0.1, kappa_base=1.0, lambda_base=0.01):
    """
    Сканирует один параметр, остальные фиксированы
    """
    results = []
    for val in variations:
        if param_name == 'gamma':
            gamma = val
            eta = eta_base
            kappa = kappa_base
            lambda_spec = lambda_base
        elif param_name == 'eta':
            gamma = gamma_base
            eta = val
            kappa = kappa_base
            lambda_spec = lambda_base
        elif param_name == 'kappa':
            gamma = gamma_base
            eta = eta_base
            kappa = val
            lambda_spec = lambda_base
        elif param_name == 'lambda_spec':
            gamma = gamma_base
            eta = eta_base
            kappa = kappa_base
            lambda_spec = val
        else:
            continue

        try:
            c_star, K_star = solve_c(gamma, eta, kappa, lambda_spec, verbose=False)
            results.append((val, c_star, K_star))
            print(f"  {param_name} = {val:.6f} -> c* = {c_star:.6f}, K* = {K_star:.6f}")
        except Exception as e:
            print(f"  {param_name} = {val:.6f} -> ОШИБКА: {e}")
            results.append((val, np.nan, np.nan))

    return results

def run_sensitivity_analysis():
    """
    Полный анализ чувствительности
    """
    print("АНАЛИЗ ЧУВСТВИТЕЛЬНОСТИ RG-КОНСТАНТЫ c*")

    # Базовые параметры
    gamma0 = 0.657
    eta0 = 0.1
    kappa0 = 1.0
    lambda0 = 0.01

    # Диапазоны вариаций (10% от базового значения)
    gamma_vals = gamma0 * np.linspace(0.657, 1.0, 5)
    eta_vals = eta0 * np.linspace(0.9, 1.1, 5)
    kappa_vals = kappa0 * np.linspace(0.9, 1.1, 5)
    lambda_vals = lambda0 * np.linspace(0.9, 1.1, 5)

    # Словарь для хранения результатов
    all_results = {}

    # 1. Вариация gamma
    print("\n1. ВАРИАЦИЯ gamma (стоимость транспорта):")
    all_results['gamma'] = scan_parameter('gamma', gamma0, gamma_vals,
                                          gamma_base=gamma0, eta_base=eta0,
                                          kappa_base=kappa0, lambda_base=lambda0)

    # 2. Вариация eta
    print("\n2. ВАРИАЦИЯ eta (нелинейное насыщение):")
    all_results['eta'] = scan_parameter('eta', eta0, eta_vals,
                                        gamma_base=gamma0, eta_base=eta0,
                                        kappa_base=kappa0, lambda_base=lambda0)

    # 3. Вариация kappa
    print("\n3. ВАРИАЦИЯ kappa (жёсткость геометрии):")
    all_results['kappa'] = scan_parameter('kappa', kappa0, kappa_vals,
                                          gamma_base=gamma0, eta_base=eta0,
                                          kappa_base=kappa0, lambda_base=lambda0)

    # 4. Вариация lambda_spec
    print("\n4. ВАРИАЦИЯ lambda_spec (спектральный коэффициент):")
    all_results['lambda_spec'] = scan_parameter('lambda_spec', lambda0, lambda_vals,
                                                gamma_base=gamma0, eta_base=eta0,
                                                kappa_base=kappa0, lambda_base=lambda0)

    return all_results

# ВИЗУАЛИЗАЦИЯ
def plot_sensitivity(results):
    """
    Строит графики зависимости c* от параметров
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axs = axes.flatten()

    param_names = ['gamma', 'eta', 'kappa', 'lambda_spec']
    titles = ['γ (стоимость транспорта)', 'η (нелинейное насыщение)',
              'κ (жёсткость геометрии)', 'λ_spec (спектральный коэффициент)']

    for i, (param, title) in enumerate(zip(param_names, titles)):
        ax = axs[i]
        data = results[param]
        vals = [d[0] for d in data]
        c_vals = [d[1] for d in data]

        ax.plot(vals, c_vals, 'o-', linewidth=2, markersize=8, color='blue')
        ax.axhline(y=c_vals[2], color='red', linestyle='--', alpha=0.5,
                   label=f'c* = {c_vals[2]:.4f} (базовое)')
        ax.set_xlabel(title, fontsize=12)
        ax.set_ylabel('c*', fontsize=12)
        ax.set_title(f'Зависимость c* от {param}', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.show()

def plot_c_vs_N_evolution():
    """
    Исследует эволюцию c(N) для разных параметров
    """
    print("ЭВОЛЮЦИЯ c(N) ПРИ РАЗНЫХ ПАРАМЕТРАХ")

    N_range = np.logspace(20, 184, 10)
    gamma0, eta0, kappa0, lambda0 = 0.657, 0.1, 1.0, 0.01

    # Варианты параметров
    variants = [
        (gamma0, eta0, kappa0, lambda0, "Базовый"),
        (gamma0 * 1.1, eta0, kappa0, lambda0, "γ +10%"),
        (gamma0, eta0 * 1.1, kappa0, lambda0, "η +10%"),
        (gamma0, eta0, kappa0 * 1.1, lambda0, "κ +10%"),
        (gamma0, eta0, kappa0, lambda0 * 1.1, "λ_spec +10%"),
    ]

    plt.figure(figsize=(10, 6))

    for gamma, eta, kappa, lam, label in variants:
        c_vals = []
        for Ni in N_range:
            # Временно меняем глобальное N
            global N, lnN
            N = Ni
            lnN = np.log(N)
            try:
                c_star, _ = solve_c(gamma, eta, kappa, lam, max_iter=25, verbose=False)
                c_vals.append(c_star)
            except:
                c_vals.append(np.nan)
        # Восстанавливаем исходное N
        N = 1e184
        lnN = np.log(N)
        plt.semilogx(N_range, c_vals, 'o-', linewidth=2, label=label)

    plt.xlabel('N', fontsize=12)
    plt.ylabel('c(N)', fontsize=12)
    plt.title('Эволюция c(N) при вариации параметров', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

# ОСНОВНАЯ ПРОГРАММА
def main():
    # 1. Анализ чувствительности
    results = run_sensitivity_analysis()

    # 2. Визуализация
    plot_sensitivity(results)

    # 3. Эволюция c(N)
    plot_c_vs_N_evolution()

    # 4. Вывод статистики
    print("СТАТИСТИКА ЧУВСТВИТЕЛЬНОСТИ")

    base_c = results['gamma'][2][1]  # центральное значение
    print(f"\nБазовое значение c* = {base_c:.6f}")

    for param in ['gamma', 'eta', 'kappa', 'lambda_spec']:
        data = results[param]
        vals = [d[0] for d in data]
        c_vals = [d[1] for d in data]
        # Исключаем NaN
        valid = [(v, c) for v, c in zip(vals, c_vals) if not np.isnan(c)]
        if valid:
            v_vals, c_vals_valid = zip(*valid)
            delta_c = max(c_vals_valid) - min(c_vals_valid)
            rel_delta = delta_c / base_c * 100
            print(f"\n{param}:")
            print(f"  Диапазон c*: {min(c_vals_valid):.6f} – {max(c_vals_valid):.6f}")
            print(f"  Относительный разброс: {rel_delta:.2f}%")

if __name__ == "__main__":
    main()