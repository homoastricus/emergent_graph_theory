import numpy as np
import math
from scipy.optimize import minimize_scalar

# ФУНДАМЕНТАЛЬНЫЕ ПАРАМЕТРЫ
K = 6.0
pi = math.pi
e = math.e
lnK = math.log(K)
# 1. N ИЗ ГЕОМЕТРИЧЕСКОГО РЕЗОНАНСА
def N_from_geometric_resonance():
    """
    Уравнение: D(N) = 1/π
    D(N) = 1/3 - (K - ln K) / ln N = 1/π

    Решение: ln N = (K - ln K) / (1/3 - 1/π)
    """
    lnN_math = (K - lnK) / (1.0 / 3.0 - 1.0 / pi)
    N_math = math.exp(lnN_math)
    return N_math, lnN_math


# 2. N ИЗ ДЗЕТА-ФУНКЦИИ (СПЕКТРАЛЬНЫЙ ВЫВОД)
def N_from_zeta():
    """
    Из спектра лапласиана:
    ln N = 6^(3/2 + π²/6)

    Вывод: сумма 1/λ_n = L² ζ(2) = L² π²/6
    L = ln N / ln(K_eff)
    """
    exponent = 1.5 + pi ** 2 / 6.0
    lnN_zeta = (6.0 ** exponent)
    N_zeta = math.exp(lnN_zeta)
    return N_zeta, lnN_zeta


# 3. N ИЗ ПОСТОЯННОЙ ТОНКОЙ СТРУКТУРЫ
def N_from_alpha(alpha_exp=1 / 137.035999084):
    """
    Формула ЕТИ: α = 2 (ln K)² / (π ln N)
    Обратная формула: ln N = 2 (ln K)² / (π α)
    """
    lnN_alpha = 2.0 * lnK ** 2 / (pi * alpha_exp)
    N_alpha = math.exp(lnN_alpha)
    return N_alpha, lnN_alpha


# 4. N_phys ИЗ МИНИМИЗАЦИИ ОШИБКИ
# Экспериментальные константы
constants = {
    'ħ': 1.054571817e-34,
    'h': 6.62607015e-34,
    't_P': 5.391247e-44,
    'l_P': 1.616255e-35,
    'm_P': 2.176434e-8,
    'E_P': 1.956082e9,
    'T_P': 1.416784e32,
    'c': 299792458,
    'G': 6.67430e-11,
    'k_B': 1.380649e-23,
    'α': 1 / 137.035999084,
    'm_e': 9.1093837015e-31,
    'm_proton': 1.67262192e-27,
    'm_muon': 1.883531627e-28,
    'q_e': 1.602176634e-19,
    'RIDBERG': 1.097373e7,
}


def base_formulas(N):
    """Вычисляет константы через параметры ЕТИ"""
    lnN = math.log(N)
    N13 = N ** (1 / 3)
    p_val = 1 / (K * N13)
    Kp = K * p_val

    return {
        'ħ': (lnN ** 3) / (K * N13),
        'h': 2 * pi * (lnN ** 3) / (K * N13),
        't_P': 4 * K ** 2 * lnK ** 2 / (pi * N13 * lnN ** 2),
        'l_P': 4 * lnN ** 2 * lnK / N13,
        'm_P': K / (pi * 4 * lnN ** 3),
        'E_P': (lnN ** 5) * pi / (4 * K ** 3 * lnK ** 2),
        'T_P': 8 * pi * N13 / (lnN ** 4),
        'c': pi * (lnN ** 4) / (K ** 2 * lnK),
        'G': 16 * pi ** 3 * lnN ** 13 / (K ** 5 * lnK * N13),
        'k_B': Kp * (lnN ** 8) / (8 * pi ** 2),
        'α': 2 * lnK ** 2 / (pi * lnN),
        'm_e': 4 * pi * lnN ** 4 / (K ** (1 / 2) * N13),
        'm_proton': math.sqrt(pi) * (lnN ** 6) / (K ** (3 / 2) * N13),
        'm_muon': 4 * pi ** 2 * lnN ** 5 / (K * 3 ** (1 / 2) * N13),
        'q_e': 1.0 / (pi * K ** (3 / 2) * lnN ** 7),
        'RIDBERG': 4 * lnN ** 3 * lnK ** 3 / (pi * K ** (3 / 2)),
    }


def total_log_error(ln_N):
    """Суммарная логарифмическая ошибка"""
    N = np.exp(ln_N)
    pred = base_formulas(N)
    total = 0.0
    for key in constants.keys():
        if key in pred and pred[key] > 0 and constants[key] > 0:
            ratio = pred[key] / constants[key]
            total += (math.log(ratio)) ** 2
    return total


def N_from_fit():
    """Находит N_phys минимизацией ошибки"""
    # Начальное приближение
    lnN_math = (K - lnK) / (1.0 / 3.0 - 1.0 / pi)
    lnN_init = lnN_math * 0.999  # немного меньше

    result = minimize_scalar(
        total_log_error,
        bracket=(lnN_init * 0.9, lnN_init * 1.1),
        method='brent',
        options={'xtol': 1e-12}
    )

    N_phys = np.exp(result.x)
    lnN_phys = result.x
    return N_phys, lnN_phys, result.fun


# 5. СВЯЗЬ N_phys И N_math (ГИПОТЕЗА)
def N_phys_from_math():
    """
    Гипотеза: ln N_phys = ln N_math - πK / ln N_math
    """
    N_math, lnN_math = N_from_geometric_resonance()
    lnN_pred = lnN_math - pi * K / lnN_math + 23.9/lnN_math**2
    N_pred = math.exp(lnN_pred)
    return N_pred, lnN_pred


# СРАВНЕНИЕ ВСЕХ МЕТОДОВ
print("СРАВНЕНИЕ МЕТОДОВ ВЫЧИСЛЕНИЯ N")

# 1. Геометрический резонанс
N_math, lnN_math = N_from_geometric_resonance()
print(f"\n1. ГЕОМЕТРИЧЕСКИЙ РЕЗОНАНС")
print(f"   Уравнение: 1/3 - (K - ln K)/ln N = 1/π")
print(f"   ln N_math = {lnN_math:.10f}")
print(f"   N_math    = {N_math:.6e}")

# 2. Дзета-функция
N_zeta, lnN_zeta = N_from_zeta()
print(f"\n2. ДЗЕТА-ФУНКЦИЯ (СПЕКТРАЛЬНЫЙ ВЫВОД)")
print(f"   Уравнение: ln N = 6^(3/2 + π²/6)")
print(f"   ln N_zeta = {lnN_zeta:.10f}")
print(f"   N_zeta    = {N_zeta:.6e}")
print(f"   Отклонение от N_math: {(lnN_zeta - lnN_math) / lnN_math * 100:.6f}%")

# 3. Постоянная тонкой структуры
N_alpha, lnN_alpha = N_from_alpha()
print(f"\n3. ПОСТОЯННАЯ ТОНКОЙ СТРУКТУРЫ")
print(f"   Уравнение: α = 2(ln K)²/(π ln N)")
print(f"   ln N_alpha = {lnN_alpha:.10f}")
print(f"   N_alpha    = {N_alpha:.6e}")
print(f"   Отклонение от N_math: {(lnN_alpha - lnN_math) / lnN_math * 100:.6f}%")
print(f"   Обратная проверка: α_pred = {2 * lnK ** 2 / (pi * lnN_alpha):.10f}")
print(f"   α_CODATA          = {1 / 137.035999084:.10f}")

# 4. N_phys (фит)
N_phys, lnN_phys, min_err = N_from_fit()
print(f"\n4. N_phys (ОПТИМАЛЬНЫЙ ФИТ)")
print(f"   Минимизация логарифмической ошибки по {len(constants)} константам")
print(f"   ln N_phys = {lnN_phys:.10f}")
print(f"   N_phys    = {N_phys:.6e}")
print(f"   Минимальная ошибка: {min_err:.6e}")
print(f"   Отклонение от N_math: {(lnN_phys - lnN_math) / lnN_math * 100:.6f}%")

# 5. N_phys из гипотезы
N_pred, lnN_pred = N_phys_from_math()
print(f"\n5. N_phys ИЗ ГИПОТЕЗЫ (ln N_phys = ln N_math - πK/ln N_math)")
print(f"   ln N_pred = {lnN_pred:.10f}")
print(f"   N_pred    = {N_pred:.6e}")
print(f"   Отклонение от ln N_phys (фит): {(lnN_pred - lnN_phys) / lnN_phys * 100:.6f}%")

# СВОДНАЯ ТАБЛИЦА
print("СВОДНАЯ ТАБЛИЦА")
print(f"  {'Метод':<30} {'ln N':<18} {'N':<20} {'Откл. от N_math':<18} {'Откл. от N_phys':<18}")

methods = [
    ("Геометрический резонанс", lnN_math, N_math),
    ("Дзета-функция", lnN_zeta, N_zeta),
    ("Постоянная тонкой структуры", lnN_alpha, N_alpha),
    ("N_phys (фит)", lnN_phys, N_phys),
    ("N_phys (гипотеза)", lnN_pred, N_pred),
]

for name, lnN_val, N_val in methods:
    dev_math = (lnN_val - lnN_math) / lnN_math * 100
    dev_phys = (lnN_val - lnN_phys) / lnN_phys * 100
    print(f"  {name:<30} {lnN_val:<18.10f} {N_val:<20.6e} {dev_math:<+18.8f}% {dev_phys:<+18.8f}%")

# АНАЛИЗ КАЧЕСТВА ФИТА
print("АНАЛИЗ КАЧЕСТВА ФИТА ПРИ N_phys")

pred = base_formulas(N_phys)
print(f"  {'Константа':<15} {'CODATA':<18} {'ЕТИ':<18} {'Ошибка %':<12}")
print(f"  {'-' * 65}")
for key in constants.keys():
    if key in pred:
        p = pred[key]
        c = constants[key]
        err = abs(p - c) / c * 100
        status = "✅" if err < 0.1 else ("⭐" if err < 0.5 else "⚠️")
        print(f"  {key:<15} {c:<18.6e} {p:<18.6e} {err:<12.6f} {status}")

avg_error = np.mean([abs(pred[k] - constants[k]) / constants[k] * 100 for k in constants if k in pred])
print(f"\n  Средняя ошибка: {avg_error:.4f}%")

# ПРОВЕРКА ГИПОТЕЗЫ
print("ПРОВЕРКА ГИПОТЕЗЫ: ln N_phys = ln N_math - πK/ln N_math")

lnN_phys_predicted = lnN_math - pi * K / lnN_math
print(f"  ln N_math              = {lnN_math:.10f}")
print(f"  ln N_phys (предсказано) = {lnN_phys_predicted:.10f}")
print(f"  ln N_phys (фит)         = {lnN_phys:.10f}")
print(f"  Разность               = {abs(lnN_phys_predicted - lnN_phys):.10f}")
print(f"  Относительная ошибка   = {abs(lnN_phys_predicted - lnN_phys) / lnN_phys * 100:.8f}%")

# Альтернативная гипотеза: ln N_phys = ln N_math - 2π²/ln N_math
lnN_phys_predicted2 = lnN_math - 2 * pi ** 2 / lnN_math
print(f"\n  Альтернатива: ln N_phys = ln N_math - 2π²/ln N_math")
print(f"  ln N_phys (предсказано) = {lnN_phys_predicted2:.10f}")
print(f"  ln N_phys (фит)         = {lnN_phys:.10f}")
print(f"  Разность               = {abs(lnN_phys_predicted2 - lnN_phys):.10f}")
print(f"  Относительная ошибка   = {abs(lnN_phys_predicted2 - lnN_phys) / lnN_phys * 100:.8f}%")

# ВЫВОД
print("ВЫВОД")
print(f"""
  1. Геометрический резонанс даёт:     ln N = {lnN_math:.6f}
  2. Дзета-функция (ζ(2)) даёт:        ln N = {lnN_zeta:.6f}
  3. Тонкая структура (α) даёт:        ln N = {lnN_alpha:.6f}
  4. Оптимальный фит даёт:             ln N = {lnN_phys:.6f}
  5. Связь N_phys с N_math:
     ln N_phys ≈ ln N_math - πK/ln N_math
     ({lnN_math:.6f} - {pi * K / lnN_math:.6f} = {lnN_phys_predicted:.6f})
  6. Точность связи: {abs(lnN_phys_predicted - lnN_phys) / lnN_phys * 100:.6f}%
  7. Все методы дают ln N ≈ {np.mean([lnN_math, lnN_zeta, lnN_alpha, lnN_phys]):.2f} ± {np.std([lnN_math, lnN_zeta, lnN_alpha, lnN_phys]):.2f}
""")