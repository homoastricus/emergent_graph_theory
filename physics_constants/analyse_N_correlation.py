import math
import numpy as np

K = 6.0
pi = math.pi
lnK = math.log(K)


# Функция для вычисления одной формулы без фактора
def compute_raw(N, formula_name):
    lnN = math.log(N)
    N13 = N ** (1 / 3)

    formulas = {
        'm_W_to_m_Z': pi ** (1 / 2) / 2,
        'm_Higgs_to_m_W': 2 * K ** (1 / 2) / pi,
        'α': 2 * lnK ** 2 / (pi * lnN),
        'm_proton_to_m_electron': lnN ** 2 / (4 * pi ** (1 / 2) * K),
        'm_tau_m_electron': K ** (5 / 2) * lnN / (4 * pi ** (1 / 2)),
        'm_plank_to_m_e': K ** (3 / 2) * N13 / (16 * pi ** 2 * lnN ** 7),
        'ħ': lnN ** 3 / (K * N13),
        'c': pi * lnN ** 4 / (K ** 2 * lnK),
        'G': 16 * pi ** 3 * lnN ** 13 / (K ** 5 * lnK * N13),
        'k_B': (1 / (K * N13)) * (lnN ** 8) / (8 * pi ** 2),  # Kp = 1/N^(1/3)
        'l_P': 4 * lnN ** 2 * lnK / N13,
        't_P': 4 * K ** 2 * lnK ** 2 / (pi * N13 * lnN ** 2),
        'm_P': K / (pi * 4 * lnN ** 3),
        'E_P': lnN ** 5 * pi / (4 * K ** 3 * lnK ** 2),
        'T_P': 8 * pi * N13 / lnN ** 4,
        'h': 2 * pi * lnN ** 3 / (K * N13),
        'q_e': 1.0 / (pi * K ** (3 / 2) * lnN ** 7),
        'ep_0': N13 / (8 * pi ** 3 * lnK * lnN ** 20),
        'mu_0': 8 * pi * K ** 4 * lnK ** 3 * lnN ** 12 / N13,
        'impedance': 8 * K ** 2 * pi ** 2 * lnK ** 2 * lnN ** 16 / N13,
        'RIDBERG': 4 * lnN ** 3 * lnK ** 3 / (pi * K ** (3 / 2)),
        'bor_radius': K ** (3 / 2) / (8 * pi * lnN ** 4 * lnK),
        'compton_e': K ** (3 / 2) * lnK / (2 * pi * lnN ** 5),
        'compton_proton': 2 * K ** (5 / 2) * lnK / (pi ** (1 / 2) * lnN ** 7),
        'Φ0_magnetic_stream': lnN ** 10 * pi ** 2 * K ** (1 / 2) / N13,
        'Lambda_cosmo': lnN ** 12 / (pi ** (1 / 2) * N ** (2 / 3)),
        'Einstein_constant': 128 * K ** 3 * lnK ** 3 / (lnN ** 3 * N13),
        'vacuum_higgs': lnN ** 6 * 8 * pi ** (3 / 2) / (2 ** (1 / 2) * N13),
    }

    return formulas.get(formula_name, None)


# Экспериментальные значения
experimental = {
    'm_W_to_m_Z': 0.8815,
    'm_Higgs_to_m_W': 1.558,
    'α': 1 / 137.035999084,
    'm_proton_to_m_electron': 1836.152673426,
    'm_tau_m_electron': 3477,
    'm_plank_to_m_e': 2.389e22,
    'ħ': 1.054571817e-34,
    'c': 299792458,
    'G': 6.67430e-11,
    'k_B': 1.380649e-23,
    'l_P': 1.616255e-35,
    't_P': 5.391247e-44,
    'm_P': 2.176434e-8,
    'E_P': 1.956082e9,
    'T_P': 1.416784e32,
    'h': 6.62607015e-34,
    'q_e': 1.602176634e-19,
    'ep_0': 8.8725366415e-12,
    'mu_0': 1.25663706127e-6,
    'impedance': 376.730313,
    'RIDBERG': 1.097373e7,
    'bor_radius': 5.29177210903e-11,
    'compton_e': 2.426e-12,
    'compton_proton': 1.32140985396e-15,
    'Φ0_magnetic_stream': 2.06783366752e-15,
    'Lambda_cosmo': 1.08929e-52,
    'Einstein_constant': 2.07664746e-43,
    'vacuum_higgs': 4.388471e-25,
}

# Анализ зависимости от N
print("=" * 90)
print("АНАЛИЗ ЗАВИСИМОСТИ КОНСТАНТ ОТ N")
print("=" * 90)

N_base = 4.197668e121
N_test = N_base * 1.001  # небольшое изменение N

print(
    f"\n{'Константа':<25} {'Значение при N_base':<20} {'Значение при N_test':<20} {'Изменение %':<15} {'Зависит от N':<15}")
print("-" * 95)

independent_constants = []
dependent_constants = []

for name in experimental.keys():
    val_base = compute_raw(N_base, name)
    val_test = compute_raw(N_test, name)

    if val_base is not None and val_test is not None and val_base != 0:
        change = abs(val_test - val_base) / abs(val_base) * 100
        depends = "ДА" if change > 0.01 else "НЕТ"

        if change <= 0.01:
            independent_constants.append(name)
        else:
            dependent_constants.append(name)

        print(f"{name:<25} {val_base:<20.6e} {val_test:<20.6e} {change:<15.6f} {depends:<15}")

# Анализ независимых от N констант
print(f"\n{'=' * 90}")
print("КОНСТАНТЫ, НЕ ЗАВИСЯЩИЕ ОТ N (ТОПОЛОГИЧЕСКИЕ ИНВАРИАНТЫ)")
print("=" * 90)

print(f"\n{'Константа':<25} {'ЕТИ (аналит.)':<20} {'Эксперимент':<20} {'Отклонение %':<15} {'Отклонение / σ':<15}")
print("-" * 95)

shifts = []
for name in independent_constants:
    val_eti = compute_raw(N_base, name)
    val_exp = experimental[name]

    if val_eti is not None and val_exp != 0:
        deviation = (val_eti - val_exp) / val_exp * 100
        shifts.append(deviation)

        # Оценка "сигмы" (предполагая погрешность измерения 0.01% для отношений)
        sigma_est = abs(val_exp) * 0.0001  # 0.01% погрешность
        sigma_dev = abs(deviation) / 0.01 if sigma_est > 0 else 0

        print(f"{name:<25} {val_eti:<20.6f} {val_exp:<20.6f} {deviation:<+15.6f} {sigma_dev:<15.2f}")

# Систематический сдвиг
if shifts:
    mean_shift = np.mean(shifts)
    std_shift = np.std(shifts)

    print(f"\n{'=' * 90}")
    print("СИСТЕМАТИЧЕСКИЙ СДВИГ НЕЗАВИСИМЫХ ОТ N КОНСТАНТ")
    print("=" * 90)
    print(f"\n  Средний сдвиг: {mean_shift:+.6f}%")
    print(f"  Стандартное отклонение: {std_shift:.6f}%")

    if abs(mean_shift) < 0.1:
        print("\n  ✅ СИСТЕМАТИЧЕСКИЙ СДВИГ ОТСУТСТВУЕТ")
        print("     Топологические инварианты ЕТИ совпадают с экспериментом")
        print("     без систематической ошибки.")
    elif abs(mean_shift) < 1.0:
        print(f"\n  🟡 СЛАБЫЙ СИСТЕМАТИЧЕСКИЙ СДВИГ ({mean_shift:+.4f}%)")
        print("     Возможна поправка порядка 1/ln N.")
    else:
        print(f"\n  ❌ ЗНАЧИТЕЛЬНЫЙ СИСТЕМАТИЧЕСКИЙ СДВИГ ({mean_shift:+.4f}%)")
        print("     Требуется пересмотр формул для топологических инвариантов.")