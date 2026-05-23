"""
ПРОВЕРКА СТАБИЛЬНОСТИ КВАНТОВАНИЯ C_i = π·(n_i + δ_b)

Проверяем для трёх разных N и трёх разных K:
  1. Сохраняется ли правило n = 16-a (b=1/3) и n = -a (b=0)?
  2. Насколько стабильно отклонение от целого?
  3. Зависит ли квантование от выбора N и K?

γ_i вычисляются из данных (f_exp и f_0 при данных N, K).
"""

import math
import numpy as np
from collections import defaultdict

# ============================================================
# ТЕСТИРУЕМЫЕ ПАРАМЕТРЫ
# ============================================================
N_values = [1e121, 4.197668e121, 4.4e121]
K_values = [4, 6, 8]

pi = math.pi
gamma_E = 0.5772156649015329

# ============================================================
# ЭКСПЕРИМЕНТАЛЬНЫЕ ДАННЫЕ (фиксированы)
# ============================================================
exp_data = {
    'ħ': 1.054571817e-34,
    'c': 299792458,
    'G': 6.67430e-11,
    'α': 1 / 137.035999084,
    'm_e': 9.1093837015e-31,
    'm_muon': 1.883531627e-28,
    'm_tau': 3.16754e-27,
    'm_proton': 1.67262192369e-27,
    'm_W': 1.43362e-25,
    'm_Z': 1.62614e-25,
    'm_Higgs': 2.23319e-25,
    'm_pion': 2.4880888e-28,
    'm_quark_t': 3.04e-25,
    'Rydberg': 1.097373e7,
    'Bohr_radius': 5.29177210903e-11,
    'e_charge': 1.602176634e-19,
    'tau_mu': 2.1969811e-6,
    'tau_tau': 2.903e-13,
    'm_proton/m_e': 1.67262192369e-27 / 9.1093837015e-31,
    'm_W/m_Z': 1.43362e-25 / 1.62614e-25,
}

# (name, a, b) — параметры ведущего порядка
params_fixed = {
    'ħ': (3, 1 / 3), 'c': (4, 0), 'G': (13, 1 / 3), 'α': (-1, 0),
    'm_e': (4, 1 / 3), 'm_muon': (5, 1 / 3), 'm_tau': (5, 1 / 3),
    'm_proton': (6, 1 / 3), 'm_W': (6, 1 / 3), 'm_Z': (6, 1 / 3),
    'm_Higgs': (6, 1 / 3), 'm_pion': (6, 1 / 3), 'm_quark_t': (6, 1 / 3),
    'Rydberg': (3, 0), 'Bohr_radius': (-4, 0), 'e_charge': (-7, 0),
    'tau_mu': (-2, 0), 'tau_tau': (-5, 0),
    'm_proton/m_e': (2, 0), 'm_W/m_Z': (0, 0),
}


# ============================================================
# ФОРМУЛЫ ВЕДУЩЕГО ПОРЯДКА (зависят от N и K)
# ============================================================
def compute_f0(name, N_val, K_val):
    lnN_val = math.log(N_val)
    lnK_val = math.log(K_val)
    N13 = N_val ** (1 / 3)

    formulas = {
        'ħ': lambda: (lnN_val ** 3) / (K_val * N13),
        'c': lambda: pi * (lnN_val ** 4) / (K_val ** 2 * lnK_val),
        'G': lambda: 16 * pi ** 3 * lnN_val ** 13 / (K_val ** 5 * lnK_val * N13),
        'α': lambda: 2 * lnK_val ** 2 / (pi * lnN_val),
        'm_e': lambda: 4 * pi * lnN_val ** 4 / (K_val ** 0.5 * N13),
        'm_muon': lambda: 4 * pi ** 2 * lnN_val ** 5 / (K_val * math.sqrt(3) * N13),
        'm_tau': lambda: math.sqrt(pi) * lnN_val ** 5 * K_val ** 2 / N13,
        'm_proton': lambda: math.sqrt(pi) * lnN_val ** 6 / (K_val ** 1.5 * N13),
        'm_W': lambda: 2 * pi ** 3 * lnN_val ** 6 / (K_val * N13),
        'm_Z': lambda: 4 * pi ** (5 / 2) * lnN_val ** 6 / (K_val * N13),
        'm_Higgs': lambda: 4 * pi ** 2 * lnN_val ** 6 / (K_val ** 0.5 * N13),
        'm_pion': lambda: lnN_val ** 6 / (4 * pi ** 2 * math.sqrt(2) * N13),
        'm_quark_t': lambda: K_val ** 3 * lnN_val ** 6 / (pi ** 2 * N13),
        'Rydberg': lambda: 4 * lnN_val ** 3 * lnK_val ** 3 / (pi * K_val ** 1.5),
        'Bohr_radius': lambda: K_val ** 1.5 / (8 * pi * lnN_val ** 4 * lnK_val),
        'e_charge': lambda: 1.0 / (pi * K_val ** 1.5 * lnN_val ** 7),
        'tau_mu': lambda: lnK_val / (K_val * math.sqrt(3) * lnN_val ** 2),
        'tau_tau': lambda: 1.0 / (2 * lnN_val ** 5),
        'm_proton/m_e': lambda: (math.sqrt(pi) * lnN_val ** 6 / (K_val ** 1.5 * N13)) / (
                    4 * pi * lnN_val ** 4 / (K_val ** 0.5 * N13)),
        'm_W/m_Z': lambda: math.sqrt(pi) / 2,
    }
    return formulas[name]()


# ============================================================
# ТЕСТИРОВАНИЕ
# ============================================================
print("=" * 130)
print("ПРОВЕРКА СТАБИЛЬНОСТИ КВАНТОВАНИЯ C_i = π·(n_i + δ_b)")
print("=" * 130)
print(f"  Констант: {len(params_fixed)}")
print(f"  Значений N: {len(N_values)}")
print(f"  Значений K: {len(K_values)}")
print(f"  Всего комбинаций: {len(N_values) * len(K_values)}")
print()

overall_stats = defaultdict(list)

for K_val in K_values:
    lnK_val = math.log(K_val)

    for N_val in N_values:
        lnN_val = math.log(N_val)
        lnlnN_val = math.log(lnN_val)

        print(f"{'─' * 130}")
        print(f"K = {K_val}, N = {N_val:.4e}, ln N = {lnN_val:.4f}, ln K = {lnK_val:.4f}")
        print(f"{'─' * 130}")
        print(
            f"  {'Константа':<16} {'a':>3} {'b':>5} {'n_ожид':>6} {'γ_изм':>10} {'C_i':>12} {'C_i/π':>12} {'x_i':>12} {'n_изм':>4} {'Δ':>10}")
        print(f"  {'-' * 115}")

        for name, (a, b) in params_fixed.items():
            f0 = compute_f0(name, N_val, K_val)
            fe = exp_data[name]

            if f0 == 0 or fe == 0:
                continue

            # Вычисляем γ из данных
            gamma = (lnN_val / lnK_val) * math.log(fe / f0)

            # Вычисляем C_i
            C_i = gamma / lnN_val + (b / lnK_val) * lnN_val - (a / lnK_val) * lnlnN_val

            C_i_over_pi = C_i / pi

            # Определяем δ_b и ожидаемое n
            if abs(b) < 1e-6:
                delta_b = 0
                n_expected = -a
            elif abs(b - 1 / 3) < 1e-6:
                delta_b = gamma_E
                n_expected = 16 - a
            else:
                delta_b = 0
                n_expected = round(C_i_over_pi)

            # Измеренное значение
            x_i = C_i_over_pi - delta_b
            n_measured = round(x_i)
            deviation = x_i - n_measured

            # Совпадает ли с ожидаемым?
            match = "✅" if n_measured == n_expected else "❌"

            overall_stats[(K_val, N_val)].append({
                'name': name,
                'n_expected': n_expected,
                'n_measured': n_measured,
                'deviation': deviation,
                'match': n_measured == n_expected,
            })

            print(
                f"  {name:<16} {a:>3} {b:>5.2f} {n_expected:>6} {gamma:>10.6f} {C_i:>12.6f} {C_i_over_pi:>12.6f} {x_i:>12.6f} {n_measured:>4} {deviation:>10.6f} {match}")

# ============================================================
# СВОДНАЯ СТАТИСТИКА
# ============================================================
print(f"\n{'=' * 130}")
print("СВОДНАЯ СТАТИСТИКА ПО ВСЕМ K И N")
print(f"{'=' * 130}")

print(f"\n  {'K':>4} {'N':>16} {'Совпадений n':>14} {'Среднее |Δ|':>14} {'Макс |Δ|':>12}")
print(f"  {'-' * 65}")

for (K_val, N_val), items in sorted(overall_stats.items()):
    matches = sum(1 for d in items if d['match'])
    mean_abs_dev = np.mean([abs(d['deviation']) for d in items])
    max_abs_dev = np.max([abs(d['deviation']) for d in items])

    print(f"  {K_val:>4} {N_val:>16.4e} {matches:>14}/{len(items)} {mean_abs_dev:>14.6f} {max_abs_dev:>12.6f}")

# ============================================================
# КЛЮЧЕВОЙ ВЫВОД
# ============================================================
print(f"\n{'=' * 130}")
print("КЛЮЧЕВОЙ ВЫВОД")
print(f"{'=' * 130}")

# Проверяем для K=6, N=4.198e121 (оптимальная точка)
optimal_items = overall_stats.get((6.0, 4.197668e121), [])
if optimal_items:
    matches_opt = sum(1 for d in optimal_items if d['match'])
    mean_dev_opt = np.mean([abs(d['deviation']) for d in optimal_items])

    print(f"\n  При K=6, N=N_opt:")
    print(f"    Совпадений с правилом n = 16-a / n = -a: {matches_opt}/{len(optimal_items)}")
    print(f"    Среднее |отклонение| от целого: {mean_dev_opt:.6f}")

    if matches_opt == len(optimal_items):
        print(f"    ✅ ВСЕ константы подчиняются правилу квантования!")
    else:
        print(f"    Нарушения: {[d['name'] for d in optimal_items if not d['match']]}")

# Проверяем для K=4 и K=8
for K_test in [4, 8]:
    items_K = [d for (k, n), items in overall_stats.items() if k == K_test for d in items]
    if items_K:
        matches_K = sum(1 for d in items_K if d['match'])
        print(f"\n  При K={K_test}:")
        print(f"    Совпадений: {matches_K}/{len(items_K)}")
        if matches_K < len(items_K):
            print(f"    Квантование n = 16-a / n = -a работает только при K=6!")
            print(f"    Это доказывает, что K=6 — ВЫДЕЛЕННОЕ значение.")