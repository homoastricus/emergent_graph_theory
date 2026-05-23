"""
МАСШТАБНОЕ ИССЛЕДОВАНИЕ СТРУКТУРЫ C_i НА 62 КОНСТАНТАХ

Гипотезы:
  H1: b=0 (безразмерные) → C_i/π ≈ целое число
  H2: b=1/3 (размерные с N^{-1/3}) → C_i/π ≈ целое + γ_E
  H3: a=0, b=0 (топологические) → C_i/π ≈ 0
  H4: Существуют другие классы с другими сдвигами

Используем γ_i из предыдущего полного анализа для 62 констант.
"""

import math
import numpy as np
from collections import defaultdict

# ============================================================
# ФУНДАМЕНТАЛЬНЫЕ КОНСТАНТЫ
# ============================================================
K = 6.0
pi = math.pi
lnK = math.log(K)
gamma_E = 0.5772156649015329  # постоянная Эйлера-Маскерони

N_opt = 4.197668e121
lnN_opt = math.log(N_opt)
lnlnN_opt = math.log(lnN_opt)

# ============================================================
# ВСЕ 62 КОНСТАНТЫ: (name, category, a, b)
# ============================================================
# a, b определены из структуры ведущего порядка f_0 ~ (ln N)^a * N^{-b}
all_constants = [
    # Квантовые (b=1/3)
    ('ħ',               'Квантовые',         3,  1/3),
    ('h',               'Квантовые',         3,  1/3),

    # Планковские
    ('l_P',             'Планковские',       2,  1/3),
    ('t_P',             'Планковские',      -2,  1/3),
    ('m_P',             'Планковские',      -3,  0),     # не зависит от N
    ('E_P',             'Планковские',       5,  0),     # не зависит от N
    ('T_P',             'Планковские',      -4, -1/3),

    # Фундаментальные
    ('c',               'Фундаментальные',   4,  0),
    ('G',               'Фундаментальные',  13,  1/3),
    ('k_B',             'Фундаментальные',   8,  1/3),
    ('α',               'Фундаментальные',  -1,  0),

    # Лептоны
    ('m_e',             'Лептоны',           4,  1/3),
    ('m_muon',          'Лептоны',           5,  1/3),
    ('m_tau',           'Лептоны',           5,  1/3),

    # Барионы
    ('m_proton',        'Барионы',           6,  1/3),
    ('m_neutron',       'Барионы',           6,  1/3),

    # Бозоны
    ('m_W',             'Бозоны',            6,  1/3),
    ('m_Z',             'Бозоны',            6,  1/3),
    ('m_Higgs',         'Бозоны',            6,  1/3),

    # Мезоны
    ('m_pion',          'Мезоны',            6,  1/3),
    ('m_pion0',         'Мезоны',            4,  1/3),
    ('m_kaon0',         'Мезоны',            6,  1/3),
    ('m_D0',            'Мезоны',            6,  1/3),
    ('m_J_psi',         'Мезоны',            5,  1/3),
    ('m_eta',           'Мезоны',            5,  1/3),
    ('m_Upsilon_1S',    'Мезоны',            6,  1/3),

    # Кварки
    ('m_quark_u',       'Кварки',            5,  1/3),
    ('m_quark_d',       'Кварки',            5,  1/3),
    ('m_quark_s',       'Кварки',            4,  1/3),
    ('m_quark_c',       'Кварки',            6,  1/3),
    ('m_quark_b',       'Кварки',            6,  1/3),
    ('m_quark_t',       'Кварки',            6,  1/3),

    # Атомные
    ('Rydberg',         'Атомные',           3,  0),
    ('Bohr_radius',     'Атомные',          -4,  0),
    ('Compton_e',       'Атомные',          -5,  0),
    ('Compton_p',       'Атомные',          -7,  0),

    # Электромагнитные
    ('e_charge',        'Электромагнитные', -7,  0),
    ('epsilon_0',       'Электромагнитные',-20, -1/3),
    ('mu_0',            'Электромагнитные', 12, -1/3),
    ('impedance',       'Электромагнитные', 16, -1/3),
    ('flux_quantum',    'Электромагнитные', 10, -1/3),

    # Космология
    ('Lambda',          'Космология',       12, -2/3),
    ('kappa_Einstein',  'Космология',       -3, -1/3),
    ('v_Higgs',         'Космология',        6,  1/3),

    # Времена жизни (все b=0, разные a)
    ('tau_mu',          'Времена жизни',    -2,  0),
    ('tau_tau',         'Времена жизни',    -5,  0),
    ('tau_pion',        'Времена жизни',    -4,  0),
    ('tau_neutron',     'Времена жизни',    -3,  1/12),  # особый случай
    ('tau_kaon',        'Времена жизни',    -3,  0),
    ('tau_D_plus',      'Времена жизни',    -4,  0),
    ('tau_B_plus',      'Времена жизни',    -5,  0),
    ('tau_Lambda_b',    'Времена жизни',    -5,  0),
    ('tau_D0',          'Времена жизни',    -4,  0),

    # Безразмерные отношения (b=0)
    ('m_proton/m_e',    'Отношения',         2,  0),
    ('m_muon/m_e',      'Отношения',         1,  0),
    ('m_tau/m_e',       'Отношения',         1,  0),
    ('m_W/m_e',         'Отношения',         2,  0),
    ('m_Z/m_e',         'Отношения',         2,  0),
    ('m_Higgs/m_e',     'Отношения',         2,  0),
    ('m_W/m_Z',         'Отношения',         0,  0),
    ('m_Higgs/m_W',     'Отношения',         0,  0),
    ('m_P/m_e',         'Отношения',         0,  0),  # a=0? проверим
]

# ============================================================
# РАНЕЕ ИЗМЕРЕННЫЕ γ_i ДЛЯ ВСЕХ КОНСТАНТ (ПРИ ОПТИМАЛЬНОМ N)
# ============================================================
# Из полной таблицы (раздел 7 отчёта)
gamma_measured = {
    'ħ': 0.192835, 'h': 0.192835,
    'l_P': -0.104361, 't_P': -0.222142, 'm_P': 0.179369, 'E_P': 0.415022, 'T_P': -0.364637,
    'c': 0.117799, 'G': -0.048154, 'k_B': 0.224670, 'α': -0.015396,
    'm_e': 0.515264, 'm_muon': 0.061400, 'm_tau': 0.248402,
    'm_proton': -0.120978, 'm_neutron': -0.462800,
    'm_W': -0.100838, 'm_Z': 0.715684, 'm_Higgs': -0.267871,
    'm_pion': 0.127663, 'm_pion0': 0.271797, 'm_kaon0': -0.193698,
    'm_D0': -1.079562, 'm_J_psi': -0.385841, 'm_eta': -0.248487, 'm_Upsilon_1S': -1.179646,
    'm_quark_u': -0.682866, 'm_quark_d': 0.730184, 'm_quark_s': -1.474050,
    'm_quark_c': 0.195006, 'm_quark_b': -0.592949, 'm_quark_t': 0.120094,
    'Rydberg': 0.409414, 'Bohr_radius': -0.424833, 'Compton_e': -0.460215, 'Compton_p': 0.196014,
    'e_charge': -0.105394, 'epsilon_0': -0.506027, 'mu_0': 0.270428,
    'impedance': 0.388227, 'flux_quantum': 0.298216,
    'Lambda': 0.340703, 'kappa_Einstein': -0.519350, 'v_Higgs': 0.585737,
    'tau_mu': -0.100283, 'tau_tau': 0.012300, 'tau_pion': 0.173669,
    'tau_neutron': -0.243215, 'tau_kaon': -0.150499,
    'tau_D_plus': -0.031627, 'tau_B_plus': 0.385605, 'tau_Lambda_b': -0.008957, 'tau_D0': 0.060468,
    'm_proton/m_e': -0.636243, 'm_muon/m_e': -0.453865, 'm_tau/m_e': -0.266862,
    'm_W/m_e': -0.616102, 'm_Z/m_e': 0.200420, 'm_Higgs/m_e': -0.783135,
    'm_W/m_Z': -0.816522, 'm_Higgs/m_W': -0.167033, 'm_P/m_e': -0.335895,
}

# ============================================================
# ВЫЧИСЛЕНИЕ C_i ДЛЯ ВСЕХ КОНСТАНТ
# ============================================================
def compute_Ci(gamma, a, b, lnN, lnK, lnlnN):
    """C_i = γ/ln N + (b/ln K)·ln N - (a/ln K)·ln(ln N)"""
    return gamma/lnN + (b/lnK)*lnN - (a/lnK)*lnlnN

results = []
for name, category, a, b in all_constants:
    if name not in gamma_measured:
        continue

    gamma = gamma_measured[name]
    ci = compute_Ci(gamma, a, b, lnN_opt, lnK, lnlnN_opt)

    results.append({
        'name': name,
        'category': category,
        'a': a,
        'b': b,
        'gamma': gamma,
        'C_i': ci,
        'C_i_over_pi': ci / pi,
    })

# ============================================================
# АНАЛИЗ ПО КЛАССАМ b
# ============================================================
print("=" * 110)
print("МАСШТАБНЫЙ АНАЛИЗ СТРУКТУРЫ C_i НА 62 КОНСТАНТАХ")
print("=" * 110)
print(f"\n  Гипотезы:")
print(f"    H1: b=0 → C_i/π ≈ целое число n")
print(f"    H2: b=1/3 → C_i/π ≈ n + γ_E  (γ_E = {gamma_E:.6f})")
print(f"    H3: a=0, b=0 → C_i/π ≈ 0")
print(f"    H4: Другие b → другие закономерности")
print(f"\n  Параметры: ln N = {lnN_opt:.6f}, ln K = {lnK:.6f}, π = {pi:.6f}")

# Группировка по b
by_b = defaultdict(list)
for r in results:
    b_key = round(r['b'], 6)
    by_b[b_key].append(r)

for b_val in sorted(by_b.keys()):
    items = by_b[b_val]
    n_items = len(items)

    print(f"\n{'─'*110}")
    print(f"КЛАСС b = {b_val}  (n = {n_items})")
    print(f"{'─'*110}")

    if b_val == 0:
        print(f"  Гипотеза: C_i/π ≈ целое число")
        print(f"\n  {'Константа':<20} {'a':>4} {'C_i':>12} {'C_i/π':>12} {'n_best':>8} {'Остаток':>12} {'Качество':>15}")
        print(f"  {'─'*90}")

        residuals = []
        for r in sorted(items, key=lambda x: x['C_i_over_pi']):
            ci_over_pi = r['C_i_over_pi']
            n_best = round(ci_over_pi)
            residual = ci_over_pi - n_best
            residuals.append(residual)

            if abs(residual) < 0.01:
                quality = "✅✅✅ ОТЛИЧНО"
            elif abs(residual) < 0.05:
                quality = "✅✅ ХОРОШО"
            elif abs(residual) < 0.1:
                quality = "✅ ПРИЕМЛЕМО"
            else:
                quality = "❌ НЕТ"

            print(f"  {r['name']:<20} {r['a']:>4} {r['C_i']:>12.4f} {ci_over_pi:>12.6f} {n_best:>8} {residual:>12.6f} {quality:>15}")

        if residuals:
            residuals_arr = np.array(residuals)
            print(f"\n  Статистика остатков: среднее = {np.mean(residuals_arr):.6f}, стд = {np.std(residuals_arr):.6f}")
            print(f"  Среднее |остаток| = {np.mean(np.abs(residuals_arr)):.6f}")

    elif abs(b_val - 1/3) < 0.001:
        print(f"  Гипотеза: C_i/π ≈ целое + γ_E")
        print(f"\n  {'Константа':<20} {'a':>4} {'C_i':>12} {'C_i/π':>12} {'C_i/π - γ_E':>14} {'n_best':>6} {'Остаток':>12} {'Качество':>15}")
        print(f"  {'─'*100}")

        residuals = []
        for r in sorted(items, key=lambda x: x['C_i_over_pi']):
            ci_over_pi = r['C_i_over_pi']
            shifted = ci_over_pi - gamma_E
            n_best = round(shifted)
            residual = shifted - n_best
            residuals.append(residual)

            if abs(residual) < 0.02:
                quality = "✅✅✅ ОТЛИЧНО"
            elif abs(residual) < 0.05:
                quality = "✅✅ ХОРОШО"
            elif abs(residual) < 0.1:
                quality = "✅ ПРИЕМЛЕМО"
            else:
                quality = "❌ НЕТ"

            print(f"  {r['name']:<20} {r['a']:>4} {r['C_i']:>12.4f} {ci_over_pi:>12.6f} {shifted:>14.6f} {n_best:>6} {residual:>12.6f} {quality:>15}")

        if residuals:
            residuals_arr = np.array(residuals)
            print(f"\n  Статистика остатков: среднее = {np.mean(residuals_arr):.6f}, стд = {np.std(residuals_arr):.6f}")
            print(f"  Среднее |остаток| = {np.mean(np.abs(residuals_arr)):.6f}")

    else:
        print(f"  Другой класс b = {b_val} — требуется отдельный анализ")
        print(f"\n  {'Константа':<20} {'a':>4} {'C_i':>12} {'C_i/π':>12}")
        print(f"  {'─'*55}")
        for r in sorted(items, key=lambda x: x['C_i_over_pi']):
            print(f"  {r['name']:<20} {r['a']:>4} {r['C_i']:>12.4f} {r['C_i_over_pi']:>12.6f}")

# ============================================================
# СВОДНАЯ ТАБЛИЦА ПО КАТЕГОРИЯМ
# ============================================================
print(f"\n{'='*110}")
print("СВОДКА ПО ФИЗИЧЕСКИМ КАТЕГОРИЯМ")
print(f"{'='*110}")

by_category = defaultdict(list)
for r in results:
    by_category[r['category']].append(r)

print(f"\n  {'Категория':<22} {'n':>4} {'Среднее C_i':>14} {'Класс b':>10} {'Закономерность':>30}")
print(f"  {'─'*85}")

for cat, items in sorted(by_category.items()):
    avg_ci = np.mean([r['C_i'] for r in items])
    b_vals = set(round(r['b'], 6) for r in items)
    b_str = ', '.join(str(b) for b in b_vals)

    # Определяем закономерность
    if all(round(r['b'], 6) == 0 for r in items):
        pattern = "C_i/π ≈ целое"
    elif all(abs(round(r['b'], 6) - 1/3) < 0.001 for r in items):
        pattern = "C_i/π ≈ целое + γ_E"
    else:
        pattern = "смешанный"

    print(f"  {cat:<22} {len(items):>4} {avg_ci:>14.4f} {b_str:>10} {pattern:>30}")

# ============================================================
# ФИНАЛЬНЫЙ ВЫВОД
# ============================================================
print(f"\n{'='*110}")
print("ФИНАЛЬНЫЙ ВЫВОД")
print(f"{'='*110}")

# Собираем статистику по качеству совпадений
excellent_0 = 0
good_0 = 0
total_0 = 0
excellent_13 = 0
good_13 = 0
total_13 = 0

for b_val, items in by_b.items():
    for r in items:
        if b_val == 0:
            total_0 += 1
            n_best = round(r['C_i_over_pi'])
            if abs(r['C_i_over_pi'] - n_best) < 0.01:
                excellent_0 += 1
            elif abs(r['C_i_over_pi'] - n_best) < 0.05:
                good_0 += 1
        elif abs(b_val - 1/3) < 0.001:
            total_13 += 1
            shifted = r['C_i_over_pi'] - gamma_E
            n_best = round(shifted)
            if abs(shifted - n_best) < 0.02:
                excellent_13 += 1
            elif abs(shifted - n_best) < 0.05:
                good_13 += 1

print(f"""
  РЕЗУЛЬТАТЫ ПРОВЕРКИ ГИПОТЕЗ:

  Класс b=0 (безразмерные, n={total_0}):
    • C_i/π ≈ целое с ошибкой < 0.01: {excellent_0}/{total_0} ({excellent_0/total_0*100:.0f}%)
    • C_i/π ≈ целое с ошибкой < 0.05: {good_0}/{total_0} ({good_0/total_0*100:.0f}%)

  Класс b=1/3 (размерные, n={total_13}):
    • C_i/π ≈ целое + γ_E с ошибкой < 0.02: {excellent_13}/{total_13} ({excellent_13/total_13*100:.0f}%)
    • C_i/π ≈ целое + γ_E с ошибкой < 0.05: {good_13}/{total_13} ({good_13/total_13*100:.0f}%)

  ИНТЕРПРЕТАЦИЯ:
  
  Если гипотезы подтверждаются на >80% констант:
    → C_i имеет ДОКАЗАННУЮ дискретную структуру
    → Физика сводится к трём числам: n (целое), π, γ_E
    → Это уровень открытия, сравнимый с квантованием заряда
  
  Если подтверждается на 50-80%:
    → Структура есть, но требует уточнения для отдельных классов
    → Возможно, существуют другие сдвиги помимо γ_E
  
  Если <50%:
    → Гипотеза отвергается в текущей форме
    → C_i имеют более сложное происхождение
""")