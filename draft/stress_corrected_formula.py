"""
ЭМЕРДЖЕНТНЫЕ ФОРМУЛЫ ЕТИ С ПОПРАВКАМИ ПЕРВОГО ПОРЯДКА
Сравнение: без поправки → с поправкой → экспериментальное значение

Поправка: f_true = f_model * exp(b₀/ln N)
         f_true ≈ f_model * (1 + b₀/ln N)
         b₀ = γ_i * ln K
"""

import math
import numpy as np

# ============================================================
# ФУНДАМЕНТАЛЬНЫЕ ПАРАМЕТРЫ
# ============================================================
K = 6.0
pi = math.pi
lnK = math.log(K)  # ≈ 1.7918
N0 = 4.197668e121
lnN = math.log(N0)  # ≈ 280.0473
N13 = N0 ** (1 / 3)

# Масштаб поправки
inv_lnN = 1.0 / lnN

# ============================================================
# ТАБЛИЦА γ_i ДЛЯ КОНСТАНТ (измеренные значения)
# ============================================================
# Для безразмерных отношений γ = b₀ / lnK
gamma_values = {
    # Безразмерные массы фермионов (рациональные приближения)
    'm_proton/m_e': -7 / 11,  # γ = -0.636364
    'm_muon/m_e': -5 / 11,  # γ = -0.454545
    'm_tau/m_e': -4 / 15,  # γ = -0.266667

    # Безразмерные массы бозонов
    'm_W/m_e': -0.6161,  # измерено
    'm_Z/m_e': +0.2004,  # измерено (положительное!)
    'm_Higgs/m_e': -0.7831,  # измерено

    # Калибровочные отношения
    'm_W/m_Z': -0.8165,  # измерено
    'm_Higgs/m_W': -0.1670,  # измерено

    # Планковские и фундаментальные (измеренные γ)
    'ħ': +0.1928,
    'l_P': -0.1044,
    't_P': -0.2221,
    'm_P': +0.1794,
    'E_P': +0.4150,
    'T_P': -0.3646,
    'c': +0.1178,
    'G': -0.0482,
    'k_B': +0.2247,

    # Массы (размерные)
    'm_e': +0.5153,
    'm_proton': -0.1210,
    'm_muon': +0.0614,
    'm_tau': +0.2484,
    'm_W': -0.1008,
    'm_Z': +0.7157,
    'm_Higgs': -0.2679,

    # Дополнительные
    'Rydberg': +0.4094,
    'Bohr_radius': -0.4248,
    'α': -0.0154,
    'm_P/m_e': -0.3359,
}

# ============================================================
# ЭКСПЕРИМЕНТАЛЬНЫЕ ДАННЫЕ (CODATA)
# ============================================================
exp_data = {
    'ħ': 1.054571817e-34,
    'l_P': 1.616255e-35,
    't_P': 5.391247e-44,
    'm_P': 2.176434e-8,
    'E_P': 1.956082e9,
    'T_P': 1.416784e32,
    'c': 299792458,
    'G': 6.67430e-11,
    'k_B': 1.380649e-23,
    'm_e': 9.1093837015e-31,
    'm_proton': 1.67262192369e-27,
    'm_muon': 1.883531627e-28,
    'm_tau': 3.16754e-27,
    'm_W': 1.43362e-25,
    'm_Z': 1.62614e-25,
    'm_Higgs': 2.23319e-25,
    'Rydberg': 1.097373e7,
    'Bohr_radius': 5.29177210903e-11,
    'α': 1 / 137.035999084,
}


# ============================================================
# ФОРМУЛЫ ЕТИ (ВЕДУЩИЙ ПОРЯДОК)
# ============================================================
def eti_formulas():
    """Возвращает словарь формул и значений ведущего порядка"""

    formulas = {}
    values = {}

    # Планковские
    formulas['ħ'] = '(ln N)³ / (K · N^(1/3))'
    values['ħ'] = (lnN ** 3) / (K * N13)

    formulas['l_P'] = '4 · (ln N)² · ln K / N^(1/3)'
    values['l_P'] = 4 * lnN ** 2 * lnK / N13

    formulas['t_P'] = '4 · K² · (ln K)² / (π · N^(1/3) · (ln N)²)'
    values['t_P'] = 4 * K ** 2 * lnK ** 2 / (pi * N13 * lnN ** 2)

    formulas['m_P'] = 'K / (4π · (ln N)³)'
    values['m_P'] = K / (pi * 4 * lnN ** 3)

    formulas['E_P'] = 'π · (ln N)⁵ / (4 · K³ · (ln K)²)'
    values['E_P'] = (lnN ** 5) * pi / (4 * K ** 3 * lnK ** 2)

    formulas['T_P'] = '8π · N^(1/3) / (ln N)⁴'
    values['T_P'] = 8 * pi * N13 / (lnN ** 4)

    # Фундаментальные
    formulas['c'] = 'π · (ln N)⁴ / (K² · ln K)'
    values['c'] = pi * (lnN ** 4) / (K ** 2 * lnK)

    formulas['G'] = '16π³ · (ln N)^13 / (K⁵ · ln K · N^(1/3))'
    values['G'] = 16 * pi ** 3 * lnN ** 13 / (K ** 5 * lnK * N13)

    formulas['k_B'] = '(Kp) · (ln N)⁸ / (8π²),  p = 1/(K·N^(1/3))'
    p_val = 1.0 / (K * N13)
    values['k_B'] = (K * p_val) * (lnN ** 8) / (8 * pi ** 2)

    # Безразмерные
    formulas['α'] = '2 · (ln K)² / (π · ln N)'
    values['α'] = 2 * lnK ** 2 / (pi * lnN)

    # Массы
    formulas['m_e'] = '4π · (ln N)⁴ / (√K · N^(1/3))'
    values['m_e'] = 4 * pi * lnN ** 4 / (K ** 0.5 * N13)

    formulas['m_proton'] = '√π · (ln N)⁶ / (K^(3/2) · N^(1/3))'
    values['m_proton'] = math.sqrt(pi) * lnN ** 6 / (K ** 1.5 * N13)

    formulas['m_muon'] = '4π² · (ln N)⁵ / (K · √3 · N^(1/3))'
    values['m_muon'] = 4 * pi ** 2 * lnN ** 5 / (K * math.sqrt(3) * N13)

    formulas['m_tau'] = '√π · (ln N)⁵ · K² / N^(1/3)'
    values['m_tau'] = math.sqrt(pi) * lnN ** 5 * K ** 2 / N13

    formulas['m_W'] = '2π³ · (ln N)⁶ / (K · N^(1/3))'
    values['m_W'] = 2 * pi ** 3 * lnN ** 6 / (K * N13)

    formulas['m_Z'] = '4π^(5/2) · (ln N)⁶ / (K · N^(1/3))'
    values['m_Z'] = 4 * pi ** (5 / 2) * lnN ** 6 / (K * N13)

    formulas['m_Higgs'] = '4π² · (ln N)⁶ / (√K · N^(1/3))'
    values['m_Higgs'] = 4 * pi ** 2 * lnN ** 6 / (K ** 0.5 * N13)

    # Дополнительные
    formulas['Rydberg'] = '4 · (ln N)³ · (ln K)³ / (π · K^(3/2))'
    values['Rydberg'] = 4 * lnN ** 3 * lnK ** 3 / (pi * K ** 1.5)

    formulas['Bohr_radius'] = 'K^(3/2) / (8π · (ln N)⁴ · ln K)'
    values['Bohr_radius'] = K ** 1.5 / (8 * pi * lnN ** 4 * lnK)

    # Безразмерные отношения
    formulas['m_proton/m_e'] = 'm_proton / m_e'
    values['m_proton/m_e'] = values['m_proton'] / values['m_e']

    formulas['m_muon/m_e'] = 'm_muon / m_e'
    values['m_muon/m_e'] = values['m_muon'] / values['m_e']

    formulas['m_tau/m_e'] = 'm_tau / m_e'
    values['m_tau/m_e'] = values['m_tau'] / values['m_e']

    formulas['m_W/m_e'] = 'm_W / m_e'
    values['m_W/m_e'] = values['m_W'] / values['m_e']

    formulas['m_Z/m_e'] = 'm_Z / m_e'
    values['m_Z/m_e'] = values['m_Z'] / values['m_e']

    formulas['m_Higgs/m_e'] = 'm_Higgs / m_e'
    values['m_Higgs/m_e'] = values['m_Higgs'] / values['m_e']

    formulas['m_W/m_Z'] = '√π / 2'
    values['m_W/m_Z'] = math.sqrt(pi) / 2

    formulas['m_Higgs/m_W'] = '2√K / π'
    values['m_Higgs/m_W'] = 2 * math.sqrt(K) / pi

    formulas['m_P/m_e'] = 'm_P / m_e'
    values['m_P/m_e'] = values['m_P'] / values['m_e']

    return formulas, values


formulas, model_values = eti_formulas()


# ============================================================
# ВЫЧИСЛЕНИЕ ЗНАЧЕНИЙ С ПОПРАВКАМИ
# ============================================================
def apply_correction(value, gamma):
    """Применяет поправку первого порядка"""
    b0 = gamma * lnK
    return value * math.exp(b0 * inv_lnN)


# ============================================================
# ФОРМИРОВАНИЕ ТАБЛИЦЫ
# ============================================================
# Определяем порядок вывода
display_order = [
    # Безразмерные
    'α',
    'm_proton/m_e', 'm_muon/m_e', 'm_tau/m_e',
    'm_W/m_e', 'm_Z/m_e', 'm_Higgs/m_e',
    'm_W/m_Z', 'm_Higgs/m_W',
    'm_P/m_e',
    # Размерные планковские
    'ħ', 'l_P', 't_P', 'm_P', 'E_P', 'T_P',
    # Фундаментальные
    'c', 'G', 'k_B',
    # Массы
    'm_e', 'm_proton', 'm_muon', 'm_tau',
    'm_W', 'm_Z', 'm_Higgs',
    # Дополнительные
    'Rydberg', 'Bohr_radius',
]

print("=" * 120)
print("ЭМЕРДЖЕНТНЫЕ ФОРМУЛЫ ЕТИ С ПОПРАВКАМИ ПЕРВОГО ПОРЯДКА")
print("=" * 120)
print(f"\n  Параметры: N = {N0:.6e}, ln N = {lnN:.6f}, K = {K}, ln K = {lnK:.6f}")
print(f"  Поправка: f_corrected = f_model × exp(γ·ln K / ln N)")
print(f"  Масштаб поправки: ln K / ln N = {lnK / lnN:.6f} ({lnK / lnN * 100:.4f}%)")
print()

# Заголовок таблицы
print(
    f"  {'Константа':<18} {'Формула (ведущий порядок)':<48} {'Формула с поправкой':<42} {'Значение до':>16} {'Значение после':>16} {'CODATA':>16} {'Ошибка до %':>12} {'Ошибка после %':>12}")
print(f"  {'-' * 180}")

for name in display_order:
    if name not in model_values:
        continue

    fm = model_values[name]
    formula_str = formulas.get(name, '—')

    # Поправка
    if name in gamma_values:
        gamma = gamma_values[name]
        b0 = gamma * lnK
        f_corrected = fm * math.exp(b0 * inv_lnN)
        correction_str = f"× exp({gamma:.4f}·ln K / ln N)"
    else:
        f_corrected = fm
        correction_str = "—"

    # Экспериментальное значение
    if name in exp_data:
        fe = exp_data[name]
    elif name.startswith('m_') and '/' in name:
        # Безразмерное отношение — вычисляем из эксперимента
        parts = name.split('/')
        if parts[0] in exp_data and parts[1] in exp_data:
            fe = exp_data[parts[0]] / exp_data[parts[1]]
        else:
            fe = None
    else:
        fe = None

    # Ошибки
    if fe is not None and fe != 0 and fm != 0:
        err_before = abs(fm / fe - 1) * 100
        err_after = abs(f_corrected / fe - 1) * 100 if f_corrected != 0 else float('inf')
    else:
        err_before = None
        err_after = None

    # Форматирование
    # Формулы
    formula_display = formula_str if len(formula_str) <= 46 else formula_str[:43] + '...'
    corr_display = correction_str if len(correction_str) <= 40 else correction_str[:37] + '...'


    # Значения — выбираем формат в зависимости от порядка
    def fmt_val(v):
        if v == 0:
            return f"{'0.0':>16}"
        if abs(v) >= 1e6 or (abs(v) <= 1e-5 and v != 0):
            return f"{v:>16.8e}"
        elif abs(v) >= 1000:
            return f"{v:>16.4f}"
        elif abs(v) >= 1:
            return f"{v:>16.8f}"
        else:
            return f"{v:>16.8f}"


    # Ошибки
    def fmt_err(e):
        if e is None:
            return f"{'—':>12}"
        return f"{e:>12.6f}"


    print(
        f"  {name:<18} {formula_display:<48} {corr_display:<42} {fmt_val(fm)} {fmt_val(f_corrected)} {fmt_val(fe) if fe else '—':>16} {fmt_err(err_before)} {fmt_err(err_after)}")

# ============================================================
# СВОДНАЯ СТАТИСТИКА
# ============================================================
print("\n" + "=" * 120)
print("СВОДНАЯ СТАТИСТИКА УЛУЧШЕНИЙ")
print("=" * 120)

improvements = []
for name in display_order:
    if name not in model_values or name not in gamma_values:
        continue
    if name not in exp_data and not (name.startswith('m_') and '/' in name):
        continue

    fm = model_values[name]
    gamma = gamma_values[name]
    b0 = gamma * lnK
    f_corrected = fm * math.exp(b0 * inv_lnN)

    if name in exp_data:
        fe = exp_data[name]
    else:
        parts = name.split('/')
        fe = exp_data[parts[0]] / exp_data[parts[1]]

    err_before = abs(fm / fe - 1) * 100
    err_after = abs(f_corrected / fe - 1) * 100

    if err_before > 0:
        improvement = (err_before - err_after) / err_before * 100
        improvements.append({
            'name': name,
            'err_before': err_before,
            'err_after': err_after,
            'improvement': improvement,
        })

if improvements:
    avg_improvement = np.mean([x['improvement'] for x in improvements])
    avg_err_before = np.mean([x['err_before'] for x in improvements])
    avg_err_after = np.mean([x['err_after'] for x in improvements])

    print(f"\n  Всего констант с поправками: {len(improvements)}")
    print(f"  Средняя ошибка до поправки:    {avg_err_before:.6f}%")
    print(f"  Средняя ошибка после поправки: {avg_err_after:.6f}%")
    print(f"  Среднее улучшение:             {avg_improvement:.1f}%")

    # Топ-5 улучшений
    improvements.sort(key=lambda x: x['improvement'], reverse=True)
    print(f"\n  ТОП-5 УЛУЧШЕНИЙ:")
    for i, imp in enumerate(improvements[:5]):
        print(
            f"    {i + 1}. {imp['name']:<18} {imp['err_before']:.6f}% → {imp['err_after']:.6f}%  (улучшение {imp['improvement']:.1f}%)")