"""
ЭМЕРДЖЕНТНЫЕ ФОРМУЛЫ ЕТИ С ПОПРАВКАМИ ПЕРВОГО ПОРЯДКА
ПОЛНАЯ ВЕРСИЯ — ВСЕ КОНСТАНТЫ И ФОРМУЛЫ
"""

import math
import numpy as np

# ============================================================
# ФУНДАМЕНТАЛЬНЫЕ ПАРАМЕТРЫ
# ============================================================
K = 6.0
pi = math.pi
lnK = math.log(K)  # ≈ 1.7918
N0 = 4.15e121#4.197668e121
lnN = math.log(N0)  # ≈ 280.0473
N13 = N0 ** (1 / 3)
N16 = N0 ** (1 / 6)

inv_lnN = 1.0 / lnN
p_val = 1.0 / (K * N13)

# ============================================================
# ЭКСПЕРИМЕНТАЛЬНЫЕ ДАННЫЕ (CODATA + PDG)
# ============================================================
exp_data = {
    # Квантовые
    'ħ': 1.054571817e-34,
    'h': 6.62607015e-34,

    # Планковские
    't_P': 5.391247e-44,
    'l_P': 1.616255e-35,
    'm_P': 2.176434e-8,
    'E_P': 1.956082e9,
    'T_P': 1.416784e32,

    # Фундаментальные
    'c': 299792458,
    'G': 6.67430e-11,
    'k_B': 1.380649e-23,

    # Безразмерные
    'α': 1 / 137.035999084,

    # Массы лептонов
    'm_e': 9.1093837015e-31,
    'm_muon': 1.883531627e-28,
    'm_tau': 3.16754e-27,

    # Массы барионов
    'm_proton': 1.67262192369e-27,
    'm_neutron': 1.67492749804e-27,

    # Массы калибровочных бозонов
    'm_W': 1.43362e-25,
    'm_Z': 1.62614e-25,
    'm_Higgs': 2.23319e-25,

    # Массы мезонов
    'm_pion': 2.4880888e-28,
    'm_pion0': 2.40609e-28,
    'm_kaon0': 8.801929e-28,
    'm_D0': 3.32479e-27,
    'm_J_psi': 5.52061e-27,
    'm_eta': 9.767732e-28,
    'm_Upsilon_1S': 1.68715e-26,

    # Массы кварков
    'm_quark_u': 2.1650e-30,
    'm_quark_d': 4.7915e-30,
    'm_quark_s': 9.635e-30,
    'm_quark_c': 1.27e-27,
    'm_quark_b': 4.180e-27,
    'm_quark_t': 3.04e-25,

    # Атомные и спектральные
    'Rydberg': 1.097373e7,
    'Bohr_radius': 5.29177210903e-11,
    'Compton_e': 2.426e-12,
    'Compton_p': 1.32140985396e-15,

    # Электромагнитные
    'e_charge': 1.602176634e-19,
    'epsilon_0': 8.8541878128e-12,
    'mu_0': 1.25663706127e-6,
    'impedance': 376.730313,
    'flux_quantum': 2.06783366752e-15,

    # Космология
    'Lambda': 1.08929e-52,
    'kappa_Einstein': 2.07664746e-43,
    'v_Higgs': 4.388471e-25,

    # Времена жизни
    'tau_mu': 2.1969811e-6,
    'tau_tau': 2.903e-13,
    'tau_pion': 2.6033e-8,
    'tau_neutron': 877.8,
    'tau_kaon': 1.2380e-8,
    'tau_D_plus': 1.040e-12,
    'tau_B_plus': 1.638e-12,
    'tau_Lambda_b': 1.471e-12,
    'tau_D0': 4.101e-13,
}


# ============================================================
# ФОРМУЛЫ ЕТИ (ПОЛНЫЙ НАБОР)
# ============================================================
def eti_formulas():
    formulas = {}
    values = {}

    # ========================
    # КВАНТОВЫЕ
    # ========================
    formulas['ħ'] = '(ln N)³ / (K · N^(1/3))'
    values['ħ'] = lnN ** 3 / (K * N13)

    formulas['h'] = '2π · ħ'
    values['h'] = 2 * pi * values['ħ']

    # ========================
    # ПЛАНКОВСКИЕ
    # ========================
    formulas['l_P'] = '4 · (ln N)² · ln K / N^(1/3)'
    values['l_P'] = 4 * lnN ** 2 * lnK / N13

    formulas['t_P'] = '4 · K² · (ln K)² / (π · N^(1/3) · (ln N)²)'
    values['t_P'] = 4 * K ** 2 * lnK ** 2 / (pi * N13 * lnN ** 2)

    formulas['m_P'] = 'K / (4π · (ln N)³)'
    values['m_P'] = K / (pi * 4 * lnN ** 3)

    formulas['E_P'] = 'π · (ln N)⁵ / (4 · K³ · (ln K)²)'
    values['E_P'] = lnN ** 5 * pi / (4 * K ** 3 * lnK ** 2)

    formulas['T_P'] = '8π · N^(1/3) / (ln N)⁴'
    values['T_P'] = 8 * pi * N13 / lnN ** 4

    # ========================
    # ФУНДАМЕНТАЛЬНЫЕ
    # ========================
    formulas['c'] = 'π · (ln N)⁴ / (K² · ln K)'
    values['c'] = pi * lnN ** 4 / (K ** 2 * lnK)

    formulas['G'] = '16π³ · (ln N)^13 / (K⁵ · ln K · N^(1/3))'
    values['G'] = 16 * pi ** 3 * lnN ** 13 / (K ** 5 * lnK * N13)

    formulas['k_B'] = '(Kp) · (ln N)⁸ / (8π²)'
    values['k_B'] = (K * p_val) * lnN ** 8 / (8 * pi ** 2)

    formulas['α'] = '2 · (ln K)² / (π · ln N)'
    values['α'] = 2 * lnK ** 2 / (pi * lnN)

    # ========================
    # МАССЫ ЛЕПТОНОВ
    # ========================
    formulas['m_e'] = '4π · (ln N)⁴ / (√K · N^(1/3))'
    values['m_e'] = 4 * pi * lnN ** 4 / (K ** 0.5 * N13)

    formulas['m_muon'] = '4π² · (ln N)⁵ / (K · √3 · N^(1/3))'
    values['m_muon'] = 4 * pi ** 2 * lnN ** 5 / (K * math.sqrt(3) * N13)

    formulas['m_tau'] = '√π · (ln N)⁵ · K² / N^(1/3)'
    values['m_tau'] = math.sqrt(pi) * lnN ** 5 * K ** 2 / N13

    # ========================
    # МАССЫ БАРИОНОВ
    # ========================
    formulas['m_proton'] = '√π · (ln N)⁶ / (K^(3/2) · N^(1/3))'
    values['m_proton'] = math.sqrt(pi) * lnN ** 6 / (K ** 1.5 * N13)

    formulas['m_neutron'] = 'm_proton · (1 + 1/lnN)'
    values['m_neutron'] = values['m_proton'] * (1 + inv_lnN)

    # ========================
    # МАССЫ БОЗОНОВ
    # ========================
    formulas['m_W'] = '2π³ · (ln N)⁶ / (K · N^(1/3))'
    values['m_W'] = 2 * pi ** 3 * lnN ** 6 / (K * N13)

    formulas['m_Z'] = '4π^(5/2) · (ln N)⁶ / (K · N^(1/3))'
    values['m_Z'] = 4 * pi ** (5 / 2) * lnN ** 6 / (K * N13)

    formulas['m_Higgs'] = '4π² · (ln N)⁶ / (√K · N^(1/3))'
    values['m_Higgs'] = 4 * pi ** 2 * lnN ** 6 / (K ** 0.5 * N13)

    # ========================
    # МАССЫ МЕЗОНОВ
    # ========================
    formulas['m_pion'] = '(ln N)⁶ / (4π² · √2 · N^(1/3))'
    values['m_pion'] = lnN ** 6 / (4 * pi ** 2 * math.sqrt(2) * N13)

    formulas['m_pion0'] = '2π · K³ · (ln N)⁴ / N^(1/3)'
    values['m_pion0'] = 2 * pi * K ** 3 * lnN ** 4 / N13

    formulas['m_kaon0'] = '(ln N)⁶ · √(2π) / (4π² · N^(1/3))'
    values['m_kaon0'] = lnN ** 6 * math.sqrt(2 * pi) / (4 * pi ** 2 * N13)

    formulas['m_D0'] = '(ln N)⁶ · √(2π) / (K · √3 · N^(1/3))'
    values['m_D0'] = lnN ** 6 * math.sqrt(2 * pi) / (K * math.sqrt(3) * N13)

    formulas['m_J_psi'] = '8π² · √2 · (ln N)⁵ / N^(1/3)'
    values['m_J_psi'] = 8 * pi ** 2 * math.sqrt(2) * lnN ** 5 / N13

    formulas['m_eta'] = '2π² · (ln N)⁵ / N^(1/3)'
    values['m_eta'] = 2 * pi ** 2 * lnN ** 5 / N13

    formulas['m_Upsilon_1S'] = '√3 · (ln N)⁶ / (√2 · N^(1/3))'
    values['m_Upsilon_1S'] = math.sqrt(3) * lnN ** 6 / (math.sqrt(2) * N13)

    # ========================
    # МАССЫ КВАРКОВ
    # ========================
    formulas['m_quark_u'] = '√3 · (ln N)⁵ / (4π² · N^(1/3))'
    values['m_quark_u'] = math.sqrt(3) * lnN ** 5 / (4 * pi ** 2 * N13)

    formulas['m_quark_d'] = '(ln N)⁵ / (K · √3 · N^(1/3))'
    values['m_quark_d'] = lnN ** 5 / (K * math.sqrt(3) * N13)

    formulas['m_quark_s'] = 'π^(7/2) · (ln N)⁴ / N^(1/3)'
    values['m_quark_s'] = pi ** (3.5) * lnN ** 4 / N13

    formulas['m_quark_c'] = '2π² · (ln N)⁶ / (K³ · N^(1/3))'
    values['m_quark_c'] = 2 * pi ** 2 * lnN ** 6 / (K ** 3 * N13)

    formulas['m_quark_b'] = 'π · (ln N)⁶ / (K · √3 · N^(1/3))'
    values['m_quark_b'] = pi * lnN ** 6 / (K * math.sqrt(3) * N13)

    formulas['m_quark_t'] = 'K³ · (ln N)⁶ / (π² · N^(1/3))'
    values['m_quark_t'] = K ** 3 * lnN ** 6 / (pi ** 2 * N13)

    # ========================
    # АТОМНЫЕ И СПЕКТРАЛЬНЫЕ
    # ========================
    formulas['Rydberg'] = '4 · (ln N)³ · (ln K)³ / (π · K^(3/2))'
    values['Rydberg'] = 4 * lnN ** 3 * lnK ** 3 / (pi * K ** 1.5)

    formulas['Bohr_radius'] = 'K^(3/2) / (8π · (ln N)⁴ · ln K)'
    values['Bohr_radius'] = K ** 1.5 / (8 * pi * lnN ** 4 * lnK)

    formulas['Compton_e'] = 'K^(3/2) · ln K / (2π · (ln N)⁵)'
    values['Compton_e'] = K ** 1.5 * lnK / (2 * pi * lnN ** 5)

    formulas['Compton_p'] = '2 · K^(5/2) · ln K / (√π · (ln N)⁷)'
    values['Compton_p'] = 2 * K ** 2.5 * lnK / (math.sqrt(pi) * lnN ** 7)

    # ========================
    # ЭЛЕКТРОМАГНИТНЫЕ
    # ========================
    formulas['e_charge'] = '1 / (π · K^(3/2) · (ln N)⁷)'
    values['e_charge'] = 1.0 / (pi * K ** 1.5 * lnN ** 7)

    formulas['epsilon_0'] = 'N^(1/3) / (8π³ · ln K · (ln N)^20)'
    values['epsilon_0'] = N13 / (8 * pi ** 3 * lnK * lnN ** 20)

    formulas['mu_0'] = '8π · K⁴ · (ln K)³ · (ln N)^12 / N^(1/3)'
    values['mu_0'] = 8 * pi * K ** 4 * lnK ** 3 * lnN ** 12 / N13

    formulas['impedance'] = '8 · K² · π² · (ln K)² · (ln N)^16 / N^(1/3)'
    values['impedance'] = 8 * K ** 2 * pi ** 2 * lnK ** 2 * lnN ** 16 / N13

    formulas['flux_quantum'] = 'π² · √K · (ln N)^10 / N^(1/3)'
    values['flux_quantum'] = pi ** 2 * math.sqrt(K) * lnN ** 10 / N13

    # ========================
    # КОСМОЛОГИЯ
    # ========================
    formulas['Lambda'] = '(ln N)^12 / (√π · N^(2/3))'
    values['Lambda'] = lnN ** 12 / (math.sqrt(pi) * N13 ** 2)

    formulas['kappa_Einstein'] = '128 · K³ · (ln K)³ / ((ln N)³ · N^(1/3))'
    values['kappa_Einstein'] = 128 * K ** 3 * lnK ** 3 / (lnN ** 3 * N13)

    formulas['v_Higgs'] = '8π^(3/2) · (ln N)⁶ / (√2 · N^(1/3))'
    values['v_Higgs'] = 8 * pi ** 1.5 * lnN ** 6 / (math.sqrt(2) * N13)

    # ========================
    # ВРЕМЕНА ЖИЗНИ
    # ========================
    formulas['tau_mu'] = 'ln K / (K · √3 · (ln N)²)'
    values['tau_mu'] = lnK / (K * math.sqrt(3) * lnN ** 2)

    formulas['tau_tau'] = '1 / (2 · (ln N)⁵)'
    values['tau_tau'] = 1.0 / (2 * lnN ** 5)

    formulas['tau_pion'] = 'K² · √2 · π / (ln N)⁴'
    values['tau_pion'] = K ** 2 * math.sqrt(2) * pi / lnN ** 4

    formulas['tau_neutron'] = '√2 · N^(1/12) / (ln N)³'
    values['tau_neutron'] = math.sqrt(2) * N16 ** 0.5 / lnN ** 3  # N^(1/12)

    formulas['tau_kaon'] = '4 / (K^(3/2) · (ln N)³)'
    values['tau_kaon'] = 4.0 / (K ** 1.5 * lnN ** 3)

    formulas['tau_D_plus'] = '1 / (√π · K^(5/2) · (ln N)⁴)'
    values['tau_D_plus'] = 1.0 / (math.sqrt(pi) * K ** 2.5 * lnN ** 4)

    formulas['tau_B_plus'] = 'ln K · π / (2 · (ln N)⁵)'
    values['tau_B_plus'] = lnK * pi / (2 * lnN ** 5)

    formulas['tau_Lambda_b'] = 'ln K · √2 / (ln N)⁵'
    values['tau_Lambda_b'] = lnK * math.sqrt(2) / lnN ** 5

    formulas['tau_D0'] = 'ln K / (2π² · K² · (ln N)⁴)'
    values['tau_D0'] = lnK / (2 * pi ** 2 * K ** 2 * lnN ** 4)

    # ========================
    # БЕЗРАЗМЕРНЫЕ ОТНОШЕНИЯ
    # ========================
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
# ВЫЧИСЛЕНИЕ γ_i ИЗ ДАННЫХ
# ============================================================
def compute_gamma(f_model, f_exp):
    """Вычисляет γ из отношения эксперимента к модели"""
    if f_model <= 0 or f_exp <= 0:
        return None
    b0 = lnN * math.log(f_exp / f_model)
    return b0 / lnK


print("=" * 120)
print("ВЫЧИСЛЕНИЕ γ_i ДЛЯ ВСЕХ КОНСТАНТ")
print("=" * 120)
print(f"\n  γ = (ln N / ln K) · ln(f_exp / f_model)")
print(f"  ln N = {lnN:.6f}, ln K = {lnK:.6f}, ln N/ln K = {lnN / lnK:.6f}")
print()

# Вычисляем γ для всех констант
gamma_values = {}
for name in model_values:
    if name in exp_data:
        fm = model_values[name]
        fe = exp_data[name]
        gamma = compute_gamma(fm, fe)
        if gamma is not None:
            gamma_values[name] = gamma
    elif '/' in name:
        parts = name.split('/')
        if parts[0] in exp_data and parts[1] in exp_data:
            fm = model_values[name]
            fe = exp_data[parts[0]] / exp_data[parts[1]]
            gamma = compute_gamma(fm, fe)
            if gamma is not None:
                gamma_values[name] = gamma

# ============================================================
# ТАБЛИЦА γ_i
# ============================================================
print(f"  {'Константа':<22} {'f_model':<18} {'f_exp':<18} {'f_exp/f_model':<16} {'γ_i':<12} {'|γ_i|':<12}")
print(f"  {'─' * 90}")

for name in sorted(gamma_values.keys()):
    fm = model_values[name]
    fe = exp_data.get(name, None)
    if fe is None and '/' in name:
        parts = name.split('/')
        fe = exp_data[parts[0]] / exp_data[parts[1]]

    ratio = fe / fm
    gamma = gamma_values[name]
    print(f"  {name:<22} {fm:<18.8e} {fe:<18.8e} {ratio:<16.8f} {gamma:<12.6f} {abs(gamma):<12.6f}")

# ============================================================
# ГРУППИРОВКА ПО ТИПАМ
# ============================================================
print("\n" + "=" * 120)
print("ГРУППИРОВКА γ_i ПО ТИПАМ КОНСТАНТ")
print("=" * 120)

groups = {
    'Квантовые': ['ħ', 'h'],
    'Планковские': ['l_P', 't_P', 'm_P', 'E_P', 'T_P'],
    'Фундаментальные': ['c', 'G', 'k_B', 'α'],
    'Лептоны': ['m_e', 'm_muon', 'm_tau'],
    'Барионы': ['m_proton', 'm_neutron'],
    'Бозоны': ['m_W', 'm_Z', 'm_Higgs'],
    'Мезоны': ['m_pion', 'm_pion0', 'm_kaon0', 'm_D0', 'm_J_psi', 'm_eta', 'm_Upsilon_1S'],
    'Кварки': ['m_quark_u', 'm_quark_d', 'm_quark_s', 'm_quark_c', 'm_quark_b', 'm_quark_t'],
    'Атомные': ['Rydberg', 'Bohr_radius', 'Compton_e', 'Compton_p'],
    'Электромагнитные': ['e_charge', 'epsilon_0', 'mu_0', 'impedance', 'flux_quantum'],
    'Космология': ['Lambda', 'kappa_Einstein', 'v_Higgs'],
    'Времена жизни': ['tau_mu', 'tau_tau', 'tau_pion', 'tau_neutron', 'tau_kaon', 'tau_D_plus', 'tau_B_plus',
                      'tau_Lambda_b', 'tau_D0'],
    'Отношения': ['m_proton/m_e', 'm_muon/m_e', 'm_tau/m_e', 'm_W/m_e', 'm_Z/m_e', 'm_Higgs/m_e', 'm_W/m_Z',
                  'm_Higgs/m_W', 'm_P/m_e'],
}

for group_name, const_list in groups.items():
    gammas = [gamma_values[name] for name in const_list if name in gamma_values]
    if gammas:
        avg_gamma = np.mean(gammas)
        std_gamma = np.std(gammas)
        print(f"\n  {group_name}:")
        print(f"    Средний γ: {avg_gamma:.6f} ± {std_gamma:.6f}")
        print(f"    Диапазон: [{min(gammas):.6f}, {max(gammas):.6f}]")
        for name in const_list:
            if name in gamma_values:
                print(f"      {name:<20} γ = {gamma_values[name]:.6f}")

# ============================================================
# ПОИСК РАЦИОНАЛЬНЫХ ПРИБЛИЖЕНИЙ
# ============================================================
print("\n" + "=" * 120)
print("ПОИСК РАЦИОНАЛЬНЫХ ПРИБЛИЖЕНИЙ ДЛЯ γ_i")
print("=" * 120)


def find_rational(x, max_denom=30):
    """Находит наилучшее рациональное приближение"""
    best_num, best_denom = 0, 1
    best_err = float('inf')

    for denom in range(1, max_denom + 1):
        num = round(x * denom)
        err = abs(x - num / denom)
        if err < best_err:
            best_err = err
            best_num, best_denom = num, denom

    return best_num, best_denom, best_err


print(f"\n  {'Константа':<22} {'γ_i':<12} {'Рац. прибл.':<16} {'Ошибка':<12}")
print(f"  {'─' * 60}")

for name in sorted(gamma_values.keys()):
    gamma = gamma_values[name]
    num, denom, err = find_rational(gamma)
    sign = '-' if num < 0 else '+'
    print(f"  {name:<22} {gamma:<12.6f} {sign}{abs(num)}/{denom:<12} {err:<12.6f}")

# ============================================================
# СОХРАНЕНИЕ В ФАЙЛ
# ============================================================
import json

with open('gamma_values.json', 'w') as f:
    json.dump(gamma_values, f, indent=2)
print(f"\n  💾 γ_i сохранены в gamma_values.json")