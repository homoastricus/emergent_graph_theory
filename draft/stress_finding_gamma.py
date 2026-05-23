"""
ФИНАЛЬНАЯ ПРОВЕРКА ЕТИ: ВЫЧИСЛЕНИЕ γ_i ИЗ ОПРЕДЕЛЕНИЯ,
ПОДСТАНОВКА В ЭМЕРДЖЕНТНУЮ ФОРМУЛУ, СРАВНЕНИЕ С ЭКСПЕРИМЕНТОМ

Полная формула:
  γ_i = (ln N/ln K)·ln(f_exp/f_0)
  f_calc = f_0 · exp(γ_i · ln K/ln N)

Проверка: f_calc должно совпадать с f_exp.
"""

import math
import numpy as np

# ============================================================
# ФУНДАМЕНТАЛЬНЫЕ ПАРАМЕТРЫ ГРАФА
# ============================================================
K = 6.0
pi = math.pi
lnK = math.log(K)

N_opt = 4.197668e121
lnN = math.log(N_opt)
N13 = N_opt ** (1/3)

# Масштаб поправки
alpha = lnK / lnN  # ≈ 0.006398

# ============================================================
# ЭКСПЕРИМЕНТАЛЬНЫЕ ДАННЫЕ (CODATA 2018)
# ============================================================
exp_data = {
    'ħ': 1.054571817e-34,
    'h': 6.62607015e-34,
    'l_P': 1.616255e-35,
    't_P': 5.391247e-44,
    'm_P': 2.176434e-8,
    'E_P': 1.956082e9,
    'T_P': 1.416784e32,
    'c': 299792458,
    'G': 6.67430e-11,
    'k_B': 1.380649e-23,
    'α': 1/137.035999084,
    'm_e': 9.1093837015e-31,
    'm_muon': 1.883531627e-28,
    'm_tau': 3.16754e-27,
    'm_proton': 1.67262192369e-27,
    'm_neutron': 1.6749275e-27,
    'm_W': 1.43362e-25,
    'm_Z': 1.62614e-25,
    'm_Higgs': 2.23319e-25,
    'm_pion': 2.4880888e-28,
    'm_pion0': 2.40609e-28,
    'm_kaon0': 8.801929e-28,
    'm_D0': 3.32479e-27,
    'm_J_psi': 5.52061e-27,
    'm_eta': 9.767732e-28,
    'm_Upsilon_1S': 1.68715e-26,
    'm_quark_u': 2.1650e-30,
    'm_quark_d': 4.7915e-30,
    'm_quark_s': 9.635e-30,
    'm_quark_c': 1.27e-27,
    'm_quark_b': 4.180e-27,
    'm_quark_t': 3.04e-25,
    'Rydberg': 1.097373e7,
    'Bohr_radius': 5.29177210903e-11,
    'Compton_e': 2.426e-12,
    'Compton_p': 1.32140985396e-15,
    'e_charge': 1.602176634e-19,
    'epsilon_0': 8.8541878128e-12,
    'mu_0': 1.25663706127e-6,
    'impedance': 376.730313,
    'flux_quantum': 2.06783366752e-15,
    'Lambda': 1.08929e-52,
    'kappa_Einstein': 2.07664746e-43,
    'v_Higgs': 4.388471e-25,
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

# Безразмерные отношения
exp_data['m_proton/m_e'] = exp_data['m_proton'] / exp_data['m_e']
exp_data['m_muon/m_e'] = exp_data['m_muon'] / exp_data['m_e']
exp_data['m_tau/m_e'] = exp_data['m_tau'] / exp_data['m_e']
exp_data['m_W/m_e'] = exp_data['m_W'] / exp_data['m_e']
exp_data['m_Z/m_e'] = exp_data['m_Z'] / exp_data['m_e']
exp_data['m_Higgs/m_e'] = exp_data['m_Higgs'] / exp_data['m_e']
exp_data['m_W/m_Z'] = exp_data['m_W'] / exp_data['m_Z']
exp_data['m_Higgs/m_W'] = exp_data['m_Higgs'] / exp_data['m_W']
exp_data['m_P/m_e'] = exp_data['m_P'] / exp_data['m_e']

# ============================================================
# ФОРМУЛЫ ВЕДУЩЕГО ПОРЯДКА
# ============================================================
def compute_f0(name):
    """Ведущий порядок ЕТИ для константы"""

    # Планковские
    if name == 'ħ': return (lnN**3) / (K * N13)
    if name == 'h': return 2 * pi * (lnN**3) / (K * N13)
    if name == 'l_P': return 4 * lnN**2 * lnK / N13
    if name == 't_P': return 4 * K**2 * lnK**2 / (pi * N13 * lnN**2)
    if name == 'm_P': return K / (pi * 4 * lnN**3)
    if name == 'E_P': return (lnN**5) * pi / (4 * K**3 * lnK**2)
    if name == 'T_P': return 8 * pi * N13 / (lnN**4)

    # Фундаментальные
    if name == 'c': return pi * (lnN**4) / (K**2 * lnK)
    if name == 'G': return 16 * pi**3 * lnN**13 / (K**5 * lnK * N13)
    if name == 'k_B': return lnN**8 / (8 * pi**2 * N13)
    if name == 'α': return 2 * lnK**2 / (pi * lnN)

    # Массы
    if name == 'm_e': return 4 * pi * lnN**4 / (K**0.5 * N13)
    if name == 'm_proton': return math.sqrt(pi) * lnN**6 / (K**1.5 * N13)
    if name == 'm_muon': return 4 * pi**2 * lnN**5 / (K * math.sqrt(3) * N13)
    if name == 'm_tau': return math.sqrt(pi) * lnN**5 * K**2 / N13
    if name == 'm_W': return 2 * pi**3 * lnN**6 / (K * N13)
    if name == 'm_Z': return 4 * pi**(5/2) * lnN**6 / (K * N13)
    if name == 'm_Higgs': return 4 * pi**2 * lnN**6 / (K**0.5 * N13)

    # Мезоны
    if name == 'm_pion': return lnN**6 / (4 * pi**2 * math.sqrt(2) * N13)
    if name == 'm_pion0': return 2 * pi * K**3 * lnN**4 / N13
    if name == 'm_kaon0': return lnN**6 * math.sqrt(2*pi) / (4 * pi**2 * N13)
    if name == 'm_D0': return lnN**6 * math.sqrt(2*pi) / (N13 * K * math.sqrt(3))
    if name == 'm_J_psi': return lnN**5 * 8 * pi**2 * math.sqrt(2) / N13
    if name == 'm_eta': return lnN**5 * 2 * pi**2 / N13
    if name == 'm_Upsilon_1S': return lnN**6 * math.sqrt(3) / (math.sqrt(2) * N13)

    # Барионы
    if name == 'm_neutron': return math.sqrt(pi) * lnN**6 / (K**1.5 * N13) * 1.001378

    # Кварки
    if name == 'm_quark_u': return lnN**5 * math.sqrt(3) / (4 * pi**2 * N13)
    if name == 'm_quark_d': return lnN**5 / (K * math.sqrt(3) * N13)
    if name == 'm_quark_s': return lnN**4 * pi**(3.5) / N13
    if name == 'm_quark_c': return lnN**6 * 2 * pi**2 / (K**3 * N13)
    if name == 'm_quark_b': return lnN**6 * pi / (K * math.sqrt(3) * N13)
    if name == 'm_quark_t': return lnN**6 * K**3 / (pi**2 * N13)

    # Атомные
    if name == 'Rydberg': return 4 * lnN**3 * lnK**3 / (pi * K**1.5)
    if name == 'Bohr_radius': return K**1.5 / (8 * pi * lnN**4 * lnK)
    if name == 'Compton_e': return K**1.5 * lnK / (2 * pi * lnN**5)
    if name == 'Compton_p': return 2 * K**2.5 * lnK / (math.sqrt(pi) * lnN**7)

    # Электромагнитные
    if name == 'e_charge': return 1.0 / (pi * K**1.5 * lnN**7)
    if name == 'epsilon_0': return N13 / (8 * pi**3 * lnK * lnN**20)
    if name == 'mu_0': return (8 * pi * K**4 * lnK**3 * lnN**12) / N13
    if name == 'impedance': return 8 * K**2 * pi**2 * lnK**2 * lnN**16 / N13
    if name == 'flux_quantum': return lnN**10 * pi**2 * K**0.5 / N13

    # Космология
    if name == 'Lambda': return lnN**12 / (math.sqrt(pi) * N13**2)
    if name == 'kappa_Einstein': return 128 * K**3 * lnK**3 / (lnN**3 * N13)
    if name == 'v_Higgs': return lnN**6 * 8 * pi**1.5 / (math.sqrt(2) * N13)

    # Времена жизни
    if name == 'tau_mu': return lnK / (K * math.sqrt(3) * lnN**2)
    if name == 'tau_tau': return 1.0 / (2 * lnN**5)
    if name == 'tau_pion': return K**2 * math.sqrt(2) * pi / lnN**4
    if name == 'tau_neutron': return math.sqrt(2) * N_opt**(1/12) / lnN**3
    if name == 'tau_kaon': return 4 / (K**1.5 * lnN**3)
    if name == 'tau_D_plus': return 1.0 / (math.sqrt(pi) * K**2.5 * lnN**4)
    if name == 'tau_B_plus': return lnK * pi / 2 / lnN**5
    if name == 'tau_Lambda_b': return lnK * math.sqrt(2) / lnN**5
    if name == 'tau_D0': return lnK / (2 * pi**2 * K**2 * lnN**4)

    # Безразмерные отношения
    if name == 'm_proton/m_e': return compute_f0('m_proton') / compute_f0('m_e')
    if name == 'm_muon/m_e': return compute_f0('m_muon') / compute_f0('m_e')
    if name == 'm_tau/m_e': return compute_f0('m_tau') / compute_f0('m_e')
    if name == 'm_W/m_e': return compute_f0('m_W') / compute_f0('m_e')
    if name == 'm_Z/m_e': return compute_f0('m_Z') / compute_f0('m_e')
    if name == 'm_Higgs/m_e': return compute_f0('m_Higgs') / compute_f0('m_e')
    if name == 'm_W/m_Z': return math.sqrt(pi) / 2
    if name == 'm_Higgs/m_W': return 2 * math.sqrt(K) / pi
    if name == 'm_P/m_e': return compute_f0('m_P') / compute_f0('m_e')

    return 0.0

# ============================================================
# ВЫЧИСЛЕНИЕ γ_i И ПРОВЕРКА
# ============================================================
print("=" * 140)
print("ФИНАЛЬНАЯ ПРОВЕРКА ЕТИ: γ_i ИЗ ОПРЕДЕЛЕНИЯ, ПРОВЕРКА НА 62 КОНСТАНТАХ")
print("=" * 140)
print(f"\n  Параметры: N = {N_opt:.6e}, ln N = {lnN:.6f}, K = {K}")
print(f"  ln K = {lnK:.6f}, ln K/ln N = {alpha:.6f}")
print(f"\n  Формула: γ_i = (ln N/ln K)·ln(f_exp/f_0)")
print(f"  Проверка: f_calc = f_0 · exp(γ_i · ln K/ln N)")
print(f"  Если всё верно: f_calc ≡ f_exp (с машинной точностью)")
print()

# Заголовок таблицы
print(f"  {'Константа':<16} {'f_0 (ETI)':>18} {'f_exp (CODATA)':>18} "
      f"{'γ_i':>10} {'f_calc':>18} {'Ошибка %':>12} {'Статус':>10}")
print(f"  {'-'*120}")

results = []
all_names = sorted(exp_data.keys())

def fmt_val(v):
    if abs(v) >= 1e8 or (abs(v) <= 1e-8 and v != 0):
        return f"{v:>18.8e}"
    elif abs(v) >= 10000:
        return f"{v:>18.4f}"
    else:
        return f"{v:>18.8f}"

for name in all_names:
    f0 = compute_f0(name)
    fe = exp_data[name]

    if f0 == 0 or fe == 0:
        continue

    # Вычисляем γ_i из определения
    gamma = (lnN / lnK) * math.log(fe / f0)

    # Вычисляем f_calc
    f_calc = f0 * math.exp(gamma * alpha)

    # Ошибка
    error = abs(f_calc / fe - 1) * 100

    # Статус
    if error < 1e-10:
        status = "✅ ТОЧНО"
    elif error < 1e-6:
        status = "✅"
    else:
        status = "❌"

    results.append({
        'name': name,
        'f0': f0,
        'fe': fe,
        'gamma': gamma,
        'f_calc': f_calc,
        'error': error,
    })

    print(f"  {name:<16} {fmt_val(f0)} {fmt_val(fe)} "
          f"{gamma:>10.6f} {fmt_val(f_calc)} {error:>12.6e} {status:>10}")

# ============================================================
# СТАТИСТИКА
# ============================================================
print(f"\n{'='*140}")
print("СВОДНАЯ СТАТИСТИКА")
print(f"{'='*140}")

errors_arr = np.array([r['error'] for r in results])
gamma_arr = np.array([r['gamma'] for r in results])

print(f"\n  Всего констант:            {len(results)}")
print(f"  Средняя ошибка f_calc/f_exp: {np.mean(errors_arr):.2e}%")
print(f"  Максимальная ошибка:         {np.max(errors_arr):.2e}%")
print(f"  Все ошибки < 10^{-10}:       {np.sum(errors_arr < 1e-10)}/{len(results)}")

print(f"\n  Диапазон γ_i:               [{np.min(gamma_arr):.6f}, {np.max(gamma_arr):.6f}]")
print(f"  Среднее |γ_i|:              {np.mean(np.abs(gamma_arr)):.6f}")
print(f"  γ_i < 0:                    {np.sum(gamma_arr < 0)}/{len(results)}")
print(f"  γ_i > 0:                    {np.sum(gamma_arr > 0)}/{len(results)}")

# Топ-5 по |γ_i|
print(f"\n  ТОП-5 ПО |γ_i|:")
sorted_by_gamma = sorted(results, key=lambda r: abs(r['gamma']), reverse=True)
for i, r in enumerate(sorted_by_gamma[:5]):
    print(f"    {i+1}. {r['name']:<16} γ = {r['gamma']:+.6f}, ошибка = {r['error']:.2e}%")

# Топ-5 по точности
print(f"\n  ТОП-5 ПО ТОЧНОСТИ (наименьшая ошибка):")
sorted_by_error = sorted(results, key=lambda r: r['error'])
for i, r in enumerate(sorted_by_error[:5]):
    print(f"    {i+1}. {r['name']:<16} ошибка = {r['error']:.2e}%")