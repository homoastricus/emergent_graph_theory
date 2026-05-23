"""
ПОЛНАЯ ТАБЛИЦА ЕТИ: ВСЕ 62 КОНСТАНТЫ С ФОРМУЛАМИ, ПАРАМЕТРАМИ И ПОПРАВКАМИ

Столбцы:
  0. Название константы
  1. Формула ведущего порядка
  2. a_i (степень при ln N)
  3. b_i (степень при N)
  4. n_i (квантовое число)
  5. γ_i (логарифмическая поправка)
  6. Значение без поправки (ведущий порядок)
  7. Значение с поправкой
  8. Экспериментальное значение (CODATA)
  9. exp/f₀
  10. exp/f_corr
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

correction_scale = lnK / lnN

# ============================================================
# ВСЕ ДАННЫЕ В ОДНОМ МЕСТЕ
# ============================================================
constants = [
    # (name, formula_str, a, b, n, f0_func, exp_val)

    # === Квантовые (b=1/3) ===
    ('ħ', '(ln N)³/(K·N^(1/3))', 3, 1/3, 13,
     lambda: (lnN**3)/(K*N13),
     1.054571817e-34),

    ('h', '2π(ln N)³/(K·N^(1/3))', 3, 1/3, 13,
     lambda: 2*pi*(lnN**3)/(K*N13),
     6.62607015e-34),

    # === Планковские ===
    ('l_P', '4(ln N)²·ln K/N^(1/3)', 2, 1/3, 14,
     lambda: 4*lnN**2*lnK/N13,
     1.616255e-35),

    ('t_P', '4K²(ln K)²/(π·N^(1/3)·(ln N)²)', -2, 1/3, 18,
     lambda: 4*K**2*lnK**2/(pi*N13*lnN**2),
     5.391247e-44),

    ('m_P', 'K/(4π(ln N)³)', -3, 0, 3,
     lambda: K/(pi*4*lnN**3),
     2.176434e-8),

    ('E_P', 'π(ln N)⁵/(4K³(ln K)²)', 5, 0, -5,
     lambda: (lnN**5)*pi/(4*K**3*lnK**2),
     1.956082e9),

    ('T_P', '8π·N^(1/3)/(ln N)⁴', -4, -1/3, None,
     lambda: 8*pi*N13/(lnN**4),
     1.416784e32),

    # === Фундаментальные ===
    ('c', 'π(ln N)⁴/(K²·ln K)', 4, 0, -4,
     lambda: pi*lnN**4/(K**2*lnK),
     299792458),

    ('G', '16π³(ln N)^13/(K⁵·ln K·N^(1/3))', 13, 1/3, 3,
     lambda: 16*pi**3*lnN**13/(K**5*lnK*N13),
     6.67430e-11),

    ('k_B', '(ln N)⁸/(8π²·N^(1/3))', 8, 1/3, 8,
     lambda: lnN**8/(8*pi**2*N13),
     1.380649e-23),

    ('α', '2(ln K)²/(π·ln N)', -1, 0, 1,
     lambda: 2*lnK**2/(pi*lnN),
     1/137.035999084),

    # === Лептоны (b=1/3) ===
    ('m_e', '4π(ln N)⁴/(√K·N^(1/3))', 4, 1/3, 12,
     lambda: 4*pi*lnN**4/(K**0.5*N13),
     9.1093837015e-31),

    ('m_muon', '4π²(ln N)⁵/(K√3·N^(1/3))', 5, 1/3, 11,
     lambda: 4*pi**2*lnN**5/(K*math.sqrt(3)*N13),
     1.883531627e-28),

    ('m_tau', '√π(ln N)⁵·K²/N^(1/3)', 5, 1/3, 11,
     lambda: math.sqrt(pi)*lnN**5*K**2/N13,
     3.16754e-27),

    # === Барионы (b=1/3) ===
    ('m_proton', '√π(ln N)⁶/(K³/²·N^(1/3))', 6, 1/3, 10,
     lambda: math.sqrt(pi)*lnN**6/(K**1.5*N13),
     1.67262192369e-27),
    #
    # ('m_neutron', '√π(ln N)⁶/(K³/²·N^(1/3))', 6, 1/3, 10,
    #  lambda: math.sqrt(pi)*lnN**6/(K**1.5*N13),
    #  1.6749275e-27),

    # === Бозоны (b=1/3) ===
    ('m_W', '2π³(ln N)⁶/(K·N^(1/3))', 6, 1/3, 10,
     lambda: 2*pi**3*lnN**6/(K*N13),
     1.43362e-25),

    ('m_Z', '4π⁵/²(ln N)⁶/(K·N^(1/3))', 6, 1/3, 10,
     lambda: 4*pi**(5/2)*lnN**6/(K*N13),
     1.62614e-25),

    ('m_Higgs', '4π²(ln N)⁶/(√K·N^(1/3))', 6, 1/3, 10,
     lambda: 4*pi**2*lnN**6/(K**0.5*N13),
     2.23319e-25),

    # === Мезоны (b=1/3) ===
    ('m_pion', '(ln N)⁶/(4π²√2·N^(1/3))', 6, 1/3, 10,
     lambda: lnN**6/(4*pi**2*math.sqrt(2)*N13),
     2.4880888e-28),

    ('m_pion0', '2πK³(ln N)⁴/N^(1/3)', 4, 1/3, 12,
     lambda: 2*pi*K**3*lnN**4/N13,
     2.40609e-28),

    ('m_kaon0', '(ln N)⁶√(2π)/(4π²·N^(1/3))', 6, 1/3, 10,
     lambda: lnN**6*math.sqrt(2*pi)/(4*pi**2*N13),
     8.801929e-28),

    ('m_D0', '(ln N)⁶√(2π)/(K√3·N^(1/3))', 6, 1/3, 10,
     lambda: lnN**6*math.sqrt(2*pi)/(N13*K*math.sqrt(3)),
     3.32479e-27),

    ('m_J_psi', '8π²√2(ln N)⁵/N^(1/3)', 5, 1/3, 11,
     lambda: lnN**5*8*pi**2*math.sqrt(2)/N13,
     5.52061e-27),

    ('m_eta', '2π²(ln N)⁵/N^(1/3)', 5, 1/3, 11,
     lambda: lnN**5*2*pi**2/N13,
     9.767732e-28),

    ('m_Upsilon_1S', '(ln N)⁶√3/(√2·N^(1/3))', 6, 1/3, 10,
     lambda: lnN**6*math.sqrt(3)/(math.sqrt(2)*N13),
     1.68715e-26),

    # === Кварки (b=1/3) ===
    ('m_quark_u', '(ln N)⁵√3/(4π²·N^(1/3))', 5, 1/3, 11,
     lambda: lnN**5*math.sqrt(3)/(4*pi**2*N13),
     2.1650e-30),

    ('m_quark_d', '(ln N)⁵/(K√3·N^(1/3))', 5, 1/3, 11,
     lambda: lnN**5/(K*math.sqrt(3)*N13),
     4.7915e-30),

    ('m_quark_s', '(ln N)⁴π^(7/2)/N^(1/3)', 4, 1/3, 12,
     lambda: lnN**4*pi**(3.5)/N13,
     9.635e-30),

    ('m_quark_c', '2π²(ln N)⁶/(K³·N^(1/3))', 6, 1/3, 10,
     lambda: lnN**6*2*pi**2/(K**3*N13),
     1.27e-27),

    ('m_quark_b', 'π(ln N)⁶/(K√3·N^(1/3))', 6, 1/3, 10,
     lambda: lnN**6*pi/(K*math.sqrt(3)*N13),
     4.180e-27),

    ('m_quark_t', 'K³(ln N)⁶/(π²·N^(1/3))', 6, 1/3, 10,
     lambda: K**3*lnN**6/(pi**2*N13),
     3.04e-25),

    # === Атомные (b=0) ===
    ('Rydberg', '4(ln N)³(ln K)³/(π·K³/²)', 3, 0, -3,
     lambda: 4*lnN**3*lnK**3/(pi*K**1.5),
     1.097373e7),

    ('Bohr_radius', 'K³/²/(8π(ln N)⁴·ln K)', -4, 0, 4,
     lambda: K**1.5/(8*pi*lnN**4*lnK),
     5.29177210903e-11),

    ('Compton_e', 'K³/²·ln K/(2π(ln N)⁵)', -5, 0, 5,
     lambda: K**1.5*lnK/(2*pi*lnN**5),
     2.426e-12),

    ('Compton_p', '2K⁵/²·ln K/(√π(ln N)⁷)', -7, 0, 7,
     lambda: 2*K**2.5*lnK/(math.sqrt(pi)*lnN**7),
     1.32140985396e-15),

    # === Электромагнитные ===
    ('e_charge', '1/(π·K³/²·(ln N)⁷)', -7, 0, 7,
     lambda: 1.0/(pi*K**1.5*lnN**7),
     1.602176634e-19),

    ('epsilon_0', 'N^(1/3)/(8π³·ln K·(ln N)²⁰)', -20, -1/3, None,
     lambda: N13/(8*pi**3*lnK*lnN**20),
     8.8541878128e-12),

    ('mu_0', '8πK⁴(ln K)³(ln N)¹²/N^(1/3)', 12, -1/3, None,
     lambda: 8*pi*K**4*lnK**3*lnN**12/N13,
     1.25663706127e-6),

    ('impedance', '8K²π²(ln K)²(ln N)¹⁶/N^(1/3)', 16, -1/3, None,
     lambda: 8*K**2*pi**2*lnK**2*lnN**16/N13,
     376.730313),

    ('flux_quantum', '(ln N)¹⁰π²√K/N^(1/3)', 10, -1/3, None,
     lambda: lnN**10*pi**2*K**0.5/N13,
     2.06783366752e-15),

    # === Космология ===
    ('Lambda', '(ln N)¹²/(√π·N^(2/3))', 12, -2/3, None,
     lambda: lnN**12/(math.sqrt(pi)*N13**2),
     1.08929e-52),

    ('kappa_Einstein', '128K³(ln K)³/((ln N)³N^(1/3))', -3, -1/3, None,
     lambda: 128*K**3*lnK**3/(lnN**3*N13),
     2.07664746e-43),

    ('v_Higgs', '8π³/²(ln N)⁶/(√2·N^(1/3))', 6, 1/3, 10,
     lambda: lnN**6*8*pi**1.5/(math.sqrt(2)*N13),
     4.388471e-25),

    # === Времена жизни (b=0) ===
    ('tau_mu', 'ln K/(K√3·(ln N)²)', -2, 0, 2,
     lambda: lnK/(K*math.sqrt(3)*lnN**2),
     2.1969811e-6),

    ('tau_tau', '1/(2(ln N)⁵)', -5, 0, 5,
     lambda: 1.0/(2*lnN**5),
     2.903e-13),

    ('tau_pion', 'K²√2·π/(ln N)⁴', -4, 0, 4,
     lambda: K**2*math.sqrt(2)*pi/lnN**4,
     2.6033e-8),

    ('tau_neutron', '√2·N^(1/12)/(ln N)³', -3, 1/12, None,
     lambda: math.sqrt(2)*N_opt**(1/12)/lnN**3,
     877.8),

    ('tau_kaon', '4/(K³/²·(ln N)³)', -3, 0, 3,
     lambda: 4/(K**1.5*lnN**3),
     1.2380e-8),

    ('tau_D_plus', '1/(√π·K⁵/²·(ln N)⁴)', -4, 0, 4,
     lambda: 1.0/(math.sqrt(pi)*K**2.5*lnN**4),
     1.040e-12),

    ('tau_B_plus', '(ln K)·π/(2(ln N)⁵)', -5, 0, 5,
     lambda: lnK*pi/2/lnN**5,
     1.638e-12),

    ('tau_Lambda_b', '(ln K)√2/(ln N)⁵', -5, 0, 5,
     lambda: lnK*math.sqrt(2)/lnN**5,
     1.471e-12),

    ('tau_D0', 'ln K/(2π²K²(ln N)⁴)', -4, 0, 4,
     lambda: lnK/(2*pi**2*K**2*lnN**4),
     4.101e-13),

    # === Безразмерные отношения (b=0) ===
    ('m_proton/m_e', 'm_p/m_e', 2, 0, -2,
     lambda: (math.sqrt(pi)*lnN**6/(K**1.5*N13))/(4*pi*lnN**4/(K**0.5*N13)),
     1.67262192369e-27 / 9.1093837015e-31),

    ('m_muon/m_e', 'm_μ/m_e', 1, 0, -1,
     lambda: (4*pi**2*lnN**5/(K*math.sqrt(3)*N13))/(4*pi*lnN**4/(K**0.5*N13)),
     1.883531627e-28 / 9.1093837015e-31),

    ('m_tau/m_e', 'm_τ/m_e', 1, 0, -1,
     lambda: (math.sqrt(pi)*lnN**5*K**2/N13)/(4*pi*lnN**4/(K**0.5*N13)),
     3.16754e-27 / 9.1093837015e-31),

    ('m_W/m_e', 'm_W/m_e', 2, 0, -2,
     lambda: (2*pi**3*lnN**6/(K*N13))/(4*pi*lnN**4/(K**0.5*N13)),
     1.43362e-25 / 9.1093837015e-31),

    ('m_Z/m_e', 'm_Z/m_e', 2, 0, -2,
     lambda: (4*pi**(5/2)*lnN**6/(K*N13))/(4*pi*lnN**4/(K**0.5*N13)),
     1.62614e-25 / 9.1093837015e-31),

    ('m_Higgs/m_e', 'm_H/m_e', 2, 0, -2,
     lambda: (4*pi**2*lnN**6/(K**0.5*N13))/(4*pi*lnN**4/(K**0.5*N13)),
     2.23319e-25 / 9.1093837015e-31),

    ('m_W/m_Z', '√π/2', 0, 0, 0,
     lambda: math.sqrt(pi)/2,
     1.43362e-25 / 1.62614e-25),

    ('m_Higgs/m_W', '2√K/π', 0, 0, 0,
     lambda: 2*math.sqrt(K)/pi,
     2.23319e-25 / 1.43362e-25),

    ('m_P/m_e', 'm_P/m_e', 0, 0, 0,
     lambda: (K/(pi*4*lnN**3))/(4*pi*lnN**4/(K**0.5*N13)),
     2.176434e-8 / 9.1093837015e-31),
]

# ============================================================
# ИЗМЕРЕННЫЕ γ_i
# ============================================================
gamma_dict = {
    'ħ': 0.192835, 'h': 0.192835,
    'l_P': -0.104361, 't_P': -0.222142, 'm_P': 0.179369, 'E_P': 0.415022, 'T_P': -0.364637,
    'c': 0.117799, 'G': -0.048154, 'k_B': 0.224670, 'α': -0.015396,
    'm_e': 0.515264, 'm_muon': 0.061400, 'm_tau': 0.248402,
    'm_proton': -0.120978,
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
# ВЫВОД ТАБЛИЦЫ
# ============================================================
print("=" * 155)
print("ПОЛНАЯ ТАБЛИЦА ЕТИ: 62 ФУНДАМЕНТАЛЬНЫЕ КОНСТАНТЫ С ПОПРАВКАМИ")
print("=" * 155)
print(f"\n  N = {N_opt:.6e}  |  ln N = {lnN:.6f}  |  K = {K}  |  ln K = {lnK:.6f}")
print(f"  Поправка: f_corr = f₀ × exp(γ_i × ln K / ln N)")
print(f"  ln K / ln N = {correction_scale:.6f}\n")

# Заголовок
print(f"  {'Константа':<16} {'Формула':<42} {'a':>3} {'b':>6} {'n':>4} {'γ_i':>8} "
      f"{'f₀':>18} {'f_corr':>18} {'f_exp':>18} "
      f"{'exp/f₀':>10} {'exp/f_corr':>12}")
print(f"  {'-'*170}")

# Форматирование
def fmt_val(v):
    if v == 0:
        return f"{'0.0':>18}"
    if abs(v) >= 1e8 or (abs(v) <= 1e-8 and v != 0):
        return f"{v:>18.8e}"
    elif abs(v) >= 10000:
        return f"{v:>18.4f}"
    elif abs(v) >= 1:
        return f"{v:>18.8f}"
    else:
        return f"{v:>18.8e}"

errors_before = []
errors_after = []

for c in constants:
    name, formula, a, b, n, f0_func, fe = c

    f0 = f0_func()
    gamma = gamma_dict.get(name, 0.0)
    f_corr = f0 * math.exp(gamma * correction_scale)

    ratio_before = fe / f0 if f0 != 0 else 0
    ratio_after = fe / f_corr if f_corr != 0 else 0

    if f0 != 0:
        errors_before.append(abs(fe/f0 - 1) * 100)
    if f_corr != 0:
        errors_after.append(abs(fe/f_corr - 1) * 100)

    n_str = f"{n}" if n is not None else "—"

    print(f"  {name:<16} {formula:<42} {a:>3} {b:>6.2f} {n_str:>4} {gamma:>8.6f} "
          f"{fmt_val(f0)} {fmt_val(f_corr)} {fmt_val(fe)} "
          f"{ratio_before:>10.6f} {ratio_after:>12.8f}")

# ============================================================
# СТАТИСТИКА
# ============================================================
print(f"\n{'='*155}")
print("СВОДНАЯ СТАТИСТИКА")
print(f"{'='*155}")

print(f"\n  Всего констант:                {len(constants)}")
print(f"  Средняя ошибка без поправки:    {np.mean(errors_before):.6f}%")
print(f"  Средняя ошибка с поправкой:     {np.mean(errors_after):.10f}%")
print(f"  Улучшение:                      {np.mean(errors_before)/np.mean(errors_after):.0f} раз")
print(f"  Макс. ошибка без поправки:      {np.max(errors_before):.4f}%")
print(f"  Макс. ошибка с поправкой:       {np.max(errors_after):.6f}%")