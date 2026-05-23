"""
ИССЛЕДОВАНИЕ ЭВОЛЮЦИИ ЭМЕРДЖЕНТНЫХ КОНСТАНТ ПРИ РОСТЕ N
От N = 10^10 до N = 10^122 с шагом в один порядок
Расширенная версия: добавлены масса и заряд электрона,
вакуум Хиггса, космология, безразмерные отношения
"""

import math
import numpy as np

# ========== КОНСТАНТЫ ==========
K = 6.0
pi = math.pi
lnK = math.log(K)

# ========== ДИАПАЗОН N ==========
powers = list(range(10, 123))  # от 10^10 до 10^122
N_values = [10.0 ** p for p in powers]


# ========== ФУНКЦИИ ДЛЯ ВЫЧИСЛЕНИЯ КОНСТАНТ ==========
def compute_constants(N):
    lnN = math.log(N)
    N13 = N ** (1 / 3)
    N16 = N ** (1 / 6)
    N23 = N ** (2 / 3)
    p_val = 1 / (K * N13)
    Kp = K * p_val
    sqrtK = math.sqrt(K)
    sqrt3 = math.sqrt(3)
    sqrt2 = math.sqrt(2)
    sqrtPi = math.sqrt(pi)

    # Планковские величины (новый самосогласованный набор)
    lP_val = 4 * (lnN ** 2) * lnK / N13
    tP_val = 4 * K ** 2 * lnK ** 2 / (pi * lnN ** 2 * N13)
    mP_val = K / (4 * pi * lnN ** 3)
    EP_val = pi * (lnN ** 5) / (4 * K ** 3 * lnK ** 2)
    TP_val = 2 * pi ** 3 * N13 / (K ** 3 * lnK ** 2 * lnN ** 3)

    # Фундаментальные эмерджентные константы
    hbar_val = (lnN ** 3) / (K * N13)
    c_val = pi * (lnN ** 4) / (K ** 2 * lnK)
    G_val = 16 * pi ** 3 * lnN ** 13 / (K ** 5 * lnK * N13)
    alpha_val = 2 * lnK ** 2 / (pi * lnN)
    kB_val = Kp * (lnN ** 8) / (8 * pi ** 2)

    # Электродинамика
    qe_val = 1.0 / (pi * K ** (3 / 2) * lnN ** 7)
    ep0_val = N13 / (8 * pi ** 3 * lnK * lnN ** 20)
    mu0_val = (8 * pi * K ** 4 * lnK ** 3 * lnN ** 12) / N13

    # Массы частиц
    me_val = 4 * pi * lnN ** 4 / (sqrtK * N13)
    m_muon = 4 * pi ** 2 * lnN ** 5 / (K * sqrt3 * N13)
    m_tau = sqrtPi * lnN ** 5 * K ** 2 / N13
    m_proton = sqrtPi * lnN ** 6 / (K ** (3 / 2) * N13)

    # Бозоны
    m_W = 2 * pi ** 3 * lnN ** 6 / (N13 * K)
    m_Z = lnN ** 6 * 4 * pi ** (5 / 2) / (N13 * K)
    m_Higgs = lnN ** 6 * 4 * pi ** 2 / (N13 * sqrtK)

    # Кварки (примеры)
    m_qu_top = lnN ** 6 * K ** 3 / (pi ** 2 * N13)
    m_qu_bottom = lnN ** 6 * pi / (K * sqrt3 * N13)
    m_qu_charm = lnN ** 6 * 2 * pi ** 2 / (K ** 3 * N13)
    m_qu_strange = lnN ** 4 * pi ** (7 / 2) / N13
    m_qu_up = lnN ** 5 * sqrt3 / (4 * pi ** 2 * N13)
    m_qu_down = lnN ** 5 / (K * sqrt3 * N13)

    # Космология
    Lambda_val = lnN ** 12 / (sqrtPi * N23)
    kappa_val = 2 * (4 * K * lnK / lnN) ** 3 / N13
    v_Higgs = lnN ** 6 * 8 * pi ** (3 / 2) / (sqrt2 * N13)

    # Постоянная Ридберга и Боровский радиус
    Rydberg = 4 * lnN ** 3 * lnK ** 3 / (pi * K ** (3 / 2))
    bor_radius = K ** (3 / 2) / (8 * pi * lnN ** 4 * lnK)

    # Характерные энергии
    E_rydberg = 8 * pi * lnK ** 2 * lnN ** 10 / (K ** (9 / 2) * N13)

    # Безразмерные отношения
    mW_mZ = sqrtPi / 2
    mH_mW = 2 * sqrtK / pi
    mp_me = lnN ** 2 / (4 * sqrtPi * K)
    m_muon_me = pi * sqrtK * lnN / sqrt3

    # λ-фактор и эффективная размерность
    lambda_factor = pi ** 2 * lnN / (4 * K ** 3 * lnK ** 2)
    d_eff = 3 * lambda_factor

    # Функция геометрического резонанса
    F_val = (K + math.log(p_val)) / (p_val - lnN)
    F_error = abs(F_val - 1 / pi) / (1 / pi) * 100

    return {
        # Базовые параметры
        'lnN': lnN,
        'p': p_val,
        'x': -1 / 3,
        'U': 3.0,

        # Функция геометрического резонанса
        'F(N)': F_val,
        'F_target': 1 / pi,
        'F_error_%': F_error,

        # Фундаментальные константы
        'ħ': hbar_val,
        'c': c_val,
        'G': G_val,
        'α': alpha_val,
        'k_B': kB_val,

        # Электродинамика
        'q_e': qe_val,
        'ε₀': ep0_val,
        'μ₀': mu0_val,

        # Планковские величины
        'l_P': lP_val,
        't_P': tP_val,
        'm_P': mP_val,
        'E_P': EP_val,
        'T_P': TP_val,

        # Массы фермионов
        'm_e': me_val,
        'm_μ': m_muon,
        'm_τ': m_tau,
        'm_p': m_proton,
        'm_u': m_qu_up,
        'm_d': m_qu_down,
        'm_s': m_qu_strange,
        'm_c': m_qu_charm,
        'm_b': m_qu_bottom,
        'm_t': m_qu_top,

        # Массы бозонов
        'm_W': m_W,
        'm_Z': m_Z,
        'm_H': m_Higgs,
        'v_Higgs': v_Higgs,

        # Космология
        'Λ': Lambda_val,
        'κ': kappa_val,

        # Атомная физика
        'R∞': Rydberg,
        'a₀': bor_radius,
        'E_Ry': E_rydberg,

        # Безразмерные отношения
        'm_W/m_Z': mW_mZ,
        'm_H/m_W': mH_mW,
        'm_p/m_e': mp_me,
        'm_μ/m_e': m_muon_me,

        # λ-фактор
        'λ': lambda_factor,
        'd_eff': d_eff,
    }


# ========== ВЫЧИСЛЕНИЕ ДЛЯ ВСЕХ N ==========
print("=" * 90)
print("ЭВОЛЮЦИЯ ЭМЕРДЖЕНТНЫХ КОНСТАНТ ПРИ РОСТЕ N")
print("Расширенная версия: фундаментальные константы + массы + заряды")
print("=" * 90)
print(f"K = {K}, lnK = {lnK:.6f}")
print(f"Диапазон N: 10^10 ... 10^122")
print(f"Шаг: 1 порядок ({len(N_values)} значений)")

# Вычисляем
data = [compute_constants(N) for N in N_values]

# Текущие значения
N0 = 4.198e121
data_current = compute_constants(N0)

# CODATA
c_exp = 299792458
hbar_exp = 1.054571817e-34
G_exp = 6.67430e-11
alpha_exp = 1 / 137.035999084
kB_exp = 1.380649e-23
qe_exp = 1.602176634e-19
ep0_exp = 8.8541878128e-12
mu0_exp = 1.25663706127e-6
me_exp = 9.1093837015e-31
m_muon_exp = 1.883531627e-28
m_tau_exp = 3.167e-27
m_proton_exp = 1.67262192e-27
m_W_exp = 1.43362e-25
m_Z_exp = 1.62614e-25
m_Higgs_exp = 2.23319e-25
m_top_exp = 3.04e-25
m_bottom_exp = 4.180e-27
m_charm_exp = 1.27e-27
m_strange_exp = 9.635e-30
m_up_exp = 2.1650e-30
m_down_exp = 4.7915e-30
lP_exp = 1.616255e-35
tP_exp = 5.391247e-44
mP_exp = 2.176434e-8
EP_exp = 1.956082e9
TP_exp = 1.416784e32
Lambda_exp = 1.08929e-52
kappa_exp = 2.07664746e-43
Rydberg_exp = 1.097373e7
bor_radius_exp = 5.29177210903e-11
v_Higgs_exp = 4.388471e-25

# ========== ТАБЛИЦА ПО КЛЮЧЕВЫМ N ==========
print("\n" + "=" * 90)
print("ТАБЛИЦА: КЛЮЧЕВЫЕ ЭПОХИ")
print("=" * 90)

key_powers = [10, 30, 50, 80, 100, 110, 115, 118, 120, 121, 122]
header = (f"{'N':>10} {'F(N)':>8} {'α':>8} {'ħ':>12} {'c':>12} "
          f"{'m_e (кг)':>14} {'q_e (Кл)':>14} {'l_P (м)':>12} {'λ':>8} {'d_eff':>7}")
print(header)
print("-" * len(header))

for p in key_powers:
    if p <= powers[-1]:
        idx = p - powers[0]
        if idx >= 0 and idx < len(data):
            d = data[idx]
            print(f"{10.0 ** p:>10.0e} {d['F(N)']:>8.5f} {d['α']:>8.5f} "
                  f"{d['ħ']:>12.4e} {d['c']:>12.4e} "
                  f"{d['m_e']:>14.6e} {d['q_e']:>14.6e} "
                  f"{d['l_P']:>12.4e} {d['λ']:>8.5f} {d['d_eff']:>7.4f}")

# ========== ТЕКУЩИЕ ЗНАЧЕНИЯ ==========
print("\n" + "=" * 90)
print(f"ТЕКУЩИЕ ЗНАЧЕНИЯ (N = {N0:.4e})")
print("=" * 90)

comparisons = [
    # Фундаментальные
    ("ħ (Дж·с)", data_current['ħ'], hbar_exp),
    ("c (м/с)", data_current['c'], c_exp),
    ("G (м³/(кг·с²))", data_current['G'], G_exp),
    ("α", data_current['α'], alpha_exp),
    ("k_B (Дж/К)", data_current['k_B'], kB_exp),
    # Электродинамика
    ("q_e (Кл)", data_current['q_e'], qe_exp),
    ("ε₀ (Ф/м)", data_current['ε₀'], ep0_exp),
    ("μ₀ (Н/А²)", data_current['μ₀'], mu0_exp),
    # Планковские
    ("l_P (м)", data_current['l_P'], lP_exp),
    ("t_P (с)", data_current['t_P'], tP_exp),
    ("m_P (кг)", data_current['m_P'], mP_exp),
    ("E_P (Дж)", data_current['E_P'], EP_exp),
    ("T_P (К)", data_current['T_P'], TP_exp),
    # Массы фермионов
    ("m_e (кг)", data_current['m_e'], me_exp),
    ("m_μ (кг)", data_current['m_μ'], m_muon_exp),
    ("m_τ (кг)", data_current['m_τ'], m_tau_exp),
    ("m_p (кг)", data_current['m_p'], m_proton_exp),
    ("m_u (кг)", data_current['m_u'], m_up_exp),
    ("m_d (кг)", data_current['m_d'], m_down_exp),
    ("m_s (кг)", data_current['m_s'], m_strange_exp),
    ("m_c (кг)", data_current['m_c'], m_charm_exp),
    ("m_b (кг)", data_current['m_b'], m_bottom_exp),
    ("m_t (кг)", data_current['m_t'], m_top_exp),
    # Массы бозонов
    ("m_W (кг)", data_current['m_W'], m_W_exp),
    ("m_Z (кг)", data_current['m_Z'], m_Z_exp),
    ("m_H (кг)", data_current['m_H'], m_Higgs_exp),
    ("v_Higgs (Дж)", data_current['v_Higgs'], v_Higgs_exp),
    # Космология
    ("Λ (м⁻²)", data_current['Λ'], Lambda_exp),
    ("κ (с²/(кг·м))", data_current['κ'], kappa_exp),
    # Атомная физика
    ("R∞ (м⁻¹)", data_current['R∞'], Rydberg_exp),
    ("a₀ (м)", data_current['a₀'], bor_radius_exp),
]

print(f"\n{'Константа':<18} {'ЕТИ':<18} {'CODATA':<18} {'Ошибка %':<12}")
print("-" * 65)
for name, eti_val, exp_val in comparisons:
    err = abs(eti_val / exp_val - 1) * 100
    status = "✅" if err < 0.1 else ("⭐" if err < 0.5 else ("🟡" if err < 1.0 else "⚠️"))
    print(f"{status} {name:<16} {eti_val:<18.6e} {exp_val:<18.6e} {err:<12.6f}")

# ========== ПРОИЗВОДНЫЕ ==========
print("\n" + "=" * 90)
print("ПРОИЗВОДНЫЕ ВЕЛИЧИНЫ")
print("=" * 90)

print(f"  F(N) = {data_current['F(N)']:.10f}")
print(f"  Цель (1/π) = {1 / pi:.10f}")
print(f"  Ошибка F(N): {data_current['F_error_%']:.6f}%")
print(f"  λ = {data_current['λ']:.6f}")
print(f"  d_eff = {data_current['d_eff']:.4f}")
print(f"  m_W/m_Z = {data_current['m_W/m_Z']:.6f}")
print(f"  m_H/m_W = {data_current['m_H/m_W']:.6f}")
print(f"  m_p/m_e = {data_current['m_p/m_e']:.4f}")
print(f"  m_μ/m_e = {data_current['m_μ/m_e']:.4f}")
print(f"  E_Ry (Дж) = {data_current['E_Ry']:.6e}")

# ========== АСИМПТОТИКИ ==========
print("\n" + "=" * 90)
print("АСИМПТОТИЧЕСКИЕ ПРЕДЕЛЫ (N → ∞)")
print("=" * 90)

print(f"""
  {'Величина':<20} {'При N=10^10':<16} {'При N=10^50':<16} {'При N=10^100':<16} {'При N_opt':<16} {'Предел N→∞':<16}
  {'-' * 90}
  {'F(N)':<20} {data[0]['F(N)']:<16.6f} {data[40]['F(N)']:<16.6f} {data[90]['F(N)']:<16.6f} {compute_constants(N_opt := math.exp((K - lnK) / (1 / 3 - 1 / pi)))['F(N)']:<16.6f} {'1/3 = 0.33333':<16}
  {'α':<20} {data[0]['α']:<16.6f} {data[40]['α']:<16.6f} {data[90]['α']:<16.6f} {compute_constants(N_opt)['α']:<16.6f} {'→ 0':<16}
  {'m_e (кг)':<20} {data[0]['m_e']:<16.4e} {data[40]['m_e']:<16.4e} {data[90]['m_e']:<16.4e} {compute_constants(N_opt)['m_e']:<16.4e} {'→ 0':<16}
  {'q_e (Кл)':<20} {data[0]['q_e']:<16.4e} {data[40]['q_e']:<16.4e} {data[90]['q_e']:<16.4e} {compute_constants(N_opt)['q_e']:<16.4e} {'→ 0':<16}
  {'l_P (м)':<20} {data[0]['l_P']:<16.4e} {data[40]['l_P']:<16.4e} {data[90]['l_P']:<16.4e} {compute_constants(N_opt)['l_P']:<16.4e} {'→ 0':<16}
  {'E_P (Дж)':<20} {data[0]['E_P']:<16.4e} {data[40]['E_P']:<16.4e} {data[90]['E_P']:<16.4e} {compute_constants(N_opt)['E_P']:<16.4e} {'→ ∞':<16}
  {'λ':<20} {data[0]['λ']:<16.6f} {data[40]['λ']:<16.6f} {data[90]['λ']:<16.6f} {compute_constants(N_opt)['λ']:<16.6f} {'→ 1':<16}
  {'d_eff':<20} {data[0]['d_eff']:<16.4f} {data[40]['d_eff']:<16.4f} {data[90]['d_eff']:<16.4f} {compute_constants(N_opt)['d_eff']:<16.4f} {'→ 3':<16}
""")

# ========== ГЕОМЕТРИЧЕСКИЙ РЕЗОНАНС ==========
print("=" * 90)
print("ГЕОМЕТРИЧЕСКИЙ РЕЗОНАНС")
print("=" * 90)

N_opt = math.exp((K - lnK) / (1 / 3 - 1 / pi))
print(f"  ln N_opt = (K - lnK) / (1/3 - 1/π) = {math.log(N_opt):.4f}")
print(f"  N_opt = {N_opt:.4e}")
print(f"  При N_opt: F(N) = {compute_constants(N_opt)['F(N)']:.10f}")
print(f"  Ошибка: {compute_constants(N_opt)['F_error_%']:.6f}%")
print(f"  λ при N_opt = {compute_constants(N_opt)['λ']:.6f}")
print(f"  d_eff при N_opt = {compute_constants(N_opt)['d_eff']:.4f}")

# ========== λ=1 ==========
print("\n" + "=" * 90)
print("ТОЧКА λ=1 (ПОЛНАЯ ПЛАНКОВСКАЯ САМОСОГЛАСОВАННОСТЬ)")
print("=" * 90)

lnN_l1 = 4 * K ** 3 * lnK ** 2 / pi ** 2
N_l1 = math.exp(lnN_l1)
print(f"  ln N = 4·K³·(lnK)²/π² = {lnN_l1:.4f}")
print(f"  N = {N_l1:.4e}")
print(f"  Отклонение от N_opt: {abs(N_l1 / N_opt - 1) * 100:.4f}%")

# ========== ЭВОЛЮЦИЯ λ ==========
print("\n" + "=" * 90)
print("ЭВОЛЮЦИЯ λ И d_eff")
print("=" * 90)

for p_label, p_val in [("10^10", 10), ("10^30", 30), ("10^50", 50),
                       ("10^80", 80), ("10^100", 100), ("10^121", 121), ("N_opt", None)]:
    if p_val is not None:
        idx = p_val - powers[0]
        d = data[idx]
        label = f"N = 10^{p_val}"
    else:
        d = compute_constants(N_opt)
        label = f"N = N_opt"
    print(f"  {label:<15} λ = {d['λ']:.6f}, d_eff = {d['d_eff']:.4f}")

# ========== ГРАФИКИ ==========
try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle('Эволюция эмерджентных констант при росте N', fontsize=16, fontweight='bold')

    powers_arr = np.array(powers)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    # 1. Функция геометрического резонанса
    ax = axes[0, 0]
    F_vals = [d['F(N)'] for d in data]
    ax.plot(powers_arr, F_vals, color=colors[0], linewidth=1.5)
    ax.axhline(y=1 / pi, color='red', linestyle='--', linewidth=1, label=f'1/π = {1 / pi:.5f}')
    ax.axhline(y=1 / 3, color='green', linestyle='--', linewidth=1, label='1/3 (предел)')
    ax.axvline(x=math.log10(N_opt), color='orange', linestyle=':', linewidth=1.5,
               label=f'N_opt ≈ 10^{math.log10(N_opt):.1f}')
    ax.set_xlabel('log₁₀(N)', fontsize=10)
    ax.set_ylabel('F(N)', fontsize=10)
    ax.set_title('Функция геометрического резонанса', fontsize=11, fontweight='bold')
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(True, alpha=0.3)

    # 2. Ошибка F(N)
    ax = axes[0, 1]
    F_err = [d['F_error_%'] for d in data]
    ax.semilogy(powers_arr, F_err, color=colors[3], linewidth=1.5)
    ax.axvline(x=math.log10(N_opt), color='orange', linestyle=':', linewidth=1.5, label=f'N_opt')
    ax.fill_between(powers_arr, 1e-8, F_err, alpha=0.2, color=colors[3])
    ax.set_xlabel('log₁₀(N)', fontsize=10)
    ax.set_ylabel('Ошибка (%)', fontsize=10)
    ax.set_title('Ошибка геометрического резонанса', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. Постоянная тонкой структуры
    ax = axes[0, 2]
    alpha_vals = [d['α'] for d in data]
    ax.plot(powers_arr, alpha_vals, color=colors[4], linewidth=1.5)
    ax.axhline(y=1 / 137.036, color='red', linestyle='--', linewidth=1, label='CODATA ≈ 1/137')
    ax.set_xlabel('log₁₀(N)', fontsize=10)
    ax.set_ylabel('α', fontsize=10)
    ax.set_title('Постоянная тонкой структуры α', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 4. Фундаментальные константы (нормировка)
    ax = axes[1, 0]
    ref_idx = -1  # нормировка на последнее (текущее) значение
    hbar_norm = np.array([d['ħ'] for d in data]) / data[ref_idx]['ħ']
    c_norm = np.array([d['c'] for d in data]) / data[ref_idx]['c']
    G_norm = np.array([d['G'] for d in data]) / data[ref_idx]['G']

    ax.plot(powers_arr, hbar_norm, color=colors[0], label='ħ/ħ₀', linewidth=1.5)
    ax.plot(powers_arr, c_norm, color=colors[1], label='c/c₀', linewidth=1.5)
    ax.plot(powers_arr, G_norm, color=colors[2], label='G/G₀', linewidth=1.5)
    ax.set_xlabel('log₁₀(N)', fontsize=10)
    ax.set_ylabel('Относительное значение', fontsize=10)
    ax.set_title('Фундаментальные константы', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 5. Масса электрона и протона
    ax = axes[1, 1]
    me_norm = np.array([d['m_e'] for d in data]) / data[ref_idx]['m_e']
    mp_norm = np.array([d['m_p'] for d in data]) / data[ref_idx]['m_p']
    ax.plot(powers_arr, me_norm, color=colors[0], label='m_e/m_e₀', linewidth=1.5)
    ax.plot(powers_arr, mp_norm, color=colors[1], label='m_p/m_p₀', linewidth=1.5)
    ax.set_xlabel('log₁₀(N)', fontsize=10)
    ax.set_ylabel('Относительное значение', fontsize=10)
    ax.set_title('Массы электрона и протона', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 6. Массы бозонов
    ax = axes[1, 2]
    mW_norm = np.array([d['m_W'] for d in data]) / data[ref_idx]['m_W']
    mZ_norm = np.array([d['m_Z'] for d in data]) / data[ref_idx]['m_Z']
    mH_norm = np.array([d['m_H'] for d in data]) / data[ref_idx]['m_H']
    ax.plot(powers_arr, mW_norm, color=colors[5], label='m_W/m_W₀', linewidth=1.5)
    ax.plot(powers_arr, mZ_norm, color=colors[6], label='m_Z/m_Z₀', linewidth=1.5)
    ax.plot(powers_arr, mH_norm, color=colors[7], label='m_H/m_H₀', linewidth=1.5)
    ax.set_xlabel('log₁₀(N)', fontsize=10)
    ax.set_ylabel('Относительное значение', fontsize=10)
    ax.set_title('Массы W, Z, Хиггс', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 7. λ и эффективная размерность
    ax = axes[2, 0]
    lambda_vals = [d['λ'] for d in data]
    d_eff_vals = [d['d_eff'] for d in data]

    ax2 = ax.twinx()
    ax.plot(powers_arr, lambda_vals, color=colors[0], linewidth=1.5, label='λ')
    ax2.plot(powers_arr, d_eff_vals, color=colors[3], linestyle='--', linewidth=1.5, label='d_eff')
    ax.axhline(y=1.0, color='green', linestyle=':', linewidth=1, label='λ=1')
    ax2.axhline(y=3.0, color='orange', linestyle=':', linewidth=1, label='d_eff=3')
    ax.set_xlabel('log₁₀(N)', fontsize=10)
    ax.set_ylabel('λ', color=colors[0], fontsize=10)
    ax2.set_ylabel('d_eff', color=colors[3], fontsize=10)
    ax.set_title('λ-фактор и эффективная размерность', fontsize=11, fontweight='bold')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc='upper left')
    ax.grid(True, alpha=0.3)

    # 8. Планковские величины (логарифмический масштаб)
    ax = axes[2, 1]
    lP_norm = np.array([d['l_P'] for d in data]) / data[ref_idx]['l_P']
    tP_norm = np.array([d['t_P'] for d in data]) / data[ref_idx]['t_P']
    EP_norm = np.array([d['E_P'] for d in data]) / data[ref_idx]['E_P']

    ax.loglog(powers_arr, lP_norm, color=colors[0], label='l_P/l_P₀', linewidth=1.5)
    ax.loglog(powers_arr, tP_norm, color=colors[1], label='t_P/t_P₀', linewidth=1.5)
    ax.loglog(powers_arr, EP_norm, color=colors[2], label='E_P/E_P₀', linewidth=1.5)
    ax.set_xlabel('log₁₀(N)', fontsize=10)
    ax.set_ylabel('Относительное значение', fontsize=10)
    ax.set_title('Планковские величины (log-log)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 9. Безразмерные отношения
    ax = axes[2, 2]
    mW_mZ_vals = [d['m_W/m_Z'] for d in data]
    mH_mW_vals = [d['m_H/m_W'] for d in data]
    mp_me_vals = [d['m_p/m_e'] for d in data]

    ax.plot(powers_arr, mp_me_vals, color=colors[0], label='m_p/m_e', linewidth=1.5)
    ax.plot(powers_arr, mW_mZ_vals, color=colors[5], label='m_W/m_Z', linewidth=1.5)
    ax.plot(powers_arr, mH_mW_vals, color=colors[6], label='m_H/m_W', linewidth=1.5)
    ax.axhline(y=pi ** 0.5 / 2, color='red', linestyle=':', linewidth=1, label=f'√π/2')
    ax.set_xlabel('log₁₀(N)', fontsize=10)
    ax.set_ylabel('Значение', fontsize=10)
    ax.set_title('Безразмерные отношения', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('evolution_of_constants_extended.png', dpi=150, bbox_inches='tight')
    print("\n[График сохранён в 'evolution_of_constants_extended.png']")

except ImportError:
    print("\n[Matplotlib не установлен — график не построен]")

print("\n" + "=" * 90)
print("ИССЛЕДОВАНИЕ ЗАВЕРШЕНО")
print("=" * 90)