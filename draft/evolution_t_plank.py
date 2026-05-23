"""
ИССЛЕДОВАНИЕ ЭВОЛЮЦИИ ЭМЕРДЖЕНТНЫХ КОНСТАНТ ПРИ РОСТЕ ln N
От ln N = 23 до ln N = 281 с шагом ~2.3 (100 точек)
Параметр эволюции: ВРЕМЯ = t_P (планковское время из модели)
t_P = 4 * K² * (lnK)² / (π * (ln N)² * N^(1/3))

ДОПОЛНИТЕЛЬНЫЙ ГРАФИК: Все константы для N от 10 до 10^150
"""

import math
import numpy as np

# ========== КОНСТАНТЫ ==========
K = 6.0
pi = math.pi
lnK = math.log(K)

# ========== ДИАПАЗОН ln N ==========
lnN_min = 2.0
lnN_max = 281.0  # примерно 10^122
n_points = 200
lnN_values = np.linspace(lnN_min, lnN_max, n_points)
N_values = np.exp(lnN_values)

# ========== НОВЫЙ ДИАПАЗОН: N от 10 до 10^150 ==========
N_new_min = 1e1
N_new_max = 1e150
n_new_points = 151  # по точке на каждый порядок
N_new_values = np.logspace(math.log10(N_new_min), math.log10(N_new_max), n_new_points)
lnN_new_values = np.log(N_new_values)

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

    # Планковские величины
    lP_val = 4 * (lnN ** 2) * lnK / N13
    tP_val = 4 * K ** 2 * lnK ** 2 / (pi * (lnN ** 2) * N13)  # ПАРАМЕТР ЭВОЛЮЦИИ
    mP_val = K / (4 * pi * lnN ** 3)
    EP_val = pi * (lnN ** 5) / (4 * K ** 3 * lnK ** 2)
    TP_val = 2 * pi ** 3 * N13 / (K ** 3 * lnK ** 2 * lnN ** 3)

    # Фундаментальные
    hbar_val = (lnN ** 3) / (K * N13)
    c_val = pi * (lnN ** 4) / (K ** 2 * lnK)
    G_val = 16 * pi ** 3 * lnN ** 13 / (K ** 5 * lnK * N13)
    alpha_val = 2 * lnK ** 2 / (pi * lnN)
    kB_val = Kp * (lnN ** 8) / (8 * pi ** 2)

    # Электродинамика
    qe_val = 1.0 / (pi * K ** (3 / 2) * lnN ** 7)
    ep0_val = N13 / (8 * pi ** 3 * lnK * lnN ** 20)
    mu0_val = (8 * pi * K ** 4 * lnK ** 3 * lnN ** 12) / N13

    # Массы
    me_val = 4 * pi * lnN ** 4 / (sqrtK * N13)
    m_muon = 4 * pi ** 2 * lnN ** 5 / (K * sqrt3 * N13)
    m_tau = sqrtPi * lnN ** 5 * K ** 2 / N13
    m_proton = sqrtPi * lnN ** 6 / (K ** (3 / 2) * N13)

    # Бозоны
    m_W = 2 * pi ** 3 * lnN ** 6 / (N13 * K)
    m_Z = lnN ** 6 * 4 * pi ** (5 / 2) / (N13 * K)
    m_Higgs = lnN ** 6 * 4 * pi ** 2 / (N13 * sqrtK)

    # Кварки
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

    # Атомная физика
    Rydberg = 4 * lnN ** 3 * lnK ** 3 / (pi * K ** (3 / 2))
    bor_radius = K ** (3 / 2) / (8 * pi * lnN ** 4 * lnK)
    E_rydberg = 8 * pi * lnK ** 2 * lnN ** 10 / (K ** (9 / 2) * N13)

    # Безразмерные отношения
    mW_mZ = sqrtPi / 2
    mH_mW = 2 * sqrtK / pi
    mp_me = lnN ** 2 / (4 * sqrtPi * K)
    m_muon_me = pi * sqrtK * lnN / sqrt3

    # λ-фактор
    lambda_factor = pi ** 2 * lnN / (4 * K ** 3 * lnK ** 2)
    d_eff = 3 * lambda_factor

    # Функция геометрического резонанса
    F_val = (K + math.log(p_val)) / (p_val - lnN)
    F_error = abs(F_val - 1 / pi) / (1 / pi) * 100

    return {
        # Базовые
        'N': N,
        'lnN': lnN,
        'p': p_val,
        'F(N)': F_val,
        'F_error_%': F_error,
        't_P': tP_val,  # ПАРАМЕТР ЭВОЛЮЦИИ

        # Фундаментальные
        'ħ': hbar_val, 'c': c_val, 'G': G_val, 'α': alpha_val, 'k_B': kB_val,

        # Электродинамика
        'q_e': qe_val, 'ε₀': ep0_val, 'μ₀': mu0_val,

        # Планковские
        'l_P': lP_val, 'm_P': mP_val, 'E_P': EP_val, 'T_P': TP_val,

        # Массы
        'm_e': me_val, 'm_μ': m_muon, 'm_τ': m_tau, 'm_p': m_proton,
        'm_u': m_qu_up, 'm_d': m_qu_down, 'm_s': m_qu_strange,
        'm_c': m_qu_charm, 'm_b': m_qu_bottom, 'm_t': m_qu_top,

        # Бозоны
        'm_W': m_W, 'm_Z': m_Z, 'm_H': m_Higgs, 'v_Higgs': v_Higgs,

        # Космология
        'Λ': Lambda_val, 'κ': kappa_val,

        # Атомная
        'R∞': Rydberg, 'a₀': bor_radius, 'E_Ry': E_rydberg,

        # Безразмерные
        'm_W/m_Z': mW_mZ, 'm_H/m_W': mH_mW, 'm_p/m_e': mp_me, 'm_μ/m_e': m_muon_me,

        # λ-фактор
        'λ': lambda_factor, 'd_eff': d_eff,
    }


# ========== ВЫЧИСЛЕНИЯ ДЛЯ ОСНОВНОГО ДИАПАЗОНА ==========
print("ЭВОЛЮЦИЯ ЭМЕРДЖЕНТНЫХ КОНСТАНТ")
print("Параметр эволюции: t_P (планковское время из модели)")
print(f"t_P = 4·K²·(lnK)² / (π·(ln N)²·N^(1/3))")
print(f"K = {K}, lnK = {lnK:.6f}")
print(f"Диапазон ln N: {lnN_min:.1f} ... {lnN_max:.1f}")
print(f"Точек: {n_points}")

# Вычисляем для основного диапазона
data = [compute_constants(N) for N in N_values]

# Вычисляем для нового диапазона
print("\nВычисление данных для графика 'Все константы'...")
data_new = [compute_constants(N) for N in N_new_values]
print(f"Вычислено {len(data_new)} точек для N от 10^1 до 10^150")

# Текущие значения
N0 = 4.198e121
lnN0 = math.log(N0)
data_current = compute_constants(N0)

# CODATA (те же)
c_exp = 299792458
hbar_exp = 1.054571817e-34
G_exp = 6.67430e-11
alpha_exp = 1 / 137.035999084
kB_exp = 1.380649e-23
qe_exp = 1.602176634e-19
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

# ========== ТАБЛИЦА ПО КЛЮЧЕВЫМ ln N ==========
print("\n" + "=" * 90)
print("ТАБЛИЦА: КЛЮЧЕВЫЕ ЭПОХИ (по ln N и t_P)")
print("=" * 90)

key_lnN = [1, 4, 8, 12, 15, 23, 50, 100, 150, 200, 230, 250, 260, 270, 275, 280, 281]
header = (f"{'ln N':>8} {'t_P (с)':>12} {'F(N)':>8} {'α':>8} "
          f"{'ħ':>12} {'c':>12} {'m_e (кг)':>14} {'l_P (м)':>12} {'λ':>8} {'d_eff':>7}")
print(header)
print("-" * len(header))

for target_lnN in key_lnN:
    idx = np.argmin(np.abs(lnN_values - target_lnN))
    d = data[idx]
    print(f"{d['lnN']:>8.1f} {d['t_P']:>12.4e} {d['F(N)']:>8.5f} {d['α']:>8.5f} "
          f"{d['ħ']:>12.4e} {d['c']:>12.4e} "
          f"{d['m_e']:>14.6e} {d['l_P']:>12.4e} {d['λ']:>8.5f} {d['d_eff']:>7.4f}")

# ========== ТЕКУЩИЕ ЗНАЧЕНИЯ ==========
print(f"ТЕКУЩИЕ ЗНАЧЕНИЯ (ln N = {lnN0:.4f}, N = {N0:.4e})")
print(f"Планковское время t_P = {data_current['t_P']:.4e} с")

comparisons = [
    ("ħ (Дж·с)", data_current['ħ'], hbar_exp),
    ("c (м/с)", data_current['c'], c_exp),
    ("G (м³/(кг·с²))", data_current['G'], G_exp),
    ("α", data_current['α'], alpha_exp),
    ("k_B (Дж/К)", data_current['k_B'], kB_exp),
    ("q_e (Кл)", data_current['q_e'], qe_exp),
    ("l_P (м)", data_current['l_P'], lP_exp),
    ("t_P (с)", data_current['t_P'], tP_exp),
    ("m_P (кг)", data_current['m_P'], mP_exp),
    ("E_P (Дж)", data_current['E_P'], EP_exp),
    ("T_P (К)", data_current['T_P'], TP_exp),
    ("m_e (кг)", data_current['m_e'], me_exp),
    ("m_μ (кг)", data_current['m_μ'], m_muon_exp),
    ("m_τ (кг)", data_current['m_τ'], m_tau_exp),
    ("m_p (кг)", data_current['m_p'], m_proton_exp),
    ("m_W (кг)", data_current['m_W'], m_W_exp),
    ("m_Z (кг)", data_current['m_Z'], m_Z_exp),
    ("m_H (кг)", data_current['m_H'], m_Higgs_exp),
    ("Λ (м⁻²)", data_current['Λ'], Lambda_exp),
    ("κ (с²/(кг·м))", data_current['κ'], kappa_exp),
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
print("ПРОИЗВОДНЫЕ ВЕЛИЧИНЫ")
print(f"  ln N = {lnN0:.4f}")
print(f"  N = {N0:.4e}")
print(f"  F(N) = {data_current['F(N)']:.10f}")
print(f"  Цель (1/π) = {1 / pi:.10f}")
print(f"  Ошибка F(N): {data_current['F_error_%']:.6f}%")
print(f"  λ = {data_current['λ']:.6f}")
print(f"  d_eff = {data_current['d_eff']:.4f}")
print(f"  m_W/m_Z = {data_current['m_W/m_Z']:.6f}")
print(f"  m_H/m_W = {data_current['m_H/m_W']:.6f}")
print(f"  m_p/m_e = {data_current['m_p/m_e']:.4f}")
print(f"  m_μ/m_e = {data_current['m_μ/m_e']:.4f}")

# ========== ГЕОМЕТРИЧЕСКИЙ РЕЗОНАНС ==========
print("ГЕОМЕТРИЧЕСКИЙ РЕЗОНАНС")

lnN_opt = (K - lnK) / (1 / 3 - 1 / pi)
N_opt = math.exp(lnN_opt)
d_opt = compute_constants(N_opt)
print(f"  ln N_opt = (K - lnK) / (1/3 - 1/π) = {lnN_opt:.4f}")
print(f"  N_opt = {N_opt:.4e}")
print(f"  При ln N_opt: F(N) = {d_opt['F(N)']:.10f}")
print(f"  Ошибка: {d_opt['F_error_%']:.6f}%")
print(f"  λ = {d_opt['λ']:.6f}, d_eff = {d_opt['d_eff']:.4f}")
print(f"  t_P при резонансе = {d_opt['t_P']:.4e} с")

# ========== λ=1 ==========
print("ТОЧКА λ=1 (ПОЛНАЯ ПЛАНКОВСКАЯ САМОСОГЛАСОВАННОСТЬ)")

lnN_l1 = 4 * K ** 3 * lnK ** 2 / pi ** 2
N_l1 = math.exp(lnN_l1)
d_l1 = compute_constants(N_l1)
print(f"  ln N = 4·K³·(lnK)²/π² = {lnN_l1:.4f}")
print(f"  N = {N_l1:.4e}")
print(f"  t_P при λ=1 = {d_l1['t_P']:.4e} с")

# ========== ПРОИЗВОДНЫЕ ПО ln N ==========
print("ПРОИЗВОДНЫЕ ПО ln N (СКОРОСТЬ ИЗМЕНЕНИЯ)")

# Берём две близкие точки для численной производной
idx_current = np.argmin(np.abs(lnN_values - lnN0))
if idx_current < len(data) - 1:
    d1 = data[idx_current]
    d2 = data[idx_current + 1]
    dlnN = d2['lnN'] - d1['lnN']

    print(f"\n  d(ln константы) / d(ln N) при ln N ≈ {lnN0:.1f}:")

    for name, key in [("ħ", 'ħ'), ("c", 'c'), ("G", 'G'), ("α", 'α'),
                      ("m_e", 'm_e'), ("m_p", 'm_p'), ("l_P", 'l_P'),
                      ("t_P", 't_P'), ("E_P", 'E_P'), ("Λ", 'Λ')]:
        deriv = (math.log(d2[key]) - math.log(d1[key])) / dlnN
        print(f"    d(ln {name})/d(ln N) = {deriv:+.6f}")

# ========== ГРАФИКИ ==========
try:
    import matplotlib.pyplot as plt

    # ========== ОСНОВНЫЕ ГРАФИКИ ЭВОЛЮЦИИ ==========
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle('Эволюция эмерджентных констант как функция времени t_P', fontsize=16, fontweight='bold')

    # Извлекаем массив времён
    tP_vals = np.array([d['t_P'] for d in data])
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    # 1. Функция геометрического резонанса
    ax = axes[0, 0]
    F_vals = [d['F(N)'] for d in data]
    ax.semilogx(tP_vals, F_vals, color=colors[0], linewidth=1.5)
    ax.axhline(y=1 / pi, color='red', linestyle='--', linewidth=1, label=f'1/π = {1 / pi:.5f}')
    ax.axhline(y=1 / 3, color='green', linestyle='--', linewidth=1, label='1/3 (предел)')
    ax.set_xlabel('t_P (с)', fontsize=10)
    ax.set_ylabel('F(N)', fontsize=10)
    ax.set_title('Функция геометрического резонанса F(t_P)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(min(tP_vals), max(tP_vals))

    # 2. Ошибка F(N)
    ax = axes[0, 1]
    F_err = [d['F_error_%'] for d in data]
    ax.semilogx(tP_vals, F_err, color=colors[3], linewidth=1.5)
    ax.set_xlabel('t_P (с)', fontsize=10)
    ax.set_ylabel('Ошибка (%)', fontsize=10)
    ax.set_title('Ошибка геометрического резонанса', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 3. Постоянная тонкой структуры
    ax = axes[0, 2]
    alpha_vals = [d['α'] for d in data]
    ax.semilogx(tP_vals, alpha_vals, color=colors[4], linewidth=1.5)
    ax.axhline(y=1 / 137.036, color='red', linestyle='--', linewidth=1, label='CODATA ≈ 1/137')
    ax.set_xlabel('t_P (с)', fontsize=10)
    ax.set_ylabel('α', fontsize=10)
    ax.set_title('Постоянная тонкой структуры α(t_P)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 4. Фундаментальные константы (логарифмический масштаб)
    ax = axes[1, 0]
    hbar_vals = np.array([d['ħ'] for d in data])
    c_vals = np.array([d['c'] for d in data])
    G_vals = np.array([d['G'] for d in data])

    ax.loglog(tP_vals, hbar_vals, color=colors[0], label='ħ (Дж·с)', linewidth=1.5)
    ax.loglog(tP_vals, G_vals, color=colors[2], label='G (м³/(кг·с²))', linewidth=1.5)
    ax2 = ax.twinx()
    ax2.semilogx(tP_vals, c_vals, color=colors[1], label='c (м/с)', linewidth=1.5)

    ax.set_xlabel('t_P (с)', fontsize=10)
    ax.set_ylabel('ħ, G', fontsize=10)
    ax2.set_ylabel('c (м/с)', fontsize=10)
    ax.set_title('Фундаментальные константы', fontsize=11, fontweight='bold')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7)
    ax.grid(True, alpha=0.3)

    # 5. Массы частиц
    ax = axes[1, 1]
    me_vals = np.array([d['m_e'] for d in data])
    mp_vals = np.array([d['m_p'] for d in data])
    mW_vals = np.array([d['m_W'] for d in data])

    ax.loglog(tP_vals, me_vals, color=colors[0], label='m_e', linewidth=1.5)
    ax.loglog(tP_vals, mp_vals, color=colors[1], label='m_p', linewidth=1.5)
    ax.loglog(tP_vals, mW_vals, color=colors[5], label='m_W', linewidth=1.5)
    ax.set_xlabel('t_P (с)', fontsize=10)
    ax.set_ylabel('Масса (кг)', fontsize=10)
    ax.set_title('Массы частиц', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 6. Планковские величины
    ax = axes[1, 2]
    lP_vals = np.array([d['l_P'] for d in data])
    EP_vals = np.array([d['E_P'] for d in data])
    mP_vals = np.array([d['m_P'] for d in data])

    ax.loglog(tP_vals, lP_vals, color=colors[0], label='l_P (м)', linewidth=1.5)
    ax.loglog(tP_vals, EP_vals, color=colors[1], label='E_P (Дж)', linewidth=1.5)
    ax.loglog(tP_vals, mP_vals, color=colors[2], label='m_P (кг)', linewidth=1.5)
    ax.set_xlabel('t_P (с)', fontsize=10)
    ax.set_ylabel('Значение', fontsize=10)
    ax.set_title('Планковские величины', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 7. λ и эффективная размерность
    ax = axes[2, 0]
    lambda_vals = [d['λ'] for d in data]
    d_eff_vals = [d['d_eff'] for d in data]

    ax2 = ax.twinx()
    ax.semilogx(tP_vals, lambda_vals, color=colors[0], linewidth=1.5, label='λ')
    ax2.semilogx(tP_vals, d_eff_vals, color=colors[3], linestyle='--', linewidth=1.5, label='d_eff')
    ax.axhline(y=1.0, color='green', linestyle=':', linewidth=1, label='λ=1')
    ax2.axhline(y=3.0, color='orange', linestyle=':', linewidth=1, label='d_eff=3')
    ax.set_xlabel('t_P (с)', fontsize=10)
    ax.set_ylabel('λ', color=colors[0], fontsize=10)
    ax2.set_ylabel('d_eff', color=colors[3], fontsize=10)
    ax.set_title('λ-фактор и эффективная размерность', fontsize=11, fontweight='bold')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc='upper left')
    ax.grid(True, alpha=0.3)

    # 8. Заряд электрона и космологическая постоянная
    ax = axes[2, 1]
    qe_vals = np.array([d['q_e'] for d in data])
    Lambda_vals = np.array([d['Λ'] for d in data])

    ax.loglog(tP_vals, qe_vals, color=colors[0], label='q_e (Кл)', linewidth=1.5)
    ax.loglog(tP_vals, Lambda_vals, color=colors[3], label='Λ (м⁻²)', linewidth=1.5)
    ax.set_xlabel('t_P (с)', fontsize=10)
    ax.set_ylabel('Значение', fontsize=10)
    ax.set_title('Заряд электрона и Λ', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 9. Безразмерные отношения
    ax = axes[2, 2]
    mp_me_vals = [d['m_p/m_e'] for d in data]
    mW_mZ_vals = [d['m_W/m_Z'] for d in data]
    mH_mW_vals = [d['m_H/m_W'] for d in data]

    ax.semilogx(tP_vals, mp_me_vals, color=colors[0], label='m_p/m_e', linewidth=1.5)
    ax.semilogx(tP_vals, mW_mZ_vals, color=colors[5], label='m_W/m_Z', linewidth=1.5)
    ax.semilogx(tP_vals, mH_mW_vals, color=colors[6], label='m_H/m_W', linewidth=1.5)
    ax.axhline(y=math.sqrt(pi)/2, color='red', linestyle=':', linewidth=1, label=f'√π/2')
    ax.set_xlabel('t_P (с)', fontsize=10)
    ax.set_ylabel('Значение', fontsize=10)
    ax.set_title('Безразмерные отношения', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('evolution_vs_tP.png', dpi=150, bbox_inches='tight')
    print("\n[График сохранён в 'evolution_vs_tP.png']")

    # ========== НОВЫЙ БОЛЬШОЙ ГРАФИК: ВСЕ КОНСТАНТЫ ==========
    print("\nСоздание большого графика 'Все константы'...")

    # Создаем большой график 5x6 для всех констант
    fig_all, axes_all = plt.subplots(5, 6, figsize=(24, 20))
    fig_all.suptitle('Эволюция всех физических констант (N от 10¹ до 10¹⁵⁰)',
                     fontsize=18, fontweight='bold')

    # Извлекаем данные для нового диапазона
    N_new = np.array([d['N'] for d in data_new])
    hbar_new = np.array([d['ħ'] for d in data_new])
    c_new = np.array([d['c'] for d in data_new])
    G_new = np.array([d['G'] for d in data_new])
    alpha_new = np.array([d['α'] for d in data_new])
    kB_new = np.array([d['k_B'] for d in data_new])
    qe_new = np.array([d['q_e'] for d in data_new])
    ep0_new = np.array([d['ε₀'] for d in data_new])
    mu0_new = np.array([d['μ₀'] for d in data_new])
    lP_new = np.array([d['l_P'] for d in data_new])
    tP_new = np.array([d['t_P'] for d in data_new])
    mP_new = np.array([d['m_P'] for d in data_new])
    EP_new = np.array([d['E_P'] for d in data_new])
    TP_new = np.array([d['T_P'] for d in data_new])
    me_new = np.array([d['m_e'] for d in data_new])
    m_muon_new = np.array([d['m_μ'] for d in data_new])
    m_tau_new = np.array([d['m_τ'] for d in data_new])
    m_proton_new = np.array([d['m_p'] for d in data_new])
    m_W_new = np.array([d['m_W'] for d in data_new])
    m_Z_new = np.array([d['m_Z'] for d in data_new])
    m_Higgs_new = np.array([d['m_H'] for d in data_new])
    m_top_new = np.array([d['m_t'] for d in data_new])
    m_bottom_new = np.array([d['m_b'] for d in data_new])
    m_charm_new = np.array([d['m_c'] for d in data_new])
    m_strange_new = np.array([d['m_s'] for d in data_new])
    m_up_new = np.array([d['m_u'] for d in data_new])
    m_down_new = np.array([d['m_d'] for d in data_new])
    Lambda_new = np.array([d['Λ'] for d in data_new])
    kappa_new = np.array([d['κ'] for d in data_new])
    v_Higgs_new = np.array([d['v_Higgs'] for d in data_new])
    Rydberg_new = np.array([d['R∞'] for d in data_new])
    bor_radius_new = np.array([d['a₀'] for d in data_new])
    E_rydberg_new = np.array([d['E_Ry'] for d in data_new])

    # Список всех констант для отображения
    all_constants = [
        # row 1: Фундаментальные
        ('Постоянная Планка ħ (Дж·с)', hbar_new, 'blue', 0, 0),
        ('Скорость света c (м/с)', c_new, 'red', 0, 1),
        ('Гравитационная G (м³/(кг·с²))', G_new, 'green', 0, 2),
        ('Тонкая структура α', alpha_new, 'purple', 0, 3),
        ('Постоянная Больцмана k_B (Дж/К)', kB_new, 'orange', 0, 4),
        ('Заряд электрона q_e (Кл)', qe_new, 'brown', 0, 5),

        # row 2: Планковские
        ('Планковская длина l_P (м)', lP_new, 'blue', 1, 0),
        ('Планковское время t_P (с)', tP_new, 'red', 1, 1),
        ('Планковская масса m_P (кг)', mP_new, 'green', 1, 2),
        ('Планковская энергия E_P (Дж)', EP_new, 'purple', 1, 3),
        ('Планковская температура T_P (К)', TP_new, 'orange', 1, 4),
        ('Электрическая постоянная ε₀', ep0_new, 'brown', 1, 5),

        # row 3: Массы (часть 1)
        ('Масса электрона m_e (кг)', me_new, 'blue', 2, 0),
        ('Масса мюона m_μ (кг)', m_muon_new, 'red', 2, 1),
        ('Масса тау m_τ (кг)', m_tau_new, 'green', 2, 2),
        ('Масса протона m_p (кг)', m_proton_new, 'purple', 2, 3),
        ('Масса W-бозона m_W (кг)', m_W_new, 'orange', 2, 4),
        ('Масса Z-бозона m_Z (кг)', m_Z_new, 'brown', 2, 5),

        # row 4: Массы (часть 2) и бозоны
        ('Масса Хиггса m_H (кг)', m_Higgs_new, 'blue', 3, 0),
        ('Масса t-кварка m_t (кг)', m_top_new, 'red', 3, 1),
        ('Масса b-кварка m_b (кг)', m_bottom_new, 'green', 3, 2),
        ('Масса c-кварка m_c (кг)', m_charm_new, 'purple', 3, 3),
        ('Масса s-кварка m_s (кг)', m_strange_new, 'orange', 3, 4),
        ('Вакуум Хиггса v_H (кг)', v_Higgs_new, 'brown', 3, 5),

        # row 5: Космология и атомная физика
        ('Космологическая Λ (м⁻²)', Lambda_new, 'blue', 4, 0),
        ('Гравитационная κ (с²/(кг·м))', kappa_new, 'red', 4, 1),
        ('Постоянная Ридберга R∞ (м⁻¹)', Rydberg_new, 'green', 4, 2),
        ('Боровский радиус a₀ (м)', bor_radius_new, 'purple', 4, 3),
        ('Энергия Ридберга E_Ry (Дж)', E_rydberg_new, 'orange', 4, 4),
        ('Магнитная постоянная μ₀', mu0_new, 'brown', 4, 5),
    ]

    # Отображаем все константы
    for title, data_arr, color, row, col in all_constants:
        ax = axes_all[row, col]
        ax.loglog(N_new, data_arr, color=color, linewidth=1.5)
        ax.set_xlabel('N', fontsize=8)
        ax.set_ylabel(title.split('(')[0].strip(), fontsize=7)
        ax.set_title(title, fontsize=8, fontweight='bold')
        ax.grid(True, alpha=0.3, which='both')
        ax.tick_params(labelsize=7)

        # Добавляем сетку по порядкам N
        ax.axvline(x=1e10, color='gray', linestyle=':', alpha=0.3)
        ax.axvline(x=1e20, color='gray', linestyle=':', alpha=0.3)
        ax.axvline(x=1e30, color='gray', linestyle=':', alpha=0.3)
        ax.axvline(x=1e40, color='gray', linestyle=':', alpha=0.3)
        ax.axvline(x=1e50, color='gray', linestyle=':', alpha=0.3)
        ax.axvline(x=1e60, color='gray', linestyle=':', alpha=0.3)
        ax.axvline(x=1e70, color='gray', linestyle=':', alpha=0.3)
        ax.axvline(x=1e80, color='gray', linestyle=':', alpha=0.3)
        ax.axvline(x=1e90, color='gray', linestyle=':', alpha=0.3)
        ax.axvline(x=1e100, color='gray', linestyle=':', alpha=0.3)
        ax.axvline(x=1e110, color='gray', linestyle=':', alpha=0.3)
        ax.axvline(x=1e120, color='gray', linestyle=':', alpha=0.3)
        ax.axvline(x=1e130, color='gray', linestyle=':', alpha=0.3)
        ax.axvline(x=1e140, color='gray', linestyle=':', alpha=0.3)

        # Отмечаем текущее N
        ax.axvline(x=N0, color='black', linestyle='--', linewidth=1, alpha=0.7)

    plt.tight_layout()
    plt.savefig('all_constants_N_range.png', dpi=150, bbox_inches='tight')
    print("[График сохранён в 'all_constants_N_range.png']")

    # ========== БОНУС: СВОДНЫЙ ГРАФИК СРАВНЕНИЯ ==========
    print("\nСоздание сводного графика сравнения...")

    fig_summary, axes_summary = plt.subplots(3, 2, figsize=(18, 16))
    fig_summary.suptitle('Сводное сравнение основных констант (N от 10¹ до 10¹⁵⁰)',
                         fontsize=16, fontweight='bold')

    # 1. Фундаментальные константы
    ax = axes_summary[0, 0]
    ax.loglog(N_new, hbar_new, label='ħ (Дж·с)', color='blue', linewidth=2)
    ax.loglog(N_new, c_new, label='c (м/с)', color='red', linewidth=2)
    ax.loglog(N_new, G_new, label='G (м³/(кг·с²))', color='green', linewidth=2)
    ax.loglog(N_new, kB_new, label='k_B (Дж/К)', color='orange', linewidth=2)
    ax.set_xlabel('N', fontsize=10)
    ax.set_ylabel('Значение', fontsize=10)
    ax.set_title('Фундаментальные константы', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which='both')
    ax.axvline(x=N0, color='black', linestyle='--', alpha=0.5, label=f'N₀ ≈ 10¹²²')

    # 2. Планковские величины
    ax = axes_summary[0, 1]
    ax.loglog(N_new, lP_new, label='l_P (м)', color='blue', linewidth=2)
    ax.loglog(N_new, tP_new, label='t_P (с)', color='red', linewidth=2)
    ax.loglog(N_new, mP_new, label='m_P (кг)', color='green', linewidth=2)
    ax.loglog(N_new, EP_new, label='E_P (Дж)', color='purple', linewidth=2)
    ax.loglog(N_new, TP_new, label='T_P (К)', color='orange', linewidth=2)
    ax.set_xlabel('N', fontsize=10)
    ax.set_ylabel('Значение', fontsize=10)
    ax.set_title('Планковские величины', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which='both')
    ax.axvline(x=N0, color='black', linestyle='--', alpha=0.5)

    # 3. Массы частиц (основные)
    ax = axes_summary[1, 0]
    ax.loglog(N_new, me_new, label='m_e', color='blue', linewidth=2)
    ax.loglog(N_new, m_muon_new, label='m_μ', color='red', linewidth=2)
    ax.loglog(N_new, m_tau_new, label='m_τ', color='green', linewidth=2)
    ax.loglog(N_new, m_proton_new, label='m_p', color='purple', linewidth=2)
    ax.loglog(N_new, m_W_new, label='m_W', color='orange', linewidth=2)
    ax.loglog(N_new, m_Z_new, label='m_Z', color='brown', linewidth=2)
    ax.loglog(N_new, m_Higgs_new, label='m_H', color='pink', linewidth=2)
    ax.set_xlabel('N', fontsize=10)
    ax.set_ylabel('Масса (кг)', fontsize=10)
    ax.set_title('Массы бозонов и лептонов', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which='both')
    ax.axvline(x=N0, color='black', linestyle='--', alpha=0.5)

    # 4. Массы кварков
    ax = axes_summary[1, 1]
    ax.loglog(N_new, m_top_new, label='t-кварк', color='blue', linewidth=2)
    ax.loglog(N_new, m_bottom_new, label='b-кварк', color='red', linewidth=2)
    ax.loglog(N_new, m_charm_new, label='c-кварк', color='green', linewidth=2)
    ax.loglog(N_new, m_strange_new, label='s-кварк', color='purple', linewidth=2)
    ax.loglog(N_new, m_up_new, label='u-кварк', color='orange', linewidth=2)
    ax.loglog(N_new, m_down_new, label='d-кварк', color='brown', linewidth=2)
    ax.set_xlabel('N', fontsize=10)
    ax.set_ylabel('Масса (кг)', fontsize=10)
    ax.set_title('Массы кварков', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which='both')
    ax.axvline(x=N0, color='black', linestyle='--', alpha=0.5)

    # 5. Космология и атомная физика
    ax = axes_summary[2, 0]
    ax.loglog(N_new, abs(Lambda_new), label='|Λ| (м⁻²)', color='blue', linewidth=2)
    ax.loglog(N_new, kappa_new, label='κ (с²/(кг·м))', color='red', linewidth=2)
    ax.loglog(N_new, Rydberg_new, label='R∞ (м⁻¹)', color='green', linewidth=2)
    ax.loglog(N_new, bor_radius_new, label='a₀ (м)', color='purple', linewidth=2)
    ax.set_xlabel('N', fontsize=10)
    ax.set_ylabel('Значение', fontsize=10)
    ax.set_title('Космология и атомная физика', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which='both')
    ax.axvline(x=N0, color='black', linestyle='--', alpha=0.5)

    # 6. Безразмерные параметры
    ax = axes_summary[2, 1]
    alpha_vals_new = np.array([d['α'] for d in data_new])
    lambda_vals_new = np.array([d['λ'] for d in data_new])
    d_eff_vals_new = np.array([d['d_eff'] for d in data_new])
    F_vals_new = np.array([d['F(N)'] for d in data_new])

    ax.semilogx(N_new, alpha_vals_new, label='α', color='blue', linewidth=2)
    ax.semilogx(N_new, lambda_vals_new, label='λ', color='red', linewidth=2)
    ax.semilogx(N_new, d_eff_vals_new, label='d_eff', color='green', linewidth=2)
    ax.semilogx(N_new, F_vals_new, label='F(N)', color='purple', linewidth=2)
    ax.axhline(y=1/137.036, color='blue', linestyle=':', alpha=0.5)
    ax.axhline(y=1.0, color='red', linestyle=':', alpha=0.5)
    ax.axhline(y=3.0, color='green', linestyle=':', alpha=0.5)
    ax.axhline(y=1/pi, color='purple', linestyle=':', alpha=0.5)
    ax.set_xlabel('N', fontsize=10)
    ax.set_ylabel('Значение', fontsize=10)
    ax.set_title('Безразмерные параметры', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axvline(x=N0, color='black', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('summary_constants.png', dpi=150, bbox_inches='tight')
    print("[График сохранён в 'summary_constants.png']")

except ImportError:
    print("\n[Matplotlib не установлен — графики не построены]")

print(f"\nПАРАМЕТР ЭВОЛЮЦИИ: t_P (планковское время из модели)")
print(f"  t_P = 4·K²·(lnK)² / (π·(ln N)²·N^(1/3))")
print(f"\n  Текущее t_P = {data_current['t_P']:.4e} с")
print(f"  CODATA t_P = {tP_exp:.4e} с")
print(f"  Относительная ошибка: {abs(data_current['t_P']/tP_exp - 1)*100:.6f}%")
print(f"\n  ln N(t_P) = {lnN0:.1f}")
print(f"  ln N_opt = {lnN_opt:.1f} (геометрический резонанс)")
print(f"  ln N(λ=1) = {lnN_l1:.1f} (полная согласованность)")
print(f"\n  ВАЖНО: t_P УБЫВАЕТ при росте ln N")
print(f"  Ранняя Вселенная (малые ln N) → большое t_P")
print(f"  Современная эпоха → малое t_P")
print(f"  Будущее → ещё меньше t_P")
print(f"\n  СОЗДАНЫ ГРАФИКИ:")
print(f"  1. 'evolution_vs_tP.png' - Эволюция констант как функция t_P")
print(f"  2. 'all_constants_N_range.png' - ВСЕ 30 констант для N от 10 до 10^150")
print(f"  3. 'summary_constants.png' - Сводный график по группам")