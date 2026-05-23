"""
ИССЛЕДОВАНИЕ ЭВОЛЮЦИИ ЭМЕРДЖЕНТНЫХ КОНСТАНТ ПРИ РОСТЕ ln N
От ln N = 23 до ln N = 281 с шагом ~2.3 (100 точек)
Гипотеза: время соразмерно логарифму N
"""

import math
import numpy as np

# ========== КОНСТАНТЫ ==========
K = 6.0
pi = math.pi
lnK = math.log(K)

# ========== ДИАПАЗОН ln N ==========
lnN_min = 23.0  # примерно 10^10
lnN_max = 281.0  # примерно 10^122
n_points = 100
lnN_values = np.linspace(lnN_min, lnN_max, n_points)
N_values = np.exp(lnN_values)


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
    tP_val = 4 * K ** 2 * lnK ** 2 / (pi * lnN ** 2 * N13)
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

        # Фундаментальные
        'ħ': hbar_val, 'c': c_val, 'G': G_val, 'α': alpha_val, 'k_B': kB_val,

        # Электродинамика
        'q_e': qe_val, 'ε₀': ep0_val, 'μ₀': mu0_val,

        # Планковские
        'l_P': lP_val, 't_P': tP_val, 'm_P': mP_val, 'E_P': EP_val, 'T_P': TP_val,

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


# ========== ВЫЧИСЛЕНИЕ ==========
print("=" * 90)
print("ЭВОЛЮЦИЯ ЭМЕРДЖЕНТНЫХ КОНСТАНТ КАК ФУНКЦИЯ ln N")
print("Гипотеза: время соразмерно логарифму N")
print("=" * 90)
print(f"K = {K}, lnK = {lnK:.6f}")
print(f"Диапазон ln N: {lnN_min:.1f} ... {lnN_max:.1f}")
print(f"Точек: {n_points}")

# Вычисляем
data = [compute_constants(N) for N in N_values]

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
print("ТАБЛИЦА: КЛЮЧЕВЫЕ ЭПОХИ (по ln N)")
print("=" * 90)

key_lnN = [23, 50, 100, 150, 200, 230, 250, 260, 270, 275, 280, 281]
header = (f"{'ln N':>8} {'N':>12} {'F(N)':>8} {'α':>8} "
          f"{'ħ':>12} {'c':>12} {'m_e (кг)':>14} {'l_P (м)':>12} {'λ':>8} {'d_eff':>7}")
print(header)
print("-" * len(header))

for target_lnN in key_lnN:
    # Находим ближайшую точку
    idx = np.argmin(np.abs(lnN_values - target_lnN))
    d = data[idx]
    print(f"{d['lnN']:>8.1f} {d['N']:>12.4e} {d['F(N)']:>8.5f} {d['α']:>8.5f} "
          f"{d['ħ']:>12.4e} {d['c']:>12.4e} "
          f"{d['m_e']:>14.6e} {d['l_P']:>12.4e} {d['λ']:>8.5f} {d['d_eff']:>7.4f}")

# ========== ТЕКУЩИЕ ЗНАЧЕНИЯ ==========
print("\n" + "=" * 90)
print(f"ТЕКУЩИЕ ЗНАЧЕНИЯ (ln N = {lnN0:.4f}, N = {N0:.4e})")
print("=" * 90)

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
print("\n" + "=" * 90)
print("ПРОИЗВОДНЫЕ ВЕЛИЧИНЫ")
print("=" * 90)

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
print("\n" + "=" * 90)
print("ГЕОМЕТРИЧЕСКИЙ РЕЗОНАНС")
print("=" * 90)

lnN_opt = (K - lnK) / (1 / 3 - 1 / pi)
N_opt = math.exp(lnN_opt)
d_opt = compute_constants(N_opt)
print(f"  ln N_opt = (K - lnK) / (1/3 - 1/π) = {lnN_opt:.4f}")
print(f"  N_opt = {N_opt:.4e}")
print(f"  При ln N_opt: F(N) = {d_opt['F(N)']:.10f}")
print(f"  Ошибка: {d_opt['F_error_%']:.6f}%")
print(f"  λ = {d_opt['λ']:.6f}, d_eff = {d_opt['d_eff']:.4f}")

# ========== λ=1 ==========
print("\n" + "=" * 90)
print("ТОЧКА λ=1 (ПОЛНАЯ ПЛАНКОВСКАЯ САМОСОГЛАСОВАННОСТЬ)")
print("=" * 90)

lnN_l1 = 4 * K ** 3 * lnK ** 2 / pi ** 2
N_l1 = math.exp(lnN_l1)
print(f"  ln N = 4·K³·(lnK)²/π² = {lnN_l1:.4f}")
print(f"  N = {N_l1:.4e}")

# ========== ПРОИЗВОДНЫЕ ПО ln N ==========
print("\n" + "=" * 90)
print("ПРОИЗВОДНЫЕ ПО ln N (СКОРОСТЬ ИЗМЕНЕНИЯ)")
print("=" * 90)

# Берём две близкие точки для численной производной
idx_current = np.argmin(np.abs(lnN_values - lnN0))
if idx_current < len(data) - 1:
    d1 = data[idx_current]
    d2 = data[idx_current + 1]
    dlnN = d2['lnN'] - d1['lnN']

    print(f"\n  d(ln константы) / d(ln N) при ln N ≈ {lnN0:.1f}:")

    for name, key in [("ħ", 'ħ'), ("c", 'c'), ("G", 'G'), ("α", 'α'),
                      ("m_e", 'm_e'), ("m_p", 'm_p'), ("l_P", 'l_P'),
                      ("E_P", 'E_P'), ("Λ", 'Λ')]:
        deriv = (math.log(d2[key]) - math.log(d1[key])) / dlnN
        print(f"    d(ln {name})/d(ln N) = {deriv:+.6f}")

# ========== ГРАФИКИ ==========
try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle('Эволюция эмерджентных констант как функция ln N', fontsize=16, fontweight='bold')

    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    # 1. Функция геометрического резонанса
    ax = axes[0, 0]
    F_vals = [d['F(N)'] for d in data]
    ax.plot(lnN_values, F_vals, color=colors[0], linewidth=1.5)
    ax.axhline(y=1 / pi, color='red', linestyle='--', linewidth=1, label=f'1/π = {1 / pi:.5f}')
    ax.axhline(y=1 / 3, color='green', linestyle='--', linewidth=1, label='1/3 (предел)')
    ax.axvline(x=lnN_opt, color='orange', linestyle=':', linewidth=1.5, label=f'ln N_opt = {lnN_opt:.1f}')
    ax.set_xlabel('ln N', fontsize=10)
    ax.set_ylabel('F(N)', fontsize=10)
    ax.set_title('Функция геометрического резонанса F(ln N)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(True, alpha=0.3)

    # 2. Ошибка F(N)
    ax = axes[0, 1]
    F_err = [d['F_error_%'] for d in data]
    ax.semilogy(lnN_values, F_err, color=colors[3], linewidth=1.5)
    ax.axvline(x=lnN_opt, color='orange', linestyle=':', linewidth=1.5, label=f'ln N_opt')
    ax.set_xlabel('ln N', fontsize=10)
    ax.set_ylabel('Ошибка (%)', fontsize=10)
    ax.set_title('Ошибка геометрического резонанса', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. Постоянная тонкой структуры
    ax = axes[0, 2]
    alpha_vals = [d['α'] for d in data]
    ax.plot(lnN_values, alpha_vals, color=colors[4], linewidth=1.5)
    ax.axhline(y=1 / 137.036, color='red', linestyle='--', linewidth=1, label='CODATA ≈ 1/137')
    ax.set_xlabel('ln N', fontsize=10)
    ax.set_ylabel('α', fontsize=10)
    ax.set_title('Постоянная тонкой структуры α', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 4. Фундаментальные константы (логарифмический масштаб)
    ax = axes[1, 0]
    hbar_vals = np.array([d['ħ'] for d in data])
    c_vals = np.array([d['c'] for d in data])
    G_vals = np.array([d['G'] for d in data])

    ax.semilogy(lnN_values, hbar_vals, color=colors[0], label='ħ (Дж·с)', linewidth=1.5)
    ax.semilogy(lnN_values, G_vals, color=colors[2], label='G (м³/(кг·с²))', linewidth=1.5)

    ax2 = ax.twinx()
    ax2.semilogy(lnN_values, c_vals, color=colors[1], label='c (м/с)', linewidth=1.5)

    ax.set_xlabel('ln N', fontsize=10)
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

    ax.loglog(lnN_values, me_vals, color=colors[0], label='m_e', linewidth=1.5)
    ax.loglog(lnN_values, mp_vals, color=colors[1], label='m_p', linewidth=1.5)
    ax.loglog(lnN_values, mW_vals, color=colors[5], label='m_W', linewidth=1.5)
    ax.set_xlabel('ln N', fontsize=10)
    ax.set_ylabel('Масса (кг)', fontsize=10)
    ax.set_title('Массы частиц', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 6. Планковские величины
    ax = axes[1, 2]
    lP_vals = np.array([d['l_P'] for d in data])
    tP_vals = np.array([d['t_P'] for d in data])
    EP_vals = np.array([d['E_P'] for d in data])

    ax.loglog(lnN_values, lP_vals, color=colors[0], label='l_P (м)', linewidth=1.5)
    ax.loglog(lnN_values, tP_vals, color=colors[1], label='t_P (с)', linewidth=1.5)
    ax.loglog(lnN_values, EP_vals, color=colors[2], label='E_P (Дж)', linewidth=1.5)
    ax.set_xlabel('ln N', fontsize=10)
    ax.set_ylabel('Значение', fontsize=10)
    ax.set_title('Планковские величины', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 7. λ и эффективная размерность
    ax = axes[2, 0]
    lambda_vals = [d['λ'] for d in data]
    d_eff_vals = [d['d_eff'] for d in data]

    ax2 = ax.twinx()
    ax.plot(lnN_values, lambda_vals, color=colors[0], linewidth=1.5, label='λ')
    ax2.plot(lnN_values, d_eff_vals, color=colors[3], linestyle='--', linewidth=1.5, label='d_eff')
    ax.axhline(y=1.0, color='green', linestyle=':', linewidth=1, label='λ=1')
    ax2.axhline(y=3.0, color='orange', linestyle=':', linewidth=1, label='d_eff=3')
    ax.set_xlabel('ln N', fontsize=10)
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

    ax.loglog(lnN_values, qe_vals, color=colors[0], label='q_e (Кл)', linewidth=1.5)
    ax.loglog(lnN_values, Lambda_vals, color=colors[3], label='Λ (м⁻²)', linewidth=1.5)
    ax.set_xlabel('ln N', fontsize=10)
    ax.set_ylabel('Значение', fontsize=10)
    ax.set_title('Заряд электрона и Λ', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 9. Безразмерные отношения
    ax = axes[2, 2]
    mW_mZ_vals = [d['m_W/m_Z'] for d in data]
    mH_mW_vals = [d['m_H/m_W'] for d in data]
    mp_me_vals = [d['m_p/m_e'] for d in data]

    ax.plot(lnN_values, mp_me_vals, color=colors[0], label='m_p/m_e', linewidth=1.5)
    ax.plot(lnN_values, mW_mZ_vals, color=colors[5], label='m_W/m_Z (const)', linewidth=1.5)
    ax.plot(lnN_values, mH_mW_vals, color=colors[6], label='m_H/m_W (const)', linewidth=1.5)
    ax.axhline(y=pi ** 0.5 / 2, color='red', linestyle=':', linewidth=1, label=f'√π/2')
    ax.set_xlabel('ln N', fontsize=10)
    ax.set_ylabel('Значение', fontsize=10)
    ax.set_title('Безразмерные отношения', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('evolution_vs_lnN.png', dpi=150, bbox_inches='tight')
    print("\n[График сохранён в 'evolution_vs_lnN.png']")

except ImportError:
    print("\n[Matplotlib не установлен — график не построен]")

print("\n" + "=" * 90)
print("ИССЛЕДОВАНИЕ ЗАВЕРШЕНО")
print("=" * 90)
print(f"\nГИПОТЕЗА: время ∝ ln N")
print(f"  Текущее ln N = {lnN0:.1f}")
print(f"  ln N_opt = {lnN_opt:.1f}")
print(f"  ln N(λ=1) = {lnN_l1:.1f}")
print(f"  Если время пропорционально ln N, то:")
print(f"    Возраст Вселенной ∝ {lnN0:.0f}")
print(f"    Точка резонанса ∝ {lnN_opt:.0f}")
print(f"    Точка полной согласованности ∝ {lnN_l1:.0f}")