"""
ИССЛЕДОВАНИЕ ЭВОЛЮЦИИ ЭМЕРДЖЕНТНЫХ КОНСТАНТ
Гипотеза 3: время ∝ ln N / N^(1/3)
"""

import math
import numpy as np

# ========== КОНСТАНТЫ ==========
K = 6.0
pi = math.pi
lnK = math.log(K)

# ========== ДИАПАЗОН τ = ln N / N^(1/3) ==========
# Сначала вычисляем N_values из равномерной шкалы ln N
lnN_min = 23.0
lnN_max = 281.0
n_points = 100
lnN_values = np.linspace(lnN_min, lnN_max, n_points)
N_values = np.exp(lnN_values)

# Вычисляем τ = ln N / N^(1/3)
tau_values = lnN_values / (N_values ** (1 / 3))


# ========== ФУНКЦИИ ДЛЯ ВЫЧИСЛЕНИЯ КОНСТАНТ ==========
def compute_constants(N):
    lnN = math.log(N)
    N13 = N ** (1 / 3)
    N23 = N ** (2 / 3)
    p_val = 1 / (K * N13)
    Kp = K * p_val
    sqrtK = math.sqrt(K)
    sqrt3 = math.sqrt(3)
    sqrt2 = math.sqrt(2)
    sqrtPi = math.sqrt(pi)

    # Время по гипотезе 3
    tau = lnN / N13

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

    # Массы
    me_val = 4 * pi * lnN ** 4 / (sqrtK * N13)
    m_muon = 4 * pi ** 2 * lnN ** 5 / (K * sqrt3 * N13)
    m_tau = sqrtPi * lnN ** 5 * K ** 2 / N13
    m_proton = sqrtPi * lnN ** 6 / (K ** (3 / 2) * N13)
    m_W = 2 * pi ** 3 * lnN ** 6 / (N13 * K)
    m_Z = lnN ** 6 * 4 * pi ** (5 / 2) / (N13 * K)
    m_Higgs = lnN ** 6 * 4 * pi ** 2 / (N13 * sqrtK)
    m_qu_top = lnN ** 6 * K ** 3 / (pi ** 2 * N13)

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
        'N': N, 'lnN': lnN, 'tau': tau,
        'F(N)': F_val, 'F_error_%': F_error,
        'ħ': hbar_val, 'c': c_val, 'G': G_val, 'α': alpha_val, 'k_B': kB_val,
        'q_e': qe_val, 'l_P': lP_val, 't_P': tP_val, 'm_P': mP_val,
        'E_P': EP_val, 'T_P': TP_val,
        'm_e': me_val, 'm_μ': m_muon, 'm_τ': m_tau, 'm_p': m_proton,
        'm_W': m_W, 'm_Z': m_Z, 'm_H': m_Higgs, 'm_t': m_qu_top,
        'v_Higgs': v_Higgs, 'Λ': Lambda_val, 'κ': kappa_val,
        'R∞': Rydberg, 'a₀': bor_radius, 'E_Ry': E_rydberg,
        'm_W/m_Z': mW_mZ, 'm_H/m_W': mH_mW, 'm_p/m_e': mp_me, 'm_μ/m_e': m_muon_me,
        'λ': lambda_factor, 'd_eff': d_eff,
    }


# ========== ВЫЧИСЛЕНИЕ ==========
print("=" * 90)
print("ЭВОЛЮЦИЯ ЭМЕРДЖЕНТНЫХ КОНСТАНТ КАК ФУНКЦИЯ τ = ln N / N^(1/3)")
print("Гипотеза 3: время ∝ ln N / N^(1/3)")
print("=" * 90)
print(f"K = {K}, lnK = {lnK:.6f}")
print(f"Диапазон ln N: {lnN_min:.1f} ... {lnN_max:.1f}")
print(f"Точек: {n_points}")

# Вычисляем
data = [compute_constants(N) for N in N_values]

# Текущие значения
N0 = 4.198e121
lnN0 = math.log(N0)
N13_0 = N0 ** (1 / 3)
tau0 = lnN0 / N13_0
data_current = compute_constants(N0)

# Экспериментальные значения (СИ)
c_exp = 299792458
hbar_exp = 1.054571817e-34
G_exp = 6.67430e-11
alpha_exp = 1 / 137.035999084
me_exp = 9.1093837015e-31
m_muon_exp = 1.883531627e-28
m_tau_exp = 3.167e-27
m_proton_exp = 1.67262192e-27
m_W_exp = 1.43362e-25
m_Z_exp = 1.62614e-25
m_Higgs_exp = 2.23319e-25
m_top_exp = 3.04e-25
lP_exp = 1.616255e-35
tP_exp = 5.391247e-44
mP_exp = 2.176434e-8
EP_exp = 1.956082e9
Lambda_exp = 1.08929e-52
Rydberg_exp = 1.097373e7

# ========== ТАБЛИЦА ПО КЛЮЧЕВЫМ τ ==========
print("\n" + "=" * 90)
print("ТАБЛИЦА: КЛЮЧЕВЫЕ ЭПОХИ (по τ = ln N / N^(1/3))")
print("=" * 90)

header = (f"{'τ':>12} {'ln N':>8} {'N':>12} {'F(N)':>8} {'α':>8} "
          f"{'ħ':>12} {'m_e (кг)':>14} {'l_P (м)':>12} {'d_eff':>7}")
print(header)
print("-" * len(header))

# Выбираем характерные точки
for idx in [0, 10, 25, 50, 75, 90, 95, 98, 99]:
    if idx < len(data):
        d = data[idx]
        print(f"{d['tau']:>12.4e} {d['lnN']:>8.1f} {d['N']:>12.4e} {d['F(N)']:>8.5f} "
              f"{d['α']:>8.5f} {d['ħ']:>12.4e} {d['m_e']:>14.6e} "
              f"{d['l_P']:>12.4e} {d['d_eff']:>7.4f}")

# ========== ТЕКУЩИЕ ЗНАЧЕНИЯ ==========
print("\n" + "=" * 90)
print(f"ТЕКУЩИЕ ЗНАЧЕНИЯ (τ = {tau0:.4e}, ln N = {lnN0:.4f}, N = {N0:.4e})")
print("=" * 90)

comparisons = [
    ("ħ (Дж·с)", data_current['ħ'], hbar_exp),
    ("c (м/с)", data_current['c'], c_exp),
    ("G (м³/(кг·с²))", data_current['G'], G_exp),
    ("α", data_current['α'], alpha_exp),
    ("l_P (м)", data_current['l_P'], lP_exp),
    ("t_P (с)", data_current['t_P'], tP_exp),
    ("m_P (кг)", data_current['m_P'], mP_exp),
    ("E_P (Дж)", data_current['E_P'], EP_exp),
    ("m_e (кг)", data_current['m_e'], me_exp),
    ("m_μ (кг)", data_current['m_μ'], m_muon_exp),
    ("m_τ (кг)", data_current['m_τ'], m_tau_exp),
    ("m_p (кг)", data_current['m_p'], m_proton_exp),
    ("m_W (кг)", data_current['m_W'], m_W_exp),
    ("m_Z (кг)", data_current['m_Z'], m_Z_exp),
    ("m_H (кг)", data_current['m_H'], m_Higgs_exp),
    ("m_t (кг)", data_current['m_t'], m_top_exp),
    ("Λ (м⁻²)", data_current['Λ'], Lambda_exp),
    ("R∞ (м⁻¹)", data_current['R∞'], Rydberg_exp),
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

print(f"  τ = ln N / N^(1/3) = {tau0:.6e}")
print(f"  ln N = {lnN0:.4f}")
print(f"  N = {N0:.4e}")
print(f"  F(N) = {data_current['F(N)']:.10f}")
print(f"  Цель (1/π) = {1 / pi:.10f}")
print(f"  Ошибка F(N): {data_current['F_error_%']:.6f}%")
print(f"  λ = {data_current['λ']:.6f}")
print(f"  d_eff = {data_current['d_eff']:.4f}")

# ========== ОСОБЫЕ ТОЧКИ ==========
print("\n" + "=" * 90)
print("ОСОБЫЕ ТОЧКИ В ШКАЛЕ τ")
print("=" * 90)

lnN_opt = (K - lnK) / (1 / 3 - 1 / pi)
N_opt = math.exp(lnN_opt)
tau_opt = lnN_opt / (N_opt ** (1 / 3))

lnN_l1 = 4 * K ** 3 * lnK ** 2 / pi ** 2
N_l1 = math.exp(lnN_l1)
tau_l1 = lnN_l1 / (N_l1 ** (1 / 3))

print(f"  {'Точка':<25} {'τ':<16} {'ln N':<12} {'N':<16}")
print(f"  {'-' * 65}")
print(f"  {'Текущая':<25} {tau0:<16.6e} {lnN0:<12.4f} {N0:<16.4e}")
print(f"  {'Геометрический резонанс':<25} {tau_opt:<16.6e} {lnN_opt:<12.4f} {N_opt:<16.4e}")
print(f"  {'λ=1 (полная согласованность)':<25} {tau_l1:<16.6e} {lnN_l1:<12.4f} {N_l1:<16.4e}")

# ========== ПРОИЗВОДНЫЕ ПО τ ==========
print("\n" + "=" * 90)
print("ПРОИЗВОДНЫЕ ПО τ (СКОРОСТЬ ИЗМЕНЕНИЯ)")
print("=" * 90)

idx_current = np.argmin(np.abs(tau_values - tau0))
if idx_current < len(data) - 1:
    d1 = data[idx_current]
    d2 = data[idx_current + 1]
    dtau = d2['tau'] - d1['tau']

    print(f"\n  d(ln константы) / dτ при τ ≈ {tau0:.4e}:")
    for name, key in [("ħ", 'ħ'), ("c", 'c'), ("G", 'G'), ("α", 'α'),
                      ("m_e", 'm_e'), ("m_p", 'm_p'), ("l_P", 'l_P'),
                      ("E_P", 'E_P'), ("Λ", 'Λ'), ("d_eff", 'd_eff')]:
        if d1[key] > 0 and d2[key] > 0:
            deriv = (math.log(d2[key]) - math.log(d1[key])) / dtau
            print(f"    d(ln {name})/dτ = {deriv:+.6e}")

# ========== ГРАФИКИ ==========
try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle('Эволюция эмерджентных констант как функция τ = ln N / N^(1/3)',
                 fontsize=14, fontweight='bold')

    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    tau_arr = np.array(tau_values)

    # 1. F(N)
    ax = axes[0, 0]
    F_vals = [d['F(N)'] for d in data]
    ax.plot(tau_arr, F_vals, color=colors[0], linewidth=1.5)
    ax.axhline(y=1 / pi, color='red', linestyle='--', linewidth=1, label=f'1/π')
    ax.axvline(x=tau_opt, color='orange', linestyle=':', linewidth=1.5, label=f'τ_opt')
    ax.set_xlabel('τ = ln N / N^(1/3)', fontsize=10)
    ax.set_ylabel('F(N)', fontsize=10)
    ax.set_title('Функция геометрического резонанса', fontsize=11, fontweight='bold')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 2. α
    ax = axes[0, 1]
    alpha_vals = [d['α'] for d in data]
    ax.plot(tau_arr, alpha_vals, color=colors[4], linewidth=1.5)
    ax.set_xlabel('τ', fontsize=10)
    ax.set_ylabel('α', fontsize=10)
    ax.set_title('Постоянная тонкой структуры', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 3. d_eff
    ax = axes[0, 2]
    d_eff_vals = [d['d_eff'] for d in data]
    ax.plot(tau_arr, d_eff_vals, color=colors[3], linewidth=1.5)
    ax.axhline(y=3.0, color='green', linestyle='--', linewidth=1, label='d_eff = 3')
    ax.set_xlabel('τ', fontsize=10)
    ax.set_ylabel('d_eff', fontsize=10)
    ax.set_title('Эффективная размерность', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 4. Фундаментальные константы
    ax = axes[1, 0]
    hbar_vals = np.array([d['ħ'] for d in data])
    G_vals = np.array([d['G'] for d in data])
    c_vals = np.array([d['c'] for d in data])

    ax.loglog(tau_arr, hbar_vals, color=colors[0], label='ħ', linewidth=1.5)
    ax.loglog(tau_arr, G_vals, color=colors[2], label='G', linewidth=1.5)
    ax.loglog(tau_arr, c_vals, color=colors[1], label='c', linewidth=1.5)
    ax.set_xlabel('τ', fontsize=10)
    ax.set_ylabel('Значение', fontsize=10)
    ax.set_title('ħ, c, G', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 5. Массы
    ax = axes[1, 1]
    me_vals = np.array([d['m_e'] for d in data])
    mp_vals = np.array([d['m_p'] for d in data])
    mW_vals = np.array([d['m_W'] for d in data])
    mH_vals = np.array([d['m_H'] for d in data])

    ax.loglog(tau_arr, me_vals, color=colors[0], label='m_e', linewidth=1.5)
    ax.loglog(tau_arr, mp_vals, color=colors[1], label='m_p', linewidth=1.5)
    ax.loglog(tau_arr, mW_vals, color=colors[5], label='m_W', linewidth=1.5)
    ax.loglog(tau_arr, mH_vals, color=colors[7], label='m_H', linewidth=1.5)
    ax.set_xlabel('τ', fontsize=10)
    ax.set_ylabel('Масса (кг)', fontsize=10)
    ax.set_title('Массы частиц', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 6. Планковские величины
    ax = axes[1, 2]
    lP_vals = np.array([d['l_P'] for d in data])
    tP_vals = np.array([d['t_P'] for d in data])
    EP_vals = np.array([d['E_P'] for d in data])

    ax.loglog(tau_arr, lP_vals, color=colors[0], label='l_P', linewidth=1.5)
    ax.loglog(tau_arr, tP_vals, color=colors[1], label='t_P', linewidth=1.5)
    ax.loglog(tau_arr, EP_vals, color=colors[2], label='E_P', linewidth=1.5)
    ax.set_xlabel('τ', fontsize=10)
    ax.set_ylabel('Значение', fontsize=10)
    ax.set_title('Планковские величины', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 7. Космология
    ax = axes[2, 0]
    Lambda_vals = np.array([d['Λ'] for d in data])
    kappa_vals = np.array([d['κ'] for d in data])
    vH_vals = np.array([d['v_Higgs'] for d in data])

    ax.loglog(tau_arr, Lambda_vals, color=colors[0], label='Λ', linewidth=1.5)
    ax.loglog(tau_arr, kappa_vals, color=colors[3], label='κ', linewidth=1.5)
    ax.loglog(tau_arr, vH_vals, color=colors[5], label='v_Higgs', linewidth=1.5)
    ax.set_xlabel('τ', fontsize=10)
    ax.set_ylabel('Значение', fontsize=10)
    ax.set_title('Космология и вакуум', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 8. Безразмерные отношения
    ax = axes[2, 1]
    mp_me_vals = [d['m_p/m_e'] for d in data]
    mmu_me_vals = [d['m_μ/m_e'] for d in data]
    mW_mZ_vals = [d['m_W/m_Z'] for d in data]

    ax.plot(tau_arr, mp_me_vals, color=colors[0], label='m_p/m_e', linewidth=1.5)
    ax.plot(tau_arr, mmu_me_vals, color=colors[1], label='m_μ/m_e', linewidth=1.5)
    ax.plot(tau_arr, mW_mZ_vals, color=colors[5], label='m_W/m_Z (const)', linewidth=1.5)
    ax.set_xlabel('τ', fontsize=10)
    ax.set_ylabel('Значение', fontsize=10)
    ax.set_title('Безразмерные отношения', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 9. Ошибка F(N)
    ax = axes[2, 2]
    F_err = [d['F_error_%'] for d in data]
    ax.semilogy(tau_arr, F_err, color=colors[3], linewidth=1.5)
    ax.axvline(x=tau_opt, color='orange', linestyle=':', linewidth=1.5, label=f'τ_opt')
    ax.set_xlabel('τ', fontsize=10)
    ax.set_ylabel('Ошибка (%)', fontsize=10)
    ax.set_title('Ошибка геометрического резонанса', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('evolution_vs_tau.png', dpi=150, bbox_inches='tight')
    print("\n[График сохранён в 'evolution_vs_tau.png']")

except ImportError:
    print("\n[Matplotlib не установлен — график не построен]")

# ========== АНАЛИЗ ГИПОТЕЗЫ ==========
print("\n" + "=" * 90)
print("АНАЛИЗ ГИПОТЕЗЫ 3: время ∝ ln N / N^(1/3)")
print("=" * 90)

# Оцениваем t0
t_universe = 4.35e17  # возраст Вселенной в секундах
t0_h3 = t_universe / tau0

print(f"\n  Если t = t₀ · τ, то:")
print(f"    τ_now = {tau0:.6e}")
print(f"    t₀ = t_now / τ_now = {t0_h3:.4e} с")
print(f"    t₀ ≈ {t0_h3 / 86400 / 365.25:.2f} лет")
print(f"    Сравнение с t_P = {data_current['t_P']:.4e} с")
print(f"    t₀ / t_P = {t0_h3 / data_current['t_P']:.4e}")

# Сравнение трёх гипотез
print(f"\n  СРАВНЕНИЕ ТРЁХ ГИПОТЕЗ:")
print(f"    Гипотеза 1 (t ∝ ln N):    t₀ = {t_universe / lnN0:.4e} с = {t_universe / lnN0 / 86400 / 365.25:.2f} лет")
print(f"    Гипотеза 2 (t ∝ N):       t₀ = {t_universe / N0:.4e} с (планковский масштаб)")
print(f"    Гипотеза 3 (t ∝ lnN/N^(1/3)): t₀ = {t0_h3:.4e} с = {t0_h3 / 86400 / 365.25:.2f} лет")

# Ключевые моменты в годах
print(f"\n  КЛЮЧЕВЫЕ МОМЕНТЫ В ГОДАХ (Гипотеза 3):")
print(f"    Сейчас:              0 лет (τ = {tau0:.4e})")
print(f"    Резонанс через:      {(tau_opt - tau0) * t0_h3 / 86400 / 365.25:.2e} лет")
print(f"    λ=1 через:           {(tau_l1 - tau0) * t0_h3 / 86400 / 365.25:.2e} лет")

# Обратный отсчёт
print(f"\n  ОБРАТНЫЙ ОТСЧЁТ (Гипотеза 3):")
for frac in [0.01, 0.1, 0.5, 0.9, 0.99]:
    tau_past = tau0 * frac
    # Находим соответствующий ln N
    # τ = lnN / N^(1/3) — решаем численно
    print(f"    {frac * 100:.0f}% возраста: τ = {tau_past:.4e}")

print("\n" + "=" * 90)
print("ИССЛЕДОВАНИЕ ЗАВЕРШЕНО")
print("=" * 90)