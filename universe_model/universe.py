import numpy as np
import matplotlib.pyplot as plt

# КОНСТАНТЫ
K = 6.0
pi = np.pi
lnK = np.log(K)

# СВЯЗЬ N И ВРЕМЕНИ (из механизма стрелы времени)
# dS/dt ~ N^(-1/3) => S ~ t * N^(-1/3)
# Но S ~ ln N (энтропия горизонта)
# ln N ~ t * N^(-1/3) => в первом приближении t ~ N^(1/3)

N_now = 4.2e121
lnN_now = np.log(N_now)
t_now = 13.8e9  # лет (возраст Вселенной)

# Калибровка: t = t0 * N^(1/3)
t0 = t_now / N_now ** (1 / 3)


def N_from_time(t_years):
    """Вычисляет N по времени (в годах)"""
    return (t_years / t0) ** 3


def time_from_N(N):
    """Вычисляет время (в годах) по N"""
    return t0 * N ** (1 / 3)


# ЭМЕРДЖЕНТНЫЕ ФОРМУЛЫ
def compute_constants(N):
    """Вычисляет все эмерджентные константы для данного N"""
    lnN = np.log(N)
    N13 = N ** (1 / 3)

    hbar_val = (lnN ** 3) / (K * N13)
    h_val = 2 * pi * hbar_val
    c_val = pi * (lnN ** 4) / (K ** 2 * lnK)
    lP_val = 4 * lnN ** 2 * lnK / N13
    tP_val = 4 * K ** 2 * lnK ** 2 / (pi * N13 * lnN ** 2)
    EP_val = (lnN ** 5) * pi / (4 * K ** 3 * lnK ** 2)
    G_val = 16 * pi ** 3 * lnN ** 13 / (K ** 5 * lnK * N13)
    mP_val = K / (pi * 4 * lnN ** 3)
    TP_val = 8 * pi * N13 / (lnN ** 4)
    k_B_val = (lnN ** 8) / (8 * pi ** 2 * N13)  # Kp = 1/N^(1/3)
    alpha_val = 2 * lnK ** 2 / (pi * lnN)
    m_e_val = 4 * pi * lnN ** 4 / (K ** 0.5 * N13)
    m_proton_val = np.sqrt(pi) * (lnN ** 6) / (K ** 1.5 * N13)

    return {
        'hbar': hbar_val, 'h': h_val, 'c': c_val,
        'l_P': lP_val, 't_P': tP_val, 'E_P': EP_val,
        'G': G_val, 'm_P': mP_val, 'T_P': TP_val,
        'k_B': k_B_val, 'alpha': alpha_val,
        'm_e': m_e_val, 'm_p': m_proton_val
    }


# ТЕКУЩИЕ ЗНАЧЕНИЯ
const_now = compute_constants(N_now)

print("ТЕКУЩИЕ ЗНАЧЕНИЯ ЭМЕРДЖЕНТНЫХ КОНСТАНТ")
for name, value in const_now.items():
    print(f"  {name:<8} = {value:.6e}")

# ВРЕМЕННАЯ ШКАЛА
# От ранней Вселенной до далёкого будущего
t_years = np.logspace(-40, 3, 500) * t_now  # от планковской эпохи до 1000*t_now

N_values = N_from_time(t_years)
lnN_values = np.log(N_values)

# Убедимся, что не выходим за разумные пределы
mask = (N_values > 100) & (N_values < 1e300)
t_years = t_years[mask]
N_values = N_values[mask]
lnN_values = lnN_values[mask]

# ВЫЧИСЛЕНИЕ КОНСТАНТ ДЛЯ ВСЕХ ВРЕМЁН
constants_over_time = {name: [] for name in const_now.keys()}
for N in N_values:
    consts = compute_constants(N)
    for name in const_now.keys():
        constants_over_time[name].append(consts[name])

# ВИЗУАЛИЗАЦИЯ
fig, axes = plt.subplots(3, 3, figsize=(18, 16))
fig.suptitle('Эволюция эмерджентных констант с космическим временем\n(нормировка на современные значения)',
             fontsize=16, fontweight='bold')

# Конфигурация графиков
plots = [
    ('hbar', 'Постоянная Планка ħ', 'ħ'),
    ('c', 'Скорость света c', 'c'),
    ('G', 'Гравитационная постоянная G', 'G'),
    ('l_P', 'Планковская длина l_P', 'l_P'),
    ('t_P', 'Планковское время t_P', 't_P'),
    ('m_P', 'Планковская масса m_P', 'm_P'),
    ('E_P', 'Планковская энергия E_P', 'E_P'),
    ('alpha', 'Постоянная тонкой структуры α', 'α'),
    ('m_p', 'Масса протона m_p', 'm_p'),
]

for idx, (key, title, label) in enumerate(plots):
    ax = axes[idx // 3, idx % 3]
    values = np.array(constants_over_time[key])

    # Нормировка на текущее значение
    values_norm = values / const_now[key]

    # Разделение на прошлое и будущее
    past_mask = t_years <= t_now
    future_mask = t_years >= t_now

    t_norm = t_years / t_now

    ax.plot(t_norm[past_mask], values_norm[past_mask], 'b-', linewidth=2, label='Прошлое')
    ax.plot(t_norm[future_mask], values_norm[future_mask], 'r-', linewidth=2, label='Будущее')

    # Отметка современной эпохи
    ax.axvline(x=1.0, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.axhline(y=1.0, color='gray', linestyle=':', linewidth=1, alpha=0.5)

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlabel('Время (в единицах $t_{now}$ = 13.8 млрд лет)', fontsize=9)
    ax.set_ylabel(f'{label} / {label}$_{{\\rm now}}$', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    if idx == 0:
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('constants_vs_time_corrected.png', dpi=150, bbox_inches='tight')
plt.show()

# ДЕТАЛЬНЫЙ АНАЛИЗ СОВРЕМЕННОЙ ЭПОХИ
print("АНАЛИЗ ЭВОЛЮЦИИ КОНСТАНТ В СОВРЕМЕННУЮ ЭПОХУ")

# Производная d(ln N)/dt в современную эпоху
# N = (t/t0)^3 => ln N = 3 ln t - 3 ln t0 => d(ln N)/dt = 3/t
dlnN_dt_now = 3.0 / t_now  # год^(-1)

print(f"\n  Современная скорость роста ln N:")
print(f"  d(ln N)/dt = {dlnN_dt_now:.2e} год^(-1)")
print(f"  Характерное время удвоения ln N: {np.log(2) / dlnN_dt_now:.2e} лет")

# Относительные скорости изменения констант
print(f"\n  Относительные скорости изменения констант:")
print(f"  {'Константа':<12} {'d(ln C)/d(ln N)':<18} {'(dC/dt)/C (год⁻¹)':<22} {'ΔC/C за 100 лет':<18}")
print(f"  {'-' * 70}")

# Вычисляем производные численно (более надёжно)
eps = 1.001
N_plus = N_now * eps
const_plus = compute_constants(N_plus)

for name in const_now.keys():
    if const_now[name] > 0 and const_plus[name] > 0:
        dlnC_dlnN = (np.log(const_plus[name]) - np.log(const_now[name])) / np.log(eps)
        dC_dt_rel = dlnC_dlnN * dlnN_dt_now
        delta_100yr = dC_dt_rel * 100

        # Оценка: можно ли измерить за 100 лет?
        if abs(delta_100yr) > 1e-12:
            measurable = "✓ возможно"
        else:
            measurable = "✗ пока нет"

        print(f"  {name:<12} {dlnC_dlnN:<+18.6f} {dC_dt_rel:<+22.6e} {delta_100yr:<+18.6e}  {measurable}")

# ФИЗИЧЕСКИЕ ЭПОХИ
print("ФИЗИЧЕСКИЕ ЭПОХИ И ЗНАЧЕНИЯ КОНСТАНТ")

epochs = [
    (1e-43 * t_now, "Планковская эпоха (10⁻⁴³ с)"),
    (1e-36 * t_now, "Инфляционная эпоха (10⁻³⁶ с)"),
    (1e-12 * t_now, "Электрослабая эпоха (10⁻¹² с)"),
    (1e-5 * t_now, "Адронная эпоха (10⁻⁵ с)"),
    (1e0 * t_now, "Современная эпоха (13.8 млрд лет)"),
    (1e2 * t_now, "Далёкое будущее (×100)"),
    (1e3 * t_now, "Далёкое будущее (×1000)"),
]

print(f"\n  {'Эпоха':<35} {'N':<15} {'ln N':<12} {'ħ/ħ_now':<12} {'c/c_now':<12} {'m_p/m_p_now':<12}")
print(f"  {'-' * 100}")

for t_epoch, name in epochs:
    N_epoch = N_from_time(t_epoch)
    if N_epoch > 10 and N_epoch < 1e300:
        const_epoch = compute_constants(N_epoch)
        lnN_epoch = np.log(N_epoch)

        print(f"  {name:<35} {N_epoch:<15.2e} {lnN_epoch:<12.1f} "
              f"{const_epoch['hbar'] / const_now['hbar']:<12.4f} "
              f"{const_epoch['c'] / const_now['c']:<12.4f} "
              f"{const_epoch['m_p'] / const_now['m_p']:<12.4f}")

# ВЫВОДЫ
print("ВЫВОДЫ")
print(f"""
  1. СВЯЗЬ N И ВРЕМЕНИ:
     t = t₀ · N^(1/3), где t₀ = {t0:.2e} лет
     Это следует из механизма стрелы времени: dS/dt ∼ N^(-1/3)

  2. ЭВОЛЮЦИЯ КОНСТАНТ:
     • Все размерные константы эволюционируют по степенным законам
     • Логарифмические поправки ∼ ln N замедляют эволюцию
     • В современную эпоху относительные изменения ∼10^(-10) за 100 лет

  3. ПРОШЛОЕ:
     • В ранней Вселенной (малые N) константы сильно отличались
     • Квантовые эффекты доминировали (ħ велико)
     • Скорость света была мала
     • Частицы были тяжелее

  4. БУДУЩЕЕ:
     • При t → ∞: ħ → 0 (классический предел)
     • При t → ∞: c → ∞ (неограниченная скорость сигнала)
     • При t → ∞: G → 0 (гравитация исчезает)
     • При t → ∞: m_p → 0 (все частицы становятся безмассовыми)

  5. СТРЕЛА ВРЕМЕНИ:
     dS/dt = N^(-1/3) / t₀ > 0 всегда
     Необратимость сохраняется на всех этапах эволюции
""")