import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Все данные
data = {
    'Квантовые и Планковские': {
        'ħ': 1.054327e-34, 'h': 6.624531e-34, 't_P': 5.404624e-44,
        'l_P': 1.618973e-35, 'm_P': 2.174010e-08, 'E_P': 1.950787e+09,
        'T_P': 1.418687e+32
    },
    'Фундаментальные': {
        'c': 2.995533e+08, 'G': 6.682305e-11, 'k_B': 1.379971e-23,
        'α': 7.298152e-03, 'q_e': 1.603382e-19
    },
    'Массы частиц': {
        'm_e': 9.088399e-31, 'm_proton': 1.675539e-27, 'm_muon': 1.884637e-28,
        'm_pi_meson': 2.488466e-28, 'm_pi0_meson': 2.404290e-28, 'm_k0_meson': 8.821381e-28,
        'm_DT': 3.351077e-27, 'm_Λ_barion': 1.990772e-27, 'm_z_bozon': 1.620279e-25,
        'm_w_bozon': 1.435935e-25, 'm_Higgs': 2.239188e-25, 'm_D0': 3.351077e-27,
        'm_J_ψ': 5.539678e-27, 'm_eta': 9.792859e-28, 'm_Υ_1S': 1.701579e-26,
        'm_qu_u': 2.176610e-30, 'm_qu_d': 4.773840e-30, 'm_qu_s': 9.735936e-30,
        'm_qu_c': 1.269645e-27, 'm_qu_b': 4.199952e-27, 'm_qu_t': 3.040608e-25,
        'Sigma_plus': 2.111533e-27, 'Ksi_0': 2.369569e-27, 'omega_minus': 2.969815e-27,
        'Ksi_plus': 4.422385e-27, 'Omega0_c': 4.765220e-27, 'lambda_B0': 1.005323e-26,
        'Ksi_minus': 2.369569e-27, 'Sigma_minus': 2.111533e-27,
        'phi_meson': 1.827665e-27, 'omega_meson': 1.384919e-27, 'eta_shtrih': 1.705508e-27,
        'm_neitrino': 1.782934e-36
    },
    'Космология': {
        'Lambda_cosmo': 1.089025e-52, 'Einstein_constant': 2.085786e-43,
        'vacuum_higgs': 4.376291e-25, 'h2_connection_energy': 2.171858e-18
    },
    'Времена жизни': {
        'mu_lifetime': 2.198440e-06, 'tau_lifetime': 2.902932e-13,
        'pion_lifetime': 2.600524e-08, 'neutron_lifetime': 8.789688e+02,
        'kaon_lifetime': 1.239234e-08, 'D_plus_lifetime': 1.040257e-12,
        'B_plus_lifetime': 1.634054e-12, 'Λ_b_lifetime': 1.471166e-12,
        'D0_lifetime': 4.099595e-13
    },
    'Безразмерные отношения': {
        'm_proton_to_m_e': 1.843601e+03, 'm_tau_m_e': 3.483131e+03,
        'm_W_to_m_Z': 8.862269e-01, 'm_plank_to_m_e': 2.392071e+22,
        'm_Higgs_to_m_W': 1.559394e+00, 'impedance': 3.761181e+02,
        'bor_radius': 5.306411e-11, 'compton_e': 2.433289e-12,
        'compton_proton': 1.319856e-15, 'Φ0_magnetic_stream': 2.065800e-15,
        'RIDBERG': 1.094466e+07, 'ep_0': 8.875681e-12, 'mu_0': 1.255596e-06
    }
}

# Экспериментальные значения
exp_data = {
    'Квантовые и Планковские': {
        'ħ': 1.054572e-34, 'h': 6.626070e-34, 't_P': 5.391247e-44,
        'l_P': 1.616255e-35, 'm_P': 2.176434e-08, 'E_P': 1.956082e+09,
        'T_P': 1.416784e+32
    },
    'Фундаментальные': {
        'c': 2.997925e+08, 'G': 6.674300e-11, 'k_B': 1.380649e-23,
        'α': 7.297353e-03, 'q_e': 1.602177e-19
    },
    'Массы частиц': {
        'm_e': 9.109384e-31, 'm_proton': 1.672622e-27, 'm_muon': 1.883532e-28,
        'm_pi_meson': 2.488089e-28, 'm_pi0_meson': 2.406090e-28, 'm_k0_meson': 8.801929e-28,
        'm_DT': 3.343584e-27, 'm_Λ_barion': 1.990161e-27, 'm_z_bozon': 1.626140e-25,
        'm_w_bozon': 1.433620e-25, 'm_Higgs': 2.233190e-25, 'm_D0': 3.324790e-27,
        'm_J_ψ': 5.520610e-27, 'm_eta': 9.767732e-28, 'm_Υ_1S': 1.687150e-26,
        'm_qu_u': 2.165000e-30, 'm_qu_d': 4.791500e-30, 'm_qu_s': 9.635000e-30,
        'm_qu_c': 1.270000e-27, 'm_qu_b': 4.180000e-27, 'm_qu_t': 3.040000e-25,
        'Sigma_plus': 2.119330e-27, 'Ksi_0': 2.345320e-27, 'omega_minus': 2.985900e-27,
        'Ksi_plus': 4.399500e-27, 'Omega0_c': 4.808000e-27, 'lambda_B0': 1.002300e-26,
        'Ksi_minus': 2.358000e-27, 'Sigma_minus': 2.132000e-27,
        'phi_meson': 1.819000e-27, 'omega_meson': 1.394000e-27, 'eta_shtrih': 1.708600e-27,
        'm_neitrino': 1.783000e-36
    },
    'Космология': {
        'Lambda_cosmo': 1.089290e-52, 'Einstein_constant': 2.076647e-43,
        'vacuum_higgs': 4.388471e-25, 'h2_connection_energy': 2.178872e-18
    },
    'Времена жизни': {
        'mu_lifetime': 2.196981e-06, 'tau_lifetime': 2.903000e-13,
        'pion_lifetime': 2.603300e-08, 'neutron_lifetime': 8.778000e+02,
        'kaon_lifetime': 1.238000e-08, 'D_plus_lifetime': 1.040000e-12,
        'B_plus_lifetime': 1.638000e-12, 'Λ_b_lifetime': 1.471000e-12,
        'D0_lifetime': 4.101000e-13
    },
    'Безразмерные отношения': {
        'm_proton_to_m_e': 1.836153e+03, 'm_tau_m_e': 3.477000e+03,
        'm_W_to_m_Z': 8.815000e-01, 'm_plank_to_m_e': 2.389000e+22,
        'm_Higgs_to_m_W': 1.558000e+00, 'impedance': 3.767303e+02,
        'bor_radius': 5.291772e-11, 'compton_e': 2.426000e-12,
        'compton_proton': 1.321410e-15, 'Φ0_magnetic_stream': 2.067834e-15,
        'RIDBERG': 1.097373e+07, 'ep_0': 8.872537e-12, 'mu_0': 1.256637e-06
    }
}

# КОНСОЛЬНЫЙ ВЫВОД
print("ПОЛНЫЙ АНАЛИЗ ОТКЛОНЕНИЙ ЕТИ ОТ ЭКСПЕРИМЕНТА")

all_deviations = []
all_abs_deviations = []
all_names_list = []
all_categories_list = []

for category in data.keys():
    print(f"КАТЕГОРИЯ: {category}")
    print(f"{'Константа':<30} {'Эксперимент':<20} {'ЕТИ':<20} {'Отклонение %':<15} {'|Δ| %':<12} {'Направление'}")
    print("-" * 120)

    devs = []
    for name in data[category]:
        eti_val = data[category][name]
        exp_val = exp_data[category][name]
        dev = (eti_val - exp_val) / exp_val * 100

        devs.append(dev)
        all_deviations.append(dev)
        all_abs_deviations.append(abs(dev))
        all_names_list.append(name)
        all_categories_list.append(category)

        direction = "▲ ВЫШЕ" if dev > 0 else ("▼ НИЖЕ" if dev < 0 else "● РАВНО")
        bar = "█" * int(min(abs(dev) * 20, 50))

        print(f"{name:<30} {exp_val:<20.6e} {eti_val:<20.6e} {dev:<+15.6f} {abs(dev):<12.6f} {direction} {bar}")

    # Статистика по категории
    devs_arr = np.array(devs)
    mean_d = np.mean(devs_arr)
    std_d = np.std(devs_arr)
    median_d = np.median(devs_arr)
    rms_d = np.sqrt(np.mean(devs_arr ** 2))
    max_pos = np.max(devs_arr)
    max_neg = np.min(devs_arr)
    skew = stats.skew(devs_arr)
    kurt = stats.kurtosis(devs_arr)

    print(f"\n  СТАТИСТИКА ПО КАТЕГОРИИ:")
    print(f"  Число констант:        {len(devs)}")
    print(f"  Среднее отклонение:    {mean_d:+.4f}%")
    print(f"  Медиана:                {median_d:+.4f}%")
    print(f"  Стандартное откл.:     {std_d:.4f}%")
    print(f"  RMS отклонение:        {rms_d:.4f}%")
    print(f"  Макс. положительное:   {max_pos:+.4f}%")
    print(f"  Макс. отрицательное:   {max_neg:+.4f}%")
    print(f"  Асимметрия:            {skew:+.4f}")
    print(f"  Эксцесс:               {kurt:+.4f}")
    print(f"  Число с |Δ| < 0.1%:   {np.sum(np.abs(devs_arr) < 0.1)}")
    print(f"  Число с |Δ| < 0.5%:   {np.sum(np.abs(devs_arr) < 0.5)}")
    print(f"  Число с |Δ| < 1.0%:   {np.sum(np.abs(devs_arr) < 1.0)}")
    print(f"  Число Δ > 0 (выше):   {np.sum(devs_arr > 0)}")
    print(f"  Число Δ < 0 (ниже):   {np.sum(devs_arr < 0)}")

# ГЛОБАЛЬНАЯ СТАТИСТИКА
all_devs = np.array(all_deviations)
all_abs_devs = np.array(all_abs_deviations)

print("ГЛОБАЛЬНАЯ СТАТИСТИКА ПО ВСЕМ КОНСТАНТАМ")
print(f"  Общее число констант:  {len(all_devs)}")
print(f"  Среднее отклонение:    {np.mean(all_devs):+.4f}%")
print(f"  Медиана:                {np.median(all_devs):+.4f}%")
print(f"  Стандартное откл.:     {np.std(all_devs):.4f}%")
print(f"  RMS отклонение:        {np.sqrt(np.mean(all_devs ** 2)):.4f}%")
print(f"  Среднее |Δ|:            {np.mean(all_abs_devs):.4f}%")
print(f"  Медиана |Δ|:            {np.median(all_abs_devs):.4f}%")
print(f"  Макс. положительное:   {np.max(all_devs):+.4f}%")
print(f"  Макс. отрицательное:   {np.min(all_devs):+.4f}%")
print(f"  Асимметрия:            {stats.skew(all_devs):+.4f}")
print(f"  Эксцесс:               {stats.kurtosis(all_devs):+.4f}")
print(f"  Число с |Δ| < 0.1%:   {np.sum(all_abs_devs < 0.1)}")
print(f"  Число с |Δ| < 0.5%:   {np.sum(all_abs_devs < 0.5)}")
print(f"  Число с |Δ| < 1.0%:   {np.sum(all_abs_devs < 1.0)}")
print(f"  Число Δ > 0 (выше):   {np.sum(all_devs > 0)}")
print(f"  Число Δ < 0 (ниже):   {np.sum(all_devs < 0)}")

# Корреляция отклонений с величиной константы
print("АНАЛИЗ КОРРЕЛЯЦИИ ОТКЛОНЕНИЙ С ВЕЛИЧИНОЙ КОНСТАНТЫ")

log_exp_vals = []
for category in data.keys():
    for name in data[category]:
        log_exp_vals.append(np.log10(abs(exp_data[category][name])))

log_exp_vals = np.array(log_exp_vals)
correlation = np.corrcoef(log_exp_vals, all_devs)[0, 1]
print(f"  Корреляция отклонения с log10(exp): {correlation:+.4f}")
print(f"  (Отрицательная → большие константы имеют тенденцию к отрицательному отклонению)")

# Распределение отклонений по квантилям
print("РАСПРЕДЕЛЕНИЕ ОТКЛОНЕНИЙ ПО ДИАПАЗОНАМ")
bins = [0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 100.0]
print(f"  {'Диапазон |Δ|':<20} {'Число':<10} {'%':<10} {'Кумулятивно':<15}")
print(f"  {'-' * 55}")
cumulative = 0
for i in range(len(bins) - 1):
    count = np.sum((all_abs_devs >= bins[i]) & (all_abs_devs < bins[i + 1]))
    cumulative += count
    pct = count / len(all_devs) * 100
    cum_pct = cumulative / len(all_devs) * 100
    print(f"  {bins[i]:.2f}% – {bins[i + 1]:.2f}%        {count:<10} {pct:<10.2f} {cum_pct:<15.2f}")

# Топ-10 наибольших отклонений
print("ТОП-10 НАИБОЛЬШИХ ОТКЛОНЕНИЙ (ПО АБСОЛЮТНОЙ ВЕЛИЧИНЕ)")
print(f"  {'Ранг':<6} {'Константа':<30} {'Категория':<25} {'Отклонение %':<15}")
print(f"  {'-' * 80}")

sorted_indices = np.argsort(all_abs_devs)[::-1]
for rank, idx in enumerate(sorted_indices[:10]):
    print(f"  {rank + 1:<6} {all_names_list[idx]:<30} {all_categories_list[idx]:<25} {all_deviations[idx]:+15.6f}")

# Топ-10 наименьших отклонений
print("ТОП-10 НАИМЕНЬШИХ ОТКЛОНЕНИЙ (НАИБОЛЕЕ ТОЧНЫЕ ПРЕДСКАЗАНИЯ)")
print(f"  {'Ранг':<6} {'Константа':<30} {'Категория':<25} {'Отклонение %':<15}")
print(f"  {'-' * 80}")

sorted_indices_asc = np.argsort(all_abs_devs)
for rank, idx in enumerate(sorted_indices_asc[:10]):
    print(f"  {rank + 1:<6} {all_names_list[idx]:<30} {all_categories_list[idx]:<25} {all_deviations[idx]:+15.6f}")

# Сравнение с нормальным распределением
print("ТЕСТ НА НОРМАЛЬНОСТЬ РАСПРЕДЕЛЕНИЯ ОТКЛОНЕНИЙ")
stat_shapiro, p_shapiro = stats.shapiro(all_devs)
print(f"  Тест Шапиро-Уилка: статистика = {stat_shapiro:.4f}, p-value = {p_shapiro:.6f}")
if p_shapiro > 0.05:
    print(f"  ✅ Распределение НЕ ОТЛИЧАЕТСЯ значимо от нормального")
else:
    print(f"  ⚠️  Распределение ОТЛИЧАЕТСЯ от нормального (p < 0.05)")

# Тест на систематический сдвиг (t-тест)
t_stat, p_ttest = stats.ttest_1samp(all_devs, 0)
print(f"\n  T-тест на систематический сдвиг (H0: среднее = 0):")
print(f"  t-статистика = {t_stat:.4f}, p-value = {p_ttest:.6f}")
if p_ttest > 0.05:
    print(f"  ✅ СИСТЕМАТИЧЕСКИЙ СДВИГ НЕ ОБНАРУЖЕН (p > 0.05)")
else:
    direction = "положительный" if np.mean(all_devs) > 0 else "отрицательный"
    print(f"  ⚠️  ОБНАРУЖЕН {direction} СИСТЕМАТИЧЕСКИЙ СДВИГ (p < 0.05)")

# Анализ по типам частиц
print("АНАЛИЗ ПО ТИПАМ ЧАСТИЦ")

particle_types = {
    'Лептоны': ['m_e', 'm_muon', 'm_neitrino'],
    'Мезоны': ['m_pi_meson', 'm_pi0_meson', 'm_k0_meson', 'm_DT', 'm_D0',
               'm_J_ψ', 'm_eta', 'm_Υ_1S', 'phi_meson', 'omega_meson', 'eta_shtrih'],
    'Барионы': ['m_proton', 'm_Λ_barion', 'Sigma_plus', 'Ksi_0', 'omega_minus',
                'Ksi_plus', 'Omega0_c', 'lambda_B0', 'Ksi_minus', 'Sigma_minus'],
    'Кварки': ['m_qu_u', 'm_qu_d', 'm_qu_s', 'm_qu_c', 'm_qu_b', 'm_qu_t'],
    'Бозоны': ['m_z_bozon', 'm_w_bozon', 'm_Higgs']
}

for ptype, names in particle_types.items():
    devs_ptype = []
    for name in names:
        for category in data.keys():
            if name in data[category]:
                dev = (data[category][name] - exp_data[category][name]) / exp_data[category][name] * 100
                devs_ptype.append(dev)
                break

    if devs_ptype:
        devs_ptype = np.array(devs_ptype)
        print(f"  {ptype:<15}: n={len(devs_ptype):<3} среднее={np.mean(devs_ptype):+.4f}%  "
              f"медиана={np.median(devs_ptype):+.4f}%  σ={np.std(devs_ptype):.4f}%  "
              f"min={np.min(devs_ptype):+.4f}%  max={np.max(devs_ptype):+.4f}%")