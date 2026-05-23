"""
ПОЛНЫЙ СТАТИСТИЧЕСКИЙ АНАЛИЗ ОШИБОК
Модель: f = f0 * (1 - sigma / ln N)
"""

import math
import numpy as np
from scipy import stats

# ПАРАМЕТРЫ
K = 6.0
pi = math.pi
lnK = math.log(K)

# Оптимальные параметры (из предыдущего анализа)
N_opt =1 # резонанса 4.203950e+121   #4.176490e+121
sigma_opt = 0.07  # Оптимальное значение из сканирования (|σ| ≈ 0.07)

lnN_opt = 280.047#math.log(N_opt)
N_opt = math.exp(lnN_opt)
print("=" * 90)
print("ПОЛНЫЙ СТАТИСТИЧЕСКИЙ АНАЛИЗ ОШИБОК")
print("Модель: f = f0 * (1 - σ / ln N)")
print("=" * 90)
print(f"""
  ПАРАМЕТРЫ:
    K = {K}
    N = {N_opt:.6e}
    ln N = {lnN_opt:.10f}
    σ = {sigma_opt}
    Поправка: 1 - {sigma_opt}/{lnN_opt:.1f} = {1 - sigma_opt/lnN_opt:.10f}
""")

# ЭКСПЕРИМЕНТАЛЬНЫЕ ЗНАЧЕНИЯ
constants = {
    'ħ': 1.054571817e-34,
    'h': 6.62607015e-34,
    't_P': 5.391247e-44,
    'l_P': 1.616255e-35,
    'm_P': 2.176434e-8,
    'E_P': 1.956082e9,
    'T_P': 1.416784e32,
    'c': 299792458,
    #'G': 6.67430e-11,
    'k_B': 1.380649e-23,
    'α': 1/137.035999084,
    'm_e': 9.1093837015e-31,
    'ep_0': 8.8725366415e-12,
    'mu_0': 1.25663706127e-6,
    'q_e': 1.602176634e-19,
    'm_proton': 1.67262192e-27,
    # 'm_muon': 1.883531627e-28,
    # 'm_pi_meson': 2.4880888e-28,
    # 'm_pi0_meson': 2.40609e-28,
    # 'm_k0_meson': 8.801929e-28,
    # 'm_DT': 3.3435837724e-27,
    # 'm_Λ_barion': 1.9901611e-27,
    # 'RIDBERG': 1.097373e7,
    # 'bor_radius': 5.29177210903e-11,
    # 'impedance': 376.730313,
    # 'Φ0_magnetic_stream': 2.06783366752e-15,
    # 'm_proton_to_m_electron': 1836.152673426,
    # #'m_tau_m_electron': 3477,
    # #'m_W_to_m_Z': 0.8815,
    # 'm_plank_to_m_e': 2.389e22,
    # 'compton_e': 2.426e-12,
    # 'compton_proton': 1.32140985396e-15,
    # #'m_Higgs_to_m_W': 1.558,
    # 'vacuum_higgs': 4.388471e-25,
    # 'h2_connection_energy': 2.178872e-18,
}

# БАЗОВЫЕ ФОРМУЛЫ
def base_formulas(N):
    """Вычисляет ВСЕ константы без поправок"""
    lnN = math.log(N)
    N13 = N ** (1/3)
    p_val = 1/(K*N13)
    Kp = K * p_val

    return {
        'ħ': (lnN ** 3) / (K * N13),
        'h': 2 * pi * (lnN ** 3) / (K * N13),
        't_P': 4 * K**2 * lnK**2 / (pi * N13 * lnN**2),
        'l_P': 4 * lnN ** 2 * lnK / N13,
        'm_P': K / (pi * 4 * lnN**3),
        'E_P': (lnN ** 5) * pi / (4 * K**3 * lnK**2),
        'T_P': 8 * pi * N13 / (lnN**4),
        'c': pi * (lnN ** 4) / (K**2 * lnK),
        'G': 16 * pi**3 * lnN**13 / (K**5 * lnK * N13),
        'k_B': Kp * (lnN**8) / (8 * pi**2),
        'α': 2 * lnK**2 / (pi * lnN),
        'm_e': 4*pi * lnN**4 / (K**(1/2) * N13),
        'ep_0': N13 / (8 * pi**3 * lnK * lnN**20),
        'mu_0': (8 * pi * K**4 * lnK**3 * lnN**12) / N13,
        'q_e': 1.0 / (pi * K**(3/2) * lnN**7),
        'm_proton': math.sqrt(pi) * (lnN**6) / (K**(3/2) * N13),
        'm_muon': 4*pi**2 * lnN**5 / (K * 3**(1/2) * N13),
        'm_pi_meson': (lnN)**6 / (4*pi**2 * 2**(1/2)) / N13,
        'm_pi0_meson': 2 * pi * K**3 * lnN**4 / N13,
        'm_k0_meson': (lnN**6 * (2*pi)**(1/2)) / (4*pi**2 * N13),
        'm_DT': lnN**6 * (2*pi)**(1/2) / (K * 3**(1/2) * N13),
        'm_Λ_barion': (lnN**6 * 2**(1/2)) / (pi**2 * N13),
        'RIDBERG': 4 * lnN**3 * lnK**3 / (pi * K**(3/2)),
        'bor_radius': K**(3/2) / (8 * pi * lnN**4 * lnK),
        'impedance': 8 * K**2 * pi**2 * lnK**2 * lnN**16 / N13,
        'Φ0_magnetic_stream': lnN**10 * pi**2 * K**(1/2) / N13,
        'm_proton_to_m_electron': lnN**2 / (4 * pi**(1/2) * K),
        'm_tau_m_electron': K**(5/2) * lnN / (4 * pi**(1/2)),
        'm_W_to_m_Z': pi**(1/2) / 2,
        'm_plank_to_m_e': K**(3/2) * N13 / (16 * pi**2 * lnN**7),
        'compton_e': K**(3/2) * lnK / (2 * pi * lnN**5),
        'compton_proton': 2 * K**(5/2) * lnK / (pi**(1/2) * lnN**7),
        'm_Higgs_to_m_W': 2 * K**(1/2) / pi,
        'h2_connection_energy': 8 * pi * lnN**10 * lnK**2 / (K**(9/2) * N13),
    }

# ВЫЧИСЛЕНИЕ ОШИБОК
base = base_formulas(N_opt)
lnN_val = math.log(N_opt)
factor = 1.0 +  0.3555/ lnN_val
# для  N из резонанса сигма  оптимальная sigma 6.1
# ┌─────────────────────────────────────────────────────┐
# │ Средняя
# ошибка: 1.3095 % │
# │ Медианная
# ошибка: 0.2575 % │
# │ Стандартное
# отклонение: 1.5062 % │
# │ Мин / Макс
# ошибка: 0.0198 % / 4.4984 % │
# └─────────────────────────────────────────────────────┘

# для N из дзета функции минмально при +0.312
# Средняя ошибка:                         0.2622% │
#   │ Медианная ошибка:                       0.1537% │
#   │ Стандартное отклонение:                 0.1965%

# N - глобальный оптимум: 280.098 при +  4.85
# ┌─────────────────────────────────────────────────────┐
# │ Средняя
# ошибка: 1.0646 % │
# │ Медианная
# ошибка: 0.2295 % │
# │ Стандартное
# отклонение: 1.2022 % │
# │ Мин / Макс
# ошибка: 0.0286 % / 3.6312 % │
# └─────────────────────────────────────────────────────┘

# для N из paperfolding
#Paperfolding	280.3946
# ┌─────────────────────────────────────────────────────┐
#   │ Средняя ошибка:                         8.1006% │
#   │ Медианная ошибка:                       1.8341% │
#   │ Стандартное отклонение:                 8.8260% │
#   │ Мин / Макс ошибка:                    0.3985% / 26.3551% │
#   └─────────────────────────────────────────────────────┘

# Массивы ошибок
relative_errors_pct = []  # Относительные ошибки в процентах
log_errors = []           # Логарифмические ошибки
abs_errors = []           # Абсолютные ошибки
names = []                # Имена констант
predictions = []          # Предсказанные значения
true_values = []          # Истинные значения

for name in constants.keys():
    if name in base:
        f0 = base[name]
        f_pred = f0 * factor
        f_true = constants[name]

        rel_err = abs(f_pred - f_true) / f_true * 100
        log_err = math.log(f_pred / f_true)
        abs_err = abs(f_pred - f_true)

        relative_errors_pct.append(rel_err)
        log_errors.append(log_err)
        abs_errors.append(abs_err)
        names.append(name)
        predictions.append(f_pred)
        true_values.append(f_true)

relative_errors_pct = np.array(relative_errors_pct)
log_errors = np.array(log_errors)
abs_errors = np.array(abs_errors)

# 1. БАЗОВАЯ СТАТИСТИКА
print("1. БАЗОВАЯ СТАТИСТИКА")

n = len(relative_errors_pct)
mean_rel = np.mean(relative_errors_pct)
median_rel = np.median(relative_errors_pct)
std_rel = np.std(relative_errors_pct, ddof=1)  # Выборочное стандартное отклонение
min_rel = np.min(relative_errors_pct)
max_rel = np.max(relative_errors_pct)

mean_log = np.mean(log_errors)
std_log = np.std(log_errors, ddof=1)

print(f"""
  Число констант (n):              {n}
  
  Относительные ошибки (%):
    Среднее (mean):                 {mean_rel:.6f}%
    Медиана (median):               {median_rel:.6f}%
    Стд. отклонение (std):          {std_rel:.6f}%
    Минимум:                        {min_rel:.6f}%
    Максимум:                       {max_rel:.6f}%
    Размах (max - min):             {max_rel - min_rel:.6f}%
    
  Логарифмические ошибки:
    Среднее:                        {mean_log:.6e}
    Стд. отклонение:                {std_log:.6e}
""")


# Корреляция между ошибкой и величиной константы
log_true_values = np.log10(np.abs(true_values))
correlation_pearson, p_pearson = stats.pearsonr(relative_errors_pct, log_true_values)
correlation_spearman, p_spearman = stats.spearmanr(relative_errors_pct, log_true_values)

print(f"""
  Корреляция ошибки с порядком величины константы (log10):
    Пирсон:   r = {correlation_pearson:.6f}, p = {p_pearson:.6f}
    Спирмен:  ρ = {correlation_spearman:.6f}, p = {p_spearman:.6f}
    
  {'Значимая корреляция' if p_pearson < 0.05 else 'Нет значимой корреляции'}
  {'(ошибка зависит от масштаба константы)' if p_pearson < 0.05 else '(ошибка не зависит от масштаба)'}
""")


print(f"""
  МОДЕЛЬ: f = f0 * (1 - σ/ln N)
  ПАРАМЕТРЫ: N = {N_opt:.4e}, σ = {sigma_opt}
  
  ОСНОВНЫЕ МЕТРИКИ:
  ┌─────────────────────────────────────────────────────┐
  │ Средняя ошибка:                     {mean_rel:>10.4f}% │
  │ Медианная ошибка:                   {median_rel:>10.4f}% │
  │ Стандартное отклонение:             {std_rel:>10.4f}% │
  │ Мин / Макс ошибка:                  {min_rel:>8.4f}% / {max_rel:.4f}% │
  └─────────────────────────────────────────────────────┘
  """)


# Сортируем по ошибке
sorted_indices = np.argsort(relative_errors_pct)

print(f"  {'Константа':<25} {'Предсказание':<18} {'CODATA':<18} {'Ошибка %':<12} {'Z-score':<10} {'Статус':<8}")
print(f"  {'─' * 91}")

for i in sorted_indices:
    name = names[i]
    pred = predictions[i]
    true = true_values[i]
    err = relative_errors_pct[i]

    if err < 0.05:
        status = "⭐⭐⭐"
    elif err < 0.2:
        status = "⭐⭐"
    elif err < 0.5:
        status = "⭐"
    else:
        status = "⚠️"

    print(f"  {name:<25} {pred:<18.6e} {true:<18.6e} {err:<12.6f}  {status:<8}")
