"""
ПОЛНЫЙ СТАТИСТИЧЕСКИЙ АНАЛИЗ ОШИБОК С ОПТИМИЗАЦИЕЙ ПАРАМЕТРОВ
Модель: f = f0 * (1 - sigma / ln N)
"""

import math
import numpy as np
from scipy import stats
from tqdm import tqdm

# ===========================
# ПАРАМЕТРЫ
# ===========================
K = 6.0
pi = math.pi
lnK = math.log(K)

print("ПОЛНЫЙ СТАТИСТИЧЕСКИЙ АНАЛИЗ ОШИБОК С ОПТИМИЗАЦИЕЙ")
print("Модель: f = f0 * (1 - σ / ln N)")
print(f"K = {K}\n")

# ===========================
# ЭКСПЕРИМЕНТАЛЬНЫЕ ЗНАЧЕНИЯ
# ===========================
constants = {
    'ħ': 1.054571817e-34,
    'h': 6.62607015e-34,
    't_P': 5.391247e-44,
    'l_P': 1.616255e-35,
    'm_P': 2.176434e-8,
    'E_P': 1.956082e9,
    'T_P': 1.416784e32,
    'c': 299792458,
    'G': 6.67430e-11,
    'k_B': 1.380649e-23,
    'α': 1 / 137.035999084,
    'm_e': 9.1093837015e-31,
    'ep_0': 8.8725366415e-12,
    'mu_0': 1.25663706127e-6,
    'q_e': 1.602176634e-19,
    'm_proton': 1.67262192e-27,
    'm_muon': 1.883531627e-28,
    'm_pi_meson': 2.4880888e-28,
    'm_pi0_meson': 2.40609e-28,
    'm_k0_meson': 8.801929e-28,
    'm_DT': 3.3435837724e-27,
    'm_Λ_barion': 1.9901611e-27,
    'RIDBERG': 1.097373e7,
    'bor_radius': 5.29177210903e-11,
    'impedance': 376.730313,
    'Φ0_magnetic_stream': 2.06783366752e-15,
    'm_proton_to_m_electron': 1836.152673426,
    'compton_e': 2.426e-12,
    'compton_proton': 1.32140985396e-15,
    'vacuum_higgs': 4.388471e-25,
    'h2_connection_energy': 2.178872e-18,
}


# ===========================
# БАЗОВЫЕ ФОРМУЛЫ
# ===========================
def base_formulas(N):
    """Вычисляет ВСЕ константы без поправок"""
    lnN = math.log(N)
    N13 = N ** (1 / 3)
    p_val = 1 / (K * N13)
    Kp = K * p_val

    return {
        'ħ': (lnN ** 3) / (K * N13),
        'h': 2 * pi * (lnN ** 3) / (K * N13),
        't_P': 4 * K ** 2 * lnK ** 2 / (pi * N13 * lnN ** 2),
        'l_P': 4 * lnN ** 2 * lnK / N13,
        'm_P': K / (pi * 4 * lnN ** 3),
        'E_P': (lnN ** 5) * pi / (4 * K ** 3 * lnK ** 2),
        'T_P': 8 * pi * N13 / (lnN ** 4),
        'c': pi * (lnN ** 4) / (K ** 2 * lnK),
        'G': 16 * pi ** 3 * lnN ** 13 / (K ** 5 * lnK * N13),
        'k_B': Kp * (lnN ** 8) / (8 * pi ** 2),
        'α': 2 * lnK ** 2 / (pi * lnN),
        'm_e': 4 * pi * lnN ** 4 / (K ** (1 / 2) * N13),
        'ep_0': N13 / (8 * pi ** 3 * lnK * lnN ** 20),
        'mu_0': (8 * pi * K ** 4 * lnK ** 3 * lnN ** 12) / N13,
        'q_e': 1.0 / (pi * K ** (3 / 2) * lnN ** 7),
        'm_proton': math.sqrt(pi) * (lnN ** 6) / (K ** (3 / 2) * N13),
        # 'm_muon': 4 * pi ** 2 * lnN ** 5 / (K * 3 ** (1 / 2) * N13),
        # 'm_pi_meson': (lnN) ** 6 / (4 * pi ** 2 * 2 ** (1 / 2)) / N13,
        # 'm_pi0_meson': 2 * pi * K ** 3 * lnN ** 4 / N13,
        # 'm_k0_meson': (lnN ** 6 * (2 * pi) ** (1 / 2)) / (4 * pi ** 2 * N13),
        # 'm_DT': lnN ** 6 * (2 * pi) ** (1 / 2) / (K * 3 ** (1 / 2) * N13),
        # 'm_Λ_barion': (lnN ** 6 * 2 ** (1 / 2)) / (pi ** 2 * N13),
        'RIDBERG': 4 * lnN ** 3 * lnK ** 3 / (pi * K ** (3 / 2)),
        'bor_radius': K ** (3 / 2) / (8 * pi * lnN ** 4 * lnK),
        'impedance': 8 * K ** 2 * pi ** 2 * lnK ** 2 * lnN ** 16 / N13,
        'Φ0_magnetic_stream': lnN ** 10 * pi ** 2 * K ** (1 / 2) / N13,
        # 'm_proton_to_m_electron': lnN ** 2 / (4 * pi ** (1 / 2) * K),
        # 'm_tau_m_electron': K ** (5 / 2) * lnN / (4 * pi ** (1 / 2)),
        # 'm_W_to_m_Z': pi ** (1 / 2) / 2,
        # 'm_plank_to_m_e': K ** (3 / 2) * N13 / (16 * pi ** 2 * lnN ** 7),
        # 'compton_e': K ** (3 / 2) * lnK / (2 * pi * lnN ** 5),
        # 'compton_proton': 2 * K ** (5 / 2) * lnK / (pi ** (1 / 2) * lnN ** 7),
        # 'm_Higgs_to_m_W': 2 * K ** (1 / 2) / pi,
        # 'vacuum_higgs': lnN ** 6 * 8 * pi ** (3 / 2) / (2 ** (1 / 2) * N13),
        # 'h2_connection_energy': 8 * pi * lnN ** 10 * lnK ** 2 / (K ** (9 / 2) * N13),
    }


def evaluate_model(N, start_delta):
    """Оценивает модель для заданных параметров"""
    base = base_formulas(N)
    lnN_val = math.log(N)
    factor = 1.0 + start_delta / lnN_val

    relative_errors = []

    for name in constants.keys():
        if name in base:
            f0 = base[name]
            f_pred = f0 * factor
            f_true = constants[name]

            rel_err = abs(f_pred - f_true) / f_true * 100
            relative_errors.append(rel_err)

    relative_errors = np.array(relative_errors)

    # Метрики качества
    median_error = np.median(relative_errors)
    mean_error = np.mean(relative_errors)
    std_error = np.std(relative_errors, ddof=1)

    # Комбинированная метрика (среднее арифметическое медианы и разброса)
    combined_metric = (median_error + mean_error) / 2 + std_error * 0.5

    return {
        'median_error': median_error,
        'mean_error': mean_error,
        'std_error': std_error,
        'combined_metric': combined_metric,
        'N': N,
        'lnN': lnN_val,
        'start_delta': start_delta,
        'errors': relative_errors
    }


# ===========================
# ОПТИМИЗАЦИЯ
# ===========================
print("=" * 90)
print("ЗАПУСК ОПТИМИЗАЦИИ")
print("=" * 90)

# Параметры сканирования
N_start = 280.038
N_end = 280.044
N_step = 0.0001

delta_start = -0.5
delta_end = 0.5
delta_step = 0.0005

# Вычисление количества шагов
N_steps = int((N_end - N_start) / N_step) + 1
delta_steps = int((delta_end - delta_start) / delta_step) + 1

print(f"Сканирование N: {N_start} → {N_end}, шаг {N_step}")
print(f"Количество точек N: {N_steps}")
print(f"Сканирование σ: {delta_start} → {delta_end}, шаг {delta_step}")
print(f"Количество точек σ: {delta_steps}")
print(f"Всего комбинаций: {N_steps * delta_steps:,}")
print()

# Хранение лучших результатов для каждого N
best_for_each_N = []  # Лучший по комбинированной метрике для каждого N
all_candidates = []  # Все кандидаты, которые улучшают предыдущие результаты

# Глобальные оптимумы
best_median_error = float('inf')
best_std_error = float('inf')
best_combined = float('inf')

# Создаем массивы для эффективного поиска
N_values = np.linspace(N_start, N_end, N_steps)
delta_values = np.linspace(delta_start, delta_end, delta_steps)

# Прогресс-бар для N
for i, lnN in enumerate(tqdm(N_values, desc="Сканирование N")):
    N_actual = math.exp(lnN)
    best_for_this_N = None
    best_combined_this_N = float('inf')

    # Для текущего N ищем лучшую delta
    for delta in delta_values:
        result = evaluate_model(N_actual, delta)

        # Обновляем лучший для этого N
        if result['combined_metric'] < best_combined_this_N:
            best_combined_this_N = result['combined_metric']
            best_for_this_N = result

        # Проверяем на глобальный рекорд
        if (result['median_error'] < best_median_error * 0.99 or
                result['std_error'] < best_std_error * 0.99 or
                result['combined_metric'] < best_combined * 0.99):
            all_candidates.append(result)

            if result['median_error'] < best_median_error:
                best_median_error = result['median_error']
            if result['std_error'] < best_std_error:
                best_std_error = result['std_error']
            if result['combined_metric'] < best_combined:
                best_combined = result['combined_metric']

    if best_for_this_N:
        best_for_each_N.append(best_for_this_N)

# Если кандидатов мало, берем лучшие для каждого N
if len(all_candidates) < 20:
    all_candidates = best_for_each_N

# Сортируем по разным критериям
sorted_by_median = sorted(all_candidates, key=lambda x: x['median_error'])
sorted_by_std = sorted(all_candidates, key=lambda x: x['std_error'])
sorted_by_combined = sorted(all_candidates, key=lambda x: x['combined_metric'])

# ===========================
# ВЫВОД РЕЗУЛЬТАТОВ
# ===========================
print("\n" + "=" * 90)
print("ТОП-10 КАНДИДАТОВ ПО МИНИМУМУ МЕДИАННОЙ ОШИБКИ")
print("=" * 90)

print(
    f"\n{'Ранг':<6} {'N':<15} {'ln N':<15} {'σ':<12} {'Медиана %':<14} {'Среднее %':<14} {'Стд %':<12} {'Комб. метрика':<15}")
print("-" * 105)

for i, candidate in enumerate(sorted_by_median[:10]):
    print(f"{i + 1:<6} {candidate['N']:<15.6e} {candidate['lnN']:<15.10f} "
          f"{candidate['start_delta']:<12.6f} {candidate['median_error']:<14.6f} "
          f"{candidate['mean_error']:<14.6f} {candidate['std_error']:<12.6f} "
          f"{candidate['combined_metric']:<15.6f}")

print("\n" + "=" * 90)
print("ТОП-10 КАНДИДАТОВ ПО МИНИМУМУ СТАНДАРТНОГО ОТКЛОНЕНИЯ (РАЗБРОСА)")
print("=" * 90)

print(
    f"\n{'Ранг':<6} {'N':<15} {'ln N':<15} {'σ':<12} {'Медиана %':<14} {'Среднее %':<14} {'Стд %':<12} {'Комб. метрика':<15}")
print("-" * 105)

for i, candidate in enumerate(sorted_by_std[:10]):
    print(f"{i + 1:<6} {candidate['N']:<15.6e} {candidate['lnN']:<15.10f} "
          f"{candidate['start_delta']:<12.6f} {candidate['median_error']:<14.6f} "
          f"{candidate['mean_error']:<14.6f} {candidate['std_error']:<12.6f} "
          f"{candidate['combined_metric']:<15.6f}")

print("\n" + "=" * 90)
print("ТОП-10 КАНДИДАТОВ ПО КОМБИНИРОВАННОЙ МЕТРИКЕ")
print("(Минимум среднего между медианой и средним + 0.5×стд)")
print("=" * 90)

print(
    f"\n{'Ранг':<6} {'N':<15} {'ln N':<15} {'σ':<12} {'Медиана %':<14} {'Среднее %':<14} {'Стд %':<12} {'Комб. метрика':<15}")
print("-" * 105)

for i, candidate in enumerate(sorted_by_combined[:10]):
    print(f"{i + 1:<6} {candidate['N']:<15.6e} {candidate['lnN']:<15.10f} "
          f"{candidate['start_delta']:<12.6f} {candidate['median_error']:<14.6f} "
          f"{candidate['mean_error']:<14.6f} {candidate['std_error']:<12.6f} "
          f"{candidate['combined_metric']:<15.6f}")

# ===========================
# ДЕТАЛЬНЫЙ АНАЛИЗ ЛУЧШЕГО КАНДИДАТА
# ===========================
best_candidate = sorted_by_combined[0]

print("\n" + "=" * 90)
print("ДЕТАЛЬНЫЙ АНАЛИЗ ЛУЧШЕГО КАНДИДАТА (по комбинированной метрике)")
print("=" * 90)

print(f"""
ПАРАМЕТРЫ:
  N = {best_candidate['N']:.10f}
  ln N = {best_candidate['lnN']:.10f}
  σ = {best_candidate['start_delta']:.6f}
  Фактор поправки: 1 + {best_candidate['start_delta']:.6f}/{best_candidate['lnN']:.10f} = {1 + best_candidate['start_delta'] / best_candidate['lnN']:.10f}

СТАТИСТИКИ:
  Медианная ошибка: {best_candidate['median_error']:.6f}%
  Средняя ошибка:   {best_candidate['mean_error']:.6f}%
  Стд. отклонение:  {best_candidate['std_error']:.6f}%
""")

# Детальный вывод всех констант для лучшего кандидата
base = base_formulas(best_candidate['N'])
factor = 1.0 + best_candidate['start_delta'] / best_candidate['lnN']

print(f"\n{'Константа':<30} {'Предсказание':<20} {'CODATA':<20} {'Ошибка %':<15}")
print("-" * 90)

for name in sorted(constants.keys()):
    if name in base:
        f_pred = base[name] * factor
        f_true = constants[name]
        rel_err = abs(f_pred - f_true) / f_true * 100

        status = "⭐⭐⭐" if rel_err < 0.05 else ("⭐⭐" if rel_err < 0.2 else ("⭐" if rel_err < 0.5 else "⚠️"))
        print(f"{name:<30} {f_pred:<20.6e} {f_true:<20.6e} {rel_err:<15.6f} {status}")

print("\n" + "=" * 90)
print("ОПТИМИЗАЦИЯ ЗАВЕРШЕНА")
print("=" * 90)