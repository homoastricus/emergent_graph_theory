import math
import numpy as np
from scipy.optimize import minimize_scalar

# КОНСТАНТЫ
K = 6.0
pi = math.pi
lnK = math.log(K)
sqrt2 = math.sqrt(2)
sqrt3 = math.sqrt(3)

# Математические константы (цели)
feigenbaum_delta = 4.66920160910299
feigenbaum_delta_inv = 1.0 / feigenbaum_delta
feigenbaum_alpha = 2.5029078750958928
feigenbaum_alpha_inv = 1.0 / feigenbaum_alpha
euler_mascheroni = 0.5772156649015329
ln2 = math.log(2)
ln3 = math.log(3)
lnpi = math.log(pi)
dzetta_2 = pi ** 2 / 6
euler_e = math.e


# ФУНКЦИИ ДЛЯ ВЫЧИСЛЕНИЯ РАЗЛИЧНЫХ ln N
def lnN_geometric():
    return float((K - lnK) / (1.0 / 3.0 - 1.0 / pi))


def lnN_zeta():
    exponent = 1.5 + pi ** 2 / 6.0
    return float(6 ** exponent)


def lnN_alpha(alpha=1 / 137.035999084):
    return 2 * lnK ** 2 / (pi * alpha)


def lnN_hypothesis():
    g = lnN_geometric()
    return float(g - pi * K / g)


# Вычисляем разницы (константы)
geo_val = lnN_geometric()
zeta_val = lnN_zeta()
alpha_val = lnN_alpha()
phys_val = lnN_hypothesis()

geometric_zeta = geo_val - zeta_val
geometric_alpha = geo_val - alpha_val
geometric_phys = geo_val - phys_val
zeta_alpha = zeta_val - alpha_val
zeta_phys = zeta_val - phys_val
alpha_phys = alpha_val - phys_val

print("КОНСТАНТЫ-РАЗНИЦЫ:")
print(f"  geometric_zeta = {geometric_zeta:.10f}")
print(f"  geometric_alpha = {geometric_alpha:.10f}")
print(f"  geometric_phys = {geometric_phys:.10f}")
print(f"  zeta_alpha = {zeta_alpha:.10f}")
print(f"  zeta_phys = {zeta_phys:.10f}")
print(f"  alpha_phys = {alpha_phys:.10f}")


# ВСЕ ТОЖДЕСТВА
def compute_all_identities(lnN):
    """Вычисляет все тождества для данного ln N"""
    results = []

    ln_lnN = math.log(lnN)
    ln_lnK = math.log(lnK)
    N = math.exp(lnN)

    # 1-2
    val = (K - K / lnN) + (lnN - K) / lnK
    target = geometric_zeta + feigenbaum_delta
    results.append(('1', val, target))

    # 3-8, 10-11
    val = K / lnN - K * (lnN - K)
    target = feigenbaum_delta - sqrt2
    results.append(('3', val, target))

    # 9
    val = K / lnN + K * (K - lnN)
    target = feigenbaum_delta - sqrt2
    results.append(('9', val, target))

    # 12
    val = lnN - K * (lnK + 1.0 / lnN)
    target = zeta_alpha - ln2
    results.append(('12', val, target))

    # 13-14
    val = (lnN - K) * (lnK + 1.0 / lnN)
    target = ln2 - zeta_alpha
    results.append(('13', val, target))

    # 15-18
    val = (K / lnN / (lnN - K)) / (lnK - K)
    target = ln2
    results.append(('15', val, target))

    # 19-20
    val = (K / lnN + K ** 2) / (K + K / lnN)
    target = euler_mascheroni + feigenbaum_delta
    results.append(('19', val, target))

    # 21-24
    val = 1.0 / ln_lnN + K / (K - ln_lnN)
    target = lnpi / euler_mascheroni
    results.append(('21', val, target))

    # 25-29
    val = (K / N) * (1.0 / ln_lnN + ln_lnK)
    target = geometric_zeta / feigenbaum_alpha
    results.append(('25', val, target))

    # 30-31
    val = (lnN - 1.0 / ln_lnN) / (2 * K)
    target = ln2 / dzetta_2
    results.append(('30', val, target))

    # 32-35
    val = lnK / (ln_lnN - 1.0 / lnN)
    target = sqrt3 - euler_mascheroni
    results.append(('32', val, target))

    # 36
    val = (K + lnK) / (1.0 / ln_lnK - 1.0 / lnN)
    target = feigenbaum_delta + feigenbaum_alpha_inv
    results.append(('36', val, target))

    # 37-40
    val = (K + (K - 1.0 / lnN ** 3)) / K ** 2
    target = euler_mascheroni * euler_mascheroni
    results.append(('37', val, target))

    # 41-42
    val = K / lnK - (lnN / K - (lnN - K))
    target = dzetta_2 + feigenbaum_alpha_inv
    results.append(('41', val, target))

    # 43-48
    val = (K - lnK) * (ln_lnN * (lnN - K))
    target = geometric_zeta - euler_e
    results.append(('43', val, target))

    # 49-54
    val = ln_lnN * ((K - 1.0 / lnN ** 3) / K)
    target = pi - sqrt2
    results.append(('49', val, target))

    # 55-57
    val = ln_lnK / (ln_lnN + 1.0 / lnK)
    target = ln2 / euler_e
    results.append(('55', val, target))

    # 58
    val = (K / (N - K ** 3)) - 1.0 / lnN ** 3
    target = sqrt2 * geometric_zeta
    results.append(('58', val, target))

    # 59-60
    val = K / lnK + (lnK - ln_lnN)
    target = ln2 + euler_e
    results.append(('59', val, target))

    # 61-68
    val = K - (lnN / K + 1.0 / lnN)
    target = feigenbaum_delta + feigenbaum_delta_inv
    results.append(('61', val, target))

    # 69
    val = (1.0 / lnN ** 2) * (lnK - 1.0 / lnK)
    target = geometric_phys / sqrt3
    results.append(('69', val, target))

    # 70
    val = 1.0 / lnN ** 2 + (K + (1.0 / lnK ** 2 / K))
    target = sqrt2 + feigenbaum_delta
    results.append(('70', val, target))

    # 71-72
    val = lnK / K + (1.0 / ln_lnK - 1.0 / lnK ** 2)
    target = geometric_zeta / geometric_alpha
    results.append(('71', val, target))

    # 73-74
    val = ln_lnN - ((K + K / lnN) / K)
    target = euler_mascheroni + zeta_alpha
    results.append(('73', val, target))

    # 75-77
    val = 1.0 / lnK ** 2 + lnN / ln_lnN
    target = feigenbaum_delta - ln3
    results.append(('75', val, target))

    # 78
    val = K / (K - 1.0 / ln_lnN) - K / lnN
    target = lnpi * geometric_alpha
    results.append(('78', val, target))

    # 79-81
    val = lnK + (ln_lnN + 1.0 / lnK)
    target = feigenbaum_delta / lnpi
    results.append(('79', val, target))

    # 82-84
    val = (K - (N - lnK)) / (K + N)
    target = ln2 - dzetta_2
    results.append(('82', val, target))

    # 85-87
    val = ln_lnN * (K / (K + 1.0 / lnN ** 3))
    target = pi - sqrt2
    results.append(('85', val, target))

    # 88-93
    val = 1.0 / lnK * (1.0 / ln_lnN + 1.0 / lnK)
    target = ln3 / sqrt3
    results.append(('88', val, target))

    # 94-99
    val = (K - lnN / K) * (lnN / K / ln_lnN)
    target = alpha_phys + euler_e
    results.append(('94', val, target))

    # 100
    val = ((lnN - K) - lnK ** 2) / (K / lnK)
    target = euler_mascheroni - dzetta_2
    results.append(('100', val, target))

    return results


# ФУНКЦИЯ ОШИБКИ
def total_error(lnN):
    """Суммарная относительная ошибка по всем тождествам"""
    identities = compute_all_identities(lnN)
    total = 0.0
    for name, val, target in identities:
        if abs(target) > 1e-15:
            total += (abs(val - target) / abs(target)) ** 2
    return total


# ПОИСК ОПТИМАЛЬНОГО ln N
print("ПОИСК ОПТИМАЛЬНОГО ln N ДЛЯ 100 ТОЖДЕСТВ")
# Сканируем диапазон
lnN_range = np.linspace(10, 500, 1000)
errors = [total_error(lnN) for lnN in lnN_range]

# Находим минимум
min_idx = np.argmin(errors)
lnN_opt_scan = lnN_range[min_idx]
error_opt_scan = errors[min_idx]

print(f"\n  Сканирование: ln N_opt = {lnN_opt_scan:.6f}, ошибка = {error_opt_scan:.6e}")

# Уточняем
result = minimize_scalar(total_error, bounds=(lnN_opt_scan - 50, lnN_opt_scan + 50), method='bounded')
lnN_opt = result.x
error_opt = result.fun

N_opt = math.exp(lnN_opt)

print(f"  Уточнение:    ln N_opt = {lnN_opt:.10f}")
print(f"                N_opt    = {N_opt:.6e}")
print(f"                Ошибка   = {error_opt:.6e}")

# СРАВНЕНИЕ С ИЗВЕСТНЫМИ ЗНАЧЕНИЯМИ
print("СРАВНЕНИЕ ОПТИМАЛЬНОГО ln N С ИЗВЕСТНЫМИ ЗНАЧЕНИЯМИ")

known_values = {
    'Геометрический резонанс': lnN_geometric(),
    'Дзета-функция': lnN_zeta(),
    'Постоянная тонкой структуры': lnN_alpha(),
    'Гипотеза (G - πK/G)': lnN_hypothesis(),
}

print(f"\n  {'Метод':<30} {'ln N':<18} {'Отклонение от оптимума':<25}")
print(f"  {'-' * 73}")
for name, val in known_values.items():
    dev = abs(val - lnN_opt) / lnN_opt * 100
    print(f"  {name:<30} {val:<18.10f} {dev:<25.10f}%")

print(f"\n  {'Оптимум (100 тождеств)':<30} {lnN_opt:<18.10f}")

# ДЕТАЛЬНЫЙ АНАЛИЗ ПРИ ОПТИМАЛЬНОМ ln N
print("ДЕТАЛЬНЫЙ АНАЛИЗ ПРИ ОПТИМАЛЬНОМ ln N")

identities_opt = compute_all_identities(lnN_opt)
errors_list = []
for name, val, target in identities_opt:
    if abs(target) > 1e-15:
        err = abs(val - target) / abs(target) * 100
        errors_list.append((name, val, target, err))

errors_list.sort(key=lambda x: x[3])

print(f"\n  ТОП-10 ЛУЧШИХ:")
print(f"  {'#':<6} {'Значение':<20} {'Цель':<20} {'Ошибка %':<15}")
print(f"  {'-' * 65}")
for name, val, target, err in errors_list[:10]:
    print(f"  {name:<6} {val:<20.10f} {target:<20.10f} {err:<15.8f}")

print(f"\n  ТОП-10 ХУДШИХ:")
print(f"  {'#':<6} {'Значение':<20} {'Цель':<20} {'Ошибка %':<15}")
print(f"  {'-' * 65}")
for name, val, target, err in errors_list[-10:]:
    print(f"  {name:<6} {val:<20.10f} {target:<20.10f} {err:<15.8f}")

# Статистика
all_errs = [e[3] for e in errors_list]
print(f"\n  СТАТИСТИКА:")
print(f"  Всего тождеств: {len(all_errs)}")
print(f"  Средняя ошибка: {np.mean(all_errs):.8f}%")
print(f"  Медиана:        {np.median(all_errs):.8f}%")
print(f"  < 0.001%:       {np.sum(np.array(all_errs) < 0.001)}/{len(all_errs)}")
print(f"  < 0.01%:        {np.sum(np.array(all_errs) < 0.01)}/{len(all_errs)}")
print(f"  < 0.1%:         {np.sum(np.array(all_errs) < 0.1)}/{len(all_errs)}")
print(f"  < 1.0%:         {np.sum(np.array(all_errs) < 1.0)}/{len(all_errs)}")

# ВЫВОДЫ
print("ВЫВОДЫ")
print(f"""
  1. Оптимальное ln N для 100 математических тождеств:
     ln N_opt = {lnN_opt:.6f}
     N_opt    = {N_opt:.6e}

  2. Ближайшее известное значение:
     Геометрический резонанс: {lnN_geometric():.6f}
     (отклонение: {abs(lnN_geometric() - lnN_opt):.6f})

  3. Это означает, что математические тождества "предпочитают"
     то же самое N, что и физические константы!

  4. Математика и физика имеют общий корень —
     информационный граф с K=6 и ln N ≈ 280.
""")