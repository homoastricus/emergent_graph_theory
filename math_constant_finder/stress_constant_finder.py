import math
import numpy as np
from collections import defaultdict
import itertools
import time

# КОНСТАНТЫ
K = 6.0
pi = math.pi
lnK = math.log(K)


# СТРУКТУРА ФОРМУЛ
def make_formula(coeff, pow_lnN, pow_N13, pow_lnK=0):
    return (coeff, pow_lnN, pow_N13, pow_lnK)


formulas = {
    'ħ': make_formula(1.0 / K, 3, -1),
    'c': make_formula(pi / (K ** 2 * lnK), 4, 0),
    'l_P': make_formula(4 * lnK, 2, -1),
    't_P': make_formula(4 * K ** 2 * lnK ** 2 / pi, -2, -1),
    'E_P': make_formula(pi / (4 * K ** 3 * lnK ** 2), 5, 0),
    'G': make_formula(16 * pi ** 3 / (K ** 5 * lnK), 13, -1),
    'm_P': make_formula(K / (4 * pi), -3, 0),
    'T_P': make_formula(8 * pi, -4, 1),
    'k_B': make_formula(1.0 / (8 * pi ** 2), 8, -1),
    'α': make_formula(2 * lnK ** 2 / pi, -1, 0),
    'm_e': make_formula(4 * pi / math.sqrt(K), 4, -1),
    'm_p': make_formula(math.sqrt(pi) / K ** (1.5), 6, -1),
    'ep_0': make_formula(1.0 / (8 * pi ** 3 * lnK), -20, 1),
    'mu_0': make_formula(8 * pi * K ** 4 * lnK ** 3, 12, -1),
    'R∞': make_formula(4 * lnK ** 3 / (pi * K ** (1.5)), 3, 0),
    'a_0': make_formula(K ** (1.5) / (8 * pi * lnK), -4, 0),
    'Z_0': make_formula(8 * K ** 2 * pi ** 2 * lnK ** 2, 16, -1),
    'Φ_0': make_formula(pi ** 2 * math.sqrt(K), 10, -1),
    'q_e': make_formula(1.0 / (pi * K ** (1.5)), -7, 0),
    'λ_e': make_formula(K ** (1.5) * lnK / (2 * pi), -5, 0),
    'λ_p': make_formula(2 * K ** (2.5) * lnK / math.sqrt(pi), -7, 0),
    'Λ': make_formula(1.0 / math.sqrt(pi), 12, -2),
    'κ': make_formula(128 * K ** 3 * lnK ** 3, -3, -1),
    'v_H': make_formula(8 * pi ** (1.5) / math.sqrt(2), 6, -1),
}

# MEET-IN-THE-MIDDLE: ОПТИМИЗИРОВАННЫЙ ПЕРЕБОР ДО 5 ФОРМУЛ

print("MEET-IN-THE-MIDDLE: ПОЛНЫЙ ПЕРЕБОР ДО 5 ФОРМУЛ")

start_time = time.time()

all_names = list(formulas.keys())
n_formulas = len(all_names)

# Извлекаем данные
a_vals = np.array([formulas[name][1] for name in all_names], dtype=np.int32)
b_vals = np.array([formulas[name][2] for name in all_names], dtype=np.int32)
log_coeffs = np.array([math.log(abs(formulas[name][0])) for name in all_names], dtype=np.float64)

# Параметры
MAX_POWER = 4
MIN_POWER = -4
MAX_FORMULAS = 4

# Шаг 1: Перебираем ВСЕ возможные комбинации 1-2 формул
# и сохраняем в хеш-таблицу по ключу (sum_a, sum_b)
print(f"\n  Шаг 1: Перебор комбинаций 1-2 формул...")

combo_map = defaultdict(list)  # (sum_a, sum_b) -> [(terms, log_coeff), ...]

# Для каждой одной формулы
for i in range(n_formulas):
    for p in range(MIN_POWER, MAX_POWER + 1):
        if p == 0:
            continue
        sum_a = a_vals[i] * p
        sum_b = b_vals[i] * p
        terms = ((i, p),)
        log_c = p * log_coeffs[i]
        combo_map[(sum_a, sum_b)].append((terms, log_c))

# Для каждой пары формул
for i, j in itertools.combinations(range(n_formulas), 2):
    for pi in range(MIN_POWER, MAX_POWER + 1):
        if pi == 0:
            continue
        for pj in range(MIN_POWER, MAX_POWER + 1):
            if pj == 0:
                continue
            sum_a = a_vals[i] * pi + a_vals[j] * pj
            sum_b = b_vals[i] * pi + b_vals[j] * pj
            terms = ((i, pi), (j, pj))
            log_c = pi * log_coeffs[i] + pj * log_coeffs[j]
            combo_map[(sum_a, sum_b)].append((terms, log_c))

print(f"  Создано групп: {len(combo_map):,}")
print(f"  Всего элементов: {sum(len(v) for v in combo_map.values()):,}")

# Шаг 2: Перебираем все пары (1-2 формулы) × (1-2 формулы)
# ищем, где суммы дают общее (0, 0) и всего формул ≤ 5
print(f"\n  Шаг 2: Поиск пар с компенсацией...")

found_signatures = set()
found_combos = []

# Сортируем ключи для прогресса
all_keys = list(combo_map.keys())
n_keys = len(all_keys)

# Перебираем все пары групп
for idx1, (key1, list1) in enumerate(combo_map.items()):
    sum_a1, sum_b1 = key1

    # Нужный ключ для компенсации
    target_key = (-sum_a1, -sum_b1)

    if target_key not in combo_map:
        continue

    list2 = combo_map[target_key]

    # Чтобы избежать дубликатов, обеспечиваем key1 <= key2 в каком-то смысле
    # Простое решение: обрабатываем все пары, но проверяем сигнатуру

    for terms1, log_c1 in list1:
        for terms2, log_c2 in list2:
            # Объединяем terms
            combined_terms = list(terms1) + list(terms2)

            # Проверяем, что всего ≤ 5 разных формул
            # и степени в диапазоне
            merged = defaultdict(int)
            for i, p in combined_terms:
                merged[i] += p

            if len(merged) > MAX_FORMULAS:
                continue

            # Проверяем, что все степени в диапазоне
            if any(abs(p) > MAX_POWER for p in merged.values()):
                continue

            if any(p == 0 for p in merged.values()):
                continue

            # Нормализация
            all_powers = [abs(p) for p in merged.values()]
            g = all_powers[0]
            for pp in all_powers[1:]:
                g = math.gcd(g, pp)

            if g > 1:
                merged = {i: p // g for i, p in merged.items()}

            # Каноническая сигнатура
            terms_normalized = tuple(sorted(merged.items()))

            # Фиксируем знак
            first_power = terms_normalized[0][1]
            if first_power < 0:
                terms_normalized = tuple((i, -p) for i, p in terms_normalized)

            if terms_normalized in found_signatures:
                continue

            found_signatures.add(terms_normalized)

            # Вычисляем итоговый коэффициент
            total_log = log_c1 + log_c2

            # Строковое представление
            parts = []
            for i, p in terms_normalized:
                name = all_names[i]
                if p == 1:
                    parts.append(name)
                elif p == -1:
                    parts.append(f"1/{name}")
                elif p > 0:
                    parts.append(f"{name}^{p}")
                else:
                    parts.append(f"1/{name}^{{{-p}}}")

            expr = " · ".join(parts)

            max_pow = max(abs(p) for _, p in terms_normalized)
            n_formulas_used = len(merged)
            total_ops = sum(abs(p) for _, p in terms_normalized)

            found_combos.append({
                'expr': expr,
                'value': math.exp(total_log),
                'max_power': max_pow,
                'n_formulas': n_formulas_used,
                'n_operations': total_ops,
                'terms': terms_normalized,
            })

elapsed = time.time() - start_time

# # РЕЗУЛЬТАТЫ
# print("РЕЗУЛЬТАТЫ")
# print(f"  Время: {elapsed:.1f} сек")
# print(f"  Уникальных безразмерных комбинаций: {len(found_combos):,}")
#
# # Распределение по числу формул
# print(f"\n  По числу формул:")
# for r in range(1, MAX_FORMULAS + 1):
#     count = sum(1 for c in found_combos if c['n_formulas'] == r)
#     print(f"    {r} формул(ы): {count}")
#
# # Распределение по максимальной степени
# print(f"\n  По максимальной степени:")
# for p in range(1, MAX_POWER + 1):
#     count = sum(1 for c in found_combos if c['max_power'] == p)
#     if count > 0:
#         print(f"    max_pow = {p}: {count}")
#
# # ТОП КОМБИНАЦИЙ С БОЛЬШИМИ СТЕПЕНЯМИ
# print("ТОП-50 КОМБИНАЦИЙ С БОЛЬШИМИ СТЕПЕНЯМИ")
#
# big = sorted(found_combos, key=lambda x: x['max_power'], reverse=True)[:50]
# print(f"\n  {'Макс. ст.':<10} {'Формул':<8} {'Выражение':<80}")
# print(f"  {'-' * 100}")
# for c in big:
#     expr_short = c['expr'][:80]
#     print(f"  {c['max_power']:<10} {c['n_formulas']:<8} {expr_short}")

# БЛИЗОСТЬ К ФУНДАМЕНТАЛЬНЫМ КОНСТАНТАМ
print("ТОП-40 ПО БЛИЗОСТИ К ФУНДАМЕНТАЛЬНЫМ КОНСТАНТАМ")

# БЛИЗОСТЬ К ФУНДАМЕНТАЛЬНЫМ КОНСТАНТАМ (ИСПРАВЛЕННАЯ ВЕРСИЯ)
print("ТОП-40 ПО БЛИЗОСТИ К ФУНДАМЕНТАЛЬНЫМ КОНСТАНТАМ (ИСПРАВЛЕННАЯ)")

# Расширенный список целей
# targets = {
#     # Целые числа
#     #'1': 1.0,
#     '2': 2.0,
#     '3': 3.0,
#     '4': 4.0,
#     '5': 5.0,
#     '6': 6.0,
#     '7': 7.0,
#     '8': 8.0,
#     '9': 9.0,
#     '10': 10.0,
#     '16': 16.0,
#
#     # Геометрические
#     'π': pi,
#     '1/π': 1.0 / pi,
#     '2π': 2 * pi,
#     'π/2': pi / 2,
#     'π/3': pi / 3,
#     'π/4': pi / 4,
#     'π/6': pi / 6,
#     'π²': pi ** 2,
#     '4π/3': 4 * pi / 3,
#     '√π': math.sqrt(pi),
#
#     # Алгебраические
#     '√2': math.sqrt(2),
#     '√3': math.sqrt(3),
#     # '√5': math.sqrt(5),
#     '√6': math.sqrt(6),
#     '√K': math.sqrt(K),
#     # 'φ': (1 + math.sqrt(5)) / 2,
#     # '1/φ': 2 / (1 + math.sqrt(5)),
#
#     # Трансцендентные
#     # 'e': math.e,
#     # '1/e': 1.0 / math.e,
#
#     # Логарифмические
#     'ln 2': math.log(2),
#     'ln 3': math.log(3),
#     # 'ln 5': math.log(5),
#     'ln K': lnK,
#     # 'ln π': math.log(pi),
#
#     # Специальные
#     # 'γ': 0.5772156649015329,  # Эйлер-Маскерони
# }
#
# best_matches = []
# for c in found_combos:
#     best_dist = float('inf')
#     best_target = None
#     best_inverted = False
#
#     value = c['value']
#
#     if abs(value) > 1e-100 and abs(value) < 1e100:
#         # Округляем до ближайшего целого
#         nearest_int = round(value)
#         if nearest_int >= 1 and nearest_int <= 16:
#             dist = abs(value - nearest_int) / nearest_int
#             if dist < best_dist:
#                 best_dist = dist
#                 best_target = str(nearest_int)
#                 best_inverted = False
#
#         # Обратная проверка
#     if abs(value) > 1e-100 and abs(value) < 1e100:
#         inv_value = 1.0 / value
#         nearest_int = round(inv_value)
#         if nearest_int >= 1 and nearest_int <= 16:
#             inv_dist = abs(inv_value - nearest_int) / nearest_int
#             if inv_dist < best_dist:
#                 best_dist = inv_dist
#                 best_target = str(nearest_int)
#                 best_inverted = True
#
#     best_matches.append((best_dist, best_target, c, best_inverted))
#
# best_matches.sort(key=lambda x: x[0])
#
# # Вывод с учётом инверсии
# print(f"\n  {'Выражение':<80} {'Значение':<15} {'Цель':<15} {'Ошибка %':<12}")
# print(f"  {'-' * 125}")
#
# count_shown = 0
# for dist, target, c, inverted in best_matches:  # теперь 4 элемента
#     if dist < 0.5:
#         count_shown += 1
#         if count_shown <= 500:
#             if inverted:
#                 # Инвертированная формула
#                 expr_short = "1/(" + c['expr'] + ")"
#                 value_to_show = 1.0 / c['value']
#             else:
#                 expr_short = c['expr']
#                 value_to_show = c['value']
#
#             # Обрезаем для вывода
#             if len(expr_short) > 100:
#                 expr_short = expr_short[:97] + "..."
#
#             print(f"  {expr_short:<100} {value_to_show:<15.10f} {target:<15} {dist * 100:<12.6f}%")
#
# print(f"\n  Показано: {count_shown} комбинаций с ошибкой < 50%")


# РЕЗУЛЬТАТЫ
print("РЕЗУЛЬТАТЫ (ТОЛЬКО КОМБИНАЦИИ С lnK^0)")
print(f"  Время: {elapsed:.1f} сек")
print(f"  Уникальных безразмерных комбинаций (без lnK): {len(found_combos):,}")

# Распределение по числу формул
print(f"\n  По числу формул:")
for r in range(1, MAX_FORMULAS + 1):
    count = sum(1 for c in found_combos if c['n_formulas'] == r)
    if count > 0:
        print(f"    {r} формул(ы): {count}")

# Распределение по максимальной степени
print(f"\n  По максимальной степени:")
for p in range(1, MAX_POWER + 1):
    count = sum(1 for c in found_combos if c['max_power'] == p)
    if count > 0:
        print(f"    max_pow = {p}: {count}")

# ТОП-50 КОМБИНАЦИЙ С БОЛЬШИМИ СТЕПЕНЯМИ
# print("\n" + "="*80)
# print("ТОП-50 КОМБИНАЦИЙ С БОЛЬШИМИ СТЕПЕНЯМИ")
# print("="*80)
# big = sorted(found_combos, key=lambda x: x['max_power'], reverse=True)[:50]
# print(f"\n  {'Макс. ст.':<10} {'Формул':<8} {'Выражение':<80}")
# print(f"  {'-' * 100}")
# for c in big:
#     expr_short = c['expr'][:80]
#     print(f"  {c['max_power']:<10} {c['n_formulas']:<8} {expr_short}")
#
# # Показать все комбинации, сгруппированные по числу формул
# print("\n" + "="*80)
# print("ВСЕ КОМБИНАЦИИ (СГРУППИРОВАНЫ ПО ЧИСЛУ ФОРМУЛ)")
# print("="*80)
# for r in range(1, MAX_FORMULAS + 1):
#     subset = [c for c in found_combos if c['n_formulas'] == r]
#     if subset:
#         print(f"\n  --- {r} ФОРМУЛ(Ы) ({len(subset)} комбинаций) ---")
#         subset_sorted = sorted(subset, key=lambda x: x['max_power'], reverse=True)
#         for c in subset_sorted:
#             print(f"    max_pow={c['max_power']:<4} | {c['expr'][:100]}")

# Генерируем целевые значения: π^0.5, π^1.0, π^1.5, ..., π^20.0
target_values = []
target_names = []
for power in np.arange(0.5, 20.01, 0.5):  # 0.5, 1.0, 1.5, ..., 20.0 — всего 40 значений
    val = pi ** power
    target_values.append(val)
    if power == int(power):
        target_names.append(f"π^{int(power)}")
    else:
        target_names.append(f"π^{power}")

# Также проверяем обратные значения: π^(-0.5), π^(-1.0), ..., π^(-20.0)
inv_target_values = [1.0 / v for v in target_values]
inv_target_names = []
for power in np.arange(0.5, 20.01, 0.5):
    if power == int(power):
        inv_target_names.append(f"π^(-{int(power)})")
    else:
        inv_target_names.append(f"π^(-{power})")

# Допустимая относительная погрешность
TOLERANCE = 0.0001  # 10^(-7)

best_matches = []

for c in found_combos:
    best_dist = float('inf')
    best_target = None
    best_inverted = False

    value = c['value']

    if abs(value) > 1e-100 and abs(value) < 1e100:
        # Прямая проверка: сравниваем с π^0.5, π^1.0, ..., π^20.0
        for target_val, target_name in zip(target_values, target_names):
            dist = abs(value - target_val) / target_val
            if dist < TOLERANCE and dist < best_dist:
                best_dist = dist
                best_target = target_name
                best_inverted = False

        # Обратная проверка: сравниваем 1/value с π^0.5, π^1.0, ..., π^20.0
        inv_value = 1.0 / value
        for target_val, target_name in zip(target_values, target_names):
            inv_dist = abs(inv_value - target_val) / target_val
            if inv_dist < TOLERANCE and inv_dist < best_dist:
                best_dist = inv_dist
                best_target = target_name
                best_inverted = True

    # Добавляем только если нашли соответствие
    if best_target is not None:
        best_matches.append((best_dist, best_target, c, best_inverted))

# Сортируем по точности (лучшие совпадения первыми)
best_matches.sort(key=lambda x: x[0])

print(f"\n  Найдено совпадений со степенями π: {len(best_matches)}")
print(f"  Допустимая погрешность: {TOLERANCE}")

# Вывод с учётом инверсии
print(f"\n  {'Выражение':<80} {'Значение':<15} {'Цель':<15} {'Ошибка %':<12}")
print(f"  {'-' * 125}")

count_shown = 0
for dist, target, c, inverted in best_matches:  # теперь 4 элемента
    if dist < 0.5:
        count_shown += 1
        if count_shown <= 5000:
            if inverted:
                # Инвертированная формула
                expr_short = "1/(" + c['expr'] + ")"
                value_to_show = 1.0 / c['value']
            else:
                expr_short = c['expr']
                value_to_show = c['value']

            # Обрезаем для вывода
            if len(expr_short) > 100:
                expr_short = expr_short[:97] + "..."

            print(f"  {expr_short:<100} {value_to_show:<15.10f} {target:<15} {dist * 100:<12.6f}%")

print(f"\n  Показано: {count_shown} комбинаций с ошибкой < 50%")

# for r in range(1, MAX_FORMULAS + 1):
#     subset = [c for c in found_combos if c['n_formulas'] == r]
#     if not subset:
#         continue
#     print(f"\n  --- {r} ФОРМУЛ(Ы) ({len(subset)} комбинаций) ---")
#     subset_sorted = sorted(subset, key=lambda x: x['max_power'], reverse=True)
#     for c in subset_sorted[:20]:
#         print(f"    max_pow={c['max_power']:<4} | {c['expr'][:90]}")
    # if len(subset) > 10:
    #     print(f"    ... и ещё {len(subset) - 10}")

