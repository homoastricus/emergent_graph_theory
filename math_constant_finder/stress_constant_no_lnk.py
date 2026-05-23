import math
import numpy as np
from collections import defaultdict, Counter
import itertools
import time

# КОНСТАНТЫ
K = 6.0
pi = math.pi
lnK = math.log(K)


# СТРУКТУРА ФОРМУЛ — теперь с тремя показателями: (coeff, pow_lnN, pow_N13, pow_lnK)
def make_formula(coeff, pow_lnN, pow_N13, pow_lnK=0):
    return (coeff, pow_lnN, pow_N13, pow_lnK)


formulas = {
    'r_e': make_formula(K ** (3/2) / (2 * pi**3), -6,0,3),
    'ħ': make_formula(1.0 / K, 3, -1, 0),
    'c': make_formula(pi / (K ** 2), 4, 0, -1),
    'l_P': make_formula(4, 2, -1, 1),
    't_P': make_formula(4 * K ** 2 / pi, -2, -1, 2),
    'E_P': make_formula(pi / (4 * K ** 3), 5, 0, -2),
    'G': make_formula(16 * pi ** 3 / (K ** 5), 13, -1, -1),
    'm_P': make_formula(K / (4 * pi), -3, 0, 0),
    'T_P': make_formula(8 * pi, -4, 1, 0),
    'k_B': make_formula(1.0 / (8 * pi ** 2), 8, -1, 0),
    'α': make_formula(2 / pi, -1, 0, 2),
    'm_e': make_formula(4 * pi / math.sqrt(K), 4, -1, 0),
    'm_p': make_formula(math.sqrt(pi) / K ** (1.5), 6, -1, 0),
    'ep_0': make_formula(1.0 / (8 * pi ** 3), -20, 1, -1),
    'mu_0': make_formula(8 * pi * K ** 4, 12, -1, 3),
    'R∞': make_formula(4 / (pi * K ** (1.5)), 3, 0, 3),
    'a_0': make_formula(K ** (1.5) / (8 * pi), -4, 0, -1),
    'Z_0': make_formula(8 * K ** 2 * pi ** 2, 16, -1, 2),
    'Φ_0': make_formula(pi ** 2 * math.sqrt(K), 10, -1, 0),
    'q_e': make_formula(1.0 / (pi * K ** (1.5)), -7, 0, 0),
    'λ_e': make_formula(K ** (1.5) / (2 * pi), -5, 0, 1),
    'λ_p': make_formula(2 * K ** (2.5) / math.sqrt(pi), -7, 0, 1),
    'Λ': make_formula(1.0 / math.sqrt(pi), 12, -2, 0),
    'κ': make_formula(128 * K ** 3, -3, -1, 3),
    'v_H': make_formula(8 * pi ** (1.5) / math.sqrt(2), 6, -1, 0),
}

# Компоненты формул для символьного представления (только K и π, без lnK)
formula_components = {
    'r_e': (0.5, {'K': 1.5, 'π': -3}),
    'ħ': (1.0, {'K': -1}),
    'c': (1.0, {'K': -2, 'π': 1}),
    'l_P': (4.0, {}),
    't_P': (4.0, {'K': 2, 'π': -1}),
    'E_P': (0.25, {'K': -3, 'π': 1}),
    'G': (16.0, {'K': -5, 'π': 3}),
    'm_P': (0.25, {'K': 1, 'π': -1}),
    'T_P': (8.0, {'π': 1}),
    'k_B': (0.125, {'π': -2}),
    'α': (2.0, {'π': -1}),
    'm_e': (4.0, {'K': -0.5, 'π': 1}),
    'm_p': (1.0, {'K': -1.5, 'π': 0.5}),
    'ep_0': (0.125, {'π': -3}),
    'mu_0': (8.0, {'K': 4, 'π': 1}),
    'R∞': (4.0, {'K': -1.5, 'π': -1}),
    'a_0': (0.125, {'K': 1.5, 'π': -1}),
    'Z_0': (8.0, {'K': 2, 'π': 2}),
    'Φ_0': (1.0, {'K': 0.5, 'π': 2}),
    'q_e': (1.0, {'K': -1.5, 'π': -1}),
    'λ_e': (0.5, {'K': 1.5, 'π': -1}),
    'λ_p': (2.0, {'K': 2.5, 'π': -0.5}),
    'Λ': (1.0, {'π': -0.5}),
    'κ': (128.0, {'K': 3}),
    'v_H': (8.0 / math.sqrt(2), {'π': 1.5}),
}


def build_symbolic_formula(merged_terms):
    """Строит символьную формулу из объединенных компонентов"""

    total_coeff = 1.0
    total_components = Counter()

    for idx, power in merged_terms.items():
        name = all_names[idx]
        coeff, comps = formula_components[name]

        total_coeff *= coeff ** power
        for comp, exp in comps.items():
            total_components[comp] += exp * power

    # Удаляем нулевые компоненты
    total_components = Counter({k: v for k, v in total_components.items() if abs(v) > 1e-10})

    # Собираем строку
    parts = []

    # Числовой коэффициент (только если не 1)
    if abs(total_coeff - 1.0) > 1e-10:
        if abs(total_coeff - round(total_coeff)) < 1e-10:
            parts.append(str(int(round(total_coeff))))
        else:
            parts.append(f"{total_coeff:.10g}")

    # Положительные степени (числитель)
    for comp in sorted(total_components.keys()):
        if total_components[comp] > 0:
            exp = total_components[comp]
            if abs(exp - 1.0) < 1e-10:
                parts.append(comp)
            elif abs(exp - round(exp)) < 1e-10:
                parts.append(f"{comp}^{int(round(exp))}")
            else:
                parts.append(f"{comp}^{exp}")

    # Отрицательные степени (знаменатель)
    denom_parts = []
    for comp in sorted(total_components.keys()):
        if total_components[comp] < 0:
            exp = -total_components[comp]
            if abs(exp - 1.0) < 1e-10:
                denom_parts.append(comp)
            elif abs(exp - round(exp)) < 1e-10:
                denom_parts.append(f"{comp}^{int(round(exp))}")
            else:
                denom_parts.append(f"{comp}^{exp}")

    if denom_parts:
        if len(denom_parts) == 1:
            parts.append(f"1/{denom_parts[0]}")
        else:
            parts.append(f"1/({'·'.join(denom_parts)})")

    if not parts:
        return "1"

    return "·".join(parts)


# MEET-IN-THE-MIDDLE: ОПТИМИЗИРОВАННЫЙ ПЕРЕБОР ДО 5 ФОРМУЛ
print("MEET-IN-THE-MIDDLE: ПОЛНЫЙ ПЕРЕБОР ДО 5 ФОРМУЛ (ТОЛЬКО С НУЛЕВОЙ СТЕПЕНЬЮ lnK)")
print("=" * 100)

start_time = time.time()

all_names = list(formulas.keys())
n_formulas = len(all_names)

# Извлекаем данные
a_vals = np.array([formulas[name][1] for name in all_names], dtype=np.int32)  # ln N
b_vals = np.array([formulas[name][2] for name in all_names], dtype=np.int32)  # N^(1/3)
c_vals = np.array([formulas[name][3] for name in all_names], dtype=np.int32)  # ln K
# ВАЖНО: log_coeffs должен браться ИМЕННО из formulas[name][0]
log_coeffs = np.array([math.log(abs(formulas[name][0])) for name in all_names], dtype=np.float64)

MAX_POWER = 5
MIN_POWER = -5
MAX_FORMULAS = 5

print(f"\nШаг 1: Перебор комбинаций 1-2 формул...")

combo_map = defaultdict(list)

# Для каждой одной формулы
for i in range(n_formulas):
    for p in range(MIN_POWER, MAX_POWER + 1):
        if p == 0:
            continue
        sum_a = a_vals[i] * p
        sum_b = b_vals[i] * p
        sum_c = c_vals[i] * p
        terms = ((i, p),)
        log_c = p * log_coeffs[i]
        combo_map[(sum_a, sum_b, sum_c)].append((terms, log_c))

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
            sum_c = c_vals[i] * pi + c_vals[j] * pj
            terms = ((i, pi), (j, pj))
            log_c = pi * log_coeffs[i] + pj * log_coeffs[j]
            combo_map[(sum_a, sum_b, sum_c)].append((terms, log_c))

print(f"  Создано групп: {len(combo_map):,}")
print(f"  Всего элементов: {sum(len(v) for v in combo_map.values()):,}")

print(f"\nШаг 2: Поиск пар с компенсацией...")

found_signatures = set()
found_combos = []

for key1, list1 in combo_map.items():
    sum_a1, sum_b1, sum_c1 = key1
    target_key = (-sum_a1, -sum_b1, -sum_c1)

    if target_key not in combo_map:
        continue

    list2 = combo_map[target_key]

    for terms1, log_c1 in list1:
        for terms2, log_c2 in list2:
            combined_terms = list(terms1) + list(terms2)

            merged = defaultdict(int)
            for i, p in combined_terms:
                merged[i] += p

            if len(merged) > MAX_FORMULAS:
                continue

            if any(abs(p) > MAX_POWER for p in merged.values()):
                continue

            if any(p == 0 for p in merged.values()):
                continue

            # Проверяем, что суммарная степень lnK = 0
            total_lnk = sum(c_vals[i] * p for i, p in merged.items())
            if total_lnk != 0:
                continue

            # Нормализация
            all_powers = [abs(p) for p in merged.values()]
            g = all_powers[0]
            for pp in all_powers[1:]:
                g = math.gcd(g, pp)

            if g > 1:
                merged = {i: p // g for i, p in merged.items()}

            terms_normalized = tuple(sorted(merged.items()))

            first_power = terms_normalized[0][1]
            if first_power < 0:
                terms_normalized = tuple((i, -p) for i, p in terms_normalized)

            if terms_normalized in found_signatures:
                continue

            found_signatures.add(terms_normalized)

            # Вычисляем значение напрямую из формул, а не из логарифмов
            total_log = log_c1 + log_c2

            # Дополнительная проверка: пересчитываем значение напрямую
            direct_value = 1.0
            for i, p in merged.items():
                direct_value *= formulas[all_names[i]][0] ** p

            coeff = direct_value  # Используем прямое вычисление!

            # Комбинация как строка
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

            combo_expr = " · ".join(parts)

            # Символьная формула через компоненты
            symbolic_formula = build_symbolic_formula(merged)

            found_combos.append({
                'combo_expr': combo_expr,
                'symbolic_formula': symbolic_formula,
                'value': coeff,
                'max_power': max(abs(p) for _, p in terms_normalized),
                'n_formulas': len(merged),
            })

elapsed = time.time() - start_time

# РЕЗУЛЬТАТЫ
print("\n" + "=" * 120)
print("РЕЗУЛЬТАТЫ (ТОЛЬКО КОМБИНАЦИИ С lnK^0)")
print("=" * 120)
print(f"Время: {elapsed:.1f} сек")
print(f"Уникальных безразмерных комбинаций (без lnK): {len(found_combos):,}")

print(f"\nПо числу формул:")
for r in range(1, MAX_FORMULAS + 1):
    count = sum(1 for c in found_combos if c['n_formulas'] == r)
    if count > 0:
        print(f"  {r} формул(ы): {count}")

print("\n" + "=" * 120)
print("ТАБЛИЦА: КОМБИНАЦИЯ → ЗНАЧЕНИЕ → СИМВОЛЬНАЯ ФОРМУЛА")
print("=" * 120)
print(f"\n{'Комбинация':<40} {'Значение':<25} {'Символьная формула':<50}")
print(f"{'-' * 40} {'-' * 25} {'-' * 50}")

sorted_combos = sorted(found_combos, key=lambda x: x['value'])

for c in sorted_combos:
    combo_short = c['combo_expr'][:40]
    sym_short = c['symbolic_formula'][:50]
    print(f"{combo_short:<40} {c['value']:<25.15e} {sym_short:<50}")

print(f"\nВсего комбинаций: {len(found_combos)}")