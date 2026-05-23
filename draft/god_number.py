import math
import itertools

# Пять фундаментальных констант
constants = {
    'π': math.pi,
    'γ_E': 0.5772156649015329,  # постоянная Эйлера-Маскерони
    'δ_F': 4.669201609102990,  # постоянная Фейгенбаума
    'ln 2': math.log(2),
    'ln 3': math.log(3),
}

names = list(constants.keys())
values = list(constants.values())

print("╔══════════════════════════════════════════════════════════════════╗")
print("║   ИССЛЕДОВАНИЕ: 5 ФУНДАМЕНТАЛЬНЫХ КОНСТАНТ                     ║")
print("╚══════════════════════════════════════════════════════════════════╝")

print("\nИсходные значения:")
for name, val in constants.items():
    print(f"  {name:<6} = {val:.10f}")

# ============================================================
# 1. АДДИТИВНЫЙ АНАЛИЗ (разности и суммы)
# ============================================================
print(f"\n{'=' * 70}")
print("1. АДДИТИВНЫЙ АНАЛИЗ")
print(f"{'=' * 70}")

print("\n1a. ПОПАРНЫЕ РАЗНОСТИ (|a - b|):")
print(f"  {'Пара':<20} {'Разность':>14} {'/π':>14} {'/γ_E':>14} {'/δ_F':>14} {'/ln2':>14} {'/ln3':>14}")
print(f"  {'-' * 90}")

interesting_diffs = []
for i, j in itertools.combinations(range(5), 2):
    diff = abs(values[i] - values[j])
    ratios = [diff / v for v in values]
    print(
        f"  {names[i] + ' - ' + names[j]:<20} {diff:>14.8f} {ratios[0]:>14.8f} {ratios[1]:>14.8f} {ratios[2]:>14.8f} {ratios[3]:>14.8f} {ratios[4]:>14.8f}")

    # Проверяем, не является ли разность близкой к целому кратному одной из констант
    for k, (name, val) in enumerate(constants.items()):
        for mult in [1, 2, 3, 4, 5, 6, 1 / 2, 1 / 3, 1 / 4, 1 / 6, 2 / 3, 3 / 2]:
            if abs(diff - mult * val) / (mult * val) < 0.05:
                interesting_diffs.append(
                    f"  |{names[i]} - {names[j]}| ≈ {mult}×{name} (откл: {abs(diff - mult * val) / (mult * val) * 100:.2f}%)")

if interesting_diffs:
    print(f"\n  ИНТЕРЕСНЫЕ СОВПАДЕНИЯ:")
    for item in interesting_diffs:
        print(item)

print("\n1b. ПОПАРНЫЕ СУММЫ:")
print(f"  {'Пара':<20} {'Сумма':>14} {'/π':>14} {'/γ_E':>14} {'/δ_F':>14} {'/ln2':>14} {'/ln3':>14}")
print(f"  {'-' * 90}")

interesting_sums = []
for i, j in itertools.combinations(range(5), 2):
    sum_val = values[i] + values[j]
    ratios = [sum_val / v for v in values]
    print(
        f"  {names[i] + ' + ' + names[j]:<20} {sum_val:>14.8f} {ratios[0]:>14.8f} {ratios[1]:>14.8f} {ratios[2]:>14.8f} {ratios[3]:>14.8f} {ratios[4]:>14.8f}")

    for k, (name, val) in enumerate(constants.items()):
        for mult in [1, 2, 3, 4, 5, 6, 1 / 2, 1 / 3, 1 / 4, 1 / 6, 2 / 3, 3 / 2]:
            if abs(sum_val - mult * val) / (mult * val) < 0.05:
                interesting_sums.append(
                    f"  {names[i]} + {names[j]} ≈ {mult}×{name} (откл: {abs(sum_val - mult * val) / (mult * val) * 100:.2f}%)")

if interesting_sums:
    print(f"\n  ИНТЕРЕСНЫЕ СОВПАДЕНИЯ:")
    for item in interesting_sums:
        print(item)

# ============================================================
# 2. МУЛЬТИПЛИКАТИВНЫЙ АНАЛИЗ
# ============================================================
print(f"\n{'=' * 70}")
print("2. МУЛЬТИПЛИКАТИВНЫЙ АНАЛИЗ")
print(f"{'=' * 70}")

print("\n2a. ПОПАРНЫЕ ПРОИЗВЕДЕНИЯ:")
for i, j in itertools.combinations(range(5), 2):
    prod = values[i] * values[j]
    print(f"  {names[i]} × {names[j]:<6} = {prod:>12.8f}")

print("\n2b. ПОПАРНЫЕ ОТНОШЕНИЯ:")
for i, j in itertools.permutations(range(5), 2):
    if i != j:
        ratio = values[i] / values[j]
        # Проверяем, не близко ли отношение к целому или известной константе
        marker = ""
        for mult in [1, 2, 3, 4, 6, 1 / 2, 1 / 3, 1 / 4, 1 / 6]:
            if abs(ratio - mult) < 0.05:
                marker = f" ≈ {mult}"
                break
        for c_name, c_val in constants.items():
            if abs(ratio - c_val) / c_val < 0.05:
                marker = f" ≈ {c_name}"
                break
        print(f"  {names[i]} / {names[j]:<6} = {ratio:>12.8f}{marker}")

# ============================================================
# 3. ПРОИЗВЕДЕНИЕ ВСЕХ ПЯТИ
# ============================================================
print(f"\n{'=' * 70}")
print("3. ПРОИЗВЕДЕНИЕ ВСЕХ ПЯТИ")
print(f"{'=' * 70}")

product_all = 1.0
for v in values:
    product_all *= v

print(f"\n  π × γ_E × δ_F × ln2 × ln3 = {product_all:.8f}")

# Проверка: не связано ли произведение с известными числами?
print(f"\n  Сравнение с известными числами:")
for name, val in [('e', math.e), ('√2', math.sqrt(2)), ('√3', math.sqrt(3)),
                  ('√π', math.sqrt(math.pi)), ('ln π', math.log(math.pi)),
                  ('π²', math.pi ** 2), ('1', 1.0), ('2', 2.0), ('3', 3.0)]:
    ratio = product_all / val
    print(f"    Π / {name:<6} = {ratio:.8f}")

# ============================================================
# 4. ТРОЙНЫЕ КОМБИНАЦИИ
# ============================================================
print(f"\n{'=' * 70}")
print("4. ТРОЙНЫЕ КОМБИНАЦИИ")
print(f"{'=' * 70}")

print("\n  (a × b) / c:")
interesting_triples = []
for i, j, k in itertools.permutations(range(5), 3):
    if i != j and j != k and i != k:
        result = values[i] * values[j] / values[k]
        # Проверяем близость к константам
        for c_name, c_val in constants.items():
            if abs(result - c_val) / c_val < 0.1:
                interesting_triples.append(
                    f"    ({names[i]} × {names[j]}) / {names[k]} = {result:.6f} ≈ {c_name} (откл: {abs(result - c_val) / c_val * 100:.2f}%)")
                break

if interesting_triples:
    for item in interesting_triples:
        print(item)

print("\n  (a × b × c):")
for i, j, k in itertools.combinations(range(5), 3):
    prod3 = values[i] * values[j] * values[k]
    for c_name, c_val in constants.items():
        if abs(prod3 - c_val) / c_val < 0.5:
            print(f"    {names[i]} × {names[j]} × {names[k]} = {prod3:.6f} ≈ {c_name}")

# ============================================================
# 5. РЕКУРРЕНТНЫЕ СООТНОШЕНИЯ
# ============================================================
print(f"\n{'=' * 70}")
print("5. ПОИСК РЕКУРРЕНТНЫХ СООТНОШЕНИЙ")
print(f"{'=' * 70}")

# Упорядочим константы по возрастанию
sorted_names = [n for _, n in sorted(zip(values, names))]
sorted_vals = sorted(values)

print(f"\n  Упорядоченные по возрастанию:")
for n, v in zip(sorted_names, sorted_vals):
    print(f"    {n:<6} = {v:.10f}")

# Разности между соседними
print(f"\n  Разности между соседними (по возрастанию):")
for i in range(4):
    diff = sorted_vals[i + 1] - sorted_vals[i]
    print(f"    {sorted_names[i + 1]} - {sorted_names[i]} = {diff:.6f}")

# Отношения соседних
print(f"\n  Отношения соседних (по возрастанию):")
for i in range(4):
    ratio = sorted_vals[i + 1] / sorted_vals[i]
    for mult in [1, 2, 3, 4, 1 / 2, 1 / 3, 1 / 4, 3 / 2, 4 / 3, 2 / 3]:
        if abs(ratio - mult) < 0.1:
            print(f"    {sorted_names[i + 1]} / {sorted_names[i]} = {ratio:.4f} ≈ {mult}")
            break
    else:
        print(f"    {sorted_names[i + 1]} / {sorted_names[i]} = {ratio:.4f}")

# ============================================================
# 6. СВЯЗЬ С N
# ============================================================
print(f"\n{'=' * 70}")
print("6. СВЯЗЬ С N (через геометрический резонанс)")
print(f"{'=' * 70}")

K = 6.0
lnK = math.log(K)
lnN_from_pi = (K - lnK) / (1 / 3 - 1 / math.pi)
print(f"\n  ln N = (K - lnK) / (1/3 - 1/π) = {lnN_from_pi:.4f}")
print(f"  Содержит: K=6, π")
print(f"  Не содержит явно: γ_E, δ_F")
print(f"  Но косвенно: γ_E входит через ζ-регуляризацию γ_i")
print(f"               δ_F входит через условие устойчивости TSCO")