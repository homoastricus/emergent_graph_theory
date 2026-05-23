import numpy as np
from itertools import product

# Константы с указанием типа
constants = {
    # Алгебраические (могут давать точные рациональные результаты)
    "√2": {"value": 1.4142135623730951, "type": "algebraic"},
    "√3": {"value": 1.7320508075688772, "type": "algebraic"},

    # Трансцендентные
    "π": {"value": 3.141592653589793, "type": "transcendental"},
    "e": {"value": 2.718281828459045, "type": "transcendental"},
    "γ_E": {"value": 0.5772156649015329, "type": "transcendental"},
    "ln2": {"value": 0.6931471805599453, "type": "transcendental"},
    "ln3": {"value": 1.0986122886681098, "type": "transcendental"},
    "δ_Feigenbaum": {"value": 4.669201609102990, "type": "transcendental"},

    # Paperfold — скорее всего трансцендентная (константа складки бумаги)
    "Paperfold": {"value": 0.85073618820186726036, "type": "transcendental"}  # исправлено!
}

# Целевые значения
targets = [1 / 6, 1 / 4, 1 / 3, 0.5, 1, 2, 3, 4, 5, 6]

# Выбираем только трансцендентные константы
transcendental_names = [name for name, info in constants.items() if info["type"] == "transcendental"]
transcendental_values = np.array([constants[name]["value"] for name in transcendental_names])

# Параметры поиска
max_power = 2
top_k = 20

print("=" * 80)
print("ИССЛЕДОВАНИЕ ТОЛЬКО ТРАНСЦЕНДЕНТНЫХ КОНСТАНТ")
print("=" * 80)
print(f"\nИсключены алгебраические: √2, √3 (они дают тривиальные точные результаты)")
print(f"\nТрансцендентные константы: {', '.join(transcendental_names)}")
print(f"Paperfold = {constants['Paperfold']['value']:.15f}...")
print(f"\nЦели: {targets}")
print(f"Диапазон степеней: -{max_power}..{max_power}")
print(f"Всего комбинаций: {(2 * max_power + 1) ** len(transcendental_names):.2e}")

# Логарифмы трансцендентных констант
log_values = np.log(transcendental_values)

# Поиск результатов
best_by_target = {target: [] for target in targets}
powers_range = range(-max_power, max_power + 1)
total = 0

for powers in product(powers_range, repeat=len(transcendental_names)):
    if all(p == 0 for p in powers):
        continue

    total += 1
    log_product = np.sum(powers * log_values)
    product_value = np.exp(log_product)

    for target in targets:
        deviation = abs(product_value - target)
        best_by_target[target].append((deviation, product_value, powers))
        best_by_target[target].sort(key=lambda x: x[0])
        best_by_target[target] = best_by_target[target][:top_k]

print(f"\nОбработано комбинаций: {total:,}")

# Вывод результатов
print("\n" + "=" * 80)
print("РЕЗУЛЬТАТЫ ДЛЯ ТРАНСЦЕНДЕНТНЫХ КОНСТАНТ")
print("=" * 80)

for target in targets:
    print(f"\n{'─' * 80}")
    print(f"ЦЕЛЬ = {target}")
    print(f"{'─' * 80}")

    for idx, (dev, prod, powers) in enumerate(best_by_target[target][:7], 1):
        # Формируем формулу
        parts = []
        for i, (name, p) in enumerate(zip(transcendental_names, powers)):
            if p > 0:
                parts.append(f"{name}^{p}" if p != 1 else name)
            elif p < 0:
                parts.append(f"1/{name}^{abs(p)}" if abs(p) != 1 else f"1/{name}")

        formula = " × ".join(parts) if parts else "1"
        print(f"{idx:2}. {formula} = {prod:.15f}  (откл. {dev:.2e})")

print("\n" + "=" * 80)
print("ГОТОВО")
print("=" * 80)