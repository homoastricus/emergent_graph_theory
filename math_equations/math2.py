import math
import itertools

components = {
    '3': 3,
    '6': 6,
    'ln6': math.log(6),
    'pi': math.pi,
    'sqrt3': math.sqrt(3),
    'sqrt2': math.sqrt(2)
}
names = list(components.keys())
vals = list(components.values())

target_low = 280.04
target_high = 280.12

pows_range = range(-2, 3)

found = []

# Предварительно вычисляем все степени для ускорения
powers_cache = {}
for i, val in enumerate(vals):
    for p in pows_range:
        powers_cache[(i, p)] = val ** p

for exp_pows in itertools.product(pows_range, repeat=len(components)):
    # Вычисляем X как произведение vals[i]^exp_pows[i]
    X = 1.0
    for i, p in enumerate(exp_pows):
        X *= powers_cache[(i, p)]

    if abs(X) > 20:
        continue

    exp_factor = math.exp(X)

    for base_mask in range(1, 2 ** len(components)):
        base_factor = 1.0
        for i in range(len(components)):
            if (base_mask >> i) & 1:
                base_factor *= vals[i]

        result = base_factor * exp_factor

        if target_low <= result <= target_high:
            base_names = [names[i] for i in range(len(components)) if (base_mask >> i) & 1]
            exp_parts = []
            for i in range(len(components)):
                p = exp_pows[i]
                if p != 0:
                    name = names[i]
                    if p == 1:
                        exp_parts.append(name)
                    elif p == -1:
                        exp_parts.append(f'1/{name}')
                    else:
                        exp_parts.append(f'{name}^{p}')

            exp_str = ' * '.join(exp_parts) if exp_parts else '1'

            found.append({
                'result': result,
                'base_factor': base_factor,
                'base_names': base_names,
                'X': X,
                'exp_factor': exp_factor,
                'exp_pows': exp_pows,
                'exp_str': exp_str,
                'num_base': len(base_names),
                'num_exp_factors': len(exp_parts),
                'has_abs_pow_gt1': any(abs(p) > 1 for p in exp_pows),
                'has_negative_pow': any(p < 0 for p in exp_pows)
            })

if not found:
    print("Ничего не найдено.")
else:
    # Сортируем: сначала по числу множителей в X, затем по числу в base, затем предпочтение степеням ±1
    found.sort(key=lambda x: (
        x['num_exp_factors'],
        x['num_base'],
        x['has_abs_pow_gt1'],
        x['has_negative_pow']
    ))

    print(f"Найдено {len(found)} вариантов. Топ-10 минималистичных:\n")

    for i, item in enumerate(found[:10], 1):
        print(f"--- Вариант {i} ---")
        print(f"Результат: {item['result']:.10f}")
        print(f"Base = произведение({', '.join(item['base_names'])}) = {item['base_factor']:.10f}")
        print(f"X = {item['exp_str']} = {item['X']:.10f}")
        print(f"e^X = {item['exp_factor']:.10f}")
        print(f"Выражение: {item['base_factor']:.10f} * e^{item['X']:.10f} = {item['result']:.10f}")
        print()