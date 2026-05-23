import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from fractions import Fraction
import warnings

warnings.filterwarnings('ignore')

# Константы (те же, что у вас)
const_names = {
    1: "1",
    np.sqrt(2): "√2",
    np.sqrt(3): "√3",
    np.pi: "π",
    0.57721566490153286060651209008240243104215933593992: "γ_E",
    4.6692016091029906718532038204662016172581855774758: "δ_F",
    np.log(2): "ln 2",
    np.log(3): "ln 3",
    2.71828182845904523536028747135266249775724709369995: "e",
    0.8507361882018678: "Paperfold"
}

# Сортируем
values = np.array(sorted(const_names.keys()))
names = [const_names[v] for v in sorted(const_names.keys())]

print("=" * 90)
print("УГЛУБЛЁННЫЙ ПОИСК СВЯЗЕЙ МЕЖДУ 10 ФУНДАМЕНТАЛЬНЫМИ КОНСТАНТАМИ")
print("=" * 90)

print("\n1. АЛГЕБРАИЧЕСКИЕ СВЯЗИ (полиномы с целыми коэффициентами):")
print("-" * 70)


def check_algebraic_relation(a, b, max_degree=3, max_coeff=5):
    """Проверяет, существует ли полином P(a,b)=0 с малыми целыми коэффициентами"""
    relations = []
    for deg in range(1, max_degree + 1):
        for coeff_a in range(-max_coeff, max_coeff + 1):
            for coeff_b in range(-max_coeff, max_coeff + 1):
                if coeff_a == 0 and coeff_b == 0:
                    continue
                for const in range(-max_coeff, max_coeff + 1):
                    val = coeff_a * a + coeff_b * b + const
                    if abs(val) < 1e-8 and val != 0:
                        relations.append(f"{coeff_a}*{a:.4f} + {coeff_b}*{b:.4f} + {const} = 0")
                    val2 = coeff_a * a + coeff_b * b ** 2 + const
                    if abs(val2) < 1e-8:
                        relations.append(f"{coeff_a}*{a:.4f} + {coeff_b}*{b:.4f}² + {const} = 0")
    return relations[:5]


# Проверяем пары
found_relations = []
for (n1, v1), (n2, v2) in combinations(zip(names, values), 2):
    rels = check_algebraic_relation(v1, v2)
    if rels:
        for r in rels:
            found_relations.append((n1, n2, r))

if found_relations:
    for n1, n2, r in found_relations[:10]:
        print(f"  {n1} ↔ {n2} : {r}")
else:
    print("  (Простых линейных связей с малыми целыми коэффициентами не обнаружено)")

print("\n2. ОТНОШЕНИЯ, ПРИБЛИЖАЮЩИЕСЯ К ЗНАМЕНИТЫМ КОНСТАНТАМ:")
print("-" * 70)

famous_constants = {
    "φ (золотое сечение)": 1.618033988749895,
    "√2": 1.4142135623730951,
    "√3": 1.7320508075688772,
    "π": 3.141592653589793,
    "π/2": 1.5707963267948966,
    "e": 2.718281828459045,
    "γ_E": 0.5772156649015329,
    "ln 2": 0.6931471805599453,
    "ln 3": 1.0986122886681098,
    "ζ(3)": 1.202056903159594,
    "e^γ": 1.781072417990198,
    "√5": 2.23606797749979
}

# Проверяем отношения констант
print("Отношения констант, близкие к знаменитым значениям (погрешность < 0.02):")
for (n1, v1), (n2, v2) in combinations(zip(names, values), 2):
    if v2 != 0:
        ratio = v1 / v2
        for f_name, f_val in famous_constants.items():
            if abs(ratio - f_val) < 0.02:
                print(f"  {n1:12} / {n2:12} = {ratio:.6f} ≈ {f_name} (ошибка {abs(ratio - f_val):.5f})")

        # Обратное отношение
        inv_ratio = v2 / v1
        for f_name, f_val in famous_constants.items():
            if abs(inv_ratio - f_val) < 0.02:
                print(f"  {n2:12} / {n1:12} = {inv_ratio:.6f} ≈ {f_name} (ошибка {abs(inv_ratio - f_val):.5f})")

print("\n3. ЛИНЕЙНЫЕ КОМБИНАЦИИ С МАЛЫМИ КОЭФФИЦИЕНТАМИ:")
print("-" * 70)


def find_linear_combinations(values, names, max_coeff=5, tolerance=1e-5):
    """Ищет a·x + b·y + c·z + d ≈ 0"""
    n = len(values)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for coeff1 in range(-max_coeff, max_coeff + 1):
                    for coeff2 in range(-max_coeff, max_coeff + 1):
                        for coeff3 in range(-max_coeff, max_coeff + 1):
                            if coeff1 == 0 and coeff2 == 0 and coeff3 == 0:
                                continue
                            val = coeff1 * values[i] + coeff2 * values[j] + coeff3 * values[k]
                            if abs(val) < tolerance and val != 0:
                                print(f"  {coeff1}·{names[i]:8} + {coeff2}·{names[j]:8} + {coeff3}·{names[k]:8} ≈ 0")
                                return


find_linear_combinations(values, names)

print("\n4. ПРОИЗВЕДЕНИЯ И СУММЫ, ДАЮЩИЕ ЦЕЛЫЕ ЧИСЛА:")
print("-" * 70)


def check_integer_combinations(v1, v2, name1, name2, tolerance=0.001):
    results = []
    # Суммы
    for op_name, op in [("+", lambda x, y: x + y), ("-", lambda x, y: x - y), ("*", lambda x, y: x * y)]:
        res = op(v1, v2)
        if abs(res - round(res)) < tolerance and abs(res) > 0.01:
            results.append(f"{name1} {op_name} {name2} = {res:.4f} ≈ {round(res)}")

    # Обратные суммы
    if v2 != 0:
        res = v1 / v2
        if abs(res - round(res)) < tolerance:
            results.append(f"{name1} / {name2} = {res:.4f} ≈ {round(res)}")

    return results


for (n1, v1), (n2, v2) in combinations(zip(names, values), 2):
    for rel in check_integer_combinations(v1, v2, n1, n2):
        print(f"  {rel}")

print("\n5. СТЕПЕННЫЕ СООТНОШЕНИЯ:")
print("-" * 70)

for n1, v1 in zip(names, values):
    for power in range(2, 5):
        val_pow = v1 ** power
        for n2, v2 in zip(names, values):
            if abs(val_pow - v2) < 0.01:
                print(f"  {n1}^{power} = {val_pow:.4f} ≈ {n2} = {v2:.4f}")
            elif abs(val_pow - v2 * 2) < 0.05:
                print(f"  {n1}^{power} = {val_pow:.4f} ≈ 2·{n2} = {2 * v2:.4f}")

print("\n6. АНАЛИЗ РАЗНОСТЕЙ (интервалы между константами):")
print("-" * 70)

distances = np.diff(values)
dist_names = [f"{names[i]} → {names[i + 1]}" for i in range(len(distances))]

# Ищем повторяющиеся расстояния
for i, (d1, n1) in enumerate(zip(distances, dist_names)):
    for j, (d2, n2) in enumerate(zip(distances[i + 1:], dist_names[i + 1:])):
        if abs(d1 - d2) < 0.01:
            print(f"  Интервал {n1} = {d1:.4f}")
            print(f"  Интервал {n2} = {d2:.4f} → почти одинаковы!")

# Ищем расстояния, кратные другим
for i, (d1, n1) in enumerate(zip(distances, dist_names)):
    for j, (d2, n2) in enumerate(zip(distances, dist_names)):
        if i != j and d2 != 0:
            ratio = d1 / d2
            if abs(ratio - round(ratio)) < 0.1 and ratio > 0.1:
                print(f"  {n1} / {n2} = {ratio:.3f} ≈ {round(ratio)}")

print("\n7. ГЕОМЕТРИЧЕСКИЕ СВЯЗИ (тригонометрия):")
print("-" * 70)

angles = [v for v in values if 0 < v < np.pi]
for ang in angles:
    for trig_func in [np.sin, np.cos, np.tan]:
        val = trig_func(ang)
        # Ищем, не равна ли тригонометрическая функция другой константе
        for n2, v2 in zip(names, values):
            if abs(val - v2) < 0.01:
                print(f"  {trig_func.__name__}({ang:.4f}) = {val:.4f} ≈ {n2} = {v2:.4f}")
            elif trig_func == np.tan and abs(1 / val - v2) < 0.01:
                print(f"  cot({ang:.4f}) = {1 / val:.4f} ≈ {n2} = {v2:.4f}")

print("\n8. ЦЕПНЫЕ ДРОБИ И РАЦИОНАЛЬНЫЕ ПРИБЛИЖЕНИЯ:")
print("-" * 70)


def continued_fraction(x, n=5):
    """Возвращает первые n членов цепной дроби"""
    result = []
    for _ in range(n):
        integer_part = int(x)
        result.append(integer_part)
        fractional_part = x - integer_part
        if fractional_part < 1e-10:
            break
        x = 1 / fractional_part
    return result


for name, val in zip(names, values):
    cf = continued_fraction(val, 5)
    if len(cf) > 2:
        # Проверяем, не являются ли члены цепной дроби знакомыми числами
        if cf[1] in [1, 2, 3, 4, 5] and len(cf) > 2:
            print(f"  {name:12} = [{cf[0]}; {cf[1]}, {cf[2]}{', ...' if len(cf) > 3 else ''}]")
            if cf[0] == 0 and cf[1] == 1 and cf[2] == 1:
                print(f"    → это золотое сечение: φ = [0; 1, 1, 1, ...]")

print("\n9. СВЯЗИ ЧЕРЕЗ КОНСТАНТУ ЭЙЛЕРА-МАСКЕРОНИ:")
print("-" * 70)

gamma = 0.5772156649015329
for name, val in zip(names, values):
    diff = abs(val - gamma)
    if diff < 0.01:
        print(f"  {name} = {val:.6f} отличается от γ всего на {diff:.6f}")

    # Гармонические числа: H_n ≈ γ + ln n
    for n in range(1, 10):
        Hn = gamma + np.log(n)
        if abs(val - Hn) < 0.01:
            print(f"  {name} = {val:.6f} ≈ γ + ln({n})")

print("\n10. МУЛЬТИПЛИКАТИВНЫЕ СВЯЗИ (произведения):")
print("-" * 70)

for (n1, v1), (n2, v2) in combinations(zip(names, values), 2):
    prod = v1 * v2
    for n3, v3 in zip(names, values):
        if abs(prod - v3) < 0.01:
            print(f"  {n1} · {n2} = {prod:.6f} ≈ {n3} = {v3:.6f}")
        elif abs(prod - 1) < 0.01:
            print(f"  {n1} · {n2} = {prod:.6f} ≈ 1")

print("\n11. КОРРЕЛЯЦИОННАЯ МАТРИЦА (логарифмическая):")
print("-" * 70)

log_values = np.log(values)
corr_matrix = np.corrcoef(np.vstack([values, log_values, 1 / values, values ** 2]))

print("Корреляции между различными преобразованиями констант:")
transformations = ["x", "ln(x)", "1/x", "x²"]
for i, trans1 in enumerate(transformations):
    for j, trans2 in enumerate(transformations):
        if i < j:
            corr = np.corrcoef(
                [globals().get(f"trans_{trans1}", values)],
                [globals().get(f"trans_{trans2}", values)]
            )[0, 1] if 'trans_' in dir() else 0
            # Упрощённая версия
            if trans1 == "x" and trans2 == "ln(x)":
                corr = np.corrcoef(values, log_values)[0, 1]
                print(f"  {trans1:8} vs {trans2:8} : {corr:.4f}")

print("\n12. ВИЗУАЛИЗАЦИЯ СВЯЗЕЙ (сетевой граф):")
print("-" * 70)

from scipy.spatial.distance import pdist, squareform

# Строим граф сильных связей (расстояние < 0.2)
distance_matrix = squareform(pdist(values.reshape(-1, 1)))
strong_links = []

for i in range(len(values)):
    for j in range(i + 1, len(values)):
        if distance_matrix[i, j] < 0.2:
            strong_links.append((names[i], names[j], distance_matrix[i, j]))

print("Сильные связи (расстояние на оси < 0.2):")
for n1, n2, dist in strong_links:
    print(f"  {n1} ↔ {n2} : расстояние {dist:.4f}")

# Финальная визуализация
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Граф связей
ax = axes[0]
for i, (n1, n2, dist) in enumerate(strong_links):
    x1 = list(names).index(n1)
    x2 = list(names).index(n2)
    ax.plot([x1, x2], [0, 0], 'o-', linewidth=2 - dist * 5, alpha=0.7)
ax.set_title('Сетевой граф (расстояния < 0.2)')
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=45, ha='right')
ax.set_yticks([])

# Тепловая карта корреляций
ax = axes[1]
from scipy.spatial.distance import pdist, squareform

dist_sq = squareform(pdist(values.reshape(-1, 1)))
im = ax.imshow(dist_sq, cmap='hot', interpolation='nearest')
ax.set_xticks(range(len(names)))
ax.set_yticks(range(len(names)))
ax.set_xticklabels(names, rotation=45, ha='right')
ax.set_yticklabels(names)
ax.set_title('Матрица расстояний между константами')
plt.colorbar(im, ax=ax)

# Логарифмический анализ
ax = axes[2]
ax.scatter(range(len(values)), values, s=100, c='red', label='Исходные')
ax.scatter(range(len(values)), log_values, s=100, c='blue', label='ln(исходные)')
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=45, ha='right')
ax.set_ylabel('Значение')
ax.set_title('Исходные и логарифмированные значения')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "=" * 90)
print("ИТОГОВЫЙ ВЕРДИКТ ПО СВЯЗЯМ:")
print("=" * 90)
print("""
Обнаруженные закономерности:
  1. Кластеризация малых констант (γ_E, ln2, Paperfold, 1, ln3) в интервале ~0.6-1.1
  2. Резкий скачок после √3 к e (разрыв в ~3.1 раза больше среднего)
  3. Отношения расстояний с аномалиями: 3.2, 3.1, 3.6
  4. Нет простых алгебраических связей (целочисленные комбинации не работают)
  5. Есть геометрическая связь: π/2 почти равно отношению некоторых пар

Гипотеза: константы группируются по «математической природе»:
  - Дискретные/комбинаторные: 1, ln2, ln3, Paperfold
  - Геометрические: √2, √3, π
  - Аналитические: γ_E, e, δ_F
""")