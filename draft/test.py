import numpy as np
import itertools
import math

# Исходные величины
K = 8
p = 0.0525
N = 1e122
h = 1.054e-34
Ru = 3.3e26
c = 299000000
G = 6.67e-11
lp = 1.6e-35

# Целевые значения
R_target = 6.02e23
KB_target = 1.38e-23

# Все доступные исходные величины
constants = {
    'K': K,
    'p': p,
    'N': N,
    'h': h,
    'Ru': Ru,
    'c': c,
    'G': G,
    'lp': lp
}

# Список всех доступных величин (включая вычисляемые)
available_terms = {
    'K': K,
    'p': p,
    'N': N,
    'h': h,
    'Ru': Ru,
    'c': c,
    'G': G,
    'lp': lp,
    'sqrt(p*K)': math.sqrt(p * K),
    'sqrt(K)': math.sqrt(K),
    'sqrt(p)': math.sqrt(p),
    #'ln(2)': math.log(2),
    'ln(p)': math.log(p) if p > 0 else 0,
    'ln(N)': math.log(N) if N > 0 else 0,
    'ln(K)': math.log(K) if K > 0 else 0,
    '1/p': 1 / p if p != 0 else float('inf'),
    '1/K': 1 / K if K != 0 else float('inf'),
    '1/N': 1 / N if N != 0 else float('inf'),
    '1/ln(p*K)': 1 / math.log(p) if p > 0 and math.log(p) != 0 else float('inf'),
    '1/ln(p)': 1 / math.log(p) if p > 0 and math.log(p) != 0 else float('inf'),
    '1/ln(N)': 1 / math.log(N) if N > 0 and math.log(N) != 0 else float('inf'),
    '1/ln(K)': 1 / math.log(K) if K > 0 and math.log(K) != 0 else float('inf'),
    '1/sqrt(p*K)': 1 / math.sqrt(p * K) if p * K > 0 else float('inf'),
    '1/sqrt(p)': 1 / math.sqrt(p) if p > 0 else float('inf'),
    '1/sqrt(K)': 1 / math.sqrt(K) if K > 0 else float('inf'),
    'p^2': p ** 2,
    'K^2': K ** 2,
    'p^3': p ** 3,
    'K^3': K ** 3,
    'p^5': p ** 5,
    'K^5': K ** 5,
    'N^2': N ** 2 if abs(N) < 1e100 else float('inf'),
    'N^(1/2)': math.sqrt(N) if N >= 0 else float('inf'),
    'N^(1/3)': N ** (1 / 3) if N >= 0 else float('inf'),
    'N^(1/4)': N ** (1 / 4) if N >= 0 else float('inf'),
    'N^(1/5)': N ** (1 / 5) if N >= 0 else float('inf'),
    'N^(1/6)': N ** (1 / 6) if N >= 0 else float('inf'),
    'N^(2/3)': N ** (2 / 3) if N >= 0 else float('inf'),
    'p^(1/2)': math.sqrt(p),
    'p^(1/3)': p ** (1 / 3),
    'p^(1/4)': p ** (1 / 4),
    'p^(1/6)': p ** (1 / 6),
    'p^(-1/2)': p ** (-1 / 2) if p > 0 else float('inf'),
    'p^(-1/3)': p ** (-1 / 3) if p > 0 else float('inf'),
    'p^(-1/4)': p ** (-1 / 4) if p > 0 else float('inf'),
    'p^(-1/6)': p ** (-1 / 6) if p > 0 else float('inf'),
    'p^(2/3)': p ** (2 / 3),
    'p^(-2/3)': p ** (-2 / 3) if p > 0 else float('inf'),
    'K^(1/2)': math.sqrt(K),
    'K^(1/3)': K ** (1 / 3),
    'K^(1/4)': K ** (1 / 4),
    'K^(1/6)': K ** (1 / 6),
    'K^(-1/2)': K ** (-1 / 2) if K > 0 else float('inf'),
    'K^(-1/3)': K ** (-1 / 3) if K > 0 else float('inf'),
    'K^(-1/4)': K ** (-1 / 4) if K > 0 else float('inf'),
    'K^(-1/6)': K ** (-1 / 6) if K > 0 else float('inf'),
    'K^(2/3)': K ** (2 / 3),
    'K^(-2/3)': K ** (-2 / 3) if K > 0 else float('inf'),
}

# Фильтруем бесконечные и некорректные значения
filtered_terms = {}
for key, value in available_terms.items():
    if (isinstance(value, (int, float)) and
            not math.isinf(value) and
            not math.isnan(value) and
            abs(value) > 0 and
            abs(value) < float('inf')):
        filtered_terms[key] = value

available_terms = filtered_terms
term_names = list(available_terms.keys())

print(f"Доступно {len(term_names)} различных членов")
print(f"Целевое R = {R_target:.3e}")
print(f"Целевое KB = {KB_target:.3e}")
print("-" * 80)


# Функция для вычисления относительной ошибки
def relative_error(value, target):
    if target == 0:
        return float('inf')
    return abs(value - target) / target


# Хранилище результатов
results_R = []
results_KB = []

# Максимальное количество точных совпадений для каждого целевого значения
max_results = 200

# Проверяем формулы разной длины (от 2 до 6 членов)
MAX = 7 #7
for num_terms in range(2, MAX):
    print(f"Проверяем формулы с {num_terms} членами...")

    # Генерируем все возможные комбинации членов
    combinations = list(itertools.combinations(term_names, num_terms))
    print(f"Всего комбинаций: {len(combinations)}")

    for combo in combinations:
        # Вычисляем произведение всех членов
        product_value = 1.0
        formula_parts = []

        for term in combo:
            value = available_terms[term]
            if math.isinf(value) or math.isnan(value) or value == 0:
                product_value = float('inf')
                break
            product_value *= value
            formula_parts.append(term)

        if math.isinf(product_value) or math.isnan(product_value):
            continue

        # Проверяем на совпадение с R
        error_R = relative_error(product_value, R_target)
        if error_R <= 0.18:  # 15% точность
            formula = " * ".join(formula_parts)
            results_R.append((error_R, product_value, formula, num_terms))

            if len(results_R) >= max_results:
                break

        # Проверяем на совпадение с KB
        error_KB = relative_error(product_value, KB_target)
        if error_KB <= 0.18:  # 15% точность
            formula = " * ".join(formula_parts)
            results_KB.append((error_KB, product_value, formula, num_terms))

            if len(results_KB) >= max_results:
                break

    # Сортируем результаты по точности
    results_R.sort(key=lambda x: x[0])
    results_KB.sort(key=lambda x: x[0])

    # Если набрали достаточно результатов, прерываемся
    if len(results_R) >= max_results and len(results_KB) >= max_results:
        break

print("\n" + "=" * 80)
print("РЕЗУЛЬТАТЫ ДЛЯ R (число Авогадро):")
print("=" * 80)

if results_R:
    print(f"\nНайдено {len(results_R)} формул с точностью 18% или лучше")
    print("\nТоп-20 результатов:")
    print("-" * 120)
    print(f"{'№':<3} {'Отн. ошибка':<12} {'Полученное значение':<20} {'Целевое':<20} {'Формула':<60} {'Членов':<6}")
    print("-" * 120)

    for i, (error, value, formula, num_terms) in enumerate(results_R[:20], 1):
        print(f"{i:<3} {error * 100:<10.4f}% {value:<20.3e} {R_target:<20.3e} {formula:<60} {num_terms:<6}")
else:
    print("Не найдено ни одной формулы для R с требуемой точностью")

print("\n" + "=" * 80)
print("РЕЗУЛЬТАТЫ ДЛЯ KB (постоянная Больцмана):")
print("=" * 80)

if results_KB:
    print(f"\nНайдено {len(results_KB)} формул с точностью 18% или лучше")
    print("\nТоп-20 результатов:")
    print("-" * 120)
    print(f"{'№':<3} {'Отн. ошибка':<12} {'Полученное значение':<20} {'Целевое':<20} {'Формула':<60} {'Членов':<6}")
    print("-" * 120)

    for i, (error, value, formula, num_terms) in enumerate(results_KB[:20], 1):
        print(f"{i:<3} {error * 100:<10.4f}% {value:<20.3e} {KB_target:<20.3e} {formula:<60} {num_terms:<6}")
else:
    print("Не найдено ни одной формулы для KB с требуемой точностью")

# Сохраняем все результаты в файлы
with open('results_R.txt', 'w', encoding='utf-8') as f:
    f.write("Результаты для R (число Авогадро):\n")
    f.write("=" * 80 + "\n")
    for error, value, formula, num_terms in results_R:
        f.write(f"Ошибка: {error * 100:.4f}%, Значение: {value:.3e}, "
                f"Формула: {formula}, Членов: {num_terms}\n")

with open('results_KB.txt', 'w', encoding='utf-8') as f:
    f.write("Результаты для KB (постоянная Больцмана):\n")
    f.write("=" * 80 + "\n")
    for error, value, formula, num_terms in results_KB:
        f.write(f"Ошибка: {error * 100:.4f}%, Значение: {value:.3e}, "
                f"Формула: {formula}, Членов: {num_terms}\n")

print(f"\nВсе результаты сохранены в файлы 'results_R.txt' и 'results_KB.txt'")
print(f"Всего найдено {len(results_R)} формул для R и {len(results_KB)} формул для KB")