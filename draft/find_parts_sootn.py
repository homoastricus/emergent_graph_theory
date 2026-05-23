import math
import itertools
import random
import numpy as np

# 1. Данные (без изменений)
ratios = {
    "m_mu/m_e": 206.7682830,
    "m_tau/m_mu": 16.817,
    "m_tau/m_e": 3477.48,
    "m_p/m_e": 1836.15267343,
    "m_n/m_p": 1.00137841893,
    "m_pi0/m_e": 264.142,
    "m_K0/m_e": 974.0,
    "m_Lambda/m_e": 2182.6,
}

constants = {
    "pi": math.pi,
    "e": math.e,
    "sqrt2": math.sqrt(2),
    "sqrt3": math.sqrt(3),
    "sqrt5": math.sqrt(5),
    "phi": (1 + math.sqrt(5)) / 2,
    "alpha": 1 / 137.035999084,
    "alpha_inv": 137.035999084,
}

# Урезанные степени (главное — целые степени, убираем дробные)
powers = [-3, -2, -1, 0, 1, 2, 3]

# Коэффициенты (меньше, только самые вероятные)
coeffs = [1 / 3, 1 / 2, 2 / 3, 1, 3 / 2, 2, 3, 4, 5, 6]

# 2. ПРЕДВЫЧИСЛЕНИЕ ВСЕХ ВОЗМОЖНЫХ base^n
# Это ключевая оптимизация: вычисляем один раз все степени констант
precomputed_pows = {}
for name, val in constants.items():
    for p in powers:
        precomputed_pows[(name, p)] = val ** p


# 3. Функция поиска с предвычислением и ранним отсечением
def find_best_formula_optimized(target, max_constants=2):
    best_err = 1.0  # начинаем с 100% ошибки
    best_expr = None
    best_val = None

    # 1 константа
    for (name1, p1), val1 in precomputed_pows.items():
        if p1 == 0: continue
        base = val1
        for k in coeffs:
            guess = k * base
            if guess == 0: continue
            err = abs(guess - target) / target
            if err < best_err and err < 0.01:  # раннее отсечение: лучше 1%
                best_err = err
                best_expr = f"{k:.4g} * {name1}^{p1}"
                best_val = guess

    # 2 константы (только если с одной не нашли очень хорошо)
    if best_err > 1e-6 and max_constants >= 2:
        # Используем комбинации индексов, но с предвычислением
        items = list(precomputed_pows.items())
        for i in range(len(items)):
            (name1, p1), val1 = items[i]
            for j in range(i + 1, len(items)):
                (name2, p2), val2 = items[j]
                base = val1 * val2
                for k in coeffs:
                    guess = k * base
                    if guess == 0: continue
                    err = abs(guess - target) / target
                    if err < best_err and err < 0.005:  # для двух констант требуем 0.5%
                        best_err = err
                        best_expr = f"{k:.4g} * {name1}^{p1} * {name2}^{p2}"
                        best_val = guess

    return best_err, best_expr, best_val


# 4. Монте-Карло (тоже оптимизированный)
def monte_carlo_optimized(target, n_random=5000, max_constants=2):
    best_real_err, _, _ = find_best_formula_optimized(target, max_constants)

    better_count = 0
    # Генерируем случайные числа в логарифмической шкале
    log_low, log_high = -1, 4  # от 0.1 до 10000
    for _ in range(n_random):
        rand_target = 10 ** random.uniform(log_low, log_high)
        rand_err, _, _ = find_best_formula_optimized(rand_target, max_constants)
        if rand_err < best_real_err:
            better_count += 1

    return better_count / n_random, best_real_err


# 5. Основной запуск (с кэшированием результатов для ускорения)
print("Оптимизированный поиск формул (предвычисление + раннее отсечение)\n")

# Кэш для Монте-Карло (чтобы не пересчитывать одно и то же)
monte_carlo_cache = {}

for name, target in ratios.items():
    print(f"\n--- {name} = {target:.6f} ---")

    err, expr, val = find_best_formula_optimized(target, max_constants=2)
    if expr is None:
        print("  Нет формул с ошибкой < 1%")
        continue

    print(f"  Лучшая: {expr} = {val:.8f}")
    print(f"  Ошибка: {err * 100:.6f}%")

    # Монте-Карло с кэшированием
    if name not in monte_carlo_cache:
        p_val, _ = monte_carlo_optimized(target, n_random=2000)
        monte_carlo_cache[name] = p_val
    else:
        p_val = monte_carlo_cache[name]

    print(f"  p-value: {p_val:.4f}")
    if p_val < 0.05 / len(ratios):
        print("  ✅ ПОТЕНЦИАЛЬНО ЗНАЧИМО (p < 0.05/popravka)")
    else:
        print("  ❌ Вероятно случайность")

# 6. Быстрая проверка известных формул
print("Известные формулы (быстрая проверка):")

alpha = constants["alpha"]
print(f"3π/(2α) = {3 * math.pi / (2 * alpha):.6f}  (m_mu/m_e = {ratios['m_mu/m_e']:.6f})")
print(f"6π^5 = {6 * math.pi ** 5:.6f}  (m_p/m_e = {ratios['m_p/m_e']:.6f})")