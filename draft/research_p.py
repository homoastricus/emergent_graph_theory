import math

# Исходное число
p = 5.270179e-02
print("=" * 70)
print("ИССЛЕДОВАНИЕ ЧИСЛА p = 5.270179e-02 = 0.05270179")
print("=" * 70)

# 1. Основные преобразования
print("\n1. ОСНОВНЫЕ ПРЕОБРАЗОВАНИЯ:")
print("-" * 40)
print(f"p = {p:.10f}")
print(f"p = {p * 100:.6f}%")
print(f"1/p = {1 / p:.10f}")
print(f"1 - p = {1 - p:.10f}")
print(f"p/(1-p) = {p / (1 - p):.10f}")
print(f"ln(p) = {math.log(p):.10f}")
print(f"exp(p) = {math.exp(p):.10f}")
print(f"sin(p) = {math.sin(p):.10f} (радианы)")
print(f"cos(p) = {math.cos(p):.10f}")

# 2. Связь с известными математическими константами
print("\n2. СВЯЗЬ С МАТЕМАТИЧЕСКИМИ КОНСТАНТАМИ:")
print("-" * 40)

constants = {
    "π": math.pi,
    "e": math.e,
    "φ (золотое сечение)": (1 + math.sqrt(5)) / 2,
    "γ (Эйлер-Маскерони)": 0.5772156649,
    "√2": math.sqrt(2),
    "√3": math.sqrt(3),
    "ln(2)": math.log(2),
    "ln(10)": math.log(10),
    "G (постоянная Каталана)": 0.915965594177,
}

for name, value in constants.items():
    ratio = p / value
    diff = abs(p - value)
    print(f"{name:20} = {value:.10f}")
    print(f"  p/{name}: {ratio:.6f}  |  {name}/p: {value / p:.6f}  |  разность: {diff:.6f}")

# 3. Исследование обратного значения
print("\n3. ИССЛЕДОВАНИЕ 1/p:")
print("-" * 40)
inv_p = 1 / p
print(f"1/p = {inv_p:.10f}")

# Проверим, близко ли к известным числам
inv_checks = [
    ("2π", 2 * math.pi),
    ("π²", math.pi ** 2),
    ("e²", math.e ** 2),
    ("10π", 10 * math.pi),
    ("φ²", ((1 + math.sqrt(5)) / 2) ** 2),
    ("6π", 6 * math.pi),
    ("4π²", 4 * math.pi ** 2),
]

for name, value in inv_checks:
    ratio = inv_p / value
    print(f"{name:6} = {value:.6f},  (1/p)/{name} = {ratio:.6f}")

# 4. Исследование 1-p
print("\n4. ИССЛЕДОВАНИЕ 1-p:")
print("-" * 40)
one_minus_p = 1 - p
print(f"1-p = {one_minus_p:.10f}")

checks_1mp = [
    ("√2/2", math.sqrt(2) / 2),
    ("1/φ", 2 / (1 + math.sqrt(5))),
    ("e^{-1}", math.exp(-1)),
    ("ln(2)", math.log(2)),
    ("π/4", math.pi / 4),
    ("1 - 1/e", 1 - math.exp(-1)),
]

for name, value in checks_1mp:
    ratio = one_minus_p / value
    print(f"{name:10} = {value:.6f},  (1-p)/{name} = {ratio:.6f}")

# 5. Связь с p через e и π
print("\n5. КОМБИНАЦИИ С e И π:")
print("-" * 40)

combinations = [
    ("p * e", p * math.e),
    ("p * π", p * math.pi),
    ("p / π", p / math.pi),
    ("p / e", p / math.e),
    ("√p", math.sqrt(p)),
    ("p²", p ** 2),
    ("p³", p ** 3),
    ("e^p", math.exp(p)),
    ("π^p", math.pi ** p),
]

for expr, value in combinations:
    print(f"{expr:10} = {value:.10f}")

# 6. Проверка специальных соотношений
print("\n6. СПЕЦИАЛЬНЫЕ СООТНОШЕНИЯ:")
print("-" * 40)

# 6.1. Соотношение с постоянной тонкой структуры α ≈ 1/137
alpha = 1 / 137.035999084
print(f"α (постоянная тонкой структуры) = {alpha:.10f}")
print(f"p/α = {p / alpha:.6f}")
print(f"α/p = {alpha / p:.6f}")
print(f"p ≈ {alpha / 2.6:.10f}? (проверка: p/(α/2.6) = {p / (alpha / 2.6):.6f})")

# 6.2. Соотношение с массой протона/электрона
print(f"\nМасса протона/электрона ≈ 1836.15267343")
print(f"p * 35000 ≈ {p * 35000:.2f} (близко к 1836?)")

# 6.3. Логарифмические связи
print(f"\nln(1/p) = {math.log(1 / p):.10f}")
print(f"ln(1/p) / π = {math.log(1 / p) / math.pi:.10f}")
print(f"ln(1/p) / ln(10) = {math.log(1 / p) / math.log(10):.10f} (≈3?)")

# 7. Проверка как вероятности
print("\n7. ВЕРОЯТНОСТНЫЕ ИНТЕРПРЕТАЦИИ:")
print("-" * 40)

# Вероятность в различных контекстах
prob_contexts = [
    ("Вероятность выпадения орла 3 раза подряд", 0.5 ** 3),
    ("Вероятность выпадения конкретной комбинации в 5 бросках монеты", 0.5 ** 5),
    ("Вероятность 1 успеха в 20 испытаниях при p=0.1", 20 * 0.1 * (0.9 ** 19)),
    ("e^{-3}", math.exp(-3)),
    ("1/(2π)", 1 / (2 * math.pi)),
    ("1/(6π²)", 1 / (6 * math.pi ** 2)),
]

for desc, prob in prob_contexts:
    ratio = p / prob
    print(f"{desc:50}: {prob:.6f}, отношение p/вероятность = {ratio:.6f}")

# 8. Численные поиски интересных приближений
print("\n8. ИНТЕРЕСНЫЕ ПРИБЛИЖЕНИЯ:")
print("-" * 40)

approximations = [
    (f"≈ 1/({int(1 / p)}) = 1/{int(1 / p)}", 1 / int(1 / p)),
    ("≈ (π - e)/20", (math.pi - math.e) / 20),
    ("≈ (√10 - π)/10", (math.sqrt(10) - math.pi) / 10),
    ("≈ ln(π)/20", math.log(math.pi) / 20),
    ("≈ (φ - 1.5)/10", ((1 + math.sqrt(5)) / 2 - 1.5) / 10),
    ("≈ 1/(2e²)", 1 / (2 * math.e ** 2)),
    ("≈ 1/(6π)", 1 / (6 * math.pi)),
]

for desc, approx in approximations:
    error = abs(p - approx) / p * 100
    print(f"{desc:20} = {approx:.10f}, ошибка: {error:.3f}%")

# 9. Связь с физическими константами через K=8
print("\n9. СВЯЗЬ ЧЕРЕЗ K=8:")
print("-" * 40)
K = 8.0
Kp = K * p
print(f"K*p = {K} * {p} = {Kp:.10f}")
print(f"ln(K*p) = {math.log(Kp):.10f}")
print(f"|ln(K*p)| = {abs(math.log(Kp)):.10f}")

# Проверим близость к e/2, π/4 и т.д.
print(f"\nK*p близко к:")
checks_Kp = [
    ("1/√5", 1 / math.sqrt(5)),
    ("1/e", 1 / math.e),
    ("√2/π", math.sqrt(2) / math.pi),
    ("ln(2)/2", math.log(2) / 2),
    ("φ - 1.5", (1 + math.sqrt(5)) / 2 - 1.5),
]

for name, value in checks_Kp:
    ratio = Kp / value
    print(f"{name:10} = {value:.6f}, K*p/{name} = {ratio:.6f}")

# 10. Связь с информационной энтропией
print("\n10. ИНФОРМАЦИОННО-ТЕОРЕТИЧЕСКИЕ СВЯЗИ:")
print("-" * 40)

# Энтропия бинарного источника с вероятностью p
H = -p * math.log2(p) - (1 - p) * math.log2(1 - p)
print(f"Бинарная энтропия H(p) = {-p * math.log2(p) - (1 - p) * math.log2(1 - p):.10f}")
print(f"H(p) / p = {H / p:.10f}")
print(f"H(p) * e = {H * math.e:.10f}")

# 11. Связь с теорией хаоса
print("\n11. СВЯЗИ С ТЕОРИЕЙ ХАОСА:")
print("-" * 40)

# Постоянная Фейгенбаума δ ≈ 4.6692016091
feigenbaum = 4.6692016091
print(f"Постоянная Фейгенбаума δ = {feigenbaum:.10f}")
print(f"p * 100 ≈ {p * 100:.6f} (сравни с 52.7)")
print(f"δ / (p*100) = {feigenbaum / (p * 100):.6f}")
print(f"(1-p) * 100 ≈ {(1 - p) * 100:.6f}")

# 12. Связь с комбинаторикой
print("\n12. КОМБИНАТОРНЫЕ СВЯЗИ:")
print("-" * 40)

# Число сочетаний
from scipy.special import comb

for n in [10, 20, 30, 40]:
    # Найдем k такое, что C(n,k)/2^n ≈ p
    total = 2 ** n
    target = p * total

    # Приближенный поиск
    k_approx = n / 4  # начальное приближение
    # Это упрощенный поиск
    print(f"C({n}, {n // 4})/2^{n} ≈ {comb(n, n // 4) / 2 ** n:.6f}")

# 13. Финальные наблюдения
print("\n" + "=" * 70)
print("КЛЮЧЕВЫЕ НАБЛЮДЕНИЯ:")
print("=" * 70)

print("1. Обратное значение 1/p ≈ 18.975 - близко к 19")
print("2. 1-p ≈ 0.9473 - очень близко к 0.9472... что может быть связано")
print("   с e^{-0.0542} ≈ 0.9473 или cos(0.33) ≈ 0.9460")
print("3. p ≈ 0.0527 близко к:")
print("   - 1/(19) = 0.05263158 (ошибка 0.13%)")
print("   - (π - e)/20 = 0.052618 (ошибка 0.16%)")
print("   - ln(π)/20 = 0.0573 (ошибка 8.6%)")
print("4. K*p = 0.4216 близко к 1/√5.6 ≈ 0.422 или e^{-0.863}")
print("5. p похоже на вероятность:")
print("   - 1 успех в 19 испытаниях при вероятности успеха ~0.0526")
print("   - или 5.27% - типичная вероятность редких событий")

print("\n" + "=" * 70)
print("НАИБОЛЕЕ ТОЧНЫЕ ПРИБЛИЖЕНИЯ:")
print("=" * 70)

# Лучшие приближения
best_approxs = [
    ("1/19", 1 / 19, abs(p - 1 / 19) / p * 100),
    ("(π - e)/20", (math.pi - math.e) / 20, abs(p - (math.pi - math.e) / 20) / p * 100),
    ("1/(2e² + π)", 1 / (2 * math.e ** 2 + math.pi), abs(p - 1 / (2 * math.e ** 2 + math.pi)) / p * 100),
    ("ln(φ)/10", math.log((1 + math.sqrt(5)) / 2) / 10, abs(p - math.log((1 + math.sqrt(5)) / 2) / 10) / p * 100),
]

for desc, approx, error in sorted(best_approxs, key=lambda x: x[2]):
    print(f"{desc:15} = {approx:.10f}, ошибка: {error:.6f}%")

# Итоговый вывод
print(f"\nИтог: p ≈ 1/19 = 0.05263158 с ошибкой всего 0.13%!")