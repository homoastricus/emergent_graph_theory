import numpy as np
import math


def test_alpha_formula_with_6():
    """Тестирование формулы α = ln(K)/ln(N×6)"""

    K = 8.0
    N = 0.95e123
    α_exact = 1 / 137.035999084

    print("=== АНАЛИЗ ФОРМУЛЫ α = ln(K)/ln(N×6) ===")
    print(f"α (эксперимент) = {α_exact:.15f}")
    print()

    # Ваша формула с 6
    α_calc = math.log(K) / math.log(N * 6)

    print(f"Ваша формула:")
    print(f"α = ln({K}) / ln({N:.2e} × 6)")
    print(f"  = ln({K}) / ln({N * 6:.2e})")
    print(f"  = {math.log(K):.10f} / {math.log(N * 6):.10f}")
    print(f"  = {α_calc:.15f}")
    print()

    error = abs(α_calc / α_exact - 1) * 100
    print(f"Ошибка: {error:.10f}%")
    print(f"Совпадение до {-math.log10(error) + 2:.1f} знаков после запятой")
    print()

    # Проверим чувствительность к 6
    print("ЧУВСТВИТЕЛЬНОСТЬ К КОЭФФИЦИЕНТУ:")
    print(f"{'Коэффициент':<15} | {'α':<20} | {'Ошибка %':<15}")
    print("-" * 60)

    for coeff in [5.9, 5.95, 5.98, 5.99, 6.0, 6.01, 6.02, 6.05, 6.1]:
        α_test = math.log(K) / math.log(N * coeff)
        error_test = abs(α_test / α_exact - 1) * 100
        print(f"{coeff:<15.3f} | {α_test:<20.15f} | {error_test:<15.10f}")

    return α_calc, error


def why_6_works():
    """Почему именно 6 работает?"""

    print("\n=== ПОЧЕМУ ИМЕННО 6? ===")

    K = 8.0
    N = 9.5e122
    α_exact = 1 / 137.035999084

    # Из формулы: α = ln(K)/ln(6N)
    # => ln(6N) = ln(K)/α
    # => 6N = exp(ln(K)/α)
    # => 6 = exp(ln(K)/α) / N

    # Вычислим "идеальный" коэффициент
    ideal_coeff = math.exp(math.log(K) / α_exact) / N

    print(f"Из формулы α = ln(K)/ln(C·N):")
    print(f"ln(C·N) = ln(K)/α")
    print(f"C·N = exp(ln(K)/α)")
    print(f"C = exp(ln(K)/α) / N")
    print()
    print(f"ln(K) = ln({K}) = {math.log(K):.10f}")
    print(f"ln(K)/α = {math.log(K):.10f} / {α_exact:.10f}")
    print(f"        = {math.log(K) / α_exact:.10f}")
    print(f"exp(ln(K)/α) = {math.exp(math.log(K) / α_exact):.10f}")
    print(f"N = {N:.10e}")
    print(f"Идеальный C = {math.exp(math.log(K) / α_exact):.10f} / {N:.10e}")
    print(f"            = {ideal_coeff:.15f}")
    print()
    print(f"Ваш коэффициент: 6")
    print(f"Идеальный коэффициент: {ideal_coeff:.15f}")
    print(f"Отношение: {6 / ideal_coeff:.10f}")

    return ideal_coeff


def physical_meaning_of_6():
    """Физический смысл числа 6"""

    print("\n=== ФИЗИЧЕСКИЙ СМЫСЛ ЧИСЛА 6 ===")

    print("1. ПРОСТРАНСТВЕННЫЕ ИЗМЕРЕНИЯ:")
    print("   6 = 2 × 3")
    print("   где 3 - пространственные измерения")
    print("   2 - возможно, спин или что-то связанное")
    print()

    print("2. ГРАФОВАЯ ИНТЕРПРЕТАЦИЯ:")
    print("   В графе малого мира:")
    print("   - K = 8 (кубическая решётка в 3D)")
    print("   - Каждый узел видит 8 соседей")
    print("   - Но эффективно связано 6 направлений?")
    print()

    print("3. СВЯЗЬ С π:")
    print(f"   2π ≈ {2 * math.pi:.6f}")
    print(f"   6/π ≈ {6 / math.pi:.6f}")
    print(f"   π² ≈ {math.pi ** 2:.6f}")
    print()

    print("4. КОМБИНАЦИИ ИЗ ВАШЕЙ МОДЕЛИ:")
    # Проверим другие константы вашей модели
    K = 8.0
    p = 0.0527
    N_val = 9.5e122

    λ = (math.log(K * p) / math.log(N_val)) ** 2
    U = math.log(N_val) / abs(math.log(K * p))

    print(f"   λ = {λ:.6e}")
    print(f"   U = {U:.6f}")
    print(f"   p = {p}")
    print(f"   e = {math.exp(1):.6f}")
    print()
    print(f"   Возможные связи:")
    print(f"   6 = 3! (факториал 3)")
    print(f"   6 = 2 × 3 (размерность × что-то)")
    print(f"   6 = π × e / 1.423...")
    print(f"   6 ≈ 2π - 0.283...")

    return 6


def alternative_formulations():
    """Альтернативные формулировки той же формулы"""

    print("\n=== АЛЬТЕРНАТИВНЫЕ ФОРМУЛИРОВКИ ===")

    K = 8.0
    N = 9.5e122
    α_exact = 1 / 137.035999084

    print("Ваша формула: α = ln(K)/ln(6N)")
    print()

    # Формулировка 1: через эффективный N
    N_eff = 6 * N
    print(f"1. Через эффективный N_eff = 6N = {N_eff:.2e}")
    print(f"   α = ln(K)/ln(N_eff)")
    print()

    # Формулировка 2: как поправка к логарифму
    print(f"2. Как поправка к ln(N):")
    print(f"   ln(6N) = ln(N) + ln(6)")
    print(f"   ln(6) = {math.log(6):.10f}")
    print(f"   ln(N) = {math.log(N):.10f}")
    print(f"   ln(6)/ln(N) = {math.log(6) / math.log(N):.10e} (очень мало)")
    print()

    # Формулировка 3: через относительную поправку
    print(f"3. Относительная поправка:")
    α_simple = math.log(K) / math.log(N)
    α_corrected = math.log(K) / math.log(6 * N)

    print(f"   α(просто) = ln(K)/ln(N) = {α_simple:.10f}")
    print(f"   α(с 6) = ln(K)/ln(6N) = {α_corrected:.10f}")
    print(f"   Отношение: {α_corrected / α_simple:.10f}")
    print(f"   1/этого отношения: {α_simple / α_corrected:.10f}")
    print()

    # Формулировка 4: что если 6 = 2π/что-то?
    test_value = 2 * math.pi / (α_corrected / α_simple)
    print(f"4. Связь с 2π:")
    print(f"   2π = {2 * math.pi:.10f}")
    print(f"   2π / (отношение) = {test_value:.10f}")
    print(f"   Близко к 6? {abs(test_value - 6):.10f}")


def find_optimal_coefficient():
    """Нахождение оптимального коэффициента"""

    print("\n=== ОПТИМАЛЬНЫЙ КОЭФФИЦИЕНТ ===")

    K = 8.0
    N = 9.5e122
    α_exact = 1 / 137.035999084

    # Решим точно: α = ln(K)/ln(C·N)
    # => ln(C·N) = ln(K)/α
    # => C·N = exp(ln(K)/α)
    # => C = exp(ln(K)/α)/N

    C_optimal = math.exp(math.log(K) / α_exact) / N

    print(f"Точное уравнение:")
    print(f"α = ln(K)/ln(C·N)")
    print(f"=> C = exp(ln(K)/α)/N")
    print()
    print(f"Подставляем:")
    print(f"ln(K) = {math.log(K):.15f}")
    print(f"α = {α_exact:.15f}")
    print(f"ln(K)/α = {math.log(K) / α_exact:.15f}")
    print(f"exp(ln(K)/α) = {math.exp(math.log(K) / α_exact):.15f}")
    print(f"N = {N:.15e}")
    print(f"C_optimal = {C_optimal:.15f}")
    print()
    print(f"Ваш коэффициент: 6")
    print(f"Оптимальный: {C_optimal:.15f}")
    print(f"Разница: {abs(6 - C_optimal):.15e}")
    print(f"Относительная ошибка: {abs(6 / C_optimal - 1) * 100:.10f}%")

    # Проверим, что получается с оптимальным C
    α_optimal = math.log(K) / math.log(C_optimal * N)
    print(f"\nС оптимальным C:")
    print(f"α = {α_optimal:.15f}")
    print(f"Точное α = {α_exact:.15f}")
    print(f"Ошибка: {abs(α_optimal / α_exact - 1) * 100:.15e}%")

    return C_optimal


# Запуск анализа
if __name__ == "__main__":
    print("ГЛУБОКИЙ АНАЛИЗ ФОРМУЛЫ α = ln(K)/ln(6N)")
    print("=" * 70)

    # 1. Тестирование формулы
    α_calc, error = test_alpha_formula_with_6()

    # 2. Почему 6?
    ideal_coeff = why_6_works()

    # 3. Физический смысл 6
    physical_meaning_of_6()

    # 4. Альтернативные формулировки
    alternative_formulations()

    # 5. Оптимальный коэффициент
    C_optimal = find_optimal_coefficient()

    # 6. Вывод
    print("\n" + "=" * 70)
    print("ВЫВОД:")
    print("=" * 70)

    print(f"""
    ВАША ФОРМУЛА: α = ln(K)/ln(6N)

    ТОЧНОСТЬ: {error:.10f}% ошибки

    ФИЗИЧЕСКИЙ СМЫСЛ ЧИСЛА 6:

    1. ВАША ФОРМУЛА ФАКТИЧЕСКИ: α = ln(K)/ln(N_eff)
       где N_eff = 6N - эффективное количество степеней свободы

    2. 6 = 2 × 3
       3 - пространственные измерения
       2 - возможно, связано с:
           - Спином частиц (фермионы имеют полуцелый спин)
           - Двумя направлениями времени? (F-теория)
           - Чётностью?

    3. ИЛИ: 6 ≈ 2π - 0.2832...
       2π естественно появляется в квантовой механике

    4. СВЯЗЬ С ГРАФОМ:
       В кубической решётке (K=8) каждый узел имеет 6 граней
       Но соединён с 8 соседями
       Возможно, эффективное число связано с 6

    МАТЕМАТИЧЕСКИ:

    Идеальный коэффициент: C = {C_optimal:.15f}
    Ваш коэффициент: 6
    Разница: {abs(6 - C_optimal):.2e}

    То есть 6 - это ПРИБЛИЖЁННОЕ значение идеального коэффициента,
    но дающее невероятную точность благодаря логарифмической зависимости!

    КЛЮЧЕВОЙ МОМЕНТ:

    ln(6N) = ln(N) + ln(6)
           = ln(N) × [1 + ln(6)/ln(N)]

    Так как ln(6)/ln(N) ≈ {math.log(6) / math.log(9.5e122):.2e} ОЧЕНЬ мало,
    то небольшие изменения в коэффициенте дают большие изменения в α!

    ЭТО ОБЪЯСНЯЕТ, почему формула так чувствительна к точному значению 6!
    """)