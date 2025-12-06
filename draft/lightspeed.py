import math
import numpy as np


def calculate_original(K, p, N):
    """Оригинальный расчет из кода"""
    lambda_param = (math.log(K * p) / math.log(N)) ** 2

    # hbar_em
    hbar_em = (np.log(K) ** 2) / (4 * lambda_param ** 2 * K ** 2)

    # R_universe
    R_universe = 2 * math.pi / (np.sqrt(K * p) * lambda_param) * N ** (1 / 6)

    # Оригинальная формула для c
    c_original = math.pi * (1 / np.sqrt(K * p) * R_universe / hbar_em) / lambda_param ** 2 * N ** (-1 / 6)

    return c_original, lambda_param, hbar_em, R_universe


def calculate_simplified(K, p, N):
    """Упрощенная формула"""
    lambda_param = (math.log(K * p) / math.log(N)) ** 2

    # Упрощенная формула: c = 8π²K / [p(ln K)²λ]
    c_simple = (8 * math.pi ** 2 * K) / (p * (math.log(K) ** 2) * lambda_param)

    return c_simple, lambda_param


def calculate_alternative(K, p, N):
    """Альтернативная форма без явного λ"""
    c_alt = (8 * math.pi ** 2 * K * (math.log(N) ** 2)) / \
            (p * (math.log(K) ** 2) * (math.log(K * p) ** 2))
    return c_alt


# Параметры
K = 8.0
p = 5.270179e-02  # 0.05270179
N = 9.702e+122

print("=" * 60)
print("ПРОВЕРКА ФОРМУЛЫ ДЛЯ СКОРОСТИ СВЕТА")
print("=" * 60)
print(f"Параметры: K = {K}, p = {p:.6e}, N = {N:.3e}")
print()

# Оригинальный расчет
c_orig, lambda_param, hbar_em, R_univ = calculate_original(K, p, N)
print("ОРИГИНАЛЬНАЯ ФОРМУЛА:")
print(f"  λ = {lambda_param:.6e}")
print(f"  ħ_em = {hbar_em:.6e}")
print(f"  R_universe = {R_univ:.6e} м")
print(f"  c = {c_orig:.6e} м/с")
print()

# Упрощенная формула
c_simple, lambda_simple = calculate_simplified(K, p, N)
print("УПРОЩЕННАЯ ФОРМУЛА (c = 8π²K/[p(ln K)²λ]):")
print(f"  λ = {lambda_simple:.6e} (должно совпадать с выше)")
print(f"  c = {c_simple:.6e} м/с")
print()

# Альтернативная форма
c_alt = calculate_alternative(K, p, N)
print("АЛЬТЕРНАТИВНАЯ ФОРМУЛА (без явного λ):")
print(f"  c = {c_alt:.6e} м/с")
print()

# Сравнение
print("=" * 60)
print("СРАВНЕНИЕ:")
print("=" * 60)
print(f"Оригинальная формула:  {c_orig:.10e} м/с")
print(f"Упрощенная формула:   {c_simple:.10e} м/с")
print(f"Альтернативная форма: {c_alt:.10e} м/с")
print()

# Отношения
ratio_simple = c_simple / c_orig
ratio_alt = c_alt / c_orig

print(f"Отношение упрощ/ориг: {ratio_simple:.12f}")
print(f"Отношение альт/ориг:  {ratio_alt:.12f}")
print()

# Проверка экспериментального значения
c_exp = 2.99792458e8
print(f"Экспериментальное значение: {c_exp:.6e} м/с")
print(f"Отношение ориг/эксп: {c_orig / c_exp:.6f}")
print(f"Отношение прост/эксп: {c_simple / c_exp:.6f}")
print()

# Проверка шаг за шагом
print("=" * 60)
print("ДЕТАЛЬНАЯ ПРОВЕРКА ШАГ ЗА ШАГОМ:")
print("=" * 60)

# Шаг 1: Вычислим промежуточные величины
lnK = math.log(K)
lnKp = math.log(K * p)
lnN = math.log(N)

print(f"1. ln(K) = {lnK:.6f}")
print(f"2. ln(Kp) = {lnKp:.6f}")
print(f"3. ln(N) = {lnN:.6f}")
print(f"4. λ = (ln(Kp)/lnN)² = ({lnKp:.6f}/{lnN:.6f})² = {lambda_param:.6e}")

# Шаг 2: Проверим коэффициент
coeff = (8 * math.pi ** 2 * K) / (p * (lnK ** 2))
print(f"\n5. Коэффициент 8π²K/[p(ln K)²] = {coeff:.6e}")

# Шаг 3: Проверим произведение
print(f"6. coeff/λ = {coeff}/{lambda_param:.6e} = {coeff / lambda_param:.6e}")
print(f"7. Что должно быть равно c_simple = {c_simple:.6e}")

# Проверим точное равенство
tolerance = 1e-10
if abs(ratio_simple - 1.0) < tolerance and abs(ratio_alt - 1.0) < tolerance:
    print("\n✅ ВСЕ ФОРМУЛЫ ДАЮТ ИДЕНТИЧНЫЙ РЕЗУЛЬТАТ!")
else:
    print(f"\n⚠️  ЕСТЬ РАСХОЖДЕНИЕ! Проверим точнее...")

    # Детальная проверка
    print(f"\nРазница упрощ-ориг: {abs(c_simple - c_orig):.2e}")
    print(f"Относительная ошибка: {abs(c_simple / c_orig - 1) * 100:.2e}%")