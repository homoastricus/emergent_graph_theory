import math


def exact_bohr_radius_no_addition(K, p, N):
    """
    100% точная формула радиуса Бора через K, p, N
    Без операции сложения, используем ln(6*N)
    Возвращает значение в метрах
    """

    # Все вычисления с максимальной точностью
    ln_K = math.log(K)
    ln_N = math.log(N)
    ln_6N = math.log(6 * N)  # Вместо ln(6) + ln(N)

    Kp = K * p
    abs_ln_Kp = abs(math.log(Kp))
    sqrt_Kp = math.sqrt(Kp)

    # Числитель
    numerator = (ln_K ** 3) * p * ln_6N * (abs_ln_Kp ** 2)

    # Знаменатель
    denominator = 2304 * (math.pi ** 3) * (K ** 3) * sqrt_Kp * (ln_N ** 2)

    # Это уже значение в метрах!
    return numerator / denominator


# Ваши параметры
K = 8.0
p = 0.0527
N = 9.7e122

# Вычисление
r_bohr_meters = exact_bohr_radius_no_addition(K, p, N)
print(f"Радиус Бора (метры): {r_bohr_meters:.10e}")
print(f"Классический радиус Бора: 5.29177210903e-11 м")
print(f"Отношение: {r_bohr_meters / 5.29177210903e-11:.6f}")