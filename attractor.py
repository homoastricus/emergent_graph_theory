import math


def calculate_N_from_pK(p, K=8.0):
    # Число Эйлера
    e_const = math.e

    # K*p
    Kp = K * p

    # |ln(Kp)|
    abs_ln_Kp = abs(math.log(Kp))

    # Вычисляем ln(N)
    ln_N = (e_const ** 2 * abs_ln_Kp) / (p ** 2 * K)

    # Вычисляем N
    N = math.exp(ln_N)

    return N, ln_N


def calculate_N_from_pK_detailed(p, K=8.0):
    """Детальный расчет с промежуточными значениями"""
    print(f"Расчет N из тождества p*sqrt(K*U) = e")
    print(f"при p = {p:.6e}, K = {K}")

    e_const = math.e
    Kp = K * p
    ln_Kp = math.log(Kp)
    abs_ln_Kp = abs(ln_Kp)

    print(f"1. e = {e_const:.6f}")
    print(f"2. K*p = {K} * {p:.6e} = {Kp:.6f}")
    print(f"3. ln(K*p) = {ln_Kp:.6f}")
    print(f"4. |ln(K*p)| = {abs_ln_Kp:.6f}")
    print(f"5. e^2 = {e_const ** 2:.6f}")
    print(f"6. p^2 = {p ** 2:.6e}")
    print(f"7. p^2 * K = {p ** 2 * K:.6e}")

    # Вычисление ln(N)
    numerator = e_const ** 2 * abs_ln_Kp
    denominator = p ** 2 * K
    ln_N = numerator / denominator

    print(f"\n8. Числитель: e^2 * |ln(Kp)| = {numerator:.6f}")
    print(f"9. Знаменатель: p^2 * K = {denominator:.6e}")
    print(f"10. ln(N) = {numerator:.6f} / {denominator:.6e} = {ln_N:.6f}")

    # Вычисление N
    N = math.exp(ln_N)
    log10_N = ln_N / math.log(10)  # перевод в log10

    print(f"\n11. N = exp({ln_N:.6f}) = {N:.6e}")
    print(f"12. log10(N) = {log10_N:.6f}")

    # Проверка тождества обратно
    U = ln_N / abs_ln_Kp
    left_side = p * math.sqrt(K * U)
    error = abs(left_side - e_const) / e_const * 100

    print(f"\nПРОВЕРКА:")
    print(f"U = lnN / |ln(Kp)| = {ln_N:.6f} / {abs_ln_Kp:.6f} = {U:.6f}")
    print(f"p*sqrt(K*U) = {p:.6e} * sqrt({K} * {U:.6f}) = {left_side:.6f}")
    print(f"e = {e_const:.6f}")
    print(f"Относительная ошибка: {error:.6e}%")

    return N, ln_N


if __name__ == "__main__":
    # Ваши параметры
    p = 5.270179e-02
    K = 8.0

    print("ПРОВЕРКА ПАРАМЕТРОВ:")

    N_calculated, ln_N_calculated = calculate_N_from_pK_detailed(p, K)

    # Сравнение с вашим N
    N_your = 9.702e+122
    print("СРАВНЕНИЕ:")
    print(f" N:      {N_your:.6e}")
    print(f"Расчетное N: {N_calculated:.6e}")

    ratio = N_calculated / N_your
    log_ratio = math.log(ratio)

    print(f"\nОтношение (расчет/теория): {ratio:.6f}")
    print(f"ln(отношения): {log_ratio:.6f}")
    print(f"Отличие в порядке: {math.log10(ratio):.6f}")

    # 4. Анализ для разных p
    print("АНАЛИЗ ЗАВИСИМОСТИ N ОТ p:")
    print(f"{'p':<12} {'N':<20} {'log10(N)':<12} {'p*sqrt(K*U)-e':<15}")
    print("-" * 60)

    p_values = [0.05]

    for p_val in p_values:
        N_val, ln_N = calculate_N_from_pK(p_val, K)
        Kp = K * p_val
        abs_ln_Kp = abs(math.log(Kp))
        U = ln_N / abs_ln_Kp
        left_side = p_val * math.sqrt(K * U)
        diff = left_side - math.e

        print(f"{p_val:<12.6f} {N_val:<20.6e} {ln_N / math.log(10):<12.3f} {diff:<15.6e}")


## 5. Обратная задача: найти p для заданного N
def find_p_for_given_N(N_target, K=8.0, tolerance=1e-10, max_iter=1000):
    """
    Находит p такое, что N(p) = N_target
    Используем метод Ньютона
    """

    def f(p):
        """Функция: ln(N(p)) - ln(N_target)"""
        Kp = K * p
        if Kp <= 0:
            return 1e100
        abs_ln_Kp = abs(math.log(Kp))
        ln_N_pred = (math.e ** 2 * abs_ln_Kp) / (p ** 2 * K)
        return ln_N_pred - math.log(N_target)

    def df(p):
        """Производная f по p"""
        Kp = K * p
        if Kp <= 0:
            return 1e100
        abs_ln_Kp = abs(math.log(Kp))
        term1 = -2 * math.e ** 2 * abs_ln_Kp / (p ** 3 * K)
        term2 = math.e ** 2 / (p ** 2 * K)  # производная от |ln(Kp)|
        return term1 + term2

    # Начальное приближение
    p_guess = 0.05
    p_current = p_guess

    for i in range(max_iter):
        f_val = f(p_current)
        df_val = df(p_current)

        if abs(f_val) < tolerance:
            break

        if df_val == 0:
            break

        p_new = p_current - f_val / df_val
        # Ограничиваем p разумными пределами
        if p_new <= 0 or p_new >= 1:
            p_new = max(1e-10, min(0.999, p_new))
        p_current = p_new
    return p_current


# Проверяем найденное p
N_check, _ = calculate_N_from_pK(0.05, K=8)
print(f"Проверка: N(p_found) = {N_check:.3e}")
