import math
import sys
# Константы
K = 6

N= 4.196e121 #4.475947352678e+121
N_sqrt = N**(-1/3)
print(math.log(N)/ (N_sqrt))

sys.exit()

# Уравнение, которое должно быть равно 0
def equation(N):
    if N <= 0:
        return 1e10
    p = N ** (-1 / 3) / K
    numerator = K + math.log(p)
    denominator = p - math.log(N)
    return numerator / denominator - 1.0 / math.pi


# Поиск интервала, где функция меняет знак
def find_bracket(start, step=1.5):
    x = start
    fx = equation(x)

    for _ in range(100):
        x_next = x * step
        fx_next = equation(x_next)

        if fx * fx_next < 0:
            return (min(x, x_next), max(x, x_next))

        x = x_next
        fx = fx_next

    raise ValueError("Не удалось найти интервал смены знака")


# Метод половинного деления для уточнения корня
def bisection(a, b, tol=1e-15, max_iter=200):
    fa = equation(a)
    fb = equation(b)

    if fa * fb > 0:
        raise ValueError("На концах интервала одинаковые знаки")

    for i in range(max_iter):
        mid = (a + b) / 2
        fmid = equation(mid)

        if abs(fmid) < tol:
            return mid

        if fa * fmid < 0:
            b = mid
            fb = fmid
        else:
            a = mid
            fa = fmid

    return (a + b) / 2


# Начальное приближение (из аналитической оценки)
N_initial = 3.14e121
print(f"Начальное приближение: N = {N_initial:.2e}")
print(f"Значение уравнения при N_initial: {equation(N_initial):.2e}\n")

# Находим интервал
try:
    low, high = find_bracket(N_initial, step=2.0)
    print(f"Найден интервал: N ∈ [{low:.2e}, {high:.2e}]")
    print(f"f(low) = {equation(low):.2e}")
    print(f"f(high) = {equation(high):.2e}\n")

    # Находим точный корень
    N_solution = bisection(low, high)
    p_solution = N_solution ** (-1 / 3) / K

    print("=" * 70)
    print(f"ТОЧНОЕ РЕШЕНИЕ:")
    print(f"N = {N_solution:.12e}")
    print(f"ln(N) = {math.log(N_solution):.12f}")
    print(f"p = {p_solution:.12e}")
    print(f"K = {K}")

    # Проверка
    lhs = (K + math.log(p_solution)) / (p_solution - math.log(N_solution))
    print(f"\nПРОВЕРКА:")
    print(f"(K+ln p)/(p-ln N) = {lhs:.15f}")
    print(f"1/π                 = {1 / math.pi:.15f}")
    print(f"РАЗНИЦА             = {lhs - 1 / math.pi:.2e} ← должна быть 0")

    # Проверка второго уравнения
    pK = p_solution * K
    N_inv3 = N_solution ** (-1 / 3)
    print(f"\nПРОВЕРКА pK = N^(-1/3):")
    print(f"p·K = {pK:.15e}")
    print(f"N^(-1/3) = {N_inv3:.15e}")
    print(f"РАЗНИЦА = {pK - N_inv3:.2e} ← должна быть 0")

    print("=" * 70)

except Exception as e:
    print(f"Ошибка: {e}")
    print("\nПробуем другой начальный интервал...")

    # Альтернативный поиск
    for test_N in [1e120, 1e121, 1e122, 1e123]:
        f_val = equation(test_N)
        print(f"N = {test_N:.1e}, f(N) = {f_val:.3e}")
        if abs(f_val) < 1e-10:
            print(f"\nНайдено решение: N = {test_N:.1e}")
            break