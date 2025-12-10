import math


def find_exact_p_for_pi():
    """Находит p, при котором ln(8+p) + 1/(1-p) = π"""
    target = math.pi

    def f(p):
        return math.log(8 + p) + 1 / (1 - p) - target

    # Метод бисекции
    a, b = 0.05, 0.06
    for _ in range(50):
        mid = (a + b) / 2
        if f(mid) * f(a) < 0:
            b = mid
        else:
            a = mid

    p_exact = (a + b) / 2
    return p_exact


p_exact_pi = find_exact_p_for_pi()
print(f"Точное p для π: {p_exact_pi:.10f}")