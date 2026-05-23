import math
from scipy.optimize import fsolve

K = 8.0

def equation(p):
    """Уравнение: ln(K+p) + 1/(1-p) - π = 0"""
    if p <= 0 or p >= 1:
        return 1e10  # штраф за выход из области
    return math.log(K + p) + 1.0 / (1.0 - p) - math.pi

# Начальное приближение (известное значение)
p0 = 0.05270179

# Поиск корня
p_opt = fsolve(equation, p0, xtol=1e-12)[0]

# Проверка
val = math.log(K + p_opt) + 1.0 / (1.0 - p_opt)
error = abs(val - math.pi) / math.pi

print("=" * 60)
print("ПОИСК p, ПРИ КОТОРОМ ln(K+p) + 1/(1-p) = π")
print("=" * 60)
print(f"K = {K}")
print(f"\nИсходное p (из модели): {p0:.10f}")
val0 = math.log(K + p0) + 1.0 / (1.0 - p0)
print(f"  Значение: {val0:.10f}")
print(f"  Ошибка относительно π: {abs(val0 - math.pi)/math.pi * 100:.8f}%")

print(f"\nОптимальное p: {p_opt:.10f}")
print(f"  Значение: {val:.10f}")
print(f"  π = {math.pi:.10f}")
print(f"  Абсолютная ошибка: {abs(val - math.pi):.2e}")
print(f"  Относительная ошибка: {error:.2e}")

print(f"\nСравнение:")
print(f"  p_opt - p0 = {p_opt - p0:.10f}")
print(f"  Относительное изменение: {abs(p_opt - p0)/p0 * 100:.6f}%")