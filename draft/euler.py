import math

# Заданные параметры
K = 8.00
p = 5.270179e-02
N = 9.702e+122

# Вычисления
lnN = math.log(N)
Kp = K * p
lnKp = math.log(Kp)
U = lnN / abs(lnKp)
left_side = p * math.sqrt(K * U)
right_side = math.e
difference = abs(left_side - right_side)
relative_error = abs(left_side/right_side - 1) * 100

print(f"Результаты:")
print(f"N = {N:.3e}")
print(f"ln(N) = {lnN:.6f}")
print(f"K*p = {Kp:.6f}")
print(f"ln(K*p) = {lnKp:.6f}")
print(f"U = {U:.6f}")
print(f"p*sqrt(K*U) = {left_side:.10f}")
print(f"e = {right_side:.10f}")
print(f"Разность = {difference:.6e}")
print(f"Относительная ошибка = {relative_error:.6f}%")

# Дополнительно: значение λ
lam = (lnKp/lnN)**2
print(f"\nλ = {lam:.6e}")
print(f"√λ = {math.sqrt(lam):.6f}")