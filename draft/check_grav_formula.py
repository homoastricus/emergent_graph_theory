import math
from math import log, sqrt, pi

K = 8.0
p = 1.25e-31
N = 9.702e122

# Базовые величины
lnN = log(N)
lnK = log(K)
lnKp = log(K * p)

x = lnKp / lnN
lambda_sq = x ** 2
U = lnN / abs(lnKp)

f1 = U / pi
f5 = K / lnK
S_glob = lnN

C = 3 * (K - 2) / (4 * (K - 1)) * (1 - p) ** 3
correction = 1 + (1 - C) / lnN
hbar_em = (lnK ** 2) / (4 * lambda_sq ** 2 * K ** 2) * correction

# Формула G
G_calc = (lambda_sq**4 * f1**(16/3) * hbar_em**6 * N**(4/3)) / (f5**(8/3) * S_glob**14)

print("=" * 60)
print("ПРОВЕРКА ФОРМУЛЫ G")
print("=" * 60)
print(f"G_calc = {G_calc:.6e}")
print(f"G_target (CODATA) = 6.67430e-11")
print(f"Отношение = {G_calc / 6.67430e-11:.6e}")