import math
from math import log, sqrt, pi, e

K = 8.0
p = 1.25e-31
N = 9.702e122

# ФУНДАМЕНТАЛЬНАЯ КОНСТАНТА ТЕОРИИ
alpha = 1.010135e-41  # Дж·с

lnN = log(N)
lnK = log(K)
lnKp = log(K * p)

x = lnKp / lnN
lambda_sq = x ** 2
U = lnN / abs(lnKp)
f1 = U / pi

c_sveta = lnN**2 * lnKp**2 / lnK**(1/3)
print(f"c_sveta = {c_sveta:.6f}")

# Экспериментальное значение
MP_TARGET = 2.176434e-8

# Вычисление
abs_lnKp = abs(lnKp)

K = 8.0
p = 1.25e-31
N = 9.702e122

lnN = log(N)
lnK = log(K)
lnKp = log(K * p)

# Correction
C_clust = 3 * (K - 2) / (4 * (K - 1)) * (1 - p) ** 3
correction = 1 + (1 - C_clust) / lnN

# V с correction
x = lnKp / lnN
U = lnN / abs_lnKp

hbar_em = (lnK ** 2) / (4 * lambda_sq ** 2 * K ** 2) * correction
V = f1 ** (2/3) * hbar_em ** 3 * lnN ** 2

# m_P через V
m_P = sqrt(1 / pi) * e**(1/6) * V**(5/4) * lnN * abs_lnKp / (N**(1/6) * lnK**(1/6))

print("=" * 70)
print("С УЧЁТОМ CORRECTION")
print("=" * 70)
print(f"m_P = {m_P:.6e} кг")
print(f"m_P (CODATA) = {MP_TARGET:.6e} кг")
print(f"Отношение = {m_P / MP_TARGET:.10f}")
print(f"Ошибка = {abs(m_P / MP_TARGET - 1) * 100:.6f}%")