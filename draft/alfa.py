import math

K = 6.0
pi = math.pi
lnK = math.log(K)
N = 4.198e121
lnN = math.log(N)

p = 1.0 / (K * N**(1/3))
lnKp = math.log(K*p)
Kp = K * p

alpha_eti = 2 * lnK**2 / (pi * lnN)

# Ваша формула
val2 =  ((K * lnKp/lnN) / (K - lnN))

print(f"α (ЕТИ) = {alpha_eti:.10f}")
print(f"Ваша формула = {val2:.10f}")
print(f"Ваша формула2 = {val2:.10f}")
print(f"Отношение = {val2/alpha_eti:.6f}")
print(f"Ошибка = {abs(val2/alpha_eti - 1)*100:.6f}%")