import math

K = 6.0
pi = math.pi
lnK = math.log(K)
N = 4.1790e121
lnN = math.log(N)

# Эмерджентные формулы
hbar_eti = lnN**3 / (K * N**(1/3))
c_eti = pi * lnN**4 / (K**2 * lnK)
mP_eti_main = K / (4 * pi * lnN**3)

# G через определение
G_from_mP = hbar_eti * c_eti / mP_eti_main**2

# Прямая эмерджентная формула G
G_direct = pi**7 * lnN**15 / (K**11 * lnK**5 * N**(1/3))

G_emer = 16 * pi**3 * lnN**13 / (K**5 * lnK * N**(1/3))

# CODATA
G_codata = 6.67430e-11

print("=" * 60)
print("ВАЛИДАЦИЯ G ЧЕРЕЗ m_P")
print("=" * 60)
print(f"G (из m_P)   = {G_from_mP:.10e}")
print(f"G (прямая)   = {G_direct:.10e}")
print(f"G (CODATA)   = {G_codata:.10e}")
print(f"Ошибка (m_P) = {abs(G_from_mP - G_codata)/G_codata*100:.4f}%")
print(f"Ошибка (прям) = {abs(G_direct - G_codata)/G_codata*100:.4f}%")
print(f"Ошибка (G_emer) = {abs(G_emer - G_codata)/G_codata*100:.4f}%")
