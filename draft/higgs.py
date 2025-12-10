import math

K = 8.0
p = 5.270179e-02
N = 9.702e122

lnK = math.log(K)
lnKp = math.log(K * p)
abs_lnKp = abs(lnKp)
lnN = math.log(N)
U = lnN / abs_lnKp

# f1-f6
f1 = U / math.pi
f2 = lnK
f3 = math.sqrt(K * p)
f4 = 1 / p
f5 = K / lnK
f6 = (K + p * K) / K

# Вычисляем m_Z/m_e по формуле
mZ_me_formula = (f1**4 * f2) / (f4**2 * f5)

# Экспериментальное значение
m_e_exp = 9.1093837e-31
m_Z_exp = 1.626e-25
mZ_me_exp = m_Z_exp / m_e_exp

print(f"f1 = {f1:.6f}")
print(f"f2 = {f2:.6f}")
print(f"f4 = {f4:.2f}")
print(f"f5 = {f5:.6f}")
print()
print(f"m_Z/m_e по формуле: {mZ_me_formula:.1f}")
print(f"m_Z/m_e экспериментально: {mZ_me_exp:.1f}")
print(f"Отношение формула/эксперимент: {mZ_me_formula/mZ_me_exp:.3f}")

# Проверим альтернативную запись
print(f"\nПроверка:")
print(f"f1⁴ = {f1**4:.6e}")
print(f"f1⁴ * f2 = {(f1**4 * f2):.6e}")
print(f"f4² = {f4**2:.6e}")
print(f"f4² * f5 = {(f4**2 * f5):.6e}")
print(f"(f1⁴ * f2) / (f4² * f5) = {(f1**4 * f2)/(f4**2 * f5):.6e}")