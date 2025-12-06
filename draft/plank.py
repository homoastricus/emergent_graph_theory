import numpy as np

K = 8
p = 0.0527
lambda_param = 1e-5

numerator = 2048 * np.pi**3 * K**5 * lambda_param**5
denominator = p**3 * (np.log(K))**8

M_planck = np.sqrt(numerator / denominator)

print("РАСЧЕТ ПЛАНКОВСКОЙ МАССЫ:")
print(f"Числитель = 2048 × π³ × K⁵ × λ⁵")
print(f"          = 2048 × {np.pi**3:.3f} × {K**5} × {lambda_param**5:.3e}")
print(f"          = {numerator:.3e}")
print()
print(f"Знаменатель = p³ × (ln K)⁸")
print(f"            = {p**3:.3e} × {np.log(K)**8:.3e}")
print(f"            = {denominator:.3e}")
print()
print(f"Дробь = {numerator/denominator:.3e}")
print(f"M_planck = √({numerator/denominator:.3e}) = {M_planck:.3e} кг")
print(f"Реальная M_planck = 2.176e-8 кг")
print(f"Отношение: {M_planck/2.176e-8:.3f}")