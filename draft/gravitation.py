import math

K = 8.0
p = 0.05270179
N = 9.702e+122

lnK = math.log(K)  # ≈ 2.07944
lnKp = math.log(K*p)  # ≈ -0.863664
lnN = math.log(N)  # ≈ 283.187
lambda_param = (lnKp/lnN)**2  # ≈ 9.299e-6

# По правильной формуле:
G_correct = (lnK**8 * p**2) / (1024 * math.pi**2 * lambda_param**8 * K**6 * N**(1/3))

print(f"lnK = {lnK:.6f}")
print(f"lnK^8 = {lnK**8:.3e}")
print(f"p^2 = {p**2:.6e}")
print(f"λ = {lambda_param:.6e}")
print(f"λ^8 = {lambda_param**8:.3e}")
print(f"K^6 = {K**6:.0f}")
print(f"N^(1/3) = {N**(1/3):.3e}")
print(f"1024π² = {1024*math.pi**2:.3e}")

print(f"\nG = {G_correct:.6e} м³/кг·с²")
print(f"Классическое G = 6.674e-11 м³/кг·с²")
print(f"Отношение: {G_correct/6.674e-11:.6f}")