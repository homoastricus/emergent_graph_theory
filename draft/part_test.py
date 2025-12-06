import math

# === Фундаментальные параметры ===
K, p = 8.0, 0.052702
N = 9.702e+122
U = 327.891
lam = p*(1-2 * math.sqrt(p/K))
print(lam)
# === Структурные функции ===
f1 = U / math.pi        # ≈ 104.371
f2 = math.log(K)        # ≈ 2.079
f3 = math.sqrt(K*p)     # ≈ 0.6493
f4 = 1/p                # ≈ 18.97
f5 = K / math.log(K)    # ≈ 3.847

# === Массы частиц по твоим формулам ===
m_e = 12 * math.sqrt(K*p) * U**4 * N**(-1/3)
m_mu = m_e * 2 * f1
m_tau = m_e * (math.sqrt(K)/(2*p)) * f1 * f4
m_u = m_e * math.sqrt(K/p)
m_d = m_u * f2
m_s = m_e * f1
m_c = m_e * (1/f5) * f1 * f5
m_b = m_e * (K*p) * f1**2
m_t = m_e * ( (f1)/(f2*(K*p)*(f5**2)) ) * f1**2 * f5
m_p = m_e * (f1/(f3*f4*f5)) * (U*K/math.pi)
m_n = m_p * 1.002  # приближенно
m_W = m_e * ((f1*f2*f3)/(f4**2) * f5**2) * f1**2 * f5
m_Z = m_e * ((f1**2*f2)/(f4**2 * f5**2)) * f1**2 * f5
m_H = m_e * (2**(0.5**2.3) * K**(0.9**2.3)) * f1**2 * f5

# === Словарь частиц ===
particles = {
    "electron": m_e,
    "muon": m_mu,
    "tau": m_tau,
    "up": m_u,
    "down": m_d,
    "strange": m_s,
    "charm": m_c,
    "bottom": m_b,
    "top": m_t,
    "proton": m_p,
    "neutron": m_n,
    "W": m_W,
    "Z": m_Z,
    "Higgs": m_H
}

# Пример вывода
for name, mass in particles.items():
    print(f"{name}: {mass:.3e} kg")
