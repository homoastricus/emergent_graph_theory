import math

K = 6.0
pi = math.pi
lnK = math.log(K)
N = 4.197668e121
lnN = math.log(N)

eps = 0.001  # 0.1% изменение
lnN_plus = lnN * (1 + eps)
N_plus = math.exp(lnN_plus)

# Функции
def alpha(N): return 2 * lnK**2 / (pi * math.log(N))
def me(N): return 4*pi * math.log(N)**4 / (K**0.5 * N**(1/3))
def mp(N): return math.sqrt(pi) * math.log(N)**6 / (K**1.5 * N**(1/3))
def me_over_mp(N): return 4 * pi**0.5 * K / math.log(N)**2
def Rinf_a0(N): return lnK**2 / (2 * pi**2 * math.log(N))

print("SLOPE CONSISTENCY TEST")
print(f"{'Величина':<20} {'d(ln f)/d(ln ln N)':<25} {'Ожидание':<15} {'Статус'}")
print("-" * 75)

# α
f0 = alpha(N)
f1 = alpha(N_plus)
slope = (math.log(f1) - math.log(f0)) / (math.log(lnN_plus) - math.log(lnN))
print(f"{'α':<20} {slope:<25.4f} {'-1':<15} {'✅' if abs(slope + 1) < 0.01 else '❌'}")

# m_e/m_p
f0 = me_over_mp(N)
f1 = me_over_mp(N_plus)
slope = (math.log(f1) - math.log(f0)) / (math.log(lnN_plus) - math.log(lnN))
print(f"{'m_e/m_p':<20} {slope:<25.4f} {'-2':<15} {'✅' if abs(slope + 2) < 0.01 else '❌'}")

# R∞·a₀
f0 = Rinf_a0(N)
f1 = Rinf_a0(N_plus)
slope = (math.log(f1) - math.log(f0)) / (math.log(lnN_plus) - math.log(lnN))
print(f"{'R∞·a₀':<20} {slope:<25.4f} {'-1':<15} {'✅' if abs(slope + 1) < 0.01 else '❌'}")

# Дополнительно: m_e
f0 = me(N)
f1 = me(N_plus)
slope = (math.log(f1) - math.log(f0)) / (math.log(lnN_plus) - math.log(lnN))
print(f"{'m_e':<20} {slope:<25.4f} {'—':<15} {'—'}")

# Дополнительно: m_p
f0 = mp(N)
f1 = mp(N_plus)
slope = (math.log(f1) - math.log(f0)) / (math.log(lnN_plus) - math.log(lnN))
print(f"{'m_p':<20} {slope:<25.4f} {'—':<15} {'—'}")