import math
import numpy as np
from scipy.optimize import minimize_scalar

# --- БАЗА ---
K = 6.0
lnK = math.log(K)

# Константы
euler_mascheroni = 0.5772156649015329
paperfolding = 0.850736
feigenbaum_delta = 4.669201609102990

# --- ТРИ ТOЖДЕСТВА (нормализованные) ---

def id_euler(lnN):
    # логарифмическая чувствительность
    return (math.log(lnN) / K)

def id_paperfolding(lnN):
    N = math.exp(lnN)
    return (lnK * N**(1/3)) / (K + N**(1/3))

def id_feigenbaum(lnN):
    # слабая зависимость, но не константа
    return K * (K + 1.0/lnN) / (K + 1.0/math.log(lnK))

IDENTITIES = [
    ("Euler-Mascheroni", id_euler, euler_mascheroni),
    ("Paperfolding", id_paperfolding, math.log(3) * paperfolding),
    ("Feigenbaum δ", id_feigenbaum, feigenbaum_delta),
]

# --- χ² ---
def chi2(lnN):
    total = 0.0
    for _, f, target in IDENTITIES:
        val = f(lnN)
        total += ((val - target) / target) ** 2
    return total / len(IDENTITIES)

# --- ПОИСК МИНИМУМА ---
res = minimize_scalar(chi2, bounds=(260, 300), method='bounded')
lnN_star = res.x
chi_min = res.fun

print("\n=== 3-CONSTANT TEST ===")
print(f"lnN* = {lnN_star:.10f}")
print(f"N*   = {math.exp(lnN_star):.3e}")
print(f"χ²   = {chi_min:.6e}")

# --- ШИРИНА ---
target = chi_min * 10
left = lnN_star
right = lnN_star

for _ in range(200):
    if chi2(left - 0.001) < target:
        left -= 0.001
    else:
        break

for _ in range(200):
    if chi2(right + 0.001) < target:
        right += 0.001
    else:
        break

width = right - left
print(f"ΔlnN = {width:.6f}")
print(f"relative width = {width/lnN_star*100:.4f}%")

# --- LOO ---
print("\nLOO TEST:")
for i in range(len(IDENTITIES)):
    subset = [IDENTITIES[j] for j in range(len(IDENTITIES)) if j != i]

    def chi_sub(lnN):
        total = 0.0
        for _, f, target in subset:
            total += ((f(lnN) - target)/target)**2
        return total / len(subset)

    r = minimize_scalar(chi_sub, bounds=(260, 300), method='bounded')
    delta = abs(r.x - lnN_star) / lnN_star * 100

    print(f"without {IDENTITIES[i][0]:<20}: Δ = {delta:.6f}%")

# --- BOOTSTRAP ---
print("\nBOOTSTRAP (2 из 3):")
samples = []

for _ in range(500):
    idx = np.random.choice(3, 2, replace=False)
    subset = [IDENTITIES[i] for i in idx]

    def chi_sub(lnN):
        total = 0.0
        for _, f, target in subset:
            total += ((f(lnN) - target)/target)**2
        return total / len(subset)

    r = minimize_scalar(chi_sub, bounds=(260, 300), method='bounded')
    samples.append(r.x)

samples = np.array(samples)

print(f"median lnN = {np.median(samples):.6f}")
print(f"std        = {np.std(samples):.6f}")
print(f"68% range  = [{np.percentile(samples,16):.3f}, {np.percentile(samples,84):.3f}]")