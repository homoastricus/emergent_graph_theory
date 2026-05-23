import matplotlib.pyplot as plt
import numpy as np

# RENORMALIZED CRITICAL GRAPH THEORY
# Emergent geometric / nonlocal graph phase

N = 4.2e121
logN = np.log(N)
mu_c = 2 / 3
Nc = N ** mu_c
K = 6.0
eps = 1.0
gamma = 2.0
T = 0.4
beta = 3.0
lam = 10.0

# FREE ENERGY DENSITY

def free_energy_density(x):
    if x <= 1e-15:
        return np.inf

    k = x * Nc
    k_eff = K + lam * k

    F_edges = eps * x
    F_frustration = gamma * x ** 2
    S = -T * x * np.log(1 / x)
    F_transport = beta / np.log(k_eff)

    F = (
        F_edges
        +
        F_frustration
        +
        S
        +
        F_transport
    )

    return F

# SCAN PARAMETER SPACE

xs = np.logspace(-8, 2, 6000)

Fs_list = []
terms_list = []

for x in xs:

    k = x * Nc
    k_eff = K + lam * k

    F_edges = eps * x
    F_frustration = gamma * x ** 2
    S = -T * x * np.log(1 / x)
    F_transport = beta / np.log(k_eff)

    F_total = F_edges + F_frustration + S + F_transport

    Fs_list.append(F_total)
    terms_list.append([F_edges, F_frustration, S, F_transport])

Fs = np.array(Fs_list)
terms = np.array(terms_list)

# VARIATIONAL MINIMUM

idx = np.argmin(Fs)
x_opt = xs[idx]

k_opt = x_opt * Nc
p_opt = k_opt / N
mu_opt = np.log(k_opt) / np.log(N)

# RESULTS

print("RENORMALIZED CRITICAL THEORY")
print(f"N = {N:.2e}")

print(f"\nfree_energy_value = {Fs[idx]:.6e}")

print("\nOPTIMAL STATE")
print(f"x_opt = {x_opt:.6e}")
print(f"k_opt = {k_opt:.6e}")
print(f"p_opt = {p_opt:.6e}")

U = logN / np.log(K * p_opt)
print(f"U = {U:.6f}")

print(f"mu_opt = {mu_opt:.6f}")

print("\nEXPECTED CRITICAL SCALING")
print(f"mu_c = 2/3 = {2 / 3:.6f}")

# EMERGENT GEOMETRY

k_eff = K + lam * k_opt
L = logN / np.log(k_eff)

print("\nEMERGENT GEOMETRY")
print(f"k_eff = {k_eff:.6e}")
print(f"Mean path length L = {L:.6f}")

# SECOND DERIVATIVE

second = np.gradient(np.gradient(Fs, xs), xs)
curvature = second[idx]

print("\nCRITICAL CURVATURE")
print(f"F''(x*) = {curvature:.6e}")

# DECOMPOSITION AT MINIMUM

F_edges_opt = terms[idx, 0]
F_frustration_opt = terms[idx, 1]
S_opt = terms[idx, 2]
F_transport_opt = terms[idx, 3]

print("\nMINIMUM DECOMPOSITION")

print(f"F_edges       = {F_edges_opt:.6e}")
print(f"F_frustration = {F_frustration_opt:.6e}")
print(f"S term        = {S_opt:.6e}")
print(f"F_transport   = {F_transport_opt:.6e}")

print("\nCHECK SUM:")
print(f"{Fs[idx]:.6e}")

# FULL FREE ENERGY PLOT

plt.figure(figsize=(10, 6))

plt.plot(np.log10(xs), Fs, linewidth=2)

plt.axvline(
    np.log10(x_opt),
    color='red',
    linestyle='--',
    label=f'x* = {x_opt:.3e}'
)

plt.xlabel(r'$\log_{10}(x)$')
plt.ylabel('Free Energy Density')
plt.title('Critical Emergent Graph Phase')
plt.grid(True)
plt.legend()
plt.show()

# LOCAL ZOOM

window = 0.6

mask = (
    (np.log10(xs) > np.log10(x_opt) - window)
    &
    (np.log10(xs) < np.log10(x_opt) + window)
)

plt.figure(figsize=(10, 6))

plt.plot(np.log10(xs[mask]), Fs[mask], linewidth=3)

plt.axvline(
    np.log10(x_opt),
    color='red',
    linestyle='--',
    label='critical saddle'
)

plt.xlabel(r'$\log_{10}(x)$')
plt.ylabel('Free Energy Density')
plt.title('Zoom Near Critical Saddle')
plt.grid(True)
plt.legend()
plt.show()

# SECOND DERIVATIVE PLOT

plt.figure(figsize=(10, 6))

plt.plot(np.log10(xs), second, linewidth=2)

plt.axvline(np.log10(x_opt), color='red', linestyle='--')
plt.axhline(0, color='black')

plt.xlabel(r'$\log_{10}(x)$')
plt.ylabel(r"$F''(x)$")
plt.title('Critical Curvature')
plt.grid(True)
plt.show()

# SCALING CHECK
print("\nSCALING TEST")
print("Expected:")
print("p ~ N^(-1/3)")
print("k ~ N^(2/3)")

print("\nПолучено:")
print(f"p * N^(1/3) = {p_opt * N ** (1 / 3):.6e}")
print(f"k / N^(2/3) = {k_opt / N ** (2 / 3):.6e}")

# EFFECTIVE FUNCTIONAL

print("\nEFFECTIVE FUNCTIONAL")
print("""
F[x] = eps*x + gamma*x^2 - T*x*log(1/x) + beta/log(K + lambda*k) где k = x*N^(2/3)
""")