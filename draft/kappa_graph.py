import numpy as np
from scipy.optimize import root

# =========================
# параметры
# =========================
d = 3
K0 = 2 * d
alpha = 1.0
beta = 1.0 / (2 * d)
gamma = 1.0
eta = 0.1
T = 0.5
lam = 0.01

N = 0.3576e122
lnN = np.log(N)

# =========================
# энтропия
# =========================
def entropy(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -p*np.log(p) - (1-p)*np.log(1-p)

# =========================
# x(c)
# =========================
def x_of_c(c):
    return c * (lnN)**(1/3)

# =========================
# лагранжиан
# =========================
def L(c, K, kappa):
    x = x_of_c(c)
    p = x / N**(1/d)

    spec = lam * (x*K**2/N**(1/d) - 1) / (x*K**2/N**(1/d) + 1)**2

    energy_K = kappa * (K - K0)**2
    local = alpha * (1 + beta * x)
    entropy_term = T * entropy(p)
    transport = gamma * (K + 2*eta*x) / (c**2)

    return energy_K + local - entropy_term - spec + transport

# =========================
# система уравнений вариации
# =========================
def equations(vars, kappa):
    c, K = vars

    eps = 1e-6

    # производные численно
    dLc = (L(c+eps, K, kappa) - L(c-eps, K, kappa)) / (2*eps)
    dLK = (L(c, K+eps, kappa) - L(c, K-eps, kappa)) / (2*eps)

    return [dLc, dLK]

# =========================
# решение
# =========================
def solve_system(kappa, guess=(1.0, 100.0)):
    sol = root(equations, guess, args=(kappa,))
    return sol.x

# =========================
# скан κ
# =========================
kappas = np.linspace(0.1, 1000, 1000)

for k in kappas:
    c, K = solve_system(k)
    print(f"κ={k:.3f} -> c={c:.4f}, K={K:.4f}")