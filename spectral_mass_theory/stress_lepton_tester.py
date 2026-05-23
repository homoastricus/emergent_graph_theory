import numpy as np
from numpy.linalg import eigh
from scipy.optimize import minimize

# =========================
# ДАННЫЕ
# =========================

d_exp = np.array([
    -13.0, -7.0, 0.0,
    -12.0, -9.0, -3.0,
    -11.5, -6.0, -3.5
])

ckm_exp = np.log10([0.225, 0.041, 0.0037])

# =========================
# BOUNDS
# =========================

bounds = [
    (0.18, 0.30),  # eps
    (0.05, 0.30),  # eps_e

    # a12
    (-2, 2), (0.2, 2), (-2, 2),

    # a23
    (-2, 2), (-2, 2), (-2, 2),

    # b13
    (-2, 2), (-2, 2), (-2, 2),

    # phases
    (0.3, 2.8),
    (0.05, 0.6),
    (0.05, 0.6),

    # shifts
    (-6, 2), (-6, 2), (-6, 2),

    # NEW
    (0.0, 4.0),   # delta_11
    (0.0, 1.5),   # delta_13_u
    (0.0, 1.5),   # delta_13_d

    (0.5, 5.0),   # kappa_d
    (0.8, 2.5)    # p23_d  ← ПАТЧ
]

# =========================
# МАТРИЦЫ
# =========================

def L_quark(eps, a12, a23, b13, phi, d11, d13, p23=1.0, kappa=1.0, kappa2=1.0):
    return kappa * np.array([
        [eps**(4+d11), a12*eps**3, b13*eps**(2+d13)*np.exp(1j*phi)],
        [a12*eps**3, kappa2*eps**2, a23*eps**p23],
        [b13*eps**(2+d13)*np.exp(-1j*phi), a23*eps**p23, 1.0]
    ], dtype=complex)


def L_lepton(eps_e, a12, a23, b13, phi):
    return np.array([
        [eps_e**4, a12 * eps_e**3, b13 * eps_e**3 * np.exp(1j * phi)],
        [a12 * eps_e**3, eps_e**2, a23 * eps_e],
        [b13 * eps_e**3 * np.exp(-1j * phi), a23 * eps_e, 1.0]
    ], dtype=complex)


def diag(L):
    L = (L + L.conj().T) / 2
    vals, vecs = eigh(L)
    idx = np.argsort(vals)
    return vals[idx], vecs[:, idx]


def safe_log(x):
    return np.log10(np.abs(x) + 1e-12)


def CKM(Uu, Ud):
    return np.abs(Uu.conj().T @ Ud)

# =========================
# LOSS
# =========================

def loss(params):
    (eps, eps_e,
     a12_u, a12_d, a12_e,
     a23_u, a23_d, a23_e,
     b13_u, b13_d, b13_e,
     phi, dphi_d, dphi_e,
     cu, cd, ce,
     d11, d13_u, d13_d,
     kappa_d, p23_d) = params

    Lu = L_quark(eps, a12_u, a23_u, b13_u, phi, d11, d13_u, p23=1.0)
    Ld = kappa_d * L_quark(eps, a12_d, a23_d, b13_d,
                           phi + dphi_d, d11, d13_d, p23=p23_d)
    Le = L_lepton(eps_e, a12_e, a23_e, b13_e, phi + dphi_e)

    mu, Uu = diag(Lu)
    md, Ud = diag(Ld)
    me, Ue = diag(Le)

    # массы
    d_pred = np.array([
        safe_log(mu[0]) + cu,
        safe_log(mu[1]) + cu,
        safe_log(mu[2]),

        safe_log(md[0]) + cd,
        safe_log(md[1]) + cd,
        safe_log(md[2]) + cd,

        safe_log(me[0]) + ce,
        safe_log(me[1]) + ce,
        safe_log(me[2]) + ce
    ])

    loss_m = np.mean((d_pred - d_exp)**2)

    # CKM
    V = CKM(Uu, Ud)
    ckm_pred = safe_log([V[0,1], V[1,2], V[0,2]])


    loss_ckm = (
            8 * (ckm_pred[0] - ckm_exp[0]) ** 2 +  # ← усилить Vus
            5 * (ckm_pred[1] - ckm_exp[1]) ** 2 +
            2 * (ckm_pred[2] - ckm_exp[2]) ** 2
    )

    penalty_positive = 0 * (
            np.sum(np.clip(-mu, 0, None) ** 2) +  # штраф за λ_up < 0
            np.sum(np.clip(-md, 0, None) ** 2) +  # штраф за λ_down < 0
            np.sum(np.clip(-me, 0, None) ** 2)  # штраф за λ_lep < 0
    )

    return (
            loss_m +
            2 * loss_ckm +
            penalty_positive
    )

# =========================
# СТАРТ
# =========================

x0 = [
    0.22, 0.18,

    0.8, 0.5, 0.8,
    0.6, 0.9, 0.4,
    0.5, 0.3, 0.6,

    1.2, 0.3, 0.2,

    -5, -5, -4,

    1.5,   # delta_11
    0.5,   # delta_13_u
    0.5,   # delta_13_d

    1.5,   # kappa_d
    1.5    # p23_d  ← ключевой параметр
]

print("ОПТИМИЗАЦИЯ С P23_d...")
res = minimize(loss, x0, method='L-BFGS-B',
               bounds=bounds,
               options={'maxiter': 50000, 'ftol': 1e-12})

# =========================
# ВЫВОД
# =========================

params = res.x

(eps, eps_e,
 a12_u, a12_d, a12_e,
 a23_u, a23_d, a23_e,
 b13_u, b13_d, b13_e,
 phi, dphi_d, dphi_e,
 cu, cd, ce,
 d11, d13_u, d13_d,
 kappa_d, p23_d) = params

Lu = L_quark(eps, a12_u, a23_u, b13_u, phi, d11, d13_u)
Ld = kappa_d * L_quark(eps, a12_d, a23_d, b13_d,
                       phi + dphi_d, d11, d13_d, p23=p23_d)
Le = L_lepton(eps_e, a12_e, a23_e, b13_e, phi + dphi_e)

mu, Uu = diag(Lu)
md, Ud = diag(Ld)
me, Ue = diag(Le)

print("\n" + "="*60)
print(f"SUCCESS: {res.success} | LOSS: {res.fun:.4f}")
print("="*60)

print(f"\nε = {eps:.4f} | ε_e = {eps_e:.4f}")
print(f"δ11 = {d11:.2f}, δ13_u = {d13_u:.2f}, δ13_d = {d13_d:.2f}")
print(f"κ_d = {kappa_d:.2f}, p23_d = {p23_d:.2f}")

print("\nEIGENVALUES:")
print("up   :", mu)
print("down :", md)
print("lep  :", me)

print("\nМАССЫ:")
labels = ['u','c','t','d','s','b','e','μ','τ']
d_pred = np.array([
    safe_log(mu[0])+cu,
    safe_log(mu[1])+cu,
    safe_log(mu[2]),
    safe_log(md[0])+cd,
    safe_log(md[1])+cd,
    safe_log(md[2])+cd,
    safe_log(me[0])+ce,
    safe_log(me[1])+ce,
    safe_log(me[2])+ce
])

for l,p,e in zip(labels, d_pred, d_exp):
    print(f"{l}: {p:.2f} (exp {e}) Δ={abs(p-e):.2f}")

print("\nCKM:")
V = CKM(Uu, Ud)
print("|Vus| =", V[0,1])
print("|Vcb| =", V[1,2])
print("|Vub| =", V[0,2])

print("\nИЕРАРХИИ:")
print("m_c/m_t =", mu[1]/mu[2])
print("m_s/m_b =", md[1]/md[2])
print("m_μ/m_τ =", me[1]/me[2])