import math
import numpy as np
from scipy.special import gammaln

# ============================================================
# КОНСТАНТЫ
# ============================================================
pi = math.pi
gamma_E = 0.5772156649015329

K = 6.0
lnK = math.log(K)

N = 4.197668e121
lnN = math.log(N)
lnlnN = math.log(lnN)
lnN2 = lnN ** 2

# ============================================================
# ДАННЫЕ
# ============================================================
# name: (A_i, a, b, sector, C2_SU3, C2_SU2)

data = {
    'm_e': (5.6456, 4, 1/3, 'lepton', 0, 0.75),
    'm_muon': (-3.5426, 5, 1/3, 'lepton', 0, 0.75),
    'm_tau': (-2.1029, 5, 1/3, 'lepton', 0, 0.75),

    # 'm_quark_u': (-0.6152, 5, 1/3, 'quark', 4/3, 0.75),
    # 'm_quark_d': (10.2639, 5, 1/3, 'quark', 4/3, 0.75),
    # 'm_quark_s': (0.2058, 4, 1/3, 'quark', 4/3, 0.75),
    # 'm_quark_c': (-0.8609, 6, 1/3, 'quark', 4/3, 0.75),
    # 'm_quark_b': (-7.5016, 6, 1/3, 'quark', 4/3, 0.75),
    # 'm_quark_t': (-1.4922, 6, 1/3, 'quark', 4/3, 0.75),

    'm_proton': (1.4404, 6, 1/3, 'baryon', 2, 0.75),
    'm_neutron': (-1.4404, 6, 1/3, 'baryon', 2, 0.75),

    'm_pion': (0.3767, 6, 1/3, 'meson', 2, 0),
    'm_pion0': (14.3832, 4, 1/3, 'meson', 2, 0),
    'm_kaon0': (-2.3316, 6, 1/3, 'meson', 2, 0),
    'm_D0': (-9.7979, 6, 1/3, 'meson', 2, 0),
    'm_J_psi': (3.4768, 5, 1/3, 'meson', 2, 0),
    'm_eta': (4.5343, 5, 1/3, 'meson', 2, 0),
    'm_Upsilon_1S': (-10.6415, 6, 1/3, 'meson', 2, 0),

    'm_W': (-1.8245, 6, 1/3, 'boson', 0, 2),
    'm_Z': (5.0566, 6, 1/3, 'boson', 0, 0),
    'm_Higgs': (-3.2322, 6, 1/3, 'boson', 0, 0.75),
}

# γ_i
gamma_measured = {
    'm_e': 0.515264, 'm_muon': 0.061400, 'm_tau': 0.248402,
    'm_quark_u': -0.682866, 'm_quark_d': 0.730184,
    'm_quark_s': -1.474050, 'm_quark_c': 0.195006,
    'm_quark_b': -0.592949, 'm_quark_t': 0.120094,
    'm_proton': -0.120978, 'm_neutron': -0.462800,
    'm_pion': 0.127663, 'm_pion0': 0.271797,
    'm_kaon0': -0.193698, 'm_D0': -1.079562,
    'm_J_psi': -0.385841, 'm_eta': -0.248487,
    'm_Upsilon_1S': -1.179646,
    'm_W': -0.100838, 'm_Z': 0.715684, 'm_Higgs': -0.267871,
}

# ============================================================
# C_i
# ============================================================
def compute_Ci(gamma, a, b):
    return gamma / lnN + (b / lnK) * lnN - (a / lnK) * lnlnN

Ci_data = {}
for name, (A_i, a, b, *_rest) in data.items():
    gamma = gamma_measured[name]
    Ci_data[name] = compute_Ci(gamma, a, b)

# ============================================================
# n_i (СПЕКТР)
# ============================================================
n_data = {}
for name, Ci in Ci_data.items():
    n_data[name] = round(Ci / pi - gamma_E)

# ============================================================
# ДИЗАЙН-МАТРИЦА
# ============================================================
names = list(data.keys())

# сектора
sectors = sorted(set(d[3] for d in data.values()))
sector_index = {s: i for i, s in enumerate(sectors)}

X = []
y = []

for name in names:
    A_i, a, b, sector, C2_3, C2_2 = data[name]
    n_i = n_data[name]

    # базовые физические признаки
    features = [
        gammaln(n_i + gamma_E),   # log Γ
        n_i,                      # линейный спектр
        C2_3,                     # SU3
        C2_2,                     # SU2
        1.0                       # константа
    ]

    # секторные dummy
    sec_vec = [0]*len(sectors)
    sec_vec[sector_index[sector]] = 1

    X.append(features + sec_vec)
    y.append(A_i)

X = np.array(X)
y = np.array(y)

# ============================================================
# РЕГРЕССИЯ
# ============================================================
coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
y_pred = X @ coeffs

residuals = y - y_pred
R2 = 1 - np.sum(residuals**2) / np.sum((y - np.mean(y))**2)

# ============================================================
# ВЫВОД
# ============================================================
print("="*80)
print("ФИНАЛЬНАЯ СПЕКТРАЛЬНАЯ МОДЕЛЬ")
print("="*80)

labels = [
    "logΓ(n+γ_E)",
    "n",
    "C2_SU3",
    "C2_SU2",
    "const"
] + [f"sector_{s}" for s in sectors]

for l, c in zip(labels, coeffs):
    print(f"{l:20s}: {c:10.5f}")

print(f"\nR² = {R2:.4f}")

print("\nТаблица:")
print(f"{'name':15s} {'n_i':>5} {'A_i':>10} {'model':>10} {'res':>10}")
print("-"*60)

for i, name in enumerate(names):
    print(f"{name:15s} {n_data[name]:5d} {y[i]:10.4f} {y_pred[i]:10.4f} {residuals[i]:10.4f}")

print("\nСтатистика:")
print(f"std = {np.std(residuals):.4f}")
print(f"max = {np.max(np.abs(residuals)):.4f}")