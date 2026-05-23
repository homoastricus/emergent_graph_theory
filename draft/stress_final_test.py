"""
ПЕРЕСТРОЕНИЕ МОДЕЛИ A(n, b) БЕЗ ЦЕНТРИРОВАНИЯ

Вместо использования A_i = (δ_i - δ₀)·ln²N (центрированные остатки),
используем АБСОЛЮТНЫЕ значения γ_i для построения модели A.

Формула: γ_i = (b/ln K)(ln N)² - (a/ln K)(ln N)ln(ln N) + C·ln N
         C = π·(n + δ_b)·exp(A/(ln N)²)

Из этого определения ИЗВЛЕКАЕМ A для каждой константы,
затем строим модель A(n, b) без вычитания среднего.
"""

import math
import numpy as np
from scipy.special import gammaln
from collections import defaultdict

# ============================================================
# ПАРАМЕТРЫ
# ============================================================
K = 6.0
pi = math.pi
lnK = math.log(K)
gamma_E = 0.5772156649015329

N_opt = 4.197668e121
lnN = math.log(N_opt)
lnlnN = math.log(lnN)
N13 = N_opt ** (1/3)
lnN2 = lnN ** 2

# ============================================================
# ИЗМЕРЕННЫЕ γ_i
# ============================================================
gamma_measured = {
    'ħ': 0.192835, 'h': 0.192835,
    'l_P': -0.104361, 't_P': -0.222142,
    'm_P': 0.179369, 'E_P': 0.415022, 'T_P': -0.364637,
    'c': 0.117799, 'G': -0.048154, 'k_B': 0.224670,
    'α': -0.015396,
    'm_e': 0.515264, 'm_muon': 0.061400, 'm_tau': 0.248402,
    'm_proton': -0.120978, 'm_neutron': -0.462800,
    'm_W': -0.100838, 'm_Z': 0.715684, 'm_Higgs': -0.267871,
    'm_pion': 0.127663, 'm_pion0': 0.271797, 'm_kaon0': -0.193698,
    'm_D0': -1.079562, 'm_J_psi': -0.385841, 'm_eta': -0.248487,
    'm_Upsilon_1S': -1.179646,
    'm_quark_u': -0.682866, 'm_quark_d': 0.730184,
    'm_quark_s': -1.474050, 'm_quark_c': 0.195006,
    'm_quark_b': -0.592949, 'm_quark_t': 0.120094,
    'Rydberg': 0.409414, 'Bohr_radius': -0.424833,
    'Compton_e': -0.460215, 'Compton_p': 0.196014,
    'e_charge': -0.105394,
    'tau_mu': -0.100283, 'tau_tau': 0.012300, 'tau_pion': 0.173669,
    'tau_neutron': -0.243215, 'tau_kaon': -0.150499,
    'tau_D_plus': -0.031627, 'tau_B_plus': 0.385605,
    'tau_Lambda_b': -0.008957, 'tau_D0': 0.060468,
    'm_proton/m_e': -0.636243, 'm_muon/m_e': -0.453865,
    'm_tau/m_e': -0.266862, 'm_W/m_e': -0.616102,
    'm_Z/m_e': 0.200420, 'm_Higgs/m_e': -0.783135,
    'm_W/m_Z': -0.816522, 'm_Higgs/m_W': -0.167033,
    'm_P/m_e': -0.335895,
}

# (name, a, b)
all_constants = [
    # b = 1/3
    ('ħ', 3, 1/3), ('h', 3, 1/3),
    ('l_P', 2, 1/3), ('t_P', -2, 1/3),
    ('G', 13, 1/3), ('k_B', 8, 1/3),
    ('m_e', 4, 1/3), ('m_muon', 5, 1/3), ('m_tau', 5, 1/3),
    ('m_proton', 6, 1/3), ('m_neutron', 6, 1/3),
    ('m_W', 6, 1/3), ('m_Z', 6, 1/3), ('m_Higgs', 6, 1/3),
    ('m_pion', 6, 1/3), ('m_pion0', 4, 1/3), ('m_kaon0', 6, 1/3),
    ('m_D0', 6, 1/3), ('m_J_psi', 5, 1/3), ('m_eta', 5, 1/3),
    ('m_Upsilon_1S', 6, 1/3),
    ('m_quark_u', 5, 1/3), ('m_quark_d', 5, 1/3), ('m_quark_s', 4, 1/3),
    ('m_quark_c', 6, 1/3), ('m_quark_b', 6, 1/3), ('m_quark_t', 6, 1/3),
    # b = 0
    ('m_P', -3, 0), ('E_P', 5, 0), ('c', 4, 0), ('α', -1, 0),
    ('Rydberg', 3, 0), ('Bohr_radius', -4, 0),
    ('Compton_e', -5, 0), ('Compton_p', -7, 0),
    ('e_charge', -7, 0),
    ('tau_mu', -2, 0), ('tau_tau', -5, 0), ('tau_pion', -4, 0),
    ('tau_kaon', -3, 0), ('tau_D_plus', -4, 0),
    ('tau_B_plus', -5, 0), ('tau_Lambda_b', -5, 0), ('tau_D0', -4, 0),
    ('m_proton/m_e', 2, 0), ('m_muon/m_e', 1, 0), ('m_tau/m_e', 1, 0),
    ('m_W/m_e', 2, 0), ('m_Z/m_e', 2, 0), ('m_Higgs/m_e', 2, 0),
    ('m_W/m_Z', 0, 0), ('m_Higgs/m_W', 0, 0), ('m_P/m_e', 0, 0),
    # b = -1/3
    ('T_P', -4, -1/3),
]

# ============================================================
# ИЗВЛЕЧЕНИЕ АБСОЛЮТНОГО A_i ИЗ γ_i
# ============================================================
def extract_A_from_gamma(gamma, a, b):
    """Извлекает A из γ через обратную формулу"""
    if abs(b) < 1e-6:
        n = -a
        delta_b = 0
    elif abs(b - 1/3) < 1e-6:
        n = 16 - a
        delta_b = gamma_E
    elif abs(b + 1/3) < 1e-6:
        n = -a
        delta_b = 0
    else:
        return None, None

    # Из γ = (b/ln K)(ln N)² - (a/ln K)(ln N)ln(ln N) + C·ln N
    # находим C
    C = (gamma - (b/lnK)*lnN**2 + (a/lnK)*lnN*lnlnN) / lnN

    # Из C = π·(n + δ_b)·exp(A/ln²N)
    # находим A
    if C <= 0 or (n + delta_b) <= 0:
        return None, None

    ratio = C / (pi * (n + delta_b))
    if ratio <= 0:
        return None, None

    A = math.log(ratio) * lnN2

    return A, C

# ============================================================
# ВЫЧИСЛЕНИЕ A ДЛЯ ВСЕХ КОНСТАНТ
# ============================================================
data_abs = []
for name, a, b in all_constants:
    if name not in gamma_measured:
        continue

    gamma = gamma_measured[name]
    A, C = extract_A_from_gamma(gamma, a, b)

    if A is None:
        continue

    if abs(b) < 1e-6:
        n = -a
    elif abs(b - 1/3) < 1e-6:
        n = 16 - a
    elif abs(b + 1/3) < 1e-6:
        n = -a
    else:
        continue

    data_abs.append({
        'name': name, 'a': a, 'b': b, 'n': n,
        'A': A, 'C': C, 'gamma': gamma,
    })

# ============================================================
# ПОСТРОЕНИЕ МОДЕЛИ A(n, b) БЕЗ ЦЕНТРИРОВАНИЯ
# ============================================================
print("=" * 100)
print("ПЕРЕСТРОЕНИЕ МОДЕЛИ A(n, b) ПО АБСОЛЮТНЫМ ЗНАЧЕНИЯМ")
print("=" * 100)

for b_target, b_label in [(0, "b=0 (безразмерные)"), (1/3, "b=1/3 (массы)")]:
    items = [d for d in data_abs if abs(d['b'] - b_target) < 1e-6]

    if len(items) < 4:
        continue

    n_arr = np.array([d['n'] for d in items])
    A_arr = np.array([d['A'] for d in items])

    print(f"\n{'─'*100}")
    print(f"КЛАСС {b_label}  (n = {len(items)})")
    print(f"{'─'*100}")

    # Модель: A = α·n + β·ln Γ(n+γ_E) + ε
    log_gamma_arr = np.array([gammaln(d['n'] + gamma_E) for d in items])
    X = np.column_stack([n_arr, log_gamma_arr, np.ones(len(items))])
    coeffs, _, _, _ = np.linalg.lstsq(X, A_arr, rcond=None)
    alpha, beta, eps = coeffs
    A_pred = X @ coeffs
    R2 = 1 - np.sum((A_arr - A_pred)**2) / np.sum((A_arr - np.mean(A_arr))**2)

    print(f"\n  МОДЕЛЬ A(n):")
    print(f"  A(n) = {alpha:.4f}·n + {beta:.4f}·ln Γ(n+γ_E) + {eps:.4f}")
    print(f"  R² = {R2:.4f}")

    # Таблица
    print(f"\n  {'Константа':<16} {'n':>4} {'A_извл':>12} {'A_модель':>12} {'Остаток':>10}")
    print(f"  {'-'*60}")
    for i, d in enumerate(items):
        residual = A_arr[i] - A_pred[i]
        print(f"  {d['name']:<16} {d['n']:>4} {A_arr[i]:>12.4f} {A_pred[i]:>12.4f} {residual:>10.4f}")

    print(f"\n  Сохранённая модель для использования:")
    print(f"  alpha_{b_target} = {alpha:.6f}")
    print(f"  beta_{b_target}  = {beta:.6f}")
    print(f"  eps_{b_target}   = {eps:.6f}")
    print(f"  R2_{b_target}    = {R2:.6f}")

# ============================================================
# ВЫВОД: ФИНАЛЬНЫЕ КОЭФФИЦИЕНТЫ
# ============================================================
print(f"\n{'='*100}")
print("ФИНАЛЬНЫЕ КОЭФФИЦИЕНТЫ ДЛЯ АНАЛИТИЧЕСКОЙ ФОРМУЛЫ")
print(f"{'='*100}")

# Сохраняем лучшие коэффициенты
final_models = {}

for b_target in [0, 1/3]:
    items = [d for d in data_abs if abs(d['b'] - b_target) < 1e-6]
    if len(items) < 4:
        continue

    n_arr = np.array([d['n'] for d in items])
    A_arr = np.array([d['A'] for d in items])
    log_gamma_arr = np.array([gammaln(d['n'] + gamma_E) for d in items])
    X = np.column_stack([n_arr, log_gamma_arr, np.ones(len(items))])
    coeffs, _, _, _ = np.linalg.lstsq(X, A_arr, rcond=None)

    final_models[b_target] = {
        'alpha': coeffs[0],
        'beta': coeffs[1],
        'eps': coeffs[2],
    }

    print(f"\n  Класс b = {b_target}:")
    print(f"    A(n) = {coeffs[0]:.6f}·n + {coeffs[1]:.6f}·ln Γ(n+γ_E) + {coeffs[2]:.6f}")

print(f"\n  Для использования в полной формуле:")
print(f"  def compute_A(n, b):")
print(f"      if abs(b) < 1e-6:")
print(f"          return {final_models[0]['alpha']:.6f} * n + {final_models[0]['beta']:.6f} * gammaln(n + gamma_E) + {final_models[0]['eps']:.6f}")
print(f"      elif abs(b - 1/3) < 1e-6:")
print(f"          return {final_models[1/3]['alpha']:.6f} * n + {final_models[1/3]['beta']:.6f} * gammaln(n + gamma_E) + {final_models[1/3]['eps']:.6f}")