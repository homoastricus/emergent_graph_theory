"""
ФИНАЛЬНЫЙ АНАЛИЗ A_i: РАЗДЕЛЕНИЕ НА КЛАССЫ И ПОИСК СТРУКТУРЫ

Ключевое наблюдение из предыдущего анализа:
  — A_i СИЛЬНО зависит от b (класса константы)
  — Для b=0: A_i ~ 80-120 (компактный кластер)
  — Для b=1/3: A_i ~ -10 до +35 (другой компактный кластер)
  — Для b=-1/3, -2/3: A_i сильно отрицательные (от -2576 до -955)

Вывод: A_i НЕ является универсальной функцией только от n_i.
       A_i = f(n_i, b) — зависит от геометрического класса.

План:
  1. Разделить константы по b
  2. Для каждого b найти модель A(n)
  3. Проверить, выражается ли A через n_i, a_i, K, π
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
lnN2 = lnN ** 2

# ============================================================
# ДАННЫЕ
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
    'tau_kaon': -0.150499, 'tau_D_plus': -0.031627,
    'tau_B_plus': 0.385605, 'tau_Lambda_b': -0.008957, 'tau_D0': 0.060468,
    'm_proton/m_e': -0.636243, 'm_muon/m_e': -0.453865,
    'm_tau/m_e': -0.266862, 'm_W/m_e': -0.616102,
    'm_Z/m_e': 0.200420, 'm_Higgs/m_e': -0.783135,
    'm_W/m_Z': -0.816522, 'm_Higgs/m_W': -0.167033,
    'm_P/m_e': -0.335895,
}

all_constants = [
    ('ħ','quantum',3,1/3), ('h','quantum',3,1/3),
    ('l_P','planck',2,1/3), ('t_P','planck',-2,1/3),
    ('m_P','planck',-3,0), ('E_P','planck',5,0),
    ('T_P','planck',-4,-1/3),
    ('c','fundamental',4,0), ('G','fundamental',13,1/3),
    ('k_B','fundamental',8,1/3), ('α','fundamental',-1,0),
    ('m_e','lepton',4,1/3), ('m_muon','lepton',5,1/3),
    ('m_tau','lepton',5,1/3),
    ('m_proton','baryon',6,1/3), ('m_neutron','baryon',6,1/3),
    ('m_W','boson',6,1/3), ('m_Z','boson',6,1/3),
    ('m_Higgs','boson',6,1/3),
    ('m_pion','meson',6,1/3), ('m_pion0','meson',4,1/3),
    ('m_kaon0','meson',6,1/3), ('m_D0','meson',6,1/3),
    ('m_J_psi','meson',5,1/3), ('m_eta','meson',5,1/3),
    ('m_Upsilon_1S','meson',6,1/3),
    ('m_quark_u','quark',5,1/3), ('m_quark_d','quark',5,1/3),
    ('m_quark_s','quark',4,1/3), ('m_quark_c','quark',6,1/3),
    ('m_quark_b','quark',6,1/3), ('m_quark_t','quark',6,1/3),
    ('Rydberg','atomic',3,0), ('Bohr_radius','atomic',-4,0),
    ('Compton_e','atomic',-5,0), ('Compton_p','atomic',-7,0),
    ('e_charge','electromagnetic',-7,0),
    ('tau_mu','lifetime',-2,0), ('tau_tau','lifetime',-5,0),
    ('tau_pion','lifetime',-4,0), ('tau_kaon','lifetime',-3,0),
    ('tau_D_plus','lifetime',-4,0), ('tau_B_plus','lifetime',-5,0),
    ('tau_Lambda_b','lifetime',-5,0), ('tau_D0','lifetime',-4,0),
    ('m_proton/m_e','ratio',2,0), ('m_muon/m_e','ratio',1,0),
    ('m_tau/m_e','ratio',1,0), ('m_W/m_e','ratio',2,0),
    ('m_Z/m_e','ratio',2,0), ('m_Higgs/m_e','ratio',2,0),
    ('m_W/m_Z','ratio',0,0), ('m_Higgs/m_W','ratio',0,0),
    ('m_P/m_e','ratio',0,0),
]

def compute_Ci(gamma, a, b):
    return gamma/lnN + (b/lnK)*lnN - (a/lnK)*lnlnN

# Вычисляем все A_i
all_data = []
for name, sector, a, b in all_constants:
    if name not in gamma_measured:
        continue
    gamma = gamma_measured[name]
    Ci = compute_Ci(gamma, a, b)
    Ci_pi = Ci / pi

    if abs(b) < 1e-6:
        n_i = round(Ci_pi)
        base = n_i
    elif abs(b - 1/3) < 1e-6:
        n_i = round(Ci_pi - gamma_E)
        base = n_i + gamma_E
    elif abs(b + 1/3) < 1e-6:
        n_i = round(Ci_pi)
        base = n_i
    else:
        n_i = round(Ci_pi)
        base = n_i

    if abs(base) < 1e-10:
        continue

    delta_i = math.log(Ci / (pi * base))

    # Используем глобальный δ₀
    delta0 = -8.276234e-06
    A_i = (delta_i - delta0) * lnN2

    all_data.append({
        'name': name, 'sector': sector, 'a': a, 'b': b,
        'n_i': n_i, 'A_i': A_i, 'Ci': Ci,
    })

# ============================================================
# АНАЛИЗ ПО КЛАССАМ b
# ============================================================
print("=" * 100)
print("АНАЛИЗ A_i ПО КЛАССАМ b")
print("=" * 100)

# Разделяем по b
by_b = defaultdict(list)
for d in all_data:
    b_rounded = round(d['b'], 6)
    by_b[b_rounded].append(d)

for b_val in sorted(by_b.keys()):
    items = by_b[b_val]
    A_vals = np.array([d['A_i'] for d in items])
    n_vals = np.array([d['n_i'] for d in items])

    print(f"\n{'─'*100}")
    print(f"КЛАСС b = {b_val}  (n = {len(items)})")
    print(f"{'─'*100}")

    print(f"\n  Статистика A_i: среднее = {np.mean(A_vals):.2f}, стд = {np.std(A_vals):.2f}")
    print(f"  Диапазон: [{np.min(A_vals):.2f}, {np.max(A_vals):.2f}]")

    # Таблица
    print(f"\n  {'Константа':<18} {'n':>4} {'a':>4} {'A_i':>12} {'Сектор':<15}")
    print(f"  {'-'*60}")
    sorted_items = sorted(items, key=lambda d: d['A_i'])
    for d in sorted_items:
        print(f"  {d['name']:<18} {d['n_i']:>4} {d['a']:>4} {d['A_i']:>12.4f} {d['sector']:<15}")

    # Модель A(n) внутри класса
    if len(items) >= 4:
        log_gamma = np.array([gammaln(d['n_i'] + gamma_E) for d in items])
        X = np.column_stack([n_vals, log_gamma, np.ones(len(items))])
        coeffs, _, _, _ = np.linalg.lstsq(X, A_vals, rcond=None)
        alpha, beta, eps = coeffs
        A_pred = X @ coeffs
        R2 = 1 - np.sum((A_vals - A_pred)**2) / np.sum((A_vals - np.mean(A_vals))**2)

        print(f"\n  МОДЕЛЬ A(n) для b={b_val}:")
        print(f"  A_i = {alpha:.2f}·n_i + {beta:.2f}·ln Γ(n_i+γ_E) + {eps:.2f}")
        print(f"  R² = {R2:.4f}")

# ============================================================
# КЛЮЧЕВОЙ ВЫВОД
# ============================================================
print(f"\n{'='*100}")
print("КЛЮЧЕВОЙ ВЫВОД")
print(f"{'='*100}")
print(f"""
  A_i СУЩЕСТВЕННО РАЗЛИЧАЕТСЯ ДЛЯ РАЗНЫХ b:
  
  b = 0 (безразмерные):
    — Компактный кластер A_i ~ 70-125
    — Слабо зависит от n_i
    — Разные сектора (атомные, времена жизни) перемешаны
    
  b = 1/3 (массы):
    — Другой компактный кластер A_i ~ -10 до +35
    — Зависит от n_i: A_i растёт с n
    — Внутри одного n есть разброс ~10
    
  b = -1/3, -2/3 (полевые):
    — Сильно отрицательные A_i ~ -2500 до -1000
    — Выделенный класс, возможно, требует отдельных формул
    
  Это означает, что A_i = f(n_i, b) — зависит от геометрического класса.
  Внутри каждого класса есть дополнительная структура по секторам.
""")