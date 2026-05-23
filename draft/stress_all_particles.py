"""
ЕДИНАЯ МОДЕЛЬ FLAVOR — ИСПРАВЛЕННАЯ
=====================================
Исправления:
1. Убран структурный штраф (α,β)
2. Раздельные α,β для секторов
3. Расширены границы масштабов
4. Усилен CKM-штраф
5. Добавлен штраф за неиерархичность масс
"""

import numpy as np
from numpy.linalg import eigh
from scipy.optimize import minimize

# ============================================================
# ДАННЫЕ
# ============================================================

d_exp = np.array([
    -13.0, -7.0, 0.0,      # u c t
    -12.0, -9.0, -3.0,     # d s b
    -11.5, -6.0, -3.5      # e μ τ
])

ckm_exp = np.log10([0.225, 0.041, 0.0037])

labels_mass = ['u', 'c', 't', 'd', 's', 'b', 'e', 'μ', 'τ']
labels_ckm = ['Vus', 'Vcb', 'Vub']

# ============================================================
# МАТРИЦА L
# ============================================================

def L_matrix(eps, a12, a23, b13, phi):
    """Матрица с РАЗДЕЛЬНЫМИ a12, a23 (как в рабочей версии)"""
    return np.array([
        [eps**4,              a12*eps**3,    b13*eps**2*np.exp(1j*phi)],
        [a12*eps**3,          eps**2,        a23*eps],
        [b13*eps**2*np.exp(-1j*phi),  a23*eps,   1.0]
    ], dtype=complex)


def diag(L):
    L = (L + L.conj().T) / 2
    vals, vecs = eigh(L)
    idx = np.argsort(vals)
    return vals[idx], vecs[:, idx]


def safe_log(x, eps=1e-12):
    return np.log10(np.abs(x) + eps)


def CKM(Uu, Ud):
    return np.abs(Uu.conj().T @ Ud)


# ============================================================
# ФУНКЦИЯ ПОТЕРЬ (ИСПРАВЛЕННАЯ)
# ============================================================

def total_loss(params):
    """
    params: eps, a12_u, a12_d, a12_e, a23_u, a23_d, a23_e,
            b13_u, b13_d, b13_e, phi, delta_d, delta_e,
            c_u, c_d, c_e
    """
    (eps,
     a12_u, a12_d, a12_e,
     a23_u, a23_d, a23_e,
     b13_u, b13_d, b13_e,
     phi, delta_d, delta_e,
     c_u, c_d, c_e) = params

    # Матрицы секторов
    Lu = L_matrix(eps, a12_u, a23_u, b13_u, phi)
    Ld = L_matrix(eps, a12_d, a23_d, b13_d, phi + delta_d)
    Le = L_matrix(eps, a12_e, a23_e, b13_e, phi + delta_e)

    # Спектр
    mu, Uu = diag(Lu)
    md, Ud = diag(Ld)
    me, Ue = diag(Le)

    # Предсказанные массы
    d_pred = np.concatenate([
        [safe_log(mu[0]) + c_u],
        [safe_log(mu[1]) + c_u],
        [safe_log(mu[2])],
        [safe_log(md[0]) + c_d],
        [safe_log(md[1]) + c_d],
        [safe_log(md[2]) + c_d],
        [safe_log(me[0]) + c_e],
        [safe_log(me[1]) + c_e],
        [safe_log(me[2]) + c_e]
    ])

    # Потери масс
    loss_mass = np.mean((d_pred - d_exp) ** 2)

    # CKM (УСИЛЕННЫЙ штраф)
    V = CKM(Uu, Ud)
    ckm_vals = np.array([V[0, 1], V[1, 2], V[0, 2]])
    ckm_pred = safe_log(ckm_vals)

    # Асимметричный штраф: Vcb критически важен
    loss_ckm = (
        3.0 * (ckm_pred[0] - ckm_exp[0]) ** 2 +   # Vus
        8.0 * (ckm_pred[1] - ckm_exp[1]) ** 2 +   # Vcb — САМЫЙ ВАЖНЫЙ
        2.0 * (ckm_pred[2] - ckm_exp[2]) ** 2     # Vub
    )

    # Штрафы
    penalty_top = 5.0 * (safe_log(mu[2]) - 0.0) ** 2
    penalty_negative = 10.0 * (
        np.sum(np.clip(-mu, 0, None) ** 2) +
        np.sum(np.clip(-md, 0, None) ** 2) +
        np.sum(np.clip(-me, 0, None) ** 2)
    )

    # Иерархия масс (должна быть правильной)
    penalty_hierarchy = (
        0.5 * ((safe_log(mu[1]) - safe_log(mu[0])) - 6.0) ** 2 +   # c/u ~ 10^6
        0.5 * ((safe_log(md[1]) - safe_log(md[0])) - 3.0) ** 2 +   # s/d ~ 10^3
        0.3 * ((safe_log(me[1]) - safe_log(me[0])) - 5.5) ** 2     # μ/e ~ 10^5.5
    )

    # Структурный штраф (ОЧЕНЬ СЛАБЫЙ)
    penalty_structure = 0.001 * (
        (abs(a12_u) - 1.0) ** 2 + (abs(a12_d) - 1.0) ** 2 + (abs(a12_e) - 1.0) ** 2 +
        (abs(a23_u) - 1.0) ** 2 + (abs(a23_d) - 1.0) ** 2 + (abs(a23_e) - 1.0) ** 2
    )

    return loss_mass + loss_ckm + penalty_top + penalty_negative + penalty_hierarchy + penalty_structure


# ============================================================
# ОПТИМИЗАЦИЯ
# ============================================================

x0 = [
    0.22,                    # eps
    0.8, 0.5, 0.8,          # a12: u, d, e
    0.6, 0.9, 0.4,          # a23: u, d, e
    0.5, 0.3, 0.6,          # b13: u, d, e
    1.2,                     # phi
    0.3, 0.2,               # delta_d, delta_e
    -8.0, -8.0, -6.0        # c_u, c_d, c_e
]

bounds = [
    (0.15, 0.28),            # eps
    (0.2, 2.0), (0.2, 2.0), (0.2, 2.0),  # a12: u, d, e
    (0.2, 2.0), (0.2, 2.0), (0.2, 2.0),  # a23: u, d, e
    (0.1, 2.0), (0.1, 2.0), (0.1, 2.0),  # b13: u, d, e
    (0.5, 2.5),              # phi
    (0.05, 0.8), (0.05, 0.8),# delta_d, delta_e
    (-12.0, 2.0), (-12.0, 2.0), (-12.0, 2.0)  # c_u, c_d, c_e
]

print("╔══════════════════════════════════════════════════════════════╗")
print("║   ЕДИНАЯ МОДЕЛЬ FLAVOR — ИСПРАВЛЕННАЯ                     ║")
print("║   16 параметров → 9 масс + 3 угла CKM                     ║")
print("╚══════════════════════════════════════════════════════════════╝")

print(f"\nОптимизация...")
res = minimize(total_loss, x0, method='L-BFGS-B', bounds=bounds,
               options={'maxiter': 50000, 'ftol': 1e-12})

# ============================================================
# РЕЗУЛЬТАТЫ
# ============================================================

(eps, a12_u, a12_d, a12_e, a23_u, a23_d, a23_e,
 b13_u, b13_d, b13_e, phi, delta_d, delta_e,
 c_u, c_d, c_e) = res.x

Lu = L_matrix(eps, a12_u, a23_u, b13_u, phi)
Ld = L_matrix(eps, a12_d, a23_d, b13_d, phi + delta_d)
Le = L_matrix(eps, a12_e, a23_e, b13_e, phi + delta_e)

mu, Uu = diag(Lu)
md, Ud = diag(Ld)
me, Ue = diag(Le)

print(f"\n{'='*60}")
print(f"СТАТУС: {'✅ УСПЕХ' if res.success else '⚠️ НЕ СОШЛОСЬ'}")
print(f"LOSS: {res.fun:.4f}")
print(f"{'='*60}")

print(f"\n  ε = {eps:.4f}")
print(f"\n  Секторные параметры:")
print(f"  {'':>8} {'a12':>8} {'a23':>8} {'b13':>8}")
print(f"  {'up':>8} {a12_u:8.3f} {a23_u:8.3f} {b13_u:8.3f}")
print(f"  {'down':>8} {a12_d:8.3f} {a23_d:8.3f} {b13_d:8.3f}")
print(f"  {'lep':>8} {a12_e:8.3f} {a23_e:8.3f} {b13_e:8.3f}")

print(f"\n  МАССЫ:")
d_pred = np.array([
    safe_log(mu[0]) + c_u, safe_log(mu[1]) + c_u, safe_log(mu[2]),
    safe_log(md[0]) + c_d, safe_log(md[1]) + c_d, safe_log(md[2]) + c_d,
    safe_log(me[0]) + c_e, safe_log(me[1]) + c_e, safe_log(me[2]) + c_e
])

print(f"  {'Час-ца':>6} {'Предск':>8} {'Эксп':>8} {'Δ':>8}")
for lab, pred, exp in zip(labels_mass, d_pred, d_exp):
    diff = abs(pred - exp)
    status = "✓" if diff < 1.0 else ("⚠" if diff < 2.0 else "✗")
    print(f"  {lab:>6} {pred:8.2f} {exp:8.1f} {diff:8.2f} {status}")

print(f"\n  CKM:")
V = CKM(Uu, Ud)
ckm_vals = np.array([V[0, 1], V[1, 2], V[0, 2]])
angles = np.arcsin(ckm_vals) * 180 / np.pi
for lab, pred, exp, ang in zip(labels_ckm, ckm_vals, [0.225, 0.041, 0.0037], angles):
    print(f"  {lab} = {pred:.4f} (exp: {exp:.4f}) θ = {ang:.1f}°")

print(f"\n  СРЕДНЯЯ ОШИБКА МАСС: {np.mean(np.abs(d_pred - d_exp)):.2f} dex")