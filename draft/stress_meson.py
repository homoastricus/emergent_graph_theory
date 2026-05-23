"""
МЕЗОНЫ 6.0 — ПОЛНЫЙ АНАЛИЗ
==========================
- считает δα, δq
- добавляет внутренние признаки
- ищет простые формулы
"""

import numpy as np
from itertools import product

# ============================================================
# ДАННЫЕ
# ============================================================

mesons = {
    'π0': (4, 8, 6, 1.0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 'meson'),
    'π+': (6, -5, 0, -2.0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 'meson'),
    'π-': (6, -5, 0, -2.0, 1, 1, 1, -1, 0, 0, 0, 1, 0, -1, 'meson'),
    'K+': (6, -3, 0, -1.5, 1, 1, 0.5, 0.5, 1, 0, 0, 1, 0, 1, 'meson'),
    'K0': (6, -3, 0, -1.5, 1, 1, 0.5, -0.5, 1, 0, 0, 1, 0, 0, 'meson'),
    'η': (5, 2, 0, 2.0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 'meson'),
    'η′': (5, 2, 6, -1.0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 'meson'),
    'φ': (5, 4, 3, 0.5, 1, 1, 0, 0, 0, 0, 0, 2, 1, 0, 'meson'),
    'ω': (5, 2, 0, 2.0, 1, 2, 0, 0, 0, 0, 0, 2, 1, 0, 'meson'),
    'J/ψ': (5, 2, 0, 2.0, 1, 8, 0, 0, 0, 0, 0, 2, 1, 0, 'meson'),
    'D0': (6, -3, -2, -0.5, 1, 1, 0.5, -0.5, 1, 0, 0, 1, 0, 0, 'meson'),
    'Υ(1S)': (6, -3, 0, 0.0, 1, 1, 0, 0, 0, 0, 0, 2, 1, 0, 'meson'),
}

# ============================================================
# ФИЧИ
# ============================================================

FEATURE_NAMES = [
    'ns',          # |S|
    'hidden_s',    # s sbar
    'mixing',      # η-like
    'is_vector',   # spin=1
    'g',           # поколение
    'p'            # составность
]

def compute_features(name, v):
    n_q, n_qbar = v[4], v[5]
    S = v[8]
    g = v[11]
    spin = v[12]

    p = (n_q + n_qbar) / 2
    ns = abs(S)

    # hidden strangeness
    hidden_s = 1 if name == 'φ' else 0

    # mixing
    mixing = 1 if name in ['η'] else 0

    # vector
    is_vector = 1 if spin == 1 else 0

    return np.array([ns, hidden_s, mixing, is_vector, g, p])


# ============================================================
# DATASET
# ============================================================

def build_dataset():
    X, y_alpha, y_q = [], [], []

    for name, v in mesons.items():
        feats = compute_features(name, v)

        alpha = v[1]
        q = v[3]
        g = v[11]

        p = feats[-1]

        delta_alpha = alpha - (-2 * p)
        delta_q = q - (p + g)

        X.append(feats)
        y_alpha.append(delta_alpha)
        y_q.append(delta_q)

    return np.array(X), np.array(y_alpha), np.array(y_q)


# ============================================================
# ОЦЕНКА
# ============================================================

def evaluate(X, y, coeffs):
    preds = X @ coeffs[:-1] + coeffs[-1]
    err = np.abs(preds - y)

    mae = np.mean(err)
    acc = np.mean(err < 1e-6)

    return acc, mae


# ============================================================
# ПОИСК
# ============================================================

def search(X, y, coef_range):
    best = []

    for coeffs in product(coef_range, repeat=X.shape[1] + 1):
        coeffs = np.array(coeffs)

        acc, mae = evaluate(X, y, coeffs)

        if acc >= 0.6:
            best.append((acc, mae, coeffs))

    best.sort(key=lambda x: (-x[0], x[1]))
    return best[:10]


# ============================================================
# ВЫВОД
# ============================================================

def print_models(models, target):
    print("\n" + "=" * 80)
    print(f"ЛУЧШИЕ МОДЕЛИ ДЛЯ {target}")
    print("=" * 80)

    for acc, mae, coeffs in models:
        terms = []

        for c, name in zip(coeffs[:-1], FEATURE_NAMES):
            if abs(c) > 1e-6:
                terms.append(f"{c}*{name}")

        formula = " + ".join(terms)
        if abs(coeffs[-1]) > 1e-6:
            formula += f" + {coeffs[-1]}"

        print(f"\nAccuracy: {acc:.2%} | MAE: {mae:.3f}")
        print(f"{target} = {formula}")


# ============================================================
# ДЕТАЛЬНЫЙ ВЫВОД
# ============================================================

def print_raw(X, y_alpha, y_q):
    print("\n" + "=" * 80)
    print("СЫРЫЕ δ-ЗНАЧЕНИЯ")
    print("=" * 80)

    for i, name in enumerate(mesons.keys()):
        feats = X[i]
        print(f"{name:3} | ns={feats[0]} hs={feats[1]} mix={feats[2]} vec={feats[3]} "
              f"| δα={y_alpha[i]:5.2f} | δq={y_q[i]:5.2f}")


# ============================================================
# ЗАПУСК
# ============================================================

print("=" * 80)
print("МЕЗОНЫ 6.0 — ПОЛНЫЙ АНАЛИЗ")
print("=" * 80)

X, y_alpha, y_q = build_dataset()

print_raw(X, y_alpha, y_q)

coef_range = np.arange(-3, 3.5, 0.5)

# Поиск формул
models_alpha = search(X, y_alpha, coef_range)
models_q = search(X, y_q, coef_range)

print_models(models_alpha, "delta_alpha")
print_models(models_q, "delta_q")