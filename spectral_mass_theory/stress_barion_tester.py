import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures

baryons = {
    # Октет (S=1/2)
    'p': {'d': 0.5, 'S': 0.5, 'I': 0.5, 'Y': 1, 'ns': 0, 'sym': 0, 'heavy': 0},
    'n': {'d': 0.5, 'S': 0.5, 'I': 0.5, 'Y': 1, 'ns': 0, 'sym': 0, 'heavy': 0},
    'Λ': {'d': -2.0, 'S': 0.5, 'I': 0.0, 'Y': 0, 'ns': 1, 'sym': 0, 'heavy': 0},
    'Σ+': {'d': -2.0, 'S': 0.5, 'I': 1.0, 'Y': 0, 'ns': 1, 'sym': 0, 'heavy': 0},
    'Ξ0': {'d': -0.5, 'S': 0.5, 'I': 0.5, 'Y': -1, 'ns': 2, 'sym': 0, 'heavy': 0},

    # Декуплет (S=3/2)
    'Ω-': {'d': 1.0, 'S': 1.5, 'I': 0.0, 'Y': -2, 'ns': 3, 'sym': 1, 'heavy': 0},
    'Δ': {'d': -1.0, 'S': 1.5, 'I': 1.5, 'Y': 1, 'ns': 0, 'sym': 1, 'heavy': 0},

    # Тяжёлые барионы (c, b)
    'Λ+_c': {'d': 1.0, 'S': 0.5, 'I': 0.0, 'Y': 1, 'ns': 0, 'sym': 0, 'heavy': 1},
    'Ξ+_c': {'d': -1.0, 'S': 0.5, 'I': 0.5, 'Y': 0, 'ns': 1, 'sym': 0, 'heavy': 1},
    'Ω0_c': {'d': -2.5, 'S': 0.5, 'I': 0.0, 'Y': -1, 'ns': 2, 'sym': 0, 'heavy': 1},
    'Λ0_b': {'d': 0.5, 'S': 0.5, 'I': 0.0, 'Y': 0, 'ns': 0, 'sym': 0, 'heavy': 1},
    'Ξ++_b': {'d': 0.0, 'S': 0.5, 'I': 0.5, 'Y': 1, 'ns': 0, 'sym': 0, 'heavy': 1},
}

# =========================
# ФУНКЦИИ
# =========================

def sigma_sigma(S):
    return -3 if S == 0.5 else 3

def is_diquark(name):
    return 1 if name.startswith('Λ') else 0

def is_strange(name):
    return 1 if 'Ω' in name else 0

# =========================
# DATAFRAME
# =========================

rows = []
for name, data in baryons.items():
    S = data['S']
    I = data['I']

    rows.append({
        'name': name,
        'd': data['d'],
        'S2': S*(S+1),
        'I2': I*(I+1),
        'Y': data['Y'],
        'sigma': sigma_sigma(S),
        'ns': data['ns'],
        'sym': data['sym'],
        'heavy': data['heavy'],
        'diquark': is_diquark(name),
        'strange_flag': is_strange(name),
    })

df = pd.DataFrame(rows)

# Квантовые фичи
features = ['S2','I2','Y','sigma','ns','sym','heavy','diquark','strange_flag']
X = df[features]
y = df['d']

# НЕЛИНЕЙНОСТЬ
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

feature_names = poly.get_feature_names_out(features)

# РЕГУЛЯРИЗАЦИЯ

model = Ridge(alpha=1.0)
model.fit(X_poly, y)

y_pred = model.predict(X_poly)

# МЕТРИКИ
print("\nКАЧЕСТВО")
print("R²  =", r2_score(y, y_pred))
print("MAE =", mean_absolute_error(y, y_pred))

# ТОП-ВАЖНЫЕ ЧЛЕНЫ
coefs = pd.Series(model.coef_, index=feature_names)
important = coefs.abs().sort_values(ascending=False).head(15)

print("\nТОП-15 ВАЖНЫХ ЧЛЕНОВ:")
for name in important.index:
    print(f"{name:25s} {coefs[name]:+.4f}")

# СРАВНЕНИЕ

df['pred'] = y_pred
df['err'] = df['pred'] - df['d']

print("\nСРАВНЕНИЕ:")
print(df[['name','d','pred','err']].to_string(index=False))