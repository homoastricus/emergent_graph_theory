"""
ФИНАЛЬНАЯ МОДЕЛЬ МЕЗОНОВ — ВСЕ ИСПРАВЛЕНИЯ
============================================
1. η/η' разделены (eta_prime)
2. Добавлена эффективная масса кварка (mq_eff)
3. Сектора исправлены (S=0 goldstone, S=1 light, heavy>0)
4. Взаимодействия: mq_eff * sigma, mq_eff * ns, eta_prime * sigma
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures

# =========================
# ДАННЫЕ МЕЗОНОВ
# =========================

mesons = {
    # ГОЛДСТОУНЫ (S=0, heavy=0)
    'π0':   {'d': 1.0,  'S': 0, 'Y': 0,  'ns': 0, 'heavy': 0, 'charge': 0, 'eta_prime': 0},
    'π±':   {'d': -2.0, 'S': 0, 'Y': 0,  'ns': 0, 'heavy': 0, 'charge': 1, 'eta_prime': 0},
    'K0':   {'d': -1.5, 'S': 0, 'Y': 1,  'ns': 1, 'heavy': 0, 'charge': 0, 'eta_prime': 0},
    'η':    {'d': 2.0,  'S': 0, 'Y': 0,  'ns': 1, 'heavy': 0, 'charge': 0, 'eta_prime': 0},
    "η'":   {'d': -1.0, 'S': 0, 'Y': 0,  'ns': 1, 'heavy': 0, 'charge': 0, 'eta_prime': 1},

    # ЛЁГКИЕ ВЕКТОРНЫЕ (S=1, heavy=0)
    'ρ':    {'d': 2.5,  'S': 1, 'Y': 0,  'ns': 0, 'heavy': 0, 'charge': 1, 'eta_prime': 0},
    'ω':    {'d': 2.0,  'S': 1, 'Y': 0,  'ns': 0, 'heavy': 0, 'charge': 0, 'eta_prime': 0},
    'φ':    {'d': 0.5,  'S': 1, 'Y': 0,  'ns': 2, 'heavy': 0, 'charge': 0, 'eta_prime': 0},
    'K*':   {'d': -2.5, 'S': 1, 'Y': 1,  'ns': 1, 'heavy': 0, 'charge': 1, 'eta_prime': 0},

    # ТЯЖЁЛЫЕ (heavy > 0)
    'J/ψ':  {'d': 2.0,  'S': 1, 'Y': 0,  'ns': 0, 'heavy': 1, 'charge': 0, 'eta_prime': 0, 'heavy_pair': 1, 'heavy_light': 0},
    'η_c':  {'d': 0.0,  'S': 0, 'Y': 0,  'ns': 0, 'heavy': 1, 'charge': 0, 'eta_prime': 0, 'heavy_pair': 1, 'heavy_light': 0},
    'Υ':    {'d': 0.0,  'S': 1, 'Y': 0,  'ns': 0, 'heavy': 2, 'charge': 0, 'eta_prime': 0, 'heavy_pair': 1, 'heavy_light': 0},
    'D0':   {'d': 0.5,  'S': 1, 'Y': 0,  'ns': 0, 'heavy': 1, 'charge': 0, 'eta_prime': 0, 'heavy_pair': 0, 'heavy_light': 1},
    'B':    {'d': 2.0,  'S': 1, 'Y': 0,  'ns': 0, 'heavy': 2, 'charge': 0, 'eta_prime': 0, 'heavy_pair': 0, 'heavy_light': 1},
    'B_s':  {'d': 0.0,  'S': 1, 'Y': 0,  'ns': 1, 'heavy': 2, 'charge': 0, 'eta_prime': 0, 'heavy_pair': 0, 'heavy_light': 1},
}

# =========================
# ФУНКЦИИ
# =========================

def sigma_mes(S):
    return -3 if S == 0 else 1

# =========================
# DATAFRAME
# =========================

rows = []
for name, data in mesons.items():
    S = data['S']
    heavy = data['heavy']

    # Эффективная масса кварка
    if heavy == 0:
        mq_eff = 0.0
    elif 'heavy_pair' in data and data['heavy_pair'] == 1:
        # QQ̄: масса пары ~ 2*m_Q
        mq_eff = 2.0 if heavy == 1 else 4.5  # c или b
    elif 'heavy_light' in data and data['heavy_light'] == 1:
        # Qq̄: масса тяжёлого кварка
        mq_eff = 1.3 if heavy == 1 else 4.5
    else:
        mq_eff = float(heavy)

    row = {
        'name': name,
        'd': data['d'],
        'S': S,
        'sigma': sigma_mes(S),
        'Y': data['Y'],
        'ns': data['ns'],
        'heavy': heavy,
        'charge': data.get('charge', 0),
        'eta_prime': data.get('eta_prime', 0),
        'heavy_pair': data.get('heavy_pair', 0),
        'heavy_light': data.get('heavy_light', 0),
        'mq_eff': mq_eff,
        # Сектора (исправленные)
        'sector_goldstone': 1 if (S == 0 and heavy == 0) else 0,
        'sector_light': 1 if (S == 1 and heavy == 0) else 0,
        'sector_heavy': 1 if heavy > 0 else 0,
    }
    rows.append(row)

df = pd.DataFrame(rows)

# =========================
# ИНЖЕНЕРИЯ ПРИЗНАКОВ
# =========================

# Секторальные взаимодействия
df['goldstone_charge'] = df['sector_goldstone'] * df['charge']
df['goldstone_ns'] = df['sector_goldstone'] * df['ns']
df['goldstone_sigma'] = df['sector_goldstone'] * df['sigma']

df['light_ns'] = df['sector_light'] * df['ns']
df['light_charge'] = df['sector_light'] * df['charge']
df['light_sigma'] = df['sector_light'] * df['sigma']

df['heavy_sigma'] = df['sector_heavy'] * df['sigma']
df['heavy_ns'] = df['sector_heavy'] * df['ns']
df['heavy_pair_sigma'] = df['heavy_pair'] * df['sigma']
df['heavy_light_sigma'] = df['heavy_light'] * df['sigma']

# Взаимодействия с массой кварка
df['mq_eff_sigma'] = df['mq_eff'] * df['sigma']
df['mq_eff_ns'] = df['mq_eff'] * df['ns']

# η/η′ взаимодействия
df['eta_prime_sigma'] = df['eta_prime'] * df['sigma']
df['eta_prime_ns'] = df['eta_prime'] * df['ns']

# =========================
# ФИЧИ (ФИНАЛЬНЫЙ НАБОР)
# =========================

features = [
    # Базовые
    'sigma', 'Y', 'ns', 'charge',

    # Секторальные взаимодействия
    'goldstone_charge', 'goldstone_ns', 'goldstone_sigma',
    'light_ns', 'light_charge', 'light_sigma',
    'heavy_sigma', 'heavy_ns',

    # Тяжёлые подтипы
    'heavy_pair_sigma', 'heavy_light_sigma',

    # Эффективная масса кварка
    'mq_eff', 'mq_eff_sigma', 'mq_eff_ns',

    # η/η′
    'eta_prime', 'eta_prime_sigma', 'eta_prime_ns',
]

X = df[features]
y = df['d']

# =========================
# МОДЕЛЬ
# =========================

poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X)
feature_names = poly.get_feature_names_out(features)

print("=" * 90)
print("ФИНАЛЬНАЯ МОДЕЛЬ: ВСЕ ИСПРАВЛЕНИЯ")
print("=" * 90)
print(f"  Признаков: {len(features)} → после poly: {len(feature_names)}")
print(f"  Объектов: {len(df)}")

model = Ridge(alpha=0.3)
model.fit(X_poly, y)
y_pred = model.predict(X_poly)

# =========================
# МЕТРИКИ
# =========================

r2 = r2_score(y, y_pred)
mae = mean_absolute_error(y, y_pred)

print(f"\n  R² = {r2:.4f}")
print(f"  MAE = {mae:.4f}")

# Топ-15 признаков
coefs = pd.Series(model.coef_, index=feature_names)
top15 = coefs.abs().sort_values(ascending=False).head(15)

print(f"\n  ТОП-15 ПРИЗНАКОВ:")
for name in top15.index:
    bar = '█' * int(abs(coefs[name]) / abs(coefs).max() * 25)
    print(f"    {name:<45s} {coefs[name]:+8.4f}  {bar}")

# =========================
# СРАВНЕНИЕ ПО СЕКТОРАМ
# =========================

df['pred'] = y_pred
df['err'] = df['pred'] - df['d']

print(f"\n{'='*90}")
print("СРАВНЕНИЕ ПО СЕКТОРАМ")
print("=" * 90)

for sector_name, sector_mask in [
    ('ГОЛДСТОУНЫ (S=0)', df['sector_goldstone'] == 1),
    ('ЛЁГКИЕ (S=1)', df['sector_light'] == 1),
    ('ТЯЖЁЛЫЕ', df['sector_heavy'] == 1),
]:
    df_s = df[sector_mask]
    if len(df_s) > 0:
        r2_s = r2_score(df_s['d'], df_s['pred'])
        mae_s = mean_absolute_error(df_s['d'], df_s['pred'])
        print(f"\n  {sector_name} (n={len(df_s)}): R² = {r2_s:.4f}, MAE = {mae_s:.4f}")
        print(f"  {'Имя':<6} {'d':>6} {'Пред':>10} {'Ош':>10}")
        print(f"  {'-'*34}")
        for _, row in df_s.iterrows():
            mark = ' ★' if abs(row['err']) < 0.15 else ''
            prefix = '⚠ ' if abs(row['err']) > 0.4 else '  '
            print(f"  {prefix}{row['name']:<4} {row['d']:>+6.1f} {row['pred']:>+10.3f} {row['err']:>+10.3f}{mark}")

# =========================
# ВЫВОД
# =========================

print("ВЫВОД")
# Ключевые признаки
goldstone_features = [f for f in top15.index if 'goldstone' in f.lower() or 'eta_prime' in f.lower()]
heavy_features = [f for f in top15.index if 'mq_eff' in f.lower() or 'heavy' in f.lower()]
light_features = [f for f in top15.index if 'light' in f.lower()]

print(f"""
  ОБЩАЯ МЕТРИКА:
    R² = {r2:.4f}
    MAE = {mae:.4f}
  
  КЛЮЧЕВЫЕ ПРИЗНАКИ (топ-15):
    Голдстоуны: {len(goldstone_features)} признаков
    Тяжёлые:    {len(heavy_features)} признаков
    Лёгкие:     {len(light_features)} признаков
""")