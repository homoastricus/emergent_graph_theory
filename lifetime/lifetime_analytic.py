"""
ЕТИ: ПРОВЕРКА АНАЛИТИЧЕСКОЙ МОДЕЛИ ДЛЯ ЯДЕРНЫХ ВРЕМЁН ЖИЗНИ
Гипотезы:
  b = α_b · (N − Z) + β_b · Z + γ_b · (тип распада)
  c = α_c · N + β_c · Z + γ_c · (тип распада)
  a, d, e ≈ константы (или медленно меняющиеся функции)
"""

import math
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import cross_val_score, LeaveOneOut
import matplotlib.pyplot as plt
import seaborn as sns

# 1. ДАННЫЕ

# Формат: (Z, N, тип_распада, a, b, c, d, e)
# тип_распада: 0 = β⁻, 1 = β⁺, 2 = EC
data = [
    # Z, N, type, a, b, c, d, e
    (0, 1, 0, 2, 0, 0, 8, -8),  # n
    (1, 2, 0, 3, -2, -2, 8, 0),  # H3
    (2, 4, 0, 1, 0, 5, -1, -7),  # He6
    (2, 6, 0, 1, 7, -4, 2, -8),  # He8
    (4, 3, 2, 2, 5, -4, -4, 6),  # Be7
    (4, 6, 0, 4, 7, -4, 5, 5),  # Be10
    (4, 7, 0, 1, -5, 4, -4, -1),  # Be11
    (6, 4, 1, 1, 4, 0, -5, -1),  # C10
    (6, 5, 1, 1, -9, -1, -1, 5),  # C11
    (6, 8, 0, 4, -3, 4, -6, 5),  # C14
    (6, 9, 0, 2, 6, -6, 0, -8),  # C15
    (7, 6, 1, 3, -9, 4, -6, -5),  # N13
    (7, 9, 0, 4, -6, -17, 0, -8),  # N16
    (7, 10, 0, 3, -1, -15, -2, -5),  # N17
    (8, 6, 1, 2, -14, -7, -3, 3),  # O14
    (8, 7, 1, 1, 10, -2, 3, -4),  # O15
    (8, 11, 0, 3, 5, -8, -7, -6),  # O19
    (8, 12, 0, 1, -2, 4, 4, -6),  # O20
    (9, 9, 1, 3, 11, -19, 2, -2),  # F18
    (11, 11, 1, 4, -5, -13, -2, 5),  # Na22
    (11, 13, 0, 3, 9, -4, -6, -3),  # Na24
    (13, 13, 1, 1, -15, 52, 3, 0),  # Al26
    (15, 17, 0, 3, 17, -14, 6, -4),  # P32
    (16, 19, 0, 3, 3, -8, 0, 2),  # S35
    (17, 19, 0, 3, 20, 9, -4, 3),  # Cl36
    (18, 21, 0, 3, -23, 39, -5, -4),  # Ar39
    (19, 23, 0, 1, 42, -12, -3, -1),  # K42
    (20, 25, 0, 3, -10, -5, 4, 3),  # Ca45
    (25, 27, 1, 4, 7, -10, -7, 0),  # Mn52
    (25, 29, 2, 2, 29, -16, 0, 4),  # Mn54
    (26, 29, 2, 2, 14, 27, -8, -7),  # Fe55
    (26, 33, 0, 4, 25, -28, -7, 5),  # Fe59
    (27, 30, 2, 4, 1, 5, -7, -4),  # Co57
    (27, 33, 0, 2, -32, 28, -6, 6),  # Co60
    (28, 35, 0, 3, 10, 6, -5, 3),  # Ni63
    (29, 35, 0, 2, -11, -15, 6, 7),  # Cu64
    (30, 35, 2, 4, -6, 8, 2, -8),  # Zn65
    (38, 52, 0, 3, -39, 21, -2, 6),  # Sr90
    (55, 80, 0, 3, -39, 55, 3, -3),  # Cs135
    (55, 82, 0, 3, 80, -54, -2, 6),  # Cs137
    (53, 76, 0, 4, 78, -36, 3, 2),  # I129
    (62, 84, 0, 3, -29, 40, 2, 5),  # Sm146
    (94, 145, 0, 2, 94, -52, 5, 8),  # Pu239
    (92, 144, 0, 1, 142, -47, 5, 2),  # U236
]

data = np.array(data)
Z = data[:, 0]
N = data[:, 1]
A = Z + N
decay_type = data[:, 2]  # 0=β⁻, 1=β⁺, 2=EC
a_true = data[:, 3]
b_true = data[:, 4]
c_true = data[:, 5]
d_true = data[:, 6]
e_true = data[:, 7]

n_nuclei = len(data)
print(f"Загружено {n_nuclei} ядер")

# 2. АНАЛИТИЧЕСКАЯ МОДЕЛЬ ДЛЯ b
print("МОДЕЛЬ 1: b = α·(N−Z) + β·Z + γ·decay_type")

# Строим матрицу признаков
X_b = np.column_stack([N - Z, Z, decay_type])

# Линейная регрессия
reg_b = LinearRegression()
reg_b.fit(X_b, b_true)
b_pred = reg_b.predict(X_b)
r2_b = reg_b.score(X_b, b_true)

# Кросс-валидация
loo = LeaveOneOut()
scores_b = cross_val_score(reg_b, X_b, b_true, cv=loo, scoring='neg_mean_squared_error')
rmse_b = np.sqrt(-scores_b.mean())

print(f"\nРезультаты:")
print(f"  R² = {r2_b:.4f}")
print(f"  RMSE (LOO) = {rmse_b:.2f}")
print(f"  Коэффициенты:")
print(f"    α (N−Z) = {reg_b.coef_[0]:+.4f}")
print(f"    β (Z)   = {reg_b.coef_[1]:+.4f}")
print(f"    γ (decay) = {reg_b.coef_[2]:+.4f}")
print(f"  Intercept = {reg_b.intercept_:+.4f}")

# Статистическая значимость
for i, name in enumerate(['N−Z', 'Z', 'decay_type']):
    # Вычисляем t-статистику
    X_i = X_b[:, i]
    r, p = stats.pearsonr(X_i, b_true)
    print(f"  Корреляция b vs {name}: r = {r:+.4f}, p = {p:.6f}")

# График
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

axes[0].scatter(N - Z, b_true, c='steelblue', s=60, alpha=0.7, edgecolors='black')
axes[0].set_xlabel('N − Z')
axes[0].set_ylabel('b (√2)')
axes[0].set_title(f'b vs N−Z (r = {stats.pearsonr(N - Z, b_true)[0]:+.4f})')
axes[0].grid(True, alpha=0.3)

axes[1].scatter(Z, b_true, c='darkorange', s=60, alpha=0.7, edgecolors='black')
axes[1].set_xlabel('Z')
axes[1].set_ylabel('b (√2)')
axes[1].set_title(f'b vs Z (r = {stats.pearsonr(Z, b_true)[0]:+.4f})')
axes[1].grid(True, alpha=0.3)

axes[2].scatter(b_true, b_pred, c='green', s=60, alpha=0.7, edgecolors='black')
axes[2].plot([b_true.min(), b_true.max()], [b_true.min(), b_true.max()], 'r--')
axes[2].set_xlabel('Реальное b')
axes[2].set_ylabel('Предсказанное b')
axes[2].set_title(f'b: R² = {r2_b:.4f}, RMSE = {rmse_b:.2f}')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('eti_model_b.png', dpi=300)
print("\n✓ Сохранено: eti_model_b.png")

# 3. АНАЛИТИЧЕСКАЯ МОДЕЛЬ ДЛЯ c
print("МОДЕЛЬ 2: c = α·N + β·Z + γ·decay_type + δ·forbidden_flag")

# Добавляем признак запрещённости (|c| > 30 считаем запрещённым)
forbidden_flag = (np.abs(c_true) > 30).astype(float)

X_c = np.column_stack([N, Z, decay_type, forbidden_flag])

reg_c = LinearRegression()
reg_c.fit(X_c, c_true)
c_pred = reg_c.predict(X_c)
r2_c = reg_c.score(X_c, c_true)

scores_c = cross_val_score(reg_c, X_c, c_true, cv=loo, scoring='neg_mean_squared_error')
rmse_c = np.sqrt(-scores_c.mean())

print(f"\nРезультаты:")
print(f"  R² = {r2_c:.4f}")
print(f"  RMSE (LOO) = {rmse_c:.2f}")
print(f"  Коэффициенты:")
print(f"    α (N)       = {reg_c.coef_[0]:+.4f}")
print(f"    β (Z)       = {reg_c.coef_[1]:+.4f}")
print(f"    γ (decay)   = {reg_c.coef_[2]:+.4f}")
print(f"    δ (forb)    = {reg_c.coef_[3]:+.4f}")
print(f"  Intercept = {reg_c.intercept_:+.4f}")

for i, name in enumerate(['N', 'Z', 'decay_type', 'forbidden']):
    X_i = X_c[:, i]
    r, p = stats.pearsonr(X_i, c_true)
    print(f"  Корреляция c vs {name}: r = {r:+.4f}, p = {p:.6f}")

# График
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

axes[0].scatter(N, c_true, c='steelblue', s=60, alpha=0.7, edgecolors='black')
axes[0].set_xlabel('N')
axes[0].set_ylabel('c (√3)')
axes[0].set_title(f'c vs N (r = {stats.pearsonr(N, c_true)[0]:+.4f})')
axes[0].grid(True, alpha=0.3)

axes[1].scatter(Z, c_true, c='darkorange', s=60, alpha=0.7, edgecolors='black')
axes[1].set_xlabel('Z')
axes[1].set_ylabel('c (√3)')
axes[1].set_title(f'c vs Z (r = {stats.pearsonr(Z, c_true)[0]:+.4f})')
axes[1].grid(True, alpha=0.3)

axes[2].scatter(c_true, c_pred, c='green', s=60, alpha=0.7, edgecolors='black')
axes[2].plot([c_true.min(), c_true.max()], [c_true.min(), c_true.max()], 'r--')
axes[2].set_xlabel('Реальное c')
axes[2].set_ylabel('Предсказанное c')
axes[2].set_title(f'c: R² = {r2_c:.4f}, RMSE = {rmse_c:.2f}')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('eti_model_c.png', dpi=300)
print("\n✓ Сохранено: eti_model_c.png")

# 4. ПРЕДСКАЗАНИЕ ДЛЯ НОВЫХ ЯДЕР
print("ПРЕДСКАЗАНИЕ ДЛЯ НОВЫХ ЯДЕР")

# Новые ядра для проверки
new_nuclei = [
    # Z, N, decay_type, is_forbidden
    (19, 21, 0, 1),  # K40 (запрещённый β⁻)
    (37, 50, 0, 1),  # Rb87 (запрещённый β⁻)
    (90, 142, 0, 0),  # Th232
    (92, 146, 0, 0),  # U238
    (75, 112, 0, 1),  # Re187 (запрещённый)
    (71, 105, 0, 1),  # Lu176 (запрещённый)
    (6, 5, 1, 0),  # C11 (β⁺) — проверка
    (8, 7, 1, 0),  # O15 (β⁺) — проверка
]

# Экспериментальные значения (для сравнения)
experimental = {
    'K40': 3.938e16,
    'Rb87': 1.568e18,
    'Th232': 4.434e17,
    'U238': 1.410e17,
    'Re187': 1.294e18,
    'Lu176': 1.186e18,
    'C11': 1220.4,
    'O15': 176.4,
}

# Обучаем Ridge-модели для a, d, e (они слабо зависят от Z, N)
X_ad = np.column_stack([Z, N, A, decay_type])
ridge_a = Ridge(alpha=1.0).fit(X_ad, a_true)
ridge_d = Ridge(alpha=1.0).fit(X_ad, d_true)
ridge_e = Ridge(alpha=1.0).fit(X_ad, e_true)

# Параметры ЕТИ
lnN = 280.044221
lnK = math.log(6.0)
sqrt2 = math.sqrt(2)
sqrt3 = math.sqrt(3)
pi = math.pi

print(f"\n{'Ядро':8s} {'a':3s} {'b':5s} {'c':5s} {'d':3s} {'e':3s} | {'T_pred':>16s} | {'T_exp':>16s} | {'Ratio':8s}")

results = []
for Z_new, N_new, dtype_new, forb_new in new_nuclei:
    A_new = Z_new + N_new

    # Предсказываем b
    X_b_new = np.array([[N_new - Z_new, Z_new, dtype_new]])
    b_new = reg_b.predict(X_b_new)[0]

    # Предсказываем c
    X_c_new = np.array([[N_new, Z_new, dtype_new, forb_new]])
    c_new = reg_c.predict(X_c_new)[0]

    # Предсказываем a, d, e
    X_ad_new = np.array([[Z_new, N_new, A_new, dtype_new]])
    a_new = ridge_a.predict(X_ad_new)[0]
    d_new = ridge_d.predict(X_ad_new)[0]
    e_new = ridge_e.predict(X_ad_new)[0]

    # Округляем до целых (физическое ограничение)
    a_int = int(round(a_new))
    b_int = int(round(b_new))
    c_int = int(round(c_new))
    d_int = int(round(d_new))
    e_int = int(round(e_new))

    # Вычисляем период полураспада
    log_T = (a_int * math.log(lnN) +
             b_int * math.log(sqrt2) +
             c_int * math.log(sqrt3) +
             d_int * math.log(lnK) +
             e_int * math.log(pi))
    T_pred = math.exp(log_T)

    # Имя ядра
    elements = ['n', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
                'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
                'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
                'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
                'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
                'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
                'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
                'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
                'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
                'Pa', 'U', 'Np', 'Pu']

    name = f"{elements[Z_new]}{A_new}" if Z_new < len(elements) else f"Z={Z_new}"

    exp_val = experimental.get(name, float('nan'))
    if not np.isnan(exp_val):
        ratio = T_pred / exp_val
    else:
        ratio = float('nan')

    results.append((name, a_int, b_int, c_int, d_int, e_int, T_pred, exp_val, ratio))
    print(
        f"{name:8s} {a_int:+3d} {b_int:+5d} {c_int:+5d} {d_int:+3d} {e_int:+3d} | {T_pred:16.6e} | {exp_val:16.6e} | {ratio:8.4f}")

# 5. СВОДНАЯ ТАБЛИЦА КОЭФФИЦИЕНТОВ
print("ИТОГОВАЯ АНАЛИТИЧЕСКАЯ МОДЕЛЬ")

print(f"""
Показатели ЕТИ-формулы для ядерных времён жизни:

  b = {reg_b.coef_[0]:+.2f} · (N−Z) {reg_b.coef_[1]:+.2f} · Z {reg_b.coef_[2]:+.2f} · decay_type {reg_b.intercept_:+.2f}
  c = {reg_c.coef_[0]:+.2f} · N {reg_c.coef_[1]:+.2f} · Z {reg_c.coef_[2]:+.2f} · decay_type {reg_c.coef_[3]:+.2f} · forbidden {reg_c.intercept_:+.2f}
  a ≈ {a_true.mean():.1f} (константа, слабо зависит от ядра)
  d ≈ {d_true.mean():.1f} (константа, слабо зависит от ядра)
  e ≈ {e_true.mean():.1f} (константа, слабо зависит от ядра)

Качество модели:
  b: R² = {r2_b:.3f}, RMSE = {rmse_b:.1f}
  c: R² = {r2_c:.3f}, RMSE = {rmse_c:.1f}
""")

# 6. СРАВНЕНИЕ С ЭКСПЕРИМЕНТОМ
print("СРАВНЕНИЕ ПРЕДСКАЗАНИЙ С ЭКСПЕРИМЕНТОМ")

valid_results = [(name, pred, exp, ratio) for name, _, _, _, _, _, pred, exp, ratio in results if not np.isnan(exp)]
valid_results.sort(key=lambda x: abs(np.log10(x[3])))

print(f"\n{'Ядро':8s} {'Предсказание':>16s} {'Эксперимент':>16s} {'Отношение':>10s} {'log10(err)':>10s}")
print("-" * 70)
for name, pred, exp, ratio in valid_results:
    log_err = abs(np.log10(ratio))
    status = "✅" if log_err < 0.5 else ("⚠️" if log_err < 1.0 else "❌")
    print(f"{name:8s} {pred:16.6e} {exp:16.6e} {ratio:10.4f} {log_err:10.4f} {status}")

if valid_results:
    log_errors = [abs(np.log10(r[3])) for r in valid_results]
    print(f"\nСтатистика для {len(valid_results)} ядер:")
    print(f"  Средняя лог-ошибка: {np.mean(log_errors):.4f} dex")
    print(f"  Медианная лог-ошибка: {np.median(log_errors):.4f} dex")
    print(f"  Минимальная: {np.min(log_errors):.4f} dex")
    print(f"  Максимальная: {np.max(log_errors):.4f} dex")

plt.show()