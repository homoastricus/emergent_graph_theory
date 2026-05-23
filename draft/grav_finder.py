"""
ЧИСЛЕННАЯ ПРОВЕРКА ПЛАНКОВСКИХ ВЕЛИЧИН
========================================
Экстремально локальный режим: p = 1.25e-31, K = 8, N = 9.702e122
Сравнение с экспериментальными значениями (CODATA)
"""

import math
from math import log, sqrt, pi

# ============================================================================
# ВХОДНЫЕ ПАРАМЕТРЫ
# ============================================================================

K = 8.0
p = 1.25e-31
N = 9.702e122

# Экспериментальные значения (CODATA)
HBAR_TARGET = 1.054571817e-34      # Дж·с
LP_TARGET = 1.616255e-35           # м
TP_TARGET = 5.391247e-44           # с
C_TARGET = 2.99792458e8            # м/с

# ============================================================================
# ВЫЧИСЛЕНИЕ БАЗОВЫХ ВЕЛИЧИН
# ============================================================================

lnN = log(N)
lnK = log(K)
lnp = log(p)
lnKp = log(K * p)

x = lnKp / lnN
lambda_sq = x ** 2
U = lnN / abs(lnKp)

f1 = U / pi
f2 = lnK
f3 = sqrt(K * p)
f5 = K / lnK
f6 = 1.0 + p

S_loc = lnK
S_nonloc = -lnp
S_glob = lnN
S_spec = -2 * log(lambda_sq) if lambda_sq > 0 else 0

C_clust = 3 * (K - 2) / (4 * (K - 1)) * (1 - p) ** 3
correction = 1 + (1 - C_clust) / lnN

# Локальный квант действия (безразмерный)
hbar_em = (lnK ** 2) / (4 * lambda_sq ** 2 * K ** 2) * correction

# Голографический фактор
N_minus_1_3 = N ** (-1/3)

print("=" * 90)
print("БАЗОВЫЕ ВЕЛИЧИНЫ")
print("=" * 90)
print(f"K = {K}")
print(f"p = {p:.3e}")
print(f"N = {N:.3e}")
print(f"\nln N = {lnN:.6f}")
print(f"ln K = {lnK:.6f}")
print(f"ln p = {lnp:.6f}")
print(f"ln(Kp) = {lnKp:.6f}")
print(f"\nx = ln(Kp)/ln N = {x:.6f}")
print(f"λ = x² = {lambda_sq:.6e}")
print(f"U = ln N/|ln(Kp)| = {U:.6f}")
print(f"\nf1 = U/π = {f1:.6f}")
print(f"f2 = ln K = {f2:.6f}")
print(f"f3 = √(Kp) = {f3:.6e}")
print(f"f5 = K/ln K = {f5:.6f}")
print(f"\nS_loc = {S_loc:.6f}")
print(f"S_nonloc = {S_nonloc:.6f}")
print(f"S_glob = {S_glob:.6f}")
print(f"S_spec = {S_spec:.6f}")
print(f"\nC (кластеризация) = {C_clust:.6f}")
print(f"correction = {correction:.6f}")
print(f"hbar_em (безразмерный) = {hbar_em:.6e}")
print(f"N^(-1/3) = {N_minus_1_3:.6e}")

# ============================================================================
# ФОРМУЛЫ ДЛЯ ПЛАНКОВСКИХ ВЕЛИЧИН
# ============================================================================

print("\n" + "=" * 90)
print("ПЛАНКОВСКИЕ ВЕЛИЧИНЫ: СРАВНЕНИЕ ФОРМУЛ")
print("=" * 90)

# ----------------------------------------------------------------------------
# 1. ПОСТОЯННАЯ ПЛАНКА ℏ
# ----------------------------------------------------------------------------
print("\n1. ПОСТОЯННАЯ ПЛАНКА ℏ")
print("-" * 60)

# Формула 1: Исходная из модели
hbar_formula1 = hbar_em * N_minus_1_3 / (6 * pi)

# Формула 2: Новая фундаментальная (найденная перебором)
hbar_formula2 = f1 ** (2/3) * hbar_em ** 3 * S_glob ** 2 * N_minus_1_3

# Формула 3: Через S_nonloc
V = S_nonloc ** 4 / f5 ** (2/3)
hbar_formula3 = V * N_minus_1_3

print(f"Формула 1 (исходная):    ℏ = hbar_em * N^(-1/3) / (6π)")
print(f"  = {hbar_em:.6e} * {N_minus_1_3:.6e} / (6π)")
print(f"  = {hbar_formula1:.6e}")
print(f"  Отношение к цели: {hbar_formula1 / HBAR_TARGET:.6e}")

print(f"\nФормула 2 (фундаментальная): ℏ = f1^(2/3) * hbar_em^3 * S_glob^2 * N^(-1/3)")
print(f"  = {f1:.6f}^(2/3) * {hbar_em:.6e}^3 * {S_glob:.6f}^2 * {N_minus_1_3:.6e}")
print(f"  = {hbar_formula2:.6e}")
print(f"  Отношение к цели: {hbar_formula2 / HBAR_TARGET:.6e}")

print(f"\nФормула 3 (через V): ℏ = (S_nonloc^4 / f5^(2/3)) * N^(-1/3)")
print(f"  = ({S_nonloc:.6f}^4 / {f5:.6f}^(2/3)) * {N_minus_1_3:.6e}")
print(f"  = {hbar_formula3:.6e}")
print(f"  Отношение к цели: {hbar_formula3 / HBAR_TARGET:.6e}")

print(f"\nЦЕЛЕВОЕ ЗНАЧЕНИЕ: ℏ = {HBAR_TARGET:.6e}")

# ----------------------------------------------------------------------------
# 2. ПЛАНКОВСКАЯ ДЛИНА ℓ_P
# ----------------------------------------------------------------------------
print("\n2. ПЛАНКОВСКАЯ ДЛИНА ℓ_P")
print("-" * 60)

# Формула 1: Исходная из модели
lp_formula1 = (2 * pi) / (K * p * lambda_sq) * N_minus_1_3

# Формула 2: Найденная перебором
lp_formula2 = lambda_sq / (f5 ** (2/3) * S_glob ** 3)

# Формула 3: Альтернативная
lp_formula3 = (2 * pi) / (K * p) * U ** 2 * N_minus_1_3

print(f"Формула 1 (исходная):    ℓ_P = (2π) / (Kp * λ) * N^(-1/3)")
print(f"  = (2π) / ({K * p:.3e} * {lambda_sq:.6e}) * {N_minus_1_3:.6e}")
print(f"  = {lp_formula1:.6e}")
print(f"  Отношение к цели: {lp_formula1 / LP_TARGET:.6e}")

print(f"\nФормула 2 (фундаментальная): ℓ_P = λ / (f5^(2/3) * S_glob^3)")
print(f"  = {lambda_sq:.6e} / ({f5:.6f}^(2/3) * {S_glob:.6f}^3)")
print(f"  = {lp_formula2:.6e}")
print(f"  Отношение к цели: {lp_formula2 / LP_TARGET:.6e}")

print(f"\nФормула 3: ℓ_P = (2π) / (Kp) * U^2 * N^(-1/3)")
print(f"  = (2π) / {K * p:.3e} * {U:.6f}^2 * {N_minus_1_3:.6e}")
print(f"  = {lp_formula3:.6e}")
print(f"  Отношение к цели: {lp_formula3 / LP_TARGET:.6e}")

print(f"\nЦЕЛЕВОЕ ЗНАЧЕНИЕ: ℓ_P = {LP_TARGET:.6e}")

# ----------------------------------------------------------------------------
# 3. ПЛАНКОВСКОЕ ВРЕМЯ t_P
# ----------------------------------------------------------------------------
print("\n3. ПЛАНКОВСКОЕ ВРЕМЯ t_P")
print("-" * 60)

# Формула 1: Исходная из модели
tp_formula1 = lambda_sq ** 2 * hbar_em * N_minus_1_3 / pi

# Формула 2: Найденная перебором
tp_formula2 = N_minus_1_3 / (f1 ** 2 * hbar_em ** 3)

# Формула 3: Через λ
tp_formula3 = lambda_sq ** 2 * hbar_em * N_minus_1_3 / (2 * pi)

print(f"Формула 1 (исходная):    t_P = λ^2 * hbar_em * N^(-1/3) / π")
print(f"  = {lambda_sq:.6e}^2 * {hbar_em:.6e} * {N_minus_1_3:.6e} / π")
print(f"  = {tp_formula1:.6e}")
print(f"  Отношение к цели: {tp_formula1 / TP_TARGET:.6e}")

print(f"\nФормула 2 (фундаментальная): t_P = N^(-1/3) / (f1^2 * hbar_em^3)")
print(f"  = {N_minus_1_3:.6e} / ({f1:.6f}^2 * {hbar_em:.6e}^3)")
print(f"  = {tp_formula2:.6e}")
print(f"  Отношение к цели: {tp_formula2 / TP_TARGET:.6e}")

print(f"\nФормула 3: t_P = λ^2 * hbar_em * N^(-1/3) / (2π)")
print(f"  = {lambda_sq:.6e}^2 * {hbar_em:.6e} * {N_minus_1_3:.6e} / (2π)")
print(f"  = {tp_formula3:.6e}")
print(f"  Отношение к цели: {tp_formula3 / TP_TARGET:.6e}")

print(f"\nЦЕЛЕВОЕ ЗНАЧЕНИЕ: t_P = {TP_TARGET:.6e}")

# ----------------------------------------------------------------------------
# 4. СКОРОСТЬ СВЕТА c
# ----------------------------------------------------------------------------
print("\n4. СКОРОСТЬ СВЕТА c")
print("-" * 60)

# Из исходных формул
c_formula1 = lp_formula1 / tp_formula1
c_formula2 = lp_formula2 / tp_formula2
c_formula3 = lp_formula3 / tp_formula3

# Прямая формула (полученная ранее)
c_direct = (2 * pi ** 2) / (K * p * lambda_sq ** 3 * hbar_em)

print(f"Формула 1: c = ℓ_P / t_P (исходные)")
print(f"  = {lp_formula1:.6e} / {tp_formula1:.6e}")
print(f"  = {c_formula1:.6e}")
print(f"  Отношение к цели: {c_formula1 / C_TARGET:.6e}")

print(f"\nФормула 2: c = ℓ_P / t_P (фундаментальные)")
print(f"  = {lp_formula2:.6e} / {tp_formula2:.6e}")
print(f"  = {c_formula2:.6e}")
print(f"  Отношение к цели: {c_formula2 / C_TARGET:.6e}")

print(f"\nФормула 3: c = (2π²) / (Kp * λ³ * hbar_em)")
print(f"  = (2π²) / ({K * p:.3e} * {lambda_sq:.6e}^(3/2) * {hbar_em:.6e})")
print(f"  = {c_direct:.6e}")
print(f"  Отношение к цели: {c_direct / C_TARGET:.6e}")

print(f"\nЦЕЛЕВОЕ ЗНАЧЕНИЕ: c = {C_TARGET:.6e}")

# ============================================================================
# СВОДНАЯ ТАБЛИЦА
# ============================================================================

print("\n" + "=" * 90)
print("СВОДНАЯ ТАБЛИЦА: ОТНОШЕНИЕ К ЭКСПЕРИМЕНТУ")
print("=" * 90)

print(f"\n{'Величина':<20} {'Формула 1':<15} {'Формула 2':<15} {'Формула 3':<15}")
print("-" * 65)
print(f"{'ℏ':<20} {hbar_formula1 / HBAR_TARGET:<15.6e} {hbar_formula2 / HBAR_TARGET:<15.6e} {hbar_formula3 / HBAR_TARGET:<15.6e}")
print(f"{'ℓ_P':<20} {lp_formula1 / LP_TARGET:<15.6e} {lp_formula2 / LP_TARGET:<15.6e} {lp_formula3 / LP_TARGET:<15.6e}")
print(f"{'t_P':<20} {tp_formula1 / TP_TARGET:<15.6e} {tp_formula2 / TP_TARGET:<15.6e} {tp_formula3 / TP_TARGET:<15.6e}")
print(f"{'c':<20} {c_formula1 / C_TARGET:<15.6e} {c_formula2 / C_TARGET:<15.6e} {c_formula3 / C_TARGET:<15.6e}")

# ============================================================================
# ВЫВОДЫ
# ============================================================================

print("\n" + "=" * 90)
print("ВЫВОДЫ")
print("=" * 90)

# Определяем лучшую формулу для каждой величины
best_hbar = min([(1, abs(hbar_formula1/HBAR_TARGET - 1)),
                 (2, abs(hbar_formula2/HBAR_TARGET - 1)),
                 (3, abs(hbar_formula3/HBAR_TARGET - 1))], key=lambda x: x[1])

best_lp = min([(1, abs(lp_formula1/LP_TARGET - 1)),
               (2, abs(lp_formula2/LP_TARGET - 1)),
               (3, abs(lp_formula3/LP_TARGET - 1))], key=lambda x: x[1])

best_tp = min([(1, abs(tp_formula1/TP_TARGET - 1)),
               (2, abs(tp_formula2/TP_TARGET - 1)),
               (3, abs(tp_formula3/TP_TARGET - 1))], key=lambda x: x[1])

print(f"\nЛучшие формулы для каждой величины:")
print(f"  ℏ   : Формула {best_hbar[0]} (отклонение {best_hbar[1]*100:.4f}%)")
print(f"  ℓ_P : Формула {best_lp[0]} (отклонение {best_lp[1]*100:.4f}%)")
print(f"  t_P : Формула {best_tp[0]} (отклонение {best_tp[1]*100:.4f}%)")

# Проверка согласованности
print(f"\nПроверка согласованности:")
print(f"  ℓ_P / t_P = {lp_formula2 / tp_formula2:.6e}")
print(f"  c (цель)  = {C_TARGET:.6e}")
print(f"  Отношение = {(lp_formula2 / tp_formula2) / C_TARGET:.6e}")