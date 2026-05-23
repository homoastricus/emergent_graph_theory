"""
ПОИСК ФУНДАМЕНТАЛЬНЫХ ФОРМУЛ ДЛЯ ПЛАНКОВСКОЙ ДЛИНЫ И ВРЕМЕНИ
При p = 1.25e-31, K = 8, N = 9.702e122
"""

import numpy as np
import math
from math import log, sqrt, pi, e

# ВХОДНЫЕ ДАННЫЕ
K = 8.0
p = 1.25e-31
N = 9.702e122

# Вычисление базовых величин
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

C = 3 * (K - 2) / (4 * (K - 1)) * (1 - p) ** 3
correction = 1 + (1 - C) / lnN
hbar_em = (lnK ** 2) / (4 * lambda_sq ** 2 * K ** 2) * correction

# Планковские величины (в относительных единицах)
lp_rel = (2 * pi) / (K * p * lambda_sq) * N ** (-1/3)
tp_rel = lambda_sq ** 2 * hbar_em * N ** (-1/3) / pi

print("=" * 80)
print("ПЛАНКОВСКИЕ ВЕЛИЧИНЫ ПРИ p = 1.25e-31")
print("=" * 80)
print(f"\nБазовые величины:")
print(f"  λ = {lambda_sq:.6e}")
print(f"  hbar_em = {hbar_em:.6e}")
print(f"  N^(-1/3) = {N ** (-1/3):.6e}")
print(f"\nОтносительные планковские величины:")
print(f"  ℓ_P (отн) = {lp_rel:.6e}")
print(f"  t_P (отн) = {tp_rel:.6e}")

# РАСШИРЕННАЯ БИБЛИОТЕКА ВЕЛИЧИН
def safe_power(val, exponent, min_val=1e-100, max_val=1e100):
    try:
        if val <= 0 and exponent not in [0, 1, 2, 3]:
            return None
        result = val ** exponent
        if min_val < abs(result) < max_val:
            return result
        return None
    except:
        return None

base_quantities = {}

def add_quantity(name, val):
    if 1e-100 < abs(val) < 1e100:
        base_quantities[name] = val

add_quantity('f1', f1)
add_quantity('f2', f2)
add_quantity('f3', f3)
add_quantity('f5', f5)
add_quantity('f6', f6)
add_quantity('x', x)
add_quantity('x_abs', abs(x))
add_quantity('lambda', lambda_sq)
add_quantity('U', U)
add_quantity('S_loc', S_loc)
add_quantity('S_nonloc', S_nonloc)
add_quantity('S_glob', S_glob)
add_quantity('S_spec', S_spec)
add_quantity('lnK', lnK)
add_quantity('lnp', lnp)
add_quantity('lnN', lnN)
add_quantity('lnKp', lnKp)
add_quantity('abs_lnKp', abs(lnKp))
add_quantity('pi', pi)
add_quantity('e', e)
add_quantity('hbar_em', hbar_em)
add_quantity('C', C)
add_quantity('K', K)
add_quantity('p', p)
add_quantity('Kp', K * p)
add_quantity('N_1_3', N ** (-1/3))

# Добавляем дробные степени
exponents = [0.5, 1/3, 2/3, 1.5, -0.5, -1/3, -2/3, -1.5, 2.0, 3.0, -1.0, -2.0, -3.0]
exp_names = {
    0.5: '½', 1/3: '⅓', 2/3: '⅔', 1.5: '³⁄²',
    -0.5: '⁻½', -1/3: '⁻⅓', -2/3: '⁻⅔', -1.5: '⁻³⁄²',
    2.0: '²', 3.0: '³', -1.0: '⁻¹', -2.0: '⁻²', -3.0: '⁻³'
}

original_keys = list(base_quantities.keys())
for key in original_keys:
    val = base_quantities[key]
    if val > 0:
        for exp in exponents:
            if exp not in [1.0, -1.0]:
                powered = safe_power(val, exp)
                if powered is not None:
                    base_quantities[f"{key}{exp_names[exp]}"] = powered

print(f"Всего величин: {len(base_quantities)}")

# ПОИСК ДЛЯ ℓ_P
def check_match(value, target, tolerance=0.15):
    if target == 0:
        return abs(value) < tolerance
    if value == 0:
        return False
    try:
        return abs(value - target) / abs(target) < tolerance
    except:
        return False

quantities = list(base_quantities.items())

print("\n" + "=" * 80)
print(f"ПОИСК ФОРМУЛ ДЛЯ ℓ_P (target = {lp_rel:.6e})")
print("=" * 80)

lp_results = []

# Простые
for name, val in quantities:
    if check_match(val, lp_rel, 0.15):
        error = abs(val - lp_rel) / lp_rel * 100
        lp_results.append((name, val, error))

# Произведения двух
for i, (n1, v1) in enumerate(quantities):
    for j, (n2, v2) in enumerate(quantities):
        if i <= j:
            prod = v1 * v2
            if check_match(prod, lp_rel, 0.15):
                error = abs(prod - lp_rel) / lp_rel * 100
                lp_results.append((f"{n1}*{n2}", prod, error))

# Отношения
for n1, v1 in quantities:
    for n2, v2 in quantities:
        if v2 != 0:
            ratio = v1 / v2
            if check_match(ratio, lp_rel, 0.15):
                error = abs(ratio - lp_rel) / lp_rel * 100
                lp_results.append((f"{n1}/{n2}", ratio, error))

# Три величины: a * b / c
for (n1, v1) in quantities:
    for (n2, v2) in quantities:
        for (n3, v3) in quantities:
            if v3 != 0:
                val = v1 * v2 / v3
                if check_match(val, lp_rel, 0.15):
                    error = abs(val - lp_rel) / lp_rel * 100
                    lp_results.append((f"{n1}*{n2}/{n3}", val, error))

lp_results.sort(key=lambda x: x[2])

print(f"\nТОП-20 ДЛЯ ℓ_P:")
print(f"{'Формула':<50} {'Значение':<15} {'Ошибка %':<10}")
print("-" * 80)
for i, (formula, val, error) in enumerate(lp_results[:20]):
    print(f"{i+1:2d}. {formula:<47} {val:<15.6e} {error:<10.4f}%")

# ПОИСК ДЛЯ t_P

print("\n" + "=" * 80)
print(f"ПОИСК ФОРМУЛ ДЛЯ t_P (target = {tp_rel:.6e})")
print("=" * 80)

tp_results = []

for name, val in quantities:
    if check_match(val, tp_rel, 0.15):
        error = abs(val - tp_rel) / tp_rel * 100
        tp_results.append((name, val, error))

for i, (n1, v1) in enumerate(quantities):
    for j, (n2, v2) in enumerate(quantities):
        if i <= j:
            prod = v1 * v2
            if check_match(prod, tp_rel, 0.15):
                error = abs(prod - tp_rel) / tp_rel * 100
                tp_results.append((f"{n1}*{n2}", prod, error))

for n1, v1 in quantities:
    for n2, v2 in quantities:
        if v2 != 0:
            ratio = v1 / v2
            if check_match(ratio, tp_rel, 0.15):
                error = abs(ratio - tp_rel) / tp_rel * 100
                tp_results.append((f"{n1}/{n2}", ratio, error))

for (n1, v1) in quantities:
    for (n2, v2) in quantities:
        for (n3, v3) in quantities:
            if v3 != 0:
                val = v1 * v2 / v3
                if check_match(val, tp_rel, 0.15):
                    error = abs(val - tp_rel) / tp_rel * 100
                    tp_results.append((f"{n1}*{n2}/{n3}", val, error))

tp_results.sort(key=lambda x: x[2])

print(f"\nТОП-20 ДЛЯ t_P:")
print(f"{'Формула':<50} {'Значение':<15} {'Ошибка %':<10}")
print("-" * 80)
for i, (formula, val, error) in enumerate(tp_results[:20]):
    print(f"{i+1:2d}. {formula:<47} {val:<15.6e} {error:<10.4f}%")

# ФУНДАМЕНТАЛЬНЫЕ КАНДИДАТЫ
print("\n" + "=" * 80)
print("ФУНДАМЕНТАЛЬНЫЕ КАНДИДАТЫ")
print("=" * 80)

# Для ℓ_P ищем формулы, содержащие hbar_em, lambda, f1
print("\nДля ℓ_P (предпочтение формулам с λ, f1, S_glob):")
lp_fundamental = [r for r in lp_results if any(x in r[0] for x in ['lambda', 'f1', 'S_glob', 'U', 'hbar_em'])]
for i, (formula, val, error) in enumerate(lp_fundamental[:10]):
    print(f"  {i+1:2d}. {formula:<45} error={error:.4f}%")

print("\nДля t_P (предпочтение формулам с λ, hbar_em, S_glob):")
tp_fundamental = [r for r in tp_results if any(x in r[0] for x in ['lambda', 'hbar_em', 'S_glob', 'U', 'f1'])]
for i, (formula, val, error) in enumerate(tp_fundamental[:10]):
    print(f"  {i+1:2d}. {formula:<45} error={error:.4f}%")