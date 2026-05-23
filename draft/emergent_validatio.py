"""
ПРОВЕРКА СОГЛАСОВАННОСТИ С НОВОЙ ФОРМУЛОЙ ДЛЯ G
=================================================
Добавлена упрощённая формула: G = 512 * π^(4/3) * |ln(Kp)|^(58/3) / (lnN)^(67/3) * (K/lnK)^(29/3)
"""

import math
import numpy as np
from math import log, sqrt, pi, e
import matplotlib.pyplot as plt

# ============================================================================
# ЦЕЛЕВЫЕ ЗНАЧЕНИЯ (CODATA)
# ============================================================================
HBAR_TARGET = 1.054571817e-34
C_TARGET = 2.99792458e8
MP_TARGET = 2.176434e-8
ME_TARGET = 9.1093837e-31
LP_TARGET = 1.616255e-35
G_TARGET = 6.67430e-11

# Номинальные параметры
K = 8.0
p_nom = 1.25e-31
N_nom = 9.702e122


# ============================================================================
# ФУНКЦИЯ ВЫЧИСЛЕНИЯ ВСЕХ ЭМЕРДЖЕНТНЫХ ВЕЛИЧИН (С НОВОЙ ФОРМУЛОЙ G)
# ============================================================================
def compute_all_calibrated(p, N, use_new_G=True):
    """Вычисление с перекалибровкой alpha по hbar"""

    lnN = log(N)
    lnK = log(K)
    lnp = log(p)
    lnKp = log(K * p)
    abs_lnKp = abs(lnKp)

    x = lnKp / lnN
    lambda_sq = x ** 2
    U = lnN / abs_lnKp
    f1 = U / pi
    f3 = sqrt(K * p)
    f5 = K / lnK

    C_clust = 3 * (K - 2) / (4 * (K - 1)) * (1 - p) ** 3
    correction = 1 + (1 - C_clust) / lnN
    hbar_em = (lnK ** 2) / (4 * lambda_sq ** 2 * K ** 2) * correction
    V = f1 ** (2 / 3) * hbar_em ** 3 * lnN ** 2

    # КАЛИБРОВКА α ПО ℏ
    alpha = HBAR_TARGET / (V * N ** (-1 / 3))

    # Эмерджентные величины
    hbar = HBAR_TARGET  # точно по определению
    c = (lnN ** 2 * abs_lnKp ** 2) / (lnK ** (1 / 3))
    m_P = sqrt(alpha / pi) * e ** (1 / 6) * V ** (5 / 4) * lnN * abs_lnKp / (N ** (1 / 6) * lnK ** (1 / 6))
    m_e = 8 * f3 * (U ** 6) * (V ** 3) * lnK * N ** (-1 / 3)
    lp = (K * p) / (V ** (1 / 3) * lnN)

    # ДВЕ ВЕРСИИ G
    if use_new_G:
        # Новая упрощённая формула
        G = 512 * pi ** (4 / 3) * (abs_lnKp ** (58 / 3)) / (lnN ** (67 / 3)) * (K ** (29 / 3)) / (lnK ** (29 / 3))
    else:
        # Старая формула
        G = pi * e ** (-1 / 3) / (V ** (3 / 2))

    return {
        'hbar': hbar, 'c': c, 'm_P': m_P, 'm_e': m_e, 'lp': lp, 'G': G,
        'V': V, 'U': U, 'lambda': lambda_sq, 'alpha': alpha
    }


# ============================================================================
# ФУНКЦИЯ ОШИБКИ
# ============================================================================
def total_error_calibrated(p, N, use_new_G=True):
    res = compute_all_calibrated(p, N, use_new_G)

    errors = {
        'c': abs(res['c'] - C_TARGET) / C_TARGET,
        'm_P': abs(res['m_P'] - MP_TARGET) / MP_TARGET,
        'm_e': abs(res['m_e'] - ME_TARGET) / ME_TARGET,
        'lp': abs(res['lp'] - LP_TARGET) / LP_TARGET,
        'G': abs(res['G'] - G_TARGET) / G_TARGET,
    }

    total = sum(errors.values())
    return total, errors, res


# ============================================================================
# ПРОВЕРКА В НОМИНАЛЬНОЙ ТОЧКЕ (СРАВНЕНИЕ СТАРОЙ И НОВОЙ G)
# ============================================================================
print("=" * 90)
print("СРАВНЕНИЕ СТАРОЙ И НОВОЙ ФОРМУЛЫ ДЛЯ G В НОМИНАЛЬНОЙ ТОЧКЕ")
print("=" * 90)

res_old = compute_all_calibrated(p_nom, N_nom, use_new_G=False)
res_new = compute_all_calibrated(p_nom, N_nom, use_new_G=True)

print(f"\nНоминальные параметры: p = {p_nom:.6e}, N = {N_nom:.6e}")
print(f"\nСтарая G: {res_old['G']:.6e} (ошибка {abs(res_old['G'] - G_TARGET) / G_TARGET * 100:.6f}%)")
print(f"Новая G:   {res_new['G']:.6e} (ошибка {abs(res_new['G'] - G_TARGET) / G_TARGET * 100:.6f}%)")
print(f"G (CODATA): {G_TARGET:.6e}")

# Проверка, совпадают ли они
print(f"\nСовпадение старой и новой G: {abs(res_old['G'] - res_new['G']) < 1e-12}")

# ============================================================================
# СРАВНЕНИЕ ОШИБОК В НОМИНАЛЬНОЙ ТОЧКЕ
# ============================================================================
print("\n" + "=" * 90)
print("СРАВНЕНИЕ ОШИБОК В НОМИНАЛЬНОЙ ТОЧКЕ")
print("=" * 90)

total_old, errors_old, _ = total_error_calibrated(p_nom, N_nom, use_new_G=False)
total_new, errors_new, _ = total_error_calibrated(p_nom, N_nom, use_new_G=True)

print(f"\n{'Величина':<6} {'Старая G':<15} {'Новая G':<15} {'Разница':<10}")
print("-" * 50)
for k in ['c', 'm_P', 'm_e', 'lp', 'G']:
    old_val = errors_old[k] * 100
    new_val = errors_new[k] * 100
    diff = new_val - old_val
    status = "✅" if new_val < old_val else "❌"
    print(f"{k:<6} {old_val:<15.6f}% {new_val:<15.6f}% {diff:+.6f}% {status}")

print(f"\n{'Сумма':<6} {total_old:<15.6f} {total_new:<15.6f} {total_new - total_old:+.6f}")

# ============================================================================
# ДВУМЕРНОЕ СКАНИРОВАНИЕ С НОВОЙ G
# ============================================================================
print("\n" + "=" * 90)
print("ДВУМЕРНОЕ СКАНИРОВАНИЕ С НОВОЙ ФОРМУЛОЙ G")
print("=" * 90)

p_factors = np.linspace(0.95, 1.05, 11)
N_factors = np.linspace(0.95, 1.05, 11)

best_total = float('inf')
best_params = None

for p_f in p_factors:
    for N_f in N_factors:
        p_val = p_nom * p_f
        N_val = N_nom * N_f
        total, errors, res = total_error_calibrated(p_val, N_val, use_new_G=True)

        if total < best_total:
            best_total = total
            best_params = (p_val, N_val, p_f, N_f, errors, res)

print(f"\nГлобальный минимум (с новой G):")
print(f"  p = {best_params[0]:.6e} (фактор {best_params[2]:.4f})")
print(f"  N = {best_params[1]:.6e} (фактор {best_params[3]:.4f})")
print(f"  Суммарная ошибка = {best_total:.6f}")

print(f"\nОшибки в глобальном минимуме:")
for name, err in best_params[4].items():
    print(f"  {name:4}: {err * 100:.6f}%")

# ============================================================================
# СРАВНЕНИЕ С ПРЕДЫДУЩИМ ОПТИМУМОМ (СТАРАЯ G)
# ============================================================================
print("\n" + "=" * 90)
print("СРАВНЕНИЕ С ПРЕДЫДУЩИМ ОПТИМУМОМ (СТАРАЯ G)")
print("=" * 90)

# Запускаем поиск со старой G для сравнения
best_total_old = float('inf')
best_params_old = None

for p_f in p_factors:
    for N_f in N_factors:
        p_val = p_nom * p_f
        N_val = N_nom * N_f
        total, errors, res = total_error_calibrated(p_val, N_val, use_new_G=False)

        if total < best_total_old:
            best_total_old = total
            best_params_old = (p_val, N_val, p_f, N_f, errors, res)

print(f"\nСтарая G:")
print(f"  p = {best_params_old[0]:.6e} (фактор {best_params_old[2]:.4f})")
print(f"  N = {best_params_old[1]:.6e} (фактор {best_params_old[3]:.4f})")
print(f"  Суммарная ошибка = {best_total_old:.6f}")

print(f"\nНовая G:")
print(f"  p = {best_params[0]:.6e} (фактор {best_params[2]:.4f})")
print(f"  N = {best_params[1]:.6e} (фактор {best_params[3]:.4f})")
print(f"  Суммарная ошибка = {best_total:.6f}")

print(f"\nРазница в суммарной ошибке: {best_total - best_total_old:+.6f}")
if best_total < best_total_old:
    print("✅ Новая G даёт ЛУЧШЕЕ согласование!")
else:
    print("⚠️ Старая G даёт лучшее согласование")

# ============================================================================
# ПРОВЕРКА, СХОДЯТСЯ ЛИ МИНИМУМЫ В ОДНУ ТОЧКУ
# ============================================================================
print("\n" + "=" * 90)
print("ПРОВЕРКА СХОДИМОСТИ МИНИМУМОВ")
print("=" * 90)


# Проверяем, все ли величины имеют минимум в одной точке
def find_best_for_each(p_factors, N_factors, use_new_G):
    best_for_each = {}
    for name in ['c', 'm_P', 'm_e', 'lp', 'G']:
        best_val = float('inf')
        best_point = None
        for p_f in p_factors:
            for N_f in N_factors:
                p_val = p_nom * p_f
                N_val = N_nom * N_f
                _, errors, _ = total_error_calibrated(p_val, N_val, use_new_G)
                if errors[name] < best_val:
                    best_val = errors[name]
                    best_point = (p_f, N_f)
        best_for_each[name] = (best_point, best_val * 100)
    return best_for_each


print("\nМинимумы для каждой величины (новая G):")
best_each_new = find_best_for_each(p_factors, N_factors, True)
for name, ((pf, Nf), err) in best_each_new.items():
    print(f"  {name:4}: p_factor={pf:.4f}, N_factor={Nf:.4f}, ошибка={err:.6f}%")

print("\nМинимумы для каждой величины (старая G):")
best_each_old = find_best_for_each(p_factors, N_factors, False)
for name, ((pf, Nf), err) in best_each_old.items():
    print(f"  {name:4}: p_factor={pf:.4f}, N_factor={Nf:.4f}, ошибка={err:.6f}%")


# Проверяем, стали ли минимумы ближе друг к другу
def spread(best_each):
    p_factors = [v[0][0] for v in best_each.values()]
    N_factors = [v[0][1] for v in best_each.values()]
    return np.std(p_factors) + np.std(N_factors)


spread_new = spread(best_each_new)
spread_old = spread(best_each_old)

print(f"\nРазброс минимумов (p_factors):")
print(f"  Старая G: {spread_old:.6f}")
print(f"  Новая G:  {spread_new:.6f}")

if spread_new < spread_old:
    print("✅ Новая G УМЕНЬШАЕТ разброс — минимумы стали ближе друг к другу!")
else:
    print("⚠️ Старая G даёт меньший разброс")

# ============================================================================
# ИТОГ
# ============================================================================
print("\n" + "=" * 90)
print("ИТОГ")
print("=" * 90)

if best_total < best_total_old and spread_new < spread_old:
    print("\n🎉 НОВАЯ ФОРМУЛА G ПРЕВОСХОДНА!")
    print("   - Меньшая суммарная ошибка")
    print("   - Минимумы всех величин ближе друг к другу")
    print("   - Это подтверждает правильность упрощённой формулы!")
elif best_total < best_total_old:
    print("\n✅ Новая G даёт меньшую суммарную ошибку, но минимумы более разбросаны")
elif spread_new < spread_old:
    print("\n✅ Новая G улучшает согласованность минимумов")
else:
    print("\n⚠️ Старая G пока показывает лучшие результаты")