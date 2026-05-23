import math
import numpy as np
from math import log, sqrt, pi, e
from scipy.optimize import minimize

# ============================================================================
# ВХОДНЫЕ ПАРАМЕТРЫ
# ============================================================================
K = 8.0
p = 1.25e-31
N = 9.702e122
M_E_TARGET = 9.1093837e-31

# ============================================================================
# ВЫЧИСЛЕНИЕ БАЗОВЫХ ВЕЛИЧИН
# ============================================================================
lnN = log(N)
lnK = log(K)
lnKp = log(K * p)
abs_lnKp = abs(lnKp)

x = lnKp / lnN
lambda_sq = x ** 2
U = lnN / abs_lnKp
f1 = U / pi
f3 = sqrt(K * p)

C_clust = 3 * (K - 2) / (4 * (K - 1)) * (1 - p) ** 3
correction = 1 + (1 - C_clust) / lnN
hbar_em = (lnK ** 2) / (4 * lambda_sq ** 2 * K ** 2) * correction
V = f1 ** (2/3) * hbar_em ** 3 * lnN ** 2

base = M_E_TARGET * N ** (1/3)

print("=" * 80)
print("УТОЧНЕНИЕ ФОРМУЛЫ ДЛЯ МАССЫ ЭЛЕКТРОНА")
print("=" * 80)
print(f"\nЦелевое значение: m_e * N^(1/3) = {base:.6e}")

# ============================================================================
# ФУНКЦИЯ ДЛЯ ОПТИМИЗАЦИИ
# ============================================================================
def objective(params):
    coeff, a, b, c = params
    val = coeff * f3 * (U ** a) * (V ** b) * (lnK ** c)
    return abs(val - base) / base

# Уточняем параметры
result = minimize(
    objective, 
    [8.0, 6.0, 3.0, 1.0],  # начальные значения
    method='Nelder-Mead',
    options={'xatol': 1e-10, 'fatol': 1e-10}
)

coeff, a, b, c = result.x
val_opt = coeff * f3 * (U ** a) * (V ** b) * (lnK ** c)
error_opt = abs(val_opt - base) / base

print(f"\nУточнённые параметры:")
print(f"  coeff = {coeff:.6f}")
print(f"  a (степень U) = {a:.6f}")
print(f"  b (степень V) = {b:.6f}")
print(f"  c (степень lnK) = {c:.6f}")
print(f"  Значение = {val_opt:.6e}")
print(f"  Ошибка = {error_opt*100:.6f}%")

# ============================================================================
# ПРОВЕРКА С РАЗНЫМИ КОЭФФИЦИЕНТАМИ
# ============================================================================
print("\n" + "=" * 80)
print("ПРОВЕРКА С ТОЧНЫМИ КОЭФФИЦИЕНТАМИ")
print("=" * 80)

# Пробуем красивые коэффициенты
nice_coeffs = [6, 8, 10, 12, 4*pi, 8*pi, pi**2, 2*pi**2, 4*pi**2]
for c_val in nice_coeffs:
    val = c_val * f3 * (U ** a) * (V ** b) * (lnK ** c)
    error = abs(val - base) / base
    print(f"  {c_val:8.4f} : ошибка {error*100:.6f}%")

# ============================================================================
# ФИНАЛЬНАЯ ФОРМУЛА
# ============================================================================
print("\n" + "=" * 80)
print("ФИНАЛЬНАЯ ФОРМУЛА")
print("=" * 80)

# Выбираем лучший коэффициент
best_coeff = min(nice_coeffs, key=lambda c: abs(c * f3 * (U ** a) * (V ** b) * (lnK ** c) - base))
m_e_calc = best_coeff * f3 * (U ** a) * (V ** b) * (lnK ** c) * N ** (-1/3)

print(f"\nm_e = {best_coeff:.4f} · f3 · U^{a:.4f} · V^{b:.4f} · (lnK)^{c:.4f} · N^(-1/3)")
print(f"\nm_e (вычисленная) = {m_e_calc:.6e} кг")
print(f"m_e (CODATA)     = {M_E_TARGET:.6e} кг")
print(f"Отношение        = {m_e_calc / M_E_TARGET:.10f}")
print(f"Ошибка           = {abs(m_e_calc / M_E_TARGET - 1) * 100:.6f}%")

# ============================================================================
# УПРОЩЁННАЯ ФОРМУЛА (ЦЕЛЫЕ СТЕПЕНИ)
# ============================================================================
print("\n" + "=" * 80)
print("УПРОЩЁННАЯ ФОРМУЛА (ЦЕЛЫЕ СТЕПЕНИ)")
print("=" * 80)

a_int = round(a)
b_int = round(b)
c_int = round(c)

for coeff in [6, 8, 10, 12, 4*pi, 8*pi]:
    val = coeff * f3 * (U ** a_int) * (V ** b_int) * (lnK ** c_int)
    error = abs(val - base) / base
    if error < 0.01:
        m_e_simple = coeff * f3 * (U ** a_int) * (V ** b_int) * (lnK ** c_int) * N ** (-1/3)
        print(f"\ncoeff = {coeff:.4f}, U^{a_int}, V^{b_int}, (lnK)^{c_int}")
        print(f"m_e = {m_e_simple:.6e} кг")
        print(f"Ошибка = {error*100:.6f}%")
        break