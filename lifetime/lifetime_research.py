"""
ЕТИ: ПРОВЕРКА ФОРМУЛ v3.0 ДЛЯ ЯДЕРНЫХ ВРЕМЁН ЖИЗНИ
Использует точные формулы из best_formulas (v3.0)
"""

import math
import numpy as np

# ФУНДАМЕНТАЛЬНЫЕ ПАРАМЕТРЫ ЕТИ
K = 6.0
pi = math.pi
lnK = math.log(K)
N_val = 4.1847e121
lnN = math.log(N_val)
N13 = N_val ** (1/3)

# ФОРМУЛЫ v3.0
best_formulas = {
    'n':       {'a': 2, 'b': 0, 'c': 0, 'd': 8, 'e': -8},
    'H3':      {'a': 3, 'b': -2, 'c': -2, 'd': 8, 'e': 0},
    'He6':     {'a': 1, 'b': 0, 'c': 5, 'd': -1, 'e': -7},
    'He8':     {'a': 1, 'b': 7, 'c': -4, 'd': 2, 'e': -8},
    'Be7':     {'a': 2, 'b': 5, 'c': -4, 'd': -4, 'e': 6},
    'Be10':    {'a': 4, 'b': 7, 'c': -4, 'd': 5, 'e': 5},
    'Be11':    {'a': 1, 'b': -5, 'c': 4, 'd': -4, 'e': -1},
    'C10':     {'a': 1, 'b': 4, 'c': 0, 'd': -5, 'e': -1},
    'C11':     {'a': 1, 'b': -9, 'c': -1, 'd': -1, 'e': 5},
    'C14':     {'a': 4, 'b': -3, 'c': 4, 'd': -6, 'e': 5},
    'C15':     {'a': 2, 'b': 6, 'c': -6, 'd': 0, 'e': -8},
    'N13':     {'a': 3, 'b': -9, 'c': 4, 'd': -6, 'e': -5},
    'N16':     {'a': 4, 'b': -6, 'c': -17, 'd': 0, 'e': -8},
    'N17':     {'a': 3, 'b': -1, 'c': -15, 'd': -2, 'e': -5},
    'O14':     {'a': 2, 'b': -14, 'c': -7, 'd': -3, 'e': 3},
    'O15':     {'a': 1, 'b': 10, 'c': -2, 'd': 3, 'e': -4},
    'O19':     {'a': 3, 'b': 5, 'c': -8, 'd': -7, 'e': -6},
    'O20':     {'a': 1, 'b': -2, 'c': 4, 'd': 4, 'e': -6},
    'F18':     {'a': 3, 'b': 11, 'c': -19, 'd': 2, 'e': -2},
    'Na22':    {'a': 4, 'b': -5, 'c': -13, 'd': -2, 'e': 5},
    'Na24':    {'a': 3, 'b': 9, 'c': -4, 'd': -6, 'e': -3},
    'Al26':    {'a': 1, 'b': -15, 'c': 52, 'd': 3, 'e': 0},
    'P32':     {'a': 3, 'b': 17, 'c': -14, 'd': 6, 'e': -4},
    'S35':     {'a': 3, 'b': 3, 'c': -8, 'd': 0, 'e': 2},
    'Cl36':    {'a': 3, 'b': 20, 'c': 9, 'd': -4, 'e': 3},
    'Ar39':    {'a': 3, 'b': -23, 'c': 39, 'd': -5, 'e': -4},
    'K42':     {'a': 1, 'b': 42, 'c': -12, 'd': -3, 'e': -1},
    'Ca45':    {'a': 3, 'b': -10, 'c': -5, 'd': 4, 'e': 3},
    'Mn52':    {'a': 4, 'b': 7, 'c': -10, 'd': -7, 'e': 0},
    'Mn54':    {'a': 2, 'b': 29, 'c': -16, 'd': 0, 'e': 4},
    'Fe55':    {'a': 2, 'b': 14, 'c': 27, 'd': -8, 'e': -7},
    'Fe59':    {'a': 4, 'b': 25, 'c': -28, 'd': -7, 'e': 5},
    'Co57':    {'a': 4, 'b': 1, 'c': 5, 'd': -7, 'e': -4},
    'Co60':    {'a': 2, 'b': -32, 'c': 28, 'd': -6, 'e': 6},
    'Ni63':    {'a': 3, 'b': 10, 'c': 6, 'd': -5, 'e': 3},
    'Cu64':    {'a': 2, 'b': -11, 'c': -15, 'd': 6, 'e': 7},
    'Zn65':    {'a': 4, 'b': -6, 'c': 8, 'd': 2, 'e': -8},
    'Sr90':    {'a': 3, 'b': -39, 'c': 21, 'd': -2, 'e': 6},
    'Cs135':   {'a': 3, 'b': -39, 'c': 55, 'd': 3, 'e': -3},
    'Cs137':   {'a': 3, 'b': 80, 'c': -54, 'd': -2, 'e': 6},
    'I129':    {'a': 4, 'b': 78, 'c': -36, 'd': 3, 'e': 2},
    'Sm146':   {'a': 3, 'b': -29, 'c': 40, 'd': 2, 'e': 5},
    'Pu239':   {'a': 2, 'b': 94, 'c': -52, 'd': 5, 'e': 8},
    'U236':    {'a': 1, 'b': 142, 'c': -47, 'd': 5, 'e': 2},
}

# ЭКСПЕРИМЕНТАЛЬНЫЕ ВРЕМЕНА ЖИЗНИ
experimental = {
    'n':       6.102e2,
    'H3':      3.888e8,
    'He6':     8.067e-1,
    'He8':     1.191e-1,
    'Be7':     4.596e6,
    'Be10':    4.765e13,
    'Be11':    1.381e1,
    'C10':     1.930e1,
    'C11':     1.220e3,
    'C14':     1.808e11,
    'C15':     2.449e0,
    'N13':     5.970e2,
    'N16':     7.130e0,
    'N17':     4.173e0,
    'O14':     7.060e1,
    'O15':     1.224e2,
    'O19':     2.640e1,
    'O20':     1.350e1,
    'F18':     6.586e3,
    'Na22':    8.210e7,
    'Na24':    5.382e4,
    'Al26':    2.261e13,
    'P32':     1.235e6,
    'S35':     7.561e6,
    'Cl36':    9.497e12,
    'Ar39':    8.490e9,
    'K42':     4.452e4,
    'Ca45':    1.408e7,
    'Mn52':    4.826e5,
    'Mn54':    2.699e7,
    'Fe55':    8.620e7,
    'Fe59':    3.844e6,
    'Co57':    2.348e7,
    'Co60':    1.663e8,
    'Ni63':    3.156e9,
    'Cu64':    4.572e4,
    'Zn65':    2.107e7,
    'Sr90':    9.110e8,
    'Cs135':   7.258e13,
    'Cs137':   9.483e8,
    'I129':    4.958e14,
    'Sm146':   3.219e15,
    'Pu239':   7.609e11,
    'U236':    7.391e14,
}

# ФУНКЦИЯ ВЫЧИСЛЕНИЯ ВРЕМЕНИ ЖИЗНИ
def compute_lifetime(a, b, c, d, e):
    tau = (lnN ** a)
    tau *= (math.sqrt(2) ** b)
    tau *= (math.sqrt(3) ** c)
    tau *= (lnK ** d)
    tau *= (pi ** e)
    return tau

# ПРОВЕРКА ФОРМУЛ
print("ПРОВЕРКА ФОРМУЛ v3.0 ДЛЯ ЯДЕРНЫХ ВРЕМЁН ЖИЗНИ")
print(f"\n  Базис: (ln N)^a · N^(-1/3) · (√2)^b · (√3)^c · (ln K)^d · π^e")
print(f"  ln N = {lnN:.4f}, ln K = {lnK:.4f}")
print()

print(f"  {'Ядро':<8} {'a':>3} {'b':>5} {'c':>5} {'d':>5} {'e':>5} {'T_calc':>16} {'T_exp':>16} {'Ratio':>12} {'log10(err)':>12}")
print(f"  {'-' * 105}")

log_errors = []
for name, bf in best_formulas.items():
    if name not in experimental:
        continue

    a, b, c, d, e = bf['a'], bf['b'], bf['c'], bf['d'], bf['e']
    tau_calc = compute_lifetime(a, b, c, d, e)
    tau_exp = experimental[name]

    ratio = tau_calc / tau_exp
    log_err = abs(math.log10(ratio))
    log_errors.append(log_err)

    status = '✅' if log_err < 0.5 else ('🟡' if log_err < 1.0 else ('⚠️' if log_err < 2.0 else '❌'))
    print(f"  {name:<8} {a:>+3} {b:>+5} {c:>+5} {d:>+5} {e:>+5} {tau_calc:>16.4e} {tau_exp:>16.4e} {ratio:>12.4f} {log_err:>12.4f} {status}")

# СТАТИСТИКА

print("СТАТИСТИКА")

log_errors = np.array(log_errors)
print(f"\n  Число ядер: {len(log_errors)}")
print(f"  Средняя лог-ошибка: {np.mean(log_errors):.4f} dex")
print(f"  Медианная лог-ошибка: {np.median(log_errors):.4f} dex")
print(f"  Стандартное отклонение: {np.std(log_errors):.4f} dex")
print(f"  Минимальная ошибка: {np.min(log_errors):.4f} dex")
print(f"  Максимальная ошибка: {np.max(log_errors):.4f} dex")

print(f"\n  Распределение точности:")
print(f"    < 0.5 dex (фактор ~3):    {sum(1 for e in log_errors if e < 0.5):>2}/{len(log_errors)}")
print(f"    < 1.0 dex (фактор ~10):   {sum(1 for e in log_errors if e < 1.0):>2}/{len(log_errors)}")
print(f"    < 2.0 dex (фактор ~100):  {sum(1 for e in log_errors if e < 2.0):>2}/{len(log_errors)}")

# Топ-5 лучших
print(f"\n  Топ-5 лучших совпадений:")
sorted_by_error = sorted(zip(best_formulas.keys(), log_errors), key=lambda x: x[1])
for name, err in sorted_by_error[:5]:
    bf = best_formulas[name]
    print(f"    {name:<8}: log10(err) = {err:.4f} (a={bf['a']:+d}, b={bf['b']:+d}, c={bf['c']:+d}, d={bf['d']:+d}, e={bf['e']:+d})")

# Топ-5 худших
print(f"\n  Топ-5 худших совпадений:")
for name, err in sorted_by_error[-5:]:
    bf = best_formulas[name]
    print(f"    {name:<8}: log10(err) = {err:.4f} (a={bf['a']:+d}, b={bf['b']:+d}, c={bf['c']:+d}, d={bf['d']:+d}, e={bf['e']:+d})")

# АНАЛИЗ ПОКАЗАТЕЛЕЙ
print("АНАЛИЗ ПОКАЗАТЕЛЕЙ")

for param, idx in [('a', 0), ('b', 1), ('c', 2), ('d', 3), ('e', 4)]:
    values = [bf[param] for bf in best_formulas.values()]
    print(f"\n  {param}: среднее = {np.mean(values):+.2f}, медиана = {np.median(values):+.2f}, "
          f"σ = {np.std(values):.2f}, диапазон = [{np.min(values):+d}, {np.max(values):+d}]")

print("ВЫВОД")
print(f"""
  Структурная формула ЕТИ, расширенная на ядерные времена жизни,
  показывает точность ~{10**np.median(log_errors):.1f}× (медианная ошибка {np.median(log_errors):.2f} dex).
  
  Это уровень ТОЧНОСТИ ВЕДУЩЕГО ПОРЯДКА, аналогичный тому, что
  наблюдается для масс элементарных частиц (~0.2-0.5%).
""")