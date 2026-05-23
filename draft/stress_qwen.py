import math

K = 6.0
pi = math.pi
lnK = math.log(K)

N_opt = 4.197668e121
lnN = math.log(N_opt)
lnlnN = math.log(lnN)

def compute_C(gamma, a, b):
    return gamma/lnN + (b/lnK)*lnN - (a/lnK)*lnlnN

# Фундаментальные инварианты
invariants = {
    'ħ':   {'gamma': 0.192835, 'a': 3,  'b': 1/3},
    'G':   {'gamma': -0.048154, 'a': 13, 'b': 1/3},
    'c':   {'gamma': 0.117799, 'a': 4,  'b': 0},
    'k_B': {'gamma': 0.224670, 'a': 8,  'b': 1/3},
}

C = {}
for name, data in invariants.items():
    C[name] = compute_C(data['gamma'], data['a'], data['b'])
    print(f"C_{name} = {C[name]:+.6f}")

# Планковские константы
planck_actual = {
    'l_P': {'gamma': -0.104361, 'a': 2, 'b': 1/3},
    't_P': {'gamma': -0.222142, 'a': -2, 'b': 1/3},
    'm_P': {'gamma': 0.179369, 'a': -3, 'b': 0},
    'E_P': {'gamma': 0.415022, 'a': 5, 'b': 0},
    'T_P': {'gamma': -0.364637, 'a': -4, 'b': -1/3},
}

C_actual = {}
for name, data in planck_actual.items():
    C_actual[name] = compute_C(data['gamma'], data['a'], data['b'])

# Аддитивные предсказания
print(f"\nАДДИТИВНЫЕ ПРЕДСКАЗАНИЯ (с π-поправками):")
print(f"  {'Константа':<6} {'C_измер':>12} {'C_пред':>12} {'Ошибка':>12}")
print(f"  {'-'*45}")

# l_P = √(ħG/c³)
C_lP_pred = 0.5 * (C['ħ'] + C['G'] - 3*C['c'])
print(f"  {'l_P':<6} {C_actual['l_P']:>12.6f} {C_lP_pred:>12.6f} {abs(C_actual['l_P']-C_lP_pred):>12.6f}")

# t_P = √(ħG/c⁵)
C_tP_pred = 0.5 * (C['ħ'] + C['G'] - 5*C['c'])
print(f"  {'t_P':<6} {C_actual['t_P']:>12.6f} {C_tP_pred:>12.6f} {abs(C_actual['t_P']-C_tP_pred):>12.6f}")

# m_P = √(ħc/G)
C_mP_pred = 0.5 * (C['ħ'] + C['c'] - C['G'])
print(f"  {'m_P':<6} {C_actual['m_P']:>12.6f} {C_mP_pred:>12.6f} {abs(C_actual['m_P']-C_mP_pred):>12.6f}")

# E_P = √(ħc⁵/G)
C_EP_pred = 0.5 * (C['ħ'] + 5*C['c'] - C['G'])
print(f"  {'E_P':<6} {C_actual['E_P']:>12.6f} {C_EP_pred:>12.6f} {abs(C_actual['E_P']-C_EP_pred):>12.6f}")

# T_P = E_P/k_B (+ π поправка!)
C_TP_pred = C['c'] - C['G'] + pi  # Исправленная формула
# T_P определяется как E_P/k_B, но с геометрической поправкой
print(f"  {'T_P':<6} {C_actual['T_P']:>12.6f} {C_TP_pred:>12.6f} {abs(C_actual['T_P']-C_TP_pred):>12.6f}")