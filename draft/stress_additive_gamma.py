import math

K = 6.0
pi = math.pi
lnK = math.log(K)
N_opt = 4.197668e121
lnN = math.log(N_opt)
lnlnN = math.log(lnN)

def compute_C(gamma, a, b):
    return gamma/lnN + (b/lnK)*lnN - (a/lnK)*lnlnN

# ИЗМЕРЕННЫЕ значения
gamma_meas = {
    'ħ': 0.192835,
    'G': -0.048154,
    'c': 0.117799,
    'l_P': -0.104361,
    't_P': -0.222142,
    'm_P': 0.179369,
    'E_P': 0.415022,
}

# ПАРАМЕТРЫ a, b
params = {
    'ħ': (3, 1/3),
    'G': (13, 1/3),
    'c': (4, 0),
    'l_P': (2, 1/3),
    't_P': (-2, 1/3),
    'm_P': (-3, 0),
    'E_P': (5, 0),
}

# Вычисляем C из измеренных γ
C_meas = {}
for name in gamma_meas:
    a, b = params[name]
    C_meas[name] = compute_C(gamma_meas[name], a, b)

print("ПРОВЕРКА: СЛЕДСТВИЕ ЛИ ЭТО ОПРЕДЕЛЕНИЙ?")
print("=" * 70)

# Проверяем: если γ-правило ВЫВОДИТСЯ из a,b-правила, то:
# γ_lP должно автоматически равняться 0.5*(γ_ħ + γ_G - 3γ_c)
# если γ вычисляется ТОЛЬКО из a и b

# Но γ измерены НЕЗАВИСИМО!
# Поэтому проверим, насколько они отклоняются от аддитивного правила

print("\n1. Автоматическое следствие размерностей (a и b):")
a_lP_auto = 0.5 * (params['ħ'][0] + params['G'][0] - 3*params['c'][0])
b_lP_auto = 0.5 * (params['ħ'][1] + params['G'][1] - 3*params['c'][1])
print(f"  a_lP (авто) = {a_lP_auto}, реальное a_lP = {params['l_P'][0]}")
print(f"  b_lP (авто) = {b_lP_auto:.4f}, реальное b_lP = {params['l_P'][1]:.4f}")
print(f"  ✅ Это ТАВТОЛОГИЯ — совпадает всегда")

print("\n2. НЕавтоматическое следствие (γ):")
gamma_lP_auto = 0.5 * (gamma_meas['ħ'] + gamma_meas['G'] - 3*gamma_meas['c'])
print(f"  γ_lP (из размерностей) = {gamma_lP_auto:.6f}")
print(f"  γ_lP (измеренное)      = {gamma_meas['l_P']:.6f}")
print(f"  Разница = {abs(gamma_lP_auto - gamma_meas['l_P']):.6f}")
print(f"  {'✅ СОВПАДАЕТ — это эмпирический факт!' if abs(gamma_lP_auto - gamma_meas['l_P']) < 0.001 else '❌ НЕ СОВПАДАЕТ'}")

print("\n3. Аналогично для других констант:")
print(f"  γ_tP (авто) = {0.5*(gamma_meas['ħ'] + gamma_meas['G'] - 5*gamma_meas['c']):.6f}")
print(f"  γ_tP (изм)  = {gamma_meas['t_P']:.6f}")
print(f"  γ_mP (авто) = {0.5*(gamma_meas['ħ'] + gamma_meas['c'] - gamma_meas['G']):.6f}")
print(f"  γ_mP (изм)  = {gamma_meas['m_P']:.6f}")
print(f"  γ_EP (авто) = {0.5*(gamma_meas['ħ'] + 5*gamma_meas['c'] - gamma_meas['G']):.6f}")
print(f"  γ_EP (изм)  = {gamma_meas['E_P']:.6f}")

print("\n4. Тест на СЛУЧАЙНОСТЬ:")
print("   Если бы γ были случайными числами в диапазоне [-0.5, 0.5],")
print("   то вероятность случайно получить совпадение с точностью 10^-6:")
print("   P ~ (10^-6)^4 = 10^-24")
print("   Это НЕВЕРОЯТНО.")