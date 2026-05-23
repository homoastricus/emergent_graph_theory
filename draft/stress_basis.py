import math

pi = math.pi
K = 6.0
lnK = math.log(K)
sqrt2 = math.sqrt(2)
sqrt3 = math.sqrt(3)
gamma_E = 0.5772156649015329

gamma_measured = {
    'h': 0.192835,
    'l_P': -0.104361, 't_P': -0.222142, 'm_P': 0.179369, 'E_P': 0.415022, 'T_P': -0.364637,
    'c': 0.117799, 'G': -0.048154, 'k_B': 0.224670, 'α': -0.015396,
    'm_e': 0.515264, 'm_muon': 0.061400, 'm_tau': 0.248402,
    'm_proton': -0.120978, 'm_neutron': -0.462800,
    'm_W': -0.100838, 'm_Z': 0.715684, 'm_Higgs': -0.267871,
    'm_pion': 0.127663, 'm_pion0': 0.271797, 'm_kaon0': -0.193698,
    'm_D0': -1.079562, 'm_J_psi': -0.385841, 'm_eta': -0.248487, 'm_Upsilon_1S': -1.179646,
    'm_quark_u': -0.682866, 'm_quark_d': 0.730184, 'm_quark_s': -1.474050,
    'm_quark_c': 0.195006, 'm_quark_b': -0.592949, 'm_quark_t': 0.120094,
}

# Базисные значения
basis_values = {
    '1/π': 1 / pi,
    'π': pi,
    '1/√π': 1 / math.sqrt(pi),
    #'1/√2π': 1 / math.sqrt(pi),
    '√π': math.sqrt(pi),
    '√π^3': math.sqrt(pi)**3,
    'π^(2/3)': pi**(2/3),
    '1/lnK': 1 / lnK,
    'lnK': lnK,
    '√2': sqrt2,
    '√3': sqrt3,
    '1/√2': 1 / sqrt2,
    '1/√3': 1 / sqrt3,
    'γ_E': gamma_E,
    'π/6': pi / 6,
    'π/4': pi / 4,
    'π/3': pi / 3,
    'π/2': pi / 2,
    'lnK/π': lnK / pi,
    'π/lnK': pi / lnK,
    '√π/K': math.sqrt(pi) / K,
    'K/π': K / pi,
    'lnK/K': lnK / K,
}

print("ПОИСК БЛИЖАЙШИХ БАЗИСНЫХ ЗНАЧЕНИЙ ДЛЯ γ_i")
print("=" * 80)
print(f"{'γ_i':>10} {'Ближайший базис':<20} {'Значение':>12} {'Отклонение':>12}")
print("-" * 60)

for name, gamma in gamma_measured.items():
    best_name = None
    best_val = None
    best_diff = float('inf')

    for basis_name, basis_val in basis_values.items():
        diff = abs(abs(gamma) - basis_val)
        if diff < best_diff:
            best_diff = diff
            best_name = basis_name
            best_val = basis_val

    # Проверяем, может ли gamma быть кратным базисному значению
    for basis_name, basis_val in basis_values.items():
        for mult in [1, 2, 3, 4, 6, 1 / 2, 1 / 3, 1 / 4, 1 / 6, 2 / 3, 3 / 2, 4 / 3]:
            candidate = mult * basis_val
            diff = abs(abs(gamma) - candidate)
            if diff < best_diff:
                best_diff = diff
                best_name = f"{mult}×{basis_name}"
                best_val = candidate

    sign = '+' if gamma >= 0 else '-'
    print(f"{sign}{abs(gamma):>9.6f} {best_name:<20} {best_val:>12.6f} {best_diff:>12.6f}")