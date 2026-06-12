"""
НУМЕРОЛОГИЧЕСКИЙ ПОИСК ПЛАНКОВСКИХ ВЕЛИЧИН ИЗ ПАРАМЕТРОВ ГИП
ИСПРАВЛЕННАЯ ВЕРСИЯ: ГАРАНТИРОВАННОЕ НАХОЖДЕНИЕ H_TEST
"""
import itertools

import math

# 1. БАЗОВЫЕ ПАРАМЕТРЫ ГИП
K = 6.0
p = 4.802764507914655e-42  # 3.2403939e-40#1.8130170251248464e-40 #3.39e-40
# N = 4.475947e121 #0.3576e122
N_math = 4.475947e+121
N_phys = math.log(N_math) - math.pi * K / math.log(N_math)
N = math.exp(N_phys)
print(N_phys)
Lambda_GIP = 5.183093e-02
p_t = N ** (-1 / 3) / K

# Производные величины
Kp = K * p
lnK = math.log(K)
lnKp = math.log(Kp)
lnN = math.log(N)
lnKp_abs = abs(lnKp)
print("={p_t}".format(p_t=p_t))
# print(f"{math.log(N)/math.log(K*p_t):.12e}")

# ТЕСТОВЫЕ ФОРМУЛЫ
H_TEST = (N ** (-1 / 3) * (lnN ** 3)) / K
H_TEST_2 = lnN / (Kp * N ** (1 / 3))

print("=" * 80)
print("🔬 ТЕСТОВЫЕ ФОРМУЛЫ (должны быть найдены):")
print("=" * 80)
print(f"   H_TEST  = N^(-1/3) * (lnN)^3 / K = {H_TEST:.12e}")
print(f"   H_TEST / ħ_SI = {H_TEST / 1.054571817e-34:.6f}")
print(f"   H_TEST_2 = lnN / (Kp * N^(1/3)) = {H_TEST_2:.12e}")
p_fix = K ** 3 * lnK / (2 * (math.pi ** 3)) * N ** (-1 / 3)

val_test = 2 ** (1 / 2) / (lnK * N ** (1 / 3))
print(f"   val_test = {val_test:.12e}")
print(f"   p_fix = {p_fix:.12e}")
# p = p_fix
# 2. ФУНДАМЕНТАЛЬНЫЕ КОНСТАНТЫ (ЦЕЛЕВЫЕ ЗНАЧЕНИЯ)

targets = {
    # 'ħ (Дж·с)': 1.054571817e-34,
    # 'h (Дж·с)': 6.62607015e-34,
    # 'l_P (м)': 1.616255e-35,
    # 't_P (с)': 5.391247e-44,
    # 'm_P (кг)': 2.176434e-8,
    # 'E_P (Дж)': 1.956082e9,
    # 'T_P (К)': 1.416784e32,
    # 'T_B':  1.380649e-23
    # 'E_P': 9.109384e-31
    # 'G (м³/(кг·с²))': 6.67430e-11,
    # 'c (м/с)': 299792458,
    # 'ω': 1.85487e34,
    # 'ε₀':8.8541878128e-12,
    # 'μ0': 1.25663706127e-6,
    # 'q_e': 1.60217663e-19,
    # 'm_proton': 1.67262192e-27,
    # 'm_muon': 1.883531627e-28,
    # 'm_tau_lepton': 3.167e-27,
    # 'pion+': 2.4880888e-28,
    # 'pion0': 2.4060e-28
    # 'Kaon+': 8.8019e-28,
    # 'm_neutron': 1.67492749804e-27,
    # 'm_DT': 3.34358377241e-27
    # 'm_Λ_barion':	1.9901611e-27,
    # 'Z-бозон':1.62614e-25,
    # 'W-бозон':1.433620e-25
    # 'm_Higgs':2.23319e-25 #higgs
    # 'm_D0': 3.32479e-27
    # 'm_J/ψ': 5.52061e-27,
    # 'm_eta': 9.767732e-28,
    # 'm_Υ_1S': 1.68715e-26,
    # 'm_qw_u': 2.1650e-30,
    # 'm_qw_d': 4.7915e-30,
     'm_qw_s': 1.66e-28,
    # 'm_qw_c': 1.27e-27,
    # 'm_qw_b': 4.180e-27,
    # 'm_qw_t': 3.00e-25
    # 'GF': 1.436e-62,
    # 'Lambda_cosmo': 1.08929e-52
    # 'vacuum_higgs': 4.388471e-25
    # 'mu_lifetime': 2.1969811e-6,
    # 'tau_lifetime': 2.903e-13
    # 'pion_lifetime': 2.6033e-8
    #'neutron_lifetime': 877.8,
     #'tritiy_lilfetime': 3.89e8,
     #'C_14_lifetime': 1.81e11,
    #'C_11_lifetime': 1220.4,
    #'Be7': 4598208,
    #'Be10': 4.37e13,
    # 'He_6': 1.163,
    # 'Li_8': 1.208,
    # 'kaon': 1.2380e-8,
    # 'D+_lifetime': 1.040e-12,
    # 'B+_ligetime': 1.638e-12
    # 'B+_lifetime': 1.471e-12,
    # 'D0_lifetime': 4.101e-13
    # 'h2_connection_energy': 2.178872e-18
    # 'm_neitrino': 1.783e-36
    # 'Σ+': 2.11933e-27,
    # 'Ξ0-': 2.34532e-27,
    # 'omega_minus': 2.9859e-27
    # 'Lambda+': 4.0737e-27
    # 'Sigma_aver': 4.367e-27
    # Ksi+': 4.3995e-27
    # "omega_zero": 4.8080e-27,
    #'lambda_B0': 1.0023e-26
    # 'Ksi-': 2.3580e-27,
    # 'Σ−': 2.1328e-27,
    # 'phi_meson': 1.8195e-27
    # 'omega_meson': 1.394e-27
    # 'eta_shtrih': 1.7086e-27
    # 'hydrogen': 3.347e-27
    # 'pi_null_meson': 2.4063446e-28
    # 'moment_e':  1159652180.73e-12
    # 'oxygen': 2.656e-26
    # 'helium_4': 6.646e-27
    # 'berily': 1.229e-26
    # 'litiy': 1.165e-26
    # 'carbon': 1.992e-26
    # 'nitro': 2.324e-26
    # 'po': 1.4900e-27
    # 'K*': 1.5900e-27
    # 'B': 9.4000e-27
    # 'eta_c': 5.319e-27
    # 'h_c-мезон': 6.285e-27
    # 'delta_barion': 2.196e-27
    # 'B_c': 1.1185e-26
    # 'B_s': 9.567e-26
    # 'Ksi++': 6.453e-27
    # 't_hagedorn': 11604.5
    # 'T_relict': 2.72548
    # 'global': 8.058027287347709e-39
    # 'tail': math.pi-3
    # 'B0': 9.413e-27,
    # 'geometric_zeta': 0.06228841681507902,
    # 'geometric_alpha': 0.03660105378020262,
    # 'geometric_phys': 0.06729304209414977,
    # 'zeta_alpha': 0.025687363034876398,
    # 'zeta_phys': 0.005004625279070751,
    # 'alpha_phys': 0.03069198831394715,
    # 'dzetta_2': 1.6449340668482264,
    # 'm_plank_diff': 2.5405355802828732e-11
    # 'sigma_plus_lifetime': 8.018e-11
    # 'avogadro':6.02214076e23
}

# =========================================================
# 3. БАЗОВЫЕ КОМПОНЕНТЫ (РАСШИРЕННЫЙ НАБОР)
# =========================================================

# ВАЖНО: явно добавляем (lnN)^3 и другие степени
components = {
    # Основные параметры
    'K': K,
    # 'p': p,
    'N': N,
    # 'Kp': Kp,

    # Логарифмы
    # 'lnK': lnK,
    'lnN': lnN,
    # 'lnKp': lnKp,
    # 'lnKp_abs': lnKp_abs,

    # ЯВНО ДОБАВЛЯЕМ СТЕПЕНИ lnN
    # '(lnN)^(p+1)': lnN**(1+p),
    '(lnN)^2': lnN ** 2,
    '(lnN)^3': lnN ** 3,
    '(lnN)^4': lnN ** 4,
    '(lnN)^5': lnN ** 5,
    '(lnN)^6': lnN ** 6,
    # '(lnN)^7': lnN**7,
    # '(lnN)^8': lnN**8,
    # '(lnN)^9': lnN**9,
    # '(lnN)^10': lnN**10,
    # '(lnN)^11': lnN**11,
    # '(lnN)^12': lnN**12,
    # '(lnN)^13': lnN**13,
    # '(lnN)^14': lnN**14,
    # '(lnN)^15': lnN**15,
    # '(lnN)^16': lnN**16,
    # '(lnN)^17': lnN**17,
    # '(lnN)^18': lnN**18,
    # '(lnN)^19': lnN**19,
    # '(lnN)^20': lnN**20,
    # '(lnN)^21': lnN**21,
    # '(lnN)^22': lnN**22,
    # '(lnN)^23': lnN**23,
    # '(lnN)^31/2': lnN**(31/2),
    # '1/(lnN)': 1 / (lnN),
    # '1/(lnN)^2': 1 / (lnN ** 2),
    # '1/(lnN)^3': 1 / (lnN ** 3),
    # '1/(lnN)^4': 1 / (lnN ** 4),
    #     '1/(lnN)^5': 1/(lnN ** 5),
    #     '1/(lnN)^6': 1/(lnN ** 6),
    # '1/(lnN)^7': 1/(lnN ** 7),
    # '1/(lnN)^8': 1/(lnN ** 8),
    # '1/(lnN)^9': 1/(lnN ** 9),
    # '1/(lnN)^10': 1/(lnN ** 10),
    # '1/(lnN)^11': 1/(lnN ** 11),
    # '1/(lnN)^12': 1/(lnN ** 12),

    # '1/(lnN)^12': 1/(lnN ** 12),
    # '1/(lnN)^13': 1/(lnN ** 13),

    # Добавить в components:

    # 'π^3': math.pi ** (3),
    # '4π²': 4 * math.pi ** 2,
    # '1/(4π²)': 1 / (4 * math.pi ** 2),
    # '2π²': 2 * math.pi ** 2,
    # '1/(2π²)': 1 / (2 * math.pi ** 2),
    # '8π²': 8 * math.pi ** 2,
    # '1/(8π²)': 1 / (8 * math.pi ** 2),
    '√(π)': math.sqrt(math.pi),
    '1/√(π)': 1 / math.sqrt(math.pi),
    # '√(2π)': math.sqrt(2 * math.pi),
    # '1/√(2π)': 1 / math.sqrt(2 * math.pi),

    'π': math.pi,
    # '2π': 2 * math.pi,
    '1/π': 1 / math.pi,
    # '1/(2π)': 1 / (2 * math.pi),
    'π²': math.pi ** 2,
    'π^3': math.pi ** 3,
     'π^4': math.pi ** 4,
    # 'π^5': math.pi ** 5,
    # 'π^6': math.pi ** 6,

    '1/π²': 1 / math.pi ** 2,
    '1/π^3': 1 / math.pi ** 3,
     '1/π^4': 1 / math.pi ** 4,
    # '1/π^5': 1 / math.pi ** 5,
    # '1/π^6': 1 / math.pi ** 6,

    'π^3/2': math.pi ** (3 / 2),
    # 'π^5/2': math.pi ** (5 / 2),
    # 'π^7/2': math.pi ** (7 / 2),
    # 'π^9/2': math.pi ** (9 / 2),
    # 'π^11/2': math.pi ** (11 / 2),
    '1/π^3/2': 1 / math.pi ** (3 / 2),
    # '1/π^5/2': 1 / math.pi ** (5 / 2),
    # '1/π^7/2': 1 / math.pi ** (7 / 2),
    # '1/π^9/2': 1 / math.pi ** (9 / 2),
    # '1/π^11/2': 1 / math.pi ** (11 / 2),

    # '1/π^6': 1 / math.pi**6,
    # '1/2': 1 / 2,
    # '1/3': 1 / 3,
    # '3/2': 3 / 2,
    # '2/3': 2 / 3,
    # '2': 2 / 1,
    # '4': 4,
    # '2√2': 2*math.sqrt(2),
    # '3√2': 3*math.sqrt(2),
    # '2√3': 2*math.sqrt(3),
    # '3√3': 3*math.sqrt(3),
    # '1/3√2': 1/(3 * math.sqrt(2)),
    # '1/2√3': 1/(2 * math.sqrt(3)),
    # '1/4': 1/4,
    #  '√2': math.sqrt(2),
    #  '√3': math.sqrt(3),
    #  '1/√2': 1/math.sqrt(2),
    #  '1/√3': 1/math.sqrt(3),
    # '2/√3': 2/math.sqrt(3),
    # '√3/2': math.sqrt(3)/2,
    # '√3/3': math.sqrt(3)/3,
    # '√2/2': math.sqrt(2)/2,

    # ЯВНО ДОБАВЛЯЕМ СТЕПЕНИ N
    # 'N^(-1/4)': N**(-1/4),
    # 'N^(1/4)': N**(1/4),
    'N^(1/3)': N ** (1 / 3),
    'N^(-1/3)': N ** (-1 / 3),
    # 'N^(1/2)': N**(1/2),
    # 'N^(-1/2)': N**(-1/2),
    # 'N^(2/3)': N**(2/3),
    # 'N^(-2/3)': N**(-2/3),
    # 'N^(1/6)': N**(1/6),
    # 'N^(-1/6)': N**(-1/6),

    # ЯВНО ДОБАВЛЯЕМ СТЕПЕНИ K
    'K^2': K ** 2,
    'K^3': K ** 3,
    'K^4': K ** 4,
    'K^1.5': K ** 1.5,
    # 'K^2.5': K ** 2.5,
    # 'K^3.5': K ** 3.5,
    # 'K^4.5': K**4.5,
    'K^5': K ** 5,
    '1/K^1.5': 1 / K ** 1.5,
    # '1/K^2.5': 1 / K ** 2.5,
    # '1/K^3.5': 1 / K ** 3.5,
    # '1/K^4.5': 1/K**4.5,
    '1/K': 1 / K,
    # '1/p': 1/p,
    '1/K^2': 1 / (K ** 2),
    '1/K^3': 1 / (K ** 3),
    # '1/K^4': 1/(K**4),
    # '1/K^5': 1/(K**5),
    # 'K^3/2': K**(3/2),

    #'(lnK)^2': lnK ** 2,
    #'(lnK)^1/2': lnK ** (1/2),
    #'1/(lnK)^1/2': 1/lnK ** (1/2),
    # '(lnK)^3/2': lnK**(3/2),
    # '(lnK)^5/2': lnK**(5/2),
    # '(lnK)^7/2': lnK**(7/2),
    #'(lnK)^3': lnK ** 3,
    # '(lnK)^4': lnK **4,
    #'1/(lnK)^2': 1 / lnK ** 2,
    # '1/(lnK)^3': 1/lnK**3,
    # '1/(lnK)^4': 1/lnK**4,
    #
    # '1/(lnK)^3/2': 1/lnK ** (3 / 2),
    # '1/(lnK)^5/2': 1/lnK ** (5 / 2),
    # '1/(lnK)^7/2': 1/lnK ** (7 / 2),

    # '1/K^7/2': 1/K**(7/2),
    # 'p^2': p ** 2,
    # 'p^3': p ** 3,

    # '1/p^3': 1/(p**3),
    # '1/p^2': 1/(p**2),

    # Математические константы
    # 'e': math.e,
    # '1/e': 1 / math.e,
    # 'e^π': math.exp(math.pi),
    # 'ζ(2)': 1.644934,
    # '1/ζ(2)': 1/1.644934,
    # Специальные комбинации
    # 'N^-1/π': N ** (-1 / math.pi),
    # 'N^-1/2π': N ** (-1 / (2 * math.pi)),
    # 'lnN/π': lnN / math.pi,
    '√K': math.sqrt(K),
    '1/√K': 1 / math.sqrt(K),
    # '√pK': math.sqrt(p*K),
    # '1/√pK': 1/math.sqrt(p * K),
}


# =========================================================
# 4. ГЕНЕРАЦИЯ ФОРМУЛ (ЦЕЛЕНАПРАВЛЕННАЯ)
# =========================================================

def generate_formulas(components):
    """Генерирует формулы вида (A * B * C) / D и A * B * C"""
    formulas = {}

    # Ключевые компоненты для H_TEST
    candidates = [
        '(lnN)^3', '(lnN)^2', 'lnN',
        'N^(-1/3)', 'N^(1/3)', 'N',
        'K', '1/K', 'K^2', 'K^3',
        'π', '2π', '1/π', '1/(2π)',
        'e', '1/e',
    ]

    # Фильтруем только существующие
    available = [c for c in components if c in components]

    print(f"\n🔧 Генерация формул из {len(available)} компонентов...")

    # Комбинации из 3 компонентов: A * B * C
    for a, b, c in itertools.combinations(available, 3):
        val = components[a] * components[b] * components[c]
        if 1e-100 < abs(val) < 1e100:
            formulas[f'{a} * {b} * {c}'] = val

    # Комбинации из 4 компонентов: (A * B * C) / D
    for a, b, c in itertools.combinations(available, 3):
        for d in available:
            if d in (a, b, c):
                continue
            val = components[a] * components[b] * components[c] / components[d]
            if 1e-100 < abs(val) < 1e100:
                formulas[f'({a} * {b} * {c}) / {d}'] = val
                formulas[f'{a} * {b} * {c} / {d}'] = val

    # Комбинации: A * B / C
    for a, b, c in itertools.permutations(available, 3):
        if components[c] != 0:
            val = components[a] * components[b] / components[c]
            if 1e-100 < abs(val) < 1e100:
                formulas[f'{a} * {b} / {c}'] = val

    print(f"   Сгенерировано {len(formulas)} формул")
    return formulas


# 5. ПОИСК СОВПАДЕНИЙ

def find_matches(formulas, targets, tolerance=0.05):
    """Ищет совпадения с целевыми константами"""
    matches = []

    for formula, value in formulas.items():
        for target_name, target_value in targets.items():
            rel_error = abs(value - target_value) / target_value
            if rel_error < tolerance:
                matches.append({
                    'formula': formula,
                    'value': value,
                    'target': target_name,
                    'target_value': target_value,
                    'rel_error': rel_error,
                    'rel_percent': rel_error * 100
                })

    matches.sort(key=lambda x: x['rel_error'])
    return matches


# 6. ГЛАВНАЯ ФУНКЦИЯ

def main():
    print("НУМЕРОЛОГИЧЕСКИЙ ПОИСК ПЛАНКОВСКИХ ВЕЛИЧИН")

    print("\n📊 ПАРАМЕТРЫ ГИП:")
    print(f"  K = {K}")
    print(f"  p = {p:.4e}")
    print(f"  N = {N:.4e}")
    print(f"  lnN = {lnN:.6f}")

    # Генерируем формулы
    formulas = generate_formulas(components)

    # Объединяем с базовыми компонентами
    all_formulas = {**components, **formulas}

    # Ищем совпадения
    print("\n🔍 ПОИСК СОВПАДЕНИЙ...")
    matches = find_matches(all_formulas, targets, tolerance=0.05)
    print(f"   Найдено: {len(matches)}")

    print("ТОП ЛУЧШИХ СОВПАДЕНИЙ")
    for i, m in enumerate(matches[:2500], 1):
        marker = " ⭐⭐⭐" if m['rel_percent'] < 0.1 else (" ⭐⭐" if m['rel_percent'] < 1 else "")
        print(f"\n{i:2d}. {m['formula']}{marker}")
        print(f"    Значение: {m['value']:.12e}")
        print(f"    Цель: {m['target']} = {m['target_value']:.12e}")
        print(f"    Отн. ошибка: {m['rel_percent']:.6f}%")

    return matches


if __name__ == "__main__":
    matches = main()
