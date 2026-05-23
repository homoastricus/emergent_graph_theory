import math

import numpy as np

# КОНСТАНТЫ
K = 6.0
pi = math.pi
lnK = math.log(K)

# ЭКСПЕРИМЕНТАЛЬНЫЕ ЗНАЧЕНИЯ (CODATA 2018)
exp_values = {
    'hbar': 1.054571817e-34,
    'c': 299792458,
    'G': 6.67430e-11,
    'l_P': 1.616255e-35,
    't_P': 5.391247e-44,
    'm_P': 2.176434e-8,
    'E_P': 1.956082e9,
    'T_P': 1.416784e32,
    'k_B': 1.380649e-23,
    'alpha': 1 / 137.035999084,
    'm_e': 9.1093837015e-31,
    'm_p': 1.67262192369e-27,
    'q_e': 1.602176634e-19,
    'eps0': 8.8541878128e-12,
    'mu0': 1.25663706212e-6,
    'Z0': 376.730313668,
    'R∞': 10973731.568157,
    'a_0': 5.29177210903e-11,
    'λ_e': 2.42631023867e-12,
    'lambda_p': 1.32140985539e-15,
    'κ': 2.07664746e-43,
    'Φ_0': 2.06783366752e-15,
    'v_H': 4.388471e-25,
    'Z_0': 376.730313,
    'Λ': 1.08929e-52,
    'r_e':2.8179402853e-15,
}

# ЭМЕРДЖЕНТНЫЕ ЗНАЧЕНИЯ ЕТИ
N_phys = 4.197668e121
lnN_phys = math.log(N_phys)
N13_phys = N_phys ** (1 / 3)


def compute_eti_values():
    lnN = lnN_phys
    N13 = N13_phys

    return {
        'hbar': (lnN ** 3) / (K * N13),
        'c': pi * (lnN ** 4) / (K ** 2 * lnK),
        'G': 16 * pi ** 3 * lnN ** 13 / (K ** 5 * lnK * N13),
        'l_P': 4 * lnN ** 2 * lnK / N13,
        't_P': 4 * K ** 2 * lnK ** 2 / (pi * N13 * lnN ** 2),
        'm_P': K / (pi * 4 * lnN ** 3),
        'E_P': (lnN ** 5) * pi / (4 * K ** 3 * lnK ** 2),
        'T_P': 8 * pi * N13 / (lnN ** 4),
        'k_B': (lnN ** 8) / (8 * pi ** 2 * N13),
        'alpha': 2 * lnK ** 2 / (pi * lnN),
        'm_e': 4 * pi * lnN ** 4 / (K ** 0.5 * N13),
        'm_p': math.sqrt(pi) * (lnN ** 6) / (K ** 1.5 * N13),
        'q_e': 1.0 / (pi * K ** 1.5 * lnN ** 7),
        'eps0': N13 / (8 * pi ** 3 * lnK * lnN ** 20),
        'mu0': (8 * pi * K ** 4 * lnK ** 3 * lnN ** 12) / N13,
        'Z0': 8 * K ** 2 * pi ** 2 * lnK ** 2 * lnN ** 16 / N13,
        'R∞': 4 * lnK ** 3 * lnN ** 3 / (pi * K ** 1.5),
        'a_0': K ** 1.5 / (8 * pi * lnK * lnN ** 4),
        'λ_e': K ** 1.5 * lnK / (2 * pi * lnN ** 5),
        'lambda_p': 2 * K ** 2.5 * lnK / (math.sqrt(pi) * lnN ** 7),
        'κ': 128 * K ** 3 * lnK ** 3 / (lnN ** 3 * N13),
        'Φ_0': lnN ** 10 * pi ** 2 * K ** (1 / 2) / N13,
        'v_H': lnN ** 6 * 8 * pi ** (3 / 2) * 1 / (2 ** (1 / 2) * N13),
        'Z_0': 8 * K ** 2 * pi ** 2 * lnK ** 2 * lnN ** 16 / N13,
        'Λ': lnN ** 12 / (pi ** (1 / 2) * N_phys ** (2 / 3)),
        'r_e': lnK ** 3 * K ** (3 / 2) / (2 * pi ** 3 * lnN ** 6)

}

eti_values = compute_eti_values()

identities = [
    # ЧАСТЬ 1: ПЛАНКОВСКАЯ САМОСОГЛАСОВАННОСТЬ (11 тождеств)
    ("ħ·c/(G·m_P²)", lambda c: c['hbar'] * c['c'] / (c['G'] * c['m_P'] ** 2), 1),
    ("c·t_P/l_P", lambda c: c['c'] * c['t_P'] / c['l_P'], 1),
    ("l_P²·c³/(ħ·G)", lambda c: c['l_P'] ** 2 * c['c'] ** 3 / (c['hbar'] * c['G']), 1),
    ("l_P³/(t_P²·G·m_P)", lambda c: c['l_P'] ** 3 / (c['t_P'] ** 2 * c['G'] * c['m_P']), 1),
    ("ħ/(c·l_P·m_P)", lambda c: c['hbar'] / (c['c'] * c['l_P'] * c['m_P']), 1),
    ("ħ/(c²·t_P·m_P)", lambda c: c['hbar'] / (c['c'] ** 2 * c['t_P'] * c['m_P']), 1),
    ("c³·t_P/(G·m_P)", lambda c: c['c'] ** 3 * c['t_P'] / (c['G'] * c['m_P']), 1),
    ("c⁵·t_P/(E_P·G)", lambda c: c['c'] ** 5 * c['t_P'] / (c['E_P'] * c['G']), 1),
    ("c²·l_P/(G·m_P)", lambda c: c['c'] ** 2 * c['l_P'] / (c['G'] * c['m_P']), 1),
    ("E_P/(m_P·c²)", lambda c: c['E_P'] / (c['m_P'] * c['c'] ** 2), 1),
    ("E_P·t_P²/(l_P²·m_P)", lambda c: c['E_P'] * c['t_P'] ** 2 / (c['l_P'] ** 2 * c['m_P']), 1),

    # ЧАСТЬ 2: ЭЛЕКТРОМАГНИТНАЯ САМОСОГЛАСОВАННОСТЬ
    ("c²·ε₀·μ₀", lambda c: c['c'] ** 2 * c['eps0'] * c['mu0'], 1),
    ("c·ε₀·Z₀", lambda c: c['c'] * c['eps0'] * c['Z0'], 1),
    ("Z₀²·ε₀/μ₀", lambda c: c['Z0'] ** 2 * c['eps0'] / c['mu0'], 1),
    ("Z₀/(c·μ₀)", lambda c: c['Z0'] / (c['c'] * c['mu0']), 1),
    ("c³·ε₀²·μ₀·Z₀", lambda c: c['c'] ** 3 * c['eps0'] ** 2 * c['mu0'] * c['Z0'], 1),

    # ЧАСТЬ 3: КРОСС-СЕКТОРАЛЬНЫЕ БАЗОВЫЕ
    ("ħ²·E_P/(G²·m_P⁵)", lambda c: c['hbar'] ** 2 * c['E_P'] / (c['G'] ** 2 * c['m_P'] ** 5), 1),
    ("ħ³/(t_P·G²·m_P⁵)", lambda c: c['hbar'] ** 3 / (c['t_P'] * c['G'] ** 2 * c['m_P'] ** 5), 1),
    ("ħ·c⁵/(E_P²·G)", lambda c: c['hbar'] * c['c'] ** 5 / (c['E_P'] ** 2 * c['G']), 1),
    ("ħ·c/(E_P·l_P)", lambda c: c['hbar'] * c['c'] / (c['E_P'] * c['l_P']), 1),
    ("ħ/(E_P·t_P)", lambda c: c['hbar'] / (c['E_P'] * c['t_P']), 1),
    ("c⁴·l_P/(E_P·G)", lambda c: c['c'] ** 4 * c['l_P'] / (c['E_P'] * c['G']), 1),
    ("r_e·1/α^{5}*R∞^2 λ_e = 1/(8*pi)", lambda c: c['r_e'] / c['alpha']**5 * c['R∞']**2 * c['λ_e'], 1/(8*pi)),

    # ЧАСТЬ 4: ЦЕЛОЧИСЛЕННЫЕ ИНВАРИАНТЫ — СТЕПЕНИ 2
    ("m_P⁶/(R∞·a_0⁴·λ_e) = 2",
     lambda c: (c['R∞'] * c['a_0'] ** 4 * c['λ_e'] / c['m_P'] ** 6), 2),
    ("l_P³/(α²·m_e³·a_0) = 2",
     lambda c: c['l_P'] ** 3 / (c['alpha'] ** 2 * c['m_e'] ** 3 * c['a_0']), 2),
    ("(α·a_0²)/m_P³ = 2",
     lambda c: (c['alpha'] * c['a_0'] ** 2) / c['m_P'] ** 3, 2),
    ("l_P·a_0/(m_P²·m_e) = 2",
     lambda c: c['l_P'] * c['a_0'] / (c['m_P'] ** 2 * c['m_e']), 2),

    # ЧАСТЬ 5: ЦЕЛОЧИСЛЕННЫЕ ИНВАРИАНТЫ — СТЕПЕНИ 3 (2 тождества)
    ("m_P·m_p²·κ/l_P³ = 3",
     lambda c: (c['m_P'] * c['m_p'] ** 2 * c['κ']) / c['l_P'] ** 3, 3),
    ("(m_p²·κ) / t_P²·G = 3",
     lambda c: (c['m_p'] ** 2 * c['κ']) / (c['t_P'] ** 2 * c['G']), 3),

    # ЧАСТЬ 6: ЦЕЛОЧИСЛЕННЫЕ ИНВАРИАНТЫ — СМЕШАННЫЕ 2^a·3^b (7 тождеств)
    ("α²/(R∞·λ_e) = 2",
     lambda c: c['alpha'] ** 2 / (c['R∞'] * c['λ_e']), 2),
    ("l_P²/(m_P·α·m_e²) = 2",
     lambda c: c['l_P'] ** 2 / (c['m_P'] * c['alpha'] * c['m_e'] ** 2), 2),
    ("ħ⁵/(m_P⁷·T_P³·m_p⁸) = 32",
     lambda c: c['hbar'] ** 5 / (c['m_P'] ** 7 * c['T_P'] ** 3 * c['m_p'] ** 8), 32),
    ("m_P⁶·Z_0/(k_B·λ_e²) = 486",
     lambda c: c['m_P'] ** 6 * c['Z0'] / (c['k_B'] * c['λ_e'] ** 2), 486),
    ("m_P³/(α·a_0²) = 2",
     lambda c: (c['alpha'] * c['a_0'] ** 2) / c['m_P'] ** 3, 2),
    ("G·a_0/E_P·m_e = 2",
     lambda c: (c['G'] * c['a_0']) / (c['E_P'] * c['m_e']), 2),

    # ЧАСТЬ 6: ЦЕЛОЧИСЛЕННЫЕ ИНВАРИАНТЫ — СТЕПЕНИ 3
    ("l_P³/(m_P·m_p²·κ) = 3",
     lambda c: (c['m_P'] * c['m_p'] ** 2 * c['κ'] / c['l_P'] ** 3), 3),
    ("t_P²·G/(m_p²·κ) = 3",
     lambda c: (c['m_p'] ** 2 * c['κ'] / (c['t_P'] ** 2 * c['G'])), 3),

    # ЧАСТЬ 7 (продолжение): СМЕШАННЫЕ 2^a·3^b
    ("G·λ_e/(T_P·m_p²) = 1/sqrt(K)",
     lambda c: c['G'] * c['λ_e'] / (c['T_P'] * c['m_p'] ** 2), 1 / math.sqrt(K)),
    ("m_P²·v_H⁶/(m_e⁵·Φ_0) = 2592",
     lambda c: c['m_P'] ** 2 * c['v_H'] ** 6 / (c['m_e'] ** 5 * c['Φ_0']), 2592),

    # ЧАСТЬ 8 (продолжение): π-ИНВАРИАНТЫ
    ("E_P·κ/l_P = 8π",
     lambda c: c['E_P'] * c['κ'] / c['l_P'], 8 * pi),
    ("c⁶·m_e³·R∞/G³ = 1/16π",
     lambda c: c['c'] ** 6 * c['m_e'] ** 3 * c['R∞'] / c['G'] ** 3, 1 / (16 * pi)),
    ("ħ/(Φ_0·q_e) = 1/π",
     lambda c: c['hbar'] / (c['Φ_0'] * c['q_e']), 1 / pi),

    # ЧАСТЬ 9: √π-СЕМЕЙСТВО
    ("c²·α⁶·λ_p⁻¹/R∞³ = 1/72√π",
     lambda c: c['c'] ** 2 * c['alpha'] ** 6 / (c['lambda_p'] * c['R∞'] ** 3), 1 / (72 * math.sqrt(pi))),
    ("a_0³·λ_e⁶/(m_P⁷·λ_p³) = 1/96√π",
     lambda c: c['a_0'] ** 3 * c['λ_e'] ** 6 / (c['m_P'] ** 7 * c['lambda_p'] ** 3), 1 / (96 * math.sqrt(pi))),
    ("m_p⁵·Φ_0⁻³·v_H²/ħ⁴ = 1/243√π",
     lambda c: c['m_p'] ** 5 * c['v_H'] ** 2 / (c['Φ_0'] ** 3 * c['hbar'] ** 4), 1 / (243 * math.sqrt(pi))),
    ("ħ⁶/(m_P⁴·m_e³·m_p³) = 1/324√π",
     lambda c: c['hbar'] ** 6 / (c['m_P'] ** 4 * c['m_e'] ** 3 * c['m_p'] ** 3), 1 / (324 * math.sqrt(pi))),
    ("λ_p/(c²·λ_e³) = 576√π",
     lambda c: c['lambda_p'] / (c['c'] ** 2 * c['λ_e'] ** 3), 576 * math.sqrt(pi)),
    ("α·a_0^{3}·λ_e / m_P^6 = 8π",
     lambda c: c['alpha'] * c['λ_e'] * c['a_0'] ** 3 / c['m_P'] ** 6, 8 * pi),

    ("E_P·α·m_e^2·1/m_p^{2} = 8π/K",
     lambda c: c['alpha'] * c['E_P'] * c['m_e'] ** 2 / c['m_p'] ** 2, 8 * pi / 6),
    ("λ_e/α·a_0 =2·π",
     lambda c: c['λ_e'] / (c['alpha'] * c['a_0']), 2 * pi),
    ("λ_e/α·a_0 =2·π",
     lambda c: c['λ_e'] / (c['alpha'] * c['a_0']), 2 * pi),
    ("λ_e·m_e·c/ħ =2·π",
     lambda c: c['λ_e'] * c['m_e'] * c['c'] / c['hbar'], 2 * pi),
    ("λ_p·m_p·c/ħ =2·π",
     lambda c: c['lambda_p'] * c['m_p'] * c['c'] / c['hbar'], 2 * pi),
    ("l_P·m_P/m_e·λ_e =1/2·π",
     lambda c: c['l_P'] * c['m_P'] / (c['λ_e'] * c['m_e']), 1 / (2 * pi)),

    # ЧАСТЬ 10: π^(3/2)-СЕМЕЙСТВО
    ("m_P⁴·λ_p³/(a_0²·λ_e⁵) = 384π^(3/2)",
     lambda c: c['m_P'] ** 4 * c['lambda_p'] ** 3 / (c['a_0'] ** 2 * c['λ_e'] ** 5), 384 * pi ** (3 / 2)),
    ("m_P·α·λ_p³/λ_e⁵ = 768π^(3/2)",
     lambda c: c['m_P'] * c['alpha'] * c['lambda_p'] ** 3 / c['λ_e'] ** 5, 768 * pi ** (3 / 2)),
    ("c^6 · 1 / E_P ^ {3} · a_0 · λ_e = 4pi",
     lambda c: c['c'] ** 6 * c['a_0'] * c['λ_e'] / c['E_P'] ** 3, 4 * pi),
    ("t_P·λ_p/(λ_e³·v_H) = 8 sqrt(2)",
     lambda c: c['t_P'] * c['lambda_p'] / (c['λ_e'] ** 3 * c['v_H']), 8 * math.sqrt(2)),
    ("ħ⁴·mu0·λ_p²/l_P⁵ = 243",
     lambda c: c['hbar'] ** 4 * c['mu0'] * c['lambda_p'] ** 2 / c['l_P'] ** 5, 243),
    ("α·Φ_0/(Z_0·q_e) = 4",
     lambda c: (c['Z0'] * c['q_e'] / (c['alpha'] * c['Φ_0'])), 4),
    ("t_P⁵·E_P⁶·Z_0/m_p⁶ = 432",
     lambda c: c['t_P'] ** 5 * c['E_P'] ** 6 * c['Z0'] / c['m_p'] ** 6, 432),
    ("l_P·m_p⁶·eps0/ħ⁶ = 432",
     lambda c: c['hbar'] ** 6 / (c['l_P'] * c['m_p'] ** 6 * c['eps0']), 432),
    ("ħ⁵·E_P·Z_0/m_p⁶ = 432",
     lambda c: c['hbar'] ** 5 * c['E_P'] * c['Z0'] / c['m_p'] ** 6, 432),
    ("m_P⁶·Z_0/(k_B·λ_e²) = 486",
     lambda c: c['m_P'] ** 6 * c['Z0'] / (c['k_B'] * c['λ_e'] ** 2), 486),
    ("m_P^3/(a_0·λ_e) = 1/(4*pi)",
     lambda c: (c['m_P'] ** 3) / (c['a_0'] * c['λ_e']), 1 / (4 * pi)),
    ("ħ·α·1/Z_0·1/q_e^{2} = 1/(4*pi)",
     lambda c: c['hbar'] * c['alpha'] / (c['Z_0'] * c['q_e'] ** 2), 1 / (4 * pi)),
    ("m_p^{3} · λ_p · κ / l_P^4 = 6*pi",
     lambda c: c['m_p'] ** 3 * c['lambda_p'] * c['κ'] / c['l_P'] ** 4, 6 * pi),
    ("R∞^{2} · λ_e^{4} ·  m_P^3 · α^5 = 2*pi^2",
     lambda c: c['R∞'] ** 2 * c['λ_e'] ** 4 / (c['m_P'] ** 3 * c['alpha'] ** 5), 2 * pi ** 2),
    ("T_P·Λ^2/k_B·v_H^2 = 2/pi",
     lambda c: (c['T_P'] * c['Λ'] ** 2) / (c['k_B'] * c['v_H'] ** 2), 2 / pi),
    ("α·Φ_0^2/Z_0·ħ = pi/4",
     lambda c: (c['alpha'] * c['Φ_0'] ** 2) / (c['Z_0'] * c['hbar']), pi / 4),
    ("ep_0·Φ_0^2/m_e·a_0 = pi/4",
     lambda c: (c['eps0'] * c['Φ_0'] ** 2) / (c['m_e'] * c['a_0']), pi / 4),
    ("Λ·t_P^3*T_P^3/κ^2) = 2/sqrt(pi)",
     lambda c: (c['Λ'] * c['t_P'] ** 3 * c['T_P']**3) / c['κ']**2, 2/math.sqrt(pi)),
    ("Φ_0·m_P^2/m_e = 27/8*pi",
     lambda c: c['Φ_0'] * c['m_P'] ** 2 / c['m_e'], 27/(8*pi)),

    # ЧАСТЬ 8: π-ИНВАРИАНТЫ
    ("m_P⁹/(R∞²·a_0⁷·λ_e) = π",
     lambda c: c['m_P'] ** 9 / (c['R∞'] ** 2 * c['a_0'] ** 7 * c['λ_e']), pi),
    ("ħ²·v_H⁴/(m_e⁵·Φ_0) = 1/π",
     lambda c: c['hbar'] ** 2 * c['v_H'] ** 4 / (c['m_e'] ** 5 * c['Φ_0']), 1 / pi),
    ("m_P³·λ_e/(α²·a_0³) = π",
     lambda c: c['m_P'] ** 3 * c['λ_e'] / (c['alpha'] ** 2 * c['a_0'] ** 3), pi),
    ("m_e⁵/(Φ_0·q_e²·v_H⁴) = 1/π",
     lambda c: c['m_e'] ** 5 / (c['Φ_0'] * c['q_e'] ** 2 * c['v_H'] ** 4), 1 / pi),
    ("α·R∞·a_0⁵/m_P⁶ = 1/π",
     lambda c: c['alpha'] * c['R∞'] * c['a_0'] ** 5 / c['m_P'] ** 6, 1 / pi),
    ("ħ/(Φ_0·q_e) = 1/π",
     lambda c: c['hbar'] / (c['Φ_0'] * c['q_e']), 1 / pi),
    ("t_P·E_P/(Φ_0·q_e) = 1/π",
     lambda c: c['t_P'] * c['E_P'] / (c['Φ_0'] * c['q_e']), 1 / pi),
]

# СРАВНЕНИЕ
print("СРАВНЕНИЕ: CODATA vs ЕТИ (ЭМЕРДЖЕНТНЫЕ ФОРМУЛЫ)")
print(f"\n{'Тождество':<55} {'CODATA':<18} {'ЕТИ':<18} {'δ(CODATA)':<14} {'δ(ЕТИ)':<14} {'ЕТИ точнее?':<15}")

for name, func, target in identities:
    # CODATA
    try:
        val_exp = func(exp_values)
        err_exp = abs(val_exp - target) / target * 100
    except:
        val_exp = float('nan')
        err_exp = float('nan')

    # ЕТИ
    try:
        val_eti = func(eti_values)
        err_eti = abs(val_eti - target) / target * 100
    except:
        val_eti = float('nan')
        err_eti = float('nan')

    eti_better = "✅ ДА" if err_eti < err_exp else ("≈ равно" if abs(err_eti - err_exp) < 0.001 else "❌ НЕТ")

    print(f"{name:<55} {val_exp:<18.10f} {val_eti:<18.10f} {err_exp:<14.8f} {err_eti:<14.8f} {eti_better:<15}")

# СТАТИСТИКА

eti_errors = []
exp_errors = []
for name, func, target in identities:
    try:
        val_exp = func(exp_values)
        err_exp = abs(val_exp - target) / target * 100
        exp_errors.append(err_exp)
    except:
        pass
    try:
        val_eti = func(eti_values)
        err_eti = abs(val_eti - target) / target * 100
        eti_errors.append(err_eti)
    except:
        pass

print(f"\n  Средняя ошибка CODATA: {np.mean(exp_errors):.6f}%")
print(f"  Средняя ошибка ЕТИ:    {np.mean(eti_errors):.10f}%")
print(f"  Макс. ошибка CODATA:   {max(exp_errors):.6f}%")
print(f"  Макс. ошибка ЕТИ:      {max(eti_errors):.10f}%")
