import math
import numpy as np
from collections import defaultdict

# ============================================================
# ФУНДАМЕНТАЛЬНЫЕ КОНСТАНТЫ
# ============================================================
K = 6.0
pi = math.pi
lnK = math.log(K)
sqrt2 = math.sqrt(2)
sqrt3 = math.sqrt(3)
gamma_E = 0.5772156649015329

N = 4.197668e121
lnN = math.log(N)
N13 = N ** (1 / 3)
lnlnN = math.log(lnN)
lnN2 = lnN ** 2

correction_scale = lnK / lnN

# ============================================================
# CDATA
# ============================================================
CDATA = {
    'ħ': 1.054571817e-34,
    'h': 6.62607015e-34,
    't_P': 5.391247e-44,
    'l_P': 1.616255e-35,
    'm_P': 2.176434e-8,
    'E_P': 1.956082e9,
    'T_P': 1.416784e32,
    'c': 299792458,
    'G': 6.67430e-11,
    'k_B': 1.380649e-23,
    'α': 1 / 137.035999084,
    'm_e': 9.1093837015e-31,
    'm_muon': 1.883531627e-28,
    'm_tau': 3.167e-27,
    'm_proton': 1.67262192e-27,
    'm_DT': 3.3435837724e-27,
    'm_Λ_barion': 1.9901611e-27,
    'Sigma_plus': 2.11933e-27,
    'Ksi_0': 2.34532e-27,
    'omega_minus': 2.9859e-27,
    'Ksi_plus': 4.3995e-27,
    'Omega0_c': 4.808e-27,
    'lambda_B0': 1.0023e-26,
    'Ksi_minus': 2.358e-27,
    'Sigma_minus': 2.132e-27,
    'm_z_bozon': 1.62614e-25,
    'm_w_bozon': 1.43362e-25,
    'm_Higgs': 2.23319e-25,
    'vacuum_higgs': 4.388471e-25,
    'm_pi_meson': 2.4880888e-28,
    'm_pi0_meson': 2.40609e-28,
    'm_k0_meson': 8.801929e-28,
    'm_D0': 3.32479e-27,
    'm_J_ψ': 5.52061e-27,
    'm_eta': 9.767732e-28,
    'm_Υ_1S': 1.68715e-26,
    'phi_meson': 1.819e-27,
    'omega_meson': 1.394e-27,
    'eta_shtrih': 1.7086e-27,
    'm_qu_u': 2.1650e-30,
    'm_qu_d': 4.7915e-30,
    'm_qu_s': 9.635e-30,
    'm_qu_c': 1.27e-27,
    'm_qu_b': 4.180e-27,
    'm_qu_t': 3.04e-25,
    'RIDBERG': 1.097373e7,
    'bor_radius': 5.29177210903e-11,
    'compton_e': 2.426e-12,
    'compton_proton': 1.32140985396e-15,
    'ep_0': 8.8725366415e-12,
    'mu_0': 1.25663706127e-6,
    'q_e': 1.602176634e-19,
    'impedance': 376.730313,
    'Φ0_magnetic_stream': 2.06783366752e-15,
    'Lambda_cosmo': 1.08929e-52,
    'Einstein_constant': 2.07664746e-43,
    'm_proton_to_m_electron': 1836.152673426,
    'm_tau_m_electron': 3477,
    'm_W_to_m_Z': 0.8815,
    'm_plank_to_m_e': 2.389e22,
    'm_Higgs_to_m_W': 1.558,
    'mu_lifetime': 2.1969811e-6,
    'tau_lifetime': 2.903e-13,
    'pion_lifetime': 2.6033e-8,
    'neutron_lifetime': 877.8,
    'kaon_lifetime': 1.2380e-8,
    'D_plus_lifetime': 1.040e-12,
    'B_plus_lifetime': 1.638e-12,
    'Λ_b_lifetime': 1.471e-12,
    'D0_lifetime': 4.101e-13,
    'h2_connection_energy': 2.178872e-18,
    'm_neitrino': 1.783e-36,
}


# ============================================================
# ВЫЧИСЛЕНИЕ ВСЕХ ЭМЕРДЖЕНТНЫХ ФОРМУЛ
# ============================================================
def compute_all_formulas(N):
    lnN = math.log(N)
    N13 = N ** (1 / 3)
    p_val = 1 / (K * N13)
    Kp = K * p_val

    return {
        'ħ': (lnN ** 3) / (K * N13),
        'h': 2 * pi * (lnN ** 3) / (K * N13),
        't_P': 4 * K ** 2 * lnK ** 2 / (pi * N13 * lnN ** 2),
        'l_P': 4 * lnN ** 2 * lnK / N13,
        'm_P': K / (pi * 4 * lnN ** 3),
        'E_P': (lnN ** 5) * pi / (4 * K ** 3 * lnK ** 2),
        'T_P': 8 * pi * N13 / (lnN ** 4),
        'c': pi * (lnN ** 4) / (K ** 2 * lnK),
        'G': 16 * pi ** 3 * lnN ** 13 / (K ** 5 * lnK * N13),
        'k_B': Kp * (lnN ** 8) / (8 * pi ** 2),
        'α': 2 * lnK ** 2 / (pi * lnN),
        'm_e': 4 * pi * lnN ** 4 / (K ** 0.5 * N13),
        'm_muon': 4 * pi ** 2 * lnN ** 5 / (K * sqrt3 * N13),
        'm_tau': math.sqrt(pi) * lnN ** 5 * K ** 2 / N13,
        'm_proton': math.sqrt(pi) * lnN ** 6 / (K ** 1.5 * N13),
        'm_DT': lnN ** 6 * math.sqrt(2 * pi) / (K * sqrt3 * N13),
        'm_Λ_barion': lnN ** 6 * sqrt2 / (pi ** 2 * N13),
        'Sigma_plus': K * lnN ** 6 / (4 * pi ** 2 * N13),
        'Ksi_0': lnN ** 6 * math.sqrt(2 * pi) / (K ** 1.5 * N13),
        'omega_minus': lnN ** 6 * pi / (K ** 1.5 * N13),
        'Ksi_plus': lnN ** 6 / (pi * N13),
        'Omega0_c': K * lnN ** 6 / (pi ** 2.5 * N13),
        'lambda_B0': math.sqrt(pi) * lnN ** 6 / (K ** 0.5 * N13),
        'Ksi_minus': lnN ** 6 * math.sqrt(2 * pi) / (K ** 1.5 * N13),
        'Sigma_minus': K * lnN ** 6 / (4 * pi ** 2 * N13),
        'm_z_bozon': 4 * pi ** 2.5 * lnN ** 6 / (K * N13),
        'm_w_bozon': 2 * pi ** 3 * lnN ** 6 / (K * N13),
        'm_Higgs': 4 * pi ** 2 * lnN ** 6 / (K ** 0.5 * N13),
        'vacuum_higgs': 8 * pi ** 1.5 * lnN ** 6 / (sqrt2 * N13),
        'm_pi_meson': lnN ** 6 / (4 * pi ** 2 * sqrt2 * N13),
        'm_pi0_meson': 2 * pi * K ** 3 * lnN ** 4 / N13,
        'm_k0_meson': lnN ** 6 * math.sqrt(2 * pi) / (4 * pi ** 2 * N13),
        'm_D0': lnN ** 6 * math.sqrt(2 * pi) / (K * sqrt3 * N13),
        'm_J_ψ': 8 * pi ** 2 * sqrt2 * lnN ** 5 / N13,
        'm_eta': 2 * pi ** 2 * lnN ** 5 / N13,
        'm_Υ_1S': lnN ** 6 * sqrt3 / (sqrt2 * N13),
        'phi_meson': math.sqrt(2 * pi) * K ** 1.5 * lnN ** 5 / N13,
        'omega_meson': 2 * pi ** 2 * sqrt2 * lnN ** 5 / N13,
        'eta_shtrih': K ** 3 * lnN ** 5 / (2 * pi * N13),
        'm_qu_u': lnN ** 5 * sqrt3 / (4 * pi ** 2 * N13),
        'm_qu_d': lnN ** 5 / (K * sqrt3 * N13),
        'm_qu_s': lnN ** 4 * pi ** 3.5 / N13,
        'm_qu_c': 2 * pi ** 2 * lnN ** 6 / (K ** 3 * N13),
        'm_qu_b': pi * lnN ** 6 / (K * sqrt3 * N13),
        'm_qu_t': K ** 3 * lnN ** 6 / (pi ** 2 * N13),
        'RIDBERG': 4 * lnN ** 3 * lnK ** 3 / (pi * K ** 1.5),
        'bor_radius': K ** 1.5 / (8 * pi * lnN ** 4 * lnK),
        'compton_e': K ** 1.5 * lnK / (2 * pi * lnN ** 5),
        'compton_proton': 2 * K ** 2.5 * lnK / (math.sqrt(pi) * lnN ** 7),
        'ep_0': N13 / (8 * pi ** 3 * lnK * lnN ** 20),
        'mu_0': 8 * pi * K ** 4 * lnK ** 3 * lnN ** 12 / N13,
        'q_e': 1.0 / (pi * K ** 1.5 * lnN ** 7),
        'impedance': 8 * K ** 2 * pi ** 2 * lnK ** 2 * lnN ** 16 / N13,
        'Φ0_magnetic_stream': lnN ** 10 * pi ** 2 * K ** 0.5 / N13,
        'Lambda_cosmo': lnN ** 12 / (math.sqrt(pi) * N13 ** 2),
        'Einstein_constant': 128 * K ** 3 * lnK ** 3 / (lnN ** 3 * N13),
        'm_proton_to_m_electron': lnN ** 2 / (4 * math.sqrt(pi) * K),
        'm_tau_m_electron': K ** 2.5 * lnN / (4 * math.sqrt(pi)),
        'm_W_to_m_Z': math.sqrt(pi) / 2,
        'm_plank_to_m_e': K ** 1.5 * N13 / (16 * pi ** 2 * lnN ** 7),
        'm_Higgs_to_m_W': 2 * K ** 0.5 / pi,
        'mu_lifetime': lnK / (K * sqrt3 * lnN ** 2),
        'tau_lifetime': 1 / (2 * lnN ** 5),
        'pion_lifetime': K ** 2 * sqrt2 * pi / lnN ** 4,
        'neutron_lifetime': sqrt2 * N ** (1 / 12) / lnN ** 3,
        'kaon_lifetime': 4 / (K ** 1.5 * lnN ** 3),
        'D_plus_lifetime': 1 / (math.sqrt(pi) * K ** 2.5 * lnN ** 4),
        'B_plus_lifetime': lnK * pi / (2 * lnN ** 5),
        'Λ_b_lifetime': lnK * sqrt2 / lnN ** 5,
        'D0_lifetime': lnK / (2 * pi ** 2 * K ** 2 * lnN ** 4),
        'h2_connection_energy': 8 * pi * lnN ** 10 * lnK ** 2 / (K ** 4.5 * N13),
        'm_neitrino': lnN ** 2 * sqrt2 / (lnK * N13),
    }


# ============================================================
# ПАРАМЕТРЫ ДЛЯ ВЫЧИСЛЕНИЯ γ ЧЕРЕЗ СПЕКТРАЛЬНУЮ ФОРМУЛУ
# ============================================================
# Для каждой константы: (a, b)
const_params = {
    # Квантовые
    'ħ': (3, 1 / 3),
    'h': (3, 1 / 3),
    # Планковские
    't_P': (-2, 1 / 3),
    'l_P': (2, 1 / 3),
    'm_P': (-3, 0),
    'E_P': (5, 0),
    'T_P': (-4, -1 / 3),
    # Фундаментальные
    'c': (4, 0),
    'G': (13, 1 / 3),
    'k_B': (8, 1 / 3),
    'α': (-1, 0),
    # Лептоны
    'm_e': (4, 1 / 3),
    'm_muon': (5, 1 / 3),
    'm_tau': (5, 1 / 3),
    # Барионы
    'm_proton': (6, 1 / 3),
    # ... остальные будут с b=1/3 и a=4,5,6
}


# Функция вычисления γ через спектральную формулу
def compute_gamma_spectral(a, b):
    """Вычисляет γ через спектральную формулу ЕТИ"""
    # Определяем n
    if abs(b) < 1e-6:
        delta_b = 0.0
        n = -a
    elif abs(b - 1 / 3) < 1e-6:
        delta_b = gamma_E
        n = 16 - a
    elif abs(b + 1 / 3) < 1e-6:
        delta_b = gamma_E
        n = 16 + a  # для b=-1/3
    else:
        return None, None

    # C = π(n + δ_b)
    C = pi * (n + delta_b)

    # γ = b/(2lnK)·(lnN)² - a/lnK·lnN·lnlnN + C·lnN
    gamma = (b / (2 * lnK)) * lnN ** 2 - (a / lnK) * lnN * lnlnN + C * lnN

    return gamma, n


# ============================================================
# ТЕСТ СПЕКТРАЛЬНОЙ ФОРМУЛЫ
# ============================================================
print("=" * 120)
print("ТЕСТ СПЕКТРАЛЬНОЙ ФОРМУЛЫ ДЛЯ γ")
print("=" * 120)
print(f"\n  K = {K}, lnN = {lnN:.6f}, lnlnN = {lnlnN:.6f}")
print(f"  γ_E = {gamma_E:.10f}")
print(f"  Поправка: f = f0 * exp(γ * lnK/lnN)")
print(f"  lnK/lnN = {correction_scale:.8f}")
print()

emergent = compute_all_formulas(N)

# Проверяем константы с известными a, b
print(
    f"  {'Константа':<22} {'a':>4} {'b':>6} {'n':>4} {'C':>12} {'γ':>12} {'f0':>16} {'f_corr':>16} {'CDATA':>16} {'f0/CDATA':>12} {'f_corr/CDATA':>14}")
print(f"  {'-' * 150}")

results = []
for const_name in sorted(emergent.keys()):
    if const_name not in CDATA:
        continue
    if const_name not in const_params:
        continue

    a, b = const_params[const_name]
    gamma, n = compute_gamma_spectral(a, b)

    if gamma is None:
        continue

    f0 = emergent[const_name]
    f_corr = f0 * math.exp(gamma * correction_scale)
    cdata_val = CDATA[const_name]

    ratio0 = f0 / cdata_val
    ratio_corr = f_corr / cdata_val
    dev0 = abs(ratio0 - 1) * 100
    dev_corr = abs(ratio_corr - 1) * 100

    results.append({
        'name': const_name,
        'a': a,
        'b': b,
        'n': n,
        'gamma': gamma,
        'f0': f0,
        'f_corr': f_corr,
        'cdata': cdata_val,
        'ratio0': ratio0,
        'ratio_corr': ratio_corr,
        'dev0': dev0,
        'dev_corr': dev_corr,
    })

    print(f"  {const_name:<22} {a:>4} {b:>6.2f} {n:>4} {pi * (n + (gamma_E if abs(b - 1 / 3) < 1e-6 else 0)):>12.4f} "
          f"{gamma:>12.6f} {f0:>16.8e} {f_corr:>16.8e} {cdata_val:>16.8e} "
          f"{ratio0:>12.8f} {ratio_corr:>14.8f}")

# Статистика
print(f"\n{'=' * 120}")
print("СТАТИСТИКА")
print("=" * 120)

devs0 = [r['dev0'] for r in results]
devs_corr = [r['dev_corr'] for r in results]

print(f"\n  Констант с известными (a,b): {len(results)}")
print(f"  Средняя ошибка без поправки: {np.mean(devs0):.6f}%")
print(f"  Средняя ошибка с поправкой:  {np.mean(devs_corr):.6f}%")
print(f"  Медианная ошибка с поправкой: {np.median(devs_corr):.6f}%")

# Распределение ошибок
excellent = sum(1 for d in devs_corr if d < 0.01)
good = sum(1 for d in devs_corr if 0.01 <= d < 0.1)
ok = sum(1 for d in devs_corr if 0.1 <= d < 1.0)
poor = sum(1 for d in devs_corr if d >= 1.0)

print(f"\n  Распределение ошибок с поправкой:")
print(f"    < 0.01%:  {excellent}")
print(f"    0.01-0.1%: {good}")
print(f"    0.1-1%:    {ok}")
print(f"    > 1%:      {poor}")

# Улучшение
improved = sum(1 for r in results if r['dev_corr'] < r['dev0'])
worsened = sum(1 for r in results if r['dev_corr'] > r['dev0'])
print(f"\n  Улучшилось: {improved}")
print(f"  Ухудшилось: {worsened}")