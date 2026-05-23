"""
СТРЕСС-ТЕСТ ЕТИ: LEAVE-ONE-OUT ПРОВЕРКА ВСЕХ ФУНДАМЕНТАЛЬНЫХ КОНСТАНТ
"""

import math

import numpy as np
from scipy.optimize import minimize_scalar

# ФУНДАМЕНТАЛЬНЫЕ ПАРАМЕТРЫ Модели
K = 6.0
pi = math.pi
e = math.e
lnK = math.log(K)

# ЭКСПЕРИМЕНТАЛЬНЫЕ ЗНАЧЕНИЯ КОНСТАНТ

constants = {
    # Квантовые
    'ħ': 1.054571817e-34,  # Дж·с
    'h': 6.62607015e-34,  # Дж·с

    # Планковские
    't_P': 5.391247e-44,  # с
    'l_P': 1.616255e-35,  # м
    'm_P': 2.176434e-8,  # кг
    'E_P': 1.956082e9,  # Дж
    'T_P': 1.416784e32,  # К

    # Фундаментальные
    'c': 299792458,  # м/с
    'G': 6.67430e-11,  # м³/(кг·с²)
    'k_B': 1.380649e-23,  # Дж/К

    # Безразмерные
    'α': 1 / 137.035999084,  # постоянная тонкой структуры
    'm_e': 9.1093837015e-31,
    'r_e':2.8179402853e-15,

    'ep_0': 8.8725366415e-12,
    'mu_0': 1.25663706127e-6,

    'q_e': 1.602176634e-19,
    'm_proton': 1.67262192e-27,
    'm_muon': 1.883531627e-28,
    'm_tau': 3.167e-27,
    'm_pi_meson': 2.4880888e-28,
    'm_pi0_meson': 2.40609e-28,
    'm_k0_meson': 8.801929e-28,
    'm_DT': 3.3435837724e-27,
    'm_Λ_barion': 1.9901611e-27,
    'm_z_bozon': 1.62614e-25,
    'm_w_bozon': 1.43362e-25,
    'm_Higgs': 2.23319e-25,
    'm_D0': 3.32479e-27,
    'm_J_ψ': 5.52061e-27,
    'm_eta': 9.767732e-28,
    'm_Υ_1S': 1.68715e-26,
    'm_qu_u': 2.1650e-30,
    'm_qu_d': 4.7915e-30,
    'm_qu_s': 1.66e-28,
    'm_qu_c': 1.27e-27,
    'm_qu_b': 4.180e-27,
    'm_qu_t': 3.04e-25,
    'RIDBERG': 1.097373e7,
    'bor_radius': 5.29177210903e-11,
    'impedance': 376.730313,
    'Φ0_magnetic_stream': 2.06783366752e-15,
    'm_proton_to_m_electron': 1836.152673426,
    'm_tau_m_electron': 3477,
    'm_W_to_m_Z': 0.8815,
    'm_plank_to_m_e': 2.389e22,
    'compton_e': 2.426e-12,
    'compton_proton': 1.32140985396e-15,
    'm_Higgs_to_m_W': 1.558,
    'Lambda_cosmo': 1.08929e-52,
    'Einstein_constant': 2.07664746e-43,
    'vacuum_higgs': 4.388471e-25,
    'mu_lifetime': 2.1969811e-6,
    'tau_lifetime': 2.903e-13,
    'pion_lifetime': 2.6033e-8,
    'neutron_lifetime': 877.8,
    'kaon_lifetime': 1.2380e-8,
    'D_plus_lifetime': 1.040e-12,
    'B_plus_lifetime': 1.638e-12,
    'Λ_b_lifetime': 1.471e-12,
    'D0_lifetime': 4.101e-13,
    # энергии
    'h2_connection_energy': 2.178872e-18,
    # 'Gf': 1.436e-62,
    'Sigma_plus': 2.11933e-27,
    'm_neitrino': 1.783e-36,
    'Ksi_0': 2.34532e-27,
    'omega_minus': 2.9859e-27,
    'Ksi_plus': 4.3995e-27,
    'Omega0_c': 4.808e-27,
    'lambda_B0': 1.0023e-26,
    'Ksi_minus': 2.358e-27,
    'Sigma_minus': 2.132e-27,
    'phi_meson': 1.819e-27,
    'omega_meson': 1.394e-27,
    'eta_shtrih': 1.7086e-27,

    'rho_meson': 1.49e-27,
    'K_star_meson': 1.59e-27,
    #'B_meson': 9.40e-27,
    'eta_c_meson': 5.319e-27,
    'h_c_meson': 6.285e-27,
    'delta_meson': 2.196e-27,
    'B_c_meson': 1.1185e-26,
    'Ksi_pp_b_baryon': 6.543e-27,
    'K_plus': 8.802e-28,
    'B_meson': 9.413e-27,

}


# ФОРМУЛЫ ЕТИ (ПОЛНЫЙ НАБОР, ВСЕ ВЫВОДЯТСЯ)
def formulas(N):
    """Вычисляет ВСЕ константы через параметры ЕТИ (ни одна не задается вручную)"""
    lnN = math.log(N)
    N13 = N ** (1 / 3)

    # Вычисляем p для данного N
    p_val = 1 / (K * N ** (1 / 3))
    Kp = K * p_val

    m_neitrino = ((lnN) ** 2 * N ** (-1 / 3) * 2 ** (1 / 2)) / lnK
    Gf = (2 ** (1 / 2) * K ** 2 * lnN ** 2 * N ** (2 / 3) / (64 * pi ** 6 * lnN ** 13))
    h2_connection_energy = 8 * pi * lnN ** 10 * lnK ** 2 / (K ** (9 / 2) * N13)

    D0_lifetime = lnK / (2 * pi ** 2 * K ** 2 * (lnN) ** 4)
    Λ_b_lifetime = lnK * 2 ** (1 / 2) / lnN ** 5
    B_plus_lifetime = lnK * pi / 2 / lnN ** 5
    neutron_lifetime = pi * lnN
    kaon_lifetime = 4 / (K ** (3 / 2) * lnN ** 3)
    mu_lifetime = lnK / (K * 3 ** (1 / 2) * (lnN) ** 2)
    tau_lifetime = 1 / (2 * (lnN) ** 5)
    pion_lifetime = K ** 2 * 2 ** (1 / 2) * pi / (lnN) ** 4
    D_plus_lifetime = 1 / (pi ** (1 / 2) * K ** (5 / 2) * lnN ** 4)

    # Квантовые
    hbar_val = (lnN ** 3) / (K * N13)
    h_val = 2 * pi * hbar_val

    # Скорость света
    c_val = pi * (lnN ** 4) / (K ** 2 * lnK)

    # Планковская длина и время
    # lP_val = pi**2 * lnN**3 / (K**3 * lnK * N13)
    lP_val = 4 * lnN ** 2 * lnK / N13
    # tP_val = pi / (K * N13 * lnN)
    tP_val = 4 * K ** 2 * lnK ** 2 / (pi * N13 * lnN ** 2)

    # Планковская энергия
    # EP_val = (lnN ** 4) / pi
    EP_val = (lnN ** 5) * pi / (4 * K ** 3 * lnK ** 2)

    # Гравитационная постоянная (полностью эмерджентная)
    G_val = 16 * pi ** 3 * lnN ** 13 / (K ** 5 * lnK * N ** (1 / 3))

    # Планковская масса (выводится через другие константы)
    mP_val = K / (pi * 4 * lnN ** 3)

    # Планковская температура
    TP_val = 8 * pi * N13 / (lnN ** 4)

    # Постоянная Больцмана (через правильное p)
    k_B_val = (lnN ** 8) / (8 * pi ** 2 * N13)

    # Постоянная тонкой структуры
    alpha_val = 2 * lnK ** 2 / (pi * lnN)

    m_e_val = 4 * pi * lnN ** 4 / (K ** (1 / 2) * N ** (1 / 3))

    m_proton_val = math.sqrt(pi) * (lnN ** 6) / (K ** (3 / 2) * (N ** (1 / 3)))

    ep_0_val = N ** (1 / 3) / (8 * pi ** 3 * lnK * lnN ** 20)

    mu_0_val = (8 * pi * K ** 4 * lnK ** 3 * lnN ** 12) / N ** (1 / 3)

    q_e_val = 1.0 / (pi * K ** (3 / 2) * lnN ** 7)

    m_muon = 4 * pi ** 2 * lnN ** 5 / (K * 3 ** (1 / 2) * N ** (1 / 3))

    m_tau = pi ** (1 / 2) * (lnN ** 5) * (K ** 2) / (N ** (1 / 3))

    m_pi_meson = (lnN) ** 6 * 1 / (4 * pi ** 2) * N ** (-1 / 3) / 2 ** (1 / 2)

    m_pi0_meson = 2 * pi * K ** 3 * lnN ** 4 / N ** (1/3)

    m_k0_meson = (lnN ** 6 * 1 / (4 * (pi ** 2)) * (2 * pi) ** (1 / 2)) / N ** (1 / 3)
    m_DT = lnN ** 6 * (2 * pi) ** (1 / 2) / (K * 3 ** (1 / 2) * N ** (1 / 3))

    m_Λ_barion = (lnN ** 6 * N ** (-1 / 3) * 2 ** (1 / 2)) / (pi ** 2)
    m_z_bozon = (lnN ** 6) * 4 * (pi ** (5 / 2)) / (N ** (1 / 3) * K)
    m_w_bozon = 2 * pi ** 3 * lnN ** 6 / (N ** (1 / 3) * K)

    m_Higgs = lnN ** 6 * 4 * (pi ** (2)) / (N ** (1 / 3) * K ** (1 / 2))
    m_D0 = lnN ** 6 * ((2 * pi) ** (1 / 2)) / (N ** (1 / 3) * K * (3 ** (1 / 2)))

    m_J_ψ = lnN ** 5 * 8 * pi ** 2 * 2 ** (1 / 2) / N ** (1 / 3)

    m_eta = lnN ** 5 * 2 * pi ** 2 / N ** (1 / 3)


    m_Υ_1S = lnN ** 6 * 3 ** (1 / 2) / (2 ** (1 / 2) * N ** (1 / 3))

    m_qu_u = lnN ** 5 * 3 ** (1 / 2) / (4 * pi ** 2 * N ** (1 / 3))

    m_qu_d = lnN ** 5 / (K * 3 ** (1 / 2) * N ** (1 / 3))

    m_qu_s = lnN ** 5 * K / (pi ** ( 1/ 2) * N ** (1 / 3)) #lnN ** 4 * pi ** (7 / 2) / N ** (1 / 3)

    m_qu_c = lnN ** 6 * 2 * pi ** 2 / (K ** 3 * N ** (1 / 3))

    m_qu_b = lnN ** 6 * pi / (K * (3 ** (1 / 2)) * N ** (1 / 3))

    m_qu_t = lnN ** 6 * K ** 3 / (pi ** 2 * N ** (1 / 3))

    RIDBERG = 4 * lnN ** 3 * lnK ** 3 / (pi * K ** (3 / 2))

    #Энергия Ридберга(Хартри)
    #E_ridberg = 2 * 4 * lnN ** 3 * lnK ** 3 / (pi * K ** (3 / 2)) * (lnN ** 3) / (K * N13) * pi * (lnN ** 4) / (K ** 2 * lnK)
    r_e = 	lnK ** 3 * K ** (3/2) / (2 * pi**3 * lnN ** 6) #α λC/2π


    bor_radius = K ** (3 / 2) / (8 * pi * lnN ** 4 * lnK)

    impedance = 8 * K ** 2 * pi ** 2 * lnK ** 2 * lnN ** 16 / N ** (1 / 3)

    Φ0_magnetic_stream = lnN ** 10 * pi ** 2 * K ** (1 / 2) / (N ** (1 / 3))

    m_proton_to_m_electron = lnN ** 2 / (4 * pi ** (1 / 2) * K)

    m_tau_m_electron = K ** (5 / 2) * lnN / (4 * pi ** (1 / 2))

    # формулы масс частиц уже поделены, зафиксирован результат
    m_W_to_m_Z = pi ** (1 / 2) / 2

    m_plank_to_m_e = K ** (3 / 2) * N ** (1 / 3) / (16 * pi ** 2 * lnN ** 7)

    compton_e = K ** (3 / 2) * lnK / (2 * pi * lnN ** 5)

    compton_proton = 2 * K ** (5 / 2) * lnK / (pi ** (1 / 2) * lnN ** 7)

    m_Higgs_to_m_W = 2 * K ** (1 / 2) / pi

    Lambda_cosmo = lnN ** 12 / (pi ** (1 / 2) * N ** (2 / 3))

    Einstein_constant = 128 * K ** 3 * lnK ** 3 / (lnN ** 3 * N ** (1 / 3))

    vacuum_higgs = lnN ** 6 * 8 * pi ** (3 / 2) * 1 / (2 ** (1 / 2) * N ** (1 / 3))

    Sigma_plus = K * lnN ** 6 * 1 / (4 * pi ** (2)) / N ** (1 / 3)

    Ksi_0 = (lnN ** 6 * (2 * pi) ** (1 / 2) * N ** (-1 / 3)) / K ** (3 / 2)

    omega_minus = (lnN ** 6 * pi * N ** (-1 / 3)) / K ** (3 / 2)

    Lambda_plus = (lnN ** 6 * pi ** (1 / 2) * N ** (-1 / 3)) / K

    Ksi_plus = lnN ** 6 / (pi * N ** (1 / 3))

    # ((lnN)^5 * 4π² * √K) / N^(1/3)
    Omega0_c = K ** (1 / 2) * lnN ** 5 * 4 * pi ** 2 / N ** (1 / 3)

    lambda_B0 = lnN ** 6 * pi**(1/2) / (K ** (1 / 2) * N ** (1 / 3))

    Ksi_minus = (lnN ** 6 * (2 * pi) ** (1 / 2) * N ** (-1 / 3)) / K ** (3 / 2)

    Sigma_minus = (lnN ** 6 * K * N ** (-1 / 3)) / (4 * pi ** 2)

    phi_meson = lnN ** 5 * (2 * pi) ** (1 / 2) * K ** (3 / 2) / N ** (1 / 3)

    omega_meson = lnN ** 5 * 2 * pi ** 2 * 2 ** (1 / 2) / N ** (1 / 3)

    eta_shtrih = lnN ** 5 * K ** 3 * 1 / (2 * pi) / N ** (1 / 3)

    m_rho = (math.sqrt(3)) * (pi ** (5 / 2)) * (lnN ** 5) / N ** (1 / 3)

    m_K_star = 2 * pi ** (-5 / 2) * lnN ** 6 / N13

    # прежняя формула
    m_B = (pi ** 2) * (lnN ** 6) / ((2 ** (3 / 2)) * (3 ** (3 / 2)) * N13)
    #m_B = lnN ** 6 * (3 ** (1/2))  / ((2*pi)**(1/2) * N13)

    m_eta_c = (math.sqrt(2)) ** 4 * (math.sqrt(3)) ** 6 * (lnN ** 5) / N13

    m_h_c = math.sqrt(2) * (lnN ** 6) / (pi * N13)

    m_delta = (lnN ** 6) / (2 * pi * N13)

    m_B_c = (math.sqrt(2)) ** 6 * (math.sqrt(3)) ** 4 * pi * (lnN ** 5) / N13

    m_B_s = (math.sqrt(2)) ** 4 * math.sqrt(3) * (lnN ** 6) / N13

    m_Ksi_pp_b = math.sqrt(2) * (lnN ** 6) / (3 * N13)

    m_K_plus = lnN**6 * N**(-1/3) / (2*pi)**(3/2)



    return {key: value * 1 for key, value in {
        'ħ': hbar_val,
        'h': h_val,
        't_P': tP_val,
        'l_P': lP_val,
        'm_P': mP_val,
        'E_P': EP_val,
        'T_P': TP_val,
        'c': c_val,
        'G': G_val,
        'k_B': k_B_val,
        'α': alpha_val,
        'm_e': m_e_val,
        'r_e':r_e,
        'ep_0': ep_0_val,
        'mu_0': mu_0_val,
        'q_e': q_e_val,
        'm_proton': m_proton_val,
        'm_muon': m_muon,
        'm_tau': m_tau,
        'm_pi_meson': m_pi_meson,
        'm_pi0_meson': m_pi0_meson,
        'm_k0_meson': m_k0_meson,
        'm_DT': m_DT,
        'm_Λ_barion': m_Λ_barion,
        'm_z_bozon': m_z_bozon,
        'm_w_bozon': m_w_bozon,
        'm_Higgs': m_Higgs,
        'm_D0': m_D0,
        'm_J_ψ': m_J_ψ,
        'm_eta': m_eta,
        'm_Υ_1S': m_Υ_1S,
        'm_qu_u': m_qu_u,
        'm_qu_d': m_qu_d,
        'm_qu_s': m_qu_s,
        'm_qu_c': m_qu_c,
        'm_qu_b': m_qu_b,
        'm_qu_t': m_qu_t,
        'RIDBERG': RIDBERG,
        'bor_radius': bor_radius,
        'impedance': impedance,
        'Φ0_magnetic_stream': Φ0_magnetic_stream,
        'm_proton_to_m_electron': m_proton_to_m_electron,
        'm_tau_m_electron': m_tau_m_electron,
        'm_W_to_m_Z': m_W_to_m_Z,
        'm_plank_to_m_e': m_plank_to_m_e,
        'compton_e': compton_e,
        'compton_proton': compton_proton,
        'm_Higgs_to_m_W': m_Higgs_to_m_W,
        'Lambda_cosmo': Lambda_cosmo,
        'Einstein_constant': Einstein_constant,
        'vacuum_higgs': vacuum_higgs,
        'mu_lifetime': mu_lifetime,
        'tau_lifetime': tau_lifetime,
        'pion_lifetime': pion_lifetime,
        'D_plus_lifetime': D_plus_lifetime,
        'kaon_lifetime': kaon_lifetime,
        'neutron_lifetime': neutron_lifetime,
        'B_plus_lifetime': B_plus_lifetime,
        'Λ_b_lifetime': Λ_b_lifetime,
        'D0_lifetime': D0_lifetime,
        'h2_connection_energy': h2_connection_energy,
        'm_neitrino': m_neitrino,
        'Sigma_plus': Sigma_plus,
        'Ksi_0': Ksi_0,
        'omega_minus': omega_minus,
        'Lambda_plus': Lambda_plus,
        'Ksi_plus': Ksi_plus,
        'Omega0_c': Omega0_c,
        'lambda_B0': lambda_B0,
        'Ksi_minus': Ksi_minus,
        'Sigma_minus': Sigma_minus,
        'phi_meson': phi_meson,
        'omega_meson': omega_meson,
        'eta_shtrih': eta_shtrih,
        'rho_meson': m_rho,
        'K_star_meson': m_K_star,
        'B_meson': m_B,
        'eta_c_meson': m_eta_c,
        'h_c_meson': m_h_c,
        'delta_meson': m_delta,
        'B_c_meson': m_B_c,
        'B_s_meson': m_B_s,
        'Ksi_pp_b_baryon': m_Ksi_pp_b,
        'K_plus': m_K_plus,
        # 'Gf': Gf
    }.items()}


def find_optimal_N(N_initial=4.183e121):
    """Находит N с минимальной суммарной лог-ошибкой по всем константам"""

    def total_log_error(ln_N):
        """Суммарная логарифмическая ошибка для всех констант"""
        N = np.exp(ln_N)
        pred = formulas(N)
        total = 0.0

        for key in constants.keys():
            if key in pred:
                ratio = pred[key] / constants[key]
                total += (np.log(ratio)) ** 2

        return total

    # Начальное приближение
    ln_N0 = np.log(N_initial)

    # Минимизация
    result = minimize_scalar(
        total_log_error,
        bracket=(ln_N0 * 0.9, ln_N0 * 1.1),  # ±10% от начального
        method='brent',
        options={'xtol': 1e-10}
    )

    N_opt = np.exp(result.x)
    min_error = result.fun

    return N_opt, min_error


# ЛОГ-ОШИБКА ДЛЯ ПРОИЗВОЛЬНОГО НАБОРА
def log_error_subset(N, subset_keys):
    """Логарифмическая ошибка для поднабора констант"""
    pred = formulas(N)
    total = 0.0

    for key in subset_keys:
        if key in pred and key in constants:
            ratio = pred[key] / constants[key]
            total += (math.log(ratio)) ** 2

    return total


# ПОИСК N ПО ПОДНАБОРУ
def fit_N(subset_keys, N0):
    """Находит оптимальное N для поднабора констант"""
    lnN0 = math.log(N0)

    result = minimize_scalar(
        lambda x: log_error_subset(math.exp(x), subset_keys),
        bracket=(lnN0 - 0.1 * lnN0, lnN0 + 0.1 * lnN0),
        method='brent'
    )

    lnN_opt = result.x
    N_opt = math.exp(lnN_opt)

    return N_opt, result.fun


# ОТНОСИТЕЛЬНАЯ И ЛОГ-ОШИБКА
def relative_error(pred, true):
    if true == 0:
        return float('inf')
    return abs(pred - true) / true * 100


def log_error_single(pred, true):
    if pred <= 0 or true <= 0:
        return float('inf')
    return abs(math.log(pred / true))


# ПОЛНЫЙ АНАЛИЗ ВСЕХ КОНСТАНТ
def analyze_all_constants(N0):
    """Вычисляет ошибки для всех констант при N0"""
    pred = formulas(N0)
    p_val = N0 ** (1 / 3) / K

    print("ПОЛНЫЙ АНАЛИЗ ВСЕХ КОНСТАНТ ПРИ N0")
    print(f"\n  p (вариационное) = {p_val:.6e}")
    print(f"  Kp = {K * p_val:.6e}")

    results = []
    for name in constants.keys():
        p_val = pred[name]
        t_val = constants[name]
        rel_err = relative_error(p_val, t_val)

        if rel_err < 0.1:
            status = "⭐⭐⭐"
        elif rel_err < 1.0:
            status = "⭐⭐"
        elif rel_err < 5.0:
            status = "⭐"
        else:
            status = "⚠️"

        results.append((name, p_val, t_val, rel_err, status))

    # Сортируем по ошибке
    results.sort(key=lambda x: x[3])

    print(f"\n{'Константа':<8} {'Предсказание':<18} {'Эксперимент':<18} {'Ошибка %':<12} {'Статус':<6}")
    print("-" * 75)

    for name, p_val, t_val, rel_err, status in results:
        print(f"{name:<8} {p_val:<18.6e} {t_val:<18.6e} {rel_err:<12.6f} {status:<6}")

    return results


# LEAVE-ONE-OUT ТЕСТ
def leave_one_out_test(N0):
    results = {}

    print("LEAVE-ONE-OUT ТЕСТ ЕТИ")
    print(f"\nТеоретическое N0 = {N0:.4e}")
    print(f"ln N0 = {math.log(N0):.6f}")
    # print(f"p(N0) = {p:.6e}\n")

    for excluded in constants.keys():
        # Формируем поднабор
        subset = [k for k in constants.keys() if k != excluded]

        # Фитим N
        N_fit, err_fit = fit_N(subset, N0)

        # Предсказание исключённой константы
        pred = formulas(N_fit)[excluded]
        true = constants[excluded]

        rel_err = relative_error(pred, true)
        log_err = log_error_single(pred, true)

        results[excluded] = {
            'N_fit': N_fit,
            'fit_error': err_fit,
            'predicted': pred,
            'true': true,
            'rel_error_%': rel_err,
            'log_error': log_err,
        }

        # Определяем статус
        if rel_err < 0.1:
            status = "✅✅✅ ОТЛИЧНО"
        elif rel_err < 1.0:
            status = "✅✅ ХОРОШО"
        elif rel_err < 5.0:
            status = "✅ ПРИЕМЛЕМО"
        else:
            status = "⚠️ ТРЕБУЕТ УТОЧНЕНИЯ"

        print(f"\n--- Исключена: {excluded} ---")
        print(f"N_fit = {N_fit:.4e}  (ΔN/N0 = {(N_fit - N0) / N0 * 100:+.2f}%)")
        print(f"Ошибка фитинга = {err_fit:.6e}")
        print(f"Предсказано: {pred:.6e}")
        print(f"Реальное:    {true:.6e}")
        print(f"Отн. ошибка: {rel_err:.6f}%")
        print(f"Статус:      {status}")

    return results


# СВОДКА ПО ГРУППАМ КОНСТАНТ
def group_summary(N0):
    """Группирует константы по категориям"""
    pred = formulas(N0)

    groups = {
        'Квантовые': ['ħ', 'h'],
        'Планковские': ['t_P', 'l_P', 'm_P', 'E_P', 'T_P'],
        'Фундаментальные': ['c', 'G', 'k_B'],
        'Безразмерные': ['α'],
    }

    print("СВОДКА ПО ГРУППАМ КОНСТАНТ")

    for group_name, const_list in groups.items():
        errors = []
        for name in const_list:
            if name in constants:
                rel_err = relative_error(pred[name], constants[name])
                errors.append(rel_err)

        if errors:
            avg_err = sum(errors) / len(errors)
            max_err = max(errors)
            min_err = min(errors)
            print(f"\n{group_name}:")
            print(f"  Средняя ошибка: {avg_err:.4f}%")
            print(f"  Максимальная:   {max_err:.4f}%")
            print(f"  Минимальная:    {min_err:.4f}%")


# ГЛАВНЫЙ ЗАПУСК
def main():
    print("СТРЕСС-ТЕСТ ЕДИНОЙ ТЕОРИИ ИНФОРМАЦИИ (ИСПРАВЛЕННАЯ ВЕРСИЯ)")

    # Теоретическое N
    # N0 =  4.197668e121 #4.179e121 #4.475947352678e+121
    # N0 = math.exp(280.11151) #math.exp(280.04176) #math.exp(280.098)
    # N0 = 4.2064263158547185e+121

    N_math = 4.475947e+121
    N_phys = math.log(N_math) - math.pi * K / math.log(N_math)
    N0 = math.exp(N_phys)

    # Параметры
    print(f"\nПАРАМЕТРЫ ЕТИ:")
    print(f"  K = {K}")
    print(f"  ln K = {lnK:.6f}")
    print(f"  N0 = {N0:.4e}")
    print(f"  ln N0 = {math.log(N0):.6f}")

    # Полный анализ при N0
    analyze_all_constants(N0)

    # Сводка по группам
    group_summary(N0)

    # Leave-One-Out тест
    results = leave_one_out_test(N0)

    # Итог LOO
    print("ИТОГ LEAVE-ONE-OUT")

    success_count = 0
    for key, res in results.items():
        if res['rel_error_%'] < 1.0:
            success_count += 1
            status = "✅"
        elif res['rel_error_%'] < 5.0:
            status = "🟡"
        else:
            status = "❌"
        print(f"{status} {key:<8}: ошибка"
              f" = {res['rel_error_%']:.6f}%  (N_fit/N0 = {res['N_fit'] / N0:.6f})")

    print(f"\nУспешно предсказано: {success_count}/{len(results)} констант")

    # Метод 1: Быстрый поиск
    N_opt_1, error_1 = find_optimal_N()
    print(f"\nБыстрый поиск:")
    print(f"  N_оптимальное = {N_opt_1:.6e}")
    print(f"  Минимальная ошибка = {error_1:.6e}")
    print(f"  Отклонение от N0 = {(N_opt_1 - N0) / N0 * 100:.2f}%")

    # МИНИМАЛЬНЫЙ БЛОК ДЛЯ ВСТАВКИ
    print("\n" + "=" * 80)
    print("ОТКЛОНЕНИЯ КОНСТАНТ (±%)")
    print("=" * 80)
    pred = formulas(N0)
    for name in constants:
        if name in pred:
            dev = ((pred[name] - constants[name]) / constants[name]) * 100
            print(f"{name:<25} {constants[name]:<18.6e} {pred[name]:<18.6e} {dev:>+10.6f}%")
    # Финальный вердикт
    print("ФИНАЛЬНЫЙ ВЕРДИКТ")

    if success_count >= len(results) * 0.8:
        print("✅ СТРЕСС-ТЕСТ ПРОЙДЕН!")
        print("   ЕТИ демонстрирует высокую предсказательную силу.")
        print("   ВСЕ константы выводятся из первых принципов.")
    elif success_count >= len(results) * 0.5:
        print("🟡 СТРЕСС-ТЕСТ ПРОЙДЕН ЧАСТИЧНО")
        print("   Большинство констант предсказывается корректно.")
    else:
        print("❌ СТРЕСС-ТЕСТ НЕ ПРОЙДЕН")
        print("   Требуется пересмотр модели.")

    return results


if __name__ == "__main__":
    results = main()
