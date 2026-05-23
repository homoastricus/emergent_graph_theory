import pandas as pd

import math

N = 4.1977e+121
K = 6.0
pi = math.pi
e = math.e
lnK = math.log(K)
lnN = math.log(N)

D0_lifetime = lnK / (2 * pi ** 2 * K ** 2 * (lnN) ** 4)
Λ_b_lifetime = lnK * 2 ** (1 / 2) / lnN ** 5
B_plus_lifetime = lnK * pi / 2 / lnN ** 5
neutron_lifetime = 2 ** (1 / 2) * N ** (1 / 12) / (lnN) ** 3
kaon_lifetime = 4 / (K ** (3 / 2) * lnN ** 3)
mu_lifetime = lnK / (K * 3 ** (1 / 2) * (lnN) ** 2)
tau_lifetime = 1 / (2 * (lnN) ** 5)
pion_lifetime = K ** 2 * 2 ** (1 / 2) * pi / (lnN) ** 4
D_plus_lifetime = 1 / (pi ** (1 / 2) * K ** (5 / 2) * lnN ** 4)

# lifetimes
Λ_b_2_B_plus_lifetime = 2 ** (1 / 2) / pi

# masses
m_qu_c_2_m_cu_t = 2 * (pi ** 4) / K ** 6
m_Λ_barion_2_m_k0_meson = 4 / (pi ** (1 / 2))

m_z_bozon_2_m_w_bozon = 2 / (pi ** (1 / 2))
m_w_bozon_2_m_Higgs = pi / (2 * 6 ** (1 / 2))

m_proton_2_m_qu_b = 1 / (math.sqrt(2) * math.sqrt(pi))
# m_Λ_barion_2_m_z_bozon = 3 * (2 ** (1 / 2)) / 2 * (pi ** (1 / 2))
m_Λ_barion_2_m_z_bozon = 3 / ((2 * pi) ** (1 / 2))
m_DT_2_m_Λ_barion = (pi) ** (3 / 2) / 6 * 3 ** (1 / 2)
m_Higgs_2_m_D0 = 12 * pi ** (3 / 2)
m_D0_2_m_Υ_1S = (pi ** (1 / 2)) / 3 * 6
m_Υ_1S_2_m_qu_c = 3 ** (1 / 2) * (6 ** 3) / ((2 ** (1 / 2)) * 2 * pi ** 2)
m_qu_c_2_m_qu_b = 2 * pi * (3 ** (1 / 2)) / (6 ** 2)
m_qu_t_2_m_qu_b = 6 ** 2 * (3 ** (1 / 2)) / pi ** 3

m_qu_t_2_m_pi_meson = 6 ** 3 * 4 / 2 ** (1 / 2)
m_w_bozon_2_m_proton_ = 2 * pi ** (3 / 2) * 6 ** (1 / 2)
m_pi_meson_2_m_k0_meson = pi ** (1 / 2)
m_z_bozon_2_m_proton = 4 * (pi ** 2) * 6 ** (1 / 2)
m_muon_2_m_tau = 2 * pi ** 3 / 2 * 3 ** (1 / 2) / 9

m_J_ψ_2_m_tau = 8 * pi ** (3 / 2) * 2 ** (1 / 2) / 6 ** 2
m_J_ψ_2_m_eta = 4 * 2 ** (1 / 2)
m_eta_2_m_qu_d = 12 * pi ** 2 * 3 ** (1 / 2)
m_tau_2_m_eta = 18 / pi ** (3 / 2)
m_eta_2_m_qu_u = 4 * pi ** 2 / 3 ** (1 / 2)
m_qu_d_2_m_qu_u = 8 * pi ** 2
m_muon_2_m_qu_u = 8 * pi ** 4 / 9

m_pi0_meson_2_m_qu_s = 2 * 6 ** 2 / pi ** (5 / 2)
m_pi0_meson_2_m_e_val = 6 ** (7 / 2) / 2
m_qu_s_2_m_e_val = pi ** (5 / 2) * 6 ** (1 / 2) / 4

m_e_val = 4 * pi * lnN ** 4 / (K ** (1 / 2) * N ** (1 / 3))

m_proton_val = math.sqrt(pi) * (lnN ** 6) / (K ** (3 / 2) * (N ** (1 / 3)))

m_k0_meson = (lnN ** 6 * 1 / (4 * (pi ** 2)) * (2 * pi) ** (1 / 2)) / N ** (1 / 3)
m_DT = lnN ** 6 * (2 * pi) ** (1 / 2) / (K * 3 ** (1 / 2) * N ** (1 / 3))

m_Λ_barion = (lnN ** 6 * N ** (-1 / 3) * 2 ** (1 / 2)) / (pi ** 2)
m_z_bozon = (lnN ** 6) * 4 * (pi ** (5 / 2)) / (N ** (1 / 3) * K)
m_w_bozon = 2 * pi ** 3 * lnN ** 6 / (N ** (1 / 3) * K)

m_Higgs = lnN ** 6 * 4 * (pi ** (2)) / (N ** (1 / 3) * K ** (1 / 2))
m_D0 = lnN ** 6 * ((2 * pi) ** (1 / 2)) / (N ** (1 / 3) * K * (3 ** (1 / 2)))

m_Υ_1S = lnN ** 6 * 3 ** (1 / 2) / (2 ** (1 / 2) * N ** (1 / 3))

m_qu_c = lnN ** 6 * 2 * pi ** 2 / (K ** 3 * N ** (1 / 3))

m_qu_b = lnN ** 6 * pi / (K * (3 ** (1 / 2)) * N ** (1 / 3))

m_qu_t = lnN ** 6 * K ** 3 / (pi ** 2 * N ** (1 / 3))

m_pi_meson = lnN ** 6 * 1 / (4 * pi ** 2) * N ** (-1 / 3) / 2 ** (1 / 2)

m_muon = 4 * pi ** 2 * lnN ** 5 / (K * 3 ** (1 / 2) * N ** (1 / 3))

m_tau = pi ** (1 / 2) * (lnN ** 5) * (K ** 2) / (N ** (1 / 3))

m_pi0_meson = 2 * pi * K ** 3 * lnN ** 4 / N ** (1 / 3)

m_J_ψ = lnN ** 5 * 8 * pi ** 2 * 2 ** (1 / 2) / N ** (1 / 3)

m_eta = lnN ** 5 * 2 * pi ** 2 / N ** (1 / 3)

m_qu_u = lnN ** 5 * 3 ** (1 / 2) / (4 * pi ** 2 * N ** (1 / 3))

m_qu_d = lnN ** 5 / (K * 3 ** (1 / 2) * N ** (1 / 3))

m_qu_s = lnN ** 4 * pi ** (7 / 2) / N ** (1 / 3)

# Новые соотношения масс
m_Higgs_2_m_z_bozon = 1 / (pi ** (1 / 2) * K ** (1 / 2))
m_Higgs_2_m_w_bozon = 2 / (pi * K ** (1 / 2))
m_DT_2_m_k0_meson = 4 * pi ** 2 / (K * 3 ** (1 / 2))
m_DT_2_m_proton = 2 ** (1 / 2) * K ** (1 / 2) / 3 ** (1 / 2)
m_qu_c_2_m_proton = 2 * pi ** (3 / 2) / K ** (3 / 2)
m_muon_2_m_qu_d = 4 * pi ** 2
m_tau_2_m_qu_d = pi ** (1 / 2) * K ** 3 * 3 ** (1 / 2)
m_J_ψ_2_m_qu_d = 8 * pi ** 2 * 2 ** (1 / 2) * K * 3 ** (1 / 2)
m_muon_2_m_eta = 2 / (K * 3 ** (1 / 2))
m_Υ_1S_2_m_proton = 3 ** (1 / 2) * K ** (3 / 2) / (2 ** (1 / 2) * pi ** (1 / 2))

constants = {
    'm_e': 9.1093837015e-31,
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
    'm_qu_s': 9.635e-30,
    'm_qu_c': 1.27e-27,
    'm_qu_b': 4.180e-27,
    'm_qu_t': 3.04e-25,
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
    'm_neitrino': 1.783e-36,
    'Sigma_plus': 2.11933e-27,
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
    'Lambda_plus': 4.0737e-27
}

# Словарь соответствия переменных и имен
var_to_name = {
    'm_e_val': 'm_e',
    'm_proton_val': 'm_proton',
    'm_k0_meson': 'm_k0_meson',
    'm_DT': 'm_DT',
    'm_Λ_barion': 'm_Λ_barion',
    'm_z_bozon': 'm_z_bozon',
    'm_w_bozon': 'm_w_bozon',
    'm_Higgs': 'm_Higgs',
    'm_D0': 'm_D0',
    'm_Υ_1S': 'm_Υ_1S',
    'm_qu_c': 'm_qu_c',
    'm_qu_b': 'm_qu_b',
    'm_qu_t': 'm_qu_t',
    'm_pi_meson': 'm_pi_meson',
    'm_muon': 'm_muon',
    'm_tau': 'm_tau',
    'm_pi0_meson': 'm_pi0_meson',
    'm_J_ψ': 'm_J_ψ',
    'm_eta': 'm_eta',
    'm_qu_u': 'm_qu_u',
    'm_qu_d': 'm_qu_d',
    'm_qu_s': 'm_qu_s',
    'mu_lifetime': 'mu_lifetime',
    'tau_lifetime': 'tau_lifetime',
    'pion_lifetime': 'pion_lifetime',
    'neutron_lifetime': 'neutron_lifetime',
    'kaon_lifetime': 'kaon_lifetime',
    'D_plus_lifetime': 'D_plus_lifetime',
    'B_plus_lifetime': 'B_plus_lifetime',
    'Λ_b_lifetime': 'Λ_b_lifetime',
    'D0_lifetime': 'D0_lifetime',
    # 'm_neitrino': 'm_n
    'm_Higgs_2_m_w_bozon': 'm_Higgs_2_m_w_bozon',
    'm_DT_2_m_k0_meson': 'm_DT_2_m_k0_meson',
    'm_DT_2_m_proton': 'm_DT_2_proton',
    'm_qu_c_2_m_proton': 'm_qu_c_2_proton',
    'm_muon_2_m_qu_d': 'm_muon_2_m_qu_d',
    'm_tau_2_m_qu_d': 'm_tau_2_m_qu_d',
    'm_J_ψ_2_m_qu_d': 'm_J_ψ_2_m_qu_d',
    'm_muon_2_m_eta': 'm_muon_2_m_eta',
    'm_Υ_1S_2_m_proton': 'm_Υ_1S_2_m_proton',
    'm_muon_2_m_tau': 'm_muon_2_m_tau'
}

# Список соотношений и их формул
relations = [

    {
        'name': 'Отношение масс: бозон Хиггса / Z-бозон',
        'formula': 'm_Higgs_2_m_z_bozon',
        'expr': '(K / pi) ** (1/2)',
        'first_var': 'm_Higgs',
        'second_var': 'm_z_bozon'
    },
    {
        'name': 'Отношение масс: бозон Хиггса / W-бозон',
        'formula': 'm_Higgs_2_m_w_bozon',
        'expr': '2 * math.sqrt(K) / pi',
        'first_var': 'm_Higgs',
        'second_var': 'm_w_bozon'
    },
    {
        'name': 'Отношение масс: DT / K0-мезон',
        'formula': 'm_DT_2_m_k0_meson',
        'expr': '4 * pi**2 / (K * 3**(1/2))',
        'first_var': 'm_DT',
        'second_var': 'm_k0_meson'
    },
    {
        'name': 'Отношение масс: DT / протон',
        'formula': 'm_DT_2_m_proton',
        'expr': '2**(1/2) * K**(1/2) / 3**(1/2)',
        'first_var': 'm_DT',
        'second_var': 'm_proton'
    },
    {
        'name': 'Отношение масс: c-кварк / протон',
        'formula': 'm_qu_c_2_m_proton',
        'expr': '2 * pi**(3/2) / K**(3/2)',
        'first_var': 'm_qu_c',
        'second_var': 'm_proton'
    },
    {
        'name': 'Отношение масс: мюон / d-кварк',
        'formula': 'm_muon_2_m_qu_d',
        'expr': '4 * pi**2',
        'first_var': 'm_muon',
        'second_var': 'm_qu_d'
    },
    {
        'name': 'Отношение масс: тау / d-кварк',
        'formula': 'm_tau_2_m_qu_d',
        'expr': 'pi**(1/2) * K**3 * 3**(1/2)',
        'first_var': 'm_tau',
        'second_var': 'm_qu_d'
    },
    {
        'name': 'Отношение масс: J/ψ / d-кварк',
        'formula': 'm_J_ψ_2_m_qu_d',
        'expr': '8 * pi**2 * 2**(1/2) * K * 3**(1/2)',
        'first_var': 'm_J_ψ',
        'second_var': 'm_qu_d'
    },
    {
        'name': 'Отношение масс: мюон / η',
        'formula': 'm_muon_2_m_eta',
        'expr': '2 / (K * 3**(1/2))',
        'first_var': 'm_muon',
        'second_var': 'm_eta'
    },
    {
        'name': 'Отношение масс: Υ(1S) / протон',
        'formula': 'm_Υ_1S_2_m_proton',
        'expr': '3**(1/2) * K**(3/2) / (2**(1/2) * pi**(1/2))',
        'first_var': 'm_Υ_1S',
        'second_var': 'm_proton'
    },

    {
        'name': 'Отношение масс: c-кварк / t-кварк',
        'formula': 'm_qu_c_2_m_cu_t',
        'expr': '2*(pi**4)/K**6',
        'first_var': 'm_qu_c',
        'second_var': 'm_qu_t'
    },
    {
        'name': 'Отношение масс: Λ-барион / K0-мезон',
        'formula': 'm_Λ_barion_2_m_k0_meson',
        'expr': '4 / (pi ** (1/2))',
        'first_var': 'm_Λ_barion',
        'second_var': 'm_k0_meson'
    },
    {
        'name': 'Отношение масс: Z-бозон / W-бозон',
        'formula': 'm_z_bozon_2_m_w_bozon',
        'expr': '2/(pi ** (1/2))',
        'first_var': 'm_z_bozon',
        'second_var': 'm_w_bozon'
    },
    {
        'name': 'Отношение масс: W-бозон / бозон Хиггса',
        'formula': 'm_w_bozon_2_m_Higgs',
        'expr': 'pi / (2 * K ** (1 / 2))',
        'first_var': 'm_w_bozon',
        'second_var': 'm_Higgs'
    },
    {
        'name': 'Отношение масс: протон / b-кварк',
        'formula': 'm_proton_2_m_qu_b',
        'expr': '1/ (math.sqrt(2) * math.sqrt(pi))',
        'first_var': 'm_proton',
        'second_var': 'm_qu_b'
    },
    {
        'name': 'Отношение масс: Λ-барион / Z-бозон',
        'formula': 'm_Λ_barion_2_m_z_bozon',
        'expr': 'K * 2**(1/2) / (4 * pi**(9/2))',
        'first_var': 'm_Λ_barion',
        'second_var': 'm_z_bozon'
    },
    {
        'name': 'Отношение масс: DT / Λ-барион',
        'formula': 'm_DT_2_m_Λ_barion',
        'expr': '(pi) ** (5 / 2) / (K * 3 ** (1 / 2))',
        'first_var': 'm_DT',
        'second_var': 'm_Λ_barion'
    },
    {
        'name': 'Отношение масс: бозон Хиггса / D0',
        'formula': 'm_Higgs_2_m_D0',
        'expr': '12 * pi ** (3/2)',
        'first_var': 'm_Higgs',
        'second_var': 'm_D0'
    },
    {
        'name': 'Отношение масс: D0 / Υ(1S)',
        'formula': 'm_D0_2_m_Υ_1S',
        'expr': '(pi ** (1 / 2))/9',
        'first_var': 'm_D0',
        'second_var': 'm_Υ_1S'
    },
    {
        'name': 'Отношение масс: Υ(1S) / c-кварк',
        'formula': 'm_Υ_1S_2_m_qu_c',
        'expr': '3 ** (1 / 2) * (K**3)/ ((2 ** (1 / 2)) * 2*pi**2)',
        'first_var': 'm_Υ_1S',
        'second_var': 'm_qu_c'
    },
    {
        'name': 'Отношение масс: c-кварк / b-кварк',
        'formula': 'm_qu_c_2_m_qu_b',
        'expr': '2 * pi * (3 ** (1 / 2))/ (K ** 2 )',
        'first_var': 'm_qu_c',
        'second_var': 'm_qu_b'
    },
    {
        'name': 'Отношение масс: t-кварк / b-кварк',
        'formula': 'm_qu_t_2_m_qu_b',
        'expr': 'K**4 * math.sqrt(3) / pi**3',
        'first_var': 'm_qu_t',
        'second_var': 'm_qu_b'
    },
    {
        'name': 'Отношение масс: t-кварк / π-мезон',
        'formula': 'm_qu_t_2_m_pi_meson',
        'expr': '4 * 2**(1/2) * K**3',
        'first_var': 'm_qu_t',
        'second_var': 'm_pi_meson'
    },
    {
        'name': 'Отношение масс: W-бозон / протон',
        'formula': 'm_w_bozon_2_m_proton_',
        'expr': '2 * pi**(5/2) * math.sqrt(K)',
        'first_var': 'm_w_bozon',
        'second_var': 'm_proton'
    },
    {
        'name': 'Отношение масс: π-мезон / K0-мезон',
        'formula': 'm_pi_meson_2_m_k0_meson',
        'expr': '1 / (2*(pi) ** (1 / 2))',
        'first_var': 'm_pi_meson',
        'second_var': 'm_k0_meson'
    },
    {
        'name': 'Отношение масс: Z-бозон / протон',
        'formula': 'm_z_bozon_2_m_proton',
        'expr': '4 * (pi ** 2) * K ** (1 / 2)',
        'first_var': 'm_z_bozon',
        'second_var': 'm_proton'
    },
    {
        'name': 'Отношение масс: мюон / тау',
        'formula': 'm_muon_2_m_tau',
        'expr': '4 * pi ** (3/2) / (K**3 * 3**(1/2))',
        'first_var': 'm_muon',
        'second_var': 'm_tau'
    },
    {
        'name': 'Отношение масс: J/ψ / тау',
        'formula': 'm_J_ψ_2_m_tau',
        'expr': '8 * pi ** (3 / 2) * 2 ** (1 / 2) / K**2',
        'first_var': 'm_J_ψ',
        'second_var': 'm_tau'
    },
    {
        'name': 'Отношение масс: J/ψ / η-мезон',
        'formula': 'm_J_ψ_2_m_eta',
        'expr': '4 * 2 ** (1 / 2)',
        'first_var': 'm_J_ψ',
        'second_var': 'm_eta'
    },
    {
        'name': 'Отношение масс: η-мезон / d-кварк',
        'formula': 'm_eta_2_m_qu_d',
        'expr': '12 * pi ** 2 * 3 ** (1 / 2)',
        'first_var': 'm_eta',
        'second_var': 'm_qu_d'
    },
    {
        'name': 'Отношение масс: тау / η-мезон',
        'formula': 'm_tau_2_m_eta',
        'expr': '18/pi ** (3 / 2)',
        'first_var': 'm_tau',
        'second_var': 'm_eta'
    },
    {
        'name': 'Отношение масс: η-мезон / u-кварк',
        'formula': 'm_eta_2_m_qu_u',
        'expr': '8 * pi**4 / math.sqrt(3)',
        'first_var': 'm_eta',
        'second_var': 'm_qu_u'
    },
    {
        'name': 'Отношение масс: d-кварк / u-кварк',
        'formula': 'm_qu_d_2_m_qu_u',
        'expr': '4 * pi**2 / (3 * K)',
        'first_var': 'm_qu_d',
        'second_var': 'm_qu_u'
    },
    {
        'name': 'Отношение масс: мюон / u-кварк',
        'formula': 'm_muon_2_m_qu_u',
        'expr': '8 * pi ** 4 / 9',
        'first_var': 'm_muon',
        'second_var': 'm_qu_u'
    },
    {
        'name': 'Отношение масс: π0-мезон / s-кварк',
        'formula': 'm_pi0_meson_2_m_qu_s',
        'expr': '2 * K**3 / pi**(5/2)',
        'first_var': 'm_pi0_meson',
        'second_var': 'm_qu_s'
    },
    {
        'name': 'Отношение масс: π0-мезон / электрон',
        'formula': 'm_pi0_meson_2_m_e_val',
        'expr': 'K**(7/2) / 2',
        'first_var': 'm_pi0_meson',
        'second_var': 'm_e'
    },
    {
        'name': 'Отношение масс: s-кварк / электрон',
        'formula': 'm_qu_s_2_m_e_val',
        'expr': 'pi ** (5 / 2) * K ** (1 / 2) / 4',
        'first_var': 'm_qu_s',
        'second_var': 'm_e'
    },
    {
        'name': 'Отношение времен жизни: Λ_b / B+',
        'formula': 'Λ_b_2_B_plus_lifetime',
        'expr': '2*2**(1/2) / pi',
        'first_var': 'Λ_b_lifetime',
        'second_var': 'B_plus_lifetime'
    },
    # new
    {
        'name': 'Отношение масс: Sigma_plus / протон',
        'formula': 'Sigma_plus_2_m_proton',
        'expr': 'K**(5/2) / (4 * pi**(5/2))',
        'first_var': 'Sigma_plus',
        'second_var': 'm_proton'
    },
    {
        'name': 'Отношение масс: Sigma_plus / K0-мезон',
        'formula': 'Sigma_plus_2_m_k0_meson',
        'expr': 'K / (2*pi)**(1/2)',
        'first_var': 'Sigma_plus',
        'second_var': 'm_k0_meson'
    },
    {
        'name': 'Отношение масс: Sigma_plus / DT',
        'formula': 'Sigma_plus_2_m_DT',
        'expr': 'K**2 * 3**(1/2) / (4 * pi**(5/2) * 2**(1/2))',
        'first_var': 'Sigma_plus',
        'second_var': 'm_DT'
    },
    {
        'name': 'Отношение масс: Sigma_plus / Λ-барион',
        'formula': 'Sigma_plus_2_m_Λ_barion',
        'expr': 'K / (4 * math.sqrt(2))',
        'first_var': 'Sigma_plus',
        'second_var': 'm_Λ_barion'
    },
    {
        'name': 'Отношение масс: Sigma_plus / Z-бозон',
        'formula': 'Sigma_plus_2_m_z_bozon',
        'expr': 'K**2 / (16 * pi**(9/2))',
        'first_var': 'Sigma_plus',
        'second_var': 'm_z_bozon'
    },
    {
        'name': 'Отношение масс: Sigma_plus / W-бозон',
        'formula': 'Sigma_plus_2_m_w_bozon',
        'expr': 'K**2 / (8 * pi**5)',
        'first_var': 'Sigma_plus',
        'second_var': 'm_w_bozon'
    },
    {
        'name': 'Отношение масс: Sigma_plus / бозон Хиггса',
        'formula': 'Sigma_plus_2_m_Higgs',
        'expr': 'K**(3/2) / (16 * pi**4)',
        'first_var': 'Sigma_plus',
        'second_var': 'm_Higgs'
    },
    {
        'name': 'Отношение масс: Sigma_plus / D0',
        'formula': 'Sigma_plus_2_m_D0',
        'expr': 'K**2 * 3**(1/2) / (4 * pi**2 * (2*pi)**(1/2))',
        'first_var': 'Sigma_plus',
        'second_var': 'm_D0'
    },
    {
        'name': 'Отношение масс: Sigma_plus / Υ(1S)',
        'formula': 'Sigma_plus_2_m_Υ_1S',
        'expr': 'K * 2**(1/2) / (4 * pi**2 * 3**(1/2))',
        'first_var': 'Sigma_plus',
        'second_var': 'm_Υ_1S'
    },
    {
        'name': 'Отношение масс: Sigma_plus / c-кварк',
        'formula': 'Sigma_plus_2_m_qu_c',
        'expr': 'K**4 / (8 * pi**4)',
        'first_var': 'Sigma_plus',
        'second_var': 'm_qu_c'
    },
    {
        'name': 'Отношение масс: Sigma_plus / b-кварк',
        'formula': 'Sigma_plus_2_m_qu_b',
        'expr': 'K**2 * 3**(1/2) / (4 * pi**3)',
        'first_var': 'Sigma_plus',
        'second_var': 'm_qu_b'
    },
    {
        'name': 'Отношение масс: Sigma_plus / t-кварк',
        'formula': 'Sigma_plus_2_m_qu_t',
        'expr': '1 / (4 * K**2)',
        'first_var': 'Sigma_plus',
        'second_var': 'm_qu_t'
    },
    {
        'name': 'Отношение масс: Sigma_plus / π-мезон',
        'formula': 'Sigma_plus_2_m_pi_meson',
        'expr': 'K * 2**(1/2)',
        'first_var': 'Sigma_plus',
        'second_var': 'm_pi_meson'
    },
    {
        'name': 'Отношение масс: Sigma_plus / Ksi_0',
        'formula': 'Sigma_plus_2_m_Ksi_0',
        'expr': 'K**(5/2) / (4 * pi**(5/2) * 2**(1/2))',
        'first_var': 'Sigma_plus',
        'second_var': 'Ksi_0'
    },
    {
        'name': 'Отношение масс: Sigma_plus / omega_minus',
        'formula': 'Sigma_plus_2_m_omega_minus',
        'expr': 'K**(5/2) / (4 * pi**3)',
        'first_var': 'Sigma_plus',
        'second_var': 'omega_minus'
    },
    {
        'name': 'Отношение масс: Sigma_plus / Lambda_plus',
        'formula': 'Sigma_plus_2_m_Lambda_plus',
        'expr': 'K**2 / (4 * pi**(5/2))',
        'first_var': 'Sigma_plus',
        'second_var': 'Lambda_plus'
    },
    {
        'name': 'Отношение масс: Sigma_plus / Ksi_plus',
        'formula': 'Sigma_plus_2_m_Ksi_plus',
        'expr': 'K / (4 * pi)',
        'first_var': 'Sigma_plus',
        'second_var': 'Ksi_plus'
    },
    {
        'name': 'Отношение масс: Sigma_plus / Omega0_c',
        'formula': 'Sigma_plus_2_m_Omega0_c',
        'expr': 'pi**(1/2) / 4',
        'first_var': 'Sigma_plus',
        'second_var': 'Omega0_c'
    },
    {
        'name': 'Отношение масс: Sigma_plus / lambda_B0',
        'formula': 'Sigma_plus_2_m_lambda_B0',
        'expr': 'K**(3/2) / (4 * pi**(5/2))',
        'first_var': 'Sigma_plus',
        'second_var': 'lambda_B0'
    },
    {
        'name': 'Отношение масс: Sigma_plus / Ksi_minus',
        'formula': 'Sigma_plus_2_m_Ksi_minus',
        'expr': 'K**(5/2) / (4 * pi**2 * (2*pi)**(1/2))',
        'first_var': 'Sigma_plus',
        'second_var': 'Ksi_minus'
    },
    {
        'name': 'Отношение масс: Sigma_plus / Sigma_minus',
        'formula': 'Sigma_plus_2_m_Sigma_minus',
        'expr': '1',
        'first_var': 'Sigma_plus',
        'second_var': 'Sigma_minus'
    },

    # KSI0
    {
        'name': 'Отношение масс: Ksi_0 / протон',
        'formula': 'Ksi_0_2_m_proton',
        'expr': '2**(1/2)',
        'first_var': 'Ksi_0',
        'second_var': 'm_proton'
    },
    {
        'name': 'Отношение масс: Ksi_0 / K0-мезон',
        'formula': 'Ksi_0_2_m_k0_meson',
        'expr': '4 * pi**2 / K**(3/2)',
        'first_var': 'Ksi_0',
        'second_var': 'm_k0_meson'
    },
    {
        'name': 'Отношение масс: Ksi_0 / DT',
        'formula': 'Ksi_0_2_m_DT',
        'expr': '3**(1/2) / K**(1/2)',
        'first_var': 'Ksi_0',
        'second_var': 'm_DT'
    },
    {
        'name': 'Отношение масс: Ksi_0 / Λ-барион',
        'formula': 'Ksi_0_2_m_Λ_barion',
        'expr': 'pi**(5/2) / K**(3/2)',
        'first_var': 'Ksi_0',
        'second_var': 'm_Λ_barion'
    },
    {
        'name': 'Отношение масс: Ksi_0 / Z-бозон',
        'formula': 'Ksi_0_2_m_z_bozon',
        'expr': '2**(1/2) / (4 * pi**2 * K**(1/2))',
        'first_var': 'Ksi_0',
        'second_var': 'm_z_bozon'
    },
    {
        'name': 'Отношение масс: Ksi_0 / W-бозон',
        'formula': 'Ksi_0_2_m_w_bozon',
        'expr': '1 / (2 * pi**5 * K)**(1/2)',
        'first_var': 'Ksi_0',
        'second_var': 'm_w_bozon'
    },
    {
        'name': 'Отношение масс: Ksi_0 / бозон Хиггса',
        'formula': 'Ksi_0_2_m_Higgs',
        'expr': '2**(1/2) / (4 * pi**(3/2) * K)',
        'first_var': 'Ksi_0',
        'second_var': 'm_Higgs'
    },
    {
        'name': 'Отношение масс: Ksi_0 / D0',
        'formula': 'Ksi_0_2_m_D0',
        'expr': '3**(1/2) / K**(1/2)',
        'first_var': 'Ksi_0',
        'second_var': 'm_D0'
    },
    {
        'name': 'Отношение масс: Ksi_0 / Υ(1S)',
        'formula': 'Ksi_0_2_m_Υ_1S',
        'expr': '2 * pi**(1/2) / (K**(3/2) * 3**(1/2))',
        'first_var': 'Ksi_0',
        'second_var': 'm_Υ_1S'
    },
    {
        'name': 'Отношение масс: Ksi_0 / c-кварк',
        'formula': 'Ksi_0_2_m_qu_c',
        'expr': 'K**(3/2) / (2**(1/2) * pi**(3/2))',
        'first_var': 'Ksi_0',
        'second_var': 'm_qu_c'
    },
    {
        'name': 'Отношение масс: Ksi_0 / b-кварк',
        'formula': 'Ksi_0_2_m_qu_b',
        'expr': '6**(1/2) / ((pi*K)**(1/2))',
        'first_var': 'Ksi_0',
        'second_var': 'm_qu_b'
    },
    {
        'name': 'Отношение масс: Ksi_0 / t-кварк',
        'formula': 'Ksi_0_2_m_qu_t',
        'expr': '2**(1/2) * pi**(5/2) / K**(9/2)',
        'first_var': 'Ksi_0',
        'second_var': 'm_qu_t'
    },
    {
        'name': 'Отношение масс: Ksi_0 / π-мезон',
        'formula': 'Ksi_0_2_m_pi_meson',
        'expr': '8 * pi**(5/2) / K**(3/2)',
        'first_var': 'Ksi_0',
        'second_var': 'm_pi_meson'
    },
    {
        'name': 'Отношение масс: Ksi_0 / Sigma_plus',
        'formula': 'Ksi_0_2_m_Sigma_plus',
        'expr': '4 * 2**(1/2) * (pi/K)**(5/2)',
        'first_var': 'Ksi_0',
        'second_var': 'Sigma_plus'
    },
    {
        'name': 'Отношение масс: Ksi_0 / omega_minus',
        'formula': 'Ksi_0_2_m_omega_minus',
        'expr': '2**(1/2) / pi**(1/2)',
        'first_var': 'Ksi_0',
        'second_var': 'omega_minus'
    },
    {
        'name': 'Отношение масс: Ksi_0 / Lambda_plus',
        'formula': 'Ksi_0_2_m_Lambda_plus',
        'expr': '2**(1/2) / K**(1/2)',
        'first_var': 'Ksi_0',
        'second_var': 'Lambda_plus'
    },
    {
        'name': 'Отношение масс: Ksi_0 / Ksi_plus',
        'formula': 'Ksi_0_2_m_Ksi_plus',
        'expr': 'pi**(3/2) * 2**(1/2) / K**(3/2)',
        'first_var': 'Ksi_0',
        'second_var': 'Ksi_plus'
    },
    {
        'name': 'Отношение масс: Ksi_0 / Omega0_c',
        'formula': 'Ksi_0_2_m_Omega0_c',
        'expr': '2**(1/2) * pi**3 / K**(5/2)',
        'first_var': 'Ksi_0',
        'second_var': 'Omega0_c'
    },
    {
        'name': 'Отношение масс: Ksi_0 / lambda_B0',
        'formula': 'Ksi_0_2_m_lambda_B0',
        'expr': '2**(1/2) / K',
        'first_var': 'Ksi_0',
        'second_var': 'lambda_B0'
    },
    {
        'name': 'Отношение масс: Ksi_0 / Ksi_minus',
        'formula': 'Ksi_0_2_m_Ksi_minus',
        'expr': '1',
        'first_var': 'Ksi_0',
        'second_var': 'Ksi_minus'
    },
    {
        'name': 'Отношение масс: Ksi_0 / Sigma_minus',
        'formula': 'Ksi_0_2_m_Sigma_minus',
        'expr': '4 * 2**(1/2) * (pi/K)**(5/2)',
        'first_var': 'Ksi_0',
        'second_var': 'Sigma_minus'
    },

    # omega_minus
    {
        'name': 'Отношение масс: omega_minus / протон',
        'formula': 'omega_minus_2_m_proton',
        'expr': 'pi**(1/2)',
        'first_var': 'omega_minus',
        'second_var': 'm_proton'
    },
    {
        'name': 'Отношение масс: omega_minus / K0-мезон',
        'formula': 'omega_minus_2_m_k0_meson',
        'expr': '2*2**(1/2) * pi**(5/2) / K**(3/2)',
        'first_var': 'omega_minus',
        'second_var': 'm_k0_meson'
    },
    {
        'name': 'Отношение масс: omega_minus / DT',
        'formula': 'omega_minus_2_m_DT',
        'expr': '(3*pi / (2*K))**(1/2)',
        'first_var': 'omega_minus',
        'second_var': 'm_DT'
    },
    {
        'name': 'Отношение масс: omega_minus / Λ-барион',
        'formula': 'omega_minus_2_m_Λ_barion',
        'expr': 'pi**3 / (K**(3/2) * 2**(1/2))',
        'first_var': 'omega_minus',
        'second_var': 'm_Λ_barion'
    },
    {
        'name': 'Отношение масс: omega_minus / Z-бозон',
        'formula': 'omega_minus_2_m_z_bozon',
        'expr': '1 / (4 * pi**(3/2) * K**(1/2))',
        'first_var': 'omega_minus',
        'second_var': 'm_z_bozon'
    },
    {
        'name': 'Отношение масс: omega_minus / W-бозон',
        'formula': 'omega_minus_2_m_w_bozon',
        'expr': '1 / (2 * pi**2 * K**(1/2))',
        'first_var': 'omega_minus',
        'second_var': 'm_w_bozon'
    },
    {
        'name': 'Отношение масс: omega_minus / бозон Хиггса',
        'formula': 'omega_minus_2_m_Higgs',
        'expr': '1 / (4 * pi * K)',
        'first_var': 'omega_minus',
        'second_var': 'm_Higgs'
    },
    {
        'name': 'Отношение масс: omega_minus / D0',
        'formula': 'omega_minus_2_m_D0',
        'expr': '(pi * 3)**(1/2) / (K**(1/2) * 2**(1/2))',
        'first_var': 'omega_minus',
        'second_var': 'm_D0'
    },
    {
        'name': 'Отношение масс: omega_minus / Υ(1S)',
        'formula': 'omega_minus_2_m_Υ_1S',
        'expr': 'pi * 2**(1/2) / (K**(3/2) * 3**(1/2))',
        'first_var': 'omega_minus',
        'second_var': 'm_Υ_1S'
    },
    {
        'name': 'Отношение масс: omega_minus / c-кварк',
        'formula': 'omega_minus_2_m_qu_c',
        'expr': 'K**(3/2) / (2 * pi)',
        'first_var': 'omega_minus',
        'second_var': 'm_qu_c'
    },
    {
        'name': 'Отношение масс: omega_minus / b-кварк',
        'formula': 'omega_minus_2_m_qu_b',
        'expr': '3**(1/2) / K**(1/2)',
        'first_var': 'omega_minus',
        'second_var': 'm_qu_b'
    },
    {
        'name': 'Отношение масс: omega_minus / t-кварк',
        'formula': 'omega_minus_2_m_qu_t',
        'expr': 'pi**3 / K**(9/2)',
        'first_var': 'omega_minus',
        'second_var': 'm_qu_t'
    },
    {
        'name': 'Отношение масс: omega_minus / π-мезон',
        'formula': 'omega_minus_2_m_pi_meson',
        'expr': '4 * pi**3 * 2**(1/2) / K**(3/2)',
        'first_var': 'omega_minus',
        'second_var': 'm_pi_meson'
    },
    {
        'name': 'Отношение масс: omega_minus / Sigma_plus',
        'formula': 'omega_minus_2_m_Sigma_plus',
        'expr': '4 * pi**3 / K**(5/2)',
        'first_var': 'omega_minus',
        'second_var': 'Sigma_plus'
    },
    {
        'name': 'Отношение масс: omega_minus / Ksi_0',
        'formula': 'omega_minus_2_m_Ksi_0',
        'expr': 'pi**(1/2) / 2**(1/2)',
        'first_var': 'omega_minus',
        'second_var': 'Ksi_0'
    },
    {
        'name': 'Отношение масс: omega_minus / Lambda_plus',
        'formula': 'omega_minus_2_m_Lambda_plus',
        'expr': 'pi**(1/2) / K**(1/2)',
        'first_var': 'omega_minus',
        'second_var': 'Lambda_plus'
    },
    {
        'name': 'Отношение масс: omega_minus / Ksi_plus',
        'formula': 'omega_minus_2_m_Ksi_plus',
        'expr': 'pi**2 / K**(3/2)',
        'first_var': 'omega_minus',
        'second_var': 'Ksi_plus'
    },
    {
        'name': 'Отношение масс: omega_minus / Omega0_c',
        'formula': 'omega_minus_2_m_Omega0_c',
        'expr': 'pi**(7/2) / K**(5/2)',
        'first_var': 'omega_minus',
        'second_var': 'Omega0_c'
    },
    {
        'name': 'Отношение масс: omega_minus / lambda_B0',
        'formula': 'omega_minus_2_m_lambda_B0',
        'expr': 'pi**(1/2) / K',
        'first_var': 'omega_minus',
        'second_var': 'lambda_B0'
    },
    {
        'name': 'Отношение масс: omega_minus / Ksi_minus',
        'formula': 'omega_minus_2_m_Ksi_minus',
        'expr': 'pi**(1/2) / 2**(1/2)',
        'first_var': 'omega_minus',
        'second_var': 'Ksi_minus'
    },
    {
        'name': 'Отношение масс: omega_minus / Sigma_minus',
        'formula': 'omega_minus_2_m_Sigma_minus',
        'expr': '4 * pi**3 / K**(5/2)',
        'first_var': 'omega_minus',
        'second_var': 'Sigma_minus'
    },
    {
        'name': 'Отношение масс: Lambda_plus / протон',
        'formula': 'Lambda_plus_2_m_proton',
        'expr': 'K**(1/2)',
        'first_var': 'Lambda_plus',
        'second_var': 'm_proton'
    },
    {
        'name': 'Отношение масс: Lambda_plus / K0-мезон',
        'formula': 'Lambda_plus_2_m_k0_meson',
        'expr': '4 * pi**2 / (K * 2**(1/2))',
        'first_var': 'Lambda_plus',
        'second_var': 'm_k0_meson'
    },
    {
        'name': 'Отношение масс: Lambda_plus / DT',
        'formula': 'Lambda_plus_2_m_DT',
        'expr': '3**(1/2) / 2**(1/2)',
        'first_var': 'Lambda_plus',
        'second_var': 'm_DT'
    },
    {
        'name': 'Отношение масс: Lambda_plus / Λ-барион',
        'formula': 'Lambda_plus_2_m_Λ_barion',
        'expr': 'pi**(5/2) / (K * 2**(1/2))',
        'first_var': 'Lambda_plus',
        'second_var': 'm_Λ_barion'
    },
    {
        'name': 'Отношение масс: Lambda_plus / Z-бозон',
        'formula': 'Lambda_plus_2_m_z_bozon',
        'expr': '1 / (4 * pi**2)',
        'first_var': 'Lambda_plus',
        'second_var': 'm_z_bozon'
    },
    {
        'name': 'Отношение масс: Lambda_plus / W-бозон',
        'formula': 'Lambda_plus_2_m_w_bozon',
        'expr': '1 / (2 * pi**(5/2))',
        'first_var': 'Lambda_plus',
        'second_var': 'm_w_bozon'
    },
    {
        'name': 'Отношение масс: Lambda_plus / бозон Хиггса',
        'formula': 'Lambda_plus_2_m_Higgs',
        'expr': '1 / (4 * pi**(3/2) * K**(1/2))',
        'first_var': 'Lambda_plus',
        'second_var': 'm_Higgs'
    },
    {
        'name': 'Отношение масс: Lambda_plus / D0',
        'formula': 'Lambda_plus_2_m_D0',
        'expr': '3**(1/2) / 2**(1/2)',
        'first_var': 'Lambda_plus',
        'second_var': 'm_D0'
    },
    {
        'name': 'Отношение масс: Lambda_plus / Υ(1S)',
        'formula': 'Lambda_plus_2_m_Υ_1S',
        'expr': 'pi**(1/2) * 2**(1/2) / (K * 3**(1/2))',
        'first_var': 'Lambda_plus',
        'second_var': 'm_Υ_1S'
    },
    {
        'name': 'Отношение масс: Lambda_plus / c-кварк',
        'formula': 'Lambda_plus_2_m_qu_c',
        'expr': 'K**2 / (2 * pi**(3/2))',
        'first_var': 'Lambda_plus',
        'second_var': 'm_qu_c'
    },
    {
        'name': 'Отношение масс: Lambda_plus / b-кварк',
        'formula': 'Lambda_plus_2_m_qu_b',
        'expr': '(3/pi)**(1/2)',
        'first_var': 'Lambda_plus',
        'second_var': 'm_qu_b'
    },
    {
        'name': 'Отношение масс: Lambda_plus / t-кварк',
        'formula': 'Lambda_plus_2_m_qu_t',
        'expr': 'pi**(5/2) / K**4',
        'first_var': 'Lambda_plus',
        'second_var': 'm_qu_t'
    },
    {
        'name': 'Отношение масс: Lambda_plus / π-мезон',
        'formula': 'Lambda_plus_2_m_pi_meson',
        'expr': '4 * pi**(5/2) * 2**(1/2) / K',
        'first_var': 'Lambda_plus',
        'second_var': 'm_pi_meson'
    },
    {
        'name': 'Отношение масс: Lambda_plus / Sigma_plus',
        'formula': 'Lambda_plus_2_m_Sigma_plus',
        'expr': '4 * pi**(5/2) / K**2',
        'first_var': 'Lambda_plus',
        'second_var': 'Sigma_plus'
    },
    {
        'name': 'Отношение масс: Lambda_plus / Ksi_0',
        'formula': 'Lambda_plus_2_m_Ksi_0',
        'expr': 'K**(1/2)/ 2**(1/2)',
        'first_var': 'Lambda_plus',
        'second_var': 'Ksi_0'
    },
    {
        'name': 'Отношение масс: Lambda_plus / omega_minus',
        'formula': 'Lambda_plus_2_m_omega_minus',
        'expr': 'K**(1/2) / pi**(1/2)',
        'first_var': 'Lambda_plus',
        'second_var': 'omega_minus'
    },
    {
        'name': 'Отношение масс: Lambda_plus / Ksi_plus',
        'formula': 'Lambda_plus_2_m_Ksi_plus',
        'expr': 'pi**(3/2) / K',
        'first_var': 'Lambda_plus',
        'second_var': 'Ksi_plus'
    },
    {
        'name': 'Отношение масс: Lambda_plus / Omega0_c',
        'formula': 'Lambda_plus_2_m_Omega0_c',
        'expr': 'pi**3 / K**2',
        'first_var': 'Lambda_plus',
        'second_var': 'Omega0_c'
    },
    {
        'name': 'Отношение масс: Lambda_plus / lambda_B0',
        'formula': 'Lambda_plus_2_m_lambda_B0',
        'expr': '1/K**(1/2)',
        'first_var': 'Lambda_plus',
        'second_var': 'lambda_B0'
    },
    {
        'name': 'Отношение масс: Lambda_plus / Ksi_minus',
        'formula': 'Lambda_plus_2_m_Ksi_minus',
        'expr': 'K**(1/2) / 2**(1/2)',
        'first_var': 'Lambda_plus',
        'second_var': 'Ksi_minus'
    },
    {
        'name': 'Отношение масс: Lambda_plus / Sigma_minus',
        'formula': 'Lambda_plus_2_m_Sigma_minus',
        'expr': '4 * pi**(5/2) / K**2',
        'first_var': 'Lambda_plus',
        'second_var': 'Sigma_minus'
    },

    # Ksi_plus
    {
        'name': 'Отношение масс: Ksi_plus / протон',
        'formula': 'Ksi_plus_2_m_proton',
        'expr': 'K**(3/2) / pi**(3/2)',
        'first_var': 'Ksi_plus',
        'second_var': 'm_proton'
    },
    {
        'name': 'Отношение масс: Ksi_plus / K0-мезон',
        'formula': 'Ksi_plus_2_m_k0_meson',
        'expr': '4 * pi**(1/2) / 2**(1/2)',
        'first_var': 'Ksi_plus',
        'second_var': 'm_k0_meson'
    },
    {
        'name': 'Отношение масс: Ksi_plus / DT',
        'formula': 'Ksi_plus_2_m_DT',
        'expr': 'K * 3**(1/2) / (pi**(3/2) * 2**(1/2))',
        'first_var': 'Ksi_plus',
        'second_var': 'm_DT'
    },
    {
        'name': 'Отношение масс: Ksi_plus / Λ-барион',
        'formula': 'Ksi_plus_2_m_Λ_barion',
        'expr': 'pi / 2**(1/2)',
        'first_var': 'Ksi_plus',
        'second_var': 'm_Λ_barion'
    },
    {
        'name': 'Отношение масс: Ksi_plus / Z-бозон',
        'formula': 'Ksi_plus_2_m_z_bozon',
        'expr': 'K / (4 * pi**(7/2))',
        'first_var': 'Ksi_plus',
        'second_var': 'm_z_bozon'
    },
    {
        'name': 'Отношение масс: Ksi_plus / W-бозон',
        'formula': 'Ksi_plus_2_m_w_bozon',
        'expr': 'K / (2 * pi**4)',
        'first_var': 'Ksi_plus',
        'second_var': 'm_w_bozon'
    },
    {
        'name': 'Отношение масс: Ksi_plus / бозон Хиггса',
        'formula': 'Ksi_plus_2_m_Higgs',
        'expr': 'K**(1/2) / (4 * pi**3)',
        'first_var': 'Ksi_plus',
        'second_var': 'm_Higgs'
    },
    {
        'name': 'Отношение масс: Ksi_plus / D0',
        'formula': 'Ksi_plus_2_m_D0',
        'expr': 'K * 3**(1/2) / (pi**(3/2) * 2**(1/2))',
        'first_var': 'Ksi_plus',
        'second_var': 'm_D0'
    },
    {
        'name': 'Отношение масс: Ksi_plus / Υ(1S)',
        'formula': 'Ksi_plus_2_m_Υ_1S',
        'expr': '2**(1/2) / (pi * 3**(1/2))',
        'first_var': 'Ksi_plus',
        'second_var': 'm_Υ_1S'
    },
    {
        'name': 'Отношение масс: Ksi_plus / c-кварк',
        'formula': 'Ksi_plus_2_m_qu_c',
        'expr': 'K**3 / (2 * pi**3)',
        'first_var': 'Ksi_plus',
        'second_var': 'm_qu_c'
    },
    {
        'name': 'Отношение масс: Ksi_plus / b-кварк',
        'formula': 'Ksi_plus_2_m_qu_b',
        'expr': 'K * 3**(1/2) / pi**2',
        'first_var': 'Ksi_plus',
        'second_var': 'm_qu_b'
    },
    {
        'name': 'Отношение масс: Ksi_plus / t-кварк',
        'formula': 'Ksi_plus_2_m_qu_t',
        'expr': 'pi / K**3',
        'first_var': 'Ksi_plus',
        'second_var': 'm_qu_t'
    },
    {
        'name': 'Отношение масс: Ksi_plus / π-мезон',
        'formula': 'Ksi_plus_2_m_pi_meson',
        'expr': '4 * pi * 2**(1/2)',
        'first_var': 'Ksi_plus',
        'second_var': 'm_pi_meson'
    },
    {
        'name': 'Отношение масс: Ksi_plus / Sigma_plus',
        'formula': 'Ksi_plus_2_m_Sigma_plus',
        'expr': '4 * pi / K',
        'first_var': 'Ksi_plus',
        'second_var': 'Sigma_plus'
    },
    {
        'name': 'Отношение масс: Ksi_plus / Ksi_0',
        'formula': 'Ksi_plus_2_m_Ksi_0',
        'expr': 'K**(3/2) / (pi**(3/2) * 2**(1/2))',
        'first_var': 'Ksi_plus',
        'second_var': 'Ksi_0'
    },
    {
        'name': 'Отношение масс: Ksi_plus / omega_minus',
        'formula': 'Ksi_plus_2_m_omega_minus',
        'expr': 'K**(3/2) / pi**2',
        'first_var': 'Ksi_plus',
        'second_var': 'omega_minus'
    },
    {
        'name': 'Отношение масс: Ksi_plus / Lambda_plus',
        'formula': 'Ksi_plus_2_m_Lambda_plus',
        'expr': 'K / pi**(3/2)',
        'first_var': 'Ksi_plus',
        'second_var': 'Lambda_plus'
    },
    {
        'name': 'Отношение масс: Ksi_plus / Omega0_c',
        'formula': 'Ksi_plus_2_m_Omega0_c',
        'expr': 'pi**(3/2) / K',
        'first_var': 'Ksi_plus',
        'second_var': 'Omega0_c'
    },
    {
        'name': 'Отношение масс: Ksi_plus / lambda_B0',
        'formula': 'Ksi_plus_2_m_lambda_B0',
        'expr': 'K**(1/2) / pi**(3/2)',
        'first_var': 'Ksi_plus',
        'second_var': 'lambda_B0'
    },
    {
        'name': 'Отношение масс: Ksi_plus / Ksi_minus',
        'formula': 'Ksi_plus_2_m_Ksi_minus',
        'expr': 'K**(3/2) / (pi**(3/2) * 2**(1/2))',
        'first_var': 'Ksi_plus',
        'second_var': 'Ksi_minus'
    },
    {
        'name': 'Отношение масс: Ksi_plus / Sigma_minus',
        'formula': 'Ksi_plus_2_m_Sigma_minus',
        'expr': '4 * pi / K',
        'first_var': 'Ksi_plus',
        'second_var': 'Sigma_minus'
    },

    # omega_c
    {
        'name': 'Отношение масс: Omega0_c / протон',
        'formula': 'Omega0_c_2_m_proton',
        'expr': 'K**(5/2) / pi**3',
        'first_var': 'Omega0_c',
        'second_var': 'm_proton'
    },
    {
        'name': 'Отношение масс: Omega0_c / K0-мезон',
        'formula': 'Omega0_c_2_m_k0_meson',
        'expr': '4 * K / (pi * 2**(1/2))',
        'first_var': 'Omega0_c',
        'second_var': 'm_k0_meson'
    },
    {
        'name': 'Отношение масс: Omega0_c / DT',
        'formula': 'Omega0_c_2_m_DT',
        'expr': 'K**2 * 3**(1/2) / (pi**3 * 2**(1/2))',
        'first_var': 'Omega0_c',
        'second_var': 'm_DT'
    },
    {
        'name': 'Отношение масс: Omega0_c / Λ-барион',
        'formula': 'Omega0_c_2_m_Λ_barion',
        'expr': 'K / ((2*pi)**(1/2))',
        'first_var': 'Omega0_c',
        'second_var': 'm_Λ_barion'
    },
    {
        'name': 'Отношение масс: Omega0_c / Z-бозон',
        'formula': 'Omega0_c_2_m_z_bozon',
        'expr': 'K**2 / (4 * pi**5)',
        'first_var': 'Omega0_c',
        'second_var': 'm_z_bozon'
    },
    {
        'name': 'Отношение масс: Omega0_c / W-бозон',
        'formula': 'Omega0_c_2_m_w_bozon',
        'expr': 'K**2 / (2 * pi**(11/2))',
        'first_var': 'Omega0_c',
        'second_var': 'm_w_bozon'
    },
    {
        'name': 'Отношение масс: Omega0_c / бозон Хиггса',
        'formula': 'Omega0_c_2_m_Higgs',
        'expr': 'K**(3/2) / (4 * pi**(9/2))',
        'first_var': 'Omega0_c',
        'second_var': 'm_Higgs'
    },
    {
        'name': 'Отношение масс: Omega0_c / D0',
        'formula': 'Omega0_c_2_m_D0',
        'expr': 'K**2 * 3**(1/2) / (pi**3 * 2**(1/2))',
        'first_var': 'Omega0_c',
        'second_var': 'm_D0'
    },
    {
        'name': 'Отношение масс: Omega0_c / Υ(1S)',
        'formula': 'Omega0_c_2_m_Υ_1S',
        'expr': 'K * 2**(1/2) / (pi**(5/2) * 3**(1/2))',
        'first_var': 'Omega0_c',
        'second_var': 'm_Υ_1S'
    },
    {
        'name': 'Отношение масс: Omega0_c / c-кварк',
        'formula': 'Omega0_c_2_m_qu_c',
        'expr': 'K**4 / (2 * pi**(9/2))',
        'first_var': 'Omega0_c',
        'second_var': 'm_qu_c'
    },
    {
        'name': 'Отношение масс: Omega0_c / b-кварк',
        'formula': 'Omega0_c_2_m_qu_b',
        'expr': 'K**2 * 3**(1/2) / pi**(7/2)',
        'first_var': 'Omega0_c',
        'second_var': 'm_qu_b'
    },
    {
        'name': 'Отношение масс: Omega0_c / t-кварк',
        'formula': 'Omega0_c_2_m_qu_t',
        'expr': '1 / (pi**(1/2) * K**2)',
        'first_var': 'Omega0_c',
        'second_var': 'm_qu_t'
    },
    {
        'name': 'Отношение масс: Omega0_c / π-мезон',
        'formula': 'Omega0_c_2_m_pi_meson',
        'expr': '4 * 2**(1/2) * K / pi**(1/2)',
        'first_var': 'Omega0_c',
        'second_var': 'm_pi_meson'
    },
    {
        'name': 'Отношение масс: Omega0_c / Sigma_plus',
        'formula': 'Omega0_c_2_m_Sigma_plus',
        'expr': '4 / pi**(1/2)',
        'first_var': 'Omega0_c',
        'second_var': 'Sigma_plus'
    },
    {
        'name': 'Отношение масс: Omega0_c / Ksi_0',
        'formula': 'Omega0_c_2_m_Ksi_0',
        'expr': 'K**(5/2) / (pi**3 * 2**(1/2))',
        'first_var': 'Omega0_c',
        'second_var': 'Ksi_0'
    },
    {
        'name': 'Отношение масс: Omega0_c / omega_minus',
        'formula': 'Omega0_c_2_m_omega_minus',
        'expr': 'K**(5/2) / pi**(7/2)',
        'first_var': 'Omega0_c',
        'second_var': 'omega_minus'
    },
    {
        'name': 'Отношение масс: Omega0_c / Lambda_plus',
        'formula': 'Omega0_c_2_m_Lambda_plus',
        'expr': 'K**2 / pi**3',
        'first_var': 'Omega0_c',
        'second_var': 'Lambda_plus'
    },
    {
        'name': 'Отношение масс: Omega0_c / Ksi_plus',
        'formula': 'Omega0_c_2_m_Ksi_plus',
        'expr': 'K / pi**(3/2)',
        'first_var': 'Omega0_c',
        'second_var': 'Ksi_plus'
    },
    {
        'name': 'Отношение масс: Omega0_c / lambda_B0',
        'formula': 'Omega0_c_2_m_lambda_B0',
        'expr': 'K**(3/2) / pi**3',
        'first_var': 'Omega0_c',
        'second_var': 'lambda_B0'
    },
    {
        'name': 'Отношение масс: Omega0_c / Ksi_minus',
        'formula': 'Omega0_c_2_m_Ksi_minus',
        'expr': 'K**(5/2) / (pi**3 * 2**(1/2))',
        'first_var': 'Omega0_c',
        'second_var': 'Ksi_minus'
    },
    {
        'name': 'Отношение масс: Omega0_c / Sigma_minus',
        'formula': 'Omega0_c_2_m_Sigma_minus',
        'expr': '4 / pi**(1/2)',
        'first_var': 'Omega0_c',
        'second_var': 'Sigma_minus'
    },

    #B0
    {
        'name': 'Отношение масс: lambda_B0 / протон',
        'formula': 'lambda_B0_2_m_proton',
        'expr': 'K',
        'first_var': 'lambda_B0',
        'second_var': 'm_proton'
    },
    {
        'name': 'Отношение масс: lambda_B0 / K0-мезон',
        'formula': 'lambda_B0_2_m_k0_meson',
        'expr': '4 * pi**2 / (K**(1/2) * 2**(1/2))',
        'first_var': 'lambda_B0',
        'second_var': 'm_k0_meson'
    },
    {
        'name': 'Отношение масс: lambda_B0 / DT',
        'formula': 'lambda_B0_2_m_DT',
        'expr': 'K**(1/2) * 3**(1/2) / 2**(1/2)',
        'first_var': 'lambda_B0',
        'second_var': 'm_DT'
    },
    {
        'name': 'Отношение масс: lambda_B0 / Λ-барион',
        'formula': 'lambda_B0_2_m_Λ_barion',
        'expr': 'pi**(5/2) / (K**(1/2) * 2**(1/2))',
        'first_var': 'lambda_B0',
        'second_var': 'm_Λ_barion'
    },
    {
        'name': 'Отношение масс: lambda_B0 / Z-бозон',
        'formula': 'lambda_B0_2_m_z_bozon',
        'expr': 'K**(1/2) / (4 * pi**2)',
        'first_var': 'lambda_B0',
        'second_var': 'm_z_bozon'
    },
    {
        'name': 'Отношение масс: lambda_B0 / W-бозон',
        'formula': 'lambda_B0_2_m_w_bozon',
        'expr': 'K**(1/2) / (2 * pi**(5/2))',
        'first_var': 'lambda_B0',
        'second_var': 'm_w_bozon'
    },
    {
        'name': 'Отношение масс: lambda_B0 / бозон Хиггса',
        'formula': 'lambda_B0_2_m_Higgs',
        'expr': '1 / (4 * pi**(3/2))',
        'first_var': 'lambda_B0',
        'second_var': 'm_Higgs'
    },
    {
        'name': 'Отношение масс: lambda_B0 / D0',
        'formula': 'lambda_B0_2_m_D0',
        'expr': 'K**(1/2) * 3**(1/2) / 2**(1/2)',
        'first_var': 'lambda_B0',
        'second_var': 'm_D0'
    },
    {
        'name': 'Отношение масс: lambda_B0 / Υ(1S)',
        'formula': 'lambda_B0_2_m_Υ_1S',
        'expr': 'pi**(1/2) * 2**(1/2) / (K**(1/2) * 3**(1/2))',
        'first_var': 'lambda_B0',
        'second_var': 'm_Υ_1S'
    },
    {
        'name': 'Отношение масс: lambda_B0 / c-кварк',
        'formula': 'lambda_B0_2_m_qu_c',
        'expr': 'K**(5/2) / (2 * pi**(3/2))',
        'first_var': 'lambda_B0',
        'second_var': 'm_qu_c'
    },
    {
        'name': 'Отношение масс: lambda_B0 / b-кварк',
        'formula': 'lambda_B0_2_m_qu_b',
        'expr': 'K**(1/2) * 3**(1/2) / pi**(1/2)',
        'first_var': 'lambda_B0',
        'second_var': 'm_qu_b'
    },
    {
        'name': 'Отношение масс: lambda_B0 / t-кварк',
        'formula': 'lambda_B0_2_m_qu_t',
        'expr': 'pi**(5/2) / K**(7/2)',
        'first_var': 'lambda_B0',
        'second_var': 'm_qu_t'
    },
    {
        'name': 'Отношение масс: lambda_B0 / π-мезон',
        'formula': 'lambda_B0_2_m_pi_meson',
        'expr': '4 * pi**(5/2) * 2**(1/2) / K**(1/2)',
        'first_var': 'lambda_B0',
        'second_var': 'm_pi_meson'
    },
    {
        'name': 'Отношение масс: lambda_B0 / Sigma_plus',
        'formula': 'lambda_B0_2_m_Sigma_plus',
        'expr': '4 * pi**(5/2) / K**(3/2)',
        'first_var': 'lambda_B0',
        'second_var': 'Sigma_plus'
    },
    {
        'name': 'Отношение масс: lambda_B0 / Ksi_0',
        'formula': 'lambda_B0_2_m_Ksi_0',
        'expr': 'K / 2**(1/2)',
        'first_var': 'lambda_B0',
        'second_var': 'Ksi_0'
    },
    {
        'name': 'Отношение масс: lambda_B0 / omega_minus',
        'formula': 'lambda_B0_2_m_omega_minus',
        'expr': 'K / pi**(1/2)',
        'first_var': 'lambda_B0',
        'second_var': 'omega_minus'
    },
    {
        'name': 'Отношение масс: lambda_B0 / Lambda_plus',
        'formula': 'lambda_B0_2_m_Lambda_plus',
        'expr': 'K**(1/2)',
        'first_var': 'lambda_B0',
        'second_var': 'Lambda_plus'
    },
    {
        'name': 'Отношение масс: lambda_B0 / Ksi_plus',
        'formula': 'lambda_B0_2_m_Ksi_plus',
        'expr': 'pi**(3/2) / K**(1/2)',
        'first_var': 'lambda_B0',
        'second_var': 'Ksi_plus'
    },
    {
        'name': 'Отношение масс: lambda_B0 / Omega0_c',
        'formula': 'lambda_B0_2_m_Omega0_c',
        'expr': 'pi**3 / K**(3/2)',
        'first_var': 'lambda_B0',
        'second_var': 'Omega0_c'
    },
    {
        'name': 'Отношение масс: lambda_B0 / Ksi_minus',
        'formula': 'lambda_B0_2_m_Ksi_minus',
        'expr': 'K / 2**(1/2)',
        'first_var': 'lambda_B0',
        'second_var': 'Ksi_minus'
    },
    {
        'name': 'Отношение масс: lambda_B0 / Sigma_minus',
        'formula': 'lambda_B0_2_m_Sigma_minus',
        'expr': '4 * pi**(5/2) / K**(3/2)',
        'first_var': 'lambda_B0',
        'second_var': 'Sigma_minus'
    },

    {
        'name': 'Отношение масс: phi_meson / мюон',
        'formula': 'phi_meson_2_m_muon',
        'expr': 'K**(5/2) * 3**(1/2) * 2**(1/2) / (4 * pi**(3/2))',
        'first_var': 'phi_meson',
        'second_var': 'm_muon'
    },
    {
        'name': 'Отношение масс: phi_meson / тау',
        'formula': 'phi_meson_2_m_tau',
        'expr': '(2/K)**(1/2)',
        'first_var': 'phi_meson',
        'second_var': 'm_tau'
    },
    {
        'name': 'Отношение масс: phi_meson / J/ψ',
        'formula': 'phi_meson_2_m_J_ψ',
        'expr': 'K**(3/2) / (8 * pi**(3/2))',
        'first_var': 'phi_meson',
        'second_var': 'm_J_ψ'
    },
    {
        'name': 'Отношение масс: phi_meson / η',
        'formula': 'phi_meson_2_m_eta',
        'expr': '2**(1/2) * K**(3/2) / (2 * pi**(3/2))',
        'first_var': 'phi_meson',
        'second_var': 'm_eta'
    },
    {
        'name': 'Отношение масс: phi_meson / u-кварк',
        'formula': 'phi_meson_2_m_qu_u',
        'expr': '4 * K**(3/2) * pi**(5/2) * 2**(1/2) / 3**(1/2)',
        'first_var': 'phi_meson',
        'second_var': 'm_qu_u'
    },
    {
        'name': 'Отношение масс: phi_meson / d-кварк',
        'formula': 'phi_meson_2_m_qu_d',
        'expr': 'K**(5/2) * 3**(1/2) * (2*pi)**(1/2)',
        'first_var': 'phi_meson',
        'second_var': 'm_qu_d'
    },
    {
        'name': 'Отношение масс: phi_meson / omega_meson',
        'formula': 'phi_meson_2_m_omega_meson',
        'expr': 'K**(3/2) / (2 * pi**(3/2))',
        'first_var': 'phi_meson',
        'second_var': 'omega_meson'
    },
    {
        'name': 'Отношение масс: phi_meson / eta_shtrih',
        'formula': 'phi_meson_2_m_eta_shtrih',
        'expr': '2 * 2**(1/2) * pi **(3/2) / K**(3/2)',
        'first_var': 'phi_meson',
        'second_var': 'eta_shtrih'
    },

    {
        'name': 'Отношение масс: omega_meson / мюон',
        'formula': 'omega_meson_2_m_muon',
        'expr': 'K * (3/2)**(1/2)',
        'first_var': 'omega_meson',
        'second_var': 'm_muon'
    },
    {
        'name': 'Отношение масс: omega_meson / тау',
        'formula': 'omega_meson_2_m_tau',
        'expr': '2 * pi**(3/2) * 2**(1/2) / K**2',
        'first_var': 'omega_meson',
        'second_var': 'm_tau'
    },
    {
        'name': 'Отношение масс: omega_meson / J/ψ',
        'formula': 'omega_meson_2_m_J_ψ',
        'expr': '1/4',
        'first_var': 'omega_meson',
        'second_var': 'm_J_ψ'
    },
    {
        'name': 'Отношение масс: omega_meson / η',
        'formula': 'omega_meson_2_m_eta',
        'expr': '2**(1/2)',
        'first_var': 'omega_meson',
        'second_var': 'm_eta'
    },
    {
        'name': 'Отношение масс: omega_meson / u-кварк',
        'formula': 'omega_meson_2_m_qu_u',
        'expr': '8 * pi**4 * 2**(1/2) / 3**(1/2)',
        'first_var': 'omega_meson',
        'second_var': 'm_qu_u'
    },
    {
        'name': 'Отношение масс: omega_meson / d-кварк',
        'formula': 'omega_meson_2_m_qu_d',
        'expr': '2 * K * pi**2 * 2**(1/2) * 3**(1/2)',
        'first_var': 'omega_meson',
        'second_var': 'm_qu_d'
    },
    {
        'name': 'Отношение масс: omega_meson / phi_meson',
        'formula': 'omega_meson_2_m_phi_meson',
        'expr': '2 * (pi/K)**(3/2)',
        'first_var': 'omega_meson',
        'second_var': 'phi_meson'
    },
    {
        'name': 'Отношение масс: omega_meson / eta_shtrih',
        'formula': 'omega_meson_2_m_eta_shtrih',
        'expr': '4 * pi**3 * 2**(1/2) / K**3',
        'first_var': 'omega_meson',
        'second_var': 'eta_shtrih'
    },

    {
        'name': 'Отношение масс: eta_shtrih / мюон',
        'formula': 'eta_shtrih_2_m_muon',
        'expr': 'K**4 * 3**(1/2) / (8 * pi**3)',
        'first_var': 'eta_shtrih',
        'second_var': 'm_muon'
    },
    {
        'name': 'Отношение масс: eta_shtrih / тау',
        'formula': 'eta_shtrih_2_m_tau',
        'expr': 'K / (2 * pi**(3/2))',
        'first_var': 'eta_shtrih',
        'second_var': 'm_tau'
    },
    {
        'name': 'Отношение масс: eta_shtrih / J/ψ',
        'formula': 'eta_shtrih_2_m_J_ψ',
        'expr': 'K**3 / (16 * pi**3 * 2**(1/2))',
        'first_var': 'eta_shtrih',
        'second_var': 'm_J_ψ'
    },
    {
        'name': 'Отношение масс: eta_shtrih / η',
        'formula': 'eta_shtrih_2_m_eta',
        'expr': 'K**3 / (4 * pi**3)',
        'first_var': 'eta_shtrih',
        'second_var': 'm_eta'
    },
    {
        'name': 'Отношение масс: eta_shtrih / u-кварк',
        'formula': 'eta_shtrih_2_m_qu_u',
        'expr': '2 * K**3 * pi / 3**(1/2)',
        'first_var': 'eta_shtrih',
        'second_var': 'm_qu_u'
    },
    {
        'name': 'Отношение масс: eta_shtrih / d-кварк',
        'formula': 'eta_shtrih_2_m_qu_d',
        'expr': 'K**4 * 3**(1/2) / (2 * pi)',
        'first_var': 'eta_shtrih',
        'second_var': 'm_qu_d'
    },
    {
        'name': 'Отношение масс: eta_shtrih / phi_meson',
        'formula': 'eta_shtrih_2_m_phi_meson',
        'expr': 'K**(3/2) / (2 * pi **(3/2) * 2**(1/2))',
        'first_var': 'eta_shtrih',
        'second_var': 'phi_meson'
    },
    {
        'name': 'Отношение масс: eta_shtrih / omega_meson',
        'formula': 'eta_shtrih_2_m_omega_meson',
        'expr': 'K**3 / (4 * pi**3 * 2**(1/2))',
        'first_var': 'eta_shtrih',
        'second_var': 'omega_meson'
    },
]

# Создаем таблицу
# Создаем таблицу
table_data = []
for rel in relations:
    formula = rel['formula']
    expr = rel['expr']

    # Вычисляем аналитическое значение
    # Вычисляем аналитическое значение
    try:
        # Всегда используем eval для expr, игнорируя переменные
        analytic_value = eval(expr)
    except:
        try:
            if formula in globals():
                analytic_value = globals()[formula]
            else:
                analytic_value = None
        except:
            analytic_value = None

    # Получаем значения величин
    first_const_name = var_to_name.get(rel['first_var'], rel['first_var'])
    second_const_name = var_to_name.get(rel['second_var'], rel['second_var'])

    first_value = constants.get(first_const_name, None)
    second_value = constants.get(second_const_name, None)

    # Реальное соотношение
    if first_value is not None and second_value is not None and second_value != 0:
        real_ratio = first_value / second_value
    else:
        real_ratio = None

    # Вычисляем Teta (расхождение)
    if analytic_value is not None and real_ratio is not None and real_ratio != 0:
        teta = analytic_value / real_ratio
    else:
        teta = None

    table_data.append({
        'Соотношение': rel['name'],
        'Аналитическая формула': expr,
        'Аналитическое значение': analytic_value,
        'Значение 1': first_value,
        'Значение 2': second_value,
        'Реальное соотношение': real_ratio,
        'Teta': teta
    })

# Создаем DataFrame
# Создаем DataFrame
df = pd.DataFrame(table_data)

# Создаем числовой столбец для сортировки
df['abs_dev'] = abs(df['Teta'] - 1)
df = df.sort_values(by='abs_dev', ascending=False)
df = df.drop(columns=['abs_dev'])
df = df.reset_index(drop=True)
df = df.sort_values(by='Teta', ascending=False).reset_index(drop=True)

# Настройка отображения
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 50)
pd.set_option('display.float_format', '{:.6e}'.format)

print("ТАБЛИЦА СООТНОШЕНИЙ МАСС И ВРЕМЕН ЖИЗНИ ЧАСТИЦ (отсортирована по величине расхождения Teta, K=6)")
print(df.to_string(index=False))

# Дополнительная статистика
print("СТАТИСТИКА ПО Teta:")
valid_teta = [t for t in df['Teta'] if t is not None]
if valid_teta:
    print(f"Количество соотношений с вычисленным Teta: {len(valid_teta)}")
    print(f"Среднее значение Teta: {sum(valid_teta) / len(valid_teta):.6e}")
    print(f"Минимальное Teta: {min(valid_teta):.6e}")
    print(f"Максимальное Teta: {max(valid_teta):.6e}")
    print(f"Количество Teta в диапазоне 0.9-1.1: {sum(1 for t in valid_teta if 0.9 <= t <= 1.1)}")
    print(f"Количество Teta в диапазоне 0.5-2.0: {sum(1 for t in valid_teta if 0.5 <= t <= 2.0)}")
