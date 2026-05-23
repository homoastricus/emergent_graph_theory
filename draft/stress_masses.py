import math

N=4.1977e+121
K = 6.0
pi = math.pi
e = math.e
lnK = math.log(K)
lnN = math.log(N)
N13 = N ** (1 / 3)
N16 = N ** (1 / 6)

p_val = 1 / (K * N ** (1 / 3))
Kp = K * p_val

m_neitrino = ((lnN) ** 2 * N ** (-1 / 3) * 2 ** (1 / 2)) / lnK


D0_lifetime = lnK / (2 * pi ** 2 * K ** 2 * (lnN) ** 4)
Λ_b_lifetime = lnK * 2 ** (1 / 2) / lnN ** 5
B_plus_lifetime = lnK * pi / 2 / lnN ** 5
neutron_lifetime = 2 ** (1 / 2) * N ** (1 / 12) / (lnN) ** 3
kaon_lifetime = 4 / (K ** (3 / 2) * lnN ** 3)
mu_lifetime = lnK / (K * 3 ** (1 / 2) * (lnN) ** 2)
tau_lifetime = 1 / (2 * (lnN) ** 5)
pion_lifetime = K ** 2 * 2 ** (1 / 2) * pi / (lnN) ** 4
D_plus_lifetime = 1 / (pi ** (1 / 2) * K ** (5 / 2) * lnN ** 4)


#lifetimes
Λ_b_2_B_plus_lifetime = 2**(1/2) / pi


#masses
m_qu_c_2_m_cu_t = 2*(pi**4)/K**6
m_Λ_barion_2_m_k0_meson = 4 * (pi ** 7/2)

m_z_bozon_2_m_w_bozon = 2/(pi ** (1/2))
m_w_bozon_2_m_Higgs = pi / (2 * 6 ** (1 / 2))
#m_proton_val_to_m_qu_b = math.sqrt(K) / math.sqrt(pi) *  math.sqrt(3)
m_proton_2_m_qu_b = math.sqrt(2) / math.sqrt(pi)
#m_Λ_barion_2_m_z_bozon = 3 * (2 ** (1 / 2)) / 2 * (pi ** (1 / 2))
m_Λ_barion_2_m_z_bozon = 3  / ((2*pi) ** (1 / 2))
m_DT_2_m_Λ_barion = (pi) ** (3 / 2) / 6 * 3 ** (1 / 2)
m_Higgs_2_m_D0 = 12 * pi ** (3/2)
m_D0_2_m_Υ_1S = (pi ** (1 / 2))/3*6
m_Υ_1S_2_m_qu_c = 3 ** (1 / 2) * (6**3)/ ((2 ** (1 / 2)) * 2*pi**2)
m_qu_c_2_m_qu_b = 2 * pi * (3 ** (1 / 2))/ (6 ** 2 )
m_qu_t_2_m_qu_b = 6**2 * (3 ** (1 / 2)) / pi**3

m_qu_t_2_m_pi_meson = 6 ** 3 * 4 / 2 ** (1 / 2)
m_w_bozon_2_m_proton_ = 2 * pi ** (3/2) * 6**(1/2)
m_pi_meson_2_m_k0_meson = pi ** (1 / 2)
m_z_bozon_2_m_proton = 4 * (pi ** 2) * 6 ** (1 / 2)
m_muon_2_m_tau = 2 * pi ** 3/2 * 3 ** (1 / 2) / 9

m_J_ψ_2_m_tau = 8 * pi ** (3 / 2) * 2 ** (1 / 2) / 6**2
m_J_ψ_2_m_eta = 4 * 2 ** (1 / 2)
m_eta_2_m_qu_d = 12 * pi ** 2 * 3 ** (1 / 2)
m_tau_2_m_eta = 18/pi ** (3 / 2)
m_eta_2_m_qu_u = 4 * pi ** 2 / 3 ** (1 / 2)
m_qu_d_2_m_qu_u = 8 * pi ** 2
m_muon_2_m_qu_u = 8 * pi ** 4 / 9

m_pi0_meson_2_m_qu_s = 2 * 6**2 / pi**(5/2)
m_pi0_meson_2_m_e_val = 6**(7/2) / 2
m_qu_s_2_m_e_val =  pi ** (5 / 2) * 6 ** (1 / 2) / 4





m_e_val = 4 * pi * lnN ** 4 / (K ** (1 / 2) * N ** (1 / 3))

m_proton_val = math.sqrt(pi) * (lnN ** 6) / (K ** (3 / 2) * (N ** (1 / 3)))

m_k0_meson = (lnN ** 6 * 1 / (4 * (pi ** 2)) * (2 * pi) ** (1 / 2)) / N ** (1 / 3)
m_DT = lnN ** 6 * (2 * pi) ** (1 / 2) / (K * 3 ** (1 / 2) * N ** (1 / 3))

m_Λ_barion = (lnN ** 6 * N ** (-1 / 3) * 2 ** (1 / 2)) / (pi ** 2)
m_z_bozon = (lnN ** 6) * 4 * (pi ** (5 / 2)) / (N ** (1 / 3) * K)
m_w_bozon = 2 * pi ** 3 * lnN ** 6 / (N ** (1 / 3) * K)

m_Higgs = lnN ** 6 * 4 * (pi ** (2)) / (N ** (1 / 3) * K ** (1 / 2))
m_D0 = lnN ** 6 * ((2 * pi) ** (1 / 2)) / (N ** (1 / 3) * K * (3 ** (1 / 2)))

# Kp * (lnN)^6 * √3 / √2
m_Υ_1S = lnN ** 6 * 3 ** (1 / 2) / (2 ** (1 / 2) * N ** (1 / 3))

m_qu_c = lnN ** 6 * 2 * pi ** 2 / (K ** 3 * N ** (1 / 3))

# (p * (lnN)^6 * π) / √3
m_qu_b = lnN ** 6 * pi / (K * (3 ** (1 / 2)) * N ** (1 / 3))

# ((lnN)^6 * N^(-1/3) * K^3) / π²
m_qu_t = lnN ** 6 * K ** 3 / (pi ** 2 * N ** (1 / 3))

m_pi_meson = lnN ** 6 * 1 / (4 * pi ** 2) * N ** (-1 / 3) / 2 ** (1 / 2)

m_muon = 4 * pi ** 2 * lnN ** 5 / (K * 3 ** (1 / 2) * N ** (1 / 3))

m_tau = pi ** (1 / 2) * (lnN ** 5) * (K ** 2) / (N ** (1 / 3))

m_pi0_meson = 2 * pi * K ** 3 * lnN ** 4 / N ** (1 / 3)  # (lnN) ** 6 * 1 / (4 * pi ** 2) * N ** (-1 / 3) / 2 ** (1 / 2)

m_J_ψ = lnN ** 5 * 8 * pi ** 2 * 2 ** (1 / 2) / N ** (1 / 3)

m_eta = lnN ** 5 * 2 * pi ** 2 / N ** (1 / 3)

# (lnN)^5 * N^(-1/3) * √3 / 4π²
m_qu_u = lnN ** 5 * 3 ** (1 / 2) / (4 * pi ** 2 * N ** (1 / 3))

# ((lnN)^5 * N^(-1/3) * 1/K) / √3
m_qu_d = lnN ** 5 / (K * 3 ** (1 / 2) * N ** (1 / 3))

# ((lnN)^4 * π:5/2 * π) / N^(1/3)
m_qu_s = lnN ** 4 * pi ** (7 / 2) / N ** (1 / 3)

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
    'm_Higgs':2.23319e-25,
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
    'm_neitrino': 1.783e-36
}