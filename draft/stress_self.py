import math
import numpy as np
from scipy.optimize import minimize_scalar

K = 6.0
pi = math.pi
e = math.e
lnK = math.log(K)
N =  4.197668e121

constants = {
    # Квантовые
    'ħ': 1.054571817e-34,        # Дж·с
    'h': 6.62607015e-34,         # Дж·с

    # Планковские
    't_P': 5.391247e-44,         # с
    'l_P': 1.616255e-35,         # м
    'm_P': 2.176434e-8,          # кг
    'E_P': 1.956082e9,           # Дж
    'T_P': 1.416784e32,          # К

    # Фундаментальные
    'c': 299792458,              # м/с
    'G': 6.67430e-11,            # м³/(кг·с²)
    'k_B': 1.380649e-23,         # Дж/К

    # Безразмерные
    'α': 1/137.035999084,        # постоянная тонкой структуры
    'm_e': 9.1093837015e-31,

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
    'RIDBERG': 1.097373e7,
    'bor_radius':  5.29177210903e-11,
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
    #'Gf': 1.436e-62,
    'm_neitrino': 1.783e-36
}


lnN = math.log(N)
N13 = N ** (1/3)
N16 = N ** (1/6)

# Вычисляем p для данного N
p_val = 1/(K*N**(1/3)) #compute_p(N)
Kp = K * p_val

m_neitrino = ((lnN)**2 * N**(-1/3) * 2**(1/2)) / lnK
Gf = (2**(1/2) * K**2 * lnN**2 * N**(2/3)/  (64 * pi**6  * lnN**13))
h2_connection_energy = 8 * pi * lnN**10 * lnK**2 / (K**(9/2) * N13)

D0_lifetime =  lnK / (2*pi**2 * K**2 * (lnN)**4)
Λ_b_lifetime = lnK * 2**(1/2)  / lnN**5
B_plus_lifetime = lnK * pi/2  / lnN**5
neutron_lifetime = 2**(1/2) * N**(1/12) / (lnN)**3
kaon_lifetime = 4 / (K**(3/2) * lnN**3)
mu_lifetime = lnK / (K * 3**(1/2) * (lnN)**2)
tau_lifetime = 1/(2*(lnN)**5)
pion_lifetime = K**2 * 2**(1/2)* pi/(lnN)**4
D_plus_lifetime = 1 / (pi**(1/2) *  K**(5/2) * lnN**4)

# Квантовые

fix_val = 5
# Скорость света
c_val = pi * (lnN ** 4) / (K**2 * lnK) #(1-1/(fix*lnN))

# Планковская длина и время
#lP_val = pi**2 * lnN**3 / (K**3 * lnK * N13)

fix_plus = (1+1/(fix_val*lnN))
fix_minus = (1-1/(fix_val*lnN))
fix_plus_sqrt = (1+1/(fix_val*lnN))**2
fix_minus_sqrt = (1-1/(fix_val*lnN))**2

lP_val = 4 * lnN ** 2 * lnK / N13 * fix_minus
#tP_val = pi / (K * N13 * lnN)
tP_val = 4 * K**2 * lnK**2 / (pi * N13 * lnN**2) * fix_minus
hbar_val = (lnN ** 3) / (K * N13) * fix_plus_sqrt
h_val = 2 * pi * hbar_val
print(f"h={constants['h']/h_val:.7f}")
print(f"t_P={constants['t_P']/tP_val:.7f}")
print(f"l_P={constants['l_P']/lP_val:.7f}")
print(f"c_val={constants['c']/c_val:.7f}")



# Планковская энергия
EP_val = (lnN ** 5) * pi / (4 * K**3  * lnK**2)

# Гравитационная постоянная (полностью эмерджентная)
G_val = 16 * pi**3 * lnN**13 / (K**5 * lnK * N**(1/3))

# Планковская масса (выводится через другие константы)
mP_val = K /(pi * 4 * lnN**3)

# Планковская температура
TP_val = 8 * pi * N13 / (lnN**4)

# Постоянная Больцмана (через правильное p)
k_B_val = Kp * (lnN**8) / (8 * pi**2)

# Постоянная тонкой структуры
alpha_val = 2 * lnK**2 / (pi * lnN)

m_e_val = 4*pi *  lnN ** 4 /  (K**(1/2) * N ** (1 / 3))

m_proton_val = math.sqrt(pi) * (lnN**6) / (K ** (3/2) * (N ** (1/3)))

ep_0_val = N ** (1 / 3) / (8 * pi ** 3 * lnK * lnN ** 20)

mu_0_val = (8 * pi * K**4 * lnK ** 3 * lnN ** 12) / N ** (1 / 3)

q_e_val = 1.0 / (pi * K ** (3 / 2) * lnN ** 7)

m_muon = 4*pi**2 * lnN**5 / (K * 3**(1/2) * N**(1/3))

m_tau = pi**(1/2) * (lnN ** 5) * (K**2)/ (N ** (1/3))

m_pi_meson = (lnN)**6 * 1 / (4*pi**2) * N**(-1 / 3) / 2**(1/2)

m_pi0_meson = 2 * pi * K**3 * lnN**4 / N**(1/3) #(lnN) ** 6 * 1 / (4 * pi ** 2) * N ** (-1 / 3) / 2 ** (1 / 2)


m_k0_meson = (lnN ** 6 * 1 / (4 * (pi ** 2)) * (2 * pi) ** (1 / 2)) / N ** (1 / 3)
m_DT = lnN ** 6 * (2 * pi) ** (1 / 2) / (K * 3 ** (1 / 2) * N ** (1 / 3))

m_Λ_barion = (lnN ** 6 * N ** (-1 / 3) * 2 ** (1 / 2)) / (pi ** 2)
m_z_bozon = (lnN**6) * 4 * (pi**(5/2)) / (N**(1/3) * K)
m_w_bozon = 2 * pi ** 3 * lnN ** 6 / (N ** (1 / 3) * K)

m_Higgs = lnN ** 6 * 4 * (pi ** (2)) / (N ** (1 / 3) * K ** (1 / 2))
m_D0 = lnN ** 6 * ((2 * pi) ** (1 / 2)) / (N ** (1 / 3) * K * (3 ** (1 / 2)))

m_J_ψ = lnN ** 5 * 8 * pi ** 2 * 2 ** (1 / 2) / N ** (1 / 3)

m_eta = lnN ** 5 * 2 * pi ** 2 / N ** (1 / 3)

# Kp * (lnN)^6 * √3 / √2
m_Υ_1S = lnN ** 6 * 3**(1/2) / (2**(1 / 2) * N**(1/3))
#(lnN)^5 * N^(-1/3) * √3 / 4π²
m_qu_u = lnN**5 * 3**(1/2) / (4 * pi**2  * N**(1/3))

#((lnN)^5 * N^(-1/3) * 1/K) / √3
m_qu_d = lnN ** 5 / (K * 3**(1/2) * N**(1 / 3))

#((lnN)^4 * π:5/2 * π) / N^(1/3)
m_qu_s = lnN**4 * pi**(7/2)/N**(1/3)

m_qu_c = lnN ** 6 * 2*pi**2 / (K**3 * N**(1/3))

#(p * (lnN)^6 * π) / √3
m_qu_b = lnN**6 * pi / (K * (3**(1/2)) * N**(1/3))

#((lnN)^6 * N^(-1/3) * K^3) / π²
m_qu_t = lnN**6 * K**3 /(pi**2 * N**(1/3))

# $R_\infty = 4 (\ln N)^3 (\ln K)^3} / {\pi K^{3/2}
RIDBERG = 4 * lnN**3 * lnK**3 / (pi * K**(3/2))

bor_radius = K**(3/2) / (8 * pi * lnN**4 * lnK)

impedance = 8 * K**2 * pi**2 * lnK**2 * lnN**16 / N**(1/3)

Φ0_magnetic_stream = lnN**10 * pi**2 * K**(1/2) / (N**(1/3))

m_proton_to_m_electron = lnN**2 / (4 * pi**(1/2) * K)

m_tau_m_electron = K**(5/2) * lnN / (4 * pi**(1/2))

# формулы масс частиц уже поделены, зафиксирован результат
m_W_to_m_Z = pi**(1/2) / 2

m_plank_to_m_e =  K**(3/2) * N**(1/3) / (16 * pi**2 * lnN**7)

compton_e = K**(3/2) * lnK / (2 * pi * lnN**5)

compton_proton = 2 * K ** (5/2) * lnK / (pi**(1/2) * lnN ** 7)

m_Higgs_to_m_W = 2 * K ** (1/2) / pi

Lambda_cosmo = lnN**12 /(pi**(1/2) * N**(2/3))

Einstein_constant = 128 * K**3 * lnK**3 / (lnN**3 * N**(1/3))

vacuum_higgs = lnN**6 * 8 * pi**(3/2) * 1 / (2**(1/2) * N**(1/3))


