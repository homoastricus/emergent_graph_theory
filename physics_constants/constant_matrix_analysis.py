"""
ETI Constants Matrix Complete Analysis Toolkit
Полный анализ 6-мерной алгебраической структуры фундаментальных констант.

Возможности:
  • Проверка ранга и ядра матрицы M (30×6)
  • LLL-редукция для поиска коротких инвариантов
  • χ²-тест с автоматической классификацией точных/нетривиальных тождеств
  • Логарифмическая калибровка ETI → CODATA
  • Корреляционный анализ отклонений
  • Извлечение NLO-поправок
  • Визуализация: PCA, тепловая карта, граф тождеств, гистограммы
Требования:
    pip install numpy scipy sympy matplotlib seaborn networkx uncertainties scikit-learn
"""

import json
import time
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sympy as sp
from scipy import stats as sp_stats
from sklearn.decomposition import PCA
from uncertainties import ufloat, nominal_value, std_dev

warnings.filterwarnings('ignore')

# 1. ДАННЫЕ: МАТРИЦА M (30 констант × 6 базисных примитивов)
# Столбцы: [s2, s3, s_pi, s_lnK, s_lnN, s_N]

M = sp.Matrix([
    # s2, s3, s_pi, s_lnK, s_lnN, s_N13

    # hbar = lnN^3 / (K × N13) = lnN^3 / (6 × N13)
    # 1/K = 1/6 = 2^(-1)×3^(-1) → s₂=-2, s₃=-2
    [-2, -2, 0, 0, 3, -1],  # hbar

    # h = 2π × hbar
    # +2 → s₂+=2
    [0, -2, 1, 0, 3, -1],  # h

    # c = π × lnN^4 / (K² × lnK)
    # 1/K² = 2^(-2)×3^(-2) → s₂=-4, s₃=-4
    # 1/lnK → s_lnK=-1
    [-4, -4, 1, -1, 4, 0],  # c

    # lP = 4 × lnN² × lnK / N13
    # 4 = 2² → s₂+=4
    # lnK → s_lnK=+1
    [4, 0, 0, 1, 2, -1],  # lP

    # tP = 4 × K² × lnK² / (π × N13 × lnN²)
    # 4 = 2² → s₂+=4
    # K² = 2²×3² → s₂+=4, s₃+=4
    # lnK² → s_lnK=+2
    # 1/π → s_π=-1
    [8, 4, -1, 2, -2, -1],  # tP

    # EP = lnN^5 × π / (4 × K³ × lnK²)
    # 1/4 = 2^(-2) → s₂-=2
    # 1/K³ = 2^(-3)×3^(-3) → s₂-=6, s₃-=6
    # 1/lnK² → s_lnK=-2
    [-10, -6, 1, -2, 5, 0],  # EP

    # G = 16 × π³ × lnN^13 / (K^5 × lnK × N13)
    # 16 = 2⁴ → s₂+=8
    # 1/K^5 = 2^(-5)×3^(-5) → s₂-=10, s₃-=10
    # 1/lnK → s_lnK=-1
    [-2, -10, 3, -1, 13, -1],  # G

    # mP = K / (4π × lnN³)
    # K = 2×3 → s₂+=2, s₃+=2
    # 1/4 = 2^(-2) → s₂-=2
    [-2, 2, -1, 0, -3, 0],  # mP

    # TP = 8π × N13 / lnN⁴
    # 8 = 2³ → s₂+=6
    [6, 0, 1, 0, -4, 1],  # TP

    # kB = lnN⁸ / (8π² × N13)
    # 1/8 = 2^(-3) → s₂-=6
    [-6, 0, -2, 0, 8, -1],  # kB

    # alpha = 2 × lnK² / (π × lnN)
    # 2 → s₂+=2
    # lnK² → s_lnK=+2
    [2, 0, -1, 2, -1, 0],  # alpha

    # me = 4π × lnN⁴ / (K^(1/2) × N13)
    # 4 = 2² → s₂+=4
    # 1/K^(1/2) = 2^(-1/2)×3^(-1/2) → s₂-=1, s₃-=1
    [3, -1, 1, 0, 4, -1],  # me

    # mp = √π × lnN⁶ / (K^(3/2) × N13)
    # 1/K^(3/2) = 2^(-3/2)×3^(-3/2) → s₂-=3, s₃-=3
    [-3, -3, sp.Rational(1, 2), 0, 6, -1],  # mp

    # eps0 = N13 / (8π³ × lnK × lnN²⁰)
    # 1/8 = 2^(-3) → s₂-=6
    # 1/lnK → s_lnK=-1
    [-6, 0, -3, -1, -20, 1],  # eps0

    # mu0 = 8π × K⁴ × lnK³ × lnN¹² / N13
    # 8 = 2³ → s₂+=6
    # K⁴ = 2⁴×3⁴ → s₂+=8, s₃+=8
    # lnK³ → s_lnK=+3
    [14, 8, 1, 3, 12, -1],  # mu0

    # qe = 1 / (π × K^(3/2) × lnN⁷)
    # 1/K^(3/2) = 2^(-3/2)×3^(-3/2) → s₂-=3, s₃-=3
    [-3, -3, -1, 0, -7, 0],  # qe

    # Rinf = 4 × lnN³ × lnK³ / (π × K^(3/2))
    # 4 = 2² → s₂+=4
    # lnK³ → s_lnK=+3
    # 1/K^(3/2) = 2^(-3/2)×3^(-3/2) → s₂-=3, s₃-=3
    [1, -3, -1, 3, 3, 0],  # Rinf

    # a0 = K^(3/2) / (8π × lnN⁴ × lnK)
    # K^(3/2) = 2^(3/2)×3^(3/2) → s₂+=3, s₃+=3
    # 1/8 = 2^(-3) → s₂-=6
    # 1/lnK → s_lnK=-1
    [-3, 3, -1, -1, -4, 0],  # a0

    # Z0 = 8 × K² × π² × lnK² × lnN¹⁶ / N13
    # 8 = 2³ → s₂+=6
    # K² = 2²×3² → s₂+=4, s₃+=4
    # lnK² → s_lnK=+2
    [10, 4, 2, 2, 16, -1],  # Z0

    # Phi0 = lnN¹⁰ × π² × K^(1/2) / N13
    # K^(1/2) = 2^(1/2)×3^(1/2) → s₂+=1, s₃+=1
    [1, 1, 2, 0, 10, -1],  # Phi0

    # lambda_e = K^(3/2) × lnK / (2π × lnN⁵)
    # K^(3/2) = 2^(3/2)×3^(3/2) → s₂+=3, s₃+=3
    # lnK → s_lnK=+1
    # 1/2 = 2^(-1) → s₂-=2
    [1, 3, -1, 1, -5, 0],  # lambda_e

    # lambda_p = 2 × K^(5/2) × lnK / (√π × lnN⁷)
    # 2 → s₂+=2
    # K^(5/2) = 2^(5/2)×3^(5/2) → s₂+=5, s₃+=5
    # lnK → s_lnK=+1
    [7, 5, -sp.Rational(1, 2), 1, -7, 0],  # lambda_p

    # vH = lnN⁶ × 8 × π^(3/2) / (√2 × N13)
    # 8 = 2³ → s₂+=6
    # 1/√2 = 2^(-1/2) → s₂-=1
    [5, 0, sp.Rational(3, 2), 0, 6, -1],  # vH

    # muon = 4π² × lnN⁵ / (K × √3 × N13)
    # 4 = 2² → s₂+=4
    # 1/K = 2^(-1)×3^(-1) → s₂-=2, s₃-=2
    # 1/√3 = 3^(-1/2) → s₃-=1
    [2, -3, 2, 0, 5, -1],  # muon

    # tau = √π × lnN⁵ × K² / N13
    # K² = 2²×3² → s₂+=4, s₃+=4
    [4, 4, sp.Rational(1, 2), 0, 5, -1],  # tau

    # pion_pm = lnN⁶ / (4π² × √2 × N13)
    # 1/4 = 2^(-2) → s₂-=4
    # 1/√2 = 2^(-1/2) → s₂-=1
    [-5, 0, -2, 0, 6, -1],  # pion_pm

    # pion0 = 2π × K³ × lnN⁴ / N13
    # 2 → s₂+=2
    # K³ = 2³×3³ → s₂+=6, s₃+=6
    [8, 6, 1, 0, 4, -1],  # pion0

    # Lambda = lnN¹² / (√π × N^(2/3))
    [0, 0, -sp.Rational(1, 2), 0, 12, -2],  # Lambda

    # kappa = 128 × K³ × lnK³ / (lnN³ × N13)
    # 128 = 2⁷ → s₂+=14
    # K³ = 2³×3³ → s₂+=6, s₃+=6
    # lnK³ → s_lnK=+3
    [20, 6, 0, 3, -3, -1],  # kappa

    # mH_mW = 2 × K^(1/2) / π
    # 2 → s₂+=2
    # K^(1/2) = 2^(1/2)×3^(1/2) → s₂+=1, s₃+=1
    [3, 1, -1, 0, 0, 0],  # mH_mW

    # ============ КВАРКИ ============
    # m_qu_u = lnN^5 * 3^(1/2) / (4 * pi^2 * N^(1/3))
    [-4, 1, -2, 0, 5, -1],  # m_qu_u

    # m_qu_d = lnN^5 / (K * 3^(1/2) * N^(1/3))
    [-2, -3, 0, 0, 5, -1],  # m_qu_d

    # m_qu_s = lnN^4 * pi^(7/2) / N^(1/3)
    [2, 2, -sp.Rational(1, 2), 0, 5, -1],  # m_qu_s

    # m_qu_c = lnN^6 * 2 * pi^2 / (K^3 * N^(1/3))
    [-4, -6, 2, 0, 6, -1],  # m_qu_c

    # m_qu_b = lnN^6 * pi / (K * 3^(1/2) * N^(1/3))
    [-2, -3, 1, 0, 6, -1],  # m_qu_b

    # m_qu_t = lnN^6 * K^3 / (pi^2 * N^(1/3))
    [6, 6, -2, 0, 6, -1],  # m_qu_t

    # ============ АТОМНЫЕ КОНСТАНТЫ ============
    # m_proton_to_m_electron = lnN^2 / (4 * pi^(1/2) * K)
    [-6, -2, -sp.Rational(1, 2), 0, 2, 0],  # m_proton_to_m_electron

    # m_tau_m_electron = K^(5/2) * lnN / (4 * pi^(1/2))
    [1, 5, -sp.Rational(1, 2), 0, 1, 0],  # m_tau_m_electron

    # ============ БАРИОНЫ ============
    # m_Λ_barion = lnN^6 * N^(-1/3) * 2^(1/2) / pi^2
    [1, 0, -2, 0, 6, -1],  # m_Λ_barion

    # m_z_bozon = lnN^6 * 4 * pi^(5/2) / (N^(1/3) * K)
    [2, -2, sp.Rational(5, 2), 0, 6, -1],  # m_z_bozon

    # m_w_bozon = 2 * pi^3 * lnN^6 / (N^(1/3) * K)
    [0, -2, 3, 0, 6, -1],  # m_w_bozon

    # m_D0 = lnN^6 * (2*pi)^(1/2) / (N^(1/3) * K * 3^(1/2))
    [-1, -3, sp.Rational(1, 2), 0, 6, -1],  # m_D0

    # m_J_ψ = lnN^5 * 8 * pi^2 * 2^(1/2) / N^(1/3)
    [7, 0, 2, 0, 5, -1],  # m_J_ψ

    # m_eta = lnN^5 * 2 * pi^2 / N^(1/3)
    [2, 0, 2, 0, 5, -1],  # m_eta

    # r_electron = lnK^3 * K^(3/2) / (2 * pi^3 * lnN^6)
    [1, 3, -3, 3, -6, 0],  # r_electron

    # ============ ВРЕМЕНА ЖИЗНИ ============
    # D0_lifetime = lnK / (2 * pi^2 * K^2 * lnN^4)
    [-6, -4, -2, 1, -4, 0],  # D0_lifetime

    # Λ_b_lifetime = lnK * 2^(1/2) / lnN^5
    [1, 0, 0, 1, -5, 0],  # Λ_b_lifetime

    # B_plus_lifetime = lnK * pi / (2 * lnN^5)
    [-2, 0, 1, 1, -5, 0],  # B_plus_lifetime

    # kaon_lifetime = 4 / (K^(3/2) * lnN^3)
    [1, -3, 0, 0, -3, 0],  # kaon_lifetime

    # mu_lifetime = lnK / (K * 3^(1/2) * lnN^2)
    [-2, -3, 0, 1, -2, 0],  # mu_lifetime

    # tau_lifetime = 1 / (2 * lnN^5)
    [-2, 0, 0, 0, -5, 0],  # tau_lifetime

    # pion_lifetime = K^2 * 2^(1/2) * pi / lnN^4
    [5, 4, 1, 0, -4, 0],  # pion_lifetime

    # D_plus_lifetime = 1 / (pi^(1/2) * K^(5/2) * lnN^4)
    [-5, -5, -sp.Rational(1, 2), 0, -4, 0],  # D_plus_lifetime

    # m_Higgs = lnN^6 * 4 * pi^2 / (N^(1/3) * K^(1/2))
    # = lnN^6 * 2^2 * pi^2 * K^(-1/2) * N^(-1/3)
    # 2^2 → s₂+=4
    # K^(-1/2) = 2^(-1/2)×3^(-1/2) → s₂-=1, s₃-=1
    # s₂ = 4 - 1 = 3, s₃ = -1, s_π = 2, s_lnK = 0, s_lnN = 6, s_N13 = -1
    [3, -1, 2, 0, 6, -1],  # m_Higgs

    # m_neitrino = lnN^2 * N^(-1/3) * 2^(1/2) / lnK
    # = lnN^2 * N^(-1/3) * 2^(+1/2) * lnK^(-1)
    # s₂ = 1, s₃ = 0, s_π = 0, s_lnK = -1, s_lnN = 2, s_N13 = -1
    [1, 0, 0, -1, 2, -1],  # m_neitrino

    # Sigma_plus = K × lnN⁶ / (4π² × N^{1/3})
    # K = (√2)²(√3)² → s₂=2, s₃=2
    # 1/4 = (√2)⁻⁴ → s₂-=4 → s₂=-2
    # 1/π² → s_π=-2
    [-2, 2, -2, 0, 6, -1],  # Sigma_plus (Σ⁺)

    # Ksi_0 = √(2π) × lnN⁶ / (K^{3/2} × N^{1/3})
    # √(2π) = (√2)¹ × π^{1/2} → s₂=1, s_π=0.5
    # K^{-3/2} = (√2)⁻³(√3)⁻³ → s₂-=3, s₃=-3 → s₂=-2
    [-2, -3, 0.5, 0, 6, -1],  # Ksi_0 (Ξ⁰)
    #
    # Omega_minus = π × lnN⁶ / (K^{3/2} × N^{1/3})
    # π → s_π=1
    # K^{-3/2} = (√2)⁻³(√3)⁻³
    [-3, -3, 1, 0, 6, -1],  # Omega_minus (Ω⁻)
    #
    # Ksi_plus = lnN⁶ / (π × N^{1/3})
    # π^{-1} → s_π=-1
    [0, 0, -1, 0, 6, -1],  # Ksi_plus (Ξ⁺)
    #
    # Omega0_c = K × lnN⁶ / (π^{5/2} × N^{1/3})
    # K = (√2)²(√3)² → s₂=2, s₃=2
    # π^{-5/2} → s_π=-2.5
    #[2, 2, -2.5, 0, 6, -1],  # Omega0_c (Ω_c⁰)

    # ((lnN)^5 * 4π² * √K) / N^(1/3) ⭐⭐
    # Значение: 4.797501566675e-27
    # Цель: omega_zero = 4.808000000000e-27
    # Отн. ошибка: 0.218353%
    [5, 1, 2, 0, 5, -1],

    #
    # lambda_B0 = √π × lnN⁶ / (K^{1/2} × N^{1/3})
    # √π → s_π=0.5
    # K^{-1/2} = (√2)⁻¹(√3)⁻¹
    #  ((lnN)^5 * π^3/2 * N^(-1/3)) / 1/K^2
    [-1, -1, 0.5, 0, 6, -1],  # lambda_B0 (Λ_b⁰)
    #
    # # Ksi_minus = √(2π) × lnN⁶ / (K^{3/2} × N^{1/3})
    [-2, -3, 0.5, 0, 6, -1],  # Ksi_minus (Ξ⁻)
    #
    # Sigma_minus = K × lnN⁶ / (4π² × N^{1/3})
    [-2, 2, -2, 0, 6, -1],  # Sigma_minus (Σ⁻)

    # phi_meson = K^{3/2} × √(2π) × lnN⁵ / N^{1/3}
    # K^{3/2} = (√2)³(√3)³ → s₂=3, s₃=3
    # √(2π) = (√2)¹ × π^{1/2} → s₂+=1, s_π=0.5
    [4, 3, 0.5, 0, 5, -1],  # phi_meson (φ)

    # omega_meson = 2√2 × π² × lnN⁵ / N^{1/3}
    # 2√2 = (√2)³ → s₂=3
    # π² → s_π=2
    [3, 0, 2, 0, 5, -1],  # omega_meson (ω)

    # eta_shtrih = K³ × lnN⁵ / (2π × N^{1/3})
    # K³ = (√2)⁶(√3)⁶ → s₂=6, s₃=6
    # 1/2 = (√2)⁻² → s₂-=2 → s₂=4
    # 1/π → s_π=-1
    [4, 6, -1, 0, 5, -1],  # eta_shtrih (η′)

    # rho_meson = √3 × π^{5/2} × lnN⁵ / N^{1/3}
    # √3 → s₃=1
    # π^{5/2} → s_π=2.5
    [0, 1, 2.5, 0, 5, -1],  # rho_meson (ρ)

    # K_star = 2 × lnN⁶ / (π^{5/2} × N^{1/3})
    # 2 = (√2)² → s₂=2
    # π^{-5/2} → s_π=-2.5
    #[2, 0, -2.5, 0, 6, -1],  # K_star (K*)

    #neutron_lifetime
    [0, 0, 1, 0, 1, 0],

    # h2_connection_energy = 8 * pi * lnN ** 10 * lnK ** 2 / (K ** (9 / 2) * N13)
    [-3, -9, 1, 2, 10, -1],
])

NAMES = [
    "hbar", "h", "c", "lP", "tP", "EP", "G", "mP", "TP", "kB",
    "alpha", "me", "mp", "eps0", "mu0", "qe", "Rinf", "a0",
    "Z0", "Phi0", "lambda_e", "lambda_p", "vH", "muon", "tau",
    "pion_pm", "pion0", "Lambda", "kappa", "mH_mW",
    "m_qu_u", "m_qu_d", "m_qu_s", "m_qu_c", "m_qu_b", "m_qu_t",
    "m_proton_to_m_electron", "m_tau_m_electron",
    "m_Λ_barion", "m_z_bozon", "m_w_bozon", "m_D0",
    "m_J_ψ", "m_eta", "r_electron",
    "D0_lifetime", "Λ_b_lifetime", "B_plus_lifetime", "kaon_lifetime",
    "mu_lifetime", "tau_lifetime", "pion_lifetime", "D_plus_lifetime",
    "m_Higgs", "m_neitrino",
    "Sigma_plus", "Ksi_0", "Omega_minus", "Ksi_plus",
    "Omega0_c",
    "lambda_B0",
    "Ksi_minus",
    "Sigma_minus","phi_meson", "omega_meson",
    "eta_shtrih","rho_meson", "neutron_lifetime",
    "h2_connection_energy"
    #"K_star_meson",
]

BASIS_NAMES = ["√2", "√3", "π", "ln K", "ln N", "N^(1/3)"]

# ============================================================
# 2. ПАРАМЕТРЫ И БАЗИС
# ============================================================

K = 6.0
N_val = 4.1847e121

BASIS = [
    np.sqrt(2),
    np.sqrt(3),
    np.pi,
    np.log(K),
    np.log(N_val),
    N_val ** (1.0 / 3.0)
]

# ============================================================
# 3. CODATA 2022
# ============================================================

CODATA = {
    "hbar": (1.054571817e-34, 1.2e-10),
    "h": (6.62607015e-34, 0),
    "c": (299792458, 0),
    "G": (6.67430e-11, 2.2e-5),
    "kB": (1.380649e-23, 0),
    "qe": (1.602176634e-19, 0),
    "alpha": (7.2973525693e-3, 1.5e-10),
    "eps0": (8.8541878128e-12, 1.5e-10),
    "mu0": (1.25663706212e-6, 1.5e-10),
    "Z0": (376.730313668, 1.5e-10),
    "Phi0": (2.067833848e-15, 1.5e-10),
    "me": (9.1093837015e-31, 3.0e-10),
    "mp": (1.67262192369e-27, 3.0e-10),
    "Rinf": (10973731.568160, 1.9e-12),
    "a0": (5.29177210903e-11, 1.5e-10),
    "lambda_e": (2.42631023867e-12, 1.5e-10),
    "lambda_p": (1.32140985538e-15, 1.5e-10),
    "lP": (1.616255e-35, 1.1e-5),
    "tP": (5.391247e-44, 1.1e-5),
    "EP": (1.956082e9, 1.1e-5),
    "mP": (2.176434e-8, 1.1e-5),
    "TP": (1.416784e32, 1.1e-5),
    "muon": (1.883531627e-28, 1.5e-10),
    "tau": (3.16747e-27, 5.0e-4),
    "pion_pm": (2.488089e-28, 1.0e-5),
    "pion0": (2.406090e-28, 1.0e-5),
    "Lambda": (1.089e-52, 1.0e-2),
    "kappa": (2.07664746e-43, 2.2e-5),
    "mH_mW": (1.558, 1.0e-3),
    "vH": (4.388471e-25, 1.0e-3),
    'm_qu_u': (2.1650e-30, 1.0e-2),
    'm_qu_d': (4.7915e-30, 1.0e-2),
    'm_qu_s': (9.635e-30, 1.0e-2),
    'm_qu_c': (1.27e-27, 1.0e-2),
    'm_qu_b': (4.180e-27, 1.0e-2),
    'm_qu_t': (3.04e-25, 1.0e-2),
    'm_proton_to_m_electron': (1836.152673426, 1.0e-3),
    'm_tau_m_electron': (3477, 1.0e-2),
    'm_eta': (9.767732e-28, 1.0e-4),
    'm_Λ_barion': (1.9901611e-27, 1.0e-4),
    'm_z_bozon': (1.62614e-25, 1.0e-3),
    'm_w_bozon': (1.43362e-25, 1.0e-3),
    # 'm_Higgs': (2.23319e-25, 1.0e-3),
    'm_D0': (3.32479e-27, 1.0e-3),
    'm_J_ψ': (5.52061e-27, 1.0e-3),
    'r_electron': (2.8179402853e-15, 1.0e-5),
    'mu_lifetime': (2.1969811e-6, 1.0e-4),
    'tau_lifetime': (2.903e-13, 1.0e-2),
    'pion_lifetime': (2.6033e-8, 1.0e-2),
    'kaon_lifetime': (1.2380e-8, 1.0e-2),
    'D_plus_lifetime': (1.040e-12, 1.0e-2),
    'B_plus_lifetime': (1.638e-12, 1.0e-2),
    'Λ_b_lifetime': (1.471e-12, 1.0e-2),
    'D0_lifetime': (4.101e-13, 1.0e-2),
    'm_Higgs': (2.23319e-25, 1.0e-3),
    'm_neitrino': (1.783e-36, 1.0e-1),
    'Sigma_plus': (2.11933e-27, 1.0e-2),
    'Ksi_0': (2.34532e-27, 1.0e-2),
    'Omega_minus': (2.9859e-27, 1.0e-2),
    'Ksi_plus': (4.3995e-27, 1.0e-2),
    'Omega0_c': (4.808e-27, 1.0e-2),
    'lambda_B0': (1.0023e-26, 1.0e-2),
    'Ksi_minus': (2.358e-27, 1.0e-2),
    'Sigma_minus': (2.132e-27, 1.0e-2),
    'phi_meson': (1.819e-27, 1.0e-2),
    'omega_meson': (1.394e-27, 1.0e-2),
    'eta_shtrih': (1.7086e-27, 1.0e-2),
    'rho_meson': (1.49e-27, 1.0e-2),
    'K_star_meson': (1.59e-27, 1.0e-2),
    'neutron_lifetime': (877.8, 1.0e-2),
    'h2_connection_energy': (2.178872e-18, 1.0e-5)
}

print("\n" + "=" * 80)
print("ДИАГНОСТИКА: ПРЯМОЕ СРАВНЕНИЕ БЕЗРАЗМЕРНЫХ ОТНОШЕНИЙ")
print("=" * 80)


# Вычисляем ETI-значения
def compute_eti_value(s_vector):
    result = 1.0
    basis = [np.sqrt(2), np.sqrt(3), np.pi, np.log(K), np.log(N_val), N_val ** (1 / 3)]
    for s, b in zip(s_vector, basis):
        if s != 0:
            result *= b ** float(s)
    return result


# Сравниваем ОТНОШЕНИЯ, которые не зависят от единиц
test_ratios = [
    # (имя1, имя2, описание)
    ("alpha", "mH_mW", "α vs mH/mW (обе безразмерные)"),
    ("me", "mp", "me/mp"),
    ("muon", "me", "mμ/me"),
    ("tau", "muon", "mτ/mμ"),
    ("muon", "mp", "mμ/mp"),
    ("pion_pm", "mp", "mπ±/mp"),
    ("vH", "mp", "vH/mp"),
    ("hbar", "c", "ℏ/c (размерное, но отношение ETI должно совпадать)"),
]

print(f"\n{'Отношение':20s} {'ETI':15s} {'CODATA':15s} {'Отклонение':15s}")
print("-" * 70)

for name1, name2, desc in test_ratios:
    i1, i2 = NAMES.index(name1), NAMES.index(name2)

    s1 = [float(M[i1, j]) for j in range(6)]
    s2 = [float(M[i2, j]) for j in range(6)]

    eti1 = compute_eti_value(s1)
    eti2 = compute_eti_value(s2)
    ratio_eti = eti1 / eti2

    codata1 = CODATA[name1][0]
    codata2 = CODATA[name2][0]
    ratio_codata = codata1 / codata2

    if ratio_codata != 0:
        dev = (ratio_eti / ratio_codata - 1) * 100
        print(f"{desc:20s} {ratio_eti:15.6e} {ratio_codata:15.6e} {dev:+10.4f}%")
    else:
        print(f"{desc:20s} {ratio_eti:15.6e} {ratio_codata:15.6e} N/A")

# Отдельно проверим α
print(f"\n--- Проверка α ---")
i_alpha = NAMES.index("alpha")
s_alpha = [float(M[i_alpha, j]) for j in range(6)]
alpha_eti = compute_eti_value(s_alpha)
alpha_codata = CODATA["alpha"][0]

print(f"α (ETI)     = {alpha_eti:.10f}")
print(f"α (CODATA)  = {alpha_codata:.10f}")
print(f"Отношение   = {alpha_eti / alpha_codata:.10f}")
print(f"Отклонение  = {(alpha_eti / alpha_codata - 1) * 100:.6f}%")

# Из α можно найти оптимальное ln N
# α = 2 * (ln K)² / (π * ln N)
ln_K = np.log(K)
ln_N_optimal = 2 * ln_K ** 2 / (np.pi * alpha_codata)
print(f"\nОптимальное ln N (из α): {ln_N_optimal:.4f}")
print(f"Текущее ln N:              {np.log(N_val):.4f}")
print(f"Отношение:                 {ln_N_optimal / np.log(N_val):.6f}")

# А также найдём оптимальное N
N_optimal = np.exp(ln_N_optimal)
print(f"Оптимальное N:  {N_optimal:.6e}")
print(f"Текущее N:      {N_val:.6e}")


# ============================================================
# 4. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def compute_predicted_value(s_vector, basis=BASIS):
    """Вычисляет предсказанное значение константы по вектору показателей."""
    result = 1.0
    for s, b in zip(s_vector, basis):
        if s != 0:
            result *= b ** float(s)
    return result


def compute_identity_value(kernel_vec, constants_dict):
    """Вычисляет значение тождества ∏ C_i^{v_i} с неопределённостью."""
    log_val = 0.0
    log_unc_sq = 0.0

    for coef, name in zip(kernel_vec, NAMES):
        if coef != 0 and name in constants_dict:
            val, rel_unc = constants_dict[name]
            if val <= 0:
                return ufloat(float('inf'), 0)
            power = int(coef)
            log_val += power * np.log(val)
            if rel_unc > 0:
                log_unc_sq += (power * rel_unc) ** 2

    log_unc = np.sqrt(log_unc_sq)

    try:
        nominal = np.exp(log_val)
        unc = nominal * log_unc
        return ufloat(nominal, unc)
    except OverflowError:
        return ufloat(float('inf'), 0)


def vector_to_expression(vec, names, threshold=0.01):
    """Преобразует вектор коэффициентов в читаемое выражение."""
    lhs, rhs = [], []
    for coef, name in zip(vec, names):
        if abs(coef) < threshold:
            continue
        if coef > 0:
            lhs.append(name if abs(coef - 1) < 1e-10 else f"{name}^{int(coef)}")
        elif coef < 0:
            c = abs(coef)
            rhs.append(name if abs(c - 1) < 1e-10 else f"{name}^{int(c)}")
    lhs_str = " × ".join(lhs) if lhs else "1"
    rhs_str = " × ".join(rhs) if rhs else "1"
    return f"{lhs_str} = {rhs_str}"


def get_kernel_vectors():
    """Извлекает целочисленный базис ядра матрицы M."""
    kernel = M.T.nullspace()
    vectors = []
    for vec in kernel:
        denoms = [sp.fraction(term)[1] for term in vec]
        lcm = sp.ilcm(*denoms) if denoms else 1
        ivec = [int(sp.simplify(v * lcm)) for v in vec]
        gcd = abs(sp.igcd(*ivec)) if any(ivec) else 1
        ivec = [v // gcd for v in ivec]
        vectors.append(ivec)
    return vectors


def matrix_to_float(sp_mat):
    """Конвертирует sympy-матрицу в numpy float."""
    rows, cols = sp_mat.shape
    arr = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            arr[i, j] = float(sp_mat[i, j])
    return arr


# ============================================================
# 5. АНАЛИЗ ЯДРА И LLL
# ============================================================

def analyze_kernel():
    """Полный анализ ядра матрицы M."""

    print("=" * 80)
    print("ЧАСТЬ 1: АНАЛИЗ ЯДРА И LLL-РЕДУКЦИЯ")
    print("=" * 80)

    t0 = time.time()

    rank = M.rank()
    print(f"\n✓ Ранг матрицы M: {rank} (ожидается 6)")
    print(f"✓ Размер матрицы: {M.rows} × {M.cols}")
    print(f"✓ Размерность ядра: {M.rows - rank}")

    M_rref, pivots = M.rref()
    print(f"✓ Пивоты RREF: {pivots}")

    kernel_vectors = get_kernel_vectors()
    print(f"\n✓ Число независимых тождеств: {len(kernel_vectors)}")

    # LLL-редукция
    print("\n" + "-" * 80)
    print("LLL-РЕДУКЦИЯ")
    print("-" * 80)

    K_int = sp.Matrix(kernel_vectors).T
    K_for_lll = K_int.T
    K_lll = K_for_lll.lll()

    print(f"✓ LLL-редуцированный базис: {K_lll.shape}")

    short_vectors = []
    for i in range(K_lll.rows):
        vec = [int(v) for v in K_lll.row(i)]
        norm_sq = sum(v * v for v in vec)
        max_abs = max(abs(v) for v in vec)
        if max_abs <= 15 and any(v != 0 for v in vec):
            short_vectors.append((norm_sq, vec, i))

    short_vectors.sort()

    print(f"\n✓ Найдено {len(short_vectors)} коротких инвариантов:")
    for rank_idx, (nsq, vec, orig_idx) in enumerate(short_vectors, 1):
        expr = vector_to_expression(vec, NAMES)
        print(f"  #{rank_idx} (‖v‖²={nsq}): {expr}")

    print(f"\n⏱ Время: {time.time() - t0:.2f} с")
    return kernel_vectors, short_vectors


# ============================================================
# 6. χ²-ТЕСТ
# ============================================================

def chi_squared_test(kernel_vectors):
    """χ²-тест для нетривиальных тождеств."""

    print("\n" + "=" * 80)
    print("ЧАСТЬ 2: χ²-ТЕСТ")
    print("=" * 80)

    exact_indices = set()
    nontrivial_indices = set()

    for i, vec in enumerate(kernel_vectors):
        norm_sq = sum(v * v for v in vec)
        has_masses = any(vec[NAMES.index(name)] != 0 for name in
                         ['me', 'mp', 'muon', 'tau', 'pion_pm', 'pion0', 'vH']
                         if name in NAMES)
        has_couplings = any(vec[NAMES.index(name)] != 0 for name in
                            ['alpha', 'G', 'kB', 'TP'] if name in NAMES)

        if not has_masses and not has_couplings and norm_sq <= 20:
            exact_indices.add(i)
        elif not has_masses and norm_sq <= 3:
            exact_indices.add(i)
        else:
            nontrivial_indices.add(i)

    print(f"\n✓ Точных определений: {len(exact_indices)}")
    print(f"✓ Нетривиальных гипотез: {len(nontrivial_indices)}")

    chi2_total = 0
    z_scores = []
    identity_results = []

    for i in sorted(nontrivial_indices):
        vec = kernel_vectors[i]
        identity_val = compute_identity_value(vec, CODATA)

        nominal = nominal_value(identity_val)
        unc = std_dev(identity_val)

        if np.isnan(nominal) or np.isnan(unc) or np.isinf(nominal) or np.isinf(unc):
            continue

        expected = 1.0

        if unc > 1e-100:
            z = (nominal - expected) / unc
            if abs(z) > 1e6:
                z = np.sign(z) * 1e6
        else:
            z = 0.0 if abs(nominal - expected) < 1e-10 else 100.0

        if unc > 1e-100:
            chi2_contrib = min(((nominal - expected) / unc) ** 2, 1e6)
        else:
            chi2_contrib = 0.0 if abs(nominal - expected) < 1e-10 else 1e4

        chi2_total += chi2_contrib
        z_scores.append(z)

        identity_results.append({
            'idx': i,
            'expression': vector_to_expression(vec, NAMES),
            'nominal': nominal,
            'unc': unc,
            'z': z,
            'chi2': chi2_contrib,
            'norm_sq': sum(v * v for v in vec)
        })

    n_tested = len(identity_results)
    nu = n_tested
    chi2_reduced = chi2_total / nu if nu > 0 else 0
    p_value = sp_stats.chi2.sf(chi2_total, nu) if nu > 0 else 1.0

    print(f"\n✓ Проверено тождеств: {n_tested}")
    print(f"✓ χ² = {chi2_total:.2f}, ν = {nu}, χ²/ν = {chi2_reduced:.2f}")
    print(f"✓ p-value = {p_value:.4f}")

    n_low = sum(1 for z in z_scores if abs(z) <= 1)
    n_moderate = sum(1 for z in z_scores if 1 < abs(z) <= 3)
    n_high = sum(1 for z in z_scores if abs(z) > 3)

    print(f"\n✓ Распределение Z-scores:")
    print(f"  |Z| ≤ 1: {n_low} ({100 * n_low / n_tested:.1f}%)")
    print(f"  1 < |Z| ≤ 3: {n_moderate} ({100 * n_moderate / n_tested:.1f}%)")
    print(f"  |Z| > 3: {n_high} ({100 * n_high / n_tested:.1f}%)")

    print(f"\n✓ Топ-3 лучших:")
    best = sorted(identity_results, key=lambda x: abs(x['z']))[:3]
    for rank, r in enumerate(best, 1):
        print(f"  #{rank} [{r['idx']}] Z = {r['z']:+.3f}")
        print(f"      {r['expression'][:100]}...")

    print(f"\n⚠ Топ-3 худших:")
    worst = sorted(identity_results, key=lambda x: abs(x['z']), reverse=True)[:3]
    for rank, r in enumerate(worst, 1):
        print(f"  #{rank} [{r['idx']}] Z = {r['z']:+.3f}")
        print(f"      {r['expression'][:100]}...")

    # Диагноз
    print(f"\n--- Диагноз ---")
    if chi2_reduced < 2:
        print(f"✅ Хорошее согласие")
    elif chi2_reduced < 10:
        print(f"⚠ Умеренное напряжение")
    else:
        print(f"❌ Сильное расхождение (LO-точность ~0.1–0.5%)")

    if z_scores:
        z_finite = [z for z in z_scores if np.isfinite(z) and abs(z) < 10]
        if z_finite:
            plt.figure(figsize=(8, 5))
            plt.hist(z_finite, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
            plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
            plt.xlabel('Z-score')
            plt.ylabel('Число тождеств')
            plt.title(f'Z-scores ({n_tested} тождеств)')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('z_scores_histogram.png', dpi=300)
            print("✓ Сохранено: z_scores_histogram.png")

    return identity_results


# ============================================================
# 7. ЛОГАРИФМИЧЕСКАЯ КАЛИБРОВКА И АНАЛИЗ ОТКЛОНЕНИЙ
# ============================================================

def logarithmic_calibration_and_deviations():
    """
    Калибрует ETI-предсказания через логарифмическую регрессию
    и анализирует остаточные отклонения.
    """

    print("\n" + "=" * 80)
    print("ЧАСТЬ 3: ЛОГАРИФМИЧЕСКАЯ КАЛИБРОВКА И АНАЛИЗ ОТКЛОНЕНИЙ")
    print("=" * 80)

    # Вычисляем безразмерные ETI-значения
    eti_dimensionless = []
    for i, name in enumerate(NAMES):
        s_vector = [float(M[i, j]) for j in range(6)]
        eti_dimensionless.append(compute_predicted_value(s_vector))

    ln_eti = np.log(eti_dimensionless)
    ln_codata = np.log([CODATA[name][0] for name in NAMES])

    # Линейная регрессия
    slope, intercept, r_value, p_value, std_err = sp_stats.linregress(ln_eti, ln_codata)

    print(f"\n--- Калибровочная регрессия ---")
    print(f"  ln(CODATA) = {slope:.6f} × ln(ETI) + {intercept:.6f}")
    print(f"  R² = {r_value ** 2:.8f}")
    print(f"  p = {p_value:.6e}")

    # Калиброванные значения
    ln_eti_calibrated = slope * ln_eti + intercept
    eti_calibrated = np.exp(ln_eti_calibrated)

    # Отклонения
    deviations = []
    for i, name in enumerate(NAMES):
        ratio = eti_calibrated[i] / CODATA[name][0]
        rel_dev_pct = (ratio - 1.0) * 100
        a = float(M[i, 4])
        b = abs(float(M[i, 5]))
        deviations.append({
            'name': name,
            'a': a,
            'b': b,
            'eti_dim': eti_dimensionless[i],
            'eti_cal': eti_calibrated[i],
            'codata': CODATA[name][0],
            'ratio': ratio,
            'rel_dev_pct': rel_dev_pct,
            'ln_dev': np.log(ratio) if ratio > 0 else 0
        })

    # Статистика
    rel_devs = [d['rel_dev_pct'] for d in deviations]
    abs_rel_devs = [abs(d) for d in rel_devs]

    print(f"\n--- Статистика отклонений ---")
    print(f"  Среднее: {np.mean(rel_devs):+.4f}%")
    print(f"  Медиана: {np.median(rel_devs):+.4f}%")
    print(f"  Стандартное откл.: {np.std(rel_devs):.4f}%")
    print(f"  Минимум: {np.min(rel_devs):+.4f}%")
    print(f"  Максимум: {np.max(rel_devs):+.4f}%")
    print(f"  Среднее |Δ|: {np.mean(abs_rel_devs):.4f}%")

    # Топ-5 лучших и худших
    print(f"\n✓ Топ-5 лучших согласий:")
    best = sorted(deviations, key=lambda x: abs(x['rel_dev_pct']))[:5]
    for d in best:
        print(f"  {d['name']:12s}: a={d['a']:+3.0f}, b={d['b']:.1f}, Δ = {d['rel_dev_pct']:+.4f}%")

    print(f"\n⚠ Топ-5 худших согласий:")
    worst = sorted(deviations, key=lambda x: abs(x['rel_dev_pct']), reverse=True)[:5]
    for d in worst:
        print(f"  {d['name']:12s}: a={d['a']:+3.0f}, b={d['b']:.1f}, Δ = {d['rel_dev_pct']:+.4f}%")

    # Корреляции
    a_vals = np.array([d['a'] for d in deviations])
    b_vals = np.array([d['b'] for d in deviations])
    dev_vals = np.array(rel_devs)
    abs_dev_vals = np.array(abs_rel_devs)

    r_pearson_a, p_pearson_a = sp_stats.pearsonr(a_vals, dev_vals)
    r_pearson_b, p_pearson_b = sp_stats.pearsonr(b_vals, dev_vals)
    rho_spearman_a, p_spearman_a = sp_stats.spearmanr(a_vals, dev_vals)
    rho_spearman_abs_a, p_spearman_abs_a = sp_stats.spearmanr(np.abs(a_vals), abs_dev_vals)

    print(f"\n--- Корреляционный анализ ---")
    print(f"  Пирсон (a vs Δ): r = {r_pearson_a:+.4f} (p = {p_pearson_a:.4f})")
    print(f"  Пирсон (b vs Δ): r = {r_pearson_b:+.4f} (p = {p_pearson_b:.4f})")
    print(f"  Спирмен (a vs Δ): ρ = {rho_spearman_a:+.4f} (p = {p_spearman_a:.4f})")
    print(f"  Спирмен (|a| vs |Δ|): ρ = {rho_spearman_abs_a:+.4f} (p = {p_spearman_abs_a:.4f})")

    # Групповой анализ по a
    print(f"\n--- Групповой анализ по a ---")
    a_groups = {}
    for d in deviations:
        a_int = int(d['a'])
        if a_int not in a_groups:
            a_groups[a_int] = []
        a_groups[a_int].append(abs(d['rel_dev_pct']))

    for a_int in sorted(a_groups.keys()):
        vals = a_groups[a_int]
        print(f"  a = {a_int:+3d}: n = {len(vals)}, среднее |Δ| = {np.mean(vals):.4f}%, "
              f"медиана = {np.median(vals):.4f}%, max = {np.max(vals):.4f}%")

    # Графики
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. ln(ETI) vs ln(CODATA)
    ax = axes[0, 0]
    ax.scatter(ln_eti, ln_codata, c='steelblue', s=50, alpha=0.7, edgecolors='black')
    ax.plot(ln_eti, slope * ln_eti + intercept, 'r-', linewidth=2,
            label=f'ln(CODATA) = {slope:.4f}·ln(ETI) + {intercept:.2f}')
    ax.set_xlabel('ln(ETI безразмерное)')
    ax.set_ylabel('ln(CODATA СИ)')
    ax.set_title(f'Калибровка: R² = {r_value ** 2:.8f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Отклонения vs a
    ax = axes[0, 1]
    ax.scatter(a_vals, dev_vals, c='darkorange', s=60, alpha=0.7, edgecolors='black')
    ax.axhline(y=0, color='red', linestyle='--')
    ax.set_xlabel('Показатель a (степень при ln N)')
    ax.set_ylabel('Отклонение Δ (%)')
    ax.set_title(f'Корреляция a vs Δ (r = {r_pearson_a:+.3f})')
    ax.grid(True, alpha=0.3)

    # 3. Гистограмма отклонений
    ax = axes[1, 0]
    ax.hist(dev_vals, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Отклонение Δ (%)')
    ax.set_ylabel('Число констант')
    ax.set_title(f'Распределение отклонений (σ = {np.std(dev_vals):.3f}%)')
    ax.grid(True, alpha=0.3)

    # 4. |Δ| vs |a|
    ax = axes[1, 1]
    ax.scatter(np.abs(a_vals), abs_dev_vals, c='green', s=60, alpha=0.7, edgecolors='black')
    ax.set_xlabel('|a|')
    ax.set_ylabel('|Δ| (%)')
    ax.set_title(f'Спирмен ρ = {rho_spearman_abs_a:+.3f} (p = {p_spearman_abs_a:.4f})')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('calibration_analysis.png', dpi=300)
    print("\n✓ Сохранено: calibration_analysis.png")

    return deviations


# ============================================================
# 8. NLO-ПОПРАВКИ
# ============================================================

def extract_nlo_corrections(deviations):
    """Извлекает NLO-поправки после калибровки."""

    print("\n" + "=" * 80)
    print("ЧАСТЬ 4: ИЗВЛЕЧЕНИЕ NLO-ПОПРАВОК")
    print("=" * 80)

    a_vals = np.array([d['a'] for d in deviations])
    b_vals = np.array([d['b'] for d in deviations])
    ln_devs = np.array([d['ln_dev'] for d in deviations])

    ln_N = np.log(N_val)
    ln_N2 = ln_N ** 2

    # Модель 1: (c1·a + c2·b) / ln N
    X1 = np.column_stack([a_vals / ln_N, b_vals / ln_N])
    coef1, res1, _, _ = np.linalg.lstsq(X1, ln_devs, rcond=None)
    r2_1 = 1 - np.sum(res1 ** 2) / np.sum((ln_devs - np.mean(ln_devs)) ** 2)

    # Модель 2: добавляем a·b/ln N
    X2 = np.column_stack([a_vals / ln_N, b_vals / ln_N, a_vals * b_vals / ln_N])
    coef2, res2, _, _ = np.linalg.lstsq(X2, ln_devs, rcond=None)
    r2_2 = 1 - np.sum(res2 ** 2) / np.sum((ln_devs - np.mean(ln_devs)) ** 2)

    # Модель 3: полная NLO + NNLO
    X3 = np.column_stack([
        a_vals / ln_N, b_vals / ln_N,
        a_vals ** 2 / ln_N2, b_vals ** 2 / ln_N2, a_vals * b_vals / ln_N2
    ])
    coef3, res3, _, _ = np.linalg.lstsq(X3, ln_devs, rcond=None)
    r2_3 = 1 - np.sum(res3 ** 2) / np.sum((ln_devs - np.mean(ln_devs)) ** 2)

    n = len(ln_devs)
    models = [
        (1, coef1, r2_1, n * np.log(np.sum(res1 ** 2) / n) + 4, "Δln = (c1·a + c2·b) / ln N"),
        (2, coef2, r2_2, n * np.log(np.sum(res2 ** 2) / n) + 6, "Δln = (c1·a + c2·b + c3·a·b) / ln N"),
        (3, coef3, r2_3, n * np.log(np.sum(res3 ** 2) / n) + 10, "Δln = NLO + NNLO (a², b², a·b)"),
    ]

    print(f"\n--- Сравнение NLO-моделей ---")
    print(f"  ln N = {ln_N:.2f}")
    for num, coefs, r2, aic, desc in models:
        print(f"  Модель {num}: R² = {r2:.4f}, AIC = {aic:.2f}")
        print(f"    {desc}")
        for j, c in enumerate(coefs):
            print(f"      c{j + 1} = {c:+.6f}")

    best_model = min(models, key=lambda m: m[3])
    print(f"\n✓ Оптимальная модель: {best_model[0]} (AIC = {best_model[3]:.2f})")

    # Улучшенные формулы
    print(f"\n--- Примеры улучшенных формул ---")
    for d in deviations[:5]:
        name = d['name']
        lo_val = np.log(d['eti_cal'])
        nlo_corr = 0
        if best_model[0] == 1:
            nlo_corr = (best_model[1][0] * d['a'] + best_model[1][1] * d['b']) / ln_N
        elif best_model[0] == 2:
            nlo_corr = (best_model[1][0] * d['a'] + best_model[1][1] * d['b'] +
                        best_model[1][2] * d['a'] * d['b']) / ln_N

        improved = np.exp(lo_val + nlo_corr)
        orig_dev = abs(d['ratio'] - 1) * 100
        impr_dev = abs(improved / d['codata'] - 1) * 100
        print(f"  {name}: LO Δ = {orig_dev:.4f}% → +NLO Δ = {impr_dev:.4f}%")

    # График
    if best_model[0] == 1:
        pred = X1 @ best_model[1]
    elif best_model[0] == 2:
        pred = X2 @ best_model[1]
    else:
        pred = X3 @ best_model[1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(pred * 100, [d['rel_dev_pct'] for d in deviations],
                    alpha=0.7, c='steelblue', s=50, edgecolors='black')
    axes[0].plot([-2, 2], [-2, 2], 'r--', linewidth=1)
    axes[0].set_xlabel('Предсказанное отклонение (%)')
    axes[0].set_ylabel('Наблюдаемое отклонение (%)')
    axes[0].set_title(f'Модель {best_model[0]}: R² = {best_model[2]:.4f}')
    axes[0].grid(True, alpha=0.3)

    residuals = ln_devs - pred
    axes[1].scatter(a_vals, residuals * 100, c='darkorange', s=50, alpha=0.7, edgecolors='black')
    axes[1].axhline(y=0, color='red', linestyle='--')
    axes[1].set_xlabel('Показатель a')
    axes[1].set_ylabel('Остаток (%)')
    axes[1].set_title('Остатки после NLO-коррекции')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('nlo_analysis.png', dpi=300)
    print("\n✓ Сохранено: nlo_analysis.png")

    return best_model


# ============================================================
# 9. ВИЗУАЛИЗАЦИЯ
# ============================================================

def visualize_all():
    """Создаёт все визуализации."""

    print("\n" + "=" * 80)
    print("ЧАСТЬ 5: ВИЗУАЛИЗАЦИЯ")
    print("=" * 80)

    M_float = matrix_to_float(M)

    # Тепловая карта
    plt.figure(figsize=(12, 10))
    sns.heatmap(M_float, annot=True, fmt='.1f', cmap='coolwarm',
                xticklabels=BASIS_NAMES, yticklabels=NAMES,
                cbar_kws={'label': 'Показатель степени'})
    plt.title('Матрица ETI: 30 констант × 6 базисных примитивов')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig('matrix_heatmap.png', dpi=300)
    print("✓ Сохранено: matrix_heatmap.png")

    # PCA
    pca = PCA(n_components=2)
    coords_2d = pca.fit_transform(M_float)

    sectors = {
        'Планк': ['hbar', 'h', 'c', 'lP', 'tP', 'EP', 'G', 'mP', 'TP'],
        'ЭМ': ['alpha', 'eps0', 'mu0', 'qe', 'Z0', 'Phi0', 'lambda_e', 'lambda_p'],
        'Атомные': ['Rinf', 'a0', 'kB', 'me', 'mp'],
        'Частицы': ['vH', 'muon', 'tau', 'pion_pm', 'pion0'],
        'Космо': ['Lambda', 'kappa', 'mH_mW']
    }

    color_map = {'Планк': 'blue', 'ЭМ': 'green', 'Атомные': 'purple',
                 'Частицы': 'red', 'Космо': 'orange'}

    plt.figure(figsize=(10, 8))
    for sector, consts in sectors.items():
        mask = [i for i, n in enumerate(NAMES) if n in consts]
        if mask:
            plt.scatter(coords_2d[mask, 0], coords_2d[mask, 1],
                        c=color_map[sector], label=sector, s=80, alpha=0.7, edgecolors='black')

    for i, (x, y, name) in enumerate(zip(coords_2d[:, 0], coords_2d[:, 1], NAMES)):
        if name in ['hbar', 'c', 'G', 'me', 'mp', 'alpha', 'vH', 'Lambda']:
            plt.annotate(name, (x, y), fontsize=8, ha='right')

    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)')
    plt.title('PCA-проекция констант в 6D ETI-базисе')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('pca_projection.png', dpi=300)
    print("✓ Сохранено: pca_projection.png")


# ============================================================
# 10. ГЛАВНАЯ ФУНКЦИЯ
# ============================================================

def main():
    """Главная функция."""

    print("=" * 80)
    print("ETI CONSTANTS MATRIX COMPLETE ANALYSIS")
    print("=" * 80)
    print(f"Время запуска: {datetime.now().isoformat()}")
    print(f"Параметры: K = {K}, N = {N_val:.6e}")
    print(f"ln N = {np.log(N_val):.6f}")

    # Часть 1: Ядро
    kernel_vectors, short_vectors = analyze_kernel()

    # Часть 2: χ²
    identity_results = chi_squared_test(kernel_vectors)

    # Часть 3: Калибровка и отклонения
    deviations = logarithmic_calibration_and_deviations()

    # Часть 4: NLO
    best_nlo = extract_nlo_corrections(deviations)

    # Часть 5: Визуализация
    visualize_all()

    # Проверка соответствия
    if M.rows != len(NAMES):
        print(f"❌ ОШИБКА: M.rows = {M.rows}, len(NAMES) = {len(NAMES)}")
        print("Несоответствие размеров! Проверьте матрицу и список имён.")
        return

    # Сохранение
    summary = {
        'timestamp': datetime.now().isoformat(),
        'rank': int(M.rank()),
        'num_identities': len(kernel_vectors),
        'num_short_invariants': len(short_vectors),
        'best_nlo_model': best_nlo[0],
        'best_nlo_r2': float(best_nlo[2]),
        'parameters': {'K': K, 'N': N_val, 'ln_N': np.log(N_val)}
    }

    with open('eti_analysis_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n" + "=" * 80)
    print("АНАЛИЗ ЗАВЕРШЁН")
    print("=" * 80)
    print(f"✓ Результаты: eti_analysis_summary.json")
    print(f"✓ Графики: z_scores_histogram.png, calibration_analysis.png")
    print(f"✓ Графики: nlo_analysis.png, matrix_heatmap.png, pca_projection.png")
    print()
    print("🎯 Ключевые выводы:")
    print(f"  • Ранг: {M.rank()}/6")
    print(f"  • Тождеств: {len(kernel_vectors)}")
    print(f"  • Коротких инвариантов: {len(short_vectors)}")
    print(f"  • NLO-модель: {best_nlo[0]} (R² = {best_nlo[2]:.4f})")


if __name__ == "__main__":

    main()
