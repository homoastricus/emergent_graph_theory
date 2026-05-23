import numpy as np
import sympy as sp

data = [

    # s2, s3, s_pi, s_lnK, s_lnN, s_N^(1/3)

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

]
A = np.array(data, dtype=np.float64)
rank = np.linalg.matrix_rank(A)
print("Rank =", rank)
u, s, vh = np.linalg.svd(A)
print("\nSingular values:")
print(s)
threshold = 1e-10
null_mask = (s < threshold)

print("\nNear-zero singular values:", s[s < threshold])

null_space = vh.T[:, s < threshold]
print("\nNull space vectors (зависимости):")
print(null_space)

# РАНГ (по порогу)
tol = 1e-10
rank = np.sum(s > tol)

print("\nRank =", rank)

# CONDITION NUMBER
cond_number = s[0] / s[-1]
print("\nCondition number =", cond_number)

# НУЛЕВОЕ ПРОСТРАНСТВО
null_space = vh.T[:, null_mask]

print("\nNear-zero singular values:")
print(s[s < tol])

print("\nNull space vectors:")
print(null_space)