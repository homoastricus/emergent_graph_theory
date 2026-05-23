import sympy as sp
import numpy as np


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
    [5, 1, 2, 0, 5, -1],  # Omega0_c (Ω_c⁰)
    #
    # lambda_B0 = √π × lnN⁶ / (K^{1/2} × N^{1/3})
    # ((lnN)^6 * 1/π:3/2 * 4) / N^(1/3)
    # √π → s_π=0.5
    # K^{-1/2} = (√2)⁻¹(√3)⁻¹
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
    [2, 0, -2.5, 0, 6, -1],  # K_star (K*)

    #neutron_lifetime
    [0, 0, 1, 0, 1, 0],

    #h2_connection_energy
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
    "eta_shtrih",
    "rho_meson",
    "K_star_meson",
    'neutron_lifetime',
    "h2_connection_energy"
]

def get_exact_invariants(M_sp):
    """
    Извлекает ТОЛЬКО точные инварианты из ядра матрицы.
    Точный инвариант — где все 6 степеней сокращаются.
    """
    # 1. Получаем базис ядра
    kernel = M_sp.T.nullspace()

    exact_invariants = []
    approximate_invariants = []

    for vec_idx, vec in enumerate(kernel):
        # 2. Приводим к целочисленному виду
        denoms = [sp.fraction(term)[1] for term in vec]
        lcm = sp.ilcm(*denoms) if denoms else 1
        ivec = [int(sp.simplify(v * lcm)) for v in vec]

        # Убираем общий делитель
        gcd = abs(sp.igcd(*ivec)) if any(ivec) else 1
        ivec = [v // gcd for v in ivec]

        # 3. Вычисляем степени для левой и правой частей
        left_powers = np.zeros(6)  # положительные коэффициенты
        right_powers = np.zeros(6)  # отрицательные коэффициенты

        for i, coef in enumerate(ivec):
            row = [float(M_sp[i, j]) for j in range(6)]
            if coef > 0:
                left_powers += coef * np.array(row)
            elif coef < 0:
                right_powers += (-coef) * np.array(row)

        # 4. Проверяем полное сокращение
        diff = left_powers - right_powers
        is_exact = np.allclose(diff, 0, atol=1e-10)

        norm_sq = sum(v * v for v in ivec)

        if is_exact:
            exact_invariants.append({
                'vector': ivec,
                'norm_sq': norm_sq,
                'diff': diff,
                'original_idx': vec_idx
            })
        else:
            approximate_invariants.append({
                'vector': ivec,
                'norm_sq': norm_sq,
                'diff': diff,
                'original_idx': vec_idx
            })

    # 5. Сортируем по норме
    exact_invariants.sort(key=lambda x: x['norm_sq'])

    return exact_invariants, approximate_invariants


def vector_to_expression(vec, names, threshold=1e-10):
    """Преобразует вектор коэффициентов в читаемое выражение."""
    num_parts = []
    den_parts = []

    for coef, name in zip(vec, names):
        if abs(coef) < threshold:
            continue
        if coef > 0:
            if abs(coef - 1) < threshold:
                num_parts.append(name)
            else:
                num_parts.append(f"{name}^{int(coef)}")
        elif coef < 0:
            c = abs(coef)
            if abs(c - 1) < threshold:
                den_parts.append(name)
            else:
                den_parts.append(f"{name}^{int(c)}")

    num_str = " × ".join(num_parts) if num_parts else "1"
    den_str = " × ".join(den_parts) if den_parts else "1"

    if den_str == "1":
        return num_str
    return f"({num_str}) / ({den_str})"


# ИСПОЛЬЗОВАНИЕ
# M — sympy-матрица
# NAMES — список имён констант

exact, approx = get_exact_invariants(M)

print("ТОЧНЫЕ ИНВАРИАНТЫ (все степени сокращаются)")
print(f"Найдено: {len(exact)}")
print()

for i, inv in enumerate(exact[:30], 1):  # первые 30
    expr = vector_to_expression(inv['vector'], NAMES)
    print(f"#{i} (‖v‖²={inv['norm_sq']}): {expr[:120]}")

print()
print("ПРИБЛИЖЁННЫЕ ИНВАРИАНТЫ (НЕ сокращаются)")
print(f"Найдено: {len(approx)}")
print()

for i, inv in enumerate(approx[:10], 1):
    expr = vector_to_expression(inv['vector'], NAMES)
    diff = inv['diff']
    print(f"#{i} (‖v‖²={inv['norm_sq']}): {expr[:100]}")
    print(f"    Остаток: s₂={diff[0]:+.1f}, s₃={diff[1]:+.1f}, s_π={diff[2]:+.1f}, "
          f"s_lnK={diff[3]:+.1f}, s_lnN={diff[4]:+.1f}, s_N={diff[5]:+.1f}")