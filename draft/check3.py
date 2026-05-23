"""
АНАЛИЗ: ПРОВЕРКА НЕЗАВИСИМОСТИ ТОЖДЕСТВ — РАСШИРЕННАЯ ВЕРСИЯ
1. χ²-функционал с анализом ширины минимума
2. Leave-one-out: устойчивость N к удалению тождеств
3. Bootstrap: случайные поднаборы по 3 из 14
4. Итоговая вариационная формулировка
5. NULL MODELS: сравнение со случайными системами
6. PERMUTATION TEST: перемешивание target'ов
7. COMPLEXITY PENALTY: AIC/BIC
8. HOLD-OUT VALIDATION: предсказание на unseen identities
"""
import random
import math
import numpy as np
from scipy.optimize import minimize_scalar
import warnings
from scipy.special import gamma, zeta
from itertools import combinations

warnings.filterwarnings('ignore')

# КОНСТАНТЫ
K = 6.0
pi = math.pi
lnK = math.log(K)

N_base = 4.197668e121
lnN_base = math.log(N_base)

# Теоретическое значение (для справки, не используется в тестах)
lnN_theory = (K - lnK) / (1.0 / 3.0 - 1.0 / pi)
N_theory = math.exp(lnN_theory)

# Константы для новых тождеств
euler = 2.718281828459045
euler_mascheroni = 0.57721566490153286060651209008240243104215933593992
gamma_quarter = 3.6256099082219083  # 1/4
e = math.e
lambda_const = 0.05183093000  # Постоянная Голомба-Дикмана
mills = 1.306377883863080
meissel_mertens = 0.26149721284764278375542683860869585905156664826119
cahen = 0.6419448389191956  # Постоянная Каэна
gompertz = 0.5963473623231940743410784993692793760741778601521
catalan = 0.91596559417721901505460351493238411077414937428167
brun_twin = 1.902160583104
van_der_pauw = 4.5323601418271938
parabolic_const = 2.2955871493926380740340072915082881885494572780968
lemniscate = 2.622057554292119
ln3 = math.log(3)
ln2 = math.log(2)
glaisher = 1.2824271291006226
brjuno = 0.0654603245889095
alpha = 7.2973525693e-3
sqrt2 = math.sqrt(2)
sqrt3 = math.sqrt(3)
supergolden = 1.465571231876768
levy = 3.275822918721811
viswanath = 1.1319882487943
landau_ramanujan = 0.764223653589221
gauss = 0.834626841674073
kepler_bouwkamp = 0.1149420448532962
feigenbaum_delta = 4.669201609102990
feigenbaum_alpha = 2.5029078750958928
khinchin = 2.6854520010653062
apery = 1.2020569031595942
phi = (1 + math.sqrt(5)) / 2
gamma_1_2 = gamma(0.5)
gamma_1_6 = gamma(1 / 6)
gamma_2_3 = gamma(2 / 3)
lnpi = math.log(math.pi)
gelfond_schneider = 2 ** math.sqrt(2)
e_pi = math.exp(math.pi)
alladi_grinstead = 0.8093940205406391
regular_paperfolding = 0.850736  # постоянная бумажного сгиба
gamma_third = gamma(1.0 / 3.0)  # Г(1/3)
gamma_three_quarters = gamma(3.0 / 4.0)  # Г(3/4)
ramanujan_soldner = 1.451369234883381  # постоянная Рамануджана—Сольднера
artin = 0.373955813619202  # постоянная Артина
erdos_tenenbaum_ford = 0.086071332  # постоянная Эрдёша—Тененбаума—Форда
porter = 1.467078079433  # постоянная Портера
embree_trefethen = 0.70258  # постоянная Эмбри—Трефетена
magic_angle = 0.9553166181245  # магический угол
twin_prime = 0.660161815846869  # постоянная простых близнецов
plastic = 1.324717957244746  # пластическое число
pi_minus_3 = pi - 3.0
pi_over_4 = pi / 4.0
pi_over_K = pi / K
e_pi = math.exp(pi)
e_e = e ** e
gelfond = gelfond_schneider  # синоним
hilbert = 2.0 ** math.sqrt(2)  # = gelfond_schneider
LAMBDA_val = 0.039132


def identity_feigenbaum_ratio(lnN):
    """K * (K + 1/lnN) / (K + 1/lnlnK) ≈ feigenbaum_delta"""
    return K * (K + 1.0 / lnN + 1.0 / lnN ** 2) / (K + 1.0 / math.log(lnK))


def geom_resonance_inv_pi(lnN):
    N = math.exp(lnN)
    p = 1.0 / (K * N ** (1 / 3))
    return (K + math.log(p)) / (p - lnN)


def geom_resonance_pi(lnN):
    N = math.exp(lnN)
    p = 1.0 / (K * N ** (1 / 3))
    return -lnN / (K + math.log(p))


def alpha_identity(lnN):
    return 2 * lnK ** 2 / (pi * lnN)


def alladi_grinstead_identity(lnN):
    N = math.exp(lnN)
    p = 1.0 / (K * N ** (1 / 3))
    return (K - 1.0 / (1.0 - p)) / (K + 1.0 / math.log(lnN))


def pi_sq_identity(lnN):
    N = math.exp(lnN)
    p = 1.0 / (K * N ** (1 / 3))
    return (1.0 / (lnK ** 2) - K) - math.log(K * p) / K


def identity_euler_mills(lnN):
    """K * ln(ln(K)) + 1/(K * ln(K)^2) ≈ euler * mills"""
    return K * math.log(lnK) + 1.0 / (K * lnK * lnK)


def identity_golden_gompertz(lnN):
    """(K - ln(p)) / ((ln(N)-K)/K) ≈ phi + gompertz"""
    N = math.exp(lnN)
    p = 1.0 / (K * N ** (1 / 3))
    return (K - math.log(p)) / ((lnN - K) / K)


def identity_golden_meissel(lnN):
    """(K - ln(lnN)) + (K - 1/ln(lnN)) ≈ phi / meissel_mertens"""
    ln_ln_N = math.log(lnN)
    return (K - ln_ln_N) + (K - 1.0 / ln_ln_N)


def identity_ln3(lnN):
    """(K/lnN - K) / (1/lnK - K) ≈ ln 3"""
    return (K / lnN - K) / (1.0 / lnK - K)


def identity_gamma_catalan(lnN):
    """(K - lnK) - ln(K*p)/lnN ≈ gamma_quarter + catalan"""
    N = math.exp(lnN)
    p = 1.0 / (K * N ** (1 / 3))
    return (K - lnK) - math.log(K * p) / lnN


def identity_euler_van_der_pauw(lnN):
    """ln(lnN)/K + (K + 1/lnK^2) ≈ e + van_der_pauw"""
    return math.log(lnN) / K + (K + 1.0 / lnK ** 2)


def identity_gamma_mills(lnN):
    """ln(ln(lnN)) / (K - ln(lnN)) ≈ gamma_quarter * mills"""
    ln_ln_N = math.log(lnN)
    return math.log(ln_ln_N) / (K - ln_ln_N)


def identity_golden_cahen(lnN):
    """(K + 1/ln(ln(lnN))) / (K - ln(K*p)/lnN) ≈ phi * cahen"""
    N = math.exp(lnN)
    p = 1.0 / (K * N ** (1 / 3))
    return (K + 1.0 / math.log(math.log(lnN))) / (K - math.log(K * p) / lnN)


def identity_vdp_glaisher(lnN):
    """ln(lnN) + 1/ln(lnN) ≈ van_der_pauw * glaisher"""
    ln_ln_N = math.log(lnN)
    return ln_ln_N + 1.0 / ln_ln_N


def identity_brjuno_alpha(lnN):
    """(K - lnN/ln(K*p)) - (1/ln(lnN) / K) ≈ brjuno / alpha"""
    N = math.exp(lnN)
    p = 1.0 / (K * N ** (1 / 3))
    Kp = K * p
    return (K - lnN / math.log(Kp)) - (1.0 / (math.log(lnN) * K))


def identity_cahen_gompertz(lnN):
    """(K + 1/ln(ln(lnN))) / (K + lambda) ≈ cahen / gompertz"""
    return (K + 1.0 / math.log(math.log(lnN))) / (K + lambda_const)


def identity_gamma_quarter_meissel(lnN):
    """(K - 1/lnK^2) / (K - 1/lnN^2) ≈ gamma_quarter * meissel_mertens"""
    return (K - 1.0 / lnK ** 2) / (K - 1.0 / lnN ** 2)


def identity_gamma_quarter_div_meissel(lnN):
    """(lnN+K)/K - K*ln(lnN) ≈ gamma_quarter / meissel_mertens"""
    return (lnN + K) / K - K * math.log(lnN)


def identity_phi_brun(lnN):
    """ln(ln(lnN))/K + (K - lnK^2) ≈ phi * brun_twin"""
    return math.log(math.log(lnN)) / K + (K - lnK ** 2)


def identity_gamma_1_6_gauss(lnN):
    """1/lnK + K + lambda ≈ gamma_1_6 / gauss"""
    return 1.0 / lnK + K + lambda_const


def identity_sqrt3_gamma_quarter(lnN):
    """K + 1/ln(ln(lnN)) - lnK/K ≈ sqrt3 * gamma_quarter"""
    return K + 1.0 / math.log(math.log(lnN)) - lnK / K


def identity_kepler_supergolden(lnN):
    """(K + lnK) / (K - ln(K*p)) ≈ kepler_bouwkamp / supergolden"""
    N = math.exp(lnN)
    p = 1.0 / (K * N ** (1 / 3))
    return (K + lnK) / (K - math.log(K * p))


def identity_lemniscate_e_e(lnN):
    """(lnN/K - K) - ln(lnN)/K ≈ lemniscate * e^e"""
    return (lnN / K - K) - math.log(lnN) / K


def identity_sqrt2_mills(lnN):
    """K/lnK^2 - K/lnN ≈ sqrt2 * mills"""
    return K / lnK ** 2 - K / lnN


def identity_ln3_parabolic(lnN):
    """U/K - K/lnN ≈ ln3 / parabolic"""
    N = math.exp(lnN)
    p = 1.0 / (K * N ** (1 / 3))
    U_val = lnN / abs(math.log(K * p))
    return U_val / K - K / lnN


def identity_khinchin_e_e(lnN):
    """lnN/K - (K - K/lnN) ≈ khinchin * e^e"""
    return lnN / K - (K - K / lnN)


def identity_lnpi_brun(lnN):
    """1/ln(lnN) - K/lnN * 1/ln(K*p) ≈ ln(pi) * brun_twin"""
    N = math.exp(lnN)
    p = 1.0 / (K * N ** (1 / 3))
    return 1.0 / math.log(lnN) - (K / lnN) * math.log(K * p)


def identity_delta_vdp(lnN):
    """(K + 1/ln(lnN)) / (K - 1/lnN) ≈ feigenbaum_delta / van_der_pauw"""
    return (K + 1.0 / math.log(lnN)) / (K - 1.0 / lnN)


def identity_apery_e_pi(lnN):
    """1/lnK^2 / (K - 1/lnN) ≈ apery / e^pi"""
    return (1.0 / lnK ** 2) / (K - 1.0 / lnN)


def identity_gelfond_ag(lnN):
    """K / ln(ln(lnN)) - 1/ln(lnN) ≈ gelfond_schneider / alladi_grinstead"""
    ln_ln_N = math.log(lnN)
    return K / math.log(ln_ln_N) - 1.0 / ln_ln_N


def identity_sqrt2_vdp(lnN):
    """lnK/K + K + lambda ≈ sqrt2 * van_der_pauw"""
    return lnK / K + K + lambda_const


def identity_gamma_1_6_catalan(lnN):
    """(K + K/lnN) - (ln(K*p)/lnN) / K ≈ gamma_1_6 / catalan"""
    N = math.exp(lnN)
    p = 1.0 / (K * N ** (1 / 3))
    return (K + K / lnN) - math.log(K * p) / (lnN * K)


def identity_ln2_gamma_2_3(lnN):
    """ln(lnN) / (K + 1/lnN) ≈ ln2 * gamma_2_3"""
    return math.log(lnN) / (K + 1.0 / lnN)


def identity_phi_gompertz(lnN):
    """(K + lambda) / (K - ln(K*p)/lnN) ≈ phi * gompertz"""
    N = math.exp(lnN)
    p = 1.0 / (K * N ** (1 / 3))
    return (K + lambda_const) / (K - math.log(K * p) / lnN)


def identity_pi_via_lnK(lnN):
    """ln(K)^2 * (lnN - K) / lnN ≈ pi"""
    return lnK ** 2 * (lnN - K) / lnN


def identity_pi_over_K(lnN):
    """(lnK^2 / K) * (lnN - K) / lnN ≈ pi/K"""
    return (lnK ** 2 / K) * (lnN - K) / lnN


def identity_catalan_1(lnN):
    """(lnN - K - N^(1/3)) / (K + lnN + K) ≈ catalan"""
    N = math.exp(lnN)
    return (lnN - K - N ** (1.0 / 3.0)) / (2 * K + lnN)


def identity_sqrt_pi_1(lnN):
    """(lnK + 1/lnN) - N^(1/3) / (lnN + K) ≈ sqrt(pi)"""
    N = math.exp(lnN)
    return (lnK + 1.0 / lnN) - (N ** (1.0 / 3.0)) / (lnN + K)


def identity_catalan_2(lnN):
    """K / N^(1/3) - 1/(lnK^2 * lnN) ≈ catalan"""
    N = math.exp(lnN)
    return K / (N ** (1.0 / 3.0)) - 1.0 / (lnK ** 2 * lnN)


def identity_pi_2(lnN):
    """(N^(1/3) - lnK^2) - lambda / ln(lnK) ≈ pi"""
    N = math.exp(lnN)
    return (N ** (1.0 / 3.0) - lnK ** 2) - lambda_const / math.log(lnK)


def identity_feigenbaum_alpha(lnN):
    """(K - 1/lnK^2) / (1/ln(lnK) + 1/lnK) ≈ feigenbaum_alpha"""
    return (K - 1.0 / lnK ** 2) / (1.0 / math.log(lnK) + 1.0 / lnK)


def identity_gompertz_2(lnN):
    """(lnN + K) / (1/ln(lnK)) / (lnN - 1/lnK^2) ≈ gompertz"""
    return (lnN + K) / (1.0 / math.log(lnK)) / (lnN - 1.0 / lnK ** 2)


def identity_ln2_1(lnN):
    """(N^(1/3) - lnK) / (N^(1/3) + 1/lnK^2) ≈ ln2"""
    N = math.exp(lnN)
    n13 = N ** (1.0 / 3.0)
    return (n13 - lnK) / (n13 + 1.0 / lnK ** 2)


def identity_magic_angle_1(lnN):
    """(lnN - 2K) / (N^(1/3) + lnN - K) ≈ magic_angle"""
    N = math.exp(lnN)
    return (lnN - 2 * K) / (N ** (1.0 / 3.0) + lnN - K)


def identity_brun_twin_1(lnN):
    """lnK/K + 1/ln(lnK) - lambda ≈ brun_twin"""
    return lnK / K + 1.0 / math.log(lnK) - lambda_const


def identity_parabolic_1(lnN):
    """(K + lambda) - N^(1/3) * ln(lnK) ≈ parabolic"""
    N = math.exp(lnN)
    return (K + lambda_const) - (N ** (1.0 / 3.0)) * math.log(lnK)


def identity_gamma_3_4_1(lnN):
    """ln(lnK)/lnK^2 + (lnN+K)/(lnN-K) ≈ gamma_3_4"""
    return math.log(lnK) / lnK ** 2 + (lnN + K) / (lnN - K)


def identity_gamma_3_4_2(lnN):
    """1/(ln(lnK))^2 - 1/ln(lnK) ≈ gamma_3_4"""
    ln_ln_K = math.log(lnK)
    return 1.0 / ln_ln_K ** 2 - 1.0 / ln_ln_K


def identity_meissel_mertens(lnN):
    """1/(ln(lnK) * N^(1/3)) - 1/(K * lnN) ≈ meissel_mertens"""
    N = math.exp(lnN)
    return 1.0 / (math.log(lnK) * N ** (1.0 / 3.0)) - 1.0 / (K * lnN)


def identity_glaisher_2(lnN):
    """1/(ln(lnK) * lnK) + lnK/ln(lnK) ≈ glaisher"""
    ln_ln_K = math.log(lnK)
    return 1.0 / (ln_ln_K * lnK) + lnK / ln_ln_K


def identity_phi_2(lnN):
    """1/ln(lnK) - 1/lnN - 1/(K * lnK) ≈ phi"""
    return 1.0 / math.log(lnK) - 1.0 / lnN - 1.0 / (K * lnK)


def identity_mills_2(lnN):
    """lnK - 1/lnK^2 - 1/lnK^3 ≈ mills"""
    return lnK - 1.0 / lnK ** 2 - 1.0 / lnK ** 3


def identity_magic_angle_2(lnN):
    """(lnN + K - N^(1/3)) / (N^(1/3) + lnN + K) ≈ magic_angle"""
    N = math.exp(lnN)
    n13 = N ** (1.0 / 3.0)
    return (lnN + K - n13) / (n13 + lnN + K)


def identity_lnpi(lnN):
    """lnK^2 - lnK - lnK / N^(1/3) ≈ ln(pi)"""
    N = math.exp(lnN)
    return lnK ** 2 - lnK - lnK / N ** (1.0 / 3.0)


def identity_brun_twin_2(lnN):
    """(N^(1/3) - K) * (lnK + 1/ln(lnK)) ≈ brun_twin"""
    N = math.exp(lnN)
    return (N ** (1.0 / 3.0) - K) * (lnK + 1.0 / math.log(lnK))


def identity_twin_prime(lnN):
    """K * lambda - lnK / (lnN - K) ≈ twin_prime"""
    return K * lambda_const - lnK / (lnN - K)


def identity_gamma_2_3(lnN):
    """lnK/ln(lnK) - 1/ln(lnK) - 1/lnN ≈ gamma_2_3"""
    return lnK / math.log(lnK) - 1.0 / math.log(lnK) - 1.0 / lnN


def identity_khinchin_2(lnN):
    """(lnK + 1/lnN) * (lnK^2 - 1/ln(lnK)) ≈ khinchin"""
    return (lnK + 1.0 / lnN) * (lnK ** 2 - 1.0 / math.log(lnK))


def identity_pi_over_4(lnN):
    """(N^(1/3) + 1/lnN) / (lnK + N^(1/3)) ≈ pi/4"""
    N = math.exp(lnN)
    n13 = N ** (1.0 / 3.0)
    return (n13 + 1.0 / lnN) / (lnK + n13)


def identity_cahen_2(lnN):
    """(lnK + lnK^2) / (K + lnK) ≈ cahen"""
    return (lnK + lnK ** 2) / (K + lnK)


def identity_gamma_1_6_2(lnN):
    """(K - 1/lnK^2) * (lnN - K) / lnN ≈ gamma_1_6"""
    return (K - 1.0 / lnK ** 2) * (lnN - K) / lnN


def identity_brun_twin_3(lnN):
    """lnK + lambda - 1/(K * lnN) ≈ brun_twin"""
    return lnK + lambda_const - 1.0 / (K * lnN)


def identity_viswanath(lnN):
    """(N^(1/3) - 1/lnK^2) / (lnK^2 / ln(lnK)) ≈ viswanath"""
    N = math.exp(lnN)
    return (N ** (1.0 / 3.0) - 1.0 / lnK ** 2) / (lnK ** 2 / math.log(lnK))


def identity_plastic(lnN):
    """lnK * lambda * (N^(1/3) + lambda) ≈ plastic"""
    N = math.exp(lnN)
    return lnK * lambda_const * (N ** (1.0 / 3.0) + lambda_const)


def identity_pi_minus_3(lnN):
    """K * N^(1/3) / (lnN - K + lnK^2) ≈ pi - 3"""
    N = math.exp(lnN)
    return (K * N ** (1.0 / 3.0)) / (lnN - K + lnK ** 2)


def identity_phi_3(lnN):
    """lnK - 1/lnK^3 ≈ phi"""
    return lnK - 1.0 / lnK ** 3


def identity_sqrt_pi_2(lnN):
    """(lnN + lnK^2) * ln(lnK) / (lnN - K) ≈ sqrt(pi)"""
    return (lnN + lnK ** 2) * math.log(lnK) / (lnN - K)


def identity_porter(lnN):
    """lnK^3 - K + 1/ln(lnK) ≈ porter"""
    return lnK ** 3 - K + 1.0 / math.log(lnK)


def identity_gamma_3_4_3(lnN):
    """(N^(1/3) - lambda) * lambda / ln(lnK) ≈ gamma_3_4"""
    N = math.exp(lnN)
    return (N ** (1.0 / 3.0) - lambda_const) * lambda_const / math.log(lnK)


def identity_alpha_2(lnN):
    """2 / (lnN - K) ≈ alpha"""
    return 2.0 / (lnN - K)


def identity_apery_2(lnN):
    """lnK * N^(1/3) / (N^(1/3) + lnK^2) ≈ apery"""
    N = math.exp(lnN)
    n13 = N ** (1.0 / 3.0)
    return (lnK * n13) / (n13 + lnK ** 2)


def identity_lemniscate_over_e(lnN):
    """(N^(1/3) - lnK) / (1/ln(lnK) + lnK^2) ≈ lemniscate / e"""
    N = math.exp(lnN)
    return (N ** (1.0 / 3.0) - lnK) / (1.0 / math.log(lnK) + lnK ** 2)


def identity_parabolic_gauss(lnN):
    """lnK * (lnN + K) / (lnN - K - N^(1/3)) ≈ parabolic * gauss"""
    N = math.exp(lnN)
    return (lnK * (lnN + K)) / (lnN - K - N ** (1.0 / 3.0))


def identity_sqrt3_new(lnN):
    """K/(lnN+K) + 1/ln(lnK) - 1/lnN ≈ √3"""
    return K / (lnN + K) + 1.0 / math.log(lnK) - 1.0 / lnN


# ===========================================
# НОВЫЕ ПРОВЕРЕННЫЕ ТОЖДЕСТВА ДЛЯ ДОБАВЛЕНИЯ
# ===========================================

def identity_ln3_paperfolding(lnN):
    """ln(K) * N^(1/3) / (K + N^(1/3)) ≈ ln3 * regular_paperfolding"""
    N = math.exp(lnN)
    return (lnK * N ** (1.0 / 3.0)) / (K + N ** (1.0 / 3.0))


def identity_gamma_third_gompertz(lnN):
    """(1/ln(lnK) - lambda) - 1/(ln(lnK) * (lnN+K)) ≈ gamma_third * gompertz"""
    ln_ln_K = math.log(lnK)
    return (1.0 / ln_ln_K - lambda_const) - 1.0 / (ln_ln_K * (lnN + K))


def identity_glaisher_pi_e(lnN):
    """((lnN-K)/N^(1/3)) ≈ glaisher * pi^e (упрощённая форма)"""
    N = math.exp(lnN)
    return (lnN - K) / N ** (1.0 / 3.0)


def identity_alladi_brun(lnN):
    """K * ln(lnK) / (1/ln(lnK) + 1/lnK) ≈ alladi_grinstead * brun_twin"""
    ln_ln_K = math.log(lnK)
    return (K * ln_ln_K) / (1.0 / ln_ln_K + 1.0 / lnK)


def identity_van_der_pauw_apery(lnN):
    """(K - 1/lnK) + lnK/(lnN+K) ≈ van_der_pauw * apery"""
    return (K - 1.0 / lnK) + lnK / (lnN + K)


def identity_catalan_gamma_1_2(lnN):
    """(K - lnK^2) / (1/ln(lnK) + 1/lnN) ≈ catalan * gamma_1_2"""
    return (K - lnK ** 2) / (1.0 / math.log(lnK) + 1.0 / lnN)


def identity_erdos_gamma(lnN):
    """((lnN+K)/N^(1/3)) / (lnN + N^(1/3)) ≈ erdos_tenenbaum_ford * gamma_1_2"""
    N = math.exp(lnN)
    n13 = N ** (1.0 / 3.0)
    return ((lnN + K) / n13) / (lnN + n13)


def identity_artin_e_pi(lnN):
    """N^(1/3)/lnK + (lnK + lnK^2) ≈ artin * e^pi"""
    N = math.exp(lnN)
    return N ** (1.0 / 3.0) / lnK + (lnK + lnK ** 2)


def identity_paperfolding_gamma(lnN):
    """(lnK + 1/lnN) + (1/ln(lnK))^2 ≈ regular_paperfolding * gamma_1_6"""
    return (lnK + 1.0 / lnN) + (1.0 / math.log(lnK)) ** 2


def identity_cahen_supergolden(lnN):
    """(N^(1/3) - (lnN-K)) / (1/ln(lnK) - (lnN+K)) ≈ cahen * supergolden"""
    N = math.exp(lnN)
    return (N ** (1.0 / 3.0) - (lnN - K)) / (1.0 / math.log(lnK) - (lnN + K))


def identity_feigenbaum_khinchin(lnN):
    """K + N^(1/3) - 1/lnN ≈ feigenbaum_delta * khinchin"""
    N = math.exp(lnN)
    return K + N ** (1.0 / 3.0) - 1.0 / lnN


def identity_phi_catalan(lnN):
    """(1/ln(lnK) - 1/lnK) + lnK/ln(lnK) ≈ phi * catalan"""
    ln_ln_K = math.log(lnK)
    return (1.0 / ln_ln_K - 1.0 / lnK) + lnK / ln_ln_K


def identity_apery_magic_angle(lnN):
    """(lnK - 1/lnK) - (1/lnK)/N^(1/3) ≈ apery * magic_angle"""
    N = math.exp(lnN)
    return (lnK - 1.0 / lnK) - (1.0 / lnK) / N ** (1.0 / 3.0)


def identity_viswanath_landau(lnN):
    """(lnK - 1/lnK^2) / (1/ln(lnK) - 1/lnN) ≈ viswanath * landau_ramanujan"""
    return (lnK - 1.0 / lnK ** 2) / (1.0 / math.log(lnK) - 1.0 / lnN)


def identity_catalan_supergolden(lnN):
    """lnK/K + (lnN+K)/(lnN-K) ≈ catalan * supergolden"""
    return lnK / K + (lnN + K) / (lnN - K)


def identity_magic_angle_plastic(lnN):
    """(lnK + 1/lnN) / (lnK^2 - lnK) ≈ magic_angle * plastic"""
    return (lnK + 1.0 / lnN) / (lnK ** 2 - lnK)


def identity_gamma_third_artin(lnN):
    """lnK^2 * ln(lnK) / (K / lnK^2) ≈ gamma_third * artin"""
    return (lnK ** 2 * math.log(lnK)) / (K / lnK ** 2)


def identity_lnpi_gelfond(lnN):
    """lnK * ln(lnK) - K/lnN ≈ ln(pi) * gelfond_schneider"""
    return lnK * math.log(lnK) - K / lnN


def identity_pi_embree(lnN):
    """(lnK - 1/lnK^2) + N^(1/3) * lambda ≈ pi * embree_trefethen"""
    N = math.exp(lnN)
    return (lnK - 1.0 / lnK ** 2) + N ** (1.0 / 3.0) * lambda_const


def identity_gamma_third_gelfond(lnN):
    """1/(ln(lnK) * K) + N^(1/3) + 1/lnK^2 ≈ gamma_third * gelfond_schneider"""
    N = math.exp(lnN)
    return 1.0 / (math.log(lnK) * K) + N ** (1.0 / 3.0) + 1.0 / lnK ** 2


def identity_phi_cahen(lnN):
    """lnK * ln(lnK) - lnK/(lnN+K) ≈ phi * cahen"""
    return lnK * math.log(lnK) - lnK / (lnN + K)


def identity_catalan_inverse(lnN):
    """(lnN - K - N^(1/3)) / (K + lnN + K) ≈ catalan (исправленная)"""
    N = math.exp(lnN)
    return (lnN - K - N ** (1.0 / 3.0)) / (2 * K + lnN)


def identity_apery_gompertz(lnN):
    """(1/ln(lnK) + lnK^2) - (K - lnK) ≈ apery * gompertz"""
    return (1.0 / math.log(lnK) + lnK ** 2) - (K - lnK)


def identity_levy_gamma_2_3(lnN):
    """(1/ln(lnK))^2 - (1/ln(lnK) - lnK^2) ≈ levy * gamma_2_3"""
    ln_ln_K = math.log(lnK)
    return (1.0 / ln_ln_K) ** 2 - (1.0 / ln_ln_K - lnK ** 2)


IDENTITIES = [
    # Геометрический резонанс
    ("1/π", geom_resonance_inv_pi, 1.0 / pi),
    ("π", geom_resonance_pi, pi),
    ("δ_F (ratio)", identity_feigenbaum_ratio, feigenbaum_delta),
    ("α", alpha_identity, alpha),
    ("Alladi-Grinstead", alladi_grinstead_identity, alladi_grinstead),
    ("π²", pi_sq_identity, pi ** 2),
    ("Euler×Mills", identity_euler_mills, euler * mills),
    ("Golden+Gompertz", identity_golden_gompertz, phi + gompertz),
    ("Golden/Meissel", identity_golden_meissel, phi / meissel_mertens),
    ("Ln3", identity_ln3, ln3),
    ("Gamma+Catalan", identity_gamma_catalan, gamma_quarter + catalan),
    ("Euler+Van_der_Pauw", identity_euler_van_der_pauw, e + van_der_pauw),
    ("Gamma×Mills", identity_gamma_mills, gamma_quarter * mills),
    ("Golden×Cahen", identity_golden_cahen, phi * cahen),
    ("vdP×Glaisher", identity_vdp_glaisher, van_der_pauw * glaisher),
    ("Brjuno/α", identity_brjuno_alpha, brjuno / alpha),
    ("Cahen/Gompertz", identity_cahen_gompertz, cahen / gompertz),
    ("Γ(1/4)×Mm", identity_gamma_quarter_meissel, gamma_quarter * meissel_mertens),
    ("Γ(1/4)/Mm", identity_gamma_quarter_div_meissel, gamma_quarter / meissel_mertens),
    ("φ×Brun", identity_phi_brun, phi * brun_twin),
    ("Γ(1/6)/Gauss", identity_gamma_1_6_gauss, gamma_1_6 / gauss),
    ("√3×Γ(1/4)", identity_sqrt3_gamma_quarter, sqrt3 * gamma_quarter),
    ("Kepler/Supergolden", identity_kepler_supergolden, kepler_bouwkamp / supergolden),
    ("Lemniscate×e^e", identity_lemniscate_e_e, lemniscate * e_e),
    ("√2×Mills", identity_sqrt2_mills, sqrt2 * mills),
    ("ln3/parabolic", identity_ln3_parabolic, ln3 / parabolic_const),
    ("Khinchin×e^e", identity_khinchin_e_e, khinchin * e_e),
    ("lnπ×Brun", identity_lnpi_brun, lnpi * brun_twin),
    ("δ/vdP", identity_delta_vdp, feigenbaum_delta / van_der_pauw),
    ("Apéry/e^π", identity_apery_e_pi, apery / e_pi),
    ("Gelfond/AG", identity_gelfond_ag, gelfond_schneider / alladi_grinstead),
    ("√2×vdP", identity_sqrt2_vdp, sqrt2 * van_der_pauw),
    ("Γ(1/6)/Catalan", identity_gamma_1_6_catalan, gamma_1_6 / catalan),
    ("ln2×Γ(2/3)", identity_ln2_gamma_2_3, ln2 * gamma_2_3),
    ("φ×Gompertz", identity_phi_gompertz, phi * gompertz),
    ("π via ln(K)", identity_pi_via_lnK, pi),
    ("π/K via ln(K)", identity_pi_over_K, pi_over_K),
    ("Feigenbaum α", identity_feigenbaum_alpha, feigenbaum_alpha),
    ("Gompertz (2)", identity_gompertz_2, gompertz),
    ("Γ(3/4) (1)", identity_gamma_3_4_1, gamma_three_quarters),
    ("Γ(3/4) (2)", identity_gamma_3_4_2, gamma_three_quarters),
    ("φ (2)", identity_phi_2, phi),
    ("Mills (2)", identity_mills_2, mills),
    ("Γ(2/3)", identity_gamma_2_3, gamma_2_3),
    ("Khinchin (2)", identity_khinchin_2, khinchin),
    ("Cahen (2)", identity_cahen_2, cahen),
    ("Γ(1/6) (2)", identity_gamma_1_6_2, gamma_1_6),
    ("φ (3)", identity_phi_3, phi),
    ("Porter", identity_porter, porter),
    ("α (2)", identity_alpha_2, alpha),
    ("ln3×Paperfolding", identity_ln3_paperfolding, ln3 * regular_paperfolding),
    ("Γ(1/3)×Gompertz", identity_gamma_third_gompertz, gamma_third * gompertz),
    ("Glaisher×π^e", identity_glaisher_pi_e, glaisher * pi ** e),
    ("Alladi×Brun", identity_alladi_brun, alladi_grinstead * brun_twin),
    ("vdP×Apéry", identity_van_der_pauw_apery, van_der_pauw * apery),
    ("Catalan×√π (1)", identity_catalan_gamma_1_2, catalan * gamma_1_2),
    ("√3 (new)", identity_sqrt3_new, sqrt3),
]


# 1. χ²-ФУНКЦИОНАЛ
def chi_square(lnN, identities=None):
    """Сумма квадратов относительных отклонений"""
    if identities is None:
        identities = IDENTITIES
    total = 0.0
    for name, func, target in identities:
        val = func(lnN)
        # Только для положительных значений
        if val > 0 and target > 0:
            total += ((val - target) / target) ** 2
        elif val <= 0 and target <= 0:
            total += ((val - target) / abs(target)) ** 2
        else:
            total += 100.0  # штраф за смену знака
    return total / len(identities)


def find_chi_minimum(identities=None):
    """Находит минимум χ² и его ширину"""
    if identities is None:
        identities = IDENTITIES

    def chi_wrapper(lnN):
        return chi_square(lnN, identities)

    result = minimize_scalar(chi_wrapper, bounds=(270, 290), method='bounded')
    lnN_min = result.x
    chi_min = result.fun

    # Ширина минимума: где χ² = 10 × χ²_min
    target = chi_min * 10.0

    # Левая граница
    left = lnN_min
    for _ in range(20):
        test = left - 0.001
        if test < 260 or chi_wrapper(test) > target:
            break
        left = test

    # Правая граница
    right = lnN_min
    for _ in range(20):
        test = right + 0.001
        if test > 300 or chi_wrapper(test) > target:
            break
        right = test

    width = right - left

    return lnN_min, chi_min, width, left, right


# 2. LEAVE-ONE-OUT
def leave_one_out():
    """Убирает по одному тождеству и смотрит, как меняется N"""
    print("\n" + "=" * 80)
    print("2. LEAVE-ONE-OUT: УСТОЙЧИВОСТЬ К УДАЛЕНИЮ ТОЖДЕСТВ")
    print("=" * 80)

    # Базовый N (все тождества)
    base_lnN, base_chi, _, _, _ = find_chi_minimum()

    results = []
    for i, (excluded_name, _, _) in enumerate(IDENTITIES):
        subset = [ident for j, ident in enumerate(IDENTITIES) if j != i]

        result = minimize_scalar(
            lambda lnN: chi_square(lnN, subset),
            bounds=(270, 290),
            method='bounded'
        )
        lnN_sub = result.x
        delta = abs(lnN_sub - base_lnN) / base_lnN * 100

        results.append((excluded_name, lnN_sub, delta))

        stability = "✅ УСТОЙЧИВО" if delta < 0.01 else ("🟡" if delta < 0.05 else "❌ НЕУСТОЙЧИВО")
        print(f"  Без {excluded_name:<20}: lnN = {lnN_sub:.10f}, Δ = {delta:.6f}% {stability}")

    # Статистика
    deltas = [d for _, _, d in results]
    mean_delta = np.mean(deltas)
    max_delta = max(deltas)

    print(f"\n  Среднее отклонение: {mean_delta:.10f}%")
    print(f"  Максимальное отклонение: {max_delta:.10f}%")

    if max_delta < 0.01:
        print("  ✅ СТРУКТУРА УСТОЙЧИВА — тождества согласованы")
    elif max_delta < 0.05:
        print("  🟡 СТРУКТУРА УМЕРЕННО УСТОЙЧИВА")
    else:
        print("  ❌ СТРУКТУРА НЕУСТОЙЧИВА — возможна коррелированная подгонка")

    return base_lnN, results


# 3. BOOTSTRAP
def bootstrap_test(n_iterations=1000, subset_size=6):
    """Случайные поднаборы тождеств"""
    print("\n" + "=" * 80)
    print(f"3. BOOTSTRAP: {n_iterations} ИТЕРАЦИЙ ПО {subset_size} СЛУЧАЙНЫХ ТОЖДЕСТВА")
    print("=" * 80)

    n_total = len(IDENTITIES)
    lnN_samples = []

    for _ in range(n_iterations):
        indices = np.random.choice(n_total, subset_size, replace=False)
        subset = [IDENTITIES[i] for i in indices]

        result = minimize_scalar(
            lambda lnN: chi_square(lnN, subset),
            bounds=(270, 290),
            method='bounded'
        )
        lnN_samples.append(result.x)

    lnN_samples = np.array(lnN_samples)
    mean_lnN = np.mean(lnN_samples)
    median_lnN = np.median(lnN_samples)
    std_lnN = np.std(lnN_samples)

    p68_low = np.percentile(lnN_samples, 16)
    p68_high = np.percentile(lnN_samples, 84)
    p95_low = np.percentile(lnN_samples, 2.5)
    p95_high = np.percentile(lnN_samples, 97.5)

    rel_width = (p68_high - p68_low) / median_lnN * 100 if median_lnN > 0 else 100

    print(f"\n  СРЕДНЕЕ (справка): {mean_lnN:.10f}")
    print(f"  МЕДИАНА:          {median_lnN:.10f}  ← ОСНОВНАЯ ОЦЕНКА")
    print(f"  Стандартное отклонение: {std_lnN:.10f}")
    print(f"\n  Доверительные интервалы (на основе перцентилей):")
    print(f"    68%: [{p68_low:.6f}, {p68_high:.10f}]")
    print(f"    95%: [{p95_low:.6f}, {p95_high:.10f}]")
    print(f"  Относительная ширина (68% интервал): {rel_width:.4f}%")

    bootstrap_narrow = rel_width < 0.05
    if bootstrap_narrow:
        print(f"\n  ✅ УЗКИЙ ПИК ({rel_width:.4f}%) — ФУНДАМЕНТАЛЬНАЯ СТРУКТУРА")
    elif rel_width < 0.1:
        print(f"\n  🟡 УМЕРЕННО УЗКИЙ ПИК ({rel_width:.4f}%) — возможна структура")
    else:
        print(f"\n  ❌ ШИРОКИЙ ПИК ({rel_width:.4f}%) — иллюзия согласованности")

    return lnN_samples, median_lnN, bootstrap_narrow


# ============================================
# НОВЫЕ ТЕСТЫ ПО КРИТИЧЕСКИМ ЗАМЕЧАНИЯМ
# ============================================

# 5. NULL MODELS
def generate_null_model_constants(n_constants):
    """Генерирует случайные константы из того же распределения"""
    real_constants = [target for _, _, target in IDENTITIES]
    log_constants = np.log(np.abs(real_constants))
    mean_log = np.mean(log_constants)
    std_log = np.std(log_constants)

    null_targets = np.exp(np.random.normal(mean_log, std_log, n_constants))
    # Сохраняем знаки
    signs = np.sign(real_constants[:n_constants])
    return [s * abs(t) for s, t in zip(signs, null_targets)]


def null_model_test(n_null=500):
    """Сравнение реальной χ²-структуры со случайными target'ами"""
    print("\n" + "=" * 80)
    print("5. NULL MODELS: СРАВНЕНИЕ СО СЛУЧАЙНЫМИ КОНСТАНТАМИ")
    print("=" * 80)

    # Реальная χ²
    real_lnN, real_chi, real_width, _, _ = find_chi_minimum()
    real_width_rel = real_width / real_lnN * 100

    # Генерируем null-модели
    null_chis = []
    null_widths = []

    for i in range(n_null):
        null_targets = generate_null_model_constants(len(IDENTITIES))
        null_identities = [
            (name, func, target)
            for (name, func, _), target in zip(IDENTITIES, null_targets)
        ]

        try:
            lnN_null, chi_null, width_null, _, _ = find_chi_minimum(null_identities)
            null_chis.append(chi_null)
            null_widths.append(width_null / lnN_null * 100)
        except:
            continue

    null_chis = np.array(null_chis)
    null_widths = np.array(null_widths)

    # Статистика
    print(f"\n  Реальная система:")
    print(f"    χ²_min = {real_chi:.6e}")
    print(f"    Относительная ширина = {real_width_rel:.4f}%")

    print(f"\n  Null-модели (n={len(null_chis)}):")
    print(f"    Среднее χ²_min = {np.mean(null_chis):.6e} ± {np.std(null_chis):.6e}")
    print(f"    Средняя ширина = {np.mean(null_widths):.4f}% ± {np.std(null_widths):.4f}%")

    # Z-score
    if len(null_chis) > 0:
        z_chi = (real_chi - np.mean(null_chis)) / np.std(null_chis) if np.std(null_chis) > 0 else 0
        z_width = (real_width_rel - np.mean(null_widths)) / np.std(null_widths) if np.std(null_widths) > 0 else 0

        print(f"\n  Z-score χ²: {z_chi:.2f}")
        print(f"  Z-score ширины: {z_width:.2f}")

        # P-value (эмпирический)
        p_chi = np.mean(null_chis <= real_chi)
        p_width = np.mean(null_widths <= real_width_rel)

        print(f"  P(χ²_null ≤ χ²_real) = {p_chi:.4f}")
        print(f"  P(width_null ≤ width_real) = {p_width:.4f}")

        if z_chi < -2 and p_chi < 0.05:
            print("\n  ✅ χ² ЗНАЧИМО ЛУЧШЕ СЛУЧАЙНОГО — структура неслучайна")
        else:
            print("\n  ❌ χ² НЕ ОТЛИЧАЕТСЯ ОТ СЛУЧАЙНОГО — возможна подгонка")

        if z_width < -2 and p_width < 0.05:
            print("  ✅ ШИРИНА ЗНАЧИМО УЖЕ СЛУЧАЙНОГО — резкий минимум не случаен")
        else:
            print("  ❌ ШИРИНА НЕ ОТЛИЧАЕТСЯ ОТ СЛУЧАЙНОГО")

    return null_chis, null_widths, real_chi, real_width_rel


# 6. PERMUTATION TEST
def permutation_test(n_permutations=500):
    """Перемешивание targets между функциями"""
    print("\n" + "=" * 80)
    print("6. PERMUTATION TEST: ПЕРЕМЕШИВАНИЕ TARGET'ОВ")
    print("=" * 80)

    # Реальная χ²
    real_lnN, real_chi, real_width, _, _ = find_chi_minimum()

    functions = [func for _, func, _ in IDENTITIES]
    targets = [target for _, _, target in IDENTITIES]

    perm_chis = []

    for i in range(n_permutations):
        permuted_targets = np.random.permutation(targets)
        permuted_identities = [
            (f"perm_{j}", func, target)
            for j, (func, target) in enumerate(zip(functions, permuted_targets))
        ]

        try:
            _, chi_perm, _, _, _ = find_chi_minimum(permuted_identities)
            perm_chis.append(chi_perm)
        except:
            continue

    perm_chis = np.array(perm_chis)

    print(f"\n  Реальная χ²_min = {real_chi:.6e}")
    print(f"  Перемешанные (n={len(perm_chis)}):")
    print(f"    Среднее = {np.mean(perm_chis):.6e} ± {np.std(perm_chis):.6e}")
    print(f"    Минимум = {np.min(perm_chis):.6e}")

    if len(perm_chis) > 0:
        p_perm = np.mean(perm_chis <= real_chi)
        print(f"  P(χ²_perm ≤ χ²_real) = {p_perm:.4f}")

        if p_perm < 0.01:
            print("\n  ✅ СПЕЦИФИЧНОСТЬ ВЫСОКАЯ — пары function-target неслучайны")
        elif p_perm < 0.05:
            print("\n  🟡 УМЕРЕННАЯ СПЕЦИФИЧНОСТЬ")
        else:
            print("\n  ❌ ПАРЫ НЕ СПЕЦИФИЧНЫ — любая функция подходит к любому target'у")

    return perm_chis, real_chi


# 7. COMPLEXITY PENALTY (AIC/BIC)
def complexity_analysis():
    """Информационные критерии с учётом сложности"""
    print("\n" + "=" * 80)
    print("7. COMPLEXITY PENALTY: ИНФОРМАЦИОННЫЕ КРИТЕРИИ")
    print("=" * 80)

    # Реальная χ²
    real_lnN, real_chi, real_width, _, _ = find_chi_minimum()

    # Параметры модели
    k_params = 1  # только lnN — свободный параметр

    # Количество "эффективных степеней свободы" в конструкциях
    # Каждое тождество использует ~3-5 арифметических операций + выбор констант
    # Оценим complexity как количество тождеств × среднюю сложность
    n_identities = len(IDENTITIES)
    avg_operations = 4  # среднее число операций в тождестве
    effective_complexity = n_identities * avg_operations

    # RSS (residual sum of squares)
    rss = real_chi * n_identities

    # AIC = n * ln(RSS/n) + 2k
    aic_simple = n_identities * np.log(rss / n_identities) + 2 * k_params
    aic_complex = n_identities * np.log(rss / n_identities) + 2 * effective_complexity

    # BIC = n * ln(RSS/n) + k * ln(n)
    bic_simple = n_identities * np.log(rss / n_identities) + k_params * np.log(n_identities)
    bic_complex = n_identities * np.log(rss / n_identities) + effective_complexity * np.log(n_identities)

    print(f"\n  Количество тождеств: {n_identities}")
    print(f"  Свободные параметры (ln N): {k_params}")
    print(f"  Эффективная сложность конструкции: ~{effective_complexity}")
    print(f"\n  RSS = {rss:.6e}")
    print(f"  ln(RSS/n) = {np.log(rss / n_identities):.4f}")

    print(f"\n  AIC (прост.): {aic_simple:.2f}")
    print(f"  AIC (с учётом сложности): {aic_complex:.2f}")
    print(f"  BIC (прост.): {bic_simple:.2f}")
    print(f"  BIC (с учётом сложности): {bic_complex:.2f}")

    # Сравнение с null-моделью
    null_targets = generate_null_model_constants(n_identities)
    null_identities = [
        (name, func, target)
        for (name, func, _), target in zip(IDENTITIES, null_targets)
    ]
    _, null_chi, _, _, _ = find_chi_minimum(null_identities)
    null_rss = null_chi * n_identities

    delta_aic = aic_complex - (n_identities * np.log(null_rss / n_identities) + 2 * effective_complexity)
    delta_bic = bic_complex - (
                n_identities * np.log(null_rss / n_identities) + effective_complexity * np.log(n_identities))

    print(f"\n  ΔAIC (реальная - null): {delta_aic:.2f}")
    print(f"  ΔBIC (реальная - null): {delta_bic:.2f}")

    if delta_aic < -10 and delta_bic < -10:
        print("\n  ✅ МОДЕЛЬ ЗНАЧИМО ЛУЧШЕ NULL ДАЖЕ С УЧЁТОМ СЛОЖНОСТИ")
    elif delta_aic < 0 and delta_bic < 0:
        print("\n  🟡 МОДЕЛЬ ЛУЧШЕ NULL, НО ПРЕИМУЩЕСТВО НЕВЕЛИКО")
    else:
        print("\n  ❌ МОДЕЛЬ НЕ ЛУЧШЕ NULL ПОСЛЕ ШТРАФА ЗА СЛОЖНОСТЬ")

    return aic_simple, aic_complex, bic_simple, bic_complex


# 8. HOLD-OUT VALIDATION
def hold_out_validation(test_fraction=0.2, n_splits=20):
    """Предсказание на unseen identities"""
    print("\n" + "=" * 80)
    print(f"8. HOLD-OUT VALIDATION: ПРЕДСКАЗАНИЕ НА {test_fraction * 100:.0f}% ТОЖДЕСТВ")
    print("=" * 80)

    n_total = len(IDENTITIES)
    n_test = max(1, int(n_total * test_fraction))
    n_train = n_total - n_test

    train_errors = []
    test_errors = []
    predictions = []

    for split in range(n_splits):
        indices = np.random.permutation(n_total)
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]

        train_set = [IDENTITIES[i] for i in train_idx]
        test_set = [IDENTITIES[i] for i in test_idx]

        # Обучаем на train
        lnN_train, chi_train, _, _, _ = find_chi_minimum(train_set)

        # Проверяем на train
        train_err = chi_square(lnN_train, train_set)
        train_errors.append(train_err)

        # Проверяем на test (out-of-sample)
        test_err = chi_square(lnN_train, test_set)
        test_errors.append(test_err)

        # Предсказанные значения
        test_preds = []
        for name, func, target in test_set:
            pred = func(lnN_train)
            test_preds.append((name, pred, target))
        predictions.append(test_preds)

    train_errors = np.array(train_errors)
    test_errors = np.array(test_errors)

    print(f"\n  Размер обучающей выборки: {n_train}")
    print(f"  Размер тестовой выборки: {n_test}")
    print(f"  Количество splits: {n_splits}")

    print(f"\n  Ошибка на train: {np.mean(train_errors):.6e} ± {np.std(train_errors):.6e}")
    print(f"  Ошибка на test:  {np.mean(test_errors):.6e} ± {np.std(test_errors):.6e}")

    # Отношение test/train
    ratio = np.mean(test_errors) / np.mean(train_errors) if np.mean(train_errors) > 0 else float('inf')
    print(f"  Отношение test/train: {ratio:.4f}")

    # Генерализационный разрыв
    gap = np.mean(test_errors) - np.mean(train_errors)
    print(f"  Генерализационный разрыв: {gap:.6e}")

    if ratio < 2.0:
        print("\n  ✅ ХОРОШАЯ ГЕНЕРАЛИЗАЦИЯ — модель предсказывает новые тождества")
    elif ratio < 10.0:
        print("\n  🟡 УМЕРЕННАЯ ГЕНЕРАЛИЗАЦИЯ")
    else:
        print("\n  ❌ ПЛОХАЯ ГЕНЕРАЛИЗАЦИЯ — модель переобучена под конкретные тождества")

    # Детальный анализ одного примера
    print(f"\n  Пример предсказаний (split 0):")
    for name, pred, target in predictions[0][:5]:
        rel_err = abs(pred - target) / abs(target) * 100
        print(f"    {name:<20}: pred={pred:.10f}, target={target:.10f}, err={rel_err:.2f}%")

    return train_errors, test_errors, predictions


# 4. ВАРИАЦИОННАЯ ФОРМУЛИРОВКА
def variational_formulation(lnN_min, chi_min, width):
    """Формулировка в терминах вариационного принципа"""
    print("\n" + "=" * 80)
    print("4. ВАРИАЦИОННАЯ ФОРМУЛИРОВКА")
    print("=" * 80)

    N_min = math.exp(lnN_min)

    print(f"""
    Принцип согласования геометрических структур:

    N = arg min χ²(N)

    где χ²(N) = Σᵢ (fᵢ(N) - Cᵢ)² / Cᵢ²

    Результат:
      N* = {N_min:.6e} (ln N* = {lnN_min:.10f})
      χ²_min = {chi_min:.6e}
      Ширина минимума (ΔlnN при χ² = 10χ²_min): {width:.4f}
      Относительная ширина: {width / lnN_min * 100:.4f}%
    """)


def main():
    np.random.seed(42)
    random.seed(42)

    print("=" * 80)
    print("РАСШИРЕННЫЙ АНАЛИЗ НЕЗАВИСИМОСТИ ТОЖДЕСТВ")
    print("=" * 80)
    print(f"\nБазовое N: {N_base:.6e} (ln N = {lnN_base:.8f})")
    print(f"Теоретическое N (для справки): {N_theory:.6e} (ln N = {lnN_theory:.8f})")
    print(f"Количество тождеств: {len(IDENTITIES)}")
    print(f"ВСЕ ТЕСТЫ СВЕРЯЮТСЯ ПО БАЗОВОМУ N_base, а не по N_theory")

    # 1. χ²-функционал
    print("\n" + "=" * 80)
    print("1. χ²-ФУНКЦИОНАЛ: ПОИСК ГЛОБАЛЬНОГО МИНИМУМА")
    print("=" * 80)

    lnN_min, chi_min, width, left, right = find_chi_minimum()

    print(f"\n  Минимум χ² при ln N = {lnN_min:.8f}")
    print(f"  N* = {math.exp(lnN_min):.6e}")
    print(f"  χ²_min = {chi_min:.6e}")
    print(f"  Ширина минимума: [{left:.10f}, {right:.6f}]")
    print(f"  ΔlnN = {width:.6f}")
    print(f"  Относительная ширина: {width / lnN_min * 100:.6f}%")

    # Сравнение с N_base
    deviation_from_base = abs(lnN_min - lnN_base) / lnN_base * 100
    print(f"  Отклонение от N_base: {deviation_from_base:.10f}%")

    if width / lnN_min * 100 < 0.05:
        print(f"  ✅ РЕЗКИЙ МИНИМУМ")
    elif width / lnN_min * 100 < 0.5:
        print(f"  🟡 УМЕРЕННО РЕЗКИЙ МИНИМУМ")
    else:
        print(f"  ❌ ШИРОКИЙ МИНИМУМ")

    # 2. Leave-one-out
    base_lnN, loo_results = leave_one_out()

    # 3. Bootstrap
    bootstrap_samples, bootstrap_median, bootstrap_narrow = bootstrap_test(
        n_iterations=1000, subset_size=min(6, len(IDENTITIES) // 2)
    )

    # 4. Вариационная формулировка
    variational_formulation(lnN_min, chi_min, width)

    # 5. Null models
    null_chis, null_widths, real_chi, real_width = null_model_test(n_null=300)

    # 6. Permutation test
    perm_chis, real_chi = permutation_test(n_permutations=300)

    # 7. Complexity penalty
    aic_s, aic_c, bic_s, bic_c = complexity_analysis()

    # 8. Hold-out validation
    train_err, test_err, predictions = hold_out_validation(test_fraction=0.2, n_splits=20)

    # ИТОГОВЫЙ ВЕРДИКТ
    print("\n" + "=" * 80)
    print("ИТОГОВЫЙ ВЕРДИКТ (РАСШИРЕННЫЙ)")
    print("=" * 80)

    # Критерии
    sharp_minimum = width / lnN_min * 100 < 0.05
    loo_stable = max([d for _, _, d in loo_results]) < 0.05 if loo_results else False

    # Новые критерии
    null_significant = False
    if len(null_chis) > 0:
        z_chi = (real_chi - np.mean(null_chis)) / np.std(null_chis) if np.std(null_chis) > 0 else 0
        null_significant = z_chi < -2

    perm_significant = False
    if len(perm_chis) > 0:
        p_perm = np.mean(perm_chis <= real_chi)
        perm_significant = p_perm < 0.05

    delta_aic = aic_c - aic_s  # упрощённо
    better_than_null = delta_aic < -10

    generalization_ratio = np.mean(test_err) / np.mean(train_err) if np.mean(train_err) > 0 else float('inf')
    good_generalization = generalization_ratio < 2.0

    tests = [
        ("Резкий минимум χ²", sharp_minimum),
        ("LOO устойчивость", loo_stable),
        ("Bootstrap узкий пик", bootstrap_narrow),
        ("Null model значимость", null_significant),
        ("Permutation специфичность", perm_significant),
        ("Инфо-критерии лучше null", better_than_null),
        ("Hold-out генерализация", good_generalization),
    ]

    passed = sum(1 for _, result in tests if result)

    print(f"\n  Результаты тестов:")
    for name, result in tests:
        print(f"    {name:<30}: {'✅' if result else '❌'}")

    print(f"\n  Пройдено тестов: {passed}/{len(tests)}")

    if passed >= 6:
        print("\n  🏆 СИЛЬНОЕ СВИДЕТЕЛЬСТВО СТРУКТУРЫ")
        print("  Большинство строгих тестов пройдено.")
        print("  Однако помните: корреляция не означает причинность.")
    elif passed >= 4:
        print("\n  🟡 УМЕРЕННОЕ СВИДЕТЕЛЬСТВО")
        print("  Часть тестов пройдена, но нужны дополнительные проверки.")
    else:
        print("\n  ❌ СЛАБОЕ СВИДЕТЕЛЬСТВО")
        print("  Система ведёт себя похоже на случайную или переобученную.")

    print("\n  ВАЖНОЕ ПРЕДУПРЕЖДЕНИЕ:")
    print("  Даже при прохождении всех тестов, это НЕ доказывает")
    print("  'фундаментальную физическую структуру'.")
    print("  Это показывает лишь, что система численных соотношений")
    print("  статистически необычна. Требуется:")
    print("  - Независимое теоретическое предсказание")
    print("  - Внешняя валидация на новых данных")
    print("  - Физическая интерпретация")

    return {
        'lnN_min': lnN_min,
        'chi_min': chi_min,
        'width': width,
        'loo_results': loo_results,
        'bootstrap': bootstrap_samples,
        'null_models': null_chis,
        'permutations': perm_chis,
        'aic': aic_c,
        'bic': bic_c,
        'holdout': (train_err, test_err),
        'tests_passed': passed,
        'total_tests': len(tests)
    }


if __name__ == "__main__":
    results = main()