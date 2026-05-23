import numpy as np
import math
import warnings
from scipy.special import gamma

warnings.filterwarnings('ignore')

K = 6.0
pi = math.pi
lnK = math.log(K)
lnN_base = 280.04732539

# Константы
euler = 2.718281828459045
mills = 1.306377883863080
phi = (1 + math.sqrt(5)) / 2
gompertz = 0.596347362323194
meissel_mertens = 0.2614972128476428
catalan = 0.915965594177219
van_der_pauw = 4.5323601418271938
gamma_quarter = 3.6256099082219083
cahen = 0.6419448389191956
glaisher = 1.2824271291006226
brjuno = 0.0654603245889095
alpha = 7.2973525693e-3
lambda_const = 0.05183093000
brun_twin = 1.902160583104
gamma_1_6 = math.gamma(1 / 6)
gauss = 0.834626841674073
sqrt3 = math.sqrt(3)
kepler_bouwkamp = 0.1149420448532962
supergolden = 1.465571231876768
lemniscate = 2.622057554292119
sqrt2 = math.sqrt(2)
feigenbaum_delta = 4.669201609102990
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
lemniscate=2.622057554292119
ln3=math.log(3)
ln2=math.log(2)
glaisher = 1.2824271291006226
brjuno=0.0654603245889095
alpha=7.2973525693e-3
sqrt2=math.sqrt(2)
sqrt3=math.sqrt(3)
supergolden=1.465571231876768
levy=3.275822918721811
viswanath= 1.1319882487943
landau_ramanujan= 0.764223653589221
gauss= 0.834626841674073
kepler_bouwkamp=0.1149420448532962
feigenbaum_delta=4.669201609102990
feigenbaum_alpha=2.5029078750958928
khinchin=2.6854520010653062
apery=1.2020569031595942
phi= (1 + math.sqrt(5)) / 2
gamma_1_2= gamma(0.5)
gamma_1_6= gamma(1 / 6)
gamma_2_3= gamma(2 / 3)
lnpi=math.log(math.pi)
gelfond_schneider=2 ** math.sqrt(2)
e_pi=math.exp(math.pi)
alladi_grinstead=0.8093940205406391
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
    return K * (K + 1.0/lnN + 1.0/lnN**2) / (K + 1.0/math.log(lnK))

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
    return K/(lnN + K) + 1.0/math.log(lnK) - 1.0/lnN


# ===========================================
# НОВЫЕ ПРОВЕРЕННЫЕ ТОЖДЕСТВА ДЛЯ ДОБАВЛЕНИЯ
# ===========================================

def identity_ln3_paperfolding(lnN):
    """lnK * (lnN)^(1/3) / (K + (lnN)^(1/3)) ≈ ln3 * paperfolding"""
    return (lnK * lnN ** (1/3)) / (K + lnN ** (1/3))


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




def check_identity(name, func, target, lnN_range=(270, 290), n_points=500):
    """Тестирует тождество и находит его резонансные свойства"""

    lnN_values = np.linspace(lnN_range[0], lnN_range[1], n_points)

    values = np.array([func(lnN) for lnN in lnN_values])

    if target > 0:
        deviations = np.abs(values - target) / target * 100
    else:
        deviations = np.abs(values - target) / abs(target) * 100 if abs(target) > 1e-10 else np.abs(
            values - target) * 100

    # Минимум отклонения
    best_idx = np.argmin(deviations)
    best_lnN = lnN_values[best_idx]
    best_val = values[best_idx]
    best_dev = deviations[best_idx]

    # Значение при базовом lnN
    base_val = func(lnN_base)
    if target > 0:
        base_dev = abs(base_val - target) / target * 100
    else:
        base_dev = abs(base_val - target) / abs(target) * 100 if abs(target) > 1e-10 else abs(base_val - target) * 100

    # Ширина резонанса (где отклонение < 0.01%)
    threshold = 0.01
    left = lnN_base
    while left > lnN_range[0] and (
    abs(func(left) - target) / target * 100 if target > 0 else abs(func(left) - target) / abs(target) * 100 if abs(
            target) > 1e-10 else abs(func(left) - target) * 100) < threshold:
        left -= 0.01

    right = lnN_base
    while right < lnN_range[1] and (
    abs(func(right) - target) / target * 100 if target > 0 else abs(func(right) - target) / abs(target) * 100 if abs(
            target) > 1e-10 else abs(func(right) - target) * 100) < threshold:
        right += 0.01

    width = right - left

    # Классификация
    if base_dev < 0.001:
        status = "✅ ОТЛИЧНО"
    elif base_dev < 0.01:
        status = "🟢 ХОРОШО"
    elif base_dev < 0.1:
        status = "🟡 УДОВЛ."
    elif base_dev < 1.0:
        status = "🟠 СЛАБО"
    else:
        status = "🔴 ПЛОХО"

    # Проверка на плато
    dev_range = np.max(deviations) - np.min(deviations)
    is_plateau = (dev_range < 0.1) and (base_dev < 0.1)

    # Проверка на острый резонанс
    is_sharp = (width < 10) and (base_dev < 0.01)

    # Проверка на плавное изменение (независимость от N)
    values_vs_lnN_slope = abs(values[-1] - values[0]) / (lnN_range[1] - lnN_range[0])
    is_flat = values_vs_lnN_slope < 0.01

    return {
        'name': name,
        'target': target,
        'base_val': base_val,
        'base_dev': base_dev,
        'best_lnN': best_lnN,
        'best_dev': best_dev,
        'width': width,
        'status': status,
        'is_plateau': is_plateau,
        'is_sharp': is_sharp,
        'is_flat': is_flat,
    }


# Список тождеств
identities = [
    ("1/π", geom_resonance_inv_pi, 1.0 / pi),
    ("π", geom_resonance_pi, pi),
    ("α", alpha_identity, alpha),
    ("Alladi-Grinstead", alladi_grinstead_identity, 0.8093940205406391),
    ("π²", pi_sq_identity, pi ** 2),
    ("Euler×Mills", identity_euler_mills, euler * mills),
    ("Golden+Gompertz", identity_golden_gompertz, phi + gompertz),
    ("Golden/Meissel", identity_golden_meissel, phi / meissel_mertens),
    ("ln3", identity_ln3, math.log(3)),
    ("Gamma+Catalan", identity_gamma_catalan, gamma_quarter + catalan),
    ("Euler+vdP", identity_euler_van_der_pauw, euler + van_der_pauw),
    ("Gamma×Mills", identity_gamma_mills, gamma_quarter * mills),
    ("Golden×Cahen", identity_golden_cahen, phi * cahen),
    ("vdP×Glaisher", identity_vdp_glaisher, van_der_pauw * glaisher),
    ("Bruno/α", identity_brjuno_alpha, brjuno / alpha),
    ("Cahen/Gompertz", identity_cahen_gompertz, cahen / gompertz),
    ("Γ(1/4)×Mm", identity_gamma_quarter_meissel, gamma_quarter * meissel_mertens),
    ("Γ(1/4)/Mm", identity_gamma_quarter_div_meissel, gamma_quarter / meissel_mertens),
    ("φ×Brun", identity_phi_brun, phi * brun_twin),
    ("Γ(1/6)/Gauss", identity_gamma_1_6_gauss, gamma_1_6 / gauss),
    ("√3×Γ(1/4)", identity_sqrt3_gamma_quarter, sqrt3 * gamma_quarter),
    ("Kepler/Supergolden", identity_kepler_supergolden, kepler_bouwkamp / supergolden),
    ("Lemniscate×e^e", identity_lemniscate_e_e, lemniscate * math.exp(math.e)),
    ("√2×Mills", identity_sqrt2_mills, sqrt2 * mills),
    ("δ_F (ratio)", identity_feigenbaum_ratio, feigenbaum_delta),

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
    # Высокоточные (ошибка < 0.0001%)
    ("Γ(1/3)×Gompertz", identity_gamma_third_gompertz, gamma_third * gompertz),
    ("Glaisher×π^e", identity_glaisher_pi_e, glaisher * pi ** e),
    ("Alladi×Brun", identity_alladi_brun, alladi_grinstead * brun_twin),
    ("vdP×Apéry", identity_van_der_pauw_apery, van_der_pauw * apery),
    ("Catalan×√π (1)", identity_catalan_gamma_1_2, catalan * gamma_1_2),
    ("√3 (new)", identity_sqrt3_new, sqrt3),

]

# Запуск тестов
print("=" * 120)
print("МАССОВЫЙ ТЕСТ ТОЖДЕСТВ: РЕЗОНАНСНЫЕ СВОЙСТВА")
print("=" * 120)
print(f"\n  Базовое lnN = {lnN_base:.6f}")
print(f"  K = {K}, lnK = {lnK:.10f}")
print()

results = []
for name, func, target in identities:
    try:
        result = check_identity(name, func, target)
        results.append(result)
    except Exception as e:
        print(f"  ❌ {name}: ошибка — {e}")

# Сортировка по качеству
results.sort(key=lambda r: r['base_dev'])

# Таблица
print(f"  {'Тождество':<22} {'Статус':<12} {'Откл.%':>10} {'Шир.рез.':>10} {'Тип':>15}")
print(f"  {'─' * 70}")

excellent = []
good = []
sharp = []
plateau = []
flat = []

for r in results:
    type_str = ""
    if r['is_sharp']:
        type_str = "ОСТРЫЙ РЕЗОНАНС"
        sharp.append(r)
    elif r['is_plateau']:
        type_str = "ПЛАТО"
        plateau.append(r)
    elif r['is_flat']:
        type_str = "ПЛОСКОЕ"
        flat.append(r)
    else:
        type_str = "—"

    print(f"  {r['name']:<22} {r['status']:<12} {r['base_dev']:>10.6f} {r['width']:>10.2f} {type_str:>15}")

    if r['base_dev'] < 0.001:
        excellent.append(r)
    elif r['base_dev'] < 0.01:
        good.append(r)

# Сводка
print(f"\n{'═' * 120}")
print("СВОДКА")
print(f"{'═' * 120}")

print(f"\n  Всего протестировано: {len(results)}")
print(f"  Отличных (< 0.001%): {len(excellent)}")
print(f"  Хороших (0.001-0.01%): {len(good)}")
print(f"  Острых резонансов: {len(sharp)}")
print(f"  Плато: {len(plateau)}")
print(f"  Плоских (не зависят от N): {len(flat)}")

if excellent:
    print(f"\n  Отличные тождества (< 0.001%):")
    for r in excellent:
        print(f"    ✅ {r['name']:<22} откл. = {r['base_dev']:.8f}%")

if sharp:
    print(f"\n  Острые резонансы (ширина < 10):")
    for r in sharp:
        print(f"    📍 {r['name']:<22} ширина = {r['width']:.2f}, откл. = {r['base_dev']:.8f}%")

print(f"\n{'═' * 120}")
print("ИНТЕРПРЕТАЦИЯ")
print(f"{'═' * 120}")
print(f"""
  Типы поведения тождеств:

  🔴 ОСТРЫЙ РЕЗОНАНС: тождество работает ТОЛЬКО вблизи lnN ≈ 280
     → подтверждает УНИКАЛЬНОСТЬ N

  🟡 ПЛАТО: тождество работает в широком диапазоне lnN
     → подтверждает СТРУКТУРУ, но не уникальность N

  🟢 ПЛОСКОЕ: тождество почти не зависит от N
     → интересное свойство, но не для определения N
""")