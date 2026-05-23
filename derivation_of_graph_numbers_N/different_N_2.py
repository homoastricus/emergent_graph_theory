import numpy as np
import math

from mpmath import lambertw
from scipy.optimize import minimize_scalar, fsolve

K = 6.0
pi = math.pi
lnK = math.log(K)

def lnN_geometric():
    return float((K - lnK) / (1.0/3.0 - 1.0/pi))

def lnN_zeta():
    # ВАЖНО: интерпретируем как ln N (НЕ exp)
    exponent = 1.5 + pi ** 2 / 6.0
    lnN_zeta = 6 ** exponent
    return lnN_zeta

def lnN_alpha(alpha=1/137.035999084):
    return 2 * lnK**2 / (pi * alpha)

def lnN_hypothesis():
    g = float(lnN_geometric())
    return float(g - pi*K/g)


def lnN_hbar(hbar=1.054571817e-34):
    """
    Решает уравнение: hbar = (ln N)^3 / (6 * N^(1/3))
    Работает в логарифмическом масштабе для устойчивости
    """
    K = 6.0


    # Определим функцию от x = ln(N)
    # hbar = x^3 / (K * exp(x/3))
    # hbar * K = x^3 * exp(-x/3)
    # Логарифмируем: ln(hbar*K) = 3*ln(x) - x/3
    # f(x) = 3*ln(x) - x/3 - ln(hbar*K) = 0

    def equation(x):
        # x = ln(N)
        if x <= 0:
            return 1e100  # x должно быть положительным
        return 3 * np.log(x) - x / 3 - np.log(hbar * K)


    # Начальное приближение
    # Из физических соображений, x = ln(N) ~ 280
    x_guess = 280.0

    x_solution = fsolve(equation, x_guess, maxfev=1000)[0]

    return x_solution

def lnN_lightspeed(lightspeed=299792458):
    # c_val = pi * (lnN ** 4) / (K ** 2 * lnK)
    return (lightspeed*(K ** 2) * lnK/ pi )**(1/4)

def lnN_plank_mass(mass=2.176434e-8):
    #mP_val = K / (pi * 4 * lnN ** 3)
    return (K/(mass*4*pi))**(1/3)

geo_val = lnN_geometric()
zeta_val = lnN_zeta()
alpha_val = lnN_alpha()
phys_val = lnN_hypothesis()
print(zeta_val)
print("geometric_zeta=", geo_val - zeta_val)
print("geometric_alpha=", geo_val - alpha_val)
print("geometric_phys=", geo_val - phys_val)
print("zeta_alpha=", zeta_val - alpha_val)
print("zeta_phys =", zeta_val - phys_val)
print("alpha_phys =", alpha_val - phys_val)

lnN_plank_mass = lnN_plank_mass()
print("lnN_plank_mass =", lnN_plank_mass)
print(lnN_plank_mass)
print("zeta_val =", zeta_val)
print("zeta_plank_mass =", zeta_val - lnN_plank_mass)


def plank_mass(zeta_val):
    #mP_val = K / (pi * 4 * lnN ** 3)
    return K / (pi * 4 * zeta_val ** 3)
# zeta_hbar_val = lnN_hbar()
# print(zeta_hbar_val)
# print("zeta_hbar =", zeta_val - zeta_hbar_val)
#
# zeta_lightspeed = lnN_lightspeed()
# print("zeta_lightspeed =", zeta_val - zeta_lightspeed)

print("plank_mass =", 2.176434e-8 - plank_mass(zeta_val))
# alfa_exp = 7.2973525693e-3
# zeta_N = math.exp(zeta_val)
# print(zeta_N)
# ALFA_TEST = 2 * lnK**2 / (pi * (zeta_N + pi**6/(math.sqrt(6)*zeta_N**2)))
# print(f"dif {ALFA_TEST-alfa_exp:.10f}")


# Базовое значение
X = K ** (1.5 + pi**2 / 6.0)  # ln N_zeta

# CODATA
mP_CODATA = 2.176434e-8

# Ведущий порядок
mP_leading = K / (4 * pi * X**3)

# Поправка
correction = K / (4*pi**2 * X**4) + K**2 / (4*pi**2 * X**5)
# Уточнённое значение
mP_corrected = mP_leading + correction
tested = 2.176434e-8 - plank_mass(zeta_val)
print(f"  mP (tested)      = {correction/tested:.10e} кг")


print("ПЛАНКОВСКАЯ МАССА: УТОЧНЁННАЯ ФОРМУЛА")
print(f"  mP (ведущий порядок) = {mP_leading:.10e} кг")
print(f"  Поправка             = {correction:.6e} кг")
print(f"  mP (уточнённое)      = {mP_corrected:.10e} кг")
print(f"  mP (CODATA)          = {mP_CODATA:.10e} кг")
print(f"  Ошибка ведущего пор. = {abs(mP_leading/mP_CODATA - 1)*100:.6f}%")
print(f"  Ошибка уточнённого   = {abs(mP_corrected/mP_CODATA - 1)*100:.6f}%")