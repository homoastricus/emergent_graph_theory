import math
import numpy as np

# =========================
# ПАРАМЕТРЫ
# =========================
K = 6.0
pi = math.pi
lnK = math.log(K)

# Текущее значение N
N_current = 4.198e121
lnN_current = math.log(N_current)

print( math.log(math.log(N_current)))
# =========================
# ВЫЧИСЛЕНИЕ ОТНОШЕНИЙ
# =========================

def compute_ratios(N):
    """Вычисляет lnN / N^(1/3), lnN^2 / N^(1/3), lnN^3 / N^(1/3)"""
    lnN = math.log(N)
    N13 = N ** (1 / 3)

    ratio_1 = lnN / N13
    ratio_2 = (lnN ** 2) / N13
    ratio_3 = (lnN ** 3) / N13

    return lnN, N13, ratio_1, ratio_2, ratio_3

