import numpy as np

# -------------------------------
# ФИЗИЧЕСКИЕ "эталонные" значения
# -------------------------------
phys_hbar = 1.054571817e-34  # Дж·с
phys_c = 2.99792458e8  # м/с
phys_G = 6.67430e-11  # м^3/(кг·с^2)
phys_lP = 1.616255e-35  # м
phys_tP = 5.391247e-44  # с
phys_kB = 1.380649e-23  # Дж/К

# -------------------------------
# ВХОДНЫЕ ПАРАМЕТРЫ модели
# -------------------------------
K = 7.62  # локальная связность
p = 0.05  # вероятность дальних связей
lambda_param = 1.0  # спектральный параметр сети (масштаб)

# Массив N для проверки зависимости
N_values = np.array([1e5, 1e10, 1e15, 1e20, 1e30, 1e50, 1e70, 1e100, 1e150, 1e180])


# -------------------------------
# ФУНКЦИИ РАСЧЕТА
# -------------------------------
def calc_c(K, p, lambda_param):
    return np.sqrt(lambda_param * p * (K - 1))


def calc_hbar(K, lambda_param, N):
    return (np.log(K) ** 2) / (4 * lambda_param ** 2 * K ** 2 * np.sqrt(N * np.log(K)))


def calc_lP(hbar, c):
    return np.sqrt(hbar / c ** 3)


def calc_tP(lP, c):
    return lP / c


def calc_G(lP, c, hbar):
    return lP ** 2 * c ** 3 / hbar


def calc_kB(N):
    return 1 / N  # Пропорционально 1/N для теста


# -------------------------------
# РАСЧЕТ И ВЫВОД
# -------------------------------
for N in N_values:
    c_val = calc_c(K, p, lambda_param)
    hbar_val = calc_hbar(K, lambda_param, N)
    lP_val = calc_lP(hbar_val, c_val)
    tP_val = calc_tP(lP_val, c_val)
    G_val = calc_G(lP_val, c_val, hbar_val)
    kB_val = calc_kB(N)

    print(f"\n=== N = {N:.3e} ===")
    print(f"c       : {c_val:.3e} | физическое: {phys_c:.3e}")
    print(f"hbar    : {hbar_val:.3e} | физическое: {phys_hbar:.3e}")
    print(f"l_P     : {lP_val:.3e} | физическое: {phys_lP:.3e}")
    print(f"t_P     : {tP_val:.3e} | физическое: {phys_tP:.3e}")
    print(f"G       : {G_val:.3e} | физическое: {phys_G:.3e}")
    print(f"k_B     : {kB_val:.3e} | физическое: {phys_kB:.3e}")
