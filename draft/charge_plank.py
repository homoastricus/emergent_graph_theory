import math

import math

import numpy as np


def before(K, p, N):
    lambda_param = (np.log(K * p) / np.log(N)) ** 2
    return math.sqrt(4 * math.pi *
                            (((np.log(K) ** 2) / (4 * lambda_param ** 2 * K ** 2)) * N ** (-1 / 3) / (6 * 3.141592)) *
                            (3.1415926 * (1 / np.sqrt(K * p) *
                                       (2 * math.pi / (np.sqrt(K * p) * lambda_param) * N ** (1 / 6)) / ((np.log(K) ** 2) /
                                                                                    (4 * lambda_param ** 2 * K ** 2)))
                             / lambda_param ** 2 * N ** (-1 / 6)) *
                            (((((np.log(K * p) / np.log(N)) ** 4) * K)
                  / (2 * math.pi * ((3.1415926 * (1 / np.sqrt(K * p) * (2 * math.pi / (np.sqrt(K * p) * lambda_param) * N ** (1 / 6)) /
                                               ((np.log(K) ** 2) / (4 * lambda_param ** 2 * K ** 2)))
                                     / lambda_param ** 2 * N ** (-1 / 6)) ** 2) * ((np.log(K) ** 2) /
                                                                                             (4 * lambda_param ** 2 * K ** 2) * N ** (-1 / 3)
                                                                                             / (6 * 3.141592)) * (N ** (1 / 3)) * (math.pi * math.log(N) ** 7 / (
                    3 * abs(math.log(K * p) ** 6)  * (p * K) ** (3 / 2) * N ** (1 / 3)))))
                     ))


import math


# Или вариант с исходной формой под корнем:
def planck_charge_compact(K, p, N):
    lnK = math.log(K)
    lnKp = math.log(K * p)
    lnN = math.log(N)
    return math.sqrt(3 * p ** (5 / 2) * K ** (1.5) * lnK ** 2 * lnKp ** 12 / (4 * math.pi ** 3 * lnN ** 13))

# Пример использования для нашей Вселенной
if __name__ == "__main__":
    # Параметры нашей Вселенной
    K = 8.0
    p = 0.0527
    N = 0.95e123  # 9.5 × 10^122

    result = planck_charge_compact(K, p, N)

    print("ПЛАНКОВСКИЙ ЗАРЯД ИЗ ПАРАМЕТРОВ ГРАФА")

    qp = 1.8755459e-18  # Кулон

    print(f"\n до упрощения q_p = {before(K, p, N):.3e} ")
    print(f"\n после q_p = {result:.3e} ")
