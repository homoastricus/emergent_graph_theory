# Требуется: pip install mpmath
import math
import mpmath as mp

# Настройка точности (количество десятичных знаков). Увеличьте при необходимости.
mp.mp.dps = 80  # ~ 80 decimal digits of precision

LOG2E = 1.0 / math.log(2.0)


def log2_comb_adaptive(k_big, E_big, tol_sparse=1e-6):
    """
    Устойчивое вычисление log2 C(k, E) для очень больших k, E.
    Параметры k_big, E_big могут быть Python int или mpmath.mpf.
    Возвращает mpmath.mpf (log2).
    Алгоритм:
     - Если E == 0 или E == k: вернуть 0
     - Если E << k (p = E/k < tol_sparse): используем асимптику k*H(p) ≈ E*ln(k/E)
     - Иначе используем loggamma с mpmath: ln C = ln Gamma(k+1) - ln Gamma(E+1) - ln Gamma(k-E+1)
    """
    # переведём в mp.mpf
    k = mp.mpf(k_big)
    E = mp.mpf(E_big)

    if E == 0:
        return mp.mpf('0.0')
    if E == k:
        return mp.mpf('0.0')
    if E > k:
        return mp.ninf

    # симметрия: возьмём min(E, k-E) для лучшей устойчивости (меньшая опция)
    if E > k/2:
        E = k - E

    p = E / k

    # Очень разрежённый режим: p << 1
    if p < tol_sparse:
        # Используем более точную форму k * H(p) с H(p)= -p ln p - (1-p) ln(1-p)
        # Но при p<<1, H(p) ≈ p ln(1/p) + p^2/2 ... — достаточно k*H(p)
        # Вычисляем в mp с хорошей точностью
        H_p = - (p * mp.log(p) + (1 - p) * mp.log(1 - p))
        lnC = k * H_p  # натурный логарифм
        return lnC / mp.log(2)
    else:
        # Общий режим: используем loggamma (mp.loggamma)
        # ln C = lnGamma(k+1) - lnGamma(E+1) - lnGamma(k-E+1)
        # mp.loggamma работает с mp.mpf и большой точностью
        lnC = mp.loggamma(k + 1) - mp.loggamma(E + 1) - mp.loggamma(k - E + 1)
        return lnC / mp.log(2)


# Вспомогательные функции для твоего анализа:
def compute_k_from_N(N):
    # N может быть int или float-like. Вернём mp.mpf
    Nmp = mp.mpf(N)
    return (Nmp * (Nmp - 1)) / 2


def entropy_system_mp(N, m):
    """
    Возвращает (H_system_log2, E, k) с высокой точностью.
    Здесь E = N*m/2 (можно ввести как mp)
    """
    Nmp = mp.mpf(N)
    m_mp = mp.mpf(m)
    E = (Nmp * m_mp) / 2
    k = compute_k_from_N(Nmp)

    # Используем адаптивную функцию
    Hsys_log2 = log2_comb_adaptive(k, E, tol_sparse=1e-9)
    return Hsys_log2, E, k


if __name__ == "__main__":
    mp.mp.dps = 80

    Nodes = 1e122
    corrs = math.log(Nodes)
    N = mp.mpf(Nodes)          # твой пример
    m = mp.mpf(corrs)        # как в выводе

    Hsys, E, k = entropy_system_mp(N, m)
    print("k (≈)   :", mp.nstr(k, 20))
    print("E (≈)   :", mp.nstr(E, 20))
    print("H_system (log2) ≈", mp.nstr(Hsys, 20))
