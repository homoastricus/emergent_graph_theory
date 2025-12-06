# УСТОЙЧИВЫЙ ОПТИМИЗАТОР ДЛЯ p и N (логарифм)
# Запусти в Python 3.8+ с numpy, scipy установленными.

import numpy as np
import math
from scipy.optimize import differential_evolution, minimize

# ---------------------------
#  Параметры/эталонные константы
# ---------------------------
K = 8.0

C_exp = {
    "hbar": 1.054571817e-34,
    "c":    2.99792458e8,
    "G":    6.67430e-11,
    "kB":   1.380649e-23,
    "alpha":7.2973525693e-3
}

# ---------------------------
#  Функции модели (защищённые)
# ---------------------------

def safe_log(x):
    # защита от отрицательных/ноль
    x = np.asarray(x)
    with np.errstate(divide='ignore'):
        return np.log(np.maximum(x, 1e-300))

def lambda_param(log10N, p):
    """Возвращаем lambda устойчиво. log10N = log10(N)."""
    N = 10.0**log10N
    Kp = K * p
    if Kp <= 0:
        return 1e-20
    lnKp = math.log(Kp)
    lnN = math.log(N)
    if abs(lnN) < 1e-12:
        return 1e-20
    lam = (lnKp / lnN) ** 2
    # защита от слишком маленьких/больших значений
    return float(np.clip(lam, 1e-20, 1e2))

def hbar_em_from_logN(log10N, p):
    lam = lambda_param(log10N, p)
    # формула: (ln K)^2 / (4 lam^2 K^2) * N^{-1/3} / (6π)
    N = 10.0**log10N
    num = (math.log(K) ** 2)
    den = 4 * (lam ** 2) * (K ** 2)
    base = num / den
    val = base * N ** (-1.0/3.0) / (6 * math.pi)
    if not np.isfinite(val) or val <= 0:
        return 1e-300
    return val

def c_em_from_logN(log10N, p):
    lam = lambda_param(log10N, p)
    N = 10.0**log10N
    Kp = K * p
    if Kp <= 0:
        return 1e300
    # R = 2π N^{1/6} / (sqrt(Kp) * lam)
    R = 2 * math.pi * N ** (1.0/6.0) / (math.sqrt(Kp) * lam)
    hbar_local = (math.log(K) ** 2) / (4 * lam ** 2 * K ** 2)  # note: same basic block
    # emergent ħ used in denominator — protect
    if hbar_local <= 0:
        return 1e300
    # c = π * (R / sqrt(Kp) / hbar_local) / lam^2 * N^{-1/6}
    val = math.pi * (R / math.sqrt(Kp) / hbar_local) / (lam ** 2) * N ** (-1.0/6.0)
    # clamp to reasonable physical window
    if not np.isfinite(val) or val <= 0:
        return 1e300
    return val

def G_em_from_logN(log10N, p):
    lam = lambda_param(log10N, p)
    N = 10.0**log10N
    hbar_local = (math.log(K) ** 2) / (4 * lam ** 2 * K ** 2)
    # G = hbar_local^4 / l_em^2 / lam^2  (approx used before)
    # we use stable variant: hbar_local**4 / (lam**2 * N^{1/3})
    val = (hbar_local ** 4) / (lam ** 2 * N ** (1.0/3.0))
    if not np.isfinite(val) or val <= 0:
        return 1e-300
    return val

def kB_em_from_logN(log10N, p):
    N = 10.0**log10N
    Kp = K * p
    if Kp <= 0:
        return 1e300
    lnN = math.log(N)
    lnKp = math.log(Kp)
    if abs(lnKp) < 1e-12:
        return 1e300
    val = math.pi * (lnN ** 7) / (3 * (abs(lnKp) ** 6) * (Kp) ** (3.0/2.0) * N ** (1.0/3.0))
    if not np.isfinite(val) or val <= 0:
        return 1e-300
    return val

def alpha_em_from_logN(log10N, p):
    N = 10.0**log10N
    # fallback stable formula used earlier
    if 6*N <= 1:
        return 1e-12
    val = math.log(K) / math.log(6*N)
    return val


# ---------------------------
#  Функция ошибки (устойчивая)
# ---------------------------
def loss_wrapped(x):
    # x = [p, log10N]
    p = float(x[0])
    log10N = float(x[1])

    # жесткие ограничения: не позволяем оптимизатору уходить далеко
    if p <= 0 or p >= 1:
        return 1e6 + 1e6 * abs(p)
    if log10N < 115 or log10N > 140:
        return 1e6 + 1e6 * abs(log10N - 127)

    # вычисляем модельные константы
    try:
        m = {}
        m["hbar"] = hbar_em_from_logN(log10N, p)
        m["c"]    = c_em_from_logN(log10N, p)
        m["G"]    = G_em_from_logN(log10N, p)
        m["kB"]   = kB_em_from_logN(log10N, p)
        m["alpha"]= alpha_em_from_logN(log10N, p)
    except Exception as e:
        # если что-то пошло не так — вернуть большой штраф
        return 1e6

    total = 0.0
    # суммируем усечённую (capped) лог-ошибку по каждому члену
    for key in ["hbar","c","G","kB","alpha"]:
        mval = m[key]
        eval_ = C_exp[key]
        # если модель дала абсурд — огромный штраф
        if mval <= 0 or not np.isfinite(mval):
            return 1e6
        # логарифмическая относительная ошибка
        term = (math.log(mval / eval_))**2
        # усечение — чтобы один член не доминировал бесконечно
        term = min(term, 100.0)
        total += term

    # дополнительный мягкий штраф за расстояние log10N от 122 (твое априорное знание)
    total += 0.001 * (log10N - 122.0)**2

    return total

# ---------------------------
#  Запуск: сначала differential_evolution в узкой области, потом L-BFGS-B
# ---------------------------
if __name__ == "__main__":
    bounds_de = [(0.02, 0.08), (118.0, 135.0)]
    print("Старт differential_evolution (узкая область)...")
    de_res = differential_evolution(loss_wrapped, bounds_de, maxiter=150, popsize=20, polish=True, disp=False)

    print("DE result:", de_res.x, "fun=", de_res.fun)

    # локальная полировка (L-BFGS-B) от найденной точки
    x0 = de_res.x
    bounds_local = [(0.01, 0.2), (115.0, 140.0)]
    print("Старт L-BFGS-B (полировка)...")
    res = minimize(loss_wrapped, x0, method="L-BFGS-B", bounds=bounds_local, options={"maxiter":2000})

    p_opt, log10N_opt = res.x
    N_opt = 10.0**log10N_opt

    print("\n=== РЕЗУЛЬТАТ ОПТИМИЗАЦИИ ===")
    print(f"p_opt    = {p_opt:.6e}")
    print(f"log10N   = {log10N_opt:.6f}")
    print(f"N_opt    = {N_opt:.3e}")
    print(f"loss     = {res.fun:.6e}")
    print("Статус:", res.message)

    # Выведем модельные константы
    model = {
        "hbar": hbar_em_from_logN(log10N_opt, p_opt),
        "c":    c_em_from_logN(log10N_opt, p_opt),
        "G":    G_em_from_logN(log10N_opt, p_opt),
        "kB":   kB_em_from_logN(log10N_opt, p_opt),
        "alpha":alpha_em_from_logN(log10N_opt, p_opt)
    }

    print("\nСРАВНЕНИЕ КОНСТАНТ:")
    for k in model:
        print(f"{k:5s} model={model[k]:.6e} exp={C_exp[k]:.6e} ratio={model[k]/C_exp[k]:.6e}")

