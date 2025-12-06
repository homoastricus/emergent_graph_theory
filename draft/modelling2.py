# УСТОЙЧИВЫЙ ОПТИМИЗАТОР ДЛЯ p и N (логарифм) с расширенными константами
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
    "c": 2.99792458e8,
    "G": 6.67430e-11,
    "kB": 1.380649e-23,
    "alpha": 7.2973525693e-3,
    "cosmo_lambda": 1.1056e-52,  # м⁻² (космологическая постоянная)
    "T_plank": 1.417e32,  # К (планковская температура)
    "ep0_em": 8.85e-12,  # Ф/м (электрическая постоянная)
    "mu0_em": 1.256e-6,  # Гн/м (магнитная постоянная)
    "e_plank": 1.87e-18,  # Кл (планковский заряд)
    "alfa_em": 7.297352e-3  # постоянная тонкой структуры (уже есть как alpha)
}


# ---------------------------
#  Вспомогательные функции (защищённые)
# ---------------------------

def safe_log(x):
    # защита от отрицательных/ноль
    x = np.asarray(x)
    with np.errstate(divide='ignore'):
        return np.log(np.maximum(x, 1e-300))


def lambda_param(log10N, p):
    """Возвращаем lambda устойчиво. log10N = log10(N)."""
    N = 10.0 ** log10N
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
    """Основная часть hbar без N^(-1/3)/(6π)"""
    lam = lambda_param(log10N, p)
    num = (math.log(K) ** 2)
    den = 4 * (lam ** 2) * (K ** 2)
    val = num / den
    if not np.isfinite(val) or val <= 0:
        return 1e-300
    return val


# ---------------------------
#  Основные функции модели (как у вас)
# ---------------------------

def hbar_emergent_from_logN(log10N, p):
    lam = lambda_param(log10N, p)
    # формула: (ln K)^2 / (4 lam^2 K^2) * N^{-1/3} / (6π)
    N = 10.0 ** log10N
    num = (math.log(K) ** 2)
    den = 4 * (lam ** 2) * (K ** 2)
    base = num / den
    val = base * N ** (-1.0 / 3.0) / (6 * math.pi)
    if not np.isfinite(val) or val <= 0:
        return 1e-300
    return val


def c_em_from_logN(log10N, p):
    lam = lambda_param(log10N, p)
    N = 10.0 ** log10N
    Kp = K * p
    if Kp <= 0:
        return 1e300

    # R = 2π N^{1/6} / (sqrt(Kp) * lam)
    R = 2 * math.pi * N ** (1.0 / 6.0) / (math.sqrt(Kp) * lam)

    # hbar_local (та же базовая часть, что в hbar_em)
    hbar_local = (math.log(K) ** 2) / (4 * lam ** 2 * K ** 2)

    # защита
    if hbar_local <= 0:
        return 1e300

    # c = π * (R / sqrt(Kp) / hbar_local) / lam^2 * N^{-1/6}
    val = math.pi * (R / math.sqrt(Kp) / hbar_local) / (lam ** 2) * N ** (-1.0 / 6.0)

    if not np.isfinite(val) or val <= 0:
        return 1e300
    return val


def G_em_from_logN(log10N, p):
    lam = lambda_param(log10N, p)
    N = 10.0 ** log10N

    # сначала рассчитаем l_em для точности
    Kp = K * p
    if Kp <= 0 or lam <= 0:
        return 1e-300

    l_em = 2 * math.pi / (K * p * lam) * N ** (1.0 / 6.0)

    # hbar_em (основная часть)
    hbar_em_val = hbar_em_from_logN(log10N, p)

    # G = (hbar_em^4 / l_em^2) * (1 / lam^2)
    if l_em <= 0:
        return 1e-300

    val = (hbar_em_val ** 4 / l_em ** 2) * (1 / lam ** 2)

    if not np.isfinite(val) or val <= 0:
        return 1e-300
    return val


def kB_em_from_logN(log10N, p):
    N = 10.0 ** log10N
    Kp = K * p
    if Kp <= 0:
        return 1e300

    lnN = math.log(N)
    lnKp = math.log(Kp)

    if abs(lnKp) < 1e-12:
        return 1e300

    val = math.pi * (lnN ** 7) / (3 * (abs(lnKp) ** 6) * (Kp) ** (3.0 / 2.0) * N ** (1.0 / 3.0))

    if not np.isfinite(val) or val <= 0:
        return 1e300
    return val


def alpha_em_from_logN(log10N, p):
    N = 10.0 ** log10N
    if 6 * N <= 1:
        return 1e-12
    val = math.log(K) / math.log(6 * N)
    return val


# ---------------------------
#  НОВЫЕ ФУНКЦИИ для дополнительных констант
# ---------------------------

def cosmo_lambda_em_from_logN(log10N, p):
    """Космологическая постоянная"""
    N = 10.0 ** log10N
    Kp = K * p

    lam = lambda_param(log10N, p)

    # cosmo_lambda = 3 * K * p / (π^2 * N^(1/3)) * (log(K*p)/log(N))^4
    # что эквивалентно: 3 * K * p / (π^2 * N^(1/3)) * lam^2

    if N <= 0 or lam <= 0:
        return 1e-300

    val = 3 * K * p / (math.pi ** 2 * N ** (1.0 / 3.0)) * lam ** 2

    if not np.isfinite(val) or val <= 0:
        return 1e-300
    return val


def T_plank_em_from_logN(log10N, p):
    """Планковская температура"""
    # T_plank = (hbar_emergent * c^5 / (G * kB^2))^0.5

    hbar_val = hbar_emergent_from_logN(log10N, p)
    c_val = c_em_from_logN(log10N, p)
    G_val = G_em_from_logN(log10N, p)
    kB_val = kB_em_from_logN(log10N, p)

    # Защита от некорректных значений
    if any(v <= 0 or not np.isfinite(v) for v in [hbar_val, c_val, G_val, kB_val]):
        return 1e300

    if kB_val == 0:
        return 1e300

    # T_plank = sqrt(hbar * c^5 / (G * kB^2))
    numerator = hbar_val * (c_val ** 5)
    denominator = G_val * (kB_val ** 2)

    if denominator <= 0:
        return 1e300

    val = math.sqrt(numerator / denominator)

    if not np.isfinite(val) or val <= 0:
        return 1e300
    return val


def ep0_em_from_logN(log10N, p):
    """Электрическая постоянная (вакуумная диэлектрическая проницаемость)"""
    N = 10.0 ** log10N

    lam = lambda_param(log10N, p)
    c_val = c_em_from_logN(log10N, p)
    hbar_val = hbar_emergent_from_logN(log10N, p)
    kB_val = kB_em_from_logN(log10N, p)  # это KB2 в вашем коде

    # ep0_em = ((λ^4 * K) / (2π * c^2 * hbar_emergent * N^(1/3) * kB))
    # где λ^4 = (log(K*p)/log(N))^4 * (K?)
    # Из вашего кода: (((np.log(self.K * self.p) / np.log(self.N)) ** 4) * self.K)

    Kp = K * p
    if Kp <= 0 or N <= 0 or lam <= 0:
        return 1e-300

    lnKp = math.log(Kp)
    lnN = math.log(N)

    if abs(lnN) < 1e-12:
        return 1e-300

    lam_factor = (lnKp / lnN) ** 4

    numerator = lam_factor * K
    denominator = 2 * math.pi * (c_val ** 2) * hbar_val * (N ** (1.0 / 3.0)) * kB_val

    if denominator <= 0:
        return 1e-300

    val = numerator / denominator

    if not np.isfinite(val) or val <= 0:
        return 1e-300
    return val


def mu0_em_from_logN(log10N, p):
    """Магнитная постоянная (вакуумная магнитная проницаемость)"""
    N = 10.0 ** log10N
    Kp = K * p

    if Kp <= 0:
        return 1e-300

    lnK = math.log(K)
    lnN = math.log(N)
    lnKp = math.log(Kp)

    if abs(lnKp) < 1e-12 or N <= 0:
        return 1e-300

    # mu0_em = (π * (ln K)^2 * (ln N)^15) / (36 * K^(9/2) * p^(3/2) * |ln(Kp)|^14 * N^(1/3))

    numerator = math.pi * (lnK ** 2) * (lnN ** 15)
    denominator = 36 * (K ** (9.0 / 2.0)) * (p ** (3.0 / 2.0)) * (abs(lnKp) ** 14) * (N ** (1.0 / 3.0))

    if denominator <= 0:
        return 1e-300

    val = numerator / denominator

    if not np.isfinite(val) or val <= 0:
        return 1e-300
    return val


def e_plank_em_from_logN(log10N, p):
    """Планковский заряд"""
    N = 10.0 ** log10N
    Kp = K * p

    if Kp <= 0:
        return 1e-300

    lnK = math.log(K)
    lnKp = math.log(Kp)
    lnN = math.log(N)

    if lnN <= 0 or abs(lnKp) < 1e-12:
        return 1e-300

    # e_plank = sqrt(3 * p^(5/2) * K^(1.5) * lnK^2 * lnKp^12 / (4 * π^3 * lnN^13))

    numerator = 3 * (p ** (5.0 / 2.0)) * (K ** 1.5) * (lnK ** 2) * (lnKp ** 12)
    denominator = 4 * (math.pi ** 3) * (lnN ** 13)

    if denominator <= 0:
        return 1e-300

    val = math.sqrt(numerator / denominator)

    if not np.isfinite(val) or val <= 0:
        return 1e-300
    return val


def alfa_em_em_from_logN(log10N, p):
    """Постоянная тонкой структуры (альтернативная формула)"""
    # В вашем коде есть две формулы для alpha:
    # 1. alpha_em_from_logN: math.log(K) / math.log(6*N)
    # 2. alfa_em: возможно та же самая или другая
    # Используем ту же, что и alpha_em_from_logN для консистентности
    return alpha_em_from_logN(log10N, p)


# ---------------------------
#  Функция ошибки (расширенная)
# ---------------------------
def loss_wrapped(x):
    # x = [p, log10N]
    p = float(x[0])
    log10N = float(x[1])

    # жесткие ограничения
    if p <= 0 or p >= 1:
        return 1e6 + 1e6 * abs(p)
    if log10N < 115 or log10N > 140:
        return 1e6 + 1e6 * abs(log10N - 127)

    # вычисляем ВСЕ модельные константы
    try:
        m = {}

        # Основные 5 констант (как было)
        m["hbar"] = hbar_emergent_from_logN(log10N, p)
        m["c"] = c_em_from_logN(log10N, p)
        m["G"] = G_em_from_logN(log10N, p)
        m["kB"] = kB_em_from_logN(log10N, p)
        m["alpha"] = alpha_em_from_logN(log10N, p)

        # Новые 6 констант
        m["cosmo_lambda"] = cosmo_lambda_em_from_logN(log10N, p)
        m["T_plank"] = T_plank_em_from_logN(log10N, p)
        m["ep0_em"] = ep0_em_from_logN(log10N, p)
        m["mu0_em"] = mu0_em_from_logN(log10N, p)
        m["e_plank"] = e_plank_em_from_logN(log10N, p)
        m["alfa_em"] = alfa_em_em_from_logN(log10N, p)

    except Exception as e:
        return 1e6

    total = 0.0
    # Для всех 11 констант
    all_keys = ["hbar", "c", "G", "kB", "alpha",
                "cosmo_lambda", "T_plank", "ep0_em",
                "mu0_em", "e_plank", "alfa_em"]

    well_matched_count = 0
    individual_errors = []

    for key in all_keys:
        mval = m[key]
        eval_ = C_exp[key]

        # если модель дала абсурд — огромный штраф
        if mval <= 0 or not np.isfinite(mval):
            return 1e6

        # относительная ошибка в процентах
        if mval > 0 and eval_ > 0:
            rel_error = abs(mval / eval_ - 1.0) * 100  # в процентах
            log_error = (math.log(mval / eval_)) ** 2

            # Проверяем, хорошо ли согласуется (0.001% = 0.00001 в долях)
            if rel_error < 0.001:  # 0.001% отклонение
                well_matched_count += 1
                # Для хорошо согласованных констант используем меньший вес ошибки
                term = log_error * 0.1  # уменьшаем вклад в 10 раз
            else:
                term = log_error

            term = min(term, 100.0)  # усечение
            individual_errors.append((key, rel_error, term))
            total += term
        else:
            return 1e6

    # ПРАВИЛО БОЛЬШИНСТВА: если 8, 9 или 10 констант хорошо согласованы (0.001%),
    # то суммарная ошибка уменьшается в 100 раз
    if well_matched_count >= 8:
        total = total / 100.0
        # Дополнительный бонус: чем больше хорошо согласованных, тем лучше
        bonus = (11 - well_matched_count) * 0.01  # штраф за плохие
        total = total * (1.0 + bonus)

        # Для отладки можно вывести информацию
        if well_matched_count >= 9:
            print(f"  Найдено хорошее решение: {well_matched_count}/11 констант с ошибкой <0.001%")

    # дополнительный мягкий штраф за расстояние log10N от 122
    total += 0.001 * (log10N - 122.0) ** 2

    return total


# ---------------------------
#  Запуск оптимизации
# ---------------------------
if __name__ == "__main__":
    bounds_de = [(0.02, 0.08), (118.0, 135.0)]
    print("Старт differential_evolution (узкая область)...")
    de_res = differential_evolution(loss_wrapped, bounds_de, maxiter=150, popsize=20, polish=True, disp=False)

    print("DE result:", de_res.x, "fun=", de_res.fun)

    # локальная полировка
    x0 = de_res.x
    bounds_local = [(0.01, 0.2), (115.0, 140.0)]
    print("Старт L-BFGS-B (полировка)...")
    res = minimize(loss_wrapped, x0, method="L-BFGS-B", bounds=bounds_local, options={"maxiter": 2000})

    p_opt, log10N_opt = res.x
    N_opt = 10.0 ** log10N_opt

    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТ ОПТИМИЗАЦИИ")
    print("=" * 60)
    print(f"p_opt    = {p_opt:.6e}")
    print(f"log10N   = {log10N_opt:.6f}")
    print(f"N_opt    = {N_opt:.3e}")
    print(f"loss     = {res.fun:.6e}")
    print("Статус:", res.message)

    # Выведем ВСЕ модельные константы
    print("\n" + "=" * 60)
    print("СРАВНЕНИЕ ВСЕХ КОНСТАНТ")
    print("=" * 60)

    # Создаем словарь со всеми константами
    model_all = {
        "hbar": hbar_emergent_from_logN(log10N_opt, p_opt),
        "c": c_em_from_logN(log10N_opt, p_opt),
        "G": G_em_from_logN(log10N_opt, p_opt),
        "kB": kB_em_from_logN(log10N_opt, p_opt),
        "alpha": alpha_em_from_logN(log10N_opt, p_opt),
        "cosmo_lambda": cosmo_lambda_em_from_logN(log10N_opt, p_opt),
        "T_plank": T_plank_em_from_logN(log10N_opt, p_opt),
        "ep0_em": ep0_em_from_logN(log10N_opt, p_opt),
        "mu0_em": mu0_em_from_logN(log10N_opt, p_opt),
        "e_plank": e_plank_em_from_logN(log10N_opt, p_opt),
        "alfa_em": alfa_em_em_from_logN(log10N_opt, p_opt)
    }

    print(f"{'Константа':<15} {'Модель':<20} {'Эксперимент':<20} {'Отношение':<12}")
    print("-" * 70)

    for k in model_all:
        ratio = model_all[k] / C_exp[k]
        print(f"{k:<15} {model_all[k]:<20.6e} {C_exp[k]:<20.6e} {ratio:<12.6f}")

    # Дополнительная информация
    print("\n" + "=" * 60)
    print("ДОПОЛНИТЕЛЬНАЯ ИНФОРМАЦИЯ")
    print("=" * 60)

    lam = lambda_param(log10N_opt, p_opt)
    print(f"λ = {lam:.6e}")
    print(f"K*p = {K * p_opt:.6e}")
    print(f"log(K*p) = {math.log(K * p_opt):.6f}")
    print(f"log(N) = {math.log(N_opt):.6f}")

    # Проверка дополнительных величин
    R_universe = 2 * math.pi / (math.sqrt(K * p_opt) * lam) * N_opt ** (1.0 / 6.0)
    l_em = 2 * math.pi / (K * p_opt * lam) * N_opt ** (1.0 / 6.0)
    hbar_em = hbar_em_from_logN(log10N_opt, p_opt)

    print(f"\nR_universe = {R_universe:.3e} m")
    print(f"l_em = {l_em:.3e} m")
    print(f"hbar_em (основная часть) = {hbar_em:.3e}")