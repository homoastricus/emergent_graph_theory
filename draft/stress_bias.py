"""
РАСЧЁТ КОНСТАНТЫ СДВИГА c₀ МЕЖДУ N_math И N_phys

Модель: ln N_phys = ln N_math - c₀/ln N_math - c₁/(ln N_math)²

Анализ для трёх наборов констант:
1. Все 59 констант
2. 9 времён жизни
3. 29 самых точных констант
"""

import math
import numpy as np
from scipy.optimize import minimize_scalar, fsolve

# =========================
# ДАННЫЕ ЭКСПЕРИМЕНТОВ
# =========================

lnN_math = 280.0497096834  # из математических тождеств (LOO, все 35 тождеств)

experiments = {
    "59 констант (все)": {
        "N_fit_ratio": 0.9980,  # N_fit / N_math
        "description": "Полный набор физических констант"
    },
    "9 времён жизни": {
        "N_fit_ratio": 0.9744,  # N_fit / N_math
        "description": "Только времена жизни частиц"
    },
    "29 точных констант": {
        "N_fit_ratio": 0.9950,  # N_fit / N_math
        "description": "Самые точно измеренные константы"
    },
    "14 тождеств (старый тест)": {
        "N_fit_ratio": 1.0000,  # N_fit / N_math = 1 (базовый тест)
        "description": "Оригинальные 14 тождеств (N_math определён из них же)"
    },
}


# =========================
# МОДЕЛЬ СДВИГА
# =========================

def compute_delta_lnN(N_fit_ratio):
    """Вычисляет Δln N = ln(N_phys) - ln(N_math)"""
    return math.log(N_fit_ratio)


def compute_c0_first_order(delta_lnN, lnN):
    """c₀ = Δln N × ln N (первый порядок)"""
    return delta_lnN * lnN


def compute_c0_with_second_order(delta_lnN, lnN, c1_guess=0):
    """
    Решает уравнение:
    Δln N = -c₀/ln N - c₁/(ln N)²

    Возвращает (c₀, c₁)
    """
    # Если c₁ известна, то c₀ = -(Δln N + c₁/lnN²) × lnN
    if c1_guess != 0:
        c0 = -(delta_lnN + c1_guess / lnN ** 2) * lnN
        return c0, c1_guess
    else:
        # Только первый порядок
        c0 = -delta_lnN * lnN
        return c0, 0.0


def predict_Nphys_from_model(lnN_math_val, c0, c1=0):
    """Предсказывает N_phys по модели"""
    correction = c0 / lnN_math_val + c1 / lnN_math_val ** 2
    lnN_phys = lnN_math_val - correction
    N_phys = math.exp(lnN_phys)
    ratio = N_phys / math.exp(lnN_math_val)
    return lnN_phys, N_phys, ratio


# =========================
# ПОИСК ОПТИМАЛЬНОГО c₀
# =========================

def find_optimal_c0():
    """
    Находит c₀, которое минимизирует дисперсию между экспериментами
    """
    print("\n" + "=" * 80)
    print("ПОИСК ОПТИМАЛЬНОГО c₀")
    print("=" * 80)

    ratios = [exp["N_fit_ratio"] for exp in experiments.values()]
    lnN_values = [lnN_math] * len(ratios)

    def variance_of_c0(c0):
        c0_values = []
        for ratio in ratios:
            delta = math.log(ratio)
            # Из delta = -c0/lnN → c0 = -delta × lnN
            c0_values.append(-delta * lnN_math)
        return np.var(c0_values)

    result = minimize_scalar(variance_of_c0, bounds=(0.1, 10.0), method='bounded')
    return result.x, result.fun


# =========================
# КАНДИДАТЫ НА c₀
# =========================

CANDIDATES = {
    "√2": math.sqrt(2),
    "π/2": math.pi / 2,
    "e/2": math.e / 2,
    "ln(4)": math.log(4),
    "γ (Euler-Mascheroni)": 0.5772156649015329,
    "1/√3": 1.0 / math.sqrt(3),
    "ln(K)/π": math.log(6) / math.pi,
    "√π": math.sqrt(math.pi),
    "π/√2": math.pi / math.sqrt(2),
    "√(π/2)": math.sqrt(math.pi / 2),
    "ln(K)": math.log(6),
    "2/π": 2.0 / math.pi,
}


# =========================
# ГЛАВНЫЙ ЗАПУСК
# =========================

def main():
    print("=" * 80)
    print("РАСЧЁТ КОНСТАНТЫ СДВИГА c₀ МЕЖДУ N_math И N_phys")
    print("=" * 80)

    print(f"\nln N_math = {lnN_math:.10f}")
    print(f"N_math = {math.exp(lnN_math):.6e}")

    # ===== ТАБЛИЦА ДЛЯ ВСЕХ ЭКСПЕРИМЕНТОВ =====
    print("\n" + "=" * 80)
    print("РЕЗУЛЬТАТЫ ПО ЭКСПЕРИМЕНТАМ")
    print("=" * 80)

    print(f"\n{'Эксперимент':<30} {'N_fit/N_math':<15} {'Δln N':<15} {'c₀ (1-й порядок)':<18} {'ln N_phys':<15}")
    print("-" * 95)

    c0_values = []

    for name, exp in experiments.items():
        ratio = exp["N_fit_ratio"]
        delta = compute_delta_lnN(ratio)
        c0 = compute_c0_first_order(delta, lnN_math)

        if ratio < 1.0:  # не базовый тест
            c0_values.append(c0)

        lnN_phys_pred = lnN_math - c0 / lnN_math
        N_phys_pred = math.exp(lnN_phys_pred)

        print(f"{name:<30} {ratio:<15.6f} {delta:<15.6f} {c0:<18.6f} {lnN_phys_pred:<15.6f}")

    # ===== СТАТИСТИКА =====
    if c0_values:
        print(f"\n  Среднее c₀ (без базового теста): {np.mean(c0_values):.6f}")
        print(f"  Стандартное отклонение c₀: {np.std(c0_values):.6f}")
        print(f"  Медиана c₀: {np.median(c0_values):.6f}")

    # ===== СРАВНЕНИЕ С КАНДИДАТАМИ =====
    print("\n" + "=" * 80)
    print("СРАВНЕНИЕ С МАТЕМАТИЧЕСКИМИ КОНСТАНТАМИ")
    print("=" * 80)

    # Для точных констант
    ratio_accurate = experiments["29 точных констант"]["N_fit_ratio"]
    delta_accurate = compute_delta_lnN(ratio_accurate)
    c0_accurate = compute_c0_first_order(delta_accurate, lnN_math)

    print(f"\n  Эталонное c₀ (29 точных констант): {c0_accurate:.6f}")
    print(f"\n  {'Константа':<25} {'Значение':<15} {'Отклонение от c₀':<15} {'Ошибка %':<12}")
    print(f"  {'-' * 70}")

    best_candidate = None
    best_error = float('inf')

    for name, value in CANDIDATES.items():
        deviation = abs(value - c0_accurate)
        error_pct = deviation / c0_accurate * 100

        if error_pct < best_error:
            best_error = error_pct
            best_candidate = (name, value)

        marker = "⭐" if error_pct < 0.5 else ("✅" if error_pct < 2.0 else ("🟡" if error_pct < 5.0 else "❌"))
        print(f"  {marker} {name:<23} {value:<15.6f} {deviation:<15.6f} {error_pct:<12.4f}")

    # ===== ЛУЧШИЙ КАНДИДАТ =====
    print("\n" + "=" * 80)
    print("ЛУЧШИЙ КАНДИДАТ")
    print("=" * 80)

    if best_candidate:
        name, value = best_candidate
        print(f"\n  c₀ ≈ {name} = {value:.10f}")
        print(f"  Ошибка: {best_error:.6f}%")

        # Предсказание с лучшим кандидатом
        lnN_phys_best = lnN_math - value / lnN_math
        N_phys_best = math.exp(lnN_phys_best)
        ratio_best = N_phys_best / math.exp(lnN_math)

        print(f"\n  Предсказание с c₀ = {name}:")
        print(f"    ln N_phys = {lnN_phys_best:.6f}")
        print(f"    N_phys = {N_phys_best:.6e}")
        print(f"    N_phys / N_math = {ratio_best:.6f}")
        print(f"    Эксперимент (29 точных): {ratio_accurate:.6f}")
        print(f"    Ошибка предсказания: {abs(ratio_best - ratio_accurate) / ratio_accurate * 100:.6f}%")

    # ===== ПРЕДСКАЗАНИЕ ДЛЯ ВСЕХ НАБОРОВ =====
    print("\n" + "=" * 80)
    print("ПРЕДСКАЗАНИЯ МОДЕЛИ ДЛЯ ВСЕХ НАБОРОВ")
    print("=" * 80)

    # Используем среднее c₀ от трёх экспериментов
    c0_mean = np.mean(c0_values) if c0_values else 1.404

    print(f"\n  Используем c₀ = {c0_mean:.6f} (среднее по экспериментам)")
    print(f"\n  {'Эксперимент':<30} {'N_fit/N_math (эксп)':<20} {'N_fit/N_math (модель)':<20} {'Ошибка %':<12}")
    print(f"  {'-' * 80}")

    for name, exp in experiments.items():
        if exp["N_fit_ratio"] < 1.0:
            delta_exp = compute_delta_lnN(exp["N_fit_ratio"])
            lnN_phys_model = lnN_math - c0_mean / lnN_math
            ratio_model = math.exp(lnN_phys_model) / math.exp(lnN_math)
            error = abs(ratio_model - exp["N_fit_ratio"]) / exp["N_fit_ratio"] * 100
            print(f"  {name:<30} {exp['N_fit_ratio']:<20.6f} {ratio_model:<20.6f} {error:<12.6f}")

    # ===== ФИНАЛЬНАЯ ФОРМУЛА =====
    print("\n" + "=" * 80)
    print("ФИНАЛЬНАЯ ФОРМУЛА СДВИГА")
    print("=" * 80)

    print(f"""
    Модель первого порядка:

    ln N_phys = ln N_math - c₀ / ln N_math

    где c₀ ≈ {c0_mean:.6f} (среднее по экспериментам)

    Для 29 самых точных констант:
    c₀ ≈ {c0_accurate:.6f}
    Лучший кандидат: {best_candidate[0] if best_candidate else 'не определён'} = {best_candidate[1] if best_candidate else 0:.6f}

    N_phys / N_math = exp(-c₀ / ln N_math) ≈ {math.exp(-c0_accurate / lnN_math):.6f}

    При N → ∞: N_phys / N_math → 1 (сдвиг исчезает)
    """)


if __name__ == "__main__":
    main()