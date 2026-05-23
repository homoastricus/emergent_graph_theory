"""
АНАЛИЗ RG-ИНВАРИАНТОВ И КРИТИЧЕСКОЙ ЛИНИИ
Проверка физичности модели через стабильность инвариантов
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# ЧАСТЬ 1: ВЫЧИСЛЕНИЕ ИНВАРИАНТОВ
# ============================================================================

def compute_invariants(p, N, K=8):
    """
    Вычисление RG-инвариантов для заданных параметров.

    Инварианты:
    - I1 = ln N / |ln(Kp)|        # Модулярный инвариант U
    - I2 = ln K - ln(Kp)/ln N      # Баланс локального и глобального
    - I3 = ln p / ln(Kp)           # Внутренняя структура p
    - I4 = (ln(Kp)/ln N)^2         # λ = x² (спектральная щель)
    - I5 = ln N + ln p             # Сумма масштабов
    - I6 = ln N / |ln p|           # Отношение масштабов
    """
    lnN = np.log(N)
    lnK = np.log(K)
    lnp = np.log(p)
    lnKp = np.log(K * p)

    I1 = lnN / abs(lnKp)  # U
    I2 = lnK - lnKp / lnN  # Баланс
    I3 = lnp / lnKp  # Структура p
    I4 = (lnKp / lnN) ** 2  # λ
    I5 = lnN + lnp  # Сумма логарифмов
    I6 = lnN / abs(lnp)  # Отношение

    # Дополнительные физические инварианты
    x = lnKp / lnN
    lambda_val = x ** 2
    U = lnN / abs(lnKp)
    f1 = U / np.pi
    hbar_em = (lnK) ** 2 / (4 * lambda_val ** 2 * K ** 2)  # Безразмерный ħ

    # RG-инвариант V (из эмерджентной квантовой гравитации)
    V = (f1 ** (2 / 3)) * (hbar_em ** 3) * (lnN ** 2)

    # Критическая линия: p ~ N^{-1/4} => p * N^{1/4} = const
    critical_parameter = p * N ** (1 / 4)

    return {
        'I1': I1, 'I2': I2, 'I3': I3,
        'I4': I4, 'I5': I5, 'I6': I6,
        'U': U, 'lambda': lambda_val, 'x': x,
        'V': V, 'f1': f1, 'hbar_em': hbar_em,
        'critical_param': critical_parameter
    }


def analyze_stability(base_p, base_N, base_K=8,
                      p_variations=None, N_variations=None):
    """
    Анализ стабильности инвариантов при вариации p и N.
    """
    if p_variations is None:
        p_variations = np.linspace(0.90, 1.10, 21)  # ±10% с мелким шагом
    if N_variations is None:
        N_variations = np.linspace(0.90, 1.10, 21)

    results = []

    print("АНАЛИЗ СТАБИЛЬНОСТИ RG-ИНВАРИАНТОВ")
    print("=" * 70)
    print(f"Базовые параметры:")
    print(f"  K = {base_K}")
    print(f"  p = {base_p:.6e}")
    print(f"  N = {base_N:.6e}")
    print(f"  ln N = {np.log(base_N):.6f}")
    print(f"  ln(Kp) = {np.log(base_K * base_p):.6f}")
    print(f"\nБазовые инварианты:")

    base_inv = compute_invariants(base_p, base_N, base_K)
    for key, val in base_inv.items():
        if not np.isnan(val) and not np.isinf(val):
            print(f"  {key:15} = {val:.8f}")

    print(f"\nВариации: p: {min(p_variations) * 100:.0f}% - {max(p_variations) * 100:.0f}%")
    print(f"         N: {min(N_variations) * 100:.0f}% - {max(N_variations) * 100:.0f}%")
    print(f"         всего комбинаций: {len(p_variations) * len(N_variations)}")

    # Собираем все значения
    all_invariants = {key: [] for key in base_inv.keys()}
    all_p = []
    all_N = []

    for p_factor in p_variations:
        for N_factor in N_variations:
            p = base_p * p_factor
            N = base_N * N_factor

            inv = compute_invariants(p, N, base_K)

            all_p.append(p)
            all_N.append(N)

            for key, val in inv.items():
                all_invariants[key].append(val)

            results.append({
                'p_factor': p_factor,
                'N_factor': N_factor,
                'p': p,
                'N': N,
                **inv
            })

    # Вычисляем статистики
    stats_results = {}
    print("\n" + "=" * 70)
    print("СТАТИСТИКА ИНВАРИАНТОВ")
    print("=" * 70)
    print(f"{'Инвариант':<18} {'Среднее':<15} {'σ':<12} {'σ/среднее':<12} {'Статус':<15}")
    print("-" * 70)

    for key, values in all_invariants.items():
        vals = np.array(values)
        vals = vals[np.isfinite(vals)]

        if len(vals) > 0:
            mean_val = np.mean(vals)
            std_val = np.std(vals)
            rel_std = std_val / abs(mean_val) if mean_val != 0 else np.inf

            # Определяем статус
            if rel_std < 0.001:
                status = "🔥 СИЛЬНЫЙ"
            elif rel_std < 0.01:
                status = "👍 ХОРОШИЙ"
            elif rel_std < 0.05:
                status = "⚠️ УМЕРЕННЫЙ"
            elif rel_std < 0.10:
                status = "🟡 СЛАБЫЙ"
            else:
                status = "❌ НЕТ"

            stats_results[key] = {
                'mean': mean_val,
                'std': std_val,
                'rel_std': rel_std,
                'status': status,
                'values': vals
            }

            print(f"{key:<18} {mean_val:<15.6f} {std_val:<12.6f} {rel_std:<12.6f} {status:<15}")

    return results, stats_results, all_p, all_N


# ============================================================================
# ЧАСТЬ 2: ПРОВЕРКА КРИТИЧЕСКОЙ ЛИНИИ p ~ N^{-1/4}
# ============================================================================

def check_critical_line(results):
    """
    Проверка гипотезы о критической линии p ~ N^{-1/d}.
    """
    print("\n" + "=" * 70)
    print("ПРОВЕРКА КРИТИЧЕСКОЙ ЛИНИИ p ~ N^{-1/d}")
    print("=" * 70)

    # Извлекаем p и N
    p_vals = np.array([r['p'] for r in results])
    N_vals = np.array([r['N'] for r in results])

    log_p = np.log(p_vals)
    log_N = np.log(N_vals)

    # Линейная регрессия
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_N, log_p)

    # Ожидаемый наклон для d=4
    expected_slope = -0.25

    print(f"\nРезультаты регрессии ln(p) vs ln(N):")
    print(f"  Наклон = {slope:.6f} ± {std_err:.6f}")
    print(f"  Ожидаемый наклон (d=4) = {expected_slope}")
    print(f"  Отклонение = {abs(slope - expected_slope):.6f}")
    print(f"  R² = {r_value ** 2:.6f}")
    print(f"  p-value = {p_value:.6e}")

    # Оценка размерности
    d_estimated = -1 / slope if slope != 0 else np.inf

    print(f"\n  Оценка размерности d = {d_estimated:.3f}")

    # Визуализация
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. График ln(p) vs ln(N)
    ax1 = axes[0]
    ax1.scatter(log_N, log_p, alpha=0.5, s=20, c='blue')
    ax1.plot(log_N, slope * log_N + intercept, 'r-', linewidth=2,
             label=f'Наклон = {slope:.4f} ± {std_err:.4f}')
    ax1.plot(log_N, expected_slope * log_N + intercept, 'g--', linewidth=2,
             label=f'Ожидаемый (d=4): -0.25')
    ax1.set_xlabel('ln(N)')
    ax1.set_ylabel('ln(p)')
    ax1.set_title('Проверка критической линии p ~ N^{-1/d}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. График p vs N в log-log
    ax2 = axes[1]
    ax2.loglog(N_vals, p_vals, 'o', alpha=0.5, markersize=3, label='Данные')

    # Теоретическая линия
    N_theory = np.logspace(np.log10(min(N_vals)), np.log10(max(N_vals)), 100)
    p_theory = np.exp(intercept) * N_theory ** slope
    ax2.loglog(N_theory, p_theory, 'r-', linewidth=2, label='Fit')

    p_expected = np.exp(intercept) * N_theory ** expected_slope
    ax2.loglog(N_theory, p_expected, 'g--', linewidth=2, label='d=4')

    ax2.set_xlabel('N')
    ax2.set_ylabel('p')
    ax2.set_title('p vs N (log-log)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('critical_line_analysis.png', dpi=150)
    plt.show()

    # Вывод
    print("\n" + "-" * 50)
    if abs(slope - expected_slope) < 3 * std_err:
        print("✅ ГИПОТЕЗА ПОДТВЕРЖДЕНА!")
        print(f"   Наклон {slope:.4f} совместим с d=4 в пределах 3σ.")
    elif abs(slope - expected_slope) < 0.1:
        print("⚠️ ЧАСТИЧНОЕ ПОДТВЕРЖДЕНИЕ")
        print(f"   Наклон {slope:.4f} близок к -0.25, но не совпадает точно.")
    else:
        print("❌ ГИПОТЕЗА НЕ ПОДТВЕРЖДЕНА")
        print(f"   Наклон {slope:.4f} значительно отличается от -0.25.")

    return {
        'slope': slope,
        'std_err': std_err,
        'expected': expected_slope,
        'r_squared': r_value ** 2,
        'p_value': p_value,
        'd_estimated': d_estimated
    }


# ============================================================================
# ЧАСТЬ 3: ВИЗУАЛИЗАЦИЯ СТАБИЛЬНОСТИ ИНВАРИАНТОВ
# ============================================================================

def plot_invariant_stability(stats_results, all_p, all_N):
    """
    Визуализация стабильности инвариантов.
    """
    # Выбираем инварианты с хорошей стабильностью
    good_invariants = [key for key, stat in stats_results.items()
                       if stat['rel_std'] < 0.05]

    if not good_invariants:
        good_invariants = list(stats_results.keys())[:6]

    n_inv = len(good_invariants)
    n_cols = min(3, n_inv)
    n_rows = (n_inv + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, key in enumerate(good_invariants):
        ax = axes[idx]
        stat = stats_results[key]
        values = stat['values']

        # Гистограмма
        ax.hist(values, bins=30, edgecolor='black', alpha=0.7, density=True)

        # Нормальное распределение для сравнения
        x = np.linspace(min(values), max(values), 100)
        y = stats.norm.pdf(x, stat['mean'], stat['std'])
        ax.plot(x, y, 'r-', linewidth=2, label=f"σ/μ = {stat['rel_std']:.4f}")

        ax.axvline(x=stat['mean'], color='green', linestyle='--', alpha=0.7)
        ax.set_xlabel(key)
        ax.set_ylabel('Плотность')
        ax.set_title(f"{key}: {stat['status']}")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # Убираем пустые subplots
    for idx in range(n_inv, len(axes)):
        axes[idx].remove()

    plt.tight_layout()
    plt.savefig('invariant_stability.png', dpi=150)
    plt.show()


def plot_invariant_correlations(stats_results):
    """
    Корреляции между инвариантами.
    """
    # Выбираем ключевые инварианты
    key_invariants = ['I1', 'I2', 'I4', 'U', 'lambda', 'V']
    available = [k for k in key_invariants if k in stats_results]

    if len(available) < 2:
        return

    n = len(available)
    fig, axes = plt.subplots(n, n, figsize=(12, 12))

    for i, key1 in enumerate(available):
        for j, key2 in enumerate(available):
            ax = axes[i, j]
            vals1 = stats_results[key1]['values']
            vals2 = stats_results[key2]['values']

            if i == j:
                # Диагональ — гистограмма
                ax.hist(vals1, bins=20, edgecolor='black', alpha=0.7)
                ax.set_xlabel(key1)
            else:
                # Scatter plot
                ax.scatter(vals1, vals2, alpha=0.3, s=10)

                # Корреляция
                corr = np.corrcoef(vals1, vals2)[0, 1]
                ax.text(0.05, 0.95, f'r = {corr:.3f}',
                        transform=ax.transAxes, fontsize=9,
                        verticalalignment='top')

            if i == n - 1:
                ax.set_xlabel(key2)
            if j == 0:
                ax.set_ylabel(key1)

            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('invariant_correlations.png', dpi=150)
    plt.show()


# ============================================================================
# ЧАСТЬ 4: ФИНАЛЬНЫЙ ВЫВОД
# ============================================================================

def final_conclusion(stats_results, critical_line_result):
    """
    Формулировка финального вывода о физичности модели.
    """
    print("\n" + "=" * 70)
    print("ФИНАЛЬНЫЙ ВЫВОД О ФИЗИЧНОСТИ МОДЕЛИ")
    print("=" * 70)

    # Критерии физичности
    strong_invariants = sum(1 for stat in stats_results.values()
                            if stat['rel_std'] < 0.01)
    good_invariants = sum(1 for stat in stats_results.values()
                          if stat['rel_std'] < 0.05)
    total_invariants = len(stats_results)

    critical_confirmed = (abs(critical_line_result['slope'] - critical_line_result['expected'])
                          < 3 * critical_line_result['std_err'])

    print(f"\nКРИТЕРИИ:")
    print(f"  Сильных инвариантов (σ/μ < 1%):    {strong_invariants}/{total_invariants}")
    print(f"  Хороших инвариантов (σ/μ < 5%):   {good_invariants}/{total_invariants}")
    print(f"  Критическая линия подтверждена:    {'✅ Да' if critical_confirmed else '❌ Нет'}")

    print("\n" + "-" * 50)

    if strong_invariants >= 2 and critical_confirmed:
        print("""
╔══════════════════════════════════════════════════════════════════════════╗
║                    🎉 МОДЕЛЬ ФИЗИЧЕСКАЯ!                                  ║
╚══════════════════════════════════════════════════════════════════════════╝

ОБНАРУЖЕНО:
  • Существуют стабильные RG-инварианты (σ/μ < 1%).
  • Подтверждена критическая линия p ~ N^{-1/4} (d=4).
  • Параметры не случайны — они связаны фундаментальной симметрией.

ИНТЕРПРЕТАЦИЯ:
  • Модель не является подгонкой.
  • Существует скрытое уравнение F(p, N, K) = 0.
  • Это кандидат на фундаментальную теорию.
""")
    elif good_invariants >= 2:
        print("""
╔══════════════════════════════════════════════════════════════════════════╗
║                  ⚠️ МОДЕЛЬ ЧАСТИЧНО ФИЗИЧЕСКАЯ                           ║
╚══════════════════════════════════════════════════════════════════════════╝

ОБНАРУЖЕНО:
  • Существуют умеренно стабильные инварианты (σ/μ < 5%).
  • Есть признаки структуры, но требуется уточнение.

РЕКОМЕНДАЦИИ:
  • Проверить формулы на большем диапазоне параметров.
  • Уточнить функционал для поиска fixed points.
  • Добавить физические ограничения (d_s ≈ 4, λ ≈ α).
""")
    else:
        print("""
╔══════════════════════════════════════════════════════════════════════════╗
║                     ❌ МОДЕЛЬ НЕСТАБИЛЬНА                                ║
╚══════════════════════════════════════════════════════════════════════════╝

ОБНАРУЖЕНО:
  • Инварианты сильно варьируют при малых изменениях параметров.
  • Критическая линия не подтверждается.

ВОЗМОЖНЫЕ ПРИЧИНЫ:
  • Текущие параметры (p, N) не являются выделенными.
  • Требуется другой функционал или дополнительные ограничения.
  • Модель в текущем виде — подгонка.
""")


# ============================================================================
# ЧАСТЬ 5: ЗАПУСК
# ============================================================================

def main():
    print("=" * 70)
    print("АНАЛИЗ RG-ИНВАРИАНТОВ И КРИТИЧЕСКОЙ ЛИНИИ")
    print("Проверка физичности модели через стабильность инвариантов")
    print("=" * 70)

    # Базовые параметры (критические)
    base_K = 8
    base_p = 1.25e-31
    base_N = 9.702e+122

    # Вариации для анализа стабильности
    p_variations = np.linspace(0.90, 1.10, 15)  # ±10%, 15 точек
    N_variations = np.linspace(0.90, 1.10, 15)

    # 1. Анализ стабильности инвариантов
    results, stats_results, all_p, all_N = analyze_stability(
        base_p, base_N, base_K, p_variations, N_variations
    )

    # 2. Проверка критической линии
    critical_result = check_critical_line(results)

    # 3. Визуализация
    plot_invariant_stability(stats_results, all_p, all_N)
    plot_invariant_correlations(stats_results)

    # 4. Финальный вывод
    final_conclusion(stats_results, critical_result)

    # 5. Сохранение результатов
    import json
    summary = {
        'base_parameters': {'K': base_K, 'p': base_p, 'N': base_N},
        'invariants': {k: {'mean': float(v['mean']), 'std': float(v['std']),
                           'rel_std': float(v['rel_std']), 'status': v['status']}
                       for k, v in stats_results.items()},
        'critical_line': {k: float(v) if not isinstance(v, bool) else v
                          for k, v in critical_result.items()}
    }

    with open('rg_invariants_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print("\n✅ Результаты сохранены в rg_invariants_results.json")

    return results, stats_results, critical_result


if __name__ == "__main__":
    results, stats, crit = main()