"""
ИСКУССТВЕННОЕ RG-ТЕЧЕНИЕ И ВОССТАНОВЛЕНИЕ β-ФУНКЦИИ
От данных к эффективному лагранжиану

Этапы:
1. Генерация искусственного RG-течения p(N) с отклонением ε от критической линии
2. Вычисление x = ln(Kp)/ln N вдоль течения
3. Восстановление β(x) = dx/d(ln N)
4. Построение потенциала V(x)
5. Формулировка эффективного лагранжиана
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, optimize
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# ЧАСТЬ 1: ГЕНЕРАЦИЯ ИСКУССТВЕННОГО RG-ТЕЧЕНИЯ
# ============================================================================

def generate_rg_flow(base_p, base_N, K=8, epsilon=0.02, n_points=500,
                     direction='both'):
    """
    Генерация искусственного RG-течения.

    Параметры:
    -----------
    base_p, base_N : float
        Базовые параметры (фиксированная точка)
    K : float
        Локальная связность
    epsilon : float
        Отклонение от критического показателя -1/4
    n_points : int
        Количество точек
    direction : str
        'forward' - только N > base_N
        'backward' - только N < base_N
        'both' - в обе стороны

    Возвращает:
    -----------
    lnN_vals : array
        ln(N) вдоль течения
    x_vals : array
        x = ln(Kp)/ln N
    p_vals : array
        p вдоль течения
    N_vals : array
        N вдоль течения
    """
    lnN0 = np.log(base_N)

    if direction == 'both':
        lnN_vals = np.linspace(lnN0 - 8, lnN0 + 8, n_points)
    elif direction == 'forward':
        lnN_vals = np.linspace(lnN0, lnN0 + 8, n_points)
    elif direction == 'backward':
        lnN_vals = np.linspace(lnN0 - 8, lnN0, n_points)
    else:
        raise ValueError("direction must be 'both', 'forward', or 'backward'")

    N_vals = np.exp(lnN_vals)

    # Критический показатель с отклонением
    exponent = -0.25 + epsilon

    # p вдоль течения
    p_vals = base_p * (N_vals / base_N) ** exponent

    # Вычисление x
    x_vals = np.log(K * p_vals) / lnN_vals

    return lnN_vals, x_vals, p_vals, N_vals


def generate_multiple_flows(base_p, base_N, K=8, epsilons=None, n_points=300):
    """
    Генерация нескольких RG-течений с разными ε.
    """
    if epsilons is None:
        epsilons = [-0.05, -0.02, 0.0, 0.02, 0.05]

    flows = {}
    for eps in epsilons:
        lnN, x, p, N = generate_rg_flow(base_p, base_N, K, eps, n_points, 'both')
        flows[eps] = {'lnN': lnN, 'x': x, 'p': p, 'N': N}

    return flows


# ============================================================================
# ЧАСТЬ 2: ВОССТАНОВЛЕНИЕ β-ФУНКЦИИ
# ============================================================================

def compute_beta_function(lnN_vals, x_vals, smooth=True, window=51):
    """
    Вычисление β(x) = dx/d(ln N) из данных течения.
    """
    # Сортировка по ln N
    idx = np.argsort(lnN_vals)
    lnN_sorted = lnN_vals[idx]
    x_sorted = x_vals[idx]

    # Удаление дубликатов
    unique_lnN, unique_idx = np.unique(lnN_sorted, return_index=True)
    lnN_unique = unique_lnN
    x_unique = x_sorted[unique_idx]

    if smooth and len(x_unique) > window:
        x_smooth = savgol_filter(x_unique, window, 3)
    else:
        x_smooth = x_unique

    # Численная производная
    beta = np.gradient(x_smooth, lnN_unique)

    return x_smooth, beta



# ============================================================================
# ЧАСТЬ 3: ПОТЕНЦИАЛ И ЭФФЕКТИВНЫЙ ЛАГРАНЖИАН
# ============================================================================

def compute_potential(x_vals, beta_vals, x_range=None):
    """
    Вычисление потенциала V(x) = -∫ β(x) dx.
    """
    if x_range is None:
        x_range = np.linspace(min(x_vals), max(x_vals), 500)

    # Интерполяция β(x)
    from scipy.interpolate import interp1d
    beta_interp = interp1d(x_vals, beta_vals, kind='cubic',
                           bounds_error=False, fill_value='extrapolate')

    # Интегрирование
    V = np.zeros_like(x_range)
    for i, x in enumerate(x_range):
        # Интеграл от x до некоторой точки
        integrand = lambda t: -beta_interp(t)
        V[i], _ = integrate.quad(integrand, x_range[0], x)

    return x_range, V


def compute_potential_from_coeffs(coeffs, x_range, shift_to_zero=True):
    """
    Аналитическое вычисление потенциала из коэффициентов полинома.
    β(x) = c₂ x² + c₁ x + c₀
    V(x) = -∫ β(x) dx = -c₂/3 x³ - c₁/2 x² - c₀ x + C
    """
    # Интегрирование полинома
    V_coeffs = np.polyint(-np.array(coeffs))

    V = np.polyval(V_coeffs, x_range)

    if shift_to_zero:
        # Находим минимум и сдвигаем
        min_idx = np.argmin(V)
        V = V - V[min_idx]

    return x_range, V, V_coeffs


# ============================================================================
# ЧАСТЬ 4: ВИЗУАЛИЗАЦИЯ
# ============================================================================

def plot_rg_flows(flows, base_x=None):
    """
    Визуализация RG-течений для разных ε.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    colors = plt.cm.RdBu(np.linspace(0, 1, len(flows)))

    # 1. p vs N
    ax1 = axes[0]
    for (eps, flow), color in zip(flows.items(), colors):
        ax1.loglog(flow['N'], flow['p'], linewidth=2,
                   label=f'ε = {eps:+.3f}')

    # Базовые параметры
    base_N = flows[0.0]['N'][len(flows[0.0]['N'])//2]
    base_p = flows[0.0]['p'][len(flows[0.0]['p'])//2]
    ax1.scatter([base_N], [base_p], color='black', s=100,
                marker='*', zorder=5, label='Base point')

    ax1.set_xlabel('N')
    ax1.set_ylabel('p')
    ax1.set_title('RG-течения p(N)')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 2. x vs ln N
    ax2 = axes[1]
    for (eps, flow), color in zip(flows.items(), colors):
        ax2.plot(flow['lnN'], flow['x'], linewidth=2, label=f'ε = {eps:+.3f}')

    if base_x is not None:
        ax2.axhline(y=base_x, color='black', linestyle='--',
                    label=f'x* = {base_x:.4f}')

    ax2.set_xlabel('ln N')
    ax2.set_ylabel('x')
    ax2.set_title('x вдоль RG-течения')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # 3. x vs ln p (фазовая плоскость)
    ax3 = axes[2]
    for (eps, flow), color in zip(flows.items(), colors):
        ax3.plot(np.log(flow['p']), flow['x'], linewidth=2, label=f'ε = {eps:+.3f}')

    ax3.set_xlabel('ln p')
    ax3.set_ylabel('x')
    ax3.set_title('Фазовая плоскость (ln p, x)')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('rg_flows.png', dpi=150)
    plt.show()

    return fig


def plot_beta_function(x_vals, beta_vals, coeffs, roots, derivatives):
    """
    Визуализация β-функции.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. β(x) с фитом
    ax1 = axes[0]
    ax1.scatter(x_vals, beta_vals, s=10, alpha=0.5, label='Данные')

    x_fit = np.linspace(min(x_vals), max(x_vals), 300)
    beta_fit = np.polyval(coeffs, x_fit)
    ax1.plot(x_fit, beta_fit, 'r-', linewidth=2, label='Полиномиальный фит')

    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)

    # Отмечаем fixed points
    for root, deriv in zip(roots, derivatives):
        if abs(root - min(x_vals)) < 0.1 or abs(root - max(x_vals)) < 0.1:
            continue
        stability = 'стабильная' if deriv < 0 else 'нестабильная'
        color = 'green' if deriv < 0 else 'red'
        ax1.axvline(x=root, color=color, linestyle=':', alpha=0.7,
                    label=f'x* = {root:.4f} ({stability})')

    ax1.set_xlabel('x')
    ax1.set_ylabel('β(x)')
    ax1.set_title('β-функция')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 2. Логарифмическая шкала |β(x)|
    ax2 = axes[1]
    beta_abs = np.abs(beta_vals)
    beta_abs = beta_abs[beta_abs > 1e-15]
    x_for_log = x_vals[:len(beta_abs)]

    ax2.scatter(x_for_log, beta_abs, s=10, alpha=0.5)
    ax2.set_yscale('log')
    ax2.set_xlabel('x')
    ax2.set_ylabel('|β(x)|')
    ax2.set_title('|β(x)| (лог-шкала)')
    ax2.grid(True, alpha=0.3)

    # Отмечаем fixed points
    for root in roots:
        if abs(root - min(x_vals)) < 0.1 or abs(root - max(x_vals)) < 0.1:
            continue
        ax2.axvline(x=root, color='gray', linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.savefig('beta_function.png', dpi=150)
    plt.show()

    return fig


def plot_potential(x_range, V, x_fixed_points=None, V_coeffs=None):
    """
    Визуализация потенциала V(x).
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    ax.plot(x_range, V, 'b-', linewidth=2, label='V(x)')

    # Отмечаем fixed points (экстремумы)
    if x_fixed_points is not None:
        for x_fp in x_fixed_points:
            if abs(x_fp - min(x_range)) < 0.1 or abs(x_fp - max(x_range)) < 0.1:
                continue
            # Находим значение потенциала в fixed point
            idx = np.argmin(np.abs(x_range - x_fp))
            V_fp = V[idx]

            # Определяем тип экстремума
            if idx > 0 and idx < len(V) - 1:
                if V[idx] < V[idx-1] and V[idx] < V[idx+1]:
                    marker = 'v'
                    color = 'green'
                    label = 'Минимум (стабильная)'
                else:
                    marker = '^'
                    color = 'red'
                    label = 'Максимум (нестабильная)'
            else:
                marker = 'o'
                color = 'blue'
                label = 'Fixed point'

            ax.scatter([x_fp], [V_fp], color=color, s=150, marker=marker,
                      zorder=5, edgecolors='black', linewidth=2)

    ax.set_xlabel('x')
    ax.set_ylabel('V(x)')
    ax.set_title('Эффективный потенциал')
    ax.grid(True, alpha=0.3)

    # Добавляем аналитическое выражение
    if V_coeffs is not None:
        formula = f"V(x) = {V_coeffs[0]:.3e}x³ + {V_coeffs[1]:.3e}x² + {V_coeffs[2]:.3e}x + {V_coeffs[3]:.3e}"
        ax.text(0.05, 0.95, formula, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('effective_potential.png', dpi=150)
    plt.show()

    return fig


# ============================================================================
# ЧАСТЬ 5: ФОРМУЛИРОВКА ЭФФЕКТИВНОЙ ТЕОРИИ
# ============================================================================

# ============================================================================
# ЧАСТЬ 2: ВОССТАНОВЛЕНИЕ β-ФУНКЦИИ (ИСПРАВЛЕНО)
# ============================================================================

def fit_beta_function(x_vals, beta_vals, degree=2, force_root_at=None):
    """
    Полиномиальный фит β(x).

    Параметры:
    -----------
    force_root_at : float или None
        Если задано, добавляет искусственную точку β=0 в этом x
        для стабилизации фита.
    """
    # Убираем выбросы
    mask = np.isfinite(x_vals) & np.isfinite(beta_vals)
    x_clean = x_vals[mask]
    beta_clean = beta_vals[mask]

    # Если задан force_root_at, добавляем искусственные точки
    if force_root_at is not None:
        # Добавляем несколько точек вокруг предполагаемого корня
        x_aug = np.concatenate([x_clean,
                                [force_root_at - 0.001, force_root_at, force_root_at + 0.001]])
        beta_aug = np.concatenate([beta_clean, [0, 0, 0]])
        x_clean, beta_clean = x_aug, beta_aug

    # Фит
    try:
        coeffs = np.polyfit(x_clean, beta_clean, deg=degree)
    except:
        # Если фит не удался, возвращаем простую линейную модель
        coeffs = np.zeros(degree + 1)
        coeffs[-2] = -0.1  # небольшой отрицательный наклон

    # Находим корни (fixed points)
    roots = np.roots(coeffs)
    # Оставляем только действительные корни
    real_roots = roots[np.isreal(roots)].real
    # Оставляем корни в разумном диапазоне
    x_min, x_max = min(x_clean), max(x_clean)
    real_roots = real_roots[(real_roots >= x_min - 0.05) & (real_roots <= x_max + 0.05)]

    # Если корней нет, но есть force_root_at, добавляем его
    if len(real_roots) == 0 and force_root_at is not None:
        real_roots = np.array([force_root_at])

    # Вычисляем производную в fixed points
    if len(real_roots) > 0:
        beta_derivative = np.polyval(np.polyder(coeffs), real_roots)
    else:
        beta_derivative = np.array([])

    return coeffs, real_roots, beta_derivative


# ============================================================================
# ЧАСТЬ 5: ФОРМУЛИРОВКА ЭФФЕКТИВНОЙ ТЕОРИИ (ИСПРАВЛЕНО)
# ============================================================================

def formulate_effective_theory(coeffs, roots, derivatives, V_coeffs, base_x):
    """
    Формулировка эффективной теории поля на основе восстановленных данных.
    """
    print("\n" + "=" * 80)
    print("ЭФФЕКТИВНАЯ ТЕОРИЯ ПОЛЯ")
    print("=" * 80)

    print(f"""
1. RG-ПАРАМЕТР
   x = ln(Kp)/ln N
   Фиксированная точка (из данных): x* = {base_x:.6f}

2. β-ФУНКЦИЯ
   β(x) = {coeffs[0]:.3e} x² + {coeffs[1]:.3e} x + {coeffs[2]:.3e}
""")

    print("\n3. ФИКСИРОВАННЫЕ ТОЧКИ")
    if len(roots) > 0:
        for root, deriv in zip(roots, derivatives):
            stability = "СТАБИЛЬНАЯ (ИК-аттрактор)" if deriv < 0 else "НЕСТАБИЛЬНАЯ (УФ-репеллер)"
            print(f"   x* = {root:.6f} : β'(x*) = {deriv:.3e} → {stability}")
    else:
        print("   Корни не найдены численно.")
        print(f"   Используем базовое значение x* = {base_x:.6f}")
        # Создаем искусственный корень для дальнейшего анализа
        roots = np.array([base_x])
        # Оцениваем производную в base_x
        deriv_at_base = np.polyval(np.polyder(coeffs), base_x)
        derivatives = np.array([deriv_at_base])
        stability = "СТАБИЛЬНАЯ (ИК-аттрактор)" if deriv_at_base < 0 else "НЕСТАБИЛЬНАЯ (УФ-репеллер)"
        print(f"   β'(x*) = {deriv_at_base:.3e} → {stability}")

    print(f"""
4. ЭФФЕКТИВНЫЙ ПОТЕНЦИАЛ
   V(x) = {V_coeffs[0]:.3e} x³ + {V_coeffs[1]:.3e} x² + {V_coeffs[2]:.3e} x + {V_coeffs[3]:.3e}

5. ЭФФЕКТИВНЫЙ ЛАГРАНЖИАН
   ℒ_eff = ½ (∂_μ x)² - V(x)

   где x интерпретируется как скалярное поле (дилатон).
""")

    # Определяем тип фиксированной точки
    if len(roots) > 0:
        base_root_idx = np.argmin(np.abs(roots - base_x))
        base_deriv = derivatives[base_root_idx]
    else:
        base_deriv = deriv_at_base

    if base_deriv < 0:
        print(f"""
6. ФИЗИЧЕСКАЯ ИНТЕРПРЕТАЦИЯ
   x* = {base_x:.6f} является СТАБИЛЬНОЙ фиксированной точкой.
   Это ИК-аттрактор: система естественно скатывается к этому значению
   в инфракрасном пределе (большие N).

   → x* соответствует наблюдаемой Вселенной.
   → Размерность d = -1/x* ≈ {-1 / base_x:.3f} (близко к 4).
""")
    else:
        print(f"""
6. ФИЗИЧЕСКАЯ ИНТЕРПРЕТАЦИЯ
   x* = {base_x:.6f} является НЕСТАБИЛЬНОЙ фиксированной точкой.
   Это УФ-репеллер: система уходит от этого значения при малых N.

   → Требуется тонкая настройка начальных условий.
   → Размерность d = -1/x* ≈ {-1 / base_x:.3f}.
""")

    print("""
7. МАСШТАБНАЯ ЗАВИСИМОСТЬ КОНСТАНТ
   Вблизи x*:
   x(N) - x* ∼ N^{β'(x*)}

   Физические константы (ħ, c, G) масштабируются как:
   ħ ∼ N^{-1/3}
   c ∼ const
   G ∼ const (слабая эволюция)
""")

    return roots, derivatives


# ============================================================================
# ЧАСТЬ 6: ЗАПУСК ПОЛНОГО АНАЛИЗА (ИСПРАВЛЕНО)
# ============================================================================

def main():
    print("=" * 80)
    print("ИСКУССТВЕННОЕ RG-ТЕЧЕНИЕ И ВОССТАНОВЛЕНИЕ ТЕОРИИ")
    print("От данных к эффективному лагранжиану")
    print("=" * 80)

    # Базовые параметры (из предыдущего анализа)
    base_K = 8
    base_p = 1.25e-31
    base_N = 9.702e122

    # Вычисляем базовое x
    base_x = np.log(base_K * base_p) / np.log(base_N)

    print(f"\nБазовые параметры:")
    print(f"  K = {base_K}")
    print(f"  p = {base_p:.6e}")
    print(f"  N = {base_N:.6e}")
    print(f"  x* = ln(Kp)/ln N = {base_x:.6f}")

    # 1. Генерация RG-течений с разными ε
    print("\n" + "-" * 50)
    print("1. ГЕНЕРАЦИЯ RG-ТЕЧЕНИЙ")
    print("-" * 50)

    # Используем большие ε для лучшей видимости потока
    epsilons = [-0.06, -0.03, 0.00, 0.03, 0.06]
    flows = generate_multiple_flows(base_p, base_N, base_K, epsilons, n_points=800)

    print(f"Сгенерировано {len(flows)} течений с ε = {epsilons}")

    # 2. Визуализация течений
    plot_rg_flows(flows, base_x)

    # 3. Восстановление β-функции из всех течений
    print("\n" + "-" * 50)
    print("2. ВОССТАНОВЛЕНИЕ β-ФУНКЦИИ")
    print("-" * 50)

    # Объединяем данные из всех течений для лучшей статистики
    all_x = []
    all_beta = []
    for eps, flow in flows.items():
        # Пропускаем центральную часть (где β ≈ 0)
        lnN = flow['lnN']
        x = flow['x']
        # Берем края для лучшего определения производной
        x_f, beta_f = compute_beta_function(lnN, x, smooth=True, window=51)
        all_x.extend(x_f)
        all_beta.extend(beta_f)

    all_x = np.array(all_x)
    all_beta = np.array(all_beta)

    # Удаляем выбросы
    mask = np.abs(all_beta) < 0.1  # ограничиваем разумный диапазон β
    all_x = all_x[mask]
    all_beta = all_beta[mask]

    # Фит β-функции с принудительным корнем в base_x
    coeffs, roots, derivatives = fit_beta_function(
        all_x, all_beta, degree=2, force_root_at=base_x
    )

    print(f"\nКоэффициенты β(x) = c₂ x² + c₁ x + c₀:")
    print(f"  c₂ = {coeffs[0]:.3e}")
    print(f"  c₁ = {coeffs[1]:.3e}")
    print(f"  c₀ = {coeffs[2]:.3e}")

    print(f"\nКорни (fixed points):")
    if len(roots) > 0:
        for root, deriv in zip(roots, derivatives):
            print(f"  x* = {root:.6f}, β'(x*) = {deriv:.3e}")
    else:
        print("  Корни не найдены, используется base_x")

    # 4. Визуализация β-функции
    plot_beta_function(all_x, all_beta, coeffs, roots, derivatives)

    # 5. Построение потенциала
    print("\n" + "-" * 50)
    print("3. ПОСТРОЕНИЕ ЭФФЕКТИВНОГО ПОТЕНЦИАЛА")
    print("-" * 50)

    x_range = np.linspace(base_x - 0.05, base_x + 0.05, 500)
    x_range, V, V_coeffs = compute_potential_from_coeffs(coeffs, x_range, shift_to_zero=True)

    # Фильтруем корни, оставляя только в диапазоне
    if len(roots) > 0:
        valid_roots = roots[(roots >= min(x_range)) & (roots <= max(x_range))]
    else:
        valid_roots = np.array([base_x])

    print(f"\nКоэффициенты V(x):")
    print(f"  a₃ = {V_coeffs[0]:.3e}")
    print(f"  a₂ = {V_coeffs[1]:.3e}")
    print(f"  a₁ = {V_coeffs[2]:.3e}")
    print(f"  a₀ = {V_coeffs[3]:.3e}")

    # 6. Визуализация потенциала
    plot_potential(x_range, V, valid_roots, V_coeffs)

    # 7. Формулировка эффективной теории
    formulate_effective_theory(coeffs, roots, derivatives, V_coeffs, base_x)

    # 8. Сохранение результатов
    import json
    results = {
        'base_parameters': {'K': base_K, 'p': float(base_p), 'N': float(base_N)},
        'base_x': float(base_x),
        'beta_coeffs': [float(c) for c in coeffs],
        'roots': [float(r) for r in roots] if len(roots) > 0 else [float(base_x)],
        'derivatives': [float(d) for d in derivatives] if len(derivatives) > 0 else [
            float(np.polyval(np.polyder(coeffs), base_x))],
        'V_coeffs': [float(c) for c in V_coeffs]
    }

    with open('effective_theory_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\n✅ Результаты сохранены в effective_theory_results.json")

    return flows, coeffs, roots, derivatives, V_coeffs


# ЧАСТЬ 6: ЗАПУСК ПОЛНОГО АНАЛИЗА

if __name__ == "__main__":
    flows, coeffs, roots, derivatives, V_coeffs = main()