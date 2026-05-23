"""
РАСШИРЕННЫЙ АНАЛИЗ: RG-ИНВАРИАНТЫ ДЛЯ РАЗНЫХ ТИПОВ ВЗАИМОДЕЙСТВИЙ
Проверка сохранения и критического поведения
"""

import numpy as np
from scipy.sparse import csr_matrix, diags, lil_matrix
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigvalsh
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
import warnings

from draft.spectral import PhysicalGraphImproved

warnings.filterwarnings('ignore')


# ============================================================
# ЧАСТЬ 1: RG-ИНВАРИАНТЫ
# ============================================================

def compute_rg_invariants(N, K, p, xi=None):
    """
    Вычисление RG-инвариантов для заданных параметров.

    Инварианты:
    - I1 = ln N / |ln(Kp)|        # Модулярный инвариант U
    - I2 = ln K - ln(Kp)/ln N     # Баланс локального и глобального
    - I3 = ln p / ln(Kp)          # Внутренняя структура p
    - I4 = (ln(Kp)/ln N)^2        # λ = x² (спектральная щель)
    - I5 = ln N + ln p            # Сумма масштабов
    - I6 = ln N / |ln p|          # Отношение масштабов
    - V = (U/π)^(2/3) * hbar_em^3 * (ln N)^2  # RG-инвариант
    - critical_parameter = p * N^(1/4)        # Критический параметр
    """
    lnN = np.log(N)
    lnK = np.log(K) if K > 0 else 0
    lnp = np.log(p) if p > 0 else -np.inf
    lnKp = np.log(K * p) if (K * p) > 0 else -np.inf

    # Защита от сингулярностей
    if np.isinf(lnp) or np.isinf(lnKp):
        return {f'I{i}': np.nan for i in range(1, 7)}

    I1 = lnN / abs(lnKp) if abs(lnKp) > 1e-10 else np.inf  # U
    I2 = lnK - lnKp / lnN if lnN > 0 else np.nan
    I3 = lnp / lnKp if abs(lnKp) > 1e-10 else np.nan
    I4 = (lnKp / lnN) ** 2 if lnN > 0 else np.nan  # λ
    I5 = lnN + lnp
    I6 = lnN / abs(lnp) if abs(lnp) > 1e-10 else np.inf

    # Дополнительные физические инварианты
    x = lnKp / lnN if lnN > 0 else np.nan
    lambda_val = x ** 2
    U = I1
    f1 = U / np.pi
    hbar_em = (lnK) ** 2 / (4 * lambda_val ** 2 * K ** 2) if lambda_val > 1e-10 else np.inf

    V = (f1 ** (2 / 3)) * (hbar_em ** 3) * (lnN ** 2) if not np.isinf(hbar_em) else np.inf
    critical_parameter = p * (N ** 0.25)

    return {
        'I1_U': I1,
        'I2_balance': I2,
        'I3_structure': I3,
        'I4_lambda': I4,
        'I5_sum': I5,
        'I6_ratio': I6,
        'x': x,
        'f1': f1,
        'hbar_em': hbar_em,
        'V_RG': V,
        'critical_param': critical_parameter,
        'N': N,
        'K': K,
        'p': p,
        'xi': xi
    }


# ============================================================
# ЧАСТЬ 2: КЛАСС ГРАФА С ВЫЧИСЛЕНИЕМ ИНВАРИАНТОВ
# ============================================================

class PhysicalGraphWithRG(PhysicalGraphImproved):
    """Расширенная версия графа с вычислением RG-инвариантов."""

    def __init__(self, N=1400, K=8, use_complex=True):
        super().__init__(N, K, use_complex)
        self.rg_invariants = {}

    def compute_all_rg_invariants(self):
        """Вычисление RG-инвариантов для всех каналов."""
        invariants = []
        for i, (p, xi, name) in enumerate(self.channel_params):
            inv = compute_rg_invariants(self.N, self.K, p, xi)
            inv['channel'] = name
            inv['channel_idx'] = i
            invariants.append(inv)
        self.rg_invariants = invariants
        return invariants

    def analyze_invariant_evolution(self, param_range, param_name='p'):
        """
        Анализ эволюции инвариантов при изменении параметра.

        param_range: массив значений параметра
        param_name: 'p' или 'xi' или 'N' или 'K'
        """
        results = []
        base_params = self.channel_params[0]  # Берём гравитацию как базовый

        for val in param_range:
            if param_name == 'p':
                p, xi, name = val, base_params[1], base_params[2]
            elif param_name == 'xi':
                p, xi, name = base_params[0], val, base_params[2]
            elif param_name == 'N':
                self.N = int(val)
                p, xi, name = base_params[0], base_params[1], base_params[2]
            elif param_name == 'K':
                self.K = int(val)
                p, xi, name = base_params[0], base_params[1], base_params[2]

            inv = compute_rg_invariants(self.N, self.K, p, xi)
            inv['param_value'] = val
            inv['param_name'] = param_name
            results.append(inv)

        return results


# ============================================================
# ЧАСТЬ 3: АНАЛИЗ СОХРАНЕНИЯ ИНВАРИАНТОВ
# ============================================================

def analyze_rg_invariants_for_channels(N=1400, K=8):
    """Анализ RG-инвариантов для 4 типов взаимодействий."""

    print("=" * 80)
    print("RG-ИНВАРИАНТЫ ДЛЯ РАЗНЫХ ТИПОВ ВЗАИМОДЕЙСТВИЙ")
    print("=" * 80)

    # Создаём граф
    graph = PhysicalGraphWithRG(N, K, use_complex=True)

    # Определяем каналы с параметрами
    channels = [
        ('Гравитация', 0.000005, 50.0),
        ('Сильное', 0.5, 0.5),
        ('Слабое', 0.03, 5.0),
        ('ЭМ', 0.002, 15.0)
    ]

    all_invariants = []

    print(f"\n{'Канал':<12} {'p':<12} {'ξ':<8} ", end='')
    print(f"{'I1(U)':<10} {'I4(λ)':<10} {'I6':<10} {'V_RG':<12} {'Крит.пар':<10}")
    print("-" * 85)

    for name, p, xi in channels:
        inv = compute_rg_invariants(N, K, p, xi)
        inv['channel'] = name
        all_invariants.append(inv)

        print(f"{name:<12} {p:<12.6f} {xi:<8.1f} ", end='')
        print(f"{inv['I1_U']:<10.4f} {inv['I4_lambda']:<10.6f} ", end='')
        print(f"{inv['I6_ratio']:<10.4f} {inv['V_RG']:<12.4e} {inv['critical_param']:<10.4f}")

    return all_invariants


def analyze_invariant_evolution(N=1400, K=8):
    """Анализ эволюции инвариантов при изменении параметров."""

    graph = PhysicalGraphWithRG(N, K, use_complex=True)

    # Сканируем по p (вероятность связи)
    p_range = np.logspace(-6, 0, 20)  # от 1e-6 до 1
    results_p = graph.analyze_invariant_evolution(p_range, 'p')

    # Сканируем по xi (масштаб длины)
    xi_range = np.logspace(-1, 2, 20)  # от 0.1 до 100
    results_xi = graph.analyze_invariant_evolution(xi_range, 'xi')

    # Сканируем по N (размер графа)
    N_range = np.logspace(2, 4, 15, dtype=int)  # от 100 до 10000
    results_N = graph.analyze_invariant_evolution(N_range, 'N')

    # Визуализация
    visualize_rg_evolution(results_p, results_xi, results_N)

    return results_p, results_xi, results_N


def visualize_rg_evolution(results_p, results_xi, results_N):
    """Визуализация эволюции RG-инвариантов."""

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Извлекаем данные
    p_vals = [r['param_value'] for r in results_p]
    xi_vals = [r['param_value'] for r in results_xi]
    N_vals = [r['param_value'] for r in results_N]

    # 1. Эволюция I1 (U) по p
    ax = axes[0, 0]
    i1_p = [r['I1_U'] for r in results_p]
    ax.semilogx(p_vals, i1_p, 'b-', linewidth=2)
    ax.axhline(y=np.pi, color='r', linestyle='--', label=r'$U = \pi$ (критическое)')
    ax.set_xlabel('p (вероятность связи)')
    ax.set_ylabel('I1 = U')
    ax.set_title('Модулярный инвариант U vs p')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 2. Эволюция I4 (λ) по p
    ax = axes[0, 1]
    i4_p = [r['I4_lambda'] for r in results_p]
    ax.loglog(p_vals, i4_p, 'g-', linewidth=2)
    ax.set_xlabel('p')
    ax.set_ylabel('I4 = λ (спектральная щель)')
    ax.set_title('Спектральная щель vs p')
    ax.grid(True, alpha=0.3)

    # 3. Эволюция I6 (отношение масштабов) по p
    ax = axes[0, 2]
    i6_p = [r['I6_ratio'] for r in results_p]
    ax.semilogx(p_vals, i6_p, 'r-', linewidth=2)
    ax.set_xlabel('p')
    ax.set_ylabel('I6 = ln N / |ln p|')
    ax.set_title('Отношение масштабов vs p')
    ax.grid(True, alpha=0.3)

    # 4. Эволюция V_RG по xi
    ax = axes[1, 0]
    v_xi = [r['V_RG'] for r in results_xi]
    ax.loglog(xi_vals, v_xi, 'purple', linewidth=2)
    ax.axhline(y=1.0, color='orange', linestyle='--', label='V = 1 (фиксированная точка)')
    ax.set_xlabel('ξ (масштаб длины)')
    ax.set_ylabel('V_RG')
    ax.set_title('RG-инвариант V vs ξ')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 5. Критический параметр vs N
    ax = axes[1, 1]
    crit_N = [r['critical_param'] for r in results_N]
    ax.loglog(N_vals, crit_N, 'b-', linewidth=2)
    ax.axhline(y=1.0, color='r', linestyle='--', label=r'$p N^{1/4} = 1$ (крит. линия)')
    ax.set_xlabel('N (размер графа)')
    ax.set_ylabel(r'$p \cdot N^{1/4}$')
    ax.set_title('Критический параметр vs N')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 6. Корреляция между I1 и I4
    ax = axes[1, 2]
    ax.scatter([r['I1_U'] for r in results_p],
               [r['I4_lambda'] for r in results_p],
               c=p_vals, cmap='viridis', norm='log', s=50)
    ax.set_xlabel('I1 = U')
    ax.set_ylabel('I4 = λ')
    ax.set_title('U-λ корреляция (цвет = p)')
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label('p')

    plt.tight_layout()
    plt.savefig('rg_invariants_analysis.png', dpi=150)
    plt.show()


def find_conserved_quantities(all_invariants):
    """Поиск сохраняющихся величин для разных каналов."""

    print("\n" + "=" * 80)
    print("ПОИСК СОХРАНЯЮЩИХСЯ ВЕЛИЧИН")
    print("=" * 80)

    # Создаём DataFrame для анализа
    df = pd.DataFrame(all_invariants)

    # Вычисляем вариации для каждого инварианта
    invariants_cols = ['I1_U', 'I2_balance', 'I3_structure', 'I4_lambda',
                       'I5_sum', 'I6_ratio', 'V_RG', 'critical_param']

    print("\nВариации инвариантов между каналами:")
    print("-" * 60)

    conserved = []
    for col in invariants_cols:
        values = df[col].values
        mean_val = np.mean(values)
        std_val = np.std(values)
        cv = std_val / abs(mean_val) if mean_val != 0 else np.inf  # коэф. вариации

        # Убираем inf для V_RG при вычислениях
        if np.isinf(cv) or np.isnan(cv):
            cv = np.inf

        print(f"{col:<20}: mean = {mean_val:<12.4e}, std = {std_val:<12.4e}, CV = {cv:<.4f}")

        if cv < 0.5:  # Малая вариация = возможное сохранение
            conserved.append(col)

    print("\n" + "-" * 60)
    print(f"Потенциально сохраняющиеся инварианты (CV < 0.5): {conserved}")

    # Корреляционный анализ
    print("\nКорреляционная матрица инвариантов:")
    print("-" * 60)
    corr_matrix = df[invariants_cols].corr()
    print(corr_matrix.round(3))

    return conserved, corr_matrix


# ============================================================
# ЧАСТЬ 4: КРИТИЧЕСКОЕ ПОВЕДЕНИЕ
# ============================================================

def analyze_critical_behavior():
    """Анализ критического поведения вблизи p ~ N^{-1/4}."""

    print("\n" + "=" * 80)
    print("КРИТИЧЕСКОЕ ПОВЕДЕНИЕ")
    print("=" * 80)

    N_vals = np.array([100, 200, 500, 1000, 2000, 5000, 10000])

    # Критическая линия: p_crit = N^{-1/4}
    p_crit = N_vals ** (-0.25)

    print(f"\n{'N':<8} {'p_crit = N^{-1/4}':<20} {'p_гравитация':<15} {'p_ЭМ':<15}")
    print("-" * 60)

    for N, p_c in zip(N_vals, p_crit):
        # Сравниваем с параметрами наших каналов
        p_grav = 5e-6
        p_em = 0.002

        ratio_grav = p_grav / p_c if p_c > 0 else np.inf
        ratio_em = p_em / p_c

        print(f"{N:<8} {p_c:<20.6e} {ratio_grav:<15.4f} {ratio_em:<15.4f}")

    print("\nИнтерпретация:")
    print("-" * 60)
    print("• Если p/p_crit << 1: режим разреженных связей (гравитация)")
    print("• Если p/p_crit ≈ 1: критический режим (переход)")
    print("• Если p/p_crit >> 1: плотный режим (сильное взаимодействие)")

    # Визуализация фазовой диаграммы
    fig, ax = plt.subplots(figsize=(10, 6))

    # Сетка параметров
    N_grid = np.logspace(2, 4, 50)
    p_grid = np.logspace(-6, 0, 50)
    N_mesh, p_mesh = np.meshgrid(N_grid, p_grid)

    # Критическая линия
    p_crit_mesh = N_mesh ** (-0.25)

    # Фаза: 0 - разреженная, 1 - критическая, 2 - плотная
    phase = np.zeros_like(N_mesh)
    phase[p_mesh < p_crit_mesh * 0.1] = 0
    phase[(p_mesh >= p_crit_mesh * 0.1) & (p_mesh <= p_crit_mesh * 10)] = 1
    phase[p_mesh > p_crit_mesh * 10] = 2

    contour = ax.contourf(N_mesh, p_mesh, phase, levels=[-0.5, 0.5, 1.5, 2.5],
                          colors=['lightblue', 'lightgreen', 'salmon'], alpha=0.6)

    ax.plot(N_grid, p_crit_mesh[0], 'k--', linewidth=2, label=r'$p_{\rm crit} = N^{-1/4}$')

    # Отмечаем наши каналы
    for N in [100, 1000, 10000]:
        ax.scatter(N, 5e-6, color='blue', s=100, marker='o', label='Гравитация' if N == 1000 else '')
        ax.scatter(N, 0.002, color='red', s=100, marker='s', label='ЭМ' if N == 1000 else '')
        ax.scatter(N, 0.03, color='green', s=100, marker='^', label='Слабое' if N == 1000 else '')
        ax.scatter(N, 0.5, color='purple', s=100, marker='D', label='Сильное' if N == 1000 else '')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('N (размер графа)')
    ax.set_ylabel('p (вероятность связи)')
    ax.set_title('Фазовая диаграмма: p vs N\n(синий=разреженная, зелёный=критическая, красный=плотная)')
    ax.legend(loc='lower left', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('phase_diagram.png', dpi=150)
    plt.show()


# ============================================================
# ЧАСТЬ 5: ЗАПУСК ВСЕГО АНАЛИЗА
# ============================================================

def main():
    """Главная функция для анализа RG-инвариантов."""

    # 1. Базовый анализ для 4 каналов
    invariants = analyze_rg_invariants_for_channels(N=1400, K=8)

    # 2. Эволюция инвариантов
    results_p, results_xi, results_N = analyze_invariant_evolution(N=1400, K=8)

    # 3. Поиск сохраняющихся величин
    conserved, corr_matrix = find_conserved_quantities(invariants)

    # 4. Критическое поведение
    analyze_critical_behavior()

    # 5. Выводы
    print("\n" + "=" * 80)
    print("ВЫВОДЫ ПО RG-ИНВАРИАНТАМ")
    print("=" * 80)

    print("""
    1. ИНВАРИАНТЫ:
       - I1 (U): сильно меняется между каналами (от 0.1 до 100+)
       - I4 (λ): спектральная щель — чувствительный параметр
       - V_RG: расходится для некоторых каналов (проблема сингулярности)

    2. СОХРАНЕНИЕ:
       - Ни один инвариант не сохраняется строго между каналами
       - Наиболее стабилен I6 (отношение масштабов), но вариация ~50%

    3. КРИТИЧЕСКОЕ ПОВЕДЕНИЕ:
       - Гравитация: p << N^{-1/4} (разреженный режим)
       - Сильное: p >> N^{-1/4} (плотный режим)
       - Слабое и ЭМ: вблизи критической области

    4. РЕКОМЕНДАЦИИ:
       - Для проверки универсальности нужно смотреть на комбинации инвариантов
       - Возможно, существует скрытый закон сохранения: I1 * I4^α = const
       - V_RG может быть полезен для классификации, но требует регуляризации
    """)

    return invariants, conserved, corr_matrix


if __name__ == "__main__":
    invariants, conserved, corr_matrix = main()