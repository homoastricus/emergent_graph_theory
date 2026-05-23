"""
АНАЛИЗ УПРАВЛЯЮЩЕГО ПАРАМЕТРА Λ = 4η²γ ln N / K³

Исследование зависимости Λ от физических параметров:
- N: число узлов (энтропия горизонта)
- K: средняя степень связности
- η: нелинейность (crowding)
- γ: транспортный коэффициент

Автор: на основе совместного исследования
Дата: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# ЧАСТЬ 1: ВЫЧИСЛЕНИЕ Λ
# ============================================================================

def calculate_Lambda(N: float, K: float = 6.0, eta: float = 0.1, gamma: float = 1.0) -> float:
    """
    Вычисление управляющего параметра Λ.

    Λ = 4η²γ ln N / K³

    Parameters:
    -----------
    N : float
        Число узлов (информационная емкость системы)
    K : float
        Средняя степень узла (локальная связность)
    eta : float
        Нелинейность (crowding effect)
    gamma : float
        Транспортный коэффициент

    Returns:
    --------
    Lambda : float
        Управляющий параметр
    """
    return 4 * eta ** 2 * gamma * np.log(N) / K ** 3


def calculate_N_from_Lambda(Lambda: float, K: float = 6.0, eta: float = 0.1, gamma: float = 1.0) -> float:
    """
    Обратное вычисление N из Λ.

    N = exp(Λ · K³ / (4η²γ))
    """
    return np.exp(Lambda * K ** 3 / (4 * eta ** 2 * gamma))


def calculate_eta_from_Lambda(Lambda: float, N: float, K: float = 6.0, gamma: float = 1.0) -> float:
    """
    Вычисление η из Λ при заданных N, K, γ.

    η = √(Λ · K³ / (4γ ln N))
    """
    return np.sqrt(Lambda * K ** 3 / (4 * gamma * np.log(N)))


def calculate_K_from_Lambda(Lambda: float, N: float, eta: float = 0.1, gamma: float = 1.0) -> float:
    """
    Вычисление K из Λ при заданных N, η, γ.

    K = ∛(4η²γ ln N / Λ)
    """
    return np.cbrt(4 * eta ** 2 * gamma * np.log(N) / Lambda)


# ============================================================================
# ЧАСТЬ 2: АНАЛИЗ ЧУВСТВИТЕЛЬНОСТИ
# ============================================================================

@dataclass
class LambdaAnalysis:
    """Результаты анализа Λ"""
    Lambda: float
    N: float
    K: float
    eta: float
    gamma: float
    ln_N: float

    @property
    def regime(self) -> str:
        """Определение режима по Λ"""
        if self.Lambda < 0.01:
            return "ГЛУБОКО ГЕОМЕТРИЧЕСКИЙ"
        elif self.Lambda < 0.1:
            return "ГЕОМЕТРИЧЕСКИЙ"
        elif self.Lambda < 1.0:
            return "ПЕРЕХОДНЫЙ (критический)"
        elif self.Lambda < 10.0:
            return "НЕЛОКАЛЬНЫЙ"
        else:
            return "УЛЬТРА-НЕЛОКАЛЬНЫЙ"

    @property
    def sensitivity_N(self) -> float:
        """d(ln Λ)/d(ln N) = 1/ln N"""
        return 1.0 / self.ln_N

    @property
    def sensitivity_K(self) -> float:
        """d(ln Λ)/d(ln K) = -3"""
        return -3.0

    @property
    def sensitivity_eta(self) -> float:
        """d(ln Λ)/d(ln η) = 2"""
        return 2.0

    @property
    def sensitivity_gamma(self) -> float:
        """d(ln Λ)/d(ln γ) = 1"""
        return 1.0


def analyze_Lambda(N: float, K: float = 6.0, eta: float = 0.1, gamma: float = 1.0) -> LambdaAnalysis:
    """Полный анализ Λ для заданных параметров"""
    Lambda = calculate_Lambda(N, K, eta, gamma)
    return LambdaAnalysis(
        Lambda=Lambda,
        N=N,
        K=K,
        eta=eta,
        gamma=gamma,
        ln_N=np.log(N)
    )


# ============================================================================
# ЧАСТЬ 3: СРАВНЕНИЕ С ФИЗИЧЕСКИМИ МАСШТАБАМИ
# ============================================================================

def physical_scales() -> dict:
    """Физически значимые масштабы N и соответствующие Λ"""

    scales = {
        'Планковский масштаб': 1.0,
        'Протон': 1e20,
        'Атом': 1e40,
        'Человек': 1e80,
        'Земля': 1e100,
        'Солнечная система': 1e110,
        'Галактика': 1e115,
        'Наблюдаемая Вселенная (энтропия)': 3.576e121,
        'Наблюдаемая Вселенная (объем)': 1e184,
        'Мультивселенная (оценка)': 1e500,
    }

    results = {}
    for name, N in scales.items():
        N_float = float(N)  # <-- ВОТ ФИКС: преобразование в float
        Lambda = calculate_Lambda(N_float)
        results[name] = {
            'N': N_float,
            'ln N': np.log(N_float),
            'Lambda': Lambda,
            'regime': analyze_Lambda(N_float).regime
        }

    return results


# ============================================================================
# ЧАСТЬ 4: ПОИСК КРИТИЧЕСКИХ ТОЧЕК
# ============================================================================

def find_critical_N(Lambda_target: float = 1.0, K: float = 6.0,
                    eta: float = 0.1, gamma: float = 1.0) -> float:
    """
    Найти N, при котором Λ = Lambda_target.

    Критические значения Λ:
    - Λ = 1: переход геометрический/нелокальный
    - Λ = 0.1: граница глубокого геометрического режима
    - Λ = 10: граница ультра-нелокального режима
    """
    return calculate_N_from_Lambda(Lambda_target, K, eta, gamma)


def find_critical_eta(Lambda_target: float, N: float, K: float = 6.0, gamma: float = 1.0) -> float:
    """Найти η, при котором Λ = Lambda_target"""
    return calculate_eta_from_Lambda(Lambda_target, N, K, gamma)


def find_critical_K(Lambda_target: float, N: float, eta: float = 0.1, gamma: float = 1.0) -> float:
    """Найти K, при котором Λ = Lambda_target"""
    return calculate_K_from_Lambda(Lambda_target, N, eta, gamma)


# ============================================================================
# ЧАСТЬ 5: ВИЗУАЛИЗАЦИЯ
# ============================================================================

def plot_Lambda_dependence():
    """Визуализация зависимости Λ от параметров"""
    fig = plt.figure(figsize=(16, 12))

    # 1. Λ(N) для разных K
    ax1 = plt.subplot(2, 3, 1)
    N_range = np.logspace(2, 200, 500)
    for K in [4, 6, 8, 10]:
        Lambda_vals = [calculate_Lambda(N, K=K) for N in N_range]
        ax1.loglog(N_range, Lambda_vals, linewidth=2, label=f'K={K}')

    # Отметки физических масштабов
    scales = physical_scales()
    for name, data in scales.items():
        if 'Вселенная' in name:
            ax1.scatter([data['N']], [data['Lambda']], s=100, marker='*',
                        edgecolors='black', linewidth=1.5)
            ax1.annotate(name.split()[0], (data['N'], data['Lambda']),
                         fontsize=8, ha='right')

    ax1.axhline(y=1.0, color='purple', linestyle=':', linewidth=2, label='Λ = 1 (переход)')
    ax1.axhline(y=0.1, color='orange', linestyle='--', alpha=0.7, label='Λ = 0.1')
    ax1.set_xlabel('N (число узлов)', fontsize=12)
    ax1.set_ylabel('Λ', fontsize=12)
    ax1.set_title('Зависимость Λ от N при разных K', fontsize=14)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 2. Λ(K) для разных N
    ax2 = plt.subplot(2, 3, 2)
    K_range = np.linspace(3, 15, 200)
    for N in [1e50, 1e100, 1e122, 1e184]:
        Lambda_vals = [calculate_Lambda(N, K=K) for K in K_range]
        ax2.semilogy(K_range, Lambda_vals, linewidth=2, label=f'N=10^{int(np.log10(N))}')

    ax2.axhline(y=1.0, color='purple', linestyle=':', linewidth=2)
    ax2.axvline(x=6.0, color='red', linestyle='--', alpha=0.7, label='K=6 (3D-решетка)')
    ax2.set_xlabel('K (средняя степень)', fontsize=12)
    ax2.set_ylabel('Λ', fontsize=12)
    ax2.set_title('Зависимость Λ от K при разных N', fontsize=14)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # 3. Λ(η) для разных N
    ax3 = plt.subplot(2, 3, 3)
    eta_range = np.logspace(-2, 1, 200)
    for N in [1e50, 1e100, 1e122, 1e184]:
        Lambda_vals = [calculate_Lambda(N, eta=eta) for eta in eta_range]
        ax3.loglog(eta_range, Lambda_vals, linewidth=2, label=f'N=10^{int(np.log10(N))}')

    ax3.axhline(y=1.0, color='purple', linestyle=':', linewidth=2)
    ax3.axvline(x=0.1, color='green', linestyle='--', alpha=0.7, label='η=0.1 (базовое)')
    ax3.set_xlabel('η (crowding)', fontsize=12)
    ax3.set_ylabel('Λ', fontsize=12)
    ax3.set_title('Зависимость Λ от η при разных N', fontsize=14)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # 4. Контурный график Λ(N, K)
    ax4 = plt.subplot(2, 3, 4)
    N_grid = np.logspace(2, 200, 100)
    K_grid = np.linspace(3, 12, 100)
    N_mesh, K_mesh = np.meshgrid(N_grid, K_grid)
    Lambda_mesh = 4 * 0.1 ** 2 * 1.0 * np.log(N_mesh) / K_mesh ** 3

    contour = ax4.contourf(np.log10(N_mesh), K_mesh, np.log10(Lambda_mesh + 1e-12),
                           levels=20, cmap='viridis')
    ax4.contour(np.log10(N_mesh), K_mesh, np.log10(Lambda_mesh + 1e-12),
                levels=[0], colors='red', linewidths=2, linestyles='-')
    ax4.contour(np.log10(N_mesh), K_mesh, np.log10(Lambda_mesh + 1e-12),
                levels=[-1], colors='orange', linewidths=1.5, linestyles='--')

    # Наша Вселенная
    ax4.scatter([np.log10(3.576e121)], [6.0], color='gold', s=200, marker='*',
                edgecolors='black', linewidth=2, label='Наша Вселенная')

    ax4.set_xlabel('log₁₀ N', fontsize=12)
    ax4.set_ylabel('K', fontsize=12)
    ax4.set_title('log₁₀ Λ в зависимости от N и K', fontsize=14)
    ax4.legend(fontsize=10)
    plt.colorbar(contour, ax=ax4, label='log₁₀ Λ')

    # 5. Чувствительность Λ к параметрам
    ax5 = plt.subplot(2, 3, 5)

    # Для нашей Вселенной
    analysis = analyze_Lambda(3.576e121)

    sensitivities = {
        'N (логарифмическая)': analysis.sensitivity_N,
        'K (кубическая)': abs(analysis.sensitivity_K),
        'η (квадратичная)': analysis.sensitivity_eta,
        'γ (линейная)': analysis.sensitivity_gamma
    }

    colors_bar = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    bars = ax5.bar(sensitivities.keys(), sensitivities.values(), color=colors_bar, alpha=0.7)
    ax5.set_ylabel('|∂ ln Λ / ∂ ln(param)|', fontsize=12)
    ax5.set_title('Чувствительность Λ к параметрам', fontsize=14)
    ax5.tick_params(axis='x', rotation=45)

    for bar, val in zip(bars, sensitivities.values()):
        ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    ax5.grid(True, alpha=0.3, axis='y')

    # 6. Таблица критических значений
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')

    # Критические N для разных Λ_target
    critical_data = []
    for Lambda_t in [0.01, 0.1, 1.0, 10.0]:
        N_crit = find_critical_N(Lambda_t)
        critical_data.append([f'Λ = {Lambda_t}', f'N = {N_crit:.2e}'])

    # Критические η для нашей Вселенной
    N_universe = 4.198e121
    for Lambda_t in [0.01, 0.1, 1.0, 10.0]:
        eta_crit = find_critical_eta(Lambda_t, N_universe)
        critical_data.append([f'Λ = {Lambda_t} (η)', f'η = {eta_crit:.4f}'])

    table_text = "КРИТИЧЕСКИЕ ЗНАЧЕНИЯ:\n\n"
    table_text += "Критические N (K=6, η=0.1, γ=1):\n"
    for Lambda_t in [0.01, 0.1, 1.0, 10.0]:
        N_crit = find_critical_N(Lambda_t)
        table_text += f"  Λ = {Lambda_t:5.2f} → N = {N_crit:.2e}\n"

    table_text += "\nКритические η (N=3.58e121, K=6, γ=1):\n"
    for Lambda_t in [0.01, 0.1, 1.0, 10.0]:
        eta_crit = find_critical_eta(Lambda_t, N_universe)
        table_text += f"  Λ = {Lambda_t:5.2f} → η = {eta_crit:.4f}\n"

    table_text += "\nКритические K (N=3.58e121, η=0.1, γ=1):\n"
    for Lambda_t in [0.01, 0.1, 1.0, 10.0]:
        K_crit = find_critical_K(Lambda_t, N_universe)
        table_text += f"  Λ = {Lambda_t:5.2f} → K = {K_crit:.2f}\n"

    ax6.text(0.1, 0.9, table_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('Lambda_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

    return fig


# ============================================================================
# ЧАСТЬ 6: ГЛАВНЫЙ ЗАПУСК
# ============================================================================

def main():
    print("АНАЛИЗ УПРАВЛЯЮЩЕГО ПАРАМЕТРА Λ = 4η²γ ln N / K³")

    # Базовый расчет для нашей Вселенной
    N_universe = 4.198e121
    K_universe = 6.0
    eta_universe = 0.1
    gamma_universe = 1.0

    Lambda_universe = calculate_Lambda(N_universe, K_universe, eta_universe, gamma_universe)
    analysis = analyze_Lambda(N_universe, K_universe, eta_universe, gamma_universe)

    print("\n📊 НАША ВСЕЛЕННАЯ:")
    print(f"  N = {N_universe:.3e}")
    print(f"  ln N = {np.log(N_universe):.3f}")
    print(f"  K = {K_universe}")
    print(f"  η = {eta_universe}")
    print(f"  γ = {gamma_universe}")
    print(f"\n🎯 Λ = {Lambda_universe:.6e}")
    print(f"  Режим: {analysis.regime}")

    print("\n📐 ЧУВСТВИТЕЛЬНОСТЬ Λ:")
    print(f"  ∂ ln Λ / ∂ ln N = {analysis.sensitivity_N:.6f} (очень слабая!)")
    print(f"  ∂ ln Λ / ∂ ln K = {analysis.sensitivity_K:.1f} (сильная, кубическая)")
    print(f"  ∂ ln Λ / ∂ ln η = {analysis.sensitivity_eta:.1f} (квадратичная)")
    print(f"  ∂ ln Λ / ∂ ln γ = {analysis.sensitivity_gamma:.1f} (линейная)")

    print("\n🔬 ФИЗИЧЕСКИЕ МАСШТАБЫ:")
    scales = physical_scales()
    for name, data in scales.items():
        if 'Вселенная' in name or name in ['Планковский масштаб', 'Протон']:
            print(f"  {name:35}: N={data['N']:.2e}, Λ={data['Lambda']:.2e} ({data['regime']})")

    print("\n🎯 КРИТИЧЕСКИЕ ЗНАЧЕНИЯ:")
    print("\n  Критические N (при K=6, η=0.1, γ=1):")
    for Lambda_t in [0.01, 0.1, 1.0, 10.0]:
        N_crit = find_critical_N(Lambda_t)
        print(f"    Λ = {Lambda_t:5.2f} → N = {N_crit:.2e}")

    print("\n  Критические η (при N=3.58e121, K=6, γ=1):")
    for Lambda_t in [0.01, 0.1, 1.0, 10.0]:
        eta_crit = find_critical_eta(Lambda_t, N_universe)
        print(f"    Λ = {Lambda_t:5.2f} → η = {eta_crit:.4f}")

    print("\n  Критические K (при N=3.58e121, η=0.1, γ=1):")
    for Lambda_t in [0.01, 0.1, 1.0, 10.0]:
        K_crit = find_critical_K(Lambda_t, N_universe)
        print(f"    Λ = {Lambda_t:5.2f} → K = {K_crit:.2f}")

    print("\n💡 ВАЖНЫЕ ВЫВОДЫ:")
    print("  1. Λ ∝ ln N — ЛОГАРИФМИЧЕСКАЯ зависимость от N")
    print("  2. Даже при изменении N на порядки, Λ меняется слабо")
    print("  3. ФУНДАМЕНТАЛЕН именно Λ, а не N")
    print("  4. 10^122 — не 'единственное' N, а естественный масштаб")
    print("  5. K влияет на Λ КУБИЧЕСКИ — самая сильная зависимость")

    print("\n📈 ПОСТРОЕНИЕ ГРАФИКОВ...")
    plot_Lambda_dependence()

    print("\n✅ Анализ завершен. График сохранен как 'Lambda_analysis.png'")

    return analysis


if __name__ == "__main__":
    analysis = main()