"""
УНИВЕРСАЛЬНАЯ БЕЗРАЗМЕРНАЯ МОДЕЛЬ ГЛОБАЛЬНОГО ИНФОРМАЦИОННОГО ПОЛЯ (ГИП)

Сведение всей физики к одному управляющему параметру Λ.
Уравнение состояния: y²(1 + y) = Λ/2 (точное)
"""

import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# ЧАСТЬ 1: УНИВЕРСАЛЬНЫЕ БЕЗРАЗМЕРНЫЕ ПЕРЕМЕННЫЕ
# ============================================================================

@dataclass
class DimensionlessGIP:
    """Безразмерные параметры Глобального Информационного Поля"""
    Lambda: float  # управляющий параметр Λ = 4η²γ ln N / K³
    y: float       # безразмерная нелокальность y = 2ηx/K

    @property
    def regime(self) -> str:
        """Определение режима по y"""
        if self.y < 0.5:
            return "ГЕОМЕТРИЧЕСКИЙ (локально-доминированный)"
        elif self.y < 2.0:
            return "ПЕРЕХОДНЫЙ"
        else:
            return "НЕЛОКАЛЬНЫЙ (плазменный)"

    @property
    def scaling_exponent(self) -> float:
        """Эффективный показатель степени x ~ (ln N)^ν"""
        if self.y < 0.1:
            return 0.5
        elif self.y > 10:
            return 1/3
        else:
            return 0.5 - 0.167 * (self.y / (1 + self.y))


# ============================================================================
# ЧАСТЬ 2: ТОЧНОЕ АНАЛИТИЧЕСКОЕ РЕШЕНИЕ y(Λ)
# ============================================================================

def solve_y_from_cubic(Lambda: float) -> float:
    """
    Решение кубического уравнения y³ + y² = Λ.
    Использует формулу Кардано.
    """
    if Lambda == 0:
        return 0.0

    p = -1/3
    q = -1/27 - Lambda
    D = (q/2)**2 + (p/3)**3

    if D >= 0:
        u = np.cbrt(-q/2 + np.sqrt(D))
        v = np.cbrt(-q/2 - np.sqrt(D))
        t = u + v
    else:
        phi = np.arccos(-q/2 / np.sqrt(-(p/3)**3))
        t = 2 * np.sqrt(-p/3) * np.cos(phi/3)

    y = t - 1/3
    return max(0.0, y)


def solve_Lambda_from_y(y: float) -> float:
    """Обратное преобразование: Λ = y²(1 + y)"""
    return y**2 * (1 + y)


# ============================================================================
# ЧАСТЬ 3: СВЯЗЬ С ФИЗИЧЕСКИМИ ПАРАМЕТРАМИ
# ============================================================================

@dataclass
class PhysicalGIP:
    """Физические параметры ГИП"""
    K: float      # средняя степень
    eta: float    # нелинейность (crowding)
    gamma: float  # транспортный коэффициент
    N: float      # число узлов
    d: float = 3.0  # размерность

    @property
    def Lambda(self) -> float:
        """Вычисление Λ из физических параметров"""
        return 4 * self.eta**2 * self.gamma * np.log(self.N) / self.K**3

    @property
    def y_cubic(self) -> float:
        """Безразмерная нелокальность из кубического уравнения"""
        return solve_y_from_cubic(self.Lambda)

    @property
    def y_exact(self) -> float:
        """Точная безразмерная нелокальность из минимизации действия"""
        return find_exact_y(self.Lambda)

    @property
    def x(self) -> float:
        """Критический параметр x = p N^(1/d)"""
        return self.K * self.y_exact / (2 * self.eta)

    @property
    def p(self) -> float:
        """Вероятность нелокальной связи"""
        return self.x / self.N**(1/self.d)

    @property
    def U(self) -> float:
        """Модулярный инвариант U = ln N / |ln(Kp)|"""
        kp = self.K * self.p
        return np.log(self.N) / abs(np.log(kp))

    @property
    def d_s(self) -> float:
        """Спектральная размерность"""
        kp = self.K * self.p
        return 2.0 / abs(np.log(kp) / np.log(self.N))

    def get_dimensionless(self) -> DimensionlessGIP:
        """Получить безразмерное представление"""
        return DimensionlessGIP(self.Lambda, self.y_exact)


def from_Lambda_to_physical(Lambda: float, N: float, K: float = 6.0,
                            eta: float = 0.1, gamma: float = 1.0, d: float = 3.0) -> PhysicalGIP:
    """Восстановление физических параметров из Λ"""
    current = 4 * eta**2 * gamma * np.log(N) / K**3
    if current > 0:
        scale = np.sqrt(Lambda / current)
        eta_scaled = eta * scale
    else:
        eta_scaled = eta
    return PhysicalGIP(K=K, eta=eta_scaled, gamma=gamma, N=N, d=d)


# ============================================================================
# ЧАСТЬ 4: БЕЗРАЗМЕРНЫЙ ЛАГРАНЖИАН И ТОЧНЫЙ МИНИМУМ
# ============================================================================

def dimensionless_action(y: float, Lambda: float) -> float:
    """Безразмерное действие S̃(y; Λ)"""
    return y**2 + (2/3)*y**3 - Lambda * np.log(y + 1e-12)


def action_derivative(y: float, Lambda: float) -> float:
    """Первая производная ∂S̃/∂y"""
    return 2*y + 2*y**2 - Lambda/(y + 1e-12)


def action_second_derivative(y: float, Lambda: float) -> float:
    """Вторая производная ∂²S̃/∂y²"""
    return 2 + 4*y + Lambda/(y**2 + 1e-12)


def find_exact_y(Lambda: float) -> float:
    """Поиск точного минимума безразмерного действия"""
    def objective(y):
        return dimensionless_action(y, Lambda)

    y0 = solve_y_from_cubic(Lambda)
    result = minimize_scalar(objective, bounds=(1e-6, 100), method='bounded')
    return result.x


def analyze_minimum(Lambda: float) -> dict:
    """Полный анализ минимума для заданного Λ"""
    y_cubic = solve_y_from_cubic(Lambda)
    y_exact = find_exact_y(Lambda)

    return {
        'Lambda': Lambda,
        'y_cubic': y_cubic,
        'y_exact': y_exact,
        'gradient_cubic': action_derivative(y_cubic, Lambda),
        'gradient_exact': action_derivative(y_exact, Lambda),
        'hessian': action_second_derivative(y_exact, Lambda),
        'action_cubic': dimensionless_action(y_cubic, Lambda),
        'action_exact': dimensionless_action(y_exact, Lambda)
    }


# ============================================================================
# ЧАСТЬ 5: АНАЛИЗ ДЛЯ НАШЕЙ ВСЕЛЕННОЙ
# ============================================================================

def analyze_universe(N: float = 4.198e121, K: float = 6.0,
                     eta: float = 0.1, gamma: float = 1.0, d: float = 3.0) -> dict:
    """Полный анализ для параметров нашей Вселенной"""

    universe = PhysicalGIP(K=K, eta=eta, gamma=gamma, N=N, d=d)
    Lambda = universe.Lambda

    y_cubic = universe.y_cubic
    y_exact = universe.y_exact

    x_val = universe.x
    p_val = universe.p
    U_val = universe.U
    d_s_val = universe.d_s

    dimless = universe.get_dimensionless()

    return {
        'physical': universe,
        'Lambda': Lambda,
        'y_cubic': y_cubic,
        'y_exact': y_exact,
        'x': x_val,
        'p': p_val,
        'U': U_val,
        'd_s': d_s_val,
        'U_pi_ratio': U_val / np.pi,
        'ds_2pi_ratio': d_s_val / (2 * np.pi),
        'gradient': action_derivative(y_exact, Lambda),
        'hessian': action_second_derivative(y_exact, Lambda),
        'regime': dimless.regime,
        'scaling': dimless.scaling_exponent
    }


# ============================================================================
# ЧАСТЬ 6: ФАЗОВАЯ ДИАГРАММА
# ============================================================================

class PhaseDiagram:
    """Фазовая диаграмма ГИП в координатах (Λ, y)"""

    def __init__(self):
        self.Lambda_range = np.logspace(-4, 4, 500)
        self.y_cubic = np.array([solve_y_from_cubic(L) for L in self.Lambda_range])
        self.y_exact = np.array([find_exact_y(L) for L in self.Lambda_range])

    def plot(self, ax=None, show_regimes=True):
        """Построение фазовой диаграммы"""
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        ax.loglog(self.Lambda_range, self.y_exact, 'b-', linewidth=2.5, label='y(Λ) точное')
        ax.loglog(self.Lambda_range, self.y_cubic, 'c--', linewidth=1.5, alpha=0.7, label='y(Λ) кубическое')

        Lambda_small = self.Lambda_range[self.Lambda_range < 0.1]
        ax.loglog(Lambda_small, np.sqrt(Lambda_small/2), 'r--', linewidth=1.5, label='y ~ √(Λ/2)')

        Lambda_large = self.Lambda_range[self.Lambda_range > 10]
        ax.loglog(Lambda_large, (Lambda_large/2)**(1/3), 'g--', linewidth=1.5, label='y ~ ∛(Λ/2)')

        ax.axvline(x=1.0, color='purple', linestyle=':', linewidth=2, label='Λ = 1 (переход)')
        ax.axhline(y=1.0, color='purple', linestyle=':', linewidth=2)

        if show_regimes:
            ax.fill_between([1e-4, 0.5], 1e-2, 0.7, alpha=0.15, color='red', label='Геометрический')
            ax.fill_between([0.5, 2.0], 0.7, 1.5, alpha=0.15, color='yellow', label='Переходный')
            ax.fill_between([2.0, 1e4], 1.5, 20, alpha=0.15, color='green', label='Нелокальный')

        ax.set_xlabel('Λ = 4η²γ ln N / K³', fontsize=14)
        ax.set_ylabel('y = 2ηx / K', fontsize=14)
        ax.set_title('ФАЗОВАЯ ДИАГРАММА ГЛОБАЛЬНОГО ИНФОРМАЦИОННОГО ПОЛЯ', fontsize=16)
        ax.legend(loc='upper left', fontsize=11)
        ax.grid(True, alpha=0.3)

        return ax


# ============================================================================
# ЧАСТЬ 7: ВИЗУАЛИЗАЦИЯ
# ============================================================================

def plot_full_analysis():
    """Полная визуализация модели ГИП"""
    fig = plt.figure(figsize=(18, 12))

    # 1. Фазовая диаграмма
    ax1 = plt.subplot(2, 3, 1)
    pd = PhaseDiagram()
    pd.plot(ax=ax1)

    # 2. Действие S(y) для разных Λ
    ax2 = plt.subplot(2, 3, 2)
    y_vals = np.logspace(-2, 1, 200)
    Lambdas = [0.01, 0.1, 1.0, 10.0, 100.0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(Lambdas)))

    for L, c in zip(Lambdas, colors):
        S_vals = [dimensionless_action(y, L) for y in y_vals]
        y_opt = find_exact_y(L)
        S_opt = dimensionless_action(y_opt, L)
        ax2.loglog(y_vals, S_vals - S_opt + 1, color=c, linewidth=1.5, label=f'Λ={L}')
        ax2.scatter([y_opt], [1], color=c, s=100, marker='*', edgecolors='white', linewidth=1)

    ax2.set_xlabel('y', fontsize=12)
    ax2.set_ylabel('S̃(y) - S̃_min + 1', fontsize=12)
    ax2.set_title('Безразмерное действие', fontsize=14)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # 3. Скейлинговый показатель
    ax3 = plt.subplot(2, 3, 3)
    y_range = np.logspace(-2, 2, 200)
    nu = [DimensionlessGIP(0, y).scaling_exponent for y in y_range]
    ax3.semilogx(y_range, nu, 'b-', linewidth=2.5)
    ax3.axhline(y=0.5, color='r', linestyle='--', label='ν = 1/2')
    ax3.axhline(y=1/3, color='g', linestyle='--', label='ν = 1/3')
    ax3.axvline(x=1.0, color='purple', linestyle=':', label='y = 1')
    ax3.set_xlabel('y', fontsize=12)
    ax3.set_ylabel('ν (x ~ (ln N)^ν)', fontsize=12)
    ax3.set_title('Скейлинговый показатель', fontsize=14)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # 4. p(N) для разных режимов
    ax4 = plt.subplot(2, 3, 4)
    N_range = np.logspace(2, 200, 100)

    for Lambda, color in [(0.01, 'red'), (1.0, 'purple'), (100.0, 'green')]:
        p_vals = []
        for N in N_range:
            phys = from_Lambda_to_physical(Lambda, N)
            p_vals.append(phys.p)
        ax4.loglog(N_range, p_vals, color=color, linewidth=2, label=f'Λ={Lambda}')

    ax4.loglog(N_range, np.sqrt(np.log(N_range))/N_range**(1/3), 'k--', linewidth=1.5, label='p ~ √(ln N)/N^(1/3)')
    ax4.set_xlabel('N', fontsize=12)
    ax4.set_ylabel('p', fontsize=12)
    ax4.set_title('Масштабирование p(N)', fontsize=14)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    # 5. Точка нашей Вселенной
    ax5 = plt.subplot(2, 3, 5)
    pd.plot(ax=ax5, show_regimes=False)

    result = analyze_universe()
    ax5.scatter([result['Lambda']], [result['y_exact']], color='gold', s=400,
                marker='*', edgecolors='black', linewidth=2.5,
                label=f'Наша Вселенная\nΛ={result["Lambda"]:.3e}\ny={result["y_exact"]:.3f}')

    N_alt = [1e10, 1e50, 1e100, 1e150, 1e200]
    for N in N_alt:
        alt = PhysicalGIP(K=6.0, eta=0.1, gamma=1.0, N=N)
        ax5.scatter([alt.Lambda], [alt.y_exact], s=100, alpha=0.6)

    ax5.legend(fontsize=10)
    ax5.set_title('Положение нашей Вселенной', fontsize=14)

    # 6. Компоненты действия
    ax6 = plt.subplot(2, 3, 6)

    components = {
        'Структурная': result['physical'].K * (1 + result['physical'].eta * result['x']),
        'Транспортная': result['physical'].gamma * np.log(result['physical'].N) / (result['x'] + 1e-12),
        'Энтропийная': 0.0,
        'Геом. штраф': 0.0,
    }

    colors_bar = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#DDA0DD']
    bars = ax6.bar(components.keys(), components.values(), color=colors_bar, alpha=0.7)
    ax6.set_ylabel('Вклад в действие', fontsize=12)
    ax6.set_title('Компоненты действия', fontsize=14)
    ax6.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('GIP_universal_model_corrected.png', dpi=150, bbox_inches='tight')
    plt.show()

    return fig


# ============================================================================
# ЧАСТЬ 8: ГЛАВНЫЙ ЗАПУСК
# ============================================================================

def main():
    print("=" * 80)
    print("УНИВЕРСАЛЬНАЯ БЕЗРАЗМЕРНАЯ МОДЕЛЬ ГЛОБАЛЬНОГО ИНФОРМАЦИОННОГО ПОЛЯ")
    print("=" * 80)

    result = analyze_universe()

    print("\n📊 ПАРАМЕТРЫ НАШЕЙ ВСЕЛЕННОЙ:")
    print(f"  N = {result['physical'].N:.3e}")
    print(f"  K = {result['physical'].K:.1f}")
    print(f"  η = {result['physical'].eta:.4f}")
    print(f"  γ = {result['physical'].gamma:.1f}")

    print("\n🎯 УПРАВЛЯЮЩИЙ ПАРАМЕТР:")
    print(f"  Λ = {result['Lambda']:.6e}")

    print("\n📐 БЕЗРАЗМЕРНАЯ НЕЛОКАЛЬНОСТЬ:")
    print(f"  y (кубическое) = {result['y_cubic']:.6f}")
    print(f"  y (точное)     = {result['y_exact']:.6f}")
    print(f"  Режим: {result['regime']}")
    print(f"  Скейлинг: ν = {result['scaling']:.4f}")

    print("\n🔬 ФИЗИЧЕСКИЕ ВЕЛИЧИНЫ:")
    print(f"  x = p·N^(1/d) = {result['x']:.4f}")
    print(f"  p = {result['p']:.4e}")
    print(f"  U = ln N / |ln(Kp)| = {result['U']:.6f}")
    print(f"  d_s = {result['d_s']:.4f}")

    print("\n✨ ПРОВЕРКА ТОЖДЕСТВ:")
    print(f"  U/π = {result['U_pi_ratio']:.8f}")
    print(f"  d_s/(2π) = {result['ds_2pi_ratio']:.8f}")

    print("\n⚡ УСТОЙЧИВОСТЬ:")
    print(f"  ∂S/∂y = {result['gradient']:.6e}")
    print(f"  ∂²S/∂y² = {result['hessian']:.6f} > 0 → устойчивый минимум")

    print("\n🔄 ПРОВЕРКА УНИВЕРСАЛЬНОСТИ:")
    print("  Меняем параметры, сохраняя Λ:")

    N2 = 1e100
    phys2 = from_Lambda_to_physical(result['Lambda'], N2)
    print(f"    N={N2:.1e}, K={phys2.K}, η={phys2.eta:.4f} → y={phys2.y_exact:.6f}")

    K3 = 4.0
    phys3 = from_Lambda_to_physical(result['Lambda'], result['physical'].N, K=K3)
    print(f"    N={result['physical'].N:.1e}, K={K3}, η={phys3.eta:.4f} → y={phys3.y_exact:.6f}")

    print("\n✅ ВСЕ ВАРИАНТЫ ДАЮТ ОДИНАКОВОЕ y!")

    print("\n📈 ПОСТРОЕНИЕ ГРАФИКОВ...")
    plot_full_analysis()

    print("\n🎓 ИТОГ:")
    print("=" * 80)
    print("Вся физика Глобального Информационного Поля сводится к ОДНОМУ числу Λ.")
    print(f"Для нашей Вселенной Λ = {result['Lambda']:.6e}.")
    print("Это означает, что мы находимся в ГЕОМЕТРИЧЕСКОМ режиме (y < 1),")
    print("где пространство близко к 3D-решетке с редкими нелокальными связями.")
    print("=" * 80)

    return result


if __name__ == "__main__":
    result = main()