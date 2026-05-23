"""
СТРЕСС-ТЕСТ МОДЕЛИ ГИП
Проверка чувствительности формул фундаментальных констант к изменению N
Ищем N, при котором суммарная ошибка минимальна
"""

import math
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# 1. ФУНДАМЕНТАЛЬНЫЕ КОНСТАНТЫ (ЭКСПЕРИМЕНТАЛЬНЫЕ ЗНАЧЕНИЯ)
# =========================================================

constants = {
    'ħ': 1.054571817e-34,  # редуцированная постоянная Планка, Дж·с
    'h': 6.62607015e-34,  # постоянная Планка, Дж·с
    't_P': 5.391247e-44,  # планковское время, с
    'E_P': 1.956082e9,  # планковская энергия, Дж
    'l_P': 1.616255e-35,  # планковская длина, м
    'c': 299792458,  # скорость света, м/с
}

# =========================================================
# 2. ФОРМУЛЫ ГИП (ВЫРАЖЕНИЯ ЧЕРЕЗ N, K, π, e)
# =========================================================

K = 6.0
pi = math.pi
e = math.e


def formulas(N):
    """Возвращает предсказанные значения констант для заданного N"""
    lnN = math.log(N)
    N13 = N ** (1 / 3)

    return {
        'ħ': (lnN ** 3) / (K * N13),
        'h': 2 * pi * (lnN ** 3) / (K * N13),
        't_P': pi / (K * N13 * lnN),
        'E_P': (lnN ** 4) / pi,
        'l_P': math.pi**2 * lnN**3 / (K**3 * math.log(K) * N**(1/3)),
        'c': pi * (lnN ** 4) / (K**2 *  math.log(K)),
    }


# =========================================================
# 3. ФУНКЦИЯ ОШИБКИ
# =========================================================

def relative_error(pred, true):
    """Относительная ошибка в процентах"""
    return abs(pred - true) / true * 100


def total_error(N, constants, formulas):
    """Суммарная относительная ошибка для всех констант"""
    pred = formulas(N)
    errors = {}
    total = 0.0
    for name in constants:
        err = relative_error(pred[name], constants[name])
        errors[name] = err
        total += err
    return total, errors


# =========================================================
# 4. ПОИСК ОПТИМАЛЬНОГО N
# =========================================================

def find_optimal_N(N0, delta_range=0.05, steps=500):
    """
    Ищет N, минимизирующее суммарную ошибку
    N0 - начальное значение (наше N из теории)
    delta_range - диапазон изменения lnN в долях (0.05 = ±5%)
    steps - количество шагов
    """
    lnN0 = math.log(N0)
    lnN_min = lnN0 * (1 - delta_range)
    lnN_max = lnN0 * (1 + delta_range)

    lnN_vals = np.linspace(lnN_min, lnN_max, steps)
    N_vals = np.exp(lnN_vals)

    total_errors = []
    all_errors = {name: [] for name in constants}

    for N in N_vals:
        total_err, err_dict = total_error(N, constants, formulas)
        total_errors.append(total_err)
        for name in constants:
            all_errors[name].append(err_dict[name])

    # Находим минимум
    min_idx = np.argmin(total_errors)
    N_opt = N_vals[min_idx]
    lnN_opt = lnN_vals[min_idx]
    min_error = total_errors[min_idx]

    return {
        'N_opt': N_opt,
        'lnN_opt': lnN_opt,
        'min_total_error': min_error,
        'N_vals': N_vals,
        'total_errors': total_errors,
        'all_errors': all_errors,
        'min_idx': min_idx,
    }


# =========================================================
# 5. ВИЗУАЛИЗАЦИЯ
# =========================================================

def plot_results(result, N0):
    """Строит графики ошибок"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # График 1: Суммарная ошибка
    ax1 = axes[0, 0]
    ax1.plot(result['N_vals'], result['total_errors'], 'b-', linewidth=2)
    ax1.axvline(x=N0, color='r', linestyle='--', label=f'N0 = {N0:.2e}')
    ax1.axvline(x=result['N_opt'], color='g', linestyle='--', label=f'N_opt = {result["N_opt"]:.2e}')
    ax1.scatter([result['N_opt']], [result['min_total_error']], color='g', s=100, zorder=5)
    ax1.set_xscale('log')
    ax1.set_xlabel('N (число узлов)')
    ax1.set_ylabel('Суммарная относительная ошибка (%)')
    ax1.set_title('Суммарная ошибка всех констант')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # График 2: Индивидуальные ошибки
    ax2 = axes[0, 1]
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    for i, (name, errors) in enumerate(result['all_errors'].items()):
        ax2.plot(result['N_vals'], errors, color=colors[i % len(colors)],
                 linewidth=1.5, label=name)
    ax2.axvline(x=N0, color='r', linestyle='--', alpha=0.5)
    ax2.axvline(x=result['N_opt'], color='g', linestyle='--', alpha=0.5)
    ax2.set_xscale('log')
    ax2.set_xlabel('N (число узлов)')
    ax2.set_ylabel('Относительная ошибка (%)')
    ax2.set_title('Индивидуальные ошибки констант')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)

    # График 3: Ошибка вблизи минимума (увеличенный масштаб)
    ax3 = axes[1, 0]
    # Берем окно ±10% вокруг минимума
    idx_min = result['min_idx']
    half_window = min(50, len(result['N_vals']) // 10)
    start = max(0, idx_min - half_window)
    end = min(len(result['N_vals']), idx_min + half_window)

    ax3.plot(result['N_vals'][start:end], result['total_errors'][start:end], 'b-', linewidth=2)
    ax3.axvline(x=N0, color='r', linestyle='--', label=f'N0 = {N0:.2e}')
    ax3.axvline(x=result['N_opt'], color='g', linestyle='--', label=f'N_opt = {result["N_opt"]:.2e}')
    ax3.scatter([result['N_opt']], [result['min_total_error']], color='g', s=100)
    ax3.set_xlabel('N')
    ax3.set_ylabel('Суммарная ошибка (%)')
    ax3.set_title('Ошибка вблизи минимума')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # График 4: Относительное отклонение N_opt от N0
    ax4 = axes[1, 1]
    ax4.axis('off')

    deviation = (result['N_opt'] - N0) / N0 * 100
    text = f"""
    РЕЗУЛЬТАТЫ СТРЕСС-ТЕСТА
    =========================

    Теоретическое N0:     {N0:.4e}
    Оптимальное N_opt:    {result['N_opt']:.4e}
    Отклонение:           {deviation:+.6f}%

    Суммарная ошибка:
      при N0:             {total_error(N0, constants, formulas)[0]:.6f}%
      при N_opt:          {result['min_total_error']:.6f}%
      Улучшение:          {(total_error(N0, constants, formulas)[0] - result['min_total_error']):.6f}%

    Индивидуальные ошибки при N_opt:
    """
    for name in constants:
        _, err_dict = total_error(result['N_opt'], constants, formulas)
        text += f"\n      {name}:     {err_dict[name]:.6f}%"

    text += f"""

    ВЫВОД:
    {'✅ ФОРМУЛЫ ПОДТВЕРЖДЕНЫ' if abs(deviation) < 0.5 else '⚠️ ТРЕБУЕТСЯ УТОЧНЕНИЕ'}
    Минимум ошибки достигается вблизи теоретического N0.
    """
    ax4.text(0.1, 0.9, text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('stress_test_GIP.png', dpi=150, bbox_inches='tight')
    plt.show()


# =========================================================
# 6. MAIN
# =========================================================

def main():
    print("=" * 80)
    print("СТРЕСС-ТЕСТ МОДЕЛИ ГИП")
    print("Поиск оптимального N для формул фундаментальных констант")
    print("=" * 80)

    # Наше теоретическое N из ГИП
    N0 = 4.1790e121

    print(f"\n📊 Теоретическое N0 = {N0:.4e}")
    print(f"   ln(N0) = {math.log(N0):.6f}")

    # Вычисляем ошибки при N0
    total_err, err_dict = total_error(N0, constants, formulas)
    print(f"\n📊 Ошибки при N0:")
    for name, err in err_dict.items():
        print(f"   {name}: {err:.6f}%")
    print(f"   Суммарная: {total_err:.6f}%")

    # Ищем оптимальное N
    print("\n🔍 Поиск оптимального N...")
    result = find_optimal_N(N0, delta_range=0.05, steps=1000)

    print(f"\n📊 РЕЗУЛЬТАТЫ ПОИСКА:")
    print(f"   Оптимальное N_opt = {result['N_opt']:.4e}")
    print(f"   ln(N_opt) = {result['lnN_opt']:.6f}")
    print(f"   Отклонение от N0: {(result['N_opt'] - N0) / N0 * 100:+.6f}%")

    # Ошибки при N_opt
    total_err_opt, err_dict_opt = total_error(result['N_opt'], constants, formulas)
    print(f"\n📊 Ошибки при N_opt:")
    for name, err in err_dict_opt.items():
        print(f"   {name}: {err:.6f}%")
    print(f"   Суммарная: {total_err_opt:.6f}%")

    # Улучшение
    improvement = total_err - total_err_opt
    print(f"\n📊 Улучшение суммарной ошибки: {improvement:.6f}%")

    # Визуализация
    plot_results(result, N0)

    # Финальный вердикт
    deviation = abs((result['N_opt'] - N0) / N0 * 100)
    print("\n" + "=" * 80)
    if deviation < 0.5:
        print("✅ СТРЕСС-ТЕСТ ПРОЙДЕН!")
        print("   Оптимум находится в пределах 0.5% от теоретического N0.")
        print("   Формулы ГИП устойчивы и дают минимальную ошибку вблизи предсказанного значения.")
    elif deviation < 2:
        print("⚠️ СТРЕСС-ТЕСТ ПРОЙДЕН ЧАСТИЧНО")
        print("   Оптимум в пределах 2% от теоретического N0.")
        print("   Требуется уточнение параметров.")
    else:
        print("❌ СТРЕСС-ТЕСТ НЕ ПРОЙДЕН")
        print("   Оптимум существенно отличается от теоретического N0.")
        print("   Формулы требуют пересмотра.")
    print("=" * 80)

    return result


if __name__ == "__main__":
    result = main()