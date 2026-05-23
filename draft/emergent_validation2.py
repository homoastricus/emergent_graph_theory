"""
ВИЗУАЛИЗАЦИЯ ОБЛАСТЕЙ ОПТИМАЛЬНЫХ ПАРАМЕТРОВ p И N
===================================================
Без калибровочных констант. Показываем, при каких p и N
каждая эмерджентная величина максимально близка к CODATA.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from math import log, sqrt, pi, e

# ============================================================================
# ЦЕЛЕВЫЕ ЗНАЧЕНИЯ (CODATA)
# ============================================================================
TARGETS = {
    'c': 2.99792458e8,  # м/с
    'G': 6.67430e-11,  # м³/(кг·с²)
    'm_e': 9.1093837e-31,  # кг
    'l_P': 1.616255e-35,  # м
    't_P': 5.391247e-44,  # с
}

# Номинальные параметры
K = 8.0
p_nom = 1.25e-31
N_nom = 9.702e122


# ============================================================================
# ФУНКЦИЯ ВЫЧИСЛЕНИЯ БЕЗРАЗМЕРНЫХ ЭМЕРДЖЕНТНЫХ ВЕЛИЧИН (БЕЗ α)
# ============================================================================
def compute_dimensionless(p, N):
    """Вычисление безразмерных величин (без калибровки)"""

    lnN = log(N)
    lnK = log(K)
    lnp = log(p)
    lnKp = log(K * p)
    abs_lnKp = abs(lnKp)

    x = lnKp / lnN
    lambda_sq = x ** 2
    U = lnN / abs_lnKp
    f1 = U / pi
    f3 = sqrt(K * p)

    C_clust = 3 * (K - 2) / (4 * (K - 1)) * (1 - p) ** 3
    correction = 1 + (1 - C_clust) / lnN
    hbar_em = (lnK ** 2) / (4 * lambda_sq ** 2 * K ** 2) * correction

    V = f1 ** (2 / 3) * hbar_em ** 3 * lnN ** 2

    # Безразмерные величины (без калибровочных констант)
    c_dimless = (lnN ** 2 * abs_lnKp ** 2) / (lnK ** (1 / 3))

    G_dimless = pi * e ** (-1 / 3) / (V ** (3 / 2))

    m_e_dimless = 8 * f3 * (U ** 6) * (V ** 3) * lnK * N ** (-1 / 3)

    l_P_dimless = (K * p) / (V ** (1 / 3) * lnN)

    t_P_dimless = ((K * p) ** (3 / 2)) / (lambda_sq ** (3 / 2) * lnK ** (1 / 3))

    return {
        'c': c_dimless,
        'G': G_dimless,
        'm_e': m_e_dimless,
        'l_P': l_P_dimless,
        't_P': t_P_dimless,
        'V': V,
        'U': U,
        'lambda': lambda_sq
    }


# ============================================================================
# ВЫЧИСЛЕНИЕ КАЛИБРОВОЧНЫХ МНОЖИТЕЛЕЙ В НОМИНАЛЬНОЙ ТОЧКЕ
# ============================================================================
res_nom = compute_dimensionless(p_nom, N_nom)
calibration = {name: TARGETS[name] / res_nom[name] for name in TARGETS}

print("=" * 90)
print("КАЛИБРОВОЧНЫЕ МНОЖИТЕЛИ (в номинальной точке)")
print("=" * 90)
for name, cal in calibration.items():
    print(f"  {name}: {cal:.6e}")


# ============================================================================
# ФУНКЦИЯ ОШИБКИ (после калибровки)
# ============================================================================
def error(p, N, name):
    """Относительная ошибка для заданной величины"""
    res = compute_dimensionless(p, N)
    val = res[name] * calibration[name]
    target = TARGETS[name]
    return abs(val - target) / target


# ============================================================================
# ПОСТРОЕНИЕ СЕТКИ
# ============================================================================
n_points = 50
p_factors = np.linspace(0.75, 1.25, n_points)  # ±25%
N_factors = np.linspace(0.75, 1.25, n_points)  # ±25%

p_vals = p_nom * p_factors
N_vals = N_nom * N_factors

P_grid, N_grid = np.meshgrid(p_vals, N_vals)

# ============================================================================
# ВЫЧИСЛЕНИЕ ОШИБОК НА СЕТКЕ
# ============================================================================
print("\n" + "=" * 90)
print("ВЫЧИСЛЕНИЕ ОШИБОК НА СЕТКЕ 50x50...")
print("=" * 90)

errors = {name: np.zeros((n_points, n_points)) for name in TARGETS}

for i, p in enumerate(p_vals):
    if i % 10 == 0:
        print(f"  Прогресс: {i}/{n_points}")
    for j, N in enumerate(N_vals):
        for name in TARGETS:
            errors[name][j, i] = error(p, N, name) * 100  # в процентах

# ============================================================================
# ПОИСК ОПТИМАЛЬНЫХ ТОЧЕК ДЛЯ КАЖДОЙ ВЕЛИЧИНЫ
# ============================================================================
print("\n" + "=" * 90)
print("ОПТИМАЛЬНЫЕ ТОЧКИ ДЛЯ КАЖДОЙ ВЕЛИЧИНЫ")
print("=" * 90)

optimal_points = {}

for name in TARGETS:
    min_idx = np.unravel_index(np.argmin(errors[name]), errors[name].shape)
    opt_p = p_vals[min_idx[1]]
    opt_N = N_vals[min_idx[0]]
    opt_err = errors[name][min_idx]
    opt_p_factor = opt_p / p_nom
    opt_N_factor = opt_N / N_nom

    optimal_points[name] = {
        'p': opt_p, 'N': opt_N,
        'p_factor': opt_p_factor, 'N_factor': opt_N_factor,
        'error': opt_err
    }

    print(f"\n{name}:")
    print(f"  p = {opt_p:.6e} (фактор {opt_p_factor:.4f})")
    print(f"  N = {opt_N:.6e} (фактор {opt_N_factor:.4f})")
    print(f"  Минимальная ошибка = {opt_err:.6f}%")

# ============================================================================
# ВИЗУАЛИЗАЦИЯ
# ============================================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

plot_names = ['c', 'G', 'm_e', 'l_P', 't_P']
plot_titles = ['Скорость света c', 'Гравитационная постоянная G',
               'Масса электрона m_e', 'Планковская длина ℓ_P', 'Планковское время t_P']

for idx, (name, title) in enumerate(zip(plot_names, plot_titles)):
    ax = axes[idx]

    # Тепловая карта ошибок (логарифмическая шкала)
    err_data = errors[name]

    # Ограничиваем для лучшей визуализации
    err_data_clipped = np.clip(err_data, 1e-4, 10)

    im = ax.contourf(N_factors, p_factors, err_data_clipped,
                     levels=np.logspace(-4, 1, 20),
                     cmap='RdYlBu_r', norm='log')

    # Контурные линии
    contour_levels = [0.01, 0.1, 1.0, 5.0]
    ax.contour(N_factors, p_factors, err_data, levels=contour_levels,
               colors='black', linewidths=0.5, alpha=0.7)

    # Оптимальная точка
    opt = optimal_points[name]
    ax.scatter([opt['N_factor']], [opt['p_factor']],
               color='white', s=150, marker='*',
               edgecolors='black', linewidth=2, zorder=10)

    # Номинальная точка
    ax.scatter([1.0], [1.0], color='green', s=80, marker='o',
               edgecolors='white', linewidth=1.5, zorder=10, label='Номинал')

    ax.set_xlabel('N / N_nom', fontsize=11)
    ax.set_ylabel('p / p_nom', fontsize=11)
    ax.set_title(f'{title}\n(мин. ошибка {opt["error"]:.4f}%)', fontsize=12)
    ax.grid(True, alpha=0.2)

    # Добавляем аннотацию с оптимальными параметрами
    ax.text(0.05, 0.95,
            f"p={opt['p_factor']:.3f}, N={opt['N_factor']:.3f}",
            transform=ax.transAxes, fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Цветовая шкала
plt.colorbar(im, ax=axes.ravel().tolist(), label='Ошибка % (log scale)',
             location='right', shrink=0.9)

# Легенда для номинальной точки
handles = [plt.Line2D([0], [0], marker='o', color='w',
                      markerfacecolor='green', markersize=8, label='Номинал'),
           plt.Line2D([0], [0], marker='*', color='w',
                      markerfacecolor='white', markeredgecolor='black',
                      markersize=12, label='Оптимум')]
fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(0.92, 0.92))

plt.tight_layout()
plt.savefig('optimal_parameters_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# СВОДНЫЙ ГРАФИК: ВСЕ ОПТИМАЛЬНЫЕ ТОЧКИ НА ОДНОЙ ПЛОСКОСТИ
# ============================================================================
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# Цвета для разных величин
colors = {'c': 'blue', 'G': 'red', 'm_e': 'green', 'l_P': 'orange', 't_P': 'purple'}
markers = {'c': 'o', 'G': 's', 'm_e': '^', 'l_P': 'v', 't_P': 'D'}

for name in TARGETS:
    opt = optimal_points[name]
    ax.scatter([opt['N_factor']], [opt['p_factor']],
               color=colors[name], s=150, marker=markers[name],
               label=f"{name} ({opt['error']:.4f}%)",
               edgecolors='black', linewidth=1.5, zorder=10)

# Номинальная точка
ax.scatter([1.0], [1.0], color='black', s=200, marker='*',
           label='Номинал', edgecolors='white', linewidth=2, zorder=10)

# Соединительные линии от номинала к оптимумам
for name in TARGETS:
    opt = optimal_points[name]
    ax.plot([1.0, opt['N_factor']], [1.0, opt['p_factor']],
            color=colors[name], linewidth=1, alpha=0.5, linestyle='--')

ax.set_xlabel('N / N_nom', fontsize=13)
ax.set_ylabel('p / p_nom', fontsize=13)
ax.set_title('Оптимальные параметры для разных величин', fontsize=14)
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0.74, 1.26)
ax.set_ylim(0.74, 1.26)

# Добавляем серую область для номинала ±10%
ax.axhspan(0.9, 1.1, alpha=0.1, color='gray')
ax.axvspan(0.9, 1.1, alpha=0.1, color='gray')

plt.tight_layout()
plt.savefig('all_optimal_points.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# ТАБЛИЦА ОПТИМАЛЬНЫХ ПАРАМЕТРОВ
# ============================================================================
print("\n" + "=" * 90)
print("СВОДНАЯ ТАБЛИЦА ОПТИМАЛЬНЫХ ПАРАМЕТРОВ")
print("=" * 90)
print(f"{'Величина':<6} {'p_opt':<18} {'N_opt':<18} {'p/p_nom':<10} {'N/N_nom':<10} {'Ошибка %':<10}")
print("-" * 85)

for name in TARGETS:
    opt = optimal_points[name]
    print(f"{name:<6} {opt['p']:<18.6e} {opt['N']:<18.6e} "
          f"{opt['p_factor']:<10.4f} {opt['N_factor']:<10.4f} {opt['error']:<10.6f}%")

# ============================================================================
# АНАЛИЗ РАЗБРОСА
# ============================================================================
print("\n" + "=" * 90)
print("АНАЛИЗ РАЗБРОСА ОПТИМАЛЬНЫХ ТОЧЕК")
print("=" * 90)

p_factors_opt = [opt['p_factor'] for opt in optimal_points.values()]
N_factors_opt = [opt['N_factor'] for opt in optimal_points.values()]

print(f"\nСтандартное отклонение p_factors: {np.std(p_factors_opt):.6f}")
print(f"Стандартное отклонение N_factors: {np.std(N_factors_opt):.6f}")

# Центр масс оптимальных точек
center_p = np.mean(p_factors_opt)
center_N = np.mean(N_factors_opt)

print(f"\nЦентр масс оптимальных точек:")
print(f"  p/p_nom = {center_p:.4f}")
print(f"  N/N_nom = {center_N:.4f}")

print("\n" + "=" * 90)
print("ГОТОВО!")
print("=" * 90)

# Поиск компромиссной точки
best_total = float('inf')
best_compromise = None

for i, p in enumerate(p_vals):
    for j, N in enumerate(N_vals):
        # Суммарная ошибка (взвешенная)
        total_err = (errors['c'][j, i] + errors['G'][j, i] +
                     errors['m_e'][j, i] + errors['l_P'][j, i] + errors['t_P'][j, i])

        if total_err < best_total:
            best_total = total_err
            best_compromise = (p, N, p / p_nom, N / N_nom)

print("=" * 50)
print("КОМПРОМИССНАЯ ТОЧКА")
print("=" * 50)
print(f"p = {best_compromise[0]:.6e} (фактор {best_compromise[2]:.4f})")
print(f"N = {best_compromise[1]:.6e} (фактор {best_compromise[3]:.4f})")
print(f"Суммарная ошибка = {best_total:.6f}%")