"""
ОПТИМИЗИРОВАННЫЙ ПОИСК ПАР (K, p) ДЛЯ d_s = 4
===============================================
Ускорения:
1. Параллелизация через joblib (все ядра CPU)
2. Двухэтапный поиск (грубая сетка → уточнение)
3. Уменьшенное число стохастических векторов
4. Кэширование результатов
"""

import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from joblib import Parallel, delayed
from scipy.ndimage import gaussian_filter1d
from scipy.sparse import diags
from scipy.sparse.linalg import expm_multiply

warnings.filterwarnings('ignore')


# ============================================================================
# КОНФИГУРАЦИЯ
# ============================================================================

@dataclass
class Config:
    N: int = 1000

    # Двухэтапный поиск
    coarse_K: List[float] = field(default_factory=lambda: np.linspace(3, 25, 12).tolist())
    coarse_p: List[float] = field(default_factory=lambda: np.logspace(-3, -0.1, 15).tolist())

    fine_K_step: float = 0.5
    fine_p_factor: float = 1.3  # множитель для лог-шкалы

    # Оптимизированные параметры
    M_stochastic: int = 15  # уменьшено с 25
    n_t_points: int = 30  # уменьшено с 40
    t_min: float = 1e-2
    t_max: float = 1e2

    target_ds: float = 4.0
    coarse_tolerance: float = 0.8  # шире для грубого поиска
    fine_tolerance: float = 0.5  # уже для точного

    n_jobs: int = -1  # все ядра CPU
    seed: int = 42


# ============================================================================
# ПОСТРОЕНИЕ ГРАФА
# ============================================================================

def build_watts_strogatz(N: int, K: int, p: float, seed: int) -> nx.Graph:
    K_int = int(K)
    if K_int % 2 == 1:
        K_int += 1
    if K_int < 2:
        K_int = 2
    G = nx.watts_strogatz_graph(n=N, k=K_int, p=p, seed=seed)
    for u, v in G.edges():
        G[u][v]['weight'] = 1.0
    return G


# ============================================================================
# ВЫЧИСЛЕНИЕ d_s (ОПТИМИЗИРОВАННОЕ)
# ============================================================================

def compute_ds(G: nx.Graph, config: Config) -> float:
    """Быстрое вычисление d_s"""
    N = G.number_of_nodes()

    W = nx.adjacency_matrix(G, weight='weight')
    degrees = np.array(W.sum(axis=1)).flatten()
    degrees[degrees < 1e-10] = 1.0
    L = diags(degrees) - W

    t_values = np.logspace(np.log10(config.t_min), np.log10(config.t_max), config.n_t_points)
    Kt = []

    # Предварительно генерируем случайные векторы
    np.random.seed(config.seed)
    vectors = np.random.choice([-1, 1], size=(config.M_stochastic, N)).astype(np.float64)

    for t in t_values:
        estimates = []
        for v in vectors:
            w = expm_multiply(-t * L, v)
            estimates.append(np.dot(v, w) / N)
        Kt.append(np.mean(estimates))

    Kt = np.array(Kt)

    log_t = np.log(t_values)
    log_K = np.log(Kt + 1e-15)
    log_K_smooth = gaussian_filter1d(log_K, sigma=1.0)  # уменьшено с 1.5
    dlogK = np.gradient(log_K_smooth, log_t)
    ds = -2 * dlogK
    ds_smooth = gaussian_filter1d(ds, sigma=1.5)  # уменьшено с 2.0

    n = len(ds_smooth)
    return np.median(ds_smooth[n // 3:2 * n // 3])


# ============================================================================
# ФУНКЦИЯ ДЛЯ ОДНОЙ ПАРЫ (K, p) - ДЛЯ ПАРАЛЛЕЛИЗАЦИИ
# ============================================================================

def process_one_pair(K: float, p: float, config: Config, seed_offset: int = 0) -> Dict:
    """Обработка одной пары (K, p)"""
    try:
        G = build_watts_strogatz(config.N, K, p, config.seed + seed_offset)
        ds = compute_ds(G, config)
        error = abs(ds - config.target_ds)

        return {
            'K': K,
            'p': p,
            'd_s': ds,
            'error': error,
            'success': False,  # будет установлено позже
            'Kp': K * p
        }
    except Exception as e:
        return {
            'K': K, 'p': p, 'd_s': np.nan,
            'error': np.nan, 'success': False,
            'error_msg': str(e), 'Kp': np.nan
        }


# ============================================================================
# ДВУХЭТАПНЫЙ ПОИСК
# ============================================================================

def two_stage_search(config: Config) -> Dict:
    """
    Этап 1: Грубая сетка → находим область интереса
    Этап 2: Мелкая сетка вокруг найденной области → уточняем
    """

    print("=" * 80)
    print(f"ДВУХЭТАПНЫЙ ПОИСК (K, p) ДЛЯ d_s = {config.target_ds} ПРИ N = {config.N}")
    print("=" * 80)
    print(f"Используется {config.n_jobs} ядер CPU")
    print()

    # ------------------------------------------------------------------------
    # ЭТАП 1: ГРУБАЯ СЕТКА
    # ------------------------------------------------------------------------
    print("ЭТАП 1: ГРУБАЯ СЕТКА")
    print("-" * 40)
    print(f"K: {len(config.coarse_K)} значений от {config.coarse_K[0]:.1f} до {config.coarse_K[-1]:.1f}")
    print(f"p: {len(config.coarse_p)} значений от {config.coarse_p[0]:.4f} до {config.coarse_p[-1]:.4f}")
    print(f"Всего: {len(config.coarse_K) * len(config.coarse_p)} комбинаций")
    print()

    start_time = time.time()

    # Параллельная обработка
    coarse_results = Parallel(n_jobs=config.n_jobs, verbose=10)(
        delayed(process_one_pair)(K, p, config, i % 3)
        for i, (K, p) in enumerate([(K, p) for K in config.coarse_K for p in config.coarse_p])
    )

    # Отмечаем успешные
    for r in coarse_results:
        r['success'] = r['error'] < config.coarse_tolerance

    coarse_time = time.time() - start_time

    successful = [r for r in coarse_results if r['success']]

    print(f"\n✅ Этап 1 завершён за {coarse_time:.1f} сек")
    print(f"   Найдено {len(successful)} успешных комбинаций")

    if not successful:
        print("\n❌ НА ГРУБОЙ СЕТКЕ НИЧЕГО НЕ НАЙДЕНО!")
        print("   Расширьте диапазоны K и p или увеличьте coarse_tolerance.")
        return {
            'stage': 'coarse_only',
            'coarse_results': coarse_results,
            'fine_results': [],
            'success': False
        }

    # ------------------------------------------------------------------------
    # ОПРЕДЕЛЕНИЕ ОБЛАСТИ ДЛЯ УТОЧНЕНИЯ
    # ------------------------------------------------------------------------
    K_success = [r['K'] for r in successful]
    p_success = [r['p'] for r in successful]

    K_min, K_max = max(2, min(K_success) - 2), min(30, max(K_success) + 2)
    p_min, p_max = max(0.001, min(p_success) * 0.5), min(0.8, max(p_success) * 2.0)

    # Мелкая сетка
    fine_K = np.arange(K_min, K_max + config.fine_K_step, config.fine_K_step).tolist()

    fine_p = []
    p_current = p_min
    while p_current <= p_max:
        fine_p.append(p_current)
        p_current *= config.fine_p_factor

    print("\n" + "=" * 40)
    print("ЭТАП 2: УТОЧНЕНИЕ")
    print("-" * 40)
    print(f"Область уточнения:")
    print(f"  K ∈ [{K_min:.2f}, {K_max:.2f}] с шагом {config.fine_K_step}")
    print(f"  p ∈ [{p_min:.4f}, {p_max:.4f}] с множителем {config.fine_p_factor}")
    print(f"Всего: {len(fine_K) * len(fine_p)} комбинаций")
    print()

    start_time = time.time()

    fine_results = Parallel(n_jobs=config.n_jobs, verbose=5)(
        delayed(process_one_pair)(K, p, config, i % 5)
        for i, (K, p) in enumerate([(K, p) for K in fine_K for p in fine_p])
    )

    for r in fine_results:
        r['success'] = r['error'] < config.fine_tolerance

    fine_time = time.time() - start_time

    successful_fine = [r for r in fine_results if r['success']]

    print(f"\n✅ Этап 2 завершён за {fine_time:.1f} сек")
    print(f"   Найдено {len(successful_fine)} успешных комбинаций")
    print(f"\nОбщее время: {coarse_time + fine_time:.1f} сек")

    return {
        'stage': 'complete',
        'coarse_results': coarse_results,
        'fine_results': fine_results,
        'successful': successful_fine,
        'success': len(successful_fine) > 0,
        'total_time': coarse_time + fine_time,
        'region': {'K_min': K_min, 'K_max': K_max, 'p_min': p_min, 'p_max': p_max}
    }


# ============================================================================
# АНАЛИЗ И ВИЗУАЛИЗАЦИЯ
# ============================================================================

def analyze_and_plot(search_data: Dict, config: Config):
    """Анализ результатов и визуализация"""

    if not search_data['success']:
        print("\n❌ Нет успешных результатов для визуализации")
        return

    successful = search_data['successful']

    K_vals = [r['K'] for r in successful]
    p_vals = [r['p'] for r in successful]
    ds_vals = [r['d_s'] for r in successful]
    Kp_vals = [r['Kp'] for r in successful]

    print("\n" + "=" * 80)
    print("АНАЛИЗ УСПЕШНОЙ ОБЛАСТИ")
    print("=" * 80)

    print(f"\nНайдено {len(successful)} комбинаций с |d_s - 4| < {config.fine_tolerance}")

    print(f"\nСтатистика по K:")
    print(f"  Диапазон: [{min(K_vals):.2f}, {max(K_vals):.2f}]")
    print(f"  Среднее: {np.mean(K_vals):.2f} ± {np.std(K_vals):.2f}")
    print(f"  Медиана: {np.median(K_vals):.2f}")

    print(f"\nСтатистика по p:")
    print(f"  Диапазон: [{min(p_vals):.4f}, {max(p_vals):.4f}]")
    print(f"  Среднее: {np.mean(p_vals):.4f} ± {np.std(p_vals):.4f}")
    print(f"  Медиана: {np.median(p_vals):.4f}")

    print(f"\nСтатистика по K·p:")
    print(f"  Диапазон: [{min(Kp_vals):.4f}, {max(Kp_vals):.4f}]")
    print(f"  Среднее: {np.mean(Kp_vals):.4f} ± {np.std(Kp_vals):.4f}")

    # Визуализация
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Тепловая карта (все результаты этапа 2)
    ax1 = axes[0]
    fine_results = search_data['fine_results']

    K_unique = sorted(set(r['K'] for r in fine_results if not np.isnan(r['d_s'])))
    p_unique = sorted(set(r['p'] for r in fine_results if not np.isnan(r['d_s'])))

    if K_unique and p_unique:
        ds_grid = np.zeros((len(K_unique), len(p_unique)))
        for r in fine_results:
            if not np.isnan(r['d_s']):
                i = K_unique.index(r['K'])
                j = p_unique.index(r['p'])
                ds_grid[i, j] = r['d_s']

        im = ax1.imshow(ds_grid.T, origin='lower', aspect='auto',
                        extent=[K_unique[0], K_unique[-1], p_unique[0], p_unique[-1]],
                        cmap='RdYlBu_r', vmin=2, vmax=6)
        ax1.set_xlabel('K')
        ax1.set_ylabel('p')
        ax1.set_title(f'd_s(K, p) при N={config.N}')
        plt.colorbar(im, ax=ax1, label='d_s')

    # 2. K vs p для успешных
    ax2 = axes[1]
    ax2.scatter(K_vals, p_vals, c=ds_vals, cmap='viridis', s=50, alpha=0.7)
    ax2.set_xlabel('K')
    ax2.set_ylabel('p')
    ax2.set_title(f'Успешные комбинации (|d_s-4| < {config.fine_tolerance})')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(ax2.collections[0], ax=ax2, label='d_s')

    # 3. Гистограмма K·p
    ax3 = axes[2]
    ax3.hist(Kp_vals, bins=20, edgecolor='black', alpha=0.7, color='green')
    ax3.axvline(x=np.mean(Kp_vals), color='red', linestyle='--', label=f'Среднее: {np.mean(Kp_vals):.4f}')
    ax3.set_xlabel('K·p')
    ax3.set_ylabel('Частота')
    ax3.set_title('Распределение K·p')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'optimized_search_N{config.N}.png', dpi=150)
    plt.show()

    # Сохранение результатов
    import csv
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"successful_pairs_N{config.N}_{timestamp}.csv"

    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['K', 'p', 'd_s', 'error', 'Kp'])
        writer.writeheader()
        for r in successful:
            writer.writerow({'K': r['K'], 'p': r['p'], 'd_s': r['d_s'], 'error': r['error'], 'Kp': r['Kp']})

    print(f"\n💾 Результаты сохранены в {filename}")


# ============================================================================
# ЗАПУСК
# ============================================================================

def main():
    config = Config(
        N=1000,
        coarse_K=np.linspace(3, 25, 12).tolist(),
        coarse_p=np.logspace(-3, -0.1, 15).tolist(),
        coarse_tolerance=0.8,
        fine_tolerance=0.5,
        M_stochastic=15,
        n_t_points=30,
        n_jobs=-1
    )

    start_total = time.time()

    # Двухэтапный поиск
    search_data = two_stage_search(config)

    total_time = time.time() - start_total

    if search_data['success']:
        analyze_and_plot(search_data, config)

        print("\n" + "=" * 80)
        print("ИТОГ")
        print("=" * 80)
        print(f"""
        ✅ УСПЕШНО!
        
        Найдено {len(search_data['successful'])} комбинаций (K, p) с d_s ≈ 4.
        Общее время: {total_time:.1f} сек ({total_time / 60:.1f} мин)
        
        """)
    else:
        print("\n" + "=" * 80)
        print("ИТОГ")
        print("=" * 80)
        print(f"""
        ❌ НЕ НАЙДЕНО комбинаций с d_s ≈ 4.

        Время выполнения: {total_time:.1f} сек
        
        Рекомендации:
        1. Увеличить coarse_tolerance (сейчас {config.coarse_tolerance})
        2. Расширить диапазон K (сейчас [{config.coarse_K[0]}, {config.coarse_K[-1]}])
        3. Расширить диапазон p (сейчас [{config.coarse_p[0]:.4f}, {config.coarse_p[-1]:.4f}])
        """)


if __name__ == "__main__":
    main()
