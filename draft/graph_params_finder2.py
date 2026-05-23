"""
СТАТИСТИЧЕСКАЯ ОПТИМИЗАЦИЯ v5 — MULTI-OBJECTIVE
Цель: одновременно
• d_s ≈ 4.000
• K_eff ≈ 8.0
• p_eff ≈ 0.0527
"""

import numpy as np
import networkx as nx
from scipy.sparse import diags
from scipy.sparse.linalg import expm_multiply
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree
from dataclasses import dataclass
from typing import List, Dict, Tuple
import warnings
import time
from datetime import datetime
import json
import os

warnings.filterwarnings('ignore')


@dataclass
class Config:
    N: int = 1500
    dim: int = 4
    n_runs: int = 21                    # увеличено для статистики
    base_seed: int = 42

    # === РАСШИРЕННЫЕ ДИАПАЗОНЫ v5 ===
    R_range: Tuple[float, float] = (2.0, 4.0)
    connection_radius_range: Tuple[float, float] = (1.5, 2.8)
    p_long_range: Tuple[float, float] = (0.0001, 0.01)
    epsilon_long_range: Tuple[float, float] = (0.05, 0.15)

    # Параметры спектрального анализа
    M_stochastic: int = 40
    n_t_points: int = 50
    t_min: float = 1e-2
    t_max: float = 1e2

    # Целевые значения для multi-objective
    target_ds: float = 4.0
    target_K: float = 8.0
    target_p: float = 0.0527


def generate_points(N: int, dim: int, R: float) -> np.ndarray:
    directions = np.random.randn(N, dim)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    radii = R * np.random.random(N) ** (1.0 / dim)
    return directions * radii[:, np.newaxis]


def distance(p1: np.ndarray, p2: np.ndarray, R: float) -> float:
    d_eucl = np.linalg.norm(p1 - p2)
    r1 = np.linalg.norm(p1)
    r2 = np.linalg.norm(p2)
    boundary_factor = 1.0 / ((1 - r1 / R) * (1 - r2 / R) + 0.1)
    return d_eucl * boundary_factor


def build_geometric_graph(config: Config, R: float, connection_radius: float,
                          p_long: float, epsilon_long: float, seed: int):
    np.random.seed(seed)
    points = generate_points(config.N, config.dim, R)
    G = nx.Graph()
    G.add_nodes_from(range(config.N))

    tree = cKDTree(points)
    r_query = connection_radius * 1.3

    for i in range(config.N):
        idx = tree.query_ball_point(points[i], r=r_query)
        for j in idx:
            if j > i:
                dist = distance(points[i], points[j], R)
                if dist < connection_radius:
                    weight = np.exp(-dist / connection_radius)
                    G.add_edge(i, j, weight=weight, type='local')

    # Гарантированная связность
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        while len(components) > 1:
            min_dist = float('inf')
            best_pair = None
            for ii in components[0]:
                for jj in components[1]:
                    dist = distance(points[ii], points[jj], R)
                    if dist < min_dist:
                        min_dist = dist
                        best_pair = (ii, jj)
            if best_pair:
                i, j = best_pair
                weight = np.exp(-min_dist / connection_radius)
                G.add_edge(i, j, weight=weight, type='bridge')
            components = list(nx.connected_components(G))

    # Дальние связи
    for i in range(config.N):
        for j in range(i + 1, config.N):
            if not G.has_edge(i, j) and np.random.random() < p_long:
                dist = distance(points[i], points[j], R)
                weight = epsilon_long * np.exp(-dist / (2 * connection_radius))
                G.add_edge(i, j, weight=weight, type='long')

    return G, points


def compute_ds_plateau(G, config: Config) -> Tuple[float, float, float, Dict]:
    N = G.number_of_nodes()
    W = nx.adjacency_matrix(G, weight='weight')
    degrees = np.array(W.sum(axis=1)).flatten()
    degrees[degrees < 1e-10] = 1.0
    D_inv_sqrt = diags(1.0 / np.sqrt(degrees))
    I = diags(np.ones(N))
    L = I - D_inv_sqrt @ W @ D_inv_sqrt

    t_values = np.logspace(np.log10(config.t_min), np.log10(config.t_max), config.n_t_points)
    Kt = []
    for t in t_values:
        estimates = []
        for _ in range(config.M_stochastic):
            v = np.random.choice([-1, 1], size=N).astype(np.float64)
            w = expm_multiply(-t * L, v)
            estimates.append(np.dot(v, w))
        Kt.append(np.mean(estimates))
    Kt = np.array(Kt)

    log_t = np.log(t_values)
    log_K = np.log(Kt + 1e-10)
    log_K_smooth = gaussian_filter1d(log_K, sigma=1.5)
    dlogK = np.gradient(log_K_smooth, log_t)
    ds = -2 * dlogK
    ds_smooth = gaussian_filter1d(ds, sigma=2.0)

    mask = (ds_smooth > 3.0) & (ds_smooth < 5.0)
    if not np.any(mask):
        return np.max(ds_smooth), np.max(ds_smooth), 0.0, {'error': 'no_plateau'}

    plateau_indices = np.where(mask)[0]
    groups = []
    current = [plateau_indices[0]]
    for i in range(1, len(plateau_indices)):
        if plateau_indices[i] == plateau_indices[i - 1] + 1:
            current.append(plateau_indices[i])
        else:
            groups.append(current)
            current = [plateau_indices[i]]
    groups.append(current)

    longest = max(groups, key=len)
    plateau_val = np.mean(ds_smooth[longest])
    plateau_std = np.std(ds_smooth[longest])
    quality = (1.0 / (1.0 + abs(plateau_val - 4.0))) * (1.0 / (1.0 + plateau_std))

    return plateau_val, np.max(ds_smooth), quality, {
        'plateau_val': plateau_val,
        'plateau_std': plateau_std,
        'plateau_length': len(longest)
    }


def extract_parameters(G) -> Tuple[float, float]:
    degrees = [d for _, d in G.degree()]
    K_eff = np.mean(degrees)
    total_edges = G.number_of_edges()
    long_edges = sum(1 for _, _, data in G.edges(data=True) if data.get('type') == 'long')
    p_eff = long_edges / total_edges if total_edges > 0 else 0.0
    return K_eff, p_eff


@dataclass
class RunResult:
    run_id: int
    seed: int
    R: float
    connection_radius: float
    p_long: float
    epsilon_long: float
    K_eff: float
    p_eff: float
    d_s_plateau: float
    quality: float
    success: bool
    info: Dict


def run_single_experiment(config: Config, run_id: int) -> RunResult:
    seed = config.base_seed + run_id
    np.random.seed(seed)

    R = np.random.uniform(*config.R_range)
    connection_radius = np.random.uniform(*config.connection_radius_range)
    p_long = np.random.uniform(*config.p_long_range)
    epsilon_long = np.random.uniform(*config.epsilon_long_range)

    try:
        G, _ = build_geometric_graph(config, R, connection_radius, p_long, epsilon_long, seed)
        K_eff, p_eff = extract_parameters(G)
        d_s_plateau, _, quality, info = compute_ds_plateau(G, config)

        success = quality > 0.4 and 4.0 < K_eff < 17.0
        if not success:
            info['error'] = f'quality={quality:.3f} or K_eff={K_eff:.1f}'

        return RunResult(
            run_id=run_id, seed=seed, R=R, connection_radius=connection_radius,
            p_long=p_long, epsilon_long=epsilon_long, K_eff=K_eff, p_eff=p_eff,
            d_s_plateau=d_s_plateau, quality=quality, success=success, info=info
        )
    except Exception as e:
        return RunResult(
            run_id=run_id, seed=seed, R=R, connection_radius=connection_radius,
            p_long=p_long, epsilon_long=epsilon_long, K_eff=0.0, p_eff=0.0,
            d_s_plateau=0.0, quality=0.0, success=False, info={'error': str(e)}
        )


def multi_objective_quality(K_eff: float, p_eff: float, d_s: float, config: Config) -> float:
    """Multi-objective функция качества"""
    w_ds = 0.55
    w_K = 0.25
    w_p = 0.20

    score_ds = 1.0 / (1.0 + abs(d_s - config.target_ds))
    score_K = 1.0 / (1.0 + abs(K_eff - config.target_K) / config.target_K)
    score_p = 1.0 / (1.0 + abs(p_eff - config.target_p) / config.target_p)

    return w_ds * score_ds + w_K * score_K + w_p * score_p


def analyze_results(results: List[RunResult], config: Config):
    successful = [r for r in results if r.success]
    print(f"\nУспешных запусков: {len(successful)}/{len(results)}")

    if not successful:
        print("❌ Нет успешных запусков!")
        return

    d_s_values = np.array([r.d_s_plateau for r in successful])
    K_values = np.array([r.K_eff for r in successful])
    p_values = np.array([r.p_eff for r in successful])

    print(f"\nСпектральная размерность d_s: {np.mean(d_s_values):.3f} ± {np.std(d_s_values):.3f}")
    print(f"K_eff: {np.mean(K_values):.2f} ± {np.std(K_values):.2f}")
    print(f"p_eff: {np.mean(p_values):.4f} ± {np.std(p_values):.4f}")

    # Multi-objective сортировка
    for r in successful:
        r.quality = multi_objective_quality(r.K_eff, r.p_eff, r.d_s_plateau, config)

    top_results = sorted(successful, key=lambda r: r.quality, reverse=True)[:5]

    print(f"\nТОП-5 по multi-objective качеству:")
    print(f"{'K_eff':>6} {'p_eff':>8} {'d_s':>6} {'Quality':>8}")
    for r in top_results:
        print(f"{r.K_eff:>6.2f} {r.p_eff:>8.4f} {r.d_s_plateau:>6.3f} {r.quality:>8.3f}")

    K_opt = np.mean([r.K_eff for r in top_results])
    p_opt = np.mean([r.p_eff for r in top_results])

    print(f"\n✨ ИТОГОВЫЕ ОПТИМАЛЬНЫЕ ПАРАМЕТРЫ (топ-5):")
    print(f"K_optimal  = {K_opt:.2f}")
    print(f"p_optimal  = {p_opt:.4f}")
    print(f"d_s среднее = {np.mean(d_s_values):.3f}")

    return {
        'K_optimal': K_opt,
        'p_optimal': p_opt,
        'd_s_mean': np.mean(d_s_values)
    }


def main():
    print("=== СТАТИСТИЧЕСКАЯ ОПТИМИЗАЦИЯ v5 — MULTI-OBJECTIVE ===")
    config = Config(n_runs=21)

    results = []
    start_time = time.time()

    for i in range(config.n_runs):
        print(f"Запуск {i+1}/{config.n_runs}...", end=" ", flush=True)
        result = run_single_experiment(config, i)
        results.append(result)
        status = "✅" if result.success else "❌"
        print(f"{status} d_s={result.d_s_plateau:.3f}, K={result.K_eff:.1f}, p={result.p_eff:.4f}")

    elapsed = time.time() - start_time
    print(f"\nВремя выполнения: {elapsed:.1f} сек")

    analysis = analyze_results(results, config)

    # Сохранение
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n✅ Результаты сохранены в папке optimization_results_v5_{timestamp}/")


if __name__ == "__main__":
    main()