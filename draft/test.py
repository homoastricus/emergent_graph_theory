"""
ПОЛНОЕ КОНКУРЕНТНОЕ ДЕЙСТВИЕ + РЕАЛЬНАЯ СПЕКТРАЛЬНАЯ ЭНТРОПИЯ
=============================================================
Проверка гипотезы: минимум полного действия даёт p ≈ 0.0527
и при этом p выполняется аттрактор (ln(Kp) + 1/(1-p))^p ≈ e.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
from scipy.optimize import minimize_scalar
from dataclasses import dataclass
from typing import Tuple, Dict, List
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# КОНФИГУРАЦИЯ
# ============================================================================

@dataclass
class Config:
    N: int = 1000  # число узлов
    K: int = 8  # локальная связность
    p_range: Tuple[float, float] = (0.01, 0.20)  # расширенный диапазон
    n_p_points: int = 40  # точек для сканирования
    n_eigenvalues: int = 100  # собственных значений для энтропии
    seed: int = 42

    # Веса компонент действия
    w_elastic: float = 1.0  # упругость
    w_curvature: float = 0.5  # кривизна
    w_flow: float = 0.3  # поток
    w_entropy: float = -0.2  # энтропия (отрицательный вес — максимизация)
    w_clustering: float = 0.4  # кластеризация


# ============================================================================
# ПОСТРОЕНИЕ ГРАФА
# ============================================================================

def build_watts_strogatz(N: int, K: int, p: float, seed: int) -> nx.Graph:
    """Построение графа Watts-Strogatz."""
    G = nx.watts_strogatz_graph(n=N, k=K, p=p, seed=seed)
    for u, v in G.edges():
        G[u][v]['weight'] = 1.0
    return G


# ============================================================================
# КОМПОНЕНТЫ ДЕЙСТВИЯ
# ============================================================================

def compute_elastic_action(G: nx.Graph, K: int) -> float:
    """
    Упругость: отклонение средней степени от K.
    S_elastic = (mean_degree - K)^2
    """
    degrees = [d for _, d in G.degree()]
    mean_deg = np.mean(degrees)
    return (mean_deg - K) ** 2


def compute_curvature_action(G: nx.Graph, p: float, N: int) -> float:
    """
    Кривизна: через спектральную щель Лапласиана.
    S_curv = lambda_1 (чем меньше, тем более "плоское" пространство)
    """
    A = nx.adjacency_matrix(G, weight='weight')
    degrees = np.array(A.sum(axis=1)).flatten()
    degrees[degrees < 1e-10] = 1.0

    L = diags(degrees) - A  # ненормализованный Лапласиан

    try:
        lambda_1 = eigsh(L, k=1, which='SM', return_eigenvectors=False)[0]
        return lambda_1
    except:
        return 1.0


def compute_flow_action(G: nx.Graph, p: float) -> float:
    """
    Информационный поток: обратный диаметр.
    S_flow = 1 / diameter (чем меньше диаметр, тем лучше поток)
    """
    try:
        diameter = nx.diameter(G)
        return 1.0 / diameter if diameter > 0 else 1.0
    except:
        return 0.0


def compute_spectral_entropy(G: nx.Graph, k: int = 100) -> float:
    """
    Реальная спектральная энтропия через нормализованный Лапласиан.
    S = -sum(lambda_tilde * ln(lambda_tilde))
    """
    N = G.number_of_nodes()
    A = nx.adjacency_matrix(G, weight='weight')
    degrees = np.array(A.sum(axis=1)).flatten()
    degrees[degrees < 1e-10] = 1.0

    D_inv_sqrt = diags(1.0 / np.sqrt(degrees))
    I = diags(np.ones(N))
    L_norm = I - D_inv_sqrt @ A @ D_inv_sqrt

    try:
        eigenvalues = eigsh(L_norm, k=min(k, N - 2), which='SM', return_eigenvectors=False)
        eigenvalues = eigenvalues[eigenvalues > 1e-15]
    except:
        return 0.0

    if len(eigenvalues) == 0:
        return 0.0

    Z = np.sum(eigenvalues)
    lambda_tilde = eigenvalues / Z
    return -np.sum(lambda_tilde * np.log(lambda_tilde + 1e-15))


def compute_clustering_action(G: nx.Graph, p: float) -> float:
    """
    Кластеризация: средний коэффициент кластеризации.
    S_clust = -C (минус, потому что хотим максимизировать)
    """
    C = nx.average_clustering(G)
    return -C


# ============================================================================
# ПОЛНОЕ ДЕЙСТВИЕ
# ============================================================================

def total_action(G: nx.Graph, p: float, config: Config) -> float:
    """
    Полное конкурентное действие.
    S_total = w_elastic * S_elastic + w_curv * S_curv + w_flow * S_flow
              + w_entropy * S_entropy + w_clust * S_clust
    """
    S_elastic = compute_elastic_action(G, config.K)
    S_curv = compute_curvature_action(G, p, config.N)
    S_flow = compute_flow_action(G, p)
    S_entropy = compute_spectral_entropy(G, config.n_eigenvalues)
    S_clust = compute_clustering_action(G, p)

    total = (config.w_elastic * S_elastic +
             config.w_curvature * S_curv +
             config.w_flow * S_flow +
             config.w_entropy * S_entropy +
             config.w_clustering * S_clust)

    return total


# ============================================================================
# АТТРАКТОР (ИСПРАВЛЕННЫЙ)
# ============================================================================

def attractor_value(K: float, p: float, N: float) -> float:
    """
    Исправленная формула аттрактора: (ln(Kp) + 1/(1-p))^p.
    Работает для p ∈ (0, 1).
    """
    if p <= 0 or p >= 1.0:
        return 0.0

    Kp = K * p
    if Kp <= 0:
        return 0.0

    try:
        val = np.log(Kp) + 1.0 / (1.0 - p)
        if val <= 0:
            return 0.0
        return val ** p
    except:
        return 0.0


# ============================================================================
# СКАНИРОВАНИЕ И ОПТИМИЗАЦИЯ
# ============================================================================

def scan_and_optimize(config: Config) -> Dict:
    """
    Сканирование по p и поиск минимума полного действия.
    """
    p_values = np.linspace(config.p_range[0], config.p_range[1], config.n_p_points)

    actions = []
    attractors = []
    entropies = []
    elastic_terms = []
    curv_terms = []
    flow_terms = []
    clust_terms = []

    print("=" * 70)
    print("СКАНИРОВАНИЕ ПОЛНОГО ДЕЙСТВИЯ")
    print("=" * 70)
    print(f"N = {config.N}, K = {config.K}")
    print(f"Веса: elastic={config.w_elastic}, curv={config.w_curvature}, "
          f"flow={config.w_flow}, entropy={config.w_entropy}, clust={config.w_clustering}")
    print()

    for p in p_values:
        G = build_watts_strogatz(config.N, config.K, p, config.seed)

        S_total = total_action(G, p, config)
        S_ent = compute_spectral_entropy(G, config.n_eigenvalues)
        S_el = compute_elastic_action(G, config.K)
        S_cu = compute_curvature_action(G, p, config.N)
        S_fl = compute_flow_action(G, p)
        S_cl = compute_clustering_action(G, p)

        attr = attractor_value(config.K, p, config.N)

        actions.append(S_total)
        attractors.append(attr)
        entropies.append(S_ent)
        elastic_terms.append(S_el)
        curv_terms.append(S_cu)
        flow_terms.append(S_fl)
        clust_terms.append(S_cl)

        if len(p_values) <= 20 or p % 0.02 < 0.001:
            print(f"p = {p:.4f}: S_total = {S_total:.4f}, attr = {attr:.4f}")

    # Поиск минимума действия
    def action_to_minimize(p):
        G = build_watts_strogatz(config.N, config.K, p, config.seed)
        return total_action(G, p, config)

    result = minimize_scalar(
        action_to_minimize,
        bounds=config.p_range,
        method='bounded',
        options={'xatol': 1e-4}
    )

    p_opt = result.x
    G_opt = build_watts_strogatz(config.N, config.K, p_opt, config.seed)
    S_opt = total_action(G_opt, p_opt, config)
    attr_opt = attractor_value(config.K, p_opt, config.N)

    # Поиск p, где аттрактор ≈ e
    attr_array = np.array(attractors)
    valid_attr = attr_array > 0
    if np.any(valid_attr):
        idx_e = np.argmin(np.abs(attr_array[valid_attr] - np.e))
        p_e = p_values[valid_attr][idx_e]
    else:
        p_e = 0.0

    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТЫ")
    print("=" * 70)
    print(f"p_opt (минимум действия) = {p_opt:.6f}")
    print(f"p_e (аттрактор ≈ e) = {p_e:.6f}")
    print(f"Исходное p из статьи = 0.0527")
    print(f"\nСравнение:")
    print(f"  |p_opt - 0.0527| = {abs(p_opt - 0.0527):.6f}")
    print(f"  |p_e - 0.0527| = {abs(p_e - 0.0527):.6f}")
    print(f"  Аттрактор при p_opt = {attr_opt:.4f}")
    print(f"  e = {np.e:.4f}")

    return {
        'p_values': p_values,
        'actions': np.array(actions),
        'attractors': np.array(attractors),
        'entropies': np.array(entropies),
        'elastic_terms': np.array(elastic_terms),
        'curv_terms': np.array(curv_terms),
        'flow_terms': np.array(flow_terms),
        'clust_terms': np.array(clust_terms),
        'p_opt': p_opt,
        'p_e': p_e,
        'attr_opt': attr_opt,
        'S_opt': S_opt
    }


# ============================================================================
# ВИЗУАЛИЗАЦИЯ
# ============================================================================

def plot_full_results(data: Dict, config: Config):
    """Визуализация всех компонент действия."""
    p_values = data['p_values']

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # 1. Полное действие
    ax1 = axes[0, 0]
    ax1.plot(p_values, data['actions'], 'k-', linewidth=2, label='S_total')
    ax1.axvline(x=data['p_opt'], color='r', linestyle='--',
                label=f"p_opt = {data['p_opt']:.4f}")
    ax1.axvline(x=0.0527, color='g', linestyle=':', label='p_статья = 0.0527')
    ax1.set_xlabel('p')
    ax1.set_ylabel('S_total')
    ax1.set_title('Полное действие')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Компоненты действия
    ax2 = axes[0, 1]
    ax2.plot(p_values, data['elastic_terms'], label='Elastic', alpha=0.7)
    ax2.plot(p_values, data['curv_terms'], label='Curvature', alpha=0.7)
    ax2.plot(p_values, data['flow_terms'], label='Flow', alpha=0.7)
    ax2.plot(p_values, -config.w_entropy * data['entropies'], label='-Entropy', alpha=0.7)
    ax2.plot(p_values, -config.w_clustering * data['clust_terms'], label='-Clustering', alpha=0.7)
    ax2.axvline(x=data['p_opt'], color='r', linestyle='--', alpha=0.5)
    ax2.set_xlabel('p')
    ax2.set_ylabel('Компоненты')
    ax2.set_title('Компоненты действия')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)

    # 3. Аттрактор
    ax3 = axes[0, 2]
    ax3.plot(p_values, data['attractors'], 'g-', linewidth=2)
    ax3.axhline(y=np.e, color='r', linestyle='--', label=f'e = {np.e:.4f}')
    ax3.axvline(x=data['p_opt'], color='b', linestyle=':', alpha=0.7)
    ax3.axvline(x=0.0527, color='orange', linestyle=':', alpha=0.7)
    ax3.set_xlabel('p')
    ax3.set_ylabel('(ln(Kp) + 1/(1-p))^p')
    ax3.set_title('Аттрактор')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Спектральная энтропия
    ax4 = axes[1, 0]
    ax4.plot(p_values, data['entropies'], 'purple', linewidth=2)
    ax4.axvline(x=data['p_opt'], color='r', linestyle='--')
    ax4.set_xlabel('p')
    ax4.set_ylabel('S_entropy')
    ax4.set_title('Спектральная энтропия')
    ax4.grid(True, alpha=0.3)

    # 5. Корреляция: действие vs аттрактор
    ax5 = axes[1, 1]
    valid = data['attractors'] > 0
    ax5.scatter(data['attractors'][valid], data['actions'][valid],
                c=p_values[valid], cmap='viridis', s=30)
    ax5.axvline(x=np.e, color='r', linestyle='--', alpha=0.5)
    ax5.set_xlabel('Аттрактор')
    ax5.set_ylabel('S_total')
    ax5.set_title('Действие vs Аттрактор')
    ax5.grid(True, alpha=0.3)
    plt.colorbar(ax5.collections[0], ax=ax5, label='p')

    # 6. Сводка
    ax6 = axes[1, 2]
    ax6.axis('off')

    # Оценка успеха
    p_opt_close = abs(data['p_opt'] - 0.0527) < 0.01
    attr_close = abs(data['attr_opt'] - np.e) < 0.5

    summary = f"""
    ПОЛНОЕ ДЕЙСТВИЕ + АТТРАКТОР
    ──────────────────────────
    N = {config.N}, K = {config.K}

    p_opt = {data['p_opt']:.6f}
    p_статья = 0.0527
    |Δp| = {abs(data['p_opt'] - 0.0527):.6f}

    Аттрактор при p_opt = {data['attr_opt']:.4f}
    e = {np.e:.4f}
    |Δattr| = {abs(data['attr_opt'] - np.e):.4f}

    РЕЗУЛЬТАТ:
    """

    if p_opt_close and attr_close:
        summary += "\n✅ ПОЛНОЕ ПОДТВЕРЖДЕНИЕ!\n"
        summary += "   p_opt ≈ 0.0527 И аттрактор ≈ e"
    elif p_opt_close:
        summary += "\n⚠️ p_opt ≈ 0.0527, но аттрактор ≠ e"
    elif attr_close:
        summary += "\n⚠️ Аттрактор ≈ e, но p_opt ≠ 0.0527"
    else:
        summary += "\n❌ Гипотеза не подтверждена"

    ax6.text(0.1, 0.5, summary, fontsize=11, family='monospace',
             verticalalignment='center', transform=ax6.transAxes)

    plt.tight_layout()
    plt.savefig('full_action_attractor.png', dpi=150)
    plt.show()


# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    print("=" * 70)
    print("ПОЛНОЕ КОНКУРЕНТНОЕ ДЕЙСТВИЕ + РЕАЛЬНАЯ ЭНТРОПИЯ + АТТРАКТОР")
    print("=" * 70)

    # Конфигурация с оптимальными весами
    config = Config(
        N=1000, K=8,
        p_range=(0.02, 0.18),
        n_p_points=50,
        w_elastic=1.0,
        w_curvature=0.5,
        w_flow=0.3,
        w_entropy=-0.15,  # отрицательный — максимизируем энтропию
        w_clustering=0.4
    )

    data = scan_and_optimize(config)
    plot_full_results(data, config)

    print("\n" + "=" * 70)
    print("ИТОГОВЫЙ ВЫВОД")
    print("=" * 70)

    if abs(data['p_opt'] - 0.0527) < 0.01:
        print("✅ МИНИМУМ ПОЛНОГО ДЕЙСТВИЯ ДАЁТ p ≈ 0.0527!")
        print("   Конкуренция упругости, кривизны, потока, энтропии и кластеризации")
        print("   выделяет то самое значение p!")
    else:
        print(f"⚠️ Минимум действия даёт p = {data['p_opt']:.4f}")
        print("   Требуется подстройка весов компонент.")

    if abs(data['attr_opt'] - np.e) < 0.5:
        print("✅ АТТРАКТОР ПРИ p_opt БЛИЗОК К e!")
        print("   (ln(Kp) + 1/(1-p))^p ≈ e выполняется.")
    else:
        print(f"⚠️ Аттрактор при p_opt = {data['attr_opt']:.4f}, а e = {np.e:.4f}")


if __name__ == "__main__":
    main()