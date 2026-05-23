"""
ФИНАЛЬНАЯ ФИЗИЧЕСКАЯ МОДЕЛЬ: ВЗАИМОДЕЙСТВИЯ КАК СПЕКТРАЛЬНЫЕ РЕЖИМЫ
С РАЗВЕДЕННЫМИ ПАРАМЕТРАМИ И ШТРАФОМ НА КОРРЕЛЯЦИЮ
"""

import numpy as np
from scipy.sparse import csr_matrix, diags, lil_matrix, block_diag
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigvalsh
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# ЧАСТЬ 1: ГРАФ С РАЗВЕДЕННЫМИ ПАРАМЕТРАМИ КАНАЛОВ

class PhysicalGraphImproved:
    """
    Граф с сильно разведенными параметрами каналов.

    Параметры подобраны из физических соображений:
    - Гравитация: очень редкие, очень длинные связи
    - Сильное: плотные, очень короткие связи
    - Слабое: промежуточная плотность и длина
    - ЭМ: редкие, длинные связи (но не такие экстремальные как гравитация)
    """

    def __init__(self, N=1400, K=8, use_complex=True):
        self.N = N
        self.K = K
        self.use_complex = use_complex

        # РАЗВЕДЕННЫЕ ПАРАМЕТРЫ КАНАЛОВ
        # Формат: (p, xi, описание)
        self.channel_params = [
            (0.000005, 50.0, 'Гравитация'),   # очень редкие, очень длинные
            (0.5,      0.5,  'Сильное'),      # плотные, очень короткие
            (0.03,     5.0,  'Слабое'),       # промежуточные
            (0.002,    15.0, 'ЭМ')            # редкие, длинные
        ]

        self.p_channels = [p for p, _, _ in self.channel_params]
        self.xi_channels = [xi for _, xi, _ in self.channel_params]
        self.channel_names = [name for _, _, name in self.channel_params]
        self.n_channels = len(self.channel_params)

        self.local_laplacian = None
        self.channel_laplacians = []
        self.is_generated = False

    def generate(self):
        """Генерация графа с сильно разведенными параметрами."""
        # 1. Локальный лапласиан
        local_adj = self._generate_local_adjacency()
        self.local_laplacian = self._adj_to_laplacian(local_adj)

        # 2. Канальные лапласианы (БЕЗ индивидуальной нормировки!)
        self.channel_laplacians = []
        for a in range(self.n_channels):
            p = self.p_channels[a]
            xi = self.xi_channels[a]
            adj_a = self._generate_channel_adjacency_fast(p, xi)
            L_a = self._adj_to_laplacian(adj_a)
            # ВАЖНО: не нормируем индивидуально! Нормировка будет общей.
            self.channel_laplacians.append(L_a)

        # 3. ЕДИНАЯ нормировка всех лапласианов
        self._unify_normalization()

        self.is_generated = True
        return self.channel_laplacians

    def _unify_normalization(self):
        """Единая нормировка для всех лапласианов."""
        # Вычисляем средний след по всем каналам
        traces = []
        for L in [self.local_laplacian] + self.channel_laplacians:
            if self.use_complex:
                traces.append(np.real(L.diagonal().sum()))
            else:
                traces.append(L.diagonal().sum())

        mean_trace = np.mean(traces)

        # Нормируем все лапласианы на средний след
        if mean_trace > 1e-10:
            self.local_laplacian = self.local_laplacian / (self.local_laplacian.diagonal().sum() / mean_trace)
            for i in range(len(self.channel_laplacians)):
                trace_i = traces[i+1]
                if trace_i > 1e-10:
                    self.channel_laplacians[i] = self.channel_laplacians[i] / (trace_i / mean_trace)

    def _generate_local_adjacency(self):
        """Локальная решетка (кольцо)."""
        data, rows, cols = [], [], []
        for i in range(self.N):
            for j in range(1, self.K // 2 + 1):
                neighbor = (i + j) % self.N
                rows.extend([i, neighbor])
                cols.extend([neighbor, i])
                data.extend([1.0, 1.0])
        return csr_matrix((data, (rows, cols)), shape=(self.N, self.N))

    def _generate_channel_adjacency_fast(self, p, xi):
        """Быстрая генерация через сэмплинг."""
        adj = lil_matrix((self.N, self.N), dtype=complex if self.use_complex else float)

        # Число рёбер масштабируется с p и xi
        n_edges = int(self.N * p * max(1, xi) * 2)
        n_edges = max(n_edges, self.N // 4)

        for _ in range(n_edges):
            i = np.random.randint(0, self.N)
            dist = int(np.random.exponential(xi)) + 1
            j = (i + dist) % self.N

            if i != j:
                if self.use_complex:
                    phase = np.exp(1j * np.random.uniform(0, 2 * np.pi))
                    adj[i, j] = phase
                    adj[j, i] = np.conj(phase)
                else:
                    adj[i, j] = 1.0
                    adj[j, i] = 1.0

        return adj.tocsr()

    def _adj_to_laplacian(self, adj):
        """Преобразование в нормированный лапласиан."""
        if self.use_complex:
            degrees = np.array(np.abs(adj).sum(axis=1)).flatten()
        else:
            degrees = np.array(adj.sum(axis=1)).flatten()

        degrees = np.maximum(degrees, 1e-6)
        D_inv_sqrt = diags(1.0 / np.sqrt(degrees))
        L = diags(np.ones(self.N)) - D_inv_sqrt @ adj @ D_inv_sqrt
        L = L + 1e-8 * diags(np.ones(self.N))
        return L

    def compute_total_laplacian(self, weights):
        """Полный лапласиан = Σ w_a L_a."""
        if not self.is_generated:
            self.generate()

        w_local, w_channels = weights[0], weights[1:]
        L_total = w_local * self.local_laplacian
        for w, L_a in zip(w_channels, self.channel_laplacians):
            L_total = L_total + w * L_a
        return L_total

    def compute_spectrum(self, weights, k=50):
        """Спектр полного лапласиана."""
        L = self.compute_total_laplacian(weights)
        return self._safe_eigsh(L, k)

    def compute_all_channel_spectra(self, k=50):
        """Спектры всех каналов."""
        if not self.is_generated:
            self.generate()
        spectra = []
        for L_a in self.channel_laplacians:
            spectra.append(self._safe_eigsh(L_a, k))
        return spectra

    def _safe_eigsh(self, L, k, maxiter=5000, tol=1e-5):
        """Безопасное вычисление собственных значений."""
        try:
            return eigsh(L, k=min(k, self.N-2), which='SM',
                        return_eigenvectors=False, maxiter=maxiter, tol=tol)
        except:
            if self.N <= 2000:
                if self.use_complex:
                    return np.real(eigvalsh(L.toarray()))[:k]
                else:
                    return eigvalsh(L.toarray())[:k]
            else:
                return np.zeros(k)


# ЧАСТЬ 2: ОПТИМИЗАЦИЯ СО ШТРАФОМ НА КОРРЕЛЯЦИЮ
def compute_operator_invariants(spectrum, use_log=True):
    """Вычисление инвариантов оператора."""
    if use_log:
        spec = np.log(spectrum + 1e-10)
    else:
        spec = spectrum
    return {
        'mean': np.mean(spec),
        'var': np.var(spec),
        'gap': spectrum[1] - spectrum[0] if len(spectrum) > 1 else 0
    }


def compute_channel_correlation(channel_spectra, use_log=True):
    """Вычисление средней корреляции между каналами."""
    if use_log:
        X = np.array([np.log(s + 1e-10) for s in channel_spectra])
    else:
        X = np.array(channel_spectra)

    corr_matrix = np.corrcoef(X)
    # Средняя абсолютная корреляция (без диагонали)
    n = len(channel_spectra)
    if n > 1:
        mean_corr = np.mean(np.abs(corr_matrix[np.triu_indices(n, 1)]))
    else:
        mean_corr = 0
    return mean_corr, corr_matrix


def find_optimal_weights_with_anticorrelation(graph, k=50, n_iter=300,
                                               lambda_anticorr=10.0, use_log=True):
    """
    Поиск оптимальных весов со ШТРАФОМ НА КОРРЕЛЯЦИЮ.
    Цель: минимизировать ошибку + максимизировать независимость каналов.
    """

    channel_spectra = graph.compute_all_channel_spectra(k)

    # Целевые инварианты
    target = {'mean': 0.0, 'var': 1.0, 'gap': 0.01}

    def objective(weights_raw):
        # Нормируем веса
        weights = np.abs(weights_raw) / np.sum(np.abs(weights_raw))

        # Полный спектр
        total_spectrum = graph.compute_spectrum(weights, k)
        total_inv = compute_operator_invariants(total_spectrum, use_log)

        # Ошибка по инвариантам
        error = 0.0
        error += (total_inv['mean'] - target['mean'])**2 * 10
        error += (total_inv['var'] - target['var'])**2
        error += max(0, 0.01 - total_inv['gap'])**2 * 100

        # Энтропия весов (поощряем разнообразие)
        entropy = -np.sum(weights * np.log(weights + 1e-10))
        error -= entropy * 0.01

        # 🔥 ШТРАФ НА КОРРЕЛЯЦИЮ МЕЖДУ КАНАЛАМИ
        # Хотим, чтобы взвешенные каналы были независимы
        weighted_spectra = []
        for i, spec in enumerate(channel_spectra):
            weighted_spectra.append(weights[i+1] * spec)  # i+1 пропускает локальный

        mean_corr, _ = compute_channel_correlation(weighted_spectra, use_log)
        error += lambda_anticorr * mean_corr  # Штраф за корреляцию

        return error

    n_components = graph.n_channels + 1
    x0 = np.ones(n_components) / n_components
    bounds = [(1e-6, None)] * n_components

    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': n_iter})

    optimal_weights = np.abs(result.x) / np.sum(np.abs(result.x))
    return optimal_weights, result.fun



# ЧАСТЬ 3: АНАЛИЗ И ВИЗУАЛИЗАЦИЯ
def analyze_improved_graph(N=1400, K=8, use_complex=True, n_runs=3, lambda_anticorr=10.0):
    """Полный анализ улучшенной модели."""
    print("=" * 80)
    print("ФИНАЛЬНАЯ МОДЕЛЬ: РАЗВЕДЕННЫЕ ПАРАМЕТРЫ + АНТИКОРРЕЛЯЦИЯ")
    print("=" * 80)

    graph = PhysicalGraphImproved(N, K, use_complex=use_complex)
    graph.generate()

    print(f"\nПараметры: N = {N}, K = {K}, U(1) фазы: {use_complex}")
    print(f"Штраф на корреляцию: λ = {lambda_anticorr}")

    print("\nПАРАМЕТРЫ КАНАЛОВ:")
    for i, name in enumerate(graph.channel_names):
        print(f"  {name}: p = {graph.p_channels[i]:.6f}, ξ = {graph.xi_channels[i]:.1f}")

    # 1. Спектры каналов
    channel_spectra = graph.compute_all_channel_spectra(k=50)

    print("\n" + "-" * 40)
    print("СТАТИСТИКА КАНАЛЬНЫХ СПЕКТРОВ (log)")
    print("-" * 40)
    for i, name in enumerate(graph.channel_names):
        spec = channel_spectra[i]
        log_spec = np.log(spec + 1e-10)
        print(f"  {name}: mean(log λ) = {np.mean(log_spec):.4f}, std(log λ) = {np.std(log_spec):.4f}")

    # 2. Корреляция каналов ДО оптимизации
    mean_corr_before, corr_matrix_before = compute_channel_correlation(channel_spectra)
    print(f"\nСредняя корреляция каналов (до оптимизации): {mean_corr_before:.4f}")

    # 3. Оптимальные веса
    print("\n" + "-" * 40)
    print("ПОИСК ОПТИМАЛЬНЫХ ВЕСОВ (с антикорреляцией)")
    print("-" * 40)

    all_weights = []
    for run in range(n_runs):
        weights, error = find_optimal_weights_with_anticorrelation(
            graph, k=50, n_iter=300, lambda_anticorr=lambda_anticorr
        )
        all_weights.append(weights)

    mean_weights = np.mean(all_weights, axis=0)
    std_weights = np.std(all_weights, axis=0)

    names = ['Локальный'] + graph.channel_names
    print(f"\n  Оптимальные веса (среднее по {n_runs} запускам):")
    for i, name in enumerate(names):
        print(f"    {name}: {mean_weights[i]:.6f} ± {std_weights[i]:.6f}")

    # 4. Корреляция ПОСЛЕ оптимизации (со взвешенными каналами)
    weighted_spectra = []
    for i, spec in enumerate(channel_spectra):
        weighted_spectra.append(mean_weights[i+1] * spec)
    mean_corr_after, corr_matrix_after = compute_channel_correlation(weighted_spectra)

    print(f"\nСредняя корреляция каналов (после оптимизации): {mean_corr_after:.4f}")
    print(f"Уменьшение корреляции: {mean_corr_before - mean_corr_after:.4f}")

    # 5. PCA
    X_log = np.array([np.log(s + 1e-10) for s in channel_spectra])
    pca = PCA()
    pca.fit(X_log)

    print("\n" + "-" * 40)
    print("PCA ПО КАНАЛАМ")
    print("-" * 40)
    print(f"  Объясненная дисперсия: {pca.explained_variance_ratio_}")
    eff_dim = np.sum(pca.explained_variance_ratio_ > 0.05)
    print(f"  Эффективная размерность: {eff_dim}")

    # 6. Визуализация
    visualize_improved(graph, mean_weights, channel_spectra, pca,
                      corr_matrix_before, corr_matrix_after,
                      mean_corr_before, mean_corr_after, eff_dim)

    return mean_weights, pca, (mean_corr_before, mean_corr_after)


def visualize_improved(graph, weights, channel_spectra, pca,
                       corr_before, corr_after, corr_val_before, corr_val_after, eff_dim):
    """Визуализация результатов."""

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    colors = ['gray', 'purple', '#d62728', '#2ca02c', '#1f77b4']
    names = ['Локальный'] + graph.channel_names

    # 1. Оптимальные веса
    ax = axes[0, 0]
    ax.bar(range(len(weights)), weights, color=colors[:len(weights)], alpha=0.7)
    ax.set_xticks(range(len(weights)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Вес')
    ax.set_title('Оптимальные веса каналов')
    ax.grid(alpha=0.3)

    # 2. Log-спектры
    ax = axes[0, 1]
    for i, (name, spec) in enumerate(zip(graph.channel_names, channel_spectra)):
        log_spec = np.log(spec + 1e-10)
        ax.plot(log_spec, alpha=0.7, label=name, color=colors[i+1])
    ax.set_xlabel('Индекс')
    ax.set_ylabel('log(λ)')
    ax.set_title('Log-спектры каналов')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 3. Корреляция ДО и ПОСЛЕ
    ax = axes[0, 2]
    x = np.arange(2)
    ax.bar(x, [corr_val_before, corr_val_after], color=['red', 'green'], alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(['До', 'После'])
    ax.set_ylabel('Средняя корреляция')
    ax.set_title('Эффект антикорреляционного штрафа')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.grid(alpha=0.3)

    # 4. Матрица корреляций ДО
    ax = axes[1, 0]
    im = ax.imshow(corr_before, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(graph.channel_names)))
    ax.set_yticks(range(len(graph.channel_names)))
    ax.set_xticklabels(graph.channel_names, rotation=45, ha='right')
    ax.set_yticklabels(graph.channel_names)
    ax.set_title('Корреляция ДО оптимизации')
    plt.colorbar(im, ax=ax)

    # 5. Матрица корреляций ПОСЛЕ
    ax = axes[1, 1]
    im = ax.imshow(corr_after, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(graph.channel_names)))
    ax.set_yticks(range(len(graph.channel_names)))
    ax.set_xticklabels(graph.channel_names, rotation=45, ha='right')
    ax.set_yticklabels(graph.channel_names)
    ax.set_title('Корреляция ПОСЛЕ оптимизации')
    plt.colorbar(im, ax=ax)

    # 6. Сводка
    ax = axes[1, 2]
    ax.axis('off')

    # Определяем иерархию весов
    channel_weights = weights[1:]
    sorted_idx = np.argsort(channel_weights)[::-1]
    hierarchy = " > ".join([f"{graph.channel_names[i]} ({channel_weights[i]:.3f})"
                            for i in sorted_idx])

    summary = f"""
    РЕЗУЛЬТАТЫ УЛУЧШЕННОЙ МОДЕЛИ
    
    U(1) фазы: {'Да' if graph.use_complex else 'Нет'}
    Штраф на корреляцию: λ = 10.0
    
    Корреляция ДО:     {corr_val_before:.4f}
    Корреляция ПОСЛЕ:  {corr_val_after:.4f}
    Уменьшение:        {corr_val_before - corr_val_after:.4f}
    
    Эффективная размерность: {eff_dim}
    
    Иерархия весов:
    {hierarchy}
    
    ВЫВОД:
    {'✅ Каналы стали более независимыми' if corr_val_after < corr_val_before else '⚠️ Корреляция не уменьшилась'}
    """

    ax.text(0.1, 0.5, summary, fontsize=10, family='monospace',
            verticalalignment='center', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('improved_physical_model.png', dpi=150)
    plt.show()


# ЧАСТЬ 4: ЗАПУСК

def main():
    print("СРАВНЕНИЕ: С АНТИКОРРЕЛЯЦИЕЙ И БЕЗ")
    print("=" * 60)

    # Без антикорреляции
    print("\n1. МОДЕЛЬ БЕЗ АНТИКОРРЕЛЯЦИИ (λ=0)")
    print("-" * 40)
    results_no_anti = analyze_improved_graph(N=1400, K=8, use_complex=True, n_runs=2, lambda_anticorr=0.0)

    # С антикорреляцией
    print("\n\n2. МОДЕЛЬ С АНТИКОРРЕЛЯЦИЕЙ (λ=10)")
    print("-" * 40)
    results_with_anti = analyze_improved_graph(N=1400, K=8, use_complex=True, n_runs=2, lambda_anticorr=10.0)

    return results_no_anti, results_with_anti


if __name__ == "__main__":
    results = main()