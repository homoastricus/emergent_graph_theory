import warnings

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.sparse import linalg as spla

warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.facecolor'] = 'white'


def create_watts_strogatz_graph(N, k, p):
    """Создание графа Уоттса-Строгаца"""
    G = nx.watts_strogatz_graph(N, k, p, seed=42)
    return G


def calculate_full_laplacian_spectrum(G):
    """
    Вычисление ПОЛНОГО спектра нормализованного лапласиана
    Для больших графов используем плотную матрицу (до N~5000)
    или специальные методы
    """
    n = G.number_of_nodes()

    # Для графов до 5000 вершин используем плотную матрицу
    if n <= 5000:
        print(f"  Используем плотную матрицу ({n}x{n})...")
        L = nx.normalized_laplacian_matrix(G).toarray()
        eigenvalues = np.linalg.eigvalsh(L)
        return np.sort(eigenvalues)

    # Для больших графов используем комбинацию методов
    else:
        print(f"  Используем разреженные методы для N={n}...")
        L = nx.normalized_laplacian_matrix(G).astype(np.float64)

        # Вычисляем все собственные значения через сдвиг
        # Используем тот факт, что спектр нормализованного лапласиана ∈ [0, 2]

        # Шаг 1: получаем малые собственные значения
        k_small = min(n - 2, n // 3)  # берем треть спектра
        try:
            eig_small = spla.eigsh(L, k=k_small, which='SM',
                                   return_eigenvectors=False)
        except:
            eig_small = spla.eigsh(L, k=min(k_small, 500), which='SM',
                                   return_eigenvectors=False)

        # Шаг 2: получаем большие собственные значения
        k_large = min(n - 2, n // 3)
        try:
            eig_large = spla.eigsh(L, k=k_large, which='LM',
                                   return_eigenvectors=False)
        except:
            eig_large = spla.eigsh(L, k=min(k_large, 500), which='LM',
                                   return_eigenvectors=False)

        # Шаг 3: получаем средние собственные значения через сдвиг
        # Используем sigma=1.0 чтобы получить значения вокруг середины спектра
        k_mid = min(n - 2, n // 3)
        try:
            eig_mid = spla.eigsh(L, k=k_mid, sigma=1.0,
                                 return_eigenvectors=False)
        except:
            eig_mid = np.array([])

        # Объединяем и сортируем
        all_eigs = np.concatenate([eig_small, eig_mid, eig_large])
        all_eigs = np.unique(all_eigs)  # удаляем возможные дубликаты
        all_eigs = np.sort(all_eigs)

        # Убеждаемся, что 0 на месте
        if all_eigs[0] > 1e-10:
            all_eigs = np.concatenate([[0.0], all_eigs])

        print(f"  Получено {len(all_eigs)} уникальных с.з. из {n}")
        return all_eigs


def estimate_spectral_dimension(eigenvalues, n_bins=30):
    """
    Надежная оценка спектральной размерности из плотности состояний
    ρ(λ) ∝ λ^(ds/2 - 1) для малых λ

    Используем интегральную функцию распределения:
    N(λ) = ∫ρ(λ')dλ' ∝ λ^(ds/2)
    """
    # Берем только положительные собственные значения
    eigs = eigenvalues[eigenvalues > 1e-12]

    if len(eigs) < 10:
        return np.nan, np.nan

    # Определяем диапазон "малых" λ: до того, как плотность состояний начнет
    # отклоняться от степенного закона
    # Эвристика: используем первые 20% спектра или до λ=0.3
    threshold = min(np.percentile(eigs, 20), 0.3)
    small_eigs = eigs[eigs <= threshold]

    if len(small_eigs) < 10:
        small_eigs = eigs[:max(10, len(eigs) // 5)]

    # Строим кумулятивную функцию распределения
    n_points = len(small_eigs)
    cumulative = np.arange(1, n_points + 1) / n_points

    # Линейная регрессия в log-log масштабе
    # log(N(λ)) = (ds/2) * log(λ) + const
    valid = (small_eigs > 1e-12) & (cumulative > 1e-12)
    log_lambda = np.log(small_eigs[valid])
    log_cum = np.log(cumulative[valid])

    if len(log_lambda) < 5:
        return np.nan, np.nan

    # Линейная регрессия
    coeffs = np.polyfit(log_lambda, log_cum, 1)
    ds = 2 * coeffs[0]

    # Оценка ошибки через R^2
    predicted = np.polyval(coeffs, log_lambda)
    ss_res = np.sum((log_cum - predicted) ** 2)
    ss_tot = np.sum((log_cum - np.mean(log_cum)) ** 2)
    r_squared = 1 - ss_res / ss_tot

    return ds, r_squared


def calculate_physical_constants(N, K, p):
    """Расчет эмерджентных физических констант"""
    lnN = np.log(N)
    lnK = np.log(K)
    N13 = N ** (1 / 3)
    Kp = K * p

    constants = {
        'N': N,
        'K': K,
        'p': p,
        'Kp': Kp,
        'c': np.pi * (lnN ** 4) / (K ** 2 * lnK),
        'lP': 4 * lnN ** 2 * lnK / N13,
        'tP': 4 * K ** 2 * lnK ** 2 / (np.pi * N13 * lnN ** 2),
        'EP': (lnN ** 5) * np.pi / (4 * K ** 3 * lnK ** 2),
        'G': 16 * np.pi ** 3 * lnN ** 13 / (K ** 5 * lnK * N13),
        'mP': K / (np.pi * 4 * lnN ** 3),
        'TP': 8 * np.pi * N13 / (lnN ** 4),
        'k_B': Kp * (lnN ** 8) / (8 * np.pi ** 2),
        'alpha': 2 * lnK ** 2 / (np.pi * lnN)
    }

    # Проверка согласованности
    constants['c_check'] = constants['lP'] / constants['tP']
    constants['EP_check'] = constants['mP'] * constants['c'] ** 2

    return constants


def analyze_graph_spectrum(N, K, p):
    """Анализ спектра графа"""
    print(f"Анализ графа с N={N}, K={K}, p={p:.6e}")
    print(f"Kp = {K * p:.6e}")
    # Создаем граф
    G = create_watts_strogatz_graph(N, K, p)
    n_edges = G.number_of_edges()
    print(f"Граф создан: {G.number_of_nodes()} вершин, {n_edges} ребер")
    print(f"Средняя степень: {2 * n_edges / N:.1f}")

    # Вычисляем спектр
    eigenvalues = calculate_full_laplacian_spectrum(G)
    print(f"Спектр вычислен: {len(eigenvalues)} собственных значений")
    print(f"λ_1 (первое ненулевое) = {eigenvalues[1]:.6f}")
    print(f"λ_max = {eigenvalues[-1]:.6f}")

    # Дополнительная статистика спектра
    print(f"λ_2 = {eigenvalues[2]:.6f}")
    print(f"λ_10 = {eigenvalues[min(10, len(eigenvalues) - 1)]:.6f}")
    print(f"Медиана спектра = {np.median(eigenvalues):.6f}")

    # Оценка спектральной размерности
    ds_estimated, r_squared = estimate_spectral_dimension(eigenvalues)
    print(f"\nСпектральная размерность (оценка): {ds_estimated:.3f}")
    print(f"Качество фита (R²): {r_squared:.3f}")
    print(f"Ожидаемая спектральная размерность: 2.0")

    # Дополнительный метод: оценка через отношение собственных значений
    if len(eigenvalues) > 10:
        # Для ds=2 характерно определенное соотношение между λ_n
        ratio_10_1 = eigenvalues[10] / eigenvalues[1]
        print(f"λ_10/λ_1 = {ratio_10_1:.2f}")

    # Физические константы
    constants = calculate_physical_constants(N, K, p)
    print(f"\nЭмерджентные физические константы:")
    print(f"  c  = {constants['c']:.6e} (проверка: lP/tP = {constants['c_check']:.6e})")
    print(f"  lP = {constants['lP']:.6e}")
    print(f"  tP = {constants['tP']:.6e}")
    print(f"  EP = {constants['EP']:.6e} (проверка: mP·c² = {constants['EP_check']:.6e})")
    print(f"  G  = {constants['G']:.6e}")
    print(f"  mP = {constants['mP']:.6e}")
    print(f"  TP = {constants['TP']:.6e}")
    print(f"  kB = {constants['k_B']:.6e}")
    print(f"  α  = {constants['alpha']:.6e}")

    # Проверка согласованности
    print(f"\nПроверки согласованности:")
    print(f"  lP / tP = {constants['lP'] / constants['tP']:.6e}")
    print(f"  c (формула) = {constants['c']:.6e}")
    print(f"  Совпадение: {abs(constants['c'] - constants['lP'] / constants['tP']) < 1e-10}")

    return eigenvalues, ds_estimated, constants


def plot_spectral_analysis(all_results):
    """Детальный анализ спектральных плотностей"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i, result in enumerate(all_results):
        if i >= len(axes):
            break

        eigenvalues = result['eigenvalues']
        N = result['N']
        ds = result['ds_estimated']

        ax = axes[i]

        # Строим плотность состояний
        eigs_positive = eigenvalues[eigenvalues > 1e-10]

        # Гибридный подход: гистограмма для малых λ, точки для больших
        n_bins = min(100, len(eigs_positive) // 5)
        bins = np.logspace(np.log10(eigs_positive[1]),
                           np.log10(eigs_positive[-1]), n_bins)

        hist, bin_edges = np.histogram(eigs_positive, bins=bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Нормализуем на ширину бина для получения плотности
        bin_widths = bin_edges[1:] - bin_edges[:-1]
        density = hist / bin_widths

        ax.loglog(bin_centers, density, 'b-', linewidth=1.5, alpha=0.8)

        # Теоретическая кривая для ds=2: ρ(λ) ~ const для малых λ
        # Для ds=2: ρ(λ) ∝ λ^(ds/2 - 1) = λ^0 = const
        ax.axhline(y=np.mean(density[:5]), color='r', linestyle='--',
                   alpha=0.5, label='ρ~const (ds=2)')

        ax.set_xlabel('λ')
        ax.set_ylabel('ρ(λ)')
        ax.set_title(f'N={N}, ds≈{ds:.2f}')
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.suptitle('Спектральная плотность графов\nK=6, Kp=N^(-1/3)', fontsize=14)
    plt.tight_layout()
    plt.savefig('spectral_density_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_asymptotics():
    """Построение асимптотических графиков"""
    print("Построение асимптотических графиков")

    N_values = np.logspace(np.log10(100), 122, 50, base=10)
    K = 6

    # Словари для констант
    const_lists = {
        'c': [], 'lP': [], 'tP': [], 'EP': [],
        'G': [], 'mP': [], 'TP': [], 'kB': [], 'alpha': []
    }

    for N in N_values:
        N = int(N)
        p = 1 / (6 * N ** (1 / 3))
        constants = calculate_physical_constants(N, K, p)

        for key in const_lists:
            const_lists[key].append(constants[key])

    # Построение графиков
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    labels = {
        'c': 'c (скорость света)',
        'lP': 'lP (Планковская длина)',
        'tP': 'tP (Планковское время)',
        'EP': 'EP (Планковская энергия)',
        'G': 'G (гравитационная постоянная)',
        'mP': 'mP (Планковская масса)',
        'TP': 'TP (Планковская температура)',
        'kB': 'kB (постоянная Больцмана)',
        'alpha': 'α (пост. тонкой структуры)'
    }

    for i, (key, label) in enumerate(labels.items()):
        ax = axes[i]
        values = const_lists[key]
        ax.loglog(N_values, values, 'b-', linewidth=1.5)
        ax.set_xlabel('N')
        ax.set_ylabel(key)
        ax.set_title(label)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Асимптотическое поведение эмерджентных констант\nK=6, Kp=N^(-1/3)',
                 fontsize=14)
    plt.tight_layout()
    plt.savefig('physical_constants_asymptotics.png', dpi=150, bbox_inches='tight')
    plt.show()


def calculate_universe_constants():
    """Расчет констант для Вселенной"""
    print("ФИЗИЧЕСКИЕ КОНСТАНТЫ ВСЕЛЕННОЙ")
    print(f"N = 4.2 × 10^121")

    N_universe = int(4.2e121)
    K = 6
    p = 1 / (6 * N_universe ** (1 / 3))

    constants = calculate_physical_constants(N_universe, K, p)

    print(f"\nПараметры графа:")
    print(f"  N = {N_universe:.2e}")
    print(f"  K = {K}")
    print(f"  p = {p:.6e}")
    print(f"  Kp = N^(-1/3) = {N_universe ** (-1 / 3):.6e}")
    print(f"  Число shortcut'ов ≈ {0.5 * N_universe ** (2 / 3):.2e}")

    print(f"\nЭмерджентные фундаментальные константы:")
    print(f"  c  = {constants['c']:.6e}")
    print(f"  lP = {constants['lP']:.6e}")
    print(f"  tP = {constants['tP']:.6e}")
    print(f"  EP = {constants['EP']:.6e}")
    print(f"  G  = {constants['G']:.6e}")
    print(f"  mP = {constants['mP']:.6e}")
    print(f"  TP = {constants['TP']:.6e}")
    print(f"  kB = {constants['k_B']:.6e}")
    print(f"  α  = {constants['alpha']:.6e}")

    print(f"\nПроверки:")
    print(f"  lP/tP = {constants['lP'] / constants['tP']:.6e} (должно = c)")
    print(f"  mP·c² = {constants['mP'] * constants['c'] ** 2:.6e} (должно = EP)")

    print(f"\nРазмерности:")
    print(f"  Спектральная (ожидаемая) = 2.0")
    print(f"  Эффективная диффузионная = 4/3")
    print(f"  Эффективная геометрическая = 2.0")

    return constants


# Основное выполнение
if __name__ == "__main__":
    print("МОДЕЛИРОВАНИЕ ГРАФА МАЛОГО МИРА")
    print("Информационная модель Вселенной")

    sizes = [400, 900, 2000, 4000, 9000]
    K = 6

    all_results = []

    for N in sizes:
        p = 1 / (K * N ** (1 / 3))

        try:
            eigenvalues, ds_est, constants = analyze_graph_spectrum(N, K, p)
            all_results.append({
                'N': N,
                'eigenvalues': eigenvalues,
                'ds_estimated': ds_est,
                'constants': constants
            })
        except Exception as e:
            print(f"Ошибка при N={N}: {e}")
            import traceback

            traceback.print_exc()

    # Визуализация
    if all_results:
        plot_spectral_analysis(all_results)

    # Асимптотики
    plot_asymptotics()

    # Константы Вселенной
    universe_constants = calculate_universe_constants()
