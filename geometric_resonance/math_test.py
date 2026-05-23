import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.linalg import eigvalsh
from scipy import stats
from collections import Counter
import warnings

warnings.filterwarnings('ignore')

# =========================
# ПАРАМЕТРЫ
# =========================

K = 6


def p_critical(N):
    return 1 / (K * N ** (1 / 3))


# =========================
# СПЕКТР
# =========================

def get_spectrum(G):
    L = nx.laplacian_matrix(G).toarray()
    eigvals = eigvalsh(L)
    return eigvals[1:]  # убираем 0


# =========================
# (A) ФРАКТАЛЬНОСТЬ
# =========================

def spectral_density(eigs, bins=50):
    hist, edges = np.histogram(eigs, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, hist


def analyze_spectrum(eigs, label=""):
    """Детальный анализ спектра"""
    print(f"\n{'=' * 60}")
    print(f"СПЕКТРАЛЬНЫЙ АНАЛИЗ {label}")
    print(f"{'=' * 60}")
    print(f"  Количество собственных значений: {len(eigs)}")
    print(f"  Минимальное λ: {eigs.min():.6f}")
    print(f"  Максимальное λ: {eigs.max():.6f}")
    print(f"  Среднее λ: {eigs.mean():.6f}")
    print(f"  Медиана λ: {np.median(eigs):.6f}")
    print(f"  Стандартное отклонение λ: {eigs.std():.6f}")

    # Асимметрия и эксцесс
    skewness = stats.skew(eigs)
    kurtosis = stats.kurtosis(eigs)
    print(f"  Асимметрия (skewness): {skewness:.6f}")
    print(f"  Эксцесс (kurtosis): {kurtosis:.6f}")

    # Процентили
    percentiles = [10, 25, 50, 75, 90]
    print(f"  Процентили:")
    for p in percentiles:
        print(f"    {p}%: {np.percentile(eigs, p):.6f}")

    # Спектральная щель
    spectral_gap = eigs[1] - eigs[0] if len(eigs) > 1 else 0
    print(f"  Спектральная щель (λ₂ - λ₁): {spectral_gap:.6f}")

    # Оценка фрактальной размерности
    x, y = spectral_density(eigs)
    mask = (x > 0) & (y > 0)
    if mask.sum() > 5:
        logx = np.log(x[mask])
        logy = np.log(y[mask])
        slope, intercept, r_value, p_value, std_err = stats.linregress(logx, logy)
        print(f"  Фрактальная размерность (наклон log-log): {abs(slope):.6f}")
        print(f"  R² линейной регрессии: {r_value ** 2:.6f}")

    return eigs


def plot_fractality(N):
    p = p_critical(N)
    print(f"\n  p_critical({N}) = {p:.6e}")

    G = nx.watts_strogatz_graph(N, K, p)
    print(f"  Граф: {G.number_of_nodes()} вершин, {G.number_of_edges()} ребер")
    print(f"  Средняя степень: {np.mean([d for _, d in G.degree()]):.2f}")
    print(f"  Коэффициент кластеризации: {nx.average_clustering(G):.6f}")

    eigs = get_spectrum(G)
    analyze_spectrum(eigs, f"N={N}")

    x, y = spectral_density(eigs)
    plt.figure(figsize=(10, 6))
    plt.loglog(x, y + 1e-10, 'o', markersize=3, alpha=0.7)
    plt.title(f"Spectral density (log-log), N={N}")
    plt.xlabel("λ")
    plt.ylabel("ρ(λ)")
    plt.grid(True, alpha=0.3)
    plt.show()


# =========================
# (B) ДИНАМИКА (автомат)
# =========================

def graph_dynamics(G, steps=50):
    N = G.number_of_nodes()
    state = np.random.randint(0, 2, size=N)
    history = []

    for _ in range(steps):
        new_state = state.copy()
        for i in range(N):
            neighbors = list(G.neighbors(i))
            if len(neighbors) > 0:
                s = sum(state[j] for j in neighbors)
                new_state[i] = 1 if s > len(neighbors) // 2 else 0
            else:
                new_state[i] = state[i]
        state = new_state
        history.append(state.copy())

    return np.array(history)


def analyze_dynamics(history):
    """Анализ динамики автомата"""
    print(f"\n{'=' * 60}")
    print(f"АНАЛИЗ ДИНАМИКИ АВТОМАТА")
    print(f"{'=' * 60}")
    print(f"  Шагов эволюции: {len(history)}")
    print(f"  Узлов: {history.shape[1]}")

    # Плотность единиц во времени
    density_time = history.mean(axis=1)
    print(f"  Начальная плотность: {density_time[0]:.4f}")
    print(f"  Конечная плотность: {density_time[-1]:.4f}")
    print(f"  Средняя плотность: {density_time.mean():.4f}")
    print(f"  Стабильность конечного состояния: {density_time[-10:].std():.6f}")

    # Сходимость
    changes = (history[1:] != history[:-1]).mean(axis=1)
    print(f"  Последнее изменение (доля узлов): {changes[-1]:.6f}")
    convergence_step = np.where(changes < 0.01)[0]
    if len(convergence_step) > 0:
        print(f"  Сходимость достигнута на шаге: {convergence_step[0]}")
    else:
        print(f"  Сходимость не достигнута за {len(history)} шагов")

    # Устойчивые узлы
    node_activity = (history[1:] != history[:-1]).mean(axis=0)
    stable_nodes = (node_activity < 0.1).sum()
    print(f"  Устойчивых узлов (активность < 10%): {stable_nodes}/{history.shape[1]}")

    # Анализ паттернов в финальном состоянии
    final_state = history[-1]
    runs = []
    current = final_state[0]
    length = 1
    for bit in final_state[1:]:
        if bit == current:
            length += 1
        else:
            runs.append(length)
            current = bit
            length = 1
    runs.append(length)

    print(f"  Количество серий 0/1: {len(runs)}")
    print(f"  Средняя длина серии: {np.mean(runs):.2f}")
    print(f"  Максимальная серия: {max(runs)}")

    return density_time


def plot_automaton(N):
    p = p_critical(N)
    print(f"\n  p_critical({N}) = {p:.6e}")

    G = nx.watts_strogatz_graph(N, K, p)
    hist = graph_dynamics(G, steps=100)
    analyze_dynamics(hist)

    plt.figure(figsize=(12, 8))
    plt.imshow(hist, aspect='auto', cmap='binary', interpolation='nearest')
    plt.title(f"Graph dynamics (automaton), N={N}")
    plt.xlabel("node")
    plt.ylabel("time")
    plt.colorbar(label='State')
    plt.show()


# =========================
# (C) БИФУРКАЦИИ - РАСШИРЕННЫЙ АНАЛИЗ
# =========================

def bifurcation_scan_detailed(N, p_range=(0.3, 2.0), n_points=100):
    """
    Детальное сканирование бифуркаций с высоким разрешением
    """
    p_crit = p_critical(N)
    ps = np.linspace(p_range[0], p_range[1], n_points) * p_crit

    all_densities = []
    all_patterns = []
    all_entropies = []
    all_complexities = []

    print(f"\n  Сканирование {n_points} точек p...")

    for p_idx, p in enumerate(ps):
        G = nx.watts_strogatz_graph(N, K, p)
        hist = graph_dynamics(G, steps=200)

        # Анализ последних 20 шагов для стабильности
        final_states = hist[-20:]
        densities_last = final_states.mean(axis=1)
        density = densities_last.mean()
        density_std = densities_last.std()

        # Финальное состояние (последний шаг)
        final_state = hist[-1]

        # Анализ паттернов
        state_str = ''.join(map(str, final_state))

        # Энтропия Шеннона по блокам размера 3
        blocks = [state_str[i:i + 3] for i in range(0, len(state_str) - 3, 3)]
        block_counts = Counter(blocks)
        total_blocks = len(blocks)
        if total_blocks > 0:
            entropy = -sum((count / total_blocks) * np.log2(count / total_blocks)
                           for count in block_counts.values())
        else:
            entropy = 0

        # Сложность паттернов (уникальные блоки)
        complexity = len(set(blocks)) / max(1, total_blocks)

        all_densities.append(density)
        all_patterns.append(final_state)
        all_entropies.append(entropy)
        all_complexities.append(complexity)

        if p_idx % 20 == 0:
            print(f"    Прогресс: {p_idx}/{n_points} (p/p_crit = {p / p_crit:.3f})", end='\r')

    print(f"\n  Сканирование завершено!")

    return (ps, np.array(all_densities), np.array(all_entropies),
            np.array(all_complexities), all_patterns)


def analyze_vertical_structures(N, ps, densities, entropies, complexities, patterns):
    """
    ПОИСК ВЕРТИКАЛЬНЫХ СТРУКТУР (ШТРИХОВКА) - аналог Busy Beaver
    """
    p_crit = p_critical(N)

    print(f"\n{'=' * 60}")
    print(f"АНАЛИЗ КРИТИЧЕСКИХ СТРУКТУР (ВЕРТИКАЛЬНАЯ ШТРИХОВКА)")
    print(f"{'=' * 60}")

    # 1. Поиск "полосатых" регионов через дисперсию по окну
    window = 10
    density_std = np.array([np.std(densities[max(0, i - window):min(len(densities), i + window)])
                            for i in range(len(densities))])

    threshold = np.mean(density_std) + np.std(density_std)
    high_variance_regions = density_std > threshold

    print(f"\n  🔍 ОБЛАСТИ С ВЫСОКОЙ ДИСПЕРСИЕЙ ПЛОТНОСТИ:")
    print(f"  Порог обнаружения: std > {threshold:.4f}")

    # Группируем последовательные регионы
    regions = []
    in_region = False
    start_idx = 0

    for i, is_high in enumerate(high_variance_regions):
        if is_high and not in_region:
            start_idx = i
            in_region = True
        elif not is_high and in_region:
            regions.append((start_idx, i - 1))
            in_region = False

    if in_region:
        regions.append((start_idx, len(high_variance_regions) - 1))

    print(f"  Найдено регионов с высокой вариативностью: {len(regions)}")

    for idx, (start, end) in enumerate(regions[:5]):
        p_start = ps[start] / p_crit
        p_end = ps[end] / p_crit
        width = p_end - p_start
        avg_std = density_std[start:end + 1].mean()
        avg_entropy = entropies[start:end + 1].mean()
        avg_complexity = complexities[start:end + 1].mean()

        print(f"\n  Регион {idx + 1}:")
        print(f"    p/p_crit: [{p_start:.4f}, {p_end:.4f}] (ширина: {width:.4f})")
        print(f"    Средняя вариативность: {avg_std:.4f}")
        print(f"    Средняя сложность: {avg_complexity:.4f}")
        print(f"    Средняя энтропия: {avg_entropy:.4f} бит")

    # 2. АНАЛИЗ ВЕРТИКАЛЬНЫХ ПАТТЕРНОВ (аналог Busy Beaver)
    print(f"\n  📊 АНАЛИЗ ПАТТЕРНОВ СОСТОЯНИЙ:")

    # Ищем повторяющиеся паттерны через длины серий
    pattern_hashes = []
    for pattern in patterns:
        # Сжимаем паттерн в кортеж длин серий
        runs = []
        if len(pattern) > 0:
            current = pattern[0]
            length = 1
            for bit in pattern[1:]:
                if bit == current:
                    length += 1
                else:
                    runs.append(length)
                    current = bit
                    length = 1
            runs.append(length)
        pattern_hashes.append(tuple(runs[:20]))  # Первые 20 серий

    # Считаем уникальные паттерны
    unique_patterns = Counter(pattern_hashes)
    print(f"  Уникальных паттернов (по сериям): {len(unique_patterns)}")
    print(f"  Топ-5 паттернов (частота):")
    for pattern_hash, count in unique_patterns.most_common(5):
        print(f"    Серии: {list(pattern_hash)[:10]}... : {count} раз")

    # 3. ПОИСК КРИТИЧЕСКИХ ТОЧЕК (пики сложности)
    print(f"\n  🎯 КРИТИЧЕСКИЕ ТОЧКИ (ПИКИ СЛОЖНОСТИ):")

    # Находим локальные максимумы сложности
    complexity_peaks = []
    mean_complexity = np.mean(complexities)
    std_complexity = np.std(complexities)

    for i in range(1, len(complexities) - 1):
        if (complexities[i] > complexities[i - 1] and
                complexities[i] > complexities[i + 1] and
                complexities[i] > mean_complexity + std_complexity):
            complexity_peaks.append(i)

    print(f"  Найдено пиков: {len(complexity_peaks)}")

    for idx, peak_idx in enumerate(complexity_peaks[:10]):
        p_val = ps[peak_idx]
        print(f"\n  Пик {idx + 1}:")
        print(f"    p/p_crit = {p_val / p_crit:.4f}")
        print(f"    Сложность: {complexities[peak_idx]:.4f}")
        print(f"    Энтропия: {entropies[peak_idx]:.4f} бит")
        print(f"    Плотность: {densities[peak_idx]:.4f}")

        # Анализ паттерна в пике
        peak_pattern = patterns[peak_idx]
        runs = []
        current = peak_pattern[0]
        length = 1
        for bit in peak_pattern[1:]:
            if bit == current:
                length += 1
            else:
                runs.append(length)
                current = bit
                length = 1
        runs.append(length)

        print(f"    Серий 0/1: {len(runs)}")
        print(f"    Средняя длина серии: {np.mean(runs):.2f}")
        print(f"    Макс. серия: {max(runs)}")

    return regions, complexity_peaks


def plot_bifurcation_advanced(N):
    """
    Улучшенная визуализация бифуркаций с выделением структур
    """
    ps, densities, entropies, complexities, patterns = bifurcation_scan_detailed(N)
    regions, peaks = analyze_vertical_structures(N, ps, densities, entropies, complexities, patterns)

    p_crit = p_critical(N)

    # Создаем фигуру с тремя подграфиками
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # 1. БИФУРКАЦИОННАЯ ДИАГРАММА
    ax1 = axes[0]

    # Определяем цвета по сложности
    mean_comp = np.mean(complexities)
    std_comp = np.std(complexities)
    colors = ['red' if c > mean_comp + std_comp else 'black' for c in complexities]

    for i, p in enumerate(ps):
        ax1.plot(p / p_crit, densities[i], '.', color=colors[i], alpha=0.5, markersize=2)

    # Выделяем регионы с высокой вариативностью
    for idx, (start, end) in enumerate(regions):
        if idx == 0:
            ax1.axvspan(ps[start] / p_crit, ps[end] / p_crit,
                        alpha=0.2, color='red', label='Высокая вариативность')
        else:
            ax1.axvspan(ps[start] / p_crit, ps[end] / p_crit, alpha=0.2, color='red')

    ax1.set_ylabel('Плотность единиц')
    ax1.set_title(f'Бифуркационная диаграмма (N={N}). Красные точки = высокая сложность')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')

    # 2. ЭНТРОПИЯ
    ax2 = axes[1]
    ax2.plot(ps / p_crit, entropies, 'g-', linewidth=1.5, alpha=0.8)
    ax2.fill_between(ps / p_crit, entropies, alpha=0.2, color='green')

    # Отмечаем пики
    for peak in peaks[:10]:
        ax2.axvline(ps[peak] / p_crit, color='darkgreen', alpha=0.3, linestyle='--', linewidth=0.5)

    ax2.set_ylabel('Энтропия (бит)')
    ax2.set_title('Информационная энтропия состояний')
    ax2.grid(True, alpha=0.3)

    # 3. СЛОЖНОСТЬ
    ax3 = axes[2]
    ax3.plot(ps / p_crit, complexities, 'b-', linewidth=1.5, alpha=0.8)
    ax3.fill_between(ps / p_crit, complexities, alpha=0.2, color='blue')

    # Отмечаем пики сложности
    for peak in peaks[:10]:
        ax3.axvline(ps[peak] / p_crit, color='darkblue', alpha=0.5,
                    linestyle='--', linewidth=0.8)
        ax3.plot(ps[peak] / p_crit, complexities[peak], 'ro', markersize=5)

    ax3.set_xlabel('p / p_critical')
    ax3.set_ylabel('Сложность паттернов')
    ax3.set_title('Структурная сложность (вертикальные структуры)')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 4. ДЕТАЛЬНЫЙ АНАЛИЗ САМОГО СЛОЖНОГО РЕГИОНА
    if peaks:
        # Находим пик с максимальной сложностью
        best_peak_idx = peaks[np.argmax(complexities[peaks])]
        p_best = ps[best_peak_idx]

        print(f"\n{'=' * 60}")
        print(f"ДЕТАЛЬНЫЙ АНАЛИЗ НАИБОЛЕЕ СЛОЖНОГО СОСТОЯНИЯ")
        print(f"{'=' * 60}")

        # Создаем граф для этого p
        G = nx.watts_strogatz_graph(N, K, p_best)
        hist = graph_dynamics(G, steps=200)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Пространственно-временная диаграмма
        ax1 = axes[0, 0]
        im = ax1.imshow(hist[-50:], aspect='auto', cmap='binary', interpolation='nearest')
        ax1.set_title(f'Динамика (последние 50 шагов)\np/p_crit = {p_best / p_crit:.4f}')
        ax1.set_xlabel('Узел')
        ax1.set_ylabel('Время')

        # Гистограмма длин серий
        ax2 = axes[0, 1]
        final_state = hist[-1]
        runs = []
        current = final_state[0]
        length = 1
        for bit in final_state[1:]:
            if bit == current:
                length += 1
            else:
                runs.append(length)
                current = bit
                length = 1
        runs.append(length)

        ax2.hist(runs, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        ax2.set_xlabel('Длина серии')
        ax2.set_ylabel('Частота')
        ax2.set_title(f'Распределение длин серий\nСредняя: {np.mean(runs):.2f}, Макс: {max(runs)}')
        ax2.grid(True, alpha=0.3)

        # Эволюция плотности
        ax3 = axes[1, 0]
        density_evolution = hist.mean(axis=1)
        ax3.plot(density_evolution, 'b-', linewidth=1)
        ax3.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Шаг')
        ax3.set_ylabel('Плотность единиц')
        ax3.set_title('Эволюция плотности во времени')
        ax3.grid(True, alpha=0.3)

        # Финальное состояние (1D паттерн)
        ax4 = axes[1, 1]
        ax4.imshow(final_state.reshape(1, -1), aspect='auto', cmap='binary', interpolation='nearest')
        ax4.set_title(f'Финальное состояние\nЕдиниц: {final_state.sum()}, Нулей: {N - final_state.sum()}')
        ax4.set_xlabel('Узел')
        ax4.set_yticks([])

        plt.tight_layout()
        plt.show()

        print(f"  p/p_crit = {p_best / p_crit:.6f}")
        print(f"  Сложность: {complexities[best_peak_idx]:.6f}")
        print(f"  Энтропия: {entropies[best_peak_idx]:.6f} бит")
        print(f"  Плотность: {densities[best_peak_idx]:.4f}")
        print(f"  Средняя длина серии: {np.mean(runs):.2f}")
        print(f"  Максимальная серия: {max(runs)}")
        print(f"  Количество серий: {len(runs)}")
        print(f"  Финальная плотность: {final_state.mean():.4f}")

        # Анализ стабильности
        last_20 = hist[-20:]
        stability = (last_20 == final_state).all(axis=1).mean()
        print(f"  Стабильность финального состояния: {stability:.2%}")

    return ps, densities, entropies, complexities


# =========================
# (D) SCALING
# =========================

def scaling_test(Ns=[3200, 6400, 12800]):
    print(f"\n{'=' * 60}")
    print(f"SCALING АНАЛИЗ")
    print(f"{'=' * 60}")

    slopes = []
    r_squareds = []
    spectral_gaps = []
    clustering_coeffs = []

    for N in Ns:
        p = p_critical(N)
        G = nx.watts_strogatz_graph(N, K, p)
        eigs = get_spectrum(G)

        # Спектральный анализ
        x, y = spectral_density(eigs)

        mask = (x > np.percentile(x, 20)) & (x < np.percentile(x, 80)) & (y > 0)

        if mask.sum() > 5:
            logx = np.log(x[mask])
            logy = np.log(y[mask])

            slope, intercept, r_value, p_value, std_err = stats.linregress(logx, logy)

            slopes.append(slope)
            r_squareds.append(r_value ** 2)

        else:
            slopes.append(np.nan)
            r_squareds.append(np.nan)

        spectral_gap = eigs[1] - eigs[0] if len(eigs) > 1 else 0
        spectral_gaps.append(spectral_gap)

        clustering = nx.average_clustering(G)
        clustering_coeffs.append(clustering)

        print(f"  N={N:<6} p={p:.6e}")
        print(f"    Фрактальная размерность: {slope:.4f} (R²={r_squareds[-1]:.4f})" if not np.isnan(
            slopes[-1]) else "    Недостаточно данных")
        print(f"    Спектральная щель: {spectral_gap:.6f}")
        print(f"    Кластеризация: {clustering:.6f}")
        print(f"    Средняя степень: {K:.1f}")

    # Проверка скейлинга
    valid_slopes = [s for s in slopes if not np.isnan(s)]
    if valid_slopes:
        print(f"\n  📈 АНАЛИЗ СКЕЙЛИНГА:")
        print(f"  Средний наклон: {np.mean(valid_slopes):.4f}")
        print(f"  STD наклона: {np.std(valid_slopes):.4f}")
        stability = np.std(valid_slopes)
        if stability < 0.05:
            print(f"  Стабильность скейлинга: ОТЛИЧНО ✅")
        elif stability < 0.1:
            print(f"  Стабильность скейлинга: ХОРОШО ✅")
        elif stability < 0.2:
            print(f"  Стабильность скейлинга: ПРИЕМЛЕМО 🟡")
        else:
            print(f"  Стабильность скейлинга: ТРЕБУЕТ ПРОВЕРКИ ⚠️")

    # Зависимость спектральной щели от N
    if len(spectral_gaps) >= 2:
        log_Ns = np.log(Ns)
        log_gaps = np.log(spectral_gaps)
        gap_slope, _, _, _, _ = stats.linregress(log_Ns, log_gaps)
        print(f"  Скейлинг спектральной щели (gap ~ N^{gap_slope:.3f}): {gap_slope:.4f}")

    return slopes


def analyze_graph_structure(G, label=""):
    """Комплексный анализ структуры графа"""
    print(f"\n{'=' * 60}")
    print(f"СТРУКТУРНЫЙ АНАЛИЗ ГРАФА {label}")
    print(f"{'=' * 60}")

    n = G.number_of_nodes()
    m = G.number_of_edges()

    print(f"  Вершин: {n}")
    print(f"  Ребер: {m}")
    print(f"  Плотность: {nx.density(G):.6f}")

    degrees = [d for _, d in G.degree()]
    print(f"  Средняя степень: {np.mean(degrees):.2f}")
    print(f"  Медианная степень: {np.median(degrees):.2f}")
    print(f"  Мин/Макс степень: {np.min(degrees)}/{np.max(degrees)}")
    print(f"  STD степени: {np.std(degrees):.2f}")

    # Компоненты связности
    components = list(nx.connected_components(G))
    print(f"  Компонент связности: {len(components)}")
    if len(components) > 0:
        largest_comp = max(components, key=len)
        print(f"  Размер крупнейшей компоненты: {len(largest_comp)} ({len(largest_comp) / n * 100:.1f}%)")

    # Кластеризация
    clustering = nx.average_clustering(G)
    print(f"  Средний коэффициент кластеризации: {clustering:.6f}")

    # Пути (только если граф связный)
    if nx.is_connected(G):
        avg_path = nx.average_shortest_path_length(G)
        diameter = nx.diameter(G)
        print(f"  Средняя длина пути: {avg_path:.4f}")
        print(f"  Диаметр: {diameter}")

        # Эффективность
        efficiency = nx.global_efficiency(G)
        print(f"  Глобальная эффективность: {efficiency:.6f}")

    # Ассортативность
    assortativity = nx.degree_assortativity_coefficient(G)
    print(f"  Ассортативность по степени: {assortativity:.6f}")

    return G


# =========================
# ГЛАВНЫЙ ЗАПУСК
# =========================

def main():
    print("=" * 60)
    print("КОМПЛЕКСНЫЙ АНАЛИЗ ГРАФА УОТТСА-СТРОГАЦА")
    print("=" * 60)
    print(f"\nПараметры модели:")
    print(f"  K = {K}")
    print(f"  p_critical = 1/(K * N^(1/3))")

    N = 20000

    # ТЕСТ 1: ФРАКТАЛЬНОСТЬ
    print(f"\n{'#' * 60}")
    print(f"# ТЕСТ 1: ФРАКТАЛЬНОСТЬ СПЕКТРА")
    print(f"{'#' * 60}")

    p = p_critical(N)
    G = nx.watts_strogatz_graph(N, K, p)
    analyze_graph_structure(G, f"при p_critical(N={N})")
    plot_fractality(N)

    # ТЕСТ 2: КЛЕТОЧНЫЙ АВТОМАТ
    print(f"\n{'#' * 60}")
    print(f"# ТЕСТ 2: КЛЕТОЧНЫЙ АВТОМАТ НА ГРАФЕ")
    print(f"{'#' * 60}")
    plot_automaton(N)

    # ТЕСТ 3: БИФУРКАЦИОННАЯ ДИАГРАММА
    print(f"\n{'#' * 60}")
    print(f"# ТЕСТ 3: БИФУРКАЦИОННАЯ ДИАГРАММА (РАСШИРЕННЫЙ)")
    print(f"{'#' * 60}")
    plot_bifurcation_advanced(N)

    # ТЕСТ 4: SCALING
    print(f"\n{'#' * 60}")
    print(f"# ТЕСТ 4: SCALING АНАЛИЗ")
    print(f"{'#' * 60}")
    scaling_test()


if __name__ == "__main__":
    main()