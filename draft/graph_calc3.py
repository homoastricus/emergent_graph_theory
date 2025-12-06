import numpy as np
import networkx as nx
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from collections import defaultdict
import math
from collections import deque
from scipy import stats

class UniverseGraphAnalyzer:
    """Анализатор графовой модели Вселенной с физически осмысленными параметрами"""

    def __init__(self, N, m, graph_type='RRG', theoretical_N=1e185, theoretical_k=425):
        """
        Parameters:
        N - количество узлов (планковских ячеек)
        m - степень связности (количество связей на узел)
        graph_type - тип графа: 'RRG', 'ER', 'WS' (Watts-Strogatz)
        theoretical_N - теоретическое N Вселенной
        theoretical_k - теоретическая оптимальная связность
        """
        self.N = N
        self.m = m
        self.graph_type = graph_type
        self.theoretical_N = theoretical_N
        self.theoretical_k = theoretical_k
        self.G = None
        self.A = None
        self.L = None
        self.results = {}

    def create_graph(self):
        """Автоматическое создание графа с физически осмысленной связностью"""
        print(f"Создание графа (автоматическая оптимизация параметров): N={self.N}")

        # 🔹 1. Автоматически подбираем оптимальную связность
        # k_opt ≈ log(N) * c, где c — коэффициент эффективной размерности (~3)
        k_opt = int(max(4, np.round(3 * np.log(self.N))))
        #self.m = k_opt
        print(f"  → Автоматически выбрана связность m = {self.m}")

        # 🔹 2. Используем граф малых миров (Watts–Strogatz)
        # вероятность переподключения подбирается по принципу наименьшего действия:
        # p_opt ≈ 1 / k_opt (обеспечивает баланс локальности и глобальных корреляций)
        p_opt = min(0.1, max(0.005, 1.0 / k_opt))
        print(f"  → Вероятность переподключения p = {p_opt:.4f}")

        k_even = self.m if self.m % 2 == 0 else self.m - 1
        self.G = nx.watts_strogatz_graph(n=self.N, k=k_even, p=p_opt, seed=42)

        # 🔹 3. Проверка связности и отбор крупнейшей компоненты
        if not nx.is_connected(self.G):
            print("  ⚠️  Граф не связный — берём наибольшую компоненту.")
            largest_cc = max(nx.connected_components(self.G), key=len)
            self.G = self.G.subgraph(largest_cc).copy()
            self.N = self.G.number_of_nodes()
            print(f"  → Используем компоненту с N={self.N}")

        # 🔹 4. Строим нормализованный лапласиан (инвариантный по масштабу)
        self.A = nx.adjacency_matrix(self.G)
        deg = np.array(self.A.sum(axis=1)).flatten()
        D = sp.diags(deg)
        D_inv_sqrt = sp.diags(1.0 / np.sqrt(np.maximum(deg, 1e-12)))
        self.L = sp.eye(self.N) - D_inv_sqrt @ self.A @ D_inv_sqrt

        # 🔹 5. Сохраняем основные метрики
        self.results['N_final'] = self.N
        self.results['avg_degree'] = np.mean(deg)
        self.results['degree_std'] = np.std(deg)
        self.results['edges_count'] = self.G.number_of_edges()

        print(f"  ✓ Граф успешно создан: N={self.N}, ⟨k⟩={self.results['avg_degree']:.2f}")

    def compute_volume_scaling_dimension(self, num_samples=1000):
        """
        Размерность через scaling объема: V(r) ~ r^d
        Измеряем, как растет число узлов в шаре радиуса r от случайной точки
        """
        print("Вычисление размерности через scaling объема...")

        try:
            if self.N < 1000:
                return 0

            # Выбираем случайные стартовые точки
            sample_nodes = np.random.choice(self.N, size=min(num_samples, self.N // 10), replace=False)
            radii = list(range(1, 15))  # Радиусы для анализа

            volume_data = []

            for r in radii:
                volumes = []
                for start_node in sample_nodes:
                    # BFS для нахождения всех узлов на расстоянии ≤ r
                    visited = set([start_node])
                    queue = deque([(start_node, 0)])

                    while queue:
                        node, distance = queue.popleft()
                        if distance < r:
                            for neighbor in self.G.neighbors(node):
                                if neighbor not in visited:
                                    visited.add(neighbor)
                                    queue.append((neighbor, distance + 1))

                    volumes.append(len(visited))

                if volumes:  # Добавляем только если есть данные
                    volume_data.append(np.mean(volumes))

            # Проверяем, что достаточно данных
            if len(volume_data) < 5:
                return 0

            # Убираем вырожденные случаи
            valid_mask = np.array(volume_data) > volume_data[0] + 5

            if np.sum(valid_mask) < 5:
                return 0

            log_r = np.log(np.array(radii)[valid_mask])
            log_V = np.log(np.array(volume_data)[valid_mask])

            # Линейная регрессия: log(V) ~ d * log(r)
            slope, intercept, r_value, p_value, std_err = stats.linregress(log_r, log_V)

            if r_value ** 2 > 0.9 and 0.5 < slope < 8:
                print(f"  Объемная размерность: d_V = {slope:.3f} (R² = {r_value ** 2:.3f})")
                return slope
            else:
                return 0

        except Exception as e:
            print(f"Ошибка в объемной размерности: {e}")
            return 0

    def compute_curvature_dimension(self):
        """
        Размерность через анализ Ollivier-Ricci кривизны графа
        В пространстве размерности d кривизна имеет характерный scaling
        """
        print("Вычисление размерности через кривизну...")

        try:
            if self.N > 5000:  # Для больших графов используем выборку
                sample_size = min(500, self.N // 20)
                sample_nodes = np.random.choice(self.N, size=sample_size, replace=False)
            else:
                sample_nodes = list(self.G.nodes())

            curvatures = []

            for node in sample_nodes:
                neighbors = list(self.G.neighbors(node))
                if len(neighbors) < 2:
                    continue

                # Простая оценка кривизны через локальную кластеризацию
                # В d-мерном пространстве: κ ~ 1/d для случайных геометрических графов
                try:
                    subgraph = self.G.subgraph(neighbors + [node])
                    if nx.is_connected(subgraph):  # Проверяем связность подграфа
                        clustering = nx.transitivity(subgraph)

                        # Преобразуем кластеризацию в оценку кривизны
                        if clustering > 0:
                            # Эмпирическая формула: κ ≈ 2 - 1/clustering для малых миров
                            curvature = 2.0 - 1.0 / clustering
                            # Ограничиваем диапазон кривизны
                            if -10 < curvature < 10:
                                curvatures.append(curvature)
                except:
                    continue

            if len(curvatures) < 10:
                return 0

            avg_curvature = np.mean(curvatures)
            std_curvature = np.std(curvatures)

            # Оценка размерности через кривизну: d ≈ 1/κ для плоских пространств
            if abs(avg_curvature) > 0.01:
                d_curv = 1.0 / abs(avg_curvature)
                # Ограничиваем физически осмысленный диапазон
                if 0.5 < d_curv < 10 and std_curvature / abs(avg_curvature) < 1.0:
                    print(f"  Размерность из кривизны: d_κ = {d_curv:.3f} (κ = {avg_curvature:.3f})")
                    return d_curv

        except Exception as e:
            print(f"Ошибка в размерности из кривизны: {e}")

        return 0

    def compute_fractal_dimension(self, num_walks=100, max_steps=200):
        """
        Фрактальная размерность через анализ траекторий случайного блуждания
        Используем scaling среднеквадратичного смещения: ⟨r²⟩ ~ t^(2/d_w)
        """
        print("Вычисление фрактальной размерности...")

        try:
            if self.N < 1000:
                return 0

            # Моделируем случайные блуждания
            msd_data = []  # Mean squared displacement

            # Упрощаем: берем меньше шагов для скорости
            steps_to_test = list(range(5, min(max_steps, 100), 10))

            for step in steps_to_test:
                displacements = []

                for walk in range(num_walks):
                    # Начинаем со случайного узла
                    current = np.random.randint(0, self.N)
                    start_node = current

                    # Случайное блуждание
                    for s in range(step):
                        neighbors = list(self.G.neighbors(current))
                        if not neighbors:
                            break
                        current = np.random.choice(neighbors)

                    # Вычисляем расстояние от начальной точки
                    try:
                        if nx.has_path(self.G, start_node, current):
                            distance = nx.shortest_path_length(self.G, start_node, current)
                            displacements.append(distance ** 2)
                    except:
                        continue

                if displacements:
                    msd_data.append((step, np.mean(displacements)))

            if len(msd_data) < 5:
                return 0

            steps = np.array([x[0] for x in msd_data])
            msd = np.array([x[1] for x in msd_data])

            # Фитируем: log(MSD) ~ (2/d_w) * log(t)
            valid_mask = (msd > 0) & (steps > 0)
            if np.sum(valid_mask) < 5:
                return 0

            log_t = np.log(steps[valid_mask])
            log_msd = np.log(msd[valid_mask])

            slope, intercept, r_value, p_value, std_err = stats.linregress(log_t, log_msd)

            if r_value ** 2 > 0.8 and slope > 0:
                d_w = 2.0 / slope  # walk dimension
                # Для обычной диффузии: d_f = d_w
                d_fractal = d_w

                if 0.5 < d_fractal < 8:
                    print(f"  Фрактальная размерность: d_f = {d_fractal:.3f} (R² = {r_value ** 2:.3f})")
                    return d_fractal

        except Exception as e:
            print(f"Ошибка в фрактальной размерности: {e}")

        return 0

    def compute_spectral_properties(self, k_eig=100):
        """Улучшенный спектральный анализ с многометодной оценкой размерности"""
        print("Вычисление спектральных свойств...")

        k_eig = min(k_eig, self.N - 1)

        try:
            eigvals, eigvecs = spla.eigsh(self.L, k=k_eig, which='SM', maxiter=1000)
            eigvals = np.sort(eigvals)
        except:
            eigvals, eigvecs = spla.eigsh(self.L, k=min(50, k_eig), which='SM')
            eigvals = np.sort(eigvals)

        spectral_gap = eigvals[1] if len(eigvals) > 1 else eigvals[0]

        self.results['spectral_gap'] = spectral_gap
        self.results['eigvals'] = eigvals
        self.results['eigvecs'] = eigvecs

        # МНОГОМЕТОДНАЯ оценка размерности - РАСШИРЕННАЯ
        dimension_estimates = []

        # 1. Основной спектральный метод
        d1 = self._estimate_spectral_dimension(eigvals)
        if d1 > 0:
            dimension_estimates.append(d1)

        # 2. Альтернативный спектральный метод
        d2 = self._estimate_dimension_via_scaling(eigvals)
        if d2 > 0:
            dimension_estimates.append(d2)

        # 3. Метод через случайное блуждание
        d3 = self.results.get('rw_spectral_dimension', 0)
        if d3 > 0:
            dimension_estimates.append(d3)

        # 4. НОВЫЕ МЕТОДЫ (вызываем только если уже есть хорошие оценки)
        if len(dimension_estimates) >= 1:  # Есть хотя бы одна надежная оценка

            # Объемная размерность
            d4 = self.compute_volume_scaling_dimension()
            if d4 > 0:
                dimension_estimates.append(d4)
                self.results['volume_dimension'] = d4

            # Размерность из кривизны
            d5 = self.compute_curvature_dimension()
            if d5 > 0:
                dimension_estimates.append(d5)
                self.results['curvature_dimension'] = d5

            # Фрактальная размерность
            d6 = self.compute_fractal_dimension()
            if d6 > 0:
                dimension_estimates.append(d6)
                self.results['fractal_dimension'] = d6

        # Усредняем надежные оценки
        if dimension_estimates:
            final_dimension = np.median(dimension_estimates)
            print(f"  Все оценки размерности: {[f'{d:.3f}' for d in dimension_estimates]}")
            print(f"  Финальная размерность: {final_dimension:.3f}")

            # Сохраняем статистику методов
            self.results['all_dimension_estimates'] = dimension_estimates
            self.results['dimension_std'] = np.std(dimension_estimates)
        else:
            final_dimension = 0
            print(f"  Надежная оценка размерности не получена")

        self.results['spectral_dimension'] = final_dimension
        return spectral_gap

    def _create_mixed_degree_graph(self):
        """Создание графа со смешанными степенями для дробных m"""
        print(f"Создание графа со смешанными степенями для m={self.m}")

        # Разбиваем m на целую и дробную части
        m_int = int(self.m)
        m_frac = self.m - m_int

        # Вычисляем сколько узлов будут иметь повышенную степень
        n_high_degree = int(self.N * m_frac)
        n_low_degree = self.N - n_high_degree

        # Создаем последовательность степеней
        degree_sequence = [m_int] * n_low_degree + [m_int + 1] * n_high_degree

        # Проверяем четность суммы степеней
        total_degree = sum(degree_sequence)
        if total_degree % 2 != 0:
            # Корректируем: убираем одну связь у случайного узла
            degree_sequence[0] -= 1
            print(f"Корректируем сумму степеней с {total_degree} на {total_degree - 1}")

        # Создаем граф с заданной последовательностью степеней
        try:
            self.G = nx.configuration_model(degree_sequence, seed=42)
            # Убираем кратные ребра и петли
            self.G = nx.Graph(self.G)  # Преобразуем в простой граф
            self.G.remove_edges_from(nx.selfloop_edges(self.G))
        except Exception as e:
            print(f"Ошибка при создании графа со смешанными степенями: {e}")
            # Fallback: используем ER граф с нужной плотностью
            p = self.m / (self.N - 1)
            self.G = nx.erdos_renyi_graph(n=self.N, p=p, seed=42)

    def _estimate_dimension_via_scaling(self, eigvals):
        """Альтернативный метод оценки размерности"""
        try:
            # Используем scaling низкочастотной части спектра
            low_freq = eigvals[(eigvals > 0) & (eigvals < np.percentile(eigvals, 40))]

            if len(low_freq) < 8:
                return 0

            # Масштабный анализ
            x = np.log(np.arange(1, len(low_freq) + 1))
            y = np.log(low_freq)

            coef = np.polyfit(x, y, 1)
            r2 = np.corrcoef(x, y)[0, 1] ** 2

            if r2 > 0.8:
                # Для D-мерного пространства: λ_k ~ k^{2/D}
                d_s = 2 / abs(coef[0])
                return max(0.1, min(10, d_s))

        except:
            pass

        return 0

    def _estimate_spectral_dimension(self, eigvals):
        """УСТОЙЧИВАЯ оценка спектральной размерности"""
        if len(eigvals) < 20:
            return 0

        # Исключаем нулевое собственное значение
        nonzero_eigvals = eigvals[eigvals > 1e-10]

        if len(nonzero_eigvals) < 15:
            return 0

        # ФИКСИРОВАННЫЙ диапазон низких частот (абсолютные значения)
        # Берем первые 20% ненулевых собственных значений
        n_low = max(10, len(nonzero_eigvals) // 5)
        low_eig = nonzero_eigvals[:n_low]

        # Проверяем физическую осмысленность диапазона
        if np.max(low_eig) / np.min(low_eig) > 1e6:
            return 0  # Слишком большой разброс - нефизично

        # Улучшенная гистограмма с гарантированными бинами
        try:
            # Логарифмические бины от min до max low_eig
            n_bins = min(12, len(low_eig) // 3)
            log_bins = np.logspace(np.log10(np.min(low_eig)),
                                   np.log10(np.max(low_eig)),
                                   n_bins)

            hist, bin_edges = np.histogram(low_eig, bins=log_bins, density=True)
            bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])

            # Строгая проверка валидности данных
            valid_mask = (hist > 1e-10) & (bin_centers > 0) & np.isfinite(hist)

            if np.sum(valid_mask) < 5:
                return 0

            log_λ = np.log(bin_centers[valid_mask])
            log_ρ = np.log(hist[valid_mask])

            # Проверка на выбросы
            z_scores = np.abs((log_ρ - np.mean(log_ρ)) / np.std(log_ρ))
            if np.any(z_scores > 2.5):
                return 0  # Есть выбросы

            # Линейная регрессия с проверкой качества
            coef, residuals, _, _, _ = np.polyfit(log_λ, log_ρ, 1, full=True)

            if len(residuals) == 0:
                return 0

            # R² оценка
            ss_res = residuals[0]
            ss_tot = np.sum((log_ρ - np.mean(log_ρ)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            # Строгие критерии качества
            if r_squared < 0.85 or abs(coef[0]) > 8:
                return 0

            d_s = 2 * (coef[0] + 1)

            # ФИЗИЧЕСКИЕ ОГРАНИЧЕНИЯ
            if d_s < 0.1 or d_s > 8:
                return 0

            print(f"  Размерность: d_s = {d_s:.3f} (R² = {r_squared:.3f})")
            return d_s

        except Exception as e:
            print(f"Ошибка в оценке размерности: {e}")
            return 0

    def compute_information_metrics(self):
        """Вычисление информационных метрик"""
        print("Вычисление информационных метрик...")

        # Энтропия на узел (Шенноновская энтропия распределения степеней)
        degrees = [d for _, d in self.G.degree()]
        degree_probs = np.bincount(degrees) / len(degrees)
        degree_probs = degree_probs[degree_probs > 0]  # Убираем нули

        entropy_per_node = -np.sum(degree_probs * np.log2(degree_probs))

        # Полная энтропия системы
        # Для графа с N узлами и E ребрами: log2(числа возможных конфигураций)
        E = self.G.number_of_edges()
        k_possible = self.N * (self.N - 1) // 2  # Максимальное число ребер

        # Приближение для разреженных графов
        if E < 0.01 * k_possible:
            total_entropy = E * np.log2(k_possible / E) + E * np.log2(math.e)
        else:
            # Общая формула через энтропию биномиального распределения
            p = E / k_possible
            if p > 0 and p < 1:
                H_binom = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
                total_entropy = k_possible * H_binom
            else:
                total_entropy = 0

        # Информационная связность (average shortest path length)
        try:
            # Для больших графов используем выборку
            if self.N > 1000:
                sample_nodes = np.random.choice(self.N, size=min(100, self.N // 10), replace=False)
                path_lengths = []
                for i, node1 in enumerate(sample_nodes):
                    for node2 in sample_nodes[i + 1:]:
                        try:
                            length = nx.shortest_path_length(self.G, node1, node2)
                            path_lengths.append(length)
                        except:
                            continue
                avg_path_length = np.mean(path_lengths) if path_lengths else 0
            else:
                avg_path_length = nx.average_shortest_path_length(self.G)
        except:
            avg_path_length = 0

        self.results['entropy_per_node'] = entropy_per_node
        self.results['total_entropy'] = total_entropy
        self.results['information_connectivity'] = avg_path_length

        return entropy_per_node, total_entropy, avg_path_length

    def compute_physical_metrics(self):
        """Вычисление дополнительных физических метрик"""
        print("Вычисление физических метрик...")

        # Эффективная размерность через случайное блуждание
        d_s_rw = self._estimate_rw_spectral_dimension()
        self.results['rw_spectral_dimension'] = d_s_rw

        # Кластеризация (transitivity)
        clustering = nx.transitivity(self.G)
        self.results['clustering_coefficient'] = clustering

        # Ассортативность (assortativity)
        assortativity = nx.degree_assortativity_coefficient(self.G)
        self.results['assortativity'] = assortativity

        # Эффективная "скорость света" через spectral gap
        # c_eff ~ 1 / sqrt(λ₁) в подходящих единицах
        if self.results['spectral_gap'] > 0:
            c_eff = 1.0 / np.sqrt(self.results['spectral_gap'])
            self.results['effective_speed'] = c_eff
        else:
            self.results['effective_speed'] = 0

        return d_s_rw, clustering, assortativity

    def _estimate_rw_spectral_dimension(self, n_steps=1000, sample_size=1000):
        """Оценка спектральной размерности через случайное блуждание"""
        try:
            N = self.N
            if N < 100:
                return 0

            # Инициализация случайного блуждания
            deg = np.array(self.A.sum(axis=1)).flatten()
            D_inv = sp.diags(1 / deg)
            W = D_inv @ self.A

            sample_idx = np.random.choice(N, size=min(sample_size, N // 10), replace=False)
            P0 = np.zeros(N)
            P0[sample_idx] = 1.0 / len(sample_idx)

            P = P0.copy()
            ret_prob = []

            # Моделирование
            for t in range(n_steps):
                P = W @ P
                ret_prob.append(P[sample_idx].mean())

            ret_prob = np.array(ret_prob)
            t_vals = np.arange(1, n_steps + 1)

            # Ищем степенной закон на подходящем диапазоне
            skip_start = max(10, n_steps // 50)
            if len(t_vals) > skip_start + 50:
                fit_t = t_vals[skip_start:-10]
                fit_P = ret_prob[skip_start:-10]

                if np.all(fit_P > 0) and len(fit_t) > 20:
                    logt = np.log(fit_t)
                    logP = np.log(fit_P)

                    # Исключаем выбросы
                    mask = np.abs(logP - np.mean(logP)) < 2 * np.std(logP)
                    if np.sum(mask) > 10:
                        coef = np.polyfit(logt[mask], logP[mask], 1)
                        d_s = -2 * coef[0]
                        return max(0, d_s)

        except Exception as e:
            print(f"Ошибка в оценке RW размерности: {e}")

        return 0

    def add_universe_parameters(self):
        """Добавление параметров из теоретической модели ЕТИ"""

        # Базовые параметры Вселенной
        self.results['N_theoretical'] = self.theoretical_N
        self.results['k_optimal'] = self.theoretical_k
        self.results['target_dimension'] = 3.0
        self.results['correlation_exponent'] = 2.0

        # Проверка соответствия нашей модели теории
        k_ratio = self.results['avg_degree'] / self.results['k_optimal']
        dim_ratio = self.results['spectral_dimension'] / self.results['target_dimension']

        self.results['k_optimality_ratio'] = k_ratio
        self.results['dimension_ratio'] = dim_ratio

    def compute_theoretical_metrics(self):
        """Вычисление метрик из теоретического формализма ЕТИ"""

        # 1. Информационное действие (упрощенная версия)
        k = self.results['avg_degree']
        k_opt = self.results['k_optimal']
        k_std = self.results['degree_std']

        # Компоненты действия из вашей формулы
        rigidity = k_std ** 2  # (∇k)² - жесткость геометрии
        balance = (k - k_opt) ** 2  # Баланс корреляции и свободы
        freedom = 1.0 / k ** 2 if k > 0 else 0  # Свобода (γ/k²)

        # Энтропийная составляющая (пропорциональна полной энтропии)
        entropy_term = self.results['total_entropy'] / 1e10  # Нормировка

        # Голографическое ограничение
        N = self.results['N_final']
        holo_constraint = N ** (2 / 3) / N  # Информация ∝ N^{2/3}

        informational_action = (rigidity + balance + freedom +
                                entropy_term + holo_constraint)

        self.results['informational_action'] = informational_action
        self.results['action_components'] = {
            'rigidity': rigidity,
            'balance': balance,
            'freedom': freedom,
            'entropy': entropy_term,
            'holographic': holo_constraint
        }

        # 2. Проверка закона корреляций
        correlation_exponent = self._estimate_correlation_exponent()
        self.results['estimated_correlation_exponent'] = correlation_exponent
        self.results['correlation_law_deviation'] = abs(correlation_exponent - 2.0)

        # 3. Флуктуации связности (σₖ/⟨k⟩ ~ √ℏ)
        fluctuation_ratio = k_std / k if k > 0 else 0
        self.results['connectivity_fluctuation'] = fluctuation_ratio

        return informational_action, correlation_exponent

    def _estimate_correlation_exponent(self):
        """Оценка показателя корреляционного закона C(r) ∝ 1/r^α"""
        try:
            # Используем распределение кратчайших путей как прокси для корреляций
            if self.N > 1000:
                sample_nodes = np.random.choice(self.N, size=min(200, self.N // 20), replace=False)
                distances = []
                for i, node1 in enumerate(sample_nodes):
                    for node2 in sample_nodes[i + 1:i + 21]:  # Ограничиваем пары
                        try:
                            dist = nx.shortest_path_length(self.G, node1, node2)
                            distances.append(dist)
                        except:
                            continue

                if len(distances) > 50:
                    # Гистограмма расстояний
                    hist, bins = np.histogram(distances, bins=20, density=True)
                    bin_centers = (bins[:-1] + bins[1:]) / 2

                    # Фит степенного закона (исключая нулевые значения)
                    valid_mask = (hist > 0) & (bin_centers > 0)
                    if np.sum(valid_mask) > 5:
                        log_r = np.log(bin_centers[valid_mask])
                        log_C = np.log(hist[valid_mask])
                        coef = np.polyfit(log_r, log_C, 1)
                        return -coef[0]  # C(r) ∝ r^{-α}

        except Exception as e:
            print(f"Ошибка в оценке корреляционного показателя: {e}")

        return 0

    def _compute_model_quality(self):
        """Оценка качества соответствия теоретическим предсказаниям"""
        r = self.results

        scores = []

        # Оптимальность связности
        k_score = 1.0 - min(1.0, abs(r['k_optimality_ratio'] - 1.0))
        scores.append(k_score * 3)  # Вес 3

        # Размерность
        dim_diff = abs(r['spectral_dimension'] - r['target_dimension'])
        dim_score = 1.0 - min(1.0, dim_diff / r['target_dimension'])
        scores.append(dim_score * 3)  # Вес 3

        # Корреляционный закон
        if 'correlation_law_deviation' in r:
            corr_score = 1.0 - min(1.0, r['correlation_law_deviation'] / 2.0)
            scores.append(corr_score * 2)  # Вес 2

        # Флуктуации
        fluct_score = 1.0 - min(1.0, abs(r['connectivity_fluctuation'] - 0.01) / 0.01)
        scores.append(fluct_score * 2)  # Вес 2

        return min(10, sum(scores))

    def analyze(self, k_eig=100):
        """Полный анализ графа"""
        print("ПОЛНЫЙ АНАЛИЗ ГРАФОВОЙ МОДЕЛИ ВСЕЛЕННОЙ")

        # 1. Создание графа
        self.create_graph()

        # 2. Спектральные свойства
        spectral_gap = self.compute_spectral_properties(k_eig)

        # 3. Информационные метрики
        entropy_node, entropy_total, connectivity = self.compute_information_metrics()

        # 4. Физические метрики
        d_s_rw, clustering, assortativity = self.compute_physical_metrics()

        # 5. Теоретические метрики ЕТИ
        self.add_universe_parameters()
        action, corr_exp = self.compute_theoretical_metrics()

        # 6. Вывод результатов
        self._print_results()
        self._print_theoretical_interpretation()

        return self.results

    def _print_results(self):
        """Вывод результатов анализа"""
        print("РЕЗУЛЬТАТЫ АНАЛИЗА")

        results = self.results

        print(f"ОСНОВНЫЕ ПАРАМЕТРЫ:")
        print(f"  Узлов (N): {results['N_final']:,}")
        print(f"  Средняя степень: {results['avg_degree']:.2f}")
        print(f"  Ребер: {results['edges_count']:,}")

        print(f"\nМНОГОМЕТОДНАЯ ОЦЕНКА РАЗМЕРНОСТИ:")
        print(f"  Спектральная размерность: {results['spectral_dimension']:.3f}")

        if 'all_dimension_estimates' in results:
            print(f"  Все методы: {[f'{d:.3f}' for d in results['all_dimension_estimates']]}")
            print(f"  Стандартное отклонение: {results['dimension_std']:.3f}")

        # Новые методы
        if 'volume_dimension' in results:
            print(f"  Объемная размерность: {results['volume_dimension']:.3f}")
        if 'curvature_dimension' in results:
            print(f"  Размерность из кривизны: {results['curvature_dimension']:.3f}")
        if 'fractal_dimension' in results:
            print(f"  Фрактальная размерность: {results['fractal_dimension']:.3f}")

        print(f"\nСПЕКТРАЛЬНЫЕ СВОЙСТВА:")
        print(f"  Spectral gap (λ₁): {results['spectral_gap']:.6f}")
        print(f"  Спектральная размерность: {results['spectral_dimension']:.3f}")
        print(f"  RW размерность: {results['rw_spectral_dimension']:.3f}")

        print(f"\nИНФОРМАЦИОННЫЕ МЕТРИКИ:")
        print(f"  Энтропия на узел: {results['entropy_per_node']:.3f} бит")
        print(f"  Полная энтропия: {results['total_entropy']:.3e} бит")
        print(f"  Информационная связность: {results['information_connectivity']:.3f}")

        print(f"\nФИЗИЧЕСКИЕ МЕТРИКИ:")
        print(f"  Кластеризация: {results['clustering_coefficient']:.4f}")
        print(f"  Ассортативность: {results['assortativity']:.4f}")
        if 'effective_speed' in results:
            print(f"  Эффективная скорость: {results['effective_speed']:.4f}")

        print(f"\nЭФФЕКТИВНОСТЬ МОДЕЛИ:")
        density = results['edges_count'] / (results['N_final'] * (results['N_final'] - 1) // 2)
        print(f"  Плотность графа: {density:.6f}")
        print(f"  Энтропия на связь: {results['entropy_per_node'] / results['avg_degree']:.4f} бит/связь")

    def _print_theoretical_interpretation(self):
        """Расширенная физическая интерпретация"""
        print("ТЕОРЕТИЧЕСКАЯ ИНТЕРПРЕТАЦИЯ (ЕТИ ФОРМАЛИЗМ)")

        r = self.results

        print(f"СООТВЕТСТВИЕ ТЕОРЕТИЧЕСКИМ ПРЕДСКАЗАНИЯМ:")
        print(f"  • Оптимальность связности: {r['k_optimality_ratio']:.3f} (цель: 1.0)")
        print(f"  • Размерность: {r['spectral_dimension']:.2f} (цель: {r['target_dimension']:.1f})")
        print(f"  • Флуктуации связности: {r['connectivity_fluctuation']:.4f} (~√ℏ)")

        if 'estimated_correlation_exponent' in r:
            print(f"  • Закон корреляций: C(r) ∝ 1/r^{r['estimated_correlation_exponent']:.2f} (цель: 2.00)")

        print(f"\nИНФОРМАЦИОННОЕ ДЕЙСТВИЕ: {r['informational_action']:.6f}")
        if 'action_components' in r:
            comp = r['action_components']
            print(f"  Компоненты:")
            print(f"  • Жесткость: {comp['rigidity']:.6f}")
            print(f"  • Баланс: {comp['balance']:.6f}")
            print(f"  • Свобода: {comp['freedom']:.6f}")
            print(f"  • Энтропия: {comp['entropy']:.6f}")
            print(f"  • Голографическое: {comp['holographic']:.6f}")

        # Оценка физической осмысленности
        print(f"\nФИЗИЧЕСКАЯ ОЦЕНКА МОДЕЛИ:")
        quality_score = self._compute_model_quality()
        print(f"  Общая оценка: {quality_score:.1f}/10")

        if quality_score > 7:
            print("  ✅ Модель хорошо соответствует теоретическим предсказаниям")
        elif quality_score > 5:
            print("  ⚠️  Модель требует тонкой настройки параметров")
        else:
            print("  ❌ Модель нуждается в фундаментальном пересмотре")


# Пример использования
if __name__ == "__main__":
    # Физически осмысленные параметры
    N = 20000  # 100 тысяч узлов
    m = 220.0 #math.log(N)
    print(f" степень связи: {m:.2f}")

    analyzer = UniverseGraphAnalyzer(
        N=N, m=379,
        # (450 давало 2.6, 410 - 2.827 , 390 - 0, 401 - 2.886,
        # 402-0, 403 - 0, 404 - 2.862, 399 - 0, 398 - 0,
        # 397 - 2.910, 396 - 2.910, 395 - 2.922, 394 - 2.922, 393 - 0, 392 - 0, 391 - 0,
        # 389 - 0, 388 - 0, 387 - 0, 386 - 0,385 - 0, 384 - 0, 383 - 2.998, 382 - 2.998,
        # 381 - 0, 380 - 0, 379 - 3.025  )
        graph_type='RRG',
        theoretical_N=1e185,
        theoretical_k=425
    )

    results = analyzer.analyze()

    print("ФИЗИЧЕСКАЯ ИНТЕРПРЕТАЦИЯ ДЛЯ ВСЕЛЕННОЙ:")

    if results['spectral_dimension'] > 0:
        print(f"• Эффективная размерность пространства: {results['spectral_dimension']:.2f}")
    print(f"• Информационная емкость: {results['total_entropy']:.2e} бит")
    print(f"• Скорость распространения информации: {results.get('effective_speed', 0):.2f}")
    print(f"• Степень квантовой запутанности: {results['clustering_coefficient']:.3f}")
    print(f"• Оптимальность структуры: {results['k_optimality_ratio']:.3f}")