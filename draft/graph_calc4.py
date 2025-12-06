import numpy as np
import networkx as nx
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from collections import defaultdict
import math


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

    def compute_spectral_properties(self, k_eig=100):
        """Улучшенный спектральный анализ"""
        print("Вычисление спектральных свойств...")

        k_eig = min(k_eig, self.N - 1)

        # Используем более стабильный алгоритм
        try:
            eigvals, eigvecs = spla.eigsh(self.L, k=k_eig, which='SM', maxiter=1000)
            eigvals = np.sort(eigvals)
        except:
            # Fallback для проблемных случаев
            eigvals, eigvecs = spla.eigsh(self.L, k=min(50, k_eig), which='SM')
            eigvals = np.sort(eigvals)

        spectral_gap = eigvals[1] if len(eigvals) > 1 else eigvals[0]

        self.results['spectral_gap'] = spectral_gap
        self.results['eigvals'] = eigvals
        self.results['eigvecs'] = eigvecs

        # МНОГОМЕТОДНАЯ оценка размерности
        dimension_estimates = []

        # 1. Основной метод
        d1 = self._estimate_spectral_dimension(eigvals)
        if d1 > 0:
            dimension_estimates.append(d1)

        # 2. Альтернативный метод через scaling
        d2 = self._estimate_dimension_via_scaling(eigvals)
        if d2 > 0:
            dimension_estimates.append(d2)

        # 3. Метод через случайное блуждание
        d3 = self.results.get('rw_spectral_dimension', 0)
        if d3 > 0:
            dimension_estimates.append(d3)

        # Усредняем надежные оценки
        if dimension_estimates:
            final_dimension = np.median(dimension_estimates)
            print(f"  Оценки размерности: {dimension_estimates}")
            print(f"  Финальная размерность: {final_dimension:.3f}")
        else:
            final_dimension = 0
            print(f"  Надежная оценка размерности не получена")

        self.results['spectral_dimension'] = final_dimension
        return spectral_gap

    def _estimate_spectral_dimension(self, eigvals):
        """
        Строгая оценка спектральной размерности d_s через scaling низких собственных значений.
        Используется теоретическое соотношение λ_k ~ k^(2/d_s).
        """
        # Удаляем нули
        nonzero = eigvals[eigvals > 1e-12]
        if len(nonzero) < 10:
            return 0

        # Берем только низкочастотную часть спектра
        M = min(100, len(nonzero))
        low = np.sort(nonzero[:M])
        k = np.arange(1, len(low) + 1)

        log_k = np.log(k)
        log_lambda = np.log(low)

        # Линейная аппроксимация log(λ_k) ~ (2/d_s)*log(k)
        slope, intercept = np.polyfit(log_k, log_lambda, 1)
        d_s = 2.0 / slope if slope != 0 else 0

        # Проверка качества аппроксимации
        residuals = log_lambda - (slope * log_k + intercept)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((log_lambda - np.mean(log_lambda))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        # Строгие физические ограничения
        if r2 < 0.85 or d_s < 0.1 or d_s > 10:
            return 0

        print(f"  Строгая оценка размерности: d_s = {d_s:.3f} (R² = {r2:.3f})")
        return d_s


    def _estimate_dimension_via_scaling(self, L=None, M=100):
        """
        Альтернативная строгая оценка размерности d_s напрямую из лапласиана L (если передан),
        без использования гистограмм. Использует те же принципы scaling низких собственных значений.
        """
        try:
            if L is None:
                L = self.L

            N = L.shape[0]
            M = min(M, N - 1)
            eigvals, _ = spla.eigsh(L, k=M, which='SM', maxiter=2000)
            eigvals = np.sort(eigvals)
            nonzero = eigvals[eigvals > 1e-12]
            if len(nonzero) < 5:
                return 0

            k = np.arange(1, len(nonzero) + 1)
            log_k = np.log(k)
            log_lambda = np.log(nonzero)

            slope, intercept = np.polyfit(log_k, log_lambda, 1)
            d_s = 2.0 / slope if slope != 0 else 0

            residuals = log_lambda - (slope * log_k + intercept)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((log_lambda - np.mean(log_lambda))**2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            if r2 < 0.85 or d_s < 0.1 or d_s > 10:
                return 0

            print(f"  Альтернативная строгая оценка: d_s = {d_s:.3f} (R² = {r2:.3f})")
            return d_s

        except Exception as e:
            print(f"Ошибка при строгой оценке размерности: {e}")
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
    N = 3000

    analyzer = UniverseGraphAnalyzer(
        N=N, m=2,
        # 46  - 2.150, 56 - 2.0, 38 - 2.349, 37 - 0, 68 - 1.966, 78 - 1.99, 90 - 2.044, 120 - 2.238
        # 150 - 2.46 170 - 2.628, 195 - 2.827, все что выше дает 0. # 190 - 2.795, 210 - 0, 207 - 0


        # 30000 - 200 - 0. 230 - 0 ,  270. 500 - 1.974, 700 - 1.866, 400 - 2.180, 300 - не работает, 324 -0, 364- 2.294
        graph_type='WS',
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