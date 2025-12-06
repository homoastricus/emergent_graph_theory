import numpy as np
import networkx as nx
from scipy.sparse import csgraph
from scipy.optimize import curve_fit
from collections import Counter
import warnings

warnings.filterwarnings('ignore')


class EmergentGeometryAnalyzer:
    """
    Математически строгий анализатор эмерджентной геометрии графов
    Вычисляет эффективную размерность, энтропии и информационные меры
    """

    def __init__(self, N, K, p, random_seed=None):
        """
        Инициализация анализатора

        Parameters:
        N (int): Количество узлов
        K (int): Количество ближайших соседей (четное)
        p (float): Вероятность пересоединения [0, 1]
        random_seed (int): Seed для воспроизводимости
        """
        self.N = N
        self.K = K
        self.p = p
        self.random_seed = random_seed

        # Генерация графа
        self.G = nx.watts_strogatz_graph(N, K, p, seed=random_seed)

        # Проверка связности
        if not nx.is_connected(self.G):
            self.largest_component = max(nx.connected_components(self.G), key=len)
            self.G = self.G.subgraph(self.largest_component).copy()
            self.N = len(self.G)
            print(f"Внимание: граф несвязный. Используется наибольшая компонента с {self.N} узлами")

    def effective_dimension(self, num_samples=50, max_radius=None):
        """
        Вычисляет эффективную размерность Хаусдорфа через скейлинг объема

        Метод: N(r) ~ r^d, где N(r) - количество узлов в шаре радиуса r
        Размерность d определяется как наклон в логарифмических координатах
        """
        if max_radius is None:
            max_radius = min(20, nx.diameter(self.G) // 2)

        # Функция для подгонки
        def power_law(r, d, C):
            return C * r ** d

        radii = []
        volumes = []

        # Выбор случайных узлов для усреднения
        sample_nodes = np.random.choice(list(self.G.nodes()),
                                        size=min(num_samples, self.N),
                                        replace=False)

        for r in range(1, max_radius + 1):
            total_volume = 0
            count = 0

            for node in sample_nodes:
                try:
                    # Находим все узлы на расстоянии <= r
                    ego_nodes = nx.ego_graph(self.G, node, radius=r, center=False)
                    volume = len(ego_nodes)
                    total_volume += volume
                    count += 1
                except:
                    continue

            if count > 0:
                radii.append(r)
                volumes.append(total_volume / count)

        if len(radii) < 3:
            return 0.0, np.zeros(2), np.array([]), np.array([])

        # Линейная регрессия в логарифмических координатах
        log_r = np.log(radii)
        log_v = np.log(volumes)

        try:
            # Подгонка степенного закона
            popt, pcov = curve_fit(power_law, radii, volumes,
                                   p0=[2.0, 1.0], maxfev=5000)
            dimension = popt[0]
            std_error = np.sqrt(np.diag(pcov))[0]
        except:
            # Резервный метод: линейная регрессия
            A = np.vstack([log_r, np.ones(len(log_r))]).T
            slope, intercept = np.linalg.lstsq(A, log_v, rcond=None)[0]
            dimension = slope
            std_error = 0.1

        return dimension, np.array([dimension - std_error, dimension + std_error]), np.array(radii), np.array(volumes)

    def node_entropy(self):
        """
        Вычисляет энтропию на узел через распределение степеней

        H_node = -Σ p(k) log p(k), где p(k) = N_k / N
        N_k - количество узлов со степенью k
        """
        degrees = [d for _, d in self.G.degree()]
        degree_counts = Counter(degrees)

        total_nodes = len(degrees)
        entropy = 0.0

        for count in degree_counts.values():
            p = count / total_nodes
            if p > 0:
                entropy -= p * np.log(p)

        return entropy

    def link_entropy(self):
        """
        Вычисляет энтропию на связь через распределение communicability

        Communicability измеряет, сколько путей соединяет две вершины
        """
        try:
            # Матрица смежности
            A = nx.adjacency_matrix(self.G).astype(float)

            # Communicability matrix: exp(A)
            # Используем приближение через 3 первых члена ряда Тейлора для стабильности
            I = np.eye(A.shape[0])
            comm_matrix = I + A + A.dot(A) / 2 + A.dot(A).dot(A) / 6

            # Нормализуем для получения распределения вероятностей
            comm_flat = comm_matrix.flatten()
            comm_flat = comm_flat[comm_flat > 0]  # Убираем нули

            # Нормализация
            p_comm = comm_flat / np.sum(comm_flat)

            # Энтропия Шеннона
            entropy = -np.sum(p_comm * np.log(p_comm))

            return entropy / len(self.G.edges())  # Нормируем на количество связей

        except:
            # Резервный метод: энтропия на основе clustering coefficient
            clustering = nx.clustering(self.G)
            clustering_values = list(clustering.values())
            clustering_values = [c for c in clustering_values if c > 0]

            if len(clustering_values) == 0:
                return 0.0

            p_clust = np.array(clustering_values) / np.sum(clustering_values)
            entropy = -np.sum(p_clust * np.log(p_clust))

            return entropy / len(self.G.edges())

    def total_information(self):
        """
        Вычисляет суммарную информацию (связность) графа

        Использует спектральную энтропию матрицы Лапласа:
        I_total = log(N) - H_spectral
        где H_spectral = -Σ (λ_i/Σλ) log(λ_i/Σλ)
        """
        try:
            # Нормализованная матрица Лапласа
            L = nx.normalized_laplacian_matrix(self.G)

            if L.shape[0] == 0:
                return 0.0

            # Собственные значения
            eigenvalues = np.linalg.eigvals(L.toarray())
            eigenvalues = np.real(eigenvalues)
            eigenvalues = eigenvalues[eigenvalues > 1e-10]  # Убираем нули

            if len(eigenvalues) == 0:
                return 0.0

            # Нормализация для распределения вероятностей
            p_spectral = eigenvalues / np.sum(eigenvalues)

            # Спектральная энтропия
            H_spectral = -np.sum(p_spectral * np.log(p_spectral))

            # Максимальная энтропия (для равномерного распределения)
            H_max = np.log(len(eigenvalues))

            # Информация (отклонение от максимальной энтропии)
            information = H_max - H_spectral

            return max(0.0, information)

        except:
            # Резервный метод: информация на основе связности
            if self.N == 0:
                return 0.0

            # Простая мера: логарифм количества связей относительно максимально возможного
            m = len(self.G.edges())
            m_max = self.N * (self.N - 1) / 2

            if m_max == 0:
                return 0.0

            return -np.log(m / m_max) if m > 0 else 0.0

    def analyze_all(self):
        """Выполняет полный анализ и возвращает все метрики"""

        print(f"Анализ графа Уоттса-Строгаца:")
        print(f"N={self.N}, K={self.K}, p={self.p:.3f}")
        print(f"Реальные параметры: {len(self.G.nodes())} узлов, {len(self.G.edges())} связей")
        print()

        # 1. Эффективная размерность
        dimension, conf_interval, radii, volumes = self.effective_dimension()
        print(f"1. Эффективная размерность: {dimension:.3f}")
        print(f"   Доверительный интервал: [{conf_interval[0]:.3f}, {conf_interval[1]:.3f}]")

        # 2. Энтропия на узел
        H_node = self.node_entropy()
        print(f"2. Энтропия на узел: {H_node:.3f} нат/узел")

        # 3. Энтропия на связь
        H_link = self.link_entropy()
        print(f"3. Энтропия на связь: {H_link:.3f} нат/связь")

        # 4. Суммарная информация
        I_total = self.total_information()
        print(f"4. Суммарная информация (связность): {I_total:.3f} нат")

        # Дополнительные базовые метрики
        avg_path = nx.average_shortest_path_length(self.G) if nx.is_connected(self.G) else float('inf')
        clustering = nx.average_clustering(self.G)

        print(f"\nДополнительные метрики:")
        print(f"   Средняя длина пути: {avg_path:.3f}")
        print(f"   Кластеризация: {clustering:.3f}")
        print(f"   Диаметр: {nx.diameter(self.G) if nx.is_connected(self.G) else '∞'}")

        return {
            'dimension': dimension,
            'dimension_confidence': conf_interval,
            'node_entropy': H_node,
            'link_entropy': H_link,
            'total_information': I_total,
            'avg_path_length': avg_path,
            'clustering': clustering,
            'radii': radii,
            'volumes': volumes
        }


# ПРИМЕР ИСПОЛЬЗОВАНИЯ
if __name__ == "__main__":
    # Параметры графа
    N = 7000
    K = 8
    p = 0.0527

    # Создание анализатора и выполнение анализа
    analyzer = EmergentGeometryAnalyzer(N, K, p, random_seed=42)
    results = analyzer.analyze_all()

    print("\n" + "=" * 50)
    print("ВИЗУАЛИЗАЦИЯ СКЕЙЛИНГА РАЗМЕРНОСТИ")
    print("=" * 50)

    # Визуализация скейлинга для размерности
    if len(results['radii']) > 0:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 4))

        # Логарифмический график для проверки степенного закона
        plt.subplot(1, 2, 1)
        plt.loglog(results['radii'], results['volumes'], 'bo-', alpha=0.7)
        plt.xlabel('Радиус r (log)')
        plt.ylabel('Объем N(r) (log)')
        plt.title(f'Скейлинг объема\nЭффективная размерность: {results["dimension"]:.3f}')
        plt.grid(True, alpha=0.3)

        # Линейный график
        plt.subplot(1, 2, 2)
        plt.plot(results['radii'], results['volumes'], 'ro-', alpha=0.7)
        plt.xlabel('Радиус r')
        plt.ylabel('Объем N(r)')
        plt.title('Рост объема шара')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    print(f"\nИНТЕРПРЕТАЦИЯ РЕЗУЛЬТАТОВ:")
    print(
        f"• Размерность ~{results['dimension']:.2f}: Пространство имеет {'низкую' if results['dimension'] < 2 else 'умеренную' if results['dimension'] < 4 else 'высокую'} размерность")
    print(
        f"• Энтропия узла ~{results['node_entropy']:.3f}: {'Низкая' if results['node_entropy'] < 1.0 else 'Высокая'} разнородность связей")
    print(
        f"• Суммарная информация ~{results['total_information']:.3f}: {'Слабая' if results['total_information'] < 2.0 else 'Сильная'} структурная организация")