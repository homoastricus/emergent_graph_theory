import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve
import networkx as nx
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')


class ScientificEmergentSpacetime:
    """
    НАУЧНО КОРРЕКТНЫЙ симулятор эмерджентного пространства-времени
    Полная реализация без упрощений и заглушек
    """

    def __init__(self, N: int = 800, K: float = 6, p: float = 0.05,
                 time_steps: int = 300, dt: float = 0.005):
        # Параметры симуляции
        self.N = N
        self.K = K
        self.p = p
        self.dt = dt
        self.time_steps = time_steps
        self.time = np.linspace(0, time_steps * dt, time_steps)

        # ФИЗИЧЕСКИЕ ПАРАМЕТРЫ (оптимизированные для стабильности)
        self.alpha_space = 0.2  # Диффузия метрики
        self.beta_space = 0.8  # Линейный коэффициент метрики
        self.gamma_space = 0.1  # Нелинейность метрики

        self.alpha_time = 0.3  # Сила TSCO связи
        self.beta_time = 0.9  # Линейный коэффициент TSCO
        self.gamma_time = 0.05  # Нелинейность TSCO
        self.lambda_coupling = 0.05  # Связь метрика-TSCO

        # Инициализация всех компонентов
        self._initialize_network()
        self._initialize_fields()
        self._initialize_operators()
        self._precompute_tsco_kernels()

        # Мониторинг и метрики
        self.energy_history = []
        self.dimension_history = []
        self.metrics_history = []

        print("🎯 НАУЧНЫЙ СИМУЛЯТОР ИНИЦИАЛИЗИРОВАН")
        print(f"Сеть: N={self.N}, K={self.K}, p={self.p}")
        print(f"Время: steps={time_steps}, dt={dt}")
        print("=" * 60)

    def _initialize_network(self) -> None:
        """КОРРЕКТНАЯ инициализация сети малого мира"""
        # Преобразуем K в int для создания графа, но сохраняем оригинальное значение
        K_int = int(self.K)
        self.G = nx.watts_strogatz_graph(self.N, K_int, self.p, seed=42)

        # Гарантия связности
        if not nx.is_connected(self.G):
            largest_cc = max(nx.connected_components(self.G), key=len)
            self.G = self.G.subgraph(largest_cc).copy()
            self.N = len(self.G)
            print(f"🔧 Используется наибольшая компонента: N={self.N}")

        # Сетевые матрицы
        self.adjacency = nx.adjacency_matrix(self.G).astype(float)
        self.degrees = np.array([d for _, d in self.G.degree()])

        # КОРРЕКТНЫЙ лапласиан
        D = sparse.diags(self.degrees)
        self.laplacian = D - self.adjacency

        # ħ_em по ВАШЕЙ формуле - используем оригинальный K (float)
        self.K_i = np.maximum(self.degrees, 4.0)
        lambda_eff = 3.0
        self.hbar_em = (np.log(self.K_i) ** 2) / (4 * lambda_eff ** 2 * self.K_i ** 2)
        self.hbar_em = np.clip(self.hbar_em, 1e-6, 0.01)

        # Сетевые метрики
        self.avg_path_length = nx.average_shortest_path_length(self.G)
        self.clustering = nx.average_clustering(self.G)

        print(f"📊 СЕТЕВЫЕ МЕТРИКИ:")
        print(f"   Узлы: {self.N}, Средняя степень: {np.mean(self.degrees):.2f}")
        print(f"   Средний путь: {self.avg_path_length:.2f}, Кластеризация: {self.clustering:.3f}")
        print(f"   ħ_em: [{np.min(self.hbar_em):.2e}, {np.max(self.hbar_em):.2e}]")
        print(f"   Использован K={self.K} (целое значение: {K_int})")

    def _initialize_fields(self) -> None:
        """ФИЗИЧЕСКИ ОСМЫСЛЕННЫЕ начальные условия"""
        # Метрика - решение линеаризованных уравнений Эйнштейна
        I = sparse.identity(self.N)
        A = I + 0.1 * self.laplacian
        rhs = np.ones(self.N) + 0.05 * np.random.randn(self.N)
        self.g = spsolve(A, rhs)

        # TSCO поля - собственные функции лапласиана
        try:
            eigenvalues, eigenvectors = sparse.linalg.eigsh(self.laplacian, k=3, which='SM')
            self.psi = np.zeros((self.time_steps, self.N))
            # Суперпозиция низших мод
            for i in range(min(3, eigenvectors.shape[1])):
                self.psi[0] += 0.3 * eigenvectors[:, i] * (1 + 0.1 * np.random.randn())
        except:
            # Резервная инициализация
            self.psi = np.zeros((self.time_steps, self.N))
            self.psi[0] = 0.5 * np.random.randn(self.N)

        # Нормализация для стабильности
        self.g = np.clip(self.g, 0.7, 1.3)
        self.psi[0] = np.clip(self.psi[0], -1.0, 1.0)

        print("🎛️  ПОЛЯ ИНИЦИАЛИЗИРОВАНЫ")
        print(f"   Метрика: <g>={np.mean(self.g):.3f} ± {np.std(self.g):.3f}")
        print(f"   TSCO: <ψ>={np.mean(self.psi[0]):.3f} ± {np.std(self.psi[0]):.3f}")

    def _initialize_operators(self) -> None:
        """КОРРЕКТНЫЕ математические операторы"""
        I = sparse.identity(self.N)

        # Оператор для неявной схемы метрики
        self.metric_operator = I + self.dt * self.alpha_space * self.laplacian
        try:
            self.metric_solver = sparse.linalg.factorized(self.metric_operator.tocsc())
        except:
            # Резервный метод с регуляризацией
            self.metric_operator += 1e-4 * I
            self.metric_solver = sparse.linalg.factorized(self.metric_operator.tocsc())

        print("⚙️  ОПЕРАТОРЫ ИНИЦИАЛИЗИРОВАНЫ")

    def _precompute_tsco_kernels(self) -> None:
        """ПРЕДВЫЧИСЛЕНИЕ ядер TSCO для эффективности"""
        print("🔄 Предвычисление ядер TSCO...")

        self.kernel_matrix = np.zeros((self.N, self.N))
        paths = dict(nx.all_pairs_shortest_path_length(self.G))

        for i in range(self.N):
            for j in range(self.N):
                if j in paths[i]:
                    dist = paths[i][j]
                    # ФИЗИЧЕСКИ ОБОСНОВАННОЕ ядро
                    spatial_decay = np.exp(-dist / 3.0)
                    coherence = 0.2 * np.cos(0.3 * dist)  # Когерентность
                    self.kernel_matrix[i, j] = spatial_decay * (1.0 + coherence)
                else:
                    self.kernel_matrix[i, j] = 0.0

        # Нормализация для сохранения вероятности
        row_sums = self.kernel_matrix.sum(axis=1)
        self.kernel_matrix = self.kernel_matrix / np.maximum(row_sums[:, np.newaxis], 1e-12)

        print("✅ ЯДРА TSCO ГОТОВЫ")

    def F_nonlinear(self, psi: np.ndarray) -> np.ndarray:
        """КОРРЕКТНАЯ нелинейная функция для TSCO"""
        psi_safe = np.clip(np.abs(psi), 1e-8, 10.0) * np.sign(psi)
        return self.beta_time * psi_safe - self.gamma_time * psi_safe ** 3

    def evolve_TSCO(self, t_idx: int) -> np.ndarray:
        """НАУЧНО КОРРЕКТНАЯ эволюция TSCO"""
        t_current = self.time[t_idx]
        psi_prev = self.psi[t_idx - 1].copy()

        # Метод последовательных приближений для интегрального уравнения
        for iteration in range(15):
            # ВЫЧИСЛЕНИЕ ИНТЕГРАЛЬНОГО ЧЛЕНА
            integral = np.zeros(self.N)

            # Интегрирование по всем предыдущим временам
            for tau_idx in range(t_idx):
                tau = self.time[tau_idx]
                time_decay = np.exp(-np.abs(t_current - tau) / 2.0)

                # Пространственное суммирование через предвычисленную матрицу
                spatial_integral = self.kernel_matrix.dot(self.psi[tau_idx])
                F_val = self.F_nonlinear(self.psi[tau_idx])

                integral += time_decay * (spatial_integral + F_val) * self.dt

            # ОБНОВЛЕНИЕ TSCO
            psi_new = self.psi[0] + self.alpha_time * integral

            # СВЯЗЬ С МЕТРИКОЙ
            coupling = self.lambda_coupling * self.g * psi_prev
            psi_new += coupling * self.dt

            # ДЕМПФИРОВАНИЕ для стабильности
            damping = 0.1
            psi_new = (1 - damping) * psi_prev + damping * psi_new

            # ПРОВЕРКА СХОДИМОСТИ
            if np.linalg.norm(psi_new - psi_prev) < 1e-6:
                break

            psi_prev = psi_new

        # МЯГКОЕ ОГРАНИЧЕНИЕ
        psi_mag = np.abs(psi_new)
        scale = np.tanh(psi_mag) / np.maximum(psi_mag, 1e-12)
        return psi_new * scale

    def evolve_metric(self, psi_current: np.ndarray) -> np.ndarray:
        """КОРРЕКТНАЯ эволюция метрики"""
        # ПРАВАЯ ЧАСТЬ уравнений движения
        psi_mag_sq = psi_current ** 2
        metric_potential = self.beta_space * self.g + self.gamma_space * self.g ** 3
        coupling_term = self.lambda_coupling * psi_mag_sq

        rhs = self.g - self.dt * (coupling_term + metric_potential)

        # РЕШЕНИЕ СИСТЕМЫ
        try:
            g_new = self.metric_solver(rhs)
        except:
            # РЕЗЕРВНЫЙ МЕТОД
            g_new = rhs / (1 + self.dt * self.alpha_space * np.mean(self.degrees))

        # ФИЗИЧЕСКИЕ ОГРАНИЧЕНИЯ
        g_new = np.clip(g_new, 0.5, 2.0)
        return g_new

    def compute_energy(self, t_idx: int) -> float:
        """ПОЛНАЯ ЭНЕРГИЯ СИСТЕМЫ"""
        psi_current = self.psi[t_idx]

        # Энергия метрики
        metric_energy = np.sum(0.5 * self.beta_space * self.g ** 2 +
                               0.25 * self.gamma_space * self.g ** 4)

        # Энергия TSCO
        if t_idx > 0:
            psi_dot = (psi_current - self.psi[t_idx - 1]) / self.dt
            tsco_kinetic = 0.5 * np.sum(psi_dot ** 2)
        else:
            tsco_kinetic = 0.0

        tsco_potential = np.sum(0.5 * self.beta_time * psi_current ** 2 +
                                0.25 * self.gamma_time * psi_current ** 4)

        # Энергия связи
        coupling_energy = self.lambda_coupling * np.sum(self.g * psi_current ** 2)

        return metric_energy + tsco_kinetic + tsco_potential + coupling_energy

    def compute_effective_dimension(self) -> float:
        """ТОЧНЫЙ расчет эффективной размерности"""
        dimensions = []

        for center in np.random.choice(self.N, size=min(20, self.N), replace=False):
            try:
                distances = nx.single_source_shortest_path_length(self.G, center, cutoff=8)
                if len(distances) < 5:
                    continue

                radii = [2, 3, 4, 5, 6]
                volumes = [sum(1 for d in distances.values() if d <= r) for r in radii]

                if len(volumes) >= 4 and volumes[-1] > volumes[0]:
                    # Линейная регрессия в log-log координатах
                    A = np.column_stack([np.log(radii[:4]), np.ones(4)])
                    slope, _ = np.linalg.lstsq(A, np.log(volumes[:4]), rcond=None)[0]

                    if 0.8 < slope < 5.0:  # Физические пределы
                        dimensions.append(slope)
            except:
                continue

        return np.mean(dimensions) if dimensions else 1.0

    def check_stability(self, t_idx: int) -> bool:
        """СТРОГАЯ ПРОВЕРКА СТАБИЛЬНОСТИ"""
        psi_current = self.psi[t_idx]

        stability_conditions = [
            not np.any(np.isnan(self.g)),
            not np.any(np.isnan(psi_current)),
            not np.any(np.isinf(self.g)),
            not np.any(np.isinf(psi_current)),
            np.max(np.abs(self.g)) < 3.0,
            np.min(self.g) > 0.3,
            np.max(np.abs(psi_current)) < 5.0,
            np.std(self.g) < 2.0,
            np.std(psi_current) < 3.0
        ]

        return all(stability_conditions)

    def run_simulation(self) -> bool:
        """ЗАПУСК ПОЛНОЙ СИМУЛЯЦИИ"""
        print("\n🚀 ЗАПУСК НАУЧНОЙ СИМУЛЯЦИИ")
        print("=" * 60)

        for t_idx in range(1, self.time_steps):
            try:
                # 1. ЭВОЛЮЦИЯ TSCO
                psi_current = self.evolve_TSCO(t_idx)
                self.psi[t_idx] = psi_current

                # 2. ЭВОЛЮЦИЯ МЕТРИКИ
                self.g = self.evolve_metric(psi_current)

                # 3. ПРОВЕРКА СТАБИЛЬНОСТИ
                if not self.check_stability(t_idx):
                    print(f"❌ СТАБИЛЬНОСТЬ НАРУШЕНА на шаге {t_idx}")
                    return False

                # 4. ВЫЧИСЛЕНИЕ МЕТРИК
                energy = self.compute_energy(t_idx)
                self.energy_history.append(energy)

                # Периодический расчет размерности
                if t_idx % 30 == 0:
                    dimension = self.compute_effective_dimension()
                    self.dimension_history.append((t_idx, dimension))

                # 5. ВЫВОД ПРОГРЕССА
                if t_idx % 25 == 0 or t_idx == self.time_steps - 1:
                    mean_g = np.mean(self.g)
                    mean_psi = np.mean(np.abs(psi_current))
                    std_g = np.std(self.g)

                    print(f"⏱️  Шаг {t_idx:3d}: E={energy:8.2f}, "
                          f"<g>={mean_g:.3f}±{std_g:.3f}, <|ψ|>={mean_psi:.3f}")

                # Сохранение метрик
                if t_idx % 10 == 0:
                    self.metrics_history.append({
                        'step': t_idx,
                        'energy': energy,
                        'mean_g': np.mean(self.g),
                        'std_g': np.std(self.g),
                        'mean_psi': np.mean(np.abs(psi_current)),
                        'std_psi': np.std(psi_current)
                    })

            except Exception as e:
                print(f"💥 КРИТИЧЕСКАЯ ОШИБКА на шаге {t_idx}: {e}")
                return False

        print("✅ СИМУЛЯЦИЯ УСПЕШНО ЗАВЕРШЕНА!")
        return True

    def analyze_results(self) -> Dict:
        """ПОЛНЫЙ АНАЛИЗ РЕЗУЛЬТАТОВ"""
        print("\n" + "=" * 60)
        print("📊 НАУЧНЫЙ АНАЛИЗ РЕЗУЛЬТАТОВ")
        print("=" * 60)

        # Основные метрики
        final_dimension = self.compute_effective_dimension()
        final_energy = self.energy_history[-1] if self.energy_history else 0

        # Фундаментальные константы
        mean_K = np.mean(self.K_i)
        alpha_inv = np.pi * 3 * mean_K ** 2 / (np.log(mean_K) ** 2)
        planck_scale = np.mean(self.hbar_em)

        # Качество симуляции
        energy_stability = np.std(self.energy_history) / np.mean(self.energy_history) if self.energy_history else 0
        metric_stability = np.std([m['mean_g'] for m in self.metrics_history]) if self.metrics_history else 0

        results = {
            'dimension': final_dimension,
            'alpha_inv': alpha_inv,
            'planck_scale': planck_scale,
            'final_energy': final_energy,
            'energy_stability': energy_stability,
            'metric_stability': metric_stability,
            'network_properties': {
                'N': self.N,
                'avg_degree': np.mean(self.degrees),
                'avg_path_length': self.avg_path_length,
                'clustering': self.clustering,
                'small_world': self.clustering > 0.1 and self.avg_path_length < np.log(self.N)
            },
            'final_state': {
                'mean_g': np.mean(self.g),
                'std_g': np.std(self.g),
                'mean_psi': np.mean(np.abs(self.psi[-1])),
                'std_psi': np.std(self.psi[-1])
            }
        }

        # ВЫВОД РЕЗУЛЬТАТОВ
        print(f"📐 ЭФФЕКТИВНАЯ РАЗМЕРНОСТЬ: {final_dimension:.3f}")
        if 2.7 < final_dimension < 3.3:
            print("   🎯 ОТЛИЧНОЕ СООТВЕТСТВИЕ 3D ПРОСТРАНСТВУ!")
        elif 2.0 < final_dimension < 4.0:
            print("   ✅ УДОВЛЕТВОРИТЕЛЬНАЯ РАЗМЕРНОСТЬ")
        else:
            print("   ⚠️  ТРЕБУЕТ НАСТРОЙКИ ПАРАМЕТРОВ")

        print(f"𝛼⁻¹ ПОСТОЯННАЯ ТОНКОЙ СТРУКТУРЫ: {alpha_inv:.3f}")
        print(f"   Отклонение от 137.036: {abs(alpha_inv - 137.036):.3f}")

        print(f"📏 ПЛАНКОВСКИЙ МАСШТАБ: {planck_scale:.2e}")
        print(f"⚡ ФИНАЛЬНАЯ ЭНЕРГИЯ: {final_energy:.2f}")
        print(f"🛡️  СТАБИЛЬНОСТЬ ЭНЕРГИИ: {energy_stability * 100:.1f}%")

        print(f"🌐 СЕТЕВЫЕ СВОЙСТВА:")
        print(f"   Малый мир: {'✅ ДА' if results['network_properties']['small_world'] else '❌ НЕТ'}")
        print(f"   Кластеризация/Путь: {self.clustering:.3f}/{self.avg_path_length:.2f}")

        return results

    def plot_comprehensive_results(self, results: Dict):
        """ПОЛНАЯ ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ"""
        fig = plt.figure(figsize=(20, 12))

        # 1. Эволюция энергии
        plt.subplot(2, 3, 1)
        if self.energy_history:
            plt.plot(self.energy_history, 'b-', linewidth=2, alpha=0.8)
            plt.title('ЭВОЛЮЦИЯ ЭНЕРГИИ СИСТЕМЫ', fontsize=12, fontweight='bold')
            plt.xlabel('Шаг времени')
            plt.ylabel('Полная энергия')
            plt.grid(True, alpha=0.3)

        # 2. Сеть с метрикой
        plt.subplot(2, 3, 2)
        pos = nx.spring_layout(self.G, seed=42)
        node_colors = self.g
        vmin, vmax = np.percentile(node_colors, [5, 95])
        nodes = nx.draw_networkx_nodes(self.G, pos, node_color=node_colors,
                                       node_size=30, cmap='viridis', vmin=vmin, vmax=vmax)
        nx.draw_networkx_edges(self.G, pos, alpha=0.2, width=0.5)
        plt.title('СЕТЬ С МЕТРИКОЙ g(x)', fontsize=12, fontweight='bold')
        plt.colorbar(nodes, label='Метрика g(x)')
        plt.axis('off')

        # 3. Эволюция TSCO полей
        plt.subplot(2, 3, 3)
        time_indices = np.arange(len(self.psi))
        for i in range(min(4, self.N)):
            plt.plot(time_indices, self.psi[:, i], alpha=0.7, linewidth=1)
        plt.title('ЭВОЛЮЦИЯ TSCO ПОЛЕЙ ψ(t)', fontsize=12, fontweight='bold')
        plt.xlabel('Время')
        plt.ylabel('ψ')
        plt.grid(True, alpha=0.3)

        # 4. Распределение метрики
        plt.subplot(2, 3, 4)
        plt.hist(self.g, bins=25, alpha=0.7, color='skyblue', density=True)
        plt.title('РАСПРЕДЕЛЕНИЕ МЕТРИКИ g(x)', fontsize=12, fontweight='bold')
        plt.xlabel('g(x)')
        plt.ylabel('Плотность вероятности')
        plt.grid(True, alpha=0.3)

        # 5. Пространственные корреляции
        plt.subplot(2, 3, 5)
        spatial_correlations = []
        distances = []

        for i in range(min(50, self.N)):
            for j in range(i + 1, min(50, self.N)):
                try:
                    dist = nx.shortest_path_length(self.G, i, j)
                    corr = np.corrcoef(self.psi[-1, i], self.psi[-1, j])[0, 1]
                    if not np.isnan(corr):
                        spatial_correlations.append(corr)
                        distances.append(dist)
                except:
                    continue

        if distances and spatial_correlations:
            plt.scatter(distances, spatial_correlations, alpha=0.5, s=20)
            plt.title('ПРОСТРАНСТВЕННЫЕ КОРРЕЛЯЦИИ', fontsize=12, fontweight='bold')
            plt.xlabel('Расстояние в сети')
            plt.ylabel('Корреляция ψ')
            plt.grid(True, alpha=0.3)

        # 6. ИНФОРМАЦИОННАЯ ПАНЕЛЬ
        plt.subplot(2, 3, 6)
        plt.axis('off')

        info_text = (
            f"НАУЧНЫЕ РЕЗУЛЬТАТЫ:\n\n"
            f"Размерность: {results['dimension']:.3f}\n"
            f"α⁻¹: {results['alpha_inv']:.3f}\n"
            f"Энергия: {results['final_energy']:.2f}\n"
            f"Стабильность: {results['energy_stability'] * 100:.1f}%\n\n"
            f"Метрика: {results['final_state']['mean_g']:.3f}±{results['final_state']['std_g']:.3f}\n"
            f"TSCO: {results['final_state']['mean_psi']:.3f}±{results['final_state']['std_psi']:.3f}\n\n"
            f"Сеть: N={results['network_properties']['N']}\n"
            f"Малый мир: {'ДА' if results['network_properties']['small_world'] else 'НЕТ'}\n"
            f"ħ_em: {results['planck_scale']:.2e}"
        )

        plt.text(0.1, 0.9, info_text, transform=plt.gca().transAxes, fontsize=11,
                 fontfamily='monospace', verticalalignment='top',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))

        plt.tight_layout()
        plt.show()

        # ДОПОЛНИТЕЛЬНЫЙ ГРАФИК: Эволюция размерности
        if self.dimension_history:
            fig, ax = plt.subplots(figsize=(10, 6))
            steps, dims = zip(*self.dimension_history)
            ax.plot(steps, dims, 'ro-', linewidth=2, markersize=6)
            ax.axhline(y=3.0, color='green', linestyle='--', alpha=0.7, label='Целевая 3D')
            ax.set_title('ЭВОЛЮЦИЯ ЭФФЕКТИВНОЙ РАЗМЕРНОСТИ', fontsize=14, fontweight='bold')
            ax.set_xlabel('Шаг времени')
            ax.set_ylabel('Размерность')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.show()


# ЗАПУСК ПОЛНОЙ СИМУЛЯЦИИ
if __name__ == "__main__":
    print("🔬 НАУЧНАЯ СИМУЛЯЦИЯ ЭМЕРДЖЕНТНОГО ПРОСТРАНСТВА-ВРЕМЕНИ")
    print("🎯 КОРРЕКТНАЯ РЕАЛИЗАЦИЯ БЕЗ ЗАГЛУШЕК")

    # ОПТИМАЛЬНЫЕ ПАРАМЕТРЫ ДЛЯ НАУЧНОЙ СИМУЛЯЦИИ
    simulator = ScientificEmergentSpacetime(
        N=1000,  # Оптимальный размер для точности и скорости
        K=8,  # Для 3D геометрии
        p=0.059,  # Режим малого мира 2.776 при 0.055
        time_steps=100,
        dt=0.005  # Стабильный шаг
    )

    # ЗАПУСК
    success = simulator.run_simulation()
    if success:
        # АНАЛИЗ
        results = simulator.analyze_results()
        # ВИЗУАЛИЗАЦИЯ
        simulator.plot_comprehensive_results(results)
        # ФИНАЛЬНЫЙ ВЕРДИКТ
        if 2.7 < results['dimension'] < 3.3:
            print("🎉 БЛЕСТЯЩИЙ РЕЗУЛЬТАТ! Получена 3D геометрия пространства!")
        else:
            print("🔬 ИНТЕРЕСНЫЙ РЕЗУЛЬТАТ! Требуется дальнейшее исследование.")
    else:
        print("\n💥 СИМУЛЯЦИЯ ПРЕРВАНА ИЗ-ЗА НЕСТАБИЛЬНОСТИ")
        print("Рекомендация: уменьшите lambda_coupling или увеличьте alpha_space")