import numpy as np


class MetricComparison:
    """Сравнение метрики для разных геометрических структур"""

    def __init__(self, total_cells=90000000):
        self.total_cells = total_cells
        self.l_p = 1.0
        self.xi = 1.648721  # e-1
        self.sigma_0 = 0.367879  # 1/e

    def generate_cubic_coords(self):
        """Генерирует кубическую решетку с ~total_cells ячеек"""
        size = int((self.total_cells ** (1 / 3)) / 2)
        coords = []
        for x in range(-size, size + 1):
            for y in range(-size, size + 1):
                for z in range(-size, size + 1):
                    coords.append([x, y, z])
        return np.array(coords)

    def generate_random_coords(self):
        """Генерирует случайное равномерное распределение"""
        side = int(self.total_cells ** (1 / 3))
        coords = np.random.rand(self.total_cells, 3) * side * 2 - side
        return coords

    def generate_spherical_coords(self):
        """Генерирует точки на сфере (радиальное распределение)"""
        # Случайные направления
        phi = np.random.uniform(0, 2 * np.pi, self.total_cells)
        costheta = np.random.uniform(-1, 1, self.total_cells)
        theta = np.arccos(costheta)

        # Радиусы с равномерной плотностью в объеме
        r = np.random.uniform(0, self.total_cells ** (1 / 3), self.total_cells) ** (1 / 3)

        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)

        return np.column_stack([x, y, z])

    def sigma_r(self, r):
        """Функция шума метрики (одинаковая для всех геометрий)"""
        r_eff = np.maximum(r, self.l_p)
        quantum = self.sigma_0 * np.exp(-r_eff / self.xi)
        residual = self.l_p / r_eff
        holographic = np.sqrt(32.0 / self.total_cells)
        return quantum + residual + holographic

    def compute_metric(self, coords):
        """Вычисляет метрику для данной геометрии"""
        # Вычисляем расстояния от центра
        center = np.array([0, 0, 0])
        distances = np.linalg.norm(coords - center, axis=1)

        # Фильтруем ненулевые расстояния
        mask = distances > 0
        r_valid = distances[mask]

        if len(r_valid) == 0:
            return 0, 0, np.array([])

        # Генерируем α(r) с одинаковой функцией шума
        sigma_values = self.sigma_r(r_valid)
        alpha = np.random.normal(2.0, sigma_values)

        # Применяем физические ограничения
        alpha = np.clip(alpha, 1.0, 3.0)

        return np.mean(alpha), np.std(alpha), alpha

    def run_comparison(self):
        """Запускает сравнение метрики для разных геометрий"""
        print("ЭКСПЕРИМЕНТ: МЕТРИКА ДЛЯ РАЗНЫХ ГЕОМЕТРИЧЕСКИХ СТРУКТУР")
        print(f"Общее число ячеек: {self.total_cells:,}")

        geometries = {
            "Кубическая решетка": self.generate_cubic_coords(),
            "Случайное распределение": self.generate_random_coords(),
            "Сферическое распределение": self.generate_spherical_coords()
        }

        results = {}

        for name, coords in geometries.items():
            print(f"\n{name}:")
            print("-" * 40)

            mean_alpha, std_alpha, alpha_values = self.compute_metric(coords)

            # Анализ по масштабам
            distances = np.linalg.norm(coords, axis=1)
            mask = distances > 0
            r_valid = distances[mask]

            # Разбиваем на масштабные зоны
            near_mask = r_valid <= 1.0
            mid_mask = (r_valid > 1.0) & (r_valid <= 5.0)
            far_mask = r_valid > 5.0

            near_alpha = alpha_values[near_mask] if np.any(near_mask) else np.array([2.0])
            mid_alpha = alpha_values[mid_mask] if np.any(mid_mask) else np.array([2.0])
            far_alpha = alpha_values[far_mask] if np.any(far_mask) else np.array([2.0])

            results[name] = {
                'mean_alpha': mean_alpha,
                'std_alpha': std_alpha,
                'near': (len(near_alpha), np.mean(near_alpha), np.std(near_alpha)),
                'mid': (len(mid_alpha), np.mean(mid_alpha), np.std(mid_alpha)),
                'far': (len(far_alpha), np.mean(far_alpha), np.std(far_alpha))
            }

            print(f"  Всего точек: {len(alpha_values):,}")
            print(f"  ⟨α⟩ = {mean_alpha:.6f} ± {std_alpha:.6f}")
            print(
                f"  Планковский (r ≤ 1): {len(near_alpha)} яч., ⟨α⟩ = {np.mean(near_alpha):.4f} ± {np.std(near_alpha):.4f}")
            print(
                f"  Промежуточный (1 < r ≤ 5): {len(mid_alpha)} яч., ⟨α⟩ = {np.mean(mid_alpha):.4f} ± {np.std(mid_alpha):.4f}")
            print(
                f"  Макроскопический (r > 5): {len(far_alpha)} яч., ⟨α⟩ = {np.mean(far_alpha):.6f} ± {np.std(far_alpha):.6f}")

        return results


# ДОПОЛНИТЕЛЬНЫЙ ЭКСПЕРИМЕНТ: Влияние размера системы
def run_size_experiment():
    """Проверяет как меняется точность с размером системы"""
    print("ЭКСПЕРИМЕНТ 2: ВЛИЯНИЕ РАЗМЕРА СИСТЕМЫ")

    sizes = [1000000, 5000000, 20000000]

    for size in sizes:
        print(f"\nРазмер системы: {size:,} ячеек")
        print("-" * 30)

        comparator = MetricComparison(size)
        coords = comparator.generate_cubic_coords()

        mean_alpha, std_alpha, alpha_values = comparator.compute_metric(coords)

        print(f"  ⟨α⟩ = {mean_alpha:.6f} ± {std_alpha:.6f}")
        print(f"  Относительная погрешность: {std_alpha / mean_alpha * 100:.4f}%")


# ЗАПУСК ЭКСПЕРИМЕНТОВ
if __name__ == "__main__":
    # Основной эксперимент
    comparator = MetricComparison(50000)
    results = comparator.run_comparison()

    # Дополнительный эксперимент с размерами
    run_size_experiment()

    # ФИНАЛЬНЫЙ АНАЛИЗ
    print("ФИНАЛЬНЫЙ АНАЛИЗ")

    print(f"\n{'Геометрия':<25} {'⟨α⟩':<12} {'σ(α)':<10} {'σ/⟨α⟩, %':<10}")

    for name, data in results.items():
        rel_error = data['std_alpha'] / data['mean_alpha'] * 100
        print(f"{name:<25} {data['mean_alpha']:.6f}  {data['std_alpha']:.6f}  {rel_error:.4f}%")

    # Проверка гипотезы
    std_values = [data['std_alpha'] for data in results.values()]
    max_std = max(std_values)
    min_std = min(std_values)
    variation = (max_std - min_std) / np.mean(std_values) * 100

    print(f"\n📊 Анализ погрешностей:")
    print(f"   Максимальная σ(α): {max_std:.6f}")
    print(f"   Минимальная σ(α): {min_std:.6f}")
    print(f"   Разброс погрешностей: {variation:.2f}%")

    if variation < 10:  # Если разброс меньше 10%
        print(f"\n🎯 ВЫВОД: Метрика НЕ ЗАВИСИТ от геометрии!")
        print("   Погрешность σ(α) практически одинакова для всех структур.")
    else:
        print(f"\n⚠️ ВЫВОД: Метрика зависит от геометрии.")
        print("   Погрешность σ(α) различается для разных структур.")