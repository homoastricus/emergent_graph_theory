import math
import numpy as np
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt


class OptimalNSearcher:
    def __init__(self, K=8.0, p=0.052702):
        self.K = K
        self.p = p
        self.kp = K * p

        # Определяем топ-10 самых точных равенств из ваших результатов
        self.equations = [
            # 1. Глайшеровская константа (самое точное)
            {
                'name': 'glaisher',
                'target': 1.282427129100622,
                'formula': lambda lnN: 1 + p + math.sqrt(p),
                'depends_on_N': False  # Не зависит от N!
            },
            # 2. Гамма(1/3)
            {
                'name': 'gamma(1/3)',
                'target': 2.678938534707748,
                'formula': lambda lnN: (1 - p) * math.sqrt(K),
                'depends_on_N': False
            },
            # 3. Фейгенбаум-2
            {
                'name': 'feigenbaum2',
                'target': 2.502907875095892,
                'formula': lambda lnN: (1 / (1 - p)) / (K * p),
                'depends_on_N': False
            },
            # 4. Фейгенбаум-2 (вторая формула)
            {
                'name': 'feigenbaum2_v2',
                'target': 2.502907875095892,
                'formula': lambda lnN: math.log(K) + ((K + p) * p),
                'depends_on_N': False
            },
            # 5. Каталан
            {
                'name': 'catalan',
                'target': 0.915965594177219,
                'formula': lambda lnN: p - math.log(K * p),
                'depends_on_N': False
            },
            # 6. Хинчин (УЖЕ ЗАВИСИТ ОТ N!)
            {
                'name': 'khinchin',
                'target': 2.685452001065306,
                'formula': lambda lnN: math.sqrt(K) / (1 + p),
                'depends_on_N': False
            },
            # 7. Мейссель-Мертенс
            {
                'name': 'meissel_mertens',
                'target': 0.261497212847642,
                'formula': lambda lnN: math.log(K) / (K - p),
                'depends_on_N': False
            },
            # 8. Фейгенбаум-2 (третья формула)
            {
                'name': 'feigenbaum2_v3',
                'target': 2.502907875095892,
                'formula': lambda lnN: (K * p) + math.log(K),
                'depends_on_N': False
            },
            # 9. Эйлер-Маскерони
            {
                'name': 'euler_mascheroni',
                'target': 0.577215664901532,
                'formula': lambda lnN: p * (K - math.log(p)),
                'depends_on_N': False
            },
            # 10. π/3
            {
                'name': 'pi/3',
                'target': math.pi / 3,
                'formula': lambda lnN: p + (K / (K + p)),
                'depends_on_N': False
            },
        ]

        # Теперь добавим уравнения, которые действительно зависят от N
        self.equations_dependent = [
            # Ваше ключевое уравнение p*sqrt(K*U) = e
            {
                'name': 'p*sqrt(K*U)=e',
                'target': math.e,
                'formula': lambda lnN: p * math.sqrt(K * (lnN / abs(math.log(K * p)))),
                'depends_on_N': True
            },
            # Ещё пример: отношение логарифмов
            {
                'name': 'ln(N)/ln(Kp)',
                'target': 330.0,  # Примерное значение U
                'formula': lambda lnN: lnN / abs(math.log(K * p)),
                'depends_on_N': True
            },
            # Формула из вашего кода для N
            {
                'name': 'N_from_formula',
                'target': 0,  # Здесь будем считать отклонение
                'formula': lambda lnN: (math.e ** 2 * abs(math.log(K * p))) / (p ** 2 * K),
                'depends_on_N': True
            }
        ]

        # Объединяем
        self.all_equations = self.equations + self.equations_dependent

    def calculate_errors(self, N: float) -> Dict[str, float]:
        """Вычисление ошибок для данного N"""
        lnN = math.log(N)
        errors = {}

        for eq in self.all_equations:
            try:
                value = eq['formula'](lnN)

                if eq['name'] == 'N_from_formula':
                    # Для этой формулы сравниваем ln(N) с предсказанным
                    predicted_lnN = value
                    error = abs(lnN - predicted_lnN) / abs(lnN)
                else:
                    # Для остальных сравниваем с целевым значением
                    error = abs(value - eq['target']) / abs(eq['target'])

                errors[eq['name']] = error

            except (ValueError, ZeroDivisionError):
                errors[eq['name']] = float('inf')

        return errors

    def search_optimal_N(self,
                         N_min: float = 1e122,
                         N_max: float = 1e145,
                         num_points: int = 1000,
                         plot_results: bool = True) -> Tuple[float, float, Dict]:
        """
        Поиск оптимального N в логарифмическом диапазоне

        Returns: (best_N, best_avg_error, results_dict)
        """
        # Создаём логарифмически равномерную сетку
        log_N_min = math.log(N_min)
        log_N_max = math.log(N_max)
        log_N_values = np.linspace(log_N_min, log_N_max, num_points)
        N_values = np.exp(log_N_values)

        # Массивы для результатов
        avg_errors = []
        best_equations_errors = []

        # Для отладки: собираем все ошибки
        all_errors_dict = {eq['name']: [] for eq in self.all_equations}

        print(f"Поиск оптимального N в диапазоне: {N_min:.1e} ... {N_max:.1e}")
        print(f"Количество точек: {num_points}")
        print("-" * 60)

        best_avg_error = float('inf')
        best_N = None
        best_errors_dict = {}

        for i, N in enumerate(N_values):
            errors = self.calculate_errors(N)

            # Игнорируем уравнения, не зависящие от N
            dependent_errors = []
            for eq in self.all_equations:
                if eq['depends_on_N'] and errors[eq['name']] < float('inf'):
                    dependent_errors.append(errors[eq['name']])

            if dependent_errors:
                avg_error = np.mean(dependent_errors)
            else:
                avg_error = float('inf')

            avg_errors.append(avg_error)

            # Сохраняем ошибки для каждого уравнения
            for name, error in errors.items():
                all_errors_dict[name].append(error)

            # Сохраняем лучшие 5 уравнений (не зависящих от N)
            independent_errors = []
            for eq in self.equations:
                if not eq['depends_on_N']:
                    independent_errors.append(errors[eq['name']])

            if independent_errors:
                best_eq_error = np.mean(sorted(independent_errors)[:5])  # Среднее 5 лучших
                best_equations_errors.append(best_eq_error)
            else:
                best_equations_errors.append(float('inf'))

            # Обновляем лучшее значение
            if avg_error < best_avg_error and avg_error > 0:
                best_avg_error = avg_error
                best_N = N
                best_errors_dict = errors.copy()

            # Прогресс
            if i % (num_points // 20) == 0:
                print(f"  N={N:.2e}, средняя ошибка={avg_error:.2e}")

        print("-" * 60)
        print(f"Наилучшее N = {best_N:.6e}")
        print(f"Лучшая средняя ошибка = {best_avg_error:.6e} ({best_avg_error * 100:.8f}%)")
        print(f"log10(N) = {math.log10(best_N):.6f}")
        print(f"ln(N) = {math.log(best_N):.6f}")

        # Выводим точные ошибки для лучшего N
        print("\nОшибки для лучшего N:")
        print("-" * 60)
        for name, error in sorted(best_errors_dict.items(), key=lambda x: x[1]):
            if error < float('inf'):
                print(f"{name:25}: {error:.6e} ({error * 100:.8f}%)")

        if plot_results:
            self._plot_results(N_values, avg_errors, best_equations_errors,
                               all_errors_dict, best_N)

        return best_N, best_avg_error, best_errors_dict

    def _plot_results(self, N_values, avg_errors, best_eq_errors,
                      all_errors_dict, best_N):
        """Построение графиков результатов"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Средняя ошибка vs N
        ax1 = axes[0, 0]
        ax1.semilogx(N_values, avg_errors, 'b-', linewidth=2, label='Средняя ошибка')
        ax1.semilogx(N_values, best_eq_errors, 'r--', linewidth=1.5,
                     label='Лучшие 5 уравнений')
        ax1.axvline(best_N, color='g', linestyle=':', linewidth=2,
                    label=f'Оптимальное N={best_N:.2e}')
        ax1.set_xlabel('N')
        ax1.set_ylabel('Относительная ошибка')
        ax1.set_title('Поиск оптимального N')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Ошибки ключевых уравнений
        ax2 = axes[0, 1]
        key_equations = ['p*sqrt(K*U)=e', 'ln(N)/ln(Kp)', 'N_from_formula']
        colors = ['r', 'g', 'b']
        for eq_name, color in zip(key_equations, colors):
            if eq_name in all_errors_dict:
                errors = all_errors_dict[eq_name]
                ax2.semilogx(N_values, errors, color + '-', linewidth=1.5,
                             label=eq_name)
        ax2.axvline(best_N, color='k', linestyle=':', linewidth=2)
        ax2.set_xlabel('N')
        ax2.set_ylabel('Относительная ошибка')
        ax2.set_title('Ключевые уравнения')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Логарифмический масштаб N
        ax3 = axes[1, 0]
        log_N = np.log10(N_values)
        ax3.plot(log_N, avg_errors, 'b-', linewidth=2)
        ax3.axvline(math.log10(best_N), color='g', linestyle=':', linewidth=2,
                    label=f'log10(N)={math.log10(best_N):.2f}')
        ax3.set_xlabel('log10(N)')
        ax3.set_ylabel('Средняя ошибка')
        ax3.set_title('Зависимость от log10(N)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Гистограмма ошибок для лучшего N
        ax4 = axes[1, 1]
        errors_for_best = []
        eq_names = []
        for eq in self.all_equations:
            if eq['depends_on_N']:
                eq_names.append(eq['name'])
                # Находим индекс best_N
                idx = np.argmin(np.abs(N_values - best_N))
                errors_for_best.append(all_errors_dict[eq['name']][idx])

        if errors_for_best:
            bars = ax4.bar(range(len(errors_for_best)), errors_for_best)
            ax4.set_xticks(range(len(errors_for_best)))
            ax4.set_xticklabels(eq_names, rotation=45, ha='right')
            ax4.set_ylabel('Относительная ошибка')
            ax4.set_title(f'Ошибки при N={best_N:.2e}')
            ax4.grid(True, alpha=0.3, axis='y')

            # Подписываем значения
            for bar, error in zip(bars, errors_for_best):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width() / 2., height,
                         f'{error:.1e}', ha='center', va='bottom', rotation=90)

        plt.tight_layout()
        plt.savefig('optimal_N_search.png', dpi=150)
        plt.show()

    def fine_search_around_best(self, initial_N: float,
                                factor_range: float = 1e3,
                                num_iterations: int = 5) -> float:
        """
        Точный поиск оптимального N вокруг начального значения

        Используем метод золотого сечения для минимизации ошибки
        """
        print(f"\nТочный поиск вокруг N = {initial_N:.6e}")
        print("=" * 60)

        def objective_function(lnN):
            """Функция для минимизации: средняя ошибка зависимых уравнений"""
            N = math.exp(lnN)
            errors = self.calculate_errors(N)

            dependent_errors = []
            for eq in self.all_equations:
                if eq['depends_on_N'] and errors[eq['name']] < float('inf'):
                    dependent_errors.append(errors[eq['name']])

            if not dependent_errors:
                return float('inf')

            return np.mean(dependent_errors)

        # Метод золотого сечения
        from scipy.optimize import minimize_scalar

        # Ищем в логарифмическом пространстве
        lnN_initial = math.log(initial_N)
        lnN_min = lnN_initial - math.log(factor_range)
        lnN_max = lnN_initial + math.log(factor_range)

        result = minimize_scalar(
            objective_function,
            bounds=(lnN_min, lnN_max),
            method='bounded',
            options={'xatol': 1e-6, 'maxiter': 50}
        )

        if result.success:
            optimal_lnN = result.x
            optimal_N = math.exp(optimal_lnN)
            min_error = result.fun

            print(f"Точный оптимум найден:")
            print(f"  N = {optimal_N:.10e}")
            print(f"  ln(N) = {optimal_lnN:.10f}")
            print(f"  log10(N) = {optimal_lnN / math.log(10):.10f}")
            print(f"  Минимальная ошибка = {min_error:.10e} ({min_error * 100:.12f}%)")

            # Выводим все уравнения для этого N
            print(f"\nПроверка всех уравнений для оптимального N:")
            print("-" * 60)
            errors = self.calculate_errors(optimal_N)
            for name, error in sorted(errors.items(), key=lambda x: x[1]):
                if error < float('inf'):
                    print(f"{name:25}: {error:.10e} ({error * 100:.12f}%)")

            return optimal_N
        else:
            print(f"Оптимизация не удалась: {result.message}")
            return initial_N


def main():
    """Основная функция поиска оптимального N"""
    print("ПОИСК ОПТИМАЛЬНОГО N ДЛЯ МАКСИМАЛЬНОЙ ТОЧНОСТИ")
    print("=" * 60)

    # Инициализация поисковика
    searcher = OptimalNSearcher(K=8.0, p=0.052702)

    # 1. Грубый поиск в широком диапазоне
    print("\n1. ГРУБЫЙ ПОИСК В ШИРОКОМ ДИАПАЗОНЕ:")
    best_N, best_error, best_errors = searcher.search_optimal_N(
        N_min=1e122,  # Ваше текущее значение
        N_max=1e145,  # До 10^145
        num_points=2000,  # 2000 точек для гладкого поиска
        plot_results=True
    )

    # 2. Точный поиск вокруг найденного значения
    print("\n2. ТОЧНЫЙ ПОИСК ВОКРУГ НАЙДЕННОГО ЗНАЧЕНИЯ:")
    optimal_N = searcher.fine_search_around_best(
        initial_N=best_N,
        factor_range=10,  # Ищем в диапазоне best_N/10 ... best_N*10
        num_iterations=10
    )

    # 3. Проверка с другим начальным приближением
    print("\n3. ПРОВЕРКА С ДРУГИМИ НАЧАЛЬНЫМИ ЗНАЧЕНИЯМИ:")
    test_points = [
        9.7e122,  # Ваше исходное значение
        1.0e124,  # Чуть больше
        1.0e123,  # Чуть меньше
        5.0e124,  # Из вашего расчёта p*sqrt(K*U)=e
        1.047e147,  # Из вашего раннего кода
    ]

    for test_N in test_points:
        errors = searcher.calculate_errors(test_N)
        dependent_errors = []
        for eq in searcher.all_equations:
            if eq['depends_on_N'] and errors[eq['name']] < float('inf'):
                dependent_errors.append(errors[eq['name']])

        if dependent_errors:
            avg_error = np.mean(dependent_errors)
            print(f"N={test_N:.2e}: средняя ошибка={avg_error:.6e}")

    # 4. Вывод итоговых результатов
    print("\n" + "=" * 60)
    print("ИТОГОВЫЕ РЕКОМЕНДАЦИИ:")
    print("=" * 60)

    print(f"\nТекущее значение: N = 9.70e+122")
    print(f"Найденный оптимум: N = {optimal_N:.6e}")

    ratio = optimal_N / 9.7e122
    print(f"Отношение: {ratio:.6f} (log10(отношения) = {math.log10(ratio):.6f})")

    # Вычисляем U для оптимального N
    U_optimal = math.log(optimal_N) / abs(math.log(8.0 * 0.052702))
    print(f"\nU = ln(N)/|ln(Kp)|:")
    print(f"  Для N=9.70e+122: U ≈ {math.log(9.7e122) / abs(math.log(8 * 0.052702)):.6f}")
    print(f"  Для оптимального N: U ≈ {U_optimal:.6f}")

    # Проверяем ключевое уравнение p*sqrt(K*U)=e
    left_side = 0.052702 * math.sqrt(8.0 * U_optimal)
    print(f"\nПроверка уравнения p*sqrt(K*U) = e:")
    print(f"  Левая часть: {left_side:.10f}")
    print(f"  Правая часть (e): {math.e:.10f}")
    print(f"  Относительная ошибка: {abs(left_side - math.e) / math.e * 100:.10f}%")

    print("\n" + "=" * 60)
    print("РЕКОМЕНДАЦИЯ:")
    if abs(ratio - 1) < 0.1:
        print("Текущее N уже близко к оптимальному (<10% отличие)")
    else:
        print(f"Рассмотрите использование N ≈ {optimal_N:.3e}")
        print(f"Это даст улучшение точности в ~{1 / best_error:.1f} раз")


if __name__ == "__main__":
    main()