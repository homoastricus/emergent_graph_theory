import numpy as np
import math
from scipy.optimize import minimize, differential_evolution
import matplotlib.pyplot as plt


class NetworkConstantsOptimizer:
    def __init__(self):
        # Известные математические константы
        self.constants = {
            'pi': math.pi,
            'e': math.e,
            'G': 1.2824271291006226,  # Константа Глайшера
            'gamma_1_3': 2.6789385347077476,  # Γ(1/3)
            'feigenbaum': 2.5029078750958928,  # α Фейгенбаума
            'planck_mass': 2.176434e-8,
            'planck_time': 5.391247e-44,
            'hbar': 1.054571817e-34,
            'c': 299792458,
            'G_grav': 6.67430e-11
        }

        # Все возможные формулы для α
        self.alpha_formulas = [
            lambda K, p: (1 / (1 - p)) * math.sqrt(K * p),  # Ваш вариант
            lambda K, p: 1 / ((1 - p) * math.sqrt(K * p)),  # Инверсный вариант
            lambda K, p: K * p / (1 - p),  # Линейный вариант
            lambda K, p: math.sqrt(K) / (1 - p),  # Альтернативный
            lambda K, p: (K + p) / (1 - p),  # Еще вариант
        ]

        self.alpha_formula_names = [
            "(1/(1-p))*√(Kp)",
            "1/((1-p)*√(Kp))",
            "Kp/(1-p)",
            "√K/(1-p)",
            "(K+p)/(1-p)"
        ]

    def calculate_U(self, N, K, p):
        """Вычисление U = lnN / |ln((K+p)p)|"""
        return math.log(N) / abs(math.log((K + p) * p))

    def find_best_alpha_formula(self, K, p):
        """Находит лучшую формулу для α среди возможных"""
        best_formula_idx = 0
        best_error = float('inf')
        best_value = 0

        for i, formula in enumerate(self.alpha_formulas):
            try:
                alpha_value = formula(K, p)
                error = abs(alpha_value - self.constants['feigenbaum']) / self.constants['feigenbaum'] * 100

                if error < best_error:
                    best_error = error
                    best_formula_idx = i
                    best_value = alpha_value
            except:
                continue

        return best_formula_idx, best_value, best_error

    def calculate_all_equations(self, K, p, N, alpha_formula_idx=None):
        """Вычисление всех тождеств"""
        U = self.calculate_U(N, K, p)

        # Находим лучшую формулу для α
        if alpha_formula_idx is None:
            alpha_formula_idx, alpha_value, _ = self.find_best_alpha_formula(K, p)
        else:
            alpha_value = self.alpha_formulas[alpha_formula_idx](K, p)

        equations = {
            # 1. π-уравнение
            'pi_eq': math.log(K + p) + 1 / (1 - p),
            'pi_target': self.constants['pi'],

            # 2. e-уравнение
            'e_eq': p * math.sqrt((K + p) * U),
            'e_target': self.constants['e'],

            # 3. G-уравнение (Глайшера)
            'G_eq': 1 + math.sqrt(p) + p,
            'G_target': self.constants['G'],

            # 4. Γ(1/3)-уравнение
            'gamma_eq': (1 - p) * math.sqrt(K),
            'gamma_target': self.constants['gamma_1_3'],

            # 5. α-уравнение (Фейгенбаума)
            'alpha_eq': alpha_value,
            'alpha_target': self.constants['feigenbaum'],
            'alpha_formula_idx': alpha_formula_idx,
            'alpha_formula_name': self.alpha_formula_names[alpha_formula_idx],

            # U для информации
            'U': U
        }

        return equations

    def calculate_errors(self, equations):
        """Вычисление относительных ошибок"""
        errors = {}

        # Вычисляем ошибки для каждого уравнения
        for key in ['pi', 'e', 'G', 'gamma', 'alpha']:
            eq_val = equations[f'{key}_eq']
            target = equations[f'{key}_target']
            if target != 0:
                errors[f'{key}_error'] = abs(eq_val - target) / target * 100
            else:
                errors[f'{key}_error'] = abs(eq_val - target)

        return errors

    def objective_function(self, params, alpha_formula_idx=None, print_details=False):
        """Целевая функция для оптимизации"""
        K, p, logN = params

        # Ограничения на параметры
        if K < 1 or K > 20:
            return 1e9
        if p <= 0 or p >= 1:
            return 1e9
        if logN < 100 or logN > 300:
            return 1e9

        N = math.exp(logN)
        equations = self.calculate_all_equations(K, p, N, alpha_formula_idx)
        errors = self.calculate_errors(equations)

        # Суммарная ошибка (взвешенная)
        total_error = (
                errors['pi_error'] * 1.0 +
                errors['e_error'] * 1.0 +  # e-уравнение очень важное
                errors['G_error'] * 0.5 +
                errors['gamma_error'] * 0.5 +
                errors['alpha_error'] * 1.0  # α тоже важно
        )

        # Штраф за большие отклонения
        penalty = 0
        for err_name, err_value in errors.items():
            if err_value > 0.1:  # Штрафуем ошибки > 0.1%
                penalty += (err_value - 0.1) * 10

        total_score = total_error + penalty

        if print_details:
            print(f"\nK={K:.6f}, p={p:.9f}, N={N:.3e}")
            print(f"U={equations['U']:.3f}, α формула: {equations['alpha_formula_name']}")
            print("-" * 60)
            for key in ['pi', 'e', 'G', 'gamma', 'alpha']:
                eq_val = equations[f'{key}_eq']
                target = equations[f'{key}_target']
                error = errors[f'{key}_error']
                print(f"{key:>6}: {eq_val:.12f} vs {target:.12f} | error={error:.6f}%")
            print(f"Total score: {total_score:.6f}")

        return total_score

    def optimize_for_alpha_formula(self, formula_idx):
        """Оптимизация для конкретной формулы α"""
        print(f"\n🔍 Оптимизация для α-формулы: {self.alpha_formula_names[formula_idx]}")

        bounds = [
            (7.5, 8.5),  # K
            (0.05, 0.06),  # p
            (math.log(1e120), math.log(1e125))  # logN
        ]

        result = differential_evolution(
            lambda params: self.objective_function(params, formula_idx),
            bounds,
            maxiter=200,
            popsize=30,
            tol=1e-10,
            disp=False
        )

        K_opt, p_opt, logN_opt = result.x
        N_opt = math.exp(logN_opt)
        score = result.fun

        print(f"  K={K_opt:.8f}, p={p_opt:.8f}, N={N_opt:.3e}, score={score:.4f}")

        return K_opt, p_opt, N_opt, score, formula_idx

    def find_optimal_parameters(self):
        """Поиск оптимальных параметров"""
        print("=" * 80)
        print("ПОИСК ОПТИМАЛЬНЫХ ПАРАМЕТРОВ СЕТИ")
        print("=" * 80)

        # Тестируем все формулы для α
        best_params = None
        best_score = float('inf')
        best_formula_idx = 0

        print("\n🔍 Тестирование различных формул для α:")
        for i in range(len(self.alpha_formulas)):
            try:
                K, p, N, score, formula_idx = self.optimize_for_alpha_formula(i)

                if score < best_score:
                    best_score = score
                    best_params = (K, p, N)
                    best_formula_idx = formula_idx
            except Exception as e:
                print(f"  Формула {i} не удалась: {e}")

        if best_params:
            K_opt, p_opt, N_opt = best_params

            # Уточняем локальной оптимизацией
            print(f"\n🔍 Уточнение лучшего решения (формула {best_formula_idx})...")
            x0 = [K_opt, p_opt, math.log(N_opt)]
            result_local = minimize(
                lambda params: self.objective_function(params, best_formula_idx),
                x0,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12}
            )

            K_final, p_final, logN_final = result_local.x
            N_final = math.exp(logN_final)
            final_score = result_local.fun

            print(f"\n✅ Оптимальные параметры:")
            print(f"K = {K_final:.10f}")
            print(f"p = {p_final:.10f}")
            print(f"N = {N_final:.6e}")
            print(f"Формула α: {self.alpha_formula_names[best_formula_idx]}")
            print(f"Счет: {final_score:.6f}")

            return K_final, p_final, N_final, best_formula_idx

        return None, None, None, None

    def verify_solution(self, K, p, N, alpha_formula_idx=None):
        """Детальная проверка решения"""
        print("\n" + "=" * 80)
        print("ДЕТАЛЬНАЯ ПРОВЕРКА РЕШЕНИЯ")
        print("=" * 80)

        if alpha_formula_idx is None:
            alpha_formula_idx, _, _ = self.find_best_alpha_formula(K, p)

        equations = self.calculate_all_equations(K, p, N, alpha_formula_idx)
        errors = self.calculate_errors(equations)

        print(f"\n📊 Параметры:")
        print(f"K = {K:.10f}")
        print(f"p = {p:.10f}")
        print(f"N = {N:.6e}")
        print(f"U = lnN/|ln(Kp)| = {equations['U']:.6f}")
        print(f"Формула α: {equations['alpha_formula_name']}")

        print("\n📈 Уравнения и ошибки:")
        print(f"{'Ураянение':<15} {'Левая часть':<20} {'Константа':<20} {'Ошибка':<15}")


        for key in ['pi', 'e', 'G', 'gamma', 'alpha']:
            eq_val = equations[f'{key}_eq']
            target = equations[f'{key}_target']
            error = errors[f'{key}_error']

            error_str = f"{error:.6f}%"
            if error < 0.001:
                error_str = f"\033[92m{error_str}\033[0m"  # Зеленый
            elif error < 0.01:
                error_str = f"\033[93m{error_str}\033[0m"  # Желтый
            elif error < 0.1:
                error_str = f"\033[96m{error_str}\033[0m"  # Голубой
            else:
                error_str = f"\033[91m{error_str}\033[0m"  # Красный

            print(f"{key:<15} {eq_val:<20.12f} {target:<20.12f} {error_str:<15}")

        print("-" * 70)

        # Средняя ошибка
        avg_error = np.mean([errors[f'{key}_error'] for key in ['pi', 'e', 'G', 'gamma', 'alpha']])
        max_error = max([errors[f'{key}_error'] for key in ['pi', 'e', 'G', 'gamma', 'alpha']])

        print(f"\n📊 Статистика:")
        print(f"Средняя ошибка: {avg_error:.8f}%")
        print(f"Максимальная ошибка: {max_error:.8f}%")

        if avg_error < 0.01:
            print("\n✅ Отличное решение! Все тождества выполняются с высокой точностью.")
        elif avg_error < 0.1:
            print("\n⚠️ Хорошее решение, можно использовать.")
        elif avg_error < 1.0:
            print("\n⚠️ Удовлетворительное решение, но требует улучшения.")
        else:
            print("\n❌ Решение требует значительной доработки.")

        return equations, errors

    def analyze_around_original(self):
        """Анализ окрестности исходных параметров"""
        print("АНАЛИЗ ОКРЕСТНОСТИ ИСХОДНЫХ ПАРАМЕТРОВ")

        K0, p0, N0 = 8.0, 0.05270179, 9.702e122

        print(f"\nИсходные параметры: K={K0}, p={p0}, N={N0:.3e}")

        # Находим лучшую формулу для α при исходных параметрах
        best_idx, best_alpha, best_error = self.find_best_alpha_formula(K0, p0)
        print(f"Лучшая формула α: {self.alpha_formula_names[best_idx]}")
        print(f"α = {best_alpha:.6f} (ошибка: {best_error:.4f}%)")

        # Проверяем исходные параметры
        print("\n🔍 Проверка исходных параметров:")
        self.verify_solution(K0, p0, N0, best_idx)

        # Пробуем небольшие вариации
        print("\n🔍 Поиск улучшений в окрестности:")

        best_K, best_p, best_N = K0, p0, N0
        best_avg_error = float('inf')

        for dK in np.linspace(-0.01, 0.01, 5):
            for dp in np.linspace(-0.0001, 0.0001, 5):
                for dlogN in np.linspace(-0.1, 0.1, 5):
                    K_test = K0 + dK
                    p_test = p0 + dp
                    N_test = N0 * math.exp(dlogN)

                    equations = self.calculate_all_equations(K_test, p_test, N_test, best_idx)
                    errors = self.calculate_errors(equations)

                    avg_error = np.mean([errors[f'{key}_error'] for key in ['pi', 'e', 'G', 'gamma', 'alpha']])

                    if avg_error < best_avg_error:
                        best_avg_error = avg_error
                        best_K, best_p, best_N = K_test, p_test, N_test

        print(f"\n✅ Улучшенные параметры:")
        print(f"K = {best_K:.10f} (Δ={best_K - K0:+.6f})")
        print(f"p = {best_p:.10f} (Δ={best_p - p0:+.6f})")
        print(f"N = {best_N:.6e} (отношение к исходному: {best_N / N0:.6f})")

        self.verify_solution(best_K, best_p, best_N, best_idx)

        return best_K, best_p, best_N, best_idx

    def run_focused_analysis(self):
        """Целевой анализ с фиксированными K=8"""
        print("🚀 ЗАПУСК ЦЕЛЕВОГО АНАЛИЗА (K=8)")
        print("=" * 80)

        # Фиксируем K=8 и ищем оптимальные p, N
        K_fixed = 8.0

        print("\n🔍 Поиск оптимальных p и N при K=8...")

        best_score = float('inf')
        best_p = None
        best_N = None
        best_formula_idx = None

        # Тестируем все формулы для α
        for formula_idx in range(len(self.alpha_formulas)):
            # Оптимизируем только p и N
            bounds = [
                (0.0526, 0.0528),  # p
                (math.log(1e122), math.log(1e123))  # logN
            ]

            result = differential_evolution(
                lambda params: self.objective_function([K_fixed, params[0], params[1]], formula_idx),
                bounds,
                maxiter=100,
                popsize=20,
                tol=1e-10,
                disp=False
            )

            p_opt, logN_opt = result.x
            N_opt = math.exp(logN_opt)
            score = result.fun

            if score < best_score:
                best_score = score
                best_p = p_opt
                best_N = N_opt
                best_formula_idx = formula_idx

        if best_p is not None:
            print(f"\n✅ Найденные параметры при K=8:")
            print(f"p = {best_p:.10f}")
            print(f"N = {best_N:.6e}")
            print(f"Формула α: {self.alpha_formula_names[best_formula_idx]}")
            print(f"Счет: {best_score:.6f}")

            self.verify_solution(K_fixed, best_p, best_N, best_formula_idx)

            return K_fixed, best_p, best_N, best_formula_idx

        return None, None, None, None

# ==================== ЗАПУСК ====================

if __name__ == "__main__":
    optimizer = NetworkConstantsOptimizer()
    # Вариант 1: Анализ окрестности исходных параметров
    print("ВАРИАНТ 1: АНАЛИЗ ОКРЕСТНОСТИ ИСХОДНЫХ ПАРАМЕТРОВ")
    best_K1, best_p1, best_N1, best_idx1 = optimizer.analyze_around_original()
    # Вариант 2: Целевой анализ с K=8
    print("\nВАРИАНТ 2: ЦЕЛЕВОЙ АНАЛИЗ С K=8")

    best_K2, best_p2, best_N2, best_idx2 = optimizer.run_focused_analysis()
    # Вариант 3: Полная оптимизаци
    best_K3, best_p3, best_N3, best_idx3 = optimizer.find_optimal_parameters()
    # Сравнение результатов
    print("СРАВНЕНИЕ РЕЗУЛЬТАТОВ")


    results = []
    if best_K1 is not None:
        equations1 = optimizer.calculate_all_equations(best_K1, best_p1, best_N1, best_idx1)
        errors1 = optimizer.calculate_errors(equations1)
        avg1 = np.mean([errors1[f'{key}_error'] for key in ['pi', 'e', 'G', 'gamma', 'alpha']])
        results.append(("Окрестность исходных", best_K1, best_p1, best_N1, avg1))

    if best_K2 is not None:
        equations2 = optimizer.calculate_all_equations(best_K2, best_p2, best_N2, best_idx2)
        errors2 = optimizer.calculate_errors(equations2)
        avg2 = np.mean([errors2[f'{key}_error'] for key in ['pi', 'e', 'G', 'gamma', 'alpha']])
        results.append(("K=8 фиксирован", best_K2, best_p2, best_N2, avg2))

    if best_K3 is not None:
        equations3 = optimizer.calculate_all_equations(best_K3, best_p3, best_N3, best_idx3)
        errors3 = optimizer.calculate_errors(equations3)
        avg3 = np.mean([errors3[f'{key}_error'] for key in ['pi', 'e', 'G', 'gamma', 'alpha']])
        results.append(("Полная оптимизация", best_K3, best_p3, best_N3, avg3))

    print(f"\n{'Метод':<25} {'K':<10} {'p':<12} {'N':<20} {'Ср. ошибка':<10}")
    for name, K, p, N, avg_err in results:
        print(f"{name:<25} {K:<10.6f} {p:<12.8f} {N:<20.3e} {avg_err:<10.6f}%")

    # Выбор лучшего
    if results:
        best_result = min(results, key=lambda x: x[4])
        print(f"\n✅ Лучшее решение: {best_result[0]}")
        print(f"   K = {best_result[1]:.10f}")
        print(f"   p = {best_result[2]:.10f}")
        print(f"   N = {best_result[3]:.6e}")
        print(f"   Средняя ошибка: {best_result[4]:.8f}%")