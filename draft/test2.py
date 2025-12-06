import numpy as np
import math
from scipy import constants


class ParameterSearcher:
    def __init__(self):
        self.classical_constants = {
            'hbar': constants.hbar,
            'c': constants.c,
            'G': constants.G,
            'kb': constants.k,
            'lp': constants.physical_constants['Planck length'][0],
            'tp': constants.physical_constants['Planck time'][0],
            'Tp': constants.physical_constants['Planck temperature'][0],
            'cosmo_lambda': 1.1056e-52,
            'ep0_em': 8.85e-12,
            'mu0_em': 1.256e-6,
            'e_plank': 1.87e-18
        }

    def lambda_param(self, K, p, N):
        return (np.log(K * p) / np.log(N)) ** 2

    def calculate_constants(self, K, p, N):
        """Ваши формулы в компактном виде"""
        λ = self.lambda_param(K, p, N)
        lnK = np.log(K)
        lnKp = np.log(K * p)
        lnN = np.log(N)

        # Ваши ключевые формулы
        hbar_em = (lnK ** 2) / (4 * λ ** 2 * K ** 2)

        R_universe = 2 * math.pi / (np.sqrt(K * p) * λ) * N ** (1 / 6)
        l_em = R_universe / np.sqrt(K * p)

        hbar_emergent = hbar_em * N ** (-1 / 3) / (6 * math.pi)

        c_emergent = (math.pi * l_em / hbar_em) / λ ** 2 * N ** (-1 / 6)

        G_emergent = (hbar_em ** 4 / l_em ** 2) * (1 / λ ** 2)

        # Постоянная Больцмана (ваша KB2 формула)
        KB2 = math.pi * lnN ** 7 / (3 * abs(lnKp) ** 6 * (p * K) ** (3 / 2) * N ** (1 / 3))

        lp_emergent = l_em * N ** (-1 / 2)

        # Космологическая постоянная
        cosmo_lambda = 3 * K * p / (math.pi ** 2 * N ** (1 / 3)) * (lnKp / lnN) ** 4

        # Диэлектрическая проницаемость
        ep0_em = (lnKp ** 4 * K) / (2 * math.pi * c_emergent ** 2 * hbar_emergent * N ** (1 / 3) * KB2)

        # Магнитная проницаемость
        mu0_em = (math.pi * lnK ** 2 * lnN ** 15) / (36 * K ** (9 / 2) * p ** (3 / 2) * abs(lnKp) ** 14 * N ** (1 / 3))

        # Заряд Планка
        e_plank = np.sqrt(3 * p ** (5 / 2) * K ** 1.5 * lnK ** 2 * lnKp ** 12 / (4 * math.pi ** 3 * lnN ** 13))

        # Температура Планка
        T_plank = (hbar_emergent * c_emergent ** 5 / (G_emergent * KB2 ** 2)) ** 0.5

        results = {
            'hbar': hbar_emergent,
            'c': c_emergent,
            'G': G_emergent,
            'kb': KB2,
            'lp': lp_emergent,
            'tp': hbar_emergent / T_plank,  # через температуру Планка
            'Tp': T_plank,
            'cosmo_lambda': cosmo_lambda,
            'ep0_em': ep0_em,
            'mu0_em': mu0_em,
            'e_plank': e_plank
        }

        return results

    def calculate_score(self, results):
        """Оценка качества совпадения (0-100%)"""
        score = 0
        errors = []

        for key in self.classical_constants:
            if key in results:
                ratio = results[key] / self.classical_constants[key]
                if ratio > 0:
                    # Логарифмическая ошибка (лучше для больших диапазонов)
                    error = abs(np.log10(ratio))
                    errors.append(error)

                    # Баллы: 0 если ошибка > 2 порядка, 100 если идеально
                    if error < 0.01:  # 1% ошибка
                        score += 100
                    elif error < 0.1:  # 25% ошибка
                        score += 80
                    elif error < 0.3:  # 50% ошибка
                        score += 50
                    elif error < 1.0:  # 90% ошибка
                        score += 20
                    elif error < 2.0:  # 99% ошибка
                        score += 5

        avg_error = np.mean(errors) if errors else 100
        normalized_score = score / len(self.classical_constants)

        return normalized_score, avg_error

    def search_parameters(self, K_range=(4, 16), p_range=(0.01, 0.1), N_range=(1e120, 1e126)):
        """Поиск лучших параметров"""
        best_score = 0
        best_params = None
        best_results = None

        # Тестовые точки (логарифмическая сетка)
        K_values = [4, 6, 8, 10, 12, 14, 16]
        p_values = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
        N_values = [10 ** i for i in range(120, 127, 2)]

        total_combinations = len(K_values) * len(p_values) * len(N_values)
        print(f"Всего комбинаций: {total_combinations}")

        count = 0
        good_solutions = []

        for K in K_values:
            for p in p_values:
                for N in N_values:
                    count += 1
                    if count % 100 == 0:
                        print(f"Проверено {count}/{total_combinations}")

                    try:
                        results = self.calculate_constants(K, p, N)
                        score, avg_error = self.calculate_score(results)

                        if score > 50:  # Удовлетворительное решение
                            good_solutions.append((score, avg_error, K, p, N, results))

                        if score > best_score:
                            best_score = score
                            best_params = (K, p, N)
                            best_results = results

                    except Exception as e:
                        continue

        return best_params, best_results, best_score, good_solutions


def analyze_solution(K, p, N, results, searcher):
    """Анализ конкретного решения"""
    print(f"\n=== Параметры: K={K}, p={p:.4f}, N={N:.2e} ===")

    print("\nКонстанта | Эмерджентная | Классическая | Отношение | Ошибка%")
    print("-" * 70)

    total_error = 0
    matched = 0

    for key in searcher.classical_constants:
        if key in results:
            em_val = results[key]
            cl_val = searcher.classical_constants[key]
            ratio = em_val / cl_val
            error_pct = abs(ratio - 1) * 100

            if error_pct < 10:
                matched += 1
                mark = "✓"
            else:
                mark = "✗"

            total_error += error_pct

            print(f"{key:12} | {em_val:.3e} | {cl_val:.3e} | {ratio:.4f} | {error_pct:.2f}% {mark}")

    avg_error = total_error / len(searcher.classical_constants)
    print(f"\nСовпало: {matched}/{len(searcher.classical_constants)}")
    print(f"Средняя ошибка: {avg_error:.2f}%")

    # Проверяем размерность
    λ = searcher.lambda_param(K, p, N)
    print(f"λ = {λ:.6e}")
    print(f"ln(K) = {np.log(K):.3f}, ln(Kp) = {np.log(K * p):.3f}, ln(N) = {np.log(N):.3f}")


def main_search():
    """Основной поиск"""
    print("=== ПОИСК АЛЬТЕРНАТИВНЫХ ПАРАМЕТРОВ ===")
    print("Ищем K, p, N, дающие похожие результаты...\n")

    searcher = ParameterSearcher()

    # 1. Ваши исходные параметры
    print("1. ВАШИ ПАРАМЕТРЫ (K=8.0, p=0.0527, N=0.95e123):")
    results_original = searcher.calculate_constants(8.0, 0.0527, 0.95e123)
    analyze_solution(8.0, 0.0527, 0.95e123, results_original, searcher)

    # 2. Поиск альтернатив
    print("\n\n2. ПОИСК ЛУЧШИХ АЛЬТЕРНАТИВНЫХ ПАРАМЕТРОВ...")
    best_params, best_results, best_score, good_solutions = searcher.search_parameters()

    if best_params:
        print(f"\nЛучшие найденные параметры: K={best_params[0]}, p={best_params[1]:.6f}, N={best_params[2]:.2e}")
        print(f"Оценка: {best_score:.1f}%")
        analyze_solution(best_params[0], best_params[1], best_params[2], best_results, searcher)

    # 3. Проверка небольших вариаций ваших параметров
    print("\n\n3. НЕБОЛЬШИЕ ВАРИАЦИИ ВАШИХ ПАРАМЕТРОВ:")

    variations = [
        (8.0, 0.0527, 0.95e123, "Оригинал"),
        (8.1, 0.0527, 0.95e123, "+ΔK"),
        (7.9, 0.0527, 0.95e123, "-ΔK"),
        (8.0, 0.0530, 0.95e123, "+Δp"),
        (8.0, 0.0524, 0.95e123, "-Δp"),
        (8.0, 0.0527, 1.00e123, "+ΔN"),
        (8.0, 0.0527, 0.90e123, "-ΔN"),
        (8.04, 0.0525, 1.00e123, "Ваш ранний вариант"),
    ]

    for K, p, N, desc in variations:
        try:
            results = searcher.calculate_constants(K, p, N)
            score, avg_error = searcher.calculate_score(results)
            print(f"\n{desc}: K={K}, p={p:.4f}, N={N:.2e}")
            print(f"  Оценка: {score:.1f}%, Средняя ошибка: {10 ** avg_error - 1:.1%}")
        except:
            print(f"\n{desc}: Ошибка вычисления")

    # 4. Проверка "очевидных" альтернатив
    print("\n\n4. 'ОЧЕВИДНЫЕ' АЛЬТЕРНАТИВЫ (округленные значения):")

    obvious_alternatives = [
        (8, 0.05, 1e123, "Округленные"),
        (10, 0.05, 1e123, "K=10"),
        (6, 0.05, 1e123, "K=6"),
        (8, 0.06, 1e123, "p=0.06"),
        (8, 0.04, 1e123, "p=0.04"),
        (4, 0.1, 1e123, "Меньше K, больше p"),
        (16, 0.025, 1e123, "Больше K, меньше p"),
    ]

    for K, p, N, desc in obvious_alternatives:
        try:
            results = searcher.calculate_constants(K, p, N)
            score, avg_error = searcher.calculate_score(results)

            # Считаем сколько констант совпало в пределах 10%
            matched = 0
            for key in searcher.classical_constants:
                if key in results:
                    ratio = results[key] / searcher.classical_constants[key]
                    if 0.9 < ratio < 1.1:
                        matched += 1

            print(f"{desc}: K={K}, p={p:.3f}, N={N:.2e}")
            print(f"  Совпало {matched}/11 констант, Оценка: {score:.1f}%")
        except Exception as e:
            print(f"{desc}: Ошибка - {e}")


def check_sensitivity():
    """Анализ чувствительности"""
    print("\n\n=== АНАЛИЗ ЧУВСТВИТЕЛЬНОСТИ ===")

    searcher = ParameterSearcher()
    base_K, base_p, base_N = 8.0, 0.0527, 0.95e123
    base_results = searcher.calculate_constants(base_K, base_p, base_N)

    # Анализ производных
    delta = 0.01  # 1% изменение

    for param_name, param_val, idx in [("K", base_K, 0), ("p", base_p, 1), ("N", base_N, 2)]:
        print(f"\nЧувствительность к {param_name}:")

        # +1%
        params_plus = [base_K, base_p, base_N]
        params_plus[idx] = param_val * (1 + delta)
        results_plus = searcher.calculate_constants(*params_plus)

        # -1%
        params_minus = [base_K, base_p, base_N]
        params_minus[idx] = param_val * (1 - delta)
        results_minus = searcher.calculate_constants(*params_minus)

        # Считаем изменение констант
        print("Константа | Изменение на 1% параметра")
        print("-" * 50)

        for key in ['c', 'hbar', 'G']:
            if key in results_plus and key in base_results:
                change_plus = (results_plus[key] - base_results[key]) / base_results[key] * 100
                change_minus = (results_minus[key] - base_results[key]) / base_results[key] * 100
                avg_change = (abs(change_plus) + abs(change_minus)) / 2
                print(f"{key:8} | {avg_change:.2f}%")


if __name__ == "__main__":
    main_search()
    check_sensitivity()