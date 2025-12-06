import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import odeint
import warnings

warnings.filterwarnings('ignore')


class FractalNetworkAttractor:
    """Модель информационно-фрактального аттрактора с динамикой"""

    def __init__(self):
        self.constants = {
            'e': np.e,
            'pi': np.pi,
            'phi': (1 + np.sqrt(5)) / 2,  # золотое сечение
            'alpha': 7.2973525693e-3,  # постоянная тонкой структуры
            'c': 1.0,  # скорость света (нормированная)
            'hbar': 1.0  # постоянная Планка (нормированная)
        }

    def fractal_depth(self, N, K, p):
        """Фрактальная глубина U = ln(N) / |ln(Kp)|"""
        if K * p == 1:
            return np.inf
        return np.log(N) / abs(np.log(K * p))

    def balance_equation(self, N, K, p):
        """Уравнение баланса: e - p*sqrt(K*U) = 0"""
        U = self.fractal_depth(N, K, p)
        if np.isinf(U):
            return np.inf
        return self.constants['e'] - p * np.sqrt(K * U)

    def find_attractor_N(self, K, p):
        """Находит аттракторное N для данных K, p"""
        if K * p >= 1:
            return np.nan

        def f(logN):
            return self.balance_equation(np.exp(logN), K, p)

        try:
            # Поиск в широком диапазоне
            logN_guesses = [10, 50, 100, 200, 300]
            for guess in logN_guesses:
                logN_sol = fsolve(f, guess, maxfev=1000, xtol=1e-12)
                if f(logN_sol[0]) < 1e-10:
                    return np.exp(logN_sol[0])
        except:
            pass

        return np.nan

    def evolution_dynamics(self, state, t, K, p_target, alpha=0.1):
        """
        Динамическая модель эволюции системы к аттрактору
        state = [logN, p, lambda1]
        """
        logN, p, lambda1 = state

        # Текущий N
        N = np.exp(logN)

        # 1. Динамика p (стремление к целевому значению)
        dp_dt = alpha * (p_target - p)

        # 2. Динамика N (реакция на изменение p)
        if K * p < 1:
            # Вычисляем "желаемое" N для текущего p
            N_target = self.find_attractor_N(K, p)
            if not np.isnan(N_target):
                dlogN_dt = 0.01 * (np.log(N_target) - logN)
            else:
                dlogN_dt = 0
        else:
            dlogN_dt = -0.01  # Сжатие при Kp >= 1

        # 3. Динамика λ₁ (релаксация к равновесию)
        # λ₁ стремится к значению, зависящему от U
        U = self.fractal_depth(N, K, p)
        if not np.isinf(U):
            # Базовая модель λ₁ = λ∞ + A/U
            lambda_target = 0.013 + 2.5 / U
            dlambda_dt = 0.1 * (lambda_target - lambda1)
        else:
            dlambda_dt = 0

        return [dlogN_dt, dp_dt, dlambda_dt]

    def universal_relations(self, K, p, N):
        """Вычисление универсальных соотношений с фундаментальными константами"""
        if np.isnan(N) or N <= 0:
            return {}

        U = self.fractal_depth(N, K, p)
        if np.isinf(U):
            return {}

        relations = {}

        # 1. Отношения с e
        relations['U/e'] = U / self.constants['e']
        relations['lnN/e'] = np.log(N) / self.constants['e']
        relations['sqrt_KU/e'] = np.sqrt(K * U) / self.constants['e']

        # 2. Отношения с π
        relations['U/pi'] = U / self.constants['pi']
        relations['lnN/pi'] = np.log(N) / self.constants['pi']
        relations['sqrt_KU/pi'] = np.sqrt(K * U) / self.constants['pi']

        # 3. Отношения с φ (золотое сечение)
        relations['U/phi'] = U / self.constants['phi']
        relations['lnN/phi'] = np.log(N) / self.constants['phi']
        relations['sqrt_KU/phi'] = np.sqrt(K * U) / self.constants['phi']

        # 4. Комбинированные соотношения
        relations['(U*pi)/e'] = (U * self.constants['pi']) / self.constants['e']
        relations['(lnN*phi)/pi'] = (np.log(N) * self.constants['phi']) / self.constants['pi']

        # 5. Экспоненциальные соотношения
        relations['exp(-U/pi)'] = np.exp(-U / self.constants['pi'])
        relations['exp(-lnN/e)'] = np.exp(-np.log(N) / self.constants['e'])

        # 6. "Квантовые" соотношения (с постоянной тонкой структуры)
        relations['alpha*U'] = self.constants['alpha'] * U
        relations['alpha*lnN'] = self.constants['alpha'] * np.log(N)

        # 7. Информационная энтропия
        S = np.log(N)  # информационная энтропия Больцмана
        relations['S/e'] = S / self.constants['e']
        relations['S/pi'] = S / self.constants['pi']
        relations['S/(e*pi)'] = S / (self.constants['e'] * self.constants['pi'])

        return relations

    def analyze_configuration(self, K, p, p_initial=None):
        """Полный анализ конфигурации с динамикой"""

        if p_initial is None:
            p_initial = p * 1.1  # начальное значение немного выше

        # 1. Находим аттрактор
        N_attr = self.find_attractor_N(K, p)

        results = {
            'K': K,
            'p_target': p,
            'p_initial': p_initial,
            'Kp': K * p,
            'has_attractor': not np.isnan(N_attr)
        }

        if results['has_attractor']:
            # 2. Параметры аттрактора
            U_attr = self.fractal_depth(N_attr, K, p)
            lambda1_attr = 0.013 + 2.5 / U_attr

            results.update({
                'N_attr': N_attr,
                'log10_N_attr': np.log10(N_attr),
                'ln_N_attr': np.log(N_attr),
                'U_attr': U_attr,
                'lambda1_attr': lambda1_attr,
                'mixing_time_attr': 1 / lambda1_attr,
                'balance_value': p * np.sqrt(K * U_attr),
                'balance_error': abs(p * np.sqrt(K * U_attr) - self.constants['e']) / self.constants['e'] * 100
            })

            # 3. Универсальные соотношения
            relations = self.universal_relations(K, p, N_attr)
            results['relations'] = relations

            # 4. Моделируем динамику достижения аттрактора
            t = np.linspace(0, 100, 1000)  # время от 0 до 100
            initial_state = [
                np.log(
                    self.find_attractor_N(K, p_initial) if not np.isnan(self.find_attractor_N(K, p_initial)) else 1e50),
                p_initial,
                0.135  # начальное λ₁ (из ваших данных)
            ]

            states = odeint(self.evolution_dynamics, initial_state, t, args=(K, p, 0.1))

            # 5. Анализ динамики
            logN_final = states[-1, 0]
            p_final = states[-1, 1]
            lambda_final = states[-1, 2]

            results['dynamics'] = {
                'time_to_convergence': t[np.argmax(states[:, 1] < p + 0.001)] if np.any(
                    states[:, 1] < p + 0.001) else 100,
                'convergence_speed': -np.gradient(states[:, 1], t).mean(),
                'N_final': np.exp(logN_final),
                'p_final': p_final,
                'lambda_final': lambda_final,
                'states': states,
                'time': t
            }

            # 6. Анализ устойчивости
            # Линеаризация вокруг аттрактора
            J = np.zeros((3, 3))  # матрица Якоби

            # Производные в окрестности аттрактора
            eps = 1e-6
            for i in range(3):
                state_plus = initial_state.copy()
                state_minus = initial_state.copy()
                state_plus[i] += eps
                state_minus[i] -= eps

                f_plus = self.evolution_dynamics(state_plus, 0, K, p, 0.1)
                f_minus = self.evolution_dynamics(state_minus, 0, K, p, 0.1)

                J[:, i] = (np.array(f_plus) - np.array(f_minus)) / (2 * eps)

            eigenvalues = np.linalg.eigvals(J)
            results['stability'] = {
                'jacobian': J,
                'eigenvalues': eigenvalues,
                'is_stable': np.all(np.real(eigenvalues) < 0),
                'oscillatory': np.any(np.imag(eigenvalues) != 0)
            }

        return results

    def analyze_multiple_configurations(self, configs):
        """Анализ нескольких конфигураций"""
        all_results = []

        for i, (K, p) in enumerate(configs):
            print(f"Анализ конфигурации {i + 1}: K={K}, p={p}")

            result = self.analyze_configuration(K, p)
            all_results.append(result)

            if result['has_attractor']:
                print(f"  Аттрактор найден: N={result['N_attr']:.2e}")
                print(f"  log10(N)={result['log10_N_attr']:.2f}, U={result['U_attr']:.2f}")
                print(f"  λ₁={result['lambda1_attr']:.6f}, τ={result['mixing_time_attr']:.1f}")
                print(f"  Ошибка баланса: {result['balance_error']:.10f}%")

                # Важные соотношения
                rel = result['relations']
                print(f"  Ключевые соотношения:")
                print(f"    U/π={rel['U/pi']:.6f}, ln(N)/e={rel['lnN/e']:.6f}")
                print(f"    √(KU)/π={rel['sqrt_KU/pi']:.6f}, U/φ={rel['U/phi']:.6f}")

                # Динамика
                dyn = result['dynamics']
                print(f"  Динамика: время сходимости={dyn['time_to_convergence']:.1f}")
                print(f"    Конечное p={dyn['p_final']:.6f}, λ₁={dyn['lambda_final']:.6f}")

                # Устойчивость
                stab = result['stability']
                print(f"  Устойчивость: {'стабильно' if stab['is_stable'] else 'нестабильно'}")
                print(f"    Собственные значения: {stab['eigenvalues']}")
            else:
                print(f"  Аттрактор не найден (Kp={K * p:.3f} >= 1)")

            print()

        return all_results

    def find_optimal_configurations(self, K_range=(4, 24, 2), p_range=(0.01, 0.1, 0.01)):
        """Поиск оптимальных конфигураций по различным критериям"""

        K_values = np.arange(K_range[0], K_range[1] + K_range[2], K_range[2])
        p_values = np.arange(p_range[0], p_range[1] + p_range[2], p_range[2])

        optima = {
            'max_N': {'value': 0, 'K': None, 'p': None, 'config': None},
            'min_error': {'value': np.inf, 'K': None, 'p': None, 'config': None},
            'min_lambda1': {'value': np.inf, 'K': None, 'p': None, 'config': None},
            'max_U': {'value': 0, 'K': None, 'p': None, 'config': None},
            'best_universal': {'score': 0, 'K': None, 'p': None, 'config': None}
        }

        for K in K_values:
            for p in p_values:
                if K * p < 1:
                    N = self.find_attractor_N(K, p)
                    if not np.isnan(N):
                        U = self.fractal_depth(N, K, p)
                        error = abs(p * np.sqrt(K * U) - self.constants['e']) / self.constants['e']
                        lambda1 = 0.013 + 2.5 / U

                        # Универсальный скор (близость к круглым числам в соотношениях)
                        relations = self.universal_relations(K, p, N)
                        universal_score = (
                                1 / (abs(relations['U/pi'] - round(relations['U/pi'])) + 1e-10) +
                                1 / (abs(relations['lnN/e'] - round(relations['lnN/e'])) + 1e-10) +
                                1 / (abs(relations['sqrt_KU/pi'] - round(relations['sqrt_KU/pi'])) + 1e-10)
                        )

                        config = {'K': K, 'p': p, 'N': N, 'U': U, 'error': error, 'lambda1': lambda1}

                        # Обновление оптимумов
                        if N > optima['max_N']['value']:
                            optima['max_N'] = {'value': N, 'K': K, 'p': p, 'config': config}

                        if error < optima['min_error']['value']:
                            optima['min_error'] = {'value': error, 'K': K, 'p': p, 'config': config}

                        if lambda1 < optima['min_lambda1']['value']:
                            optima['min_lambda1'] = {'value': lambda1, 'K': K, 'p': p, 'config': config}

                        if U > optima['max_U']['value']:
                            optima['max_U'] = {'value': U, 'K': K, 'p': p, 'config': config}

                        if universal_score > optima['best_universal']['score']:
                            optima['best_universal'] = {'score': universal_score, 'K': K, 'p': p, 'config': config}

        return optima


def main():
    """Основной анализ"""

    model = FractalNetworkAttractor()

    # 1. Ваши ключевые конфигурации
    key_configs = [
        (8, 0.0527),  # Исходная
        (8, 0.05),  # Аттрактор
        (4, 0.07),  # Интересный случай с U/π ≈ 120
        (6, 0.05),  # Для сравнения
        (12, 0.03),  # Максимальный N
        (20, 0.03),  # Средний K
        (24, 0.03)  # Большой K
    ]

    print("АНАЛИЗ КЛЮЧЕВЫХ КОНФИГУРАЦИЙ")
    print("=" * 60)

    results = model.analyze_multiple_configurations(key_configs)

    # 2. Поиск оптимальных конфигураций
    print("\nПОИСК ОПТИМАЛЬНЫХ КОНФИГУРАЦИЙ")
    print("=" * 60)

    optima = model.find_optimal_configurations()

    for criterion, data in optima.items():
        if data['config']:
            print(f"\n{criterion.replace('_', ' ').title()}:")
            print(f"  K={data['K']}, p={data['p']:.4f}")
            print(f"  N={data['config']['N']:.2e}, U={data['config']['U']:.2f}")
            print(f"  Ошибка={data['config']['error'] * 100:.6f}%, λ₁={data['config']['lambda1']:.6f}")

    # 3. Анализ универсальных соотношений для лучших конфигураций
    print("\nУНИВЕРСАЛЬНЫЕ СООТНОШЕНИЯ ДЛЯ ОПТИМАЛЬНЫХ КОНФИГУРАЦИЙ")
    print("=" * 60)

    best_configs = [
        (optima['min_error']['K'], optima['min_error']['p']),
        (optima['best_universal']['K'], optima['best_universal']['p']),
        (8, 0.05)  # Ваш аттрактор
    ]

    for K, p in best_configs:
        if K is not None:
            result = model.analyze_configuration(K, p)
            if result['has_attractor']:
                print(f"\nK={K}, p={p}:")
                rel = result['relations']

                print("  Отношения с фундаментальными константами:")
                print(f"    U/π = {rel['U/pi']:.6f} (близко к {round(rel['U/pi'])}?)")
                print(f"    ln(N)/e = {rel['lnN/e']:.6f} (близко к {round(rel['lnN/e'])}?)")
                print(f"    √(KU)/π = {rel['sqrt_KU/pi']:.6f} (близко к {round(rel['sqrt_KU/pi'])}?)")
                print(f"    U/φ = {rel['U/phi']:.6f}")
                print(f"    U/e = {rel['U/e']:.6f}")
                print(f"    α·U = {rel['alpha*U']:.6e} (постоянная тонкой структуры)")

                # Проверка на "магические" числа
                interesting = []
                for name, value in rel.items():
                    if abs(value - round(value)) < 0.01:
                        interesting.append((name, value, round(value)))

                if interesting:
                    print("  Интересные приближения к целым числам:")
                    for name, value, nearest in interesting:
                        print(f"    {name} = {value:.6f} ≈ {nearest}")

    # 4. Фазовый анализ
    print("\nФАЗОВЫЙ АНАЛИЗ СИСТЕМЫ")
    print("=" * 60)

    # Исследуем границу Kp = 1
    print("Исследование границы Kp = 1:")
    for K in [4, 8, 12, 16, 20, 24]:
        p_critical = 1 / K
        print(f"\nK={K}:")
        print(f"  Критическое p = {p_critical:.6f}")

        # Немного ниже критического
        p_below = p_critical * 0.999
        result_below = model.analyze_configuration(K, p_below)

        # Немного выше критического
        p_above = p_critical * 1.001
        result_above = model.analyze_configuration(K, p_above)

        if result_below['has_attractor']:
            print(f"  p={p_below:.6f} (< критического): аттрактор существует")
            print(f"    N={result_below['N_attr']:.2e}, U={result_below['U_attr']:.2f}")
        else:
            print(f"  p={p_below:.6f}: аттрактор не найден")

        print(f"  p={p_above:.6f} (> критического): аттрактор не существует")

    # 5. Динамические режимы
    print("\nДИНАМИЧЕСКИЕ РЕЖИМЫ ЭВОЛЮЦИИ")
    print("=" * 60)

    # Тест для вашей конфигурации
    K, p = 8, 0.05
    result = model.analyze_configuration(K, p)

    if result['has_attractor']:
        dyn = result['dynamics']
        stab = result['stability']

        print(f"\nДинамика для K={K}, p={p}:")
        print(f"  Время сходимости: {dyn['time_to_convergence']:.1f}")
        print(f"  Скорость сходимости: {dyn['convergence_speed']:.6f}")
        print(f"  Устойчивость: {'стабильно' if stab['is_stable'] else 'нестабильно'}")
        print(f"  Осцилляции: {'есть' if stab['oscillatory'] else 'нет'}")
        print(f"  Собственные значения: {stab['eigenvalues']}")

        # Анализ режимов релаксации
        real_parts = np.real(stab['eigenvalues'])
        imag_parts = np.imag(stab['eigenvalues'])

        print("  Режимы релаксации:")
        for i, (re, im) in enumerate(zip(real_parts, imag_parts)):
            if im == 0:
                print(f"    Мода {i + 1}: экспоненциальная, время релаксации = {1 / abs(re):.2f}")
            else:
                print(f"    Мода {i + 1}: осциллирующая, частота = {abs(im):.4f}, затухание = {abs(re):.4f}")


if __name__ == "__main__":
    main()