"""
АНАЛИЗ FIXED POINTS ДЛЯ ПРОВЕРКИ ФИЗИЧНОСТИ МОДЕЛИ
====================================================
Проверяем, сходятся ли оптимальные параметры к одним и тем же
значениям при малых вариациях p и N.
"""

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution

import math


# ============================================================================
# КОНСТАНТЫ ДЛЯ ПОИСКА
# ============================================================================
class ConstantLibrary:
    def __init__(self):
        self.math_constants = {
            'pi': math.pi, 'e': math.e, 'phi': (1 + math.sqrt(5)) / 2,
            'sqrt2': math.sqrt(2), 'sqrt3': math.sqrt(3), 'sqrt5': math.sqrt(5),
            'gamma_euler': 0.5772156649015329,
            'catalan': 0.915965594177219, 'apery': 1.2020569031595942,
            'khinchin': 2.6854520010653062, 'glaisher': 1.2824271291006226,
            'feigenbaum': 4.669201609102990, 'feigenbaum2': 2.5029078750958928,

            'golden_ratio': (1 + math.sqrt(5)) / 2,

            # Корни
            'sqrt7': math.sqrt(7),

            # Логарифмы
            'ln2': math.log(2),
            'ln3': math.log(3),
            'ln10': math.log(10),
            'lnpi': math.log(math.pi),

            # === ПОПУЛЯРНЫЕ МАТЕМАТИЧЕСКИЕ КОНСТАНТЫ ===
            'euler_mascheroni': 0.577215664901532,  # γ
            'mills': 1.306377883863080,  # θ
            'porter': 1.467078079433975,  # C

            # Константы из теории чисел
            'brun_twin': 1.902160583104,  # B₂
            'twin_prime': 0.660161815846869,  # C₂
            'meissel_mertens': 0.261497212847642,  # M
            'artin': 0.373955813619202,  # C_Artin
            'ramanujan_soldner': 1.451369234883381,  # μ

            # Геометрические константы
            'lemniscate': 2.622057554292119,  # ϖ
            'magic_angle': 0.955316618124509,  # θ_m
            'parabolic': 2.295587149392638,  # P

            # Алгебраические константы
            'plastic': 1.324717957244746,  # ρ
            'supergolden': 1.465571231876768,  # ψ
            'conway': 1.303577269034296,  # λ

            # === ДРУГИЕ ИНТЕРЕСНЫЕ КОНСТАНТЫ ===
            'gompertz': 0.596347362323194,  # δ
            'levy': 3.275822918721811,  # γ
            'erdos_borwein': 1.606695152415291,  # E
            'viswanath': 1.1319882487943,  # K
            'sierpinski': 2.584981759579253,  # K
            'landau_ramanujan': 0.764223653589221,  # K
            'backhouse': 1.456074948582689,  # B
            'gauss': 0.834626841674073,  # G
            'niven': 0.705971,  # C
            'omega': 0.567143290,  # Ω
            'laplace_limit': 0.662743419,  # ε
            'mrb': 0.187859,  # C

            # === КОНСТАНТЫ СВЯЗАННЫЕ С π и e ===
            'pi/2': math.pi / 2,
            'pi/3': math.pi / 3,
            'pi/4': math.pi / 4,
            'pi/6': math.pi / 6,
            '2pi': 2 * math.pi,
            'e^pi': math.exp(math.pi),
            'pi^e': math.pi ** math.e,
            'e^e': math.exp(math.e),

            # === КОНСТАНТЫ ИЗ АНАЛИЗА ===
            'gamma(1/2)': math.sqrt(math.pi),  # Γ(1/2)
            'gamma(1/3)': 2.678938534707748,  # Γ(1/3)
            'gamma(1/4)': 3.625609908221908,  # Γ(1/4)

            # === СПЕЦИАЛЬНЫЕ КОНСТАНТЫ ===
            'ramanujan_constant': 262537412640768743.99999999999925,  # e^{π√163}
            'gelfond': 23.140692632779269,  # e^π
            'gelfond_schneider': 2.665144142690225,  # 2^√2
            'hilbert': 2.665144142690225,  # 2^√2
            # 'geometric_zeta': 0.06228841681507902,
            # 'geometric_alpha': 0.03660105378020262,
            # 'geometric_phys': 0.06729304209414977,
            # 'zeta_alpha': -0.025687363034876398,
            # 'zeta_phys': 0.005004625279070751,
            # 'alpha_phys': 0.03069198831394715,
        }
        self.all_constants = self.math_constants


# ГЕНЕРАЦИЯ ФОРМУЛ И ПОИСК ОПТИМАЛЬНЫХ ПАРАМЕТРОВ
class FixedPointAnalyzer:
    def __init__(self, base_K=8, base_p=0.0527, base_N=9.5e+122):
        self.base_K = base_K
        self.base_p = base_p
        self.base_N = base_N
        self.constants = ConstantLibrary().all_constants

    def compute_all_expressions(self, K, p, N):
        """Вычисление всех базовых выражений для данных параметров"""
        lnN = math.log(N)
        lnK = math.log(K)
        lnp = math.log(p) if p > 0 else float('nan')
        lnKp = math.log(K * p) if K * p > 0 else float('nan')

        exprs = {
            'K': K, 'p': p, 'N': N, 'K*p': K * p,
            'ln(N)': lnN, 'ln(K)': lnK,
            'ln(p)': lnp, 'ln(K*p)': lnKp,
            '1+p': 1 + p, '1-p': 1 - p,
            'sqrt(K*p)': math.sqrt(K * p) if K * p >= 0 else float('nan'),
            'sqrt(K)': math.sqrt(K), 'sqrt(p)': math.sqrt(p) if p >= 0 else float('nan'),
        }

        if lnKp != 0 and not math.isnan(lnKp):
            exprs['ln(N)/ln(K*p)'] = lnN / abs(lnKp)
            exprs['ln(K*p)/ln(N)'] = lnKp / lnN

        if p != 0:
            exprs['K/p'] = K / p
            exprs['p/K'] = p / K

        if p != 1:
            exprs['1/(1-p)'] = 1 / (1 - p)

        return exprs

    def generate_formulas(self, expressions, max_depth=3):
        """Генерация формул через перебор операций"""
        formulas = []
        expr_list = list(expressions.items())

        # Уровень 1: базовые выражения
        for name, val in expr_list:
            if not math.isnan(val) and not math.isinf(val):
                formulas.append((name, val))

        # Уровень 2: бинарные операции
        for i, (n1, v1) in enumerate(expr_list):
            for j, (n2, v2) in enumerate(expr_list):
                if i <= j:
                    if not (math.isnan(v1) or math.isnan(v2) or math.isinf(v1) or math.isinf(v2)):
                        formulas.append((f"({n1} + {n2})", v1 + v2))
                        formulas.append((f"({n1} * {n2})", v1 * v2))
                        if v2 != 0:
                            formulas.append((f"({n1} / {n2})", v1 / v2))
                        if v1 - v2 != 0:
                            formulas.append((f"({n1} - {n2})", v1 - v2))

        return formulas[:500]  # ограничиваем количество

    def find_best_matches(self, formulas, target_constants, tolerance=0.01):
        """Поиск формул, близких к целевым константам"""
        matches = []
        for formula, value in formulas:
            if math.isnan(value) or math.isinf(value) or value == 0:
                continue
            for const_name, const_val in target_constants.items():
                if const_val == 0:
                    continue
                error = abs(value - const_val) / const_val
                if error < tolerance:
                    matches.append({
                        'formula': formula,
                        'value': value,
                        'constant': const_name,
                        'target': const_val,
                        'error': error
                    })
        matches.sort(key=lambda x: x['error'])
        return matches[:30]  # топ-30 самых точных

    def extract_equation(self, formula_str, target_val):
        """
        Извлечение уравнения из формулы.
        Например: "ln(N)/ln(K*p)" = target_val -> ln(N) = target_val * ln(K*p)
        """
        # Упрощённый парсинг — ищем p и N в формуле
        equation_type = None
        if 'ln(N)' in formula_str and 'ln(K*p)' in formula_str:
            if '/ln(K*p)' in formula_str:
                # formula = ln(N) / ln(K*p) ≈ target
                # => ln(N) = target * |ln(K*p)|
                return ('U_equation', lambda p, N, target: abs(math.log(N) - target * abs(math.log(8 * p))))
            elif '*ln(K*p)' in formula_str:
                return ('U_equation', lambda p, N, target: abs(math.log(N) / abs(math.log(8 * p)) - target))

        if 'ln(N)' in formula_str and 'ln(p)' in formula_str:
            return ('ln_ratio', lambda p, N, target: abs(math.log(N) / abs(math.log(p)) - target))

        if 'sqrt(K*p)' in formula_str:
            return ('sqrt_Kp', lambda p, N, target: abs(math.sqrt(8 * p) - target))

        if 'K/p' in formula_str:
            return ('K_p_ratio', lambda p, N, target: abs(8 / p - target))

        # По умолчанию — минимизация разности
        return ('generic', lambda p, N, target: 0)

    def find_optimal_parameters(self, matches, p_range=(0.01, 0.6), N_range=(1e122, 1e124)):
        """Поиск оптимальных p и N для каждой формулы"""
        results = []

        for match in matches[:10]:  # анализируем топ-10
            formula = match['formula']
            target = match['target']
            eq_type, eq_func = self.extract_equation(formula, target)

            # Функция ошибки
            def error_func(params):
                p, N = params
                if p <= 0 or N <= 0:
                    return 1e10
                try:
                    return eq_func(p, N, target)
                except:
                    return 1e10

            # Поиск минимума
            result = differential_evolution(
                error_func,
                bounds=[p_range, N_range],
                maxiter=100,
                popsize=10,
                disp=False
            )

            p_opt, N_opt = result.x

            results.append({
                'formula': formula,
                'constant': match['constant'],
                'target': target,
                'eq_type': eq_type,
                'p_opt': p_opt,
                'N_opt': N_opt,
                'error': result.fun,
                'p_rel': p_opt / self.base_p,
                'N_rel': N_opt / self.base_N
            })

        return results

    def analyze_stability(self, p_variations=None, N_variations=None):
        """Анализ стабильности fixed points при вариациях параметров"""
        if p_variations is None:
            p_variations = [0.9, 0.95, 1.0, 1.05, 1.1]
        if N_variations is None:
            N_variations = [0.9, 0.95, 1.0, 1.05, 1.1]

        all_results = []

        for p_factor in p_variations:
            for N_factor in N_variations:
                p_test = self.base_p * p_factor
                N_test = self.base_N * N_factor

                print(f"Тест: p = {p_test:.3e} (x{p_factor:.2f}), N = {N_test:.3e} (x{N_factor:.2f})")
                # Вычисляем выражения
                exprs = self.compute_all_expressions(self.base_K, p_test, N_test)

                # Генерируем формулы
                formulas = self.generate_formulas(exprs)

                # Находим лучшие совпадения
                matches = self.find_best_matches(formulas, self.constants, tolerance=0.05)

                if matches:
                    print(f"Найдено {len(matches)} совпадений")
                    print("\nТоп-5 формул:")
                    for i, m in enumerate(matches[:5]):
                        print(f"  {i + 1}. {m['formula']} ≈ {m['constant']} (ошибка {m['error'] * 100:.4f}%)")

                    # Ищем оптимальные параметры
                    opt_results = self.find_optimal_parameters(matches)

                    for res in opt_results:
                        res['p_factor_input'] = p_factor
                        res['N_factor_input'] = N_factor
                        all_results.append(res)
                else:
                    print("Совпадений не найдено")

        return all_results

    def plot_fixed_points(self, results):
        """Визуализация fixed points"""
        if not results:
            print("Нет данных для визуализации")
            return

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # 1. Распределение оптимальных p
        ax1 = axes[0, 0]
        p_opts = [r['p_opt'] for r in results]
        ax1.hist(p_opts, bins=20, edgecolor='black', alpha=0.7)
        ax1.axvline(x=self.base_p, color='red', linestyle='--', label=f'base p={self.base_p:.2e}')
        ax1.set_xlabel('p_opt')
        ax1.set_ylabel('Частота')
        ax1.set_title('Распределение оптимальных p')
        ax1.legend()
        ax1.set_xscale('log')
        ax1.grid(True, alpha=0.3)

        # 2. Распределение оптимальных N
        ax2 = axes[0, 1]
        N_opts = [r['N_opt'] for r in results]
        ax2.hist(N_opts, bins=20, edgecolor='black', alpha=0.7)
        ax2.axvline(x=self.base_N, color='red', linestyle='--', label=f'base N={self.base_N:.2e}')
        ax2.set_xlabel('N_opt')
        ax2.set_ylabel('Частота')
        ax2.set_title('Распределение оптимальных N')
        ax2.legend()
        ax2.set_xscale('log')
        ax2.grid(True, alpha=0.3)

        # 3. p_opt / p_input vs N_opt / N_input
        ax3 = axes[0, 2]
        p_ratios = [r['p_opt'] / (self.base_p * r['p_factor_input']) for r in results]
        N_ratios = [r['N_opt'] / (self.base_N * r['N_factor_input']) for r in results]
        ax3.scatter(N_ratios, p_ratios, alpha=0.6, s=30)
        ax3.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
        ax3.axvline(x=1.0, color='red', linestyle='--', alpha=0.5)
        ax3.set_xlabel('N_opt / N_input')
        ax3.set_ylabel('p_opt / p_input')
        ax3.set_title('Отношение оптимальных к входным параметрам')
        ax3.grid(True, alpha=0.3)

        # 4. Кластеризация по типам уравнений
        ax4 = axes[1, 0]
        eq_types = defaultdict(list)
        for r in results:
            eq_types[r['eq_type']].append(r)

        colors = plt.cm.Set3(np.linspace(0, 1, len(eq_types)))
        for (eq_type, res_list), color in zip(eq_types.items(), colors):
            p_vals = [r['p_opt'] for r in res_list]
            N_vals = [r['N_opt'] for r in res_list]
            ax4.scatter(N_vals, p_vals, label=eq_type, alpha=0.6, s=30, color=color)

        ax4.scatter([self.base_N], [self.base_p], color='red', s=200, marker='*',
                    label='Base', edgecolors='black', linewidth=2)
        ax4.set_xlabel('N_opt')
        ax4.set_ylabel('p_opt')
        ax4.set_title('Кластеризация по типам уравнений')
        ax4.set_xscale('log')
        ax4.set_yscale('log')
        ax4.legend(loc='best', fontsize=8)
        ax4.grid(True, alpha=0.3)

        # 5. Статистика по константам
        ax5 = axes[1, 1]
        const_counts = defaultdict(int)
        const_errors = defaultdict(list)
        for r in results:
            const_counts[r['constant']] += 1
            const_errors[r['constant']].append(r['error'])

        const_names = list(const_counts.keys())
        counts = [const_counts[c] for c in const_names]
        ax5.barh(const_names, counts)
        ax5.set_xlabel('Количество')
        ax5.set_title('Частота появления констант')
        ax5.grid(True, alpha=0.3)

        # 6. Сводка
        ax6 = axes[1, 2]
        ax6.axis('off')

        # Анализ стабильности
        p_std = np.std([r['p_rel'] for r in results])
        N_std = np.std([r['N_rel'] for r in results])

        summary = f"""
        АНАЛИЗ FIXED POINTS
        Базовые параметры:
          p = {self.base_p:.3e}
          N = {self.base_N:.3e}

        Проанализировано формул: {len(results)}
        Уникальных констант: {len(const_counts)}

        СТАБИЛЬНОСТЬ:
          σ(p_rel) = {p_std:.4f}
          σ(N_rel) = {N_std:.4f}

        ВЫВОД:
        """

        if p_std < 0.1 and N_std < 0.1:
            summary += "\n✅ ВЫСОКАЯ СТАБИЛЬНОСТЬ!"
            summary += "\n   Fixed points сходятся к одним значениям."
            summary += "\n   Это признак ФИЗИЧЕСКОЙ модели!"
        elif p_std < 0.3 and N_std < 0.3:
            summary += "\n⚠️ УМЕРЕННАЯ СТАБИЛЬНОСТЬ"
            summary += "\n   Есть разброс, но есть и кластеризация."
        else:
            summary += "\n❌ НИЗКАЯ СТАБИЛЬНОСТЬ"
            summary += "\n   Fixed points разбросаны хаотично."
            summary += "\n   Возможно, это ПОДГОНКА."

        ax6.text(0.1, 0.5, summary, fontsize=11, family='monospace',
                 verticalalignment='center', transform=ax6.transAxes)

        plt.tight_layout()
        plt.savefig('fixed_points_analysis.png', dpi=150)
        plt.show()

        return p_std, N_std


# ЗАПУСК АНАЛИЗА
def main():
    print("АНАЛИЗ FIXED POINTS ДЛЯ ПРОВЕРКИ ФИЗИЧНОСТИ МОДЕЛИ")

    # Базовые параметры (оптимальные из предыдущих тестов)
    base_K = 8
    base_p = 0.0527
    base_N = 9.702e+122

    analyzer = FixedPointAnalyzer(base_K, base_p, base_N)

    # Вариации параметров (±5%, ±10%)
    p_variations = [0.90, 0.95, 1.00, 1.05, 1.10]
    N_variations = [0.90, 0.95, 1.00, 1.05, 1.10]

    print(f"\nБазовые параметры:")
    print(f"  K = {base_K}")
    print(f"  p = {base_p:.6e}")
    print(f"  N = {base_N:.6e}")
    print(f"\nВариации p: {[f'{v * 100:.0f}%' for v in p_variations]}")
    print(f"Вариации N: {[f'{v * 100:.0f}%' for v in N_variations]}")
    print(f"Всего комбинаций: {len(p_variations) * len(N_variations)}")

    # Запуск анализа
    results = analyzer.analyze_stability(p_variations, N_variations)

    # Визуализация
    p_std, N_std = analyzer.plot_fixed_points(results)

    # Итоговый вывод
    print("ИТОГОВЫЙ ВЫВОД")

    if p_std < 0.1 and N_std < 0.1:
        print("\n🎉 МОДЕЛЬ ФИЗИЧЕСКАЯ!")
        print("   Fixed points устойчивы и сходятся к одним значениям.")
        print("   Это доказывает, что за формулами стоит реальная структура,")
        print("   а не просто подгонка под конкретные числа.")
    elif p_std < 0.3 and N_std < 0.3:
        print("\n⚠️ МОДЕЛЬ ЧАСТИЧНО ФИЗИЧЕСКАЯ")
        print("   Есть устойчивые кластеры, но также есть разброс.")
        print("   Некоторые формулы требуют уточнения.")
    else:
        print("\n❌ МОДЕЛЬ НЕСТАБИЛЬНА")
        print("   Fixed points хаотично разбросаны.")
        print("   Это признак подгонки, а не физической теории.")

    # Сохраняем результаты
    import json
    with open('fixed_points_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print("\n✅ Результаты сохранены в fixed_points_results.json")


if __name__ == "__main__":
    main()
