import json
import re
import time
import warnings
from itertools import combinations, product
from typing import Dict
import math

warnings.filterwarnings('ignore')


class UltraPrecisionFinder:
    def __init__(self, max_components=5, min_quantum=1e-300, max_quantum=0.0001, constant_variable=False):
        self.max_components = max_components
        self.min_quantum = min_quantum
        self.max_quantum = max_quantum
        self.constant_variable = constant_variable

        self.K = 6
        self.N = 4.197668e+121

        self.lnN = math.log(self.N)
        self.lnK = math.log(self.K)
        self.lnlnN = math.log(self.lnN)
        self.lnlnK = math.log(self.lnK)

        self.base_values = self._create_base_values()
        self.all_constants = self._create_all_constants()
        self.results = []
        self.checked_combinations = 0

    def _calculate_relative_error_percent(self, calculated_value, target_value):
        if abs(target_value) < 1e-15:
            return 0.0 if abs(calculated_value) < 1e-15 else float('inf')
        if abs(calculated_value) > 1e100 or abs(target_value) > 1e100:
            return 0.0 if calculated_value == target_value else float('inf')
        return abs((calculated_value - target_value) / target_value) * 100.0

    def _is_error_in_range(self, rel_error_percent):
        if rel_error_percent == float('inf'):
            return False
        return self.min_quantum <= rel_error_percent <= self.max_quantum

    def _create_base_values(self) -> Dict[str, float]:
        K = self.K
        lnN = self.lnN
        lnK = self.lnK
        lnlnN = self.lnlnN
        lnlnK = self.lnlnK

        return {
            'K': K, 'lnN': lnN, 'lnK': lnK, 'lnlnN': lnlnN, 'lnlnK': lnlnK,
            '1/lnN': 1 / lnN, '1/lnK': 1 / lnK, '1/lnlnN': 1 / lnlnN, '1/lnlnK': 1 / lnlnK,
            'lnN/K': lnN / K, 'K/lnN': K / lnN, 'lnK/lnN': lnK / lnN, 'lnN/lnK': lnN / lnK,
            'lnK**2': lnK ** 2, 'K**2': K ** 2, '1/K**2': 1 / (K ** 2), 'K**3': K ** 3,
            'lnN**2': lnN ** 2, 'lnN**3': lnN ** 3, '1/lnN**3': 1 / (lnN ** 3),
            '1/2': 1 / 2, '1/3': 1 / 3, '2/3': 2 / 3, '2': 2, '3': 3,
        }

    def _create_all_constants(self) -> Dict[str, float]:
        return {
            # 'pi': math.pi,
            # 'e': math.e,
            # 'phi': (1 + math.sqrt(5)) / 2,
            # 'sqrt2': math.sqrt(2),
            # 'sqrt3': math.sqrt(3),
            # 'gamma_euler': 0.5772156649015329,
            # 'ln2': math.log(2),
            # 'ln3': math.log(3),
            # 'lnpi': math.log(math.pi),
            # 'gamma_1_3': gamma(1 / 3),
            # 'gamma_1_4': gamma(0.25),
            # 'LAMBDA': 0.05183093,
            # 'p^2': 3.141592653589793 ** 2,
            # 'euler': 2.718281828459045,
            # 'euler_mascheroni': 0.5772156649015329,
            # 'golden_ratio': 1.618033988749895,
            # 'gamma_quarter': 3.6256099082219083,
            # 'gamma_third': 2.6789385347077476,
            # 'gamma_three_quarters': 1.225416702465177,
            # 'cahen': 0.6419448389191956,
            # 'erdos_tenenbaum_ford': 0.0860713320559343,
            # 'regular_paperfolding': 0.8507361882018673,
            # 'van_der_pauw': 4.5323601418271938,
            # 'embree_trefethen': 0.70258,
            # 'kepler_bouwkamp': 0.1149420448532962,
            # 'alladi_grinstead': 0.8093940205406391,
            # 'brjuno': 0.0654603245889095,
            'feigenbaum_delta': 4.669201609102990,
            # 'feigenbaum_alpha': 2.5029078750958928,
            # 'glaisher': 1.2824271291006226,
            # 'apery': 1.2020569031595942,
            # 'catalan': 0.9159655941772190,
            # 'khinchin': 2.6854520010653062,
            # 'mills': 1.306377883863080,
            # 'porter': 1.467078079433975,
            # 'brun_twin': 1.902160583104,
            # 'twin_prime': 0.660161815846869,
            # 'meissel_mertens': 0.261497212847642,
            # 'artin': 0.373955813619202,
            # 'ramanujan_soldner': 1.451369234883381,
            # 'lemniscate': 2.622057554292119,
            # 'magic_angle': 0.955316618124509,
            # 'parabolic': 2.295587149392638,
            # 'plastic': 1.324717957244746,
            # 'supergolden': 1.465571231876768,
            # 'gompertz': 0.596347362323194,
            # 'levy': 3.275822918721811,
            # 'viswanath': 1.1319882487943,
            # 'landau_ramanujan': 0.764223653589221,
            # 'gauss': 0.834626841674073,
            # 'gamma_1_2': gamma(0.5),
            # 'gamma_1_6': gamma(1 / 6),
            # 'gamma_2_3': gamma(2 / 3),
            # 'pi/4': math.pi / 4,
            # 'pi/K': math.pi / 6,
            # 'e^pi': math.exp(math.pi),
            # 'pi^e': math.pi ** math.e,
            # 'gelfond': math.exp(math.pi),
            # 'gelfond_schneider': 2 ** math.sqrt(2),
            # 'hilbert': 2 ** math.sqrt(2),
            # 'alpha': 7.2973525693e-3,
        }

    def try_exact_combinations(self, value, const_name, const_value, tolerance=1e-12):
        """
        ИСПРАВЛЕННАЯ ВЕРСИЯ: используем eval вместо _evaluate
        """
        base_names = list(self.base_values.keys())
        n = len(base_names)
        results = []

        for complexity in range(2, self.max_components + 1):
            for indices in combinations(range(n), complexity):
                selected_names = [base_names[i] for i in indices]

                for ops in product(['+', '-', '*', '/'], repeat=complexity - 1):
                    # Строим формулу
                    formula = self._build_formula(selected_names, ops)

                    # Вычисляем через eval (надежно!)
                    result = self._evaluate_formula_string(formula)
                    if result is None:
                        continue

                    if const_value != 0:
                        rel_error_percent = self._calculate_relative_error_percent(result, const_value)
                        if self._is_error_in_range(rel_error_percent):
                            results.append({
                                'formula': formula,
                                'value': result,
                                'constant': const_name,
                                'const_value': const_value,
                                'rel_error': rel_error_percent
                            })

        return results

    def _evaluate_formula_string(self, formula_str):
        """Вычисление строки формулы с правильной подстановкой значений"""
        try:
            expr = formula_str

            # Заменяем специальные имена с **
            expr = expr.replace('K**2', str(self.base_values['K**2']))
            expr = expr.replace('K**3', str(self.base_values['K**3']))
            expr = expr.replace('lnN**2', str(self.base_values['lnN**2']))
            expr = expr.replace('lnN**3', str(self.base_values['lnN**3']))
            expr = expr.replace('1/K**2', str(self.base_values['1/K**2']))
            expr = expr.replace('1/lnN**3', str(self.base_values['1/lnN**3']))
            expr = expr.replace('lnK**2', str(self.base_values['lnK**2']))

            # Заменяем остальные переменные
            replacements = {
                'lnlnN': str(self.lnlnN), 'lnlnK': str(self.lnlnK),
                'lnN/K': str(self.lnN / self.K), 'K/lnN': str(self.K / self.lnN),
                'lnK/lnN': str(self.lnK / self.lnN), 'lnN/lnK': str(self.lnN / self.lnK),
                '1/lnlnN': str(1 / self.lnlnN), '1/lnlnK': str(1 / self.lnlnK),
                '1/lnN': str(1 / self.lnN), '1/lnK': str(1 / self.lnK),
                'lnN': str(self.lnN), 'lnK': str(self.lnK),
                '1/3': str(1 / 3), '2/3': str(2 / 3), '1/2': str(1 / 2),
                'K': str(self.K), '2': '2', '3': '3'
            }

            for name, val in replacements.items():
                expr = expr.replace(name, val)

            # Вычисляем
            result = eval(expr, {"__builtins__": {}}, {})

            if math.isnan(result) or math.isinf(result):
                return None
            return result
        except:
            return None

    def _evaluate_formula(self, formula_str, values_dict):
        """Упрощенная версия для обратной совместимости"""
        return self._evaluate_formula_string(formula_str)

    def _build_formula(self, names, ops):
        if not ops:
            return str(names[0])
        if len(names) == 1:
            return str(names[0])

        terms = []
        current_term = str(names[0])

        for i, op in enumerate(ops):
            next_name = str(names[i + 1])
            if op in ('*', '/'):
                current_term = f"{current_term}{op}{next_name}"
            else:
                terms.append(current_term)
                terms.append(op)
                current_term = next_name

        terms.append(current_term)
        return "".join(terms)

    def optimize_to_constant(self, const_name, const_value):
        """
        ИСПРАВЛЕННАЯ ВЕРСИЯ: только проверенные методы
        """
        results = []
        base_items = list(self.base_values.items())
        n = len(base_items)

        # 1. Обратные значения
        for name, val in base_items:
            if abs(val) > 1e-15:
                inv_val = 1.0 / val
                if const_value != 0:
                    rel_error_percent = self._calculate_relative_error_percent(inv_val, const_value)
                    if self._is_error_in_range(rel_error_percent):
                        results.append({
                            'formula': f"1/{name}",
                            'value': inv_val,
                            'constant': const_name,
                            'const_value': const_value,
                            'rel_error': rel_error_percent
                        })

        # 2. Комбинации из 2-5 элементов через eval
        for complexity in range(2, min(self.max_components + 1, 6)):
            for indices in combinations(range(n), complexity):
                selected_names = [base_items[i][0] for i in indices]

                for ops in product(['+', '-', '*', '/'], repeat=complexity - 1):
                    formula = self._build_formula(selected_names, ops)
                    result = self._evaluate_formula_string(formula)

                    if result is None:
                        continue

                    if const_value != 0:
                        rel_error_percent = self._calculate_relative_error_percent(result, const_value)
                        if self._is_error_in_range(rel_error_percent):
                            results.append({
                                'formula': formula,
                                'value': result,
                                'constant': const_name,
                                'const_value': const_value,
                                'rel_error': rel_error_percent
                            })

        return results

    def _format_coeff(self, coeff):
        if coeff == 1:
            return ""
        elif coeff == -1:
            return "-"
        elif coeff == int(coeff):
            return str(int(coeff))
        else:
            return f"{coeff:.4g}"

    def _format_linear_combo(self, a, name1, b, name2):
        parts = []
        if a == 1:
            parts.append(name1)
        elif a == -1:
            parts.append(f"-{name1}")
        elif a != 0:
            parts.append(f"{a}*{name1}")

        if b > 0 and parts:
            parts.append(f"+ {b}*{name2}")
        elif b == 1:
            parts.append(f"+ {name2}")
        elif b == -1:
            parts.append(f"- {name2}")
        elif b != 0:
            parts.append(f"{b}*{name2}")

        return " ".join(parts)

    def search_ultra_precision(self):
        print(f"\n{'=' * 80}")
        print(f"🎯 ПОИСК СВЕРХТОЧНЫХ СОВПАДЕНИЙ")
        print(f"{'=' * 80}")

        total_start = time.time()
        all_results = []

        for const_name, const_value in self.all_constants.items():
            print(f"  Проверка {const_name} = {const_value:.10f}...")

            direct_results = self.try_exact_combinations(None, const_name, const_value)
            if direct_results:
                print(f"    ✅ Найдено {len(direct_results)} прямых совпадений!")
                all_results.extend(direct_results)

            algebra_results = self.optimize_to_constant(const_name, const_value)
            if algebra_results:
                print(f"    ✅ Найдено {len(algebra_results)} алгебраических приближений!")
                all_results.extend(algebra_results)

        if all_results:
            all_results.sort(key=lambda x: x['rel_error'])
            unique = []
            seen = set()
            for r in all_results:
                key = (r['formula'], r['constant'])
                if key not in seen:
                    seen.add(key)
                    unique.append(r)
            all_results = unique[:500]

        total_time = time.time() - total_start
        print(f"\n✅ ПОИСК ЗАВЕРШЕН за {total_time:.1f}с")
        print(f"Всего найдено: {len(all_results)} сверхточных совпадений")

        return all_results

    def print_results(self, results):
        if not results:
            print("\n❌ Сверхточных совпадений не найдено.")
            return

        print(f"\n{'=' * 100}")
        print(f"ЛУЧШИЕ СВЕРХТОЧНЫЕ СОВПАДЕНИЯ (показано {min(30, len(results))})")
        print(f"{'=' * 100}")
        print(f"{'Формула':<50} {'Константа':<20} {'Значение':<15} {'Отн. ошибка %':<15}")
        print("─" * 100)

        for i, r in enumerate(results[:30]):
            formula = r['formula'][:48]
            const = r['constant'][:18]
            value = f"{r['value']:.10f}"
            error = f"{r['rel_error']:.8f}"
            print(f"{formula:<50} {const:<20} {value:<15} {error:<15}")

        if len(results) > 30:
            print(f"... и еще {len(results) - 30} результатов")

    def run(self):
        results = self.search_ultra_precision()
        self.print_results(results)
        if results:
            self.save_results(results)

    def save_results(self, results, filename="ultra_precision_matches.json"):
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results[:5000], f, indent=2, ensure_ascii=False)
        print(f"\n💾 Сохранено в {filename}")


if __name__ == "__main__":
    finder = UltraPrecisionFinder(
        max_components=5,
        min_quantum=1e-300,
        max_quantum=0.001,
        constant_variable=False
    )
    finder.run()