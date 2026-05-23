"""
ОБЪЕДИНЁННЫЙ ПОИСК МАТЕМАТИЧЕСКИХ ЗАКОНОМЕРНОСТЕЙ
МЕЖДУ ПАРАМЕТРАМИ ГРАФА МАЛОГО МИРА И ФУНДАМЕНТАЛЬНЫМИ КОНСТАНТАМИ

Включает:
- ConstantHunter (поиск нетривиальных формул)
Поиск связей с:
- e, π, φ, γ (Эйлера-Маскерони)
- Константы Фейгенбаума, Глайшера, Апери, Каталана, Хинчина
- Γ(1/3), Γ(1/4), ζ(3)
- Широкий спектр математических констант (из ConstantHunter)
"""

import math
import json
import re
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import time
from math import log, exp, sqrt, pi, e
from scipy.special import gamma, zeta
from collections import Counter
import warnings

warnings.filterwarnings('ignore')

def lnN_zeta():
    # ВАЖНО: интерпретируем как ln N (НЕ exp)
    exponent = 1.5 + math.pi ** 2 / 6.0
    lnN_zeta = (6.0 ** exponent)
    return lnN_zeta

N_test = math.exp(lnN_zeta())#4.197668e+121
K0 = 6
p0 = 1 / (K0 * N_test ** (1 / 3))
print(f"p0 = {p0}")


def generate_constant_variations(constants_dict: Dict[str, float]) -> Dict[str, float]:
    """
    Автоматически генерирует вариации констант:
    - 1/C (обратные значения)
    - C1 + C2, C1 - C2, C1 * C2, C1 / C2
    """
    variations = {}
    variations.update(constants_dict)

    # Обратные значения
    for name, value in constants_dict.items():
        if abs(value) > 1e-10:
            inv_name = f"1/{name}"
            if inv_name not in variations:
                variations[inv_name] = 1.0 / value

    # Комбинации C1 + C2, C1 - C2, C1 * C2, C1 / C2
    pure_constants = {k: v for k, v in constants_dict.items()
                      if not any(op in k for op in ['/', '+', '-', '*'])}

    pure_names = list(pure_constants.keys())
    pure_values = list(pure_constants.values())

    max_combinations = 800000
    combinations_generated = 0

    for i in range(len(pure_names)):
        for j in range(i + 1, len(pure_names)):
            if combinations_generated >= max_combinations:
                break

            name1, val1 = pure_names[i], pure_values[i]
            name2, val2 = pure_names[j], pure_values[j]

            # Сложение
            sum_val = val1 + val2
            sum_name = f"({name1} + {name2})"
            if sum_name not in variations:
                variations[sum_name] = sum_val
                combinations_generated += 1

            #Умножение
            prod_val = val1 * val2
            prod_name = f"({name1} * {name2})"
            if prod_name not in variations:
                variations[prod_name] = prod_val
                combinations_generated += 1

            # Разность (обе версии)
            if val1 != val2:
                diff1_val = val1 - val2
                diff1_name = f"({name1} - {name2})"
                if abs(diff1_val) > 1e-100 and diff1_name not in variations:
                    variations[diff1_name] = diff1_val
                    combinations_generated += 1

                diff2_val = val2 - val1
                diff2_name = f"({name2} - {name1})"
                if abs(diff2_val) > 1e-100 and diff2_name not in variations:
                    variations[diff2_name] = diff2_val
                    combinations_generated += 1

            #Деление (обе версии)
            if abs(val2) > 1e-100:
                div1_val = val1 / val2
                div1_name = f"({name1} / {name2})"
                if div1_name not in variations:
                    variations[div1_name] = div1_val
                    combinations_generated += 1

            if abs(val1) > 1e-100:
                div2_val = val2 / val1
                div2_name = f"({name2} / {name1})"
                if div2_name not in variations:
                    variations[div2_name] = div2_val
                    combinations_generated += 1

    return variations


# СТРУКТУРА ДЛЯ РЕЗУЛЬТАТОВ СОВПАДЕНИЙ
@dataclass
class MatchResult:
    formula: str
    calculated_value: float
    constant_name: str
    constant_value: float
    absolute_error: float
    relative_percent: float

    def __str__(self):
        return f"{self.formula} ≈ {self.constant_name} ({self.relative_percent:.8f}%)"


# ЧАСТЬ 1: БАЗА КОНСТАНТ
class ConstantLibrary:
    """Библиотека фундаментальных математических и физических констант"""

    def __init__(self, include_variations=True, include_special_variations=False):
        self.math_constants = {
            # 'LAMBDA': 0.05183093,
            #'phi': (1 + math.sqrt(5)) / 2,
            #'sqrt2': math.sqrt(2),
            #'sqrt3': math.sqrt(3),
            #'gamma_euler': 0.5772156649015329,
            #'ln2': math.log(2),
            #'ln3': math.log(3),
            #'lnpi': math.log(math.pi),
        }

        self.special_constants = {
            #'geometric_zeta': 0.06228841681507902,
            #'geometric_alpha': 0.03660105378020262,
            # 'geometric_phys': 0.06729304209414977,
            # 'zeta_alpha': 0.025687363034876398,
            # 'zeta_phys': 0.005004625279070751,
            # 'alpha_phys': 0.03069198831394715,
            # 'zeta_hbar': 0.0057246855806738495,
            # 'zeta_lightspeed':0.05087153198951455,
            'zeta_plank_mass': 2.5405355802828732e-11,
            #'dzetta_2': 1.6449340668482264,
            #'1/sqrt_pi': 1/(3.141592653589793**(1/2)),
            #'p^2': 3.141592653589793**2,
            #'pi-3': 3.141592653589793 - 3,
            #'e-1': 2.718281828459045 - 1,
            #'euler': 2.718281828459045,
            #'euler_mascheroni': 0.5772156649015329,
            #'pi': 3.141592653589793,
            #'golden_ratio': 1.618033988749895,
             # 'gamma_quarter': 3.6256099082219083,
             # 'gamma_third': 2.6789385347077476,
             # 'gamma_three_quarters': 1.225416702465177,
            # 'cahen': 0.6419448389191956,
            # 'erdos_tenenbaum_ford': 0.0860713320559343,
            #'regular_paperfolding': 0.8507361882018673,
            # 'van_der_pauw': 4.5323601418271938,
            #'embree_trefethen': 0.70258,
            # 'kepler_bouwkamp': 0.1149420448532962,
            # 'alladi_grinstead': 0.8093940205406391,
            # 'brjuno': 0.0654603245889095,
            #'feigenbaum_delta': 4.669201609102990,
            #'feigenbaum_delta_inv': 1/4.669201609102990,
            #'feigenbaum_alpha': 2.5029078750958928,
            #'feigenbaum_alpha_inv': 1/2.5029078750958928,
            #'glaisher': 1.2824271291006226,
            #'apery': 1.2020569031595942,
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
            #'viswanath': 1.1319882487943,
            # 'landau_ramanujan': 0.764223653589221,
            # 'gauss': 0.834626841674073,
        }

        self.gamma_values = {
             #'gamma_1_2': gamma(0.5),
             #'gamma_1_6': gamma(1/6),
             #'gamma_2_3': gamma(2/3),
        }

        self.pi_e_constants = {
            #'pi/4': pi / 4,
            #'pi/K': pi / 6,
            # 'e^pi': exp(pi),
            # 'pi^e': pi ** e,
            # 'e^e': exp(e),
            # 'gelfond': exp(pi),
            # 'gelfond_schneider': 2 ** sqrt(2),
            # 'hilbert': 2 ** sqrt(2),
        }

        self.physics_constants = {
            #'alpha': 7.2973525693e-3,
        }

        self.base_constants = {}
        self.base_constants.update(self.math_constants)
        self.base_constants.update(self.special_constants)
        self.base_constants.update(self.gamma_values)
        self.base_constants.update(self.pi_e_constants)
        self.base_constants.update(self.physics_constants)

        self.all_constants = dict(self.base_constants)

        if include_variations:
            print("🔄 Генерация вариаций констант (C1±C2, C1*C2, C1/C2, 1/C)...")
            variations = generate_constant_variations(self.base_constants)
            self.all_constants.update(variations)
            print(f"   Добавлено {len(variations)} вариаций")

        print(f"📚 Всего констант для проверки: {len(self.all_constants)}")


class GraphQuantities:
    """Вычисление всех величин из параметров графа K, N"""

    def __init__(self, K=6.0, N=N_test):
        self.K = K
        self.N = N
        self.update()

    def update(self, K=None, N=None):
        if K is not None:
            self.K = K

        if N is not None:
            self.N = N

        self.lnN = log(self.N)
        self.lnK = log(self.K)
        self.all_quantities = {
            'K': self.K,
            'N': self.N,
            'lnK': self.lnK,
            'lnN': self.lnN,
        }

        return self.all_quantities


# ЧАСТЬ 3: CONSTANT HUNTER (ПОИСК НЕТРИВИАЛЬНЫХ ФОРМУЛ)
class ConstantHunter:
    """Поиск нетривиальных совпадений с константами через перебор выражений"""

    def __init__(self, K=6, N=N_test):
        self.K = K
        self.N = N

        self._log_cache = {}
        self._sqrt_cache = {}

        self.base_expressions = self._generate_base_expressions()
        self.constants = ConstantLibrary().all_constants
        self.constants_list = list(self.constants.items())

    def _cached_log(self, x):
        if x not in self._log_cache:
            self._log_cache[x] = math.log(x) if x > 0 else float('nan')
        return self._log_cache[x]

    def _cached_sqrt(self, x):
        if x not in self._sqrt_cache:
            self._sqrt_cache[x] = math.sqrt(x) if x >= 0 else float('nan')
        return self._sqrt_cache[x]

    def _generate_base_expressions(self) -> Dict[str, float]:
        exprs = {}

        K = self.K
        N = self.N
        ln_N = self._cached_log(N)
        ln_K = self._cached_log(K)

        exprs['K'] = float(K)
        #exprs['N'] = float(N)
        exprs['ln(N)'] = ln_N
        exprs['ln(K)'] = ln_K
        exprs['sqrt(K)'] = math.sqrt(K)
        exprs['1/sqrt(K)'] = 1/math.sqrt(K)
        exprs['2'] = 2
        exprs['1/2'] = 1/2
        exprs['1/3'] = 1/3
        exprs['3'] = 3
        exprs['1/sqrt(2)'] = 1/math.sqrt(2)
        exprs['sqrt(2)'] = math.sqrt(2)
        exprs['1/sqrt(3)'] = 1/math.sqrt(3)
        exprs['sqrt(3)'] = math.sqrt(3)

        exprs['(pi^2)/6'] = pi**2/6
        exprs['6/(pi^2)'] = 6/pi**2
        exprs['ln(N)^2'] = self._cached_log(N)**2
        exprs['1/ln(N)^2'] = 1/self._cached_log(N) ** 2
        exprs['1/ln(N)^3'] = 1 / self._cached_log(N) ** 3
        exprs['1/ln(N)^4'] = 1 / self._cached_log(N) ** 4
        exprs['1/ln(N)^5'] = 1 / self._cached_log(N) ** 5
        exprs['1/ln(N)^6'] = 1 / self._cached_log(N) ** 6
        exprs['1/ln(N)^2'] = 1 / self._cached_log(N) ** 2
        exprs['lnN/K'] = self._cached_log(N) / K
        exprs['K/lnN'] =  K / self._cached_log(N)
        #exprs['pi^6'] = pi**6
        #exprs['1/pi^6'] = 1/pi ** 6
        exprs['pi^2'] = pi ** 2
        exprs['1/pi^2'] = 1/pi ** 2
        exprs['K^2'] = K**2
        exprs['K^3'] = K**3
        #exprs['ln(ln(ln(N)))'] = self._cached_log(self._cached_log(self._cached_log(N)))
        #exprs['1/ln(ln(ln(N)))'] = 1/(self._cached_log(self._cached_log(self._cached_log(N))))
        exprs['ln(ln(N))'] = self._cached_log(self._cached_log(N))
        exprs['1/ln(ln(N))'] = 1/(self._cached_log(self._cached_log(N)))
        exprs['ln(ln(K))'] = self._cached_log(self._cached_log(K))
        exprs['1/ln(ln(K))'] = 1 / (self._cached_log(self._cached_log(K)))
        exprs['ln(N)-K'] = self._cached_log(N) - K
        exprs['ln(N)+K'] = self._cached_log(N) + K
        exprs['ln(K)^2'] = self._cached_log(K) * self._cached_log(K)
        exprs['1/ln(N)'] = 1 / (self._cached_log(N))
        exprs['1/ln(N)^2'] = 1 / (self._cached_log(N)**2)
        exprs['ln(N)^2'] = (self._cached_log(N)**2)
        exprs['1/ln(K)'] = 1 / (self._cached_log(K))
        exprs['1/ln(K)^2'] = 1 / (self._cached_log(K)**2)

        valid_exprs = {}
        for k, v in exprs.items():
            if not (math.isnan(v) or math.isinf(v)):
                valid_exprs[k] = float(v)

        return valid_exprs

    def _is_trivial_value(self, value: float) -> bool:
        """Фильтрует тривиальные значения"""
        # Базовые тривиальные значения
        return False
        if abs(value - 1.0) < 1e-10:
            return True
        if abs(value) < 1e-10:
            return True
        if abs(value - math.e) < 1e-10:
            return True
        if abs(value - math.log(6)) < 1e-10:  # ln(K) = ln(6)
            return True
        if abs(value - math.pi) < 1e-10:
            return True
        if abs(value - math.sqrt(self.K) / 2) < 1e-10:
            return True
        if abs(value - 4 / math.sqrt(self.K)) < 1e-10:
            return True
        return False

    def _is_tautology(self, formula: str, value: float, const_name: str) -> bool:
        """
        Определяет, является ли совпадение тривиальной тавтологией
        """
        return False
        # 1. Проверка на циклические выражения (X/X, X-X, X*1/X и т.д.)
        tautology_patterns = [
            (r'\(([^)]+)\)\s*\*\s*\(1/\1\)', 'X * 1/X'),
            (r'\(1/([^)]+)\)\s*\*\s*\1', '1/X * X'),
            (r'\(([^)]+)\)\s*/\s*\(\1\)', 'X / X'),
            (r'\(([^)]+)\)\s*-\s*\(\1\)', 'X - X'),
            (r'\(([^)]+)\s*\+\s*([^)]+)\)\s*-\s*\2', '(X+Y)-Y = X'),
            (r'\(([^)]+)\s*-\s*([^)]+)\)\s*\+\s*\2', '(X-Y)+Y = X'),
            (r'\(([^)]+)\s*\*\s*([^)]+)\)\s*/\s*\2', '(X*Y)/Y = X'),
            (r'\(([^)]+)\s*/\s*([^)]+)\)\s*\*\s*\2', '(X/Y)*Y = X'),
        ]

        for pattern, desc in tautology_patterns:
            if re.search(pattern, formula):
                return True

        # 2. Проверка на свойство логарифма: ln(a*b) = ln(a) + ln(b)
        if 'ln2' in const_name and 'ln3' in const_name:
            if abs(value - (math.log(2) + math.log(3))) < 1e-10:
                if any(x in formula for x in ['ln(K)', 'ln(6)', 'ln(K*p)']):
                    return True

        # 4. Проверка на ln(K) = f2 тавтологии
        if abs(value - math.log(6)) < 1e-10:
            if 'ln(K)' in formula or 'ln(6)' in formula:
                return True


        # 7. Масштабирование: (K/X)*X = K
        scale_pattern = r'\(K\s*/\s*([^)]+)\)\s*\*\s*\1'
        if re.search(scale_pattern, formula):
            return True

        # 8. Обратное масштабирование: (K*X)/X = K
        scale_pattern2 = r'\(K\s*\*\s*([^)]+)\)\s*/\s*\1'
        if re.search(scale_pattern2, formula):
            return True

        return False

    def _fast_check_constant(self, value: float, const_value: float,
                             min_percent: float, max_percent: float) -> Optional[float]:
        if const_value == 0:
            return None
        rel_error = abs((value - const_value) / const_value) * 100
        if min_percent <= rel_error <= max_percent:
            return rel_error
        return None

    def search_matches_fast(self, max_operations: int = 3,
                            min_percent: float = 1e-7,
                            max_percent: float = 0.5,
                            max_expressions: int = 20000) -> List[MatchResult]:
        matches = []
        tautologies_filtered = 0

        current_expressions = list(self.base_expressions.items())
        all_expressions = dict(self.base_expressions)

        print(f"\n🔍 CONSTANT HUNTER: Поиск от {min_percent:.2e}% до {max_percent:.2f}%")
        print(f"Базовых выражений: {len(current_expressions)}")
        print(f"Констант для проверки: {len(self.constants_list)}")

        total_start = time.time()

        for op_count in range(1, max_operations + 1):
            level_start = time.time()
            new_expressions = []

            print(f"\nУровень {op_count}: {len(current_expressions)} выражений")

            n = len(current_expressions)
            max_pairs = min(8000000, n * (n - 1) // 2)

            checked_pairs = 0
            formulas_generated = 0

            for i in range(min(n, 4000)):
                for j in range(i, min(n, 4000)):
                    if checked_pairs >= max_pairs:
                        break

                    expr1 = current_expressions[i]
                    expr2 = current_expressions[j]
                    checked_pairs += 1

                    ops = []

                    val_sum = expr1[1] + expr2[1]
                    if not self._is_trivial_value(val_sum):
                        ops.append((f"({expr1[0]} + {expr2[0]})", val_sum))

                    val_sub1 = expr1[1] - expr2[1]
                    if abs(val_sub1) > 1e-300 and not self._is_trivial_value(val_sub1):
                        ops.append((f"({expr1[0]} - {expr2[0]})", val_sub1))

                    val_sub2 = expr2[1] - expr1[1]
                    if abs(val_sub2) > 1e-300 and not self._is_trivial_value(val_sub2):
                        ops.append((f"({expr2[0]} - {expr1[0]})", val_sub2))

                    val_mul = expr1[1] * expr2[1]
                    if abs(val_mul) < 1e300 and not self._is_trivial_value(val_mul):
                        ops.append((f"({expr1[0]} * {expr2[0]})", val_mul))

                    if abs(expr2[1]) > 1e-300:
                        val_div1 = expr1[1] / expr2[1]
                        if not self._is_trivial_value(val_div1):
                            ops.append((f"({expr1[0]} / {expr2[0]})", val_div1))

                    if abs(expr1[1]) > 1e-300:
                        val_div2 = expr2[1] / expr1[1]
                        if not self._is_trivial_value(val_div2):
                            ops.append((f"({expr2[0]} / {expr1[0]})", val_div2))

                    for formula, value in ops:
                        formulas_generated += 1

                        for const_name, const_value in self.constants_list:
                            rel_error = self._fast_check_constant(
                                value, const_value, min_percent, max_percent
                            )

                            if rel_error is not None:
                                # Фильтрация тавтологий
                                if self._is_tautology(formula, value, const_name):
                                    tautologies_filtered += 1
                                    continue

                                matches.append(MatchResult(
                                    formula=formula,
                                    calculated_value=value,
                                    constant_name=const_name,
                                    constant_value=const_value,
                                    absolute_error=abs(value - const_value),
                                    relative_percent=rel_error
                                ))

                        if len(new_expressions) < 1000 and formula not in all_expressions:
                            all_expressions[formula] = value
                            new_expressions.append((formula, value))

                if checked_pairs >= max_pairs:
                    break

            if new_expressions:
                current_expressions.extend(new_expressions)
                if len(current_expressions) > max_expressions:
                    current_expressions.sort(key=lambda x: abs(x[1]))
                    current_expressions = current_expressions[:max_expressions]

            level_time = time.time() - level_start
            print(f"  Проверено пар: {checked_pairs:,}, формул: {formulas_generated:,}, "
                  f"совпадений: {len(matches)}, отфильтровано тавтологий: {tautologies_filtered}, "
                  f"время: {level_time:.1f}с")

        total_time = time.time() - total_start

        matches.sort(key=lambda x: x.relative_percent)

        unique_matches = []
        seen = set()
        for match in matches:
            key = (match.formula, match.constant_name)
            if key not in seen:
                seen.add(key)
                unique_matches.append(match)

        print(f"\n✅ Итого: {total_time:.1f}с, уникальных совпадений: {len(unique_matches)}, "
              f"отфильтровано тавтологий: {tautologies_filtered}")

        return unique_matches[:2000]

    def save_results(self, matches: List[MatchResult], filename: str = "hunter_matches.json"):
        results_data = []
        for match in matches:
            results_data.append({
                'formula': match.formula,
                'calculated_value': float(match.calculated_value),
                'constant_name': match.constant_name,
                'constant_value': float(match.constant_value),
                'absolute_error': float(match.absolute_error),
                'relative_percent': float(match.relative_percent)
            })

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)

        print(f"\n💾 Сохранено {len(results_data)} результатов в {filename}")


# ЧАСТЬ 4: ГЛАВНАЯ ФУНКЦИЯ
def main():
    print("ОБЪЕДИНЁННЫЙ АНАЛИЗ МАТЕМАТИЧЕСКИХ ЗАКОНОМЕРНОСТЕЙ")
    print("ConstantHunter + Спектральные величины + RG-параметры")

    K0 = 6
    N0 = N_test
    #p0 = 1/(K0*N_test**(1/3))

    print(f"\n📊 БАЗОВЫЕ ПАРАМЕТРЫ:")
    print(f"  K = {K0}")
    print(f"  N = {N0:.6e}")

    # ConstantHunter
    print("\n" + "=" * 70)
    print("CONSTANT HUNTER: ПОИСК НЕТРИВИАЛЬНЫХ СОВПАДЕНИЙ")
    print("=" * 70)

    hunter = ConstantHunter(K=K0, N=N0)

    constant_lib = ConstantLibrary(
        include_variations=True,
        include_special_variations=False
    )
    hunter.constants = constant_lib.all_constants
    hunter.constants_list = list(hunter.constants.items())

    matches = hunter.search_matches_fast(
        max_operations=4,
        min_percent=1e-300,
        max_percent=0.05,
        max_expressions=8000
    )

    if matches:
        hunter.save_results(matches, "unified_hunter_matches.json")

        print("\n📊 ТОП-1000 ЛУЧШИХ СОВПАДЕНИЙ:")
        print("=" * 70)
        for i, match in enumerate(matches[:1000], 1):
            print(f"{i:2d}. {match}")

        # Статистика по типам констант
        stats = {}
        for match in matches:
            stats[match.constant_name] = stats.get(match.constant_name, 0) + 1

        print("\n📊 СТАТИСТИКА ПО КОНСТАНТАМ:")
        for const_name, count in sorted(stats.items(), key=lambda x: x[1], reverse=True)[:15]:
            print(f"  {const_name:25}: {count:4d} совпадений")

        # Категоризация
        physics = [m for m in matches if m.constant_name in ['alpha', 'LAMBDA']]
        math_consts = [m for m in matches if m.constant_name in ['pi', 'euler', 'phi', 'golden_ratio']]
        special = [m for m in matches if m.constant_name in ['feigenbaum_delta', 'feigenbaum_alpha',
                                                              'glaisher', 'apery', 'catalan', 'khinchin']]

        print(f"\n📊 КАТЕГОРИИ СОВПАДЕНИЙ:")
        print(f"  Физические константы: {len(physics)}")
        print(f"  Математические константы: {len(math_consts)}")
        print(f"  Специальные константы: {len(special)}")

if __name__ == "__main__":
    main()