import math
import json
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple, Set
from functools import lru_cache


@dataclass
class MatchResult:
    formula: str
    calculated_value: float
    constant_name: str
    constant_value: float
    absolute_error: float
    relative_percent: float

    def __lt__(self, other):
        return self.relative_percent < other.relative_percent


class AdvancedConstantHunter:
    def __init__(self, K=8, p=0.052702, N=9.7e122):
        self.K = K
        self.p = p
        self.N = N

        # Ключевые предвычисления
        self._precompute()

        # Фундаментальные константы с расширенным набором
        self.constants = self._load_extended_constants()

        # Кэш для быстрого доступа к попарным операциям
        self._pair_cache = {}

    def _precompute(self):
        """Предвычисление всех базовых величин"""
        self.kp = self.K * self.p
        self.ln_N = math.log(self.N)
        self.ln_K = math.log(self.K)
        self.ln_p = math.log(self.p) if self.p > 0 else float('nan')
        self.ln_kp = math.log(self.kp) if self.kp > 0 else float('nan')
        self.ln_k_p = math.log(self.K+self.p) if self.K+self.p > 0 else float('nan')
        self.ln_U = math.log(self.N)/ math.log(self.K + self.p) #if self.kp > 0 else float('nan')

        # Основные величины
        self.base_values = {
            'K': self.K,
            'p': self.p,
            'N': self.N,
            'K*p': self.kp,
            'ln(N)': self.ln_N,
            'ln(K)': self.ln_K,
            'ln(p)': self.ln_p if self.p > 0 else None,
            'ln(K*p)': self.ln_kp if self.kp > 0 else None,
            'ln(K+p)': self.ln_k_p if self.ln_k_p > 0 else None,
            'ln(N)/ln((K+p)*p)': self.ln_U if self.ln_U > 0 else None,
            '1+p': 1 + self.p,
            '1-p': 1 - self.p,
            'K+p': self.K + self.p,
            'K-p': self.K - self.p,
            '(K+p)*p': (self.K + self.p) * self.p,
            'K/p': self.K / self.p if self.p != 0 else None,
            'p/K': self.p / self.K if self.K != 0 else None,
            'sqrt(K)': math.sqrt(self.K),
            'sqrt(p)': math.sqrt(self.p) if self.p >= 0 else None,
            'sqrt(K*p)': math.sqrt(self.kp) if self.kp >= 0 else None,
            '1/sqrt(K*p)': 1 / math.sqrt(self.kp) if self.kp > 0 else None,
            '1/(1-p)': 1 / (1 - self.p) if self.p != 1 else None,
        }

        # Убираем None значения
        self.base_values = {k: v for k, v in self.base_values.items()
                            if v is not None and not math.isnan(v)}

        # Ключевые отношения
        if self.ln_kp != 0:
            self.base_values['ln(N)/ln(K*p)'] = self.ln_N / self.ln_kp
            self.base_values['ln(K*p)/ln(N)'] = self.ln_kp / self.ln_N
            self.U = abs(self.ln_N / self.ln_kp)  # Ваш параметр U
            self.base_values['U'] = self.U

    def _load_extended_constants(self) -> Dict[str, float]:
        """Расширенный набор математических констант"""
        constants = {
            # Основные константы
            'pi': math.pi,
            'e': math.e,
            'golden_ratio': (1 + math.sqrt(5)) / 2,
            'sqrt2': math.sqrt(2),
            'sqrt3': math.sqrt(3),
            'sqrt5': math.sqrt(5),

            # Гамма-функции (важны для вашей модели)
            'gamma(1/2)': math.sqrt(math.pi),
            'gamma(1/3)': 2.678938534707748,
            'gamma(1/4)': 3.625609908221908,

            # Константы теории чисел
            'catalan': 0.915965594177219,
            'apery': 1.202056903159594,
            'khinchin': 2.685452001065306,
            'glaisher': 1.282427129100622,
            'meissel_mertens': 0.261497212847642,

            # Динамические системы
            'feigenbaum': 4.669201609102990,
            'feigenbaum2': 2.502907875095892,

            # Алгебраические константы
            'plastic': 1.324717957244746,
            'supergolden': 1.465571231876768,
            'conway': 1.303577269034296,

            # Связанные с e и π
            'e^pi': math.exp(math.pi),
            'pi^e': math.pi ** math.e,
            'e^e': math.exp(math.e),
            '2pi': 2 * math.pi,
            'pi/2': math.pi / 2,
            'pi/3': math.pi / 3,
            'pi/4': math.pi / 4,
            'pi/6': math.pi / 6,

            # Логарифмические константы
            'ln2': math.log(2),
            'ln3': math.log(3),
            'ln10': math.log(10),
            'lnpi': math.log(math.pi),

            # Специальные константы
            'euler_mascheroni': 0.577215664901532,
            'gauss': 0.834626841674073,
            'lemniscate': 2.622057554292119,
            'laplace_limit': 0.662743419,
            'omega': 0.567143290,
        }
        return constants

    @lru_cache(maxsize=10000)
    def _check_constant(self, value: float, const_value: float) -> float:
        """Кэшированная проверка совпадения с константой"""
        if const_value == 0:
            return float('inf')
        return abs((value - const_value) / const_value) * 100

    def _is_interesting(self, value: float) -> bool:
        """Быстрая проверка, интересно ли значение"""
        # Игнорируем слишком большие/малые значения
        if abs(value) < 1e-15 or abs(value) > 1e15:
            return False

        # Игнорируем значения, близкие к тривиальным
        trivial_values = {0, 1, math.e, math.pi, self.K, self.p}
        for tv in trivial_values:
            if abs(value - tv) < 1e-10:
                return False

        return True

    def _generate_expressions_level(self, expressions: List[Tuple[str, float]],
                                    max_new: int = 1000) -> List[Tuple[str, float]]:
        """Генерация нового уровня выражений"""
        new_expressions = []
        seen = set()

        n = len(expressions)
        # Используем стратегию: проверяем самые перспективные пары
        for i in range(min(n, 100)):
            for j in range(i, min(n, 100)):
                expr1_name, expr1_val = expressions[i]
                expr2_name, expr2_val = expressions[j]

                # Генерируем операции
                ops = [
                    (f"({expr1_name} + {expr2_name})", expr1_val + expr2_val),
                    (f"({expr1_name} - {expr2_name})", expr1_val - expr2_val),
                    (f"({expr2_name} - {expr1_name})", expr2_val - expr1_val),
                ]

                # Умножение
                mul_val = expr1_val * expr2_val
                if abs(mul_val) < 1e20:
                    ops.append((f"({expr1_name} * {expr2_name})", mul_val))

                # Деление
                if abs(expr2_val) > 1e-10:
                    div_val = expr1_val / expr2_val
                    if abs(div_val) < 1e20:
                        ops.append((f"({expr1_name} / {expr2_name})", div_val))

                if abs(expr1_val) > 1e-10:
                    div_val = expr2_val / expr1_val
                    if abs(div_val) < 1e20:
                        ops.append((f"({expr2_name} / {expr1_name})", div_val))

                for name, val in ops:
                    if name in seen or not self._is_interesting(val):
                        continue

                    seen.add(name)
                    new_expressions.append((name, val))

                    if len(new_expressions) >= max_new:
                        return new_expressions

        return new_expressions

    def search_deep_matches(self, max_levels: int = 3,
                            min_error: float = 1e-7,
                            max_error: float = 0.5) -> List[MatchResult]:
        """Поиск совпадений с многоуровневым перебором"""
        all_matches = []
        expressions = list(self.base_values.items())

        print(f"Начинаем поиск с {len(expressions)} базовых выражений")
        print(f"Уровней: {max_levels}, диапазон ошибок: {min_error:.2e}% - {max_error:.2f}%")

        for level in range(max_levels):
            print(f"\nУровень {level + 1}:")
            print(f"  Выражений: {len(expressions)}")

            # Поиск совпадений среди текущих выражений
            level_matches = self._find_matches_in_expressions(expressions, min_error, max_error)
            all_matches.extend(level_matches)

            print(f"  Найдено совпадений: {len(level_matches)}")

            # Генерация нового уровня (если не последний)
            if level < max_levels - 1:
                new_exprs = self._generate_expressions_level(expressions)
                expressions.extend(new_exprs)
                print(f"  Добавлено новых выражений: {len(new_exprs)}")

        # Сортировка и удаление дубликатов
        all_matches.sort(key=lambda x: x.relative_percent)

        # Убираем дубликаты
        seen_formulas = set()
        unique_matches = []
        for match in all_matches:
            if match.formula not in seen_formulas:
                seen_formulas.add(match.formula)
                unique_matches.append(match)

        return unique_matches[:500]

    def _find_matches_in_expressions(self, expressions: List[Tuple[str, float]],
                                     min_error: float, max_error: float) -> List[MatchResult]:
        """Поиск совпадений среди списка выражений"""
        matches = []

        for expr_name, expr_value in expressions:
            if not self._is_interesting(expr_value):
                continue

            for const_name, const_value in self.constants.items():
                error_percent = self._check_constant(expr_value, const_value)

                if min_error <= error_percent <= max_error:
                    matches.append(MatchResult(
                        formula=expr_name,
                        calculated_value=expr_value,
                        constant_name=const_name,
                        constant_value=const_value,
                        absolute_error=abs(expr_value - const_value),
                        relative_percent=error_percent
                    ))

        return matches

    def analyze_best_matches(self, matches: List[MatchResult], top_n: int = 50):
        """Анализ лучших совпадений"""
        print(f"ТОП-{top_n} ЛУЧШИХ СОВПАДЕНИЙ:")

        for i, match in enumerate(matches[:top_n], 1):
            print(f"{i:3d}. {match.formula}")
            print(f"     = {match.calculated_value:.12g}")
            print(f"     ≈ {match.constant_name} = {match.constant_value:.12g}")
            print(f"     Ошибка: {match.relative_percent:.10f}%")
            print(f"     Абс. ошибка: {match.absolute_error:.6g}")
            print()

    def save_results(self, matches: List[MatchResult], filename: str):
        """Сохранение результатов в JSON"""
        results = []
        for match in matches:
            results.append({
                'formula': match.formula,
                'calculated': float(match.calculated_value),
                'constant': match.constant_name,
                'constant_value': float(match.constant_value),
                'error_percent': float(match.relative_percent),
                'abs_error': float(match.absolute_error)
            })

        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nРезультаты сохранены в {filename} ({len(results)} совпадений)")


def main():
    """Основная функция с улучшенным поиском"""
    start_time = time.time()

    # Инициализация с вашими параметрами
    hunter = AdvancedConstantHunter(K=8, p=0.052702, N=9.70e122)
    print("ПОИСК ФУНДАМЕНТАЛЬНЫХ СТРУКТУР В ПАРАМЕТРАХ СЕТИ")
    print(f"Параметры: K={hunter.K}, p={hunter.p:.6f}, N={hunter.N:.2e}")
    print(f"U = ln(N)/|ln(Kp)| = {hunter.U:.6f}")
    print()

    # Запуск поиска
    matches = hunter.search_deep_matches(
        max_levels=3,
        min_error=1e-7,  # 0.0000001%
        max_error=0.5  # 0.5%
    )

    # Анализ результатов
    if matches:
        hunter.analyze_best_matches(matches, top_n=50)
        hunter.save_results(matches, "deep_matches.json")

        # Статистика
        stats = {}
        for match in matches:
            stats[match.constant_name] = stats.get(match.constant_name, 0) + 1

        print("СТАТИСТИКА ПО КОНСТАНТАМ:")
        print(f"{'=' * 60}")
        for const, count in sorted(stats.items(), key=lambda x: x[1], reverse=True):
            print(f"  {const:20}: {count:4d} совпадений")

    total_time = time.time() - start_time
    print(f"\nОбщее время выполнения: {total_time:.1f} секунд")


if __name__ == "__main__":
    main()