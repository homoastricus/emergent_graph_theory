import json
import re
import time
from typing import List, Dict, Optional

from scipy.special import gamma

import math


def lnN_zeta():
    # ВАЖНО: интерпретируем как ln N (НЕ exp)
    exponent = 1.5 + math.pi ** 2 / 6.0
    lnN_zeta = (6.0 ** exponent)
    return lnN_zeta




class SmartConstantFinder:
    def __init__(self, max_components=5, min_quantum=1e-300, max_quantum=0.01,
                 constant_variable=False, max_expressions=300000000000, max_pairs_per_level=500000000000):
        self.max_components = max_components
        self.min_quantum = min_quantum
        self.max_quantum = max_quantum
        self.constant_variable = constant_variable
        self.max_expressions = max_expressions
        self.max_pairs_per_level = max_pairs_per_level

        # Кэши для ускорения
        self._log_cache = {}
        self._sqrt_cache = {}

        # Инициализация K и N
        #N_test = lnN_zeta()
        #p_test = 1 / (6 * N_test ** (1 / 3))
        self.K = 6
        self.N = lnN_zeta()

        # Вычисление всех компонентов
        self.base_expressions = self.create_base_expressions()

        # Создание всех констант
        self.all_constants = self.create_all_constants()
        self.constants_list = list(self.all_constants.items())

        # Результаты
        self.results = []

        # Статистика
        self.tautologies_filtered = 0

    def _cached_log(self, x):
        """Кэшированное вычисление логарифма"""
        if x not in self._log_cache:
            self._log_cache[x] = math.log(x) if x > 0 else float('nan')
        return self._log_cache[x]

    def _cached_sqrt(self, x):
        """Кэшированное вычисление квадратного корня"""
        if x not in self._sqrt_cache:
            self._sqrt_cache[x] = math.sqrt(x) if x >= 0 else float('nan')
        return self._sqrt_cache[x]

    def create_base_expressions(self) -> Dict[str, float]:
        """Создание словаря базовых выражений (как в образце)"""
        K = self.K
        N = self.N
        ln_N = self._cached_log(N)
        ln_K = self._cached_log(K)

        exprs = {
            'K': float(K),
            # 'N': float(N),
            'ln(N)': ln_N,
            'ln(K)': ln_K,
            'ln(ln(N))': self._cached_log(ln_N),
            'ln(ln(K))': self._cached_log(ln_K),
            # 'ln(ln(ln(N)))': self._cached_log(self._cached_log(ln_N)),
            # '1/ln(N)': 1 / ln_N,
            # '1/ln(K)': 1 / ln_K,
            # '1/ln(ln(N))': 1 / self._cached_log(ln_N),
            # '1/ln(ln(K))': 1 / self._cached_log(ln_K),
            # '1/ln(ln(ln(N)))': 1 / self._cached_log(self._cached_log(ln_N)),
            'ln(N)^2': ln_N ** 2,
            'ln(N)^3': ln_N ** 3,
            'ln(K)^2': ln_K ** 2,
            'ln(K)^3': ln_K ** 3,
            # 'ln(K)/ln(N)': ln_K / ln_N,
            # 'ln(N)/ln(K)': ln_N / ln_K,
            # '(ln(K)/ln(N))^2': (ln_K / ln_N) ** 2,
            # '(ln(N)/ln(K))^2': (ln_N / ln_K) ** 2,
            # 'ln(K)^2/ln(N)^2': (ln_K ** 2) / (ln_N ** 2),
            # 'ln(N)^2/ln(K)^2': (ln_N ** 2) / (ln_K ** 2),
            # 'ln(K)^3/ln(N)^3': (ln_K ** 3) / (ln_N ** 3),
            # 'K/ln(N)': K / ln_N,
            # 'ln(N)/K': ln_N / K,
            'K^3': K ** 3,
            'K^2': K ** 2,
            # 'ln(N)/K^3': ln_N / K ** 3,
            # 'ln(N)-K': ln_N - K,
            # 'ln(N)+K': ln_N + K,
            # 'K - ln(N)^3': K - ln_N ** 3,
            '2': 2.0,
            '3': 3.0,
            # '1/2': 0.5,
            # '1/3': 1 / 3,
            # '1/K': 1 / K,
        }

        # Фильтруем некорректные значения
        valid_exprs = {}
        for k, v in exprs.items():
            if not (math.isnan(v) or math.isinf(v)):
                valid_exprs[k] = float(v)

        return valid_exprs

    def create_all_constants(self):
        """Создание всех констант (как в образце)"""
        constants = {}

        # Базовые константы
        constants.update({
            'LAMBDA': 0.05183093,
            'phi': (1 + math.sqrt(5)) / 2,
            'sqrt2': math.sqrt(2),
            'sqrt3': math.sqrt(3),
            'gamma_euler': 0.5772156649015329,
            'ln2': math.log(2),
            'ln3': math.log(3),
            'lnpi': math.log(math.pi),
            'pi': math.pi,
            'euler': math.e,
            'euler_mascheroni': 0.5772156649015329,
            'golden_ratio': 1.618033988749895,
            'feigenbaum_delta': 4.669201609102990,
            'feigenbaum_alpha': 2.5029078750958928,
            'glaisher': 1.2824271291006226,
            'apery': 1.2020569031595942,
            'catalan': 0.9159655941772190,
            'khinchin': 2.6854520010653062,
            'alpha': 7.2973525693e-3,
            'gamma_1_2': gamma(0.5),
            'gamma_1_3': gamma(1 / 3),
            'gamma_1_4': gamma(0.25),
            'gamma_2_3': gamma(2 / 3),
            'paperfolding': 0.8507361882018673
        })

        # Генерация вариаций если нужно
        if self.constant_variable:
            print("🔄 Генерация вариаций констант...")
            variations = self.generate_constant_variations(constants)
            constants.update(variations)
            print(f"   Добавлено {len(variations)} вариаций")

        print(f"📚 Всего констант для проверки: {len(constants)}")
        return constants

    def generate_constant_variations(self, constants_dict: Dict[str, float]) -> Dict[str, float]:
        """Генерация вариаций констант (взято из образца)"""
        variations = {}

        # Обратные значения
        for name, value in constants_dict.items():
            if abs(value) > 1e-10:
                inv_name = f"1/{name}"
                if inv_name not in variations:
                    variations[inv_name] = 1.0 / value

        # Комбинации
        pure_constants = {k: v for k, v in constants_dict.items()
                          if not any(op in k for op in ['/', '+', '-', '*'])}

        pure_names = list(pure_constants.keys())
        pure_values = list(pure_constants.values())

        max_combinations = 500000000000
        combinations_generated = 0

        for i in range(len(pure_names)):
            for j in range(i + 1, len(pure_names)):
                if combinations_generated >= max_combinations:
                    break

                name1, val1 = pure_names[i], pure_values[i]
                name2, val2 = pure_names[j], pure_values[j]

                # Умножение
                prod_val = val1 * val2
                prod_name = f"({name1} * {name2})"
                if prod_name not in variations:
                    variations[prod_name] = prod_val
                    combinations_generated += 1

                # Деление
                if abs(val2) > 1e-10:
                    div1_val = val1 / val2
                    div1_name = f"({name1} / {name2})"
                    if div1_name not in variations:
                        variations[div1_name] = div1_val
                        combinations_generated += 1

                if abs(val1) > 1e-10:
                    div2_val = val2 / val1
                    div2_name = f"({name2} / {name1})"
                    if div2_name not in variations:
                        variations[div2_name] = div2_val
                        combinations_generated += 1

        return variations

    def _is_trivial_value(self, value: float) -> bool:
        """Фильтрует тривиальные значения (взято из образца)"""
        if abs(value - 1.0) < 1e-10:
            return True
        if abs(value) < 1e-10:
            return True
        if abs(value - math.e) < 1e-10:
            return True
        if abs(value - math.log(6)) < 1e-10:
            return True
        if abs(value - math.pi) < 1e-10:
            return True
        return False

    def _is_tautology(self, formula: str, value: float, const_name: str) -> bool:
        """Проверка на тавтологию (улучшенная версия из образца)"""
        # Циклические выражения
        tautology_patterns = [
            (r'\(([^)]+)\)\s*\*\s*\(1/\1\)', 'X * 1/X'),
            (r'\(1/([^)]+)\)\s*\*\s*\1', '1/X * X'),
            (r'\(([^)]+)\)\s*/\s*\(\1\)', 'X / X'),
            (r'\(([^)]+)\)\s*-\s*\(\1\)', 'X - X'),
            (r'\(([^)]+)\s*\+\s*([^)]+)\)\s*-\s*\2', '(X+Y)-Y = X'),
            (r'\(([^)]+)\s*\*\s*([^)]+)\)\s*/\s*\2', '(X*Y)/Y = X'),
        ]

        for pattern, _ in tautology_patterns:
            if re.search(pattern, formula):
                return True

        # Проверка на ln(6) = ln(2*3) = ln2 + ln3
        if 'ln2' in const_name and 'ln3' in const_name:
            if abs(value - (math.log(2) + math.log(3))) < 1e-10:
                if any(x in formula for x in ['ln(K)', 'ln(6)']):
                    return True

        return False

    def _fast_check_constant(self, value: float, const_value: float,
                             min_percent: float, max_percent: float) -> Optional[float]:
        """Быстрая проверка совпадения с константой"""
        if const_value == 0:
            return None
        rel_error = abs((value - const_value) / const_value) * 100
        if min_percent <= rel_error <= max_percent:
            return rel_error
        return None

    def search_matches_smart(self) -> List:
        matches = []
        tautologies_filtered = 0

        current_expressions = list(self.base_expressions.items())
        all_expressions = dict(self.base_expressions)

        print(f"\n🔍 НАЧАЛО УМНОГО ПОИСКА СОВПАДЕНИЙ")
        print(f"Базовых выражений: {len(current_expressions)}")
        print(f"Констант для проверки: {len(self.constants_list)}")
        print(f"Диапазон ошибки: {self.min_quantum}% - {self.max_quantum}%")
        print("=" * 80)

        total_start = time.time()
        total_formulas_checked = 0

        for level in range(1, self.max_components + 1):
            level_start = time.time()

            n = len(current_expressions)
            total_pairs = n * (n + 1) // 2

            # ОЦЕНКА ВРЕМЕНИ И АВТО-ОГРАНИЧЕНИЕ
            est_speed = 40000  # пар в секунду (из вашей статистики)
            est_time = total_pairs / est_speed

            print(f"\n[{time.strftime('%H:%M:%S')}] Уровень {level}: {n:,} выражений")
            print(f"  Всего пар: {total_pairs:,}")
            print(f"  Примерное время: {est_time:.0f}с ({est_time / 60:.1f}мин)")

            # ЕСЛИ ВРЕМЯ > 5 МИНУТ - СПРАШИВАЕМ ИЛИ АВТО-ОГРАНИЧИВАЕМ
            MAX_TIME_PER_LEVEL = 300  # 5 минут максимум на уровень

            if est_time > MAX_TIME_PER_LEVEL:
                # Вычисляем, сколько выражений можем обработать за 5 минут
                max_pairs = MAX_TIME_PER_LEVEL * est_speed
                # n*(n+1)/2 = max_pairs → n ≈ sqrt(2*max_pairs)
                max_n = int((2 * max_pairs) ** 0.5)
                max_n = min(max_n, n)  # не больше доступного

                print(f"  ⚠️ ВНИМАНИЕ: Уровень займет {est_time / 60:.1f} минут!")
                print(f"  Авто-ограничение: обработка только {max_n:,} выражений (~{MAX_TIME_PER_LEVEL}с)")

                # Сортируем по "интересности" и берем топ
                if level > 1:
                    current_expressions.sort(key=lambda x: self._score_value(x[1]), reverse=True)

                n = max_n
                current_expressions = current_expressions[:n]
                total_pairs = n * (n + 1) // 2
                print(f"  Скорректированное время: {total_pairs / est_speed:.0f}с")

            new_expressions = []
            checked_pairs = 0
            formulas_generated = 0
            level_matches = 0

            last_report_time = time.time()

            for i in range(n):
                for j in range(i, n):
                    checked_pairs += 1

                    # Индикатор прогресса каждые 2 секунды
                    current_time = time.time()
                    if current_time - last_report_time >= 2:
                        progress = checked_pairs / total_pairs * 100
                        elapsed = current_time - level_start
                        eta = (elapsed / progress * 100) - elapsed if progress > 0 else 0

                        print(f"  [{time.strftime('%H:%M:%S')}] Прогресс: {progress:.1f}% | "
                              f"Пар: {checked_pairs:,}/{total_pairs:,} | "
                              f"Найдено: {level_matches} | "
                              f"ETA: {eta:.0f}с")
                        last_report_time = current_time

                    expr1 = current_expressions[i]
                    expr2 = current_expressions[j]

                    # Генерация операций
                    ops = []

                    val_sum = expr1[1] + expr2[1]
                    if not self._is_trivial_value(val_sum):
                        ops.append((f"({expr1[0]} + {expr2[0]})", val_sum))

                    val_sub1 = expr1[1] - expr2[1]
                    if abs(val_sub1) > 1e-10 and not self._is_trivial_value(val_sub1):
                        ops.append((f"({expr1[0]} - {expr2[0]})", val_sub1))

                    val_sub2 = expr2[1] - expr1[1]
                    if abs(val_sub2) > 1e-10 and not self._is_trivial_value(val_sub2):
                        ops.append((f"({expr2[0]} - {expr1[0]})", val_sub2))

                    val_mul = expr1[1] * expr2[1]
                    if abs(val_mul) < 1e50 and not self._is_trivial_value(val_mul):
                        ops.append((f"({expr1[0]} * {expr2[0]})", val_mul))

                    if abs(expr2[1]) > 1e-10:
                        val_div1 = expr1[1] / expr2[1]
                        if not self._is_trivial_value(val_div1):
                            ops.append((f"({expr1[0]} / {expr2[0]})", val_div1))

                    if abs(expr1[1]) > 1e-10:
                        val_div2 = expr2[1] / expr1[1]
                        if not self._is_trivial_value(val_div2):
                            ops.append((f"({expr2[0]} / {expr1[0]})", val_div2))

                    for formula, value in ops:
                        formulas_generated += 1
                        total_formulas_checked += 1

                        # Проверка констант
                        for const_name, const_value in self.constants_list:
                            rel_error = self._fast_check_constant(
                                value, const_value, self.min_quantum, self.max_quantum
                            )

                            if rel_error is not None:
                                if self._is_tautology(formula, value, const_name):
                                    tautologies_filtered += 1
                                    continue

                                matches.append({
                                    'formula': formula,
                                    'value': value,
                                    'constant': const_name,
                                    'const_value': const_value,
                                    'abs_error': abs(value - const_value),
                                    'rel_error': rel_error
                                })

                                level_matches += 1

                                if len(matches) > 1000:
                                    matches.sort(key=lambda x: x['rel_error'])
                                    matches = matches[:500]

                        # Добавляем новые выражения (ТОЛЬКО ЛУЧШИЕ)
                        if len(new_expressions) < self.max_expressions:
                            if formula not in all_expressions:
                                all_expressions[formula] = value
                                new_expressions.append((formula, value))

            level_time = time.time() - level_start
            print(f"\n  ✅ Уровень {level} завершен за {level_time:.1f}с")
            print(f"     Пар: {checked_pairs:,}, Формул: {formulas_generated:,}")
            print(f"     Совпадений: {level_matches}, Всего: {len(matches)}")

            # Обновляем для следующего уровня
            if new_expressions:
                # Берем только топ-N по качеству для следующего уровня
                if len(new_expressions) > self.max_expressions:
                    new_expressions.sort(key=lambda x: self._score_value(x[1]), reverse=True)
                    new_expressions = new_expressions[:self.max_expressions]
                current_expressions = new_expressions
            else:
                print("  Нет новых выражений. Остановка.")
                break

        total_time = time.time() - total_start

        # Дедупликация
        unique_matches = []
        seen = set()
        for match in matches:
            key = (match['formula'], match['constant'])
            if key not in seen:
                seen.add(key)
                unique_matches.append(match)

        unique_matches.sort(key=lambda x: x['rel_error'])

        print(f"\n{'=' * 80}")
        print(f"✅ ПОИСК ЗАВЕРШЕН за {total_time:.1f}с")
        print(f"Уникальных совпадений: {len(unique_matches)}")

        return unique_matches[:500]

    def _score_value(self, value: float) -> float:
        """Оценка "интересности" значения (чем выше, тем лучше)"""
        # Штраф за тривиальные значения
        if abs(value) < 1e-10:
            return 0.0
        if abs(value - 1.0) < 1e-10:
            return 0.0
        if abs(value - round(value)) < 1e-10 and abs(value) < 10:
            return 0.0

        # Бонус за значения в "разумном" диапазоне
        if 1e-5 < abs(value) < 1e5:
            return 1.0

        # Значения далеко от 0 и не слишком большие
        if 1e-10 < abs(value) < 1e10:
            return 0.5

        return 0.1

    def print_results(self, matches):
        """Вывод результатов в виде таблицы"""
        if not matches:
            print("Не найдено совпадений в заданном диапазоне ошибки.")
            return

        print(f"ЛУЧШИЕ СОВПАДЕНИЯ (показано {len(matches)})")
        print(f"{'Формула':<45} {'Значение':<15} {'Константа':<25} {'Абс. ошибка':<12} {'Отн. ошибка %':<12}")

        for i, match in enumerate(matches[:1000], 1):
            formula = match['formula'][:43]
            value = f"{match['value']:.10f}"
            constant = match['constant'][:23]
            abs_err = f"{match['abs_error']:.2e}"
            rel_err = f"{match['rel_error']:.8f}"

            if i <= 999:  # Показываем только первые 30 для краткости
                print(f"{formula:<45} {value:<15} {constant:<25} {abs_err:<12} {rel_err:<12}")

        if len(matches) > 999:
            print(f"... и еще {len(matches) - 999} результатов")

        print(f"{'=' * 120}")

    def run(self):
        """Запуск всего процесса поиска"""
        matches = self.search_matches_smart()
        self.print_results(matches)

        # Сохранение результатов
        if matches:
            self.save_results(matches)

    def save_results(self, matches, filename="smart_matches.json"):
        """Сохранение результатов в JSON"""
        results_data = []
        for match in matches[:500]:
            results_data.append({
                'formula': match['formula'],
                'value': float(match['value']),
                'constant': match['constant'],
                'const_value': float(match['const_value']),
                'abs_error': float(match['abs_error']),
                'rel_error': float(match['rel_error'])
            })

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)

        print(f"\n💾 Сохранено {len(results_data)} результатов в {filename}")


# Запуск
if __name__ == "__main__":
    # Настройка параметров
    MAX_COMPONENTS = 5
    MIN_QUANTUM = 1e-300
    MAX_QUANTUM = 0.0001
    CONSTANT_VARIABLE = False
    MAX_EXPRESSIONS = 300000000

    print("Конфигурация поиска:")
    print(f"  Максимум уровней: {MAX_COMPONENTS}")
    print(f"  Диапазон ошибки: {MIN_QUANTUM}% - {MAX_QUANTUM}%")
    print(f"  Комбинации констант: {'Вкл' if CONSTANT_VARIABLE else 'Выкл'}")
    print(f"  Максимум выражений: {MAX_EXPRESSIONS}")
    print()

    finder = SmartConstantFinder(
        max_components=MAX_COMPONENTS,
        min_quantum=MIN_QUANTUM,
        max_quantum=MAX_QUANTUM,
        constant_variable=CONSTANT_VARIABLE,
        max_expressions=MAX_EXPRESSIONS
    )

    finder.run()
