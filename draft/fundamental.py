import math
import json
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import time


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


class ConstantHunter:
    def __init__(self, K=8, p=0.0527, N=9.7e122):
        self.K = K
        self.p = p
        self.N = N

        # Кэш для ускорения
        self._log_cache = {}
        self._sqrt_cache = {}

        # Базовые комбинации
        self.base_expressions = self._generate_base_expressions()

        # Фундаментальные константы
        self.constants = self._load_fundamental_constants()

        # Константы для быстрого доступа
        self.constants_list = list(self.constants.items())

    def _cached_log(self, x):
        """Кэшированный логарифм"""
        if x not in self._log_cache:
            self._log_cache[x] = math.log(x) if x > 0 else float('nan')
        return self._log_cache[x]

    def _cached_sqrt(self, x):
        """Кэшированный квадратный корень"""
        if x not in self._sqrt_cache:
            self._sqrt_cache[x] = math.sqrt(x) if x >= 0 else float('nan')
        return self._sqrt_cache[x]

    def _generate_base_expressions(self) -> Dict[str, float]:
        """Генерация базовых выражений (оптимизированная)"""
        exprs = {}

        # Предвычисленные значения
        K = self.K
        p = self.p
        N = self.N
        kp = K * p
        ln_N = self._cached_log(N)
        ln_K = self._cached_log(K)

        # Базовые величины
        exprs['K'] = float(K)
        exprs['p'] = float(p)
        exprs['N'] = float(N)
        exprs['K*p'] = kp
        exprs['ln(N)'] = ln_N
        exprs['ln(K)'] = ln_K

        if p > 0:
            exprs['ln(p)'] = self._cached_log(p)

        if kp > 0:
            exprs['ln(K*p)'] = self._cached_log(kp)

        # Ключевые отношения
        if kp > 0:
            ln_kp = self._cached_log(kp)
            if ln_kp != 0:
                exprs['ln(N)/ln(K*p)'] = ln_N / ln_kp
                exprs['ln(K*p)/ln(N)'] = ln_kp / ln_N

        # Основные комбинации
        exprs['1+p'] = 1 + p
        exprs['1-p'] = 1 - p
        exprs['(K+p)*p'] = (K + p) * p

        if kp >= 0:
            exprs['sqrt(K*p)'] = self._cached_sqrt(kp)

        if kp > 0:
            exprs['1/sqrt(K*p)'] = 1 / self._cached_sqrt(kp)

        if p != 1:
            exprs['1/(1-p)'] = 1 / (1 - p)

        # Минимальный набор дополнительных выражений
        exprs['K+p'] = K + p
        exprs['K-p'] = K - p

        if p != 0:
            exprs['K/p'] = K / p
            exprs['p/K'] = p / K

        exprs['sqrt(K)'] = self._cached_sqrt(K)

        if p >= 0:
            exprs['sqrt(p)'] = self._cached_sqrt(p)

        # Убираем тригонометрию и сокращаем набор
        valid_exprs = {}
        for k, v in exprs.items():
            if not (math.isnan(v) or math.isinf(v)):
                valid_exprs[k] = float(v)

        return valid_exprs

    def _load_fundamental_constants(self) -> Dict[str, float]:
        """Загрузка только самых важных констант"""
        constants = {
            # Самые важные математические константы
            # Основные
            'pi': math.pi,
            'golden_ratio': (1 + math.sqrt(5)) / 2,

            # Корни
            'sqrt2': math.sqrt(2),
            'sqrt3': math.sqrt(3),
            'sqrt5': math.sqrt(5),
            'sqrt7': math.sqrt(7),
            'sqrt10': math.sqrt(10),

            # Логарифмы
            'ln2': math.log(2),
            'ln3': math.log(3),
            'ln10': math.log(10),
            'lnpi': math.log(math.pi),

            # === ПОПУЛЯРНЫЕ МАТЕМАТИЧЕСКИЕ КОНСТАНТЫ ===
            'euler_mascheroni': 0.577215664901532,  # γ
            'catalan': 0.915965594177219,  # G
            'apery': 1.202056903159594,  # ζ(3)
            'khinchin': 2.685452001065306,  # K₀
            'glaisher': 1.282427129100622,  # A
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

            # === КОНСТАНТЫ ИЗ ТЕОРИИ ХАОСА ===
            'feigenbaum': 4.669201609102990,  # δ
            'feigenbaum2': 2.502907875095892,  # α

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
        }
        return constants

    def _is_trivial_value(self, value: float) -> bool:
        """Быстрая проверка тривиальных значений"""
        # Проверяем простые соотношения
        sqrt_K_over_2 = math.sqrt(self.K) / 2
        four_over_sqrt_K = 4 / math.sqrt(self.K)

        if abs(value - sqrt_K_over_2) < 1e-10:
            return True
        if abs(value - four_over_sqrt_K) < 1e-10:
            return True
        if abs(value - 1.0) < 1e-10:  # Единица часто тривиальна
            return True
        if abs(value - math.e) < 1e-10:  # Точное e обычно тривиально
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

    def search_matches_fast(self, max_operations: int = 3,
                            min_percent: float = 1e-7,
                            max_percent: float = 0.5,
                            max_expressions: int = 2000) -> List[MatchResult]:
        """Оптимизированный поиск совпадений"""
        matches = []

        # Начинаем с базовых выражений
        current_expressions = list(self.base_expressions.items())
        all_expressions = dict(self.base_expressions)

        print(f"Поиск от {min_percent:.2e}% до {max_percent:.2f}%")
        print(f"Базовых выражений: {len(current_expressions)}")
        print(f"Констант для проверки: {len(self.constants_list)}")

        total_start = time.time()

        for op_count in range(1, max_operations + 1):
            level_start = time.time()
            new_expressions = []

            print(f"\nУровень {op_count}:")
            print(f"  Текущих выражений: {len(current_expressions)}")

            # Ограничиваем количество пар для проверки
            n = len(current_expressions)
            max_pairs = min(50000, n * (n - 1) // 2)  # Ограничение пар

            checked_pairs = 0
            formulas_generated = 0

            # Проверяем ограниченное количество пар
            for i in range(min(n, 300)):  # Ограничиваем первый индекс
                for j in range(i, min(n, 300)):  # Ограничиваем второй индекс
                    if checked_pairs >= max_pairs:
                        break

                    expr1 = current_expressions[i]
                    expr2 = current_expressions[j]
                    checked_pairs += 1

                    # Простые операции (самые быстрые)
                    ops = []

                    # Сложение
                    val_sum = expr1[1] + expr2[1]
                    ops.append((f"({expr1[0]} + {expr2[0]})", val_sum))

                    # Вычитание (если не тривиально)
                    val_sub1 = expr1[1] - expr2[1]
                    if abs(val_sub1) > 1e-10 and not self._is_trivial_value(val_sub1):
                        ops.append((f"({expr1[0]} - {expr2[0]})", val_sub1))

                    val_sub2 = expr2[1] - expr1[1]
                    if abs(val_sub2) > 1e-10 and not self._is_trivial_value(val_sub2):
                        ops.append((f"({expr2[0]} - {expr1[0]})", val_sub2))

                    # Умножение
                    val_mul = expr1[1] * expr2[1]
                    if abs(val_mul) < 1e50:
                        ops.append((f"({expr1[0]} * {expr2[0]})", val_mul))

                    # Деление (если не тривиально)
                    if abs(expr2[1]) > 1e-10:
                        val_div1 = expr1[1] / expr2[1]
                        if not self._is_trivial_value(val_div1):
                            ops.append((f"({expr1[0]} / {expr2[0]})", val_div1))

                    if abs(expr1[1]) > 1e-10:
                        val_div2 = expr2[1] / expr1[1]
                        if not self._is_trivial_value(val_div2):
                            ops.append((f"({expr2[0]} / {expr1[0]})", val_div2))

                    # Проверяем каждую операцию
                    for formula, value in ops:
                        formulas_generated += 1

                        # Быстрая проверка тривиальности
                        if self._is_trivial_value(value):
                            continue

                        # Проверяем с константами
                        for const_name, const_value in self.constants_list:
                            rel_error = self._fast_check_constant(
                                value, const_value, min_percent, max_percent
                            )

                            if rel_error is not None:
                                matches.append(MatchResult(
                                    formula=formula,
                                    calculated_value=value,
                                    constant_name=const_name,
                                    constant_value=const_value,
                                    absolute_error=abs(value - const_value),
                                    relative_percent=rel_error
                                ))

                        # Добавляем в новые выражения (ограниченное количество)
                        if len(new_expressions) < 1000 and formula not in all_expressions:
                            all_expressions[formula] = value
                            new_expressions.append((formula, value))

                if checked_pairs >= max_pairs:
                    break

            # Обновляем список выражений
            if new_expressions:
                # Ограничиваем общее количество выражений
                current_expressions.extend(new_expressions)
                if len(current_expressions) > max_expressions:
                    # Оставляем наиболее разнообразные значения
                    current_expressions.sort(key=lambda x: abs(x[1]))
                    current_expressions = current_expressions[:max_expressions]

            level_time = time.time() - level_start
            print(f"  Проверено пар: {checked_pairs:,}")
            print(f"  Сгенерировано формул: {formulas_generated:,}")
            print(f"  Найдено совпадений: {len(matches)}")
            print(f"  Время уровня: {level_time:.1f} сек")

            if level_time > 300:  # Если уровень занимает больше 5 минут
                print(f"  ⚠️  Уровень {op_count + 1} и далее будут очень долгими")
                print(f"  Рекомендуется остановиться или уменьшить max_operations")
                break

        total_time = time.time() - total_start

        # Сортируем и фильтруем результаты
        matches.sort(key=lambda x: x.relative_percent)

        # Убираем дубликаты
        unique_matches = []
        seen = set()
        for match in matches:
            key = (match.formula, match.constant_name)
            if key not in seen:
                seen.add(key)
                unique_matches.append(match)

        print(f"\nИтого:")
        print(f"  Общее время: {total_time:.1f} сек")
        print(f"  Всего формул проверено: ~{formulas_generated:,}")
        print(f"  Уникальных совпадений: {len(unique_matches)}")

        return unique_matches[:500]

    def save_results(self, matches: List[MatchResult], filename: str = "nontrivial_matches.json"):
        """Сохранение результатов в файл"""
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

        print(f"\nСохранено {len(results_data)} результатов в {filename}")

        # Также сохраняем текстовый отчет
        self.save_text_report(matches, filename.replace('.json', '.txt'))

    def save_text_report(self, matches: List[MatchResult], filename: str = "nontrivial_matches.txt"):
        """Сохранение текстового отчета"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("НЕТРИВИАЛЬНЫЕ СОВПАДЕНИЯ С ФУНДАМЕНТАЛЬНЫМИ КОНСТАНТАМИ\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Исходные значения:\n")
            f.write(f"  K = {self.K}\n")
            f.write(f"  p = {self.p}\n")
            f.write(f"  N = {self.N:.2e}\n")
            f.write(f"  K*p = {self.K * self.p:.6f}\n")
            f.write(f"  ln(N) = {math.log(self.N):.6f}\n")
            f.write(f"  ln(K*p) = {math.log(self.K * self.p):.6f}\n\n")

            f.write(f"Диапазон поиска: {1e-7:.2e}% - {0.5:.2f}%\n")
            f.write(f"Найдено совпадений: {len(matches)}\n\n")

            f.write("Лучшие совпадения:\n")
            f.write("-" * 80 + "\n")

            # Группируем по константам
            matches_by_constant = {}
            for match in matches:
                if match.constant_name not in matches_by_constant:
                    matches_by_constant[match.constant_name] = []
                matches_by_constant[match.constant_name].append(match)

            # Выводим топ-5 для каждой константы
            for const_name, const_matches in sorted(matches_by_constant.items()):
                f.write(f"\n{const_name.upper()}:\n")
                for i, match in enumerate(const_matches[:5], 1):
                    f.write(f"{i:3d}. {match.formula}\n")
                    f.write(f"     = {match.calculated_value:.10g}\n")
                    f.write(f"     ≈ {match.constant_name} = {match.constant_value:.10g}\n")
                    f.write(f"     Относительная погрешность: {match.relative_percent:.8f}%\n")
                    f.write(f"     Абсолютная погрешность: {match.absolute_error:.6g}\n")
                    f.write("     " + "-" * 40 + "\n")

            f.write(f"\nВсего найдено совпадений: {len(matches)}\n")


def main():
    """Основная функция с разумными настройками"""
    #1.2e147
    hunter = ConstantHunter(K=8, p=0.052702, N=9.7e123)

    print("Поиск нетривиальных приближённых совпадений")
    print("=" * 60)

    # Быстрый поиск с разумными ограничениями
    matches = hunter.search_matches_fast(
        max_operations=3,  # 3 уровня достаточно
        min_percent=1e-7,  # 0.0000001%
        max_percent=0.5,  # 0.5%
        max_expressions=1500  # Ограничение выражений
    )

    if matches:
        hunter.save_results(matches, "fast_nontrivial_matches.json")

        print("\nТОП-10 лучших совпадений:")
        print("=" * 60)
        for i, match in enumerate(matches[:200], 1):
            print(f"{i:2d}. {match}")

        # Статистика по константам
        stats = {}
        for match in matches:
            stats[match.constant_name] = stats.get(match.constant_name, 0) + 1

        print("\nСтатистика по константам:")
        for const_name, count in sorted(stats.items(), key=lambda x: x[1], reverse=True):
            print(f"  {const_name:20}: {count:4d} совпадений")
    else:
        print("\nСовпадений не найдено в заданном диапазоне.")


if __name__ == "__main__":
    main()