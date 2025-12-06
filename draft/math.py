import numpy as np
import pandas as pd
from itertools import combinations, product
import warnings

# Подавляем предупреждения о переполнении
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Заданные константы
constants = {
    '8': 8.0,
    '0.05': 0.05,
    'e': np.e,
    'pi': np.pi,
    #'delta': 4.669201609,
    '369.49': 369.49,
    '2': 2.0,
    'X': np.sqrt(8 * 369.49),  # ≈ 54.36837315940215
    'Y': 1.0 / np.sqrt(8 * 369.49)  # ≈ 0.018393046212144493
}

# Имена и значения отдельно для удобства
names = list(constants.keys())
values = list(constants.values())


# Операции с безопасной обработкой
def apply_operation(a, b, op):
    """Применить операцию между a и b с безопасной обработкой ошибок"""
    try:
        if op == '+':
            return a + b
        elif op == '-':
            return a - b
        elif op == '*':
            return a * b
        elif op == '/':
            if abs(b) < 1e-15:
                return None
            return a / b
        elif op == '^':
            # Проверяем на допустимость возведения в степень
            if a < 0 and not np.isclose(b, int(b)):
                return None  # Комплексные числа пропускаем
            # Проверяем на возможное переполнение
            if abs(a) > 1e100 and abs(b) > 10:
                return None
            if abs(a) < 1e-100 and abs(b) > 10:
                return None
            result = np.power(a, b)
            if not np.isfinite(result):
                return None
            return result
        elif op == 'log':
            if a <= 0:
                return None
            return np.log(a)
    except (OverflowError, ValueError, ZeroDivisionError, FloatingPointError):
        return None
    return None


def safe_power(base, exponent):
    """Безопасное возведение в степень с проверкой на переполнение"""
    try:
        # Проверка на потенциальное переполнение
        if abs(base) > 1e100 and abs(exponent) > 10:
            return None
        if abs(base) < 1e-100 and abs(exponent) > 10:
            return None

        result = base ** exponent
        if not np.isfinite(result):
            return None
        return result
    except (OverflowError, ValueError, ZeroDivisionError):
        return None


def get_single_ops(x, name):
    """Одноаргументные операции с безопасной обработкой"""
    results = []

    # Специальные операции с безопасной проверкой
    operations = [
        ('√', 0.5, lambda v: safe_power(v, 0.5)),
        ('∛', 1 / 3, lambda v: safe_power(v, 1 / 3)),
        ('∜', 1 / 4, lambda v: safe_power(v, 1 / 4)),
        ('²', 2, lambda v: safe_power(v, 2)),
        ('³', 3, lambda v: safe_power(v, 3)),
        ('⁴', 4, lambda v: safe_power(v, 4)),
        ('⁻¹', -1, lambda v: safe_power(v, -1) if abs(v) > 1e-15 else None),
        ('⁻²', -2, lambda v: safe_power(v, -2) if abs(v) > 1e-7 else None),
        ('⁻³', -3, lambda v: safe_power(v, -3) if abs(v) > 1e-5 else None),
        ('⁻⁴', -4, lambda v: safe_power(v, -4) if abs(v) > 1e-4 else None),
    ]

    for sym, p, func in operations:
        try:
            val = func(x)
            if val is not None and np.isfinite(val):
                if sym in ['√', '∛', '∜']:
                    results.append((f"{sym}({name})", val))
                else:
                    results.append((f"({name}){sym}", val))
        except:
            continue

    # Дробные степени
    for p in [1 / 2, 1 / 3, 1 / 4, 2 / 3, 3 / 2, 3 / 4, 4 / 3]:
        try:
            if x < 0 and not np.isclose(p * 2, int(p * 2)):
                continue  # Избегаем комплексных чисел

            val = safe_power(x, p)
            if val is not None and np.isfinite(val):
                if abs(p - 0.5) < 1e-10:
                    results.append((f"√({name})", val))
                elif abs(p - 1 / 3) < 1e-10:
                    results.append((f"∛({name})", val))
                else:
                    results.append((f"({name})^{p:.3f}", val))
        except:
            continue

    # Логарифмы
    if x > 1e-15:
        try:
            log_val = np.log(x)
            if np.isfinite(log_val):
                results.append((f"ln({name})", log_val))

            log10_val = np.log10(x)
            if np.isfinite(log10_val):
                results.append((f"log10({name})", log10_val))
        except:
            pass

    # Обратное значение
    if abs(x) > 1e-15:
        try:
            inv = 1.0 / x
            if np.isfinite(inv):
                results.append((f"1/({name})", inv))
        except:
            pass

    # Экспонента (только для умеренных значений)
    if -10 < x < 10:
        try:
            exp_val = np.exp(x)
            if np.isfinite(exp_val):
                results.append((f"exp({name})", exp_val))
        except:
            pass

    return results


def generate_expressions(max_complexity=4):
    """Генерация выражений заданной сложности с безопасной обработкой"""
    expressions = []
    added_values = set()

    # Добавляем округленные значения для устранения дубликатов
    def add_expression(expr, val):
        if val is None or not np.isfinite(val):
            return

        # Игнорируем слишком большие или маленькие значения
        if abs(val) > 1e100 or (abs(val) < 1e-100 and abs(val) > 0):
            return

        # Округляем до 14 знаков для более точного сравнения
        try:
            key = round(float(val), 14)
        except:
            return

        if key not in added_values:
            added_values.add(key)
            expressions.append((expr, val))

    # Одночлены
    print("Генерация одночленов...")
    for name, val in constants.items():
        # Само значение
        add_expression(name, val)

        # Степени и операции над одним числом
        for label, res in get_single_ops(val, name):
            add_expression(label, res)

    # Пары чисел с бинарными операциями
    print("Генерация парных выражений...")
    constant_items = list(constants.items())
    for (name1, val1), (name2, val2) in combinations(constant_items, 2):
        for op in ['+', '-', '*', '/', '^']:
            # Прямой порядок
            res = apply_operation(val1, val2, op)
            if res is not None:
                add_expression(f"({name1} {op} {name2})", res)

            # Обратный порядок для некоммутативных операций
            if op in ['-', '/', '^']:
                res_rev = apply_operation(val2, val1, op)
                if res_rev is not None:
                    add_expression(f"({name2} {op} {name1})", res_rev)

    # Тройки чисел
    if max_complexity >= 3:
        print("Генерация тройных выражений...")
        for (name1, val1), (name2, val2), (name3, val3) in combinations(constant_items, 3):
            # Различные порядки операций
            for op1 in ['+', '-', '*', '/']:
                for op2 in ['+', '-', '*', '/']:
                    # (a op1 b) op2 c
                    try:
                        res1 = apply_operation(val1, val2, op1)
                        if res1 is not None:
                            res2 = apply_operation(res1, val3, op2)
                            if res2 is not None:
                                add_expression(f"(({name1} {op1} {name2}) {op2} {name3})", res2)
                    except:
                        pass

                    # a op1 (b op2 c)
                    try:
                        res1 = apply_operation(val2, val3, op2)
                        if res1 is not None:
                            res2 = apply_operation(val1, res1, op1)
                            if res2 is not None:
                                add_expression(f"({name1} {op1} ({name2} {op2} {name3}))", res2)
                    except:
                        pass

    # Четверки чисел (ограниченное количество для производительности)
    if max_complexity >= 4:
        print("Генерация четверных выражений...")
        count = 0
        max_combinations = 500  # Ограничение для производительности

        for (name1, val1), (name2, val2), (name3, val3), (name4, val4) in combinations(constant_items, 4):
            if count >= max_combinations:
                break
            count += 1

            # Простые цепочки операций
            for ops in product(['+', '-', '*', '/'], repeat=2):
                try:
                    # Разные порядки вычислений
                    orders = [
                        f"(({name1} {ops[0]} {name2}) {ops[1]} {name3}) {op4} {name4}"
                        for op4 in ['+', '-', '*', '/']
                    ]

                    for expr_template in orders:
                        try:
                            # Вычисляем последовательно
                            if ops[0] == '+':
                                res = val1 + val2
                            elif ops[0] == '-':
                                res = val1 - val2
                            elif ops[0] == '*':
                                res = val1 * val2
                            elif ops[0] == '/':
                                if abs(val2) < 1e-15:
                                    continue
                                res = val1 / val2

                            if ops[1] == '+':
                                res = res + val3
                            elif ops[1] == '-':
                                res = res - val3
                            elif ops[1] == '*':
                                res = res * val3
                            elif ops[1] == '/':
                                if abs(val3) < 1e-15:
                                    continue
                                res = res / val3

                            # Третья операция (извлекаем из шаблона)
                            if ' + ' in expr_template.split('}')[-1]:
                                res = res + val4
                            elif ' - ' in expr_template.split('}')[-1]:
                                res = res - val4
                            elif ' * ' in expr_template.split('}')[-1]:
                                res = res * val4
                            elif ' / ' in expr_template.split('}')[-1]:
                                if abs(val4) < 1e-15:
                                    continue
                                res = res / val4

                            if res is not None and np.isfinite(res):
                                add_expression(expr_template, res)
                        except:
                            continue
                except:
                    continue

    print(f"Сгенерировано выражений: {len(expressions)}")
    return expressions


def find_accurate_relations(expressions, threshold_percent=0.1):  # Изменено с 0.5 на 0.1
    """Найти точные соотношения между выражениями с точностью 0.1%"""
    results = []
    n = len(expressions)

    print(f"Поиск соотношений среди {n} выражений (порог: {threshold_percent}%)...")

    for i in range(n):
        expr1, val1 = expressions[i]

        # Пропускаем некорректные значения
        if val1 is None or not np.isfinite(val1):
            continue

        if abs(val1) < 1e-15 or abs(val1) > 1e15:
            continue

        for j in range(i + 1, n):
            expr2, val2 = expressions[j]

            if val2 is None or not np.isfinite(val2):
                continue

            if abs(val2) < 1e-15 or abs(val2) > 1e15:
                continue

            try:
                # Проверка отношения близкого к 1 (val1/val2 ≈ 1)
                if abs(val2) > 1e-15:
                    ratio = val1 / val2
                    if np.isfinite(ratio) and 1e-15 < abs(ratio) < 1e15:
                        error_percent = abs(ratio - 1) * 100
                        if error_percent < threshold_percent:
                            results.append({
                                'expression1': expr1,
                                'value1': val1,
                                'expression2': expr2,
                                'value2': val2,
                                'ratio': ratio,
                                'error_%': error_percent,
                                'type': 'ratio ≈ 1'
                            })

                # Проверка разности близкой к 0 (val1 - val2 ≈ 0)
                min_abs = min(abs(val1), abs(val2))
                if min_abs > 1e-15:
                    diff_ratio = abs(val1 - val2) / min_abs
                    if diff_ratio * 100 < threshold_percent:
                        results.append({
                            'expression1': expr1,
                            'value1': val1,
                            'expression2': expr2,
                            'value2': val2,
                            'ratio': val1 / val2 if abs(val2) > 1e-15 else float('inf'),
                            'error_%': diff_ratio * 100,
                            'type': 'difference ≈ 0'
                        })
            except (ZeroDivisionError, OverflowError, FloatingPointError):
                continue

    # Удаление дубликатов с учетом порядка выражений
    unique_results = []
    seen_pairs = set()

    for result in results:
        expr1 = result['expression1']
        expr2 = result['expression2']
        # Создаем уникальный ключ с учетом порядка
        pair_key = (min(expr1, expr2), max(expr1, expr2))

        if pair_key not in seen_pairs:
            seen_pairs.add(pair_key)
            unique_results.append(result)

    print(f"Найдено {len(unique_results)} уникальных соотношений с погрешностью < {threshold_percent}%")
    return unique_results


def main():
    print("=" * 60)
    print("ПОИСК МАТЕМАТИЧЕСКИХ СООТНОШЕНИЙ С ТОЧНОСТЬЮ 0.1%")
    print("=" * 60)

    print("\nГенерация выражений...")
    expressions = generate_expressions(max_complexity=3)
    print(f"Сгенерировано {len(expressions)} уникальных выражений")

    print("\nПоиск соотношений с точностью <0.1%...")
    results = find_accurate_relations(expressions, threshold_percent=0.1)  # 0.1% вместо 0.5%

    # Обработка результатов
    if results:
        df = pd.DataFrame(results)

        # Удаление дубликатов
        df['expr_pair'] = df.apply(lambda x: tuple(sorted([x['expression1'], x['expression2']])), axis=1)
        df = df.drop_duplicates(subset=['expr_pair'], keep='first')

        # Сортировка по точности
        df = df.sort_values('error_%').reset_index(drop=True)

        print(f"\nНайдено {len(df)} соотношений с точностью <0.1%")
        print("\nСамые точные соотношения:")
        print("-" * 100)

        # Группируем по точности
        print("\n1. Сверхточные соотношения (ошибка < 0.001%):")
        super_accurate = df[df['error_%'] < 0.001]
        if len(super_accurate) > 0:
            for idx, row in super_accurate.iterrows():
                print(f"   {row['expression1']:40s} ≈ {row['expression2']:40s}")
                print(f"        Отношение: {row['ratio']:.12f}, Погрешность: {row['error_%']:.10f}%")
                print()

        print("\n2. Очень точные соотношения (0.001% ≤ ошибка < 0.01%):")
        very_accurate = df[(df['error_%'] >= 0.001) & (df['error_%'] < 0.01)]
        if len(very_accurate) > 0:
            for idx, row in very_accurate.iterrows():
                print(f"   {row['expression1']:40s} ≈ {row['expression2']:40s}")
                print(f"        Отношение: {row['ratio']:.10f}, Погрешность: {row['error_%']:.6f}%")
                print()

        print("\n3. Точные соотношения (0.01% ≤ ошибка < 0.1%):")
        accurate = df[(df['error_%'] >= 0.01) & (df['error_%'] < 0.1)]
        if len(accurate) > 0:
            for idx, row in accurate.head(20).iterrows():  # Покажем первые 20
                print(f"   {row['expression1']:40s} ≈ {row['expression2']:40s}")
                print(f"        Отношение: {row['ratio']:.8f}, Погрешность: {row['error_%']:.4f}%")
                print()
            if len(accurate) > 20:
                print(f"   ... и еще {len(accurate) - 20} соотношений")

        # Сохранение в файл
        output_file = 'accurate_relations_0.1percent.csv'
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\nПолные результаты сохранены в '{output_file}'")

        # Статистика
        print("\n" + "=" * 60)
        print("СТАТИСТИКА:")
        print(f"Всего соотношений: {len(df)}")
        print(f"Со сверхвысокой точностью (<0.001%): {len(super_accurate)}")
        print(f"С очень высокой точностью (0.001-0.01%): {len(very_accurate)}")
        print(f"С высокой точностью (0.01-0.1%): {len(accurate)}")

    else:
        print("Соотношений с точностью 0.1% не найдено")

    # Проверка известных соотношений с новым порогом
    print("\n" + "=" * 60)
    print("ПРОВЕРКА ИЗВЕСТНЫХ СООТНОШЕНИЙ (порог 0.1%):")
    print("=" * 60)

    known_relations = [
        ("8 * 0.0527", 8 * 0.0527, "π - e", np.pi - np.e),
        ("0.0527", 0.0527, "(e - π/2)/δ²", (np.e - np.pi / 2) / (4.669201609 ** 2)),
        ("Y", constants['Y'], "(π - e)/δ²", (np.pi - np.e) / (4.669201609 ** 2)),
        ("0.0527 + Y", 0.0527 + constants['Y'], "π/(2δ²)", np.pi / (2 * (4.669201609 ** 2))),
        ("327.85", 327.85, "3πδe²", 3 * np.pi * 4.669201609 * (np.e ** 2)),
        ("X", constants['X'], "e√(24πδ)", np.e * np.sqrt(24 * np.pi * 4.669201609)),
    ]

    for name1, val1, name2, val2 in known_relations:
        try:
            if abs(val2) > 1e-15:
                error = abs(val1 / val2 - 1) * 100
                if error < 0.1:  # 0.1% вместо 0.5%
                    status = "✓ ПРОШЛО"
                else:
                    status = "✗ НЕ ПРОШЛО"
                print(f"{status} {name1:20s} ≈ {name2:30s} | погрешность: {error:.6f}%")
            else:
                print(f"✗ {name1:20s} ≈ {name2:30s} | значение слишком мало")
        except:
            print(f"✗ {name1:20s} ≈ {name2:30s} | ошибка вычисления")


if __name__ == "__main__":
    main()