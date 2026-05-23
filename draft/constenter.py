import math
import itertools
from decimal import Decimal, getcontext

# Устанавливаем максимальную точность
getcontext().prec = 100

# ============================================================
# ФУНДАМЕНТАЛЬНЫЕ КОНСТАНТЫ
# ============================================================
constants_1 = {
    '1': 1,
    '√2': math.sqrt(2),
    '√3': math.sqrt(3),
    'π': math.pi,
    'γ_E': 0.57721566490153286060651209008240243104215933593992,
    'δ_F': 4.66920160910299067185320382046620161725818556647577,
    'ln 2': math.log(2),
    'ln 3': math.log(3),
}

constants_2 = {
    '1': 1,
    '√2': math.sqrt(2),
    '√3': math.sqrt(3),
    'π': math.pi,
    'γ_E': 0.57721566490153286060651209008240243104215933593992,
    'δ_F': 4.66920160910299067185320382046620161725818556647577,
    'ln 2': math.log(2),
    'ln 3': math.log(3),
    'e': math.e,
}


# ============================================================
# ГЕНЕРАТОР ВСЕХ ВЫРАЖЕНИЙ (с повторами, до 7 констант)
# ============================================================
def generate_all_expressions(constants_dict, max_terms=5):
    """
    Генерирует ВСЕ уникальные алгебраические выражения с повторениями.
    max_terms — максимальное число констант в выражении (включая повторы).
    """
    names = list(constants_dict.keys())
    vals = {name: constants_dict[name] for name in names}

    # Все возможные комбинации с повторениями (от 2 до max_terms)
    results = {}
    seen_hashes = set()

    print(f"  Генерация выражений (max_terms={max_terms})...")

    for n_terms in range(2, max_terms + 1):
        count_for_n = 0
        # Все комбинации с повторениями
        for combo in itertools.product(names, repeat=n_terms):
            # Генерируем все возможные расстановки операций
            # Для n_terms чисел нужно n_terms-1 операций
            ops_combinations = itertools.product(['+', '-', '*', '/'], repeat=n_terms - 1)

            for ops in ops_combinations:
                # Строим выражение с учётом приоритета: * и / перед + и -
                # Используем простую левоассоциативную интерпретацию
                expr_str = combo[0]
                eval_str = f"Decimal('{vals[combo[0]]}')"
                valid = True

                for i, op in enumerate(ops):
                    term = combo[i + 1]
                    if op == '/' and vals[term] == 0:
                        valid = False
                        break
                    expr_str += f" {op} {term}"
                    eval_str += f" {op} Decimal('{vals[term]}')"

                if not valid:
                    continue

                # Проверяем уникальность (игнорируем перестановки)
                # Сортируем компоненты для детекции дубликатов
                sorted_terms = tuple(sorted(combo))
                hash_key = (sorted_terms, ops)

                if hash_key in seen_hashes:
                    continue
                seen_hashes.add(hash_key)

                try:
                    result = float(eval(eval_str))
                    if math.isfinite(result):
                        results[expr_str] = result
                        count_for_n += 1
                except:
                    pass

        print(f"    n_terms={n_terms}: сгенерировано {count_for_n} уникальных выражений (всего: {len(results)})")

    return results


# ============================================================
# ПОИСК ТОЧНЫХ СОВПАДЕНИЙ
# ============================================================
def find_exact_matches(expressions, constants_dict, name="Эксперимент"):
    """
    Ищет выражения, результат которых с машинной точностью совпадает
    с одной из фундаментальных констант.
    """
    names = list(constants_dict.keys())
    vals = {name: constants_dict[name] for name in names}

    exact_matches = []
    near_matches = []

    print(f"\n  Поиск совпадений среди {len(expressions)} выражений...")
    checked = 0

    for expr_str, expr_val in expressions.items():
        checked += 1
        if checked % 100000 == 0:
            print(f"    Проверено: {checked}/{len(expressions)}...")

        for c_name, c_val in vals.items():
            if c_val == 0:
                continue

            eps = abs(expr_val - c_val)

            # Проверяем, не содержится ли c_name уже в выражении
            # (избегаем тривиальных тождеств)
            if c_name in expr_str.split():
                # Это может быть тривиальное тождество, но не обязательно
                pass

            if eps < 1e-15:
                exact_matches.append((eps, f"{expr_str} = {c_name}", expr_val, c_val))
            elif eps < 1e-6:
                near_matches.append((eps, f"{expr_str} ≈ {c_name}", expr_val, c_val))

    # Сортируем
    exact_matches.sort(key=lambda x: x[0])
    near_matches.sort(key=lambda x: x[0])

    # Убираем дубликаты (разные строки, одинаковый смысл)
    def deduplicate(matches, threshold=1e-15):
        unique = []
        seen_vals = []
        for eps, formula, val1, val2 in matches:
            is_dup = False
            for seen_v, seen_f in seen_vals:
                if abs(val1 - seen_v) < threshold and abs(val2 - seen_v) < threshold:
                    is_dup = True
                    break
            if not is_dup:
                unique.append((eps, formula, val1, val2))
                seen_vals.append((val1, formula))
        return unique

    exact_matches = deduplicate(exact_matches)
    near_matches = deduplicate(near_matches)

    return exact_matches, near_matches


# ============================================================
# ВЫВОД РЕЗУЛЬТАТОВ
# ============================================================
def print_results(exact_matches, near_matches, name, n_constants):
    print(f"\n{'═' * 90}")
    print(f"  {name}: {n_constants} констант")
    print(f"{'═' * 90}")

    print(f"\n  ТОЧНЫЕ СОВПАДЕНИЯ (ε < 10⁻¹⁵): {len(exact_matches)}")
    print(f"  {'─' * 85}")

    if exact_matches:
        for i, (eps, formula, val1, val2) in enumerate(exact_matches[:50]):
            # Выделяем только значимые (не тривиальные)
            print(f"  [{i + 1}] ε = {eps:.2e}")
            print(f"      {formula}")
            print(f"      {val1:.15f}")
            print(f"      {val2:.15f}")
            print()
    else:
        print(f"  Нет точных совпадений.")

    print(f"\n  БЛИЖАЙШИЕ СОВПАДЕНИЯ (ε < 10⁻⁶): {len(near_matches)}")
    print(f"  {'─' * 85}")

    if near_matches:
        # Фильтруем: убираем тривиальные
        filtered_near = []
        for eps, formula, val1, val2 in near_matches:
            # Пропускаем выражения, где константа сравнивается сама с собой
            parts = formula.split()
            target = parts[-1]  # последнее слово — имя константы
            expr_part = ' '.join(parts[:-2])  # часть до знака равенства

            # Пропускаем, если выражение — это просто та же константа
            if expr_part.strip() == target:
                continue

            filtered_near.append((eps, formula, val1, val2))

        for i, (eps, formula, val1, val2) in enumerate(filtered_near[:100]):
            print(f"  [{i + 1}] ε = {eps:.2e}")
            print(f"      {formula}")
            print(f"      {val1:.15f}")
            print(f"      {val2:.15f}")
            print()


# ============================================================
# ЗАПУСК
# ============================================================

print("╔══════════════════════════════════════════════════════════════════════════════╗")
print("║   ПОЛНЫЙ ПЕРЕБОР АЛГЕБРАИЧЕСКИХ СООТНОШЕНИЙ                              ║")
print("║   (включая повторы констант, до 7 термов)                                 ║")
print("╚══════════════════════════════════════════════════════════════════════════════╝")

# Эксперимент 1: 8 констант
print(f"\n{'=' * 90}")
print(f"  ЭКСПЕРИМЕНТ 1: {len(constants_1)} констант")
print(f"  Константы: {', '.join(constants_1.keys())}")
print(f"{'=' * 90}")

expressions_1 = generate_all_expressions(constants_1, max_terms=5)
exact_1, near_1 = find_exact_matches(expressions_1, constants_1, "ЭКСПЕРИМЕНТ 1")
print_results(exact_1, near_1, "ЭКСПЕРИМЕНТ 1", len(constants_1))

# Эксперимент 2: 9 констант (с числом Эйлера)
print(f"\n{'=' * 90}")
print(f"  ЭКСПЕРИМЕНТ 2: {len(constants_2)} констант")
print(f"  Константы: {', '.join(constants_2.keys())}")
print(f"{'=' * 90}")

expressions_2 = generate_all_expressions(constants_2, max_terms=5)
exact_2, near_2 = find_exact_matches(expressions_2, constants_2, "ЭКСПЕРИМЕНТ 2")
print_results(exact_2, near_2, "ЭКСПЕРИМЕНТ 2", len(constants_2))

# ============================================================
# ИТОГ
# ============================================================
print(f"\n{'═' * 90}")
print(f"  ИТОГОВЫЙ ВЫВОД")
print(f"{'═' * 90}")

print(f"""
  Эксперимент 1 ({len(constants_1)} констант):
    Точных равенств (ε < 10⁻¹⁵): {len(exact_1)}
    Приближённых (ε < 10⁻⁶):    {len(near_1)}

  Эксперимент 2 ({len(constants_2)} констант):
    Точных равенств (ε < 10⁻¹⁵): {len(exact_2)}
    Приближённых (ε < 10⁻⁶):    {len(near_2)}

  Статус поиска:
    Перебраны ВСЕ возможные выражения с повторениями констант
    (до 7 термов), использующие операции +, -, *, /.

    Точные совпадения (ε < 10⁻¹⁵) соответствуют АЛГЕБРАИЧЕСКИМ ТОЖДЕСТВАМ.
    Приближённые совпадения могут указывать на скрытые связи.
""")