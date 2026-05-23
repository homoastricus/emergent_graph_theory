import itertools

import math

K = 6.0

lnK = math.log(K)
lnN = 280.111514
N = math.exp(lnN)
targets = {
    #'tail': math.pi - 3,
    #'1/tail': 1 / (math.pi - 3),
#    'feigenbaum_delta': 4.669201609102990,
#    'feigenbaum_alpha': 2.5029078750958928,
#'1/feigenbaum_delta': 1/4.669201609102990,
    'euler': 2.718281828459045,
#'1/euler': 1/2.718281828459045,
    'euler_mascheroni': 0.57721566490153286060651209008240243104215933593992,
    #'ln3': math.log(3),
    #'ln2': math.log(2),
    'pi_inv': 1 / math.pi,
    'pi^2':  math.pi**2
}

components_base = {
    #'1': 1,
    'K': K,
    # 'N': N,
    'lnK': lnK,
    'lnlnK': math.log(lnK),
    '1/lnlnK': 1/math.log(lnK),
'1/lnlnN': 1/math.log(lnN),
'lnlnN': math.log(lnN),
    'ln2': math.log(2),
    'ln3': math.log(3),
    '1/ln2': 1 / math.log(2),
    '1/ln3': 1 / math.log(3),
    'lnN': lnN,
    #'(lnN)^3': lnN ** 3,
    # '(lnN)^6': lnN ** 6,
    '1/lnN': 1 / lnN,
    '1/lnN^2': 1 / lnN ** 2,
    #'1/lnN^3': 1 / lnN ** 3,
    # '1/(lnN)^3': 1 / (lnN ** 3),
    # '1/(lnN)^6': 1 / (lnN ** 6),
    '1/2': 1 / 2,
    '1/3': 1 / 3,
    '2': 2.0,
    # 'N^(1/3)': N ** (1 / 3),
    'N^(-1/3)': N ** (-1 / 3),
    # '(lnK)^2': lnK ** 2,
    # '1/(lnK)^2': 1 / lnK ** 2,
    '1/K': 1 / K,
    '1/K^2': 1 / (K ** 2),
    # '1/K^3': 1 / (K ** 3),
    # '√2': math.sqrt(2),
    # '√3': math.sqrt(3),
    # '√K': math.sqrt(K),
    # '1/√K': 1 / math.sqrt(K),
}


def safe_val(val):
    """Проверяет, что значение не слишком большое/маленькое и не NaN"""
    try:
        if isinstance(val, complex):
            return False
        if not math.isfinite(val):
            return False
        if abs(val) > 1e150 or (abs(val) < 1e-150 and abs(val) > 0):
            return False
        return True
    except:
        return False


def generate_expressions(components, max_depth=2):
    """
    Генерирует составные выражения с +, -, *, /
    max_depth: глубина вложенности
    """
    available = list(components.keys())
    print(f"\n🔧 Генерация выражений из {len(available)} компонентов")

    expressions = {}

    # Уровень 0: базовые компоненты
    for name, val in components.items():
        if safe_val(val):
            expressions[name] = val

    ops = {
        '+': lambda a, b: a + b,
        '-': lambda a, b: a - b,
        '*': lambda a, b: a * b,
        '/': lambda a, b: a / b if abs(b) > 1e-100 else None,
    }

    # Уровень 1: все бинарные операции над базовыми компонентами
    print("   Уровень 1: бинарные операции над базовыми...")
    level1 = {}

    for a_name, b_name in itertools.permutations(available, 2):
        a, b = components[a_name], components[b_name]
        for op_name, op_func in ops.items():
            try:
                result = op_func(a, b)
                if result is not None and safe_val(result):
                    name = f'({a_name} {op_name} {b_name})'
                    level1[name] = result
            except:
                pass

    # Унарный минус для уровня 1
    for name, val in list(level1.items()):
        if safe_val(-val):
            level1[f'-{name}'] = -val

    expressions.update(level1)
    print(f"   Уровень 1 завершён: {len(level1)} шт.")

    if max_depth >= 2:
        print("   Уровень 2: вложенные выражения...")
        level2 = {}

        # Собираем операнды: базовые + уровень 1
        all_operands = {}
        all_operands.update(components)
        all_operands.update(level1)

        # Ограничиваем для скорости
        sample_base = available
        sample_level1 = list(level1.keys())[:100000]

        # === ТИП 1: (A op B) op C - тройные комбинации ===
        print("      Тип 1: (A op B) op C ...")
        for ab_name in sample_level1:
            for c_name in sample_base[:100000]:
                if c_name in ab_name:  # избегаем дублирования
                    continue
                ab_val = all_operands[ab_name]
                c_val = components[c_name]

                for op_name, op_func in ops.items():
                    try:
                        result = op_func(ab_val, c_val)
                        if result is not None and safe_val(result):
                            name = f'({ab_name} {op_name} {c_name})'
                            if name not in expressions:
                                level2[name] = result
                    except:
                        pass

        # === ТИП 2: A op (B op C) - меняем порядок ===
        print("      Тип 2: A op (B op C) ...")
        for a_name in sample_base[:1000000]:
            for bc_name in sample_level1:
                if a_name in bc_name:
                    continue
                a_val = components[a_name]
                bc_val = all_operands[bc_name]

                # A / (B op C) - деление на составное выражение
                for op_name in ['/', '*', '+', '-']:
                    try:
                        if op_name == '/' and abs(bc_val) < 1e-100:
                            continue
                        result = ops[op_name](a_val, bc_val)
                        if result is not None and safe_val(result):
                            name = f'({a_name} {op_name} {bc_name})'
                            if name not in expressions:
                                level2[name] = result
                    except:
                        pass

        # === ТИП 3: (A op B) op (C op D) - два составных выражения ===
        print("      Тип 3: (A op B) op (C op D) ...")
        for ab_name in sample_level1[:1000000]:
            for cd_name in sample_level1[:1000000]:
                if ab_name == cd_name:
                    continue
                ab_val = all_operands[ab_name]
                cd_val = all_operands[cd_name]

                for op_name, op_func in ops.items():
                    try:
                        if op_name == '/' and abs(cd_val) < 1e-100:
                            continue
                        result = op_func(ab_val, cd_val)
                        if result is not None and safe_val(result):
                            name = f'({ab_name} {op_name} {cd_name})'
                            if name not in expressions:
                                level2[name] = result
                    except:
                        pass

        # === ТИП 4: A - B/(C - D) и подобные ===
        print("      Тип 4: A op (B / (C op D)) ...")
        # Исправлено: используем itertools.islice вместо слайса
        sample_permutations = list(itertools.islice(
            itertools.permutations(sample_base, 3), 5000
        ))

        for a_name in sample_base[:1000000]:
            for b_name, c_name, d_name in sample_permutations:
                # Пропускаем если a совпадает с одним из b,c,d
                if a_name in (b_name, c_name, d_name):
                    continue

                a_val = components[a_name]
                b_val = components[b_name]
                c_val = components[c_name]
                d_val = components[d_name]

                # Формируем (C op D)
                for op1_name, op1_func in [('+', ops['+']), ('-', ops['-'])]:
                    try:
                        cd_val = op1_func(c_val, d_val)
                        if not safe_val(cd_val) or abs(cd_val) < 1e-100:
                            continue

                        # B / (C op D)
                        b_div_cd = b_val / cd_val
                        if not safe_val(b_div_cd):
                            continue

                        cd_name = f'({c_name} {op1_name} {d_name})'
                        b_div_name = f'({b_name} / {cd_name})'

                        # Сохраняем промежуточный результат
                        if b_div_name not in expressions:
                            level2[b_div_name] = b_div_cd

                        # A - B/(C-D) и A + B/(C-D)
                        for op2_name, op2_func in [('-', ops['-']), ('+', ops['+'])]:
                            result = op2_func(a_val, b_div_cd)
                            if safe_val(result):
                                name = f'({a_name} {op2_name} {b_div_name})'
                                if name not in expressions:
                                    level2[name] = result

                        # A * B/(C-D) и A / (B/(C-D))
                        for op2_name, op2_func in [('*', ops['*']), ('/', ops['/'])]:
                            try:
                                if op2_name == '/' and abs(b_div_cd) < 1e-100:
                                    continue
                                result = op2_func(a_val, b_div_cd)
                                if safe_val(result):
                                    name = f'({a_name} {op2_name} {b_div_name})'
                                    if name not in expressions:
                                        level2[name] = result
                            except:
                                pass
                    except:
                        pass

        print(f"   Уровень 2 завершён: {len(level2)} шт.")
        expressions.update(level2)

    if max_depth >= 3:
        print("   Уровень 3: глубокие вложенные выражения...")
        level3 = {}

        sample_level2 = list(level2.keys())[:1000000]

        # (L2) op (Base)
        for l2_name in sample_level2:
            for base_name in available[:15]:
                if base_name in l2_name:
                    continue
                l2_val = expressions[l2_name]
                base_val = components[base_name]

                for op_name, op_func in ops.items():
                    try:
                        if op_name == '/' and abs(base_val) < 1e-100:
                            continue
                        result = op_func(l2_val, base_val)
                        if safe_val(result):
                            name = f'({l2_name} {op_name} {base_name})'
                            if name not in expressions:
                                level3[name] = result
                    except:
                        pass

        # Base / (L2) - деление на вложенное
        for base_name in available[:1000000]:
            for l2_name in sample_level2:
                if base_name in l2_name:
                    continue
                l2_val = expressions[l2_name]
                base_val = components[base_name]

                if abs(l2_val) < 1e-100:
                    continue
                result = base_val / l2_val
                if safe_val(result):
                    name = f'({base_name} / {l2_name})'
                    if name not in expressions:
                        level3[name] = result

        print(f"   Уровень 3 завершён: {len(level3)} шт.")
        expressions.update(level3)

    print(f"   ✅ Всего сгенерировано: {len(expressions)} выражений")
    return expressions


def find_matches(formulas, targets, tolerance=0.05):
    """Ищет совпадения с целевыми константами"""
    matches = []

    for formula, value in formulas.items():
        for target_name, target_value in targets.items():
            if abs(target_value) < 1e-100:
                continue
            rel_error = abs(value - target_value) / abs(target_value)
            if rel_error < tolerance:
                matches.append({
                    'formula': formula,
                    'value': value,
                    'target': target_name,
                    'target_value': target_value,
                    'rel_error': rel_error,
                    'rel_percent': rel_error * 100
                })

    matches.sort(key=lambda x: x['rel_error'])
    return matches


def main():
    print("НУМЕРОЛОГИЧЕСКИЙ ПОИСК (РАСШИРЕННЫЙ)")
    print(f"  K = {K}")
    print(f"  N = {N:.4e}")
    print(f"  lnN = {lnN:.6f}")
    print(f"  Цель: π - 3 = {math.pi - 3:.12e}")
    expressions = generate_expressions(components_base, max_depth=3)

    print("\n🔍 ПОИСК СОВПАДЕНИЙ...")
    matches = find_matches(expressions, targets, tolerance=0.005)
    print(f"   Найдено: {len(matches)}")

    print("ТОП СОВПАДЕНИЙ")

    for i, m in enumerate(matches[:100], 1):
        if m['rel_percent'] < 0.01:
            marker = " ⭐⭐⭐"
        elif m['rel_percent'] < 0.1:
            marker = " ⭐⭐"
        elif m['rel_percent'] < 1:
            marker = " ⭐"
        else:
            marker = ""

        print(f"\n{i:3d}. {m['formula']}{marker}")
        print(f"     Значение: {m['value']:.12e}")
        print(f"     Цель: {m['target']} = {m['target_value']:.12e}")
        print(f"     Ошибка: {m['rel_percent']:.6f}%")

    print("СТАТИСТИКА")
    print(f"  Всего выражений: {len(expressions)}")
    print(f"  Совпадений (<5%): {len(matches)}")
    print(f"  <0.1%: {sum(1 for m in matches if m['rel_percent'] < 0.1)}")
    print(f"  <0.01%: {sum(1 for m in matches if m['rel_percent'] < 0.01)}")

    return matches


if __name__ == "__main__":
    matches = main()
