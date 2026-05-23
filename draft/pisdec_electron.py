"""
ПОИСК ФОРМУЛЫ ДЛЯ МАССЫ ЭЛЕКТРОНА ИЗ ПАРАМЕТРОВ ГИП
m_e = m_P × F, где F — формула из 1-4 компонентов
Включает поиск постоянной тонкой структуры α
"""

import math
import itertools

# =========================================================
# 1. БАЗОВЫЕ ПАРАМЕТРЫ ГИП
# =========================================================

K = 6.0
N = 4.1790e121

# Вычисляем p из согласованной формулы
lnK = math.log(K)
lnN = math.log(N)
pi = math.pi

p_fix = p = 4.802764507914655e-42#K ** 3 * lnK / (2 * (pi ** 3)) * N ** (-1 / 3)

# Производные величины
Kp = K * p_fix
lnKp = math.log(Kp)
lnKp_abs = abs(lnKp)

print("=" * 80)
print("ПОИСК ФОРМУЛЫ ДЛЯ МАССЫ ЭЛЕКТРОНА")
print("=" * 80)

print(f"\n📊 ПАРАМЕТРЫ ГИП:")
print(f"  K = {K}")
print(f"  p = {p_fix:.6e}")
print(f"  N = {N:.6e}")
print(f"  lnN = {lnN:.6f}")
print(f"  lnK = {lnK:.6f}")

# =========================================================
# 2. ФУНДАМЕНТАЛЬНЫЕ КОНСТАНТЫ (ЦЕЛЕВЫЕ ЗНАЧЕНИЯ)
# =========================================================

# Планковская масса
m_P = 2.176434e-8  # кг

# Масса электрона
m_e = 9.1093837015e-31  # кг

# Отношение m_e / m_P
ratio_me_mp = m_e / m_P  # ≈ 4.185 × 10⁻²³

# Постоянная тонкой структуры
alpha = 1 / 137.035999084  # ≈ 0.00729735

# Другие безразмерные отношения
targets = {
    'm_e': m_e,
    'm_e / m_P': ratio_me_mp,
    'm_P': m_P,
    'α (пост. тонкой структуры)': alpha,
    #'α²': alpha ** 2,
    #'α³': alpha ** 3,
    #'√α': math.sqrt(alpha),
    #'1/α': 1 / alpha,
    #'2πα': 2 * pi * alpha,
    #'α/(2π)': alpha / (2 * pi),
}

print(f"\n🎯 ЦЕЛЕВЫЕ ЗНАЧЕНИЯ:")
print(f"  m_P = {m_P:.6e} кг")
print(f"  m_e = {m_e:.6e} кг")
print(f"  m_e / m_P = {ratio_me_mp:.6e}")
print(f"  α = {alpha:.10f}")
print(f"  α² = {alpha ** 2:.10f}")
print(f"  α³ = {alpha ** 3:.6e}")

# =========================================================
# 3. БАЗОВЫЕ КОМПОНЕНТЫ (РАСШИРЕННЫЙ НАБОР)
# =========================================================

components = {
    # Основные параметры
    'K': K,
    'p': p_fix,
    'N': N,
    'Kp': Kp,

    # Логарифмы
    'lnK': lnK,
    'lnN': lnN,
    'lnKp': lnKp,
    'lnKp_abs': lnKp_abs,

    # Степени lnN
    '(lnN)^2': lnN ** 2,
    '(lnN)^3': lnN ** 3,
    '(lnN)^4': lnN ** 4,
    '(lnN)^5': lnN ** 5,
    '(lnN)^6': lnN ** 6,
    '(lnN)^7': lnN ** 7,
    '(lnN)^8': lnN ** 8,
    '(lnN)^9': lnN ** 9,
    '(lnN)^10': lnN ** 10,
    '(lnN)^11': lnN ** 11,
    '(lnN)^12': lnN ** 12,

    '1/(lnN)^2': 1 / (lnN ** 2),
    '1/(lnN)^3': 1 / (lnN ** 3),
    '1/(lnN)^4': 1 / (lnN ** 4),
    '1/(lnN)^5': 1 / (lnN ** 5),
    '1/(lnN)^6': 1 / (lnN ** 6),
    '1/(lnN)^7': 1 / (lnN ** 7),
    '1/(lnN)^8': 1 / (lnN ** 8),
    '1/(lnN)^9': 1 / (lnN ** 9),
    '1/(lnN)^10': 1 / (lnN ** 10),
    '1/(lnN)^11': 1 / (lnN ** 12),
    '1/(lnN)^12': 1 / (lnN ** 11),

    # Степени lnK
    '(lnK)^2': lnK ** 2,
    '(lnK)^3': lnK ** 3,
    '(lnK)^4': lnK ** 4,
    '(lnK)^5': lnK ** 5,
    '(lnK)^6': lnK ** 6,
    '1/lnK': 1 / lnK,
    '1/(lnK)^2': 1 / (lnK ** 2),
    '1/(lnK)^3': 1 / (lnK ** 3),
    '1/(lnK)^4': 1 / (lnK ** 4),
    '1/(lnK)^5': 1 / (lnK ** 5),
    '1/(lnK)^6': 1 / (lnK ** 6),

    # Степени N
    'N^(-1/4)': N ** (-1 / 4),
    'N^(1/4)': N ** (1 / 4),
    'N^(1/3)': N ** (1 / 3),
    'N^(-1/3)': N ** (-1 / 3),
    'N^(1/2)': N ** (1 / 2),
    'N^(-1/2)': N ** (-1 / 2),
    'N^(2/3)': N ** (2 / 3),
    'N^(-2/3)': N ** (-2 / 3),
    'N^(1/6)': N ** (1 / 6),
    'N^(-1/6)': N ** (-1 / 6),
    '1/N': 1 / N,

    # Степени K
    'K^2': K ** 2,
    'K^3': K ** 3,
    'K^4': K ** 4,
    'K^5': K ** 5,
    'K^6': K ** 6,
    '1/K': 1 / K,
    '1/K^2': 1 / (K ** 2),
    '1/K^3': 1 / (K ** 3),
    '1/K^4': 1 / (K ** 4),
    '√K': math.sqrt(K),
    '1/√K': 1 / math.sqrt(K),

    # Степени p
    'p^2': p_fix ** 2,
    'p^3': p_fix ** 3,
    '1/p': 1 / p_fix,
    '1/p^2': 1 / (p_fix ** 2),
    '1/p^3': 1 / (p_fix ** 3),
    '√p': math.sqrt(p_fix),

    # Математические константы
    '√2': math.sqrt(2),
    '√3': math.sqrt(3),
    '√5': math.sqrt(5),
    '√6': math.sqrt(6),
    '√7': math.sqrt(7),
    'π': pi,
    '2π': 2 * pi,
    '3π': 3 * pi,
    '4π': 4 * pi,
    '1/π': 1 / pi,
    '1/(2π)': 1 / (2 * pi),
    '1/(3π)': 1 / (3 * pi),
    '1/(4π)': 1 / (4 * pi),
    'π²': pi ** 2,
    'π³': pi ** 3,
    'π⁴': pi ** 4,
    '1/π²': 1 / (pi ** 2),
    '1/π³': 1 / (pi ** 3),
    '√π': math.sqrt(pi),
    '1/√π': 1 / math.sqrt(pi),

    'e': math.e,
    'e²': math.e ** 2,
    '1/e': 1 / math.e,
    '1/e²': 1 / (math.e ** 2),
    'e^π': math.exp(pi),
    '1/e^π': 1 / math.exp(pi),

    # Комбинации с π
    '4π²': 4 * pi ** 2,
    '1/(4π²)': 1 / (4 * pi ** 2),
    '2π²': 2 * pi ** 2,
    '1/(2π²)': 1 / (2 * pi ** 2),
    '8π²': 8 * pi ** 2,
    '1/(8π²)': 1 / (8 * pi ** 2),
    '√(2π)': math.sqrt(2 * pi),
    '1/√(2π)': 1 / math.sqrt(2 * pi),

    # Специальные комбинации
    # 'N^-1/π': N ** (-1 / pi),
    # 'N^-1/2π': N ** (-1 / (2 * pi)),
    # 'N^-1/3π': N ** (-1 / (3 * pi)),
    # #'lnN/π': lnN / pi,
    # 'lnN/(2π)': lnN / (2 * pi),
    # 'lnN/π²': lnN / (pi ** 2),

    # Комбинации логарифмов
    'lnN/lnK': lnN / lnK,
    'lnK/lnN': lnK / lnN,
    '(lnN)/(lnK)^2': lnN / (lnK ** 2),
    '(lnN)^2/lnK': (lnN ** 2) / lnK,

    # Планковские отношения (для проверки)
    '√(ħG/c³)': 1.616255e-35,  # l_P
    '√(ħG/c⁵)': 5.391247e-44,  # t_P
}


# =========================================================
# 4. ГЕНЕРАЦИЯ ФОРМУЛ
# =========================================================

def generate_formulas(components):
    """Генерирует формулы вида (A * B * C) / D и другие комбинации"""
    formulas = {}

    # Расширенный список кандидатов
    candidates = [
        # Степени lnN
        '(lnN)^2', '(lnN)^3', '(lnN)^4', '(lnN)^5', '(lnN)^6', '(lnN)^7', '(lnN)^8',
        '1/(lnN)^2', '1/(lnN)^3', '1/(lnN)^4',

        # Степени lnK
        'lnK', '(lnK)^2', '(lnK)^3', '1/lnK', '1/(lnK)^2',

        # Степени N
        'N^(-1/3)', 'N^(-1/4)', 'N^(-1/6)', 'N^(1/3)', 'N^(1/4)',
        'N^(-2/3)', 'N^(2/3)', '1/N',

        # Степени K
        'K', '1/K', 'K^2', '1/K^2', 'K^3', '1/K^3', 'K^4', '1/K^4', '√K', '1/√K',

        # Степени p
        'p', '1/p', 'p^2', '1/p^2', '√p',

        # π и комбинации
        'π', '2π', '3π', '4π', '1/π', '1/(2π)', '1/(3π)', '1/(4π)',
        'π²', 'π³', 'π⁴', '1/π²', '1/π³', '√π', '1/√π',
        '4π²', '1/(4π²)', '2π²', '1/(2π²)', '8π²', '1/(8π²)',

        # e и комбинации
        'e', 'e²', '1/e', '1/e²', 'e^π', '1/e^π',

        # Корни
        '√2', '√3', '√5', '√6', '√7',

        # Отношения логарифмов
        'lnN/lnK', 'lnK/lnN', '(lnN)/(lnK)^2', '(lnN)^2/lnK',
        'lnN/π', 'lnN/(2π)',

        # Специальные
        'N^-1/π', 'N^-1/2π', 'N^-1/3π',
    ]

    available = [c for c in candidates if c in components]
    print(f"\n🔧 Генерация формул из {len(available)} компонентов...")

    # 1. Прямые компоненты
    for c in available:
        formulas[c] = components[c]

    # 2. Произведения 2 компонентов
    for a, b in itertools.combinations(available, 2):
        val = components[a] * components[b]
        if 1e-100 < abs(val) < 1e100:
            formulas[f'{a} * {b}'] = val

    # 3. Отношения 2 компонентов
    for a, b in itertools.permutations(available, 2):
        if components[b] != 0:
            val = components[a] / components[b]
            if 1e-100 < abs(val) < 1e100:
                formulas[f'{a} / {b}'] = val

    # 4. Произведения 3 компонентов
    for a, b, c in itertools.combinations(available, 3):
        val = components[a] * components[b] * components[c]
        if 1e-100 < abs(val) < 1e100:
            formulas[f'{a} * {b} * {c}'] = val

    # 5. Комбинации вида A * B / C
    for a, b, c in itertools.permutations(available, 3):
        if components[c] != 0:
            val = components[a] * components[b] / components[c]
            if 1e-100 < abs(val) < 1e100:
                formulas[f'{a} * {b} / {c}'] = val

    # 6. Комбинации вида (A * B * C) / D
    for a, b, c in itertools.combinations(available, 3):
        for d in available:
            if d in (a, b, c):
                continue
            if components[d] != 0:
                val = components[a] * components[b] * components[c] / components[d]
                if 1e-100 < abs(val) < 1e100:
                    formulas[f'({a} * {b} * {c}) / {d}'] = val

    # 7. Комбинации вида (A * B) / (C * D)
    for a, b in itertools.combinations(available, 2):
        for c, d in itertools.combinations(available, 2):
            if components[c] != 0 and components[d] != 0:
                val = (components[a] * components[b]) / (components[c] * components[d])
                if 1e-100 < abs(val) < 1e100:
                    formulas[f'({a} * {b}) / ({c} * {d})'] = val

    print(f"   Сгенерировано {len(formulas)} формул")
    return formulas


# =========================================================
# 5. ПОИСК СОВПАДЕНИЙ
# =========================================================

def find_matches(formulas, targets, tolerance=0.10):
    """Ищет совпадения с целевыми константами"""
    matches = []

    for formula, value in formulas.items():
        for target_name, target_value in targets.items():
            if target_value != 0:
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


# 6. ГЛАВНАЯ ФУНКЦИЯ

def main():
    # Генерируем формулы
    formulas = generate_formulas(components)

    # Ищем совпадения
    print("\n🔍 ПОИСК СОВПАДЕНИЙ...")
    matches = find_matches(formulas, targets, tolerance=0.10)
    print(f"   Найдено: {len(matches)}")

    # Группируем по целям
    by_target = {}
    for m in matches:
        target = m['target']
        if target not in by_target:
            by_target[target] = []
        by_target[target].append(m)

    # Выводим результаты по каждой цели
    for target in targets:
        if target in by_target:
            print(f"🎯 ЦЕЛЬ: {target} = {targets[target]:.6e}")

            target_matches = by_target[target][:80]  # топ-20 для каждой цели

            for i, m in enumerate(target_matches, 1):
                marker = " ⭐⭐⭐" if m['rel_percent'] < 0.1 else (" ⭐⭐" if m['rel_percent'] < 1 else "")
                print(f"\n{i:2d}. {m['formula']}{marker}")
                print(f"    Значение: {m['value']:.12e}")
                print(f"    Отн. ошибка: {m['rel_percent']:.6f}%")

                # Вычисляем массу электрона для m_e / m_P
                if target == 'm_e / m_P':
                    m_e_pred = m['value'] * m_P
                    print(f"    → m_e (предсказано): {m_e_pred:.6e} кг")
                    print(f"    → m_e (реальное):    {m_e:.6e} кг")
                elif target == 'α (пост. тонкой структуры)':
                    print(f"    → 1/α (предсказано): {1 / m['value']:.6f}")
                    print(f"    → 1/α (реальное):    137.036")

    # Сводная таблица лучших для m_e / m_P
    print("🏆 ЛУЧШИЕ ФОРМУЛЫ ДЛЯ m_e / m_P")

    if 'm_e / m_P' in by_target:
        best_me = by_target['m_e / m_P'][:10]
        print(f"\n{'Формула':<50} {'m_e/m_P':<15} {'Ошибка %':<10} {'m_e (кг)':<15}")
        for m in best_me:
            formula_short = m['formula'][:47] + "..." if len(m['formula']) > 50 else m['formula']
            print(f"{formula_short:<50} {m['value']:.6e}  {m['rel_percent']:>8.4f}%  {m['value'] * m_P:.6e}")

    # Сводная таблица для α
    print("🏆 ЛУЧШИЕ ФОРМУЛЫ ДЛЯ ПОСТОЯННОЙ ТОНКОЙ СТРУКТУРЫ α")

    if 'α (пост. тонкой структуры)' in by_target:
        best_alpha = by_target['α (пост. тонкой структуры)'][:10]
        print(f"\n{'Формула':<50} {'α':<15} {'Ошибка %':<10} {'1/α':<10}")
        for m in best_alpha:
            formula_short = m['formula'][:47] + "..." if len(m['formula']) > 50 else m['formula']
            print(f"{formula_short:<50} {m['value']:.10f}  {m['rel_percent']:>8.4f}%  {1 / m['value']:.2f}")

    return matches


if __name__ == "__main__":
    matches = main()