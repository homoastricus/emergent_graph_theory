import re
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict, Optional, Union
import json
from dataclasses import dataclass
from math import pi, e as e_constant

# ==========================================
# КОНФИГУРАЦИЯ
# ==========================================

INPUT_FILE = "output"
OUTPUT_FILE = "formula_analysis_v2.json"


# ==========================================
# ПРАВИЛЬНОЕ ПРЕДСТАВЛЕНИЕ ФОРМУЛЫ
# ==========================================

@dataclass
class Formula:
    """
    Представление: scalar * Π component^power

    Пример: 8π² * K^2 / (lnN)^4
    scalar = 8
    powers = {'pi': 2, 'K': 2, 'lnN': -4}
    """
    scalar: float
    powers: Dict[str, float]

    def __mul__(self, other: 'Formula') -> 'Formula':
        """Умножение формул"""
        new_scalar = self.scalar * other.scalar
        new_powers = self.powers.copy()

        for comp, power in other.powers.items():
            new_powers[comp] = new_powers.get(comp, 0) + power

        # Удаляем нулевые степени
        new_powers = {k: v for k, v in new_powers.items() if abs(v) > 1e-10}

        return Formula(scalar=new_scalar, powers=new_powers)

    def __truediv__(self, other: 'Formula') -> 'Formula':
        """Деление формул"""
        new_scalar = self.scalar / other.scalar
        new_powers = self.powers.copy()

        for comp, power in other.powers.items():
            new_powers[comp] = new_powers.get(comp, 0) - power

        new_powers = {k: v for k, v in new_powers.items() if abs(v) > 1e-10}

        return Formula(scalar=new_scalar, powers=new_powers)

    def inverse(self) -> 'Formula':
        """Обратная формула"""
        new_powers = {k: -v for k, v in self.powers.items()}
        return Formula(scalar=1.0 / self.scalar, powers=new_powers)

    def is_dimensionless(self) -> bool:
        """Проверка, безразмерна ли формула (все степени = 0)"""
        return len(self.powers) == 0

    def get_vector(self, components_order: List[str]) -> np.ndarray:
        """Получение вектора степеней в заданном порядке"""
        vec = np.zeros(len(components_order))
        for i, comp in enumerate(components_order):
            if comp in self.powers:
                vec[i] = self.powers[comp]
        return vec

    def __str__(self) -> str:
        """Строковое представление"""
        if not self.powers:
            return f"{self.scalar:.6g}"

        # Числитель и знаменатель
        numerator_parts = []
        denominator_parts = []

        # Добавляем скаляр если не 1
        if abs(self.scalar - 1.0) > 1e-10:
            numerator_parts.append(f"{self.scalar:.6g}")

        for comp, power in sorted(self.powers.items()):
            if power > 0:
                if abs(power - 1) < 1e-10:
                    numerator_parts.append(comp)
                else:
                    numerator_parts.append(f"{comp}^{power:.3g}")
            else:
                abs_power = abs(power)
                if abs(abs_power - 1) < 1e-10:
                    denominator_parts.append(comp)
                else:
                    denominator_parts.append(f"{comp}^{abs_power:.3g}")

        num_str = ' · '.join(numerator_parts) if numerator_parts else '1'
        denom_str = ' · '.join(denominator_parts) if denominator_parts else '1'

        if denom_str == '1':
            return num_str
        return f"({num_str}) / ({denom_str})"


# ==========================================
# БАЗОВЫЕ НЕЗАВИСИМЫЕ КОМПОНЕНТЫ
# ==========================================

# Минимальный набор независимых переменных
INDEPENDENT_COMPONENTS = [
    'K',  # Константа связи
    'lnK',  # ln(K)
    'lnN',  # ln(N)
    'N',  # Размерность
    'p',  # p-константа
    'pi',  # π
    'e',  # число e
    'sqrt2',  # √2
    'sqrt3',  # √3
    'epi',  # e^π (отдельный базис, т.к. это трансцендентная комбинация)
]

# Порядок для векторизации
COMPONENTS_ORDER = INDEPENDENT_COMPONENTS
comp_to_idx = {comp: i for i, comp in enumerate(COMPONENTS_ORDER)}


# ==========================================
# ПРАВИЛЬНЫЙ ПАРСЕР ФОРМУЛ
# ==========================================

def parse_atomic(term: str) -> Tuple[float, Dict[str, float]]:
    """
    Парсинг атомарного терма с правильным учётом коэффициентов.

    Возвращает: (scalar, {component: power})

    Примеры:
    '8π²' -> (8, {'pi': 2})
    'K^2' -> (1, {'K': 2})
    'lnN' -> (1, {'lnN': 1})
    '√2' -> (1, {'sqrt2': 1})
    """
    term = term.strip()
    powers = {}
    scalar = 1.0

    # Извлекаем числовой коэффициент
    # Ищем числа в начале (включая дроби)
    num_match = re.match(r'^(\d+(?:\.\d+)?(?:\/\d+(?:\.\d+)?)?)\s*\*?\s*(.*)', term)
    if num_match:
        num_str = num_match.group(1)
        if '/' in num_str:
            num, den = num_str.split('/')
            scalar = float(num) / float(den)
        else:
            scalar = float(num_str)
        term = num_match.group(2).strip()

    # Если после извлечения числа ничего не осталось
    if not term:
        return scalar, powers

    # Словарь базовых компонентов и их нормализация
    base_components = {
        # K
        'K': 'K',
        # lnK
        'lnK': 'lnK',
        # lnN с разными написаниями
        'lnN': 'lnN',
        '(lnN)': 'lnN',
        # N
        'N': 'N',
        # p
        'p': 'p',
        # pi с разными написаниями
        'π': 'pi',
        'pi': 'pi',
        # e
        'e': 'e',
        # Корни
        '√2': 'sqrt2',
        '√(2)': 'sqrt2',
        'sqrt(2)': 'sqrt2',
        '√3': 'sqrt3',
        '√(3)': 'sqrt3',
        'sqrt(3)': 'sqrt3',
        # e^π
        'e^π': 'epi',
        'e^pi': 'epi',
        'exp(π)': 'epi',
    }

    # Обработка степени
    if '^' in term:
        base, power_str = term.rsplit('^', 1)
        base = base.strip()
        power_str = power_str.strip()

        # Парсинг степени
        if '/' in power_str:
            num, den = power_str.split('/')
            power = float(num) / float(den)
        else:
            try:
                power = float(power_str)
            except ValueError:
                # Может быть, это не степень а часть имени (например, e^π)
                if term in base_components:
                    powers[base_components[term]] = 1.0
                    return scalar, powers
                return scalar, {}
    else:
        base = term
        power = 1.0

    # Нормализация базы
    base = base.strip()

    # Обработка специальных конструкций
    if base.startswith('(lnN)') and '^' in term:
        # Уже обработано выше
        pass

    if base in base_components:
        comp = base_components[base]
        powers[comp] = power
    elif base == 'lnN':
        powers['lnN'] = power
    else:
        # Пробуем match без скобок
        base_clean = base.replace('(', '').replace(')', '')
        if base_clean in base_components:
            powers[base_components[base_clean]] = power

    # Обработка составных множителей внутри атома
    # Например: 8π² -> scalar=8, pi^2
    if 'π' in term and 'pi' not in str(powers):
        # Извлекаем коэффициент при π
        pi_coeff = 1.0
        pi_power = 1.0

        # Ищем 2π, 4π, 8π, 2π², 4π², 8π²
        pi_patterns = [
            (r'^2π²$', 2, 2),
            (r'^2π\^2$', 2, 2),
            (r'^4π²$', 4, 2),
            (r'^4π\^2$', 4, 2),
            (r'^8π²$', 8, 2),
            (r'^8π\^2$', 8, 2),
            (r'^2π$', 2, 1),
            (r'^4π$', 4, 1),
            (r'^8π$', 8, 1),
        ]

        for pattern, coeff, pp in pi_patterns:
            if re.match(pattern, term):
                scalar *= coeff
                powers['pi'] = pp
                break

    # Особый случай: √(2π) = √2 · √π
    if '√(2π)' in term or '√2π' in term:
        powers['sqrt2'] = 1.0
        powers['pi'] = 0.5

    # Особый случай: √(pK) - составной, разложим позже
    if '√(pK)' in term or '√pK' in term:
        # Это sqrt(p*K), но p и K у нас независимые
        # Пока пропускаем
        pass

    return scalar, powers


def parse_expression(expr: str) -> Formula:
    """
    Парсинг выражения в Formula.
    Поддерживает умножение, деление, скобки.
    """
    expr = expr.strip()

    # Убираем внешние скобки если они окружают всё выражение
    while expr.startswith('(') and expr.endswith(')'):
        depth = 0
        for i, char in enumerate(expr):
            if char == '(':
                depth += 1
            elif char == ')':
                depth -= 1
            if depth == 0 and i < len(expr) - 1:
                break
        else:
            expr = expr[1:-1].strip()
        if depth != 0:
            break

    # Находим главное деление (вне скобок)
    split_idx = -1
    depth = 0
    for i, char in enumerate(expr):
        if char == '(':
            depth += 1
        elif char == ')':
            depth -= 1
        elif char == '/' and depth == 0:
            # Проверяем что это не 1/что-то
            if i == 0 or expr[i - 1] != '1':
                split_idx = i
                break

    if split_idx > 0:
        numerator = expr[:split_idx].strip()
        denominator = expr[split_idx + 1:].strip()

        num_formula = parse_product(numerator)
        den_formula = parse_product(denominator)

        return num_formula / den_formula
    else:
        return parse_product(expr)


def parse_product(expr: str) -> Formula:
    """
    Парсинг произведения термов.
    """
    expr = expr.strip()

    # Убираем внешние скобки
    while expr.startswith('(') and expr.endswith(')'):
        expr = expr[1:-1].strip()

    # Разделяем по умножению
    terms = re.split(r'\s*\*\s*', expr)

    total_scalar = 1.0
    total_powers = {}

    for term in terms:
        term = term.strip()
        if not term:
            continue

        # Обработка 1/что-то
        if term.startswith('1/'):
            # Это отрицательная степень
            sub_expr = term[2:]
            sub_formula = parse_expression(sub_expr)
            # Инвертируем
            total_scalar /= sub_formula.scalar
            for comp, power in sub_formula.powers.items():
                total_powers[comp] = total_powers.get(comp, 0) - power
            continue

        scalar, powers = parse_atomic(term)
        total_scalar *= scalar

        for comp, power in powers.items():
            total_powers[comp] = total_powers.get(comp, 0) + power

    # Удаляем нулевые степени
    total_powers = {k: v for k, v in total_powers.items() if abs(v) > 1e-10}

    return Formula(scalar=total_scalar, powers=total_powers)


def parse_formula_v2(formula_str: str) -> Optional[Formula]:
    """
    Безопасный парсинг формулы в Formula объект.
    """
    try:
        # Предварительная очистка
        formula_str = formula_str.strip()
        # Убираем звёзды рейтинга
        formula_str = re.sub(r'[⭐]+', '', formula_str)

        return parse_expression(formula_str)
    except Exception as e:
        return None


# ==========================================
# ПАРСИНГ ВЫХОДНОГО ФАЙЛА
# ==========================================

def parse_output_file(filename: str) -> Tuple[List[str], List[str]]:
    """Парсинг файла вывода программы"""
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()

    eps0_formulas = []
    mu0_formulas = []

    lines = content.split('\n')

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        match = re.match(r'^\s*(\d+)\.\s+(.*)', line)
        if match:
            formula_text = match.group(2).strip()
            formula_text = re.sub(r'[⭐]+', '', formula_text).strip()

            # Ищем цель
            for j in range(i + 1, min(i + 5, len(lines))):
                next_line = lines[j].strip()

                if 'ε₀' in next_line or 'ε0' in next_line:
                    eps0_formulas.append(formula_text)
                    break
                elif 'μ0' in next_line or 'μ₀' in next_line:
                    mu0_formulas.append(formula_text)
                    break

                if re.match(r'^\s*\d+\.\s+', next_line):
                    break

        i += 1

    return eps0_formulas, mu0_formulas


# ==========================================
# ОПТИМИЗИРОВАННЫЙ ПОИСК С КОЭФФИЦИЕНТАМИ
# ==========================================

class FormulaSearchV2:
    def __init__(self, scalar_tolerance: float = 1e-6):
        self.scalar_tolerance = scalar_tolerance

    def search_combinations(self, eps0_formulas: List[str],
                            mu0_formulas: List[str]) -> List[Dict]:
        """
        Поиск комбинаций где ε₀ · μ₀ · test = 1
        с учётом КОЭФФИЦИЕНТОВ и степеней.
        """
        print("Парсинг формул ε₀...")
        eps0_parsed = []
        for i, f in enumerate(eps0_formulas):
            formula = parse_formula_v2(f)
            if formula is not None:
                eps0_parsed.append((i, formula))

        print("Парсинг формул μ₀...")
        mu0_parsed = []
        for i, f in enumerate(mu0_formulas):
            formula = parse_formula_v2(f)
            if formula is not None:
                mu0_parsed.append((i, formula))

        print(f"Успешно разобрано: {len(eps0_parsed)} ε₀, {len(mu0_parsed)} μ₀")

        print("Поиск совпадений с учётом коэффициентов...")
        results = []
        checked = 0

        for i_eps, eps_formula in eps0_parsed:
            for i_mu, mu_formula in mu0_parsed:
                # Вычисляем произведение ε₀ · μ₀
                product = eps_formula * mu_formula

                # Тестовая формула должна быть обратной к произведению
                # ε₀ · μ₀ · test = 1 → test = 1/(ε₀ · μ₀)
                test_formula = product.inverse()

                # Проверяем, что тестовая формула "разумная"
                # (не слишком большие степени)
                max_power = max(abs(p) for p in test_formula.powers.values()) if test_formula.powers else 0

                results.append({
                    'eps0_idx': i_eps + 1,
                    'mu0_idx': i_mu + 1,
                    'eps0_formula': eps0_formulas[i_eps],
                    'mu0_formula': mu0_formulas[i_mu],
                    'product_scalar': product.scalar,
                    'test_formula': test_formula,
                    'test_str': str(test_formula),
                    'max_power': max_power,
                    'is_simple': max_power <= 3 and len(test_formula.powers) <= 5
                })

                checked += 1
                if checked % 1000 == 0:
                    print(f"  Проверено {checked} пар, найдено {len(results)} вариантов")

        print(f"Всего проверено: {checked} пар")
        print(f"Найдено вариантов: {len(results)}")

        # Сортируем: сначала простые (маленькие степени, мало компонентов)
        results.sort(key=lambda x: (
            not x['is_simple'],  # Сначала простые
            x['max_power'],  # Затем с маленькими степенями
            len(x['test_formula'].powers)  # Затем с меньшим числом компонентов
        ))

        return results


# ==========================================
# АНАЛИЗ РЕЗУЛЬТАТОВ
# ==========================================

def analyze_results_v2(results: List[Dict]):
    """Расширенный анализ результатов"""

    print("\n" + "=" * 80)
    print("ТОП-20 ПРОСТЕЙШИХ ТЕСТОВЫХ ФОРМУЛ")
    print("=" * 80)

    for idx, res in enumerate(results[:20]):
        print(f"\n{idx + 1}. Комбинация (ε₀#{res['eps0_idx']} · μ₀#{res['mu0_idx']})")
        print(f"   ε₀: {res['eps0_formula']}")
        print(f"   μ₀: {res['mu0_formula']}")
        print(f"   ε₀·μ₀ скаляр: {res['product_scalar']:.6g}")
        print(f"   Тест: {res['test_str']}")
        print(f"   Степени: {dict(res['test_formula'].powers)}")
        print(f"   max|power|: {res['max_power']:.1f}")

        # Проверка на физический смысл
        test = res['test_formula']
        if test.scalar == 1.0 and test.is_dimensionless():
            print(f"   ★ ПОЛНОЕ СОКРАЩЕНИЕ ДО 1!")
        elif abs(test.scalar - 1.0) < 0.01:
            print(f"   ✓ Почти 1 (скаляр = {test.scalar:.6f})")

    # Группировка по тестовым формулам
    print("\n" + "=" * 80)
    print("УНИКАЛЬНЫЕ ТЕСТОВЫЕ ФОРМУЛЫ (топ-10 по частоте)")
    print("=" * 80)

    test_groups = defaultdict(list)
    for res in results:
        if res['is_simple']:
            test_groups[res['test_str']].append(res)

    sorted_groups = sorted(test_groups.items(), key=lambda x: len(x[1]), reverse=True)

    for test_str, group in sorted_groups[:10]:
        print(f"\nТест: {test_str}")
        print(f"Встречается: {len(group)} раз")
        scalars = [f"{r['product_scalar']:.4g}" for r in group[:5]]
        print(f"Скаляры ε₀·μ₀: {scalars}")

    return test_groups


# ==========================================
# ГЛАВНАЯ ФУНКЦИЯ
# ==========================================

def main():
    print("=" * 80)
    print("АНАЛИЗ ФОРМУЛ v2.0 — С КОЭФФИЦИЕНТАМИ")
    print("=" * 80)

    # Чтение файла
    print(f"\nЧтение файла: {INPUT_FILE}")
    try:
        eps0_formulas, mu0_formulas = parse_output_file(INPUT_FILE)
    except FileNotFoundError:
        print(f"Ошибка: файл '{INPUT_FILE}' не найден!")
        return

    print(f"Найдено: {len(eps0_formulas)} формул ε₀, {len(mu0_formulas)} формул μ₀")

    # Показываем примеры парсинга
    print("\n" + "=" * 80)
    print("ПРОВЕРКА ПАРСИНГА")
    print("=" * 80)

    test_cases = [
        "8π² * K^2 * e / (lnN)^4",
        "(lnK * 1/(lnN)^5 * e^π) / e",
        "1/(lnN)^4 * 8π² * e / 1/K^2",
        "N^(-1/3) * (lnN)^3 / K",
    ]

    for test in test_cases:
        formula = parse_formula_v2(test)
        if formula:
            print(f"\nВход: {test}")
            print(f"  Скаляр: {formula.scalar:.6g}")
            print(f"  Степени: {dict(formula.powers)}")
            print(f"  Строка: {str(formula)}")
        else:
            print(f"\nОШИБКА парсинга: {test}")

    # Поиск
    print("\n" + "=" * 80)
    print("ПОИСК КОМБИНАЦИЙ")
    print("=" * 80)

    searcher = FormulaSearchV2()
    results = searcher.search_combinations(eps0_formulas, mu0_formulas)

    if results:
        test_groups = analyze_results_v2(results)

        # Сохранение
        save_data = {
            'total_eps0': len(eps0_formulas),
            'total_mu0': len(mu0_formulas),
            'found_combinations': len(results),
            'simple_combinations': sum(1 for r in results if r['is_simple']),
            'top_results': []
        }

        for res in results[:30]:
            test = res['test_formula']
            save_data['top_results'].append({
                'eps0_idx': res['eps0_idx'],
                'mu0_idx': res['mu0_idx'],
                'eps0_formula': res['eps0_formula'],
                'mu0_formula': res['mu0_formula'],
                'product_scalar': res['product_scalar'],
                'test_formula': res['test_str'],
                'test_powers': test.powers,
                'test_scalar': test.scalar,
                'max_power': res['max_power']
            })

        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)

        print(f"\nРезультаты сохранены в {OUTPUT_FILE}")

        # Проверка H_TEST
        print("\n" + "=" * 80)
        print("ПРОВЕРКА H_TEST = N^(-1/3) * (lnN)^3 / K")
        print("=" * 80)

        h_test = parse_formula_v2("N^(-1/3) * (lnN)^3 / K")
        if h_test and results:
            best = results[0]
            eps_f = parse_formula_v2(best['eps0_formula'])
            mu_f = parse_formula_v2(best['mu0_formula'])

            if eps_f and mu_f:
                total = eps_f * mu_f * h_test
                print(f"ε₀ · μ₀ · H_TEST = {total}")
                if total.is_dimensionless():
                    print(f"★ БЕЗРАЗМЕРНАЯ ВЕЛИЧИНА! Скаляр = {total.scalar:.6g}")
                    if abs(total.scalar - 1.0) < 0.01:
                        print("★ ★ ★ РАВНА 1! ФОРМУЛА ПОДТВЕРЖДЕНА!")
                else:
                    print("Остались размерные компоненты")


if __name__ == "__main__":
    main()