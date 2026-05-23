import sympy as sp
from itertools import product

# ============================================================================
# СИМВОЛЬНЫЕ ПЕРЕМЕННЫЕ
# ============================================================================
pi = sp.Symbol('pi')
e = sp.Symbol('e')
lnN = sp.Symbol('lnN')
lnK = sp.Symbol('lnK')
K = sp.Symbol('K')
N = sp.Symbol('N')
p = sp.Symbol('p')

# ============================================================================
# СИМВОЛЬНАЯ ФОРМУЛА ДЛЯ c (из ЕТИ)
# ============================================================================
c_eti = pi * lnN ** 4 / (K ** 2 * lnK)

# ============================================================================
# ВСЕ УНИКАЛЬНЫЕ СИМВОЛЬНЫЕ ФОРМУЛЫ ДЛЯ ε₀
# ============================================================================
eps_formulas = [
    ("ε₀_A", lnK * e ** pi / (e * lnN ** 5)),
    ("ε₀_B", lnK / (e * lnN ** 5) * e ** pi),
    ("ε₀_C", lnK * e ** pi / (e * lnN ** 5)),
    ("ε₀_D", lnK * N ** (-sp.Integer(1) / 6) / (sp.sqrt(p * K) * lnN ** 5)),
    ("ε₀_E", lnN ** 6 * N ** (-sp.Integer(1) / 4) * K ** 3 / (1 / K ** 3)),
    ("ε₀_F", lnN ** 4 * sp.sqrt(3) * sp.sqrt(p * K) / (4 * pi ** 2)),
    ("ε₀_G", p * lnN ** (sp.Rational(31, 2)) / (8 * pi ** 2 * lnN ** 3)),
    ("ε₀_H", 2 * pi ** 2 * K ** 3 / lnN ** 6),
    ("ε₀_I", sp.sqrt(2) * N ** (-1 / pi) / (p * lnN ** 5)),
    ("ε₀_J", sp.sqrt(pi) * N ** (sp.Integer(1) / 4) * N ** (-sp.Integer(1) / 3) / K ** (sp.Rational(3, 2))),
    ("ε₀_K", K / (lnN ** 4 * 2 * pi * pi ** (sp.Rational(5, 2)))),
    ("ε₀_L", p * N ** (sp.Integer(1) / 3) * sp.sqrt(K) / lnN ** 5),
]

# ============================================================================
# ВСЕ УНИКАЛЬНЫЕ СИМВОЛЬНЫЕ ФОРМУЛЫ ДЛЯ μ₀
# ============================================================================
mu_formulas = [
    ("μ₀_A", 8 * pi ** 2 * K ** 2 * e / lnN ** 4),
    ("μ₀_B", 8 * pi ** 2 * e / (lnN ** 4 / K ** 2)),
    ("μ₀_C", lnN ** 4 / (lnN ** 5 * 8 * pi ** 2 * K ** 2)),
    ("μ₀_D", K / (8 * pi ** 2 * K ** 3 * lnN)),
    ("μ₀_E", lnK * lnN ** 2 * N ** (-sp.Integer(1) / 4) * N ** (1 / (2 * pi))),
    ("μ₀_F", 2 * pi ** 2 * sp.sqrt(2 * pi) / (lnN ** 3 * lnK)),
    ("μ₀_G", lnK * 2 * pi * sp.sqrt(K) / lnN ** 3),
    ("μ₀_H", sp.Abs(sp.log(K * p)) * sp.sqrt(pi) / (K * lnN ** 3)),
    ("μ₀_I", lnN ** 6 * pi ** (sp.Rational(5, 2)) * N ** (-sp.Integer(1) / 6) / K ** 2),
    ("μ₀_J", e ** pi / (lnN ** 2 * sp.sqrt(2 * pi) * sp.Abs(sp.log(K * p)))),
    ("μ₀_K", 8 * pi ** 2 * N ** (sp.Integer(1) / 4) * K ** 3 / N ** (sp.Integer(1) / 3)),
    ("μ₀_L", pi ** (sp.Rational(5, 2)) * e / (lnN ** 3 * sp.sqrt(3))),
]


# ============================================================================
# ФУНКЦИЯ ПРОВЕРКИ АНАЛИТИЧЕСКОГО СОКРАЩЕНИЯ
# ============================================================================
def check_analytic_cancellation(c_expr, eps_expr, mu_expr):
    product = sp.simplify(c_expr ** 2 * eps_expr * mu_expr)

    # Проверяем, является ли результат чистой константой (не зависит от переменных)
    vars_in_result = product.free_symbols

    if len(vars_in_result) == 0:
        simplified = sp.simplify(product)
        if sp.simplify(simplified - 1) == 0:
            return True, simplified, "≡ 1 (тождество)"
        else:
            return True, simplified, f"= {simplified} (константа, не 1)"

    return False, product, f"зависит от: {vars_in_result}"


# ============================================================================
# ПЕРЕБОР ВСЕХ КОМБИНАЦИЙ
# ============================================================================
results_identity = []
results_constant = []

for eps_id, eps_expr in eps_formulas:
    for mu_id, mu_expr in mu_formulas:
        is_cancelled, result, msg = check_analytic_cancellation(c_eti, eps_expr, mu_expr)

        if is_cancelled:
            if "≡ 1" in msg:
                results_identity.append((eps_id, mu_id, eps_expr, mu_expr, result))
            else:
                results_constant.append((eps_id, mu_id, eps_expr, mu_expr, result, msg))

# ============================================================================
# ВЫВОД РЕЗУЛЬТАТОВ
# ============================================================================
print("c = π·(lnN)⁴ / (K²·lnK)\n")

if results_identity:
    print("=" * 70)
    print("ТОЖДЕСТВЕННО РАВНО 1:")
    print("=" * 70)
    for eps_id, mu_id, eps_expr, mu_expr, result in results_identity:
        print(f"\nε₀ = {eps_id}")
        sp.pprint(eps_expr)
        print(f"\nμ₀ = {mu_id}")
        sp.pprint(mu_expr)
        print(f"\nc²·ε₀·μ₀ = ", end="")
        sp.pprint(result)
        print("-" * 50)
else:
    print("НЕТ ПАР С ТОЖДЕСТВОМ c²·ε₀·μ₀ ≡ 1")

if results_constant:
    print("\n" + "=" * 70)
    print("СОКРАЩАЕТСЯ ДО КОНСТАНТЫ (НЕ 1):")
    print("=" * 70)
    for eps_id, mu_id, eps_expr, mu_expr, result, msg in results_constant:
        print(f"\nε₀ = {eps_id}")
        sp.pprint(eps_expr)
        print(f"\nμ₀ = {mu_id}")
        sp.pprint(mu_expr)
        print(f"\nc²·ε₀·μ₀ {msg}")
        print("-" * 50)

# ============================================================================
# ПРОВЕРКА ФОРМУЛ ИЗ РАЗНЫХ ГРУПП
# ============================================================================
print("\n" + "=" * 70)
print("АНАЛИЗ КЛЮЧЕВЫХ ГРУПП:")
print("=" * 70)

# Группа 1: ε₀_A + μ₀_A
eps_A = lnK * e ** pi / (e * lnN ** 5)
mu_A = 8 * pi ** 2 * K ** 2 * e / lnN ** 4
prod_A = sp.simplify(c_eti ** 2 * eps_A * mu_A)
print(f"\nГруппа 1 (логарифмическая):")
print(f"ε₀ = lnK·e^π / (e·(lnN)^5)")
print(f"μ₀ = 8π²·K²·e / (lnN)^4")
print(f"c²·ε₀·μ₀ = ", end="")
sp.pprint(prod_A)
print(f"Остались переменные: {prod_A.free_symbols}")

# Группа 2: ε₀_K + μ₀_G
eps_K = K / (lnN ** 4 * 2 * pi * pi ** (sp.Rational(5, 2)))
mu_G = lnK * 2 * pi * sp.sqrt(K) / lnN ** 3
prod_KG = sp.simplify(c_eti ** 2 * eps_K * mu_G)
print(f"\nГруппа 2:")
print(f"ε₀ = K / ((lnN)^4·2π·π^(5/2))")
print(f"μ₀ = lnK·2π·√K / (lnN)^3")
print(f"c²·ε₀·μ₀ = ", end="")
sp.pprint(prod_KG)
print(f"Остались переменные: {prod_KG.free_symbols}")

# Группа 3: ε₀_J + μ₀_K
eps_J = sp.sqrt(pi) * N ** (sp.Integer(1) / 4) * N ** (-sp.Integer(1) / 3) / K ** (sp.Rational(3, 2))
mu_K = 8 * pi ** 2 * N ** (sp.Integer(1) / 4) * K ** 3 / N ** (sp.Integer(1) / 3)
prod_JK = sp.simplify(c_eti ** 2 * eps_J * mu_K)
print(f"\nГруппа 3:")
print(f"ε₀ = √π·N^(1/4)·N^(-1/3) / K^(3/2)")
print(f"μ₀ = 8π²·N^(1/4)·K³ / N^(1/3)")
print(f"c²·ε₀·μ₀ = ", end="")
sp.pprint(prod_JK)
print(f"Остались переменные: {prod_JK.free_symbols}")

# Группа 4: ε₀_D + μ₀_E
eps_D = lnK * N ** (-sp.Integer(1) / 6) / (sp.sqrt(p * K) * lnN ** 5)
mu_E = lnK * lnN ** 2 * N ** (-sp.Integer(1) / 4) * N ** (1 / (2 * pi))
prod_DE = sp.simplify(c_eti ** 2 * eps_D * mu_E)
print(f"\nГруппа 4:")
print(f"ε₀ = lnK·N^(-1/6) / (√(pK)·(lnN)^5)")
print(f"μ₀ = lnK·(lnN)^2·N^(-1/4)·N^(1/(2π))")
print(f"c²·ε₀·μ₀ = ", end="")
sp.pprint(prod_DE)
print(f"Остались переменные: {prod_DE.free_symbols}")

# Группа 5: ε₀_H + μ₀_C
eps_H = 2 * pi ** 2 * K ** 3 / lnN ** 6
mu_C = lnN ** 4 / (lnN ** 5 * 8 * pi ** 2 * K ** 2)
prod_HC = sp.simplify(c_eti ** 2 * eps_H * mu_C)
print(f"\nГруппа 5:")
print(f"ε₀ = 2π²·K³ / (lnN)^6")
print(f"μ₀ = (lnN)^4 / ((lnN)^5·8π²·K²)")
print(f"c²·ε₀·μ₀ = ", end="")
sp.pprint(prod_HC)
print(f"Остались переменные: {prod_HC.free_symbols}")

# ============================================================================
# ПОИСК ОПТИМАЛЬНЫХ ПОКАЗАТЕЛЕЙ
# ============================================================================
print("\n" + "=" * 70)
print("ПОИСК ФОРМУЛ С ПОЛНЫМ СОКРАЩЕНИЕМ:")
print("=" * 70)

# Общий вид: ε₀ ~ lnK^a · lnN^b · K^c · π^d · e^f · N^g
#            μ₀ ~ lnK^h · lnN^i · K^j · π^k · e^l · N^m
# c² = π² · lnN^8 · K^(-4) · lnK^(-2)
# c²·ε₀·μ₀ ~ lnK^(a+h-2) · lnN^(b+i+8) · K^(c+j-4) · π^(d+k+2) · e^(f+l) · N^(g+m)

print("Условия сокращения:")
print("  lnK: a + h - 2 = 0  →  a + h = 2")
print("  lnN: b + i + 8 = 0  →  b + i = -8")
print("  K:   c + j - 4 = 0  →  c + j = 4")
print("  π:   d + k + 2 = 0  →  d + k = -2")
print("  e:   f + l = 0      →  l = -f")
print("  N:   g + m = 0      →  m = -g")

print("\nПримеры валидных разложений:")
print("1) ε₀ = lnK^2 / (π · lnN^5),  μ₀ = K^4 / (π · lnN^3)")
print("   Проверка: ", end="")
eps_test1 = lnK ** 2 / (pi * lnN ** 5)
mu_test1 = K ** 4 / (pi * lnN ** 3)
sp.pprint(sp.simplify(c_eti ** 2 * eps_test1 * mu_test1))

print("2) ε₀ = 1/(π · lnN^4),  μ₀ = lnK^2 · K^4 / (π · lnN^4)")
print("   Проверка: ", end="")
eps_test2 = 1 / (pi * lnN ** 4)
mu_test2 = lnK ** 2 * K ** 4 / (pi * lnN ** 4)
sp.pprint(sp.simplify(c_eti ** 2 * eps_test2 * mu_test2))

print("3) ε₀ = lnK^2 · K^2 / (π · lnN^5),  μ₀ = K^2 / (π · lnN^3)")
print("   Проверка: ", end="")
eps_test3 = lnK ** 2 * K ** 2 / (pi * lnN ** 5)
mu_test3 = K ** 2 / (pi * lnN ** 3)
sp.pprint(sp.simplify(c_eti ** 2 * eps_test3 * mu_test3))

print("\n4) ε₀ = K^4 / (π · lnN^5 · lnK),  μ₀ = lnK^3 / (π · lnN^3)")
print("   Проверка: ", end="")
eps_test4 = K ** 4 / (pi * lnN ** 5 * lnK)
mu_test4 = lnK ** 3 / (pi * lnN ** 3)
sp.pprint(sp.simplify(c_eti ** 2 * eps_test4 * mu_test4))

print("\n5) ε₀ = lnK^2 · e^π / (e · π · lnN^5),  μ₀ = K^4 · e / (e^π · π · lnN^3)")
print("   Проверка: ", end="")
eps_test5 = lnK ** 2 * e ** pi / (e * pi * lnN ** 5)
mu_test5 = K ** 4 * e / (e ** pi * pi * lnN ** 3)
sp.pprint(sp.simplify(c_eti ** 2 * eps_test5 * mu_test5))