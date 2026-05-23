import math
from math import gcd
from functools import reduce
from collections import defaultdict

import numpy as np
from sympy import Matrix
from sympy.matrices.normalforms import smith_normal_form

# ============================================================
# ЕТИ: ПОИСК БЕЗРАЗМЕРНЫХ КОМБИНАЦИЙ
# ОПТИМИЗИРОВАННАЯ ВЕРСИЯ
#
# ИДЕЯ:
# Вместо brute-force перебора:
#
#     Σ p_i a_i = 0
#     Σ p_i b_i = 0
#
# решаем integer nullspace:
#
#     A p = 0
#
# где:
#   a_i — степень ln(N)
#   b_i — степень N^(1/3)

K = 6.0
pi = math.pi
lnK = math.log(K)

# ============================================================
# СТРУКТУРА ФОРМУЛ
# ============================================================

def make_formula(coeff, pow_lnN, pow_N13):
    return {
        "coeff": coeff,
        "a": pow_lnN,
        "b": pow_N13,
    }


formulas = {
    'ħ': make_formula(1.0 / K, 3, -1),

    'c': make_formula(
        pi / (K ** 2 * lnK),
        4,
        0
    ),

    'l_P': make_formula(
        4 * lnK,
        2,
        -1
    ),

    't_P': make_formula(
        4 * K ** 2 * lnK ** 2 / pi,
        -2,
        -1
    ),

    'E_P': make_formula(
        pi / (4 * K ** 3 * lnK ** 2),
        5,
        0
    ),

    'G': make_formula(
        16 * pi ** 3 / (K ** 5 * lnK),
        13,
        -1
    ),

    'm_P': make_formula(
        K / (4 * pi),
        -3,
        0
    ),

    'T_P': make_formula(
        8 * pi,
        -4,
        1
    ),

    'k_B': make_formula(
        1.0 / (8 * pi ** 2),
        8,
        -1
    ),

    'α': make_formula(
        2 * lnK ** 2 / pi,
        -1,
        0
    ),

    'm_e': make_formula(
        4 * pi / math.sqrt(K),
        4,
        -1
    ),

    'm_p': make_formula(
        math.sqrt(pi) / K ** 1.5,
        6,
        -1
    ),

    'ep_0': make_formula(
        1.0 / (8 * pi ** 3 * lnK),
        -20,
        1
    ),

    'mu_0': make_formula(
        8 * pi * K ** 4 * lnK ** 3,
        12,
        -1
    ),

    'R∞': make_formula(
        4 * lnK ** 3 / (pi * K ** 1.5),
        3,
        0
    ),

    'a_0': make_formula(
        K ** 1.5 / (8 * pi * lnK),
        -4,
        0
    ),

    'Z_0': make_formula(
        8 * K ** 2 * pi ** 2 * lnK ** 2,
        16,
        -1
    ),

    'Φ_0': make_formula(
        pi ** 2 * math.sqrt(K),
        10,
        -1
    ),

    'q_e': make_formula(
        1.0 / (pi * K ** 1.5),
        -7,
        0
    ),

    'λ_e': make_formula(
        K ** 1.5 * lnK / (2 * pi),
        -5,
        0
    ),

    'λ_p': make_formula(
        2 * K ** 2.5 * lnK / math.sqrt(pi),
        -7,
        0
    ),

    'Λ': make_formula(
        1.0 / math.sqrt(pi),
        12,
        -2
    ),

    'κ': make_formula(
        128 * K ** 3 * lnK ** 3,
        -3,
        -1
    ),

    'v_H': make_formula(
        8 * pi ** 1.5 / math.sqrt(2),
        6,
        -1
    ),
}

# СПИСКИ
names = list(formulas.keys())
n = len(names)

coeffs = np.array([formulas[x]["coeff"] for x in names], dtype=float)
a_vec = np.array([formulas[x]["a"] for x in names], dtype=int)
b_vec = np.array([formulas[x]["b"] for x in names], dtype=int)

# МАТРИЦА ОГРАНИЧЕНИЙ
#
# A p = 0

A = Matrix([
    list(a_vec),
    list(b_vec)
])

print("МАТРИЦА ОГРАНИЧЕНИЙ")

print("\nA =")
print(A)

print("\nРазмерность:")
print(f"  rows = {A.rows}")
print(f"  cols = {A.cols}")

# INTEGER NULLSPACE

print("ПОИСК INTEGER NULLSPACE")

null_basis = A.nullspace()

print(f"\nРазмерность nullspace = {len(null_basis)}")

# НОРМАЛИЗАЦИЯ ВЕКТОРОВ

def vector_gcd(v):
    vals = [abs(int(x)) for x in v if int(x) != 0]
    if not vals:
        return 1
    return reduce(gcd, vals)

def normalize_vector(v):
    arr = np.array([int(x) for x in v])

    g = vector_gcd(arr)

    if g > 1:
        arr = arr // g

    # фиксируем знак
    for x in arr:
        if x != 0:
            if x < 0:
                arr = -arr
            break

    return arr

# СОХРАНЯЕМ БАЗИС
basis_vectors = []
for i, vec in enumerate(null_basis):

    dense = np.array(vec).astype(np.int64).flatten()

    dense = normalize_vector(dense)

    basis_vectors.append(dense)

    print(f"\nБазисный вектор #{i+1}")
    print(dense)

# СТРОКОВОЕ ПРЕДСТАВЛЕНИЕ
def vector_to_expression(v):
    numerator = []
    denominator = []
    for name, p in zip(names, v):

        if p > 0:

            if p == 1:
                numerator.append(name)
            else:
                numerator.append(f"{name}^{p}")

        elif p < 0:

            p2 = abs(p)

            if p2 == 1:
                denominator.append(name)
            else:
                denominator.append(f"{name}^{p2}")

    if not numerator:
        numerator = ["1"]

    num = " · ".join(numerator)

    if denominator:
        den = " · ".join(denominator)
        return f"({num}) / ({den})"

    return num

# ============================================================
# ВЫЧИСЛЕНИЕ ЧИСЛЕННОГО ЗНАЧЕНИЯ
def evaluate_vector(v):

    log_val = 0.0

    for coeff, p in zip(coeffs, v):

        if p != 0:
            log_val += p * math.log(abs(coeff))

    return math.exp(log_val)

# АНАЛИЗ БАЗИСНЫХ РЕШЕНИЙ

print("БАЗИСНЫЕ БЕЗРАЗМЕРНЫЕ КОМБИНАЦИИ")

for i, v in enumerate(basis_vectors):

    expr = vector_to_expression(v)

    val = evaluate_vector(v)

    print(f"\n[{i+1}]")
    print(expr)
    print(f"≈ {val:.12e}")

# КОМПАКТНЫЕ КОМБИНАЦИИ

print("КОМПАКТНЫЕ КОМБИНАЦИИ")
compact = []

for v in basis_vectors:

    sparsity = np.count_nonzero(v)
    l1 = np.sum(np.abs(v))
    compact.append((sparsity, l1, v))

compact.sort(key=lambda x: (x[0], x[1]))

for i, (sparsity, l1, v) in enumerate(compact[:20]):

    expr = vector_to_expression(v)

    val = evaluate_vector(v)

    print(f"\n#{i+1}")
    print(f"terms = {sparsity}, L1 = {l1}")
    print(expr)
    print(f"value = {val:.12f}")

# ПОИСК БЛИЗОСТИ К ФУНДАМЕНТАЛЬНЫМ ЧИСЛАМ
targets = {
    "π": pi,
    "1/π": 1/pi,
    "e": math.e,
    "φ": (1 + math.sqrt(5))/2,
    "√2": math.sqrt(2),
    "√3": math.sqrt(3),
    "ln2": math.log(2),
    "ln3": math.log(3),
    "lnK": lnK,
}

print("БЛИЗОСТЬ К ФУНДАМЕНТАЛЬНЫМ ЧИСЛАМ")
matches = []

for v in basis_vectors:

    val = evaluate_vector(v)

    for tname, tval in targets.items():

        err = abs(val - tval)/abs(tval)

        matches.append({
            "expr": vector_to_expression(v),
            "value": val,
            "target": tname,
            "target_value": tval,
            "error": err
        })

matches.sort(key=lambda x: x["error"])

for m in matches[:30]:

    print("\n--------------------------------------")
    print(m["expr"])
    print(f"value     = {m['value']:.12f}")
    print(f"target    = {m['target']}")
    print(f"targetval = {m['target_value']:.12f}")
    print(f"error     = {100*m['error']:.8f}%")

# SMITH NORMAL FORM
print("SMITH NORMAL FORM")

S = smith_normal_form(A)

print("\nSmith normal form:")
print(S)

# СТАТИСТИКА
print("СТАТИСТИКА")
print(f"\nЧисло формул: {n}")
print(f"Размерность пространства решений: {len(null_basis)}")

nonzero_counts = [np.count_nonzero(v) for v in basis_vectors]

print(f"\nСреднее число формул в relation:")
print(f"{np.mean(nonzero_counts):.2f}")

print(f"\nМинимальное число формул:")
print(f"{np.min(nonzero_counts)}")

print(f"\nМаксимальное число формул:")
print(f"{np.max(nonzero_counts)}")