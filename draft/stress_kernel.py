import sympy as sp

# ============================================================
# МАТРИЦА ЕТИ
# строки = константы
# столбцы = [s2, s3, s_pi, s_lnK, s_lnN, s_N]
# ============================================================

M = sp.Matrix([

    # s2, s3, s_pi, s_lnK, s_lnN, s_N

    [0, 0, 0, 0, 3, -sp.Rational(1, 3)],  # hbar
    [2, 0, 1, 0, 3, -sp.Rational(1, 3)],  # h
    [0, 0, 1, -1, 4, 0],  # c
    [4, 0, 0, 1, 2, -sp.Rational(1, 3)],  # lP
    [4, 0, -1, 2, -2, -sp.Rational(1, 3)],  # tP
    [-8, -6, 1, -2, 5, 0],  # EP
    [-6, -10, 3, -1, 13, -sp.Rational(1, 3)],  # G
    [0, 2, -1, 0, -3, 0],  # mP
    [6, 0, 1, 0, -4, sp.Rational(1, 3)],  # TP
    [-6, 0, -2, 0, 8, 0],  # kB
    [2, 0, -1, 2, -1, 0],  # alpha
    [0, 0, 1, 0, 4, -sp.Rational(1, 3)],  # me
    [0, 0, sp.Rational(1, 2), 0, 6, -sp.Rational(1, 3)],  # mp
    [-6, 0, -3, -1, -20, sp.Rational(1, 3)],  # eps0
    [6, 0, 1, 3, 12, -sp.Rational(1, 3)],  # mu0
    [0, 0, -1, 0, -7, 0],  # qe
    [4, 0, -1, 3, 3, 0],  # Rinf
    [-6, 0, -1, -1, -4, 0],  # a0
    [6, 0, 2, 2, 16, -sp.Rational(1, 3)],  # Z0
    [0, 0, 2, 0, 10, -sp.Rational(1, 3)],  # Phi0
    [-2, 0, -1, 1, -5, 0],  # lambda_e
    [2, 0, -sp.Rational(1, 2), 1, -7, 0],  # lambda_p
    [5, 0, sp.Rational(3, 2), 0, 6, -sp.Rational(1, 3)],  # vH
    [4, -1, 2, 0, 5, -sp.Rational(1, 3)],  # muon
    [0, 0, sp.Rational(1, 2), 0, 5, -sp.Rational(1, 3)],  # tau
    [-1, 0, -2, 0, 6, -sp.Rational(1, 3)],  # pion_pm
    [2, 0, 1, 0, 4, -sp.Rational(1, 3)],  # pion0
    [0, 0, -sp.Rational(1, 2), 0, 12, -sp.Rational(2, 3)],  # Lambda
    [8, 6, 0, 3, -3, -sp.Rational(1, 3)],  # kappa
    [2, 0, -1, 0, 0, 0],  # mH/mW
])

# ============================================================
# СПИСОК ИМЕН
# ============================================================

names = [
    "hbar", "h", "c", "lP", "tP", "EP", "G", "mP", "TP", "kB",
    "alpha", "me", "mp", "eps0", "mu0", "qe", "Rinf", "a0",
    "Z0", "Phi0", "lambda_e", "lambda_p", "vH", "muon", "tau",
    "pion_pm", "pion0", "Lambda", "kappa", "mH_mW"
]

# ============================================================
# РАНГ
# ============================================================

print("=" * 80)
print("RANK ANALYSIS")
print("=" * 80)

rank = M.rank()
print(f"rank(M) = {rank}")
print(f"shape = {M.shape}")
print(f"left-kernel dimension = {M.rows - rank}")
print(f"independent constraints = {M.rows - rank}")
print(f"free parameters = {rank}")

# ============================================================
# ЯДРО M^T
# ============================================================

print("\n" + "=" * 80)
print("LEFT KERNEL (RAW BASIS)")
print("=" * 80)

kernel = M.T.nullspace()
print(f"Number of basis vectors: {len(kernel)}")

# ============================================================
# ПРЕОБРАЗОВАНИЕ В ЦЕЛОЧИСЛЕННУЮ МАТРИЦУ ЯДРА
# ============================================================

print("\n" + "=" * 80)
print("CONVERTING TO INTEGER KERNEL MATRIX")
print("=" * 80)

kernel_int_rows = []
for vec in kernel:
    # находим общий знаменатель
    denoms = [sp.fraction(term)[1] for term in vec]
    lcm = sp.ilcm(*denoms)
    # умножаем на lcm
    ivec = [int(sp.simplify(v * lcm)) for v in vec]
    # сокращаем на gcd
    gcd = abs(sp.igcd(*ivec))
    ivec = [v // gcd for v in ivec]
    kernel_int_rows.append(ivec)

# Транспонируем: теперь это матрица 30 × 24
K_int = sp.Matrix(kernel_int_rows).T

print(f"Integer kernel matrix shape: {K_int.shape}")
print(f"Max absolute entry: {max(abs(v) for row in kernel_int_rows for v in row)}")

# ============================================================
# LLL-РЕДУКЦИЯ
# ============================================================

print("\n" + "=" * 80)
print("LLL REDUCTION")
print("=" * 80)

# Для LLL нужно, чтобы векторы были строками, а не столбцами
# Транспонируем обратно: каждая строка = один вектор ядра
K_for_lll = K_int.T  # 24 × 30

# Применяем LLL
K_lll = K_for_lll.lll()

print(f"LLL-reduced shape: {K_lll.shape}")
print(f"Max absolute entry after LLL: {max(abs(v) for row in K_lll.tolist() for v in row)}")

# ============================================================
# ВЫБОР КОРОТКИХ ВЕКТОРОВ
# ============================================================

print("\n" + "=" * 80)
print("SHORT INVARIANTS (max|coef| <= 10)")
print("=" * 80)


def vector_to_expression(vec, names):
    """Преобразует вектор показателей в строку-выражение"""
    lhs = []
    rhs = []

    for coef, name in zip(vec, names):
        if coef > 0:
            if coef == 1:
                lhs.append(name)
            else:
                lhs.append(f"{name}^{coef}")
        elif coef < 0:
            c = abs(coef)
            if c == 1:
                rhs.append(name)
            else:
                rhs.append(f"{name}^{c}")

    lhs_str = " * ".join(lhs) if lhs else "1"
    rhs_str = " * ".join(rhs) if rhs else "1"
    return f"{lhs_str} = {rhs_str}"


def norm_sq(vec):
    """Квадрат нормы вектора"""
    return sum(v * v for v in vec)


# Собираем все короткие векторы из LLL-базиса
short_vectors = []
for i in range(K_lll.rows):
    vec = [int(v) for v in K_lll.row(i)]
    max_abs = max(abs(v) for v in vec)
    if max_abs <= 10 and any(v != 0 for v in vec):
        short_vectors.append((norm_sq(vec), vec, i))

# Сортируем по норме
short_vectors.sort()

print(f"Found {len(short_vectors)} short vectors\n")

for rank_idx, (nsq, vec, orig_idx) in enumerate(short_vectors, 1):
    expr = vector_to_expression(vec, names)
    print(f"#{rank_idx} (norm²={nsq}, LLL-row={orig_idx})")
    print(f"    {expr}")
    print()

# ============================================================
# ТАКЖЕ ПРОВЕРИМ RREF БАЗИС
# ============================================================

print("=" * 80)
print("RREF-REDUCED INVARIANTS (max|coef| <= 10)")
print("=" * 80)

# Приводим целочисленную матрицу ядра к RREF
K_rref = K_int.rref()[0]

print(f"RREF shape: {K_rref.shape}")

rref_vectors = []
for i in range(K_rref.cols):
    vec = [int(K_rref[j, i]) for j in range(K_rref.rows)]
    max_abs = max(abs(v) for v in vec)
    if max_abs <= 10 and any(v != 0 for v in vec):
        rref_vectors.append((norm_sq(vec), vec, i))

rref_vectors.sort()

print(f"Found {len(rref_vectors)} short RREF vectors\n")

for rank_idx, (nsq, vec, orig_idx) in enumerate(rref_vectors, 1):
    expr = vector_to_expression(vec, names)
    print(f"#{rank_idx} (norm²={nsq}, col={orig_idx})")
    print(f"    {expr}")
    print()

# ============================================================
# ПОИСК ИНВАРИАНТОВ С МАССАМИ ЧАСТИЦ
# ============================================================

print("=" * 80)
print("INVARIANTS INVOLVING PARTICLE MASSES (max|coef| <= 20)")
print("=" * 80)

mass_indices = {
    'me': 11, 'mp': 12, 'muon': 23, 'tau': 24,
    'pion_pm': 25, 'pion0': 26, 'vH': 22, 'mH_mW': 29
}

all_vectors = []
for i in range(K_lll.rows):
    vec = [int(v) for v in K_lll.row(i)]
    max_abs = max(abs(v) for v in vec)
    if max_abs <= 20 and any(v != 0 for v in vec):
        all_vectors.append((norm_sq(vec), vec))

all_vectors.sort()

mass_vectors = []
for nsq, vec in all_vectors:
    has_mass = any(vec[idx] != 0 for name, idx in mass_indices.items())
    if has_mass:
        mass_vectors.append((nsq, vec))

print(f"Found {len(mass_vectors)} invariants involving masses\n")

for rank_idx, (nsq, vec) in enumerate(mass_vectors[:20], 1):
    expr = vector_to_expression(vec, names)
    print(f"#{rank_idx} (norm²={nsq})")
    print(f"    {expr}")
    print()

# ============================================================
# ПРОВЕРКА ВСЕХ ВЕКТОРОВ
# ============================================================

print("=" * 80)
print("VERIFICATION OF LLL BASIS")
print("=" * 80)

all_ok = True
for i in range(K_lll.rows):
    vec = sp.Matrix([int(v) for v in K_lll.row(i)])
    check = M.T * vec
    if not all(sp.simplify(x) == 0 for x in check):
        print(f"LLL-row {i}: FAIL")
        all_ok = False

if all_ok:
    print(f"All {K_lll.rows} LLL basis vectors verified OK")
else:
    print("SOME VECTORS FAILED VERIFICATION")

# ============================================================
# СТАТИСТИКА
# ============================================================

print("\n" + "=" * 80)
print("STATISTICS")
print("=" * 80)

print(f"Total constants: {M.rows}")
print(f"Basis dimensions: {rank}")
print(f"Total invariants: {M.rows - rank}")
print(f"LLL reduced basis size: {K_lll.rows}")
print(f"Short invariants (max|coef|≤10): {len(short_vectors)}")
print(f"Mass-related invariants (max|coef|≤20): {len(mass_vectors)}")

# Распределение норм
norms = [norm_sq([int(v) for v in K_lll.row(i)]) for i in range(K_lll.rows)]
norms.sort()
print(f"\nNorm distribution of LLL basis:")
print(f"  Min norm²: {min(norms)}")
print(f"  Max norm²: {max(norms)}")
print(f"  Median norm²: {norms[len(norms) // 2]}")
print(f"  Vectors with norm² ≤ 50: {sum(1 for n in norms if n <= 50)}")
print(f"  Vectors with norm² ≤ 100: {sum(1 for n in norms if n <= 100)}")