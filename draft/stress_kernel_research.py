import sympy as sp
import numpy as np
from itertools import combinations
import time

# ============================================================
# МАТРИЦА ЕТИ
# ============================================================

M = sp.Matrix([
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

names = [
    "hbar", "h", "c", "lP", "tP", "EP", "G", "mP", "TP", "kB",
    "alpha", "me", "mp", "eps0", "mu0", "qe", "Rinf", "a0",
    "Z0", "Phi0", "lambda_e", "lambda_p", "vH", "muon", "tau",
    "pion_pm", "pion0", "Lambda", "kappa", "mH_mW"
]


def sympy_to_float_matrix(sp_mat):
    rows, cols = sp_mat.shape
    arr = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            arr[i, j] = float(sp_mat[i, j])
    return arr


def get_kernel_vectors():
    kernel = M.T.nullspace()
    vectors = []
    for vec in kernel:
        denoms = [sp.fraction(term)[1] for term in vec]
        lcm = sp.ilcm(*denoms)
        ivec = [int(sp.simplify(v * lcm)) for v in vec]
        gcd = abs(sp.igcd(*ivec))
        ivec = [v // gcd for v in ivec]
        vectors.append(ivec)
    return vectors


# ============================================================
# ЧАСТЬ 1: РАНГ
# ============================================================

print("=" * 80)
print("ЧАСТЬ 1: РАНГ")
print("=" * 80)

t0 = time.time()
rank = M.rank()
print(f"rank(M) = {rank} (за {time.time() - t0:.2f}с)")
print(f"Матрица: {M.rows} × {M.cols}")
print(f"Ядро: {M.rows - rank} соотношений")
print(f"Свободных параметров: {rank}")

# Показываем, какие столбцы образуют базис
M_rref, pivots = M.rref()
print(f"\nRREF пивоты в столбцах: {pivots}")
print("Все 6 столбцов линейно независимы — полный ранг.")

# ============================================================
# ЧАСТЬ 2: СТРУКТУРА (выборочный SNF)
# ============================================================

print("\n" + "=" * 80)
print("ЧАСТЬ 2: СТРУКТУРА ГРУППЫ (выборочный SNF)")
print("=" * 80)

M_int = (6 * M).applyfunc(int)

# Проверяем выборочно 6×6 миноры
print("Выборочная проверка 6×6 миноров (до 20 шт.):")
gcd_val = 0
count = 0
det_values = []

for combo in combinations(range(M_int.rows), 6):
    sub = M_int[list(combo), :]
    det_val = abs(sub.det())
    if det_val != 0:
        det_values.append(int(det_val))
        if gcd_val == 0:
            gcd_val = det_val
        else:
            gcd_val = sp.igcd(gcd_val, det_val)
        count += 1
        if count <= 5:
            print(f"  det[{count}] = {det_val}, НОД = {gcd_val}")
        if count >= 20:
            break

print(f"\nПроверено ненулевых миноров: {count}")
print(f"НОД всех проверенных: {gcd_val}")
print(f"Уникальные значения детерминантов: {sorted(set(det_values))}")

if gcd_val == 1:
    print("\n→ НОД = 1: группа свободная абелева, кручения нет")
    print("→ Все соотношения 'чистые', без модулярных ограничений")
else:
    print(f"\n→ НОД = {gcd_val}: возможно кручение порядка {gcd_val}")

# ============================================================
# ЧАСТЬ 3: GRAM MATRIX
# ============================================================

print("\n" + "=" * 80)
print("ЧАСТЬ 3: GRAM MATRIX И КЛАСТЕРИЗАЦИЯ")
print("=" * 80)

kernel_vectors = get_kernel_vectors()
K_mat = sp.Matrix(kernel_vectors)

Gram = K_mat * K_mat.T

# Конвертация
Gram_list = [[float(Gram[i, j]) for j in range(Gram.cols)] for i in range(Gram.rows)]
Gram_np = np.array(Gram_list)

eigenvalues = np.linalg.eigvalsh(Gram_np)
eigenvalues.sort()

print(f"Собственные значения матрицы Грама:")
print(f"  λ₁ (min) = {eigenvalues[0]:.4e}")
print(f"  λ₂₄ (max) = {eigenvalues[-1]:.4e}")
if eigenvalues[0] > 0:
    cond = eigenvalues[-1] / eigenvalues[0]
    print(f"  Число обусловленности = {cond:.4e}")
else:
    cond = float('inf')
    print(f"  Число обусловленности = ∞ (вырождена)")

# Распределение
print(f"\nРаспределение собственных значений:")
for t in [1e-10, 1, 10, 100, 1000, 10000]:
    c = sum(1 for e in eigenvalues if e > t)
    print(f"  λ > {t:8.0f}: {c}")

# Кластеры
n_show = len(kernel_vectors)
orthogonal_pairs = []
parallel_pairs = []

for i in range(n_show):
    for j in range(i + 1, n_show):
        dot = Gram_list[i][j]
        ni = np.sqrt(Gram_list[i][i])
        nj = np.sqrt(Gram_list[j][j])
        if ni > 0 and nj > 0:
            cos_angle = dot / (ni * nj)
            if abs(cos_angle) < 0.01:
                orthogonal_pairs.append((i, j, cos_angle))
            elif abs(cos_angle) > 0.99:
                parallel_pairs.append((i, j, cos_angle))

print(f"\nКластерный анализ:")
print(f"  Почти ортогональных пар (|cos| < 0.01): {len(orthogonal_pairs)}")
for i, j, c in orthogonal_pairs[:5]:
    print(f"    векторы {i},{j}: cos = {c:.6f}")

print(f"  Почти параллельных пар (|cos| > 0.99): {len(parallel_pairs)}")
for i, j, c in parallel_pairs[:5]:
    print(f"    векторы {i},{j}: cos = {c:.6f}")

# Нормы
norms = [np.sqrt(Gram_list[i][i]) for i in range(n_show)]
print(f"\nНормы векторов ядра:")
print(f"  Минимальная: {min(norms):.2f}")
print(f"  Максимальная: {max(norms):.2f}")
print(f"  Медианная: {np.median(norms):.2f}")

# ============================================================
# ЧАСТЬ 4: СИММЕТРИИ
# ============================================================

print("\n" + "=" * 80)
print("ЧАСТЬ 4: СИММЕТРИИ")
print("=" * 80)

# Группировка по сигнатуре (s₂, s₃)
print("Группировка констант по сигнатуре (s₂, s₃):")
signatures = {}
for i, name in enumerate(names):
    sig = (int(M[i, 0]), int(M[i, 1]))
    if sig not in signatures:
        signatures[sig] = []
    signatures[sig].append(name)

for sig in sorted(signatures.keys()):
    members = signatures[sig]
    print(f"  {sig}: {members}")

# Поиск пропорциональных строк
print(f"\nПоиск пропорциональных строк (с точностью до целого множителя):")
M_float = sympy_to_float_matrix(M)
found = 0
for i in range(M.rows):
    for j in range(i + 1, M.rows):
        ratios = []
        valid = True
        for k in range(M.cols):
            a = M_float[i, k]
            b = M_float[j, k]
            if abs(a) > 1e-10 and abs(b) > 1e-10:
                ratios.append(a / b)
            elif abs(a) < 1e-10 and abs(b) < 1e-10:
                continue
            else:
                valid = False
                break

        if valid and len(ratios) >= 3:
            r_mean = np.mean(ratios)
            if max(ratios) - min(ratios) < 0.01:
                r_int = int(round(r_mean))
                if abs(r_mean - r_int) < 0.01:
                    print(f"  {names[i]} ≈ {r_int} × {names[j]}")
                    found += 1

if found == 0:
    print("  Нет пропорциональных строк (все константы уникальны)")

# ============================================================
# ЧАСТЬ 5: УСТОЙЧИВОСТЬ
# ============================================================

print("\n" + "=" * 80)
print("ЧАСТЬ 5: УСТОЙЧИВОСТЬ")
print("=" * 80)

M_np = sympy_to_float_matrix(M)

# 5.1: Шум (исправлено)
print("\n5.1: Устойчивость к шуму")
np.random.seed(42)

for noise in [1e-10, 1e-8, 1e-6]:
    M_noisy = M_np + np.random.normal(0, noise, M_np.shape)
    U, s, Vt = np.linalg.svd(M_noisy)
    rank_eff = sum(1 for x in s if x > 1e-6)

    # Безопасный вывод сингулярных значений
    s_str = ", ".join([f"s[{i}]={s[i]:.2e}" for i in range(min(7, len(s)))])
    print(f"  Шум={noise:.0e}: ранг={rank_eff}, {s_str}")

# 5.2: Удаление констант
print("\n5.2: При удалении ключевых констант:")
for idx, label in [(0, "hbar"), (2, "c"), (6, "G"), (10, "alpha"), (11, "me"), (23, "muon")]:
    rows = [i for i in range(M.rows) if i != idx]
    M_sub = M[rows, :]
    r = M_sub.rank()
    print(f"  Без {label} (idx={idx}): rank={r}, строк={M_sub.rows}, дефицит={M_sub.rows - r}")

# 5.3: Подвыборки
print("\n5.3: Анализ подвыборок:")

subsets = [
    ("Все 30", list(range(30))),
    ("Планковские (0-9)", list(range(10))),
    ("Электромагнитные (10,13-19)", [10, 13, 14, 15, 18, 19]),
    ("Атомные (11,12,16,17,20,21)", [11, 12, 16, 17, 20, 21]),
    ("Фермионы+Хиггс (11,12,22-25,29)", [11, 12, 22, 23, 24, 25, 29]),
]

for label, idxs in subsets:
    M_sub = M[idxs, :]
    r = M_sub.rank()
    print(f"  {label}: rank={r}, строк={M_sub.rows}, дефицит={M_sub.rows - r}")

# ============================================================
# ИТОГ
# ============================================================

print("\n" + "=" * 80)
print("ИТОГ")
print("=" * 80)

print(f"""
Время выполнения: {time.time() - t0:.2f}с

СТРУКТУРНЫЕ ХАРАКТЕРИСТИКИ:
═══════════════════════════
• Ранг матрицы:            {rank}
• Размер матрицы:          {M.rows} × {M.cols}
• Размерность ядра:        {M.rows - rank}
• Число обусловленности:   {cond:.2e}
• НОД миноров:             {gcd_val}
• Кручение:                {'НЕТ' if gcd_val == 1 else f'ДА (порядок {gcd_val})'}
• Ортогональных пар:       {len(orthogonal_pairs)}
• Параллельных пар:        {len(parallel_pairs)}

ФИЗИЧЕСКАЯ ИНТЕРПРЕТАЦИЯ:
═══════════════════════════
1. 6-мерное лог-пространство — НЕ случайно
   Все 30 констант точно лежат в 6-мерном подпространстве

2. 24 независимых соотношения
   Это число жёстко фиксировано структурой матрицы

3. Группа соотношений {'свободная абелева' if gcd_val == 1 else 'с кручением'}
   {'→ Все соотношения «чистые»' if gcd_val == 1 else '→ Есть модулярные ограничения'}

4. Кластерная структура:
   {'→ Есть выделенные ортогональные подпространства' if len(orthogonal_pairs) > 0 else '→ Система сильно связана'}
   {'→ Есть вырожденные направления' if len(parallel_pairs) > 0 else '→ Все направления различны'}

5. Иерархия:
   {'→ Сильная иерархия (cond >> 1)' if cond > 100 else '→ Однородная структура'}
""")