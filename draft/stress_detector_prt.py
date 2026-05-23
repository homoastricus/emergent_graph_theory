"""
СИСТЕМАТИЧЕСКИЙ АНАЛИЗ СВЯЗИ КВАНТОВЫХ СВОЙСТВ СО СТРУКТУРНЫМИ ПАРАМЕТРАМИ ФОРМУЛ

Анализируемые параметры формул:
  1. Степень lnN (a)
  2. Положение lnN^a (числитель/знаменатель)
  3. Степень K
  4. Наличие и степень π
  5. Наличие корней (√2, √3)
  6. Знак "голого" коэффициента C (без lnN и N)

Квантовые свойства частиц:
  - Спин s
  - s(s+1)
  - Барионное число B
  - Лептонное число L
  - Электрический заряд Q
  - Изоспин I
  - Цвет (триплет/синглет/октет)
  - Поколение
  - Тип частицы (фермион/бозон)
"""

import math
import numpy as np
from scipy import stats
from collections import defaultdict

# ============================================================
# ПАРАМЕТРЫ
# ============================================================
K_val = 6.0
pi_val = math.pi
lnK_val = math.log(K_val)

# ============================================================
# ВСЕ ЧАСТИЦЫ С ФОРМУЛАМИ И КВАНТОВЫМИ СВОЙСТВАМИ
# ============================================================
# (name, formula_str, a, b, K_power, pi_power, has_sqrt2, has_sqrt3,
#  s, B, L, Q, I, I3, color, generation, fermion)

particles = [
    # ===== ЛЕПТОНЫ =====
    ('e', '4π(lnN)⁴/(√K·N^(1/3))', 4, 1 / 3, -0.5, 1, False, False,
     0.5, 0, 1, -1, 0.5, -0.5, 'singlet', 1, True),

    ('μ', '4π²(lnN)⁵/(K√3·N^(1/3))', 5, 1 / 3, -1, 2, False, True,
     0.5, 0, 1, -1, 0.5, -0.5, 'singlet', 2, True),

    ('τ', '√π(lnN)⁵·K²/N^(1/3)', 5, 1 / 3, 2, 0.5, False, False,
     0.5, 0, 1, -1, 0.5, -0.5, 'singlet', 3, True),

    # ===== КВАРКИ =====
    ('u', '(lnN)⁵√3/(4π²·N^(1/3))', 5, 1 / 3, 0, -2, False, True,
     0.5, 1 / 3, 0, 2 / 3, 0.5, 0.5, 'triplet', 1, True),

    ('d', '(lnN)⁵/(K√3·N^(1/3))', 5, 1 / 3, -1, 0, False, True,
     0.5, 1 / 3, 0, -1 / 3, 0.5, -0.5, 'triplet', 1, True),

    ('s', '(lnN)⁴π^(7/2)/N^(1/3)', 4, 1 / 3, 0, 3.5, False, False,
     0.5, 1 / 3, 0, -1 / 3, 0, 0, 'triplet', 2, True),

    ('c', '2π²(lnN)⁶/(K³·N^(1/3))', 6, 1 / 3, -3, 2, False, False,
     0.5, 1 / 3, 0, 2 / 3, 0, 0, 'triplet', 2, True),

    ('b', 'π(lnN)⁶/(K√3·N^(1/3))', 6, 1 / 3, -1, 1, False, True,
     0.5, 1 / 3, 0, -1 / 3, 0, 0, 'triplet', 3, True),

    ('t', 'K³(lnN)⁶/(π²·N^(1/3))', 6, 1 / 3, 3, -2, False, False,
     0.5, 1 / 3, 0, 2 / 3, 0, 0, 'triplet', 3, True),

    # ===== БАРИОНЫ =====
    ('p', '√π(lnN)⁶/(K^(3/2)·N^(1/3))', 6, 1 / 3, -1.5, 0.5, False, False,
     0.5, 1, 0, 1, 0.5, 0.5, 'singlet', 1, True),

    ('n', '√π(lnN)⁶/(K^(3/2)·N^(1/3))', 6, 1 / 3, -1.5, 0.5, False, False,
     0.5, 1, 0, 0, 0.5, -0.5, 'singlet', 1, True),

    ('Λ', '√2(lnN)⁶/(π²·N^(1/3))', 6, 1 / 3, 0, -2, True, False,
     0.5, 1, 0, 0, 0, 0, 'singlet', 2, True),

    ('Σ+', 'K(lnN)⁶/(4π²·N^(1/3))', 6, 1 / 3, 1, -2, False, False,
     0.5, 1, 0, 1, 0.5, 1, 'singlet', 2, True),

    ('Σ-', 'K(lnN)⁶/(4π²·N^(1/3))', 6, 1 / 3, 1, -2, False, False,
     0.5, 1, 0, -1, 0.5, -1, 'singlet', 2, True),

    ('Ξ0', '(lnN)⁶√(2π)/(K^(3/2)·N^(1/3))', 6, 1 / 3, -1.5, -0.5, False, False,
     0.5, 1, 0, 0, 0.5, -0.5, 'singlet', 2, True),

    ('Ξ-', '(lnN)⁶√(2π)/(K^(3/2)·N^(1/3))', 6, 1 / 3, -1.5, -0.5, False, False,
     0.5, 1, 0, -1, 0.5, -0.5, 'singlet', 2, True),

    ('Ω-', '(lnN)⁶π/(K^(3/2)·N^(1/3))', 6, 1 / 3, -1.5, 1, False, False,
     1.5, 1, 0, -1, 0, 0, 'singlet', 3, True),

    # ===== МЕЗОНЫ =====
    ('π±', '(lnN)⁶/(4π²√2·N^(1/3))', 6, 1 / 3, 0, -2, True, False,
     0, 0, 0, 1, 1, 1, 'singlet', 1, False),

    ('π0', '2πK³(lnN)⁴/N^(1/3)', 4, 1 / 3, 3, 1, False, False,
     0, 0, 0, 0, 1, 0, 'singlet', 1, False),

    ('K0', '(lnN)⁶√(2π)/(4π²·N^(1/3))', 6, 1 / 3, 0, -1.5, False, False,
     0, 0, 0, 0, 0.5, -0.5, 'singlet', 2, False),

    ('D0', '(lnN)⁶√(2π)/(K√3·N^(1/3))', 6, 1 / 3, -1, -0.5, False, True,
     0, 0, 0, 0, 0.5, -0.5, 'singlet', 2, False),

    ('J/ψ', '8π²√2(lnN)⁵/N^(1/3)', 5, 1 / 3, 0, 2, True, False,
     1, 0, 0, 0, 0, 0, 'singlet', 2, False),

    ('η', '2π²(lnN)⁵/N^(1/3)', 5, 1 / 3, 0, 2, False, False,
     0, 0, 0, 0, 0, 0, 'singlet', 2, False),

    ('Υ(1S)', '(lnN)⁶√3/(√2·N^(1/3))', 6, 1 / 3, 0, 0, True, True,
     1, 0, 0, 0, 0, 0, 'singlet', 3, False),

    ('φ', '(lnN)⁵√(2π)·K^(3/2)/N^(1/3)', 5, 1 / 3, 1.5, -0.5, False, False,
     1, 0, 0, 0, 0, 0, 'singlet', 2, False),

    # ===== БОЗОНЫ =====
    ('W±', '2π³(lnN)⁶/(K·N^(1/3))', 6, 1 / 3, -1, 3, False, False,
     1, 0, 0, 1, 1, 1, 'singlet', 1, False),

    ('Z0', '4π^(5/2)(lnN)⁶/(K·N^(1/3))', 6, 1 / 3, -1, 2.5, False, False,
     1, 0, 0, 0, 1, 0, 'singlet', 1, False),

    ('H', '4π²(lnN)⁶/(√K·N^(1/3))', 6, 1 / 3, -0.5, 2, False, False,
     0, 0, 0, 0, 0.5, -0.5, 'singlet', 1, False),

    # Барионы (a=6)
    ('Σ+', 'K(lnN)⁶/(4π²·N^(1/3))', 6, 1 / 3, 1, -2, False, False,
     0.5, 1, 0, 1, 0.5, 1, 'singlet', 2, True),

    ('Σ-', 'K(lnN)⁶/(4π²·N^(1/3))', 6, 1 / 3, 1, -2, False, False,
     0.5, 1, 0, -1, 0.5, -1, 'singlet', 2, True),

    ('Ξ0', '(lnN)⁶√(2π)/(K^(3/2)·N^(1/3))', 6, 1 / 3, -1.5, -0.5, False, False,
     0.5, 1, 0, 0, 0.5, -0.5, 'singlet', 2, True),

    ('Ξ-', '(lnN)⁶√(2π)/(K^(3/2)·N^(1/3))', 6, 1 / 3, -1.5, -0.5, False, False,
     0.5, 1, 0, -1, 0.5, -0.5, 'singlet', 2, True),

    ('Ξ+', '(lnN)⁶/(π·N^(1/3))', 6, 1 / 3, 0, -1, False, False,
     0.5, 1, 0, 1, 0.5, 1, 'singlet', 2, True),

    ('Ω-', '(lnN)⁶π/(K^(3/2)·N^(1/3))', 6, 1 / 3, -1.5, 1, False, False,
     1.5, 1, 0, -1, 0, 0, 'singlet', 3, True),

    ('Ω0_c', 'K(lnN)⁶/(π^(5/2)·N^(1/3))', 6, 1 / 3, 1, -2.5, False, False,
     0.5, 1, 0, 0, 0, 0, 'singlet', 3, True),

    ('Λ0_b', '(lnN)⁶√π/(√K·N^(1/3))', 6, 1 / 3, -0.5, 0.5, False, False,
     0.5, 1, 0, 0, 0, 0, 'singlet', 3, True),

    ('Λ+_c', '(lnN)⁶√π/(K·N^(1/3))', 6, 1 / 3, -1, 0.5, False, False,
     0.5, 1, 0, 1, 0, 0, 'singlet', 2, True),

    # Мезоны (a=5)
    ('φ', '(lnN)⁵√(2π)·K^(3/2)/N^(1/3)', 5, 1 / 3, 1.5, -0.5, False, False,
     1, 0, 0, 0, 0, 0, 'singlet', 2, False),

    ('ω', '(lnN)⁵2π²√2/N^(1/3)', 5, 1 / 3, 0, 2, True, False,
     1, 0, 0, 0, 0, 0, 'singlet', 2, False),

    ("η'", '(lnN)⁵K³/(2π·N^(1/3))', 5, 1 / 3, 3, -1, False, False,
     0, 0, 0, 0, 0, 0, 'singlet', 2, False),

]


# ============================================================
# ВЫЧИСЛЕНИЕ "ГОЛОГО" КОЭФФИЦИЕНТА C
# ============================================================
def compute_C0(K_pow, pi_pow, has_sqrt2, has_sqrt3):
    """Вычисляет числовой коэффициент формулы без lnN и N"""
    C = 1.0

    if K_pow != 0:
        C *= K_val ** K_pow

    if pi_pow != 0:
        C *= pi_val ** pi_pow

    if has_sqrt2:
        C *= math.sqrt(2)

    if has_sqrt3:
        C *= math.sqrt(3)

    return C


# ============================================================
# АНАЛИЗ
# ============================================================
print("=" * 110)
print("СИСТЕМАТИЧЕСКИЙ АНАЛИЗ: КВАНТОВЫЕ СВОЙСТВА ↔ СТРУКТУРА ФОРМУЛ")
print("=" * 110)

# Собираем все признаки
features = []
for p in particles:
    name, formula, a, b, K_pow, pi_pow, has_s2, has_s3, s, B, L, Q, I, I3, color, gen, ferm = p

    C0 = compute_C0(K_pow, pi_pow, has_s2, has_s3)
    sign = '+' if C0 >= 1 else '-'

    features.append({
        'name': name,
        'a': a,
        'K_pow': K_pow,
        'pi_pow': pi_pow,
        'has_sqrt2': has_s2,
        'has_sqrt3': has_s3,
        'C0': C0,
        'sign': sign,
        's': s,
        's_splus1': s * (s + 1),
        'B': B,
        'L': L,
        'Q': Q,
        'I': I,
        'color': color,
        'generation': gen,
        'fermion': ferm,
    })

# ============================================================
# ТАБЛИЦА 1: ВСЕ ПРИЗНАКИ
# ============================================================
print(f"\n{'─' * 110}")
print("ТАБЛИЦА 1: СТРУКТУРНЫЕ ПАРАМЕТРЫ ФОРМУЛ")
print(f"{'─' * 110}")

print(
    f"\n  {'Частица':<6} {'a':>3} {'K^p':>6} {'π^q':>6} {'√2':>4} {'√3':>4} {'C0':>10} {'Знак':>5} {'Спин':>5} {'s(s+1)':>8} {'B':>4} {'L':>4} {'Q':>4} {'Цвет':>10} {'Пок':>4}")
print(f"  {'-' * 100}")

for f in features:
    sqrt2_str = '✓' if f['has_sqrt2'] else '·'
    sqrt3_str = '✓' if f['has_sqrt3'] else '·'
    print(
        f"  {f['name']:<6} {f['a']:>3} {f['K_pow']:>6.1f} {f['pi_pow']:>6.1f} {sqrt2_str:>4} {sqrt3_str:>4} {f['C0']:>10.4f} {f['sign']:>5} {f['s']:>5.1f} {f['s_splus1']:>8.2f} {f['B']:>4} {f['L']:>4} {f['Q']:>4} {f['color']:>10} {f['generation']:>4}")

# ============================================================
# ТАБЛИЦА 2: КОРРЕЛЯЦИИ
# ============================================================
print(f"\n{'─' * 110}")
print("ТАБЛИЦА 2: КОРРЕЛЯЦИИ КВАНТОВЫХ ЧИСЕЛ СО СТРУКТУРНЫМИ ПАРАМЕТРАМИ")
print(f"{'─' * 110}")

# Числовые признаки
numeric_features = ['a', 'K_pow', 'pi_pow', 'C0']
numeric_quantum = ['s', 's_splus1', 'B', 'L', 'Q', 'I', 'generation']

print(f"\n  {'Признак':<12}", end='')
for q in numeric_quantum:
    print(f"  {q:>10}", end='')
print(f"\n  {'-' * (12 + 11 * len(numeric_quantum))}")

for feat in numeric_features:
    feat_vals = np.array([f[feat] for f in features])
    print(f"  {feat:<12}", end='')
    for q in numeric_quantum:
        q_vals = np.array([f[q] for f in features])
        if len(set(q_vals)) > 1 and len(set(feat_vals)) > 1:
            corr, p_val = stats.pearsonr(feat_vals, q_vals)
            stars = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else ''))
            print(f"  {corr:>7.3f}{stars:>3}", end='')
        else:
            print(f"  {'—':>10}", end='')
    print()

# ============================================================
# ТАБЛИЦА 3: ГРУППИРОВКА ПО ЗНАКУ C0
# ============================================================
print(f"\n{'─' * 110}")
print("ТАБЛИЦА 3: СВЯЗЬ ЗНАКА C0 С ФИЗИЧЕСКИМИ СВОЙСТВАМИ")
print(f"{'─' * 110}")

positive = [f for f in features if f['sign'] == '+']
negative = [f for f in features if f['sign'] == '-']

print(f"\n  ЗНАК '+' (C0 ≥ 1, n={len(positive)}):")
print(f"    Частицы: {', '.join([f['name'] for f in positive])}")
print(f"    Средний спин: {np.mean([f['s'] for f in positive]):.2f}")
print(f"    Среднее B: {np.mean([f['B'] for f in positive]):.2f}")
print(f"    Среднее L: {np.mean([f['L'] for f in positive]):.2f}")
print(f"    Фермионов: {sum(1 for f in positive if f['fermion'])}/{len(positive)}")

print(f"\n  ЗНАК '-' (C0 < 1, n={len(negative)}):")
print(f"    Частицы: {', '.join([f['name'] for f in negative])}")
print(f"    Средний спин: {np.mean([f['s'] for f in negative]):.2f}")
print(f"    Среднее B: {np.mean([f['B'] for f in negative]):.2f}")
print(f"    Среднее L: {np.mean([f['L'] for f in negative]):.2f}")
print(f"    Фермионов: {sum(1 for f in negative if f['fermion'])}/{len(negative)}")

# ============================================================
# ТАБЛИЦА 4: ГРУППИРОВКА ПО НАЛИЧИЮ √2, √3
# ============================================================
print(f"\n{'─' * 110}")
print("ТАБЛИЦА 4: СВЯЗЬ НАЛИЧИЯ √2, √3 С ФИЗИЧЕСКИМИ СВОЙСТВАМИ")
print(f"{'─' * 110}")

sqrt2_particles = [f for f in features if f['has_sqrt2']]
sqrt3_particles = [f for f in features if f['has_sqrt3']]
both = [f for f in features if f['has_sqrt2'] and f['has_sqrt3']]
neither = [f for f in features if not f['has_sqrt2'] and not f['has_sqrt3']]

print(f"\n  С √2 (n={len(sqrt2_particles)}): {', '.join([f['name'] for f in sqrt2_particles])}")
print(f"    Средний спин: {np.mean([f['s'] for f in sqrt2_particles]):.2f}")
print(f"    Фермионов: {sum(1 for f in sqrt2_particles if f['fermion'])}/{len(sqrt2_particles)}")

print(f"\n  С √3 (n={len(sqrt3_particles)}): {', '.join([f['name'] for f in sqrt3_particles])}")
print(f"    Средний спин: {np.mean([f['s'] for f in sqrt3_particles]):.2f}")
print(f"    Фермионов: {sum(1 for f in sqrt3_particles if f['fermion'])}/{len(sqrt3_particles)}")

print(f"\n  С ОБОИМИ (n={len(both)}): {', '.join([f['name'] for f in both])}")
print(f"\n  БЕЗ ОБОИХ (n={len(neither)}): {', '.join([f['name'] for f in neither])}")

# ============================================================
# ТАБЛИЦА 5: АНАЛИЗ ПО УРОВНЯМ a
# ============================================================
print(f"\n{'─' * 110}")
print("ТАБЛИЦА 5: ХАРАКТЕРИСТИКИ ПО УРОВНЯМ a")
print(f"{'─' * 110}")

by_a = defaultdict(list)
for f in features:
    by_a[f['a']].append(f)

print(
    f"\n  {'a':>4} {'N':>4} {'Ср. спин':>10} {'Ср. s(s+1)':>12} {'Ср. K^p':>10} {'Ср. π^q':>10} {'Ср. C0':>12} {'Фермионы':>10} {'Примеры'}")
print(f"  {'-' * 90}")

for a in sorted(by_a.keys()):
    items = by_a[a]
    print(f"  {a:>4} {len(items):>4} {np.mean([f['s'] for f in items]):>10.2f} "
          f"{np.mean([f['s_splus1'] for f in items]):>12.2f} "
          f"{np.mean([f['K_pow'] for f in items]):>10.2f} "
          f"{np.mean([f['pi_pow'] for f in items]):>10.2f} "
          f"{np.mean([f['C0'] for f in items]):>12.4f} "
          f"{sum(1 for f in items if f['fermion']):>10}/{len(items)} "
          f"{', '.join([f['name'] for f in items[:5]])}{'...' if len(items) > 5 else ''}")


