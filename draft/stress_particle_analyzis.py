import math
import numpy as np
from collections import defaultdict
from itertools import combinations

# ============================================================
# ВСЕ ЧАСТИЦЫ С КООРДИНАТАМИ (b, c, d) — без a
# ============================================================
particles = {
    # ЛЕПТОНЫ
    'e': ('lepton', -1, 0, 1.0),
    'μ': ('lepton', -5, -1, 2.0),
    'τ': ('lepton', 6, 0, 0.5),
    # КВАРКИ
    'u': ('quark', 0, -1, -2.0),
    'd': ('quark', 0, -1, -1.0),
    's': ('quark', 0, 0, 3.5),
    'c': ('quark', -6, 0, 2.0),
    'b': ('quark', 0, -2, 1.0),
    't': ('quark', 0, 6, -2.0),
    # МЕЗОНЫ
    'π0': ('meson', 8, 6, 1.0),
    'π±': ('meson', -5, 0, -2.0),
    'K0': ('meson', -3, 0, -1.5),
    'η': ('meson', 2, 0, 2.0),
    "η'": ('meson', 4, 6, -1.0),
    'φ': ('meson', 4, 3, 0.5),
    'ω': ('meson', 3, 0, 2.0),
    'ρ': ('meson', 0, 1, 2.5),
    'K*': ('meson', 2, 0, -2.5),
    'δ': ('meson', -2, 0, -1.0),
    'J/ψ': ('meson', 7, 0, 2.0),
    'η_c': ('meson', 4, 6, 0.0),
    'h_c': ('meson', 1, 0, -1.0),
    'Υ(1S)': ('meson', -1, 1, 0.0),
    'B': ('meson', -3, -3, 2.0),
    'B_s': ('meson', 4, 1, 0.0),
    'B_c': ('meson', 6, 4, 1.0),
    'D0': ('meson', -1, -3, 0.5),
    'Ξ++_b': ('meson', 1, -2, 0.0),
    # БАРИОНЫ
    'p': ('baryon', 0, 0, 0.5),
    'n': ('baryon', 0, 0, 0.5),
    'Λ': ('baryon', 1, 0, -2.0),
    'Σ+': ('baryon', -1, 2, -2.0),
    'Ξ0': ('baryon', 1, -2, -0.5),
    'Ω-': ('baryon', 0, -3, 1.0),
    'Λ+_c': ('baryon', 0, 0, 1.0),
    'Ξ+_c': ('baryon', -3, 0, -1.0),
    'Ω0_c': ('baryon', 0, -2, -2.5),
    'Λ0_b': ('baryon', 0, -1, 0.5),
    # БОЗОНЫ
    'W±': ('boson', -2, -2, 3.0),
    'Z0': ('boson', -2, -2, 2.5),
    'H': ('boson', 0, -1, 2.0),
}


# ============================================================
# ФУНКЦИЯ ВЫЧИСЛЕНИЯ C (алгебраический блок)
# ============================================================
def compute_C(b, c, d):
    return (math.sqrt(2) ** b) * (math.sqrt(3) ** c) * (math.pi ** d)


# ============================================================
# ТЕСТ 1: ОДИНАКОВЫЕ СООТНОШЕНИЯ C
# ============================================================
print("=" * 110)
print("ТЕСТ 1: ОДИНАКОВЫЕ СООТНОШЕНИЯ C_i / C_j = C_k / C_l")
print("=" * 110)
print()

# Создаём список всех пар частиц с их отношениями C
pairs = []
names = list(particles.keys())

for i, name1 in enumerate(names):
    for j, name2 in enumerate(names):
        if i < j:
            cat1, b1, c1, d1 = particles[name1]
            cat2, b2, c2, d2 = particles[name2]
            C1 = compute_C(b1, c1, d1)
            C2 = compute_C(b2, c2, d2)
            if C2 != 0 and C1 != 0:
                ratio = C1 / C2
                pairs.append((name1, name2, ratio, b1 - b2, c1 - c2, d1 - d2))

# Группируем по значению отношения
ratio_groups = defaultdict(list)
for name1, name2, ratio, db, dc, dd in pairs:
    # Округляем до 6 знаков для группировки
    key = round(ratio, 6)
    ratio_groups[key].append((name1, name2, db, dc, dd))

# Находим группы с >1 совпадением
print("ГРУППЫ С ОДИНАКОВЫМ ОТНОШЕНИЕМ C (≥2 пар):")
print(f"  {'Отношение C':<16} {'Пары частиц':<50} {'Δb, Δc, Δd'}")
print(f"  {'─' * 90}")

count = 0
for ratio, group in sorted(ratio_groups.items()):
    if len(group) >= 2:
        count += 1
        if count <= 30:  # Ограничим вывод
            ratio_str = f"{ratio:.6f}"
            pair_strs = []
            for name1, name2, db, dc, dd in group:
                pair_strs.append(f"{name1}/{name2}")
            print(f"  {ratio_str:<16} {', '.join(pair_strs[:4]):<50} Δ={group[0][1]},{group[0][2]},{group[0][3]}")
            if len(group) > 4:
                print(f"  {'':<16} ... и ещё {len(group) - 4} пар")

print(f"\n  Всего групп с одинаковым отношением: {count}")

# ============================================================
# ТЕСТ 2: СООТНОШЕНИЯ, ГДЕ ОСТАЁТСЯ 2^n ИЛИ 3^n
# ============================================================
print(f"\n{'=' * 110}")
print("ТЕСТ 2: СООТНОШЕНИЯ, ДАЮЩИЕ ЦЕЛУЮ СТЕПЕНЬ 2 ИЛИ 3")
print("=" * 110)
print()

power_of_2 = []
power_of_3 = []
power_of_6 = []

for name1, name2, ratio, db, dc, dd in pairs:
    # Проверяем, является ли отношение степенью 2
    # C1/C2 = (√2)^{b1-b2} · (√3)^{c1-c2} · π^{d1-d2}
    # Если dc=0 и dd=0, то C1/C2 = (√2)^{db} = 2^{db/2}
    # Если db=0 и dd=0, то C1/C2 = (√3)^{dc} = 3^{dc/2}

    if dc == 0 and dd == 0 and db != 0:
        # Только степень √2
        power_of_2.append((name1, name2, db, ratio))

    if db == 0 and dd == 0 and dc != 0:
        # Только степень √3
        power_of_3.append((name1, name2, dc, ratio))

    if dd == 0 and db != 0 and dc != 0 and db == dc:
        # Степень √2 и √3 одинаковые → степень √6
        power_of_6.append((name1, name2, db, ratio))

print(f"ЧИСТЫЕ СТЕПЕНИ 2 (только b различается, c и d равны):")
print(f"  {'Пара':<30} {'Δb':>5} {'C₁/C₂':<18} {'2^{Δb/2}':<18}")
print(f"  {'─' * 75}")

# Группируем по Δb
by_db = defaultdict(list)
for name1, name2, db, ratio in power_of_2:
    by_db[db].append((name1, name2, ratio))

for db in sorted(by_db.keys()):
    group = by_db[db]
    expected = 2 ** (db / 2)
    for name1, name2, ratio in group[:5]:
        print(f"  {name1:<10} / {name2:<15} {db:>5}  {ratio:<18.6f} {expected:<18.6f}")
    if len(group) > 5:
        print(f"  ... и ещё {len(group) - 5} пар")

print(f"\n  Всего пар с чистым √2: {len(power_of_2)}")

print(f"\nЧИСТЫЕ СТЕПЕНИ 3 (только c различается, b и d равны):")
print(f"  {'Пара':<30} {'Δc':>5} {'C₁/C₂':<18} {'3^{Δc/2}':<18}")
print(f"  {'─' * 75}")

by_dc = defaultdict(list)
for name1, name2, dc, ratio in power_of_3:
    by_dc[dc].append((name1, name2, ratio))

for dc in sorted(by_dc.keys()):
    group = by_dc[dc]
    expected = 3 ** (dc / 2)
    for name1, name2, ratio in group[:5]:
        print(f"  {name1:<10} / {name2:<15} {dc:>5}  {ratio:<18.6f} {expected:<18.6f}")
    if len(group) > 5:
        print(f"  ... и ещё {len(group) - 5} пар")

print(f"\n  Всего пар с чистым √3: {len(power_of_3)}")

print(f"\nЧИСТЫЕ СТЕПЕНИ 6 (b=c, d равны):")
print(f"  {'Пара':<30} {'Δb=Δc':>5} {'C₁/C₂':<18} {'6^{Δ/2}':<18}")
print(f"  {'─' * 75}")

for name1, name2, db, ratio in power_of_6[:15]:
    expected = 6 ** (db / 2)
    print(f"  {name1:<10} / {name2:<15} {db:>5}  {ratio:<18.6f} {expected:<18.6f}")

print(f"\n  Всего пар с чистым √6: {len(power_of_6)}")

# ============================================================
# ТЕСТ 3: СИММЕТРИЧНЫЕ ПАРЫ (b₁=-b₂, c₁=-c₂, d₁=-d₂)
# ============================================================
print(f"\n{'=' * 110}")
print("ТЕСТ 3: СИММЕТРИЧНЫЕ ПАРЫ (b₁=-b₂, c₁=-c₂, d₁=-d₂)")
print("=" * 110)
print()

symmetric_pairs = []
for name1, name2, ratio, db, dc, dd in pairs:
    b1, c1, d1 = particles[name1][1:]
    b2, c2, d2 = particles[name2][1:]

    # Проверяем b₁=-b₂, c₁=-c₂, d₁=-d₂
    if b1 == -b2 and c1 == -c2 and d1 == -d2 and (b1 != 0 or c1 != 0 or d1 != 0):
        symmetric_pairs.append((name1, name2, b1, c1, d1))

print(f"  {'Пара':<25} {'b':>5} {'c':>5} {'d':>6}")
print(f"  {'─' * 45}")
for name1, name2, b, c, d in symmetric_pairs:
    print(f"  {name1:<10} ↔ {name2:<10} {b:>5} {c:>5} {d:>6}")

print(f"\n  Всего симметричных пар: {len(symmetric_pairs)}")