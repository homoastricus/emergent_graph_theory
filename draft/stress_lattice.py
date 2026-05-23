import numpy as np
from itertools import combinations
from collections import defaultdict

# ============================================================
# ДАННЫЕ (твои текущие лучшие)
# ============================================================

particles = {
    "pi0":  (8, 6, 1.0),
    "pi":   (-5, 0, -2.0),
    "K":    (-3, 0, -1.5),
    "eta":  (2, 0, 2.0),
    "eta'": (4, 6, -1.0),
    "phi":  (4, 3, 0.5),
    "omega":(3, 0, 2.0),
    "J/psi":(7, 0, 2.0),
    "D":    (-1, -3, 0.5),
    "Upsilon":(-1, 1, 0.0),
    "rho":  (0, 1, 2.5),
    "K*":   (2, 0, -2.5),
    "B":    (-3, -3, 2.0),
}

# ============================================================
# ВЕКТОРЫ
# ============================================================

names = list(particles.keys())
points = np.array([particles[n] for n in names])

# ============================================================
# 1. РАССТОЯНИЯ
# ============================================================

print("\n=== РАССТОЯНИЯ МЕЖДУ ТОЧКАМИ ===")

distances = []
for (i, p1), (j, p2) in combinations(enumerate(points), 2):
    d = np.linalg.norm(p1 - p2)
    distances.append(round(d, 2))

unique_dist = sorted(set(distances))
print("Уникальные расстояния:", unique_dist)

# ============================================================
# 2. РАЗНОСТИ (решётка?)
# ============================================================

print("\n=== АНАЛИЗ РАЗНОСТЕЙ ===")

diffs = []

for p1, p2 in combinations(points, 2):
    d = p1 - p2
    diffs.append(tuple(np.round(d, 2)))

# частоты
freq = defaultdict(int)
for d in diffs:
    freq[d] += 1

# топ повторяющихся векторов
top = sorted(freq.items(), key=lambda x: -x[1])[:15]

for vec, count in top:
    print(f"{vec} -> {count}")

# ============================================================
# 3. ПРОВЕРКА РЕШЁТКИ
# ============================================================

print("\n=== ПРОВЕРКА БАЗИСА РЕШЁТКИ ===")

# пытаемся найти базис из 3 векторов
basis_candidates = diffs[:50]

def is_integer_combination(v, basis):
    try:
        B = np.array(basis).T
        coeffs = np.linalg.lstsq(B, v, rcond=None)[0]
        return np.allclose(coeffs, np.round(coeffs), atol=0.1)
    except:
        return False

found_basis = None

for b in combinations(basis_candidates, 3):
    good = True
    for v in diffs[:100]:
        if not is_integer_combination(np.array(v), b):
            good = False
            break
    if good:
        found_basis = b
        break

if found_basis:
    print("НАЙДЕН БАЗИС РЕШЁТКИ:")
    for v in found_basis:
        print(v)
else:
    print("Чистой решётки нет (или нужна другая метрика)")

# ============================================================
# 4. ПЛОСКОСТИ
# ============================================================

print("\n=== ПРОВЕРКА ПЛОСКОСТЕЙ ===")

# проверяем линейную зависимость (плоскость)
for combo in combinations(range(len(points)), 3):
    p1, p2, p3 = points[list(combo)]
    normal = np.cross(p2 - p1, p3 - p1)

    distances = []
    for p in points:
        dist = abs(np.dot(p - p1, normal))
        distances.append(dist)

    if sum(d < 1e-6 for d in distances) >= 6:
        print("Плоскость через:", [names[i] for i in combo])
        break

# ============================================================
# 5. КЛАСТЕРЫ ПО d
# ============================================================

print("\n=== СЛОИ ПО d ===")

layers = defaultdict(list)

for name, (b, c, d) in particles.items():
    layers[d].append(name)

for d, group in sorted(layers.items()):
    print(f"d = {d}: {group}")

# ============================================================
# 6. КЛАСТЕРЫ ПО c
# ============================================================

print("\n=== СЛОИ ПО c ===")

layers_c = defaultdict(list)

for name, (b, c, d) in particles.items():
    layers_c[c].append(name)

for c, group in sorted(layers_c.items()):
    print(f"c = {c}: {group}")