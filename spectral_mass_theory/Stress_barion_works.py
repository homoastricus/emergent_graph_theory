import numpy as np
from collections import defaultdict

# ============================================================
# ДАННЫЕ (барионы — лучше начать с них)
# ============================================================

particles = {
    'p': (6, -3, -3, 0.5, 3, 0, 0.5, 0.5, 0, 0, 0, 1, 0.5, 1, 'baryon'),
    'n': (6, -3, -3, 0.5, 3, 0, 0.5, -0.5, 0, 0, 0, 1, 0.5, 0, 'baryon'),
    'Λ': (6, 1, 0, -2.0, 3, 0, 0, 0, -1, 0, 0, 1, 0.5, 0, 'baryon'),
    'Σ+': (6, -2, 2, -2.0, 3, 0, 1, 1, -1, 0, 0, 1, 0.5, 1, 'baryon'),
    'Σ0': (6, -2, 2, -2.0, 3, 0, 1, 0, -1, 0, 0, 1, 0.5, 0, 'baryon'),
    'Σ-': (6, -2, 2, -2.0, 3, 0, 1, -1, -1, 0, 0, 1, 0.5, -1, 'baryon'),
    'Ξ0': (6, -2, -3, 0.5, 3, 0, 0.5, 0.5, -2, 0, 0, 1, 0.5, 0, 'baryon'),
    'Ξ-': (6, -2, -3, 0.5, 3, 0, 0.5, -0.5, -2, 0, 0, 1, 0.5, -1, 'baryon'),
    'Ω-': (6, -3, -3, 1.0, 3, 0, 0, 0, -3, 0, 0, 1, 1.5, -1, 'baryon'),
}

# ============================================================
# ВСПОМОГАТЕЛЬНОЕ
# ============================================================

def compute_p(v):
    return (v[4] + v[5]) / 2


def compute_deltas():
    data = []

    for name, v in particles.items():
        alpha, q = v[1], v[3]
        g = v[11]
        p = compute_p(v)

        delta_alpha = alpha - (-2*p)
        delta_q = q - (p + g)

        data.append((name, delta_alpha, delta_q))

    return data


# ============================================================
# ПРОСТАЯ КЛАСТЕРИЗАЦИЯ (по близости)
# ============================================================

def cluster(values, tol=0.6):
    clusters = []

    for val in values:
        placed = False
        for cluster in clusters:
            if abs(cluster[0] - val) < tol:
                cluster.append(val)
                placed = True
                break
        if not placed:
            clusters.append([val])

    return clusters


# ============================================================
# АНАЛИЗ
# ============================================================

data = compute_deltas()

print("\n=== Δ ДАННЫЕ ===")
for name, da, dq in data:
    print(f"{name:3} | δα = {da:5.2f} | δq = {dq:5.2f}")

# --- кластеризация ---
alpha_vals = [d[1] for d in data]
q_vals = [d[2] for d in data]

alpha_clusters = cluster(alpha_vals)
q_clusters = cluster(q_vals)

print("\n=== КЛАСТЕРЫ δα ===")
for c in alpha_clusters:
    print(sorted([round(x,2) for x in c]))

print("\n=== КЛАСТЕРЫ δq ===")
for c in q_clusters:
    print(sorted([round(x,2) for x in c]))

# ============================================================
# ПРИВЯЗКА К ЧАСТИЦАМ
# ============================================================

def assign_cluster(val, clusters, tol=0.6):
    for i, c in enumerate(clusters):
        if abs(c[0] - val) < tol:
            return i
    return -1

print("\n=== ГРУППЫ (кандидаты в мультиплеты) ===")

for name, da, dq in data:
    ca = assign_cluster(da, alpha_clusters)
    cq = assign_cluster(dq, q_clusters)
    print(f"{name:3} → α-кластер {ca}, q-кластер {cq}")