import matplotlib.pyplot as plt
import numpy as np
from collections import Counter, defaultdict

# ============================================================
# ДАННЫЕ ВСЕХ ЧАСТИЦ С КООРДИНАТАМИ (a, b, c, d)
# ============================================================
all_particles = {
    # ЛЕПТОНЫ
    'e':     ('lepton', 4, -1, 0, 1.0),
    'μ':     ('lepton', 5, -5, -1, 2.0),
    'τ':     ('lepton', 5, 6, 0, 0.5),
    # КВАРКИ
    'u':     ('quark', 5, 0, -1, -2.0),
    'd':     ('quark', 5, 0, -1, -1.0),
    's':     ('quark', 4, 0, 0, 3.5),
    'c':     ('quark', 6, -6, 0, 2.0),
    'b':     ('quark', 6, 0, -2, 1.0),
    't':     ('quark', 6, 0, 6, -2.0),
    # МЕЗОНЫ
    'π0':    ('meson', 4, 8, 6, 1.0),
    'π±':    ('meson', 6, -5, 0, -2.0),
    'K0':    ('meson', 6, -3, 0, -1.5),
    'η':     ('meson', 5, 2, 0, 2.0),
    "η'":    ('meson', 5, 4, 6, -1.0),
    'φ':     ('meson', 5, 4, 3, 0.5),
    'ω':     ('meson', 5, 3, 0, 2.0),
    'ρ':     ('meson', 5, 0, 1, 2.5),
    'K*':    ('meson', 6, 2, 0, -2.5),
    'δ':     ('meson', 6, -2, 0, -1.0),
    'J/ψ':   ('meson', 5, 7, 0, 2.0),
    'η_c':   ('meson', 5, 4, 6, 0.0),
    'h_c':   ('meson', 6, 1, 0, -1.0),
    'Υ(1S)': ('meson', 6, -1, 1, 0.0),
    'B':     ('meson', 6, -3, -3, 2.0),
    'B_s':   ('meson', 6, 4, 1, 0.0),
    'B_c':   ('meson', 5, 6, 4, 1.0),
    'D0':    ('meson', 6, -1, -3, 0.5),
    'Ξ++_b': ('meson', 6, 1, -2, 0.0),
    # БАРИОНЫ
    'p':     ('baryon', 6, 0, 0, 0.5),
    'n':     ('baryon', 6, 0, 0, 0.5),
    'Λ':     ('baryon', 6, 1, 0, -2.0),
    'Σ+':    ('baryon', 6, -1, 2, -2.0),
    'Ξ0':    ('baryon', 6, 1, -2, -0.5),
    'Ω-':    ('baryon', 6, 0, -3, 1.0),
    'Λ+_c':  ('baryon', 6, 0, 0, 1.0),
    'Ξ+_c':  ('baryon', 6, -3, 0, -1.0),
    'Ω0_c':  ('baryon', 6, 0, -2, -2.5),
    'Λ0_b':  ('baryon', 6, 0, -1, 0.5),
    # БОЗОНЫ
    'W±':    ('boson', 6, -2, -2, 3.0),
    'Z0':    ('boson', 6, -2, -2, 2.5),
    'H':     ('boson', 6, 0, -1, 2.0),
}

# ============================================================
# ЦВЕТА ДЛЯ КАТЕГОРИЙ
# ============================================================
colors = {
    'lepton': '#e41a1c',
    'quark': '#377eb8',
    'meson': '#4daf4a',
    'baryon': '#ff7f00',
    'boson': '#984ea3',
}

markers = {
    'lepton': 'o',
    'quark': 's',
    'meson': 'D',
    'baryon': '^',
    'boson': 'P',
}

sizes = {
    'lepton': 80,
    'quark': 60,
    'meson': 100,
    'baryon': 120,
    'boson': 150,
}

# ============================================================
# СОЗДАЁМ ФИГУРУ С 6 ПАНЕЛЯМИ
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(20, 14))
fig.suptitle('Структурные координаты частиц в базисе ЕТИ\n'
             '(a = степень ln N, b = степень √2, c = степень √3, d = степень π)',
             fontsize=16, fontweight='bold')

# ============================================================
# 1. ПЛОСКОСТЬ (b, c) — групповые факторы √2 и √3
# ============================================================
ax = axes[0, 0]
for cat in ['lepton', 'quark', 'meson', 'baryon', 'boson']:
    xs = [p[2] for p in all_particles.values() if p[0] == cat]
    ys = [p[3] for p in all_particles.values() if p[0] == cat]
    names = [name for name, p in all_particles.items() if p[0] == cat]
    ax.scatter(xs, ys, c=colors[cat], marker=markers[cat], s=sizes[cat],
               edgecolors='black', linewidth=0.5, alpha=0.8, label=cat, zorder=5)
    for x, y, name in zip(xs, ys, names):
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(5, 5),
                   fontsize=7, alpha=0.8)

# Выделяем паттерны
# Линия d-кварк → s-кварк → c-кварк → b-кварк для b-мезонов
b_meson_b = {'B': -3, 'B_s': 4, 'B_c': 6}
b_meson_c = {'B': -3, 'B_s': 1, 'B_c': 4}
for name in b_meson_b:
    if name in all_particles:
        ax.annotate('', xy=(b_meson_b[name], b_meson_c[name]),
                   fontsize=9, color='darkgreen', fontweight='bold')

ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
ax.axvline(x=0, color='gray', linewidth=0.5, linestyle='--')
ax.set_xlabel('b (степень √2)', fontsize=12)
ax.set_ylabel('c (степень √3)', fontsize=12)
ax.set_title('Плоскость групповых факторов (b, c)', fontsize=13, fontweight='bold')
ax.legend(loc='upper left', fontsize=8)
ax.grid(True, alpha=0.3)

# ============================================================
# 2. ПЛОСКОСТЬ (a, d) — спектральный уровень и π
# ============================================================
ax = axes[0, 1]
for cat in ['lepton', 'quark', 'meson', 'baryon', 'boson']:
    xs = [p[1] for p in all_particles.values() if p[0] == cat]
    ys = [p[4] for p in all_particles.values() if p[0] == cat]
    names = [name for name, p in all_particles.items() if p[0] == cat]
    ax.scatter(xs, ys, c=colors[cat], marker=markers[cat], s=sizes[cat],
               edgecolors='black', linewidth=0.5, alpha=0.8, label=cat, zorder=5)
    for x, y, name in zip(xs, ys, names):
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(5, 5),
                   fontsize=7, alpha=0.8)

ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
ax.set_xlabel('a (степень ln N)', fontsize=12)
ax.set_ylabel('d (степень π)', fontsize=12)
ax.set_title('Плоскость спектрального уровня (a, d)', fontsize=13, fontweight='bold')
ax.legend(loc='upper left', fontsize=8)
ax.grid(True, alpha=0.3)

# ============================================================
# 3. ПЛОСКОСТЬ (b, d) — √2 и π
# ============================================================
ax = axes[0, 2]
for cat in ['lepton', 'quark', 'meson', 'baryon', 'boson']:
    xs = [p[2] for p in all_particles.values() if p[0] == cat]
    ys = [p[4] for p in all_particles.values() if p[0] == cat]
    names = [name for name, p in all_particles.items() if p[0] == cat]
    ax.scatter(xs, ys, c=colors[cat], marker=markers[cat], s=sizes[cat],
               edgecolors='black', linewidth=0.5, alpha=0.8, label=cat, zorder=5)
    for x, y, name in zip(xs, ys, names):
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(5, 5),
                   fontsize=7, alpha=0.8)

ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
ax.axvline(x=0, color='gray', linewidth=0.5, linestyle='--')
ax.set_xlabel('b (степень √2)', fontsize=12)
ax.set_ylabel('d (степень π)', fontsize=12)
ax.set_title('Плоскость (b, d): SU(2) × спектр', fontsize=13, fontweight='bold')
ax.legend(loc='upper left', fontsize=8)
ax.grid(True, alpha=0.3)

# ============================================================
# 4. ПЛОСКОСТЬ (c, d) — √3 и π
# ============================================================
ax = axes[1, 0]
for cat in ['lepton', 'quark', 'meson', 'baryon', 'boson']:
    xs = [p[3] for p in all_particles.values() if p[0] == cat]
    ys = [p[4] for p in all_particles.values() if p[0] == cat]
    names = [name for name, p in all_particles.items() if p[0] == cat]
    ax.scatter(xs, ys, c=colors[cat], marker=markers[cat], s=sizes[cat],
               edgecolors='black', linewidth=0.5, alpha=0.8, label=cat, zorder=5)
    for x, y, name in zip(xs, ys, names):
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(5, 5),
                   fontsize=7, alpha=0.8)

ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
ax.axvline(x=0, color='gray', linewidth=0.5, linestyle='--')
ax.set_xlabel('c (степень √3)', fontsize=12)
ax.set_ylabel('d (степень π)', fontsize=12)
ax.set_title('Плоскость (c, d): SU(3) × спектр', fontsize=13, fontweight='bold')
ax.legend(loc='upper left', fontsize=8)
ax.grid(True, alpha=0.3)

# ============================================================
# 5. 3D-проекция (b, c, d) — цвет = категория
# ============================================================
ax = axes[1, 1]
for cat in ['lepton', 'quark', 'meson', 'baryon', 'boson']:
    xs = [p[2] for p in all_particles.values() if p[0] == cat]
    ys = [p[3] for p in all_particles.values() if p[0] == cat]
    ax.scatter(xs, ys, c=colors[cat], marker=markers[cat], s=sizes[cat],
               edgecolors='black', linewidth=0.5, alpha=0.8, label=cat, zorder=5)

# Добавляем информацию о d через размер точек
for cat in ['lepton', 'quark', 'meson', 'baryon', 'boson']:
    xs = [p[2] for p in all_particles.values() if p[0] == cat]
    ys = [p[3] for p in all_particles.values() if p[0] == cat]
    ds = [abs(p[4]) for p in all_particles.values() if p[0] == cat]
    names = [name for name, p in all_particles.items() if p[0] == cat]
    for x, y, d, name in zip(xs, ys, ds, names):
        ax.annotate(f'{name}\n(d={d})', (x, y), textcoords="offset points",
                   xytext=(5, 5), fontsize=6, alpha=0.7)

ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
ax.axvline(x=0, color='gray', linewidth=0.5, linestyle='--')
ax.set_xlabel('b (√2)', fontsize=12)
ax.set_ylabel('c (√3)', fontsize=12)
ax.set_title('Групповые факторы с величиной |d|', fontsize=13, fontweight='bold')
ax.legend(loc='upper left', fontsize=8)
ax.grid(True, alpha=0.3)

# ============================================================
# 6. ТЕПЛОВАЯ КАРТА РАСПРЕДЕЛЕНИЯ ПО КЛЕТКАМ (b, c)
# ============================================================
ax = axes[1, 2]

# Собираем все уникальные значения b и c
all_b = sorted(set(p[2] for p in all_particles.values()))
all_c = sorted(set(p[3] for p in all_particles.values()))

# Создаём сетку
b_range = np.arange(min(all_b) - 0.5, max(all_b) + 1.5, 1)
c_range = np.arange(min(all_c) - 0.5, max(all_c) + 1.5, 1)

# Считаем количество частиц в каждой клетке
heatmap = defaultdict(int)
for p in all_particles.values():
    b_bin = round(p[2])
    c_bin = round(p[3])
    heatmap[(b_bin, c_bin)] += 1

# Создаём матрицу для тепловой карты
b_centers = np.arange(min(all_b), max(all_b) + 1)
c_centers = np.arange(min(all_c), max(all_c) + 1)
heat_matrix = np.zeros((len(c_centers), len(b_centers)))
for i, c_val in enumerate(c_centers):
    for j, b_val in enumerate(b_centers):
        heat_matrix[i, j] = heatmap.get((b_val, c_val), 0)

# Рисуем тепловую карту
im = ax.imshow(heat_matrix, cmap='YlOrRd', aspect='auto', origin='lower',
               extent=[min(all_b)-0.5, max(all_b)+0.5, min(all_c)-0.5, max(all_c)+0.5])
plt.colorbar(im, ax=ax, label='Число частиц')

# Добавляем аннотации
for (b_val, c_val), count in heatmap.items():
    if count > 0:
        ax.text(b_val, c_val, str(count), ha='center', va='center',
               fontsize=10, fontweight='bold', color='black' if count < 3 else 'white')

ax.set_xlabel('b (степень √2)', fontsize=12)
ax.set_ylabel('c (степень √3)', fontsize=12)
ax.set_title('Тепловая карта: плотность частиц\nна плоскости (b, c)', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.show()

# ============================================================
# СТАТИСТИЧЕСКИЙ АНАЛИЗ
# ============================================================
print("=" * 80)
print("СТАТИСТИКА РАСПРЕДЕЛЕНИЯ КООРДИНАТ")
print("=" * 80)

# Распределение по a
print(f"\nРаспределение по a (спектральный уровень):")
a_counts = Counter(p[1] for p in all_particles.values())
for a_val in sorted(a_counts):
    particles = [name for name, p in all_particles.items() if p[1] == a_val]
    print(f"  a = {a_val}: {a_counts[a_val]} частиц — {', '.join(particles)}")

# Распределение по b
print(f"\nРаспределение по b (степень √2):")
b_counts = Counter(p[2] for p in all_particles.values())
for b_val in sorted(b_counts):
    particles = [name for name, p in all_particles.items() if p[2] == b_val]
    print(f"  b = {b_val:>3}: {b_counts[b_val]} частиц — {', '.join(particles)}")

# Распределение по c
print(f"\nРаспределение по c (степень √3):")
c_counts = Counter(p[3] for p in all_particles.values())
for c_val in sorted(c_counts):
    particles = [name for name, p in all_particles.items() if p[3] == c_val]
    print(f"  c = {c_val:>3}: {c_counts[c_val]} частиц — {', '.join(particles)}")

# Распределение по d
print(f"\nРаспределение по d (степень π):")
d_counts = Counter(p[4] for p in all_particles.values())
for d_val in sorted(d_counts):
    particles = [name for name, p in all_particles.items() if p[4] == d_val]
    print(f"  d = {d_val:>5}: {d_counts[d_val]} частиц — {', '.join(particles)}")

# Симметрии
print(f"\nСИММЕТРИИ КООРДИНАТ:")
print(f"  Диапазон b: [{min(all_b)}, {max(all_b)}]")
print(f"  Диапазон c: [{min(all_c)}, {max(all_c)}]")
print(f"  Диапазон d: [{min(p[4] for p in all_particles.values())}, {max(p[4] for p in all_particles.values())}]")

# Проверка b ↔ -b симметрии
b_vals = [p[2] for p in all_particles.values()]
b_positive = sum(1 for b in b_vals if b > 0)
b_negative = sum(1 for b in b_vals if b < 0)
b_zero = sum(1 for b in b_vals if b == 0)
print(f"\n  Симметрия b: +{b_positive} / 0:{b_zero} / -{b_negative}")

c_vals = [p[3] for p in all_particles.values()]
c_positive = sum(1 for c in c_vals if c > 0)
c_negative = sum(1 for c in c_vals if c < 0)
c_zero = sum(1 for c in c_vals if c == 0)
print(f"  Симметрия c: +{c_positive} / 0:{c_zero} / -{c_negative}")