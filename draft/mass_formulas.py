import math

K = 6.0
pi = math.pi
lnN = 280.04176  # оптимальное значение
N13 = (4.197668e121) ** (1 / 3)


def compute_mass(a, b, c, d):
    """Вычисляет массу частицы по координатам (a,b,c,d)"""
    C = (math.sqrt(2) ** b) * (math.sqrt(3) ** c) * (pi ** d)
    return C * (lnN ** a) / N13


def compute_formula(a, b, c, d):
    """Возвращает строку с развёрнутой формулой"""
    parts = []

    # Множитель (√2)^b
    if b != 0:
        if b == 1:
            parts.append("√2")
        elif b == -1:
            parts.append("1/√2")
        else:
            parts.append(f"(√2)^{{{b}}}")

    # Множитель (√3)^c
    if c != 0:
        if c == 1:
            parts.append("√3")
        elif c == -1:
            parts.append("1/√3")
        else:
            parts.append(f"(√3)^{{{c}}}")

    # Множитель π^d
    if d != 0:
        if d == 1:
            parts.append("π")
        elif d == -1:
            parts.append("1/π")
        elif d == 0.5:
            parts.append("√π")
        elif d == -0.5:
            parts.append("1/√π")
        elif d == 2.5:
            parts.append("π^{5/2}")
        elif d == -2.5:
            parts.append("1/π^{5/2}")
        else:
            parts.append(f"π^{{{d}}}")

    # Множитель (ln N)^a
    parts.append(f"(\\ln N)^{{{a}}}")

    # Множитель N^{-1/3}
    parts.append("N^{-1/3}")

    return " \\cdot ".join(parts)


# Все частицы
particles = [
    {'name': 'ρ', 'a': 5, 'b': 0, 'c': 1, 'd': 2.5, 'mass_exp': 1.49e-27},
    {'name': 'K*', 'a': 6, 'b': 2, 'c': 0, 'd': -2.5, 'mass_exp': 1.59e-27},
    {'name': 'B', 'a': 6, 'b': -3, 'c': -3, 'd': 2.0, 'mass_exp': 9.40e-27},
    {'name': 'η_c', 'a': 5, 'b': 4, 'c': 6, 'd': 0, 'mass_exp': 5.319e-27},
    {'name': 'h_c', 'a': 6, 'b': 1, 'c': 0, 'd': -1, 'mass_exp': 6.285e-27},
    {'name': 'δ', 'a': 6, 'b': -2, 'c': 0, 'd': -1, 'mass_exp': 2.196e-27},
    {'name': 'B_c', 'a': 5, 'b': 6, 'c': 4, 'd': 1, 'mass_exp': 1.1185e-26},
    {'name': 'B_s', 'a': 6, 'b': 4, 'c': 1, 'd': 0, 'mass_exp': 9.567e-27},
    {'name': 'Ξ^{++}_b', 'a': 6, 'b': 1, 'c': -2, 'd': 0, 'mass_exp': 6.453e-27},
]

print("=" * 120)
print("ЭМЕРДЖЕНТНЫЕ ФОРМУЛЫ ДЛЯ МАСС ЧАСТИЦ")
print("=" * 120)
print(f"\n  K = {K},  ln N = {lnN:.6f},  N^(1/3) = {N13:.6e}")
print(f"  Общая структура: m = (√2)^b · (√3)^c · π^d · (ln N)^a · N^(-1/3)")
print()

for p in particles:
    name = p['name']
    a, b, c, d = p['a'], p['b'], p['c'], p['d']
    mass_exp = p['mass_exp']
    mass_eti = compute_mass(a, b, c, d)
    formula = compute_formula(a, b, c, d)
    deviation = (mass_eti - mass_exp) / mass_exp * 100

    print(f"{'─' * 120}")
    print(f"\n  {name}:")
    print(f"  Координаты: a={a}, b={b}, c={c}, d={d}")
    print(f"\n  m_{{{name}}} = {formula}")
    print(f"\n  Развёрнутая формула:")

    # Детальный расчёт
    sqrt2_part = math.sqrt(2) ** b
    sqrt3_part = math.sqrt(3) ** c
    pi_part = pi ** d
    lnN_part = lnN ** a
    C = sqrt2_part * sqrt3_part * pi_part

    print(f"  (√2)^{b} = {sqrt2_part:.6f}")
    print(f"  (√3)^{c} = {sqrt3_part:.6f}")
    print(f"  π^{d} = {pi_part:.6f}")
    print(f"  C = (√2)^b · (√3)^c · π^d = {C:.6f}")
    print(f"  (ln N)^{a} = {lnN_part:.6e}")
    print(f"  N^(-1/3) = {1 / N13:.6e}")
    print(f"\n  m_{{ETI}} = {mass_eti:.6e} кг")
    print(f"  m_{{exp}} = {mass_exp:.6e} кг")
    print(f"  Отклонение: {deviation:+.4f}%")

    if abs(deviation) < 0.1:
        print(f"  Статус: ★★★ ОТЛИЧНО")
    elif abs(deviation) < 1.0:
        print(f"  Статус: ★★ ХОРОШО")
    elif abs(deviation) < 5.0:
        print(f"  Статус: ★ ПРИЕМЛЕМО")
    else:
        print(f"  Статус: ТРЕБУЕТ УТОЧНЕНИЯ")

# Сводная таблица
print(f"\n{'=' * 120}")
print("СВОДНАЯ ТАБЛИЦА")
print(f"{'=' * 120}")
print(
    f"{'Частица':<12} {'a':>3} {'b':>5} {'c':>5} {'d':>6} {'m_ETI (кг)':<18} {'m_exp (кг)':<18} {'Откл. %':<12} {'C':<12}")
print(f"{'─' * 90}")

for p in sorted(particles, key=lambda x: compute_mass(x['a'], x['b'], x['c'], x['d'])):
    name = p['name']
    a, b, c, d = p['a'], p['b'], p['c'], p['d']
    mass_eti = compute_mass(a, b, c, d)
    mass_exp = p['mass_exp']
    deviation = (mass_eti - mass_exp) / mass_exp * 100
    C = (math.sqrt(2) ** b) * (math.sqrt(3) ** c) * (pi ** d)

    bar = "█" * min(int(abs(deviation) * 10), 20)
    status = "✅" if abs(deviation) < 0.5 else ("🟡" if abs(deviation) < 1.0 else "⚠️")

    print(
        f"{status} {name:<10} {a:>3} {b:>5} {c:>5} {d:>6} {mass_eti:<18.6e} {mass_exp:<18.6e} {deviation:>+10.4f}% {C:<12.6f}")

print(f"\n  ✅ = отклонение < 0.5%")
print(f"  🟡 = отклонение 0.5% – 1.0%")
print(f"  ⚠️  = отклонение > 1.0%")