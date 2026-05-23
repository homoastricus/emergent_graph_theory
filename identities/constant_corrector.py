import math
import numpy as np

# ============================================================
# ФУНДАМЕНТАЛЬНЫЕ ПАРАМЕТРЫ ЕТИ
# ============================================================
K = 6.0
pi = math.pi
lnK = math.log(K)

# Эмпирическое оптимальное N
N_phys = 4.197668e121
lnN_phys = math.log(N_phys)
N13_phys = N_phys ** (1 / 3)


# ============================================================
# ЭМЕРДЖЕНТНЫЕ ФОРМУЛЫ ЕТИ
# ============================================================
def compute_eti_constants(N):
    """Вычисляет все эмерджентные константы через N"""
    lnN = math.log(N)
    N13 = N ** (1 / 3)

    hbar = (lnN ** 3) / (K * N13)
    h = 2 * pi * hbar
    c_val = pi * (lnN ** 4) / (K ** 2 * lnK)
    lP = 4 * lnN ** 2 * lnK / N13
    tP = 4 * K ** 2 * lnK ** 2 / (pi * N13 * lnN ** 2)
    EP = (lnN ** 5) * pi / (4 * K ** 3 * lnK ** 2)
    G_val = 16 * pi ** 3 * lnN ** 13 / (K ** 5 * lnK * N13)
    mP = K / (pi * 4 * lnN ** 3)
    TP = 8 * pi * N13 / (lnN ** 4)
    kB = (lnN ** 8) / (8 * pi ** 2 * N13)
    alpha = 2 * lnK ** 2 / (pi * lnN)
    me = 4 * pi * lnN ** 4 / (K ** 0.5 * N13)
    mp = math.sqrt(pi) * (lnN ** 6) / (K ** 1.5 * N13)
    qe = 1.0 / (pi * K ** 1.5 * lnN ** 7)
    eps0 = N13 / (8 * pi ** 3 * lnK * lnN ** 20)
    mu0 = (8 * pi * K ** 4 * lnK ** 3 * lnN ** 12) / N13
    Z0 = 8 * K ** 2 * pi ** 2 * lnK ** 2 * lnN ** 16 / N13

    return {
        'hbar': hbar, 'h': h, 'c': c_val, 'l_P': lP, 't_P': tP,
        'E_P': EP, 'G': G_val, 'm_P': mP, 'T_P': TP, 'k_B': kB,
        'alpha': alpha, 'm_e': me, 'm_p': mp, 'q_e': qe,
        'eps0': eps0, 'mu0': mu0, 'Z0': Z0
    }


# ============================================================
# ЭКСПЕРИМЕНТАЛЬНЫЕ ЗНАЧЕНИЯ (CODATA 2018)
# ============================================================
exp_values = {
    'hbar': 1.054571817e-34,  # Дж·с
    'c': 299792458,  # м/с
    'G': 6.67430e-11,  # м³/(кг·с²)
    'l_P': 1.616255e-35,  # м
    't_P': 5.391247e-44,  # с
    'm_P': 2.176434e-8,  # кг
    'E_P': 1.956082e9,  # Дж
    'T_P': 1.416784e32,  # К
    'k_B': 1.380649e-23,  # Дж/К
    'alpha': 1 / 137.035999084,  # безразмерная
    'm_e': 9.1093837015e-31,  # кг
    'm_p': 1.67262192369e-27,  # кг
    'q_e': 1.602176634e-19,  # Кл
    'eps0': 8.8541878128e-12,  # Ф/м
    'mu0': 1.25663706212e-6,  # Н/А²
    'Z0': 376.730313668  # Ом
}

# ТОЖДЕСТВА ДЛЯ ПРОВЕРКИ

identities = [
    # === ПЛАНКОВСКАЯ САМОСОГЛАСОВАННОСТЬ ===
    ("ħ·c/(G·m_P²)", lambda c: c['hbar']*c['c']/(c['G']*c['m_P']**2), 1),
    ("c·t_P/l_P", lambda c: c['c']*c['t_P']/c['l_P'], 1),
    ("l_P²·c³/(ħ·G)", lambda c: c['l_P']**2*c['c']**3/(c['hbar']*c['G']), 1),
    ("l_P³/(t_P²·G·m_P)", lambda c: c['l_P']**3/(c['t_P']**2*c['G']*c['m_P']), 1),  # ИСПРАВЛЕНО
    ("ħ/(c·l_P·m_P)", lambda c: c['hbar']/(c['c']*c['l_P']*c['m_P']), 1),
    ("ħ/(c²·t_P·m_P)", lambda c: c['hbar']/(c['c']**2*c['t_P']*c['m_P']), 1),
    ("c³·t_P/(G·m_P)", lambda c: c['c']**3*c['t_P']/(c['G']*c['m_P']), 1),
    ("c⁵·t_P/(E_P·G)", lambda c: c['c']**5*c['t_P']/(c['E_P']*c['G']), 1),
    ("c²·l_P/(G·m_P)", lambda c: c['c']**2*c['l_P']/(c['G']*c['m_P']), 1),
    ("E_P/(m_P·c²)", lambda c: c['E_P']/(c['m_P']*c['c']**2), 1),  # ИСПРАВЛЕНО
    ("E_P·t_P²/(l_P²·m_P)", lambda c: c['E_P']*c['t_P']**2/(c['l_P']**2*c['m_P']), 1),  # ИСПРАВЛЕНО

    # === ЭЛЕКТРОМАГНИТНАЯ САМОСОГЛАСОВАННОСТЬ ===
    ("c²·ε₀·μ₀", lambda c: c['c']**2*c['eps0']*c['mu0'], 1),
    ("c·ε₀·Z₀", lambda c: c['c']*c['eps0']*c['Z0'], 1),
    ("Z₀²·ε₀/μ₀", lambda c: c['Z0']**2*c['eps0']/c['mu0'], 1),  # ИСПРАВЛЕНО
    ("Z₀/(c·μ₀)", lambda c: c['Z0']/(c['c']*c['mu0']), 1),
    ("c³·ε₀²·μ₀·Z₀", lambda c: c['c']**3*c['eps0']**2*c['mu0']*c['Z0'], 1),

    # === КРОСС-СЕКТОРАЛЬНЫЕ (все величины эмерджентны) ===
    ("ħ²·E_P/(G²·m_P⁵)", lambda c: c['hbar']**2*c['E_P']/(c['G']**2*c['m_P']**5), 1),
    ("ħ³/(t_P·G²·m_P⁵)", lambda c: c['hbar']**3/(c['t_P']*c['G']**2*c['m_P']**5), 1),
    ("ħ·c⁵/(E_P²·G)", lambda c: c['hbar']*c['c']**5/(c['E_P']**2*c['G']), 1),
    ("ħ·c/(E_P·l_P)", lambda c: c['hbar']*c['c']/(c['E_P']*c['l_P']), 1),
    ("ħ/(E_P·t_P)", lambda c: c['hbar']/(c['E_P']*c['t_P']), 1),

    # === ДОПОЛНИТЕЛЬНЫЕ (исправленные) ===
    ("c⁴·l_P/(E_P·G)", lambda c: c['c']**4*c['l_P']/(c['E_P']*c['G']), 1),  # из meet-in-the-middle
    ("ħ·t_P/(l_P²·m_P)", lambda c: c['hbar']*c['t_P']/(c['l_P']**2*c['m_P']), 1),  # комбинация

    # Вместо ħ·c/(E_P·t_P) — правильное: ħ = E_P·t_P (определение)
    ("ħ/(E_P·t_P)",
     lambda c: c['hbar'] / (c['E_P'] * c['t_P']), 1),

    # Вместо ħ²·l_P/(G·m_P³) — правильное из meet-in-the-middle:
    ("l_P³/(t_P²·G·m_P)",
     lambda c: c['l_P']**3 / (c['t_P']**2 * c['G'] * c['m_P']), 1),

    # Вместо ħ·c²/(l_P·E_P) — правильное: ħ·c/(l_P·E_P) = 1
    ("ħ·c/(l_P·E_P)",
     lambda c: c['hbar'] * c['c'] / (c['l_P'] * c['E_P']), 1),

    # Вместо ħ·c·t_P/(l_P²·m_P) — правильное из meet-in-the-middle:
    ("ħ/(c·l_P·m_P)",
     lambda c: c['hbar'] / (c['c'] * c['l_P'] * c['m_P']), 1),

    # Вместо c³·ε₀/(Z₀·μ₀) — правильное: Z₀ = √(μ₀/ε₀)
    ("Z₀²·ε₀/μ₀",
     lambda c: c['Z0']**2 * c['eps0'] / c['mu0'], 1),

    # Вместо ħ·l_P·E_P/(c³·G) — правильное из meet-in-the-middle:
    ("ħ·c⁵/(E_P²·G)",
     lambda c: c['hbar'] * c['c']**5 / (c['E_P']**2 * c['G']), 1),
]
# ВЫЧИСЛЕНИЯ
eti_constants = compute_eti_constants(N_phys)

print("=" * 120)
print("ПРОВЕРКА АЛГЕБРАИЧЕСКОЙ САМОСОГЛАСОВАННОСТИ ЕТИ")
print("=" * 120)
print(f"\n  N = {N_phys:.6e}")
print(f"  ln N = {lnN_phys:.6f}")
print(f"  K = {K}")
print()

# ============================================================
# СРАВНЕНИЕ ЭМЕРДЖЕНТНЫХ ЗНАЧЕНИЙ С ЭКСПЕРИМЕНТОМ
# ============================================================
print("=" * 120)
print("ЧАСТЬ 1: СРАВНЕНИЕ ЭМЕРДЖЕНТНЫХ КОНСТАНТ С ЭКСПЕРИМЕНТОМ")
print("=" * 120)
print(f"\n{'Константа':<12} {'ЕТИ значение':<20} {'Эксперимент':<20} {'Отклонение %':<15} {'Статус'}")
print("-" * 100)

total_deviation = 0
count = 0
for name, exp_val in exp_values.items():
    if name in eti_constants:
        eti_val = eti_constants[name]
        dev = abs(eti_val - exp_val) / exp_val * 100
        total_deviation += dev
        count += 1

        if dev < 0.1:
            status = "⭐⭐⭐"
        elif dev < 1.0:
            status = "⭐⭐"
        elif dev < 5.0:
            status = "⭐"
        else:
            status = "⚠️"

        print(f"{name:<12} {eti_val:<20.6e} {exp_val:<20.6e} {dev:<15.8f} {status}")

if count > 0:
    print(f"\n  Среднее отклонение: {total_deviation / count:.6f}%")

# ============================================================
# ПРОВЕРКА ТОЖДЕСТВ
# ============================================================
print(f"\n{'=' * 120}")
print("ЧАСТЬ 2: АЛГЕБРАИЧЕСКИЕ ТОЖДЕСТВА")
print("=" * 120)
print(f"\n{'Тождество':<45} {'ЕТИ значение':<20} {'Цель':<15} {'Отклонение %':<15} {'Статус'}")
print("-" * 100)

results = []
for name, func, target in identities:
    try:
        value = func(eti_constants)
        if target != 0:
            deviation = abs(value - target) / abs(target) * 100
        else:
            deviation = abs(value - target)

        if deviation < 1e-10:
            status = "✅ ТОЧНО"
        elif deviation < 0.001:
            status = "⭐⭐⭐"
        elif deviation < 0.1:
            status = "⭐⭐"
        elif deviation < 1.0:
            status = "⭐"
        else:
            status = "⚠️"

        results.append((name, value, target, deviation, status))
        print(f"{name:<45} {value:<20.10f} {target:<15} {deviation:<15.6e} {status}")
    except Exception as e:
        print(f"{name:<45} {'ОШИБКА':<20} {str(e):<30}")

# ============================================================
# СТАТИСТИКА ТОЖДЕСТВ
# ============================================================
exact_count = sum(1 for r in results if r[3] < 1e-10)
perfect_count = sum(1 for r in results if r[3] < 0.001)
good_count = sum(1 for r in results if r[3] < 0.1)

print(f"\n{'=' * 120}")
print("СТАТИСТИКА ТОЖДЕСТВ")
print("=" * 120)
print(f"""
  Проверено тождеств: {len(results)}
  • Точных (Δ < 10⁻¹⁰): {exact_count} ({exact_count / len(results) * 100:.1f}%)
  • Отличных (Δ < 0.001%): {perfect_count} ({perfect_count / len(results) * 100:.1f}%)
  • Хороших (Δ < 0.1%): {good_count} ({good_count / len(results) * 100:.1f}%)

  ВЫВОД: {'✅ ВСЕ ТОЖДЕСТВА ВЫПОЛНЯЮТСЯ С ВЫСОЧАЙШЕЙ ТОЧНОСТЬЮ' if perfect_count == len(results) else '⚠️ Есть отклонения, требующие анализа'}
""")

# ============================================================
# ДОПОЛНИТЕЛЬНО: ПРОВЕРКА С ЭКСПЕРИМЕНТАЛЬНЫМИ ЗНАЧЕНИЯМИ
# ============================================================
print(f"\n{'=' * 120}")
print("ЧАСТЬ 3: ТОЖДЕСТВА С ЭКСПЕРИМЕНТАЛЬНЫМИ ЗНАЧЕНИЯМИ")
print("=" * 120)
print(f"\n{'Тождество':<45} {'Из эксп. данных':<20} {'Цель':<15} {'Отклонение %':<15}")
print("-" * 100)

# Проверяем те же тождества, но используя экспериментальные значения
for name, func, target in identities[:10]:  # первые 10 для примера
    try:
        value = func(exp_values)
        if target != 0:
            deviation = abs(value - target) / abs(target) * 100
        else:
            deviation = abs(value - target)
        print(f"{name:<45} {value:<20.10f} {target:<15} {deviation:<15.6e}")
    except:
        pass

print(f"\n  (Экспериментальные значения дают те же тождества по определению —")
print(f"   планковские единицы определены через ħ, c, G. Отличие ЕТИ в том,")
print(f"   что она ВЫВОДИТ эти соотношения, а не постулирует их.)")