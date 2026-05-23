import numpy as np
import math

# ============================================================
# ФУНДАМЕНТАЛЬНЫЕ ПАРАМЕТРЫ
# ============================================================
K = 6.0
pi = math.pi
lnK = math.log(K)
gamma_E = 0.5772156649015329

N = 4.197668e121
lnN = math.log(N)
lnlnN = math.log(lnN)
inv_lnN = 1.0 / lnN
N13 = N ** (1 / 3)


# ============================================================
# ФОРМУЛЫ ВЕДУЩЕГО ПОРЯДКА
# ============================================================
def compute_leading_order():
    f0 = {}
    f0['m_e'] = 4 * pi * lnN ** 4 / (K ** 0.5 * N13)
    f0['m_muon'] = 4 * pi ** 2 * lnN ** 5 / (K * math.sqrt(3) * N13)
    f0['m_tau'] = math.sqrt(pi) * lnN ** 5 * K ** 2 / N13
    f0['m_proton'] = math.sqrt(pi) * lnN ** 6 / (K ** 1.5 * N13)
    f0['m_neutron'] = f0['m_proton'] * (1 + inv_lnN)
    f0['m_W'] = 2 * pi ** 3 * lnN ** 6 / (K * N13)
    f0['m_Z'] = 4 * pi ** (2.5) * lnN ** 6 / (K * N13)
    f0['m_Higgs'] = 4 * pi ** 2 * lnN ** 6 / (K ** 0.5 * N13)
    f0['m_pion'] = lnN ** 6 / (4 * pi ** 2 * math.sqrt(2) * N13)
    f0['m_kaon0'] = lnN ** 6 * math.sqrt(2 * pi) / (4 * pi ** 2 * N13)
    f0['m_D0'] = lnN ** 6 * math.sqrt(2 * pi) / (K * math.sqrt(3) * N13)
    f0['m_J_psi'] = 8 * pi ** 2 * math.sqrt(2) * lnN ** 5 / N13
    f0['m_Upsilon_1S'] = math.sqrt(3) * lnN ** 6 / (math.sqrt(2) * N13)
    f0['m_quark_u'] = math.sqrt(3) * lnN ** 5 / (4 * pi ** 2 * N13)
    f0['m_quark_d'] = lnN ** 5 / (K * math.sqrt(3) * N13)
    f0['m_quark_s'] = pi ** 3.5 * lnN ** 4 / N13
    f0['m_quark_c'] = 2 * pi ** 2 * lnN ** 6 / (K ** 3 * N13)
    f0['m_quark_b'] = pi * lnN ** 6 / (K * math.sqrt(3) * N13)
    f0['m_quark_t'] = K ** 3 * lnN ** 6 / (pi ** 2 * N13)

    # Отношения
    f0['m_proton/m_e'] = f0['m_proton'] / f0['m_e']
    f0['m_muon/m_e'] = f0['m_muon'] / f0['m_e']
    f0['m_tau/m_e'] = f0['m_tau'] / f0['m_e']
    f0['m_W/m_e'] = f0['m_W'] / f0['m_e']
    f0['m_Z/m_e'] = f0['m_Z'] / f0['m_e']
    f0['m_Higgs/m_e'] = f0['m_Higgs'] / f0['m_e']
    f0['m_pion/m_e'] = f0['m_pion'] / f0['m_e']
    f0['m_kaon0/m_e'] = f0['m_kaon0'] / f0['m_e']
    f0['m_D0/m_e'] = f0['m_D0'] / f0['m_e']
    f0['m_J_psi/m_e'] = f0['m_J_psi'] / f0['m_e']
    f0['m_Upsilon_1S/m_e'] = f0['m_Upsilon_1S'] / f0['m_e']

    return f0


f0 = compute_leading_order()

# ============================================================
# ЭКСПЕРИМЕНТАЛЬНЫЕ ДАННЫЕ
# ============================================================
exp_data = {
    'm_e': 9.1093837015e-31, 'm_muon': 1.883531627e-28,
    'm_tau': 3.16754e-27, 'm_proton': 1.67262192369e-27,
    'm_neutron': 1.67492749804e-27, 'm_W': 1.43362e-25,
    'm_Z': 1.62614e-25, 'm_Higgs': 2.23319e-25,
    'm_pion': 2.4880888e-28, 'm_kaon0': 8.801929e-28,
    'm_D0': 3.32479e-27, 'm_J_psi': 5.52061e-27,
    'm_Upsilon_1S': 1.68715e-26,
    'm_quark_u': 2.1650e-30, 'm_quark_d': 4.7915e-30,
    'm_quark_s': 9.635e-30, 'm_quark_c': 1.27e-27,
    'm_quark_b': 4.180e-27, 'm_quark_t': 3.04e-25,
    'm_proton/m_e': 1836.15267343, 'm_muon/m_e': 206.768283,
    'm_tau/m_e': 3477.23, 'm_W/m_e': 157378.0,
    'm_Z/m_e': 178450.0, 'm_Higgs/m_e': 245150.0,
}

# ============================================================
# ТЕСТ: УНИВЕРСАЛЬНОСТЬ k
# ============================================================
print("=" * 100)
print("ТЕСТ: УНИВЕРСАЛЬНОСТЬ ПОКАЗАТЕЛЯ k")
print("=" * 100)
print(f"\n  Гипотеза: Ω_i ∼ m_i^k")
print(f"  ln Ω_i = δ_i · ln N")
print(f"  Следовательно: k = δ_i · ln N / ln(m_i/m_e)")
print(f"  ln N = {lnN:.4f}")
print()

# Список отношений масс для тестирования
test_ratios = [
    ('m_proton/m_e', 'm_proton', 'm_e'),
    ('m_muon/m_e', 'm_muon', 'm_e'),
    ('m_tau/m_e', 'm_tau', 'm_e'),
    ('m_W/m_e', 'm_W', 'm_e'),
    ('m_Z/m_e', 'm_Z', 'm_e'),
    ('m_Higgs/m_e', 'm_Higgs', 'm_e'),
    ('m_pion/m_e', 'm_pion', 'm_e'),
    ('m_kaon0/m_e', 'm_kaon0', 'm_e'),
    ('m_D0/m_e', 'm_D0', 'm_e'),
    ('m_J_psi/m_e', 'm_J_psi', 'm_e'),
    ('m_Upsilon_1S/m_e', 'm_Upsilon_1S', 'm_e'),
]

print(
    f"  {'Отношение':<22} {'exp ratio':<14} {'model ratio':<14} {'Θ=f_exp/f₀':<14} {'δ':<12} {'ln Ω':<12} {'ln(ratio)':<14} {'k':<12}")
print(f"  {'─' * 110}")

k_values = []
k_labels = []

for ratio_name, mass_name, ref_name in test_ratios:
    # Вычисляем f0 для отношения
    if ratio_name in f0:
        f0_ratio = f0[ratio_name]
    else:
        f0_ratio = f0[mass_name] / f0[ref_name]

    # Экспериментальное отношение
    if ratio_name in exp_data:
        exp_ratio = exp_data[ratio_name]
    else:
        exp_ratio = exp_data[mass_name] / exp_data[ref_name]

    Theta = exp_ratio / f0_ratio
    ln_Theta = math.log(Theta)

    # Определяем a, b для этого отношения
    if ratio_name == 'm_proton/m_e':
        a, b = 2, 0
    elif ratio_name == 'm_muon/m_e':
        a, b = 1, 0
    elif ratio_name == 'm_tau/m_e':
        a, b = 1, 0
    elif ratio_name == 'm_W/m_e':
        a, b = 2, 0
    elif ratio_name == 'm_Z/m_e':
        a, b = 2, 0
    elif ratio_name == 'm_Higgs/m_e':
        a, b = 2, 0
    elif ratio_name == 'm_pion/m_e':
        a, b = 2, 0  # приблизительно
    elif ratio_name == 'm_kaon0/m_e':
        a, b = 2, 0
    elif ratio_name == 'm_D0/m_e':
        a, b = 2, 0
    elif ratio_name == 'm_J_psi/m_e':
        a, b = 1, 0
    elif ratio_name == 'm_Upsilon_1S/m_e':
        a, b = 2, 0
    else:
        a, b = 1, 0

    # Вычисляем C
    C_exp = ln_Theta / lnK + (b / lnK) * lnN - (a / lnK) * lnlnN

    # n и δ_b для отношений (b=0)
    n = -a
    delta_b = 0.0
    f0_struct = pi * (n + delta_b)

    # δ = ln(C_exp / f0_struct)
    if C_exp > 0 and f0_struct > 0:
        delta = math.log(C_exp / f0_struct)
    elif C_exp < 0 and f0_struct < 0:
        delta = math.log(abs(C_exp) / abs(f0_struct))
    else:
        delta = None

    if delta is not None:
        ln_Omega = delta * lnN
        ln_ratio = math.log(exp_ratio)
        k = ln_Omega / ln_ratio if ln_ratio != 0 else None

        if k is not None:
            k_values.append(k)
            k_labels.append(ratio_name)

        print(
            f"  {ratio_name:<22} {exp_ratio:<14.4f} {f0_ratio:<14.4f} {Theta:<14.8f} {delta:<12.6f} {ln_Omega:<12.6f} {ln_ratio:<14.6f} {k if k else '—':<12}")

# ============================================================
# СТАТИСТИКА k
# ============================================================
k_arr = np.array(k_values)

print(f"\n{'=' * 100}")
print("СТАТИСТИКА ПОКАЗАТЕЛЯ k")
print(f"{'=' * 100}")

print(f"\n  k для каждого отношения:")
for label, k in zip(k_labels, k_values):
    marker = " ← протон/электрон" if 'proton' in label else ""
    print(f"    {label:<22} k = {k:+.6f}{marker}")

print(f"\n  Статистика:")
print(f"    Среднее k = {np.mean(k_arr):+.6f}")
print(f"    Стд откл. = {np.std(k_arr):.6f}")
print(f"    Медиана   = {np.median(k_arr):+.6f}")
print(f"    Диапазон  = [{np.min(k_arr):+.6f}, {np.max(k_arr):+.6f}]")

# Исключаем составные частицы (мезоны)
fundamental_indices = [i for i, label in enumerate(k_labels)
                       if not any(meson in label for meson in ['pion', 'kaon', 'D0', 'J_psi', 'Upsilon'])]
k_fundamental = k_arr[fundamental_indices]

if len(k_fundamental) > 0:
    print(f"\n  Только фундаментальные частицы (без мезонов):")
    print(f"    Среднее k = {np.mean(k_fundamental):+.6f}")
    print(f"    Стд откл. = {np.std(k_fundamental):.6f}")
    for i in fundamental_indices:
        print(f"      {k_labels[i]:<22} k = {k_values[i]:+.6f}")

# ============================================================
# ВЫВОД
# ============================================================
print(f"\n{'=' * 100}")
print("ВЫВОД")
print(f"{'=' * 100}")

cv = abs(np.std(k_arr) / np.mean(k_arr)) if abs(np.mean(k_arr)) > 1e-10 else float('inf')

print(f"""
  АНАЛИЗ УНИВЕРСАЛЬНОСТИ k:

  Ожидание при k = const: стд.откл. ≪ |среднее|
  Наблюдается:
    Среднее k = {np.mean(k_arr):+.6f}
    Стд.откл.  = {np.std(k_arr):.6f}
    Коэф. вариации = {cv:.2f}

  {('✅ k УНИВЕРСАЛЕН!' if cv < 0.3 else '🟡 k ПРИМЕРНО ПОСТОЯНЕН' if cv < 0.5 else '❌ k НЕ универсален')}

  Интерпретация:
  {'─' * 50}

  Если k ≈ const (около -0.15):
    → Существует УНИВЕРСАЛЬНЫЙ скейлинг: Ω ∼ m^k
    → δ_i = (k/ln N) · ln(m_i/m_e)
    → Это НЕ RG-эффект, а ГЛОБАЛЬНАЯ масштабная зависимость
    → Энтропийная природа: более тяжёлые частицы имеют МЕНЬШЕ доступных состояний

  Если k сильно варьируется:
    → Зависимость сложнее, чем простая степенная
    → Нужна более детальная модель Ω(m)

  Численные значения:
    k ≈ {np.mean(k_arr):.4f} означает:
    При увеличении массы в 10 раз → Ω уменьшается в 10^{abs(np.mean(k_arr)):.2f} ≈ {10 ** abs(np.mean(k_arr)):.2f} раз
""")