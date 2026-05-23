import math

# =========================================================
# ФУНДАМЕНТАЛЬНЫЕ ПАРАМЕТРЫ
# =========================================================
K = 6.0
pi = math.pi
lnK = math.log(K)

# Энтропия горизонта (N из голографического принципа)
N = 4.1790e121
lnN = math.log(N)

print("=" * 70)
print("ВЫЧИСЛЕНИЕ ε₀ ИЗ ЭМЕРДЖЕНТНЫХ ФОРМУЛ ЕТИ")
print("=" * 70)
print(f"K          = {K}")
print(f"π          = {pi:.10f}")
print(f"ln K       = {lnK:.10f}")
print(f"N          = {N:.6e}")
print(f"ln N       = {lnN:.10f}")
print(f"N^(1/3)    = {N ** (1 / 3):.6e}")

# =========================================================
# ЭМЕРДЖЕНТНЫЕ ФОРМУЛЫ
# =========================================================

# 1. Постоянная тонкой структуры
alpha_eti = 2 * lnK ** 2 / (pi * lnN)

# 2. Приведённая постоянная Планка
hbar_eti = lnN ** 3 / (K * N ** (1 / 3))

# 3. Скорость света
c_eti = pi * lnN ** 4 / (K ** 2 * lnK)

# 4. Элементарный заряд
q_e_eti = 1.0 / (pi * K ** (3 / 2) * lnN ** 7)

# =========================================================
# ВЫЧИСЛЕНИЕ ε₀ ДВУМЯ СПОСОБАМИ
# =========================================================

# Способ 1: Из определения через альфа
epsilon0_from_alpha = q_e_eti ** 2 / (4 * pi * alpha_eti * hbar_eti * c_eti)

# Способ 2: Прямая эмерджентная формула
epsilon0_direct = N ** (1 / 3) / (8 * pi ** 3 * lnK * lnN ** 20)

# =========================================================
# ЭКСПЕРИМЕНТАЛЬНЫЕ ЗНАЧЕНИЯ
# =========================================================
alpha_codata = 1 / 137.035999084
hbar_codata = 1.054571817e-34
c_codata = 299792458
q_e_codata = 1.602176634e-19
epsilon0_codata = 8.8541878128e-12

# =========================================================
# ВЫВОД РЕЗУЛЬТАТОВ
# =========================================================

print("\n" + "=" * 70)
print("ПРОМЕЖУТОЧНЫЕ КОНСТАНТЫ")
print("=" * 70)
print(f"{'Константа':<12} {'ЕТИ':<22} {'CODATA':<22} {'Ошибка %':<12}")
print("-" * 70)

for name, eti_val, codata_val in [
    ("α", alpha_eti, alpha_codata),
    ("ħ", hbar_eti, hbar_codata),
    ("c", c_eti, c_codata),
    ("q_e", q_e_eti, q_e_codata),
]:
    err = abs(eti_val - codata_val) / codata_val * 100
    print(f"{name:<12} {eti_val:<22.10e} {codata_val:<22.10e} {err:<12.6f}")

print("\n" + "=" * 70)
print("ДИЭЛЕКТРИЧЕСКАЯ ПРОНИЦАЕМОСТЬ ВАКУУМА")
print("=" * 70)

err_from_alpha = abs(epsilon0_from_alpha - epsilon0_codata) / epsilon0_codata * 100
err_direct = abs(epsilon0_direct - epsilon0_codata) / epsilon0_codata * 100

print(f"\nИз определения α:  {epsilon0_from_alpha:.10e} Ф/м")
print(f"Прямая формула:     {epsilon0_direct:.10e} Ф/м")
print(f"CODATA:             {epsilon0_codata:.10e} Ф/м")
print(f"\nОшибка (из α):      {err_from_alpha:.4f}%")
print(f"Ошибка (прямая):    {err_direct:.4f}%")

# Проверка: обе формулы должны совпадать
diff = abs(epsilon0_from_alpha - epsilon0_direct)
print(f"\nРасхождение методов: {diff:.2e} (должно быть ~0)")

# =========================================================
# АНАЛИЗ ЧУВСТВИТЕЛЬНОСТИ
# =========================================================
print("\n" + "=" * 70)
print("АНАЛИЗ ЧУВСТВИТЕЛЬНОСТИ ε₀ К N")
print("=" * 70)

# Показатель степени при lnN в формуле
power = 20  # (lnN)^20 в знаменателе
print(f"Показатель степени при ln N: -{power}")
print(f"Изменение N на 1% → изменение ε₀ на ~{power / 100 * 100:.1f}%")

# При каком N ошибка стала бы нулевой?
print("\nПоиск N, при котором ошибка ε₀ = 0...")


def find_N_for_epsilon0(target_eps, N_start):
    """Ищет N, дающее целевую ε₀ (прямая формула)"""

    def error(logN):
        N_val = math.exp(logN)
        eps = N_val ** (1 / 3) / (8 * pi ** 3 * lnK * math.log(N_val) ** 20)
        return abs(eps - target_eps)

    # Простой перебор
    best_N = N_start
    best_err = float('inf')

    for factor in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]:
        N_test = N_start * factor
        eps = N_test ** (1 / 3) / (8 * pi ** 3 * lnK * math.log(N_test) ** 20)
        err = abs(eps - target_eps)
        if err < best_err:
            best_err = err
            best_N = N_test

    return best_N, best_err


N_opt, err_opt = find_N_for_epsilon0(epsilon0_codata, N)
eps_opt = N_opt ** (1 / 3) / (8 * pi ** 3 * lnK * math.log(N_opt) ** 20)
print(f"N_opt     = {N_opt:.6e}")
print(f"ln N_opt  = {math.log(N_opt):.6f}")
print(f"ε₀(N_opt) = {eps_opt:.10e}")
print(f"Ошибка    = {abs(eps_opt - epsilon0_codata) / epsilon0_codata * 100:.6f}%")

# =========================================================
# СТРУКТУРА ФОРМУЛЫ
# =========================================================
print("\n" + "=" * 70)
print("АНАТОМИЯ ФОРМУЛЫ ε₀")
print("=" * 70)

components = {
    "N^(1/3)": N ** (1 / 3),
    "8π³": 8 * pi ** 3,
    "ln K": lnK,
    "(ln N)^20": lnN ** 20,
}

for name, val in components.items():
    print(f"  {name:<15} = {val:.6e}")

eps_check = components["N^(1/3)"] / (components["8π³"] * components["ln K"] * components["(ln N)^20"])
print(f"\n  ε₀ = {eps_check:.10e}")

print("\n" + "=" * 70)
print("ВЫВОД")
print("=" * 70)
print(f"Формула: ε₀ = N^(1/3) / (8π³ · ln K · (ln N)^20)")
print(f"Точность: {err_from_alpha:.2f}%")
print(f"Статус: требуется уточнение (цель <0.1%)")