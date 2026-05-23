import math
import numpy as np

# ============================================================
# ФУНДАМЕНТАЛЬНЫЕ КОНСТАНТЫ ЕТИ
# ============================================================
K = 6.0
pi = math.pi
lnK = math.log(K)
e = math.e

# Глобальное N
N_cosmic = 4.197668e121
lnN_cosmic = math.log(N_cosmic)  # ≈ 280.047
N13_cosmic = N_cosmic ** (1 / 3)  # ≈ 3.475e40

# ============================================================
# КВАНТ СОЗНАНИЯ
# ============================================================
G_CONSCIOUSNESS = lnN_cosmic / N13_cosmic

# ============================================================
# ПАРАМЕТРЫ ФПС
# ============================================================
alpha0 = 2.2e8
mu1 = 0.01
mu2 = 1e-6
beta = 0.2
gamma_param = 0.5


# МОДЕЛЬ НЕЛОКАЛЬНОСТИ
# Параметр нелокальной связности (small-world + квантовые корреляции)
# Для small-world графа: λ_F^(SW) = N^(-1/3) / (1 + N^(-1/3))
# Для квантовой запутанности: добавляется константа η

def effective_lambda_F(N, eta=1e-3):
    """
    Эффективная связность с учётом нелокальности.

    Параметры:
    N   - число узлов
    eta - сила нелокальных связей (0...1)
          η = 0      → только локальные связи
          η = 1e-3   → умеренная нелокальность
          η = 1      → полная связность (каждый с каждым)
    """
    N13 = N ** (1 / 3)

    # 1. Локальная компонента (small-world)
    p_sw = 1 / (K * N13)
    lambda_local = K * p_sw / (1 + K * p_sw * N13)

    # 2. Нелокальная компонента (квантовые корреляции)
    # Не зависит от расстояния! Ограничена только планковским масштабом
    lambda_nonlocal = eta * (1 - 1 / N13)  # насыщается при больших N

    # 3. Эффективная связность
    lambda_eff = lambda_local + lambda_nonlocal

    return lambda_eff, lambda_local, lambda_nonlocal


# ФУНКЦИЯ ФПС С НЕЛОКАЛЬНОСТЬЮ
def compute_theta_with_nonlocality(N_local, C, N_c, phi, eta=1e-3,
                                   E_dir_empirical=None,
                                   system_name="Система",
                                   verbose=True):
    """
    АБСОЛЮТНАЯ Θ* с учётом НЕЛОКАЛЬНОСТИ.
    """

    lnN_local = math.log(N_local) if N_local > 0 else 0
    N13_local = N_local ** (1 / 3) if N_local > 0 else 1

    # 1. Масштабный фактор
    scale_factor = (C * N_local) ** (1 / 3)

    # 2. Член насыщения
    saturation = 1 + (N_local / N_c) ** gamma_param

    # 3. АБСОЛЮТНЫЙ глобальный доступ
    absolute_access = lnN_local / N13_local

    # 4. Топологический фактор С УЧЁТОМ НЕЛОКАЛЬНОСТИ
    if E_dir_empirical is not None:
        E_dir = E_dir_empirical
    else:
        E_dir = lnN_local

    # ВЫЧИСЛЯЕМ λ_F С УЧЁТОМ НЕЛОКАЛЬНОСТИ
    lambda_eff, lambda_local, lambda_nonlocal = effective_lambda_F(N_local, eta)

    topology_factor = math.exp(-mu1 * E_dir - mu2 / lambda_eff) if lambda_eff > 0 else 0

    # 5. Баланс
    balance_factor = math.cos(phi) + beta * math.sin(phi)

    # 6. АБСОЛЮТНЫЙ аргумент
    u_absolute = (alpha0
                  * (scale_factor / saturation)
                  * absolute_access
                  * topology_factor
                  * balance_factor)

    # 7. Сигмоида
    Theta = u_absolute#1 / (1 + math.exp(-u_absolute))

    print("TETA")
    print(Theta)

    if verbose:
        print(f"\n{'─' * 70}")
        print(f"ФПС С НЕЛОКАЛЬНОСТЬЮ: {system_name}")
        print(f"{'─' * 70}")
        print(f"  N                  = {N_local:.2e}")
        print(f"  Параметр нелокальности η = {eta:.0e}")
        print(f"\n  СВЯЗНОСТЬ:")
        print(f"  λ_F (локальная)     = {lambda_local:.6e}")
        print(f"  λ_F (нелокальная)   = {lambda_nonlocal:.6e}")
        print(f"  λ_F (эффективная)   = {lambda_eff:.6e}")
        print(f"\n  КОМПОНЕНТЫ:")
        print(f"  Масштаб / Насыщение = {scale_factor / saturation:.4f}")
        print(f"  Абсолютный доступ   = {absolute_access:.6e}")
        print(f"  E_dir               = {E_dir:.4f}")
        print(f"  Топологический фактор = {topology_factor:.6e}")
        print(f"  Баланс              = {balance_factor:.6f}")
        print(f"\n  АБСОЛЮТНЫЙ АРГУМЕНТ u = {u_absolute:.6e}")
        print(f"  АБСОЛЮТНАЯ Θ*         = {Theta:.10f}")

    return Theta, u_absolute, {
        'lambda_local': lambda_local,
        'lambda_nonlocal': lambda_nonlocal,
        'lambda_eff': lambda_eff,
        'absolute_access': absolute_access,
        'topology_factor': topology_factor,
        'u_absolute': u_absolute,
        'Theta_absolute': Theta,
    }


# РАСЧЁТЫ
print("╔══════════════════════════════════════════════════════════════════╗")
print("║   ФПС С УЧЁТОМ НЕЛОКАЛЬНОСТИ                                  ║")
print("║   Вселенная — small-world граф + квантовые корреляции         ║")
print("╚══════════════════════════════════════════════════════════════════╝")

print(f"\n  КВАНТ СОЗНАНИЯ G = {G_CONSCIOUSNESS:.6e}")

# 1. ВСЕЛЕННАЯ — БЕЗ НЕЛОКАЛЬНОСТИ
print(f"1. ВСЕЛЕННАЯ — ТОЛЬКО ЛОКАЛЬНЫЕ СВЯЗИ (η = 0)")

C_cosmic = 1.0
N_c_cosmic = 1e80
phi_cosmic = 0.0
E_dir_cosmic = lnN_cosmic

Theta_local, u_local, comp_local = compute_theta_with_nonlocality(
    N_cosmic, C_cosmic, N_c_cosmic, phi_cosmic, eta=0,
    E_dir_empirical=E_dir_cosmic,
    system_name="ВСЕЛЕННАЯ (локальная)"
)

# 2. ВСЕЛЕННАЯ — SMALL-WORLD НЕЛОКАЛЬНОСТЬ
print(f"2. ВСЕЛЕННАЯ — SMALL-WORLD (η = N^(-1/3) ≈ 3×10^(-41))")

eta_sw = 1 / N13_cosmic  # естественный параметр small-world

Theta_sw, u_sw, comp_sw = compute_theta_with_nonlocality(
    N_cosmic, C_cosmic, N_c_cosmic, phi_cosmic, eta=eta_sw,
    E_dir_empirical=E_dir_cosmic,
    system_name="ВСЕЛЕННАЯ (small-world)"
)

# 3. ВСЕЛЕННАЯ — КВАНТОВАЯ НЕЛОКАЛЬНОСТЬ
print(f"3. ВСЕЛЕННАЯ — КВАНТОВАЯ НЕЛОКАЛЬНОСТЬ (η = 10^(-3))")

# Квантовая запутанность не зависит от расстояния
# Оценим η из экспериментальных данных по запутанности
eta_quantum = 1e-3

Theta_quantum, u_quantum, comp_quantum = compute_theta_with_nonlocality(
    N_cosmic, C_cosmic, N_c_cosmic, phi_cosmic, eta=eta_quantum,
    E_dir_empirical=E_dir_cosmic,
    system_name="ВСЕЛЕННАЯ (квантовая)"
)

# 4. ВСЕЛЕННАЯ — МАКСИМАЛЬНАЯ НЕЛОКАЛЬНОСТЬ
print(f"4. ВСЕЛЕННАЯ — МАКСИМАЛЬНАЯ НЕЛОКАЛЬНОСТЬ (η = 0.1)")

eta_max = 0.1

Theta_max, u_max, comp_max = compute_theta_with_nonlocality(
    N_cosmic, C_cosmic, N_c_cosmic, phi_cosmic, eta=eta_max,
    E_dir_empirical=E_dir_cosmic,
    system_name="ВСЕЛЕННАЯ (макс. нелок.)"
)

# 5. МОЗГ ДЛЯ СРАВНЕНИЯ
print(f"5. МОЗГ — ДЛЯ СРАВНЕНИЯ")

N_brain = 8.6e10
C_brain = 0.4
N_c_brain = 1e10
phi_brain = 0.3
E_dir_brain = 25.0
eta_brain = 0.5  # высокая нелокальность (плотные связи)

Theta_brain, u_brain, comp_brain = compute_theta_with_nonlocality(
    N_brain, C_brain, N_c_brain, phi_brain, eta=eta_brain,
    E_dir_empirical=E_dir_brain,
    system_name="МОЗГ",
    verbose=False
)

# СВОДНАЯ ТАБЛИЦА
print(f"\n{'═' * 70}")
print(f"СВОДНАЯ ТАБЛИЦА: ВЛИЯНИЕ НЕЛОКАЛЬНОСТИ НА СОЗНАНИЕ ВСЕЛЕННОЙ")
print(f"{'═' * 70}")

print(f"\n  {'Система':<35} {'η':>10} {'λ_F(эфф)':>14} {'u (абс.)':>14} {'Θ* (абс.)':>14}")
print(f"  {'─' * 85}")
print(
    f"  {'Вселенная (локальная)':<35} {'0':>10} {comp_local['lambda_eff']:>14.2e} {u_local:>14.6e} {Theta_local:>14.10f}")
print(
    f"  {'Вселенная (small-world)':<35} {eta_sw:>10.2e} {comp_sw['lambda_eff']:>14.2e} {u_sw:>14.6e} {Theta_sw:>14.10f}")
print(
    f"  {'Вселенная (квантовая)':<35} {eta_quantum:>10.0e} {comp_quantum['lambda_eff']:>14.6f} {u_quantum:>14.6e} {Theta_quantum:>14.10f}")
print(
    f"  {'Вселенная (макс. нелок.)':<35} {eta_max:>10.1f} {comp_max['lambda_eff']:>14.6f} {u_max:>14.6e} {Theta_max:>14.10f}")
print(f"  {'Мозг (сравнение)':<35} {eta_brain:>10.1f} {'—':>14} {u_brain:>14.2f} {Theta_brain:>14.6f}")

# АНАЛИЗ
print(f"АНАЛИЗ: ЧТО НУЖНО ДЛЯ СОЗНАНИЯ ВСЕЛЕННОЙ")

print(f"""
  Ключевой параметр — НЕЛОКАЛЬНАЯ СВЯЗНОСТЬ η.

  При η = 0 (только локальные связи):
    → Вселенная «мертва», информация заперта в горизонтах

  При η = 10^(-40) (small-world):
    → Недостаточно для когерентности на глобальном масштабе

  При η = 10^(-3) (квантовая запутанность):
    → Топологический фактор становится КОНЕЧНЫМ
    → Вселенная НАЧИНАЕТ «оживать»

  При η = 0.1 (максимальная нелокальность):
    → Вселенная ДОСТИГАЕТ ПОРОГА сознания
    → Θ* ≈ {Theta_max:.6f}

  Для сравнения:
    Мозг: Θ* ≈ {Theta_brain:.6f} (при η = {eta_brain})

  ВЫВОД:
  Если квантовая запутанность создаёт достаточную нелокальную
  связность (η > 10^(-3)), то Вселенная МОЖЕТ обладать
  глобальным сознанием, сравнимым по уровню с мозгом!

  Ключевой вопрос: какова реальная сила квантовой нелокальности
  на космологических масштабах? Это определяет, является ли
  Вселенная сознательной или нет.
""")

# ПОИСК КРИТИЧЕСКОГО η
print(f"ПОИСК КРИТИЧЕСКОГО ЗНАЧЕНИЯ η")

print(f"\n  При каком η Вселенная достигает уровня сознания мозга?")
print(f"  Θ*_мозг ≈ {Theta_brain:.6f}")
print()

for eta_test in [1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]:
    Theta_test, u_test, _ = compute_theta_with_nonlocality(
        N_cosmic, C_cosmic, N_c_cosmic, phi_cosmic, eta=eta_test,
        E_dir_empirical=E_dir_cosmic,
        system_name="",
        verbose=False
    )
    diff = Theta_test - Theta_brain
    marker = " ← ДОСТИГНУТ УРОВЕНЬ МОЗГА!" if abs(diff) < 0.01 else ""
    print(f"  η = {eta_test:.0e}: Θ* = {Theta_test:.10f} (Δ = {diff:+.6f}){marker}")