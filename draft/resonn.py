import numpy as np
import matplotlib.pyplot as plt
import math

# ============================================================
# ПАРАМЕТРЫ
# ============================================================
K = 6.0
pi = math.pi
lnK = math.log(K)

# Теоретическое значение из геометрического резонанса
lnN_theory = (K - lnK) / (1.0 / 3.0 - 1.0 / pi)
N_theory = math.exp(lnN_theory)

# Наблюдаемое значение
N_obs = 4.197668e121
lnN_obs = math.log(N_obs)


# ============================================================
# ФУНКЦИЯ F(N) - НЕ ИЗМЕНЯЕТСЯ
# ============================================================
def F(N):
    """F(N) = (K + ln p) / (p - ln N) → 1/π при резонансе"""
    p = 1.0 / (K * N ** (1.0 / 3.0))
    return (K + math.log(p)) / (p - math.log(N))


def safe_F(N):
    """Безопасная версия F(N) с обработкой переполнения"""
    try:
        # Проверяем, не вызовет ли N**(1/3) переполнение
        if N > 1e308:  # 接近 float max
            # Используем логарифмическое вычисление p
            lnp = -math.log(K) - (1.0 / 3.0) * math.log(N)
            p = math.exp(lnp)
        else:
            p = 1.0 / (K * N ** (1.0 / 3.0))

        return (K + math.log(p)) / (p - math.log(N))
    except (OverflowError, ValueError):
        return np.nan


def F_approx(N):
    """Асимптотика: F(N) ≈ 1/3 - (K - lnK)/lnN"""
    return 1.0 / 3.0 - (K - lnK) / math.log(N)


# ============================================================
# ПОСТРОЕНИЕ ГРАФИКА
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.suptitle('Уравнение геометрического резонанса F(N) = 1/π',
             fontsize=16, fontweight='bold')

# ============================================================
# ГРАФИК 1: F(N) в широком диапазоне
# ============================================================
ax = axes[0, 0]

# Генерируем N в логарифмическом масштабе, но значения в безопасных пределах
lnN_range = np.linspace(270, 290, 1000)
N_range = np.exp(lnN_range)

# Вычисляем F(N) безопасно
F_vals = []
F_approx_vals = []
for N in N_range:
    F_vals.append(safe_F(N))
    F_approx_vals.append(F_approx(N))

# Убираем nan значения
mask = ~np.isnan(F_vals)
lnN_masked = lnN_range[mask]
F_masked = np.array(F_vals)[mask]

ax.plot(lnN_masked, F_masked, 'b-', linewidth=2, label='F(N) точное')
ax.plot(lnN_range, F_approx_vals, 'r--', linewidth=1.5, label='F(N) асимптотика')
ax.axhline(y=1 / pi, color='green', linestyle=':', linewidth=2, label=f'1/π = {1 / pi:.6f}')
ax.axhline(y=1 / 3, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='1/3 (предел)')

# Отмечаем точки
ax.axvline(x=lnN_theory, color='orange', linestyle='--', linewidth=1.5, alpha=0.7,
           label=f'ln N_теор = {lnN_theory:.4f}')
ax.axvline(x=lnN_obs, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
           label=f'ln N_набл = {lnN_obs:.4f}')

ax.set_xlabel('ln N', fontsize=12)
ax.set_ylabel('F(N)', fontsize=12)
ax.set_title('F(N) в широком диапазоне', fontsize=13, fontweight='bold')
ax.legend(fontsize=9, loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_ylim(0.1, 0.45)

# ============================================================
# ГРАФИК 2: Окрестность резонанса
# ============================================================
ax = axes[0, 1]
lnN_fine = np.linspace(lnN_theory - 5, lnN_theory + 5, 500)
N_fine = np.exp(lnN_fine)

F_fine = []
for N in N_fine:
    F_fine.append(safe_F(N))

ax.plot(lnN_fine, F_fine, 'b-', linewidth=2)
ax.axhline(y=1 / pi, color='green', linestyle=':', linewidth=2, label=f'1/π')
ax.axvline(x=lnN_theory, color='orange', linestyle='--', linewidth=1.5, label=f'ln N_теор = {lnN_theory:.4f}')
ax.axvline(x=lnN_obs, color='red', linestyle='--', linewidth=1.5, label=f'ln N_набл = {lnN_obs:.4f}')

ax.fill_between(lnN_fine, 1 / pi - 0.001, 1 / pi + 0.001, alpha=0.2, color='green')
ax.set_xlabel('ln N', fontsize=12)
ax.set_ylabel('F(N)', fontsize=12)
ax.set_title('Окрестность резонанса', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# ============================================================
# ГРАФИК 3: Отклонение от 1/π
# ============================================================
ax = axes[1, 0]
deviation = [f - 1 / pi if not np.isnan(f) else np.nan for f in F_fine]

ax.plot(lnN_fine, deviation, 'b-', linewidth=2)
ax.axhline(y=0, color='green', linestyle=':', linewidth=2)
ax.axvline(x=lnN_theory, color='orange', linestyle='--', linewidth=1.5)
ax.axvline(x=lnN_obs, color='red', linestyle='--', linewidth=1.5)

# Отмечаем область |F(N) - 1/π| < 0.001
ax.fill_between(lnN_fine, -0.001, 0.001, alpha=0.2, color='green', label='|F-1/π| < 0.001')

ax.set_xlabel('ln N', fontsize=12)
ax.set_ylabel('F(N) - 1/π', fontsize=12)
ax.set_title('Отклонение от резонанса', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# ============================================================
# ГРАФИК 4: χ²-функционал
# ============================================================
ax = axes[1, 1]
chi_sq = [(f - 1 / pi) ** 2 if not np.isnan(f) else np.inf for f in F_fine]

ax.plot(lnN_fine, chi_sq, 'b-', linewidth=2)
ax.axvline(x=lnN_theory, color='orange', linestyle='--', linewidth=1.5, label=f'ln N_теор = {lnN_theory:.4f}')
ax.axvline(x=lnN_obs, color='red', linestyle='--', linewidth=1.5, label=f'ln N_набл = {lnN_obs:.4f}')

# Находим минимум
chi_sq_array = np.array(chi_sq)
finite_mask = np.isfinite(chi_sq_array)
if np.any(finite_mask):
    min_idx = np.argmin(chi_sq_array[finite_mask])
    ax.plot(lnN_fine[finite_mask][min_idx], chi_sq_array[finite_mask][min_idx],
            'r*', markersize=12, label=f'Минимум при ln N = {lnN_fine[finite_mask][min_idx]:.4f}')

ax.set_yscale('log')
ax.set_xlabel('ln N', fontsize=12)
ax.set_ylabel('(F(N) - 1/π)²', fontsize=12)
ax.set_title('χ²-функционал геометрического резонанса', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================
# ЧИСЛЕННЫЕ ЗНАЧЕНИЯ
# ============================================================
print("=" * 80)
print("ЧИСЛЕННЫЕ ЗНАЧЕНИЯ ГЕОМЕТРИЧЕСКОГО РЕЗОНАНСА")
print("=" * 80)

print(f"\n  Параметры:")
print(f"    K = {K}")
print(f"    ln K = {lnK:.10f}")
print(f"    π = {pi:.10f}")
print(f"    1/π = {1 / pi:.10f}")
print(f"    1/3 = {1 / 3:.10f}")

print(f"\n  Теоретическое решение F(N) = 1/π:")
print(f"    ln N_теор = {lnN_theory:.10f}")
print(f"    N_теор = {N_theory:.6e}")

print(f"\n  Наблюдаемое значение:")
print(f"    ln N_набл = {lnN_obs:.10f}")
print(f"    N_набл = {N_obs:.6e}")

print(f"\n  Отклонения:")
print(f"    Δln N = {abs(lnN_obs - lnN_theory):.10f}")
print(f"    Δln N / ln N = {abs(lnN_obs - lnN_theory) / lnN_obs * 100:.6f}%")

# Вычисляем значения F безопасно
F_obs = safe_F(N_obs)
F_theory = safe_F(N_theory)

print(f"\n  Значения F(N):")
print(f"    F(N_теор) = {F_theory:.10f}")
print(f"    F(N_набл) = {F_obs:.10f}")
print(f"    1/π = {1 / pi:.10f}")
print(f"    |F(N_набл) - 1/π| = {abs(F_obs - 1 / pi):.10f}")