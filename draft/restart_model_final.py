"""
Безразмерная модель графа ГИП — исправленная версия
С физическими ограничениями: p ≤ 1/K (чтобы log не взрывался)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
from scipy.special import entr

# =========================================================
# 1. Безразмерные параметры (калибруются)
# =========================================================

# Физические константы
K = 6.0                # K = 2d, d=3
d = 3.0
eta = 0.1              # нелинейность (crowding)

# Безразмерные комбинации
gamma_tilde = 1.0      # γ / η
alpha_tilde = 1.0      # α / (ηK)
beta_tilde = 0.2       # αβK/(2η) / η
kappa_tilde = 0.1      # κ(K-2d)² / η
T_tilde = 0.05         # T / η
lambda_tilde = 0.01    # λ / η

# 2. Универсальное уравнение состояния

def y_from_Lambda(Lambda_val):
    """Решает y³ + y² - Λ = 0, возвращает y > 0"""
    def eq(y):
        return y**3 + y**2 - Lambda_val

    if Lambda_val < 1e-6:
        return np.sqrt(Lambda_val)     # Λ ≪ 1
    elif Lambda_val > 1e6:
        return Lambda_val**(1/3)       # Λ ≫ 1
    else:
        y0 = np.sqrt(Lambda_val) if Lambda_val < 1 else Lambda_val**(1/3)
        sol = root(eq, y0)
        return sol.x[0] if sol.success else np.nan

# 3. Вычисление физических величин (с защитой)

def compute_quantities(Lambda_val, y=None):
    if y is None:
        y = y_from_Lambda(Lambda_val)

    if np.isnan(y):
        return None

    # Базовые величины
    x = K/(2*eta) * y
    lnN = (K**3)/(4*eta**2) * Lambda_val / gamma_tilde

    # Физическое ограничение: N >= 1
    if lnN < 0:
        lnN = 0.0
    N = np.exp(lnN) if lnN < 700 else np.exp(700)  # защита от overflow

    # p с ограничением: не может быть > 1/K (иначе log(Kp)=0)
    p_raw = K/(2*eta) * y / (N ** (1/d))
    p_max = 0.99 / K   # оставляем запас до сингулярности Kp=1
    p = min(p_raw, p_max)

    # Защита от log(0) и log(1)
    Kp = K * p
    if Kp <= 0:
        log_Kp = -1e-10
    elif abs(Kp - 1.0) < 1e-8:
        log_Kp = 1e-8 if Kp > 1 else -1e-8
    else:
        log_Kp = np.log(Kp)

    # U = lnN / |ln(Kp)|
    if abs(log_Kp) < 1e-10:
        U = 1e10  # сингулярность — большой, но конечный
    else:
        U = lnN / abs(log_Kp)

    # Ограничиваем U разумными пределами
    U = min(U, 1e6)

    f1 = U / np.pi
    d_s = 2.0 / abs(log_Kp / lnN) if lnN > 1e-10 else 0.0
    d_s = min(d_s, 100.0)  # ограничиваем

    return {
        'Λ': Lambda_val,
        'y': y,
        'x': x,
        'lnN': lnN,
        'N': N,
        'p': p,
        'Kp': Kp,
        'log_Kp': log_Kp,
        'U': U,
        'f1': f1,
        'd_s': d_s,
        'K': K
    }

# 4. Безразмерный лагранжиан

def dimensionless_action(Lambda_val, y=None):
    if y is None:
        y = y_from_Lambda(Lambda_val)
    if np.isnan(y):
        return np.nan

    # Получаем p (нужно для энтропии и спектра)
    x = K/(2*eta) * y
    lnN = (K**3)/(4*eta**2) * Lambda_val / gamma_tilde
    if lnN < 0:
        lnN = 0.0
    N = np.exp(lnN) if lnN < 700 else np.exp(700)
    p = K/(2*eta) * y / (N ** (1/d))

    # Члены действия
    S_local = alpha_tilde
    S_long = beta_tilde * y
    S_nonlin = y**2
    S_geom = kappa_tilde * (K - 2*d)**2 / (K**2)
    S_trans = gamma_tilde / y if y > 1e-10 else 1e10

    # Энтропийный член (с защитой)
    if p > 0 and p < 1:
        Hp = entr(p) + entr(1-p)  # -p log p - (1-p) log(1-p)
    else:
        Hp = 0.0
    S_entropy = -T_tilde * K * Hp

    # Спектральный член
    denom = p*K + 1/K
    S_spec = lambda_tilde / denom if denom > 1e-10 else 1e10

    S = S_local + S_long + S_nonlin + S_geom + S_trans + S_entropy + S_spec
    return S

# 5. RG-уравнение

def rg_flow(Lambda_val, y=None):
    if y is None:
        y = y_from_Lambda(Lambda_val)
    if np.isnan(y):
        return np.nan
    x = K/(2*eta) * y
    dx_dt = gamma_tilde / (2*K*x + 6*eta*x**2) if x > 0 else 0
    return dx_dt

# 6. УНИВЕРСАЛЬНАЯ КРИВАЯ y(Λ)

def plot_universal_curve():
    Lambda_vals = np.logspace(-3, 2, 200)  # ограничиваем диапазон
    y_vals = []
    for L in Lambda_vals:
        y = y_from_Lambda(L)
        if not np.isnan(y):
            y_vals.append(y)
        else:
            y_vals.append(np.nan)

    plt.figure(figsize=(8,6))
    plt.loglog(Lambda_vals, y_vals, 'b-', linewidth=2)
    plt.xlabel(r'$\Lambda = \frac{4\eta^2 \gamma \ln N}{K^3}$', fontsize=14)
    plt.ylabel(r'$y = \frac{2\eta x}{K}$', fontsize=14)
    plt.title('Универсальное уравнение состояния сети ГИП', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.axvline(1, color='r', linestyle='--', alpha=0.5, label=r'$\Lambda=1$')
    plt.legend()
    plt.tight_layout()
    plt.show()

# 7. ЗАВИСИМОСТИ ОТ Λ

def plot_quantities_vs_Lambda():
    Lambda_vals = np.logspace(-2, 1, 80)  # ограниченный диапазон
    results = []
    for L in Lambda_vals:
        q = compute_quantities(L)
        if q is not None:
            results.append(q)

    if not results:
        print("Нет данных для построения")
        return

    y_vals = [r['y'] for r in results]
    p_vals = [r['p'] for r in results]
    U_vals = [r['U'] for r in results]
    d_s_vals = [r['d_s'] for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0,0].loglog(Lambda_vals[:len(y_vals)], y_vals, 'b-', linewidth=2)
    axes[0,0].set_xlabel(r'$\Lambda$')
    axes[0,0].set_ylabel(r'$y$')
    axes[0,0].set_title('Параметр нелокальности')
    axes[0,0].grid(True, alpha=0.3)

    axes[0,1].loglog(Lambda_vals[:len(p_vals)], p_vals, 'r-', linewidth=2)
    axes[0,1].set_xlabel(r'$\Lambda$')
    axes[0,1].set_ylabel(r'$p$')
    axes[0,1].set_title('Вероятность дальних связей')
    axes[0,1].grid(True, alpha=0.3)

    axes[1,0].semilogx(Lambda_vals[:len(U_vals)], U_vals, 'g-', linewidth=2)
    axes[1,0].set_xlabel(r'$\Lambda$')
    axes[1,0].set_ylabel(r'$U$')
    axes[1,0].set_title('Модулярный инвариант')
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].axhline(np.pi, color='k', linestyle='--', alpha=0.5, label=r'$\pi$')
    axes[1,0].legend()

    axes[1,1].semilogx(Lambda_vals[:len(d_s_vals)], d_s_vals, 'm-', linewidth=2)
    axes[1,1].set_xlabel(r'$\Lambda$')
    axes[1,1].set_ylabel(r'$d_s$')
    axes[1,1].set_title('Спектральная размерность')
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].axhline(2*np.pi, color='k', linestyle='--', alpha=0.5, label=r'$2\pi$')
    axes[1,1].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("БЕЗРАЗМЕРНАЯ МОДЕЛЬ ГРАФА ГИП (ИСПРАВЛЕННАЯ)")
    print("Универсальное уравнение: y² + y³ = Λ")

    # Диапазон Λ, где модель работает
    Lambda_test = 2.0
    print(f"\nТестовое Λ = {Lambda_test}")
    y0 = y_from_Lambda(Lambda_test)
    print(f"y = {y0:.6f}")

    quant = compute_quantities(Lambda_test, y0)
    if quant:
        print(f"x = {quant['x']:.6f}")
        print(f"lnN = {quant['lnN']:.6f}")
        print(f"N = {quant['N']:.6e}")
        print(f"p = {quant['p']:.6e}")
        print(f"Kp = {quant['Kp']:.6e}")
        print(f"log(Kp) = {quant['log_Kp']:.6f}")
        print(f"U = {quant['U']:.6f}")
        print(f"d_s = {quant['d_s']:.6f}")

        S = dimensionless_action(Lambda_test, y0)
        print(f"Безразмерное действие S = {S:.6f}")

        rg = rg_flow(Lambda_test, y0)
        print(f"RG-поток dx/dt = {rg:.6e}")
    else:
        print("Ошибка: решение не найдено")

    # Графики
    plot_universal_curve()
    plot_quantities_vs_Lambda()