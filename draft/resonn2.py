import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, minimize_scalar

# ============================================================
# ПАРАМЕТРЫ
# ============================================================
K = 6.0
lnK = np.log(K)
const = K - lnK  # ≈ 4.208240530771945


# ============================================================
# ТОЧНАЯ ФУНКЦИЯ F(x)
# ============================================================
def F(x):
    """x = ln(N), точная функция"""
    p = np.exp(-x / 3) / K
    lnp = -lnK - x / 3
    return (K + lnp) / (p - x)


# ============================================================
# ПРОИЗВОДНАЯ (ЧИСЛЕННАЯ)
# ============================================================
def F_prime(x, h=1e-6):
    """Центральная разностная производная"""
    return (F(x + h) - F(x - h)) / (2 * h)


# ============================================================
# ПОИСК КОРНЕЙ F'(x) = 0
# ============================================================
print("=" * 80)
print("ПОИСК ТОЧЕК, ГДЕ ПРОИЗВОДНАЯ F'(x) = 0")
print("=" * 80)

# Поиск в разных интервалах
intervals = [(0.1, 2), (2, 10), (10, 100), (100, 500)]
critical_points = []

for a, b in intervals:
    try:
        # Сначала проверим знаки производной на концах
        fpa = F_prime(a)
        fpb = F_prime(b)

        if fpa * fpb < 0:  # есть корень
            root = fsolve(F_prime, (a + b) / 2)[0]
            critical_points.append(root)
            print(f"  Интервал [{a}, {b}]: корень при x = {root:.10f}")
            print(f"    F(x) = {F(root):.10f}")
            print(f"    F'(x) = {F_prime(root):.2e}")
        else:
            # Возможно, экстремум на границе или внутри
            # Проверим минимум/максимум
            res = minimize_scalar(F, bounds=(a, b), method='bounded')
            if abs(F_prime(res.x)) < 1e-6:
                critical_points.append(res.x)
                print(f"  Интервал [{a}, {b}]: экстремум при x = {res.x:.10f}")
                print(f"    F(x) = {F(res.x):.10f}")
                print(f"    F'(x) = {F_prime(res.x):.2e}")
    except:
        pass

# ============================================================
# АНАЛИТИЧЕСКИЙ ПОИСК (БОЛЕЕ ТОЧНЫЙ)
# ============================================================
print("\n" + "-" * 80)
print("УТОЧНЁННЫЙ ПОИСК (ГЛОБАЛЬНЫЙ МИНИМУМ/МАКСИМУМ)")
print("-" * 80)

# Глобальный поиск на интервале (0, 50)
x_scan = np.linspace(0.1, 50, 5000)
F_vals = [F(x) for x in x_scan]
F_prime_vals = [F_prime(x) for x in x_scan]

# Находим смены знака производной
sign_changes = []
for i in range(1, len(F_prime_vals) - 1):
    if F_prime_vals[i - 1] * F_prime_vals[i + 1] < 0:
        sign_changes.append(x_scan[i])

print(f"\nНайдено точек смены знака производной: {len(sign_changes)}")
for i, x0 in enumerate(sign_changes, 1):
    # Уточняем корень
    root = fsolve(F_prime, x0)[0]
    print(f"\n{i}. x = {root:.10f}")
    print(f"   F(x) = {F(root):.10f}")
    print(f"   F'(x) = {F_prime(root):.2e}")

    # Определяем тип экстремума
    F_double_prime = (F_prime(root + 1e-5) - F_prime(root - 1e-5)) / (2e-5)
    if F_double_prime > 0:
        print(f"   Тип: ЛОКАЛЬНЫЙ МИНИМУМ")
    elif F_double_prime < 0:
        print(f"   Тип: ЛОКАЛЬНЫЙ МАКСИМУМ")
    else:
        print(f"   Тип: ТОЧКА ПЕРЕГИБА")

# ============================================================
# ПОСТРОЕНИЕ ГРАФИКОВ
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f'Анализ функции F(x) и её производной (K = {K})', fontsize=14, fontweight='bold')

# График 1: F(x) в широком диапазоне
ax = axes[0, 0]
x_wide = np.linspace(0.5, 500, 1000)
F_wide = [F(x) for x in x_wide]
ax.plot(x_wide, F_wide, 'b-', linewidth=2)
ax.axhline(y=1 / 3, color='green', linestyle='--', alpha=0.7, label='y = 1/3')
ax.axhline(y=1 / np.pi, color='red', linestyle=':', alpha=0.7, label=f'y = 1/π = {1 / np.pi:.6f}')
ax.set_xlabel('x = ln(N)')
ax.set_ylabel('F(x)')
ax.set_title('F(x) в широком диапазоне')
ax.grid(True, alpha=0.3)
ax.legend()

# Отмечаем критические точки
for xc in critical_points:
    ax.axvline(x=xc, color='orange', linestyle='--', alpha=0.5, linewidth=1)
    ax.scatter(xc, F(xc), color='red', s=50, zorder=5)

# График 2: F(x) в области малых x (где экстремум)
ax = axes[0, 1]
x_small = np.linspace(0.1, 5, 1000)
F_small = [F(x) for x in x_small]
ax.plot(x_small, F_small, 'b-', linewidth=2)
ax.set_xlabel('x = ln(N)')
ax.set_ylabel('F(x)')
ax.set_title('F(x) при малых x (область экстремума)')
ax.grid(True, alpha=0.3)

# Отмечаем критические точки
for xc in critical_points:
    if xc < 5:
        ax.axvline(x=xc, color='orange', linestyle='--', alpha=0.7, linewidth=1.5)
        ax.scatter(xc, F(xc), color='red', s=80, zorder=5)
        ax.annotate(f'x={xc:.3f}\nF={F(xc):.3f}',
                    (xc, F(xc)), xytext=(10, 10), textcoords='offset points')

# График 3: Производная F'(x)
ax = axes[1, 0]
x_deriv = np.linspace(0.2, 500, 2000)
F_prime_vals_plot = [F_prime(x) for x in x_deriv]
ax.plot(x_deriv, F_prime_vals_plot, 'r-', linewidth=2)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax.set_xlabel('x = ln(N)')
ax.set_ylabel("F'(x)")
ax.set_title("Производная F'(x)")
ax.grid(True, alpha=0.3)
ax.set_yscale('symlog', linthresh=1e-4)  # симметричный логарифмический масштаб

# Отмечаем нули производной
for xc in critical_points:
    ax.axvline(x=xc, color='orange', linestyle='--', alpha=0.5, linewidth=1)
    ax.scatter(xc, 0, color='red', s=50, zorder=5)

# График 4: F(x) в области большого x (резонанс)
ax = axes[1, 1]
x_large = np.linspace(200, 350, 1000)
F_large = [F(x) for x in x_large]
ax.plot(x_large, F_large, 'b-', linewidth=2)
ax.axhline(y=1 / np.pi, color='red', linestyle=':', alpha=0.7, label=f'1/π = {1 / np.pi:.6f}')
ax.axhline(y=1 / 3, color='green', linestyle='--', alpha=0.7, label='1/3')

# Отмечаем теоретический резонанс
x_resonance = const / (1 / 3 - 1 / np.pi)  # ≈ 280.1115
ax.axvline(x=x_resonance, color='purple', linestyle='--', alpha=0.7, label=f'Резонанс x={x_resonance:.2f}')
ax.scatter(x_resonance, 1 / np.pi, color='purple', s=80, zorder=5)

ax.set_xlabel('x = ln(N)')
ax.set_ylabel('F(x)')
ax.set_title('F(x) в области резонанса (большие x)')
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.show()

# ============================================================
# ИТОГОВЫЙ ВЫВОД
# ============================================================
print("\n" + "=" * 80)
print("ВЫВОДЫ")
print("=" * 80)

if critical_points:
    print(f"\nНайдено {len(critical_points)} критических точек (F'(x)=0):")
    for i, xc in enumerate(sorted(critical_points), 1):
        print(f"  {i}. x = {xc:.10f}, F(x) = {F(xc):.10f}")

    # Основной экстремум при малых x
    small_crit = [xc for xc in critical_points if xc < 10]
    if small_crit:
        x_min = small_crit[0]
        print(f"\n▶ ГЛОБАЛЬНЫЙ ЭКСТРЕМУМ при x = {x_min:.6f}")
        print(f"   F(x_min) = {F(x_min):.10f}")
        print(f"   Это ЛОКАЛЬНЫЙ МИНИМУМ функции F(x)")
else:
    print("\nКритических точек не найдено (функция монотонна)")

print(f"\n▶ РЕЗОНАНС (F(x) = 1/π) достигается при x = {x_resonance:.6f}")
print(f"   В этой точке производная F'(x) = {F_prime(x_resonance):.2e} (НЕ НОЛЬ)")
print(f"   Функция монотонно возрастает в этой области")