import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

# 1. ПАРАМЕТРЫ
L = 200
d = 3
N = L ** d
V = N

print(f"Решётка: {L}×{L}×{L} = {N} узлов")

# 2. СПЕКТР (используем сэмплирование для ускорения)
# Для больших L берём подвыборку в k-пространстве
sample_rate = max(1, L // 130)  # адаптивное сэмплирование
k_vals = 2 * np.pi * np.arange(0, L, sample_rate) / L

print(f"Сэмплирование k-пространства: {len(k_vals)}×{len(k_vals)}×{len(k_vals)} точек")

eigvals = []

for kx in k_vals:
    for ky in k_vals:
        for kz in k_vals:
            lam = 2 * (
                    (1 - np.cos(kx)) +
                    (1 - np.cos(ky)) +
                    (1 - np.cos(kz))
            )
            eigvals.append(lam)

eigvals = np.array(eigvals)
eigvals = eigvals[eigvals > 1e-12]

# Корректируем на сэмплирование
sampling_factor = sample_rate ** d
print(f"Коэффициент сэмплирования: {sampling_factor}")


# 3. ТЕОРЕТИЧЕСКАЯ ФОРМУЛА ДЛЯ НЕПРЕРЫВНОГО ПРИБЛИЖЕНИЯ
def heat_kernel_continuous(t, volume):
    """
    Точная формула для heat kernel в непрерывном R^d
    K(t) = V / (4πt)^(d/2)
    """
    return volume / (4 * np.pi * t) ** (d / 2)


def heat_kernel_discrete(t):
    """Дискретный heat kernel с учётом сэмплирования"""
    return sampling_factor * np.sum(np.exp(-t * eigvals))


# 4. ВЫЧИСЛЕНИЕ ДЛЯ РАЗНЫХ t
t_values = np.logspace(-3, 2, 200)
K_discrete = np.array([heat_kernel_discrete(t) for t in t_values])
K_continuous = np.array([heat_kernel_continuous(t, V) for t in t_values])

# 5. МЕТОД 1: ПРЯМОЕ СРАВНЕНИЕ С НЕПРЕРЫВНОЙ ФОРМУЛОЙ
print("МЕТОД 1: Прямая подгонка")

# Ищем область, где дискретный и непрерывный совпадают
ratio = K_discrete / K_continuous

# Берём область, где отношение стабильно
mask_stable = (t_values > 0.1) & (t_values < 10)  # промежуточные времена
if np.sum(mask_stable) > 10:
    pi_estimates_1 = (V / K_discrete[mask_stable]) ** (2 / d) / (4 * t_values[mask_stable])
    pi_est_1 = np.median(pi_estimates_1)
    print(f"π (метод 1) ≈ {pi_est_1:.6f}")
    print(f"Ошибка: {abs(pi_est_1 - np.pi):.6f}")

# 6. МЕТОД 2: ЛОГАРИФМИЧЕСКАЯ ПОДГОНКА
print("МЕТОД 2: Логарифмическая подгонка")

log_t = np.log(t_values)
log_K = np.log(K_discrete)

# Находим область с наклоном -1.5
slopes = np.gradient(log_K, log_t)
curvature = np.gradient(slopes, log_t)

target_slope = -1.5
mask = (
        (np.abs(slopes - target_slope) < 0.1) &
        (np.abs(curvature) < 0.05)
)

if np.sum(mask) < 5:
    mask = (np.abs(slopes - target_slope) < 0.2)

if np.sum(mask) >= 5:
    # Подгонка: log K = log(V) - (d/2)*log(4π) - (d/2)*log(t)
    coeffs = np.polyfit(log_t[mask], log_K[mask], 1)
    slope_fit, intercept = coeffs

    # Из intercept извлекаем π
    # intercept = log(V) - (d/2)*log(4π)
    log_4pi = (np.log(V) - intercept) * 2 / d
    pi_est_2 = np.exp(log_4pi) / 4

    print(f"Наклон: {slope_fit:.6f} (ожидаем {-d / 2})")
    print(f"π (метод 2) ≈ {pi_est_2:.6f}")
    print(f"Ошибка: {abs(pi_est_2 - np.pi):.6f}")

# 7. МЕТОД 3: ИНТЕГРАЛЬНЫЙ МЕТОД
print("МЕТОД 3: Интегральный метод")


# ∫₀^∞ K(t) dt = V * ∫₀^∞ dt/(4πt)^(d/2) - зависит от размерности
# Для d=3 этот интеграл расходится, используем регуляризацию

def integrand(t):
    return heat_kernel_discrete(t) * t ** (d / 2 - 1)


# Интегрируем в разумных пределах
t_int = np.logspace(-2, 1, 100)
K_int = np.array([integrand(t) for t in t_int])
integral = np.trapz(K_int, t_int)

# Для d=3: ∫₀^∞ t^(1/2) * K(t) dt = V * Γ(1/2) / (4π)^(3/2) = V/(8π)
pi_est_3 = V / (8 * integral)

print(f"π (метод 3) ≈ {pi_est_3:.6f}")
print(f"Ошибка: {abs(pi_est_3 - np.pi):.6f}")

# 8. МЕТОД 4: ПРЯМОЕ ИЗВЛЕЧЕНИЕ ИЗ АСИМПТОТИКИ
print("\n" + "=" * 50)
print("МЕТОД 4: Прямое извлечение из асимптотики")
print("=" * 50)

# Для малых t: K(t) ≈ V/(4πt)^(3/2)
mask_small_t = t_values < 0.1
if np.sum(mask_small_t) > 10:
    A = np.mean(K_discrete[mask_small_t] * t_values[mask_small_t] ** (3 / 2))
    pi_est_4 = (V / A) ** (2 / 3) / 4
    print(f"A = K * t^(3/2) ≈ {A:.6f}")
    print(f"π (метод 4) ≈ {pi_est_4:.6f}")
    print(f"Ошибка: {abs(pi_est_4 - np.pi):.6f}")

# 9. ВИЗУАЛИЗАЦИЯ
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 9.1 Heat kernels
ax = axes[0, 0]
ax.loglog(t_values, K_discrete, 'b-', label='Дискретный', alpha=0.7)
ax.loglog(t_values, K_continuous, 'r--', label='Непрерывный (4πt)^(-3/2)')
ax.set_xlabel('t')
ax.set_ylabel('K(t)')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_title('Heat kernel: дискретный vs непрерывный')

# 9.2 Отношение K_discrete / K_continuous
ax = axes[0, 1]
ax.semilogx(t_values, ratio, 'g-')
ax.axhline(y=1, color='r', linestyle='--')
ax.set_xlabel('t')
ax.set_ylabel('K_discrete / K_continuous')
ax.grid(True, alpha=0.3)
ax.set_title('Отношение к непрерывному приближению')

# 9.3 Локальный наклон
ax = axes[0, 2]
ax.semilogx(t_values, slopes, 'b-', label='d log K / d log t')
ax.axhline(y=-1.5, color='r', linestyle='--', label='-3/2')
if 'mask' in locals():
    ax.semilogx(t_values[mask], slopes[mask], 'ro', markersize=3)
ax.set_xlabel('t')
ax.set_ylabel('slope')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_title('Локальный наклон')

# 9.4 Оценки π разными методами
ax = axes[1, 0]
methods = []
estimates = []
if 'pi_est_1' in locals():
    methods.append('Метод 1')
    estimates.append(pi_est_1)
if 'pi_est_2' in locals():
    methods.append('Метод 2')
    estimates.append(pi_est_2)
if 'pi_est_3' in locals():
    methods.append('Метод 3')
    estimates.append(pi_est_3)
if 'pi_est_4' in locals():
    methods.append('Метод 4')
    estimates.append(pi_est_4)

methods.append('Истина')
estimates.append(np.pi)

colors = ['blue', 'green', 'orange', 'purple', 'red']
ax.bar(methods, estimates, color=colors)
ax.axhline(y=np.pi, color='red', linestyle='--', alpha=0.5)
ax.set_ylabel('π')
ax.set_title('Сравнение оценок π')
ax.grid(True, alpha=0.3)

# 9.5 Сходимость к непрерывному пределу
ax = axes[1, 1]
A_t = K_discrete * t_values ** (3 / 2)
ax.loglog(t_values, A_t, 'b-')
ax.axhline(y=V / (4 * np.pi) ** (3 / 2), color='r', linestyle='--',
           label=f'V/(4π)^(3/2) = {V / (4 * np.pi) ** (3 / 2):.2e}')
ax.set_xlabel('t')
ax.set_ylabel('A = K * t^(3/2)')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_title('Коэффициент A(t)')

# 9.6 Спектр
ax = axes[1, 2]
ax.hist(eigvals, bins=50, density=True, alpha=0.7)
ax.set_xlabel('λ')
ax.set_ylabel('Плотность состояний')
ax.set_title(f'Спектр (первые {len(eigvals)} собственных значений)')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 10. АНАЛИЗ ОШИБОК
print("АНАЛИЗ ОШИБОК")

# Теоретическая поправка для дискретной решётки
# K_disc(t) ≈ K_cont(t) * (1 - t/6 + ...) для малых t
correction_factor = 1 + t_values / 6  # первая поправка решётки
K_corrected = K_continuous * correction_factor

# Находим оптимальное t для оценки
relative_error = np.abs(K_discrete - K_continuous) / K_continuous
best_t_idx = np.argmin(relative_error)
best_t = t_values[best_t_idx]

print(f"Оптимальное t для оценки: {best_t:.6f}")
print(f"Минимальная относительная ошибка: {relative_error[best_t_idx]:.6f}")

# Оценка π при оптимальном t
A_opt = K_discrete[best_t_idx] * best_t ** (3 / 2)
pi_opt = (V / A_opt) ** (2 / 3) / 4
print(f"π (оптимальное t) ≈ {pi_opt:.6f}")
print(f"Ошибка: {abs(pi_opt - np.pi):.6f}")

# Усреднение по нескольким лучшим t
n_best = 20
best_indices = np.argsort(relative_error)[:n_best]
pi_best = np.mean([(V / (K_discrete[i] * t_values[i] ** (3 / 2))) ** (2 / 3) / 4
                   for i in best_indices])
print(f"π (усреднение {n_best} лучших t) ≈ {pi_best:.6f}")
print(f"Ошибка: {abs(pi_best - np.pi):.6f}")