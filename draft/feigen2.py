import numpy as np
from scipy.optimize import minimize_scalar
import warnings

warnings.filterwarnings('ignore')

K = 6.0


def mu_N(lnN):
    """
    Эффективный параметр μ через масштаб графа.
    μ(N) = K * (ln N)^(1/3) / (K + (ln N)^(1/3))
    """
    x = lnN ** (1 / 3)
    return K * x / (K + x)


def logistic(x, mu):
    """Логистическая карта"""
    return mu * x * (1 - x)


def iterate_map(mu, n_iter=10000, n_discard=9000, x0=0.5):
    """
    Итерация логистической карты.
    Возвращает последние значения после discard.
    """
    x = x0
    trajectory = []
    for i in range(n_iter):
        x = logistic(x, mu)
        if i >= n_discard:
            trajectory.append(x)
    return np.array(trajectory)


def detect_period(trajectory, tol=1e-6, max_period=32):
    """
    Определяет период аттрактора.
    Сравнивает последнюю точку с предыдущими.
    """
    if len(trajectory) < max_period * 2:
        return None

    last = trajectory[-1]

    for period in range(1, max_period + 1):
        if abs(trajectory[-1 - period] - last) < tol:
            # Проверяем устойчивость периода
            all_close = True
            for j in range(period):
                if abs(trajectory[-1 - j] - trajectory[-1 - period - j]) > tol * 10:
                    all_close = False
                    break
            if all_close:
                return period

    # Если период не найден, это хаос (или период > max_period)
    if len(set(np.round(trajectory[-100:], 4))) > 20:
        return -1  # хаос
    return None


def find_bifurcation_points(lnN_range, step=0.001):
    """
    Находит точки бифуркаций, сканируя lnN.
    Возвращает список (lnN, mu, period).
    """
    results = []
    prev_period = None

    for lnN in np.arange(lnN_range[0], lnN_range[1], step):
        mu = mu_N(lnN)

        # Для μ < 1 логистическая карта имеет только фиксированную точку 0
        if mu < 1.0:
            period = 1
        elif mu < 3.0:
            period = 1
        else:
            traj = iterate_map(mu, n_iter=5000, n_discard=4000)
            period = detect_period(traj)

        if period is not None and period != prev_period:
            results.append((lnN, mu, period))
            prev_period = period

    return results


def compute_feigenbaum_delta(bifurcations):
    """
    Вычисляет δ_n по последовательности бифуркаций.
    """
    mu_vals = np.array([b[1] for b in bifurcations if b[2] > 0])
    periods = [b[2] for b in bifurcations if b[2] > 0]

    deltas = []
    for i in range(2, len(mu_vals)):
        if mu_vals[i] != mu_vals[i - 1]:
            d = (mu_vals[i - 1] - mu_vals[i - 2]) / (mu_vals[i] - mu_vals[i - 1])
            deltas.append((periods[i], d))

    return deltas


# ============================================================
# ЗАПУСК
# ============================================================
print("=" * 80)
print("ПОИСК ТОЧЕК БИФУРКАЦИИ И ВЫЧИСЛЕНИЕ δ_n")
print("=" * 80)

# Диапазон lnN: μ должен пройти от ~1 до ~4
# μ = 1 при lnN^(1/3) = K/5 ≈ 1.2 → lnN ≈ 1.73
# μ = 3 при lnN^(1/3) = K/1 ≈ 6 → lnN ≈ 216
# μ = 3.5699 (начало хаоса) при lnN^(1/3) = K * 3.5699 / (K - 3.5699)
#   при K=6: x = 6*3.5699/2.4301 ≈ 8.81 → lnN ≈ 684
# μ = 4 при lnN^(1/3) = 2K/2 = 6 → lnN = 216?
#   Проверим: μ = 6*x/(6+x) = 4 → 6x = 24+4x → 2x = 24 → x=12 → lnN = 1728

# Возьмём широкий диапазон
lnN_range = (1.0, 20000.0)
step = 0.01

print(f"\nСканирование lnN ∈ [{lnN_range[0]}, {lnN_range[1]}] с шагом {step}")
print("K = 6")
print("\nОжидаемые точки бифуркации в логистической карте:")
print("  μ₁ = 3.0        (1 → 2)")
print("  μ₂ ≈ 3.44949    (2 → 4)")
print("  μ₃ ≈ 3.54409    (4 → 8)")
print("  μ₄ ≈ 3.56441    (8 → 16)")
print("  μ₅ ≈ 3.56876    (16 → 32)")
print("  μ∞ ≈ 3.5699456  (начало хаоса)\n")

bifurcations = find_bifurcation_points(lnN_range, step)

print(f"Найдено точек бифуркации: {len(bifurcations)}")
print(f"\n{'lnN':>12} {'μ':>12} {'Период':>8}")
print("-" * 35)
for lnN, mu, period in bifurcations[:20]:
    print(f"{lnN:>12.4f} {mu:>12.8f} {period:>8}")

# Вычисляем δ
deltas = compute_feigenbaum_delta(bifurcations)

print(f"\n{'─' * 60}")
print("СХОДИМОСТЬ δ_n → δ_F")
print(f"{'─' * 60}")
print(f"{'Период':>8} {'δ_n':>12} {'δ_F':>12} {'Отклонение %':>15}")
print("-" * 50)

delta_F = 4.669201609102990

for period, d in deltas:
    dev = abs(d - delta_F) / delta_F * 100
    print(f"2^{period:>4} {d:>12.8f} {delta_F:>12.8f} {dev:>15.8f}")

# Более точный поиск
print(f"\n{'─' * 60}")
print("УТОЧНЁННЫЙ ПОИСК (около μ ≈ 3.0, 3.449, 3.544, 3.564, 3.569)")
print(f"{'─' * 60}")


def refined_search(mu_target, delta_mu=0.1, tol=1e-8):
    """
    Уточняет точку бифуркации около mu_target.
    """

    def f(mu):
        traj = iterate_map(mu, n_iter=10000, n_discard=9000)
        period = detect_period(traj)
        return period if period else -1

    mu_left = mu_target - delta_mu
    mu_right = mu_target + delta_mu

    # Бинарный поиск точки смены периода
    for _ in range(40):
        mu_mid = (mu_left + mu_right) / 2
        period_mid = f(mu_mid)
        period_left = f(mu_left)

        if period_mid != period_left:
            mu_right = mu_mid
        else:
            mu_left = mu_mid

    # Возвращаем mu при обратном переходе к N
    mu_bif = (mu_left + mu_right) / 2
    # Решаем mu = K * x / (K + x) относительно x
    # mu*(K+x) = K*x → mu*K + mu*x = K*x → mu*K = K*x - mu*x → x = mu*K/(K-mu)
    x = mu_bif * K / (K - mu_bif)
    lnN = x ** 3

    return mu_bif, lnN


# Ищем первые 5 бифуркаций
mu_targets = [3.0, 3.44949, 3.54409, 3.56441, 3.56876]
refined = []

for mu_t in mu_targets:
    try:
        mu_b, lnN_b = refined_search(mu_t)
        refined.append((lnN_b, mu_b))
        print(f"μ_target ≈ {mu_t:.6f} → μ = {mu_b:.10f}, lnN = {lnN_b:.8f}")
    except Exception as e:
        print(f"Ошибка при μ_target ≈ {mu_t}: {e}")

# Вычисляем δ из уточнённых значений
if len(refined) >= 3:
    print(f"\n{'─' * 60}")
    print("УТОЧНЁННЫЕ ЗНАЧЕНИЯ δ_n")
    print(f"{'─' * 60}")

    mu_ref = np.array([r[1] for r in refined])

    deltas_refined = []
    for i in range(2, len(mu_ref)):
        d = (mu_ref[i - 1] - mu_ref[i - 2]) / (mu_ref[i] - mu_ref[i - 1])
        deltas_refined.append(d)
        dev = abs(d - delta_F) / delta_F * 100
        print(f"δ_{i} = {d:.10f} (отклонение {dev:.8f}%)")

    if len(deltas_refined) >= 3:
        print(f"\nПредельная оценка: δ ≈ {deltas_refined[-1]:.10f}")
        print(f"δ_F = {delta_F:.10f}")
        print(f"Совпадение: {abs(deltas_refined[-1] - delta_F) / delta_F * 100:.8f}%")