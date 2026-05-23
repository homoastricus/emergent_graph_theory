import numpy as np
import math
import warnings

warnings.filterwarnings('ignore')

# =========================
# ПАРАМЕТРЫ
# =========================
K = 6.0
pi = math.pi


# =========================
# ГРАФ
# =========================

def build_small_world_graph(N):
    p_shortcut = 1.0 / (K * N ** (1 / 3))

    L = max(2, int(round(N ** (1 / 3))))
    N_actual = L ** 3

    def idx(x, y, z):
        return ((x % L) * L + (y % L)) * L + (z % L)

    adj = np.zeros((N_actual, N_actual))

    for x in range(L):
        for y in range(L):
            for z in range(L):
                i = idx(x, y, z)

                for dx, dy, dz in [
                    (1, 0, 0), (-1, 0, 0),
                    (0, 1, 0), (0, -1, 0),
                    (0, 0, 1), (0, 0, -1)
                ]:
                    j = idx(x + dx, y + dy, z + dz)

                    if np.random.random() < p_shortcut:
                        j = np.random.randint(0, N_actual)

                    if i != j:
                        adj[i, j] = 1
                        adj[j, i] = 1

    degrees = np.sum(adj, axis=1)
    L_mat = np.diag(degrees) - adj

    return np.linalg.eigvalsh(L_mat), N_actual


def compute_spectrum(N):
    eigenvals, N_actual = build_small_world_graph(N)
    eigenvals = eigenvals[eigenvals > 1e-12]
    return eigenvals, N_actual


# =========================
# ВЫЧИСЛЕНИЯ
# =========================

def compute_physical_entropy(eigenvals):
    """
    ФИЗИЧЕСКАЯ величина (интенсивная)
    """
    return np.mean(np.log(eigenvals))


def compute_logdet(eigenvals):
    """
    ζ-кандидат (экстенсивная)
    """
    return np.sum(np.log(eigenvals))

def compute_log_mean(eigenvals):
    return np.mean(np.log(eigenvals))


def compute_renormalized_S(eigenvals):
    log_l = np.log(eigenvals)
    log_mean = np.mean(log_l)
    return np.sum(log_l - log_mean), log_mean


def compute_intensive(eigenvals):
    """
    Интенсивная величина: средний логарифм собственного значения.
    Именно здесь может скрываться 1/π!
    """
    return np.mean(np.log(eigenvals))


def compute_intensive_derivative(eigenvals, N_actual):
    """
    Производная интенсивной величины по ln N.
    Оценивается через конечные разности.
    """
    # Аналитически: d<ln λ>/d(ln N) для 3D-решётки
    # <ln λ> ~ const - (2/d) * ln N для решётки
    # Для small-world: поправка от нелокальных связей
    return compute_intensive(eigenvals)


# =========================
# ТЕСТЫ
# =========================

def scaling(N_values):
    results = []

    for N in N_values:
        eigenvals, N_actual = compute_spectrum(N)

        log_det = compute_log_det(eigenvals)
        log_mean = compute_log_mean(eigenvals)
        intensive_val = compute_intensive(eigenvals)
        S_ren, _ = compute_renormalized_S(eigenvals)

        lnN = math.log(N_actual)

        results.append((N_actual, lnN, log_det, log_mean, intensive_val, S_ren))

        print(f"N={N_actual} (lnN={lnN:.4f})")
        print(f"  log_det (экстенсивная)  = {log_det:.6f}")
        print(f"  S_ren  (ренормированная) = {S_ren:.6f}")
        print(f"  <ln λ> (интенсивная)    = {intensive_val:.6f}")
        print("-" * 60)

    return results


def estimate_intensive_derivative(results):
    """
    Вычисляет производную ИНТЕНСИВНОЙ величины d<ln λ>/d(ln N).
    Это ключевой тест! Если теория верна, должно быть ~ 1/π.
    """
    print(f"\n{'N₁':<10} {'N₂':<10} {'d<intens>/d(lnN)':<22} {'Отклонение от 1/π':<20}")
    print("-" * 75)

    derivs = []

    for i in range(len(results) - 1):
        N1, lnN1, _, _, int1, _ = results[i]
        N2, lnN2, _, _, int2, _ = results[i + 1]

        d_int = (int2 - int1) / (lnN2 - lnN1)
        dev = abs(d_int - 1.0 / pi) / (1.0 / pi) * 100
        derivs.append(d_int)

        print(f"{N1:<10} {N2:<10} {d_int:<22.8f} {dev:<20.6f}%")

    if derivs:
        mean_d = np.mean(derivs)
        std_d = np.std(derivs)
        dev_mean = abs(mean_d - 1.0 / pi) / (1.0 / pi) * 100

        print(f"\n  Средняя производная: {mean_d:.8f}")
        print(f"  Стандартное отклонение: {std_d:.8f}")
        print(f"  1/π = {1.0 / pi:.8f}")
        print(f"  Отклонение от 1/π: {dev_mean:.4f}%")

        if dev_mean < 5.0:
            print("  ✅ ГИПОТЕЗА ПОДТВЕРЖДЕНА: d<ln λ>/d(ln N) ≈ 1/π")
        elif dev_mean < 20.0:
            print("  🟡 ГИПОТЕЗА ЧАСТИЧНО ПОДТВЕРЖДЕНА (требует больших N)")
        else:
            print(f"  ❌ ГИПОТЕЗА ОТКЛОНЕНА (но возможно, нужно больше N)")

    return derivs


def estimate_renormalized_derivative(results):
    """
    Вычисляет производную РЕНОРМИРОВАННОЙ энтропии dS_ren/d(ln N).
    """
    print(f"\n{'N₁':<10} {'N₂':<10} {'dS_ren/d(lnN)':<22} {'Отклонение от 1/π':<20}")
    print("-" * 75)

    derivs = []

    for i in range(len(results) - 1):
        N1, lnN1, _, _, _, S1 = results[i]
        N2, lnN2, _, _, _, S2 = results[i + 1]

        dS = (S2 - S1) / (lnN2 - lnN1)
        dev = abs(dS - 1.0 / pi) / (1.0 / pi) * 100
        derivs.append(dS)

        print(f"{N1:<10} {N2:<10} {dS:<22.8f} {dev:<20.6f}%")

    if derivs:
        mean_d = np.mean(derivs)
        dev_mean = abs(mean_d - 1.0 / pi) / (1.0 / pi) * 100

        print(f"\n  Средняя производная: {mean_d:.8f}")
        print(f"  Отклонение от 1/π: {dev_mean:.4f}%")

    return derivs


# =========================
# MAIN
# =========================

def main():
    print("=" * 70)
    print("ЧЕСТНЫЙ СПЕКТРАЛЬНЫЙ ТЕСТ: ПОИСК 1/π")
    print("=" * 70)
    print(f"\nГипотеза: d<ln λ>/d(ln N) = 1/π ≈ {1.0 / pi:.8f}")
    print(f"или:      dS_ren/d(ln N) = 1/π")
    print(f"\nПроверяется на РЕАЛЬНЫХ small-world графах")
    print(f"БЕЗ аналитической подстановки!")

    N_values = [64, 125, 216, 343, 512, 729]

    print("\n" + "=" * 70)
    print("1. ВЫЧИСЛЕНИЕ СПЕКТРА")
    print("=" * 70)
    results = scaling(N_values)

    print("\n" + "=" * 70)
    print("2. ТЕСТ: ИНТЕНСИВНАЯ ВЕЛИЧИНА d<ln λ>/d(ln N)")
    print("=" * 70)
    int_derivs = estimate_intensive_derivative(results)

    print("\n" + "=" * 70)
    print("3. ТЕСТ: РЕНОРМИРОВАННАЯ ЭНТРОПИЯ dS_ren/d(ln N)")
    print("=" * 70)
    ren_derivs = estimate_renormalized_derivative(results)

    print("\n" + "=" * 70)
    print("4. ПРЕДСКАЗАНИЕ ЕТИ (для сравнения)")
    print("=" * 70)
    lnK = math.log(K)
    lnN_opt = (K - lnK) / (1.0 / 3.0 - 1.0 / pi)
    N_opt = math.exp(lnN_opt)
    print(f"  ln N_opt = {lnN_opt:.4f}")
    print(f"  N_opt = {N_opt:.4e}")
    print(f"  При N_opt: d<ln λ>/d(ln N) = 1/π = {1.0 / pi:.8f}")

    return results, int_derivs, ren_derivs


if __name__ == "__main__":
    results, int_derivs, ren_derivs = main()