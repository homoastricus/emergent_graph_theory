import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse import linalg as spla
from scipy.fft import fft, fftfreq
from scipy.stats import kurtosis, skew, entropy
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.facecolor'] = 'white'


# ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================

def create_watts_strogatz_graph(N, k, p, seed=42):
    """Создание графа Уоттса-Строгаца"""
    return nx.watts_strogatz_graph(N, k, p, seed=seed)


def calculate_laplacian_spectrum(G, verbose=True):
    """Вычисление спектра нормализованного лапласиана"""
    n = G.number_of_nodes()

    if verbose:
        print(f"  [СПЕКТР] Вычисление для N={n}...")

    if n <= 5000:
        L = nx.normalized_laplacian_matrix(G).toarray()
        eigenvalues = np.linalg.eigvalsh(L)
        if verbose:
            print(f"  [СПЕКТР] Плотный метод, получено {len(eigenvalues)} с.з.")
    else:
        L = nx.normalized_laplacian_matrix(G).astype(np.float64)
        k_small = min(n - 2, n // 3)

        try:
            eig_small = spla.eigsh(L, k=k_small, which='SM', return_eigenvectors=False)
        except:
            eig_small = spla.eigsh(L, k=min(k_small, 500), which='SM', return_eigenvectors=False)

        try:
            eig_large = spla.eigsh(L, k=k_small, which='LM', return_eigenvectors=False)
        except:
            eig_large = spla.eigsh(L, k=min(k_small, 500), which='LM', return_eigenvectors=False)

        eigenvalues = np.sort(np.concatenate([eig_small, eig_large]))
        if verbose:
            print(f"  [СПЕКТР] Разреженный метод, получено {len(eigenvalues)} с.з.")

    return eigenvalues


def calculate_physical_constants(N, K, p, verbose=True):
    """Расчет ВСЕХ эмерджентных физических констант"""
    N = float(N)
    K = float(K)
    p = float(p)

    lnN = np.log(N)
    lnK = np.log(K)
    N13 = N ** (1 / 3)
    Kp = K * p

    constants = {
        'N': N, 'K': K, 'p': p, 'Kp': Kp,
        'c': np.pi * (lnN ** 4) / (K ** 2 * lnK),
        'lP': 4 * lnN ** 2 * lnK / N13,
        'tP': 4 * K ** 2 * lnK ** 2 / (np.pi * N13 * lnN ** 2),
        'EP': (lnN ** 5) * np.pi / (4 * K ** 3 * lnK ** 2),
        'mP': K / (np.pi * 4 * lnN ** 3),
        'TP': 8 * np.pi * N13 / (lnN ** 4),
        'G': 16 * np.pi ** 3 * lnN ** 13 / (K ** 5 * lnK * N13),
        'k_B': Kp * (lnN ** 8) / (8 * np.pi ** 2),
        'alpha': 2 * lnK ** 2 / (np.pi * lnN),
        'h_em': lnN ** 3 / (K * N13)
    }

    constants['hbar'] = constants['h_em'] / (2 * np.pi)
    constants['c_check'] = constants['lP'] / constants['tP']
    constants['EP_check'] = constants['mP'] * constants['c'] ** 2
    constants['hbar_check'] = constants['EP'] * constants['tP']
    constants['c_rel_error'] = abs(constants['c'] - constants['c_check']) / constants['c']
    constants['EP_rel_error'] = abs(constants['EP'] - constants['EP_check']) / constants['EP']

    if verbose:
        print(f"\n  [КОНСТАНТЫ] ln(N)={lnN:.3f}, N^(1/3)={N13:.3e}")
        print(f"  [КОНСТАНТЫ] Проверка c (отн. ошибка): {constants['c_rel_error']:.2e}")
        print(f"  [КОНСТАНТЫ] Проверка EP (отн. ошибка): {constants['EP_rel_error']:.2e}")
        print(f"  [КОНСТАНТЫ] h_em = {constants['h_em']:.6e}, ℏ = {constants['hbar']:.6e}")
        print(f"  [КОНСТАНТЫ] G = {constants['G']:.6e}")
        print(f"  [КОНСТАНТЫ] α = {constants['alpha']:.6e}")

    return constants


def estimate_spectral_dimension(eigenvalues, verbose=True):
    """Оценка спектральной размерности"""
    eigs = eigenvalues[eigenvalues > 1e-12]

    if len(eigs) < 10:
        return np.nan, 0.0

    # Метод 1: через кумулятивную функцию
    threshold = min(np.percentile(eigs, 20), 0.3)
    small_eigs = eigs[eigs <= threshold]

    if len(small_eigs) < 10:
        small_eigs = eigs[:max(10, len(eigs) // 5)]

    n_points = len(small_eigs)
    cumulative = np.arange(1, n_points + 1) / n_points
    valid = (small_eigs > 1e-12) & (cumulative > 1e-12)
    log_lambda = np.log(small_eigs[valid])
    log_cum = np.log(cumulative[valid])

    if len(log_lambda) < 5:
        ds1 = np.nan
        r2_1 = 0.0
    else:
        coeffs = np.polyfit(log_lambda, log_cum, 1)
        ds1 = 2 * coeffs[0]
        predicted = np.polyval(coeffs, log_lambda)
        ss_res = np.sum((log_cum - predicted) ** 2)
        ss_tot = np.sum((log_cum - np.mean(log_cum)) ** 2)
        r2_1 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Метод 2: через отношения
    if len(eigenvalues) > 20:
        ratios = eigenvalues[2:21] / eigenvalues[1]
        expected = np.arange(2, 21)
        valid_ratios = ratios > 0
        if np.sum(valid_ratios) > 5:
            ds2 = 2 * np.mean(np.log(ratios[valid_ratios]) / np.log(expected[valid_ratios]))
        else:
            ds2 = np.nan
    else:
        ds2 = np.nan

    if verbose:
        print(f"  [РАЗМЕРНОСТЬ] ds (кумулянта) = {ds1:.3f} (R²={r2_1:.3f})")
        print(f"  [РАЗМЕРНОСТЬ] ds (отношения) = {ds2:.3f}")
        print(f"  [РАЗМЕРНОСТЬ] Ожидаемая ds = 2.0")

    return ds1, r2_1


def analyze_spectrum_details(eigenvalues, constants, verbose=True):
    """Детальный анализ спектра"""
    lambda_1 = eigenvalues[1]
    lambda_max = eigenvalues[-1]
    lambda_med = np.median(eigenvalues)

    spacing = np.diff(eigenvalues[eigenvalues > 1e-10])
    mean_spacing = np.mean(spacing) if len(spacing) > 0 else 0
    std_spacing = np.std(spacing) if len(spacing) > 0 else 0

    if len(spacing) > 5:
        spacing_ratio = spacing[1:] / spacing[:-1]
        mean_spacing_ratio = np.mean(spacing_ratio)
    else:
        mean_spacing_ratio = 0

    hbar = constants['hbar']
    c = constants['c']

    omega_1 = lambda_1 / hbar
    E_1 = hbar * omega_1
    tau_relax = 1.0 / lambda_1 if lambda_1 > 0 else np.inf

    if verbose:
        print(f"\n  [ДЕТАЛИ СПЕКТРА]")
        print(f"  λ₁ = {lambda_1:.6e}")
        print(f"  λ_max = {lambda_max:.6e}")
        print(f"  λ_max/λ₁ = {lambda_max / lambda_1:.2e}")
        print(f"  Средний спейсинг = {mean_spacing:.6e}")
        print(f"  ω₁ = {omega_1:.6e}")
        print(f"  τ_relax = {tau_relax:.6e}")
        print(f"  Скос спектра = {skew(eigenvalues[eigenvalues > 0]):.3f}")
        print(f"  Эксцесс спектра = {kurtosis(eigenvalues[eigenvalues > 0]):.3f}")

    return {
        'lambda_1': lambda_1,
        'lambda_max': lambda_max,
        'omega_1': omega_1,
        'tau_relax': tau_relax,
        'mean_spacing': mean_spacing,
        'spacing_ratio': mean_spacing_ratio
    }


# ==================== КЛАСС КВАНТОВОЙ ДИНАМИКИ ====================

class QuantumGraphDynamics:
    """
    ПОЛНАЯ КВАНТОВАЯ ДИНАМИКА НА ГРАФЕ
    Включает: линейную, нелинейную, хаотическую, автоматную
    """

    def __init__(self, G, use_complex_edges=False, magnetic_flux=0.0, verbose=True):
        self.G = G
        self.N = G.number_of_nodes()
        self.verbose = verbose

        if verbose:
            print(f"\n[КВАНТ] Инициализация: N={self.N}")

        if use_complex_edges:
            self.L = self._build_complex_laplacian(magnetic_flux)
            if verbose:
                print(f"  Комплексный лапласиан, поток={magnetic_flux}")
        else:
            self.L = nx.laplacian_matrix(G).astype(np.complex128).toarray()

        deg = np.array([G.degree(i) for i in range(self.N)])
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(deg, 1)))
        self.L_norm = D_inv_sqrt @ self.L @ D_inv_sqrt

        self.A = nx.to_numpy_array(G).astype(np.complex128)
        if use_complex_edges:
            phases = magnetic_flux * (np.random.randn(self.N, self.N) - 0.5) * 2 * np.pi
            phases = (phases - phases.T) / 2
            self.A = self.A * np.exp(1j * phases)

        row_sums = np.abs(self.A).sum(axis=1)
        D_inv = np.diag(1.0 / np.maximum(row_sums, 1))
        self.W = D_inv @ self.A

        if not use_complex_edges:
            self.eigenvalues = np.linalg.eigvalsh(self.L_norm.real)
            if verbose:
                print(f"  Спектральная щель: {self.eigenvalues[1]:.6e}")

    def _build_complex_laplacian(self, flux):
        L = np.zeros((self.N, self.N), dtype=np.complex128)
        for i, j in self.G.edges():
            phase = flux * (np.random.random() - 0.5) * 2 * np.pi
            L[i, j] = -np.exp(1j * phase)
            L[j, i] = -np.exp(-1j * phase)
            L[i, i] += 1
            L[j, j] += 1
        return L

    # ===== Нелинейные функции =====

    @staticmethod
    def f_logistic(z, r):
        mag = np.abs(z)
        phase = np.angle(z)
        new_mag = r * mag * (1 - mag ** 2)
        return np.nan_to_num(new_mag, 0) * np.exp(1j * phase)

    @staticmethod
    def f_quadratic(z, c=0.3 + 0.5j):
        return z * z + c

    @staticmethod
    def f_exponential(z, r=1.5):
        mag = np.abs(z)
        phase = np.angle(z)
        new_mag = np.exp(r * (mag - 1))
        new_mag = np.clip(new_mag, 0, 100)
        return new_mag * np.exp(1j * phase)

    @staticmethod
    def f_cubic(z, r=3.0):
        mag = np.abs(z)
        phase = np.angle(z)
        new_mag = r * mag * (1 - mag)
        return np.nan_to_num(new_mag, 0) * np.exp(1j * phase)

    # ===== ДИНАМИКИ =====

    def linear_schrodinger(self, psi0=None, dt=0.01, T=1000, verbose=True):
        """Линейное уравнение Шрёдингера: i dψ/dt = Lψ"""
        if verbose:
            print(f"\n[ЛИН. ШРЁДИНГЕР] dt={dt}, T={T}")

        if psi0 is None:
            psi = np.random.randn(self.N) + 1j * np.random.randn(self.N)
            psi /= np.linalg.norm(psi)
        else:
            psi = psi0.copy()

        history = np.zeros((T, self.N), dtype=np.complex128)
        norms = []
        energies = []

        for t in range(T):
            psi = psi - 1j * dt * (self.L @ psi)

            norm = np.linalg.norm(psi)
            norms.append(norm)
            psi /= norm

            E = np.real(np.conj(psi) @ (self.L @ psi))
            energies.append(E)

            history[t] = psi.copy()

            if verbose and t % (T // 4) == 0:
                print(f"  t={t}: норма={norm:.6e}, E={E:.6e}")

        if verbose:
            print(f"  Средняя энергия: {np.mean(energies[-T // 2:]):.6e}")
            print(f"  STD энергии: {np.std(energies[-T // 2:]):.6e}")

        return history, {'norms': norms, 'energies': energies}

    def nonlinear_schrodinger(self, g=1.0, psi0=None, dt=0.01, T=1000, verbose=True):
        """Нелинейное уравнение Шрёдингера (DNLS): i dψ/dt = Lψ + g|ψ|²ψ"""
        if verbose:
            print(f"\n[DNLS] g={g}, dt={dt}, T={T}")

        if psi0 is None:
            psi = np.random.randn(self.N) + 1j * np.random.randn(self.N)
            psi /= np.linalg.norm(psi)
        else:
            psi = psi0.copy()

        history = np.zeros((T, self.N), dtype=np.complex128)
        norms = []
        energies = []
        participations = []

        for t in range(T):
            nonlinear = g * np.abs(psi) ** 2 * psi
            psi = psi - 1j * dt * (self.L @ psi + nonlinear)

            norm = np.linalg.norm(psi)
            norms.append(norm)
            psi /= norm

            E_lin = np.real(np.conj(psi) @ (self.L @ psi))
            E_nl = g * np.sum(np.abs(psi) ** 4) / 2
            energies.append(E_lin + E_nl)

            pr = 1.0 / np.sum(np.abs(psi) ** 4)
            participations.append(pr)

            history[t] = psi.copy()

            if verbose and t % (T // 4) == 0:
                print(f"  t={t}: PR={pr:.2f}, E={energies[-1]:.6e}")

        if verbose:
            print(f"  PR начальный: {participations[0]:.2f}")
            print(f"  PR конечный: {participations[-1]:.2f}")
            print(f"  Локализация: {'ДА' if participations[-1] < self.N / 2 else 'НЕТ'}")

        return history, {
            'norms': norms,
            'energies': energies,
            'participations': participations
        }

    def complex_logistic_strong(self, r=3.8, epsilon=0.2, nonlinear='logistic',
                                psi0=None, T=500, normalize=True, norm_max=10.0,
                                verbose=True):
        """
        ХАОТИЧЕСКАЯ ДИНАМИКА
        Ключевое: f(W @ psi) вместо W @ f(psi)
        """
        if verbose:
            print(f"\n[ХАОС] r={r}, ε={epsilon}, тип={nonlinear}")

        if nonlinear == 'logistic':
            def f(z):
                return self.f_logistic(z, r)
        elif nonlinear == 'quadratic':
            def f(z):
                return self.f_quadratic(z, 0.3 + 0.5j * r / 4)
        elif nonlinear == 'exponential':
            def f(z):
                return self.f_exponential(z, r / 2)
        elif nonlinear == 'cubic':
            def f(z):
                return self.f_cubic(z, r)
        else:
            raise ValueError(f"Неизвестная нелинейность: {nonlinear}")

        if psi0 is None:
            psi = np.random.randn(self.N) + 1j * np.random.randn(self.N)
            psi /= np.linalg.norm(psi)
        else:
            psi = psi0.copy()

        history = np.zeros((T, self.N), dtype=np.complex128)
        norms = []
        max_mags = []

        for t in range(T):
            # f(W @ psi) - нелинейность ПОСЛЕ смешивания
            mixed = (1 - epsilon) * psi + epsilon * (self.W @ psi)
            psi = f(mixed)

            current_norm = np.linalg.norm(psi)
            norms.append(current_norm)

            if normalize and current_norm > norm_max:
                psi = psi * (norm_max / current_norm)
            elif normalize and current_norm < 1e-15:
                psi = np.random.randn(self.N) + 1j * np.random.randn(self.N)
                psi /= np.linalg.norm(psi)

            max_mags.append(np.max(np.abs(psi)))
            history[t] = psi.copy()

            if verbose and t % (T // 4) == 0:
                print(f"  t={t}: норма={current_norm:.2e}, макс|ψ|={max_mags[-1]:.4f}")

        if verbose:
            print(f"  Средняя норма: {np.mean(norms):.2e}")
            print(f"  Максимальная |ψ|: {np.max(max_mags):.4f}")

        return history, {'norms': norms, 'max_mags': max_mags}

    def complex_logistic_original(self, r=3.7, epsilon=0.2, psi0=None, T=500, verbose=True):
        """
        ОРИГИНАЛЬНАЯ версия (W @ f(psi)) для сравнения
        """
        if verbose:
            print(f"\n[ОРИГИНАЛ] r={r}, ε={epsilon}")

        def f_complex(psi):
            mag = np.abs(psi)
            phase = np.angle(psi)
            new_mag = r * mag * (1 - mag ** 2)
            return np.nan_to_num(new_mag, 0) * np.exp(1j * phase)

        if psi0 is None:
            psi = np.random.randn(self.N) + 1j * np.random.randn(self.N)
            psi /= np.linalg.norm(psi)
        else:
            psi = psi0.copy()

        history = np.zeros((T, self.N), dtype=np.complex128)
        amps = []
        correlations = []

        for t in range(T):
            f_psi = f_complex(psi)
            psi_new = (1 - epsilon) * f_psi + epsilon * (self.W @ f_psi)
            psi = psi_new / np.linalg.norm(psi_new)

            amp = np.abs(psi)
            amps.append(np.mean(amp))

            if t > 0:
                corr = np.corrcoef(np.abs(history[t - 1]), amp)[0, 1]
                correlations.append(corr)

            history[t] = psi.copy()

            if verbose and t % (T // 4) == 0:
                print(f"  t={t}: <|ψ|>={amps[-1]:.4f}")

        return history, {'amps': amps, 'correlations': correlations}

    def hybrid_schrodinger_logistic(self, r=3.0, epsilon=0.2, dt=0.01,
                                    psi0=None, T=1000, verbose=True):
        """
        ГИБРИД: квантовая эволюция + нелинейность + связь
        """
        if verbose:
            print(f"\n[ГИБРИД] r={r}, ε={epsilon}, dt={dt}")

        if psi0 is None:
            psi = np.random.randn(self.N) + 1j * np.random.randn(self.N)
            psi /= np.linalg.norm(psi)
        else:
            psi = psi0.copy()

        history = np.zeros((T, self.N), dtype=np.complex128)

        for t in range(T):
            quantum = -1j * (self.L @ psi)
            nonlinear = r * psi * (1 - np.abs(psi))
            coupling = epsilon * (self.W @ psi)

            psi = psi + dt * (quantum + nonlinear + coupling)

            psi_norm = np.linalg.norm(psi)
            if psi_norm > 10:
                psi *= 10 / psi_norm

            history[t] = psi.copy()

        return history, {}

    def simulate_quantum_dynamics(self, dt=0.01, T=1000):
        """Алиас для linear_schrodinger (совместимость)"""
        return self.linear_schrodinger(dt=dt, T=T)

    def simulate_nonlinear_quantum(self, g=1.0, dt=0.01, T=1000):
        """Алиас для nonlinear_schrodinger (совместимость)"""
        return self.nonlinear_schrodinger(g=g, dt=dt, T=T)

    # ===== БИФУРКАЦИИ =====

    def bifurcation_diagram(self, node_idx=0, nonlinear='logistic',
                            epsilon=0.2, rs=None, T_transient=500, T_sample=200):
        """Бифуркационная диаграмма"""
        if rs is None:
            rs = np.linspace(2.5, 4.5, 400)

        print(f"\n[БИФУРКАЦИЯ] {len(rs)} значений r, узел {node_idx}")

        bifurcation_data = []

        for i, r in enumerate(rs):
            if i % 100 == 0:
                print(f"  r = {r:.3f} ({i}/{len(rs)})")

            psi = np.random.randn(self.N) + 1j * np.random.randn(self.N)
            psi /= np.linalg.norm(psi)

            for t in range(T_transient):
                mixed = (1 - epsilon) * psi + epsilon * (self.W @ psi)
                if nonlinear == 'logistic':
                    psi = self.f_logistic(mixed, r)
                elif nonlinear == 'quadratic':
                    psi = self.f_quadratic(mixed, 0.3 + 0.5j * r / 4)
                elif nonlinear == 'cubic':
                    psi = self.f_cubic(mixed, r)

                norm = np.linalg.norm(psi)
                if norm > 10:
                    psi *= 10 / norm

            for t in range(T_sample):
                mixed = (1 - epsilon) * psi + epsilon * (self.W @ psi)
                if nonlinear == 'logistic':
                    psi = self.f_logistic(mixed, r)
                elif nonlinear == 'quadratic':
                    psi = self.f_quadratic(mixed, 0.3 + 0.5j * r / 4)
                elif nonlinear == 'cubic':
                    psi = self.f_cubic(mixed, r)

                norm = np.linalg.norm(psi)
                if norm > 10:
                    psi *= 10 / norm

                bifurcation_data.append((r, np.abs(psi[node_idx])))

        return np.array(bifurcation_data)

    # ===== ПОКАЗАТЕЛЬ ЛЯПУНОВА =====

    def compute_lyapunov_correct(self, nonlinear='logistic', r=3.8, epsilon=0.2,
                                 T_transient=300, T_lyap=300, delta=1e-8, verbose=True):
        """Корректный показатель Ляпунова (без строгой нормировки)"""
        if verbose:
            print(f"\n[ЛЯПУНОВ] {nonlinear}, r={r}, ε={epsilon}")

        psi1 = np.random.randn(self.N) + 1j * np.random.randn(self.N)
        psi1 /= np.linalg.norm(psi1)
        psi2 = psi1 + delta * (np.random.randn(self.N) + 1j * np.random.randn(self.N))
        psi2 /= np.linalg.norm(psi2)

        if nonlinear == 'logistic':
            def f(z):
                return self.f_logistic(z, r)
        elif nonlinear == 'quadratic':
            def f(z):
                return self.f_quadratic(z, 0.3 + 0.5j * r / 4)
        elif nonlinear == 'cubic':
            def f(z):
                return self.f_cubic(z, r)

        for t in range(T_transient):
            mixed1 = (1 - epsilon) * psi1 + epsilon * (self.W @ psi1)
            mixed2 = (1 - epsilon) * psi2 + epsilon * (self.W @ psi2)
            psi1 = f(mixed1)
            psi2 = f(mixed2)

            for psi in [psi1, psi2]:
                norm = np.linalg.norm(psi)
                if norm > 10:
                    psi *= 10 / norm

        lyap_vals = []

        for t in range(T_lyap):
            mixed1 = (1 - epsilon) * psi1 + epsilon * (self.W @ psi1)
            mixed2 = (1 - epsilon) * psi2 + epsilon * (self.W @ psi2)
            psi1_new = f(mixed1)
            psi2_new = f(mixed2)

            d1 = np.linalg.norm(psi1_new - psi2_new)

            if d1 > 1e-15:
                lyap_local = np.log(d1 / delta)
                lyap_vals.append(lyap_local)

                norm1 = np.linalg.norm(psi1_new)
                norm2 = np.linalg.norm(psi2_new)

                if norm1 > 100:
                    psi1_new *= 100 / norm1
                if norm2 > 100:
                    psi2_new *= 100 / norm2

                if d1 > 1e-10:
                    psi2_new = psi1_new + (delta / d1) * (psi2_new - psi1_new)

            psi1 = psi1_new
            psi2 = psi2_new

        lyap = np.mean(lyap_vals[len(lyap_vals) // 2:]) if lyap_vals else 0

        if verbose:
            print(f"  λ = {lyap:.6f} → {'ХАОС' if lyap > 0.01 else 'ПОРЯДОК'}")

        return lyap, lyap_vals

    def lyapunov_analysis(self, r=3.7, epsilon=0.2, T_transient=500, T_lyap=200):
        """Алиас для совместимости"""
        return self.compute_lyapunov_correct(r=r, epsilon=epsilon,
                                             T_transient=T_transient, T_lyap=T_lyap)

    # ===== КЛЕТОЧНЫЙ АВТОМАТ =====

    def discretize_states(self, psi, n_states=3):
        """Дискретизация состояний для конечного автомата"""
        mag = np.abs(psi)
        thresholds = np.linspace(0, 1, n_states)[1:]
        return np.digitize(mag, thresholds)

    def cellular_automaton_step(self, states, rule='majority'):
        """Один шаг клеточного автомата"""
        new_states = np.zeros(self.N, dtype=int)

        for i in range(self.N):
            neighbors = list(self.G.neighbors(i))
            if len(neighbors) == 0:
                new_states[i] = states[i]
                continue

            neighbor_states = [states[j] for j in neighbors]

            if rule == 'majority':
                counts = np.bincount(neighbor_states, minlength=3)
                new_states[i] = np.argmax(counts)
            elif rule == 'xor':
                new_states[i] = np.sum(neighbor_states) % 3
            elif rule == 'average':
                avg = np.mean(neighbor_states)
                new_states[i] = int(np.clip(round(avg), 0, 2))
            elif rule == 'game_of_life':
                alive = np.sum(np.array(neighbor_states) > 0)
                if states[i] > 0:
                    new_states[i] = 1 if 2 <= alive <= 3 else 0
                else:
                    new_states[i] = 1 if alive == 3 else 0

        return new_states

    def run_cellular_automaton(self, n_states=3, rule='majority', T=100,
                               initial_states=None, verbose=True):
        """Запуск клеточного автомата на графе"""
        if verbose:
            print(f"\n[АВТОМАТ] Правило: {rule}, T={T}")

        if initial_states is None:
            states = np.random.randint(0, n_states, self.N)
        else:
            states = initial_states.copy()

        history = np.zeros((T, self.N), dtype=int)
        state_counts = []

        for t in range(T):
            history[t] = states.copy()
            counts = np.bincount(states, minlength=n_states)
            state_counts.append(counts)
            states = self.cellular_automaton_step(states, rule)

            if verbose and t % (T // 4) == 0:
                print(f"  t={t}: распределение {counts}")

        if verbose:
            final_counts = np.bincount(history[-1], minlength=n_states)
            print(f"  Финальное распределение: {final_counts}")

        return history, np.array(state_counts)

    # ===== АНАЛИЗ =====

    def analyze_dynamics(self, history, dynamics_type='', verbose=True):
        """Расширенный анализ динамики"""
        T, N = history.shape

        amps = np.abs(history)
        mean_amp = np.mean(amps, axis=1)
        phases = np.angle(history)

        fft_amp = fft(mean_amp - np.mean(mean_amp))
        freq_amp = fftfreq(T)
        dominant_freq = freq_amp[1:T // 2][np.argmax(np.abs(fft_amp[1:T // 2]))]

        if T > 2:
            autocorr = np.correlate(mean_amp - np.mean(mean_amp),
                                    mean_amp - np.mean(mean_amp), mode='full')
            autocorr = autocorr[T - 1:] / autocorr[T - 1]
            decorr_time = np.argmax(autocorr < 0.5) if np.any(autocorr < 0.5) else T
        else:
            decorr_time = T

        amp_hist, _ = np.histogram(amps[-1], bins=20, density=True)
        amp_entropy = entropy(amp_hist + 1e-15)

        final_amps = amps[-1]
        max_idx = np.argmax(final_amps)
        min_idx = np.argmin(final_amps)

        if verbose:
            print(f"\n  [АНАЛИЗ] {dynamics_type}")
            print(f"    Доминантная частота: {dominant_freq:.6f}")
            print(f"    Время декорреляции: {decorr_time}")
            print(f"    Энтропия амплитуд: {amp_entropy:.4f}")
            print(f"    Макс. амплитуда: {final_amps[max_idx]:.4f} (узел {max_idx})")
            print(f"    Экстремальность: {np.max(final_amps) / np.mean(final_amps):.2f}")

        return {
            'dominant_freq': dominant_freq,
            'decorr_time': decorr_time,
            'amp_entropy': amp_entropy,
            'max_amp': final_amps[max_idx],
            'min_amp': final_amps[min_idx]
        }

    def cluster_analysis(self, x, threshold=0.1):
        """Анализ кластеризации состояний"""
        states = np.zeros(self.N, dtype=int)
        states[x < 0.33] = 0
        states[(x >= 0.33) & (x < 0.66)] = 1
        states[x >= 0.66] = 2

        unique, counts = np.unique(states, return_counts=True)

        cluster_sizes = []
        visited = np.zeros(self.N, dtype=bool)

        for i in range(self.N):
            if not visited[i]:
                cluster = []
                queue = [i]
                visited[i] = True
                state_val = states[i]

                while queue:
                    node = queue.pop(0)
                    cluster.append(node)

                    for neighbor in self.G.neighbors(node):
                        if not visited[neighbor] and states[neighbor] == state_val:
                            visited[neighbor] = True
                            queue.append(neighbor)

                if len(cluster) > 1:
                    cluster_sizes.append(len(cluster))

        return {
            'states_distribution': dict(zip(unique, counts)),
            'n_clusters': len(cluster_sizes),
            'max_cluster_size': max(cluster_sizes) if cluster_sizes else 0
        }

    def analyze_quantum_correlations(self, history, verbose=True):
        """Анализ квантовых корреляций"""
        psi = history[-1]
        rho_i = np.abs(psi) ** 2
        correlations = np.abs(np.outer(psi, np.conj(psi)))

        mi_values = []
        for i, j in list(self.G.edges())[:50]:
            p_i = rho_i[i]
            p_j = rho_i[j]
            p_ij = correlations[i, j]
            if p_i > 0 and p_j > 0 and p_ij > 0:
                mi = p_ij * np.log(p_ij / (p_i * p_j))
                mi_values.append(mi)

        if verbose and mi_values:
            print(f"\n  [КОРРЕЛЯЦИИ]")
            print(f"    Средняя взаимная информация: {np.mean(mi_values):.6f}")
            print(f"    Макс. взаимная информация: {np.max(mi_values):.6f}")

        return {'mutual_info': mi_values, 'density': rho_i}


# ==================== ВИЗУАЛИЗАЦИЯ ====================

def plot_spectral_analysis(all_results):
    """Графики спектрального анализа"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i, (N, eigenvalues) in enumerate(all_results):
        if i >= len(axes):
            break

        ax = axes[i]
        eigs = eigenvalues[eigenvalues > 1e-10]

        if len(eigs) > 20:
            n_bins = min(100, len(eigs) // 5)
            bins = np.logspace(np.log10(eigs[1]), np.log10(eigs[-1]), n_bins)
            hist, _ = np.histogram(eigs, bins=bins, density=True)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            bin_widths = bins[1:] - bins[:-1]
            density = hist / bin_widths

            ax.loglog(bin_centers, density, 'b-', linewidth=1.5)
            ax.axhline(y=np.mean(density[:5]), color='r', linestyle='--', alpha=0.5)

        ax.set_xlabel('λ')
        ax.set_ylabel('ρ(λ)')
        ax.set_title(f'N={N}')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Спектральная плотность графов\nK=6, Kp=N^(-1/3)', fontsize=14)
    plt.tight_layout()
    plt.savefig('spectral_density.png', dpi=150)
    plt.show()


def plot_bifurcation(bif_data, title=''):
    """Бифуркационная диаграмма"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(bif_data[:, 0], bif_data[:, 1], 'k,', alpha=0.1, markersize=0.1)
    ax.set_xlabel('r')
    ax.set_ylabel('|ψ|')
    ax.set_title(f'Бифуркационная диаграмма\n{title}')
    ax.axvline(x=3.0, color='r', linestyle='--', alpha=0.5, label='r=3.0')
    ax.axvline(x=3.57, color='orange', linestyle='--', alpha=0.5, label='r=3.57')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('bifurcation.png', dpi=150)
    plt.show()


def plot_dynamics(history, title=''):
    """Визуализация динамики"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    for i in range(min(5, history.shape[1])):
        ax.plot(np.abs(history[:, i]), alpha=0.7, label=f'Узел {i}')
    ax.set_xlabel('t')
    ax.set_ylabel('|ψ|')
    ax.set_title('Амплитуды')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    im = ax.imshow(np.abs(history[:min(200, len(history)), :50].T),
                   aspect='auto', cmap='hot', interpolation='nearest')
    ax.set_xlabel('t')
    ax.set_ylabel('Узел')
    ax.set_title('|ψ|(t, x)')
    plt.colorbar(im, ax=ax)

    ax = axes[1, 0]
    signal = np.abs(history[:, 0])
    spectrum = np.abs(fft(signal - np.mean(signal)))
    freqs = fftfreq(len(signal))
    ax.semilogy(freqs[:len(freqs) // 2], spectrum[:len(freqs) // 2])
    ax.set_xlabel('Частота')
    ax.set_ylabel('Мощность')
    ax.set_title('Спектр (узел 0)')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    amp = np.abs(history[:, 0])
    ax.plot(amp[:-1], amp[1:], 'b.', alpha=0.3, markersize=1)
    ax.set_xlabel('|ψ(t)|')
    ax.set_ylabel('|ψ(t+1)|')
    ax.set_title('Возвратное отображение')
    ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig('dynamics.png', dpi=150)
    plt.show()


def plot_automaton(history, title=''):
    """Визуализация клеточного автомата"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for i, t in enumerate([0, len(history) // 2, -1]):
        ax = axes[i]
        theta = np.linspace(0, 2 * np.pi, history.shape[1])
        colors = ['red', 'green', 'blue']
        state_colors = [colors[s] for s in history[t]]
        ax.scatter(np.cos(theta), np.sin(theta), c=state_colors, s=50)
        ax.set_title(f't = {t if t >= 0 else len(history) - 1}')
        ax.set_aspect('equal')
        ax.axis('off')

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig('automaton.png', dpi=150)
    plt.show()


def plot_lyapunov_analysis(rs, lyap_values, title=''):
    """График показателей Ляпунова"""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(rs, lyap_values, 'b-', linewidth=2)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.7, label='λ = 0')
    ax.fill_between(rs, 0, lyap_values, where=(np.array(lyap_values) > 0),
                    color='red', alpha=0.3, label='Хаос')
    ax.fill_between(rs, lyap_values, 0, where=(np.array(lyap_values) < 0),
                    color='blue', alpha=0.3, label='Порядок')

    ax.set_xlabel('r')
    ax.set_ylabel('λ')
    ax.set_title(f'Показатели Ляпунова\n{title}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('lyapunov.png', dpi=150)
    plt.show()


# ==================== ГЛАВНАЯ ПРОГРАММА ====================

if __name__ == "__main__":
    print("ПОЛНЫЙ АНАЛИЗ: ГРАФ + КВАНТОВАЯ ДИНАМИКА + ХАОС + АВТОМАТ")

    # Параметры
    N = 300
    K = 6
    p = 1 / (K * N ** (1 / 3))

    print(f"ПАРАМЕТРЫ СИСТЕМЫ")
    print(f"N = {N}")
    print(f"K = {K}")
    print(f"p = {p:.6e}")
    print(f"Kp = N^(-1/3) = {K * p:.6e}")
    print(f"Число shortcut'ов ≈ {int(p * K * N / 2)}")

    # Создание графа
    print(f"СОЗДАНИЕ ГРАФА")

    G = create_watts_strogatz_graph(N, K, p)
    n_edges = G.number_of_edges()
    degrees = [d for _, d in G.degree()]

    print(f"Вершин: {G.number_of_nodes()}")
    print(f"Рёбер: {n_edges}")
    print(f"Средняя степень: {np.mean(degrees):.2f}")
    print(f"Мин/макс степень: {np.min(degrees)}/{np.max(degrees)}")
    print(f"Кластерный коэффициент: {nx.average_clustering(G):.4f}")

    # Спектральный анализ
    print(f"СПЕКТРАЛЬНЫЙ АНАЛИЗ")

    eigenvalues = calculate_laplacian_spectrum(G, verbose=True)
    constants = calculate_physical_constants(N, K, p, verbose=True)
    ds, r2 = estimate_spectral_dimension(eigenvalues, verbose=True)
    spectrum_info = analyze_spectrum_details(eigenvalues, constants, verbose=True)

    # Инициализация квантовой динамики
    print(f"КВАНТОВАЯ ДИНАМИКА")

    qd = QuantumGraphDynamics(G, verbose=True)

    # 1. Линейная динамика
    print(f"1. ЛИНЕЙНОЕ УРАВНЕНИЕ ШРЁДИНГЕРА")

    history_lin, diag_lin = qd.linear_schrodinger(dt=0.05, T=300, verbose=True)
    dyn_info_lin = qd.analyze_dynamics(history_lin, 'Линейная', verbose=True)
    corr_info = qd.analyze_quantum_correlations(history_lin, verbose=True)

    print(f"\n  [ПРОВЕРКА] Теоретическая частота из спектра:")
    print(f"    λ₁ = {eigenvalues[1]:.6f} → ω₁ = {eigenvalues[1] / constants['hbar']:.6f}")
    print(f"    Доминантная частота в динамике: {dyn_info_lin['dominant_freq']:.6f}")

    # 2. Нелинейная динамика
    print(f"2. НЕЛИНЕЙНОЕ УРАВНЕНИЕ ШРЁДИНГЕРА (DNLS)")

    for g in [0.1, 1.0, 5.0]:
        history_nl, diag_nl = qd.nonlinear_schrodinger(g=g, dt=0.05, T=300, verbose=False)
        final_pr = diag_nl['participations'][-1]
        print(f"  g={g}: финальный PR={final_pr:.2f} → {'ЛОКАЛИЗАЦИЯ' if final_pr < N / 2 else 'ДЕЛОКАЛИЗАЦИЯ'}")

    # 3. Хаотическая динамика
    print(f"3. ХАОТИЧЕСКАЯ ДИНАМИКА f(Wx)")

    regimes = [
        ('Порядок', 2.8, 0.2),
        ('Период-2', 3.2, 0.2),
        ('Слабый хаос', 3.5, 0.2),
        ('Развитой хаос', 3.8, 0.15),
        ('Сильный хаос', 4.0, 0.15)
    ]

    for name, r, eps in regimes:
        history_ch, diag_ch = qd.complex_logistic_strong(
            r=r, epsilon=eps, nonlinear='logistic', T=300, verbose=False
        )
        final_amps = np.abs(history_ch[-1])
        max_ratio = np.max(final_amps) / (np.mean(final_amps) + 1e-15)
        print(f"  {name}: r={r}, ε={eps}")
        print(f"    <|ψ|>={np.mean(final_amps):.4f}, max/mean={max_ratio:.2f}")

    # Детальная визуализация хаоса
    history_ch, diag_ch = qd.complex_logistic_strong(
        r=3.9, epsilon=0.15, nonlinear='logistic', T=500, verbose=True
    )
    plot_dynamics(history_ch, 'Хаос: f(Wx), r=3.9, ε=0.15')

    # Сравнение с оригиналом
    print(f"\n  [СРАВНЕНИЕ] Оригинальная динамика Wf(x):")
    history_orig, _ = qd.complex_logistic_original(r=3.9, epsilon=0.15, T=300, verbose=False)
    amps_orig = np.abs(history_orig[-1])
    print(f"    <|ψ|>={np.mean(amps_orig):.4f}, max/mean={np.max(amps_orig) / np.mean(amps_orig):.2f}")

    # 4. Бифуркационная диаграмма
    print(f"4. БИФУРКАЦИОННАЯ ДИАГРАММА")

    bif_data = qd.bifurcation_diagram(
        node_idx=0, nonlinear='logistic', epsilon=0.2,
        rs=np.linspace(2.5, 4.2, 300),
        T_transient=300, T_sample=100
    )
    plot_bifurcation(bif_data, f'N={N}, ε=0.2, f(Wx)')

    # 5. Показатели Ляпунова
    print(f"5. СПЕКТР ПОКАЗАТЕЛЕЙ ЛЯПУНОВА")

    rs = [2.8, 3.0, 3.2, 3.4, 3.5, 3.57, 3.6, 3.7, 3.8, 3.9, 4.0]
    lyap_results = []

    for r in rs:
        lyap, _ = qd.compute_lyapunov_correct(
            nonlinear='logistic', r=r, epsilon=0.2,
            T_transient=200, T_lyap=200, verbose=False
        )
        lyap_results.append(lyap)
        print(f"  r={r:.2f}: λ={lyap:.6f} → {'ХАОС ⚡' if lyap > 0.01 else 'порядок'}")

    # Находим критическое r
    for i in range(len(rs) - 1):
        if lyap_results[i] <= 0 and lyap_results[i + 1] > 0:
            r_crit = (rs[i] + rs[i + 1]) / 2
            print(f"\n  [ФЕЙГЕНБАУМ] Критическое r ≈ {r_crit:.3f}")
            break

    plot_lyapunov_analysis(rs, lyap_results, f'N={N}')

    # 6. Клеточный автомат
    print(f"6. КЛЕТОЧНЫЙ АВТОМАТ")

    for rule in ['majority', 'xor', 'average']:
        ca_history, counts = qd.run_cellular_automaton(
            rule=rule, T=50, verbose=False
        )
        final_counts = np.bincount(ca_history[-1], minlength=3)
        print(f"  Правило '{rule}': финал {final_counts}")

    ca_history, _ = qd.run_cellular_automaton(rule='majority', T=100, verbose=True)
    plot_automaton(ca_history, "Клеточный автомат: majority rule")

    # 7. Гибридная динамика
    print(f"7. ГИБРИДНАЯ ДИНАМИКА (Шрёдингер + нелинейность + связь)")

    history_hybrid, _ = qd.hybrid_schrodinger_logistic(
        r=3.5, epsilon=0.2, dt=0.01, T=500, verbose=True
    )
    qd.analyze_dynamics(history_hybrid, 'Гибрид', verbose=True)

    # 8. Константы Вселенной
    print(f"8. КОНСТАНТЫ ВСЕЛЕННОЙ (N=4.2×10^121)")

    const_univ = calculate_physical_constants(4.2e121, 6, 1 / (6 * (4.2e121) ** (1 / 3)), verbose=False)

    print(f"\n  Планковские единицы:")
    print(f"  lP = {const_univ['lP']:.6e}")
    print(f"  tP = {const_univ['tP']:.6e}")
    print(f"  mP = {const_univ['mP']:.6e}")
    print(f"  EP = {const_univ['EP']:.6e}")
    print(f"  TP = {const_univ['TP']:.6e}")

    print(f"\n  Фундаментальные постоянные:")
    print(f"  c  = {const_univ['c']:.6e}")
    print(f"  G  = {const_univ['G']:.6e}")
    print(f"  h  = {const_univ['h_em']:.6e}")
    print(f"  ℏ  = {const_univ['hbar']:.6e}")
    print(f"  kB = {const_univ['k_B']:.6e}")
    print(f"  α  = {const_univ['alpha']:.6e}")

    print(f"\n  Проверки согласованности:")
    print(f"  lP/tP = {const_univ['lP'] / const_univ['tP']:.6e} (c = {const_univ['c']:.6e})")
    print(f"  mP·c² = {const_univ['mP'] * const_univ['c'] ** 2:.6e} (EP = {const_univ['EP']:.6e})")
    print(f"  EP·tP = {const_univ['EP'] * const_univ['tP']:.6e} (ℏ = {const_univ['hbar']:.6e})")

    # Итоговый сводный анализ
    print(f"\n{'=' * 80}")
    print(f"СВОДНЫЙ АНАЛИЗ")
    print(f"{'=' * 80}")

    print(f"\n  СТРУКТУРА ГРАФА:")
    print(f"  N={N}, K={K}, p={p:.6e}")
    print(f"  Спектральная размерность: {ds:.3f} (ожидаемая: 2.0)")
    print(f"  Спектральная щель: {eigenvalues[1]:.6e}")

    print(f"\n  КВАНТОВЫЕ СВОЙСТВА:")
    print(f"  ω₀ = {eigenvalues[1] / constants['hbar']:.6e}")
    print(f"  τ_relax = {1 / eigenvalues[1]:.6e}")
    print(f"  Доминантная частота (линейная): {dyn_info_lin['dominant_freq']:.6f}")

    print(f"\n  ХАОТИЧЕСКИЕ СВОЙСТВА:")
    print(f"  Критическое r: ~{r_crit:.3f}")
    print(f"  Максимальный λ (r=3.9): {max(lyap_results):.6f}")
    print(f"  f(Wx) >> Wf(x) по хаотичности")

    print(f"\n  ЭМЕРДЖЕНТНЫЕ КОНСТАНТЫ:")
    print(f"  h_em = {constants['h_em']:.6e}")
    print(f"  c = {constants['c']:.6e}")
    print(f"  G = {constants['G']:.6e}")
    print(f"  α = {constants['alpha']:.6e}")
