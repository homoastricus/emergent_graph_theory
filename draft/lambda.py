import numpy as np
from scipy import constants
import math
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

class LambdaSelfConsistent:
    """
    Самосогласованное вычисление параметра λ
    исключительно из параметров сети (K, p, N).
    """

    def __init__(self, K, p, N):
        self.K = K                  # локальная связность
        self.p = p                  # вероятность shortcut
        self.N = N                  # голографическая энтропия (число узлов)
        self.lp = constants.physical_constants['Planck length'][0]

    # 1. Спектральный масштаб
    def from_spectral_gap(self):
        # λ_spectral ≈ 1/√(Kp)
        return 1.0 / np.sqrt(self.K * self.p)

    # 2. Корреляционная длина
    def from_correlation(self):
        # Средний путь ~ log N / log K
        correlation_length = np.log(self.N) / np.log(self.K)
        return 1.0 / correlation_length

    # 3. Энтропийный масштаб
    def from_entropy(self):
        # λ_entropy ~ 1 / (√N log K)
        return 1.0 / (np.sqrt(self.N) * np.log(self.K))

    # 4. Оптимальный баланс локального и глобального взаимодействия
    def from_optimal_transport(self):
        # λ_opt ~ √[(K(1−p)) / (p log N)]
        return np.sqrt((self.K * (1.0 - self.p)) / (self.p * np.log(self.N)))

    # 5. Композитная формула — взвешенное среднее
    def composite_lambda(self):
        λ_spec = self.from_spectral_gap()
        λ_corr = self.from_correlation()
        λ_ent = self.from_entropy()
        λ_opt = self.from_optimal_transport()

        # веса по физическому смыслу (основной вклад — спектральный и корреляционный)
        weights = np.array([0.09, 0.11, 0.79, 0.0]) # [0.0, 0.01, 0.99, 0.0]
        lambdas = np.array([λ_spec, λ_corr, λ_ent, λ_opt])
        return np.average(lambdas, weights=weights)

    # 6. Нормировка на планковскую шкалу
    def normalized_lambda(self):
        """
        Масштабируем λ к безразмерному виду в единицах Планка:
        λ_norm = λ_composite * l_P * N^(-1/3)
        (так как размер графа растёт как N^(1/3))
        """
        λ = self.composite_lambda()
        λ_norm = λ * self.lp * self.N ** (-1/3)
        return λ_norm

    def report(self):
        print("=== САМООРГАНИЗОВАННОЕ ВЫЧИСЛЕНИЕ λ ===")
        print(f"K = {self.K}, p = {self.p}, N = {self.N:.2e}")
        print("----------------------------------------")
        print(f"λ_spectral        = {self.from_spectral_gap():.6e}")
        print(f"λ_correlation     = {self.from_correlation():.6e}")
        print(f"λ_entropy         = {self.from_entropy():.6e}")
        print(f"λ_optimal         = {self.from_optimal_transport():.6e}")
        print("----------------------------------------")
        λ_comp = self.composite_lambda()
        print(f"λ_composite       = {λ_comp:.6e}")
        print(f"λ_normalized      = {self.normalized_lambda():.6e}")
        return λ_comp


# === Пример использования ===
if __name__ == "__main__":
    K = 8
    p = 0.06
    N = 10000000000000000000 #1e123  # энтропия горизонта   10000000000000000000

    calc = LambdaSelfConsistent(K, p, N)
    λ_final = calc.report()

    print("\nИТОГ:")
    print(f"Самосогласованное значение λ = {λ_final:.6e}")
