import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import constants


class AlternativeUniverse:
    def __init__(self, base_K=8.0, base_p=0.0527, base_N=0.95e123):
        self.base_K = base_K
        self.base_p = base_p
        self.base_N = base_N

        self.classical_constants = {
            'hbar': constants.hbar,
            'c': constants.c,
            'G': constants.G,
            'kb': constants.k,
            'lp': constants.physical_constants['Planck length'][0],
            'tp': constants.physical_constants['Planck time'][0],
            'Tp': constants.physical_constants['Planck temperature'][0],
            'cosmo_lambda': 1.1056e-52,
            'ep0_em': 8.85e-12,
            'mu0_em': 1.256e-6,
            'e_plank': 1.87e-18
        }

    def lambda_param(self, K, p, N):
        return (np.log(K * p) / np.log(N)) ** 2

    def calculate_universe(self, p_variation, K_variation=1.0, N_variation=1.0):
        """Рассчитывает вселенную с измененными параметрами"""
        K = self.base_K * K_variation
        p = self.base_p * p_variation
        N = self.base_N * N_variation

        # Ограничиваем p физически осмысленными значениями
        if p >= 1.0 or p <= 1e-10:
            return None

        λ = self.lambda_param(K, p, N)
        lnK = np.log(K)
        lnKp = np.log(K * p)
        lnN = np.log(N)

        # Основные формулы (как в вашем коде)
        hbar_em = (lnK ** 2) / (4 * λ ** 2 * K ** 2)
        R_universe = 2 * math.pi / (np.sqrt(K * p) * λ) * N ** (1 / 6)
        l_em = R_universe / np.sqrt(K * p)
        hbar_emergent = hbar_em * N ** (-1 / 3) / (6 * math.pi)
        c_emergent = (math.pi * l_em / hbar_em) / λ ** 2 * N ** (-1 / 6)
        G_emergent = (hbar_em ** 4 / l_em ** 2) * (1 / λ ** 2)
        KB2 = math.pi * lnN ** 7 / (3 * abs(lnKp) ** 6 * (p * K) ** (3 / 2) * N ** (1 / 3))
        cosmo_lambda = 3 * K * p / (math.pi ** 2 * N ** (1 / 3)) * (lnKp / lnN) ** 4

        # Эффективная размерность
        d_eff = 1 + 4 * (1 - np.exp(-0.15 * (K - 3))) * np.exp(-20 * abs(p - 0.05) ** 1.5)

        # Критические отношения
        alpha_G = G_emergent * 1.67e-27 ** 2 / (
                    constants.hbar * constants.c)  # гравитационная постоянная тонкой структуры
        alpha_em = 1 / 137  # предполагаем постоянной для простоты

        return {
            'p': p,
            'K': K,
            'N': N,
            'd_eff': d_eff,
            'c': c_emergent,
            'hbar': hbar_emergent,
            'G': G_emergent,
            'lambda': cosmo_lambda,
            'kb': KB2,
            'alpha_G': alpha_G,
            'R': R_universe,
            'ratio_c': c_emergent / self.classical_constants['c'],
            'ratio_G': G_emergent / self.classical_constants['G'],
            'ratio_hbar': hbar_emergent / self.classical_constants['hbar']
        }

    def analyze_alternative_universes(self):
        """Анализ альтернативных вселенных"""

        print("=== АЛЬТЕРНАТИВНЫЕ ВСЕЛЕННЫЕ ===")
        print(f"Базовая вселенная: K={self.base_K}, p={self.base_p}, N={self.base_N:.2e}")
        print("p = вероятность дальнодействующих связей")
        print("=" * 60)

        # Диапазон изменения p
        p_factors = [0.1, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 5.0, 10.0]

        universes = []
        for factor in p_factors:
            universe = self.calculate_universe(factor)
            if universe:
                universes.append((factor, universe))

        # Вывод результатов
        print("\np (отн.) | Размерность | c/c₀     | G/G₀     | ħ/ħ₀     | α_G      | Λ/Λ₀")
        print("-" * 80)

        for factor, uni in universes:
            # Критерии обитаемости
            habitable = self.check_habitability(uni)
            marker = "🏠" if habitable else " "

            print(f"{factor:7.2f}x  | {uni['d_eff']:9.2f}  | {uni['ratio_c']:7.3f}  | "
                  f"{uni['ratio_G']:7.3f}  | {uni['ratio_hbar']:7.3f}  | "
                  f"{uni['alpha_G']:9.2e} | {uni['lambda'] / self.classical_constants['cosmo_lambda']:7.3f} {marker}")

        # Детальный анализ ключевых случаев
        print("\n\n=== ФИЗИКА АЛЬТЕРНАТИВНЫХ ВСЕЛЕННЫХ ===")

        key_cases = [
            (0.1, "Меньше дальнодействия (p=0.1×)"),
            (0.5, "Значительно меньше дальнодействия"),
            (1.0, "Наша Вселенная"),
            (1.5, "Больше дальнодействия"),
            (2.0, "Значительно больше дальнодействия"),
            (5.0, "Очень много дальнодействия"),
        ]

        for factor, desc in key_cases:
            uni = self.calculate_universe(factor)
            if uni:
                print(f"\n{desc}:")
                print(f"  p = {uni['p']:.6f} (в {factor:.1f} раз от нашего)")
                print(f"  Эффективная размерность: {uni['d_eff']:.2f}")
                print(f"  Скорость света: {uni['c']:.3e} м/с ({uni['ratio_c']:.1%} от нашей)")
                print(f"  Гравитационная постоянная: {uni['G']:.3e} ({uni['ratio_G']:.1%})")
                print(f"  Квант действия: {uni['hbar']:.3e} ({uni['ratio_hbar']:.1%})")
                print(
                    f"  Космологическая постоянная: {uni['lambda']:.3e} ({uni['lambda'] / self.classical_constants['cosmo_lambda']:.1%})")

                # Физические следствия
                consequences = self.get_consequences(uni, factor)
                for cons in consequences:
                    print(f"  → {cons}")

    def check_habitability(self, universe):
        """Проверяет, может ли вселенная быть обитаемой"""
        # Критерии обитаемости (упрощенные)
        criteria = [
            universe['d_eff'] >= 2.9 and universe['d_eff'] <= 3.1,  # ~3 измерения
            abs(universe['ratio_c'] - 1) < 0.5,  # c не слишком отличается
            abs(universe['ratio_G'] - 1) < 10,  # G в разумных пределах
            universe['alpha_G'] < 1e-36,  # гравитация слабее электромагнетизма
            universe['lambda'] > 0,  # положительная Λ
        ]

        return all(criteria)

    def get_consequences(self, universe, p_factor):
        """Физические следствия изменения p"""
        consequences = []

        # 1. Влияние на квантовую механику
        if p_factor < 0.5:
            consequences.append("Слабая квантовая запутанность, мало нелокальных корреляций")
        elif p_factor > 2.0:
            consequences.append("Сильная квантовая запутанность, возможно макроскопические квантовые эффекты")

        # 2. Влияние на гравитацию
        G_ratio = universe['ratio_G']
        if G_ratio > 10:
            consequences.append("ОЧЕНЬ СИЛЬНАЯ гравитация - звезды быстро сжигают топливо")
        elif G_ratio > 2:
            consequences.append("Сильная гравитация - меньшие звезды, более быстрая эволюция")
        elif G_ratio < 0.5:
            consequences.append("Слабая гравитация - огромные звезды, медленная эволюция")
        elif G_ratio < 0.1:
            consequences.append("ОЧЕНЬ СЛАБАЯ гравитация - невозможность образования звезд")

        # 3. Влияние на скорость света
        c_ratio = universe['ratio_c']
        if c_ratio > 2:
            consequences.append("Быстрая связь между регионами вселенной")
        elif c_ratio < 0.5:
            consequences.append("Медленная коммуникация, изолированные регионы")

        # 4. Размерность
        d = universe['d_eff']
        if d < 2.5:
            consequences.append(f"Суб-3D пространство ({d:.1f} измерений) - ограниченная сложность структур")
        elif d > 3.5:
            consequences.append(f"Сверх-3D пространство ({d:.1f} измерений) - возможна дополнительная физика")

        # 5. Космологическая постоянная
        lambda_ratio = universe['lambda'] / self.classical_constants['cosmo_lambda']
        if lambda_ratio > 100:
            consequences.append("ОЧЕНЬ быстрое расширение - разрыв структур")
        elif lambda_ratio > 10:
            consequences.append("Быстрое расширение - мало галактических скоплений")
        elif lambda_ratio < 0.1:
            consequences.append("Медленное расширение - возможен коллапс вселенной")

        # 6. Возможность сложности
        hbar_ratio = universe['ratio_hbar']
        if 0.1 < hbar_ratio < 10 and 0.1 < G_ratio < 10 and 0.5 < c_ratio < 2:
            consequences.append("Возможны сложные структуры (звезды, планеты, жизнь)")
        else:
            consequences.append("Вероятно слишком простая или слишком хаотичная физика для сложности")

        return consequences

    def plot_universes(self):
        """Визуализация альтернативных вселенных"""
        p_values = np.linspace(0.01, 0.2, 50)

        metrics = {
            'd_eff': [],
            'c_ratio': [],
            'G_ratio': [],
            'hbar_ratio': [],
            'lambda_ratio': []
        }

        for p_val in p_values:
            uni = self.calculate_universe(p_val / self.base_p)
            if uni:
                metrics['d_eff'].append(uni['d_eff'])
                metrics['c_ratio'].append(uni['ratio_c'])
                metrics['G_ratio'].append(uni['ratio_G'])
                metrics['hbar_ratio'].append(uni['ratio_hbar'])
                metrics['lambda_ratio'].append(uni['lambda'] / self.classical_constants['cosmo_lambda'])

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # График 1: Размерность
        axes[0, 0].plot(p_values, metrics['d_eff'], 'b-', linewidth=2)
        axes[0, 0].axvline(self.base_p, color='r', linestyle='--', label='Наша Вселенная')
        axes[0, 0].axhline(3.0, color='g', linestyle=':', alpha=0.5)
        axes[0, 0].set_xlabel('Вероятность дальних связей (p)')
        axes[0, 0].set_ylabel('Эффективная размерность')
        axes[0, 0].set_title('Размерность пространства')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # График 2: Скорость света
        axes[0, 1].plot(p_values, metrics['c_ratio'], 'g-', linewidth=2)
        axes[0, 1].axvline(self.base_p, color='r', linestyle='--')
        axes[0, 1].axhline(1.0, color='g', linestyle=':', alpha=0.5)
        axes[0, 1].set_xlabel('Вероятность дальних связей (p)')
        axes[0, 1].set_ylabel('c/c₀ (отношение к нашей)')
        axes[0, 1].set_title('Скорость света')
        axes[0, 1].grid(True, alpha=0.3)

        # График 3: Гравитационная постоянная (логарифмический!)
        axes[0, 2].plot(p_values, metrics['G_ratio'], 'r-', linewidth=2)
        axes[0, 2].axvline(self.base_p, color='r', linestyle='--')
        axes[0, 2].axhline(1.0, color='g', linestyle=':', alpha=0.5)
        axes[0, 2].set_xlabel('Вероятность дальних связей (p)')
        axes[0, 2].set_ylabel('G/G₀ (отношение к нашей)')
        axes[0, 2].set_title('Гравитационная постоянная')
        axes[0, 2].set_yscale('log')
        axes[0, 2].grid(True, alpha=0.3)

        # График 4: Квант действия
        axes[1, 0].plot(p_values, metrics['hbar_ratio'], 'purple', linewidth=2)
        axes[1, 0].axvline(self.base_p, color='r', linestyle='--')
        axes[1, 0].axhline(1.0, color='g', linestyle=':', alpha=0.5)
        axes[1, 0].set_xlabel('Вероятность дальних связей (p)')
        axes[1, 0].set_ylabel('ħ/ħ₀ (отношение к нашей)')
        axes[1, 0].set_title('Квант действия')
        axes[1, 0].grid(True, alpha=0.3)

        # График 5: Космологическая постоянная (логарифмический!)
        axes[1, 1].plot(p_values, metrics['lambda_ratio'], 'orange', linewidth=2)
        axes[1, 1].axvline(self.base_p, color='r', linestyle='--')
        axes[1, 1].axhline(1.0, color='g', linestyle=':', alpha=0.5)
        axes[1, 1].set_xlabel('Вероятность дальних связей (p)')
        axes[1, 1].set_ylabel('Λ/Λ₀ (отношение к нашей)')
        axes[1, 1].set_title('Космологическая постоянная')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)

        # График 6: Область обитаемости
        habitable = []
        for i, p_val in enumerate(p_values):
            uni = self.calculate_universe(p_val / self.base_p)
            if uni and self.check_habitability(uni):
                habitable.append(p_val)

        axes[1, 2].scatter(habitable, [1] * len(habitable), color='green', s=50, alpha=0.6,
                           label='Возможно обитаемые')
        axes[1, 2].axvline(self.base_p, color='r', linestyle='--', label='Наша Вселенная')
        axes[1, 2].set_xlabel('Вероятность дальних связей (p)')
        axes[1, 2].set_title('Область возможной обитаемости')
        axes[1, 2].set_ylim(0.5, 1.5)
        axes[1, 2].get_yaxis().set_visible(False)
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


def simulate_extreme_universes():
    """Моделирование экстремальных случаев"""
    print("\n" + "=" * 70)
    print("ЭКСТРЕМАЛЬНЫЕ СЛУЧАИ:")
    print("=" * 70)

    simulator = AlternativeUniverse()

    extreme_cases = [
        (0.001, "p → 0 (почти нет дальнодействия)"),
        (0.01, "p очень мало"),
        (0.1, "p мало"),
        (0.3, "p большое"),
        (0.7, "p очень большое"),
        (0.99, "p → 1 (почти все связи дальнодействующие)")
    ]

    for factor, desc in extreme_cases:
        print(f"\n{desc}:")
        uni = simulator.calculate_universe(factor / 0.0527)  # нормализуем к нашему p

        if uni:
            print(f"  Размерность: {uni['d_eff']:.2f}")
            print(f"  G/G₀ = {uni['ratio_G']:.2e}")
            print(f"  c/c₀ = {uni['ratio_c']:.2f}")

            # Что происходит в таких вселенных?
            if factor < 0.01:
                print("  → Пространство почти дискретно, сильная локальность")
                print("  → Слабая квантовая запутанность")
                print("  → Возможно, классическая физика доминирует")
            elif factor > 0.5:
                print("  → Пространство сильно нелокально")
                print("  → Сильная квантовая запутанность на всех масштабах")
                print("  → Возможно, квантовые эффекты доминируют")
        else:
            print("  Невозможная конфигурация")


# Запуск анализа
if __name__ == "__main__":
    universe_sim = AlternativeUniverse()

    # 1. Табличный анализ
    universe_sim.analyze_alternative_universes()

    # 2. Графики
    universe_sim.plot_universes()

    # 3. Экстремальные случаи
    simulate_extreme_universes()