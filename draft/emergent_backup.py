import numpy as np
from scipy import constants

import math

class EmergentPhysicsCalculator:
    def __init__(self, K, p, lambda_param, N, M):
        """
        Инициализация параметров сети малого мира

        Parameters:
        K - локальная связность
        p - вероятность случайной связи
        lambda_param - спектральный масштаб лапласиана
        N - голографическая энтропия (площадь горизонта)
        """
        self.K = K
        self.p = p
        self.lambda_param = lambda_param
        self.N = N
        self.M = M

        # Классические физические константы (CODATA 2018)
        self.classical_constants = {
            'hbar': constants.hbar,  # 1.054571817e-34 J·s
            'c': constants.c,  # 299792458 m/s
            'G': constants.G,  # 6.67430e-11 m³/kg·s²
            'kb': constants.k,  # 1.380649e-23 J/K
            'lp': constants.physical_constants['Planck length'][0],  # 1.616255e-35 m
            'tp': constants.physical_constants['Planck time'][0],  # 5.391247e-44 s
            'Tp': constants.physical_constants['Planck temperature'][0],  # 1.416784e+32 K
            'cosmo_lambda': 1.1056e-52,
            'T_plank': 1.417e32,
            'ep0_em': 8.85e-12,
            'mu0_em': 1.256e-6,
            'e_plank': 1.87e-18,
            'electron_charge': 1.60e-19,
            'alfa_em': 7.297352e-3,
            'electron_mass': 9.109e-31,
            'plank_mass': 2.176e-8,
            'muon': 1.899e-28,
            'tau': 3.167e-27,
            'up_part': 2.162e-30,
            'down_part': 4.658e-30,
            'strange': 9.495e-29,
            'charm': 1.269e-27,
            'bottom_part': 4.178e-27,
            'top_part': 3.067e-25,
            'proton_part': 1.673e-27,
            'neutron_part': 1.677e-27,
            'W_boson': 1.434e-25,
            'HIGGS': 2.244e-25,
            'Z_boson': 1.621e-25,
            'deuterium': 3.304e-27,
            'lithium6': 9.988e-27,
            'lithium7': 1.165e-26,
            'uran_238': 3.952e-25,
            'thoriy_232': 3.8526e-25,
            'alpha_He': 6.333e-27,
            'pion': 2.391e-28,
            'kaon': 8.808e-28,
            'eta_meson': 9.739e-28,
            'rho_meson': 1.286e-27,
            'neutrino_e': 1.8e-38,
            'neutrino_mu': 9e-38,
            'neutrino_tau': 1.8e-37,
            'bor_orbital_radius': 5.291e-11,
            'compton_electron_em': 2.426e-12,
            'compton_pi_meson_em': 1.460e-15,
            'W_boson_compton_em': 2.45e-18
        }

    def calculate_emergent_constants(self):
        """Вычисление всех эмерджентных физических констант"""

        # 1. Локальный квант действия
        hbar_em = (np.log(self.K) ** 2) / (4 * self.lambda_param ** 2 * self.K ** 2)
        print(f"hbar_em {hbar_em:.3f}")

        C = 3 * (self.K - 2) / (4 * (self.K - 1)) * (1 - self.p) ** 3
        correction = 1 + (1 - C) / np.log(self.N)
        hbar_em = hbar_em * correction
        # коррекционная кластерная функция, описана в статье в разделе приложения, следует ка результат энтропийного распределения конфигураций графа.


        # 2. Постоянная Планка
        hbar_emergent = hbar_em * self.N ** (-1 / 3) / (6 * math.pi)
        print(f"hbar_emergent {hbar_emergent:.3e}")

        # 3. Диаметр Вселенной R_universe = 3e26
        R_universe = 2 * math.pi / (np.sqrt(self.K * self.p) * self.lambda_param) * self.N ** (1 / 6)
        print(f"R_universe {R_universe:.3e}")

        # 4. Локальный масштаб длины через спектр лапласиана
        l_em = 2 * math.pi / (self.K * self.p * self.lambda_param) * self.N ** (1 / 6)
        print(f"l_em {l_em:.3e}")

        # 5. Планковская длина
        lp_emergent = 1 / np.sqrt(self.K * self.p) * R_universe * self.N ** (-1 / 2)
        print(f"lp_emergent {lp_emergent:.3e}")
        # N = 4πR² / ℓ_P² ⇒ R = √(N * ℓ_P² / 4π)

        # 6. Планковское время
        tp_emergent = self.lambda_param ** 2 * hbar_em * self.N ** (-1 / 3) / math.pi
        print(f"tp_emergent {tp_emergent:.3e}")
        tp_emergent_final = self.lambda_param ** 2 * ((np.log(self.K) ** 2) / (4 * self.lambda_param ** 2 * self.K ** 2)) * self.N ** (-1 / 3) / math.pi
        print(f"tp_emergent_final {tp_emergent_final:.3e}")

        # 7. Скорость света
        # c_emergent = (l_em / hbar_em) / self.lambda_param ** 2 * self.N ** (-1 / 6)
        c_emergent = math.pi * (1 / np.sqrt(self.K * self.p) * R_universe / ((np.log(self.K) ** 2) / (
                4 * self.lambda_param ** 2 * self.K ** 2))) / self.lambda_param ** 2 * self.N ** (-1 / 6)
        print(f"c_emergent {c_emergent:.3e}")

        c_emergent_final = 8 * math.pi**2 * self.K * np.log(self.N)**2  / (self.p * np.log(self.K)**2 * abs(np.log(self.p * self.K))**2 )
        print(f"c_emergent_final {c_emergent_final:.3e}")

        # 8. Гравитационная постоянная
        G_emergent = (hbar_em ** 4 / l_em ** 2) * (1 / self.lambda_param ** 2)
        print(f"G_emergent {G_emergent:.3e}")

        G_emergent_final =  (np.log(self.K)**8 * self.p**2) / (1024 * math.pi**2 * self.lambda_param**8 * self.K**6 * self.N**(1/3))
        print(f"G_emergent_final {G_emergent_final:.3e}")

        # 9. Планковская энергия
        Ep_emergent = hbar_emergent / tp_emergent

        # 10. Масса Планка Постоянная тонкой структуры
        M_planck_test = np.sqrt(hbar_emergent * c_emergent / G_emergent)
        print(f"M_planck_test {M_planck_test:.3e}")

        M_planck_final_middle = np.sqrt(
            (((np.log(self.K) ** 2) / (4 * self.lambda_param ** 2 * self.K ** 2)) * self.N ** (-1 / 3) / (6 * math.pi))
            * (math.pi * (1 / np.sqrt(self.K * self.p)
                          * (2 * math.pi / (np.sqrt(self.K * self.p) * self.lambda_param) * self.N ** (1 / 6)) / (
                                  (np.log(self.K) ** 2) / (
                                  4 * self.lambda_param ** 2 * self.K ** 2))) / self.lambda_param ** 2 * self.N ** (
                       -1 / 6))
            / ((((np.log(self.K) ** 2) / (4 * self.lambda_param ** 2 * self.K ** 2)) ** 4 / (
                    2 * math.pi / (self.K * self.p * self.lambda_param) * self.N ** (1 / 6)) ** 2) * (
                       1 / self.lambda_param ** 2)))
        print(f"M_planck_final_middle {M_planck_final_middle:.3e}")

        M_planck_final = (32 / math.sqrt(3)) * (math.pi ** 1.5) * (
                abs((math.log(self.K * self.p)) / math.log(self.N)) ** 5) * (self.K ** 2.5) / (
                                 (math.log(self.K) ** 4) * (self.p ** 1.5))
        print(f"M_planck_final {M_planck_final:.3e}")

        Ms = 1
        R_schwarzschild_middle = (2 * (
                (((np.log(self.K) ** 2) / (4 * self.lambda_param ** 2 * self.K ** 2)) ** 4 / l_em ** 2) * (
                1 / self.lambda_param ** 2)) * Ms /
                                  (math.pi * (1 / np.sqrt(self.K * self.p) * (
                                          2 * math.pi / (np.sqrt(self.K * self.p) * self.lambda_param) * self.N ** (
                                          1 / 6)) / ((np.log(self.K) ** 2) / (
                                          4 * self.lambda_param ** 2 * self.K ** 2))) / self.lambda_param ** 2 * self.N ** (
                                           -1 / 6)) ** 2)

        R_schwarzschild_final = (
                (self.p ** 4 * Ms) / (32768 * (math.pi ** 6) * (self.K ** 8) * (self.N ** (1 / 3)))
                * ((math.log(self.K) * math.log(self.N) / math.log(self.K * self.p)) ** 12)
        )
        print(f"R_schwarzschild {R_schwarzschild_final:.3e}")

        effective_dimension = 3
        # cosmo_lambda = 2 * effective_dimension / R_universe**2 * (effective_dimension-1)
        # или после преобразований: Λ=3/4π^2 * K * p * λ^2 * N^−1/3
        # Финальный упрощенный код:
        # cosmo_lambda = (3 / (4 * math.pi ** 2)) * self.K * self.p * self.lambda_param ** 2 * self.N ** (-1 / 3)
        cosmo_lambda = 3 * self.K * self.p / (math.pi ** 2 * self.N ** (1 / 3)) * (
                np.log(self.K * self.p) / np.log(self.N)) ** 4
        print(f"cosmo_lambda {cosmo_lambda:.3e}")

        # 8. Постоянная Больцмана
        KB = 1.3e-23
        #  T = mp c²/k
        T_plank = ((32 / math.sqrt(3)) * (math.pi ** 1.5) * (
                abs((math.log(self.K * self.p)) / math.log(self.N)) ** 5) * (self.K ** 2.5) / (
                                 (math.log(self.K) ** 4) * (self.p ** 1.5))) * (8 * math.pi**2 * self.K * np.log(self.N)**2
                                / (self.p * np.log(self.K)**2 * abs(np.log(self.p * self.K))**2 ))**2 / (math.pi * math.log(self.N) ** 7 / (
                3 * abs(math.log(self.K * self.p) ** 6) * (self.p * self.K) ** (3 / 2) * self.N ** (1 / 3)))
        print(f"T_plank {T_plank:.3e}")

        T_plank_final = ((6144 * math.pi**4.5 / math.sqrt(3)) * (abs(math.log(self.K * self.p))**7 * self.K**6 * self.N**(1/3))
                         / (self.p**2 * math.log(self.K)**8 * math.log(self.N)**8))

        print(f"T_plank_final {T_plank_final:.3e}")
        print(f"lambda_param {self.lambda_param:.3e}")

        KB_start = (((np.log(self.K) ** 2) / (4 * self.lambda_param ** 2 * self.K ** 2)) * self.N ** (-1 / 3) / (
                6 * math.pi)) * (math.pi * (1 / np.sqrt(self.K * self.p) * (
                2 * math.pi / (np.sqrt(self.K * self.p) * self.lambda_param) * self.N ** (1 / 6)) / (
                                                    (np.log(self.K) ** 2) / (
                                                    4 * self.lambda_param ** 2 * self.K ** 2))) / self.lambda_param ** 2 * self.N ** (
                                         -1 / 6)) * math.log(self.N) / math.sqrt(self.K * self.p)
        print(f"KB_start ======  {KB_start:.3e}")

        # в результате получается финальное математически упрощенное выражение для постоянной Больцмана
        KB2 = math.pi * math.log(self.N) ** 7 / (
                3 * abs(math.log(self.K * self.p) ** 6) * (self.p * self.K) ** (3 / 2) * self.N ** (1 / 3))
        print(f"KB2 ======  {KB2:.3e}")

        # Температура Хокинга TH = ℏc38πGMkB
        Th = 2 * math.pi * hbar_emergent * c_emergent ** 3 / (16 * math.pi * KB2 * Ms * G_emergent)
        Th_hocking_final_middle = (2 * math.pi *
                                   (((np.log(self.K) ** 2) / (4 * self.lambda_param ** 2 * self.K ** 2)) * self.N ** (
                                           -1 / 3)
                                    / (6 * math.pi)) * (math.pi * (1 / np.sqrt(self.K * self.p) * (
                        2 * math.pi / (np.sqrt(self.K * self.p) * self.lambda_param) * self.N ** (1 / 6)) / (
                                                                           (np.log(self.K) ** 2) / (
                                                                           4 * self.lambda_param ** 2 * self.K ** 2))) / self.lambda_param ** 2 * self.N ** (
                                                                -1 / 6)) ** 3 /
                                   (16 * math.pi * (math.pi * math.log(self.N) ** 7 / (
                                           3 * abs(math.log(self.K * self.p) ** 6) * (self.p * self.K) ** (
                                           3 / 2) * self.N ** (1 / 3)))
                                    * Ms * ((((np.log(self.K) ** 2) / (4 * self.lambda_param ** 2 * self.K ** 2)) ** 4 /
                                             (2 * math.pi / (self.K * self.p * self.lambda_param) * self.N ** (
                                                     1 / 6)) ** 2)
                                            * (1 / self.lambda_param ** 2))))

        print(f"Th_hocking_final_middle {Th_hocking_final_middle:.3e}")
        Th_hocking_final = (8192 * (math.pi ** 6) * (abs(math.log(self.K * self.p)) ** 12) * (self.K ** (17 / 2)) * (
                self.N ** (1 / 3))
                            / (Ms * (math.log(self.N) ** 13) * (self.p ** (7 / 2)) * (math.log(self.K) ** 12)))
        print(f"Th_hocking_final {Th_hocking_final:.3e}")

        # ε₀ = (λ² K) / (4π c² ℏ_em N ^ {1 / 3})
        ep0_em = ((((np.log(self.K * self.p) / np.log(self.N)) ** 4) * self.K)
                  / (2 * math.pi * ((8 * math.pi**2 * self.K * np.log(self.N)**2  /
                                     (self.p * np.log(self.K)**2 * abs(np.log(self.p * self.K))**2 )) ** 2)
                     * (hbar_em * self.N ** (-1 / 3) / (6 * math.pi)) * (self.N ** (1 / 3)) * (math.pi * math.log(self.N) ** 7 / (
                3 * abs(math.log(self.K * self.p) ** 6) * (self.p * self.K) ** (3 / 2) * self.N ** (1/3)))))
        print(f"ep0_em {ep0_em:.3e}")
        # после математического упрощения получаем
        epsilon_0_emergent = (9 * (self.lambda_param ** 2) * (self.K ** (5/2)) * (self.p ** (7/2)) *
          (self.N ** (1/3)) * (np.log(self.K) ** 2) *
          (np.log(self.K * self.p) ** 14)) / (
          16 * (np.pi ** 5) * (np.log(self.N) ** 15))
        print(f"epsilon_0_emergent {epsilon_0_emergent:.3e}")

        mu0_test = 1 / (ep0_em * c_emergent ** 2)
        print(f"mu0_test {mu0_test:.3e}")

        mu0_em = ((math.log(self.K)) ** 2) / (14 * KB2 * (self.K ** 3) * (self.lambda_param ** 4))
        # получается упрощенное выражение после математических преобразований
        mu0_em = (math.pi * (math.log(self.K) ** 2) * (math.log(self.N) ** 15) /
                  (36 * (self.K ** (9 / 2)) * (self.p ** (3 / 2)) * (abs(math.log(self.K * self.p)) ** 14) * (
                          self.N ** (1 / 3))))
        print(f"mu0_result {mu0_em:.3e}")

        # πλK² / (ln K)²
        alfa_em = np.log(self.K) / np.log(self.M)
        print(f"alfa_em {alfa_em:.3e}")

        '''
        e_plank_long = math.sqrt(4 * math.pi *
                                 (((np.log(self.K) ** 2) / (4 * self.lambda_param ** 2 * self.K ** 2)) * self.N ** (
                                         -1 / 3) / (6 * math.pi)) *
                                 (math.pi * (1 / np.sqrt(self.K * self.p) *
                                             (2 * math.pi / (np.sqrt(self.K * self.p) * self.lambda_param) * self.N ** (
                                                     1 / 6)) / ((np.log(self.K) ** 2) /
                                                                (4 * self.lambda_param ** 2 * self.K ** 2)))
                                  / self.lambda_param ** 2 * self.N ** (-1 / 6)) *
                                 (((((np.log(self.K * self.p) / np.log(self.N)) ** 4) * self.K)
                                   / (2 * math.pi * ((math.pi * (1 / np.sqrt(self.K * self.p) * (2 * math.pi / (
                                                 np.sqrt(self.K * self.p) * self.lambda_param) * self.N ** (
                                                                                                         1 / 6)) /
                                                                 ((np.log(self.K) ** 2) / (
                                                                         4 * self.lambda_param ** 2 * self.K ** 2)))
                                                      / self.lambda_param ** 2 * self.N ** (-1 / 6)) ** 2) * (
                                              (np.log(self.K) ** 2) /
                                              (4 * self.lambda_param ** 2 * self.K ** 2) * self.N ** (-1 / 3)
                                              / (6 * math.pi)) * (self.N ** (1 / 3)) * (
                                              math.pi * math.log(self.N) ** 7 / (
                                              3 * abs(math.log(self.K * self.p) ** 6) * (self.p * self.K) ** (
                                              3 / 2) * self.N ** (1 / 3))))))
                                 )
        '''
        lnK = math.log(self.K)
        lnKp = math.log(self.K * self.p)
        lnN = math.log(self.N)
        e_plank = math.sqrt(
            3 * self.p ** (5 / 2) * self.K ** (1.5) * lnK ** 2 * lnKp ** 12 / (4 * math.pi ** 3 * lnN ** 13))


        def e_emergent_charge(N, K, p):
            num = (3 / (4 * math.pi ** 3)) * (K ** (3/2)) * (p ** (5/2))
            num *= (math.log(K) ** 3) * (math.log(K * p) ** 14)
            den = (abs(math.log(K * p)) ** 2) * (math.log(N) ** 14)
            return math.sqrt(num / den)

        electron_charge = e_emergent_charge(N=self.N, K=self.K, p=self.p)
        # температура планка
        T_plank = (hbar_emergent * c_emergent ** 5 / (G_emergent * KB2 ** 2)) ** 0.5

        def mass_function(m_planck, particle):
            particle_koef = 1
            if particle == "electron":
                particle_koef = 1
            # Физический смысл: учитывает геометрию упаковки в графе
            C = (np.log(self.K) / np.log(2)) ** (1 / 2)
            # phase = math.sin(math.pi * np.log(self.K * self.p) / np.log(self.N))
            base_scaling = particle_koef * 2 * math.pi * (self.p * np.log(self.K)) ** 3 * self.N ** (-1 / 6)
            return m_planck * base_scaling * C  #

        # m_pl = 2.176e-8
        def electron_mass_holomorphic_calculation(m_planck, particle):
            """голографический подход"""
            # Полная масса Вселенной
            # R_universe = 3e26
            rho_critical = 9.31e-27
            M_universe = (4 / 3) * np.pi * R_universe ** 3 * rho_critical
            # Используем универсальную формулу для калибровки
            calc_mass = mass_function(M_planck_test, particle)
            # Вычисляем соответствующий масштабирующий фактор
            scaling_factor = calc_mass * self.N * np.sqrt(self.K) / M_universe
            # Корректируем формулу
            electron_mass = M_universe * scaling_factor / (self.N * np.sqrt(self.K))
            electron_mass = calc_mass
            return electron_mass

        # расcчет массы электрона
        # electron_mass_holomorphic_calculation(M_planck_final, "electron")
        # [√(Kp) × (lnK ^ 4 / K ^ 2)] × (1 / U) × M_planck × (π / 10)
        # electron_mass = (m_planck * math.pi /10 * np.sqrt(self.K * self.p) * ((np.log(self.K))**4 / self.K**2 ) *
        # 1/(math.log(self.N) / abs(math.log(self.K * self.p))))
        electron_mass = 12 * np.sqrt(self.K * self.p) * ((math.log(self.N) / abs(math.log(self.K * self.p))) ** 4) * (
                self.N ** (-1 / 3))

        # радиус Бора  5,291e-11
        r_bor_emergent = hbar_emergent / (electron_mass * alfa_em * c_emergent)
        print(f"r_bor_emergent {r_bor_emergent:.3e}")

        r_bor_emergent_final = (((math.log(self.K) ** 3) * self.p * math.log(6 * self.N) * (abs(math.log(self.K * self.p)) ** 2))
                                / (2304 * (math.pi ** 3) * (self.K ** 3) * math.sqrt(self.K * self.p) * (math.log(self.N) ** 2)))
        print(f"r_bor_emergent_final {r_bor_emergent_final:.3e}")

        # длина волны комптона для электрона
        compton_electron_em = 2 * (math.log(self.K)**4 * math.log(self.K * self.p)**2 * math.sqrt(self.p)) / (2304 * math.pi**2 * self.K**3.5 * math.log(self.N)**2)

        # def eta(K, p, N):
        #     # кластеризация локальная
        #     C_cluster = (3 * (K - 2) / (4 * (K - 1))) * (1 - p) ** 3
        #     # фрактальный параметр
        #     U = np.log(N) / abs(np.log(K * p))
        #     # вторая гармоника / первая
        #     lambda1 = (np.log(K * p) / np.log(N)) ** 2
        #     lambda2 = K * (1 - C_cluster) * (1 - 1 / U)
        #     eta = lambda2 / lambda1
        #     return eta
        # print(f" eta { eta(self.K, self.p, self.N)}") * eta(self.K, self.p, self.N)
        # масса электрона

        # массы частиц
        # Базовые величины
        lnK = math.log(self.K)
        U = math.log(self.N) / abs(math.log(self.K * self.p))

        # Структурные функции
        # фрактальный масштаб
        f1 = U / math.pi  # U/π

        # энтропия узла
        f2 = lnK  # lnK

        # (локальная скорость, локальная частота)
        f3 = math.sqrt(self.K * self.p)  # √(Kp)

        # нелокальность
        f4 = 1 / self.p  # 1/p

        # регулярность (структурная симметрия)
        f5 = self.K / lnK  # K/lnK

        f6 = (self.K + self.p * self.K) / self.K  # 1.053#
        print( "f1-6 values " +str(f1) + " " + str(f2) + " " + str(f3) + " " + str(f4) + " " + str(f5) + " " + str(f6))
        print(f" f6:{f6}")
        # Базовая масса электрона
        m_e = 12 * f3 * (U ** 4) * (self.N ** (-1 / 3))
        """Мюон - подтверждённая формула"""
        muon = m_e * 2 * f1
        """Тау-лептон - исправленная формула"""
        # m_τ = m_e × (U/π)² / (K/lnK)
        # tau             = m_e * (math.sqrt(self.K)/(2 * self.p)) * f1  * (1/self.p) #m_e * f1**2 / f5
        tau = m_e * f1 * 1 / f2 ** 2 * 1 / f3 * f4 ** 2 * 1 / f5
        """Up кварк - исправленная формула"""
        # m_u = m_e × √(K/p) / 10

        up_part = m_e * f3 ** 2 * f4 ** 2 / (f5 ** 2 * f2 ** 2)
        """Down кварк - через up кварк"""
        # m_d = m_u × lnK     f1^1 × f2^2 × f3^-1 × f4^-1 × f5^-2
        down_part = m_e * f2 ** 2 * f1 / (f3 * f4 * f5 ** 2) * f2
        """Strange кварк - подтверждённая формула"""
        strange = m_e * f1

        """Charm кварк - исправленная формула"""
        # m_c = m_e × (U/π)² (K/lnK)
        charm = m_e * f4 ** 2 * f5

        """Bottom кварк - исправленная формула"""
        # m_b = m_e × (U/π)² × p
        bottom_part = 8 * m_e * (f1 ** 2) * self.p

        """Top кварк - через bottom кварк"""
        # m_t = m_b × (K/lnK) × (1/p)    mₑ × 8.0781 × (U/π)² × (K/lnK)
        top_part = 8 * m_e * (f1 ** 2) * self.p * f5 / self.p

        """Протон - исправленная формула"""
        proton_part = m_e * f1 ** 2 * self.K / (f3 * f4 * f5)
        neutron_part = m_e * f1 ** 2 * self.K / (f3 * f4 * f5) * (1 + (self.K * self.p * self.p) / 10)
        W_boson = m_e * f2 * f3 ** 2 * f5 ** 3 * f1 ** 3 / f4 ** 2
        Z_boson = m_e * (((U / math.pi) ** 2 * lnK) / ((1 / self.p) ** 2 * (self.K / lnK) ** 2)) * (
                U / math.pi) ** 2 * (self.K / lnK)

        W_boson_compton_em = ( hbar_emergent /
                                      (((12 * np.sqrt(self.K * self.p) * ((math.log(self.N) / abs(math.log(self.K * self.p))) ** 4) * (
                self.N ** (-1 / 3))) * f2 * (f3 ** 2) * (f5 ** 3) * (f1 ** 3) / (f4 ** 2)) * c_emergent))

        print(f" W_boson_compton_em: {W_boson_compton_em:.3e}")
        W_boson_compton_test =  (math.log(self.K)**6 * abs(math.log(self.K*self.p))**5
                                         / (2304  * self.K**(15/2) * self.p**(5/2) * (math.log(self.N)**5)))
        print(f" W_boson_compton_test: {W_boson_compton_test:.3e}")

        HIGGS = m_e * f1 ** 2 * f5 / f3 * f5
        deuterium = (proton_part + neutron_part) * (1 - self.p / f5)
        alpha_He = 2 * (proton_part + neutron_part) * (1 - 4 * self.p / f5)

        # mₑ·f₁⁻¹·f₂²·f₃⁻¹·f₄·f₅⁰
        pion = m_e * f2 ** 3 * 1 / f3 * f4
        compton_pi_meson = (hbar_emergent) / (12 * np.sqrt(self.K * self.p) * ((math.log(self.N) / abs(math.log(self.K * self.p))) ** 4) * (
                self.N ** (-1 / 3)) *  (math.pi * (1 / np.sqrt(self.K * self.p) * R_universe / ((np.log(self.K) ** 2) / (
                4 * self.lambda_param ** 2 * self.K ** 2))) / self.lambda_param ** 2 * self.N ** (-1 / 6)))

        compton_pi_meson_em = ((np.log(self.K) * (self.p**2) * abs(np.log(self.K*self.p)**2))
                            / (2304 * math.pi** 3 * self.K**3 * np.log(self.N)**2))
        print(f" compton_pi_meson_em: {compton_pi_meson_em:.3e}")

        # (f₁·f₂·f₃²·f₄⁻²·f₅) · (U/π)·(1/p)
        # kaon            = m_e * f1 * f4/f2
        kaon = m_e * f1 * f4 / f2 * (f6 ** (1 / 2))

        eta_meson = m_e * f2 * f4 / f5 * f1
        # (f₁·f₂²·f₃²·f₄⁻¹) · (U / π)·lnK
        rho_meson = m_e * f1 ** 2 * f2 ** 3 * f3 ** 3 * 1 / f4

        neutrino_e = m_e * 1 / f4 ** 5 * 1 / f4
        neutrino_mu = m_e * f5 / f4 ** 5 * 1 / f4
        neutrino_tau = m_e * 1 / (f2 * (f4 ** 5))


        #atomic nucleos
        a_volume = (f5 ** 2) * (f6 ** 1)  # (K/lnK)^2 * (1+p)^1
        # Поверхностный коэффициент
        a_surface = (f5 ** 2) * (f6 ** (5 / 2))  # (K/lnK)^2 * (1+p)^(5/2)
        # Кулоновский коэффициент
        a_coulomb = (f2 ** 2 / self.K) * (f6 ** (11 / 2))  # (lnK)^2/K * (1+p)^(11/2)
        # Асимметрийный коэффициент
        a_asymmetry = (f5 ** 2) * (f6 ** 9)  # (K/lnK)^2 * (1+p)^9

        def nuclear_binding_correction(A, Z):
            """Semi-empirical binding в терминах структурных функций графа."""
            # Используем предвычисленные f1-f6
            a_volume = (f5 ** 2) * (f6 ** 1)  # (K/lnK)^2 * (1+p)^1
            a_surface = (f5 ** 2) * (f6 ** (5 / 2))  # (K/lnK)^2 * (1+p)^(5/2)
            a_coulomb = (f2 ** 2 / self.K) * (f6 ** (11 / 2))  # (lnK)^2/K * (1+p)^(11/2)
            a_asymmetry = (f5 ** 2) * (f6 ** 9)  # (K/lnK)^2 * (1+p)^9

            # Четностный член
            if A % 2 == 1:
                pairing_sign = 0
            elif Z % 2 == 0 and (A - Z) % 2 == 0:
                pairing_sign = 1
            elif Z % 2 == 1 and (A - Z) % 2 == 1:
                pairing_sign = -1
            else:
                pairing_sign = 0

            a_pairing = 12.0 * pairing_sign / math.sqrt(A)

            # Энергия связи ядра
            B = (a_volume * A
                 - a_surface * (A ** (2 / 3))
                 - a_coulomb * Z * (Z - 1) / (A ** (1 / 3))
                 - a_asymmetry * (A - 2 * Z) ** 2 / A
                 + a_pairing)

            # Коррекция массы
            return 1 - (B / A) / 931.494


        uran_238_avg_nucleon = (92 * proton_part + 146 * neutron_part) / 238
        uran_238 = 238 * uran_238_avg_nucleon * nuclear_binding_correction(238, Z=92)

        thoriy_232_avg_nucleon = (90 * proton_part + 142 * neutron_part) / 232
        thoriy_232 = 232 * thoriy_232_avg_nucleon * nuclear_binding_correction(232, Z=90)

        lithium6_avg_nucleon = (3 * proton_part + 3 * neutron_part) / 6
        lithium6 = 6 * lithium6_avg_nucleon * nuclear_binding_correction(6, Z=3)

        lithium7_avg_nucleon = (3 * proton_part + 4 * neutron_part) / 7
        lithium7 = 7 * lithium7_avg_nucleon * nuclear_binding_correction(7, Z=3)

        #lithium = 3*(proton_part+neutron_part)*(1 - 6*self.p/f5 + (self.p/f5)**2)*(f6**15/2)

        return {
            'hbar_emergent': hbar_emergent,
            'hbar_em': hbar_em,
            'l_em': l_em,
            'hbar': hbar_emergent,
            'lp': lp_emergent,
            'tp': tp_emergent,
            'c': c_emergent,
            'G': G_emergent,
            'kb': KB2,
            'Ep': Ep_emergent,
            'cosmo_lambda': cosmo_lambda,
            'R_universe': R_universe,
            'T_plank': T_plank,
            'ep0_em': ep0_em,
            'mu0_em': mu0_em,
            'e_plank': e_plank,
            'electron_charge': electron_charge,
            'alfa_em': alfa_em,
            'bor_orbital_radius': r_bor_emergent_final,
            'compton_electron_em': compton_electron_em,
            'compton_pi_meson_em': compton_pi_meson_em,
            'W_boson_compton_em': W_boson_compton_em,
            'plank_mass': M_planck_final,
            'electron_mass': electron_mass,
            'muon': muon,
            'tau': tau,
            'up_part': up_part,
            'down_part': down_part,
            'strange': strange,
            'charm': charm,
            'bottom_part': bottom_part,
            'top_part': top_part,
            'proton_part': proton_part,
            'neutron_part': neutron_part,
            'W_boson': W_boson,
            'HIGGS': HIGGS,
            'Z_boson': Z_boson,
            'deuterium': deuterium,
            'lithium6': lithium6,
            'lithium7': lithium7,
            'uran_238': uran_238,
            'thoriy_232': thoriy_232,
            'alpha_He': alpha_He,
            'pion': pion,
            'kaon': kaon,
            'eta_meson': eta_meson,
            'rho_meson': rho_meson,
            'neutrino_e': neutrino_e,
            #'neutrino_mu': neutrino_mu,
            'neutrino_tau': neutrino_tau,
        }

    def compare_with_classical(self, emergent_constants):
        """Сравнение с классическими значениями"""

        comparison = {}
        for key in ['hbar', 'lp', 'tp', 'c', 'G', 'kb',
                    'cosmo_lambda',
                    'T_plank',
                    'ep0_em',
                    'mu0_em',
                    'e_plank',
                    'electron_charge',
                    'alfa_em',
                    'bor_orbital_radius',
                    'compton_electron_em',
                    'compton_pi_meson_em',
                    'W_boson_compton_em',
                    'electron_mass',
                    'plank_mass',
                    'muon',
                    'neutrino_e',
                    'neutrino_mu',
                    'neutrino_tau'
                    'tau',
                    'up_part',
                    'down_part',
                    'strange',
                    'charm',
                    'bottom_part',
                    'top_part',
                    'proton_part',
                    'neutron_part',
                    'W_boson',
                    'HIGGS',
                    'Z_boson',
                    'deuterium',
                    'lithium6',
                    'lithium7',
                    'uran_238',
                    'thoriy_232',
                    'alpha_He',
                    'pion',
                    'kaon',
                    'eta_meson',
                    'rho_meson',
                    'neutrino_e',
                    'neutrino_mu',
                    'neutrino_tau'
                    ]:
            if key in emergent_constants and key in self.classical_constants:
                emergent_val = emergent_constants[key]
                classical_val = self.classical_constants[key]
                ratio = emergent_val / classical_val
                difference_orders = np.log10(abs(ratio)) if ratio != 0 else -np.inf

                comparison[key] = {
                    'emergent': emergent_val,
                    'classical': classical_val,
                    'ratio': ratio,
                    'difference_orders': difference_orders,
                    'match': abs(difference_orders) < 2.0  # Совпадение в пределах 2 порядков
                }

        return comparison

    def calculate_network_parameters(self):
        """Вычисление дополнительных параметров сети"""

        # Число узлов в объёме (M ~ N^{3/2})
        M = self.N ** (3 / 2)

        # Эффективная размерность (из вашей работы)
        d_eff = 1 + 4 * (1 - np.exp(-0.15 * (self.K - 3))) * np.exp(-20 * abs(self.p - 0.0527) ** 1.5)

        # Эмерджентная скорость (c_em = √(Kp))
        c_em = np.sqrt(self.K * self.p)

        return {
            'M_nodes': M,
            'effective_dimension': d_eff,
            'c_emergent_raw': c_em
        }


def print_results(calculator, emergent_constants, comparison, network_params):
    """Красивый вывод результатов"""
    print("ЭМЕРДЖЕНТНАЯ ФИЗИКА ИЗ СЕТИ МАЛОГО МИРА")

    print(f"\nПАРАМЕТРЫ СЕТИ:")
    print(f"K (локальная связность) = {calculator.K}")
    print(f"p (вероятность связи) = {calculator.p}")
    print(f"λ (спектральный масштаб) = {calculator.lambda_param:.2e}")
    print(f"N (голографическая энтропия) = {calculator.N:.2e}")
    print(f"M (узлов в объёме) = {network_params['M_nodes']:.2e}")
    print(f"Эффективная размерность = {network_params['effective_dimension']:.3f}")

    print(f"\nЛОКАЛЬНЫЕ СЕТЕВЫЕ ПАРАМЕТРЫ:")
    print(f"ħ_em (лок. квант действия) = {emergent_constants['hbar_em']:.6f}")
    print(f"l_em (лок. масштаб длины) = {emergent_constants['l_em']:.6f}")
    print(f"c_em (сырая скорость) = {network_params['c_emergent_raw']:.6f}")

    print(f"\nЭМЕРДЖЕНТНЫЕ ФИЗИЧЕСКИЕ КОНСТАНТЫ:")
    for key in ['hbar_emergent', 'hbar', 'c', 'G', 'kb', 'lp', 'tp', 'Ep',
                'cosmo_lambda',
                'ep0_em',
                'mu0_em',
                'electron_mass',
                'compton_pi_meson_em'
                'muon',
                'tau',
                'up_part',
                'down_part',
                'strange',
                'charm',
                'bottom_part',
                'top_part',
                'proton_part',
                'neutron_part',
                'W_boson',
                'HIGGS',
                'Z_boson',
                'deuterium',
                'lithium6',
                'lithium7',
                'uran_238',
                'thoriy_232',
                'alpha_He',
                'pion',
                'kaon',
                'eta_meson',
                'rho_meson'
                'neutrino_e',
                'neutrino_mu',
                'neutrino_tau'
                ]:
        if key in emergent_constants:
            val = emergent_constants[key]
            unit = {
                'hbar': 'Дж·с', 'c': 'м/с', 'G': 'м³/кг·с²',
                'kb': 'Дж/К', 'lp': 'м', 'tp': 'с', 'Ep': 'Дж', 'cosmo_lambda': 'м⁻²', 'T_plank': 'k', 'ep0_em': ' t ',
                'mu0_em': ' t', 'neutrino_e': ' ', 'neutrino_mu': ' ', 'neutrino_tau': ''
            }.get(key, '')
            print(f"{key:4} = {val:.6e} {unit}")

    print(f"\nСРАВНЕНИЕ С КЛАССИЧЕСКИМИ ЗНАЧЕНИЯМИ:")
    print("Константа      | Эмерджентная       | Классическая       | Отношение | Совпадение")

    for key, data in comparison.items():
        emergent = data['emergent']
        classical = data['classical']
        ratio = data['ratio']
        match = "✓" if data['match'] else "✗"

        print(f"{key:14} | {emergent:.4e} | {classical:.4e} | {ratio:8.3f} | {match}")

def lambda_emergent(N, K, p):
    """ Эмерджентный спектральный масштаб λ(N, K, p). """
    N = float(N)  # ключевая строка!
    # return 0.04414688903133314**2
    return (np.log(K * p) / np.log(N)) ** 2

def lambda_emergent2(N, p, K):
    lnKp = np.log(K * p)
    lnN = np.log(N)
    U = lnN / abs(lnKp)
    base = (lnKp / lnN) ** 2
    a = 0.9
    b = -1.11
    correction = 1 + a * abs(lnKp) / lnN + b / lnN
    return base * correction

def main():
    """Основная функция с параметрами"""

    # параметры сети
    K = 8.00
    p = 5.270179e-02  #  моделированое значение  5.270179e-02  в аттракторе - 5e-02
    N = 9.702e+122  #  моделированое значение 9.702e+122   в аттракторе - 1.047e+147
    M = 6 * N
    lambda_param = lambda_emergent(N, K, p)

    # Создание калькулятора
    calc = EmergentPhysicsCalculator(K, p, lambda_param, N, M)

    # Вычисления
    emergent_constants = calc.calculate_emergent_constants()
    comparison = calc.compare_with_classical(emergent_constants)
    network_params = calc.calculate_network_parameters()

    # Вывод основных результатов (исправленная строка)
    print_results(calc, emergent_constants, comparison, network_params)  # передаем network_params, а не particles

    # Анализ качества совпадения
    matches = sum(1 for data in comparison.values() if data['match'])
    total = len(comparison)
    print(f"Совпавших констант: {matches}/{total}")

    avg_error_orders = np.mean([abs(data['difference_orders']) for data in comparison.values()])
    print(f"Средняя ошибка (порядки): {avg_error_orders:.4f}")

    if matches >= 4 and avg_error_orders < 1.5:
        print("\n🎉 ОТЛИЧНОЕ СОВПАДЕНИЕ! Модель работает корректно.")
    elif matches >= 3:
        print("\n✅ ХОРОШЕЕ СОВПАДЕНИЕ. Модель требует небольшой настройки.")
    else:
        print("\n⚠️  ТРЕБУЕТСЯ ДОРАБОТКА. Проверьте параметры сети.")

if __name__ == "__main__":
    main()