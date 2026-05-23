"""
ПОЛНЫЙ АНАЛИЗ СПЕКТРА МАСС В НОВОЙ ГРАФОВОЙ ТЕОРИИ
Полная версия с расширенным перебором для p = 1.243622e-31, N = 9.3555e+122
"""

from collections import defaultdict
import numpy as np
import math
from math import log, sqrt, pi, e

print("=" * 100)
print("ПОЛНЫЙ АНАЛИЗ СПЕКТРА МАСС В НОВОЙ ГРАФОВОЙ ТЕОРИИ")
print("=" * 100)


class CompleteParticleSpectrum:
    def __init__(self):
        # ПАРАМЕТРЫ НОВОЙ ТЕОРИИ (ОПТИМАЛЬНЫЕ)
        self.K = 8.0
        self.p = 1.243622e-31
        self.N = 9.3555e+122

        # ВЫЧИСЛЕНИЕ ВСЕХ БАЗОВЫХ ВЕЛИЧИН
        self.lnK = math.log(self.K)
        self.lnN = math.log(self.N)
        self.lnp = math.log(self.p)
        self.lnKp = math.log(self.K * self.p)
        self.abs_lnKp = abs(self.lnKp)

        # RG-параметры
        self.x = self.lnKp / self.lnN
        self.lambda_val = self.x ** 2
        self.U = self.lnN / self.abs_lnKp

        # Кластеризация и коррекция
        self.C_clust = 3 * (self.K - 2) / (4 * (self.K - 1)) * (1 - self.p) ** 3
        self.correction = 1 + (1 - self.C_clust) / self.lnN

        # Локальный квант действия
        self.hbar_em = (self.lnK ** 2) / (4 * self.lambda_val ** 2 * self.K ** 2) * self.correction

        # СТРУКТУРНЫЕ ФУНКЦИИ
        self.f1 = self.U / pi
        self.f2 = self.lnK
        self.f3 = math.sqrt(self.K * self.p)
        self.f4 = 1.0 / self.p
        self.f5 = self.K / self.lnK
        self.f6 = 1.0 + self.p

        # Фундаментальные величины
        self.V = self.f1 ** (2 / 3) * self.hbar_em ** 3 * self.lnN ** 2
        self.S_nonloc = -self.lnp
        self.S_loc = self.lnK
        self.S_glob = self.lnN
        self.S_spec = -2 * math.log(self.lambda_val) if self.lambda_val > 0 else 0

        # Дополнительные комбинации
        self.U_inv = 1.0 / self.U
        self.V_inv = 1.0 / self.V
        self.lambda_inv = 1.0 / self.lambda_val if self.lambda_val > 0 else 0
        self.N_minus_1_3 = self.N ** (-1 / 3)
        self.N_minus_1_6 = self.N ** (-1 / 6)

        print(f"\n⚙️  ПАРАМЕТРЫ СЕТИ (НОВАЯ ТЕОРИЯ):")
        print(f"   K = {self.K}")
        print(f"   p = {self.p:.6e}")
        print(f"   N = {self.N:.6e}")
        print(f"   U = lnN/|ln(Kp)| = {self.U:.6f}")
        print(f"   λ = (ln(Kp)/lnN)² = {self.lambda_val:.6e}")
        print(f"   V = {self.V:.6e}")

        print(f"\n🎯 СТРУКТУРНЫЕ ФУНКЦИИ:")
        print(f"   f₁ = U/π = {self.f1:.6f}")
        print(f"   f₂ = lnK = {self.f2:.6f}")
        print(f"   f₃ = √(Kp) = {self.f3:.6e}")
        print(f"   f₄ = 1/p = {self.f4:.6e}")
        print(f"   f₅ = K/lnK = {self.f5:.6f}")
        print(f"   f₆ = 1+p = {self.f6:.6f}")
        print(f"   S_nonloc = -ln(p) = {self.S_nonloc:.6f}")
        print(f"   S_glob = ln(N) = {self.S_glob:.6f}")

        # МАССА ЭЛЕКТРОНА
        self.m_e_theoretical = 8 * self.f3 * (self.U ** 6) * (self.V ** 3) * self.lnK * self.N ** (-1 / 3)
        self.m_e_exp = 9.1093837e-31
        self.calibration = self.m_e_exp / self.m_e_theoretical

        print(f"\n📊 МАССА ЭЛЕКТРОНА:")
        print(f"   m_e (теоретическая) = {self.m_e_theoretical:.6e} кг")
        print(f"   m_e (эксперимент)   = {self.m_e_exp:.6e} кг")
        print(f"   Отношение = {self.m_e_theoretical / self.m_e_exp:.6f}")

        # БИБЛИОТЕКА ВСЕХ ДОСТУПНЫХ ВЕЛИЧИН ДЛЯ ПЕРЕБОРА
        self.all_variables = {
            # Базовые структурные функции
            'f1': self.f1, 'f2': self.f2, 'f3': self.f3,
            'f4': self.f4, 'f5': self.f5, 'f6': self.f6,
            # Фундаментальные величины
            'V': self.V, 'U': self.U, 'lambda': self.lambda_val,
            # Энтропии
            'S_nonloc': self.S_nonloc, 'S_loc': self.S_loc, 'S_glob': self.S_glob,
            'S_spec': self.S_spec,
            # Обратные величины
            'U_inv': self.U_inv, 'V_inv': self.V_inv, 'lambda_inv': self.lambda_inv,
            # Логарифмы
            'lnK': self.lnK, 'lnN': self.lnN, 'lnp': self.lnp,
            'abs_lnKp': self.abs_lnKp,
            # Голографические факторы
            'N_1_3': self.N_minus_1_3, 'N_1_6': self.N_minus_1_6,
            # Константы
            'pi': pi, 'e': e,
            # Базовые параметры
            'K': self.K, 'p': self.p, 'Kp': self.K * self.p,
        }

        # Добавляем квадраты и кубы
        for key in list(self.all_variables.keys()):
            val = self.all_variables[key]
            if abs(val) < 1e100 and abs(val) > 1e-100:
                self.all_variables[f"{key}²"] = val ** 2
                self.all_variables[f"{key}³"] = val ** 3
            if val > 0 and val < 1e50:
                self.all_variables[f"√{key}"] = math.sqrt(val)

        print(f"\n📚 БИБЛИОТЕКА: {len(self.all_variables)} величин")

    def calculate_mass_from_list(self, var_list):
        """Вычисляет массу из списка переменных (каждая в степени 1)"""
        mass = self.m_e_theoretical * self.calibration
        for var_name in var_list:
            if var_name in self.all_variables:
                mass *= self.all_variables[var_name]
        return mass

    def calculate_mass_from_exponents(self, exponents):
        """Вычисляет массу из словаря {переменная: степень}"""
        mass = self.m_e_theoretical * self.calibration
        for var_name, power in exponents.items():
            if var_name in self.all_variables:
                mass *= self.all_variables[var_name] ** power
        return mass

    def get_particle_catalog(self):
        """Полный каталог частиц"""
        return {
            # ====== ЛЕПТОНЫ ======
            'e⁻': {'mass_kg': 9.10938356e-31, 'type': 'lepton', 'charge': -1},
            'e⁺': {'mass_kg': 9.10938356e-31, 'type': 'lepton', 'charge': 1},
            'μ⁻': {'mass_kg': 1.883531627e-28, 'type': 'lepton', 'charge': -1},
            'μ⁺': {'mass_kg': 1.883531627e-28, 'type': 'lepton', 'charge': 1},
            'τ⁻': {'mass_kg': 3.16754e-27, 'type': 'lepton', 'charge': -1},
            'τ⁺': {'mass_kg': 3.16754e-27, 'type': 'lepton', 'charge': 1},
            'ν_e': {'mass_kg': 1.8e-38, 'type': 'lepton', 'charge': 0},
            'ν_μ': {'mass_kg': 9.0e-38, 'type': 'lepton', 'charge': 0},
            'ν_τ': {'mass_kg': 1.8e-37, 'type': 'lepton', 'charge': 0},

            # ====== КВАРКИ ======
            'u': {'mass_kg': 2.16e-30, 'type': 'quark', 'charge': 2 / 3},
            'd': {'mass_kg': 4.67e-30, 'type': 'quark', 'charge': -1 / 3},
            's': {'mass_kg': 9.36e-29, 'type': 'quark', 'charge': -1 / 3},
            'c': {'mass_kg': 1.27e-27, 'type': 'quark', 'charge': 2 / 3},
            'b': {'mass_kg': 4.18e-27, 'type': 'quark', 'charge': -1 / 3},
            't': {'mass_kg': 3.08e-25, 'type': 'quark', 'charge': 2 / 3},

            # ====== КАЛИБРОВОЧНЫЕ БОЗОНЫ ======
            'γ': {'mass_kg': 0, 'type': 'boson', 'charge': 0},
            'W⁺': {'mass_kg': 1.433e-25, 'type': 'boson', 'charge': 1},
            'W⁻': {'mass_kg': 1.433e-25, 'type': 'boson', 'charge': -1},
            'Z⁰': {'mass_kg': 1.625e-25, 'type': 'boson', 'charge': 0},
            'g': {'mass_kg': 0, 'type': 'boson', 'charge': 0},
            'H⁰': {'mass_kg': 2.246e-25, 'type': 'boson', 'charge': 0},

            # ====== ЛЁГКИЕ МЕЗОНЫ ======
            'π⁰': {'mass_kg': 2.406e-28, 'type': 'meson', 'charge': 0},
            'π⁺': {'mass_kg': 2.488e-28, 'type': 'meson', 'charge': 1},
            'π⁻': {'mass_kg': 2.488e-28, 'type': 'meson', 'charge': -1},
            'K⁺': {'mass_kg': 8.806e-28, 'type': 'meson', 'charge': 1},
            'K⁻': {'mass_kg': 8.806e-28, 'type': 'meson', 'charge': -1},
            'K⁰': {'mass_kg': 8.954e-28, 'type': 'meson', 'charge': 0},
            'η': {'mass_kg': 9.491e-28, 'type': 'meson', 'charge': 0},
            'η\'': {'mass_kg': 1.708e-27, 'type': 'meson', 'charge': 0},

            # ====== ВЕКТОРНЫЕ МЕЗОНЫ ======
            'ρ⁺': {'mass_kg': 1.253e-27, 'type': 'meson', 'charge': 1},
            'ρ⁰': {'mass_kg': 1.253e-27, 'type': 'meson', 'charge': 0},
            'ρ⁻': {'mass_kg': 1.253e-27, 'type': 'meson', 'charge': -1},
            'ω': {'mass_kg': 1.410e-27, 'type': 'meson', 'charge': 0},
            'φ': {'mass_kg': 1.838e-27, 'type': 'meson', 'charge': 0},
            'K*⁺': {'mass_kg': 1.415e-27, 'type': 'meson', 'charge': 1},
            'K*⁰': {'mass_kg': 1.419e-27, 'type': 'meson', 'charge': 0},

            # ====== ТЯЖЁЛЫЕ КВАРКОНИИ ======
            'J/ψ': {'mass_kg': 5.525e-27, 'type': 'meson', 'charge': 0},
            'ψ(2S)': {'mass_kg': 6.124e-27, 'type': 'meson', 'charge': 0},
            'Υ(1S)': {'mass_kg': 1.694e-26, 'type': 'meson', 'charge': 0},
            'Υ(2S)': {'mass_kg': 1.835e-26, 'type': 'meson', 'charge': 0},
            'Υ(3S)': {'mass_kg': 1.900e-26, 'type': 'meson', 'charge': 0},

            # ====== ОЧАРОВАННЫЕ МЕЗОНЫ ======
            'D⁰': {'mass_kg': 3.340e-27, 'type': 'meson', 'charge': 0},
            'D⁺': {'mass_kg': 3.354e-27, 'type': 'meson', 'charge': 1},
            'D*⁰': {'mass_kg': 3.403e-27, 'type': 'meson', 'charge': 0},
            'D*⁺': {'mass_kg': 3.414e-27, 'type': 'meson', 'charge': 1},
            'D_s⁺': {'mass_kg': 3.672e-27, 'type': 'meson', 'charge': 1},

            # ====== ПРЕЛЕСТНЫЕ МЕЗОНЫ ======
            'B⁰': {'mass_kg': 9.430e-27, 'type': 'meson', 'charge': 0},
            'B⁺': {'mass_kg': 9.424e-27, 'type': 'meson', 'charge': 1},
            'B_s⁰': {'mass_kg': 1.004e-26, 'type': 'meson', 'charge': 0},
            'B_c⁺': {'mass_kg': 1.783e-26, 'type': 'meson', 'charge': 1},

            # ====== ЛЁГКИЕ БАРИОНЫ ======
            'p': {'mass_kg': 1.6726219e-27, 'type': 'baryon', 'charge': 1},
            'n': {'mass_kg': 1.6749275e-27, 'type': 'baryon', 'charge': 0},
            'Λ': {'mass_kg': 1.992e-27, 'type': 'baryon', 'charge': 0},
            'Σ⁺': {'mass_kg': 2.129e-27, 'type': 'baryon', 'charge': 1},
            'Σ⁰': {'mass_kg': 2.134e-27, 'type': 'baryon', 'charge': 0},
            'Σ⁻': {'mass_kg': 2.139e-27, 'type': 'baryon', 'charge': -1},
            'Ξ⁰': {'mass_kg': 2.347e-27, 'type': 'baryon', 'charge': 0},
            'Ξ⁻': {'mass_kg': 2.359e-27, 'type': 'baryon', 'charge': -1},
            'Ω⁻': {'mass_kg': 2.989e-27, 'type': 'baryon', 'charge': -1},

            # ====== ОЧАРОВАННЫЕ БАРИОНЫ ======
            'Λ_c⁺': {'mass_kg': 3.733e-27, 'type': 'baryon', 'charge': 1},
            'Σ_c⁺⁺': {'mass_kg': 3.867e-27, 'type': 'baryon', 'charge': 2},
            'Σ_c⁺': {'mass_kg': 3.864e-27, 'type': 'baryon', 'charge': 1},
            'Σ_c⁰': {'mass_kg': 3.861e-27, 'type': 'baryon', 'charge': 0},
            'Ξ_c⁺': {'mass_kg': 4.066e-27, 'type': 'baryon', 'charge': 1},
            'Ξ_c⁰': {'mass_kg': 4.069e-27, 'type': 'baryon', 'charge': 0},
            'Ω_c⁰': {'mass_kg': 4.376e-27, 'type': 'baryon', 'charge': 0},

            # ====== ПРЕЛЕСТНЫЕ БАРИОНЫ ======
            'Λ_b⁰': {'mass_kg': 1.133e-26, 'type': 'baryon', 'charge': 0},
            'Σ_b⁺': {'mass_kg': 1.167e-26, 'type': 'baryon', 'charge': 1},
            'Σ_b⁻': {'mass_kg': 1.168e-26, 'type': 'baryon', 'charge': -1},
            'Ξ_b⁰': {'mass_kg': 1.192e-26, 'type': 'baryon', 'charge': 0},
            'Ξ_b⁻': {'mass_kg': 1.193e-26, 'type': 'baryon', 'charge': -1},
            'Ω_b⁻': {'mass_kg': 1.212e-26, 'type': 'baryon', 'charge': -1},

            # ====== ЭКЗОТИЧЕСКИЕ ЧАСТИЦЫ ======
            'X(3872)': {'mass_kg': 6.918e-27, 'type': 'exotic', 'charge': 0},
            'Z_c(3900)': {'mass_kg': 6.975e-27, 'type': 'exotic', 'charge': 1},
            'Y(4260)': {'mass_kg': 9.135e-27, 'type': 'exotic', 'charge': 0},
            'P_c(4380)': {'mass_kg': 7.825e-27, 'type': 'exotic', 'charge': 1},
            'P_c(4450)': {'mass_kg': 7.950e-27, 'type': 'exotic', 'charge': 1},
            'T_cc⁺': {'mass_kg': 6.850e-27, 'type': 'exotic', 'charge': 1},
        }

    def find_best_formula(self, target_mass, particle_type, max_vars=4):
        """Поиск лучшей формулы для заданной массы"""
        m_e = self.m_e_exp
        target_ratio = target_mass / m_e

        var_names = list(self.all_variables.keys())

        # Ограничиваем набор в зависимости от типа частицы
        if particle_type == 'lepton':
            priority_vars = ['f1', 'f2', 'f3', 'V', 'U', 'S_nonloc']
        elif particle_type == 'quark':
            priority_vars = ['f1', 'f2', 'f3', 'f4', 'f5', 'V', 'U']
        elif particle_type == 'boson':
            priority_vars = ['f1', 'V', 'U', 'lambda', 'S_glob']
        elif particle_type == 'meson':
            priority_vars = ['f1', 'f2', 'f3', 'f4', 'V', 'U', 'S_nonloc']
        elif particle_type == 'baryon':
            priority_vars = ['f1', 'f2', 'f3', 'f4', 'f5', 'V', 'U', 'S_glob']
        else:
            priority_vars = var_names[:20]

        # Ограничиваем количество переменных
        search_vars = [v for v in priority_vars if v in self.all_variables][:15]

        best_match = None
        best_error = float('inf')

        # Перебор 1-3 переменных с целыми степенями
        from itertools import combinations, product

        for n_vars in range(1, max_vars + 1):
            for combo in combinations(search_vars, n_vars):
                # Степени от -3 до 3
                for powers in product(range(-3, 4), repeat=n_vars):
                    if all(p == 0 for p in powers):
                        continue

                    exponents = {var: pow for var, pow in zip(combo, powers) if pow != 0}

                    try:
                        mass = self.calculate_mass_from_exponents(exponents)
                        ratio = mass / m_e

                        if ratio <= 0:
                            continue

                        # Логарифмическая ошибка для больших масс
                        if target_ratio > 100:
                            error = abs(math.log10(ratio) - math.log10(target_ratio))
                        else:
                            error = abs(ratio - target_ratio) / target_ratio

                        # Штраф за сложность
                        complexity = len(exponents) + sum(abs(p) for p in powers)
                        error *= (1 + 0.05 * complexity)

                        if error < best_error:
                            best_error = error
                            best_match = {
                                'exponents': exponents,
                                'theoretical': mass,
                                'target': target_mass,
                                'error': error,
                                'formula': self.format_exponents(exponents),
                                'complexity': complexity
                            }
                    except:
                        continue

        return best_match

    def format_exponents(self, exponents):
        """Форматирует экспоненты в читаемую строку"""
        if not exponents:
            return "1"
        parts = []
        for var, pow in sorted(exponents.items()):
            if pow == 1:
                parts.append(var)
            elif pow == -1:
                parts.append(f"{var}⁻¹")
            else:
                parts.append(f"{var}^{pow}")
        return " × ".join(parts)

    def analyze_all_particles(self):
        """Анализ всех частиц"""
        catalog = self.get_particle_catalog()
        results = []

        print(f"\n🔍 АНАЛИЗ {len(catalog)} ЧАСТИЦ")
        print(
            f"{'Статус':<4} {'Частица':<10} {'Тип':<10} {'Масса эксп.':<14} {'Масса теор.':<14} {'Ошибка %':<10} {'Формула':<35}")

        for name, data in catalog.items():
            target_mass = data['mass_kg']

            if target_mass == 0:
                results.append({
                    'name': name,
                    'type': data['type'],
                    'charge': data['charge'],
                    'target': 0,
                    'theoretical': 0,
                    'error': 0,
                    'formula': '0'
                })
                print(f"{'✅':<4} {name:<10} {data['type']:<10} {'0':<14} {'0':<14} {'0.00':<10} {'-':<35}")
                continue

            best = self.find_best_formula(target_mass, data['type'])

            if best:
                error_percent = best['error'] * 100
                results.append({
                    'name': name,
                    'type': data['type'],
                    'charge': data['charge'],
                    'target': target_mass,
                    'theoretical': best['theoretical'],
                    'error': best['error'],
                    'formula': best['formula'],
                    'exponents': best['exponents']
                })

                if error_percent < 10:
                    status = "✅"
                elif error_percent < 30:
                    status = "⚠️"
                else:
                    status = "❌"

                print(f"{status:<4} {name:<10} {data['type']:<10} {target_mass:<14.3e} "
                      f"{best['theoretical']:<14.3e} {error_percent:<10.2f} "
                      f"{best['formula'][:35]:<35}")
            else:
                results.append({
                    'name': name,
                    'type': data['type'],
                    'charge': data['charge'],
                    'target': target_mass,
                    'theoretical': 0,
                    'error': 1.0,
                    'formula': '?'
                })
                print(f"{'❌':<4} {name:<10} {data['type']:<10} {target_mass:<14.3e} {'---':<14} {'>100':<10} {'?':<35}")

        return results

    def print_statistics(self, results):
        """Выводит подробную статистику"""
        massive = [r for r in results if r['target'] > 0]

        if not massive:
            return

        print("\n" + "=" * 100)
        print("СТАТИСТИКА ПО ТИПАМ ЧАСТИЦ")
        print("=" * 100)

        by_type = defaultdict(list)
        for r in massive:
            by_type[r['type']].append(r)

        print(f"{'Тип':<12} {'Всего':<8} {'<5%':<8} {'<10%':<8} {'<20%':<8} {'<30%':<8} {'>30%':<8}")
        print("-" * 60)

        for ptype in ['lepton', 'quark', 'boson', 'meson', 'baryon', 'exotic']:
            if ptype in by_type:
                particles = by_type[ptype]
                total = len(particles)
                lt5 = len([p for p in particles if p['error'] * 100 < 5])
                lt10 = len([p for p in particles if p['error'] * 100 < 10])
                lt20 = len([p for p in particles if p['error'] * 100 < 20])
                lt30 = len([p for p in particles if p['error'] * 100 < 30])
                gt30 = total - lt30

                print(f"{ptype:<12} {total:<8} {lt5:<8} {lt10:<8} {lt20:<8} {lt30:<8} {gt30:<8}")

        # Общая статистика
        total = len(massive)
        excellent = len([r for r in massive if r['error'] * 100 < 10])
        good = len([r for r in massive if r['error'] * 100 < 30])

        print("\n" + "=" * 100)
        print("ОБЩАЯ СТАТИСТИКА")
        print(f"Всего массивных частиц: {total}")
        print(f"Отлично описано (<10%): {excellent} ({excellent / total * 100:.1f}%)")
        print(f"Хорошо описано (<30%): {good} ({good / total * 100:.1f}%)")

        errors = [r['error'] * 100 for r in massive]
        print(f"Средняя ошибка: {np.mean(errors):.2f}%")
        print(f"Медианная ошибка: {np.median(errors):.2f}%")

    def predict_new_particles(self, num_predictions=50):
        """Предсказывает новые частицы"""
        print(f"ПРЕДСКАЗАНИЕ НОВЫХ ЧАСТИЦ (ТОП-{num_predictions})")

        catalog = self.get_particle_catalog()
        known_masses = [data['mass_kg'] for data in catalog.values() if data['mass_kg'] > 0]
        m_e = self.m_e_exp

        predictions = []

        # Перебор возможных формул
        search_vars = ['f1', 'f2', 'f3', 'V', 'U', 'S_nonloc', 'lambda']
        from itertools import combinations, product

        for n_vars in range(1, 4):
            for combo in combinations(search_vars, n_vars):
                for powers in product(range(-3, 4), repeat=n_vars):
                    if all(p == 0 for p in powers):
                        continue

                    exponents = {var: pow for var, pow in zip(combo, powers) if pow != 0}

                    try:
                        mass = self.calculate_mass_from_exponents(exponents)
                        ratio = mass / m_e

                        # Разумный диапазон масс (от 0.1 eV до 100 TeV)
                        if ratio < 0.1 or ratio > 2e11:
                            continue

                        # Проверяем, не совпадает ли с известной частицей
                        is_new = True
                        for known_mass in known_masses:
                            if known_mass > 0:
                                known_ratio = known_mass / m_e
                                if 0.7 < ratio / known_ratio < 1.4:  # ±30%
                                    is_new = False
                                    break

                        if is_new:
                            complexity = len(exponents) + sum(abs(p) for p in powers)
                            predictions.append({
                                'mass_kg': mass,
                                'mass_me': ratio,
                                'mass_gev': mass * 5.61e35 / 1e9,
                                'exponents': exponents,
                                'formula': self.format_exponents(exponents),
                                'complexity': complexity
                            })
                    except:
                        continue

        # Убираем дубликаты
        unique_preds = []
        seen_masses = set()
        for p in sorted(predictions, key=lambda x: (x['complexity'], x['mass_kg'])):
            mass_key = round(p['mass_kg'], 30)
            if mass_key not in seen_masses:
                seen_masses.add(mass_key)
                unique_preds.append(p)

        # Выводим топ
        print(f"{'№':<4} {'Масса (GeV)':<14} {'Масса (кг)':<14} {'Сложность':<10} {'Формула':<40}")
        print("-" * 90)

        for i, p in enumerate(unique_preds[:num_predictions]):
            print(
                f"{i + 1:<4} {p['mass_gev']:<14.6f} {p['mass_kg']:<14.3e} {p['complexity']:<10} {p['formula'][:40]:<40}")

        return unique_preds

    def export_to_excel(self, results, filename="particle_spectrum_new_theory.xlsx"):
        """Экспорт в Excel"""
        try:
            import pandas as pd

            data = []
            for r in results:
                if r['target'] > 0:
                    error_percent = r['error'] * 100
                else:
                    error_percent = 0

                row = {
                    'Частица': r['name'],
                    'Тип': r['type'],
                    'Заряд': r.get('charge', 0),
                    'Масса эксп. (кг)': r['target'],
                    'Масса теор. (кг)': r['theoretical'],
                    'Ошибка (%)': error_percent,
                    'Формула': r['formula']
                }

                if 'exponents' in r and r['exponents']:
                    for var, exp in r['exponents'].items():
                        row[f'exp_{var}'] = exp

                data.append(row)

            df = pd.DataFrame(data)
            df = df.sort_values(['Тип', 'Масса эксп. (кг)'])

            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Все частицы', index=False)

                for ptype in df['Тип'].unique():
                    df_type = df[df['Тип'] == ptype]
                    df_type.to_excel(writer, sheet_name=ptype[:31], index=False)

                df_best = df[df['Ошибка (%)'] < 30]
                df_best.to_excel(writer, sheet_name='Лучшие (<30%)', index=False)

            print(f"\n✅ Результаты сохранены в {filename}")
            return True
        except Exception as e:
            print(f"\n⚠️ Ошибка экспорта: {e}")
            return False

    def run_full_analysis(self):
        """Запуск полного анализа"""
        # 1. Анализ известных частиц
        results = self.analyze_all_particles()

        # 2. Статистика
        self.print_statistics(results)

        # 3. Предсказание новых частиц
        predictions = self.predict_new_particles(50)

        # 4. Экспорт
        self.export_to_excel(results)

        return results, predictions

# ЗАПУСК
if __name__ == "__main__":
    spectrum = CompleteParticleSpectrum()
    results, predictions = spectrum.run_full_analysis()
