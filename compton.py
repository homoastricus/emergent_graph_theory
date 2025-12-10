import numpy as np

import math


class FinalComptonFormulas:
    def __init__(self):
        self.K = 8.0
        self.p = 5.270179e-02
        self.N = 9.702e122

        self.h = 6.62607015e-34
        self.c = 299792458

        # Базовые величины
        self.lnK = math.log(self.K)
        self.lnKp = math.log(self.K * self.p)
        self.abs_lnKp = abs(self.lnKp)
        self.lnN = math.log(self.N)

        # f1-f6
        self.U = self.lnN / self.abs_lnKp
        self.f1 = self.lnN / (math.pi * self.abs_lnKp)  # U/π
        self.f2 = self.lnK  # lnK
        self.f3 = math.sqrt(self.K * self.p)  # √(Kp)
        self.f4 = 1 / self.p  # 1/p
        self.f5 = self.K / self.lnK  # K/lnK
        self.f6 = (self.K + self.p * self.K) / self.K  # 1 + p

        # Экспериментальные данные (согласованные λ и m)
        self.exp_data = {
            # Основные частицы (λ и m согласованы)
            "Электрон": {"λ": 2.4263102367e-12, "m": 9.1093837e-31},
            "Мюон": {"λ": 1.173444110e-14, "m": 1.883531627e-28},
            "Тау": {"λ": 6.977e-16, "m": 3.16754e-27},
            "Протон": {"λ": 1.32140985539e-15, "m": 1.67262192369e-27},
            "Нейтрон": {"λ": 1.31959090581e-15, "m": 1.67492749804e-27},
            "Пион": {"λ": 1.413e-15, "m": 2.398e-28},
            "Kaon": {"λ": 4.016e-16, "m": 8.802e-28},
            "Eta": {"λ": 3.729e-16, "m": 9.499e-28},
            "Rho": {"λ": 1.50e-15, "m": 1.46e-27},
            "W": {"λ": 2.450e-18, "m": 1.434e-25},
            "Z": {"λ": 1.36e-17, "m": 1.626e-25},
            "Higgs": {"λ": 9.92e-18, "m": 2.246e-25},

            # Кварки: λ вычисляем из масс через λ = ħ/(mc)
            "Up": {"m": 2.162e-30, "λ": 5.6e-13},  # Вычислим ниже
            "Down": {"m": 4.65e-30, "λ": 2.68e-13},
            "Strange": {"m": 9.4950e-29, "λ": 1.3e-14},
            "Charm": {"m": 1.2602e-27, "λ": 1.67e-15},
            "Bottom": {"m": 4.1780e-27, "λ": 5.09e-16},
            "Top": {"m": 3.0670e-25, "λ": 6.91e-18},
        }

        # Вычисляем λ из масс для кварков
        for name in ["Up", "Down", "Strange", "Charm", "Bottom", "Top"]:
            m = self.exp_data[name]["m"]
            self.exp_data[name]["λ"] = self.h / (m * self.c)

    # ТОЧНЫЕ ФОРМУЛЫ λ (из ваших рабочих формул)
    def electron_compton(self):
        return (2 * self.f2 ** 4 * self.lnKp ** 2 * math.sqrt(self.p)) / \
            (2304 * math.pi ** 2 * self.K ** 3.5 * self.lnN ** 2)

    def muon_compton(self):
        return (self.f2 ** 6 * self.abs_lnKp ** 2 * math.sqrt(self.p)) / \
            (4608 * math.pi ** 4 * self.K ** 5 * self.lnN ** 2)

    def tau_compton(self):
        return (self.f2 ** 5 * self.abs_lnKp ** 2 * self.p) / \
            (4608 * self.K ** 7.5 * self.lnN ** 2)

    def proton_compton(self):
        return (self.f2 ** 5 * self.abs_lnKp ** 2 * (self.p ** 1.5)) / \
            (1152 * math.pi ** 3 * self.K ** 5.5 * self.lnN ** 2)

    def neutron_compton(self):
        return (self.f2 ** 4 * self.abs_lnKp ** 4 * self.p) / \
            (1152 * math.pi ** 3 * self.K ** 3 * self.lnN ** 3)

    def pion_compton(self):
        return (self.f2 * self.p ** 2 * self.abs_lnKp ** 2) / \
            (2304 * math.pi ** 3 * self.K ** 3 * self.lnN ** 2)

    def kaon_compton(self):
        return (self.f2 ** 5 * self.abs_lnKp ** 3 * self.p ** (3 / 2)) / \
            (2304 * math.pi ** 2 * self.K ** (7 / 2) * self.lnN ** 3)

    def eta_compton(self):
        return (self.f2 ** 5 * self.abs_lnKp ** 5 * self.p ** 1.5) / \
            (2304 * self.K ** 4.5 * self.lnN ** 3)

    def rho_compton(self):
        return (self.f2 ** 3 * self.abs_lnKp ** 2 * self.p ** 2) / \
            (1152 * math.pi ** 4 * self.K ** 3.5 * self.lnN ** 2)

    def W_compton(self):
        return (self.f2 ** 6 * self.abs_lnKp ** 5) / \
            (2304 * self.K ** (15 / 2) * self.p ** (5 / 2) * self.lnN ** 5)

    def Z_compton(self):
        λ_e = self.electron_compton()
        mass_ratio = (self.f1**4 * self.f2) / (self.f4**2 * self.f5)
        return λ_e / mass_ratio

    def Higgs_compton(self):
        λ_e = self.electron_compton()
        mass_ratio = (self.f1 ** 2) * self.f5 / self.f3 * self.f5
        return λ_e / mass_ratio

    # ФОРМУЛЫ λ ДЛЯ КВАРКОВ ЧЕРЕЗ МАССОВЫЕ ОТНОШЕНИЯ (f1-f6)
    def up_quark_compton(self):
        """λ_up = λ_e / (m_up/m_e) где m_up/m_e = f3²·f4²/(f5²·f2²)"""
        λ_e = self.electron_compton()
        mass_ratio = (self.f3 ** 2 * self.f4 ** 2) / (self.f5 ** 2 * self.f2 ** 2)
        return λ_e / mass_ratio

    def down_quark_compton(self):
        """λ_down = λ_e / (m_down/m_e) где m_down/m_e = f2²·f1/(f3·f4·f5²)·f2"""
        λ_e = self.electron_compton()
        # Из вашей формулы: m_down = m_e * f2²·f1/(f3·f4·f5²)·f2
        mass_ratio = (self.f2 ** 2) * self.f1 / (self.f3 * self.f4 * self.f5 ** 2) * self.f2
        return λ_e / mass_ratio

    def strange_quark_compton(self):
        """λ_strange = λ_e / (m_strange/m_e) где m_strange = m_e * f1"""
        λ_e = self.electron_compton()
        mass_ratio = self.f1
        return λ_e / mass_ratio

    def charm_quark_compton(self):
        """λ_charm = λ_e / (m_charm/m_e) где m_charm = m_e * f4²·f5"""
        λ_e = self.electron_compton()
        mass_ratio = self.f4 ** 2 * self.f5
        return λ_e / mass_ratio

    def bottom_quark_compton(self):
        """λ_bottom = λ_e / (m_bottom/m_e) где m_bottom = 8·m_e·f1²·p"""
        λ_e = self.electron_compton()
        mass_ratio = 8 * (self.f1 ** 2) * self.p
        return λ_e / mass_ratio

    def top_quark_compton(self):
        """λ_top = λ_e / (m_top/m_e) где m_top = 8·m_e·f1²·f5"""
        λ_e = self.electron_compton()
        mass_ratio = 8 * (self.f1 ** 2) * self.f5
        return λ_e / mass_ratio

    def mass_from_lambda(self, lambda_c):
        """Стандартное преобразование λ → m"""
        if lambda_c <= 0:
            return float('inf')
        return self.h / (lambda_c * self.c)

    def mass_from_formula(self, particle_name):
        """Масса из ваших формул через f1-f6"""
        λ_func = getattr(self, f"{particle_name.lower().replace(' ', '_').replace('+', '').replace('-', '')}_compton")
        λ = λ_func()
        return self.mass_from_lambda(λ)

    def calculate_all(self):
        """Вычисление всех частиц"""
        particles = [
            ("Электрон", self.electron_compton),
            ("Мюон", self.muon_compton),
            ("Тау", self.tau_compton),
            ("Протон", self.proton_compton),
            ("Нейтрон", self.neutron_compton),
            ("Пион", self.pion_compton),
            ("Kaon", self.kaon_compton),
            ("Eta", self.eta_compton),
            ("Rho", self.rho_compton),
            ("W", self.W_compton),
            ("Z", self.Z_compton),
            ("Higgs", self.Higgs_compton),
            ("Up", self.up_quark_compton),
            ("Down", self.down_quark_compton),
            ("Strange", self.strange_quark_compton),
            ("Charm", self.charm_quark_compton),
            ("Bottom", self.bottom_quark_compton),
            ("Top", self.top_quark_compton),
        ]

        results = []
        for name, λ_func in particles:
            λ_calc = λ_func()
            m_calc = self.mass_from_lambda(λ_calc)

            exp = self.exp_data.get(name)
            if exp:
                λ_exp = exp.get("λ")
                m_exp = exp.get("m")

                λ_error = m_error = None
                if λ_exp is not None:
                    λ_error = abs(λ_calc - λ_exp) / λ_exp * 100 if λ_exp != 0 else float('inf')
                if m_exp is not None:
                    m_error = abs(m_calc - m_exp) / m_exp * 100 if m_exp != 0 else float('inf')
            else:
                λ_exp = m_exp = λ_error = m_error = None

            results.append({
                "name": name,
                "λ_calc": λ_calc,
                "λ_exp": λ_exp,
                "λ_error": λ_error,
                "m_calc": m_calc,
                "m_exp": m_exp,
                "m_error": m_error
            })

        return results

    def print_results(self, results):
        print(f"{'Частица':<12} {'λ расч (м)':<20} {'λ эксп (м)':<20} {'Ошибка λ%':<12} "
              f"{'m расч (кг)':<20} {'m эксп (кг)':<20} {'Ошибка m%':<12}")

        for r in results:
            name = r["name"]
            λ_calc = f"{r['λ_calc']:.3e}"
            m_calc = f"{r['m_calc']:.3e}"

            λ_exp = f"{r['λ_exp']:.3e}" if r['λ_exp'] is not None else "N/A"
            m_exp = f"{r['m_exp']:.3e}" if r['m_exp'] is not None else "N/A"

            def fmt_error(error):
                if error is None:
                    return "N/A"
                if error == float('inf'):
                    return "∞"
                if error < 1:
                    return f"\033[92m{error:.2f}%\033[0m"
                elif error < 10:
                    return f"\033[93m{error:.2f}%\033[0m"
                elif error < 50:
                    return f"\033[91m{error:.2f}%\033[0m"
                else:
                    return f"\033[91m{error:.1f}%\033[0m"

            λ_err = fmt_error(r['λ_error'])
            m_err = fmt_error(r['m_error'])

            print(f"{name:<12} {λ_calc:<20} {λ_exp:<20} {λ_err:<12} "
                  f"{m_calc:<20} {m_exp:<20} {m_err:<12}")


    def analyze_formulas(self):
        """Анализ формул через f1-f6"""

        print("АНАЛИЗ ФОРМУЛ ЧЕРЕЗ f1-f6:")


        # Значения f1-f6
        print(f"\nЗначения функций при K={self.K}, p={self.p}, N={self.N:.3e}:")
        print(f"f1 = U/π = {self.f1:.6f}")
        print(f"f2 = lnK = {self.f2:.6f}")
        print(f"f3 = √(Kp) = {self.f3:.6f}")
        print(f"f4 = 1/p = {self.f4:.2f}")
        print(f"f5 = K/lnK = {self.f5:.6f}")
        print(f"f6 = 1 + p = {self.f6:.6f}")

        # Массовые отношения
        print(f"\nМассовые отношения m/m_e:")
        print(f"  m_μ/m_e = 1/(λ_μ/λ_e) ≈ {1 / (self.muon_compton() / self.electron_compton()):.1f}")
        print(f"  m_p/m_e = 1/(λ_p/λ_e) ≈ {1 / (self.proton_compton() / self.electron_compton()):.1f}")
        print(f"  m_W/m_e = 1/(λ_W/λ_e) ≈ {1 / (self.W_compton() / self.electron_compton()):.1f}")

        # Проверка кварковых формул
        print(f"\nПроверка кварковых формул:")

        # Up кварк
        m_up_ratio = (self.f3 ** 2 * self.f4 ** 2) / (self.f5 ** 2 * self.f2 ** 2)
        print(f"  m_up/m_e = f3²·f4²/(f5²·f2²) = {m_up_ratio:.6f}")
        print(f"  Эксперимент: m_up/m_e ≈ {self.exp_data['Up']['m'] / self.exp_data['Электрон']['m']:.6f}")

        # Down кварк
        m_down_ratio = (self.f2 ** 2) * self.f1 / (self.f3 * self.f4 * self.f5 ** 2) * self.f2
        print(f"  m_down/m_e = f2³·f1/(f3·f4·f5²) = {m_down_ratio:.6f}")
        print(f"  Эксперимент: m_down/m_e ≈ {self.exp_data['Down']['m'] / self.exp_data['Электрон']['m']:.6f}")

        # Strange кварк
        print(f"  m_strange/m_e = f1 = {self.f1:.6f}")
        print(f"  Эксперимент: m_strange/m_e ≈ {self.exp_data['Strange']['m'] / self.exp_data['Электрон']['m']:.6f}")


# Запуск
if __name__ == "__main__":
    calc = FinalComptonFormulas()

    print("ФИНАЛЬНЫЕ ФОРМУЛЫ КОМПТОНОВСКИХ ДЛИН")

    # Вычисляем
    results = calc.calculate_all()
    calc.print_results(results)

    # Анализ
    calc.analyze_formulas()
