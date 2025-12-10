import math


class UnifiedTheoryTable:
    def __init__(self):
        self.K = 8.0
        self.p = 5.270179e-02
        self.N = 9.702e122
        self.t_P = 5.39e-44

        # Вычисляем все базовые величины
        self.lnK = math.log(self.K)
        self.lnKp = math.log(self.K * self.p)
        self.lnN = math.log(self.N)
        self.U = self.lnN / abs(self.lnKp)

        # Основные f-функции
        self.f1 = self.U / math.pi
        self.f2 = self.lnK
        self.f3 = math.sqrt(self.K * self.p)
        self.f4 = 1.0 / self.p
        self.f5 = self.K / self.lnK
        self.f6 = 1.0 + self.p

        # Дополнительные функции
        self.g1 = self.U / (math.pi * self.p)
        self.g2 = math.log(self.U) / self.lnK
        self.q1 = 1.0 / (self.K * self.p ** 2)
        self.q3 = self.f2 ** 2 / self.f5

        # Специальные множители
        self.U_p = self.U / self.p

    def get_mass_kg(self, particle_name):
        """Массы частиц в кг"""
        masses = {
            'W_boson': 1.433e-25,
            'Z_boson': 1.625e-25,
            'Higgs': 2.246e-25,
            'muon': 1.884e-28,
            'tau': 3.167e-27,
            'electron': 9.109e-31,
            'neutron': 1.675e-27,
            'pion+': 2.488e-28,
            'kaon+': 8.806e-28,
            'pion0': 2.406e-28,
            'eta': 9.491e-28,
            'rho': 1.253e-27,
            'Delta++': 2.208e-27,
            'top_quark': 3.08e-25,
        }
        return masses.get(particle_name, 0)

    def get_tau_exp(self, particle_name):
        """Экспериментальные времена жизни"""
        taus = {
            'W_boson': 3.07e-25,
            'Z_boson': 3.08e-25,
            'Higgs': 1.56e-22,
            'muon': 2.197e-6,
            'tau': 2.906e-13,
            'electron': float('inf'),
            'neutron': 879.4,
            'pion+': 2.603e-8,
            'kaon+': 1.238e-8,
            'pion0': 8.52e-17,
            'eta': 5.0e-19,
            'rho': 4.45e-24,
            'Delta++': 5.6e-24,
            'top_quark': 5.0e-25,
        }
        return taus.get(particle_name, 0)

    def get_particle_type(self, particle_name):
        """Типы частиц"""
        types = {
            'W_boson': 'gauge_boson',
            'Z_boson': 'gauge_boson',
            'Higgs': 'gauge_boson',
            'muon': 'fermion',
            'tau': 'fermion',
            'electron': 'fermion',
            'neutron': 'baryon',
            'pion+': 'meson',
            'kaon+': 'meson',
            'pion0': 'meson',
            'eta': 'meson',
            'rho': 'meson',
            'Delta++': 'baryon',
            'top_quark': 'quark',
        }
        return types.get(particle_name, 'unknown')

    def expand_formula(self, particle_name):
        """Раскрывает формулу через базовые параметры"""
        formulas = {
            'W_boson': {
                'compact': 'τ = t_P × f₁¹¹ f₂⁻⁴ f₃⁴ f₄⁻³ f₅⁴ f₆³',
                'expanded': f'τ = t_P × (U/π)¹¹ × (lnK)⁻⁴ × (√(Kp))⁴ × (1/p)⁻³ × (K/lnK)⁴ × (1+p)³'
            },
            'Z_boson': {
                'compact': 'τ = t_P × f₁¹¹ f₂⁴ f₃⁻² f₄⁻³ f₅⁻² f₆⁻⁴',
                'expanded': f'τ = t_P × (U/π)¹¹ × (lnK)⁴ × (√(Kp))⁻² × (1/p)⁻³ × (K/lnK)⁻² × (1+p)⁻⁴'
            },
            'Higgs': {
                'compact': 'τ = t_P × f₁¹¹ f₂⁰ f₃⁻⁴ f₄⁻³ f₅⁴ f₆⁰',
                'expanded': f'τ = t_P × (U/π)¹¹ × (√(Kp))⁻⁴ × (1/p)⁻³ × (K/lnK)⁴'
            },
            'muon': {
                'compact': 'τ = t_P × f₁¹¹ f₄⁻³ × g₁² g₂² × (U/p)⁴',
                'expanded': f'τ = t_P × (U/π)¹¹ × (1/p)⁻³ × (U/(πp))² × (lnU/lnK)² × (U/p)⁴'
            },
            'tau': {
                'compact': 'τ = t_P × f₁¹¹ f₄⁻³ × g₁⁻¹ g₂⁰ × (U/p)⁴',
                'expanded': f'τ = t_P × (U/π)¹¹ × (1/p)⁻³ × (U/(πp))⁻¹ × (U/p)⁴'
            },
            'electron': {
                'compact': 'τ = ∞ (стабильная)',
                'expanded': 'τ = ∞ (стабильная: g₁→∞ или (U/p)⁰)'
            },
            'neutron': {
                'compact': 'τ = t_P × f₁¹¹ f₄³ × (U/p)⁵',
                'expanded': f'τ = t_P × (U/π)¹¹ × (1/p)³ × (U/p)⁵'
            },
            'pion+': {
                'compact': 'τ = t_P × f₁⁴ f₄³ × q₁² q₃² × (U/p)⁵',
                'expanded': f'τ = t_P × (U/π)⁴ × (1/p)³ × [1/(Kp²)]² × [(lnK)²/(K/lnK)]² × (U/p)⁵'
            },
            'kaon+': {
                'compact': 'τ = t_P × f₁⁹ f₄⁻² × q₃² × (U/p)⁵',
                'expanded': f'τ = t_P × (U/π)⁹ × (1/p)⁻² × [(lnK)²/(K/lnK)]² × (U/p)⁵'
            },
            'pion0': {
                'compact': 'τ = t_P × f₁⁷ f₄¹ × (U/p)³',
                'expanded': f'τ = t_P × (U/π)⁷ × (1/p)¹ × (U/p)³'
            },
            'eta': {
                'compact': 'τ = t_P × f₁⁷ f₄⁻² × q₁² q₃¹ × (U/p)³',
                'expanded': f'τ = t_P × (U/π)⁷ × (1/p)⁻² × [1/(Kp²)]² × [(lnK)²/(K/lnK)]¹ × (U/p)³'
            },
            'rho': {
                'compact': 'τ = t_P × f₁⁶ f₄² × q₁² q₃²',
                'expanded': f'τ = t_P × (U/π)⁶ × (1/p)² × [1/(Kp²)]² × [(lnK)²/(K/lnK)]²'
            },
            'Delta++': {
                'compact': 'τ = t_P × f₁⁶ f₄³ × q₁² q₃¹',
                'expanded': f'τ = t_P × (U/π)⁶ × (1/p)³ × [1/(Kp²)]² × [(lnK)²/(K/lnK)]¹'
            },
            'top_quark': {
                'compact': 'τ = t_P × f₁⁷ f₄⁰ × q₁² q₃²',
                'expanded': f'τ = t_P × (U/π)⁷ × [1/(Kp²)]² × [(lnK)²/(K/lnK)]²'
            },
        }
        return formulas.get(particle_name, {'compact': '', 'expanded': ''})


def generate_complete_table():
    """Генерирует полную таблицу результатов"""

    theory = UnifiedTheoryTable()

    particles = [
        'W_boson', 'Z_boson', 'Higgs',
        'muon', 'tau', 'electron',
        'neutron', 'pion+', 'kaon+',
        'pion0', 'eta',
        'rho', 'Delta++', 'top_quark'
    ]

    print("=" * 150)
    print("ПОЛНАЯ ТАБЛИЦА СТРУКТУРНЫХ ФОРМУЛ ВРЕМЁН ЖИЗНИ")
    print("=" * 150)
    print("Параметры модели:")
    print(f"  K = {theory.K}")
    print(f"  p = {theory.p:.6f}")
    print(f"  N = {theory.N:.2e}")
    print(f"  t_P = {theory.t_P:.2e} с")
    print(f"  U = lnN/|ln(Kp)| = {theory.U:.2f}")
    print(f"  U/p = {theory.U / theory.p:.1f}")
    print("=" * 150)

    header = f"{'ЧАСТИЦА':<12} {'ТИП':<10} {'МАССА (кг)':<15} {'τ (с)':<15} {'СТРУКТУРНАЯ ФОРМУЛА (f-q)':<40} {'ДЕТАЛЬНАЯ ФОРМУЛА':<60}"
    print(header)
    print("-" * 150)

    for particle in particles:
        mass = theory.get_mass_kg(particle)
        tau = theory.get_tau_exp(particle)
        ptype = theory.get_particle_type(particle)
        formulas = theory.expand_formula(particle)

        # Форматируем массу
        if mass > 0:
            mass_str = f"{mass:.3e}"
        else:
            mass_str = "—"

        # Форматируем время жизни
        if tau == float('inf'):
            tau_str = "∞ (стабильная)"
        else:
            tau_str = f"{tau:.3e}"

        # Выводим строку
        compact = formulas['compact']
        expanded = formulas['expanded']

        # Перенос строки, если формула слишком длинная
        if len(expanded) > 60:
            parts = [expanded[i:i + 60] for i in range(0, len(expanded), 60)]
            expanded = "\n" + " " * 100 + parts[0]
            for part in parts[1:]:
                expanded += "\n" + " " * 100 + part

        print(f"{particle:<12} {ptype:<10} {mass_str:<15} {tau_str:<15} {compact:<40} {expanded}")

    print("=" * 150)

    # Вывод определений функций
    print("\n" + "=" * 150)
    print("ОПРЕДЕЛЕНИЯ СТРУКТУРНЫХ ФУНКЦИЙ:")
    print("=" * 150)
    definitions = [
        ("f₁ = U/π", f"= {theory.U}/π = {theory.f1:.1f}"),
        ("f₂ = lnK", f"= ln({theory.K}) = {theory.f2:.3f}"),
        ("f₃ = √(Kp)", f"= √({theory.K}×{theory.p}) = {theory.f3:.4f}"),
        ("f₄ = 1/p", f"= 1/{theory.p} = {theory.f4:.1f}"),
        ("f₅ = K/lnK", f"= {theory.K}/ln({theory.K}) = {theory.f5:.3f}"),
        ("f₆ = 1+p", f"= 1+{theory.p} = {theory.f6:.4f}"),
        ("", ""),
        ("g₁ = U/(πp)", f"= {theory.U}/(π×{theory.p}) = {theory.g1:.1f}"),
        ("g₂ = lnU/lnK", f"= ln({theory.U})/ln({theory.K}) = {theory.g2:.3f}"),
        ("", ""),
        ("q₁ = 1/(Kp²)", f"= 1/({theory.K}×{theory.p}²) = {theory.q1:.2e}"),
        ("q₃ = f₂²/f₅", f"= (lnK)²/(K/lnK) = {theory.q3:.2f}"),
        ("", ""),
        ("U = lnN/|ln(Kp)|", f"= ln({theory.N})/|ln({theory.K}×{theory.p})| = {theory.U:.2f}"),
        ("U/p", f"= {theory.U}/{theory.p} = {theory.U / theory.p:.1f}"),
    ]

    for def_name, def_value in definitions:
        if def_name:
            print(f"{def_name:<20} {def_value}")
        else:
            print()

    print("=" * 150)

    # Физические интерпретации
    print("\n" + "=" * 150)
    print("ФИЗИЧЕСКИЕ ИНТЕРПРЕТАЦИИ:")
    print("=" * 150)
    interpretations = [
        "1. ВСЕ СЛАБЫЕ РАСПАДЫ содержат ядро: f₁¹¹ f₄⁻³",
        "2. КАЛИБРОВОЧНЫЕ БОЗОНЫ: только f₁-f₆ (без g, q)",
        "3. ФЕРМИОНЫ: требуют g₁ (киральность) и g₂ (поколения)",
        "4. АДРОНЫ: требуют q₁ (QCD-масштаб) и q₃ (изоспин)",
        "5. ТИП РАСПАДА определяет степень (U/p):",
        "   • Слабые: (U/p)⁴-⁵",
        "   • Электромагнитные: (U/p)³",
        "   • Сильные: (U/p)⁰",
        "6. МАССА влияет на степень f₁ и знак f₄:",
        "   • Лёгкие частицы: большие степени f₁, f₄<0",
        "   • Тяжёлые частицы: меньшие степени f₁, f₄>0",
        "7. СТАБИЛЬНОСТЬ электрона обеспечивается предельными",
        "   значениями g₁→∞ или специальными условиями симметрии",
    ]

    for i, interp in enumerate(interpretations, 1):
        print(f"{interp}")

    print("=" * 150)

    # Проверка согласованности
    print("\n" + "=" * 150)
    print("ПРОВЕРКА СОГЛАСОВАННОСТИ МОДЕЛИ:")
    print("=" * 150)

    # Вычисляем некоторые ключевые отношения
    print("Ключевые безразмерные отношения:")
    print(f"  f₁/f₄ = {theory.f1 / theory.f4:.3f} (отношение масштабов)")
    print(f"  U/p = {theory.U / theory.p:.1f} (универсальный слабый фактор)")
    print(f"  g₁/f₁ = {theory.g1 / theory.f1:.1f} (фермионный фактор)")
    print(f"  q₁ × Kp² = {theory.q1 * theory.K * theory.p ** 2:.2f} (проверка определения q₁)")

    # Проверка ядра формулы
    print("\nПроверка ядра формулы для слабых распадов:")
    core_value = theory.f1 ** 11 * theory.f4 ** (-3) * theory.t_P
    print(f"  t_P × f₁¹¹ × f₄⁻³ = {core_value:.2e} с")
    print(f"  Отношение к τ_W: {core_value / theory.get_tau_exp('W_boson'):.3f}")
    print(f"  Отношение к τ_μ: {core_value / theory.get_tau_exp('muon'):.3e}")

    print("=" * 150)


if __name__ == "__main__":
    generate_complete_table()