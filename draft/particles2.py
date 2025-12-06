import math

# Параметры теории
K = 8.0
p = 5.270179e-02
N = 9.702e+122

# Базовые величины
lnK = math.log(K)
lnKp = math.log(K * p)
lnN = math.log(N)
U = lnN / abs(lnKp)

# Структурные функции
f1 = U / math.pi  # ~104.37
f2 = lnK  # ~2.079
f3 = math.sqrt(K * p)  # ~0.6493
f4 = 1 / p  # ~18.97
f5 = K / lnK  # ~3.847

# Базовый электрон
m_e = 12 * f3 * (U ** 4) * (N ** (-1 / 3))


class UnifiedParticleMasses:
    """Универсальная система масс элементарных частиц"""

    def __init__(self):
        self.m_e = m_e

    # Лептоны
    def electron(self):
        return self.m_e

    def muon(self):
        return self.m_e * 2 * f1

    def tau(self):
        return self.m_e * f1 * f4

    def neutrino(self, generation=1):
        """Массы нейтрино (очень малые)"""
        base = self.m_e * (p * f2) ** 4
        if generation == 1:
            return base
        elif generation == 2:
            return base * math.sqrt(f1)
        else:  # generation == 3
            return base * f1

    # Кварки
    def up(self):
        return self.m_e * (1 / f3) * (1 + p)

    def down(self):
        return self.up() * f2 * (1 + p)

    def strange(self):
        return self.m_e * f1

    def charm(self):
        return self.m_e * f1 * f5 * math.sqrt(p)

    def bottom(self):
        return self.m_e * (f1 ** 2) * (p / f2)

    def top(self):
        return self.bottom() * f5 * f4

    # Адроны
    def proton(self):
        return self.m_e * f1 * f2 * (1 + 1 / f4)

    def neutron(self):
        return self.proton() * (1 + p / f2)

    # Бозоны
    def W_boson(self):
        return self.m_e * (f1 ** 2) * f5

    def Z_boson(self):
        return self.W_boson() * (1 + p)

    def Higgs(self):
        return self.W_boson() * (math.pi / 2)


# Тестирование системы
print("=" * 80)
print("УНИВЕРСАЛЬНАЯ СИСТЕМА МАСС ЭЛЕМЕНТАРНЫХ ЧАСТИЦ")
print("=" * 80)

model = UnifiedParticleMasses()

# Экспериментальные значения (кг)
exp_values = {
    'electron': 9.1093837015e-31,
    'muon': 1.883531627e-28,
    'tau': 3.16754e-27,
    'up': 2.16e-30,
    'down': 4.67e-30,
    'strange': 93.4e-30,
    'charm': 1.27e-27,
    'bottom': 4.18e-27,
    'top': 3.08e-25,
    'proton': 1.67262192369e-27,
    'neutron': 1.67492749804e-27,
    'W': 1.433e-25,
    'Z': 1.626e-25,
    'Higgs': 2.246e-25
}

print(f"{'Частица':<12} {'Теория (кг)':<20} {'Эксперимент (кг)':<20} {'Отношение':<10} {'Ошибка (%)':<10}")
print("-" * 90)

particles = [
    ('electron', model.electron),
    ('muon', model.muon),
    ('tau', model.tau),
    ('up', model.up),
    ('down', model.down),
    ('strange', model.strange),
    ('charm', model.charm),
    ('bottom', model.bottom),
    ('top', model.top),
    ('proton', model.proton),
    ('neutron', model.neutron),
    ('W', model.W_boson),
    ('Z', model.Z_boson),
    ('Higgs', model.Higgs)
]

for name, func in particles:
    theory = func()
    exp = exp_values.get(name, 0)
    if exp > 0:
        ratio = theory / exp
        error = abs(theory - exp) / exp * 100
        print(f"{name:<12} {theory:<20.3e} {exp:<20.3e} {ratio:<10.3f} {error:<10.1f}")