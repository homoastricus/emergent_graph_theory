
import math
import numpy as np
from scipy.optimize import minimize
from itertools import product

# ===============================
# БАЗОВЫЕ ПАРАМЕТРЫ
# ===============================
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

# Классические массы (кг) — расширен для тяжёлых частиц, адронов и т.д.
m_classical = {
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
    'deuterium': 3.3435837724e-27,  # m_p + m_n - binding (~2.224 MeV/c²)
    'alpha': 6.644657230e-27,      # 4He = 2p + 2n - binding (~28.3 MeV/c²)
    'neutrino1': 1e-36,            # Оценка, очень малая
    'neutrino2': 1e-36,            # Оценка
    'neutrino3': 5e-35,            # Оценка
    'pion': 2.39e-28,              # Пион заряженный
    'kaon': 8.77e-28,              # Каон
    'eta': 9.77e-28,               # Эта-мезон
    'rho': 1.37e-27,               # Ро-мезон
    'W': 1.433e-25,                # W-бозон
    'Z': 1.626e-25,                # Z-бозон
    'Higgs': 2.246e-25,            # Хиггс-бозон
}

# ===============================
# ФУНКЦИЯ ДЛЯ БАЗОВОГО ЭЛЕКТРОНА С ВАРИАЦИЯМИ КОЭФФИЦИЕНТА
# ===============================
def calculate_m_e(coeff_type='fixed', coeff=12, t=None, n=None):
    """Расчёт m_e с вариациями коэффициента: fixed=12, 2*K^n или 2^(1/2)^t * K^n^t"""
    if coeff_type == 'fixed':
        return coeff * f3 * (U ** 4) * (N ** (-1 / 3))
    elif coeff_type == '2_K_n':
        if n is None:
            raise ValueError("n required for 2 * K^n")
        return 2 * (K ** n) * f3 * (U ** 4) * (N ** (-1 / 3))
    elif coeff_type == '2_half_t_K_n_t':
        if t is None or n is None:
            raise ValueError("t and n required for 2^(1/2)^t * K^n^t")
        return (2 ** (0.5 ** t)) * (K ** (n ** t)) * f3 * (U ** 4) * (N ** (-1 / 3))
    else:
        raise ValueError("Неверный тип коэффициента")

# ===============================
# КЛАСС ДЛЯ РАСЧЁТА МАСС (С АНАЛИТИЧЕСКИМИ C_i)
# ===============================
class UnifiedParticleMasses:
    """Универсальная система масс элементарных частиц и адронов"""
    def __init__(self, coeff_type='fixed', coeff=12, t=None, n=None):
        self.m_e = calculate_m_e(coeff_type, coeff, t, n)
        # Аналитические C_i из лучших совпадений (с ошибкой <1%, упрощённые)
        self.C_muon = 2  # От 2 * K^0.00
        self.C_tau = (2 ** (0.5 ** 3.9)) * (K ** (0.7 ** 3.9))  # 1.7574
        self.C_up = (2 ** (0.5 ** 2.8)) * (K ** (0.7 ** 2.8))  # 2.3762
        self.C_down = 2 * K ** 0.1  # 2.4623
        self.C_charm = (2 ** (0.5 ** 2.7)) * (K ** (0.8 ** 2.7))  # 3.4731
        self.C_bottom = f3 ** 2  # 0.4216
        self.C_top = f1 * f2 ** -1 * f3 ** -2 * f5 ** -2  # 8.0432
        self.C_proton = f1 * f3 ** -1 * f4 ** -1 * f5 ** -1  # 2.2019
        self.C_neutron = (2 ** (0.5 ** 1.2)) * (K ** (0.3 ** 1.2))  # 2.2079
        self.C_deuterium = (2 ** (0.5 ** 3.6)) * (K ** (0.9 ** 3.6))  # 4.3939
        self.C_alpha = (2 ** (0.5 ** 3.0)) * (K ** (1.0 ** 3.0))  # 8.7241
        self.C_pion = f1 ** -1 * f2 ** 2 * f3 ** -1 * f4 ** 1 * f5 ** 0  # 1.2107
        self.C_kaon = f1 ** 1 * f2 ** 1 * f3 ** 2 * f4 ** -2 * f5 ** 1  # 0.9778
        self.C_eta = f1 ** 0 * f2 ** 1 * f3 ** 0 * f4 ** 1 * f5 ** -2  # 2.6658
        self.C_rho = f1 ** 1 * f2 ** 2 * f3 ** 2 * f4 ** -1 * f5 ** 0  # 10.0280
        self.C_W = f1 * f2 * f3 ** 2 * f4 ** -2 * f5 ** 2  # 3.7617
        self.C_Z = f1 ** 2 * f2 * f4 ** -2 * f5 ** -2  # 4.2508
        self.C_Higgs = (2 ** (0.5 ** 2.3)) * (K ** (0.9 ** 2.3))  # 5.8867

    # Лептоны
    def electron(self):
        return self.m_e

    def muon(self):
        return self.m_e * self.C_muon * f1

    def tau(self):
        return self.m_e * self.C_tau * f1 * f4

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
        return self.m_e * self.C_up

    def down(self):
        return self.m_e * self.C_down * f2

    def strange(self):
        return self.m_e * f1

    def charm(self):
        return self.m_e * self.C_charm * f1 * f5

    def bottom(self):
        return self.m_e * self.C_bottom * f1 ** 2

    def top(self):
        return self.m_e * self.C_top * f1 ** 2 * f5

    # Адроны
    def proton(self):
        return self.m_e * self.C_proton * U * K / math.pi

    def neutron(self):
        return self.m_e * self.C_neutron * U * K / math.pi

    def deuterium(self):
        # Дейтерий = протон + нейтрон - энергия связи
        binding_factor = 1 - p / f5  # ~0.986
        return (self.proton() + self.neutron()) * binding_factor

    def alpha(self):
        # Альфа-частица = 2 протона + 2 нейтрона - сильная binding
        binding_factor = 1 - 4 * p / f5  # ~0.945
        return 2 * (self.proton() + self.neutron()) * binding_factor

    # Дополнительные частицы (мезоны, etc.)
    def pion(self):
        return self.m_e * self.C_pion * f1 * f2

    def kaon(self):
        return self.m_e * self.C_kaon * f1 * f4 / 2

    def eta(self):
        return self.m_e * self.C_eta * f1 * f5

    def rho(self):
        return self.m_e * self.C_rho * f1 * f2 * f3

    # Бозоны
    def W_boson(self):
        return self.m_e * self.C_W * (f1 ** 2) * f5

    def Z_boson(self):
        return self.m_e * self.C_Z * (f1 ** 2) * f5

    def Higgs(self):
        return self.m_e * self.C_Higgs * (f1 ** 2) * f5

# ===============================
# ОПТИМИЗАЦИЯ КОЭФФИЦИЕНТОВ
# ===============================
def objective_function(coeffs):
    """Минимизируем сумму квадратов относительных ошибок"""
    # coeffs: [C_muon, C_tau, C_up, C_down, C_charm, C_bottom, C_top, C_proton, C_neutron, C_deuterium, C_alpha, C_pion, C_kaon, C_eta, C_rho, C_W, C_Z, C_Higgs]
    m_e = calculate_m_e()
    predictions = {
        'muon': m_e * coeffs[0] * f1,
        'tau': m_e * coeffs[1] * f1 * f4,
        'up': m_e * coeffs[2],
        'down': m_e * coeffs[3] * f2,
        'charm': m_e * coeffs[4] * f1 * f5,
        'bottom': m_e * coeffs[5] * f1 ** 2,
        'top': m_e * coeffs[6] * f1 ** 2 * f5,
        'proton': m_e * coeffs[7] * U * K / math.pi,
        'neutron': m_e * coeffs[8] * U * K / math.pi,
        'deuterium': m_e * coeffs[9] * U * K / math.pi,
        'alpha': m_e * coeffs[10] * U * K / math.pi,
        'pion': m_e * coeffs[11] * f1 * f2,
        'kaon': m_e * coeffs[12] * f1 * f4 / 2,
        'eta': m_e * coeffs[13] * f1 * f5,
        'rho': m_e * coeffs[14] * f1 * f2 * f3,
        'W': m_e * coeffs[15] * (f1 ** 2) * f5,
        'Z': m_e * coeffs[16] * (f1 ** 2) * f5,
        'Higgs': m_e * coeffs[17] * (f1 ** 2) * f5,
    }
    total_error = 0
    for particle, pred in predictions.items():
        classic = m_classical.get(particle, 0)
        if classic > 0:
            error = (pred - classic) / classic
            total_error += error ** 2
    return total_error

# Начальные приближения (расширены для новых частиц)
initial_guess = [2.0, 1.0, 0.5, 0.25, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

# Границы для коэффициентов
bounds = [(0.1, 10)] * len(initial_guess)

# Оптимизация
result = minimize(objective_function, initial_guess, bounds=bounds, method='L-BFGS-B')

optimized_coeffs = result.x

# ===============================
# ПОИСК АНАЛИТИЧЕСКИХ ВЫРАЖЕНИЙ ДЛЯ КОЭФФИЦИЕНТОВ
# ===============================
def search_analytic_expressions(optimized_coeffs):
    """Brute-force поиск аналитических выражений для коэффициентов, включая вариации с 2, K^n, 2^(1/2)^t * K^n^t"""
    print("\n" + "=" * 80)
    print("ПОИСК АНАЛИТИЧЕСКИХ ВЫРАЖЕНИЙ ДЛЯ КОЭФФИЦИЕНТОВ")
    print("=" * 80)

    coeff_names = ['C_muon', 'C_tau', 'C_up', 'C_down', 'C_charm', 'C_bottom', 'C_top', 'C_proton', 'C_neutron', 'C_deuterium', 'C_alpha', 'C_pion', 'C_kaon', 'C_eta', 'C_rho', 'C_W', 'C_Z', 'C_Higgs']

    for idx, coeff in enumerate(optimized_coeffs):
        print(f"\n{coeff_names[idx]} = {coeff:.4f}")

        best_matches = []

        # Вариации с 2 * K^n
        for n in np.linspace(-2, 2, 41):  # Шаги 0.1
            value = 2 * (K ** n)
            error = abs(value - coeff) / coeff * 100
            if error < 5:
                best_matches.append((f"2 * K^{n:.2f}", value, error))

        # Вариации с 2^(1/2)^t * K^n^t
        for t in np.linspace(0, 4, 41):  # Шаги 0.1
            for n in np.linspace(0, 2, 21):
                value = (2 ** (0.5 ** t)) * (K ** (n ** t))
                error = abs(value - coeff) / coeff * 100
                if error < 5:
                    best_matches.append((f"2^(1/2)^ {t:.2f} * K^{n:.2f}^ {t:.2f}", value, error))

        # Другие вариации (с p, lnK, f_i, e, π и их комбинациями)
        expressions = [
            lambda: 1 / f2, lambda: f2, lambda: 1 / f4, lambda: f4 / 2, lambda: f5 / 2,
            lambda: lnK / 2, lambda: math.pi / lnK, lambda: math.e / f2, lambda: math.sqrt(2) * lnK,
            lambda: 2 / math.sqrt(K), lambda: math.sqrt(2) ** lnK, lambda: math.e ** (-p),
            lambda: 2 * math.pi / f5, lambda: f1 / f4, lambda: f3 * f2, lambda: 1 / (p * lnK)
        ]
        for expr_func in expressions:
            value = expr_func()
            error = abs(value - coeff) / coeff * 100
            if error < 5:
                best_matches.append((f"Выражение: {value:.4f}", value, error))

        # Комбинации f_i
        for combo in product(range(-2, 3), repeat=5):  # Степени для f1..f5
            value = (f1 ** combo[0]) * (f2 ** combo[1]) * (f3 ** combo[2]) * (f4 ** combo[3]) * (f5 ** combo[4])
            if value > 0:
                error = abs(value - coeff) / coeff * 100
                if error < 5:
                    best_matches.append((f"f1^{combo[0]} * f2^{combo[1]} * f3^{combo[2]} * f4^{combo[3]} * f5^{combo[4]}", value, error))

        # Сортируем по ошибке
        best_matches.sort(key=lambda x: x[2])

        if best_matches:
            print("Лучшие совпадения:")
            for pattern, value, error in best_matches[:10]:  # Топ-10
                print(f" {pattern:50} = {value:.4f} (ошибка: {error:.1f}%)")
        else:
            print("Нет совпадений с ошибкой <5%.")

# ===============================
# ТЕСТИРОВАНИЕ СИСТЕМЫ
# ===============================
model = UnifiedParticleMasses(coeff_type='fixed', coeff=12)  # Можно менять coeff_type на '2_K_n' или '2_half_t_K_n_t' с t,n

print("=" * 80)
print("УНИВЕРСАЛЬНАЯ СИСТЕМА МАСС ЭЛЕМЕНТАРНЫХ ЧАСТИЦ")
print("=" * 80)

print(f"{'Частица':<12} {'Теория (кг)':<20} {'Эксперимент (кг)':<20} {'Отношение':<10} {'Ошибка (%)':<10}")
print("-" * 90)

particles = [
    ('electron', model.electron()),
    ('muon', model.muon()),
    ('tau', model.tau()),
    ('up', model.up()),
    ('down', model.down()),
    ('strange', model.strange()),
    ('charm', model.charm()),
    ('bottom', model.bottom()),
    ('top', model.top()),
    ('proton', model.proton()),
    ('neutron', model.neutron()),
    ('deuterium', model.deuterium()),
    ('alpha', model.alpha()),
    ('pion', model.pion()),
    ('kaon', model.kaon()),
    ('eta', model.eta()),
    ('rho', model.rho()),
    ('W', model.W_boson()),
    ('Z', model.Z_boson()),
    ('Higgs', model.Higgs()),
    ('neutrino1', model.neutrino(1)),
    ('neutrino2', model.neutrino(2)),
    ('neutrino3', model.neutrino(3)),
]

for name, theory in particles:
    exp = m_classical.get(name, 0)
    if exp > 0:
        ratio = theory / exp
        error = abs(theory - exp) / exp * 100
        print(f"{name:<12} { theory:<20.3e} {exp:<20.3e} {ratio:<10.3f} {error:<10.1f}")

# ===============================
# ОПТИМИЗАЦИЯ КОЭФФИЦИЕНТОВ
# ===============================
print("\n" + "=" * 80)
print("ОПТИМИЗАЦИЯ КОЭФФИЦИЕНТОВ")
print("=" * 80)

result = minimize(objective_function, initial_guess, bounds=bounds, method='L-BFGS-B')
optimized_coeffs = result.x

print(f"Оптимизированные коэффициенты:")
for i, coeff in enumerate(optimized_coeffs):
    print(f"C_{i+1} = {coeff:.4f}")

# ===============================
# ПОИСК АНАЛИТИЧЕСКИХ ВЫРАЖЕНИЙ
# ===============================
search_analytic_expressions(optimized_coeffs)

# ===============================
# ФИНАЛЬНЫЙ ВЫВОД АНАЛИТИЧЕСКИХ ФОРМУЛ
# ===============================
print("\n" + "=" * 100)
print("ФИНАЛЬНЫЕ АНАЛИТИЧЕСКИЕ ФОРМУЛЫ МАСС ЧАСТИЦ")
print("=" * 100)

formulas = {
    "electron":  "mₑ = 12·√(Kp)·U⁴·N⁻¹ᐟ³",
    "muon":      "mμ = mₑ · 2·(U/π)",
    "tau":       "mτ = mₑ · (2^(½)^3.9 · K^(0.7)^3.9) · (U/π)·(1/p)",
    "up":        "mᵤ = mₑ · (2^(½)^2.8 · K^(0.7)^2.8)",
    "down":      "m_d = mₑ · (2·K^0.1) · lnK",
    "strange":   "m_s = mₑ · (U/π)",
    "charm":     "m_c = mₑ · (2^(½)^2.7 · K^(0.8)^2.7) · (U/π)·(K/lnK)",
    "bottom":    "m_b = mₑ · f₃² · (U/π)²",
    "top":       "m_t = mₑ · (f₁·f₂⁻¹·f₃⁻²·f₅⁻²) · (U/π)²·(K/lnK)",
    "proton":    "m_p = mₑ · (f₁·f₃⁻¹·f₄⁻¹·f₅⁻¹) · (U·K/π)",
    "neutron":   "mₙ = mₑ · (2^(½)^1.2 · K^(0.3)^1.2) · (U·K/π)",
    "deuterium": "m_D = (m_p + mₙ)·(1 − p/f₅)",
    "alpha":     "m_α = 2·(m_p + mₙ)·(1 − 4p/f₅)",
    "pion":      "m_π = mₑ · (f₁⁻¹·f₂²·f₃⁻¹·f₄·f₅⁰) · (U/π)·lnK",
    "kaon":      "m_K = mₑ · (f₁·f₂·f₃²·f₄⁻²·f₅) · (U/π)·(1/p)",
    "eta":       "m_η = mₑ · (f₂·f₄·f₅⁻²) · (U/π)·(K/lnK)",
    "rho":       "m_ρ = mₑ · (f₁·f₂²·f₃²·f₄⁻¹) · (U/π)·lnK",
    "W":         "m_W = mₑ · (f₁·f₂·f₃²·f₄⁻²·f₅²) · (U/π)²·(K/lnK)",
    "Z":         "m_Z = mₑ · (f₁²·f₂·f₄⁻²·f₅⁻²) · (U/π)²·(K/lnK)",
    "Higgs":     "m_H = mₑ · (2^(½)^2.3 · K^(0.9)^2.3) · (U/π)²·(K/lnK)",
    "ν₁":        "m_ν₁ = mₑ·(p·lnK)⁴",
    "ν₂":        "m_ν₂ = mₑ·(p·lnK)⁴·√(U/π)",
    "ν₃":        "m_ν₃ = mₑ·(p·lnK)⁴·(U/π)",
}

for k,v in formulas.items():
    print(f"{k:<10} : {v}")

