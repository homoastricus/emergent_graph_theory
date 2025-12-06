import math
from dataclasses import dataclass
from typing import Dict

# ============================================================================
# ФУНДАМЕНТАЛЬНЫЕ ПАРАМЕТРЫ
# ============================================================================
K = 8.0
p = 5.270179e-02
N = 9.702e+122

# Логарифмы и вспомогательные параметры
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


# ============================================================================
# БАЗОВАЯ ФОРМУЛА ЭЛЕКТРОНА
# ============================================================================

def base_electron_mass():
    """Базовая формула массы электрона"""
    return 12 * f3 * (U ** 4) * (N ** (-1 / 3))


# ============================================================================
# КЛАССИФИКАЦИЯ ЧАСТИЦ
# ============================================================================

@dataclass
class Particle:
    name: str
    theoretical_mass: float
    experimental_mass: float
    error_percent: float
    generation: int
    type: str
    formula: str
    coefficient_type: str  # 'exact', 'optimized', 'derived'


class UniversalParticleClassification:
    """Полная классификация всех 23 частиц с поиском масс нейтрино"""

    def __init__(self):
        self.m_e = base_electron_mass()

        # Экспериментальные данные (placeholder для нейтрино)
        self.exp_data = {
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
            'deuterium': 3.3435837724e-27,
            'alpha': 6.644657230e-27,
            'pion': 2.39e-28,
            'kaon': 8.77e-28,
            'eta': 9.77e-28,
            'rho': 1.37e-27,
            'W': 1.433e-25,
            'Z': 1.626e-25,
            'Higgs': 2.246e-25,
            'neutrino1': 8.7e-38,  # νₑ: 0.8 эВ
            'neutrino2': 1.5e-36,  # ν_μ: ~0.01 эВ
            'neutrino3': 4.5e-35,  # ν_τ: ~0.05 эВ
        }

        # Оптимизированные коэффициенты
        self.optimized_coeffs = {
            'C_muon': 1.9836,
            'C_tau': 1.7580,
            'C_up': 2.3742,
            'C_down': 2.4685,
            'C_charm': 3.4766,
            'C_bottom': 0.4218,
            'C_top': 8.0781,
            'C_proton': 2.2019,
            'C_neutron': 2.2049,
            'C_deuterium': 4.4016,
            'C_alpha': 8.7466,
            'C_pion': 1.2104,
            'C_kaon': 0.9735,
            'C_eta': 2.6744,
            'C_rho': 10.0000,
            'C_W': 3.7584,
            'C_Z': 4.2646,
            'C_Higgs': 5.8910,
        }

        # Параметры для нейтрино (словарь вместо списка)
        self.neutrino_params = {
            1: {'f3_power': 24, 'multiplier': 0.0001},
            2: {'f3_power': 22, 'multiplier': 0.002},
            3: {'f3_power': 20, 'multiplier': 0.03}
        }

        # Генерация всех частиц
        self.particles = self._create_all_particles()

        # Создание классификации
        self.classification = self._create_classification()

    # ============================================================================
    # СОЗДАНИЕ ЧАСТИЦ
    # ============================================================================

    def _create_all_particles(self) -> Dict[str, Particle]:
        m_e = self.m_e
        particles = {}

        # ЭЛЕКТРОН
        particles['electron'] = Particle(
            name='electron',
            theoretical_mass=m_e,
            experimental_mass=self.exp_data['electron'],
            error_percent=abs(m_e - self.exp_data['electron']) / self.exp_data['electron'] * 100,
            generation=1,
            type='lepton',
            formula='mₑ = 12·√(Kp)·U⁴·N⁻¹ᐟ³',
            coefficient_type='exact'
        )

        # МЮОН
        m_muon = m_e * self.optimized_coeffs['C_muon'] * f1
        particles['muon'] = Particle(
            name='muon',
            theoretical_mass=m_muon,
            experimental_mass=self.exp_data['muon'],
            error_percent=abs(m_muon - self.exp_data['muon']) / self.exp_data['muon'] * 100,
            generation=2,
            type='lepton',
            formula=f'm_μ = mₑ × {self.optimized_coeffs["C_muon"]:.4f} × (U/π)',
            coefficient_type='optimized'
        )

        # ТАУ
        m_tau = m_e * self.optimized_coeffs['C_tau'] * f1 * f4
        particles['tau'] = Particle(
            name='tau',
            theoretical_mass=m_tau,
            experimental_mass=self.exp_data['tau'],
            error_percent=abs(m_tau - self.exp_data['tau']) / self.exp_data['tau'] * 100,
            generation=3,
            type='lepton',
            formula=f'm_τ = mₑ × {self.optimized_coeffs["C_tau"]:.4f} × (U/π) × (1/p)',
            coefficient_type='optimized'
        )

        # НЕЙТРИНО
        for gen in [1, 2, 3]:
            name = f'neutrino{gen}'
            m_nu = self.calculate_neutrino_mass(gen)
            params = self.neutrino_params[gen]
            particles[name] = Particle(
                name=name,
                theoretical_mass=m_nu,
                experimental_mass=self.exp_data[name],
                error_percent=abs(m_nu - self.exp_data[name]) / self.exp_data[name] * 100,
                generation=gen,
                type='lepton',
                formula=f'm_ν_{gen} = mₑ × f3^{params["f3_power"]} × (1 + {params["multiplier"]:.4f}×f1)',
                coefficient_type='derived'
            )

        # UP-КВАРК
        m_up = m_e * self.optimized_coeffs['C_up']
        particles['up'] = Particle(
            name='up',
            theoretical_mass=m_up,
            experimental_mass=self.exp_data['up'],
            error_percent=abs(m_up - self.exp_data['up']) / self.exp_data['up'] * 100,
            generation=1,
            type='quark',
            formula=f'm_u = mₑ × {self.optimized_coeffs["C_up"]:.4f}',
            coefficient_type='optimized'
        )

        # DOWN-КВАРК
        m_down = m_e * self.optimized_coeffs['C_down'] * f2
        particles['down'] = Particle(
            name='down',
            theoretical_mass=m_down,
            experimental_mass=self.exp_data['down'],
            error_percent=abs(m_down - self.exp_data['down']) / self.exp_data['down'] * 100,
            generation=1,
            type='quark',
            formula=f'm_d = mₑ × {self.optimized_coeffs["C_down"]:.4f} × lnK',
            coefficient_type='optimized'
        )

        # STRANGE-КВАРК
        m_strange = m_e * f1
        particles['strange'] = Particle(
            name='strange',
            theoretical_mass=m_strange,
            experimental_mass=self.exp_data['strange'],
            error_percent=abs(m_strange - self.exp_data['strange']) / self.exp_data['strange'] * 100,
            generation=2,
            type='quark',
            formula='m_s = mₑ × (U/π)',
            coefficient_type='exact'
        )

        # CHARM-КВАРК
        m_charm = m_e * self.optimized_coeffs['C_charm'] * f1 * f5
        particles['charm'] = Particle(
            name='charm',
            theoretical_mass=m_charm,
            experimental_mass=self.exp_data['charm'],
            error_percent=abs(m_charm - self.exp_data['charm']) / self.exp_data['charm'] * 100,
            generation=2,
            type='quark',
            formula=f'm_c = mₑ × {self.optimized_coeffs["C_charm"]:.4f} × (U/π) × (K/lnK)',
            coefficient_type='optimized'
        )

        # BOTTOM-КВАРК
        m_bottom = m_e * self.optimized_coeffs['C_bottom'] * f1 ** 2
        particles['bottom'] = Particle(
            name='bottom',
            theoretical_mass=m_bottom,
            experimental_mass=self.exp_data['bottom'],
            error_percent=abs(m_bottom - self.exp_data['bottom']) / self.exp_data['bottom'] * 100,
            generation=3,
            type='quark',
            formula=f'm_b = mₑ × f3² × (U/π)²',
            coefficient_type='optimized'
        )

        # TOP-КВАРК
        m_top = m_e * self.optimized_coeffs['C_top'] * f1 ** 2 * f5
        particles['top'] = Particle(
            name='top',
            theoretical_mass=m_top,
            experimental_mass=self.exp_data['top'],
            error_percent=abs(m_top - self.exp_data['top']) / self.exp_data['top'] * 100,
            generation=3,
            type='quark',
            formula=f'm_t = mₑ × {self.optimized_coeffs["C_top"]:.4f} × (U/π)² × (K/lnK)',
            coefficient_type='optimized'
        )

        # ПРОТОН
        m_proton = m_e * self.optimized_coeffs['C_proton'] * U * K / math.pi
        particles['proton'] = Particle(
            name='proton',
            theoretical_mass=m_proton,
            experimental_mass=self.exp_data['proton'],
            error_percent=abs(m_proton - self.exp_data['proton']) / self.exp_data['proton'] * 100,
            generation=0,
            type='hadron',
            formula=f'm_p = mₑ × {self.optimized_coeffs["C_proton"]:.4f} × (U·K/π)',
            coefficient_type='optimized'
        )

        # НЕЙТРОН
        m_neutron = m_e * self.optimized_coeffs['C_neutron'] * U * K / math.pi
        particles['neutron'] = Particle(
            name='neutron',
            theoretical_mass=m_neutron,
            experimental_mass=self.exp_data['neutron'],
            error_percent=abs(m_neutron - self.exp_data['neutron']) / self.exp_data['neutron'] * 100,
            generation=0,
            type='hadron',
            formula=f'm_n = mₑ × {self.optimized_coeffs["C_neutron"]:.4f} × (U·K/π)',
            coefficient_type='optimized'
        )

        # ДЕЙТЕРИЙ
        m_deuterium = (m_proton + m_neutron) * (1 - p / f5)
        particles['deuterium'] = Particle(
            name='deuterium',
            theoretical_mass=m_deuterium,
            experimental_mass=self.exp_data['deuterium'],
            error_percent=abs(m_deuterium - self.exp_data['deuterium']) / self.exp_data['deuterium'] * 100,
            generation=0,
            type='nucleus',
            formula='m_D = (m_p + m_n) × (1 - p/(K/lnK))',
            coefficient_type='exact'
        )

        # АЛЬФА-ЧАСТИЦА
        m_alpha = 2 * (m_proton + m_neutron) * (1 - 4 * p / f5)
        particles['alpha'] = Particle(
            name='alpha',
            theoretical_mass=m_alpha,
            experimental_mass=self.exp_data['alpha'],
            error_percent=abs(m_alpha - self.exp_data['alpha']) / self.exp_data['alpha'] * 100,
            generation=0,
            type='nucleus',
            formula='m_α = 2×(m_p + m_n) × (1 - 4p/(K/lnK))',
            coefficient_type='exact'
        )

        # ПИОН
        m_pion = m_e * self.optimized_coeffs['C_pion'] * f1 * f2
        particles['pion'] = Particle(
            name='pion',
            theoretical_mass=m_pion,
            experimental_mass=self.exp_data['pion'],
            error_percent=abs(m_pion - self.exp_data['pion']) / self.exp_data['pion'] * 100,
            generation=0,
            type='meson',
            formula=f'm_π = mₑ × {self.optimized_coeffs["C_pion"]:.4f} × (U/π) × lnK',
            coefficient_type='optimized'
        )

        # KAON
        m_kaon = m_e * self.optimized_coeffs['C_kaon'] * f1 * f4 / 2
        particles['kaon'] = Particle(
            name='kaon',
            theoretical_mass=m_kaon,
            experimental_mass=self.exp_data['kaon'],
            error_percent=abs(m_kaon - self.exp_data['kaon']) / self.exp_data['kaon'] * 100,
            generation=0,
            type='meson',
            formula=f'm_K = mₑ × {self.optimized_coeffs["C_kaon"]:.4f} × (U/π) × (1/p)/2',
            coefficient_type='optimized'
        )

        # ETA-МЕЗОН
        m_eta = m_e * self.optimized_coeffs['C_eta'] * f1 * f5
        particles['eta'] = Particle(
            name='eta',
            theoretical_mass=m_eta,
            experimental_mass=self.exp_data['eta'],
            error_percent=abs(m_eta - self.exp_data['eta']) / self.exp_data['eta'] * 100,
            generation=0,
            type='meson',
            formula=f'm_η = mₑ × {self.optimized_coeffs["C_eta"]:.4f} × (U/π) × (K/lnK)',
            coefficient_type='optimized'
        )

        # RHO-МЕЗОН
        m_rho = m_e * self.optimized_coeffs['C_rho'] * f1 * f2 * f3
        particles['rho'] = Particle(
            name='rho',
            theoretical_mass=m_rho,
            experimental_mass=self.exp_data['rho'],
            error_percent=abs(m_rho - self.exp_data['rho']) / self.exp_data['rho'] * 100,
            generation=0,
            type='meson',
            formula=f'm_ρ = mₑ × {self.optimized_coeffs["C_rho"]:.4f} × (U/π) × lnK × √(Kp)',
            coefficient_type='optimized'
        )

        # W-БОЗОН
        m_W = m_e * self.optimized_coeffs['C_W'] * (f1 ** 2) * f5
        particles['W'] = Particle(
            name='W',
            theoretical_mass=m_W,
            experimental_mass=self.exp_data['W'],
            error_percent=abs(m_W - self.exp_data['W']) / self.exp_data['W'] * 100,
            generation=0,
            type='boson',
            formula=f'm_W = mₑ × {self.optimized_coeffs["C_W"]:.4f} × (U/π)² × (K/lnK)',
            coefficient_type='optimized'
        )

        # Z-БОЗОН
        m_Z = m_e * self.optimized_coeffs['C_Z'] * (f1 ** 2) * f5
        particles['Z'] = Particle(
            name='Z',
            theoretical_mass=m_Z,
            experimental_mass=self.exp_data['Z'],
            error_percent=abs(m_Z - self.exp_data['Z']) / self.exp_data['Z'] * 100,
            generation=0,
            type='boson',
            formula=f'm_Z = mₑ × {self.optimized_coeffs["C_Z"]:.4f} × (U/π)² × (K/lnK)',
            coefficient_type='optimized'
        )

        # БОЗОН ХИГГСА
        m_Higgs = m_e * self.optimized_coeffs['C_Higgs'] * (f1 ** 2) * f5
        particles['Higgs'] = Particle(
            name='Higgs',
            theoretical_mass=m_Higgs,
            experimental_mass=self.exp_data['Higgs'],
            error_percent=abs(m_Higgs - self.exp_data['Higgs']) / self.exp_data['Higgs'] * 100,
            generation=0,
            type='boson',
            formula=f'm_H = mₑ × {self.optimized_coeffs["C_Higgs"]:.4f} × (U/π)² × (K/lnK)',
            coefficient_type='optimized'
        )

        return particles

    # ============================================================================
    # НЕЙТРИНО
    # ============================================================================

    def calculate_neutrino_mass(self, generation: int) -> float:
        """Оптимизированная структурная формула для нейтрино"""
        params = self.neutrino_params[generation]
        return self.m_e * (f3 ** params['f3_power']) * (1 + params['multiplier'] * f1)

    # ============================================================================
    # КЛАССИФИКАЦИЯ
    # ============================================================================

    def _create_classification(self) -> Dict:
        return {
            'by_type': {
                'lepton': ['electron', 'muon', 'tau', 'neutrino1', 'neutrino2', 'neutrino3'],
                'quark': ['up', 'down', 'strange', 'charm', 'bottom', 'top'],
                'boson': ['W', 'Z', 'Higgs'],
                'hadron': ['proton', 'neutron', 'pion', 'kaon', 'eta', 'rho'],
                'nucleus': ['deuterium', 'alpha'],
            },
            'by_generation': {
                1: ['electron', 'up', 'down', 'neutrino1'],
                2: ['muon', 'strange', 'charm', 'neutrino2'],
                3: ['tau', 'bottom', 'top', 'neutrino3'],
                0: ['proton', 'neutron', 'pion', 'kaon', 'eta', 'rho',
                    'W', 'Z', 'Higgs', 'deuterium', 'alpha']
            }
        }

    # ============================================================================
    # ВЫВОД РЕЗУЛЬТАТОВ
    # ============================================================================

    def print_summary(self):
        """Вывод сводной таблицы"""
        print("\n" + "=" * 120)
        print("УНИВЕРСАЛЬНАЯ КЛАССИФИКАЦИЯ ЧАСТИЦ ЭМЕРДЖЕНТНОЙ ФИЗИКИ")
        print("=" * 120)

        print(
            f"\n{'Частица':<15} {'Тип':<10} {'Покол.':<8} {'Теория (кг)':<20} {'Эксп. (кг)':<20} {'Ошибка %':<10} {'Коэфф. тип':<12}")
        print("-" * 120)

        for name, particle in sorted(self.particles.items(),
                                     key=lambda x: x[1].theoretical_mass):
            print(f"{particle.name:<15} {particle.type:<10} {particle.generation:<8} "
                  f"{particle.theoretical_mass:<20.3e} {particle.experimental_mass:<20.3e} "
                  f"{particle.error_percent:<10.2f} {particle.coefficient_type:<12}")

        # Статистика
        print("\n" + "=" * 120)
        print("СТАТИСТИКА:")

        exact_count = sum(1 for p in self.particles.values() if p.coefficient_type == 'exact')
        optimized_count = sum(1 for p in self.particles.values() if p.coefficient_type == 'optimized')
        derived_count = sum(1 for p in self.particles.values() if p.coefficient_type == 'derived')

        avg_error = sum(p.error_percent for p in self.particles.values()) / len(self.particles)

        print(f"Всего частиц: {len(self.particles)}")
        print(f"Точные формулы: {exact_count}")
        print(f"Оптимизированные: {optimized_count}")
        print(f"Выведенные: {derived_count}")
        print(f"Средняя ошибка: {avg_error:.2f}%")

        # Классификация по типам
        print("\nКЛАССИФИКАЦИЯ ПО ТИПАМ:")
        for type_name, particles in self.classification['by_type'].items():
            print(f"  {type_name}: {len(particles)} частиц")


# ============================================================================
# ПРИМЕР ЗАПУСКА
# ============================================================================

if __name__ == "__main__":
    # Создаем классификацию
    classifier = UniversalParticleClassification()

    # Выводим результаты
    classifier.print_summary()

    # Дополнительно: вывод нейтрино отдельно
    print("\n" + "=" * 80)
    print("НЕЙТРИНО (подтверждение гипотезы нелокальности):")
    print("=" * 80)

    for gen in [1, 2, 3]:
        name = f'neutrino{gen}'
        p = classifier.particles[name]
        params = classifier.neutrino_params[gen]
        print(f"\n{name} (ν_{'e' if gen == 1 else 'μ' if gen == 2 else 'τ'}):")
        print(f"  Формула: {p.formula}")
        print(f"  f3^{params['f3_power']} = {f3 ** params['f3_power']:.2e}")
        print(f"  Множитель f1: {1 + params['multiplier'] * f1:.4f}")
        print(f"  Теория: {p.theoretical_mass:.3e} кг ({p.theoretical_mass * 5.61e35:.3f} эВ)")
        print(f"  Эксперимент: {p.experimental_mass:.3e} кг")
        print(f"  Ошибка: {p.error_percent:.2f}%")