
import math

K = 8.00  #
p = 5.270179e-02  #
N = 9.702e+122  #
lnK = math.log(K)
U = math.log(N) / abs(math.log(K * p))
neitron_part = 1.67262e-27
proton_part = 1.67493e-27
# Структурные функции
# фрактальный масштаб
f1 = U / math.pi  # U/π

# энтропия узла
f2 = lnK  # lnK

# (локальная скорость, локальная частота)
f3 = math.sqrt(K * p)  # √(Kp)

# нелокальность
f4 = 1 / p  # 1/p

# регулярность (структурная симметрия)
f5 = K / lnK  # K/lnK

f6 = (K + p * K) / K
def nuclear_binding_correction(A, Z):
    """Semi-empirical binding в терминах структурных функций графа."""
    a_volume = 15.5 * f3**2
    a_surface = 16.8 * (1 / f5)**0.7
    a_coulomb = 0.717 * f4**(-0.5) * p**1.5
    a_asymmetry = 23.5 / f1
    a_pairing = 12.0 * (-1 if A%2==1 else (1 if Z%2==0 and (A-Z)%2==0 else -1 if Z%2==1 and (A-Z)%2==1 else 0)) / math.sqrt(A)

    B = a_volume * A - a_surface * A**(2/3) - a_coulomb * Z*(Z-1) / A**(1/3) - a_asymmetry * (A - 2*Z)**2 / A + a_pairing

    # B в МэВ, B/A — энергия связи на нуклон в МэВ
    binding_per_nucleon_MeV = B / A
    # Переводим в атомные единицы массы (1 а.е.м. = 931.494 МэВ)
    mass_defect_fraction = binding_per_nucleon_MeV / 931.494

    # Коэффициент коррекции массы нуклона
    return 1 - mass_defect_fraction

# Пример использования:
Z = 146
A = 238
PR = 92

avg_nucleon = (PR * proton_part + Z * neitron_part) / A
mass = A * avg_nucleon * nuclear_binding_correction(A, Z=92)
print(mass)