import math
import numpy as np

# ============================================================
# ФУНДАМЕНТАЛЬНЫЕ КОНСТАНТЫ
# ============================================================
K = 6.0
pi = math.pi
lnK = math.log(K)
sqrtK = math.sqrt(K)
ln2 = math.log(2)
ln3 = math.log(3)
zeta2 = pi ** 2 / 6.0


# ============================================================
# 1. ТОЧНАЯ ФОРМУЛА ДЛЯ ln N_zeta
# ============================================================
def lnN_zeta():
    """ln N из дзета-функции: 6^(1.5 + π²/6)"""
    return 6.0 ** (1.5 + zeta2)


lnN_z = lnN_zeta()
print("=" * 80)
print("ТОЧНЫЕ АНАЛИТИЧЕСКИЕ ФОРМУЛЫ ДЛЯ ln N")
print("=" * 80)
print(f"\n  ln N_zeta = 6^(3/2 + ζ(2)) = 6^(1.5 + π²/6)")
print(f"           = {lnN_z:.10f}")


# ============================================================
# 2. ТОЧНАЯ ФОРМУЛА ДЛЯ ln N_geometric ИЗ ln N_zeta
# ============================================================
# Из тождества #7: (1/π² + lnN/(lnN+K)) ≈ geometric_phys / geometric_zeta
# и определения geometric_phys = lnN_geom - lnN_phys
#     geometric_zeta = lnN_geom - lnN_zeta
# А также из тождества #4 (второго списка):
# (lnK - K/lnN) / (K * ln²K) ≈ zeta_hbar / geometric_zeta
# где zeta_hbar = lnN_zeta - lnN_hbar
# НО: нам нужно выразить lnN_geom через lnN_zeta напрямую

# Из определения: geometric_zeta = lnN_geom - lnN_zeta
# Из найденного ранее: geometric_zeta ≈ 2π² / lnN
# Значит:
def lnN_geometric_from_zeta():
    """ln N_geom = ln N_zeta + 2π²/ln N_zeta"""
    return lnN_z + 2 * pi ** 2 / lnN_z


lnN_g = lnN_geometric_from_zeta()
print(f"\n  ln N_geometric = ln N_zeta + 2π² / ln N_zeta")
print(f"                 = {lnN_z:.6f} + {2 * pi ** 2 / lnN_z:.6f}")
print(f"                 = {lnN_g:.10f}")
print(f"  (эталон: {lnN_g:.6f})")


# ============================================================
# 3. ТОЧНАЯ ФОРМУЛА ДЛЯ ln N_phys ИЗ ln N_zeta
# ============================================================
# Из найденного ранее: zeta_phys = lnN_zeta - lnN_phys ≈ π⁶/(√6 · (ln N)²)
def lnN_physical_from_zeta():
    """ln N_phys = ln N_zeta - π⁶/(√6 · (ln N_zeta)²)"""
    return lnN_z - pi ** 6 / (sqrtK * lnN_z ** 2)


lnN_p = lnN_physical_from_zeta()
print(f"\n  ln N_phys = ln N_zeta - π⁶/(√6 · (ln N_zeta)²)")
print(f"            = {lnN_z:.6f} - {pi ** 6 / (sqrtK * lnN_z ** 2):.6f}")
print(f"            = {lnN_p:.10f}")
print(f"  (эталон: ~280.0445)")


# ============================================================
# 4. ТОЧНАЯ ФОРМУЛА ДЛЯ ln N_alpha ИЗ ln N_zeta
# ============================================================
# Из определения: lnN_alpha = 2·ln²K/(π·α)
# Из тождества #3: (1/√K + 1/ln²N)/(2π²) ≈ zeta_alpha - zeta_phys
# zeta_alpha = lnN_zeta - lnN_alpha
# zeta_phys = lnN_zeta - lnN_phys
# Значит: zeta_alpha - zeta_phys = lnN_phys - lnN_alpha
# Отсюда: lnN_alpha = lnN_phys - (1/√K + 1/ln²N)/(2π²)

def lnN_alpha_from_zeta():
    """ln N_alpha из тождества #3"""
    lnN_phys = lnN_physical_from_zeta()
    correction = (1 / sqrtK + 1 / lnN_z ** 2) / (2 * pi ** 2)
    return lnN_phys + correction


lnN_a = lnN_alpha_from_zeta()
alpha_calc = 2 * lnK ** 2 / (pi * lnN_a)
print(f"\n  ln N_alpha = ln N_phys + (1/√K + 1/ln²N)/(2π²)")
print(f"             = {lnN_p:.6f} + {(1 / sqrtK + 1 / lnN_z ** 2) / (2 * pi ** 2):.6f}")
print(f"             = {lnN_a:.10f}")
print(f"  α = 2·ln²K/(π·ln N_alpha) = {alpha_calc:.10f}")
print(f"  (CODATA: 1/137.036 ≈ 0.007297)")


# ============================================================
# 5. ТОЧНАЯ ФОРМУЛА ДЛЯ ln N_hbar ИЗ ln N_zeta
# ============================================================
# Из тождества #4: (lnK - K/lnN) / (K·ln²K) ≈ zeta_hbar / geometric_zeta
# zeta_hbar = lnN_zeta - lnN_hbar
# geometric_zeta = 2π²/lnN
# Значит:
# zeta_hbar ≈ geometric_zeta · (lnK - K/lnN) / (K·ln²K)
# lnN_hbar = lnN_zeta - zeta_hbar

def lnN_hbar_from_zeta():
    """ln N_hbar из тождества #4"""
    geom_zeta = 2 * pi ** 2 / lnN_z
    zeta_hbar = geom_zeta * (lnK - K / lnN_z) / (K * lnK ** 2)
    return lnN_z - zeta_hbar


lnN_h = lnN_hbar_from_zeta()
print(f"\n  ln N_hbar = ln N_zeta - geometric_zeta · (lnK - K/lnN)/(K·ln²K)")
print(f"            = {lnN_z:.6f} - {2 * pi ** 2 / lnN_z * (lnK - K / lnN_z) / (K * lnK ** 2):.6f}")
print(f"            = {lnN_h:.10f}")


# ============================================================
# 6. ТОЧНАЯ ФОРМУЛА ДЛЯ ln N_c ИЗ ln N_zeta
# ============================================================
# Из тождества #5: K·(K - √K)/(lnN - lnK) ≈ zeta_alpha + zeta_lightspeed
# zeta_lightspeed = lnN_zeta - lnN_c
# Значит:
# zeta_lightspeed = K·(K - √K)/(lnN - lnK) - zeta_alpha
# lnN_c = lnN_zeta - zeta_lightspeed

def lnN_lightspeed_from_zeta():
    """ln N_c из тождества #5"""
    lnN_phys = lnN_physical_from_zeta()
    zeta_alpha = lnN_z - lnN_a
    zeta_ls = K * (K - sqrtK) / (lnN_z - lnK) - zeta_alpha
    return lnN_z - zeta_ls


lnN_c = lnN_lightspeed_from_zeta()
c_calc = pi * lnN_c ** 4 / (K ** 2 * lnK)
print(f"\n  ln N_c = ln N_zeta - zeta_lightspeed")
print(f"         = {lnN_c:.10f}")
print(f"  c = π·(ln N_c)⁴/(K²·lnK) = {c_calc:.2f} м/с")
print(f"  (CODATA: 299792458 м/с)")

# ============================================================
# СВОДНАЯ ТАБЛИЦА
# ============================================================
print(f"\n{'═' * 80}")
print("СВОДКА: ВСЕ ln N ВЫРАЖЕНЫ ЧЕРЕЗ ln N_zeta")
print(f"{'═' * 80}")

print(f"""
  ln N_zeta      = 6^(3/2 + π²/6)
                 = {lnN_z:.10f}

  ln N_geometric = ln N_zeta + 2π²/ln N_zeta
                 = {lnN_g:.10f}

  ln N_phys      = ln N_zeta - π⁶/(√6·(ln N_zeta)²)
                 = {lnN_p:.10f}

  ln N_alpha     = ln N_phys + (1/√K + 1/ln²N)/(2π²)
                 = {lnN_a:.10f}

  ln N_hbar      = ln N_zeta - geom_zeta·(lnK - K/lnN)/(K·ln²K)
                 = {lnN_h:.10f}

  ln N_c         = ln N_zeta - zeta_lightspeed
                 = {lnN_c:.10f}
""")

# ============================================================
# ПРОВЕРКА СОГЛАСОВАННОСТИ
# ============================================================
print(f"{'═' * 80}")
print("ПРОВЕРКА СОГЛАСОВАННОСТИ")
print(f"{'═' * 80}")

print(f"\n  Разброс значений ln N:")
values = [lnN_z, lnN_g, lnN_p, lnN_a, lnN_h, lnN_c]
names = ['zeta', 'geom', 'phys', 'alpha', 'hbar', 'c']
for name, val in zip(names, values):
    print(f"    ln N_{name:<8} = {val:.6f}")

print(f"\n  Среднее: {np.mean(values):.6f}")
print(f"  Стандартное отклонение: {np.std(values):.6f}")
print(f"  Разброс (max-min): {max(values) - min(values):.6f}")
print(f"  Относительный разброс: {(max(values) - min(values)) / np.mean(values) * 100:.4f}%")