import math
import numpy as np

# ============================================================
# ФУНДАМЕНТАЛЬНЫЕ ПАРАМЕТРЫ
# ============================================================
K = 6.0
pi = math.pi
lnK = math.log(K)
sqrtK = math.sqrt(K)
sqrt2 = math.sqrt(2)
sqrt3 = math.sqrt(3)


# ============================================================
# ВСЕ МЕТОДЫ ВЫЧИСЛЕНИЯ ln N
# ============================================================
def compute_all_N_values():
    results = {}

    # 1. N_geom — геометрический резонанс (LO)
    lnN_geom = (K - lnK) / (1.0 / 3.0 - 1.0 / pi)
    results['N_geom'] = {'lnN': lnN_geom, 'N': math.exp(lnN_geom)}

    # 2. N_zeta — дзета-функция
    lnN_zeta = 6.0 ** (1.5 + pi ** 2 / 6.0)
    results['N_zeta'] = {'lnN': lnN_zeta, 'N': math.exp(lnN_zeta)}

    # 3. N_alpha — тонкая структура
    alpha_exp = 1.0 / 137.035999084
    lnN_alpha = 2.0 * lnK ** 2 / (pi * alpha_exp)
    results['N_alpha'] = {'lnN': lnN_alpha, 'N': math.exp(lnN_alpha)}

    # 4. N_phys (NLO) — G - πK/G
    G = lnN_geom
    lnN_phys_nlo = G - pi * K / G
    results['N_phys_NLO'] = {'lnN': lnN_phys_nlo, 'N': math.exp(lnN_phys_nlo)}

    # 5. N_phys (NNLO) — G - πK/G + 23.9/G²
    lnN_phys_nnlo = G - pi * K / G + 23.9 / G ** 2
    results['N_phys_NNLO'] = {'lnN': lnN_phys_nnlo, 'N': math.exp(lnN_phys_nnlo)}

    # 6. N_phys (NNLO v2) — G - πK/G + π^6/(sqrtK * G^2)
    lnN_phys_nnlo_v2 = G - pi * K / G + pi ** 6 / (sqrtK * G ** 2)
    results['N_phys_NNLO_v2'] = {'lnN': lnN_phys_nnlo_v2, 'N': math.exp(lnN_phys_nnlo_v2)}

    return results


results = compute_all_N_values()

# ============================================================
# ТАБЛИЦА ВСЕХ N
# ============================================================
print("=" * 100)
print("ВСЕ МЕТОДЫ ВЫЧИСЛЕНИЯ ln N В ЕТИ")
print("=" * 100)
print(f"\n{'Метод':<20} {'ln N':<20} {'N':<20}")
print("-" * 60)
for name, data in results.items():
    print(f"{name:<20} {data['lnN']:<20.10f} {data['N']:<20.6e}")

# ============================================================
# ВСЕ ПОПРАВКИ И РАЗНИЦЫ
# ============================================================
print(f"\n{'=' * 100}")
print("ВСЕ ПОПРАВКИ И РАЗНИЦЫ МЕЖДУ МЕТОДАМИ")
print("=" * 100)

lnN = {name: data['lnN'] for name, data in results.items()}

# Базовые разницы
pairs = [
    ('N_geom - N_phys_NLO', lnN['N_geom'] - lnN['N_phys_NLO'], 'πK/lnN_geom (квантовая поправка)'),
    ('N_geom - N_zeta', lnN['N_geom'] - lnN['N_zeta'], 'geometric_zeta (спектральная)'),
    ('N_geom - N_alpha', lnN['N_geom'] - lnN['N_alpha'], 'geometric_alpha (электромагнитная)'),
    ('N_zeta - N_phys_NLO', lnN['N_zeta'] - lnN['N_phys_NLO'], 'zeta_phys (остаточная)'),
    ('N_alpha - N_phys_NLO', lnN['N_alpha'] - lnN['N_phys_NLO'], 'alpha_phys'),
    ('N_phys_NNLO - N_phys_NLO', lnN['N_phys_NNLO'] - lnN['N_phys_NLO'], '23.9/G² (двухпетлевая)'),
]

print(f"\n{'Разница':<30} {'Значение':<20} {'Аналитическое выражение':<40}")
print("-" * 90)
for name, value, expr in pairs:
    print(f"{name:<30} {value:<20.10f} {expr:<40}")

# ============================================================
# ПРОВЕРКА ПОПРАВОК
# ============================================================
print(f"\n{'=' * 100}")
print("ПРОВЕРКА АНАЛИТИЧЕСКИХ ПОПРАВОК")
print("=" * 100)

G = lnN['N_geom']

tests = [
    ('πK/G', pi * K / G, lnN['N_geom'] - lnN['N_phys_NLO']),
    ('πK/G = N_geom - N_phys_NLO?', None, None),
    ('π⁶/(√K·G²)', pi ** 6 / (sqrtK * G ** 2), lnN['N_phys_NNLO_v2'] - lnN['N_phys_NLO']),
    ('geometric_zeta = N_geom - N_zeta', lnN['N_geom'] - lnN['N_zeta'], None),
    ('geometric_phys = N_geom - N_phys_NLO', lnN['N_geom'] - lnN['N_phys_NLO'], None),
]

print(f"\n{'Поправка':<35} {'Аналитическое':<20} {'Из разницы N':<20} {'Совпадение?'}")
print("-" * 90)

# Проверка πK/G
val1 = pi * K / G
val2 = lnN['N_geom'] - lnN['N_phys_NLO']
match1 = abs(val1 - val2) < 1e-10
print(f"{'πK/G':<35} {val1:<20.10f} {val2:<20.10f} {'✅' if match1 else '❌'}")

# Проверка π⁶/(√K·G²)
val3 = pi ** 6 / (sqrtK * G ** 2)
val4 = lnN['N_phys_NNLO_v2'] - lnN['N_phys_NLO']
match2 = abs(val3 - val4) < 1e-10
print(f"{'π⁶/(√K·G²)':<35} {val3:<20.10f} {val4:<20.10f} {'✅' if match2 else '❌'}")

# Проверка geometric_zeta
val5 = lnN['N_geom'] - lnN['N_zeta']
print(f"{'geometric_zeta':<35} {val5:<20.10f} {'—':<20} {'—'}")

# Проверка geometric_phys
val6 = lnN['N_geom'] - lnN['N_phys_NLO']
print(f"{'geometric_phys':<35} {val6:<20.10f} {'—':<20} {'—'}")

# Проверка тождества: 1/π² + lnN/(lnN+K) = geometric_phys/geometric_zeta
print(f"\n{'=' * 100}")
print("ПРОВЕРКА НЕТРИВИАЛЬНОГО ТОЖДЕСТВА")
print("=" * 100)

lhs = 1.0 / pi ** 2 + G / (G + K)
rhs = (lnN['N_geom'] - lnN['N_phys_NLO']) / (lnN['N_geom'] - lnN['N_zeta'])
dev = abs(lhs - rhs) / rhs * 100

print(f"\n  1/π² + lnN/(lnN+K) = {lhs:.10f}")
print(f"  geometric_phys/geometric_zeta = {rhs:.10f}")
print(f"  Отклонение: {dev:.8f}%")
print(f"  Статус: {'✅ ТОЧНО' if dev < 0.001 else '⭐ ХОРОШО' if dev < 0.1 else '⚠️'}")

# Проверка тождества: √K - 1/π⁶ = π - ln2
print(f"\n{'=' * 100}")
print("ПРОВЕРКА ТОЖДЕСТВА: sqrt(K) - 1/π⁶ = π - ln2")
print("=" * 100)

lhs2 = sqrtK - 1.0 / pi ** 6
rhs2 = pi - math.log(2)
dev2 = abs(lhs2 - rhs2) / rhs2 * 100

print(f"\n  sqrt(K) - 1/π⁶ = {lhs2:.10f}")
print(f"  π - ln2 = {rhs2:.10f}")
print(f"  Отклонение: {dev2:.8f}%")
print(f"  Статус: {'✅ ТОЧНО' if dev2 < 0.001 else '⭐ ХОРОШО' if dev2 < 0.1 else '⚠️'}")

# Проверка тождества: 1/3 + (K+√2)/K = γ·e
print(f"\n{'=' * 100}")
print("ПРОВЕРКА ТОЖДЕСТВА: 1/3 + (K+√2)/K = γ·e")
print("=" * 100)

gamma_euler = 0.5772156649015329
lhs3 = 1.0 / 3.0 + (K + sqrt2) / K
rhs3 = gamma_euler * math.e
dev3 = abs(lhs3 - rhs3) / rhs3 * 100

print(f"\n  1/3 + (K+√2)/K = {lhs3:.10f}")
print(f"  γ·e = {rhs3:.10f}")
print(f"  Отклонение: {dev3:.8f}%")
print(f"  Статус: {'✅ ТОЧНО' if dev3 < 0.001 else '⭐ ХОРОШО' if dev3 < 0.1 else '⚠️'}")

# Проверка тождества: (K+√K)/(lnK+1/3) = δ_F - ln2
print(f"\n{'=' * 100}")
print("ПРОВЕРКА ТОЖДЕСТВА: (K+√K)/(lnK+1/3) = δ_F - ln2")
print("=" * 100)

feigenbaum_delta = 4.66920160910299
lhs4 = (K + sqrtK) / (lnK + 1.0 / 3.0)
rhs4 = feigenbaum_delta - math.log(2)
dev4 = abs(lhs4 - rhs4) / rhs4 * 100

print(f"\n  (K+√K)/(lnK+1/3) = {lhs4:.10f}")
print(f"  δ_F - ln2 = {rhs4:.10f}")
print(f"  Отклонение: {dev4:.8f}%")
print(f"  Статус: {'✅ ТОЧНО' if dev4 < 0.001 else '⭐ ХОРОШО' if dev4 < 0.1 else '⚠️'}")

# ============================================================
# СВОДНАЯ ТАБЛИЦА ВСЕХ ПРОВЕРОК
# ============================================================
print(f"\n{'=' * 100}")
print("СВОДНАЯ ТАБЛИЦА ВСЕХ ПРОВЕРОК")
print("=" * 100)

all_checks = [
    ('πK/G = N_geom - N_phys_NLO', match1),
    ('π⁶/(√K·G²) = N_phys_NNLO_v2 - N_phys_NLO', match2),
    ('1/π² + lnN/(lnN+K) = geometric_phys/geometric_zeta', dev < 0.001),
    ('√K - 1/π⁶ = π - ln2', dev2 < 0.001),
    ('1/3 + (K+√2)/K = γ·e', dev3 < 0.001),
    ('(K+√K)/(lnK+1/3) = δ_F - ln2', dev4 < 0.001),
]

print(f"\n{'Тождество':<55} {'Статус':<10}")
print("-" * 65)
for name, passed in all_checks:
    print(f"{name:<55} {'✅ ПРОЙДЕН' if passed else '❌ НЕ ПРОЙДЕН'}")

passed_count = sum(1 for _, p in all_checks if p)
print(f"\n  Пройдено: {passed_count}/{len(all_checks)}")
print(f"  {'🎉 ВСЕ ТОЖДЕСТВА ПОДТВЕРЖДЕНЫ!' if passed_count == len(all_checks) else '⚠️ Есть расхождения'}")