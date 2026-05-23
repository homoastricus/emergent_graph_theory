import math

# ============================================================
# ФУНДАМЕНТАЛЬНЫЕ КОНСТАНТЫ
# ============================================================
K = 6.0
pi = math.pi
lnK = math.log(K)
sqrtK = math.sqrt(K)

# ============================================================
# CODATA
# ============================================================
alpha_CODATA = 1 / 137.035999084

# ============================================================
# ВЫЧИСЛЕНИЕ ln N_zeta
# ============================================================
X = K ** (1.5 + pi ** 2 / 6.0)  # X = ln N_zeta

print("╔══════════════════════════════════════════════════════════════════╗")
print("║   ИСПРАВЛЕННАЯ ФОРМУЛА ДЛЯ α ЧЕРЕЗ ln N_zeta                  ║")
print("╚══════════════════════════════════════════════════════════════════╝")
print(f"\n  K = {K}")
print(f"  π = {pi:.10f}")
print(f"  ln K = {lnK:.10f}")
print(f"  X = ln N_zeta = K^(3/2 + π²/6) = {X:.10f}")

# ============================================================
# ИСПРАВЛЕННАЯ ПОЛНАЯ ФОРМУЛА ДЛЯ α
# ============================================================
print(f"\n{'─' * 70}")
print("ИСПРАВЛЕННАЯ ФОРМУЛА (точный знаменатель в α-поправке)")
print(f"{'─' * 70}")

# Числитель
numerator = 2 * lnK ** 2 / pi
print(f"\n  Числитель: 2·ln²K / π = {numerator:.10f}")

# Знаменатель: три члена (ИСПРАВЛЕННЫЙ)
term1 = X  # главный член
term2 = -pi ** 6 / (sqrtK * X ** 2)  # квантовая поправка
term3_corrected = lnK ** 2 / (2 * pi ** 2 * (K - 1 / X))  # α-поправка (ТОЧНАЯ)

print(f"\n  Знаменатель ln N_α:")
print(f"    Главный член (X):                         {term1:.10f}")
print(f"    Квантовая поправка (-π⁶/(√K·X²)):       {term2:.10f}")
print(f"    α-поправка (ТОЧНАЯ, +ln²K/(2π²(K-1/X))): {term3_corrected:.10f}")

denominator = term1 + term2 + term3_corrected
print(f"    СУММА (ln N_α):                           {denominator:.10f}")

# α
alpha_calculated = numerator / denominator
print(f"\n  α = числитель / знаменатель")
print(f"    = {numerator:.10f} / {denominator:.10f}")
print(f"    = {alpha_calculated:.15f}")

# ============================================================
# СРАВНЕНИЕ
# ============================================================
print(f"\n{'─' * 70}")
print("СРАВНЕНИЕ С ЭКСПЕРИМЕНТОМ")
print(f"{'─' * 70}")

print(f"\n  α (вычисленное) = {alpha_calculated:.15f}")
print(f"  α (CODATA)      = {alpha_CODATA:.15f}")
print(f"  1/α (вычисл.)   = {1 / alpha_calculated:.10f}")
print(f"  1/α (CODATA)    = {1 / alpha_CODATA:.10f}")

relative_error = abs(alpha_calculated / alpha_CODATA - 1) * 100
print(f"\n  Относительная ошибка: {relative_error:.10f}%")

# ============================================================
# СРАВНЕНИЕ С ВЕДУЩИМ ПОРЯДКОМ
# ============================================================
print(f"\n{'─' * 70}")
print("СРАВНЕНИЕ С ВЕДУЩИМ ПОРЯДКОМ")
print(f"{'─' * 70}")

alpha_leading = 2 * lnK ** 2 / (pi * X)
print(f"\n  α (ведущий порядок) = 2·ln²K/(π·X) = {alpha_leading:.15f}")
print(f"  Ошибка ведущего порядка: {abs(alpha_leading / alpha_CODATA - 1) * 100:.10f}%")
print(f"  Ошибка полной формулы:   {relative_error:.10f}%")
if relative_error > 1e-15:
    print(f"  Улучшение в {abs(alpha_leading / alpha_CODATA - 1) * 100 / relative_error:.1f} раз")
else:
    print(f"  Улучшение: МАШИННАЯ ТОЧНОСТЬ")

# ============================================================
# ПРОВЕРКА КОМПОНЕНТ
# ============================================================
print(f"\n{'─' * 70}")
print("ПРОВЕРКА КОМПОНЕНТ (детали)")
print(f"{'─' * 70}")

# Вычисляем geometric_zeta
lnN_geom = (K - lnK) / (1 / 3 - 1 / pi)
geom_zeta = lnN_geom - X
print(f"\n  ln N_geom = {lnN_geom:.10f}")
print(f"  geometric_zeta = ln N_geom - X = {geom_zeta:.10f}")

# Вычисляем alpha_phys из тождества
alpha_phys_identity = (lnK ** 2 / X) / ((K - 1 / X) * geom_zeta)
print(f"  alpha_phys (из тождества)  = {alpha_phys_identity:.10f}")

# Вычисляем alpha_phys из ИСПРАВЛЕННОЙ формулы
alpha_phys_formula = lnK ** 2 / (2 * pi ** 2 * (K - 1 / X))
print(f"  alpha_phys (из формулы)    = {alpha_phys_formula:.10f}")
print(f"  Совпадение: {abs(alpha_phys_identity / alpha_phys_formula - 1) * 100:.10f}%")

# ============================================================
# РАЗЛОЖЕНИЕ ПОПРАВОК
# ============================================================
print(f"\n{'─' * 70}")
print("РАЗЛОЖЕНИЕ ПОПРАВОК")
print(f"{'─' * 70}")

alpha_1 = numerator / term1  # только главный член
alpha_2 = numerator / (term1 + term2)  # + квантовая поправка
alpha_3 = numerator / (term1 + term2 + term3_corrected)  # полная формула

print(f"\n  α (только главный член):     {alpha_1:.15f}  (ошибка: {abs(alpha_1 / alpha_CODATA - 1) * 100:.8f}%)")
print(f"  α (+ квантовая поправка):    {alpha_2:.15f}  (ошибка: {abs(alpha_2 / alpha_CODATA - 1) * 100:.8f}%)")
print(f"  α (полная формула):          {alpha_3:.15f}  (ошибка: {abs(alpha_3 / alpha_CODATA - 1) * 100:.8f}%)")

# ============================================================
# ИТОГ
# ============================================================
print(f"\n{'═' * 70}")
print("ИТОГ")
print(f"{'═' * 70}")
print(f"""
  ИСПРАВЛЕННАЯ ФОРМУЛА ДЛЯ α (только через π и K=6):

  X = K^(3/2 + π²/6) = ln N_zeta

  α = (2·ln²K / π) / (X - π⁶/(√K·X²) + ln²K/(2π²·(K - 1/X)))

  Вычисленное значение: α = {alpha_calculated:.15f}
  CODATA:               α = {alpha_CODATA:.15f}
  1/α (вычисл.):        {1 / alpha_calculated:.10f}
  1/α (CODATA):         {1 / alpha_CODATA:.10f}
  Отклонение:           {relative_error:.10f}%

  Улучшение относительно ведущего порядка: {abs(alpha_leading / alpha_CODATA - 1) * 100 / relative_error:.1f} раз
""")