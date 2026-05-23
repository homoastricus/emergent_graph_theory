import math

K = 6.0
lnN_base = 280.11152

print("ПРОВЕРКА УСТОЙЧИВОСТИ ТОЖДЕСТВА")
print(f"""
  K·(ln 3)² - (ln 3)/K + (ln 3)/ln N ≈ 1/(π - 3)

  Целевое значение: 1/(π - 3) = {1 / (math.pi - 3):.10f}
""")

print(f"{'ln N':<15} {'RHS':<20} {'Ошибка':<15} {'Статус'}")
print("-" * 60)

best_err = float('inf')
best_lnN = lnN_base

for dN in [-5.0, -2.0, -1.0, -0.5, -0.2, -0.1, -0.05, -0.01, -0.001,
           0,
           0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]:
    lnN_test = lnN_base + dN
    rhs = K * (math.log(3) ** 2) - math.log(3) / K + math.log(3) / lnN_test
    lhs = 1 / (math.pi - 3)
    err = abs(rhs - lhs) / lhs

    if err < best_err:
        best_err = err
        best_lnN = lnN_test

    if err < 0.001:
        status = "⭐⭐⭐ ОТЛИЧНО"
    elif err < 0.01:
        status = "⭐⭐ ХОРОШО"
    elif err < 0.05:
        status = "⭐ ПРИЕМЛЕМО"
    else:
        status = "⚠️"

    marker = " ← БАЗОВОЕ" if dN == 0 else ""
    print(f"{lnN_test:<15.6f} {rhs:<20.10f} {err:<15.8f} {status}{marker}")

print(f"\nЛучшее совпадение при ln N = {best_lnN:.6f}, ошибка = {best_err:.8f}")

# Анализ чувствительности
print("АНАЛИЗ ЧУВСТВИТЕЛЬНОСТИ")

lnN_sens = lnN_base
rhs_base = K * (math.log(3) ** 2) - math.log(3) / K + math.log(3) / lnN_sens
derivative = -math.log(3) / lnN_sens ** 2

print(f"""
  d(RHS)/d(ln N) = -ln 3 / (ln N)² = {derivative:.6e}

  При изменении ln N на 1:
    RHS изменяется на {derivative:.6e}
    Относительное изменение: {abs(derivative) / rhs_base:.6e}

  Для изменения ошибки на 0.01% требуется:
    Δ(ln N) ≈ {0.0001 * rhs_base / abs(derivative):.1f}
""")

# Где ошибка минимальна?
print(f"""
  ОПТИМАЛЬНОЕ ln N:
    Базовое: {lnN_base:.6f} (ошибка = {abs(K * (math.log(3) ** 2) - math.log(3) / K + math.log(3) / lnN_base - 1 / (math.pi - 3)) / (1 / (math.pi - 3)):.8f})
    Лучшее:  {best_lnN:.6f} (ошибка = {best_err:.8f})
    Сдвиг:   {best_lnN - lnN_base:+.6f}
""")