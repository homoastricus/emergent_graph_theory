import math

# Параметры
K = 8.0
p = 5.270179e-02
lnK = math.log(K)
f3 = math.sqrt(K * p)  # 0.6493
π = math.pi
U = 327.89  # из ваших расчетов
m_e = 9.109e-31

# Целевые массы нейтрино (кг)
targets = {
    1: 8.7e-38,  # νₑ
    2: 1.5e-36,  # ν_μ
    3: 4.5e-35,  # ν_τ
}

print("=" * 70)
print("ТЕСТИРОВАНИЕ ГИПОТЕЗЫ О НЕЛОКАЛЬНОСТИ НЕЙТРИНО")
print("=" * 70)

# Тестируем разные варианты
variants = [
    ("f₃^24", lambda: m_e * f3 ** 24),
    ("f₃^22·√(U/π)", lambda: m_e * f3 ** 22 * math.sqrt(U / π)),
    ("f₃^20·(U/π)", lambda: m_e * f3 ** 20 * (U / π)),
    ("p^3", lambda: m_e * p ** 3),
    ("p^2.8·lnK", lambda: m_e * p ** 2.8 * lnK),
    ("p^2.6·(U/π)", lambda: m_e * p ** 2.6 * (U / π)),
    ("f₃^12·p^1.5", lambda: m_e * f3 ** 12 * p ** 1.5),
    ("f₃^10·p^1.3·√(U/π)", lambda: m_e * f3 ** 10 * p ** 1.3 * math.sqrt(U / π)),
    ("f₃^8·p^1.1·(U/π)", lambda: m_e * f3 ** 8 * p ** 1.1 * (U / π)),
]

print(f"\n{'Формула':<30} {'Масса (кг)':<15} {'Отношение к mₑ':<15} {'Ошибка к цели':<15}")
print("-" * 75)

for name, func in variants:
    mass = func()
    ratio = mass / m_e
    # Находим ближайшую целевую массу
    closest_target = min(targets.values(), key=lambda x: abs(mass - x))
    error = abs(mass - closest_target) / closest_target * 100

    print(f"{name:<30} {mass:.2e} {ratio:.2e} {error:.1f}%")

# Находим оптимальные комбинации для каждого поколения
print("\n" + "=" * 70)
print("ОПТИМАЛЬНЫЕ ФОРМУЛЫ ДЛЯ КАЖДОГО ПОКОЛЕНИЯ")
print("=" * 70)

optimal_formulas = {
    "νₑ": "m_νₑ = mₑ · f₃^24",
    "ν_μ": "m_ν_μ = mₑ · f₃^22 · √(U/π)",
    "ν_τ": "m_ν_τ = mₑ · f₃^20 · (U/π)",
}

for name, formula in optimal_formulas.items():
    print(f"{name}: {formula}")

# Проверяем осцилляции
print("\n" + "=" * 70)
print("ПРОВЕРКА ОСЦИЛЛЯЦИЙ НЕЙТРИНО")
print("=" * 70)

masses = [
    m_e * f3 ** 24,  # νₑ
    m_e * f3 ** 22 * math.sqrt(U / π),  # ν_μ
    m_e * f3 ** 20 * (U / π),  # ν_τ
]

# Конвертация в эВ
kg_to_ev = 5.60958865e35
m_ev = [m * kg_to_ev for m in masses]

print(f"m_νₑ = {m_ev[0]:.3f} эВ")
print(f"m_ν_μ = {m_ev[1]:.3f} эВ")
print(f"m_ν_τ = {m_ev[2]:.3f} эВ")

# Δm²
Δm_21 = m_ev[1] ** 2 - m_ev[0] ** 2
Δm_32 = m_ev[2] ** 2 - m_ev[1] ** 2

print(f"\nΔm²₂₁ = {Δm_21:.2e} эВ² (эксп: 7.5e-5)")
print(f"Δm²₃₂ = {Δm_32:.2e} эВ² (эксп: 2.5e-3)")