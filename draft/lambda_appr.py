import numpy as np
from scipy import constants
import math
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# Ваши данные с новыми весами
N_values = np.array([1e3, 1e6, 1e9, 1e12, 1e15, 1e18, 1e19])
lambda_values = np.array([1.767989e-01, 1.483236e-01, 1.423774e-01,
                          1.395783e-01, 1.379055e-01,  1.367906e-01, 1.364972e-01])  # последнее - с новыми весами

print("АНАЛИЗ С НОВЫМИ ВЕСАМИ [0.1, 0.1, 0.7, 0.1]")
print("=" * 50)
print("N\t\tλ")
for N, lmbda in zip(N_values, lambda_values):
    print(f"{N:.1e}\t{lmbda:.3e}")

# Логарифмический анализ
logN = np.log10(N_values)
loglambda = np.log10(lambda_values)

# Линейная регрессия в логарифмической шкале
slope, intercept = np.polyfit(logN, loglambda, 1)

print(f"\nЛОГАРИФМИЧЕСКАЯ АППРОКСИМАЦИЯ:")
print(f"logλ = {slope:.4f} * logN + {intercept:.4f}")
print(f"То есть: λ ~ N^({slope:.4f})")

# Прогноз для N = 10^123
N_target = 1e123
lambda_predicted = 10**(slope * np.log10(N_target) + intercept)

print(f"\nПРОГНОЗ ДЛЯ N = 10^123:")
print(f"λ_predicted = {lambda_predicted:.3e}")
print(f"Целевое λ = 1.000e-05")
print(f"Отношение: {lambda_predicted/1e-5:.3f}")

# Анализ точности
error_orders = np.log10(lambda_predicted/1e-5)
print(f"Ошибка в порядках: {error_orders:.2f}")

# Проверка сходимости
if abs(slope + 1/3) < 0.05:
    print("✅ Идеальная сходимость: λ ~ N^(-1/3)")
elif abs(slope + 0.5) < 0.05:
    print("✅ Хорошая сходимость: λ ~ N^(-1/2)")
elif abs(slope + 0.4) < 0.1:
    print("✅ Умеренная сходимость к λ ~ N^(-0.4)")
else:
    print(f"⚠️ Нестандартная сходимость: λ ~ N^({slope:.3f})")

# Детальный анализ поведения
print(f"\nДЕТАЛЬНЫЙ АНАЛИЗ:")
print(f"Текущий наклон: {slope:.4f}")

# Аппроксимация степенным законом
def power_law(N, a, b):
    return a * N**b

# Используем последние 3 точки для лучшей аппроксимации асимптотики
N_recent = N_values[2:]
lambda_recent = lambda_values[2:]
popt, pcov = curve_fit(power_law, N_recent, lambda_recent)
a_power, b_power = popt

print(f"Степенная аппроксимация (по последним точкам):")
print(f"λ = {a_power:.3e} * N^({b_power:.4f})")

lambda_power_predicted = a_power * N_target**b_power
print(f"Прогноз (степенной): {lambda_power_predicted:.3e}")
print(f"Отношение к целевому: {lambda_power_predicted/1e-5:.3f}")

# Визуализация тренда
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

# Логарифмический график
plt.subplot(1, 2, 1)
plt.loglog(N_values, lambda_values, 'bo-', label='Данные', markersize=8)
N_extended = np.logspace(19, 123, 100)
plt.loglog(N_extended, 10**(slope * np.log10(N_extended) + intercept),
           'r--', label=f'Линейная аппроксимация')
plt.axhline(y=1e-5, color='g', linestyle=':', label='λ=1e-5')
plt.xlabel('N')
plt.ylabel('λ')
plt.legend()
plt.grid(True, alpha=0.3)
plt.title('Логарифмическая шкала')

# Линейная шкала λ
plt.subplot(1, 2, 2)
plt.semilogx(N_values, lambda_values, 'bo-', label='Данные', markersize=8)
plt.semilogx(N_extended, 10**(slope * np.log10(N_extended) + intercept),
           'r--', label='Аппроксимация')
plt.axhline(y=1e-5, color='g', linestyle=':', label='λ=1e-5')
plt.xlabel('N')
plt.ylabel('λ')
plt.legend()
plt.grid(True, alpha=0.3)
plt.title('Линейная шкала λ')

plt.tight_layout()
plt.show()

# Окончательная оценка
print(f"\n🎯 ИТОГОВАЯ ОЦЕНКА:")
print(f"При N = 10^123:")
print(f"λ ≈ {lambda_predicted:.2e} (линейная аппроксимация)")
print(f"λ ≈ {lambda_power_predicted:.2e} (степенная аппроксимация)")

if abs(lambda_predicted - 1e-5) < 1e-6:
    print("🎉 ИДЕАЛЬНОЕ СОВПАДЕНИЕ!")
elif abs(np.log10(lambda_predicted/1e-5)) < 1:
    print("✅ ОТЛИЧНОЕ СОВПАДЕНИЕ (в пределах 1 порядка)")
elif abs(np.log10(lambda_predicted/1e-5)) < 2:
    print("⚠️ ХОРОШЕЕ СОВПАДЕНИЕ (в пределах 2 порядков)")
else:
    print("❌ ТРЕБУЕТСЯ ДОРАБОТКА")