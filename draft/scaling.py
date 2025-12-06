import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Ваши данные (приведем в порядок)
data = [
    (1000, 0.0135),
    (5000, 0.0201),
    (10000, 0.0185),
    (12000, 0.1350),  # ваше новое значение
    (50000, 0.0128)
]

# Сортируем по N
data.sort(key=lambda x: x[0])
N_vals = np.array([d[0] for d in data])
lambda_vals = np.array([d[1] for d in data])


# Модель 1: С насыщением
def model_saturation(N, a, b, N0):
    """λ₁ = a * (1 - exp(-(N/N0)^b))"""
    return a * (1 - np.exp(-(N / N0) ** b))


# Модель 2: Степенная с насыщением
def model_power_sat(N, alpha, beta, gamma):
    """λ₁ = alpha * N^beta / (1 + gamma * N^beta)"""
    return alpha * N ** beta / (1 + gamma * N ** beta)


# Модель 3: Простая степенная (ваша)
def model_power(N, a, b):
    """λ₁ = a * N^b"""
    return a * N ** b


# Подгонка
try:
    # Пробуем степенную модель
    popt_power, _ = curve_fit(model_power, N_vals, lambda_vals,
                              p0=[0.01, 0.1], bounds=([0, -2], [10, 2]))
    a_power, b_power = popt_power

    # Пробуем модель с насыщением
    popt_sat, _ = curve_fit(model_power_sat, N_vals, lambda_vals,
                            p0=[0.1, 0.1, 0.001],
                            bounds=([0, -1, 0], [10, 1, 1]))
    a_sat, b_sat, c_sat = popt_sat

    print(f"Степенная модель: λ₁ = {a_power:.4f} * N^{b_power:.4f}")
    print(f"Модель с насыщением: λ₁ = {a_sat:.4f} * N^{b_sat:.4f} / (1 + {c_sat:.4f} * N^{b_sat:.4f})")

    # Предсказание для больших N
    N_test = np.logspace(3, 10, 100)
    lambda_power = model_power(N_test, a_power, b_power)
    lambda_sat = model_power_sat(N_test, a_sat, b_sat, c_sat)

    # Визуализация
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.loglog(N_vals, lambda_vals, 'bo', label='Данные')
    plt.loglog(N_test, lambda_power, 'r-', label=f'Степенная: ∝ N^{b_power:.3f}')
    plt.loglog(N_test, lambda_sat, 'g--', label='С насыщением')
    plt.xlabel('N')
    plt.ylabel('λ₁')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(N_vals, lambda_vals, 'bo', label='Данные')
    plt.plot(N_test, lambda_power, 'r-')
    plt.plot(N_test, lambda_sat, 'g--')
    plt.xscale('log')
    plt.yscale('linear')
    plt.xlabel('N (log)')
    plt.ylabel('λ₁')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"Ошибка подгонки: {e}")
    print("Возможно, данные недостаточно согласованы")