import numpy as np
import matplotlib.pyplot as plt

# =========================
# фиксируем K
# =========================
K = 6
inv_pi = 1 / np.pi


# =========================
# аналитическая функция F(N) через логарифмы
# p(N) = 1 / (K * N^(1/3))
# ln(p) = -ln(K) - (1/3)*ln(N)
# =========================
def F_log(log_N, K):
    """Вычисляет F(N) используя логарифм N вместо самого N"""
    ln_N = log_N * np.log(10)  # переводим из log10 в натуральный логарифм
    ln_K = np.log(K)

    # ln(p) = -ln(K) - (1/3)*ln(N)
    ln_p = -ln_K - (1 / 3) * ln_N

    # p = exp(ln_p)
    p_val = np.exp(ln_p)

    # Числитель: K + ln(p)
    numerator = K + ln_p

    # Знаменатель: p - ln(N)
    denominator = p_val - ln_N

    return numerator / denominator


# =========================
# диапазон: степени от 0 до 300
# =========================
exponents = np.linspace(0, 300, num=1000)  # 1000 точек для гладкого графика

# Вычисляем F(N) через логарифмы
F_values = F_log(exponents, K)

# Вычисляем ошибку
error_values = np.abs((F_values - inv_pi) / inv_pi) * 100

# =========================
# построение графиков
# =========================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# График F(N) в зависимости от степени
ax1.plot(exponents, F_values, 'b-', linewidth=1)
ax1.axhline(y=inv_pi, color='r', linestyle='--', label=f'1/π = {inv_pi:.6f}')
ax1.set_xlabel('Степень (log₁₀ N)')
ax1.set_ylabel('F(N)')
ax1.set_title('График функции F(N) для степеней от 0 до 300')
ax1.legend()
ax1.grid(True, alpha=0.3)

# График относительной ошибки
ax2.plot(exponents, error_values, 'r-', linewidth=1)
ax2.set_xlabel('Степень (log₁₀ N)')
ax2.set_ylabel('Относительная ошибка (%)')
ax2.set_title('Относительная ошибка |F(N) - 1/π| / (1/π) × 100%')
ax2.grid(True, alpha=0.3)
ax2.set_yscale('log')  # логарифмическая шкала для ошибки

plt.tight_layout()
plt.show()

# =========================
# вывод некоторых значений
# =========================
print(f"{'Степень':>10} | {'N':>15} | {'F(N)':>20} | {'error %':>15}")
print("-" * 70)

# Выводим несколько точек
sample_exponents = [0, 10, 50, 100, 150, 200, 250, 300]
for exp in sample_exponents:
    idx = np.argmin(np.abs(exponents - exp))
    F_val = F_values[idx]
    err = error_values[idx]
    N_val = 10 ** exp if exp <= 300 else float('inf')
    if exp <= 15:  # для маленьких степеней показываем точное значение
        print(f"{exp:10.0f} | {N_val:15.3e} | {F_val:20.10f} | {err:15.8f}")
    else:
        print(f"{exp:10.0f} | 10^{exp:<11} | {F_val:20.10f} | {err:15.8f}")

# Дополнительно: покажем сходимость к 1/π
print(f"\nПредельное значение 1/π = {inv_pi:.10f}")
print(f"Значение F(N) при степени 300: {F_values[-1]:.10f}")
print(f"Разница с 1/π: {abs(F_values[-1] - inv_pi):.2e}")