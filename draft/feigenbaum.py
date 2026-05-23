import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import warnings
warnings.filterwarnings('ignore')

rcParams['figure.figsize'] = (16, 10)
rcParams['font.size'] = 11

FEIGENBAUM_DELTA = 4.669201609102990671853203820466
K = 6
N_special = 4.197e121

# Создаем очень детальный диапазон вокруг особой точки
N_zoom = np.logspace(120, 122, 5000)

def formula(K, N):
    return K * (K + 1/np.log(N)) / (K + 1/np.log(np.log(K)))

# Вычисляем значения
values_zoom = formula(K, N_zoom)
errors_zoom = np.abs(values_zoom - FEIGENBAUM_DELTA) / FEIGENBAUM_DELTA * 100

# Находим точный минимум ошибки
min_error_idx = np.argmin(errors_zoom)
N_min_error = N_zoom[min_error_idx]
min_error_value = errors_zoom[min_error_idx]

# Создаем комплексную визуализацию
fig = plt.figure(figsize=(16, 12))

# 1. Основной график: формула vs N в широком диапазоне
ax1 = plt.subplot(2, 2, 1)
N_wide = np.logspace(1, 300, 1000)
values_wide = formula(K, N_wide)

ax1.semilogx(N_wide, values_wide, 'b-', linewidth=2, alpha=0.7, label='Approximation')
ax1.axhline(y=FEIGENBAUM_DELTA, color='r', linestyle='--', linewidth=2,
            label=f'δ = {FEIGENBAUM_DELTA:.8f}...')
ax1.axvline(x=N_special, color='orange', linestyle=':', linewidth=2, alpha=0.7,
            label=f'N = 4.197e121')
ax1.fill_between(N_wide, FEIGENBAUM_DELTA - 0.001, FEIGENBAUM_DELTA + 0.001,
                  alpha=0.2, color='green')
ax1.set_xlabel('N (log scale)')
ax1.set_ylabel('Value')
ax1.set_title('Global View: Formula vs Feigenbaum Delta')
ax1.legend(loc='center right')
ax1.grid(True, alpha=0.3)
ax1.set_ylim([4.66, 5.05])

# 2. Детальный график вокруг особой точки
ax2 = plt.subplot(2, 2, 2)
ax2.semilogx(N_zoom, values_zoom, 'b-', linewidth=1.5, alpha=0.8)
ax2.axhline(y=FEIGENBAUM_DELTA, color='r', linestyle='--', linewidth=2)
ax2.axvline(x=N_special, color='orange', linestyle=':', linewidth=2, alpha=0.7)
ax2.scatter([N_min_error], [values_zoom[min_error_idx]], color='red', s=150, zorder=5,
           label=f'Minimum error at N≈{N_min_error:.2e}')
ax2.fill_between(N_zoom, FEIGENBAUM_DELTA - 1e-6, FEIGENBAUM_DELTA + 1e-6,
                  alpha=0.2, color='green')
ax2.set_xlabel('N (log scale)')
ax2.set_ylabel('Value')
ax2.set_title('Zoom: Around N = 4.197e121')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. График ошибки в широком диапазоне
ax3 = plt.subplot(2, 2, 3)
errors_wide = np.abs(values_wide - FEIGENBAUM_DELTA) / FEIGENBAUM_DELTA * 100

ax3.loglog(N_wide, errors_wide, 'purple', linewidth=2)
ax3.scatter([N_special], [0.0000370947], color='red', s=150, zorder=5,
           label=f'Your point: 0.000037% error')
ax3.axhline(y=0.059441, color='orange', linestyle='--', alpha=0.7,
           label=f'Asymptotic limit: 0.059%')
ax3.fill_between(N_wide, 1e-6, 1e-4, alpha=0.2, color='green',
                 label='High precision zone')
ax3.set_xlabel('N (log scale)')
ax3.set_ylabel('Relative Error (%)')
ax3.set_title('Error Evolution with N')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Градиент ошибки (скорость сходимости)
ax4 = plt.subplot(2, 2, 4)
gradient = np.gradient(np.log10(errors_wide), np.log10(N_wide))

ax4.semilogx(N_wide[1:], gradient[1:], 'green', linewidth=1.5)
ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax4.axvline(x=N_special, color='red', linestyle=':', linewidth=2, alpha=0.7,
           label=f'N = 4.197e121 (gradient ≈ 0)')
ax4.set_xlabel('N (log scale)')
ax4.set_ylabel('d(log₁₀ error)/d(log₁₀ N)')
ax4.set_title('Convergence Rate (Gradient)')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Дополнительный анализ: чувствительность к K
fig2, (ax5, ax6) = plt.subplots(1, 2, figsize=(16, 6))

# Варьируем K вокруг 6
K_values = np.linspace(5, 7, 1000)
N_for_K = 4.197e121  # фиксируем N

values_vs_K = np.array([formula(k, N_for_K) for k in K_values])
errors_vs_K = np.abs(values_vs_K - FEIGENBAUM_DELTA) / FEIGENBAUM_DELTA * 100

ax5.plot(K_values, values_vs_K, 'b-', linewidth=2, label='Formula value')
ax5.axhline(y=FEIGENBAUM_DELTA, color='r', linestyle='--', linewidth=2, label=f'δ')
ax5.axvline(x=6, color='orange', linestyle=':', linewidth=2, alpha=0.7, label='K=6')
ax5.set_xlabel('K')
ax5.set_ylabel('Value')
ax5.set_title('Sensitivity to K (at N = 4.197e121)')
ax5.legend()
ax5.grid(True, alpha=0.3)

ax6.semilogy(K_values, errors_vs_K, 'purple', linewidth=2)
ax6.axvline(x=6, color='orange', linestyle=':', linewidth=2, alpha=0.7, label='K=6')
ax6.scatter([6], [errors_vs_K[np.argmin(np.abs(K_values - 6))]],
           color='red', s=150, zorder=5, label=f'Error at K=6')
ax6.set_xlabel('K')
ax6.set_ylabel('Relative Error (%)')
ax6.set_title('Error Sensitivity to K')
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Финальный анализ и выводы
print("\n" + "="*70)
print("DETAILED ANALYSIS")
print("="*70)
print(f"\nOPTIMAL POINT ANALYSIS:")
print(f"  Your N = {N_special:.4e}")
print(f"  Actual minimum error occurs at N ≈ {N_min_error:.4e}")
print(f"  Minimum possible error = {min_error_value:.10f}%")
print(f"  Match quality: {min_error_value/0.0000370947*100:.2f}% of your reported error")
print(f"\nFORMULA BEHAVIOR:")
print(f"  At N=10:               {formula(K, 10):.10f} (error: {errors_wide[0]:.4f}%)")
print(f"  At optimal N:          {values_zoom[min_error_idx]:.10f} (error: {min_error_value:.8f}%)")
print(f"  At N→∞:                {K*K/(K+1/np.log(np.log(K))):.10f} (error: 0.0594%)")
print(f"  At N=e^300:            {formula(K, np.exp(300)):.10f} (error: 0.0039%)")
print(f"\nSIGNIFICANCE:")
print(f"  Improvement over asymptotic: {0.059441/min_error_value:.0f}x better at optimal point")
print(f"  Error reduction from N=10:   {errors_wide[0]/min_error_value:.0f}x improvement")
print(f"\nMATHEMATICAL INSIGHT:")
print(f"  The term 1/ln(N) at optimal N = {1/np.log(N_special):.2e}")
print(f"  The term 1/ln(ln(K)) = {1/np.log(np.log(K)):.6f}")
print(f"  K*(K+1/lnN)/(K+1/ln(lnK)) balances when 1/lnN ≈ specific value")