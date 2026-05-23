import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


def T_CMB_from_ln(lnN):
    lnN = np.asarray(lnN, dtype=float)

    mask = (lnN > 0) & np.isfinite(lnN)
    result = np.full_like(lnN, np.nan, dtype=float)

    valid_lnN = lnN[mask]

    # 🔥 НЕТ exp — только lnN
    numerator = np.sqrt(2) * (valid_lnN ** 17)
    denominator = 6 * np.exp(valid_lnN / 3)

    result[mask] = numerator / denominator
    return result if result.size > 1 else float(result)

def cosmic_time_correct(lnN):
    lnN = np.asarray(lnN, dtype=float)

    mask = (lnN > 0.1) & np.isfinite(lnN)
    result = np.full_like(lnN, np.nan, dtype=float)

    valid_lnN = lnN[mask]

    # 🔥 считаем через ln
    log_t_raw = (2/3) * valid_lnN - 2 * np.log(valid_lnN)

    # нормировка
    lnN_current = 280.047
    log_t_current = (2/3) * lnN_current - 2 * np.log(lnN_current)

    # перевод в реальные годы
    t_norm = 13.8e9

    log_t = log_t_raw - log_t_current + np.log(t_norm)

    result[mask] = np.exp(log_t)
    return result if result.size > 1 else float(result)


# ПАРАМЕТРЫ

lnN_current = 280.047
t_current = cosmic_time_correct(lnN_current)
T_current = T_CMB_from_ln(lnN_current)

# Диапазон lnN — начинаем с 3.1 вместо 3.0 для избежания сингулярности
lnN_values = np.linspace(3.1, 350.0, 3000)
t_values = cosmic_time_correct(lnN_values)
T_values = T_CMB_from_ln(lnN_values)

# Убираем NaN
valid = ~np.isnan(t_values) & ~np.isnan(T_values) & np.isfinite(t_values) & np.isfinite(T_values)
lnN_values_clean = lnN_values[valid]
t_values_clean = t_values[valid]
T_values_clean = T_values[valid]

# ГРАФИКИ
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('ТЕМПЕРАТУРА РЕЛИКТОВОГО ИЗЛУЧЕНИЯ В ЕТИ\n'
             r'$t \sim N^{2/3} / (\ln N)^2$',
             fontsize=16, fontweight='bold')

# --- График 1: T_CMB(t) — вся история ---
ax1 = axes[0, 0]
ax1.loglog(t_values_clean, T_values_clean, 'b-', linewidth=2, label=r'$T_{\rm CMB}(t)$')
ax1.axvline(x=t_current, color='red', linestyle='--', linewidth=1.5,
            label=f'Сейчас: t = {t_current / 1e9:.2f} млрд лет')
ax1.axhline(y=T_current, color='green', linestyle=':', linewidth=1.5,
            label=f'T = {T_current:.2f} K')
ax1.set_xlabel('Космическое время t (лет)', fontsize=12)
ax1.set_ylabel('T_CMB (K)', fontsize=12)
ax1.set_title('Полная история Вселенной', fontsize=13)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=9)

# --- График 2: T_CMB(t) — современная эпоха ---
ax2 = axes[0, 1]
t_mask = (t_values_clean > 1e9) & (t_values_clean < 2e10)
t_modern = t_values_clean[t_mask]
T_modern = T_values_clean[t_mask]
ax2.plot(t_modern / 1e9, T_modern, 'b-', linewidth=2)
ax2.axvline(x=t_current / 1e9, color='red', linestyle='--', linewidth=1.5)
ax2.axhline(y=2.725, color='orange', linestyle=':', linewidth=1.5, label='CODATA: 2.725 K')
ax2.set_xlabel('Космическое время t (млрд лет)', fontsize=12)
ax2.set_ylabel('T_CMB (K)', fontsize=12)
ax2.set_title('Современная эпоха', fontsize=13)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10)

# --- График 3: T_CMB(ln N) ---
ax3 = axes[0, 2]
ax3.semilogy(lnN_values_clean, T_values_clean, 'b-', linewidth=2)
ax3.axvline(x=lnN_current, color='red', linestyle='--', linewidth=1.5)
ax3.axhline(y=T_current, color='green', linestyle=':', linewidth=1.5)
ax3.set_xlabel(r'$\ln N$', fontsize=12)
ax3.set_ylabel('T_CMB (K)', fontsize=12)
ax3.set_title(r'Зависимость от $\ln N$', fontsize=13)
ax3.grid(True, alpha=0.3)

# --- График 4: Связь t и ln N ---
ax4 = axes[1, 0]
ax4.semilogy(lnN_values_clean, t_values_clean, 'g-', linewidth=2)
ax4.axvline(x=lnN_current, color='red', linestyle='--', linewidth=1.5)
ax4.axhline(y=t_current, color='red', linestyle='--', linewidth=1.5)
ax4.set_xlabel(r'$\ln N$', fontsize=12)
ax4.set_ylabel('Космическое время t (лет)', fontsize=12)
ax4.set_title(r'$t \sim N^{2/3} / (\ln N)^2$', fontsize=13)
ax4.grid(True, alpha=0.3)

# --- График 5: dT/dt ---
ax5 = axes[1, 1]
dT_dt = np.gradient(T_values_clean, t_values_clean)
ax5.loglog(t_values_clean, -dT_dt, 'r-', linewidth=2)
ax5.axvline(x=t_current, color='red', linestyle='--', linewidth=1.5)
ax5.set_xlabel('Космическое время t (лет)', fontsize=12)
ax5.set_ylabel('-dT/dt (K/год)', fontsize=12)
ax5.set_title('Скорость остывания', fontsize=13)
ax5.grid(True, alpha=0.3)

# --- График 6: d(ln T)/d(ln t) ---
ax6 = axes[1, 2]
dlnT_dlnt = np.gradient(np.log(T_values_clean), np.log(t_values_clean + 1e-10))
ax6.plot(lnN_values_clean, dlnT_dlnt, 'm-', linewidth=2, label=r'$d\ln T/d\ln t$')
ax6.axvline(x=lnN_current, color='red', linestyle='--', linewidth=1.5)
ax6.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax6.set_xlabel(r'$\ln N$', fontsize=12)
ax6.set_ylabel('Относительная скорость', fontsize=12)
ax6.set_title('Относительная скорость остывания', fontsize=13)
ax6.grid(True, alpha=0.3)
ax6.legend(fontsize=10)

plt.tight_layout()
plt.show()

# ТАБЛИЦА
print("ТЕМПЕРАТУРА РЕЛИКТОВОГО ИЗЛУЧЕНИЯ В ЕТИ")
print(f"\n  Формула T_CMB: (√2 · (ln N)^17) / (6 · N^(1/3))")
print(f"  Модель времени: t ~ N^(2/3) / (ln N)^2")
print(f"  Нормировка: t(сейчас) = 13.8 млрд лет")
print(f"\n  Текущее состояние:")
print(f"    ln N = {lnN_current:.3f}")
print(f"    t = {t_current:.4e} лет = {t_current / 1e9:.2f} млрд лет")
print(f"    T_CMB(теория) = {T_current:.6f} K")
print(f"    T_CMB(CODATA) = 2.72548 K")
print(f"    Ошибка = {abs(T_current - 2.72548) / 2.72548 * 100:.4f}%")
print()

print(f"  {'ln N':<10} {'t (млрд лет)':<18} {'T_CMB (K)':<18} {'Эпоха':<20}")
print(f"  {'-' * 70}")

epochs = [
    (5.0, "Инфляция"),
    (23.0, "Адронная эра"),
    (50.0, "Электрослабая эра"),
    (101.0, "Формирование галактик"),
    (200.0, "Современная эпоха (начало)"),
    (250.0, "Близкое прошлое"),
    (270.0, "Близкое прошлое"),
    (280.047, "НАСТОЯЩЕЕ"),
    (290.0, "Близкое будущее"),
    (300.0, "Далекое будущее"),
    (350.0, "Асимптотическое будущее"),
]

for lnN_val, epoch in epochs:
    t_val = cosmic_time_correct(lnN_val)
    T_val = T_CMB_from_ln(lnN_val)
    marker = " ← СЕЙЧАС" if abs(lnN_val - lnN_current) < 0.1 else ""
    if not np.isnan(t_val) and not np.isnan(T_val):
        print(f"  {lnN_val:<10.3f} {t_val / 1e9:<18.6f} {T_val:<18.6e} {epoch + marker:<20}")

# Производная в текущей точке
dlnT_dt_current = np.gradient(np.log(T_values_clean), t_values_clean)
idx_current = np.argmin(np.abs(lnN_values_clean - lnN_current))
print(f"\n  В текущей точке:")
print(f"    d(ln T)/dt ≈ {dlnT_dt_current[idx_current]:.6e} год⁻¹")
print(f"    dT/dt ≈ {dlnT_dt_current[idx_current] * T_current:.4e} K/год")
print(f"    Температура {'падает' if dlnT_dt_current[idx_current] < 0 else 'растёт'} со временем")