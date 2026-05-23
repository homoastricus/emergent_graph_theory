import numpy as np
import matplotlib.pyplot as plt

# =========================
# ДАННЫЕ ИЗ ТВОЕГО ЭКСПЕРИМЕНТА
# =========================
K = 6

N = np.array([216, 512, 1000, 1728, 2744])
lnN = np.log(N)

# =========================
# ТВОЯ ГИПОТЕЗА
# =========================
hbar_geom = (lnN**3) / (K * N**(1/3))

# =========================
# ДЛЯ СРАВНЕНИЯ: ФАКТИЧЕСКИЙ hbar_eff
# =========================
hbar_eff = np.array([
    1.5955e-01,
    1.5626e-01,
    1.5551e-01,
    1.4917e-01,
    1.5242e-01
])

# =========================
# НОРМИРОВКА (чтобы сравнивать формы)
# =========================
hbar_geom_norm = hbar_geom / hbar_geom[0]
hbar_eff_norm = hbar_eff / hbar_eff[0]

# =========================
# ФИТ СТЕПЕНИ ПО N
# =========================
logN = np.log(N)
logh = np.log(hbar_geom)

coeffs = np.polyfit(logN, logh, 1)
alpha = -coeffs[0]

# =========================
# ГРАФИКИ
# =========================
plt.figure(figsize=(12,5))

# --- 1. сравнение ---
plt.subplot(1,2,1)
plt.plot(N, hbar_geom_norm, 'o-', label='hbar_geom')
plt.plot(N, hbar_eff_norm, 's--', label='hbar_eff (measured)')
plt.xlabel("N")
plt.ylabel("normalized hbar")
plt.title("Comparison: geometry vs measurement")
plt.legend()
plt.grid(True)

# --- 2. scaling check ---
plt.subplot(1,2,2)
plt.loglog(N, hbar_geom, 'o-', label='hbar_geom')
plt.xlabel("N")
plt.ylabel("hbar_geom")
plt.title(f"Scaling check: slope ≈ {-alpha:.3f}")
plt.grid(True)

plt.tight_layout()
plt.show()

# =========================
# ЧИСЛОВОЙ ВЫВОД
# =========================
print("=== ANALYSIS ===")
print("hbar_geom:")
for n, h in zip(N, hbar_geom):
    print(f"N={n:5d} -> {h:.6e}")

print("\nEffective scaling exponent:")
print(f"alpha ≈ {alpha:.4f}")

print("\nRatio hbar_eff / hbar_geom:")
print(hbar_eff / hbar_geom)