"""
ЧИСЛЕННАЯ ПРОВЕРКА 35 LLL-ИНВАРИАНТОВ ЕТИ
"""

import math
import numpy as np

# ФУНДАМЕНТАЛЬНЫЕ ПАРАМЕТРЫ
K = 6.0
N_val = 4.1847e121
lnK = math.log(K)
lnN = math.log(N_val)
N13 = N_val ** (1.0 / 3.0)
pi = math.pi

# ВЫЧИСЛЕНИЕ ВСЕХ КОНСТАНТ
C = {}

# Квантовые
C['hbar'] = (lnN ** 3) / (K * N13)
C['h'] = 2 * pi * C['hbar']

# Скорость света
C['c'] = pi * (lnN ** 4) / (K ** 2 * lnK)

# Планковские
C['lP'] = 4 * lnN ** 2 * lnK / N13
C['tP'] = 4 * K ** 2 * lnK ** 2 / (pi * N13 * lnN ** 2)
C['EP'] = (lnN ** 5) * pi / (4 * K ** 3 * lnK ** 2)
C['mP'] = K / (pi * 4 * lnN ** 3)
C['TP'] = 8 * pi * N13 / (lnN ** 4)
C['G'] = 16 * pi ** 3 * lnN ** 13 / (K ** 5 * lnK * N13)
C['kB'] = (lnN ** 8) / (8 * pi ** 2 * N13)

# Безразмерные
C['alpha'] = 2 * lnK ** 2 / (pi * lnN)
C['m_proton_to_m_electron'] = lnN ** 2 / (4 * pi ** (1 / 2) * K)

# Электромагнитные
C['eps0'] = N13 / (8 * pi ** 3 * lnK * lnN ** 20)
C['mu0'] = (8 * pi * K ** 4 * lnK ** 3 * lnN ** 12) / N13
C['Z0'] = 8 * K ** 2 * pi ** 2 * lnK ** 2 * lnN ** 16 / N13

# Массы частиц
C['pion_pm'] = lnN ** 6 / (4 * pi ** 2 * math.sqrt(2) * N13)
C['pion0'] = 2 * pi * K ** 3 * lnN ** 4 / N13
C['m_Λ_barion'] = (lnN ** 6 * math.sqrt(2)) / (pi ** 2 * N13)
C['m_z_bozon'] = lnN ** 6 * 4 * pi ** (5 / 2) / (N13 * K)
C['m_w_bozon'] = 2 * pi ** 3 * lnN ** 6 / (N13 * K)
C['m_qu_s'] = lnN ** 4 * pi ** (7 / 2) / N13
C['m_qu_c'] = lnN ** 6 * 2 * pi ** 2 / (K ** 3 * N13)
C['m_qu_t'] = lnN ** 6 * K ** 3 / (pi ** 2 * N13)
C['m_neitrino'] = (lnN ** 2 * math.sqrt(2)) / (lnK * N13)

# Мезоны
C['rho_meson'] = math.sqrt(3) * pi ** (5 / 2) * lnN ** 5 / N13
C['omega_meson'] = lnN ** 5 * 2 * pi ** 2 * math.sqrt(2) / N13
C['eta_shtrih'] = lnN ** 5 * K ** 3 / (2 * pi * N13)
C['m_J_ψ'] = lnN ** 5 * 8 * pi ** 2 * math.sqrt(2) / N13
C['m_eta'] = lnN ** 5 * 2 * pi ** 2 / N13

# Барионы
C['Sigma_plus'] = K * lnN ** 6 / (4 * pi ** 2 * N13)
C['Sigma_minus'] = K * lnN ** 6 / (4 * pi ** 2 * N13)
C['Ksi_plus'] = lnN ** 6 / (pi * N13)

# Времена жизни
C['D0_lifetime'] = lnK / (2 * pi ** 2 * K ** 2 * lnN ** 4)
C['Λ_b_lifetime'] = lnK * math.sqrt(2) / lnN ** 5
C['B_plus_lifetime'] = lnK * pi / (2 * lnN ** 5)
C['tau_lifetime'] = 1 / (2 * lnN ** 5)
C['pion_lifetime'] = K ** 2 * math.sqrt(2) * pi / lnN ** 4

# Гравитация и космология
C['kappa'] = 128 * K ** 3 * lnK ** 3 / (lnN ** 3 * N13)
C['vH'] = lnN ** 6 * 8 * pi ** (3 / 2) / (math.sqrt(2) * N13)
C['Lambda_cosmo'] = lnN ** 12 / (pi ** (1 / 2) * N_val ** (2 / 3))
C['Lambda'] = C['Lambda_cosmo']  # синоним

# ОПРЕДЕЛЕНИЕ ТОЖДЕСТВ
identities = []

# #4 (‖v‖²=7): G × mP^2 = hbar × c
identities.append(("#4",
                   lambda C: C['G'] * C['mP'] ** 2,
                   lambda C: C['hbar'] * C['c']))

# #5 (‖v‖²=11): pion_pm^2 = hbar × mP × TP × kB^2
identities.append(("#5",
                   lambda C: C['pion_pm'] ** 2,
                   lambda C: C['hbar'] * C['mP'] * C['TP'] * C['kB'] ** 2))

# #6 (‖v‖²=12): c × EP × kappa = h^2 × TP^2 × kB
identities.append(("#6",
                   lambda C: C['c'] * C['EP'] * C['kappa'],
                   lambda C: C['h'] ** 2 * C['TP'] ** 2 * C['kB']))

# #7 (‖v‖²=17): EP^2 × mP × kappa^2 × rho_meson = lP × tP × G × pion_lifetime^2
identities.append(("#7",
                   lambda C: C['EP'] ** 2 * C['mP'] * C['kappa'] ** 2 * C['rho_meson'],
                   lambda C: C['lP'] * C['tP'] * C['G'] * C['pion_lifetime'] ** 2))

# #8 (‖v‖²=21): tP × m_Λ_barion^2 × rho_meson^2 = h × G × mP × kB^2 × kappa × pion_lifetime^2
identities.append(("#8",
                   lambda C: C['tP'] * C['m_Λ_barion'] ** 2 * C['rho_meson'] ** 2,
                   lambda C: C['h'] * C['G'] * C['mP'] * C['kB'] ** 2 * C['kappa'] * C['pion_lifetime'] ** 2))

# #9 (‖v‖²=23): h × m_Λ_barion^2 × rho_meson = hbar × lP × G × mP × TP^2 × kB × pion_pm^2 × pion_lifetime^2
identities.append(("#9",
                   lambda C: C['h'] * C['m_Λ_barion'] ** 2 * C['rho_meson'],
                   lambda C: C['hbar'] * C['lP'] * C['G'] * C['mP'] * C['TP'] ** 2 * C['kB'] * C['pion_pm'] ** 2 * C[
                       'pion_lifetime'] ** 2))

# #10 (‖v‖²=23): hbar × h^2 × tP × mP = m_Λ_barion^2 × D0_lifetime^2 × pion_lifetime^2 × rho_meson^2
identities.append(("#10",
                   lambda C: C['hbar'] * C['h'] ** 2 * C['tP'] * C['mP'],
                   lambda C: C['m_Λ_barion'] ** 2 * C['D0_lifetime'] ** 2 * C['pion_lifetime'] ** 2 * C[
                       'rho_meson'] ** 2))

# #11 (‖v‖²=24): vH^4 × D0_lifetime^2 = hbar × lP × tP × G
identities.append(("#11",
                   lambda C: C['vH'] ** 4 * C['D0_lifetime'] ** 2,
                   lambda C: C['hbar'] * C['lP'] * C['tP'] * C['G']))

# #12 (‖v‖²=26): Ksi_plus^4 = TP × pion_pm^2 × m_Λ_barion^2 × rho_meson
identities.append(("#12",
                   lambda C: C['Ksi_plus'] ** 4,
                   lambda C: C['TP'] * C['pion_pm'] ** 2 * C['m_Λ_barion'] ** 2 * C['rho_meson']))

# #13 (‖v‖²=34): TP × Lambda^4 × D0_lifetime^2 × pion_lifetime^2 = hbar × h × tP × kB × m_Λ_barion^2 × rho_meson
identities.append(("#13",
                   lambda C: C['TP'] * C['Lambda'] ** 4 * C['D0_lifetime'] ** 2 * C['pion_lifetime'] ** 2,
                   lambda C: C['hbar'] * C['h'] * C['tP'] * C['kB'] * C['m_Λ_barion'] ** 2 * C['rho_meson']))

# #14 (‖v‖²=35): EP × G × eps0^4 × Z0^4 = lP
identities.append(("#14",
                   lambda C: C['EP'] * C['G'] * C['eps0'] ** 4 * C['Z0'] ** 4,
                   lambda C: C['lP']))

# #15 (‖v‖²=36): c × lP × tP × G × D0_lifetime^2 × eta_shtrih^4 = h × mP × kB^2 × kappa × m_Λ_barion^2 × rho_meson
identities.append(("#15",
                   lambda C: C['c'] * C['lP'] * C['tP'] * C['G'] * C['D0_lifetime'] ** 2 * C['eta_shtrih'] ** 4,
                   lambda C: C['h'] * C['mP'] * C['kB'] ** 2 * C['kappa'] * C['m_Λ_barion'] ** 2 * C['rho_meson']))

# #16 (‖v‖²=40): c × EP^2 × G × eps0^4 × mu0^4 = tP × mP
identities.append(("#16",
                   lambda C: C['c'] * C['EP'] ** 2 * C['G'] * C['eps0'] ** 4 * C['mu0'] ** 4,
                   lambda C: C['tP'] * C['mP']))

# #17 (‖v‖²=41): lP × TP × m_qu_s^4 × m_Λ_barion^2 = EP × kB × kappa × rho_meson^4
identities.append(("#17",
                   lambda C: C['lP'] * C['TP'] * C['m_qu_s'] ** 4 * C['m_Λ_barion'] ** 2,
                   lambda C: C['EP'] * C['kB'] * C['kappa'] * C['rho_meson'] ** 4))

# #18 (‖v‖²=60): lP × tP × G × TP × kB^2 × m_neitrino^4 = h^3 × c × EP^2 × mP × kappa × m_Λ_barion^4 × pion_lifetime^2
identities.append(("#18",
                   lambda C: C['lP'] * C['tP'] * C['G'] * C['TP'] * C['kB'] ** 2 * C['m_neitrino'] ** 4,
                   lambda C: C['h'] ** 3 * C['c'] * C['EP'] ** 2 * C['mP'] * C['kappa'] * C['m_Λ_barion'] ** 4 * C[
                       'pion_lifetime'] ** 2))

# #19 (‖v‖²=85): EP × TP × D0_lifetime^2 × omega_meson^8 = h^2 × lP × G × rho_meson^3
identities.append(("#19",
                   lambda C: C['EP'] * C['TP'] * C['D0_lifetime'] ** 2 * C['omega_meson'] ** 8,
                   lambda C: C['h'] ** 2 * C['lP'] * C['G'] * C['rho_meson'] ** 3))

# #20 (‖v‖²=90): TP × m_Λ_barion^2 × m_eta^8 × D0_lifetime^2 = hbar × h × G × pion_pm^2 × kappa × rho_meson^3
identities.append(("#20",
                   lambda C: C['TP'] * C['m_Λ_barion'] ** 2 * C['m_eta'] ** 8 * C['D0_lifetime'] ** 2,
                   lambda C: C['hbar'] * C['h'] * C['G'] * C['pion_pm'] ** 2 * C['kappa'] * C['rho_meson'] ** 3))

# #21 (‖v‖²=90): G × TP × D0_lifetime^2 × Sigma_minus^8 × rho_meson = h × lP × kB^3 × pion_pm^2 × m_Λ_barion^2
identities.append(("#21",
                   lambda C: C['G'] * C['TP'] * C['D0_lifetime'] ** 2 * C['Sigma_minus'] ** 8 * C['rho_meson'],
                   lambda C: C['h'] * C['lP'] * C['kB'] ** 3 * C['pion_pm'] ** 2 * C['m_Λ_barion'] ** 2))

# #22 (‖v‖²=90): G × TP × D0_lifetime^2 × Sigma_plus^8 × rho_meson = h × lP × kB^3 × pion_pm^2 × m_Λ_barion^2
identities.append(("#22",
                   lambda C: C['G'] * C['TP'] * C['D0_lifetime'] ** 2 * C['Sigma_plus'] ** 8 * C['rho_meson'],
                   lambda C: C['h'] * C['lP'] * C['kB'] ** 3 * C['pion_pm'] ** 2 * C['m_Λ_barion'] ** 2))

# #23 (‖v‖²=93): m_J_ψ^8 × D0_lifetime^2 = hbar × h × lP × G^2 × mP × TP^2 × kappa × m_Λ_barion^2 × pion_lifetime^2 × rho_meson^2
identities.append(("#23",
                   lambda C: C['m_J_ψ'] ** 8 * C['D0_lifetime'] ** 2,
                   lambda C: C['hbar'] * C['h'] * C['lP'] * C['G'] ** 2 * C['mP'] * C['TP'] ** 2 * C['kappa'] * C[
                       'm_Λ_barion'] ** 2 * C['pion_lifetime'] ** 2 * C['rho_meson'] ** 2))

# #24 (‖v‖²=93): EP × m_Λ_barion^2 × Λ_b_lifetime^8 = hbar × tP × TP × D0_lifetime^4 × pion_lifetime^2 × rho_meson
identities.append(("#24",
                   lambda C: C['EP'] * C['m_Λ_barion'] ** 2 * C['Λ_b_lifetime'] ** 8,
                   lambda C: C['hbar'] * C['tP'] * C['TP'] * C['D0_lifetime'] ** 4 * C['pion_lifetime'] ** 2 * C[
                       'rho_meson']))

# #25 (‖v‖²=94): hbar × h × c × mP × kappa × m_proton_to_m_electron^8 × m_Λ_barion^2 × pion_lifetime^2 = TP × kB^2 × pion_pm^2 × D0_lifetime^2 × rho_meson^2
identities.append(("#25",
                   lambda C: C['hbar'] * C['h'] * C['c'] * C['mP'] * C['kappa'] * C['m_proton_to_m_electron'] ** 8 * C[
                       'm_Λ_barion'] ** 2 * C['pion_lifetime'] ** 2,
                   lambda C: C['TP'] * C['kB'] ** 2 * C['pion_pm'] ** 2 * C['D0_lifetime'] ** 2 * C['rho_meson'] ** 2))

# #26 (‖v‖²=95): G × TP × kB^3 × kappa × tau_lifetime^8 = hbar × h^2 × tP × EP × mP^2 × D0_lifetime^2 × pion_lifetime^2
identities.append(("#26",
                   lambda C: C['G'] * C['TP'] * C['kB'] ** 3 * C['kappa'] * C['tau_lifetime'] ** 8,
                   lambda C: C['hbar'] * C['h'] ** 2 * C['tP'] * C['EP'] * C['mP'] ** 2 * C['D0_lifetime'] ** 2 * C[
                       'pion_lifetime'] ** 2))

# #27 (‖v‖²=95): TP × m_Λ_barion^2 × m_z_bozon^8 × D0_lifetime^2 × pion_lifetime^2 = hbar × h × lP × G^2 × mP × kappa × rho_meson^3
identities.append(("#27",
                   lambda C: C['TP'] * C['m_Λ_barion'] ** 2 * C['m_z_bozon'] ** 8 * C['D0_lifetime'] ** 2 * C[
                       'pion_lifetime'] ** 2,
                   lambda C: C['hbar'] * C['h'] * C['lP'] * C['G'] ** 2 * C['mP'] * C['kappa'] * C['rho_meson'] ** 3))

# #28 (‖v‖²=96): tP × mP × m_Λ_barion^2 × m_w_bozon^8 = h × c × lP × G × kB^2 × kappa × m_qu_s^4 × rho_meson
identities.append(("#28",
                   lambda C: C['tP'] * C['mP'] * C['m_Λ_barion'] ** 2 * C['m_w_bozon'] ** 8,
                   lambda C: C['h'] * C['c'] * C['lP'] * C['G'] * C['kB'] ** 2 * C['kappa'] * C['m_qu_s'] ** 4 * C[
                       'rho_meson']))

# #29 (‖v‖²=97): h × kB^2 × pion_pm^2 × kappa × pion_lifetime^2 × rho_meson = c × TP × tau^8 × D0_lifetime^4
identities.append(("#29",
                   lambda C: C['h'] * C['kB'] ** 2 * C['pion_pm'] ** 2 * C['kappa'] * C['pion_lifetime'] ** 2 * C[
                       'rho_meson'],
                   lambda C: C['c'] * C['TP'] * C['tau_lifetime'] ** 8 * C['D0_lifetime'] ** 4))

# #30 (‖v‖²=101): G × kB^4 × pion_pm^2 × kappa^3 × rho_meson^2 = hbar^2 × h × EP^2 × mP × TP × mu0^4 × m_Λ_barion^2 × D0_lifetime^4 × pion_lifetime^2 × m_neitrino^4
identities.append(("#30",
                   lambda C: C['G'] * C['kB'] ** 4 * C['pion_pm'] ** 2 * C['kappa'] ** 3 * C['rho_meson'] ** 2,
                   lambda C: C['hbar'] ** 2 * C['h'] * C['EP'] ** 2 * C['mP'] * C['TP'] * C['mu0'] ** 4 * C[
                       'm_Λ_barion'] ** 2 * C['D0_lifetime'] ** 4 * C['pion_lifetime'] ** 2 * C['m_neitrino'] ** 4))

# #31 (‖v‖²=105): mP × TP^2 × pion_pm^2 × m_qu_c^8 × m_Λ_barion^2 × pion_lifetime^2 = hbar × h × c × lP × tP × EP × G^2 × kB × D0_lifetime^2 × rho_meson^3
identities.append(("#31",
                   lambda C: C['mP'] * C['TP'] ** 2 * C['pion_pm'] ** 2 * C['m_qu_c'] ** 8 * C['m_Λ_barion'] ** 2 * C[
                       'pion_lifetime'] ** 2,
                   lambda C: C['hbar'] * C['h'] * C['c'] * C['lP'] * C['tP'] * C['EP'] * C['G'] ** 2 * C['kB'] * C[
                       'D0_lifetime'] ** 2 * C['rho_meson'] ** 3))

# #32 (‖v‖²=106): c × TP × kB × pion0^8 × D0_lifetime^4 = h^2 × mP × kappa × eta_shtrih^4 × rho_meson
identities.append(("#32",
                   lambda C: C['c'] * C['TP'] * C['kB'] * C['pion0'] ** 8 * C['D0_lifetime'] ** 4,
                   lambda C: C['h'] ** 2 * C['mP'] * C['kappa'] * C['eta_shtrih'] ** 4 * C['rho_meson']))

# #33 (‖v‖²=107): c × EP^2 × alpha^8 × m_Λ_barion^2 × pion_lifetime^2 × m_neitrino^4 = lP × tP × G × TP × kB × pion_pm^2 × kappa × D0_lifetime^2
identities.append(("#33",
                   lambda C: C['c'] * C['EP'] ** 2 * C['alpha'] ** 8 * C['m_Λ_barion'] ** 2 * C['pion_lifetime'] ** 2 *
                             C['m_neitrino'] ** 4,
                   lambda C: C['lP'] * C['tP'] * C['G'] * C['TP'] * C['kB'] * C['pion_pm'] ** 2 * C['kappa'] * C[
                       'D0_lifetime'] ** 2))

# #34 (‖v‖²=119): hbar × EP × G^2 × pion_pm^2 × kappa × m_Λ_barion^2 × B_plus_lifetime^8 × pion_lifetime^2 = h × lP × tP × m_qu_s^4 × D0_lifetime^4 × rho_meson
identities.append(("#34",
                   lambda C: C['hbar'] * C['EP'] * C['G'] ** 2 * C['pion_pm'] ** 2 * C['kappa'] * C['m_Λ_barion'] ** 2 *
                             C['B_plus_lifetime'] ** 8 * C['pion_lifetime'] ** 2,
                   lambda C: C['h'] * C['lP'] * C['tP'] * C['m_qu_s'] ** 4 * C['D0_lifetime'] ** 4 * C['rho_meson']))

# #35 (‖v‖²=125): EP × TP × m_qu_t^8 × D0_lifetime^4 × rho_meson^3 = hbar × tP × m_Λ_barion^4 × eta_shtrih^4
identities.append(("#35",
                   lambda C: C['EP'] * C['TP'] * C['m_qu_t'] ** 8 * C['D0_lifetime'] ** 4 * C['rho_meson'] ** 3,
                   lambda C: C['hbar'] * C['tP'] * C['m_Λ_barion'] ** 4 * C['eta_shtrih'] ** 4))

# ПРОВЕРКА И ВЫВОД
print("ЧИСЛЕННАЯ ПРОВЕРКА 35 LLL-ИНВАРИАНТОВ ЕТИ")
print(f"  N = {N_val:.4e}")
print(f"  ln N = {lnN:.6f}")
print(f"  K = {K}")
print()

print(
    f"{'ID':<5} {'Норма':<7} {'Левая часть':<18} {'Правая часть':<18} {'Отношение':<14} {'Отклонение':<14} {'Статус':<12}")
print("-" * 100)

results = []
for id_str, left_fn, right_fn in identities:
    left_val = left_fn(C)
    right_val = right_fn(C)

    if right_val != 0:
        ratio = left_val / right_val
    else:
        ratio = float('inf')

    deviation_pct = (ratio - 1.0) * 100

    # Норма из названия
    norm = id_str.split('(')[1].split('‖')[1].split('=')[1].rstrip(')') if '‖' in id_str else '?'

    # Статус
    if abs(deviation_pct) < 0.01:
        status = "✅ ТОЧНОЕ"
    elif abs(deviation_pct) < 0.3:
        status = "🟡 NLO"
    elif abs(deviation_pct) < 1.0:
        status = "🟠 ~1%"
    elif abs(deviation_pct) < 10:
        status = "🔴 >1%"
    else:
        status = "❌ НЕТ"

    results.append((id_str, norm, left_val, right_val, ratio, deviation_pct, status))

    print(
        f"{id_str:<5} {norm:<7} {left_val:<18.6e} {right_val:<18.6e} {ratio:<14.6f} {deviation_pct:<+14.6f}% {status:<12}")

print("-" * 100)

# Статистика
exact = sum(1 for r in results if abs(r[5]) < 0.01)
nlo = sum(1 for r in results if 0.01 <= abs(r[5]) < 0.3)
pct1 = sum(1 for r in results if 0.3 <= abs(r[5]) < 1.0)
bad = sum(1 for r in results if abs(r[5]) >= 1.0)

print(f"\nСТАТИСТИКА:")
print(f"  ✅ Точные (<0.01%):     {exact}")
print(f"  🟡 NLO (0.01-0.3%):     {nlo}")
print(f"  🟠 ~1% (0.3-1.0%):      {pct1}")
print(f"  🔴❌ >1%:                {bad}")
print(f"  Всего:                  {len(results)}")
print()

# Топ-5 лучших
print("Топ-5 лучших:")
best = sorted(results, key=lambda x: abs(x[5]))[:5]
for r in best:
    print(f"  {r[0]:<5} отклонение = {r[5]:+.8f}%")

# Топ-5 худших
print("\nТоп-5 худших:")
worst = sorted(results, key=lambda x: abs(x[5]), reverse=True)[:5]
for r in worst:
    print(f"  {r[0]:<5} отклонение = {r[5]:+.4f}%")

