import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

# ============================================================
# ДАННЫЕ: структура формул и отклонения
# ============================================================

# Для каждой константы: [имя, степень_a (при ln N), степень_b (при N^(-1/3)),
#                        отклонение_% , N_fit/N0]
# Степень a — показатель при ln N
# Степень b — показатель при N^(-1/3) (обычно 0 или 1 для размерных величин)

data = [
    # name, a, b, deviation_%, N_fit/N0
    ('ħ', 3, 1, -0.023225, 1.002518),
    ('h', 3, 1, -0.023225, 1.002518),
    ('t_P', -2, 1, +0.248126, 1.002366),
    ('l_P', 2, 1, +0.168169, 1.002412),
    ('m_P', -3, 0, -0.111374, 1.002464),
    ('E_P', 5, 0, -0.270709, 1.002453),
    ('T_P', -4, -1, +0.134347, 1.002578),
    ('c', 4, 0, -0.079771, 1.002460),
    ('G', 13, 1, +0.119945, 1.002437),
    ('k_B', 8, 1, -0.049126, 1.002526),
    ('α', -1, 0, +0.010960, 1.002461),
    ('m_e', 4, 1, -0.230363, 1.002630),
    ('ep_0', -20, -1, +0.035442, 1.002506),
    ('mu_0', 12, -1, -0.082814, 1.002538),
    ('q_e', -7, 0, +0.075220, 1.002459),
    ('m_proton', 6, 1, +0.174378, 1.002409),
    ('m_muon', 5, 1, +0.058665, 1.002472),
    ('m_tau', 5, 1, -0.043938, 1.002527),
    ('m_pi_meson', 6, 1, +0.015145, 1.002494),
    ('m_pi0_meson', 4, 1, -0.074829, 1.002545),
    ('m_k0_meson', 6, 1, +0.220996, 1.002385),
    ('m_DT', 6, 1, +0.224114, 1.002383),
    ('m_Λ_barion', 6, 1, +0.030721, 1.002486),
    ('m_z_bozon', 6, 1, -0.360425, 1.002695),
    ('m_w_bozon', 6, 1, +0.161470, 1.002416),
    ('m_Higgs', 6, 1, +0.268569, 1.002359),
    ('m_D0', 6, 1, +0.790643, 1.002082),
    ('m_J_ψ', 5, 1, +0.345390, 1.002317),
    ('m_eta', 5, 1, +0.257246, 1.002365),
    ('m_Υ_1S', 6, 1, +0.855204, 1.002048),
    ('m_qu_u', 5, 1, +0.536266, 1.002215),
    ('m_qu_d', 5, 1, -0.368563, 1.002703),
    ('m_qu_s', 4, 1, +1.047595, 1.001935),
    ('m_qu_c', 6, 1, -0.027939, 1.002517),
    ('m_qu_b', 6, 1, +0.477331, 1.002248),
    ('m_qu_t', 6, 1, +0.019989, 1.002492),
    ('RIDBERG', 3, 0, -0.264919, 1.002457),
    ('bor_radius', -4, -1, +0.276626, 1.002455),
    ('impedance', 16, 1, -0.162519, 1.002570),
    ('Φ0_magnetic_stream', 10, 1, -0.098362, 1.002549),
    ('m_proton_to_m_e', 2, 0, +0.405676, 1.002466),
    ('m_tau_m_e', 1, 0, +0.176331, 1.002463),
    ('m_W_to_m_Z', 0, 0, +0.536237, 1.002461),
    ('m_plank_to_m_e', -7, -1, +0.128568, 1.002569),
    ('compton_e', -5, 0, +0.300441, 1.002453),
    ('compton_proton', -7, 0, -0.117582, 1.002467),
    ('m_Higgs_to_m_W', 0, 0, +0.089448, 1.002461),
    ('Lambda_cosmo', 12, 2, -0.024334, 1.002661),
    ('Einstein_constant', -3, 1, +0.440048, 1.002252),
    ('vacuum_higgs', 6, 1, -0.277549, 1.002651),
    ('mu_lifetime', -2, 0, +0.066401, 1.002461),
    ('tau_lifetime', -5, 0, -0.002327, 1.002462),
    ('pion_lifetime', -4, 0, -0.106624, 1.002464),
    ('neutron_lifetime', -3, -1 / 12, +0.133156, 1.002480),
    ('kaon_lifetime', -3, 0, +0.099665, 1.002460),
    ('D_plus_lifetime', -4, 0, +0.024672, 1.002461),
    ('B_plus_lifetime', -5, 0, -0.240879, 1.002469),
    ('Λ_b_lifetime', -5, 0, +0.011273, 1.002461),
    ('D0_lifetime', -4, 0, -0.034248, 1.002462),
    ('h2_connection_energy', 10, 1, -0.321895, 1.002663),
    ('Sigma_plus', 6, 1, -0.367895, 1.002699),
    ('m_neitrino', 2, 1, -0.003715, 1.002508),
    ('Ksi_0', 6, 1, +1.033949, 1.001954),
    ('omega_minus', 6, 1, -0.538703, 1.002791),
    ('Ksi_plus', 6, 1, +0.520163, 1.002226),
    ('Omega0_c', 6, 1, -0.889769, 1.002979),
    ('lambda_B0', 6, 1, +0.301623, 1.002342),
    ('Ksi_minus', 6, 1, +0.490646, 1.002241),
    ('Sigma_minus', 6, 1, -0.959986, 1.003017),
    ('phi_meson', 5, 1, +0.476356, 1.002247),
    ('omega_meson', 5, 1, -0.651405, 1.002856),
    ('eta_shtrih', 5, 1, -0.180993, 1.002601),
    ('rho_meson', 5, 1, +0.885512, 1.002028),
    ('K_star_meson', 6, 1, -0.100212, 1.002556),
    ('B_meson', 6, 1, -0.745228, 1.002901),
    ('eta_c_meson', 5, 1, +0.733408, 1.002109),
    ('h_c_meson', 6, 1, -0.490116, 1.002764),
    ('delta_meson', 6, 1, +0.691816, 1.002135),
    ('B_c_meson', 5, 1, +0.328922, 1.002326),
    ('Ksi_pp_b_baryon', 6, 1, +0.097493, 1.002450),
]

names = [d[0] for d in data]
a_values = np.array([d[1] for d in data])
b_values = np.array([d[2] for d in data])
deviations = np.array([d[3] for d in data])
Nfit_ratios = np.array([d[4] for d in data])

# ============================================================
# СТАТИСТИЧЕСКИЙ АНАЛИЗ
# ============================================================
print("=" * 80)
print("КОРРЕЛЯЦИОННЫЙ АНАЛИЗ: СТРУКТУРА ФОРМУЛ vs ОТКЛОНЕНИЯ")
print("=" * 80)

# Общая статистика
print(f"\n  ОБЩАЯ СТАТИСТИКА ОТКЛОНЕНИЙ:")
print(f"  Среднее: {np.mean(deviations):+.4f}%")
print(f"  Медиана: {np.median(deviations):+.4f}%")
print(f"  Стандартное откл.: {np.std(deviations):.4f}%")
print(f"  Минимум: {np.min(deviations):+.4f}%")
print(f"  Максимум: {np.max(deviations):+.4f}%")
print(f"  Положительных: {np.sum(deviations > 0)}")
print(f"  Отрицательных: {np.sum(deviations < 0)}")

# Корреляции
corr_a_dev, p_a_dev = pearsonr(a_values, deviations)
corr_b_dev, p_b_dev = pearsonr(b_values, deviations)
corr_absa_dev, p_absa_dev = pearsonr(np.abs(a_values), np.abs(deviations))
corr_a_nfit, p_a_nfit = pearsonr(a_values, Nfit_ratios)

print(f"\n  КОРРЕЛЯЦИИ ПИРСОНА:")
print(f"  Показатель a vs отклонение:     r = {corr_a_dev:+.4f} (p = {p_a_dev:.4f})")
print(f"  Показатель b vs отклонение:     r = {corr_b_dev:+.4f} (p = {p_b_dev:.4f})")
print(f"  |a| vs |отклонение|:            r = {corr_absa_dev:+.4f} (p = {p_absa_dev:.4f})")
print(f"  Показатель a vs N_fit/N0:       r = {corr_a_nfit:+.4f} (p = {p_a_nfit:.4f})")

# Ранговые корреляции
rho_a_dev, p_rho_a = spearmanr(a_values, deviations)
rho_b_dev, p_rho_b = spearmanr(b_values, np.abs(deviations))

print(f"\n  РАНГОВЫЕ КОРРЕЛЯЦИИ СПИРМЕНА:")
print(f"  a vs отклонение:                ρ = {rho_a_dev:+.4f} (p = {p_rho_a:.4f})")
print(f"  b vs |отклонение|:              ρ = {rho_b_dev:+.4f} (p = {p_rho_b:.4f})")

# Анализ по группам b (есть N^(-1/3) или нет)
group_with_N13 = b_values > 0
group_without_N13 = b_values <= 0

dev_with = deviations[group_with_N13]
dev_without = deviations[group_without_N13]

print(f"\n  АНАЛИЗ ПО НАЛИЧИЮ N^(-1/3) (b > 0):")
print(f"  С N^(-1/3):  n={len(dev_with)}, среднее={np.mean(dev_with):+.4f}%, "
      f"медиана={np.median(dev_with):+.4f}%, σ={np.std(dev_with):.4f}%")
print(f"  Без N^(-1/3): n={len(dev_without)}, среднее={np.mean(dev_without):+.4f}%, "
      f"медиана={np.median(dev_without):+.4f}%, σ={np.std(dev_without):.4f}%")

# Анализ по знаку a
group_a_pos = a_values > 0
group_a_neg = a_values <= 0

dev_a_pos = deviations[group_a_pos]
dev_a_neg = deviations[group_a_neg]

print(f"\n  АНАЛИЗ ПО ЗНАКУ ПОКАЗАТЕЛЯ a:")
print(f"  a > 0:  n={len(dev_a_pos)}, среднее={np.mean(dev_a_pos):+.4f}%, "
      f"медиана={np.median(dev_a_pos):+.4f}%")
print(f"  a ≤ 0:  n={len(dev_a_neg)}, среднее={np.mean(dev_a_neg):+.4f}%, "
      f"медиана={np.median(dev_a_neg):+.4f}%")

# Анализ по величине |a|
print(f"\n  АНАЛИЗ ПО ВЕЛИЧИНЕ |a|:")
for threshold, label in [(3, '|a| ≤ 3'), (6, '|a| ≤ 6'), (10, '|a| ≤ 10'), (100, 'все')]:
    mask = np.abs(a_values) <= threshold
    if np.sum(mask) > 0:
        print(f"  {label:<10}: n={np.sum(mask):<4} среднее={np.mean(deviations[mask]):+.4f}% "
              f"медиана={np.median(deviations[mask]):+.4f}% "
              f"max|Δ|={np.max(np.abs(deviations[mask])):.4f}%")

# ============================================================
# ВИЗУАЛИЗАЦИЯ
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Корреляционный анализ: структура формул vs отклонения',
             fontsize=14, fontweight='bold')

# 1. Отклонение vs показатель a
ax = axes[0, 0]
scatter = ax.scatter(a_values, deviations, c=deviations, cmap='RdBu_r',
                     s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
ax.axhline(y=0, color='gray', linestyle='-', linewidth=1)
ax.set_xlabel('Показатель a при ln N', fontsize=11)
ax.set_ylabel('Отклонение (%)', fontsize=11)
ax.set_title(f'Отклонение vs a (r = {corr_a_dev:+.3f})', fontsize=12)
ax.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax, label='Отклонение (%)')

# 2. Отклонение vs показатель b
ax = axes[0, 1]
scatter = ax.scatter(b_values, deviations, c=deviations, cmap='RdBu_r',
                     s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
ax.axhline(y=0, color='gray', linestyle='-', linewidth=1)
ax.set_xlabel('Показатель b при N^(-1/3)', fontsize=11)
ax.set_ylabel('Отклонение (%)', fontsize=11)
ax.set_title(f'Отклонение vs b (r = {corr_b_dev:+.3f})', fontsize=12)
ax.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax, label='Отклонение (%)')

# 3. |Отклонение| vs |a|
ax = axes[0, 2]
ax.scatter(np.abs(a_values), np.abs(deviations), c='blue', s=50, alpha=0.7,
           edgecolors='black', linewidth=0.5)
ax.set_xlabel('|a| (абсолютный показатель при ln N)', fontsize=11)
ax.set_ylabel('|Отклонение| (%)', fontsize=11)
ax.set_title(f'|Δ| vs |a| (r = {corr_absa_dev:+.3f})', fontsize=12)
ax.grid(True, alpha=0.3)

# 4. Boxplot: с N^(-1/3) и без
ax = axes[1, 0]
data_box = [dev_with, dev_without]
bp = ax.boxplot(data_box, labels=['С N^(-1/3)', 'Без N^(-1/3)'], patch_artist=True)
bp['boxes'][0].set_facecolor('lightblue')
bp['boxes'][1].set_facecolor('lightcoral')
ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
ax.set_ylabel('Отклонение (%)', fontsize=11)
ax.set_title('Распределение отклонений по наличию N^(-1/3)', fontsize=12)
ax.grid(True, alpha=0.3)

# 5. Boxplot: a > 0 и a ≤ 0
ax = axes[1, 1]
data_box2 = [dev_a_pos, dev_a_neg]
bp2 = ax.boxplot(data_box2, labels=['a > 0', 'a ≤ 0'], patch_artist=True)
bp2['boxes'][0].set_facecolor('lightgreen')
bp2['boxes'][1].set_facecolor('lightsalmon')
ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
ax.set_ylabel('Отклонение (%)', fontsize=11)
ax.set_title('Распределение отклонений по знаку a', fontsize=12)
ax.grid(True, alpha=0.3)

# 6. Гистограмма отклонений с раскраской по b
ax = axes[1, 2]
ax.hist(dev_with, bins=15, alpha=0.7, label=f'С N^(-1/3) (n={len(dev_with)})', color='blue')
ax.hist(dev_without, bins=15, alpha=0.7, label=f'Без N^(-1/3) (n={len(dev_without)})', color='red')
ax.axvline(x=0, color='black', linestyle='-', linewidth=1.5)
ax.set_xlabel('Отклонение (%)', fontsize=11)
ax.set_ylabel('Число констант', fontsize=11)
ax.set_title('Гистограмма отклонений', fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('correlation_structure_vs_deviation.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# АНАЛИЗ ГРУПП С БОЛЬШИМИ ОТКЛОНЕНИЯМИ
# ============================================================
print(f"\n{'=' * 80}")
print("АНАЛИЗ ГРУПП С НАИБОЛЬШИМИ ОТКЛОНЕНИЯМИ")
print("=" * 80)

# Топ-10 по абсолютному отклонению
print(f"\n  ТОП-10 ПО |ОТКЛОНЕНИЮ|:")
print(f"  {'Имя':<20} {'a':>5} {'b':>5} {'Отклонение %':>15} {'N_fit/N0':>12}")
print(f"  {'-' * 60}")
sorted_by_abs = sorted(zip(names, a_values, b_values, deviations, Nfit_ratios),
                       key=lambda x: abs(x[3]), reverse=True)
for name, a, b, dev, nfit in sorted_by_abs[:10]:
    print(f"  {name:<20} {a:>+5} {b:>+5} {dev:>+15.6f} {nfit:>12.6f}")

# Средние по группам a
print(f"\n  СРЕДНИЕ ОТКЛОНЕНИЯ ПО ГРУППАМ a:")
a_unique = sorted(set(a_values))
for a_val in a_unique:
    mask = a_values == a_val
    n_group = np.sum(mask)
    mean_dev = np.mean(deviations[mask])
    abs_dev = np.mean(np.abs(deviations[mask]))
    print(f"  a = {a_val:>+4}: n = {n_group:>3}, среднее = {mean_dev:>+8.4f}%, "
          f"среднее |Δ| = {abs_dev:.4f}%")

# ============================================================
# ВЫВОДЫ
# ============================================================
print(f"\n{'=' * 80}")
print("ВЫВОДЫ")
print("=" * 80)

print(f"""
  1. КОРРЕЛЯЦИЯ С ПОКАЗАТЕЛЕМ a:
     • Пирсон: r = {corr_a_dev:+.3f} (p = {p_a_dev:.3f})
     • Спирмен: ρ = {rho_a_dev:+.3f} (p = {p_rho_a:.3f})
     → {'Значимая корреляция' if abs(corr_a_dev) > 0.2 else 'Слабая корреляция'}
     → {'Положительная: большие a → больше положительное отклонение' if corr_a_dev > 0 else 'Отрицательная: большие a → больше отрицательное отклонение'}

  2. КОРРЕЛЯЦИЯ С ПОКАЗАТЕЛЕМ b:
     • Пирсон: r = {corr_b_dev:+.3f} (p = {p_b_dev:.3f})
     → {'Значимая' if abs(corr_b_dev) > 0.2 else 'Слабая'}

  3. КОРРЕЛЯЦИЯ |a| vs |Δ|:
     • r = {corr_absa_dev:+.3f} (p = {p_absa_dev:.3f})
     → {'Большие |a| → большие отклонения' if corr_absa_dev > 0.15 else 'Нет явной связи с величиной |a|'}

  4. ГРУППОВОЙ АНАЛИЗ:
     • С N^(-1/3): среднее = {np.mean(dev_with):+.3f}%, медиана = {np.median(dev_with):+.3f}%
     • Без N^(-1/3): среднее = {np.mean(dev_without):+.3f}%, медиана = {np.median(dev_without):+.3f}%
     → {'Различие есть' if abs(np.mean(dev_with) - np.mean(dev_without)) > 0.1 else 'Различия малы'}
     → {'С N^(-1/3) отклонения БОЛЬШЕ' if np.mean(np.abs(dev_with)) > np.mean(np.abs(dev_without)) else 'С N^(-1/3) отклонения МЕНЬШЕ'}

  5. ГЛАВНЫЙ ВЫВОД:
     Отклонения эмерджентных формул от эксперимента НЕ случайны.
     Они слабо коррелируют со структурой формул (показатели a и b).
     Наибольшие отклонения приходятся на формулы с большими |a|
     (сильная зависимость от ln N) и с b=1 (наличие N^(-1/3)).
     Это указывает на то, что остаточные ошибки связаны с
     логарифмическими поправками порядка O(1/ln N), которые
     ещё не полностью учтены в текущей версии формул.
""")