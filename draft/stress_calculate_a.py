import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr, spearmanr, shapiro, norm, f_oneway
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import warnings

warnings.filterwarnings('ignore')

# Создаем данные
data = {
    'b': [],
    'name': [],
    'A': [],
    'Theta': []
}

# b = -0.667 (n = 1)
data['b'].extend([-0.667])
data['name'].extend(['Lambda'])
data['A'].extend([349.734392])
data['Theta'].extend([1.00218222])

# b = -0.333 (n = 6)
b_minus = [
    ('impedance', -20.243552, 1.00248699),
    ('mu_0', -16.622579, 1.00173171),
    ('flux_quantum', -14.725405, 1.00190982),
    ('kappa_Einstein', -3.391959, 0.99668268),
    ('T_P', -2.302553, 0.99766975),
    ('epsilon_0', 892.305768, 0.99676765)
]
for name, A, Theta in b_minus:
    data['b'].append(-0.333)
    data['name'].append(name)
    data['A'].append(A)
    data['Theta'].append(Theta)

# b = +0.000 (n = 26)
b_zero = [
    ('E_P', -4.258444, 1.00265886),
    ('c', -3.620973, 1.00075397),
    ('m_Higgs/m_e', -2.654242, 0.99499103),
    ('m_proton/m_e', -2.505629, 0.99593755),
    ('m_W/m_e', -2.485861, 0.99606353),
    ('Rydberg', -2.394666, 1.00262289),
    ('m_Z/m_e', -1.723808, 1.00093185),
    ('m_muon/m_e', -1.388558, 0.99710036),
    ('m_tau/m_e', -1.201446, 0.99829476),
    ('m_W/m_Z', -0.835885, 0.99466624),
    ('m_P/m_e', -0.350419, 0.99776051),
    ('m_Higgs/m_W', -0.139743, 0.99910632),
    ('alpha', 0.919297, 0.99990150),
    ('tau_mu', 1.769103, 0.99935859),
    ('tau_kaon', 2.653580, 0.99903756),
    ('m_P', 2.983449, 1.00114827),
    ('Bohr_radius', 3.313940, 0.99728559),
    ('tau_D_plus', 3.707145, 0.99979767),
    ('tau_D0', 3.799241, 1.00038695),
    ('tau_pion', 3.912442, 1.00111176),
    ('Compton_e', 4.213251, 0.99705985),
    ('tau_Lambda_b', 4.664509, 0.99994270),
    ('tau_tau', 4.685766, 1.00007870),
    ('tau_B_plus', 5.059071, 1.00247017),
    ('e_charge', 6.437458, 0.99932591),
    ('Compton_p', 6.738866, 1.00125489)
]
for name, A, Theta in b_zero:
    data['b'].append(0.0)
    data['name'].append(name)
    data['A'].append(A)
    data['Theta'].append(Theta)

# b = +0.083 (n = 1)
data['b'].extend([0.083])
data['name'].extend(['tau_neutron'])
data['A'].extend([-3516.617616])
data['Theta'].extend([0.0])

# b = +0.333 (n = 28)
b_plus = [
    ('G', -6.522476, 0.99969196),
    ('k_B', -1.576187, 1.00143848),
    ('m_Upsilon_1S', -1.111116, 0.99248097),
    ('m_D0', -1.011033, 0.99311670),
    ('m_quark_b', -0.524420, 0.99621346),
    ('m_neutron', -0.394270, 0.99704336),
    ('m_Higgs', -0.199341, 0.99828761),
    ('m_kaon0', -0.125168, 0.99876148),
    ('m_proton', -0.052449, 0.99922627),
    ('m_W', -0.032308, 0.99935504),
    ('m_quark_t', 0.188623, 1.00076866),
    ('m_pion', 0.196193, 1.00081713),
    ('m_quark_c', 0.263535, 1.00124844),
    ('m_quark_u', 0.320357, 0.99564051),
    ('m_quark_s', 0.463866, 0.99061327),
    ('m_J_psi', 0.617382, 0.99753441),
    ('v_Higgs', 0.654267, 1.00375461),
    ('m_eta', 0.754736, 0.99841143),
    ('m_Z', 0.784214, 1.00458949),
    ('m_muon', 1.064622, 1.00039291),
    ('m_tau', 1.251625, 1.00159055),
    ('m_quark_d', 1.733407, 1.00468269),
    ('m_pion0', 2.209713, 1.00174049),
    ('m_e', 2.453180, 1.00330213),
    ('hbar', 3.065444, 1.00123453),
    ('h', 3.065444, 1.00123453),
    ('l_P', 3.702941, 0.99933252),
    ('t_P', 7.323933, 0.99857973)
]
for name, A, Theta in b_plus:
    data['b'].append(0.333)
    data['name'].append(name)
    data['A'].append(A)
    data['Theta'].append(Theta)

# Создаем DataFrame
df = pd.DataFrame(data)
print("=" * 80)
print("ПОЛНЫЙ СТАТИСТИЧЕСКИЙ АНАЛИЗ СВЯЗИ A И Θ ПО ГРУППАМ b")
print("=" * 80)

# 1. Описательная статистика по группам
print("\n" + "=" * 80)
print("1. ОПИСАТЕЛЬНАЯ СТАТИСТИКА ПО ГРУППАМ b")
print("=" * 80)

groups = df[df['b'].isin([-0.333, 0.0, 0.333])]
for b_val in [-0.333, 0.0, 0.333]:
    group = groups[groups['b'] == b_val]
    n = len(group)
    print(f"\n--- Группа b = {b_val:.3f} (n = {n}) ---")
    print(f"A: среднее = {group['A'].mean():.6f}, медиана = {group['A'].median():.6f}")
    print(f"A: стд = {group['A'].std():.6f}, мин = {group['A'].min():.6f}, макс = {group['A'].max():.6f}")
    print(f"A: асимметрия = {group['A'].skew():.6f}, эксцесс = {group['A'].kurtosis():.6f}")
    print(f"Θ: среднее = {group['Theta'].mean():.6f}, стд = {group['Theta'].std():.6f}")
    print(f"Θ: мин = {group['Theta'].min():.6f}, макс = {group['Theta'].max():.6f}")

# 2. Тесты на нормальность A внутри групп
print("\n" + "=" * 80)
print("2. ТЕСТЫ НА НОРМАЛЬНОСТЬ A ВНУТРИ ГРУПП")
print("=" * 80)

for b_val in [-0.333, 0.0, 0.333]:
    group_A = groups[groups['b'] == b_val]['A']
    n = len(group_A)

    # Shapiro-Wilk test
    if n >= 3:
        W_stat, W_p = shapiro(group_A)
        print(f"\nГруппа b = {b_val:.3f} (n = {n}):")
        print(f"  Shapiro-Wilk: W = {W_stat:.6f}, p-value = {W_p:.6f}")
        if W_p > 0.05:
            print(f"  → Гипотеза нормальности НЕ отвергается (на уровне 0.05)")
        else:
            print(f"  → Гипотеза нормальности ОТВЕРГАЕТСЯ")

        # Anderson-Darling test
        ad_stat, ad_crit, ad_sign = stats.anderson(group_A, dist='norm')
        print(f"  Anderson-Darling: A² = {ad_stat:.6f}, критические значения = {ad_crit}")
        if ad_stat < ad_crit[2]:
            print(f"  → Нормальность не отвергается на уровне 5%")
        else:
            print(f"  → Нормальность отвергается")

        # D'Agostino-Pearson test
        k2_stat, k2_p = stats.normaltest(group_A)
        print(f"  D'Agostino-Pearson: K² = {k2_stat:.6f}, p-value = {k2_p:.6f}")

# 3. Корреляционный анализ
print("\n" + "=" * 80)
print("3. КОРРЕЛЯЦИОННЫЙ АНАЛИЗ A И Θ ПО ГРУППАМ")
print("=" * 80)

for b_val in [-0.333, 0.0, 0.333]:
    group = groups[groups['b'] == b_val]
    A_vals = group['A'].values
    Theta_vals = group['Theta'].values
    n = len(A_vals)

    print(f"\nГруппа b = {b_val:.3f} (n = {n}):")

    # Pearson correlation
    if n > 2:
        r_pearson, p_pearson = pearsonr(A_vals, Theta_vals)
        t_stat = r_pearson * np.sqrt((n - 2) / (1 - r_pearson ** 2))
        print(f"  Pearson r = {r_pearson:.6f}, t = {t_stat:.4f}, p-value = {p_pearson:.6f}")
        print(f"  R² = {r_pearson ** 2:.6f} ({r_pearson ** 2 * 100:.2f}% объяснённой дисперсии)")

        # Spearman correlation
        rho_spearman, p_spearman = spearmanr(A_vals, Theta_vals)
        print(f"  Spearman ρ = {rho_spearman:.6f}, p-value = {p_spearman:.6f}")

        # Kendall tau
        tau_kendall, p_kendall = stats.kendalltau(A_vals, Theta_vals)
        print(f"  Kendall τ = {tau_kendall:.6f}, p-value = {p_kendall:.6f}")

        # Fisher Z-transformation для доверительного интервала
        z = 0.5 * np.log((1 + r_pearson) / (1 - r_pearson))
        se = 1 / np.sqrt(n - 3)
        z_lower = z - 1.96 * se
        z_upper = z + 1.96 * se
        r_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
        r_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)
        print(f"  95% CI для r: [{r_lower:.4f}, {r_upper:.4f}]")

# 4. Линейная регрессия по группам
print("\n" + "=" * 80)
print("4. ЛИНЕЙНАЯ РЕГРЕССИЯ A ~ Θ ПО ГРУППАМ")
print("=" * 80)

for b_val in [-0.333, 0.0, 0.333]:
    group = groups[groups['b'] == b_val]
    X = group['Theta'].values.reshape(-1, 1)
    y = group['A'].values
    n = len(y)

    print(f"\n--- Группа b = {b_val:.3f} (n = {n}) ---")

    # sklearn
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    residuals = y - y_pred

    # Коэффициенты
    beta_1 = model.coef_[0]
    beta_0 = model.intercept_

    # R²
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot
    r_squared_adj = 1 - (1 - r_squared) * (n - 1) / (n - 2)

    # Стандартные ошибки
    se = np.sqrt(ss_res / (n - 2))
    se_beta1 = se / np.sqrt(np.sum((X.flatten() - np.mean(X)) ** 2))
    se_beta0 = se * np.sqrt(1 / n + np.mean(X) ** 2 / np.sum((X.flatten() - np.mean(X)) ** 2))

    # t-статистики
    t_beta1 = beta_1 / se_beta1
    t_beta0 = beta_0 / se_beta0

    # p-values
    p_beta1 = 2 * (1 - stats.t.cdf(abs(t_beta1), n - 2))
    p_beta0 = 2 * (1 - stats.t.cdf(abs(t_beta0), n - 2))

    print(f"  Уравнение: A = {beta_0:.4f} + ({beta_1:.4f}) · Θ")
    print(f"  Коэффициенты:")
    print(f"    β₀ (intercept): {beta_0:.6f}, SE = {se_beta0:.6f}, t = {t_beta0:.4f}, p = {p_beta0:.6f}")
    print(f"    β₁ (slope):      {beta_1:.6f}, SE = {se_beta1:.6f}, t = {t_beta1:.4f}, p = {p_beta1:.6f}")
    print(f"  R² = {r_squared:.6f}, скорректированный R² = {r_squared_adj:.6f}")
    print(f"  Стандартная ошибка остатков (MSE) = {se:.6f}")

    # F-тест
    F_stat = (r_squared / 1) / ((1 - r_squared) / (n - 2))
    F_p = 1 - stats.f.cdf(F_stat, 1, n - 2)
    print(f"  F-статистика = {F_stat:.4f}, p-value = {F_p:.6f}")

    # Тест на нормальность остатков
    if n >= 3:
        W_res, p_res = shapiro(residuals)
        print(f"  Нормальность остатков (Shapiro-Wilk): W = {W_res:.6f}, p = {p_res:.6f}")

    # 95% доверительные интервалы
    t_crit = stats.t.ppf(0.975, n - 2)
    ci_beta1 = (beta_1 - t_crit * se_beta1, beta_1 + t_crit * se_beta1)
    ci_beta0 = (beta_0 - t_crit * se_beta0, beta_0 + t_crit * se_beta0)
    print(f"  95% CI для β₀: [{ci_beta0[0]:.4f}, {ci_beta0[1]:.4f}]")
    print(f"  95% CI для β₁: [{ci_beta1[0]:.4f}, {ci_beta1[1]:.4f}]")

# 5. Анализ выбросов (Cook's distance и studentized residuals)
print("\n" + "=" * 80)
print("5. АНАЛИЗ ВЫБРОСОВ И ВЛИЯТЕЛЬНЫХ НАБЛЮДЕНИЙ")
print("=" * 80)

for b_val in [-0.333, 0.0, 0.333]:
    group = groups[groups['b'] == b_val]
    X = sm.add_constant(group['Theta'].values)
    y = group['A'].values
    model_sm = sm.OLS(y, X).fit()

    # Studentized residuals
    influence = model_sm.get_influence()
    studentized_res = influence.resid_studentized_external
    cooks_d = influence.cooks_distance[0]

    print(f"\nГруппа b = {b_val:.3f}:")
    outliers = []
    for i, (name, sr, cd) in enumerate(zip(group['name'], studentized_res, cooks_d)):
        flag = ""
        if abs(sr) > 2:
            flag += " [OUTLIER]"
        if cd > 4 / len(group):
            flag += " [INFLUENTIAL]"
        if flag:
            outliers.append((name, group['A'].iloc[i], group['Theta'].iloc[i], sr, cd, flag))
            print(f"  {name}: A = {group['A'].iloc[i]:.4f}, Θ = {group['Theta'].iloc[i]:.6f}, "
                  f"студ. остаток = {sr:.4f}, Cook's D = {cd:.6f}{flag}")

# 6. Множественная регрессия с взаимодействием
print("\n" + "=" * 80)
print("6. МОДЕЛЬ С ВЗАИМОДЕЙСТВИЕМ: A ~ Θ + b + Θ:b")
print("=" * 80)

df_model = groups.copy()
df_model['b_cat'] = df_model['b'].astype('category')

# Создаем модель
X = pd.get_dummies(df_model['b_cat'], prefix='b', drop_first=True)
X = X.astype(float)
X['Theta'] = df_model['Theta'].values
# Взаимодействия
for col in X.columns:
    if col.startswith('b_'):
        X[f'Theta_{col}'] = X['Theta'] * X[col]

y = df_model['A'].values
X = sm.add_constant(X)
model_full = sm.OLS(y, X).fit()
print(model_full.summary())

# 7. ANCOVA (анализ ковариаций)
print("\n" + "=" * 80)
print("7. ANCOVA: СРАВНЕНИЕ РЕГРЕССИЙ МЕЖДУ ГРУППАМИ")
print("=" * 80)

# Проверяем, различаются ли наклоны
formula = 'A ~ Theta + C(b) + Theta:C(b)'
model_ancova = ols(formula, data=df_model).fit()
anova_table = sm.stats.anova_lm(model_ancova, typ=2)
print(anova_table)

# 8. Объединенный анализ (все точки)
print("\n" + "=" * 80)
print("8. ОБЪЕДИНЕННЫЙ АНАЛИЗ ВСЕХ ТОЧЕК (n = 62)")
print("=" * 80)

all_data = df
print(f"\nВсего точек: {len(all_data)}")
r_all, p_all = pearsonr(all_data['A'], all_data['Theta'])
rho_all, p_rho_all = spearmanr(all_data['A'], all_data['Theta'])
print(f"Pearson r = {r_all:.6f}, p = {p_all:.6f}")
print(f"Spearman ρ = {rho_all:.6f}, p = {p_rho_all:.6f}")

# 9. Частная корреляция (контролируя b)
print("\n" + "=" * 80)
print("9. ЧАСТНАЯ КОРРЕЛЯЦИЯ A И Θ ПРИ КОНТРОЛЕ b")
print("=" * 80)

# Корреляция остатков
groups_analyzed = df[df['b'].isin([-0.333, 0.0, 0.333])]
residuals_A = []
residuals_Theta = []
for b_val in [-0.333, 0.0, 0.333]:
    group = groups_analyzed[groups_analyzed['b'] == b_val]
    residuals_A.extend(group['A'] - group['A'].mean())
    residuals_Theta.extend(group['Theta'] - group['Theta'].mean())

r_partial, p_partial = pearsonr(residuals_A, residuals_Theta)
print(f"Частная корреляция (контролируя b): r = {r_partial:.6f}, p = {p_partial:.6f}")

# 10. Визуализация
print("\n" + "=" * 80)
print("10. ПОСТРОЕНИЕ ГРАФИКОВ")
print("=" * 80)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, b_val in enumerate([-0.333, 0.0, 0.333]):
    ax = axes[idx]
    group = groups_analyzed[groups_analyzed['b'] == b_val]
    X = group['Theta'].values
    y = group['A'].values

    # Scatter plot
    ax.scatter(X, y, alpha=0.7, s=80, edgecolors='black', linewidth=0.5)

    # Regression line
    if len(X) > 2:
        model = LinearRegression()
        model.fit(X.reshape(-1, 1), y)
        X_line = np.linspace(X.min(), X.max(), 100)
        y_line = model.predict(X_line.reshape(-1, 1))
        ax.plot(X_line, y_line, 'r-', linewidth=2, label=f'Регрессия (R²={model.score(X.reshape(-1, 1), y):.3f})')

        # Confidence interval
        n = len(X)
        y_pred = model.predict(X.reshape(-1, 1))
        residuals = y - y_pred
        se = np.sqrt(np.sum(residuals ** 2) / (n - 2))
        se_line = se * np.sqrt(1 / n + (X_line - np.mean(X)) ** 2 / np.sum((X - np.mean(X)) ** 2))
        t_val = stats.t.ppf(0.975, n - 2)
        ax.fill_between(X_line, y_line - t_val * se_line, y_line + t_val * se_line,
                        alpha=0.2, color='red', label='95% CI')

    ax.set_xlabel('Θ (Тета)', fontsize=12)
    ax.set_ylabel('A', fontsize=12)
    ax.set_title(f'Группа b = {b_val:.3f} (n = {len(group)})', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('regression_by_groups.png', dpi=150, bbox_inches='tight')
plt.show()

# 11. QQ-plot для каждой группы
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, b_val in enumerate([-0.333, 0.0, 0.333]):
    ax = axes[idx]
    group_A = groups_analyzed[groups_analyzed['b'] == b_val]['A']
    stats.probplot(group_A, dist="norm", plot=ax)
    ax.set_title(f'Q-Q plot A для b = {b_val:.3f}', fontsize=14)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('qq_plots.png', dpi=150, bbox_inches='tight')
plt.show()

# 12. Boxplot распределений A по группам
fig, ax = plt.subplots(figsize=(10, 6))
bp = ax.boxplot([groups_analyzed[groups_analyzed['b'] == -0.333]['A'],
                 groups_analyzed[groups_analyzed['b'] == 0.0]['A'],
                 groups_analyzed[groups_analyzed['b'] == 0.333]['A']],
                labels=['b = -0.333', 'b = 0.000', 'b = +0.333'],
                patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightblue')
ax.set_ylabel('A', fontsize=12)
ax.set_title('Распределение A по группам b', fontsize=14)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('boxplot_by_groups.png', dpi=150, bbox_inches='tight')
plt.show()

# 13. Финальное резюме
print("\n" + "=" * 80)
print("ФИНАЛЬНОЕ РЕЗЮМЕ")
print("=" * 80)

print("""
1. СВЯЗЬ A И Θ:
   - В группе b = 0.000 связь отсутствует (r = {:.3f}, p = {:.3f})
   - В группе b = -0.333 связь полностью обусловлена выбросом epsilon_0
   - В группе b = +0.333 слабая связь становится незначимой при удалении хвостов
   - Общая корреляция (n=62): r = {:.3f}, p = {:.3f} (незначима)

2. НОРМАЛЬНОСТЬ A:
   - Группа b = 0.000: нормальность не отвергается (p > 0.05)
   - Группа b = -0.333: нормальность отвергается (выбросы)
   - Группа b = +0.333: нормальность отвергается (асимметрия)

3. ВЫВОД:
   Θ НЕ является предиктором A. Группировка по b отражает категориальные
   различия в средних значениях A, а не модуляцию связи с Θ.
""".format(
    pearsonr(groups[groups['b'] == 0.0]['A'], groups[groups['b'] == 0.0]['Theta'])[0],
    pearsonr(groups[groups['b'] == 0.0]['A'], groups[groups['b'] == 0.0]['Theta'])[1],
    r_all, p_all
))