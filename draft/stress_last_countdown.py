"""
ПОЛНЫЙ СТАТИСТИЧЕСКИЙ АНАЛИЗ γ-ЗНАЧЕНИЙ ДЛЯ ВСЕХ ФИЗИЧЕСКИХ КОНСТАНТ
На основе обновленного списка из 60+ параметров
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import gaussian_kde
from fractions import Fraction
from collections import Counter, defaultdict
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# ВСЕ ДАННЫЕ ИЗ ОБНОВЛЕННОГО СПИСКА
# ============================================================================

# Полный словарь всех констант и их γ-значений
gamma_full = {
    # Квантовые
    'ħ': 0.192835,
    'h': 0.192835,

    # Планковские
    'l_P': -0.104361,
    't_P': -0.222142,
    'm_P': 0.179369,
    'E_P': 0.415022,
    'T_P': -0.364637,

    # Фундаментальные
    'c': 0.117799,
    'G': -0.048154,
    'k_B': 0.224670,
    'α': -0.015396,

    # Лептоны
    'm_e': 0.515264,
    'm_muon': 0.061400,
    'm_tau': 0.248402,

    # Барионы
    'm_proton': -0.120978,
    'm_neutron': -0.462800,

    # Бозоны
    'm_W': -0.100838,
    'm_Z': 0.715684,
    'm_Higgs': -0.267871,

    # Мезоны
    'm_pion': 0.127663,
    'm_pion0': 0.271797,
    'm_kaon0': -0.193698,
    'm_D0': -1.079562,
    'm_J_psi': -0.385841,
    'm_eta': -0.248487,
    'm_Upsilon_1S': -1.179646,

    # Кварки
    'm_quark_u': -0.682866,
    'm_quark_d': 0.730184,
    'm_quark_s': -1.474050,
    'm_quark_c': 0.195006,
    'm_quark_b': -0.592949,
    'm_quark_t': 0.120094,

    # Атомные
    'Rydberg': 0.409414,
    'Bohr_radius': -0.424833,
    'Compton_e': -0.460215,
    'Compton_p': 0.196014,

    # Электромагнитные
    'e_charge': -0.105394,
    'epsilon_0': -0.506027,
    'mu_0': 0.270428,
    'impedance': 0.388227,
    'flux_quantum': 0.298216,

    # Космология
    'Lambda': 0.340703,
    'kappa_Einstein': -0.519350,
    'v_Higgs': 0.585737,

    # Времена жизни
    'tau_mu': -0.100283,
    'tau_tau': 0.012300,
    'tau_pion': 0.173669,
    'tau_neutron': -0.243215,
    'tau_kaon': -0.150499,
    'tau_D_plus': -0.031627,
    'tau_B_plus': 0.385605,
    'tau_Lambda_b': -0.008957,
    'tau_D0': 0.060468,

    # Отношения
    'm_proton/m_e': -0.636243,
    'm_muon/m_e': -0.453865,
    'm_tau/m_e': -0.266862,
    'm_W/m_e': -0.616102,
    'm_Z/m_e': 0.200420,
    'm_Higgs/m_e': -0.783135,
    'm_W/m_Z': -0.816522,
    'm_Higgs/m_W': -0.167033,
    'm_P/m_e': -0.335895,
}

# Категории для группировки
categories = {
    'Квантовые': ['ħ', 'h'],
    'Планковские': ['l_P', 't_P', 'm_P', 'E_P', 'T_P'],
    'Фундаментальные': ['c', 'G', 'k_B', 'α'],
    'Лептоны': ['m_e', 'm_muon', 'm_tau'],
    'Барионы': ['m_proton', 'm_neutron'],
    'Бозоны': ['m_W', 'm_Z', 'm_Higgs'],
    'Мезоны': ['m_pion', 'm_pion0', 'm_kaon0', 'm_D0', 'm_J_psi', 'm_eta', 'm_Upsilon_1S'],
    'Кварки': ['m_quark_u', 'm_quark_d', 'm_quark_s', 'm_quark_c', 'm_quark_b', 'm_quark_t'],
    'Атомные': ['Rydberg', 'Bohr_radius', 'Compton_e', 'Compton_p'],
    'Электромагнитные': ['e_charge', 'epsilon_0', 'mu_0', 'impedance', 'flux_quantum'],
    'Космология': ['Lambda', 'kappa_Einstein', 'v_Higgs'],
    'Времена жизни': ['tau_mu', 'tau_tau', 'tau_pion', 'tau_neutron', 'tau_kaon',
                      'tau_D_plus', 'tau_B_plus', 'tau_Lambda_b', 'tau_D0'],
    'Отношения': ['m_proton/m_e', 'm_muon/m_e', 'm_tau/m_e', 'm_W/m_e', 'm_Z/m_e',
                  'm_Higgs/m_e', 'm_W/m_Z', 'm_Higgs/m_W', 'm_P/m_e'],
}

# Цвета для категорий
category_colors = {
    'Квантовые': '#1f77b4',
    'Планковские': '#ff7f0e',
    'Фундаментальные': '#2ca02c',
    'Лептоны': '#d62728',
    'Барионы': '#9467bd',
    'Бозоны': '#8c564b',
    'Мезоны': '#e377c2',
    'Кварки': '#7f7f7f',
    'Атомные': '#bcbd22',
    'Электромагнитные': '#17becf',
    'Космология': '#aec7e8',
    'Времена жизни': '#ffbb78',
    'Отношения': '#98df8a',
}

# Извлекаем массивы
names_full = list(gamma_full.keys())
values_full = np.array(list(gamma_full.values()))
n_full = len(values_full)

# Создаем массив категорий для каждого значения
cat_array = []
for name in names_full:
    for cat, members in categories.items():
        if name in members:
            cat_array.append(cat)
            break
cat_array = np.array(cat_array)

# ============================================================================
# 1. РАСШИРЕННАЯ ОПИСАТЕЛЬНАЯ СТАТИСТИКА
# ============================================================================

print("=" * 90)
print("СТАТИСТИЧЕСКИЙ АНАЛИЗ γ-ЗНАЧЕНИЙ ДЛЯ {0} ФИЗИЧЕСКИХ КОНСТАНТ".format(n_full))
print("=" * 90)

print("\n" + "─" * 90)
print("1. ОПИСАТЕЛЬНАЯ СТАТИСТИКА ПОЛНОЙ ВЫБОРКИ")
print("─" * 90)

print(f"\n  Размер выборки:               {n_full}")
print(f"  Среднее арифметическое:        {np.mean(values_full):.6f}")
print(f"  Медиана:                       {np.median(values_full):.6f}")
print(f"  Стандартное отклонение:        {np.std(values_full, ddof=1):.6f}")
print(f"  Дисперсия:                     {np.var(values_full, ddof=1):.6f}")
print(f"  Минимальное значение:          {np.min(values_full):.6f} ({names_full[np.argmin(values_full)]})")
print(f"  Максимальное значение:         {np.max(values_full):.6f} ({names_full[np.argmax(values_full)]})")
print(f"  Размах:                        {np.max(values_full) - np.min(values_full):.6f}")
print(f"  Асимметрия (skewness):         {stats.skew(values_full):.6f}")
print(f"  Эксцесс (kurtosis):            {stats.kurtosis(values_full):.6f}")
print(f"  Коэффициент вариации:          {np.std(values_full, ddof=1)/abs(np.mean(values_full)):.4f}")

q1, q2, q3 = np.percentile(values_full, [25, 50, 75])
print(f"  Q1 (25-й процентиль):          {q1:.6f}")
print(f"  Q2 (50-й процентиль):          {q2:.6f}")
print(f"  Q3 (75-й процентиль):          {q3:.6f}")
print(f"  IQR (межквартильный размах):   {q3 - q1:.6f}")

# Выбросы по методу Тьюки
lower_fence = q1 - 1.5 * (q3 - q1)
upper_fence = q3 + 1.5 * (q3 - q1)
outliers = values_full[(values_full < lower_fence) | (values_full > upper_fence)]
print(f"  Границы выбросов:              [{lower_fence:.4f}, {upper_fence:.4f}]")
print(f"  Количество выбросов:           {len(outliers)}")
if len(outliers) > 0:
    for out_val in outliers:
        idx = np.where(values_full == out_val)[0][0]
        print(f"    • {names_full[idx]}: γ = {out_val:.6f}")

# ============================================================================
# 2. СТАТИСТИКА ПО КАТЕГОРИЯМ
# ============================================================================

print("\n" + "─" * 90)
print("2. СТАТИСТИКА ПО КАТЕГОРИЯМ")
print("─" * 90)

print(f"\n  {'Категория':<20} {'N':>3} {'Среднее':>10} {'Стд.откл.':>10} {'Мин.':>10} {'Макс.':>10} {'Диапазон':>10}")
print("  " + "─" * 75)

category_stats = {}
for cat, members in categories.items():
    cat_values = [gamma_full[m] for m in members if m in gamma_full]
    if cat_values:
        cat_values = np.array(cat_values)
        category_stats[cat] = {
            'n': len(cat_values),
            'mean': np.mean(cat_values),
            'std': np.std(cat_values, ddof=1) if len(cat_values) > 1 else 0,
            'min': np.min(cat_values),
            'max': np.max(cat_values),
            'range': np.max(cat_values) - np.min(cat_values),
            'values': cat_values
        }
        print(f"  {cat:<20} {len(cat_values):>3} {np.mean(cat_values):>10.4f} "
              f"{np.std(cat_values, ddof=1) if len(cat_values) > 1 else 0:>10.4f} "
              f"{np.min(cat_values):>10.4f} {np.max(cat_values):>10.4f} "
              f"{np.max(cat_values) - np.min(cat_values):>10.4f}")

# ============================================================================
# 3. ТЕСТЫ НА НОРМАЛЬНОСТЬ И СЛУЧАЙНОСТЬ
# ============================================================================

print("\n" + "─" * 90)
print("3. ТЕСТЫ НА НОРМАЛЬНОСТЬ И РАВНОМЕРНОСТЬ")
print("─" * 90)

# Shapiro-Wilk
shapiro_stat, shapiro_p = stats.shapiro(values_full)
print(f"\n  Shapiro-Wilk test:")
print(f"    Статистика: {shapiro_stat:.6f}")
print(f"    p-value:    {shapiro_p:.6f}")
print(f"    Вывод: {'НЕ нормальное' if shapiro_p < 0.05 else 'Возможно нормальное'} распределение")

# D'Agostino-Pearson
dagostino_stat, dagostino_p = stats.normaltest(values_full)
print(f"\n  D'Agostino-Pearson test:")
print(f"    Статистика: {dagostino_stat:.6f}")
print(f"    p-value:    {dagostino_p:.6f}")
print(f"    Вывод: {'НЕ нормальное' if dagostino_p < 0.05 else 'Возможно нормальное'} распределение")

# Anderson-Darling
ad_result = stats.anderson(values_full, dist='norm')
print(f"\n  Anderson-Darling test:")
print(f"    Статистика: {ad_result.statistic:.6f}")
for cv, sl in zip(ad_result.critical_values, ad_result.significance_level):
    marker = " ← ОТКЛОНЕНИЕ" if ad_result.statistic > cv else ""
    print(f"    Уровень {sl:.0f}%: критическое значение = {cv:.6f}{marker}")

# Тест на равномерность
ks_stat, ks_p = stats.kstest(values_full, 'uniform',
                              args=(np.min(values_full), np.max(values_full) - np.min(values_full)))
print(f"\n  Kolmogorov-Smirnov test (равномерность):")
print(f"    Статистика: {ks_stat:.6f}")
print(f"    p-value:    {ks_p:.6f}")
print(f"    Вывод: {'НЕ равномерное' if ks_p < 0.05 else 'Возможно равномерное'} распределение")

# ============================================================================
# 4. КЛАСТЕРНЫЙ АНАЛИЗ
# ============================================================================

print("\n" + "─" * 90)
print("4. КЛАСТЕРНЫЙ АНАЛИЗ")
print("─" * 90)

# Определяем оптимальное число кластеров
max_k = min(15, n_full - 1)
silhouette_scores = []

for k in range(2, max_k + 1):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(values_full.reshape(-1, 1))
    score = silhouette_score(values_full.reshape(-1, 1), labels)
    silhouette_scores.append(score)

best_k = np.argmax(silhouette_scores) + 2
best_score = max(silhouette_scores)

print(f"\n  Silhouette scores для разного числа кластеров:")
for k, score in enumerate(silhouette_scores, start=2):
    marker = " ← ОПТИМУМ" if k == best_k else ""
    print(f"    k={k:2d}: {score:.4f}{marker}")

# Финальная кластеризация
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(values_full.reshape(-1, 1))

print(f"\n  Состав кластеров (k={best_k}, silhouette={best_score:.4f}):")
for cluster_id in range(best_k):
    cluster_mask = clusters == cluster_id
    cluster_values = values_full[cluster_mask]
    cluster_names = [names_full[i] for i in range(n_full) if clusters[i] == cluster_id]
    cluster_cats = [cat_array[i] for i in range(n_full) if clusters[i] == cluster_id]

    print(f"\n  Кластер {cluster_id + 1}:")
    print(f"    Центр: {np.mean(cluster_values):.6f}")
    print(f"    Диапазон: [{np.min(cluster_values):.6f}, {np.max(cluster_values):.6f}]")
    print(f"    Размер: {len(cluster_values)}")
    print(f"    Категории: {dict(Counter(cluster_cats))}")
    print(f"    Константы: {', '.join(cluster_names)}")

# ============================================================================
# 5. АНАЛИЗ РАЦИОНАЛЬНЫХ ПРИБЛИЖЕНИЙ
# ============================================================================

print("\n" + "─" * 90)
print("5. ПОИСК РАЦИОНАЛЬНЫХ ПРИБЛИЖЕНИЙ")
print("─" * 90)

def find_best_rational(value, max_denom=100):
    """Находит лучшее рациональное приближение с малым знаменателем"""
    best_frac = None
    best_error = float('inf')

    for denom in range(1, max_denom + 1):
        num = round(value * denom)
        approx = num / denom
        error = abs(value - approx)

        if error < best_error:
            best_error = error
            best_frac = (num, denom)

    return best_frac, best_error

def quality_string(error):
    if error < 1e-12:
        return "✦✦✦ ТОЧНОЕ"
    elif error < 0.001:
        return "✦✦ ОТЛИЧНОЕ"
    elif error < 0.005:
        return "✦ ХОРОШЕЕ"
    elif error < 0.01:
        return "ПРИЕМЛЕМОЕ"
    else:
        return "ГРУБОЕ"

print(f"\n  {'Константа':<22} {'γ':>10} {'Дробь':>12} {'Прибл.':>10} {'Ошибка':>10} {'Качество':<16}")
print("  " + "─" * 82)

exact_count = 0
good_count = 0
rational_results = []

for name in names_full:
    value = gamma_full[name]
    (num, denom), error = find_best_rational(value, max_denom=100)

    sign = '-' if num < 0 else '+'
    frac_str = f"{sign}{abs(num)}/{denom}" if num != 0 else "0"
    approx = num / denom if denom != 0 else 0
    quality = quality_string(error)

    if error < 1e-12:
        exact_count += 1
    if error < 0.01:
        good_count += 1

    rational_results.append({
        'name': name,
        'value': value,
        'num': num,
        'denom': denom,
        'error': error,
        'quality': quality
    })

    print(f"  {name:<22} {value:>10.6f} {frac_str:>12} {approx:>10.6f} {error:>10.6f} {quality:<16}")

print("  " + "─" * 82)
print(f"\n  Точных совпадений (ε < 10⁻¹²): {exact_count} из {n_full}")
print(f"  Хороших приближений (ε < 0.01): {good_count} из {n_full}")
print(f"  Доля хороших: {good_count/n_full:.1%}")

# Анализ знаменателей
denominators = [r['denom'] for r in rational_results if r['error'] < 0.01 and r['denom'] > 0]
if denominators:
    denom_counter = Counter(denominators)
    print(f"\n  Наиболее частые знаменатели (топ-15):")
    for denom, count in denom_counter.most_common(15):
        print(f"    Знаменатель {denom:3d}: {count} раз(а)")

# ============================================================================
# 6. ПОИСК ТОЧНЫХ РАЦИОНАЛЬНЫХ ОТНОШЕНИЙ МЕЖДУ ПАРАМИ
# ============================================================================

print("\n" + "─" * 90)
print("6. ТОЧНЫЕ РАЦИОНАЛЬНЫЕ ОТНОШЕНИЯ МЕЖДУ ПАРАМИ")
print("─" * 90)

exact_relations = []
for i in range(n_full):
    for j in range(i + 1, n_full):
        val1, val2 = values_full[i], values_full[j]
        name1, name2 = names_full[i], names_full[j]

        if abs(val2) < 1e-15:
            continue

        ratio = val1 / val2

        # Ищем точное или очень хорошее рациональное отношение
        for max_d in [10, 20, 30, 50]:
            frac = Fraction(abs(ratio)).limit_denominator(max_d)
            approx = frac.numerator / frac.denominator
            if ratio < 0:
                approx = -approx
            error = abs(ratio - approx)

            if error < 1e-10:  # практически точное
                sign = '-' if ratio < 0 else '+'
                frac_str = f"{sign}{frac.numerator}/{frac.denominator}"
                exact_relations.append((name1, name2, ratio, frac_str, error, val1, val2))
                break

exact_relations.sort(key=lambda x: abs(x[2]))  # сортировка по близости к 1

print(f"\n  Найдено {len(exact_relations)} точных рациональных отношений:")
print(f"\n  {'Пара':<45} {'Отношение':>12} {'Дробь':>12} {'Проверка':>20}")
print("  " + "─" * 90)

for name1, name2, ratio, frac_str, error, val1, val2 in exact_relations[:40]:
    check = f"{val1:.6f} / {val2:.6f} = {ratio:.6f}"
    print(f"  {name1:<22} / {name2:<22} {ratio:>12.6f} {frac_str:>12} {check:>20}")

if len(exact_relations) > 40:
    print(f"  ... и еще {len(exact_relations) - 40} точных отношений")

# ============================================================================
# 7. ПОИСК ЛИНЕЙНЫХ КОМБИНАЦИЙ
# ============================================================================

print("\n" + "─" * 90)
print("7. ПОИСК ЦЕЛОЧИСЛЕННЫХ ЛИНЕЙНЫХ КОМБИНАЦИЙ")
print("─" * 90)

print("\n  Проверка гипотезы: n₁·γ₁ + n₂·γ₂ ≈ 0 (целые n₁, n₂ < 10)")
print(f"\n  {'Пара':<45} {'Уравнение':<30} {'Сумма':>12} {'Ошибка':>10}")
print("  " + "─" * 90)

linear_relations = []
for i in range(n_full):
    for j in range(i + 1, n_full):
        val1, val2 = values_full[i], values_full[j]
        name1, name2 = names_full[i], names_full[j]

        for n1 in range(-10, 11):
            if n1 == 0:
                continue
            for n2 in range(-10, 11):
                if n2 == 0:
                    continue

                sum_val = n1 * val1 + n2 * val2
                if abs(sum_val) < 0.01:
                    linear_relations.append((name1, name2, n1, n2, sum_val, val1, val2))

linear_relations.sort(key=lambda x: abs(x[4]))

for name1, name2, n1, n2, sum_val, val1, val2 in linear_relations[:30]:
    eq = f"{n1}·γ({name1}) + {n2}·γ({name2})"
    print(f"  {name1:<22} + {name2:<22} {eq:<30} {sum_val:>12.6f} {abs(sum_val):>10.6f}")

print(f"\n  Всего найдено линейных комбинаций (|сумма| < 0.01): {len(linear_relations)}")

# ============================================================================
# 8. АНАЛИЗ РАСПРЕДЕЛЕНИЯ ЗНАКОВ
# ============================================================================

print("\n" + "─" * 90)
print("8. АНАЛИЗ ЗНАКОВ И СИММЕТРИИ")
print("─" * 90)

pos_mask = values_full > 0
neg_mask = values_full < 0
zero_mask = values_full == 0

pos_count = np.sum(pos_mask)
neg_count = np.sum(neg_mask)
zero_count = np.sum(zero_mask)

print(f"\n  Положительные γ: {pos_count} ({pos_count/n_full:.1%})")
print(f"  Отрицательные γ: {neg_count} ({neg_count/n_full:.1%})")
print(f"  Нулевые γ:       {zero_count} ({zero_count/n_full:.1%})")

# Биномиальный тест

from scipy.stats import binomtest
result = binomtest(min(pos_count, neg_count), n=pos_count + neg_count, p=0.5)
p_value_sign = result.pvalue

print(f"\n  Биномиальный тест на равновероятность знаков:")
print(f"    p-value: {p_value_sign:.6f}")
print(f"    Вывод: {'ЗНАЧИМАЯ асимметрия знаков' if p_value_sign < 0.05 else 'Нет значимой асимметрии'}")

# Тест на симметрию распределения
pos_vals = values_full[pos_mask]
neg_vals_abs = -values_full[neg_mask]

if len(pos_vals) > 0 and len(neg_vals_abs) > 0:
    ks_sym_stat, ks_sym_p = stats.ks_2samp(pos_vals, neg_vals_abs)
    print(f"\n  Тест Колмогорова-Смирнова на симметрию:")
    print(f"    Статистика: {ks_sym_stat:.6f}")
    print(f"    p-value:    {ks_sym_p:.6f}")
    print(f"    Вывод: {'Распределения РАЗЛИЧНЫ' if ks_sym_p < 0.05 else 'Нет оснований отвергать симметрию'}")

# ============================================================================
# 9. ВИЗУАЛИЗАЦИЯ
# ============================================================================

print("\n" + "=" * 90)
print("СОЗДАНИЕ ГРАФИКОВ...")
print("=" * 90)

plt.rcParams.update({
    'figure.figsize': (22, 16),
    'font.size': 9,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
})

fig = plt.figure(figsize=(24, 18))
fig.suptitle(f'КОМПЛЕКСНЫЙ АНАЛИЗ γ-ЗНАЧЕНИЙ ({n_full} ФИЗИЧЕСКИХ КОНСТАНТ)',
             fontsize=16, fontweight='bold', y=0.98)

# --- График 1: Категориальный box plot ---
ax1 = plt.subplot(3, 4, 1)
cat_names = list(category_stats.keys())
cat_data = [category_stats[cat]['values'] for cat in cat_names]
bp = ax1.boxplot(cat_data, patch_artist=True, vert=True)
for patch, cat in zip(bp['boxes'], cat_names):
    patch.set_facecolor(category_colors.get(cat, 'gray'))
ax1.set_xticklabels(cat_names, rotation=45, ha='right', fontsize=7)
ax1.set_ylabel('γ значение')
ax1.set_title('Box plot по категориям')
ax1.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
ax1.grid(True, alpha=0.3, axis='y')

# --- График 2: Гистограмма + KDE ---
ax2 = plt.subplot(3, 4, 2)
n_bins = int(np.sqrt(n_full)) * 2
ax2.hist(values_full, bins=n_bins, density=True, alpha=0.6, color='steelblue', edgecolor='black')
kde = gaussian_kde(values_full)
x_range = np.linspace(np.min(values_full) - 0.2, np.max(values_full) + 0.2, 300)
ax2.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
ax2.axvline(x=0, color='black', linestyle='-', linewidth=1.5)
ax2.axvline(x=np.mean(values_full), color='green', linestyle='--', linewidth=1.5, label=f'μ={np.mean(values_full):.3f}')
ax2.axvline(x=np.median(values_full), color='orange', linestyle='--', linewidth=1.5, label=f'Me={np.median(values_full):.3f}')
ax2.set_xlabel('γ значение')
ax2.set_ylabel('Плотность')
ax2.set_title(f'Гистограмма + KDE (bins={n_bins})')
ax2.legend(fontsize=7)
ax2.grid(True, alpha=0.3)

# --- График 3: Q-Q plot ---
ax3 = plt.subplot(3, 4, 3)
stats.probplot(values_full, dist="norm", plot=ax3)
ax3.set_title('Q-Q plot (нормальное распределение)')
ax3.grid(True, alpha=0.3)

# --- График 4: Распределение на числовой оси ---
ax4 = plt.subplot(3, 4, 4)
np.random.seed(42)
y_jitter = np.random.uniform(-0.4, 0.4, n_full)
colors_by_cat = [category_colors.get(cat_array[i], 'gray') for i in range(n_full)]
ax4.scatter(values_full, y_jitter, c=colors_by_cat, alpha=0.7, s=60, edgecolors='black', linewidth=0.3)
ax4.axvline(x=0, color='black', linestyle='-', linewidth=1.5)
ax4.set_xlabel('γ значение')
ax4.set_yticks([])
ax4.set_title('Все значения на числовой оси')
ax4.grid(True, alpha=0.3, axis='x')

# --- График 5: Сортированные значения с категориями ---
ax5 = plt.subplot(3, 4, 5)
sorted_idx = np.argsort(values_full)
sorted_vals = values_full[sorted_idx]
sorted_cats = cat_array[sorted_idx]
bar_colors = [category_colors.get(c, 'gray') for c in sorted_cats]
bars = ax5.bar(range(n_full), sorted_vals, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=0.2)
ax5.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
ax5.set_xlabel('Индекс (сортировка)')
ax5.set_ylabel('γ значение')
ax5.set_title('Сортированные значения')
ax5.grid(True, alpha=0.3, axis='y')

# --- График 6: Кластеры ---
ax6 = plt.subplot(3, 4, 6)
cluster_colors = plt.cm.tab10(np.linspace(0, 1, best_k))
for cluster_id in range(best_k):
    mask = clusters == cluster_id
    y_pos = np.random.uniform(-0.3, 0.3, np.sum(mask))
    ax6.scatter(values_full[mask], y_pos, c=[cluster_colors[cluster_id]],
               alpha=0.7, s=60, edgecolors='black', linewidth=0.3, label=f'Кл.{cluster_id+1}')
    ax6.axvline(x=np.mean(values_full[mask]), color=cluster_colors[cluster_id],
                linestyle='--', linewidth=2, alpha=0.8)
ax6.axvline(x=0, color='black', linestyle='-', linewidth=1.5)
ax6.set_xlabel('γ значение')
ax6.set_yticks([])
ax6.set_title(f'Кластеры (k={best_k}, sil={best_score:.3f})')
ax6.legend(fontsize=7, ncol=2)
ax6.grid(True, alpha=0.3, axis='x')

# --- График 7: Pie chart знаков ---
ax7 = plt.subplot(3, 4, 7)
sizes = [pos_count, neg_count, zero_count]
labels = [f'+ ({pos_count})', f'- ({neg_count})', f'0 ({zero_count})']
colors_pie = ['#1f77b4', '#d62728', '#7f7f7f']
ax7.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%', startangle=90)
ax7.set_title('Распределение знаков γ')

# --- График 8: Анализ ошибок рациональных приближений ---
ax8 = plt.subplot(3, 4, 8)
errors = [r['error'] for r in rational_results]
ax8.hist(np.log10(errors), bins=30, color='steelblue', edgecolor='black', alpha=0.7)
ax8.axvline(x=-12, color='green', linestyle='--', label='Точные (10⁻¹²)')
ax8.axvline(x=-2, color='orange', linestyle='--', label='Хорошие (10⁻²)')
ax8.set_xlabel('log₁₀(ошибка)')
ax8.set_ylabel('Частота')
ax8.set_title('Распределение ошибок приближений')
ax8.legend(fontsize=7)
ax8.grid(True, alpha=0.3)

# --- График 9: Кумулятивная функция ---
ax9 = plt.subplot(3, 4, 9)
sorted_ecdf = np.sort(values_full)
y_ecdf = np.arange(1, n_full + 1) / n_full
ax9.step(sorted_ecdf, y_ecdf, where='post', linewidth=2, color='steelblue', label='ECDF')
x_theor = np.linspace(sorted_ecdf[0], sorted_ecdf[-1], 200)
ax9.plot(x_theor, stats.norm.cdf(x_theor, np.mean(values_full), np.std(values_full)),
         'r--', linewidth=2, label='Норм. CDF')
ax9.set_xlabel('γ значение')
ax9.set_ylabel('F(γ)')
ax9.set_title('Эмпирическая функция распределения')
ax9.legend(fontsize=7)
ax9.grid(True, alpha=0.3)

# --- График 10: Диаграмма рассеяния категорий ---
ax10 = plt.subplot(3, 4, 10)
unique_cats = list(category_stats.keys())
for i, cat in enumerate(unique_cats):
    cat_vals = category_stats[cat]['values']
    x_pos = np.full_like(cat_vals, i)
    ax10.scatter(x_pos + np.random.uniform(-0.2, 0.2, len(cat_vals)), cat_vals,
                c=category_colors[cat], alpha=0.7, s=50, edgecolors='black', linewidth=0.3, label=cat)
ax10.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax10.set_xticks(range(len(unique_cats)))
ax10.set_xticklabels(unique_cats, rotation=45, ha='right', fontsize=6)
ax10.set_ylabel('γ значение')
ax10.set_title('Распределение по категориям')
ax10.grid(True, alpha=0.3, axis='y')

# --- График 11: Частота знаменателей ---
ax11 = plt.subplot(3, 4, 11)
if denominators:
    top_denoms = denom_counter.most_common(20)
    denoms, counts = zip(*top_denoms)
    ax11.bar(range(len(denoms)), counts, color='steelblue', alpha=0.7, edgecolor='black')
    ax11.set_xticks(range(len(denoms)))
    ax11.set_xticklabels(denoms, fontsize=7)
    ax11.set_xlabel('Знаменатель')
    ax11.set_ylabel('Частота')
    ax11.set_title('Популярные знаменатели')

# --- График 12: Зависимость |γ| от категории ---
ax12 = plt.subplot(3, 4, 12)
abs_values = np.abs(values_full)
cat_means = {}
cat_stds = {}
for cat in unique_cats:
    cat_vals = np.abs(category_stats[cat]['values'])
    cat_means[cat] = np.mean(cat_vals)
    cat_stds[cat] = np.std(cat_vals) if len(cat_vals) > 1 else 0

cats_ordered = sorted(unique_cats, key=lambda c: cat_means[c], reverse=True)
x_pos = range(len(cats_ordered))
means = [cat_means[c] for c in cats_ordered]
stds = [cat_stds[c] for c in cats_ordered]
colors_bar = [category_colors[c] for c in cats_ordered]
ax12.bar(x_pos, means, yerr=stds, color=colors_bar, alpha=0.7, edgecolor='black', capsize=5)
ax12.set_xticks(x_pos)
ax12.set_xticklabels(cats_ordered, rotation=45, ha='right', fontsize=6)
ax12.set_ylabel('Средний |γ|')
ax12.set_title('Средний модуль γ по категориям')
ax12.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# ============================================================================
# 10. ДЕТАЛЬНЫЙ ГРАФИК С ПОДПИСЯМИ
# ============================================================================

fig2, ax = plt.subplots(figsize=(28, 12))
fig2.suptitle('ВСЕ γ-ЗНАЧЕНИЯ С ПОДПИСЯМИ И КАТЕГОРИЯМИ', fontsize=14, fontweight='bold')

# Создаем позиции по спирали для лучшей видимости
n = n_full
theta = np.linspace(0, 4 * np.pi, n)
r = np.linspace(0.2, 1.0, n)
x_spiral = r * np.cos(theta)
y_spiral = r * np.sin(theta) * 5

# Масштабируем x_spiral к диапазону значений
x_positions = values_full
y_positions = y_spiral * 0.5 + np.random.normal(0, 0.1, n)

ax.scatter(x_positions, y_positions, c=colors_by_cat, alpha=0.8, s=100,
           edgecolors='black', linewidth=0.5, zorder=3)

# Подписываем выборочно (чтобы не перегружать)
step = max(1, n // 40)
for i in range(0, n, step):
    name = names_full[i]
    val = values_full[i]
    cat = cat_array[i]
    ax.annotate(f'{name}\n({val:.4f})',
                xy=(val, y_positions[i]),
                xytext=(np.random.randint(-60, 60), np.random.randint(-30, 30)),
                textcoords='offset points',
                fontsize=7, alpha=0.9,
                bbox=dict(boxstyle='round,pad=0.2', facecolor=category_colors[cat], alpha=0.3),
                arrowprops=dict(arrowstyle='->', color='gray', alpha=0.4, lw=0.5))

ax.axvline(x=0, color='black', linestyle='-', linewidth=2, alpha=0.8)
ax.axvline(x=np.mean(values_full), color='green', linestyle='--', linewidth=1.5, alpha=0.7, label=f'μ = {np.mean(values_full):.4f}')
ax.axvline(x=np.median(values_full), color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Me = {np.median(values_full):.4f}')
ax.set_xlabel('γ значение', fontsize=12)
ax.set_ylabel('Ордината (для визуализации)', fontsize=10)
ax.legend(fontsize=10, loc='upper left')
ax.grid(True, alpha=0.3, axis='x')
ax.set_ylim(-4, 4)

plt.tight_layout()
plt.show()

# ============================================================================
# 11. ИТОГОВОЕ ЗАКЛЮЧЕНИЕ
# ============================================================================

print("\n" + "=" * 90)
print("ИТОГОВОЕ ЗАКЛЮЧЕНИЕ")
print("=" * 90)

print(f"""
  1. ОБЩАЯ ХАРАКТЕРИСТИКА ВЫБОРКИ ({n_full} констант):
     • Среднее γ = {np.mean(values_full):.4f} (95% ДИ: [{np.mean(values_full) - 1.96*np.std(values_full, ddof=1)/np.sqrt(n_full):.4f}, {np.mean(values_full) + 1.96*np.std(values_full, ddof=1)/np.sqrt(n_full):.4f}])
     • Медиана = {np.median(values_full):.4f}
     • Диапазон: [{np.min(values_full):.4f}, {np.max(values_full):.4f}]
     • Асимметрия = {stats.skew(values_full):.3f} ({"близко к симметричному" if abs(stats.skew(values_full)) < 0.5 else "асимметричное"})
     • Эксцесс = {stats.kurtosis(values_full):.3f} ({"нормальный" if abs(stats.kurtosis(values_full)) < 0.5 else "отличается от нормального"})

  2. ТЕСТЫ НА СЛУЧАЙНОСТЬ:
     • Нормальность: {"НЕ ОТВЕРГАЕТСЯ" if shapiro_p >= 0.05 else "ОТКЛОНЯЕТСЯ"} (Shapiro-Wilk p={shapiro_p:.4f})
     • Равномерность: {"НЕ ОТВЕРГАЕТСЯ" if ks_p >= 0.05 else "ОТКЛОНЯЕТСЯ"} (KS p={ks_p:.4f})
     • Симметрия знаков: {"НЕ ЗНАЧИМА" if p_value_sign >= 0.05 else "ЗНАЧИМА"} (бином. тест p={p_value_sign:.4f})

  3. СТРУКТУРА ДАННЫХ:
     • Оптимальное число кластеров: {best_k} (silhouette={best_score:.3f})
     • Точных рациональных γ: {exact_count} из {n_full}
     • Хороших приближений (ε<0.01): {good_count} из {n_full}
     • Точных рациональных отношений: {len(exact_relations)}
     • Целочисленных линейных комбинаций: {len(linear_relations)}

  4. КЛЮЧЕВЫЕ НАБЛЮДЕНИЯ:
     • {exact_count} значений являются точными рациональными дробями
     • {len(exact_relations)} пар имеют точные рациональные отношения
     • Распределение {"НЕ случайно" if shapiro_p < 0.05 or best_score > 0.5 else "близко к случайному"}
     • Присутствует {"значимая" if best_score > 0.5 else "умеренная"} кластерная структура
     • Доминирующие знаменатели: {', '.join([str(d) for d, _ in denom_counter.most_common(5)]) if denominators else 'нет данных'}

  5. ФИЗИЧЕСКАЯ ИНТЕРПРЕТАЦИЯ:
     • Отрицательные γ (n={neg_count}) соответствуют величинам, которые МЕНЬШЕ модельных значений
     • Положительные γ (n={pos_count}) — величины БОЛЬШЕ модельных
     • Близость к нулю указывает на хорошее согласие с моделью
     • Категории с наибольшим разбросом: мезоны, кварки (возможно, указывает на новую физику)
""")

print("=" * 90)
print("АНАЛИЗ ЗАВЕРШЕН")
print("=" * 90)