import numpy as np
import math
from scipy import stats
import matplotlib.pyplot as plt

# 🔬 DENSITY-CORRECTED DISTANCE (DCD) ТЕСТ

def compute_f_functions():
    K = 6.0
    p = 5.270179e-02
    N = 9.702e+122

    lnK = math.log(K)
    lnKp = math.log(K * p)
    lnN = math.log(N)

    U = lnN / abs(lnKp)

    return {
        1: U / math.pi,
        2: lnK,
        3: math.sqrt(K * p),
        4: 1 / p,
        5: K / lnK,
        6: 1 + p
    }


def generate_spectrum(f_powers, max_complexity):
    """Генерирует спектр для заданных ограничений"""
    f = compute_f_functions()
    spectrum = []

    ranges = {i: range(f_powers[i][0], f_powers[i][1] + 1) for i in range(1, 7)}

    for a1 in ranges[1]:
        for a2 in ranges[2]:
            for a3 in ranges[3]:
                for a4 in ranges[4]:
                    for a5 in ranges[5]:
                        for a6 in ranges[6]:
                            if sum(map(abs, [a1, a2, a3, a4, a5, a6])) > max_complexity:
                                continue
                            try:
                                val = (f[1] ** a1 * f[2] ** a2 * f[3] ** a3 *
                                       f[4] ** a4 * f[5] ** a5 * f[6] ** a6)
                                if val > 0:
                                    spectrum.append(math.log10(val))
                            except:
                                continue

    return np.array(sorted(spectrum))


def get_particles():
    return {
        'electron': 9.10938356e-31,
        'muon': 1.883531627e-28,
        'tau': 3.16754e-27,
        'proton': 1.6726219e-27,
        'neutron': 1.6749275e-27,
        'up_part': 2.16e-30,
        'down_part': 4.67e-30,
        'strange': 9.36e-29,
        'charm': 1.27e-27,
        'bottom_part': 4.18e-27,
        'top_part': 3.08e-25,
        'W_boson': 1.433e-25,
        'Z_boson': 1.625e-25,
        'HIGGS': 2.246e-25,
        'pion': 2.406e-28,
        'kaon': 8.806e-28,
        'eta_meson': 9.491e-28,
        'rho_meson': 1.253e-27,
    }


def calculate_z_scores(masses, spectrum, m_e):
    """
    Вычисляет нормализованные расстояния z = d / local_spacing

    Для случайного распределения z ~ Uniform(0, 0.5)
    """
    z_scores = []
    details = []

    for mass in masses:
        target = math.log10(mass / m_e)

        # Находим ближайшие линии спектра
        idx = np.searchsorted(spectrum, target)

        if idx == 0:
            # Цель левее всего спектра
            nearest = spectrum[0]
            distance = abs(target - nearest)
            if len(spectrum) > 1:
                local_spacing = spectrum[1] - spectrum[0]
            else:
                local_spacing = 1.0
            left, right = None, spectrum[0]
        elif idx == len(spectrum):
            # Цель правее всего спектра
            nearest = spectrum[-1]
            distance = abs(target - nearest)
            if len(spectrum) > 1:
                local_spacing = spectrum[-1] - spectrum[-2]
            else:
                local_spacing = 1.0
            left, right = spectrum[-1], None
        else:
            left = spectrum[idx - 1]
            right = spectrum[idx]
            local_spacing = right - left

            dist_left = abs(target - left)
            dist_right = abs(right - target)

            distance = min(dist_left, dist_right)
            nearest = left if dist_left < dist_right else right

        # Нормализованное расстояние
        if local_spacing > 0:
            z = distance / local_spacing
        else:
            z = 0.0

        z_scores.append(z)
        details.append({
            'mass': mass,
            'target': target,
            'nearest': nearest,
            'distance': distance,
            'local_spacing': local_spacing,
            'z': z,
            'left': left,
            'right': right
        })

    return np.array(z_scores), details


def dcd_permutation_test(real_masses, spectrum, m_e, n_perm=10000, label=""):
    """
    Density-Corrected Distance permutation test.

    Сравнивает распределение z-оценок для реальных масс
    с распределением для случайных масс.
    """
    print(f"\n🎲 DCD Permutation test ({label}): {n_perm} итераций...")

    # Реальные z-оценки
    z_real, real_details = calculate_z_scores(real_masses, spectrum, m_e)

    real_mean = np.mean(z_real)
    real_std = np.std(z_real)

    # Логарифмический диапазон реальных масс
    logs = [math.log10(m / m_e) for m in real_masses]
    min_log, max_log = min(logs), max(logs)

    random_means = []
    random_stds = []
    all_random_z = []

    for i in range(n_perm):
        if (i + 1) % 2000 == 0:
            print(f"  Прогресс: {i + 1}/{n_perm}")

        fake_logs = np.random.uniform(min_log, max_log, len(real_masses))
        fake_masses = [m_e * (10 ** x) for x in fake_logs]
        z_fake, _ = calculate_z_scores(fake_masses, spectrum, m_e)

        random_means.append(np.mean(z_fake))
        random_stds.append(np.std(z_fake))
        all_random_z.extend(z_fake)

    random_means = np.array(random_means)
    random_stds = np.array(random_stds)
    all_random_z = np.array(all_random_z)

    # Статистические тесты
    # 1. Сравнение средних (односторонний: реальные МЕНЬШЕ?)
    p_value_mean = np.mean(random_means <= real_mean)
    z_score_mean = (real_mean - np.mean(random_means)) / np.std(random_means) if np.std(random_means) > 0 else 0

    # 2. KS-тест: реальные z vs случайные z
    # Берем выборку случайных z того же размера для KS-теста
    np.random.shuffle(all_random_z)
    random_sample = all_random_z[:len(z_real)]
    ks_stat, ks_pvalue = stats.ks_2samp(z_real, random_sample)

    # 3. Тест на равномерность (z_real должны быть uniform(0, 0.5) при H0)
    # Нормализуем на [0, 1] для теста
    z_normalized = z_real / 0.5
    uniform_stat, uniform_pvalue = stats.kstest(z_normalized, 'uniform')

    # 4. Log-likelihood тест
    # При H0 (равномерное распределение) ожидаемая плотность = 2 на [0, 0.5]
    # Логарифмическая функция правдоподобия: sum(log(2)) = n * log(2)
    # Отклонение: -2 * (sum(log(p_i)) - n*log(2)) ~ chi-square
    eps = 1e-10
    # Используем эмпирическую плотность через KDE
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(z_real, bw_method='scott')
    log_likelihood_real = np.sum(np.log(kde(z_real) + eps))

    # Для случайных данных
    random_ll = []
    for _ in range(min(1000, n_perm)):
        fake_logs = np.random.uniform(min_log, max_log, len(real_masses))
        fake_masses = [m_e * (10 ** x) for x in fake_logs]
        z_fake, _ = calculate_z_scores(fake_masses, spectrum, m_e)
        kde_fake = gaussian_kde(z_fake, bw_method='scott')
        ll_fake = np.sum(np.log(kde_fake(z_fake) + eps))
        random_ll.append(ll_fake)

    random_ll = np.array(random_ll)
    p_value_ll = np.mean(random_ll >= log_likelihood_real)  # Большее LL = лучшее соответствие данным

    return {
        'z_real': z_real,
        'real_details': real_details,
        'real_mean': real_mean,
        'real_std': real_std,
        'random_mean_avg': np.mean(random_means),
        'random_mean_std': np.std(random_means),
        'p_value_mean': p_value_mean,
        'z_score_mean': z_score_mean,
        'ks_stat': ks_stat,
        'ks_pvalue': ks_pvalue,
        'uniform_stat': uniform_stat,
        'uniform_pvalue': uniform_pvalue,
        'log_likelihood': log_likelihood_real,
        'random_ll_mean': np.mean(random_ll),
        'random_ll_std': np.std(random_ll),
        'p_value_ll': p_value_ll,
        'all_random_z': all_random_z
    }


def analyze_dcd_results(results_dict, blind_particles_names, all_particles):
    """
    Анализ результатов DCD теста
    """
    print("📊 АНАЛИЗ DCD РЕЗУЛЬТАТОВ")

    for config_name, res in results_dict.items():
        print(f"\n🔹 {config_name}:")
        print(f"   Среднее z (реальное):      {res['real_mean']:.4f}")
        print(f"   Среднее z (случайное):     {res['random_mean_avg']:.4f} ± {res['random_mean_std']:.4f}")
        print(f"   Z-оценка среднего:         {res['z_score_mean']:.2f}")
        print(f"   P-значение (среднее):      {res['p_value_mean']:.6f}")
        print(f"   KS-тест P-значение:        {res['ks_pvalue']:.6f}")
        print(f"   Uniform-тест P-значение:   {res['uniform_pvalue']:.6f}")
        print(f"   Log-likelihood P-значение: {res['p_value_ll']:.6f}")

        # Интерпретация
        if res['p_value_mean'] < 0.05 and res['ks_pvalue'] < 0.05:
            print(f"   🔥 СИГНАЛ ОБНАРУЖЕН! Реальные массы ЗНАЧИМО ближе к спектру.")
        elif res['p_value_mean'] < 0.1:
            print(f"   ✅ СЛАБЫЙ СИГНАЛ. Требуется дополнительная проверка.")
        else:
            print(f"   ⚠️ СИГНАЛ НЕ ОБНАРУЖЕН. Распределение неотличимо от случайного.")


def plot_dcd_results(results_dict, title_suffix=""):
    """
    Визуализация результатов DCD теста
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    configs = list(results_dict.keys())

    for idx, (config_name, res) in enumerate(results_dict.items()):
        row, col = idx // 3, idx % 3
        ax = axes[row, col]

        # Гистограмма z-оценок
        ax.hist(res['z_real'], bins=15, alpha=0.7, color='red',
                edgecolor='black', density=True, label='Реальные')
        ax.hist(res['all_random_z'], bins=30, alpha=0.5, color='blue',
                edgecolor='black', density=True, label='Случайные')

        # Ожидаемое равномерное распределение
        x = np.linspace(0, 0.5, 100)
        ax.plot(x, [2] * len(x), 'g--', linewidth=2, label='Uniform(0, 0.5)')

        ax.set_xlabel('z = d / local_spacing')
        ax.set_ylabel('Плотность')
        ax.set_title(f"{config_name}\nmean z = {res['real_mean']:.4f}, p = {res['p_value_mean']:.4f}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 0.5])

        # Добавляем аннотацию
        if res['p_value_mean'] < 0.05:
            ax.text(0.05, 0.95, "СИГНАЛ!", transform=ax.transAxes,
                    fontsize=12, fontweight='bold', color='darkred')

    for idx in range(len(configs), 6):
        row, col = idx // 3, idx % 3
        axes[row, col].axis('off')

    plt.suptitle(f'Density-Corrected Distance (DCD) Test\n{title_suffix}', fontsize=14)
    plt.tight_layout()
    plt.savefig('dcd_test_results.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_z_distribution_comparison(z_real, z_random, config_name):
    """
    Детальное сравнение распределений z-оценок
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Гистограммы
    axes[0].hist(z_real, bins=12, alpha=0.7, color='red',
                 edgecolor='black', density=True, label='Реальные массы')
    axes[0].hist(z_random, bins=30, alpha=0.5, color='blue',
                 edgecolor='black', density=True, label='Случайные массы')

    x = np.linspace(0, 0.5, 100)
    axes[0].plot(x, [2] * len(x), 'g--', linewidth=2, label='Ожидаемое Uniform(0,0.5)')

    axes[0].set_xlabel('z = d / local_spacing')
    axes[0].set_ylabel('Плотность')
    axes[0].set_title(f'Распределение z-оценок: {config_name}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([0, 0.5])

    # 2. Q-Q plot против равномерного распределения
    from scipy import stats
    stats.probplot(z_real, dist=stats.uniform(loc=0, scale=0.5), plot=axes[1])
    axes[1].set_title(f'Q-Q Plot против Uniform(0, 0.5)\n{config_name}')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'z_distribution_{config_name}.png', dpi=150, bbox_inches='tight')
    plt.show()


def detailed_particle_analysis(details, blind_set):
    """
    Детальный анализ z-оценок для каждой частицы
    """
    print("🔍 ДЕТАЛЬНЫЙ АНАЛИЗ ПО ЧАСТИЦАМ")
    print(
        f"\n{'Частица':<15} {'Тип':<8} {'log10(m/me)':<14} {'Ближ. линия':<14} {'Расст. d':<12} {'Шаг':<10} {'z':<10}")

    # Маппинг имен (details приходит без имен, нужно сопоставить)
    particles = get_particles()
    sorted_names = sorted(particles.keys(), key=lambda x: particles[x])

    for i, (name, mass) in enumerate([(n, particles[n]) for n in sorted_names]):
        if i < len(details):
            det = details[i]
            ptype = "BLIND" if name in blind_set else "TRAIN"
            print(f"{name:<15} {ptype:<8} {det['target']:<14.4f} {det['nearest']:<14.4f} "
                  f"{det['distance']:<12.6f} {det['local_spacing']:<10.6f} {det['z']:<10.4f}")


# 🚀 ЗАПУСК

def main():
    print("🔬 DENSITY-CORRECTED DISTANCE (DCD) ТЕСТ")
    print("\nИдея: Нормализовать расстояние до спектра на локальную плотность.")
    print("При H0 (нет структуры) z ~ Uniform(0, 0.5)")

    particles = get_particles()
    m_e = 9.10938356e-31

    BLIND_SET = {'W_boson', 'Z_boson', 'HIGGS', 'top_part', 'tau', 'charm', 'bottom_part'}

    all_masses = list(particles.values())
    blind_masses = [v for k, v in particles.items() if k in BLIND_SET]

    print(f"\n📊 ДАННЫЕ:")
    print(f"  Всего частиц: {len(all_masses)}")
    print(f"  BLIND: {len(blind_masses)}")

    # Конфигурации для тестирования
    configs = {
        "СВЕРХЖЕСТКАЯ": ({1: (-1, 1), 2: (-1, 1), 3: (-1, 1), 4: (-1, 1), 5: (-1, 1), 6: (-1, 1)}, 3),
        "ЖЕСТКАЯ": ({1: (-2, 2), 2: (-1, 1), 3: (-2, 2), 4: (-1, 1), 5: (-2, 2), 6: (-1, 1)}, 4),
        "УМЕРЕННАЯ": ({1: (-2, 2), 2: (-1, 1), 3: (-2, 2), 4: (-1, 1), 5: (-2, 2), 6: (-1, 1)}, 5),
    }

    all_results = {}
    all_details = {}

    for config_name, (powers, max_comp) in configs.items():
        print(f"📋 ТЕСТИРОВАНИЕ КОНФИГУРАЦИИ: {config_name}")

        spectrum = generate_spectrum(powers, max_comp)
        print(f"  Спектр: {len(spectrum)} линий")

        # DCD тест на всех частицах
        results = dcd_permutation_test(
            all_masses, spectrum, m_e, n_perm=5000, label=f"ALL ({config_name})"
        )

        all_results[config_name] = results
        all_details[config_name] = results['real_details']

        print(f"\n  📈 РЕЗУЛЬТАТЫ:")
        print(f"    Среднее z:            {results['real_mean']:.4f} (случ: {results['random_mean_avg']:.4f})")
        print(f"    P-значение (среднее): {results['p_value_mean']:.6f}")
        print(f"    KS-тест P-значение:   {results['ks_pvalue']:.6f}")
        print(f"    Uniform P-значение:   {results['uniform_pvalue']:.6f}")
        print(f"    LL P-значение:        {results['p_value_ll']:.6f}")

        # Строим Q-Q plot для лучшей конфигурации
        if config_name == "УМЕРЕННАЯ":
            plot_z_distribution_comparison(
                results['z_real'],
                results['all_random_z'][:5000],
                config_name
            )

    # Анализ результатов
    analyze_dcd_results(all_results, BLIND_SET, particles)

    # Детальный анализ для умеренной конфигурации
    if "УМЕРЕННАЯ" in all_details:
        detailed_particle_analysis(all_details["УМЕРЕННАЯ"], BLIND_SET)

    # Визуализация
    plot_dcd_results(all_results, "Все частицы")

    # ФИНАЛЬНОЕ ЗАКЛЮЧЕНИЕ
    print("🏆 ФИНАЛЬНОЕ ЗАКЛЮЧЕНИЕ DCD ТЕСТА")

    significant = []
    for name, res in all_results.items():
        if res['p_value_mean'] < 0.05 and res['ks_pvalue'] < 0.05:
            significant.append(name)

    if significant:
        print(f"\n🔥🔥🔥 СТРУКТУРА ОБНАРУЖЕНА! 🔥🔥🔥")
        print(f"\nКонфигурации с сигналом: {', '.join(significant)}")
        print("\n✅ ЭТО ОЗНАЧАЕТ:")
        print("  • После коррекции на плотность спектра,")
        print("    реальные массы находятся ЗНАЧИМО БЛИЖЕ к линиям спектра,")
        print("    чем случайные массы.")
        print("  • Эффект НЕ объясняется артефактами плотности.")
        print("  • Базис f1...f6 отражает РЕАЛЬНУЮ СТРУКТУРУ масс частиц.")
    else:
        print("\n⚠️ СТРУКТУРА НЕ ОБНАРУЖЕНА")
        print("\n  После коррекции на плотность спектра,")
        print("  реальные массы НЕ находятся значимо ближе к спектру,")
        print("  чем случайные массы.")
        print("\n  Это означает:")
        print("  • Наблюдаемая близость полностью объясняется плотностью спектра.")
        print("  • Базис f1...f6 является ХОРОШЕЙ АППРОКСИМАЦИОННОЙ СХЕМОЙ,")
        print("    но не доказано, что он отражает фундаментальную структуру.")

    print("ТЕСТ ЗАВЕРШЕН")


if __name__ == "__main__":
    main()