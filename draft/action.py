import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


def quantum_thermodynamic_graph(N=100, K=8, p=0.0527, T_range=None):
    """
    Расширенный квантово-термодинамический анализ графа Watts-Strogatz
    """

    if T_range is None:
        T_range = np.logspace(-2, 2, 50)  # Температуры от 0.01 до 100

    # --------------------------
    # 1. Генерация графа
    # --------------------------
    print(f"КВАНТОВО-ТЕРМОДИНАМИЧЕСКИЙ АНАЛИЗ ГРАФА")
    print(f"Параметры: N={N}, K={K}, p={p}")

    G = nx.watts_strogatz_graph(N, K, p, seed=42)
    L = nx.laplacian_matrix(G).astype(float).toarray()

    # Структурные характеристики
    avg_path = nx.average_shortest_path_length(G)
    clustering = nx.average_clustering(G)
    diameter = nx.diameter(G)

    # --------------------------
    # 2. Спектральный анализ
    # --------------------------
    eigenvalues, eigenvectors = eigh(L)

    # Фильтруем нулевое собственное значение
    nonzero_indices = eigenvalues > 1e-10
    energies = eigenvalues[nonzero_indices]
    eigenvectors = eigenvectors[:, nonzero_indices]

    n_modes = len(energies)

    # Физические величины для каждой моды
    momenta = np.sqrt(energies)  # p = √E (волновой вектор)
    wavelengths = 2 * np.pi / momenta  # λ = 2π/p
    frequencies = energies  # ω = E (в единицах ħ=1)

    # --------------------------
    # 3. Термодинамический анализ по температурам
    # --------------------------
    F_list = []  # Свободная энергия
    U_list = []  # Внутренняя энергия
    S_list = []  # Энтропия
    Cv_list = []  # Теплоёмкость
    Z_list = []  # Статсумма

    for T in T_range:
        if T == 0:
            beta = 1e10  # Бесконечно большой для T→0
        else:
            beta = 1.0 / T

        # Статистическая сумма
        Z = np.sum(np.exp(-beta * energies))

        # Вероятности состояний
        p_n = np.exp(-beta * energies) / Z

        # Термодинамические величины
        F = -T * np.log(Z)  # Свободная энергия
        U = np.sum(p_n * energies)  # Внутренняя энергия
        S = -np.sum(p_n * np.log(p_n + 1e-12))  # Энтропия
        Cv = beta ** 2 * (np.sum(p_n * energies ** 2) - U ** 2)  # Теплоёмкость

        F_list.append(F)
        U_list.append(U)
        S_list.append(S)
        Cv_list.append(Cv)
        Z_list.append(Z)

    # --------------------------
    # 4. Анализ при T=1 (реперная точка)
    # --------------------------
    T_ref = 1.0
    beta_ref = 1.0 / T_ref
    Z_ref = np.sum(np.exp(-beta_ref * energies))
    p_n_ref = np.exp(-beta_ref * energies) / Z_ref

    F_ref = -T_ref * np.log(Z_ref)
    U_ref = np.sum(p_n_ref * energies)
    S_ref = -np.sum(p_n_ref * np.log(p_n_ref + 1e-12))

    # --------------------------
    # 5. Вектор Фидлера и квантовое действие
    # --------------------------
    # Фидлер вектор (вторая мода)
    fidler_idx = 1 if len(energies) > 1 else 0
    lambda2 = energies[fidler_idx]
    psi2 = eigenvectors[:, fidler_idx]

    # Действие для Фидлера вектора
    action_fidler = psi2.T @ L @ psi2

    # Среднее действие по всем модам при T=1
    actions = []
    for i in range(n_modes):
        psi_i = eigenvectors[:, i]
        action_i = psi_i.T @ L @ psi_i
        actions.append(action_i)

    avg_action_T1 = np.sum(p_n_ref * np.array(actions))

    # Квантовый параметр: эффективная постоянная Планка
    # Из соотношения неопределённостей: Δx·Δp ≥ ħ/2
    # На графе: Δx ~ 1 (межвершинное расстояние), Δp ~ средний градиент

    # Вычисляем градиенты для Фидлера вектора
    edge_gradients = []
    for u, v in G.edges():
        grad = (psi2[u] - psi2[v]) ** 2
        edge_gradients.append(grad)

    avg_gradient = np.mean(edge_gradients)
    hbar_eff = 2 * avg_gradient  # из Δx·Δp ≥ ħ/2, где Δx=1
    m_eff = hbar_eff ** 2 / lambda2


    # --------------------------
    # 6. Анализ спектральных свойств
    # --------------------------
    # Спектральная плотность состояний
    hist, bin_edges = np.histogram(energies, bins=30, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Плотность состояний ρ(E)
    rho_E = hist

    # Средний уровень и флуктуации
    mean_level_spacing = np.mean(np.diff(np.sort(energies)))
    level_variance = np.var(energies)

    # Распределение Вигнера-Дайсона (индикатор хаоса)
    # s = разности соседних уровней
    if len(energies) > 2:
        level_spacings = np.diff(np.sort(energies))
        normalized_spacings = level_spacings / np.mean(level_spacings)
    else:
        normalized_spacings = np.array([])

    # --------------------------
    # 7. Вывод результатов
    # --------------------------
    print(f"\n1. СТРУКТУРНЫЕ ХАРАКТЕРИСТИКИ:")
    print(f"   Средняя длина пути: {avg_path:.4f}")
    print(f"   Коэффициент кластеризации: {clustering:.4f}")
    print(f"   Диаметр графа: {diameter}")
    print(f"   Число рёбер: {G.number_of_edges()}")
    print(f"   Средняя степень: {2 * G.number_of_edges() / N:.2f}")

    print(f"\n2. СПЕКТРАЛЬНЫЕ СВОЙСТВА:")
    print(f"   Число ненулевых мод: {n_modes}")
    print(f"   λ₂ (алгебраическая связность): {lambda2:.6f}")
    print(f"   Среднее расстояние между уровнями: {mean_level_spacing:.6f}")
    print(f"   Спектральная ширина: {energies.max() - energies.min():.6f}")
    print(f"   Спектральная плотность при E=λ₂: {rho_E[np.argmin(np.abs(bin_centers - lambda2))]:.6f}")

    print(f"\n3. ТЕРМОДИНАМИКА ПРИ T=1:")
    print(f"   Статистическая сумма Z = {Z_ref:.6f}")
    print(f"   Свободная энергия F = {F_ref:.6f}")
    print(f"   Внутренняя энергия U = {U_ref:.6f}")
    print(f"   Энтропия S = {S_ref:.6f}")
    print(f"   Эффективная температура из S: T_eff = {U_ref / S_ref if S_ref > 0 else 0:.6f}")

    print(f"\n4. КВАНТОВЫЕ ПАРАМЕТРЫ:")
    print(f"   Действие для Фидлера вектора: S_Ψ₂ = {action_fidler:.6f}")
    print(f"   Среднее действие при T=1: ⟨S⟩ = {avg_action_T1:.6f}")
    print(f"   Эффективная ħ: ħ_eff = {hbar_eff:.6f}")
    print(f"   Отношение λ₂/ħ_eff² = {lambda2 / (hbar_eff ** 2) if hbar_eff > 0 else 0:.6f}")
    print(f"Эффективная квантовая масса графа: m_eff = {m_eff:.6f}")

    print(f"\n5. РЕЖИМЫ ГРАФА:")
    if p < 0.01:
        print("   РЕЖИМ: Почти регулярное кольцо")
    elif p < 0.1:
        print("   РЕЖИМ: Маленький мир (оптимальный баланс)")
    elif p < 0.5:
        print("   РЕЖИМ: Переходный режим")
    else:
        print("   РЕЖИМ: Случайный граф")

    # Проверка соотношения неопределённостей
    delta_x = 1.0  # минимальное расстояние
    delta_p = avg_gradient
    uncertainty_product = delta_x * delta_p

    print(f"\n6. СООТНОШЕНИЕ НЕОПРЕДЕЛЁННОСТЕЙ:")
    print(f"   Δx ≈ {delta_x} (межвершинное расстояние)")
    print(f"   Δp ≈ {delta_p:.6f} (средний градиент)")
    print(f"   Δx·Δp = {uncertainty_product:.6f}")
    print(f"   ħ_eff/2 = {hbar_eff / 2:.6f}")
    print(f"   Выполняется: {uncertainty_product >= hbar_eff / 2}")

    # --------------------------
    # 8. Расширенная визуализация
    # --------------------------
    fig = plt.figure(figsize=(20, 16))

    # 8.1 Спектр энергий
    ax1 = plt.subplot(4, 4, 1)
    ax1.plot(energies, 'o-', markersize=3, linewidth=1, alpha=0.7)
    ax1.set_xlabel('Номер моды (k)')
    ax1.set_ylabel('Энергия E_k')
    ax1.set_title('Спектр энергий графа')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=lambda2, color='r', linestyle='--', alpha=0.5, label=f'λ₂={lambda2:.3f}')
    ax1.legend()

    # 8.2 Плотность состояний
    ax2 = plt.subplot(4, 4, 2)
    ax2.bar(bin_centers, rho_E, width=bin_edges[1] - bin_edges[0], alpha=0.7)
    ax2.set_xlabel('Энергия E')
    ax2.set_ylabel('Плотность состояний ρ(E)')
    ax2.set_title('Спектральная плотность')
    ax2.grid(True, alpha=0.3)
    ax2.axvline(x=lambda2, color='r', linestyle='--', alpha=0.5)

    # 8.3 Фидлер вектор
    ax3 = plt.subplot(4, 4, 3)
    ax3.plot(psi2, 'b-', linewidth=1.5, alpha=0.7)
    ax3.fill_between(range(len(psi2)), psi2, 0, where=psi2 > 0, color='blue', alpha=0.3)
    ax3.fill_between(range(len(psi2)), psi2, 0, where=psi2 < 0, color='red', alpha=0.3)
    ax3.set_xlabel('Вершина')
    ax3.set_ylabel('Ψ₂(x)')
    ax3.set_title(f'Фидлер вектор (λ₂={lambda2:.3f})')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)

    # 8.4 Термодинамика: свободная энергия
    ax4 = plt.subplot(4, 4, 4)
    ax4.semilogx(T_range, F_list, 'b-', linewidth=2)
    ax4.axvline(x=1.0, color='r', linestyle='--', alpha=0.5, label='T=1')
    ax4.set_xlabel('Температура T')
    ax4.set_ylabel('Свободная энергия F(T)')
    ax4.set_title('Температурная зависимость свободной энергии')
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    # 8.5 Энтропия
    ax5 = plt.subplot(4, 4, 5)
    ax5.semilogx(T_range, S_list, 'g-', linewidth=2)
    ax5.axvline(x=1.0, color='r', linestyle='--', alpha=0.5)
    ax5.set_xlabel('Температура T')
    ax5.set_ylabel('Энтропия S(T)')
    ax5.set_title('Температурная зависимость энтропии')
    ax5.grid(True, alpha=0.3)

    # 8.6 Теплоёмкость
    ax6 = plt.subplot(4, 4, 6)
    ax6.semilogx(T_range, Cv_list, 'r-', linewidth=2)
    ax6.axvline(x=1.0, color='r', linestyle='--', alpha=0.5)
    ax6.set_xlabel('Температура T')
    ax6.set_ylabel('Теплоёмкость Cv(T)')
    ax6.set_title('Теплоёмкость системы')
    ax6.grid(True, alpha=0.3)

    # 8.7 Распределение Больцмана при T=1
    ax7 = plt.subplot(4, 4, 7)
    ax7.plot(p_n_ref, 'o-', markersize=3, linewidth=1, alpha=0.7)
    ax7.set_xlabel('Номер моды')
    ax7.set_ylabel('Вероятность p_k')
    ax7.set_title(f'Распределение Больцмана при T=1')
    ax7.grid(True, alpha=0.3)
    ax7.set_yscale('log')

    # 8.8 Граф с раскраской по Фидлер вектору
    ax8 = plt.subplot(4, 4, 8)
    pos = nx.circular_layout(G)

    # Нормализуем для цветовой карты
    psi_norm = (psi2 - psi2.min()) / (psi2.max() - psi2.min())
    colors = plt.cm.RdYlBu(psi_norm)

    nx.draw(G, pos, node_color=colors, node_size=30,
            edge_color='gray', alpha=0.6, width=0.5, ax=ax8)
    ax8.set_title('Граф: цвет = значение Ψ₂')

    # 8.9 Распределение градиентов
    ax9 = plt.subplot(4, 4, 9)
    ax9.hist(edge_gradients, bins=30, alpha=0.7, edgecolor='black')
    ax9.axvline(x=avg_gradient, color='r', linestyle='--',
                label=f'Среднее = {avg_gradient:.4f}')
    ax9.set_xlabel('(Ψ₂(i) - Ψ₂(j))²')
    ax9.set_ylabel('Частота')
    ax9.set_title('Распределение градиентов на рёбрах')
    ax9.legend()
    ax9.grid(True, alpha=0.3)

    # 8.10 Распределение Вигнера-Дайсона
    ax10 = plt.subplot(4, 4, 10)
    if len(normalized_spacings) > 5:
        ax10.hist(normalized_spacings, bins=20, density=True,
                  alpha=0.7, edgecolor='black', label='Данные')

        # Теоретические распределения
        s = np.linspace(0, 5, 100)
        # Wigner-Dyson (GOE) для хаотических систем
        p_wigner = (np.pi * s / 2) * np.exp(-np.pi * s ** 2 / 4)
        # Пуассон для регулярных систем
        p_poisson = np.exp(-s)

        ax10.plot(s, p_wigner, 'r-', linewidth=2, alpha=0.7, label='Wigner-Dyson')
        ax10.plot(s, p_poisson, 'g-', linewidth=2, alpha=0.7, label='Poisson')

        ax10.set_xlabel('Нормированное расстояние s')
        ax10.set_ylabel('P(s)')
        ax10.set_title('Распределение расстояний между уровнями')
        ax10.legend()
        ax10.grid(True, alpha=0.3)

    # 8.11 Волновая функция в импульсном представлении
    ax11 = plt.subplot(4, 4, 11)
    psi_fft = np.abs(np.fft.fft(psi2)) ** 2
    freqs = np.fft.fftfreq(len(psi2))
    idx = np.argsort(freqs)
    ax11.plot(freqs[idx], psi_fft[idx], 'b-', alpha=0.7)
    ax11.set_xlabel('Волновое число k')
    ax11.set_ylabel('|FFT(Ψ₂)|²')
    ax11.set_title('Фурье-образ Фидлер вектора')
    ax11.grid(True, alpha=0.3)

    # 8.12 Фазовая диаграмма p vs C (отношение пути к кластеризации)
    ax12 = plt.subplot(4, 4, 12)
    # Исследуем несколько значений p
    p_test = np.logspace(-4, 0, 10)
    path_vals = []
    clust_vals = []

    for p_val in p_test:
        G_test = nx.watts_strogatz_graph(N, K, p_val, seed=42)
        path_vals.append(nx.average_shortest_path_length(G_test))
        clust_vals.append(nx.average_clustering(G_test))

    ax12.loglog(p_test, path_vals, 'bo-', label='Длина пути L(p)')
    ax12.loglog(p_test, clust_vals, 'ro-', label='Кластеризация C(p)')
    ax12.axvline(x=p, color='k', linestyle='--', label=f'p={p}')
    ax12.set_xlabel('Вероятность перестройки p')
    ax12.set_ylabel('L(p), C(p)')
    ax12.set_title('Фазовая диаграмма малого мира')
    ax12.legend()
    ax12.grid(True, alpha=0.3)

    # 8.13 Связь λ₂ с p
    ax13 = plt.subplot(4, 4, 13)
    lambda2_vals = []
    for p_val in p_test:
        G_test = nx.watts_strogatz_graph(N, K, p_val, seed=42)
        L_test = nx.laplacian_matrix(G_test).astype(float).toarray()
        eigvals_test = np.linalg.eigvalsh(L_test)
        eigvals_nonzero = eigvals_test[eigvals_test > 1e-10]
        lambda2_vals.append(eigvals_nonzero[1] if len(eigvals_nonzero) > 1 else 0)

    ax13.loglog(p_test, lambda2_vals, 'go-')
    ax13.axvline(x=p, color='k', linestyle='--', label=f'p={p}')
    ax13.set_xlabel('p')
    ax13.set_ylabel('λ₂(p)')
    ax13.set_title('Зависимость λ₂ от перестройки')
    ax13.legend()
    ax13.grid(True, alpha=0.3)

    # 8.14 Информация о системе
    ax14 = plt.subplot(4, 4, 14)
    info_text = (
        f"Граф Watts-Strogatz\n"
        f"N={N}, K={K}, p={p:.4f}\n"
        f"\nСтруктура:\n"
        f"L={avg_path:.3f}\n"
        f"C={clustering:.3f}\n"
        f"d={diameter}\n"
        f"\nТермодинамика (T=1):\n"
        f"F={F_ref:.3f}\n"
        f"U={U_ref:.3f}\n"
        f"S={S_ref:.3f}\n"
        f"\nКвантовые параметры:\n"
        f"ħ_eff={hbar_eff:.4f}\n"
        f"λ₂/ħ²={lambda2 / (hbar_eff ** 2):.2f}\n"
        f"⟨S⟩={avg_action_T1:.3f}"
    )
    ax14.text(0.1, 0.5, info_text, fontsize=9, va='center',
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax14.axis('off')

    # 8.15 Температурная зависимость действия
    ax15 = plt.subplot(4, 4, 15)
    # Вычисляем среднее действие для каждой температуры
    mean_actions = []
    for T_val in T_range:
        beta_val = 1.0 / T_val if T_val > 0 else 1e10
        p_val = np.exp(-beta_val * energies) / np.sum(np.exp(-beta_val * energies))
        mean_action = np.sum(p_val * np.array(actions))
        mean_actions.append(mean_action)

    ax15.semilogx(T_range, mean_actions, 'm-', linewidth=2)
    ax15.axvline(x=1.0, color='r', linestyle='--', alpha=0.5)
    ax15.set_xlabel('Температура T')
    ax15.set_ylabel('⟨S⟩(T)')
    ax15.set_title('Температурная зависимость среднего действия')
    ax15.grid(True, alpha=0.3)

    # 8.16 Режимы системы в зависимости от p
    ax16 = plt.subplot(4, 4, 16)
    # Вычисляем характеристику "малость мира"
    small_worldness = []
    for p_val in p_test:
        G_test = nx.watts_strogatz_graph(N, K, p_val, seed=42)
        L_test = nx.average_shortest_path_length(G_test)
        C_test = nx.average_clustering(G_test)
        # Для регулярного графа с тем же K
        L_rand = np.log(N) / np.log(K)
        C_rand = K / N
        sigma = (C_test / C_rand) / (L_test / L_rand)
        small_worldness.append(sigma)

    ax16.loglog(p_test, small_worldness, 'c-', linewidth=2)
    ax16.axvline(x=p, color='k', linestyle='--', label=f'p={p}')
    ax16.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='σ=1')
    ax16.set_xlabel('p')
    ax16.set_ylabel('σ (малость мира)')
    ax16.set_title('Параметр "малого мира" σ = (C/C_rand)/(L/L_rand)')
    ax16.legend()
    ax16.grid(True, alpha=0.3)

    plt.suptitle(f'Полный квантово-термодинамический анализ графа малого мира (N={N}, K={K}, p={p})',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

    return {
        'graph': G,
        'energies': energies,
        'eigenvectors': eigenvectors,
        'lambda2': lambda2,
        'psi2': psi2,
        'T_range': T_range,
        'F_list': F_list,
        'U_list': U_list,
        'S_list': S_list,
        'Cv_list': Cv_list,
        'Z_list': Z_list,
        'hbar_eff': hbar_eff,
        'avg_gradient': avg_gradient,
        'avg_action_T1': avg_action_T1,
        'actions': actions,
        'avg_path': avg_path,
        'clustering': clustering,
        'diameter': diameter
    }


# ======================
#    ЗАПУСК АНАЛИЗА
# ======================
if __name__ == "__main__":
    # Основной анализ
    results = quantum_thermodynamic_graph(N=10000, K=8, p=0.0527)

    # Дополнительный анализ для разных p

    print("СРАВНИТЕЛЬНЫЙ АНАЛИЗ ДЛЯ РАЗНЫХ p:")

    p_values = [0.0527]
    for p_val in p_values:
        print(f"\np = {p_val}:")
        try:
            G_test = nx.watts_strogatz_graph(10000, 8, p_val, seed=42)
            L_test = nx.laplacian_matrix(G_test).astype(float).toarray()
            eigvals = np.linalg.eigvalsh(L_test)
            eigvals_nonzero = eigvals[eigvals > 1e-10]
            lambda2_val = eigvals_nonzero[1] if len(eigvals_nonzero) > 1 else 0
            print(f"  λ₂ = {lambda2_val:.6f}, L = {nx.average_shortest_path_length(G_test):.3f}, "
                  f"C = {nx.average_clustering(G_test):.3f}")
        except:
            continue

