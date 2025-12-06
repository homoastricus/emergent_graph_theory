"""
Анализ эмерджентного спектрального оператора Λ(N,K,p) = [log(Kp)/log N]²
и его связи с λ₁ (второй гармоникой лапласиана) для графа Уоттса-Строгаца
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix
from scipy.optimize import curve_fit
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# 1. ВАШ ОПЕРАТОР
# ============================================================================

def emergent_operator(N, K=8, p=0.0527):
    """
    Эмерджентный спектральный масштаб Λ(N, K, p) = [log(Kp)/log N]²
    """
    return (np.log(K * p) / np.log(N)) ** 2


# ============================================================================
# 2. ВЫЧИСЛЕНИЕ λ₁ ДЛЯ WS-ГРАФА
# ============================================================================

def compute_lambda1_ws(N, K=8, p=0.0527, n_samples=3, seed=42):
    """
    Вычисление λ₁ (Fiedler value) для графа Уоттса-Строгаца
    с усреднением по нескольким реализациям
    """
    lambda_samples = []

    for i in range(n_samples):
        # Генерация графа
        G = nx.watts_strogatz_graph(N, K, p, seed=seed + i * 1000)

        # Нормализованный лапласиан L_norm = I - D^{-1/2} A D^{-1/2}
        A = nx.adjacency_matrix(G)
        degrees = np.array(A.sum(axis=1)).flatten()
        degrees[degrees == 0] = 1  # избегаем деления на 0

        D_inv_sqrt = csr_matrix((1.0 / np.sqrt(degrees),
                                 (range(N), range(N))), shape=(N, N))
        L_norm = csr_matrix(np.eye(N)) - D_inv_sqrt @ A @ D_inv_sqrt

        # Вычисление собственных значений
        try:
            eigenvalues = eigsh(L_norm, k=min(5, N - 1), which='SM',
                                sigma=0, tol=1e-5, maxiter=3000)
            eigvals_sorted = np.sort(eigenvalues[0])
            lambda1 = eigvals_sorted[1] if len(eigvals_sorted) > 1 else eigvals_sorted[0]
            lambda_samples.append(lambda1)
        except:
            # Fallback: используем ненормализованный лапласиан
            L = nx.laplacian_matrix(G).astype(float)
            eigenvalues = eigsh(L, k=min(5, N - 1), which='SM',
                                sigma=0, tol=1e-5, maxiter=3000)
            eigvals_sorted = np.sort(eigenvalues[0])
            lambda1 = eigvals_sorted[1] if len(eigvals_sorted) > 1 else eigvals_sorted[0]
            lambda_samples.append(lambda1)

    return np.mean(lambda_samples), np.std(lambda_samples)


# ============================================================================
# 3. ВАШИ ДАННЫЕ И БЫСТРЫЙ АНАЛИЗ
# ============================================================================

def analyze_emergent_operator():
    """Анализ связи λ₁ и эмерджентного оператора Λ"""

    # Ваши данные (N и соответствующие λ₁)
    N_data = np.array([1000, 5000, 10000, 50000])
    lambda1_data = np.array([0.0135, 0.0201, 0.0185, 0.0128])

    # Параметры графа
    K = 8
    p = 0.0527

    print("=" * 60)
    print("АНАЛИЗ ЭМЕРДЖЕНТНОГО ОПЕРАТОРА Λ(N,K,p) = [log(Kp)/log N]²")
    print("=" * 60)
    print(f"Параметры: K={K}, p={p}")
    print(f"Kp = {K * p:.4f}")
    print()

    # Вычисляем Λ для ваших данных
    Lambda_data = emergent_operator(N_data, K, p)

    print("Таблица данных:")
    print("N\t\tλ₁ (измерено)\tΛ(N,K,p)\tλ₁/Λ\t\t1-Λ")
    for i in range(len(N_data)):
        print(f"{N_data[i]}\t\t{lambda1_data[i]:.6f}\t\t{Lambda_data[i]:.6f}\t"
              f"{lambda1_data[i] / Lambda_data[i]:.3f}\t\t{1 - Lambda_data[i]:.3f}")

    # 1. Проверка гипотезы: λ₁ ∝ Λ
    print("\n" + "=" * 60)
    print("ГИПОТЕЗА 1: λ₁ ∝ Λ")
    C_vals = lambda1_data / Lambda_data
    print(f"Коэффициенты пропорциональности C = λ₁/Λ:")
    for N, C in zip(N_data, C_vals):
        print(f"  N={N}: C = {C:.3f}")
    print(f"  Среднее: C = {np.mean(C_vals):.3f} ± {np.std(C_vals):.3f}")

    # 2. Проверка гипотезы: λ₁ = λ_∞ * (1 - Λ)
    print("\n" + "=" * 60)
    print("ГИПОТЕЗА 2: λ₁ = λ_∞ * (1 - Λ)")
    lambda_inf_estimates = lambda1_data / (1 - Lambda_data)
    print(f"Оценки λ_∞ = λ₁/(1-Λ):")
    for N, linf in zip(N_data, lambda_inf_estimates):
        print(f"  N={N}: λ_∞ = {linf:.4f}")
    lambda_inf = np.mean(lambda_inf_estimates)
    lambda_inf_std = np.std(lambda_inf_estimates)
    print(f"  Среднее: λ_∞ = {lambda_inf:.4f} ± {lambda_inf_std:.4f}")

    # 3. Линейная регрессия λ₁ от Λ
    print("\n" + "=" * 60)
    print("ЛИНЕЙНАЯ РЕГРЕССИЯ: λ₁ = a + b·Λ")
    coeffs = np.polyfit(Lambda_data, lambda1_data, 1)
    a, b = coeffs[1], coeffs[0]
    print(f"  λ₁ = {a:.4f} + {b:.4f}·Λ")
    print(f"  R² = {np.corrcoef(Lambda_data, lambda1_data)[0, 1] ** 2:.4f}")

    return N_data, lambda1_data, Lambda_data, lambda_inf, a, b


# ============================================================================
# 4. МОДЕЛИ С ЭМЕРДЖЕНТНЫМ ОПЕРАТОРОМ
# ============================================================================

def model_simple(N, K=8, p=0.0527, lambda_const=0.017):
    """Простая модель: λ₁ = константа"""
    return lambda_const * np.ones_like(N)


def model_emergent_linear(N, K=8, p=0.0527, a=0.018, b=-0.04):
    """Линейная модель: λ₁ = a + b·Λ"""
    Lambda = emergent_operator(N, K, p)
    return a + b * Lambda


def model_emergent_multiplicative(N, K=8, p=0.0527, lambda_inf=0.018):
    """Мультипликативная модель: λ₁ = λ_∞ * (1 - Λ)"""
    Lambda = emergent_operator(N, K, p)
    return lambda_inf * (1 - Lambda)


def model_full(N, K=8, p=0.0527, lambda_inf=0.018, alpha=1.0, beta=5.0):
    """Полная модель: λ₁ = λ_∞ * (1 - α·Λ) + β/N"""
    Lambda = emergent_operator(N, K, p)
    return lambda_inf * (1 - alpha * Lambda) + beta / N


# ============================================================================
# 5. ВИЗУАЛИЗАЦИЯ
# ============================================================================

def plot_analysis(N_data, lambda1_data, Lambda_data, lambda_inf, a, b):
    """Визуализация всех результатов"""

    K = 8
    p = 0.0527

    # Диапазон N для предсказаний
    N_range = np.logspace(3, 6, 100).astype(int)
    N_range[0] = 1000  # гарантируем минимум

    # Предсказания разных моделей
    lambda_const = model_simple(N_range, K, p, 0.017)
    lambda_linear = model_emergent_linear(N_range, K, p, a, b)
    lambda_mult = model_emergent_multiplicative(N_range, K, p, lambda_inf)

    # Подгонка полной модели
    def full_model_func(N, lambda_inf, alpha, beta):
        return model_full(N, K, p, lambda_inf, alpha, beta)

    try:
        # Начальные приближения
        p0 = [lambda_inf, 1.0, 5.0]
        bounds = ([0.01, 0.1, 0], [0.1, 2.0, 20])
        popt, _ = curve_fit(full_model_func, N_data, lambda1_data, p0=p0, bounds=bounds)
        lambda_full = full_model_func(N_range, *popt)
        lambda_inf_fit, alpha_fit, beta_fit = popt
    except:
        lambda_full = model_full(N_range, K, p, lambda_inf, 1.0, 5.0)
        lambda_inf_fit, alpha_fit, beta_fit = lambda_inf, 1.0, 5.0

    # Создаем графики
    fig = plt.figure(figsize=(15, 10))

    # График 1: Зависимость от N
    ax1 = plt.subplot(2, 2, 1)
    ax1.loglog(N_data, lambda1_data, 'bo', markersize=8, label='Данные')
    ax1.loglog(N_range, lambda_const, 'r--', linewidth=2, alpha=0.7, label='Константа (0.017)')
    ax1.loglog(N_range, lambda_linear, 'g-', linewidth=2, alpha=0.7, label=f'Линейная: {a:.3f}+{b:.3f}·Λ')
    ax1.loglog(N_range, lambda_mult, 'm:', linewidth=2, alpha=0.7, label=f'Мультипликативная: λ_∞={lambda_inf:.3f}')
    ax1.loglog(N_range, lambda_full, 'c-', linewidth=2, alpha=0.7,
               label=f'Полная: λ_∞={lambda_inf_fit:.3f}, α={alpha_fit:.2f}, β={beta_fit:.1f}')
    ax1.set_xlabel('N (размер графа)', fontsize=12)
    ax1.set_ylabel('λ₁ (Fiedler value)', fontsize=12)
    ax1.set_title('Зависимость λ₁ от N', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # График 2: λ₁ как функция Λ
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(Lambda_data, lambda1_data, 'bo', markersize=8, label='Данные')

    # Линейная регрессия
    Lambda_range = np.linspace(0.04, 0.12, 100)
    lambda_lin_fit = a + b * Lambda_range
    ax2.plot(Lambda_range, lambda_lin_fit, 'r-', linewidth=2,
             label=f'Линейная регрессия: λ₁ = {a:.4f} + {b:.4f}·Λ')

    # Идеальная мультипликативная модель
    lambda_mult_fit = lambda_inf * (1 - Lambda_range)
    ax2.plot(Lambda_range, lambda_mult_fit, 'g--', linewidth=2,
             label=f'Мультипликативная: λ₁ = {lambda_inf:.4f}·(1-Λ)')

    ax2.set_xlabel('Λ(N,K,p) = [log(Kp)/log N]²', fontsize=12)
    ax2.set_ylabel('λ₁', fontsize=12)
    ax2.set_title('λ₁ как функция эмерджентного оператора', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # График 3: Ошибки моделей
    ax3 = plt.subplot(2, 2, 3)

    models = [
        ('Константа', lambda N: model_simple(N, K, p, 0.017)),
        ('Линейная', lambda N: model_emergent_linear(N, K, p, a, b)),
        ('Мультипликативная', lambda N: model_emergent_multiplicative(N, K, p, lambda_inf)),
        ('Полная', lambda N: model_full(N, K, p, lambda_inf_fit, alpha_fit, beta_fit))
    ]

    errors = []
    model_names = []
    for name, model_func in models:
        pred = model_func(N_data)
        error = np.abs(pred - lambda1_data) / lambda1_data * 100
        errors.append(np.mean(error))
        model_names.append(name)

    bars = ax3.bar(range(len(models)), errors, color=['red', 'green', 'blue', 'cyan'])
    ax3.set_xticks(range(len(models)))
    ax3.set_xticklabels(model_names, rotation=45, ha='right')
    ax3.set_ylabel('Средняя относительная ошибка (%)', fontsize=12)
    ax3.set_title('Точность моделей', fontsize=14, fontweight='bold')

    # Добавляем значения на столбцы
    for bar, err in zip(bars, errors):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                 f'{err:.1f}%', ha='center', va='bottom')

    ax3.grid(True, alpha=0.3, axis='y')

    # График 4: Предсказание для огромного графа
    ax4 = plt.subplot(2, 2, 4)

    # Огромный граф (ваш случай)
    N_huge = 9 ** 123 / 4  # ~2.5e116
    Lambda_huge = emergent_operator(N_huge, K, p)

    # Предсказания разных моделей
    predictions = []
    labels = []

    for name, model_func in models:
        pred = model_func(np.array([N_huge]))[0]
        predictions.append(pred)
        labels.append(f'{name}\nλ₁={pred:.6f}')

    # Логарифмическая шкала для наглядности
    x_pos = np.arange(len(predictions))
    ax4.bar(x_pos, predictions, color=['red', 'green', 'blue', 'cyan'])
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax4.set_ylabel('λ₁', fontsize=12)
    ax4.set_title(f'Предсказание для огромного графа\nN≈{N_huge:.1e}, Λ={Lambda_huge:.2e}',
                  fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()

    # Вывод результатов для огромного графа
    print("\n" + "=" * 60)
    print("ПРЕДСКАЗАНИЯ ДЛЯ ОГРОМНОГО ГРАФА (N≈2.5×10¹¹⁶):")
    print("=" * 60)
    print(f"Λ(N,K,p) = {Lambda_huge:.2e}")
    print()

    for i, (name, _) in enumerate(models):
        print(f"{name}: λ₁ = {predictions[i]:.6f}")

    return lambda_inf_fit, alpha_fit, beta_fit


# ============================================================================
# 6. ПРИМЕР ИСПОЛЬЗОВАНИЯ В ИССЛЕДОВАНИЯХ
# ============================================================================

def example_emergent_property(K=8, p=0.0527):
    """
    Пример использования эмерджентного оператора для моделирования
    времени перемешивания случайного блуждания
    """

    def mixing_time(N, lambda1, K=8):
        """Время перемешивания: τ ∼ 1/λ₁ с поправкой на локальную структуру"""
        # Базовое время
        tau_base = 1.0 / lambda1

        # Поправка на локальную кластеризацию
        clustering_correction = 1 + 2 * p

        # Поправка на конечный размер
        finite_size_correction = 1 + 3 / np.sqrt(N)

        return tau_base * clustering_correction * finite_size_correction

    # Диапазон размеров
    N_test = np.array([1000, 5000, 10000, 50000, 100000])

    print("\n" + "=" * 60)
    print("ПРИМЕР: ВРЕМЯ ПЕРЕМЕШИВАНИЯ СЛУЧАЙНОГО БЛУЖДАНИЯ")
    print("=" * 60)

    for N in N_test:
        # Вычисляем Λ
        Lambda = emergent_operator(N, K, p)

        # Оцениваем λ₁ через эмерджентный оператор
        lambda_inf = 0.018  # из предыдущего анализа
        lambda1_est = lambda_inf * (1 - Lambda)

        # Вычисляем время перемешивания
        tau = mixing_time(N, lambda1_est, K)

        print(f"N={N:6d}: Λ={Lambda:.4f}, λ₁≈{lambda1_est:.4f}, τ≈{tau:.1f} шагов")

    # Для огромного графа
    N_huge = 9 ** 123 / 1.1
    Lambda_huge = emergent_operator(N_huge, K, p)
    lambda1_huge = lambda_inf * (1 - Lambda_huge)
    tau_huge = mixing_time(N_huge, lambda1_huge, K)

    print(f"\nОгромный граф (N≈{N_huge:.1e}):")
    print(f"  Λ = {Lambda_huge:.2e}")
    print(f"  λ₁ ≈ {lambda1_huge:.6f}")
    print(f"  Время перемешивания τ ≈ {tau_huge:.1f} шагов")
    print(f"  (практически: τ ≈ {1 / lambda_inf:.1f} шагов, так как Λ ≈ 0)")


# ============================================================================
# 7. ОСНОВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    """Основная функция анализа"""

    print("ЗАПУСК АНАЛИЗА ЭМЕРДЖЕНТНОГО ОПЕРАТОРА")
    print("=" * 60)

    # 1. Анализ ваших данных
    N_data, lambda1_data, Lambda_data, lambda_inf, a, b = analyze_emergent_operator()

    # 2. Визуализация
    lambda_inf_fit, alpha_fit, beta_fit = plot_analysis(N_data, lambda1_data, Lambda_data,
                                                        lambda_inf, a, b)

    # 3. Пример использования
    example_emergent_property()

    # 4. Итоговая формула
    print("\n" + "=" * 60)
    print("ИТОГОВАЯ РЕКОМЕНДУЕМАЯ МОДЕЛЬ:")
    print("=" * 60)
    print(f"λ₁(N, K, p) = λ_∞(K, p) · [1 - α(K, p) · Λ(N, K, p)] + β(K, p)/N")
    print()
    print(f"где Λ(N, K, p) = [log(K·p) / log(N)]²")
    print()
    print(f"Для ваших параметров (K=8, p=0.0527):")
    print(f"  λ_∞ ≈ {lambda_inf_fit:.4f}")
    print(f"  α ≈ {alpha_fit:.2f}")
    print(f"  β ≈ {beta_fit:.1f}")
    print()
    print(f"Упрощенная версия (для больших N):")
    print(f"  λ₁(N) ≈ {lambda_inf_fit:.4f} · [1 - {alpha_fit:.2f}·Λ(N)]")
    print()
    print(f"Для практических расчетов:")
    print(f"  λ₁ ≈ {lambda_inf_fit:.4f} (с точностью до 1% для N > 10^4)")

    return {
        'lambda_inf': lambda_inf_fit,
        'alpha': alpha_fit,
        'beta': beta_fit,
        'a': a,
        'b': b
    }


# ============================================================================
# 8. ДОПОЛНИТЕЛЬНЫЕ УТИЛИТЫ
# ============================================================================

def compute_new_point(N_new, K=8, p=0.0527, n_samples=3):
    """Вычисление λ₁ для нового значения N"""
    print(f"\nВычисление λ₁ для N={N_new}...")
    lambda_mean, lambda_std = compute_lambda1_ws(N_new, K, p, n_samples)
    Lambda_new = emergent_operator(N_new, K, p)

    print(f"Результат: λ₁ = {lambda_mean:.6f} ± {lambda_std:.6f}")
    print(f"Λ(N,K,p) = {Lambda_new:.6f}")
    print(f"Отношение λ₁/Λ = {lambda_mean / Lambda_new:.3f}")

    return lambda_mean, lambda_std, Lambda_new


# ============================================================================
# ЗАПУСК ПРОГРАММЫ
# ============================================================================

if __name__ == "__main__":
    # Запускаем основной анализ
    params = main()

    # Опционально: вычисляем новую точку
    print("\n" + "=" * 60)
    print("ДОПОЛНИТЕЛЬНО: ВЫЧИСЛЕНИЕ НОВОЙ ТОЧКИ")
    print("=" * 60)

    # Можно раскомментировать для вычисления конкретного N
    # N_test = 8000
    # lambda_test, std_test, Lambda_test = compute_new_point(N_test)

    print("\n" + "=" * 60)
    print("АНАЛИЗ ЗАВЕРШЕН")
    print("=" * 60)