import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Константы
e = np.e
K = 8.0


def equation_for_p(p, N):
    """Уравнение: e - p*sqrt((K+p)*ln(N)/|ln(p*(K+p))|) = 0"""
    if p <= 0 or p >= 1:
        return 1e10  # Большое число для выхода за границы

    # Вычисляем аргумент логарифма
    arg = p * (K + p)

    # Избегаем численных проблем
    if arg <= 0:
        return 1e10

    # Вычисляем логарифм (может быть положительным или отрицательным)
    log_val = np.log(arg)
    abs_log = np.abs(log_val)

    # Вычисляем U
    U = np.log(N) / abs_log if abs_log > 1e-12 else 1e10

    # Вычисляем левую часть уравнения
    sqrt_val = np.sqrt((K + p) * U)
    left_side = p * sqrt_val

    # Возвращаем разность
    return left_side - e


def find_p_for_N(N, p_guess=0.1):
    """Находит p для заданного N"""
    try:
        # Решаем уравнение
        sol = fsolve(equation_for_p, p_guess, args=(N,), full_output=True)
        p_sol = sol[0][0]

        # Проверяем, что решение в допустимом диапазоне
        if 0 < p_sol < 1:
            # Уточняем точность решения
            error = abs(equation_for_p(p_sol, N))
            if error < 1e-6:
                return p_sol
            else:
                # Пробуем другие начальные приближения
                for guess in [0.05, 0.2, 0.5, 0.8]:
                    sol = fsolve(equation_for_p, guess, args=(N,), full_output=True)
                    p_sol = sol[0][0]
                    if 0 < p_sol < 1 and abs(equation_for_p(p_sol, N)) < 1e-6:
                        return p_sol
    except:
        pass

    return np.nan


# Диапазон N от 100 до 10^123
N_min = 100
N_max = 1e123
num_points = 200  # Уменьшим количество точек для скорости

# Создаём логарифмическую сетку для N
N_vals = np.logspace(np.log10(N_min), np.log10(N_max), num_points)

# Находим p для каждого N
p_vals = []
for i, N in enumerate(N_vals):
    # Используем предыдущее решение как начальное приближение (если есть)
    if i == 0:
        p_guess = 0.1
    else:
        p_guess = p_vals[-1] if not np.isnan(p_vals[-1]) else 0.1

    p_val = find_p_for_N(N, p_guess)
    p_vals.append(p_val)

    # Прогресс
    if i % 20 == 0:
        print(f"Выполнено: {i / num_points * 100:.1f}%")

p_vals = np.array(p_vals)

# Фильтруем только валидные значения
valid_indices = ~np.isnan(p_vals)
N_valid = N_vals[valid_indices]
p_valid = p_vals[valid_indices]

# Построение графика
plt.figure(figsize=(12, 8))

# График 1: p от N в логарифмическом масштабе
plt.subplot(2, 1, 1)
plt.loglog(N_valid, p_valid, 'b-', linewidth=2, marker='o', markersize=3)
plt.xlabel('N (логарифмическая шкала)', fontsize=12)
plt.ylabel('p (логарифмическая шкала)', fontsize=12)
plt.title(r'Зависимость $p$ от $N$: $e = p\sqrt{(K+p) \cdot \frac{\ln N}{|\ln(p(K+p))|}}$', fontsize=14)
plt.grid(True, alpha=0.3)
plt.xlim(N_min, N_max)

# Добавляем информацию
plt.text(1e60, 0.1, f'K = {K}', fontsize=12,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

# График 2: p от ln(N) в линейных координатах для наглядности
plt.subplot(2, 1, 2)
plt.semilogx(N_valid, p_valid, 'r-', linewidth=2)
plt.xlabel('N (логарифмическая шкала)', fontsize=12)
plt.ylabel('p', fontsize=12)
plt.title('Тот же график в полулогарифмических координатах', fontsize=14)
plt.grid(True, alpha=0.3)
plt.ylim(0, max(p_valid) * 1.1)

plt.tight_layout()
plt.show()

# Проверка точности для 10 точек
print("\n" + "=" * 80)
print("Проверка точности решения (10 точек):")
print("=" * 80)
print(f"{'N':>20} {'p':>15} {'p*sqrt(...)':>20} {'e':>15} {'Ошибка,%':>10}")
print("-" * 80)

# Создаем 10 значений N равномерно в логарифмическом масштабе
test_Ns = np.logspace(np.log10(N_min), np.log10(N_max), 10)

for N in test_Ns:
    # Для надежности используем несколько начальных приближений
    p = None
    for guess in [0.1, 0.05, 0.2, 0.15]:
        p_try = find_p_for_N(N, guess)
        if not np.isnan(p_try):
            p = p_try
            break

    if p is not None and not np.isnan(p):
        # Вычисляем левую часть уравнения для проверки
        arg = p * (K + p)
        log_val = np.log(arg)
        abs_log = np.abs(log_val)
        U = np.log(N) / abs_log
        left_side = p * np.sqrt((K + p) * U)
        error = abs(left_side - e) / e * 100

        print(f"{N:20.2e} {p:15.6e} {left_side:20.10f} {e:15.10f} {error:10.6f}")
    else:
        print(f"{N:20.2e} {'Решение не найдено':>15}")

# Дополнительная информация
print("\n" + "=" * 80)
print("Сводная информация:")
print(f"Диапазон N: от {N_min} до {N_max:.2e}")
print(f"Диапазон p: от {min(p_valid):.6f} до {max(p_valid):.6f}")
print(f"Число валидных точек: {len(p_valid)} из {len(N_vals)}")
print("=" * 80)