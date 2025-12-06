import numpy as np
from scipy.optimize import fsolve

# Оптимальные параметры сети
K_opt = 7.62
p_opt = 0.05
pi_target = 3.1415

# Уравнение для поиска alpha и beta
def equation(vars):
    alpha, beta = vars
    xi = K_opt**alpha / p_opt**beta
    pi_eff = 3 / (4 * xi**3)
    return pi_eff - pi_target

# Функция для fsolve с фиксированным beta, ищем alpha
def solve_alpha(beta_fixed):
    func = lambda alpha: equation([alpha, beta_fixed])
    alpha_solution = fsolve(func, 0.3)  # начальное приближение 0.3
    return alpha_solution[0]

# Подбор beta
beta_values = np.linspace(0.1, 1.0, 50)
solutions = []
for beta in beta_values:
    alpha = solve_alpha(beta)
    solutions.append((alpha, beta))

# Находим лучший вариант (alpha > 0)
solutions = np.array(solutions)
solutions = solutions[(solutions[:,0] > 0)]
best_alpha, best_beta = solutions[0]

print(f"Подобранные коэффициенты: alpha = {best_alpha:.4f}, beta = {best_beta:.4f}")
print(f"Проверка: π_eff = {3 / (4 * (K_opt**best_alpha / p_opt**best_beta)**3):.5f}")
