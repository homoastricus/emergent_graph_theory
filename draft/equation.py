import math

# Исходные значения
K = 8.0
p = 5e-02#5.270179e-02
N = 1.3e145#9.410e+122
e_const = math.e

print("Исходные значения:")
print(f"K = {K}")
print(f"p = {p}")
print(f"N = {N:.3e}")
print(f"e = {e_const:.6f}\n")

# 1. Вычисляем p*K
pK = p * K
print(f"1. p*K = {p:.6f} * {K} = {pK:.6f}")

# 2. Вычисляем ln(N)
ln_N = math.log(N)
print(f"2. ln(N) = ln({N:.3e}) = {ln_N:.6f}")

# 3. Вычисляем |ln(p*K)|
abs_ln_pK = abs(math.log(p*(K+p)))
print(f"3. |ln(p*K)| = |ln({pK:.6f})| = {abs_ln_pK:.6f}")

# 4. Вычисляем U = ln(N) / |ln(p*K)|
U = ln_N / abs_ln_pK
print(f"4. U = ln(N) / |ln(p*K)| = {ln_N:.6f} / {abs_ln_pK:.6f} = {U:.6f}")

# 5. Вычисляем левую часть уравнения: p * sqrt(K * U)
left_side = p * math.sqrt((K+p) * U)
print(f"5. p * sqrt(K * U) = {p:.6f} * sqrt({K} * {U:.6f})")
print(f"   = {p:.6f} * sqrt({K*U:.6f})")
print(f"   = {p:.6f} * {math.sqrt(K*U):.6f}")
print(f"   = {left_side:.6f}")

# 6. Сравниваем с e
print(f"\n6. Сравнение:")
print(f"   Левая часть: {left_side:.6f}")
print(f"   e (правая часть): {e_const:.6f}")
print(f"   Разность: {left_side - e_const:.6f}")

# 7. Вычисляем относительную ошибку
error_percent = abs((left_side - e_const) / e_const) * 100
print(f"   Относительная ошибка: {error_percent:.4f}%")

# 8. Проверка точности
if error_percent < 0.1:
    print("\n✓ Уравнение выполняется с хорошей точностью")
else:
    print("\n✗ Уравнение не выполняется точно")