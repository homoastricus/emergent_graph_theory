import math

# Ваши параметры
K = 8.0
p = 0.05
N = 1.2e147

# Вычисление формулы
e = math.exp(1)
lnN = math.log(N)
Kp = K * p
lnKp = math.log(Kp)
U = lnN / abs(lnKp)

left = p# + e * U#
right =  e / math.sqrt(K * U)# + e * U
#p + eU = eU + 1
#p = e
print(f"Левая часть: {left}")
print(f"Правая часть: {right}")
print(f"Разница: {abs(left - right)}")
print(f"Относительная ошибка: {abs(left/right - 1)*100:.10f}%")