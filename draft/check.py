# Давайте точно проверим:
import math

K = 8.0
p = 0.05270179
N = 9.702e122

left = math.e
right = p * math.sqrt((p + K) * math.log(N) / abs(math.log(p * (p + K))))

print(f"Левая часть (e):  {left:.15f}")
print(f"Правая часть:     {right:.15f}")
print(f"Отношение:        {right/left:.15f}")
print(f"Ошибка:           {(right-left)/left*1e6:.3f} ppm")