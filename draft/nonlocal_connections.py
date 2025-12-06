# Псевдокод для визуализации распределения
import matplotlib.pyplot as plt
import numpy as np

N = 50000  # Большое число узлов
d = np.arange(1, N//2 + 1)  # Все возможные расстояния от 1 до N/2
probability = np.ones_like(d) * (2/N)  # Равномерная вероятность

plt.figure(figsize=(10, 6))
plt.bar(d, probability, width=1.0, alpha=0.7)
plt.xlabel('Расстояние d между узлами')
plt.ylabel('Вероятность P(d)')
plt.title('Распределение длин нелокальных связей в кольцевой сети\n(РАВНОМЕРНОЕ РАСПРЕДЕЛЕНИЕ)')
plt.show()