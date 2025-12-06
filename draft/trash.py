import numpy as np

# eigenvalues — массив собственных значений Лапласиана графа
eigenvalues = np.array([0.00001])

# 1. Сортируем значения
eig_sorted = np.sort(eigenvalues)

# 2. Вычисляем относительные разрывы между соседними значениями
gaps = np.diff(eig_sorted)

# 3. Находим индекс с наибольшим разрывом в области малых собственных значений
#    обычно первые несколько маленьких значений
threshold_index = np.argmax(gaps[:10])  # первые 10 значений, например
threshold = (eig_sorted[threshold_index] + eig_sorted[threshold_index+1]) / 2

# 4. Количество ненулевых собственных значений
DoF_quantum = np.sum(eigenvalues > threshold)

print("Порог для нулевых значений:", threshold)
print("Квантовые степени свободы:", DoF_quantum)
