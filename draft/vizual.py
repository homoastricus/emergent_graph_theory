import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Параметры графа
N = 7000                # число узлов (для наглядности)
K = 8                 # локальная степень узла
p = 0.06              # вероятность дальней связи (малый мир)

# Создаем граф малого мира (Watts-Strogatz)
G = nx.watts_strogatz_graph(N, K, p)

# Задаем комплексные корреляции для каждой связи
# Допустим, корреляция — комплексное число: амплитуда и фаза
correlations = {}
for u, v in G.edges():
    amplitude = np.random.rand()         # случайная амплитуда [0,1]
    phase = np.random.uniform(0, 2*np.pi)  # случайная фаза [0,2pi]
    correlations[(u,v)] = amplitude * np.exp(1j * phase)

# Присвоим их как атрибут ребрам
nx.set_edge_attributes(G, correlations, "complex_corr")

# Визуализация графа (амплитуда = толщина, фаза = цвет)
pos = nx.circular_layout(G)  # круговая раскладка для наглядности
edges = G.edges(data=True)

# Подготовка цветов и толщины
edge_colors = [np.angle(data['complex_corr']) for u,v,data in edges]
edge_widths = [np.abs(data['complex_corr'])*5 for u,v,data in edges]  # масштабируем

plt.figure(figsize=(10,10))
nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=300)
nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths, edge_cmap=plt.cm.hsv)
nx.draw_networkx_labels(G, pos, font_size=10)

plt.title("Прототип графа Вселенной с комплексными корреляциями")
plt.axis('off')
plt.show()
