import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class QuantumGraphDynamics:

    def __init__(self, G):
        self.G = G
        self.N = G.number_of_nodes()

        self.A = nx.to_numpy_array(G)
        deg = self.A.sum(axis=1, keepdims=True)
        self.W = self.A / np.maximum(deg, 1)

    # -----------------------------
    # НЕЛИНЕЙНАЯ ФУНКЦИЯ
    # -----------------------------
    def f_complex(self, psi, r):
        mag = np.abs(psi)
        phase = np.angle(psi)

        new_mag = r * mag * (1 - mag**2)
        new_mag = np.nan_to_num(new_mag)

        return new_mag * np.exp(1j * phase)

    # -----------------------------
    # ДИНАМИКА (исправленная)
    # -----------------------------
    def step(self, psi, r, epsilon):
        mixed = (1 - epsilon) * psi + epsilon * (self.W @ psi)
        psi_new = self.f_complex(mixed, r)

        # мягкая нормировка (не убивает хаос)
        norm = np.linalg.norm(psi_new)
        if norm > 10:
            psi_new /= norm

        return psi_new

    # -----------------------------
    # СИМУЛЯЦИЯ
    # -----------------------------
    def simulate(self, r, epsilon, T=1000, discard=500):
        psi = np.random.randn(self.N) + 1j*np.random.randn(self.N)
        psi /= np.linalg.norm(psi)

        history = []

        for t in range(T):
            psi = self.step(psi, r, epsilon)

            if t >= discard:
                history.append(psi.copy())

        return np.array(history)

    # -----------------------------
    # БИФУРКАЦИОННАЯ ДИАГРАММА
    # -----------------------------
    def bifurcation_diagram(self,
                            r_min=2.5,
                            r_max=4.2,
                            n_r=200,
                            epsilon=0.2,
                            T=1000,
                            discard=500,
                            node=0,
                            use_magnitude=True):

        rs = np.linspace(r_min, r_max, n_r)

        r_vals = []
        x_vals = []

        print("\n[БИФУРКАЦИИ] Старт...")

        for i, r in enumerate(rs):
            history = self.simulate(r, epsilon, T=T, discard=discard)

            if use_magnitude:
                series = np.abs(history[:, node])
            else:
                series = np.real(history[:, node])

            # добавляем все точки (не усредняем!)
            r_vals.extend([r] * len(series))
            x_vals.extend(series)

            if i % (n_r // 10) == 0:
                print(f"  r = {r:.3f}")

        return np.array(r_vals), np.array(x_vals)

    # -----------------------------
    # ПЛОТ БИФУРКАЦИЙ
    # -----------------------------
    def plot_bifurcation(self, r_vals, x_vals, title="Bifurcation Diagram"):
        plt.figure(figsize=(10, 6))
        plt.scatter(r_vals, x_vals, s=0.2)
        plt.xlabel("r")
        plt.ylabel("|ψ|")
        plt.title(title)
        plt.grid(alpha=0.3)
        plt.show()

    # -----------------------------
    # ДИСКРЕТНЫЙ АВТОМАТ
    # -----------------------------
    def to_automaton(self, psi):
        mag = np.abs(psi)

        return np.digitize(mag, [0.33, 0.66])

    # -----------------------------
    # БИФУРКАЦИИ ДЛЯ АВТОМАТА
    # -----------------------------
    def bifurcation_automaton(self,
                              r_min=2.5,
                              r_max=4.2,
                              n_r=200,
                              epsilon=0.2,
                              T=500,
                              discard=300,
                              node=0):

        rs = np.linspace(r_min, r_max, n_r)

        r_vals = []
        states = []

        print("\n[БИФУРКАЦИИ АВТОМАТА]")

        for r in rs:
            history = self.simulate(r, epsilon, T=T, discard=discard)

            for psi in history:
                state = self.to_automaton(psi)[node]
                r_vals.append(r)
                states.append(state)

        return np.array(r_vals), np.array(states)


# Параметры графа
N = 400  # число вершин
K = 6
p = 1 / (K * N**(1/3))  # вероятность связи

print(f"N = {N}")
print(f"p = 1/({K} * {N}^(1/3)) = 1/({K} * {N**(1/3):.2f}) = {p:.6f}")

# Создаём граф Эрдёша-Реньи
G = nx.erdos_renyi_graph(n=N, p=p, seed=42)

print(f"Число вершин: {G.number_of_nodes()}")
print(f"Число рёбер: {G.number_of_edges()}")
print(f"Средняя степень: {np.mean([d for _, d in G.degree()]):.2f}")
print(f"Плотность графа: {nx.density(G):.6f}")

# Создаем объект
q = QuantumGraphDynamics(G)

# Запускаем бифуркационную диаграмму
r_vals, x_vals = q.bifurcation_diagram(
    r_min=2.5,
    r_max=4.2,
    epsilon=0.2
)

# Отображаем результат
q.plot_bifurcation(r_vals, x_vals)