import numpy as np
import networkx as nx
from scipy.linalg import eigvalsh
import matplotlib.pyplot as plt

# HEAT KERNEL
def heat_kernel(eigvals, t):
    return np.sum(np.exp(-t * eigvals))


# GEOMETRICITY MEASURE (MAIN ORDER PARAMETER)
def geometry_quality(t_vals, K_vals):
    """
    R^2 для log-log линейной модели:
    log K = a log t + b
    """

    x = np.log(t_vals)
    y = np.log(K_vals + 1e-12)

    A = np.vstack([x, np.ones_like(x)]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]

    y_pred = a * x + b

    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    r2 = 1 - ss_res / ss_tot

    slope = a

    return r2, slope


# FULL GRAPH ANALYSIS
def analyze_graph(N, k, p, t_vals):

    G = nx.watts_strogatz_graph(N, k, p)

    L = nx.laplacian_matrix(G).toarray()
    eigvals = eigvalsh(L)
    eigvals = eigvals[eigvals > 1e-12]

    K_vals = np.array([
        heat_kernel(eigvals, t)
        for t in t_vals
    ])

    r2, slope = geometry_quality(t_vals, K_vals)

    spectral_dim = -2 * slope

    return r2, spectral_dim


# PARAMETER GRID
N_list = [50, 100, 200, 400, 800]
p_list = np.logspace(-4, 0, 20)
k = 6

t_vals = np.logspace(-2, 2, 80)

# PHASE MAPS
R2_map = np.zeros((len(N_list), len(p_list)))
D_map = np.zeros_like(R2_map)

# COMPUTATION
for i, N in enumerate(N_list):
    for j, p in enumerate(p_list):

        print(f"N={N}, p={p:.5f}")

        r2, d_s = analyze_graph(N, k, p, t_vals)

        R2_map[i, j] = r2
        D_map[i, j] = d_s


# CRITICAL LINE p_c(N)
threshold = 0.95  # геометрическая фаза

p_c = []

for i, N in enumerate(N_list):

    row = R2_map[i]

    good = np.where(row > threshold)[0]

    if len(good) > 0:
        p_c.append(p_list[good[-1]])
    else:
        p_c.append(np.nan)

p_c = np.array(p_c)

# FIT p_c(N)
valid = ~np.isnan(p_c)

logN = np.log(np.array(N_list)[valid])
logp = np.log(p_c[valid])

alpha, b = np.polyfit(logN, logp, 1)
alpha = -alpha

print("\n==============================")
print("CRITICAL SCALING")
print("==============================")
print(f"p_c(N) ~ N^(-alpha)")
print(f"alpha ≈ {alpha:.4f}")

# PHASE DIAGRAM (HEATMAP)
plt.figure(figsize=(8, 5))

plt.imshow(
    R2_map,
    aspect='auto',
    origin='lower',
    extent=[
        np.log10(p_list[0]),
        np.log10(p_list[-1]),
        N_list[0],
        N_list[-1]
    ],
    cmap='viridis'
)

plt.colorbar(label='R^2 (geometry quality)')
plt.xlabel("log10(p)")
plt.ylabel("N")
plt.title("Geometric phase diagram")
plt.show()
# CRITICAL LINE
plt.figure()

plt.loglog(N_list, p_c, 'o-', label="measured")

plt.loglog(
    N_list,
    np.exp(b) * np.array(N_list) ** (-alpha),
    '--',
    label=f"fit α={alpha:.2f}"
)

plt.xlabel("N")
plt.ylabel("p_c(N)")
plt.title("Critical scaling of geometry breakdown")
plt.legend()
plt.grid()
plt.show()