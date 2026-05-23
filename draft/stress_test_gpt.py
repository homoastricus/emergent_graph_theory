import numpy as np
import networkx as nx
from scipy.sparse.linalg import eigsh
from collections import defaultdict

# =========================
# ГРАФ
# =========================
def generate_graph(N=5000, K=6):
    p = 1 / (K* N**0.33333)  # можешь заменить на свою модель
    G = nx.erdos_renyi_graph(N, p, seed=None)

    # берём крупнейшую компоненту
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()

    return G


# =========================
# СПЕКТР
# =========================
def compute_spectrum(G, num_eigs=120):
    L = nx.laplacian_matrix(G).astype(float)
    vals, vecs = eigsh(L, k=num_eigs, which='SM')
    idx = np.argsort(vals)
    return vals[idx], vecs[:, idx]


# =========================
# LDOS / CLUSTER веса
# =========================
def compute_A_k(eigvecs, mode='ldos', vertex=0, cluster=None):
    if mode == 'ldos':
        return np.abs(eigvecs[vertex, :])**2

    elif mode == 'cluster':
        A = np.zeros(eigvecs.shape[1])
        for v in cluster:
            A += np.abs(eigvecs[v, :])**2
        return A

    else:
        raise ValueError("Unknown mode")


# =========================
# ЭНЕРГИЯ
# =========================
def compute_energy(eigvals, A_k, lam, t, sigma, gamma=1.0):
    return t * eigvals + ((eigvals - lam)**2) / sigma - gamma * A_k


# =========================
# SOFTMAX
# =========================
def softmax(x, beta):
    x = -beta * x
    x -= np.max(x)  # стабильность
    e = np.exp(x)
    return e / np.sum(e)


# =========================
# FIXED POINT
# =========================
def find_lambda_star(eigvals, A_k, t, sigma, beta,
                     gamma=1.0,
                     lam_init=0.1,
                     max_iter=100,
                     tol=1e-6):

    lam = lam_init

    for _ in range(max_iter):
        E = compute_energy(eigvals, A_k, lam, t, sigma, gamma)
        p = softmax(E, beta)
        lam_new = np.sum(p * eigvals)

        if abs(lam_new - lam) < tol:
            return lam_new

        lam = lam_new

    return lam


# =========================
# КЛАСТЕР
# =========================
def get_cluster(G, center, size=10):
    nodes = list(nx.bfs_tree(G, center, depth_limit=3))
    return nodes[:size]


# =========================
# ОСНОВНОЙ ЭКСПЕРИМЕНТ
# =========================
def run_experiment():

    # параметры
    N = 5000
    K = 6
    NUM_EIGS = 120

    t_list = [0.5, 1.0, 2.0]
    sigma_list = [0.05, 0.1, 0.5]
    beta_list = [0.5, 1, 2, 5, 10, 20, 50]

    n_inits = 5

    # граф
    G = generate_graph(N, K)
    eigvals, eigvecs = compute_spectrum(G, NUM_EIGS)

    # точка
    vertex = np.random.randint(0, len(G))
    cluster = get_cluster(G, vertex, size=12)

    print(f"Vertex: {vertex}, cluster size: {len(cluster)}")

    results = []

    for mode in ['ldos', 'cluster']:

        A_k = compute_A_k(eigvecs, mode=mode,
                          vertex=vertex,
                          cluster=cluster)

        for t in t_list:
            for sigma in sigma_list:
                for beta in beta_list:

                    lambdas = []

                    for _ in range(n_inits):
                        lam0 = np.random.uniform(0, np.max(eigvals))
                        lam_star = find_lambda_star(
                            eigvals, A_k,
                            t, sigma, beta,
                            lam_init=lam0
                        )
                        lambdas.append(lam_star)

                    lambdas = np.array(lambdas)

                    mean = np.mean(lambdas)
                    std = np.std(lambdas)

                    # считаем число уникальных решений
                    unique = np.unique(np.round(lambdas, 4))
                    n_unique = len(unique)

                    results.append((mode, t, sigma, beta, mean, std, n_unique))

                    print(f"[mode={mode}] t={t}, σ={sigma}, β={beta} "
                          f"→ λ*≈{mean:.4f}, std={std:.4e}, states={n_unique}")

    return results


# =========================
# ЗАПУСК
# =========================
if __name__ == "__main__":
    results = run_experiment()