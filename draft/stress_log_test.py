import numpy as np
import networkx as nx

def compute_spectral_quantities(N, K=6):
    p = 1 / (K * N**(1/3))

    G = nx.watts_strogatz_graph(N, K, p)

    L = nx.laplacian_matrix(G).astype(float).toarray()

    eigvals = np.linalg.eigvalsh(L)

    # убираем нулевое собственное
    eigvals = eigvals[1:]

    lambda1 = eigvals[0]

    S = np.sum(1.0 / eigvals)

    return lambda1, S


for N in [200, 400, 800, 1600]:
    lam1, S = compute_spectral_quantities(N)

    print(f"N={N}")
    print("lambda1 =", lam1)
    print("lambda1 * ln N =", lam1 * np.log(N))
    print("S =", S)
    print("S / ln N =", S / np.log(N))
    print("-" * 40)