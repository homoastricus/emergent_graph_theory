import numpy as np

def lambda_emergent(N, K, p):
    """ Эмерджентный спектральный масштаб λ(N, K, p). """
    N = float(N)   # ключевая строка!
    return (np.log(K * p) / np.log(N)) ** 2

# пример:
K = 8
p = 0.06
N = 10**122

lam = lambda_emergent(N, K, p)
print(lam)
