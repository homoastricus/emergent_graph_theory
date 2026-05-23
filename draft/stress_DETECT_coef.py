import math
import numpy as np

gamma_E = 0.5772156649015329

alpha = 22.960202
beta = -5.527557

def lambda_k(k):
    return math.exp(alpha) * (k + gamma_E) ** (-beta)

def A_from_spectrum(n):
    return sum(math.log(lambda_k(k)) for k in range(1, n+1))

# тест
test_n = [10, 11, 12]

for n in test_n:
    print(f"n = {n}, A(n) = {A_from_spectrum(n):.4f}")