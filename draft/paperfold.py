import math

K = 6.0
lnK = math.log(K)
ln3 = math.log(3)
paperfolding = 0.8507361882
target = ln3 * paperfolding

def S_eff(lnN):
    x = lnN ** (1/3)
    return lnK * x / (K + x)

# Тест при разных N
test_lnN = [280.0, 280.0473, 280.1, 280.5, 281.0, 285.0, 300.0]
for lnN in test_lnN:
    S = S_eff(lnN)
    delta = abs(S - target) / target * 100
    print(f"lnN={lnN:.4f}: S={S:.10f}, отклонение={delta:.6f}%")