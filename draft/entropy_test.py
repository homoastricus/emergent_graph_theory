from mpmath import mp

# Устанавливаем точность
mp.dps = 50  # 50 десятичных знаков

N = mp.mpf('4.179e121')
M = mp.e ** N
print(f"M ≈ {M}")