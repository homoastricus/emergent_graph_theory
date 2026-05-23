import math
import numpy as np


# ============================================================
# ФУНДАМЕНТАЛЬНЫЕ КОНСТАНТЫ
# ============================================================
pi = math.pi
e = math.e
K = 6.0
lnK = math.log(K)

N = 4.197668e121
lnN = math.log(N)
N13 = N ** (1/3)

s_3 = math.sqrt(3)
s_2 = math.sqrt(2)
N_start =np.exp(math.pi * math.log(K)) #m math.exp(
N=np.exp(math.pi * math.log(K)) #m math.exp(

N_reso= 280.111514
N_djet=280.0492258637
x = pi**2*lnK #17.444
dif = N_reso - N_djet - pi**2* lnK/(N_djet) - 0 /N_djet**2 - 54*0/N_djet**3
print(x)
t = N_reso - N_djet
print(f"N_reso - N_djet={dif:.6e}")
#
print(f"N={N:.6e}")
#
# N_res=np.exp(N)
# print(f"N_res={N_res:.6e}")
