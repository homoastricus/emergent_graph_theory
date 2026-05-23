import math

# Параметры ГИП
K = 6.0
N = 4.179e121
lnN = math.log(N)
N_13 = N**(1/3)

# В СИ
h_SI = 6.62607015e-34
print(f"h_SI = {h_SI:.8e} Дж·с  (по определению СИ с 2019)")


p = 1.8130170251248464e-40
C_test = math.sqrt(p/(1+(K**2) * p))
print(f"C_test = {C_test*h_SI}  (безразмерное, ~1)")

print(f"C_test_2pi = {C_test*h_SI/(2*math.pi)} ")


print(f"C_test_norm = {h_SI/(C_test*N_13)} ")
# Графовое значение h (безразмерное)
h_graph = (lnN**3) / (K * N_13)
print(f"h_graph = {h_graph}  (безразмерное, ~1)")



# В планковских единицах
h_planck = 1.0
print(f"h_planck = {h_planck}  (по определению)")


# Калибровочный коэффициент
C_action = h_SI / h_graph
print(f"\nКалибровочный коэффициент C_action = {C_action:.8e} Дж·с")
print(f"Это просто переводной множитель между графовыми единицами и СИ!")