import math
import warnings

warnings.filterwarnings('ignore')

K = 6.0
pi = math.pi
lnK = math.log(K)
lnN_math = 280.11158 # выведено из геометрического резонанса
lnN_phys = math.log(4.183e121)
dif = lnN_math - lnN_phys
res = math.exp(dif)
print(f"res={res:.10f}")

test = lnN_math - lnN_phys - 18.96/lnN_math
#test = math.exp(test)
print(f"test={test:.10f}")

H_emer = lnN_phys**3/(K*math.exp(lnN_phys)**(1/3))
print(f"H_emer={H_emer:.50}")

print(f"pi*K/lnN_math={pi*K/lnN_math:.10f}")

print(f"pi*lnN={pi*lnN_phys:.10f}")
print(f"pi*lnN^2={pi*lnN_phys**2:.10f}")
print(f"pi*lnN^3={pi*lnN_phys**3:.10f}")
print(f"pi*lnN^4={pi*lnN_phys**4:.10f}")
print(f"pi*lnN^5={pi*lnN_phys**5:.10f}")


#K * (K + 1.0/lnN ) / (K + 1.0/math.log(lnK)) = delta
#K + 1.0/lnN = (1 + 1.0/(K*math.log(lnK))) * delta
#1.0/lnN = (1 + 1.0/(K*math.log(lnK))) * delta - K
#lnN = 1/(1 + 1.0/(K*math.log(lnK))) * delta - K)
#N= exp(1/(1 + 1.0/(K*math.log(lnK))) * delta - K))

#lnN = (K - ln K) / (1/pi-1/3) = 1/(1 + 1.0/(K*math.log(lnK))) * delta - K)

#(K - ln K) / ln N=1/pi-1/3
#lnN = (K - ln K) / (1/pi-1/3)

def identity_feigenbaum_ratio(lnN):
    """K * (K + 1/lnN) / (K + 1/lnlnK) ≈ feigenbaum_delta"""
    return K * (K + 1.0/lnN ) / (K + 1.0/math.log(lnK))

feigenbaum_delta=4.669201609102990
res = feigenbaum_delta -  identity_feigenbaum_ratio(lnN_math)
print(f"res={res:.10f}")

#p = 1.0 / (K * N ** (1 / 3))

# геометрический резонанс
#1/pi = 1/3 - (K-lnK)/lnN_math + fix/lnN_math
#1/pi - 1/3 = (K-lnK)/lnN_math
ln_math = (K-lnK)/(1/pi - 1/3)
#0 = lnN_math/pi  - lnN_math/3 + (K-lnK)
print(f"ln_math={ln_math:.10f}")

# рассчет для N_phys с остатком
# 1/pi = 1/3 - (K-lnK)/lnN_phys + fix/lnN_phys
# 1/pi - 1/3 + (K-lnK)/lnN_phys = fix/lnN_phys
fix  =  lnN_phys*(1/pi - 1/3) + (K-lnK)
print(f"fix={fix:.10f}")