import numpy as np

# Параметры графа
K = 8
p = 0.06

# Масштаб N
N_values = np.logspace(60, 122, num=10, base=10)

results = []

for N in N_values:
    # Масса электрона (3-циклы)
    p_3cycle = K * p
    m_e = (p_3cycle ** 3) * N ** (-1 / 6)

    # Масса W/Z (4-циклы)
    m_boson = (p_3cycle ** 2) * N ** (-1 / 6)

    # Энергия фотона (глобальное возбуждение)
    E_photon = 1 / N ** (1 / 3)

    # Гравитационная постоянная
    G = (p_3cycle ** 2) / N

    # Энтропия / Больцман
    S = N * np.log(K)

    results.append({
        'N': N,
        'm_e': m_e,
        'm_boson': m_boson,
        'E_photon': E_photon,
        'G': G,
        'S': S
    })

# Выведем результаты
for r in results:
    print(f"N={r['N']:.2e}, m_e={r['m_e']:.2e}, m_boson={r['m_boson']:.2e}, "
          f"E_photon={r['E_photon']:.2e}, G={r['G']:.2e}, S={r['S']:.2e}")
