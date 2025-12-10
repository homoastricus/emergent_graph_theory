# Ваши значения
m_e = 9.0978e-31
alpha = 0.007297
c = 2.9800e+08

E_ion_from_alpha = 0.5 * alpha**2 * m_e * c**2
E_ion_eV_from_alpha = E_ion_from_alpha / 1.602e-19

print(f"E_ion из α²mc²/2: {E_ion_eV_from_alpha:.2f} эВ")
# Должно быть ~13.6 эВ