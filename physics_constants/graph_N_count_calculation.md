ТОЧНЫЕ АНАЛИТИЧЕСКИЕ ФОРМУЛЫ ДЛЯ $\ln N$
================================================================================

$$
\ln N_{\zeta} = 6^{\frac{3}{2} + \zeta(2)} = 6^{1.5 + \frac{\pi^2}{6}}
               = 280.0492258637
$$

$$
\ln N_{\text{geometric}} = \ln N_{\zeta} + \frac{2\pi^2}{\ln N_{\zeta}}
                         = 280.049226 + 0.070485
                         = 280.1197106463
$$
$( \text{эталон: } 280.119711 )$

$$
\ln N_{\text{phys}} = \ln N_{\zeta} - \frac{\pi^6}{\sqrt{6} \cdot (\ln N_{\zeta})^2}
                    = 280.049226 - 0.005004
                    = 280.0442214310
$$
$( \text{эталон: } \approx 280.0445 )$

$$
\ln N_{\alpha} = \ln N_{\text{phys}} + \frac{ \frac{1}{\sqrt{K}} + \frac{1}{\ln^2 N} }{2\pi^2}
               = 280.044221 + 0.020683
               = 280.0649041769
$$
$$
\alpha = \frac{2 \ln^2 K}{\pi \cdot \ln N_{\alpha}} = 0.0072976134
$$
$( \text{CODATA: } 1/137.036 \approx 0.007297 )$

$$
\ln N_{\hbar} = \ln N_{\zeta} - \text{geometric\_zeta} \cdot \frac{\ln K - \frac{K}{\ln N}}{K \cdot \ln^2 K}
              = 280.049226 - 0.006478
              = 280.0427478767
$$

$$
\ln N_c = \ln N_{\zeta} - \text{zeta\_lightspeed}
        = 279.9569887368
$$
$$
c = \frac{\pi \cdot (\ln N_c)^4}{K^2 \cdot \ln K} = 299180247.89 \ \text{м/с}
$$
$( \text{CODATA: } 299792458 \ \text{м/с} )$

---

### СВОДКА: ВСЕ $\ln N$ ВЫРАЖЕНЫ ЧЕРЕЗ $\ln N_{\zeta}$

$$
\ln N_{\zeta} = 6^{\frac{3}{2} + \frac{\pi^2}{6}} = 280.0492258637
$$

$$
\ln N_{\text{geometric}} = \ln N_{\zeta} + \frac{2\pi^2}{\ln N_{\zeta}} = 280.1197106463
$$

$$
\ln N_{\text{phys}} = \ln N_{\zeta} - \frac{\pi^6}{\sqrt{6} \cdot (\ln N_{\zeta})^2} = 280.0442214310
$$

$$
\ln N_{\alpha} = \ln N_{\text{phys}} + \frac{ \frac{1}{\sqrt{K}} + \frac{1}{\ln^2 N} }{2\pi^2} = 280.0649041769
$$

$$
\ln N_{\hbar} = \ln N_{\zeta} - \text{geom\_zeta} \cdot \frac{\ln K - \frac{K}{\ln N}}{K \cdot \ln^2 K} = 280.0427478767
$$

$$
\ln N_c = \ln N_{\zeta} - \text{zeta\_lightspeed} = 279.9569887368
$$

---

### ПРОВЕРКА СОГЛАСОВАННОСТИ

Разброс значений $\ln N$:

| Параметр          | Значение      |
|-------------------|---------------|
| $\ln N_{\zeta}$   | 280.049226    |
| $\ln N_{\text{geom}}$ | 280.119711 |
| $\ln N_{\text{phys}}$ | 280.044221 |
| $\ln N_{\alpha}$  | 280.064904    |
| $\ln N_{\hbar}$   | 280.042748    |
| $\ln N_c$         | 279.956989    |

- **Среднее:** 280.046300
- **Стандартное отклонение:** 0.047849
- **Разброс (max − min):** 0.162722
- **Относительный разброс:** 0.0581%