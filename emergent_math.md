
<script type="text/javascript"
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
</script>
# Аналитические формулы из кода

| Название переменной                   | Аналитическая формула                                                                                                                                    |
|---------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Локальный квант действия**          | $$ \hbar_{\text{em}} = \frac{(\ln K)^2}{4 \lambda^2 K^2} \cdot \left[1 + \frac{1 - \frac{3(K-2)(1-p)^3}{4(K-1)}}{\ln N} \right] $$                       |
| **Постоянная Планка**                 | $$ \hbar_{\text{emergent}} = \hbar_{\text{em}} \cdot \frac{N^{-1/3}}{6\pi} $$                                                                            |
| **Диаметр Вселенной**                 | $$ R_{\text{universe}} = \frac{2\pi}{\sqrt{K p} \cdot \lambda} \cdot N^{1/6} $$                                                                          |
| **Локальный масштаб длины**           | $$ l_{\text{em}} = \frac{2\pi}{K p \lambda} \cdot N^{1/6} $$                                                                                             |
| **Планковская длина**                 | $$ \ell_{\text{P, emergent}} = \frac{1}{\sqrt{K p}} \cdot R_{\text{universe}} \cdot N^{-1/2} $$                                                          |
| **Планковское время**                 | $$ t_{\text{P, emergent}} = \lambda^2 \cdot \hbar_{\text{em}} \cdot N^{-1/3} \cdot \frac{1}{\pi} $$                                                      |
| **Планковское время (final)**         | $$ t_{\text{P, emergent\_final}} = \lambda^2 \cdot \frac{(\ln K)^2}{4 \lambda^2 K^2} \cdot N^{-1/3} \cdot \frac{1}{\pi} $$                               |
| **Скорость света**                    | $$ c_{emergent} = \pi \cdot \frac{\frac{1}{\sqrt{K p}} \cdot R_{\text{universe}}}{\frac{(\ln K)^2}{4 \lambda^2 K^2}} \cdot \frac{1}{\lambda^2} \cdot N^{-1/6} $$ |
| **Скорость света (final)**            | $$ c_{emergent_final} = \frac{8\pi^2 K (\ln N)^2}{p (\ln K)^2 \ln(pK)^2} $$                                                                              |
| **Гравитационная постоянная**         | $$ G_{\text{emergent}} = \frac{\hbar_{\text{em}}^4}{l_{\text{em}}^2} \cdot \frac{1}{\lambda^2} $$                                                        |
| **Гравитационная постоянная (final)** | $$ G_{emergent_final} = \frac{(\ln K)^8 p^2}{1024 \pi^2 \lambda^8 K^6 N^{1/3}} $$                                                                        |
| **Планковская энергия**               | $$ E_{\text{P, emergent}} = \frac{\hbar_{\text{emergent}}}{t_{\text{P, emergent}}} $$                                                                    |
| **Масса Планка (start)**              | $$ M_{planck} = \sqrt{\frac{\hbar_{\text{emergent}} \cdot c_{\text{emergent}}}{G_{\text{emergent}}}} $$                                                  |
| **Масса Планка (final)**              | $$ M_{planck-final} = \frac{32}{\sqrt{3}} \pi^{1.5} \left(\frac{\ln(K p)}{\ln N}\right)^5 \frac{K^{2.5}}{(\ln K)^4 p^{1.5}} $$                           |
| **Радиус Шварцшильда (final)**        | $$ R_{schwarzschild-final} = \frac{p^4 M_s}{32768 \pi^6 K^8 N^{1/3}} \left(\frac{\ln K \cdot \ln N}{\ln(K p)}\right)^{12} $$                             |
| **Космологическая постоянная**        | $$ \Lambda_{\text{cosmo}} = \frac{3 K p}{\pi^2 N^{1/3}} \left(\frac{\ln(K p)}{\ln N}\right)^4 $$                                                         |
| **Температура Планка**                | $$ T_{\text{plank}} = \frac{M_{planck-final} \cdot c_{\text{emergent\_final}}^2}{k_{\text{B2}}} $$                                                       |
| **Температура Планка (final)**        | $$ T_{\text{plank\_final}} = \frac{6144 \pi^{4.5}}{\sqrt{3}} \cdot \frac{\ln(K p)^7 K^6 N^{1/3}}{p^2 (\ln K)^8 (\ln N)^8} $$                             |                                                                                                                                                                                                                                                                                                                        |
| **Постоянная Больцмана (KB2)**        | $$ k_{\text{B2}} = \frac{\pi (\ln N)^7}{3 \ln(K p)^6 (p K)^{3/2} N^{1/3}} $$                                                                             |
| **Температура Хокинга**               | $$ T_{H-final} = \frac{8192 \pi^6 \ln(K p)^{12} K^{17/2} N^{1/3}}{M_s (\ln N)^{13} p^{7/2} (\ln K)^{12}} $$                                              |
| **Электрическая постоянная**          | $$ \varepsilon_{0\_emergent} = \frac{9 \lambda^2 K^{5/2} p^{7/2} N^{1/3} (\ln K)^2 \left(\ln(K p)\right)^{14}}{16 \pi^5 (\ln N)^{15}} $$                 |
| **Магнитная постоянная (test)**       | $$ \mu_{0\_test} = \frac{1}{\varepsilon_{0\_em} \cdot c_{\text{emergent}}^2} $$                                                                          |
| **Магнитная постоянная**              | $$ \mu_{0\_em} = \frac{(\ln K)^2}{14 \cdot k_{\text{B2}} \cdot K^3 \cdot \lambda^4} $$                                                                   |
| **Магнитная постоянная (result)**     | $$ \mu_{0-result} = \frac{\pi (\ln K)^2 (\ln N)^{15}}{36 K^{9/2} p^{3/2}\ln(K p)^{14} N^{1/3}} $$                                                        |
| **Постоянная тонкой структуры**       | $$ \alpha_{\text{em}} = \frac{\ln K}{\ln M} $$                                                                                                           |
| **Элементарный заряд (e_plank)**      | $$ e_{\text{plank}} = \sqrt{\frac{3 p^{5/2} K^{1.5} (\ln K)^2 \left(\ln(K p)\right)^{12}}{4 \pi^3 (\ln N)^{13}}} $$                                      |
| **Масса электрона**                   | $$ m_e = 12 \sqrt{K p} (\frac{\ln N}{\ln(K p)})^4 N^{-1/3} $$                                                                                            |
| **Радиус Бора**                       | $$ r_{bor_emergent} = \frac{\hbar_{\text{emergent}}}{m_e \cdot \alpha_{\text{em}} \cdot c_{\text{emergent}}} $$                                          |
| **Радиус Бора (final)**               | $$ r_{bor-final} = \frac{(\ln K)^3 p \ln(6N)\ln(K p)^2}{2304 \pi^3 K^3 \sqrt{K p} (\ln N)^2} $$                                                          |
| **Длина волны Комптона (электрон)**   | $$ \lambda_{\text{C,e}} = \frac{2 (\ln K)^4 \left(\ln(K p)\right)^2 \sqrt{p}}{2304 \pi^2 K^{3.5} (\ln N)^2} $$                                           |
| **Длина волны Комптона (W-бозон)**    | $$ \lambda_{\text{C,W}} = \frac{(\ln K)^6 \ln(K p)^5}{2304 K^{15/2} p^{5/2} (\ln N)^5} $$                                                                |
| **Длина волны Комптона (пи-мезон)**   | $$ \lambda_{C,π} = \frac{\ln K \cdot p^2 \cdot \ln(K p)^2}{2304 \pi^3 K^3 (\ln N)^2} $$                                                                  |

---

### Структурные функции (используются в массовых формулах):

$$ f_1 = \frac{U}{\pi} $$
$$ f_2 = \ln K $$
$$ f_3 = \sqrt{K p} $$
$$ f_4 = \frac{1}{p} $$
$$ f_5 = \frac{K}{\ln K} $$
$$ f_6 = \frac{K + pK}{K} $$

где $$ U = \frac{\ln N}{\left|\ln(K p)\right|} $$

---

### Массы элементарных частиц:

| Частица | Формула |
|---|---|
| **Мюон** | $$ m_\mu = m_e \cdot 2 \cdot f_1 $$ |
| **Тау-лептон** | $$ m_\tau = m_e \cdot \frac{f_1}{f_2^2} \cdot \frac{1}{f_3} \cdot f_4^2 \cdot \frac{1}{f_5} $$ |
| **Up-кварк** | $$ m_u = m_e \cdot f_3^2 \cdot f_4^2 \cdot \frac{1}{f_5^2} \cdot \frac{1}{f_2^2} $$ |
| **Down-кварк** | $$ m_d = m_e \cdot f_2^2 \cdot f_1 \cdot \frac{1}{f_3} \cdot \frac{1}{f_4} \cdot \frac{1}{f_5^2} \cdot f_2 $$ |
| **Strange-кварк** | $$ m_s = m_e \cdot f_1 $$ |
| **Charm-кварк** | $$ m_c = m_e \cdot f_4^2 \cdot f_5 $$ |
| **Bottom-кварк** | $$ m_b = 8 m_e \cdot f_1^2 \cdot p $$ |
| **Top-кварк** | $$ m_t = 8 m_e \cdot f_1^2 \cdot f_5 $$ |
| **Протон** | $$ m_p = m_e \cdot f_1^2 \cdot \frac{K}{f_3} \cdot \frac{1}{f_4} \cdot \frac{1}{f_5} $$ |
| **Нейтрон** | $$ m_n = m_p \cdot \left(1 + \frac{K p^2}{10}\right) $$ |
| **W-бозон** | $$ m_W = m_e \cdot f_2 \cdot f_3^2 \cdot f_5^3 \cdot f_1^3 \cdot \frac{1}{f_4^2} $$ |
| **Z-бозон** | $$ m_Z = m_e \cdot \frac{f_1^4 \cdot f_2}{f_4^2 \cdot f_5} $$ |
| **Бозон Хиггса** | $$ m_H = m_e \cdot f_1^2 \cdot \frac{f_5}{f_3} \cdot f_5 $$ |
| **Дейтрон** | $$ m_D = (m_p + m_n) \cdot \left(1 - \frac{p}{f_5}\right) $$ |
| **Альфа-частица** | $$ m_\alpha = 2 \cdot (m_p + m_n) \cdot \left(1 - \frac{4p}{f_5}\right) $$ |
| **Пи-мезон** | $$ m_\pi = m_e \cdot f_2^3 \cdot \frac{1}{f_3} \cdot f_4 $$ |
| **Kaон** | $$ m_K = m_e \cdot f_1 \cdot \frac{1}{f_4} \cdot \frac{1}{f_2} \cdot f_6^{1/2} $$ |
| **Эта-мезон** | $$ m_\eta = m_e \cdot f_2 \cdot f_4 \cdot \frac{1}{f_5} \cdot f_1 $$ |
| **Ро-мезон** | $$ m_\rho = m_e \cdot f_1^2 \cdot f_2^3 \cdot f_3^3 \cdot \frac{1}{f_4} $$ |
| **Электронное нейтрино** | $$ m_{\nu_e} = m_e \cdot \frac{1}{f_4^5} \cdot \frac{1}{f_4} $$ |
| **Мюонное нейтрино** | $$ m_{\nu_\mu} = m_e \cdot \frac{f_5}{f_4^5} \cdot \frac{1}{f_4} $$ |
| **Тау-нейтрино** | $$ m_{\nu_\tau} = m_e \cdot \frac{1}{f_2 \cdot f_4^5} $$ |

---

### Обозначения:
- $$ K $$ - степень вершин графа
- $$ p $$ - параметр вероятности связи
- $$ N $$ - число вершин графа
- $$ \lambda $$ - параметр масштаба
- $$ M $$ - дополнительный параметр (используется в $$ \alpha_{\text{em}} $$)
- $$ M_s $$ - масса объекта (для радиуса Шварцшильда)