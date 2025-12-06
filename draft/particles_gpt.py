import math
import pandas as pd

# ------------------------------
# Параметры модели
# ------------------------------
K = 8.0
p = 5.270179e-02
N = 9.702e+122
lnK = math.log(K)
lnKp = math.log(K * p)
lnN = math.log(N)
U = lnN / abs(lnKp)

# ------------------------------
# Базовая масса электрона
# ------------------------------
m_e_base = 12 * math.sqrt(K * p) * (U ** 4) * (N ** (-1 / 3))
m_e_exp = 9.1093837e-31

# ------------------------------
# Экспериментальные массы (кг)
# ------------------------------
masses_exp = {
    "electron": 9.109e-31,
    "muon": 1.884e-28,
    "tau": 3.168e-27,
    "up": 2.16e-30,
    "down": 4.67e-30,
    "strange": 9.34e-29,
    "charm": 1.27e-27,
    "bottom": 4.18e-27,
    "top": 1.731e-25,
    "proton": 1.673e-27,
}

# ------------------------------
# Структурные функции
# ------------------------------
f1 = U / math.pi
f2 = lnK
f3 = math.sqrt(K * p)
f4 = 1 / p
f5 = K / lnK

# ------------------------------
# Универсальная таблица формул
# ------------------------------
particles = {
    "electron":  {"dU": 4, "n_pi": 0, "n_p": 0, "n_lnK": 0, "n_KlnK": 0, "desc": "базовая мода"},
    "muon":      {"dU": 5, "n_pi": 1, "n_p": 0, "n_lnK": 0, "n_KlnK": 0, "coeff": 2, "desc": "вторая гармоника"},
    "tau":       {"dU": 5, "n_pi": 1, "n_p": 1, "n_lnK": 0, "n_KlnK": 0, "desc": "третья гармоника"},
    "proton":    {"dU": 5, "n_pi": 1, "n_p": 0, "n_lnK": 1, "n_KlnK": 1, "desc": "композитная мода"},
    "up":        {"dU": 4, "n_pi": 1, "n_p": 0, "n_lnK": -1, "n_KlnK": 0, "desc": "кварк 1-го поколения"},
    "down":      {"dU": 4, "n_pi": 1, "n_p": 0, "n_lnK": 0, "n_KlnK": 0, "desc": "кварк 1-го поколения"},
    "strange":   {"dU": 4, "n_pi": 1, "n_p": 0, "n_lnK": 1, "n_KlnK": 0, "desc": "кварк 2-го поколения"},
    "charm":     {"dU": 5, "n_pi": 1, "n_p": 0, "n_lnK": 2, "n_KlnK": 0, "desc": "кварк 2-го поколения"},
    "bottom":    {"dU": 5, "n_pi": 1, "n_p": 1, "n_lnK": 0, "n_KlnK": 0, "desc": "кварк 3-го поколения"},
    "top":       {"dU": 6, "n_pi": 2, "n_p": 0, "n_lnK": -1, "n_KlnK": 1, "desc": "высшая гармоника"},
}

# ------------------------------
# Вычисление
# ------------------------------
data = []
for particle, exps in particles.items():
    coeff = exps.get("coeff", 1.0)
    mass_model = (
        coeff
        * 12
        * math.sqrt(K * p)
        * (U ** exps["dU"])
        * (N ** (-1 / 3))
        / (math.pi ** exps["n_pi"])
        / (p ** exps["n_p"])
        * (lnK ** exps["n_lnK"])
        * ((K / lnK) ** exps["n_KlnK"])
    )
    m_exp = masses_exp.get(particle, None)
    if m_exp:
        error = abs(mass_model / m_exp - 1) * 100
        ratio = mass_model / m_exp
    else:
        error = None
        ratio = None
    data.append(
        {
            "Частица": particle,
            "Модельная масса (кг)": mass_model,
            "Эксп. масса (кг)": m_exp,
            "Отношение": ratio,
            "Ошибка (%)": error,
            "Гармоника U": exps["dU"],
            "Описание": exps["desc"],
        }
    )

df = pd.DataFrame(data)
print(df.to_string(index=False, float_format=lambda x: f"{x:.3e}"))
