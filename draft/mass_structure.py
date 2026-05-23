import numpy as np
import random
from itertools import product
from collections import defaultdict

# ==========================================
# НАСТРОЙКИ
# ==========================================

K = 6

P_VALUES = [-3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3]
Q_VALUES = [-3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3]

LAMBDA_COMPLEXITY = 0.1  # штраф

N_PERM = 2000

# ==========================================
# ВСЕ ФОРМУЛЫ МАСС ЧАСТИЦ (из ЕТИ)
# ==========================================

masses = {
    # Лептоны
    "e": 0.511,  # МэВ
    "mu": 105.66,
    "tau": 1776.86,

    # Кварки (конституэнтные массы, МэВ)
    "u": 336,
    "d": 340,
    "s": 486,
    "c": 1550,
    "b": 4730,
    "t": 173000,

    # Барионы (МэВ)
    "proton": 938.27,
    "neutron": 939.57,
    "Lambda": 1115.68,
    "Sigma_plus": 1189.37,
    "Sigma_minus": 1197.45,
    "Sigma0": 1192.64,
    "Ksi_0": 1314.86,
    "Ksi_minus": 1321.71,
    "Ksi_plus": 1318.0,  # ~ оценка
    "Omega_minus": 1672.45,
    "Omega0_c": 2695.2,
    "Lambda0_b": 5619.6,
    "Lambda_plus_c": 2286.46,

    # Мезоны (МэВ)
    "pi_plus": 139.57,
    "pi0": 134.98,
    "K0": 497.61,
    "K_plus": 493.68,
    "D0": 1864.84,
    "J_psi": 3096.9,
    "eta": 547.86,
    "Upsilon_1S": 9460.3,
    "phi": 1019.46,
    "omega": 782.66,
    "eta_prime": 957.78,

    # Бозоны (МэВ)
    "W": 80379,
    "Z": 91187.6,
    "Higgs": 125100,
}

spins = {
    # Лептоны
    "e": 0.5, "mu": 0.5, "tau": 0.5,

    # Кварки
    "u": 0.5, "d": 0.5, "s": 0.5, "c": 0.5, "b": 0.5, "t": 0.5,

    # Барионы
    "proton": 0.5, "neutron": 0.5, "Lambda": 0.5,
    "Sigma_plus": 0.5, "Sigma_minus": 0.5, "Sigma0": 0.5,
    "Ksi_0": 0.5, "Ksi_minus": 0.5, "Ksi_plus": 0.5,
    "Omega_minus": 1.5, "Omega0_c": 0.5,
    "Lambda0_b": 0.5, "Lambda_plus_c": 0.5,

    # Мезоны
    "pi_plus": 0, "pi0": 0, "K0": 0, "K_plus": 0,
    "D0": 0, "J_psi": 1, "eta": 0, "Upsilon_1S": 1,
    "phi": 1, "omega": 1, "eta_prime": 0,

    # Бозоны
    "W": 1, "Z": 1, "Higgs": 0,
}


# ==========================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==========================================

def is_fermion(spin):
    return spin % 1 != 0


def predict_ratio(p, q):
    return (K ** p) * (np.pi ** q)


def log_error(r_real, r_pred):
    return abs(np.log(r_real / r_pred))


def score_function(err, p, q):
    return err + LAMBDA_COMPLEXITY * (abs(p) + abs(q))


# ==========================================
# ПОИСК ЛУЧШЕЙ ФОРМУЛЫ
# ==========================================

def find_best_pq(ratio):
    best = None

    for p, q in product(P_VALUES, Q_VALUES):
        r_pred = predict_ratio(p, q)
        err = log_error(ratio, r_pred)
        score = score_function(err, p, q)

        if best is None or score < best[0]:
            best = (score, p, q, err)

    return best  # (score, p, q, err)


# ==========================================
# СЧИТАЕМ p ДЛЯ ВСЕХ ПАР
# ==========================================

def compute_p_signs(masses):
    results = []

    names = list(masses.keys())

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]

            ratio = masses[a] / masses[b]

            _, p, q, err = find_best_pq(ratio)

            results.append({
                "pair": (a, b),
                "p": p,
                "q": q,
                "sign": np.sign(p),
                "err": err
            })

    return results


# ==========================================
# МЕТРИКА ТОЧНОСТИ
# ==========================================

def compute_accuracy(results, spins, use_sign=True):
    correct = 0
    total = 0

    for r in results:
        a, b = r["pair"]

        if use_sign:
            sign = r["sign"]
            if sign == 0:
                continue
            fermion_a = is_fermion(spins[a])
            predicted_fermion = (sign < 0)
            if predicted_fermion == fermion_a:
                correct += 1
        else:
            # Проверка через q: q ≥ 0 для целого спина
            q = r["q"]
            if q == 0:
                continue
            boson_a = not is_fermion(spins[a])
            predicted_boson = (q >= 0)
            if predicted_boson == boson_a:
                correct += 1

        total += 1

    return correct / total if total > 0 else 0


# ==========================================
# ПЕРМУТАЦИОННЫЙ ТЕСТ
# ==========================================

def permutation_test(masses, spins):
    real_results = compute_p_signs(masses)
    A_real = compute_accuracy(real_results, spins)

    perm_scores = []

    names = list(masses.keys())
    values = list(masses.values())

    for _ in range(N_PERM):
        random.shuffle(values)
        perm_masses = dict(zip(names, values))

        results = compute_p_signs(perm_masses)
        A_perm = compute_accuracy(results, spins)

        perm_scores.append(A_perm)

    p_value = np.mean([x >= A_real for x in perm_scores])

    return A_real, perm_scores, p_value


# ==========================================
# НУЛЕВАЯ МОДЕЛЬ
# ==========================================

def random_model_test(spins, n_samples=100):
    accs = []

    names = list(spins.keys())

    for _ in range(n_samples):
        masses_rand = {
            name: np.exp(np.random.uniform(0, 5))
            for name in names
        }

        results = compute_p_signs(masses_rand)
        acc = compute_accuracy(results, spins)

        accs.append(acc)

    return np.mean(accs), np.std(accs)


# ==========================================
# TRAIN / TEST
# ==========================================

def train_test_split_dict(masses, spins, test_ratio=0.3):
    names = list(masses.keys())
    random.shuffle(names)

    split = int(len(names) * (1 - test_ratio))

    train_names = names[:split]
    test_names = names[split:]

    masses_train = {k: masses[k] for k in train_names}
    masses_test = {k: masses[k] for k in test_names}

    spins_train = {k: spins[k] for k in train_names}
    spins_test = {k: spins[k] for k in test_names}

    return masses_train, masses_test, spins_train, spins_test


# ==========================================
# ЗАПУСК
# ==========================================

if __name__ == "__main__":

    print("=" * 70)
    print("ПРОВЕРКА ПРАВИЛА: ЗНАК K^p ↔ СТАТИСТИКА")
    print("=" * 70)
    print(f"\n  Всего частиц: {len(masses)}")
    print(f"  Фермионов: {sum(1 for s in spins.values() if is_fermion(s))}")
    print(f"  Бозонов: {sum(1 for s in spins.values() if not is_fermion(s))}")
    print(f"  Пар для анализа: {len(masses) * (len(masses) - 1) // 2}")
    print()

    print("=== 1. ПРАВИЛО ЗНАКА K^p ===")
    results = compute_p_signs(masses)
    acc_sign = compute_accuracy(results, spins, use_sign=True)
    print(f"  Точность (знак K^p → фермион/бозон): {acc_sign:.4f}")

    # Детали: какие пары нарушают правило
    violations = []
    for r in results:
        a, b = r["pair"]
        if r["sign"] != 0:
            fermion_a = is_fermion(spins[a])
            fermion_b = is_fermion(spins[b])
            predicted_a_fermion = (r["sign"] < 0)
            predicted_b_fermion = (r["sign"] > 0)

            if predicted_a_fermion != fermion_a or predicted_b_fermion != fermion_b:
                violations.append(r)

    if violations:
        print(f"\n  Нарушения ({len(violations)}/{len(results)}):")
        for v in violations[:10]:
            a, b = v["pair"]
            print(f"    {a} ({spins[a]:.1f}) vs {b} ({spins[b]:.1f}): "
                  f"p={v['p']}, sign={'+' if v['sign'] > 0 else '-'}")

    print("\n=== 2. ПРАВИЛО ЗНАКА π^q ===")
    acc_q = compute_accuracy(results, spins, use_sign=False)
    print(f"  Точность (знак π^q → фермион/бозон): {acc_q:.4f}")

    print("\n=== 3. ПЕРМУТАЦИОННЫЙ ТЕСТ (ЗНАК K^p) ===")
    A_real, perm_scores, p_val = permutation_test(masses, spins)
    print(f"  Реальная точность: {A_real:.4f}")
    print(f"  Средняя случайная: {np.mean(perm_scores):.4f} ± {np.std(perm_scores):.4f}")
    print(f"  p-value: {p_val:.6f}")

    if p_val < 0.001:
        print(f"  ✅ ВЫСОКАЯ ЗНАЧИМОСТЬ! Правило не случайно.")
    elif p_val < 0.05:
        print(f"  🟡 ЗНАЧИМО на уровне 5%.")
    else:
        print(f"  ❌ НЕ ЗНАЧИМО. Правило может быть случайным.")

    print("\n=== 4. НУЛЕВАЯ МОДЕЛЬ (СЛУЧАЙНЫЕ МАССЫ) ===")
    mean_acc, std_acc = random_model_test(spins)
    print(f"  Случайные массы: точность = {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"  Отличие от реальной: {(A_real - mean_acc) / std_acc:.1f} сигм")

    print("\n=== 5. TRAIN / TEST ===")
    m_tr, m_te, s_tr, s_te = train_test_split_dict(masses, spins)

    res_tr = compute_p_signs(m_tr)
    res_te = compute_p_signs(m_te)

    acc_tr = compute_accuracy(res_tr, s_tr)
    acc_te = compute_accuracy(res_te, s_te)

    print(f"  Train accuracy: {acc_tr:.4f}")
    print(f"  Test accuracy:  {acc_te:.4f}")

    if abs(acc_tr - acc_te) < 0.1:
        print(f"  ✅ Правило устойчиво: train ≈ test")
    else:
        print(f"  ⚠️ Переобучение? Разница > 10%")

    print("\n" + "=" * 70)
    print("ВЫВОД")
    print("=" * 70)

    if p_val < 0.01 and A_real > 0.7:
        print(f"""
    ПРАВИЛО ПОДТВЕРЖДЕНО СТАТИСТИЧЕСКИ:

    1. Знак K^p предсказывает статистику (фермион/бозон)
       с точностью {A_real:.1%} (p = {p_val:.4f}).

    2. Это НЕ случайное совпадение:
       — Случайные массы дают точность {mean_acc:.1%}
       — Отличие на {(A_real - mean_acc) / std_acc:.0f} сигм

    3. Правило работает на независимых данных:
       — Train: {acc_tr:.1%}
       — Test:  {acc_te:.1%}

    ФИЗИЧЕСКИЙ СМЫСЛ:
    K^p < 0 → полуцелый спин → ФЕРМИОНЫ (вещество)
    K^p ≥ 0 → целый спин → БОЗОНЫ (поля)

    Это СТРУКТУРНОЕ ПРАВИЛО, закодированное в формулах масс.
    """)
    else:
        print(f"""
    Правило требует дальнейшей проверки.
    Точность = {A_real:.1%}, p-value = {p_val:.4f}.

    Возможно, нужно уточнить модель или увеличить выборку.
    """)