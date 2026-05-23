"""
ИСПРАВЛЕННЫЕ: fit_N и formulas_with_params с защитой от деления на ноль
"""

import math
import numpy as np
from scipy.optimize import minimize_scalar
import random

# ========== ПАРАМЕТРЫ ==========
K_true = 6.0
pi_true = math.pi
lnK_true = math.log(K_true)
N0 = 4.198e121
lnN0 = math.log(N0)

# ========== ВСЕ КОНСТАНТЫ ==========
ALL_CONSTANTS = {
    'c': 299792458, 'ħ': 1.054571817e-34, 'G': 6.67430e-11,
    'k_B': 1.380649e-23, 'α': 1/137.035999084,
    'm_e': 9.1093837015e-31, 'm_proton': 1.67262192e-27,
    'm_muon': 1.883531627e-28, 'm_tau': 3.167e-27,
    'm_pi_meson': 2.4880888e-28, 'm_z_bozon': 1.62614e-25,
    'm_w_bozon': 1.43362e-25, 'm_Higgs': 2.23319e-25,
    'Lambda_cosmo': 1.08929e-52, 'vacuum_higgs': 4.388471e-25,
    'mu_lifetime': 2.1969811e-6, 'tau_lifetime': 2.903e-13,
    'pion_lifetime': 2.6033e-8, 'neutron_lifetime': 877.8,
}

# ========== ФОРМУЛЫ ЕТИ (С ЗАЩИТОЙ) ==========
def formulas_with_params(N, K_val, pi_val, lnK_val):
    """
    Формулы ЕТИ с заданными параметрами.
    ЗАЩИТА: если K_val <= 0 или K_val == 1, возвращаем заглушку с огромной ошибкой.
    """
    # Проверка на валидность параметров
    if K_val <= 0 or K_val == 1.0:
        # Возвращаем словарь с нулевыми значениями — вызовет большую ошибку
        return {k: 0.0 for k in ALL_CONSTANTS.keys()}

    if pi_val <= 0:
        return {k: 0.0 for k in ALL_CONSTANTS.keys()}

    if lnK_val == 0:
        return {k: 0.0 for k in ALL_CONSTANTS.keys()}

    try:
        lnN = math.log(N)
    except (ValueError, OverflowError):
        return {k: 0.0 for k in ALL_CONSTANTS.keys()}

    if lnN <= 0:
        return {k: 0.0 for k in ALL_CONSTANTS.keys()}

    N13 = N ** (1/3) if N > 0 else 0
    if N13 == 0:
        return {k: 0.0 for k in ALL_CONSTANTS.keys()}

    Kp = K_val / (K_val * N13) if N13 != 0 else 0

    try:
        return {
            'ħ': (lnN**3) / (K_val * N13),
            'c': pi_val * (lnN**4) / (K_val**2 * lnK_val),
            'G': 16 * pi_val**3 * lnN**13 / (K_val**5 * lnK_val * N13),
            'k_B': Kp * (lnN**8) / (8 * pi_val**2),
            'α': 2 * lnK_val**2 / (pi_val * lnN),
            'm_e': 4*pi_val * lnN**4 / (K_val**0.5 * N13),
            'm_proton': pi_val**0.5 * lnN**6 / (K_val**1.5 * N13),
            'm_muon': 4*pi_val**2 * lnN**5 / (K_val * 3**0.5 * N13),
            'm_tau': pi_val**0.5 * lnN**5 * K_val**2 / N13,
            'm_pi_meson': lnN**6 / (4*pi_val**2 * 2**0.5) / N13,
            'm_z_bozon': lnN**6 * 4 * pi_val**2.5 / (N13 * K_val),
            'm_w_bozon': 2 * pi_val**3 * lnN**6 / (N13 * K_val),
            'm_Higgs': lnN**6 * 4 * pi_val**2 / (N13 * K_val**0.5),
            'vacuum_higgs': lnN**6 * 8 * pi_val**1.5 / (2**0.5 * N13),
            'Lambda_cosmo': lnN**12 / (pi_val**0.5 * N**(2/3)),
            'mu_lifetime': lnK_val / (K_val * 3**0.5 * lnN**2),
            'tau_lifetime': 1/(2 * lnN**5),
            'pion_lifetime': K_val**2 * 2**0.5 * pi_val / lnN**4,
            'neutron_lifetime': 2**0.5 * N**(1/12) / lnN**3,
        }
    except (ZeroDivisionError, OverflowError, ValueError):
        return {k: 0.0 for k in ALL_CONSTANTS.keys()}


def formulas_default(N):
    return formulas_with_params(N, K_true, pi_true, lnK_true)


def fit_N(keys, N_start, K_val, pi_val, lnK_val):
    """Подгонка N с защитой"""
    def error(ln_N):
        try:
            N_val = math.exp(ln_N)
        except OverflowError:
            return 1e100

        pred = formulas_with_params(N_val, K_val, pi_val, lnK_val)
        err = 0.0
        for k in keys:
            if k in pred and ALL_CONSTANTS[k] > 0:
                if pred[k] <= 0:
                    err += 1e6  # большой штраф за невалидное предсказание
                else:
                    ratio = pred[k] / ALL_CONSTANTS[k]
                    if ratio > 0:
                        err += (math.log(ratio))**2
                    else:
                        err += 1e6
        return err if math.isfinite(err) else 1e100

    try:
        result = minimize_scalar(
            error,
            bounds=(lnN0*0.3, lnN0*3.0),  # расширенные границы
            method='bounded'
        )
        return math.exp(result.x), result.fun
    except Exception:
        return N0, 1e100


# ================================================================
# ТЕСТ Z: OUT-OF-DISTRIBUTION
# ================================================================
def Z_out_of_distribution():
    print("="*70)
    print("ТЕСТ Z: OUT-OF-DISTRIBUTION — ФИТ НА 50%, ТЕСТ НА 50%")
    print("="*70)

    all_keys = list(ALL_CONSTANTS.keys())
    n_total = len(all_keys)
    n_train = n_total // 2

    n_trials = 100
    train_errors = []
    test_errors = []
    N_values = []

    random.seed(42)
    np.random.seed(42)

    for trial in range(n_trials):
        shuffled = all_keys.copy()
        random.shuffle(shuffled)
        train_keys = shuffled[:n_train]
        test_keys = shuffled[n_train:]

        N_fit, train_err = fit_N(train_keys, N0, K_true, pi_true, lnK_true)

        pred = formulas_default(N_fit)
        test_err = 0.0
        for k in test_keys:
            if k in pred and pred[k] > 0 and ALL_CONSTANTS[k] > 0:
                test_err += (math.log(pred[k] / ALL_CONSTANTS[k]))**2

        if math.isfinite(train_err) and math.isfinite(test_err):
            train_errors.append(train_err)
            test_errors.append(test_err)
            N_values.append(N_fit)

    if not N_values:
        print("  ❌ Все trial'ы провалились")
        return {'std_lnN_pct': 100, 'ratio_test_train': 100}

    train_errors = np.array(train_errors)
    test_errors = np.array(test_errors)
    N_values = np.array(N_values)

    ln_N_values = np.log(N_values)
    mean_lnN = np.mean(ln_N_values)
    std_lnN = np.std(ln_N_values)

    print(f"\n  Валидных trial'ов: {len(N_values)}/{n_trials}")
    print(f"  N_fit: среднее ln(N) = {mean_lnN:.4f} ± {std_lnN:.4f}")
    print(f"  Разброс ln(N): {std_lnN/lnN0*100:.4f}% от ln(N0)")
    print(f"  Train ошибка: {np.mean(train_errors):.6f} ± {np.std(train_errors):.6f}")
    print(f"  Test ошибка:  {np.mean(test_errors):.6f} ± {np.std(test_errors):.6f}")

    ratio = np.mean(test_errors) / np.mean(train_errors) if np.mean(train_errors) > 0 else float('inf')
    print(f"  Отношение test/train: {ratio:.2f}")

    if std_lnN/lnN0 < 0.01 and ratio < 3.0:
        print(f"\n  ✅✅✅ ТЕСТ Z ПРОЙДЕН")
    elif std_lnN/lnN0 < 0.05 and ratio < 5.0:
        print(f"\n  ✅ ТЕСТ Z ПРОЙДЕН")
    else:
        print(f"\n  ❌ ТЕСТ Z НЕ ПРОЙДЕН")

    return {
        'std_lnN_pct': std_lnN/lnN0*100,
        'ratio_test_train': ratio,
        'good_trials': len(N_values),
        'n_trials': n_trials
    }


# ================================================================
# ТЕСТ W: PARAMETER ABLATION
# ================================================================
def W_parameter_ablation():
    print("\n" + "="*70)
    print("ТЕСТ W: PARAMETER ABLATION — ЗАМЕНА K И π")
    print("="*70)

    all_keys = list(ALL_CONSTANTS.keys())

    # Эталонная ошибка
    pred_eti = formulas_default(N0)
    eti_error = sum((math.log(pred_eti[k]/ALL_CONSTANTS[k]))**2
                     for k in all_keys if k in pred_eti and pred_eti[k] > 0)
    print(f"\n  Эталонная ошибка ЕТИ (K=6, π=3.14): {eti_error:.6f}")

    # Абляция K
    print(f"\n  АБЛЯЦИЯ K (π = π_true):")
    print(f"  {'K':<8} {'Опт. N':<18} {'Ошибка':<14} {'Avg rel %':<12} {'Деградация':<10}")
    print(f"  {'─'*60}")

    K_results = []
    for K_test in [2, 3, 4, 5, 6, 7, 8, 10, 12, 20]:
        if K_test == 1:
            continue  # пропускаем K=1 (lnK=0)
        lnK_test = math.log(K_test)
        N_opt, err = fit_N(all_keys, N0, K_test, pi_true, lnK_test)

        pred = formulas_with_params(N_opt, K_test, pi_true, lnK_test)
        rel_errors = []
        for k in all_keys:
            if k in pred and pred[k] > 0 and ALL_CONSTANTS[k] > 0:
                rel_errors.append(abs(pred[k]/ALL_CONSTANTS[k] - 1) * 100)

        avg_rel = np.mean(rel_errors) if rel_errors else float('inf')
        degradation = err / eti_error if eti_error > 0 else float('inf')
        marker = " ← ИСТИННОЕ" if K_test == 6 else ""

        print(f"  {K_test:<8} {N_opt:<18.4e} {err:<14.6f} {avg_rel:<12.2f} {degradation:<10.1f}{marker}")
        K_results.append({'K': K_test, 'error': err, 'degradation': degradation})

    # Абляция π
    print(f"\n  АБЛЯЦИЯ π (K = 6):")
    print(f"  {'π':<10} {'Опт. N':<18} {'Ошибка':<14} {'Avg rel %':<12} {'Деградация':<10}")
    print(f"  {'─'*60}")

    pi_results = []
    for pi_mult in [0.5, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.5, 2.0]:
        pi_test = pi_true * pi_mult
        N_opt, err = fit_N(all_keys, N0, K_true, pi_test, lnK_true)

        pred = formulas_with_params(N_opt, K_true, pi_test, lnK_true)
        rel_errors = []
        for k in all_keys:
            if k in pred and pred[k] > 0 and ALL_CONSTANTS[k] > 0:
                rel_errors.append(abs(pred[k]/ALL_CONSTANTS[k] - 1) * 100)

        avg_rel = np.mean(rel_errors) if rel_errors else float('inf')
        degradation = err / eti_error if eti_error > 0 else float('inf')
        marker = " ← ИСТИННОЕ" if abs(pi_mult - 1.0) < 0.001 else ""

        print(f"  {pi_test:<10.4f} {N_opt:<18.4e} {err:<14.6f} {avg_rel:<12.2f} {degradation:<10.1f}{marker}")
        pi_results.append({'pi': pi_test, 'error': err, 'degradation': degradation})

    # Анализ чувствительности
    print(f"\n  АНАЛИЗ ЧУВСТВИТЕЛЬНОСТИ:")

    k6_err = [r['error'] for r in K_results if r['K'] == 6]
    k5_err = [r['error'] for r in K_results if r['K'] == 5]
    k7_err = [r['error'] for r in K_results if r['K'] == 7]

    if k6_err and k5_err and k7_err:
        sens_K_up = (k7_err[0] - k6_err[0]) / k6_err[0] * 100
        sens_K_down = (k5_err[0] - k6_err[0]) / k6_err[0] * 100
        print(f"  K: 5→+{sens_K_down:.0f}%, 7→+{sens_K_up:.0f}% ошибки")
    else:
        sens_K_up = 0

    pi0 = [r['error'] for r in pi_results if abs(r['pi']/pi_true - 1) < 0.001]
    pi05 = [r['error'] for r in pi_results if abs(r['pi']/pi_true - 0.95) < 0.01]
    pi105 = [r['error'] for r in pi_results if abs(r['pi']/pi_true - 1.05) < 0.01]

    if pi0 and pi05 and pi105:
        sens_pi_up = (pi105[0] - pi0[0]) / pi0[0] * 100
        sens_pi_down = (pi05[0] - pi0[0]) / pi0[0] * 100
        print(f"  π: -5%→+{sens_pi_down:.0f}%, +5%→+{sens_pi_up:.0f}% ошибки")
    else:
        sens_pi_up = 0

    # Вердикт
    print(f"\n  ВЕРДИКТ:")
    if sens_K_up > 50 and sens_pi_up > 50:
        print(f"  ✅✅✅ ТЕСТ W ПРОЙДЕН: модель катастрофически зависит от K=6 и π")
    elif sens_K_up > 20:
        print(f"  ✅ ТЕСТ W ПРОЙДЕН (умеренная чувствительность)")
    else:
        print(f"  ❌ ТЕСТ W НЕ ПРОЙДЕН")

    return {
        'sensitivity_K': sens_K_up,
        'sensitivity_pi': sens_pi_up,
        'eti_error': eti_error
    }


# ================================================================
# ГЛАВНЫЙ ЗАПУСК
# ================================================================
def main():
    print("="*70)
    print("ТЕСТЫ Z И W — ИСПРАВЛЕННАЯ ВЕРСИЯ")
    print("="*70)
    print(f"K = {K_true}, π = {pi_true:.6f}")
    print(f"N0 = {N0:.4e}, lnN0 = {lnN0:.4f}")

    results_Z = Z_out_of_distribution()
    results_W = W_parameter_ablation()

    print("\n" + "="*70)
    print("ФИНАЛЬНЫЙ ВЕРДИКТ")
    print("="*70)

    z_pass = results_Z['std_lnN_pct'] < 0.01 and results_Z['ratio_test_train'] < 3.0
    w_pass = results_W['sensitivity_K'] > 50 and results_W['sensitivity_pi'] > 50

    print(f"  Тест Z: {'✅' if z_pass else '❌'} (разброс N: {results_Z['std_lnN_pct']:.4f}%)")
    print(f"  Тест W: {'✅' if w_pass else '❌'} (чувств. K: +{results_W['sensitivity_K']:.0f}%, π: +{results_W['sensitivity_pi']:.0f}%)")

    if z_pass and w_pass:
        print("\n  ✅✅✅ ОБА ТЕСТА ПРОЙДЕНЫ")

if __name__ == "__main__":
    results = main()