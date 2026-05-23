import math
import numpy as np
from collections import defaultdict
import itertools
import time
import pickle
import os

K = 6.0
pi = math.pi
lnK = math.log(K)

# Имя файла для кэширования
CACHE_FILE = "processed_unknown_cache.pkl"

def make_formula(coeff, pow_lnN, pow_N13):
    return (coeff, pow_lnN, pow_N13)

formulas = {
    'ħ': make_formula(1.0 / K, 3, -1),
    'c': make_formula(pi / (K ** 2 * lnK), 4, 0),
    'l_P': make_formula(4 * lnK, 2, -1),
    't_P': make_formula(4 * K ** 2 * lnK ** 2 / pi, -2, -1),
    'E_P': make_formula(pi / (4 * K ** 3 * lnK ** 2), 5, 0),
    'G': make_formula(16 * pi ** 3 / (K ** 5 * lnK), 13, -1),
    'm_P': make_formula(K / (4 * pi), -3, 0),
    'T_P': make_formula(8 * pi, -4, 1),
    'k_B': make_formula(1.0 / (8 * pi ** 2), 8, -1),
    'α': make_formula(2 * lnK ** 2 / pi, -1, 0),
    'm_e': make_formula(4 * pi / math.sqrt(K), 4, -1),
    'm_p': make_formula(math.sqrt(pi) / K ** (1.5), 6, -1),
    'ep_0': make_formula(1.0 / (8 * pi ** 3 * lnK), -20, 1),
    'mu_0': make_formula(8 * pi * K ** 4 * lnK ** 3, 12, -1),
    'R∞': make_formula(4 * lnK ** 3 / (pi * K ** (1.5)), 3, 0),
    'a_0': make_formula(K ** (1.5) / (8 * pi * lnK), -4, 0),
    'Z_0': make_formula(8 * K ** 2 * pi ** 2 * lnK ** 2, 16, -1),
    'Φ_0': make_formula(pi ** 2 * math.sqrt(K), 10, -1),
    'q_e': make_formula(1.0 / (pi * K ** (1.5)), -7, 0),
    'λ_e': make_formula(K ** (1.5) * lnK / (2 * pi), -5, 0),
    'λ_p': make_formula(2 * K ** (2.5) * lnK / math.sqrt(pi), -7, 0),
    'Λ': make_formula(1.0 / math.sqrt(pi), 12, -2),
    'κ': make_formula(128 * K ** 3 * lnK ** 3, -3, -1),
    'v_H': make_formula(8 * pi ** (1.5) / math.sqrt(2), 6, -1),
    'm_muon': make_formula(4 * pi ** 2/ (K * 3 ** (1 / 2)), 5, -1),
    'm_tau': make_formula(pi ** (1 / 2) * (K ** 2), 5, -1),

#     m_muon = 4 * pi ** 2 * lnN ** 5 / (K * 3 ** (1 / 2) * N ** (1 / 3))
#
# m_tau = pi ** (1 / 2) * (lnN ** 5) * (K ** 2) / (N ** (1 / 3))
#
# m_pi_meson = (lnN) ** 6 * 1 / (4 * pi ** 2) * N ** (-1 / 3) / 2 ** (1 / 2)
#
# m_pi0_meson = 2 * pi * K ** 3 * lnN ** 4 / N ** (1 / 3)
}

all_names = list(formulas.keys())
n_formulas = len(all_names)

a_vals = np.array([formulas[name][1] for name in all_names], dtype=np.int32)
b_vals = np.array([formulas[name][2] for name in all_names], dtype=np.int32)
coeff_vals = np.array([formulas[name][0] for name in all_names], dtype=np.float64)
log_coeffs = np.array([math.log(abs(c)) for c in coeff_vals], dtype=np.float64)


# ============================================================
# СИМВОЛЬНАЯ ФОРМУЛА
# ============================================================
def get_symbolic_formula(terms_normalized):
    pi_power = 0.0
    K_power = 0.0
    lnK_power = 0.0
    sqrt2_power = 0.0
    numeric_factor = 1.0

    for idx, p in terms_normalized:
        name = all_names[idx]
        coeff = coeff_vals[idx]

        if name == 'ħ':
            numeric_factor *= (1.0 / K) ** p
            K_power -= p
        elif name == 'c':
            numeric_factor *= (pi / (K ** 2 * lnK)) ** p
            pi_power += p
            K_power -= 2 * p
            lnK_power -= p
        elif name == 'l_P':
            numeric_factor *= (4 * lnK) ** p
            lnK_power += p
        elif name == 't_P':
            numeric_factor *= (4 * K ** 2 * lnK ** 2 / pi) ** p
            K_power += 2 * p
            lnK_power += 2 * p
            pi_power -= p
        elif name == 'E_P':
            numeric_factor *= (pi / (4 * K ** 3 * lnK ** 2)) ** p
            pi_power += p
            K_power -= 3 * p
            lnK_power -= 2 * p
        elif name == 'G':
            numeric_factor *= (16 * pi ** 3 / (K ** 5 * lnK)) ** p
            pi_power += 3 * p
            K_power -= 5 * p
            lnK_power -= p
        elif name == 'm_P':
            numeric_factor *= (K / (4 * pi)) ** p
            K_power += p
            pi_power -= p
        elif name == 'T_P':
            numeric_factor *= (8 * pi) ** p
            pi_power += p
        elif name == 'k_B':
            numeric_factor *= (1.0 / (8 * pi ** 2)) ** p
            pi_power -= 2 * p
        elif name == 'α':
            numeric_factor *= (2 * lnK ** 2 / pi) ** p
            lnK_power += 2 * p
            pi_power -= p
        elif name == 'm_e':
            numeric_factor *= (4 * pi / math.sqrt(K)) ** p
            pi_power += p
            K_power -= 0.5 * p
        elif name == 'm_p':
            numeric_factor *= (math.sqrt(pi) / K ** 1.5) ** p
            pi_power += 0.5 * p
            K_power -= 1.5 * p
        elif name == 'ep_0':
            numeric_factor *= (1.0 / (8 * pi ** 3 * lnK)) ** p
            pi_power -= 3 * p
            lnK_power -= p
        elif name == 'mu_0':
            numeric_factor *= (8 * pi * K ** 4 * lnK ** 3) ** p
            pi_power += p
            K_power += 4 * p
            lnK_power += 3 * p
        elif name == 'R∞':
            numeric_factor *= (4 * lnK ** 3 / (pi * K ** 1.5)) ** p
            lnK_power += 3 * p
            pi_power -= p
            K_power -= 1.5 * p
        elif name == 'a_0':
            numeric_factor *= (K ** 1.5 / (8 * pi * lnK)) ** p
            K_power += 1.5 * p
            pi_power -= p
            lnK_power -= p
        elif name == 'Z_0':
            numeric_factor *= (8 * K ** 2 * pi ** 2 * lnK ** 2) ** p
            K_power += 2 * p
            pi_power += 2 * p
            lnK_power += 2 * p
        elif name == 'Φ_0':
            numeric_factor *= (pi ** 2 * math.sqrt(K)) ** p
            pi_power += 2 * p
            K_power += 0.5 * p
        elif name == 'q_e':
            numeric_factor *= (1.0 / (pi * K ** 1.5)) ** p
            pi_power -= p
            K_power -= 1.5 * p
        elif name == 'λ_e':
            numeric_factor *= (K ** 1.5 * lnK / (2 * pi)) ** p
            K_power += 1.5 * p
            lnK_power += p
            pi_power -= p
        elif name == 'λ_p':
            numeric_factor *= (2 * K ** 2.5 * lnK / math.sqrt(pi)) ** p
            K_power += 2.5 * p
            lnK_power += p
            pi_power -= 0.5 * p
        elif name == 'Λ':
            numeric_factor *= (1.0 / math.sqrt(pi)) ** p
            pi_power -= 0.5 * p
        elif name == 'κ':
            numeric_factor *= (128 * K ** 3 * lnK ** 3) ** p
            K_power += 3 * p
            lnK_power += 3 * p
        elif name == 'v_H':
            numeric_factor *= (8 * pi ** 1.5 / math.sqrt(2)) ** p
            pi_power += 1.5 * p
            sqrt2_power -= 0.5 * p
        else:
            numeric_factor *= coeff ** p

    # Строим символьное представление
    parts = []

    # Числовой коэффициент
    if abs(numeric_factor - 1.0) > 1e-10:
        if abs(numeric_factor) >= 1000000000 or (abs(numeric_factor) <= 0.000000001 and numeric_factor != 0):
            parts.append(f"{numeric_factor:.6e}")
        else:
            rounded = round(numeric_factor)
            if abs(numeric_factor - rounded) < 1e-10:
                parts.append(str(rounded))
            else:
                parts.append(f"{numeric_factor:.10f}")

    # π
    if abs(pi_power) > 1e-10:
        p = pi_power
        if abs(p - round(p)) < 1e-10:
            p_int = int(round(p))
            if p_int == 1:
                parts.append("π")
            elif p_int == -1:
                parts.append("1/π")
            elif p_int > 0:
                parts.append(f"π^{p_int}")
            else:
                parts.append(f"1/π^{{-p_int}}")
        else:
            parts.append(f"π^{{{p:.6f}}}")

    # K
    if abs(K_power) > 1e-10:
        p = K_power
        if abs(p - round(p)) < 1e-10:
            p_int = int(round(p))
            if p_int == 1:
                parts.append("K")
            elif p_int == -1:
                parts.append("1/K")
            elif p_int > 0:
                parts.append(f"K^{p_int}")
            else:
                parts.append(f"1/K^{{-p_int}}")
        else:
            parts.append(f"K^{{{p:.6f}}}")

    # lnK
    if abs(lnK_power) > 1e-10:
        p = lnK_power
        if abs(p - round(p)) < 1e-10:
            p_int = int(round(p))
            if p_int == 1:
                parts.append("lnK")
            elif p_int == -1:
                parts.append("1/lnK")
            elif p_int > 0:
                parts.append(f"(lnK)^{p_int}")
            else:
                parts.append(f"1/(lnK)^{{-p_int}}")
        else:
            parts.append(f"(lnK)^{{{p:.6f}}}")

    # √2
    if abs(sqrt2_power) > 1e-10:
        p = sqrt2_power
        if abs(p - round(p)) < 1e-10:
            p_int = int(round(p))
            if p_int == 1:
                parts.append("√2")
            elif p_int == -1:
                parts.append("1/√2")
            elif p_int > 0:
                parts.append(f"(√2)^{p_int}")
            else:
                parts.append(f"1/(√2)^{{-p_int}}")
        else:
            parts.append(f"(√2)^{{{p:.6f}}}")

    if not parts:
        parts.append("1")

    return " · ".join(parts)


# ============================================================
# ЗАГРУЗКА ИЛИ ВЫЧИСЛЕНИЕ
# ============================================================

def load_or_compute():
    """Загружает processed_unknown из кэша, либо вычисляет заново."""

    # Проверяем, существует ли файл кэша
    if os.path.exists(CACHE_FILE):
        print(f"\n{'=' * 80}")
        print(f"ЗАГРУЗКА ИЗ КЭША: {CACHE_FILE}")
        print(f"{'=' * 80}")

        try:
            with open(CACHE_FILE, 'rb') as f:
                cache_data = pickle.load(f)

            processed_unknown = cache_data['processed_unknown']
            found_known = cache_data.get('found_known', [])
            found_unknown = cache_data.get('found_unknown', [])

            print(f"  Загружено UNKNOWN: {len(processed_unknown)}")
            print(f"  Загружено KNOWN: {len(found_known)}")
            print(f"  Всего UNKNOWN (оригинальных): {len(found_unknown)}")

            return processed_unknown, found_known, found_unknown

        except Exception as e:
            print(f"  ОШИБКА при загрузке кэша: {e}")
            print(f"  Будет выполнен новый перебор...")

    # Если кэша нет или загрузка не удалась — выполняем перебор
    print(f"\n{'=' * 80}")
    print("ВЫПОЛНЕНИЕ ПОЛНОГО ПЕРЕБОРА")
    print(f"{'=' * 80}")

    processed_unknown, found_known, found_unknown = run_meet_in_the_middle()

    # Сохраняем в кэш
    print(f"\n{'=' * 80}")
    print(f"СОХРАНЕНИЕ В КЭШ: {CACHE_FILE}")
    print(f"{'=' * 80}")

    try:
        cache_data = {
            'processed_unknown': processed_unknown,
            'found_known': found_known,
            'found_unknown': found_unknown,
        }
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"  Сохранено успешно!")
        print(f"  Размер файла: {os.path.getsize(CACHE_FILE):,} байт")
    except Exception as e:
        print(f"  ОШИБКА при сохранении: {e}")

    return processed_unknown, found_known, found_unknown


def run_meet_in_the_middle():
    """Выполняет meet-in-the-middle перебор и возвращает результаты."""

    # ============================================================
    # KNOWN: известные физические тождества
    # ============================================================
    print("=" * 80)
    print("ФОРМИРОВАНИЕ БАЗЫ KNOWN (ИЗВЕСТНЫЕ ФИЗИЧЕСКИЕ ТОЖДЕСТВА)")
    print("=" * 80)

    known_signatures = set()

    def add_known(parts_list):
        powers = [p for _, p in parts_list]
        g = powers[0]
        for pp in powers[1:]:
            g = math.gcd(g, abs(pp))
        if g > 1:
            parts_list = [(n, p // g) for n, p in parts_list]
        first_p = parts_list[0][1]
        if first_p < 0:
            parts_list = [(n, -p) for n, p in parts_list]
        parts_list.sort()
        name_to_idx = {name: i for i, name in enumerate(all_names)}
        sig = tuple((name_to_idx[n], p) for n, p in parts_list if p != 0)
        known_signatures.add(sig)

    # Планковские
    for parts in [
        [('c', 1), ('t_P', 1), ('l_P', -1)],
        [('ħ', 1), ('c', -1), ('l_P', -1), ('m_P', -1)],
        [('ħ', 1), ('c', -2), ('t_P', -1), ('m_P', -1)],
        [('l_P', 2), ('c', 3), ('ħ', -1), ('G', -1)],
        [('E_P', 1), ('m_P', -1), ('c', -2)],
        [('ħ', 1), ('E_P', -1), ('t_P', -1)],
        [('l_P', 3), ('t_P', -2), ('G', -1), ('m_P', -1)],
        [('ħ', 1), ('c', 1), ('G', -1), ('m_P', -2)],
    ]:
        add_known(parts)

    # Электромагнитные
    for parts in [
        [('c', 2), ('ep_0', 1), ('mu_0', 1)],
        [('Z_0', 2), ('ep_0', 1), ('mu_0', -1)],
        [('c', 1), ('ep_0', 1), ('Z_0', 1)],
        [('Z_0', 1), ('c', -1), ('mu_0', -1)],
    ]:
        add_known(parts)

    # Атомные
    add_known([('a_0', 1), ('α', 1), ('m_e', 1), ('c', 1), ('ħ', -1)])

    print(f"  Загружено KNOWN сигнатур: {len(known_signatures)}")


    print("MEET-IN-THE-MIDDLE: ПОИСК ВСЕХ БЕЗРАЗМЕРНЫХ КОМБИНАЦИЙ")
    start_time = time.time()
    MAX_POWER = 6
    MIN_POWER = -6
    MAX_FORMULAS = 6

    print(f"\n  Шаг 1: Перебор комбинаций 1-2 формул...")
    combo_map = defaultdict(list)

    for i in range(n_formulas):
        for p in range(MIN_POWER, MAX_POWER + 1):
            if p == 0:
                continue
            sum_a = a_vals[i] * p
            sum_b = b_vals[i] * p
            terms = ((i, p),)
            log_c = p * log_coeffs[i]
            combo_map[(sum_a, sum_b)].append((terms, log_c))

    for i, j in itertools.combinations(range(n_formulas), 2):
        for pi in range(MIN_POWER, MAX_POWER + 1):
            if pi == 0:
                continue
            for pj in range(MIN_POWER, MAX_POWER + 1):
                if pj == 0:
                    continue
                sum_a = a_vals[i] * pi + a_vals[j] * pj
                sum_b = b_vals[i] * pi + b_vals[j] * pj
                terms = ((i, pi), (j, pj))
                log_c = pi * log_coeffs[i] + pj * log_coeffs[j]
                combo_map[(sum_a, sum_b)].append((terms, log_c))

    print(f"  Создано групп: {len(combo_map):,}")

    print(f"\n  Шаг 2: Поиск пар с компенсацией...")
    found_known = []
    found_unknown = []
    seen_sigs = set()

    for key1, list1 in combo_map.items():
        target_key = (-key1[0], -key1[1])
        if target_key not in combo_map:
            continue
        list2 = combo_map[target_key]

        for terms1, log_c1 in list1:
            for terms2, log_c2 in list2:
                merged = defaultdict(int)
                for i, p in terms1 + terms2:
                    merged[i] += p

                if len(merged) > MAX_FORMULAS:
                    continue
                if any(abs(p) > MAX_POWER for p in merged.values()):
                    continue
                if any(p == 0 for p in merged.values()):
                    continue

                all_powers = [abs(p) for p in merged.values()]
                g = all_powers[0]
                for pp in all_powers[1:]:
                    g = math.gcd(g, pp)
                if g > 1:
                    merged = {i: p // g for i, p in merged.items()}

                terms_norm = tuple(sorted(merged.items()))
                first_p = terms_norm[0][1]
                if first_p < 0:
                    terms_norm = tuple((i, -p) for i, p in terms_norm)

                if terms_norm in seen_sigs:
                    continue
                seen_sigs.add(terms_norm)

                value = math.exp(log_c1 + log_c2)
                symbolic = get_symbolic_formula(terms_norm)
                is_known = terms_norm in known_signatures

                # Формируем строку "как в коде"
                parts = []
                for i, p in terms_norm:
                    name = all_names[i]
                    if p == 1:
                        parts.append(name)
                    elif p == -1:
                        parts.append(f"1/{name}")
                    elif p > 0:
                        parts.append(f"{name}^{p}")
                    else:
                        parts.append(f"1/{name}^{{{-p}}}")
                expr = " · ".join(parts)

                entry = {
                    'expr': expr,
                    'value': value,
                    'symbolic': symbolic,
                    'terms': terms_norm,
                }

                if is_known:
                    found_known.append(entry)
                else:
                    found_unknown.append(entry)

    elapsed = time.time() - start_time
    print(f"  Время перебора: {elapsed:.1f} сек")
    print(f"  Найдено KNOWN: {len(found_known)}")
    print(f"  Найдено UNKNOWN: {len(found_unknown)}")

    # Обработка UNKNOWN с инверсией
    processed_unknown = []
    for entry in found_unknown:
        value = entry['value']
        expr = entry['expr']
        symbolic = entry['symbolic']

        if value < 1.0:
            display_value = 1.0 / value
            display_expr = "1/(" + expr + ")"
            display_symbolic = "1/(" + symbolic + ")"
        else:
            display_value = value
            display_expr = expr
            display_symbolic = symbolic

        processed_unknown.append({
            'display_value': display_value,
            'display_expr': display_expr,
            'display_symbolic': display_symbolic,
            'original_value': value,
            'original_expr': expr,
            'original_symbolic': symbolic,
            'terms': entry['terms'],
        })

    processed_unknown.sort(key=lambda x: x['display_value'])

    return processed_unknown, found_known, found_unknown


# ============================================================
# ГЛАВНЫЙ КОД
# ============================================================

if __name__ == "__main__":

    # Загружаем или вычисляем
    processed_unknown, found_known, found_unknown = load_or_compute()

    print("UNKNOWN: ПЕРВЫЕ 1200 НОВЫХ ПРЕДСКАЗАНИЙ")
    print(f"\n  {'Комбинация констант':<55} {'Эмерджентная формула':<30} {'Значение':<15}")
    print(f"  {'-' * 100}")

    count = 0
    for entry in processed_unknown:
        # if count >= 5000:
        #     break

        val = entry['display_value']
        if val <= 2*math.pi-0.0001:
           continue
        # if val > 1000:#3*math.pi:
        #    continue
        #
        #
        if val >=  2*math.pi+0.0001:
            continue
        # val = entry['display_value']
        # if abs(val - round(val)) > 1e-6:
        #     continue

        count += 1
        expr_short = entry['display_expr'][:90]
        print(f"  {expr_short:<90}  {entry['display_value']:<15.10f}")

    print(f"\n  Показано: {count} результатов")

    # ============================================================
    # СТАТИСТИКА
    # ============================================================
    values_for_stats = []
    for entry in found_unknown:
        v = entry['value']
        if v < 1.0:
            v = 1.0 / v
        values_for_stats.append(v)

    values_for_stats = np.array(values_for_stats)

    print(f"\n{'=' * 80}")
    print("СТАТИСТИКА")
    print(f"{'=' * 80}")
    print(f"  Всего UNKNOWN: {len(found_unknown)}")
    print(f"  Диапазон значений: [{values_for_stats.min():.4f}, {values_for_stats.max():.4f}]")
    print(f"  Из них с value ≈ 1 (|v-1|<0.01): {np.sum(abs(values_for_stats - 1.0) < 0.01)}")
    print(f"  Из них с value ≈ 2: {np.sum(abs(values_for_stats - 2.0) < 0.02)}")
    print(f"  Из них с value ≈ π: {np.sum(abs(values_for_stats - pi) < 0.1)}")
    print(f"  Из них с value ≈ √π: {np.sum(abs(values_for_stats - math.sqrt(pi)) < 0.05)}")
    print(f"  Целочисленные (|v-round(v)|<1e-6): {np.sum(abs(values_for_stats - np.round(values_for_stats)) < 1e-6)}")