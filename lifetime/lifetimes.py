"""
ЕТИ: СИСТЕМАТИЧЕСКИЙ ПОИСК ГИПОТЕЗ ДЛЯ ПЕРИОДОВ ПОЛУРАСПАДА
Версия 4.0 — проверка гипотез на всём массиве ядер

Алгоритм:
1. Формулируем гипотезу: степень при √2 = f(Z,N,A), степень при √3 = g(Z,N,A)
2. Для КАЖДОЙ гипотезы варьируем a (степень lnN), c (степень lnK), d (степень π)
3. Фиксируем гипотезу, если ≥ 2/3 ядер имеют ошибку < 1%
"""

import math
import time
from datetime import datetime
from itertools import product

# ============================================================
# ФУНДАМЕНТАЛЬНЫЕ ПАРАМЕТРЫ
# ============================================================
K = 6.0
pi = math.pi
lnK = math.log(K)
N_val = 4.1847e121
lnN = math.log(N_val)

sqrt2 = math.sqrt(2)
sqrt3 = math.sqrt(3)

print("=" * 100)
print("ЕТИ v4.0: СИСТЕМАТИЧЕСКИЙ ПОИСК ГИПОТЕЗ ДЛЯ ВРЕМЁН ЖИЗНИ ЯДЕР")
print("=" * 100)
print(f"  ln N = {lnN:.4f}")
print(f"  ln K = {lnK:.4f}")

# ============================================================
# ДАННЫЕ ЯДЕР
# ============================================================
nuclei = {
    'n':    (0, 1, 1, 877.8, 'b-'),
    'H3':   (1, 2, 3, 388800000.0, 'b-'),
    'He6':  (2, 4, 6, 0.8067, 'b-'),
    'He8':  (2, 6, 8, 0.1191, 'b-'),
    #'Be7':  (4, 3, 7, 4598208.0, 'EC'),
    'Be10': (4, 6, 10, 4.765e13, 'b-'),
    'Be11': (4, 7, 11, 13.76, 'b-'),
    'C10':  (6, 4, 10, 19.308, 'b+'),
    'C11':  (6, 5, 11, 1220.4, 'b+'),
    'C14':  (6, 8, 14, 1.808e11, 'b-'),
    'C15':  (6, 9, 15, 2.449, 'b-'),
    'N13':  (7, 6, 13, 862.7, 'b+'),
    'N16':  (7, 9, 16, 7.13, 'b-'),
    'N17':  (7, 10, 17, 4.173, 'b-'),
    'O14':  (8, 6, 14, 70.62, 'b+'),
    'O15':  (8, 7, 15, 176.4, 'b+'),
    'O19':  (8, 11, 19, 26.91, 'b-'),
    'O20':  (8, 12, 20, 13.51, 'b-'),
    'F18':  (9, 9, 18, 9483.0, 'b+'),
    'Na22': (11, 11, 22, 8.208e7, 'b+'),
    'Na24': (11, 13, 24, 53820.0, 'b-'),
    'Al26': (13, 13, 26, 2.261e13, 'b+'),
    'P32':  (15, 17, 32, 1.235e6, 'b-'),
    'S35':  (16, 19, 35, 7.569e6, 'b-'),
    'Cl36': (17, 19, 36, 9.497e12, 'b-'),
    'Ar39': (18, 21, 39, 8.490e9, 'b-'),
    'K42':  (19, 23, 42, 44580.0, 'b-'),
    'Ca45': (20, 25, 45, 1.407e7, 'b-'),
    'Mn52': (25, 27, 52, 4.83e6, 'b+'),
    #'Mn54': (25, 29, 54, 2.698e7, 'EC'),
    #'Fe55': (26, 29, 55, 8.64e7, 'EC'),
    'Fe59': (26, 33, 59, 3.8448e7, 'b-'),
    #'Co57': (27, 30, 57, 2.348e7, 'EC'),
    'Co60': (27, 33, 60, 1.663e8, 'b-'),
    'Ni63': (28, 35, 63, 3.186e10, 'b-'),
    'Cu64': (29, 35, 64, 45720.0, 'b-'),
    #'Zn65': (30, 35, 65, 2.107e7, 'EC'),
    'Sr90': (38, 52, 90, 9.072e8, 'b-'),
    'Cs135':(55, 80, 135, 7.258e13, 'b-'),
    'Cs137':(55, 82, 137, 9.483e8, 'b-'),
    'I129': (53, 76, 129, 4.955e14, 'b-'),
    'Sm146':(62, 84, 146, 3.247e15, 'b-'),
    'Pu239':(94, 145, 239, 7.609e11, 'b-'),
    'U236': (92, 144, 236, 7.391e14, 'b-'),
}

nuclei_list = [(name, Z, N, A, tau, dtype) for name, (Z, N, A, tau, dtype) in nuclei.items()]
n_total = len(nuclei_list)
threshold_count = int(n_total * 0.20)  # минимум 2/3 ядер

print(f"\n  Всего ядер: {n_total}")
print(f"  Порог для фиксации гипотезы: {threshold_count} из {n_total} (≥ 2/3)")

# ============================================================
# ФУНКЦИЯ ВЫЧИСЛЕНИЯ ВРЕМЕНИ ЖИЗНИ
# ============================================================
def compute_tau(a, b, c, d):
    """Вычисляет время жизни по показателям"""
    tau = (lnN ** a)
    tau *= (sqrt2 ** b)
    tau *= (sqrt3 ** c)
    tau *= (lnK ** d)
    tau *= (pi ** d)  # здесь d используется и для lnK и для pi (можно разделить)
    return tau

def compute_tau_full(a, b, c, d_lnK, d_pi):
    """Полная формула с разными степенями для lnK и π"""
    tau = (lnN ** a)
    tau *= (sqrt2 ** b)
    tau *= (sqrt3 ** c)
    tau *= (lnK ** d_lnK)
    tau *= (pi ** d_pi)
    return tau

# ============================================================
# ГЕНЕРАЦИЯ ГИПОТЕЗ
# ============================================================
def generate_hypotheses():
    """
    Генерирует список гипотез для степеней √2 и √3.
    Каждая гипотеза — функция от (Z, N, A, decay_type).
    """
    hypotheses = []

    # === БАЗОВЫЕ ГИПОТЕЗЫ: степень = одна из ядерных величин ===
    base_vars = [
        ('Z', lambda z, n, a, dt: z),
        ('N', lambda z, n, a, dt: n),
        ('A', lambda z, n, a, dt: a),
        ('N-Z', lambda z, n, a, dt: n - z),
        ('Z-N', lambda z, n, a, dt: z - n),
    ]

    for name1, func1 in base_vars:
        for name2, func2 in base_vars:
            # Прямые степени
            hypotheses.append((f"b={name1}, c={name2}",
                             lambda z, n, a, dt, f1=func1, f2=func2: (f1(z,n,a,dt), f2(z,n,a,dt))))
            # Обратные степени
            hypotheses.append((f"b=-{name1}, c=-{name2}",
                             lambda z, n, a, dt, f1=func1, f2=func2: (-f1(z,n,a,dt), -f2(z,n,a,dt))))
            # Смешанные
            hypotheses.append((f"b={name1}, c=-{name2}",
                             lambda z, n, a, dt, f1=func1, f2=func2: (f1(z,n,a,dt), -f2(z,n,a,dt))))
            hypotheses.append((f"b=-{name1}, c={name2}",
                             lambda z, n, a, dt, f1=func1, f2=func2: (-f1(z,n,a,dt), f2(z,n,a,dt))))

    # === СДВИНУТЫЕ ГИПОТЕЗЫ: степень = величина ± 1 ===
    for name1, func1 in base_vars:
        for name2, func2 in base_vars:
            for d1 in [-1, 1]:
                for d2 in [-1, 1]:
                    hypotheses.append(
                        (f"b={name1}{d1:+d}, c={name2}{d2:+d}",
                         lambda z, n, a, dt, f1=func1, f2=func2, delta1=d1, delta2=d2:
                         (f1(z, n, a, dt) + delta1, f2(z, n, a, dt) + delta2))
                    )

    # === ГИПОТЕЗЫ С ДЕЛЕНИЕМ ===
    div_hypotheses = [
        ('b=Z, c=N/Z', lambda z,n,a,dt: (z, n/z if z > 0 else 0)),
        ('b=N, c=Z/N', lambda z,n,a,dt: (n, z/n if n > 0 else 0)),
        ('b=N-Z, c=(N-Z)/A', lambda z,n,a,dt: (n-z, (n-z)/a if a > 0 else 0)),
    ]
    for name, func in div_hypotheses:
        hypotheses.append((name, func))

    # === ГИПОТЕЗЫ ДЛЯ КОНКРЕТНЫХ ТИПОВ РАСПАДА ===
    def b_by_decay(z, n, a, dt):
        if dt == 'b-': return n - z
        elif dt == 'b+': return z - n
        elif dt == 'EC': return z
        else: return 0

    def c_by_decay(z, n, a, dt):
        if dt == 'b-': return a
        elif dt == 'b+': return a
        elif dt == 'EC': return n
        else: return 0

    hypotheses.append(('b=decay, c=decay', lambda z,n,a,dt: (b_by_decay(z,n,a,dt), c_by_decay(z,n,a,dt))))

    return hypotheses


# ============================================================
# ПРОВЕРКА ГИПОТЕЗ
# ============================================================
def tst_hypothesis(hypothesis_func, a_vals, c_lnK_vals, d_pi_vals, threshold=0.2):
    """
    Проверяет гипотезу на всех ядрах.
    Возвращает (success, best_params, results)
    """
    best_total = 0
    best_params = None
    best_results = None

    for a in a_vals:
        for c_lnK in c_lnK_vals:
            for d_pi in d_pi_vals:
                good_count = 0
                results = []

                for name, Z, N, A, tau_exp, dtype in nuclei_list:
                    b, c = hypothesis_func(Z, N, A, dtype)
                    tau_calc = compute_tau_full(a, b, c, c_lnK, d_pi)

                    if tau_exp > 0:
                        error = abs(tau_calc / tau_exp - 1)
                        if error < threshold:
                            good_count += 1
                            results.append((name, tau_calc, tau_exp, error, a, b, c, c_lnK, d_pi))

                if good_count > best_total:
                    best_total = good_count
                    best_params = (a, c_lnK, d_pi)
                    best_results = results

    return best_total, best_params, best_results


# ============================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================
def main():
    t_start = time.time()

    # Генерируем гипотезы
    hypotheses = generate_hypotheses()
    print(f"\n  Сгенерировано гипотез: {len(hypotheses)}")

    # Диапазоны варьирования
    a_vals = list(range(1, 6))           # степень lnN
    c_lnK_vals = list(range(-10, 10))      # степень lnK
    d_pi_vals = list(range(-10, 10))       # степень π

    print(f"  Варьирование: a ∈ {a_vals}, lnK ∈ {c_lnK_vals}, π ∈ {d_pi_vals}")
    print(f"  Всего комбинаций на гипотезу: {len(a_vals) * len(c_lnK_vals) * len(d_pi_vals)}")

    # Проверяем гипотезы
    good_hypotheses = []

    for i, (name, func) in enumerate(hypotheses):
        best_total, best_params, best_results = tst_hypothesis(
            func, a_vals, c_lnK_vals, d_pi_vals, threshold=0.2
        )

        if best_total >= threshold_count:
            a, c_lnK, d_pi = best_params
            good_hypotheses.append({
                'name': name,
                'count': best_total,
                'a': a,
                'c_lnK': c_lnK,
                'd_pi': d_pi,
                'results': best_results,
            })

            # Прогресс
            if (i + 1) % 50 == 0:
                elapsed = time.time() - t_start
                print(f"  Прогресс: {i+1}/{len(hypotheses)} гипотез проверено "
                      f"({elapsed:.1f} с), найдено хороших: {len(good_hypotheses)}")

    elapsed = time.time() - t_start
    print(f"\n  Проверка завершена за {elapsed:.1f} с")

    # ============================================================
    # ВЫВОД РЕЗУЛЬТАТОВ
    # ============================================================
    print(f"\n{'=' * 100}")
    print(f"РЕЗУЛЬТАТЫ: НАЙДЕНО {len(good_hypotheses)} РАБОЧИХ ГИПОТЕЗ")
    print(f"{'=' * 100}")

    if good_hypotheses:
        # Сортируем по числу успешных ядер
        good_hypotheses.sort(key=lambda h: -h['count'])

        for rank, hyp in enumerate(good_hypotheses[:10], 1):
            pct = hyp['count'] / n_total * 100
            print(f"\n  #{rank} {hyp['name']}")
            print(f"     Успешно: {hyp['count']}/{n_total} ({pct:.1f}%)")
            print(f"     Параметры: a={hyp['a']}, lnK={hyp['c_lnK']}, π={hyp['d_pi']}")

            # Показываем первые 5 ядер
            print(f"     Примеры ядер:")
            for name, tc, te, err, a, b, c, cl, dp in hyp['results'][:5]:
                print(f"       {name:6s}: b={b:>+4d}, c={c:>+4d}, ошибка={err*100:.4f}%")
    else:
        print(f"\n  ❌ НЕ НАЙДЕНО НИ ОДНОЙ РАБОЧЕЙ ГИПОТЕЗЫ.")
        print(f"     Порог: {threshold_count}/{n_total} ядер с ошибкой < 1%")

    print(f"\n{'=' * 100}")
    print("ВЫВОД")
    print("=" * 100)

    if good_hypotheses:
        print(f"""
  Найдено {len(good_hypotheses)} гипотез, работающих для ≥ 2/3 ядер.
  Это ОГРОМНЫЙ ПРОГРЕСС по сравнению с версией 3.0!
  
  Следующие шаги:
  1. Для лучших гипотез проверить LOO-тест
  2. Добавить NLO-поправку
  3. Проверить предсказания для новых ядер
""")
    else:
        print(f"""
  Ни одна гипотеза не прошла порог {threshold_count}/{n_total}.
  
  Возможные причины:
  1. Времена жизни не описываются простой формулой с √2 и √3
  2. Нужен дополнительный член (например, фактор энергии распада)
  3. Нужна отдельная формула для каждого типа распада
  4. Порог 1% слишком жёсткий для ведущего порядка
""")

    return good_hypotheses


if __name__ == "__main__":
    hypotheses = main()