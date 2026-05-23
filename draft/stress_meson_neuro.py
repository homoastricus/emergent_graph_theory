import numpy as np
from itertools import product

# Ваши данные
mesons = {
    'π0': (4, 8, 6, 1.0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 'meson'),
    'π+': (6, -5, 0, -2.0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 'meson'),
    'π-': (6, -5, 0, -2.0, 1, 1, 1, -1, 0, 0, 0, 1, 0, -1, 'meson'),
    'K+': (6, -3, 0, -1.5, 1, 1, 0.5, 0.5, 1, 0, 0, 1, 0, 1, 'meson'),
    'K0': (6, -3, 0, -1.5, 1, 1, 0.5, -0.5, 1, 0, 0, 1, 0, 0, 'meson'),
    'η': (5, 2, 0, 2.0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 'meson'),
    'η′': (5, 2, 6, -1.0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 'meson'),
    'φ': (5, 4, 3, 0.5, 1, 1, 0, 0, 0, 0, 0, 2, 1, 0, 'meson'),
    'ω': (5, 2, 0, 2.0, 1, 2, 0, 0, 0, 0, 0, 2, 1, 0, 'meson'),
    'J/ψ': (5, 2, 0, 2.0, 1, 8, 0, 0, 0, 0, 0, 2, 1, 0, 'meson'),
    'D0': (6, -3, -2, -0.5, 1, 1, 0.5, -0.5, 1, 0, 0, 1, 0, 0, 'meson'),
    'Υ(1S)': (6, -3, 0, 0.0, 1, 1, 0, 0, 0, 0, 0, 2, 1, 0, 'meson'),
}

FEATURE_NAMES = ['ns', 'hidden_s', 'mixing', 'is_vector', 'g', 'p']


def build_dataset():
    X, y_alpha, y_q = [], [], []
    for name, v in mesons.items():
        n_q, n_qbar = v[4], v[5]
        S, g, spin = v[8], v[11], v[12]

        p = (n_q + n_qbar) / 2
        ns = abs(S)
        hidden_s = 1 if name == 'φ' else 0
        mixing = 1 if name in ['η'] else 0
        is_vector = 1 if spin == 1 else 0

        feats = np.array([ns, hidden_s, mixing, is_vector, g, p])

        alpha = v[1]
        q = v[3]

        delta_alpha = alpha - (-2 * p)
        delta_q = q - (p + g)

        X.append(feats)
        y_alpha.append(delta_alpha)
        y_q.append(delta_q)

    return np.array(X), np.array(y_alpha), np.array(y_q)


def fast_search(X, y, coef_range, max_combinations=2000000):
    """Быстрый поиск с выводом прогресса"""
    n_features = X.shape[1]
    np.random.seed(42)
    best = []

    for i in range(max_combinations):
        coeffs = np.random.choice(coef_range, n_features + 1)
        w = coeffs[:-1]
        b = coeffs[-1]

        preds = X @ w + b
        mae = np.mean(np.abs(preds - y))
        acc = np.mean(np.abs(preds - y) < 1e-6)

        if acc > 0.3:
            best.append((acc, mae, coeffs))

            if len(best) > 100:
                best.sort(key=lambda x: -x[0])
                best = best[:20]

        # Прогресс каждые 50000 итераций
        if (i + 1) % 50000 == 0:
            print(f"  Прогресс: {i + 1}/{max_combinations} ({(i + 1) / max_combinations * 100:.1f}%)")

    best.sort(key=lambda x: (-x[0], x[1]))
    return best[:10]


def print_models(models, target_name, X_data, y_data):
    """ВЫВОД ФОРМУЛ НА ЭКРАН"""
    print("\n" + "=" * 80)
    print(f"ЛУЧШИЕ МОДЕЛИ ДЛЯ {target_name}")
    print("=" * 80)

    if not models:
        print("Модели не найдены (попробуйте увеличить max_combinations или уменьшить порог acc)")
        return

    for i, (acc, mae, coeffs) in enumerate(models, 1):
        print(f"\n{i}. Точность: {acc:.2%} | MAE: {mae:.4f}")

        # Формируем формулу
        terms = []
        for j, (coef, name) in enumerate(zip(coeffs[:-1], FEATURE_NAMES)):
            if abs(coef) > 1e-6:
                if coef > 0 and terms:
                    terms.append(f"+ {coef}*{name}")
                else:
                    terms.append(f"{coef}*{name}")

        formula = " + ".join(terms)
        if abs(coeffs[-1]) > 1e-6:
            if coeffs[-1] > 0 and formula:
                formula += f" + {coeffs[-1]}"
            elif not formula:
                formula = f"{coeffs[-1]}"

        print(f"{target_name} = {formula}")

        # Показываем предсказания для первых 3 частиц
        print(f"  Примеры предсказаний:")
        names = list(mesons.keys())[:3]
        for idx, name in enumerate(names):
            feat = X_data[idx]
            pred = np.dot(feat, coeffs[:-1]) + coeffs[-1]
            actual = y_data[idx]
            print(f"    {name}: предсказано={pred:.3f}, реально={actual:.3f}")


# ============================================================
# ЗАПУСК С ВЫВОДОМ
# ============================================================

print("=" * 80)
print("МЕЗОНЫ 6.0 — БЫСТРЫЙ ПОИСК ФОРМУЛ")
print("=" * 80)

X, y_alpha, y_q = build_dataset()
coef_range = np.arange(-3, 3.5, 0.5)

print("\nИсходные данные:")
print(f"  Объектов: {X.shape[0]}")
print(f"  Признаков: {X.shape[1]}")
print(f"  Диапазон коэффициентов: {coef_range[0]}..{coef_range[-1]} шаг {coef_range[1] - coef_range[0]}")

print("\n" + "=" * 80)
print("ПОИСК ДЛЯ delta_alpha")
print("=" * 80)
models_alpha = fast_search(X, y_alpha, coef_range, max_combinations=4000000)
print_models(models_alpha, "delta_alpha", X, y_alpha)

print("\n" + "=" * 80)
print("ПОИСК ДЛЯ delta_q")
print("=" * 80)
models_q = fast_search(X, y_q, coef_range, max_combinations=4000000)
print_models(models_q, "delta_q", X, y_q)

# Дополнительно: показываем сами значения, которые мы ищем
print("\n" + "=" * 80)
print("РЕАЛЬНЫЕ ЗНАЧЕНИЯ ДЛЯ ПРОВЕРКИ")
print("=" * 80)
print("\ndelta_alpha для каждого мезона:")
for i, name in enumerate(mesons.keys()):
    print(f"  {name:5}: {y_alpha[i]:6.2f}")

print("\ndelta_q для каждого мезона:")
for i, name in enumerate(mesons.keys()):
    print(f"  {name:5}: {y_q[i]:6.2f}")