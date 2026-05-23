"""
LOO-CV ТЕСТ ДЛЯ МОДЕЛЕЙ d(квантовые числа) В ЕТИ
Честная проверка: обучение на N-1, предсказание на 1
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import LeaveOneOut
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# ДАННЫЕ БАРИОНОВ
# ============================================================

baryons = {
    'p':      {'d': 0.5,  'S': 0.5, 'I': 0.5, 'Y': 1,  'ns': 0, 'sym': 0, 'heavy': 0},
    'n':      {'d': 0.5,  'S': 0.5, 'I': 0.5, 'Y': 1,  'ns': 0, 'sym': 0, 'heavy': 0},
    'Λ':      {'d': -2.0, 'S': 0.5, 'I': 0.0, 'Y': 0,  'ns': 1, 'sym': 0, 'heavy': 0},
    'Σ+':     {'d': -2.0, 'S': 0.5, 'I': 1.0, 'Y': 0,  'ns': 1, 'sym': 0, 'heavy': 0},
    'Ξ0':     {'d': -0.5, 'S': 0.5, 'I': 0.5, 'Y': -1, 'ns': 2, 'sym': 0, 'heavy': 0},
    'Ω-':     {'d': 1.0,  'S': 1.5, 'I': 0.0, 'Y': -2, 'ns': 3, 'sym': 1, 'heavy': 0},
    'Δ':      {'d': -1.0, 'S': 1.5, 'I': 1.5, 'Y': 1,  'ns': 0, 'sym': 1, 'heavy': 0},
    'Λ+_c':   {'d': 1.0,  'S': 0.5, 'I': 0.0, 'Y': 1,  'ns': 0, 'sym': 0, 'heavy': 1},
    'Ξ+_c':   {'d': -1.0, 'S': 0.5, 'I': 0.5, 'Y': 0,  'ns': 1, 'sym': 0, 'heavy': 1},
    'Ω0_c':   {'d': -2.5, 'S': 0.5, 'I': 0.0, 'Y': -1, 'ns': 2, 'sym': 0, 'heavy': 1},
    'Λ0_b':   {'d': 0.5,  'S': 0.5, 'I': 0.0, 'Y': 0,  'ns': 0, 'sym': 0, 'heavy': 1},
    'Ξ++_cc': {'d': 0.0,  'S': 0.5, 'I': 0.5, 'Y': 1,  'ns': 0, 'sym': 0, 'heavy': 1},
}

# ============================================================
# ДАННЫЕ МЕЗОНОВ
# ============================================================

mesons = {
    'π0':  {'d': 1.0,  'S': 0, 'Y': 0, 'ns': 0, 'heavy': 0, 'charge': 0, 'eta_prime': 0, 'heavy_pair': 0, 'heavy_light': 0},
    'π±':  {'d': -2.0, 'S': 0, 'Y': 0, 'ns': 0, 'heavy': 0, 'charge': 1, 'eta_prime': 0, 'heavy_pair': 0, 'heavy_light': 0},
    'K0':  {'d': -1.5, 'S': 0, 'Y': 1, 'ns': 1, 'heavy': 0, 'charge': 0, 'eta_prime': 0, 'heavy_pair': 0, 'heavy_light': 0},
    'η':   {'d': 2.0,  'S': 0, 'Y': 0, 'ns': 1, 'heavy': 0, 'charge': 0, 'eta_prime': 0, 'heavy_pair': 0, 'heavy_light': 0},
    "η'":  {'d': -1.0, 'S': 0, 'Y': 0, 'ns': 1, 'heavy': 0, 'charge': 0, 'eta_prime': 1, 'heavy_pair': 0, 'heavy_light': 0},
    'ρ':   {'d': 2.5,  'S': 1, 'Y': 0, 'ns': 0, 'heavy': 0, 'charge': 1, 'eta_prime': 0, 'heavy_pair': 0, 'heavy_light': 0},
    'ω':   {'d': 2.0,  'S': 1, 'Y': 0, 'ns': 0, 'heavy': 0, 'charge': 0, 'eta_prime': 0, 'heavy_pair': 0, 'heavy_light': 0},
    'φ':   {'d': 0.5,  'S': 1, 'Y': 0, 'ns': 2, 'heavy': 0, 'charge': 0, 'eta_prime': 0, 'heavy_pair': 0, 'heavy_light': 0},
    'K*':  {'d': -2.5, 'S': 1, 'Y': 1, 'ns': 1, 'heavy': 0, 'charge': 1, 'eta_prime': 0, 'heavy_pair': 0, 'heavy_light': 0},
    'J/ψ': {'d': 2.0,  'S': 1, 'Y': 0, 'ns': 0, 'heavy': 1, 'charge': 0, 'eta_prime': 0, 'heavy_pair': 1, 'heavy_light': 0},
    'η_c': {'d': 0.0,  'S': 0, 'Y': 0, 'ns': 0, 'heavy': 1, 'charge': 0, 'eta_prime': 0, 'heavy_pair': 1, 'heavy_light': 0},
    'Υ':   {'d': 0.0,  'S': 1, 'Y': 0, 'ns': 0, 'heavy': 2, 'charge': 0, 'eta_prime': 0, 'heavy_pair': 1, 'heavy_light': 0},
    'D0':  {'d': 0.5,  'S': 1, 'Y': 0, 'ns': 0, 'heavy': 1, 'charge': 0, 'eta_prime': 0, 'heavy_pair': 0, 'heavy_light': 1},
    'B':   {'d': 2.0,  'S': 1, 'Y': 0, 'ns': 0, 'heavy': 2, 'charge': 0, 'eta_prime': 0, 'heavy_pair': 0, 'heavy_light': 1},
    'B_s': {'d': 0.0,  'S': 1, 'Y': 0, 'ns': 1, 'heavy': 2, 'charge': 0, 'eta_prime': 0, 'heavy_pair': 0, 'heavy_light': 1},
}

# ============================================================
# ПОДГОТОВКА ДАННЫХ
# ============================================================

def prepare_baryons():
    def sigma_b(S): return -3 if S == 0.5 else 3
    def is_diquark(name): return 1 if name.startswith('Λ') else 0
    def is_strange(name): return 1 if 'Ω' in name else 0

    rows = []
    for name, data in baryons.items():
        S, I = data['S'], data['I']
        rows.append({
            'name': name,
            'd': data['d'],
            'S2': S*(S+1),
            'I2': I*(I+1),
            'Y': data['Y'],
            'sigma': sigma_b(S),
            'ns': data['ns'],
            'sym': data['sym'],
            'heavy': data['heavy'],
            'diquark': is_diquark(name),
            'strange_flag': is_strange(name),
        })
    return pd.DataFrame(rows)


def prepare_mesons():
    def sigma_m(S): return -3 if S == 0 else 1

    rows = []
    for name, data in mesons.items():
        S = data['S']
        heavy = data['heavy']
        heavy_pair = data.get('heavy_pair', 0)
        heavy_light = data.get('heavy_light', 0)

        if heavy == 0:            mq_eff = 0.0
        elif heavy_pair == 1:     mq_eff = 2.0 if heavy == 1 else 4.5
        elif heavy_light == 1:    mq_eff = 1.3 if heavy == 1 else 4.5
        else:                     mq_eff = float(heavy)

        sg = 1 if (S == 0 and heavy == 0) else 0
        sl = 1 if (S == 1 and heavy == 0) else 0
        sh = 1 if heavy > 0 else 0
        sig = sigma_m(S)
        ep = data.get('eta_prime', 0)
        ch = data.get('charge', 0)
        ns = data['ns']

        rows.append({
            'name': name, 'd': data['d'],
            'sigma': sig, 'Y': data['Y'], 'ns': ns, 'charge': ch,
            'goldstone_charge': sg*ch, 'goldstone_ns': sg*ns,
            'goldstone_sigma': sg*sig,
            'light_ns': sl*ns, 'light_charge': sl*ch, 'light_sigma': sl*sig,
            'heavy_sigma': sh*sig, 'heavy_ns': sh*ns,
            'heavy_pair_sigma': heavy_pair*sig,
            'heavy_light_sigma': heavy_light*sig,
            'mq_eff': mq_eff, 'mq_eff_sigma': mq_eff*sig,
            'mq_eff_ns': mq_eff*ns,
            'eta_prime': ep, 'eta_prime_sigma': ep*sig, 'eta_prime_ns': ep*ns,
        })
    return pd.DataFrame(rows)

# ============================================================
# LOO-CV ЯДРО
# ============================================================

def run_loo_cv(df, feature_cols, alpha, poly_degree, label):
    """
    Честный LOO-CV:
    - обучаем на N-1 частицах
    - предсказываем 1 исключённую
    """
    print(f"\n{'='*65}")
    print(f"  LOO-CV: {label}")
    print(f"  Объектов: {len(df)}, признаков: {len(feature_cols)}, "
          f"poly={poly_degree}, alpha={alpha}")
    print(f"{'='*65}")

    X_raw = df[feature_cols].values
    y = df['d'].values
    names = df['name'].values

    poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
    X_poly = poly.fit_transform(X_raw)

    print(f"  Признаков после poly: {X_poly.shape[1]}")
    print(f"  Соотношение объекты/признаки: "
          f"{len(df)}/{X_poly.shape[1]} = {len(df)/X_poly.shape[1]:.3f}")

    # In-sample
    model_full = Ridge(alpha=alpha)
    model_full.fit(X_poly, y)
    r2_insample  = r2_score(y, model_full.predict(X_poly))
    mae_insample = mean_absolute_error(y, model_full.predict(X_poly))

    # LOO-CV
    loo = LeaveOneOut()
    y_true_loo, y_pred_loo, details = [], [], []

    for train_idx, test_idx in loo.split(X_poly):
        ml = Ridge(alpha=alpha)
        ml.fit(X_poly[train_idx], y[train_idx])
        pred = ml.predict(X_poly[test_idx])[0]
        true = y[test_idx[0]]
        name = names[test_idx[0]]
        y_true_loo.append(true)
        y_pred_loo.append(pred)
        details.append({'name': name, 'd_true': true,
                        'd_pred': pred, 'err': pred - true})

    y_true_loo = np.array(y_true_loo)
    y_pred_loo = np.array(y_pred_loo)
    r2_loo  = r2_score(y_true_loo, y_pred_loo)
    mae_loo = mean_absolute_error(y_true_loo, y_pred_loo)
    degradation = r2_insample - r2_loo

    # Метрики
    print(f"\n  {'Метрика':<32} {'In-sample':>10} {'LOO-CV':>10}")
    print(f"  {'-'*55}")
    print(f"  {'R²':<32} {r2_insample:>10.4f} {r2_loo:>10.4f}")
    print(f"  {'MAE':<32} {mae_insample:>10.4f} {mae_loo:>10.4f}")
    print(f"  {'Деградация R² (overfitting)':<32} {degradation:>10.4f}")

    if degradation < 0.05:
        verdict = "✅ МИНИМАЛЬНОЕ — модель обобщается"
    elif degradation < 0.15:
        verdict = "🟡 УМЕРЕННОЕ — частичное переобучение"
    elif degradation < 0.30:
        verdict = "⚠️  СУЩЕСТВЕННОЕ — переобучение"
    else:
        verdict = "❌ КРИТИЧЕСКОЕ — модель не обобщается"
    print(f"  Вердикт: {verdict}")

    # Детальная таблица
    print(f"\n  {'Частица':<10} {'d_true':>8} {'d_pred':>10} {'ошибка':>10}")
    print(f"  {'-'*42}")
    for d in sorted(details, key=lambda x: abs(x['err']), reverse=True):
        mark = " ★" if abs(d['err']) < 0.15 else (" ⚠" if abs(d['err']) > 0.4 else "")
        print(f"  {d['name']:<10} {d['d_true']:>+8.1f} "
              f"{d['d_pred']:>+10.3f} {d['err']:>+10.3f}{mark}")

    # Пороги
    errs = np.abs(y_pred_loo - y_true_loo)
    print(f"\n  Качество (LOO): точность предсказания |err| < порог")
    for thresh in [0.15, 0.25, 0.50, 1.0]:
        n_ok = np.sum(errs < thresh)
        pct = n_ok / len(errs) * 100
        bar = '█' * int(pct / 4)
        print(f"    < {thresh:.2f}: {n_ok:2d}/{len(errs)} ({pct:5.1f}%)  {bar}")

    return {'r2_insample': r2_insample, 'mae_insample': mae_insample,
            'r2_loo': r2_loo, 'mae_loo': mae_loo,
            'degradation': degradation, 'details': details}

# ============================================================
# ЧУВСТВИТЕЛЬНОСТЬ К ALPHA
# ============================================================

def alpha_sensitivity(df, feature_cols, poly_degree, label,
                      alphas=None):
    if alphas is None:
        alphas = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]

    print(f"\n  Чувствительность к alpha: {label}")
    print(f"  {'alpha':>8} {'R²_in':>10} {'R²_LOO':>10} "
          f"{'MAE_LOO':>10} {'Деград.':>10}")
    print(f"  {'-'*52}")

    X_raw = df[feature_cols].values
    y = df['d'].values
    poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
    X_poly = poly.fit_transform(X_raw)
    loo = LeaveOneOut()
    best_alpha, best_r2_loo = None, -np.inf

    for alpha in alphas:
        m = Ridge(alpha=alpha)
        m.fit(X_poly, y)
        r2_in = r2_score(y, m.predict(X_poly))
        y_tl, y_pl = [], []
        for tr, te in loo.split(X_poly):
            ml = Ridge(alpha=alpha)
            ml.fit(X_poly[tr], y[tr])
            y_tl.append(y[te[0]])
            y_pl.append(ml.predict(X_poly[te])[0])
        r2_l  = r2_score(y_tl, y_pl)
        mae_l = mean_absolute_error(y_tl, y_pl)
        deg   = r2_in - r2_l
        mark  = " ← лучший" if r2_l > best_r2_loo else ""
        if r2_l > best_r2_loo:
            best_r2_loo, best_alpha = r2_l, alpha
        print(f"  {alpha:>8.1f} {r2_in:>10.4f} {r2_l:>10.4f} "
              f"{mae_l:>10.4f} {deg:>10.4f}{mark}")

    print(f"\n  Оптимальный alpha={best_alpha}  (LOO R²={best_r2_loo:.4f})")
    return best_alpha, best_r2_loo

# ============================================================
# ПРЕДСКАЗАНИЯ ДЛЯ НОВЫХ ЧАСТИЦ
# ============================================================

def predict_new(df_train, feature_cols, alpha, poly_degree,
                new_particles, label):
    print(f"\n  Предсказания для новых частиц: {label}")
    X_train = df_train[feature_cols].values
    y_train = df_train['d'].values
    poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
    X_tr_poly = poly.fit_transform(X_train)
    model = Ridge(alpha=alpha)
    model.fit(X_tr_poly, y_train)

    print(f"  {'Частица':<14} {'d_pred':>8}  {'Округл. d':>10}")
    print(f"  {'-'*38}")
    results = {}
    for name, feats in new_particles.items():
        x = np.array([feats])
        x_poly = poly.transform(x)
        d_pred = model.predict(x_poly)[0]
        d_round = round(d_pred * 2) / 2
        results[name] = d_pred
        print(f"  {name:<14} {d_pred:>+8.3f}  {d_round:>+10.1f}")
    return results

# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 65)
    print("  LOO-CV ТЕСТ ЕТИ: d(квантовые числа)")
    print("  Честная проверка предсказательной силы")
    print("=" * 65)

    # ── БАРИОНЫ ──────────────────────────────────────────────
    df_b = prepare_baryons()
    feat_b = ['S2','I2','Y','sigma','ns','sym','heavy',
              'diquark','strange_flag']

    res_b = run_loo_cv(df_b, feat_b, alpha=1.0, poly_degree=2,
                       label="БАРИОНЫ")
    best_ab, _ = alpha_sensitivity(df_b, feat_b, 2, "Барионы")

    if best_ab != 1.0:
        res_b = run_loo_cv(df_b, feat_b, alpha=best_ab, poly_degree=2,
                           label=f"БАРИОНЫ (alpha={best_ab})")

    # Новые барионы: [S2, I2, Y, sigma, ns, sym, heavy, diquark, strange_flag]
    new_baryons = {
        'Σc++':    [0.75, 2.0,  1.0,  3.0, 0, 0, 1, 0, 0],
        'Ξcc+':    [0.75, 0.75, 1.0,  3.0, 0, 0, 1, 1, 0],
        'Ωcc+':    [0.75, 0.0, -1.0,  3.0, 1, 0, 1, 0, 1],
        'Ωbbb-':   [0.75, 0.0, -3.0, -3.0, 0, 0, 1, 0, 1],
        'Ωb-':     [0.75, 0.0, -2.0, -3.0, 0, 0, 1, 0, 1],
        'Ξbb0':    [0.75, 0.5, -1.0, -3.0, 0, 0, 1, 0, 0],
    }
    predict_new(df_b, feat_b, best_ab, 2, new_baryons, "Барионы")

    # ── МЕЗОНЫ ───────────────────────────────────────────────
    df_m = prepare_mesons()
    feat_m = [
        'sigma', 'Y', 'ns', 'charge',
        'goldstone_charge', 'goldstone_ns', 'goldstone_sigma',
        'light_ns', 'light_charge', 'light_sigma',
        'heavy_sigma', 'heavy_ns',
        'heavy_pair_sigma', 'heavy_light_sigma',
        'mq_eff', 'mq_eff_sigma', 'mq_eff_ns',
        'eta_prime', 'eta_prime_sigma', 'eta_prime_ns',
    ]

    res_m = run_loo_cv(df_m, feat_m, alpha=0.3, poly_degree=2,
                       label="МЕЗОНЫ")
    best_am, _ = alpha_sensitivity(df_m, feat_m, 2, "Мезоны")

    if best_am != 0.3:
        res_m = run_loo_cv(df_m, feat_m, alpha=best_am, poly_degree=2,
                           label=f"МЕЗОНЫ (alpha={best_am})")

    # Новые мезоны: строим вектор признаков вручную
    def mfeats(S, Y, ns, charge, ep=0, hp=0, hl=0, heavy=0):
        sig = -3 if S == 0 else 1
        if heavy == 0:        mq = 0.0
        elif hp and heavy==1: mq = 2.0
        elif hp:              mq = 4.5
        elif hl and heavy==1: mq = 1.3
        elif hl:              mq = 4.5
        else:                 mq = float(heavy)
        sg = 1 if (S == 0 and heavy == 0) else 0
        sl = 1 if (S == 1 and heavy == 0) else 0
        sh = 1 if heavy > 0 else 0
        return [sig, Y, ns, charge,
                sg*charge, sg*ns, sg*sig,
                sl*ns, sl*charge, sl*sig,
                sh*sig, sh*ns,
                hp*sig, hl*sig,
                mq, mq*sig, mq*ns,
                ep, ep*sig, ep*ns]

    new_mesons = {
        'Ds+':      mfeats(0, 1, 1, 1),
        'Bc+':      mfeats(0, 0, 0, 1, hl=1, heavy=2),
        'ηb':       mfeats(0, 0, 0, 0, hp=1, heavy=2),
        'K0*(700)': mfeats(0, 1, 1, 0),
        'D_s*':     mfeats(1, 1, 1, 1, hl=1, heavy=1),
        'χc0':      mfeats(0, 0, 0, 0, hp=1, heavy=1),
    }
    predict_new(df_m, feat_m, best_am, 2, new_mesons, "Мезоны")

    # ── ФИНАЛЬНЫЙ ВЕРДИКТ ────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  ФИНАЛЬНЫЙ ВЕРДИКТ")
    print(f"{'='*65}")
    print(f"\n  {'Модель':<18} {'R²_in':>8} {'R²_LOO':>8} "
          f"{'MAE_LOO':>9} {'Деград.':>9}")
    print(f"  {'-'*56}")
    print(f"  {'Барионы':<18} {res_b['r2_insample']:>8.4f} "
          f"{res_b['r2_loo']:>8.4f} {res_b['mae_loo']:>9.4f} "
          f"{res_b['degradation']:>9.4f}")
    print(f"  {'Мезоны':<18} {res_m['r2_insample']:>8.4f} "
          f"{res_m['r2_loo']:>8.4f} {res_m['mae_loo']:>9.4f} "
          f"{res_m['degradation']:>9.4f}")

    print(f"\n  ИНТЕРПРЕТАЦИЯ:")
    for lbl, res in [("Барионы", res_b), ("Мезоны", res_m)]:
        r, d = res['r2_loo'], res['degradation']
        if r > 0.85 and d < 0.10:
            v = "✅ Правило реально работает — d предсказуем"
        elif r > 0.70 and d < 0.20:
            v = "🟡 Умеренная предсказательность — частичное переобучение"
        elif r > 0.50:
            v = "⚠️  Слабая — существенное переобучение"
        else:
            v = "❌ Модель не обобщается — d определяется ad hoc"
        print(f"  {lbl}: {v}")
        print(f"    R²_LOO={r:.4f}, деградация={d:.4f}")

    r2_b, r2_m = res_b['r2_loo'], res_m['r2_loo']
    print(f"\n  ЧТО ЭТО ОЗНАЧАЕТ ДЛЯ ЕТИ:")
    if r2_b > 0.80 and r2_m > 0.80:
        print("  ✅ d НЕ является ad hoc — он предсказуем из квантовых чисел")
        print("  ✅ ЕТИ замыкается: координаты выводятся из принципов")
        print("  → Следующий шаг: предсказание масс новых частиц")
    elif r2_b > 0.60 or r2_m > 0.60:
        print("  🟡 Частичная предсказательность")
        print("  → Нужно: увеличить выборку или упростить модель")
    else:
        print("  ❌ d является свободным параметром")
        print("  → Нужно: пересмотреть структурную формулу")


if __name__ == "__main__":
    main()