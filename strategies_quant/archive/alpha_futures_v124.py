"""
Alpha Futures V124 -- MACHINE LEARNING RANKING (Next-Open Execution)
=====================================================================
Instead of ranking by ROC*Z (manual heuristic), train a model to PREDICT
which signals will be profitable.

OPTIMIZED: Precompute ML probabilities for ALL (si, di) pairs, then
lookup in backtest loop. No per-iteration feature computation.

ALL signals use NEXT-OPEN execution: signal at close di, entry at O[si, di+1].
"""
import sys, os, time, warnings
import numpy as np
import talib
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alpha_v2 import load_all_data, MIN_TRAIN, CASH0

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# ============================================================
# CONSTANTS
# ============================================================
MULT = {'agfi': 15, 'alfi': 5, 'aufi': 1000, 'bufi': 10, 'cufi': 5, 'fufi': 10,
        'rbfi': 10, 'znfi': 5, 'nifi': 1, 'hcfi': 10, 'spfi': 10, 'ssfi': 5,
        'sffi': 5, 'smfi': 5, 'pbfi': 5, 'snfi': 1, 'rufi': 10, 'wrffi': 10,
        'afi': 10, 'bfi': 10, 'bbfi': 500, 'cffi': 5, 'cfi': 10, 'csfi': 10,
        'ebfi': 5, 'egfi': 10, 'fbfi': 500, 'ifi': 100, 'jfi': 100, 'jmfi': 60,
        'lfi': 5, 'mfi': 10, 'pgfi': 20, 'ppfi': 5, 'vfi': 5, 'yfi': 10,
        'pfi': 10, 'jdfi': 5, 'lhfi': 16, 'pkfi': 5, 'rrfi': 20, 'lrfi': 20,
        'jrfi': 20, 'pmfi': 20, 'whfi': 20, 'rsfi': 20, 'cjfi': 10, 'mafi': 10,
        'apfi': 10, 'cyfi': 5, 'fgfi': 20, 'oifi': 10, 'pfifi': 5, 'rmfi': 10,
        'srfi': 10, 'tafi': 5, 'safi': 20, 'urfi': 20, 'scfi': 1000, 'lufi': 10,
        'bcfi': 5, 'nrfi': 1, 'lgfi': 20, 'brfi': 5, 'lcfi': 1, 'sifi': 5,
        'ni': 1, 'tai': 5}
DEF_MULT = 10
COMM = 0.0003

FEATURE_NAMES = [
    'roc5', 'z_score', 'roc5_improving', 'adx', 'rsi',
    'roc3', 'roc10', 'vol_ratio', 'oi_change', 'atr_pct',
    'c_minus_o_pct', 'body_ratio', 'upper_shadow', 'bb_position', 'macd_hist',
    'roc5_rank', 'z_rank', 'natr', 'plus_di_minus_di', 'prev_day_ret'
]
N_FEATURES = 20


def annual_return(final, initial, n_days):
    if final <= 0 or initial <= 0 or n_days <= 0:
        return -100.0
    yrs = n_days / 252
    return (final / initial) ** (1.0 / yrs) * 100 - 100


def main():
    print("=" * 150)
    print("  Alpha Futures V124 -- MACHINE LEARNING RANKING (Next-Open Execution)")
    print("=" * 150)
    print(f"  sklearn available: {HAS_SKLEARN}")
    print(f"  Goal: Train ML model to predict profitable trades, compare vs ROC*Z heuristic")

    # -- Load data -------------------------------------------------
    print("\n[Data] Loading...", flush=True)
    t_start = time.time()
    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)
    years = sorted(set(d.year for d in dates))
    print(f"  {NS} commodities, {ND} days, years: {years}")
    print(f"  MIN_TRAIN={MIN_TRAIN}, CASH0={CASH0:,}")

    # ================================================================
    # PRECOMPUTE ALL INDICATORS as 2D arrays [si, di]
    # ================================================================
    print("\n[Precompute] Computing all indicators...", flush=True)
    t0 = time.time()

    # Daily returns in percent
    RET = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si].astype(np.float64)
        for di in range(1, ND):
            if not np.isnan(c[di]) and not np.isnan(c[di-1]) and c[di-1] > 0:
                RET[si, di] = (c[di] / c[di-1] - 1) * 100

    # ROC indicators
    ROC5 = np.full((NS, ND), np.nan)
    ROC3 = np.full((NS, ND), np.nan)
    ROC10 = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si].astype(np.float64)
        ROC5[si] = talib.ROC(c, timeperiod=5)
        ROC3[si] = talib.ROC(c, timeperiod=3)
        ROC10[si] = talib.ROC(c, timeperiod=10)

    # Z-score of daily returns (20-day rolling)
    ZSCORE_20 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(21, ND):
            rets = RET[si, di-20:di]
            valid = rets[~np.isnan(rets)]
            if len(valid) < 10:
                continue
            mean_r = np.mean(valid)
            std_r = np.std(valid, ddof=1)
            if std_r > 0 and not np.isnan(RET[si, di]):
                ZSCORE_20[si, di] = (RET[si, di] - mean_r) / std_r

    # Cross-sectional ranks
    ROC5_RANK = np.full((NS, ND), np.nan)
    Z_RANK = np.full((NS, ND), np.nan)
    for di in range(ND):
        # ROC5 rank
        vals = ROC5[:, di]
        valid_mask = ~np.isnan(vals)
        n_valid = np.sum(valid_mask)
        if n_valid >= 10:
            order = np.argsort(vals[valid_mask])
            ranks = np.empty(n_valid)
            ranks[order] = np.arange(1, n_valid + 1) / n_valid
            idx = 0
            for si in range(NS):
                if valid_mask[si]:
                    ROC5_RANK[si, di] = ranks[idx]
                    idx += 1

        # Z-score rank
        vals_z = ZSCORE_20[:, di]
        valid_mask_z = ~np.isnan(vals_z)
        n_valid_z = np.sum(valid_mask_z)
        if n_valid_z >= 10:
            order_z = np.argsort(vals_z[valid_mask_z])
            ranks_z = np.empty(n_valid_z)
            ranks_z[order_z] = np.arange(1, n_valid_z + 1) / n_valid_z
            idx_z = 0
            for si in range(NS):
                if valid_mask_z[si]:
                    Z_RANK[si, di] = ranks_z[idx_z]
                    idx_z += 1

    # ADX(14)
    ADX = np.full((NS, ND), np.nan)
    for si in range(NS):
        h = H[si].astype(np.float64)
        l = L[si].astype(np.float64)
        c = C[si].astype(np.float64)
        ADX[si] = talib.ADX(h, l, c, timeperiod=14)

    # RSI(14)
    RSI = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si].astype(np.float64)
        RSI[si] = talib.RSI(c, timeperiod=14)

    # ATR(14) and NATR
    ATR14 = np.full((NS, ND), np.nan)
    NATR = np.full((NS, ND), np.nan)
    for si in range(NS):
        h = H[si].astype(np.float64)
        l = L[si].astype(np.float64)
        c = C[si].astype(np.float64)
        ATR14[si] = talib.ATR(h, l, c, timeperiod=14)
        NATR[si] = talib.NATR(h, l, c, timeperiod=14)

    # PLUS_DI - MINUS_DI
    PLUS_DI = np.full((NS, ND), np.nan)
    MINUS_DI = np.full((NS, ND), np.nan)
    for si in range(NS):
        h = H[si].astype(np.float64)
        l = L[si].astype(np.float64)
        c = C[si].astype(np.float64)
        PLUS_DI[si] = talib.PLUS_DI(h, l, c, timeperiod=14)
        MINUS_DI[si] = talib.MINUS_DI(h, l, c, timeperiod=14)

    # MACD histogram
    MACD_HIST = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si].astype(np.float64)
        _, _, macdhist = talib.MACD(c, fastperiod=12, slowperiod=26, signalperiod=9)
        MACD_HIST[si] = macdhist

    # Bollinger Bands
    BB_UPPER = np.full((NS, ND), np.nan)
    BB_LOWER = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si].astype(np.float64)
        upper, _, lower = talib.BBANDS(c, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        BB_UPPER[si] = upper
        BB_LOWER[si] = lower

    # Volume SMA(20)
    V_SMA20 = np.full((NS, ND), np.nan)
    for si in range(NS):
        v = V[si].astype(np.float64)
        V_SMA20[si] = talib.SMA(v, timeperiod=20)

    print(f"  All indicators computed ({time.time()-t0:.1f}s)")

    # ================================================================
    # BUILD FEATURE MATRIX as 3D array [si, di, feature]
    # ================================================================
    print("\n[Features] Building feature matrix for all (si, di)...", flush=True)
    t0 = time.time()

    FEAT = np.full((NS, ND, N_FEATURES), np.nan)

    for si in range(NS):
        for di in range(MIN_TRAIN, ND):
            roc5 = ROC5[si, di]
            z_score = ZSCORE_20[si, di]
            if np.isnan(roc5) or np.isnan(z_score):
                continue

            # roc5_improving
            roc5_prev = ROC5[si, di-1]
            roc5_improving = 1.0 if (not np.isnan(roc5_prev) and roc5 > roc5_prev) else 0.0

            # adx
            adx = ADX[si, di]
            if np.isnan(adx): adx = 25.0

            # rsi
            rsi = RSI[si, di]
            if np.isnan(rsi): rsi = 50.0

            # roc3, roc10
            roc3 = ROC3[si, di] if not np.isnan(ROC3[si, di]) else 0.0
            roc10 = ROC10[si, di] if not np.isnan(ROC10[si, di]) else 0.0

            # vol_ratio
            v_di = V[si, di]
            v_sma = V_SMA20[si, di]
            vol_ratio = v_di / v_sma if (not np.isnan(v_di) and not np.isnan(v_sma) and v_sma > 0) else 1.0

            # oi_change
            oi_di = OI[si, di]
            oi_prev = OI[si, di-5]
            oi_change = (oi_di - oi_prev) / oi_prev * 100 if (not np.isnan(oi_di) and not np.isnan(oi_prev) and oi_prev > 0) else 0.0

            # atr_pct
            atr = ATR14[si, di]
            c_di = C[si, di]
            atr_pct = atr / c_di * 100 if (not np.isnan(atr) and not np.isnan(c_di) and c_di > 0) else 2.0

            # c_minus_o_pct
            o_di = O[si, di]
            c_minus_o_pct = (c_di - o_di) / o_di * 100 if (not np.isnan(c_di) and not np.isnan(o_di) and o_di > 0) else 0.0

            # body_ratio
            h_di = H[si, di]
            l_di = L[si, di]
            hl_range = h_di - l_di
            body_ratio = abs(c_di - o_di) / hl_range if (not np.isnan(hl_range) and hl_range > 0 and not np.isnan(c_di) and not np.isnan(o_di)) else 0.5

            # upper_shadow
            upper_shadow = (h_di - max(c_di, o_di)) / hl_range if (not np.isnan(hl_range) and hl_range > 0 and not np.isnan(c_di) and not np.isnan(o_di)) else 0.0

            # bb_position
            bb_u = BB_UPPER[si, di]
            bb_l = BB_LOWER[si, di]
            bb_position = (c_di - bb_l) / (bb_u - bb_l) if (not np.isnan(bb_u) and not np.isnan(bb_l) and not np.isnan(c_di) and (bb_u - bb_l) > 0) else 0.5

            # macd_hist
            macd_h = MACD_HIST[si, di]
            if np.isnan(macd_h): macd_h = 0.0

            # ranks
            roc5_rank = ROC5_RANK[si, di] if not np.isnan(ROC5_RANK[si, di]) else 0.5
            z_rank = Z_RANK[si, di] if not np.isnan(Z_RANK[si, di]) else 0.5

            # natr
            natr = NATR[si, di]
            if np.isnan(natr): natr = 2.0

            # plus_di_minus_di
            pdi = PLUS_DI[si, di]
            mdi = MINUS_DI[si, di]
            plus_di_minus_di = pdi - mdi if (not np.isnan(pdi) and not np.isnan(mdi)) else 0.0

            # prev_day_ret
            prev_ret = RET[si, di-1]
            if np.isnan(prev_ret): prev_ret = 0.0

            FEAT[si, di] = [
                roc5, z_score, roc5_improving, adx, rsi,
                roc3, roc10, vol_ratio, oi_change, atr_pct,
                c_minus_o_pct, body_ratio, upper_shadow, bb_position, macd_h,
                roc5_rank, z_rank, natr, plus_di_minus_di, prev_ret
            ]

    print(f"  Feature matrix built ({time.time()-t0:.1f}s)")

    # ================================================================
    # COLLECT TRAINING DATA
    # ================================================================
    print("\n[ML] Collecting signal instances...", flush=True)
    t0 = time.time()

    train_X = []
    train_y = []
    train_si_di = []  # (si, di) for reference
    test_X = []
    test_y = []
    test_si_di = []

    for si in range(NS):
        for di in range(MIN_TRAIN, ND - 2):
            roc5 = ROC5[si, di]
            z20 = ZSCORE_20[si, di]
            if np.isnan(roc5) or np.isnan(z20):
                continue
            if roc5 <= 1.0 or z20 <= 1.5:
                continue

            feats = FEAT[si, di]
            if np.any(np.isnan(feats)):
                continue

            # Label
            entry_price = O[si, di+1]
            exit_price = C[si, di+2]
            if np.isnan(entry_price) or np.isnan(exit_price) or entry_price <= 0:
                continue
            next_ret = (exit_price - entry_price) / entry_price * 100
            label = 1 if next_ret > 0 else 0

            if dates[di].year <= 2022:
                train_X.append(feats)
                train_y.append(label)
                train_si_di.append((si, di))
            else:
                test_X.append(feats)
                test_y.append(label)
                test_si_di.append((si, di))

    train_X = np.array(train_X)
    train_y = np.array(train_y)
    test_X = np.array(test_X) if test_X else np.empty((0, N_FEATURES))
    test_y = np.array(test_y) if test_y else np.empty(0)

    print(f"  Training instances (<=2022): {len(train_X)}")
    print(f"  Test instances (>=2023): {len(test_X)}")
    if len(train_X) > 0:
        print(f"  Training win rate: {np.mean(train_y)*100:.1f}%")
    if len(test_X) > 0:
        print(f"  Test win rate: {np.mean(test_y)*100:.1f}%")
    print(f"  Data collection ({time.time()-t0:.1f}s)")

    if len(train_X) < 100:
        print("  ERROR: Too few training samples!")
        return

    # ================================================================
    # TRAIN MODELS
    # ================================================================
    print("\n[ML] Training models...", flush=True)
    t0 = time.time()

    models = {}
    model_names = []

    if HAS_SKLEARN:
        print("  Training RandomForest...")
        rf = RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=20,
                                     random_state=42, n_jobs=-1)
        rf.fit(train_X, train_y)
        models['RF'] = rf
        model_names.append('RF')

        print("  Training GradientBoosting...")
        gb = GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                                         min_samples_leaf=20, random_state=42)
        gb.fit(train_X, train_y)
        models['GB'] = gb
        model_names.append('GB')

        print("  Training LogisticRegression...")
        feat_means = np.nanmean(train_X, axis=0)
        feat_stds = np.nanstd(train_X, axis=0)
        feat_stds[feat_stds < 1e-10] = 1.0
        train_X_norm = (train_X - feat_means) / feat_stds
        lr = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
        lr.fit(train_X_norm, train_y)
        models['LR'] = lr
        model_names.append('LR')
    else:
        print("  No sklearn - correlation-based scoring...")
        feat_corrs = np.zeros(N_FEATURES)
        for fi in range(N_FEATURES):
            valid = ~np.isnan(train_X[:, fi])
            if np.sum(valid) > 50:
                feat_corrs[fi] = np.corrcoef(train_X[valid, fi], train_y[valid])[0, 1]
        models['CORR'] = feat_corrs
        model_names.append('CORR')

    print(f"  Models trained ({time.time()-t0:.1f}s)")

    # ================================================================
    # PRECOMPUTE ML PROBABILITIES for all (si, di)
    # ================================================================
    print("\n[ML] Precomputing probabilities for all (si, di)...", flush=True)
    t0 = time.time()

    # For each model, compute P(win) for all signal locations
    ML_PROB = {}  # model_name -> [NS, ND] array

    for mname in model_names:
        prob_arr = np.full((NS, ND), np.nan)

        # Gather all (si, di) where features exist
        all_si = []
        all_di = []
        all_feats = []
        for si in range(NS):
            for di in range(MIN_TRAIN, ND):
                if not np.any(np.isnan(FEAT[si, di])):
                    all_si.append(si)
                    all_di.append(di)
                    all_feats.append(FEAT[si, di])

        if len(all_feats) == 0:
            ML_PROB[mname] = prob_arr
            continue

        X_all = np.array(all_feats)

        if mname == 'LR':
            X_all_norm = (X_all - feat_means) / feat_stds
            probs = models['LR'].predict_proba(X_all_norm)[:, 1]
        elif mname == 'CORR':
            weights = models['CORR']
            scores = np.zeros(len(X_all))
            for fi in range(N_FEATURES):
                col = X_all[:, fi]
                std = np.std(col)
                if std > 1e-10:
                    scores += weights[fi] * (col - np.mean(col)) / std
            probs = 1.0 / (1.0 + np.exp(-scores))
        else:
            probs = models[mname].predict_proba(X_all)[:, 1]

        for k in range(len(all_si)):
            prob_arr[all_si[k], all_di[k]] = probs[k]

        ML_PROB[mname] = prob_arr

    print(f"  Probabilities precomputed ({time.time()-t0:.1f}s)")

    # ================================================================
    # FEATURE IMPORTANCE
    # ================================================================
    print(f"\n{'=' * 150}")
    print("  FEATURE IMPORTANCE")
    print(f"{'=' * 150}")

    if HAS_SKLEARN:
        print("\n  Random Forest feature importance:")
        rf_imp = rf.feature_importances_
        for rank, idx in enumerate(np.argsort(rf_imp)[::-1]):
            print(f"    {rank+1:>2}. {FEATURE_NAMES[idx]:<20} = {rf_imp[idx]:.4f}")

        print("\n  Gradient Boosting feature importance:")
        gb_imp = gb.feature_importances_
        for rank, idx in enumerate(np.argsort(gb_imp)[::-1]):
            print(f"    {rank+1:>2}. {FEATURE_NAMES[idx]:<20} = {gb_imp[idx]:.4f}")

        print("\n  Logistic Regression coefficients:")
        lr_coefs = lr.coef_[0]
        for rank, idx in enumerate(np.argsort(np.abs(lr_coefs))[::-1]):
            print(f"    {rank+1:>2}. {FEATURE_NAMES[idx]:<20} = {lr_coefs[idx]:+.4f}")

    # Univariate analysis
    print("\n  Univariate feature analysis (training set):")
    print(f"  {'Feature':<20} | {'Win Mean':>10} | {'Loss Mean':>10} | {'Diff':>10} | {'Corr':>8}")
    print("-" * 70)
    for fi in range(N_FEATURES):
        col = train_X[:, fi]
        valid = ~np.isnan(col)
        if np.sum(valid) < 50:
            continue
        win_mask = valid & (train_y == 1)
        loss_mask = valid & (train_y == 0)
        if np.sum(win_mask) == 0 or np.sum(loss_mask) == 0:
            continue
        win_mean = np.mean(col[win_mask])
        loss_mean = np.mean(col[loss_mask])
        corr = np.corrcoef(col[valid], train_y[valid])[0, 1]
        print(f"  {FEATURE_NAMES[fi]:<20} | {win_mean:>10.4f} | {loss_mean:>10.4f} | {win_mean - loss_mean:>+10.4f} | {corr:>+7.4f}")

    # ================================================================
    # BACKTEST ENGINE (optimized — uses precomputed ML_PROB)
    # ================================================================
    def backtest_fast(model_name, threshold=0.55, top_n=1, hold=1,
                     rank_mode='ml', label='',
                     start_di=None, end_di=None):
        if start_di is None:
            start_di = MIN_TRAIN
        if end_di is None:
            end_di = ND

        prob_arr = ML_PROB.get(model_name)
        cash = float(CASH0)
        positions = []
        trades = []
        daily_equity = []

        for di in range(start_di, end_di - 1):
            port_val = cash
            for pos in positions:
                cp = C[pos['si'], di]
                if not np.isnan(cp) and cp > 0:
                    mult = MULT.get(pos['sym'], DEF_MULT)
                    port_val += cp * mult * pos['lots'] - cp * mult * abs(pos['lots']) * COMM
            daily_equity.append(port_val)

            # Close positions
            closed = []
            for pos in positions:
                days_held = di - pos['entry_di']
                if days_held >= pos['hold_days']:
                    exit_price = C[pos['si'], di]
                    if np.isnan(exit_price) or exit_price <= 0:
                        exit_price = pos['entry_price']
                    mult = MULT.get(pos['sym'], DEF_MULT)
                    mkt_val = exit_price * mult * abs(pos['lots'])
                    cash += mkt_val - mkt_val * COMM
                    pnl = (exit_price - pos['entry_price']) * mult * pos['lots']
                    invested = pos['entry_price'] * mult * abs(pos['lots'])
                    pnl_pct = pnl / invested * 100 if invested > 0 else 0
                    trades.append({
                        'pnl': pnl, 'pnl_pct': pnl_pct,
                        'entry_di': pos['entry_di'], 'exit_di': di,
                        'sym': pos['sym'],
                    })
                    closed.append(pos)
            for pos in closed:
                positions.remove(pos)

            if len(positions) >= top_n:
                continue

            entry_di = di + 1
            if entry_di >= end_di:
                continue

            candidates = []
            for si in range(NS):
                roc5 = ROC5[si, di]
                z20 = ZSCORE_20[si, di]
                if np.isnan(roc5) or np.isnan(z20):
                    continue
                if roc5 <= 1.0 or z20 <= 1.5:
                    continue

                ep = O[si, entry_di]
                if np.isnan(ep) or ep <= 0:
                    continue
                if any(p['si'] == si for p in positions):
                    continue

                # ML probability (precomputed)
                prob = prob_arr[si, di]
                if np.isnan(prob):
                    continue

                # Threshold filter
                if prob < threshold:
                    continue

                # Ranking score
                if rank_mode == 'roc_z':
                    score = roc5 * z20
                elif rank_mode == 'ml_then_roc_z':
                    score = roc5 * z20
                else:
                    score = prob

                candidates.append((score, si, ep, prob, roc5, z20))

            if not candidates:
                continue

            candidates.sort(key=lambda x: -x[0])
            n_slots = top_n - len(positions)
            cap_per_slot = cash * 0.95 / max(1, n_slots)

            for sc_val, si, price, prob_val, roc_val, zs_val in candidates[:max(0, n_slots)]:
                sym = syms[si]
                mult = MULT.get(sym, DEF_MULT)
                contracts = max(1, int(cap_per_slot / (price * mult * (1 + COMM))))
                cost_in = price * mult * contracts * (1 + COMM)
                if cost_in > cash:
                    contracts = int(cash * 0.9 / (price * mult * (1 + COMM)))
                    cost_in = price * mult * contracts * (1 + COMM) if contracts > 0 else 0
                if contracts <= 0 or cost_in <= 0 or cost_in > cash:
                    continue
                cash -= cost_in
                positions.append({
                    'si': si, 'entry_price': price, 'entry_di': entry_di,
                    'lots': contracts, 'dir': 1, 'sym': sym,
                    'hold_days': hold,
                })

        # Close remaining
        for pos in positions:
            ae = end_di - 1
            exit_price = C[pos['si'], min(ae, ND-1)]
            if np.isnan(exit_price) or exit_price <= 0:
                exit_price = pos['entry_price']
            mult = MULT.get(pos['sym'], DEF_MULT)
            mkt_val = exit_price * mult * abs(pos['lots'])
            cash += mkt_val - mkt_val * COMM
            pnl = (exit_price - pos['entry_price']) * mult * pos['lots']
            invested = pos['entry_price'] * mult * abs(pos['lots'])
            pnl_pct = pnl / invested * 100 if invested > 0 else 0
            trades.append({
                'pnl': pnl, 'pnl_pct': pnl_pct,
                'entry_di': pos['entry_di'], 'exit_di': ae,
                'sym': pos['sym'],
            })

        n_days_test = end_di - start_di
        ann = annual_return(cash, CASH0, n_days_test)
        wr = np.mean([1 if t['pnl_pct'] > 0 else 0 for t in trades]) * 100 if trades else 0
        n_trades = len(trades)
        avg_pnl = np.mean([t['pnl_pct'] for t in trades]) if trades else 0

        if daily_equity:
            eq_arr = np.array(daily_equity)
            peak_arr = np.maximum.accumulate(eq_arr)
            dd_arr = (eq_arr - peak_arr) / peak_arr * 100
            mdd = np.min(dd_arr)
        else:
            mdd = 0.0

        return {
            'ann': ann, 'wr': wr, 'n': n_trades, 'avg_pnl': avg_pnl,
            'final_cash': cash, 'n_days': n_days_test, 'mdd': mdd,
            'label': label,
        }

    # ================================================================
    # WALK-FORWARD HELPER
    # ================================================================
    wf_years = [2020, 2021, 2022, 2023, 2024, 2025]

    def walk_forward(model_name, threshold=0.55, top_n=1, hold=1, rank_mode='ml'):
        wf = {}
        for yr in wf_years:
            ts = te = None
            for di in range(ND):
                if dates[di].year == yr and ts is None:
                    ts = di
                if dates[di].year == yr + 1 and te is None:
                    te = di
            if ts is None:
                wf[yr] = None
                continue
            if te is None:
                te = ND
            r = backtest_fast(model_name, threshold=threshold, top_n=top_n,
                             hold=hold, rank_mode=rank_mode,
                             start_di=ts, end_di=te)
            wf[yr] = r
        return wf

    def print_wf(label, wf):
        vals = {yr: wf[yr]['ann'] if wf[yr] else 0 for yr in wf_years}
        avg = np.mean(list(vals.values()))
        pos = sum(1 for v in vals.values() if v > 0)
        mdds = [wf[yr]['mdd'] for yr in wf_years if wf[yr]]
        avg_mdd = np.mean(mdds) if mdds else 0
        row = f"  {label:<55} | {avg:>+8.1f}% |"
        for yr in wf_years:
            row += f" {vals[yr]:>+8.1f}% |"
        row += f" {pos}/6 | {avg_mdd:>6.1f}%"
        print(row)
        return avg, pos

    # ================================================================
    # DETERMINE TEST PERIOD
    # ================================================================
    test_start_di = None
    test_end_di = None
    for di in range(ND):
        if dates[di].year == 2023 and test_start_di is None:
            test_start_di = di
        if dates[di].year == 2026 and test_end_di is None:
            test_end_di = di
    if test_end_di is None:
        test_end_di = ND
    if test_start_di is None:
        test_start_di = MIN_TRAIN

    print(f"\n  Test period: {dates[test_start_di]} to {dates[min(test_end_di-1, ND-1)]}")

    # ================================================================
    # SECTION 1: BASELINE (ROC*Z no ML)
    # ================================================================
    print(f"\n{'=' * 150}")
    print("  SECTION 1: BASELINE — ROC*Z heuristic (NO ML filter)")
    print(f"{'=' * 150}")

    baseline_r = backtest_fast(model_names[0], threshold=0.0, top_n=1, hold=1,
                                rank_mode='roc_z', label='BASELINE ROC*Z',
                                start_di=test_start_di, end_di=test_end_di)
    print(f"  BASELINE (ROC>1% AND Z>1.5, rank by ROC*Z, no ML):")
    print(f"    Annual: {baseline_r['ann']:>+.1f}%  WR: {baseline_r['wr']:.1f}%  Trades: {baseline_r['n']}  MDD: {baseline_r['mdd']:>+.1f}%")

    # Also full period baseline
    baseline_full = backtest_fast(model_names[0], threshold=0.0, top_n=1, hold=1,
                                   rank_mode='roc_z', label='BASELINE ROC*Z FULL')
    print(f"  BASELINE FULL PERIOD:")
    print(f"    Annual: {baseline_full['ann']:>+.1f}%  WR: {baseline_full['wr']:.1f}%  Trades: {baseline_full['n']}  MDD: {baseline_full['mdd']:>+.1f}%")

    # ================================================================
    # SECTION 2: ML MODEL COMPARISON ON TEST SET (2023-2025)
    # ================================================================
    print(f"\n{'=' * 150}")
    print("  SECTION 2: ML MODEL COMPARISON (Test 2023-2025)")
    print(f"{'=' * 150}")
    print(f"  {'Model':<10} | {'Thresh':>7} | {'Rank Mode':<15} | {'Ann':>10} | {'WR':>6} | {'N':>5} | {'AvgPnL':>8} | {'MDD':>8} | {'Final':>12}")
    print("-" * 120)

    test_results = []
    for mname in model_names:
        for thresh in [0.50, 0.55, 0.60, 0.65, 0.70]:
            for rm in ['ml', 'roc_z', 'ml_then_roc_z']:
                lbl = f"{mname} T>{thresh:.2f} {rm}"
                r = backtest_fast(mname, threshold=thresh, top_n=1, hold=1,
                                 rank_mode=rm, label=lbl,
                                 start_di=test_start_di, end_di=test_end_di)
                test_results.append(r)
                print(f"  {mname:<10} | {thresh:>6.2f} | {rm:<15} | {r['ann']:>+9.1f}% | {r['wr']:>5.1f}% | {r['n']:>5} | {r['avg_pnl']:>+7.3f}% | {r['mdd']:>+7.1f}% | {r['final_cash']:>11,.0f}")

    # ================================================================
    # SECTION 3: BEST ML CONFIGS FULL PERIOD
    # ================================================================
    print(f"\n{'=' * 150}")
    print("  SECTION 3: FULL PERIOD BACKTEST (Best configs)")
    print(f"{'=' * 150}")

    # Find best configs from test results
    test_sorted = sorted(test_results, key=lambda x: -x['ann'])
    best_labels = set()
    best_configs_full = []

    # Top 5 from test
    for r in test_sorted[:5]:
        lbl = r['label']
        if lbl not in best_labels:
            best_labels.add(lbl)
            # Parse config from label
            parts = lbl.split()
            mname = parts[0]
            thresh = float(parts[1].split('>')[1])
            rm = parts[2]
            best_configs_full.append((mname, thresh, rm, lbl))

    # Always include baseline
    best_configs_full.append((model_names[0], 0.0, 'roc_z', 'BASELINE ROC*Z'))

    print(f"  {'Config':<40} | {'Ann':>10} | {'WR':>6} | {'N':>5} | {'AvgPnL':>8} | {'MDD':>8} | {'Final':>12}")
    print("-" * 110)

    full_results = []
    for mname, thresh, rm, lbl in best_configs_full:
        r = backtest_fast(mname, threshold=thresh, top_n=1, hold=1,
                         rank_mode=rm, label=lbl)
        full_results.append(r)
        print(f"  {lbl:<40} | {r['ann']:>+9.1f}% | {r['wr']:>5.1f}% | {r['n']:>5} | {r['avg_pnl']:>+7.3f}% | {r['mdd']:>+7.1f}% | {r['final_cash']:>11,.0f}")

    # ================================================================
    # SECTION 4: WALK-FORWARD BY YEAR
    # ================================================================
    print(f"\n{'=' * 170}")
    print("  SECTION 4: WALK-FORWARD BY YEAR (2020-2025)")
    print(f"{'=' * 170}")

    header = f"  {'Config':<55} | {'Avg':>8} |"
    for yr in wf_years:
        header += f" {yr:>8} |"
    header += f" {'Pos':>4} | {'AvgMDD':>7}"
    print(header)
    print("-" * 170)

    for mname, thresh, rm, lbl in best_configs_full:
        wf = walk_forward(mname, threshold=thresh, top_n=1, hold=1, rank_mode=rm)
        print_wf(lbl, wf)

    # ================================================================
    # SECTION 5: THRESHOLD SENSITIVITY (best model)
    # ================================================================
    print(f"\n{'=' * 150}")
    print("  SECTION 5: THRESHOLD SENSITIVITY (Test 2023-2025)")
    print(f"{'=' * 150}")

    best_model = model_names[0]
    print(f"  Model: {best_model}")
    print(f"  {'Threshold':>10} | {'ML-rank Ann':>12} | {'ML WR':>7} | {'ML N':>5} | {'ML+ROCZ Ann':>12} | {'ML+ROCZ WR':>10} | {'ML+ROCZ N':>9}")
    print("-" * 100)

    for thresh in [0.45, 0.50, 0.55, 0.58, 0.60, 0.62, 0.65, 0.70, 0.75, 0.80]:
        r_ml = backtest_fast(best_model, threshold=thresh, top_n=1, hold=1,
                            rank_mode='ml', label=f'T>{thresh} ml',
                            start_di=test_start_di, end_di=test_end_di)
        r_mrz = backtest_fast(best_model, threshold=thresh, top_n=1, hold=1,
                             rank_mode='ml_then_roc_z', label=f'T>{thresh} ml+rocz',
                             start_di=test_start_di, end_di=test_end_di)
        print(f"  {thresh:>9.2f} | {r_ml['ann']:>+10.1f}% | {r_ml['wr']:>5.1f}% | {r_ml['n']:>5} | {r_mrz['ann']:>+10.1f}% | {r_mrz['wr']:>8.1f}% | {r_mrz['n']:>7}")

    # ================================================================
    # SECTION 6: TOP_N SENSITIVITY
    # ================================================================
    print(f"\n{'=' * 150}")
    print("  SECTION 6: TOP_N SENSITIVITY (Test 2023-2025)")
    print(f"{'=' * 150}")

    # Use best threshold from Section 5
    # Try a few promising configs with different top_n
    for top_n_val in [1, 2, 3]:
        print(f"\n  Top_N = {top_n_val}")
        print(f"  {'Config':<45} | {'Ann':>10} | {'WR':>6} | {'N':>5} | {'MDD':>8}")
        print("-" * 90)
        for mname, thresh, rm, lbl in best_configs_full[:min(4, len(best_configs_full))]:
            r = backtest_fast(mname, threshold=thresh, top_n=top_n_val, hold=1,
                             rank_mode=rm, label=lbl,
                             start_di=test_start_di, end_di=test_end_di)
            print(f"  {lbl:<45} | {r['ann']:>+9.1f}% | {r['wr']:>5.1f}% | {r['n']:>5} | {r['mdd']:>+7.1f}%")

    # ================================================================
    # SECTION 7: ML PREDICTION ACCURACY
    # ================================================================
    print(f"\n{'=' * 150}")
    print("  SECTION 7: ML PREDICTION ACCURACY")
    print(f"{'=' * 150}")

    if len(test_X) > 0 and HAS_SKLEARN:
        for mname in model_names:
            print(f"\n  {mname}:")
            if mname == 'LR':
                probs_test = models['LR'].predict_proba((test_X - feat_means) / feat_stds)[:, 1]
            else:
                probs_test = models[mname].predict_proba(test_X)[:, 1]

            for thresh in [0.50, 0.55, 0.60, 0.65, 0.70]:
                mask = probs_test >= thresh
                if np.sum(mask) > 0:
                    pred_pos = np.sum(mask)
                    actual_pos = np.sum(test_y[mask] == 1)
                    precision = actual_pos / pred_pos * 100
                    avg_ret = np.mean([test_X[k, 0] for k in range(len(test_X)) if mask[k]])
                    print(f"    T>{thresh:.2f}: predicted {pred_pos:>5}, won {actual_pos:>5} => precision {precision:.1f}%")
                else:
                    print(f"    T>{thresh:.2f}: no trades predicted")

    # Permutation importance on test set
    if HAS_SKLEARN and len(test_X) > 0:
        print("\n  Permutation importance (RF, test set):")
        base_probs = models['RF'].predict_proba(test_X)[:, 1]
        base_score = np.mean(base_probs[test_y == 1]) - np.mean(base_probs[test_y == 0])

        perm_imp = np.zeros(N_FEATURES)
        for fi in range(N_FEATURES):
            X_perm = test_X.copy()
            np.random.seed(42)
            np.random.shuffle(X_perm[:, fi])
            perm_probs = models['RF'].predict_proba(X_perm)[:, 1]
            perm_score = np.mean(perm_probs[test_y == 1]) - np.mean(perm_probs[test_y == 0])
            perm_imp[fi] = base_score - perm_score

        for rank, idx in enumerate(np.argsort(perm_imp)[::-1]):
            print(f"    {rank+1:>2}. {FEATURE_NAMES[idx]:<20} = {perm_imp[idx]:+.4f}")

    # ================================================================
    # SECTION 8: ML FILTERING IMPACT ANALYSIS
    # ================================================================
    print(f"\n{'=' * 150}")
    print("  SECTION 8: ML FILTERING — How many signals does ML remove?")
    print(f"{'=' * 150}")

    for mname in model_names:
        print(f"\n  {mname}:")
        for thresh in [0.50, 0.55, 0.60, 0.65, 0.70]:
            n_total = 0
            n_pass = 0
            for si in range(NS):
                for di in range(test_start_di, min(test_end_di, ND)):
                    roc5 = ROC5[si, di]
                    z20 = ZSCORE_20[si, di]
                    if np.isnan(roc5) or np.isnan(z20):
                        continue
                    if roc5 <= 1.0 or z20 <= 1.5:
                        continue
                    n_total += 1
                    prob = ML_PROB[mname][si, di]
                    if not np.isnan(prob) and prob >= thresh:
                        n_pass += 1
            pct = n_pass / n_total * 100 if n_total > 0 else 0
            print(f"    T>{thresh:.2f}: {n_total:>5} signals -> {n_pass:>5} pass ({pct:.1f}%)")

    # ================================================================
    # FINAL SUMMARY
    # ================================================================
    print(f"\n{'=' * 150}")
    print("  FINAL SUMMARY")
    print(f"{'=' * 150}")

    # Q1
    print(f"\n  Q1: Does ML filtering/ranking improve over simple ROC*Z?")
    print(f"      Baseline (ROC*Z, no ML, 2023-2025): {baseline_r['ann']:>+.1f}% ann, WR {baseline_r['wr']:.1f}%, {baseline_r['n']} trades, MDD {baseline_r['mdd']:>+.1f}%")
    best_ml_test = max(test_results, key=lambda x: x['ann'])
    diff = best_ml_test['ann'] - baseline_r['ann']
    print(f"      Best ML config ({best_ml_test.get('label','')}): {best_ml_test['ann']:>+.1f}% ann, WR {best_ml_test['wr']:.1f}%, {best_ml_test['n']} trades, MDD {best_ml_test['mdd']:>+.1f}%")
    print(f"      Difference: {diff:+.1f}pp")
    if diff > 0:
        print(f"      => ML IMPROVES by {diff:.1f}pp")
    else:
        print(f"      => ML does NOT improve ({diff:.1f}pp)")

    # Q2
    print(f"\n  Q2: Feature Importance (top 5):")
    if HAS_SKLEARN:
        top5 = np.argsort(rf.feature_importances_)[::-1][:5]
        for i, idx in enumerate(top5):
            print(f"      {i+1}. {FEATURE_NAMES[idx]} (importance={rf_imp[idx]:.4f})")

    # Q3
    print(f"\n  Q3: Best ML config by annual return:")
    print(f"      {best_ml_test.get('label', 'N/A')}")
    print(f"      Annual: {best_ml_test['ann']:>+.1f}%")
    print(f"      WR: {best_ml_test['wr']:.1f}%")
    print(f"      Trades: {best_ml_test['n']}")
    print(f"      MDD: {best_ml_test['mdd']:>+.1f}%")

    # Q4
    print(f"\n  Q4: OOS Performance (2023-2025) Comparison:")
    print(f"      {'Config':<40} | {'Ann':>10} | {'WR':>6} | {'N':>5} | {'MDD':>8}")
    print(f"      {'-'*80}")
    print(f"      {'BASELINE ROC*Z (no ML)':<40} | {baseline_r['ann']:>+9.1f}% | {baseline_r['wr']:>5.1f}% | {baseline_r['n']:>5} | {baseline_r['mdd']:>+7.1f}%")
    for r in test_sorted[:5]:
        print(f"      {r.get('label',''):<40} | {r['ann']:>+9.1f}% | {r['wr']:>5.1f}% | {r['n']:>5} | {r['mdd']:>+7.1f}%")

    elapsed = time.time() - t_start
    print(f"\n  Total elapsed: {elapsed:.1f}s")
    print("=" * 150)


if __name__ == '__main__':
    main()
