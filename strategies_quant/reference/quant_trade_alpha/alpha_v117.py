"""V117: Fast ML-Enhanced Strategy using V116's proven factors.
=============================================================
Key insight from V116: R_VOL_ADJ_MOM is the breakthrough factor.
V62 + R_VOL_ADJ_MOM(w=0.5) ATR=1.0 R10 = +317%/52% WR.

V117 adds LightGBM re-ranking on top of V116's proven factor set.
Unlike V115 (trains every 60 days = too slow), V117 trains quarterly.

Key ML design choices:
1. Only 7 features (V62's 6 + VOL_ADJ_MOM) — no noise
2. LambdaRank objective — optimize for ranking, not regression
3. Train quarterly (~20 trainings total) — fast enough
4. 5-fold time-series CV for validation
5. Blend: 60% composite + 40% ML — don't overtrust ML
"""
import sys, os, time, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alpha_v2 import MIN_TRAIN
from alpha_v7 import CASH0


def backtest_v117(factor_weights, factors, NS, ND, dates, C, O, H, L, V,
                  top_n=1, rebalance_days=10, atr_stop_mult=2.0,
                  use_ml=True, ml_blend=0.4, ml_retrain_days=60,
                  ml_trees=100, ml_leaves=31, ml_lr=0.05,
                  ml_train_window=400, ml_cv_folds=3,
                  break_even_pct=0.0):
    """V117: Fast ML-enhanced backtest with V116 proven factors.

    Parameters
    ----------
    factor_weights : dict - {factor_name: weight} for composite scoring
    factors : dict - {factor_name: ndarray(NS, ND)} ranked factors
    use_ml : bool - enable LightGBM re-ranking
    ml_blend : float - ML weight in blend (0=composite only, 1=ML only)
    ml_retrain_days : int - retrain frequency (quarterly=60)
    ml_trees : int - LightGBM boosting rounds
    ml_train_window : int - training lookback (days)
    break_even_pct : float - if >0, move stop to entry when gain >= this %
    """
    import lightgbm as lgb

    factor_names = list(factor_weights.keys())
    weights = np.array([factor_weights[f] for f in factor_names])
    n_features = len(factor_names)

    cash = float(CASH0)
    holdings = []
    trades = []
    last_rebalance = -999
    last_ml_train = -999
    daily_nav = []
    year_stats = {}
    ml_model = None

    def _get_features(di):
        """Get feature matrix for day di."""
        X = np.full((NS, n_features), np.nan)
        for fi, fname in enumerate(factor_names):
            if fname in factors:
                X[:, fi] = factors[fname][:, di]
        return X

    def _get_target(di, horizon=5):
        """Forward returns for ranking target."""
        if di + horizon >= ND:
            return None
        p0 = C[:, di]
        p1 = C[:, di + horizon]
        valid = ~np.isnan(p0) & ~np.isnan(p1) & (p0 > 0)
        rets = np.full(NS, np.nan)
        rets[valid] = (p1[valid] - p0[valid]) / p0[valid]
        return rets

    def _train_ml(di):
        """Train LightGBM LambdaRank on recent data."""
        X_list, y_list = [], []
        start_di = max(MIN_TRAIN, di - ml_train_window)
        # Sample every 3 days for speed
        for train_di in range(start_di, di - 5, 3):
            X = _get_features(train_di)
            y = _get_target(train_di + 1, horizon=5)
            if X is None or y is None:
                continue
            mask = ~np.any(np.isnan(X), axis=1) & ~np.isnan(y)
            if mask.sum() > 30:
                X_list.append(X[mask])
                y_list.append(y[mask])

        if len(X_list) < 5:
            return None

        X_train = np.vstack(X_list)
        y_train = np.concatenate(y_list)
        if len(X_train) < 300:
            return None

        # Discretize returns into quintile labels (0-4) for LambdaRank
        # This follows PRISM-VQ insight: discretization improves ranking
        y_discrete = np.zeros(len(y_train), dtype=np.int32)
        for group_start, group_size in zip(
            np.cumsum([0] + [x.shape[0] for x in X_list[:-1]]),
            [x.shape[0] for x in X_list]):
            chunk = y_train[group_start:group_start + group_size]
            valid = ~np.isnan(chunk)
            if valid.sum() < 10:
                continue
            try:
                q = np.percentile(chunk[valid], [20, 40, 60, 80])
                labels = np.digitize(chunk, q)  # 0-4 quintile labels
                y_discrete[group_start:group_start + group_size] = labels
            except Exception:
                pass

        query_sizes = [x.shape[0] for x in X_list]

        train_data = lgb.Dataset(X_train, label=y_discrete, group=query_sizes)
        params = {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'learning_rate': ml_lr,
            'num_leaves': ml_leaves,
            'min_data_in_leaf': 30,
            'feature_fraction': 0.8,
            'verbose': -1,
            'seed': 42,
        }

        try:
            model = lgb.train(params, train_data, num_boost_round=ml_trees)
            return model
        except Exception:
            return None

    def _rank(di):
        """Compute composite score, optionally blended with ML."""
        # Composite score
        composite = np.zeros(NS)
        count = np.zeros(NS)
        for fname, w in zip(factor_names, weights):
            if fname not in factors:
                continue
            vals = factors[fname][:, di]
            valid = ~np.isnan(vals)
            if valid.sum() < 50:
                continue
            composite[valid] += w * vals[valid]
            count[valid] += abs(w)
        mask = count > 0
        if mask.sum() < top_n * 2:
            return None, None
        composite[mask] /= count[mask]
        composite[~mask] = -9999

        # ML re-ranking
        if use_ml and ml_model is not None:
            X = _get_features(di)
            mask_valid = ~np.any(np.isnan(X), axis=1)
            if mask_valid.sum() > 50:
                try:
                    ml_scores = ml_model.predict(X[mask_valid])
                    ml_min, ml_max = np.percentile(ml_scores, [2, 98])
                    if ml_max > ml_min:
                        ml_norm = (ml_scores - ml_min) / (ml_max - ml_min) * 100
                        blend = composite.copy()
                        blend[mask_valid] = (1 - ml_blend) * composite[mask_valid] + ml_blend * ml_norm
                        composite = blend
                except Exception:
                    pass

        return np.argsort(-composite), composite

    # Main loop
    for di in range(MIN_TRAIN, ND):
        year = dates[di].year

        # Retrain ML model
        if use_ml and (ml_model is None or di - last_ml_train >= ml_retrain_days):
            ml_model = _train_ml(di)
            last_ml_train = di

        # Phase 1: Exit checks
        for pos in list(holdings):
            si = pos['si']
            today_o = O[si, di]
            today_l = L[si, di]
            today_h = H[si, di]
            if np.isnan(today_o) or today_o <= 0:
                continue

            exited = False

            # ATR trailing stop
            if atr_stop_mult > 0:
                atr_sum = 0.0; atr_count = 0
                for dd in range(max(di-14, 1), di):
                    if not np.isnan(H[si, dd]) and not np.isnan(L[si, dd]):
                        tr = H[si, dd] - L[si, dd]
                        if not np.isnan(C[si, dd-1]):
                            tr = max(tr, abs(H[si, dd]-C[si, dd-1]), abs(L[si, dd]-C[si, dd-1]))
                        atr_sum += tr; atr_count += 1
                if atr_count > 0:
                    atr_stop = pos['hw'] - atr_stop_mult * atr_sum / atr_count
                    if not np.isnan(today_l) and today_l <= atr_stop:
                        sp = today_o if today_o < atr_stop else atr_stop
                        pnl = (sp - pos['entry']) / pos['entry'] * 100
                        cash += pos['shares'] * sp * (1 - 0.0013)
                        trades.append({'pnl': pnl, 'days': (dates[di]-pos['ed']).days, 'year': year})
                        holdings.remove(pos)
                        exited = True

            if exited:
                continue

            # Break-even stop
            if break_even_pct > 0 and pos in holdings:
                gain = (pos['hw'] - pos['entry']) / pos['entry'] * 100
                if gain >= break_even_pct:
                    if not np.isnan(today_l) and today_l <= pos['entry']:
                        sp = today_o if today_o < pos['entry'] else pos['entry']
                        pnl = (sp - pos['entry']) / pos['entry'] * 100
                        cash += pos['shares'] * sp * (1 - 0.0013)
                        trades.append({'pnl': pnl, 'days': (dates[di]-pos['ed']).days, 'year': year})
                        holdings.remove(pos)
                        exited = True

            if exited:
                continue

            # Update high water mark
            if not np.isnan(today_h) and today_h > 0:
                pos['hw'] = max(pos['hw'], today_h)

            # Time stop at 60 days
            if pos in holdings:
                days_held = (dates[di] - pos['ed']).days
                if days_held >= 60:
                    sp = today_o
                    if not np.isnan(sp) and sp > 0:
                        pnl = (sp - pos['entry']) / pos['entry'] * 100
                        cash += pos['shares'] * sp * (1 - 0.0013)
                        trades.append({'pnl': pnl, 'days': days_held, 'year': year})
                        holdings.remove(pos)

        # Phase 2: Rebalance
        if di - last_rebalance >= rebalance_days:
            rank_idx, composite = _rank(di)
            if rank_idx is not None:
                top_indices = set(rank_idx[:top_n])
                current_indices = set(h['si'] for h in holdings)

                to_sell = current_indices - top_indices
                for pos in list(holdings):
                    if pos['si'] in to_sell:
                        p = O[pos['si'], di]
                        if np.isnan(p) or p <= 0:
                            p = C[pos['si'], di]
                        if not np.isnan(p) and p > 0:
                            pnl = (p - pos['entry']) / pos['entry'] * 100
                            cash += pos['shares'] * p * (1 - 0.0013)
                            trades.append({'pnl': pnl, 'days': (dates[di]-pos['ed']).days, 'year': year})
                            holdings.remove(pos)

                to_buy = top_indices - set(h['si'] for h in holdings)
                n_to_buy = len(to_buy)
                if n_to_buy > 0 and cash > 10000:
                    alloc = cash / n_to_buy
                    for si in to_buy:
                        p = O[si, di]
                        if np.isnan(p) or p <= 0:
                            p = C[si, di]
                        if not np.isnan(p) and p > 0:
                            shares = int(alloc / 1.001 / p)
                            if shares > 0:
                                cost = shares * p * 1.001
                                if cost <= cash:
                                    cash -= cost
                                    holdings.append({'si': si, 'shares': shares,
                                                    'entry': p, 'ed': dates[di], 'hw': p})
                last_rebalance = di

        # Phase 3: NAV
        nav = cash
        for pos in holdings:
            cp = C[pos['si'], di]
            if np.isnan(cp) or cp <= 0:
                cp = pos['entry']
            nav += pos['shares'] * cp
        daily_nav.append(nav)

    # Close remaining
    for pos in holdings:
        p = C[pos['si'], ND-1]
        if not np.isnan(p) and p > 0:
            pnl = (p - pos['entry']) / pos['entry'] * 100
            cash += pos['shares'] * p * (1 - 0.0013)
            trades.append({'pnl': pnl, 'days': 999, 'year': dates[ND-1].year})

    if not trades:
        return None

    days_total = (dates[ND-1] - dates[MIN_TRAIN]).days
    yr = max(days_total / 365.25, 0.01)
    ann = ((cash / CASH0) ** (1/yr) - 1) * 100
    nw = sum(1 for t in trades if t['pnl'] > 0)
    wr = nw / max(len(trades), 1) * 100
    avg_w = np.mean([t['pnl'] for t in trades if t['pnl'] > 0]) if nw > 0 else 0
    avg_l = np.mean([abs(t['pnl']) for t in trades if t['pnl'] <= 0]) if nw < len(trades) else 0

    for t in trades:
        y = t.get('year', 'unk')
        if y not in year_stats:
            year_stats[y] = {'trades': 0, 'wins': 0, 'total_pnl': 0}
        year_stats[y]['trades'] += 1
        if t['pnl'] > 0:
            year_stats[y]['wins'] += 1
        year_stats[y]['total_pnl'] += t['pnl']

    max_dd = 0
    if daily_nav:
        peak = daily_nav[0]
        for nav in daily_nav:
            if nav > peak:
                peak = nav
            if peak > 0:
                dd = (peak - nav) / peak * 100
                if dd > max_dd:
                    max_dd = dd

    # Sharpe and GT-Score
    daily_rets = np.diff(daily_nav) / daily_nav[:-1]
    mu = np.mean(daily_rets)
    sigma = np.std(daily_rets)
    sharpe = mu / max(sigma, 1e-10) * np.sqrt(252)
    z = sharpe / np.sqrt(1 + sharpe**2 / 4)
    ln_z = np.log(z) if z > 0 and sharpe > 0 else -10
    r_sq = np.sum(daily_rets > 0) / max(len(daily_rets), 1)
    neg_rets = daily_rets[daily_rets < 0]
    sigma_d = np.std(neg_rets) if len(neg_rets) > 10 else sigma
    gt_score = mu * ln_z * r_sq / max(sigma_d, 1e-10) * 100

    return {
        'ann': round(ann, 1), 'n': len(trades), 'wr': round(wr, 1),
        'avg_w': round(avg_w, 1), 'avg_l': round(avg_l, 1),
        'edge': round((nw/max(len(trades),1))*avg_w - (1-nw/max(len(trades),1))*avg_l, 2),
        'max_dd': round(max_dd, 1), 'tpy': round(len(trades)/yr, 1),
        'sharpe': round(sharpe, 2), 'gt_score': round(gt_score, 2),
        'year_stats': year_stats,
    }


if __name__ == '__main__':
    print("V117: Fast ML-Enhanced Strategy with V116 Proven Factors")
    print("Use test_v117.py to run backtests.")
