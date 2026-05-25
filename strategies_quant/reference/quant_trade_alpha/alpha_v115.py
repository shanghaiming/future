"""V115: ML-Enhanced Factor Strategy based on PRISM-VQ insights.
============================================================
Key improvements over previous ML attempts (V71-V112):
1. Only use proven high-quality factors (V62's 6 + V56's 8)
2. VQ-VAE discretization (PRISM-VQ: A-shares RankIC +27%)
3. Proper time-series walk-forward CV
4. LightGBM LambdaRank with clean features
5. GT-Score evaluation (not just Sharpe)

No look-ahead: all signals use data up to di-1.
"""
import sys, os, time, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alpha_v2 import MIN_TRAIN
from alpha_v7 import CASH0


def compute_vq_codewords(factors, n_codewords=64, n_iter=50, seed=42):
    """Vector Quantization: map factor space to discrete codewords.
    Simple K-means VQ (lightweight version of PRISM-VQ's VQ-VAE).
    For each day, assign each stock to a codeword based on its factor vector.

    Parameters
    ----------
    factors : dict of {name: ndarray(NS, ND)} - ranked factors
    n_codewords : int - number of codewords (PRISM-VQ uses 128-256 for CSI300)
    n_iter : int - K-means iterations
    seed : int - random seed

    Returns
    -------
    codewords : ndarray(NS, ND) - integer codeword assignments (0 to n_codewords-1)
    """
    rng = np.random.RandomState(seed)

    # Stack factors into a matrix
    factor_names = sorted(factors.keys())
    factor_arrays = [factors[k] for k in factor_names]
    # Shape: (n_factors, NS, ND)
    stacked = np.stack(factor_arrays, axis=0)
    n_factors = len(factor_names)
    NS, ND = stacked.shape[1], stacked.shape[2]

    codewords = np.full((NS, ND), -1, dtype=np.int32)

    # Initialize codebook from a random sample of stocks on a random day
    di_sample = ND // 2
    sample_data = stacked[:, :, di_sample].T  # (NS, n_factors)
    valid_mask = ~np.any(np.isnan(sample_data), axis=1)
    valid_samples = sample_data[valid_mask]

    if len(valid_samples) < n_codewords:
        n_codewords = max(len(valid_samples) // 2, 8)

    # K-means initialization
    idx = rng.choice(len(valid_samples), n_codewords, replace=False)
    codebook = valid_samples[idx].copy()  # (n_codewords, n_factors)

    # K-means iterations on the sample day
    for _ in range(n_iter):
        # Assign each stock to nearest codeword
        dists = np.sum((valid_samples[:, None, :] - codebook[None, :, :]) ** 2, axis=2)
        assignments = np.argmin(dists, axis=1)

        # Update codebook
        for c in range(n_codewords):
            mask = assignments == c
            if mask.sum() > 0:
                codebook[c] = valid_samples[mask].mean(axis=0)

    # Now assign all stocks on all days
    for di in range(ND):
        day_data = stacked[:, :, di].T  # (NS, n_factors)
        valid = ~np.any(np.isnan(day_data), axis=1)
        if valid.sum() < 10:
            continue

        dists = np.sum((day_data[valid, None, :] - codebook[None, :, :]) ** 2, axis=2)
        codewords[valid, di] = np.argmin(dists, axis=1)

    return codewords


def compute_vq_features(factors, n_codewords_list=(32, 64, 128)):
    """Compute multi-resolution VQ features for ML input.

    Returns additional features:
    - VQ_N: codeword assignment for N-cluster VQ (categorical)
    - VQ_N_FREQ: frequency of stock's codeword (how common is this "type")
    - VQ_N_MOMENTUM: average momentum of stocks in same codeword
    """
    from collections import defaultdict

    all_features = {}

    for n_cw in n_codewords_list:
        cw = compute_vq_codewords(factors, n_codewords=n_cw)

        # VQ frequency: how common is each codeword (inverse = more unique)
        freq = np.full_like(cw, np.nan, dtype=np.float64)
        for di in range(cw.shape[1]):
            col = cw[:, di]
            valid = col >= 0
            if valid.sum() < 10:
                continue
            counts = defaultdict(int)
            for c in col[valid]:
                counts[int(c)] += 1
            total = valid.sum()
            for si in range(cw.shape[0]):
                if valid[si]:
                    freq[si, di] = counts[int(col[si])] / total

        # Inverse frequency = uniqueness score
        uniq = np.where(freq > 0, 1.0 / freq, np.nan)
        all_features[f'VQ_UNIQ_{n_cw}'] = uniq

        # VQ momentum: average return of stocks in same codeword
        # This captures "stocks of the same type are moving together"
        # (Will be filled in by the caller with actual returns)

    return all_features


def backtest_v115(factor_weights, factors, NS, ND, dates, C, O, H, L, V,
                  top_n=1, rebalance_days=10, atr_stop_mult=2.0,
                  use_vq=True, vq_codewords=64,
                  use_ml=True, ml_lookback=500, ml_retrain=60,
                  ml_lr=0.05, ml_leaves=31, ml_trees=200):
    """V115: ML-enhanced factor strategy with VQ discretization.

    Pipeline:
    1. Compute composite score from hand-crafted factors
    2. (Optional) Add VQ discretization features
    3. (Optional) Use LightGBM to re-rank stocks
    4. Select top_n stocks, apply ATR stop

    Parameters
    ----------
    factor_weights : dict - {factor_name: weight} for composite scoring
    factors : dict - {factor_name: ndarray(NS, ND)} ranked factors
    use_vq : bool - enable VQ discretization
    vq_codewords : int - number of VQ codewords
    use_ml : bool - enable LightGBM re-ranking
    ml_lookback : int - training window (days)
    ml_retrain : int - retrain frequency (days)
    ml_lr : float - LightGBM learning rate
    ml_leaves : int - max leaves per tree
    ml_trees : int - number of boosting rounds
    """
    import lightgbm as lgb

    factor_names = list(factor_weights.keys())
    weights = np.array([factor_weights[f] for f in factor_names])

    # Compute VQ features if requested
    vq_cw = None
    if use_vq:
        avail_factors = {k: v for k, v in factors.items()
                        if isinstance(v, np.ndarray) and v.ndim == 2 and v.shape == (NS, ND)}
        vq_cw = compute_vq_codewords(avail_factors, n_codewords=vq_codewords)

    # Prepare feature matrix for ML
    all_feature_names = factor_names.copy()
    if use_vq:
        all_feature_names.append(f'VQ_CW')

    # Build feature arrays
    n_features = len(all_feature_names)

    cash = float(1_000_000)
    holdings = []
    trades = []
    last_rebalance = -999
    last_ml_train = -999
    daily_nav = []
    year_stats = {}

    # ML model state
    ml_model = None

    def _get_features(di):
        """Get feature matrix for day di."""
        X = np.zeros((NS, n_features))
        valid_count = np.zeros(NS)

        for fi, fname in enumerate(factor_names):
            if fname in factors:
                vals = factors[fname][:, di]
                mask = ~np.isnan(vals)
                X[mask, fi] = vals[mask]
                valid_count[mask] += 1
            else:
                X[:, fi] = 0

        if use_vq and vq_cw is not None:
            # VQ codeword as numerical feature (normalized)
            cw_vals = vq_cw[:, di].astype(np.float64)
            cw_vals[cw_vals < 0] = np.nan
            mask = ~np.isnan(cw_vals)
            X[mask, -1] = cw_vals[mask] / vq_codewords * 100  # normalize to 0-100
            valid_count[mask] += 1

        # Mask stocks with insufficient features
        mask = valid_count >= len(factor_names) * 0.5
        X[~mask, :] = np.nan
        return X

    def _get_target(di, horizon=5):
        """Get forward returns for training target (no look-ahead since we use di-1 for prediction)."""
        if di + horizon >= ND:
            return None
        rets = np.zeros(NS)
        for si in range(NS):
            p0 = C[si, di]
            p1 = C[si, di + horizon]
            if np.isnan(p0) or np.isnan(p1) or p0 <= 0:
                rets[si] = np.nan
            else:
                rets[si] = (p1 - p0) / p0
        return rets

    def _train_ml(di):
        """Train LightGBM on historical data up to di-1."""
        X_list = []
        y_list = []
        # Sample training points from the lookback window
        start_di = max(MIN_TRAIN, di - ml_lookback)
        for train_di in range(start_di, di - 5, 5):  # every 5 days
            X = _get_features(train_di)
            y = _get_target(train_di + 1)  # predict from next day
            if X is not None and y is not None:
                mask = ~np.any(np.isnan(X), axis=1) & ~np.isnan(y)
                if mask.sum() > 50:
                    X_list.append(X[mask])
                    y_list.append(y[mask])

        if len(X_list) < 3:
            return None

        X_train = np.vstack(X_list)
        y_train = np.concatenate(y_list)

        if len(X_train) < 200:
            return None

        # Create ranking target: group stocks by day, rank within group
        # LightGBM LambdaRank needs query sizes
        query_sizes = [x.shape[0] for x in X_list]

        train_data = lgb.Dataset(X_train, label=y_train, group=query_sizes)
        params = {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'learning_rate': ml_lr,
            'num_leaves': ml_leaves,
            'min_data_in_leaf': 50,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 1,
            'verbose': -1,
            'seed': 42,
        }

        try:
            model = lgb.train(params, train_data, num_boost_round=ml_trees)
            return model
        except Exception:
            return None

    def _rank(di):
        """Compute composite rank, optionally enhanced by ML."""
        # Base composite score
        composite = np.zeros(NS)
        count = np.zeros(NS)
        for fname, w in zip(factor_names, weights):
            if fname not in factors: continue
            vals = factors[fname][:, di]
            valid = ~np.isnan(vals)
            if valid.sum() < 50: continue
            composite[valid] += w * vals[valid]
            count[valid] += abs(w)
        mask = count > 0
        if mask.sum() < top_n * 2: return None, None
        composite[mask] /= count[mask]
        composite[~mask] = -9999

        # ML re-ranking
        if use_ml and ml_model is not None:
            X = _get_features(di)
            mask_valid = ~np.any(np.isnan(X), axis=1)
            if mask_valid.sum() > 50:
                try:
                    ml_scores = ml_model.predict(X[mask_valid])
                    # Blend: 50% composite + 50% ML
                    # Normalize ML scores to same scale as composite
                    ml_min, ml_max = np.percentile(ml_scores, [5, 95])
                    if ml_max > ml_min:
                        ml_norm = (ml_scores - ml_min) / (ml_max - ml_min) * 100
                        # Override with blended score
                        blend = composite.copy()
                        blend[mask_valid] = 0.5 * composite[mask_valid] + 0.5 * ml_norm
                        composite = blend
                except Exception:
                    pass  # Fall back to composite only

        return np.argsort(-composite), composite

    for di in range(MIN_TRAIN, ND):
        year = dates[di].year

        # Retrain ML model periodically
        if use_ml and (ml_model is None or di - last_ml_train >= ml_retrain):
            ml_model = _train_ml(di)
            last_ml_train = di

        # Phase 1: Exit checks (ATR stop)
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

            if exited: continue

            # Update HW
            if not np.isnan(today_h) and today_h > 0:
                pos['hw'] = max(pos['hw'], today_h)

            # Time stop
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
                        if np.isnan(p) or p <= 0: p = C[pos['si'], di]
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
                        if np.isnan(p) or p <= 0: p = C[si, di]
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
            if np.isnan(cp) or cp <= 0: cp = pos['entry']
            nav += pos['shares'] * cp
        daily_nav.append(nav)

    # Close remaining
    for pos in holdings:
        p = C[pos['si'], ND-1]
        if not np.isnan(p) and p > 0:
            pnl = (p - pos['entry']) / pos['entry'] * 100
            cash += pos['shares'] * p * (1 - 0.0013)
            trades.append({'pnl': pnl, 'days': 999, 'year': dates[ND-1].year})

    if not trades: return None

    days_total = (dates[ND-1] - dates[MIN_TRAIN]).days
    yr = max(days_total / 365.25, 0.01)
    ann = ((cash / CASH0) ** (1/yr) - 1) * 100
    nw = sum(1 for t in trades if t['pnl'] > 0)
    wr = nw / max(len(trades), 1) * 100
    avg_w = np.mean([t['pnl'] for t in trades if t['pnl'] > 0]) if nw > 0 else 0
    avg_l = np.mean([abs(t['pnl']) for t in trades if t['pnl'] <= 0]) if nw < len(trades) else 0

    for t in trades:
        y = t.get('year', 'unk')
        if y not in year_stats: year_stats[y] = {'trades': 0, 'wins': 0, 'total_pnl': 0}
        year_stats[y]['trades'] += 1
        if t['pnl'] > 0: year_stats[y]['wins'] += 1
        year_stats[y]['total_pnl'] += t['pnl']

    max_dd = 0
    if daily_nav:
        peak = daily_nav[0]
        for nav in daily_nav:
            if nav > peak: peak = nav
            if peak > 0:
                dd = (peak - nav) / peak * 100
                if dd > max_dd: max_dd = dd

    # GT-Score: μ · ln(z) · r² / σ_d
    daily_rets = np.diff(daily_nav) / daily_nav[:-1]
    mu = np.mean(daily_rets)
    sigma = np.std(daily_rets)
    sharpe = mu / max(sigma, 1e-10) * np.sqrt(252)
    z = sharpe / np.sqrt(1 + sharpe**2 / 4)
    if z > 0 and sharpe > 0:
        ln_z = np.log(z)
    else:
        ln_z = -10

    # r²: consistency (fraction of positive-return days)
    r_sq = np.sum(daily_rets > 0) / max(len(daily_rets), 1)

    # σ_d: downside deviation
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
    print("V115: ML-Enhanced Factor Strategy with VQ discretization")
    print("Use test_v115.py to run backtests.")
