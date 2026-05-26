"""
V40: Breadth-Filtered Cross-Sectional Rank Mean Reversion
==========================================================
Extension of V18: adds market breadth confirmation to rank-based MR.

Thesis: V18 doesn't consider overall market health. Mean reversion works
best when the broader commodity market is also oversold -- many commodities
declining simultaneously indicates systemic oversold conditions, not
idiosyncratic noise.

Signal architecture:
  1. Same V18 cross-sectional rank: 7 factors, composite score
  2. Market breadth indicators:
     a. Advance/Decline ratio: % of commodities with positive 5d return
     b. Average composite rank: mean rank across all 50 commodities
     c. New lows count: # of commodities at 20d low
  3. Breadth filter rules:
     - STRONG_BEARISH (A/D < 0.3, avg_rank > 0.60): best for MR, full sizing
     - MODERATE_BEARISH (A/D 0.3-0.45): good for MR, normal sizing
     - NEUTRAL (A/D 0.45-0.55): skip
     - BULLISH (A/D > 0.55): definitely skip
  4. Entry: high composite rank AND breadth confirms bearish conditions
  5. KER gate, hold 5d, ATR stop 3.0

Signal at close[di], enter at open[di+1]. No look-ahead. No gap signals.
No leverage.
"""
import sys
import os
import time
import warnings
import numpy as np
import pandas as pd
from collections import defaultdict

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from alpha_futures_data import load_all_data

try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False

CASH0 = 1_000_000
COMM = 0.0005

DEFAULT_WEIGHTS = {
    'rank_ret5d':  0.25,
    'rank_oi5d':   0.20,
    'rank_rsi':    0.15,
    'rank_vol':    0.15,
    'rank_ret10d': 0.10,
    'rank_range':  0.10,
    'rank_atrp':   0.05,
}

# Breadth regime constants
BREADTH_STRONG_BEARISH = 2
BREADTH_MODERATE_BEARISH = 1
BREADTH_NEUTRAL = 0
BREADTH_BULLISH = -1


# ============================================================
# FACTOR COMPUTATION (same as V18)
# ============================================================
def compute_rsi_manual(C, NS, ND, period=14):
    rsi = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si]
        gains = np.full(ND, np.nan)
        losses = np.full(ND, np.nan)
        for di in range(1, ND):
            if np.isnan(c[di]) or np.isnan(c[di - 1]):
                continue
            delta = c[di] - c[di - 1]
            gains[di] = max(delta, 0.0)
            losses[di] = max(-delta, 0.0)

        avg_gain = np.nan
        avg_loss = np.nan
        for di in range(1, ND):
            if np.isnan(gains[di]):
                continue
            if np.isnan(avg_gain):
                valid_g = []
                valid_l = []
                for j in range(di, min(di + period, ND)):
                    if not np.isnan(gains[j]):
                        valid_g.append(gains[j])
                        valid_l.append(losses[j] if not np.isnan(losses[j]) else 0.0)
                if len(valid_g) >= period:
                    avg_gain = np.mean(valid_g)
                    avg_loss = np.mean(valid_l)
                    if avg_loss == 0:
                        rsi[si, di + period - 1] = 100.0
                    else:
                        rs = avg_gain / avg_loss
                        rsi[si, di + period - 1] = 100.0 - 100.0 / (1.0 + rs)
                continue

            avg_gain = (avg_gain * (period - 1) + gains[di]) / period
            avg_loss = (avg_loss * (period - 1) + losses[di]) / period
            if avg_loss == 0:
                rsi[si, di] = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi[si, di] = 100.0 - 100.0 / (1.0 + rs)
    return rsi


def compute_raw_factors(C, O, H, L, V, OI, NS, ND):
    t0 = time.time()
    print("[V40] Computing raw factors...", flush=True)

    ret_5d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            if not np.isnan(C[si, di]) and not np.isnan(C[si, di - 5]) and C[si, di - 5] > 0:
                ret_5d[si, di] = C[si, di] / C[si, di - 5] - 1.0

    ret_10d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(10, ND):
            if not np.isnan(C[si, di]) and not np.isnan(C[si, di - 10]) and C[si, di - 10] > 0:
                ret_10d[si, di] = C[si, di] / C[si, di - 10] - 1.0

    oi_5d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            if not np.isnan(OI[si, di]) and not np.isnan(OI[si, di - 5]) and OI[si, di - 5] > 0:
                oi_5d[si, di] = OI[si, di] / OI[si, di - 5] - 1.0

    vol_5d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            vals = V[si, di - 5:di]
            valid = vals[~np.isnan(vals)]
            if len(valid) >= 3:
                vol_5d[si, di] = np.mean(valid)

    daily_range = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(1, ND):
            if not np.isnan(H[si, di]) and not np.isnan(L[si, di]) and not np.isnan(C[si, di]):
                if C[si, di] > 0 and H[si, di] > L[si, di]:
                    daily_range[si, di] = (H[si, di] - L[si, di]) / C[si, di]

    rsi14 = np.full((NS, ND), np.nan)
    if HAS_TALIB:
        for si in range(NS):
            c = np.where(np.isnan(C[si]), 0, C[si]).astype(np.float64)
            nan_mask = np.isnan(C[si])
            try:
                r = talib.RSI(c, 14)
                rsi14[si] = np.where(nan_mask, np.nan, r)
            except Exception:
                pass
    needs_fallback = np.all(np.isnan(rsi14), axis=1)
    if needs_fallback.any():
        rsi_manual = compute_rsi_manual(C, NS, ND, 14)
        for si in range(NS):
            if needs_fallback[si]:
                rsi14[si] = rsi_manual[si]

    atrp = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(14, ND):
            atr_vals = []
            for j in range(di - 14, di):
                hh, ll, cc = H[si, j], L[si, j], C[si, j]
                if not any(np.isnan([hh, ll, cc])):
                    prev_c = C[si, j - 1] if j > 0 and not np.isnan(C[si, j - 1]) else cc
                    atr_vals.append(max(hh - ll, abs(hh - prev_c), abs(ll - prev_c)))
            if atr_vals and not np.isnan(C[si, di]) and C[si, di] > 0:
                atrp[si, di] = np.mean(atr_vals) / C[si, di]

    print(f"  Raw factors done: {time.time() - t0:.1f}s", flush=True)
    return {
        'ret_5d': ret_5d,
        'ret_10d': ret_10d,
        'oi_5d': oi_5d,
        'vol_5d': vol_5d,
        'daily_range': daily_range,
        'rsi14': rsi14,
        'atrp': atrp,
    }


def compute_cross_sectional_ranks(raw_factors, NS, ND, min_count=10):
    t0 = time.time()
    print("[V40] Computing cross-sectional ranks...", flush=True)

    factors_to_rank = {
        'rank_ret5d': raw_factors['ret_5d'],
        'rank_ret10d': raw_factors['ret_10d'],
        'rank_oi5d': raw_factors['oi_5d'],
        'rank_vol': raw_factors['vol_5d'],
        'rank_range': raw_factors['daily_range'],
        'rank_rsi': raw_factors['rsi14'],
        'rank_atrp': raw_factors['atrp'],
    }

    INVERT_FACTORS = {'rank_ret5d', 'rank_ret10d', 'rank_oi5d', 'rank_rsi'}

    ranks = {}
    for name, factor in factors_to_rank.items():
        rank_arr = np.full((NS, ND), np.nan)
        for di in range(ND):
            vals = factor[:, di]
            valid_count = np.sum(~np.isnan(vals))
            if valid_count < min_count:
                continue
            ranked = pd.Series(vals).rank(pct=True, na_option='keep').values
            if name in INVERT_FACTORS:
                ranked = 1.0 - ranked
            rank_arr[:, di] = ranked
        ranks[name] = rank_arr

    print(f"  CS ranks done: {time.time() - t0:.1f}s", flush=True)
    return ranks


def compute_ker(C, NS, ND):
    ker_10 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(10, ND):
            closes = C[si, di - 10:di + 1]
            valid = closes[~np.isnan(closes)]
            if len(valid) < 10 or valid[0] <= 0:
                continue
            net_change = abs(valid[-1] - valid[0])
            total_change = np.sum(np.abs(np.diff(valid)))
            if total_change > 1e-10:
                ker_10[si, di] = net_change / total_change

    ker_regime = np.zeros((NS, ND), dtype=int)
    for si in range(NS):
        for di in range(ND):
            if np.isnan(ker_10[si, di]):
                continue
            if ker_10[si, di] < 0.15:
                ker_regime[si, di] = 1
            elif ker_10[si, di] > 0.3:
                ker_regime[si, di] = -1
    return ker_regime


def build_composite_signal(ranks, weights, NS, ND, min_factors=4):
    t0 = time.time()
    print("[V40] Building composite signal...", flush=True)

    composite = np.full((NS, ND), np.nan)
    n_confirm = np.zeros((NS, ND), dtype=int)

    factor_names = list(weights.keys())
    weight_vals = np.array([weights[k] for k in factor_names])

    for di in range(ND):
        for si in range(NS):
            vals = []
            w_sum = 0.0
            confirm_count = 0
            for idx, name in enumerate(factor_names):
                rank_val = ranks[name][si, di]
                if np.isnan(rank_val):
                    continue
                vals.append(rank_val * weight_vals[idx])
                w_sum += weight_vals[idx]
                if rank_val > 0.5:
                    confirm_count += 1

            if w_sum > 0 and confirm_count >= min_factors:
                composite[si, di] = sum(vals) / w_sum
                n_confirm[si, di] = confirm_count

    print(f"  Composite done: {time.time() - t0:.1f}s", flush=True)
    return composite, n_confirm


# ============================================================
# MARKET BREADTH COMPUTATION (V40 new)
# ============================================================
def compute_market_breadth(C, composite, NS, ND):
    """Compute daily market breadth indicators.

    Returns:
        ad_ratio: array[ND] -- fraction of commodities with positive 5d return
        avg_rank: array[ND] -- mean composite rank across all commodities
        new_lows_count: array[ND] -- # of commodities at 20d low
    """
    t0 = time.time()
    print("[V40] Computing market breadth...", flush=True)

    ad_ratio = np.full(ND, np.nan)
    avg_rank = np.full(ND, np.nan)
    new_lows_count = np.full(ND, np.nan)

    for di in range(20, ND):
        # Advance/Decline: % with positive 5d return
        rets = []
        ranks = []
        lows = 0
        low_count_valid = 0
        for si in range(NS):
            c_now = C[si, di]
            c_5d = C[si, di - 5]
            if not np.isnan(c_now) and not np.isnan(c_5d) and c_5d > 0:
                rets.append(c_now / c_5d - 1.0)

            if not np.isnan(composite[si, di]):
                ranks.append(composite[si, di])

            # New 20d low check
            if not np.isnan(c_now):
                window = C[si, di - 20:di + 1]
                valid_w = window[~np.isnan(window)]
                if len(valid_w) >= 10:
                    low_count_valid += 1
                    if c_now <= np.min(valid_w):
                        lows += 1

        if len(rets) >= 10:
            ad_ratio[di] = sum(1 for r in rets if r > 0) / len(rets)
        if len(ranks) >= 10:
            avg_rank[di] = np.mean(ranks)
        if low_count_valid >= 10:
            new_lows_count[di] = lows

    print(f"  Breadth done: {time.time() - t0:.1f}s", flush=True)
    return ad_ratio, avg_rank, new_lows_count


def classify_breadth(ad_ratio, avg_rank, new_lows_count, ND,
                     ad_strong=0.30, ad_moderate=0.45,
                     avg_rank_thresh=0.55,
                     use_new_lows_filter=True):
    """Classify daily breadth regime.

    Returns:
        breadth_regime: array[ND] int -- one of BREADTH_* constants
        size_mult: array[ND] float -- position sizing multiplier
    """
    breadth_regime = np.zeros(ND, dtype=int)
    size_mult = np.ones(ND) * 0.0  # default: no trades

    for di in range(ND):
        ad = ad_ratio[di]
        ar = avg_rank[di]
        nl = new_lows_count[di]

        if np.isnan(ad) or np.isnan(ar):
            continue

        # New lows gate: if enabled, require some new lows for bearish
        nl_ok = True
        if use_new_lows_filter and not np.isnan(nl):
            nl_ok = nl >= 5  # at least 5 commodities at 20d low

        if ad < ad_strong and ar > avg_rank_thresh and nl_ok:
            breadth_regime[di] = BREADTH_STRONG_BEARISH
            size_mult[di] = 1.0
        elif ad < ad_moderate and ar > avg_rank_thresh * 0.9 and nl_ok:
            breadth_regime[di] = BREADTH_MODERATE_BEARISH
            size_mult[di] = 0.7
        elif ad < 0.55:
            breadth_regime[di] = BREADTH_NEUTRAL
            size_mult[di] = 0.0
        else:
            breadth_regime[di] = BREADTH_BULLISH
            size_mult[di] = 0.0

    return breadth_regime, size_mult


# ============================================================
# SIGNAL PIPELINE
# ============================================================
def compute_all_signals(C, O, H, L, V, OI, NS, ND,
                        weights=None,
                        ad_strong=0.30, ad_moderate=0.45,
                        avg_rank_thresh=0.55,
                        use_new_lows_filter=True):
    if weights is None:
        weights = DEFAULT_WEIGHTS

    raw = compute_raw_factors(C, O, H, L, V, OI, NS, ND)
    ranks = compute_cross_sectional_ranks(raw, NS, ND)
    ker_regime = compute_ker(C, NS, ND)
    composite, n_confirm = build_composite_signal(ranks, weights, NS, ND)

    ad_ratio, avg_rank, new_lows_count = compute_market_breadth(C, composite, NS, ND)
    breadth_regime, breadth_size_mult = classify_breadth(
        ad_ratio, avg_rank, new_lows_count, ND,
        ad_strong=ad_strong, ad_moderate=ad_moderate,
        avg_rank_thresh=avg_rank_thresh,
        use_new_lows_filter=use_new_lows_filter,
    )

    # Print breadth distribution
    counts = {
        'STRONG_BEAR': int(np.sum(breadth_regime == BREADTH_STRONG_BEARISH)),
        'MOD_BEAR': int(np.sum(breadth_regime == BREADTH_MODERATE_BEARISH)),
        'NEUTRAL': int(np.sum(breadth_regime == BREADTH_NEUTRAL)),
        'BULLISH': int(np.sum(breadth_regime == BREADTH_BULLISH)),
    }
    total_classified = sum(counts.values())
    if total_classified > 0:
        print(f"  Breadth dist: {counts}")

    return {
        'composite': composite,
        'n_confirm': n_confirm,
        'ker_regime': ker_regime,
        'ranks': ranks,
        'ad_ratio': ad_ratio,
        'avg_rank': avg_rank,
        'new_lows_count': new_lows_count,
        'breadth_regime': breadth_regime,
        'breadth_size_mult': breadth_size_mult,
    }


# ============================================================
# BACKTEST
# ============================================================
def backtest_v40(C, O, H, L, NS, ND, dates, syms, sigs,
                 top_n=1, min_rank=0.75, atr_stop=3.0,
                 min_confidence=3, use_ker_gate=True,
                 hold_days=5,
                 pyramid_ratio=0.5,
                 pyramid_day=1,
                 start_di=60, end_di=None):
    composite = sigs['composite']
    ker_regime = sigs['ker_regime']
    n_confirm = sigs['n_confirm']
    breadth_size_mult = sigs['breadth_size_mult']

    if end_di is None:
        end_di = ND - 1

    equity = CASH0
    peak = equity
    max_dd = 0.0
    positions = []
    trades = []

    for di in range(max(start_di, 1), end_di):
        d = dates[di]
        daily_pnl = 0
        new_positions = []

        pos_by_si = defaultdict(list)
        for si, edi, ep, sp, alloc, is_pyr in positions:
            pos_by_si[si].append((edi, ep, sp, alloc, is_pyr))

        for si, pos_list in pos_by_si.items():
            c = C[si, di]
            if np.isnan(c):
                for edi, ep, sp, alloc, is_pyr in pos_list:
                    new_positions.append((si, edi, ep, sp, alloc, is_pyr))
                continue

            earliest_edi = min(p[0] for p in pos_list)
            hold = di - earliest_edi

            stopped = any(c < sp for _, _, sp, _, _ in pos_list)

            if stopped:
                for edi, ep, sp, alloc, is_pyr in pos_list:
                    pnl = (c - ep) / ep - COMM
                    profit = equity * alloc * pnl
                    daily_pnl += profit
                    trades.append({
                        'pnl_abs': profit, 'pnl_pct': pnl * 100,
                        'days': di - edi + 1, 'di': di, 'year': d.year,
                        'sym': syms[si], 'reason': 'stop', 'pyr': is_pyr,
                    })
            elif hold >= hold_days:
                for edi, ep, sp, alloc, is_pyr in pos_list:
                    pnl = (c - ep) / ep - COMM
                    profit = equity * alloc * pnl
                    daily_pnl += profit
                    trades.append({
                        'pnl_abs': profit, 'pnl_pct': pnl * 100,
                        'days': di - edi + 1, 'di': di, 'year': d.year,
                        'sym': syms[si], 'reason': 'hold', 'pyr': is_pyr,
                    })
            else:
                for edi, ep, sp, alloc, is_pyr in pos_list:
                    new_positions.append((si, edi, ep, sp, alloc, is_pyr))

        # Pyramid check
        if pyramid_ratio > 0:
            held_with_pos = defaultdict(list)
            for si, edi, ep, sp, alloc, is_pyr in new_positions:
                held_with_pos[si].append((edi, ep, sp, alloc, is_pyr))

            additions = []
            for si, pos_list in held_with_pos.items():
                has_pyr = any(is_pyr for _, _, _, _, is_pyr in pos_list)
                if has_pyr:
                    continue
                earliest_edi = min(p[0] for p in pos_list)
                hold = di - earliest_edi
                if hold == pyramid_day and not np.isnan(C[si, di]):
                    avg_ep = np.mean([ep for _, ep, _, _, _ in pos_list])
                    if C[si, di] > avg_ep:
                        base_alloc = sum(a for _, _, _, a, _ in pos_list)
                        pyr_alloc = base_alloc * pyramid_ratio
                        c_now = C[si, di]
                        atr_v = []
                        for j in range(max(start_di, di - 14), di):
                            hh, ll, cc = H[si, j], L[si, j], C[si, j]
                            if not any(np.isnan([hh, ll, cc])):
                                atr_v.append(max(hh - ll, abs(hh - cc), abs(ll - cc)))
                        if atr_v:
                            atr = np.mean(atr_v)
                            additions.append(
                                (si, di, c_now, c_now - atr_stop * atr, pyr_alloc, True))
            new_positions.extend(additions)

        positions = new_positions
        equity += daily_pnl
        if equity > peak:
            peak = equity
        if peak > 0:
            dd = (peak - equity) / peak * 100
            if dd > max_dd:
                max_dd = dd
        if equity <= 0:
            break

        held = {p[0] for p in positions}
        if len(positions) >= top_n:
            continue

        # BREADTH FILTER: check market-wide conditions at close[di]
        breadth_mult = breadth_size_mult[di]
        if breadth_mult <= 0:
            # Market not oversold enough for MR -- skip entry
            continue

        # Entry signal at close[di], enter at open[di+1]
        candidates = []
        for si in range(NS):
            if si in held:
                continue
            if np.isnan(composite[si, di]):
                continue
            if composite[si, di] < min_rank:
                continue
            if n_confirm[si, di] < min_confidence:
                continue
            if use_ker_gate and ker_regime[si, di] < 0:
                continue
            if di + 1 >= ND or np.isnan(O[si, di + 1]):
                continue
            # Apply breadth sizing multiplier to allocation
            alloc = (1.0 / max(top_n, 1)) * breadth_mult
            candidates.append((composite[si, di], si, alloc))

        candidates.sort(key=lambda x: -x[0])
        for rank_val, si, alloc in candidates[:top_n]:
            if len(positions) >= top_n or si in held:
                break
            ep = O[si, di + 1]
            if np.isnan(ep) or ep <= 0:
                continue
            atr_v = []
            for j in range(max(start_di, di - 14), di):
                hh, ll, cc = H[si, j], L[si, j], C[si, j]
                if not any(np.isnan([hh, ll, cc])):
                    atr_v.append(max(hh - ll, abs(hh - cc), abs(ll - cc)))
            if not atr_v:
                continue
            atr = np.mean(atr_v)
            positions.append((si, di + 1, ep, ep - atr_stop * atr, alloc, False))
            held.add(si)

    # Close remaining positions
    for si, edi, ep, sp, alloc, is_pyr in positions:
        c = C[si, ND - 1]
        if not np.isnan(c) and c > 0:
            pnl = (c - ep) / ep - COMM
            equity += equity * alloc * pnl

    return trades, equity, max_dd


# ============================================================
# ANALYSIS
# ============================================================
def analyze(trades, equity, max_dd, label=""):
    if not trades:
        print(f"  {label}: no trades")
        return None
    nw = sum(1 for t in trades if t['pnl_pct'] > 0)
    wr = nw / len(trades) * 100
    n_days = max(1, trades[-1]['di'] - trades[0]['di'])
    ann = ((equity / CASH0) ** (1 / max(1.0, n_days / 252)) - 1) * 100
    ap = [t['pnl_abs'] for t in sorted(trades, key=lambda x: x['di'])]
    rets = np.array(ap) / CASH0
    sh = np.mean(rets) / np.std(rets) * np.sqrt(252) if np.std(rets) > 0 else 0

    n_pyr = sum(1 for t in trades if t.get('pyr'))
    n_base = len(trades) - n_pyr
    n_stop = sum(1 for t in trades if t['reason'] == 'stop')
    n_hold = sum(1 for t in trades if t['reason'] == 'hold')

    print(f"  {label}: {len(trades)}t (base:{n_base} pyr:{n_pyr} stop:{n_stop} hold:{n_hold}) "
          f"WR={wr:.1f}% ann={ann:+.1f}% DD={max_dd:.1f}% Sh={sh:.2f} eq={equity:,.0f}")

    yr = {}
    for t in trades:
        y = t['year']
        if y not in yr:
            yr[y] = {'n': 0, 'w': 0, 'pnl': []}
        yr[y]['n'] += 1
        if t['pnl_pct'] > 0:
            yr[y]['w'] += 1
        yr[y]['pnl'].append(t['pnl_pct'])
    for y in sorted(yr.keys()):
        ys = yr[y]
        cum = np.prod([1 + p / 100 for p in ys['pnl']]) - 1
        print(f"    {y}: {ys['n']}t WR={ys['w']/ys['n']*100:.1f}% cum={cum:+.1%}")

    return {'n': len(trades), 'wr': wr, 'dd': max_dd, 'ann': ann, 'sh': sh, 'eq': equity}


# ============================================================
# WALK-FORWARD
# ============================================================
def walk_forward(C, O, H, L, V, OI, NS, ND, dates, syms, sigs,
                 pyramid_ratio=0.5, pyramid_day=1,
                 top_n=1, min_confidence=3, hold_days=5, atr_stop=3.0,
                 min_rank=0.75):
    print(f"\n{'=' * 70}")
    print(f"  WALK-FORWARD V40 (pyr={pyramid_ratio}, day={pyramid_day})")
    print(f"{'=' * 70}")

    years = sorted(set(d.year for d in dates))
    all_trades = []

    for test_year in range(2019, years[-1] + 1):
        test_start = None
        test_end_idx = None
        for i, d in enumerate(dates):
            if d.year == test_year and test_start is None:
                test_start = i
            if d.year == test_year:
                test_end_idx = i
        if test_start is None:
            continue

        trades, _, _ = backtest_v40(
            C, O, H, L, NS, ND, dates, syms, sigs,
            top_n=top_n, hold_days=hold_days, atr_stop=atr_stop,
            min_rank=min_rank,
            min_confidence=min_confidence, use_ker_gate=True,
            pyramid_ratio=pyramid_ratio, pyramid_day=pyramid_day,
            start_di=test_start, end_di=test_end_idx + 1)

        test_trades = [t for t in trades if dates[t['di']].year == test_year]
        all_trades.extend(test_trades)

        if test_trades:
            n = len(test_trades)
            nw = sum(1 for t in test_trades if t['pnl_pct'] > 0)
            wr_val = nw / n * 100
            avg = np.mean([t['pnl_pct'] for t in test_trades])
            print(f"  {test_year}: {n}t WR={wr_val:.1f}% avg={avg:+.2f}%", flush=True)
        else:
            print(f"  {test_year}: no trades", flush=True)

    if all_trades:
        nw = sum(1 for t in all_trades if t['pnl_pct'] > 0)
        wr_val = nw / len(all_trades) * 100
        avg = np.mean([t['pnl_pct'] for t in all_trades])
        cum = np.prod([1 + t['pnl_pct'] / 100 for t in all_trades]) - 1
        n_pyr = sum(1 for t in all_trades if t.get('pyr'))
        print(f"\n  WF TOTAL: {len(all_trades)}t (pyr:{n_pyr}) WR={wr_val:.1f}% "
              f"avg={avg:+.2f}% cum={cum:+.1%}")
        return all_trades
    return []


# ============================================================
# PARAMETER SWEEP
# ============================================================
def compute_metrics(trades, equity):
    """Compute standard metrics from trades list."""
    if len(trades) < 10:
        return None
    nw = sum(1 for t in trades if t['pnl_pct'] > 0)
    wr = nw / len(trades) * 100
    n_days = max(1, trades[-1]['di'] - trades[0]['di'])
    ann = ((equity / CASH0) ** (1 / max(1.0, n_days / 252)) - 1) * 100
    ap = [t['pnl_abs'] for t in sorted(trades, key=lambda x: x['di'])]
    rets_arr = np.array(ap) / CASH0
    sh_val = (np.mean(rets_arr) / np.std(rets_arr) * np.sqrt(252)
              if np.std(rets_arr) > 0 else 0)
    return {
        'n': len(trades), 'wr': wr, 'ann': ann, 'sharpe': sh_val, 'eq': equity,
    }


def main():
    t0 = time.time()
    print("=" * 70)
    print("  V40: BREADTH-FILTERED CROSS-SECTIONAL RANK MEAN REVERSION")
    print("  V18 rank + market breadth confirmation")
    print("=" * 70)

    C, O, H, L, V, OI, NS, ND, dates, syms = load_all_data(start='2016-01-01')
    print(f"  {NS} sym, {ND} days, "
          f"{dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")

    # Find 2019 start index
    bt_2019 = None
    for i, d in enumerate(dates):
        if d >= pd.Timestamp('2019-01-01'):
            bt_2019 = i
            break

    # === 1. Default config walk-forward ===
    print("\n" + "=" * 70)
    print("  BASELINE: DEFAULT BREADTH PARAMS")
    print("=" * 70)

    sigs = compute_all_signals(C, O, H, L, V, OI, NS, ND)

    for ratio in [0.0, 0.5]:
        walk_forward(C, O, H, L, V, OI, NS, ND, dates, syms, sigs,
                     pyramid_ratio=ratio, pyramid_day=1,
                     top_n=1, min_confidence=3)

    # === 2. Full 10-year with default breadth ===
    print("\n" + "=" * 70)
    print("  FULL 2016-2026 (10 years) -- BREADTH FILTERED")
    print("=" * 70)

    for ratio in [0.0, 0.5]:
        trades, eq, dd = backtest_v40(
            C, O, H, L, NS, ND, dates, syms, sigs,
            top_n=1, hold_days=5, atr_stop=3.0,
            min_rank=0.75, min_confidence=3, use_ker_gate=True,
            pyramid_ratio=ratio, pyramid_day=1,
            start_di=60)
        label = f"V40 pyr={ratio:.1f}"
        print(f"\n  {label}")
        analyze(trades, eq, dd, label)

    # === 3. BREADTH PARAMETER SWEEP ===
    print("\n" + "=" * 70)
    print("  BREADTH PARAMETER SWEEP (2019-2026)")
    print("=" * 70)

    breadth_results = []

    for ad_strong in [0.25, 0.30, 0.35, 0.45]:
        for avg_rank_thresh in [0.50, 0.55, 0.60]:
            for use_nl in [True, False]:
                b_sigs = compute_all_signals(
                    C, O, H, L, V, OI, NS, ND,
                    ad_strong=ad_strong, ad_moderate=0.45,
                    avg_rank_thresh=avg_rank_thresh,
                    use_new_lows_filter=use_nl,
                )
                for tn in [1, 2]:
                    for mr in [0.75, 0.80]:
                        for atr in [2.5, 3.0]:
                            for pyr in [0.0, 0.5]:
                                trades, eq, dd = backtest_v40(
                                    C, O, H, L, NS, ND, dates, syms, b_sigs,
                                    top_n=tn, hold_days=5, atr_stop=atr,
                                    min_rank=mr, min_confidence=3,
                                    use_ker_gate=True,
                                    pyramid_ratio=pyr, pyramid_day=1,
                                    start_di=bt_2019)
                                m = compute_metrics(trades, eq)
                                if m is None:
                                    continue
                                breadth_results.append({
                                    'ad_s': ad_strong,
                                    'ar_t': avg_rank_thresh,
                                    'nl': use_nl,
                                    'tn': tn, 'mr': mr,
                                    'atr': atr, 'pyr': pyr,
                                    **m, 'dd': dd,
                                })

    breadth_results.sort(key=lambda x: -x['sharpe'])
    print(f"\n{'ADs':>4} {'ARt':>4} {'NL':>4} {'TN':>3} {'MR':>4} "
          f"{'ATR':>4} {'Pyr':>4} "
          f"{'N':>5} {'WR':>5} {'Ann':>8} {'DD':>6} {'Sh':>5}")
    print("-" * 80)
    for r in breadth_results[:30]:
        print(f"{r['ad_s']:>4.2f} {r['ar_t']:>4.2f} {str(r['nl']):>4} "
              f"{r['tn']:>3} {r['mr']:>4.2f} "
              f"{r['atr']:>4.1f} {r['pyr']:>4.1f} "
              f"{r['n']:>5} {r['wr']:>5.1f} {r['ann']:>+8.1f} "
              f"{r['dd']:>6.1f} {r['sharpe']:>5.2f}")

    # === 4. Best configs: full 10-year ===
    if breadth_results:
        print("\n" + "=" * 70)
        print("  TOP 5 CONFIGS -- FULL 10-YEAR")
        print("=" * 70)

        seen = set()
        unique_top = []
        for r in breadth_results:
            key = (r['ad_s'], r['ar_t'], r['nl'], r['tn'], r['mr'], r['atr'], r['pyr'])
            if key not in seen:
                seen.add(key)
                unique_top.append(r)
            if len(unique_top) >= 5:
                break

        for r in unique_top:
            b_sigs = compute_all_signals(
                C, O, H, L, V, OI, NS, ND,
                ad_strong=r['ad_s'], ad_moderate=0.45,
                avg_rank_thresh=r['ar_t'],
                use_new_lows_filter=r['nl'],
            )
            trades, eq, dd = backtest_v40(
                C, O, H, L, NS, ND, dates, syms, b_sigs,
                top_n=r['tn'], hold_days=5, atr_stop=r['atr'],
                min_rank=r['mr'], min_confidence=3,
                use_ker_gate=True,
                pyramid_ratio=r['pyr'], pyramid_day=1,
                start_di=60)
            label = (f"ad={r['ad_s']:.2f} ar={r['ar_t']:.2f} nl={r['nl']} "
                     f"tn={r['tn']} mr={r['mr']:.2f} atr={r['atr']:.1f} pyr={r['pyr']:.1f}")
            print(f"\n  FULL {label}")
            analyze(trades, eq, dd, label)

        # === 5. Walk-forward for best ===
        best = unique_top[0]
        print("\n" + "=" * 70)
        print(f"  BEST WF: ad={best['ad_s']:.2f} ar={best['ar_t']:.2f} nl={best['nl']} "
              f"tn={best['tn']} mr={best['mr']:.2f} atr={best['atr']:.1f} pyr={best['pyr']:.1f}")
        print("=" * 70)

        best_sigs = compute_all_signals(
            C, O, H, L, V, OI, NS, ND,
            ad_strong=best['ad_s'], ad_moderate=0.45,
            avg_rank_thresh=best['ar_t'],
            use_new_lows_filter=best['nl'],
        )
        walk_forward(C, O, H, L, V, OI, NS, ND, dates, syms, best_sigs,
                     pyramid_ratio=best['pyr'], pyramid_day=1,
                     top_n=best['tn'], min_confidence=3,
                     hold_days=5, min_rank=best['mr'])

    # === 6. Comparison: V18 baseline (no breadth) ===
    print("\n" + "=" * 70)
    print("  COMPARISON: V18 (no breadth) vs V40 (with breadth)")
    print("=" * 70)

    # V18 equivalent: ad_moderate=1.0 -> all days pass breadth filter
    sigs_v18 = compute_all_signals(
        C, O, H, L, V, OI, NS, ND,
        ad_strong=0.0, ad_moderate=1.0,
        avg_rank_thresh=0.0,
        use_new_lows_filter=False,
    )
    for pyr in [0.0, 0.5]:
        trades_v18, eq_v18, dd_v18 = backtest_v40(
            C, O, H, L, NS, ND, dates, syms, sigs_v18,
            top_n=1, hold_days=5, atr_stop=3.0,
            min_rank=0.75, min_confidence=3, use_ker_gate=True,
            pyramid_ratio=pyr, pyramid_day=1,
            start_di=bt_2019)
        print(f"\n  V18-equiv pyr={pyr:.1f}")
        analyze(trades_v18, eq_v18, dd_v18, f"V18 pyr={pyr:.1f}")

    print(f"\n[V40] Done. {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
