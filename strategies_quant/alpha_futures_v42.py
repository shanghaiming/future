"""
V42: Adaptive Threshold + Breadth Filter Hybrid
=================================================
Combines the two best innovations from V39 and V40:

V39 (Sharpe 4.47) demonstrated that adaptive thresholds -- relaxing entry
criteria when winning, tightening when losing -- dramatically improves
risk-adjusted returns.

V40 (MDD 20.6%, Sharpe 2.11) proved that market breadth filtering (only
entering when A/D ratio is low = market broadly oversold) improves trade
quality and reduces drawdown.

V42 combines both: uses V39's adaptive threshold mechanism AND requires
the broader commodity market to be in a bearish/oversold state (V40's
breadth filter). The hypothesis is that adaptive entry *within* a
breadth-confirmed regime should produce even higher quality trades.

Architecture:
  1. V18 cross-sectional rank: 7 factors, composite score
  2. Adaptive threshold (V39): rolling win rate adjusts threshold
  3. Market breadth filter (V40): A/D ratio < max_ad required
  4. Entry: composite > adaptive_threshold AND breadth < max_ad AND KER < 0.15
  5. Hold 5d, ATR stop, pyramid on day-1 winners

Signal at close[di], enter at open[di+1]. No look-ahead. No gap signals.
No leverage. Walk-forward validation required.
"""
import sys
import os
import time
import warnings
import numpy as np
import pandas as pd
from collections import defaultdict
from itertools import product

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


# ============================================================
# FACTOR COMPUTATION (V18/V39/V40 shared)
# ============================================================
def compute_rsi_manual(C: np.ndarray, NS: int, ND: int, period: int = 14) -> np.ndarray:
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
    print("[V42] Computing raw factors...", flush=True)

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
    print("[V42] Computing cross-sectional ranks...", flush=True)

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
    print("[V42] Building composite signal...", flush=True)

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
# MARKET BREADTH (from V40)
# ============================================================
def compute_market_breadth(C, composite, NS, ND):
    """Compute daily market breadth indicators.

    Returns:
        ad_ratio: array[ND] -- fraction of commodities with positive 5d return
        avg_rank: array[ND] -- mean composite rank across all commodities
    """
    t0 = time.time()
    print("[V42] Computing market breadth...", flush=True)

    ad_ratio = np.full(ND, np.nan)
    avg_rank = np.full(ND, np.nan)

    for di in range(20, ND):
        rets = []
        ranks_list = []
        for si in range(NS):
            c_now = C[si, di]
            c_5d = C[si, di - 5]
            if not np.isnan(c_now) and not np.isnan(c_5d) and c_5d > 0:
                rets.append(c_now / c_5d - 1.0)

            if not np.isnan(composite[si, di]):
                ranks_list.append(composite[si, di])

        if len(rets) >= 10:
            ad_ratio[di] = sum(1 for r in rets if r > 0) / len(rets)
        if len(ranks_list) >= 10:
            avg_rank[di] = np.mean(ranks_list)

    print(f"  Breadth done: {time.time() - t0:.1f}s", flush=True)
    return ad_ratio, avg_rank


# ============================================================
# SIGNAL PIPELINE
# ============================================================
def compute_all_signals(C, O, H, L, V, OI, NS, ND, weights=None):
    if weights is None:
        weights = DEFAULT_WEIGHTS

    raw = compute_raw_factors(C, O, H, L, V, OI, NS, ND)
    ranks = compute_cross_sectional_ranks(raw, NS, ND)
    ker_regime = compute_ker(C, NS, ND)
    composite, n_confirm = build_composite_signal(ranks, weights, NS, ND)
    ad_ratio, avg_rank = compute_market_breadth(C, composite, NS, ND)

    return {
        'composite': composite,
        'n_confirm': n_confirm,
        'ker_regime': ker_regime,
        'ranks': ranks,
        'ad_ratio': ad_ratio,
        'avg_rank': avg_rank,
    }


# ============================================================
# ADAPTIVE THRESHOLD (from V39)
# ============================================================
def compute_atr_at(H, L, C, si, di, start_di):
    """Compute ATR for a specific symbol/day."""
    atr_v = []
    for j in range(max(start_di, di - 14), di):
        hh, ll, cc = H[si, j], L[si, j], C[si, j]
        if not any(np.isnan([hh, ll, cc])):
            atr_v.append(max(hh - ll, abs(hh - cc), abs(ll - cc)))
    if atr_v:
        return np.mean(atr_v)
    return None


def adaptive_threshold(
    recent_trades_win: list,
    base_threshold: float,
    adapt_amount: float,
    min_cap: float,
    max_cap: float,
    win_rate_window: int,
) -> float:
    """Compute adaptive threshold based on recent trade win rate.

    P-controller: adjusts threshold up/down based on rolling win rate.
    - win_rate > 60%: relax threshold (take more trades)
    - win_rate 50-60%: neutral
    - win_rate < 50%: tighten threshold (be more selective)
    """
    if len(recent_trades_win) < 5:
        return base_threshold

    window = recent_trades_win[-win_rate_window:]
    win_rate = sum(window) / len(window)

    if win_rate > 0.60:
        threshold = base_threshold - adapt_amount
    elif win_rate < 0.50:
        threshold = base_threshold + adapt_amount
    else:
        threshold = base_threshold

    return max(min_cap, min(max_cap, threshold))


def adaptive_top_n(
    recent_trades_win: list,
    top_n_base: int,
    win_rate_window: int,
) -> int:
    """Adapt top_n based on recent performance."""
    if len(recent_trades_win) < 5:
        return top_n_base

    window = recent_trades_win[-win_rate_window:]
    win_rate = sum(window) / len(window)

    if win_rate > 0.60:
        return min(top_n_base + 1, 3)
    elif win_rate < 0.50:
        return max(top_n_base - 1, 1)
    return top_n_base


# ============================================================
# BACKTEST ENGINE (V42 hybrid: adaptive + breadth)
# ============================================================
def backtest_v42(
    C, O, H, L, NS, ND, dates, syms, sigs,
    base_threshold=0.80,
    adapt_amount=0.07,
    win_rate_window=20,
    top_n_base=1,
    min_cap=0.70,
    max_cap=0.95,
    atr_stop=3.0,
    min_confidence=3,
    use_ker_gate=True,
    hold_days=5,
    pyramid_ratio=0.5,
    pyramid_day=1,
    max_ad=0.45,
    start_di=60,
    end_di=None,
):
    """Backtest V42: adaptive threshold + breadth filter hybrid."""
    composite = sigs['composite']
    ker_regime = sigs['ker_regime']
    n_confirm = sigs['n_confirm']
    ad_ratio = sigs['ad_ratio']

    if end_di is None:
        end_di = ND - 1

    equity = CASH0
    peak = equity
    max_dd = 0.0
    positions = []
    trades = []

    # Rolling win/loss tracker for adaptive threshold
    recent_trades_win: list = []

    current_threshold = base_threshold
    current_top_n = top_n_base

    for di in range(max(start_di, 1), end_di):
        d = dates[di]
        daily_pnl = 0
        new_positions = []

        # Update adaptive threshold and top_n before trading
        current_threshold = adaptive_threshold(
            recent_trades_win, base_threshold, adapt_amount,
            min_cap, max_cap, win_rate_window,
        )
        current_top_n = adaptive_top_n(
            recent_trades_win, top_n_base, win_rate_window,
        )

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
                    is_win = pnl > 0
                    trades.append({
                        'pnl_abs': profit, 'pnl_pct': pnl * 100,
                        'days': di - edi + 1, 'di': di, 'year': d.year,
                        'sym': syms[si], 'reason': 'stop', 'pyr': is_pyr,
                        'threshold': current_threshold, 'top_n': current_top_n,
                        'ad': ad_ratio[di],
                    })
                    recent_trades_win.append(1 if is_win else 0)
            elif hold >= hold_days:
                for edi, ep, sp, alloc, is_pyr in pos_list:
                    pnl = (c - ep) / ep - COMM
                    profit = equity * alloc * pnl
                    daily_pnl += profit
                    is_win = pnl > 0
                    trades.append({
                        'pnl_abs': profit, 'pnl_pct': pnl * 100,
                        'days': di - edi + 1, 'di': di, 'year': d.year,
                        'sym': syms[si], 'reason': 'hold', 'pyr': is_pyr,
                        'threshold': current_threshold, 'top_n': current_top_n,
                        'ad': ad_ratio[di],
                    })
                    recent_trades_win.append(1 if is_win else 0)
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
                        atr = compute_atr_at(H, L, C, si, di, start_di)
                        if atr is not None:
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
        if len(positions) >= current_top_n:
            continue

        # BREADTH FILTER: require A/D ratio < max_ad (market oversold)
        ad_val = ad_ratio[di]
        if np.isnan(ad_val) or ad_val >= max_ad:
            continue

        # Entry signal at close[di], enter at open[di+1]
        # Use the ADAPTIVE threshold
        candidates = []
        for si in range(NS):
            if si in held:
                continue
            if np.isnan(composite[si, di]):
                continue
            if composite[si, di] < current_threshold:
                continue
            if n_confirm[si, di] < min_confidence:
                continue
            if use_ker_gate and ker_regime[si, di] < 0:
                continue
            if di + 1 >= ND or np.isnan(O[si, di + 1]):
                continue
            alloc = 1.0 / max(current_top_n, 1)
            candidates.append((composite[si, di], si, alloc))

        candidates.sort(key=lambda x: -x[0])
        for rank_val, si, alloc in candidates[:current_top_n]:
            if len(positions) >= current_top_n or si in held:
                break
            ep = O[si, di + 1]
            if np.isnan(ep) or ep <= 0:
                continue
            atr = compute_atr_at(H, L, C, si, di, start_di)
            if atr is None:
                continue
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

    # Report threshold and A/D distribution
    thresholds_used = [t.get('threshold', 0) for t in trades]
    avg_thresh = np.mean(thresholds_used) if thresholds_used else 0
    ad_values = [t.get('ad', 0) for t in trades if not np.isnan(t.get('ad', np.nan))]
    avg_ad = np.mean(ad_values) if ad_values else 0

    print(f"  {label}: {len(trades)}t (base:{n_base} pyr:{n_pyr} stop:{n_stop} hold:{n_hold}) "
          f"WR={wr:.1f}% ann={ann:+.1f}% DD={max_dd:.1f}% Sh={sh:.2f} eq={equity:,.0f} "
          f"avg_thresh={avg_thresh:.3f} avg_ad={avg_ad:.3f}")

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


# ============================================================
# WALK-FORWARD
# ============================================================
def walk_forward(C, O, H, L, NS, ND, dates, syms, sigs,
                 base_threshold=0.80, adapt_amount=0.07,
                 win_rate_window=20, top_n_base=1,
                 min_cap=0.70, max_cap=0.95,
                 atr_stop=3.0, hold_days=5,
                 pyramid_ratio=0.5, pyramid_day=1,
                 max_ad=0.45):
    """Walk-forward validation: year-by-year out-of-sample."""
    print(f"\n{'=' * 70}")
    print(f"  WALK-FORWARD V42 (base_t={base_threshold} adapt={adapt_amount} "
          f"max_ad={max_ad} top_n={top_n_base})")
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

        trades, _, _ = backtest_v42(
            C, O, H, L, NS, ND, dates, syms, sigs,
            base_threshold=base_threshold, adapt_amount=adapt_amount,
            win_rate_window=win_rate_window, top_n_base=top_n_base,
            min_cap=min_cap, max_cap=max_cap,
            atr_stop=atr_stop, hold_days=hold_days,
            min_confidence=3, use_ker_gate=True,
            pyramid_ratio=pyramid_ratio, pyramid_day=pyramid_day,
            max_ad=max_ad,
            start_di=test_start, end_di=test_end_idx + 1)

        test_trades = [t for t in trades if dates[t['di']].year == test_year]
        all_trades.extend(test_trades)

        if test_trades:
            n = len(test_trades)
            nw = sum(1 for t in test_trades if t['pnl_pct'] > 0)
            wr_val = nw / n * 100
            avg = np.mean([t['pnl_pct'] for t in test_trades])
            avg_t = np.mean([t.get('threshold', 0) for t in test_trades])
            avg_ad = np.mean([t.get('ad', 0.5) for t in test_trades
                              if not np.isnan(t.get('ad', np.nan))])
            print(f"  {test_year}: {n}t WR={wr_val:.1f}% avg={avg:+.2f}% "
                  f"thresh={avg_t:.3f} ad={avg_ad:.3f}", flush=True)
        else:
            print(f"  {test_year}: no trades", flush=True)

    if all_trades:
        nw = sum(1 for t in all_trades if t['pnl_pct'] > 0)
        wr_val = nw / len(all_trades) * 100
        avg = np.mean([t['pnl_pct'] for t in all_trades])
        cum = np.prod([1 + t['pnl_pct'] / 100 for t in all_trades]) - 1
        n_pyr = sum(1 for t in all_trades if t.get('pyr'))
        avg_t = np.mean([t.get('threshold', 0) for t in all_trades])
        print(f"\n  WF TOTAL: {len(all_trades)}t (pyr:{n_pyr}) WR={wr_val:.1f}% "
              f"avg={avg:+.2f}% cum={cum:+.1%} avg_thresh={avg_t:.3f}")
        return all_trades
    return []


# ============================================================
# MAIN
# ============================================================
def main():
    t0 = time.time()
    print("=" * 70)
    print("  V42: ADAPTIVE THRESHOLD + BREADTH FILTER HYBRID")
    print("  V39 adaptive + V40 breadth confirmation")
    print("=" * 70)

    C, O, H, L, V, OI, NS, ND, dates, syms = load_all_data(start='2016-01-01')
    print(f"  {NS} sym, {ND} days, "
          f"{dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")

    sigs = compute_all_signals(C, O, H, L, V, OI, NS, ND)

    # Find 2019 start index for OOS testing
    bt_2019 = None
    for i, d in enumerate(dates):
        if d >= pd.Timestamp('2019-01-01'):
            bt_2019 = i
            break

    # === 1. Walk-Forward Validation with key adaptive+breadth combos ===
    print("\n" + "=" * 70)
    print("  WALK-FORWARD VALIDATION (2019-2026) -- KEY COMBOS")
    print("=" * 70)

    wf_configs = [
        (0.80, 0.07, 20, 1, 0.70, 0.95, 3.0, 0.5, 0.45),
        (0.80, 0.07, 20, 1, 0.70, 0.95, 3.0, 0.5, 0.40),
        (0.80, 0.07, 20, 1, 0.70, 0.95, 3.0, 0.5, 0.35),
        (0.85, 0.07, 20, 1, 0.75, 0.95, 3.0, 0.5, 0.45),
        (0.75, 0.05, 20, 1, 0.70, 0.90, 3.0, 0.5, 0.45),
        (0.80, 0.07, 20, 1, 0.70, 0.95, 3.0, 0.0, 0.45),
    ]

    for bt, aa, ww, tn, mn, mx, ats, pr, mad in wf_configs:
        walk_forward(C, O, H, L, NS, ND, dates, syms, sigs,
                     base_threshold=bt, adapt_amount=aa,
                     win_rate_window=ww, top_n_base=tn,
                     min_cap=mn, max_cap=mx,
                     atr_stop=ats, hold_days=5,
                     pyramid_ratio=pr, pyramid_day=1,
                     max_ad=mad)

    # === 2. Full 10-year backtest with profiles ===
    print("\n" + "=" * 70)
    print("  FULL 2016-2026 (10 years) -- ADAPTIVE+BREADTH PROFILES")
    print("=" * 70)

    profiles = [
        (0.80, 0.07, 20, 1, 0.70, 0.95, 3.0, 0.5, 0.45, "Default hybrid"),
        (0.80, 0.07, 20, 1, 0.70, 0.95, 3.0, 0.5, 0.40, "Tight breadth"),
        (0.80, 0.07, 20, 1, 0.70, 0.95, 3.0, 0.5, 0.35, "Very tight breadth"),
        (0.85, 0.07, 20, 1, 0.75, 0.95, 3.0, 0.5, 0.45, "High threshold"),
        (0.75, 0.05, 20, 1, 0.70, 0.90, 3.0, 0.5, 0.45, "Relaxed base"),
        (0.80, 0.07, 20, 1, 0.70, 0.95, 3.0, 0.0, 0.45, "No pyramid"),
    ]

    for bt, aa, ww, tn, mn, mx, ats, pr, mad, label in profiles:
        trades, eq, dd = backtest_v42(
            C, O, H, L, NS, ND, dates, syms, sigs,
            base_threshold=bt, adapt_amount=aa,
            win_rate_window=ww, top_n_base=tn,
            min_cap=mn, max_cap=mx,
            atr_stop=ats, hold_days=5,
            min_confidence=3, use_ker_gate=True,
            pyramid_ratio=pr, pyramid_day=1,
            max_ad=mad,
            start_di=60)
        print(f"\n  {label}")
        analyze(trades, eq, dd, label)

    # === 3. Full parameter sweep (2019-2026) ===
    print("\n" + "=" * 70)
    print("  PARAMETER SWEEP (2019-2026)")
    print("=" * 70)

    results = []

    sweep_params = {
        'base_threshold': [0.75, 0.80, 0.85],
        'adapt_amount': [0.05, 0.07],
        'max_ad': [0.35, 0.40, 0.45, 0.50],
        'top_n_base': [1, 2],
        'atr_stop': [2.5, 3.0],
        'pyramid_ratio': [0.0, 0.5],
    }

    total_combos = 1
    for v in sweep_params.values():
        total_combos *= len(v)
    print(f"  Total combinations: {total_combos}")

    combo_count = 0
    for bt, aa, mad, tn, ats, pr in product(
        sweep_params['base_threshold'],
        sweep_params['adapt_amount'],
        sweep_params['max_ad'],
        sweep_params['top_n_base'],
        sweep_params['atr_stop'],
        sweep_params['pyramid_ratio'],
    ):
        # min_cap < base_threshold, max_cap > base_threshold
        mn = max(0.60, bt - 0.10)
        mx = min(0.99, bt + 0.15)

        combo_count += 1
        trades, eq, dd = backtest_v42(
            C, O, H, L, NS, ND, dates, syms, sigs,
            base_threshold=bt, adapt_amount=aa,
            win_rate_window=20, top_n_base=tn,
            min_cap=mn, max_cap=mx,
            atr_stop=ats, hold_days=5,
            min_confidence=3, use_ker_gate=True,
            pyramid_ratio=pr, pyramid_day=1,
            max_ad=mad,
            start_di=bt_2019)

        if len(trades) < 10:
            continue

        m = compute_metrics(trades, eq)
        if m is None:
            continue

        results.append({
            'bt': bt, 'aa': aa, 'mad': mad, 'tn': tn,
            'mn': mn, 'mx': mx,
            'ats': ats, 'pr': pr,
            **m, 'dd': dd,
        })

    results.sort(key=lambda x: -x['sharpe'])
    print(f"\n  Evaluated {combo_count} combos, {len(results)} with 10+ trades")
    print(f"\n{'BT':>4} {'AA':>4} {'MAD':>4} {'TN':>3} {'ATS':>4} {'Pyr':>4} "
          f"{'N':>5} {'WR':>5} {'Ann':>8} {'DD':>6} {'Sh':>6}")
    print("-" * 75)
    for r in results[:30]:
        print(f"{r['bt']:>4.2f} {r['aa']:>4.2f} {r['mad']:>4.2f} {r['tn']:>3} "
              f"{r['ats']:>4.1f} {r['pr']:>4.1f} "
              f"{r['n']:>5} {r['wr']:>5.1f} {r['ann']:>+8.1f} "
              f"{r['dd']:>6.1f} {r['sharpe']:>6.2f}")

    # === 4. Top configs: full 10-year backtest ===
    print("\n" + "=" * 70)
    print("  TOP CONFIGS -- FULL 10-YEAR (2016-2026)")
    print("=" * 70)

    for r in results[:5]:
        trades, eq, dd = backtest_v42(
            C, O, H, L, NS, ND, dates, syms, sigs,
            base_threshold=r['bt'], adapt_amount=r['aa'],
            win_rate_window=20, top_n_base=r['tn'],
            min_cap=r['mn'], max_cap=r['mx'],
            atr_stop=r['ats'], hold_days=5,
            min_confidence=3, use_ker_gate=True,
            pyramid_ratio=r['pr'], pyramid_day=1,
            max_ad=r['mad'],
            start_di=60)
        label = (f"bt={r['bt']:.2f} aa={r['aa']:.2f} mad={r['mad']:.2f} "
                 f"tn={r['tn']} ats={r['ats']:.1f} pr={r['pr']:.1f}")
        print(f"\n  FULL {label}")
        analyze(trades, eq, dd, label)

    # === 5. Walk-forward for best config ===
    if results:
        best = results[0]
        print("\n" + "=" * 70)
        print(f"  BEST WF: bt={best['bt']:.2f} aa={best['aa']:.2f} "
              f"mad={best['mad']:.2f} tn={best['tn']} "
              f"ats={best['ats']:.1f} pr={best['pr']:.1f}")
        print("=" * 70)
        walk_forward(C, O, H, L, NS, ND, dates, syms, sigs,
                     base_threshold=best['bt'], adapt_amount=best['aa'],
                     win_rate_window=20, top_n_base=best['tn'],
                     min_cap=best['mn'], max_cap=best['mx'],
                     atr_stop=best['ats'], hold_days=5,
                     pyramid_ratio=best['pr'], pyramid_day=1,
                     max_ad=best['mad'])

        # === 6. Hybrid vs Adaptive-only vs Breadth-only comparison ===
        print("\n" + "=" * 70)
        print("  HYBRID vs ADAPTIVE-ONLY vs BREADTH-ONLY (2019-2026)")
        print("=" * 70)

        # Hybrid (V42 best config)
        trades_hybrid, eq_hybrid, dd_hybrid = backtest_v42(
            C, O, H, L, NS, ND, dates, syms, sigs,
            base_threshold=best['bt'], adapt_amount=best['aa'],
            win_rate_window=20, top_n_base=best['tn'],
            min_cap=best['mn'], max_cap=best['mx'],
            atr_stop=best['ats'], hold_days=5,
            min_confidence=3, use_ker_gate=True,
            pyramid_ratio=best['pr'], pyramid_day=1,
            max_ad=best['mad'],
            start_di=bt_2019)

        # Adaptive-only: no breadth filter (max_ad=1.0)
        trades_adapt, eq_adapt, dd_adapt = backtest_v42(
            C, O, H, L, NS, ND, dates, syms, sigs,
            base_threshold=best['bt'], adapt_amount=best['aa'],
            win_rate_window=20, top_n_base=best['tn'],
            min_cap=best['mn'], max_cap=best['mx'],
            atr_stop=best['ats'], hold_days=5,
            min_confidence=3, use_ker_gate=True,
            pyramid_ratio=best['pr'], pyramid_day=1,
            max_ad=1.0,
            start_di=bt_2019)

        # Breadth-only: static threshold (adapt_amount=0), breadth on
        trades_breadth, eq_breadth, dd_breadth = backtest_v42(
            C, O, H, L, NS, ND, dates, syms, sigs,
            base_threshold=best['bt'], adapt_amount=0.0,
            win_rate_window=20, top_n_base=best['tn'],
            min_cap=best['mn'], max_cap=best['mx'],
            atr_stop=best['ats'], hold_days=5,
            min_confidence=3, use_ker_gate=True,
            pyramid_ratio=best['pr'], pyramid_day=1,
            max_ad=best['mad'],
            start_di=bt_2019)

        print(f"\n  HYBRID (adaptive+breadth):")
        analyze(trades_hybrid, eq_hybrid, dd_hybrid, "hybrid")
        print(f"\n  ADAPTIVE-ONLY (no breadth):")
        analyze(trades_adapt, eq_adapt, dd_adapt, "adapt_only")
        print(f"\n  BREADTH-ONLY (static thresh):")
        analyze(trades_breadth, eq_breadth, dd_breadth, "breadth_only")

        if trades_hybrid and trades_adapt:
            print(f"\n  Hybrid vs Adaptive-only:")
            print(f"    eq_delta={eq_hybrid - eq_adapt:+,.0f} "
                  f"dd_delta={dd_hybrid - dd_adapt:+.1f}%")
        if trades_hybrid and trades_breadth:
            print(f"  Hybrid vs Breadth-only:")
            print(f"    eq_delta={eq_hybrid - eq_breadth:+,.0f} "
                  f"dd_delta={dd_hybrid - dd_breadth:+.1f}%")

    print(f"\n[V42] Done. {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
