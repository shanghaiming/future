"""
V39: Adaptive Threshold Rank Mean Reversion
============================================
Builds on V18 cross-sectional rank framework. The key innovation:
the entry threshold adapts based on recent trade performance.

When recent trades are winning (win_rate > 60%), relax the threshold
to take more trades. When recent trades are losing (win_rate < 50%),
tighten the threshold to be more selective. Acts as a P-controller
on the entry threshold.

Adaptive logic:
  - Track rolling win rate over last N trades (configurable window)
  - If win_rate > 60%: threshold = base - adapt_amount (relax)
  - If win_rate 50-60%: threshold = base (neutral)
  - If win_rate < 50%: threshold = base + adapt_amount (tighten)
  - Cap threshold in [min_cap, max_cap]
  - Also adapt top_n: expand when winning, contract when losing

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
    print("[V39] Computing raw factors...", flush=True)

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
    print("[V39] Computing cross-sectional ranks...", flush=True)

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
    print("[V39] Building composite signal...", flush=True)

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


def compute_all_signals(C, O, H, L, V, OI, NS, ND, weights=None):
    if weights is None:
        weights = DEFAULT_WEIGHTS

    raw = compute_raw_factors(C, O, H, L, V, OI, NS, ND)
    ranks = compute_cross_sectional_ranks(raw, NS, ND)
    ker_regime = compute_ker(C, NS, ND)
    composite, n_confirm = build_composite_signal(ranks, weights, NS, ND)

    return {
        'composite': composite,
        'n_confirm': n_confirm,
        'ker_regime': ker_regime,
        'ranks': ranks,
    }


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


def backtest_v39(
    C, O, H, L, NS, ND, dates, syms, sigs,
    base_threshold=0.80,
    adapt_amount=0.05,
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
    start_di=60,
    end_di=None,
):
    """Backtest V39 with adaptive threshold and adaptive top_n."""
    composite = sigs['composite']
    ker_regime = sigs['ker_regime']
    n_confirm = sigs['n_confirm']

    if end_di is None:
        end_di = ND - 1

    equity = CASH0
    peak = equity
    max_dd = 0.0
    positions = []
    trades = []

    # Rolling win/loss tracker for adaptive threshold
    recent_trades_win: list = []

    # Track threshold/top_n used per trade for diagnostics
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

    # Report threshold distribution
    thresholds_used = [t.get('threshold', 0) for t in trades]
    avg_thresh = np.mean(thresholds_used) if thresholds_used else 0

    print(f"  {label}: {len(trades)}t (base:{n_base} pyr:{n_pyr} stop:{n_stop} hold:{n_hold}) "
          f"WR={wr:.1f}% ann={ann:+.1f}% DD={max_dd:.1f}% Sh={sh:.2f} eq={equity:,.0f} "
          f"avg_thresh={avg_thresh:.3f}")

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


def walk_forward(C, O, H, L, NS, ND, dates, syms, sigs,
                 base_threshold=0.80, adapt_amount=0.05,
                 win_rate_window=20, top_n_base=1,
                 min_cap=0.70, max_cap=0.95,
                 atr_stop=3.0, hold_days=5,
                 pyramid_ratio=0.5, pyramid_day=1):
    """Walk-forward validation: year-by-year out-of-sample with adaptive threshold.

    Note: The adaptive state (recent_trades_win) is NOT reset between years,
    mimicking real trading where the adaptation persists.
    """
    print(f"\n{'=' * 70}")
    print(f"  WALK-FORWARD V39 (base_t={base_threshold} adapt={adapt_amount} "
          f"win_w={win_rate_window} top_n={top_n_base})")
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

        trades, _, _ = backtest_v39(
            C, O, H, L, NS, ND, dates, syms, sigs,
            base_threshold=base_threshold, adapt_amount=adapt_amount,
            win_rate_window=win_rate_window, top_n_base=top_n_base,
            min_cap=min_cap, max_cap=max_cap,
            atr_stop=atr_stop, hold_days=hold_days,
            min_confidence=3, use_ker_gate=True,
            pyramid_ratio=pyramid_ratio, pyramid_day=pyramid_day,
            start_di=test_start, end_di=test_end_idx + 1)

        test_trades = [t for t in trades if dates[t['di']].year == test_year]
        all_trades.extend(test_trades)

        if test_trades:
            n = len(test_trades)
            nw = sum(1 for t in test_trades if t['pnl_pct'] > 0)
            wr_val = nw / n * 100
            avg = np.mean([t['pnl_pct'] for t in test_trades])
            avg_t = np.mean([t.get('threshold', 0) for t in test_trades])
            print(f"  {test_year}: {n}t WR={wr_val:.1f}% avg={avg:+.2f}% "
                  f"thresh={avg_t:.3f}", flush=True)
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


def main():
    t0 = time.time()
    print("=" * 70)
    print("  V39: ADAPTIVE THRESHOLD RANK MEAN REVERSION")
    print("  V18 rank signals + self-tuning entry thresholds")
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

    # === 1. Walk-Forward Validation with default adaptive params ===
    print("\n" + "=" * 70)
    print("  WALK-FORWARD VALIDATION (2019-2026) -- DEFAULT ADAPTIVE")
    print("=" * 70)

    for bt, aa, ww in [(0.80, 0.05, 20), (0.75, 0.05, 20), (0.85, 0.05, 20)]:
        walk_forward(C, O, H, L, NS, ND, dates, syms, sigs,
                     base_threshold=bt, adapt_amount=aa,
                     win_rate_window=ww, top_n_base=1,
                     min_cap=0.70, max_cap=0.95,
                     atr_stop=3.0, hold_days=5,
                     pyramid_ratio=0.5, pyramid_day=1)

    # === 2. Full 10-year backtest with adaptive profiles ===
    print("\n" + "=" * 70)
    print("  FULL 2016-2026 (10 years) -- ADAPTIVE PROFILES")
    print("=" * 70)

    profiles = [
        (0.80, 0.05, 20, 1, 0.70, 0.95, 3.0, 0.5, "Default adaptive"),
        (0.75, 0.05, 20, 1, 0.70, 0.90, 3.0, 0.5, "Relaxed base"),
        (0.85, 0.05, 20, 1, 0.75, 0.95, 3.0, 0.5, "Tight base"),
        (0.80, 0.07, 15, 1, 0.70, 0.95, 3.0, 0.5, "Aggressive adapt"),
        (0.80, 0.03, 30, 1, 0.70, 0.95, 3.0, 0.5, "Conservative adapt"),
    ]

    for bt, aa, ww, tn, mn, mx, ats, pr, label in profiles:
        trades, eq, dd = backtest_v39(
            C, O, H, L, NS, ND, dates, syms, sigs,
            base_threshold=bt, adapt_amount=aa,
            win_rate_window=ww, top_n_base=tn,
            min_cap=mn, max_cap=mx,
            atr_stop=ats, hold_days=5,
            min_confidence=3, use_ker_gate=True,
            pyramid_ratio=pr, pyramid_day=1,
            start_di=60)
        print(f"\n  {label}")
        analyze(trades, eq, dd, label)

    # === 3. Full parameter sweep ===
    print("\n" + "=" * 70)
    print("  PARAMETER SWEEP (2019-2026)")
    print("=" * 70)

    results = []

    sweep_params = {
        'base_threshold': [0.75, 0.80, 0.85],
        'adapt_amount': [0.03, 0.05, 0.07],
        'win_rate_window': [15, 20, 30],
        'top_n_base': [1, 2],
        'min_cap': [0.70, 0.75],
        'max_cap': [0.90, 0.95],
        'atr_stop': [2.5, 3.0],
        'pyramid_ratio': [0.0, 0.5],
    }

    total_combos = 1
    for v in sweep_params.values():
        total_combos *= len(v)
    print(f"  Total combinations: {total_combos}")

    combo_count = 0
    for bt, aa, ww, tn, mn, mx, ats, pr in product(
        sweep_params['base_threshold'],
        sweep_params['adapt_amount'],
        sweep_params['win_rate_window'],
        sweep_params['top_n_base'],
        sweep_params['min_cap'],
        sweep_params['max_cap'],
        sweep_params['atr_stop'],
        sweep_params['pyramid_ratio'],
    ):
        # Skip invalid: min_cap must be < base_threshold < max_cap
        if mn >= bt or bt >= mx:
            continue

        combo_count += 1
        trades, eq, dd = backtest_v39(
            C, O, H, L, NS, ND, dates, syms, sigs,
            base_threshold=bt, adapt_amount=aa,
            win_rate_window=ww, top_n_base=tn,
            min_cap=mn, max_cap=mx,
            atr_stop=ats, hold_days=5,
            min_confidence=3, use_ker_gate=True,
            pyramid_ratio=pr, pyramid_day=1,
            start_di=bt_2019)

        if len(trades) < 10:
            continue

        nw = sum(1 for t in trades if t['pnl_pct'] > 0)
        wr = nw / len(trades) * 100
        n_days = max(1, trades[-1]['di'] - trades[0]['di'])
        ann = ((eq / CASH0) ** (1 / max(1.0, n_days / 252)) - 1) * 100
        ap = [t['pnl_abs'] for t in sorted(trades, key=lambda x: x['di'])]
        rets_arr = np.array(ap) / CASH0
        sh_val = (np.mean(rets_arr) / np.std(rets_arr) * np.sqrt(252)
                  if np.std(rets_arr) > 0 else 0)

        results.append({
            'bt': bt, 'aa': aa, 'ww': ww, 'tn': tn,
            'mn': mn, 'mx': mx, 'ats': ats, 'pr': pr,
            'n': len(trades), 'wr': wr, 'ann': ann,
            'dd': dd, 'sharpe': sh_val, 'eq': eq,
        })

    results.sort(key=lambda x: -x['sharpe'])
    print(f"\n  Evaluated {combo_count} valid combinations, {len(results)} with 10+ trades")
    print(f"\n{'BT':>4} {'AA':>4} {'WW':>3} {'TN':>3} {'MnC':>4} {'MxC':>4} "
          f"{'ATS':>4} {'Pyr':>4} {'N':>5} {'WR':>5} {'Ann':>8} {'DD':>6} {'Sh':>6}")
    print("-" * 85)
    for r in results[:30]:
        print(f"{r['bt']:>4.2f} {r['aa']:>4.2f} {r['ww']:>3} {r['tn']:>3} "
              f"{r['mn']:>4.2f} {r['mx']:>4.2f} {r['ats']:>4.1f} {r['pr']:>4.1f} "
              f"{r['n']:>5} {r['wr']:>5.1f} {r['ann']:>+8.1f} "
              f"{r['dd']:>6.1f} {r['sharpe']:>6.2f}")

    # === 4. Top configs: full 10-year backtest ===
    print("\n" + "=" * 70)
    print("  TOP CONFIGS -- FULL 10-YEAR (2016-2026)")
    print("=" * 70)

    for r in results[:5]:
        trades, eq, dd = backtest_v39(
            C, O, H, L, NS, ND, dates, syms, sigs,
            base_threshold=r['bt'], adapt_amount=r['aa'],
            win_rate_window=r['ww'], top_n_base=r['tn'],
            min_cap=r['mn'], max_cap=r['mx'],
            atr_stop=r['ats'], hold_days=5,
            min_confidence=3, use_ker_gate=True,
            pyramid_ratio=r['pr'], pyramid_day=1,
            start_di=60)
        label = (f"bt={r['bt']:.2f} aa={r['aa']:.2f} ww={r['ww']} tn={r['tn']} "
                 f"mn={r['mn']:.2f} mx={r['mx']:.2f} ats={r['ats']:.1f} pr={r['pr']:.1f}")
        print(f"\n  FULL {label}")
        analyze(trades, eq, dd, label)

    # === 5. Walk-forward for top config ===
    if results:
        best = results[0]
        print("\n" + "=" * 70)
        print(f"  BEST WF: bt={best['bt']:.2f} aa={best['aa']:.2f} "
              f"ww={best['ww']} tn={best['tn']} mn={best['mn']:.2f} "
              f"mx={best['mx']:.2f} ats={best['ats']:.1f} pr={best['pr']:.1f}")
        print("=" * 70)
        walk_forward(C, O, H, L, NS, ND, dates, syms, sigs,
                     base_threshold=best['bt'], adapt_amount=best['aa'],
                     win_rate_window=best['ww'], top_n_base=best['tn'],
                     min_cap=best['mn'], max_cap=best['mx'],
                     atr_stop=best['ats'], hold_days=5,
                     pyramid_ratio=best['pr'], pyramid_day=1)

        # === 6. Compare: adaptive vs static threshold ===
        print("\n" + "=" * 70)
        print("  ADAPTIVE vs STATIC THRESHOLD COMPARISON (2019-2026)")
        print("=" * 70)

        # Adaptive version
        trades_adapt, eq_adapt, dd_adapt = backtest_v39(
            C, O, H, L, NS, ND, dates, syms, sigs,
            base_threshold=best['bt'], adapt_amount=best['aa'],
            win_rate_window=best['ww'], top_n_base=best['tn'],
            min_cap=best['mn'], max_cap=best['mx'],
            atr_stop=best['ats'], hold_days=5,
            min_confidence=3, use_ker_gate=True,
            pyramid_ratio=best['pr'], pyramid_day=1,
            start_di=bt_2019)

        # Static version: adapt_amount=0 disables adaptation
        trades_static, eq_static, dd_static = backtest_v39(
            C, O, H, L, NS, ND, dates, syms, sigs,
            base_threshold=best['bt'], adapt_amount=0.0,
            win_rate_window=best['ww'], top_n_base=best['tn'],
            min_cap=best['mn'], max_cap=best['mx'],
            atr_stop=best['ats'], hold_days=5,
            min_confidence=3, use_ker_gate=True,
            pyramid_ratio=best['pr'], pyramid_day=1,
            start_di=bt_2019)

        print(f"\n  ADAPTIVE:")
        analyze(trades_adapt, eq_adapt, dd_adapt, "adaptive")
        print(f"\n  STATIC:")
        analyze(trades_static, eq_static, dd_static, "static")

        if trades_adapt:
            adapt_wr = sum(1 for t in trades_adapt if t['pnl_pct'] > 0) / len(trades_adapt)
            print(f"\n  Adaptive improvement: "
                  f"eq_delta={eq_adapt - eq_static:+,.0f} "
                  f"dd_delta={dd_adapt - dd_static:+.1f}%")

    print(f"\n[V39] Done. {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
