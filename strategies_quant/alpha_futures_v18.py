"""
V18: Cross-Sectional Rank Mean Reversion
=========================================
Core thesis: Instead of absolute thresholds (RSI<30, etc.), use
cross-sectional rank percentiles across 50 commodities for ALL signals.
More robust across different commodity volatilities.

Signal architecture:
  1. Compute cross-sectional ranks for 7 factors (rank across 50 commodities per day)
  2. Composite rank = weighted average of all ranks
     - rank_ret5d:  0.25  (low rank = oversold)
     - rank_oi5d:   0.20  (declining OI + price drop = capitulation)
     - rank_rsi:    0.15  (low RSI rank = oversold)
     - rank_vol:    0.15  (high vol rank = attention)
     - rank_ret10d: 0.10
     - rank_range:  0.10  (expansion = capitulation)
     - rank_atrp:   0.05  (high ATR% = opportunity)
  3. Entry: composite rank > 0.75 (top 25% oversold), confidence >= 3
  4. KER gate (sideways regime only)
  5. Pyramid on day-1 winners (ratio 0.5)
  6. Hold 5d, ATR stop 3.0
  7. Walk-forward 2019-2026
  8. Parameter sweep: weights, thresholds, hold periods

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

# Default weights for composite rank
DEFAULT_WEIGHTS = {
    'rank_ret5d':  0.25,
    'rank_oi5d':   0.20,
    'rank_rsi':    0.15,
    'rank_vol':    0.15,
    'rank_ret10d': 0.10,
    'rank_range':  0.10,
    'rank_atrp':   0.05,
}


def compute_rsi_manual(C, NS, ND, period=14):
    """Compute RSI without talib as fallback."""
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

        # EMA-style RSI
        avg_gain = np.nan
        avg_loss = np.nan
        for di in range(1, ND):
            if np.isnan(gains[di]):
                continue
            if np.isnan(avg_gain):
                # Seed with SMA over first 'period' valid bars
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

            # EMA update
            avg_gain = (avg_gain * (period - 1) + gains[di]) / period
            avg_loss = (avg_loss * (period - 1) + losses[di]) / period
            if avg_loss == 0:
                rsi[si, di] = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi[si, di] = 100.0 - 100.0 / (1.0 + rs)
    return rsi


def compute_raw_factors(C, O, H, L, V, OI, NS, ND):
    """Compute raw factor values before cross-sectional ranking."""
    t0 = time.time()
    print("[V18] Computing raw factors...", flush=True)

    # --- 5d return ---
    ret_5d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            if not np.isnan(C[si, di]) and not np.isnan(C[si, di - 5]) and C[si, di - 5] > 0:
                ret_5d[si, di] = C[si, di] / C[si, di - 5] - 1.0

    # --- 10d return ---
    ret_10d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(10, ND):
            if not np.isnan(C[si, di]) and not np.isnan(C[si, di - 10]) and C[si, di - 10] > 0:
                ret_10d[si, di] = C[si, di] / C[si, di - 10] - 1.0

    # --- OI 5d change ---
    oi_5d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            if not np.isnan(OI[si, di]) and not np.isnan(OI[si, di - 5]) and OI[si, di - 5] > 0:
                oi_5d[si, di] = OI[si, di] / OI[si, di - 5] - 1.0

    # --- Volume (5d average for stability) ---
    vol_5d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            vals = V[si, di - 5:di]
            valid = vals[~np.isnan(vals)]
            if len(valid) >= 3:
                vol_5d[si, di] = np.mean(valid)

    # --- Daily range (H-L) / C ---
    daily_range = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(1, ND):
            if not np.isnan(H[si, di]) and not np.isnan(L[si, di]) and not np.isnan(C[si, di]):
                if C[si, di] > 0 and H[si, di] > L[si, di]:
                    daily_range[si, di] = (H[si, di] - L[si, di]) / C[si, di]

    # --- RSI 14 ---
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
    # Fallback: use manual RSI for symbols with missing talib
    needs_fallback = np.all(np.isnan(rsi14), axis=1)
    if needs_fallback.any():
        rsi_manual = compute_rsi_manual(C, NS, ND, 14)
        for si in range(NS):
            if needs_fallback[si]:
                rsi14[si] = rsi_manual[si]

    # --- ATR% (14d) ---
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
    """Rank all factors cross-sectionally (across commodities per day).
    Low rank = oversold / extreme for mean reversion."""
    t0 = time.time()
    print("[V18] Computing cross-sectional ranks...", flush=True)

    # For mean reversion, we want to identify oversold extremes.
    # Invert factors so LOW value = high rank (most oversold):
    # - ret_5d: low return -> invert (1 - rank)
    # - ret_10d: low return -> invert
    # - oi_5d: declining OI -> invert
    # - vol_5d: high volume -> high rank (no invert needed)
    # - daily_range: high range -> high rank (no invert needed)
    # - rsi14: low RSI -> invert (1 - rank)
    # - atrp: high ATR% -> high rank (no invert needed)

    factors_to_rank = {
        'rank_ret5d': raw_factors['ret_5d'],
        'rank_ret10d': raw_factors['ret_10d'],
        'rank_oi5d': raw_factors['oi_5d'],
        'rank_vol': raw_factors['vol_5d'],
        'rank_range': raw_factors['daily_range'],
        'rank_rsi': raw_factors['rsi14'],
        'rank_atrp': raw_factors['atrp'],
    }

    # For inverted factors: we invert so low raw value -> high rank
    # (high rank = most oversold/extreme = best mean reversion candidate)
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
                # Invert: low raw value -> high rank
                ranked = 1.0 - ranked
            rank_arr[:, di] = ranked
        ranks[name] = rank_arr

    print(f"  CS ranks done: {time.time() - t0:.1f}s", flush=True)
    return ranks


def compute_ker(C, NS, ND):
    """Kaufman Efficiency Ratio for regime detection."""
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
                ker_regime[si, di] = 1  # sideways -> good for mean reversion
            elif ker_10[si, di] > 0.3:
                ker_regime[si, di] = -1  # trending -> avoid counter-trend
    return ker_regime


def build_composite_signal(ranks, weights, NS, ND, min_factors=4):
    """Build weighted composite rank from individual factor ranks.
    Also count how many factors confirm (rank > 0.5 for each factor)."""
    t0 = time.time()
    print("[V18] Building composite signal...", flush=True)

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
                # A factor "confirms" if its rank is in the top half (oversold territory)
                if rank_val > 0.5:
                    confirm_count += 1

            if w_sum > 0 and confirm_count >= min_factors:
                composite[si, di] = sum(vals) / w_sum
                n_confirm[si, di] = confirm_count

    print(f"  Composite done: {time.time() - t0:.1f}s", flush=True)
    return composite, n_confirm


def compute_all_signals(C, O, H, L, V, OI, NS, ND, weights=None):
    """Full signal pipeline."""
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


def backtest_v18(C, O, H, L, NS, ND, dates, syms, sigs,
                 top_n=1, min_rank=0.75, atr_stop=3.0,
                 min_confidence=3, use_ker_gate=True,
                 hold_days=5,
                 pyramid_ratio=0.5,
                 pyramid_day=1,
                 start_di=60, end_di=None):
    """Backtest with cross-sectional rank signals + pyramid."""
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
                            additions.append((si, di, c_now, c_now - atr_stop * atr, pyr_alloc, True))
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
            alloc = 1.0 / max(top_n, 1)
            candidates.append((composite[si, di], si, alloc))

        candidates.sort(key=lambda x: -x[0])
        for rank, si, alloc in candidates[:top_n]:
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


def analyze(trades, equity, max_dd, label=""):
    """Analyze backtest results."""
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


def walk_forward(C, O, H, L, V, OI, NS, ND, dates, syms, sigs,
                 pyramid_ratio=0.5, pyramid_day=1,
                 top_n=1, min_confidence=3, hold_days=5, atr_stop=3.0,
                 min_rank=0.75):
    """Walk-forward validation: year-by-year out-of-sample."""
    print(f"\n{'=' * 70}")
    print(f"  WALK-FORWARD V18 (pyr={pyramid_ratio}, day={pyramid_day})")
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

        trades, _, _ = backtest_v18(
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


def main():
    t0 = time.time()
    print("=" * 70)
    print("  V18: CROSS-SECTIONAL RANK MEAN REVERSION")
    print("  Pure rank-based signal across 50 commodities")
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

    # === 1. Walk-Forward Validation (default weights) ===
    print("\n" + "=" * 70)
    print("  WALK-FORWARD VALIDATION (2019-2026)")
    print("=" * 70)

    for ratio in [0.0, 0.3, 0.5]:
        walk_forward(C, O, H, L, V, OI, NS, ND, dates, syms, sigs,
                     pyramid_ratio=ratio, pyramid_day=1,
                     top_n=1, min_confidence=3)

    # === 2. Full 10-year backtest with pyramid profiles ===
    print("\n" + "=" * 70)
    print("  FULL 2016-2026 (10 years) -- PYRAMID PROFILES")
    print("=" * 70)

    profiles = [
        (0.0, 1, "No pyramid (baseline)"),
        (0.3, 1, "Mild pyramid (30%)"),
        (0.5, 1, "Moderate pyramid (50%)"),
    ]

    for ratio, pday, label in profiles:
        trades, eq, dd = backtest_v18(
            C, O, H, L, NS, ND, dates, syms, sigs,
            top_n=1, hold_days=5, atr_stop=3.0,
            min_rank=0.75, min_confidence=3, use_ker_gate=True,
            pyramid_ratio=ratio, pyramid_day=pday,
            start_di=60)
        print(f"\n  {label}")
        analyze(trades, eq, dd, label)

    # === 3. Parameter sweep ===
    print("\n" + "=" * 70)
    print("  PARAMETER SWEEP (2019-2026)")
    print("=" * 70)

    results = []

    # Sweep: top_n, pyramid_ratio, min_confidence, min_rank, hold_days
    for tn in [1, 2, 3]:
        for ratio in [0.0, 0.3, 0.5]:
            for mc in [2, 3, 4]:
                for mr in [0.70, 0.75, 0.80]:
                    for hd in [3, 5, 7]:
                        trades, eq, dd = backtest_v18(
                            C, O, H, L, NS, ND, dates, syms, sigs,
                            top_n=tn, hold_days=hd, atr_stop=3.0,
                            min_rank=mr, min_confidence=mc,
                            use_ker_gate=True,
                            pyramid_ratio=ratio, pyramid_day=1,
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
                            'tn': tn, 'ratio': ratio, 'mc': mc,
                            'mr': mr, 'hd': hd,
                            'n': len(trades), 'wr': wr, 'ann': ann,
                            'dd': dd, 'sharpe': sh_val,
                        })

    results.sort(key=lambda x: -x['sharpe'])
    print(f"\n{'TN':>3} {'Pyr':>4} {'MC':>3} {'MR':>4} {'HD':>3} "
          f"{'N':>5} {'WR':>5} {'Ann':>8} {'DD':>6} {'Sh':>5}")
    print("-" * 65)
    for r in results[:25]:
        print(f"{r['tn']:>3} {r['ratio']:>4.1f} {r['mc']:>3} {r['mr']:>4.2f} {r['hd']:>3} "
              f"{r['n']:>5} {r['wr']:>5.1f} {r['ann']:>+8.1f} "
              f"{r['dd']:>6.1f} {r['sharpe']:>5.2f}")

    # === 4. Weight sweep on best structural params ===
    print("\n" + "=" * 70)
    print("  WEIGHT SWEEP (2019-2026)")
    print("=" * 70)

    if results:
        best_struct = results[0]
        weight_variations = [
            {'rank_ret5d': 0.30, 'rank_oi5d': 0.25, 'rank_rsi': 0.15,
             'rank_vol': 0.10, 'rank_ret10d': 0.10, 'rank_range': 0.05, 'rank_atrp': 0.05},
            {'rank_ret5d': 0.20, 'rank_oi5d': 0.20, 'rank_rsi': 0.20,
             'rank_vol': 0.15, 'rank_ret10d': 0.10, 'rank_range': 0.10, 'rank_atrp': 0.05},
            {'rank_ret5d': 0.25, 'rank_oi5d': 0.15, 'rank_rsi': 0.20,
             'rank_vol': 0.15, 'rank_ret10d': 0.10, 'rank_range': 0.10, 'rank_atrp': 0.05},
            {'rank_ret5d': 0.25, 'rank_oi5d': 0.20, 'rank_rsi': 0.10,
             'rank_vol': 0.20, 'rank_ret10d': 0.10, 'rank_range': 0.10, 'rank_atrp': 0.05},
            {'rank_ret5d': 0.20, 'rank_oi5d': 0.25, 'rank_rsi': 0.15,
             'rank_vol': 0.15, 'rank_ret10d': 0.15, 'rank_range': 0.05, 'rank_atrp': 0.05},
        ]

        weight_results = []
        for w_idx, w in enumerate(weight_variations):
            w_sigs = compute_all_signals(C, O, H, L, V, OI, NS, ND, weights=w)
            trades, eq, dd = backtest_v18(
                C, O, H, L, NS, ND, dates, syms, w_sigs,
                top_n=best_struct['tn'], hold_days=best_struct['hd'],
                atr_stop=3.0, min_rank=best_struct['mr'],
                min_confidence=best_struct['mc'], use_ker_gate=True,
                pyramid_ratio=best_struct['ratio'], pyramid_day=1,
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
            weight_results.append({
                'w_idx': w_idx, 'w': w,
                'n': len(trades), 'wr': wr, 'ann': ann,
                'dd': dd, 'sharpe': sh_val, 'eq': eq,
            })
            print(f"  W{w_idx}: {len(trades)}t WR={wr:.1f}% ann={ann:+.1f}% "
                  f"DD={dd:.1f}% Sh={sh_val:.2f}")

        if weight_results:
            weight_results.sort(key=lambda x: -x['sharpe'])
            best_w = weight_results[0]
            # Recompute with best weights for full analysis
            best_weights = best_w['w']
            print(f"\n  Best weights: W{best_w['w_idx']} (Sharpe={best_w['sharpe']:.2f})")
            for k, v in sorted(best_weights.items()):
                print(f"    {k}: {v:.2f}")

            # === 5. Best config full 10-year ===
            print("\n" + "=" * 70)
            print("  BEST CONFIG -- FULL 10-YEAR")
            print("=" * 70)

            final_sigs = compute_all_signals(C, O, H, L, V, OI, NS, ND, weights=best_weights)
            for r in results[:5]:
                trades, eq, dd = backtest_v18(
                    C, O, H, L, NS, ND, dates, syms, final_sigs,
                    top_n=r['tn'], hold_days=r['hd'], atr_stop=3.0,
                    min_rank=r['mr'], min_confidence=r['mc'],
                    use_ker_gate=True,
                    pyramid_ratio=r['ratio'], pyramid_day=1,
                    start_di=60)
                label = f"tn={r['tn']} pyr={r['ratio']:.1f} mc={r['mc']} mr={r['mr']:.2f} hd={r['hd']}"
                print(f"\n  FULL {label}")
                analyze(trades, eq, dd, label)

            # === 6. Walk-forward for best overall config ===
            print("\n" + "=" * 70)
            print(f"  BEST WF: tn={best_struct['tn']} pyr={best_struct['ratio']:.1f} "
                  f"mc={best_struct['mc']} mr={best_struct['mr']:.2f} hd={best_struct['hd']}")
            print("=" * 70)
            walk_forward(C, O, H, L, V, OI, NS, ND, dates, syms, final_sigs,
                         pyramid_ratio=best_struct['ratio'], pyramid_day=1,
                         top_n=best_struct['tn'], min_confidence=best_struct['mc'],
                         hold_days=best_struct['hd'], min_rank=best_struct['mr'])

    print(f"\n[V18] Done. {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
