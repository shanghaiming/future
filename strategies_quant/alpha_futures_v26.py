"""
V26: Rank Momentum + Mean Reversion
====================================
Extends V18's cross-sectional ranking with a "rank momentum" dimension.

Core thesis: Commodities whose oversold rank is improving rapidly (getting
more oversold faster) may offer better entry timing for mean reversion.

Signal architecture:
  1. Same 7 cross-sectional ranks as V18
  2. NEW: Rank momentum = current_composite_rank - rank_5d_ago
     Captures "accelerating oversold" which may be a stronger signal
  3. Composite = V18 weighted average (0.85) + rank_momentum (0.15)
  4. KER gate, confidence >= 2, hold 5d, ATR stop 3.0
  5. Pyramid on day-1 winners (ratio 0.5)

Parameter sweep:
  - rank_momentum_weight: 0.05, 0.10, 0.15, 0.20
  - top_n: 1, 2, 3
  - min_rank: 0.70, 0.75, 0.80
  - min_confidence: 2, 3
  - pyramid: 0.0, 0.5
  - atr_stop: 2.5, 3.0

Walk-forward 2019-2026, full 10-year for top configs.

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

# Default weights for V18 composite rank (baseline)
BASE_WEIGHTS = {
    'rank_ret5d':  0.25,
    'rank_oi5d':   0.20,
    'rank_rsi':    0.15,
    'rank_vol':    0.15,
    'rank_ret10d': 0.10,
    'rank_range':  0.10,
    'rank_atrp':   0.05,
}

DEFAULT_RANK_MOMENTUM_WEIGHT = 0.15


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
    """Compute raw factor values before cross-sectional ranking."""
    t0 = time.time()
    print("[V26] Computing raw factors...", flush=True)

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
    """Rank all factors cross-sectionally (across commodities per day).
    Low rank = oversold / extreme for mean reversion."""
    t0 = time.time()
    print("[V26] Computing cross-sectional ranks...", flush=True)

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


def build_base_composite(ranks, weights, NS, ND, min_factors=4):
    """Build V18-style weighted composite rank from individual factor ranks.
    Returns composite and confirmation count."""
    t0 = time.time()
    print("[V26] Building base composite signal...", flush=True)

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

    print(f"  Base composite done: {time.time() - t0:.1f}s", flush=True)
    return composite, n_confirm


def compute_rank_momentum(base_composite, NS, ND, lookback=5):
    """Rank momentum = current_composite - composite_lookback_days_ago.
    Positive = rank is improving (getting more oversold faster).
    This captures accelerating oversold which may be a stronger entry signal."""
    t0 = time.time()
    print(f"[V26] Computing rank momentum (lookback={lookback})...", flush=True)

    momentum = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(lookback, ND):
            current = base_composite[si, di]
            past = base_composite[si, di - lookback]
            if not np.isnan(current) and not np.isnan(past):
                momentum[si, di] = current - past

    print(f"  Rank momentum done: {time.time() - t0:.1f}s", flush=True)
    return momentum


def build_final_signal(base_composite, rank_momentum, rank_momentum_weight,
                       NS, ND):
    """Blend base composite with rank momentum.
    Final = base * (1 - rmw) + momentum * rmw
    Rank momentum is re-scaled to [0, 1] range via cross-sectional rank."""
    t0 = time.time()
    print(f"[V26] Building final signal (rmw={rank_momentum_weight})...", flush=True)

    # Rank momentum cross-sectionally to normalize to [0, 1]
    ranked_momentum = np.full((NS, ND), np.nan)
    for di in range(ND):
        vals = rank_momentum[:, di]
        valid_count = np.sum(~np.isnan(vals))
        if valid_count < 10:
            continue
        ranked_momentum[:, di] = pd.Series(vals).rank(
            pct=True, na_option='keep'
        ).values

    # Blend: base_weight * base + rmw * ranked_momentum
    base_weight = 1.0 - rank_momentum_weight
    final = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(ND):
            b = base_composite[si, di]
            m = ranked_momentum[si, di]
            if not np.isnan(b) and not np.isnan(m):
                final[si, di] = base_weight * b + rank_momentum_weight * m
            elif not np.isnan(b):
                final[si, di] = b

    print(f"  Final signal done: {time.time() - t0:.1f}s", flush=True)
    return final


def compute_all_signals(C, O, H, L, V, OI, NS, ND,
                        weights=None, rank_momentum_weight=0.15):
    """Full signal pipeline with rank momentum."""
    if weights is None:
        weights = BASE_WEIGHTS

    raw = compute_raw_factors(C, O, H, L, V, OI, NS, ND)
    ranks = compute_cross_sectional_ranks(raw, NS, ND)
    ker_regime = compute_ker(C, NS, ND)
    base_composite, n_confirm = build_base_composite(ranks, weights, NS, ND)
    rank_momentum = compute_rank_momentum(base_composite, NS, ND, lookback=5)
    final_signal = build_final_signal(
        base_composite, rank_momentum, rank_momentum_weight, NS, ND
    )

    return {
        'composite': final_signal,
        'base_composite': base_composite,
        'rank_momentum': rank_momentum,
        'n_confirm': n_confirm,
        'ker_regime': ker_regime,
        'ranks': ranks,
    }


def backtest_v26(C, O, H, L, NS, ND, dates, syms, sigs,
                 top_n=1, min_rank=0.75, atr_stop=3.0,
                 min_confidence=2, use_ker_gate=True,
                 hold_days=5,
                 pyramid_ratio=0.5,
                 pyramid_day=1,
                 start_di=60, end_di=None):
    """Backtest with rank momentum signals + pyramid."""
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
                            additions.append(
                                (si, di, c_now, c_now - atr_stop * atr, pyr_alloc, True)
                            )
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
                 top_n=1, min_confidence=2, hold_days=5, atr_stop=3.0,
                 min_rank=0.75):
    """Walk-forward validation: year-by-year out-of-sample."""
    print(f"\n{'=' * 70}")
    print(f"  WALK-FORWARD V26 (pyr={pyramid_ratio}, day={pyramid_day})")
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

        trades, _, _ = backtest_v26(
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
    print("  V26: RANK MOMENTUM + MEAN REVERSION")
    print("  Extends V18 cross-sectional rank with rank momentum dimension")
    print("=" * 70)

    C, O, H, L, V, OI, NS, ND, dates, syms = load_all_data(start='2016-01-01')
    print(f"  {NS} sym, {ND} days, "
          f"{dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")

    # Find 2019 start index for OOS testing
    bt_2019 = None
    for i, d in enumerate(dates):
        if d >= pd.Timestamp('2019-01-01'):
            bt_2019 = i
            break

    # === 1. Compare rank momentum weights ===
    print("\n" + "=" * 70)
    print("  RANK MOMENTUM WEIGHT COMPARISON (2019-2026)")
    print("=" * 70)

    rmw_results = []
    for rmw in [0.0, 0.05, 0.10, 0.15, 0.20]:
        sigs = compute_all_signals(C, O, H, L, V, OI, NS, ND,
                                   rank_momentum_weight=rmw)
        trades, eq, dd = backtest_v26(
            C, O, H, L, NS, ND, dates, syms, sigs,
            top_n=1, hold_days=5, atr_stop=3.0,
            min_rank=0.75, min_confidence=2, use_ker_gate=True,
            pyramid_ratio=0.5, pyramid_day=1,
            start_di=bt_2019)
        if len(trades) >= 10:
            nw = sum(1 for t in trades if t['pnl_pct'] > 0)
            wr = nw / len(trades) * 100
            n_days = max(1, trades[-1]['di'] - trades[0]['di'])
            ann = ((eq / CASH0) ** (1 / max(1.0, n_days / 252)) - 1) * 100
            ap = [t['pnl_abs'] for t in sorted(trades, key=lambda x: x['di'])]
            rets_arr = np.array(ap) / CASH0
            sh_val = (np.mean(rets_arr) / np.std(rets_arr) * np.sqrt(252)
                      if np.std(rets_arr) > 0 else 0)
            rmw_results.append({
                'rmw': rmw, 'n': len(trades), 'wr': wr,
                'ann': ann, 'dd': dd, 'sh': sh_val, 'eq': eq,
            })
            print(f"  RMW={rmw:.2f}: {len(trades)}t WR={wr:.1f}% ann={ann:+.1f}% "
                  f"DD={dd:.1f}% Sh={sh_val:.2f} eq={eq:,.0f}")
        else:
            print(f"  RMW={rmw:.2f}: insufficient trades ({len(trades)})")

    # Pick best RMW
    if rmw_results:
        best_rmw = max(rmw_results, key=lambda x: x['sh'])
        print(f"\n  Best RMW: {best_rmw['rmw']:.2f} (Sh={best_rmw['sh']:.2f})")
    else:
        best_rmw = {'rmw': 0.15}

    # === 2. Parameter Sweep ===
    print("\n" + "=" * 70)
    print("  PARAMETER SWEEP (2019-2026)")
    print("=" * 70)

    results = []

    for rmw in [0.05, 0.10, 0.15, 0.20]:
        sigs = compute_all_signals(C, O, H, L, V, OI, NS, ND,
                                   rank_momentum_weight=rmw)
        for tn in [1, 2, 3]:
            for ratio in [0.0, 0.5]:
                for mc in [2, 3]:
                    for mr in [0.70, 0.75, 0.80]:
                        for atr_s in [2.5, 3.0]:
                            trades, eq, dd = backtest_v26(
                                C, O, H, L, NS, ND, dates, syms, sigs,
                                top_n=tn, hold_days=5, atr_stop=atr_s,
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
                                'rmw': rmw, 'tn': tn, 'ratio': ratio, 'mc': mc,
                                'mr': mr, 'atr': atr_s,
                                'n': len(trades), 'wr': wr, 'ann': ann,
                                'dd': dd, 'sharpe': sh_val, 'eq': eq,
                            })

    results.sort(key=lambda x: -x['sharpe'])
    print(f"\n{'RMW':>4} {'TN':>3} {'Pyr':>4} {'MC':>3} {'MR':>4} {'ATR':>4} "
          f"{'N':>5} {'WR':>5} {'Ann':>8} {'DD':>6} {'Sh':>5}")
    print("-" * 75)
    for r in results[:30]:
        print(f"{r['rmw']:>4.2f} {r['tn']:>3} {r['ratio']:>4.1f} {r['mc']:>3} "
              f"{r['mr']:>4.2f} {r['atr']:>4.1f} "
              f"{r['n']:>5} {r['wr']:>5.1f} {r['ann']:>+8.1f} "
              f"{r['dd']:>6.1f} {r['sharpe']:>5.2f}")

    # === 3. Walk-forward for top configs ===
    if results:
        print("\n" + "=" * 70)
        print("  WALK-FORWARD FOR TOP 3 CONFIGS")
        print("=" * 70)

        seen = set()
        top_unique = []
        for r in results:
            key = (r['rmw'], r['tn'], r['ratio'], r['mc'], r['mr'], r['atr'])
            if key not in seen:
                seen.add(key)
                top_unique.append(r)
            if len(top_unique) >= 3:
                break

        for cfg in top_unique:
            sigs = compute_all_signals(C, O, H, L, V, OI, NS, ND,
                                       rank_momentum_weight=cfg['rmw'])
            walk_forward(C, O, H, L, V, OI, NS, ND, dates, syms, sigs,
                         pyramid_ratio=cfg['ratio'], pyramid_day=1,
                         top_n=cfg['tn'], min_confidence=cfg['mc'],
                         hold_days=5, min_rank=cfg['mr'], atr_stop=cfg['atr'])

    # === 4. Full 10-year backtest for top configs ===
    if results:
        print("\n" + "=" * 70)
        print("  FULL 10-YEAR (2016-2026) -- TOP 5 CONFIGS")
        print("=" * 70)

        seen2 = set()
        top5 = []
        for r in results:
            key = (r['rmw'], r['tn'], r['ratio'], r['mc'], r['mr'], r['atr'])
            if key not in seen2:
                seen2.add(key)
                top5.append(r)
            if len(top5) >= 5:
                break

        for cfg in top5:
            sigs = compute_all_signals(C, O, H, L, V, OI, NS, ND,
                                       rank_momentum_weight=cfg['rmw'])
            trades, eq, dd = backtest_v26(
                C, O, H, L, NS, ND, dates, syms, sigs,
                top_n=cfg['tn'], hold_days=5, atr_stop=cfg['atr'],
                min_rank=cfg['mr'], min_confidence=cfg['mc'],
                use_ker_gate=True,
                pyramid_ratio=cfg['ratio'], pyramid_day=1,
                start_di=60)
            label = (f"rmw={cfg['rmw']:.2f} tn={cfg['tn']} pyr={cfg['ratio']:.1f} "
                     f"mc={cfg['mc']} mr={cfg['mr']:.2f} atr={cfg['atr']:.1f}")
            print(f"\n  FULL {label}")
            analyze(trades, eq, dd, label)

    print(f"\n[V26] Done. {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
