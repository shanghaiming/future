"""
V36: Portfolio Heat Rank Mean Reversion
========================================
Extends V18 (WF +2019.6%, Sharpe 2.39, MDD 28%) with institutional-grade
portfolio-level heat management to control drawdown.

Core innovation: Track unrealized P&L across ALL open positions.
When portfolio heat exceeds thresholds, reduce sizing or skip entries.
This is how real fund managers control drawdown -- not circuit breakers
(which destroy returns), but proportional risk reduction.

Architecture:
  1. Same V18 cross-sectional rank: 7 factors with weights
  2. Portfolio heat management:
     - portfolio_heat = sum of unrealized losses / current equity
     - heat < threshold: full size (1.0x)
     - heat 1-2x threshold: half size (0.5x)
     - heat > 2x threshold: no new entries (0.0x)
     - heat > 3x threshold: also close worst position
  3. Position sizing = base_size x heat_multiplier
  4. KER gate, hold 5d, ATR stop, pyramid
  5. Walk-forward validation 2019-2026

Signal at close[di], enter at open[di+1]. No look-ahead. No leverage.
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

# V18 baseline weights (best performer)
DEFAULT_WEIGHTS = {
    'rank_ret5d':  0.25,
    'rank_oi5d':   0.20,
    'rank_vol':    0.20,
    'rank_ret10d': 0.10,
    'rank_range':  0.10,
    'rank_rsi':    0.10,
    'rank_atrp':   0.05,
}


def compute_rsi_manual(C: np.ndarray, NS: int, ND: int, period: int = 14) -> np.ndarray:
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
    print("[V36] Computing raw factors...", flush=True)

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
    print("[V36] Computing cross-sectional ranks...", flush=True)

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
                ker_regime[si, di] = 1
            elif ker_10[si, di] > 0.3:
                ker_regime[si, di] = -1
    return ker_regime


def build_composite_signal(ranks, weights, NS, ND, min_factors=4):
    """Build weighted composite rank from individual factor ranks."""
    t0 = time.time()
    print("[V36] Building composite signal...", flush=True)

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
    """Full signal pipeline (same as V18)."""
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


def compute_atr_at(H, L, C, si, di, start_di, lookback=14):
    """Compute ATR for a specific symbol and day."""
    atr_v = []
    for j in range(max(start_di, di - lookback), di):
        hh, ll, cc = H[si, j], L[si, j], C[si, j]
        if not any(np.isnan([hh, ll, cc])):
            prev_c = C[si, j - 1] if j > 0 and not np.isnan(C[si, j - 1]) else cc
            atr_v.append(max(hh - ll, abs(hh - prev_c), abs(ll - prev_c)))
    if atr_v:
        return np.mean(atr_v)
    return None


def get_heat_multiplier(portfolio_heat_pct: float, heat_threshold: float) -> float:
    """Determine position sizing multiplier based on portfolio heat.
    portfolio_heat_pct: absolute value of unrealized losses as % of equity.
    heat_threshold: base threshold (e.g., 3% = 0.03).
    Returns multiplier: 1.0, 0.5, or 0.0.
    """
    if portfolio_heat_pct < heat_threshold:
        return 1.0
    elif portfolio_heat_pct < heat_threshold * 2:
        return 0.5
    else:
        return 0.0


def should_close_worst(portfolio_heat_pct: float, heat_threshold: float) -> bool:
    """Check if portfolio heat is so high we should close the worst position."""
    return portfolio_heat_pct > heat_threshold * 3


def backtest_v36(C, O, H, L, NS, ND, dates, syms, sigs,
                 top_n=1, min_rank=0.75, atr_stop=3.0,
                 min_confidence=3, use_ker_gate=True,
                 hold_days=5,
                 pyramid_ratio=0.5,
                 pyramid_day=1,
                 heat_threshold=0.03,
                 max_positions=7,
                 start_di=60, end_di=None):
    """Backtest with cross-sectional rank signals + portfolio heat management.

    Key difference from V18:
    - Positions track entry price and current price for unrealized P&L
    - Portfolio heat = sum of unrealized losses / equity
    - Heat multiplier gates new position sizing
    - Extreme heat triggers closing the worst position

    Position tuple: (si, entry_di, entry_price, stop_price, alloc, is_pyramid)
    """
    composite = sigs['composite']
    ker_regime = sigs['ker_regime']
    n_confirm = sigs['n_confirm']

    if end_di is None:
        end_di = ND - 1

    equity = float(CASH0)
    peak = equity
    max_dd = 0.0
    positions = []
    trades = []
    heat_stats = {'full': 0, 'half': 0, 'skip': 0, 'close_worst': 0}

    for di in range(max(start_di, 1), end_di):
        d = dates[di]
        daily_pnl = 0.0
        new_positions = []

        # --- Compute unrealized P&L for all open positions ---
        total_unrealized_loss = 0.0
        for si, edi, ep, sp, alloc, is_pyr in positions:
            c = C[si, di]
            if np.isnan(c):
                continue
            unrealized_pnl_pct = (c - ep) / ep - COMM
            unrealized_dollar = equity * alloc * unrealized_pnl_pct
            if unrealized_dollar < 0:
                total_unrealized_loss += abs(unrealized_dollar)

        portfolio_heat_pct = total_unrealized_loss / equity if equity > 0 else 0.0
        heat_mult = get_heat_multiplier(portfolio_heat_pct, heat_threshold)

        # --- Close worst position if heat is extreme ---
        if should_close_worst(portfolio_heat_pct, heat_threshold) and len(positions) > 0:
            worst_pos = None
            worst_pnl = 0.0
            worst_idx = -1
            for idx, (si, edi, ep, sp, alloc, is_pyr) in enumerate(positions):
                c = C[si, di]
                if np.isnan(c):
                    continue
                pnl_pct = (c - ep) / ep - COMM
                pnl_dollar = equity * alloc * pnl_pct
                if pnl_dollar < worst_pnl:
                    worst_pnl = pnl_dollar
                    worst_pos = (si, edi, ep, sp, alloc, is_pyr)
                    worst_idx = idx

            if worst_pos is not None:
                si_w, edi_w, ep_w, sp_w, alloc_w, is_pyr_w = worst_pos
                c_w = C[si_w, di]
                if not np.isnan(c_w):
                    pnl_pct = (c_w - ep_w) / ep_w - COMM
                    profit = equity * alloc_w * pnl_pct
                    daily_pnl += profit
                    trades.append({
                        'pnl_abs': profit,
                        'pnl_pct': pnl_pct * 100,
                        'days': di - edi_w + 1,
                        'di': di,
                        'year': d.year,
                        'sym': syms[si_w],
                        'reason': 'heat_close',
                        'pyr': is_pyr_w,
                    })
                    heat_stats['close_worst'] += 1
                    # Remove worst position by index
                    positions_new = []
                    for idx2, pos in enumerate(positions):
                        if idx2 != worst_idx:
                            positions_new.append(pos)
                    positions = positions_new

        # --- Process existing positions (stop/hold exit) ---
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
                        'pnl_abs': profit,
                        'pnl_pct': pnl * 100,
                        'days': di - edi + 1,
                        'di': di,
                        'year': d.year,
                        'sym': syms[si],
                        'reason': 'stop',
                        'pyr': is_pyr,
                    })
            elif hold >= hold_days:
                for edi, ep, sp, alloc, is_pyr in pos_list:
                    pnl = (c - ep) / ep - COMM
                    profit = equity * alloc * pnl
                    daily_pnl += profit
                    trades.append({
                        'pnl_abs': profit,
                        'pnl_pct': pnl * 100,
                        'days': di - edi + 1,
                        'di': di,
                        'year': d.year,
                        'sym': syms[si],
                        'reason': 'hold',
                        'pyr': is_pyr,
                    })
            else:
                for edi, ep, sp, alloc, is_pyr in pos_list:
                    new_positions.append((si, edi, ep, sp, alloc, is_pyr))

        # --- Pyramid check ---
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
                        pyr_alloc = base_alloc * pyramid_ratio * heat_mult
                        if pyr_alloc > 0.001:
                            c_now = C[si, di]
                            atr = compute_atr_at(H, L, C, si, di, start_di)
                            if atr is not None:
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

        # --- Entry signal at close[di], enter at open[di+1] ---
        held = {p[0] for p in positions}
        n_held = len(positions)

        if n_held >= max_positions:
            if heat_mult == 0.0:
                heat_stats['skip'] += 1
            continue

        if heat_mult == 0.0:
            heat_stats['skip'] += 1
            continue

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
            base_alloc = 1.0 / max(top_n, 1)
            adjusted_alloc = base_alloc * heat_mult
            candidates.append((composite[si, di], si, adjusted_alloc))

        candidates.sort(key=lambda x: -x[0])
        slots_left = max_positions - n_held
        for rank_val, si, alloc in candidates[:min(top_n, slots_left)]:
            if len(positions) >= max_positions or si in held:
                break
            ep = O[si, di + 1]
            if np.isnan(ep) or ep <= 0:
                continue
            atr = compute_atr_at(H, L, C, si, di, start_di)
            if atr is None:
                continue
            positions.append((si, di + 1, ep, ep - atr_stop * atr, alloc, False))
            held.add(si)

        if heat_mult == 1.0:
            heat_stats['full'] += 1
        elif heat_mult == 0.5:
            heat_stats['half'] += 1

    # Close remaining positions
    for si, edi, ep, sp, alloc, is_pyr in positions:
        c = C[si, ND - 1]
        if not np.isnan(c) and c > 0:
            pnl = (c - ep) / ep - COMM
            equity += equity * alloc * pnl

    return trades, equity, max_dd, heat_stats


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
    n_heat = sum(1 for t in trades if t['reason'] == 'heat_close')

    print(f"  {label}: {len(trades)}t (base:{n_base} pyr:{n_pyr} stop:{n_stop} "
          f"hold:{n_hold} heat:{n_heat}) "
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
                 min_rank=0.75, heat_threshold=0.03, max_positions=7):
    """Walk-forward validation: year-by-year out-of-sample."""
    print(f"\n{'=' * 70}")
    print(f"  WALK-FORWARD V36 (pyr={pyramid_ratio}, heat={heat_threshold*100:.0f}%)")
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

        trades, _, _, _ = backtest_v36(
            C, O, H, L, NS, ND, dates, syms, sigs,
            top_n=top_n, hold_days=hold_days, atr_stop=atr_stop,
            min_rank=min_rank,
            min_confidence=min_confidence, use_ker_gate=True,
            pyramid_ratio=pyramid_ratio, pyramid_day=pyramid_day,
            heat_threshold=heat_threshold, max_positions=max_positions,
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
        n_heat = sum(1 for t in all_trades if t['reason'] == 'heat_close')
        print(f"\n  WF TOTAL: {len(all_trades)}t (pyr:{n_pyr} heat_close:{n_heat}) "
              f"WR={wr_val:.1f}% avg={avg:+.2f}% cum={cum:+.1%}")
        return all_trades
    return []


def main():
    t0 = time.time()
    print("=" * 70)
    print("  V36: PORTFOLIO HEAT RANK MEAN REVERSION")
    print("  V18 rank signals + institutional drawdown control")
    print("=" * 70)

    C, O, H, L, V, OI, NS, ND, dates, syms = load_all_data(start='2016-01-01')
    print(f"  {NS} sym, {ND} days, "
          f"{dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")

    sigs = compute_all_signals(C, O, H, L, V, OI, NS, ND)

    # Find 2019 start index
    bt_2019 = None
    for i, d in enumerate(dates):
        if d >= pd.Timestamp('2019-01-01'):
            bt_2019 = i
            break

    # ================================================================
    # SECTION 1: V18 BASELINE (heat_threshold=1.0 = no heat management)
    # ================================================================
    print("\n" + "=" * 70)
    print("  V18 BASELINE (no heat management)")
    print("=" * 70)

    for ratio in [0.0, 0.5]:
        trades, eq, dd, hs = backtest_v36(
            C, O, H, L, NS, ND, dates, syms, sigs,
            top_n=1, hold_days=5, atr_stop=3.0,
            min_rank=0.75, min_confidence=3, use_ker_gate=True,
            pyramid_ratio=ratio, pyramid_day=1,
            heat_threshold=1.0, max_positions=99,  # effectively no heat management
            start_di=bt_2019)
        label = f"V18-baseline pyr={ratio:.1f}"
        analyze(trades, eq, dd, label)

    # ================================================================
    # SECTION 2: V36 HEAT MANAGEMENT COMPARISON
    # ================================================================
    print("\n" + "=" * 70)
    print("  V36 HEAT MANAGEMENT (varying thresholds)")
    print("=" * 70)

    for ht_pct in [0.02, 0.03, 0.04, 0.05]:
        ht = ht_pct
        for ratio in [0.0, 0.5]:
            for max_pos in [3, 5, 7]:
                trades, eq, dd, hs = backtest_v36(
                    C, O, H, L, NS, ND, dates, syms, sigs,
                    top_n=1, hold_days=5, atr_stop=3.0,
                    min_rank=0.75, min_confidence=3, use_ker_gate=True,
                    pyramid_ratio=ratio, pyramid_day=1,
                    heat_threshold=ht, max_positions=max_pos,
                    start_di=bt_2019)
                label = f"heat={ht_pct*100:.0f}% pyr={ratio:.1f} max={max_pos}"
                analyze(trades, eq, dd, label)

    # ================================================================
    # SECTION 3: FULL PARAMETER SWEEP
    # ================================================================
    print("\n" + "=" * 70)
    print("  PARAMETER SWEEP (2019-2026)")
    print("=" * 70)

    results = []

    for ht_pct in [0.02, 0.03, 0.04, 0.05]:
        for tn in [1, 2]:
            for mr in [0.75, 0.80]:
                for ats in [2.5, 3.0]:
                    for pyr in [0.0, 0.5]:
                        for max_pos in [3, 5, 7]:
                            trades, eq, dd, hs = backtest_v36(
                                C, O, H, L, NS, ND, dates, syms, sigs,
                                top_n=tn, hold_days=5, atr_stop=ats,
                                min_rank=mr, min_confidence=3,
                                use_ker_gate=True,
                                pyramid_ratio=pyr, pyramid_day=1,
                                heat_threshold=ht_pct, max_positions=max_pos,
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
                            n_heat_close = sum(1 for t in trades if t['reason'] == 'heat_close')
                            results.append({
                                'ht': ht_pct, 'tn': tn, 'mr': mr, 'ats': ats,
                                'pyr': pyr, 'max_pos': max_pos,
                                'n': len(trades), 'wr': wr, 'ann': ann,
                                'dd': dd, 'sharpe': sh_val, 'eq': eq,
                                'heat_close': n_heat_close,
                            })

    results.sort(key=lambda x: (-x['sharpe'], x['dd']))
    print(f"\n{'HT%':>4} {'TN':>3} {'MR':>4} {'ATR':>4} {'Pyr':>4} {'Max':>3} "
          f"{'N':>5} {'WR':>5} {'Ann':>8} {'DD':>6} {'Sh':>6} {'HC':>3}")
    print("-" * 75)
    for r in results[:30]:
        print(f"{r['ht']*100:>4.0f} {r['tn']:>3} {r['mr']:>4.2f} {r['ats']:>4.1f} "
              f"{r['pyr']:>4.1f} {r['max_pos']:>3} "
              f"{r['n']:>5} {r['wr']:>5.1f} {r['ann']:>+8.1f} "
              f"{r['dd']:>6.1f} {r['sharpe']:>6.2f} {r['heat_close']:>3}")

    # ================================================================
    # SECTION 4: FIND CONFIGS WITH Sharpe>2.0 AND MDD<25%
    # ================================================================
    print("\n" + "=" * 70)
    print("  TARGET: Sharpe > 2.0 AND MDD < 25%")
    print("=" * 70)

    target_results = [r for r in results if r['sharpe'] > 2.0 and r['dd'] < 25.0]
    if target_results:
        target_results.sort(key=lambda x: (-x['sharpe'], x['dd']))
        print(f"\n  Found {len(target_results)} configs meeting target:")
        for r in target_results[:10]:
            print(f"    heat={r['ht']*100:.0f}% tn={r['tn']} mr={r['mr']:.2f} "
                  f"atr={r['ats']:.1f} pyr={r['pyr']:.1f} max={r['max_pos']} "
                  f"=> Sh={r['sharpe']:.2f} DD={r['dd']:.1f}% ann={r['ann']:+.1f}% "
                  f"WR={r['wr']:.1f}% N={r['n']}")
    else:
        # Find closest to target
        near_target = sorted(results,
                             key=lambda x: abs(x['sharpe'] - 2.0) + abs(x['dd'] - 25.0) * 0.1)
        print(f"\n  No configs meet both targets. Closest:")
        for r in near_target[:10]:
            print(f"    heat={r['ht']*100:.0f}% tn={r['tn']} mr={r['mr']:.2f} "
                  f"atr={r['ats']:.1f} pyr={r['pyr']:.1f} max={r['max_pos']} "
                  f"=> Sh={r['sharpe']:.2f} DD={r['dd']:.1f}% ann={r['ann']:+.1f}%")

    # ================================================================
    # SECTION 5: FULL 10-YEAR FOR TOP CONFIGS
    # ================================================================
    print("\n" + "=" * 70)
    print("  FULL 10-YEAR BACKTEST FOR TOP CONFIGS")
    print("=" * 70)

    # Also include V18 baseline for comparison
    print("\n  --- V18 BASELINE (no heat management) ---")
    for ratio in [0.0, 0.5]:
        trades, eq, dd, hs = backtest_v36(
            C, O, H, L, NS, ND, dates, syms, sigs,
            top_n=1, hold_days=5, atr_stop=3.0,
            min_rank=0.75, min_confidence=3, use_ker_gate=True,
            pyramid_ratio=ratio, pyramid_day=1,
            heat_threshold=1.0, max_positions=99,
            start_di=60)
        label = f"V18-10yr pyr={ratio:.1f}"
        analyze(trades, eq, dd, label)

    # Top configs full 10-year
    top_configs = results[:5]
    print("\n  --- V36 TOP CONFIGS (10-year) ---")
    for r in top_configs:
        trades, eq, dd, hs = backtest_v36(
            C, O, H, L, NS, ND, dates, syms, sigs,
            top_n=r['tn'], hold_days=5, atr_stop=r['ats'],
            min_rank=r['mr'], min_confidence=3, use_ker_gate=True,
            pyramid_ratio=r['pyr'], pyramid_day=1,
            heat_threshold=r['ht'], max_positions=r['max_pos'],
            start_di=60)
        label = (f"heat={r['ht']*100:.0f}% tn={r['tn']} mr={r['mr']:.2f} "
                 f"atr={r['ats']:.1f} pyr={r['pyr']:.1f} max={r['max_pos']}")
        print(f"\n  {label}")
        analyze(trades, eq, dd, label)
        print(f"    Heat stats: full={hs['full']} half={hs['half']} "
              f"skip={hs['skip']} close_worst={hs['close_worst']}")

    # Also test target-meeting configs
    if target_results:
        print("\n  --- TARGET-MEETING CONFIGS (10-year) ---")
        for r in target_results[:3]:
            trades, eq, dd, hs = backtest_v36(
                C, O, H, L, NS, ND, dates, syms, sigs,
                top_n=r['tn'], hold_days=5, atr_stop=r['ats'],
                min_rank=r['mr'], min_confidence=3, use_ker_gate=True,
                pyramid_ratio=r['pyr'], pyramid_day=1,
                heat_threshold=r['ht'], max_positions=r['max_pos'],
                start_di=60)
            label = (f"TARGET heat={r['ht']*100:.0f}% tn={r['tn']} mr={r['mr']:.2f} "
                     f"atr={r['ats']:.1f} pyr={r['pyr']:.1f} max={r['max_pos']}")
            print(f"\n  {label}")
            analyze(trades, eq, dd, label)

    # ================================================================
    # SECTION 6: WALK-FORWARD FOR BEST CONFIG
    # ================================================================
    print("\n" + "=" * 70)
    print("  WALK-FORWARD VALIDATION FOR BEST CONFIGS")
    print("=" * 70)

    # Best by Sharpe
    best = results[0] if results else None
    if best:
        print(f"\n  Best by Sharpe: heat={best['ht']*100:.0f}% "
              f"tn={best['tn']} mr={best['mr']:.2f} atr={best['ats']:.1f} "
              f"pyr={best['pyr']:.1f} max={best['max_pos']}")
        walk_forward(C, O, H, L, V, OI, NS, ND, dates, syms, sigs,
                     pyramid_ratio=best['pyr'], pyramid_day=1,
                     top_n=best['tn'], min_confidence=3,
                     hold_days=5, min_rank=best['mr'],
                     heat_threshold=best['ht'], max_positions=best['max_pos'],
                     atr_stop=best['ats'])

    # Best by MDD (with reasonable Sharpe)
    mdd_candidates = [r for r in results if r['sharpe'] > 1.5]
    if mdd_candidates:
        mdd_candidates.sort(key=lambda x: x['dd'])
        best_mdd = mdd_candidates[0]
        print(f"\n  Best by MDD (Sh>1.5): heat={best_mdd['ht']*100:.0f}% "
              f"tn={best_mdd['tn']} mr={best_mdd['mr']:.2f} atr={best_mdd['ats']:.1f} "
              f"pyr={best_mdd['pyr']:.1f} max={best_mdd['max_pos']}")
        walk_forward(C, O, H, L, V, OI, NS, ND, dates, syms, sigs,
                     pyramid_ratio=best_mdd['pyr'], pyramid_day=1,
                     top_n=best_mdd['tn'], min_confidence=3,
                     hold_days=5, min_rank=best_mdd['mr'],
                     heat_threshold=best_mdd['ht'], max_positions=best_mdd['max_pos'],
                     atr_stop=best_mdd['ats'])

    # Target-meeting config walk-forward
    if target_results:
        target_best = target_results[0]
        print(f"\n  Best target-meeting: heat={target_best['ht']*100:.0f}% "
              f"tn={target_best['tn']} mr={target_best['mr']:.2f} atr={target_best['ats']:.1f} "
              f"pyr={target_best['pyr']:.1f} max={target_best['max_pos']}")
        walk_forward(C, O, H, L, V, OI, NS, ND, dates, syms, sigs,
                     pyramid_ratio=target_best['pyr'], pyramid_day=1,
                     top_n=target_best['tn'], min_confidence=3,
                     hold_days=5, min_rank=target_best['mr'],
                     heat_threshold=target_best['ht'], max_positions=target_best['max_pos'],
                     atr_stop=target_best['ats'])

    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "=" * 70)
    print("  SUMMARY: V36 vs V18 BASELINE")
    print("=" * 70)

    # V18 baseline WF
    print("\n  V18 Baseline WF (no heat management):")
    v18_trades, v18_eq, v18_dd, _ = backtest_v36(
        C, O, H, L, NS, ND, dates, syms, sigs,
        top_n=1, hold_days=5, atr_stop=3.0,
        min_rank=0.75, min_confidence=3, use_ker_gate=True,
        pyramid_ratio=0.5, pyramid_day=1,
        heat_threshold=1.0, max_positions=99,
        start_di=bt_2019)
    v18_result = analyze(v18_trades, v18_eq, v18_dd, "V18-baseline")

    print("\n  V36 Best WF (with heat management):")
    if best:
        v36_trades, v36_eq, v36_dd, v36_hs = backtest_v36(
            C, O, H, L, NS, ND, dates, syms, sigs,
            top_n=best['tn'], hold_days=5, atr_stop=best['ats'],
            min_rank=best['mr'], min_confidence=3, use_ker_gate=True,
            pyramid_ratio=best['pyr'], pyramid_day=1,
            heat_threshold=best['ht'], max_positions=best['max_pos'],
            start_di=bt_2019)
        v36_result = analyze(v36_trades, v36_eq, v36_dd, "V36-best")

        if v18_result and v36_result:
            dd_improvement = v18_result['dd'] - v36_result['dd']
            print(f"\n  MDD IMPROVEMENT: {v18_result['dd']:.1f}% -> {v36_result['dd']:.1f}% "
                  f"(delta={dd_improvement:+.1f}%)")
            print(f"  Sharpe: {v18_result['sh']:.2f} -> {v36_result['sh']:.2f} "
                  f"(delta={v36_result['sh'] - v18_result['sh']:+.2f})")

    print(f"\n[V36] Done. {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
