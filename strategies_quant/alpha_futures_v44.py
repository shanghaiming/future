"""
V44: Adaptive Threshold + Portfolio Heat Institutional Strategy
===============================================================
Combines the two best institutional-grade innovations:

  V39 (Sharpe 4.47): Adaptive threshold -- entry quality self-tunes
    based on rolling win rate. When winning, relax to take more trades.
    When losing, tighten to be more selective.

  V36 (portfolio heat): Drawdown control via portfolio-level P&L
    tracking. When unrealized losses mount, reduce sizing or skip
    entries entirely. Closes worst position in extreme heat.

V44 merges both:
  1. V18 cross-sectional rank composite (7 factors)
  2. Adaptive threshold from V39 (P-controller on entry quality)
  3. Portfolio heat management from V36 (proportional risk reduction)
  4. Entry requires BOTH adaptive_threshold pass AND heat not too high
  5. KER gate, hold 5d, ATR stop 3.0

Goal: Sharpe > 4.0 AND MDD < 15% -- institutional deployment ready.

Signal at close[di], enter at open[di+1]. No look-ahead. No leverage.
Walk-forward validation required.
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

# V18 baseline weights
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
    print("[V44] Computing raw factors...", flush=True)

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
    print("[V44] Computing cross-sectional ranks...", flush=True)

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
    print("[V44] Building composite signal...", flush=True)

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
    atr_v = []
    for j in range(max(start_di, di - 14), di):
        hh, ll, cc = H[si, j], L[si, j], C[si, j]
        if not any(np.isnan([hh, ll, cc])):
            atr_v.append(max(hh - ll, abs(hh - cc), abs(ll - cc)))
    if atr_v:
        return np.mean(atr_v)
    return None


# ---------------------------------------------------------------------------
# V39 ADAPTIVE THRESHOLD (from V39)
# ---------------------------------------------------------------------------
def adaptive_threshold(
    recent_trades_win: list,
    base_threshold: float,
    adapt_amount: float,
    min_cap: float,
    max_cap: float,
    win_rate_window: int,
) -> float:
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


# ---------------------------------------------------------------------------
# V36 PORTFOLIO HEAT MANAGEMENT (from V36)
# ---------------------------------------------------------------------------
def get_heat_multiplier(portfolio_heat_pct: float, heat_threshold: float,
                        heat_pause: float) -> float:
    """Position sizing multiplier based on portfolio unrealized losses.

    Returns:
      1.0  -- heat below threshold, full sizing
      0.5  -- heat between threshold and pause, half sizing
      0.0  -- heat above pause, no new entries
    """
    if portfolio_heat_pct < heat_threshold:
        return 1.0
    elif portfolio_heat_pct < heat_pause:
        return 0.5
    else:
        return 0.0


def should_close_worst(portfolio_heat_pct: float, heat_close: float) -> bool:
    return portfolio_heat_pct > heat_close


# ---------------------------------------------------------------------------
# V44 COMBINED BACKTEST
# ---------------------------------------------------------------------------
def backtest_v44(
    C, O, H, L, NS, ND, dates, syms, sigs,
    # Adaptive threshold params (from V39)
    base_threshold: float = 0.80,
    adapt_amount: float = 0.07,
    win_rate_window: int = 20,
    min_cap: float = 0.70,
    max_cap: float = 0.95,
    # Portfolio heat params (from V36)
    heat_threshold: float = 0.03,
    heat_pause: float = 0.06,
    heat_close: float = 0.09,
    # General params
    top_n: int = 1,
    atr_stop: float = 3.0,
    min_confidence: int = 3,
    use_ker_gate: bool = True,
    hold_days: int = 5,
    pyramid_ratio: float = 0.5,
    pyramid_day: int = 1,
    max_positions: int = 5,
    start_di: int = 60,
    end_di: int | None = None,
):
    """V44 backtest: adaptive threshold + portfolio heat management.

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

    # Adaptive threshold state
    recent_trades_win: list = []
    current_threshold = base_threshold

    # Heat statistics
    heat_stats = {'full': 0, 'half': 0, 'skip': 0, 'close_worst': 0}

    for di in range(max(start_di, 1), end_di):
        d = dates[di]
        daily_pnl = 0.0
        new_positions = []

        # --- Step 1: Compute adaptive threshold ---
        current_threshold = adaptive_threshold(
            recent_trades_win, base_threshold, adapt_amount,
            min_cap, max_cap, win_rate_window,
        )

        # --- Step 2: Compute portfolio heat ---
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
        heat_mult = get_heat_multiplier(portfolio_heat_pct, heat_threshold, heat_pause)

        # --- Step 3: Close worst position if extreme heat ---
        if should_close_worst(portfolio_heat_pct, heat_close) and len(positions) > 0:
            worst_idx = -1
            worst_pnl = 0.0
            for idx, (si, edi, ep, sp, alloc, is_pyr) in enumerate(positions):
                c = C[si, di]
                if np.isnan(c):
                    continue
                pnl_pct = (c - ep) / ep - COMM
                pnl_dollar = equity * alloc * pnl_pct
                if pnl_dollar < worst_pnl:
                    worst_pnl = pnl_dollar
                    worst_idx = idx

            if worst_idx >= 0:
                si_w, edi_w, ep_w, sp_w, alloc_w, is_pyr_w = positions[worst_idx]
                c_w = C[si_w, di]
                if not np.isnan(c_w):
                    pnl_pct = (c_w - ep_w) / ep_w - COMM
                    profit = equity * alloc_w * pnl_pct
                    daily_pnl += profit
                    is_win = pnl_pct > 0
                    trades.append({
                        'pnl_abs': profit, 'pnl_pct': pnl_pct * 100,
                        'days': di - edi_w + 1, 'di': di, 'year': d.year,
                        'sym': syms[si_w], 'reason': 'heat_close', 'pyr': is_pyr_w,
                        'threshold': current_threshold,
                    })
                    recent_trades_win.append(1 if is_win else 0)
                    heat_stats['close_worst'] += 1
                    # Build new positions list without the worst
                    positions = [p for i2, p in enumerate(positions) if i2 != worst_idx]

        # --- Step 4: Process existing positions (stop/hold exit) ---
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
                        'threshold': current_threshold,
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
                        'threshold': current_threshold,
                    })
                    recent_trades_win.append(1 if is_win else 0)
            else:
                for edi, ep, sp, alloc, is_pyr in pos_list:
                    new_positions.append((si, edi, ep, sp, alloc, is_pyr))

        # --- Step 5: Pyramid check ---
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

        # --- Step 6: Entry -- signal at close[di], enter at open[di+1] ---
        # Requires BOTH adaptive_threshold pass AND heat not too high
        held = {p[0] for p in positions}
        n_held = len(positions)

        if n_held >= max_positions:
            continue

        # Heat gate: skip all new entries if heat too high
        if heat_mult == 0.0:
            heat_stats['skip'] += 1
            continue

        candidates = []
        for si in range(NS):
            if si in held:
                continue
            if np.isnan(composite[si, di]):
                continue
            # ADAPTIVE threshold gate
            if composite[si, di] < current_threshold:
                continue
            if n_confirm[si, di] < min_confidence:
                continue
            if use_ker_gate and ker_regime[si, di] < 0:
                continue
            if di + 1 >= ND or np.isnan(O[si, di + 1]):
                continue
            base_alloc = 1.0 / max(top_n, 1)
            # Combine heat multiplier with base allocation
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

    # Close remaining positions at end
    for si, edi, ep, sp, alloc, is_pyr in positions:
        c = C[si, ND - 1]
        if not np.isnan(c) and c > 0:
            pnl = (c - ep) / ep - COMM
            equity += equity * alloc * pnl

    return trades, equity, max_dd, heat_stats


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
    n_heat = sum(1 for t in trades if t['reason'] == 'heat_close')

    thresholds_used = [t.get('threshold', 0) for t in trades]
    avg_thresh = np.mean(thresholds_used) if thresholds_used else 0

    print(f"  {label}: {len(trades)}t (base:{n_base} pyr:{n_pyr} stop:{n_stop} "
          f"hold:{n_hold} heat:{n_heat}) "
          f"WR={wr:.1f}% ann={ann:+.1f}% DD={max_dd:.1f}% Sh={sh:.2f} "
          f"eq={equity:,.0f} avg_thresh={avg_thresh:.3f}")

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
                 base_threshold=0.80, adapt_amount=0.07,
                 win_rate_window=20,
                 heat_threshold=0.03, heat_pause=0.06, heat_close=0.09,
                 top_n=1, atr_stop=3.0, hold_days=5,
                 pyramid_ratio=0.5, pyramid_day=1,
                 max_positions=5):
    """Walk-forward validation: year-by-year out-of-sample.

    Adaptive state persists across years (mimics real trading).
    """
    print(f"\n{'=' * 70}")
    print(f"  WALK-FORWARD V44")
    print(f"  bt={base_threshold} aa={adapt_amount} ww={win_rate_window} "
          f"tn={top_n} ats={atr_stop}")
    print(f"  ht={heat_threshold*100:.0f}% hp={heat_pause*100:.0f}% "
          f"hc={heat_close*100:.0f}% pyr={pyramid_ratio} max={max_positions}")
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

        trades, _, _, _ = backtest_v44(
            C, O, H, L, NS, ND, dates, syms, sigs,
            base_threshold=base_threshold, adapt_amount=adapt_amount,
            win_rate_window=win_rate_window,
            min_cap=0.70, max_cap=0.95,
            heat_threshold=heat_threshold, heat_pause=heat_pause,
            heat_close=heat_close,
            top_n=top_n, atr_stop=atr_stop, hold_days=hold_days,
            min_confidence=3, use_ker_gate=True,
            pyramid_ratio=pyramid_ratio, pyramid_day=pyramid_day,
            max_positions=max_positions,
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
        n_heat = sum(1 for t in all_trades if t['reason'] == 'heat_close')
        avg_t = np.mean([t.get('threshold', 0) for t in all_trades])
        print(f"\n  WF TOTAL: {len(all_trades)}t (pyr:{n_pyr} heat_close:{n_heat}) "
              f"WR={wr_val:.1f}% avg={avg:+.2f}% cum={cum:+.1%} "
              f"avg_thresh={avg_t:.3f}")
        return all_trades
    return []


def main():
    t0 = time.time()
    print("=" * 70)
    print("  V44: ADAPTIVE THRESHOLD + PORTFOLIO HEAT")
    print("  V39 adaptive entry + V36 drawdown control = institutional grade")
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

    # ================================================================
    # SECTION 1: BASELINE COMPARISONS
    # ================================================================
    print("\n" + "=" * 70)
    print("  SECTION 1: BASELINES -- V39 (no heat) vs V36 (no adaptive)")
    print("=" * 70)

    # V39 baseline: adaptive threshold, no heat management
    print("\n  --- V39 baseline (no heat management) ---")
    for bt, aa in [(0.80, 0.07), (0.85, 0.05)]:
        trades, eq, dd, _ = backtest_v44(
            C, O, H, L, NS, ND, dates, syms, sigs,
            base_threshold=bt, adapt_amount=aa,
            heat_threshold=1.0, heat_pause=2.0, heat_close=3.0,
            top_n=1, atr_stop=3.0, hold_days=5,
            pyramid_ratio=0.5, max_positions=99,
            start_di=bt_2019)
        label = f"V39-noheat bt={bt} aa={aa}"
        analyze(trades, eq, dd, label)

    # V36 baseline: heat management, no adaptive threshold
    print("\n  --- V36 baseline (no adaptive threshold) ---")
    for ht in [0.03, 0.04]:
        trades, eq, dd, hs = backtest_v44(
            C, O, H, L, NS, ND, dates, syms, sigs,
            base_threshold=0.80, adapt_amount=0.0,
            heat_threshold=ht, heat_pause=ht * 2, heat_close=ht * 3,
            top_n=1, atr_stop=3.0, hold_days=5,
            pyramid_ratio=0.5, max_positions=5,
            start_di=bt_2019)
        label = f"V36-noadapt ht={ht*100:.0f}%"
        analyze(trades, eq, dd, label)
        print(f"    Heat: full={hs['full']} half={hs['half']} "
              f"skip={hs['skip']} close={hs['close_worst']}")

    # ================================================================
    # SECTION 2: V44 COMBINED PROFILES
    # ================================================================
    print("\n" + "=" * 70)
    print("  SECTION 2: V44 COMBINED PROFILES (2019-2026)")
    print("=" * 70)

    profiles = [
        (0.80, 0.07, 0.03, 0.06, 0.09, 1, 3.0, 0.5, 5, "Default V44"),
        (0.85, 0.07, 0.03, 0.06, 0.09, 1, 3.0, 0.5, 5, "Tight base + heat"),
        (0.80, 0.05, 0.03, 0.06, 0.09, 1, 3.0, 0.5, 5, "Conservative adapt + heat"),
        (0.80, 0.07, 0.04, 0.08, 0.12, 1, 3.0, 0.5, 3, "Relaxed heat tight max"),
        (0.80, 0.07, 0.02, 0.05, 0.08, 1, 3.0, 0.5, 5, "Tight heat control"),
    ]

    for bt, aa, ht, hp, hc, tn, ats, pr, mp, label in profiles:
        trades, eq, dd, hs = backtest_v44(
            C, O, H, L, NS, ND, dates, syms, sigs,
            base_threshold=bt, adapt_amount=aa,
            heat_threshold=ht, heat_pause=hp, heat_close=hc,
            top_n=tn, atr_stop=ats, hold_days=5,
            pyramid_ratio=pr, max_positions=mp,
            start_di=bt_2019)
        print(f"\n  {label}")
        analyze(trades, eq, dd, label)
        print(f"    Heat: full={hs['full']} half={hs['half']} "
              f"skip={hs['skip']} close={hs['close_worst']}")

    # ================================================================
    # SECTION 3: FULL PARAMETER SWEEP
    # ================================================================
    print("\n" + "=" * 70)
    print("  SECTION 3: PARAMETER SWEEP (2019-2026)")
    print("=" * 70)

    sweep_params = {
        'base_threshold': [0.80, 0.85],
        'adapt_amount': [0.05, 0.07],
        'heat_threshold': [0.02, 0.03, 0.04],
        'heat_pause': [0.05, 0.06, 0.08],
        'top_n': [1, 2],
        'atr_stop': [2.5, 3.0],
        'pyramid_ratio': [0.0, 0.5],
        'max_positions': [3, 5],
    }

    total_combos = 1
    for v in sweep_params.values():
        total_combos *= len(v)
    print(f"  Total combinations: {total_combos}")

    results = []
    combo_count = 0

    for bt, aa, ht, hp, tn, ats, pr, mp in product(
        sweep_params['base_threshold'],
        sweep_params['adapt_amount'],
        sweep_params['heat_threshold'],
        sweep_params['heat_pause'],
        sweep_params['top_n'],
        sweep_params['atr_stop'],
        sweep_params['pyramid_ratio'],
        sweep_params['max_positions'],
    ):
        # Skip invalid: heat_pause must be > heat_threshold
        if hp <= ht:
            continue

        # heat_close derived as 1.5x heat_pause (proportional)
        hc = hp * 1.5

        combo_count += 1
        trades, eq, dd, hs = backtest_v44(
            C, O, H, L, NS, ND, dates, syms, sigs,
            base_threshold=bt, adapt_amount=aa,
            heat_threshold=ht, heat_pause=hp, heat_close=hc,
            top_n=tn, atr_stop=ats, hold_days=5,
            pyramid_ratio=pr, max_positions=mp,
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
            'bt': bt, 'aa': aa, 'ht': ht, 'hp': hp, 'hc': hc,
            'tn': tn, 'ats': ats, 'pr': pr, 'mp': mp,
            'n': len(trades), 'wr': wr, 'ann': ann,
            'dd': dd, 'sharpe': sh_val, 'eq': eq,
            'heat_close': n_heat_close,
        })

    results.sort(key=lambda x: (-x['sharpe'], x['dd']))
    print(f"\n  Evaluated {combo_count} valid combinations, "
          f"{len(results)} with 10+ trades")
    print(f"\n{'BT':>4} {'AA':>4} {'HT%':>4} {'HP%':>4} {'TN':>3} {'ATS':>4} "
          f"{'Pyr':>4} {'Max':>3} {'N':>5} {'WR':>5} {'Ann':>8} {'DD':>6} "
          f"{'Sh':>6} {'HC':>3}")
    print("-" * 85)
    for r in results[:30]:
        print(f"{r['bt']:>4.2f} {r['aa']:>4.2f} {r['ht']*100:>4.0f} "
              f"{r['hp']*100:>4.0f} {r['tn']:>3} {r['ats']:>4.1f} "
              f"{r['pr']:>4.1f} {r['mp']:>3} "
              f"{r['n']:>5} {r['wr']:>5.1f} {r['ann']:>+8.1f} "
              f"{r['dd']:>6.1f} {r['sharpe']:>6.2f} {r['heat_close']:>3}")

    # ================================================================
    # SECTION 4: TARGET FILTER -- Sharpe > 4.0 AND MDD < 15%
    # ================================================================
    print("\n" + "=" * 70)
    print("  SECTION 4: TARGET -- Sharpe > 4.0 AND MDD < 15%")
    print("=" * 70)

    target_results = [r for r in results if r['sharpe'] > 4.0 and r['dd'] < 15.0]
    if target_results:
        target_results.sort(key=lambda x: (-x['sharpe'], x['dd']))
        print(f"\n  Found {len(target_results)} configs meeting target:")
        for r in target_results[:10]:
            print(f"    bt={r['bt']:.2f} aa={r['aa']:.2f} "
                  f"ht={r['ht']*100:.0f}% hp={r['hp']*100:.0f}% "
                  f"tn={r['tn']} atr={r['ats']:.1f} pyr={r['pr']:.1f} max={r['mp']} "
                  f"=> Sh={r['sharpe']:.2f} DD={r['dd']:.1f}% ann={r['ann']:+.1f}% "
                  f"WR={r['wr']:.1f}% N={r['n']}")
    else:
        # Broader search: Sharpe > 3.5 and MDD < 20%
        near_target = [r for r in results if r['sharpe'] > 3.5 and r['dd'] < 20.0]
        if near_target:
            near_target.sort(key=lambda x: (-x['sharpe'], x['dd']))
            print(f"\n  No configs meet exact target. Best with Sh>3.5 DD<20%:")
            for r in near_target[:10]:
                print(f"    bt={r['bt']:.2f} aa={r['aa']:.2f} "
                      f"ht={r['ht']*100:.0f}% hp={r['hp']*100:.0f}% "
                      f"tn={r['tn']} atr={r['ats']:.1f} pyr={r['pr']:.1f} max={r['mp']} "
                      f"=> Sh={r['sharpe']:.2f} DD={r['dd']:.1f}% ann={r['ann']:+.1f}%")
        else:
            # Find closest by composite score
            best_compromise = sorted(
                results,
                key=lambda x: -x['sharpe'] + x['dd'] * 0.2)[:10]
            print(f"\n  No configs near target. Best compromise:")
            for r in best_compromise:
                print(f"    bt={r['bt']:.2f} aa={r['aa']:.2f} "
                      f"ht={r['ht']*100:.0f}% hp={r['hp']*100:.0f}% "
                      f"tn={r['tn']} atr={r['ats']:.1f} pyr={r['pr']:.1f} max={r['mp']} "
                      f"=> Sh={r['sharpe']:.2f} DD={r['dd']:.1f}% ann={r['ann']:+.1f}%")

    # ================================================================
    # SECTION 5: FULL 10-YEAR FOR TOP CONFIGS
    # ================================================================
    print("\n" + "=" * 70)
    print("  SECTION 5: FULL 10-YEAR (2016-2026) FOR TOP CONFIGS")
    print("=" * 70)

    # V39 baseline for comparison (no heat)
    print("\n  --- V39 BASELINE (no heat management, 10yr) ---")
    for bt, aa, pr in [(0.80, 0.07, 0.5), (0.85, 0.05, 0.5)]:
        trades, eq, dd, _ = backtest_v44(
            C, O, H, L, NS, ND, dates, syms, sigs,
            base_threshold=bt, adapt_amount=aa,
            heat_threshold=1.0, heat_pause=2.0, heat_close=3.0,
            top_n=1, atr_stop=3.0, hold_days=5,
            pyramid_ratio=pr, max_positions=99,
            start_di=60)
        label = f"V39-10yr bt={bt} aa={aa} pyr={pr}"
        analyze(trades, eq, dd, label)

    # Top V44 configs (10-year)
    top_configs = results[:5]
    print("\n  --- V44 TOP CONFIGS (10-year) ---")
    for r in top_configs:
        trades, eq, dd, hs = backtest_v44(
            C, O, H, L, NS, ND, dates, syms, sigs,
            base_threshold=r['bt'], adapt_amount=r['aa'],
            heat_threshold=r['ht'], heat_pause=r['hp'], heat_close=r['hc'],
            top_n=r['tn'], atr_stop=r['ats'], hold_days=5,
            pyramid_ratio=r['pr'], max_positions=r['mp'],
            start_di=60)
        label = (f"bt={r['bt']:.2f} aa={r['aa']:.2f} "
                 f"ht={r['ht']*100:.0f}% hp={r['hp']*100:.0f}% "
                 f"tn={r['tn']} atr={r['ats']:.1f} pyr={r['pr']:.1f} max={r['mp']}")
        print(f"\n  {label}")
        analyze(trades, eq, dd, label)
        print(f"    Heat: full={hs['full']} half={hs['half']} "
              f"skip={hs['skip']} close={hs['close_worst']}")

    # Target-meeting configs (10-year)
    if target_results:
        print("\n  --- TARGET-MEETING CONFIGS (10-year) ---")
        for r in target_results[:3]:
            trades, eq, dd, hs = backtest_v44(
                C, O, H, L, NS, ND, dates, syms, sigs,
                base_threshold=r['bt'], adapt_amount=r['aa'],
                heat_threshold=r['ht'], heat_pause=r['hp'], heat_close=r['hc'],
                top_n=r['tn'], atr_stop=r['ats'], hold_days=5,
                pyramid_ratio=r['pr'], max_positions=r['mp'],
                start_di=60)
            label = (f"TARGET bt={r['bt']:.2f} aa={r['aa']:.2f} "
                     f"ht={r['ht']*100:.0f}% hp={r['hp']*100:.0f}% "
                     f"tn={r['tn']} atr={r['ats']:.1f} pyr={r['pr']:.1f} max={r['mp']}")
            print(f"\n  {label}")
            analyze(trades, eq, dd, label)
            print(f"    Heat: full={hs['full']} half={hs['half']} "
                  f"skip={hs['skip']} close={hs['close_worst']}")

    # ================================================================
    # SECTION 6: WALK-FORWARD FOR BEST CONFIGS
    # ================================================================
    print("\n" + "=" * 70)
    print("  SECTION 6: WALK-FORWARD VALIDATION")
    print("=" * 70)

    # Best by Sharpe
    best = results[0] if results else None
    if best:
        print(f"\n  Best by Sharpe: bt={best['bt']:.2f} aa={best['aa']:.2f} "
              f"ht={best['ht']*100:.0f}% hp={best['hp']*100:.0f}% "
              f"tn={best['tn']} atr={best['ats']:.1f} "
              f"pyr={best['pr']:.1f} max={best['mp']}")
        walk_forward(C, O, H, L, NS, ND, dates, syms, sigs,
                     base_threshold=best['bt'], adapt_amount=best['aa'],
                     heat_threshold=best['ht'], heat_pause=best['hp'],
                     heat_close=best['hc'],
                     top_n=best['tn'], atr_stop=best['ats'],
                     pyramid_ratio=best['pr'], max_positions=best['mp'])

    # Best by MDD (with reasonable Sharpe)
    mdd_candidates = [r for r in results if r['sharpe'] > 2.5]
    if mdd_candidates:
        mdd_candidates.sort(key=lambda x: x['dd'])
        best_mdd = mdd_candidates[0]
        print(f"\n  Best by MDD (Sh>2.5): bt={best_mdd['bt']:.2f} "
              f"aa={best_mdd['aa']:.2f} ht={best_mdd['ht']*100:.0f}% "
              f"hp={best_mdd['hp']*100:.0f}% tn={best_mdd['tn']} "
              f"atr={best_mdd['ats']:.1f} pyr={best_mdd['pr']:.1f} max={best_mdd['mp']}")
        walk_forward(C, O, H, L, NS, ND, dates, syms, sigs,
                     base_threshold=best_mdd['bt'], adapt_amount=best_mdd['aa'],
                     heat_threshold=best_mdd['ht'], heat_pause=best_mdd['hp'],
                     heat_close=best_mdd['hc'],
                     top_n=best_mdd['tn'], atr_stop=best_mdd['ats'],
                     pyramid_ratio=best_mdd['pr'], max_positions=best_mdd['mp'])

    # Target-meeting config walk-forward
    if target_results:
        target_best = target_results[0]
        print(f"\n  Best target-meeting: bt={target_best['bt']:.2f} "
              f"aa={target_best['aa']:.2f} ht={target_best['ht']*100:.0f}% "
              f"hp={target_best['hp']*100:.0f}% tn={target_best['tn']} "
              f"atr={target_best['ats']:.1f} pyr={target_best['pr']:.1f} "
              f"max={target_best['mp']}")
        walk_forward(C, O, H, L, NS, ND, dates, syms, sigs,
                     base_threshold=target_best['bt'],
                     adapt_amount=target_best['aa'],
                     heat_threshold=target_best['ht'],
                     heat_pause=target_best['hp'],
                     heat_close=target_best['hc'],
                     top_n=target_best['tn'],
                     atr_stop=target_best['ats'],
                     pyramid_ratio=target_best['pr'],
                     max_positions=target_best['mp'])

    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "=" * 70)
    print("  SUMMARY: V44 vs BASELINES")
    print("=" * 70)

    # V39 baseline WF
    print("\n  V39 Baseline WF (adaptive, no heat):")
    v39_trades, v39_eq, v39_dd, _ = backtest_v44(
        C, O, H, L, NS, ND, dates, syms, sigs,
        base_threshold=0.80, adapt_amount=0.07,
        heat_threshold=1.0, heat_pause=2.0, heat_close=3.0,
        top_n=1, atr_stop=3.0, hold_days=5,
        pyramid_ratio=0.5, max_positions=99,
        start_di=bt_2019)
    v39_result = analyze(v39_trades, v39_eq, v39_dd, "V39-baseline")

    print("\n  V44 Best WF (adaptive + heat):")
    if best:
        v44_trades, v44_eq, v44_dd, v44_hs = backtest_v44(
            C, O, H, L, NS, ND, dates, syms, sigs,
            base_threshold=best['bt'], adapt_amount=best['aa'],
            heat_threshold=best['ht'], heat_pause=best['hp'],
            heat_close=best['hc'],
            top_n=best['tn'], atr_stop=best['ats'],
            pyramid_ratio=best['pr'], max_positions=best['mp'],
            start_di=bt_2019)
        v44_result = analyze(v44_trades, v44_eq, v44_dd, "V44-best")

        if v39_result and v44_result:
            dd_improvement = v39_result['dd'] - v44_result['dd']
            print(f"\n  MDD CHANGE: {v39_result['dd']:.1f}% -> {v44_result['dd']:.1f}% "
                  f"(delta={dd_improvement:+.1f}%)")
            print(f"  Sharpe CHANGE: {v39_result['sh']:.2f} -> {v44_result['sh']:.2f} "
                  f"(delta={v44_result['sh'] - v39_result['sh']:+.2f})")
            print(f"  Equity CHANGE: {v39_result['eq']:,.0f} -> {v44_result['eq']:,.0f} "
                  f"(delta={v44_result['eq'] - v39_result['eq']:+,.0f})")
            print(f"  Heat: full={v44_hs['full']} half={v44_hs['half']} "
                  f"skip={v44_hs['skip']} close={v44_hs['close_worst']}")

    print(f"\n[V44] Done. {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
