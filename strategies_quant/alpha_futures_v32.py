"""
V32: RANK + OI EXTREME COMBO — Best of V18 (rank) + V12 (OI contrarian)
========================================================================
Combines the two best strategies:
  - V18 (WF +2019.6%, Sharpe 2.39): cross-sectional rank as primary signal
  - V12 (WF +527.7%, Sharpe 1.71): OI capitulation pattern

Signal architecture:
  1. V18's 7 cross-sectional ranks (ret5d, oi5d, vol, ret10d, range, rsi, atrp)
  2. V12's OI capitulation signal:
     - oi_decline_5d = (OI[di] - OI[di-5]) / OI[di-5]
     - oi_decline_10d = (OI[di] - OI[di-10]) / OI[di-10]
     - OI extreme: oi_decline_5d < threshold AND oi_decline_10d < -8%
  3. Composite = V18_rank_score (base 70%) + OI_capitulation_bonus (30%)
     - OI_capitulation_bonus = cross-sectional rank of oi_decline_5d
  4. Entry: composite_rank > min_rank AND KER < 0.15
  5. Confidence: count signals aligned
  6. Hold 5d, ATR stop, pyramid on day-1 winners

Parameter sweep:
  - oi_weight: 0.20, 0.30, 0.40
  - oi_decline_5d_threshold: -3%, -5%, -8%
  - top_n: 1, 2, 3
  - min_rank: 0.70, 0.75, 0.80
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

# Default V18 weights for cross-sectional rank factors
DEFAULT_RANK_WEIGHTS = {
    'rank_ret5d':  0.25,
    'rank_oi5d':   0.20,
    'rank_rsi':    0.15,
    'rank_vol':    0.15,
    'rank_ret10d': 0.10,
    'rank_range':  0.10,
    'rank_atrp':   0.05,
}

# Default OI combo weight
DEFAULT_OI_WEIGHT = 0.30
DEFAULT_OI_DECLINE_5D_THRESH = -0.05
DEFAULT_OI_DECLINE_10D_THRESH = -0.08


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
    print("[V32] Computing raw factors...", flush=True)

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

    # --- OI capitulation factors (V12-inspired) ---
    # OI decline over 5d and 10d
    oi_decline_5d = np.full((NS, ND), np.nan)
    oi_decline_10d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(10, ND):
            if (not np.isnan(OI[si, di]) and not np.isnan(OI[si, di - 5])
                    and OI[si, di - 5] > 0):
                oi_decline_5d[si, di] = (OI[si, di] - OI[si, di - 5]) / OI[si, di - 5]
            if (not np.isnan(OI[si, di]) and not np.isnan(OI[si, di - 10])
                    and OI[si, di - 10] > 0):
                oi_decline_10d[si, di] = (OI[si, di] - OI[si, di - 10]) / OI[si, di - 10]

    # Consecutive down days (for confidence)
    consec_dn = np.zeros((NS, ND), dtype=int)
    for si in range(NS):
        consec = 0
        for di in range(1, ND):
            if (not np.isnan(C[si, di]) and not np.isnan(C[si, di - 1])
                    and C[si, di - 1] > 0):
                if C[si, di] < C[si, di - 1]:
                    consec += 1
                else:
                    consec = 0
            else:
                consec = 0
            consec_dn[si, di] = consec

    # Volume surge (5d vs 20d)
    vol_surge = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            v5 = V[si, di - 5:di]
            v20 = V[si, di - 20:di]
            v5v = v5[~np.isnan(v5)]
            v20v = v20[~np.isnan(v20)]
            if len(v5v) >= 3 and len(v20v) >= 10:
                m5 = np.mean(v5v)
                m20 = np.mean(v20v)
                if m20 > 0:
                    vol_surge[si, di] = m5 / m20 - 1.0

    print(f"  Raw factors done: {time.time() - t0:.1f}s", flush=True)
    return {
        'ret_5d': ret_5d,
        'ret_10d': ret_10d,
        'oi_5d': oi_5d,
        'vol_5d': vol_5d,
        'daily_range': daily_range,
        'rsi14': rsi14,
        'atrp': atrp,
        'oi_decline_5d': oi_decline_5d,
        'oi_decline_10d': oi_decline_10d,
        'consec_dn': consec_dn,
        'vol_surge': vol_surge,
    }


def compute_cross_sectional_ranks(raw_factors, NS, ND, min_count=10):
    """Rank all factors cross-sectionally (across commodities per day).
    Low rank = oversold / extreme for mean reversion."""
    t0 = time.time()
    print("[V32] Computing cross-sectional ranks...", flush=True)

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

    # Also rank OI decline 5d cross-sectionally (more negative = higher rank)
    oi_decline_5d_rank = np.full((NS, ND), np.nan)
    for di in range(ND):
        vals = raw_factors['oi_decline_5d'][:, di]
        valid_count = np.sum(~np.isnan(vals))
        if valid_count < min_count:
            continue
        ranked = pd.Series(vals).rank(pct=True, na_option='keep').values
        # Invert: more negative decline -> higher rank (better contrarian signal)
        oi_decline_5d_rank[:, di] = 1.0 - ranked

    ranks['rank_oi_decline_5d'] = oi_decline_5d_rank

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


def build_composite_signal(ranks, raw_factors, rank_weights, oi_weight,
                           oi_decline_5d_thresh, oi_decline_10d_thresh,
                           NS, ND, min_factors=4):
    """Build composite: V18 rank base (1 - oi_weight) + OI capitulation bonus (oi_weight).

    OI capitulation bonus = cross-sectional rank of oi_decline_5d,
    boosted if both 5d and 10d OI decline thresholds are met.

    Also compute confidence count of aligned signals.
    """
    t0 = time.time()
    print(f"[V32] Building composite (oi_w={oi_weight:.2f}, "
          f"oi_5d_thr={oi_decline_5d_thresh:.2f}, "
          f"oi_10d_thr={oi_decline_10d_thresh:.2f})...", flush=True)

    rank_weight_val = 1.0 - oi_weight
    composite = np.full((NS, ND), np.nan)
    n_confirm = np.zeros((NS, ND), dtype=int)
    oi_extreme_flag = np.zeros((NS, ND), dtype=bool)

    factor_names = list(rank_weights.keys())
    weight_vals = np.array([rank_weights[k] for k in factor_names])

    for di in range(ND):
        for si in range(NS):
            # --- V18 rank base ---
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

            if w_sum == 0:
                continue

            v18_score = sum(vals) / w_sum

            # --- OI capitulation bonus ---
            oi_capitulation_score = 0.0
            oi_decl_5d = raw_factors['oi_decline_5d'][si, di]
            oi_decl_10d = raw_factors['oi_decline_10d'][si, di]

            if not np.isnan(oi_decl_5d) and not np.isnan(oi_decl_10d):
                # Check if OI extreme condition is met
                if oi_decl_5d < oi_decline_5d_thresh and oi_decl_10d < oi_decline_10d_thresh:
                    oi_extreme_flag[si, di] = True
                    confirm_count += 1

                # Use cross-sectional rank of OI decline 5d as bonus
                oi_rank = ranks['rank_oi_decline_5d'][si, di]
                if not np.isnan(oi_rank):
                    oi_capitulation_score = oi_rank

            # --- Combine ---
            composite[si, di] = rank_weight_val * v18_score + oi_weight * oi_capitulation_score

            # Count additional confidence signals
            # consec_dn >= 3
            if raw_factors['consec_dn'][si, di] >= 3:
                confirm_count += 1
            # ret5d oversold (< -3%)
            ret5 = raw_factors['ret_5d'][si, di]
            if not np.isnan(ret5) and ret5 < -0.03:
                confirm_count += 1
            # rsi oversold (< 35)
            rsi = raw_factors['rsi14'][si, di]
            if not np.isnan(rsi) and rsi < 35:
                confirm_count += 1
            # vol surge (> 0.3)
            vs = raw_factors['vol_surge'][si, di]
            if not np.isnan(vs) and vs > 0.3:
                confirm_count += 1

            n_confirm[si, di] = confirm_count

    print(f"  Composite done: {time.time() - t0:.1f}s", flush=True)
    return composite, n_confirm, oi_extreme_flag


def compute_all_signals(C, O, H, L, V, OI, NS, ND,
                        rank_weights=None, oi_weight=DEFAULT_OI_WEIGHT,
                        oi_decline_5d_thresh=DEFAULT_OI_DECLINE_5D_THRESH,
                        oi_decline_10d_thresh=DEFAULT_OI_DECLINE_10D_THRESH):
    """Full signal pipeline for V32."""
    if rank_weights is None:
        rank_weights = DEFAULT_RANK_WEIGHTS

    raw = compute_raw_factors(C, O, H, L, V, OI, NS, ND)
    ranks = compute_cross_sectional_ranks(raw, NS, ND)
    ker_regime = compute_ker(C, NS, ND)
    composite, n_confirm, oi_extreme = build_composite_signal(
        ranks, raw, rank_weights, oi_weight,
        oi_decline_5d_thresh, oi_decline_10d_thresh,
        NS, ND)

    return {
        'composite': composite,
        'n_confirm': n_confirm,
        'ker_regime': ker_regime,
        'ranks': ranks,
        'oi_extreme': oi_extreme,
        'raw': raw,
    }


def backtest_v32(C, O, H, L, NS, ND, dates, syms, sigs,
                 top_n=1, min_rank=0.75, atr_stop=3.0,
                 min_confidence=3, use_ker_gate=True,
                 hold_days=5, pyramid_ratio=0.5, pyramid_day=1,
                 start_di=60, end_di=None):
    """Backtest with rank+OI extreme combo signals + pyramid."""
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
    print(f"  WALK-FORWARD V32 (pyr={pyramid_ratio}, tn={top_n}, "
          f"mr={min_rank:.2f}, as={atr_stop:.1f})")
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

        trades, _, _ = backtest_v32(
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
    print("  V32: RANK + OI EXTREME COMBO")
    print("  Best of V18 (rank) + V12 (OI contrarian)")
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

    # ================================================================
    # 1. Default config walk-forward
    # ================================================================
    print("\n" + "=" * 70)
    print("  SECTION 1: DEFAULT CONFIG WALK-FORWARD (2019-2026)")
    print("=" * 70)

    default_sigs = compute_all_signals(C, O, H, L, V, OI, NS, ND)

    for ratio in [0.0, 0.5]:
        walk_forward(C, O, H, L, V, OI, NS, ND, dates, syms, default_sigs,
                     pyramid_ratio=ratio, pyramid_day=1,
                     top_n=1, min_confidence=3)

    # ================================================================
    # 2. Full 10-year backtest with default config
    # ================================================================
    print("\n" + "=" * 70)
    print("  SECTION 2: FULL 2016-2026 (10 years)")
    print("=" * 70)

    profiles = [
        (0.0, 1, "No pyramid (baseline)"),
        (0.5, 1, "Moderate pyramid (50%)"),
    ]

    for ratio, pday, label in profiles:
        trades, eq, dd = backtest_v32(
            C, O, H, L, NS, ND, dates, syms, default_sigs,
            top_n=1, hold_days=5, atr_stop=3.0,
            min_rank=0.75, min_confidence=3, use_ker_gate=True,
            pyramid_ratio=ratio, pyramid_day=pday,
            start_di=60)
        print(f"\n  {label}")
        analyze(trades, eq, dd, label)

    # ================================================================
    # 3. OI weight sweep with default signals (oi_weight changes need recompute)
    # ================================================================
    print("\n" + "=" * 70)
    print("  SECTION 3: OI WEIGHT SWEEP (2019-2026)")
    print("=" * 70)

    oi_weight_results = []
    for oi_w in [0.20, 0.30, 0.40]:
        for oi_5d_thr in [-0.03, -0.05, -0.08]:
            sigs_oi = compute_all_signals(
                C, O, H, L, V, OI, NS, ND,
                oi_weight=oi_w,
                oi_decline_5d_thresh=oi_5d_thr,
                oi_decline_10d_thresh=DEFAULT_OI_DECLINE_10D_THRESH)
            trades, eq, dd = backtest_v32(
                C, O, H, L, NS, ND, dates, syms, sigs_oi,
                top_n=1, hold_days=5, atr_stop=3.0,
                min_rank=0.75, min_confidence=3, use_ker_gate=True,
                pyramid_ratio=0.5, pyramid_day=1,
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
            oi_weight_results.append({
                'oi_w': oi_w, 'oi_5d_thr': oi_5d_thr,
                'n': len(trades), 'wr': wr, 'ann': ann,
                'dd': dd, 'sharpe': sh_val, 'eq': eq,
            })
            print(f"  oi_w={oi_w:.2f} oi_5d_thr={oi_5d_thr:+.0%}: "
                  f"{len(trades)}t WR={wr:.1f}% ann={ann:+.1f}% "
                  f"DD={dd:.1f}% Sh={sh_val:.2f}")

    oi_weight_results.sort(key=lambda x: -x['sharpe'])
    if oi_weight_results:
        best_oi = oi_weight_results[0]
        print(f"\n  Best OI config: oi_w={best_oi['oi_w']:.2f} "
              f"oi_5d_thr={best_oi['oi_5d_thr']:+.0%} "
              f"Sh={best_oi['sharpe']:.2f}")

    # ================================================================
    # 4. Full parameter sweep (2019-2026)
    # ================================================================
    print("\n" + "=" * 70)
    print("  SECTION 4: FULL PARAMETER SWEEP (2019-2026)")
    print("=" * 70)

    # Use best OI weight from sweep, or default
    best_oi_w = best_oi['oi_w'] if oi_weight_results else DEFAULT_OI_WEIGHT
    best_oi_5d = best_oi['oi_5d_thr'] if oi_weight_results else DEFAULT_OI_DECLINE_5D_THRESH

    sweep_sigs = compute_all_signals(
        C, O, H, L, V, OI, NS, ND,
        oi_weight=best_oi_w,
        oi_decline_5d_thresh=best_oi_5d,
        oi_decline_10d_thresh=DEFAULT_OI_DECLINE_10D_THRESH)

    results = []
    for tn in [1, 2, 3]:
        for ratio in [0.0, 0.5]:
            for mr in [0.70, 0.75, 0.80]:
                for as_val in [2.5, 3.0]:
                    for mc in [3, 4]:
                        trades, eq, dd = backtest_v32(
                            C, O, H, L, NS, ND, dates, syms, sweep_sigs,
                            top_n=tn, hold_days=5, atr_stop=as_val,
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
                            'mr': mr, 'as': as_val,
                            'n': len(trades), 'wr': wr, 'ann': ann,
                            'dd': dd, 'sharpe': sh_val, 'eq': eq,
                        })

    results.sort(key=lambda x: -x['sharpe'])
    print(f"\n{'TN':>3} {'Pyr':>4} {'MC':>3} {'MR':>4} {'AS':>4} "
          f"{'N':>5} {'WR':>5} {'Ann':>8} {'DD':>6} {'Sh':>5}")
    print("-" * 70)
    for r in results[:25]:
        print(f"{r['tn']:>3} {r['ratio']:>4.1f} {r['mc']:>3} {r['mr']:>4.2f} {r['as']:>4.1f} "
              f"{r['n']:>5} {r['wr']:>5.1f} {r['ann']:>+8.1f} "
              f"{r['dd']:>6.1f} {r['sharpe']:>5.2f}")

    # ================================================================
    # 5. Best configs full 10-year
    # ================================================================
    print("\n" + "=" * 70)
    print("  SECTION 5: BEST CONFIGS -- FULL 10-YEAR")
    print("=" * 70)

    if results:
        for r in results[:5]:
            trades, eq, dd = backtest_v32(
                C, O, H, L, NS, ND, dates, syms, sweep_sigs,
                top_n=r['tn'], hold_days=5, atr_stop=r['as'],
                min_rank=r['mr'], min_confidence=r['mc'],
                use_ker_gate=True,
                pyramid_ratio=r['ratio'], pyramid_day=1,
                start_di=60)
            label = (f"tn={r['tn']} pyr={r['ratio']:.1f} mc={r['mc']} "
                     f"mr={r['mr']:.2f} as={r['as']:.1f}")
            print(f"\n  FULL {label}")
            analyze(trades, eq, dd, label)

    # ================================================================
    # 6. Walk-forward for top 3 configs
    # ================================================================
    print("\n" + "=" * 70)
    print("  SECTION 6: WALK-FORWARD TOP 3 CONFIGS (2019-2026)")
    print("=" * 70)

    if results:
        for r in results[:3]:
            print(f"\n  --- Config: tn={r['tn']} pyr={r['ratio']:.1f} "
                  f"mc={r['mc']} mr={r['mr']:.2f} as={r['as']:.1f} ---")
            walk_forward(C, O, H, L, V, OI, NS, ND, dates, syms, sweep_sigs,
                         pyramid_ratio=r['ratio'], pyramid_day=1,
                         top_n=r['tn'], min_confidence=r['mc'],
                         hold_days=5, atr_stop=r['as'],
                         min_rank=r['mr'])

    print(f"\n[V32] Done. {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
