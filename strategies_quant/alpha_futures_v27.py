"""
V27: Multi-Timeframe Rank Mean Reversion
==========================================
Core thesis: V9 (multi-timeframe) failed because it used raw signals.
V27 applies V18's proven rank methodology across timeframes:
rank 5d oversold AND rank 20d oversold simultaneously.
When a commodity is oversold on BOTH timeframes, the reversal is
more reliable.

Signal architecture:
  1. Short-term (5d) cross-sectional ranks:
     - rank_ret5d, rank_oi5d_5d, rank_vol_5d, rank_rsi_5d
  2. Medium-term (20d) cross-sectional ranks:
     - rank_ret20d, rank_oi_20d, rank_vol_20d
  3. Composite = st_weight * short_term_score + (1-st_weight) * medium_term_score
  4. Only enter when BOTH timeframes agree (rank > min_rank on both)
  5. KER gate, confidence >= min_confidence, hold 5d, ATR stop
  6. Pyramid on day-1 winners (configurable ratio)

Parameter sweep:
  - st_weight: 0.50, 0.60, 0.70
  - top_n: 1, 2, 3
  - min_rank: 0.65, 0.70, 0.75
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

# Default weights for short-term (5d) and medium-term (20d) composites
ST_WEIGHTS = {
    'rank_ret5d':  0.30,
    'rank_oi5d':   0.25,
    'rank_rsi5d':  0.25,
    'rank_vol5d':  0.20,
}

MT_WEIGHTS = {
    'rank_ret20d': 0.40,
    'rank_oi20d':  0.35,
    'rank_vol20d': 0.25,
}


def compute_rsi_manual(C: np.ndarray, NS: int, ND: int,
                       period: int = 14) -> np.ndarray:
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


def compute_raw_factors(C: np.ndarray, O: np.ndarray, H: np.ndarray,
                        L: np.ndarray, V: np.ndarray, OI: np.ndarray,
                        NS: int, ND: int) -> dict:
    """Compute raw factor values for both short-term (5d) and medium-term (20d)."""
    t0 = time.time()
    print("[V27] Computing raw factors (5d + 20d)...", flush=True)

    # === Short-term (5d) factors ===
    ret_5d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            if not np.isnan(C[si, di]) and not np.isnan(C[si, di - 5]) and C[si, di - 5] > 0:
                ret_5d[si, di] = C[si, di] / C[si, di - 5] - 1.0

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

    # RSI 5 (short-term momentum)
    rsi5 = np.full((NS, ND), np.nan)
    if HAS_TALIB:
        for si in range(NS):
            c = np.where(np.isnan(C[si]), 0, C[si]).astype(np.float64)
            nan_mask = np.isnan(C[si])
            try:
                r = talib.RSI(c, 5)
                rsi5[si] = np.where(nan_mask, np.nan, r)
            except Exception:
                pass
    needs_fallback_rsi5 = np.all(np.isnan(rsi5), axis=1)
    if needs_fallback_rsi5.any():
        rsi5_manual = compute_rsi_manual(C, NS, ND, 5)
        for si in range(NS):
            if needs_fallback_rsi5[si]:
                rsi5[si] = rsi5_manual[si]

    # === Medium-term (20d) factors ===
    ret_20d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            if not np.isnan(C[si, di]) and not np.isnan(C[si, di - 20]) and C[si, di - 20] > 0:
                ret_20d[si, di] = C[si, di] / C[si, di - 20] - 1.0

    oi_20d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            if not np.isnan(OI[si, di]) and not np.isnan(OI[si, di - 20]) and OI[si, di - 20] > 0:
                oi_20d[si, di] = OI[si, di] / OI[si, di - 20] - 1.0

    vol_20d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            vals = V[si, di - 20:di]
            valid = vals[~np.isnan(vals)]
            if len(valid) >= 10:
                vol_20d[si, di] = np.mean(valid)

    print(f"  Raw factors done: {time.time() - t0:.1f}s", flush=True)
    return {
        'ret_5d': ret_5d,
        'oi_5d': oi_5d,
        'vol_5d': vol_5d,
        'rsi5': rsi5,
        'ret_20d': ret_20d,
        'oi_20d': oi_20d,
        'vol_20d': vol_20d,
    }


def compute_cross_sectional_ranks(raw_factors: dict, NS: int, ND: int,
                                   min_count: int = 10) -> dict:
    """Rank all factors cross-sectionally (across commodities per day).
    Inverted so LOW raw value = high rank (most oversold = best mean reversion)."""
    t0 = time.time()
    print("[V27] Computing cross-sectional ranks...", flush=True)

    factors_to_rank = {
        'rank_ret5d': raw_factors['ret_5d'],
        'rank_oi5d': raw_factors['oi_5d'],
        'rank_rsi5d': raw_factors['rsi5'],
        'rank_vol5d': raw_factors['vol_5d'],
        'rank_ret20d': raw_factors['ret_20d'],
        'rank_oi20d': raw_factors['oi_20d'],
        'rank_vol20d': raw_factors['vol_20d'],
    }

    # All inverted: low raw value -> high rank (most oversold)
    # For vol factors: high volume -> high rank (attention, no invert needed for vol)
    INVERT_FACTORS = {'rank_ret5d', 'rank_oi5d', 'rank_rsi5d',
                      'rank_ret20d', 'rank_oi20d'}

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


def compute_ker(C: np.ndarray, NS: int, ND: int) -> np.ndarray:
    """Kaufman Efficiency Ratio for regime detection."""
    ker_regime = np.zeros((NS, ND), dtype=int)
    for si in range(NS):
        for di in range(10, ND):
            closes = C[si, di - 10:di + 1]
            valid = closes[~np.isnan(closes)]
            if len(valid) < 10 or valid[0] <= 0:
                continue
            net_change = abs(valid[-1] - valid[0])
            total_change = np.sum(np.abs(np.diff(valid)))
            if total_change > 1e-10:
                ker_val = net_change / total_change
                if ker_val < 0.15:
                    ker_regime[si, di] = 1   # sideways -> good for MR
                elif ker_val > 0.3:
                    ker_regime[si, di] = -1  # trending -> avoid
    return ker_regime


def build_multi_tf_signal(ranks: dict, st_weights: dict, mt_weights: dict,
                          st_weight: float, NS: int, ND: int,
                          min_factors: int = 2) -> tuple:
    """Build multi-timeframe composite signal.
    Returns (composite, st_composite, mt_composite, n_confirm_st, n_confirm_mt).
    """
    t0 = time.time()
    print(f"[V27] Building multi-TF signal (st_w={st_weight:.2f})...", flush=True)

    mt_weight = 1.0 - st_weight

    composite = np.full((NS, ND), np.nan)
    st_comp = np.full((NS, ND), np.nan)
    mt_comp = np.full((NS, ND), np.nan)
    n_confirm_st = np.zeros((NS, ND), dtype=int)
    n_confirm_mt = np.zeros((NS, ND), dtype=int)

    st_names = list(st_weights.keys())
    st_wvals = np.array([st_weights[k] for k in st_names])
    mt_names = list(mt_weights.keys())
    mt_wvals = np.array([mt_weights[k] for k in mt_names])

    for di in range(ND):
        for si in range(NS):
            # Short-term composite
            st_vals = []
            st_wsum = 0.0
            st_confirm = 0
            for idx, name in enumerate(st_names):
                rv = ranks[name][si, di]
                if np.isnan(rv):
                    continue
                st_vals.append(rv * st_wvals[idx])
                st_wsum += st_wvals[idx]
                if rv > 0.5:
                    st_confirm += 1

            if st_wsum > 0 and st_confirm >= min_factors:
                st_comp[si, di] = sum(st_vals) / st_wsum
                n_confirm_st[si, di] = st_confirm

            # Medium-term composite
            mt_vals = []
            mt_wsum = 0.0
            mt_confirm = 0
            for idx, name in enumerate(mt_names):
                rv = ranks[name][si, di]
                if np.isnan(rv):
                    continue
                mt_vals.append(rv * mt_wvals[idx])
                mt_wsum += mt_wvals[idx]
                if rv > 0.5:
                    mt_confirm += 1

            if mt_wsum > 0 and mt_confirm >= min_factors:
                mt_comp[si, di] = sum(mt_vals) / mt_wsum
                n_confirm_mt[si, di] = mt_confirm

            # Combined composite: only when both timeframes available
            if not np.isnan(st_comp[si, di]) and not np.isnan(mt_comp[si, di]):
                composite[si, di] = (st_weight * st_comp[si, di] +
                                     mt_weight * mt_comp[si, di])

    print(f"  Multi-TF signal done: {time.time() - t0:.1f}s", flush=True)
    return composite, st_comp, mt_comp, n_confirm_st, n_confirm_mt


def compute_all_signals(C: np.ndarray, O: np.ndarray, H: np.ndarray,
                        L: np.ndarray, V: np.ndarray, OI: np.ndarray,
                        NS: int, ND: int, st_weight: float = 0.60,
                        st_weights: dict = None,
                        mt_weights: dict = None) -> dict:
    """Full signal pipeline for V27."""
    if st_weights is None:
        st_weights = ST_WEIGHTS
    if mt_weights is None:
        mt_weights = MT_WEIGHTS

    raw = compute_raw_factors(C, O, H, L, V, OI, NS, ND)
    ranks = compute_cross_sectional_ranks(raw, NS, ND)
    ker_regime = compute_ker(C, NS, ND)
    composite, st_comp, mt_comp, ncf_st, ncf_mt = build_multi_tf_signal(
        ranks, st_weights, mt_weights, st_weight, NS, ND)

    return {
        'composite': composite,
        'st_comp': st_comp,
        'mt_comp': mt_comp,
        'n_confirm_st': ncf_st,
        'n_confirm_mt': ncf_mt,
        'ker_regime': ker_regime,
        'ranks': ranks,
    }


def backtest_v27(C: np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
                 NS: int, ND: int, dates: list, syms: list, sigs: dict,
                 top_n: int = 1, min_rank: float = 0.70,
                 atr_stop: float = 3.0, min_confidence: int = 2,
                 use_ker_gate: bool = True, hold_days: int = 5,
                 pyramid_ratio: float = 0.5, pyramid_day: int = 1,
                 st_weight: float = 0.60,
                 start_di: int = 60, end_di: int = None) -> tuple:
    """Backtest V27: multi-timeframe rank mean reversion with pyramid."""
    composite = sigs['composite']
    st_comp = sigs['st_comp']
    mt_comp = sigs['mt_comp']
    ncf_st = sigs['n_confirm_st']
    ncf_mt = sigs['n_confirm_mt']
    ker_regime = sigs['ker_regime']

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
                            additions.append((si, di, c_now,
                                              c_now - atr_stop * atr, pyr_alloc, True))
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
        # BOTH timeframes must agree: rank > min_rank on ST and MT
        candidates = []
        for si in range(NS):
            if si in held:
                continue
            # Need composite signal
            if np.isnan(composite[si, di]):
                continue
            # Both ST and MT must exceed min_rank
            if np.isnan(st_comp[si, di]) or st_comp[si, di] < min_rank:
                continue
            if np.isnan(mt_comp[si, di]) or mt_comp[si, di] < min_rank:
                continue
            # Total composite must exceed min_rank
            if composite[si, di] < min_rank:
                continue
            # Confidence: total confirming factors across both timeframes
            total_confirm = ncf_st[si, di] + ncf_mt[si, di]
            if total_confirm < min_confidence:
                continue
            # KER gate
            if use_ker_gate and ker_regime[si, di] < 0:
                continue
            if di + 1 >= ND or np.isnan(O[si, di + 1]):
                continue
            alloc = 1.0 / max(top_n, 1)
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


def analyze(trades: list, equity: float, max_dd: float, label: str = "") -> dict | None:
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

    return {'n': len(trades), 'wr': wr, 'dd': max_dd, 'ann': ann,
            'sh': sh, 'eq': equity}


def walk_forward(C: np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
                 V: np.ndarray, OI: np.ndarray, NS: int, ND: int,
                 dates: list, syms: list, sigs: dict,
                 pyramid_ratio: float = 0.5, pyramid_day: int = 1,
                 top_n: int = 1, min_confidence: int = 2,
                 hold_days: int = 5, atr_stop: float = 3.0,
                 min_rank: float = 0.70, st_weight: float = 0.60) -> list:
    """Walk-forward validation: year-by-year out-of-sample."""
    print(f"\n{'=' * 70}")
    print(f"  WALK-FORWARD V27 (pyr={pyramid_ratio}, st_w={st_weight:.2f})")
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

        trades, _, _ = backtest_v27(
            C, O, H, L, NS, ND, dates, syms, sigs,
            top_n=top_n, hold_days=hold_days, atr_stop=atr_stop,
            min_rank=min_rank, min_confidence=min_confidence,
            use_ker_gate=True, pyramid_ratio=pyramid_ratio,
            pyramid_day=pyramid_day, st_weight=st_weight,
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
    print("  V27: MULTI-TIMEFRAME RANK MEAN REVERSION")
    print("  Short-term (5d) + Medium-term (20d) cross-sectional ranks")
    print("  Both timeframes must agree for entry")
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

    # === 1. Walk-Forward Validation with default params ===
    print("\n" + "=" * 70)
    print("  WALK-FORWARD VALIDATION (2019-2026) -- DEFAULT PARAMS")
    print("=" * 70)

    for st_w in [0.50, 0.60, 0.70]:
        sigs = compute_all_signals(C, O, H, L, V, OI, NS, ND, st_weight=st_w)
        for ratio in [0.0, 0.5]:
            walk_forward(C, O, H, L, V, OI, NS, ND, dates, syms, sigs,
                         pyramid_ratio=ratio, pyramid_day=1,
                         top_n=1, min_confidence=2,
                         hold_days=5, atr_stop=3.0,
                         min_rank=0.70, st_weight=st_w)

    # === 2. Parameter sweep ===
    print("\n" + "=" * 70)
    print("  PARAMETER SWEEP (2019-2026)")
    print("=" * 70)

    results = []

    for st_w in [0.50, 0.60, 0.70]:
        sigs = compute_all_signals(C, O, H, L, V, OI, NS, ND, st_weight=st_w)
        for tn in [1, 2, 3]:
            for ratio in [0.0, 0.5]:
                for mc in [2, 3]:
                    for mr in [0.65, 0.70, 0.75]:
                        for as_val in [2.5, 3.0]:
                            trades, eq, dd = backtest_v27(
                                C, O, H, L, NS, ND, dates, syms, sigs,
                                top_n=tn, hold_days=5, atr_stop=as_val,
                                min_rank=mr, min_confidence=mc,
                                use_ker_gate=True,
                                pyramid_ratio=ratio, pyramid_day=1,
                                st_weight=st_w,
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
                                'st_w': st_w, 'tn': tn, 'ratio': ratio,
                                'mc': mc, 'mr': mr, 'atr': as_val,
                                'n': len(trades), 'wr': wr, 'ann': ann,
                                'dd': dd, 'sharpe': sh_val,
                            })

    results.sort(key=lambda x: -x['sharpe'])
    print(f"\n{'STw':>4} {'TN':>3} {'Pyr':>4} {'MC':>3} {'MR':>4} {'ATR':>4} "
          f"{'N':>5} {'WR':>5} {'Ann':>8} {'DD':>6} {'Sh':>5}")
    print("-" * 75)
    for r in results[:30]:
        print(f"{r['st_w']:>4.2f} {r['tn']:>3} {r['ratio']:>4.1f} {r['mc']:>3} "
              f"{r['mr']:>4.2f} {r['atr']:>4.1f} "
              f"{r['n']:>5} {r['wr']:>5.1f} {r['ann']:>+8.1f} "
              f"{r['dd']:>6.1f} {r['sharpe']:>5.2f}")

    # === 3. Top configs -- full 10-year ===
    if results:
        print("\n" + "=" * 70)
        print("  TOP CONFIGS -- FULL 10-YEAR (2016-2026)")
        print("=" * 70)

        seen = set()
        unique_top = []
        for r in results:
            key = (r['st_w'], r['tn'], r['ratio'], r['mc'], r['mr'], r['atr'])
            if key not in seen:
                seen.add(key)
                unique_top.append(r)
            if len(unique_top) >= 5:
                break

        for r in unique_top:
            sigs = compute_all_signals(C, O, H, L, V, OI, NS, ND,
                                       st_weight=r['st_w'])
            trades, eq, dd = backtest_v27(
                C, O, H, L, NS, ND, dates, syms, sigs,
                top_n=r['tn'], hold_days=5, atr_stop=r['atr'],
                min_rank=r['mr'], min_confidence=r['mc'],
                use_ker_gate=True,
                pyramid_ratio=r['ratio'], pyramid_day=1,
                st_weight=r['st_w'],
                start_di=60)
            label = (f"st_w={r['st_w']:.2f} tn={r['tn']} pyr={r['ratio']:.1f} "
                     f"mc={r['mc']} mr={r['mr']:.2f} atr={r['atr']:.1f}")
            print(f"\n  FULL {label}")
            analyze(trades, eq, dd, label)

        # === 4. Walk-forward for best config ===
        best = unique_top[0]
        print("\n" + "=" * 70)
        print(f"  BEST WALK-FORWARD: st_w={best['st_w']:.2f} tn={best['tn']} "
              f"pyr={best['ratio']:.1f} mc={best['mc']} mr={best['mr']:.2f} "
              f"atr={best['atr']:.1f}")
        print("=" * 70)

        best_sigs = compute_all_signals(C, O, H, L, V, OI, NS, ND,
                                        st_weight=best['st_w'])
        walk_forward(C, O, H, L, V, OI, NS, ND, dates, syms, best_sigs,
                     pyramid_ratio=best['ratio'], pyramid_day=1,
                     top_n=best['tn'], min_confidence=best['mc'],
                     hold_days=5, atr_stop=best['atr'],
                     min_rank=best['mr'], st_weight=best['st_w'])

    print(f"\n[V27] Done. {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
