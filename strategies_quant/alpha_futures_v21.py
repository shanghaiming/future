"""
V21: Volume Profile Mean Reversion
====================================
Core thesis: Build a simple volume profile (VWAP + bands) to identify price
levels where most volume occurred. Use these as "fair value" anchors for
mean reversion entries when price deviates significantly from volume-weighted
fair value.

Signal architecture:
  1. Rolling VWAP (volume-weighted average price) as "fair value" anchor
  2. Volume-weighted std bands (VWAP +/- N * std) as overbought/oversold
  3. Signal: price closes below lower VWAP band (oversold vs volume profile)
  4. Confirm with: consecutive down days >= 3, OI declining, VDP exhaustion
  5. Cross-sectional rank "distance from VWAP" across all 50 commodities
  6. KER gate: only enter when KER < 0.15 (mean-reverting regime)
  7. Pyramid on day-1 winners (ratio 0.5)

Signal at close[di], enter at open[di+1]. No look-ahead. No gap signals.
No leverage. Fixed 5-day hold. Walk-forward 2019-2026 + full 10-year.
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


def compute_vwap_and_bands(C, H, L, V, NS, ND, vwap_window=20, band_width=1.5):
    """Compute rolling VWAP and volume-weighted standard deviation bands.

    VWAP = sum(typical_price * volume) / sum(volume) over window
    typical_price = (H + L + C) / 3
    VWAP std = sqrt(sum(volume * (tp - vwap)^2) / sum(volume))
    """
    t0 = time.time()
    print(f"[V21] Computing VWAP (window={vwap_window}, band={band_width})...",
          flush=True)

    vwap = np.full((NS, ND), np.nan)
    vwap_upper = np.full((NS, ND), np.nan)
    vwap_lower = np.full((NS, ND), np.nan)
    vwap_std = np.full((NS, ND), np.nan)

    for si in range(NS):
        c = C[si]
        h = H[si]
        l = L[si]
        v = V[si]

        for di in range(vwap_window, ND):
            # Gather window data
            tp_vals = []
            vol_vals = []
            for j in range(di - vwap_window, di):
                if (np.isnan(c[j]) or np.isnan(h[j]) or np.isnan(l[j])
                        or np.isnan(v[j]) or v[j] <= 0):
                    continue
                tp = (h[j] + l[j] + c[j]) / 3.0
                tp_vals.append(tp)
                vol_vals.append(v[j])

            if len(tp_vals) < vwap_window // 2:
                continue

            tp_arr = np.array(tp_vals)
            v_arr = np.array(vol_vals)
            v_sum = np.sum(v_arr)

            if v_sum <= 0:
                continue

            # VWAP = volume-weighted mean of typical price
            vwap_val = np.sum(tp_arr * v_arr) / v_sum

            # Volume-weighted standard deviation
            diff_sq = (tp_arr - vwap_val) ** 2
            vw_std = np.sqrt(np.sum(v_arr * diff_sq) / v_sum)

            vwap[si, di] = vwap_val
            vwap_std[si, di] = vw_std
            vwap_upper[si, di] = vwap_val + band_width * vw_std
            vwap_lower[si, di] = vwap_val - band_width * vw_std

    print(f"  VWAP done: {time.time() - t0:.1f}s", flush=True)
    return vwap, vwap_upper, vwap_lower, vwap_std


def compute_vdp(C, V, NS, ND, window=20):
    """Volume-Weighted Directional Percentage: measures selling exhaustion.

    VDP = sum(volume * sign(close - prev_close)) / sum(volume) over window.
    Low VDP = persistent selling = potential exhaustion.
    """
    t0 = time.time()
    print("[V21] Computing VDP selling exhaustion...", flush=True)

    vdp = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si]
        v = V[si]
        for di in range(window + 1, ND):
            signed_vol = []
            total_vol = []
            for j in range(di - window, di):
                if np.isnan(c[j]) or np.isnan(c[j - 1]) or np.isnan(v[j]):
                    continue
                direction = 1.0 if c[j] > c[j - 1] else -1.0
                signed_vol.append(direction * v[j])
                total_vol.append(v[j])

            if len(total_vol) >= window // 2 and sum(total_vol) > 0:
                vdp[si, di] = sum(signed_vol) / sum(total_vol)

    print(f"  VDP done: {time.time() - t0:.1f}s", flush=True)
    return vdp


def compute_consecutive_down(C, NS, ND):
    """Count consecutive down days (for selling exhaustion confirmation)."""
    cons_down = np.zeros((NS, ND), dtype=int)
    for si in range(NS):
        c = C[si]
        count = 0
        for di in range(1, ND):
            if np.isnan(c[di]) or np.isnan(c[di - 1]):
                count = 0
                continue
            if c[di] < c[di - 1]:
                count += 1
            else:
                count = 0
            cons_down[si, di] = count
    return cons_down


def compute_oi_change(OI, NS, ND, period=5):
    """OI change over period. Declining OI + price drop = capitulation."""
    oi_chg = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(period, ND):
            if (not np.isnan(OI[si, di]) and not np.isnan(OI[si, di - period])
                    and OI[si, di - period] > 0):
                oi_chg[si, di] = OI[si, di] / OI[si, di - period] - 1.0
    return oi_chg


def compute_ker(C, NS, ND, period=10):
    """Kaufman Efficiency Ratio for regime detection."""
    ker = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(period, ND):
            closes = C[si, di - period:di + 1]
            valid = closes[~np.isnan(closes)]
            if len(valid) < period or valid[0] <= 0:
                continue
            net_change = abs(valid[-1] - valid[0])
            total_change = np.sum(np.abs(np.diff(valid)))
            if total_change > 1e-10:
                ker[si, di] = net_change / total_change
    return ker


def compute_atrp(C, H, L, NS, ND, period=14):
    """ATR as percentage of close price."""
    atrp = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(period, ND):
            atr_vals = []
            for j in range(di - period, di):
                hh, ll, cc = H[si, j], L[si, j], C[si, j]
                if any(np.isnan([hh, ll, cc])):
                    continue
                prev_c = C[si, j - 1] if j > 0 and not np.isnan(C[si, j - 1]) else cc
                atr_vals.append(max(hh - ll, abs(hh - prev_c), abs(ll - prev_c)))
            if atr_vals and not np.isnan(C[si, di]) and C[si, di] > 0:
                atrp[si, di] = np.mean(atr_vals) / C[si, di]
    return atrp


def compute_all_signals(C, O, H, L, V, OI, NS, ND,
                        vwap_window=20, band_width=1.5):
    """Full signal pipeline: VWAP bands + confirmations + CS rank + KER gate."""
    t0 = time.time()
    print("[V21] Computing all signals...", flush=True)

    # Core VWAP bands
    vwap, vwap_upper, vwap_lower, vwap_std = compute_vwap_and_bands(
        C, H, L, V, NS, ND, vwap_window=vwap_window, band_width=band_width)

    # Distance from VWAP as a fraction of VWAP (how far price strayed)
    dist_from_vwap = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(ND):
            if (not np.isnan(C[si, di]) and not np.isnan(vwap[si, di])
                    and vwap[si, di] > 0):
                dist_from_vwap[si, di] = (C[si, di] - vwap[si, di]) / vwap[si, di]

    # Confirmation factors
    vdp = compute_vdp(C, V, NS, ND, window=20)
    cons_down = compute_consecutive_down(C, NS, ND)
    oi_chg = compute_oi_change(OI, NS, ND, period=5)
    ker = compute_ker(C, NS, ND, period=10)
    atrp = compute_atrp(C, H, L, NS, ND, period=14)

    # RSI for additional confirmation
    rsi14 = np.full((NS, ND), np.nan)
    if HAS_TALIB:
        for si in range(NS):
            c_arr = np.where(np.isnan(C[si]), 0, C[si]).astype(np.float64)
            nan_mask = np.isnan(C[si])
            try:
                r = talib.RSI(c_arr, 14)
                rsi14[si] = np.where(nan_mask, np.nan, r)
            except Exception:
                pass
    needs_fallback = np.all(np.isnan(rsi14), axis=1)
    if needs_fallback.any():
        rsi_manual = compute_rsi_manual(C, NS, ND, 14)
        for si in range(NS):
            if needs_fallback[si]:
                rsi14[si] = rsi_manual[si]

    # Cross-sectional rank of distance from VWAP
    # Low distance = most oversold vs volume profile = best MR candidate
    cs_rank_vwap = np.full((NS, ND), np.nan)
    for di in range(ND):
        vals = dist_from_vwap[:, di]
        valid_count = np.sum(~np.isnan(vals))
        if valid_count < 10:
            continue
        # Rank: low distance -> high rank (most oversold)
        ranked = pd.Series(vals).rank(pct=True, na_option='keep').values
        cs_rank_vwap[:, di] = 1.0 - ranked  # invert: oversold = high rank

    # KER regime: 1 = sideways (good for MR), -1 = trending (avoid)
    ker_regime = np.zeros((NS, ND), dtype=int)
    for si in range(NS):
        for di in range(ND):
            if np.isnan(ker[si, di]):
                continue
            if ker[si, di] < 0.15:
                ker_regime[si, di] = 1
            elif ker[si, di] > 0.3:
                ker_regime[si, di] = -1

    print(f"  All signals done: {time.time() - t0:.1f}s", flush=True)

    return {
        'vwap': vwap,
        'vwap_upper': vwap_upper,
        'vwap_lower': vwap_lower,
        'vwap_std': vwap_std,
        'dist_from_vwap': dist_from_vwap,
        'cs_rank_vwap': cs_rank_vwap,
        'vdp': vdp,
        'cons_down': cons_down,
        'oi_chg': oi_chg,
        'ker': ker,
        'ker_regime': ker_regime,
        'rsi14': rsi14,
        'atrp': atrp,
    }


def backtest_v21(C, O, H, L, NS, ND, dates, syms, sigs,
                 top_n=1, min_rank=0.75, atr_stop=3.0,
                 min_confidence=2, use_ker_gate=True,
                 hold_days=5,
                 pyramid_ratio=0.5,
                 pyramid_day=1,
                 start_di=60, end_di=None):
    """Backtest V21: Volume Profile Mean Reversion.

    Signal: price closes below lower VWAP band.
    Confirmation: cons_down >= 3 OR OI declining OR VDP selling exhaustion.
    Entry: at open[di+1] after signal at close[di].
    """
    vwap_lower = sigs['vwap_lower']
    vwap = sigs['vwap']
    cs_rank_vwap = sigs['cs_rank_vwap']
    ker_regime = sigs['ker_regime']
    cons_down = sigs['cons_down']
    oi_chg = sigs['oi_chg']
    vdp = sigs['vdp']
    rsi14 = sigs['rsi14']

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

        # Pyramid check: add to day-1 winners
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
                    if C[si, di] > avg_ep:  # winner: price above entry
                        base_alloc = sum(a for _, _, _, a, _ in pos_list)
                        pyr_alloc = base_alloc * pyramid_ratio
                        c_now = C[si, di]
                        atr_v = []
                        for j in range(max(start_di, di - 14), di):
                            hh, ll, cc = H[si, j], L[si, j], C[si, j]
                            if not any(np.isnan([hh, ll, cc])):
                                atr_v.append(max(hh - ll, abs(hh - cc),
                                                 abs(ll - cc)))
                        if atr_v:
                            atr = np.mean(atr_v)
                            additions.append((si, di, c_now,
                                              c_now - atr_stop * atr,
                                              pyr_alloc, True))
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

        # --- Entry signals ---
        held = {p[0] for p in positions}
        if len(positions) >= top_n:
            continue

        candidates = []
        for si in range(NS):
            if si in held:
                continue

            # Core signal: close below lower VWAP band
            if np.isnan(vwap_lower[si, di]) or np.isnan(C[si, di]):
                continue
            if C[si, di] >= vwap_lower[si, di]:
                continue

            # CS rank filter: must be in top min_rank percentile of oversold
            if np.isnan(cs_rank_vwap[si, di]):
                continue
            if cs_rank_vwap[si, di] < min_rank:
                continue

            # Confirmation: count how many confirm
            confirm_count = 0

            # 1. Consecutive down days >= 3
            if cons_down[si, di] >= 3:
                confirm_count += 1

            # 2. OI declining (5d change < -2%)
            if not np.isnan(oi_chg[si, di]) and oi_chg[si, di] < -0.02:
                confirm_count += 1

            # 3. VDP selling exhaustion (VDP < -0.3 = heavy selling)
            if not np.isnan(vdp[si, di]) and vdp[si, di] < -0.3:
                confirm_count += 1

            # 4. RSI oversold (< 40, not extreme since we already have VWAP)
            if not np.isnan(rsi14[si, di]) and rsi14[si, di] < 40:
                confirm_count += 1

            if confirm_count < min_confidence:
                continue

            # KER gate: avoid counter-trend in trending markets
            if use_ker_gate and ker_regime[si, di] < 0:
                continue

            # Check next day open is available
            if di + 1 >= ND or np.isnan(O[si, di + 1]):
                continue

            alloc = 1.0 / max(top_n, 1)
            candidates.append((cs_rank_vwap[si, di], si, alloc))

        # Sort by CS rank (most oversold first) and take top_n
        candidates.sort(key=lambda x: -x[0])
        for rank, si, alloc in candidates[:top_n]:
            if len(positions) >= top_n or si in held:
                break
            ep = O[si, di + 1]
            if np.isnan(ep) or ep <= 0:
                continue
            # ATR-based stop loss
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

    print(f"  {label}: {len(trades)}t (base:{n_base} pyr:{n_pyr} "
          f"stop:{n_stop} hold:{n_hold}) "
          f"WR={wr:.1f}% ann={ann:+.1f}% DD={max_dd:.1f}% "
          f"Sh={sh:.2f} eq={equity:,.0f}")

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
        print(f"    {y}: {ys['n']}t WR={ys['w']/ys['n']*100:.1f}% "
              f"cum={cum:+.1%}")

    return {'n': len(trades), 'wr': wr, 'dd': max_dd, 'ann': ann,
            'sh': sh, 'eq': equity}


def walk_forward(C, O, H, L, V, OI, NS, ND, dates, syms,
                 vwap_window=20, band_width=1.5,
                 top_n=1, min_rank=0.75, atr_stop=3.0,
                 min_confidence=2, hold_days=5,
                 pyramid_ratio=0.5, pyramid_day=1):
    """Walk-forward validation: year-by-year out-of-sample."""
    print(f"\n{'=' * 70}")
    print(f"  WALK-FORWARD V21 (vwap_w={vwap_window} bw={band_width} "
          f"tn={top_n} mr={min_rank} mc={min_confidence} "
          f"as={atr_stop} pyr={pyramid_ratio})")
    print(f"{'=' * 70}")

    years = sorted(set(d.year for d in dates))
    all_trades = []

    # Pre-compute signals on full dataset (no look-ahead since signals
    # only use rolling window of past data)
    sigs = compute_all_signals(C, O, H, L, V, OI, NS, ND,
                               vwap_window=vwap_window,
                               band_width=band_width)

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

        trades, _, _ = backtest_v21(
            C, O, H, L, NS, ND, dates, syms, sigs,
            top_n=top_n, hold_days=hold_days, atr_stop=atr_stop,
            min_rank=min_rank, min_confidence=min_confidence,
            use_ker_gate=True,
            pyramid_ratio=pyramid_ratio, pyramid_day=pyramid_day,
            start_di=test_start, end_di=test_end_idx + 1)

        test_trades = [t for t in trades if dates[t['di']].year == test_year]
        all_trades.extend(test_trades)

        if test_trades:
            n = len(test_trades)
            nw_t = sum(1 for t in test_trades if t['pnl_pct'] > 0)
            wr_val = nw_t / n * 100
            avg = np.mean([t['pnl_pct'] for t in test_trades])
            print(f"  {test_year}: {n}t WR={wr_val:.1f}% avg={avg:+.2f}%",
                  flush=True)
        else:
            print(f"  {test_year}: no trades", flush=True)

    if all_trades:
        nw = sum(1 for t in all_trades if t['pnl_pct'] > 0)
        wr_val = nw / len(all_trades) * 100
        avg = np.mean([t['pnl_pct'] for t in all_trades])
        cum = np.prod([1 + t['pnl_pct'] / 100 for t in all_trades]) - 1
        n_pyr = sum(1 for t in all_trades if t.get('pyr'))
        print(f"\n  WF TOTAL: {len(all_trades)}t (pyr:{n_pyr}) "
              f"WR={wr_val:.1f}% avg={avg:+.2f}% cum={cum:+.1%}")
        return all_trades
    return []


def parameter_sweep(C, O, H, L, V, OI, NS, ND, dates, syms, bt_2019):
    """Parameter sweep over V21-specific parameters."""
    print("\n" + "=" * 70)
    print("  PARAMETER SWEEP (2019-2026)")
    print("=" * 70)

    results = []

    for vwap_w in [10, 20, 30]:
        for bw in [1.0, 1.5, 2.0]:
            sigs = compute_all_signals(C, O, H, L, V, OI, NS, ND,
                                       vwap_window=vwap_w, band_width=bw)
            for tn in [1, 2, 3]:
                for mr in [0.65, 0.70, 0.75, 0.80]:
                    for as_val in [2.5, 3.0]:
                        for mc in [2, 3]:
                            for ratio in [0.0, 0.5]:
                                trades, eq, dd = backtest_v21(
                                    C, O, H, L, NS, ND, dates, syms, sigs,
                                    top_n=tn, hold_days=5,
                                    atr_stop=as_val,
                                    min_rank=mr,
                                    min_confidence=mc,
                                    use_ker_gate=True,
                                    pyramid_ratio=ratio,
                                    pyramid_day=1,
                                    start_di=bt_2019)
                                if len(trades) < 10:
                                    continue
                                nw = sum(1 for t in trades
                                         if t['pnl_pct'] > 0)
                                wr = nw / len(trades) * 100
                                n_days = max(1, trades[-1]['di']
                                             - trades[0]['di'])
                                ann = ((eq / CASH0) ** (
                                    1 / max(1.0, n_days / 252)) - 1) * 100
                                ap = [t['pnl_abs'] for t in
                                      sorted(trades, key=lambda x: x['di'])]
                                rets_arr = np.array(ap) / CASH0
                                sh_val = (np.mean(rets_arr)
                                          / np.std(rets_arr) * np.sqrt(252)
                                          if np.std(rets_arr) > 0 else 0)
                                results.append({
                                    'vw': vwap_w, 'bw': bw,
                                    'tn': tn, 'mr': mr,
                                    'as': as_val, 'mc': mc,
                                    'ratio': ratio,
                                    'n': len(trades), 'wr': wr,
                                    'ann': ann, 'dd': dd,
                                    'sharpe': sh_val, 'eq': eq,
                                })

    results.sort(key=lambda x: -x['sharpe'])
    print(f"\n{'VW':>3} {'BW':>4} {'TN':>3} {'MR':>4} {'AS':>4} "
          f"{'MC':>3} {'Pyr':>4} "
          f"{'N':>5} {'WR':>5} {'Ann':>8} {'DD':>6} {'Sh':>5}")
    print("-" * 75)
    for r in results[:25]:
        print(f"{r['vw']:>3} {r['bw']:>4.1f} {r['tn']:>3} "
              f"{r['mr']:>4.2f} {r['as']:>4.1f} {r['mc']:>3} "
              f"{r['ratio']:>4.1f} "
              f"{r['n']:>5} {r['wr']:>5.1f} {r['ann']:>+8.1f} "
              f"{r['dd']:>6.1f} {r['sharpe']:>5.2f}")

    return results


def main():
    t0 = time.time()
    print("=" * 70)
    print("  V21: VOLUME PROFILE MEAN REVERSION")
    print("  VWAP bands + CS rank + KER gate + pyramid")
    print("=" * 70)

    C, O, H, L, V, OI, NS, ND, dates, syms = load_all_data(start='2016-01-01')
    print(f"  {NS} sym, {ND} days, "
          f"{dates[0].strftime('%Y-%m-%d')} to "
          f"{dates[-1].strftime('%Y-%m-%d')}")

    # Find 2019 start index for OOS testing
    bt_2019 = None
    for i, d in enumerate(dates):
        if d >= pd.Timestamp('2019-01-01'):
            bt_2019 = i
            break

    # === 1. Default config full 10-year ===
    print("\n" + "=" * 70)
    print("  FULL 10-YEAR (2016-2026) -- DEFAULT CONFIG")
    print("=" * 70)

    sigs_default = compute_all_signals(C, O, H, L, V, OI, NS, ND,
                                       vwap_window=20, band_width=1.5)

    for ratio in [0.0, 0.5]:
        label = f"V21-default pyr={ratio:.1f}"
        trades, eq, dd = backtest_v21(
            C, O, H, L, NS, ND, dates, syms, sigs_default,
            top_n=1, hold_days=5, atr_stop=3.0,
            min_rank=0.75, min_confidence=2, use_ker_gate=True,
            pyramid_ratio=ratio, pyramid_day=1,
            start_di=60)
        print(f"\n  {label}")
        analyze(trades, eq, dd, label)

    # === 2. Walk-Forward Validation ===
    print("\n" + "=" * 70)
    print("  WALK-FORWARD VALIDATION (2019-2026)")
    print("=" * 70)

    for ratio in [0.0, 0.5]:
        walk_forward(C, O, H, L, V, OI, NS, ND, dates, syms,
                     vwap_window=20, band_width=1.5,
                     top_n=1, min_rank=0.75, atr_stop=3.0,
                     min_confidence=2, pyramid_ratio=ratio,
                     pyramid_day=1)

    # === 3. Parameter Sweep (2019-2026 OOS) ===
    results = parameter_sweep(C, O, H, L, V, OI, NS, ND, dates, syms,
                              bt_2019)

    # === 4. Best config: full 10-year + walk-forward ===
    if results:
        best = results[0]
        print(f"\n{'=' * 70}")
        print(f"  BEST CONFIG: vw={best['vw']} bw={best['bw']} "
              f"tn={best['tn']} mr={best['mr']:.2f} "
              f"as={best['as']:.1f} mc={best['mc']} "
              f"pyr={best['ratio']:.1f} Sharpe={best['sharpe']:.2f}")
        print(f"{'=' * 70}")

        best_sigs = compute_all_signals(C, O, H, L, V, OI, NS, ND,
                                        vwap_window=best['vw'],
                                        band_width=best['bw'])

        # Full 10-year with best config
        for r in results[:5]:
            label = (f"vw={r['vw']} bw={r['bw']:.1f} tn={r['tn']} "
                     f"mr={r['mr']:.2f} as={r['as']:.1f} "
                     f"mc={r['mc']} pyr={r['ratio']:.1f}")
            trades, eq, dd = backtest_v21(
                C, O, H, L, NS, ND, dates, syms, best_sigs,
                top_n=r['tn'], hold_days=5, atr_stop=r['as'],
                min_rank=r['mr'], min_confidence=r['mc'],
                use_ker_gate=True,
                pyramid_ratio=r['ratio'], pyramid_day=1,
                start_di=60)
            print(f"\n  FULL 10y {label}")
            analyze(trades, eq, dd, label)

        # Walk-forward with best config
        print(f"\n  BEST WF:")
        walk_forward(C, O, H, L, V, OI, NS, ND, dates, syms,
                     vwap_window=best['vw'], band_width=best['bw'],
                     top_n=best['tn'], min_rank=best['mr'],
                     atr_stop=best['as'],
                     min_confidence=best['mc'],
                     pyramid_ratio=best['ratio'], pyramid_day=1)

    print(f"\n[V21] Done. {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
