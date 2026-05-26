"""
V17: MOMENTUM CRASH PROTECTION — Dynamic allocation based on market stress
===========================================================================
Core thesis: Mean-reversion strategies suffer during market stress/crash
periods when momentum dominates. By detecting cross-sectional stress levels,
we reduce exposure during high-volatility crashes and increase during calm
mean-reversion-friendly periods.

Architecture:
  1. Market stress indicator (cross-sectional return dispersion, avg ATR%,
     rolling std of market-wide returns)
  2. Regime classification: LOW_STRESS, NORMAL, HIGH_STRESS, CRASH
  3. Position sizing multiplier by regime
  4. Base signal: V1 multi-alpha oversold (consec_dn, ret5d, OI, VDP, RSI,
     BB, CCI) with KER gate and confidence >= 3
  5. ATR-based dynamic hold period (low vol -> 3d, high vol -> 7d, normal -> 5d)
  6. Walk-forward 2019-2026
  7. Parameter sweep over stress thresholds and size multipliers

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

# Regime constants
REGIME_LOW_STRESS = 0
REGIME_NORMAL = 1
REGIME_HIGH_STRESS = 2
REGIME_CRASH = 3


# ============================================================
# MARKET STRESS INDICATOR
# ============================================================
def compute_market_stress(C, H, L, NS, ND):
    """
    Compute market-wide stress indicators.
    Returns per-day arrays for stress metrics and regime classification.
    """
    t0 = time.time()
    print("[V17] Computing market stress indicators...", flush=True)

    # --- 1. Cross-sectional return dispersion (std of 5d returns) ---
    ret_5d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            if not np.isnan(C[si, di]) and not np.isnan(C[si, di - 5]) and C[si, di - 5] > 0:
                ret_5d[si, di] = C[si, di] / C[si, di - 5] - 1

    cs_dispersion = np.full(ND, np.nan)
    for di in range(ND):
        vals = ret_5d[:, di]
        valid = vals[~np.isnan(vals)]
        if len(valid) >= 10:
            cs_dispersion[di] = np.std(valid)

    # --- 2. Average ATR% across all commodities ---
    atr_pct = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(14, ND):
            atr_vals = []
            for j in range(di - 14, di):
                hh, ll, cc = H[si, j], L[si, j], C[si, j]
                if not any(np.isnan([hh, ll, cc])) and cc > 0:
                    atr_vals.append(max(hh - ll, abs(hh - cc), abs(ll - cc)))
            if atr_vals:
                atr_pct[si, di] = np.mean(atr_vals) / C[si, di]

    avg_atr_pct = np.full(ND, np.nan)
    for di in range(ND):
        vals = atr_pct[:, di]
        valid = vals[~np.isnan(vals)]
        if len(valid) >= 10:
            avg_atr_pct[di] = np.mean(valid)

    # --- 3. VIX-equivalent: rolling std of market-wide returns ---
    # Market-wide return = equal-weighted return across all commodities
    mkt_ret = np.full(ND, np.nan)
    for di in range(1, ND):
        rets = []
        for si in range(NS):
            if not np.isnan(C[si, di]) and not np.isnan(C[si, di - 1]) and C[si, di - 1] > 0:
                rets.append(C[si, di] / C[si, di - 1] - 1)
        if len(rets) >= 10:
            mkt_ret[di] = np.mean(rets)

    # Rolling 20d std of market returns (annualized)
    mkt_vol = np.full(ND, np.nan)
    for di in range(20, ND):
        window = mkt_ret[di - 20:di]
        valid = window[~np.isnan(window)]
        if len(valid) >= 10:
            mkt_vol[di] = np.std(valid) * np.sqrt(252)

    # --- 4. Composite stress z-score (normalized) ---
    # Use 120d rolling window to compute z-scores for each metric
    stress_zscore = np.full(ND, np.nan)
    for di in range(120, ND):
        scores = []

        # Dispersion z-score
        disp_w = cs_dispersion[di - 120:di]
        disp_valid = disp_w[~np.isnan(disp_w)]
        if len(disp_valid) >= 60 and not np.isnan(cs_dispersion[di]):
            mu, sig = np.mean(disp_valid), np.std(disp_valid)
            if sig > 0:
                scores.append((cs_dispersion[di] - mu) / sig)

        # ATR% z-score
        atr_w = avg_atr_pct[di - 120:di]
        atr_valid = atr_w[~np.isnan(atr_w)]
        if len(atr_valid) >= 60 and not np.isnan(avg_atr_pct[di]):
            mu, sig = np.mean(atr_valid), np.std(atr_valid)
            if sig > 0:
                scores.append((avg_atr_pct[di] - mu) / sig)

        # Market vol z-score
        vol_w = mkt_vol[di - 120:di]
        vol_valid = vol_w[~np.isnan(vol_w)]
        if len(vol_valid) >= 60 and not np.isnan(mkt_vol[di]):
            mu, sig = np.mean(vol_valid), np.std(vol_valid)
            if sig > 0:
                scores.append((mkt_vol[di] - mu) / sig)

        if len(scores) >= 2:
            stress_zscore[di] = np.mean(scores)

    print(f"  Stress indicators done: {time.time() - t0:.1f}s", flush=True)

    return {
        'cs_dispersion': cs_dispersion,
        'avg_atr_pct': avg_atr_pct,
        'mkt_vol': mkt_vol,
        'stress_zscore': stress_zscore,
    }


def classify_regime(stress_zscore, ND, crash_z=2.0, high_z=1.0, low_z=-0.5):
    """Classify each day into a stress regime."""
    regime = np.full(ND, REGIME_NORMAL, dtype=int)
    for di in range(ND):
        if np.isnan(stress_zscore[di]):
            continue
        z = stress_zscore[di]
        if z >= crash_z:
            regime[di] = REGIME_CRASH
        elif z >= high_z:
            regime[di] = REGIME_HIGH_STRESS
        elif z <= low_z:
            regime[di] = REGIME_LOW_STRESS
        else:
            regime[di] = REGIME_NORMAL
    return regime


# ============================================================
# V1-STYLE MULTI-ALPHA SIGNALS
# ============================================================
def compute_signals(C, O, H, L, V, OI, NS, ND):
    """Compute all signals for V17 (V1 multi-alpha lineage)."""
    t0 = time.time()
    print("[V17] Computing signals...", flush=True)

    # --- 1. Consecutive down days ---
    consec_dn = np.zeros((NS, ND), dtype=int)
    for si in range(NS):
        consec = 0
        for di in range(1, ND):
            if not np.isnan(C[si, di]) and not np.isnan(C[si, di - 1]) and C[si, di - 1] > 0:
                if C[si, di] < C[si, di - 1]:
                    consec += 1
                else:
                    consec = 0
            else:
                consec = 0
            consec_dn[si, di] = consec

    # --- 2. 5d return ---
    ret_5d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            if not np.isnan(C[si, di]) and not np.isnan(C[si, di - 5]) and C[si, di - 5] > 0:
                ret_5d[si, di] = C[si, di] / C[si, di - 5] - 1

    # --- 3. 20d volatility ---
    vol_20d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            rets = []
            for j in range(di - 20, di):
                if not np.isnan(C[si, j]) and not np.isnan(C[si, j - 1]) and C[si, j - 1] > 0:
                    rets.append(C[si, j] / C[si, j - 1] - 1)
            if len(rets) >= 10:
                vol_20d[si, di] = np.std(rets) * np.sqrt(252)

    # --- 4. OI capitulation ---
    oi_decline = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            if np.isnan(OI[si, di]) or np.isnan(OI[si, di - 5]):
                continue
            if np.isnan(C[si, di]) or np.isnan(C[si, di - 5]) or C[si, di - 5] <= 0:
                continue
            oi_chg = OI[si, di] / OI[si, di - 5] - 1
            price_chg = C[si, di] / C[si, di - 5] - 1
            if oi_chg < -0.02 and price_chg < -0.02:
                oi_decline[si, di] = min(abs(oi_chg), 0.2) / 0.2 * min(abs(price_chg), 0.1) / 0.1
            else:
                oi_decline[si, di] = 0.0

    # --- 5. VDP (Volume Delta Pressure) ---
    vdp = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(1, ND):
            if np.isnan(H[si, di]) or np.isnan(L[si, di]) or np.isnan(C[si, di]) or np.isnan(V[si, di]):
                continue
            bar_range = H[si, di] - L[si, di]
            if bar_range > 0 and V[si, di] > 0:
                vdp[si, di] = V[si, di] * (2 * C[si, di] - H[si, di] - L[si, di]) / bar_range

    vdp_10 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(10, ND):
            vals = vdp[si, di - 10:di]
            valid = vals[~np.isnan(vals)]
            if len(valid) >= 5:
                vdp_10[si, di] = np.mean(valid)

    vdp_exhaust = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            if np.isnan(vdp_10[si, di]):
                continue
            window = vdp_10[si, max(0, di - 20):di]
            wv = window[~np.isnan(window)]
            if len(wv) >= 10:
                mu, sig = np.mean(wv), np.std(wv)
                if sig > 0:
                    z = (vdp_10[si, di] - mu) / sig
                    vdp_exhaust[si, di] = min(-z, 3.0) / 3.0 if z < 0 else 0.0

    # --- 6. KER (Kaufman Efficiency Ratio) ---
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
                ker_regime[si, di] = 1  # mean-reverting
            elif ker_10[si, di] > 0.3:
                ker_regime[si, di] = -1  # trending

    # --- 7. TA-Lib indicators ---
    rsi14 = np.full((NS, ND), np.nan)
    bb_pos = np.full((NS, ND), np.nan)
    cci14 = np.full((NS, ND), np.nan)

    if HAS_TALIB:
        for si in range(NS):
            h = np.where(np.isnan(H[si]), 0, H[si]).astype(np.float64)
            l = np.where(np.isnan(L[si]), 0, L[si]).astype(np.float64)
            c = np.where(np.isnan(C[si]), 0, C[si]).astype(np.float64)
            nan_mask = np.isnan(C[si])
            try:
                rsi = talib.RSI(c, 14)
                rsi14[si] = np.where(nan_mask, np.nan, rsi)
            except Exception:
                pass
            try:
                upper, mid, lower = talib.BBANDS(c, 20, 2.0, 2.0)
                bb_range = upper - lower
                valid_bb = (~nan_mask) & (bb_range > 1e-10)
                bb_pos[si] = np.where(valid_bb, (c - lower) / bb_range, np.nan)
            except Exception:
                pass
            try:
                cci = talib.CCI(h, l, c, 14)
                cci14[si] = np.where(nan_mask, np.nan, cci)
            except Exception:
                pass

    # --- 8. Per-instrument ATR for dynamic hold period ---
    atr_14 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(14, ND):
            atr_vals = []
            for j in range(di - 14, di):
                hh, ll, cc = H[si, j], L[si, j], C[si, j]
                if not any(np.isnan([hh, ll, cc])):
                    atr_vals.append(max(hh - ll, abs(hh - cc), abs(ll - cc)))
            if atr_vals:
                atr_14[si, di] = np.mean(atr_vals)

    # --- 9. Composite score with cross-sectional ranking ---
    raw_score = np.full((NS, ND), np.nan)
    for di in range(ND):
        scores = np.full(NS, np.nan)
        for si in range(NS):
            if np.isnan(C[si, di]) or C[si, di] <= 0:
                continue

            s = 0.0
            w_total = 0.0

            # Consecutive down days (weight 0.20)
            cd = consec_dn[si, di]
            s += min(cd / 5.0, 1.0) * 0.20
            w_total += 0.20

            # 5d return oversold (weight 0.20)
            if not np.isnan(ret_5d[si, di]):
                s += min(max(-ret_5d[si, di] / 0.1, 0), 1.0) * 0.20
                w_total += 0.20

            # OI capitulation (weight 0.20)
            if not np.isnan(oi_decline[si, di]):
                s += oi_decline[si, di] * 0.20
                w_total += 0.20

            # VDP selling exhaustion (weight 0.15)
            if not np.isnan(vdp_exhaust[si, di]):
                s += vdp_exhaust[si, di] * 0.15
                w_total += 0.15

            # RSI oversold (weight 0.10)
            if not np.isnan(rsi14[si, di]):
                if rsi14[si, di] < 30:
                    s += (30 - rsi14[si, di]) / 30.0 * 0.10
                w_total += 0.10

            # Bollinger lower band (weight 0.10)
            if not np.isnan(bb_pos[si, di]):
                if bb_pos[si, di] < 0.2:
                    s += (0.2 - bb_pos[si, di]) / 0.2 * 0.10
                w_total += 0.10

            # CCI oversold (weight 0.05)
            if not np.isnan(cci14[si, di]):
                if cci14[si, di] < -100:
                    s += min((-100 - cci14[si, di]) / 200.0, 1.0) * 0.05
                w_total += 0.05

            if w_total > 0:
                scores[si] = s / w_total

        valid = ~np.isnan(scores)
        if valid.sum() >= 5:
            raw_score[:, di] = pd.Series(scores).rank(pct=True, na_option='keep').values

    # --- 10. Confidence level ---
    n_signals = np.zeros((NS, ND), dtype=int)
    for di in range(ND):
        for si in range(NS):
            if np.isnan(C[si, di]) or C[si, di] <= 0:
                continue
            n = 0
            if consec_dn[si, di] >= 3:
                n += 1
            if not np.isnan(ret_5d[si, di]) and ret_5d[si, di] < -0.03:
                n += 1
            if not np.isnan(oi_decline[si, di]) and oi_decline[si, di] > 0.1:
                n += 1
            if not np.isnan(vdp_exhaust[si, di]) and vdp_exhaust[si, di] > 0.3:
                n += 1
            if not np.isnan(rsi14[si, di]) and rsi14[si, di] < 35:
                n += 1
            if not np.isnan(bb_pos[si, di]) and bb_pos[si, di] < 0.15:
                n += 1
            if not np.isnan(cci14[si, di]) and cci14[si, di] < -100:
                n += 1
            n_signals[si, di] = n

    elapsed = time.time() - t0
    print(f"  Signals done: {elapsed:.1f}s (TA-Lib: {HAS_TALIB})", flush=True)

    return {
        'combo_rank': raw_score,
        'consec_dn': consec_dn,
        'vol_20d': vol_20d,
        'ker_regime': ker_regime,
        'n_signals': n_signals,
        'atr_14': atr_14,
    }


# ============================================================
# DYNAMIC HOLD PERIOD
# ============================================================
def compute_dynamic_hold(atr_14, C, NS, ND, window=60):
    """
    ATR-based dynamic hold period per instrument per day.
    Low vol -> hold 3d, High vol -> hold 7d, Normal -> hold 5d.
    """
    hold_map = np.full((NS, ND), 5, dtype=int)  # default 5d

    for si in range(NS):
        for di in range(window, ND):
            atr_w = atr_14[si, di - window:di]
            atr_valid = atr_w[~np.isnan(atr_w)]
            if len(atr_valid) < 30:
                continue
            if np.isnan(atr_14[si, di]) or np.isnan(C[si, di]) or C[si, di] <= 0:
                continue

            current_atr_pct = atr_14[si, di] / C[si, di]
            avg_atr_pct = np.mean(atr_valid)

            if avg_atr_pct > 0:
                ratio = current_atr_pct / avg_atr_pct
                if ratio < 0.7:
                    hold_map[si, di] = 3  # low vol -> quick MR
                elif ratio > 1.3:
                    hold_map[si, di] = 7  # high vol -> more time
                else:
                    hold_map[si, di] = 5  # normal
            else:
                hold_map[si, di] = 5

    return hold_map


# ============================================================
# BACKTEST ENGINE
# ============================================================
def backtest_v17(C, O, H, L, NS, ND, dates, syms, sigs, regime,
                 hold_map,
                 top_n=1, min_rank=0.7, atr_stop=2.5,
                 min_confidence=3, use_ker_gate=True,
                 size_multipliers=None,
                 use_dynamic_hold=True,
                 start_di=60, end_di=None):
    """
    V17 backtest with market-stress-based position sizing.
    Signal at close[di], enter at open[di+1]. No look-ahead.
    """
    combo_rank = sigs['combo_rank']
    ker_regime = sigs['ker_regime']
    n_signals = sigs['n_signals']

    if size_multipliers is None:
        size_multipliers = {
            REGIME_LOW_STRESS: 1.5,
            REGIME_NORMAL: 1.0,
            REGIME_HIGH_STRESS: 0.5,
            REGIME_CRASH: 0.0,  # skip new entries
        }

    if end_di is None:
        end_di = ND - 1

    equity = CASH0
    peak = equity
    max_dd = 0.0
    positions = []  # (si, edi, ep, sp, alloc, direction, hold_target)
    trades = []

    for di in range(max(start_di, 1), end_di):
        d = dates[di]
        daily_pnl = 0
        new_positions = []

        # Exit positions
        for si, edi, ep, sp, alloc, direction, hold_target in positions:
            c = C[si, di]
            if np.isnan(c):
                new_positions.append((si, edi, ep, sp, alloc, direction, hold_target))
                continue
            exit_r = None
            if direction > 0 and c < sp:
                exit_r = 'stop'
            elif direction < 0 and c > sp:
                exit_r = 'stop'
            elif use_dynamic_hold and (di - edi >= hold_target):
                exit_r = 'hold'
            elif (not use_dynamic_hold) and (di - edi >= 5):
                exit_r = 'hold'
            if exit_r:
                pnl = direction * (c - ep) / ep - COMM
                profit = equity * alloc * pnl
                daily_pnl += profit
                trades.append({
                    'pnl_abs': profit, 'pnl_pct': pnl * 100,
                    'days': di - edi + 1, 'di': di, 'year': d.year,
                    'sym': syms[si], 'reason': exit_r, 'dir': direction,
                    'regime': int(regime[di]),
                })
            else:
                new_positions.append((si, edi, ep, sp, alloc, direction, hold_target))

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

        # Current market regime
        current_regime = int(regime[di])
        size_mult = size_multipliers.get(current_regime, 1.0)

        # CRASH regime: skip new entries
        if size_mult <= 0:
            continue

        # --- Long candidates ---
        candidates = []
        for si in range(NS):
            if si in held:
                continue
            if np.isnan(combo_rank[si, di]):
                continue
            if combo_rank[si, di] < min_rank:
                continue
            if n_signals[si, di] < min_confidence:
                continue
            if use_ker_gate and ker_regime[si, di] < 0:
                continue
            if di + 1 >= ND or np.isnan(O[si, di + 1]):
                continue

            # Position sizing with regime multiplier
            alloc = size_mult / max(top_n, 1)

            candidates.append((combo_rank[si, di], si, alloc))

        candidates.sort(key=lambda x: -x[0])
        for rank, si, alloc in candidates[:top_n]:
            if len(positions) >= top_n or si in held:
                break
            ep = O[si, di + 1]
            if np.isnan(ep) or ep <= 0:
                continue
            # ATR stop
            atr_v = []
            for j in range(max(start_di, di - 14), di):
                hh, ll, cc = H[si, j], L[si, j], C[si, j]
                if not any(np.isnan([hh, ll, cc])):
                    atr_v.append(max(hh - ll, abs(hh - cc), abs(ll - cc)))
            if not atr_v:
                continue
            atr = np.mean(atr_v)

            # Dynamic hold period
            if use_dynamic_hold:
                hold_target = int(hold_map[si, di]) if not np.isnan(hold_map[si, di]) else 5
            else:
                hold_target = 5

            positions.append((si, di + 1, ep, ep - atr_stop * atr, alloc, 1, hold_target))
            held.add(si)

    # Close remaining
    for si, edi, ep, sp, alloc, direction, hold_target in positions:
        c = C[si, ND - 1]
        if not np.isnan(c) and c > 0:
            pnl = direction * (c - ep) / ep - COMM
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

    n_stop = sum(1 for t in trades if t['reason'] == 'stop')
    n_hold = sum(1 for t in trades if t['reason'] == 'hold')

    # Regime breakdown
    regime_pnl = defaultdict(list)
    for t in trades:
        r = t.get('regime', REGIME_NORMAL)
        regime_pnl[r].append(t['pnl_pct'])

    regime_names = {
        REGIME_LOW_STRESS: 'LOW',
        REGIME_NORMAL: 'NORMAL',
        REGIME_HIGH_STRESS: 'HIGH',
        REGIME_CRASH: 'CRASH',
    }

    print(f"  {label}: {len(trades)}t (stop:{n_stop} hold:{n_hold}) "
          f"WR={wr:.1f}% ann={ann:+.1f}% DD={max_dd:.1f}% "
          f"Sh={sh:.2f} eq={equity:,.0f}")

    # Per-regime stats
    for r in [REGIME_LOW_STRESS, REGIME_NORMAL, REGIME_HIGH_STRESS, REGIME_CRASH]:
        rp = regime_pnl.get(r, [])
        if rp:
            rwr = sum(1 for p in rp if p > 0) / len(rp) * 100
            print(f"    {regime_names.get(r, '?'):>6}: {len(rp):>3}t WR={rwr:.1f}% "
                  f"avg={np.mean(rp):+.2f}%")

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
# WALK-FORWARD VALIDATION
# ============================================================
def walk_forward(C, O, H, L, NS, ND, dates, syms, sigs, hold_map,
                 size_multipliers, top_n=1, min_confidence=3,
                 atr_stop=2.5, use_dynamic_hold=True):
    """Walk-forward validation: year-by-year out-of-sample."""
    print(f"\n{'=' * 70}")
    print(f"  WALK-FORWARD V17 (tn={top_n} mc={min_confidence} "
          f"dyn_hold={use_dynamic_hold})")
    sm = size_multipliers
    print(f"  Size mult: LOW={sm.get(0,0):.1f} NORM={sm.get(1,0):.1f} "
          f"HIGH={sm.get(2,0):.1f} CRASH={sm.get(3,0):.1f}")
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

        trades, _, _ = backtest_v17(
            C, O, H, L, NS, ND, dates, syms, sigs,
            regime=classify_regime(
                compute_market_stress(C, H, L, NS, ND)['stress_zscore'],
                ND),
            hold_map=hold_map,
            top_n=top_n, min_rank=0.7, atr_stop=atr_stop,
            min_confidence=min_confidence, use_ker_gate=True,
            size_multipliers=size_multipliers,
            use_dynamic_hold=use_dynamic_hold,
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
        print(f"\n  WF TOTAL: {len(all_trades)}t WR={wr_val:.1f}% "
              f"avg={avg:+.2f}% cum={cum:+.1%}")
        return all_trades
    return []


# ============================================================
# MAIN
# ============================================================
def main():
    t0 = time.time()
    print("=" * 70)
    print("  V17: MOMENTUM CRASH PROTECTION")
    print("  Dynamic allocation based on market stress")
    print("=" * 70)

    C, O, H, L, V, OI, NS, ND, dates, syms = load_all_data(start='2016-01-01')
    print(f"  {NS} sym, {ND} days, "
          f"{dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")

    # Compute signals
    sigs = compute_signals(C, O, H, L, V, OI, NS, ND)
    stress_data = compute_market_stress(C, H, L, NS, ND)
    hold_map = compute_dynamic_hold(sigs['atr_14'], C, NS, ND)

    # Find 2019 start index
    bt_2019 = None
    for i, d in enumerate(dates):
        if d >= pd.Timestamp('2019-01-01'):
            bt_2019 = i
            break

    # === 1. Default regime thresholds ===
    print("\n" + "=" * 70)
    print("  SECTION 1: DEFAULT REGIME (crash_z=2.0, high_z=1.0, low_z=-0.5)")
    print("=" * 70)

    regime_default = classify_regime(stress_data['stress_zscore'], ND)
    regime_names = {0: 'LOW', 1: 'NORMAL', 2: 'HIGH', 3: 'CRASH'}
    for r in [0, 1, 2, 3]:
        count = (regime_default == r).sum()
        print(f"  {regime_names[r]:>6}: {count:>5} days")

    default_mult = {0: 1.5, 1: 1.0, 2: 0.5, 3: 0.0}

    configs = [
        # (label, size_multipliers, dynamic_hold, min_conf, top_n, atr_stop)
        ("V1 baseline (no stress)", {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0}, False, 3, 1, 2.5),
        ("V17 stress gate (1.5/1.0/0.5/0)", default_mult, False, 3, 1, 2.5),
        ("V17 stress+dynhold", default_mult, True, 3, 1, 2.5),
        ("V17 aggressive (2.0/1.0/0.3/0)", {0: 2.0, 1: 1.0, 2: 0.3, 3: 0.0}, True, 3, 1, 2.5),
        ("V17 conservative (1.2/1.0/0.7/0)", {0: 1.2, 1: 1.0, 2: 0.7, 3: 0.0}, True, 3, 1, 2.5),
        ("V17 conf=2", default_mult, True, 2, 1, 2.5),
        ("V17 conf=4", default_mult, True, 4, 1, 2.5),
        ("V17 tn=2", default_mult, True, 3, 2, 2.5),
        ("V17 atr=3.0", default_mult, True, 3, 1, 3.0),
    ]

    for label, sm, dh, mc, tn, ats in configs:
        trades, eq, dd = backtest_v17(
            C, O, H, L, NS, ND, dates, syms, sigs,
            regime=regime_default, hold_map=hold_map,
            top_n=tn, min_rank=0.7, atr_stop=ats,
            min_confidence=mc, use_ker_gate=True,
            size_multipliers=sm, use_dynamic_hold=dh,
            start_di=bt_2019)
        analyze(trades, eq, dd, label)

    # === 2. Regime threshold sweep ===
    print("\n" + "=" * 70)
    print("  SECTION 2: REGIME THRESHOLD SWEEP (2019-2026)")
    print("=" * 70)

    results_threshold = []
    for crash_z in [1.5, 2.0, 2.5, 3.0]:
        for high_z in [0.5, 1.0, 1.5]:
            for low_z in [-1.0, -0.5, 0.0]:
                if crash_z <= high_z:
                    continue
                regime_sw = classify_regime(
                    stress_data['stress_zscore'], ND,
                    crash_z=crash_z, high_z=high_z, low_z=low_z)
                trades, eq, dd = backtest_v17(
                    C, O, H, L, NS, ND, dates, syms, sigs,
                    regime=regime_sw, hold_map=hold_map,
                    top_n=1, min_rank=0.7, atr_stop=2.5,
                    min_confidence=3, use_ker_gate=True,
                    size_multipliers=default_mult,
                    use_dynamic_hold=True,
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
                results_threshold.append({
                    'crash_z': crash_z, 'high_z': high_z, 'low_z': low_z,
                    'n': len(trades), 'wr': wr, 'ann': ann, 'dd': dd,
                    'sharpe': sh_val, 'eq': eq,
                })

    results_threshold.sort(key=lambda x: -x['sharpe'])
    print(f"\n{'CrashZ':>6} {'HighZ':>6} {'LowZ':>6} "
          f"{'N':>5} {'WR':>5} {'Ann':>8} {'DD':>6} {'Sh':>5}")
    print("-" * 60)
    for r in results_threshold[:20]:
        print(f"{r['crash_z']:>6.1f} {r['high_z']:>6.1f} {r['low_z']:>6.1f} "
              f"{r['n']:>5} {r['wr']:>5.1f} {r['ann']:>+8.1f} "
              f"{r['dd']:>6.1f} {r['sharpe']:>5.2f}")

    # === 3. Size multiplier sweep ===
    print("\n" + "=" * 70)
    print("  SECTION 3: SIZE MULTIPLIER SWEEP (2019-2026)")
    print("=" * 70)

    results_size = []
    for low_m in [1.0, 1.2, 1.5, 2.0]:
        for high_m in [0.3, 0.5, 0.7]:
            for crash_m in [0.0, 0.1]:
                sm = {0: low_m, 1: 1.0, 2: high_m, 3: crash_m}
                trades, eq, dd = backtest_v17(
                    C, O, H, L, NS, ND, dates, syms, sigs,
                    regime=regime_default, hold_map=hold_map,
                    top_n=1, min_rank=0.7, atr_stop=2.5,
                    min_confidence=3, use_ker_gate=True,
                    size_multipliers=sm, use_dynamic_hold=True,
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
                results_size.append({
                    'low_m': low_m, 'high_m': high_m, 'crash_m': crash_m,
                    'n': len(trades), 'wr': wr, 'ann': ann, 'dd': dd,
                    'sharpe': sh_val, 'eq': eq,
                })

    results_size.sort(key=lambda x: -x['sharpe'])
    print(f"\n{'LowM':>5} {'HighM':>5} {'CrashM':>6} "
          f"{'N':>5} {'WR':>5} {'Ann':>8} {'DD':>6} {'Sh':>5}")
    print("-" * 55)
    for r in results_size[:20]:
        print(f"{r['low_m']:>5.1f} {r['high_m']:>5.1f} {r['crash_m']:>6.1f} "
              f"{r['n']:>5} {r['wr']:>5.1f} {r['ann']:>+8.1f} "
              f"{r['dd']:>6.1f} {r['sharpe']:>5.2f}")

    # === 4. Combined best config sweep ===
    print("\n" + "=" * 70)
    print("  SECTION 4: COMBINED PARAMETER SWEEP (2019-2026)")
    print("=" * 70)

    results = []
    for tn in [1, 2, 3]:
        for mc in [2, 3]:
            for ats in [2.5, 3.0]:
                for dh in [True, False]:
                    for crash_m in [0.0, 0.1]:
                        sm = {0: 1.5, 1: 1.0, 2: 0.5, 3: crash_m}
                        trades, eq, dd = backtest_v17(
                            C, O, H, L, NS, ND, dates, syms, sigs,
                            regime=regime_default, hold_map=hold_map,
                            top_n=tn, min_rank=0.7, atr_stop=ats,
                            min_confidence=mc, use_ker_gate=True,
                            size_multipliers=sm, use_dynamic_hold=dh,
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
                            'tn': tn, 'mc': mc, 'ats': ats,
                            'dh': dh, 'cm': crash_m,
                            'n': len(trades), 'wr': wr, 'ann': ann, 'dd': dd,
                            'sharpe': sh_val, 'eq': eq,
                        })

    results.sort(key=lambda x: -x['sharpe'])
    print(f"\n{'TN':>3} {'MC':>3} {'ATR':>4} {'DH':>3} {'CM':>4} "
          f"{'N':>5} {'WR':>5} {'Ann':>8} {'DD':>6} {'Sh':>5}")
    print("-" * 60)
    for r in results[:25]:
        print(f"{r['tn']:>3} {r['mc']:>3} {r['ats']:>4.1f} "
              f"{'Y' if r['dh'] else 'N':>3} {r['cm']:>4.1f} "
              f"{r['n']:>5} {r['wr']:>5.1f} {r['ann']:>+8.1f} "
              f"{r['dd']:>6.1f} {r['sharpe']:>5.2f}")

    # === 5. Best configs — full 10-year ===
    print("\n" + "=" * 70)
    print("  SECTION 5: BEST CONFIGS — FULL 10-YEAR (2016-2026)")
    print("=" * 70)

    best = results[:5]
    for r in best:
        sm = {0: 1.5, 1: 1.0, 2: 0.5, 3: r['cm']}
        trades, eq, dd = backtest_v17(
            C, O, H, L, NS, ND, dates, syms, sigs,
            regime=regime_default, hold_map=hold_map,
            top_n=r['tn'], min_rank=0.7, atr_stop=r['ats'],
            min_confidence=r['mc'], use_ker_gate=True,
            size_multipliers=sm, use_dynamic_hold=r['dh'],
            start_di=60)
        label = (f"tn={r['tn']} mc={r['mc']} atr={r['ats']} "
                 f"dh={'Y' if r['dh'] else 'N'} cm={r['cm']}")
        print(f"\n  FULL {label}")
        analyze(trades, eq, dd, label)

    # === 6. Walk-forward for best config ===
    if best:
        best_r = best[0]
        sm = {0: 1.5, 1: 1.0, 2: 0.5, 3: best_r['cm']}
        print("\n" + "=" * 70)
        print(f"  SECTION 6: WALK-FORWARD BEST CONFIG")
        print(f"  tn={best_r['tn']} mc={best_r['mc']} atr={best_r['ats']} "
              f"dh={'Y' if best_r['dh'] else 'N'} cm={best_r['cm']}")
        print("=" * 70)

        walk_forward(C, O, H, L, NS, ND, dates, syms, sigs, hold_map,
                     size_multipliers=sm,
                     top_n=best_r['tn'], min_confidence=best_r['mc'],
                     atr_stop=best_r['ats'], use_dynamic_hold=best_r['dh'])

    # === 7. Walk-forward for top-3 configs ===
    print("\n" + "=" * 70)
    print("  SECTION 7: WALK-FORWARD TOP-3 CONFIGS")
    print("=" * 70)

    for r in best[:3]:
        sm = {0: 1.5, 1: 1.0, 2: 0.5, 3: r['cm']}
        label = (f"tn={r['tn']} mc={r['mc']} atr={r['ats']} "
                 f"dh={'Y' if r['dh'] else 'N'} cm={r['cm']}")
        print(f"\n  --- {label} ---")
        wf_trades = walk_forward(C, O, H, L, NS, ND, dates, syms, sigs, hold_map,
                                 size_multipliers=sm,
                                 top_n=r['tn'], min_confidence=r['mc'],
                                 atr_stop=r['ats'], use_dynamic_hold=r['dh'])
        if wf_trades:
            cum = np.prod([1 + t['pnl_pct'] / 100 for t in wf_trades]) - 1
            print(f"  WF CUMULATIVE: {cum:+.1%}")

    print(f"\n[V17] Done. {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
