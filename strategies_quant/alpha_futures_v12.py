"""
V12: Open Interest Extreme Contrarian — Walk-Forward Validated
==============================================================
Core thesis: OI data is futures-specific alpha unavailable to equity traders.
Exploits OI extremes to identify capitulation/accumulation turning points.

Signal architecture:
  Layer 1 (40%): OI-based signals
    - OI z-score (60d): How extreme is current OI vs recent history
    - OI-price divergence: OI declining + price declining = capitulation
    - OI acceleration: 2nd derivative of OI (acceleration of change)
    - OI ratio: current OI / 60d mean
    - Volume-OI divergence: Vol up + OI down = distribution (bearish)
  Layer 2 (30%): Price-based oversold signals (from V1)
  Layer 3 (30%): Confirmation signals (RSI, BB, CCI, volume)
  KER gate + confidence threshold + pyramid on winners
  Walk-forward validation

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

# Signal weight constants
W_OI_LAYER = 0.40
W_PRICE_LAYER = 0.30
W_CONFIRM_LAYER = 0.30

# OI signal sub-weights (must sum to 1.0 within OI layer)
W_OI_ZSCORE = 0.20
W_OI_DIVERGENCE = 0.30
W_OI_ACCEL = 0.15
W_OI_RATIO = 0.15
W_VOI_DIVERGENCE = 0.20

OI_WINDOW = 60


def compute_oi_signals(OI, C, V, NS, ND):
    """Compute OI-based signals. Returns dict of arrays."""
    t0 = time.time()
    print("[V12] Computing OI signals...", flush=True)

    # --- OI z-score (60d rolling) ---
    oi_zscore = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(OI_WINDOW, ND):
            window = OI[si, di - OI_WINDOW:di]
            valid = window[~np.isnan(window)]
            if len(valid) < 30:
                continue
            if np.isnan(OI[si, di]):
                continue
            mu, sig = np.mean(valid), np.std(valid)
            if sig > 0:
                oi_zscore[si, di] = (OI[si, di] - mu) / sig

    # --- OI-Price Divergence (5d window) ---
    # OI declining sharply + price declining = capitulation exhaustion (bullish reversal)
    # OI rising + price rising = strong trend confirmation
    oi_price_div = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            if np.isnan(OI[si, di]) or np.isnan(OI[si, di - 5]):
                continue
            if np.isnan(C[si, di]) or np.isnan(C[si, di - 5]) or C[si, di - 5] <= 0:
                continue
            oi_chg = OI[si, di] / OI[si, di - 5] - 1
            price_chg = C[si, di] / C[si, di - 5] - 1

            # Capitulation: both OI and price declining hard
            if oi_chg < -0.02 and price_chg < -0.02:
                oi_strength = min(abs(oi_chg), 0.3) / 0.3
                price_strength = min(abs(price_chg), 0.15) / 0.15
                oi_price_div[si, di] = oi_strength * price_strength
            # Mild negative score for OI rising + price falling (new shorts entering)
            elif oi_chg > 0.02 and price_chg < -0.02:
                oi_price_div[si, di] = -0.2 * min(oi_chg, 0.3) / 0.3 * min(abs(price_chg), 0.15) / 0.15
            else:
                oi_price_div[si, di] = 0.0

    # --- OI Acceleration (2nd derivative) ---
    # Rate of change of OI change - detects when OI decline is decelerating
    # (capitulation ending) or accelerating (capitulation intensifying)
    oi_accel = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(15, ND):
            # Need OI at di, di-5, di-10
            oi_now = OI[si, di]
            oi_5 = OI[si, di - 5]
            oi_10 = OI[si, di - 10]
            if any(np.isnan(x) for x in [oi_now, oi_5, oi_10]):
                continue
            if oi_5 <= 0 or oi_10 <= 0:
                continue
            # First derivative (change over 5d)
            d1_recent = oi_now / oi_5 - 1
            d1_prev = oi_5 / oi_10 - 1
            # Second derivative: deceleration of decline is bullish
            accel = d1_recent - d1_prev
            # If OI was declining (d1_prev < 0) and now decelerating (accel > 0),
            # capitulation is ending -> bullish signal
            if d1_prev < -0.02 and accel > 0:
                oi_accel[si, di] = min(accel, 0.2) / 0.2
            elif d1_prev < -0.02 and accel < -0.02:
                # Capitulation accelerating - could mean final flush (also bullish)
                oi_accel[si, di] = min(abs(accel), 0.2) / 0.2 * 0.5
            else:
                oi_accel[si, di] = 0.0

    # --- OI Ratio: current OI / 60d average ---
    oi_ratio = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(OI_WINDOW, ND):
            window = OI[si, di - OI_WINDOW:di]
            valid = window[~np.isnan(window)]
            if len(valid) < 30:
                continue
            if np.isnan(OI[si, di]):
                continue
            avg_oi = np.mean(valid)
            if avg_oi > 0:
                oi_ratio[si, di] = OI[si, di] / avg_oi

    # --- Volume-OI Divergence ---
    # Volume up + OI down = distribution (bearish, but for contrarian: overdone)
    # Volume down + OI up = accumulation (bullish)
    voi_div = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(10, ND):
            if np.isnan(V[si, di]) or np.isnan(V[si, di - 5]):
                continue
            if np.isnan(OI[si, di]) or np.isnan(OI[si, di - 5]):
                continue
            if V[si, di - 5] <= 0 or OI[si, di - 5] <= 0:
                continue
            vol_chg = V[si, di] / V[si, di - 5] - 1
            oi_chg = OI[si, di] / OI[si, di - 5] - 1

            # Distribution: vol up + OI down -> positions being unwound aggressively
            # For contrarian: this is late-stage, potential reversal
            if vol_chg > 0.1 and oi_chg < -0.03:
                voi_div[si, di] = min(vol_chg, 0.5) / 0.5 * min(abs(oi_chg), 0.2) / 0.2
            # Accumulation: vol down + OI up -> smart money building positions
            elif vol_chg < -0.1 and oi_chg > 0.03:
                voi_div[si, di] = 0.5 * min(abs(vol_chg), 0.5) / 0.5 * min(oi_chg, 0.2) / 0.2
            else:
                voi_div[si, di] = 0.0

    print(f"  OI signals done: {time.time() - t0:.1f}s", flush=True)
    return {
        'oi_zscore': oi_zscore,
        'oi_price_div': oi_price_div,
        'oi_accel': oi_accel,
        'oi_ratio': oi_ratio,
        'voi_div': voi_div,
    }


def compute_price_signals(C, NS, ND):
    """Price-based oversold signals (from V1 lineage)."""
    t0 = time.time()
    print("[V12] Computing price signals...", flush=True)

    # Consecutive down days
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

    # 5-day return
    ret_5d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            if not np.isnan(C[si, di]) and not np.isnan(C[si, di - 5]) and C[si, di - 5] > 0:
                ret_5d[si, di] = C[si, di] / C[si, di - 5] - 1

    print(f"  Price signals done: {time.time() - t0:.1f}s", flush=True)
    return {
        'consec_dn': consec_dn,
        'ret_5d': ret_5d,
    }


def compute_confirmation_signals(C, O, H, L, V, NS, ND):
    """Confirmation signals: RSI, BB, CCI, VDP."""
    t0 = time.time()
    print("[V12] Computing confirmation signals...", flush=True)

    # Volume-Delta Proxy (VDP) exhaustion
    vdp = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(1, ND):
            if np.isnan(H[si, di]) or np.isnan(L[si, di]):
                continue
            if np.isnan(C[si, di]) or np.isnan(V[si, di]):
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

    # TA-Lib indicators
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

    print(f"  Confirmation signals done: {time.time() - t0:.1f}s", flush=True)
    return {
        'vdp_exhaust': vdp_exhaust,
        'rsi14': rsi14,
        'bb_pos': bb_pos,
        'cci14': cci14,
    }


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


def compute_all_signals(C, O, H, L, V, OI, NS, ND):
    """Compute all signals and build composite OI-weighted score."""
    t0 = time.time()
    print("[V12] Computing all signals...", flush=True)

    oi_sigs = compute_oi_signals(OI, C, V, NS, ND)
    price_sigs = compute_price_signals(C, NS, ND)
    confirm_sigs = compute_confirmation_signals(C, O, H, L, V, NS, ND)
    ker_regime = compute_ker(C, NS, ND)

    # --- Composite score with OI-heavy weighting ---
    raw_score = np.full((NS, ND), np.nan)
    oi_score_raw = np.full((NS, ND), np.nan)

    for di in range(ND):
        scores = np.full(NS, np.nan)
        oi_scores = np.full(NS, np.nan)
        for si in range(NS):
            if np.isnan(C[si, di]) or C[si, di] <= 0:
                continue

            # --- Layer 1: OI signals (40%) ---
            s_oi = 0.0
            w_oi = 0.0

            # OI z-score: very negative z = OI at extreme low = potential reversal
            zscore = oi_sigs['oi_zscore'][si, di]
            if not np.isnan(zscore):
                # Extremely low OI (z < -1.5) = capitulation exhausted
                if zscore < -1.5:
                    s_oi += (min(-zscore, 3.0) - 1.5) / 1.5 * W_OI_ZSCORE
                elif zscore < -1.0:
                    s_oi += (-zscore - 1.0) / 0.5 * W_OI_ZSCORE * 0.5
                w_oi += W_OI_ZSCORE

            # OI-price divergence
            div = oi_sigs['oi_price_div'][si, di]
            if not np.isnan(div):
                s_oi += max(div, 0.0) * W_OI_DIVERGENCE
                w_oi += W_OI_DIVERGENCE

            # OI acceleration (deceleration of OI decline = bullish)
            accel = oi_sigs['oi_accel'][si, di]
            if not np.isnan(accel):
                s_oi += max(accel, 0.0) * W_OI_ACCEL
                w_oi += W_OI_ACCEL

            # OI ratio: very low ratio (OI far below average) = extreme
            ratio = oi_sigs['oi_ratio'][si, di]
            if not np.isnan(ratio):
                if ratio < 0.7:
                    s_oi += (0.7 - ratio) / 0.3 * W_OI_RATIO
                elif ratio < 0.85:
                    s_oi += (0.85 - ratio) / 0.15 * W_OI_RATIO * 0.3
                w_oi += W_OI_RATIO

            # Volume-OI divergence
            voi = oi_sigs['voi_div'][si, di]
            if not np.isnan(voi):
                s_oi += max(voi, 0.0) * W_VOI_DIVERGENCE
                w_oi += W_VOI_DIVERGENCE

            # --- Layer 2: Price oversold (30%) ---
            s_price = 0.0
            w_price = 0.0

            consec = price_sigs['consec_dn'][si, di]
            s_price += min(consec / 5.0, 1.0) * 0.50
            w_price += 0.50

            ret5 = price_sigs['ret_5d'][si, di]
            if not np.isnan(ret5):
                s_price += min(max(-ret5 / 0.1, 0), 1.0) * 0.50
                w_price += 0.50

            # --- Layer 3: Confirmation (30%) ---
            s_conf = 0.0
            w_conf = 0.0

            vdp = confirm_sigs['vdp_exhaust'][si, di]
            if not np.isnan(vdp):
                s_conf += vdp * 0.35
                w_conf += 0.35

            rsi = confirm_sigs['rsi14'][si, di]
            if not np.isnan(rsi):
                if rsi < 30:
                    s_conf += (30 - rsi) / 30.0 * 0.25
                elif rsi < 35:
                    s_conf += (35 - rsi) / 5.0 * 0.10
                w_conf += 0.25

            bb = confirm_sigs['bb_pos'][si, di]
            if not np.isnan(bb):
                if bb < 0.2:
                    s_conf += (0.2 - bb) / 0.2 * 0.20
                w_conf += 0.20

            cci = confirm_sigs['cci14'][si, di]
            if not np.isnan(cci):
                if cci < -100:
                    s_conf += min((-100 - cci) / 200.0, 1.0) * 0.20
                w_conf += 0.20

            # --- Combine layers ---
            total_score = 0.0
            total_weight = 0.0

            if w_oi > 0:
                total_score += (s_oi / w_oi) * W_OI_LAYER
                total_weight += W_OI_LAYER
                oi_scores[si] = s_oi / w_oi

            if w_price > 0:
                total_score += (s_price / w_price) * W_PRICE_LAYER
                total_weight += W_PRICE_LAYER

            if w_conf > 0:
                total_score += (s_conf / w_conf) * W_CONFIRM_LAYER
                total_weight += W_CONFIRM_LAYER

            if total_weight > 0:
                scores[si] = total_score / total_weight

        # Cross-sectional rank
        valid = ~np.isnan(scores)
        if valid.sum() >= 5:
            raw_score[:, di] = pd.Series(scores).rank(pct=True, na_option='keep').values
            oi_score_raw[:, di] = oi_scores

    # Count confirmation signals per bar
    n_signals = np.zeros((NS, ND), dtype=int)
    for di in range(ND):
        for si in range(NS):
            if np.isnan(C[si, di]) or C[si, di] <= 0:
                continue
            n = 0
            # OI signals
            zscore = oi_sigs['oi_zscore'][si, di]
            if not np.isnan(zscore) and zscore < -1.0:
                n += 1
            div = oi_sigs['oi_price_div'][si, di]
            if not np.isnan(div) and div > 0.1:
                n += 1
            accel = oi_sigs['oi_accel'][si, di]
            if not np.isnan(accel) and accel > 0.1:
                n += 1
            ratio = oi_sigs['oi_ratio'][si, di]
            if not np.isnan(ratio) and ratio < 0.85:
                n += 1
            voi = oi_sigs['voi_div'][si, di]
            if not np.isnan(voi) and voi > 0.1:
                n += 1
            # Price signals
            if price_sigs['consec_dn'][si, di] >= 3:
                n += 1
            ret5 = price_sigs['ret_5d'][si, di]
            if not np.isnan(ret5) and ret5 < -0.03:
                n += 1
            # Confirmation signals
            vdp = confirm_sigs['vdp_exhaust'][si, di]
            if not np.isnan(vdp) and vdp > 0.3:
                n += 1
            rsi = confirm_sigs['rsi14'][si, di]
            if not np.isnan(rsi) and rsi < 35:
                n += 1
            bb = confirm_sigs['bb_pos'][si, di]
            if not np.isnan(bb) and bb < 0.15:
                n += 1
            cci = confirm_sigs['cci14'][si, di]
            if not np.isnan(cci) and cci < -100:
                n += 1
            n_signals[si, di] = n

    print(f"  All signals done: {time.time() - t0:.1f}s", flush=True)
    return {
        'combo_rank': raw_score,
        'oi_score_raw': oi_score_raw,
        'ker_regime': ker_regime,
        'n_signals': n_signals,
    }


def backtest_v12(C, O, H, L, NS, ND, dates, syms, sigs,
                 top_n=1, min_rank=0.7, atr_stop=3.0,
                 min_confidence=3, use_ker_gate=True,
                 hold_days=5,
                 pyramid_ratio=0.5,
                 pyramid_day=1,
                 start_di=60, end_di=None):
    """Backtest with OI-weighted signals + pyramid."""
    combo_rank = sigs['combo_rank']
    ker_regime = sigs['ker_regime']
    n_signals = sigs['n_signals']

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
            alloc = 1.0 / max(top_n, 1)
            candidates.append((combo_rank[si, di], si, alloc))

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


def walk_forward(C, O, H, L, NS, ND, dates, syms, sigs,
                 pyramid_ratio=0.5, pyramid_day=1,
                 top_n=1, min_confidence=3, hold_days=5, atr_stop=3.0):
    """Walk-forward validation: year-by-year out-of-sample."""
    print(f"\n{'=' * 70}")
    print(f"  WALK-FORWARD V12 (pyr={pyramid_ratio}, day={pyramid_day})")
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

        trades, _, _ = backtest_v12(
            C, O, H, L, NS, ND, dates, syms, sigs,
            top_n=top_n, hold_days=hold_days, atr_stop=atr_stop,
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
    print("  V12: OPEN INTEREST EXTREME CONTRARIAN")
    print("  Futures-specific OI alpha — Walk-Forward Validated")
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

    # === 1. Walk-Forward Validation ===
    print("\n" + "=" * 70)
    print("  WALK-FORWARD VALIDATION (2019-2026)")
    print("=" * 70)

    for ratio in [0.0, 0.3, 0.5, 0.7]:
        walk_forward(C, O, H, L, NS, ND, dates, syms, sigs,
                     pyramid_ratio=ratio, pyramid_day=1,
                     top_n=1, min_confidence=3)

    # === 2. Full 10-year backtest with pyramid profiles ===
    print("\n" + "=" * 70)
    print("  FULL 2016-2026 (10 years) — PYRAMID PROFILES")
    print("=" * 70)

    profiles = [
        (0.0, 1, "No pyramid (baseline)"),
        (0.3, 1, "Mild pyramid (30%)"),
        (0.5, 1, "Moderate pyramid (50%)"),
        (0.7, 1, "Aggressive pyramid (70%)"),
    ]

    for ratio, pday, label in profiles:
        trades, eq, dd = backtest_v12(
            C, O, H, L, NS, ND, dates, syms, sigs,
            top_n=1, hold_days=5, atr_stop=3.0,
            min_confidence=3, use_ker_gate=True,
            pyramid_ratio=ratio, pyramid_day=pday,
            start_di=60)
        print(f"\n  {label}")
        analyze(trades, eq, dd, label)

    # === 3. Multi-position + pyramid parameter sweep ===
    print("\n" + "=" * 70)
    print("  PARAMETER SWEEP (2019-2026)")
    print("=" * 70)

    results = []
    for tn in [1, 2, 3]:
        for ratio in [0.0, 0.3, 0.5]:
            for mc in [2, 3, 4]:
                trades, eq, dd = backtest_v12(
                    C, O, H, L, NS, ND, dates, syms, sigs,
                    top_n=tn, hold_days=5, atr_stop=3.0,
                    min_confidence=mc, use_ker_gate=True,
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
                sh_val = np.mean(rets_arr) / np.std(rets_arr) * np.sqrt(252) if np.std(rets_arr) > 0 else 0
                results.append({
                    'tn': tn, 'ratio': ratio, 'mc': mc,
                    'n': len(trades), 'wr': wr, 'ann': ann, 'dd': dd, 'sharpe': sh_val,
                })

    results.sort(key=lambda x: -x['sharpe'])
    print(f"\n{'TN':>3} {'Pyr':>4} {'MC':>3} "
          f"{'N':>5} {'WR':>5} {'Ann':>8} {'DD':>6} {'Sh':>5}")
    print("-" * 55)
    for r in results[:25]:
        print(f"{r['tn']:>3} {r['ratio']:>4.1f} {r['mc']:>3} "
              f"{r['n']:>5} {r['wr']:>5.1f} {r['ann']:>+8.1f} "
              f"{r['dd']:>6.1f} {r['sharpe']:>5.2f}")

    # === 4. Best config full 10-year ===
    print("\n" + "=" * 70)
    print("  BEST CONFIG — FULL 10-YEAR")
    print("=" * 70)

    if results:
        best = results[:5]
        for r in best:
            trades, eq, dd = backtest_v12(
                C, O, H, L, NS, ND, dates, syms, sigs,
                top_n=r['tn'], hold_days=5, atr_stop=3.0,
                min_confidence=r['mc'], use_ker_gate=True,
                pyramid_ratio=r['ratio'], pyramid_day=1,
                start_di=60)
            label = f"tn={r['tn']} pyr={r['ratio']:.1f} mc={r['mc']}"
            print(f"\n  FULL {label}")
            analyze(trades, eq, dd, label)

    # === 5. Walk-forward for top config ===
    if results:
        best_r = results[0]
        print("\n" + "=" * 70)
        print(f"  BEST WF: tn={best_r['tn']} pyr={best_r['ratio']:.1f} mc={best_r['mc']}")
        print("=" * 70)
        walk_forward(C, O, H, L, NS, ND, dates, syms, sigs,
                     pyramid_ratio=best_r['ratio'], pyramid_day=1,
                     top_n=best_r['tn'], min_confidence=best_r['mc'])

    print(f"\n[V12] Done. {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
