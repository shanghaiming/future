"""
V37: Multi-Alpha Ensemble Vote — Combine 3 Uncorrelated Alpha Sources
=====================================================================
Core thesis: Institutional funds use multiple alpha generators. V37 combines
V18's cross-sectional rank, V12's OI capitulation, and V22's strict signal
gate into a voting system. A trade only fires when at least 2 of 3 alpha
sources agree. This produces fewer but higher-quality trades.

Three independent alpha generators:
  Alpha A (V18): Cross-sectional rank of 7 factors (composite rank percentile)
  Alpha B (V12): OI capitulation (OI 5d decline < -5%, consec down, VDP exhaustion)
  Alpha C (V22): Strict gate (>=4 of 7 V1 signals: consec_dn, ret5d, OI decline,
                  VDP, RSI, BB, CCI)

Each alpha generates a binary signal: 1 (buy) or 0 (skip).
Voting: require min_votes of 3 to enter.
Position sizing scales with vote count:
  - 3/3 agreement: full position (1.0x)
  - 2/3 agreement: half position (0.5x)
  - 1/3 or 0/3: no trade
KER gate still applies as additional filter.
Hold 5d, ATR stop 3.0.

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

# V18 default weights for composite rank
DEFAULT_WEIGHTS = {
    'rank_ret5d':  0.25,
    'rank_oi5d':   0.20,
    'rank_rsi':    0.15,
    'rank_vol':    0.15,
    'rank_ret10d': 0.10,
    'rank_range':  0.10,
    'rank_atrp':   0.05,
}


# ============================================================
# ALPHA A: V18 Cross-Sectional Rank
# ============================================================
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


def compute_alpha_a_signals(C, O, H, L, V, OI, NS, ND, weights=None, min_rank=0.75):
    """Alpha A: V18 cross-sectional rank mean reversion.
    Returns binary signal array: 1 if composite rank >= min_rank, else 0."""
    t0 = time.time()
    print("[V37-AlphaA] Computing V18 cross-sectional rank...", flush=True)

    if weights is None:
        weights = DEFAULT_WEIGHTS

    # --- Raw factors ---
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

    # --- Cross-sectional ranking ---
    factors_to_rank = {
        'rank_ret5d': ret_5d,
        'rank_ret10d': ret_10d,
        'rank_oi5d': oi_5d,
        'rank_vol': vol_5d,
        'rank_range': daily_range,
        'rank_rsi': rsi14,
        'rank_atrp': atrp,
    }
    INVERT_FACTORS = {'rank_ret5d', 'rank_ret10d', 'rank_oi5d', 'rank_rsi'}

    ranks = {}
    for name, factor in factors_to_rank.items():
        rank_arr = np.full((NS, ND), np.nan)
        for di in range(ND):
            vals = factor[:, di]
            valid_count = np.sum(~np.isnan(vals))
            if valid_count < 10:
                continue
            ranked = pd.Series(vals).rank(pct=True, na_option='keep').values
            if name in INVERT_FACTORS:
                ranked = 1.0 - ranked
            rank_arr[:, di] = ranked
        ranks[name] = rank_arr

    # --- Composite ---
    factor_names = list(weights.keys())
    weight_vals = np.array([weights[k] for k in factor_names])

    composite = np.full((NS, ND), np.nan)
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
            if w_sum > 0 and confirm_count >= 4:
                composite[si, di] = sum(vals) / w_sum

    # --- Binary signal ---
    signal_a = np.zeros((NS, ND), dtype=int)
    for si in range(NS):
        for di in range(ND):
            if not np.isnan(composite[si, di]) and composite[si, di] >= min_rank:
                signal_a[si, di] = 1

    print(f"  AlphaA done: {time.time() - t0:.1f}s", flush=True)
    return signal_a, composite


# ============================================================
# ALPHA B: V12 OI Capitulation
# ============================================================
def compute_alpha_b_signals(C, O, H, L, V, OI, NS, ND):
    """Alpha B: V12 OI capitulation detection.
    Signals when OI 5d decline < -5%, with consecutive down days and VDP exhaustion."""
    t0 = time.time()
    print("[V37-AlphaB] Computing V12 OI capitulation...", flush=True)

    # --- Consecutive down days ---
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

    # --- VDP exhaustion ---
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

    # --- OI 5d change ---
    oi_5d_chg = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            if not np.isnan(OI[si, di]) and not np.isnan(OI[si, di - 5]) and OI[si, di - 5] > 0:
                oi_5d_chg[si, di] = OI[si, di] / OI[si, di - 5] - 1.0

    # --- Binary signal: OI capitulation criteria ---
    signal_b = np.zeros((NS, ND), dtype=int)
    for si in range(NS):
        for di in range(20, ND):
            if np.isnan(C[si, di]) or C[si, di] <= 0:
                continue
            # Core: OI 5d decline < -5%
            oi_ok = (not np.isnan(oi_5d_chg[si, di]) and oi_5d_chg[si, di] < -0.05)
            # Supporting: consecutive down days >= 2
            consec_ok = consec_dn[si, di] >= 2
            # Supporting: VDP exhaustion > 0.3
            vdp_ok = (not np.isnan(vdp_exhaust[si, di]) and vdp_exhaust[si, di] > 0.3)
            # Fire if core + at least one supporting
            if oi_ok and (consec_ok or vdp_ok):
                signal_b[si, di] = 1

    print(f"  AlphaB done: {time.time() - t0:.1f}s", flush=True)
    return signal_b


# ============================================================
# ALPHA C: V22 Strict Signal Gate
# ============================================================
def compute_alpha_c_signals(C, O, H, L, V, OI, NS, ND, min_signals=4):
    """Alpha C: V22 strict signal gate.
    Require >=min_signals of 7 V1 signals to agree."""
    t0 = time.time()
    print(f"[V37-AlphaC] Computing V22 strict gate (min>={min_signals})...", flush=True)

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

    # --- 3. OI capitulation ---
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

    # --- 4. VDP exhaustion ---
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

    # --- 5-7. TA-Lib indicators ---
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

    # --- Count firing signals ---
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

    # --- Binary signal ---
    signal_c = np.zeros((NS, ND), dtype=int)
    for si in range(NS):
        for di in range(ND):
            if n_signals[si, di] >= min_signals:
                signal_c[si, di] = 1

    print(f"  AlphaC done: {time.time() - t0:.1f}s", flush=True)
    return signal_c


# ============================================================
# KER (shared regime filter)
# ============================================================
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
                ker_regime[si, di] = -1  # trending -> avoid
    return ker_regime


# ============================================================
# COMPUTE ALL ALPHA SIGNALS
# ============================================================
def compute_all_alpha_signals(C, O, H, L, V, OI, NS, ND,
                              min_rank_a=0.75, min_signals_c=4):
    """Compute all three alpha signals + KER regime."""
    t0 = time.time()
    print("[V37] Computing all alpha signals...", flush=True)

    signal_a, composite_a = compute_alpha_a_signals(
        C, O, H, L, V, OI, NS, ND, weights=DEFAULT_WEIGHTS, min_rank=min_rank_a)
    signal_b = compute_alpha_b_signals(C, O, H, L, V, OI, NS, ND)
    signal_c = compute_alpha_c_signals(C, O, H, L, V, OI, NS, ND,
                                       min_signals=min_signals_c)
    ker_regime = compute_ker(C, NS, ND)

    # --- Vote count ---
    vote_count = signal_a + signal_b + signal_c

    print(f"  All alphas done: {time.time() - t0:.1f}s", flush=True)

    # Print alpha agreement stats
    total_bars = NS * ND
    a_only = np.sum((signal_a == 1) & (signal_b == 0) & (signal_c == 0))
    b_only = np.sum((signal_a == 0) & (signal_b == 1) & (signal_c == 0))
    c_only = np.sum((signal_a == 0) & (signal_b == 0) & (signal_c == 1))
    ab = np.sum((signal_a == 1) & (signal_b == 1) & (signal_c == 0))
    ac = np.sum((signal_a == 1) & (signal_c == 1) & (signal_b == 0))
    bc = np.sum((signal_b == 1) & (signal_c == 1) & (signal_a == 0))
    abc = np.sum((vote_count == 3))

    print(f"  Alpha agreement (out of {total_bars:,} bars):")
    print(f"    A only: {a_only:>8,} ({a_only / total_bars * 100:.3f}%)")
    print(f"    B only: {b_only:>8,} ({b_only / total_bars * 100:.3f}%)")
    print(f"    C only: {c_only:>8,} ({c_only / total_bars * 100:.3f}%)")
    print(f"    A+B:    {ab:>8,} ({ab / total_bars * 100:.3f}%)")
    print(f"    A+C:    {ac:>8,} ({ac / total_bars * 100:.3f}%)")
    print(f"    B+C:    {bc:>8,} ({bc / total_bars * 100:.3f}%)")
    print(f"    A+B+C:  {abc:>8,} ({abc / total_bars * 100:.3f}%)")

    return {
        'signal_a': signal_a,
        'signal_b': signal_b,
        'signal_c': signal_c,
        'vote_count': vote_count,
        'composite_a': composite_a,
        'ker_regime': ker_regime,
    }


# ============================================================
# BACKTEST ENGINE
# ============================================================
def backtest_v37(C, O, H, L, NS, ND, dates, syms, sigs,
                 top_n=1, atr_stop=3.0,
                 min_votes=2,
                 vote_size_2=0.5,
                 vote_size_3=1.0,
                 use_ker_gate=True,
                 hold_days=5,
                 pyramid_ratio=0.5,
                 pyramid_day=1,
                 start_di=60, end_di=None):
    """Backtest with multi-alpha ensemble voting."""
    vote_count = sigs['vote_count']
    composite_a = sigs['composite_a']
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
        daily_pnl = 0.0
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
                            additions.append((si, di, c_now, c_now - atr_stop * atr,
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

        held = {p[0] for p in positions}
        if len(positions) >= top_n:
            continue

        # Entry: signal at close[di], enter at open[di+1]
        candidates = []
        for si in range(NS):
            if si in held:
                continue
            if vote_count[si, di] < min_votes:
                continue
            # KER regime gate
            if use_ker_gate and ker_regime[si, di] < 0:
                continue
            if di + 1 >= ND or np.isnan(O[si, di + 1]):
                continue

            # Position sizing based on vote count
            votes = vote_count[si, di]
            if votes >= 3:
                size_mult = vote_size_3
            elif votes >= 2:
                size_mult = vote_size_2
            else:
                continue

            alloc = size_mult / max(top_n, 1)
            # Use composite_a as tiebreaker for ranking candidates
            rank_val = composite_a[si, di] if not np.isnan(composite_a[si, di]) else 0.0
            candidates.append((rank_val, si, alloc))

        # Sort by V18 composite rank (most oversold first)
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


# ============================================================
# ANALYSIS
# ============================================================
def analyze(trades, equity, max_dd, label=""):
    """Analyze backtest results with yearly breakdown."""
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

    cum_ret = equity / CASH0 - 1

    print(f"  {label}: {len(trades)}t (base:{n_base} pyr:{n_pyr} stop:{n_stop} hold:{n_hold}) "
          f"WR={wr:.1f}% ann={ann:+.1f}% DD={max_dd:.1f}% "
          f"Sh={sh:.2f} cum={cum_ret:+.1%} eq={equity:,.0f}")

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
def walk_forward(C, O, H, L, NS, ND, dates, syms, sigs,
                 top_n=1, hold_days=5, atr_stop=3.0,
                 min_votes=2, vote_size_2=0.5, vote_size_3=1.0,
                 pyramid_ratio=0.5):
    """Walk-forward validation: year-by-year out-of-sample."""
    print(f"\n{'=' * 70}")
    print(f"  WALK-FORWARD V37 (votes>={min_votes}, "
          f"sz2={vote_size_2}, sz3={vote_size_3}, pyr={pyramid_ratio})")
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

        trades, _, _ = backtest_v37(
            C, O, H, L, NS, ND, dates, syms, sigs,
            top_n=top_n, hold_days=hold_days, atr_stop=atr_stop,
            min_votes=min_votes, vote_size_2=vote_size_2,
            vote_size_3=vote_size_3, use_ker_gate=True,
            pyramid_ratio=pyramid_ratio, pyramid_day=1,
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


# ============================================================
# MAIN
# ============================================================
def main():
    t0 = time.time()
    print("=" * 70)
    print("  V37: MULTI-ALPHA ENSEMBLE VOTE")
    print("  V18 rank + V12 OI capitulation + V22 strict gate")
    print("  Trade only when 2+ of 3 alpha sources agree")
    print("=" * 70)

    C, O, H, L, V, OI, NS, ND, dates, syms = load_all_data(start='2016-01-01')
    print(f"  {NS} sym, {ND} days, "
          f"{dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")

    # === 1. Compute signals with default params ===
    sigs = compute_all_alpha_signals(C, O, H, L, V, OI, NS, ND,
                                     min_rank_a=0.75, min_signals_c=4)

    # Find 2019 start index
    bt_2019 = None
    for i, d in enumerate(dates):
        if d >= pd.Timestamp('2019-01-01'):
            bt_2019 = i
            break

    # === 2. Quick vote-level ablation (2019-2026) ===
    print("\n" + "=" * 70)
    print("  VOTE-LEVEL ABLATION (tn=1 hd=5 atr=3.0, 2019-2026)")
    print("  Key question: does voting improve selectivity?")
    print("=" * 70)

    for mv in [1, 2, 3]:
        for pyr in [0.0, 0.5]:
            trades, eq, dd = backtest_v37(
                C, O, H, L, NS, ND, dates, syms, sigs,
                top_n=1, hold_days=5, atr_stop=3.0,
                min_votes=mv, vote_size_2=0.5, vote_size_3=1.0,
                use_ker_gate=True, pyramid_ratio=pyr, pyramid_day=1,
                start_di=bt_2019)
            label = f"votes>={mv} pyr={pyr:.1f}"
            analyze(trades, eq, dd, label)

    # === 3. Full parameter sweep ===
    print("\n" + "=" * 70)
    print("  PARAMETER SWEEP (2019-2026)")
    print("=" * 70)

    results = []

    for mv in [2, 3]:
        for tn in [1, 2, 3]:
            for vs2 in [0.3, 0.5, 0.7]:
                for vs3 in [0.7, 1.0]:
                    for atr_s in [2.5, 3.0]:
                        for pyr in [0.0, 0.5]:
                            trades, eq, dd = backtest_v37(
                                C, O, H, L, NS, ND, dates, syms, sigs,
                                top_n=tn, hold_days=5, atr_stop=atr_s,
                                min_votes=mv, vote_size_2=vs2,
                                vote_size_3=vs3, use_ker_gate=True,
                                pyramid_ratio=pyr, pyramid_day=1,
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
                                'mv': mv, 'tn': tn, 'vs2': vs2, 'vs3': vs3,
                                'atr': atr_s, 'pyr': pyr,
                                'n': len(trades), 'wr': wr, 'ann': ann,
                                'dd': dd, 'sharpe': sh_val, 'eq': eq,
                            })

    results.sort(key=lambda x: -x['sharpe'])
    print(f"\n{'MV':>3} {'TN':>3} {'VS2':>4} {'VS3':>4} {'ATR':>4} {'Pyr':>4} "
          f"{'N':>5} {'WR':>5} {'Ann':>8} {'DD':>6} {'Sh':>6}")
    print("-" * 80)
    for r in results[:30]:
        print(f"{r['mv']:>3} {r['tn']:>3} {r['vs2']:>4.1f} {r['vs3']:>4.1f} "
              f"{r['atr']:>4.1f} {r['pyr']:>4.1f} "
              f"{r['n']:>5} {r['wr']:>5.1f} {r['ann']:>+8.1f} "
              f"{r['dd']:>6.1f} {r['sharpe']:>6.2f}")

    # === 4. Sweep min_rank for Alpha A ===
    print("\n" + "=" * 70)
    print("  MIN_RANK SWEEP FOR ALPHA A (2019-2026)")
    print("=" * 70)

    for mr in [0.70, 0.80]:
        sigs_mr = compute_all_alpha_signals(C, O, H, L, V, OI, NS, ND,
                                            min_rank_a=mr, min_signals_c=4)
        for mv in [2, 3]:
            for pyr in [0.0, 0.5]:
                trades, eq, dd = backtest_v37(
                    C, O, H, L, NS, ND, dates, syms, sigs_mr,
                    top_n=1, hold_days=5, atr_stop=3.0,
                    min_votes=mv, vote_size_2=0.5, vote_size_3=1.0,
                    use_ker_gate=True, pyramid_ratio=pyr, pyramid_day=1,
                    start_di=bt_2019)
                label = f"mr={mr:.2f} votes>={mv} pyr={pyr:.1f}"
                analyze(trades, eq, dd, label)

    # === 5. Walk-forward for top configs ===
    print("\n" + "=" * 70)
    print("  WALK-FORWARD FOR TOP 5 CONFIGS")
    print("=" * 70)

    for r in results[:5]:
        wf_trades = walk_forward(
            C, O, H, L, NS, ND, dates, syms, sigs,
            top_n=r['tn'], hold_days=5, atr_stop=r['atr'],
            min_votes=r['mv'], vote_size_2=r['vs2'],
            vote_size_3=r['vs3'], pyramid_ratio=r['pyr'])

    # === 6. Full 10-year for top configs ===
    print("\n" + "=" * 70)
    print("  FULL 10-YEAR (2016-2026) FOR TOP 5 CONFIGS")
    print("=" * 70)

    for r in results[:5]:
        trades, eq, dd = backtest_v37(
            C, O, H, L, NS, ND, dates, syms, sigs,
            top_n=r['tn'], hold_days=5, atr_stop=r['atr'],
            min_votes=r['mv'], vote_size_2=r['vs2'],
            vote_size_3=r['vs3'], use_ker_gate=True,
            pyramid_ratio=r['pyr'], pyramid_day=1,
            start_di=60)
        label = (f"full mv={r['mv']} tn={r['tn']} vs2={r['vs2']:.1f} "
                 f"vs3={r['vs3']:.1f} atr={r['atr']:.1f} pyr={r['pyr']:.1f}")
        print(f"\n  {label}")
        analyze(trades, eq, dd, label)

    # === 7. Head-to-head: V18 baseline vs ensemble ===
    print("\n" + "=" * 70)
    print("  HEAD-TO-HEAD: SINGLE ALPHA vs ENSEMBLE (2019-2026)")
    print("=" * 70)

    # Best single-alpha configs
    comparisons = [
        ("V18-only (votes>=1)", 1, 0.5, 1.0),
        ("V12-only (votes>=1)", 1, 0.5, 1.0),
        ("Ensemble 2/3", 2, 0.5, 1.0),
        ("Ensemble 3/3", 3, 0.7, 1.0),
    ]

    for label, mv, vs2, vs3 in comparisons:
        for pyr in [0.0, 0.5]:
            trades, eq, dd = backtest_v37(
                C, O, H, L, NS, ND, dates, syms, sigs,
                top_n=1, hold_days=5, atr_stop=3.0,
                min_votes=mv, vote_size_2=vs2, vote_size_3=vs3,
                use_ker_gate=True, pyramid_ratio=pyr, pyramid_day=1,
                start_di=bt_2019)
            full_label = f"{label} pyr={pyr:.1f}"
            analyze(trades, eq, dd, full_label)

    print(f"\n[V37] Done. {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
