"""
V6: Correlation-Based Portfolio Construction
=============================================
Best V1 signal (consec_dn, ret5d, OI capitulation, VDP, RSI, BB, CCI)
+ KER gate + confidence>=3, but now with correlation-aware position selection.

Key innovation:
  1. Compute rolling 60-day correlation between all commodity pairs
  2. When selecting top-N candidates, prefer those with LOW correlation to
     existing positions (true diversification, not just "more positions")
  3. Test with 2-5 positions and various correlation thresholds
  4. Walk-forward validation for robustness

Signal at close[di], enter at open[di+1]. No look-ahead. No gap signals.
No leverage. Import from alpha_futures_data. Use TA-Lib if available.
"""
import sys
import os
import time
import warnings
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd

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

CORR_WINDOW = 60
MIN_CORR_OVERLAP = 30


# ============================================================
# SIGNAL COMPUTATION (same V1 multi-alpha as V5)
# ============================================================
def compute_signals(
    C: np.ndarray,
    O: np.ndarray,
    H: np.ndarray,
    L: np.ndarray,
    V: np.ndarray,
    OI: np.ndarray,
    NS: int,
    ND: int,
) -> dict:
    """Core signal computation — identical to V5 for fair comparison."""
    t0 = time.time()
    print("[V6] Computing signals...", flush=True)

    # Consecutive down days
    consec_dn = np.zeros((NS, ND), dtype=int)
    for si in range(NS):
        consec = 0
        for di in range(1, ND):
            if (
                not np.isnan(C[si, di])
                and not np.isnan(C[si, di - 1])
                and C[si, di - 1] > 0
            ):
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
            if (
                not np.isnan(C[si, di])
                and not np.isnan(C[si, di - 5])
                and C[si, di - 5] > 0
            ):
                ret_5d[si, di] = C[si, di] / C[si, di - 5] - 1

    # OI capitulation
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
                oi_decline[si, di] = (
                    min(abs(oi_chg), 0.2) / 0.2 * min(abs(price_chg), 0.1) / 0.1
                )
            else:
                oi_decline[si, di] = 0.0

    # VDP (Volume Differential Pressure)
    vdp = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(1, ND):
            if np.isnan(H[si, di]) or np.isnan(L[si, di]):
                continue
            if np.isnan(C[si, di]) or np.isnan(V[si, di]):
                continue
            bar_range = H[si, di] - L[si, di]
            if bar_range > 0 and V[si, di] > 0:
                vdp[si, di] = (
                    V[si, di] * (2 * C[si, di] - H[si, di] - L[si, di]) / bar_range
                )

    vdp_10 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(10, ND):
            vals = vdp[si, di - 10 : di]
            valid = vals[~np.isnan(vals)]
            if len(valid) >= 5:
                vdp_10[si, di] = np.mean(valid)

    vdp_exhaust = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            if np.isnan(vdp_10[si, di]):
                continue
            window = vdp_10[si, max(0, di - 20) : di]
            wv = window[~np.isnan(window)]
            if len(wv) >= 10:
                mu, sig = np.mean(wv), np.std(wv)
                if sig > 0:
                    z = (vdp_10[si, di] - mu) / sig
                    vdp_exhaust[si, di] = min(-z, 3.0) / 3.0 if z < 0 else 0.0

    # KER (Kaufman Efficiency Ratio)
    ker_10 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(10, ND):
            closes = C[si, di - 10 : di + 1]
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

    # RSI, BB, CCI via TA-Lib
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

    # Composite score + cross-sectional rank
    raw_score = np.full((NS, ND), np.nan)
    for di in range(ND):
        scores = np.full(NS, np.nan)
        for si in range(NS):
            if np.isnan(C[si, di]) or C[si, di] <= 0:
                continue
            s = 0.0
            w_total = 0.0
            s += min(consec_dn[si, di] / 5.0, 1.0) * 0.20
            w_total += 0.20
            if not np.isnan(ret_5d[si, di]):
                s += min(max(-ret_5d[si, di] / 0.1, 0), 1.0) * 0.20
                w_total += 0.20
            if not np.isnan(oi_decline[si, di]):
                s += oi_decline[si, di] * 0.20
                w_total += 0.20
            if not np.isnan(vdp_exhaust[si, di]):
                s += vdp_exhaust[si, di] * 0.15
                w_total += 0.15
            if not np.isnan(rsi14[si, di]):
                if rsi14[si, di] < 30:
                    s += (30 - rsi14[si, di]) / 30.0 * 0.10
                w_total += 0.10
            if not np.isnan(bb_pos[si, di]):
                if bb_pos[si, di] < 0.2:
                    s += (0.2 - bb_pos[si, di]) / 0.2 * 0.10
                w_total += 0.10
            if not np.isnan(cci14[si, di]):
                if cci14[si, di] < -100:
                    s += min((-100 - cci14[si, di]) / 200.0, 1.0) * 0.05
                w_total += 0.05
            if w_total > 0:
                scores[si] = s / w_total
        valid = ~np.isnan(scores)
        if valid.sum() >= 5:
            raw_score[:, di] = (
                pd.Series(scores).rank(pct=True, na_option='keep').values
            )

    # Signal count (confidence)
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

    print(f"  Done: {time.time() - t0:.1f}s", flush=True)
    return {
        'combo_rank': raw_score,
        'ker_regime': ker_regime,
        'n_signals': n_signals,
    }


# ============================================================
# ROLLING CORRELATION MATRIX
# ============================================================
def compute_rolling_correlations(
    C: np.ndarray, NS: int, ND: int, window: int = CORR_WINDOW
) -> np.ndarray:
    """
    Compute rolling pairwise correlations of daily returns.

    Returns:
        corr_matrix: (NS, NS, ND) array where corr_matrix[i, j, di] is the
                     rolling correlation between symbol i and j at day di.
                     Only filled for di >= window.
    """
    t0 = time.time()
    print(f"[V6] Computing rolling {window}-day correlations...", flush=True)

    # Compute daily returns
    returns = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(1, ND):
            if (
                not np.isnan(C[si, di])
                and not np.isnan(C[si, di - 1])
                and C[si, di - 1] > 0
            ):
                returns[si, di] = C[si, di] / C[si, di - 1] - 1

    corr_matrix = np.full((NS, NS, ND), np.nan)

    for di in range(window, ND):
        # Get return window for all symbols
        ret_window = returns[:, di - window : di]  # (NS, window)
        for si in range(NS):
            ri = ret_window[si]
            valid_i = ~np.isnan(ri)
            if valid_i.sum() < MIN_CORR_OVERLAP:
                continue
            for sj in range(si, NS):
                rj = ret_window[sj]
                valid_j = ~np.isnan(rj)
                overlap = valid_i & valid_j
                n_overlap = overlap.sum()
                if n_overlap < MIN_CORR_OVERLAP:
                    continue
                ri_clean = ri[overlap]
                rj_clean = rj[overlap]
                std_i = np.std(ri_clean)
                std_j = np.std(rj_clean)
                if std_i > 1e-10 and std_j > 1e-10:
                    corr_val = np.corrcoef(ri_clean, rj_clean)[0, 1]
                    if not np.isnan(corr_val):
                        corr_matrix[si, sj, di] = corr_val
                        corr_matrix[sj, si, di] = corr_val

    n_pairs = NS * (NS - 1) // 2
    filled = 0
    sample_di = min(window + 100, ND - 1)
    for si in range(NS):
        for sj in range(si + 1, NS):
            if not np.isnan(corr_matrix[si, sj, sample_di]):
                filled += 1
    print(
        f"  Corr matrix sample (di={sample_di}): "
        f"{filled}/{n_pairs} pairs filled, {time.time() - t0:.1f}s",
        flush=True,
    )
    return corr_matrix


def get_avg_corr_to_positions(
    corr_matrix: np.ndarray,
    candidate_si: int,
    held_sis: list,
    di: int,
) -> float:
    """
    Get the average absolute correlation between a candidate symbol and
    all currently held positions. Returns 0.0 if no valid correlations found
    (conservative — allows entry).
    """
    if not held_sis:
        return 0.0

    corr_vals = []
    for held_si in held_sis:
        c = corr_matrix[candidate_si, held_si, di]
        if not np.isnan(c):
            corr_vals.append(abs(c))

    return np.mean(corr_vals) if corr_vals else 0.0


# ============================================================
# BACKTEST WITH CORRELATION-AWARE SELECTION
# ============================================================
def backtest_v6(
    C: np.ndarray,
    O: np.ndarray,
    H: np.ndarray,
    L: np.ndarray,
    NS: int,
    ND: int,
    dates: list,
    syms: list,
    sigs: dict,
    corr_matrix: np.ndarray,
    top_n: int = 2,
    min_rank: float = 0.7,
    atr_stop: float = 3.0,
    min_confidence: int = 3,
    use_ker_gate: bool = True,
    hold_days: int = 5,
    max_corr: float = 0.6,
    corr_penalty: float = 0.3,
    start_di: int = 60,
    end_di: Optional[int] = None,
) -> tuple:
    """
    Backtest with correlation-aware position selection.

    When selecting among top candidates for new positions, we penalize
    symbols that are highly correlated with existing positions. This
    creates a truly diversified portfolio rather than just "more bets".

    Parameters:
        max_corr: reject candidates whose avg abs correlation to held
                  positions exceeds this threshold
        corr_penalty: weight of correlation penalty in ranking score.
                      The adjusted score = rank - corr_penalty * avg_abs_corr
    """
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

        # Exit logic
        pos_by_si = defaultdict(list)
        for si, edi, ep, sp, alloc in positions:
            pos_by_si[si].append((edi, ep, sp, alloc))

        for si, pos_list in pos_by_si.items():
            c = C[si, di]
            if np.isnan(c):
                for edi, ep, sp, alloc in pos_list:
                    new_positions.append((si, edi, ep, sp, alloc))
                continue

            earliest_edi = min(p[0] for p in pos_list)
            hold = di - earliest_edi

            stopped = any(c < sp for _, _, sp, _ in pos_list)

            if stopped:
                for edi, ep, sp, alloc in pos_list:
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
                    })
            elif hold >= hold_days:
                for edi, ep, sp, alloc in pos_list:
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
                    })
            else:
                for edi, ep, sp, alloc in pos_list:
                    new_positions.append((si, edi, ep, sp, alloc))

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

        # Entry selection
        held = {p[0] for p in positions}
        if len(positions) >= top_n:
            continue

        # Build candidate list with correlation-aware scoring
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

            rank = combo_rank[si, di]

            # Correlation penalty: how correlated is this candidate
            # to our current positions?
            held_sis = list(held)
            avg_corr = get_avg_corr_to_positions(
                corr_matrix, si, held_sis, di
            )

            # Hard filter: skip if too correlated
            if len(held) > 0 and avg_corr > max_corr:
                continue

            # Adjusted score: high rank + low correlation = better
            adjusted_score = rank - corr_penalty * avg_corr

            candidates.append((adjusted_score, rank, avg_corr, si))

        # Sort by adjusted score (best first)
        candidates.sort(key=lambda x: -x[0])

        for adjusted_score, rank, avg_corr, si in candidates:
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

            alloc = 1.0 / top_n
            positions.append((si, di + 1, ep, ep - atr_stop * atr, alloc))
            held.add(si)

    # Close remaining positions at final price
    for si, edi, ep, sp, alloc in positions:
        c = C[si, ND - 1]
        if not np.isnan(c) and c > 0:
            pnl = (c - ep) / ep - COMM
            equity += equity * alloc * pnl

    return trades, equity, max_dd


# ============================================================
# ANALYSIS
# ============================================================
def analyze(
    trades: list, equity: float, max_dd: float, label: str = ""
) -> Optional[dict]:
    """Print analysis and return summary dict."""
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

    print(
        f"  {label}: {len(trades)}t (stop:{n_stop} hold:{n_hold}) "
        f"WR={wr:.1f}% ann={ann:+.1f}% DD={max_dd:.1f}% Sh={sh:.2f} "
        f"eq={equity:,.0f}"
    )

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
def walk_forward_v6(
    C: np.ndarray,
    O: np.ndarray,
    H: np.ndarray,
    L: np.ndarray,
    NS: int,
    ND: int,
    dates: list,
    syms: list,
    sigs: dict,
    corr_matrix: np.ndarray,
    top_n: int = 2,
    max_corr: float = 0.6,
    corr_penalty: float = 0.3,
    min_confidence: int = 3,
) -> list:
    """Walk-forward validation: one year at a time, no look-ahead."""
    print(f"\n{'='*70}")
    print(
        f"  WALK-FORWARD V6 (tn={top_n}, max_corr={max_corr}, "
        f"penalty={corr_penalty}, mc={min_confidence})"
    )
    print(f"{'='*70}")

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

        trades, _, _ = backtest_v6(
            C, O, H, L, NS, ND, dates, syms, sigs, corr_matrix,
            top_n=top_n, hold_days=5, atr_stop=3.0,
            min_confidence=min_confidence, use_ker_gate=True,
            max_corr=max_corr, corr_penalty=corr_penalty,
            start_di=test_start, end_di=test_end_idx + 1,
        )

        test_trades = [t for t in trades if dates[t['di']].year == test_year]
        all_trades.extend(test_trades)

        if test_trades:
            n = len(test_trades)
            nw = sum(1 for t in test_trades if t['pnl_pct'] > 0)
            wr = nw / n * 100
            avg = np.mean([t['pnl_pct'] for t in test_trades])
            print(
                f"  {test_year}: {n}t WR={wr:.1f}% avg={avg:+.2f}%", flush=True
            )
        else:
            print(f"  {test_year}: no trades", flush=True)

    if all_trades:
        nw = sum(1 for t in all_trades if t['pnl_pct'] > 0)
        wr = nw / len(all_trades) * 100
        avg = np.mean([t['pnl_pct'] for t in all_trades])
        cum = np.prod([1 + t['pnl_pct'] / 100 for t in all_trades]) - 1
        print(
            f"\n  WF TOTAL: {len(all_trades)}t WR={wr:.1f}% "
            f"avg={avg:+.2f}% cum={cum:+.1%}"
        )
        return all_trades
    return []


# ============================================================
# MAIN
# ============================================================
def main():
    t0 = time.time()
    print("=" * 70)
    print("  V6: CORRELATION-BASED PORTFOLIO CONSTRUCTION")
    print("=" * 70)

    C, O, H, L, V, OI, NS, ND, dates, syms = load_all_data(start='2016-01-01')
    print(
        f"  {NS} sym, {ND} days, "
        f"{dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}"
    )

    sigs = compute_signals(C, O, H, L, V, OI, NS, ND)
    corr_matrix = compute_rolling_correlations(C, NS, ND, window=CORR_WINDOW)

    bt_2019 = None
    for i, d in enumerate(dates):
        if d >= pd.Timestamp('2019-01-01'):
            bt_2019 = i
            break

    # ===== SECTION 1: BASELINE (no correlation filter) =====
    print("\n" + "=" * 70)
    print("  SECTION 1: BASELINE (no correlation filter, 2019-2026)")
    print("=" * 70)

    for tn in [1, 2, 3, 4, 5]:
        trades, eq, dd = backtest_v6(
            C, O, H, L, NS, ND, dates, syms, sigs, corr_matrix,
            top_n=tn, hold_days=5, atr_stop=3.0,
            min_confidence=3, use_ker_gate=True,
            max_corr=1.0, corr_penalty=0.0,  # no filter
            start_di=bt_2019,
        )
        analyze(trades, eq, dd, f"baseline tn={tn}")

    # ===== SECTION 2: CORRELATION FILTER SWEEP =====
    print("\n" + "=" * 70)
    print("  SECTION 2: CORRELATION FILTER SWEEP (2019-2026)")
    print("=" * 70)

    results = []
    for tn in [2, 3, 4, 5]:
        for mc_thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            for penalty in [0.0, 0.2, 0.3, 0.5]:
                for mc in [2, 3]:
                    trades, eq, dd = backtest_v6(
                        C, O, H, L, NS, ND, dates, syms, sigs, corr_matrix,
                        top_n=tn, hold_days=5, atr_stop=3.0,
                        min_confidence=mc, use_ker_gate=True,
                        max_corr=mc_thresh, corr_penalty=penalty,
                        start_di=bt_2019,
                    )
                    if len(trades) < 10:
                        continue
                    nw = sum(1 for t in trades if t['pnl_pct'] > 0)
                    wr = nw / len(trades) * 100
                    n_days = max(1, trades[-1]['di'] - trades[0]['di'])
                    ann = ((eq / CASH0) ** (1 / max(1.0, n_days / 252)) - 1) * 100
                    ap = [t['pnl_abs'] for t in sorted(trades, key=lambda x: x['di'])]
                    rets_arr = np.array(ap) / CASH0
                    sh_val = (
                        np.mean(rets_arr) / np.std(rets_arr) * np.sqrt(252)
                        if np.std(rets_arr) > 0
                        else 0
                    )
                    results.append({
                        'tn': tn,
                        'max_corr': mc_thresh,
                        'penalty': penalty,
                        'mc': mc,
                        'n': len(trades),
                        'wr': wr,
                        'ann': ann,
                        'dd': dd,
                        'sharpe': sh_val,
                        'eq': eq,
                    })

    results.sort(key=lambda x: -x['sharpe'])
    print(
        f"\n{'TN':>3} {'MaxC':>5} {'Pen':>5} {'MC':>3} "
        f"{'N':>5} {'WR':>5} {'Ann':>8} {'DD':>6} {'Sh':>6}"
    )
    print("-" * 60)
    for r in results[:30]:
        print(
            f"{r['tn']:>3} {r['max_corr']:>5.1f} {r['penalty']:>5.1f} "
            f"{r['mc']:>3} {r['n']:>5} {r['wr']:>5.1f} "
            f"{r['ann']:>+8.1f} {r['dd']:>6.1f} {r['sharpe']:>6.2f}"
        )

    # ===== SECTION 3: BEST 10-YEAR (2016-2026) =====
    print("\n" + "=" * 70)
    print("  SECTION 3: BEST CONFIGS — FULL 10-YEAR (2016-2026)")
    print("=" * 70)

    best_configs = results[:10]
    for r in best_configs:
        trades, eq, dd = backtest_v6(
            C, O, H, L, NS, ND, dates, syms, sigs, corr_matrix,
            top_n=r['tn'], hold_days=5, atr_stop=3.0,
            min_confidence=r['mc'], use_ker_gate=True,
            max_corr=r['max_corr'], corr_penalty=r['penalty'],
            start_di=60,
        )
        label = (
            f"tn={r['tn']} maxC={r['max_corr']:.1f} "
            f"pen={r['penalty']:.1f} mc={r['mc']}"
        )
        print(f"\n  10Y {label}")
        analyze(trades, eq, dd, label)

    # ===== SECTION 4: WALK-FORWARD FOR TOP CONFIGS =====
    print("\n" + "=" * 70)
    print("  SECTION 4: WALK-FORWARD VALIDATION")
    print("=" * 70)

    # Run WF for top-3 Sharpe configs
    for r in best_configs[:3]:
        walk_forward_v6(
            C, O, H, L, NS, ND, dates, syms, sigs, corr_matrix,
            top_n=r['tn'], max_corr=r['max_corr'],
            corr_penalty=r['penalty'], min_confidence=r['mc'],
        )

    # ===== SECTION 5: COMPARISON SUMMARY =====
    print("\n" + "=" * 70)
    print("  SECTION 5: V1-BASELINE vs BEST-CORRELATED (2019-2026)")
    print("=" * 70)

    # V1 baseline (tn=1, no corr filter)
    trades_base, eq_base, dd_base = backtest_v6(
        C, O, H, L, NS, ND, dates, syms, sigs, corr_matrix,
        top_n=1, hold_days=5, atr_stop=3.0,
        min_confidence=3, use_ker_gate=True,
        max_corr=1.0, corr_penalty=0.0,
        start_di=bt_2019,
    )
    print("\n  V1-BASELINE (tn=1):")
    analyze(trades_base, eq_base, dd_base, "V1-base")

    if best_configs:
        r_best = best_configs[0]
        trades_corr, eq_corr, dd_corr = backtest_v6(
            C, O, H, L, NS, ND, dates, syms, sigs, corr_matrix,
            top_n=r_best['tn'], hold_days=5, atr_stop=3.0,
            min_confidence=r_best['mc'], use_ker_gate=True,
            max_corr=r_best['max_corr'], corr_penalty=r_best['penalty'],
            start_di=bt_2019,
        )
        label_best = (
            f"BEST-CORR (tn={r_best['tn']} maxC={r_best['max_corr']:.1f} "
            f"pen={r_best['penalty']:.1f} mc={r_best['mc']}):"
        )
        print(f"\n  {label_best}")
        analyze(trades_corr, eq_corr, dd_corr, "BEST-CORR")

        # Multi-position without correlation (same tn)
        trades_nocorr, eq_nocorr, dd_nocorr = backtest_v6(
            C, O, H, L, NS, ND, dates, syms, sigs, corr_matrix,
            top_n=r_best['tn'], hold_days=5, atr_stop=3.0,
            min_confidence=r_best['mc'], use_ker_gate=True,
            max_corr=1.0, corr_penalty=0.0,
            start_di=bt_2019,
        )
        print(f"\n  SAME-TN NO-CORR (tn={r_best['tn']} mc={r_best['mc']}):")
        analyze(trades_nocorr, eq_nocorr, dd_nocorr, "NO-CORR")

    print(f"\n[V6] Done. {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
