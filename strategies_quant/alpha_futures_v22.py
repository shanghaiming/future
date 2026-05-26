"""
V22: Strict Signal Gate — Require High Signal Agreement for Entry
=================================================================
Core hypothesis: From V18 (best strategy, WF +2019.6%), we know
cross-sectional ranking works. This strategy tests whether requiring
MORE signals to agree (>=4 of 7, or >=5 of 7) improves selectivity
and returns. Quality over quantity.

Signal architecture:
  1. Same 7 signals as V1:
     - consec_dn:  consecutive down days (>=3)
     - ret_5d:     5-day return oversold (< -3%)
     - oi_capit:   OI capitulation (OI declining + price declining)
     - vdp_exhaust: VDP selling exhaustion
     - rsi:        RSI < 35
     - bb:         Bollinger position < 0.15
     - cci:        CCI < -100
  2. BUT gate at higher confidence: require >=4 signals (vs V1's >=3)
  3. Also test >=5, >=6, all 7
  4. Cross-sectional rank the composite score
  5. KER gate (KER < 0.15 = sideways regime)
  6. Pyramid on day-1 winners (ratio 0.5)
  7. Hold 5d, ATR stop 2.5/3.0
  8. Walk-forward 2019-2026

Parameter sweep:
  - min_confidence: 3, 4, 5, 6, 7
  - top_n: 1, 2, 3
  - pyramid: 0.0, 0.5
  - atr_stop: 2.5, 3.0
  - min_rank: 0.65, 0.70, 0.75, 0.80

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


# ============================================================
# SIGNAL COMPUTATION (Same 7 signals as V1)
# ============================================================
def compute_signals(C, O, H, L, V, OI, NS, ND):
    """Compute all 7 signals from V1 + composite score + cross-sectional rank."""
    t0 = time.time()
    print("[V22] Computing 7 signals from V1...", flush=True)

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

    # --- 3. OI capitulation: OI declining + price declining ---
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

    # --- 4. VDP (Volume Delta Pressure) ---
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

    # --- 5. KER (Kaufman Efficiency Ratio) ---
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

    # --- 6. TA-Lib indicators: RSI, Bollinger, CCI ---
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

    # --- 7. Count firing signals (binary) ---
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

    # --- 8. Composite score with cross-sectional ranking ---
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

        # Cross-sectional rank
        valid = ~np.isnan(scores)
        if valid.sum() >= 5:
            raw_score[:, di] = pd.Series(scores).rank(pct=True, na_option='keep').values

    # --- 9. KER regime ---
    ker_regime = np.zeros((NS, ND), dtype=int)
    for si in range(NS):
        for di in range(ND):
            if np.isnan(ker_10[si, di]):
                continue
            if ker_10[si, di] < 0.15:
                ker_regime[si, di] = 1  # mean-reverting
            elif ker_10[si, di] > 0.3:
                ker_regime[si, di] = -1  # trending

    elapsed = time.time() - t0
    print(f"  Signals done: {elapsed:.1f}s (TA-Lib: {HAS_TALIB})", flush=True)

    return {
        'combo_rank': raw_score,
        'ker_regime': ker_regime,
        'n_signals': n_signals,
        'rsi14': rsi14,
        'bb_pos': bb_pos,
        'cci14': cci14,
        'oi_decline': oi_decline,
        'vdp_exhaust': vdp_exhaust,
        'consec_dn': consec_dn,
    }


# ============================================================
# BACKTEST ENGINE
# ============================================================
def backtest_v22(C, O, H, L, NS, ND, dates, syms, sigs,
                 top_n=1, min_rank=0.75, atr_stop=3.0,
                 min_confidence=3, use_ker_gate=True,
                 hold_days=5,
                 pyramid_ratio=0.5,
                 pyramid_day=1,
                 start_di=60, end_di=None):
    """Backtest with strict signal gate + cross-sectional rank + pyramid."""
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
        daily_pnl = 0.0
        new_positions = []

        # Group positions by symbol for pyramid support
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

            # ATR stop: check if price hit stop
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

        # Entry: signal at close[di], enter at open[di+1]
        candidates = []
        for si in range(NS):
            if si in held:
                continue
            if np.isnan(combo_rank[si, di]):
                continue
            # Cross-sectional rank gate
            if combo_rank[si, di] < min_rank:
                continue
            # STRICT CONFIDENCE GATE: key differentiator from V1
            if n_signals[si, di] < min_confidence:
                continue
            # KER regime gate
            if use_ker_gate and ker_regime[si, di] < 0:
                continue
            if di + 1 >= ND or np.isnan(O[si, di + 1]):
                continue

            alloc = 1.0 / max(top_n, 1)
            candidates.append((combo_rank[si, di], si, alloc))

        # Sort by rank (most oversold first)
        candidates.sort(key=lambda x: -x[0])
        for rank, si, alloc in candidates[:top_n]:
            if len(positions) >= top_n or si in held:
                break
            ep = O[si, di + 1]
            if np.isnan(ep) or ep <= 0:
                continue
            # ATR stop loss
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

    # Close remaining positions at last available price
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
                 top_n=1, min_confidence=3, hold_days=5, atr_stop=3.0,
                 min_rank=0.75, pyramid_ratio=0.5):
    """Walk-forward validation: year-by-year out-of-sample."""
    print(f"\n{'=' * 70}")
    print(f"  WALK-FORWARD V22 (conf>={min_confidence}, pyr={pyramid_ratio})")
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

        trades, _, _ = backtest_v22(
            C, O, H, L, NS, ND, dates, syms, sigs,
            top_n=top_n, hold_days=hold_days, atr_stop=atr_stop,
            min_rank=min_rank,
            min_confidence=min_confidence, use_ker_gate=True,
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
    print("  V22: STRICT SIGNAL GATE MEAN REVERSION")
    print("  Quality over quantity: require more signals to agree")
    print("=" * 70)

    C, O, H, L, V, OI, NS, ND, dates, syms = load_all_data(start='2016-01-01')
    print(f"  {NS} sym, {ND} days, "
          f"{dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")

    sigs = compute_signals(C, O, H, L, V, OI, NS, ND)

    # Find 2019 start index for OOS testing
    bt_2019 = None
    for i, d in enumerate(dates):
        if d >= pd.Timestamp('2019-01-01'):
            bt_2019 = i
            break

    # === 1. Confidence Level Ablation (the core hypothesis test) ===
    print("\n" + "=" * 70)
    print("  CONFIDENCE LEVEL ABLATION (tn=1 hd=5 pyr=0.5, 2019-2026)")
    print("  Key question: does >=4 or >=5 signals beat >=3?")
    print("=" * 70)

    for mc in [3, 4, 5, 6, 7]:
        for pyr in [0.0, 0.5]:
            trades, eq, dd = backtest_v22(
                C, O, H, L, NS, ND, dates, syms, sigs,
                top_n=1, hold_days=5, min_rank=0.75,
                min_confidence=mc, use_ker_gate=True,
                atr_stop=3.0, pyramid_ratio=pyr, pyramid_day=1,
                start_di=bt_2019)
            label = f"conf>={mc} pyr={pyr:.1f}"
            analyze(trades, eq, dd, label)

    # === 2. Parameter Sweep ===
    print("\n" + "=" * 70)
    print("  PARAMETER SWEEP (2019-2026)")
    print("=" * 70)

    results = []

    for mc in [3, 4, 5, 6, 7]:
        for tn in [1, 2, 3]:
            for pyr in [0.0, 0.5]:
                for atr_s in [2.5, 3.0]:
                    for mr in [0.65, 0.70, 0.75, 0.80]:
                        trades, eq, dd = backtest_v22(
                            C, O, H, L, NS, ND, dates, syms, sigs,
                            top_n=tn, hold_days=5, min_rank=mr,
                            min_confidence=mc, use_ker_gate=True,
                            atr_stop=atr_s, pyramid_ratio=pyr, pyramid_day=1,
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
                            'mc': mc, 'tn': tn, 'pyr': pyr, 'atr': atr_s,
                            'mr': mr,
                            'n': len(trades), 'wr': wr, 'ann': ann,
                            'dd': dd, 'sharpe': sh_val, 'eq': eq,
                        })

    results.sort(key=lambda x: -x['sharpe'])
    print(f"\n{'MC':>3} {'TN':>3} {'Pyr':>4} {'ATR':>4} {'MR':>4} "
          f"{'N':>5} {'WR':>5} {'Ann':>8} {'DD':>6} {'Sh':>6}")
    print("-" * 70)
    for r in results[:30]:
        print(f"{r['mc']:>3} {r['tn']:>3} {r['pyr']:>4.1f} {r['atr']:>4.1f} {r['mr']:>4.2f} "
              f"{r['n']:>5} {r['wr']:>5.1f} {r['ann']:>+8.1f} "
              f"{r['dd']:>6.1f} {r['sharpe']:>6.2f}")

    # === 3. Walk-Forward for top configs ===
    print("\n" + "=" * 70)
    print("  WALK-FORWARD FOR TOP 5 CONFIGS")
    print("=" * 70)

    for r in results[:5]:
        wf_trades = walk_forward(
            C, O, H, L, NS, ND, dates, syms, sigs,
            top_n=r['tn'], min_confidence=r['mc'],
            hold_days=5, atr_stop=r['atr'],
            min_rank=r['mr'], pyramid_ratio=r['pyr'])

    # === 4. Full 10-year for top configs ===
    print("\n" + "=" * 70)
    print("  FULL 10-YEAR (2016-2026) FOR TOP 5 CONFIGS")
    print("=" * 70)

    for r in results[:5]:
        trades, eq, dd = backtest_v22(
            C, O, H, L, NS, ND, dates, syms, sigs,
            top_n=r['tn'], hold_days=5, min_rank=r['mr'],
            min_confidence=r['mc'], use_ker_gate=True,
            atr_stop=r['atr'], pyramid_ratio=r['pyr'], pyramid_day=1,
            start_di=60)
        label = f"full mc={r['mc']} tn={r['tn']} pyr={r['pyr']:.1f} atr={r['atr']:.1f} mr={r['mr']:.2f}"
        print(f"\n  {label}")
        analyze(trades, eq, dd, label)

    # === 5. Direct comparison: V1 baseline (conf>=3) vs strict gate (conf>=4,5) ===
    print("\n" + "=" * 70)
    print("  HEAD-TO-HEAD: V1 BASELINE vs STRICT GATE (2019-2026, best params)")
    print("=" * 70)

    # Find best conf>=3 config
    best_c3 = [r for r in results if r['mc'] == 3]
    best_c4 = [r for r in results if r['mc'] == 4]
    best_c5 = [r for r in results if r['mc'] == 5]
    best_c6 = [r for r in results if r['mc'] == 6]
    best_c7 = [r for r in results if r['mc'] == 7]

    comparisons = [
        ("BEST conf>=3", best_c3[0] if best_c3 else None),
        ("BEST conf>=4", best_c4[0] if best_c4 else None),
        ("BEST conf>=5", best_c5[0] if best_c5 else None),
        ("BEST conf>=6", best_c6[0] if best_c6 else None),
        ("BEST conf>=7", best_c7[0] if best_c7 else None),
    ]

    for label, best in comparisons:
        if best is None:
            print(f"  {label}: no qualifying trades")
            continue
        trades, eq, dd = backtest_v22(
            C, O, H, L, NS, ND, dates, syms, sigs,
            top_n=best['tn'], hold_days=5, min_rank=best['mr'],
            min_confidence=best['mc'], use_ker_gate=True,
            atr_stop=best['atr'], pyramid_ratio=best['pyr'], pyramid_day=1,
            start_di=bt_2019)
        print(f"\n  {label} (tn={best['tn']} pyr={best['pyr']:.1f} atr={best['atr']:.1f} mr={best['mr']:.2f})")
        analyze(trades, eq, dd, label)

    print(f"\n[V22] Done. {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
