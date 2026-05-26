"""
V3: Walk-Forward Validation + Enhanced Entry
=============================================
V1's best config (conf≥3+KER, tn=1, hd=5) showed 27.9% ann on 2019-2026.
But 10-year was only 10.1%. V3 investigates:
  1. Proper walk-forward validation (train/test rolling windows)
  2. Wider stop loss to reduce stop-outs
  3. Calendar effects (month-of-year, day-of-week)
  4. Sector-weighted diversification
  5. Adaptive parameters based on recent regime

Signal at close[di], enter at open[di+1]. No look-ahead.
"""
import sys, os, time, warnings
import numpy as np, pandas as pd
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

COMMODITY_GROUPS = {
    'BLACK':    ['rbfi', 'hcfi', 'ifi', 'jfi', 'jmfi'],
    'METAL':    ['cufi', 'alfi', 'znfi', 'nifi', 'snfi'],
    'PRECIOUS': ['aufi', 'agfi'],
    'ENERGY':   ['scfi', 'bufi', 'fufi', 'tafi', 'mafi'],
    'CHEM':     ['ppfi', 'lfi', 'vfi', 'egfi', 'ebfi', 'safi'],
    'OILCHAIN': ['mfi', 'yfi', 'ofi', 'pfi', 'rmfi'],
    'GRAIN':    ['cfi', 'csfi', 'srfi', 'cffi'],
}


def compute_signals(C, O, H, L, V, OI, NS, ND):
    """Compute all signals."""
    t0 = time.time()
    print("[V3] Computing signals...", flush=True)

    # Consecutive down
    consec_dn = np.zeros((NS, ND), dtype=int)
    for si in range(NS):
        consec = 0
        for di in range(1, ND):
            if not np.isnan(C[si, di]) and not np.isnan(C[si, di-1]) and C[si, di-1] > 0:
                if C[si, di] < C[si, di-1]:
                    consec += 1
                else:
                    consec = 0
            else:
                consec = 0
            consec_dn[si, di] = consec

    # 5d return
    ret_5d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            if not np.isnan(C[si, di]) and not np.isnan(C[si, di-5]) and C[si, di-5] > 0:
                ret_5d[si, di] = C[si, di] / C[si, di-5] - 1

    # 20d volatility
    vol_20d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            rets = []
            for j in range(di - 20, di):
                if not np.isnan(C[si, j]) and not np.isnan(C[si, j-1]) and C[si, j-1] > 0:
                    rets.append(C[si, j] / C[si, j-1] - 1)
            if len(rets) >= 10:
                vol_20d[si, di] = np.std(rets) * np.sqrt(252)

    # OI capitulation
    oi_decline = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            if np.isnan(OI[si, di]) or np.isnan(OI[si, di-5]):
                continue
            if np.isnan(C[si, di]) or np.isnan(C[si, di-5]) or C[si, di-5] <= 0:
                continue
            oi_chg = OI[si, di] / OI[si, di-5] - 1
            price_chg = C[si, di] / C[si, di-5] - 1
            if oi_chg < -0.02 and price_chg < -0.02:
                oi_decline[si, di] = min(abs(oi_chg), 0.2) / 0.2 * min(abs(price_chg), 0.1) / 0.1
            else:
                oi_decline[si, di] = 0.0

    # VDP
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
            vals = vdp[si, di-10:di]
            valid = vals[~np.isnan(vals)]
            if len(valid) >= 5:
                vdp_10[si, di] = np.mean(valid)

    vdp_exhaust = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            if np.isnan(vdp_10[si, di]):
                continue
            window = vdp_10[si, max(0, di-20):di]
            wv = window[~np.isnan(window)]
            if len(wv) >= 10:
                mu, sig = np.mean(wv), np.std(wv)
                if sig > 0:
                    z = (vdp_10[si, di] - mu) / sig
                    vdp_exhaust[si, di] = min(-z, 3.0) / 3.0 if z < 0 else 0.0

    # KER
    ker_10 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(10, ND):
            closes = C[si, di-10:di+1]
            valid = closes[~np.isnan(closes)]
            if len(valid) < 10 or valid[0] <= 0:
                continue
            net_change = abs(valid[-1] - valid[0])
            total_change = np.sum(np.abs(np.diff(valid)))
            if total_change > 1e-10:
                ker_10[si, di] = net_change / total_change

    # TA-Lib
    rsi14 = np.full((NS, ND), np.nan)
    bb_pos = np.full((NS, ND), np.nan)
    cci14 = np.full((NS, ND), np.nan)
    atr14 = np.full((NS, ND), np.nan)

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
            try:
                atr = talib.ATR(h, l, c, 14)
                atr14[si] = np.where(nan_mask, np.nan, atr)
            except Exception:
                pass

    # Composite score + rank
    raw_score = np.full((NS, ND), np.nan)
    for di in range(ND):
        scores = np.full(NS, np.nan)
        for si in range(NS):
            if np.isnan(C[si, di]) or C[si, di] <= 0:
                continue
            s = 0.0; w_total = 0.0
            s += min(consec_dn[si, di] / 5.0, 1.0) * 0.20; w_total += 0.20
            if not np.isnan(ret_5d[si, di]):
                s += min(max(-ret_5d[si, di] / 0.1, 0), 1.0) * 0.20; w_total += 0.20
            if not np.isnan(oi_decline[si, di]):
                s += oi_decline[si, di] * 0.20; w_total += 0.20
            if not np.isnan(vdp_exhaust[si, di]):
                s += vdp_exhaust[si, di] * 0.15; w_total += 0.15
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
            raw_score[:, di] = pd.Series(scores).rank(pct=True, na_option='keep').values

    # Confidence
    n_signals = np.zeros((NS, ND), dtype=int)
    for di in range(ND):
        for si in range(NS):
            if np.isnan(C[si, di]) or C[si, di] <= 0:
                continue
            n = 0
            if consec_dn[si, di] >= 3: n += 1
            if not np.isnan(ret_5d[si, di]) and ret_5d[si, di] < -0.03: n += 1
            if not np.isnan(oi_decline[si, di]) and oi_decline[si, di] > 0.1: n += 1
            if not np.isnan(vdp_exhaust[si, di]) and vdp_exhaust[si, di] > 0.3: n += 1
            if not np.isnan(rsi14[si, di]) and rsi14[si, di] < 35: n += 1
            if not np.isnan(bb_pos[si, di]) and bb_pos[si, di] < 0.15: n += 1
            if not np.isnan(cci14[si, di]) and cci14[si, di] < -100: n += 1
            n_signals[si, di] = n

    # KER regime
    ker_regime = np.zeros((NS, ND), dtype=int)
    for si in range(NS):
        for di in range(ND):
            if np.isnan(ker_10[si, di]):
                continue
            if ker_10[si, di] < 0.15:
                ker_regime[si, di] = 1
            elif ker_10[si, di] > 0.3:
                ker_regime[si, di] = -1

    print(f"  Done: {time.time() - t0:.1f}s", flush=True)

    return {
        'combo_rank': raw_score,
        'consec_dn': consec_dn,
        'vol_20d': vol_20d,
        'ker_regime': ker_regime,
        'n_signals': n_signals,
        'atr14': atr14,
    }


# ============================================================
# CALENDAR ANALYSIS
# ============================================================
def analyze_calendar(trades, dates, sigs, label=""):
    """Analyze performance by month and day-of-week."""
    if not trades:
        return

    print(f"\n  {label} Calendar Analysis:")

    # By month
    month_pnl = defaultdict(list)
    for t in trades:
        di = t['di']
        if di < len(dates):
            m = dates[di].month
            month_pnl[m].append(t['pnl_pct'])

    print(f"    Month  {'N':>5} {'WR':>6} {'AvgPnL':>8} {'Cum':>8}")
    for m in range(1, 13):
        if m in month_pnl:
            pnls = month_pnl[m]
            n = len(pnls)
            wr = sum(1 for p in pnls if p > 0) / n * 100
            avg = np.mean(pnls)
            cum = np.prod([1 + p / 100 for p in pnls]) - 1
            print(f"    {m:>5} {n:>5} {wr:>5.1f}% {avg:>+7.2f}% {cum:>+7.1%}")

    # By day-of-week
    dow_pnl = defaultdict(list)
    dow_names = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri'}
    for t in trades:
        di = t['di']
        if di < len(dates):
            d = dates[di].dayofweek
            dow_pnl[d].append(t['pnl_pct'])

    print(f"    DoW    {'N':>5} {'WR':>6} {'AvgPnL':>8} {'Cum':>8}")
    for d in range(5):
        if d in dow_pnl:
            pnls = dow_pnl[d]
            n = len(pnls)
            wr = sum(1 for p in pnls if p > 0) / n * 100
            avg = np.mean(pnls)
            cum = np.prod([1 + p / 100 for p in pnls]) - 1
            print(f"    {dow_names[d]:>5} {n:>5} {wr:>5.1f}% {avg:>+7.2f}% {cum:>+7.1%}")


# ============================================================
# BACKTEST
# ============================================================
def backtest_v3(C, O, H, L, NS, ND, dates, syms, sigs,
                top_n=1, min_rank=0.7, atr_stop=2.5,
                min_confidence=3, use_ker_gate=True,
                hold_days=5,
                sector_limit=1,
                filter_months=None,   # list of months to skip
                filter_dow=None,      # list of DOWs to skip
                leverage=1.0,
                start_di=60, end_di=None):
    """Backtest with calendar filters."""
    combo_rank = sigs['combo_rank']
    ker_regime = sigs['ker_regime']
    n_signals = sigs['n_signals']

    if end_di is None:
        end_di = ND - 1

    # Build sector map
    sym_to_sector = {}
    for gname, gsyms in COMMODITY_GROUPS.items():
        for s in gsyms:
            sym_to_sector[s] = gname

    equity = CASH0
    peak = equity
    max_dd = 0.0
    positions = []
    trades = []

    for di in range(max(start_di, 1), end_di):
        d = dates[di]

        # Calendar filter
        if filter_months and d.month in filter_months:
            # Still manage existing positions, just skip new entries
            pass
        if filter_dow and d.dayofweek in filter_dow:
            pass

        daily_pnl = 0
        new_positions = []

        for si, edi, ep, sp, alloc in positions:
            c = C[si, di]
            if np.isnan(c):
                new_positions.append((si, edi, ep, sp, alloc))
                continue
            exit_r = None
            if c < sp:
                exit_r = 'stop'
            elif di - edi >= hold_days:
                exit_r = 'hold'
            if exit_r:
                pnl = (c - ep) / ep - COMM
                profit = equity * alloc * leverage * pnl
                daily_pnl += profit
                trades.append({
                    'pnl_abs': profit, 'pnl_pct': pnl * 100 * leverage,
                    'days': di - edi + 1, 'di': di, 'year': d.year,
                    'sym': syms[si], 'reason': exit_r,
                })
            else:
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

        held = {p[0] for p in positions}
        if len(positions) >= top_n:
            continue

        # Skip entry on filtered calendar days
        if filter_months and d.month in filter_months:
            continue
        if filter_dow and d.dayofweek in filter_dow:
            continue

        # Sector count
        sector_count = defaultdict(int)
        for si_p, *_ in positions:
            sname = syms[si_p] if si_p < len(syms) else ''
            sec = sym_to_sector.get(sname, 'OTHER')
            sector_count[sec] += 1

        # Entry
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

            # Sector limit
            sname = syms[si] if si < len(syms) else ''
            sec = sym_to_sector.get(sname, 'OTHER')
            if sector_count[sec] >= sector_limit:
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
            positions.append((si, di + 1, ep, ep - atr_stop * atr, alloc))
            held.add(si)

    for si, edi, ep, sp, alloc in positions:
        c = C[si, ND - 1]
        if not np.isnan(c) and c > 0:
            pnl = (c - ep) / ep - COMM
            equity += equity * alloc * leverage * pnl

    return trades, equity, max_dd


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
    print(f"  {label}: {len(trades)}t WR={wr:.1f}% ann={ann:+.1f}% DD={max_dd:.1f}% "
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
        print(f"    {y}: {ys['n']}t WR={ys['w']/ys['n']*100:.1f}% cum={cum:+.1%}")
    return {'n': len(trades), 'wr': wr, 'dd': max_dd, 'ann': ann, 'sh': sh, 'eq': equity}


# ============================================================
# WALK-FORWARD VALIDATION
# ============================================================
def walk_forward(C, O, H, L, NS, ND, dates, syms, sigs,
                 train_years=3, test_years=1,
                 top_n=1, hold_days=5, atr_stop=2.5,
                 min_confidence=3, use_ker_gate=True):
    """
    Rolling walk-forward: train on past N years, test on next 1 year.
    Parameters are NOT optimized — just testing robustness over time.
    """
    print(f"\n{'='*70}")
    print(f"  WALK-FORWARD VALIDATION (train={train_years}y, test={test_years}y)")
    print(f"  Config: tn={top_n}, hd={hold_days}, conf≥{min_confidence}, KER={use_ker_gate}")
    print(f"{'='*70}")

    years = sorted(set(d.year for d in dates))
    all_trades = []

    start_year = years[0]
    while True:
        train_end = start_year + train_years - 1
        test_year = train_end + 1
        if test_year > years[-1]:
            break

        # Find test period indices
        test_start = None
        test_end_idx = None
        for i, d in enumerate(dates):
            if d.year == test_year and test_start is None:
                test_start = i
            if d.year == test_year:
                test_end_idx = i
        if test_start is None:
            start_year += 1
            continue

        # Run backtest on test year only
        trades, eq, dd = backtest_v3(
            C, O, H, L, NS, ND, dates, syms, sigs,
            top_n=top_n, hold_days=hold_days, atr_stop=atr_stop,
            min_confidence=min_confidence, use_ker_gate=use_ker_gate,
            start_di=test_start, end_di=test_end_idx + 1)

        # Filter to test-year trades only
        test_trades = [t for t in trades if dates[t['di']].year == test_year]
        all_trades.extend(test_trades)

        if test_trades:
            n = len(test_trades)
            nw = sum(1 for t in test_trades if t['pnl_pct'] > 0)
            wr = nw / n * 100
            avg = np.mean([t['pnl_pct'] for t in test_trades])
            print(f"  {test_year}: {n}t WR={wr:.1f}% avg={avg:+.2f}%", flush=True)
        else:
            print(f"  {test_year}: no trades", flush=True)

        start_year += 1

    # Aggregate WF results
    if all_trades:
        nw = sum(1 for t in all_trades if t['pnl_pct'] > 0)
        wr = nw / len(all_trades) * 100
        avg = np.mean([t['pnl_pct'] for t in all_trades])
        cum = np.prod([1 + t['pnl_pct'] / 100 for t in all_trades]) - 1
        print(f"\n  WF Total: {len(all_trades)}t WR={wr:.1f}% avg={avg:+.2f}% cum={cum:+.1%}")
        return all_trades
    return []


def main():
    t0 = time.time()
    print("=" * 70)
    print("  V3: WALK-FORWARD + CALENDAR + STOP OPTIMIZATION")
    print("=" * 70)

    C, O, H, L, V, OI, NS, ND, dates, syms = load_all_data(start='2016-01-01')
    print(f"  {NS} sym, {ND} days, "
          f"{dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")

    sigs = compute_signals(C, O, H, L, V, OI, NS, ND)

    bt_2019 = None
    for i, d in enumerate(dates):
        if d >= pd.Timestamp('2019-01-01'):
            bt_2019 = i
            break

    # === 1. Walk-Forward Validation ===
    wf_trades = walk_forward(C, O, H, L, NS, ND, dates, syms, sigs,
                             train_years=3, test_years=1,
                             top_n=1, hold_days=5, atr_stop=2.5,
                             min_confidence=3, use_ker_gate=True)

    # Also WF with different params
    print()
    wf_trades2 = walk_forward(C, O, H, L, NS, ND, dates, syms, sigs,
                              train_years=3, test_years=1,
                              top_n=1, hold_days=5, atr_stop=3.5,
                              min_confidence=2, use_ker_gate=True)

    print()
    wf_trades3 = walk_forward(C, O, H, L, NS, ND, dates, syms, sigs,
                              train_years=3, test_years=1,
                              top_n=3, hold_days=5, atr_stop=2.5,
                              min_confidence=2, use_ker_gate=True)

    # === 2. Stop Loss Optimization ===
    print("\n" + "=" * 70)
    print("  STOP LOSS OPTIMIZATION (2019-2026)")
    print("=" * 70)

    for sl in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]:
        trades, eq, dd = backtest_v3(
            C, O, H, L, NS, ND, dates, syms, sigs,
            top_n=1, hold_days=5, atr_stop=sl,
            min_confidence=3, use_ker_gate=True,
            start_di=bt_2019)
        analyze(trades, eq, dd, f"stop={sl}")

    # === 3. Calendar Analysis ===
    print("\n" + "=" * 70)
    print("  CALENDAR ANALYSIS")
    print("=" * 70)

    # Baseline trades for calendar analysis
    trades_base, _, _ = backtest_v3(
        C, O, H, L, NS, ND, dates, syms, sigs,
        top_n=1, hold_days=5, atr_stop=2.5,
        min_confidence=3, use_ker_gate=True,
        start_di=bt_2019)
    analyze_calendar(trades_base, dates, sigs, "Baseline")

    # Calendar filter sweep
    print("\n  Calendar filter sweep:")
    for skip_months in [None, [1], [12], [1, 12], [7, 8], [10, 11, 12]]:
        for skip_dow in [None, [0], [4], [0, 4]]:
            trades, eq, dd = backtest_v3(
                C, O, H, L, NS, ND, dates, syms, sigs,
                top_n=1, hold_days=5, atr_stop=2.5,
                min_confidence=3, use_ker_gate=True,
                filter_months=skip_months, filter_dow=skip_dow,
                start_di=bt_2019)
            if not trades or len(trades) < 10:
                continue
            nw = sum(1 for t in trades if t['pnl_pct'] > 0)
            wr = nw / len(trades) * 100
            n_days = max(1, trades[-1]['di'] - trades[0]['di'])
            ann = ((eq / CASH0) ** (1 / max(1.0, n_days / 252)) - 1) * 100
            ap = [t['pnl_abs'] for t in sorted(trades, key=lambda x: x['di'])]
            rets = np.array(ap) / CASH0
            sh = np.mean(rets) / np.std(rets) * np.sqrt(252) if np.std(rets) > 0 else 0
            m_s = str(skip_months) if skip_months else "None"
            d_s = str(skip_dow) if skip_dow else "None"
            print(f"    skip_m={m_s:>20} skip_dow={d_s:>10} → {len(trades)}t WR={wr:.1f}% ann={ann:+.1f}% DD={dd:.1f}% Sh={sh:.2f}")

    # === 4. Top-N with sector limits ===
    print("\n" + "=" * 70)
    print("  TOP-N + SECTOR LIMITS (2019-2026)")
    print("=" * 70)

    for tn in [1, 2, 3, 5]:
        for sl in [1, 2, 99]:
            for atr_stop in [2.5, 3.5]:
                trades, eq, dd = backtest_v3(
                    C, O, H, L, NS, ND, dates, syms, sigs,
                    top_n=tn, hold_days=5, atr_stop=atr_stop,
                    min_confidence=3, use_ker_gate=True,
                    sector_limit=sl,
                    start_di=bt_2019)
                if not trades or len(trades) < 10:
                    continue
                nw = sum(1 for t in trades if t['pnl_pct'] > 0)
                wr = nw / len(trades) * 100
                n_days = max(1, trades[-1]['di'] - trades[0]['di'])
                ann = ((eq / CASH0) ** (1 / max(1.0, n_days / 252)) - 1) * 100
                ap = [t['pnl_abs'] for t in sorted(trades, key=lambda x: x['di'])]
                rets_arr = np.array(ap) / CASH0
                sh_val = np.mean(rets_arr) / np.std(rets_arr) * np.sqrt(252) if np.std(rets_arr) > 0 else 0
                print(f"  tn={tn} sec_lim={sl:>2} atr={atr_stop} → {len(trades)}t WR={wr:.1f}% ann={ann:+.1f}% DD={dd:.1f}% Sh={sh_val:.2f}")

    # === 5. Full 10-year best config ===
    print("\n" + "=" * 70)
    print("  FULL 2016-2026 BEST CONFIGS")
    print("=" * 70)

    configs = [
        (1, 2.5, 3, True, None, None, "V1 best: tn1 stop2.5 conf3 KER"),
        (1, 3.5, 3, True, None, None, "tn1 stop3.5 conf3 KER"),
        (1, 4.0, 3, True, None, None, "tn1 stop4.0 conf3 KER"),
        (3, 2.5, 3, True, None, None, "tn3 stop2.5 conf3 KER"),
        (3, 3.5, 2, True, None, None, "tn3 stop3.5 conf2 KER"),
    ]

    for tn, atr_s, mc, kg, fm, fd, label in configs:
        trades, eq, dd = backtest_v3(
            C, O, H, L, NS, ND, dates, syms, sigs,
            top_n=tn, hold_days=5, atr_stop=atr_s,
            min_confidence=mc, use_ker_gate=kg,
            filter_months=fm, filter_dow=fd,
            start_di=60)
        print(f"\n  {label}")
        analyze(trades, eq, dd, label)

    print(f"\n[V3] Done. {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
