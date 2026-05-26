"""
V2: Dynamic Exit Mean Reversion
================================
V1's entry signal is solid (27.9% ann, Sharpe 1.40) but exit is crude
(fixed 5-day hold). V2 explores:
  1. Trailing stop (lock in gains as position moves in favor)
  2. Mean-reversion profit target (exit when z-score returns to 0)
  3. Time-decay exit (reduce hold after N days if no recovery)
  4. RSI recovery exit (exit when RSI crosses back above threshold)
  5. Multi-position with sector diversification

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
    """Compute signals — same as V1 for entry, plus extras for exit."""
    t0 = time.time()
    print("[V2] Computing signals...", flush=True)

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

    # TA-Lib: RSI, Bollinger, CCI
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

    # Z-score (20d)
    zscore_20 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            w = C[si, di-20:di]
            vv = w[~np.isnan(w)]
            if len(vv) >= 15 and np.std(vv) > 0 and not np.isnan(C[si, di]):
                zscore_20[si, di] = (C[si, di] - np.mean(vv)) / np.std(vv)

    # Composite score + cross-sectional rank
    raw_score = np.full((NS, ND), np.nan)
    for di in range(ND):
        scores = np.full(NS, np.nan)
        for si in range(NS):
            if np.isnan(C[si, di]) or C[si, di] <= 0:
                continue
            s = 0.0
            w_total = 0.0
            cd = consec_dn[si, di]
            s += min(cd / 5.0, 1.0) * 0.20; w_total += 0.20
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

    # Confidence (number of signals firing)
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

    # Sector mapping
    sym_to_sector = {}
    sym_idx = {s: i for i, s in enumerate(range(NS))}  # placeholder
    # Will be set in main()

    print(f"  Done: {time.time() - t0:.1f}s", flush=True)

    return {
        'combo_rank': raw_score,
        'consec_dn': consec_dn,
        'vol_20d': vol_20d,
        'ker_10': ker_10,
        'ker_regime': ker_regime,
        'n_signals': n_signals,
        'rsi14': rsi14,
        'atr14': atr14,
        'zscore_20': zscore_20,
    }


# ============================================================
# BACKTEST WITH DYNAMIC EXITS
# ============================================================
def backtest_v2(C, O, H, L, NS, ND, dates, syms, sigs,
                top_n=1, min_rank=0.7, atr_stop=2.5,
                min_confidence=3, use_ker_gate=True,
                max_hold=10,        # max days to hold
                trail_atr=0,        # trailing stop in ATR multiples (0=off)
                profit_target=0,    # profit target as % (0=off)
                zscore_exit=False,  # exit when z-score returns to 0
                rsi_exit=False,     # exit when RSI crosses above 50
                sector_limit=1,     # max positions per sector
                leverage=1.0,
                start_di=60, end_di=None):
    """
    Mean reversion with dynamic exit strategies.
    Signal at close[di], enter at open[di+1]. No look-ahead.
    """
    combo_rank = sigs['combo_rank']
    vol_20d = sigs['vol_20d']
    ker_regime = sigs['ker_regime']
    n_signals = sigs['n_signals']
    rsi14 = sigs['rsi14']
    atr14 = sigs['atr14']
    zscore_20 = sigs['zscore_20']

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
    # positions: (si, entry_di, entry_price, stop_price, alloc, direction, highest_since_entry)
    positions = []
    trades = []

    for di in range(max(start_di, 1), end_di):
        d = dates[di]
        daily_pnl = 0
        new_positions = []

        # Exit logic
        for si, edi, ep, sp, alloc, direction, high_water in positions:
            c = C[si, di]
            if np.isnan(c):
                new_positions.append((si, edi, ep, sp, alloc, direction, high_water))
                continue

            # Update high water mark for trailing stop
            if direction > 0:
                new_high = max(high_water, c)
            else:
                new_high = min(high_water, c) if high_water > 0 else c

            exit_r = None
            hold_days = di - edi

            # 1. ATR stop loss
            if direction > 0 and c < sp:
                exit_r = 'stop'
            elif direction < 0 and c > sp:
                exit_r = 'stop'

            # 2. Trailing stop
            if exit_r is None and trail_atr > 0 and hold_days >= 2:
                if direction > 0:
                    trail = new_high - trail_atr * atr14[si, di] if not np.isnan(atr14[si, di]) else sp
                    if c < trail and c > ep:  # only trail if in profit
                        exit_r = 'trail'
                elif direction < 0:
                    trail = new_high + trail_atr * atr14[si, di] if not np.isnan(atr14[si, di]) else sp
                    if c > trail and c < ep:
                        exit_r = 'trail'

            # 3. Profit target
            if exit_r is None and profit_target > 0:
                pnl_pct = direction * (c - ep) / ep
                if pnl_pct >= profit_target:
                    exit_r = 'target'

            # 4. Z-score exit (mean reversion complete)
            if exit_r is None and zscore_exit and hold_days >= 2:
                zs = zscore_20[si, di]
                if not np.isnan(zs):
                    if direction > 0 and zs > -0.2:
                        exit_r = 'zscore'
                    elif direction < 0 and zs < 0.2:
                        exit_r = 'zscore'

            # 5. RSI recovery exit
            if exit_r is None and rsi_exit and hold_days >= 2:
                rsi = rsi14[si, di]
                if not np.isnan(rsi):
                    if direction > 0 and rsi > 50:
                        exit_r = 'rsi'
                    elif direction < 0 and rsi < 50:
                        exit_r = 'rsi'

            # 6. Max hold period
            if exit_r is None and hold_days >= max_hold:
                exit_r = 'hold'

            if exit_r:
                pnl = direction * (c - ep) / ep - COMM
                profit = equity * alloc * leverage * pnl
                daily_pnl += profit
                trades.append({
                    'pnl_abs': profit, 'pnl_pct': pnl * 100 * leverage,
                    'days': hold_days + 1, 'di': di, 'year': d.year,
                    'sym': syms[si], 'reason': exit_r, 'dir': direction,
                })
            else:
                new_positions.append((si, edi, ep, sp, alloc, direction, new_high))

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
            # ATR stop
            atr_v = []
            for j in range(max(start_di, di - 14), di):
                hh, ll, cc = H[si, j], L[si, j], C[si, j]
                if not any(np.isnan([hh, ll, cc])):
                    atr_v.append(max(hh - ll, abs(hh - cc), abs(ll - cc)))
            if not atr_v:
                continue
            atr = np.mean(atr_v)
            positions.append((si, di + 1, ep, ep - atr_stop * atr, alloc, 1, ep))
            held.add(si)

    # Close remaining
    for si, edi, ep, sp, alloc, direction, _ in positions:
        c = C[si, ND - 1]
        if not np.isnan(c) and c > 0:
            pnl = direction * (c - ep) / ep - COMM
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

    # Exit reason breakdown
    reasons = defaultdict(lambda: {'n': 0, 'w': 0, 'pnl': []})
    for t in trades:
        r = t['reason']
        reasons[r]['n'] += 1
        if t['pnl_pct'] > 0:
            reasons[r]['w'] += 1
        reasons[r]['pnl'].append(t['pnl_pct'])

    print(f"  {label}: {len(trades)}t WR={wr:.1f}% ann={ann:+.1f}% DD={max_dd:.1f}% "
          f"Sh={sh:.2f} eq={equity:,.0f}")

    # Exit breakdown
    for r in ['stop', 'hold', 'trail', 'target', 'zscore', 'rsi']:
        if r in reasons:
            rs = reasons[r]
            rwr = rs['w'] / rs['n'] * 100
            avg_pnl = np.mean(rs['pnl'])
            print(f"    {r:>7}: {rs['n']:>4}t WR={rwr:.1f}% avg={avg_pnl:+.2f}%")

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


def main():
    t0 = time.time()
    print("=" * 70)
    print("  V2: DYNAMIC EXIT MEAN REVERSION")
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

    # === Exit Strategy Ablation ===
    print("\n" + "=" * 70)
    print("  EXIT STRATEGY ABLATION (tn=1, conf≥3, KER, 2019-2026)")
    print("=" * 70)

    exit_configs = [
        # (max_hold, trail_atr, profit_target, zscore_exit, rsi_exit, label)
        (5,  0,    0,    False, False, "V1 baseline (hd=5)"),
        (7,  0,    0,    False, False, "hold 7d"),
        (10, 0,    0,    False, False, "hold 10d"),
        (15, 0,    0,    False, False, "hold 15d"),
        (5,  1.5,  0,    False, False, "hd5+trail1.5ATR"),
        (5,  2.0,  0,    False, False, "hd5+trail2.0ATR"),
        (7,  1.5,  0,    False, False, "hd7+trail1.5ATR"),
        (10, 1.5,  0,    False, False, "hd10+trail1.5ATR"),
        (10, 2.0,  0,    False, False, "hd10+trail2.0ATR"),
        (10, 0,    0.03, False, False, "hd10+target3%"),
        (10, 0,    0.05, False, False, "hd10+target5%"),
        (10, 0,    0,    True,  False, "hd10+zscore exit"),
        (10, 0,    0,    False, True,  "hd10+RSI exit"),
        (10, 1.5,  0.03, False, False, "hd10+trail+target3%"),
        (10, 1.5,  0,    True,  False, "hd10+trail+zscore"),
        (10, 1.5,  0,    False, True,  "hd10+trail+RSI"),
        (10, 1.5,  0,    True,  True,  "hd10+trail+zscore+RSI"),
        (15, 2.0,  0.05, True,  True,  "hd15+trail+target+zscore+RSI"),
    ]

    for mh, ta, pt, ze, re, label in exit_configs:
        trades, eq, dd = backtest_v2(
            C, O, H, L, NS, ND, dates, syms, sigs,
            top_n=1, min_rank=0.7, atr_stop=2.5,
            min_confidence=3, use_ker_gate=True,
            max_hold=mh, trail_atr=ta, profit_target=pt,
            zscore_exit=ze, rsi_exit=re,
            leverage=1, start_di=bt_2019)
        analyze(trades, eq, dd, label)

    # === Top-N with best exit ===
    print("\n" + "=" * 70)
    print("  TOP-N + BEST EXIT SWEEP (2019-2026)")
    print("=" * 70)

    results = []
    for tn in [1, 2, 3, 5]:
        for mh, ta in [(5, 0), (7, 1.5), (10, 1.5), (10, 2.0), (15, 2.0)]:
            for ze, re in [(False, False), (True, False), (True, True)]:
                trades, eq, dd = backtest_v2(
                    C, O, H, L, NS, ND, dates, syms, sigs,
                    top_n=tn, min_rank=0.7, atr_stop=2.5,
                    min_confidence=3, use_ker_gate=True,
                    max_hold=mh, trail_atr=ta,
                    zscore_exit=ze, rsi_exit=re,
                    leverage=1, start_di=bt_2019)
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
                    'tn': tn, 'mh': mh, 'ta': ta, 'ze': ze, 're': re,
                    'n': len(trades), 'wr': wr, 'ann': ann, 'dd': dd, 'sharpe': sh_val,
                })

    results.sort(key=lambda x: -x['sharpe'])
    print(f"\n{'TN':>3} {'MH':>3} {'TA':>4} {'Z':>3} {'R':>3} "
          f"{'N':>5} {'WR':>5} {'Ann':>8} {'DD':>6} {'Sh':>5}")
    print("-" * 60)
    for r in results[:25]:
        print(f"{r['tn']:>3} {r['mh']:>3} {r['ta']:>4.1f} "
              f"{'Y' if r['ze'] else 'N':>3} {'Y' if r['re'] else 'N':>3} "
              f"{r['n']:>5} {r['wr']:>5.1f} {r['ann']:>+8.1f} "
              f"{r['dd']:>6.1f} {r['sharpe']:>5.2f}")

    # === Best configs — yearly ===
    print("\n" + "=" * 70)
    print("  BEST CONFIGS — YEARLY (2019-2026)")
    print("=" * 70)

    best = results[:5]
    for r in best:
        trades, eq, dd = backtest_v2(
            C, O, H, L, NS, ND, dates, syms, sigs,
            top_n=r['tn'], min_rank=0.7, atr_stop=2.5,
            min_confidence=3, use_ker_gate=True,
            max_hold=r['mh'], trail_atr=r['ta'],
            zscore_exit=r['ze'], rsi_exit=r['re'],
            leverage=1, start_di=bt_2019)
        label = f"tn={r['tn']} mh={r['mh']} ta={r['ta']:.1f}"
        print(f"\n  --- {label} ---")
        analyze(trades, eq, dd, label)

    # === Full 10-year ===
    print("\n" + "=" * 70)
    print("  FULL 2016-2026 (10 years)")
    print("=" * 70)

    for r in best[:3]:
        trades, eq, dd = backtest_v2(
            C, O, H, L, NS, ND, dates, syms, sigs,
            top_n=r['tn'], min_rank=0.7, atr_stop=2.5,
            min_confidence=3, use_ker_gate=True,
            max_hold=r['mh'], trail_atr=r['ta'],
            zscore_exit=r['ze'], rsi_exit=r['re'],
            leverage=1, start_di=60)
        label = f"full tn={r['tn']} mh={r['mh']} ta={r['ta']:.1f}"
        print(f"\n  {label}")
        analyze(trades, eq, dd, label)

    print(f"\n[V2] Done. {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
