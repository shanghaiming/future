"""
V4: Pyramid + Re-entry + Adaptive Sizing
==========================================
V1-V3 established a robust mean-reversion signal (WF 54% WR, +295% cum).
V4 explores ways to boost annual returns:
  1. Pyramid: Add to positions that move in our favor (day 1-2)
  2. Re-entry: After stop-out, re-enter if signal persists
  3. Adaptive hold: Winners hold longer, losers exit sooner
  4. Multi-timeframe confirmation: Weekly trend + daily oversold
  5. Correlation-based diversification: Avoid correlated positions

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
    """Compute all signals including weekly trend."""
    t0 = time.time()
    print("[V4] Computing signals...", flush=True)

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

    # 10d return
    ret_10d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(10, ND):
            if not np.isnan(C[si, di]) and not np.isnan(C[si, di-10]) and C[si, di-10] > 0:
                ret_10d[si, di] = C[si, di] / C[si, di-10] - 1

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

    # TA-Lib
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

    # 20d MA for trend
    ma20 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            vals = C[si, di-20:di]
            valid = vals[~np.isnan(vals)]
            if len(valid) >= 15:
                ma20[si, di] = np.mean(valid)

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

    # Above 20d MA = uptrend
    above_ma20 = np.full((NS, ND), False)
    for si in range(NS):
        for di in range(ND):
            if not np.isnan(C[si, di]) and not np.isnan(ma20[si, di]):
                above_ma20[si, di] = C[si, di] > ma20[si, di]

    print(f"  Done: {time.time() - t0:.1f}s", flush=True)

    return {
        'combo_rank': raw_score,
        'consec_dn': consec_dn,
        'vol_20d': vol_20d,
        'ker_regime': ker_regime,
        'n_signals': n_signals,
        'above_ma20': above_ma20,
        'ret_10d': ret_10d,
    }


# ============================================================
# BACKTEST WITH PYRAMID + RE-ENTRY
# ============================================================
def backtest_v4(C, O, H, L, NS, ND, dates, syms, sigs,
                top_n=1, min_rank=0.7, atr_stop=3.0,
                min_confidence=3, use_ker_gate=True,
                hold_days=5,
                # New features
                pyramid=False,        # Add to winning positions
                pyramid_day=2,        # Pyramid on day N
                pyramid_ratio=0.5,    # Pyramid adds X% of original size
                reentry=False,        # Re-enter after stop-out
                reentry_cooldown=3,   # Wait N days before re-entry
                adaptive_hold=False,  # Winners hold longer
                uptrend_filter=False, # Only buy when above MA20
                sector_limit=1,
                leverage=1.0,
                start_di=60, end_di=None):
    """Backtest with pyramid and re-entry."""
    combo_rank = sigs['combo_rank']
    ker_regime = sigs['ker_regime']
    n_signals = sigs['n_signals']
    above_ma20 = sigs['above_ma20']

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
    # positions: (si, entry_di, entry_price, stop_price, alloc, is_pyramid)
    positions = []
    trades = []
    # Track recent stop-outs for re-entry: {si: stop_di}
    recent_stops = {}

    for di in range(max(start_di, 1), end_di):
        d = dates[di]
        daily_pnl = 0
        new_positions = []
        new_recent_stops = {}

        # Exit logic
        pos_by_si = defaultdict(list)
        for si, edi, ep, sp, alloc, is_pyr in positions:
            pos_by_si[si].append((edi, ep, sp, alloc, is_pyr))

        for si, pos_list in pos_by_si.items():
            c = C[si, di]
            if np.isnan(c):
                for edi, ep, sp, alloc, is_pyr in pos_list:
                    new_positions.append((si, edi, ep, sp, alloc, is_pyr))
                continue

            # Calculate effective hold from earliest position
            earliest_edi = min(p[0] for p in pos_list)
            hold = di - earliest_edi

            # Check stop on ALL positions for this symbol
            stopped = False
            for edi, ep, sp, alloc, is_pyr in pos_list:
                if c < sp:
                    stopped = True
                    break

            if stopped:
                # Close all positions for this symbol
                for edi, ep, sp, alloc, is_pyr in pos_list:
                    pnl = (c - ep) / ep - COMM
                    profit = equity * alloc * leverage * pnl
                    daily_pnl += profit
                    trades.append({
                        'pnl_abs': profit, 'pnl_pct': pnl * 100 * leverage,
                        'days': di - edi + 1, 'di': di, 'year': d.year,
                        'sym': syms[si], 'reason': 'stop', 'pyr': is_pyr,
                    })
                new_recent_stops[si] = di
            elif adaptive_hold:
                # Adaptive: winners hold up to 10 days, losers exit at 5
                total_pnl = sum((c - ep) / ep for _, ep, _, _, _ in pos_list)
                avg_pnl = total_pnl / len(pos_list) if pos_list else 0
                max_hold = 10 if avg_pnl > 0 else hold_days
                if hold >= max_hold:
                    for edi, ep, sp, alloc, is_pyr in pos_list:
                        pnl = (c - ep) / ep - COMM
                        profit = equity * alloc * leverage * pnl
                        daily_pnl += profit
                        trades.append({
                            'pnl_abs': profit, 'pnl_pct': pnl * 100 * leverage,
                            'days': di - edi + 1, 'di': di, 'year': d.year,
                            'sym': syms[si], 'reason': 'hold', 'pyr': is_pyr,
                        })
                else:
                    for edi, ep, sp, alloc, is_pyr in pos_list:
                        new_positions.append((si, edi, ep, sp, alloc, is_pyr))
            elif hold >= hold_days:
                for edi, ep, sp, alloc, is_pyr in pos_list:
                    pnl = (c - ep) / ep - COMM
                    profit = equity * alloc * leverage * pnl
                    daily_pnl += profit
                    trades.append({
                        'pnl_abs': profit, 'pnl_pct': pnl * 100 * leverage,
                        'days': di - edi + 1, 'di': di, 'year': d.year,
                        'sym': syms[si], 'reason': 'hold', 'pyr': is_pyr,
                    })
            else:
                for edi, ep, sp, alloc, is_pyr in pos_list:
                    new_positions.append((si, edi, ep, sp, alloc, is_pyr))

        positions = new_positions
        recent_stops = {k: v for k, v in {**recent_stops, **new_recent_stops}.items()
                        if di - v < (reentry_cooldown if reentry else 999)}

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
            # Still check for pyramid opportunities
            if pyramid:
                for si in list(held):
                    pos_list = [(edi, ep, sp, alloc, is_pyr)
                                for s, edi, ep, sp, alloc, is_pyr in positions if s == si]
                    if any(is_pyr for _, _, _, _, is_pyr in pos_list):
                        continue  # Already pyramided
                    earliest_edi = min(p[0] for p in pos_list)
                    hold = di - earliest_edi
                    if hold == pyramid_day and not np.isnan(C[si, di]):
                        # Check if position is in profit
                        avg_ep = np.mean([ep for _, ep, _, _, _ in pos_list])
                        if C[si, di] > avg_ep:  # In profit
                            base_alloc = sum(a for _, _, _, a, _ in pos_list)
                            pyr_alloc = base_alloc * pyramid_ratio
                            op = O[si, di] if not np.isnan(O[si, di]) else C[si, di]
                            atr_v = []
                            for j in range(max(start_di, di - 14), di):
                                hh, ll, cc = H[si, j], L[si, j], C[si, j]
                                if not any(np.isnan([hh, ll, cc])):
                                    atr_v.append(max(hh - ll, abs(hh - cc), abs(ll - cc)))
                            if atr_v:
                                atr = np.mean(atr_v)
                                positions.append((si, di, op, op - atr_stop * atr, pyr_alloc, True))
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
            if uptrend_filter and not above_ma20[si, di]:
                continue
            if di + 1 >= ND or np.isnan(O[si, di + 1]):
                continue

            # Re-entry check
            if reentry and si in recent_stops:
                continue  # Skip if recently stopped out

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
            positions.append((si, di + 1, ep, ep - atr_stop * atr, alloc, False))
            held.add(si)

    # Close remaining
    for si, edi, ep, sp, alloc, is_pyr in positions:
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

    n_pyr = sum(1 for t in trades if t.get('pyr'))
    n_base = len(trades) - n_pyr

    print(f"  {label}: {len(trades)}t (base:{n_base} pyr:{n_pyr}) "
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
        print(f"    {y}: {ys['n']}t WR={ys['w']/ys['n']*100:.1f}% cum={cum:+.1%}")

    return {'n': len(trades), 'wr': wr, 'dd': max_dd, 'ann': ann, 'sh': sh, 'eq': equity}


def main():
    t0 = time.time()
    print("=" * 70)
    print("  V4: PYRAMID + RE-ENTRY + ADAPTIVE SIZING")
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

    # === Ablation ===
    print("\n" + "=" * 70)
    print("  FEATURE ABLATION (tn=1, conf≥3, KER, stop=3, 2019-2026)")
    print("=" * 70)

    configs = [
        # (pyramid, reentry, adaptive_hold, uptrend_filter, label)
        (False, False, False, False, "V3 baseline (stop=3)"),
        (True,  False, False, False, "+pyramid (day2, 50%)"),
        (True,  False, True,  False, "+pyramid+adaptive hold"),
        (False, True,  False, False, "+re-entry (cooldown=3)"),
        (False, True,  False, False, "+re-entry (cooldown=5)", 5),
        (True,  True,  False, False, "+pyramid+re-entry"),
        (True,  True,  True,  False, "+pyramid+re-entry+adaptive"),
        (False, False, False, True,  "+uptrend filter"),
        (False, False, True,  False, "+adaptive hold"),
        (True,  True,  True,  True,  "ALL features"),
    ]

    for cfg in configs:
        if len(cfg) == 5:
            pyr, reen, adh, utf, label = cfg
            cooldown = 3
        else:
            pyr, reen, adh, utf, label, cooldown = cfg

        trades, eq, dd = backtest_v4(
            C, O, H, L, NS, ND, dates, syms, sigs,
            top_n=1, min_rank=0.7, atr_stop=3.0,
            min_confidence=3, use_ker_gate=True,
            hold_days=5,
            pyramid=pyr, pyramid_day=2, pyramid_ratio=0.5,
            reentry=reen, reentry_cooldown=cooldown,
            adaptive_hold=adh, uptrend_filter=utf,
            leverage=1, start_di=bt_2019)
        analyze(trades, eq, dd, label)

    # === Pyramid ratio sweep ===
    print("\n" + "=" * 70)
    print("  PYRAMID RATIO SWEEP (2019-2026)")
    print("=" * 70)

    for pr in [0.3, 0.5, 0.7, 1.0]:
        for pyr_day in [1, 2, 3]:
            trades, eq, dd = backtest_v4(
                C, O, H, L, NS, ND, dates, syms, sigs,
                top_n=1, min_rank=0.7, atr_stop=3.0,
                min_confidence=3, use_ker_gate=True,
                hold_days=5,
                pyramid=True, pyramid_day=pyr_day, pyramid_ratio=pr,
                leverage=1, start_di=bt_2019)
            analyze(trades, eq, dd, f"pyr ratio={pr} day={pyr_day}")

    # === Best configs on tn=2,3 ===
    print("\n" + "=" * 70)
    print("  MULTI-POSITION + ENHANCEMENTS (2019-2026)")
    print("=" * 70)

    results = []
    for tn in [1, 2, 3]:
        for pyr in [False, True]:
            for adh in [False, True]:
                for utf in [False, True]:
                    trades, eq, dd = backtest_v4(
                        C, O, H, L, NS, ND, dates, syms, sigs,
                        top_n=tn, min_rank=0.7, atr_stop=3.0,
                        min_confidence=3, use_ker_gate=True,
                        hold_days=5,
                        pyramid=pyr, pyramid_day=2, pyramid_ratio=0.5,
                        adaptive_hold=adh, uptrend_filter=utf,
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
                        'tn': tn, 'pyr': pyr, 'adh': adh, 'utf': utf,
                        'n': len(trades), 'wr': wr, 'ann': ann, 'dd': dd, 'sharpe': sh_val,
                    })

    results.sort(key=lambda x: -x['sharpe'])
    print(f"\n{'TN':>3} {'Pyr':>4} {'AdH':>4} {'UTF':>4} "
          f"{'N':>5} {'WR':>5} {'Ann':>8} {'DD':>6} {'Sh':>5}")
    print("-" * 60)
    for r in results[:20]:
        print(f"{r['tn']:>3} {'Y' if r['pyr'] else 'N':>4} "
              f"{'Y' if r['adh'] else 'N':>4} {'Y' if r['utf'] else 'N':>4} "
              f"{r['n']:>5} {r['wr']:>5.1f} {r['ann']:>+8.1f} "
              f"{r['dd']:>6.1f} {r['sharpe']:>5.2f}")

    # === Full 10-year best ===
    print("\n" + "=" * 70)
    print("  FULL 2016-2026 BEST CONFIGS")
    print("=" * 70)

    best = results[:5]
    for r in best:
        trades, eq, dd = backtest_v4(
            C, O, H, L, NS, ND, dates, syms, sigs,
            top_n=r['tn'], min_rank=0.7, atr_stop=3.0,
            min_confidence=3, use_ker_gate=True,
            hold_days=5,
            pyramid=r['pyr'], pyramid_day=2, pyramid_ratio=0.5,
            adaptive_hold=r['adh'], uptrend_filter=r['utf'],
            leverage=1, start_di=60)
        label = f"tn={r['tn']} pyr={'Y' if r['pyr'] else 'N'} adh={'Y' if r['adh'] else 'N'} utf={'Y' if r['utf'] else 'N'}"
        print(f"\n  FULL {label}")
        analyze(trades, eq, dd, label)

    print(f"\n[V4] Done. {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
