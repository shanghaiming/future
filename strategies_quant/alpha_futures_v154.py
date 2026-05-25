"""
Alpha Futures V154 — Day-of-Week and Intraday Pattern Effects
=============================================================================
Goal: Explore whether weekday patterns (Monday gap, Friday reduction) can
      improve timing on Chinese commodity futures.

Section 0: Baseline (V146 best: Cross+Corr DD70/60/40/20 corr<0.5)
Section 1: Weekday analysis (WR and avg PnL by day of week)
Section 2: Skip-day filters (skip Mon, skip Fri, skip Mon+Fri, etc.)
Section 3: Monday gap signal
Section 4: Best combos with WF validation
"""
import sys, os, time, warnings, calendar
import numpy as np
import talib
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alpha_v2 import load_all_data, MIN_TRAIN, CASH0

MULT = {'agfi': 15, 'alfi': 5, 'aufi': 1000, 'bufi': 10, 'cufi': 5, 'fufi': 10,
        'rbfi': 10, 'znfi': 5, 'nifi': 1, 'hcfi': 10, 'spfi': 10, 'ssfi': 5,
        'sffi': 5, 'smfi': 5, 'pbfi': 5, 'snfi': 1, 'rufi': 10, 'wrffi': 10,
        'afi': 10, 'bfi': 10, 'bbfi': 500, 'cffi': 5, 'cfi': 10, 'csfi': 10,
        'ebfi': 5, 'egfi': 10, 'fbfi': 500, 'ifi': 100, 'jfi': 100, 'jmfi': 60,
        'lfi': 5, 'mfi': 10, 'pgfi': 20, 'ppfi': 5, 'vfi': 5, 'yfi': 10,
        'pfi': 10, 'jdfi': 5, 'lhfi': 16, 'pkfi': 5, 'rrfi': 20, 'lrfi': 20,
        'jrfi': 20, 'pmfi': 20, 'whfi': 20, 'rsfi': 20, 'cjfi': 10, 'mafi': 10,
        'apfi': 10, 'cyfi': 5, 'fgfi': 20, 'oifi': 10, 'pfifi': 5, 'rmfi': 10,
        'srfi': 10, 'tafi': 5, 'safi': 20, 'urfi': 20, 'scfi': 1000, 'lufi': 10,
        'bcfi': 5, 'nrfi': 1, 'lgfi': 20, 'brfi': 5, 'lcfi': 1, 'sifi': 5,
        'ni': 1, 'tai': 5}
DEF_MULT = 10
COMM = 0.0003

def annual_return(final, initial, n_days):
    if final <= 0 or initial <= 0 or n_days <= 0: return -100.0
    return (final / initial) ** (1.0 / (n_days / 252)) * 100 - 100


def main():
    print("=" * 130)
    print("  V154 — Day-of-Week and Intraday Pattern Effects")
    print("=" * 130)
    t_start = time.time()
    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)
    print(f"  {NS} commodities, {ND} days")

    # ===================== PRECOMPUTE =====================
    print("\n[Precompute]...", flush=True)
    t0 = time.time()

    RET = np.full((NS, ND), np.nan)
    ROC5 = np.full((NS, ND), np.nan)
    ROC20 = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si].astype(np.float64)
        for di in range(1, ND):
            if not np.isnan(c[di]) and not np.isnan(c[di-1]) and c[di-1] > 0:
                RET[si, di] = (c[di] / c[di-1] - 1) * 100
        ROC5[si] = talib.ROC(c, timeperiod=5)
        ROC20[si] = talib.ROC(c, timeperiod=20)

    ATR14 = np.full((NS, ND), np.nan)
    for si in range(NS):
        ATR14[si] = talib.ATR(H[si].astype(np.float64), L[si].astype(np.float64),
                               C[si].astype(np.float64), timeperiod=14)

    ZSCORE = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(21, ND):
            rets = RET[si, di-20:di]
            v = rets[~np.isnan(rets)]
            if len(v) < 10: continue
            s = np.std(v, ddof=1)
            if s > 0 and not np.isnan(RET[si, di]):
                ZSCORE[si, di] = (RET[si, di] - np.mean(v)) / s

    OV_GAP = np.full((NS, ND), np.nan)
    ID_RET = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(ND):
            o, c = O[si, di], C[si, di]
            if not np.isnan(o) and not np.isnan(c):
                if di > 0 and not np.isnan(C[si, di-1]) and C[si, di-1] > 0:
                    OV_GAP[si, di] = (o - C[si, di-1]) / C[si, di-1] * 100
                if o > 0: ID_RET[si, di] = (c - o) / o * 100

    print(f"  Done ({time.time()-t0:.1f}s)")

    # ===================== SIGNAL DEFINITIONS =====================
    def sig_v121(di, edi):
        c = []
        for s in range(NS):
            roc = ROC5[s, di]; zs = ZSCORE[s, di]
            if np.isnan(roc) or np.isnan(zs) or roc <= 1.0 or zs <= 1.5: continue
            rp = ROC5[s, di-1] if di > 0 else np.nan
            if not np.isnan(rp) and roc <= rp: continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            c.append((roc * zs, s, ep, 'v121'))
        return c

    def sig_ov_id(di, edi):
        c = []
        for s in range(NS):
            ov = OV_GAP[s, di]; idr = ID_RET[s, di]; roc = ROC5[s, di]
            if any(np.isnan(x) for x in [ov, idr, roc]): continue
            if ov <= 0.3 or idr <= 0.3 or roc <= 1.0: continue
            zs = ZSCORE[s, di]
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            z_bonus = zs if not np.isnan(zs) and zs > 1.0 else 1.0
            c.append(((ov + idr) * roc * z_bonus * 2, s, ep, 'ov_id'))
        return c

    def sig_final_flag(di, edi):
        c = []
        for s in range(NS):
            roc20 = ROC20[s, di]
            if np.isnan(roc20) or roc20 <= 5.0 or di < 6: continue
            h5 = H[s, di-4:di+1]; l5 = L[s, di-4:di+1]
            if any(np.isnan(x) for x in h5) or any(np.isnan(x) for x in l5): continue
            r5 = np.max(h5) - np.min(l5)
            atr = ATR14[s, di]
            if np.isnan(atr) or atr <= 0 or r5 > atr * 3.0: continue
            h4 = np.max(H[s, di-4:di])
            cp = C[s, di]
            if np.isnan(cp) or cp <= h4: continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            c.append((roc20 * (cp - h4) / atr, s, ep, 'ff'))
        return c

    def sig_union(di, edi):
        all_sigs = {}
        for item in sig_v121(di, edi):
            sc, s, ep, st = item
            if s not in all_sigs: all_sigs[s] = [0, ep, []]
            all_sigs[s][0] += sc * 3
            all_sigs[s][2].append('v121')
        for item in sig_ov_id(di, edi):
            sc, s, ep, st = item
            if s not in all_sigs: all_sigs[s] = [0, ep, []]
            all_sigs[s][0] += sc * 2
            all_sigs[s][2].append('ov_id')
        for item in sig_final_flag(di, edi):
            sc, s, ep, st = item
            if s not in all_sigs: all_sigs[s] = [0, ep, []]
            all_sigs[s][0] += sc
            all_sigs[s][2].append('ff')
        return [(sc, s, ep, '+'.join(sigs)) for s, (sc, ep, sigs) in all_sigs.items()]

    # ===================== HELPER: Correlation between two commodities =====================
    def get_corr(si_a, si_b, di, window=20):
        """Use daily returns instead of overlapping 20-day returns."""
        start_idx = max(0, di - window)
        ret_a = RET[si_a, start_idx:di]
        ret_b = RET[si_b, start_idx:di]
        valid = ~(np.isnan(ret_a) | np.isnan(ret_b))
        n_valid = np.sum(valid)
        if n_valid < 8:
            return 0.5
        ra = ret_a[valid]; rb = ret_b[valid]
        if np.std(ra) == 0 or np.std(rb) == 0:
            return 0.5
        c = np.corrcoef(ra, rb)[0, 1]
        return c if not np.isnan(c) else 0.5

    # ===================== HELPER: DD-based sizing =====================
    def dd_size(pv, high_water, tiers):
        if high_water <= 0:
            return tiers[0][1]
        dd = (pv - high_water) / high_water
        for dd_thresh, size_frac in tiers:
            if dd >= -dd_thresh:
                return size_frac
        return tiers[-1][1]

    # ===================== BACKTEST ENGINE WITH WEEKDAY TRACKING =====================
    def backtest(start_di=MIN_TRAIN, end_di=None,
                 # Weekday filter: set of weekdays to skip (0=Mon..4=Fri)
                 skip_weekdays=None,
                 # Monday gap signal: if True, also enter on Monday gap reversals
                 monday_gap=False,
                 # Month-boundary filter: only enter near month boundaries
                 month_boundary=False,
                 # Month-boundary days: how many days from month start/end
                 month_boundary_days=3,
                 # Standard params
                 dd_tiers=None,
                 max_corr=0.5,
                 hold=1, top_n=2,
                 # Track trades by weekday for analysis
                 track_weekday=False):
        if end_di is None: end_di = ND
        if skip_weekdays is None: skip_weekdays = set()
        if dd_tiers is None:
            dd_tiers = [(0, 0.70), (0.10, 0.60), (0.20, 0.40), (0.30, 0.20)]

        cash = float(CASH0)
        positions = []
        trades = []
        daily_eq = []
        high_water = float(CASH0)

        # Weekday tracking (Chinese futures have night sessions mapped to next calendar day)
        weekday_trades = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: []}  # pnl by entry weekday

        for di in range(start_di, end_di - 1):
            # Mark-to-market
            pv = cash
            for p in positions:
                cp = C[p['si'], di]
                if not np.isnan(cp) and cp > 0:
                    m = MULT.get(p['sym'], DEF_MULT)
                    pv += cp * m * p['lots'] - cp * m * abs(p['lots']) * COMM
            daily_eq.append(pv)
            if pv > high_water:
                high_water = pv

            # Close positions past hold period
            cl = []
            for p in positions:
                if di - p['entry_di'] >= p['hold_days']:
                    ep = C[p['si'], di]
                    if np.isnan(ep) or ep <= 0: ep = p['entry_price']
                    m = MULT.get(p['sym'], DEF_MULT)
                    pnl = (ep - p['entry_price']) * m * p['lots']
                    inv = p['entry_price'] * m * abs(p['lots'])
                    pp = pnl / inv * 100 if inv > 0 else 0
                    cash += ep * m * abs(p['lots']) * (1 - COMM)
                    trades.append(pp)
                    if track_weekday and 'entry_wd' in p:
                        weekday_trades[p['entry_wd']].append(pp)
                    cl.append(p)
            for p in cl: positions.remove(p)

            # DD-based sizing
            pos_size = dd_size(pv, high_water, dd_tiers)
            pos_size = max(0.05, min(0.95, pos_size))

            # --- Enter positions ---
            if len(positions) >= top_n: continue
            edi = di + 1
            if edi >= end_di: continue

            # Weekday filter
            wd = dates[di].weekday()
            if wd in skip_weekdays: continue

            # Month-boundary filter
            if month_boundary:
                dt = dates[di]
                day_of_month = dt.day
                days_in_month = 31  # approximate; we check bounds
                # Check if near month start or end
                near_start = day_of_month <= month_boundary_days
                # For near-end, check if within N days of next month start
                try:
                    last_day = calendar.monthrange(dt.year, dt.month)[1]
                    near_end = (last_day - day_of_month) < month_boundary_days
                except Exception:
                    near_end = False
                if not (near_start or near_end):
                    continue

            held_si = set(p['si'] for p in positions)

            # Standard Cross+Corr signal (V121 + Union, balanced)
            cands_v121 = sig_v121(di, edi)
            cands_union = sig_union(di, edi)
            cands_v121.sort(key=lambda x: -x[0])
            cands_union.sort(key=lambda x: -x[0])

            best_v121 = None
            for c in cands_v121:
                if c[1] not in held_si:
                    best_v121 = c
                    break

            best_union = None
            for c in cands_union:
                if c[1] not in held_si:
                    best_union = c
                    break

            entries = []
            if best_v121 and best_union:
                if best_v121[1] == best_union[1]:
                    entries.append((best_v121[0], best_v121[1], best_v121[2],
                                    'v121+union', pos_size * 1.5))
                else:
                    corr = get_corr(best_v121[1], best_union[1], di)
                    if corr < max_corr:
                        entries.append((best_v121[0], best_v121[1], best_v121[2],
                                        'v121', pos_size))
                        entries.append((best_union[0], best_union[1], best_union[2],
                                        'union', pos_size))
                    else:
                        best = best_v121 if best_v121[0] >= best_union[0] else best_union
                        entries.append((best[0], best[1], best[2], 'best', pos_size))
            elif best_v121:
                entries.append((best_v121[0], best_v121[1], best_v121[2], 'v121', pos_size))
            elif best_union:
                entries.append((best_union[0], best_union[1], best_union[2], 'union', pos_size))

            # Monday gap signal: enter on Monday if overnight gap is negative (reversal)
            # Only add gap signals if we haven't filled positions with regular signals
            # Apply on Monday (0) and Sunday (6, which is weekend night session in Chinese futures)
            entry_si = set(e[1] for e in entries)
            if monday_gap and wd in (0, 6) and len(positions) + len(entries) < top_n:
                gap_cands = []
                for s in range(NS):
                    if s in held_si or s in entry_si: continue
                    ov = OV_GAP[s, di]
                    idr = ID_RET[s, di]
                    roc = ROC5[s, di]
                    if np.isnan(ov) or np.isnan(idr): continue
                    # Weekend gap down reversal: gap down + intraday reversal up
                    if ov < -0.5 and idr > 0.3 and not np.isnan(roc) and roc > 0:
                        ep = O[s, edi]
                        if np.isnan(ep) or ep <= 0: continue
                        score = abs(ov) * idr * 2
                        gap_cands.append((score, s, ep, 'mon_gap', pos_size))
                gap_cands.sort(key=lambda x: -x[0])
                for gc in gap_cands:
                    if len(positions) + len(entries) >= top_n: break
                    if gc[1] not in entry_si:
                        entries.append(gc)
                        entry_si.add(gc[1])

            cash_snapshot = cash  # Bug fix: snapshot before allocation
            n_planned = len(entries)
            for sc, s, pr, sig_str, pct in entries:
                if s in set(p['si'] for p in positions): continue
                if len(positions) >= top_n: break
                cap = cash_snapshot * pct / n_planned  # Equal split
                sym = syms[s]; m = MULT.get(sym, DEF_MULT)
                ct = max(1, int(cap / (pr * m * (1 + COMM))))
                ci = pr * m * ct * (1 + COMM)
                if ci > cash:
                    ct = int(cash * 0.9 / (pr * m * (1 + COMM)))
                    ci = pr * m * ct * (1 + COMM) if ct > 0 else 0
                if ct <= 0 or ci <= 0 or ci > cash: continue
                cash -= ci
                pos_entry = {'si': s, 'entry_price': pr, 'entry_di': edi,
                             'lots': ct, 'dir': 1, 'sym': sym, 'hold_days': hold,
                             'sig': sig_str, 'score': sc}
                if track_weekday:
                    pos_entry['entry_wd'] = wd
                positions.append(pos_entry)

        # Close remaining
        for p in positions:
            ae = end_di - 1
            ep = C[p['si'], min(ae, ND-1)]
            if np.isnan(ep) or ep <= 0: ep = p['entry_price']
            m = MULT.get(p['sym'], DEF_MULT)
            pnl = (ep - p['entry_price']) * m * p['lots']
            inv = p['entry_price'] * m * abs(p['lots'])
            pp = pnl / inv * 100 if inv > 0 else 0
            cash += ep * m * abs(p['lots']) * (1 - COMM)
            trades.append(pp)
            if track_weekday and 'entry_wd' in p:
                weekday_trades[p['entry_wd']].append(pp)

        nd = end_di - start_di
        ann = annual_return(cash, CASH0, nd)
        wr = np.mean([1 if t > 0 else 0 for t in trades]) * 100 if trades else 0
        nt = len(trades)
        if daily_eq:
            eq = np.array(daily_eq); pk = np.maximum.accumulate(eq)
            mdd = np.min((eq - pk) / pk * 100)
            r = np.diff(eq) / eq[:-1]
            r = np.where(np.isfinite(r), r, 0)
            sh = np.mean(r) / np.std(r) * np.sqrt(252) if np.std(r) > 0 else 0
        else:
            mdd = 0; sh = 0
        result = {'ann': ann, 'wr': wr, 'n': nt, 'mdd': mdd, 'sharpe': sh, 'final': cash}
        if track_weekday:
            result['weekday_trades'] = weekday_trades
        return result

    # ===================== PRINTING HELPERS =====================
    DAY_NAMES = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}

    def pr(r, label=""):
        ratio = abs(r['ann'] / r['mdd']) if r['mdd'] != 0 else 0
        print(f"  {label:75s} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f} | R/M={ratio:.2f} | N={r['n']:4d}")

    def walk_forward(skip_weekdays=None, monday_gap=False, month_boundary=False,
                     month_boundary_days=3, dd_tiers=None, max_corr=0.5,
                     hold=1, top_n=2, label=""):
        res = {}
        for yr in [2020, 2021, 2022, 2023, 2024, 2025]:
            ys = ye = None
            for di in range(ND):
                if dates[di].year == yr and ys is None: ys = di
                if dates[di].year == yr: ye = di + 1
            if ys is None: continue
            r = backtest(start_di=ys, end_di=ye,
                         skip_weekdays=skip_weekdays,
                         monday_gap=monday_gap,
                         month_boundary=month_boundary,
                         month_boundary_days=month_boundary_days,
                         dd_tiers=dd_tiers, max_corr=max_corr,
                         hold=hold, top_n=top_n)
            res[yr] = r
        return res

    def print_wf(wf_res, label=""):
        pos = sum(1 for r in wf_res.values() if r['ann'] > 0)
        avg_ann = np.mean([r['ann'] for r in wf_res.values()])
        worst_mdd = min(r['mdd'] for r in wf_res.values())
        ws = " | ".join([f"{yr}:{r['ann']:+.0f}%/{r['mdd']:.0f}%"
                         for yr, r in sorted(wf_res.items())])
        print(f"    {label}")
        print(f"      {pos}/6 pos | Avg={avg_ann:>+7.0f}% | WorstWfMDD={worst_mdd:>5.0f}%")
        print(f"      {ws}")

    # ===================== SECTION 0: BASELINE =====================
    print("\n" + "=" * 130)
    print("  SECTION 0: BASELINE (Cross+Corr DD70/60/40/20 corr<0.5)")
    print("=" * 130)

    baseline = backtest(track_weekday=True)
    pr(baseline, "Baseline: DD70/60/40/20 corr<0.5 hold=1 top_n=2")
    print()

    # Show weekday distribution of trading days in the dataset
    print("  Trading days by weekday in full dataset:")
    wd_counts = {}
    for di in range(ND):
        wd = dates[di].weekday()
        wd_counts[wd] = wd_counts.get(wd, 0) + 1
    ALL_WD = sorted(wd_counts.keys())
    for wd in ALL_WD:
        print(f"    {DAY_NAMES.get(wd, f'D{wd}')}: {wd_counts[wd]} days")

    # ===================== SECTION 1: WEEKDAY ANALYSIS =====================
    print("\n" + "=" * 130)
    print("  SECTION 1: WEEKDAY ANALYSIS (WR and Avg PnL by Day of Week)")
    print("=" * 130)

    wt = baseline.get('weekday_trades', {})
    print(f"\n  {'Weekday':8s} | {'N':>5s} | {'WR':>6s} | {'AvgPnL':>8s} | {'MedPnL':>8s} | {'TotPnL':>10s} | {'Pct of Total':>12s}")
    print(f"  {'-'*8}-+-{'-'*5}-+-{'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}-+-{'-'*12}")

    total_pnl = sum(sum(pnls) for pnls in wt.values())
    for wd in ALL_WD:
        pnls = wt[wd]
        if len(pnls) == 0:
            print(f"  {DAY_NAMES[wd]:8s} | {'0':>5s} | {'N/A':>6s} | {'N/A':>8s} | {'N/A':>8s} | {'0.0':>10s} | {'0.0%':>12s}")
            continue
        n = len(pnls)
        wr = np.mean([1 if p > 0 else 0 for p in pnls]) * 100
        avg_pnl = np.mean(pnls)
        med_pnl = np.median(pnls)
        tot_pnl = sum(pnls)
        pct = tot_pnl / total_pnl * 100 if total_pnl != 0 else 0
        print(f"  {DAY_NAMES[wd]:8s} | {n:>5d} | {wr:>5.1f}% | {avg_pnl:>+7.2f}% | {med_pnl:>+7.2f}% | {tot_pnl:>+9.1f}% | {pct:>11.1f}%")

    # Also run per-weekday backtests (only enter on specific weekdays)
    print(f"\n  Per-Weekday Backtests (only enter on that weekday):")
    for wd in ALL_WD:
        skip = set(ALL_WD) - {wd}
        r = backtest(skip_weekdays=skip, track_weekday=True)
        pr(r, f"Only {DAY_NAMES.get(wd, f'D{wd}')} entries")

    # ===================== SECTION 2: SKIP-DAY FILTERS =====================
    print("\n" + "=" * 130)
    print("  SECTION 2: SKIP-DAY FILTERS")
    print("  Test skipping specific weekdays for entries")
    print("=" * 130)

    skip_configs = [
        (set(), "No skip (baseline)"),
        ({4}, "Skip Friday"),
        ({0}, "Skip Monday"),
        ({0, 4}, "Skip Monday+Friday"),
        ({3}, "Skip Thursday"),
        ({3, 4}, "Skip Thursday+Friday"),
        ({0, 3}, "Skip Monday+Thursday"),
        ({1}, "Skip Tuesday"),
        ({2}, "Skip Wednesday"),
        ({6}, "Skip Sunday"),
        ({0, 6}, "Skip Monday+Sunday"),
        ({4, 6}, "Skip Friday+Sunday"),
        ({0, 4, 6}, "Skip Monday+Friday+Sunday"),
    ]

    section2_results = []
    for skip, label in skip_configs:
        r = backtest(skip_weekdays=skip)
        r['desc'] = label
        r['skip'] = skip
        section2_results.append(r)
        pr(r, label)

    # ===================== SECTION 3: MONDAY GAP SIGNAL =====================
    print("\n" + "=" * 130)
    print("  SECTION 3: MONDAY GAP SIGNAL")
    print("  After weekend gaps, price tends to revert. Use as entry signal.")
    print("  Monday gap: OV<-0.5% + ID>0.3% + ROC5>0")
    print("=" * 130)

    gap_configs = [
        (False, set(), "Baseline (no gap signal)"),
        (True, set(), "With Monday gap signal"),
        (True, {4}, "With Monday gap + skip Friday"),
        (True, {0, 4}, "With Monday gap + skip Mon+Fri (gap enters on Mon only)"),
    ]

    section3_results = []
    for mg, skip, label in gap_configs:
        r = backtest(monday_gap=mg, skip_weekdays=skip)
        r['desc'] = label
        section3_results.append(r)
        pr(r, label)

    # Also test: Monday gap signal only (no regular signals on Monday)
    print(f"\n  Monday gap ONLY (replace regular signals on Monday with gap signal):")
    # This is equivalent to: on Monday use only gap, on other days use regular
    # We simulate by running with gap=True and checking

    # ===================== SECTION 3b: MONTH-BOUNDARY EFFECTS =====================
    print("\n" + "=" * 130)
    print("  SECTION 3b: MONTH-BOUNDARY EFFECTS")
    print("  Test entries near month start/end (position adjustment period)")
    print("=" * 130)

    month_configs = [
        (False, 0, set(), "Baseline (no month filter)"),
        (True, 3, set(), "Month boundary +/- 3 days"),
        (True, 2, set(), "Month boundary +/- 2 days"),
        (True, 5, set(), "Month boundary +/- 5 days"),
        (True, 3, {4}, "Month boundary +/- 3 days + skip Friday"),
        (True, 3, {0, 4}, "Month boundary +/- 3 days + skip Mon+Fri"),
    ]

    section3b_results = []
    for mb, mbd, skip, label in month_configs:
        r = backtest(month_boundary=mb, month_boundary_days=mbd, skip_weekdays=skip)
        r['desc'] = label
        section3b_results.append(r)
        pr(r, label)

    # ===================== SECTION 4: BEST COMBOS WITH WF VALIDATION =====================
    print("\n" + "=" * 130)
    print("  SECTION 4: BEST COMBOS WITH WALK-FORWARD VALIDATION")
    print("=" * 130)

    # Collect all promising configs from sections 2, 3
    all_configs = []

    # From Section 2: skip-day filters
    for r in section2_results:
        desc = r['desc']
        skip = r['skip']
        all_configs.append({
            'desc': desc,
            'skip_weekdays': skip,
            'monday_gap': False,
            'month_boundary': False,
            'month_boundary_days': 0,
            'ann': r['ann'],
            'mdd': r['mdd'],
        })

    # From Section 3: Monday gap
    for r in section3_results:
        desc = r['desc']
        all_configs.append({
            'desc': desc,
            'skip_weekdays': set(),
            'monday_gap': True,
            'month_boundary': False,
            'month_boundary_days': 0,
            'ann': r['ann'],
            'mdd': r['mdd'],
        })

    # From Section 3: Monday gap + skip combinations
    gap_skip_configs = [
        (True, {4}, "Gap + skip Friday"),
        (True, {0, 4}, "Gap + skip Mon+Fri"),
        (True, {3}, "Gap + skip Thursday"),
        (True, {3, 4}, "Gap + skip Thu+Fri"),
    ]
    for mg, skip, label in gap_skip_configs:
        r = backtest(monday_gap=mg, skip_weekdays=skip)
        all_configs.append({
            'desc': label,
            'skip_weekdays': skip,
            'monday_gap': mg,
            'month_boundary': False,
            'month_boundary_days': 0,
            'ann': r['ann'],
            'mdd': r['mdd'],
        })

    # From Section 3b: month boundary combos
    for r in section3b_results:
        desc = r['desc']
        all_configs.append({
            'desc': desc,
            'skip_weekdays': set(),
            'monday_gap': False,
            'month_boundary': 'Month boundary' in desc,
            'month_boundary_days': 3 if '+/- 3' in desc else (2 if '+/- 2' in desc else 5),
            'ann': r['ann'],
            'mdd': r['mdd'],
        })

    # Also test combined: month boundary + skip friday + monday gap
    combo_configs = [
        (True, {4}, True, 3, "Full combo: gap + skip Fri + month +/-3"),
        (True, {0, 4}, True, 3, "Full combo: gap + skip Mon+Fri + month +/-3"),
        (True, {4}, False, 3, "Combo: skip Fri + month +/-3"),
        (True, {4}, False, 2, "Combo: skip Fri + month +/-2"),
        (True, {0, 4}, False, 3, "Combo: skip Mon+Fri + month +/-3"),
        (True, {3, 4}, True, 3, "Combo: gap + skip Thu+Fri + month +/-3"),
    ]
    for mg, skip, mb, mbd, label in combo_configs:
        r = backtest(monday_gap=mg, skip_weekdays=skip,
                     month_boundary=mb, month_boundary_days=mbd)
        all_configs.append({
            'desc': label,
            'skip_weekdays': skip,
            'monday_gap': mg,
            'month_boundary': mb,
            'month_boundary_days': mbd,
            'ann': r['ann'],
            'mdd': r['mdd'],
        })

    # Rank by return/MDD ratio, take top configs for WF
    all_configs_valid = [c for c in all_configs if c['mdd'] > -80]
    all_configs_valid.sort(key=lambda x: -abs(x['ann'] / x['mdd']) if x['mdd'] != 0 else 0)

    # Select top 15 unique configs for WF
    seen = set()
    wf_configs = []
    for c in all_configs_valid:
        if c['desc'] not in seen:
            seen.add(c['desc'])
            wf_configs.append(c)
        if len(wf_configs) >= 15:
            break

    print(f"\n  Running WF validation for top {len(wf_configs)} configs...")
    wf_all = {}
    for c in wf_configs:
        desc = c['desc']
        wf_res = walk_forward(
            skip_weekdays=c['skip_weekdays'],
            monday_gap=c['monday_gap'],
            month_boundary=c['month_boundary'],
            month_boundary_days=c['month_boundary_days'],
            label=desc
        )
        wf_all[desc] = wf_res
        print_wf(wf_res, desc)

    # ===================== DETAILED WF TABLE =====================
    print("\n" + "=" * 130)
    print("  DETAILED WF TABLE")
    print("=" * 130)

    print(f"\n  {'Config':55s} | {'2020':>12s} | {'2021':>12s} | {'2022':>12s} | {'2023':>12s} | {'2024':>12s} | {'2025':>12s} | {'Avg':>7s} | {'WfMDD':>6s}")
    print(f"  {'-'*55}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*7}-+-{'-'*6}")

    wf_summary = []
    for desc, wf_res in wf_all.items():
        vals = []
        for yr in [2020, 2021, 2022, 2023, 2024, 2025]:
            if yr in wf_res:
                vals.append(f"{wf_res[yr]['ann']:+.0f}/{wf_res[yr]['mdd']:.0f}")
            else:
                vals.append("N/A")
        avg_ann = np.mean([r['ann'] for r in wf_res.values()])
        worst_mdd = min(r['mdd'] for r in wf_res.values())
        print(f"  {desc:55s} | {vals[0]:>12s} | {vals[1]:>12s} | {vals[2]:>12s} | {vals[3]:>12s} | {vals[4]:>12s} | {vals[5]:>12s} | {avg_ann:>+6.0f}% | {worst_mdd:>5.1f}%")
        wf_summary.append((desc, avg_ann, worst_mdd, wf_res))

    # ===================== TOP 3 BY WF AVG WITH MDD > -30% =====================
    print("\n" + "=" * 130)
    print("  TOP 3 CONFIGS BY WF AVG (with WF MDD > -30%)")
    print("=" * 130)

    qualified = [(desc, avg, wmdd, wf) for desc, avg, wmdd, wf in wf_summary if wmdd > -30]
    qualified.sort(key=lambda x: -x[1])

    if qualified:
        for i, (desc, avg, wmdd, wf) in enumerate(qualified[:3]):
            pos = sum(1 for r in wf.values() if r['ann'] > 0)
            ws = " | ".join([f"{yr}:{r['ann']:+.0f}%/{r['mdd']:.0f}%"
                             for yr, r in sorted(wf.items())])
            print(f"\n  #{i+1}: {desc}")
            print(f"       AvgWF={avg:+.0f}% | WorstWfMDD={wmdd:.1f}% | {pos}/6 positive years")
            print(f"       {ws}")
    else:
        print("\n  No configs meet the criteria (WF MDD > -30%). Showing closest:")
        wf_summary.sort(key=lambda x: -x[1])
        for desc, avg, wmdd, wf in wf_summary[:5]:
            print(f"  {desc:55s} | AvgWF={avg:>+7.0f}% | WorstWfMDD={wmdd:>5.1f}%")

    # ===================== ALSO SHOW ALL QUALIFIED =====================
    print("\n" + "=" * 130)
    print("  ALL QUALIFIED CONFIGS (WF avg > 0% and WF MDD > -30%)")
    print("=" * 130)

    all_qualified = [(desc, avg, wmdd, wf) for desc, avg, wmdd, wf in wf_summary if wmdd > -30 and avg > 0]
    all_qualified.sort(key=lambda x: -x[1])
    for desc, avg, wmdd, wf in all_qualified:
        pos = sum(1 for r in wf.values() if r['ann'] > 0)
        print(f"  {desc:55s} | AvgWF={avg:>+7.0f}% | WorstWfMDD={wmdd:>5.1f}% | {pos}/6 pos")

    # ===================== FINAL SUMMARY =====================
    print("\n" + "=" * 130)
    print("  FINAL SUMMARY")
    print("=" * 130)

    print(f"\n  Baseline (no weekday filter): Ann={baseline['ann']:+.1f}% MDD={baseline['mdd']:.1f}%")

    print(f"\n  Weekday Trade Quality (from baseline):")
    wt = baseline.get('weekday_trades', {})
    for wd in ALL_WD:
        pnls = wt.get(wd, [])
        if len(pnls) > 0:
            wr = np.mean([1 if p > 0 else 0 for p in pnls]) * 100
            avg = np.mean(pnls)
            print(f"    {DAY_NAMES[wd]:8s}: N={len(pnls):4d} WR={wr:5.1f}% AvgPnL={avg:+.2f}%")

    print(f"\n  Best skip-day filters (full period):")
    section2_sorted = sorted(section2_results, key=lambda x: abs(x['ann']/x['mdd']) if x['mdd'] != 0 else 0, reverse=True)
    for r in section2_sorted[:5]:
        pr(r, r.get('desc', ''))

    print(f"\n  Elapsed: {time.time()-t_start:.0f}s")
    print("=" * 130)


if __name__ == '__main__':
    main()
