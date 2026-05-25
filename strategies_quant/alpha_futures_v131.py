"""
Alpha Futures V131 — CROSS-COMMODITY MOMENTUM SPILLOVER + MULTI-DAY MOMENTUM
=============================================================================
New alpha sources to reach +600%:

A) Spillover: When one commodity surges (ROC>3%, Z>2), buy its supply-chain
   partners next day. E.g., iron ore↑ → buy rebar, coke, coking coal.
B) Multi-day momentum: ROC(5)>1% for 3+ consecutive days = persistent momentum
C) Intra-group relative strength: Buy the strongest in a group when group is rising
D) V121 on steroids: V121 signal + ROC(3) > 0 + ROC(10) > 0 + ROC(20) > 0
   (aligned momentum across all timeframes)
E) Sector rotation: Allocate to sectors with most V121 signals

ALL signals use NEXT-OPEN execution.
"""
import sys, os, time, warnings
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

# Supply chain groups
GROUPS = {
    'steel': ['rbfi', 'hcfi', 'ifi', 'jfi', 'jmfi'],       # rebar, hrc, iron ore, coke, coking coal
    'base_metal': ['cufi', 'alfi', 'znfi', 'nifi', 'sffi'],  # copper, aluminum, zinc, nickel, tin
    'precious': ['aufi', 'agfi'],                             # gold, silver
    'energy': ['scfi', 'fufi', 'bfi', 'mfi', 'pgfi', 'tafi'], # crude, fuel oil, bitumen, methanol, LPG, PTA
    'oilseed': ['afi', 'mfi', 'yfi', 'pfi', 'cfi'],          # soybean, meal, oil, palm, corn
    'soft': ['srfi', 'cfi', 'whfi', 'rrfi'],                  # sugar, cotton, wheat, early rice
}

def annual_return(final, initial, n_days):
    if final <= 0 or initial <= 0 or n_days <= 0:
        return -100.0
    yrs = n_days / 252
    return (final / initial) ** (1.0 / yrs) * 100 - 100


def main():
    print("=" * 120)
    print("  Alpha Futures V131 — CROSS-COMMODITY MOMENTUM SPILLOVER + MULTI-DAY MOMENTUM")
    print("=" * 120)

    print("\n[Data] Loading...", flush=True)
    t_start = time.time()
    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)
    years = sorted(set(d.year for d in dates))
    print(f"  {NS} commodities, {ND} days")

    # Build symbol index
    sym_idx = {syms[i]: i for i in range(NS)}

    print("\n[Precompute]...", flush=True)
    t0 = time.time()

    RET = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si].astype(np.float64)
        for di in range(1, ND):
            if not np.isnan(c[di]) and not np.isnan(c[di-1]) and c[di-1] > 0:
                RET[si, di] = (c[di] / c[di-1] - 1) * 100

    ROC3 = np.full((NS, ND), np.nan)
    ROC5 = np.full((NS, ND), np.nan)
    ROC10 = np.full((NS, ND), np.nan)
    ROC20 = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si].astype(np.float64)
        ROC3[si] = talib.ROC(c, timeperiod=3)
        ROC5[si] = talib.ROC(c, timeperiod=5)
        ROC10[si] = talib.ROC(c, timeperiod=10)
        ROC20[si] = talib.ROC(c, timeperiod=20)

    ATR14 = np.full((NS, ND), np.nan)
    for si in range(NS):
        ATR14[si] = talib.ATR(H[si].astype(np.float64), L[si].astype(np.float64),
                               C[si].astype(np.float64), timeperiod=14)

    ZSCORE = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(21, ND):
            rets = RET[si, di-20:di]
            valid = rets[~np.isnan(rets)]
            if len(valid) < 10:
                continue
            std_r = np.std(valid, ddof=1)
            if std_r > 0 and not np.isnan(RET[si, di]):
                ZSCORE[si, di] = (RET[si, di] - np.mean(valid)) / std_r

    print(f"  Indicators done ({time.time()-t0:.1f}s)")

    # Group membership for each symbol
    sym_group = {}
    for gname, members in GROUPS.items():
        for m in members:
            sym_group[m] = gname

    # ================================================================
    # BACKTEST ENGINE (same as V121 for fair comparison)
    # ================================================================
    def backtest(signal_func, hold_days=1, top_n=1, start_di=MIN_TRAIN, end_di=None, desc=""):
        if end_di is None:
            end_di = ND
        cash = float(CASH0)
        positions = []
        trades = []
        daily_equity = []
        for di in range(start_di, end_di - 1):
            port_val = cash
            for pos in positions:
                cp = C[pos['si'], di]
                if not np.isnan(cp) and cp > 0:
                    mult = MULT.get(pos['sym'], DEF_MULT)
                    port_val += cp * mult * pos['lots'] - cp * mult * abs(pos['lots']) * COMM
            daily_equity.append(port_val)
            closed = []
            for pos in positions:
                if di - pos['entry_di'] >= pos['hold_days']:
                    exit_price = C[pos['si'], di]
                    if np.isnan(exit_price) or exit_price <= 0:
                        exit_price = pos['entry_price']
                    mult = MULT.get(pos['sym'], DEF_MULT)
                    pnl = (exit_price - pos['entry_price']) * mult * pos['lots']
                    invested = pos['entry_price'] * mult * abs(pos['lots'])
                    pnl_pct = pnl / invested * 100 if invested > 0 else 0
                    cash += exit_price * mult * abs(pos['lots']) * (1 - COMM)
                    trades.append({'pnl_pct': pnl_pct})
                    closed.append(pos)
            for pos in closed:
                positions.remove(pos)
            if len(positions) >= top_n:
                continue
            entry_di = di + 1
            if entry_di >= end_di:
                continue
            candidates = signal_func(di, entry_di)
            if not candidates:
                continue
            candidates.sort(key=lambda x: -x[0])
            n_slots = top_n - len(positions)
            cap_per_slot = cash / max(1, n_slots)
            for sc, s, price in candidates[:max(0, n_slots)]:
                sym = syms[s]
                mult = MULT.get(sym, DEF_MULT)
                contracts = max(1, int(cap_per_slot * 0.95 / (price * mult * (1 + COMM))))
                cost_in = price * mult * contracts * (1 + COMM)
                if cost_in > cash:
                    contracts = int(cash * 0.9 / (price * mult * (1 + COMM)))
                    cost_in = price * mult * contracts * (1 + COMM) if contracts > 0 else 0
                if contracts <= 0 or cost_in <= 0 or cost_in > cash:
                    continue
                cash -= cost_in
                positions.append({
                    'si': s, 'entry_price': price, 'entry_di': entry_di,
                    'lots': contracts, 'dir': 1, 'sym': sym, 'hold_days': hold_days,
                })
        for pos in positions:
            ae = end_di - 1
            exit_price = C[pos['si'], min(ae, ND-1)]
            if np.isnan(exit_price) or exit_price <= 0:
                exit_price = pos['entry_price']
            mult = MULT.get(pos['sym'], DEF_MULT)
            cash += exit_price * mult * abs(pos['lots']) * (1 - COMM)
        n_days_test = end_di - start_di
        ann = annual_return(cash, CASH0, n_days_test)
        wr = np.mean([1 if t['pnl_pct'] > 0 else 0 for t in trades]) * 100 if trades else 0
        n_trades = len(trades)
        avg_pnl = np.mean([t['pnl_pct'] for t in trades]) if trades else 0
        if daily_equity:
            eq = np.array(daily_equity)
            pk = np.maximum.accumulate(eq)
            mdd = np.min((eq - pk) / pk * 100)
            rets = np.diff(eq) / eq[:-1]
            sharpe = np.mean(rets) / np.std(rets) * np.sqrt(252) if np.std(rets) > 0 else 0
        else:
            mdd = 0; sharpe = 0
        return {'ann': ann, 'wr': wr, 'n': n_trades, 'avg_pnl': avg_pnl,
                'mdd': mdd, 'sharpe': sharpe, 'desc': desc}

    def pr(r, label=""):
        print(f"  {label:55s} | Ann={r['ann']:+8.1f}% | WR={r['wr']:5.1f}% | "
              f"N={r['n']:4d} | Avg={r['avg_pnl']:+5.2f}% | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f}")

    def wf(signal_func, hold=1, topn=1, desc=""):
        res = {}
        for yr in [2020, 2021, 2022, 2023, 2024, 2025]:
            ys = ye = None
            for di in range(ND):
                if dates[di].year == yr and ys is None: ys = di
                if dates[di].year == yr: ye = di + 1
            if ys is None: continue
            r = backtest(signal_func, hold_days=hold, top_n=topn, start_di=ys, end_di=ye)
            res[yr] = r['ann']
        return res

    # ================================================================
    # SIGNAL DEFINITIONS
    # ================================================================

    # BASELINE: V121
    def signal_v121(di, edi):
        cands = []
        for s in range(NS):
            roc = ROC5[s, di]; zs = ZSCORE[s, di]
            if np.isnan(roc) or np.isnan(zs) or roc <= 1.0 or zs <= 1.5:
                continue
            rp = ROC5[s, di-1] if di > 0 else np.nan
            if not np.isnan(rp) and roc <= rp:
                continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0:
                continue
            cands.append((roc * zs, s, ep))
        return cands

    # A) Multi-timeframe aligned momentum: ROC3>0 AND ROC5>0 AND ROC10>0 AND ROC20>0
    #    + ROC5 > 1% + Z > 1.5 (V121 base) + all timeframes aligned
    def signal_aligned(di, edi):
        cands = []
        for s in range(NS):
            roc3 = ROC3[s, di]; roc5 = ROC5[s, di]; roc10 = ROC10[s, di]; roc20 = ROC20[s, di]
            zs = ZSCORE[s, di]
            if any(np.isnan(x) for x in [roc3, roc5, roc10, roc20, zs]):
                continue
            if roc3 <= 0 or roc5 <= 0 or roc10 <= 0 or roc20 <= 0:
                continue
            if roc5 <= 1.0 or zs <= 1.5:
                continue
            rp = ROC5[s, di-1] if di > 0 else np.nan
            if not np.isnan(rp) and roc5 <= rp:
                continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0:
                continue
            # Score: sum of all ROCs * Z (strongest multi-timeframe alignment)
            score = (roc3 + roc5 + roc10 + roc20) * zs
            cands.append((score, s, ep))
        return cands

    # B) Persistent momentum: ROC(5)>1% for 3+ consecutive days
    def signal_persistent(di, edi):
        cands = []
        for s in range(NS):
            zs = ZSCORE[s, di]
            if np.isnan(zs) or zs <= 1.0:
                continue
            # Check if ROC5 > 1% for last 3 days
            if di < 2:
                continue
            persistent = True
            for k in range(3):
                r = ROC5[s, di-k]
                if np.isnan(r) or r <= 1.0:
                    persistent = False
                    break
            if not persistent:
                continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0:
                continue
            score = ROC5[s, di] * zs * 3  # 3x for 3-day persistence
            cands.append((score, s, ep))
        return cands

    # C) Cross-commodity spillover: when a group leader surges, buy laggards
    def signal_spillover(di, edi):
        cands = []
        for gname, members in GROUPS.items():
            # Find group members that are surging (ROC5 > 3%, Z > 2)
            surging = []
            for m in members:
                si = sym_idx.get(m)
                if si is None:
                    continue
                roc = ROC5[si, di]
                zs = ZSCORE[si, di]
                if not np.isnan(roc) and not np.isnan(zs) and roc > 3.0 and zs > 2.0:
                    surging.append((si, roc, zs))

            if len(surging) < 1:
                continue

            # Buy other group members that have positive but moderate momentum
            for m in members:
                si = sym_idx.get(m)
                if si is None:
                    continue
                if any(s[0] == si for s in surging):
                    continue  # skip the surger itself
                roc = ROC5[si, di]
                zs = ZSCORE[si, di]
                if np.isnan(roc) or np.isnan(zs):
                    continue
                if roc <= 0.5 or roc > 5.0:  # moderate momentum, not already surging
                    continue
                if zs <= 0.5:
                    continue
                ep = O[si, edi]
                if np.isnan(ep) or ep <= 0:
                    continue
                # Score by surger's strength * target's momentum
                surger_avg_roc = np.mean([s[1] for s in surging])
                score = surger_avg_roc * roc * zs
                cands.append((score, si, ep))
        return cands

    # D) Sector rotation: find sector with most V121 signals, buy top 2
    def signal_sector(di, edi):
        # Count V121 signals per group
        group_sigs = {}
        for gname, members in GROUPS.items():
            group_sigs[gname] = []
            for m in members:
                si = sym_idx.get(m)
                if si is None:
                    continue
                roc = ROC5[si, di]; zs = ZSCORE[si, di]
                if np.isnan(roc) or np.isnan(zs) or roc <= 1.0 or zs <= 1.5:
                    continue
                rp = ROC5[si, di-1] if di > 0 else np.nan
                if not np.isnan(rp) and roc <= rp:
                    continue
                group_sigs[gname].append((roc * zs, si))

        # Find best group
        if not group_sigs:
            return []
        best_group = max(group_sigs, key=lambda g: len(group_sigs[g]))
        if not group_sigs[best_group]:
            return []

        cands = []
        for score, si in group_sigs[best_group]:
            ep = O[si, edi]
            if np.isnan(ep) or ep <= 0:
                continue
            cands.append((score, si, ep))
        return cands

    # E) Combined: V121 + aligned momentum bonus
    def signal_v121_aligned(di, edi):
        cands = []
        for s in range(NS):
            roc5 = ROC5[s, di]; zs = ZSCORE[s, di]
            if np.isnan(roc5) or np.isnan(zs) or roc5 <= 1.0 or zs <= 1.5:
                continue
            rp = ROC5[s, di-1] if di > 0 else np.nan
            if not np.isnan(rp) and roc5 <= rp:
                continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0:
                continue
            # Alignment bonus
            bonus = 1.0
            roc3 = ROC3[s, di]; roc10 = ROC10[s, di]; roc20 = ROC20[s, di]
            if not np.isnan(roc3) and roc3 > 0:
                bonus += 0.2
            if not np.isnan(roc10) and roc10 > 0:
                bonus += 0.3
            if not np.isnan(roc20) and roc20 > 0:
                bonus += 0.5
            cands.append((roc5 * zs * bonus, s, ep))
        return cands

    # F) Spillover + V121 combined
    def signal_spill_v121(di, edi):
        v121 = signal_v121(di, edi)
        if v121:
            return v121
        # If no V121 signal, try spillover
        spill = signal_spillover(di, edi)
        if spill:
            return spill
        # Try persistent
        pers = signal_persistent(di, edi)
        if pers:
            return pers
        return []

    # ================================================================
    # RUN ALL
    # ================================================================
    print("\n" + "=" * 120)
    print("  SECTION 1: ALL NEW SIGNALS HEAD-TO-HEAD")
    print("=" * 120)

    strategies = [
        ("V121 Champion (baseline)", signal_v121),
        ("A) Aligned multi-timeframe", signal_aligned),
        ("B) Persistent momentum 3d", signal_persistent),
        ("C) Cross-commodity spillover", signal_spillover),
        ("D) Sector rotation", signal_sector),
        ("E) V121 + alignment bonus", signal_v121_aligned),
        ("F) Spill+V121+Persist fallback", signal_spill_v121),
    ]

    results = {}
    for name, func in strategies:
        r = backtest(func, hold_days=1, top_n=1, desc=name)
        results[name] = r
        pr(r, label=name)

    # ================================================================
    # SECTION 2: TOP_N SWEEP FOR BEST
    # ================================================================
    print("\n" + "=" * 120)
    print("  SECTION 2: TOP_N x HOLD for best strategies")
    print("=" * 120)

    best3 = sorted(results.items(), key=lambda x: -x[1]['ann'])[:3]
    for name, r in best3:
        func = dict(strategies)[name]
        print(f"\n  {name}:")
        for topn in [1, 2, 3, 5]:
            for hold in [1, 2, 3]:
                r = backtest(func, hold_days=hold, top_n=topn, desc=f"{name} t={topn} h={hold}")
                print(f"    top_n={topn} hold={hold}: Ann={r['ann']:+8.1f}% | "
                      f"WR={r['wr']:5.1f}% | N={r['n']:4d} | MDD={r['mdd']:6.1f}%")

    # ================================================================
    # SECTION 3: WALK-FORWARD
    # ================================================================
    print("\n" + "=" * 120)
    print("  SECTION 3: WALK-FORWARD")
    print("=" * 120)

    for name, func in strategies:
        w = wf(func, desc=name)
        ws = " | ".join([f"{yr}:{v:+.0f}%" for yr, v in sorted(w.items())])
        pos = sum(1 for v in w.values() if v > 0)
        avg = np.mean(list(w.values())) if w else 0
        print(f"  {name:55s} | {pos}/6 | Avg={avg:>+7.0f}% | {ws}")

    # ================================================================
    # SECTION 4: ALIGNED MOMENTUM PARAMETER SWEEP
    # ================================================================
    print("\n" + "=" * 120)
    print("  SECTION 4: ALIGNED MOMENTUM SENSITIVITY")
    print("=" * 120)

    # What if we require fewer timeframes aligned?
    for n_tf in [1, 2, 3, 4]:  # number of positive timeframes needed
        def make_aligned(ntf):
            def sig(di, edi):
                cands = []
                for s in range(NS):
                    roc5 = ROC5[s, di]; zs = ZSCORE[s, di]
                    if np.isnan(roc5) or np.isnan(zs) or roc5 <= 1.0 or zs <= 1.5:
                        continue
                    rp = ROC5[s, di-1] if di > 0 else np.nan
                    if not np.isnan(rp) and roc5 <= rp:
                        continue
                    ep = O[s, edi]
                    if np.isnan(ep) or ep <= 0:
                        continue
                    # Count aligned timeframes
                    aligned = 0
                    for roc_arr in [ROC3, ROC5, ROC10, ROC20]:
                        v = roc_arr[s, di]
                        if not np.isnan(v) and v > 0:
                            aligned += 1
                    if aligned < ntf:
                        continue
                    score = roc5 * zs * aligned
                    cands.append((score, s, ep))
                return cands
            return sig
        r = backtest(make_aligned(n_tf), desc=f"{n_tf} TF aligned")
        pr(r, label=f"{n_tf} TFs aligned")

    # ================================================================
    # SECTION 5: PERSISTENT MOMENTUM SENSITIVITY
    # ================================================================
    print("\n" + "=" * 120)
    print("  SECTION 5: PERSISTENT MOMENTUM SENSITIVITY")
    print("=" * 120)

    for persist_days in [2, 3, 4, 5]:
        for roc_thresh in [0.5, 1.0, 1.5]:
            def make_persist(pd, rt):
                def sig(di, edi):
                    cands = []
                    for s in range(NS):
                        zs = ZSCORE[s, di]
                        if np.isnan(zs) or zs <= 1.0:
                            continue
                        if di < pd - 1:
                            continue
                        ok = True
                        for k in range(pd):
                            r = ROC5[s, di-k]
                            if np.isnan(r) or r <= rt:
                                ok = False
                                break
                        if not ok:
                            continue
                        ep = O[s, edi]
                        if np.isnan(ep) or ep <= 0:
                            continue
                        cands.append((ROC5[s, di] * zs, s, ep))
                    return cands
                return sig
            r = backtest(make_persist(persist_days, roc_thresh),
                         desc=f"persist={persist_days}d ROC>{roc_thresh}")
            pr(r, label=f"persist={persist_days}d ROC>{roc_thresh}")

    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "=" * 120)
    print("  SUMMARY")
    print("=" * 120)

    sorted_res = sorted(results.items(), key=lambda x: -x[1]['ann'])
    for i, (name, r) in enumerate(sorted_res):
        print(f"  #{i+1}: {name:55s} | Ann={r['ann']:+8.1f}% | WR={r['wr']:5.1f}% | "
              f"N={r['n']:4d} | MDD={r['mdd']:6.1f}%")

    print(f"\n  Total elapsed: {time.time()-t_start:.0f}s")
    print("=" * 120)


if __name__ == '__main__':
    main()
