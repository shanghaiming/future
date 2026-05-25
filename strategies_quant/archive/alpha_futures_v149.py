"""
Alpha Futures V149 — Trailing Stop / Trend Continuation
=============================================================================
Goal: Explore whether holding winners longer improves returns over V146's
      fixed 1-day hold. Test trailing stops, trend continuation holds,
      and partial profit taking.

Base from V146 fixed: top_n=2, DD70/60/40/20 sizing, corr<0.5, balanced method.
Sections:
  0: Baseline (hold=1, no trailing stop)
  1: Fixed hold period sweep (hold=2,3,5)
  2: Trailing stop (ATR-based, various multipliers)
  3: Trend continuation hold (ROC5-based)
  4: Combined best
  5: WF validation for top 10 configs
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

def annual_return(final, initial, n_days):
    if final <= 0 or initial <= 0 or n_days <= 0: return -100.0
    return (final / initial) ** (1.0 / (n_days / 252)) * 100 - 100


def main():
    print("=" * 130)
    print("  V149 — Trailing Stop / Trend Continuation")
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

    RET20 = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si].astype(np.float64)
        for di in range(20, ND):
            if not np.isnan(c[di]) and not np.isnan(c[di-20]) and c[di-20] > 0:
                RET20[si, di] = (c[di] / c[di-20] - 1) * 100

    # ===================== REGIME INDICATORS =====================
    print("  Computing regime indicators...", flush=True)

    BREADTH = np.full(ND, np.nan)
    for di in range(5, ND):
        pos_count = 0; total = 0
        for si in range(NS):
            r = ROC5[si, di]
            if not np.isnan(r):
                total += 1
                if r > 0: pos_count += 1
        if total > 0:
            BREADTH[di] = pos_count / total

    MKT_RET = np.full(ND, np.nan)
    for di in range(ND):
        rets_day = RET[:, di]
        valid = rets_day[~np.isnan(rets_day)]
        if len(valid) > 10:
            MKT_RET[di] = np.mean(valid)

    MKT_VOL = np.full(ND, np.nan)
    for di in range(20, ND):
        window = MKT_RET[di-20:di]
        valid = window[~np.isnan(window)]
        if len(valid) >= 10:
            MKT_VOL[di] = np.std(valid, ddof=1)

    valid_vols = MKT_VOL[~np.isnan(MKT_VOL)]
    VOL_MEDIAN = np.median(valid_vols) if len(valid_vols) > 0 else 1.0
    print(f"  Market vol median: {VOL_MEDIAN:.4f}%")

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

    # ===================== HELPER: Correlation =====================
    def get_corr(si_a, si_b, di, window=20):
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

    # ===================== UNIFIED BACKTEST ENGINE WITH TRAILING STOP =====================
    def backtest_v149(start_di=MIN_TRAIN, end_di=None,
                      # Core params
                      max_corr=0.5, top_n=2, hold=1,
                      # DD sizing tiers
                      dd_tiers=None,
                      # Stop-loss (intraday, fraction of invested)
                      sl_pct=0.0,
                      # --- NEW: Trailing stop ---
                      trailing_stop=False,       # enable ATR trailing stop
                      trail_atr_mult=1.5,        # ATR * multiplier for trailing distance
                      # --- NEW: Trend continuation ---
                      trend_cont=False,           # enable trend continuation hold
                      trend_cont_threshold=1.0,   # ROC5 > this to extend hold
                      trend_cont_max=5,           # max extra days to extend
                      # --- NEW: Partial profit ---
                      partial_profit=False,       # enable partial profit taking
                      partial_day=2,              # after N days of profit
                      partial_pct=0.5,            # fraction to close
                      ):
        if end_di is None: end_di = ND
        if dd_tiers is None:
            dd_tiers = [(0, 0.70), (0.10, 0.60), (0.20, 0.40), (0.30, 0.20)]

        cash = float(CASH0)
        positions = []
        trades = []
        daily_eq = []
        high_water = float(CASH0)

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

            # --- Stop-loss check (intraday) ---
            if sl_pct > 0:
                cl_early = []
                for p in positions:
                    cp = C[p['si'], di]
                    if np.isnan(cp) or cp <= 0: continue
                    m = MULT.get(p['sym'], DEF_MULT)
                    unrealized = (cp - p['entry_price']) * m * p['lots']
                    invested = p['entry_price'] * m * abs(p['lots'])
                    if invested > 0:
                        loss_pct = unrealized / invested
                        if loss_pct < -sl_pct:
                            cash += cp * m * abs(p['lots']) * (1 - COMM)
                            pnl_pct = unrealized / invested * 100
                            trades.append(pnl_pct)
                            cl_early.append(p)
                for p in cl_early: positions.remove(p)

            # --- Trailing stop check ---
            if trailing_stop:
                cl_trail = []
                for p in positions:
                    cp = C[p['si'], di]
                    atr = ATR14[p['si'], di]
                    if np.isnan(cp) or np.isnan(atr) or atr <= 0:
                        continue
                    # Only trail if position has been held at least 1 day after entry
                    days_held = di - p['entry_di']
                    if days_held < 1:
                        continue
                    m = MULT.get(p['sym'], DEF_MULT)
                    invested = p['entry_price'] * m * abs(p['lots'])

                    # Only activate trailing stop when position is profitable
                    unrealized = (cp - p['entry_price']) * m * p['lots']
                    if unrealized <= 0:
                        continue

                    # Track highest close since entry for trailing anchor
                    if 'trail_highest' not in p or cp > p['trail_highest']:
                        p['trail_highest'] = cp

                    # Trailing level = highest close - ATR * mult
                    trail_level = p['trail_highest'] - atr * trail_atr_mult

                    # Check if close breached trailing level
                    if cp <= trail_level:
                        pnl = (cp - p['entry_price']) * m * p['lots']
                        pnl_pct = pnl / invested * 100 if invested > 0 else 0
                        cash += cp * m * abs(p['lots']) * (1 - COMM)
                        trades.append(pnl_pct)
                        cl_trail.append(p)
                for p in cl_trail: positions.remove(p)

            # --- Partial profit taking ---
            if partial_profit:
                for p in list(positions):
                    if p.get('partial_taken'): continue
                    days_held = di - p['entry_di']
                    if days_held >= partial_day:
                        cp = C[p['si'], di]
                        if np.isnan(cp) or cp <= 0: continue
                        m = MULT.get(p['sym'], DEF_MULT)
                        pnl = (cp - p['entry_price']) * m * p['lots']
                        invested = p['entry_price'] * m * abs(p['lots'])
                        if invested > 0 and pnl > 0:
                            # Take partial profit: close partial_pct of lots
                            lots_to_close = max(1, int(p['lots'] * partial_pct))
                            if lots_to_close >= p['lots']:
                                lots_to_close = p['lots'] - 1  # keep at least 1
                            if lots_to_close > 0:
                                cash += cp * m * lots_to_close * (1 - COMM)
                                pnl_partial = (cp - p['entry_price']) * m * lots_to_close
                                pnl_pct = pnl_partial / invested * 100
                                trades.append(pnl_pct)
                                p['lots'] -= lots_to_close
                                p['partial_taken'] = True

            # --- Trend continuation: extend hold if trend remains strong ---
            if trend_cont:
                for p in positions:
                    if p.get('trend_extended_max'): continue
                    roc = ROC5[p['si'], di]
                    days_held = di - p['entry_di']
                    # Only check on the day before expiry
                    if days_held == p['hold_days'] - 1:
                        if not np.isnan(roc) and roc > trend_cont_threshold:
                            extra = p.get('trend_extra', 0)
                            if extra < trend_cont_max:
                                p['hold_days'] += 1
                                p['trend_extra'] = extra + 1
                            else:
                                p['trend_extended_max'] = True

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
                    cl.append(p)
            for p in cl: positions.remove(p)

            # --- Compute position size (DD-based) ---
            pos_size = dd_size(pv, high_water, dd_tiers)
            pos_size = max(0.05, min(0.95, pos_size))

            # --- Enter positions (cross_corr mode from V146) ---
            if len(positions) >= top_n: continue
            edi = di + 1
            if edi >= end_di: continue

            held_si = set(p['si'] for p in positions)

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

            cash_snapshot = cash  # BUG PREVENTION: snapshot before entry loop
            n_planned = len(entries)
            for sc, s, pr, sig_str, pct in entries:
                if s in set(p['si'] for p in positions): continue
                if len(positions) >= top_n: break
                cap = cash_snapshot * pct / n_planned
                sym = syms[s]; m = MULT.get(sym, DEF_MULT)
                ct = max(1, int(cap / (pr * m * (1 + COMM))))
                ci = pr * m * ct * (1 + COMM)
                if ci > cash:
                    ct = int(cash * 0.9 / (pr * m * (1 + COMM)))
                    ci = pr * m * ct * (1 + COMM) if ct > 0 else 0
                if ct <= 0 or ci <= 0 or ci > cash: continue
                cash -= ci
                positions.append({'si': s, 'entry_price': pr, 'entry_di': edi,
                                  'lots': ct, 'dir': 1, 'sym': sym, 'hold_days': hold,
                                  'sig': sig_str, 'score': sc})

        # Close remaining
        for p in positions:
            ae = end_di - 1
            ep = C[p['si'], min(ae, ND-1)]
            if np.isnan(ep) or ep <= 0: ep = p['entry_price']
            m = MULT.get(p['sym'], DEF_MULT)
            cash += ep * m * abs(p['lots']) * (1 - COMM)

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
        return {'ann': ann, 'wr': wr, 'n': nt, 'mdd': mdd, 'sharpe': sh, 'final': cash}

    # ===================== PRINTING HELPERS =====================
    def pr(r, label=""):
        ratio = abs(r['ann'] / r['mdd']) if r['mdd'] != 0 else 0
        print(f"  {label:80s} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f} | R/M={ratio:.2f} | N={r['n']:4d}")

    def walk_forward(label="", **kwargs):
        res = {}
        for yr in [2020, 2021, 2022, 2023, 2024, 2025]:
            ys = ye = None
            for di in range(ND):
                if dates[di].year == yr and ys is None: ys = di
                if dates[di].year == yr: ye = di + 1
            if ys is None: continue
            r = backtest_v149(start_di=ys, end_di=ye, **kwargs)
            res[yr] = r
        return res

    def print_wf(wf_res, label=""):
        pos = sum(1 for r in wf_res.values() if r['ann'] > 0)
        avg_ann = np.mean([r['ann'] for r in wf_res.values()])
        worst_mdd = min(r['mdd'] for r in wf_res.values())
        ws = " | ".join([f"{yr}:{r['ann']:+.0f}%/{r['mdd']:.0f}%"
                         for yr, r in sorted(wf_res.items())])
        print(f"    {label:80s}")
        print(f"      {pos}/6 pos | Avg={avg_ann:>+7.0f}% | WorstWfMDD={worst_mdd:>5.0f}%")
        print(f"      {ws}")

    # Default DD tiers
    DD_DEFAULT = [(0, 0.70), (0.10, 0.60), (0.20, 0.40), (0.30, 0.20)]

    # ===================== SECTION 0: BASELINE =====================
    print("\n" + "=" * 130)
    print("  SECTION 0: BASELINE (hold=1, no trailing stop, DD sizing)")
    print("=" * 130)

    r_base = backtest_v149(hold=1, sl_pct=0.03, dd_tiers=DD_DEFAULT)
    r_base['desc'] = "S0: Baseline hold=1 SL=3% DD70/60/40/20"
    pr(r_base, r_base['desc'])

    r_base_nosl = backtest_v149(hold=1, sl_pct=0.0, dd_tiers=DD_DEFAULT)
    r_base_nosl['desc'] = "S0: Baseline hold=1 NO SL DD70/60/40/20"
    pr(r_base_nosl, r_base_nosl['desc'])

    # ===================== SECTION 1: FIXED HOLD PERIOD SWEEP =====================
    print("\n" + "=" * 130)
    print("  SECTION 1: FIXED HOLD PERIOD SWEEP (hold=2,3,5)")
    print("=" * 130)

    s1_results = []
    for h in [2, 3, 5]:
        for sl in [0.0, 0.03, 0.05]:
            r = backtest_v149(hold=h, sl_pct=sl, dd_tiers=DD_DEFAULT)
            desc = f"S1: hold={h} SL={sl*100:.0f}% DD70/60/40/20"
            r['desc'] = desc
            s1_results.append(r)
            pr(r, desc)

    # ===================== SECTION 2: TRAILING STOP (ATR-based) =====================
    print("\n" + "=" * 130)
    print("  SECTION 2: TRAILING STOP (ATR-based)")
    print("  Trailing stop level = current_close - ATR * mult")
    print("  Only triggers when position is profitable")
    print("=" * 130)

    s2_results = []
    for mult in [1.0, 1.5, 2.0, 2.5]:
        for h in [2, 3, 5]:
            for sl in [0.0, 0.03]:
                r = backtest_v149(hold=h, sl_pct=sl, dd_tiers=DD_DEFAULT,
                                  trailing_stop=True, trail_atr_mult=mult)
                desc = f"S2: trail ATR*{mult:.1f} hold={h} SL={sl*100:.0f}%"
                r['desc'] = desc
                r['trail_mult'] = mult
                r['hold'] = h
                r['sl_pct'] = sl
                s2_results.append(r)
                pr(r, desc)

    # ===================== SECTION 3: TREND CONTINUATION HOLD =====================
    print("\n" + "=" * 130)
    print("  SECTION 3: TREND CONTINUATION HOLD")
    print("  If ROC5 > threshold at base exit day, extend hold")
    print("=" * 130)

    s3_results = []
    for threshold in [0.5, 1.0, 2.0]:
        for max_ext in [2, 3, 5]:
            for sl in [0.0, 0.03]:
                r = backtest_v149(hold=1, sl_pct=sl, dd_tiers=DD_DEFAULT,
                                  trend_cont=True,
                                  trend_cont_threshold=threshold,
                                  trend_cont_max=max_ext)
                desc = f"S3: trend ROC5>{threshold:.1f}% max_ext={max_ext} SL={sl*100:.0f}%"
                r['desc'] = desc
                r['trend_thresh'] = threshold
                r['trend_max'] = max_ext
                r['sl_pct'] = sl
                s3_results.append(r)
                pr(r, desc)

    # Also test trend continuation with base hold=2
    for threshold in [1.0, 2.0]:
        for max_ext in [2, 3]:
            for sl in [0.0, 0.03]:
                r = backtest_v149(hold=2, sl_pct=sl, dd_tiers=DD_DEFAULT,
                                  trend_cont=True,
                                  trend_cont_threshold=threshold,
                                  trend_cont_max=max_ext)
                desc = f"S3: trend ROC5>{threshold:.1f}% max_ext={max_ext} hold_base=2 SL={sl*100:.0f}%"
                r['desc'] = desc
                r['trend_thresh'] = threshold
                r['trend_max'] = max_ext
                r['sl_pct'] = sl
                s3_results.append(r)
                pr(r, desc)

    # ===================== SECTION 4: COMBINED BEST =====================
    print("\n" + "=" * 130)
    print("  SECTION 4: COMBINED BEST")
    print("  Trailing stop + trend continuation + fixed hold variants")
    print("=" * 130)

    s4_results = []
    # Combine trailing stop + trend continuation
    for mult in [1.5, 2.0]:
        for threshold in [1.0, 2.0]:
            for max_ext in [2, 3]:
                for sl in [0.0, 0.03]:
                    r = backtest_v149(hold=1, sl_pct=sl, dd_tiers=DD_DEFAULT,
                                      trailing_stop=True, trail_atr_mult=mult,
                                      trend_cont=True,
                                      trend_cont_threshold=threshold,
                                      trend_cont_max=max_ext)
                    desc = f"S4: trail*{mult:.1f}+trend ROC5>{threshold:.1f}% ext={max_ext} SL={sl*100:.0f}%"
                    r['desc'] = desc
                    r['trail_mult'] = mult
                    r['trend_thresh'] = threshold
                    r['trend_max'] = max_ext
                    r['sl_pct'] = sl
                    s4_results.append(r)
                    pr(r, desc)

    # Combine hold=2 or 3 with trailing stop
    for h in [2, 3]:
        for mult in [1.5, 2.0]:
            for sl in [0.0, 0.03]:
                r = backtest_v149(hold=h, sl_pct=sl, dd_tiers=DD_DEFAULT,
                                  trailing_stop=True, trail_atr_mult=mult)
                desc = f"S4: hold={h} trail*{mult:.1f} SL={sl*100:.0f}%"
                r['desc'] = desc
                r['trail_mult'] = mult
                r['hold'] = h
                r['sl_pct'] = sl
                s4_results.append(r)
                pr(r, desc)

    # Partial profit taking variants
    print("\n  --- Partial profit taking ---")
    for pday in [2, 3]:
        for ppct in [0.3, 0.5]:
            for sl in [0.0, 0.03]:
                r = backtest_v149(hold=3, sl_pct=sl, dd_tiers=DD_DEFAULT,
                                  partial_profit=True,
                                  partial_day=pday, partial_pct=ppct)
                desc = f"S4: partial day={pday} pct={ppct*100:.0f}% hold=3 SL={sl*100:.0f}%"
                r['desc'] = desc
                r['sl_pct'] = sl
                s4_results.append(r)
                pr(r, desc)

    # ===================== SECTION 5: WF VALIDATION FOR TOP CONFIGS =====================
    print("\n" + "=" * 130)
    print("  SECTION 5: WF VALIDATION FOR TOP 10 CONFIGS")
    print("=" * 130)

    # Collect all results
    all_results = [r_base, r_base_nosl] + s1_results + s2_results + s3_results + s4_results
    all_valid = [r for r in all_results if r.get('desc', '') and r['mdd'] > -80]

    # Rank by annual return
    all_valid.sort(key=lambda x: -x['ann'])
    print(f"\n  Top 15 by Annual Return (full period):")
    for i, r in enumerate(all_valid[:15]):
        desc = r.get('desc', '')
        ratio = abs(r['ann'] / r['mdd']) if r['mdd'] != 0 else 0
        print(f"  #{i+1:2d}: {desc:80s} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f} | R/M={ratio:.2f}")

    # Rank by R/M ratio
    all_with_ratio = [(r, abs(r['ann'] / r['mdd']) if r['mdd'] != 0 else 0) for r in all_valid]
    all_with_ratio.sort(key=lambda x: -x[1])
    print(f"\n  Top 15 by Ann/MDD Ratio:")
    for i, (r, ratio) in enumerate(all_with_ratio[:15]):
        desc = r.get('desc', '')
        print(f"  #{i+1:2d}: {desc:80s} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f} | R/M={ratio:.2f}")

    # Pick top 10 unique configs for WF
    seen = set()
    wf_configs = []
    # First pick by R/M ratio
    for r, ratio in all_with_ratio:
        desc = r.get('desc', '')
        if desc not in seen:
            seen.add(desc)
            wf_configs.append(r)
        if len(wf_configs) >= 5:
            break
    # Then fill from top annual return
    for r in all_valid:
        desc = r.get('desc', '')
        if desc not in seen:
            seen.add(desc)
            wf_configs.append(r)
        if len(wf_configs) >= 10:
            break

    # Reconstruct params for each config and run WF
    wf_all = {}
    for r in wf_configs:
        desc = r.get('desc', '')
        # Parse the desc to reconstruct params
        kwargs = {'dd_tiers': DD_DEFAULT, 'max_corr': 0.5, 'top_n': 2}

        # Parse hold
        if 'hold=1' in desc:
            kwargs['hold'] = 1
        elif 'hold=2' in desc:
            kwargs['hold'] = 2
        elif 'hold=3' in desc:
            kwargs['hold'] = 3
        elif 'hold=5' in desc:
            kwargs['hold'] = 5
        else:
            kwargs['hold'] = 1

        # Parse SL
        if 'SL=3%' in desc:
            kwargs['sl_pct'] = 0.03
        elif 'SL=5%' in desc:
            kwargs['sl_pct'] = 0.05
        elif 'NO SL' in desc:
            kwargs['sl_pct'] = 0.0
        else:
            kwargs['sl_pct'] = 0.0

        # Parse trailing stop
        if 'trail' in desc:
            kwargs['trailing_stop'] = True
            # Parse multiplier
            for token in desc.split():
                if token.startswith('trail*'):
                    try:
                        kwargs['trail_atr_mult'] = float(token.split('*')[1])
                    except: pass
        else:
            kwargs['trailing_stop'] = False

        # Parse trend continuation
        if 'trend' in desc:
            kwargs['trend_cont'] = True
            for token in desc.split():
                if token.startswith('ROC5>'):
                    try:
                        kwargs['trend_cont_threshold'] = float(token.split('>')[1].rstrip('%'))
                    except: pass
                if token.startswith('ext='):
                    try:
                        kwargs['trend_cont_max'] = int(token.split('=')[1])
                    except: pass
        else:
            kwargs['trend_cont'] = False

        # Parse partial profit
        if 'partial' in desc:
            kwargs['partial_profit'] = True
            tokens = desc.split()
            for i, token in enumerate(tokens):
                if token == 'day=' and i+1 < len(tokens):
                    try: kwargs['partial_day'] = int(tokens[i+1])
                    except: pass
                if token == 'pct=' and i+1 < len(tokens):
                    try: kwargs['partial_pct'] = float(tokens[i+1].rstrip('%')) / 100
                    except: pass
        else:
            kwargs['partial_profit'] = False

        wf_res = walk_forward(label=desc, **kwargs)
        wf_all[desc] = wf_res
        print_wf(wf_res, desc)

    # ===================== HIGHLIGHT: BEST WF CONFIGS =====================
    print("\n" + "=" * 130)
    print("  HIGHLIGHT: Configs with WF Avg > +150% and WF Worst MDD > -30%")
    print("=" * 130)

    highlight = []
    for desc, wf_res in wf_all.items():
        avg_ann = np.mean([r['ann'] for r in wf_res.values()])
        worst_mdd = min(r['mdd'] for r in wf_res.values())
        pos_years = sum(1 for r in wf_res.values() if r['ann'] > 0)
        highlight.append((desc, avg_ann, worst_mdd, pos_years, wf_res))

    # Filter for target
    target_configs = [(d, a, m, p, w) for d, a, m, p, w in highlight if a > 0 and m > -30]
    target_configs.sort(key=lambda x: -x[1])

    if target_configs:
        for desc, avg_ann, worst_mdd, pos_years, wf_res in target_configs[:10]:
            ws = " | ".join([f"{yr}:{r['ann']:+.0f}%/{r['mdd']:.0f}%"
                             for yr, r in sorted(wf_res.items())])
            print(f"\n  *** {desc}")
            print(f"      AvgWF={avg_ann:+.0f}% | WorstWfMDD={worst_mdd:.1f}% | {pos_years}/6 positive")
            print(f"      {ws}")
    else:
        print("\n  No configs meet target. Showing all by avg WF return:")
        highlight.sort(key=lambda x: -x[1])
        for desc, avg_ann, worst_mdd, pos_years, wf_res in highlight[:10]:
            ws = " | ".join([f"{yr}:{r['ann']:+.0f}%/{r['mdd']:.0f}%"
                             for yr, r in sorted(wf_res.items())])
            print(f"\n  {desc}")
            print(f"      AvgWF={avg_ann:+.0f}% | WorstWfMDD={worst_mdd:.1f}% | {pos_years}/6 positive")
            print(f"      {ws}")

    # ===================== DETAILED WF TABLE =====================
    print("\n" + "=" * 130)
    print("  DETAILED WF TABLE: ALL TESTED CONFIGS")
    print("=" * 130)

    print(f"\n  {'Config':80s} | {'2020':>12s} | {'2021':>12s} | {'2022':>12s} | {'2023':>12s} | {'2024':>12s} | {'2025':>12s} | {'Avg':>7s} | {'WfMDD':>6s}")
    print(f"  {'-'*80}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*7}-+-{'-'*6}")

    for desc, wf_res in wf_all.items():
        vals = []
        for yr in [2020, 2021, 2022, 2023, 2024, 2025]:
            if yr in wf_res:
                vals.append(f"{wf_res[yr]['ann']:+.0f}/{wf_res[yr]['mdd']:.0f}")
            else:
                vals.append("N/A")
        avg_ann = np.mean([r['ann'] for r in wf_res.values()])
        worst_mdd = min(r['mdd'] for r in wf_res.values())
        print(f"  {desc:80s} | {vals[0]:>12s} | {vals[1]:>12s} | {vals[2]:>12s} | {vals[3]:>12s} | {vals[4]:>12s} | {vals[5]:>12s} | {avg_ann:>+6.0f}% | {worst_mdd:>5.1f}%")

    # ===================== FINAL SUMMARY: TOP 3 by WF avg with MDD < -30% =====================
    print("\n" + "=" * 130)
    print("  FINAL SUMMARY: TOP 3 CONFIGS by WF avg return (WF MDD < -30%)")
    print("=" * 130)

    # This is the key output requested by the user
    final_candidates = []
    for desc, wf_res in wf_all.items():
        avg_ann = np.mean([r['ann'] for r in wf_res.values()])
        worst_mdd = min(r['mdd'] for r in wf_res.values())
        pos_years = sum(1 for r in wf_res.values() if r['ann'] > 0)
        avg_wr = np.mean([r['wr'] for r in wf_res.values()])
        final_candidates.append({
            'desc': desc, 'avg_ann': avg_ann, 'worst_mdd': worst_mdd,
            'pos_years': pos_years, 'avg_wr': avg_wr, 'wf_res': wf_res
        })

    # Filter: WF MDD < -30% means worst_mdd > -30 (MDD is negative)
    filtered = [c for c in final_candidates if c['worst_mdd'] > -30]
    filtered.sort(key=lambda x: -x['avg_ann'])

    if filtered:
        for i, c in enumerate(filtered[:3]):
            print(f"\n  TOP #{i+1}: {c['desc']}")
            print(f"      AvgWF Ann = {c['avg_ann']:+.1f}% | Worst WF MDD = {c['worst_mdd']:.1f}% | {c['pos_years']}/6 pos | Avg WR = {c['avg_wr']:.1f}%")
            ws = " | ".join([f"{yr}:{r['ann']:+.0f}%/{r['mdd']:.0f}%"
                             for yr, r in sorted(c['wf_res'].items())])
            print(f"      {ws}")
    else:
        print("\n  No configs have WF worst MDD > -30%. Showing top 5 regardless:")
        final_candidates.sort(key=lambda x: -x['avg_ann'])
        for i, c in enumerate(final_candidates[:5]):
            print(f"\n  #{i+1}: {c['desc']}")
            print(f"      AvgWF Ann = {c['avg_ann']:+.1f}% | Worst WF MDD = {c['worst_mdd']:.1f}% | {c['pos_years']}/6 pos")
            ws = " | ".join([f"{yr}:{r['ann']:+.0f}%/{r['mdd']:.0f}%"
                             for yr, r in sorted(c['wf_res'].items())])
            print(f"      {ws}")

    print(f"\n  Elapsed: {time.time()-t_start:.0f}s")
    print("=" * 130)


if __name__ == '__main__':
    main()
