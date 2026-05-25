"""
Alpha Futures V134 — SEASONALITY-ENHANCED MOMENTUM
=============================================================================
Commodity futures have strong seasonal patterns. Agriculturals rally before
planting, energy rallies in winter. Enhance V121's momentum signal with
seasonality knowledge for higher win rate and larger moves.

Strategies:
  A) Monthly Win Rate Filter — only trade in months with WR > 55%
  B) Monthly Momentum (same month last year) — boost if last year was positive
  C) Seasonal OI Pattern — boost when OI historically increases this month
  D) V121 + Seasonality Confirm — V121 AND historically top-3 month
  E) Month-of-year Momentum Ranking — adaptive ROC thresholds by month rank
  F) Combined Seasonality Mega — A + B + D all confirming

Walk-forward: 2020-2025, expanding window for seasonality parameters.
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
    print("=" * 120)
    print("  V134 — SEASONALITY-ENHANCED MOMENTUM")
    print("=" * 120)
    t_start = time.time()
    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)
    print(f"  {NS} commodities, {ND} days")

    # Precompute month array
    MONTHS = np.array([d.month for d in dates])  # shape (ND,)
    YEARS = np.array([d.year for d in dates])    # shape (ND,)

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

    # Precompute OI change (daily)
    OI_CHANGE = np.full((NS, ND), np.nan)
    for si in range(NS):
        oi = OI[si].astype(np.float64)
        for di in range(1, ND):
            if not np.isnan(oi[di]) and not np.isnan(oi[di-1]) and oi[di-1] > 0:
                OI_CHANGE[si, di] = (oi[di] / oi[di-1] - 1) * 100

    print(f"  Done ({time.time()-t0:.1f}s)")

    # =====================================================================
    # SEASONALITY PRECOMPUTATION (expanding window)
    # =====================================================================
    # For each test year, compute seasonality stats from ALL data before that year.
    # We precompute lookup tables keyed by (test_year, si, month).

    def compute_seasonality_stats(up_to_di):
        """
        Compute seasonality stats using data from di=0 to di=up_to_di (exclusive).
        Returns:
          monthly_wr[si, month]    — win rate of ROC5>0 signals per (commodity, month)
          monthly_avg_roc[si, month] — average ROC5 per (commodity, month)
          monthly_oi_chg[si, month] — average OI change per (commodity, month)
          monthly_top3[si]         — set of top 3 months for each commodity
          same_month_ret[si, year-1, month] — return in same month last year
        """
        monthly_wr = np.full((NS, 13), np.nan)       # 1-indexed months
        monthly_avg_roc = np.full((NS, 13), np.nan)
        monthly_oi_chg = np.full((NS, 13), np.nan)
        monthly_top3 = [set() for _ in range(NS)]

        for si in range(NS):
            # Gather ROC5, RET, OI_CHANGE per month
            month_wins = {m: [0, 0] for m in range(1, 13)}  # [wins, total]
            month_roc_sum = {m: [] for m in range(1, 13)}
            month_oi_sum = {m: [] for m in range(1, 13)}

            for di in range(21, up_to_di):
                m = MONTHS[di]
                roc = ROC5[si, di]
                if np.isnan(roc): continue
                # Win = ROC5 > 0 on that day
                month_wins[m][1] += 1
                if roc > 0:
                    month_wins[m][0] += 1
                month_roc_sum[m].append(roc)
                oi_c = OI_CHANGE[si, di]
                if not np.isnan(oi_c):
                    month_oi_sum[m].append(oi_c)

            for m in range(1, 13):
                w, t = month_wins[m]
                if t >= 5:  # need at least 5 observations
                    monthly_wr[si, m] = w / t * 100
                if month_roc_sum[m]:
                    monthly_avg_roc[si, m] = np.mean(month_roc_sum[m])
                if month_oi_sum[m]:
                    monthly_oi_chg[si, m] = np.mean(month_oi_sum[m])

            # Top 3 months by average ROC5
            valid_months = [(m, monthly_avg_roc[si, m]) for m in range(1, 13)
                           if not np.isnan(monthly_avg_roc[si, m])]
            if len(valid_months) >= 3:
                valid_months.sort(key=lambda x: -x[1])
                monthly_top3[si] = set(m for m, _ in valid_months[:3])

        return monthly_wr, monthly_avg_roc, monthly_oi_chg, monthly_top3

    def get_same_month_last_year_return(si, di):
        """Get the average daily return in the same calendar month last year."""
        cur_month = MONTHS[di]
        cur_year = YEARS[di]
        prev_year = cur_year - 1
        rets = []
        for d in range(max(0, di - 400), di):
            if YEARS[d] == prev_year and MONTHS[d] == cur_month:
                r = RET[si, d]
                if not np.isnan(r):
                    rets.append(r)
        if len(rets) >= 3:
            return np.mean(rets)
        return np.nan

    # Cache seasonality stats per test year
    seas_cache = {}

    def get_seas_stats(test_year):
        if test_year in seas_cache:
            return seas_cache[test_year]
        # Find the first di of test_year
        up_to_di = MIN_TRAIN
        for di in range(ND):
            if dates[di].year == test_year:
                up_to_di = di
                break
        stats = compute_seasonality_stats(up_to_di)
        seas_cache[test_year] = stats
        return stats

    # Pre-fill cache for all test years
    print("\n[Seasonality Stats]...", flush=True)
    t1 = time.time()
    for yr in [2020, 2021, 2022, 2023, 2024, 2025]:
        get_seas_stats(yr)
        print(f"  Year {yr} stats computed")
    print(f"  Done ({time.time()-t1:.1f}s)")

    # =====================================================================
    # BACKTEST + WALK-FORWARD
    # =====================================================================

    def backtest(signal_func, hold=1, top_n=1, start_di=MIN_TRAIN, end_di=None, desc=""):
        if end_di is None: end_di = ND
        cash = float(CASH0); positions = []; trades = []; daily_eq = []
        for di in range(start_di, end_di - 1):
            pv = cash
            for p in positions:
                cp = C[p['si'], di]
                if not np.isnan(cp) and cp > 0:
                    m = MULT.get(p['sym'], DEF_MULT)
                    pv += cp * m * p['lots'] - cp * m * abs(p['lots']) * COMM
            daily_eq.append(pv)
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
                    trades.append({'pnl_pct': pp, 'sig': p.get('sig', '')})
                    cl.append(p)
            for p in cl: positions.remove(p)
            if len(positions) >= top_n: continue
            edi = di + 1
            if edi >= end_di: continue
            cands = signal_func(di, edi)
            if not cands: continue
            cands.sort(key=lambda x: -x[0])
            ns = top_n - len(positions)
            cap = cash / max(1, ns)
            for item in cands[:ns]:
                if len(item) == 3: sc, s, pr = item; sig = ''
                else: sc, s, pr, sig = item
                sym = syms[s]; m = MULT.get(sym, DEF_MULT)
                ct = max(1, int(cap * 0.95 / (pr * m * (1 + COMM))))
                ci = pr * m * ct * (1 + COMM)
                if ci > cash:
                    ct = int(cash * 0.9 / (pr * m * (1 + COMM)))
                    ci = pr * m * ct * (1 + COMM) if ct > 0 else 0
                if ct <= 0 or ci <= 0 or ci > cash: continue
                cash -= ci
                positions.append({'si': s, 'entry_price': pr, 'entry_di': edi,
                                  'lots': ct, 'dir': 1, 'sym': sym, 'hold_days': hold, 'sig': sig})
        for p in positions:
            ae = end_di - 1
            ep = C[p['si'], min(ae, ND-1)]
            if np.isnan(ep) or ep <= 0: ep = p['entry_price']
            m = MULT.get(p['sym'], DEF_MULT)
            cash += ep * m * abs(p['lots']) * (1 - COMM)
        nd = end_di - start_di
        ann = annual_return(cash, CASH0, nd)
        wr = np.mean([1 if t['pnl_pct'] > 0 else 0 for t in trades]) * 100 if trades else 0
        nt = len(trades)
        ap = np.mean([t['pnl_pct'] for t in trades]) if trades else 0
        if daily_eq:
            eq = np.array(daily_eq); pk = np.maximum.accumulate(eq)
            mdd = np.min((eq - pk) / pk * 100)
            r = np.diff(eq) / eq[:-1]
            sh = np.mean(r) / np.std(r) * np.sqrt(252) if np.std(r) > 0 else 0
        else: mdd = 0; sh = 0
        return {'ann': ann, 'wr': wr, 'n': nt, 'avg_pnl': ap, 'mdd': mdd, 'sharpe': sh,
                'desc': desc}

    def pr(r, label=""):
        print(f"  {label:60s} | Ann={r['ann']:+8.1f}% | WR={r['wr']:5.1f}% | "
              f"N={r['n']:4d} | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f}")

    def wf(func, hold=1, topn=1, use_seasonality=False):
        res = {}
        for yr in [2020, 2021, 2022, 2023, 2024, 2025]:
            ys = ye = None
            for di in range(ND):
                if dates[di].year == yr and ys is None: ys = di
                if dates[di].year == yr: ye = di + 1
            if ys:
                # For walk-forward, use seasonality stats from before the test year
                r = backtest(func, hold=hold, top_n=topn, start_di=ys, end_di=ye)
                res[yr] = r['ann']
        return res

    def wf_with_seas(func, hold=1, topn=1):
        """Walk-forward with per-year seasonality stats."""
        res = {}
        for yr in [2020, 2021, 2022, 2023, 2024, 2025]:
            ys = ye = None
            for di in range(ND):
                if dates[di].year == yr and ys is None: ys = di
                if dates[di].year == yr: ye = di + 1
            if ys:
                # Get seasonality stats for this year
                monthly_wr, monthly_avg_roc, monthly_oi_chg, monthly_top3 = get_seas_stats(yr)
                r = backtest(func, hold=hold, top_n=topn, start_di=ys, end_di=ye,
                             desc=f"wf_{yr}")
                res[yr] = r['ann']
        return res

    # =====================================================================
    # SIGNAL FUNCTIONS
    # =====================================================================

    # V121 baseline (for comparison)
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

    # A) Monthly Win Rate Filter
    # In training period, compute win rate of ROC(5)>0 signals for each (commodity, month) pair
    # Only trade when current month has historical WR > 55% for that commodity
    # Score: ROC(5) * Z-score * monthly_WR
    def sig_monthly_wr_filter(di, edi):
        cur_year = dates[di].year
        monthly_wr, monthly_avg_roc, monthly_oi_chg, monthly_top3 = get_seas_stats(cur_year)
        c = []
        for s in range(NS):
            roc = ROC5[s, di]; zs = ZSCORE[s, di]
            if np.isnan(roc) or np.isnan(zs) or roc <= 0.5 or zs <= 1.0: continue
            m = MONTHS[di]
            wr = monthly_wr[s, m]
            if np.isnan(wr) or wr <= 55.0: continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            score = roc * zs * (wr / 100.0)
            c.append((score, s, ep, 'monthly_wr'))
        return c

    # B) Monthly Momentum (same month last year)
    # If that month was positive last year, boost the signal score
    # Score: ROC(5) * Z-score * (1 + last_year_same_month_return/10)
    def sig_monthly_momentum(di, edi):
        c = []
        for s in range(NS):
            roc = ROC5[s, di]; zs = ZSCORE[s, di]
            if np.isnan(roc) or np.isnan(zs) or roc <= 0.5 or zs <= 1.0: continue
            smr = get_same_month_last_year_return(s, di)
            if np.isnan(smr): continue
            boost = 1.0 + smr / 10.0
            if boost <= 0.5: continue  # Skip if last year same month was very negative
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            score = roc * zs * boost
            c.append((score, s, ep, 'monthly_mom'))
        return c

    # C) Seasonal OI Pattern
    # When current month historically has OI increasing, boost momentum signals
    # Score: ROC(5) * Z-score * (1 + avg_monthly_oi_change)
    def sig_seasonal_oi(di, edi):
        cur_year = dates[di].year
        monthly_wr, monthly_avg_roc, monthly_oi_chg, monthly_top3 = get_seas_stats(cur_year)
        c = []
        for s in range(NS):
            roc = ROC5[s, di]; zs = ZSCORE[s, di]
            if np.isnan(roc) or np.isnan(zs) or roc <= 0.5 or zs <= 1.0: continue
            m = MONTHS[di]
            oi_c = monthly_oi_chg[s, m]
            if np.isnan(oi_c): continue
            # Only boost when OI is historically increasing
            if oi_c <= 0: continue
            boost = 1.0 + oi_c / 100.0  # oi_c is in pct, normalize
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            score = roc * zs * boost
            c.append((score, s, ep, 'seasonal_oi'))
        return c

    # D) V121 + Seasonality Confirm
    # V121 signal (ROC>1%, Z>1.5, ROC improving) AND historically strong month (top 3 months)
    def sig_v121_seasonal_confirm(di, edi):
        cur_year = dates[di].year
        monthly_wr, monthly_avg_roc, monthly_oi_chg, monthly_top3 = get_seas_stats(cur_year)
        c = []
        for s in range(NS):
            # V121 criteria
            roc = ROC5[s, di]; zs = ZSCORE[s, di]
            if np.isnan(roc) or np.isnan(zs) or roc <= 1.0 or zs <= 1.5: continue
            rp = ROC5[s, di-1] if di > 0 else np.nan
            if not np.isnan(rp) and roc <= rp: continue
            # Seasonality: must be top 3 month for this commodity
            m = MONTHS[di]
            if m not in monthly_top3[s]: continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            # Bonus for top month
            avg_roc = monthly_avg_roc[s, m]
            bonus = 1.0
            if not np.isnan(avg_roc) and avg_roc > 0:
                bonus = 1.0 + avg_roc / 100.0
            score = roc * zs * bonus
            c.append((score, s, ep, 'v121_seas'))
        return c

    # E) Month-of-year Momentum Ranking
    # Top months: lower ROC threshold (0.8%); Bottom months: higher ROC threshold (1.5%)
    def sig_adaptive_threshold(di, edi):
        cur_year = dates[di].year
        monthly_wr, monthly_avg_roc, monthly_oi_chg, monthly_top3 = get_seas_stats(cur_year)
        c = []
        for s in range(NS):
            zs = ZSCORE[s, di]
            if np.isnan(zs) or zs <= 1.0: continue
            roc = ROC5[s, di]
            if np.isnan(roc): continue
            m = MONTHS[di]
            avg_roc = monthly_avg_roc[s, m]
            # Adaptive threshold based on seasonal strength
            if np.isnan(avg_roc):
                threshold = 1.0  # Default if no data
            elif m in monthly_top3[s]:
                threshold = 0.8  # Easier for strong months
            elif avg_roc > 0:
                threshold = 1.0  # Normal for decent months
            else:
                threshold = 1.5  # Harder for weak months
            if roc <= threshold: continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            score = roc * zs
            c.append((score, s, ep, 'adaptive_thr'))
        return c

    # F) Combined Seasonality Mega
    # V121 signal + monthly WR>55% + same-month-last-year positive
    def sig_seasonal_mega(di, edi):
        cur_year = dates[di].year
        monthly_wr, monthly_avg_roc, monthly_oi_chg, monthly_top3 = get_seas_stats(cur_year)
        c = []
        for s in range(NS):
            # V121 criteria
            roc = ROC5[s, di]; zs = ZSCORE[s, di]
            if np.isnan(roc) or np.isnan(zs) or roc <= 1.0 or zs <= 1.5: continue
            rp = ROC5[s, di-1] if di > 0 else np.nan
            if not np.isnan(rp) and roc <= rp: continue
            m = MONTHS[di]
            # Filter A: monthly WR > 55%
            wr = monthly_wr[s, m]
            if np.isnan(wr) or wr <= 55.0: continue
            # Filter B: same month last year positive
            smr = get_same_month_last_year_return(s, di)
            if np.isnan(smr) or smr <= 0: continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            # Combined boost
            boost = (wr / 100.0) * (1.0 + smr / 10.0)
            score = roc * zs * boost
            c.append((score, s, ep, 'seas_mega'))
        return c

    # =====================================================================
    # SECTION 1: ALL CONFIGURATIONS
    # =====================================================================
    print("\n" + "=" * 120)
    print("  SECTION 1: SEASONALITY-ENHANCED MOMENTUM CONFIGURATIONS")
    print("=" * 120)

    configs = [
        ("V121 baseline", sig_v121, 1, 1),
        ("A) Monthly WR Filter (WR>55%)", sig_monthly_wr_filter, 1, 1),
        ("B) Monthly Momentum (last year)", sig_monthly_momentum, 1, 1),
        ("C) Seasonal OI Pattern", sig_seasonal_oi, 1, 1),
        ("D) V121 + Seasonal Confirm (top3)", sig_v121_seasonal_confirm, 1, 1),
        ("E) Adaptive Threshold by Month", sig_adaptive_threshold, 1, 1),
        ("F) Seasonal Mega (A+B+D)", sig_seasonal_mega, 1, 1),
    ]

    results = {}
    for name, func, hold, topn in configs:
        r = backtest(func, hold=hold, top_n=topn, desc=name)
        results[name] = r
        pr(r, label=name)

    # =====================================================================
    # SECTION 2: TOP_N x HOLD VARIATIONS
    # =====================================================================
    print("\n" + "=" * 120)
    print("  SECTION 2: TOP_N x HOLD for ALL strategies")
    print("=" * 120)

    for name, func, _, _ in configs:
        print(f"\n  {name}:")
        for topn in [1, 2]:
            for hold in [1, 2]:
                r = backtest(func, hold=hold, top_n=topn, desc=f"{name} t={topn} h={hold}")
                print(f"    top_n={topn} hold={hold}: Ann={r['ann']:+8.1f}% | "
                      f"WR={r['wr']:5.1f}% | N={r['n']:4d} | MDD={r['mdd']:6.1f}%")

    # =====================================================================
    # SECTION 3: WALK-FORWARD 2020-2025
    # =====================================================================
    print("\n" + "=" * 120)
    print("  SECTION 3: WALK-FORWARD 2020-2025 (expanding window seasonality)")
    print("=" * 120)

    for name, func, hold, topn in configs:
        w = {}
        for yr in [2020, 2021, 2022, 2023, 2024, 2025]:
            ys = ye = None
            for di in range(ND):
                if dates[di].year == yr and ys is None: ys = di
                if dates[di].year == yr: ye = di + 1
            if ys:
                r = backtest(func, hold=hold, top_n=topn, start_di=ys, end_di=ye, desc=f"wf_{yr}")
                w[yr] = r['ann']
        ws = " | ".join([f"{yr}:{v:+.0f}%" for yr, v in sorted(w.items())])
        pos = sum(1 for v in w.values() if v > 0)
        avg = np.mean(list(w.values())) if w else 0
        print(f"  {name:60s} | {pos}/6 | Avg={avg:>+7.0f}% | {ws}")

    # =====================================================================
    # SECTION 4: WALK-FORWARD with TOP_N=2
    # =====================================================================
    print("\n" + "=" * 120)
    print("  SECTION 4: WALK-FORWARD TOP_N=2")
    print("=" * 120)

    for name, func, hold, _ in configs:
        w = {}
        for yr in [2020, 2021, 2022, 2023, 2024, 2025]:
            ys = ye = None
            for di in range(ND):
                if dates[di].year == yr and ys is None: ys = di
                if dates[di].year == yr: ye = di + 1
            if ys:
                r = backtest(func, hold=hold, top_n=2, start_di=ys, end_di=ye, desc=f"wf_{yr}")
                w[yr] = r['ann']
        ws = " | ".join([f"{yr}:{v:+.0f}%" for yr, v in sorted(w.items())])
        pos = sum(1 for v in w.values() if v > 0)
        avg = np.mean(list(w.values())) if w else 0
        print(f"  {name:60s} | {pos}/6 | Avg={avg:>+7.0f}% | {ws}")

    # =====================================================================
    # SECTION 5: COMBINED SEASONALITY PORTFOLIO
    # =====================================================================
    print("\n" + "=" * 120)
    print("  SECTION 5: PORTFOLIO COMBINATIONS (top_n=2-3, hold=1-2)")
    print("=" * 120)

    portfolio_configs = [
        ("A) WR Filter t=2 h=1", sig_monthly_wr_filter, 1, 2),
        ("A) WR Filter t=2 h=2", sig_monthly_wr_filter, 2, 2),
        ("B) Monthly Mom t=2 h=1", sig_monthly_momentum, 1, 2),
        ("D) V121+Seas t=2 h=1", sig_v121_seasonal_confirm, 1, 2),
        ("E) Adaptive t=2 h=1", sig_adaptive_threshold, 1, 2),
        ("E) Adaptive t=3 h=1", sig_adaptive_threshold, 1, 3),
        ("F) Mega t=2 h=1", sig_seasonal_mega, 1, 2),
        ("F) Mega t=2 h=2", sig_seasonal_mega, 2, 2),
    ]

    for name, func, hold, topn in portfolio_configs:
        r = backtest(func, hold=hold, top_n=topn, desc=name)
        pr(r, label=name)

    # =====================================================================
    # SUMMARY
    # =====================================================================
    print("\n" + "=" * 120)
    print("  SUMMARY: TOP 20 BY ANNUAL RETURN")
    print("=" * 120)

    all_r = {**results}
    for name, func, hold, topn in portfolio_configs:
        r = backtest(func, hold=hold, top_n=topn, desc=name)
        all_r[name] = r

    sorted_r = sorted(all_r.items(), key=lambda x: -x[1]['ann'])
    for i, (name, r) in enumerate(sorted_r[:20]):
        print(f"  #{i+1}: {name:60s} | Ann={r['ann']:+8.1f}% | WR={r['wr']:5.1f}% | "
              f"N={r['n']:4d} | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f}")

    sorted_sh = sorted(all_r.items(), key=lambda x: -x[1]['sharpe'])
    print(f"\n  TOP 10 BY SHARPE:")
    for i, (name, r) in enumerate(sorted_sh[:10]):
        print(f"  #{i+1}: {name:60s} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f}")

    print(f"\n  Elapsed: {time.time()-t_start:.0f}s")
    print("=" * 120)


if __name__ == '__main__':
    main()
