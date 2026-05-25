"""
Alpha Futures V150 — Kelly Criterion & Optimal Position Sizing
==============================================================================
Goal: Explore whether mathematically optimal sizing (Kelly Criterion) improves
      the return/MDD frontier over fixed sizing and DD tiers.

Kelly Criterion: f* = (p * b - q) / b
  where p = win probability, b = avg win/avg loss ratio, q = 1-p
  Use rolling window (last 20-50 trades) to estimate p and b.
  Apply fractional Kelly (25%, 50%, 75%) for safety.

Sections:
  0: Baselines (fixed sizing 55%, DD tiers 70/60/40/20)
  1: Kelly criterion sizing sweep (full/75/50/25% Kelly)
  2: Kelly + DD cap (min(Kelly, DD_tier_size))
  3: Rolling window comparison (20/30/50 trades)
  4: WF validation for top 10 configs
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
    print("  V150 — KELLY CRITERION & OPTIMAL POSITION SIZING")
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

    # ===================== HELPER: Kelly Criterion =====================
    def kelly_fraction(trades, window, kelly_frac=1.0, default=0.55):
        """
        Compute Kelly fraction from rolling trade history.

        f* = (p * b - q) / b
          p = win probability
          b = avg win / avg loss ratio (odds)
          q = 1 - p

        Returns: position size fraction (0.05 to 0.95)
                 kelly_frac: fractional Kelly multiplier (0.25, 0.50, 0.75, 1.0)
        """
        if len(trades) < max(10, window // 2):
            return default  # Not enough data

        recent = trades[-window:]
        wins = [t for t in recent if t > 0]
        losses = [t for t in recent if t <= 0]

        n_total = len(recent)
        n_wins = len(wins)
        n_losses = len(losses)

        if n_wins == 0 or n_losses == 0:
            # All wins or all losses -> use default with caution
            if n_wins == 0:
                return 0.10  # Very conservative
            return min(default * 1.2, 0.80)  # Mildly aggressive

        p = n_wins / n_total
        q = 1.0 - p
        avg_win = np.mean(wins)
        avg_loss = abs(np.mean(losses))

        if avg_loss < 1e-9:
            return min(default * 1.2, 0.80)

        b = avg_win / avg_loss  # odds ratio

        # Kelly: f* = (p*b - q) / b
        kelly_f = (p * b - q) / b

        if kelly_f <= 0:
            # Negative Kelly: don't bet
            return 0.10

        # Apply fractional Kelly
        f = kelly_f * kelly_frac

        # Clamp to [0.05, 0.95]
        return max(0.05, min(0.95, f))

    # ===================== UNIFIED BACKTEST ENGINE =====================
    def backtest_v150(start_di=MIN_TRAIN, end_di=None,
                      # Sizing mode
                      sizing='fixed',
                      # Fixed sizing params
                      base_size=0.55,
                      # DD tiers
                      dd_tiers=None,
                      # Kelly params
                      kelly_frac=1.0,       # 1.0 = full Kelly, 0.5 = half Kelly
                      kelly_window=30,      # rolling window of trades
                      kelly_default=0.55,   # default size when not enough trades
                      # Kelly + DD cap
                      kelly_dd_cap=False,   # if True, min(kelly, dd_tier)
                      # General
                      hold=1, top_n=2, max_corr=0.5):
        if end_di is None: end_di = ND
        if dd_tiers is None:
            dd_tiers = [(0, 0.70), (0.10, 0.60), (0.20, 0.40), (0.30, 0.20)]

        cash = float(CASH0)
        positions = []
        trades = []       # list of pnl_pct (percentage)
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

            # --- Compute position size based on sizing mode ---
            if sizing == 'fixed':
                pos_size = base_size

            elif sizing == 'dd_tiers':
                pos_size = dd_size(pv, high_water, dd_tiers)

            elif sizing == 'kelly':
                pos_size = kelly_fraction(trades, kelly_window, kelly_frac, kelly_default)

            elif sizing == 'kelly_dd':
                kelly_size = kelly_fraction(trades, kelly_window, kelly_frac, kelly_default)
                dd_sz = dd_size(pv, high_water, dd_tiers)
                if kelly_dd_cap:
                    pos_size = min(kelly_size, dd_sz)
                else:
                    pos_size = kelly_size  # pure kelly

            else:
                pos_size = base_size

            # Clamp
            pos_size = max(0.05, min(0.95, pos_size))

            # --- Enter positions (balanced: V121 + Union with corr filter) ---
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

            # BUG PREVENTION: snapshot cash before entry loop
            cash_snapshot = cash
            n_planned = len(entries)
            for sc, s, pr, sig_str, pct in entries:
                if s in set(p['si'] for p in positions): continue
                if len(positions) >= top_n: break
                cap = cash_snapshot * pct / n_planned  # Equal split among planned entries
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
        print(f"  {label:80s} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | WR={r['wr']:5.1f}% | Sh={r['sharpe']:4.2f} | R/M={ratio:.2f} | N={r['n']:4d}")

    def walk_forward(sizing='fixed', base_size=0.55,
                     dd_tiers=None, kelly_frac=1.0, kelly_window=30,
                     kelly_default=0.55, kelly_dd_cap=False,
                     hold=1, top_n=2, max_corr=0.5, label=""):
        res = {}
        for yr in [2020, 2021, 2022, 2023, 2024, 2025]:
            ys = ye = None
            for di in range(ND):
                if dates[di].year == yr and ys is None: ys = di
                if dates[di].year == yr: ye = di + 1
            if ys is None: continue
            r = backtest_v150(start_di=ys, end_di=ye, sizing=sizing,
                              base_size=base_size, dd_tiers=dd_tiers,
                              kelly_frac=kelly_frac, kelly_window=kelly_window,
                              kelly_default=kelly_default, kelly_dd_cap=kelly_dd_cap,
                              hold=hold, top_n=top_n, max_corr=max_corr)
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

    # ===================== SECTION 0: BASELINES =====================
    print("\n" + "=" * 130)
    print("  SECTION 0: BASELINES")
    print("=" * 130)

    baseline_results = []

    # Fixed sizing sweep
    for bs in [0.45, 0.50, 0.55]:
        r = backtest_v150(sizing='fixed', base_size=bs)
        r['desc'] = f"Fixed {bs*100:.0f}%"
        baseline_results.append(r)
        pr(r, f"S0: Fixed sizing {bs*100:.0f}%")

    # DD tiers baseline
    dd_default = [(0, 0.70), (0.10, 0.60), (0.20, 0.40), (0.30, 0.20)]
    r = backtest_v150(sizing='dd_tiers', dd_tiers=dd_default)
    r['desc'] = "DD tiers 70/60/40/20%"
    baseline_results.append(r)
    pr(r, "S0: DD tiers 70/60/40/20%")

    # DD tiers more aggressive
    dd_aggressive = [(0, 0.80), (0.10, 0.65), (0.20, 0.50), (0.30, 0.35)]
    r = backtest_v150(sizing='dd_tiers', dd_tiers=dd_aggressive)
    r['desc'] = "DD tiers 80/65/50/35%"
    baseline_results.append(r)
    pr(r, "S0: DD tiers 80/65/50/35%")

    # ===================== SECTION 1: KELLY CRITERION SIZING SWEEP =====================
    print("\n" + "=" * 130)
    print("  SECTION 1: KELLY CRITERION SIZING SWEEP")
    print("  f* = (p*b - q)/b, fractional Kelly for safety")
    print("=" * 130)

    kelly_results = []

    # Full Kelly with different windows
    for win in [20, 30, 50]:
        r = backtest_v150(sizing='kelly', kelly_frac=1.0, kelly_window=win, kelly_default=0.55)
        desc = f"Full Kelly w={win} default=55%"
        r['desc'] = desc
        kelly_results.append(r)
        pr(r, f"S1: {desc}")

    # 75% Kelly
    for win in [20, 30, 50]:
        r = backtest_v150(sizing='kelly', kelly_frac=0.75, kelly_window=win, kelly_default=0.55)
        desc = f"75% Kelly w={win} default=55%"
        r['desc'] = desc
        kelly_results.append(r)
        pr(r, f"S1: {desc}")

    # 50% Kelly (half Kelly - classic recommendation)
    for win in [20, 30, 50]:
        for dfl in [0.45, 0.55, 0.65]:
            r = backtest_v150(sizing='kelly', kelly_frac=0.50, kelly_window=win, kelly_default=dfl)
            desc = f"50% Kelly w={win} default={dfl*100:.0f}%"
            r['desc'] = desc
            kelly_results.append(r)
            pr(r, f"S1: {desc}")

    # 25% Kelly (quarter Kelly - very conservative)
    for win in [20, 30, 50]:
        r = backtest_v150(sizing='kelly', kelly_frac=0.25, kelly_window=win, kelly_default=0.55)
        desc = f"25% Kelly w={win} default=55%"
        r['desc'] = desc
        kelly_results.append(r)
        pr(r, f"S1: {desc}")

    # ===================== SECTION 2: KELLY + DD CAP =====================
    print("\n" + "=" * 130)
    print("  SECTION 2: KELLY + DD CAP")
    print("  min(Kelly_size, DD_tier_size)")
    print("=" * 130)

    kelly_dd_results = []

    dd_configs = [
        ([(0, 0.70), (0.10, 0.60), (0.20, 0.40), (0.30, 0.20)], "DD 70/60/40/20"),
        ([(0, 0.80), (0.10, 0.65), (0.20, 0.50), (0.30, 0.35)], "DD 80/65/50/35"),
        ([(0, 0.60), (0.10, 0.50), (0.20, 0.35), (0.30, 0.20)], "DD 60/50/35/20"),
    ]

    for kfrac, kfrac_label in [(1.0, "Full"), (0.75, "75%"), (0.50, "50%"), (0.25, "25%")]:
        for win in [20, 30, 50]:
            for dd_t, dd_label in dd_configs:
                r = backtest_v150(sizing='kelly_dd', kelly_frac=kfrac,
                                  kelly_window=win, kelly_default=0.55,
                                  kelly_dd_cap=True, dd_tiers=dd_t)
                desc = f"{kfrac_label} Kelly cap {dd_label} w={win}"
                r['desc'] = desc
                kelly_dd_results.append(r)
                pr(r, f"S2: {desc}")

    # ===================== SECTION 3: ROLLING WINDOW COMPARISON =====================
    print("\n" + "=" * 130)
    print("  SECTION 3: ROLLING WINDOW COMPARISON")
    print("  Focus on 50% Kelly with varying windows and defaults")
    print("=" * 130)

    window_results = []

    for kfrac in [0.50, 0.75]:
        for win in [15, 20, 25, 30, 40, 50, 60]:
            for dfl in [0.45, 0.55, 0.65]:
                r = backtest_v150(sizing='kelly', kelly_frac=kfrac,
                                  kelly_window=win, kelly_default=dfl)
                desc = f"{kfrac*100:.0f}%Kelly w={win} dfl={dfl*100:.0f}%"
                r['desc'] = desc
                window_results.append(r)
                pr(r, f"S3: {desc}")

    # ===================== COMPREHENSIVE RANKING =====================
    print("\n" + "=" * 130)
    print("  COMPREHENSIVE RANKING")
    print("=" * 130)

    all_results = baseline_results + kelly_results + kelly_dd_results + window_results
    all_valid = [r for r in all_results if r.get('desc', '') and r['mdd'] > -80]

    # Sort by annual return
    all_valid.sort(key=lambda x: -x['ann'])
    print(f"\n  Top 20 by Annual Return:")
    for i, r in enumerate(all_valid[:20]):
        desc = r.get('desc', '')
        ratio = abs(r['ann'] / r['mdd']) if r['mdd'] != 0 else 0
        print(f"  #{i+1:2d}: {desc:80s} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f} | R/M={ratio:.2f}")

    # Sort by return/MDD ratio
    all_with_ratio = [(r, abs(r['ann'] / r['mdd']) if r['mdd'] != 0 else 0) for r in all_valid]
    all_with_ratio.sort(key=lambda x: -x[1])
    print(f"\n  Top 20 by Ann/MDD Ratio:")
    for i, (r, ratio) in enumerate(all_with_ratio[:20]):
        desc = r.get('desc', '')
        print(f"  #{i+1:2d}: {desc:80s} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f} | R/M={ratio:.2f}")

    # Sort by Sharpe
    all_valid_sh = list(all_valid)
    all_valid_sh.sort(key=lambda x: -x['sharpe'])
    print(f"\n  Top 20 by Sharpe:")
    for i, r in enumerate(all_valid_sh[:20]):
        desc = r.get('desc', '')
        ratio = abs(r['ann'] / r['mdd']) if r['mdd'] != 0 else 0
        print(f"  #{i+1:2d}: {desc:80s} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f} | R/M={ratio:.2f}")

    # ===================== SECTION 4: WALK-FORWARD FOR TOP CONFIGS =====================
    print("\n" + "=" * 130)
    print("  SECTION 4: WALK-FORWARD VALIDATION FOR TOP 10 CONFIGS")
    print("=" * 130)

    # Select top 10 by R/M ratio, unique configs
    seen = set()
    wf_configs = []
    for r, ratio in all_with_ratio:
        desc = r.get('desc', '')
        if desc not in seen:
            seen.add(desc)
            wf_configs.append(r)
        if len(wf_configs) >= 10:
            break

    # Also add the 3 baselines for comparison
    for r in baseline_results:
        desc = r.get('desc', '')
        if desc not in seen:
            seen.add(desc)
            wf_configs.append(r)

    wf_all = {}

    # Helper to parse config desc and run walk-forward
    def run_wf_from_desc(r):
        desc = r.get('desc', '')
        sizing = 'fixed'
        base_size = 0.55
        dd_tiers = None
        kelly_frac = 1.0
        kelly_window = 30
        kelly_default = 0.55
        kelly_dd_cap = False

        if desc.startswith("Fixed"):
            sizing = 'fixed'
            for val in [0.45, 0.50, 0.55]:
                if f"{val*100:.0f}%" in desc:
                    base_size = val
                    break

        elif desc.startswith("DD tiers"):
            sizing = 'dd_tiers'
            if "80/65/50/35" in desc:
                dd_tiers = [(0, 0.80), (0.10, 0.65), (0.20, 0.50), (0.30, 0.35)]
            else:
                dd_tiers = [(0, 0.70), (0.10, 0.60), (0.20, 0.40), (0.30, 0.20)]

        elif "Kelly cap" in desc:
            sizing = 'kelly_dd'
            kelly_dd_cap = True
            # Parse Kelly fraction
            if desc.startswith("Full"):
                kelly_frac = 1.0
            elif desc.startswith("75%"):
                kelly_frac = 0.75
            elif desc.startswith("50%"):
                kelly_frac = 0.50
            elif desc.startswith("25%"):
                kelly_frac = 0.25
            # Parse window
            if "w=20" in desc: kelly_window = 20
            elif "w=30" in desc: kelly_window = 30
            elif "w=50" in desc: kelly_window = 50
            # Parse DD tiers
            if "DD 80/65/50/35" in desc:
                dd_tiers = [(0, 0.80), (0.10, 0.65), (0.20, 0.50), (0.30, 0.35)]
            elif "DD 60/50/35/20" in desc:
                dd_tiers = [(0, 0.60), (0.10, 0.50), (0.20, 0.35), (0.30, 0.20)]
            else:
                dd_tiers = [(0, 0.70), (0.10, 0.60), (0.20, 0.40), (0.30, 0.20)]

        elif "Kelly" in desc:
            sizing = 'kelly'
            # Parse Kelly fraction
            if desc.startswith("Full"):
                kelly_frac = 1.0
            elif desc.startswith("75%"):
                kelly_frac = 0.75
            elif desc.startswith("50%"):
                kelly_frac = 0.50
            elif desc.startswith("25%"):
                kelly_frac = 0.25
            # Parse window
            if "w=15" in desc: kelly_window = 15
            elif "w=20" in desc: kelly_window = 20
            elif "w=25" in desc: kelly_window = 25
            elif "w=30" in desc: kelly_window = 30
            elif "w=40" in desc: kelly_window = 40
            elif "w=50" in desc: kelly_window = 50
            elif "w=60" in desc: kelly_window = 60
            # Parse default
            if "dfl=45" in desc or "default=45" in desc:
                kelly_default = 0.45
            elif "dfl=65" in desc or "default=65" in desc:
                kelly_default = 0.65
            else:
                kelly_default = 0.55

        return walk_forward(sizing=sizing, base_size=base_size,
                            dd_tiers=dd_tiers, kelly_frac=kelly_frac,
                            kelly_window=kelly_window, kelly_default=kelly_default,
                            kelly_dd_cap=kelly_dd_cap, label=desc)

    for r in wf_configs:
        desc = r.get('desc', '')
        wf_res = run_wf_from_desc(r)
        wf_all[desc] = wf_res
        print_wf(wf_res, desc)

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

    # ===================== FINAL SUMMARY: TOP 3 BY WF AVG WITH WF MDD < -35% =====================
    print("\n" + "=" * 130)
    print("  FINAL SUMMARY: TOP CONFIGS BY WF AVG RETURN (WF MDD < -35%)")
    print("=" * 130)

    wf_summary = []
    for desc, wf_res in wf_all.items():
        avg_ann = np.mean([r['ann'] for r in wf_res.values()])
        worst_mdd = min(r['mdd'] for r in wf_res.values())
        avg_mdd = np.mean([r['mdd'] for r in wf_res.values()])
        pos_years = sum(1 for r in wf_res.values() if r['ann'] > 0)
        avg_wr = np.mean([r['wr'] for r in wf_res.values()])
        wf_summary.append({
            'desc': desc, 'avg_ann': avg_ann, 'worst_mdd': worst_mdd,
            'avg_mdd': avg_mdd, 'pos_years': pos_years, 'avg_wr': avg_wr,
            'wf_res': wf_res
        })

    # Filter: WF MDD < -35% (i.e. worse than -35%)
    filtered = [s for s in wf_summary if s['worst_mdd'] < -35]
    filtered.sort(key=lambda x: -x['avg_ann'])

    if filtered:
        print(f"\n  Configs with WF WorstMDD < -35% (sorted by avg WF return):")
        for i, s in enumerate(filtered[:10]):
            ws = " | ".join([f"{yr}:{r['ann']:+.0f}%/{r['mdd']:.0f}%"
                             for yr, r in sorted(s['wf_res'].items())])
            print(f"\n  #{i+1}: {s['desc']}")
            print(f"       AvgWF={s['avg_ann']:+.0f}% | WorstWfMDD={s['worst_mdd']:.1f}% | {s['pos_years']}/6 pos | AvgWR={s['avg_wr']:.1f}%")
            print(f"       {ws}")
    else:
        print("\n  No configs have WF worst MDD < -35%. Showing all sorted by avg WF return:")
        wf_summary.sort(key=lambda x: -x['avg_ann'])
        for i, s in enumerate(wf_summary[:10]):
            ws = " | ".join([f"{yr}:{r['ann']:+.0f}%/{r['mdd']:.0f}%"
                             for yr, r in sorted(s['wf_res'].items())])
            print(f"\n  #{i+1}: {s['desc']}")
            print(f"       AvgWF={s['avg_ann']:+.0f}% | WorstWfMDD={s['worst_mdd']:.1f}% | {s['pos_years']}/6 pos | AvgWR={s['avg_wr']:.1f}%")
            print(f"       {ws}")

    # Also show configs with best WF MDD (least negative) among positive return configs
    print("\n" + "=" * 130)
    print("  BEST WF MDD CONTROL: Configs with avg WF return > 0%, sorted by best (least negative) WF MDD")
    print("=" * 130)

    positive_wf = [s for s in wf_summary if s['avg_ann'] > 0]
    positive_wf.sort(key=lambda x: -x['worst_mdd'])  # least negative MDD first

    for i, s in enumerate(positive_wf[:10]):
        ws = " | ".join([f"{yr}:{r['ann']:+.0f}%/{r['mdd']:.0f}%"
                         for yr, r in sorted(s['wf_res'].items())])
        print(f"\n  #{i+1}: {s['desc']}")
        print(f"       AvgWF={s['avg_ann']:+.0f}% | WorstWfMDD={s['worst_mdd']:.1f}% | {s['pos_years']}/6 pos | AvgWR={s['avg_wr']:.1f}%")
        print(f"       {ws}")

    # ===================== KELLY vs BASELINE COMPARISON =====================
    print("\n" + "=" * 130)
    print("  KELLY vs BASELINE COMPARISON")
    print("=" * 130)

    # Get baseline avg WF
    baseline_wf = {}
    for r in baseline_results:
        desc = r.get('desc', '')
        if desc in wf_all:
            avg = np.mean([r2['ann'] for r2 in wf_all[desc].values()])
            wmdd = min(r2['mdd'] for r2 in wf_all[desc].values())
            baseline_wf[desc] = (avg, wmdd)

    print(f"\n  Baseline WF results:")
    for desc, (avg, wmdd) in baseline_wf.items():
        print(f"    {desc:40s} | AvgWF={avg:>+7.0f}% | WorstWfMDD={wmdd:>5.1f}%")

    # Find Kelly configs that beat best baseline
    best_bl_ann = max(avg for avg, wmdd in baseline_wf.values()) if baseline_wf else 0
    best_bl_mdd = max(wmdd for avg, wmdd in baseline_wf.values()) if baseline_wf else 0  # least negative

    print(f"\n  Best baseline avg WF return: {best_bl_ann:+.0f}%")
    print(f"  Best baseline WF MDD: {best_bl_mdd:.1f}%")

    kelly_better = []
    for desc, wf_res in wf_all.items():
        if 'Kelly' not in desc and 'kelly' not in desc.lower():
            continue
        avg_ann = np.mean([r['ann'] for r in wf_res.values()])
        worst_mdd = min(r['mdd'] for r in wf_res.values())
        if avg_ann > best_bl_ann:
            kelly_better.append((desc, avg_ann, worst_mdd, wf_res))

    if kelly_better:
        kelly_better.sort(key=lambda x: -x[1])
        print(f"\n  Kelly configs beating best baseline avg WF return ({best_bl_ann:+.0f}%):")
        for desc, avg_ann, worst_mdd, wf_res in kelly_better[:10]:
            print(f"    {desc:80s} | AvgWF={avg_ann:>+7.0f}% | WorstWfMDD={worst_mdd:>5.1f}%")
    else:
        print(f"\n  No Kelly configs beat the best baseline avg WF return.")

    # ===================== SECTION: KELLY SIZE DISTRIBUTION =====================
    print("\n" + "=" * 130)
    print("  KELLY SIZE DISTRIBUTION ANALYSIS (best Kelly config, full period)")
    print("=" * 130)

    # Run a modified backtest that records Kelly sizes for the best Kelly config
    # Pick the best Kelly config from full-period results
    best_kelly_configs = [r for r in all_results if 'Kelly' in r.get('desc', '')]
    best_kelly_configs.sort(key=lambda x: -abs(x['ann'] / x['mdd']) if x['mdd'] != 0 else 0)

    if best_kelly_configs:
        best_k = best_kelly_configs[0]
        desc = best_k.get('desc', '')
        print(f"\n  Analyzing Kelly size distribution for: {desc}")
        print(f"  Full period: Ann={best_k['ann']:+.1f}%, MDD={best_k['mdd']:.1f}%")

        # Parse the desc to get params
        sizing_mode = 'kelly_dd' if 'cap' in desc else 'kelly'
        kfrac = 1.0
        if desc.startswith("75%"): kfrac = 0.75
        elif desc.startswith("50%"): kfrac = 0.50
        elif desc.startswith("25%"): kfrac = 0.25
        kwin = 30
        if "w=20" in desc: kwin = 20
        elif "w=50" in desc: kwin = 50

        # Run a quick analysis of Kelly sizes
        # We'll track what sizes Kelly recommends over a full-period run
        kelly_sizes = []
        cash_t = float(CASH0)
        positions_t = []
        trades_t = []

        for di in range(MIN_TRAIN, ND - 1):
            pv = cash_t
            for p in positions_t:
                cp = C[p['si'], di]
                if not np.isnan(cp) and cp > 0:
                    m = MULT.get(p['sym'], DEF_MULT)
                    pv += cp * m * p['lots'] - cp * m * abs(p['lots']) * COMM

            # Close
            cl = []
            for p in positions_t:
                if di - p['entry_di'] >= 1:
                    ep = C[p['si'], di]
                    if np.isnan(ep) or ep <= 0: ep = p['entry_price']
                    m = MULT.get(p['sym'], DEF_MULT)
                    pnl = (ep - p['entry_price']) * m * p['lots']
                    inv = p['entry_price'] * m * abs(p['lots'])
                    pp = pnl / inv * 100 if inv > 0 else 0
                    cash_t += ep * m * abs(p['lots']) * (1 - COMM)
                    trades_t.append(pp)
                    cl.append(p)
            for p in cl: positions_t.remove(p)

            if len(positions_t) >= 2: continue
            edi = di + 1
            if edi >= ND: continue

            ks = kelly_fraction(trades_t, kwin, kfrac, 0.55)
            kelly_sizes.append(ks)

        if kelly_sizes:
            ks_arr = np.array(kelly_sizes)
            print(f"  Kelly size stats over {len(kelly_sizes)} entry points:")
            print(f"    Mean: {np.mean(ks_arr):.3f} ({np.mean(ks_arr)*100:.1f}%)")
            print(f"    Median: {np.median(ks_arr):.3f} ({np.median(ks_arr)*100:.1f}%)")
            print(f"    Std: {np.std(ks_arr):.3f}")
            print(f"    Min: {np.min(ks_arr):.3f} ({np.min(ks_arr)*100:.1f}%)")
            print(f"    Max: {np.max(ks_arr):.3f} ({np.max(ks_arr)*100:.1f}%)")
            # Histogram buckets
            buckets = [0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
            for i in range(len(buckets)-1):
                count = np.sum((ks_arr >= buckets[i]) & (ks_arr < buckets[i+1]))
                if count > 0:
                    print(f"    [{buckets[i]*100:.0f}%-{buckets[i+1]*100:.0f}%): {count:4d} ({count/len(ks_arr)*100:.1f}%)")

    print(f"\n  Total elapsed: {time.time()-t_start:.0f}s")
    print("=" * 130)


if __name__ == '__main__':
    main()
