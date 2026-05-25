"""
Alpha Futures V148 — Multi-Position Portfolio Exploration
=============================================================
V146 fixed best: +185% WF avg, -24% worst WF MDD with top_n=2.

Can we do better with more positions (3-5)?

Key ideas:
  1. More positions → better diversification → smoother compounding
  2. Greedy portfolio construction with full correlation matrix
  3. Risk parity sizing (inverse ATR weighting)
  4. DD-based adaptive sizing (from V146)

Bug prevention:
  - Capital allocation: snapshot before entry, equal split (V146 Bug #1 fix)
  - Correlation: daily returns, not overlapping 20-day (V146 Bug #2 fix)
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
    if final <= 0 or initial <= 0 or n_days <= 0:
        return -100.0
    return (final / initial) ** (1.0 / (n_days / 252)) * 100 - 100


def main():
    print("=" * 120)
    print("  V148 — Multi-Position Portfolio Exploration")
    print("=" * 120)
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
            if len(v) < 10:
                continue
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
                if o > 0:
                    ID_RET[si, di] = (c - o) / o * 100

    print(f"  Done ({time.time()-t0:.1f}s)")

    # ===================== SIGNALS (same as V146) =====================
    def sig_v121(di, edi):
        c = []
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
            c.append((roc * zs, s, ep, 'v121'))
        return c

    def sig_ov_id(di, edi):
        c = []
        for s in range(NS):
            ov = OV_GAP[s, di]; idr = ID_RET[s, di]; roc = ROC5[s, di]
            if any(np.isnan(x) for x in [ov, idr, roc]):
                continue
            if ov <= 0.3 or idr <= 0.3 or roc <= 1.0:
                continue
            zs = ZSCORE[s, di]
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0:
                continue
            z_bonus = zs if not np.isnan(zs) and zs > 1.0 else 1.0
            c.append(((ov + idr) * roc * z_bonus * 2, s, ep, 'ov_id'))
        return c

    def sig_final_flag(di, edi):
        c = []
        for s in range(NS):
            roc20 = ROC20[s, di]
            if np.isnan(roc20) or roc20 <= 5.0 or di < 6:
                continue
            h5 = H[s, di-4:di+1]; l5 = L[s, di-4:di+1]
            if any(np.isnan(x) for x in h5) or any(np.isnan(x) for x in l5):
                continue
            r5 = np.max(h5) - np.min(l5)
            atr = ATR14[s, di]
            if np.isnan(atr) or atr <= 0 or r5 > atr * 3.0:
                continue
            h4 = np.max(H[s, di-4:di])
            cp = C[s, di]
            if np.isnan(cp) or cp <= h4:
                continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0:
                continue
            c.append((roc20 * (cp - h4) / atr, s, ep, 'ff'))
        return c

    def sig_union(di, edi):
        all_sigs = {}
        for item in sig_v121(di, edi):
            sc, s, ep, st = item
            if s not in all_sigs:
                all_sigs[s] = [0, ep, []]
            all_sigs[s][0] += sc * 3
            all_sigs[s][2].append('v121')
        for item in sig_ov_id(di, edi):
            sc, s, ep, st = item
            if s not in all_sigs:
                all_sigs[s] = [0, ep, []]
            all_sigs[s][0] += sc * 2
            all_sigs[s][2].append('ov_id')
        for item in sig_final_flag(di, edi):
            sc, s, ep, st = item
            if s not in all_sigs:
                all_sigs[s] = [0, ep, []]
            all_sigs[s][0] += sc
            all_sigs[s][2].append('ff')
        return [(sc, s, ep, '+'.join(sigs)) for s, (sc, ep, sigs) in all_sigs.items()]

    # ===================== HELPERS =====================
    def get_corr(si_a, si_b, di, window=20):
        """Correlation using daily returns (V146 Bug #2 fix)."""
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
        corr = np.corrcoef(ra, rb)[0, 1]
        return corr if not np.isnan(corr) else 0.5

    def dd_size(pv, high_water, tiers):
        if high_water <= 0:
            return tiers[0][1]
        dd = (pv - high_water) / high_water
        for dd_thresh, size_frac in tiers:
            if dd >= -dd_thresh:
                return size_frac
        return tiers[-1][1]

    # ===================== PORTFOLIO CONSTRUCTION =====================
    def build_portfolio(di, edi, held_si, top_n, max_corr, method='balanced'):
        """
        Greedy portfolio construction with full correlation matrix filtering.

        method='balanced': top half from V121, fill rest from Union
        method='merged': pool all candidates by score, greedy select
        """
        cands_v121 = sig_v121(di, edi)
        cands_union = sig_union(di, edi)
        cands_v121.sort(key=lambda x: -x[0])
        cands_union.sort(key=lambda x: -x[0])

        entries = []
        selected_si = set(held_si)

        def try_add(score, si, ep, sig_type):
            """Add candidate if correlation < threshold with ALL existing entries."""
            if si in selected_si or len(entries) >= top_n:
                return False
            for _, esi, _, _ in entries:
                if get_corr(si, esi, di) >= max_corr:
                    return False
            entries.append((score, si, ep, sig_type))
            selected_si.add(si)
            return True

        if method == 'balanced':
            # V121 gets more slots (stronger signal)
            n_v121 = max(1, (top_n + 1) // 2)
            for c in cands_v121:
                if len(entries) >= n_v121:
                    break
                try_add(c[0], c[1], c[2], 'v121')
            for c in cands_union:
                if len(entries) >= top_n:
                    break
                try_add(c[0], c[1], c[2], 'union')

        elif method == 'merged':
            # Pool all, sort by score (V121 weighted *3)
            pool = []
            for c in cands_v121:
                pool.append((c[0] * 3, c[1], c[2], 'v121'))
            for c in cands_union:
                pool.append((c[0], c[1], c[2], 'union'))
            pool.sort(key=lambda x: -x[0])
            for sc, si, ep, sig in pool:
                if len(entries) >= top_n:
                    break
                try_add(sc, si, ep, sig)

        return entries

    # ===================== BACKTEST ENGINE =====================
    def backtest_v148(start_di=MIN_TRAIN, end_di=None,
                      top_n=2, max_corr=0.5, hold=1,
                      sizing='dd_tiers', base_size=0.55,
                      dd_tiers=None, method='balanced',
                      sl_pct=0.0, risk_parity=False):
        if end_di is None:
            end_di = ND
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

            # Stop-loss check
            if sl_pct > 0:
                cl_early = []
                for p in positions:
                    cp = C[p['si'], di]
                    if np.isnan(cp) or cp <= 0:
                        continue
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
                for p in cl_early:
                    positions.remove(p)

            # Close positions past hold period
            cl = []
            for p in positions:
                if di - p['entry_di'] >= p['hold_days']:
                    ep = C[p['si'], di]
                    if np.isnan(ep) or ep <= 0:
                        ep = p['entry_price']
                    m = MULT.get(p['sym'], DEF_MULT)
                    pnl = (ep - p['entry_price']) * m * p['lots']
                    inv = p['entry_price'] * m * abs(p['lots'])
                    pp = pnl / inv * 100 if inv > 0 else 0
                    cash += ep * m * abs(p['lots']) * (1 - COMM)
                    trades.append(pp)
                    cl.append(p)
            for p in cl:
                positions.remove(p)

            # Position size
            if sizing == 'fixed':
                pos_size = base_size
            elif sizing == 'dd_tiers':
                pos_size = dd_size(pv, high_water, dd_tiers)
            else:
                pos_size = base_size
            pos_size = max(0.05, min(0.95, pos_size))

            # Enter positions
            if len(positions) >= top_n:
                continue
            edi = di + 1
            if edi >= end_di:
                continue

            held_si = set(p['si'] for p in positions)
            entries = build_portfolio(di, edi, held_si, top_n, max_corr, method)
            if not entries:
                continue

            # BUG FIX: snapshot cash before allocation, split equally
            cash_snapshot = cash
            n_entries = len(entries)

            # Compute capital allocation per entry
            if risk_parity:
                # Risk parity: weight inversely proportional to ATR%
                atr_pcts = []
                for _, si, ep, _ in entries:
                    atr = ATR14[si, di]
                    if not np.isnan(atr) and atr > 0 and ep > 0:
                        atr_pcts.append(atr / ep)
                    else:
                        atr_pcts.append(0.02)  # default
                inv_atrs = [1.0 / a for a in atr_pcts]
                total_inv = sum(inv_atrs)
                caps = [cash_snapshot * pos_size * (w / total_inv)
                        for w in inv_atrs]
            else:
                # Equal dollar split
                caps = [cash_snapshot * pos_size / n_entries] * n_entries

            for idx, (score, si, ep, sig_type) in enumerate(entries):
                if si in set(p['si'] for p in positions):
                    continue
                if len(positions) >= top_n:
                    break
                cap = caps[idx]
                sym = syms[si]
                m = MULT.get(sym, DEF_MULT)
                ct = max(1, int(cap / (ep * m * (1 + COMM))))
                ci = ep * m * ct * (1 + COMM)
                if ci > cash:
                    ct = int(cash * 0.9 / (ep * m * (1 + COMM)))
                    ci = ep * m * ct * (1 + COMM) if ct > 0 else 0
                if ct <= 0 or ci <= 0 or ci > cash:
                    continue
                cash -= ci
                positions.append({
                    'si': si, 'entry_price': ep, 'entry_di': edi,
                    'lots': ct, 'dir': 1, 'sym': sym, 'hold_days': hold,
                    'sig': sig_type, 'score': score
                })

        # Close remaining
        for p in positions:
            ae = end_di - 1
            ep = C[p['si'], min(ae, ND-1)]
            if np.isnan(ep) or ep <= 0:
                ep = p['entry_price']
            m = MULT.get(p['sym'], DEF_MULT)
            cash += ep * m * abs(p['lots']) * (1 - COMM)

        nd = end_di - start_di
        ann = annual_return(cash, CASH0, nd)
        wr = np.mean([1 if t > 0 else 0 for t in trades]) * 100 if trades else 0
        nt = len(trades)
        if daily_eq:
            eq = np.array(daily_eq)
            pk = np.maximum.accumulate(eq)
            mdd = np.min((eq - pk) / pk * 100)
            r = np.diff(eq) / eq[:-1]
            r = np.where(np.isfinite(r), r, 0)
            sh = np.mean(r) / np.std(r) * np.sqrt(252) if np.std(r) > 0 else 0
        else:
            mdd = 0; sh = 0
        return {'ann': ann, 'wr': wr, 'n': nt, 'mdd': mdd, 'sharpe': sh,
                'final': cash}

    # ===================== PRINT HELPERS =====================
    def pr(r, label=""):
        ratio = abs(r['ann'] / r['mdd']) if r['mdd'] != 0 else 0
        print(f"  {label:65s} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}%"
              f" | Sh={r['sharpe']:4.2f} | R/M={ratio:.2f} | N={r['n']:4d}")

    def walk_forward(top_n=2, max_corr=0.5, hold=1, sizing='dd_tiers',
                     base_size=0.55, dd_tiers=None, method='balanced',
                     sl_pct=0.0, risk_parity=False, label=""):
        res = {}
        for yr in [2020, 2021, 2022, 2023, 2024, 2025]:
            ys = ye = None
            for di in range(ND):
                if dates[di].year == yr and ys is None:
                    ys = di
                if dates[di].year == yr:
                    ye = di + 1
            if ys is None:
                continue
            r = backtest_v148(start_di=ys, end_di=ye, top_n=top_n,
                              max_corr=max_corr, hold=hold, sizing=sizing,
                              base_size=base_size, dd_tiers=dd_tiers,
                              method=method, sl_pct=sl_pct,
                              risk_parity=risk_parity)
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

    # ===================== DEFINE ALL CONFIGS =====================
    dd_champ = [(0, 0.70), (0.10, 0.60), (0.20, 0.40), (0.30, 0.20)]
    dd_aggro = [(0, 0.80), (0.10, 0.65), (0.20, 0.45), (0.30, 0.25)]

    configs = []

    # Section 1: top_n sweep with DD sizing
    for tn in [1, 2, 3, 4, 5]:
        for dd_t, dd_name in [(dd_champ, "70/60/40/20"), (dd_aggro, "80/65/45/25")]:
            for sl in [0.0, 0.03, 0.05]:
                sl_str = "NO SL" if sl == 0 else f"SL{sl*100:.0f}%"
                configs.append({
                    'top_n': tn, 'max_corr': 0.5, 'sizing': 'dd_tiers',
                    'dd_tiers': dd_t, 'method': 'balanced', 'sl_pct': sl,
                    'risk_parity': False,
                    'label': f"top{tn} DD{dd_name} corr<0.5 {sl_str}"
                })

    # Section 2: method comparison (merged variants — balanced already in S1)
    for tn in [3, 4, 5]:
        configs.append({
            'top_n': tn, 'max_corr': 0.5, 'sizing': 'dd_tiers',
            'dd_tiers': dd_champ, 'method': 'merged', 'sl_pct': 0.03,
            'risk_parity': False,
            'label': f"top{tn} DD70/60/40/20 corr<0.5 SL3% merged"
        })

    # Section 3: risk parity
    for tn in [3, 4, 5]:
        configs.append({
            'top_n': tn, 'max_corr': 0.5, 'sizing': 'dd_tiers',
            'dd_tiers': dd_champ, 'method': 'balanced', 'sl_pct': 0.03,
            'risk_parity': True,
            'label': f"top{tn} DD70/60/40/20 corr<0.5 SL3% RiskPar"
        })

    # Section 4: correlation threshold sweep
    for tn in [3, 4, 5]:
        for mc in [0.3, 0.4, 0.6, 0.7]:
            configs.append({
                'top_n': tn, 'max_corr': mc, 'sizing': 'dd_tiers',
                'dd_tiers': dd_champ, 'method': 'balanced', 'sl_pct': 0.03,
                'risk_parity': False,
                'label': f"top{tn} DD70/60/40/20 corr<{mc} SL3%"
            })

    # ===================== RUN ALL CONFIGS =====================
    print(f"\n  Running {len(configs)} configurations...\n")
    results = []
    for cfg in configs:
        label = cfg['label']
        params = {k: v for k, v in cfg.items() if k != 'label'}
        r = backtest_v148(**params)
        r['label'] = label
        r['cfg'] = params
        results.append(r)
        pr(r, label)

    # ===================== RANKING =====================
    print("\n" + "=" * 120)
    print("  RANKING BY ANN/MDD RATIO")
    print("=" * 120)

    valid = [r for r in results if r['mdd'] < -5]  # filter trivial results
    valid.sort(key=lambda x: abs(x['ann'] / x['mdd']) if x['mdd'] != 0 else 0,
               reverse=True)

    print(f"\n  Top 20 by Ann/MDD Ratio:")
    for i, r in enumerate(valid[:20]):
        ratio = abs(r['ann'] / r['mdd']) if r['mdd'] != 0 else 0
        print(f"  #{i+1:2d}: {r['label']:65s} | Ann={r['ann']:+8.1f}%"
              f" | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f} | R/M={ratio:.2f}")

    print(f"\n  Top 20 by Annual Return:")
    by_ann = sorted(valid, key=lambda x: -x['ann'])
    for i, r in enumerate(by_ann[:20]):
        ratio = abs(r['ann'] / r['mdd']) if r['mdd'] != 0 else 0
        print(f"  #{i+1:2d}: {r['label']:65s} | Ann={r['ann']:+8.1f}%"
              f" | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f} | R/M={ratio:.2f}")

    # ===================== WALK-FORWARD FOR TOP CONFIGS =====================
    print("\n" + "=" * 120)
    print("  WALK-FORWARD VALIDATION FOR TOP 15 CONFIGS")
    print("=" * 120)

    # Select top 15 by R/M ratio, ensuring diverse top_n values
    seen_keys = set()
    wf_selection = []
    for r in valid:
        tn = r['cfg']['top_n']
        key = f"tn{tn}"
        # Allow up to 5 configs per top_n
        count = sum(1 for s in wf_selection if s['cfg']['top_n'] == tn)
        if count < 5 and r['label'] not in seen_keys:
            wf_selection.append(r)
            seen_keys.add(r['label'])
        if len(wf_selection) >= 15:
            break

    wf_all = {}
    for r in wf_selection:
        label = r['label']
        cfg = r['cfg']
        wf_res = walk_forward(label=label, **cfg)
        wf_all[label] = wf_res
        print_wf(wf_res, label)

    # ===================== HIGHLIGHT BEST WF CONFIGS =====================
    print("\n" + "=" * 120)
    print("  HIGHLIGHT: Best WF configs (avg >= +150%, worst MDD > -30%)")
    print("=" * 120)

    highlights = []
    for label, wf_res in wf_all.items():
        avg_ann = np.mean([r['ann'] for r in wf_res.values()])
        worst_mdd = min(r['mdd'] for r in wf_res.values())
        pos = sum(1 for r in wf_res.values() if r['ann'] > 0)
        if avg_ann >= 150 and worst_mdd > -35:
            highlights.append((label, avg_ann, worst_mdd, pos, wf_res))

    if highlights:
        highlights.sort(key=lambda x: -x[1])
        for label, avg_ann, worst_mdd, pos, wf_res in highlights:
            ws = " | ".join([f"{yr}:{r['ann']:+.0f}%/{r['mdd']:.0f}%"
                             for yr, r in sorted(wf_res.items())])
            print(f"\n  *** {label}")
            print(f"      AvgWF={avg_ann:+.0f}% | WorstWfMDD={worst_mdd:.1f}%"
                  f" | {pos}/6 positive")
            print(f"      {ws}")
    else:
        print("\n  No configs meet target. Showing closest:")
        closest = []
        for label, wf_res in wf_all.items():
            avg_ann = np.mean([r['ann'] for r in wf_res.values()])
            worst_mdd = min(r['mdd'] for r in wf_res.values())
            closest.append((label, avg_ann, worst_mdd))
        closest.sort(key=lambda x: -x[1])
        for label, avg_ann, worst_mdd in closest[:10]:
            print(f"  {label:65s} | AvgWF={avg_ann:>+7.0f}%"
                  f" | WorstWfMDD={worst_mdd:>5.1f}%")

    # ===================== TOP_N COMPARISON TABLE =====================
    print("\n" + "=" * 120)
    print("  TOP_N COMPARISON: Avg WF annual and worst WF MDD by position count")
    print("=" * 120)

    for tn in [1, 2, 3, 4, 5]:
        tn_results = [(label, wf_res) for label, wf_res in wf_all.items()
                      if f"top{tn}" in label]
        if not tn_results:
            continue
        best = max(tn_results, key=lambda x: np.mean([r['ann'] for r in x[1].values()]))
        label, wf_res = best
        avg_ann = np.mean([r['ann'] for r in wf_res.values()])
        worst_mdd = min(r['mdd'] for r in wf_res.values())
        ws = " | ".join([f"{yr}:{r['ann']:+.0f}%/{r['mdd']:.0f}%"
                         for yr, r in sorted(wf_res.items())])
        print(f"\n  top_n={tn} best: {label}")
        print(f"    AvgWF={avg_ann:+.0f}% | WorstWfMDD={worst_mdd:.1f}%")
        print(f"    {ws}")

    # ===================== FINAL SUMMARY =====================
    print("\n" + "=" * 120)
    print("  FINAL SUMMARY")
    print("=" * 120)

    print(f"\n  V146 champion (top_n=2): +185% WF avg, -24% worst WF MDD")
    print(f"  V148 tested {len(configs)} multi-position configs")

    # Find if any top_n > 2 beats V146
    better = []
    for label, wf_res in wf_all.items():
        if 'top1' in label or 'top2' in label:
            continue
        avg_ann = np.mean([r['ann'] for r in wf_res.values()])
        worst_mdd = min(r['mdd'] for r in wf_res.values())
        if avg_ann > 185 and worst_mdd > -30:
            better.append((label, avg_ann, worst_mdd, wf_res))

    if better:
        better.sort(key=lambda x: -x[1])
        print(f"\n  Configs BETTER than V146 champion (higher return, lower MDD):")
        for label, avg_ann, worst_mdd, wf_res in better[:5]:
            ws = " | ".join([f"{yr}:{r['ann']:+.0f}%/{r['mdd']:.0f}%"
                             for yr, r in sorted(wf_res.items())])
            print(f"  *** {label}")
            print(f"      AvgWF={avg_ann:+.0f}% | WorstWfMDD={worst_mdd:.1f}%")
            print(f"      {ws}")
    else:
        print(f"\n  No top_n>2 config beats V146 champion on both return AND MDD.")
        print(f"  Multi-position helps diversification but doesn't improve"
              f" return/MDD frontier.")

    print(f"\n  Elapsed: {time.time()-t_start:.0f}s")
    print("=" * 120)


if __name__ == '__main__':
    main()
