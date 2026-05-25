"""
Alpha Futures V145 — REGIME-AWARE POSITION SIZING
=============================================================================
Problem: V137 regime FILTERS destroy returns by killing trade count.
         V140 circuit breakers also destroy returns when used as hard stops.
         Only position sizing works for MDD control.

New approach: Use regime indicators to ADJUST position SIZE, not filter trades.
  Every signal still gets traded, but size varies from 25% to 80% of capital
  depending on market conditions.

Baseline: Union/V121 50/50 @50% sizing = +155% annual, -24% MDD worst WF

Regime indicators:
  A. Market Breadth: fraction of commodities with positive 5-day ROC
  B. Volatility Regime: 20-day rolling std of equal-weighted market return
  C. Strategy Performance: rolling 20-day equity curve slope
  D. Drawdown-Based: gentle DD-based sizing (not hard stop)
  E. Combined: average normalized scores from A-D -> composite 0-1

Test on:
  1. Union signal at various base sizes
  2. Union/V121 50/50 portfolio
  3. Annual return, full MDD, per-year WF (2020-2025) with per-year MDD
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
    print("  V145 — REGIME-AWARE POSITION SIZING")
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

    # ===================== REGIME INDICATORS =====================
    print("  Computing regime indicators...", flush=True)

    # A. Market Breadth: fraction of commodities with positive 5-day ROC
    BREADTH = np.full(ND, np.nan)
    for di in range(5, ND):
        pos_count = 0; total = 0
        for si in range(NS):
            r = ROC5[si, di]
            if not np.isnan(r):
                total += 1
                if r > 0: pos_count += 1
        if total > 0:
            BREADTH[di] = pos_count / total  # 0..1

    # B. Volatility Regime: 20-day rolling std of equal-weighted market return
    # First compute equal-weighted market return per day
    MKT_RET = np.full(ND, np.nan)
    for di in range(ND):
        rets_day = RET[:, di]
        valid = rets_day[~np.isnan(rets_day)]
        if len(valid) > 10:
            MKT_RET[di] = np.mean(valid)

    # Rolling 20-day std of market return
    MKT_VOL = np.full(ND, np.nan)
    for di in range(20, ND):
        window = MKT_RET[di-20:di]
        valid = window[~np.isnan(window)]
        if len(valid) >= 10:
            MKT_VOL[di] = np.std(valid, ddof=1)

    # Compute median vol for normalization
    valid_vols = MKT_VOL[~np.isnan(MKT_VOL)]
    if len(valid_vols) > 0:
        VOL_MEDIAN = np.median(valid_vols)
    else:
        VOL_MEDIAN = 1.0
    print(f"  Market vol median: {VOL_MEDIAN:.4f}%")

    # C. Strategy Performance: rolling 20-day equity slope
    #    We'll compute this INSIDE the backtest from equity curve

    # D. Drawdown-Based: computed inside backtest from equity curve

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

    # ===================== REGIME-AWARE BACKTEST ENGINE =====================
    def backtest_regime(signal_func, hold=1, top_n=1, start_di=MIN_TRAIN, end_di=None,
                        base_size=0.50,          # Base position size fraction
                        regime_mode='none',      # 'none','breadth','vol','perf','dd','combined'
                        # Breadth thresholds
                        breadth_high=0.70, breadth_low=0.40,
                        size_breadth_high=0.80, size_breadth_mid=0.55, size_breadth_low=0.25,
                        # Vol thresholds
                        vol_low_ratio=0.8, vol_high_ratio=1.5,
                        size_vol_low=0.75, size_vol_mid=0.55, size_vol_high=0.25,
                        # Perf thresholds
                        perf_window=20,
                        size_perf_high=0.80, size_perf_mid=0.55, size_perf_low=0.25,
                        # DD thresholds
                        size_dd_peak=0.70, size_dd_small=0.60, size_dd_med=0.45,
                        size_dd_large=0.25, size_dd_extreme=0.10,
                        dd_small=0.10, dd_med=0.20, dd_large=0.30,
                        # Combined thresholds
                        size_combo_high=0.80, size_combo_mid=0.55, size_combo_low=0.25,
                        combo_high=0.7, combo_low=0.4):
        if end_di is None: end_di = ND
        cash = float(CASH0)
        positions = []; trades = []; daily_eq = []
        high_water = float(CASH0)

        for di in range(start_di, end_di - 1):
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
                    trades.append({'pnl_pct': pp, 'sig': p.get('sig', ''), 'di': p['entry_di']})
                    cl.append(p)
            for p in cl: positions.remove(p)

            # --- Compute regime-based position size ---
            if regime_mode == 'none':
                size = base_size

            elif regime_mode == 'breadth':
                bth = BREADTH[di]
                if np.isnan(bth):
                    size = base_size
                elif bth > breadth_high:
                    size = size_breadth_high
                elif bth > breadth_low:
                    size = size_breadth_mid
                else:
                    size = size_breadth_low

            elif regime_mode == 'vol':
                vol = MKT_VOL[di]
                if np.isnan(vol) or VOL_MEDIAN <= 0:
                    size = base_size
                elif vol < VOL_MEDIAN * vol_low_ratio:
                    size = size_vol_low
                elif vol > VOL_MEDIAN * vol_high_ratio:
                    size = size_vol_high
                else:
                    size = size_vol_mid

            elif regime_mode == 'perf':
                # Rolling equity curve slope
                if len(daily_eq) < perf_window:
                    size = base_size
                else:
                    eq_window = np.array(daily_eq[-perf_window:])
                    # Linear regression slope
                    x = np.arange(perf_window)
                    try:
                        slope = np.polyfit(x, eq_window, 1)[0]
                        # Normalize by mean equity
                        eq_mean = np.mean(eq_window)
                        if eq_mean > 0:
                            norm_slope = slope / eq_mean * 100  # % per day
                        else:
                            norm_slope = 0
                        # Compute std of daily returns for normalization
                        eq_rets = np.diff(eq_window) / eq_window[:-1] * 100
                        eq_rets = eq_rets[np.isfinite(eq_rets)]
                        eq_std = np.std(eq_rets) if len(eq_rets) > 5 else 1.0

                        if eq_std > 0:
                            z_slope = norm_slope / eq_std
                        else:
                            z_slope = 0

                        if z_slope > 1.0:
                            size = size_perf_high
                        elif z_slope < 0:
                            size = size_perf_low
                        else:
                            size = size_perf_mid
                    except Exception:
                        size = base_size

            elif regime_mode == 'dd':
                cur_dd = (pv - high_water) / high_water if high_water > 0 else 0
                if cur_dd >= 0:
                    size = size_dd_peak  # New high
                elif cur_dd > -dd_small:
                    size = size_dd_small
                elif cur_dd > -dd_med:
                    size = size_dd_med
                elif cur_dd > -dd_large:
                    size = size_dd_large
                else:
                    size = size_dd_extreme

            elif regime_mode == 'combined':
                # Average normalized scores from A, B, C, D -> composite 0..1
                scores = []

                # A. Breadth -> 0..1 (0.4 to 0.7 mapped to 0..1)
                bth = BREADTH[di]
                if not np.isnan(bth):
                    s_breadth = np.clip((bth - combo_low) / (combo_high - combo_low), 0, 1)
                    scores.append(s_breadth)

                # B. Vol -> 0..1 (low vol = 1, high vol = 0)
                vol = MKT_VOL[di]
                if not np.isnan(vol) and VOL_MEDIAN > 0:
                    vol_ratio = vol / VOL_MEDIAN
                    # vol_ratio < 0.8 -> 1.0, vol_ratio > 1.5 -> 0.0
                    s_vol = np.clip((vol_high_ratio - vol_ratio) / (vol_high_ratio - vol_low_ratio), 0, 1)
                    scores.append(s_vol)

                # C. Performance -> 0..1
                if len(daily_eq) >= perf_window:
                    eq_window = np.array(daily_eq[-perf_window:])
                    x = np.arange(perf_window)
                    try:
                        slope = np.polyfit(x, eq_window, 1)[0]
                        eq_mean = np.mean(eq_window)
                        if eq_mean > 0:
                            norm_slope = slope / eq_mean * 100
                        else:
                            norm_slope = 0
                        eq_rets = np.diff(eq_window) / eq_window[:-1] * 100
                        eq_rets = eq_rets[np.isfinite(eq_rets)]
                        eq_std = np.std(eq_rets) if len(eq_rets) > 5 else 1.0
                        z_slope = norm_slope / eq_std if eq_std > 0 else 0
                        # z_slope > 1 -> 1.0, z_slope < -1 -> 0.0
                        s_perf = np.clip((z_slope + 1.0) / 2.0, 0, 1)
                        scores.append(s_perf)
                    except Exception:
                        pass

                # D. DD -> 0..1 (new high = 1, deep DD = 0)
                cur_dd = (pv - high_water) / high_water if high_water > 0 else 0
                s_dd = np.clip(1.0 + cur_dd / 0.3, 0, 1)  # 0 at -30% DD
                scores.append(s_dd)

                if len(scores) == 0:
                    size = base_size
                else:
                    composite = np.mean(scores)
                    if composite > combo_high:
                        size = size_combo_high
                    elif composite > combo_low:
                        size = size_combo_mid
                    else:
                        size = size_combo_low

            else:
                size = base_size

            # --- Enter positions ---
            if len(positions) >= top_n: continue
            edi = di + 1
            if edi >= end_di: continue
            cands = signal_func(di, edi)
            if not cands: continue
            cands.sort(key=lambda x: -x[0])
            ns = top_n - len(positions)
            cap = cash * size / max(1, ns)

            for item in cands[:ns]:
                if len(item) == 3: sc, s, pr = item; sig = ''
                else: sc, s, pr, sig = item
                sym = syms[s]; m = MULT.get(sym, DEF_MULT)
                ct = max(1, int(cap / (pr * m * (1 + COMM))))
                ci = pr * m * ct * (1 + COMM)
                if ci > cash:
                    ct = int(cash * 0.9 / (pr * m * (1 + COMM)))
                    ci = pr * m * ct * (1 + COMM) if ct > 0 else 0
                if ct <= 0 or ci <= 0 or ci > cash: continue
                cash -= ci
                positions.append({'si': s, 'entry_price': pr, 'entry_di': edi,
                                  'lots': ct, 'dir': 1, 'sym': sym, 'hold_days': hold, 'sig': sig})

        # Close remaining
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
            r = np.where(np.isfinite(r), r, 0)
            sh = np.mean(r) / np.std(r) * np.sqrt(252) if np.std(r) > 0 else 0
        else: mdd = 0; sh = 0

        # Avg position size
        size_info = ""
        if trades:
            avg_size = np.mean([t.get('size_used', base_size) for t in trades]) if 'size_used' in trades[0] else base_size

        return {'ann': ann, 'wr': wr, 'n': nt, 'avg_pnl': ap, 'mdd': mdd, 'sharpe': sh,
                'final': cash, 'daily_eq': daily_eq}

    def pr(r, label=""):
        print(f"  {label:70s} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f} | N={r['n']:4d}")

    def walk_forward(signal_func, regime_mode='none', base_size=0.50, hold=1, topn=1, **kwargs):
        """Run per-year walk-forward with per-year MDD."""
        res = {}
        for yr in [2020, 2021, 2022, 2023, 2024, 2025]:
            ys = ye = None
            for di in range(ND):
                if dates[di].year == yr and ys is None: ys = di
                if dates[di].year == yr: ye = di + 1
            if ys is None: continue
            r = backtest_regime(signal_func, hold=hold, top_n=topn, start_di=ys, end_di=ye,
                                base_size=base_size, regime_mode=regime_mode, **kwargs)
            res[yr] = {'ann': r['ann'], 'mdd': r['mdd'], 'n': r['n']}
        return res

    def print_wf(wf, label=""):
        pos = sum(1 for v in wf.values() if v['ann'] > 0)
        avg_ann = np.mean([v['ann'] for v in wf.values()])
        worst_mdd = min(v['mdd'] for v in wf.values())
        ws = " | ".join([f"{yr}:{v['ann']:+.0f}%/{v['mdd']:.0f}%"
                         for yr, v in sorted(wf.items())])
        print(f"    {label:40s} {pos}/6 | Avg={avg_ann:>+7.0f}% | WfMDD={worst_mdd:>5.0f}% | {ws}")

    # ===================== SECTION 0: BASELINES =====================
    print("\n" + "=" * 130)
    print("  SECTION 0: BASELINES (Fixed Position Sizing)")
    print("=" * 130)

    # Fixed 50% sizing baseline
    r_union_50 = backtest_regime(sig_union, regime_mode='none', base_size=0.50)
    pr(r_union_50, "Union fixed 50% (baseline)")
    r_union_55 = backtest_regime(sig_union, regime_mode='none', base_size=0.55)
    pr(r_union_55, "Union fixed 55%")
    r_union_60 = backtest_regime(sig_union, regime_mode='none', base_size=0.60)
    pr(r_union_60, "Union fixed 60%")
    r_union_40 = backtest_regime(sig_union, regime_mode='none', base_size=0.40)
    pr(r_union_40, "Union fixed 40%")

    r_v121_50 = backtest_regime(sig_v121, regime_mode='none', base_size=0.50)
    pr(r_v121_50, "V121 fixed 50%")

    # ===================== SECTION 1: MARKET BREADTH REGIME =====================
    print("\n" + "=" * 130)
    print("  SECTION 1: MARKET BREADTH REGIME (fraction of commodities with positive 5d ROC)")
    print("=" * 130)

    breadth_configs = [
        # (breadth_high, breadth_low, size_high, size_mid, size_low, label)
        (0.70, 0.40, 0.80, 0.55, 0.25, "Breadth: 70/40 -> 80/55/25%"),
        (0.70, 0.40, 0.75, 0.50, 0.30, "Breadth: 70/40 -> 75/50/30%"),
        (0.75, 0.35, 0.80, 0.55, 0.25, "Breadth: 75/35 -> 80/55/25%"),
        (0.75, 0.35, 0.75, 0.55, 0.30, "Breadth: 75/35 -> 75/55/30%"),
        (0.65, 0.45, 0.80, 0.55, 0.25, "Breadth: 65/45 -> 80/55/25%"),
        (0.65, 0.45, 0.70, 0.55, 0.35, "Breadth: 65/45 -> 70/55/35%"),
        (0.80, 0.30, 0.80, 0.55, 0.25, "Breadth: 80/30 -> 80/55/25%"),
        (0.70, 0.40, 0.85, 0.55, 0.20, "Breadth: 70/40 -> 85/55/20%"),
    ]

    breadth_results = []
    print(f"\n  --- Union Signal ---")
    for bh, bl, sh, sm, sl, label in breadth_configs:
        r = backtest_regime(sig_union, regime_mode='breadth', base_size=0.55,
                            breadth_high=bh, breadth_low=bl,
                            size_breadth_high=sh, size_breadth_mid=sm, size_breadth_low=sl)
        r['desc'] = label
        breadth_results.append(r)
        pr(r, f"Union {label}")

    print(f"\n  --- V121 Signal ---")
    for bh, bl, sh, sm, sl, label in breadth_configs[:4]:
        r = backtest_regime(sig_v121, regime_mode='breadth', base_size=0.55,
                            breadth_high=bh, breadth_low=bl,
                            size_breadth_high=sh, size_breadth_mid=sm, size_breadth_low=sl)
        r['desc'] = label
        breadth_results.append(r)
        pr(r, f"V121 {label}")

    # ===================== SECTION 2: VOLATILITY REGIME =====================
    print("\n" + "=" * 130)
    print("  SECTION 2: VOLATILITY REGIME (20-day rolling std of market return)")
    print("=" * 130)

    vol_configs = [
        # (vol_low_ratio, vol_high_ratio, size_low, size_mid, size_high, label)
        (0.8, 1.5, 0.75, 0.55, 0.25, "Vol: 0.8x/1.5x -> 75/55/25%"),
        (0.8, 1.5, 0.70, 0.55, 0.30, "Vol: 0.8x/1.5x -> 70/55/30%"),
        (0.7, 1.3, 0.75, 0.55, 0.25, "Vol: 0.7x/1.3x -> 75/55/25%"),
        (0.9, 1.5, 0.75, 0.55, 0.25, "Vol: 0.9x/1.5x -> 75/55/25%"),
        (0.8, 1.3, 0.75, 0.55, 0.25, "Vol: 0.8x/1.3x -> 75/55/25%"),
        (0.8, 1.5, 0.80, 0.50, 0.20, "Vol: 0.8x/1.5x -> 80/50/20%"),
        (0.8, 2.0, 0.75, 0.55, 0.30, "Vol: 0.8x/2.0x -> 75/55/30%"),
    ]

    vol_results = []
    print(f"\n  --- Union Signal ---")
    for vlr, vhr, sl, sm, sh, label in vol_configs:
        r = backtest_regime(sig_union, regime_mode='vol', base_size=0.55,
                            vol_low_ratio=vlr, vol_high_ratio=vhr,
                            size_vol_low=sl, size_vol_mid=sm, size_vol_high=sh)
        r['desc'] = label
        vol_results.append(r)
        pr(r, f"Union {label}")

    print(f"\n  --- V121 Signal ---")
    for vlr, vhr, sl, sm, sh, label in vol_configs[:4]:
        r = backtest_regime(sig_v121, regime_mode='vol', base_size=0.55,
                            vol_low_ratio=vlr, vol_high_ratio=vhr,
                            size_vol_low=sl, size_vol_mid=sm, size_vol_high=sh)
        r['desc'] = label
        vol_results.append(r)
        pr(r, f"V121 {label}")

    # ===================== SECTION 3: STRATEGY PERFORMANCE REGIME =====================
    print("\n" + "=" * 130)
    print("  SECTION 3: STRATEGY PERFORMANCE REGIME (rolling 20-day equity slope)")
    print("=" * 130)

    perf_configs = [
        # (perf_window, size_high, size_mid, size_low, label)
        (20, 0.80, 0.55, 0.25, "Perf: 20d -> 80/55/25%"),
        (20, 0.75, 0.55, 0.30, "Perf: 20d -> 75/55/30%"),
        (20, 0.70, 0.55, 0.35, "Perf: 20d -> 70/55/35%"),
        (15, 0.80, 0.55, 0.25, "Perf: 15d -> 80/55/25%"),
        (10, 0.80, 0.55, 0.25, "Perf: 10d -> 80/55/25%"),
        (30, 0.80, 0.55, 0.25, "Perf: 30d -> 80/55/25%"),
        (20, 0.85, 0.55, 0.20, "Perf: 20d -> 85/55/20%"),
    ]

    perf_results = []
    print(f"\n  --- Union Signal ---")
    for pw, sh, sm, sl, label in perf_configs:
        r = backtest_regime(sig_union, regime_mode='perf', base_size=0.55,
                            perf_window=pw,
                            size_perf_high=sh, size_perf_mid=sm, size_perf_low=sl)
        r['desc'] = label
        perf_results.append(r)
        pr(r, f"Union {label}")

    print(f"\n  --- V121 Signal ---")
    for pw, sh, sm, sl, label in perf_configs[:4]:
        r = backtest_regime(sig_v121, regime_mode='perf', base_size=0.55,
                            perf_window=pw,
                            size_perf_high=sh, size_perf_mid=sm, size_perf_low=sl)
        r['desc'] = label
        perf_results.append(r)
        pr(r, f"V121 {label}")

    # ===================== SECTION 4: DRAWDOWN-BASED REGIME =====================
    print("\n" + "=" * 130)
    print("  SECTION 4: DRAWDOWN-BASED REGIME (gentle DD-based sizing)")
    print("=" * 130)

    dd_configs = [
        # Standard as specified
        (0.70, 0.60, 0.45, 0.25, 0.10, 0.10, 0.20, 0.30, "DD: peak/10/20/30 -> 70/60/45/25/10%"),
        # Gentler versions
        (0.65, 0.55, 0.45, 0.35, 0.20, 0.10, 0.20, 0.30, "DD: peak/10/20/30 -> 65/55/45/35/20%"),
        (0.75, 0.65, 0.50, 0.35, 0.20, 0.10, 0.20, 0.30, "DD: peak/10/20/30 -> 75/65/50/35/20%"),
        # More aggressive DD reduction
        (0.80, 0.65, 0.45, 0.25, 0.10, 0.10, 0.20, 0.30, "DD: peak/10/20/30 -> 80/65/45/25/10%"),
        # Smoother transition
        (0.70, 0.60, 0.50, 0.40, 0.30, 0.10, 0.20, 0.30, "DD: peak/10/20/30 -> 70/60/50/40/30%"),
        # Less extreme floor
        (0.70, 0.60, 0.50, 0.35, 0.25, 0.10, 0.15, 0.25, "DD: peak/10/15/25 -> 70/60/50/35/25%"),
    ]

    dd_results = []
    print(f"\n  --- Union Signal ---")
    for sp, ss, sm, sl, se, ds, dm, dl, label in dd_configs:
        r = backtest_regime(sig_union, regime_mode='dd', base_size=0.55,
                            size_dd_peak=sp, size_dd_small=ss, size_dd_med=sm,
                            size_dd_large=sl, size_dd_extreme=se,
                            dd_small=ds, dd_med=dm, dd_large=dl)
        r['desc'] = label
        dd_results.append(r)
        pr(r, f"Union {label}")

    print(f"\n  --- V121 Signal ---")
    for sp, ss, sm, sl, se, ds, dm, dl, label in dd_configs[:4]:
        r = backtest_regime(sig_v121, regime_mode='dd', base_size=0.55,
                            size_dd_peak=sp, size_dd_small=ss, size_dd_med=sm,
                            size_dd_large=sl, size_dd_extreme=se,
                            dd_small=ds, dd_med=dm, dd_large=dl)
        r['desc'] = label
        dd_results.append(r)
        pr(r, f"V121 {label}")

    # ===================== SECTION 5: COMBINED REGIME SCORE =====================
    print("\n" + "=" * 130)
    print("  SECTION 5: COMBINED REGIME SCORE (average of A+B+C+D normalized)")
    print("=" * 130)

    combo_configs = [
        # (combo_high, combo_low, size_high, size_mid, size_low, perf_window, label)
        (0.70, 0.40, 0.80, 0.55, 0.25, 20, "Combo: 0.7/0.4 -> 80/55/25%"),
        (0.70, 0.40, 0.75, 0.55, 0.30, 20, "Combo: 0.7/0.4 -> 75/55/30%"),
        (0.65, 0.35, 0.80, 0.55, 0.25, 20, "Combo: 0.65/0.35 -> 80/55/25%"),
        (0.75, 0.45, 0.80, 0.55, 0.25, 20, "Combo: 0.75/0.45 -> 80/55/25%"),
        (0.70, 0.40, 0.85, 0.55, 0.20, 20, "Combo: 0.7/0.4 -> 85/55/20%"),
        (0.70, 0.40, 0.70, 0.55, 0.35, 20, "Combo: 0.7/0.4 -> 70/55/35%"),
        (0.70, 0.40, 0.80, 0.55, 0.25, 15, "Combo: 0.7/0.4 15d -> 80/55/25%"),
        (0.70, 0.40, 0.80, 0.55, 0.25, 30, "Combo: 0.7/0.4 30d -> 80/55/25%"),
    ]

    combo_results = []
    print(f"\n  --- Union Signal ---")
    for ch, cl, sh, sm, sl, pw, label in combo_configs:
        r = backtest_regime(sig_union, regime_mode='combined', base_size=0.55,
                            combo_high=ch, combo_low=cl,
                            size_combo_high=sh, size_combo_mid=sm, size_combo_low=sl,
                            perf_window=pw)
        r['desc'] = label
        combo_results.append(r)
        pr(r, f"Union {label}")

    print(f"\n  --- V121 Signal ---")
    for ch, cl, sh, sm, sl, pw, label in combo_configs[:4]:
        r = backtest_regime(sig_v121, regime_mode='combined', base_size=0.55,
                            combo_high=ch, combo_low=cl,
                            size_combo_high=sh, size_combo_mid=sm, size_combo_low=sl,
                            perf_window=pw)
        r['desc'] = label
        combo_results.append(r)
        pr(r, f"V121 {label}")

    # ===================== SECTION 6: 50/50 PORTFOLIO WITH REGIME SIZING =====================
    print("\n" + "=" * 130)
    print("  SECTION 6: UNION/V121 50/50 PORTFOLIO WITH REGIME-AWARE SIZING")
    print("=" * 130)

    def backtest_portfolio_regime(regime_mode='none', base_size=0.50, **kwargs):
        """Run 50/50 portfolio of Union + V121, each with independent regime sizing."""
        def run_sub(sig_func):
            cash = float(CASH0); positions = []; daily_eq = []
            high_water = float(CASH0)

            for di in range(MIN_TRAIN, ND - 1):
                pv = cash
                for p in positions:
                    cp = C[p['si'], di]
                    if not np.isnan(cp) and cp > 0:
                        m = MULT.get(p['sym'], DEF_MULT)
                        pv += cp * m * p['lots'] - cp * m * abs(p['lots']) * COMM
                daily_eq.append(pv)
                if pv > high_water: high_water = pv

                # Close positions
                cl = []
                for p in positions:
                    if di - p['entry_di'] >= p['hold_days']:
                        ep = C[p['si'], di]
                        if np.isnan(ep) or ep <= 0: ep = p['entry_price']
                        m = MULT.get(p['sym'], DEF_MULT)
                        cash += ep * m * abs(p['lots']) * (1 - COMM)
                        cl.append(p)
                for p in cl: positions.remove(p)

                # Compute regime size (same logic as backtest_regime)
                if regime_mode == 'none':
                    size = base_size
                elif regime_mode == 'breadth':
                    bth = BREADTH[di]
                    if np.isnan(bth): size = base_size
                    elif bth > kwargs.get('breadth_high', 0.70): size = kwargs.get('size_breadth_high', 0.80)
                    elif bth > kwargs.get('breadth_low', 0.40): size = kwargs.get('size_breadth_mid', 0.55)
                    else: size = kwargs.get('size_breadth_low', 0.25)
                elif regime_mode == 'vol':
                    vol = MKT_VOL[di]
                    if np.isnan(vol) or VOL_MEDIAN <= 0: size = base_size
                    elif vol < VOL_MEDIAN * kwargs.get('vol_low_ratio', 0.8): size = kwargs.get('size_vol_low', 0.75)
                    elif vol > VOL_MEDIAN * kwargs.get('vol_high_ratio', 1.5): size = kwargs.get('size_vol_high', 0.25)
                    else: size = kwargs.get('size_vol_mid', 0.55)
                elif regime_mode == 'perf':
                    pw = kwargs.get('perf_window', 20)
                    if len(daily_eq) < pw: size = base_size
                    else:
                        eq_window = np.array(daily_eq[-pw:])
                        x = np.arange(pw)
                        try:
                            slope = np.polyfit(x, eq_window, 1)[0]
                            eq_mean = np.mean(eq_window)
                            norm_slope = slope / eq_mean * 100 if eq_mean > 0 else 0
                            eq_rets = np.diff(eq_window) / eq_window[:-1] * 100
                            eq_rets = eq_rets[np.isfinite(eq_rets)]
                            eq_std = np.std(eq_rets) if len(eq_rets) > 5 else 1.0
                            z_slope = norm_slope / eq_std if eq_std > 0 else 0
                            if z_slope > 1.0: size = kwargs.get('size_perf_high', 0.80)
                            elif z_slope < 0: size = kwargs.get('size_perf_low', 0.25)
                            else: size = kwargs.get('size_perf_mid', 0.55)
                        except Exception: size = base_size
                elif regime_mode == 'dd':
                    cur_dd = (pv - high_water) / high_water if high_water > 0 else 0
                    ds = kwargs.get('dd_small', 0.10)
                    dm = kwargs.get('dd_med', 0.20)
                    dl = kwargs.get('dd_large', 0.30)
                    if cur_dd >= 0: size = kwargs.get('size_dd_peak', 0.70)
                    elif cur_dd > -ds: size = kwargs.get('size_dd_small', 0.60)
                    elif cur_dd > -dm: size = kwargs.get('size_dd_med', 0.45)
                    elif cur_dd > -dl: size = kwargs.get('size_dd_large', 0.25)
                    else: size = kwargs.get('size_dd_extreme', 0.10)
                elif regime_mode == 'combined':
                    scores = []
                    bth = BREADTH[di]
                    if not np.isnan(bth):
                        ch = kwargs.get('combo_high', 0.7)
                        cl = kwargs.get('combo_low', 0.4)
                        scores.append(np.clip((bth - cl) / (ch - cl), 0, 1))
                    vol = MKT_VOL[di]
                    if not np.isnan(vol) and VOL_MEDIAN > 0:
                        vr = vol / VOL_MEDIAN
                        vhr = kwargs.get('vol_high_ratio', 1.5)
                        vlr = kwargs.get('vol_low_ratio', 0.8)
                        scores.append(np.clip((vhr - vr) / (vhr - vlr), 0, 1))
                    pw = kwargs.get('perf_window', 20)
                    if len(daily_eq) >= pw:
                        eq_window = np.array(daily_eq[-pw:])
                        x = np.arange(pw)
                        try:
                            slope = np.polyfit(x, eq_window, 1)[0]
                            eq_mean = np.mean(eq_window)
                            norm_slope = slope / eq_mean * 100 if eq_mean > 0 else 0
                            eq_rets = np.diff(eq_window) / eq_window[:-1] * 100
                            eq_rets = eq_rets[np.isfinite(eq_rets)]
                            eq_std = np.std(eq_rets) if len(eq_rets) > 5 else 1.0
                            z = norm_slope / eq_std if eq_std > 0 else 0
                            scores.append(np.clip((z + 1.0) / 2.0, 0, 1))
                        except Exception: pass
                    cur_dd = (pv - high_water) / high_water if high_water > 0 else 0
                    scores.append(np.clip(1.0 + cur_dd / 0.3, 0, 1))
                    if not scores: size = base_size
                    else:
                        composite = np.mean(scores)
                        if composite > kwargs.get('combo_high', 0.7): size = kwargs.get('size_combo_high', 0.80)
                        elif composite > kwargs.get('combo_low', 0.4): size = kwargs.get('size_combo_mid', 0.55)
                        else: size = kwargs.get('size_combo_low', 0.25)
                else:
                    size = base_size

                # Enter positions
                if len(positions) >= 1: continue
                edi = di + 1
                if edi >= ND: continue
                cands = sig_func(di, edi)
                if not cands: continue
                cands.sort(key=lambda x: -x[0])
                item = cands[0]
                if len(item) == 3: sc, s, pr = item; sig = ''
                else: sc, s, pr, sig = item
                sym = syms[s]; m = MULT.get(sym, DEF_MULT)
                cap = cash * size
                ct = max(1, int(cap / (pr * m * (1 + COMM))))
                ci = pr * m * ct * (1 + COMM)
                if ci > cash:
                    ct = int(cash * 0.9 / (pr * m * (1 + COMM)))
                    ci = pr * m * ct * (1 + COMM) if ct > 0 else 0
                if ct <= 0 or ci <= 0 or ci > cash: continue
                cash -= ci
                positions.append({'si': s, 'entry_price': pr, 'entry_di': edi,
                                  'lots': ct, 'dir': 1, 'sym': sym, 'hold_days': 1, 'sig': sig})

            for p in positions:
                ep = C[p['si'], min(ND-1, ND-1)]
                if np.isnan(ep) or ep <= 0: ep = p['entry_price']
                m = MULT.get(p['sym'], DEF_MULT)
                cash += ep * m * abs(p['lots']) * (1 - COMM)
            return np.array(daily_eq), cash

        eq_A, final_A = run_sub(sig_union)
        eq_B, final_B = run_sub(sig_v121)

        ml = min(len(eq_A), len(eq_B))
        if ml <= 1:
            return {'ann': -100.0, 'mdd': 0, 'sharpe': 0, 'final': CASH0}

        ret_A = np.diff(eq_A[:ml]) / eq_A[:ml-1]
        ret_B = np.diff(eq_B[:ml]) / eq_B[:ml-1]
        ret_A = np.where(np.isfinite(ret_A), ret_A, 0)
        ret_B = np.where(np.isfinite(ret_B), ret_B, 0)

        combined = 0.5 * ret_A + 0.5 * ret_B
        eq = np.zeros(ml)
        eq[0] = float(CASH0)
        for i in range(ml - 1):
            eq[i+1] = eq[i] * (1 + combined[i])

        final = eq[-1]
        nd = ml
        ann = annual_return(final, CASH0, nd)
        pk = np.maximum.accumulate(eq)
        mdd = np.min((eq - pk) / pk * 100)
        sh = np.mean(combined) / np.std(combined) * np.sqrt(252) if np.std(combined) > 0 else 0
        return {'ann': ann, 'mdd': mdd, 'sharpe': sh, 'final': final}

    def wf_portfolio_regime(regime_mode='none', base_size=0.50, **kwargs):
        """Walk-forward for portfolio with regime sizing."""
        res = {}
        for yr in [2020, 2021, 2022, 2023, 2024, 2025]:
            ys = ye = None
            for di in range(ND):
                if dates[di].year == yr and ys is None: ys = di
                if dates[di].year == yr: ye = di + 1
            if ys is None: continue

            # Run sub-strategies for this year only
            def run_sub_year(sig_func, start_di, end_di):
                cash = float(CASH0); positions = []; daily_eq = []
                high_water = float(CASH0)
                for di in range(start_di, end_di - 1):
                    pv = cash
                    for p in positions:
                        cp = C[p['si'], di]
                        if not np.isnan(cp) and cp > 0:
                            m = MULT.get(p['sym'], DEF_MULT)
                            pv += cp * m * p['lots'] - cp * m * abs(p['lots']) * COMM
                    daily_eq.append(pv)
                    if pv > high_water: high_water = pv
                    cl = []
                    for p in positions:
                        if di - p['entry_di'] >= p['hold_days']:
                            ep = C[p['si'], di]
                            if np.isnan(ep) or ep <= 0: ep = p['entry_price']
                            m = MULT.get(p['sym'], DEF_MULT)
                            cash += ep * m * abs(p['lots']) * (1 - COMM)
                            cl.append(p)
                    for p in cl: positions.remove(p)

                    # Regime sizing
                    if regime_mode == 'none': size = base_size
                    elif regime_mode == 'breadth':
                        bth = BREADTH[di]
                        if np.isnan(bth): size = base_size
                        elif bth > kwargs.get('breadth_high', 0.70): size = kwargs.get('size_breadth_high', 0.80)
                        elif bth > kwargs.get('breadth_low', 0.40): size = kwargs.get('size_breadth_mid', 0.55)
                        else: size = kwargs.get('size_breadth_low', 0.25)
                    elif regime_mode == 'vol':
                        vol = MKT_VOL[di]
                        if np.isnan(vol) or VOL_MEDIAN <= 0: size = base_size
                        elif vol < VOL_MEDIAN * kwargs.get('vol_low_ratio', 0.8): size = kwargs.get('size_vol_low', 0.75)
                        elif vol > VOL_MEDIAN * kwargs.get('vol_high_ratio', 1.5): size = kwargs.get('size_vol_high', 0.25)
                        else: size = kwargs.get('size_vol_mid', 0.55)
                    elif regime_mode == 'dd':
                        cur_dd = (pv - high_water) / high_water if high_water > 0 else 0
                        ds = kwargs.get('dd_small', 0.10)
                        dm = kwargs.get('dd_med', 0.20)
                        dl = kwargs.get('dd_large', 0.30)
                        if cur_dd >= 0: size = kwargs.get('size_dd_peak', 0.70)
                        elif cur_dd > -ds: size = kwargs.get('size_dd_small', 0.60)
                        elif cur_dd > -dm: size = kwargs.get('size_dd_med', 0.45)
                        elif cur_dd > -dl: size = kwargs.get('size_dd_large', 0.25)
                        else: size = kwargs.get('size_dd_extreme', 0.10)
                    elif regime_mode == 'combined':
                        scores = []
                        bth = BREADTH[di]
                        if not np.isnan(bth):
                            scores.append(np.clip((bth - kwargs.get('combo_low', 0.4)) /
                                                   (kwargs.get('combo_high', 0.7) - kwargs.get('combo_low', 0.4)), 0, 1))
                        vol = MKT_VOL[di]
                        if not np.isnan(vol) and VOL_MEDIAN > 0:
                            vr = vol / VOL_MEDIAN
                            scores.append(np.clip((kwargs.get('vol_high_ratio', 1.5) - vr) /
                                                   (kwargs.get('vol_high_ratio', 1.5) - kwargs.get('vol_low_ratio', 0.8)), 0, 1))
                        pw = kwargs.get('perf_window', 20)
                        if len(daily_eq) >= pw:
                            eq_w = np.array(daily_eq[-pw:]); x = np.arange(pw)
                            try:
                                slope = np.polyfit(x, eq_w, 1)[0]
                                em = np.mean(eq_w); ns = slope / em * 100 if em > 0 else 0
                                er = np.diff(eq_w) / eq_w[:-1] * 100; er = er[np.isfinite(er)]
                                es = np.std(er) if len(er) > 5 else 1.0
                                z = ns / es if es > 0 else 0
                                scores.append(np.clip((z + 1.0) / 2.0, 0, 1))
                            except Exception: pass
                        cur_dd = (pv - high_water) / high_water if high_water > 0 else 0
                        scores.append(np.clip(1.0 + cur_dd / 0.3, 0, 1))
                        if not scores: size = base_size
                        else:
                            composite = np.mean(scores)
                            if composite > kwargs.get('combo_high', 0.7): size = kwargs.get('size_combo_high', 0.80)
                            elif composite > kwargs.get('combo_low', 0.4): size = kwargs.get('size_combo_mid', 0.55)
                            else: size = kwargs.get('size_combo_low', 0.25)
                    else: size = base_size

                    if len(positions) >= 1: continue
                    edi = di + 1
                    if edi >= end_di: continue
                    cands = sig_func(di, edi)
                    if not cands: continue
                    cands.sort(key=lambda x: -x[0])
                    item = cands[0]
                    if len(item) == 3: sc, s, pr = item; sig = ''
                    else: sc, s, pr, sig = item
                    sym = syms[s]; m = MULT.get(sym, DEF_MULT)
                    cap = cash * size
                    ct = max(1, int(cap / (pr * m * (1 + COMM))))
                    ci = pr * m * ct * (1 + COMM)
                    if ci > cash:
                        ct = int(cash * 0.9 / (pr * m * (1 + COMM)))
                        ci = pr * m * ct * (1 + COMM) if ct > 0 else 0
                    if ct <= 0 or ci <= 0 or ci > cash: continue
                    cash -= ci
                    positions.append({'si': s, 'entry_price': pr, 'entry_di': edi,
                                      'lots': ct, 'dir': 1, 'sym': sym, 'hold_days': 1, 'sig': sig})
                for p in positions:
                    ep = C[p['si'], min(end_di-1, ND-1)]
                    if np.isnan(ep) or ep <= 0: ep = p['entry_price']
                    m = MULT.get(p['sym'], DEF_MULT)
                    cash += ep * m * abs(p['lots']) * (1 - COMM)
                return np.array(daily_eq), cash

            eq_A, _ = run_sub_year(sig_union, ys, ye)
            eq_B, _ = run_sub_year(sig_v121, ys, ye)
            ml = min(len(eq_A), len(eq_B))
            if ml <= 1: res[yr] = {'ann': -100.0, 'mdd': 0}; continue
            ret_A = np.diff(eq_A[:ml]) / eq_A[:ml-1]
            ret_B = np.diff(eq_B[:ml]) / eq_B[:ml-1]
            ret_A = np.where(np.isfinite(ret_A), ret_A, 0)
            ret_B = np.where(np.isfinite(ret_B), ret_B, 0)
            comb = 0.5 * ret_A + 0.5 * ret_B
            eq = np.zeros(ml); eq[0] = float(CASH0)
            for i in range(ml-1): eq[i+1] = eq[i] * (1 + comb[i])
            final = eq[-1]
            ann = annual_return(final, CASH0, ml)
            pk = np.maximum.accumulate(eq)
            mdd = np.min((eq - pk) / pk * 100)
            res[yr] = {'ann': ann, 'mdd': mdd}
        return res

    # Portfolio baselines first
    print(f"\n  --- Portfolio Baselines (Fixed Sizing) ---")
    r_port_50 = backtest_portfolio_regime(regime_mode='none', base_size=0.50)
    print(f"  {'Port 50/50 fixed 50%':70s} | Ann={r_port_50['ann']:+8.1f}% | MDD={r_port_50['mdd']:6.1f}% | Sh={r_port_50['sharpe']:4.2f}")

    r_port_55 = backtest_portfolio_regime(regime_mode='none', base_size=0.55)
    print(f"  {'Port 50/50 fixed 55%':70s} | Ann={r_port_55['ann']:+8.1f}% | MDD={r_port_55['mdd']:6.1f}% | Sh={r_port_55['sharpe']:4.2f}")

    r_port_60 = backtest_portfolio_regime(regime_mode='none', base_size=0.60)
    print(f"  {'Port 50/50 fixed 60%':70s} | Ann={r_port_60['ann']:+8.1f}% | MDD={r_port_60['mdd']:6.1f}% | Sh={r_port_60['sharpe']:4.2f}")

    # Now test each regime on portfolio
    port_regime_configs = [
        ('breadth', 0.55, {'breadth_high': 0.70, 'breadth_low': 0.40,
                           'size_breadth_high': 0.80, 'size_breadth_mid': 0.55, 'size_breadth_low': 0.25},
         "Port Breadth 70/40 -> 80/55/25%"),
        ('breadth', 0.55, {'breadth_high': 0.75, 'breadth_low': 0.35,
                           'size_breadth_high': 0.75, 'size_breadth_mid': 0.55, 'size_breadth_low': 0.30},
         "Port Breadth 75/35 -> 75/55/30%"),
        ('vol', 0.55, {'vol_low_ratio': 0.8, 'vol_high_ratio': 1.5,
                       'size_vol_low': 0.75, 'size_vol_mid': 0.55, 'size_vol_high': 0.25},
         "Port Vol 0.8x/1.5x -> 75/55/25%"),
        ('vol', 0.55, {'vol_low_ratio': 0.8, 'vol_high_ratio': 1.5,
                       'size_vol_low': 0.70, 'size_vol_mid': 0.55, 'size_vol_high': 0.30},
         "Port Vol 0.8x/1.5x -> 70/55/30%"),
        ('perf', 0.55, {'perf_window': 20,
                        'size_perf_high': 0.80, 'size_perf_mid': 0.55, 'size_perf_low': 0.25},
         "Port Perf 20d -> 80/55/25%"),
        ('perf', 0.55, {'perf_window': 20,
                        'size_perf_high': 0.75, 'size_perf_mid': 0.55, 'size_perf_low': 0.30},
         "Port Perf 20d -> 75/55/30%"),
        ('dd', 0.55, {'size_dd_peak': 0.70, 'size_dd_small': 0.60, 'size_dd_med': 0.45,
                      'size_dd_large': 0.25, 'size_dd_extreme': 0.10,
                      'dd_small': 0.10, 'dd_med': 0.20, 'dd_large': 0.30},
         "Port DD peak/10/20/30 -> 70/60/45/25/10%"),
        ('dd', 0.55, {'size_dd_peak': 0.75, 'size_dd_small': 0.65, 'size_dd_med': 0.50,
                      'size_dd_large': 0.35, 'size_dd_extreme': 0.20,
                      'dd_small': 0.10, 'dd_med': 0.20, 'dd_large': 0.30},
         "Port DD peak/10/20/30 -> 75/65/50/35/20%"),
        ('dd', 0.55, {'size_dd_peak': 0.70, 'size_dd_small': 0.60, 'size_dd_med': 0.50,
                      'size_dd_large': 0.40, 'size_dd_extreme': 0.30,
                      'dd_small': 0.10, 'dd_med': 0.20, 'dd_large': 0.30},
         "Port DD peak/10/20/30 -> 70/60/50/40/30%"),
        ('combined', 0.55, {'combo_high': 0.70, 'combo_low': 0.40,
                            'size_combo_high': 0.80, 'size_combo_mid': 0.55, 'size_combo_low': 0.25,
                            'perf_window': 20, 'vol_low_ratio': 0.8, 'vol_high_ratio': 1.5},
         "Port Combo 0.7/0.4 -> 80/55/25%"),
        ('combined', 0.55, {'combo_high': 0.70, 'combo_low': 0.40,
                            'size_combo_high': 0.75, 'size_combo_mid': 0.55, 'size_combo_low': 0.30,
                            'perf_window': 20, 'vol_low_ratio': 0.8, 'vol_high_ratio': 1.5},
         "Port Combo 0.7/0.4 -> 75/55/30%"),
    ]

    port_results = []
    print(f"\n  --- Portfolio with Regime Sizing ---")
    for regime_mode, base_size, kw, label in port_regime_configs:
        r = backtest_portfolio_regime(regime_mode=regime_mode, base_size=base_size, **kw)
        r['desc'] = label
        r['regime_mode'] = regime_mode
        r['kw'] = kw
        port_results.append(r)
        print(f"  {label:70s} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f}")

    # ===================== SECTION 7: RANKING AND WALK-FORWARD =====================
    print("\n" + "=" * 130)
    print("  SECTION 7: RANKING — ALL SINGLE-SIGNAL RESULTS")
    print("=" * 130)

    all_single = breadth_results + vol_results + perf_results + dd_results + combo_results
    # Filter to those with reasonable MDD
    all_single_valid = [r for r in all_single if r['mdd'] > -70 and r.get('desc', '')]

    # Sort by return
    all_single_valid.sort(key=lambda x: -x['ann'])
    print(f"\n  Top 20 by Annual Return (MDD > -70%):")
    for i, r in enumerate(all_single_valid[:20]):
        desc = r.get('desc', '')
        print(f"  #{i+1:2d}: {desc:70s} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f} | N={r['n']:4d}")

    # Sort by Sharpe
    all_single_valid.sort(key=lambda x: -x['sharpe'])
    print(f"\n  Top 10 by Sharpe (MDD > -70%):")
    for i, r in enumerate(all_single_valid[:10]):
        desc = r.get('desc', '')
        print(f"  #{i+1:2d}: {desc:70s} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f} | N={r['n']:4d}")

    # Sort by return/MDD ratio
    all_with_ratio = [(r, abs(r['ann'] / r['mdd']) if r['mdd'] != 0 else 0) for r in all_single_valid]
    all_with_ratio.sort(key=lambda x: -x[1])
    print(f"\n  Top 10 by Ann/MDD Ratio:")
    for i, (r, ratio) in enumerate(all_with_ratio[:10]):
        desc = r.get('desc', '')
        print(f"  #{i+1:2d}: {desc:70s} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | Ratio={ratio:.2f}")

    # ===================== SECTION 8: WALK-FORWARD FOR BEST CONFIGS =====================
    print("\n" + "=" * 130)
    print("  SECTION 8: WALK-FORWARD VALIDATION")
    print("=" * 130)

    # WF for baselines
    print(f"\n  --- Baselines Walk-Forward ---")
    wf_base_50 = walk_forward(sig_union, regime_mode='none', base_size=0.50)
    print_wf(wf_base_50, "Union fixed 50%")

    wf_base_55 = walk_forward(sig_union, regime_mode='none', base_size=0.55)
    print_wf(wf_base_55, "Union fixed 55%")

    # Detailed WF for the actual winning configs with correct params
    print(f"\n  --- Best Single-Signal Regime Walk-Forward (with correct params) ---")

    # Vol: 0.8x/2.0x -> 75/55/30% (best single-signal ratio = 3.02)
    wf_vol = walk_forward(sig_union, regime_mode='vol', base_size=0.55,
                          vol_low_ratio=0.8, vol_high_ratio=2.0,
                          size_vol_low=0.75, size_vol_mid=0.55, size_vol_high=0.30)
    print_wf(wf_vol, "Vol 0.8x/2.0x -> 75/55/30%")

    # DD: peak/10/20/30 -> 70/60/50/40/30% (ratio = 2.99)
    wf_dd = walk_forward(sig_union, regime_mode='dd', base_size=0.55,
                         size_dd_peak=0.70, size_dd_small=0.60, size_dd_med=0.50,
                         size_dd_large=0.40, size_dd_extreme=0.30,
                         dd_small=0.10, dd_med=0.20, dd_large=0.30)
    print_wf(wf_dd, "DD 70/60/50/40/30%")

    # Vol: 0.8x/1.5x -> 80/50/20% (ratio = 2.67)
    wf_vol2 = walk_forward(sig_union, regime_mode='vol', base_size=0.55,
                           vol_low_ratio=0.8, vol_high_ratio=1.5,
                           size_vol_low=0.80, size_vol_mid=0.50, size_vol_high=0.20)
    print_wf(wf_vol2, "Vol 0.8x/1.5x -> 80/50/20%")

    # Perf: 10d -> 80/55/25% (best Sharpe = 1.54)
    wf_perf = walk_forward(sig_union, regime_mode='perf', base_size=0.55,
                           perf_window=10,
                           size_perf_high=0.80, size_perf_mid=0.55, size_perf_low=0.25)
    print_wf(wf_perf, "Perf 10d -> 80/55/25%")

    # Perf: 15d -> 80/55/25% (Sharpe = 1.53)
    wf_perf15 = walk_forward(sig_union, regime_mode='perf', base_size=0.55,
                             perf_window=15,
                             size_perf_high=0.80, size_perf_mid=0.55, size_perf_low=0.25)
    print_wf(wf_perf15, "Perf 15d -> 80/55/25%")

    # Combo: 0.7/0.4 -> 70/55/35% (best combo ratio = 2.62)
    wf_combo = walk_forward(sig_union, regime_mode='combined', base_size=0.55,
                            combo_high=0.70, combo_low=0.40,
                            size_combo_high=0.70, size_combo_mid=0.55, size_combo_low=0.35,
                            perf_window=20, vol_low_ratio=0.8, vol_high_ratio=1.5)
    print_wf(wf_combo, "Combo 0.7/0.4 -> 70/55/35%")

    # DD gentle: 65/55/45/35/20% (ratio = 2.86, good MDD)
    wf_dd_gentle = walk_forward(sig_union, regime_mode='dd', base_size=0.55,
                                size_dd_peak=0.65, size_dd_small=0.55, size_dd_med=0.45,
                                size_dd_large=0.35, size_dd_extreme=0.20,
                                dd_small=0.10, dd_med=0.20, dd_large=0.30)
    print_wf(wf_dd_gentle, "DD 65/55/45/35/20%")

    # ===================== SECTION 9: DETAILED WF FOR TOP PORTFOLIO CONFIGS =====================
    print("\n" + "=" * 130)
    print("  SECTION 9: PORTFOLIO WALK-FORWARD (with per-year MDD)")
    print("=" * 130)

    # WF for portfolio baselines
    print(f"\n  --- Portfolio Baseline WF ---")
    wf_port_base = wf_portfolio_regime(regime_mode='none', base_size=0.50)
    print_wf(wf_port_base, "Port fixed 50%")

    wf_port_55 = wf_portfolio_regime(regime_mode='none', base_size=0.55)
    print_wf(wf_port_55, "Port fixed 55%")

    # WF for best portfolio regime configs
    print(f"\n  --- Portfolio Regime WF ---")
    # Sort portfolio results by Ann/MDD ratio
    port_valid = [r for r in port_results if r['mdd'] > -70]
    port_with_ratio = [(r, abs(r['ann'] / r['mdd']) if r['mdd'] != 0 else 0) for r in port_valid]
    port_with_ratio.sort(key=lambda x: -x[1])

    for r, ratio in port_with_ratio[:5]:
        desc = r.get('desc', '')
        regime_mode = r.get('regime_mode', 'none')
        kw = r.get('kw', {})
        wf_r = wf_portfolio_regime(regime_mode=regime_mode, base_size=0.55, **kw)
        print_wf(wf_r, f"{desc} (ratio={ratio:.2f})")

    # ===================== SECTION 10: BEST OVERALL COMPARISON TABLE =====================
    print("\n" + "=" * 130)
    print("  SECTION 10: BEST OVERALL COMPARISON")
    print("=" * 130)

    print(f"\n  Baseline: Union/V121 50/50 @50% sizing = +155% annual, -24% MDD")
    print(f"\n  {'Config':70s} | {'Ann':>8s} | {'MDD':>6s} | {'Sh':>4s} | {'Ratio':>5s}")
    print(f"  {'-'*70}-+-{'-'*8}-+-{'-'*6}-+-{'-'*4}-+-{'-'*5}")

    # Collect all results (single + portfolio)
    everything = all_single_valid + [r for r in port_results if r.get('desc', '') and r['mdd'] > -70]
    everything_with_ratio = [(r, abs(r['ann'] / r['mdd']) if r['mdd'] != 0 else 0) for r in everything]
    everything_with_ratio.sort(key=lambda x: -x[1])

    for i, (r, ratio) in enumerate(everything_with_ratio[:25]):
        desc = r.get('desc', '')
        print(f"  #{i+1:2d}: {desc:70s} | {r['ann']:+8.1f}% | {r['mdd']:6.1f}% | {r['sharpe']:4.2f} | {ratio:5.2f}")

    # Final answer: which regime sizing actually helps?
    print(f"\n\n  {'='*130}")
    print(f"  VERDICT: Does regime-aware sizing improve the return/MDD tradeoff?")
    print(f"  {'='*130}")

    base_ann = r_union_50['ann']; base_mdd = r_union_50['mdd']
    base_ratio = abs(base_ann / base_mdd) if base_mdd != 0 else 0
    print(f"\n  Baseline: Ann={base_ann:+.1f}% MDD={base_mdd:.1f}% Ratio={base_ratio:.2f}")

    better_ratio = [(r, abs(r['ann']/r['mdd'])) for r in everything
                    if r['mdd'] != 0 and abs(r['ann']/r['mdd']) > base_ratio]
    better_ratio.sort(key=lambda x: -x[1])

    if better_ratio:
        print(f"\n  Configs that BEAT baseline Ann/MDD ratio ({base_ratio:.2f}):")
        for i, (r, ratio) in enumerate(better_ratio[:10]):
            desc = r.get('desc', '')
            print(f"  #{i+1}: {desc:70s} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | Ratio={ratio:.2f}")
    else:
        print(f"\n  NO configs beat the baseline ratio. Fixed sizing wins.")

    print(f"\n  Elapsed: {time.time()-t_start:.0f}s")
    print("=" * 130)


if __name__ == '__main__':
    main()
