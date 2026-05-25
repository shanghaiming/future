"""
Alpha Futures V163 — Cross-Sectional Rank-Based Selection
==============================================================================
Currently, signals are computed per-commodity and the best score wins.
V163 tests rank-based selection across ALL commodities:

1. Momentum rank: Rank all NS commodities by ROC(5), take top N ranks
2. Multi-factor rank: Rank by combined score (ROC + Z-score + breakout energy)
3. Sector-balanced rank: Rank within each sector, take top 1 per sector
4. Volume-weighted rank: Prefer commodities with volume anomaly (V > 2* 20-day avg)
5. Momentum persistence rank: Rank by ROC5, but only if ROC5 > ROC10 > ROC20

Key insight: Instead of binary signal filters, rank ALL commodities and take top N.

Parameters to sweep:
- rank_method: ['roc5', 'roc10', 'roc20', 'multi_factor', 'sector_balanced',
                'vol_weighted', 'persistence']
- top_n: [3, 4, 5]
- min_roc: [0.5, 1.0, 1.5, 2.0]
- sector_balanced: [True, False]
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

# Sector groups for sector-balanced ranking
SECTORS = {
    'black':     {'rbfi', 'hcfi', 'ifi', 'jfi', 'jmfi'},
    'nonferrous': {'cufi', 'alfi', 'znfi', 'aufi', 'agfi', 'nifi'},
    'energy':    {'scfi', 'mfi', 'ptafi', 'bfi', 'fufi', 'egfi', 'pgfi', 'bcfi'},
    'ags':       {'afi', 'mfi', 'yfi', 'cfi', 'srfi', 'cffi', 'whfi', 'rrfi', 'lrfi'},
}


def annual_return(final, initial, n_days):
    if final <= 0 or initial <= 0 or n_days <= 0: return -100.0
    return (final / initial) ** (1.0 / (n_days / 252)) * 100 - 100


def main():
    print("=" * 130)
    print("  V163 — Cross-Sectional Rank-Based Selection")
    print("=" * 130)
    t_start = time.time()
    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)
    print(f"  {NS} commodities, {ND} days")

    # ===================== BUILD SECTOR MAP =====================
    # Map each symbol index to its sector name
    sym_to_sector = {}
    for si, s in enumerate(syms):
        for sector_name, sector_syms in SECTORS.items():
            if s in sector_syms:
                sym_to_sector[si] = sector_name
                break
        else:
            sym_to_sector[si] = None  # not in any defined sector

    # ===================== PRECOMPUTE =====================
    print("\n[Precompute]...", flush=True)
    t0 = time.time()

    RET = np.full((NS, ND), np.nan)
    ROC5 = np.full((NS, ND), np.nan)
    ROC10 = np.full((NS, ND), np.nan)
    ROC20 = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si].astype(np.float64)
        for di in range(1, ND):
            if not np.isnan(c[di]) and not np.isnan(c[di-1]) and c[di-1] > 0:
                RET[si, di] = (c[di] / c[di-1] - 1) * 100
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
            v = rets[~np.isnan(rets)]
            if len(v) < 10: continue
            s = np.std(v, ddof=1)
            if s > 0 and not np.isnan(RET[si, di]):
                ZSCORE[si, di] = (RET[si, di] - np.mean(v)) / s

    # Breakout energy: close vs 20-day high, normalized by ATR
    BREAKOUT = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            cp = C[si, di]
            h20 = H[si, di-19:di+1]
            h20v = h20[~np.isnan(h20)]
            if len(h20v) < 10 or np.isnan(cp) or cp <= 0: continue
            hmax = np.max(h20v)
            atr = ATR14[si, di]
            if np.isnan(atr) or atr <= 0: continue
            BREAKOUT[si, di] = (cp - hmax) / atr  # positive = breakout above 20d high

    # Volume ratio: current volume / 20-day average volume
    VOL_RATIO = np.full((NS, ND), np.nan)
    VOL_MA20 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            vw = V[si, di-19:di+1]
            vv = vw[~np.isnan(vw)]
            if len(vv) < 10: continue
            avg_v = np.mean(vv)
            VOL_MA20[si, di] = avg_v
            cv = V[si, di]
            if not np.isnan(cv) and avg_v > 0:
                VOL_RATIO[si, di] = cv / avg_v

    # ROC improving: ROC5 today vs ROC5 yesterday
    ROC5_IMPROVING = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(1, ND):
            r0 = ROC5[si, di]
            r1 = ROC5[si, di-1]
            if not np.isnan(r0) and not np.isnan(r1):
                ROC5_IMPROVING[si, di] = r0 - r1

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

    # ===================== RANK-BASED SIGNAL FUNCTION =====================
    def rank_commodities(di, edi, method='roc5', min_roc=1.0, top_n=3,
                         sector_balanced=False):
        """
        Rank all NS commodities by the specified method and return top N candidates.
        Returns list of (score, si, entry_price, method_label).
        """
        candidates = []

        for si in range(NS):
            ep = O[si, edi]
            if np.isnan(ep) or ep <= 0:
                continue

            if method == 'roc5':
                roc = ROC5[si, di]
                if np.isnan(roc) or roc < min_roc: continue
                score = roc

            elif method == 'roc10':
                roc = ROC10[si, di]
                if np.isnan(roc) or roc < min_roc: continue
                score = roc

            elif method == 'roc20':
                roc = ROC20[si, di]
                if np.isnan(roc) or roc < min_roc: continue
                score = roc

            elif method == 'multi_factor':
                roc = ROC5[si, di]
                zs = ZSCORE[si, di]
                bo = BREAKOUT[si, di]
                if np.isnan(roc) or roc < min_roc: continue
                # Build combined score from available factors
                score = roc  # base: momentum
                if not np.isnan(zs) and zs > 0:
                    score += zs * 2  # z-score bonus
                if not np.isnan(bo) and bo > 0:
                    score += bo * 3  # breakout bonus

            elif method == 'vol_weighted':
                roc = ROC5[si, di]
                vr = VOL_RATIO[si, di]
                if np.isnan(roc) or roc < min_roc: continue
                score = roc
                if not np.isnan(vr) and vr > 2.0:
                    score *= (1.0 + (vr - 2.0) * 0.5)  # volume anomaly boost

            elif method == 'persistence':
                roc5 = ROC5[si, di]
                roc10 = ROC10[si, di]
                roc20 = ROC20[si, di]
                if np.isnan(roc5) or np.isnan(roc10) or np.isnan(roc20): continue
                if roc5 < min_roc: continue
                # Must have momentum persistence: ROC5 > ROC10 > ROC20
                if not (roc5 > roc10 > roc20): continue
                # Score = weighted momentum with steepness bonus
                steepness = (roc5 - roc20) / max(abs(roc10), 0.1)
                score = roc5 + steepness * 2

            elif method == 'sector_balanced':
                # handled below in a separate path
                roc = ROC5[si, di]
                zs = ZSCORE[si, di]
                bo = BREAKOUT[si, di]
                vr = VOL_RATIO[si, di]
                if np.isnan(roc) or roc < min_roc: continue
                score = roc
                if not np.isnan(zs) and zs > 0:
                    score += zs * 2
                if not np.isnan(bo) and bo > 0:
                    score += bo * 3
                if not np.isnan(vr) and vr > 2.0:
                    score *= (1.0 + (vr - 2.0) * 0.3)

            else:
                continue

            candidates.append((score, si, ep, method))

        # --- Sector-balanced path ---
        if sector_balanced and method in ('roc5', 'roc10', 'roc20', 'multi_factor',
                                          'vol_weighted', 'persistence'):
            # Group candidates by sector, pick top 1 per sector, then rank overall
            sector_cands = {}  # sector_name -> list of (score, si, ep, method)
            for item in candidates:
                si = item[1]
                sec = sym_to_sector.get(si)
                if sec is not None:
                    if sec not in sector_cands:
                        sector_cands[sec] = []
                    sector_cands[sec].append(item)

            # Take top 1 from each sector
            finalists = []
            for sec_name, sec_items in sector_cands.items():
                sec_items.sort(key=lambda x: -x[0])
                finalists.append(sec_items[0])

            # Also include commodities not in any defined sector
            no_sector = [item for item in candidates if sym_to_sector.get(item[1]) is None]
            no_sector.sort(key=lambda x: -x[0])
            finalists.extend(no_sector[:2])  # allow up to 2 non-sector

            finalists.sort(key=lambda x: -x[0])
            return finalists[:top_n]

        elif method == 'sector_balanced':
            # Direct sector_balanced method: same multi-factor but always sector-balanced
            sector_cands = {}
            for item in candidates:
                si = item[1]
                sec = sym_to_sector.get(si)
                if sec is not None:
                    if sec not in sector_cands:
                        sector_cands[sec] = []
                    sector_cands[sec].append(item)

            finalists = []
            for sec_name, sec_items in sector_cands.items():
                sec_items.sort(key=lambda x: -x[0])
                finalists.append(sec_items[0])

            no_sector = [item for item in candidates if sym_to_sector.get(item[1]) is None]
            no_sector.sort(key=lambda x: -x[0])
            finalists.extend(no_sector[:2])

            finalists.sort(key=lambda x: -x[0])
            return finalists[:top_n]

        # Standard: sort by score descending, take top N
        candidates.sort(key=lambda x: -x[0])
        return candidates[:top_n]

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

    # ===================== HELPER: Compute composite regime score =====================
    def compute_composite(di, daily_eq, high_water, perf_window=20):
        scores = []
        bth = BREADTH[di]
        if not np.isnan(bth):
            scores.append(np.clip((bth - 0.4) / (0.7 - 0.4), 0, 1))
        vol = MKT_VOL[di]
        if not np.isnan(vol) and VOL_MEDIAN > 0:
            vol_ratio = vol / VOL_MEDIAN
            scores.append(np.clip((1.5 - vol_ratio) / (1.5 - 0.8), 0, 1))
        if len(daily_eq) >= perf_window:
            eq_window = np.array(daily_eq[-perf_window:])
            x = np.arange(perf_window)
            try:
                slope = np.polyfit(x, eq_window, 1)[0]
                eq_mean = np.mean(eq_window)
                norm_slope = slope / eq_mean * 100 if eq_mean > 0 else 0
                eq_rets = np.diff(eq_window) / eq_window[:-1] * 100
                eq_rets = eq_rets[np.isfinite(eq_rets)]
                eq_std = np.std(eq_rets) if len(eq_rets) > 5 else 1.0
                z = norm_slope / eq_std if eq_std > 0 else 0
                scores.append(np.clip((z + 1.0) / 2.0, 0, 1))
            except Exception:
                pass
        if high_water > 0:
            cur_dd = (daily_eq[-1] - high_water) / high_water
        else:
            cur_dd = 0
        scores.append(np.clip(1.0 + cur_dd / 0.3, 0, 1))
        return np.mean(scores) if scores else 0.5

    # ===================== HELPER: DD-based sizing =====================
    def dd_size(pv, high_water, tiers):
        if high_water <= 0:
            return tiers[0][1]
        dd = (pv - high_water) / high_water
        for dd_thresh, size_frac in tiers:
            if dd >= -dd_thresh:
                return size_frac
        return tiers[-1][1]

    # ===================== UNIFIED BACKTEST ENGINE =====================
    def backtest(start_di=MIN_TRAIN, end_di=None,
                 rank_method='roc5',
                 min_roc=1.0,
                 dd_tiers=None,
                 regime_lo=0.5, regime_hi=1.5,
                 hold=1, top_n=3,
                 sector_balanced=False,
                 max_corr=0.7):
        if end_di is None: end_di = ND
        if dd_tiers is None:
            dd_tiers = [(0, 1.00), (0.10, 0.90), (0.20, 0.70), (0.30, 0.50)]

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

            # --- Kitchen Sink sizing ---
            dd_sz = dd_size(pv, high_water, dd_tiers)
            composite = compute_composite(di, daily_eq, high_water)
            regime_mult = regime_lo + composite * (regime_hi - regime_lo)
            pos_size = dd_sz * regime_mult
            pos_size = max(0.05, min(0.99, pos_size))

            # --- Enter positions ---
            if len(positions) >= top_n: continue
            edi = di + 1
            if edi >= end_di: continue

            held_si = set(p['si'] for p in positions)
            slots = top_n - len(positions)

            # Use cash_snapshot before entry loop for capital allocation
            cash_snapshot = cash

            # Get ranked candidates
            ranked = rank_commodities(di, edi, method=rank_method,
                                       min_roc=min_roc, top_n=slots,
                                       sector_balanced=sector_balanced)

            if not ranked: continue

            # Filter out already-held symbols
            ranked = [r for r in ranked if r[1] not in held_si]
            if not ranked: continue

            # Correlation filter: remove highly correlated entries
            entries = []
            for sc, s, pr, mlabel in ranked:
                # Check correlation with already-selected entries
                too_corr = False
                for _, es, _, _ in entries:
                    corr = get_corr(s, es, di)
                    if corr >= max_corr:
                        too_corr = True
                        break
                if not too_corr:
                    entries.append((sc, s, pr, mlabel))

            if not entries: continue

            n_planned = len(entries)
            for sc, s, pr, mlabel in entries:
                if s in set(p['si'] for p in positions): continue
                if len(positions) >= top_n: break
                cap = cash_snapshot * pos_size / n_planned
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
                                  'sig': mlabel, 'score': sc})

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
        print(f"  {label:85s} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f} | R/M={ratio:.2f} | N={r['n']:4d}")

    def walk_forward(rank_method='roc5', min_roc=1.0, dd_tiers=None,
                     regime_lo=0.5, regime_hi=1.5, hold=1, top_n=3,
                     sector_balanced=False, max_corr=0.7, label=""):
        res = {}
        for yr in [2020, 2021, 2022, 2023, 2024, 2025]:
            ys = ye = None
            for di in range(ND):
                if dates[di].year == yr and ys is None: ys = di
                if dates[di].year == yr: ye = di + 1
            if ys is None: continue
            r = backtest(start_di=ys, end_di=ye, rank_method=rank_method,
                         min_roc=min_roc, dd_tiers=dd_tiers,
                         regime_lo=regime_lo, regime_hi=regime_hi,
                         hold=hold, top_n=top_n,
                         sector_balanced=sector_balanced, max_corr=max_corr)
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

    # ===================== CONFIG DEFINITIONS =====================
    DD_TIERS = {
        'aggro100': [(0, 1.00), (0.10, 0.90), (0.20, 0.70), (0.30, 0.50)],
    }

    RANK_METHODS = ['roc5', 'roc10', 'roc20', 'multi_factor',
                    'sector_balanced', 'vol_weighted', 'persistence']
    TOP_N_VALUES = [3, 4, 5]
    MIN_ROC_VALUES = [0.5, 1.0, 1.5, 2.0]
    SECTOR_BALANCED_FLAGS = [False, True]

    # ===================== SECTION 0: BASELINE =====================
    print("\n" + "=" * 120)
    print("  SECTION 0: BASELINES — Rank-based with default params")
    print("=" * 120)

    dd_base = DD_TIERS['aggro100']

    for method in RANK_METHODS:
        for tn in [3, 5]:
            for sb in [False, True]:
                sb_str = "SB" if sb else "--"
                label = f"rank={method:15s} top{tn} minROC=1.0 {sb_str}"
                r = backtest(rank_method=method, min_roc=1.0, dd_tiers=dd_base,
                             top_n=tn, sector_balanced=sb)
                pr(r, label)

    # ===================== SECTION 1: FULL PARAMETER GRID =====================
    print("\n" + "=" * 120)
    print("  SECTION 1: FULL PARAMETER GRID — Rank-based selection")
    print("  Methods x top_n x min_roc x sector_balanced")
    print("=" * 120)

    all_results = []  # (ann, mdd, sharpe, n, label, config_dict)

    for method in RANK_METHODS:
        for tn in TOP_N_VALUES:
            for mroc in MIN_ROC_VALUES:
                for sb in SECTOR_BALANCED_FLAGS:
                    sb_str = "SB" if sb else "--"
                    label = f"rank={method:15s} top{tn} minROC={mroc:.1f} {sb_str}"
                    r = backtest(rank_method=method, min_roc=mroc, dd_tiers=dd_base,
                                 top_n=tn, sector_balanced=sb)
                    pr(r, label)
                    all_results.append((r['ann'], r['mdd'], r['sharpe'], r['n'],
                                        label,
                                        {'method': method, 'top_n': tn,
                                         'min_roc': mroc, 'sector_balanced': sb}))

    # ===================== SECTION 2: TOP 20 FULL-PERIOD =====================
    print("\n" + "=" * 120)
    print("  SECTION 2: TOP 20 FULL-PERIOD by annual return")
    print("=" * 120)

    all_results.sort(key=lambda x: -x[0])
    for i, (ann, mdd, sh, n, label, cfg) in enumerate(all_results[:20]):
        ratio = abs(ann / mdd) if mdd != 0 else 0
        print(f"  #{i+1:2d} | Ann={ann:+8.1f}% | MDD={mdd:6.1f}% | Sh={sh:4.2f} | R/M={ratio:.2f} | N={n:4d} | {label}")

    # ===================== SECTION 3: TOP 20 BY RISK-ADJUSTED =====================
    print("\n" + "=" * 120)
    print("  SECTION 3: TOP 20 FULL-PERIOD by Ann/MDD ratio")
    print("=" * 120)

    all_results_ra = sorted(all_results,
                             key=lambda x: -abs(x[0] / x[1]) if x[1] != 0 else 0)
    for i, (ann, mdd, sh, n, label, cfg) in enumerate(all_results_ra[:20]):
        ratio = abs(ann / mdd) if mdd != 0 else 0
        print(f"  #{i+1:2d} | Ann={ann:+8.1f}% | MDD={mdd:6.1f}% | Sh={sh:4.2f} | R/M={ratio:.2f} | N={n:4d} | {label}")

    # ===================== SECTION 4: WALK-FORWARD VALIDATION — TOP 30 =====================
    print("\n" + "=" * 120)
    print("  SECTION 4: WALK-FORWARD VALIDATION — TOP 30 by full-period ann")
    print("=" * 120)

    wf_all = {}
    for i, (ann, mdd, sh, n, label, cfg) in enumerate(all_results[:30]):
        if label in wf_all: continue
        wf_res = walk_forward(rank_method=cfg['method'],
                              min_roc=cfg['min_roc'],
                              dd_tiers=dd_base,
                              top_n=cfg['top_n'],
                              sector_balanced=cfg['sector_balanced'],
                              label=label)
        wf_all[label] = (wf_res, cfg)
        print_wf(wf_res, label)

    # ===================== SECTION 5: BEST PER RANK METHOD =====================
    print("\n" + "=" * 120)
    print("  SECTION 5: BEST PER RANK METHOD (WF avg)")
    print("=" * 120)

    for method in RANK_METHODS:
        method_results = []
        for label, (wf_res, cfg) in wf_all.items():
            if cfg['method'] == method:
                avg_ann = np.mean([r['ann'] for r in wf_res.values()])
                worst_mdd = min(r['mdd'] for r in wf_res.values())
                pos = sum(1 for r in wf_res.values() if r['ann'] > 0)
                method_results.append((avg_ann, worst_mdd, pos, label, cfg, wf_res))
        if not method_results:
            print(f"\n  {method}: no WF results")
            continue
        method_results.sort(key=lambda x: -x[0])
        best = method_results[0]
        avg_ann, worst_mdd, pos, label, cfg, wf_res = best
        ws = " | ".join([f"{yr}:{r['ann']:+.0f}%/{r['mdd']:.0f}%"
                         for yr, r in sorted(wf_res.items())])
        print(f"\n  {method:15s} best: WF AVG={avg_ann:+.0f}% | WorstMDD={worst_mdd:.0f}% | {pos}/6 pos")
        print(f"     {label}")
        print(f"     {ws}")

        # Show top 3 for this method
        for j, (avg, wmdd, pos_j, lbl, _, wfr) in enumerate(method_results[:3]):
            ws_j = " | ".join([f"{yr}:{r['ann']:+.0f}%/{r['mdd']:.0f}%"
                               for yr, r in sorted(wfr.items())])
            print(f"    #{j+1} WF AVG={avg:+.0f}% | {lbl}")
            print(f"       {ws_j}")

    # ===================== SECTION 6: SECTOR BALANCED vs NOT =====================
    print("\n" + "=" * 120)
    print("  SECTION 6: SECTOR-BALANCED vs STANDARD — WF comparison")
    print("=" * 120)

    for method in RANK_METHODS:
        sb_results = [(label, wf_res, cfg) for label, (wf_res, cfg) in wf_all.items()
                      if cfg['method'] == method]
        if not sb_results: continue

        sb_avg = [np.mean([r['ann'] for r in wfr.values()])
                  for _, wfr, cfg in sb_results if cfg['sector_balanced']]
        nsb_avg = [np.mean([r['ann'] for r in wfr.values()])
                   for _, wfr, cfg in sb_results if not cfg['sector_balanced']]

        sb_best = max(sb_avg) if sb_avg else float('-inf')
        nsb_best = max(nsb_avg) if nsb_avg else float('-inf')

        sb_wmdd = min(min(r['mdd'] for r in wfr.values())
                      for _, wfr, cfg in sb_results if cfg['sector_balanced']) \
            if any(cfg['sector_balanced'] for _, _, cfg in sb_results) else 0
        nsb_wmdd = min(min(r['mdd'] for r in wfr.values())
                       for _, wfr, cfg in sb_results if not cfg['sector_balanced']) \
            if any(not cfg['sector_balanced'] for _, _, cfg in sb_results) else 0

        print(f"  {method:15s}: SB best={sb_best:+.0f}% (worstMDD={sb_wmdd:.0f}%) | "
              f"Standard best={nsb_best:+.0f}% (worstMDD={nsb_wmdd:.0f}%)")

    # ===================== SECTION 7: TOP 5 BY WF AVG =====================
    print("\n" + "=" * 120)
    print("  SECTION 7: TOP 5 CONFIGS BY WF AVERAGE")
    print("=" * 120)

    wf_ranked = []
    for label, (wf_res, cfg) in wf_all.items():
        avg_ann = np.mean([r['ann'] for r in wf_res.values()])
        worst_mdd = min(r['mdd'] for r in wf_res.values())
        best_ann = max(r['ann'] for r in wf_res.values())
        pos = sum(1 for r in wf_res.values() if r['ann'] > 0)
        total_n = sum(r['n'] for r in wf_res.values())
        avg_wr = np.mean([r['wr'] for r in wf_res.values()])
        wf_ranked.append((avg_ann, worst_mdd, best_ann, pos, total_n, avg_wr,
                          label, cfg, wf_res))

    wf_ranked.sort(key=lambda x: -x[0])

    for i, (avg_ann, worst_mdd, best_ann, pos, total_n, avg_wr, label, cfg, wf_res) \
            in enumerate(wf_ranked[:5]):
        ws = " | ".join([f"{yr}:{r['ann']:+.0f}%/{r['mdd']:.0f}%"
                         for yr, r in sorted(wf_res.items())])
        print(f"\n  #{i+1} WF AVG={avg_ann:+.0f}% | WorstMDD={worst_mdd:.0f}% | "
              f"{pos}/6 pos | TotalTrades={total_n} | AvgWR={avg_wr:.1f}%")
        print(f"     {label}")
        print(f"     {ws}")

    # ===================== SECTION 8: BEST RISK-ADJUSTED =====================
    print("\n" + "=" * 120)
    print("  SECTION 8: BEST RISK-ADJUSTED (WF avg / |worst MDD|)")
    print("=" * 120)

    wf_ra = sorted(wf_ranked,
                    key=lambda x: -abs(x[0] / x[1]) if x[1] != 0 else 0)
    for i, (avg_ann, worst_mdd, best_ann, pos, total_n, avg_wr, label, cfg, wf_res) \
            in enumerate(wf_ra[:5]):
        ratio = abs(avg_ann / worst_mdd) if worst_mdd != 0 else 0
        ws = " | ".join([f"{yr}:{r['ann']:+.0f}%/{r['mdd']:.0f}%"
                         for yr, r in sorted(wf_res.items())])
        print(f"\n  #{i+1} WF AVG={avg_ann:+.0f}% | WorstMDD={worst_mdd:.0f}% | "
              f"R/M={ratio:.2f} | {pos}/6 pos")
        print(f"     {label}")
        print(f"     {ws}")

    # ===================== SECTION 9: CONFIGS MEETING TARGETS =====================
    print("\n" + "=" * 120)
    print("  SECTION 9: CONFIGS MEETING TARGET (>100% WF avg AND >-25% worst WF MDD)")
    print("=" * 120)

    targets_met = [(avg, wmdd, bann, pos, tn, awr, lbl, cfg, wfr)
                   for avg, wmdd, bann, pos, tn, awr, lbl, cfg, wfr in wf_ranked
                   if avg > 100 and wmdd > -25]
    if targets_met:
        for i, (avg_ann, worst_mdd, best_ann, pos, total_n, avg_wr, label, cfg, wf_res) \
                in enumerate(targets_met[:10]):
            ws = " | ".join([f"{yr}:{r['ann']:+.0f}%/{r['mdd']:.0f}%"
                             for yr, r in sorted(wf_res.items())])
            print(f"\n  *** TARGET MET #{i+1}: WF AVG={avg_ann:+.0f}% | "
                  f"WorstMDD={worst_mdd:.0f}% | {pos}/6 pos")
            print(f"      {label}")
            print(f"      {ws}")
    else:
        print("\n  No configs met both targets. Closest configs:")
        for avg, wmdd, bann, pos, tn, awr, lbl, cfg, wfr in wf_ranked[:5]:
            ws = " | ".join([f"{yr}:{r['ann']:+.0f}%/{r['mdd']:.0f}%"
                             for yr, r in sorted(wfr.items())])
            print(f"  WF AVG={avg:+.0f}% | WorstMDD={wmdd:.0f}% | {lbl}")
            print(f"    {ws}")

    # ===================== SECTION 10: MIN_ROC SENSITIVITY =====================
    print("\n" + "=" * 120)
    print("  SECTION 10: MIN_ROC SENSITIVITY (best method, top_n=3)")
    print("=" * 120)

    # Find the best method from WF results
    if wf_ranked:
        best_method = wf_ranked[0][7]['method']
    else:
        best_method = 'roc5'

    for mroc in MIN_ROC_VALUES:
        for sb in [False, True]:
            sb_str = "SB" if sb else "--"
            label = f"rank={best_method:15s} top3 minROC={mroc:.1f} {sb_str}"
            if label not in wf_all:
                wf_res = walk_forward(rank_method=best_method, min_roc=mroc,
                                      dd_tiers=dd_base, top_n=3,
                                      sector_balanced=sb, label=label)
                wf_all[label] = (wf_res, {'method': best_method, 'top_n': 3,
                                           'min_roc': mroc, 'sector_balanced': sb})
            wf_res, _ = wf_all[label]
            avg_ann = np.mean([r['ann'] for r in wf_res.values()])
            worst_mdd = min(r['mdd'] for r in wf_res.values())
            ws = " | ".join([f"{yr}:{r['ann']:+.0f}%/{r['mdd']:.0f}%"
                             for yr, r in sorted(wf_res.items())])
            print(f"  minROC={mroc:.1f} {sb_str}: WF AVG={avg_ann:+.0f}% | WorstMDD={worst_mdd:.0f}%")
            print(f"    {ws}")

    # ===================== SECTION 11: TOP_N SENSITIVITY =====================
    print("\n" + "=" * 120)
    print("  SECTION 11: TOP_N SENSITIVITY (best method, min_roc=1.0)")
    print("=" * 120)

    for tn in TOP_N_VALUES:
        for sb in [False, True]:
            sb_str = "SB" if sb else "--"
            label = f"rank={best_method:15s} top{tn} minROC=1.0 {sb_str}"
            if label not in wf_all:
                wf_res = walk_forward(rank_method=best_method, min_roc=1.0,
                                      dd_tiers=dd_base, top_n=tn,
                                      sector_balanced=sb, label=label)
                wf_all[label] = (wf_res, {'method': best_method, 'top_n': tn,
                                           'min_roc': 1.0, 'sector_balanced': sb})
            wf_res, _ = wf_all[label]
            avg_ann = np.mean([r['ann'] for r in wf_res.values()])
            worst_mdd = min(r['mdd'] for r in wf_res.values())
            ws = " | ".join([f"{yr}:{r['ann']:+.0f}%/{r['mdd']:.0f}%"
                             for yr, r in sorted(wf_res.items())])
            print(f"  top{tn} {sb_str}: WF AVG={avg_ann:+.0f}% | WorstMDD={worst_mdd:.0f}%")
            print(f"    {ws}")

    # ===================== SECTION 12: ALL WF RESULTS RANKED =====================
    print("\n" + "=" * 120)
    print("  SECTION 12: ALL WF RESULTS — RANKED")
    print("=" * 120)

    # Re-rank including all new configs from sections 10-11
    wf_ranked_all = []
    for label, (wf_res, cfg) in wf_all.items():
        avg_ann = np.mean([r['ann'] for r in wf_res.values()])
        worst_mdd = min(r['mdd'] for r in wf_res.values())
        best_ann = max(r['ann'] for r in wf_res.values())
        pos = sum(1 for r in wf_res.values() if r['ann'] > 0)
        total_n = sum(r['n'] for r in wf_res.values())
        avg_wr = np.mean([r['wr'] for r in wf_res.values()])
        wf_ranked_all.append((avg_ann, worst_mdd, best_ann, pos, total_n, avg_wr,
                               label, cfg, wf_res))

    wf_ranked_all.sort(key=lambda x: -x[0])

    for i, (avg_ann, worst_mdd, best_ann, pos, total_n, avg_wr, label, cfg, wf_res) \
            in enumerate(wf_ranked_all[:20]):
        ws = " | ".join([f"{yr}:{r['ann']:+.0f}%/{r['mdd']:.0f}%"
                         for yr, r in sorted(wf_res.items())])
        print(f"  #{i+1:2d} WF AVG={avg_ann:>+7.0f}% | WorstMDD={worst_mdd:>5.0f}% | "
              f"{pos}/6 | {label}")
        print(f"       {ws}")

    # ===================== FINAL SUMMARY =====================
    print("\n" + "=" * 120)
    print("  FINAL SUMMARY")
    print("=" * 120)

    if wf_ranked_all:
        best = wf_ranked_all[0]
        avg_ann, worst_mdd, best_ann, pos, total_n, avg_wr, label, cfg, wf_res = best
        print(f"\n  Best V163: {label}")
        print(f"    WF AVG={avg_ann:+.0f}% | WorstMDD={worst_mdd:.0f}% | {pos}/6 pos | AvgWR={avg_wr:.1f}%")
        ws = " | ".join([f"{yr}:{r['ann']:+.0f}%/{r['mdd']:.0f}%"
                         for yr, r in sorted(wf_res.items())])
        print(f"    {ws}")

        # Best per method summary
        print(f"\n  --- Best per method:")
        for method in RANK_METHODS:
            method_best = [(avg, wmdd, pos_j, lbl) for avg, wmdd, bann, pos_j, tn, awr, lbl, c, wfr
                           in wf_ranked_all if c['method'] == method]
            if method_best:
                method_best.sort(key=lambda x: -x[0])
                avg, wmdd, pos_j, lbl = method_best[0]
                print(f"    {method:15s}: WF AVG={avg:+.0f}% | WorstMDD={wmdd:.0f}% | "
                      f"{pos_j}/6 pos | {lbl}")

        # Best risk-adjusted
        wf_ra_all = sorted(wf_ranked_all,
                            key=lambda x: -abs(x[0] / x[1]) if x[1] != 0 else 0)
        if wf_ra_all:
            ba = wf_ra_all[0]
            ratio = abs(ba[0] / ba[1]) if ba[1] != 0 else 0
            print(f"\n  Best risk-adjusted: {ba[6]}")
            print(f"    WF AVG={ba[0]:+.0f}% | WorstMDD={ba[1]:.0f}% | R/M={ratio:.2f}")

    print(f"\n  Elapsed: {time.time()-t_start:.0f}s")
    print("=" * 120)


if __name__ == '__main__':
    main()
