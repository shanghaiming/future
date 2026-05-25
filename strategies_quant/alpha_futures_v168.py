"""
Alpha Futures V168 — Sector Rotation & Timing Effects
==============================================================================
V157 champion gives +267%/-26% WF with V121+Union momentum signals on 68
commodities. Signal tuning is saturated (all converge to +267%).

V168 explores whether SECTOR ROTATION adds alpha:
  A. Sector momentum rank: only trade top-1/2 sectors by avg ROC(5)
  B. Sector breadth filter: only trade sectors where >50% have positive ROC5
  C. Sector rotation timing: defensive (ags/chem) in DD, aggressive (energy/black) near HWM
  D. Month-of-year bias: test seasonal sector preferences
  E. Day-of-week sector preference: test weekday sector patterns

Sector groups:
  black:    rbfi, hcfi, ifi, jfi, jmfi
  nonferrous: cufi, alfi, znfi, aufi, agfi, nifi
  energy:   scfi, mfi, ptafi, bfi, fufi, egfi, pgfi, bcfi
  ags:      afi, mfi, yfi, cfi, srfi, cffi, whfi, rrfi, lrfi
  chem:     mafi, tafi, fgfi, sffi, lufi, vfi, ppfi, ebfi
  other:    everything else

Base: V157 Kitchen Sink sizing (dd*wr*regime), regime 0.5-1.5, max_corr=0.5,
      no SL, ov>0.3 id>0.3, top_n=3.
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

# ===================== SECTOR DEFINITIONS =====================
SECTOR_MEMBERS = {
    'black':      {'rbfi', 'hcfi', 'ifi', 'jfi', 'jmfi'},
    'nonferrous': {'cufi', 'alfi', 'znfi', 'aufi', 'agfi', 'nifi'},
    'energy':     {'scfi', 'mfi', 'ptafi', 'bfi', 'fufi', 'egfi', 'pgfi', 'bcfi'},
    'ags':        {'afi', 'mfi', 'yfi', 'cfi', 'srfi', 'cffi', 'whfi', 'rrfi', 'lrfi'},
    'chem':       {'mafi', 'tafi', 'fgfi', 'sffi', 'lufi', 'vfi', 'ppfi', 'ebfi'},
}
SECTOR_NAMES = ['black', 'nonferrous', 'energy', 'ags', 'chem']

# Defensive sectors (lower beta) / Aggressive sectors (higher beta)
DEFENSIVE_SECTORS = {'ags', 'chem'}
AGGRESSIVE_SECTORS = {'energy', 'black'}

# Month-of-year sector biases (hypothesis-driven)
# Winter (Dec-Feb): energy demand up, black steel production
# Spring (Mar-May): ags planting season
# Summer (Jun-Aug): energy peak demand
# Autumn (Sep-Nov): harvest, chem demand
MONTH_SECTOR_MAP = {
    1:  ['energy', 'black'],      # Winter: heating demand
    2:  ['energy', 'black'],
    3:  ['ags', 'nonferrous'],    # Spring: planting, construction restart
    4:  ['ags', 'nonferrous'],
    5:  ['ags', 'chem'],
    6:  ['energy', 'nonferrous'], # Summer: peak energy
    7:  ['energy', 'nonferrous'],
    8:  ['energy', 'black'],
    9:  ['ags', 'chem'],          # Autumn: harvest
    10: ['ags', 'chem'],
    11: ['black', 'nonferrous'],  # Pre-winter restocking
    12: ['energy', 'black'],
}

# Day-of-week sector preferences (hypothesis: Monday gap fill, Friday positioning)
DOW_SECTOR_MAP = {
    0: ['energy', 'nonferrous'],  # Monday: catch weekend gaps
    1: ['black', 'chem'],
    2: ['ags', 'nonferrous'],
    3: ['energy', 'black'],
    4: ['ags', 'chem'],           # Friday: positioning for weekend
}


def annual_return(final, initial, n_days):
    if final <= 0 or initial <= 0 or n_days <= 0: return -100.0
    return (final / initial) ** (1.0 / (n_days / 252)) * 100 - 100


def main():
    print("=" * 130)
    print("  V168 — Sector Rotation & Timing Effects")
    print("=" * 130)
    t_start = time.time()
    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)
    print(f"  {NS} commodities, {ND} days")

    # ===================== BUILD SECTOR INDEX =====================
    # sym_to_sector[si] -> sector name
    sym_to_sector = {}
    for si in range(NS):
        s = syms[si]
        assigned = False
        for sec_name, sec_set in SECTOR_MEMBERS.items():
            if s in sec_set:
                sym_to_sector[si] = sec_name
                assigned = True
                break
        if not assigned:
            sym_to_sector[si] = 'other'

    # sector_si[sector_name] -> list of si indices
    sector_si = {sec: [] for sec in SECTOR_NAMES}
    sector_si['other'] = []
    for si in range(NS):
        sec = sym_to_sector[si]
        sector_si[sec].append(si)

    print(f"  Sector distribution:")
    for sec in SECTOR_NAMES + ['other']:
        print(f"    {sec:12s}: {len(sector_si[sec]):3d} commodities")

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

    OV_GAP = np.full((NS, ND), np.nan)
    ID_RET = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(ND):
            o, c = O[si, di], C[si, di]
            if not np.isnan(o) and not np.isnan(c):
                if di > 0 and not np.isnan(C[si, di-1]) and C[si, di-1] > 0:
                    OV_GAP[si, di] = (o - C[si, di-1]) / C[si, di-1] * 100
                if o > 0: ID_RET[si, di] = (c - o) / o * 100

    # ===================== SECTOR-LEVEL INDICATORS =====================
    print("  Computing sector-level indicators...", flush=True)

    # SECTOR_AVG_ROC5[sector_idx, di] = average ROC5 across commodities in that sector
    SECTOR_AVG_ROC5 = np.full((len(SECTOR_NAMES), ND), np.nan)
    SECTOR_BREADTH = np.full((len(SECTOR_NAMES), ND), np.nan)  # % with positive ROC5
    for sec_i, sec_name in enumerate(SECTOR_NAMES):
        sis = sector_si[sec_name]
        if not sis: continue
        for di in range(ND):
            rocs = [ROC5[si, di] for si in sis if not np.isnan(ROC5[si, di])]
            if rocs:
                SECTOR_AVG_ROC5[sec_i, di] = np.mean(rocs)
                SECTOR_BREADTH[sec_i, di] = sum(1 for r in rocs if r > 0) / len(rocs)

    # BREADTH (market-wide) for regime
    BREADTH_MKT = np.full(ND, np.nan)
    for di in range(5, ND):
        pos_count = 0; total = 0
        for si in range(NS):
            r = ROC5[si, di]
            if not np.isnan(r):
                total += 1
                if r > 0: pos_count += 1
        if total > 0:
            BREADTH_MKT[di] = pos_count / total

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

    # Date helpers for calendar effects
    DATE_MONTH = np.array([d.month for d in dates], dtype=np.int32)
    DATE_DOW = np.array([d.weekday() for d in dates], dtype=np.int32)

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

    # ===================== HELPER: Compute composite regime score =====================
    def compute_composite(di, daily_eq, high_water, perf_window=20):
        scores = []
        bth = BREADTH_MKT[di]
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

    # ===================== HELPER: WR-adaptive sizing =====================
    def wr_size(trades, window=20):
        if len(trades) < window:
            return 1.0
        recent = trades[-window:]
        wr = np.mean([1 if t > 0 else 0 for t in recent])
        if wr > 0.65:
            return 1.3
        elif wr >= 0.50:
            return 1.0
        else:
            return 0.5

    # ===================== HELPER: Get allowed sectors for a given day =====================
    def get_allowed_sectors(di, pv, high_water, sector_method, max_sectors):
        """
        Returns set of allowed sector names, or None if no filtering (all allowed).
        """
        if sector_method == 'none':
            return None  # all sectors allowed

        if sector_method == 'rank':
            # Rank sectors by avg ROC5, return top max_sectors
            sector_roc = []
            for sec_i, sec_name in enumerate(SECTOR_NAMES):
                avg = SECTOR_AVG_ROC5[sec_i, di]
                if not np.isnan(avg):
                    sector_roc.append((avg, sec_name))
            if not sector_roc:
                return None
            sector_roc.sort(key=lambda x: -x[0])
            allowed = set(name for _, name in sector_roc[:max_sectors])
            return allowed

        if sector_method == 'breadth':
            # Only sectors where >50% of commodities have positive ROC5
            allowed = set()
            for sec_i, sec_name in enumerate(SECTOR_NAMES):
                br = SECTOR_BREADTH[sec_i, di]
                if not np.isnan(br) and br > 0.5:
                    allowed.add(sec_name)
            if not allowed:
                return None  # fallback: all sectors
            # If more than max_sectors, keep those with highest breadth
            if len(allowed) > max_sectors:
                br_list = [(SECTOR_BREADTH[SECTOR_NAMES.index(s), di], s)
                           for s in allowed if not np.isnan(SECTOR_BREADTH[SECTOR_NAMES.index(s), di])]
                br_list.sort(key=lambda x: -x[0])
                allowed = set(name for _, name in br_list[:max_sectors])
            return allowed

        if sector_method == 'rotation':
            # Defensive (ags/chem) when in drawdown, aggressive (energy/black) near HWM
            if high_water > 0:
                dd_pct = (pv - high_water) / high_water
            else:
                dd_pct = 0
            if dd_pct < -0.10:
                # Deep drawdown: defensive only, pick top max_sectors by breadth
                candidates = []
                for s in DEFENSIVE_SECTORS:
                    sec_i = SECTOR_NAMES.index(s)
                    br = SECTOR_BREADTH[sec_i, di]
                    if not np.isnan(br):
                        candidates.append((br, s))
                candidates.sort(key=lambda x: -x[0])
                allowed = set(name for _, name in candidates[:max_sectors])
            elif dd_pct < -0.05:
                # Mild drawdown: mix defensive + some aggressive
                def_breadth = []
                for s in DEFENSIVE_SECTORS:
                    sec_i = SECTOR_NAMES.index(s)
                    br = SECTOR_BREADTH[sec_i, di]
                    if not np.isnan(br):
                        def_breadth.append((br, s))
                def_breadth.sort(key=lambda x: -x[0])
                agg_breadth = []
                for s in AGGRESSIVE_SECTORS:
                    sec_i = SECTOR_NAMES.index(s)
                    br = SECTOR_BREADTH[sec_i, di]
                    if not np.isnan(br):
                        agg_breadth.append((br, s))
                agg_breadth.sort(key=lambda x: -x[0])
                allowed = set()
                for _, s in def_breadth[:1]:
                    allowed.add(s)
                if agg_breadth:
                    allowed.add(agg_breadth[0][1])
                # Trim to max_sectors
                if len(allowed) > max_sectors:
                    all_cands = []
                    for s in allowed:
                        sec_i = SECTOR_NAMES.index(s)
                        br = SECTOR_BREADTH[sec_i, di]
                        if not np.isnan(br):
                            all_cands.append((br, s))
                    all_cands.sort(key=lambda x: -x[0])
                    allowed = set(name for _, name in all_cands[:max_sectors])
            else:
                # Near or above HWM: aggressive sectors, pick top max_sectors by breadth
                candidates = []
                for s in AGGRESSIVE_SECTORS:
                    sec_i = SECTOR_NAMES.index(s)
                    br = SECTOR_BREADTH[sec_i, di]
                    if not np.isnan(br):
                        candidates.append((br, s))
                candidates.sort(key=lambda x: -x[0])
                allowed = set(name for _, name in candidates[:max_sectors])
            return allowed if allowed else None

        if sector_method == 'month':
            # Month-of-year sector preference
            month = DATE_MONTH[di]
            preferred = MONTH_SECTOR_MAP.get(month, SECTOR_NAMES)
            # Take top max_sectors by breadth within the preferred set
            br_list = []
            for s in preferred:
                if s in SECTOR_NAMES:
                    sec_i = SECTOR_NAMES.index(s)
                    br = SECTOR_BREADTH[sec_i, di]
                    if not np.isnan(br):
                        br_list.append((br, s))
            br_list.sort(key=lambda x: -x[0])
            allowed = set(name for _, name in br_list[:max_sectors])
            return allowed if allowed else None

        if sector_method == 'dow':
            # Day-of-week sector preference
            dow = DATE_DOW[di]
            preferred = DOW_SECTOR_MAP.get(dow, SECTOR_NAMES)
            # Take top max_sectors by breadth within the preferred set
            br_list = []
            for s in preferred:
                if s in SECTOR_NAMES:
                    sec_i = SECTOR_NAMES.index(s)
                    br = SECTOR_BREADTH[sec_i, di]
                    if not np.isnan(br):
                        br_list.append((br, s))
            br_list.sort(key=lambda x: -x[0])
            allowed = set(name for _, name in br_list[:max_sectors])
            return allowed if allowed else None

        return None

    # ===================== UNIFIED BACKTEST ENGINE =====================
    def backtest(start_di=MIN_TRAIN, end_di=None,
                 sector_method='none', max_sectors=2,
                 max_corr=0.5,
                 dd_tiers=None,
                 regime_lo=0.5, regime_hi=1.5,
                 sl_pct=0.0,
                 hold=1, top_n=3):
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

            # --- Stop-loss check ---
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
            wr_mult_val = wr_size(trades, window=20)
            composite = compute_composite(di, daily_eq, high_water)
            regime_mult = regime_lo + composite * (regime_hi - regime_lo)
            pos_size = dd_sz * wr_mult_val * regime_mult
            pos_size = max(0.05, min(0.99, pos_size))

            # --- Enter positions ---
            if len(positions) >= top_n: continue
            edi = di + 1
            if edi >= end_di: continue

            held_si = set(p['si'] for p in positions)

            # Get allowed sectors for this day
            allowed_sectors = get_allowed_sectors(di, pv, high_water,
                                                   sector_method, max_sectors)

            # Get best V121 and best Union signal
            cands_v121 = sig_v121(di, edi)
            cands_union = sig_union(di, edi)

            # Filter candidates by allowed sectors
            if allowed_sectors is not None:
                cands_v121 = [(sc, s, ep, sig) for sc, s, ep, sig in cands_v121
                              if sym_to_sector.get(s) in allowed_sectors]
                cands_union = [(sc, s, ep, sig) for sc, s, ep, sig in cands_union
                               if sym_to_sector.get(s) in allowed_sectors]

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

            cash_snapshot = cash
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
        print(f"  {label:85s} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f} | R/M={ratio:.2f} | N={r['n']:4d}")

    def walk_forward(sector_method='none', max_sectors=2, max_corr=0.5,
                     dd_tiers=None, regime_lo=0.5, regime_hi=1.5,
                     sl_pct=0.0, hold=1, top_n=3, label=""):
        res = {}
        for yr in [2020, 2021, 2022, 2023, 2024, 2025]:
            ys = ye = None
            for di in range(ND):
                if dates[di].year == yr and ys is None: ys = di
                if dates[di].year == yr: ye = di + 1
            if ys is None: continue
            r = backtest(start_di=ys, end_di=ye,
                         sector_method=sector_method, max_sectors=max_sectors,
                         max_corr=max_corr, dd_tiers=dd_tiers,
                         regime_lo=regime_lo, regime_hi=regime_hi,
                         sl_pct=sl_pct, hold=hold, top_n=top_n)
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

    SECTOR_METHODS = ['none', 'rank', 'breadth', 'rotation', 'month', 'dow']
    MAX_SECTORS = [1, 2, 3, 4]

    # ===================== SECTION 0: BASELINE (no sector filter) =====================
    print("\n" + "=" * 120)
    print("  SECTION 0: BASELINE — V157 champion (no sector filter, aggro100)")
    print("=" * 120)

    r_base = backtest(sector_method='none', max_sectors=99,
                      dd_tiers=DD_TIERS['aggro100'],
                      regime_lo=0.5, regime_hi=1.5, max_corr=0.5,
                      sl_pct=0.0, top_n=3)
    pr(r_base, "BASELINE: V157 aggro100 noSL reg0.5-1.5 corr<0.5 top3")

    # ===================== SECTION 1: SECTOR MOMENTUM RANK =====================
    print("\n" + "=" * 120)
    print("  SECTION 1A: SECTOR MOMENTUM RANK (full period)")
    print("  Rank sectors by avg ROC5, trade only top-N sectors")
    print("=" * 120)

    all_results = []

    for ms in MAX_SECTORS:
        label = f"rank max_sec={ms} aggro100 noSL reg0.5-1.5 corr<0.5 top3"
        r = backtest(sector_method='rank', max_sectors=ms,
                     dd_tiers=DD_TIERS['aggro100'],
                     regime_lo=0.5, regime_hi=1.5, max_corr=0.5,
                     sl_pct=0.0, top_n=3)
        pr(r, label)
        all_results.append((r['ann'], r['mdd'], r['sharpe'], r['n'],
                            label, {'method': 'rank', 'max_sectors': ms}))

    # ===================== SECTION 1B: SECTOR BREADTH FILTER =====================
    print("\n" + "=" * 120)
    print("  SECTION 1B: SECTOR BREADTH FILTER (full period)")
    print("  Only trade in sectors where >50% of commodities have positive ROC5")
    print("=" * 120)

    for ms in MAX_SECTORS:
        label = f"breadth max_sec={ms} aggro100 noSL reg0.5-1.5 corr<0.5 top3"
        r = backtest(sector_method='breadth', max_sectors=ms,
                     dd_tiers=DD_TIERS['aggro100'],
                     regime_lo=0.5, regime_hi=1.5, max_corr=0.5,
                     sl_pct=0.0, top_n=3)
        pr(r, label)
        all_results.append((r['ann'], r['mdd'], r['sharpe'], r['n'],
                            label, {'method': 'breadth', 'max_sectors': ms}))

    # ===================== SECTION 1C: SECTOR ROTATION TIMING =====================
    print("\n" + "=" * 120)
    print("  SECTION 1C: SECTOR ROTATION TIMING (full period)")
    print("  Defensive (ags/chem) in drawdown, aggressive (energy/black) near HWM")
    print("=" * 120)

    label = "rotation aggro100 noSL reg0.5-1.5 corr<0.5 top3"
    r = backtest(sector_method='rotation', max_sectors=2,
                 dd_tiers=DD_TIERS['aggro100'],
                 regime_lo=0.5, regime_hi=1.5, max_corr=0.5,
                 sl_pct=0.0, top_n=3)
    pr(r, label)
    all_results.append((r['ann'], r['mdd'], r['sharpe'], r['n'],
                        label, {'method': 'rotation', 'max_sectors': 2}))

    # ===================== SECTION 1D: MONTH-OF-YEAR BIAS =====================
    print("\n" + "=" * 120)
    print("  SECTION 1D: MONTH-OF-YEAR SECTOR BIAS (full period)")
    print("  Seasonal sector preferences (e.g. energy in winter, ags in spring)")
    print("=" * 120)

    for ms in MAX_SECTORS:
        label = f"month max_sec={ms} aggro100 noSL reg0.5-1.5 corr<0.5 top3"
        r = backtest(sector_method='month', max_sectors=ms,
                     dd_tiers=DD_TIERS['aggro100'],
                     regime_lo=0.5, regime_hi=1.5, max_corr=0.5,
                     sl_pct=0.0, top_n=3)
        pr(r, label)
        all_results.append((r['ann'], r['mdd'], r['sharpe'], r['n'],
                            label, {'method': 'month', 'max_sectors': ms}))

    # ===================== SECTION 1E: DAY-OF-WEEK SECTOR PREFERENCE =====================
    print("\n" + "=" * 120)
    print("  SECTION 1E: DAY-OF-WEEK SECTOR PREFERENCE (full period)")
    print("  Weekday-specific sector allocation")
    print("=" * 120)

    for ms in MAX_SECTORS:
        label = f"dow max_sec={ms} aggro100 noSL reg0.5-1.5 corr<0.5 top3"
        r = backtest(sector_method='dow', max_sectors=ms,
                     dd_tiers=DD_TIERS['aggro100'],
                     regime_lo=0.5, regime_hi=1.5, max_corr=0.5,
                     sl_pct=0.0, top_n=3)
        pr(r, label)
        all_results.append((r['ann'], r['mdd'], r['sharpe'], r['n'],
                            label, {'method': 'dow', 'max_sectors': ms}))

    # ===================== SECTION 2: TOP FULL-PERIOD RESULTS =====================
    print("\n" + "=" * 120)
    print("  SECTION 2: TOP 15 FULL-PERIOD by annual return")
    print("=" * 120)

    all_results.sort(key=lambda x: -x[0])
    for i, (ann, mdd, sh, n, label, cfg) in enumerate(all_results[:15]):
        ratio = abs(ann / mdd) if mdd != 0 else 0
        print(f"  #{i+1:2d} | Ann={ann:+8.1f}% | MDD={mdd:6.1f}% | Sh={sh:4.2f} | R/M={ratio:.2f} | N={n:4d} | {label}")

    # ===================== SECTION 3: WALK-FORWARD VALIDATION =====================
    print("\n" + "=" * 120)
    print("  SECTION 3: WALK-FORWARD VALIDATION — ALL CONFIGS")
    print("=" * 120)

    wf_all = {}
    for ann, mdd, sh, n, label, cfg in all_results:
        if label not in wf_all:
            wf_res = walk_forward(sector_method=cfg['method'],
                                  max_sectors=cfg['max_sectors'],
                                  max_corr=0.5,
                                  dd_tiers=DD_TIERS['aggro100'],
                                  regime_lo=0.5, regime_hi=1.5,
                                  sl_pct=0.0, hold=1, top_n=3,
                                  label=label)
            wf_all[label] = (wf_res, cfg)
            print_wf(wf_res, label)

    # Also WF for baseline
    label_base = "BASELINE: none aggro100 noSL reg0.5-1.5 corr<0.5 top3"
    if label_base not in wf_all:
        wf_res = walk_forward(sector_method='none', max_sectors=99,
                              max_corr=0.5,
                              dd_tiers=DD_TIERS['aggro100'],
                              regime_lo=0.5, regime_hi=1.5,
                              sl_pct=0.0, hold=1, top_n=3,
                              label=label_base)
        wf_all[label_base] = (wf_res, {'method': 'none', 'max_sectors': 99})
        print_wf(wf_res, label_base)

    # ===================== SECTION 4: RANK BY WF AVERAGE =====================
    print("\n" + "=" * 120)
    print("  SECTION 4: ALL CONFIGS RANKED BY WF AVERAGE")
    print("=" * 120)

    wf_ranked = []
    for label, (wf_res, cfg) in wf_all.items():
        avg_ann = np.mean([r['ann'] for r in wf_res.values()])
        worst_mdd = min(r['mdd'] for r in wf_res.values())
        best_ann = max(r['ann'] for r in wf_res.values())
        pos = sum(1 for r in wf_res.values() if r['ann'] > 0)
        total_n = sum(r['n'] for r in wf_res.values())
        avg_wr = np.mean([r['wr'] for r in wf_res.values()])
        wf_ranked.append((avg_ann, worst_mdd, best_ann, pos, total_n, avg_wr, label, cfg, wf_res))

    wf_ranked.sort(key=lambda x: -x[0])

    for i, (avg_ann, worst_mdd, best_ann, pos, total_n, avg_wr, label, cfg, wf_res) in enumerate(wf_ranked):
        ws = " | ".join([f"{yr}:{r['ann']:+.0f}%/{r['mdd']:.0f}%"
                         for yr, r in sorted(wf_res.items())])
        print(f"\n  #{i+1} WF AVG={avg_ann:+.0f}% | WorstMDD={worst_mdd:.0f}% | {pos}/6 pos | TotalTrades={total_n} | AvgWR={avg_wr:.1f}%")
        print(f"     {label}")
        print(f"     {ws}")

    # ===================== SECTION 5: BEST RISK-ADJUSTED =====================
    print("\n" + "=" * 120)
    print("  SECTION 5: BEST RISK-ADJUSTED (WF avg / |worst MDD|)")
    print("=" * 120)

    wf_ra = sorted(wf_ranked, key=lambda x: -abs(x[0] / x[1]) if x[1] != 0 else 0)
    for i, (avg_ann, worst_mdd, best_ann, pos, total_n, avg_wr, label, cfg, wf_res) in enumerate(wf_ra[:10]):
        ratio = abs(avg_ann / worst_mdd) if worst_mdd != 0 else 0
        ws = " | ".join([f"{yr}:{r['ann']:+.0f}%/{r['mdd']:.0f}%"
                         for yr, r in sorted(wf_res.items())])
        print(f"\n  #{i+1} WF AVG={avg_ann:+.0f}% | WorstMDD={worst_mdd:.0f}% | R/M={ratio:.2f} | {pos}/6 pos")
        print(f"     {label}")
        print(f"     {ws}")

    # ===================== SECTION 6: SECTOR METHOD COMPARISON =====================
    print("\n" + "=" * 120)
    print("  SECTION 6: BEST PER SECTOR METHOD (WF avg)")
    print("=" * 120)

    for method in SECTOR_METHODS:
        method_results = [(avg, wmdd, bann, pos, tn, awr, lbl, cfg, wfr)
                          for avg, wmdd, bann, pos, tn, awr, lbl, cfg, wfr in wf_ranked
                          if cfg['method'] == method]
        if not method_results:
            print(f"\n  {method}: no results")
            continue
        method_results.sort(key=lambda x: -x[0])
        best = method_results[0]
        avg_ann, worst_mdd, best_ann, pos, total_n, avg_wr, label, cfg, wf_res = best
        ws = " | ".join([f"{yr}:{r['ann']:+.0f}%/{r['mdd']:.0f}%"
                         for yr, r in sorted(wf_res.items())])
        print(f"\n  {method:12s} best: WF AVG={avg_ann:+.0f}% | WorstMDD={worst_mdd:.0f}% | {pos}/6 pos | max_sec={cfg.get('max_sectors', 'n/a')}")
        print(f"     {label}")
        print(f"     {ws}")

    # ===================== SECTION 7: DELTA vs BASELINE =====================
    print("\n" + "=" * 120)
    print("  SECTION 7: DELTA vs BASELINE — Does sector rotation add alpha?")
    print("=" * 120)

    base_wf = None
    for label, (wf_res, cfg) in wf_all.items():
        if cfg['method'] == 'none':
            base_wf = wf_res
            base_avg = np.mean([r['ann'] for r in wf_res.values()])
            base_mdd = min(r['mdd'] for r in wf_res.values())
            break

    if base_wf:
        print(f"  BASELINE WF AVG = {base_avg:+.0f}%, WorstMDD = {base_mdd:.0f}%")
        print()
        for avg_ann, worst_mdd, best_ann, pos, total_n, avg_wr, label, cfg, wf_res in wf_ranked:
            if cfg['method'] == 'none': continue
            delta_ann = avg_ann - base_avg
            delta_mdd = worst_mdd - base_mdd
            # Per-year deltas
            yr_deltas = []
            for yr in sorted(base_wf.keys()):
                if yr in wf_res:
                    d = wf_res[yr]['ann'] - base_wf[yr]['ann']
                    yr_deltas.append(f"{yr}:{d:+.0f}%")
            yr_str = " | ".join(yr_deltas)
            print(f"  {label}")
            print(f"    WF AVG delta: {delta_ann:+.0f}% | MDD delta: {delta_mdd:+.0f}% | {pos}/6 pos")
            print(f"    Per-year: {yr_str}")
    else:
        print("  No baseline found for comparison.")

    # ===================== SECTION 8: COMBINED SECTOR FILTERS =====================
    print("\n" + "=" * 120)
    print("  SECTION 8: COMBINED SECTOR FILTERS (full period + WF)")
    print("  Test rank+breadth combo, rank+month combo, etc.")
    print("=" * 120)

    # We'll test combined methods by modifying the sector filter
    # rank_and_breadth: must pass both rank and breadth filters
    # rank_or_breadth: pass either filter
    # month_and_rank: must pass both month preference AND rank
    combined_configs = [
        ('rank+breadth', 'rank', 2),   # re-use rank but also require breadth
        ('rank+breadth', 'rank', 3),
    ]

    # For combined, we override get_allowed_sectors logic inline
    def backtest_combined(start_di=MIN_TRAIN, end_di=None,
                          max_corr=0.5, dd_tiers=None,
                          regime_lo=0.5, regime_hi=1.5,
                          sl_pct=0.0, hold=1, top_n=3,
                          combo='rank_and_breadth', max_sectors=2):
        """Backtest with combined sector filters."""
        if end_di is None: end_di = ND
        if dd_tiers is None:
            dd_tiers = DD_TIERS['aggro100']

        cash = float(CASH0)
        positions = []
        trades = []
        daily_eq = []
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

            dd_sz = dd_size(pv, high_water, dd_tiers)
            wr_mult_val = wr_size(trades, window=20)
            composite = compute_composite(di, daily_eq, high_water)
            regime_mult = regime_lo + composite * (regime_hi - regime_lo)
            pos_size = dd_sz * wr_mult_val * regime_mult
            pos_size = max(0.05, min(0.99, pos_size))

            if len(positions) >= top_n: continue
            edi = di + 1
            if edi >= end_di: continue

            held_si = set(p['si'] for p in positions)

            # Combined sector filter
            allowed_sectors = None
            if combo == 'rank_and_breadth':
                # Must be in top-N by rank AND have breadth > 50%
                # Rank
                sector_roc = []
                for sec_i, sec_name in enumerate(SECTOR_NAMES):
                    avg = SECTOR_AVG_ROC5[sec_i, di]
                    if not np.isnan(avg):
                        sector_roc.append((avg, sec_name))
                if sector_roc:
                    sector_roc.sort(key=lambda x: -x[0])
                    rank_set = set(name for _, name in sector_roc[:max_sectors])
                else:
                    rank_set = set(SECTOR_NAMES)
                # Breadth
                breadth_set = set()
                for sec_i, sec_name in enumerate(SECTOR_NAMES):
                    br = SECTOR_BREADTH[sec_i, di]
                    if not np.isnan(br) and br > 0.5:
                        breadth_set.add(sec_name)
                allowed_sectors = rank_set & breadth_set if (rank_set and breadth_set) else None

            elif combo == 'rank_or_breadth':
                sector_roc = []
                for sec_i, sec_name in enumerate(SECTOR_NAMES):
                    avg = SECTOR_AVG_ROC5[sec_i, di]
                    if not np.isnan(avg):
                        sector_roc.append((avg, sec_name))
                rank_set = set(name for _, name in sector_roc[:max_sectors]) if sector_roc else set()
                breadth_set = set()
                for sec_i, sec_name in enumerate(SECTOR_NAMES):
                    br = SECTOR_BREADTH[sec_i, di]
                    if not np.isnan(br) and br > 0.5:
                        breadth_set.add(sec_name)
                allowed_sectors = rank_set | breadth_set if (rank_set or breadth_set) else None

            elif combo == 'month_and_rank':
                month = DATE_MONTH[di]
                preferred = set(MONTH_SECTOR_MAP.get(month, SECTOR_NAMES))
                sector_roc = []
                for sec_i, sec_name in enumerate(SECTOR_NAMES):
                    avg = SECTOR_AVG_ROC5[sec_i, di]
                    if not np.isnan(avg):
                        sector_roc.append((avg, sec_name))
                if sector_roc:
                    sector_roc.sort(key=lambda x: -x[0])
                    rank_set = set(name for _, name in sector_roc[:max_sectors])
                else:
                    rank_set = set(SECTOR_NAMES)
                allowed_sectors = preferred & rank_set if (preferred and rank_set) else None

            elif combo == 'rotation_and_rank':
                # Rotation base + rank to pick within allowed
                if high_water > 0:
                    dd_pct = (pv - high_water) / high_water
                else:
                    dd_pct = 0
                if dd_pct < -0.10:
                    rot_set = DEFENSIVE_SECTORS
                else:
                    rot_set = AGGRESSIVE_SECTORS
                # Within rotation set, rank by ROC5
                sector_roc = []
                for sec_name in rot_set:
                    sec_i = SECTOR_NAMES.index(sec_name)
                    avg = SECTOR_AVG_ROC5[sec_i, di]
                    if not np.isnan(avg):
                        sector_roc.append((avg, sec_name))
                if sector_roc:
                    sector_roc.sort(key=lambda x: -x[0])
                    allowed_sectors = set(name for _, name in sector_roc[:max_sectors])
                else:
                    allowed_sectors = rot_set

            cands_v121 = sig_v121(di, edi)
            cands_union = sig_union(di, edi)

            if allowed_sectors is not None:
                cands_v121 = [(sc, s, ep, sig) for sc, s, ep, sig in cands_v121
                              if sym_to_sector.get(s) in allowed_sectors]
                cands_union = [(sc, s, ep, sig) for sc, s, ep, sig in cands_union
                               if sym_to_sector.get(s) in allowed_sectors]

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

            cash_snapshot = cash
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

    combo_configs = [
        ('rank_and_breadth', 2),
        ('rank_and_breadth', 3),
        ('rank_and_breadth', 4),
        ('rank_or_breadth',  2),
        ('rank_or_breadth',  3),
        ('month_and_rank',   2),
        ('month_and_rank',   3),
        ('rotation_and_rank', 1),
        ('rotation_and_rank', 2),
    ]

    combo_results = []
    for combo, ms in combo_configs:
        label = f"{combo} ms={ms} aggro100 noSL reg0.5-1.5 corr<0.5 top3"
        r = backtest_combined(combo=combo, max_sectors=ms,
                              dd_tiers=DD_TIERS['aggro100'],
                              regime_lo=0.5, regime_hi=1.5,
                              max_corr=0.5, sl_pct=0.0, top_n=3)
        pr(r, label)
        combo_results.append((r['ann'], r['mdd'], r['sharpe'], r['n'],
                              label, {'combo': combo, 'max_sectors': ms}))

    # WF for top combos
    print("\n  --- WF for top combos ---")
    combo_results.sort(key=lambda x: -x[0])
    for ann, mdd, sh, n, label, cfg in combo_results[:5]:
        dd_t = DD_TIERS['aggro100']
        # Use combined backtest for WF
        wf_res = {}
        for yr in [2020, 2021, 2022, 2023, 2024, 2025]:
            ys = ye = None
            for di in range(ND):
                if dates[di].year == yr and ys is None: ys = di
                if dates[di].year == yr: ye = di + 1
            if ys is None: continue
            r = backtest_combined(start_di=ys, end_di=ye,
                                  combo=cfg['combo'],
                                  max_sectors=cfg['max_sectors'],
                                  dd_tiers=dd_t,
                                  max_corr=0.5, sl_pct=0.0, top_n=3)
            wf_res[yr] = r
        wf_all[label] = (wf_res, cfg)
        print_wf(wf_res, label)

    # ===================== SECTION 9: SECTOR PERFORMANCE ANALYSIS =====================
    print("\n" + "=" * 120)
    print("  SECTION 9: SECTOR-LEVEL PERFORMANCE ANALYSIS")
    print("  How each sector contributes to the baseline strategy")
    print("=" * 120)

    # Count trades per sector from baseline
    for sec in SECTOR_NAMES:
        sis = set(sector_si[sec])
        # Simple count of how often sector commodities generate signals
        signal_counts = {2020: 0, 2021: 0, 2022: 0, 2023: 0, 2024: 0, 2025: 0}
        for yr in signal_counts:
            for di in range(ND):
                if dates[di].year != yr: continue
                for si in sis:
                    roc = ROC5[si, di]; zs = ZSCORE[si, di]
                    if not np.isnan(roc) and not np.isnan(zs) and roc > 1.0 and zs > 1.5:
                        rp = ROC5[si, di-1] if di > 0 else np.nan
                        if np.isnan(rp) or roc > rp:
                            signal_counts[yr] += 1
        total = sum(signal_counts.values())
        yr_str = " | ".join(f"{yr}:{c}" for yr, c in sorted(signal_counts.items()))
        print(f"  {sec:12s}: total V121 signals = {total:5d} | {yr_str}")

    # ===================== FINAL SUMMARY =====================
    print("\n" + "=" * 120)
    print("  FINAL SUMMARY")
    print("=" * 120)

    # Re-rank all WF results
    final_ranked = []
    for label, (wf_res, cfg) in wf_all.items():
        avg_ann = np.mean([r['ann'] for r in wf_res.values()])
        worst_mdd = min(r['mdd'] for r in wf_res.values())
        pos = sum(1 for r in wf_res.values() if r['ann'] > 0)
        final_ranked.append((avg_ann, worst_mdd, pos, label, cfg, wf_res))

    final_ranked.sort(key=lambda x: -x[0])

    # Find baseline for delta
    base_avg = None
    for avg_ann, worst_mdd, pos, label, cfg, wf_res in final_ranked:
        if cfg.get('method') == 'none' or cfg.get('method') is None:
            base_avg = avg_ann
            break
    if base_avg is None and final_ranked:
        base_avg = final_ranked[0][0]

    print(f"\n  V168 Sector Rotation & Timing Tests")
    print(f"  Baseline WF avg: {base_avg:+.0f}%")
    print()

    # Best overall
    if final_ranked:
        avg_ann, worst_mdd, pos, label, cfg, wf_res = final_ranked[0]
        delta = avg_ann - base_avg if base_avg else 0
        ws = " | ".join([f"{yr}:{r['ann']:+.0f}%/{r['mdd']:.0f}%"
                         for yr, r in sorted(wf_res.items())])
        print(f"  BEST OVERALL:")
        print(f"    {label}")
        print(f"    WF AVG={avg_ann:+.0f}% (delta={delta:+.0f}%) | WorstMDD={worst_mdd:.0f}% | {pos}/6 pos")
        print(f"    {ws}")

    # Best per method
    print(f"\n  Best per method:")
    seen_methods = set()
    for avg_ann, worst_mdd, pos, label, cfg, wf_res in final_ranked:
        method = cfg.get('method', cfg.get('combo', '?'))
        if method in seen_methods: continue
        seen_methods.add(method)
        delta = avg_ann - base_avg if base_avg else 0
        print(f"    {str(method):20s}: WF AVG={avg_ann:+.0f}% (delta={delta:+.0f}%) | MDD={worst_mdd:.0f}% | {pos}/6 pos | {label}")

    # Conclusion
    print(f"\n  CONCLUSION:")
    beat_base = [(avg, wmdd, pos, lbl, cfg, wfr)
                 for avg, wmdd, pos, lbl, cfg, wfr in final_ranked
                 if avg > base_avg and cfg.get('method') != 'none']
    if beat_base:
        print(f"    {len(beat_base)} sector rotation configs BEAT the baseline (WF avg).")
        best_delta = max(beat_base, key=lambda x: x[0])
        print(f"    Best delta: {best_delta[0] - base_avg:+.0f}% by {best_delta[3]}")
    else:
        print(f"    No sector rotation config beats the baseline.")
        print(f"    Sector rotation does NOT add alpha over V157 champion on 68 commodities.")

    print(f"\n  Elapsed: {time.time()-t_start:.0f}s")
    print("=" * 120)


if __name__ == '__main__':
    main()
