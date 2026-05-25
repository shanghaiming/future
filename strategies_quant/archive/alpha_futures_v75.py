"""
Alpha Futures V75 -- Regime-Adaptive Lookback
=============================================
V74 champion: extended groups, 44 commodities, LB=1, 1-day hold => +2185% annual.

Hypothesis: Dynamic lookback based on volatility regime improves Sharpe.
  - High vol regime => LB=1 (faster signal, capture rapid mean reversion)
  - Low vol regime  => LB=3 (slower signal, more stable trend)
  - Medium vol      => LB=2

Four modes tested:
  A: Regime-switching (discrete LB selection based on vol percentile)
  B: Vol-weighted (continuous blend of LB=1,2,3 signals)
  C: Vol-scaled position sizing (full lots in low vol, reduced in high vol)
  D: Vol filter (skip trades when vol is in extreme percentiles)

Walk-forward: 6 windows (2020-2025), reset cash at each window start.
"""
import sys, os, time, warnings
import numpy as np
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

# Extended group map (same as V74 champion)
GROUP_MAP = {}
# Ferrous
for s in ['rbfi', 'hcfi', 'ifi', 'jfi', 'jmfi']:
    GROUP_MAP[s] = 'ferrous'
# Nonferrous
for s in ['cufi', 'alfi', 'znfi', 'nifi', 'pbfi', 'snfi', 'ssfi', 'sffi']:
    GROUP_MAP[s] = 'nonferrous'
# Precious
for s in ['aufi', 'agfi']:
    GROUP_MAP[s] = 'precious'
# Oils
for s in ['afi', 'mfi', 'yfi', 'pfi', 'cfi', 'csfi', 'rrfi', 'lrfi']:
    GROUP_MAP[s] = 'oils'
# Energy
for s in ['scfi', 'mafi', 'bfi', 'fufi', 'pgfi', 'ebfi', 'fbfi']:
    GROUP_MAP[s] = 'energy'
# Chemical
for s in ['ppfi', 'vfi', 'egfi', 'srfi', 'tafi', 'fgfi', 'lfi']:
    GROUP_MAP[s] = 'chemical'
# Soft
for s in ['whfi', 'apfi', 'cjfi', 'oifi', 'rmfi', 'srfi', 'cffi']:
    GROUP_MAP[s] = 'soft'
# Livestock
for s in ['jdfi', 'lhfi', 'pkfi']:
    GROUP_MAP[s] = 'livestock'


def annual_return(final, initial, n_days):
    if final <= 0 or initial <= 0 or n_days <= 0:
        return -100.0
    yrs = n_days / 252
    return (final / initial) ** (1.0 / yrs) * 100 - 100


def main():
    t_start = time.time()
    print("=" * 98)
    print("V75 -- Regime-Adaptive Lookback")
    print("=" * 98)

    # Load data
    print("\n[Data] Loading...", flush=True)
    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)
    years = sorted(set(d.year for d in dates))
    print(f"  {NS} commodities, {ND} days, years: {years}")

    # ================================================================
    # PRECOMPUTE MOMENTUM AT MULTIPLE LOOKBACKS
    # ================================================================
    print("\n[Signals] Computing momentum at LB=1,2,3...", flush=True)
    t0 = time.time()

    all_lbs = [1, 2, 3]
    mom = {}
    for lag in all_lbs:
        m = np.full((NS, ND), np.nan)
        for si in range(NS):
            for di in range(lag, ND):
                cn = C[si, di]
                cp = C[si, di - lag]
                if not np.isnan(cn) and not np.isnan(cp) and cp > 0:
                    m[si, di] = (cn - cp) / cp
        mom[lag] = m

    # Compute group momentum for each LB
    def compute_grp_mom(gmap):
        gm_map = {}
        for si in range(NS):
            g = gmap.get(syms[si])
            if g:
                gm_map.setdefault(g, []).append(si)

        result = {}
        for lag in all_lbs:
            gm = np.full((NS, ND), np.nan)
            for grp, members in gm_map.items():
                for di in range(lag, ND):
                    for sj in members:
                        ms = [mom[lag][sk, di] for sk in members
                              if sk != sj and not np.isnan(mom[lag][sk, di])]
                        if ms:
                            gm[sj, di] = np.mean(ms)
            result[lag] = gm
        return result, gm_map

    grp, grp_members = compute_grp_mom(GROUP_MAP)

    # Compute divergence (group_avg - own) at each LB
    # Positive divergence => commodity lagging group => buy signal
    div = {}
    for lag in all_lbs:
        d = np.full((NS, ND), np.nan)
        for si in range(NS):
            for di in range(ND):
                own = mom[lag][si, di]
                g = grp[lag][si, di]
                if not np.isnan(own) and not np.isnan(g):
                    d[si, di] = g - own
        div[lag] = d

    print(f"  Momentum + divergence computed ({time.time()-t0:.1f}s)")

    # ================================================================
    # PRECOMPUTE VOLATILITY REGIME
    # ================================================================
    print("\n[Vol] Computing volatility regimes...", flush=True)
    t0 = time.time()

    VOL_WINDOW = 20   # rolling vol window (days)
    VOL_LOOKBACK = 60  # percentile lookback (days)

    # 20-day rolling volatility (std of daily returns)
    ret = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(1, ND):
            if not np.isnan(C[si, di]) and not np.isnan(C[si, di-1]) and C[si, di-1] > 0:
                ret[si, di] = (C[si, di] - C[si, di-1]) / C[si, di-1]

    vol_20 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(VOL_WINDOW, ND):
            window = ret[si, di-VOL_WINDOW:di]
            valid = window[~np.isnan(window)]
            if len(valid) >= VOL_WINDOW // 2:
                vol_20[si, di] = np.std(valid)

    # Vol percentile rank: where does today's vol sit vs past VOL_LOOKBACK days?
    # vol_rank[si, di] in [0, 1] -- 1 = highest vol in past 60 days
    vol_rank = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(VOL_LOOKBACK, ND):
            past = vol_20[si, di-VOL_LOOKBACK:di]
            valid = past[~np.isnan(past)]
            if len(valid) >= 10 and not np.isnan(vol_20[si, di]):
                vol_rank[si, di] = np.searchsorted(np.sort(valid), vol_20[si, di]) / len(valid)

    print(f"  Vol regimes computed ({time.time()-t0:.1f}s)")

    # ================================================================
    # TRADEABLE SYMBOLS
    # ================================================================
    trade_sis = [si for si in range(NS) if GROUP_MAP.get(syms[si])]
    print(f"  Tradeable commodities: {len(trade_sis)}")

    # ================================================================
    # BACKTEST ENGINE
    # ================================================================
    def run_backtest(config, wf_test_year=None):
        """
        Config:
            mode: 'fixed' | 'regime_switch' | 'vol_weighted' | 'vol_scaled' | 'vol_filter'
            fixed_lb: int (for mode='fixed')
            threshold: float
            top_n: int
            vol_lo: float (low vol percentile cutoff, e.g. 0.3)
            vol_hi: float (high vol percentile cutoff, e.g. 0.7)
            vol_filter_lo: float (skip below this percentile)
            vol_filter_hi: float (skip above this percentile)
            vol_scale_factor: float (scale in high vol, e.g. 0.5)
        """
        mode = config['mode']
        threshold = config['threshold']
        top_n = config['top_n']
        vol_lo = config.get('vol_lo', 0.3)
        vol_hi = config.get('vol_hi', 0.7)
        vol_filter_lo = config.get('vol_filter_lo', 0.1)
        vol_filter_hi = config.get('vol_filter_hi', 0.9)
        vol_scale_factor = config.get('vol_scale_factor', 0.5)
        fixed_lb = config.get('fixed_lb', 1)

        # Date range setup
        wf_mode = wf_test_year is not None
        if wf_mode:
            test_start_di = None
            test_end_di = None
            for di in range(ND):
                if dates[di].year == wf_test_year and test_start_di is None:
                    test_start_di = di
                if dates[di].year == wf_test_year + 1 and test_end_di is None:
                    test_end_di = di
            if test_start_di is None:
                return None
            if test_end_di is None:
                test_end_di = ND
            end_di = test_end_di
        else:
            test_start_di = MIN_TRAIN
            end_di = ND

        cash = float(CASH0)
        positions = []
        trades = []

        for di in range(MIN_TRAIN, end_di):
            # Reset cash at start of test window (WF only)
            if wf_mode and di == test_start_di:
                cash = float(CASH0)
                positions = []

            # --- Close positions entered yesterday ---
            closed = []
            for pos in positions:
                if di - pos['entry_di'] >= 1:
                    cn = C[pos['si'], di]
                    if np.isnan(cn) or cn <= 0:
                        cn = pos['entry']
                    mult = MULT.get(pos['sym'], DEF_MULT)
                    mkt_val = cn * mult * pos['lots']
                    cash += mkt_val - mkt_val * COMM
                    pnl = (cn - pos['entry']) * mult * pos['lots'] * pos['dir']
                    invested = pos['entry'] * mult * pos['lots']
                    pnl_pct = pnl / invested * 100 if invested > 0 else 0
                    trades.append({
                        'pnl_pct': pnl_pct,
                        'di': pos['entry_di'],
                        'year': dates[di].year if di < ND else dates[-1].year,
                        'dir': pos['dir'],
                    })
                    closed.append(pos)
            for pos in closed:
                positions.remove(pos)

            # --- Score candidates ---
            candidates = []
            for si in trade_sis:
                sym = syms[si]
                if np.isnan(C[si, di]) or C[si, di] <= 0:
                    continue
                if any(p['si'] == si for p in positions):
                    continue

                vr = vol_rank[si, di]

                # Mode D: Vol filter -- skip extreme vol
                if mode == 'vol_filter':
                    if np.isnan(vr):
                        continue
                    if vr < vol_filter_lo or vr > vol_filter_hi:
                        continue

                # Compute score based on mode
                if mode == 'fixed':
                    score = div[fixed_lb][si, di]
                    if np.isnan(score) or score <= threshold:
                        continue
                    lot_scale = 1.0

                elif mode == 'regime_switch':
                    # Pick LB based on vol regime
                    if np.isnan(vr):
                        score = div[1][si, di]  # default to LB=1
                        lb_used = 1
                    elif vr > vol_hi:
                        score = div[1][si, di]  # high vol -> fast
                        lb_used = 1
                    elif vr < vol_lo:
                        score = div[3][si, di]  # low vol -> slow
                        lb_used = 3
                    else:
                        score = div[2][si, di]  # medium vol -> LB=2
                        lb_used = 2
                    if np.isnan(score) or score <= threshold:
                        continue
                    lot_scale = 1.0

                elif mode == 'vol_weighted':
                    # Continuous blend: weight LB=1 more in high vol, LB=3 more in low vol
                    s1 = div[1][si, di]
                    s2 = div[2][si, di]
                    s3 = div[3][si, di]
                    if np.isnan(s1) or np.isnan(s2) or np.isnan(s3):
                        continue
                    if np.isnan(vr):
                        vr_mid = 0.5
                    else:
                        vr_mid = vr
                    # Weight: w1 high when vr high, w3 high when vr low
                    # Linear interpolation: w1 = vr, w3 = 1-vr, w2 = 2*min(vr, 1-vr)
                    w1 = vr_mid
                    w3 = 1.0 - vr_mid
                    w2 = 1.0 - abs(vr_mid - 0.5) * 2  # peaks at 0.5
                    wsum = w1 + w2 + w3
                    if wsum <= 0:
                        continue
                    score = (w1 * s1 + w2 * s2 + w3 * s3) / wsum
                    if score <= threshold:
                        continue
                    lot_scale = 1.0

                elif mode == 'vol_scaled':
                    # Fixed LB=1 signal, but scale position size by vol
                    score = div[1][si, di]
                    if np.isnan(score) or score <= threshold:
                        continue
                    if np.isnan(vr):
                        lot_scale = 1.0
                    elif vr > vol_hi:
                        lot_scale = vol_scale_factor  # reduce size in high vol
                    elif vr < vol_lo:
                        lot_scale = 1.0  # full size in low vol
                    else:
                        # Linear interpolation between 1.0 and vol_scale_factor
                        lot_scale = 1.0 - (vr - vol_lo) / (vol_hi - vol_lo) * (1.0 - vol_scale_factor)

                elif mode == 'vol_filter':
                    # Use LB=1, but skip extreme vol
                    score = div[1][si, di]
                    if np.isnan(score) or score <= threshold:
                        continue
                    lot_scale = 1.0

                else:
                    continue

                candidates.append((si, score, 1, lot_scale))  # long

            if not candidates:
                continue

            # Sort by score (highest divergence first)
            candidates.sort(key=lambda x: -x[1])

            # Open positions
            n_slots = top_n - len(positions)
            for si, score, direction, lot_scale in candidates[:n_slots]:
                c = C[si, di]
                if np.isnan(c) or c <= 0:
                    continue
                mult = MULT.get(syms[si], DEF_MULT)
                notional = c * mult
                lots_full = int(cash / (notional * (1 + COMM)))
                lots = max(1, int(lots_full * lot_scale))
                if lots <= 0:
                    continue
                cost_in = notional * lots * (1 + COMM)
                if cost_in > cash:
                    lots = int(cash * 0.95 / (notional * (1 + COMM)))
                    cost_in = notional * lots * (1 + COMM) if lots > 0 else 0
                if lots <= 0 or cost_in <= 0 or cost_in > cash:
                    continue

                cash -= cost_in
                positions.append({
                    'si': si, 'entry': c, 'entry_di': di,
                    'lots': lots, 'dir': direction, 'sym': syms[si],
                })

        # Close remaining positions
        for pos in positions:
            ae = ND - 1
            cn = C[pos['si'], ae]
            if np.isnan(cn) or cn <= 0:
                cn = pos['entry']
            mult = MULT.get(pos['sym'], DEF_MULT)
            mkt_val = cn * mult * pos['lots']
            cash += mkt_val - mkt_val * COMM

        # Calculate results
        test_trades = trades
        if wf_mode:
            n_days_test = test_end_di - test_start_di
        else:
            n_days_test = ND - MIN_TRAIN
        ann = annual_return(cash, CASH0, n_days_test)
        wr = np.mean([1 if t['pnl_pct'] > 0 else 0 for t in test_trades]) * 100 if test_trades else 0
        n_trades = len(test_trades)

        return {
            'ann': ann, 'wr': wr, 'n': n_trades,
            'final_cash': cash, 'n_days': n_days_test,
        }

    # ================================================================
    # BUILD CONFIGURATIONS
    # ================================================================
    print("\n[Sweep] Building configurations...", flush=True)

    configs = []
    cid = 0

    # --- Baseline: Fixed LB=1 (V74 champion) ---
    for thresh in [0.001, 0.003, 0.005, 0.01]:
        for tn in [1, 3]:
            cid += 1
            configs.append({
                'id': cid, 'mode': 'fixed', 'fixed_lb': 1,
                'threshold': thresh, 'top_n': tn,
                'label': f"Fixed_LB1_T{thresh}_TN{tn}",
            })

    # --- Fixed LB=2,3 baselines ---
    for lb in [2, 3]:
        for thresh in [0.001, 0.003, 0.005, 0.01]:
            for tn in [1, 3]:
                cid += 1
                configs.append({
                    'id': cid, 'mode': 'fixed', 'fixed_lb': lb,
                    'threshold': thresh, 'top_n': tn,
                    'label': f"Fixed_LB{lb}_T{thresh}_TN{tn}",
                })

    # --- Mode A: Regime-switching ---
    for vol_lo, vol_hi in [(0.2, 0.8), (0.3, 0.7), (0.25, 0.75)]:
        for thresh in [0.001, 0.003, 0.005, 0.01]:
            for tn in [1, 3]:
                cid += 1
                configs.append({
                    'id': cid, 'mode': 'regime_switch',
                    'vol_lo': vol_lo, 'vol_hi': vol_hi,
                    'threshold': thresh, 'top_n': tn,
                    'label': f"Regime_vl{vol_lo}_vh{vol_hi}_T{thresh}_TN{tn}",
                })

    # --- Mode B: Vol-weighted ---
    for thresh in [0.001, 0.003, 0.005, 0.01]:
        for tn in [1, 3]:
            cid += 1
            configs.append({
                'id': cid, 'mode': 'vol_weighted',
                'threshold': thresh, 'top_n': tn,
                'label': f"VolWgt_T{thresh}_TN{tn}",
            })

    # --- Mode C: Vol-scaled position sizing ---
    for vol_lo, vol_hi in [(0.3, 0.7), (0.25, 0.75)]:
        for sf in [0.3, 0.5, 0.7]:
            for thresh in [0.001, 0.003, 0.005, 0.01]:
                for tn in [1, 3]:
                    cid += 1
                    configs.append({
                        'id': cid, 'mode': 'vol_scaled',
                        'vol_lo': vol_lo, 'vol_hi': vol_hi,
                        'vol_scale_factor': sf,
                        'threshold': thresh, 'top_n': tn,
                        'label': f"VolScl_sf{sf}_T{thresh}_TN{tn}",
                    })

    # --- Mode D: Vol filter ---
    for vf_lo, vf_hi in [(0.05, 0.95), (0.1, 0.9), (0.15, 0.85), (0.2, 0.8)]:
        for thresh in [0.001, 0.003, 0.005, 0.01]:
            for tn in [1, 3]:
                cid += 1
                configs.append({
                    'id': cid, 'mode': 'vol_filter',
                    'vol_filter_lo': vf_lo, 'vol_filter_hi': vf_hi,
                    'threshold': thresh, 'top_n': tn,
                    'label': f"VolFlt_{vf_lo}_{vf_hi}_T{thresh}_TN{tn}",
                })

    print(f"  Total configs: {len(configs)}")

    # ================================================================
    # RUN FULL-PERIOD BACKTEST
    # ================================================================
    print("\n[Backtest] Running full-period sweep...", flush=True)
    t0 = time.time()

    results = []
    for i, cfg in enumerate(configs):
        r = run_backtest(cfg)
        if r:
            r['config'] = cfg
            r['label'] = cfg['label']
            results.append(r)
        if (i + 1) % 50 == 0:
            print(f"  ... {i+1}/{len(configs)} done ({time.time()-t0:.1f}s)", flush=True)

    results.sort(key=lambda x: -x['ann'])

    # Print top 15
    print("\n" + "=" * 98)
    print("  FULL-PERIOD RESULTS (Top 15)")
    print("=" * 98)
    print(f"  {'#':>3} | {'Label':<50} | {'Ann':>8} | {'WR':>5} | {'N':>5} | {'Mode':<15}")
    print("-" * 100)
    for i, r in enumerate(results[:15]):
        mode = r['config']['mode']
        print(f"  {i+1:>3} | {r['label']:<50} | {r['ann']:>+7.1f}% | {r['wr']:>4.1f}% | {r['n']:>5} | {mode:<15}")

    # ================================================================
    # MODE COMPARISON
    # ================================================================
    print("\n" + "=" * 98)
    print("  MODE COMPARISON")
    print("=" * 98)

    mode_labels = {
        'fixed': 'Fixed LB',
        'regime_switch': 'Regime-switch',
        'vol_weighted': 'Vol-weighted',
        'vol_scaled': 'Vol-scaled',
        'vol_filter': 'Vol filter',
    }
    for mode_key, mode_label in mode_labels.items():
        subset = [r for r in results if r['config']['mode'] == mode_key]
        if subset:
            best = subset[0]
            print(f"  {mode_label:<20}: Best Ann = {best['ann']:>+8.1f}%  WR={best['wr']:.1f}%  N={best['n']}  [{best['label']}]")
        else:
            print(f"  {mode_label:<20}: No results")

    # Specific V74 baseline comparison
    v74_baseline = [r for r in results if r['config']['mode'] == 'fixed'
                    and r['config'].get('fixed_lb') == 1]
    if v74_baseline:
        best_v74 = v74_baseline[0]
        print(f"\n  V74 baseline (Fixed LB=1 best): {best_v74['ann']:>+8.1f}%  [{best_v74['label']}]")

    # Best adaptive
    adaptive = [r for r in results if r['config']['mode'] != 'fixed']
    if adaptive:
        best_adaptive = adaptive[0]
        print(f"  Best adaptive:                  {best_adaptive['ann']:>+8.1f}%  [{best_adaptive['label']}]")
        if v74_baseline:
            delta = best_adaptive['ann'] - best_v74['ann']
            print(f"  Delta vs V74:                   {delta:>+8.1f}%")

    # ================================================================
    # WALK-FORWARD (Top 10)
    # ================================================================
    print("\n" + "=" * 98)
    print("  WALK-FORWARD (Top 10 configs)")
    print("=" * 98)

    wf_years = [2020, 2021, 2022, 2023, 2024, 2025]

    # Get top 10 unique configs (deduplicate by mode if needed)
    top10 = results[:10]

    wf_results = []
    for i, r in enumerate(top10):
        cfg = r['config']
        wf_row = {'label': cfg['label'], 'mode': cfg['mode'], 'windows': {}}
        for yr in wf_years:
            wr = run_backtest(cfg, wf_test_year=yr)
            if wr:
                wf_row['windows'][yr] = wr['ann']
        wf_results.append(wf_row)
        print(f"  WF {i+1}/10 done: {cfg['label']}", flush=True)

    # Print WF table
    hdr = f"  {'#':>3} | {'Config':<50} | {'Avg':>8} |"
    for yr in wf_years:
        hdr += f"  {yr:>7} |"
    hdr += f"  {'Pos':>4}"
    print(hdr)
    print("-" * len(hdr))
    for i, wf in enumerate(wf_results):
        vals = [wf['windows'].get(yr, 0) for yr in wf_years]
        avg = np.mean(vals) if vals else 0
        pos = sum(1 for v in vals if v > 0)
        line = f"  {i+1:>3} | {wf['label']:<50} | {avg:>+7.1f}% |"
        for v in vals:
            line += f"  {v:>+7.1f}% |"
        line += f"  {pos}/6"
        print(line)

    # ================================================================
    # KEY FINDINGS
    # ================================================================
    print("\n" + "=" * 98)
    print("  KEY FINDINGS")
    print("=" * 98)

    # Does vol-adaptive LB beat fixed LB=1?
    fixed_lb1 = [r for r in results if r['config']['mode'] == 'fixed' and r['config'].get('fixed_lb') == 1]
    regime = [r for r in results if r['config']['mode'] == 'regime_switch']
    volwgt = [r for r in results if r['config']['mode'] == 'vol_weighted']
    volscl = [r for r in results if r['config']['mode'] == 'vol_scaled']
    volflt = [r for r in results if r['config']['mode'] == 'vol_filter']

    best_fixed = fixed_lb1[0] if fixed_lb1 else None
    best_regime = regime[0] if regime else None
    best_volwgt = volwgt[0] if volwgt else None
    best_volscl = volscl[0] if volscl else None
    best_volflt = volflt[0] if volflt else None

    print(f"\n  Fixed LB=1 (V74 champion baseline):")
    if best_fixed:
        print(f"    Best: {best_fixed['ann']:>+8.1f}%  WR={best_fixed['wr']:.1f}%  [{best_fixed['label']}]")

    print(f"\n  Regime-switching (discrete LB by vol):")
    if best_regime:
        print(f"    Best: {best_regime['ann']:>+8.1f}%  WR={best_regime['wr']:.1f}%  [{best_regime['label']}]")
        if best_fixed:
            d = best_regime['ann'] - best_fixed['ann']
            tag = "BEATS" if d > 0 else "LOSES TO"
            print(f"    => {tag} fixed LB=1 by {abs(d):+.1f}%")

    print(f"\n  Vol-weighted (continuous blend):")
    if best_volwgt:
        print(f"    Best: {best_volwgt['ann']:>+8.1f}%  WR={best_volwgt['wr']:.1f}%  [{best_volwgt['label']}]")
        if best_fixed:
            d = best_volwgt['ann'] - best_fixed['ann']
            tag = "BEATS" if d > 0 else "LOSES TO"
            print(f"    => {tag} fixed LB=1 by {abs(d):+.1f}%")

    print(f"\n  Vol-scaled (position sizing by vol):")
    if best_volscl:
        print(f"    Best: {best_volscl['ann']:>+8.1f}%  WR={best_volscl['wr']:.1f}%  [{best_volscl['label']}]")
        if best_fixed:
            d = best_volscl['ann'] - best_fixed['ann']
            tag = "BEATS" if d > 0 else "LOSES TO"
            print(f"    => {tag} fixed LB=1 by {abs(d):+.1f}%")

    print(f"\n  Vol filter (skip extreme vol):")
    if best_volflt:
        print(f"    Best: {best_volflt['ann']:>+8.1f}%  WR={best_volflt['wr']:.1f}%  [{best_volflt['label']}]")
        if best_fixed:
            d = best_volflt['ann'] - best_fixed['ann']
            tag = "BEATS" if d > 0 else "LOSES TO"
            print(f"    => {tag} fixed LB=1 by {abs(d):+.1f}%")

    # WF comparison
    print(f"\n  Walk-Forward Stability:")
    for i, wf in enumerate(wf_results[:5]):
        vals = [wf['windows'].get(yr, 0) for yr in wf_years]
        pos = sum(1 for v in vals if v > 0)
        avg = np.mean(vals)
        worst = min(vals)
        print(f"    #{i+1} {wf['label'][:45]}: Avg={avg:>+.1f}% Pos={pos}/6 Worst={worst:>+.1f}%")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.1f}s")
    print("=" * 98)


if __name__ == '__main__':
    main()
