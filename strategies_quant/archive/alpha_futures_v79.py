"""
Alpha Futures V79 — Adaptive Threshold Modes
=============================================
V74 champion: +2185% with LB=1, fixed threshold 0.003-0.01, extended groups.
V75 showed adapting LB doesn't help. V79 tests adapting the THRESHOLD.

Modes tested:
  A: Percentile threshold (P60, P70, P80, P90 of rolling 60d divergence dist)
  B: Rolling z-score (div / rolling_std > threshold)
  C: Winning-streak filter (skip after N consecutive losses, resume after win)
  D: Time-of-week (Mon/Fri different from Tue-Thu)
  E: Signal strength filter (only trade when div in top N% of today's signals)
  fixed_baseline: V74 with fixed threshold for comparison

Walk-forward: 6 windows (2020-2025)
"""
import sys, os, time, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alpha_v2 import load_all_data, MIN_TRAIN, CASH0

# ============================================================
# CONSTANTS
# ============================================================
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
        'bcfi': 5, 'nrfi': 1, 'lgfi': 20, 'brfi': 5, 'lcfi': 1, 'sisi': 5,
        'ni': 1, 'tai': 5}
DEF_MULT = 10
COMM = 0.0003

# Extended group map (same as V74)
GROUP_MAP = {}
for s in ['rbfi', 'hcfi', 'ifi', 'jfi', 'jmfi']:
    GROUP_MAP[s] = 'ferrous'
for s in ['cufi', 'alfi', 'znfi', 'nifi', 'pbfi', 'snfi', 'ssfi', 'sffi']:
    GROUP_MAP[s] = 'nonferrous'
for s in ['aufi', 'agfi']:
    GROUP_MAP[s] = 'precious'
for s in ['afi', 'mfi', 'yfi', 'pfi', 'cfi', 'csfi', 'rrfi', 'lrfi']:
    GROUP_MAP[s] = 'oils'
for s in ['scfi', 'mafi', 'bfi', 'fufi', 'pgfi', 'ebfi', 'fbfi']:
    GROUP_MAP[s] = 'energy'
for s in ['ppfi', 'vfi', 'egfi', 'srfi', 'tafi', 'fgfi', 'lfi']:
    GROUP_MAP[s] = 'chemical'
for s in ['whfi', 'apfi', 'cjfi', 'oifi', 'rmfi', 'srfi', 'cffi']:
    GROUP_MAP[s] = 'soft'
for s in ['jdfi', 'lhfi', 'pkfi']:
    GROUP_MAP[s] = 'livestock'


def annual_return(final, initial, n_days):
    if final <= 0 or initial <= 0 or n_days <= 0:
        return -100.0
    yrs = n_days / 252
    return (final / initial) ** (1.0 / yrs) * 100 - 100


def main():
    print("=" * 110)
    print("Alpha Futures V79 -- Adaptive Threshold Modes")
    print("=" * 110)

    # ================================================================
    # LOAD DATA
    # ================================================================
    print("\n[Data] Loading...", flush=True)
    t_start = time.time()
    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)
    years = sorted(set(d.year for d in dates))
    print(f"  {NS} commodities, {ND} days, years: {years}")

    # ================================================================
    # PRECOMPUTE: Momentum, Group Momentum, Divergence
    # ================================================================
    print("\n[Signals] Computing momentum and divergence...", flush=True)
    t0 = time.time()

    # 1-day momentum (LB=1)
    mom1 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(1, ND):
            cn = C[si, di]
            cp = C[si, di - 1]
            if not np.isnan(cn) and not np.isnan(cp) and cp > 0:
                mom1[si, di] = (cn - cp) / cp

    # Group momentum: for each commodity, avg momentum of group peers
    gm_map = {}
    for si in range(NS):
        g = GROUP_MAP.get(syms[si])
        if g:
            gm_map.setdefault(g, []).append(si)

    grp_mom1 = np.full((NS, ND), np.nan)
    for grp, members in gm_map.items():
        for di in range(1, ND):
            for sj in members:
                ms = [mom1[sk, di] for sk in members
                      if sk != sj and not np.isnan(mom1[sk, di])]
                if ms:
                    grp_mom1[sj, di] = np.mean(ms)

    # Divergence = group_mom - own_mom (positive = commodity lagging group)
    divergence = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(1, ND):
            own = mom1[si, di]
            grp = grp_mom1[si, di]
            if not np.isnan(own) and not np.isnan(grp):
                divergence[si, di] = grp - own

    # Tradeable symbols
    trade_sis = [si for si in range(NS) if GROUP_MAP.get(syms[si])]
    print(f"  Tradeable: {len(trade_sis)} commodities in {len(gm_map)} groups")

    # ================================================================
    # PRECOMPUTE: Mode A — Rolling 60-day percentile thresholds
    # ================================================================
    print("  Computing adaptive threshold arrays...", flush=True)
    ROLL_WIN = 60

    # Collect all valid divergences per day (across all commodities)
    # For percentile computation, we use the absolute value distribution
    # We'll compute rolling percentiles of |divergence| over a 60-day window
    # Precompute daily valid divergence arrays for rolling percentile

    # For efficiency: precompute rolling percentile for each percentile level
    # We need P60, P70, P80, P90 of the |divergence| distribution
    pct_levels = [60, 70, 80, 90]
    # rolling_pcts[pct][di] = threshold value
    rolling_pcts = {}
    for pct in pct_levels:
        arr = np.full(ND, np.nan)
        for di in range(ROLL_WIN, ND):
            vals = []
            for dd in range(di - ROLL_WIN, di):
                for si in trade_sis:
                    v = divergence[si, dd]
                    if not np.isnan(v):
                        vals.append(abs(v))
            if len(vals) >= 20:
                arr[di] = np.percentile(vals, pct)
        rolling_pcts[pct] = arr
        valid_count = np.sum(~np.isnan(arr))
        mean_val = np.nanmean(arr)
        print(f"    P{pct}: {valid_count} valid days, mean threshold = {mean_val:.5f}")

    # ================================================================
    # PRECOMPUTE: Mode B — Rolling z-score of divergence
    # ================================================================
    # For each commodity, compute rolling 60-day mean and std of divergence
    # Then z = (div - rolling_mean) / rolling_std
    ROLL_WIN_B = 60
    div_rolling_mean = np.full((NS, ND), np.nan)
    div_rolling_std = np.full((NS, ND), np.nan)
    div_zscore = np.full((NS, ND), np.nan)
    for si in trade_sis:
        for di in range(ROLL_WIN_B, ND):
            window = divergence[si, di - ROLL_WIN_B:di]
            valid = window[~np.isnan(window)]
            if len(valid) >= 20:
                mu = np.mean(valid)
                sigma = np.std(valid)
                div_rolling_mean[si, di] = mu
                div_rolling_std[si, di] = sigma
                if sigma > 1e-10:
                    div_zscore[si, di] = (divergence[si, di] - mu) / sigma

    # ================================================================
    # PRECOMPUTE: Mode D — Day of week
    # ================================================================
    day_of_week = np.array([d.weekday() for d in dates])  # 0=Mon .. 4=Fri

    print(f"  All signals computed ({time.time()-t0:.1f}s)")

    # ================================================================
    # BACKTEST ENGINE
    # ================================================================
    def run_backtest(mode, params, wf_test_year=None):
        """
        mode: 'A_percentile', 'B_zscore', 'C_streak', 'D_weekday',
              'E_strength_filter', 'fixed_baseline'
        params: dict with mode-specific parameters
          - threshold: float (base threshold for fixed mode)
          - top_n: 1 or 3
          - For A: pct_level (60, 70, 80, 90)
          - For B: z_threshold (1.0, 1.5, 2.0)
          - For C: skip_after (2, 3, 5)
          - For D: allowed_days list (e.g., [0,1,2,3] for Mon-Thu)
          - For E: strength_pct (10, 20, 30)
        """
        top_n = params.get('top_n', 3)

        # Date range
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
            test_end_di = ND
            end_di = ND

        cash = float(CASH0)
        positions = []
        trades = []
        consecutive_losses = 0  # for Mode C

        for di in range(MIN_TRAIN, end_di):
            # Reset cash for WF mode
            if wf_mode and di == test_start_di:
                cash = float(CASH0)
                positions = []
                trades = []
                consecutive_losses = 0

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

                    # Track consecutive losses for Mode C
                    if pnl_pct <= 0:
                        consecutive_losses += 1
                    else:
                        consecutive_losses = 0

                    in_test = (not wf_mode) or (test_start_di <= pos['entry_di'] < test_end_di)
                    if in_test:
                        trades.append({
                            'pnl_pct': pnl_pct,
                            'di': pos['entry_di'],
                            'year': dates[di].year if di < ND else dates[-1].year,
                            'dir': pos['dir'],
                        })
                    closed.append(pos)
            for pos in closed:
                positions.remove(pos)

            # --- Mode C: Streak filter ---
            if mode == 'C_streak':
                skip_after = params.get('skip_after', 3)
                if consecutive_losses >= skip_after:
                    continue  # skip trading until we get a win (reset above)

            # --- Mode D: Day-of-week filter ---
            if mode == 'D_weekday':
                allowed_days = params.get('allowed_days', [0, 1, 2, 3, 4])
                if day_of_week[di] not in allowed_days:
                    continue

            # --- Gather all valid divergences for today ---
            today_divs = []
            for si in trade_sis:
                if np.isnan(C[si, di]) or C[si, di] <= 0:
                    continue
                if any(p['si'] == si for p in positions):
                    continue
                div = divergence[si, di]
                if np.isnan(div):
                    continue
                today_divs.append((si, div))

            if not today_divs:
                continue

            # --- Score candidates with mode-specific threshold ---
            candidates = []
            for si, div in today_divs:
                sym = syms[si]

                if mode == 'fixed_baseline':
                    threshold = params.get('threshold', 0.003)
                    if div <= threshold:
                        continue
                    candidates.append((si, div))

                elif mode == 'A_percentile':
                    pct_level = params.get('pct_level', 70)
                    # Need absolute divergence > percentile threshold
                    # AND positive divergence (commodity lagging)
                    abs_div = abs(div)
                    pct_thresh = rolling_pcts[pct_level][di]
                    if np.isnan(pct_thresh):
                        continue
                    # Require div > 0 AND |div| > adaptive threshold
                    if div > pct_thresh:
                        candidates.append((si, div))

                elif mode == 'B_zscore':
                    z_thresh = params.get('z_threshold', 1.5)
                    z = div_zscore[si, di]
                    if np.isnan(z):
                        continue
                    # Only go long when divergence is positive and z-score high
                    if div > 0 and z > z_thresh:
                        candidates.append((si, div))

                elif mode == 'C_streak':
                    # Same as fixed but with streak filter (applied above)
                    threshold = params.get('threshold', 0.003)
                    if div <= threshold:
                        continue
                    candidates.append((si, div))

                elif mode == 'D_weekday':
                    # Same as fixed but with day filter (applied above)
                    threshold = params.get('threshold', 0.003)
                    if div <= threshold:
                        continue
                    candidates.append((si, div))

                elif mode == 'E_strength_filter':
                    # Only trade when div is in top N% of today's signals
                    strength_pct = params.get('strength_pct', 20)
                    threshold = params.get('threshold', 0.003)
                    if div <= threshold:
                        continue
                    candidates.append((si, div))
                else:
                    continue

            if not candidates:
                continue

            # --- Mode E: Rank-based strength filter ---
            if mode == 'E_strength_filter':
                strength_pct = params.get('strength_pct', 20)
                # Only keep top strength_pct% of today's candidates
                candidates.sort(key=lambda x: -x[1])
                keep_n = max(1, int(len(candidates) * strength_pct / 100.0))
                candidates = candidates[:keep_n]

            # Sort by divergence (highest first) and open positions
            candidates.sort(key=lambda x: -x[1])
            n_slots = top_n - len(positions)
            for si, score in candidates[:n_slots]:
                c = C[si, di]
                if np.isnan(c) or c <= 0:
                    continue
                mult = MULT.get(syms[si], DEF_MULT)
                notional = c * mult
                lots = int(cash / (notional * (1 + COMM)))
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
                    'lots': lots, 'dir': 1, 'sym': syms[si],
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
            in_test = (not wf_mode) or (test_start_di <= pos['entry_di'] < test_end_di)
            if in_test:
                pnl = (cn - pos['entry']) * mult * pos['lots'] * pos['dir']
                invested = pos['entry'] * mult * pos['lots']
                pnl_pct = pnl / invested * 100 if invested > 0 else 0
                trades.append({
                    'pnl_pct': pnl_pct,
                    'di': pos['entry_di'],
                    'year': dates[ae].year,
                    'dir': pos['dir'],
                })

        # Compute results
        if wf_mode:
            n_days_test = test_end_di - test_start_di
        else:
            n_days_test = ND - MIN_TRAIN

        ann = annual_return(cash, CASH0, n_days_test)
        wr = np.mean([1 if t['pnl_pct'] > 0 else 0 for t in trades]) * 100 if trades else 0
        n_trades = len(trades)

        # Profit factor
        wins = [t['pnl_pct'] for t in trades if t['pnl_pct'] > 0]
        losses = [-t['pnl_pct'] for t in trades if t['pnl_pct'] < 0]
        gross_profit = sum(wins) if wins else 0
        gross_loss = sum(losses) if losses else 0.001
        pf = gross_profit / gross_loss if gross_loss > 0 else 0

        # Avg trade
        avg_trade = np.mean([t['pnl_pct'] for t in trades]) if trades else 0

        return {
            'ann': ann, 'wr': wr, 'n': n_trades,
            'pf': pf, 'avg': avg_trade,
            'final_cash': cash, 'n_days': n_days_test,
        }

    # ================================================================
    # BUILD CONFIGURATIONS
    # ================================================================
    print("\n" + "=" * 110)
    print("  BUILDING CONFIGURATIONS")
    print("=" * 110)

    configs = []
    cid = 0

    # --- Fixed baseline (V74 champion configs) ---
    for thresh in [0.003, 0.005, 0.01]:
        for tn in [1, 3]:
            cid += 1
            configs.append({
                'id': cid,
                'mode': 'fixed_baseline',
                'params': {'threshold': thresh, 'top_n': tn},
                'label': f"FIXED_T{thresh}_TN{tn}",
            })

    # --- Mode A: Percentile threshold ---
    for pct in pct_levels:
        for tn in [1, 3]:
            cid += 1
            configs.append({
                'id': cid,
                'mode': 'A_percentile',
                'params': {'pct_level': pct, 'top_n': tn},
                'label': f"A_P{pct}_TN{tn}",
            })

    # --- Mode B: Z-score ---
    for z_thresh in [1.0, 1.5, 2.0]:
        for tn in [1, 3]:
            cid += 1
            configs.append({
                'id': cid,
                'mode': 'B_zscore',
                'params': {'z_threshold': z_thresh, 'top_n': tn},
                'label': f"B_Z{z_thresh}_TN{tn}",
            })

    # --- Mode C: Winning-streak filter ---
    for skip_after in [2, 3, 5]:
        for thresh in [0.003, 0.005]:
            for tn in [1, 3]:
                cid += 1
                configs.append({
                    'id': cid,
                    'mode': 'C_streak',
                    'params': {'skip_after': skip_after, 'threshold': thresh, 'top_n': tn},
                    'label': f"C_SKIP{skip_after}_T{thresh}_TN{tn}",
                })

    # --- Mode D: Day-of-week ---
    day_configs = [
        ([0, 1, 2, 3, 4], "ALL"),
        ([1, 2, 3], "TUE-THU"),
        ([0, 4], "MON+FRI"),
        ([0, 1, 2, 3], "MON-THU"),
        ([1, 2, 3, 4], "TUE-FRI"),
        ([0], "MON_ONLY"),
        ([4], "FRI_ONLY"),
    ]
    for allowed, day_label in day_configs:
        for tn in [1, 3]:
            cid += 1
            configs.append({
                'id': cid,
                'mode': 'D_weekday',
                'params': {'allowed_days': allowed, 'threshold': 0.003, 'top_n': tn},
                'label': f"D_{day_label}_TN{tn}",
            })

    # --- Mode E: Strength filter ---
    for strength_pct in [10, 20, 30]:
        for thresh in [0.003, 0.005]:
            for tn in [1, 3]:
                cid += 1
                configs.append({
                    'id': cid,
                    'mode': 'E_strength_filter',
                    'params': {'strength_pct': strength_pct, 'threshold': thresh, 'top_n': tn},
                    'label': f"E_TOP{strength_pct}pct_T{thresh}_TN{tn}",
                })

    print(f"  Total configurations: {len(configs)}")

    # ================================================================
    # RUN FULL-PERIOD BACKTEST
    # ================================================================
    print("\n" + "=" * 110)
    print("  FULL-PERIOD BACKTEST")
    print("=" * 110)
    print(f"  {'#':>4} | {'Label':<35} | {'Ann':>10} | {'WR':>6} | {'PF':>6} | {'Avg':>7} | {'N':>5}")
    print("-" * 90)

    results = []
    for i, cfg in enumerate(configs):
        r = run_backtest(cfg['mode'], cfg['params'])
        if r:
            r['config'] = cfg
            r['label'] = cfg['label']
            r['mode'] = cfg['mode']
            results.append(r)
        if (i + 1) % 20 == 0:
            print(f"  ... {i+1}/{len(configs)} done", flush=True)

    # Sort by annual return
    results.sort(key=lambda x: -x['ann'])

    # Print top 30
    for i, r in enumerate(results[:30]):
        print(f"  {i+1:>4} | {r['label']:<35} | {r['ann']:>+9.1f}% | {r['wr']:>5.1f}% | {r['pf']:>5.2f} | {r['avg']:>+6.2f}% | {r['n']:>5}")

    # ================================================================
    # MODE COMPARISON SUMMARY
    # ================================================================
    print("\n" + "=" * 110)
    print("  MODE COMPARISON (best of each mode)")
    print("=" * 110)
    print(f"  {'Mode':<20} | {'Best Ann':>10} | {'Best WR':>8} | {'Best PF':>8} | {'Best N':>7} | {'Label'}")
    print("-" * 100)

    mode_names = {
        'fixed_baseline': 'Fixed Baseline',
        'A_percentile': 'A: Percentile',
        'B_zscore': 'B: Z-score',
        'C_streak': 'C: Streak Filter',
        'D_weekday': 'D: Day-of-Week',
        'E_strength_filter': 'E: Strength Filter',
    }

    mode_best = {}
    for r in results:
        m = r['mode']
        if m not in mode_best or r['ann'] > mode_best[m]['ann']:
            mode_best[m] = r

    for mode_key, mode_label in mode_names.items():
        if mode_key in mode_best:
            b = mode_best[mode_key]
            print(f"  {mode_label:<20} | {b['ann']:>+9.1f}% | {b['wr']:>7.1f}% | {b['pf']:>7.2f} | {b['n']:>7} | {b['label']}")

    # ================================================================
    # DETAILED MODE A BREAKDOWN
    # ================================================================
    print("\n" + "=" * 110)
    print("  MODE A: PERCENTILE THRESHOLD DETAIL")
    print("=" * 110)
    print(f"  {'Label':<20} | {'Ann':>10} | {'WR':>6} | {'PF':>6} | {'N':>5}")
    print("-" * 60)
    for r in results:
        if r['mode'] == 'A_percentile':
            print(f"  {r['label']:<20} | {r['ann']:>+9.1f}% | {r['wr']:>5.1f}% | {r['pf']:>5.2f} | {r['n']:>5}")

    # ================================================================
    # DETAILED MODE B BREAKDOWN
    # ================================================================
    print("\n" + "=" * 110)
    print("  MODE B: Z-SCORE DETAIL")
    print("=" * 110)
    print(f"  {'Label':<20} | {'Ann':>10} | {'WR':>6} | {'PF':>6} | {'N':>5}")
    print("-" * 60)
    for r in results:
        if r['mode'] == 'B_zscore':
            print(f"  {r['label']:<20} | {r['ann']:>+9.1f}% | {r['wr']:>5.1f}% | {r['pf']:>5.2f} | {r['n']:>5}")

    # ================================================================
    # DETAILED MODE D BREAKDOWN
    # ================================================================
    print("\n" + "=" * 110)
    print("  MODE D: DAY-OF-WEEK DETAIL")
    print("=" * 110)
    print(f"  {'Label':<20} | {'Ann':>10} | {'WR':>6} | {'PF':>6} | {'N':>5}")
    print("-" * 60)
    for r in results:
        if r['mode'] == 'D_weekday':
            print(f"  {r['label']:<20} | {r['ann']:>+9.1f}% | {r['wr']:>5.1f}% | {r['pf']:>5.2f} | {r['n']:>5}")

    # ================================================================
    # WALK-FORWARD: Top 10 configs + each mode's best
    # ================================================================
    print("\n" + "=" * 110)
    print("  WALK-FORWARD VALIDATION")
    print("=" * 110)

    wf_years = [2020, 2021, 2022, 2023, 2024, 2025]

    # Select configs for WF: top 5 overall + best of each mode (dedup)
    wf_configs = []
    seen_labels = set()

    # Top 5 overall
    for r in results[:5]:
        lbl = r['label']
        if lbl not in seen_labels:
            wf_configs.append(r['config'])
            seen_labels.add(lbl)

    # Best of each mode
    for mode_key in mode_names:
        if mode_key in mode_best:
            lbl = mode_best[mode_key]['label']
            if lbl not in seen_labels:
                wf_configs.append(mode_best[mode_key]['config'])
                seen_labels.add(lbl)

    print(f"  Testing {len(wf_configs)} configs across {len(wf_years)} WF windows...")
    print(f"  {'#':>3} | {'Config':<35} | {'Avg':>9} | ", end="")
    for yr in wf_years:
        print(f" {yr:>7} |", end="")
    print(f" {'Pos':>4} | {'WR_avg':>7} | {'PF_avg':>7}")
    print("-" * 140)

    for i, cfg in enumerate(wf_configs):
        wf_row = {}
        for yr in wf_years:
            wr = run_backtest(cfg['mode'], cfg['params'], wf_test_year=yr)
            if wr:
                wf_row[yr] = wr

        vals = [wf_row.get(yr, {}).get('ann', 0) for yr in wf_years]
        wrs = [wf_row.get(yr, {}).get('wr', 0) for yr in wf_years]
        pfs = [wf_row.get(yr, {}).get('pf', 0) for yr in wf_years]
        avg = np.mean(vals) if vals else 0
        avg_wr = np.mean(wrs) if wrs else 0
        avg_pf = np.mean(pfs) if pfs else 0
        pos = sum(1 for v in vals if v > 0)

        label = cfg['label']
        print(f"  {i+1:>3} | {label:<35} | {avg:>+8.1f}% |", end="")
        for v in vals:
            print(f" {v:>+7.1f}% |", end="")
        print(f" {pos}/6 | {avg_wr:>6.1f}% | {avg_pf:>6.2f}")

    # ================================================================
    # KEY FINDINGS
    # ================================================================
    print("\n" + "=" * 110)
    print("  KEY FINDINGS")
    print("=" * 110)

    # Does adaptive beat fixed?
    fixed_best_ann = mode_best.get('fixed_baseline', {}).get('ann', 0)
    adaptive_best_ann = 0
    adaptive_best_mode = ''
    for mode_key in ['A_percentile', 'B_zscore', 'C_streak', 'D_weekday', 'E_strength_filter']:
        if mode_key in mode_best:
            if mode_best[mode_key]['ann'] > adaptive_best_ann:
                adaptive_best_ann = mode_best[mode_key]['ann']
                adaptive_best_mode = mode_key

    print(f"\n  Fixed baseline best:  {fixed_best_ann:>+9.1f}%  ({mode_best.get('fixed_baseline', {}).get('label', 'N/A')})")
    print(f"  Adaptive best:        {adaptive_best_ann:>+9.1f}%  ({mode_best.get(adaptive_best_mode, {}).get('label', 'N/A')})")

    if adaptive_best_ann > fixed_best_ann:
        diff = adaptive_best_ann - fixed_best_ann
        print(f"  --> ADAPTIVE WINS by {diff:+.1f}pp!")
        print(f"      Best adaptive mode: {mode_names.get(adaptive_best_mode, adaptive_best_mode)}")
    else:
        diff = fixed_best_ann - adaptive_best_ann
        print(f"  --> FIXED BASELINE WINS by {diff:+.1f}pp")
        print(f"      Adaptive threshold does NOT improve over fixed threshold.")

    # Individual mode verdicts
    print(f"\n  Mode verdicts:")
    for mode_key, mode_label in mode_names.items():
        if mode_key in mode_best:
            b = mode_best[mode_key]
            vs_fixed = b['ann'] - fixed_best_ann
            verdict = "BEATS" if vs_fixed > 0 else "LOSES TO"
            print(f"    {mode_label:<22}: {b['ann']:>+8.1f}%  ({verdict} fixed by {abs(vs_fixed):+.1f}pp)  WR={b['wr']:.1f}%  PF={b['pf']:.2f}  N={b['n']}")

    print(f"\n  Total time: {time.time()-t_start:.1f}s")
    print("=" * 110)


if __name__ == '__main__':
    main()
