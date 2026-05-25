"""
Alpha Futures V96 -- Practical Execution Test (Next-Day Open Entry)
====================================================================
CRITICAL QUESTION: Does the alpha from V82/V92 survive when we enter at
NEXT DAY'S OPEN instead of SAME DAY'S CLOSE?

All previous strategies (V74/V82/V92) entered at today's close based on
today's data. But in real trading, you see today's close AFTER the market
closes, so you can only enter at TOMORROW'S OPEN. This adds a 1-day delay.

SIGNALS TESTED (12 configs + 2 baselines):
  A) V82 cc z-score:        signal at di, enter O[di+1], exit C[di+1] (hold 1)
  B) V82 cc z-score:        signal at di, enter O[di+1], exit C[di+2] (hold 2)
  C) V82 cc z-score:        signal at di, enter O[di+1], exit C[di+3] (hold 3)
  D) V92 overnight z-score:  signal at di, enter O[di+1], exit C[di+1] (hold 1)
  E) V92 overnight z-score:  signal at di, enter O[di+1], exit C[di+2] (hold 2)
  F) V92 overnight z-score:  signal at di, enter O[di+1], exit C[di+3] (hold 3)
  G) V74 within-group div:   signal at di, enter O[di+1], exit C[di+1] (hold 1)
  H) V74 within-group div:   signal at di, enter O[di+1], exit C[di+2] (hold 2)
  I) COMBINED V82 NextOpen:  signal at di, enter O[di+1], exit C[di+1]
  J) Overnight Gap MeanRev:  signal at di, enter O[di+1], exit C[di+1]
  K) BASELINE V82:           signal at di, enter C[di],   exit C[di+1] (THEORETICAL)
  L) BASELINE V92:           signal at di, enter C[di],   exit C[di+1] (THEORETICAL)

Walk-forward: top 15 configs across 2020-2025.
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
        'bcfi': 5, 'nrfi': 1, 'lgfi': 20, 'brfi': 5, 'lcfi': 1, 'sifi': 5,
        'ni': 1, 'tai': 5}
DEF_MULT = 10
COMM = 0.0003

# Group map -- srfi ONLY in soft (fixed from earlier versions)
GROUP_MAP = {}
for _s in ['rbfi', 'hcfi', 'ifi', 'jfi', 'jmfi']:
    GROUP_MAP[_s] = 'ferrous'
for _s in ['cufi', 'alfi', 'znfi', 'nifi', 'pbfi', 'snfi', 'ssfi', 'sffi']:
    GROUP_MAP[_s] = 'nonferrous'
for _s in ['aufi', 'agfi']:
    GROUP_MAP[_s] = 'precious'
for _s in ['afi', 'mfi', 'yfi', 'pfi', 'cfi', 'csfi', 'rrfi', 'lrfi']:
    GROUP_MAP[_s] = 'oils'
for _s in ['scfi', 'mafi', 'bfi', 'fufi', 'pgfi', 'ebfi', 'fbfi']:
    GROUP_MAP[_s] = 'energy'
for _s in ['ppfi', 'vfi', 'egfi', 'tafi', 'fgfi', 'lfi']:
    GROUP_MAP[_s] = 'chemical'
for _s in ['whfi', 'apfi', 'cjfi', 'oifi', 'rmfi', 'srfi', 'cffi']:
    GROUP_MAP[_s] = 'soft'
for _s in ['jdfi', 'lhfi', 'pkfi']:
    GROUP_MAP[_s] = 'livestock'


def annual_return(final, initial, n_days):
    if final <= 0 or initial <= 0 or n_days <= 0:
        return -100.0
    yrs = n_days / 252
    return (final / initial) ** (1.0 / yrs) * 100 - 100


def main():
    print("=" * 130)
    print("Alpha Futures V96 -- Practical Execution Test (Next-Day Open Entry)")
    print("=" * 130)
    print("\n  CRITICAL: Testing whether alpha survives 1-day execution delay")
    print("  Signal computed at close of day di --> Entry at OPEN of day di+1")
    print("  Baseline: same-close entry (theoretical, NOT practically executable)")

    # ── Load data ────────────────────────────────────────────────────
    print("\n[Data] Loading...", flush=True)
    t_start = time.time()
    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)
    years = sorted(set(d.year for d in dates))
    print(f"  {NS} commodities, {ND} days, years: {years}")

    # ── Build group membership ───────────────────────────────────────
    gm_map = {}
    si_group = {}
    for si in range(NS):
        g = GROUP_MAP.get(syms[si])
        if g:
            gm_map.setdefault(g, []).append(si)
            si_group[si] = g

    trade_sis = [si for si in range(NS) if si in si_group]
    group_names = sorted(gm_map.keys())
    print(f"  Tradeable: {len(trade_sis)} commodities in {len(group_names)} groups")
    for gn in group_names:
        print(f"    {gn}: {len(gm_map[gn])} commodities")

    # ================================================================
    # PRECOMPUTE RETURNS
    # ================================================================
    print("\n[Signals] Computing returns...", flush=True)
    t0 = time.time()

    # Close-to-close return: ret1[si, di] = (C[di] - C[di-1]) / C[di-1]
    ret1 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(1, ND):
            cn = C[si, di]
            cp = C[si, di - 1]
            if not np.isnan(cn) and not np.isnan(cp) and cp > 0:
                ret1[si, di] = (cn - cp) / cp

    # Overnight return: on_ret[si, di] = (O[di] - C[di-1]) / C[di-1]
    on_ret = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(1, ND):
            c_prev = C[si, di - 1]
            o_now = O[si, di]
            if not np.isnan(c_prev) and c_prev > 0 and not np.isnan(o_now) and o_now > 0:
                on_ret[si, di] = (o_now - c_prev) / c_prev

    # ================================================================
    # PRECOMPUTE GROUP-LEVEL AGGREGATES
    # ================================================================
    print("[Signals] Computing group-level aggregates...", flush=True)

    # group_total_avg[group_name][di] = average return of that group
    grp_total = {}
    for grp in group_names:
        arr = np.full(ND, np.nan)
        members = gm_map[grp]
        for di in range(1, ND):
            vals = [ret1[sk, di] for sk in members if not np.isnan(ret1[sk, di])]
            if vals:
                arr[di] = np.mean(vals)
        grp_total[grp] = arr

    # group_avg_ret[si, di] = average return of own group (excluding self)
    grp_avg = np.full((NS, ND), np.nan)
    for grp, members in gm_map.items():
        for di in range(1, ND):
            for sj in members:
                vals = [ret1[sk, di] for sk in members
                        if sk != sj and not np.isnan(ret1[sk, di])]
                if vals:
                    grp_avg[sj, di] = np.mean(vals)

    # all_groups_avg[di] = grand mean across groups
    all_groups_avg = np.full(ND, np.nan)
    all_groups_std = np.full(ND, np.nan)
    for di in range(1, ND):
        vals = [grp_total[g][di] for g in group_names if not np.isnan(grp_total[g][di])]
        if len(vals) >= 2:
            all_groups_avg[di] = np.mean(vals)
            all_groups_std[di] = np.std(vals)

    # Cross-group z-score of close-to-close return (V82 signal)
    z_cc = np.full((NS, ND), np.nan)
    for si in trade_sis:
        for di in range(1, ND):
            own = ret1[si, di]
            if np.isnan(own) or np.isnan(all_groups_avg[di]) or np.isnan(all_groups_std[di]):
                continue
            if all_groups_std[di] < 1e-8:
                continue
            z_cc[si, di] = (own - all_groups_avg[di]) / all_groups_std[di]

    # Cross-group z-score of overnight return (V92 signal)
    # For overnight, we need group-level aggregates from overnight returns
    grp_total_on = {}
    for grp in group_names:
        arr = np.full(ND, np.nan)
        members = gm_map[grp]
        for di in range(1, ND):
            vals = [on_ret[sk, di] for sk in members if not np.isnan(on_ret[sk, di])]
            if vals:
                arr[di] = np.mean(vals)
        grp_total_on[grp] = arr

    all_groups_avg_on = np.full(ND, np.nan)
    all_groups_std_on = np.full(ND, np.nan)
    for di in range(1, ND):
        vals = [grp_total_on[g][di] for g in group_names if not np.isnan(grp_total_on[g][di])]
        if len(vals) >= 2:
            all_groups_avg_on[di] = np.mean(vals)
            all_groups_std_on[di] = np.std(vals)

    z_on = np.full((NS, ND), np.nan)
    for si in trade_sis:
        for di in range(1, ND):
            own = on_ret[si, di]
            if np.isnan(own) or np.isnan(all_groups_avg_on[di]) or np.isnan(all_groups_std_on[di]):
                continue
            if all_groups_std_on[di] < 1e-8:
                continue
            z_on[si, di] = (own - all_groups_avg_on[di]) / all_groups_std_on[di]

    print(f"  Signals computed ({time.time()-t0:.1f}s)")

    # ================================================================
    # BACKTEST ENGINE
    # ================================================================
    def run_backtest(config, wf_test_year=None):
        """
        Config:
            signal: 'V82_cc_zscore' | 'V92_overnight_zscore' | 'V74_within_group' |
                    'COMBINED_V82' | 'OVERNIGHT_GAP_MEANREV' |
                    'BASELINE_V82' | 'BASELINE_V92'
            entry: 'next_open' | 'same_close'
            hold_days: 1 | 2 | 3
            threshold: float
            top_n: 1 | 3 | 5
            comm: float
        """
        sig_type = config['signal']
        entry_mode = config['entry']
        hold_days = config['hold_days']
        threshold = config['threshold']
        top_n = config['top_n']
        comm = config.get('comm', COMM)

        # Date boundaries
        if wf_test_year is not None:
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
            start_di = MIN_TRAIN
            end_di = test_end_di
        else:
            test_start_di = MIN_TRAIN
            start_di = MIN_TRAIN
            end_di = ND
            test_end_di = ND

        # We need enough room for hold_days + 1 (entry day buffer)
        max_hold = hold_days + 1
        if end_di < start_di + max_hold:
            return None

        cash = float(CASH0)
        positions = []   # list of dicts: {si, entry_price, entry_di, lots, dir, sym, hold_days}
        trades = []

        for di in range(start_di, end_di):
            # Reset cash at test window start (WF mode)
            if wf_test_year is not None and di == test_start_di:
                cash = float(CASH0)
                positions = []

            # ── Close positions that have been held long enough ───────
            closed = []
            for pos in positions:
                days_held = di - pos['entry_di']
                if days_held >= pos['hold_days']:
                    exit_price = C[pos['si'], di]
                    if np.isnan(exit_price) or exit_price <= 0:
                        exit_price = pos['entry_price']
                    mult = MULT.get(pos['sym'], DEF_MULT)
                    mkt_val = exit_price * mult * abs(pos['lots'])
                    cash += mkt_val - mkt_val * comm
                    pnl = (exit_price - pos['entry_price']) * mult * pos['lots'] * pos['dir']
                    invested = pos['entry_price'] * mult * abs(pos['lots'])
                    pnl_pct = pnl / invested * 100 if invested > 0 else 0
                    trades.append({
                        'pnl_pct': pnl_pct,
                        'entry_di': pos['entry_di'],
                        'exit_di': di,
                        'year': dates[di].year if di < ND else dates[-1].year,
                        'dir': pos['dir'],
                        'sym': pos['sym'],
                    })
                    closed.append(pos)
            for pos in closed:
                positions.remove(pos)

            # ── Generate signals at day di ───────────────────────────
            # Signal is computed using data up to and including day di
            candidates = []  # (si, score, direction, sym)

            # --- V82: cross-group z-score of cc return ---
            if sig_type in ('V82_cc_zscore', 'COMBINED_V82'):
                for si in trade_sis:
                    if any(p['si'] == si for p in positions):
                        continue
                    z = z_cc[si, di]
                    if np.isnan(z):
                        continue
                    if z < -threshold:
                        score = -z
                        candidates.append((si, score, 1, syms[si]))

            # --- V92: cross-group z-score of overnight return ---
            elif sig_type in ('V92_overnight_zscore',):
                for si in trade_sis:
                    if any(p['si'] == si for p in positions):
                        continue
                    z = z_on[si, di]
                    if np.isnan(z):
                        continue
                    if z < -threshold:
                        score = -z
                        candidates.append((si, score, 1, syms[si]))

            # --- V74: within-group divergence ---
            elif sig_type == 'V74_within_group':
                for si in trade_sis:
                    if any(p['si'] == si for p in positions):
                        continue
                    own = ret1[si, di]
                    ga = grp_avg[si, di]
                    if np.isnan(own) or np.isnan(ga):
                        continue
                    div = ga - own
                    if div > threshold:
                        candidates.append((si, div, 1, syms[si]))

            # --- COMBINED V82: same as V82 cc z-score, practical entry ---
            # (handled above in V82_cc_zscore branch)

            # --- Overnight Gap MeanRev ---
            elif sig_type == 'OVERNIGHT_GAP_MEANREV':
                for si in trade_sis:
                    if any(p['si'] == si for p in positions):
                        continue
                    oret = on_ret[si, di]
                    z = z_on[si, di]
                    if np.isnan(oret) or np.isnan(z):
                        continue
                    # Overnight gap < -1% AND z < -0.3
                    if oret < -0.01 and z < -threshold:
                        score = -z + abs(oret) * 10
                        candidates.append((si, score, 1, syms[si]))

            # --- BASELINE V82: enter at same close ---
            elif sig_type == 'BASELINE_V82':
                for si in trade_sis:
                    if any(p['si'] == si for p in positions):
                        continue
                    z = z_cc[si, di]
                    if np.isnan(z):
                        continue
                    cc = C[si, di]
                    if np.isnan(cc) or cc <= 0:
                        continue
                    if z < -threshold:
                        score = -z
                        candidates.append((si, score, 1, syms[si]))

            # --- BASELINE V92: enter at same close ---
            elif sig_type == 'BASELINE_V92':
                for si in trade_sis:
                    if any(p['si'] == si for p in positions):
                        continue
                    z = z_on[si, di]
                    if np.isnan(z):
                        continue
                    cc = C[si, di]
                    if np.isnan(cc) or cc <= 0:
                        continue
                    if z < -threshold:
                        score = -z
                        candidates.append((si, score, 1, syms[si]))

            if not candidates:
                continue

            # Sort by score descending
            candidates.sort(key=lambda x: -x[1])

            # Determine entry price and actual hold
            if entry_mode == 'next_open':
                # Entry at O[si, di+1]
                entry_di = di + 1
                if entry_di >= end_di:
                    continue
                actual_hold = hold_days
            else:  # same_close
                # Entry at C[si, di]
                entry_di = di
                actual_hold = hold_days

            # Open positions (up to top_n slots)
            n_slots = top_n - len(positions)
            for si, score, direction, sym in candidates[:max(0, n_slots)]:
                if entry_mode == 'next_open':
                    price = O[si, entry_di]
                else:
                    price = C[si, entry_di]

                if np.isnan(price) or price <= 0:
                    continue
                # Check exit day is valid
                exit_di = entry_di + actual_hold
                if exit_di >= end_di:
                    continue
                mult = MULT.get(sym, DEF_MULT)
                notional = price * mult
                lots = int(cash / (notional * (1 + comm)))
                if lots <= 0:
                    continue
                cost_in = notional * lots * (1 + comm)
                if cost_in > cash:
                    lots = int(cash * 0.95 / (notional * (1 + comm)))
                    cost_in = notional * lots * (1 + comm) if lots > 0 else 0
                if lots <= 0 or cost_in <= 0 or cost_in > cash:
                    continue

                cash -= cost_in
                positions.append({
                    'si': si, 'entry_price': price, 'entry_di': entry_di,
                    'lots': lots, 'dir': direction, 'sym': sym,
                    'hold_days': actual_hold,
                })

        # Close remaining positions at end
        for pos in positions:
            ae = end_di - 1 if end_di < ND else ND - 1
            exit_price = C[pos['si'], ae]
            if np.isnan(exit_price) or exit_price <= 0:
                exit_price = pos['entry_price']
            mult = MULT.get(pos['sym'], DEF_MULT)
            mkt_val = exit_price * mult * abs(pos['lots'])
            cash += mkt_val - mkt_val * comm

        # Results
        wf_mode = wf_test_year is not None
        n_days_test = (test_end_di - test_start_di) if wf_mode else (end_di - start_di)
        ann = annual_return(cash, CASH0, n_days_test)
        wr = np.mean([1 if t['pnl_pct'] > 0 else 0 for t in trades]) * 100 if trades else 0
        n_trades = len(trades)
        avg_pnl = np.mean([t['pnl_pct'] for t in trades]) if trades else 0

        # Max drawdown from equity curve
        eq = float(CASH0)
        peak = eq
        mdd = 0.0
        for t in trades:
            eq *= (1 + t['pnl_pct'] / 100)
            if eq > peak:
                peak = eq
            dd = (eq - peak) / peak * 100
            if dd < mdd:
                mdd = dd

        return {
            'ann': ann, 'wr': wr, 'n': n_trades, 'avg_pnl': avg_pnl,
            'final_cash': cash, 'n_days': n_days_test, 'mdd': mdd,
        }

    # ================================================================
    # BUILD CONFIGURATIONS
    # ================================================================
    print("\n[Sweep] Building configurations...", flush=True)
    configs = []
    cid = 0

    # --- A/B/C: V82 cc z-score, NextOpen entry, hold 1/2/3 ---
    for thresh in [0.3, 0.5, 0.7, 1.0]:
        for tn in [1, 3, 5]:
            for hd in [1, 2, 3]:
                cid += 1
                configs.append({
                    'id': cid, 'signal': 'V82_cc_zscore', 'entry': 'next_open',
                    'hold_days': hd, 'threshold': thresh, 'top_n': tn, 'comm': COMM,
                    'label': f"V82_NextOpen_H{hd}_Z{thresh}_TN{tn}",
                })

    # --- D/E/F: V92 overnight z-score, NextOpen entry, hold 1/2/3 ---
    for thresh in [0.3, 0.5, 0.7]:
        for tn in [1, 3, 5]:
            for hd in [1, 2, 3]:
                cid += 1
                configs.append({
                    'id': cid, 'signal': 'V92_overnight_zscore', 'entry': 'next_open',
                    'hold_days': hd, 'threshold': thresh, 'top_n': tn, 'comm': COMM,
                    'label': f"V92_NextOpen_H{hd}_Z{thresh}_TN{tn}",
                })

    # --- G/H: V74 within-group divergence, NextOpen entry, hold 1/2 ---
    for thresh in [0.003, 0.005, 0.01]:
        for tn in [1, 3, 5]:
            for hd in [1, 2]:
                cid += 1
                configs.append({
                    'id': cid, 'signal': 'V74_within_group', 'entry': 'next_open',
                    'hold_days': hd, 'threshold': thresh, 'top_n': tn, 'comm': COMM,
                    'label': f"V74_NextOpen_H{hd}_T{thresh}_TN{tn}",
                })

    # --- I: COMBINED V82 NextOpen (same signal as A, explicit label) ---
    for thresh in [0.3, 0.5, 0.7]:
        for tn in [1, 3, 5]:
            cid += 1
            configs.append({
                'id': cid, 'signal': 'COMBINED_V82', 'entry': 'next_open',
                'hold_days': 1, 'threshold': thresh, 'top_n': tn, 'comm': COMM,
                'label': f"COMBINED_NextOpen_Z{thresh}_TN{tn}",
            })

    # --- J: Overnight Gap MeanRev, NextOpen entry ---
    for thresh in [0.3, 0.5]:
        for tn in [1, 3, 5]:
            cid += 1
            configs.append({
                'id': cid, 'signal': 'OVERNIGHT_GAP_MEANREV', 'entry': 'next_open',
                'hold_days': 1, 'threshold': thresh, 'top_n': tn, 'comm': COMM,
                'label': f"GapMR_NextOpen_Z{thresh}_TN{tn}",
            })

    # --- K: BASELINE V82 (same-close entry, theoretical) ---
    for thresh in [0.3, 0.5, 0.7, 1.0]:
        for tn in [1, 3, 5]:
            cid += 1
            configs.append({
                'id': cid, 'signal': 'BASELINE_V82', 'entry': 'same_close',
                'hold_days': 1, 'threshold': thresh, 'top_n': tn, 'comm': COMM,
                'label': f"BASELINE_V82_Z{thresh}_TN{tn}",
            })

    # --- L: BASELINE V92 (same-close entry, theoretical) ---
    for thresh in [0.3, 0.5, 0.7]:
        for tn in [1, 3, 5]:
            cid += 1
            configs.append({
                'id': cid, 'signal': 'BASELINE_V92', 'entry': 'same_close',
                'hold_days': 1, 'threshold': thresh, 'top_n': tn, 'comm': COMM,
                'label': f"BASELINE_V92_Z{thresh}_TN{tn}",
            })

    print(f"  Total configs: {len(configs)}")

    # ================================================================
    # RUN FULL-PERIOD BACKTEST
    # ================================================================
    print("\n[Backtest] Running full-period sweep...", flush=True)
    results = []
    for i, cfg in enumerate(configs):
        r = run_backtest(cfg)
        if r:
            r['config'] = cfg
            r['label'] = cfg['label']
            results.append(r)
        if (i + 1) % 20 == 0 or i == len(configs) - 1:
            print(f"  ... {i+1}/{len(configs)} done", flush=True)

    results.sort(key=lambda x: -x['ann'])

    # ================================================================
    # FULL-PERIOD RESULTS (Top 30)
    # ================================================================
    print(f"\n{'=' * 130}")
    print("  FULL-PERIOD RESULTS (Top 30)")
    print(f"{'=' * 130}")
    print(f"  {'#':>3} | {'Label':<40} | {'Ann':>9} | {'WR':>6} | {'N':>5} | {'AvgPnL':>7} | {'MDD':>7}")
    print("-" * 110)
    for i, r in enumerate(results[:30]):
        print(f"  {i+1:>3} | {r['label']:<40} | {r['ann']:>+8.1f}% | {r['wr']:>5.1f}% | {r['n']:>5} | {r['avg_pnl']:>+6.3f}% | {r['mdd']:>6.1f}%")

    # ================================================================
    # BEST PER SIGNAL TYPE (full period)
    # ================================================================
    print(f"\n{'=' * 130}")
    print("  BEST PER SIGNAL TYPE (Full Period)")
    print(f"{'=' * 130}")
    print(f"  {'Signal':<30} | {'Entry':>10} | {'Ann':>9} | {'WR':>6} | {'N':>5} | {'AvgPnL':>7} | {'MDD':>7} | Best Config")
    print("-" * 130)

    best_per_sig = {}
    for r in results:
        key = r['config']['signal']
        if key not in best_per_sig:
            best_per_sig[key] = r

    sig_order = ['V82_cc_zscore', 'V92_overnight_zscore', 'V74_within_group',
                 'COMBINED_V82', 'OVERNIGHT_GAP_MEANREV',
                 'BASELINE_V82', 'BASELINE_V92']
    for sig in sig_order:
        if sig in best_per_sig:
            b = best_per_sig[sig]
            entry = b['config']['entry']
            hd = b['config']['hold_days']
            print(f"  {sig:<30} | {entry:>10} | {b['ann']:>+8.1f}% | {b['wr']:>5.1f}% | {b['n']:>5} | {b['avg_pnl']:>+6.3f}% | {b['mdd']:>6.1f}% | {b['label']}")

    # ================================================================
    # THEORETICAL vs PRACTICAL COMPARISON
    # ================================================================
    print(f"\n{'=' * 130}")
    print("  *** THEORETICAL vs PRACTICAL EXECUTION COMPARISON ***")
    print(f"{'=' * 130}")
    print("  This is the MOST IMPORTANT table: how much alpha is lost by 1-day delay?")
    print()

    # For each baseline, find matching practical configs
    # V82: signal at di, theoretical enters C[di], practical enters O[di+1]
    print(f"  {'Signal Type':<35} | {'Theoretical':>12} | {'Practical':>12} | {'Alpha Lost':>11} | {'Survival':>8}")
    print(f"  {'':35} | {'(same-close)':>12} | {'(next-open)':>12} | {'':>11} | {'Rate':>8}")
    print("-" * 110)

    # V82 comparison: for each threshold/top_n combo, compare baseline vs next_open
    for thresh in [0.3, 0.5, 0.7, 1.0]:
        for tn in [1, 3, 5]:
            # Theoretical
            theo_list = [r for r in results
                         if r['config']['signal'] == 'BASELINE_V82'
                         and abs(r['config']['threshold'] - thresh) < 0.01
                         and r['config']['top_n'] == tn]
            # Practical: best hold_days for this combo
            prac_list = [r for r in results
                         if r['config']['signal'] == 'V82_cc_zscore'
                         and r['config']['entry'] == 'next_open'
                         and abs(r['config']['threshold'] - thresh) < 0.01
                         and r['config']['top_n'] == tn]
            if theo_list and prac_list:
                theo = theo_list[0]
                prac_best = max(prac_list, key=lambda x: x['ann'])
                alpha_lost = theo['ann'] - prac_best['ann']
                survival = prac_best['ann'] / theo['ann'] * 100 if theo['ann'] != 0 else 0
                print(f"  V82 Z{thresh} TN{tn} H{prac_best['config']['hold_days']:<17} | {theo['ann']:>+11.1f}% | {prac_best['ann']:>+11.1f}% | {alpha_lost:>+10.1f}% | {survival:>7.1f}%")

    print()
    for thresh in [0.3, 0.5, 0.7]:
        for tn in [1, 3, 5]:
            theo_list = [r for r in results
                         if r['config']['signal'] == 'BASELINE_V92'
                         and abs(r['config']['threshold'] - thresh) < 0.01
                         and r['config']['top_n'] == tn]
            prac_list = [r for r in results
                         if r['config']['signal'] == 'V92_overnight_zscore'
                         and r['config']['entry'] == 'next_open'
                         and abs(r['config']['threshold'] - thresh) < 0.01
                         and r['config']['top_n'] == tn]
            if theo_list and prac_list:
                theo = theo_list[0]
                prac_best = max(prac_list, key=lambda x: x['ann'])
                alpha_lost = theo['ann'] - prac_best['ann']
                survival = prac_best['ann'] / theo['ann'] * 100 if theo['ann'] != 0 else 0
                print(f"  V92 Z{thresh} TN{tn} H{prac_best['config']['hold_days']:<17} | {theo['ann']:>+11.1f}% | {prac_best['ann']:>+11.1f}% | {alpha_lost:>+10.1f}% | {survival:>7.1f}%")

    # ================================================================
    # HOLD DAYS COMPARISON
    # ================================================================
    print(f"\n{'=' * 130}")
    print("  HOLD DAYS COMPARISON (V82 + V92 NextOpen, best per hold)")
    print(f"{'=' * 130}")
    print(f"  {'Signal':<35} | {'Hold':>4} | {'Ann':>9} | {'WR':>6} | {'N':>5} | {'AvgPnL':>7} | {'MDD':>7} | Best Config")
    print("-" * 130)

    for sig in ['V82_cc_zscore', 'V92_overnight_zscore']:
        for hd in [1, 2, 3]:
            sub = [r for r in results
                   if r['config']['signal'] == sig
                   and r['config']['entry'] == 'next_open'
                   and r['config']['hold_days'] == hd]
            if sub:
                best = sub[0]  # already sorted by ann desc
                print(f"  {sig:<35} | {hd:>4} | {best['ann']:>+8.1f}% | {best['wr']:>5.1f}% | {best['n']:>5} | {best['avg_pnl']:>+6.3f}% | {best['mdd']:>6.1f}% | {best['label']}")

    # ================================================================
    # WALK-FORWARD (Top 15 configs)
    # ================================================================
    wf_years = [2020, 2021, 2022, 2023, 2024, 2025]

    # Collect top 15 overall + best per signal type
    wf_configs = list(results[:15])
    for sig in sig_order:
        if sig in best_per_sig:
            r = best_per_sig[sig]
            if r['config'] not in [w['config'] for w in wf_configs]:
                wf_configs.append(r)

    print(f"\n{'=' * 150}")
    print(f"  WALK-FORWARD ({len(wf_configs)} configs)")
    print(f"{'=' * 150}")

    header = f"  {'#':>3} | {'Config':<40} | {'Avg':>8} |"
    for yr in wf_years:
        header += f" {yr:>7} |"
    header += f" {'Pos':>4} | {'MDD':>7}"
    print(header)
    print("-" * 150)

    wf_rows = []
    for i, r in enumerate(wf_configs):
        cfg = r['config']
        wf_row = {'label': cfg['label'], 'signal': cfg['signal'],
                  'entry': cfg['entry'], 'windows': {}, 'mdd': {}}
        for yr in wf_years:
            wr = run_backtest(cfg, wf_test_year=yr)
            if wr:
                wf_row['windows'][yr] = wr['ann']
                wf_row['mdd'][yr] = wr['mdd']
        wf_rows.append(wf_row)

        vals = [wf_row['windows'].get(yr, 0) for yr in wf_years]
        avg = np.mean(vals) if vals else 0
        pos = sum(1 for v in vals if v > 0)
        avg_mdd = np.mean(list(wf_row['mdd'].values())) if wf_row['mdd'] else 0

        row_str = f"  {i+1:>3} | {wf_row['label']:<40} | {avg:>+7.1f}% |"
        for v in vals:
            row_str += f" {v:>+7.1f}% |"
        row_str += f" {pos}/6 | {avg_mdd:>6.1f}%"
        print(row_str)

    # ================================================================
    # WF COMPARISON PER SIGNAL
    # ================================================================
    print(f"\n{'=' * 130}")
    print("  WALK-FORWARD COMPARISON (Best per signal type)")
    print(f"{'=' * 130}")
    header2 = f"  {'Signal':<30} | {'Entry':>10} | {'WF Avg':>8} |"
    for yr in wf_years:
        header2 += f" {yr:>7} |"
    header2 += f" {'Pos':>4} | Avg MDD"
    print(header2)
    print("-" * 130)

    for sig in sig_order:
        wf_match = [w for w in wf_rows if w['signal'] == sig]
        if wf_match:
            wf = wf_match[0]
            vals = [wf['windows'].get(yr, 0) for yr in wf_years]
            avg = np.mean(vals) if vals else 0
            pos = sum(1 for v in vals if v > 0)
            avg_mdd = np.mean(list(wf['mdd'].values())) if wf['mdd'] else 0
            # Get entry mode from original config
            entry = 'next_open'
            for r in results:
                if r['config']['signal'] == sig:
                    entry = r['config']['entry']
                    break
            row_str = f"  {sig:<30} | {entry:>10} | {avg:>+7.1f}% |"
            for v in vals:
                row_str += f" {v:>+7.1f}% |"
            row_str += f" {pos}/6 | {avg_mdd:>6.1f}%"
            print(row_str)

    # ================================================================
    # FINAL VERDICT
    # ================================================================
    print(f"\n{'=' * 130}")
    print("  FINAL VERDICT: HOW MUCH ALPHA SURVIVES PRACTICAL EXECUTION?")
    print(f"{'=' * 130}")

    # Best theoretical
    theo_v82 = best_per_sig.get('BASELINE_V82')
    theo_v92 = best_per_sig.get('BASELINE_V92')

    # Best practical
    prac_v82_list = [r for r in results if r['config']['signal'] == 'V82_cc_zscore'
                     and r['config']['entry'] == 'next_open']
    prac_v92_list = [r for r in results if r['config']['signal'] == 'V92_overnight_zscore'
                     and r['config']['entry'] == 'next_open']
    prac_v74_list = [r for r in results if r['config']['signal'] == 'V74_within_group'
                     and r['config']['entry'] == 'next_open']
    prac_gap_list = [r for r in results if r['config']['signal'] == 'OVERNIGHT_GAP_MEANREV'
                     and r['config']['entry'] == 'next_open']

    print()
    if theo_v82:
        print(f"  V82 Theoretical (same-close entry):  {theo_v82['ann']:>+8.1f}%  ({theo_v82['label']})")
    if prac_v82_list:
        best_prac = prac_v82_list[0]
        print(f"  V82 Practical (next-open entry):     {best_prac['ann']:>+8.1f}%  ({best_prac['label']})")
        if theo_v82 and theo_v82['ann'] != 0:
            survival = best_prac['ann'] / theo_v82['ann'] * 100
            lost = theo_v82['ann'] - best_prac['ann']
            print(f"  --> Alpha survival rate: {survival:.1f}%  (lost {lost:+.1f}%)")

    print()
    if theo_v92:
        print(f"  V92 Theoretical (same-close entry):  {theo_v92['ann']:>+8.1f}%  ({theo_v92['label']})")
    if prac_v92_list:
        best_prac = prac_v92_list[0]
        print(f"  V92 Practical (next-open entry):     {best_prac['ann']:>+8.1f}%  ({best_prac['label']})")
        if theo_v92 and theo_v92['ann'] != 0:
            survival = best_prac['ann'] / theo_v92['ann'] * 100
            lost = theo_v92['ann'] - best_prac['ann']
            print(f"  --> Alpha survival rate: {survival:.1f}%  (lost {lost:+.1f}%)")

    print()
    if prac_v74_list:
        best_prac = prac_v74_list[0]
        print(f"  V74 Practical (next-open entry):     {best_prac['ann']:>+8.1f}%  ({best_prac['label']})")

    if prac_gap_list:
        best_prac = prac_gap_list[0]
        print(f"  Gap MeanRev Practical (next-open):   {best_prac['ann']:>+8.1f}%  ({best_prac['label']})")

    # Overall best practical
    all_prac = [r for r in results if r['config']['entry'] == 'next_open']
    if all_prac:
        best_overall_prac = all_prac[0]
        print(f"\n  BEST PRACTICAL EXECUTION STRATEGY:")
        print(f"    {best_overall_prac['label']}")
        print(f"    Annual: {best_overall_prac['ann']:>+8.1f}%")
        print(f"    WR:     {best_overall_prac['wr']:>5.1f}%")
        print(f"    N:      {best_overall_prac['n']:>5}")
        print(f"    MDD:    {best_overall_prac['mdd']:>6.1f}%")

    # Survival summary
    print(f"\n  SUMMARY:")
    if theo_v82 and theo_v82['ann'] > 0 and prac_v82_list:
        s82 = prac_v82_list[0]['ann'] / theo_v82['ann'] * 100
        print(f"    V82 alpha survival: {s82:.1f}%")
    if theo_v92 and theo_v92['ann'] > 0 and prac_v92_list:
        s92 = prac_v92_list[0]['ann'] / theo_v92['ann'] * 100
        print(f"    V92 alpha survival: {s92:.1f}%")

    print(f"\n  Total time: {time.time()-t_start:.1f}s")
    print("=" * 130)


if __name__ == '__main__':
    main()
