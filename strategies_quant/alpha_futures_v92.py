"""
Alpha Futures V92 -- Decomposed Cross-Group Z-Score (Overnight + Intraday)
==========================================================================
Combines V82 champion (+3305%): cross-group z-score on close-to-close returns
with V85 insight: overnight returns carry more predictive power (70/30 split).

V92 IDEA: Apply the z-score computation SEPARATELY to overnight returns and
intraday returns, then combine with alpha weighting.

HYPOTESIS:
  z_overnight  = z-score of overnight return vs cross-group overnight distribution
  z_intraday   = z-score of intraday return vs cross-group intraday distribution
  z_combined   = alpha * z_overnight + (1-alpha) * z_intraday
  Optimal alpha may be > 0.5 (overnight carries more weight)

SIGNALS:
  A) z_overnight_only:     z-score of overnight return; open entry, close exit (same day)
  B) z_intraday_only:      z-score of intraday return; close entry, next close exit
  C) z_combined_open:      alpha*z_on + (1-alpha)*z_id; open entry, close exit
  D) z_combined_close:     alpha*z_on + (1-alpha)*z_id; close entry, next close exit
  E) z_close_baseline:     V82 exact baseline (z-score of 1-day ctc return)

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

# ── Group map (same as V82) ──────────────────────────────────────────
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
for _s in ['ppfi', 'vfi', 'egfi', 'srfi', 'tafi', 'fgfi', 'lfi']:
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
    print("=" * 120)
    print("Alpha Futures V92 -- Decomposed Cross-Group Z-Score (Overnight + Intraday)")
    print("=" * 120)

    # ── Load data ────────────────────────────────────────────────────
    print("\n[Data] Loading...", flush=True)
    t_start = time.time()
    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)
    years = sorted(set(d.year for d in dates))
    print(f"  {NS} commodities, {ND} days, years: {years}")

    # ── Build group membership ───────────────────────────────────────
    gm_map = {}           # group_name -> [si, ...]
    si_group = {}         # si -> group_name
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
    print("\n[Signals] Computing decomposed returns...", flush=True)
    t0 = time.time()

    # overnight_ret[si, di] = (O[di] - C[di-1]) / C[di-1]
    # intraday_ret[si, di]  = (C[di] - O[di]) / O[di]
    # ctc_ret[si, di]       = (C[di] - C[di-1]) / C[di-1]
    overnight_ret = np.full((NS, ND), np.nan)
    intraday_ret  = np.full((NS, ND), np.nan)
    ctc_ret       = np.full((NS, ND), np.nan)

    for si in range(NS):
        for di in range(1, ND):
            c_prev = C[si, di - 1]
            o_now  = O[si, di]
            c_now  = C[si, di]
            if not np.isnan(c_prev) and c_prev > 0 and not np.isnan(o_now) and o_now > 0:
                overnight_ret[si, di] = (o_now - c_prev) / c_prev
            if not np.isnan(o_now) and o_now > 0 and not np.isnan(c_now) and c_now > 0:
                intraday_ret[si, di] = (c_now - o_now) / o_now
            if not np.isnan(c_prev) and c_prev > 0 and not np.isnan(c_now) and c_now > 0:
                ctc_ret[si, di] = (c_now - c_prev) / c_prev

    # ================================================================
    # PRECOMPUTE GROUP-LEVEL AGGREGATES FOR EACH RETURN TYPE
    # ================================================================
    print("[Signals] Computing cross-group z-scores...", flush=True)

    # For each return type, compute:
    #   group_total_avg[group_name][di] = average return of that group
    # Then:
    #   all_groups_avg[di] = grand mean across groups
    #   all_groups_std[di] = std across groups
    # Then z-score for each commodity:
    #   z[si, di] = (own_return - all_groups_avg) / all_groups_std

    def compute_cross_group_zscores(ret_array):
        """Compute z-scores of each commodity's return vs cross-group distribution."""
        # group_total: group_name -> array[ND]
        grp_total = {}
        for grp in group_names:
            arr = np.full(ND, np.nan)
            members = gm_map[grp]
            for di in range(1, ND):
                vals = [ret_array[sk, di] for sk in members if not np.isnan(ret_array[sk, di])]
                if vals:
                    arr[di] = np.mean(vals)
            grp_total[grp] = arr

        # all_groups_avg/std per day
        aga = np.full(ND, np.nan)
        ags = np.full(ND, np.nan)
        for di in range(1, ND):
            vals = [grp_total[g][di] for g in group_names if not np.isnan(grp_total[g][di])]
            if len(vals) >= 2:
                aga[di] = np.mean(vals)
                ags[di] = np.std(vals)

        # z-score per commodity
        z = np.full((NS, ND), np.nan)
        for si in trade_sis:
            for di in range(1, ND):
                own = ret_array[si, di]
                if np.isnan(own) or np.isnan(aga[di]) or np.isnan(ags[di]) or ags[di] < 1e-8:
                    continue
                z[si, di] = (own - aga[di]) / ags[di]

        return z, aga, ags, grp_total

    z_overnight, on_avg, on_std, on_grp = compute_cross_group_zscores(overnight_ret)
    z_intraday,  id_avg, id_std, id_grp = compute_cross_group_zscores(intraday_ret)
    z_ctc,       ctc_avg, ctc_std, ctc_grp = compute_cross_group_zscores(ctc_ret)

    # Also precompute V82 baseline z-scores (same as z_ctc, but verify naming)
    # z_ctc is the V82 D_zscore signal

    print(f"  Signals computed ({time.time()-t0:.1f}s)")
    print(f"  z_overnight: mean={np.nanmean(z_overnight):.4f}, std={np.nanstd(z_overnight):.4f}")
    print(f"  z_intraday:  mean={np.nanmean(z_intraday):.4f}, std={np.nanstd(z_intraday):.4f}")
    print(f"  z_ctc:       mean={np.nanmean(z_ctc):.4f}, std={np.nanstd(z_ctc):.4f}")

    # ================================================================
    # BACKTEST ENGINE
    # ================================================================
    def run_backtest(config, wf_test_year=None):
        """
        Config keys:
            signal:  'A_z_overnight' | 'B_z_intraday' | 'C_z_combined_open' |
                     'D_z_combined_close' | 'E_z_close_baseline'
            threshold: float (z-score cutoff, negative)
            top_n: 1 | 3
            alpha: float (for C, D signals)
            comm: float
        """
        sig_type   = config['signal']
        threshold  = config['threshold']   # z < -threshold -> buy
        top_n      = config['top_n']
        comm       = config.get('comm', COMM)
        alpha      = config.get('alpha', 0.5)

        # Determine entry/exit timing
        # Open entry (A, C): signal from di-1, enter at O[di], exit at C[di]
        # Close entry (B, D, E): signal from di, enter at C[di], exit at C[di+1]
        is_open_entry = sig_type in ('A_z_overnight', 'C_z_combined_open')

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

        cash = float(CASH0)
        positions = []
        trades = []

        for di in range(start_di, end_di):
            # Reset cash at test window start (WF mode)
            if wf_test_year is not None and di == test_start_di:
                cash = float(CASH0)
                positions = []

            # ── Close positions ──────────────────────────────────────
            closed = []
            for pos in positions:
                should_close = False
                if is_open_entry:
                    # Open-to-close: entered at today's open, exit at today's close
                    if di == pos['entry_di']:
                        should_close = True
                else:
                    # Close-to-close: entered yesterday, exit today
                    if di - pos['entry_di'] >= 1:
                        should_close = True

                if should_close:
                    cn = C[pos['si'], di]
                    if np.isnan(cn) or cn <= 0:
                        cn = pos['entry']
                    mult = MULT.get(pos['sym'], DEF_MULT)
                    mkt_val = cn * mult * abs(pos['lots'])
                    cash += mkt_val - mkt_val * comm
                    pnl = (cn - pos['entry']) * mult * pos['lots'] * pos['dir']
                    invested = pos['entry'] * mult * abs(pos['lots'])
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

            # ── Generate signals ─────────────────────────────────────
            candidates = []

            if sig_type == 'A_z_overnight':
                # z-score of overnight return vs cross-group overnight distribution
                # Signal uses data from di-1 (overnight ret[si,di] known at open of di)
                # Entry: O[si, di], Exit: C[si, di]
                for si in trade_sis:
                    if any(p['si'] == si for p in positions):
                        continue
                    o_now = O[si, di]
                    if np.isnan(o_now) or o_now <= 0:
                        continue
                    z = z_overnight[si, di]
                    if np.isnan(z):
                        continue
                    if z < -threshold:
                        candidates.append((si, -z, 1, syms[si]))

            elif sig_type == 'B_z_intraday':
                # z-score of intraday return vs cross-group intraday distribution
                # Signal uses data from di (intraday ret[si,di] known at close of di)
                # Entry: C[si, di], Exit: C[si, di+1]
                for si in trade_sis:
                    if any(p['si'] == si for p in positions):
                        continue
                    c_now = C[si, di]
                    if np.isnan(c_now) or c_now <= 0:
                        continue
                    z = z_intraday[si, di]
                    if np.isnan(z):
                        continue
                    if z < -threshold:
                        candidates.append((si, -z, 1, syms[si]))

            elif sig_type == 'C_z_combined_open':
                # z_combined = alpha * z_overnight + (1-alpha) * z_intraday
                # Both z-scores from di (overnight known at open, intraday from prior day)
                # For open entry: we can use overnight z from today (known at open)
                # and intraday z from yesterday (known at yesterday's close)
                # Entry: O[si, di], Exit: C[si, di]
                if di < 1:
                    continue
                for si in trade_sis:
                    if any(p['si'] == si for p in positions):
                        continue
                    o_now = O[si, di]
                    if np.isnan(o_now) or o_now <= 0:
                        continue
                    z_on = z_overnight[si, di]     # today's overnight z (known at open)
                    z_id = z_intraday[si, di - 1]   # yesterday's intraday z (known at close)
                    if np.isnan(z_on) or np.isnan(z_id):
                        continue
                    z_comb = alpha * z_on + (1 - alpha) * z_id
                    if z_comb < -threshold:
                        candidates.append((si, -z_comb, 1, syms[si]))

            elif sig_type == 'D_z_combined_close':
                # z_combined = alpha * z_overnight + (1-alpha) * z_intraday
                # Both from di (both known at close of di)
                # Entry: C[si, di], Exit: C[si, di+1]
                for si in trade_sis:
                    if any(p['si'] == si for p in positions):
                        continue
                    c_now = C[si, di]
                    if np.isnan(c_now) or c_now <= 0:
                        continue
                    z_on = z_overnight[si, di]
                    z_id = z_intraday[si, di]
                    if np.isnan(z_on) or np.isnan(z_id):
                        continue
                    z_comb = alpha * z_on + (1 - alpha) * z_id
                    if z_comb < -threshold:
                        candidates.append((si, -z_comb, 1, syms[si]))

            elif sig_type == 'E_z_close_baseline':
                # V82 exact baseline: z-score of 1-day close-to-close return
                # Entry: C[si, di], Exit: C[si, di+1]
                for si in trade_sis:
                    if any(p['si'] == si for p in positions):
                        continue
                    c_now = C[si, di]
                    if np.isnan(c_now) or c_now <= 0:
                        continue
                    z = z_ctc[si, di]
                    if np.isnan(z):
                        continue
                    if z < -threshold:
                        candidates.append((si, -z, 1, syms[si]))

            if not candidates:
                continue

            # Sort by score descending (most negative z = highest -z = strongest signal)
            candidates.sort(key=lambda x: -x[1])

            # Open positions (up to top_n slots)
            n_slots = top_n - len(positions)
            for si, score, direction, sym in candidates[:max(0, n_slots)]:
                if is_open_entry:
                    price = O[si, di]
                else:
                    price = C[si, di]
                if np.isnan(price) or price <= 0:
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
                    'si': si, 'entry': price, 'entry_di': di,
                    'lots': lots, 'dir': direction, 'sym': sym,
                })

        # Close remaining positions
        for pos in positions:
            ae = end_di - 1 if end_di < ND else ND - 1
            cn = C[pos['si'], ae]
            if np.isnan(cn) or cn <= 0:
                cn = pos['entry']
            mult = MULT.get(pos['sym'], DEF_MULT)
            mkt_val = cn * mult * abs(pos['lots'])
            cash += mkt_val - mkt_val * comm

        # Calculate results
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

    # Signal A: z_overnight_only (open entry, close exit)
    for thresh in [0.3, 0.5, 0.7]:
        for tn in [1, 3]:
            cid += 1
            configs.append({
                'id': cid, 'signal': 'A_z_overnight',
                'threshold': thresh, 'top_n': tn, 'comm': COMM,
                'label': f"A_zON_Z{thresh}_TN{tn}",
            })

    # Signal B: z_intraday_only (close entry, next close exit)
    for thresh in [0.3, 0.5, 0.7]:
        for tn in [1, 3]:
            cid += 1
            configs.append({
                'id': cid, 'signal': 'B_z_intraday',
                'threshold': thresh, 'top_n': tn, 'comm': COMM,
                'label': f"B_zID_Z{thresh}_TN{tn}",
            })

    # Signal C: z_combined_open (open entry, close exit)
    for alpha in [0.3, 0.5, 0.7, 0.9, 1.0]:
        for thresh in [0.3, 0.5, 0.7]:
            for tn in [1, 3]:
                cid += 1
                configs.append({
                    'id': cid, 'signal': 'C_z_combined_open',
                    'threshold': thresh, 'top_n': tn,
                    'alpha': alpha, 'comm': COMM,
                    'label': f"C_zCombO_a{alpha:.1f}_Z{thresh}_TN{tn}",
                })

    # Signal D: z_combined_close (close entry, next close exit)
    for alpha in [0.0, 0.3, 0.5, 0.7, 1.0]:
        for thresh in [0.3, 0.5, 0.7]:
            for tn in [1, 3]:
                cid += 1
                configs.append({
                    'id': cid, 'signal': 'D_z_combined_close',
                    'threshold': thresh, 'top_n': tn,
                    'alpha': alpha, 'comm': COMM,
                    'label': f"D_zCombC_a{alpha:.1f}_Z{thresh}_TN{tn}",
                })

    # Signal E: V82 baseline (close entry, next close exit)
    for thresh in [0.3, 0.5, 0.7]:
        for tn in [1, 3]:
            cid += 1
            configs.append({
                'id': cid, 'signal': 'E_z_close_baseline',
                'threshold': thresh, 'top_n': tn, 'comm': COMM,
                'label': f"E_V82base_Z{thresh}_TN{tn}",
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
    # FULL-PERIOD RESULTS
    # ================================================================
    print(f"\n{'=' * 130}")
    print("  FULL-PERIOD RESULTS (Top 30)")
    print(f"{'=' * 130}")
    print(f"  {'#':>3} | {'Label':<40} | {'Ann':>9} | {'WR':>6} | {'N':>5} | {'AvgPnL':>7} | {'MDD':>7}")
    print("-" * 110)
    for i, r in enumerate(results[:30]):
        print(f"  {i+1:>3} | {r['label']:<40} | {r['ann']:>+8.1f}% | {r['wr']:>5.1f}% | {r['n']:>5} | {r['avg_pnl']:>+6.3f}% | {r['mdd']:>6.1f}%")

    # ================================================================
    # SIGNAL COMPARISON (best per signal type, full period)
    # ================================================================
    print(f"\n{'=' * 130}")
    print("  SIGNAL COMPARISON (Best per signal type, full period)")
    print(f"{'=' * 130}")
    print(f"  {'Signal':<25} | {'Ann':>9} | {'WR':>6} | {'N':>5} | {'AvgPnL':>7} | {'MDD':>7} | Best Config")
    print("-" * 130)

    best_per_sig = {}
    for r in results:
        s = r['config']['signal']
        if s not in best_per_sig:
            best_per_sig[s] = r

    sig_order = ['A_z_overnight', 'B_z_intraday', 'C_z_combined_open',
                 'D_z_combined_close', 'E_z_close_baseline']
    for sig in sig_order:
        if sig in best_per_sig:
            b = best_per_sig[sig]
            print(f"  {sig:<25} | {b['ann']:>+8.1f}% | {b['wr']:>5.1f}% | {b['n']:>5} | {b['avg_pnl']:>+6.3f}% | {b['mdd']:>6.1f}% | {b['label']}")

    # Alpha vs V82 baseline
    v82_base = best_per_sig.get('E_z_close_baseline')
    if v82_base:
        print(f"\n  V82 Baseline (E_z_close_baseline): {v82_base['ann']:>+8.1f}%")
        for sig in ['A_z_overnight', 'B_z_intraday', 'C_z_combined_open', 'D_z_combined_close']:
            if sig in best_per_sig:
                diff = best_per_sig[sig]['ann'] - v82_base['ann']
                tag = "BETTER" if diff > 0 else "WORSE"
                print(f"  {sig:<25} {best_per_sig[sig]['ann']:>+8.1f}%  ({tag} {diff:>+.1f}%)")

    # ================================================================
    # ALPHA SENSITIVITY (for C and D signals)
    # ================================================================
    print(f"\n{'=' * 130}")
    print("  ALPHA SENSITIVITY (best threshold/tn per alpha, for combined signals)")
    print(f"{'=' * 130}")
    print(f"  {'Signal':<20} | {'Alpha':>5} | {'Best Ann':>10} | {'WR':>6} | {'N':>5} | {'MDD':>7} | Best Config")
    print("-" * 100)

    for sig in ['C_z_combined_open', 'D_z_combined_close']:
        for alpha in ([0.3, 0.5, 0.7, 0.9, 1.0] if sig == 'C_z_combined_open'
                      else [0.0, 0.3, 0.5, 0.7, 1.0]):
            sub = [r for r in results
                   if r['config']['signal'] == sig
                   and abs(r['config'].get('alpha', -1) - alpha) < 0.01]
            if sub:
                best = sub[0]
                print(f"  {sig:<20} | {alpha:>5.1f} | {best['ann']:>+9.1f}% | {best['wr']:>5.1f}% | {best['n']:>5} | {best['mdd']:>6.1f}% | {best['label']}")
            else:
                print(f"  {sig:<20} | {alpha:>5.1f} | {'N/A':>10} |")

    # ================================================================
    # THRESHOLD SENSITIVITY
    # ================================================================
    print(f"\n{'=' * 130}")
    print("  THRESHOLD SENSITIVITY (best per threshold, across all signals)")
    print(f"{'=' * 130}")
    print(f"  {'Signal':<25} | {'Z_thresh':>8} | {'Best Ann':>10} | {'WR':>6} | {'N':>5} | Best Config")
    print("-" * 100)

    for sig in sig_order:
        for thresh in [0.3, 0.5, 0.7]:
            sub = [r for r in results
                   if r['config']['signal'] == sig
                   and abs(r['config']['threshold'] - thresh) < 0.01]
            if sub:
                best = sub[0]
                print(f"  {sig:<25} | {thresh:>8.1f} | {best['ann']:>+9.1f}% | {best['wr']:>5.1f}% | {best['n']:>5} | {best['label']}")

    # ================================================================
    # WALK-FORWARD (Top 15 configs + best per signal)
    # ================================================================
    wf_years = [2020, 2021, 2022, 2023, 2024, 2025]

    # Collect configs for WF: top 15 overall + best per signal type
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
        wf_row = {'label': cfg['label'], 'signal': cfg['signal'], 'windows': {}, 'mdd': {}}
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
    header2 = f"  {'Signal':<25} | {'WF Avg':>8} |"
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
            row_str = f"  {sig:<25} | {avg:>+7.1f}% |"
            for v in vals:
                row_str += f" {v:>+7.1f}% |"
            row_str += f" {pos}/6 | {avg_mdd:>6.1f}%"
            print(row_str)

    # ================================================================
    # OVERNIGHT vs INTRADAY Z-SCORE CORRELATION
    # ================================================================
    print(f"\n{'=' * 130}")
    print("  Z-SCORE CORRELATION ANALYSIS")
    print(f"{'=' * 130}")
    on_vals = z_overnight[~np.isnan(z_overnight)]
    id_vals = z_intraday[~np.isnan(z_intraday)]
    ctc_vals = z_ctc[~np.isnan(z_ctc)]
    print(f"  z_overnight: N={len(on_vals)}, mean={np.mean(on_vals):.4f}, std={np.std(on_vals):.4f}")
    print(f"  z_intraday:  N={len(id_vals)}, mean={np.mean(id_vals):.4f}, std={np.std(id_vals):.4f}")
    print(f"  z_ctc:       N={len(ctc_vals)}, mean={np.mean(ctc_vals):.4f}, std={np.std(ctc_vals):.4f}")

    # Pairwise correlation between z-scores for same commodity-day
    valid_mask = ~np.isnan(z_overnight) & ~np.isnan(z_intraday)
    if valid_mask.sum() > 100:
        corr_on_id = np.corrcoef(z_overnight[valid_mask].flatten(),
                                  z_intraday[valid_mask].flatten())[0, 1]
        print(f"  Correlation(z_overnight, z_intraday): {corr_on_id:.4f}")

    valid_mask2 = ~np.isnan(z_overnight) & ~np.isnan(z_ctc)
    if valid_mask2.sum() > 100:
        corr_on_ctc = np.corrcoef(z_overnight[valid_mask2].flatten(),
                                   z_ctc[valid_mask2].flatten())[0, 1]
        print(f"  Correlation(z_overnight, z_ctc):      {corr_on_ctc:.4f}")

    valid_mask3 = ~np.isnan(z_intraday) & ~np.isnan(z_ctc)
    if valid_mask3.sum() > 100:
        corr_id_ctc = np.corrcoef(z_intraday[valid_mask3].flatten(),
                                   z_ctc[valid_mask3].flatten())[0, 1]
        print(f"  Correlation(z_intraday, z_ctc):       {corr_id_ctc:.4f}")

    # ================================================================
    # FINAL VERDICT
    # ================================================================
    print(f"\n{'=' * 130}")
    print("  FINAL VERDICT")
    print(f"{'=' * 130}")

    if v82_base:
        v82_ann = v82_base['ann']
        print(f"  V82 Baseline (E_z_close_baseline):     {v82_ann:>+8.1f}%  ({v82_base['label']})")

        # Best overall
        if results:
            best = results[0]
            print(f"  Best overall:                          {best['ann']:>+8.1f}%  ({best['label']})")
            diff_best = best['ann'] - v82_ann
            if diff_best > 0:
                print(f"  >>> DECOMPOSED Z-SCORE ADDS +{diff_best:.1f}% OVER V82 <<<")
            else:
                print(f"  >>> DECOMPOSED Z-SCORE DOES NOT IMPROVE ON V82 ({diff_best:+.1f}%) <<<")

        # Best overnight-only
        if 'A_z_overnight' in best_per_sig:
            a = best_per_sig['A_z_overnight']
            diff_a = a['ann'] - v82_ann
            print(f"  A_z_overnight (open entry):            {a['ann']:>+8.1f}%  ({diff_a:+.1f}% vs V82)")

        # Best intraday-only
        if 'B_z_intraday' in best_per_sig:
            b = best_per_sig['B_z_intraday']
            diff_b = b['ann'] - v82_ann
            print(f"  B_z_intraday (close entry):            {b['ann']:>+8.1f}%  ({diff_b:+.1f}% vs V82)")

        # Best combined-open
        if 'C_z_combined_open' in best_per_sig:
            c = best_per_sig['C_z_combined_open']
            diff_c = c['ann'] - v82_ann
            print(f"  C_z_combined_open:                     {c['ann']:>+8.1f}%  ({diff_c:+.1f}% vs V82)")

        # Best combined-close
        if 'D_z_combined_close' in best_per_sig:
            d = best_per_sig['D_z_combined_close']
            diff_d = d['ann'] - v82_ann
            print(f"  D_z_combined_close:                    {d['ann']:>+8.1f}%  ({diff_d:+.1f}% vs V82)")

        # Which alpha is best?
        print(f"\n  Best alpha for combined signals:")
        for sig_label, sig_key in [('C (open entry)', 'C_z_combined_open'),
                                    ('D (close entry)', 'D_z_combined_close')]:
            sub = [r for r in results if r['config']['signal'] == sig_key]
            if sub:
                best_alpha = sub[0]['config'].get('alpha', 0)
                print(f"    {sig_label}: best alpha={best_alpha:.1f}, ann={sub[0]['ann']:>+8.1f}%")

        # Is overnight z-score more predictive than intraday?
        a_ann = best_per_sig.get('A_z_overnight', {}).get('ann', 0)
        b_ann = best_per_sig.get('B_z_intraday', {}).get('ann', 0)
        if a_ann > b_ann:
            print(f"\n  >>> OVERNIGHT z-score IS MORE PREDICTIVE ({a_ann:+.1f}% vs {b_ann:+.1f}%) <<<")
        else:
            print(f"\n  >>> INTRADAY z-score IS MORE PREDICTIVE ({b_ann:+.1f}% vs {a_ann:+.1f}%) <<<")

    print(f"\n  Total time: {time.time()-t_start:.1f}s")
    print("=" * 120)


if __name__ == '__main__':
    main()
