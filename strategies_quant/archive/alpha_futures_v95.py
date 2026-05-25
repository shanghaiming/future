"""
Alpha Futures V95 -- Regime-Filtered Overnight Cross-Group Z-Score
===================================================================
V92 champion: +4282% annual with overnight cross-group z-score.
Signal: z_overnight = (own_overnight - all_groups_avg) / all_groups_std;
        z < -threshold -> buy at C[di], sell at C[di+1].

V95 IDEA: Test whether REGIME FILTERING can improve V92.
Maybe the overnight z-score signal works better in certain market conditions.

SIGNALS:
  A) V92_baseline:     Pure overnight z-score (control)
  B) HIGH_SPREAD_DAYS: Only trade when cross-group dispersion is high
  C) TRENDING_MARKET:  Only trade when |all_groups_avg| < 0.005 (flat/choppy)
  D) CONTRARIAN_REGIME: Only trade when market gapped UP overnight
  E) MOMENTUM_REGIME:  Only trade when market gapped DOWN overnight
  F) VOLATILITY_REGIME: Trade only in specific volatility quintile
  G) CONSECUTIVE_SIGNAL: Two consecutive days of overnight weakness
  H) REVERSAL_AFTER_STRENGTH: Today's gap reverses yesterday's direction

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

# -- Group map (srfi removed from chemical, added cffi to soft) --
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
    print("Alpha Futures V95 -- Regime-Filtered Overnight Cross-Group Z-Score")
    print("=" * 130)

    # -- Load data --
    print("\n[Data] Loading...", flush=True)
    t_start = time.time()
    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)
    years = sorted(set(d.year for d in dates))
    print(f"  {NS} commodities, {ND} days, years: {years}")

    # -- Build group membership --
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
    # PRECOMPUTE OVERNIGHT RETURNS AND CROSS-GROUP Z-SCORES
    # ================================================================
    print("\n[Signals] Computing overnight returns and cross-group z-scores...", flush=True)
    t0 = time.time()

    # overnight_ret[si, di] = (O[di] - C[di-1]) / C[di-1]
    overnight_ret = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(1, ND):
            c_prev = C[si, di - 1]
            o_now = O[si, di]
            if not np.isnan(c_prev) and c_prev > 0 and not np.isnan(o_now) and o_now > 0:
                overnight_ret[si, di] = (o_now - c_prev) / c_prev

    # Group-level aggregates
    # group_avg_on[grp][di] = mean overnight return of group members
    group_avg_on = {}
    for grp in group_names:
        arr = np.full(ND, np.nan)
        members = gm_map[grp]
        for di in range(1, ND):
            vals = [overnight_ret[sk, di] for sk in members
                    if not np.isnan(overnight_ret[sk, di])]
            if vals:
                arr[di] = np.mean(vals)
        group_avg_on[grp] = arr

    # all_groups_avg[di], all_groups_std[di] = mean and std of group averages
    aga = np.full(ND, np.nan)  # all_groups_avg_overnight
    ags = np.full(ND, np.nan)  # all_groups_std_overnight
    for di in range(1, ND):
        vals = [group_avg_on[g][di] for g in group_names
                if not np.isnan(group_avg_on[g][di])]
        if len(vals) >= 2:
            aga[di] = np.mean(vals)
            ags[di] = np.std(vals)

    # z_overnight[si, di] for each commodity
    z_overnight = np.full((NS, ND), np.nan)
    for si in trade_sis:
        for di in range(1, ND):
            own = overnight_ret[si, di]
            if np.isnan(own) or np.isnan(aga[di]) or np.isnan(ags[di]) or ags[di] < 1e-8:
                continue
            z_overnight[si, di] = (own - aga[di]) / ags[di]

    # ================================================================
    # PRECOMPUTE REGIME FILTERS
    # ================================================================
    print("[Signals] Computing regime filters...", flush=True)

    # B) HIGH_SPREAD: median of ags over all valid days
    ags_valid = ags[~np.isnan(ags)]
    ags_median = np.median(ags_valid)
    print(f"  Cross-group spread: median={ags_median:.6f}, "
          f"mean={np.mean(ags_valid):.6f}, std={np.std(ags_valid):.6f}")

    # C) TRENDING_MARKET: |aga| < 0.005 means flat
    n_flat = np.sum((~np.isnan(aga)) & (np.abs(aga) < 0.005))
    n_trend = np.sum((~np.isnan(aga)) & (np.abs(aga) >= 0.005))
    print(f"  Flat days (|aga|<0.005): {n_flat}, Trending days: {n_trend}")

    # D) CONTRARIAN: aga > 0 (market gapped up)
    n_up = np.sum((~np.isnan(aga)) & (aga > 0))
    n_down = np.sum((~np.isnan(aga)) & (aga < 0))
    print(f"  Market gapped UP: {n_up}, DOWN: {n_down}")

    # F) VOLATILITY quintiles: 20-day rolling std of aga
    vol_20 = np.full(ND, np.nan)
    for di in range(20, ND):
        window = aga[max(0, di - 20):di]
        valid = window[~np.isnan(window)]
        if len(valid) >= 10:
            vol_20[di] = np.std(valid)
    # Assign quintiles (computed over expanding window to avoid lookahead)
    vol_quintile = np.full(ND, np.nan)
    for di in range(40, ND):
        if np.isnan(vol_20[di]):
            continue
        past_vols = vol_20[40:di]
        valid_past = past_vols[~np.isnan(past_vols)]
        if len(valid_past) >= 20:
            qcut = np.percentile(valid_past, [20, 40, 60, 80])
            if vol_20[di] <= qcut[0]:
                vol_quintile[di] = 1
            elif vol_20[di] <= qcut[1]:
                vol_quintile[di] = 2
            elif vol_20[di] <= qcut[2]:
                vol_quintile[di] = 3
            elif vol_20[di] <= qcut[3]:
                vol_quintile[di] = 4
            else:
                vol_quintile[di] = 5

    for q in range(1, 6):
        cnt = np.sum(vol_quintile == q)
        print(f"  Vol quintile Q{q}: {cnt} days")

    # G) CONSECUTIVE: z < -threshold both yesterday and today
    # H) REVERSAL: z today < -threshold AND z yesterday > 0
    # These are checked inline in the signal function

    print(f"  Signals computed ({time.time()-t0:.1f}s)")

    # ================================================================
    # BACKTEST ENGINE
    # ================================================================
    def run_backtest(config, wf_test_year=None):
        """
        Config keys:
            signal: 'A_baseline' | 'B_high_spread' | 'C_flat_market' |
                    'D_contrarian' | 'E_momentum' |
                    'F_vol_Q1' .. 'F_vol_Q5' |
                    'G_consecutive' | 'H_reversal'
            threshold: float
            top_n: 1 | 3
            comm: float
        """
        sig_type  = config['signal']
        threshold = config['threshold']
        top_n     = config['top_n']
        comm      = config.get('comm', COMM)

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

            # -- Close positions (1-day hold, close entry -> next close exit) --
            closed = []
            for pos in positions:
                if di - pos['entry_di'] >= 1:
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
                    })
                    closed.append(pos)
            for pos in closed:
                positions.remove(pos)

            # -- Regime filter: skip this day if not in regime? --
            skip_day = False

            if sig_type == 'A_baseline':
                pass  # no filter

            elif sig_type == 'B_high_spread':
                if np.isnan(ags[di]) or ags[di] <= ags_median:
                    skip_day = True

            elif sig_type == 'C_flat_market':
                if np.isnan(aga[di]) or abs(aga[di]) >= 0.005:
                    skip_day = True

            elif sig_type == 'D_contrarian':
                if np.isnan(aga[di]) or aga[di] <= 0:
                    skip_day = True

            elif sig_type == 'E_momentum':
                if np.isnan(aga[di]) or aga[di] >= 0:
                    skip_day = True

            elif sig_type.startswith('F_vol_Q'):
                target_q = int(sig_type[-1])
                if np.isnan(vol_quintile[di]) or int(vol_quintile[di]) != target_q:
                    skip_day = True

            elif sig_type == 'G_consecutive':
                # Need z from yesterday too -- filter applied per commodity below
                pass

            elif sig_type == 'H_reversal':
                # Need z from yesterday too -- filter applied per commodity below
                pass

            if skip_day:
                continue

            # -- Generate signals --
            candidates = []
            for si in trade_sis:
                if any(p['si'] == si for p in positions):
                    continue
                c_now = C[si, di]
                if np.isnan(c_now) or c_now <= 0:
                    continue
                z = z_overnight[si, di]
                if np.isnan(z):
                    continue
                if z >= -threshold:
                    continue

                # Additional per-commodity regime filters
                if sig_type == 'G_consecutive':
                    if di < 1:
                        continue
                    z_yest = z_overnight[si, di - 1]
                    if np.isnan(z_yest) or z_yest >= -threshold:
                        continue

                elif sig_type == 'H_reversal':
                    if di < 1:
                        continue
                    z_yest = z_overnight[si, di - 1]
                    if np.isnan(z_yest) or z_yest <= 0:
                        continue

                candidates.append((si, -z, syms[si]))

            if not candidates:
                continue

            # Sort by strength (most negative z = highest -z)
            candidates.sort(key=lambda x: -x[1])

            # Open positions (up to top_n slots)
            n_slots = top_n - len(positions)
            for si, score, sym in candidates[:max(0, n_slots)]:
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
                    'lots': lots, 'dir': 1, 'sym': sym,
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

    signal_types = [
        'A_baseline', 'B_high_spread', 'C_flat_market',
        'D_contrarian', 'E_momentum',
        'F_vol_Q1', 'F_vol_Q2', 'F_vol_Q3', 'F_vol_Q4', 'F_vol_Q5',
        'G_consecutive', 'H_reversal',
    ]

    for sig in signal_types:
        for thresh in [0.3, 0.5]:
            for tn in [1, 3]:
                cid += 1
                configs.append({
                    'id': cid, 'signal': sig,
                    'threshold': thresh, 'top_n': tn, 'comm': COMM,
                    'label': f"{sig}_Z{thresh}_TN{tn}",
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
    print(f"\n{'=' * 140}")
    print("  FULL-PERIOD RESULTS (All configs, sorted by annual return)")
    print(f"{'=' * 140}")
    print(f"  {'#':>3} | {'Label':<35} | {'Ann':>10} | {'WR':>6} | {'N':>5} | {'AvgPnL':>7} | {'MDD':>7} | {'Final':>14}")
    print("-" * 130)
    for i, r in enumerate(results):
        print(f"  {i+1:>3} | {r['label']:<35} | {r['ann']:>+9.1f}% | {r['wr']:>5.1f}% | "
              f"{r['n']:>5} | {r['avg_pnl']:>+6.3f}% | {r['mdd']:>6.1f}% | {r['final_cash']:>14,.0f}")

    # ================================================================
    # SIGNAL COMPARISON (best per signal type)
    # ================================================================
    print(f"\n{'=' * 140}")
    print("  REGIME COMPARISON (Best per regime, full period)")
    print(f"{'=' * 140}")
    print(f"  {'Regime':<35} | {'Ann':>10} | {'WR':>6} | {'N':>5} | {'AvgPnL':>7} | {'MDD':>7} | Best Config")
    print("-" * 140)

    best_per_sig = {}
    for r in results:
        s = r['config']['signal']
        if s not in best_per_sig:
            best_per_sig[s] = r

    for sig in signal_types:
        if sig in best_per_sig:
            b = best_per_sig[sig]
            print(f"  {sig:<35} | {b['ann']:>+9.1f}% | {b['wr']:>5.1f}% | "
                  f"{b['n']:>5} | {b['avg_pnl']:>+6.3f}% | {b['mdd']:>6.1f}% | {b['label']}")

    # ================================================================
    # VS BASELINE COMPARISON
    # ================================================================
    print(f"\n{'=' * 140}")
    print("  REGIME FILTER IMPACT (vs A_baseline)")
    print(f"{'=' * 140}")

    baseline = best_per_sig.get('A_baseline')
    if baseline:
        b_ann = baseline['ann']
        print(f"  A_baseline (V92 control): {b_ann:>+9.1f}%  WR={baseline['wr']:.1f}%  "
              f"N={baseline['n']}  MDD={baseline['mdd']:.1f}%  ({baseline['label']})")
        print()
        print(f"  {'Regime':<35} | {'Ann':>10} | {'Delta':>10} | {'WR':>6} | {'N':>5} | {'MDD':>7} | Verdict")
        print("-" * 130)

        for sig in signal_types[1:]:
            if sig in best_per_sig:
                r = best_per_sig[sig]
                delta = r['ann'] - b_ann
                tag = "BETTER" if delta > 100 else ("SIMILAR" if delta > -100 else "WORSE")
                print(f"  {sig:<35} | {r['ann']:>+9.1f}% | {delta:>+9.1f}% | "
                      f"{r['wr']:>5.1f}% | {r['n']:>5} | {r['mdd']:>6.1f}% | {tag}")

    # ================================================================
    # VOLATILITY QUINTILE DETAIL
    # ================================================================
    print(f"\n{'=' * 140}")
    print("  VOLATILITY QUINTILE ANALYSIS (Which vol regime gives best edge?)")
    print(f"{'=' * 140}")
    print(f"  {'Quintile':<40} | {'Ann':>10} | {'WR':>6} | {'N':>5} | {'AvgPnL':>7} | {'MDD':>7}")
    print("-" * 100)
    for q in range(1, 6):
        sig = f'F_vol_Q{q}'
        if sig in best_per_sig:
            r = best_per_sig[sig]
            print(f"  Q{q} ({'calmest' if q==1 else 'most volatile' if q==5 else 'mid':>15}):   {sig:<25} | "
                  f"{r['ann']:>+9.1f}% | {r['wr']:>5.1f}% | {r['n']:>5} | {r['avg_pnl']:>+6.3f}% | {r['mdd']:>6.1f}%")

    # ================================================================
    # DIRECTIONAL REGIME COMPARISON
    # ================================================================
    print(f"\n{'=' * 140}")
    print("  DIRECTIONAL REGIME: Contrarian (D) vs Momentum (E)")
    print(f"{'=' * 140}")
    contra = best_per_sig.get('D_contrarian')
    momentum = best_per_sig.get('E_momentum')
    if contra and momentum:
        print(f"  D_contrarian (market gapped UP, buy losers):   {contra['ann']:>+9.1f}%  WR={contra['wr']:.1f}%  N={contra['n']}")
        print(f"  E_momentum  (market gapped DOWN, buy losers):  {momentum['ann']:>+9.1f}%  WR={momentum['wr']:.1f}%  N={momentum['n']}")
        if contra['ann'] > momentum['ann']:
            print(f"  >>> CONTRARIAN (buy losers when market up) is STRONGER <<<")
        else:
            print(f"  >>> MOMENTUM (buy losers when market down) is STRONGER <<<")

    # ================================================================
    # CONSECUTIVE vs REVERSAL
    # ================================================================
    print(f"\n{'=' * 140}")
    print("  SIGNAL PATTERN: Consecutive (G) vs Reversal (H)")
    print(f"{'=' * 140}")
    consec = best_per_sig.get('G_consecutive')
    reversal = best_per_sig.get('H_reversal')
    if consec and reversal:
        print(f"  G_consecutive (2 days weak):  {consec['ann']:>+9.1f}%  WR={consec['wr']:.1f}%  N={consec['n']}")
        print(f"  H_reversal (weak after str):  {reversal['ann']:>+9.1f}%  WR={reversal['wr']:.1f}%  N={reversal['n']}")
        if consec['ann'] > reversal['ann']:
            print(f"  >>> CONSECUTIVE WEAKNESS is STRONGER <<<")
        else:
            print(f"  >>> REVERSAL AFTER STRENGTH is STRONGER <<<")

    # ================================================================
    # TRADE COUNT ANALYSIS (which regime trades most/least)
    # ================================================================
    print(f"\n{'=' * 140}")
    print("  TRADE FREQUENCY BY REGIME")
    print(f"{'=' * 140}")
    print(f"  {'Regime':<35} | {'N Trades':>8} | {'AvgPnL':>7} | {'Ann':>10} | Efficiency")
    print("-" * 100)
    for sig in signal_types:
        if sig in best_per_sig:
            r = best_per_sig[sig]
            eff = r['ann'] / max(r['n'], 1)
            print(f"  {sig:<35} | {r['n']:>8} | {r['avg_pnl']:>+6.3f}% | {r['ann']:>+9.1f}% | {eff:>+.2f}%/trade")

    # ================================================================
    # WALK-FORWARD
    # ================================================================
    wf_years = [2020, 2021, 2022, 2023, 2024, 2025]

    # Collect WF configs: top 15 + best per signal type
    wf_configs = list(results[:15])
    for sig in signal_types:
        if sig in best_per_sig:
            r = best_per_sig[sig]
            if r['config'] not in [w['config'] for w in wf_configs]:
                wf_configs.append(r)

    print(f"\n{'=' * 160}")
    print(f"  WALK-FORWARD ({len(wf_configs)} configs)")
    print(f"{'=' * 160}")

    header = f"  {'#':>3} | {'Config':<35} | {'Avg':>8} |"
    for yr in wf_years:
        header += f" {yr:>7} |"
    header += f" {'Pos':>4} | {'MDD':>7}"
    print(header)
    print("-" * 160)

    wf_rows = []
    for i, r in enumerate(wf_configs):
        cfg = r['config']
        wf_row = {'label': cfg['label'], 'signal': cfg['signal'],
                  'windows': {}, 'mdd': {}}
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

        row_str = f"  {i+1:>3} | {wf_row['label']:<35} | {avg:>+7.1f}% |"
        for v in vals:
            row_str += f" {v:>+7.1f}% |"
        row_str += f" {pos}/6 | {avg_mdd:>6.1f}%"
        print(row_str)

    # ================================================================
    # WF COMPARISON PER REGIME (best per signal type)
    # ================================================================
    print(f"\n{'=' * 140}")
    print("  WALK-FORWARD COMPARISON (Best per regime type)")
    print(f"{'=' * 140}")
    header2 = f"  {'Regime':<35} | {'WF Avg':>8} |"
    for yr in wf_years:
        header2 += f" {yr:>7} |"
    header2 += f" {'Pos':>4} | Avg MDD"
    print(header2)
    print("-" * 140)

    for sig in signal_types:
        wf_match = [w for w in wf_rows if w['signal'] == sig]
        if wf_match:
            wf = wf_match[0]
            vals = [wf['windows'].get(yr, 0) for yr in wf_years]
            avg = np.mean(vals) if vals else 0
            pos = sum(1 for v in vals if v > 0)
            avg_mdd = np.mean(list(wf['mdd'].values())) if wf['mdd'] else 0
            row_str = f"  {sig:<35} | {avg:>+7.1f}% |"
            for v in vals:
                row_str += f" {v:>+7.1f}% |"
            row_str += f" {pos}/6 | {avg_mdd:>6.1f}%"
            print(row_str)

    # ================================================================
    # FINAL VERDICT
    # ================================================================
    print(f"\n{'=' * 140}")
    print("  FINAL VERDICT")
    print(f"{'=' * 140}")

    if baseline:
        b_ann = baseline['ann']
        print(f"  V92 Baseline (A_baseline, no filter): {b_ann:>+9.1f}%  ({baseline['label']})")

        # Find best regime
        best_regime = None
        best_regime_ann = -1e18
        for sig in signal_types[1:]:
            if sig in best_per_sig:
                r = best_per_sig[sig]
                if r['ann'] > best_regime_ann:
                    best_regime_ann = r['ann']
                    best_regime = sig

        if best_regime:
            delta = best_regime_ann - b_ann
            r = best_per_sig[best_regime]
            print(f"  Best regime filter ({best_regime}):  {best_regime_ann:>+9.1f}%  "
                  f"WR={r['wr']:.1f}%  N={r['n']}  MDD={r['mdd']:.1f}%  ({r['label']})")
            print(f"  Delta vs baseline:                   {delta:>+9.1f}%")
            if delta > 100:
                print(f"  >>> REGIME FILTERING IMPROVES V92 by +{delta:.0f}% <<<")
            elif delta > 0:
                print(f"  >>> REGIME FILTERING MARGINALLY IMPROVES V92 <<<")
            else:
                print(f"  >>> REGIME FILTERING DOES NOT IMPROVE V92 <<<")

        # Key findings
        print(f"\n  KEY FINDINGS:")

        # Which volatility quintile is best?
        best_q = None
        best_q_ann = -1e18
        for q in range(1, 6):
            sig = f'F_vol_Q{q}'
            if sig in best_per_sig:
                if best_per_sig[sig]['ann'] > best_q_ann:
                    best_q_ann = best_per_sig[sig]['ann']
                    best_q = q
        if best_q:
            print(f"    Best volatility quintile: Q{best_q} ({best_q_ann:>+9.1f}%)")

        # Contrarian vs Momentum
        if contra and momentum:
            winner = "Contrarian (D)" if contra['ann'] > momentum['ann'] else "Momentum (E)"
            print(f"    Directional winner: {winner}")

        # Consecutive vs Reversal
        if consec and reversal:
            winner = "Consecutive (G)" if consec['ann'] > reversal['ann'] else "Reversal (H)"
            print(f"    Pattern winner: {winner}")

        # High spread helpful?
        hs = best_per_sig.get('B_high_spread')
        if hs:
            print(f"    High-spread filter: {hs['ann']:>+9.1f}% ({'helps' if hs['ann'] > b_ann else 'hurts'})")

        # Flat market helpful?
        fm = best_per_sig.get('C_flat_market')
        if fm:
            print(f"    Flat-market filter: {fm['ann']:>+9.1f}% ({'helps' if fm['ann'] > b_ann else 'hurts'})")

    print(f"\n  Total time: {time.time()-t_start:.1f}s")
    print("=" * 130)


if __name__ == '__main__':
    main()
