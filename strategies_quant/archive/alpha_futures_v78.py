"""
Alpha Futures V78 -- Rank-Based Group Divergence
=================================================
V74 champion: +2185% with extended groups, LB=1, 1-day hold.
Signal: group_mean_return > own_return -> buy (catch-up).

V78 NEW IDEA: Rank-based divergence within each group.
For each group, rank all members by today's return (1=best, N=worst).
If a commodity's rank is worse than its MEDIAN rank over past W days
-> it's unusually weak relative to its group -> buy (expect rank mean reversion).

This is a DIFFERENT signal: not "group moved more than me" but
"my rank within group dropped unusually low." The rank transformation
is robust to outliers and non-normal distributions.

SIGNALS:
  A: Rank divergence (own_rank - median_rank > threshold -> long)
  B: Z-score of rank (rank is more than 1 std below median -> long)
  C: Quantile signal (rank in bottom 20% of rolling distribution -> long)
  D: Combined rank + V74 momentum (both agree -> trade)

Walk-forward: 6 windows (2020-2025).
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
    print("=" * 100)
    print("Alpha Futures V78 -- Rank-Based Group Divergence")
    print("=" * 100)

    # Load data
    print("\n[Data] Loading...", flush=True)
    t_start = time.time()
    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)
    years = sorted(set(d.year for d in dates))
    print(f"  {NS} commodities, {ND} days, years: {years}")

    # ================================================================
    # PRECOMPUTE: daily returns, group membership, ranks
    # ================================================================
    print("\n[Signals] Computing daily returns and ranks...", flush=True)
    t0 = time.time()

    # Daily returns (1-day momentum)
    ret1 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(1, ND):
            cn = C[si, di]
            cp = C[si, di - 1]
            if not np.isnan(cn) and not np.isnan(cp) and cp > 0:
                ret1[si, di] = (cn - cp) / cp

    # Group membership
    grp_members = {}  # group_name -> list of si
    for si in range(NS):
        g = GROUP_MAP.get(syms[si])
        if g:
            grp_members.setdefault(g, []).append(si)

    trade_sis = [si for si in range(NS) if GROUP_MAP.get(syms[si])]
    grp_of = {}  # si -> group_name
    for si in trade_sis:
        grp_of[si] = GROUP_MAP[syms[si]]

    print(f"  Groups: {len(grp_members)} groups, {len(trade_sis)} tradeable commodities")
    for g, members in sorted(grp_members.items()):
        print(f"    {g}: {len(members)} members")

    # Compute daily ranks within each group (1=best return, N=worst return)
    # Higher return -> lower rank number (rank 1 = best)
    daily_rank = np.full((NS, ND), np.nan)
    for di in range(1, ND):
        for grp, members in grp_members.items():
            rets = [(si, ret1[si, di]) for si in members if not np.isnan(ret1[si, di])]
            if len(rets) < 2:
                continue
            rets.sort(key=lambda x: -x[1])  # sort descending by return
            n = len(rets)
            for rank_idx, (si, _) in enumerate(rets):
                daily_rank[si, di] = rank_idx + 1  # 1=best, n=worst

    # Precompute rolling rank statistics for windows [10, 20, 40]
    rank_windows = [10, 20, 40]
    rank_median = {}  # window -> (NS, ND) array of rolling median rank
    rank_std = {}     # window -> (NS, ND) array of rolling std rank
    rank_pctl = {}    # window -> (NS, ND) array of rolling percentile (pct of time rank is at or below current)

    for win in rank_windows:
        rmed = np.full((NS, ND), np.nan)
        rstd = np.full((NS, ND), np.nan)
        rpct = np.full((NS, ND), np.nan)
        for si in trade_sis:
            for di in range(win, ND):
                window_ranks = daily_rank[si, di - win:di]
                valid = window_ranks[~np.isnan(window_ranks)]
                if len(valid) >= 5:  # need at least 5 observations
                    rmed[si, di] = np.median(valid)
                    rstd[si, di] = np.std(valid) if np.std(valid) > 0 else 1.0
                    # percentile: fraction of time rank was >= current (worse)
                    # higher daily_rank = worse performer
                    current_rank = daily_rank[si, di]
                    if not np.isnan(current_rank):
                        rpct[si, di] = np.mean(valid >= current_rank)
        rank_median[win] = rmed
        rank_std[win] = rstd
        rank_pctl[win] = rpct

    # Also precompute V74-style group mean momentum for Signal D
    grp_mean_ret = np.full((NS, ND), np.nan)
    for di in range(1, ND):
        for grp, members in grp_members.items():
            rets = [ret1[si, di] for si in members if not np.isnan(ret1[si, di])]
            if rets:
                gm = np.mean(rets)
                for si in members:
                    grp_mean_ret[si, di] = gm

    print(f"  Signals computed ({time.time()-t0:.1f}s)")

    # ================================================================
    # SIGNAL GENERATORS
    # ================================================================

    def signal_A_rank_div(si, di, win, threshold):
        """Signal A: rank divergence. own_rank - median_rank > threshold -> long.
        Higher rank = worse performer. If current rank is much worse than median,
        rank_div > threshold -> expect rank mean reversion upward."""
        mr = rank_median[win][si, di]
        cr = daily_rank[si, di]
        if np.isnan(mr) or np.isnan(cr):
            return None, None
        rank_div = cr - mr  # positive means rank is worse than usual
        if rank_div > threshold:
            return rank_div, 1  # long
        return None, None

    def signal_B_zscore(si, di, win, threshold):
        """Signal B: z-score of rank. (rank - median) / std > threshold -> long.
        Higher rank = worse. If z-score > threshold, rank is unusually bad -> buy."""
        mr = rank_median[win][si, di]
        sr = rank_std[win][si, di]
        cr = daily_rank[si, di]
        if np.isnan(mr) or np.isnan(sr) or np.isnan(cr) or sr <= 0:
            return None, None
        z = (cr - mr) / sr
        if z > threshold:
            return z, 1  # long
        return None, None

    def signal_C_quantile(si, di, win, threshold):
        """Signal C: quantile signal. If pct of time rank was worse >= threshold -> long.
        rank_pctl measures fraction of past ranks that are >= current (worse).
        If threshold=0.8, rank is in the worst 20% historically -> buy."""
        pct = rank_pctl[win][si, di]
        cr = daily_rank[si, di]
        if np.isnan(pct) or np.isnan(cr):
            return None, None
        if pct >= threshold:
            return pct, 1  # long
        return None, None

    def signal_D_combined(si, di, win, threshold):
        """Signal D: combined rank divergence + V74 momentum.
        Both rank is unusually bad AND group return > own return -> buy."""
        mr = rank_median[win][si, di]
        cr = daily_rank[si, di]
        own = ret1[si, di]
        gm = grp_mean_ret[si, di]
        if np.isnan(mr) or np.isnan(cr) or np.isnan(own) or np.isnan(gm):
            return None, None
        rank_div = cr - mr
        mom_div = gm - own  # V74 style: group > own -> positive
        # Both conditions must hold: rank is worse than median AND group outperformed
        if rank_div > 0 and mom_div > 0:
            combined_score = rank_div * mom_div
            if combined_score > threshold:
                return combined_score, 1  # long
        return None, None

    SIGNAL_FUNCS = {
        'A_rank_div': signal_A_rank_div,
        'B_z_score': signal_B_zscore,
        'C_quantile': signal_C_quantile,
        'D_combined': signal_D_combined,
    }

    # ================================================================
    # BACKTEST ENGINE
    # ================================================================
    def run_backtest(config, wf_test_year=None):
        sig_type = config['signal_type']
        win = config['window']
        threshold = config['threshold']
        direction = config['direction']  # 'long_only' or 'long_and_short'
        top_n = config['top_n']
        sig_func = SIGNAL_FUNCS[sig_type]

        # Date range setup
        wf_mode = wf_test_year is not None
        start_di = MIN_TRAIN
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
            test_start_di = start_di
            test_end_di = ND
            end_di = ND

        cash = float(CASH0)
        positions = []
        trades = []

        for di in range(start_di, end_di):
            if wf_mode and di == test_start_di:
                cash = float(CASH0)
                positions = []

            # Close positions entered yesterday
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

            # Score candidates
            candidates = []
            for si in trade_sis:
                sym = syms[si]
                if np.isnan(C[si, di]) or C[si, di] <= 0:
                    continue
                if any(p['si'] == si for p in positions):
                    continue

                score, dirn = sig_func(si, di, win, threshold)
                if score is not None and dirn == 1:
                    candidates.append((si, score, 1))

                # Short side: when rank is unusually HIGH (much better than median),
                # short the commodity (expect rank reversal downward)
                if direction == 'long_and_short':
                    mr = rank_median[win][si, di]
                    cr = daily_rank[si, di]
                    if not np.isnan(mr) and not np.isnan(cr):
                        # Rank is unusually good (much lower = better) -> short
                        rank_adv = mr - cr  # positive means rank better than usual
                        if rank_adv > threshold:
                            candidates.append((si, rank_adv, -1))

            if not candidates:
                continue

            candidates.sort(key=lambda x: -x[1])

            n_slots = top_n - len(positions)
            for si, score, dirn in candidates[:n_slots]:
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
                    'lots': lots, 'dir': dirn, 'sym': syms[si],
                })

        # Close remaining
        for pos in positions:
            ae = ND - 1
            cn = C[pos['si'], ae]
            if np.isnan(cn) or cn <= 0:
                cn = pos['entry']
            mult = MULT.get(pos['sym'], DEF_MULT)
            mkt_val = cn * mult * pos['lots']
            cash += mkt_val - mkt_val * COMM

        if wf_mode:
            test_trades = trades
            n_days_test = test_end_di - test_start_di
            ann = annual_return(cash, CASH0, n_days_test)
        else:
            test_trades = trades
            n_days_test = ND - start_di
            ann = annual_return(cash, CASH0, n_days_test)

        wr = np.mean([1 if t['pnl_pct'] > 0 else 0 for t in test_trades]) * 100 if test_trades else 0
        n_trades = len(test_trades)

        # Max drawdown
        equity_curve = [CASH0]
        for t in test_trades:
            equity_curve.append(equity_curve[-1] * (1 + t['pnl_pct'] / 100))
        peak = CASH0
        mdd = 0
        for eq in equity_curve:
            if eq > peak:
                peak = eq
            dd = (eq - peak) / peak * 100
            if dd < mdd:
                mdd = dd

        return {
            'ann': ann, 'wr': wr, 'n': n_trades,
            'final_cash': cash, 'n_days': n_days_test,
            'mdd': mdd,
        }

    # ================================================================
    # V74 BASELINE REPRODUCTION
    # ================================================================
    def run_v74_baseline(wf_test_year=None):
        """Reproduce V74: group_mean - own > threshold -> long, LB=1, top_n=1"""
        wf_mode = wf_test_year is not None
        start_di = MIN_TRAIN
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
            test_start_di = start_di
            test_end_di = ND
            end_di = ND

        cash = float(CASH0)
        positions = []
        trades = []

        for di in range(start_di, end_di):
            if wf_mode and di == test_start_di:
                cash = float(CASH0)
                positions = []

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
                    trades.append({'pnl_pct': pnl_pct, 'di': pos['entry_di'],
                                   'year': dates[di].year if di < ND else dates[-1].year,
                                   'dir': pos['dir']})
                    closed.append(pos)
            for pos in closed:
                positions.remove(pos)

            candidates = []
            for si in trade_sis:
                if np.isnan(C[si, di]) or C[si, di] <= 0:
                    continue
                if any(p['si'] == si for p in positions):
                    continue
                own = ret1[si, di]
                gm = grp_mean_ret[si, di]
                if np.isnan(own) or np.isnan(gm):
                    continue
                div = gm - own
                if div > 0:  # V74 threshold = any positive divergence
                    candidates.append((si, div, 1))

            if not candidates:
                continue

            candidates.sort(key=lambda x: -x[1])

            n_slots = 1 - len(positions)  # V74 top_n=1
            for si, score, dirn in candidates[:max(0, n_slots)]:
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

        for pos in positions:
            ae = ND - 1
            cn = C[pos['si'], ae]
            if np.isnan(cn) or cn <= 0:
                cn = pos['entry']
            mult = MULT.get(pos['sym'], DEF_MULT)
            mkt_val = cn * mult * pos['lots']
            cash += mkt_val - mkt_val * COMM

        if wf_mode:
            n_days_test = test_end_di - test_start_di
            ann = annual_return(cash, CASH0, n_days_test)
        else:
            n_days_test = ND - start_di
            ann = annual_return(cash, CASH0, n_days_test)

        wr = np.mean([1 if t['pnl_pct'] > 0 else 0 for t in trades]) * 100 if trades else 0
        n_trades = len(trades)

        return {'ann': ann, 'wr': wr, 'n': n_trades, 'final_cash': cash, 'n_days': n_days_test}

    # ================================================================
    # SWEEP CONFIGURATIONS
    # ================================================================
    print("\n[Sweep] Generating configurations...", flush=True)

    signal_types = ['A_rank_div', 'B_z_score', 'C_quantile', 'D_combined']
    windows = [10, 20, 40]
    thresholds_by_signal = {
        'A_rank_div': [0.5, 1.0, 2.0],
        'B_z_score': [0.5, 1.0, 2.0],
        'C_quantile': [0.7, 0.8, 0.9],
        'D_combined': [0.0001, 0.001, 0.005],
    }
    directions = ['long_only', 'long_and_short']
    top_ns = [1, 3]

    configs = []
    config_id = 0
    for sig in signal_types:
        for win in windows:
            for thresh in thresholds_by_signal[sig]:
                for dirn in directions:
                    for tn in top_ns:
                        config_id += 1
                        dir_label = 'LO' if dirn == 'long_only' else 'LS'
                        configs.append({
                            'id': config_id,
                            'signal_type': sig,
                            'window': win,
                            'threshold': thresh,
                            'direction': dirn,
                            'top_n': tn,
                            'label': f"{sig}_W{win}_T{thresh}_{dir_label}_TN{tn}",
                        })

    print(f"  Total configs: {len(configs)}")

    # Run all configs (full period)
    print("\n[Sweep] Running full-period backtests...", flush=True)
    results = []
    for i, cfg in enumerate(configs):
        r = run_backtest(cfg)
        if r:
            r['config'] = cfg
            r['label'] = cfg['label']
            results.append(r)
        if (i + 1) % 50 == 0:
            print(f"  ... {i+1}/{len(configs)} done", flush=True)

    results.sort(key=lambda x: -x['ann'])

    # ================================================================
    # PRINT RESULTS BY SIGNAL TYPE
    # ================================================================
    print("\n" + "=" * 100)
    print("  FULL-PERIOD RESULTS -- TOP 5 PER SIGNAL TYPE")
    print("=" * 100)

    for sig in signal_types:
        sig_results = [r for r in results if r['config']['signal_type'] == sig]
        if not sig_results:
            continue
        print(f"\n  --- {sig} ---")
        print(f"  {'#':>3} | {'Config':<55} | {'Ann':>8} | {'WR':>5} | {'N':>5} | {'MDD':>6}")
        print("-" * 95)
        for i, r in enumerate(sig_results[:5]):
            print(f"  {i+1:>3} | {r['label']:<55} | {r['ann']:>+7.1f}% | {r['wr']:>4.1f}% | {r['n']:>5} | {r['mdd']:>5.1f}%")

    # Overall top 20
    print("\n" + "=" * 100)
    print("  OVERALL TOP 20")
    print("=" * 100)
    print(f"  {'#':>3} | {'Signal':<15} | {'Config':<45} | {'Ann':>8} | {'WR':>5} | {'N':>5} | {'MDD':>6}")
    print("-" * 100)
    for i, r in enumerate(results[:20]):
        sig_short = r['config']['signal_type']
        print(f"  {i+1:>3} | {sig_short:<15} | {r['label']:<45} | {r['ann']:>+7.1f}% | {r['wr']:>4.1f}% | {r['n']:>5} | {r['mdd']:>5.1f}%")

    # ================================================================
    # V74 BASELINE COMPARISON (FULL PERIOD)
    # ================================================================
    print("\n" + "=" * 100)
    print("  V74 BASELINE (FULL PERIOD)")
    print("=" * 100)
    v74_full = run_v74_baseline()
    print(f"  V74 baseline: Ann={v74_full['ann']:>+8.1f}%  WR={v74_full['wr']:>5.1f}%  N={v74_full['n']:>5}")

    # ================================================================
    # WALK-FORWARD FOR TOP 10 + V74
    # ================================================================
    print("\n" + "=" * 100)
    print("  WALK-FORWARD (Top 10 V78 configs + V74 baseline)")
    print("=" * 100)

    wf_years = [2020, 2021, 2022, 2023, 2024, 2025]
    wf_results = []

    # Top 10 V78
    for i, r in enumerate(results[:10]):
        cfg = r['config']
        wf_row = {'label': cfg['label'], 'signal': cfg['signal_type'], 'windows': {}}
        for yr in wf_years:
            wr = run_backtest(cfg, wf_test_year=yr)
            if wr:
                wf_row['windows'][yr] = wr['ann']
        wf_results.append(wf_row)

    # V74 baseline WF
    v74_wf = {'label': 'V74_BASELINE_LB1_TN1_LO', 'signal': 'v74', 'windows': {}}
    for yr in wf_years:
        wr = run_v74_baseline(wf_test_year=yr)
        if wr:
            v74_wf['windows'][yr] = wr['ann']
    wf_results.append(v74_wf)

    # Print WF table
    print(f"  {'#':>3} | {'Signal':<15} | {'Config':<40} | {'Avg':>8} |", end="")
    for yr in wf_years:
        print(f" {yr:>7} |", end="")
    print(f" {'Pos':>4}")
    print("-" * 140)
    for i, wf in enumerate(wf_results):
        vals = [wf['windows'].get(yr, 0) for yr in wf_years]
        avg = np.mean(vals) if vals else 0
        pos = sum(1 for v in vals if v > 0)
        sig = wf.get('signal', '')
        print(f"  {i+1:>3} | {sig:<15} | {wf['label']:<40} | {avg:>+7.1f}% |", end="")
        for v in vals:
            print(f" {v:>+7.1f}% |", end="")
        print(f" {pos}/6")

    # ================================================================
    # SIGNAL TYPE COMPARISON SUMMARY
    # ================================================================
    print("\n" + "=" * 100)
    print("  SIGNAL TYPE COMPARISON SUMMARY")
    print("=" * 100)
    print(f"  {'Signal':<20} | {'Best Ann':>10} | {'Avg Top3 Ann':>12} | {'Avg Top3 WR':>11} | {'Avg Top3 N':>10} | {'Beats V74?':>10}")
    print("-" * 95)

    v74_ann = v74_full['ann']
    for sig in signal_types:
        sig_results = [r for r in results if r['config']['signal_type'] == sig]
        if not sig_results:
            continue
        best = sig_results[0]['ann']
        top3 = sig_results[:3]
        avg_ann = np.mean([r['ann'] for r in top3])
        avg_wr = np.mean([r['wr'] for r in top3])
        avg_n = np.mean([r['n'] for r in top3])
        beats = "YES" if best > v74_ann else "NO"
        print(f"  {sig:<20} | {best:>+9.1f}% | {avg_ann:>+11.1f}% | {avg_wr:>10.1f}% | {avg_n:>10.0f} | {beats:>10}")

    print(f"  {'V74 baseline':<20} | {v74_ann:>+9.1f}% | {'---':>12} | {'---':>11} | {'---':>10} | {'---':>10}")

    # ================================================================
    # DIRECTION COMPARISON: long_only vs long_and_short
    # ================================================================
    print("\n" + "=" * 100)
    print("  DIRECTION COMPARISON: Long-Only vs Long+Short")
    print("=" * 100)
    for sig in signal_types:
        lo = [r for r in results if r['config']['signal_type'] == sig and r['config']['direction'] == 'long_only']
        ls = [r for r in results if r['config']['signal_type'] == sig and r['config']['direction'] == 'long_and_short']
        if lo:
            print(f"  {sig} Long-Only  best: {lo[0]['ann']:>+8.1f}%  {lo[0]['label']}")
        if ls:
            print(f"  {sig} Long+Short best: {ls[0]['ann']:>+8.1f}%  {ls[0]['label']}")

    # ================================================================
    # WINDOW SIZE COMPARISON
    # ================================================================
    print("\n" + "=" * 100)
    print("  WINDOW SIZE COMPARISON (best per window)")
    print("=" * 100)
    for win in windows:
        wr = [r for r in results if r['config']['window'] == win]
        if wr:
            print(f"  W={win:>2}: best={wr[0]['ann']:>+8.1f}%  {wr[0]['label']}")

    # ================================================================
    # TOP_N COMPARISON
    # ================================================================
    print("\n" + "=" * 100)
    print("  TOP_N COMPARISON (best per top_n)")
    print("=" * 100)
    for tn in top_ns:
        tr = [r for r in results if r['config']['top_n'] == tn]
        if tr:
            print(f"  TN={tn}: best={tr[0]['ann']:>+8.1f}%  {tr[0]['label']}")

    # ================================================================
    # FINAL VERDICT
    # ================================================================
    print("\n" + "=" * 100)
    print("  FINAL VERDICT")
    print("=" * 100)
    best_v78 = results[0] if results else None
    if best_v78:
        print(f"  Best V78 config: {best_v78['label']}")
        print(f"    Ann={best_v78['ann']:>+8.1f}%  WR={best_v78['wr']:>5.1f}%  N={best_v78['n']:>5}  MDD={best_v78['mdd']:>5.1f}%")
        print(f"  V74 baseline:    Ann={v74_ann:>+8.1f}%")
        if best_v78['ann'] > v74_ann:
            print(f"  --> V78 WINS by {best_v78['ann'] - v74_ann:>+.1f}% annualized!")
        else:
            print(f"  --> V74 still better by {v74_ann - best_v78['ann']:>+.1f}% annualized")

    print(f"\n  Total time: {time.time()-t_start:.1f}s")
    print("=" * 100)


if __name__ == '__main__':
    main()
