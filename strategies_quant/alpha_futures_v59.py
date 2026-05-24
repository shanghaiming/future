"""
Alpha Futures V59 — Shared Capital: V52 Pair Trading + V34b Momentum
====================================================================
Combines two 1-day hold strategies with shared capital:

  Strategy A (V52): Pair mean-reversion
    - Z-score on raw spread, LB10
    - Z threshold: [0.8, 1.0, 1.2]
    - 1-day hold, MP1

  Strategy B (V34b): Group momentum lag
    - grp_mom_excl_self - own_mom5
    - Entry: score > threshold, take top commodity
    - 1-day hold

Key insight: Capital turns over daily with 1-day hold, so both strategies
can use full capital on different days. No need to permanently split.

Portfolio allocation schemes:
  1. priority  - pairs first; if no pair signal, use momentum
  2. split7030 - always 70% pairs, 30% momentum
  3. split5050 - equal allocation
  4. regime    - vol_ratio regime (low=pairs, high=momentum, mid=both 50/50)
  5. both      - run both simultaneously at all times, 50/50

~135 configs with walk-forward validation for best (2023, 2024).
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

GROUP_MAP = {
    'rbfi': 'ferrous', 'hcfi': 'ferrous', 'ifi': 'ferrous', 'jfi': 'ferrous', 'jmfi': 'ferrous',
    'cufi': 'nonferrous', 'alfi': 'nonferrous', 'znfi': 'nonferrous', 'nifi': 'nonferrous',
    'afi': 'oils', 'mfi': 'oils', 'yfi': 'oils', 'pfi': 'oils', 'cfi': 'oils',
    'scfi': 'energy', 'mafi': 'energy', 'bfi': 'energy', 'fufi': 'energy',
    'ppfi': 'chemical', 'vfi': 'chemical', 'egfi': 'chemical', 'pgfi': 'chemical',
}

PAIRS = [
    ('rbfi', 'ifi'), ('hcfi', 'ifi'), ('hcfi', 'rbfi'),
    ('jfi', 'jmfi'), ('mafi', 'scfi'), ('fufi', 'scfi'),
    ('bfi', 'scfi'), ('mfi', 'afi'), ('yfi', 'afi'),
    ('pfi', 'yfi'), ('ppfi', 'mafi'), ('vfi', 'mafi'),
    ('egfi', 'mafi'),
]

PAIR_LABEL = {
    ('rbfi', 'ifi'):  'rebar/iron_ore',
    ('hcfi', 'ifi'):  'hotcoil/iron_ore',
    ('hcfi', 'rbfi'): 'hotcoil/rebar',
    ('jfi', 'jmfi'):  'coke/coal',
    ('mafi', 'scfi'): 'methanol/crude',
    ('fufi', 'scfi'): 'fueloil/crude',
    ('bfi', 'scfi'):  'bitumen/crude',
    ('mfi', 'afi'):   'meal/soybean',
    ('yfi', 'afi'):   'soyoil/soybean',
    ('pfi', 'yfi'):   'palm/soyoil',
    ('ppfi', 'mafi'): 'PP/methanol',
    ('vfi', 'mafi'):  'PVC/methanol',
    ('egfi', 'mafi'): 'EG/methanol',
}


def main():
    t_start = time.time()
    print("=" * 140)
    print("Alpha Futures V59 -- Shared Capital: V52 Pair Trading + V34b Momentum (1-Day Hold)")
    print("Core: Both strategies use 1-day hold, capital turns over daily")
    print("Allocation: [priority, 70/30, 50/50, regime, both] x [Z threshold] x [mom threshold]")
    print("=" * 140)

    NS, ND, dates, C, O, H, L, V, syms, sym_set = load_all_data()
    sym_to_si = {syms[si]: si for si in range(NS)}

    # ========================================
    # BUILD GROUP / PAIR STRUCTURES
    # ========================================
    group_members = {}
    for si in range(NS):
        grp = GROUP_MAP.get(syms[si])
        if grp is None:
            continue
        if grp not in group_members:
            group_members[grp] = []
        group_members[grp].append(si)

    pair_indices = []
    for down_sym, up_sym in PAIRS:
        down_si = sym_to_si.get(down_sym, -1)
        up_si = sym_to_si.get(up_sym, -1)
        if down_si >= 0 and up_si >= 0:
            pair_indices.append((down_si, up_si, down_sym, up_sym))

    print(f"  {NS} commodities, {ND} days, {len(group_members)} groups, {len(pair_indices)} active pairs")

    # ========================================
    # PRECOMPUTE ALL SIGNALS
    # ========================================
    print("\n[Signals] Computing all signals...", flush=True)
    t0 = time.time()

    # --- Pair spreads ---
    spreads = {}
    for down_si, up_si, down_sym, up_sym in pair_indices:
        spread = np.full(ND, np.nan)
        for di in range(ND):
            pd_val = C[down_si, di]
            pu_val = C[up_si, di]
            if not np.isnan(pd_val) and not np.isnan(pu_val):
                spread[di] = pd_val - pu_val
        spreads[(down_si, up_si)] = spread

    # --- Pair z-scores (LB=10) ---
    PAIR_LB = 10
    pair_data = {}
    for down_si, up_si, down_sym, up_sym in pair_indices:
        sp = spreads[(down_si, up_si)]
        z = np.full(ND, np.nan)
        for di in range(PAIR_LB, ND):
            window = sp[di - PAIR_LB:di]
            valid = window[~np.isnan(window)]
            if len(valid) >= PAIR_LB * 0.8:
                sp_mean = np.mean(valid)
                sp_std = np.std(valid, ddof=1)
                if sp_std > 1e-10:
                    z[di] = (sp[di] - sp_mean) / sp_std
        pair_data[(down_si, up_si)] = {
            'z': z,
            'down_sym': down_sym,
            'up_sym': up_sym,
        }

    # --- Momentum at lag 5 ---
    mom5 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            c_now = C[si, di]
            c_prev = C[si, di - 5]
            if not np.isnan(c_now) and not np.isnan(c_prev) and c_prev > 0:
                mom5[si, di] = (c_now - c_prev) / c_prev

    # --- Group momentum excluding self (lag=5) ---
    grp_mom5 = np.full((NS, ND), np.nan)
    for grp, members in group_members.items():
        for di in range(5, ND):
            for sj in members:
                ms = []
                for sk in members:
                    if sk == sj:
                        continue
                    m = mom5[sk, di]
                    if not np.isnan(m):
                        ms.append(m)
                if ms:
                    grp_mom5[sj, di] = np.mean(ms)

    # --- Volatility ratio for regime detection ---
    daily_ret = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(1, ND):
            c0 = C[si, di - 1]
            c1 = C[si, di]
            if not np.isnan(c0) and not np.isnan(c1) and c0 > 0:
                daily_ret[si, di] = (c1 - c0) / c0

    vol20 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            window = daily_ret[si, di - 20:di]
            valid = window[~np.isnan(window)]
            if len(valid) >= 15:
                vol20[si, di] = np.std(valid, ddof=1)

    vol60_avg = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(60, ND):
            window = vol20[si, di - 60:di]
            valid = window[~np.isnan(window)]
            if len(valid) >= 40:
                vol60_avg[si, di] = np.mean(valid)

    vol_ratio = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(60, ND):
            v20 = vol20[si, di]
            v60 = vol60_avg[si, di]
            if not np.isnan(v20) and not np.isnan(v60) and v60 > 1e-12:
                vol_ratio[si, di] = v20 / v60

    print(f"  All signals computed ({time.time()-t0:.1f}s)", flush=True)

    # ========================================
    # HELPER: Compute market-wide vol regime
    # ========================================
    def get_market_regime(di):
        """Average vol_ratio across all commodities with data.
        Returns 'low', 'mid', or 'high'.
        """
        vals = []
        for si in range(NS):
            vr = vol_ratio[si, di]
            if not np.isnan(vr):
                vals.append(vr)
        if not vals:
            return 'mid'
        avg = np.mean(vals)
        if avg < 0.85:
            return 'low'
        elif avg > 1.15:
            return 'high'
        return 'mid'

    # ========================================
    # COMBINED BACKTEST ENGINE
    # ========================================
    def run_combined(
        # Allocation scheme
        alloc_scheme='priority',   # priority, split7030, split5050, regime, both
        # Pair params (V52)
        z_thresh=1.0,
        # Momentum params (V34b)
        mom_threshold=0.005,
        # Walk-forward
        wf_split_year=None,
        config_name="",
    ):
        """
        Combined 1-day hold backtest with shared capital.
        Both strategies hold for exactly 1 day (exit next day).

        Allocation schemes:
          priority:    pair signal first, momentum if no pair signal
          split7030:   70% cash for pairs, 30% for momentum
          split5050:   50% cash for pairs, 50% for momentum
          regime:      low vol=pairs only, high vol=momentum only, mid=50/50
          both:        always run both with 50/50 split
        """
        cash = float(CASH0)
        trades = []

        for di in range(MIN_TRAIN, ND):
            year = dates[di].year
            if wf_split_year is not None and year < wf_split_year:
                continue

            # === End of previous day: all positions are 1-day hold, close them ===
            # We track yesterday's positions implicitly via the cash flow.
            # For 1-day hold: open today, close tomorrow.
            # Instead, we use a simpler approach: each day is a fresh trading day.
            # At the start of each day, cash = total equity (all positions were closed).

            # === DETERMINE ALLOCATION FOR TODAY ===
            # Compute regime
            regime = get_market_regime(di)

            # Determine pair/momentum capital split
            pair_frac = 0.0
            mom_frac = 0.0

            if alloc_scheme == 'priority':
                # Will decide after checking signals below
                pair_frac = -1  # sentinel: decide dynamically
                mom_frac = -1
            elif alloc_scheme == 'split7030':
                pair_frac = 0.7
                mom_frac = 0.3
            elif alloc_scheme == 'split5050':
                pair_frac = 0.5
                mom_frac = 0.5
            elif alloc_scheme == 'regime':
                if regime == 'low':
                    pair_frac = 1.0
                    mom_frac = 0.0
                elif regime == 'high':
                    pair_frac = 0.0
                    mom_frac = 1.0
                else:
                    pair_frac = 0.5
                    mom_frac = 0.5
            elif alloc_scheme == 'both':
                pair_frac = 0.5
                mom_frac = 0.5

            # === FIND PAIR SIGNALS ===
            pair_signals = []
            occupied_by_pairs = set()
            for down_si, up_si, down_sym, up_sym in pair_indices:
                z_val = pair_data[(down_si, up_si)]['z'][di]
                if np.isnan(z_val):
                    continue
                if abs(z_val) < z_thresh:
                    continue
                pair_signals.append((abs(z_val), down_si, up_si, down_sym, up_sym, z_val))

            pair_signals.sort(key=lambda x: -x[0])

            # === FIND MOMENTUM SIGNALS ===
            mom_signals = []
            for si in range(NS):
                sym = syms[si]
                if GROUP_MAP.get(sym) is None:
                    continue
                own = mom5[si, di]
                grp = grp_mom5[si, di]
                if np.isnan(own) or np.isnan(grp):
                    continue
                divergence = grp - own
                if divergence < mom_threshold:
                    continue
                mom_signals.append((divergence, si, sym))

            mom_signals.sort(key=lambda x: -x[0])

            # === RESOLVE PRIORITY SCHEME ===
            if alloc_scheme == 'priority':
                if pair_signals:
                    pair_frac = 1.0
                    mom_frac = 0.0
                elif mom_signals:
                    pair_frac = 0.0
                    mom_frac = 1.0
                else:
                    pair_frac = 0.0
                    mom_frac = 0.0

            # === EXECUTE PAIR TRADES (at most 1 pair, 1-day hold) ===
            if pair_frac > 0 and pair_signals:
                # Take top pair signal
                _, down_si, up_si, down_sym, up_sym, z_val = pair_signals[0]

                c_down = C[down_si, di]
                c_up = C[up_si, di]
                if not np.isnan(c_down) and c_down > 0 and not np.isnan(c_up) and c_up > 0:
                    mult_down = MULT.get(down_sym, DEF_MULT)
                    mult_up = MULT.get(up_sym, DEF_MULT)

                    pair_cash = cash * pair_frac
                    cash_per_leg = pair_cash / 2

                    lots_down = int(cash_per_leg / (c_down * mult_down * (1 + COMM)))
                    lots_up = int(cash_per_leg / (c_up * mult_up * (1 + COMM)))
                    if lots_down <= 0 or lots_up <= 0:
                        lots_down = max(1, lots_down)
                        lots_up = max(1, lots_up)
                        # scale down if needed
                        cost_test = (c_down * mult_down * lots_down + c_up * mult_up * lots_up) * (1 + COMM)
                        if cost_test > cash:
                            lots_down = 0
                            lots_up = 0

                    if lots_down > 0 and lots_up > 0:
                        cost_down = c_down * mult_down * lots_down * (1 + COMM)
                        cost_up = c_up * mult_up * lots_up * (1 + COMM)
                        total_cost = cost_down + cost_up

                        if total_cost > cash:
                            scale = cash * 0.95 / total_cost
                            lots_down = max(1, int(lots_down * scale))
                            lots_up = max(1, int(lots_up * scale))
                            cost_down = c_down * mult_down * lots_down * (1 + COMM)
                            cost_up = c_up * mult_up * lots_up * (1 + COMM)
                            total_cost = cost_down + cost_up
                            if total_cost > cash:
                                lots_down = 0
                                lots_up = 0

                        if lots_down > 0 and lots_up > 0:
                            # Direction
                            if z_val > 0:
                                pos_dir = -1  # short down + long up
                            else:
                                pos_dir = 1   # long down + short up

                            # Close next day (di+1)
                            di_exit = di + 1
                            if di_exit < ND:
                                c_down_exit = C[down_si, di_exit]
                                c_up_exit = C[up_si, di_exit]
                                if np.isnan(c_down_exit) or c_down_exit <= 0:
                                    c_down_exit = c_down
                                if np.isnan(c_up_exit) or c_up_exit <= 0:
                                    c_up_exit = c_up
                            else:
                                c_down_exit = c_down
                                c_up_exit = c_up

                            mult_d = MULT.get(down_sym, DEF_MULT)
                            mult_u = MULT.get(up_sym, DEF_MULT)

                            if pos_dir == 1:
                                pnl_down = (c_down_exit - c_down) * mult_d * lots_down
                                pnl_up = (c_up - c_up_exit) * mult_u * lots_up
                            else:
                                pnl_down = (c_down - c_down_exit) * mult_d * lots_down
                                pnl_up = (c_up_exit - c_up) * mult_u * lots_up

                            entry_val = c_down * mult_d * lots_down + c_up * mult_u * lots_up
                            exit_val = c_down_exit * mult_d * lots_down + c_up_exit * mult_u * lots_up
                            cost = entry_val * COMM + exit_val * COMM

                            total_pnl = pnl_down + pnl_up - cost
                            invested = entry_val
                            pnl_pct = total_pnl / invested * 100 if invested > 0 else 0

                            # Cash flow: pay entry, receive exit
                            if pos_dir == 1:
                                cash_flow = -c_down * mult_d * lots_down * (1 + COMM) + \
                                            c_up * mult_u * lots_up * (1 + COMM)  # long down, short up
                                cash_flow_next = c_down_exit * mult_d * lots_down * (1 - COMM) - \
                                                 c_up_exit * mult_u * lots_up * (1 - COMM)  # close
                            else:
                                cash_flow = c_down * mult_d * lots_down * (1 - COMM) - \
                                            c_up * mult_u * lots_up * (1 + COMM)  # short down, long up
                                cash_flow_next = -c_down_exit * mult_d * lots_down * (1 + COMM) + \
                                                 c_up_exit * mult_u * lots_up * (1 - COMM)  # close

                            cash += total_pnl

                            trades.append({
                                'pnl_abs': total_pnl,
                                'pnl_pct': pnl_pct,
                                'days': 1,
                                'di': di,
                                'year': year,
                                'pair': (down_sym, up_sym),
                                'pair_label': PAIR_LABEL.get((down_sym, up_sym), ''),
                                'dir': pos_dir,
                                'reason': '1day',
                                'strategy': 'pair',
                                'regime': regime,
                            })

                            occupied_by_pairs.add(down_si)
                            occupied_by_pairs.add(up_si)

            # === EXECUTE MOMENTUM TRADE (at most 1 position, 1-day hold) ===
            if mom_frac > 0 and mom_signals:
                # Take top momentum signal, avoiding pair-occupied symbols
                best_mom = None
                for _, si, sym in mom_signals:
                    if si not in occupied_by_pairs:
                        best_mom = (si, sym)
                        break

                if best_mom is not None:
                    si, sym = best_mom
                    c = C[si, di]
                    if not np.isnan(c) and c > 0:
                        mult = MULT.get(sym, DEF_MULT)
                        notional = c * mult
                        if notional > 0:
                            mom_cash = cash * mom_frac
                            lots = int(mom_cash / (notional * (1 + COMM)))
                            if lots <= 0:
                                lots = 1
                                if notional * lots * (1 + COMM) > cash:
                                    lots = 0

                            if lots > 0:
                                cost_in = notional * lots * (1 + COMM)
                                if cost_in > cash:
                                    lots = int(cash / (notional * (1 + COMM)))
                                    if lots <= 0:
                                        lots = 0
                                    cost_in = notional * lots * (1 + COMM)

                            if lots > 0:
                                # Close next day
                                di_exit = di + 1
                                if di_exit < ND:
                                    c_exit = C[si, di_exit]
                                    if np.isnan(c_exit) or c_exit <= 0:
                                        c_exit = c
                                else:
                                    c_exit = c

                                pnl = (c_exit - c) * mult * lots
                                cost_out = c_exit * mult * lots * COMM
                                total_pnl = pnl - (notional * lots * COMM) - cost_out
                                pnl_pct = total_pnl / (c * mult * lots) * 100 if c > 0 else 0

                                cash += total_pnl

                                trades.append({
                                    'pnl_abs': total_pnl,
                                    'pnl_pct': pnl_pct,
                                    'days': 1,
                                    'di': di,
                                    'year': year,
                                    'sym': sym,
                                    'dir': 1,
                                    'reason': '1day',
                                    'strategy': 'momentum',
                                    'regime': regime,
                                })

        if len(trades) < 5:
            return None

        # === COMPUTE STATS ===
        equity = float(CASH0)
        peak = float(CASH0)
        max_dd = 0.0
        for t in sorted(trades, key=lambda x: x['di']):
            equity += t['pnl_abs']
            if equity > peak:
                peak = equity
            if peak > 0:
                dd = (peak - equity) / peak * 100
                if dd > max_dd:
                    max_dd = dd

        nw = sum(1 for t in trades if t['pnl_abs'] > 0)
        wr = nw / len(trades) * 100
        avg_win = np.mean([t['pnl_pct'] for t in trades if t['pnl_abs'] > 0]) if nw > 0 else 0
        avg_loss = np.mean([abs(t['pnl_pct']) for t in trades if t['pnl_abs'] <= 0]) if nw < len(trades) else 0
        pf = (sum(t['pnl_abs'] for t in trades if t['pnl_abs'] > 0) /
              max(abs(sum(t['pnl_abs'] for t in trades if t['pnl_abs'] < 0)), 1))

        days_total = (dates[ND - 1] - dates[MIN_TRAIN]).days
        yr = max(days_total / 365.25, 0.01)
        if wf_split_year:
            first_test_di = None
            for di in range(MIN_TRAIN, ND):
                if dates[di].year >= wf_split_year:
                    first_test_di = di
                    break
            if first_test_di:
                days_total = (dates[ND - 1] - dates[first_test_di]).days
                yr = max(days_total / 365.25, 0.01)

        ann = ((cash / CASH0) ** (1 / yr) - 1) * 100

        # Sharpe approximation
        trade_pnls = [t['pnl_abs'] for t in sorted(trades, key=lambda x: x['di'])]
        if len(trade_pnls) > 1:
            rets = np.array(trade_pnls) / float(CASH0)
            mean_ret = np.mean(rets)
            std_ret = np.std(rets)
            sharpe_approx = mean_ret / std_ret * np.sqrt(252) if std_ret > 0 else 0
        else:
            sharpe_approx = 0

        # Exit reason breakdown
        reasons = {}
        for t in trades:
            r = t['reason']
            if r not in reasons:
                reasons[r] = {'n': 0, 'w': 0, 'pnl': 0.0, 'pnl_pct_sum': 0.0}
            reasons[r]['n'] += 1
            if t['pnl_abs'] > 0:
                reasons[r]['w'] += 1
            reasons[r]['pnl'] += t['pnl_abs']
            reasons[r]['pnl_pct_sum'] += t['pnl_pct']

        # Yearly breakdown
        year_stats = {}
        for t in trades:
            y = t['year']
            if y not in year_stats:
                year_stats[y] = {'n': 0, 'w': 0, 'pnl': 0.0, 'pnl_abs': 0.0,
                                 'pair_pnl': 0.0, 'mom_pnl': 0.0}
            year_stats[y]['n'] += 1
            if t['pnl_abs'] > 0:
                year_stats[y]['w'] += 1
            year_stats[y]['pnl'] += t['pnl_pct']
            year_stats[y]['pnl_abs'] += t['pnl_abs']
            if t['strategy'] == 'pair':
                year_stats[y]['pair_pnl'] += t['pnl_abs']
            else:
                year_stats[y]['mom_pnl'] += t['pnl_abs']

        # Strategy breakdown
        strat_stats = {}
        for t in trades:
            st = t['strategy']
            if st not in strat_stats:
                strat_stats[st] = {'n': 0, 'w': 0, 'pnl_abs': 0.0}
            strat_stats[st]['n'] += 1
            if t['pnl_abs'] > 0:
                strat_stats[st]['w'] += 1
            strat_stats[st]['pnl_abs'] += t['pnl_abs']

        # Per-pair breakdown
        pair_stats = {}
        for t in trades:
            if t['strategy'] == 'pair':
                p = t['pair_label']
                if p not in pair_stats:
                    pair_stats[p] = {'n': 0, 'w': 0, 'pnl': 0.0}
                pair_stats[p]['n'] += 1
                if t['pnl_abs'] > 0:
                    pair_stats[p]['w'] += 1
                pair_stats[p]['pnl'] += t['pnl_abs']

        # Regime breakdown
        regime_stats = {}
        for t in trades:
            rg = t.get('regime', 'unknown')
            if rg not in regime_stats:
                regime_stats[rg] = {'n': 0, 'w': 0, 'pnl_abs': 0.0}
            regime_stats[rg]['n'] += 1
            if t['pnl_abs'] > 0:
                regime_stats[rg]['w'] += 1
            regime_stats[rg]['pnl_abs'] += t['pnl_abs']

        # Momentum commodity breakdown
        mom_sym_stats = {}
        for t in trades:
            if t['strategy'] == 'momentum':
                s = t.get('sym', '')
                if s not in mom_sym_stats:
                    mom_sym_stats[s] = {'n': 0, 'w': 0, 'pnl': 0.0}
                mom_sym_stats[s]['n'] += 1
                if t['pnl_abs'] > 0:
                    mom_sym_stats[s]['w'] += 1
                mom_sym_stats[s]['pnl'] += t['pnl_abs']

        return {
            'name': config_name,
            'ann': round(ann, 1),
            'n': len(trades),
            'wr': round(wr, 1),
            'dd': round(max_dd, 1),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'pf': round(pf, 2),
            'sharpe': round(sharpe_approx, 2),
            'cash': round(cash, 0),
            'reasons': reasons,
            'yearly': year_stats,
            'strat_stats': strat_stats,
            'pair_stats': pair_stats,
            'regime_stats': regime_stats,
            'mom_sym_stats': mom_sym_stats,
            'trades': trades,
            'alloc_scheme': alloc_scheme,
            'z_thresh': z_thresh,
            'mom_threshold': mom_threshold,
        }

    # ========================================
    # PARAMETER SWEEP
    # ========================================
    print("\n[Backtest] Building configurations...", flush=True)
    configs = []

    alloc_schemes = ['priority', 'split7030', 'split5050', 'regime', 'both']
    z_thresholds = [0.8, 1.0, 1.2]
    mom_thresholds = [0.003, 0.005, 0.01]

    # Full-period configs: 5 x 3 x 3 = 45
    for alloc in alloc_schemes:
        for zt in z_thresholds:
            for mt in mom_thresholds:
                name = f"A_{alloc}_Z{zt:.1f}_M{mt:.3f}"
                configs.append({
                    'alloc_scheme': alloc,
                    'z_thresh': zt,
                    'mom_threshold': mt,
                    'wf_split_year': None,
                    'config_name': name,
                })

    # Walk-forward for all combos: 5 x 3 x 3 x 2 = 90
    for alloc in alloc_schemes:
        for zt in z_thresholds:
            for mt in mom_thresholds:
                for wf_year in [2023, 2024]:
                    name = f"A_{alloc}_Z{zt:.1f}_M{mt:.3f}_WF{wf_year}"
                    configs.append({
                        'alloc_scheme': alloc,
                        'z_thresh': zt,
                        'mom_threshold': mt,
                        'wf_split_year': wf_year,
                        'config_name': name,
                    })

    print(f"  {len(configs)} total configurations ({len(configs) - 90} full-period, 90 walk-forward)", flush=True)

    # ========================================
    # RUN ALL CONFIGS
    # ========================================
    print("\n[Backtest] Running...", flush=True)
    results = []

    for ci, cfg in enumerate(configs):
        r = run_combined(**cfg)
        if r is not None:
            results.append(r)
            if r['ann'] > 20:
                # Strategy breakdown
                pair_n = r['strat_stats'].get('pair', {}).get('n', 0)
                mom_n = r['strat_stats'].get('momentum', {}).get('n', 0)
                pair_pnl = r['strat_stats'].get('pair', {}).get('pnl_abs', 0)
                mom_pnl = r['strat_stats'].get('momentum', {}).get('pnl_abs', 0)
                print(f"  {r['name']:45s} | Ann {r['ann']:+7.1f}% | WR {r['wr']:5.1f}% | "
                      f"N={r['n']:5d}(P{pair_n}/M{mom_n}) | DD {r['dd']:6.1f}% | "
                      f"PF {r['pf']:4.2f} | Sh {r['sharpe']:5.2f} | "
                      f"PnL P={pair_pnl:+.0f} M={mom_pnl:+.0f}")

        if (ci + 1) % 30 == 0:
            print(f"  [{ci+1}/{len(configs)}] {len(results)} configs with results", flush=True)

    # ========================================
    # RESULTS
    # ========================================
    results.sort(key=lambda x: -x['ann'])

    wf_results = [r for r in results if '_WF' in r['name']]
    full_results = [r for r in results if '_WF' not in r['name']]

    # --- TOP 20 FULL-PERIOD ---
    print(f"\n{'=' * 160}")
    print(f"  TOP 20 FULL-PERIOD RESULTS (sorted by annual return)")
    print(f"{'=' * 160}")
    hdr = (f"  {'Config':45s} | {'Ann':>7s} | {'WR':>5s} | {'N':>7s} | {'DD':>6s} | "
           f"{'PF':>4s} | {'Sharpe':>6s} | {'AvgW':>7s} | {'AvgL':>6s} | "
           f"{'PairN':>5s} | {'MomN':>5s} | {'PairPnL':>12s} | {'MomPnL':>12s}")
    print(hdr)
    print(f"  {'-' * 155}")
    for r in full_results[:20]:
        pair_n = r['strat_stats'].get('pair', {}).get('n', 0)
        mom_n = r['strat_stats'].get('momentum', {}).get('n', 0)
        pair_pnl = r['strat_stats'].get('pair', {}).get('pnl_abs', 0)
        mom_pnl = r['strat_stats'].get('momentum', {}).get('pnl_abs', 0)
        print(f"  {r['name']:45s} | {r['ann']:+7.1f}% | {r['wr']:5.1f}% | "
              f"{r['n']:7d} | {r['dd']:6.1f}% | {r['pf']:4.2f} | "
              f"{r['sharpe']:6.2f} | {r['avg_win']:+6.2f}% | {r['avg_loss']:5.2f}% | "
              f"{pair_n:5d} | {mom_n:5d} | {pair_pnl:+12.0f} | {mom_pnl:+12.0f}")

    # --- TOP 10 WALK-FORWARD ---
    if wf_results:
        wf_results.sort(key=lambda x: -x['ann'])
        print(f"\n  TOP 10 WALK-FORWARD RESULTS (out-of-sample)")
        print(f"  {'-' * 155}")
        for r in wf_results[:10]:
            pair_n = r['strat_stats'].get('pair', {}).get('n', 0)
            mom_n = r['strat_stats'].get('momentum', {}).get('n', 0)
            print(f"  {r['name']:55s} | Ann {r['ann']:+7.1f}% | WR {r['wr']:5.1f}% | "
                  f"N={r['n']:4d}(P{pair_n}/M{mom_n}) | DD {r['dd']:6.1f}% | "
                  f"PF {r['pf']:4.2f} | Sh {r['sharpe']:5.2f}")

    # --- ALLOCATION SCHEME COMPARISON ---
    print(f"\n{'=' * 160}")
    print(f"  ALLOCATION SCHEME COMPARISON (best per scheme)")
    print(f"{'=' * 160}")
    best_by_scheme = {}
    for r in full_results:
        sc = r['alloc_scheme']
        if sc not in best_by_scheme or r['ann'] > best_by_scheme[sc]['ann']:
            best_by_scheme[sc] = r

    for sc in alloc_schemes:
        if sc in best_by_scheme:
            r = best_by_scheme[sc]
            pair_n = r['strat_stats'].get('pair', {}).get('n', 0)
            mom_n = r['strat_stats'].get('momentum', {}).get('n', 0)
            pair_pnl = r['strat_stats'].get('pair', {}).get('pnl_abs', 0)
            mom_pnl = r['strat_stats'].get('momentum', {}).get('pnl_abs', 0)
            total_pnl = pair_pnl + mom_pnl
            pair_pct = pair_pnl / total_pnl * 100 if total_pnl != 0 else 0
            mom_pct = mom_pnl / total_pnl * 100 if total_pnl != 0 else 0
            print(f"\n  {sc:12s}: {r['name']}")
            print(f"    Ann={r['ann']:+.1f}%  WR={r['wr']:.1f}%  N={r['n']}(P{pair_n}/M{mom_n})  "
                  f"DD={r['dd']:.1f}%  PF={r['pf']:.2f}  Sharpe={r['sharpe']:.2f}")
            print(f"    PnL contribution: Pairs {pair_pct:.0f}%({pair_pnl:+.0f}) | "
                  f"Momentum {mom_pct:.0f}%({mom_pnl:+.0f})")

    # --- Z THRESHOLD COMPARISON ---
    print(f"\n  Z-THRESHOLD COMPARISON (averaged across all allocation schemes):")
    for zt in z_thresholds:
        zt_results = [r for r in full_results if r['z_thresh'] == zt]
        if zt_results:
            avg_ann = np.mean([r['ann'] for r in zt_results])
            avg_wr = np.mean([r['wr'] for r in zt_results])
            avg_n = np.mean([r['n'] for r in zt_results])
            best = max(zt_results, key=lambda x: x['ann'])
            print(f"    Z={zt:.1f}: {len(zt_results)} configs | "
                  f"Avg Ann={avg_ann:+.1f}%  Avg WR={avg_wr:.1f}%  Avg N={avg_n:.0f} | "
                  f"Best Ann={best['ann']:+.1f}% ({best['name']})")

    # --- MOMENTUM THRESHOLD COMPARISON ---
    print(f"\n  MOMENTUM THRESHOLD COMPARISON (averaged across all allocation schemes):")
    for mt in mom_thresholds:
        mt_results = [r for r in full_results if r['mom_threshold'] == mt]
        if mt_results:
            avg_ann = np.mean([r['ann'] for r in mt_results])
            avg_wr = np.mean([r['wr'] for r in mt_results])
            avg_n = np.mean([r['n'] for r in mt_results])
            best = max(mt_results, key=lambda x: x['ann'])
            print(f"    Mom={mt:.3f}: {len(mt_results)} configs | "
                  f"Avg Ann={avg_ann:+.1f}%  Avg WR={avg_wr:.1f}%  Avg N={avg_n:.0f} | "
                  f"Best Ann={best['ann']:+.1f}% ({best['name']})")

    # --- BEST CONFIG FULL DETAIL ---
    if full_results:
        best = full_results[0]
        print(f"\n{'=' * 160}")
        print(f"  BEST CONFIG DETAIL: {best['name']}")
        print(f"  Ann={best['ann']:+.1f}%  WR={best['wr']:.1f}%  N={best['n']}  "
              f"DD={best['dd']:.1f}%  PF={best['pf']:.2f}  Sharpe={best['sharpe']:.2f}  "
              f"Final={best['cash']:.0f}")
        print(f"{'=' * 160}")

        # Strategy contribution breakdown
        print(f"\n  STRATEGY CONTRIBUTION BREAKDOWN:")
        total_pnl = sum(st['pnl_abs'] for st in best['strat_stats'].values())
        for st_name in ['pair', 'momentum']:
            if st_name in best['strat_stats']:
                st = best['strat_stats'][st_name]
                st_wr = st['w'] / max(st['n'], 1) * 100
                st_pct = st['pnl_abs'] / total_pnl * 100 if total_pnl != 0 else 0
                print(f"    {st_name:12s}: {st['n']:5d} trades  WR={st_wr:5.1f}%  "
                      f"PnL={st['pnl_abs']:+12.0f} ({st_pct:.0f}%)")

        # Regime distribution
        print(f"\n  REGIME DISTRIBUTION:")
        total_trades = sum(rs['n'] for rs in best['regime_stats'].values())
        for rg in sorted(best['regime_stats'].keys(), key=lambda x: -best['regime_stats'][x]['n']):
            rs = best['regime_stats'][rg]
            rwr = rs['w'] / max(rs['n'], 1) * 100
            pct = rs['n'] / max(total_trades, 1) * 100
            print(f"    {rg:12s}: {rs['n']:5d} trades ({pct:5.1f}%)  "
                  f"WR={rwr:5.1f}%  Abs={rs['pnl_abs']:+12.0f}")

        # Per-pair breakdown
        if best.get('pair_stats'):
            print(f"\n  PER-PAIR BREAKDOWN:")
            for p in sorted(best['pair_stats'].keys(), key=lambda x: -best['pair_stats'][x]['n']):
                ps = best['pair_stats'][p]
                wr_p = ps['w'] / max(ps['n'], 1) * 100
                print(f"    {p:25s}: {ps['n']:5d} trades  WR={wr_p:5.1f}%  Abs={ps['pnl']:+12.0f}")

        # Per-momentum-commodity breakdown (top 10)
        if best.get('mom_sym_stats'):
            print(f"\n  TOP MOMENTUM COMMODITIES:")
            sorted_mom = sorted(best['mom_sym_stats'].items(), key=lambda x: -x[1]['pnl'])
            for sym, ms in sorted_mom[:10]:
                wr_s = ms['w'] / max(ms['n'], 1) * 100
                print(f"    {sym:10s}: {ms['n']:4d} trades  WR={wr_s:5.1f}%  Abs={ms['pnl']:+10.0f}")

        print(f"\n  YEARLY BREAKDOWN (Pair vs Momentum contribution):")
        for y in sorted(best['yearly'].keys()):
            s = best['yearly'][y]
            wr_y = s['w'] / max(s['n'], 1) * 100
            total_y = s['pair_pnl'] + s['mom_pnl']
            pair_pct = s['pair_pnl'] / total_y * 100 if total_y != 0 else 0
            mom_pct = s['mom_pnl'] / total_y * 100 if total_y != 0 else 0
            print(f"    {y}: {s['n']:5d}t  WR={wr_y:5.1f}%  Total={total_y:+12.0f}  "
                  f"Pair={s['pair_pnl']:+10.0f}({pair_pct:4.0f}%)  "
                  f"Mom={s['mom_pnl']:+10.0f}({mom_pct:4.0f}%)")

        print(f"\n  EXIT REASON BREAKDOWN:")
        for reason, s in sorted(best['reasons'].items(), key=lambda x: -x[1]['n']):
            rwr = s['w'] / max(s['n'], 1) * 100
            print(f"    {reason:12s}: {s['n']:5d} trades  WR={rwr:5.1f}%  "
                  f"PnL={s['pnl_pct_sum']:+.1f}%  Abs={s['pnl']:+.0f}")

    # --- YEARLY FOR TOP 5 ---
    if len(full_results) >= 2:
        print(f"\n  YEARLY BREAKDOWN FOR TOP 5 CONFIGS:")
        for idx, r in enumerate(full_results[:5]):
            print(f"\n  #{idx+1}: {r['name']} (Ann={r['ann']:+.1f}%, WR={r['wr']:.1f}%, "
                  f"DD={r['dd']:.1f}%, Sharpe={r['sharpe']:.2f}, N={r['n']})")
            for y in sorted(r['yearly'].keys()):
                ys = r['yearly'][y]
                wr_y = ys['w'] / ys['n'] * 100 if ys['n'] > 0 else 0
                total_y = ys['pair_pnl'] + ys['mom_pnl']
                print(f"    {y}: {ys['n']:4d}t  WR={wr_y:5.1f}%  Total={total_y:+10.0f}  "
                      f"(P={ys['pair_pnl']:+8.0f} M={ys['mom_pnl']:+8.0f})")

    # --- STRATEGY CONTRIBUTION ACROSS TOP 20 ---
    if full_results:
        print(f"\n  STRATEGY CONTRIBUTION ACROSS TOP 20 CONFIGS:")
        strat_agg = {}
        for r in full_results[:20]:
            for st, ss in r['strat_stats'].items():
                if st not in strat_agg:
                    strat_agg[st] = {'n': 0, 'w': 0, 'pnl_abs': 0.0}
                strat_agg[st]['n'] += ss['n']
                strat_agg[st]['w'] += ss['w']
                strat_agg[st]['pnl_abs'] += ss['pnl_abs']

        total_agg = sum(sa['n'] for sa in strat_agg.values())
        for st in sorted(strat_agg.keys(), key=lambda x: -strat_agg[x]['pnl_abs']):
            sa = strat_agg[st]
            swr = sa['w'] / max(sa['n'], 1) * 100
            pct = sa['n'] / max(total_agg, 1) * 100
            print(f"    {st:12s}: {sa['n']:6d} trades ({pct:5.1f}%)  "
                  f"WR={swr:5.1f}%  Total Abs={sa['pnl_abs']:+14.0f}")

    # --- REGIME PROFITABILITY ACROSS TOP 20 ---
    if full_results:
        print(f"\n  REGIME PROFITABILITY ACROSS TOP 20 CONFIGS:")
        regime_agg = {}
        for r in full_results[:20]:
            for rg, rs in r['regime_stats'].items():
                if rg not in regime_agg:
                    regime_agg[rg] = {'n': 0, 'w': 0, 'pnl_abs': 0.0}
                regime_agg[rg]['n'] += rs['n']
                regime_agg[rg]['w'] += rs['w']
                regime_agg[rg]['pnl_abs'] += rs['pnl_abs']

        total_agg = sum(ra['n'] for ra in regime_agg.values())
        for rg in sorted(regime_agg.keys(), key=lambda x: -regime_agg[x]['n']):
            ra = regime_agg[rg]
            rwr = ra['w'] / max(ra['n'], 1) * 100
            pct = ra['n'] / max(total_agg, 1) * 100
            print(f"    {rg:12s}: {ra['n']:6d} trades ({pct:5.1f}%)  "
                  f"WR={rwr:5.1f}%  Total Abs={ra['pnl_abs']:+14.0f}")

    # --- PAIR PROFITABILITY ACROSS TOP 20 ---
    if full_results:
        print(f"\n  PAIR PROFITABILITY ACROSS TOP 20 CONFIGS:")
        pair_agg = {}
        for r in full_results[:20]:
            for p, ps in r['pair_stats'].items():
                if p not in pair_agg:
                    pair_agg[p] = {'n': 0, 'w': 0, 'pnl': 0.0}
                pair_agg[p]['n'] += ps['n']
                pair_agg[p]['w'] += ps['w']
                pair_agg[p]['pnl'] += ps['pnl']

        for p in sorted(pair_agg.keys(), key=lambda x: -pair_agg[x]['pnl']):
            ps = pair_agg[p]
            wr_p = ps['w'] / max(ps['n'], 1) * 100
            print(f"    {p:25s}: {ps['n']:5d} trades  WR={wr_p:5.1f}%  Total Abs={ps['pnl']:+12.0f}")

    # --- V52 AND V34b STANDALONE COMPARISON ---
    print(f"\n  STANDALONE STRATEGY COMPARISON:")
    pair_only = [r for r in full_results if r['strat_stats'].get('momentum', {}).get('n', 0) == 0]
    mom_only = [r for r in full_results if r['strat_stats'].get('pair', {}).get('n', 0) == 0]
    combined = [r for r in full_results
                if r['strat_stats'].get('pair', {}).get('n', 0) > 0
                and r['strat_stats'].get('momentum', {}).get('n', 0) > 0]

    if pair_only:
        best_p = max(pair_only, key=lambda x: x['ann'])
        print(f"    Best PAIRS_ONLY:   {best_p['name']:45s}  Ann={best_p['ann']:+.1f}%  "
              f"WR={best_p['wr']:.1f}%  N={best_p['n']}  DD={best_p['dd']:.1f}%")
    if mom_only:
        best_m = max(mom_only, key=lambda x: x['ann'])
        print(f"    Best MOM_ONLY:     {best_m['name']:45s}  Ann={best_m['ann']:+.1f}%  "
              f"WR={best_m['wr']:.1f}%  N={best_m['n']}  DD={best_m['dd']:.1f}%")
    if combined:
        best_c = max(combined, key=lambda x: x['ann'])
        print(f"    Best COMBINED:     {best_c['name']:45s}  Ann={best_c['ann']:+.1f}%  "
              f"WR={best_c['wr']:.1f}%  N={best_c['n']}  DD={best_c['dd']:.1f}%")

    # How many combined beat standalone?
    if pair_only and combined:
        best_pair_ann = max(r['ann'] for r in pair_only)
        beating = sum(1 for r in combined if r['ann'] > best_pair_ann)
        print(f"    Combined configs beating best pair-only ({best_pair_ann:+.1f}%): "
              f"{beating}/{len(combined)}")
    if mom_only and combined:
        best_mom_ann = max(r['ann'] for r in mom_only)
        beating = sum(1 for r in combined if r['ann'] > best_mom_ann)
        print(f"    Combined configs beating best mom-only ({best_mom_ann:+.1f}%): "
              f"{beating}/{len(combined)}")

    # --- WALK-FORWARD DETAILED ---
    if wf_results:
        wf_results.sort(key=lambda x: -x['ann'])
        print(f"\n{'=' * 160}")
        print(f"  WALK-FORWARD DETAILED RESULTS")
        print(f"{'=' * 160}")
        for r in wf_results[:15]:
            pair_n = r['strat_stats'].get('pair', {}).get('n', 0)
            mom_n = r['strat_stats'].get('momentum', {}).get('n', 0)
            pair_pnl = r['strat_stats'].get('pair', {}).get('pnl_abs', 0)
            mom_pnl = r['strat_stats'].get('momentum', {}).get('pnl_abs', 0)
            print(f"  {r['name']:55s} | Ann {r['ann']:+7.1f}% | WR {r['wr']:5.1f}% | "
                  f"N={r['n']:4d}(P{pair_n}/M{mom_n}) | DD {r['dd']:6.1f}% | "
                  f"PF {r['pf']:4.2f} | Sh {r['sharpe']:5.2f} | "
                  f"PnL P={pair_pnl:+.0f} M={mom_pnl:+.0f}")

        # WF by allocation scheme
        print(f"\n  WALK-FORWARD BEST PER ALLOCATION SCHEME:")
        for sc in alloc_schemes:
            sc_wf = [r for r in wf_results if sc in r['name']]
            if sc_wf:
                best_wf = max(sc_wf, key=lambda x: x['ann'])
                pair_n = best_wf['strat_stats'].get('pair', {}).get('n', 0)
                mom_n = best_wf['strat_stats'].get('momentum', {}).get('n', 0)
                print(f"    {sc:12s}: {best_wf['name']}")
                print(f"      Ann={best_wf['ann']:+.1f}%  WR={best_wf['wr']:.1f}%  "
                      f"N={best_wf['n']}(P{pair_n}/M{mom_n})  DD={best_wf['dd']:.1f}%  "
                      f"PF={best_wf['pf']:.2f}  Sharpe={best_wf['sharpe']:.2f}")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.1f}s")
    print("=" * 160)


if __name__ == '__main__':
    main()
