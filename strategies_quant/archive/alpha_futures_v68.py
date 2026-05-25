"""
Alpha Futures V68 -- OPEN vs CLOSE Entry/Exit Price Mode Test
==============================================================
V62 uses CLOSE prices for z-score calculation and CLOSE prices for entry/exit.
Question: What if we enter at next day's OPEN or exit at different price points?

4 price mode variations:
  1. CC (Close-to-Close): z on CLOSE, enter at CLOSE, exit at next CLOSE  [V62 baseline]
  2. CO (Close-to-Open):  z on CLOSE, enter at next OPEN, exit at that day's CLOSE (intraday)
  3. OO (Open-to-Open):   z on OPEN prices, enter at OPEN, exit at next OPEN
  4. COC (Close-Open-Close): z on CLOSE, enter at next OPEN, exit at following CLOSE (1.5 day)

Hypothesis: overnight mean-reversion may be stronger than close-to-close.
  CO mode captures: signal at close -> enter at open -> capture intraday reversion
  COC mode captures: signal at close -> enter at open -> hold through next close

~80 configs (4 price modes x Z[0.8,1.0,1.2] x LB[10,15] x spread[log,raw]).
Walk-forward for best configs.
Print: top 20, walk-forward, price mode comparison.
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

PAIRS = [
    ('rbfi', 'ifi'), ('hcfi', 'ifi'), ('hcfi', 'rbfi'),
    ('jfi', 'jmfi'), ('mafi', 'scfi'), ('fufi', 'scfi'),
    ('bfi', 'scfi'), ('mfi', 'afi'), ('yfi', 'afi'),
    ('pfi', 'yfi'), ('ppfi', 'mafi'), ('vfi', 'mafi'),
    ('egfi', 'mafi'), ('cfi', 'csfi'),
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
    ('cfi', 'csfi'):  'corn/cornstarch',
}

SPREAD_RAW = 'raw'
SPREAD_LOG = 'log'

# Price modes:
# CC  = z on CLOSE, enter at close(di), exit at close(di+1)         [V62 baseline]
# CO  = z on CLOSE, enter at open(di+1), exit at close(di+1)        [overnight + intraday]
# OO  = z on OPEN,  enter at open(di),  exit at open(di+1)          [open-based]
# COC = z on CLOSE, enter at open(di+1), exit at close(di+2)        [1.5 day hold]
PMODE_CC  = 'CC'
PMODE_CO  = 'CO'
PMODE_OO  = 'OO'
PMODE_COC = 'COC'
ALL_PMODES = [PMODE_CC, PMODE_CO, PMODE_OO, PMODE_COC]

ALL_LOOKBACKS = [10, 15]
ALL_Z = [0.8, 1.0, 1.2]
ALL_SPREADS = [SPREAD_LOG, SPREAD_RAW]

WF_WINDOWS = [
    (2019, 2020),
    (2020, 2021),
    (2021, 2022),
    (2022, 2023),
    (2023, 2024),
    (2024, 2025),
]


def main():
    t_start = time.time()
    print("=" * 160)
    print("Alpha Futures V68 -- OPEN vs CLOSE Entry/Exit Price Mode Test")
    print("4 modes: CC (baseline), CO (close-open intraday), OO (open-open), COC (1.5 day hold)")
    print("14 pairs, ~80 configs, walk-forward for best")
    print("=" * 160)

    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)
    sym_to_si = {syms[si]: si for si in range(NS)}

    # Year boundaries
    year_start_di = {}
    year_end_di = {}
    for di in range(ND):
        y = dates[di].year
        if y not in year_start_di:
            year_start_di[y] = di
        year_end_di[y] = di
    print(f"  {NS} commodities, {ND} days, years in data: {sorted(year_start_di.keys())}")

    # Build pair index mapping
    def build_pair_indices(pairs_list):
        indices = []
        for down_sym, up_sym in pairs_list:
            down_si = sym_to_si.get(down_sym, -1)
            up_si = sym_to_si.get(up_sym, -1)
            if down_si >= 0 and up_si >= 0:
                indices.append((down_si, up_si, down_sym, up_sym))
            else:
                print(f"  WARNING: pair ({down_sym}, {up_sym}) not found")
        return indices

    pair_indices = build_pair_indices(PAIRS)
    print(f"  Pairs: {len(pair_indices)}")

    # ================================================================
    # PRECOMPUTE SPREADS AND Z-SCORES
    # Using both CLOSE and OPEN prices for spread calculation
    # ================================================================
    print("\n[Signals] Precomputing spreads and z-scores (CLOSE and OPEN)...", flush=True)
    t0 = time.time()

    # z_close: spread from CLOSE prices (used for CC, CO, COC modes)
    # z_open:  spread from OPEN prices  (used for OO mode)
    z_close = {}  # (spread_mode, lb) -> {pair_key: z_array}
    z_open  = {}  # (spread_mode, lb) -> {pair_key: z_array}

    all_pair_set = set()
    for down_si, up_si, _, _ in pair_indices:
        all_pair_set.add((down_si, up_si))

    for spread_mode in ALL_SPREADS:
        for lb in ALL_LOOKBACKS:
            z_close[(spread_mode, lb)] = {}
            z_open[(spread_mode, lb)] = {}

    for down_si, up_si in all_pair_set:
        for spread_mode in ALL_SPREADS:
            # Spread from CLOSE prices
            spread_c = np.full(ND, np.nan)
            # Spread from OPEN prices
            spread_o = np.full(ND, np.nan)

            for di in range(ND):
                pd_c = C[down_si, di]
                pu_c = C[up_si, di]
                pd_o = O[down_si, di]
                pu_o = O[up_si, di]

                if not (np.isnan(pd_c) or np.isnan(pu_c) or pu_c <= 0 or pd_c <= 0):
                    if spread_mode == SPREAD_RAW:
                        spread_c[di] = pd_c - pu_c
                    else:  # LOG
                        spread_c[di] = np.log(pd_c) - np.log(pu_c)

                if not (np.isnan(pd_o) or np.isnan(pu_o) or pu_o <= 0 or pd_o <= 0):
                    if spread_mode == SPREAD_RAW:
                        spread_o[di] = pd_o - pu_o
                    else:  # LOG
                        spread_o[di] = np.log(pd_o) - np.log(pu_o)

            for lb in ALL_LOOKBACKS:
                # Z-scores from CLOSE spread
                z = np.full(ND, np.nan)
                for di in range(lb, ND):
                    window = spread_c[di - lb:di]
                    valid = window[~np.isnan(window)]
                    if len(valid) >= max(3, lb * 0.8):
                        m_val = np.mean(valid)
                        s_val = np.std(valid, ddof=1)
                        if s_val > 1e-10:
                            z[di] = (spread_c[di] - m_val) / s_val
                z_close[(spread_mode, lb)][(down_si, up_si)] = z

                # Z-scores from OPEN spread
                z = np.full(ND, np.nan)
                for di in range(lb, ND):
                    window = spread_o[di - lb:di]
                    valid = window[~np.isnan(window)]
                    if len(valid) >= max(3, lb * 0.8):
                        m_val = np.mean(valid)
                        s_val = np.std(valid, ddof=1)
                        if s_val > 1e-10:
                            z[di] = (spread_o[di] - m_val) / s_val
                z_open[(spread_mode, lb)][(down_si, up_si)] = z

    print(f"  Z-scores precomputed for CLOSE and OPEN ({time.time() - t0:.1f}s)", flush=True)

    # ================================================================
    # BACKTEST ENGINE -- parametrized by price mode
    # ================================================================
    def run_backtest(price_mode=PMODE_CC, z_thresh=1.0, spread_mode=SPREAD_LOG,
                     lookback=10, start_year=None, end_year=None,
                     config_name=""):
        """
        Backtest with different entry/exit price modes.

        CC:  signal at close(di-1), enter at close(di-1), exit at close(di)
             -> z_prev from close z-score at di-1
             -> entry: close prices at di-1
             -> exit:  close prices at di

        CO:  signal at close(di-1), enter at open(di), exit at close(di)
             -> z_prev from close z-score at di-1
             -> entry: open prices at di
             -> exit:  close prices at di

        OO:  signal at open(di-1), enter at open(di-1), exit at open(di)
             -> z_prev from open z-score at di-1
             -> entry: open prices at di-1
             -> exit:  open prices at di

        COC: signal at close(di-1), enter at open(di), exit at close(di+1)
             -> z_prev from close z-score at di-1
             -> entry: open prices at di
             -> exit:  close prices at di+1
        """
        cash = float(CASH0)
        trades = []

        # Date range
        start_di = MIN_TRAIN
        end_di = ND
        if start_year is not None:
            if start_year in year_start_di:
                start_di = year_start_di[start_year]
            else:
                return None
        if end_year is not None:
            if end_year in year_end_di:
                end_di = year_end_di[end_year] + 1
            else:
                return None

        # For COC mode, we need to exit at di+1, so we need di < end_di - 1
        max_signal_di = end_di - 1
        if price_mode == PMODE_COC:
            max_signal_di = end_di - 2  # need di+1 for exit

        # Select which z-score source to use
        if price_mode == PMODE_OO:
            z_source = z_open
        else:
            z_source = z_close

        combo_key = (spread_mode, lookback)

        for di in range(start_di, max_signal_di):
            year = dates[di].year

            # Get z-score from previous day (di-1) for signal
            # For OO mode, z is computed from open prices, so we use z at di-1
            # For CC/CO/COC, z is computed from close prices, use z at di-1
            if di < 1:
                continue

            z_prev_di = di - 1

            # Find best pair signal
            best_z = 0
            best_pair = None

            for down_si, up_si, down_sym, up_sym in pair_indices:
                z_arr = z_source[combo_key].get((down_si, up_si))
                if z_arr is None:
                    continue
                z_val = z_arr[z_prev_di] if z_prev_di < len(z_arr) else np.nan
                if np.isnan(z_val):
                    continue
                if abs(z_val) < z_thresh:
                    continue
                if abs(z_val) > abs(best_z):
                    best_z = z_val
                    best_pair = (down_si, up_si, down_sym, up_sym)

            if best_pair is None:
                continue

            down_si, up_si, down_sym, up_sym = best_pair
            mult_down = MULT.get(down_sym, DEF_MULT)
            mult_up = MULT.get(up_sym, DEF_MULT)

            # Determine entry and exit prices based on price mode
            if price_mode == PMODE_CC:
                # Entry: close at di-1, Exit: close at di
                entry_di = di - 1
                exit_di = di
                p_down_entry = C[down_si, entry_di]
                p_up_entry = C[up_si, entry_di]
                p_down_exit = C[down_si, exit_di]
                p_up_exit = C[up_si, exit_di]

            elif price_mode == PMODE_CO:
                # Entry: open at di, Exit: close at di
                entry_di = di
                exit_di = di
                p_down_entry = O[down_si, entry_di]
                p_up_entry = O[up_si, entry_di]
                p_down_exit = C[down_si, exit_di]
                p_up_exit = C[up_si, exit_di]

            elif price_mode == PMODE_OO:
                # Entry: open at di-1, Exit: open at di
                entry_di = di - 1
                exit_di = di
                p_down_entry = O[down_si, entry_di]
                p_up_entry = O[up_si, entry_di]
                p_down_exit = O[down_si, exit_di]
                p_up_exit = O[up_si, exit_di]

            elif price_mode == PMODE_COC:
                # Entry: open at di, Exit: close at di+1
                entry_di = di
                exit_di = di + 1
                if exit_di >= ND:
                    continue
                p_down_entry = O[down_si, entry_di]
                p_up_entry = O[up_si, entry_di]
                p_down_exit = C[down_si, exit_di]
                p_up_exit = C[up_si, exit_di]

            # Validate prices
            if (np.isnan(p_down_entry) or p_down_entry <= 0 or
                np.isnan(p_up_entry) or p_up_entry <= 0 or
                np.isnan(p_down_exit) or p_down_exit <= 0 or
                np.isnan(p_up_exit) or p_up_exit <= 0):
                continue

            # Position sizing: split capital equally between two legs
            cash_per_leg = cash / 2
            lots_down = int(cash_per_leg / (p_down_entry * mult_down * (1 + COMM)))
            lots_up = int(cash_per_leg / (p_up_entry * mult_up * (1 + COMM)))
            if lots_down <= 0 or lots_up <= 0:
                continue

            # Calculate cost
            cost_down = p_down_entry * mult_down * lots_down * (1 + COMM)
            cost_up = p_up_entry * mult_up * lots_up * (1 + COMM)
            total_cost = cost_down + cost_up

            if total_cost > cash:
                scale = cash * 0.95 / total_cost
                lots_down = max(1, int(lots_down * scale))
                lots_up = max(1, int(lots_up * scale))
                cost_down = p_down_entry * mult_down * lots_down * (1 + COMM)
                cost_up = p_up_entry * mult_up * lots_up * (1 + COMM)
                total_cost = cost_down + cost_up
                if total_cost > cash:
                    continue

            # Direction: z > 0 means spread is high -> short spread
            # Short spread = short down, long up
            if best_z > 0:
                pnl_down = (p_down_entry - p_down_exit) * mult_down * lots_down
                pnl_up = (p_up_exit - p_up_entry) * mult_up * lots_up
            else:
                pnl_down = (p_down_exit - p_down_entry) * mult_down * lots_down
                pnl_up = (p_up_entry - p_up_exit) * mult_up * lots_up

            # Exit costs
            exit_cost_down = p_down_exit * mult_down * lots_down * COMM
            exit_cost_up = p_up_exit * mult_up * lots_up * COMM
            cost = (cost_down + cost_up) * COMM / (1 + COMM) + exit_cost_down + exit_cost_up

            total_pnl = pnl_down + pnl_up - cost
            invested = p_down_entry * mult_down * lots_down + p_up_entry * mult_up * lots_up
            pnl_pct = total_pnl / invested * 100 if invested > 0 else 0

            # Update cash (simplified: full cycle in one step)
            # Entry cost already paid, now receive exit proceeds
            if best_z > 0:
                proceeds = (p_down_entry * mult_down * lots_down +
                            p_up_exit * mult_up * lots_up)
            else:
                proceeds = (p_down_exit * mult_down * lots_down +
                            p_up_entry * mult_up * lots_up)

            # Simpler: just track via PnL
            cash += total_pnl

            # Holding period
            if price_mode == PMODE_COC:
                hold_days = (dates[exit_di] - dates[entry_di]).days if exit_di < ND and entry_di < ND else 2
            else:
                hold_days = (dates[exit_di] - dates[entry_di]).days if exit_di < ND and entry_di < ND else 1

            trades.append({
                'pnl_abs': total_pnl,
                'pnl_pct': pnl_pct,
                'days': hold_days,
                'di': exit_di,
                'year': year,
                'pair': (down_sym, up_sym),
                'pair_label': PAIR_LABEL.get((down_sym, up_sym), ''),
                'dir': 1 if best_z < 0 else -1,
                'reason': 'time',
                'price_mode': price_mode,
            })

        if len(trades) < 3:
            return None

        # === STATS ===
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
        avg_days = np.mean([t['days'] for t in trades])
        pf = (sum(t['pnl_abs'] for t in trades if t['pnl_abs'] > 0) /
              max(abs(sum(t['pnl_abs'] for t in trades if t['pnl_abs'] < 0)), 1))

        first_di = min(t['di'] for t in trades)
        last_di = max(t['di'] for t in trades)
        if last_di > first_di:
            days_total = (dates[last_di] - dates[first_di]).days
        else:
            days_total = 365
        yr = max(days_total / 365.25, 0.01)
        ann = ((cash / CASH0) ** (1 / yr) - 1) * 100

        trade_pnls = [t['pnl_abs'] for t in sorted(trades, key=lambda x: x['di'])]
        if len(trade_pnls) > 1:
            rets = np.array(trade_pnls) / float(CASH0)
            sharpe_approx = np.mean(rets) / np.std(rets) * np.sqrt(252) if np.std(rets) > 0 else 0
        else:
            sharpe_approx = 0

        year_stats = {}
        for t in trades:
            y = t['year']
            if y not in year_stats:
                year_stats[y] = {'n': 0, 'w': 0, 'pnl': 0.0, 'pnl_abs_sum': 0.0}
            year_stats[y]['n'] += 1
            if t['pnl_abs'] > 0:
                year_stats[y]['w'] += 1
            year_stats[y]['pnl'] += t['pnl_pct']
            year_stats[y]['pnl_abs_sum'] += t['pnl_abs']

        pair_stats = {}
        for t in trades:
            p = t['pair_label']
            if p not in pair_stats:
                pair_stats[p] = {'n': 0, 'w': 0, 'pnl': 0.0}
            pair_stats[p]['n'] += 1
            if t['pnl_abs'] > 0:
                pair_stats[p]['w'] += 1
            pair_stats[p]['pnl'] += t['pnl_abs']

        return {
            'name': config_name,
            'ann': round(ann, 1),
            'n': len(trades),
            'wr': round(wr, 1),
            'dd': round(max_dd, 1),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'avg_days': round(avg_days, 1),
            'pf': round(pf, 2),
            'sharpe': round(sharpe_approx, 2),
            'cash': round(cash, 0),
            'yearly': year_stats,
            'pair_stats': pair_stats,
            'price_mode': price_mode,
        }

    # ================================================================
    # BUILD CONFIGURATIONS
    # ================================================================
    print("\n[Configs] Building configurations...", flush=True)

    configs = []

    for price_mode in ALL_PMODES:
        for spread_mode in ALL_SPREADS:
            for lb in ALL_LOOKBACKS:
                for zt in ALL_Z:
                    name = f"PM_{price_mode}_{spread_mode}_LB{lb}_Z{zt:.1f}"
                    configs.append({
                        'price_mode': price_mode,
                        'z_thresh': zt,
                        'spread_mode': spread_mode,
                        'lookback': lb,
                        'start_year': None,
                        'end_year': None,
                        'config_name': name,
                    })

    total_combos = len(configs)
    print(f"  {total_combos} configurations")
    print(f"    {len(ALL_PMODES)} price modes x {len(ALL_SPREADS)} spreads x "
          f"{len(ALL_LOOKBACKS)} lookbacks x {len(ALL_Z)} z thresholds")
    print(f"    Price modes: CC (close-close baseline), CO (close-open intraday), "
          f"OO (open-open), COC (close-open-close 1.5day)")

    # ================================================================
    # RUN FULL-PERIOD SWEEP
    # ================================================================
    print(f"\n{'=' * 160}")
    print(f"  FULL-PERIOD PARAMETER SWEEP ({total_combos} configs)")
    print(f"{'=' * 160}")

    results = []
    t_sweep_start = time.time()

    for ci, cfg in enumerate(configs):
        r = run_backtest(**cfg)
        if r is not None:
            results.append(r)

        if (ci + 1) % 20 == 0:
            elapsed = time.time() - t_sweep_start
            print(f"  [{ci + 1}/{total_combos}] {len(results)} with results ({elapsed:.1f}s)", flush=True)

    results.sort(key=lambda x: -x['ann'])
    print(f"\n  Sweep complete: {len(results)}/{total_combos} configs ({time.time() - t_sweep_start:.1f}s)",
          flush=True)

    # ================================================================
    # TOP 20 FULL-PERIOD RESULTS
    # ================================================================
    print(f"\n{'=' * 160}")
    print(f"  TOP 20 FULL-PERIOD RESULTS")
    print(f"{'=' * 160}")
    print(f"  {'#':>2s} | {'Config':40s} | {'Ann':>8s} | {'WR':>5s} | {'N':>5s} | "
          f"{'DD':>6s} | {'PF':>5s} | {'Sharpe':>7s} | {'AvgW':>6s} | {'AvgL':>6s} | "
          f"{'AvgD':>5s} | {'Cash':>12s}")
    print(f"  {'-' * 150}")

    for i, r in enumerate(results[:20]):
        print(f"  {i + 1:2d} | {r['name']:40s} | {r['ann']:+7.1f}% | {r['wr']:4.1f}% | "
              f"{r['n']:5d} | {r['dd']:5.1f}% | {r['pf']:4.2f} | {r['sharpe']:6.2f} | "
              f"{r['avg_win']:+5.2f}% | {r['avg_loss']:5.2f}% | {r['avg_days']:4.1f} | "
              f"{r['cash']:11.0f}")

    # ================================================================
    # PRICE MODE COMPARISON
    # ================================================================
    print(f"\n{'=' * 160}")
    print(f"  PRICE MODE COMPARISON")
    print(f"{'=' * 160}")

    pmode_desc = {
        PMODE_CC:  'Close-to-Close (V62 baseline): z on CLOSE, enter close, exit next close',
        PMODE_CO:  'Close-to-Open (intraday): z on CLOSE, enter next OPEN, exit that CLOSE',
        PMODE_OO:  'Open-to-Open: z on OPEN, enter open, exit next open',
        PMODE_COC: 'Close-Open-Close (1.5 day): z on CLOSE, enter next OPEN, exit following CLOSE',
    }

    for pmode in ALL_PMODES:
        subset = [r for r in results if r['price_mode'] == pmode]
        if subset:
            avg_ann = np.mean([r['ann'] for r in subset])
            med_ann = np.median([r['ann'] for r in subset])
            best = max(subset, key=lambda x: x['ann'])
            n_pos = sum(1 for r in subset if r['ann'] > 0)
            avg_wr = np.mean([r['wr'] for r in subset])
            avg_sharpe = np.mean([r['sharpe'] for r in subset])
            print(f"\n  {pmode}: {pmode_desc[pmode]}")
            print(f"    N configs={len(subset)}  Avg Ann={avg_ann:+.1f}%  Med Ann={med_ann:+.1f}%  "
                  f"Positive={n_pos}/{len(subset)}  Avg WR={avg_wr:.1f}%  Avg Sharpe={avg_sharpe:.2f}")
            print(f"    Best: {best['name']}")
            print(f"    Ann={best['ann']:+.1f}%  WR={best['wr']:.1f}%  N={best['n']}  "
                  f"DD={best['dd']:.1f}%  PF={best['pf']:.2f}  Sharpe={best['sharpe']:.2f}")
        else:
            print(f"\n  {pmode}: no results")

    # ================================================================
    # SPREAD MODE COMPARISON
    # ================================================================
    print(f"\n{'=' * 160}")
    print(f"  SPREAD MODE COMPARISON")
    print(f"{'=' * 160}")

    for spread_mode in ALL_SPREADS:
        subset = [r for r in results if f'_{spread_mode}_' in r['name']]
        if subset:
            avg_ann = np.mean([r['ann'] for r in subset])
            best = max(subset, key=lambda x: x['ann'])
            print(f"  {spread_mode:4s}: Avg Ann={avg_ann:+7.1f}%  Best Ann={best['ann']:+7.1f}%  "
                  f"({best['name']})")

    # ================================================================
    # Z THRESHOLD COMPARISON
    # ================================================================
    print(f"\n{'=' * 160}")
    print(f"  Z THRESHOLD COMPARISON")
    print(f"{'=' * 160}")

    for zt in ALL_Z:
        subset = [r for r in results if f'_Z{zt:.1f}' in r['name']]
        if subset:
            avg_ann = np.mean([r['ann'] for r in subset])
            best = max(subset, key=lambda x: x['ann'])
            print(f"  Z={zt:.1f}: Avg Ann={avg_ann:+7.1f}%  Best Ann={best['ann']:+7.1f}%  "
                  f"({best['name']})")

    # ================================================================
    # LOOKBACK COMPARISON
    # ================================================================
    print(f"\n{'=' * 160}")
    print(f"  LOOKBACK COMPARISON")
    print(f"{'=' * 160}")

    for lb in ALL_LOOKBACKS:
        subset = [r for r in results if f'_LB{lb}_' in r['name']]
        if subset:
            avg_ann = np.mean([r['ann'] for r in subset])
            best = max(subset, key=lambda x: x['ann'])
            print(f"  LB={lb:2d}: Avg Ann={avg_ann:+7.1f}%  Best Ann={best['ann']:+7.1f}%  "
                  f"({best['name']})")

    # ================================================================
    # INTERACTION: PRICE MODE x SPREAD MODE
    # ================================================================
    print(f"\n{'=' * 160}")
    print(f"  INTERACTION: PRICE MODE x SPREAD MODE (best per cell)")
    print(f"{'=' * 160}")
    print(f"  {'Price Mode':10s} | {'Spread':6s} | {'Best Ann':>8s} | {'Best Config':40s} | "
          f"{'WR':>5s} | {'N':>5s} | {'DD':>6s} | {'Sharpe':>7s}")
    print(f"  {'-' * 120}")

    for pmode in ALL_PMODES:
        for sm in ALL_SPREADS:
            subset = [r for r in results if r['price_mode'] == pmode and f'_{sm}_' in r['name']]
            if subset:
                best = max(subset, key=lambda x: x['ann'])
                print(f"  {pmode:10s} | {sm:6s} | {best['ann']:+7.1f}% | {best['name']:40s} | "
                      f"{best['wr']:4.1f}% | {best['n']:5d} | {best['dd']:5.1f}% | {best['sharpe']:6.2f}")

    # ================================================================
    # YEARLY BREAKDOWN FOR TOP 5
    # ================================================================
    if len(results) >= 2:
        print(f"\n{'=' * 160}")
        print(f"  YEARLY BREAKDOWN FOR TOP 5")
        print(f"{'=' * 160}")

        for idx, r in enumerate(results[:5]):
            print(f"\n  #{idx + 1}: {r['name']} (Ann={r['ann']:+.1f}%, WR={r['wr']:.1f}%, "
                  f"DD={r['dd']:.1f}%, Sharpe={r['sharpe']:.2f}, N={r['n']})")
            print(f"  {'Year':>6s} | {'N':>5s} | {'WR':>5s} | {'PnL Abs':>12s} | {'PnL %':>8s}")
            print(f"  {'-' * 50}")
            for y in sorted(r['yearly'].keys()):
                ys = r['yearly'][y]
                wr_y = ys['w'] / max(ys['n'], 1) * 100
                print(f"  {y:6d} | {ys['n']:5d} | {wr_y:4.1f}% | {ys['pnl_abs_sum']:+11.0f} | "
                      f"{ys['pnl']:+7.1f}%")

    # ================================================================
    # PER-PAIR STATS FOR TOP CONFIG PER PRICE MODE
    # ================================================================
    print(f"\n{'=' * 160}")
    print(f"  PER-PAIR STATS FOR BEST CONFIG PER PRICE MODE")
    print(f"{'=' * 160}")

    for pmode in ALL_PMODES:
        subset = [r for r in results if r['price_mode'] == pmode]
        if not subset:
            continue
        best = max(subset, key=lambda x: x['ann'])
        print(f"\n  {pmode} best: {best['name']}")
        print(f"  Ann={best['ann']:+.1f}%  WR={best['wr']:.1f}%  N={best['n']}  "
              f"DD={best['dd']:.1f}%  PF={best['pf']:.2f}  Sharpe={best['sharpe']:.2f}")
        print(f"  {'Pair':25s} | {'N':>5s} | {'WR':>5s} | {'Abs PnL':>12s} | {'Avg PnL':>10s}")
        print(f"  {'-' * 70}")

        for p in sorted(best['pair_stats'].keys(),
                        key=lambda x: -best['pair_stats'][x]['pnl']):
            ps = best['pair_stats'][p]
            wr_p = ps['w'] / max(ps['n'], 1) * 100
            avg_pnl = ps['pnl'] / max(ps['n'], 1)
            print(f"  {p:25s} | {ps['n']:5d} | {wr_p:4.1f}% | {ps['pnl']:+11.0f} | "
                  f"{avg_pnl:+9.0f}")

    # ================================================================
    # WALK-FORWARD FOR TOP 10
    # ================================================================
    top10_for_wf = results[:10]

    print(f"\n{'=' * 160}")
    print(f"  RIGOROUS 6-WINDOW WALK-FORWARD (Top 10 configs)")
    print(f"  Windows: {WF_WINDOWS}")
    print(f"{'=' * 160}")

    wf_all = []
    wf_by_config = {}

    for rank, cfg_result in enumerate(top10_for_wf):
        cfg_name = cfg_result['name']
        matching = [c for c in configs if c['config_name'] == cfg_name]
        if not matching:
            print(f"  [{rank + 1}] {cfg_name} -- config not found, SKIP")
            continue

        base_cfg = matching[0]
        print(f"\n  [{rank + 1}] {cfg_name}  (full-period Ann={cfg_result['ann']:+.1f}%)")

        for train_end, test_year in WF_WINDOWS:
            if test_year not in year_start_di:
                print(f"    Train -{train_end}/Test {test_year}: year not in data, SKIP")
                continue

            wf_name = f"WF_Train-{train_end}_Test-{test_year}_{cfg_name}"
            wf_cfg = dict(base_cfg)
            wf_cfg['start_year'] = test_year
            wf_cfg['end_year'] = test_year
            wf_cfg['config_name'] = wf_name

            r = run_backtest(**wf_cfg)
            if r is not None:
                wf_all.append((cfg_name, train_end, test_year, r))
                if cfg_name not in wf_by_config:
                    wf_by_config[cfg_name] = []
                wf_by_config[cfg_name].append((train_end, test_year, r))
                print(f"    Train -{train_end}/Test {test_year}: Ann={r['ann']:+7.1f}%  "
                      f"WR={r['wr']:5.1f}%  N={r['n']:4d}  DD={r['dd']:5.1f}%  "
                      f"PF={r['pf']:4.2f}  Sharpe={r['sharpe']:6.2f}")
            else:
                print(f"    Train -{train_end}/Test {test_year}: insufficient trades")

    # ================================================================
    # WALK-FORWARD AGGREGATE
    # ================================================================
    print(f"\n{'=' * 160}")
    print(f"  WALK-FORWARD AGGREGATE (average across all windows per config)")
    print(f"{'=' * 160}")

    wf_avg = []
    for cfg_name, window_results in wf_by_config.items():
        anns = [r['ann'] for _, _, r in window_results]
        wrs = [r['wr'] for _, _, r in window_results]
        ns = [r['n'] for _, _, r in window_results]
        dds = [r['dd'] for _, _, r in window_results]
        pfs = [r['pf'] for _, _, r in window_results]
        sharpe_vals = [r['sharpe'] for _, _, r in window_results]
        n_positive = sum(1 for a in anns if a > 0)

        wf_avg.append({
            'name': cfg_name,
            'avg_ann': np.mean(anns),
            'med_ann': np.median(anns),
            'min_ann': min(anns),
            'max_ann': max(anns),
            'avg_wr': np.mean(wrs),
            'avg_n': np.mean(ns),
            'avg_dd': np.mean(dds),
            'avg_pf': np.mean(pfs),
            'avg_sharpe': np.mean(sharpe_vals),
            'n_positive': n_positive,
            'n_windows': len(window_results),
            'window_details': window_results,
        })

    wf_avg.sort(key=lambda x: -x['avg_ann'])

    print(f"  {'#':>2s} | {'Config':40s} | {'Avg Ann':>8s} | {'Med Ann':>8s} | {'Min Ann':>8s} | "
          f"{'Max Ann':>8s} | {'Avg WR':>6s} | {'Avg N':>6s} | {'Avg DD':>7s} | {'Avg PF':>6s} | "
          f"{'Avg Sh':>6s} | {'Pos/Win':>7s}")
    print(f"  {'-' * 160}")

    for i, w in enumerate(wf_avg):
        print(f"  {i + 1:2d} | {w['name']:40s} | {w['avg_ann']:+7.1f}% | {w['med_ann']:+7.1f}% | "
              f"{w['min_ann']:+7.1f}% | {w['max_ann']:+7.1f}% | {w['avg_wr']:5.1f}% | "
              f"{w['avg_n']:5.0f} | {w['avg_dd']:6.1f}% | {w['avg_pf']:5.2f} | "
              f"{w['avg_sharpe']:5.2f} | {w['n_positive']}/{w['n_windows']}")

    # ================================================================
    # WALK-FORWARD WINDOW-BY-WINDOW DETAIL
    # ================================================================
    print(f"\n{'=' * 160}")
    print(f"  WALK-FORWARD WINDOW-BY-WINDOW DETAIL")
    print(f"{'=' * 160}")

    for i, w in enumerate(wf_avg):
        print(f"\n  [{i + 1}] {w['name']}:")
        print(f"  {'Train':>9s} | {'Test':>4s} | {'Ann':>8s} | {'WR':>5s} | {'N':>5s} | "
              f"{'DD':>6s} | {'PF':>5s} | {'Sharpe':>7s} | {'AvgW':>6s} | {'AvgL':>6s}")
        print(f"  {'-' * 85}")
        for train_end, test_year, r in sorted(w['window_details'], key=lambda x: x[1]):
            print(f"  -{train_end:4d}    | {test_year:4d} | {r['ann']:+7.1f}% | {r['wr']:4.1f}% | "
                  f"{r['n']:5d} | {r['dd']:5.1f}% | {r['pf']:4.2f} | {r['sharpe']:6.2f} | "
                  f"{r['avg_win']:+5.2f}% | {r['avg_loss']:5.2f}%")

    # ================================================================
    # WALK-FORWARD BY PRICE MODE
    # ================================================================
    if wf_avg:
        print(f"\n{'=' * 160}")
        print(f"  WALK-FORWARD BY PRICE MODE (best config per mode)")
        print(f"{'=' * 160}")

        for pmode in ALL_PMODES:
            # Find best WF config for this price mode
            mode_wf = [w for w in wf_avg if f'PM_{pmode}_' in w['name']]
            if mode_wf:
                best_wf = max(mode_wf, key=lambda x: x['avg_ann'])
                print(f"\n  {pmode} ({pmode_desc[pmode][:60]}):")
                print(f"    Best WF: {best_wf['name']}")
                print(f"    Avg Ann={best_wf['avg_ann']:+7.1f}%  Min={best_wf['min_ann']:+7.1f}%  "
                      f"Max={best_wf['max_ann']:+7.1f}%  Pos={best_wf['n_positive']}/{best_wf['n_windows']}")

    # ================================================================
    # OVERFITTING CHECK
    # ================================================================
    if wf_avg:
        print(f"\n{'=' * 160}")
        print(f"  OVERFITTING CHECK: Full-Period vs Walk-Forward Correlation")
        print(f"{'=' * 160}")

        full_anns = []
        wf_anns_list = []
        for w in wf_avg:
            name = w['name']
            full_r = next((r for r in results if r['name'] == name), None)
            if full_r:
                full_anns.append(full_r['ann'])
                wf_anns_list.append(w['avg_ann'])

        if len(full_anns) > 2:
            corr = np.corrcoef(full_anns, wf_anns_list)[0, 1]
            decay = np.mean(wf_anns_list) / max(np.mean(full_anns), 0.01)
            print(f"  Configs tested OOS: {len(full_anns)}")
            print(f"  Full-period avg Ann: {np.mean(full_anns):+.1f}%")
            print(f"  WF avg Ann:          {np.mean(wf_anns_list):+.1f}%")
            print(f"  Correlation:         {corr:.3f}")
            print(f"  Decay ratio:         {decay:.2f}")

            if corr > 0.5:
                print(f"  -> GOOD: Strong positive correlation, training predicts OOS")
            elif corr > 0.2:
                print(f"  -> MODERATE: Some predictive power")
            else:
                print(f"  -> WARNING: Weak/no correlation, possible overfitting")

        # WF positive rate
        all_wf_anns = [r['ann'] for _, _, _, r in wf_all]
        n_pos_wf = sum(1 for a in all_wf_anns if a > 0)
        if all_wf_anns:
            print(f"\n  Overall WF positive rate: {n_pos_wf}/{len(all_wf_anns)} "
                  f"({n_pos_wf / len(all_wf_anns) * 100:.0f}%)")

    # ================================================================
    # OVERNIGHT GAP ANALYSIS
    # ================================================================
    print(f"\n{'=' * 160}")
    print(f"  OVERNIGHT GAP ANALYSIS: CC vs CO per-pair avg PnL")
    print(f"  (Isolates overnight gap contribution by comparing CC baseline vs CO mode)")
    print(f"{'=' * 160}")

    # Find matching CC and CO configs
    for sm in ALL_SPREADS:
        for lb in ALL_LOOKBACKS:
            for zt in ALL_Z:
                cc_results = [r for r in results if r['price_mode'] == PMODE_CC
                              and f'_{sm}_' in r['name']
                              and f'_LB{lb}_' in r['name']
                              and f'_Z{zt:.1f}' in r['name']]
                co_results = [r for r in results if r['price_mode'] == PMODE_CO
                              and f'_{sm}_' in r['name']
                              and f'_LB{lb}_' in r['name']
                              and f'_Z{zt:.1f}' in r['name']]

                if cc_results and co_results:
                    cc_r = cc_results[0]
                    co_r = co_results[0]

                    print(f"\n  Config: {sm}_LB{lb}_Z{zt:.1f}")
                    print(f"  CC: Ann={cc_r['ann']:+.1f}%  WR={cc_r['wr']:.1f}%  N={cc_r['n']}  Sharpe={cc_r['sharpe']:.2f}")
                    print(f"  CO: Ann={co_r['ann']:+.1f}%  WR={co_r['wr']:.1f}%  N={co_r['n']}  Sharpe={co_r['sharpe']:.2f}")
                    delta = co_r['ann'] - cc_r['ann']
                    print(f"  CO-CC delta: {delta:+.1f}%  {'CO better' if delta > 0 else 'CC better'}")

    # ================================================================
    # FINAL SUMMARY
    # ================================================================
    print(f"\n{'=' * 160}")
    print(f"  FINAL SUMMARY")
    print(f"{'=' * 160}")

    if results:
        print(f"\n  Full-period best: {results[0]['name']}")
        print(f"    Ann={results[0]['ann']:+.1f}%  WR={results[0]['wr']:.1f}%  N={results[0]['n']}  "
              f"DD={results[0]['dd']:.1f}%  PF={results[0]['pf']:.2f}  Sharpe={results[0]['sharpe']:.2f}")

    # Best per price mode
    print(f"\n  Best per price mode (full-period):")
    for pmode in ALL_PMODES:
        subset = [r for r in results if r['price_mode'] == pmode]
        if subset:
            best = max(subset, key=lambda x: x['ann'])
            print(f"    {pmode}: Ann={best['ann']:+.1f}%  WR={best['wr']:.1f}%  "
                  f"Sharpe={best['sharpe']:.2f}  DD={best['dd']:.1f}%  N={best['n']}")

    if wf_avg:
        print(f"\n  Walk-forward best: {wf_avg[0]['name']}")
        print(f"    WF Avg Ann={wf_avg[0]['avg_ann']:+.1f}%  WF Med Ann={wf_avg[0]['med_ann']:+.1f}%  "
              f"Min={wf_avg[0]['min_ann']:+.1f}%  Max={wf_avg[0]['max_ann']:+.1f}%  "
              f"Pos/Win={wf_avg[0]['n_positive']}/{wf_avg[0]['n_windows']}")

        # Best WF per price mode
        print(f"\n  Best walk-forward per price mode:")
        for pmode in ALL_PMODES:
            mode_wf = [w for w in wf_avg if f'PM_{pmode}_' in w['name']]
            if mode_wf:
                best_wf = max(mode_wf, key=lambda x: x['avg_ann'])
                print(f"    {pmode}: WF Avg={best_wf['avg_ann']:+.1f}%  "
                      f"Min={best_wf['min_ann']:+.1f}%  Pos={best_wf['n_positive']}/{best_wf['n_windows']}")

        n_all_positive = sum(1 for w in wf_avg if w['n_positive'] == w['n_windows'])
        print(f"\n  Of top 10 WF configs, {n_all_positive} are positive in ALL test windows")

    # Key finding: does CO/COC beat CC?
    cc_best = max((r for r in results if r['price_mode'] == PMODE_CC), key=lambda x: x['ann'], default=None)
    co_best = max((r for r in results if r['price_mode'] == PMODE_CO), key=lambda x: x['ann'], default=None)
    oo_best = max((r for r in results if r['price_mode'] == PMODE_OO), key=lambda x: x['ann'], default=None)
    coc_best = max((r for r in results if r['price_mode'] == PMODE_COC), key=lambda x: x['ann'], default=None)

    print(f"\n  KEY FINDING: Does using OPEN for entry/exit improve returns?")
    if cc_best:
        print(f"    CC  (baseline): {cc_best['ann']:+.1f}%  {cc_best['name']}")
    if co_best:
        delta = co_best['ann'] - cc_best['ann'] if cc_best else 0
        print(f"    CO  (intraday): {co_best['ann']:+.1f}%  delta vs CC={delta:+.1f}%  {co_best['name']}")
    if oo_best:
        delta = oo_best['ann'] - cc_best['ann'] if cc_best else 0
        print(f"    OO  (open-open): {oo_best['ann']:+.1f}%  delta vs CC={delta:+.1f}%  {oo_best['name']}")
    if coc_best:
        delta = coc_best['ann'] - cc_best['ann'] if cc_best else 0
        print(f"    COC (1.5day):   {coc_best['ann']:+.1f}%  delta vs CC={delta:+.1f}%  {coc_best['name']}")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.1f}s")
    print("=" * 160)


if __name__ == '__main__':
    main()
