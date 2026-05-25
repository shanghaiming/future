"""
Alpha Futures V42 — Combined Portfolio: Group Momentum Lag + Supply Chain Pair Trading
========================================================================================
Combines two orthogonal strategies into a single portfolio with shared capital:

  Strategy A (V34b): Group Momentum Lag — directional trend-following
    - When a commodity lags its supply-chain group, it catches up
    - Long-only, bets on price direction
    - Works on individual commodities within defined groups

  Strategy B (V39): Supply Chain Pair Trading — market-neutral mean-reversion
    - Spread between upstream/downstream deviates from rolling mean
    - Long one leg + short another, bets on spread convergence
    - Market-neutral, works on pairs

Capital allocation between the two strategies is tested at multiple fixed ratios.
Walk-forward validation on the best combined configs.
"""
import sys, os, time, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alpha_v2 import load_all_data, MIN_TRAIN, CASH0

MULT = {
    'agfi': 15, 'alfi': 5, 'aufi': 1000, 'bufi': 10, 'cufi': 5,
    'fufi': 10, 'rbfi': 10, 'znfi': 5, 'nifi': 1, 'hcfi': 10,
    'spfi': 10, 'ssfi': 5, 'sffi': 5, 'smfi': 5, 'pbfi': 5,
    'snfi': 1, 'rufi': 10, 'wrffi': 10,
    'afi': 10, 'bfi': 10, 'bbfi': 500, 'cffi': 5, 'cfi': 10,
    'csfi': 10, 'ebfi': 5, 'egfi': 10, 'fbfi': 500,
    'ifi': 100, 'jfi': 100, 'jmfi': 60, 'lfi': 5, 'mfi': 10,
    'pgfi': 20, 'ppfi': 5, 'vfi': 5, 'yfi': 10, 'pfi': 10,
    'jdfi': 5, 'lhfi': 16, 'pkfi': 5, 'rrfi': 20, 'lrfi': 20,
    'jrfi': 20, 'pmfi': 20, 'whfi': 20, 'rsfi': 20, 'cjfi': 10,
    'mafi': 10, 'apfi': 10, 'cyfi': 5, 'fgfi': 20, 'oifi': 10,
    'pfifi': 5, 'rmfi': 10, 'srfi': 10, 'tafi': 5, 'safi': 20,
    'urfi': 20, 'scfi': 1000, 'lufi': 10, 'bcfi': 5, 'nrfi': 1,
    'lgfi': 20, 'brfi': 5, 'lcfi': 1, 'sifi': 5,
    'ni': 1, 'tai': 5,
}
DEF_MULT = 10
COMM = 0.0003

GROUP_MAP = {
    'rbfi': 'ferrous', 'hcfi': 'ferrous', 'ifi': 'ferrous', 'jfi': 'ferrous', 'jmfi': 'ferrous',
    'cufi': 'nonferrous', 'alfi': 'nonferrous', 'znfi': 'nonferrous', 'nifi': 'nonferrous',
    'afi': 'oils', 'mfi': 'oils', 'yfi': 'oils', 'pfi': 'oils', 'cfi': 'oils',
    'scfi': 'energy', 'mafi': 'energy', 'bfi': 'energy', 'fufi': 'energy',
    'ppfi': 'chemical', 'vfi': 'chemical', 'egfi': 'chemical', 'pgfi': 'chemical',
}

UPSTREAM = {
    'rbfi': 'ifi', 'hcfi': 'rbfi', 'jfi': 'jmfi',
    'mafi': 'scfi', 'bfi': 'scfi', 'fufi': 'scfi',
    'mfi': 'afi', 'yfi': 'afi', 'pfi': 'yfi',
    'ppfi': 'mafi', 'vfi': 'mafi', 'egfi': 'mafi',
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
    print("=" * 130)
    print("Alpha Futures V42 — Combined Portfolio: Group Momentum Lag + Supply Chain Pair Trading")
    print("Directional trend-following (V34b) + Market-neutral mean-reversion (V39)")
    print("=" * 130)

    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)
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

    upstream_si = {}
    for si in range(NS):
        up_sym = UPSTREAM.get(syms[si])
        if up_sym and up_sym in sym_to_si:
            upstream_si[si] = sym_to_si[up_sym]
        else:
            upstream_si[si] = -1

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

    # --- Directional signals (V34b) ---
    # Momentum at multiple lookbacks
    mom = {}
    for lag in [3, 5, 7, 10]:
        m = np.full((NS, ND), np.nan)
        for si in range(NS):
            for di in range(lag, ND):
                c_now = C[si, di]
                c_prev = C[si, di - lag]
                if not np.isnan(c_now) and not np.isnan(c_prev) and c_prev > 0:
                    m[si, di] = (c_now - c_prev) / c_prev
        mom[lag] = m

    # Group momentum (excluding self)
    grp_mom = {}
    for lag in [3, 5, 7, 10]:
        gm = np.full((NS, ND), np.nan)
        for grp, members in group_members.items():
            for di in range(lag, ND):
                for sj in members:
                    ms = []
                    for sk in members:
                        if sk == sj:
                            continue
                        m = mom[lag][sk, di]
                        if not np.isnan(m):
                            ms.append(m)
                    if ms:
                        gm[sj, di] = np.mean(ms)
        grp_mom[lag] = gm

    # ATR for trailing stops
    atr10 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(11, ND):
            trs = []
            for dd in range(di - 10, di):
                hi, lo, pc = H[si, dd], L[si, dd], C[si, dd - 1]
                if np.isnan(hi) or np.isnan(lo):
                    continue
                tr = hi - lo
                if not np.isnan(pc):
                    tr = max(tr, abs(hi - pc), abs(lo - pc))
                trs.append(tr)
            if trs:
                atr10[si, di] = np.mean(trs)

    # --- Pair signals (V39) ---
    spreads = {}
    for down_si, up_si, down_sym, up_sym in pair_indices:
        spread = np.full(ND, np.nan)
        for di in range(ND):
            pd = C[down_si, di]
            pu = C[up_si, di]
            if not np.isnan(pd) and not np.isnan(pu):
                spread[di] = pd - pu
        spreads[(down_si, up_si)] = spread

    print(f"  All signals computed ({time.time()-t0:.1f}s)", flush=True)

    # ========================================
    # SCORING FUNCTIONS
    # ========================================
    def make_glag_score(mom_lag=5, min_lag=0.003):
        """Group momentum lag scorer: positive when commodity lags its group."""
        def score(si, di):
            own = mom[mom_lag][si, di]
            grp = grp_mom[mom_lag][si, di]
            if np.isnan(own) or np.isnan(grp):
                return np.nan
            divergence = grp - own
            if divergence < min_lag:
                return np.nan
            return divergence
        return score

    # ========================================
    # PAIR DATA COMPUTATION (per-config)
    # ========================================
    def compute_pair_data(lookback):
        """Compute rolling z-scores for all pairs at given lookback."""
        pd = {}
        for down_si, up_si, down_sym, up_sym in pair_indices:
            sp = spreads[(down_si, up_si)]
            sp_mean = np.full(ND, np.nan)
            sp_std = np.full(ND, np.nan)
            z = np.full(ND, np.nan)
            for di in range(lookback, ND):
                window = sp[di - lookback:di]
                valid = window[~np.isnan(window)]
                if len(valid) >= lookback * 0.8:
                    sp_mean[di] = np.mean(valid)
                    sp_std[di] = np.std(valid, ddof=1)
                    if sp_std[di] > 1e-10:
                        z[di] = (sp[di] - sp_mean[di]) / sp_std[di]
            pd[(down_si, up_si)] = {
                'z': z, 'down_sym': down_sym, 'up_sym': up_sym,
            }
        return pd

    # ========================================
    # COMBINED BACKTEST ENGINE
    # ========================================
    def run_combined_backtest(
        # Strategy A: Directional (Group Momentum Lag)
        glag_fn, top_n=1, hold_min=2, hold_max=3, trail_atr_mult=3.0,
        # Strategy B: Pair Trading
        pair_data=None, z_thresh=1.5, pair_hold_max=3, max_pairs=2,
        # Capital allocation
        alloc_dir=0.5,
        # Walk-forward
        wf_split_year=None,
        config_name="",
    ):
        """
        Combined backtest: runs both strategies with shared capital pool.
        alloc_dir: fraction of capital allocated to directional strategy.
        """
        cash = float(CASH0)
        dir_positions = []   # directional (V34b) positions
        pair_positions = []  # pair (V39) positions
        dir_trades = []
        pair_trades = []

        for di in range(MIN_TRAIN, ND):
            year = dates[di].year
            if wf_split_year is not None and year < wf_split_year:
                continue

            # === COMPUTE CURRENT CAPITAL ALLOCATION ===
            # Market value of all open positions
            dir_mkt_val = 0.0
            for pos in dir_positions:
                c = C[pos['si'], di]
                if np.isnan(c) or c <= 0:
                    c = pos['entry']
                mult = MULT.get(pos['sym'], DEF_MULT)
                dir_mkt_val += c * mult * pos['lots']

            pair_mkt_val = 0.0
            for pos in pair_positions:
                c_down = C[pos['down_si'], di]
                c_up = C[pos['up_si'], di]
                if np.isnan(c_down) or c_down <= 0:
                    c_down = pos['entry_down']
                if np.isnan(c_up) or c_up <= 0:
                    c_up = pos['entry_up']
                mult_down = MULT.get(pos['down_sym'], DEF_MULT)
                mult_up = MULT.get(pos['up_sym'], DEF_MULT)
                pair_mkt_val += c_down * mult_down * pos['lots_down']
                pair_mkt_val += c_up * mult_up * pos['lots_up']

            total_equity = cash + dir_mkt_val + pair_mkt_val
            dir_budget = total_equity * alloc_dir
            pair_budget = total_equity * (1 - alloc_dir)

            # =============================================
            # MANAGE DIRECTIONAL POSITIONS (V34b)
            # =============================================
            new_dir_pos = []
            for pos in dir_positions:
                c = C[pos['si'], di]
                if np.isnan(c) or c <= 0:
                    c = pos['entry']
                mult = MULT.get(pos['sym'], DEF_MULT)
                pnl = (c - pos['entry']) * mult * pos['lots'] * pos['dir']
                pnl_pct = pnl / (pos['entry'] * mult * pos['lots']) * 100 if pos['entry'] > 0 else 0
                days_held = di - pos['entry_di']

                exit_reason = None

                # Trailing stop
                if trail_atr_mult > 0 and days_held >= 2:
                    atr = pos.get('atr', 0)
                    if atr > 0 and pos['dir'] == 1:
                        new_trail = c - trail_atr_mult * atr
                        if new_trail > pos.get('trail_price', pos['entry']):
                            pos['trail_price'] = new_trail
                        if c < pos['trail_price']:
                            exit_reason = 'trail'

                # Signal flip (after min hold)
                if exit_reason is None and days_held >= hold_min:
                    cur_score = glag_fn(pos['si'], di)
                    if not np.isnan(cur_score) and cur_score < -0.01:
                        exit_reason = 'signal_flip'

                # Time exit
                if exit_reason is None and days_held >= hold_max:
                    exit_reason = 'time'

                if exit_reason:
                    mkt_val = c * mult * pos['lots']
                    cost_out = mkt_val * COMM
                    cash += mkt_val - cost_out
                    dir_trades.append({
                        'pnl_pct': pnl_pct, 'pnl_abs': pnl,
                        'days': days_held, 'di': di, 'year': year,
                        'sym': pos['sym'], 'dir': pos['dir'],
                        'reason': exit_reason, 'strategy': 'directional',
                    })
                else:
                    new_dir_pos.append(pos)

            dir_positions = new_dir_pos

            # =============================================
            # MANAGE PAIR POSITIONS (V39)
            # =============================================
            new_pair_pos = []
            for pos in pair_positions:
                p_down_si = pos['down_si']
                p_up_si = pos['up_si']
                z_now = pair_data[(p_down_si, p_up_si)]['z'][di]
                days_held = di - pos['entry_di']
                entry_z = pos['entry_z']
                pos_dir = pos['dir']

                exit_reason = None

                # Exit 1: z crosses 0 (mean reversion complete)
                if not np.isnan(z_now):
                    if pos_dir == 1 and z_now <= 0:
                        exit_reason = 'mean_rev'
                    elif pos_dir == -1 and z_now >= 0:
                        exit_reason = 'mean_rev'

                # Exit 2: Stop loss (z worsens by 1.0 from entry)
                if exit_reason is None and not np.isnan(z_now):
                    if pos_dir == 1 and z_now < entry_z - 1.0:
                        exit_reason = 'stop_loss'
                    elif pos_dir == -1 and z_now > entry_z + 1.0:
                        exit_reason = 'stop_loss'

                # Exit 3: Time exit
                if exit_reason is None and days_held >= pair_hold_max:
                    exit_reason = 'time'

                if exit_reason:
                    c_down = C[p_down_si, di]
                    c_up = C[p_up_si, di]
                    if np.isnan(c_down) or c_down <= 0:
                        c_down = pos['entry_down']
                    if np.isnan(c_up) or c_up <= 0:
                        c_up = pos['entry_up']

                    mult_down = MULT.get(pos['down_sym'], DEF_MULT)
                    mult_up = MULT.get(pos['up_sym'], DEF_MULT)
                    lots_down = pos['lots_down']
                    lots_up = pos['lots_up']

                    if pos_dir == 1:
                        pnl_down = (c_down - pos['entry_down']) * mult_down * lots_down
                        pnl_up = (pos['entry_up'] - c_up) * mult_up * lots_up
                    else:
                        pnl_down = (pos['entry_down'] - c_down) * mult_down * lots_down
                        pnl_up = (c_up - pos['entry_up']) * mult_up * lots_up

                    entry_val_down = pos['entry_down'] * mult_down * lots_down
                    entry_val_up = pos['entry_up'] * mult_up * lots_up
                    exit_val_down = c_down * mult_down * lots_down
                    exit_val_up = c_up * mult_up * lots_up
                    cost = (entry_val_down + entry_val_up) * COMM + \
                           (exit_val_down + exit_val_up) * COMM

                    total_pnl = pnl_down + pnl_up - cost
                    invested = entry_val_down + entry_val_up
                    pnl_pct = total_pnl / invested * 100 if invested > 0 else 0

                    if pos_dir == 1:
                        cash_return = c_down * mult_down * lots_down - c_up * mult_up * lots_up
                    else:
                        cash_return = -c_down * mult_down * lots_down + c_up * mult_up * lots_up

                    cash += pos['cash_invested'] + cash_return - (exit_val_down + exit_val_up) * COMM

                    pair_trades.append({
                        'pnl_abs': total_pnl, 'pnl_pct': pnl_pct,
                        'days': days_held, 'di': di, 'year': year,
                        'pair': (pos['down_sym'], pos['up_sym']),
                        'pair_label': PAIR_LABEL.get((pos['down_sym'], pos['up_sym']), ''),
                        'dir': pos_dir, 'reason': exit_reason, 'strategy': 'pair',
                    })
                else:
                    new_pair_pos.append(pos)

            pair_positions = new_pair_pos

            # =============================================
            # OPEN NEW DIRECTIONAL POSITIONS
            # =============================================
            n_dir_open = len(dir_positions)
            if n_dir_open < top_n:
                slots = top_n - n_dir_open
                scored = []
                for si in range(NS):
                    sc = glag_fn(si, di)
                    if np.isnan(sc) or sc <= 0:
                        continue
                    sym = syms[si]
                    if any(p['sym'] == sym for p in dir_positions):
                        continue
                    # Check not already used in a pair position
                    if any(p['down_si'] == si or p['up_si'] == si for p in pair_positions):
                        continue
                    scored.append((si, sc, sym))

                if scored:
                    scored.sort(key=lambda x: -x[1])
                    # Available cash for directional = budget - current dir mkt val
                    dir_cash_avail = max(dir_budget - dir_mkt_val, cash * alloc_dir * 0.5)
                    # But also cannot exceed actual cash
                    dir_cash_avail = min(dir_cash_avail, cash)
                    if dir_cash_avail < 1000:
                        dir_cash_avail = cash * 0.3 if cash > 3000 else 0

                    cash_per_slot = dir_cash_avail / slots if slots > 0 else dir_cash_avail

                    for best_si, best_sc, best_sym in scored[:slots]:
                        c = C[best_si, di]
                        if np.isnan(c) or c <= 0:
                            continue
                        mult = MULT.get(best_sym, DEF_MULT)
                        notional = c * mult
                        if notional <= 0:
                            continue

                        lots = int(cash_per_slot / (notional * (1 + COMM)))
                        if lots <= 0:
                            continue
                        cost_in = notional * lots * (1 + COMM)
                        if cost_in > cash:
                            lots = int(cash / (notional * (1 + COMM)))
                            if lots <= 0:
                                continue
                            cost_in = notional * lots * (1 + COMM)

                        atr_val = atr10[best_si, di] if not np.isnan(atr10[best_si, di]) else 0
                        cash -= cost_in
                        trail_price = c - trail_atr_mult * atr_val
                        dir_positions.append({
                            'si': best_si, 'entry': c, 'entry_di': di,
                            'lots': lots, 'dir': 1, 'sym': best_sym,
                            'atr': atr_val, 'trail_price': trail_price,
                        })
                        dir_mkt_val += cost_in

            # =============================================
            # OPEN NEW PAIR POSITIONS
            # =============================================
            occupied = set()
            for pos in pair_positions:
                occupied.add(pos['down_si'])
                occupied.add(pos['up_si'])
            for pos in dir_positions:
                occupied.add(pos['si'])

            n_can_open = max_pairs - len(pair_positions)
            if n_can_open > 0:
                candidates = []
                for down_si, up_si, down_sym, up_sym in pair_indices:
                    if down_si in occupied or up_si in occupied:
                        continue
                    z_val = pair_data[(down_si, up_si)]['z'][di]
                    if np.isnan(z_val):
                        continue
                    if abs(z_val) < z_thresh:
                        continue
                    candidates.append((abs(z_val), down_si, up_si, down_sym, up_sym, z_val))

                if candidates:
                    candidates.sort(key=lambda x: -x[0])

                    # Available cash for pairs = budget - current pair mkt val
                    pair_cash_avail = max(pair_budget - pair_mkt_val, cash * (1 - alloc_dir) * 0.5)
                    pair_cash_avail = min(pair_cash_avail, cash)

                    for _, down_si, up_si, down_sym, up_sym, z_val in candidates[:n_can_open]:
                        c_down = C[down_si, di]
                        c_up = C[up_si, di]
                        if np.isnan(c_down) or c_down <= 0 or np.isnan(c_up) or c_up <= 0:
                            continue

                        mult_down = MULT.get(down_sym, DEF_MULT)
                        mult_up = MULT.get(up_sym, DEF_MULT)

                        cash_per_leg = pair_cash_avail / 2
                        lots_down = int(cash_per_leg / (c_down * mult_down * (1 + COMM)))
                        lots_up = int(cash_per_leg / (c_up * mult_up * (1 + COMM)))
                        if lots_down <= 0 or lots_up <= 0:
                            continue

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
                                continue

                        pos_dir = -1 if z_val > 0 else 1

                        cash -= total_cost
                        pair_positions.append({
                            'down_si': down_si, 'up_si': up_si,
                            'down_sym': down_sym, 'up_sym': up_sym,
                            'entry_down': c_down, 'entry_up': c_up,
                            'lots_down': lots_down, 'lots_up': lots_up,
                            'entry_di': di, 'entry_z': z_val,
                            'dir': pos_dir, 'cash_invested': total_cost,
                        })
                        pair_mkt_val += total_cost

        # === CLOSE REMAINING POSITIONS AT END ===
        for pos in dir_positions:
            c = C[pos['si'], ND - 1]
            if np.isnan(c) or c <= 0:
                c = pos['entry']
            mult = MULT.get(pos['sym'], DEF_MULT)
            pnl = (c - pos['entry']) * mult * pos['lots'] * pos['dir']
            mkt_val = c * mult * pos['lots']
            cash += mkt_val * (1 - COMM)
            dir_trades.append({
                'pnl_pct': pnl / (pos['entry'] * mult * pos['lots']) * 100,
                'pnl_abs': pnl,
                'days': ND - 1 - pos['entry_di'], 'di': ND - 1,
                'year': dates[ND - 1].year,
                'sym': pos['sym'], 'dir': pos['dir'],
                'reason': 'end', 'strategy': 'directional',
            })

        for pos in pair_positions:
            p_down_si = pos['down_si']
            p_up_si = pos['up_si']
            c_down = C[p_down_si, ND - 1]
            c_up = C[p_up_si, ND - 1]
            if np.isnan(c_down) or c_down <= 0:
                c_down = pos['entry_down']
            if np.isnan(c_up) or c_up <= 0:
                c_up = pos['entry_up']

            mult_down = MULT.get(pos['down_sym'], DEF_MULT)
            mult_up = MULT.get(pos['up_sym'], DEF_MULT)
            lots_down = pos['lots_down']
            lots_up = pos['lots_up']

            if pos['dir'] == 1:
                pnl_down = (c_down - pos['entry_down']) * mult_down * lots_down
                pnl_up = (pos['entry_up'] - c_up) * mult_up * lots_up
            else:
                pnl_down = (pos['entry_down'] - c_down) * mult_down * lots_down
                pnl_up = (c_up - pos['entry_up']) * mult_up * lots_up

            entry_val_down = pos['entry_down'] * mult_down * lots_down
            entry_val_up = pos['entry_up'] * mult_up * lots_up
            exit_val_down = c_down * mult_down * lots_down
            exit_val_up = c_up * mult_up * lots_up
            cost = (entry_val_down + entry_val_up) * COMM + (exit_val_down + exit_val_up) * COMM

            total_pnl = pnl_down + pnl_up - cost
            invested = entry_val_down + entry_val_up
            pnl_pct = total_pnl / invested * 100 if invested > 0 else 0

            if pos['dir'] == 1:
                cash_return = c_down * mult_down * lots_down - c_up * mult_up * lots_up
            else:
                cash_return = -c_down * mult_down * lots_down + c_up * mult_up * lots_up

            cash += pos['cash_invested'] + cash_return - (exit_val_down + exit_val_up) * COMM

            pair_trades.append({
                'pnl_abs': total_pnl, 'pnl_pct': pnl_pct,
                'days': ND - 1 - pos['entry_di'], 'di': ND - 1,
                'year': dates[ND - 1].year,
                'pair': (pos['down_sym'], pos['up_sym']),
                'pair_label': PAIR_LABEL.get((pos['down_sym'], pos['up_sym']), ''),
                'dir': pos['dir'], 'reason': 'end', 'strategy': 'pair',
            })

        # === COMPUTE STATS ===
        all_trades = dir_trades + pair_trades
        if len(all_trades) < 5:
            return None

        equity = float(CASH0)
        peak = float(CASH0)
        max_dd = 0.0
        # Track equity curve separately for each strategy
        equity_dir = 0.0
        equity_pair = 0.0
        for t in sorted(all_trades, key=lambda x: x['di']):
            equity += t['pnl_abs']
            if t['strategy'] == 'directional':
                equity_dir += t['pnl_abs']
            else:
                equity_pair += t['pnl_abs']
            if equity > peak:
                peak = equity
            if peak > 0:
                dd = (peak - equity) / peak * 100
                if dd > max_dd:
                    max_dd = dd

        days_total = (dates[ND - 1] - dates[MIN_TRAIN]).days
        yr = max(days_total / 365.25, 0.01)
        if wf_split_year:
            first_test_di = None
            for d in range(MIN_TRAIN, ND):
                if dates[d].year >= wf_split_year:
                    first_test_di = d
                    break
            if first_test_di:
                days_total = (dates[ND - 1] - dates[first_test_di]).days
                yr = max(days_total / 365.25, 0.01)

        ann = ((cash / CASH0) ** (1 / yr) - 1) * 100
        nw = sum(1 for t in all_trades if t['pnl_abs'] > 0)
        wr = nw / len(all_trades) * 100

        # Per-strategy stats
        n_dir = len(dir_trades)
        n_pair = len(pair_trades)
        nw_dir = sum(1 for t in dir_trades if t['pnl_abs'] > 0)
        nw_pair = sum(1 for t in pair_trades if t['pnl_abs'] > 0)
        wr_dir = nw_dir / n_dir * 100 if n_dir > 0 else 0
        wr_pair = nw_pair / n_pair * 100 if n_pair > 0 else 0
        pnl_dir_total = sum(t['pnl_abs'] for t in dir_trades)
        pnl_pair_total = sum(t['pnl_abs'] for t in pair_trades)
        avg_win_dir = np.mean([t['pnl_pct'] for t in dir_trades if t['pnl_abs'] > 0]) if nw_dir > 0 else 0
        avg_loss_dir = np.mean([abs(t['pnl_pct']) for t in dir_trades if t['pnl_abs'] <= 0]) if nw_dir < n_dir else 0
        avg_win_pair = np.mean([t['pnl_pct'] for t in pair_trades if t['pnl_abs'] > 0]) if nw_pair > 0 else 0
        avg_loss_pair = np.mean([abs(t['pnl_pct']) for t in pair_trades if t['pnl_abs'] <= 0]) if nw_pair < n_pair else 0
        pf_dir = (sum(t['pnl_abs'] for t in dir_trades if t['pnl_abs'] > 0) /
                  max(abs(sum(t['pnl_abs'] for t in dir_trades if t['pnl_abs'] < 0)), 1))
        pf_pair = (sum(t['pnl_abs'] for t in pair_trades if t['pnl_abs'] > 0) /
                   max(abs(sum(t['pnl_abs'] for t in pair_trades if t['pnl_abs'] < 0)), 1))
        pf_all = (sum(t['pnl_abs'] for t in all_trades if t['pnl_abs'] > 0) /
                  max(abs(sum(t['pnl_abs'] for t in all_trades if t['pnl_abs'] < 0)), 1))

        # Yearly breakdown (combined)
        year_stats = {}
        for t in all_trades:
            y = t['year']
            if y not in year_stats:
                year_stats[y] = {'n': 0, 'w': 0, 'pnl': 0.0, 'dir_pnl': 0.0, 'pair_pnl': 0.0}
            year_stats[y]['n'] += 1
            if t['pnl_abs'] > 0:
                year_stats[y]['w'] += 1
            year_stats[y]['pnl'] += t['pnl_pct']
            if t['strategy'] == 'directional':
                year_stats[y]['dir_pnl'] += t['pnl_abs']
            else:
                year_stats[y]['pair_pnl'] += t['pnl_abs']

        # Exit reason breakdown
        reasons = {}
        for t in all_trades:
            r = t['reason']
            if r not in reasons:
                reasons[r] = {'n': 0, 'w': 0, 'pnl': 0.0}
            reasons[r]['n'] += 1
            if t['pnl_abs'] > 0:
                reasons[r]['w'] += 1
            reasons[r]['pnl'] += t['pnl_pct']

        # Per-pair breakdown
        pair_stats = {}
        for t in pair_trades:
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
            'n': len(all_trades),
            'n_dir': n_dir, 'n_pair': n_pair,
            'wr': round(wr, 1),
            'wr_dir': round(wr_dir, 1), 'wr_pair': round(wr_pair, 1),
            'dd': round(max_dd, 1),
            'pf': round(pf_all, 2),
            'pf_dir': round(pf_dir, 2), 'pf_pair': round(pf_pair, 2),
            'avg_win_dir': round(avg_win_dir, 2), 'avg_loss_dir': round(avg_loss_dir, 2),
            'avg_win_pair': round(avg_win_pair, 2), 'avg_loss_pair': round(avg_loss_pair, 2),
            'pnl_dir': round(pnl_dir_total, 0), 'pnl_pair': round(pnl_pair_total, 0),
            'cash': round(cash, 0),
            'alloc_dir': alloc_dir,
            'reasons': reasons, 'yearly': year_stats,
            'pair_stats': pair_stats,
            'dir_trades': dir_trades, 'pair_trades': pair_trades,
        }

    # ========================================
    # BASELINE: EACH STRATEGY ALONE
    # ========================================
    print("\n[Baseline] Running directional-only and pair-only backtests...", flush=True)
    baseline_results = []

    # --- Directional-only (alloc_dir = 1.0) ---
    for lag in [5, 7]:
        for tn in [1]:
            for hm in [3, 5]:
                for trail in [2.5, 3.0]:
                    fn = make_glag_score(mom_lag=lag)
                    name = f"DIR_ONLY_LAG{lag}_N{tn}_H{hm}_TR{trail:.0f}"
                    # Use pair_data with lookback 10 (not really used since alloc=1.0)
                    pd_dummy = compute_pair_data(10)
                    r = run_combined_backtest(
                        glag_fn=fn, top_n=tn, hold_max=hm, trail_atr_mult=trail,
                        pair_data=pd_dummy, z_thresh=1.5, pair_hold_max=3, max_pairs=0,
                        alloc_dir=1.0, config_name=name,
                    )
                    if r and r['ann'] > 0:
                        baseline_results.append(r)
                        print(f"  {name:50s} | Ann {r['ann']:+7.1f}% | WR {r['wr']:5.1f}% | "
                              f"N={r['n']:3d} | DD {r['dd']:6.1f}%")

    # --- Pair-only (alloc_dir = 0.0) ---
    for lb in [10, 20]:
        for zt in [1.5, 2.0]:
            for hd in [3, 5]:
                for mp in [1, 2]:
                    pd = compute_pair_data(lb)
                    fn_dummy = make_glag_score(mom_lag=5)
                    name = f"PAIR_ONLY_LB{lb}_Z{zt:.1f}_H{hd}_MP{mp}"
                    r = run_combined_backtest(
                        glag_fn=fn_dummy, top_n=0, hold_max=3, trail_atr_mult=3.0,
                        pair_data=pd, z_thresh=zt, pair_hold_max=hd, max_pairs=mp,
                        alloc_dir=0.0, config_name=name,
                    )
                    if r and r['ann'] > 0:
                        baseline_results.append(r)
                        print(f"  {name:50s} | Ann {r['ann']:+7.1f}% | WR {r['wr']:5.1f}% | "
                              f"N={r['n']:3d} | DD {r['dd']:6.1f}%")

    # ========================================
    # COMBINED PARAMETER SWEEP
    # ========================================
    print("\n[Combined] Running combined portfolio sweep...", flush=True)
    combined_results = []

    # Precompute pair data for the lookbacks we'll test
    pair_data_cache = {}
    for lb in [10, 20]:
        pair_data_cache[lb] = compute_pair_data(lb)
        print(f"  Pair data LB={lb} computed", flush=True)

    configs = []

    # Capital allocation ratios (directional / pairs)
    for alloc_dir in [0.3, 0.4, 0.5, 0.6, 0.7]:
        # Directional params
        for lag in [5, 7]:
            for hold_max_d in [3, 5]:
                for trail in [3.0]:
                    # Pair params
                    for lb in [10, 20]:
                        for zt in [1.5, 2.0]:
                            for hold_max_p in [3, 5]:
                                for mp in [1, 2]:
                                    name = (f"A{alloc_dir:.1f}_LAG{lag}_H{hold_max_d}_"
                                            f"LB{lb}_Z{zt:.1f}_HP{hold_max_p}_MP{mp}")
                                    configs.append({
                                        'alloc_dir': alloc_dir,
                                        'lag': lag, 'hold_max_d': hold_max_d, 'trail': trail,
                                        'lb': lb, 'zt': zt, 'hold_max_p': hold_max_p, 'mp': mp,
                                        'name': name, 'wf': None,
                                    })

    # Walk-forward configs for promising parameter ranges
    for alloc_dir in [0.3, 0.5, 0.7]:
        for lag in [5, 7]:
            for hold_max_d in [3, 5]:
                for lb in [10, 20]:
                    for zt in [1.5, 2.0]:
                        for hold_max_p in [3]:
                            for mp in [2]:
                                for wf_year in [2023, 2024]:
                                    name = (f"A{alloc_dir:.1f}_LAG{lag}_H{hold_max_d}_"
                                            f"LB{lb}_Z{zt:.1f}_HP{hold_max_p}_MP{mp}_WF{wf_year}")
                                    configs.append({
                                        'alloc_dir': alloc_dir,
                                        'lag': lag, 'hold_max_d': hold_max_d, 'trail': 3.0,
                                        'lb': lb, 'zt': zt, 'hold_max_p': hold_max_p, 'mp': mp,
                                        'name': name, 'wf': wf_year,
                                    })

    print(f"  {len(configs)} combined configurations", flush=True)

    for ci, cfg in enumerate(configs):
        fn = make_glag_score(mom_lag=cfg['lag'])
        pd = pair_data_cache[cfg['lb']]
        r = run_combined_backtest(
            glag_fn=fn, top_n=1, hold_min=2, hold_max=cfg['hold_max_d'],
            trail_atr_mult=cfg['trail'],
            pair_data=pd, z_thresh=cfg['zt'], pair_hold_max=cfg['hold_max_p'],
            max_pairs=cfg['mp'],
            alloc_dir=cfg['alloc_dir'],
            wf_split_year=cfg['wf'],
            config_name=cfg['name'],
        )
        if r and r['ann'] > 0:
            combined_results.append(r)
            if r['ann'] > 50:
                print(f"  {r['name']:60s} | Ann {r['ann']:+7.1f}% | WR {r['wr']:5.1f}% | "
                      f"N={r['n']:3d}(D{r['n_dir']:2d}/P{r['n_pair']:2d}) | "
                      f"DD {r['dd']:6.1f}% | PnL D={r['pnl_dir']:+.0f} P={r['pnl_pair']:+.0f}")

        if (ci + 1) % 50 == 0:
            print(f"  [{ci+1}/{len(configs)}] {len(combined_results)} profitable", flush=True)

    # ========================================
    # RESULTS
    # ========================================
    all_results = baseline_results + combined_results
    all_results.sort(key=lambda x: -x['ann'])

    # Separate categories
    dir_only = [r for r in all_results if r['alloc_dir'] == 1.0]
    pair_only = [r for r in all_results if r['alloc_dir'] == 0.0]
    combined = [r for r in combined_results if r['ann'] > 0]
    combined.sort(key=lambda x: -x['ann'])
    wf_combined = [r for r in combined if '_WF' in r['name']]
    full_combined = [r for r in combined if '_WF' not in r['name']]

    # --- Baseline comparison table ---
    print(f"\n{'=' * 130}")
    print(f"  BASELINE COMPARISON: Directional-Only vs Pair-Only")
    print(f"{'=' * 130}")
    print(f"  {'Config':50s} | {'Ann':>7s} | {'WR':>5s} | {'N':>4s} | {'DD':>6s} | {'PF':>5s} | Type")
    print(f"  {'-' * 100}")
    for r in dir_only[:5]:
        print(f"  {r['name']:50s} | {r['ann']:+7.1f}% | {r['wr']:5.1f}% | "
              f"{r['n']:4d} | {r['dd']:6.1f}% | {r['pf']:5.2f} | DIR")
    for r in pair_only[:5]:
        print(f"  {r['name']:50s} | {r['ann']:+7.1f}% | {r['wr']:5.1f}% | "
              f"{r['n']:4d} | {r['dd']:6.1f}% | {r['pf']:5.2f} | PAIR")

    # --- Top 10 combined ---
    print(f"\n{'=' * 130}")
    print(f"  TOP 10 COMBINED STRATEGIES (by annual return)")
    print(f"{'=' * 130}")
    print(f"  {'Config':60s} | {'Ann':>7s} | {'WR':>5s} | {'N':>5s} | {'DD':>6s} | "
          f"{'PF':>5s} | {'PnL_Dir':>10s} | {'PnL_Pair':>10s}")
    print(f"  {'-' * 130}")
    for r in full_combined[:10]:
        print(f"  {r['name']:60s} | {r['ann']:+7.1f}% | {r['wr']:5.1f}% | "
              f"{r['n']:3d}(D{r['n_dir']:2d}/P{r['n_pair']:2d}) | "
              f"{r['dd']:6.1f}% | {r['pf']:5.2f} | {r['pnl_dir']:+10.0f} | {r['pnl_pair']:+10.0f}")

    # --- Allocation comparison (aggregate) ---
    print(f"\n  ALLOCATION RATIO COMPARISON (best config per allocation):")
    best_by_alloc = {}
    for r in full_combined:
        ad = r['alloc_dir']
        if ad not in best_by_alloc or r['ann'] > best_by_alloc[ad]['ann']:
            best_by_alloc[ad] = r
    for ad in sorted(best_by_alloc.keys()):
        r = best_by_alloc[ad]
        pct_d = r['pnl_dir'] / max(r['pnl_dir'] + r['pnl_pair'], 1) * 100
        print(f"    Dir={ad:.0%}/Pair={1-ad:.0%}: Ann={r['ann']:+.1f}%  WR={r['wr']:.1f}%  "
              f"DD={r['dd']:.1f}%  N={r['n']}(D{r['n_dir']}/P{r['n_pair']})  "
              f"PF={r['pf']:.2f}  PnL% D={pct_d:.0f}/P={100-pct_d:.0f}")

    # --- Best combined: full detail ---
    if full_combined:
        best = full_combined[0]
        print(f"\n{'=' * 130}")
        print(f"  BEST COMBINED: {best['name']}")
        print(f"  Ann={best['ann']:+.1f}%  WR={best['wr']:.1f}%  N={best['n']}  "
              f"DD={best['dd']:.1f}%  PF={best['pf']:.2f}  Final={best['cash']:.0f}")
        print(f"  Allocation: Dir={best['alloc_dir']:.0%}/Pair={1-best['alloc_dir']:.0%}")
        print(f"{'=' * 130}")

        print(f"\n  STRATEGY CONTRIBUTION:")
        total_pnl = best['pnl_dir'] + best['pnl_pair']
        if total_pnl != 0:
            print(f"    Directional: {best['n_dir']:3d} trades  WR={best['wr_dir']:.1f}%  "
                  f"PnL={best['pnl_dir']:+.0f} ({best['pnl_dir']/total_pnl*100:.0f}%)  "
                  f"PF={best['pf_dir']:.2f}  AvgW={best['avg_win_dir']:+.2f}%  AvgL={best['avg_loss_dir']:.2f}%")
            print(f"    Pairs:       {best['n_pair']:3d} trades  WR={best['wr_pair']:.1f}%  "
                  f"PnL={best['pnl_pair']:+.0f} ({best['pnl_pair']/total_pnl*100:.0f}%)  "
                  f"PF={best['pf_pair']:.2f}  AvgW={best['avg_win_pair']:+.2f}%  AvgL={best['avg_loss_pair']:.2f}%")

        print(f"\n  YEARLY BREAKDOWN (Directional vs Pair contribution):")
        for y in sorted(best['yearly'].keys()):
            s = best['yearly'][y]
            wr_y = s['w'] / max(s['n'], 1) * 100
            total_y = s['dir_pnl'] + s['pair_pnl']
            dir_pct = s['dir_pnl'] / total_y * 100 if total_y != 0 else 0
            print(f"    {y}: {s['n']:3d}t  WR={wr_y:5.1f}%  TotalPnL={total_y:+12.0f}  "
                  f"Dir={s['dir_pnl']:+10.0f}({dir_pct:4.0f}%)  Pair={s['pair_pnl']:+10.0f}({100-dir_pct:4.0f}%)")

        if best.get('pair_stats'):
            print(f"\n  PER-PAIR BREAKDOWN:")
            for p in sorted(best['pair_stats'].keys(), key=lambda x: -best['pair_stats'][x]['n']):
                ps = best['pair_stats'][p]
                wr_p = ps['w'] / max(ps['n'], 1) * 100
                print(f"    {p:25s}: {ps['n']:3d}t  WR={wr_p:5.1f}%  Abs={ps['pnl']:+10.0f}")

        print(f"\n  EXIT REASON BREAKDOWN:")
        for reason, s in sorted(best['reasons'].items(), key=lambda x: -x[1]['n']):
            rwr = s['w'] / max(s['n'], 1) * 100
            print(f"    {reason:12s}: {s['n']:4d} trades  WR={rwr:5.1f}%  PnL={s['pnl']:+.1f}%")

    # --- Walk-forward results ---
    if wf_combined:
        wf_combined.sort(key=lambda x: -x['ann'])
        print(f"\n{'=' * 130}")
        print(f"  WALK-FORWARD VALIDATION (out-of-sample)")
        print(f"{'=' * 130}")
        print(f"  {'Config':60s} | {'Ann':>7s} | {'WR':>5s} | {'N':>5s} | {'DD':>6s} | {'PF':>5s}")
        print(f"  {'-' * 110}")
        for r in wf_combined[:15]:
            print(f"  {r['name']:60s} | {r['ann']:+7.1f}% | {r['wr']:5.1f}% | "
                  f"{r['n']:3d}(D{r['n_dir']:2d}/P{r['n_pair']:2d}) | "
                  f"{r['dd']:6.1f}% | {r['pf']:5.2f}")

    # --- Top 5 combined: yearly detail ---
    if len(full_combined) >= 2:
        print(f"\n  YEARLY BREAKDOWN FOR TOP 5 COMBINED:")
        for idx, r in enumerate(full_combined[:5]):
            print(f"\n  #{idx+1}: {r['name']} (Ann={r['ann']:+.1f}%, WR={r['wr']:.1f}%, DD={r['dd']:.1f}%)")
            for y in sorted(r['yearly'].keys()):
                ys = r['yearly'][y]
                wr_y = ys['w'] / ys['n'] * 100 if ys['n'] > 0 else 0
                total_y = ys['dir_pnl'] + ys['pair_pnl']
                print(f"    {y}: {ys['n']:3d}t  WR={wr_y:5.1f}%  PnL={total_y:+10.0f}  "
                      f"(D={ys['dir_pnl']:+8.0f} P={ys['pair_pnl']:+8.0f})")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.1f}s")
    print("=" * 130)


if __name__ == '__main__':
    main()
