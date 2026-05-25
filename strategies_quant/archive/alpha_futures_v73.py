"""
Alpha Futures V73 -- Cross-Pair Momentum Signals
=================================================
V62 pair trading uses mean reversion on the spread (z-score).
V69 group momentum lag uses directional momentum (1-day hold, LB=3).

V73 hypothesis: What if instead of mean-reverting the spread, we TREND-FOLLOW
the spread? When the spread is widening, bet it continues widening.
When z-score is increasing (becoming more extreme), bet it continues.

THREE SIGNAL TYPES:
  A: Spread Momentum (trend-following on spread) -- PAIR trade
  B: Z-Score Momentum (z-score change) -- PAIR trade
  C: Cross-Pair Confirmation (momentum + pair mean-reversion) -- DIRECTIONAL trade
  combined_all: Best signal across all types

14 pairs, 1-day hold, walk-forward validation (6 windows: 2020-2025).
"""
import sys, os, time, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alpha_v2 import load_all_data, MIN_TRAIN, CASH0

# ================================================================
# CONSTANTS
# ================================================================
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
    'aufi': 'precious', 'agfi': 'precious',
    'afi': 'oils', 'mfi': 'oils', 'yfi': 'oils', 'pfi': 'oils', 'cfi': 'oils',
    'scfi': 'energy', 'mafi': 'energy', 'bfi': 'energy', 'fufi': 'energy', 'pgfi': 'energy',
    'ppfi': 'chemical', 'vfi': 'chemical', 'egfi': 'chemical', 'srfi': 'chemical',
}

# 14 supply chain pairs (downstream, upstream)
PAIRS_14 = [
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

WF_YEARS = [2020, 2021, 2022, 2023, 2024, 2025]


def main():
    t_start = time.time()
    print("=" * 130)
    print("V73 -- Cross-Pair Momentum Signals")
    print("Test: Spread trend-following, Z-score momentum, Cross-pair confirmation")
    print("=" * 130)

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
    print(f"  {NS} commodities, {ND} days, years: {sorted(year_start_di.keys())}")

    # ================================================================
    # BUILD PAIR INDICES
    # ================================================================
    pair_indices = []
    for down_sym, up_sym in PAIRS_14:
        down_si = sym_to_si.get(down_sym, -1)
        up_si = sym_to_si.get(up_sym, -1)
        if down_si >= 0 and up_si >= 0:
            pair_indices.append((down_si, up_si, down_sym, up_sym))
        else:
            print(f"  WARNING: pair ({down_sym}, {up_sym}) not found in data")
    print(f"  Pairs loaded: {len(pair_indices)}/14")

    # ================================================================
    # PRECOMPUTE GROUP MEMBERSHIP (for Signal C)
    # ================================================================
    group_members = {}
    for si in range(NS):
        grp = GROUP_MAP.get(syms[si])
        if grp is None:
            continue
        if grp not in group_members:
            group_members[grp] = []
        group_members[grp].append(si)

    # ================================================================
    # PRECOMPUTE LOG SPREADS FOR ALL PAIRS
    # ================================================================
    print("\n[Signals] Computing spreads...", flush=True)
    t0 = time.time()

    # spread[pair_key][di] = log(C_down) - log(C_up)
    spreads = {}
    for down_si, up_si, down_sym, up_sym in pair_indices:
        key = (down_si, up_si)
        sp = np.full(ND, np.nan)
        for di in range(ND):
            pd_val = C[down_si, di]
            pu_val = C[up_si, di]
            if not np.isnan(pd_val) and not np.isnan(pu_val) and pd_val > 0 and pu_val > 0:
                sp[di] = np.log(pd_val) - np.log(pu_val)
        spreads[key] = sp
    print(f"  Log spreads done ({time.time()-t0:.1f}s)", flush=True)

    # ================================================================
    # PRECOMPUTE SPREAD MOMENTUM (Signal A)
    # spread_mom[lb][pair_key][di] = spread[di] - spread[di-lb]
    # ================================================================
    print("  Computing spread momentum...", flush=True)
    spread_lookbacks = [3, 5, 10]
    spread_mom = {}  # lb -> pair_key -> array
    for lb in spread_lookbacks:
        spread_mom[lb] = {}
        for down_si, up_si, _, _ in pair_indices:
            key = (down_si, up_si)
            sp = spreads[key]
            sm = np.full(ND, np.nan)
            for di in range(lb, ND):
                s_now = sp[di]
                s_prev = sp[di - lb]
                if not np.isnan(s_now) and not np.isnan(s_prev):
                    sm[di] = s_now - s_prev
            spread_mom[lb][key] = sm
    print(f"  Spread momentum done ({time.time()-t0:.1f}s)", flush=True)

    # ================================================================
    # PRECOMPUTE Z-SCORES AND Z-SCORE MOMENTUM (Signal B)
    # z_score[lb][pair_key][di]
    # z_mom[pair_key][di] = z_score[di] - z_score[di-1]
    # ================================================================
    print("  Computing z-scores and z-momentum...", flush=True)
    z_lookbacks = [10, 15]
    z_scores = {}  # lb -> pair_key -> array
    z_mom = {}     # lb -> pair_key -> array

    for lb in z_lookbacks:
        z_scores[lb] = {}
        z_mom[lb] = {}
        for down_si, up_si, _, _ in pair_indices:
            key = (down_si, up_si)
            sp = spreads[key]
            z = np.full(ND, np.nan)
            for di in range(lb, ND):
                window = sp[di - lb:di]
                valid = window[~np.isnan(window)]
                if len(valid) >= max(3, int(lb * 0.7)):
                    m_val = np.mean(valid)
                    s_val = np.std(valid, ddof=1)
                    if s_val > 1e-10:
                        z[di] = (sp[di] - m_val) / s_val
            z_scores[lb][key] = z

            # Z-score momentum: z[di] - z[di-1]
            zm = np.full(ND, np.nan)
            for di in range(1, ND):
                z_now = z[di]
                z_prev = z[di - 1]
                if not np.isnan(z_now) and not np.isnan(z_prev):
                    zm[di] = z_now - z_prev
            z_mom[lb][key] = zm
    print(f"  Z-scores + z-momentum done ({time.time()-t0:.1f}s)", flush=True)

    # ================================================================
    # PRECOMPUTE GROUP MOMENTUM LAG (for Signal C)
    # grp_mom_excl_self[lag][si, di]
    # own_mom[lag][si, di]
    # ================================================================
    print("  Computing group momentum (Signal C)...", flush=True)
    mom_lag_c = 3
    own_mom = np.full((NS, ND), np.nan)
    grp_mom_excl = np.full((NS, ND), np.nan)

    for si in range(NS):
        for di in range(mom_lag_c, ND):
            c_now = C[si, di]
            c_prev = C[si, di - mom_lag_c]
            if not np.isnan(c_now) and not np.isnan(c_prev) and c_prev > 0:
                own_mom[si, di] = (c_now - c_prev) / c_prev

    for grp, members in group_members.items():
        for di in range(mom_lag_c, ND):
            for sj in members:
                ms = []
                for sk in members:
                    if sk == sj:
                        continue
                    mv = own_mom[sk, di]
                    if not np.isnan(mv):
                        ms.append(mv)
                if ms:
                    grp_mom_excl[sj, di] = np.mean(ms)

    # For Signal C: commodities in any pair (unique set)
    all_pair_commodities = set()
    for down_si, up_si, down_sym, up_sym in pair_indices:
        all_pair_commodities.add((down_si, down_sym))
        all_pair_commodities.add((up_si, up_sym))

    # Map commodity -> list of pairs it appears in, and which leg (down/up)
    commodity_pair_info = {}  # si -> [(pair_key, role, other_si), ...]
    for down_si, up_si, down_sym, up_sym in pair_indices:
        key = (down_si, up_si)
        commodity_pair_info.setdefault(down_si, []).append((key, 'down', up_si))
        commodity_pair_info.setdefault(up_si, []).append((key, 'up', down_si))

    print(f"  Group momentum done ({time.time()-t0:.1f}s)", flush=True)

    # ================================================================
    # BACKTEST ENGINE
    # ================================================================
    def run_backtest(
        signal_type='spread_mom',   # 'spread_mom', 'z_mom', 'cross_pair_confirm', 'combined_all'
        spread_lb=3,
        z_lb=15,
        threshold=0.003,
        max_pairs=1,
        comm=COMM,
        start_year=None,
        end_year=None,
        config_name="",
    ):
        """
        Unified backtest for all 4 signal types.

        Signal A (spread_mom): PAIR trade
          - spread_mom > threshold: short downstream + long upstream (spread widening)
          - spread_mom < -threshold: long downstream + short upstream (spread narrowing)
          - 1-day hold

        Signal B (z_mom): PAIR trade
          - z_mom > threshold: z increasing -> short down + long up
          - z_mom < -threshold: z decreasing -> long down + short up
          - 1-day hold

        Signal C (cross_pair_confirm): DIRECTIONAL trade
          - For each commodity, check group momentum lag signal (long only)
          - If the commodity also appears in a pair where z-score is extreme
            AND the z-score direction supports the momentum signal -> trade
          - 1-day hold, single commodity

        Signal D (combined_all): Take best signal across A, B, C
        """
        cash = float(CASH0)
        trades = []
        positions = []  # list of open positions

        start_di = MIN_TRAIN
        end_di = ND
        if start_year is not None:
            if start_year in year_start_di:
                start_di = max(year_start_di[start_year], MIN_TRAIN)
            else:
                return None
        if end_year is not None:
            if end_year in year_end_di:
                end_di = year_end_di[end_year] + 1
            else:
                return None

        for di in range(start_di, end_di):
            year = dates[di].year

            # --- Close positions held from previous day (1-day hold) ---
            closed = []
            for pos in positions:
                si_a = pos.get('si_a', -1)
                si_b = pos.get('si_b', -1)
                is_pair = pos.get('is_pair', False)

                if is_pair:
                    # Pair trade: long one leg, short the other
                    c_a = C[si_a, di]
                    c_b = C[si_b, di]
                    if np.isnan(c_a) or c_a <= 0:
                        c_a = pos['entry_a']
                    if np.isnan(c_b) or c_b <= 0:
                        c_b = pos['entry_b']

                    mult_a = MULT.get(pos['sym_a'], DEF_MULT)
                    mult_b = MULT.get(pos['sym_b'], DEF_MULT)
                    lots_a = pos['lots_a']
                    lots_b = pos['lots_b']

                    # dir=1 means long A / short B; dir=-1 means short A / long B
                    if pos['dir'] == 1:
                        pnl_a = (c_a - pos['entry_a']) * mult_a * lots_a
                        pnl_b = (pos['entry_b'] - c_b) * mult_b * lots_b
                    else:
                        pnl_a = (pos['entry_a'] - c_a) * mult_a * lots_a
                        pnl_b = (c_b - pos['entry_b']) * mult_b * lots_b

                    entry_val = pos['entry_a'] * mult_a * lots_a + pos['entry_b'] * mult_b * lots_b
                    exit_val = c_a * mult_a * lots_a + c_b * mult_b * lots_b
                    cost = entry_val * comm + exit_val * comm
                    total_pnl = pnl_a + pnl_b - cost

                    # Return cash: recover original investment + realized
                    if pos['dir'] == 1:
                        cash_return = c_a * mult_a * lots_a - c_b * mult_b * lots_b
                    else:
                        cash_return = -c_a * mult_a * lots_a + c_b * mult_b * lots_b
                    cash += pos['cash_invested'] + cash_return - exit_val * comm

                    pnl_pct = total_pnl / entry_val * 100 if entry_val > 0 else 0

                    trades.append({
                        'pnl_abs': total_pnl, 'pnl_pct': pnl_pct,
                        'days': 1, 'di': di, 'year': year,
                        'pair': (pos['sym_a'], pos['sym_b']),
                        'pair_label': PAIR_LABEL.get((pos['sym_a'], pos['sym_b']),
                                                     f"{pos['sym_a']}/{pos['sym_b']}"),
                        'dir': pos['dir'], 'reason': 'time',
                        'signal': pos.get('signal', signal_type),
                    })
                else:
                    # Directional trade: single commodity
                    si = si_a
                    cn = C[si, di]
                    if np.isnan(cn) or cn <= 0:
                        cn = pos['entry_a']
                    mult = MULT.get(pos['sym_a'], DEF_MULT)
                    mkt_val = cn * mult * pos['lots_a']
                    pnl = (cn - pos['entry_a']) * mult * pos['lots_a'] * pos['dir']
                    invested = pos['entry_a'] * mult * pos['lots_a']
                    cost = invested * comm + mkt_val * comm
                    pnl_after_cost = pnl - cost
                    pnl_pct = pnl_after_cost / invested * 100 if invested > 0 else 0
                    cash += mkt_val - mkt_val * comm
                    trades.append({
                        'pnl_abs': pnl_after_cost, 'pnl_pct': pnl_pct,
                        'days': 1, 'di': di, 'year': year,
                        'pair': (pos['sym_a'], ''),
                        'pair_label': pos['sym_a'],
                        'dir': pos['dir'], 'reason': 'time',
                        'signal': pos.get('signal', signal_type),
                    })
                closed.append(pos)
            for pos in closed:
                positions.remove(pos)

            # --- Find occupied commodities ---
            occupied = set()
            for pos in positions:
                occupied.add(pos.get('si_a', -1))
                occupied.add(pos.get('si_b', -1))

            # --- Score candidates and open positions ---
            candidates = []  # (score, info_dict)

            if signal_type == 'spread_mom':
                # Signal A: Spread Momentum (PAIR trade)
                for down_si, up_si, down_sym, up_sym in pair_indices:
                    if down_si in occupied or up_si in occupied:
                        continue
                    key = (down_si, up_si)
                    sm = spread_mom[spread_lb][key]
                    val = sm[di] if di < ND else np.nan
                    if np.isnan(val):
                        continue
                    if abs(val) < threshold:
                        continue

                    # val > 0: spread widening (downstream gaining on upstream)
                    # -> short downstream + long upstream (bet spread continues widening)
                    # val < 0: spread narrowing
                    # -> long downstream + short upstream
                    if val > 0:
                        trade_dir = -1  # short down, long up
                    else:
                        trade_dir = 1   # long down, short up

                    candidates.append((abs(val), {
                        'is_pair': True,
                        'si_a': down_si, 'si_b': up_si,
                        'sym_a': down_sym, 'sym_b': up_sym,
                        'dir': trade_dir,
                        'signal': 'spread_mom',
                    }))

            elif signal_type == 'z_mom':
                # Signal B: Z-Score Momentum (PAIR trade)
                for down_si, up_si, down_sym, up_sym in pair_indices:
                    if down_si in occupied or up_si in occupied:
                        continue
                    key = (down_si, up_si)
                    zm = z_mom[z_lb][key]
                    val = zm[di] if di < ND else np.nan
                    if np.isnan(val):
                        continue
                    if abs(val) < threshold:
                        continue

                    # z_mom > 0: z increasing (becoming more extreme or reverting toward 0 from below)
                    # If z > 0 and z_mom > 0: z going more positive -> down expensive -> short down, long up
                    # If z < 0 and z_mom < 0: z going more negative -> down cheap -> long down, short up
                    # Simplify: z_mom > thresh -> short down + long up
                    #           z_mom < -thresh -> long down + short up
                    if val > 0:
                        trade_dir = -1  # short down, long up
                    else:
                        trade_dir = 1   # long down, short up

                    candidates.append((abs(val), {
                        'is_pair': True,
                        'si_a': down_si, 'si_b': up_si,
                        'sym_a': down_sym, 'sym_b': up_sym,
                        'dir': trade_dir,
                        'signal': 'z_mom',
                    }))

            elif signal_type == 'cross_pair_confirm':
                # Signal C: Cross-Pair Confirmation (DIRECTIONAL trade)
                # For each commodity in a pair, check:
                #   1. Group momentum lag signal (grp_mom_excl - own_mom > threshold) -> bullish
                #   2. The commodity appears in a pair with extreme z-score
                #   3. If the z-score direction supports the momentum signal -> STRONG
                for si, sym in all_pair_commodities:
                    if si in occupied:
                        continue
                    if np.isnan(C[si, di]) or C[si, di] <= 0:
                        continue

                    # Check group momentum lag
                    own = own_mom[si, di]
                    grp = grp_mom_excl[si, di]
                    if np.isnan(own) or np.isnan(grp):
                        continue
                    mom_div = grp - own
                    if mom_div <= threshold:
                        continue

                    # Check pair confirmation: is this commodity in a pair with extreme z?
                    pair_infos = commodity_pair_info.get(si, [])
                    best_z_support = 0.0
                    for pair_key, role, other_si in pair_infos:
                        for zlb in z_lookbacks:
                            z_arr = z_scores[zlb].get(pair_key)
                            if z_arr is None:
                                continue
                            z_val = z_arr[di]
                            if np.isnan(z_val):
                                continue
                            # z > 0: down expensive relative to up
                            # If this commodity is the "down" leg and z > 0:
                            #   mean reversion says down will fall (bearish for down)
                            #   BUT momentum says this commodity is bullish
                            #   Conflict -> skip
                            # If this commodity is the "down" leg and z < 0:
                            #   mean reversion says down will rise (bullish for down)
                            #   AND momentum says bullish -> CONFIRM
                            # If this commodity is the "up" leg and z > 0:
                            #   mean reversion says up will fall (bearish for up)
                            #   But momentum is for THIS commodity, not the pair leg
                            # Actually: momentum is for the commodity itself.
                            # Pair z-score provides additional push:
                            #   - If z < 0 for this pair and this is down leg:
                            #     mean reversion will push down UP -> supports long
                            #   - If z > 0 for this pair and this is up leg:
                            #     mean reversion will push up DOWN -> conflicts with long
                            # So we need: (role='down' and z < -0.5) OR (role='up' and z > 0.5)
                            if role == 'down' and z_val < -0.5:
                                support = abs(z_val)
                            elif role == 'up' and z_val > 0.5:
                                support = abs(z_val)
                            else:
                                support = 0.0
                            if support > best_z_support:
                                best_z_support = support

                    if best_z_support < 0.3:
                        continue

                    score = mom_div * best_z_support
                    candidates.append((score, {
                        'is_pair': False,
                        'si_a': si, 'si_b': -1,
                        'sym_a': sym, 'sym_b': '',
                        'dir': 1,  # long
                        'signal': 'cross_pair_confirm',
                    }))

            elif signal_type == 'combined_all':
                # Collect candidates from all three signals, take the best
                # Signal A candidates
                for down_si, up_si, down_sym, up_sym in pair_indices:
                    if down_si in occupied or up_si in occupied:
                        continue
                    key = (down_si, up_si)

                    # Spread momentum
                    sm = spread_mom[spread_lb][key]
                    sm_val = sm[di] if di < ND else np.nan
                    if not np.isnan(sm_val) and abs(sm_val) >= threshold:
                        candidates.append((abs(sm_val) * 10, {  # scale for comparability
                            'is_pair': True,
                            'si_a': down_si, 'si_b': up_si,
                            'sym_a': down_sym, 'sym_b': up_sym,
                            'dir': -1 if sm_val > 0 else 1,
                            'signal': 'spread_mom',
                        }))

                    # Z-score momentum
                    for zlb in z_lookbacks:
                        zm = z_mom[zlb][key]
                        zm_val = zm[di] if di < ND else np.nan
                        if not np.isnan(zm_val) and abs(zm_val) >= threshold:
                            candidates.append((abs(zm_val) * 10, {
                                'is_pair': True,
                                'si_a': down_si, 'si_b': up_si,
                                'sym_a': down_sym, 'sym_b': up_sym,
                                'dir': -1 if zm_val > 0 else 1,
                                'signal': 'z_mom',
                            }))

                # Signal C candidates
                for si, sym in all_pair_commodities:
                    if si in occupied:
                        continue
                    if np.isnan(C[si, di]) or C[si, di] <= 0:
                        continue

                    own = own_mom[si, di]
                    grp = grp_mom_excl[si, di]
                    if np.isnan(own) or np.isnan(grp):
                        continue
                    mom_div = grp - own
                    if mom_div <= threshold:
                        continue

                    pair_infos = commodity_pair_info.get(si, [])
                    best_z_support = 0.0
                    for pair_key, role, other_si in pair_infos:
                        for zlb in z_lookbacks:
                            z_arr = z_scores[zlb].get(pair_key)
                            if z_arr is None:
                                continue
                            z_val = z_arr[di]
                            if np.isnan(z_val):
                                continue
                            if role == 'down' and z_val < -0.5:
                                support = abs(z_val)
                            elif role == 'up' and z_val > 0.5:
                                support = abs(z_val)
                            else:
                                support = 0.0
                            if support > best_z_support:
                                best_z_support = support

                    if best_z_support >= 0.3:
                        score = mom_div * best_z_support
                        candidates.append((score, {
                            'is_pair': False,
                            'si_a': si, 'si_b': -1,
                            'sym_a': sym, 'sym_b': '',
                            'dir': 1,
                            'signal': 'cross_pair_confirm',
                        }))

            if not candidates:
                continue

            # Sort by score (highest first)
            candidates.sort(key=lambda x: -x[0])

            # Open positions up to max_pairs
            n_opened = 0
            for score, info in candidates:
                if n_opened >= max_pairs:
                    break
                if info['si_a'] in occupied or info['si_b'] in occupied:
                    continue

                if info['is_pair']:
                    # Pair trade: long leg A + short leg B (or reverse)
                    si_a = info['si_a']
                    si_b = info['si_b']
                    c_a = C[si_a, di]
                    c_b = C[si_b, di]
                    if np.isnan(c_a) or c_a <= 0 or np.isnan(c_b) or c_b <= 0:
                        continue

                    mult_a = MULT.get(info['sym_a'], DEF_MULT)
                    mult_b = MULT.get(info['sym_b'], DEF_MULT)

                    capital_per_pair = cash / max(1, max_pairs)
                    cash_per_leg = capital_per_pair / 2

                    lots_a = int(cash_per_leg / (c_a * mult_a * (1 + comm)))
                    lots_b = int(cash_per_leg / (c_b * mult_b * (1 + comm)))
                    if lots_a <= 0 or lots_b <= 0:
                        continue

                    cost_a = c_a * mult_a * lots_a * (1 + comm)
                    cost_b = c_b * mult_b * lots_b * (1 + comm)
                    total_cost = cost_a + cost_b
                    if total_cost > cash:
                        scale = cash * 0.95 / total_cost
                        lots_a = max(1, int(lots_a * scale))
                        lots_b = max(1, int(lots_b * scale))
                        cost_a = c_a * mult_a * lots_a * (1 + comm)
                        cost_b = c_b * mult_b * lots_b * (1 + comm)
                        total_cost = cost_a + cost_b
                        if total_cost > cash:
                            continue

                    cash -= total_cost
                    positions.append({
                        'is_pair': True,
                        'si_a': si_a, 'si_b': si_b,
                        'sym_a': info['sym_a'], 'sym_b': info['sym_b'],
                        'entry_a': c_a, 'entry_b': c_b,
                        'lots_a': lots_a, 'lots_b': lots_b,
                        'dir': info['dir'],
                        'entry_di': di,
                        'cash_invested': total_cost,
                        'signal': info['signal'],
                    })
                    occupied.add(si_a)
                    occupied.add(si_b)
                else:
                    # Directional trade: single commodity
                    si = info['si_a']
                    c = C[si, di]
                    if np.isnan(c) or c <= 0:
                        continue

                    mult = MULT.get(info['sym_a'], DEF_MULT)
                    notional = c * mult
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
                        'is_pair': False,
                        'si_a': si, 'si_b': -1,
                        'sym_a': info['sym_a'], 'sym_b': '',
                        'entry_a': c, 'entry_b': 0,
                        'lots_a': lots, 'lots_b': 0,
                        'dir': info['dir'],
                        'entry_di': di,
                        'cash_invested': cost_in,
                        'signal': info['signal'],
                    })
                    occupied.add(si)

                n_opened += 1

        # Close remaining positions at end
        ae = min(end_di, ND) - 1
        for pos in positions:
            si_a = pos.get('si_a', -1)
            si_b = pos.get('si_b', -1)
            is_pair = pos.get('is_pair', False)

            if is_pair:
                c_a = C[si_a, ae]
                c_b = C[si_b, ae]
                if np.isnan(c_a) or c_a <= 0:
                    c_a = pos['entry_a']
                if np.isnan(c_b) or c_b <= 0:
                    c_b = pos['entry_b']

                mult_a = MULT.get(pos['sym_a'], DEF_MULT)
                mult_b = MULT.get(pos['sym_b'], DEF_MULT)
                lots_a = pos['lots_a']
                lots_b = pos['lots_b']

                if pos['dir'] == 1:
                    pnl_a = (c_a - pos['entry_a']) * mult_a * lots_a
                    pnl_b = (pos['entry_b'] - c_b) * mult_b * lots_b
                else:
                    pnl_a = (pos['entry_a'] - c_a) * mult_a * lots_a
                    pnl_b = (c_b - pos['entry_b']) * mult_b * lots_b

                entry_val = pos['entry_a'] * mult_a * lots_a + pos['entry_b'] * mult_b * lots_b
                exit_val = c_a * mult_a * lots_a + c_b * mult_b * lots_b
                cost = entry_val * comm + exit_val * comm
                total_pnl = pnl_a + pnl_b - cost

                if pos['dir'] == 1:
                    cash_return = c_a * mult_a * lots_a - c_b * mult_b * lots_b
                else:
                    cash_return = -c_a * mult_a * lots_a + c_b * mult_b * lots_b
                cash += pos['cash_invested'] + cash_return - exit_val * comm

                pnl_pct = total_pnl / entry_val * 100 if entry_val > 0 else 0
                trades.append({
                    'pnl_abs': total_pnl, 'pnl_pct': pnl_pct,
                    'days': ae - pos['entry_di'], 'di': ae,
                    'year': dates[ae].year,
                    'pair': (pos['sym_a'], pos['sym_b']),
                    'pair_label': PAIR_LABEL.get((pos['sym_a'], pos['sym_b']),
                                                 f"{pos['sym_a']}/{pos['sym_b']}"),
                    'dir': pos['dir'], 'reason': 'end',
                    'signal': pos.get('signal', signal_type),
                })
            else:
                cn = C[si_a, ae]
                if np.isnan(cn) or cn <= 0:
                    cn = pos['entry_a']
                mult = MULT.get(pos['sym_a'], DEF_MULT)
                mkt_val = cn * mult * pos['lots_a']
                pnl = (cn - pos['entry_a']) * mult * pos['lots_a'] * pos['dir']
                invested = pos['entry_a'] * mult * pos['lots_a']
                cost = invested * comm + mkt_val * comm
                pnl_after_cost = pnl - cost
                pnl_pct = pnl_after_cost / invested * 100 if invested > 0 else 0
                cash += mkt_val - mkt_val * comm
                trades.append({
                    'pnl_abs': pnl_after_cost, 'pnl_pct': pnl_pct,
                    'days': ae - pos['entry_di'], 'di': ae,
                    'year': dates[ae].year,
                    'pair': (pos['sym_a'], ''),
                    'pair_label': pos['sym_a'],
                    'dir': pos['dir'], 'reason': 'end',
                    'signal': pos.get('signal', signal_type),
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
        days_total = (dates[last_di] - dates[first_di]).days if last_di > first_di else 365
        yr = max(days_total / 365.25, 0.01)
        ann = ((cash / CASH0) ** (1 / yr) - 1) * 100

        tp = [t['pnl_abs'] for t in sorted(trades, key=lambda x: x['di'])]
        sharpe = np.mean(tp) / np.std(tp) * np.sqrt(252) / float(CASH0) if len(tp) > 1 and np.std(tp) > 0 else 0

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

        # Signal type breakdown
        signal_stats = {}
        for t in trades:
            sig = t.get('signal', '?')
            if sig not in signal_stats:
                signal_stats[sig] = {'n': 0, 'w': 0, 'pnl': 0.0}
            signal_stats[sig]['n'] += 1
            if t['pnl_abs'] > 0:
                signal_stats[sig]['w'] += 1
            signal_stats[sig]['pnl'] += t['pnl_abs']

        # Pair breakdown
        pair_stats = {}
        for t in trades:
            pl = t.get('pair_label', '')
            if pl not in pair_stats:
                pair_stats[pl] = {'n': 0, 'w': 0, 'pnl': 0.0}
            pair_stats[pl]['n'] += 1
            if t['pnl_abs'] > 0:
                pair_stats[pl]['w'] += 1
            pair_stats[pl]['pnl'] += t['pnl_abs']

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
            'sharpe': round(sharpe, 2),
            'cash': round(cash, 0),
            'yearly': year_stats,
            'signal_stats': signal_stats,
            'pair_stats': pair_stats,
            'trades': trades,
        }

    # ================================================================
    # BUILD CONFIGURATIONS
    # ================================================================
    print("\n[Configs] Building configurations...", flush=True)
    configs = []

    # Signal A: Spread Momentum
    # spread_lb x threshold x max_pairs
    for slb in [3, 5, 10]:
        for thr in [0.001, 0.003, 0.005, 0.01]:
            for mp in [1, 3]:
                configs.append({
                    'signal_type': 'spread_mom',
                    'spread_lb': slb,
                    'z_lb': 15,
                    'threshold': thr,
                    'max_pairs': mp,
                    'comm': COMM,
                    'config_name': f"A_SLB{slb}_T{thr*1000:.0f}_MP{mp}",
                })

    # Signal B: Z-Score Momentum
    # z_lb x threshold x max_pairs
    for zlb in [10, 15]:
        for thr in [0.001, 0.003, 0.005, 0.01]:
            for mp in [1, 3]:
                configs.append({
                    'signal_type': 'z_mom',
                    'spread_lb': 3,
                    'z_lb': zlb,
                    'threshold': thr,
                    'max_pairs': mp,
                    'comm': COMM,
                    'config_name': f"B_ZLB{zlb}_T{thr*1000:.0f}_MP{mp}",
                })

    # Signal C: Cross-Pair Confirmation
    # threshold x max_pairs
    for thr in [0.001, 0.003, 0.005, 0.01]:
        for mp in [1, 3]:
            configs.append({
                'signal_type': 'cross_pair_confirm',
                'spread_lb': 3,
                'z_lb': 15,
                'threshold': thr,
                'max_pairs': mp,
                'comm': COMM,
                'config_name': f"C_T{thr*1000:.0f}_MP{mp}",
            })

    # Signal D: Combined All
    # spread_lb x z_lb x threshold x max_pairs
    for slb in [3, 5]:
        for zlb in [10, 15]:
            for thr in [0.001, 0.003, 0.005]:
                for mp in [1, 3]:
                    configs.append({
                        'signal_type': 'combined_all',
                        'spread_lb': slb,
                        'z_lb': zlb,
                        'threshold': thr,
                        'max_pairs': mp,
                        'comm': COMM,
                        'config_name': f"D_SLB{slb}_ZLB{zlb}_T{thr*1000:.0f}_MP{mp}",
                    })

    total_configs = len(configs)
    n_a = sum(1 for c in configs if c['signal_type'] == 'spread_mom')
    n_b = sum(1 for c in configs if c['signal_type'] == 'z_mom')
    n_c = sum(1 for c in configs if c['signal_type'] == 'cross_pair_confirm')
    n_d = sum(1 for c in configs if c['signal_type'] == 'combined_all')
    print(f"  {total_configs} total configurations")
    print(f"    Signal A (spread_mom): {n_a}")
    print(f"    Signal B (z_mom): {n_b}")
    print(f"    Signal C (cross_pair_confirm): {n_c}")
    print(f"    Signal D (combined_all): {n_d}")

    # ================================================================
    # RUN FULL-PERIOD SWEEP
    # ================================================================
    print(f"\n{'=' * 130}")
    print(f"  FULL-PERIOD PARAMETER SWEEP ({total_configs} configs)")
    print(f"{'=' * 130}")

    results = []
    t_sw = time.time()
    for ci, cfg in enumerate(configs):
        r = run_backtest(**cfg)
        if r is not None:
            results.append(r)
        if (ci + 1) % 20 == 0:
            print(f"  [{ci+1}/{total_configs}] {len(results)} with results ({time.time()-t_sw:.1f}s)", flush=True)

    results.sort(key=lambda x: -x['ann'])
    print(f"\n  Sweep complete: {len(results)}/{total_configs} configs ({time.time()-t_sw:.1f}s)", flush=True)

    # ================================================================
    # RESULTS BY SIGNAL TYPE
    # ================================================================
    print(f"\n{'=' * 130}")
    print(f"  V73 -- Cross-Pair Momentum Signals")
    print(f"{'=' * 130}")

    # --- Signal A: Spread Momentum ---
    sig_a_results = [r for r in results if r['name'].startswith('A_')]
    print(f"\n--- Signal A: Spread Momentum ---")
    if sig_a_results:
        best_a = sig_a_results[0]
        print(f"  Best: Ann={best_a['ann']:+.1f}% | WR={best_a['wr']:.1f}% | "
              f"DD={best_a['dd']:.1f}% | PF={best_a['pf']:.2f} | N={best_a['n']} | "
              f"Sharpe={best_a['sharpe']:.2f}")
        print(f"  Config: {best_a['name']}")
        print(f"  Top 5:")
        for i, r in enumerate(sig_a_results[:5]):
            print(f"    {i+1}. {r['name']:35s} Ann={r['ann']:+7.1f}%  WR={r['wr']:5.1f}%  "
                  f"N={r['n']:4d}  DD={r['dd']:5.1f}%  PF={r['pf']:4.2f}")
    else:
        print("  No valid results")

    # --- Signal B: Z-Score Momentum ---
    sig_b_results = [r for r in results if r['name'].startswith('B_')]
    print(f"\n--- Signal B: Z-Score Momentum ---")
    if sig_b_results:
        best_b = sig_b_results[0]
        print(f"  Best: Ann={best_b['ann']:+.1f}% | WR={best_b['wr']:.1f}% | "
              f"DD={best_b['dd']:.1f}% | PF={best_b['pf']:.2f} | N={best_b['n']} | "
              f"Sharpe={best_b['sharpe']:.2f}")
        print(f"  Config: {best_b['name']}")
        print(f"  Top 5:")
        for i, r in enumerate(sig_b_results[:5]):
            print(f"    {i+1}. {r['name']:35s} Ann={r['ann']:+7.1f}%  WR={r['wr']:5.1f}%  "
                  f"N={r['n']:4d}  DD={r['dd']:5.1f}%  PF={r['pf']:4.2f}")
    else:
        print("  No valid results")

    # --- Signal C: Cross-Pair Confirmation ---
    sig_c_results = [r for r in results if r['name'].startswith('C_')]
    print(f"\n--- Signal C: Cross-Pair Confirmation ---")
    if sig_c_results:
        best_c = sig_c_results[0]
        print(f"  Best: Ann={best_c['ann']:+.1f}% | WR={best_c['wr']:.1f}% | "
              f"DD={best_c['dd']:.1f}% | PF={best_c['pf']:.2f} | N={best_c['n']} | "
              f"Sharpe={best_c['sharpe']:.2f}")
        print(f"  Config: {best_c['name']}")
        print(f"  Top 5:")
        for i, r in enumerate(sig_c_results[:5]):
            print(f"    {i+1}. {r['name']:35s} Ann={r['ann']:+7.1f}%  WR={r['wr']:5.1f}%  "
                  f"N={r['n']:4d}  DD={r['dd']:5.1f}%  PF={r['pf']:4.2f}")
    else:
        print("  No valid results")

    # --- Combined ---
    sig_d_results = [r for r in results if r['name'].startswith('D_')]
    print(f"\n--- Combined ---")
    if sig_d_results:
        best_d = sig_d_results[0]
        print(f"  Best: Ann={best_d['ann']:+.1f}% | WR={best_d['wr']:.1f}% | "
              f"DD={best_d['dd']:.1f}% | PF={best_d['pf']:.2f} | N={best_d['n']} | "
              f"Sharpe={best_d['sharpe']:.2f}")
        print(f"  Config: {best_d['name']}")
        if best_d.get('signal_stats'):
            print(f"  Signal breakdown:")
            for sig, ss in sorted(best_d['signal_stats'].items(), key=lambda x: -x[1]['pnl']):
                wr_s = ss['w'] / max(ss['n'], 1) * 100
                print(f"    {sig:25s}: N={ss['n']:4d}  WR={wr_s:5.1f}%  PnL={ss['pnl']:+.0f}")
    else:
        print("  No valid results")

    # ================================================================
    # TOP 20 OVERALL
    # ================================================================
    print(f"\n{'=' * 130}")
    print(f"  TOP 20 OVERALL")
    print(f"{'=' * 130}")
    hdr = (f"  {'#':>2s} | {'Config':40s} | {'Ann':>8s} | {'WR':>5s} | {'N':>5s} | "
           f"{'DD':>6s} | {'PF':>5s} | {'Sh':>6s} | {'AvgW':>6s} | {'AvgL':>6s}")
    print(hdr)
    print(f"  {'-' * 110}")
    for i, r in enumerate(results[:20]):
        print(f"  {i+1:2d} | {r['name']:40s} | {r['ann']:+7.1f}% | {r['wr']:4.1f}% | "
              f"{r['n']:5d} | {r['dd']:5.1f}% | {r['pf']:4.2f} | {r['sharpe']:5.2f} | "
              f"{r['avg_win']:+5.2f}% | {r['avg_loss']:5.2f}%")

    # ================================================================
    # SIGNAL TYPE COMPARISON
    # ================================================================
    print(f"\n{'=' * 130}")
    print(f"  SIGNAL TYPE COMPARISON")
    print(f"{'=' * 130}")
    for sig_name, sig_results in [('A: Spread Momentum', sig_a_results),
                                   ('B: Z-Score Momentum', sig_b_results),
                                   ('C: Cross-Pair Confirm', sig_c_results),
                                   ('D: Combined All', sig_d_results)]:
        if sig_results:
            best = sig_results[0]
            avg_ann = np.mean([r['ann'] for r in sig_results])
            print(f"  {sig_name:25s}: Best={best['ann']:+7.1f}%  Avg={avg_ann:+7.1f}%  "
                  f"Best WR={best['wr']:5.1f}%  Best PF={best['pf']:4.2f}  "
                  f"Best DD={best['dd']:5.1f}%  N_configs={len(sig_results)}")

    # ================================================================
    # YEARLY BREAKDOWN TOP 5
    # ================================================================
    top5 = results[:5]
    print(f"\n{'=' * 130}")
    print(f"  YEARLY BREAKDOWN FOR TOP 5")
    print(f"{'=' * 130}")
    for idx, r in enumerate(top5):
        print(f"\n  #{idx+1}: {r['name']}")
        print(f"    Ann={r['ann']:+.1f}%  WR={r['wr']:.1f}%  DD={r['dd']:.1f}%  "
              f"PF={r['pf']:.2f}  Sharpe={r['sharpe']:.2f}  N={r['n']}")
        print(f"    {'Year':>6s} | {'N':>5s} | {'WR':>5s} | {'PnL%':>8s} | {'PnL Abs':>12s}")
        print(f"    {'-' * 55}")
        for y in sorted(r['yearly'].keys()):
            ys = r['yearly'][y]
            wr_y = ys['w'] / max(ys['n'], 1) * 100
            print(f"    {y:6d} | {ys['n']:5d} | {wr_y:4.1f}% | {ys['pnl']:+7.1f}% | {ys['pnl_abs_sum']:+11.0f}")

    # ================================================================
    # PAIR BREAKDOWN FOR #1
    # ================================================================
    if results:
        best = results[0]
        print(f"\n{'=' * 130}")
        print(f"  PAIR BREAKDOWN for #1: {best['name']}")
        print(f"{'=' * 130}")
        print(f"  {'Pair':30s} | {'N':>5s} | {'WR':>5s} | {'Abs PnL':>12s}")
        print(f"  {'-' * 65}")
        for p in sorted(best['pair_stats'].keys(), key=lambda x: -best['pair_stats'][x]['pnl']):
            ps = best['pair_stats'][p]
            wr_p = ps['w'] / max(ps['n'], 1) * 100
            print(f"  {p:30s} | {ps['n']:5d} | {wr_p:4.1f}% | {ps['pnl']:+11.0f}")

    # ================================================================
    # WALK-FORWARD TOP 5
    # ================================================================
    print(f"\n{'=' * 130}")
    print(f"  WALK-FORWARD VALIDATION (Top 5, 6 windows: 2020-2025)")
    print(f"{'=' * 130}")

    wf_all = []
    wf_by_config = {}
    for rank, top_r in enumerate(top5):
        cn = top_r['name']
        matching = [c for c in configs if c['config_name'] == cn]
        if not matching:
            continue
        bc = matching[0]
        print(f"\n  [{rank+1}] {cn}  (full-period Ann={top_r['ann']:+.1f}%)")
        for ty in WF_YEARS:
            if ty not in year_start_di:
                continue
            wc = dict(bc)
            wc['start_year'] = ty
            wc['end_year'] = ty
            wc['config_name'] = f"WF_{ty}_{cn}"
            r = run_backtest(**wc)
            if r is not None:
                wf_all.append((cn, ty, r))
                wf_by_config.setdefault(cn, []).append((ty, r))
                print(f"    {ty}: Ann={r['ann']:+7.1f}%  WR={r['wr']:5.1f}%  "
                      f"N={r['n']:4d}  DD={r['dd']:5.1f}%  PF={r['pf']:4.2f}  "
                      f"Sharpe={r['sharpe']:5.2f}")
            else:
                print(f"    {ty}: insufficient trades")

    # ================================================================
    # WALK-FORWARD AGGREGATE TABLE
    # ================================================================
    if wf_by_config:
        print(f"\n{'=' * 130}")
        print(f"  WALK-FORWARD AGGREGATE TABLE")
        print(f"{'=' * 130}")
        wf_summary = []
        for cn, wr_list in wf_by_config.items():
            anns = [r['ann'] for _, r in wr_list]
            aby = {ty: r['ann'] for ty, r in wr_list}
            n_pos = sum(1 for a in anns if a > 0)
            wf_summary.append({
                'name': cn,
                'avg_ann': np.mean(anns),
                'med_ann': np.median(anns),
                'min_ann': min(anns),
                'max_ann': max(anns),
                'avg_wr': np.mean([r['wr'] for _, r in wr_list]),
                'avg_dd': np.mean([r['dd'] for _, r in wr_list]),
                'avg_pf': np.mean([r['pf'] for _, r in wr_list]),
                'n_positive': n_pos,
                'n_windows': len(wr_list),
                'by_year': aby,
            })
        wf_summary.sort(key=lambda x: -x['avg_ann'])

        hdr = (f"  {'#':>2s} | {'Config':35s} | {'Avg':>7s} | "
               f"{'2020':>7s} | {'2021':>7s} | {'2022':>7s} | {'2023':>7s} | {'2024':>7s} | {'2025':>7s} | "
               f"{'WR':>5s} | {'DD':>5s} | {'Pos':>4s}")
        print(hdr)
        print(f"  {'-' * 130}")
        for i, w in enumerate(wf_summary):
            ycols = ""
            for y in WF_YEARS:
                a = w['by_year'].get(y, float('nan'))
                ycols += f" | {a:+6.1f}%" if not np.isnan(a) else " |    N/A"
            print(f"  {i+1:2d} | {w['name']:35s} | {w['avg_ann']:+6.1f}%{ycols} | "
                  f"{w['avg_wr']:4.1f}% | {w['avg_dd']:4.1f}% | {w['n_positive']}/{w['n_windows']}")

    # ================================================================
    # OVERFITTING CHECK
    # ================================================================
    if wf_by_config and len(wf_summary) > 2:
        print(f"\n{'=' * 130}")
        print(f"  OVERFITTING CHECK")
        print(f"{'=' * 130}")

        full_anns = []
        wf_anns_list = []
        for w in wf_summary:
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
            print(f"  Decay ratio (WF/IS): {decay:.2f}")
            if corr > 0.5:
                print(f"  -> GOOD: Strong positive correlation")
            elif corr > 0.2:
                print(f"  -> MODERATE: Some predictive power")
            else:
                print(f"  -> WARNING: Weak correlation, possible overfitting")

        all_wf_anns = [r['ann'] for _, _, r in wf_all]
        n_pos_wf = sum(1 for a in all_wf_anns if a > 0)
        print(f"\n  Overall WF positive rate: {n_pos_wf}/{len(all_wf_anns)} "
              f"({n_pos_wf / len(all_wf_anns) * 100:.0f}%)")

        if wf_all:
            best_wf = max(wf_all, key=lambda x: x[2]['ann'])
            worst_wf = min(wf_all, key=lambda x: x[2]['ann'])
            print(f"  Best single window:  Test {best_wf[1]} = {best_wf[2]['ann']:+.1f}%")
            print(f"  Worst single window: Test {worst_wf[1]} = {worst_wf[2]['ann']:+.1f}%")

    # ================================================================
    # KEY FINDINGS
    # ================================================================
    print(f"\n{'=' * 130}")
    print(f"  KEY FINDINGS")
    print(f"{'=' * 130}")

    # Compare trend-following vs mean reversion on spreads
    if sig_a_results:
        best_a = sig_a_results[0]
        print(f"\n  1. Spread Momentum (Trend Following) Best:")
        print(f"     {best_a['name']}: Ann={best_a['ann']:+.1f}%  WR={best_a['wr']:.1f}%  "
              f"DD={best_a['dd']:.1f}%  PF={best_a['pf']:.2f}  N={best_a['n']}")
        print(f"     -> Spread trend-following {'WORKS' if best_a['ann'] > 50 else 'WEAK' if best_a['ann'] > 0 else 'FAILS'} "
              f"compared to mean reversion (V62: +334.3%)")

    if sig_b_results:
        best_b = sig_b_results[0]
        print(f"\n  2. Z-Score Momentum Best:")
        print(f"     {best_b['name']}: Ann={best_b['ann']:+.1f}%  WR={best_b['wr']:.1f}%  "
              f"DD={best_b['dd']:.1f}%  PF={best_b['pf']:.2f}  N={best_b['n']}")
        print(f"     -> Z-score momentum {'WORKS' if best_b['ann'] > 50 else 'WEAK' if best_b['ann'] > 0 else 'FAILS'}")

    if sig_c_results:
        best_c = sig_c_results[0]
        print(f"\n  3. Cross-Pair Confirmation Best:")
        print(f"     {best_c['name']}: Ann={best_c['ann']:+.1f}%  WR={best_c['wr']:.1f}%  "
              f"DD={best_c['dd']:.1f}%  PF={best_c['pf']:.2f}  N={best_c['n']}")
        print(f"     -> Pair info + momentum {'BOOSTS' if best_c['ann'] > 100 else 'HELPS' if best_c['ann'] > 0 else 'HURTS'} momentum signal")

    if results:
        best_overall = results[0]
        print(f"\n  4. Overall Best:")
        print(f"     {best_overall['name']}: Ann={best_overall['ann']:+.1f}%  WR={best_overall['wr']:.1f}%  "
              f"DD={best_overall['dd']:.1f}%  PF={best_overall['pf']:.2f}  Sharpe={best_overall['sharpe']:.2f}")

    # Compare signal types
    print(f"\n  5. Signal Type Rankings:")
    sig_best = []
    if sig_a_results:
        sig_best.append(('A: Spread Momentum', sig_a_results[0]['ann'], sig_a_results[0]['wr'],
                         sig_a_results[0]['pf'], sig_a_results[0]['n']))
    if sig_b_results:
        sig_best.append(('B: Z-Score Momentum', sig_b_results[0]['ann'], sig_b_results[0]['wr'],
                         sig_b_results[0]['pf'], sig_b_results[0]['n']))
    if sig_c_results:
        sig_best.append(('C: Cross-Pair Confirm', sig_c_results[0]['ann'], sig_c_results[0]['wr'],
                         sig_c_results[0]['pf'], sig_c_results[0]['n']))
    if sig_d_results:
        sig_best.append(('D: Combined All', sig_d_results[0]['ann'], sig_d_results[0]['wr'],
                         sig_d_results[0]['pf'], sig_d_results[0]['n']))
    sig_best.sort(key=lambda x: -x[1])
    for rank, (name, ann, wr, pf, n) in enumerate(sig_best):
        print(f"     {rank+1}. {name:25s} Ann={ann:+7.1f}%  WR={wr:5.1f}%  PF={pf:4.2f}  N={n}")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.1f}s")
    print("=" * 130)


if __name__ == '__main__':
    main()
