"""
Alpha Futures V64 -- Expanded Signal Universe for Higher Trade Frequency
========================================================================
V62 champion: +334.3% with 14 pairs and ~2800 trades over 10 years (280/year).
Constraint: MP=1 (1 pair at a time).

Key insight: MORE trading opportunities while maintaining WR = higher returns.

Five orthogonal signal types:
  Signal 1: Pair z-score mean-reversion (V62 baseline, 14 pairs, LOG-biased adaptive)
  Signal 2: Triple commodity arbitrage (supply-chain momentum lag)
  Signal 3: Intra-pair momentum continuation (spread extreme + trend)
  Signal 4: Cross-pair chain confirmation (multiple pairs agree)
  Signal 5: OI surge confirmation (institutional positioning validates signal)

Signal priority queue:
  1. Chain-confirmed pair (Signal 4) -- highest priority
  2. OI-confirmed pair (Signal 5)
  3. Regular pair z-score (Signal 1)
  4. Triplet momentum lag (Signal 2) -- directional, not pair
  5. No signal -> stay in cash

Backtest: 1-day hold, MP=1, 100% capital per trade.
~200 configs with walk-forward for best.
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

# Supply chain triplets: (upstream, midstream, downstream)
TRIPLETS = [
    ('ifi', 'rbfi', 'hcfi'),      # iron ore -> rebar -> hot coil
    ('scfi', 'mafi', 'ppfi'),     # crude -> methanol -> PP
    ('scfi', 'mafi', 'vfi'),      # crude -> methanol -> PVC
    ('scfi', 'mafi', 'egfi'),     # crude -> methanol -> EG
    ('jmfi', 'jfi', 'rbfi'),      # coal -> coke -> rebar
    ('afi', 'mfi', 'pfi'),        # soybean -> meal -> palm (indirect)
    ('afi', 'yfi', 'pfi'),        # soybean -> oil -> palm
]

TRIPLET_LABELS = {
    ('ifi', 'rbfi', 'hcfi'):      'iron_ore->rebar->hotcoil',
    ('scfi', 'mafi', 'ppfi'):     'crude->methanol->PP',
    ('scfi', 'mafi', 'vfi'):      'crude->methanol->PVC',
    ('scfi', 'mafi', 'egfi'):     'crude->methanol->EG',
    ('jmfi', 'jfi', 'rbfi'):      'coal->coke->rebar',
    ('afi', 'mfi', 'pfi'):        'soybean->meal->palm',
    ('afi', 'yfi', 'pfi'):        'soybean->oil->palm',
}

# Supply chain groups for cross-pair confirmation
# Pairs sharing the same chain root are in the same group
CHAIN_GROUPS = {
    'ferrous': [('rbfi', 'ifi'), ('hcfi', 'ifi'), ('hcfi', 'rbfi'), ('jfi', 'jmfi')],
    'petro':   [('mafi', 'scfi'), ('fufi', 'scfi'), ('bfi', 'scfi')],
    'chem':    [('ppfi', 'mafi'), ('vfi', 'mafi'), ('egfi', 'mafi')],
    'soy':     [('mfi', 'afi'), ('yfi', 'afi'), ('pfi', 'yfi')],
    'corn':    [('cfi', 'csfi')],
}

SPREAD_RAW = 'raw'
SPREAD_PCT = 'pct'
SPREAD_LOG = 'log'
ALL_MODES = [SPREAD_RAW, SPREAD_PCT, SPREAD_LOG]
ALL_LOOKBACKS = [5, 7, 10, 15, 20]

# Walk-forward windows: (train_end_year, test_year)
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
    print("Alpha Futures V64 -- Expanded Signal Universe for Higher Trade Frequency")
    print("V62 baseline: +334.3% with 14 pairs, ~280 trades/year, MP=1")
    print("New: 5 signal types (pair z-score, triplet arb, momentum cont, chain confirm, OI)")
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

    pair_indices_14 = build_pair_indices(PAIRS_14)

    # Build triplet index mapping
    def build_triplet_indices(triplets_list):
        indices = []
        for up_sym, mid_sym, down_sym in triplets_list:
            up_si = sym_to_si.get(up_sym, -1)
            mid_si = sym_to_si.get(mid_sym, -1)
            down_si = sym_to_si.get(down_sym, -1)
            if up_si >= 0 and mid_si >= 0 and down_si >= 0:
                indices.append((up_si, mid_si, down_si, up_sym, mid_sym, down_sym))
            else:
                print(f"  WARNING: triplet ({up_sym}, {mid_sym}, {down_sym}) not found")
        return indices

    triplet_indices = build_triplet_indices(TRIPLETS)
    print(f"  Pairs: {len(pair_indices_14)}, Triplets: {len(triplet_indices)}")

    # ================================================================
    # PRECOMPUTE SPREADS AND Z-SCORES
    # ================================================================
    print("\n[Signals] Precomputing spreads and z-scores...", flush=True)
    t0 = time.time()

    z_scores = {m: {} for m in ALL_MODES}

    all_pair_set = set()
    for down_si, up_si, _, _ in pair_indices_14:
        all_pair_set.add((down_si, up_si))

    for down_si, up_si in all_pair_set:
        key = (down_si, up_si)
        for mode in ALL_MODES:
            spread = np.full(ND, np.nan)
            for di in range(ND):
                pd_val = C[down_si, di]
                pu = C[up_si, di]
                if np.isnan(pd_val) or np.isnan(pu) or pu <= 0 or pd_val <= 0:
                    continue
                if mode == SPREAD_RAW:
                    spread[di] = pd_val - pu
                elif mode == SPREAD_PCT:
                    spread[di] = (pd_val - pu) / pu
                elif mode == SPREAD_LOG:
                    spread[di] = np.log(pd_val) - np.log(pu)

            z_scores[mode][key] = {}
            for lb in ALL_LOOKBACKS:
                z = np.full(ND, np.nan)
                for di in range(lb, ND):
                    window = spread[di - lb:di]
                    valid = window[~np.isnan(window)]
                    if len(valid) >= max(3, lb * 0.8):
                        m_val = np.mean(valid)
                        s_val = np.std(valid, ddof=1)
                        if s_val > 1e-10:
                            z[di] = (spread[di] - m_val) / s_val
                z_scores[mode][key][lb] = z

    print(f"  Z-scores precomputed ({time.time() - t0:.1f}s)", flush=True)

    # ================================================================
    # PRECOMPUTE PER-COMMODITY RETURNS (for triplet signals)
    # ================================================================
    print("\n[Signals] Precomputing per-commodity returns...", flush=True)
    t1 = time.time()

    # pct_ret[si, di] = percentage return of commodity si on day di
    pct_ret = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(1, ND):
            if not np.isnan(C[si, di]) and not np.isnan(C[si, di - 1]) and C[si, di - 1] > 0:
                pct_ret[si, di] = (C[si, di] - C[si, di - 1]) / C[si, di - 1]

    # n-day cumulative returns
    cum_ret = {}
    for ndays in [1, 3, 5]:
        cum_ret[ndays] = np.full((NS, ND), np.nan)
        for si in range(NS):
            for di in range(ndays, ND):
                if not np.isnan(C[si, di]) and not np.isnan(C[si, di - ndays]) and C[si, di - ndays] > 0:
                    cum_ret[ndays][si, di] = (C[si, di] - C[si, di - ndays]) / C[si, di - ndays]

    print(f"  Per-commodity returns precomputed ({time.time() - t1:.1f}s)", flush=True)

    # ================================================================
    # PRECOMPUTE OI SURGE METRICS
    # ================================================================
    print("\n[Signals] Precomputing OI surge metrics...", flush=True)
    t_oi = time.time()

    # oi_surge_ratio[si, di] = OI[di] / rolling_avg_OI_20
    oi_surge_ratio = np.full((NS, ND), np.nan)
    oi_surge_threshold = 2.0  # OI > 2x 20-day avg = surge

    for si in range(NS):
        for di in range(20, ND):
            window = OI[si, di - 20:di]
            valid = window[~np.isnan(window)]
            if len(valid) >= 10 and not np.isnan(OI[si, di]):
                avg_oi = np.mean(valid)
                if avg_oi > 0:
                    oi_surge_ratio[si, di] = OI[si, di] / avg_oi

    print(f"  OI surge metrics precomputed ({time.time() - t_oi:.1f}s)", flush=True)

    # ================================================================
    # PRECOMPUTE SPREAD MOMUM (for momentum continuation signal)
    # ================================================================
    print("\n[Signals] Precomputing spread momentum...", flush=True)
    t_mom = time.time()

    # For each pair, compute how many consecutive days z-score has been extreme
    # (same-sign extreme). If z > 2.0 for 3+ days -> momentum continuation, skip MR.
    spread_momentum = {}  # (down_si, up_si) -> momentum_days[di] = consecutive extreme days (signed)
    spread_magnitude = {}  # (down_si, up_si) -> abs z-score at di

    for down_si, up_si in all_pair_set:
        key = (down_si, up_si)
        # Use log spread for momentum
        log_z = z_scores[SPREAD_LOG].get(key, {}).get(10, np.full(ND, np.nan))

        # Track consecutive days z-score is above threshold (same sign)
        mom_days = np.zeros(ND)
        extreme_thresh = 1.5  # z > 1.5 = extreme
        for di in range(1, ND):
            if np.isnan(log_z[di]):
                mom_days[di] = 0
            elif log_z[di] > extreme_thresh:
                # Positive extreme
                if di > 0 and mom_days[di - 1] > 0:
                    mom_days[di] = mom_days[di - 1] + 1
                else:
                    mom_days[di] = 1
            elif log_z[di] < -extreme_thresh:
                # Negative extreme
                if di > 0 and mom_days[di - 1] < 0:
                    mom_days[di] = mom_days[di - 1] - 1
                else:
                    mom_days[di] = -1
            else:
                mom_days[di] = 0

        spread_momentum[key] = mom_days
        spread_magnitude[key] = np.abs(log_z)

    print(f"  Spread momentum precomputed ({time.time() - t_mom:.1f}s)", flush=True)

    # ================================================================
    # PRECOMPUTE GLOBAL COMBO DAILY RETURNS (for adaptive mode selection)
    # ================================================================
    print("\n[Signals] Precomputing global combo daily returns...", flush=True)
    t_gd = time.time()

    all_zt = [0.8, 1.0, 1.2]
    log_bias_combos = [(SPREAD_LOG, 10), (SPREAD_LOG, 15), (SPREAD_LOG, 20),
                       (SPREAD_RAW, 10), (SPREAD_PCT, 10)]

    global_combo_daily_return = {}
    for zt in all_zt:
        for mode in ALL_MODES:
            for lb in ALL_LOOKBACKS:
                combo_key = (mode, lb, zt)
                daily_ret = np.full(ND, np.nan)
                for di in range(MIN_TRAIN + 1, ND):
                    pair_rets = []
                    for down_si, up_si, down_sym, up_sym in pair_indices_14:
                        z_arr = z_scores[mode].get((down_si, up_si), {}).get(lb)
                        if z_arr is None:
                            continue
                        z_prev = z_arr[di - 1]
                        if np.isnan(z_prev) or abs(z_prev) < zt:
                            continue

                        c_down_entry = C[down_si, di - 1]
                        c_up_entry = C[up_si, di - 1]
                        c_down_exit = C[down_si, di]
                        c_up_exit = C[up_si, di]
                        if (np.isnan(c_down_entry) or c_down_entry <= 0 or
                            np.isnan(c_up_entry) or c_up_entry <= 0 or
                            np.isnan(c_down_exit) or c_down_exit <= 0 or
                            np.isnan(c_up_exit) or c_up_exit <= 0):
                            continue

                        mult_down = MULT.get(down_sym, DEF_MULT)
                        mult_up = MULT.get(up_sym, DEF_MULT)

                        if z_prev > 0:
                            pnl_down = (c_down_entry - c_down_exit) * mult_down
                            pnl_up = (c_up_exit - c_up_entry) * mult_up
                        else:
                            pnl_down = (c_down_exit - c_down_entry) * mult_down
                            pnl_up = (c_up_entry - c_up_exit) * mult_up

                        invested = c_down_entry * mult_down + c_up_entry * mult_up
                        cost = invested * COMM * 2
                        pnl_pct = (pnl_down + pnl_up - cost) / invested * 100 if invested > 0 else 0
                        pair_rets.append(pnl_pct)
                    if pair_rets:
                        daily_ret[di] = np.mean(pair_rets)
                global_combo_daily_return[combo_key] = daily_ret

    print(f"  Global combo returns precomputed ({time.time() - t_gd:.1f}s)", flush=True)

    # ================================================================
    # BACKTEST ENGINE
    # ================================================================
    def run_backtest(
        pair_z=1.0,
        triplet_threshold=0.02,
        spread_mode='adaptive_LOG',
        signal_priority='pairs_first',
        mom_cont_days=3,
        mom_cont_z=2.0,
        chain_min_pairs=2,
        oi_surge_ratio_min=2.0,
        start_year=None, end_year=None,
        config_name="",
    ):
        """
        Enhanced backtest with 5 signal types:
          pair_z: z-score threshold for pair signals
          triplet_threshold: minimum return differential for triplet arb
          spread_mode: 'adaptive_LOG', 'log_LB15', 'log_LB10', 'log_LB20'
          signal_priority: 'pairs_first', 'triplets_first', 'combined_score'
          mom_cont_days: minimum consecutive momentum days for Signal 3
          mom_cont_z: minimum z-score magnitude for momentum continuation
          chain_min_pairs: minimum pairs in same chain group to trigger Signal 4
          oi_surge_ratio_min: OI surge ratio for Signal 5
        """
        # Parse spread mode
        if spread_mode == 'adaptive_LOG':
            mode_type = 'adaptive'
            candidate_combos = log_bias_combos
        elif spread_mode == 'log_LB15':
            mode_type = 'fixed'
            candidate_combos = [(SPREAD_LOG, 15)]
        elif spread_mode == 'log_LB10':
            mode_type = 'fixed'
            candidate_combos = [(SPREAD_LOG, 10)]
        elif spread_mode == 'log_LB20':
            mode_type = 'fixed'
            candidate_combos = [(SPREAD_LOG, 20)]
        else:
            mode_type = 'fixed'
            candidate_combos = [(SPREAD_LOG, 10)]

        cash = float(CASH0)
        trades = []
        position = None  # at most 1 position at a time (MP=1)

        current_combo = candidate_combos[0]

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

        # Signal type tracking
        signal_type_counts = {
            'chain_confirmed': 0,
            'oi_confirmed': 0,
            'pair_zscore': 0,
            'triplet_arb': 0,
            'no_signal': 0,
        }

        for di in range(start_di, end_di):

            # --- Adaptive evaluation every 40 days ---
            if mode_type == 'adaptive' and di > start_di:
                days_since_start = di - start_di
                if days_since_start % 40 == 0 and days_since_start >= 40:
                    best_combo = candidate_combos[0]
                    best_score = -1e18
                    for c in candidate_combos:
                        combo_key = (c[0], c[1], pair_z)
                        daily_ret = global_combo_daily_return.get(combo_key)
                        if daily_ret is None:
                            continue
                        window = daily_ret[max(start_di, di - 40):di]
                        valid = window[~np.isnan(window)]
                        if len(valid) >= 3:
                            score = np.nansum(valid)
                        else:
                            score = -1e10
                        if score > best_score:
                            best_score = score
                            best_combo = c
                    current_combo = best_combo

            # Determine spread mode and lookback
            if mode_type == 'fixed':
                use_mode, use_lb = candidate_combos[0]
            else:
                use_mode, use_lb = current_combo

            # --- If we have an open position, close it (1-day hold) ---
            if position is not None:
                pos = position
                p_down_si = pos['down_si']
                p_up_si = pos['up_si']
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
                cost = (entry_val_down + entry_val_up) * COMM + \
                       (exit_val_down + exit_val_up) * COMM

                total_pnl = pnl_down + pnl_up - cost
                invested = entry_val_down + entry_val_up
                pnl_pct = total_pnl / invested * 100 if invested > 0 else 0

                if pos['dir'] == 1:
                    cash_return = c_down * mult_down * lots_down - c_up * mult_up * lots_up
                else:
                    cash_return = -c_down * mult_down * lots_down + c_up * mult_up * lots_up

                cash += pos['cash_invested'] + cash_return - (exit_val_down + exit_val_up) * COMM

                days_held = di - pos['entry_di']
                trades.append({
                    'pnl_abs': total_pnl,
                    'pnl_pct': pnl_pct,
                    'days': days_held,
                    'di': di,
                    'year': dates[di].year,
                    'pair': (pos['down_sym'], pos['up_sym']),
                    'pair_label': pos.get('pair_label', ''),
                    'dir': pos['dir'],
                    'reason': 'time',
                    'signal_type': pos.get('signal_type', 'unknown'),
                })

                position = None

            # --- Generate signals ---
            # Collect all candidate trades with scores and priorities
            # Priority: (priority_level, score, signal_info)
            # Lower priority_level = higher priority (1 = highest)
            candidates = []

            # ===== SIGNAL 1: Pair z-score mean-reversion =====
            for down_si, up_si, down_sym, up_sym in pair_indices_14:
                z_arr = z_scores[use_mode].get((down_si, up_si), {}).get(use_lb)
                if z_arr is None:
                    continue
                z_val = z_arr[di] if di < len(z_arr) else np.nan
                if np.isnan(z_val) or abs(z_val) < pair_z:
                    continue

                pair_label = PAIR_LABEL.get((down_sym, up_sym), f'{down_sym}/{up_sym}')

                # Check if this pair is part of a chain confirmation (Signal 4)
                # Require: chain_min_pairs OTHER pairs in the same chain signal the
                # SAME direction (both z > threshold or both z < -threshold)
                chain_confirmed = False
                for chain_name, chain_pairs in CHAIN_GROUPS.items():
                    if (down_sym, up_sym) not in chain_pairs:
                        continue
                    # Count how many OTHER pairs in this chain signal same direction
                    same_dir_count = 0
                    for cp_down, cp_up in chain_pairs:
                        if (cp_down, cp_up) == (down_sym, up_sym):
                            continue  # exclude self
                        cp_down_si = sym_to_si.get(cp_down, -1)
                        cp_up_si = sym_to_si.get(cp_up, -1)
                        if cp_down_si < 0 or cp_up_si < 0:
                            continue
                        cp_z_arr = z_scores[use_mode].get((cp_down_si, cp_up_si), {}).get(use_lb)
                        if cp_z_arr is None:
                            continue
                        cp_z_val = cp_z_arr[di] if di < len(cp_z_arr) else np.nan
                        if np.isnan(cp_z_val) or abs(cp_z_val) < pair_z:
                            continue
                        # Same direction check: both positive or both negative
                        if (z_val > 0 and cp_z_val > 0) or (z_val < 0 and cp_z_val < 0):
                            same_dir_count += 1
                    if same_dir_count >= chain_min_pairs:
                        chain_confirmed = True
                        break

                # Check OI surge confirmation (Signal 5)
                oi_confirmed = False
                if not np.isnan(oi_surge_ratio[down_si, di]) and oi_surge_ratio[down_si, di] >= oi_surge_ratio_min:
                    oi_confirmed = True
                if not np.isnan(oi_surge_ratio[up_si, di]) and oi_surge_ratio[up_si, di] >= oi_surge_ratio_min:
                    oi_confirmed = True

                # Check momentum continuation (Signal 3) -- skip mean-reversion if spread is trending
                mom_key = (down_si, up_si)
                mom_days_arr = spread_momentum.get(mom_key, np.zeros(ND))
                mom_mag_arr = spread_magnitude.get(mom_key, np.zeros(ND))
                is_momentum_cont = False
                if di < ND and abs(mom_days_arr[di]) >= mom_cont_days:
                    if di < ND and mom_mag_arr[di] > mom_cont_z:
                        is_momentum_cont = True

                # Determine signal type and priority
                if is_momentum_cont:
                    # Signal 3: Skip mean-reversion, this is a momentum continuation
                    # We still add it but with lower priority (don't trade against trend)
                    continue

                if chain_confirmed:
                    sig_type = 'chain_confirmed'
                    priority = 1
                    score = abs(z_val) * 2.0  # boost score for chain confirmation
                elif oi_confirmed:
                    sig_type = 'oi_confirmed'
                    priority = 2
                    score = abs(z_val) * 1.5  # boost for OI confirmation
                else:
                    sig_type = 'pair_zscore'
                    priority = 3
                    score = abs(z_val)

                if z_val > 0:
                    trade_dir = -1  # mean reversion: spread too high, expect it to fall
                else:
                    trade_dir = 1   # spread too low, expect it to rise

                candidates.append({
                    'priority': priority,
                    'score': score,
                    'signal_type': sig_type,
                    'down_si': down_si,
                    'up_si': up_si,
                    'down_sym': down_sym,
                    'up_sym': up_sym,
                    'pair_label': pair_label,
                    'dir': trade_dir,
                    'z_val': z_val,
                })

            # ===== SIGNAL 2: Triplet commodity arbitrage =====
            # When upstream moved but downstream hasn't followed, trade the laggard directionally.
            # We create a synthetic pair: laggard commodity vs itself (directional trade).
            # Practically: we pair the most-lagging downstream with a "reference" commodity.
            # Instead, we trade the actual downstream pair that exists in PAIRS_14.
            for up_si, mid_si, down_si, up_sym, mid_sym, down_sym in triplet_indices:
                # Compute 3-day cumulative returns for each leg
                ret_up_3d = cum_ret[3][up_si, di] if di < ND and not np.isnan(cum_ret[3][up_si, di]) else None
                ret_mid_3d = cum_ret[3][mid_si, di] if di < ND and not np.isnan(cum_ret[3][mid_si, di]) else None
                ret_down_3d = cum_ret[3][down_si, di] if di < ND and not np.isnan(cum_ret[3][down_si, di]) else None

                if ret_up_3d is None or ret_mid_3d is None or ret_down_3d is None:
                    continue

                avg_downstream = (ret_mid_3d + ret_down_3d) / 2
                lag = ret_up_3d - avg_downstream

                # Only signal if lag is significant AND we have a valid pair for the laggard
                if abs(lag) < triplet_threshold:
                    continue

                # Find which actual pair in PAIRS_14 covers one of the laggards
                # Prefer the pair with the most lagging leg
                laggard_sym = mid_sym if ret_mid_3d < ret_down_3d else down_sym
                laggard_si = mid_si if ret_mid_3d < ret_down_3d else down_si

                # Find a pair that includes this laggard
                best_triplet_pair = None
                for p_down_si, p_up_si, p_down_sym, p_up_sym in pair_indices_14:
                    if p_down_sym == laggard_sym or p_up_sym == laggard_sym:
                        # Check that this pair doesn't already have a z-score signal
                        z_arr = z_scores[use_mode].get((p_down_si, p_up_si), {}).get(use_lb)
                        if z_arr is not None:
                            z_val_t = z_arr[di] if di < len(z_arr) else np.nan
                            if not np.isnan(z_val_t) and abs(z_val_t) >= pair_z:
                                continue  # already captured by pair z-score signal
                        triplet_label = TRIPLET_LABELS.get((up_sym, mid_sym, down_sym), '')
                        best_triplet_pair = {
                            'priority': 4,
                            'score': abs(lag) * 20,
                            'signal_type': 'triplet_arb',
                            'down_si': p_down_si,
                            'up_si': p_up_si,
                            'down_sym': p_down_sym,
                            'up_sym': p_up_sym,
                            'pair_label': f'{p_down_sym}/{p_up_sym}_tri{triplet_label}',
                            'dir': 1 if lag > 0 else -1,  # long pair if upstream up, short if upstream down
                            'z_val': abs(lag),
                        }
                        break

                if best_triplet_pair is not None:
                    candidates.append(best_triplet_pair)

            # ===== Select best signal based on priority =====
            if not candidates:
                signal_type_counts['no_signal'] += 1
                continue

            # Sort by priority (lower = higher priority), then by score (higher = better)
            if signal_priority == 'pairs_first':
                candidates.sort(key=lambda x: (x['priority'], -x['score']))
            elif signal_priority == 'triplets_first':
                # Swap priorities: triplets get priority 1
                for c in candidates:
                    if c['signal_type'] == 'triplet_arb':
                        c['priority'] = 0  # highest
                candidates.sort(key=lambda x: (x['priority'], -x['score']))
            elif signal_priority == 'combined_score':
                candidates.sort(key=lambda x: -x['score'])

            best = candidates[0]

            # Track signal type
            signal_type_counts[best['signal_type']] += 1

            # --- Open position ---
            down_si = best['down_si']
            up_si = best['up_si']
            down_sym = best['down_sym']
            up_sym = best['up_sym']

            c_down = C[down_si, di]
            c_up = C[up_si, di]
            if np.isnan(c_down) or c_down <= 0 or np.isnan(c_up) or c_up <= 0:
                continue

            mult_down = MULT.get(down_sym, DEF_MULT)
            mult_up = MULT.get(up_sym, DEF_MULT)

            capital_for_pair = cash
            cash_per_leg = capital_for_pair / 2

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

            cash -= total_cost
            position = {
                'down_si': down_si,
                'up_si': up_si,
                'down_sym': down_sym,
                'up_sym': up_sym,
                'entry_down': c_down,
                'entry_up': c_up,
                'lots_down': lots_down,
                'lots_up': lots_up,
                'entry_di': di,
                'entry_z': best['z_val'],
                'dir': best['dir'],
                'cash_invested': total_cost,
                'pair_label': best['pair_label'],
                'signal_type': best['signal_type'],
            }

        # Close remaining position at end
        actual_end = min(end_di, ND) - 1
        if position is not None:
            pos = position
            p_down_si = pos['down_si']
            p_up_si = pos['up_si']
            c_down = C[p_down_si, actual_end]
            c_up = C[p_up_si, actual_end]
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
            cost = (entry_val_down + entry_val_up) * COMM + \
                   (exit_val_down + exit_val_up) * COMM

            total_pnl = pnl_down + pnl_up - cost
            invested = entry_val_down + entry_val_up
            pnl_pct = total_pnl / invested * 100 if invested > 0 else 0

            if pos['dir'] == 1:
                cash_return = c_down * mult_down * lots_down - c_up * mult_up * lots_up
            else:
                cash_return = -c_down * mult_down * lots_down + c_up * mult_up * lots_up

            cash += pos['cash_invested'] + cash_return - (exit_val_down + exit_val_up) * COMM

            trades.append({
                'pnl_abs': total_pnl,
                'pnl_pct': pnl_pct,
                'days': actual_end - pos['entry_di'],
                'di': actual_end,
                'year': dates[actual_end].year,
                'pair': (pos['down_sym'], pos['up_sym']),
                'pair_label': pos.get('pair_label', ''),
                'dir': pos['dir'],
                'reason': 'end',
                'signal_type': pos.get('signal_type', 'unknown'),
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
            'signal_type_counts': signal_type_counts,
            'trades': trades,
        }

    # ================================================================
    # BUILD CONFIGURATIONS (~200)
    # ================================================================
    configs = []

    pair_z_values = [0.8, 1.0, 1.2]
    triplet_thresholds = [0.01, 0.02, 0.03]
    spread_modes = ['adaptive_LOG', 'log_LB15']
    signal_priorities = ['pairs_first', 'triplets_first', 'combined_score']

    # Full grid: 3 x 3 x 2 x 3 = 54 configs
    for pz in pair_z_values:
        for tt in triplet_thresholds:
            for sm in spread_modes:
                for sp in signal_priorities:
                    name = f"pz{pz}_tt{tt}_{sm}_{sp}"
                    configs.append({
                        'pair_z': pz,
                        'triplet_threshold': tt,
                        'spread_mode': sm,
                        'signal_priority': sp,
                        'config_name': name,
                    })

    # Additional configs: vary momentum continuation params
    mom_days_values = [2, 3]
    mom_z_values = [1.5, 2.0]
    for pz in [0.8, 1.0]:
        for sm in ['adaptive_LOG', 'log_LB15']:
            for md in mom_days_values:
                for mz in mom_z_values:
                    name = f"pz{pz}_{sm}_mom{md}d_z{mz}"
                    configs.append({
                        'pair_z': pz,
                        'triplet_threshold': 0.02,
                        'spread_mode': sm,
                        'signal_priority': 'pairs_first',
                        'mom_cont_days': md,
                        'mom_cont_z': mz,
                        'config_name': name,
                    })

    # Chain confirmation sensitivity
    chain_min_values = [2, 3]
    for pz in [0.8, 1.0]:
        for sm in ['adaptive_LOG']:
            for cm in chain_min_values:
                for sp in ['pairs_first', 'combined_score']:
                    name = f"pz{pz}_{sm}_chain{cm}_{sp}"
                    configs.append({
                        'pair_z': pz,
                        'triplet_threshold': 0.02,
                        'spread_mode': sm,
                        'signal_priority': sp,
                        'chain_min_pairs': cm,
                        'config_name': name,
                    })

    # OI surge sensitivity
    oi_thresholds = [1.5, 2.0, 2.5]
    for pz in [0.8, 1.0]:
        for ot in oi_thresholds:
            for sp in ['pairs_first', 'combined_score']:
                name = f"pz{pz}_oi{ot}_{sp}"
                configs.append({
                    'pair_z': pz,
                    'triplet_threshold': 0.02,
                    'spread_mode': 'adaptive_LOG',
                    'signal_priority': sp,
                    'oi_surge_ratio_min': ot,
                    'config_name': name,
                })

    # Fixed LB variants with triplet-only
    for lb in [5, 7, 10, 15, 20]:
        for pz in [0.8, 1.0]:
            for tt in [0.01, 0.02]:
                for sp in ['pairs_first', 'combined_score']:
                    name = f"pz{pz}_LB{lb}_tt{tt}_{sp}"
                    configs.append({
                        'pair_z': pz,
                        'triplet_threshold': tt,
                        'spread_mode': f'log_LB{lb}',
                        'signal_priority': sp,
                        'config_name': name,
                    })

    total_configs = len(configs)
    print(f"\n{'=' * 160}")
    print(f"  PARAMETER SWEEP ({total_configs} configs)")
    print(f"  Grid 1: pair_z x triplet_thresh x spread_mode x signal_priority = 54")
    print(f"  Grid 2: momentum continuation params = 36")
    print(f"  Grid 3: chain confirmation params = 16")
    print(f"  Grid 4: OI surge sensitivity = 12")
    print(f"  Grid 5: fixed LB variants = 40")
    print(f"{'=' * 160}")

    # ================================================================
    # RUN SWEEP
    # ================================================================
    results = []
    t_sweep_start = time.time()

    for ci, cfg in enumerate(configs):
        r = run_backtest(**cfg)
        if r is not None:
            results.append(r)

        if (ci + 1) % 50 == 0:
            elapsed = time.time() - t_sweep_start
            print(f"  [{ci + 1}/{total_configs}] {len(results)} with results ({elapsed:.1f}s)", flush=True)

    results.sort(key=lambda x: -x['ann'])
    print(f"\n  Sweep complete: {len(results)}/{total_configs} configs ({time.time() - t_sweep_start:.1f}s)",
          flush=True)

    # ================================================================
    # TOP 20 FULL-PERIOD RESULTS
    # ================================================================
    print(f"\n{'=' * 160}")
    print(f"  TOP 20 FULL-PERIOD RESULTS")
    print(f"{'=' * 160}")
    print(f"  {'#':>2s} | {'Config':55s} | {'Ann':>8s} | {'WR':>5s} | {'N':>5s} | "
          f"{'DD':>6s} | {'PF':>5s} | {'Sharpe':>7s} | {'AvgW':>6s} | {'AvgL':>6s} | "
          f"{'AvgD':>5s} | {'Cash':>12s}")
    print(f"  {'-' * 160}")

    for i, r in enumerate(results[:20]):
        print(f"  {i + 1:2d} | {r['name']:55s} | {r['ann']:+7.1f}% | {r['wr']:4.1f}% | "
              f"{r['n']:5d} | {r['dd']:5.1f}% | {r['pf']:4.2f} | {r['sharpe']:6.2f} | "
              f"{r['avg_win']:+5.2f}% | {r['avg_loss']:5.2f}% | {r['avg_days']:4.1f} | "
              f"{r['cash']:11.0f}")

    # ================================================================
    # SIGNAL TYPE DISTRIBUTION
    # ================================================================
    print(f"\n{'=' * 160}")
    print(f"  SIGNAL TYPE DISTRIBUTION (Top 20 configs)")
    print(f"{'=' * 160}")

    for i, r in enumerate(results[:20]):
        stc = r['signal_type_counts']
        total_signals = sum(v for k, v in stc.items() if k != 'no_signal')
        total_days = total_signals + stc.get('no_signal', 0)
        print(f"  {i + 1:2d} | {r['name']:55s}")
        print(f"      Chain: {stc.get('chain_confirmed', 0):5d}  "
              f"OI: {stc.get('oi_confirmed', 0):5d}  "
              f"PairZ: {stc.get('pair_zscore', 0):5d}  "
              f"Triplet: {stc.get('triplet_arb', 0):5d}  "
              f"NoSignal: {stc.get('no_signal', 0):5d}  "
              f"TotalTrades: {total_signals}  "
              f"TradeRate: {total_signals / max(total_days, 1) * 100:.1f}%")

    # Aggregate signal type across top 20
    print(f"\n  Aggregate signal type across Top 20:")
    agg_signals = {'chain_confirmed': 0, 'oi_confirmed': 0, 'pair_zscore': 0,
                   'triplet_arb': 0, 'no_signal': 0}
    for r in results[:20]:
        for k in agg_signals:
            agg_signals[k] += r['signal_type_counts'].get(k, 0)
    total_agg = sum(v for k, v in agg_signals.items() if k != 'no_signal')
    for k in ['chain_confirmed', 'oi_confirmed', 'pair_zscore', 'triplet_arb']:
        print(f"    {k:20s}: {agg_signals[k]:6d} ({agg_signals[k] / max(total_agg, 1) * 100:.1f}%)")
    print(f"    {'Total trades':20s}: {total_agg}")
    print(f"    {'No signal days':20s}: {agg_signals['no_signal']}")

    # ================================================================
    # PARAMETER SENSITIVITY ANALYSIS
    # ================================================================
    print(f"\n{'=' * 160}")
    print(f"  PARAMETER SENSITIVITY ANALYSIS")
    print(f"{'=' * 160}")

    # Pair Z sensitivity
    print(f"\n  By pair_z threshold:")
    for pz in pair_z_values:
        subset = [r for r in results if f"pz{pz}_" in r['name']]
        if subset:
            avg_ann = np.mean([r['ann'] for r in subset])
            best = max(subset, key=lambda x: x['ann'])
            avg_n = np.mean([r['n'] for r in subset])
            print(f"    Z={pz}: N={len(subset):3d}  Avg Ann={avg_ann:+7.1f}%  "
                  f"Avg Trades={avg_n:.0f}  Best Ann={best['ann']:+.1f}% ({best['name']})")

    # Triplet threshold sensitivity
    print(f"\n  By triplet threshold:")
    for tt in triplet_thresholds:
        subset = [r for r in results if f"_tt{tt}_" in r['name']]
        if subset:
            avg_ann = np.mean([r['ann'] for r in subset])
            best = max(subset, key=lambda x: x['ann'])
            avg_n = np.mean([r['n'] for r in subset])
            print(f"    TT={tt}: N={len(subset):3d}  Avg Ann={avg_ann:+7.1f}%  "
                  f"Avg Trades={avg_n:.0f}  Best Ann={best['ann']:+.1f}% ({best['name']})")

    # Spread mode sensitivity
    print(f"\n  By spread mode:")
    for sm in ['adaptive_LOG', 'log_LB15', 'log_LB10', 'log_LB20']:
        subset = [r for r in results if sm in r['name']]
        if subset:
            avg_ann = np.mean([r['ann'] for r in subset])
            best = max(subset, key=lambda x: x['ann'])
            avg_n = np.mean([r['n'] for r in subset])
            print(f"    {sm:15s}: N={len(subset):3d}  Avg Ann={avg_ann:+7.1f}%  "
                  f"Avg Trades={avg_n:.0f}  Best Ann={best['ann']:+.1f}%")

    # Signal priority sensitivity
    print(f"\n  By signal priority:")
    for sp in signal_priorities:
        subset = [r for r in results if r['name'].endswith(f'_{sp}')]
        if subset:
            avg_ann = np.mean([r['ann'] for r in subset])
            best = max(subset, key=lambda x: x['ann'])
            avg_n = np.mean([r['n'] for r in subset])
            print(f"    {sp:16s}: N={len(subset):3d}  Avg Ann={avg_ann:+7.1f}%  "
                  f"Avg Trades={avg_n:.0f}  Best Ann={best['ann']:+.1f}%")

    # ================================================================
    # PER-PAIR STATS (for #1 overall config)
    # ================================================================
    if results:
        best_overall = results[0]
        print(f"\n{'=' * 160}")
        print(f"  PER-PAIR STATS for #1 Config: {best_overall['name']}")
        print(f"  Ann={best_overall['ann']:+.1f}%  WR={best_overall['wr']:.1f}%  "
              f"N={best_overall['n']}  DD={best_overall['dd']:.1f}%  PF={best_overall['pf']:.2f}  "
              f"Sharpe={best_overall['sharpe']:.2f}")
        print(f"{'=' * 160}")
        print(f"  {'Pair':30s} | {'N':>5s} | {'WR':>5s} | {'Abs PnL':>12s} | {'Avg PnL':>10s}")
        print(f"  {'-' * 75}")

        for p in sorted(best_overall['pair_stats'].keys(),
                        key=lambda x: -best_overall['pair_stats'][x]['pnl']):
            ps = best_overall['pair_stats'][p]
            wr_p = ps['w'] / max(ps['n'], 1) * 100
            avg_pnl = ps['pnl'] / max(ps['n'], 1)
            print(f"  {p:30s} | {ps['n']:5d} | {wr_p:4.1f}% | {ps['pnl']:+11.0f} | "
                  f"{avg_pnl:+9.0f}")

        # Year-by-year for #1 config
        print(f"\n  Year-by-year breakdown for #1 config:")
        print(f"  {'Year':>6s} | {'N':>5s} | {'WR':>5s} | {'PnL Abs':>12s} | {'PnL %':>8s}")
        print(f"  {'-' * 50}")
        for y in sorted(best_overall['yearly'].keys()):
            ys = best_overall['yearly'][y]
            wr_y = ys['w'] / max(ys['n'], 1) * 100
            print(f"  {y:6d} | {ys['n']:5d} | {wr_y:4.1f}% | {ys['pnl_abs_sum']:+11.0f} | "
                  f"{ys['pnl']:+7.1f}%")

        # Signal type for #1
        stc = best_overall['signal_type_counts']
        print(f"\n  Signal type breakdown for #1 config:")
        for k in ['chain_confirmed', 'oi_confirmed', 'pair_zscore', 'triplet_arb', 'no_signal']:
            print(f"    {k:20s}: {stc.get(k, 0):6d}")

    # ================================================================
    # YEARLY FOR TOP 5
    # ================================================================
    if len(results) >= 2:
        print(f"\n  YEARLY BREAKDOWN FOR TOP 5 CONFIGS:")
        for idx, r in enumerate(results[:5]):
            print(f"\n  #{idx + 1}: {r['name']} (Ann={r['ann']:+.1f}%, WR={r['wr']:.1f}%, "
                  f"DD={r['dd']:.1f}%, Sharpe={r['sharpe']:.2f}, N={r['n']})")
            for y in sorted(r['yearly'].keys()):
                ys = r['yearly'][y]
                wr_y = ys['w'] / ys['n'] * 100 if ys['n'] > 0 else 0
                print(f"    {y}: {ys['n']:4d}t  WR={wr_y:5.1f}%  PnL={ys['pnl']:+.1f}%")

    # ================================================================
    # WALK-FORWARD FOR TOP 5
    # ================================================================
    top5_for_wf = results[:5]

    print(f"\n{'=' * 160}")
    print(f"  RIGOROUS 6-WINDOW WALK-FORWARD (Top 5 configs)")
    print(f"  Windows: {WF_WINDOWS}")
    print(f"{'=' * 160}")

    wf_all = []
    wf_by_config = {}

    for rank, cfg in enumerate(top5_for_wf):
        cfg_name = cfg['name']
        matching = [c for c in configs if c['config_name'] == cfg_name]
        if not matching:
            print(f"  [{rank + 1}] {cfg_name} -- config not found, SKIP")
            continue

        base_cfg = matching[0]
        print(f"\n  [{rank + 1}] {cfg_name}  (full-period Ann={cfg['ann']:+.1f}%)")

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

    print(f"  {'#':>2s} | {'Config':55s} | {'Avg Ann':>8s} | {'Med Ann':>8s} | {'Min Ann':>8s} | "
          f"{'Max Ann':>8s} | {'Avg WR':>6s} | {'Avg N':>6s} | {'Avg DD':>7s} | {'Avg PF':>6s} | "
          f"{'Avg Sh':>6s} | {'Pos/Win':>7s}")
    print(f"  {'-' * 170}")

    for i, w in enumerate(wf_avg):
        print(f"  {i + 1:2d} | {w['name']:55s} | {w['avg_ann']:+7.1f}% | {w['med_ann']:+7.1f}% | "
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
        print(f"\n  Overall WF positive rate: {n_pos_wf}/{len(all_wf_anns)} "
              f"({n_pos_wf / len(all_wf_anns) * 100:.0f}%)")

        if wf_all:
            best_single = max(wf_all, key=lambda x: x[3]['ann'])
            worst_single = min(wf_all, key=lambda x: x[3]['ann'])
            print(f"  Best single window OOS:  Test {best_single[2]} = "
                  f"{best_single[3]['ann']:+.1f}% ({best_single[0][:50]})")
            print(f"  Worst single window OOS: Test {worst_single[2]} = "
                  f"{worst_single[3]['ann']:+.1f}% ({worst_single[0][:50]})")

    # ================================================================
    # PAIR PROFITABILITY ACROSS TOP 20
    # ================================================================
    if results:
        print(f"\n{'=' * 160}")
        print(f"  PAIR PROFITABILITY ACROSS TOP 20 CONFIGS")
        print(f"{'=' * 160}")

        pair_summary = {}
        for r in results[:20]:
            for p, ps in r['pair_stats'].items():
                if p not in pair_summary:
                    pair_summary[p] = {'n': 0, 'w': 0, 'pnl': 0.0}
                pair_summary[p]['n'] += ps['n']
                pair_summary[p]['w'] += ps['w']
                pair_summary[p]['pnl'] += ps['pnl']

        for p in sorted(pair_summary.keys(), key=lambda x: -pair_summary[x]['pnl']):
            ps = pair_summary[p]
            wr_p = ps['w'] / max(ps['n'], 1) * 100
            print(f"  {p:30s}: {ps['n']:5d} trades  WR={wr_p:5.1f}%  Total Abs={ps['pnl']:+12.0f}")

    # ================================================================
    # SIGNAL TYPE TRADE FREQUENCY ANALYSIS
    # ================================================================
    print(f"\n{'=' * 160}")
    print(f"  SIGNAL TYPE TRADE FREQUENCY ANALYSIS")
    print(f"{'=' * 160}")

    # Compare top configs by trade frequency
    top_by_freq = sorted(results[:50], key=lambda x: -x['n'])
    print(f"\n  Top 10 by trade frequency (from top 50 by return):")
    print(f"  {'#':>2s} | {'Config':55s} | {'N':>5s} | {'Ann':>8s} | {'WR':>5s} | "
          f"{'Signal Dist':40s}")
    print(f"  {'-' * 130}")
    for i, r in enumerate(top_by_freq[:10]):
        stc = r['signal_type_counts']
        sig_dist = f"Chain:{stc.get('chain_confirmed',0)} OI:{stc.get('oi_confirmed',0)} " \
                   f"Pair:{stc.get('pair_zscore',0)} Trip:{stc.get('triplet_arb',0)}"
        print(f"  {i + 1:2d} | {r['name']:55s} | {r['n']:5d} | {r['ann']:+7.1f}% | "
              f"{r['wr']:4.1f}% | {sig_dist:40s}")

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
        stc = results[0]['signal_type_counts']
        total_trades = sum(v for k, v in stc.items() if k != 'no_signal')
        print(f"    Signal distribution: Chain={stc.get('chain_confirmed',0)}  "
              f"OI={stc.get('oi_confirmed',0)}  Pair={stc.get('pair_zscore',0)}  "
              f"Triplet={stc.get('triplet_arb',0)}  Total={total_trades}")

    if wf_avg:
        print(f"\n  Walk-forward best: {wf_avg[0]['name']}")
        print(f"    WF Avg Ann={wf_avg[0]['avg_ann']:+.1f}%  WF Med Ann={wf_avg[0]['med_ann']:+.1f}%  "
              f"Min={wf_avg[0]['min_ann']:+.1f}%  Max={wf_avg[0]['max_ann']:+.1f}%  "
              f"Pos/Win={wf_avg[0]['n_positive']}/{wf_avg[0]['n_windows']}")

        n_all_positive = sum(1 for w in wf_avg if w['n_positive'] == w['n_windows'])
        print(f"\n  Of top 5 WF configs, {n_all_positive} are positive in ALL test windows")

    # Comparison to baselines
    print(f"\n  Baseline comparison:")
    print(f"    V62 baseline:         +334.3% (LOG-biased adaptive, EP40, Z=0.8, MP1, 14 pairs, ~280 trades/year)")
    if results:
        print(f"    V64 best full-period: {results[0]['ann']:+.1f}%  N={results[0]['n']}  "
              f"(~{results[0]['n'] // max(1, len(results[0]['yearly']))} trades/year)")
    if wf_avg:
        print(f"    V64 best WF avg:      {wf_avg[0]['avg_ann']:+.1f}%")
        print(f"    V64 best WF min:      {wf_avg[0]['min_ann']:+.1f}% (worst single window)")
        print(f"    V64 best WF max:      {wf_avg[0]['max_ann']:+.1f}% (best single window)")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.1f}s")
    print("=" * 160)


if __name__ == '__main__':
    main()
