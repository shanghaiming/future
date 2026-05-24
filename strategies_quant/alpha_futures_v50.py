"""
Alpha Futures V50 — Volatility Regime Rotation
================================================
Instead of filtering out high-vol periods (which hurt V36), use volatility
regime to SWITCH between strategies:

  - Low vol regime (calm):     Use mean-reversion / pair trading (V39)
  - High vol regime (trending): Use momentum following (V34b)
  - The regime itself is the signal — not a filter to skip trades, but
    a selector for WHICH strategy to use.

Precompute:
  1. Rolling 20-day realized volatility
  2. Rolling 60-day average volatility
  3. Volatility ratio = vol20 / vol60_avg
  4. KER-based regime detection
  5. Group momentum lag (V34b signal)
  6. Pair z-scores (V39 signal)

Strategy rotation:
  if vol_ratio < low_threshold:   # calm -> mean-reversion via pairs
      Use V39 pair signal
  elif vol_ratio > high_threshold:  # trending -> momentum following
      Use V34b momentum signal
  else:
      Use both signals, take whichever is stronger

~200 configs tested with walk-forward validation.
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
    'rbfi': 'ferrous', 'hcfi': 'ferrous', 'ifi': 'ferrous',
    'jfi': 'ferrous', 'jmfi': 'ferrous',
    'cufi': 'nonferrous', 'alfi': 'nonferrous',
    'znfi': 'nonferrous', 'nifi': 'nonferrous',
    'afi': 'oils', 'mfi': 'oils', 'yfi': 'oils',
    'pfi': 'oils', 'cfi': 'oils',
    'scfi': 'energy', 'mafi': 'energy',
    'bfi': 'energy', 'fufi': 'energy',
    'ppfi': 'chemical', 'vfi': 'chemical',
    'egfi': 'chemical', 'pgfi': 'chemical',
}

PAIRS = [
    ('rbfi', 'ifi'), ('hcfi', 'ifi'), ('hcfi', 'rbfi'),
    ('jfi', 'jmfi'), ('mafi', 'scfi'), ('fufi', 'scfi'),
    ('bfi', 'scfi'), ('mfi', 'afi'), ('yfi', 'afi'),
    ('pfi', 'yfi'), ('ppfi', 'mafi'), ('vfi', 'mafi'),
    ('egfi', 'mafi'),
]


def main():
    t_start = time.time()
    print("=" * 130)
    print("Alpha Futures V50 -- Volatility Regime Rotation")
    print("Core: Low vol -> pair mean-reversion | High vol -> group momentum | Regime = signal")
    print("=" * 130)

    NS, ND, dates, C, O, H, L, V, syms, sym_set = load_all_data()
    sym_to_si = {syms[si]: si for si in range(NS)}

    # Build group membership
    group_members = {}
    for si in range(NS):
        grp = GROUP_MAP.get(syms[si])
        if grp is None:
            continue
        if grp not in group_members:
            group_members[grp] = []
        group_members[grp].append(si)

    # Build pair index mapping
    pair_indices = []
    for down_sym, up_sym in PAIRS:
        down_si = sym_to_si.get(down_sym, -1)
        up_si = sym_to_si.get(up_sym, -1)
        if down_si >= 0 and up_si >= 0:
            pair_indices.append((down_si, up_si, down_sym, up_sym))
        else:
            print(f"  WARNING: pair ({down_sym}, {up_sym}) not found in data")

    print(f"  {NS} commodities, {ND} days, {len(pair_indices)} active pairs, "
          f"{len(group_members)} groups")

    # ========================================================
    # PRECOMPUTE ALL SIGNALS
    # ========================================================
    print("\n[Signals] Computing all signals...", flush=True)
    t0 = time.time()

    # --- 1. Daily returns ---
    daily_ret = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(1, ND):
            c0 = C[si, di - 1]
            c1 = C[si, di]
            if not np.isnan(c0) and not np.isnan(c1) and c0 > 0:
                daily_ret[si, di] = (c1 - c0) / c0

    # --- 2. Realized volatility: rolling 20-day std of daily returns ---
    vol20 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            window = daily_ret[si, di - 20:di]
            valid = window[~np.isnan(window)]
            if len(valid) >= 15:
                vol20[si, di] = np.std(valid, ddof=1)

    # --- 3. Average volatility: rolling 60-day mean of vol20 ---
    vol60_avg = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(60, ND):
            window = vol20[si, di - 60:di]
            valid = window[~np.isnan(window)]
            if len(valid) >= 40:
                vol60_avg[si, di] = np.mean(valid)

    # --- 4. Volatility ratio ---
    vol_ratio = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(60, ND):
            v20 = vol20[si, di]
            v60 = vol60_avg[si, di]
            if not np.isnan(v20) and not np.isnan(v60) and v60 > 1e-12:
                vol_ratio[si, di] = v20 / v60

    # --- 5. KER (Kaufman Efficiency Ratio, 20-day) ---
    ker = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            c_now = C[si, di]
            c_20 = C[si, di - 20]
            if np.isnan(c_now) or np.isnan(c_20) or c_20 <= 0:
                continue
            net = abs(c_now - c_20)
            total = 0
            for dd in range(di - 19, di + 1):
                c1 = C[si, dd]
                c0 = C[si, dd - 1]
                if not np.isnan(c1) and not np.isnan(c0):
                    total += abs(c1 - c0)
            if total > 0:
                ker[si, di] = net / total

    # --- 6. Momentum at multiple lookbacks ---
    mom = {}
    for lag in [3, 5, 7, 10, 15]:
        m = np.full((NS, ND), np.nan)
        for si in range(NS):
            for di in range(lag, ND):
                c_now = C[si, di]
                c_prev = C[si, di - lag]
                if not np.isnan(c_now) and not np.isnan(c_prev) and c_prev > 0:
                    m[si, di] = (c_now - c_prev) / c_prev
        mom[lag] = m

    # --- 7. Group momentum (excluding self) ---
    grp_mom = {}
    for lag in [5, 7]:
        gm = np.full((NS, ND), np.nan)
        for grp, members in group_members.items():
            for di in range(lag, ND):
                for sj in members:
                    ms = []
                    for sk in members:
                        if sk == sj:
                            continue
                        val = mom[lag][sk, di]
                        if not np.isnan(val):
                            ms.append(val)
                    if ms:
                        gm[sj, di] = np.mean(ms)
        grp_mom[lag] = gm

    # --- 8. ATR 10-day ---
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

    # --- 9. Pair spreads and z-scores ---
    pair_spreads = {}
    pair_zscores = {}
    for down_si, up_si, down_sym, up_sym in pair_indices:
        spread = np.full(ND, np.nan)
        for di in range(ND):
            pd_val = C[down_si, di]
            pu_val = C[up_si, di]
            if not np.isnan(pd_val) and not np.isnan(pu_val):
                spread[di] = pd_val - pu_val
        pair_spreads[(down_si, up_si)] = spread

    print(f"  Signals computed ({time.time()-t0:.1f}s)", flush=True)

    # ========================================================
    # SIGNAL GENERATORS
    # ========================================================

    def momentum_score(si, di, mom_lag=5, min_lag=0.003, scale=10.0):
        """V34b group momentum lag score: own lags group -> catches up."""
        own = mom[mom_lag][si, di]
        grp = grp_mom[mom_lag][si, di]
        if np.isnan(own) or np.isnan(grp):
            return np.nan
        divergence = grp - own
        if abs(divergence) < min_lag:
            return np.nan
        sc = np.clip(divergence * scale, -1, 1)
        if sc <= 0:
            return np.nan
        return sc

    def pair_zscore(down_si, up_si, di, lookback=20):
        """V39 pair z-score for spread mean-reversion."""
        sp = pair_spreads.get((down_si, up_si))
        if sp is None:
            return np.nan
        if di < lookback:
            return np.nan
        window = sp[di - lookback:di]
        valid = window[~np.isnan(window)]
        if len(valid) < lookback * 0.8:
            return np.nan
        mean_val = np.mean(valid)
        std_val = np.std(valid, ddof=1)
        if std_val < 1e-10:
            return np.nan
        cur = sp[di]
        if np.isnan(cur):
            return np.nan
        return (cur - mean_val) / std_val

    def get_vol_regime(si, di, low_thresh, high_thresh):
        """Return volatility regime: 'low', 'mid', or 'high'."""
        vr = vol_ratio[si, di]
        if np.isnan(vr):
            return 'mid'
        if vr < low_thresh:
            return 'low'
        elif vr > high_thresh:
            return 'high'
        return 'mid'

    def get_ker_regime(si, di, ker_trend=0.3, ker_range=0.15):
        """Return KER regime: 'trending', 'ranging', or 'neutral'."""
        k = ker[si, di]
        if np.isnan(k):
            return 'neutral'
        if k > ker_trend:
            return 'trending'
        elif k < ker_range:
            return 'ranging'
        return 'neutral'

    # ========================================================
    # BACKTEST ENGINE — REGIME ROTATION
    # ========================================================

    def run_backtest(
        # Regime thresholds
        vol_low=0.7, vol_high=1.3,
        # Regime source: 'vol', 'ker', 'both'
        regime_src='vol',
        # KER thresholds
        ker_trend=0.3, ker_range=0.15,
        # Momentum params (V34b)
        mom_lag=5, min_lag=0.003, mom_scale=10.0,
        # Pair params (V39)
        pair_lookback=20, pair_z_thresh=1.5,
        pair_hold_max=7,
        # Mid-vol behaviour: 'both_equal', 'momentum_heavy', 'pairs_heavy'
        mid_behaviour='both_equal',
        # Position management
        max_positions=3, hold_max=5, trail_atr_mult=2.5,
        # Walk-forward
        wf_split_year=None,
        config_name="",
    ):
        """
        Regime rotation backtest. Shared capital pool.
        Can hold both directional (momentum) and pair positions simultaneously.
        Max total positions = max_positions (counting each pair as 1 position).
        """
        cash = float(CASH0)
        trades = []
        positions = []  # each pos is a dict

        for di in range(MIN_TRAIN, ND):
            year = dates[di].year
            if wf_split_year is not None and year < wf_split_year:
                continue

            # --- Manage existing positions ---
            new_positions = []
            for pos in positions:
                pos_type = pos['type']  # 'momentum' or 'pair'

                if pos_type == 'momentum':
                    c = C[pos['si'], di]
                    if np.isnan(c) or c <= 0:
                        c = pos['entry']
                    mult = MULT.get(pos['sym'], DEF_MULT)
                    mkt_val = c * mult * pos['lots']
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

                    # Signal flip exit
                    if exit_reason is None and days_held >= 2:
                        cur_score = momentum_score(pos['si'], di, mom_lag, min_lag, mom_scale)
                        if not np.isnan(cur_score) and cur_score < -0.01:
                            exit_reason = 'signal_flip'

                    # Time exit
                    if exit_reason is None and days_held >= hold_max:
                        exit_reason = 'time'

                    if exit_reason:
                        cost_out = mkt_val * COMM
                        cash += mkt_val - cost_out
                        trades.append({
                            'pnl_pct': pnl_pct, 'pnl_abs': pnl,
                            'days': days_held, 'di': di, 'year': year,
                            'sym': pos['sym'], 'type': 'momentum',
                            'dir': pos['dir'], 'reason': exit_reason,
                            'regime_entry': pos['regime'],
                        })
                    else:
                        new_positions.append(pos)

                elif pos_type == 'pair':
                    p_down_si = pos['down_si']
                    p_up_si = pos['up_si']
                    z_now = pair_zscore(p_down_si, p_up_si, di, pair_lookback)
                    days_held = di - pos['entry_di']
                    entry_z = pos['entry_z']
                    pos_dir = pos['dir']

                    exit_reason = None

                    # Mean reversion exit
                    if not np.isnan(z_now):
                        if pos_dir == 1 and z_now <= 0:
                            exit_reason = 'mean_rev'
                        elif pos_dir == -1 and z_now >= 0:
                            exit_reason = 'mean_rev'

                    # Stop loss
                    if exit_reason is None and not np.isnan(z_now):
                        if pos_dir == 1 and z_now < entry_z - 1.0:
                            exit_reason = 'stop_loss'
                        elif pos_dir == -1 and z_now > entry_z + 1.0:
                            exit_reason = 'stop_loss'

                    # Time exit
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

                        trades.append({
                            'pnl_pct': pnl_pct, 'pnl_abs': total_pnl,
                            'days': days_held, 'di': di, 'year': year,
                            'pair': (pos['down_sym'], pos['up_sym']),
                            'type': 'pair', 'dir': pos_dir,
                            'reason': exit_reason,
                            'regime_entry': pos['regime'],
                        })
                    else:
                        new_positions.append(pos)

            positions = new_positions

            # --- Count open positions (pair = 1 position) ---
            n_open = len(positions)
            if n_open >= max_positions:
                continue

            # --- Collect occupied symbols ---
            occupied = set()
            for pos in positions:
                if pos['type'] == 'momentum':
                    occupied.add(pos['si'])
                elif pos['type'] == 'pair':
                    occupied.add(pos['down_si'])
                    occupied.add(pos['up_si'])

            slots = max_positions - n_open
            if slots <= 0:
                continue

            # --- Score all possible trades ---
            # Momentum candidates (V34b style)
            momentum_candidates = []
            for si in range(NS):
                if si in occupied:
                    continue
                sym = syms[si]
                if GROUP_MAP.get(sym) is None:
                    continue
                sc = momentum_score(si, di, mom_lag, min_lag, mom_scale)
                if np.isnan(sc) or sc <= 0.01:
                    continue
                # Determine regime for this symbol
                vr = get_vol_regime(si, di, vol_low, vol_high)
                kr = get_ker_regime(si, di, ker_trend, ker_range)
                if regime_src == 'vol':
                    regime = vr
                elif regime_src == 'ker':
                    regime = 'high' if kr == 'trending' else ('low' if kr == 'ranging' else 'mid')
                else:  # 'both'
                    if vr == 'high' or kr == 'trending':
                        regime = 'high'
                    elif vr == 'low' and kr == 'ranging':
                        regime = 'low'
                    else:
                        regime = 'mid'

                # Weight based on regime
                weight = 0.0
                if regime == 'high':
                    weight = 1.0  # Momentum thrives in high vol
                elif regime == 'mid':
                    if mid_behaviour == 'both_equal':
                        weight = 0.5
                    elif mid_behaviour == 'momentum_heavy':
                        weight = 0.7
                    elif mid_behaviour == 'pairs_heavy':
                        weight = 0.3
                    else:
                        weight = 0.5
                # regime == 'low': momentum not used

                if weight > 0:
                    momentum_candidates.append((si, sc * weight, sym, regime, 'momentum'))

            # Pair candidates (V39 style)
            pair_candidates = []
            for down_si, up_si, down_sym, up_sym in pair_indices:
                if down_si in occupied or up_si in occupied:
                    continue
                z_val = pair_zscore(down_si, up_si, di, pair_lookback)
                if np.isnan(z_val):
                    continue
                if abs(z_val) < pair_z_thresh:
                    continue

                # Use average vol_ratio of the two legs for regime
                vr_down = get_vol_regime(down_si, di, vol_low, vol_high)
                vr_up = get_vol_regime(up_si, di, vol_low, vol_high)

                if regime_src == 'vol':
                    regime = vr_down if vr_down == vr_up else 'mid'
                elif regime_src == 'ker':
                    kr_d = get_ker_regime(down_si, di, ker_trend, ker_range)
                    kr_u = get_ker_regime(up_si, di, ker_trend, ker_range)
                    if kr_d == 'ranging' or kr_u == 'ranging':
                        regime = 'low'
                    elif kr_d == 'trending' and kr_u == 'trending':
                        regime = 'high'
                    else:
                        regime = 'mid'
                else:
                    vol_high_any = (vr_down == 'high' or vr_up == 'high')
                    vol_low_both = (vr_down == 'low' and vr_up == 'low')
                    kr_d = get_ker_regime(down_si, di, ker_trend, ker_range)
                    kr_u = get_ker_regime(up_si, di, ker_trend, ker_range)
                    if vol_high_any or (kr_d == 'trending' and kr_u == 'trending'):
                        regime = 'high'
                    elif vol_low_both and (kr_d == 'ranging' or kr_u == 'ranging'):
                        regime = 'low'
                    else:
                        regime = 'mid'

                # Weight based on regime
                weight = 0.0
                if regime == 'low':
                    weight = 1.0  # Pairs thrive in low vol
                elif regime == 'mid':
                    if mid_behaviour == 'both_equal':
                        weight = 0.5
                    elif mid_behaviour == 'momentum_heavy':
                        weight = 0.3
                    elif mid_behaviour == 'pairs_heavy':
                        weight = 0.7
                    else:
                        weight = 0.5
                # regime == 'high': pairs not used

                if weight > 0:
                    pair_candidates.append(
                        (abs(z_val) * weight, down_si, up_si, down_sym, up_sym,
                         z_val, regime, 'pair')
                    )

            # Combine and sort all candidates
            all_candidates = []
            for c in momentum_candidates:
                all_candidates.append({
                    'score': c[1],
                    'type': 'momentum',
                    'si': c[0], 'sym': c[2], 'regime': c[3],
                })
            for c in pair_candidates:
                all_candidates.append({
                    'score': c[0],
                    'type': 'pair',
                    'down_si': c[1], 'up_si': c[2],
                    'down_sym': c[3], 'up_sym': c[4],
                    'z_val': c[5], 'regime': c[6],
                })

            all_candidates.sort(key=lambda x: -x['score'])

            # Open positions
            opened = 0
            for cand in all_candidates:
                if opened >= slots:
                    break
                if cand['type'] == 'momentum':
                    si = cand['si']
                    if si in occupied:
                        continue
                    sym = cand['sym']
                    c = C[si, di]
                    if np.isnan(c) or c <= 0:
                        continue
                    mult = MULT.get(sym, DEF_MULT)
                    notional = c * mult
                    if notional <= 0:
                        continue
                    cash_per_slot = cash / max(slots - opened, 1)
                    lots = int(cash_per_slot / (notional * (1 + COMM)))
                    if lots <= 0:
                        continue
                    cost_in = notional * lots * (1 + COMM)
                    if cost_in > cash:
                        lots = int(cash / (notional * (1 + COMM)))
                        if lots <= 0:
                            continue
                        cost_in = notional * lots * (1 + COMM)

                    atr_val = atr10[si, di] if not np.isnan(atr10[si, di]) else 0
                    cash -= cost_in
                    trail_price = c - trail_atr_mult * atr_val
                    positions.append({
                        'type': 'momentum',
                        'si': si, 'entry': c, 'entry_di': di,
                        'lots': lots, 'dir': 1, 'sym': sym,
                        'atr': atr_val, 'trail_price': trail_price,
                        'regime': cand['regime'],
                    })
                    occupied.add(si)
                    opened += 1

                elif cand['type'] == 'pair':
                    down_si = cand['down_si']
                    up_si = cand['up_si']
                    if down_si in occupied or up_si in occupied:
                        continue
                    c_down = C[down_si, di]
                    c_up = C[up_si, di]
                    if np.isnan(c_down) or c_down <= 0 or np.isnan(c_up) or c_up <= 0:
                        continue

                    mult_down = MULT.get(cand['down_sym'], DEF_MULT)
                    mult_up = MULT.get(cand['up_sym'], DEF_MULT)

                    cash_per_leg = cash / max(2, slots - opened + 1)
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

                    z_val = cand['z_val']
                    if z_val > 0:
                        pos_dir = -1  # short down + long up
                    else:
                        pos_dir = 1   # long down + short up

                    cash -= total_cost
                    positions.append({
                        'type': 'pair',
                        'down_si': down_si, 'up_si': up_si,
                        'down_sym': cand['down_sym'],
                        'up_sym': cand['up_sym'],
                        'entry_down': c_down, 'entry_up': c_up,
                        'lots_down': lots_down, 'lots_up': lots_up,
                        'entry_di': di, 'entry_z': z_val,
                        'dir': pos_dir,
                        'cash_invested': total_cost,
                        'regime': cand['regime'],
                    })
                    occupied.add(down_si)
                    occupied.add(up_si)
                    opened += 1

        # Close remaining positions at end
        for pos in positions:
            if pos['type'] == 'momentum':
                c = C[pos['si'], ND - 1]
                if np.isnan(c) or c <= 0:
                    c = pos['entry']
                mult = MULT.get(pos['sym'], DEF_MULT)
                pnl = (c - pos['entry']) * mult * pos['lots'] * pos['dir']
                cash += c * mult * pos['lots'] * (1 - COMM)
                trades.append({
                    'pnl_pct': pnl / (pos['entry'] * mult * pos['lots']) * 100,
                    'pnl_abs': pnl, 'days': ND - 1 - pos['entry_di'],
                    'di': ND - 1, 'year': dates[ND - 1].year,
                    'sym': pos['sym'], 'type': 'momentum',
                    'dir': pos['dir'], 'reason': 'end',
                    'regime_entry': pos['regime'],
                })
            elif pos['type'] == 'pair':
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
                    'pnl_pct': pnl_pct, 'pnl_abs': total_pnl,
                    'days': ND - 1 - pos['entry_di'],
                    'di': ND - 1, 'year': dates[ND - 1].year,
                    'pair': (pos['down_sym'], pos['up_sym']),
                    'type': 'pair', 'dir': pos['dir'], 'reason': 'end',
                    'regime_entry': pos['regime'],
                })

        if len(trades) < 5:
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

        # Sharpe approximation
        trade_pnls = [t['pnl_abs'] for t in sorted(trades, key=lambda x: x['di'])]
        if len(trade_pnls) > 1:
            rets = np.array(trade_pnls) / float(CASH0)
            mean_ret = np.mean(rets)
            std_ret = np.std(rets)
            sharpe_approx = mean_ret / std_ret * np.sqrt(252) if std_ret > 0 else 0
        else:
            sharpe_approx = 0

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

        # Exit reason breakdown
        reasons = {}
        for t in trades:
            r = t['reason']
            if r not in reasons:
                reasons[r] = {'n': 0, 'w': 0, 'pnl': 0.0}
            reasons[r]['n'] += 1
            if t['pnl_abs'] > 0:
                reasons[r]['w'] += 1
            reasons[r]['pnl'] += t['pnl_pct']

        # Yearly breakdown
        year_stats = {}
        for t in trades:
            y = t['year']
            if y not in year_stats:
                year_stats[y] = {'n': 0, 'w': 0, 'pnl': 0.0, 'pnl_abs': 0.0}
            year_stats[y]['n'] += 1
            if t['pnl_abs'] > 0:
                year_stats[y]['w'] += 1
            year_stats[y]['pnl'] += t['pnl_pct']
            year_stats[y]['pnl_abs'] += t['pnl_abs']

        # Trade type breakdown
        type_stats = {}
        for t in trades:
            tp = t['type']
            if tp not in type_stats:
                type_stats[tp] = {'n': 0, 'w': 0, 'pnl': 0.0, 'pnl_abs': 0.0}
            type_stats[tp]['n'] += 1
            if t['pnl_abs'] > 0:
                type_stats[tp]['w'] += 1
            type_stats[tp]['pnl'] += t['pnl_pct']
            type_stats[tp]['pnl_abs'] += t['pnl_abs']

        # Regime breakdown
        regime_stats = {}
        for t in trades:
            rg = t.get('regime_entry', 'unknown')
            if rg not in regime_stats:
                regime_stats[rg] = {'n': 0, 'w': 0, 'pnl': 0.0, 'pnl_abs': 0.0}
            regime_stats[rg]['n'] += 1
            if t['pnl_abs'] > 0:
                regime_stats[rg]['w'] += 1
            regime_stats[rg]['pnl'] += t['pnl_pct']
            regime_stats[rg]['pnl_abs'] += t['pnl_abs']

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
            'reasons': reasons,
            'yearly': year_stats,
            'type_stats': type_stats,
            'regime_stats': regime_stats,
            'trades': trades,
        }

    # ========================================================
    # PARAMETER SWEEP
    # ========================================================
    print("\n[Backtest] Building configurations...", flush=True)
    configs = []

    # --- A. Vol-regime rotation configs ---
    for vol_low in [0.5, 0.7, 0.8]:
        for vol_high in [1.2, 1.3, 1.5]:
            for mid_beh in ['both_equal', 'momentum_heavy', 'pairs_heavy']:
                for mom_lag in [5, 7]:
                    for pair_lb in [20]:
                        name = (f"VL{vol_low}_VH{vol_high}_MID{mid_beh[:3]}_"
                                f"ML{mom_lag}_PLB{pair_lb}")
                        configs.append({
                            'vol_low': vol_low, 'vol_high': vol_high,
                            'regime_src': 'vol',
                            'mom_lag': mom_lag,
                            'pair_lookback': pair_lb,
                            'mid_behaviour': mid_beh,
                            'max_positions': 3, 'hold_max': 5,
                            'trail_atr_mult': 2.5,
                            'wf_split_year': None,
                            'config_name': name,
                        })

    # --- B. KER-regime rotation configs ---
    for ker_trend in [0.3]:
        for ker_range in [0.15]:
            for mid_beh in ['both_equal', 'momentum_heavy', 'pairs_heavy']:
                for mom_lag in [5, 7]:
                    name = (f"KER_KT{ker_trend}_KR{ker_range}_MID{mid_beh[:3]}_"
                            f"ML{mom_lag}")
                    configs.append({
                        'vol_low': 0.7, 'vol_high': 1.3,
                        'regime_src': 'ker',
                        'ker_trend': ker_trend, 'ker_range': ker_range,
                        'mom_lag': mom_lag,
                        'mid_behaviour': mid_beh,
                        'max_positions': 3, 'hold_max': 5,
                        'trail_atr_mult': 2.5,
                        'wf_split_year': None,
                        'config_name': name,
                    })

    # --- C. Both-regime rotation configs ---
    for vol_low in [0.7, 0.8]:
        for vol_high in [1.2, 1.3]:
            for mid_beh in ['both_equal', 'momentum_heavy', 'pairs_heavy']:
                for mom_lag in [5, 7]:
                    name = (f"BOTH_VL{vol_low}_VH{vol_high}_MID{mid_beh[:3]}_"
                            f"ML{mom_lag}")
                    configs.append({
                        'vol_low': vol_low, 'vol_high': vol_high,
                        'regime_src': 'both',
                        'mom_lag': mom_lag,
                        'mid_behaviour': mid_beh,
                        'max_positions': 3, 'hold_max': 5,
                        'trail_atr_mult': 2.5,
                        'wf_split_year': None,
                        'config_name': name,
                    })

    # --- D. Pair z-threshold sweep ---
    for z_thresh in [1.0, 1.5, 2.0]:
        for vol_low in [0.7]:
            for vol_high in [1.3]:
                for mid_beh in ['both_equal']:
                    name = (f"ZT{z_thresh}_VL{vol_low}_VH{vol_high}_"
                            f"MID{mid_beh[:3]}")
                    configs.append({
                        'vol_low': vol_low, 'vol_high': vol_high,
                        'regime_src': 'vol',
                        'pair_z_thresh': z_thresh,
                        'mid_behaviour': mid_beh,
                        'max_positions': 3, 'hold_max': 5,
                        'trail_atr_mult': 2.5,
                        'wf_split_year': None,
                        'config_name': name,
                    })

    # --- E. Hold period sweep ---
    for hold_max in [3, 5, 7]:
        for pair_hold_max in [5, 7, 10]:
            for vol_low in [0.7]:
                for vol_high in [1.3]:
                    name = (f"H{hold_max}_PH{pair_hold_max}_VL{vol_low}_VH{vol_high}")
                    configs.append({
                        'vol_low': vol_low, 'vol_high': vol_high,
                        'regime_src': 'vol',
                        'pair_hold_max': pair_hold_max,
                        'max_positions': 3, 'hold_max': hold_max,
                        'trail_atr_mult': 2.5,
                        'wf_split_year': None,
                        'config_name': name,
                    })

    # --- F. Max positions sweep ---
    for max_pos in [1, 2, 3]:
        for vol_low in [0.7]:
            for vol_high in [1.3]:
                name = f"MAX{max_pos}_VL{vol_low}_VH{vol_high}"
                configs.append({
                    'vol_low': vol_low, 'vol_high': vol_high,
                    'regime_src': 'vol',
                    'max_positions': max_pos, 'hold_max': 5,
                    'trail_atr_mult': 2.5,
                    'wf_split_year': None,
                    'config_name': name,
                })

    # --- G. Baseline: always momentum only ---
    for mom_lag in [5, 7]:
        for hold_max in [3, 5]:
            name = f"MOM_ONLY_ML{mom_lag}_H{hold_max}"
            configs.append({
                'vol_low': 0.0, 'vol_high': 999.0,  # always high regime -> always momentum
                'regime_src': 'vol',
                'mom_lag': mom_lag,
                'max_positions': 3, 'hold_max': hold_max,
                'trail_atr_mult': 2.5,
                'wf_split_year': None,
                'config_name': name,
            })

    # --- H. Baseline: always pairs only ---
    for pair_lb in [10, 20, 30]:
        for z_thresh in [1.0, 1.5, 2.0]:
            name = f"PAIR_ONLY_LB{pair_lb}_ZT{z_thresh}"
            configs.append({
                'vol_low': 999.0, 'vol_high': 999.0,  # always low regime -> always pairs
                'regime_src': 'vol',
                'pair_lookback': pair_lb,
                'pair_z_thresh': z_thresh,
                'max_positions': 3, 'hold_max': 5,
                'trail_atr_mult': 2.5,
                'wf_split_year': None,
                'config_name': name,
            })

    # --- I. Walk-forward for promising configs ---
    for vol_low in [0.7]:
        for vol_high in [1.3]:
            for mid_beh in ['both_equal', 'momentum_heavy', 'pairs_heavy']:
                for mom_lag in [5, 7]:
                    for wf_year in [2023, 2024]:
                        name = (f"VL{vol_low}_VH{vol_high}_MID{mid_beh[:3]}_"
                                f"ML{mom_lag}_WF{wf_year}")
                        configs.append({
                            'vol_low': vol_low, 'vol_high': vol_high,
                            'regime_src': 'vol',
                            'mom_lag': mom_lag,
                            'mid_behaviour': mid_beh,
                            'max_positions': 3, 'hold_max': 5,
                            'trail_atr_mult': 2.5,
                            'wf_split_year': wf_year,
                            'config_name': name,
                        })

    # Walk-forward for both-regime
    for vol_low in [0.7]:
        for vol_high in [1.3]:
            for mid_beh in ['both_equal']:
                for mom_lag in [5, 7]:
                    for wf_year in [2023, 2024]:
                        name = (f"BOTH_VL{vol_low}_VH{vol_high}_MID{mid_beh[:3]}_"
                                f"ML{mom_lag}_WF{wf_year}")
                        configs.append({
                            'vol_low': vol_low, 'vol_high': vol_high,
                            'regime_src': 'both',
                            'mom_lag': mom_lag,
                            'mid_behaviour': mid_beh,
                            'max_positions': 3, 'hold_max': 5,
                            'trail_atr_mult': 2.5,
                            'wf_split_year': wf_year,
                            'config_name': name,
                        })

    # Walk-forward for baselines
    for mom_lag in [5]:
        for wf_year in [2023, 2024]:
            name = f"MOM_ONLY_ML{mom_lag}_H5_WF{wf_year}"
            configs.append({
                'vol_low': 0.0, 'vol_high': 999.0,
                'regime_src': 'vol',
                'mom_lag': mom_lag,
                'max_positions': 3, 'hold_max': 5,
                'trail_atr_mult': 2.5,
                'wf_split_year': wf_year,
                'config_name': name,
            })

    for pair_lb in [20]:
        for z_thresh in [1.5]:
            for wf_year in [2023, 2024]:
                name = f"PAIR_ONLY_LB{pair_lb}_ZT{z_thresh}_WF{wf_year}"
                configs.append({
                    'vol_low': 999.0, 'vol_high': 999.0,
                    'regime_src': 'vol',
                    'pair_lookback': pair_lb,
                    'pair_z_thresh': z_thresh,
                    'max_positions': 3, 'hold_max': 5,
                    'trail_atr_mult': 2.5,
                    'wf_split_year': wf_year,
                    'config_name': name,
                })

    print(f"  {len(configs)} configurations to test", flush=True)

    # ========================================================
    # RUN ALL CONFIGS
    # ========================================================
    print("\n[Backtest] Running...", flush=True)
    results = []

    for ci, cfg in enumerate(configs):
        r = run_backtest(**cfg)
        if r is not None:
            results.append(r)
            if r['ann'] > 10:
                # Type breakdown
                type_parts = []
                for tp, ts in sorted(r['type_stats'].items()):
                    twr = ts['w'] / max(ts['n'], 1) * 100
                    type_parts.append(f"{tp}:{ts['n']}({twr:.0f}%)")
                print(f"  {r['name']:50s} | Ann {r['ann']:+7.1f}% | WR {r['wr']:5.1f}% | "
                      f"N {r['n']:4d} | DD {r['dd']:6.1f}% | PF {r['pf']:4.2f} | "
                      f"Sh {r['sharpe']:5.2f} | {' | '.join(type_parts)}")

        if (ci + 1) % 50 == 0:
            print(f"  [{ci+1}/{len(configs)}] {len(results)} configs with results", flush=True)

    # ========================================================
    # RESULTS
    # ========================================================
    results.sort(key=lambda x: -x['ann'])

    wf_results = [r for r in results if '_WF' in r['name']]
    full_results = [r for r in results if '_WF' not in r['name']]

    # --- TOP 20 FULL-PERIOD ---
    print(f"\n{'=' * 140}")
    print(f"  TOP 20 FULL-PERIOD RESULTS (sorted by annual return)")
    print(f"{'=' * 140}")
    hdr = (f"  {'Config':50s} | {'Ann':>7s} | {'WR':>5s} | {'N':>4s} | {'DD':>6s} | "
           f"{'PF':>4s} | {'Sharpe':>6s} | {'AvgW':>7s} | {'AvgL':>6s} | {'AvgD':>4s}")
    print(hdr)
    print(f"  {'-' * 135}")
    for r in full_results[:20]:
        print(f"  {r['name']:50s} | {r['ann']:+7.1f}% | {r['wr']:5.1f}% | "
              f"{r['n']:4d} | {r['dd']:6.1f}% | {r['pf']:4.2f} | "
              f"{r['sharpe']:6.2f} | {r['avg_win']:+6.2f}% | {r['avg_loss']:5.2f}% | "
              f"{r['avg_days']:4.1f}")

    # --- TOP 10 WALK-FORWARD ---
    if wf_results:
        wf_results.sort(key=lambda x: -x['ann'])
        print(f"\n  TOP 10 WALK-FORWARD RESULTS (out-of-sample)")
        print(f"  {'-' * 135}")
        for r in wf_results[:10]:
            print(f"  {r['name']:50s} | Ann {r['ann']:+7.1f}% | WR {r['wr']:5.1f}% | "
                  f"N {r['n']:4d} | DD {r['dd']:6.1f}% | PF {r['pf']:4.2f} | "
                  f"Sh {r['sharpe']:6.2f}")

    # --- BEST CONFIG DETAIL ---
    if full_results:
        best = full_results[0]
        print(f"\n{'=' * 140}")
        print(f"  BEST CONFIG DETAIL: {best['name']}")
        print(f"  Ann={best['ann']:+.1f}%  WR={best['wr']:.1f}%  N={best['n']}  "
              f"DD={best['dd']:.1f}%  PF={best['pf']:.2f}  Sharpe={best['sharpe']:.2f}  "
              f"Final={best['cash']:.0f}")
        print(f"{'=' * 140}")

        # Trade type breakdown
        print(f"\n  TRADE TYPE BREAKDOWN:")
        for tp in sorted(best['type_stats'].keys()):
            ts = best['type_stats'][tp]
            twr = ts['w'] / max(ts['n'], 1) * 100
            print(f"    {tp:12s}: {ts['n']:4d} trades  WR={twr:5.1f}%  "
                  f"PnL={ts['pnl']:+.1f}%  Abs={ts['pnl_abs']:+.0f}")

        # Regime distribution
        print(f"\n  REGIME DISTRIBUTION (how often in each regime):")
        total_trades = sum(rs['n'] for rs in best['regime_stats'].values())
        for rg in sorted(best['regime_stats'].keys(), key=lambda x: -best['regime_stats'][x]['n']):
            rs = best['regime_stats'][rg]
            rwr = rs['w'] / max(rs['n'], 1) * 100
            pct = rs['n'] / max(total_trades, 1) * 100
            print(f"    {rg:12s}: {rs['n']:4d} trades ({pct:5.1f}%)  "
                  f"WR={rwr:5.1f}%  PnL={rs['pnl']:+.1f}%  Abs={rs['pnl_abs']:+.0f}")

        print(f"\n  YEARLY BREAKDOWN:")
        for y in sorted(best['yearly'].keys()):
            s = best['yearly'][y]
            wr_y = s['w'] / max(s['n'], 1) * 100
            print(f"    {y}: {s['n']:3d} trades  WR={wr_y:5.1f}%  "
                  f"PnL={s['pnl']:+.1f}%  Abs={s['pnl_abs']:+.0f}")

        print(f"\n  EXIT REASON BREAKDOWN:")
        for reason, s in sorted(best['reasons'].items(), key=lambda x: -x[1]['n']):
            rwr = s['w'] / max(s['n'], 1) * 100
            print(f"    {reason:12s}: {s['n']:4d} trades  WR={rwr:5.1f}%  "
                  f"PnL={s['pnl']:+.1f}%")

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
            print(f"    {rg:12s}: {ra['n']:5d} trades ({pct:5.1f}%)  "
                  f"WR={rwr:5.1f}%  Total Abs={ra['pnl_abs']:+12.0f}")

    # --- TYPE PROFITABILITY ACROSS TOP 20 ---
    if full_results:
        print(f"\n  TRADE TYPE PROFITABILITY ACROSS TOP 20 CONFIGS:")
        type_agg = {}
        for r in full_results[:20]:
            for tp, ts in r['type_stats'].items():
                if tp not in type_agg:
                    type_agg[tp] = {'n': 0, 'w': 0, 'pnl_abs': 0.0}
                type_agg[tp]['n'] += ts['n']
                type_agg[tp]['w'] += ts['w']
                type_agg[tp]['pnl_abs'] += ts['pnl_abs']

        for tp in sorted(type_agg.keys(), key=lambda x: -type_agg[x]['pnl_abs']):
            ta = type_agg[tp]
            twr = ta['w'] / max(ta['n'], 1) * 100
            print(f"    {tp:12s}: {ta['n']:5d} trades  WR={twr:5.1f}%  "
                  f"Total Abs={ta['pnl_abs']:+12.0f}")

    # --- YEARLY BREAKDOWN FOR TOP 5 ---
    if len(full_results) >= 2:
        print(f"\n  YEARLY BREAKDOWN FOR TOP 5 CONFIGS:")
        for idx, r in enumerate(full_results[:5]):
            print(f"\n  #{idx+1}: {r['name']} (Ann={r['ann']:+.1f}%, WR={r['wr']:.1f}%, "
                  f"DD={r['dd']:.1f}%, Sharpe={r['sharpe']:.2f})")
            for y in sorted(r['yearly'].keys()):
                ys = r['yearly'][y]
                wr_y = ys['w'] / ys['n'] * 100 if ys['n'] > 0 else 0
                print(f"    {y}: {ys['n']:3d}t  WR={wr_y:5.1f}%  "
                      f"PnL={ys['pnl']:+.1f}%  Abs={ys['pnl_abs']:+.0f}")

    # --- BASELINE COMPARISON ---
    mom_baselines = [r for r in full_results if r['name'].startswith('MOM_ONLY')]
    pair_baselines = [r for r in full_results if r['name'].startswith('PAIR_ONLY')]
    rotation_configs = [r for r in full_results
                        if not r['name'].startswith('MOM_ONLY')
                        and not r['name'].startswith('PAIR_ONLY')]

    print(f"\n  BASELINE COMPARISON:")
    if mom_baselines:
        best_mom = max(mom_baselines, key=lambda x: x['ann'])
        print(f"    Best MOM_ONLY:    {best_mom['name']:40s} Ann={best_mom['ann']:+.1f}%  "
              f"WR={best_mom['wr']:.1f}%  N={best_mom['n']}  DD={best_mom['dd']:.1f}%  "
              f"Sharpe={best_mom['sharpe']:.2f}")
    if pair_baselines:
        best_pair = max(pair_baselines, key=lambda x: x['ann'])
        print(f"    Best PAIR_ONLY:   {best_pair['name']:40s} Ann={best_pair['ann']:+.1f}%  "
              f"WR={best_pair['wr']:.1f}%  N={best_pair['n']}  DD={best_pair['dd']:.1f}%  "
              f"Sharpe={best_pair['sharpe']:.2f}")
    if rotation_configs:
        best_rot = max(rotation_configs, key=lambda x: x['ann'])
        print(f"    Best ROTATION:    {best_rot['name']:40s} Ann={best_rot['ann']:+.1f}%  "
              f"WR={best_rot['wr']:.1f}%  N={best_rot['n']}  DD={best_rot['dd']:.1f}%  "
              f"Sharpe={best_rot['sharpe']:.2f}")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.1f}s")
    print("=" * 130)


if __name__ == '__main__':
    main()
