"""
Alpha Futures V56 -- Hybrid Pair Trading with Momentum Confirmation
====================================================================
Core idea: Not all pair mean-reversion signals are equal. When the spread
deviation is in the SAME direction as the underlying commodity's momentum,
the mean-reversion is more reliable.

Specifically for a pair (downstream, upstream) where we short downstream
+ long upstream:
- If downstream has POSITIVE 5-day momentum (was going up) AND the spread
  is wide -> downstream was overbought relative to upstream -> stronger
  mean-reversion signal (momentum-confirmed).
- If downstream has NEGATIVE 5-day momentum AND the spread is wide ->
  downstream was already falling -> weaker signal (might not revert).

Confirmation types tested:
  1. none        = Pure V52 baseline (no confirmation filter)
  2. mom_confirm = Only enter when overvalued leg has positive momentum
                   (confirming it was indeed overextended)
  3. counter_mom = Only enter when overvalued leg has NEGATIVE momentum
                   (already starting to revert -- counter-momentum)
  4. oi          = Only enter when OI is rising on the pair
                   (institutional positioning confirms)
  5. vdp         = Only enter when VDP supports the direction
                   (buying pressure on the undervalued leg)

Parameter sweep:
  Confirmation: [none, mom_confirm, counter_mom, oi, vdp]
  Z threshold:  [0.8, 1.0, 1.2]
  Hold days:    [1, 2]
  Lookback:     10 (fixed, same as V52 best)
  Max pairs:    1 (fixed, same as V52 best)

~150 configs. Print: top 20, walk-forward, per-confirmation comparison.
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

CONFIRM_TYPES = ['none', 'mom_confirm', 'counter_mom', 'oi', 'vdp']


def main():
    t_start = time.time()
    print("=" * 140)
    print("Alpha Futures V56 -- Hybrid Pair Trading with Momentum Confirmation")
    print("Core: Pair mean-reversion filtered by momentum/OI/VDP confirmation")
    print("Hypothesis: Momentum-confirmed pair signals have higher WR and returns")
    print("=" * 140)

    # Load data with OI
    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)
    sym_to_si = {syms[si]: si for si in range(NS)}

    # Build pair index mapping
    pair_indices = []
    for down_sym, up_sym in PAIRS:
        down_si = sym_to_si.get(down_sym, -1)
        up_si = sym_to_si.get(up_sym, -1)
        if down_si >= 0 and up_si >= 0:
            pair_indices.append((down_si, up_si, down_sym, up_sym))
        else:
            print(f"  WARNING: pair ({down_sym}, {up_sym}) not found in data "
                  f"(down_si={down_si}, up_si={up_si})")

    print(f"  {NS} commodities, {ND} days, {len(pair_indices)} active pairs")

    # ========================================
    # PRECOMPUTE SPREADS
    # ========================================
    print("\n[Signals] Computing spreads...", flush=True)
    t0 = time.time()

    spreads = {}
    for down_si, up_si, down_sym, up_sym in pair_indices:
        spread = np.full(ND, np.nan)
        for di in range(ND):
            pd = C[down_si, di]
            pu = C[up_si, di]
            if not np.isnan(pd) and not np.isnan(pu):
                spread[di] = pd - pu
        spreads[(down_si, up_si)] = spread

    print(f"  Spreads computed ({time.time()-t0:.1f}s)", flush=True)

    # ========================================
    # PRECOMPUTE 5-DAY MOMENTUM (per commodity)
    # ========================================
    print("[Signals] Computing 5-day momentum...", flush=True)
    t0 = time.time()

    mom5 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            c_now = C[si, di]
            c_prev = C[si, di - 5]
            if not np.isnan(c_now) and not np.isnan(c_prev) and c_prev > 0:
                mom5[si, di] = (c_now - c_prev) / c_prev * 100

    print(f"  5-day momentum computed ({time.time()-t0:.1f}s)", flush=True)

    # ========================================
    # PRECOMPUTE VDP EMA (Volume Delta Pressure)
    # ========================================
    print("[Signals] Computing VDP EMA...", flush=True)
    t0 = time.time()

    vdp_ema = np.full((NS, ND), np.nan)
    for si in range(NS):
        vdp_e = 0.0
        alpha = 2.0 / 11  # 10-day EMA span
        for di in range(1, ND):
            d = di - 1  # use yesterday's data to avoid lookahead
            cd, hd, ld, vd = C[si, d], H[si, d], L[si, d], V[si, d]
            if np.isnan(cd) or np.isnan(hd) or np.isnan(ld) or np.isnan(vd):
                continue
            rng = hd - ld
            if rng <= 0:
                continue
            vdp_val = vd * (2 * cd - hd - ld) / rng
            vdp_e = alpha * vdp_val + (1 - alpha) * vdp_e
            vdp_ema[si, di] = vdp_e

    print(f"  VDP EMA computed ({time.time()-t0:.1f}s)", flush=True)

    # ========================================
    # PRECOMPUTE OI EMA TREND (per commodity)
    # ========================================
    print("[Signals] Computing OI EMA trend...", flush=True)
    t0 = time.time()

    oi_ema = np.full((NS, ND), np.nan)
    oi_trend = np.full((NS, ND), np.nan)  # 1 = rising, -1 = falling, nan = unknown
    for si in range(NS):
        oi_e = 0.0
        alpha_oi = 2.0 / 11  # 10-day EMA span
        initialized = False
        for di in range(ND):
            oi_val = OI[si, di]
            if np.isnan(oi_val):
                continue
            if not initialized:
                oi_e = oi_val
                initialized = True
            else:
                oi_e = alpha_oi * oi_val + (1 - alpha_oi) * oi_e
            oi_ema[si, di] = oi_e
            # Trend: compare current OI to its EMA
            if oi_val > oi_e * 1.0:
                oi_trend[si, di] = 1  # rising
            else:
                oi_trend[si, di] = -1  # falling

    print(f"  OI EMA trend computed ({time.time()-t0:.1f}s)", flush=True)

    # ========================================
    # BACKTEST ENGINE
    # ========================================
    LOOKBACK = 10  # Fixed at V52 best

    def run_backtest(z_thresh, hold_max, confirm_type,
                     wf_split_year=None, config_name=""):
        """
        Pair trading with optional momentum/OI/VDP confirmation.

        confirm_type:
          'none'        - No filter, pure V52-style pair mean-reversion
          'mom_confirm' - Only enter when overvalued leg has positive momentum
                          (confirming it was indeed overextended / overbought)
          'counter_mom' - Only enter when overvalued leg has NEGATIVE momentum
                          (already starting to revert)
          'oi'          - Only enter when OI is rising on at least one leg
                          (institutional positioning confirms the spread)
          'vdp'         - Only enter when VDP supports direction
                          (buying pressure on the undervalued leg,
                           selling pressure on the overvalued leg)
        """
        cash = float(CASH0)
        trades = []
        pair_positions = []

        # Pre-compute per-pair rolling z-scores (LB=10)
        pair_data = {}
        for down_si, up_si, down_sym, up_sym in pair_indices:
            sp = spreads[(down_si, up_si)]
            sp_mean = np.full(ND, np.nan)
            sp_std = np.full(ND, np.nan)
            z = np.full(ND, np.nan)

            for di in range(LOOKBACK, ND):
                window = sp[di - LOOKBACK:di]
                valid = window[~np.isnan(window)]
                if len(valid) >= LOOKBACK * 0.8:
                    sp_mean[di] = np.mean(valid)
                    sp_std[di] = np.std(valid, ddof=1)
                    if sp_std[di] > 1e-10:
                        z[di] = (sp[di] - sp_mean[di]) / sp_std[di]

            pair_data[(down_si, up_si)] = {
                'spread': sp,
                'mean': sp_mean,
                'std': sp_std,
                'z': z,
                'down_sym': down_sym,
                'up_sym': up_sym,
            }

        # Confirmation filter stats
        confirm_checked = 0
        confirm_passed = 0

        for di in range(MIN_TRAIN, ND):
            year = dates[di].year
            if wf_split_year is not None and year < wf_split_year:
                continue

            # --- Manage existing pair positions ---
            new_positions = []
            for pos in pair_positions:
                p_down_si = pos['down_si']
                p_up_si = pos['up_si']
                z_now = pair_data[(p_down_si, p_up_si)]['z'][di]
                days_held = di - pos['entry_di']
                entry_z = pos['entry_z']
                pos_dir = pos['dir']

                exit_reason = None

                # Exit 1: Z mean-reversion (z crosses 0)
                if not np.isnan(z_now):
                    if pos_dir == 1 and z_now >= 0:
                        exit_reason = 'mean_rev'
                    elif pos_dir == -1 and z_now <= 0:
                        exit_reason = 'mean_rev'

                # Exit 2: Stop loss -- z moves further by 1.5 from entry
                if exit_reason is None and not np.isnan(z_now):
                    if pos_dir == 1 and z_now < entry_z - 1.5:
                        exit_reason = 'stop_loss'
                    elif pos_dir == -1 and z_now > entry_z + 1.5:
                        exit_reason = 'stop_loss'

                # Exit 3: Time exit
                if exit_reason is None and days_held >= hold_max:
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
                        'pnl_abs': total_pnl,
                        'pnl_pct': pnl_pct,
                        'days': days_held,
                        'di': di,
                        'year': year,
                        'pair': (pos['down_sym'], pos['up_sym']),
                        'pair_label': PAIR_LABEL.get((pos['down_sym'], pos['up_sym']), ''),
                        'dir': pos_dir,
                        'reason': exit_reason,
                    })
                else:
                    new_positions.append(pos)

            pair_positions = new_positions

            # --- Check occupied commodities ---
            occupied = set()
            for pos in pair_positions:
                occupied.add(pos['down_si'])
                occupied.add(pos['up_si'])

            # --- Open new pair positions ---
            if len(pair_positions) >= 1:  # max_pairs = 1 (V52 best)
                continue

            candidates = []
            for down_si, up_si, down_sym, up_sym in pair_indices:
                if down_si in occupied or up_si in occupied:
                    continue
                pd = pair_data[(down_si, up_si)]
                z_val = pd['z'][di]
                if np.isnan(z_val):
                    continue
                if abs(z_val) < z_thresh:
                    continue

                # Determine which leg is overvalued
                # z > 0: spread is wide -> downstream is overvalued relative to upstream
                #         -> we SHORT downstream + LONG upstream
                # z < 0: spread is narrow -> downstream is undervalued relative to upstream
                #         -> we LONG downstream + SHORT upstream
                if z_val > 0:
                    overvalued_si = down_si
                    undervalued_si = up_si
                    overvalued_sym = down_sym
                    undervalued_sym = up_sym
                else:
                    overvalued_si = up_si
                    undervalued_si = down_si
                    overvalued_sym = up_sym
                    undervalued_sym = down_sym

                # Apply confirmation filter
                confirm_checked += 1
                signal_passed = True

                if confirm_type == 'mom_confirm':
                    # Overvalued leg should have POSITIVE momentum
                    # (was going up -> overbought -> stronger mean-reversion expected)
                    m = mom5[overvalued_si, di]
                    if np.isnan(m) or m <= 0:
                        signal_passed = False

                elif confirm_type == 'counter_mom':
                    # Overvalued leg should have NEGATIVE momentum
                    # (already starting to fall -> mean-reversion underway)
                    m = mom5[overvalued_si, di]
                    if np.isnan(m) or m >= 0:
                        signal_passed = False

                elif confirm_type == 'oi':
                    # OI should be rising on at least one leg
                    # (institutional positioning confirms the divergence)
                    ov_trend = oi_trend[overvalued_si, di]
                    uv_trend = oi_trend[undervalued_si, di]
                    if np.isnan(ov_trend) and np.isnan(uv_trend):
                        signal_passed = False
                    elif np.isnan(ov_trend) and uv_trend != 1:
                        signal_passed = False
                    elif np.isnan(uv_trend) and ov_trend != 1:
                        signal_passed = False
                    elif ov_trend != 1 and uv_trend != 1:
                        signal_passed = False

                elif confirm_type == 'vdp':
                    # VDP should support the direction:
                    # VDP positive on undervalued leg (buying pressure)
                    # OR VDP negative on overvalued leg (selling pressure)
                    vdp_uv = vdp_ema[undervalued_si, di]
                    vdp_ov = vdp_ema[overvalued_si, di]
                    if np.isnan(vdp_uv) and np.isnan(vdp_ov):
                        signal_passed = False
                    elif np.isnan(vdp_uv) and vdp_ov >= 0:
                        signal_passed = False
                    elif np.isnan(vdp_ov) and vdp_uv <= 0:
                        signal_passed = False
                    elif vdp_uv <= 0 and vdp_ov >= 0:
                        signal_passed = False

                if not signal_passed:
                    continue

                confirm_passed += 1
                candidates.append((abs(z_val), down_si, up_si, down_sym, up_sym, z_val))

            if not candidates:
                continue

            candidates.sort(key=lambda x: -x[0])

            # Open top candidate (max_pairs=1)
            _, down_si, up_si, down_sym, up_sym, z_val = candidates[0]

            c_down = C[down_si, di]
            c_up = C[up_si, di]
            if np.isnan(c_down) or c_down <= 0 or np.isnan(c_up) or c_up <= 0:
                continue

            mult_down = MULT.get(down_sym, DEF_MULT)
            mult_up = MULT.get(up_sym, DEF_MULT)

            cash_per_leg = cash * 0.95 / 2
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

            if z_val > 0:
                pos_dir = -1  # short down + long up
            else:
                pos_dir = 1   # long down + short up

            cash -= total_cost
            pair_positions.append({
                'down_si': down_si,
                'up_si': up_si,
                'down_sym': down_sym,
                'up_sym': up_sym,
                'entry_down': c_down,
                'entry_up': c_up,
                'lots_down': lots_down,
                'lots_up': lots_up,
                'entry_di': di,
                'entry_z': z_val,
                'dir': pos_dir,
                'cash_invested': total_cost,
            })

        # Close remaining positions at end
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
                'days': ND - 1 - pos['entry_di'],
                'di': ND - 1,
                'year': dates[ND - 1].year,
                'pair': (pos['down_sym'], pos['up_sym']),
                'pair_label': PAIR_LABEL.get((pos['down_sym'], pos['up_sym']), ''),
                'dir': pos['dir'],
                'reason': 'end',
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

        # Sharpe approximation from per-trade PnLs
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
                year_stats[y] = {'n': 0, 'w': 0, 'pnl': 0.0, 'pnl_abs_sum': 0.0}
            year_stats[y]['n'] += 1
            if t['pnl_abs'] > 0:
                year_stats[y]['w'] += 1
            year_stats[y]['pnl'] += t['pnl_pct']
            year_stats[y]['pnl_abs_sum'] += t['pnl_abs']

        # Per-pair breakdown
        pair_stats = {}
        for t in trades:
            p = t['pair_label']
            if p not in pair_stats:
                pair_stats[p] = {'n': 0, 'w': 0, 'pnl': 0.0}
            pair_stats[p]['n'] += 1
            if t['pnl_abs'] > 0:
                pair_stats[p]['w'] += 1
            pair_stats[p]['pnl'] += t['pnl_abs']

        pass_rate = confirm_passed / max(confirm_checked, 1) * 100

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
            'pair_stats': pair_stats,
            'trades': trades,
            'confirm': confirm_type,
            'pass_rate': round(pass_rate, 1),
        }

    # ========================================
    # PARAMETER SWEEP (~150 configs)
    # ========================================
    print("\n[Backtest] Running parameter sweep...", flush=True)
    results = []
    configs = []

    z_thresholds = [0.8, 1.0, 1.2]
    hold_days_list = [1, 2]
    confirm_types = CONFIRM_TYPES

    for ct in confirm_types:
        for zt in z_thresholds:
            for hd in hold_days_list:
                name = f"LB10_Z{zt:.1f}_H{hd}_{ct}"
                configs.append((zt, hd, ct, None, name))

    print(f"  {len(configs)} full-period configurations", flush=True)

    for ci, (zt, hd, ct, wf, name) in enumerate(configs):
        r = run_backtest(zt, hd, ct, wf_split_year=wf, config_name=name)
        if r is not None:
            results.append(r)
            if r['ann'] > 10:
                print(f"  {r['name']:40s} | Ann {r['ann']:+7.1f}% | WR {r['wr']:5.1f}% | "
                      f"N {r['n']:5d} | DD {r['dd']:6.1f}% | PF {r['pf']:4.2f} | "
                      f"Sh {r['sharpe']:5.2f} | AvgD {r['avg_days']:.1f} | "
                      f"Pass {r['pass_rate']:5.1f}%")
        if (ci + 1) % 10 == 0:
            print(f"  [{ci+1}/{len(configs)}] {len(results)} configs with results", flush=True)

    # Walk-forward for top configs
    print(f"\n[Walk-Forward] Testing top configs out-of-sample...", flush=True)
    full_results = [r for r in results]
    full_results.sort(key=lambda x: -x['ann'])

    # Select top configs for walk-forward: best per confirmation type
    wf_candidates = {}
    for r in full_results:
        ct = r['confirm']
        if ct not in wf_candidates:
            wf_candidates[ct] = []
        if len(wf_candidates[ct]) < 3:  # top 3 per confirmation type
            wf_candidates[ct].append(r)

    wf_results = []
    wf_configs_run = 0
    for ct, ct_results in wf_candidates.items():
        for r in ct_results:
            parts = r['name'].split('_')
            zt = float(parts[1][1:])
            hd = int(parts[2][1:])
            for wf_year in [2023, 2024]:
                name = f"{r['name']}_WF{wf_year}"
                wr = run_backtest(zt, hd, ct, wf_split_year=wf_year, config_name=name)
                if wr is not None:
                    wf_results.append(wr)
                wf_configs_run += 1

    print(f"  {wf_configs_run} walk-forward configs tested, {len(wf_results)} with results", flush=True)

    # ========================================
    # RESULTS
    # ========================================
    all_results = results + wf_results
    full_results = [r for r in all_results if '_WF' not in r['name']]
    wf_only = [r for r in all_results if '_WF' in r['name']]
    full_results.sort(key=lambda x: -x['ann'])
    wf_only.sort(key=lambda x: -x['ann'])

    # --- TOP 20 FULL-PERIOD ---
    print(f"\n{'=' * 150}")
    print(f"  TOP 20 FULL-PERIOD RESULTS (sorted by annual return)")
    print(f"{'=' * 150}")
    hdr = (f"  {'Config':40s} | {'Ann':>7s} | {'WR':>5s} | {'N':>5s} | {'DD':>6s} | "
           f"{'PF':>4s} | {'Sharpe':>6s} | {'AvgW':>7s} | {'AvgL':>6s} | {'AvgD':>4s} | "
           f"{'Pass%':>5s}")
    print(hdr)
    print(f"  {'-' * 145}")
    for r in full_results[:20]:
        print(f"  {r['name']:40s} | {r['ann']:+7.1f}% | {r['wr']:5.1f}% | "
              f"{r['n']:5d} | {r['dd']:6.1f}% | {r['pf']:4.2f} | "
              f"{r['sharpe']:6.2f} | {r['avg_win']:+6.2f}% | {r['avg_loss']:5.2f}% | "
              f"{r['avg_days']:4.1f} | {r['pass_rate']:5.1f}%")

    # --- TOP 10 WALK-FORWARD ---
    if wf_only:
        print(f"\n  TOP 10 WALK-FORWARD RESULTS (out-of-sample)")
        print(f"  {'-' * 145}")
        for r in wf_only[:10]:
            print(f"  {r['name']:50s} | Ann {r['ann']:+7.1f}% | WR {r['wr']:5.1f}% | "
                  f"N {r['n']:4d} | DD {r['dd']:6.1f}% | PF {r['pf']:4.2f} | "
                  f"Sh {r['sharpe']:6.2f} | AvgD {r['avg_days']:.1f}")

    # ========================================
    # PER-CONFIRMATION COMPARISON
    # ========================================
    print(f"\n{'=' * 150}")
    print(f"  PER-CONFIRMATION TYPE COMPARISON")
    print(f"{'=' * 150}")
    print(f"\n  {'Confirm':15s} | {'Avg Ann':>8s} | {'Best Ann':>9s} | {'Avg WR':>7s} | "
          f"{'Avg N':>6s} | {'Avg DD':>7s} | {'Avg PF':>6s} | {'Avg Sh':>6s} | "
          f"{'Avg Pass%':>9s} | {'#Cfgs':>5s} | Best Config")
    print(f"  {'-' * 140}")

    for ct in CONFIRM_TYPES:
        ct_results = [r for r in full_results if r.get('confirm') == ct]
        if ct_results:
            avg_ann = np.mean([r['ann'] for r in ct_results])
            best_ann = max(r['ann'] for r in ct_results)
            avg_wr = np.mean([r['wr'] for r in ct_results])
            avg_n = np.mean([r['n'] for r in ct_results])
            avg_dd = np.mean([r['dd'] for r in ct_results])
            avg_pf = np.mean([r['pf'] for r in ct_results])
            avg_sh = np.mean([r['sharpe'] for r in ct_results])
            avg_pass = np.mean([r['pass_rate'] for r in ct_results])
            best = max(ct_results, key=lambda x: x['ann'])
            print(f"  {ct:15s} | {avg_ann:+7.1f}% | {best_ann:+8.1f}% | {avg_wr:5.1f}% | "
                  f"{avg_n:6.0f} | {avg_dd:6.1f}% | {avg_pf:5.2f} | {avg_sh:5.2f} | "
                  f"{avg_pass:8.1f}% | {len(ct_results):5d} | {best['name']}")

    # ========================================
    # PER-CONFIRMATION DETAIL: BEST CONFIG
    # ========================================
    print(f"\n{'=' * 150}")
    print(f"  BEST CONFIG PER CONFIRMATION TYPE")
    print(f"{'=' * 150}")

    for ct in CONFIRM_TYPES:
        ct_results = [r for r in full_results if r.get('confirm') == ct]
        if not ct_results:
            continue
        best = max(ct_results, key=lambda x: x['ann'])
        print(f"\n  --- {ct.upper()} ---")
        print(f"  Config: {best['name']}")
        print(f"  Ann={best['ann']:+.1f}%  WR={best['wr']:.1f}%  N={best['n']}  "
              f"DD={best['dd']:.1f}%  PF={best['pf']:.2f}  Sharpe={best['sharpe']:.2f}  "
              f"Final={best['cash']:.0f}  PassRate={best['pass_rate']:.1f}%")

        print(f"  PER-PAIR BREAKDOWN:")
        for p in sorted(best['pair_stats'].keys(), key=lambda x: -best['pair_stats'][x]['n']):
            ps = best['pair_stats'][p]
            wr_p = ps['w'] / max(ps['n'], 1) * 100
            print(f"    {p:25s}: {ps['n']:4d} trades  WR={wr_p:5.1f}%  Abs PnL={ps['pnl']:+12.0f}")

        print(f"  YEARLY BREAKDOWN:")
        for y in sorted(best['yearly'].keys()):
            s = best['yearly'][y]
            wr_y = s['w'] / max(s['n'], 1) * 100
            print(f"    {y}: {s['n']:4d} trades  WR={wr_y:5.1f}%  PnL={s['pnl']:+.1f}%  "
                  f"Abs={s['pnl_abs_sum']:+.0f}")

        print(f"  EXIT REASON BREAKDOWN:")
        for reason, s in sorted(best['reasons'].items(), key=lambda x: -x[1]['n']):
            rwr = s['w'] / max(s['n'], 1) * 100
            print(f"    {reason:12s}: {s['n']:4d} trades  WR={rwr:5.1f}%  "
                  f"PnL={s['pnl_pct_sum']:+.1f}%  Abs={s['pnl']:+.0f}")

    # ========================================
    # WALK-FORWARD PER CONFIRMATION
    # ========================================
    if wf_only:
        print(f"\n{'=' * 150}")
        print(f"  WALK-FORWARD PER CONFIRMATION TYPE (best per type)")
        print(f"{'=' * 150}")

        for ct in CONFIRM_TYPES:
            ct_wf = [r for r in wf_only if r.get('confirm') == ct]
            if not ct_wf:
                continue
            ct_wf.sort(key=lambda x: -x['ann'])
            print(f"\n  --- {ct.upper()} Walk-Forward ---")
            for r in ct_wf[:4]:
                print(f"    {r['name']:55s} | Ann {r['ann']:+7.1f}% | WR {r['wr']:5.1f}% | "
                      f"N {r['n']:4d} | DD {r['dd']:6.1f}% | PF {r['pf']:4.2f} | "
                      f"Sh {r['sharpe']:6.2f}")

    # ========================================
    # YEARLY FOR TOP 5
    # ========================================
    if len(full_results) >= 2:
        print(f"\n  YEARLY BREAKDOWN FOR TOP 5 CONFIGS:")
        for idx, r in enumerate(full_results[:5]):
            print(f"\n  #{idx+1}: {r['name']} (Ann={r['ann']:+.1f}%, WR={r['wr']:.1f}%, "
                  f"DD={r['dd']:.1f}%, Sharpe={r['sharpe']:.2f}, N={r['n']})")
            for y in sorted(r['yearly'].keys()):
                ys = r['yearly'][y]
                wr_y = ys['w'] / ys['n'] * 100 if ys['n'] > 0 else 0
                print(f"    {y}: {ys['n']:4d}t  WR={wr_y:5.1f}%  PnL={ys['pnl']:+.1f}%")

    # ========================================
    # PAIR PROFITABILITY ACROSS TOP 20
    # ========================================
    if full_results:
        print(f"\n  PAIR PROFITABILITY ACROSS TOP 20 CONFIGS:")
        pair_summary = {}
        for r in full_results[:20]:
            for p, ps in r['pair_stats'].items():
                if p not in pair_summary:
                    pair_summary[p] = {'n': 0, 'w': 0, 'pnl': 0.0}
                pair_summary[p]['n'] += ps['n']
                pair_summary[p]['w'] += ps['w']
                pair_summary[p]['pnl'] += ps['pnl']

        for p in sorted(pair_summary.keys(), key=lambda x: -pair_summary[x]['pnl']):
            ps = pair_summary[p]
            wr_p = ps['w'] / max(ps['n'], 1) * 100
            print(f"    {p:25s}: {ps['n']:5d} trades  WR={wr_p:5.1f}%  Total Abs={ps['pnl']:+12.0f}")

    # ========================================
    # MOMENTUM CONFIRMATION vs BASELINE ANALYSIS
    # ========================================
    print(f"\n{'=' * 150}")
    print(f"  MOMENTUM CONFIRMATION vs BASELINE ANALYSIS")
    print(f"{'=' * 150}")

    none_results = [r for r in full_results if r.get('confirm') == 'none']
    mom_results = [r for r in full_results if r.get('confirm') == 'mom_confirm']
    counter_results = [r for r in full_results if r.get('confirm') == 'counter_mom']
    oi_results = [r for r in full_results if r.get('confirm') == 'oi']
    vdp_results = [r for r in full_results if r.get('confirm') == 'vdp']

    print(f"\n  Comparison: Does confirmation filter improve pair trading?")
    print(f"\n  {'Z':>4s} {'H':>2s} | {'None':>10s} | {'MomConf':>10s} | "
          f"{'Counter':>10s} | {'OI':>10s} | {'VDP':>10s} | {'Best':>15s}")
    print(f"  {'-' * 80}")

    for zt in z_thresholds:
        for hd in hold_days_list:
            tag = f"Z{zt:.1f}_H{hd}"
            vals = {}
            for ct, ct_list in [('none', none_results), ('mom_confirm', mom_results),
                                ('counter_mom', counter_results), ('oi', oi_results),
                                ('vdp', vdp_results)]:
                match = [r for r in ct_list if f"_Z{zt:.1f}_" in r['name'] and f"_H{hd}_" in r['name']]
                if match:
                    vals[ct] = max(r['ann'] for r in match)
                else:
                    vals[ct] = None

            none_v = f"{vals['none']:+.1f}%" if vals.get('none') is not None else "N/A"
            mom_v = f"{vals['mom_confirm']:+.1f}%" if vals.get('mom_confirm') is not None else "N/A"
            cnt_v = f"{vals['counter_mom']:+.1f}%" if vals.get('counter_mom') is not None else "N/A"
            oi_v = f"{vals['oi']:+.1f}%" if vals.get('oi') is not None else "N/A"
            vdp_v = f"{vals['vdp']:+.1f}%" if vals.get('vdp') is not None else "N/A"

            all_vals = {k: v for k, v in vals.items() if v is not None}
            if all_vals:
                best_ct = max(all_vals, key=all_vals.get)
                best_v = f"{best_ct}={all_vals[best_ct]:+.1f}%"
            else:
                best_v = "N/A"

            print(f"  {zt:.1f}  {hd:d}  | {none_v:>10s} | {mom_v:>10s} | "
                  f"{cnt_v:>10s} | {oi_v:>10s} | {vdp_v:>10s} | {best_v:>15s}")

    # ========================================
    # WIN RATE COMPARISON
    # ========================================
    print(f"\n  WIN RATE COMPARISON BY CONFIRMATION TYPE:")
    print(f"  {'-' * 80}")
    print(f"  {'Z':>4s} {'H':>2s} | {'None WR':>8s} | {'MomConf':>8s} | "
          f"{'Counter':>8s} | {'OI':>8s} | {'VDP':>8s}")
    print(f"  {'-' * 60}")

    for zt in z_thresholds:
        for hd in hold_days_list:
            vals = {}
            for ct, ct_list in [('none', none_results), ('mom_confirm', mom_results),
                                ('counter_mom', counter_results), ('oi', oi_results),
                                ('vdp', vdp_results)]:
                match = [r for r in ct_list if f"_Z{zt:.1f}_" in r['name'] and f"_H{hd}_" in r['name']]
                if match:
                    vals[ct] = max(r['wr'] for r in match)
                else:
                    vals[ct] = None

            none_v = f"{vals['none']:.1f}%" if vals.get('none') is not None else "N/A"
            mom_v = f"{vals['mom_confirm']:.1f}%" if vals.get('mom_confirm') is not None else "N/A"
            cnt_v = f"{vals['counter_mom']:.1f}%" if vals.get('counter_mom') is not None else "N/A"
            oi_v = f"{vals['oi']:.1f}%" if vals.get('oi') is not None else "N/A"
            vdp_v = f"{vals['vdp']:.1f}%" if vals.get('vdp') is not None else "N/A"

            print(f"  {zt:.1f}  {hd:d}  | {none_v:>8s} | {mom_v:>8s} | "
                  f"{cnt_v:>8s} | {oi_v:>8s} | {vdp_v:>8s}")

    # ========================================
    # TRADE COUNT ANALYSIS
    # ========================================
    print(f"\n  TRADE COUNT ANALYSIS BY CONFIRMATION TYPE:")
    print(f"  {'-' * 80}")
    print(f"  {'Confirm':15s} | {'Avg N':>6s} | {'Min N':>6s} | {'Max N':>6s} | "
          f"{'Avg WR':>7s} | {'Avg Ann':>8s} | {'Avg Pass%':>9s}")
    print(f"  {'-' * 80}")

    for ct in CONFIRM_TYPES:
        ct_results_list = [r for r in full_results if r.get('confirm') == ct]
        if ct_results_list:
            avg_n = np.mean([r['n'] for r in ct_results_list])
            min_n = min(r['n'] for r in ct_results_list)
            max_n = max(r['n'] for r in ct_results_list)
            avg_wr = np.mean([r['wr'] for r in ct_results_list])
            avg_ann = np.mean([r['ann'] for r in ct_results_list])
            avg_pass = np.mean([r['pass_rate'] for r in ct_results_list])
            print(f"  {ct:15s} | {avg_n:6.0f} | {min_n:6d} | {max_n:6d} | "
                  f"{avg_wr:5.1f}% | {avg_ann:+7.1f}% | {avg_pass:8.1f}%")

    # ========================================
    # V52 BASELINE COMPARISON
    # ========================================
    print(f"\n  === V56 vs V52 BASELINE COMPARISON ===")
    print(f"  V52 best (MP1, LB10): +303.5%")
    if full_results:
        print(f"  V56 best: {full_results[0]['name']}")
        print(f"    Ann={full_results[0]['ann']:+.1f}%  N={full_results[0]['n']}  "
              f"WR={full_results[0]['wr']:.1f}%  DD={full_results[0]['dd']:.1f}%  "
              f"Sharpe={full_results[0]['sharpe']:.2f}")
        delta = full_results[0]['ann'] - 303.5
        print(f"    Delta vs V52: {delta:+.1f}%")

        # Per-confirmation: how many beat V52?
        print(f"\n  Configs beating V52 (+303.5%) per confirmation type:")
        for ct in CONFIRM_TYPES:
            ct_results_list = [r for r in full_results if r.get('confirm') == ct]
            beating = sum(1 for r in ct_results_list if r['ann'] > 303.5)
            print(f"    {ct:15s}: {beating}/{len(ct_results_list)} configs beat V52")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.1f}s")
    print("=" * 140)


if __name__ == '__main__':
    main()
