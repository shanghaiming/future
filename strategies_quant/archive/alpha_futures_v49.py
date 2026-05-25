"""
Alpha Futures V49 — Deep Seasonal Optimization + V39 Pair Trading Combination
==============================================================================
Context: V44 seasonal showed +43.2% full period but +106.7% walk-forward (WF2024).
         V39 pair trading was best at +188.1%.

Part 1: Optimize V44 seasonal parameters
  - Expanded windows: [7, 10, 15, 20]
  - Seasonal Sharpe filter (new)
  - Seasonal + Momentum alignment (multi-horizon)
  - Anti-seasonal reversal with finer thresholds
  - Seasonal + OI with expanded params

Part 2: Combine best seasonal with V39 pair trading
  - Run both simultaneously with capital allocation X% pairs / (1-X)% seasonal
  - Test X = [0.3, 0.5, 0.7, 0.8]

Configs: ~200 total
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


def main():
    t_start = time.time()
    print("=" * 130)
    print("Alpha Futures V49 — Deep Seasonal Optimization + V39 Pair Trading Combination")
    print("Part 1: Optimize seasonal (4 signal types, expanded params)")
    print("Part 2: Combine best seasonal with V39 pairs (capital allocation sweep)")
    print("=" * 130)

    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)
    sym_to_si = {syms[si]: si for si in range(NS)}

    print(f"  {NS} commodities, {ND} days")

    # ========================================
    # PRECOMPUTE BASIC SIGNALS
    # ========================================
    print("\n[Signals] Computing returns, momentum, OI, ATR...", flush=True)
    t0 = time.time()

    # Daily returns
    ret = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(1, ND):
            c_now = C[si, di]
            c_prev = C[si, di - 1]
            if not np.isnan(c_now) and not np.isnan(c_prev) and c_prev > 0:
                ret[si, di] = (c_now - c_prev) / c_prev

    # Day-of-year and year arrays
    doy_arr = np.zeros(ND, dtype=np.int32)
    year_arr = np.zeros(ND, dtype=np.int32)
    for di in range(ND):
        doy_arr[di] = dates[di].timetuple().tm_yday
        year_arr[di] = dates[di].year

    # Multi-horizon momentum
    mom3 = np.full((NS, ND), np.nan)
    mom5 = np.full((NS, ND), np.nan)
    mom10 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(1, ND):
            for m_arr, m_period in [(mom3, 3), (mom5, 5), (mom10, 10)]:
                if di >= m_period:
                    c_now = C[si, di]
                    c_prev = C[si, di - m_period]
                    if not np.isnan(c_now) and not np.isnan(c_prev) and c_prev > 0:
                        m_arr[si, di] = (c_now - c_prev) / c_prev

    # OI rising: 5-day EMA trend
    oi_ema = np.full((NS, ND), np.nan)
    oi_rising = np.full((NS, ND), False)
    for si in range(NS):
        oe = 0.0
        alpha_oi = 2.0 / 6
        for di in range(1, ND):
            oi_val = OI[si, di]
            if np.isnan(oi_val):
                continue
            oe = alpha_oi * oi_val + (1 - alpha_oi) * oe
            oi_ema[si, di] = oe
        for di in range(6, ND):
            cur = oi_ema[si, di]
            prev = oi_ema[si, di - 5]
            if not np.isnan(cur) and not np.isnan(prev) and prev > 0:
                oi_rising[si, di] = (cur - prev) / prev > 0.01

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

    print(f"  Basic signals done ({time.time() - t0:.1f}s)", flush=True)

    # ========================================
    # PRECOMPUTE SEASONAL STATISTICS
    # ========================================
    print("\n[Signals] Computing seasonal statistics...", flush=True)
    t1 = time.time()

    # Collect returns by (si, doy) across all years
    ret_by_si_doy = [[[] for _ in range(367)] for _ in range(NS)]
    for si in range(NS):
        for di in range(1, ND):
            r = ret[si, di]
            if not np.isnan(r):
                ret_by_si_doy[si][doy_arr[di]].append((year_arr[di], r))

    def compute_seasonal_stats(window_days):
        """
        Returns arrays:
          seasonal_ret[si, di] = mean return on this DOY (+-window) from prior years
          seasonal_hit[si, di] = fraction of positive returns
          seasonal_std[si, di] = std of returns (for Sharpe)
          seasonal_n[si, di]   = number of prior-year observations
        """
        seasonal_ret = np.full((NS, ND), np.nan)
        seasonal_hit = np.full((NS, ND), np.nan)
        seasonal_std = np.full((NS, ND), np.nan)
        seasonal_n = np.full((NS, ND), 0, dtype=np.int32)

        for si in range(NS):
            for di in range(1, ND):
                y = year_arr[di]
                d = doy_arr[di]

                prior_rets = []
                for wd in range(d - window_days, d + window_days + 1):
                    wd_adj = wd
                    if wd_adj < 1:
                        wd_adj += 365
                    elif wd_adj > 365:
                        wd_adj -= 365
                    if wd_adj < 1 or wd_adj > 366:
                        continue
                    for yr, r in ret_by_si_doy[si][wd_adj]:
                        if yr < y:
                            prior_rets.append(r)

                if len(prior_rets) >= 5:
                    arr = np.array(prior_rets)
                    seasonal_ret[si, di] = np.mean(arr)
                    seasonal_hit[si, di] = np.sum(arr > 0) / len(arr)
                    seasonal_std[si, di] = np.std(arr, ddof=1) if len(arr) > 1 else 0
                    seasonal_n[si, di] = len(arr)

        return seasonal_ret, seasonal_hit, seasonal_std, seasonal_n

    seasonal_cache = {}
    for wd in [7, 10, 15, 20]:
        print(f"  Computing seasonal stats for window={wd}...", flush=True)
        sr, sh, ss, sn = compute_seasonal_stats(wd)
        seasonal_cache[wd] = (sr, sh, ss, sn)
        print(f"    window={wd} done ({time.time() - t1:.1f}s)", flush=True)

    print(f"  All seasonal stats computed ({time.time() - t0:.1f}s)", flush=True)

    # ========================================
    # SEASONAL SIGNAL SCORING FUNCTIONS
    # ========================================

    def make_seasonal_score(signal_type, params, window):
        """
        signal_type: 'A' (anti-seasonal reversal), 'B' (seasonal+OI),
                     'C' (seasonal Sharpe), 'D' (seasonal+momentum alignment)
        params: dict of signal-specific params
        window: DOY window size
        """
        sr, sh, ss, sn = seasonal_cache[window]

        def score(si, di):
            s_ret = sr[si, di]
            s_hit = sh[si, di]
            s_std = ss[si, di]
            s_n = sn[si, di]

            if s_n < 20:
                return np.nan
            if np.isnan(s_ret) or np.isnan(s_hit):
                return np.nan

            # Seasonal Sharpe
            s_sharpe = s_ret / s_std if s_std > 1e-10 else 0

            if signal_type == 'A':
                # Anti-seasonal reversal: bullish season but price dipped -> buy the dip
                threshold = params.get('threshold', 0.02)
                hit_min = params.get('hit_min', 0.55)
                if s_ret > 0.001 and s_hit > hit_min:
                    m = mom5[si, di]
                    if np.isnan(m):
                        return np.nan
                    if m < -threshold:
                        dip_strength = min(abs(m) / 0.1, 2.0)
                        return s_ret * s_hit * 100 * (1 + dip_strength)
                return np.nan

            elif signal_type == 'B':
                # Seasonal + OI confirmation
                threshold = params.get('threshold', 0.003)
                hit_min = params.get('hit_min', 0.55)
                if s_ret > threshold and s_hit > hit_min:
                    if oi_rising[si, di]:
                        return s_ret * s_hit * 100 * 1.5
                return np.nan

            elif signal_type == 'C':
                # Seasonal Sharpe filter: only trade when pattern is strong AND consistent
                sharpe_min = params.get('sharpe_min', 0.5)
                hit_min = params.get('hit_min', 0.55)
                if s_sharpe > sharpe_min and s_hit > hit_min:
                    return s_sharpe * s_hit * 100
                return np.nan

            elif signal_type == 'D':
                # Seasonal + multi-horizon momentum alignment
                threshold = params.get('threshold', 0.001)
                hit_min = params.get('hit_min', 0.55)
                if s_ret > threshold and s_hit > hit_min:
                    m3 = mom3[si, di]
                    m10_val = mom10[si, di]
                    if np.isnan(m3) or np.isnan(m10_val):
                        return np.nan
                    if m3 > 0 and m10_val > 0:
                        # Both momentum horizons aligned with seasonal
                        alignment = 1 + m3 * 5 + m10_val * 5
                        return s_ret * s_hit * 100 * alignment
                return np.nan

            return np.nan

        return score

    # ========================================
    # BACKTEST ENGINE — SEASONAL ONLY
    # ========================================
    def run_seasonal_backtest(score_fn, name, top_n=1, hold_max=5,
                              trail_atr_mult=3.0, wf_split_year=None):
        """Single position per symbol, long only. Exit on time or trailing stop."""
        cash = float(CASH0)
        trades = []
        positions = []

        for di in range(MIN_TRAIN, ND):
            year = dates[di].year
            if wf_split_year is not None and year < wf_split_year:
                continue

            # Manage existing positions
            new_positions = []
            for pos in positions:
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

                # Time exit
                if exit_reason is None and days_held >= hold_max:
                    exit_reason = 'time'

                if exit_reason:
                    cost_out = mkt_val * COMM
                    cash += mkt_val - cost_out
                    trades.append({
                        'pnl_pct': pnl_pct, 'pnl_abs': pnl,
                        'days': days_held, 'di': di, 'year': year,
                        'sym': pos['sym'], 'dir': pos['dir'], 'reason': exit_reason,
                    })
                else:
                    new_positions.append(pos)

            positions = new_positions

            # Open new positions
            n_open = len(positions)
            if n_open < top_n:
                slots = top_n - n_open
                scored = []
                for si in range(NS):
                    sc = score_fn(si, di)
                    if np.isnan(sc) or sc <= 0.01:
                        continue
                    sym = syms[si]
                    if any(p['sym'] == sym for p in positions):
                        continue
                    scored.append((si, sc, sym))

                if scored:
                    scored.sort(key=lambda x: -x[1])
                    cash_per_slot = cash / slots if slots > 0 else cash

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
                        trail_price = c - trail_atr_mult * atr_val if atr_val > 0 else c * 0.97
                        positions.append({
                            'si': best_si, 'entry': c, 'entry_di': di,
                            'lots': lots, 'dir': 1, 'sym': best_sym,
                            'atr': atr_val, 'trail_price': trail_price,
                        })

        # Close remaining
        for pos in positions:
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
                'sym': pos['sym'], 'dir': pos['dir'], 'reason': 'end',
            })

        if len(trades) < 10:
            return None

        # Stats
        equity = float(CASH0)
        peak = float(CASH0)
        max_dd = 0
        for t in sorted(trades, key=lambda x: x['di']):
            equity += t['pnl_abs']
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
            for di in range(MIN_TRAIN, ND):
                if dates[di].year >= wf_split_year:
                    first_test_di = di
                    break
            if first_test_di:
                days_total = (dates[ND - 1] - dates[first_test_di]).days
                yr = max(days_total / 365.25, 0.01)

        ann = ((cash / CASH0) ** (1 / yr) - 1) * 100

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
            rets_arr = np.array(trade_pnls) / float(CASH0)
            mean_ret = np.mean(rets_arr)
            std_ret = np.std(rets_arr)
            sharpe_approx = mean_ret / std_ret * np.sqrt(252) if std_ret > 0 else 0
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

        reasons = {}
        for t in trades:
            r = t['reason']
            if r not in reasons:
                reasons[r] = {'n': 0, 'w': 0, 'pnl': 0.0}
            reasons[r]['n'] += 1
            if t['pnl_abs'] > 0:
                reasons[r]['w'] += 1
            reasons[r]['pnl'] += t['pnl_pct']

        return {
            'name': name, 'ann': round(ann, 1), 'n': len(trades),
            'wr': round(wr, 1), 'dd': round(max_dd, 1),
            'avg_win': round(avg_win, 2), 'avg_loss': round(avg_loss, 2),
            'avg_days': round(avg_days, 1), 'pf': round(pf, 2),
            'sharpe': round(sharpe_approx, 2), 'cash': round(cash, 0),
            'reasons': reasons, 'yearly': year_stats,
        }

    # ========================================
    # PRECOMPUTE PAIR SPREADS (for Part 2)
    # ========================================
    print("\n[Signals] Computing pair spreads for V39 combination...", flush=True)
    t_pair = time.time()

    pair_indices = []
    for down_sym, up_sym in PAIRS:
        down_si = sym_to_si.get(down_sym, -1)
        up_si = sym_to_si.get(up_sym, -1)
        if down_si >= 0 and up_si >= 0:
            pair_indices.append((down_si, up_si, down_sym, up_sym))

    spreads = {}
    for down_si, up_si, down_sym, up_sym in pair_indices:
        spread = np.full(ND, np.nan)
        for di in range(ND):
            pd = C[down_si, di]
            pu = C[up_si, di]
            if not np.isnan(pd) and not np.isnan(pu):
                spread[di] = pd - pu
        spreads[(down_si, up_si)] = spread

    print(f"  Spreads computed for {len(pair_indices)} pairs ({time.time() - t_pair:.1f}s)", flush=True)

    # ========================================
    # COMBINED BACKTEST ENGINE (Seasonal + Pairs)
    # ========================================
    def run_combined_backtest(seasonal_score_fn, pair_lookback, pair_z_thresh,
                              pair_hold_max, pair_max_pairs, capital_pairs_pct,
                              name, seasonal_top_n=1, seasonal_hold=5,
                              wf_split_year=None):
        """
        Run seasonal directional + pair trading simultaneously.
        capital_pairs_pct: fraction of initial capital allocated to pairs (rest to seasonal)
        """
        cash_pairs = float(CASH0) * capital_pairs_pct
        cash_seasonal = float(CASH0) * (1 - capital_pairs_pct)
        trades = []
        pair_positions = []
        seasonal_positions = []

        # Pre-compute per-pair z-scores
        pair_data = {}
        for down_si, up_si, down_sym, up_sym in pair_indices:
            sp = spreads[(down_si, up_si)]
            sp_mean = np.full(ND, np.nan)
            sp_std = np.full(ND, np.nan)
            z = np.full(ND, np.nan)
            for di in range(pair_lookback, ND):
                window = sp[di - pair_lookback:di]
                valid = window[~np.isnan(window)]
                if len(valid) >= pair_lookback * 0.8:
                    sp_mean[di] = np.mean(valid)
                    sp_std[di] = np.std(valid, ddof=1)
                    if sp_std[di] > 1e-10:
                        z[di] = (sp[di] - sp_mean[di]) / sp_std[di]
            pair_data[(down_si, up_si)] = {
                'spread': sp, 'mean': sp_mean, 'std': sp_std, 'z': z,
                'down_sym': down_sym, 'up_sym': up_sym,
            }

        for di in range(MIN_TRAIN, ND):
            year = dates[di].year
            if wf_split_year is not None and year < wf_split_year:
                continue

            # ---- MANAGE PAIR POSITIONS ----
            new_pair_pos = []
            for pos in pair_positions:
                p_down_si = pos['down_si']
                p_up_si = pos['up_si']
                z_now = pair_data[(p_down_si, p_up_si)]['z'][di]
                days_held = di - pos['entry_di']
                entry_z = pos['entry_z']
                pos_dir = pos['dir']

                exit_reason = None
                if not np.isnan(z_now):
                    if pos_dir == 1 and z_now <= 0:
                        exit_reason = 'mean_rev'
                    elif pos_dir == -1 and z_now >= 0:
                        exit_reason = 'mean_rev'
                if exit_reason is None and not np.isnan(z_now):
                    if pos_dir == 1 and z_now < entry_z - 1.0:
                        exit_reason = 'stop_loss'
                    elif pos_dir == -1 and z_now > entry_z + 1.0:
                        exit_reason = 'stop_loss'
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

                    cash_pairs += pos['cash_invested'] + cash_return - (exit_val_down + exit_val_up) * COMM

                    trades.append({
                        'pnl_abs': total_pnl, 'pnl_pct': pnl_pct,
                        'days': days_held, 'di': di, 'year': year,
                        'pair': (pos['down_sym'], pos['up_sym']),
                        'type': 'pair', 'dir': pos_dir, 'reason': exit_reason,
                    })
                else:
                    new_pair_pos.append(pos)

            pair_positions = new_pair_pos

            # Open new pair positions
            occupied = set()
            for pos in pair_positions:
                occupied.add(pos['down_si'])
                occupied.add(pos['up_si'])

            n_can_open = pair_max_pairs - len(pair_positions)
            if n_can_open > 0:
                candidates = []
                for down_si, up_si, down_sym, up_sym in pair_indices:
                    if down_si in occupied or up_si in occupied:
                        continue
                    pd = pair_data[(down_si, up_si)]
                    z_val = pd['z'][di]
                    if np.isnan(z_val):
                        continue
                    if abs(z_val) < pair_z_thresh:
                        continue
                    candidates.append((abs(z_val), down_si, up_si, down_sym, up_sym, z_val))

                if candidates:
                    candidates.sort(key=lambda x: -x[0])
                    for _, down_si, up_si, down_sym, up_sym, z_val in candidates[:n_can_open]:
                        c_down = C[down_si, di]
                        c_up = C[up_si, di]
                        if np.isnan(c_down) or c_down <= 0 or np.isnan(c_up) or c_up <= 0:
                            continue

                        mult_down = MULT.get(down_sym, DEF_MULT)
                        mult_up = MULT.get(up_sym, DEF_MULT)
                        cash_per_leg = cash_pairs / 2
                        lots_down = int(cash_per_leg / (c_down * mult_down * (1 + COMM)))
                        lots_up = int(cash_per_leg / (c_up * mult_up * (1 + COMM)))
                        if lots_down <= 0 or lots_up <= 0:
                            continue

                        cost_down = c_down * mult_down * lots_down * (1 + COMM)
                        cost_up = c_up * mult_up * lots_up * (1 + COMM)
                        total_cost = cost_down + cost_up
                        if total_cost > cash_pairs:
                            scale = cash_pairs * 0.95 / total_cost
                            lots_down = max(1, int(lots_down * scale))
                            lots_up = max(1, int(lots_up * scale))
                            cost_down = c_down * mult_down * lots_down * (1 + COMM)
                            cost_up = c_up * mult_up * lots_up * (1 + COMM)
                            total_cost = cost_down + cost_up
                            if total_cost > cash_pairs:
                                continue

                        pos_dir = -1 if z_val > 0 else 1
                        cash_pairs -= total_cost
                        pair_positions.append({
                            'down_si': down_si, 'up_si': up_si,
                            'down_sym': down_sym, 'up_sym': up_sym,
                            'entry_down': c_down, 'entry_up': c_up,
                            'lots_down': lots_down, 'lots_up': lots_up,
                            'entry_di': di, 'entry_z': z_val,
                            'dir': pos_dir, 'cash_invested': total_cost,
                        })

            # ---- MANAGE SEASONAL POSITIONS ----
            new_seasonal_pos = []
            for pos in seasonal_positions:
                c = C[pos['si'], di]
                if np.isnan(c) or c <= 0:
                    c = pos['entry']
                mult = MULT.get(pos['sym'], DEF_MULT)
                mkt_val = c * mult * pos['lots']
                pnl = (c - pos['entry']) * mult * pos['lots']
                pnl_pct = pnl / (pos['entry'] * mult * pos['lots']) * 100 if pos['entry'] > 0 else 0
                days_held = di - pos['entry_di']

                exit_reason = None
                if days_held >= 2:
                    atr = pos.get('atr', 0)
                    if atr > 0:
                        new_trail = c - 3.0 * atr
                        if new_trail > pos.get('trail_price', pos['entry']):
                            pos['trail_price'] = new_trail
                        if c < pos['trail_price']:
                            exit_reason = 'trail'

                if exit_reason is None and days_held >= seasonal_hold:
                    exit_reason = 'time'

                if exit_reason:
                    cost_out = mkt_val * COMM
                    cash_seasonal += mkt_val - cost_out
                    trades.append({
                        'pnl_abs': pnl, 'pnl_pct': pnl_pct,
                        'days': days_held, 'di': di, 'year': year,
                        'sym': pos['sym'], 'type': 'seasonal',
                        'dir': 1, 'reason': exit_reason,
                    })
                else:
                    new_seasonal_pos.append(pos)

            seasonal_positions = new_seasonal_pos

            # Open new seasonal positions
            n_seas_open = seasonal_top_n - len(seasonal_positions)
            if n_seas_open > 0:
                scored = []
                for si in range(NS):
                    sc = seasonal_score_fn(si, di)
                    if np.isnan(sc) or sc <= 0.01:
                        continue
                    sym = syms[si]
                    if any(p['sym'] == sym for p in seasonal_positions):
                        continue
                    scored.append((si, sc, sym))

                if scored:
                    scored.sort(key=lambda x: -x[1])
                    cash_per_slot = cash_seasonal / n_seas_open if n_seas_open > 0 else cash_seasonal
                    for best_si, best_sc, best_sym in scored[:n_seas_open]:
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
                        if cost_in > cash_seasonal:
                            lots = int(cash_seasonal / (notional * (1 + COMM)))
                            if lots <= 0:
                                continue
                            cost_in = notional * lots * (1 + COMM)

                        atr_val = atr10[best_si, di] if not np.isnan(atr10[best_si, di]) else 0
                        cash_seasonal -= cost_in
                        trail_price = c - 3.0 * atr_val if atr_val > 0 else c * 0.97
                        seasonal_positions.append({
                            'si': best_si, 'entry': c, 'entry_di': di,
                            'lots': lots, 'dir': 1, 'sym': best_sym,
                            'atr': atr_val, 'trail_price': trail_price,
                        })

        # Close all remaining positions
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

            if pos['dir'] == 1:
                cash_return = c_down * mult_down * lots_down - c_up * mult_up * lots_up
            else:
                cash_return = -c_down * mult_down * lots_down + c_up * mult_up * lots_up
            cash_pairs += pos['cash_invested'] + cash_return - (exit_val_down + exit_val_up) * COMM

            trades.append({
                'pnl_abs': total_pnl,
                'pnl_pct': total_pnl / (entry_val_down + entry_val_up) * 100 if (entry_val_down + entry_val_up) > 0 else 0,
                'days': ND - 1 - pos['entry_di'],
                'di': ND - 1, 'year': dates[ND - 1].year,
                'pair': (pos['down_sym'], pos['up_sym']),
                'type': 'pair', 'dir': pos['dir'], 'reason': 'end',
            })

        for pos in seasonal_positions:
            c = C[pos['si'], ND - 1]
            if np.isnan(c) or c <= 0:
                c = pos['entry']
            mult = MULT.get(pos['sym'], DEF_MULT)
            pnl = (c - pos['entry']) * mult * pos['lots']
            cash_seasonal += c * mult * pos['lots'] * (1 - COMM)
            trades.append({
                'pnl_abs': pnl,
                'pnl_pct': pnl / (pos['entry'] * mult * pos['lots']) * 100 if pos['entry'] > 0 else 0,
                'days': ND - 1 - pos['entry_di'],
                'di': ND - 1, 'year': dates[ND - 1].year,
                'sym': pos['sym'], 'type': 'seasonal',
                'dir': 1, 'reason': 'end',
            })

        total_cash = cash_pairs + cash_seasonal
        if len(trades) < 10 or total_cash <= 0:
            return None

        # Combined stats
        equity = float(CASH0)
        peak = float(CASH0)
        max_dd = 0
        for t in sorted(trades, key=lambda x: x['di']):
            equity += t['pnl_abs']
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
            for di in range(MIN_TRAIN, ND):
                if dates[di].year >= wf_split_year:
                    first_test_di = di
                    break
            if first_test_di:
                days_total = (dates[ND - 1] - dates[first_test_di]).days
                yr = max(days_total / 365.25, 0.01)

        ann = ((total_cash / CASH0) ** (1 / yr) - 1) * 100

        nw = sum(1 for t in trades if t['pnl_abs'] > 0)
        wr = nw / len(trades) * 100
        avg_win = np.mean([t['pnl_pct'] for t in trades if t['pnl_abs'] > 0]) if nw > 0 else 0
        avg_loss = np.mean([abs(t['pnl_pct']) for t in trades if t['pnl_abs'] <= 0]) if nw < len(trades) else 0
        avg_days = np.mean([t['days'] for t in trades])
        pf = (sum(t['pnl_abs'] for t in trades if t['pnl_abs'] > 0) /
              max(abs(sum(t['pnl_abs'] for t in trades if t['pnl_abs'] < 0)), 1))

        trade_pnls = [t['pnl_abs'] for t in sorted(trades, key=lambda x: x['di'])]
        if len(trade_pnls) > 1:
            rets_arr = np.array(trade_pnls) / float(CASH0)
            sharpe_approx = np.mean(rets_arr) / np.std(rets_arr) * np.sqrt(252) if np.std(rets_arr) > 0 else 0
        else:
            sharpe_approx = 0

        # Breakdown by type
        pair_trades = [t for t in trades if t['type'] == 'pair']
        seas_trades = [t for t in trades if t['type'] == 'seasonal']
        pair_pnl = sum(t['pnl_abs'] for t in pair_trades)
        seas_pnl = sum(t['pnl_abs'] for t in seas_trades)

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

        return {
            'name': name, 'ann': round(ann, 1), 'n': len(trades),
            'wr': round(wr, 1), 'dd': round(max_dd, 1),
            'avg_win': round(avg_win, 2), 'avg_loss': round(avg_loss, 2),
            'avg_days': round(avg_days, 1), 'pf': round(pf, 2),
            'sharpe': round(sharpe_approx, 2), 'cash': round(total_cash, 0),
            'pair_pnl': round(pair_pnl, 0), 'seas_pnl': round(seas_pnl, 0),
            'pair_n': len(pair_trades), 'seas_n': len(seas_trades),
            'cash_pairs': round(cash_pairs, 0), 'cash_seasonal': round(cash_seasonal, 0),
            'yearly': year_stats,
        }

    # ========================================
    # PART 1: SEASONAL PARAMETER SWEEP
    # ========================================
    print("\n" + "=" * 130)
    print("  PART 1: SEASONAL PARAMETER OPTIMIZATION")
    print("=" * 130)

    seasonal_configs = []
    seasonal_results = []

    # Signal A: Anti-seasonal reversal
    # threshold (mom5 dip) = [0.01, 0.02, 0.03, 0.05], hit_min = [0.50, 0.55, 0.60]
    for threshold in [0.01, 0.02, 0.03, 0.05]:
        for hit_min in [0.50, 0.55, 0.60]:
            for window in [7, 10, 15, 20]:
                for hold in [5, 10]:
                    for top_n in [1, 3]:
                        params = {'threshold': threshold, 'hit_min': hit_min}
                        name = f"A_AntiSeas_TH{threshold:.2f}_HM{hit_min:.2f}_W{window}_D{hold}_N{top_n}"
                        seasonal_configs.append((
                            make_seasonal_score('A', params, window),
                            name, top_n, hold, 3.0, None
                        ))

    # Signal B: Seasonal + OI
    # threshold (seasonal ret) = [0.001, 0.003, 0.005], hit_min = [0.50, 0.55, 0.60]
    for threshold in [0.001, 0.003, 0.005]:
        for hit_min in [0.50, 0.55, 0.60]:
            for window in [7, 10, 15, 20]:
                for hold in [5, 10]:
                    for top_n in [1, 3]:
                        params = {'threshold': threshold, 'hit_min': hit_min}
                        name = f"B_SeasOI_TH{threshold*1000:.0f}_HM{hit_min:.2f}_W{window}_D{hold}_N{top_n}"
                        seasonal_configs.append((
                            make_seasonal_score('B', params, window),
                            name, top_n, hold, 3.0, None
                        ))

    # Signal C: Seasonal Sharpe filter
    # sharpe_min = [0.3, 0.5, 0.8], hit_min = [0.50, 0.55, 0.60]
    for sharpe_min in [0.3, 0.5, 0.8]:
        for hit_min in [0.50, 0.55, 0.60]:
            for window in [7, 10, 15, 20]:
                for hold in [5, 10]:
                    for top_n in [1, 3]:
                        params = {'sharpe_min': sharpe_min, 'hit_min': hit_min}
                        name = f"C_SeasSharpe_SM{sharpe_min:.1f}_HM{hit_min:.2f}_W{window}_D{hold}_N{top_n}"
                        seasonal_configs.append((
                            make_seasonal_score('C', params, window),
                            name, top_n, hold, 3.0, None
                        ))

    # Signal D: Seasonal + multi-horizon momentum alignment
    # threshold = [0.001, 0.003, 0.005], hit_min = [0.50, 0.55, 0.60]
    for threshold in [0.001, 0.003, 0.005]:
        for hit_min in [0.50, 0.55, 0.60]:
            for window in [7, 10, 15, 20]:
                for hold in [5, 10]:
                    for top_n in [1, 3]:
                        params = {'threshold': threshold, 'hit_min': hit_min}
                        name = f"D_SeasMom_TH{threshold*1000:.0f}_HM{hit_min:.2f}_W{window}_D{hold}_N{top_n}"
                        seasonal_configs.append((
                            make_seasonal_score('D', params, window),
                            name, top_n, hold, 3.0, None
                        ))

    print(f"  {len(seasonal_configs)} seasonal configs to test", flush=True)

    for ci, (fn, name, tn, h, trail, wf) in enumerate(seasonal_configs):
        r = run_seasonal_backtest(fn, name, top_n=tn, hold_max=h,
                                  trail_atr_mult=trail, wf_split_year=wf)
        if r and r['ann'] > 0:
            seasonal_results.append(r)
        if (ci + 1) % 50 == 0:
            print(f"  [{ci+1}/{len(seasonal_configs)}] {len(seasonal_results)} profitable seasonal configs",
                  flush=True)

    seasonal_results.sort(key=lambda x: -x['ann'])

    print(f"\n  TOP 20 SEASONAL-ONLY RESULTS:")
    print(f"  {'Config':60s} | {'Ann':>7s} | {'WR':>5s} | {'N':>4s} | {'DD':>6s} | "
          f"{'PF':>4s} | {'Sh':>5s} | {'AvgW':>7s} | {'AvgL':>6s}")
    print(f"  {'-' * 120}")
    for r in seasonal_results[:20]:
        print(f"  {r['name']:60s} | {r['ann']:+7.1f}% | {r['wr']:5.1f}% | "
              f"{r['n']:4d} | {r['dd']:6.1f}% | {r['pf']:4.2f} | "
              f"{r['sharpe']:5.2f} | {r['avg_win']:+6.2f}% | {r['avg_loss']:5.2f}%")

    # Identify best seasonal config by Sharpe (risk-adjusted)
    if seasonal_results:
        best_sharpe_seas = max(seasonal_results, key=lambda x: x.get('sharpe', 0))
        best_ann_seas = seasonal_results[0]
        print(f"\n  BEST BY ANN:  {best_ann_seas['name']}  Ann={best_ann_seas['ann']:+.1f}%  "
              f"WR={best_ann_seas['wr']:.1f}%  DD={best_ann_seas['dd']:.1f}%  Sharpe={best_ann_seas['sharpe']:.2f}")
        print(f"  BEST BY SHARPE: {best_sharpe_seas['name']}  Ann={best_sharpe_seas['ann']:+.1f}%  "
              f"WR={best_sharpe_seas['wr']:.1f}%  DD={best_sharpe_seas['dd']:.1f}%  Sharpe={best_sharpe_seas['sharpe']:.2f}")

    # Signal type effectiveness across top 20
    sig_counts = {}
    for r in seasonal_results[:20]:
        sig_label = r['name'].split('_')[0]
        if sig_label not in sig_counts:
            sig_counts[sig_label] = {'count': 0, 'ann_sum': 0.0, 'best_ann': -999, 'best_sharpe': 0}
        sig_counts[sig_label]['count'] += 1
        sig_counts[sig_label]['ann_sum'] += r['ann']
        sig_counts[sig_label]['best_ann'] = max(sig_counts[sig_label]['best_ann'], r['ann'])
        sig_counts[sig_label]['best_sharpe'] = max(sig_counts[sig_label]['best_sharpe'], r.get('sharpe', 0))

    print(f"\n  SIGNAL TYPE EFFECTIVENESS (top 20):")
    for sig, ss in sorted(sig_counts.items(), key=lambda x: -x[1]['best_ann']):
        avg = ss['ann_sum'] / ss['count'] if ss['count'] > 0 else 0
        print(f"    {sig}: appears {ss['count']}x  best_ann={ss['best_ann']:+.1f}%  "
              f"avg_ann={avg:+.1f}%  best_sharpe={ss['best_sharpe']:.2f}")

    # ========================================
    # WALK-FORWARD FOR TOP SEASONAL CONFIGS
    # ========================================
    print("\n  WALK-FORWARD VALIDATION FOR TOP 10 SEASONAL CONFIGS:", flush=True)
    wf_seasonal_results = []

    # Reconstruct configs from top seasonal results
    for r in seasonal_results[:10]:
        name = r['name']
        # Parse the config name to reconstruct params
        parts = name.split('_')
        sig_type = parts[0]
        # Extract params from name
        params = {}
        window = 10
        hold = 5
        top_n = 1
        for p in parts[1:]:
            if p.startswith('TH'):
                val = float(p[2:])
                if val > 0.1:
                    params['threshold'] = val  # anti-seasonal threshold
                else:
                    params['threshold'] = val
            elif p.startswith('HM'):
                params['hit_min'] = float(p[2:])
            elif p.startswith('SM'):
                params['sharpe_min'] = float(p[2:])
            elif p.startswith('W') and p[1:].isdigit():
                window = int(p[1:])
            elif p.startswith('D') and p[1:].isdigit():
                hold = int(p[1:])
            elif p.startswith('N') and p[1:].isdigit():
                top_n = int(p[1:])

        fn = make_seasonal_score(sig_type, params, window)
        for wf_year in [2023, 2024]:
            wf_name = f"{name}_WF{wf_year}"
            wf_r = run_seasonal_backtest(fn, wf_name, top_n=top_n, hold_max=hold,
                                         trail_atr_mult=3.0, wf_split_year=wf_year)
            if wf_r and wf_r['ann'] > 0:
                wf_seasonal_results.append(wf_r)

    if wf_seasonal_results:
        wf_seasonal_results.sort(key=lambda x: -x['ann'])
        print(f"\n  TOP 10 WALK-FORWARD SEASONAL RESULTS:")
        for r in wf_seasonal_results[:10]:
            print(f"    {r['name']:70s} | Ann {r['ann']:+7.1f}% | WR {r['wr']:5.1f}% | "
                  f"N {r['n']:4d} | DD {r['dd']:6.1f}% | Sh {r['sharpe']:5.2f}")

    # ========================================
    # PART 2: COMBINED SEASONAL + PAIR TRADING
    # ========================================
    print("\n" + "=" * 130)
    print("  PART 2: SEASONAL + PAIR TRADING COMBINATION")
    print("=" * 130)

    if not seasonal_results:
        print("  No seasonal results — skipping combination.")
        elapsed = time.time() - t_start
        print(f"\n  Total time: {elapsed:.1f}s")
        print("=" * 130)
        return

    # Use top 5 seasonal configs for combination
    combined_results = []
    combined_configs = []

    # V39 best pair config: LB10, Z1.5, H3, MP2
    pair_lb = 10
    pair_zt = 1.5
    pair_hd = 3
    pair_mp = 2

    for r in seasonal_results[:5]:
        name = r['name']
        parts = name.split('_')
        sig_type = parts[0]
        params = {}
        window = 10
        hold = 5
        top_n = 1
        for p in parts[1:]:
            if p.startswith('TH'):
                params['threshold'] = float(p[2:])
            elif p.startswith('HM'):
                params['hit_min'] = float(p[2:])
            elif p.startswith('SM'):
                params['sharpe_min'] = float(p[2:])
            elif p.startswith('W') and p[1:].isdigit():
                window = int(p[1:])
            elif p.startswith('D') and p[1:].isdigit():
                hold = int(p[1:])
            elif p.startswith('N') and p[1:].isdigit():
                top_n = int(p[1:])

        seasonal_fn = make_seasonal_score(sig_type, params, window)

        for cap_pct in [0.3, 0.5, 0.7, 0.8]:
            comb_name = f"COMB_P{cap_pct:.0f}pct_{name}"
            combined_configs.append((
                seasonal_fn, pair_lb, pair_zt, pair_hd, pair_mp,
                cap_pct, comb_name, top_n, hold, None
            ))
            # Walk-forward versions
            for wf_year in [2023, 2024]:
                wf_name = f"COMB_P{cap_pct:.0f}pct_{name}_WF{wf_year}"
                combined_configs.append((
                    seasonal_fn, pair_lb, pair_zt, pair_hd, pair_mp,
                    cap_pct, wf_name, top_n, hold, wf_year
                ))

    print(f"  {len(combined_configs)} combined configs to test", flush=True)

    for ci, (seas_fn, plb, pzt, phd, pmp, cap_pct, cname, stn, shld, wf) in enumerate(combined_configs):
        cr = run_combined_backtest(
            seas_fn, plb, pzt, phd, pmp, cap_pct, cname,
            seasonal_top_n=stn, seasonal_hold=shld, wf_split_year=wf
        )
        if cr and cr['ann'] > 0:
            combined_results.append(cr)
        if (ci + 1) % 20 == 0:
            print(f"  [{ci+1}/{len(combined_configs)}] {len(combined_results)} profitable combined configs",
                  flush=True)

    combined_results.sort(key=lambda x: -x['ann'])

    # Separate walk-forward
    comb_full = [r for r in combined_results if '_WF' not in r['name']]
    comb_wf = [r for r in combined_results if '_WF' in r['name']]

    print(f"\n  TOP 20 COMBINED (SEASONAL + PAIRS) FULL-PERIOD RESULTS:")
    print(f"  {'Config':75s} | {'Ann':>7s} | {'WR':>5s} | {'N':>4s} | {'DD':>6s} | "
          f"{'PF':>4s} | {'Sh':>5s} | {'PairPnL':>9s} | {'SeasPnL':>9s} | {'PN':>3s} | {'SN':>3s}")
    print(f"  {'-' * 140}")
    for r in comb_full[:20]:
        print(f"  {r['name']:75s} | {r['ann']:+7.1f}% | {r['wr']:5.1f}% | "
              f"{r['n']:4d} | {r['dd']:6.1f}% | {r['pf']:4.2f} | "
              f"{r['sharpe']:5.2f} | {r['pair_pnl']:+9.0f} | {r['seas_pnl']:+9.0f} | "
              f"{r['pair_n']:3d} | {r['seas_n']:3d}")

    if comb_wf:
        comb_wf.sort(key=lambda x: -x['ann'])
        print(f"\n  TOP 10 COMBINED WALK-FORWARD RESULTS:")
        print(f"  {'-' * 140}")
        for r in comb_wf[:10]:
            print(f"  {r['name']:75s} | Ann {r['ann']:+7.1f}% | WR {r['wr']:5.1f}% | "
                  f"N {r['n']:4d} | DD {r['dd']:6.1f}% | Sh {r['sharpe']:5.2f} | "
                  f"PairPnL {r['pair_pnl']:+.0f} | SeasPnL {r['seas_pnl']:+.0f}")

    # Best combined detail
    if comb_full:
        best_comb = comb_full[0]
        print(f"\n  BEST COMBINED CONFIG DETAIL: {best_comb['name']}")
        print(f"  Ann={best_comb['ann']:+.1f}%  WR={best_comb['wr']:.1f}%  N={best_comb['n']}  "
              f"DD={best_comb['dd']:.1f}%  PF={best_comb['pf']:.2f}  Sharpe={best_comb['sharpe']:.2f}  "
              f"Final={best_comb['cash']:.0f}")
        print(f"  Pair trades: {best_comb['pair_n']}  Pair PnL: {best_comb['pair_pnl']:+.0f}  "
              f"Pair final cash: {best_comb['cash_pairs']:.0f}")
        print(f"  Seas trades: {best_comb['seas_n']}  Seas PnL: {best_comb['seas_pnl']:+.0f}  "
              f"Seas final cash: {best_comb['cash_seasonal']:.0f}")

        if best_comb.get('yearly'):
            print(f"\n  YEARLY BREAKDOWN:")
            for y in sorted(best_comb['yearly'].keys()):
                ys = best_comb['yearly'][y]
                wr_y = ys['w'] / ys['n'] * 100 if ys['n'] > 0 else 0
                print(f"    {y}: {ys['n']:3d}t  WR={wr_y:5.1f}%  "
                      f"PnL={ys['pnl']:+.1f}%  Abs={ys['pnl_abs_sum']:+.0f}")

    # Capital allocation comparison (for the best seasonal config)
    print(f"\n  CAPITAL ALLOCATION COMPARISON:")
    if seasonal_results:
        best_seas_name = seasonal_results[0]['name']
        cap_results = [r for r in comb_full if best_seas_name in r['name']]
        for r in sorted(cap_results, key=lambda x: -x['ann']):
            cap_pct_str = r['name'].split('_')[1]
            print(f"    {cap_pct_str}: Ann={r['ann']:+.1f}%  WR={r['wr']:.1f}%  "
                  f"DD={r['dd']:.1f}%  Sharpe={r['sharpe']:.2f}  "
                  f"PairPnL={r['pair_pnl']:+.0f}  SeasPnL={r['seas_pnl']:+.0f}")

    # ========================================
    # FINAL SUMMARY
    # ========================================
    print(f"\n{'=' * 130}")
    print(f"  FINAL SUMMARY")
    print(f"{'=' * 130}")

    print(f"\n  PART 1 — SEASONAL OPTIMIZATION:")
    if seasonal_results:
        print(f"    Total profitable configs: {len(seasonal_results)}")
        print(f"    Best full-period:  {seasonal_results[0]['name']}")
        print(f"      Ann={seasonal_results[0]['ann']:+.1f}%  WR={seasonal_results[0]['wr']:.1f}%  "
              f"DD={seasonal_results[0]['dd']:.1f}%  Sharpe={seasonal_results[0]['sharpe']:.2f}")
        if wf_seasonal_results:
            print(f"    Best walk-forward: {wf_seasonal_results[0]['name']}")
            print(f"      Ann={wf_seasonal_results[0]['ann']:+.1f}%  WR={wf_seasonal_results[0]['wr']:.1f}%  "
                  f"DD={wf_seasonal_results[0]['dd']:.1f}%  Sharpe={wf_seasonal_results[0]['sharpe']:.2f}")

    print(f"\n  PART 2 — COMBINED SEASONAL + PAIRS:")
    if comb_full:
        print(f"    Total profitable combined configs: {len(comb_full)}")
        print(f"    Best combined: {comb_full[0]['name']}")
        print(f"      Ann={comb_full[0]['ann']:+.1f}%  WR={comb_full[0]['wr']:.1f}%  "
              f"DD={comb_full[0]['dd']:.1f}%  Sharpe={comb_full[0]['sharpe']:.2f}")
        print(f"      Pair contribution: {comb_full[0]['pair_pnl']:+.0f}  "
              f"Seasonal contribution: {comb_full[0]['seas_pnl']:+.0f}")
    if comb_wf:
        print(f"    Best combined WF: {comb_wf[0]['name']}")
        print(f"      Ann={comb_wf[0]['ann']:+.1f}%  WR={comb_wf[0]['wr']:.1f}%  "
              f"DD={comb_wf[0]['dd']:.1f}%  Sharpe={comb_wf[0]['sharpe']:.2f}")

    # Comparison: standalone seasonal vs standalone pairs vs combined
    print(f"\n  STRATEGY COMPARISON:")
    if seasonal_results:
        print(f"    Seasonal-only best:  Ann={seasonal_results[0]['ann']:+.1f}%  "
              f"Sharpe={seasonal_results[0]['sharpe']:.2f}  DD={seasonal_results[0]['dd']:.1f}%")
    if comb_full:
        print(f"    Combined best:       Ann={comb_full[0]['ann']:+.1f}%  "
              f"Sharpe={comb_full[0]['sharpe']:.2f}  DD={comb_full[0]['dd']:.1f}%")
        # Implied pairs-only (from capital allocation)
        # The pair portion alone: pair_pnl / (CASH0 * cap_pct)
        if comb_full:
            r = comb_full[0]
            # Extract cap_pct from name
            parts = r['name'].split('_')
            cap_pct_str = parts[1]  # e.g. "P80pct"
            try:
                cap_pct = int(cap_pct_str.replace('P', '').replace('pct', '')) / 100.0
                pair_ann_approx = r['pair_pnl'] / (CASH0 * cap_pct) / (
                    (dates[ND - 1] - dates[MIN_TRAIN]).days / 365.25) * 100
                seas_ann_approx = r['seas_pnl'] / (CASH0 * (1 - cap_pct)) / (
                    (dates[ND - 1] - dates[MIN_TRAIN]).days / 365.25) * 100
                print(f"    Implied pair-only ann (approx): {pair_ann_approx:+.1f}%")
                print(f"    Implied seasonal-only ann (approx): {seas_ann_approx:+.1f}%")
            except (ValueError, ZeroDivisionError):
                pass

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.1f}s")
    print("=" * 130)


if __name__ == '__main__':
    main()
