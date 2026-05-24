"""
Alpha Futures V45 — Cross-Timeframe Momentum Alignment
=======================================================
Core idea: When short-term (3d), medium-term (10d), and long-term (20d) momentum
all agree on direction, the signal is much stronger than any single timeframe.
Multi-timeframe alignment as a new signal source.

5 signals:
  1. Triple Momentum Alignment — all TFs agree, score = sum of aligned momentums
  2. Momentum Acceleration + Alignment — aligned AND accelerating
  3. Trend Quality Gate — aligned AND high R-squared (clean trend)
  4. Breakout from Consolidation — low vol + flat → breakout
  5. Cross-timeframe with Group Confirmation — group lag + TF alignment

Configs (~250): TF combos x min_momentum x hold x trail x top_n x signal
Walk-forward validation on best configs.
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

TF_COMBOS = {
    '3_10_20': [3, 10, 20],
    '3_7_15':  [3, 7, 15],
    '5_10_20': [5, 10, 20],
    '5_15_30': [5, 15, 30],
}


def main():
    t_start = time.time()
    print("=" * 120)
    print("Alpha Futures V45 — Cross-Timeframe Momentum Alignment")
    print("When short/medium/long-term momentum all agree, the signal is much stronger.")
    print("=" * 120)

    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)
    sym_to_si = {syms[si]: si for si in range(NS)}

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

    print(f"  {NS} commodities, {ND} days, {len(group_members)} groups")

    # ========================================
    # PRECOMPUTE ALL SIGNALS
    # ========================================
    print("\n[Signals] Computing momentum, regression, acceleration, vol...", flush=True)
    t0 = time.time()

    # 1. Momentum at multiple lookbacks
    print("  Computing momentum at 7 lookbacks...", flush=True)
    mom = {}
    for lag in [3, 5, 7, 10, 15, 20, 30]:
        m = np.full((NS, ND), np.nan)
        for si in range(NS):
            for di in range(lag, ND):
                c_now = C[si, di]
                c_prev = C[si, di - lag]
                if not np.isnan(c_now) and not np.isnan(c_prev) and c_prev > 0:
                    m[si, di] = (c_now - c_prev) / c_prev
        mom[lag] = m

    # 2. Linear regression slope + R-squared at multiple lookbacks
    print("  Computing linear regression slopes & R-squared...", flush=True)
    linreg_slope = {}
    r_squared = {}
    for lag in [5, 10, 20]:
        slope_arr = np.full((NS, ND), np.nan)
        rsq_arr = np.full((NS, ND), np.nan)
        for si in range(NS):
            for di in range(lag, ND):
                prices = C[si, di - lag + 1:di + 1]
                valid = ~np.isnan(prices)
                nv = valid.sum()
                if nv < lag * 0.7:
                    continue
                y = prices[valid]
                n = len(y)
                if n < 4:
                    continue
                t_arr = np.arange(n, dtype=float)
                t_mean = t_arr.mean()
                y_mean = y.mean()
                ss_xy = np.sum((t_arr - t_mean) * (y - y_mean))
                ss_xx = np.sum((t_arr - t_mean) ** 2)
                ss_yy = np.sum((y - y_mean) ** 2)
                if ss_xx < 1e-12 or ss_yy < 1e-12:
                    continue
                b = ss_xy / ss_xx
                # Slope as fraction of mean price for comparability
                if y_mean > 0:
                    slope_arr[si, di] = b / y_mean
                # R-squared
                rsq_arr[si, di] = (ss_xy ** 2) / (ss_xx * ss_yy)
        linreg_slope[lag] = slope_arr
        r_squared[lag] = rsq_arr

    # 3. Acceleration: mom[5] now vs mom[5] 3 days ago
    print("  Computing acceleration...", flush=True)
    accel = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(8, ND):  # need 5+3 lag
            m_now = mom[5][si, di]
            m_prev = mom[5][si, di - 3]
            if not np.isnan(m_now) and not np.isnan(m_prev):
                accel[si, di] = m_now - m_prev

    # 4. Volatility regime: rolling std of daily returns over 20 days
    print("  Computing volatility regime...", flush=True)
    vol_20 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(21, ND):
            rets = []
            for dd in range(di - 20, di):
                c1 = C[si, dd]
                c0 = C[si, dd - 1]
                if not np.isnan(c1) and not np.isnan(c0) and c0 > 0:
                    rets.append((c1 - c0) / c0)
            if len(rets) >= 10:
                vol_20[si, di] = np.std(rets, ddof=1)

    # Rolling mean of vol_20 over 60 days
    vol_20_ma60 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(81, ND):
            vals = vol_20[si, di - 60:di]
            valid = vals[~np.isnan(vals)]
            if len(valid) >= 20:
                vol_20_ma60[si, di] = np.mean(valid)

    # 5. KER at 20 (for consolidation detection)
    print("  Computing KER...", flush=True)
    ker20 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            c_now = C[si, di]
            c_20 = C[si, di - 20]
            if np.isnan(c_now) or np.isnan(c_20) or c_20 <= 0:
                continue
            net = abs(c_now - c_20)
            total = 0.0
            for dd in range(di - 19, di + 1):
                c1 = C[si, dd]
                c0 = C[si, dd - 1]
                if not np.isnan(c1) and not np.isnan(c0):
                    total += abs(c1 - c0)
            if total > 0:
                ker20[si, di] = net / total

    # 6. Group momentum (excluding self) at multiple lookbacks
    print("  Computing group momentum...", flush=True)
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

    # 7. ATR for trailing stops
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

    print(f"  All signals computed ({time.time() - t0:.1f}s)", flush=True)

    # ========================================
    # SCORING FUNCTIONS (5 signals)
    # ========================================

    def make_signal1_align(tf_key, min_mom=0.0, scale=10.0):
        """Signal 1: Triple Momentum Alignment.
        All three timeframes must agree on direction (positive for long).
        Score = sum of the aligned momentums.
        """
        lags = TF_COMBOS[tf_key]
        def score(si, di):
            vals = []
            for lag in lags:
                v = mom[lag][si, di]
                if np.isnan(v):
                    return np.nan
                vals.append(v)
            # All must be positive (long alignment)
            if any(v <= min_mom for v in vals):
                return np.nan
            return np.clip(sum(vals) * scale, 0, 1)
        return score

    def make_signal2_accel(tf_key, min_mom=0.0, accel_scale=5.0, scale=10.0):
        """Signal 2: Momentum Acceleration + Alignment.
        Aligned AND accelerating = strong trend entry.
        """
        lags = TF_COMBOS[tf_key]
        def score(si, di):
            vals = []
            for lag in lags:
                v = mom[lag][si, di]
                if np.isnan(v):
                    return np.nan
                vals.append(v)
            if any(v <= min_mom for v in vals):
                return np.nan
            a = accel[si, di]
            if np.isnan(a) or a <= 0:
                return np.nan
            alignment_sum = sum(vals)
            return np.clip(alignment_sum * (1 + a * accel_scale) * scale, 0, 1)
        return score

    def make_signal3_quality(tf_key, min_mom=0.0, rsq_threshold=0.5, scale=10.0):
        """Signal 3: Trend Quality Gate.
        Aligned AND high R-squared (clean trend, not noisy).
        """
        lags = TF_COMBOS[tf_key]
        # Use the longest lag in the combo for R-squared
        max_lag = max(lags)
        rsq_lag = None
        for rl in [20, 10, 5]:
            if rl >= max_lag:
                rsq_lag = rl
                break
        if rsq_lag is None:
            rsq_lag = 20
        def score(si, di):
            vals = []
            for lag in lags:
                v = mom[lag][si, di]
                if np.isnan(v):
                    return np.nan
                vals.append(v)
            if any(v <= min_mom for v in vals):
                return np.nan
            rsq = r_squared[rsq_lag][si, di]
            if np.isnan(rsq) or rsq < rsq_threshold:
                return np.nan
            alignment_sum = sum(vals)
            return np.clip(alignment_sum * rsq * scale, 0, 1)
        return score

    def make_signal4_breakout(breakout_threshold=0.01, scale=10.0):
        """Signal 4: Breakout from Consolidation.
        Detect consolidation (low vol + flat momentum + narrow range),
        then enter when momentum breaks out.
        """
        def score(si, di):
            vr = vol_20[si, di]
            vr_ma = vol_20_ma60[si, di]
            if np.isnan(vr) or np.isnan(vr_ma) or vr_ma <= 0:
                return np.nan
            vol_ratio = vr / vr_ma
            if vol_ratio >= 0.7:
                return np.nan  # Not in consolidation
            k = ker20[si, di]
            if np.isnan(k) or k >= 0.15:
                return np.nan  # Not flat enough
            m3 = mom[3][si, di]
            if np.isnan(m3) or m3 <= breakout_threshold:
                return np.nan  # No breakout
            # Stronger breakout from flatter consolidation
            ker_factor = 1.0 / max(k, 0.01)
            return np.clip(m3 * ker_factor * scale, 0, 1)
        return score

    def make_signal5_group_tf(tf_key, mom_lag=5, min_lag=0.003, align_mult=1.5, scale=10.0):
        """Signal 5: Cross-timeframe with Group Confirmation.
        Group momentum lag signal AND multi-TF alignment.
        """
        lags = TF_COMBOS[tf_key]
        def score(si, di):
            # Group lag component
            own = mom[mom_lag][si, di]
            grp = grp_mom[mom_lag][si, di]
            if np.isnan(own) or np.isnan(grp):
                return np.nan
            divergence = grp - own
            if divergence < min_lag:
                return np.nan

            # TF alignment component
            vals = []
            for lag in lags:
                v = mom[lag][si, di]
                if np.isnan(v):
                    return np.nan
                vals.append(v)
            # All timeframes must be positive
            if any(v <= 0 for v in vals):
                return np.nan

            group_score = divergence * scale
            return np.clip(group_score * align_mult, 0, 1)
        return score

    # ========================================
    # BACKTEST ENGINE (same as V34b)
    # ========================================
    def run_backtest(score_fn, name, top_n=1, hold_min=2, hold_max=3,
                     trail_atr_mult=2.5, wf_split_year=None):
        """Single position, long only, cash/(price*mult) lots."""
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

                # Signal flip (after min hold)
                if exit_reason is None and days_held >= hold_min:
                    cur_score = score_fn(pos['si'], di)
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
                        trail_price = c - trail_atr_mult * atr_val
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
            for d in range(MIN_TRAIN, ND):
                if dates[d].year >= wf_split_year:
                    first_test_di = d
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

        reasons = {}
        for t in trades:
            r = t['reason']
            if r not in reasons:
                reasons[r] = {'n': 0, 'w': 0, 'pnl': 0.0}
            reasons[r]['n'] += 1
            if t['pnl_abs'] > 0:
                reasons[r]['w'] += 1
            reasons[r]['pnl'] += t['pnl_pct']

        year_stats = {}
        for t in trades:
            y = t['year']
            if y not in year_stats:
                year_stats[y] = {'n': 0, 'w': 0, 'pnl': 0.0}
            year_stats[y]['n'] += 1
            if t['pnl_abs'] > 0:
                year_stats[y]['w'] += 1
            year_stats[y]['pnl'] += t['pnl_pct']

        grp_counts = {}
        for t in trades:
            g = GROUP_MAP.get(t['sym'], 'other')
            if g not in grp_counts:
                grp_counts[g] = {'n': 0, 'w': 0, 'pnl': 0.0}
            grp_counts[g]['n'] += 1
            if t['pnl_abs'] > 0:
                grp_counts[g]['w'] += 1
            grp_counts[g]['pnl'] += t['pnl_abs']

        return {
            'name': name, 'ann': round(ann, 1), 'n': len(trades),
            'wr': round(wr, 1), 'dd': round(max_dd, 1),
            'avg_win': round(avg_win, 2), 'avg_loss': round(avg_loss, 2),
            'avg_days': round(avg_days, 1), 'pf': round(pf, 2),
            'cash': round(cash, 0),
            'reasons': reasons, 'yearly': year_stats, 'grp_counts': grp_counts,
        }

    # ========================================
    # PARAMETER SWEEP
    # ========================================
    print("\n[Backtest] Building configurations...", flush=True)
    configs = []

    # Signal 1: Triple Momentum Alignment
    for tf_key in TF_COMBOS:
        for min_mom in [0, 0.005, 0.01, 0.02]:
            for hold in [3, 5, 7]:
                for trail in [2.5, 3.0, 4.0]:
                    for tn in [1, 3]:
                        configs.append((
                            make_signal1_align(tf_key, min_mom=min_mom),
                            f"S1_TF{tf_key}_MM{min_mom*1000:.0f}_H{hold}_TR{trail:.0f}_N{tn}",
                            tn, 2, hold, trail, None
                        ))

    # Signal 2: Acceleration + Alignment
    for tf_key in TF_COMBOS:
        for min_mom in [0, 0.005, 0.01]:
            for hold in [3, 5, 7]:
                for trail in [2.5, 3.0, 4.0]:
                    for tn in [1, 3]:
                        configs.append((
                            make_signal2_accel(tf_key, min_mom=min_mom),
                            f"S2_TF{tf_key}_MM{min_mom*1000:.0f}_H{hold}_TR{trail:.0f}_N{tn}",
                            tn, 2, hold, trail, None
                        ))

    # Signal 3: Trend Quality Gate
    for tf_key in TF_COMBOS:
        for min_mom in [0, 0.005, 0.01]:
            for hold in [3, 5, 7]:
                for trail in [2.5, 3.0, 4.0]:
                    for tn in [1, 3]:
                        configs.append((
                            make_signal3_quality(tf_key, min_mom=min_mom),
                            f"S3_TF{tf_key}_MM{min_mom*1000:.0f}_H{hold}_TR{trail:.0f}_N{tn}",
                            tn, 2, hold, trail, None
                        ))

    # Signal 4: Breakout from Consolidation
    for bt in [0.005, 0.01, 0.02]:
        for hold in [3, 5, 7]:
            for trail in [2.5, 3.0, 4.0]:
                for tn in [1, 3]:
                    configs.append((
                        make_signal4_breakout(breakout_threshold=bt),
                        f"S4_BT{bt*1000:.0f}_H{hold}_TR{trail:.0f}_N{tn}",
                        tn, 2, hold, trail, None
                    ))

    # Signal 5: Cross-TF with Group Confirmation
    for tf_key in TF_COMBOS:
        for hold in [3, 5, 7]:
            for trail in [2.5, 3.0, 4.0]:
                for tn in [1, 3]:
                    configs.append((
                        make_signal5_group_tf(tf_key),
                        f"S5_TF{tf_key}_H{hold}_TR{trail:.0f}_N{tn}",
                        tn, 2, hold, trail, None
                    ))

    print(f"  {len(configs)} full-period configurations", flush=True)

    # Walk-forward configs for best parameter combos
    wf_configs = []
    for sig in [1, 2, 3, 5]:
        for tf_key in ['3_10_20', '5_10_20']:
            for hold in [3, 5]:
                for trail in [2.5, 3.0]:
                    for tn in [1, 3]:
                        for wf_year in [2023, 2024]:
                            if sig == 1:
                                fn = make_signal1_align(tf_key, min_mom=0.005)
                            elif sig == 2:
                                fn = make_signal2_accel(tf_key, min_mom=0.005)
                            elif sig == 3:
                                fn = make_signal3_quality(tf_key, min_mom=0.005)
                            else:
                                fn = make_signal5_group_tf(tf_key)
                            wf_configs.append((
                                fn,
                                f"S{sig}_TF{tf_key}_H{hold}_TR{trail:.0f}_N{tn}_WF{wf_year}",
                                tn, 2, hold, trail, wf_year
                            ))
    # Signal 4 walk-forward
    for bt in [0.01]:
        for hold in [3, 5]:
            for tn in [1, 3]:
                for wf_year in [2023, 2024]:
                    wf_configs.append((
                        make_signal4_breakout(breakout_threshold=bt),
                        f"S4_BT{bt*1000:.0f}_H{hold}_TR30_N{tn}_WF{wf_year}",
                        tn, 2, hold, 3.0, wf_year
                    ))

    all_configs = configs + wf_configs
    print(f"  {len(all_configs)} total configurations ({len(configs)} full + {len(wf_configs)} WF)", flush=True)

    # ========================================
    # RUN BACKTESTS
    # ========================================
    print("\n[Backtest] Running...", flush=True)
    results = []
    t_backtest_start = time.time()

    for ci, (fn, name, tn, hmin, hmax, trail, wf) in enumerate(all_configs):
        r = run_backtest(fn, name, top_n=tn, hold_min=hmin, hold_max=hmax,
                         trail_atr_mult=trail, wf_split_year=wf)
        if r and r['ann'] > 0:
            results.append(r)
            if r['ann'] > 50:
                parts = []
                for reason, stats in sorted(r['reasons'].items()):
                    wr_r = stats['w'] / stats['n'] * 100 if stats['n'] > 0 else 0
                    parts.append(f"{reason}:{stats['n']}({wr_r:.0f}%)")
                print(f"  {r['name']:55s} | Ann {r['ann']:+7.1f}% | WR {r['wr']:5.1f}% | "
                      f"N {r['n']:4d} | DD {r['dd']:6.1f}% | PF {r['pf']:4.2f} | "
                      f"AvgW {r['avg_win']:+.2f}% | AvgL {r['avg_loss']:.2f}% | AvgD {r['avg_days']:.1f}")
                print(f"  {'':55s} | Exits: {' | '.join(parts)}")

        if (ci + 1) % 50 == 0:
            elapsed = time.time() - t_backtest_start
            rate = (ci + 1) / elapsed
            eta = (len(all_configs) - ci - 1) / rate
            print(f"  [{ci + 1}/{len(all_configs)}] {len(results)} profitable "
                  f"({elapsed:.0f}s elapsed, ETA {eta:.0f}s)", flush=True)

    print(f"  Backtests done ({time.time() - t_backtest_start:.1f}s)", flush=True)

    # ========================================
    # RESULTS
    # ========================================
    results.sort(key=lambda x: -x['ann'])

    wf_results = [r for r in results if '_WF' in r['name']]
    full_results = [r for r in results if '_WF' not in r['name']]

    # --- Top 20 full-period ---
    print(f"\n{'=' * 130}")
    print(f"  TOP 20 FULL-PERIOD RESULTS")
    print(f"{'=' * 130}")
    print(f"  {'Strategy':55s} | {'Ann':>7s} | {'WR':>5s} | {'N':>4s} | {'DD':>6s} | "
          f"{'PF':>4s} | {'AvgW':>7s} | {'AvgL':>6s} | {'AvgD':>4s}")
    print(f"  {'-' * 130}")
    for r in full_results[:20]:
        print(f"  {r['name']:55s} | {r['ann']:+7.1f}% | {r['wr']:5.1f}% | "
              f"{r['n']:4d} | {r['dd']:6.1f}% | {r['pf']:4.2f} | "
              f"{r['avg_win']:+6.2f}% | {r['avg_loss']:5.2f}% | "
              f"{r['avg_days']:4.1f}")

    # --- Top 10 walk-forward ---
    if wf_results:
        print(f"\n  TOP 10 WALK-FORWARD RESULTS (out-of-sample)")
        print(f"  {'-' * 130}")
        wf_results.sort(key=lambda x: -x['ann'])
        for r in wf_results[:10]:
            print(f"  {r['name']:55s} | {r['ann']:+7.1f}% | {r['wr']:5.1f}% | "
                  f"{r['n']:4d} | {r['dd']:6.1f}% | {r['pf']:4.2f}")

    # --- Best per signal ---
    print(f"\n  BEST PER SIGNAL:")
    for sig_num in [1, 2, 3, 4, 5]:
        sig_results = [r for r in full_results if r['name'].startswith(f'S{sig_num}_')]
        if sig_results:
            best = sig_results[0]
            print(f"    Signal {sig_num}: {best['name']:55s} | "
                  f"Ann {best['ann']:+7.1f}% | WR {best['wr']:5.1f}% | "
                  f"N {best['n']:4d} | DD {best['dd']:6.1f}% | PF {best['pf']:4.2f}")

    # --- Best config detail ---
    if full_results:
        best = full_results[0]
        print(f"\n{'=' * 130}")
        print(f"  BEST CONFIG DETAIL")
        print(f"  {best['name']}")
        print(f"  Ann={best['ann']:+.1f}%  WR={best['wr']:.1f}%  N={best['n']}  "
              f"DD={best['dd']:.1f}%  PF={best['pf']:.2f}")
        print(f"  AvgWin={best['avg_win']:+.2f}%  AvgLoss={best['avg_loss']:.2f}%  "
              f"AvgDays={best['avg_days']:.1f}  Final={best['cash']:.0f}")
        print(f"{'=' * 130}")

        print(f"\n  EXIT REASON BREAKDOWN:")
        for reason, s in sorted(best['reasons'].items(), key=lambda x: -x[1]['n']):
            rwr = s['w'] / max(s['n'], 1) * 100
            print(f"    {reason:12s}: {s['n']:4d} trades  WR={rwr:5.1f}%  PnL={s['pnl']:+.1f}%")

        print(f"\n  YEARLY BREAKDOWN:")
        for y in sorted(best['yearly'].keys()):
            s = best['yearly'][y]
            wr_y = s['w'] / max(s['n'], 1) * 100
            print(f"    {y}: {s['n']:3d} trades  WR={wr_y:5.1f}%  PnL={s['pnl']:+.1f}%")

        print(f"\n  GROUP BREAKDOWN:")
        for g in sorted(best['grp_counts'].keys(), key=lambda x: -best['grp_counts'][x]['n']):
            gs = best['grp_counts'][g]
            wr_g = gs['w'] / max(gs['n'], 1) * 100
            print(f"    {g:15s}: {gs['n']:3d}t  WR={wr_g:5.1f}%  Abs={gs['pnl']:+.0f}")

    # --- Yearly for top 5 ---
    if len(full_results) >= 2:
        print(f"\n  YEARLY BREAKDOWN FOR TOP 5:")
        for idx, r in enumerate(full_results[:5]):
            print(f"\n  #{idx + 1}: {r['name']} (Ann={r['ann']:+.1f}%, WR={r['wr']:.1f}%, DD={r['dd']:.1f}%)")
            for y in sorted(r['yearly'].keys()):
                ys = r['yearly'][y]
                wr_y = ys['w'] / ys['n'] * 100 if ys['n'] > 0 else 0
                print(f"    {y}: {ys['n']:3d}t  WR={wr_y:5.1f}%  PnL={ys['pnl']:+.1f}%")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.1f}s")
    print("=" * 130)


if __name__ == '__main__':
    main()
