"""
Alpha Futures V38 — Multi-Position Portfolio Strategies
========================================================
Best single-position strategy is v34b's group momentum lag at +86.8% annual.
V38 tests running multiple positions and multiple independent signals simultaneously
to push toward 600% annual.

Three approaches:
1. Multi-position with group diversification (1-8 slots, max 1 per group)
2. Multi-strategy portfolio (group lag + VDP + OI surge)
3. Aggressive raw signal (no filters) with multiple positions

Key rule: max 1 position per group (ferrous, nonferrous, oils, energy, chemical)
to ensure positions are independent.
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


def main():
    t_start = time.time()
    print("=" * 120)
    print("Alpha Futures V38 -- Multi-Position Portfolio Strategies")
    print("Goal: combine best signals with multi-position to push toward 600% annual")
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

    # Build reverse map: si -> group name (None if not in any group)
    si_group = {}
    for si in range(NS):
        si_group[si] = GROUP_MAP.get(syms[si])

    print(f"  {NS} stocks, {ND} days, Groups: {len(group_members)}")

    # ========================================
    # PRECOMPUTE SIGNALS
    # ========================================
    print("\n[Signals] Computing all factors...", flush=True)
    t0 = time.time()

    # Momentum at multiple lookbacks
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

    # Group momentum excluding self
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
                        mv = mom[lag][sk, di]
                        if not np.isnan(mv):
                            ms.append(mv)
                    if ms:
                        gm[sj, di] = np.mean(ms)
        grp_mom[lag] = gm

    # VDP EMA (Volume Delta Pressure)
    vdp_ema = np.full((NS, ND), np.nan)
    for si in range(NS):
        vdp_e = 0.0
        alpha = 2.0 / 11
        for di in range(1, ND):
            d = di - 1
            cd, hd, ld, vd = C[si, d], H[si, d], L[si, d], V[si, d]
            if np.isnan(cd) or np.isnan(hd) or np.isnan(ld) or np.isnan(vd):
                continue
            rng = hd - ld
            if rng <= 0:
                continue
            vdp_val = vd * (2 * cd - hd - ld) / rng
            vdp_e = alpha * vdp_val + (1 - alpha) * vdp_e
            vdp_ema[si, di] = vdp_e

    # VDP momentum: rate of change of VDP EMA
    vdp_mom = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(6, ND):
            cur = vdp_ema[si, di]
            prev = vdp_ema[si, di - 5]
            if not np.isnan(cur) and not np.isnan(prev) and prev != 0:
                vdp_mom[si, di] = (cur - prev) / abs(prev)

    # OI surge: when OI rises > 2x 20-day average AND price is up
    oi_avg20 = np.full((NS, ND), np.nan)
    oi_surge = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(21, ND):
            ois = OI[si, di - 20:di]
            valid = ois[~np.isnan(ois)]
            if len(valid) >= 10:
                avg_oi = np.mean(valid)
                oi_avg20[si, di] = avg_oi
                cur_oi = OI[si, di]
                if not np.isnan(cur_oi) and avg_oi > 0:
                    ratio = cur_oi / avg_oi
                    m5 = mom[5][si, di]
                    if not np.isnan(m5):
                        # OI surge + price up = institutional buying
                        if ratio > 2.0 and m5 > 0:
                            oi_surge[si, di] = min((ratio - 1.0) * m5 * 10, 1.0)
                        elif ratio > 1.5 and m5 > 0:
                            oi_surge[si, di] = min((ratio - 1.0) * m5 * 5, 0.7)

    # ATR 10-day
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

    print(f"  Done ({time.time()-t0:.1f}s)", flush=True)

    # ========================================
    # SCORING FUNCTIONS
    # ========================================

    def score_group_lag(si, di, lag=5, scale=10.0):
        """Group momentum lag: when own lags group, it catches up."""
        own = mom[lag][si, di]
        grp = grp_mom[lag][si, di]
        if np.isnan(own) or np.isnan(grp):
            return np.nan
        divergence = grp - own
        if divergence <= 0:
            return np.nan
        sc = np.clip(divergence * scale, -1, 1)
        return sc if sc > 0 else np.nan

    def score_vdp_ema(si, di):
        """VDP EMA momentum signal."""
        vd = vdp_ema[si, di]
        vm = vdp_mom[si, di]
        m5 = mom[5][si, di]
        if np.isnan(vd) or np.isnan(m5):
            return np.nan
        # VDP positive and rising + price momentum positive
        if vd > 0 and not np.isnan(vm) and vm > 0 and m5 > 0:
            sc = np.clip(m5 * 8, 0, 1)
            sc *= min(1.0 + abs(vd) / 5e6, 1.5)
            return sc
        return np.nan

    def score_oi_surge(si, di):
        """OI surge signal: OI > 2x avg + price up."""
        os_val = oi_surge[si, di]
        if np.isnan(os_val) or os_val <= 0:
            return np.nan
        return os_val

    # ========================================
    # MULTI-POSITION BACKTEST ENGINE
    # ========================================

    def run_backtest_multi(score_fns, name, max_positions=1, max_per_group=1,
                            hold_days=3, trail_atr=3.0, wf_split_year=None):
        """
        Multi-position backtest with group diversification.

        score_fns: list of (score_function, weight) tuples
        max_positions: max concurrent positions
        max_per_group: max positions per group (1 = diversified)
        hold_days: max hold before time exit
        trail_atr: trailing stop = price - trail_atr * ATR
        wf_split_year: if set, only test on years >= this value
        """
        cash = float(CASH0)
        trades = []
        positions = []

        for di in range(MIN_TRAIN, ND):
            year = dates[di].year
            if wf_split_year is not None and year < wf_split_year:
                continue

            # === Manage existing positions ===
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

                # Trailing stop (after day 1)
                if trail_atr > 0 and days_held >= 2:
                    atr = pos.get('atr', 0)
                    if atr > 0 and pos['dir'] == 1:
                        new_trail = c - trail_atr * atr
                        if new_trail > pos.get('trail_price', pos['entry']):
                            pos['trail_price'] = new_trail
                        if c < pos['trail_price']:
                            exit_reason = 'trail'

                # Signal flip (after 1 day)
                if exit_reason is None and days_held >= 2:
                    # Check merged score for this commodity
                    merged = 0.0
                    total_w = 0.0
                    for sfn, w in score_fns:
                        sc = sfn(pos['si'], di)
                        if not np.isnan(sc):
                            merged += sc * w
                            total_w += w
                    if total_w > 0:
                        merged /= total_w
                    if merged < -0.01:
                        exit_reason = 'signal_flip'

                # Time exit
                if exit_reason is None and days_held >= hold_days:
                    exit_reason = 'time'

                if exit_reason:
                    cost_out = mkt_val * COMM
                    cash += mkt_val - cost_out
                    trades.append({
                        'pnl_pct': pnl_pct, 'pnl_abs': pnl,
                        'days': days_held, 'di': di, 'year': year,
                        'sym': pos['sym'], 'dir': pos['dir'], 'reason': exit_reason,
                        'strategy': pos.get('strategy', ''),
                    })
                else:
                    new_positions.append(pos)

            positions = new_positions

            # === Open new positions ===
            n_open = len(positions)
            if n_open < max_positions:
                slots = max_positions - n_open

                # Count positions per group
                group_count = {}
                for pos in positions:
                    grp = si_group.get(pos['si'])
                    if grp:
                        group_count[grp] = group_count.get(grp, 0) + 1

                # Score all commodities, merge across strategies
                scored = []
                for si in range(NS):
                    sym = syms[si]

                    # Skip if already holding
                    if any(p['sym'] == sym for p in positions):
                        continue

                    # Check group limit
                    grp = si_group.get(si)
                    if grp and max_per_group > 0 and group_count.get(grp, 0) >= max_per_group:
                        continue

                    # Merge scores across all strategy functions
                    merged = 0.0
                    total_w = 0.0
                    best_strategy = ''
                    best_sc = 0.0
                    for sfn, w in score_fns:
                        sc = sfn(si, di)
                        if not np.isnan(sc):
                            merged += sc * w
                            total_w += w
                            if abs(sc) > best_sc:
                                best_sc = abs(sc)
                                best_strategy = sfn.__name__ if hasattr(sfn, '__name__') else ''

                    if total_w == 0:
                        continue
                    merged /= total_w
                    if merged <= 0.01:
                        continue

                    scored.append((si, merged, sym, best_strategy))

                if scored:
                    scored.sort(key=lambda x: -x[1])
                    # Take top slots, respecting group limits
                    opened_groups = dict(group_count)  # copy current counts
                    taken = 0
                    for best_si, best_sc, best_sym, best_strat in scored:
                        if taken >= slots:
                            break

                        grp = si_group.get(best_si)
                        if grp and max_per_group > 0 and opened_groups.get(grp, 0) >= max_per_group:
                            continue

                        c = C[best_si, di]
                        if np.isnan(c) or c <= 0:
                            continue
                        mult = MULT.get(best_sym, DEF_MULT)
                        notional = c * mult
                        if notional <= 0:
                            continue

                        cash_per_slot = cash / (slots - taken) if (slots - taken) > 0 else cash
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
                        trail_price = c - trail_atr * atr_val
                        positions.append({
                            'si': best_si, 'entry': c, 'entry_di': di,
                            'lots': lots, 'dir': 1, 'sym': best_sym,
                            'atr': atr_val, 'trail_price': trail_price,
                            'strategy': best_strat,
                        })
                        if grp:
                            opened_groups[grp] = opened_groups.get(grp, 0) + 1
                        taken += 1

        # Close remaining positions
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
                'strategy': pos.get('strategy', ''),
            })

        if len(trades) < 10:
            return None

        # Stats
        equity = float(CASH0); peak = float(CASH0); max_dd = 0
        for t in sorted(trades, key=lambda x: x['di']):
            equity += t['pnl_abs']
            if equity > peak: peak = equity
            if peak > 0:
                dd = (peak - equity) / peak * 100
                if dd > max_dd: max_dd = dd

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
    # DEFINE CONFIGS
    # ========================================
    print("\n[Backtest] Building configurations...", flush=True)
    configs = []

    # --- Approach 1: Single signal (group lag), multi-position ---
    for lag in [5]:
        for top_n in [1, 2, 3, 5, 8]:
            for hold in [3, 5]:
                for trail in [2.5, 3.0]:
                    fn = lambda si, di, _l=lag: score_group_lag(si, di, lag=_l)
                    fn.__name__ = f'lag{lag}'
                    configs.append((
                        [(fn, 1.0)],
                        f"A1_LAG{lag}_N{top_n}_H{hold}_TR{trail:.0f}",
                        top_n, 1, hold, trail, None
                    ))

    # --- Approach 1b: Different lag lookbacks ---
    for lag in [3, 7, 10]:
        for top_n in [3, 5]:
            fn = lambda si, di, _l=lag: score_group_lag(si, di, lag=_l)
            fn.__name__ = f'lag{lag}'
            configs.append((
                [(fn, 1.0)],
                f"A1_LAG{lag}_N{top_n}_H3_TR3",
                top_n, 1, 3, 3.0, None
            ))

    # --- Approach 2: Multi-strategy portfolio ---
    # LAG5 + VDP_EMA
    for top_n in [2, 3, 5]:
        for hold in [3, 5]:
            for trail in [2.5, 3.0]:
                fn_lag = lambda si, di: score_group_lag(si, di, lag=5)
                fn_lag.__name__ = 'lag5'
                fn_vdp = lambda si, di: score_vdp_ema(si, di)
                fn_vdp.__name__ = 'vdp'
                configs.append((
                    [(fn_lag, 1.0), (fn_vdp, 1.0)],
                    f"A2_LAG5+VDP_N{top_n}_H{hold}_TR{trail:.0f}",
                    top_n, 1, hold, trail, None
                ))

    # LAG5 + OI surge
    for top_n in [2, 3, 5]:
        for hold in [3, 5]:
            fn_lag = lambda si, di: score_group_lag(si, di, lag=5)
            fn_lag.__name__ = 'lag5'
            fn_oi = lambda si, di: score_oi_surge(si, di)
            fn_oi.__name__ = 'oi'
            configs.append((
                [(fn_lag, 1.0), (fn_oi, 1.0)],
                f"A2_LAG5+OI_N{top_n}_H{hold}_TR3",
                top_n, 1, hold, 3.0, None
            ))

    # LAG5 + VDP + OI (triple combo)
    for top_n in [3, 5]:
        for hold in [3, 5]:
            for w_lag, w_vdp, w_oi in [(1.0, 1.0, 1.0), (2.0, 1.0, 1.0), (1.0, 1.0, 2.0)]:
                fn_lag = lambda si, di: score_group_lag(si, di, lag=5)
                fn_lag.__name__ = 'lag5'
                fn_vdp = lambda si, di: score_vdp_ema(si, di)
                fn_vdp.__name__ = 'vdp'
                fn_oi = lambda si, di: score_oi_surge(si, di)
                fn_oi.__name__ = 'oi'
                configs.append((
                    [(fn_lag, w_lag), (fn_vdp, w_vdp), (fn_oi, w_oi)],
                    f"A2_LAG5+VDP+OI_N{top_n}_H{hold}_W{w_lag:.0f}{w_vdp:.0f}{w_oi:.0f}",
                    top_n, 1, hold, 3.0, None
                ))

    # --- Approach 3: Raw signal (no filters) aggressive with more positions ---
    # Use max_per_group=0 (no limit) vs max_per_group=1 (diversified)
    for top_n in [3, 5, 8]:
        for hold in [3, 5]:
            for mpg in [0, 1]:
                fn = lambda si, di: score_group_lag(si, di, lag=5)
                fn.__name__ = 'lag5_raw'
                configs.append((
                    [(fn, 1.0)],
                    f"A3_LAG5_N{top_n}_H{hold}_MPG{mpg}",
                    top_n, mpg, hold, 3.0, None
                ))

    # Walk-forward for best approach 1 configs
    for top_n in [3, 5]:
        for hold in [3, 5]:
            for wf_year in [2022, 2023, 2024]:
                fn = lambda si, di: score_group_lag(si, di, lag=5)
                fn.__name__ = 'lag5'
                configs.append((
                    [(fn, 1.0)],
                    f"WF_A1_LAG5_N{top_n}_H{hold}_TR3_WF{wf_year}",
                    top_n, 1, hold, 3.0, wf_year
                ))

    # Walk-forward for multi-strategy
    for top_n in [3, 5]:
        for wf_year in [2022, 2023, 2024]:
            fn_lag = lambda si, di: score_group_lag(si, di, lag=5)
            fn_lag.__name__ = 'lag5'
            fn_vdp = lambda si, di: score_vdp_ema(si, di)
            fn_vdp.__name__ = 'vdp'
            configs.append((
                [(fn_lag, 1.0), (fn_vdp, 1.0)],
                f"WF_A2_LAG5+VDP_N{top_n}_H3_WF{wf_year}",
                top_n, 1, 3, 3.0, wf_year
            ))

    print(f"  {len(configs)} total configurations", flush=True)

    # ========================================
    # RUN ALL CONFIGS
    # ========================================
    print("\n[Backtest] Running all configs...", flush=True)
    results = []

    for ci, (score_fns, name, mp, mpg, hold, trail, wf) in enumerate(configs):
        r = run_backtest_multi(score_fns, name, max_positions=mp, max_per_group=mpg,
                               hold_days=hold, trail_atr=trail, wf_split_year=wf)
        if r and r['ann'] > 0:
            results.append(r)

        if (ci + 1) % 20 == 0:
            print(f"  [{ci+1}/{len(configs)}] {len(results)} profitable", flush=True)

    # ========================================
    # RESULTS
    # ========================================
    results.sort(key=lambda x: -x['ann'])

    # Separate walk-forward from full-period
    wf_results = [r for r in results if r['name'].startswith('WF_')]
    full_results = [r for r in results if not r['name'].startswith('WF_')]

    # Also separate by approach
    a1_results = [r for r in full_results if r['name'].startswith('A1_')]
    a2_results = [r for r in full_results if r['name'].startswith('A2_')]
    a3_results = [r for r in full_results if r['name'].startswith('A3_')]

    print(f"\n{'=' * 120}")
    print(f"  TOP 20 FULL-PERIOD RESULTS")
    print(f"{'=' * 120}")
    print(f"  {'Strategy':50s} | {'Ann':>7s} | {'WR':>5s} | {'N':>4s} | {'DD':>6s} | "
          f"{'PF':>4s} | {'AvgW':>7s} | {'AvgL':>6s} | {'AvgD':>4s} | {'Final':>12s}")
    print(f"  {'-' * 120}")
    for r in full_results[:20]:
        print(f"  {r['name']:50s} | {r['ann']:+7.1f}% | {r['wr']:5.1f}% | "
              f"{r['n']:4d} | {r['dd']:6.1f}% | {r['pf']:4.2f} | "
              f"{r['avg_win']:+6.2f}% | {r['avg_loss']:5.2f}% | "
              f"{r['avg_days']:4.1f} | {r['cash']:>12.0f}")

    # Best per approach
    for label, res_list in [("Approach 1 (single signal, multi-position)", a1_results),
                            ("Approach 2 (multi-strategy)", a2_results),
                            ("Approach 3 (raw signal, aggressive)", a3_results)]:
        if res_list:
            best = res_list[0]
            print(f"\n  Best {label}:")
            print(f"    {best['name']:50s}  Ann={best['ann']:+.1f}%  WR={best['wr']:.1f}%  "
                  f"N={best['n']}  DD={best['dd']:.1f}%  PF={best['pf']:.2f}")

    # Walk-forward results
    if wf_results:
        wf_results.sort(key=lambda x: -x['ann'])
        print(f"\n  TOP 10 WALK-FORWARD RESULTS (out-of-sample)")
        print(f"  {'-' * 110}")
        for r in wf_results[:10]:
            print(f"  {r['name']:50s} | {r['ann']:+7.1f}% | {r['wr']:5.1f}% | "
                  f"{r['n']:4d} | {r['dd']:6.1f}% | {r['pf']:4.2f}")

    # ========================================
    # DEEP DIVE ON BEST CONFIG
    # ========================================
    if full_results:
        best = full_results[0]
        print(f"\n{'=' * 120}")
        print(f"  DEEP DIVE: {best['name']}")
        print(f"  Ann={best['ann']:+.1f}%  WR={best['wr']:.1f}%  N={best['n']}  "
              f"DD={best['dd']:.1f}%  PF={best['pf']:.2f}  Final={best['cash']:.0f}")
        print(f"{'=' * 120}")

        print(f"\n  EXIT REASON BREAKDOWN:")
        for reason, s in sorted(best['reasons'].items(), key=lambda x: -x[1]['n']):
            rwr = s['w'] / max(s['n'], 1) * 100
            print(f"    {reason:12s}: {s['n']:4d} trades  WR={rwr:5.1f}%  PnL={s['pnl']:+.1f}%")

        print(f"\n  YEARLY BREAKDOWN:")
        for y in sorted(best['yearly'].keys()):
            s = best['yearly'][y]
            wr = s['w'] / max(s['n'], 1) * 100
            print(f"    {y}: {s['n']:3d} trades  WR={wr:5.1f}%  PnL={s['pnl']:+.1f}%")

        print(f"\n  GROUP BREAKDOWN:")
        for g in sorted(best['grp_counts'].keys(), key=lambda x: -best['grp_counts'][x]['n']):
            gs = best['grp_counts'][g]
            wr_g = gs['w'] / max(gs['n'], 1) * 100
            print(f"    {g:15s}: {gs['n']:3d}t  WR={wr_g:5.1f}%  Abs={gs['pnl']:+.0f}")

    # ========================================
    # YEARLY BREAKDOWN FOR TOP 5
    # ========================================
    if len(full_results) >= 2:
        print(f"\n  YEARLY BREAKDOWN FOR TOP 5:")
        for rank, r in enumerate(full_results[:5]):
            print(f"\n  #{rank+1}: {r['name']} (Ann={r['ann']:+.1f}%, WR={r['wr']:.1f}%, DD={r['dd']:.1f}%)")
            for y in sorted(r['yearly'].keys()):
                ys = r['yearly'][y]
                wr_y = ys['w'] / ys['n'] * 100 if ys['n'] > 0 else 0
                print(f"    {y}: {ys['n']:3d}t  WR={wr_y:5.1f}%  PnL={ys['pnl']:+.1f}%")

    # ========================================
    # MULTI-POSITION SCALING ANALYSIS
    # ========================================
    print(f"\n{'=' * 120}")
    print(f"  SCALING ANALYSIS: How performance changes with position count")
    print(f"{'=' * 120}")
    # Find the best config at each position count for approach 1 (LAG5 only)
    for top_n in [1, 2, 3, 5, 8]:
        matching = [r for r in a1_results
                    if f"LAG5_N{top_n}_" in r['name'] and r['name'].endswith('_TR3')]
        if matching:
            best_at_n = matching[0]
            print(f"  N={top_n}: {best_at_n['name']:45s}  Ann={best_at_n['ann']:+.1f}%  "
                  f"WR={best_at_n['wr']:.1f}%  DD={best_at_n['dd']:.1f}%  N_trades={best_at_n['n']}")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.1f}s")
    print("=" * 120)


if __name__ == '__main__':
    main()
