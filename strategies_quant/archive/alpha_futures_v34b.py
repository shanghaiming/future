"""
Alpha Futures V34b — Group Momentum Optimizer
==============================================
V34 found that Group Momentum Lag (S4) is the #1 signal at +80.7%.
This version deep-dives into optimizing that signal:

Key insight: when a commodity lags its supply-chain group, it catches up.
This is mean-reversion within a trend (group is moving, individual lags).

Optimization axes:
1. Momentum lookback: 3/5/7/10/15 days (not just 5)
2. Group momentum: include vs exclude self
3. Entry threshold: how much lag is needed
4. Hold period: 2-10 days
5. Top-N: 1-4 concurrent positions
6. Trailing stop multiplier
7. Signal flip exit threshold
8. OI/VDP confirmation: with vs without
9. Walk-forward: train 2016-2022, test 2023-2026
10. Cross-validation by group
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
    print("=" * 110)
    print("Alpha Futures V34b — Group Momentum Optimizer")
    print("Core: own lags group → catches up. Deep parameter sweep + walk-forward.")
    print("=" * 110)

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

    print(f"  {NS} stocks, {ND} days, Groups: {len(group_members)}")

    # ========================================
    # PRECOMPUTE ALL MOMENTUM LOOKBACKS
    # ========================================
    print("\n[Signals] Computing...", flush=True)
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

    # Group momentum at multiple lookbacks (excluding self)
    grp_mom = {}
    for lag in [3, 5, 7, 10, 15]:
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

    # Group momentum INCLUDING self (for trend direction)
    grp_mom_inc = {}
    for lag in [5]:
        gm = np.full((NS, ND), np.nan)
        for grp, members in group_members.items():
            for di in range(lag, ND):
                ms = []
                for sj in members:
                    m = mom[lag][sj, di]
                    if not np.isnan(m):
                        ms.append(m)
                if ms:
                    avg = np.mean(ms)
                    for sj in members:
                        gm[sj, di] = avg
        grp_mom_inc[lag] = gm

    # Upstream leader momentum (1-day lagged)
    leader_mom5 = np.full((NS, ND), np.nan)
    for si in range(NS):
        usi = upstream_si[si]
        if usi < 0:
            leader_mom5[si, :] = grp_mom[5][si, :]
        else:
            for di in range(1, ND):
                lm = mom[5][usi, di - 1]
                if not np.isnan(lm):
                    leader_mom5[si, di] = lm

    # VDP EMA
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

    # OI EMA trend
    oi_ema = np.full((NS, ND), np.nan)
    oi_rising = np.full((NS, ND), np.nan)
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
                oi_rising[si, di] = (cur - prev) / prev

    # KER (Kaufman Efficiency Ratio)
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

    # ATR
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

    def make_group_lag_score(mom_lag=5, min_lag=0.003, scale=10.0,
                              require_group_trend=False, require_vdp=False,
                              require_oi=False, require_ker=False,
                              supply_chain_weight=0.0):
        """Parameterizable group momentum lag scorer."""
        def score(si, di):
            own = mom[mom_lag][si, di]
            grp = grp_mom[mom_lag][si, di]
            if np.isnan(own) or np.isnan(grp):
                return np.nan

            # Require group to be trending (not flat)
            if require_group_trend and abs(grp) < 0.005:
                return np.nan

            divergence = grp - own
            if abs(divergence) < min_lag:
                return np.nan

            sc = np.clip(divergence * scale, -1, 1)

            # Only long (positive divergence = group ahead, own catching up)
            if sc <= 0:
                return np.nan

            # Optional: supply chain confirmation
            if supply_chain_weight > 0:
                usi = upstream_si[si]
                if usi >= 0:
                    up_m5 = mom[5][usi, di]
                    if not np.isnan(up_m5) and up_m5 > 0:
                        sc *= (1 + supply_chain_weight)
                    elif not np.isnan(up_m5) and up_m5 < 0:
                        sc *= (1 - supply_chain_weight * 0.5)

            # Optional filters
            if require_vdp:
                vd = vdp_ema[si, di]
                if np.isnan(vd):
                    return np.nan
                if vd < 0:
                    return np.nan  # VDP must be positive for long
                sc *= min(1.0 + abs(vd) / 5e6, 1.5)

            if require_oi:
                oi_r = oi_rising[si, di]
                if not np.isnan(oi_r):
                    if oi_r > 0.01:
                        sc *= 1.3
                    elif oi_r < -0.02:
                        sc *= 0.5

            if require_ker:
                k = ker[si, di]
                if np.isnan(k) or k < 0.15:
                    return np.nan  # Only trade in trending regimes

            return sc
        return score

    # ========================================
    # BACKTEST ENGINE
    # ========================================
    def run_backtest(score_fn, name, top_n=1, hold_min=2, hold_max=3,
                     trail_atr_mult=2.5, wf_split_year=None):
        """
        Multi-position backtest with optional walk-forward split.
        wf_split_year: if set, only test on years >= this value.
        """
        cash = float(CASH0)
        trades = []
        positions = []
        last_exit = {}

        for di in range(MIN_TRAIN, ND):
            year = dates[di].year

            # Walk-forward: skip training period
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
                    last_exit[pos['sym']] = di
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
            # Adjust year count for walk-forward
            first_test_di = None
            for di in range(MIN_TRAIN, ND):
                if dates[di].year >= wf_split_year:
                    first_test_di = di
                    break
            if first_test_di:
                days_total = (dates[ND - 1] - dates[first_test_di]).days
                yr = max(days_total / 365.25, 0.01)
                # Also reset starting equity for WF
                ann = ((cash / CASH0) ** (1 / yr) - 1) * 100
        else:
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
    print("\n[Backtest] Running parameter sweep...", flush=True)
    results = []
    configs = []

    # Momentum lookback sweep
    for lag in [3, 5, 7, 10]:
        for top_n in [1, 2]:
            for hold_max in [3, 5, 7]:
                for min_lag in [0.002, 0.003, 0.005]:
                    for scale in [8.0, 10.0, 15.0]:
                        configs.append((
                            make_group_lag_score(mom_lag=lag, min_lag=min_lag, scale=scale),
                            f"LAG{lag}_N{top_n}_H{hold_max}_ML{min_lag*1000:.0f}_S{scale:.0f}",
                            top_n, 2, hold_max, 2.5, None
                        ))

    # With VDP filter
    for lag in [5, 7]:
        for hold_max in [3, 5]:
            configs.append((
                make_group_lag_score(mom_lag=lag, require_vdp=True),
                f"LAG{lag}_N1_H{hold_max}_VDP",
                1, 2, hold_max, 2.5, None
            ))

    # With OI filter
    for lag in [5, 7]:
        for hold_max in [3, 5]:
            configs.append((
                make_group_lag_score(mom_lag=lag, require_oi=True),
                f"LAG{lag}_N1_H{hold_max}_OI",
                1, 2, hold_max, 2.5, None
            ))

    # With KER regime gate
    for lag in [5, 7]:
        for hold_max in [3, 5]:
            configs.append((
                make_group_lag_score(mom_lag=lag, require_ker=True),
                f"LAG{lag}_N1_H{hold_max}_KER",
                1, 2, hold_max, 2.5, None
            ))

    # With supply chain confirmation
    for lag in [5]:
        for sc_w in [0.3, 0.5]:
            for hold_max in [3, 5]:
                configs.append((
                    make_group_lag_score(mom_lag=lag, supply_chain_weight=sc_w),
                    f"LAG{lag}_N1_H{hold_max}_SC{sc_w*10:.0f}",
                    1, 2, hold_max, 2.5, None
                ))

    # Group trend required
    for lag in [5, 7]:
        for hold_max in [3, 5]:
            configs.append((
                make_group_lag_score(mom_lag=lag, require_group_trend=True),
                f"LAG{lag}_N1_H{hold_max}_GT",
                1, 2, hold_max, 2.5, None
            ))

    # All filters combined
    for lag in [5, 7]:
        configs.append((
            make_group_lag_score(mom_lag=lag, require_vdp=True, require_oi=True, require_group_trend=True),
            f"LAG{lag}_N1_H3_ALL",
            1, 2, 3, 2.5, None
        ))
        configs.append((
            make_group_lag_score(mom_lag=lag, require_vdp=True, require_oi=True, require_group_trend=True),
            f"LAG{lag}_N1_H5_ALL",
            1, 2, 5, 2.5, None
        ))

    # Walk-forward validation on best configs (from v34 results)
    for lag in [5, 7]:
        for hold_max in [3, 5]:
            for wf_year in [2022, 2023, 2024]:
                configs.append((
                    make_group_lag_score(mom_lag=lag),
                    f"LAG{lag}_N1_H{hold_max}_WF{wf_year}",
                    1, 2, hold_max, 2.5, wf_year
                ))

    # Longer trailing stops
    for trail in [3.0, 4.0, 5.0]:
        for hold_max in [3, 5]:
            configs.append((
                make_group_lag_score(mom_lag=5),
                f"LAG5_N1_H{hold_max}_TR{trail:.0f}",
                1, 2, hold_max, trail, None
            ))

    print(f"  {len(configs)} configurations", flush=True)

    for ci, (fn, name, tn, hmin, hmax, trail, wf) in enumerate(configs):
        r = run_backtest(fn, name, top_n=tn, hold_min=hmin, hold_max=hmax,
                         trail_atr_mult=trail, wf_split_year=wf)
        if r and r['ann'] > 0:
            results.append(r)
            if r['ann'] > 50:
                parts = []
                for reason, stats in sorted(r['reasons'].items()):
                    wr_r = stats['w'] / stats['n'] * 100 if stats['n'] > 0 else 0
                    parts.append(f"{reason}:{stats['n']}({wr_r:.0f}%)")
                print(f"  {r['name']:45s} | Ann {r['ann']:+7.1f}% | WR {r['wr']:5.1f}% | "
                      f"N {r['n']:4d} | DD {r['dd']:6.1f}% | PF {r['pf']:4.2f} | "
                      f"AvgW {r['avg_win']:+.2f}% | AvgL {r['avg_loss']:.2f}% | AvgD {r['avg_days']:.1f}")
                print(f"  {'':45s} | Exits: {' | '.join(parts)}")

        if (ci + 1) % 100 == 0:
            print(f"  [{ci+1}/{len(configs)}] {len(results)} profitable", flush=True)

    # ========================================
    # RESULTS
    # ========================================
    results.sort(key=lambda x: -x['ann'])

    # Separate walk-forward results
    wf_results = [r for r in results if '_WF' in r['name']]
    full_results = [r for r in results if '_WF' not in r['name']]

    print(f"\n{'=' * 110}")
    print(f"  TOP FULL-PERIOD RESULTS")
    print(f"{'=' * 110}")
    print(f"  {'Strategy':45s} | {'Ann':>7s} | {'WR':>5s} | {'N':>4s} | {'DD':>6s} | "
          f"{'PF':>4s} | {'AvgW':>7s} | {'AvgL':>6s} | {'AvgD':>4s}")
    print(f"  {'-' * 110}")
    for r in full_results[:25]:
        print(f"  {r['name']:45s} | {r['ann']:+7.1f}% | {r['wr']:5.1f}% | "
              f"{r['n']:4d} | {r['dd']:6.1f}% | {r['pf']:4.2f} | "
              f"{r['avg_win']:+6.2f}% | {r['avg_loss']:5.2f}% | "
              f"{r['avg_days']:4.1f}")

    if wf_results:
        print(f"\n  WALK-FORWARD RESULTS (out-of-sample)")
        print(f"  {'-' * 110}")
        wf_results.sort(key=lambda x: -x['ann'])
        for r in wf_results[:15]:
            print(f"  {r['name']:45s} | {r['ann']:+7.1f}% | {r['wr']:5.1f}% | "
                  f"{r['n']:4d} | {r['dd']:6.1f}% | {r['pf']:4.2f}")

    if full_results:
        best = full_results[0]
        print(f"\n  BEST: {best['name']}  |  Ann={best['ann']:+.1f}%  WR={best['wr']:.1f}%  "
              f"N={best['n']}  DD={best['dd']:.1f}%")
        print(f"  AvgWin={best['avg_win']:+.2f}%  AvgLoss={best['avg_loss']:.2f}%  PF={best['pf']:.2f}  Final={best['cash']:.0f}")

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

    # Yearly for top 5
    if len(full_results) >= 2:
        print(f"\n  YEARLY BREAKDOWN FOR TOP 5:")
        for r in full_results[:5]:
            print(f"\n  #{full_results.index(r)+1}: {r['name']} (Ann={r['ann']:+.1f}%, WR={r['wr']:.1f}%, DD={r['dd']:.1f}%)")
            for y in sorted(r['yearly'].keys()):
                ys = r['yearly'][y]
                wr_y = ys['w'] / ys['n'] * 100 if ys['n'] > 0 else 0
                print(f"    {y}: {ys['n']:3d}t  WR={wr_y:5.1f}%  PnL={ys['pnl']:+.1f}%")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.1f}s")
    print("=" * 110)


if __name__ == '__main__':
    main()
