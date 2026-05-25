"""
Alpha Futures V34c — Adaptive Group Momentum Optimizer
======================================================
V34b best: LAG5_N1_H3_TR3 = +86.8% annual, 55.4% WR, 23.5% DD, PF 2.38

Optimizations over V34b:
1. Adaptive lookback window (arXiv 2106.08420, Sharpe +66%):
   - KER > 0.3 + low vol_ratio -> trending -> longer lookback (7-10 days)
   - KER < 0.15 + high vol_ratio -> choppy -> shorter lookback (3-5 days)
   - Medium -> default 5 days
2. KER regime gate: only enter when KER > threshold (skip choppy markets)
3. Sortino-based ranking: divergence / downside_deviation (robust in high-vol)
4. Group trend strength filter: |group_mom| > threshold for clear direction
5. Volatility-adaptive trailing stop:
   - High ATR/vol -> tighter trail (2.0x)
   - Low ATR/vol -> wider trail (3.0x)

Data: alpha_v2.load_all_data(load_oi=True) -> NS, ND, dates, C, O, H, L, V, OI, syms, sym_set
MIN_TRAIN=250, CASH0=500000. Long only. No stop loss. Trailing + time + signal flip exits.
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


def main():
    t_start = time.time()
    print("=" * 120)
    print("Alpha Futures V34c — Adaptive Group Momentum Optimizer")
    print("V34b baseline: LAG5_N1_H3_TR3 = +86.8% ann, 55.4% WR, 23.5% DD, PF 2.38")
    print("New: adaptive lookback, KER gate, Sortino ranking, group trend filter, vol-adaptive trail")
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

    print(f"  {NS} stocks, {ND} days, Groups: {len(group_members)}")

    # ========================================
    # PRECOMPUTE ALL SIGNALS
    # ========================================
    print("\n[Signals] Computing all indicators...", flush=True)
    t0 = time.time()

    # --- Momentum at multiple lookbacks ---
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

    # --- Group momentum (excluding self) at all lookbacks ---
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

    # --- KER (Kaufman Efficiency Ratio, 20-day) ---
    ker = np.full((NS, ND), np.nan)
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
                ker[si, di] = net / total

    # --- ATR (10-day) ---
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

    # --- Volatility ratio: current ATR / 60-day ATR percentile ---
    # Low vol_ratio = current vol is low relative to history -> trending more cleanly
    # High vol_ratio = vol spike -> choppy
    atr60 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(61, ND):
            trs = []
            for dd in range(di - 60, di):
                hi, lo, pc = H[si, dd], L[si, dd], C[si, dd - 1]
                if np.isnan(hi) or np.isnan(lo):
                    continue
                tr = hi - lo
                if not np.isnan(pc):
                    tr = max(tr, abs(hi - pc), abs(lo - pc))
                trs.append(tr)
            if trs:
                atr60[si, di] = np.mean(trs)

    vol_ratio = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(61, ND):
            a10 = atr10[si, di]
            a60 = atr60[si, di]
            if not np.isnan(a10) and not np.isnan(a60) and a60 > 0:
                vol_ratio[si, di] = a10 / a60

    # --- Downside deviation for Sortino ranking (rolling 20-day) ---
    downside_dev = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            rets = []
            for dd in range(di - 19, di + 1):
                c1 = C[si, dd]
                c0 = C[si, dd - 1]
                if not np.isnan(c1) and not np.isnan(c0) and c0 > 0:
                    rets.append((c1 - c0) / c0)
            if len(rets) >= 10:
                neg = [r for r in rets if r < 0]
                if neg:
                    downside_dev[si, di] = np.sqrt(np.mean(np.array(neg) ** 2))
                else:
                    downside_dev[si, di] = 1e-6  # no downside = very safe

    # --- Adaptive lookback selection ---
    # For each (si, di): choose best lookback based on KER + vol_ratio
    adaptive_lookback = np.full((NS, ND), 5, dtype=np.int32)  # default 5
    for si in range(NS):
        for di in range(20, ND):
            k = ker[si, di]
            vr = vol_ratio[si, di]
            if np.isnan(k) or np.isnan(vr):
                adaptive_lookback[si, di] = 5
            elif k > 0.3 and vr < 1.0:
                # Strong trend, low relative vol -> longer lookback
                adaptive_lookback[si, di] = 10
            elif k > 0.25 and vr < 1.2:
                adaptive_lookback[si, di] = 7
            elif k < 0.15 and vr > 1.3:
                # Choppy, high vol -> shorter lookback
                adaptive_lookback[si, di] = 3
            elif k < 0.2 and vr > 1.1:
                adaptive_lookback[si, di] = 3
            else:
                adaptive_lookback[si, di] = 5

    # --- VDP EMA ---
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

    # --- OI EMA trend ---
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

    print(f"  Done ({time.time()-t0:.1f}s)", flush=True)

    # ========================================
    # SCORING FUNCTIONS
    # ========================================

    def make_score(
        # Lookback mode: 'fixed', 'adaptive'
        lookback_mode='fixed',
        fixed_lag=5,
        # Entry threshold
        min_lag=0.003,
        scale=10.0,
        # KER regime gate
        ker_threshold=0.0,       # 0 = disabled, >0 = minimum KER to enter
        # Sortino ranking: divide divergence by downside_dev
        use_sortino=False,
        # Group trend strength filter
        group_trend_threshold=0.0,  # 0 = disabled, >0 = min |group_mom|
        # VDP filter
        require_vdp=False,
        # OI filter
        require_oi=False,
        # Adaptive trailing stop mode
        adaptive_trail=False,
    ):
        """Parameterizable scorer with all V34c optimizations."""
        def score(si, di):
            # --- Determine lookback ---
            if lookback_mode == 'adaptive':
                lag = int(adaptive_lookback[si, di])
            else:
                lag = fixed_lag

            own = mom[lag][si, di]
            grp = grp_mom[lag][si, di]
            if np.isnan(own) or np.isnan(grp):
                return np.nan

            # --- KER regime gate ---
            if ker_threshold > 0:
                k = ker[si, di]
                if np.isnan(k) or k < ker_threshold:
                    return np.nan

            # --- Group trend strength filter ---
            if group_trend_threshold > 0:
                if abs(grp) < group_trend_threshold:
                    return np.nan

            # --- Core divergence ---
            divergence = grp - own
            if abs(divergence) < min_lag:
                return np.nan

            # --- Sortino-based ranking ---
            if use_sortino:
                dd = downside_dev[si, di]
                if np.isnan(dd) or dd <= 0:
                    return np.nan
                sc = divergence / dd
                sc = np.clip(sc * scale * 0.01, -1, 1)  # normalize
            else:
                sc = np.clip(divergence * scale, -1, 1)

            # Only long (positive divergence = group ahead, own catching up)
            if sc <= 0:
                return np.nan

            # --- Optional VDP filter ---
            if require_vdp:
                vd = vdp_ema[si, di]
                if np.isnan(vd):
                    return np.nan
                if vd < 0:
                    return np.nan
                sc *= min(1.0 + abs(vd) / 5e6, 1.5)

            # --- Optional OI filter ---
            if require_oi:
                oi_r = oi_rising[si, di]
                if not np.isnan(oi_r):
                    if oi_r > 0.01:
                        sc *= 1.3
                    elif oi_r < -0.02:
                        sc *= 0.5

            return sc
        return score

    # ========================================
    # BACKTEST ENGINE
    # ========================================
    def run_backtest(score_fn, name, top_n=1, hold_min=2, hold_max=3,
                     trail_atr_mult=2.5, adaptive_trail=False,
                     wf_split_year=None):
        """
        Single/multi-position backtest with optional walk-forward split.
        adaptive_trail: if True, adjust trail multiplier based on vol_ratio.
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

                # Volatility-adaptive trailing stop
                if days_held >= 2:
                    # Determine effective trail mult
                    if adaptive_trail:
                        vr = vol_ratio[pos['si'], di]
                        if not np.isnan(vr):
                            if vr > 1.3:
                                eff_trail = max(trail_atr_mult * 0.7, 1.5)  # tight in high vol
                            elif vr < 0.8:
                                eff_trail = trail_atr_mult * 1.3  # wide in low vol
                            else:
                                eff_trail = trail_atr_mult
                        else:
                            eff_trail = trail_atr_mult
                    else:
                        eff_trail = trail_atr_mult

                    atr = pos.get('atr', 0)
                    if atr > 0 and pos['dir'] == 1 and eff_trail > 0:
                        new_trail = c - eff_trail * atr
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

                        # Initial trail price depends on mode
                        if adaptive_trail:
                            vr = vol_ratio[best_si, di]
                            if not np.isnan(vr):
                                if vr > 1.3:
                                    eff_mult = max(trail_atr_mult * 0.7, 1.5)
                                elif vr < 0.8:
                                    eff_mult = trail_atr_mult * 1.3
                                else:
                                    eff_mult = trail_atr_mult
                            else:
                                eff_mult = trail_atr_mult
                        else:
                            eff_mult = trail_atr_mult

                        trail_price = c - eff_mult * atr_val
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
    # PARAMETER SWEEP
    # ========================================
    print("\n[Backtest] Building configurations...", flush=True)
    results = []
    configs = []

    # ---- (A) BASELINE: Fixed lookback from v34b (reproduce best result) ----
    for lag in [3, 5, 7]:
        for hold_max in [3, 5]:
            for trail in [2.5, 3.0]:
                configs.append((
                    make_score(lookback_mode='fixed', fixed_lag=lag),
                    f"BASE_LAG{lag}_H{hold_max}_TR{trail*10:.0f}",
                    1, 2, hold_max, trail, False, None
                ))

    # ---- (B) ADAPTIVE LOOKBACK (core new feature) ----
    for hold_max in [3, 5, 7]:
        for trail in [2.5, 3.0]:
            configs.append((
                make_score(lookback_mode='adaptive'),
                f"ADAPT_H{hold_max}_TR{trail*10:.0f}",
                1, 2, hold_max, trail, False, None
            ))

    # ---- (C) KER REGIME GATE ----
    for ker_th in [0.15, 0.20, 0.25]:
        for lag in [5, 7]:
            for hold_max in [3, 5]:
                configs.append((
                    make_score(lookback_mode='fixed', fixed_lag=lag, ker_threshold=ker_th),
                    f"KER{ker_th*100:.0f}_LAG{lag}_H{hold_max}",
                    1, 2, hold_max, 2.5, False, None
                ))

    # Adaptive lookback + KER gate
    for ker_th in [0.15, 0.20, 0.25]:
        for hold_max in [3, 5]:
            configs.append((
                make_score(lookback_mode='adaptive', ker_threshold=ker_th),
                f"ADAPT_KER{ker_th*100:.0f}_H{hold_max}",
                1, 2, hold_max, 2.5, False, None
            ))

    # ---- (D) SORTINO-BASED RANKING ----
    for lag in [5, 7]:
        for hold_max in [3, 5]:
            for scale in [8.0, 12.0]:
                configs.append((
                    make_score(lookback_mode='fixed', fixed_lag=lag,
                               use_sortino=True, scale=scale),
                    f"SORT_LAG{lag}_H{hold_max}_S{scale:.0f}",
                    1, 2, hold_max, 2.5, False, None
                ))

    # Adaptive + Sortino
    for hold_max in [3, 5]:
        for scale in [8.0, 12.0]:
            configs.append((
                make_score(lookback_mode='adaptive', use_sortino=True, scale=scale),
                f"ADAPT_SORT_H{hold_max}_S{scale:.0f}",
                1, 2, hold_max, 2.5, False, None
            ))

    # ---- (E) GROUP TREND STRENGTH FILTER ----
    for gt_th in [0.005, 0.01, 0.015]:
        for lag in [5, 7]:
            for hold_max in [3, 5]:
                configs.append((
                    make_score(lookback_mode='fixed', fixed_lag=lag,
                               group_trend_threshold=gt_th),
                    f"GT{gt_th*1000:.0f}_LAG{lag}_H{hold_max}",
                    1, 2, hold_max, 2.5, False, None
                ))

    # Adaptive + group trend
    for gt_th in [0.005, 0.01, 0.015]:
        for hold_max in [3, 5]:
            configs.append((
                make_score(lookback_mode='adaptive', group_trend_threshold=gt_th),
                f"ADAPT_GT{gt_th*1000:.0f}_H{hold_max}",
                1, 2, hold_max, 2.5, False, None
            ))

    # ---- (F) VOLATILITY-ADAPTIVE TRAILING STOP ----
    for lag in [5, 7]:
        for base_trail in [2.5, 3.0]:
            for hold_max in [3, 5]:
                configs.append((
                    make_score(lookback_mode='fixed', fixed_lag=lag),
                    f"VTRAIL_LAG{lag}_H{hold_max}_TR{base_trail*10:.0f}",
                    1, 2, hold_max, base_trail, True, None
                ))

    # Adaptive lookback + adaptive trail
    for hold_max in [3, 5, 7]:
        for base_trail in [2.5, 3.0]:
            configs.append((
                make_score(lookback_mode='adaptive'),
                f"ADAPT_VTRAIL_H{hold_max}_TR{base_trail*10:.0f}",
                1, 2, hold_max, base_trail, True, None
            ))

    # ---- (G) COMBO: Adaptive + KER + Group Trend ----
    for ker_th in [0.15, 0.20]:
        for gt_th in [0.005, 0.01]:
            for hold_max in [3, 5]:
                configs.append((
                    make_score(lookback_mode='adaptive',
                               ker_threshold=ker_th,
                               group_trend_threshold=gt_th),
                    f"ADAPT_KER{ker_th*100:.0f}_GT{gt_th*1000:.0f}_H{hold_max}",
                    1, 2, hold_max, 2.5, False, None
                ))

    # ---- (H) COMBO: Adaptive + Sortino + KER ----
    for ker_th in [0.15, 0.20]:
        for hold_max in [3, 5]:
            configs.append((
                make_score(lookback_mode='adaptive',
                           ker_threshold=ker_th,
                           use_sortino=True, scale=10.0),
                f"ADAPT_SORT_KER{ker_th*100:.0f}_H{hold_max}",
                1, 2, hold_max, 2.5, False, None
            ))

    # ---- (I) COMBO: Adaptive + Sortino + Group Trend + KER + VTrail ----
    for ker_th in [0.15, 0.20]:
        for gt_th in [0.005, 0.01]:
            for hold_max in [3, 5]:
                configs.append((
                    make_score(lookback_mode='adaptive',
                               ker_threshold=ker_th,
                               use_sortino=True, scale=10.0,
                               group_trend_threshold=gt_th),
                    f"ALL_ADAPT_KER{ker_th*100:.0f}_GT{gt_th*1000:.0f}_H{hold_max}_VT",
                    1, 2, hold_max, 2.5, True, None
                ))

    # ---- (J) VDP/OI enhanced combos with adaptive ----
    for lag_mode in ['adaptive']:
        for require_vdp in [True, False]:
            for require_oi in [True, False]:
                if not require_vdp and not require_oi:
                    continue  # already covered above
                for hold_max in [3, 5]:
                    for ker_th in [0.0, 0.15]:
                        vdp_tag = 'VDP' if require_vdp else ''
                        oi_tag = 'OI' if require_oi else ''
                        ker_tag = f"_KER{ker_th*100:.0f}" if ker_th > 0 else ''
                        configs.append((
                            make_score(lookback_mode=lag_mode,
                                       require_vdp=require_vdp,
                                       require_oi=require_oi,
                                       ker_threshold=ker_th),
                            f"ADAPT_{vdp_tag}{oi_tag}{ker_tag}_H{hold_max}",
                            1, 2, hold_max, 2.5, False, None
                        ))

    # ---- (K) Walk-forward validation on promising configs ----
    for wf_year in [2023, 2024]:
        # Baseline
        configs.append((
            make_score(lookback_mode='fixed', fixed_lag=5),
            f"WF{wf_year}_BASE_LAG5_H3_TR25",
            1, 2, 3, 2.5, False, wf_year
        ))
        # Adaptive
        for hold_max in [3, 5]:
            configs.append((
                make_score(lookback_mode='adaptive'),
                f"WF{wf_year}_ADAPT_H{hold_max}",
                1, 2, hold_max, 2.5, False, wf_year
            ))
        # Adaptive + KER + GT
        for hold_max in [3, 5]:
            configs.append((
                make_score(lookback_mode='adaptive', ker_threshold=0.15,
                           group_trend_threshold=0.005),
                f"WF{wf_year}_ADAPT_KER15_GT5_H{hold_max}",
                1, 2, hold_max, 2.5, False, wf_year
            ))
        # Adaptive + Sortino + KER
        configs.append((
            make_score(lookback_mode='adaptive', ker_threshold=0.15,
                       use_sortino=True, scale=10.0),
            f"WF{wf_year}_ADAPT_SORT_KER15_H3",
            1, 2, 3, 2.5, False, wf_year
        ))
        # Full combo
        configs.append((
            make_score(lookback_mode='adaptive', ker_threshold=0.15,
                       use_sortino=True, scale=10.0,
                       group_trend_threshold=0.005),
            f"WF{wf_year}_ALL_KER15_GT5_SORT_H3_VT",
            1, 2, 3, 2.5, True, wf_year
        ))

    print(f"  {len(configs)} configurations", flush=True)

    # ---- Run all configs ----
    print("\n[Backtest] Running sweep...", flush=True)
    for ci, (fn, name, tn, hmin, hmax, trail, ad_trail, wf) in enumerate(configs):
        r = run_backtest(fn, name, top_n=tn, hold_min=hmin, hold_max=hmax,
                         trail_atr_mult=trail, adaptive_trail=ad_trail,
                         wf_split_year=wf)
        if r and r['ann'] > 0:
            results.append(r)
            if r['ann'] > 50:
                parts = []
                for reason, stats in sorted(r['reasons'].items()):
                    wr_r = stats['w'] / stats['n'] * 100 if stats['n'] > 0 else 0
                    parts.append(f"{reason}:{stats['n']}({wr_r:.0f}%)")
                print(f"  {r['name']:50s} | Ann {r['ann']:+7.1f}% | WR {r['wr']:5.1f}% | "
                      f"N {r['n']:4d} | DD {r['dd']:6.1f}% | PF {r['pf']:4.2f} | "
                      f"AvgW {r['avg_win']:+.2f}% | AvgL {r['avg_loss']:.2f}% | AvgD {r['avg_days']:.1f}")
                print(f"  {'':50s} | Exits: {' | '.join(parts)}")

        if (ci + 1) % 100 == 0:
            print(f"  [{ci+1}/{len(configs)}] {len(results)} profitable", flush=True)

    # ========================================
    # RESULTS
    # ========================================
    results.sort(key=lambda x: -x['ann'])

    # Separate walk-forward from full-period
    wf_results = [r for r in results if r['name'].startswith('WF')]
    full_results = [r for r in results if not r['name'].startswith('WF')]

    print(f"\n{'=' * 120}")
    print(f"  TOP 20 FULL-PERIOD RESULTS")
    print(f"{'=' * 120}")
    print(f"  {'Strategy':50s} | {'Ann':>7s} | {'WR':>5s} | {'N':>4s} | {'DD':>6s} | "
          f"{'PF':>4s} | {'AvgW':>7s} | {'AvgL':>6s} | {'AvgD':>4s}")
    print(f"  {'-' * 120}")
    for r in full_results[:20]:
        print(f"  {r['name']:50s} | {r['ann']:+7.1f}% | {r['wr']:5.1f}% | "
              f"{r['n']:4d} | {r['dd']:6.1f}% | {r['pf']:4.2f} | "
              f"{r['avg_win']:+6.2f}% | {r['avg_loss']:5.2f}% | "
              f"{r['avg_days']:4.1f}")

    # Group results by optimization type for analysis
    print(f"\n  BEST BY CATEGORY:")
    categories = {
        'BASE': 'Baseline (fixed lookback)',
        'ADAPT': 'Adaptive lookback',
        'KER': 'KER regime gate',
        'SORT': 'Sortino ranking',
        'GT': 'Group trend filter',
        'VTRAIL': 'Vol-adaptive trail',
        'ALL': 'Full combo',
    }
    for prefix, desc in categories.items():
        cat_results = [r for r in full_results if r['name'].startswith(prefix)]
        if cat_results:
            best_cat = cat_results[0]
            print(f"    {desc:30s} -> {best_cat['name']:50s} | Ann {best_cat['ann']:+.1f}% | WR {best_cat['wr']:.1f}% | DD {best_cat['dd']:.1f}% | PF {best_cat['pf']:.2f}")

    # Walk-forward results
    if wf_results:
        wf_results.sort(key=lambda x: -x['ann'])
        print(f"\n  WALK-FORWARD RESULTS (out-of-sample)")
        print(f"  {'-' * 120}")
        for r in wf_results[:20]:
            print(f"  {r['name']:55s} | Ann {r['ann']:+7.1f}% | WR {r['wr']:5.1f}% | "
                  f"N {r['n']:4d} | DD {r['dd']:6.1f}% | PF {r['pf']:4.2f}")

    # Best config details
    if full_results:
        best = full_results[0]
        print(f"\n  BEST: {best['name']}")
        print(f"  Ann={best['ann']:+.1f}%  WR={best['wr']:.1f}%  N={best['n']}  "
              f"DD={best['dd']:.1f}%  PF={best['pf']:.2f}  Final={best['cash']:.0f}")

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
        for idx, r in enumerate(full_results[:5]):
            print(f"\n  #{idx+1}: {r['name']} (Ann={r['ann']:+.1f}%, WR={r['wr']:.1f}%, DD={r['dd']:.1f}%)")
            for y in sorted(r['yearly'].keys()):
                ys = r['yearly'][y]
                wr_y = ys['w'] / ys['n'] * 100 if ys['n'] > 0 else 0
                print(f"    {y}: {ys['n']:3d}t  WR={wr_y:5.1f}%  PnL={ys['pnl']:+.1f}%")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.1f}s")
    print("=" * 120)


if __name__ == '__main__':
    main()
