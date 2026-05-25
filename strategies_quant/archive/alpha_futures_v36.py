"""
Alpha Futures V36 — HAR-RV Volatility Timing Strategy
=====================================================
Implements Corsi (2009) Heterogeneous Autoregressive Realized Volatility model
to predict next-day volatility and adjust position sizing accordingly.

Core idea:
  RV_t+1 = b0 + b1*RV_daily + b2*RV_weekly + b3*RV_monthly + e

  - Predicted vol > long-term avg (EXPANSION)  --> reduce / skip position
  - Predicted vol < long-term avg (CONTRACTION) --> increase / full position
  - Combined with proven Group Momentum Lag signal from v34b

Configurations tested:
  1. Vol expansion thresholds: 1.2, 1.3, 1.5
  2. Vol contraction thresholds: 0.7, 0.8
  3. Position multipliers: (0.0, 1.0), (0.5, 1.0), (0.5, 1.2)
  4. Hold periods: 3, 5, 7
  5. Trail: 2.5, 3.0
  6. Pure vol timing (no group mom) vs combined
  7. Skip expansion vs half-position in expansion
  8. v34b baseline for comparison
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


def _ols_fit(X, y):
    """Simple OLS: solve (X'X) beta = X'y using np.linalg.lstsq."""
    # X shape (n, 4), y shape (n,)
    # Returns beta (4,) or None
    try:
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        return beta
    except Exception:
        return None


def main():
    t_start = time.time()
    print("=" * 120)
    print("Alpha Futures V36 — HAR-RV Volatility Timing + Group Momentum Lag")
    print("Corsi (2009) realized vol prediction for position sizing")
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

    print(f"  {NS} stocks, {ND} days, Groups: {len(group_members)}")

    # ========================================
    # 1. REALIZED VOLATILITY COMPONENTS
    # ========================================
    print("\n[HAR-RV] Computing realized volatility components...", flush=True)
    t0 = time.time()

    # RV_daily = squared daily log return
    rv_daily = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(1, ND):
            c_now = C[si, di]
            c_prev = C[si, di - 1]
            if not np.isnan(c_now) and not np.isnan(c_prev) and c_prev > 0 and c_now > 0:
                rv_daily[si, di] = (np.log(c_now) - np.log(c_prev)) ** 2

    # RV_weekly = rolling 5-day mean of RV_daily
    rv_weekly = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            window = rv_daily[si, di - 5:di]
            valid = window[~np.isnan(window)]
            if len(valid) >= 3:
                rv_weekly[si, di] = np.mean(valid)

    # RV_monthly = rolling 20-day mean of RV_daily
    rv_monthly = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            window = rv_daily[si, di - 20:di]
            valid = window[~np.isnan(window)]
            if len(valid) >= 10:
                rv_monthly[si, di] = np.mean(valid)

    print(f"  RV components done ({time.time()-t0:.1f}s)", flush=True)

    # ========================================
    # 2. HAR-RV ROLLING PREDICTION
    # ========================================
    print("  Running rolling HAR-RV OLS predictions...", flush=True)
    t1 = time.time()

    # predicted_vol[si, di] = E[RV_{t+1} | info at time t]
    # actual_vol[si, di] = RV_daily[si, di+1] (for diagnostics)
    predicted_vol = np.full((NS, ND), np.nan)
    actual_vol = np.full((NS, ND), np.nan)
    MIN_HAR_TRAIN = 60  # minimum samples for OLS

    for si in range(NS):
        # Collect all valid data points for this symbol
        # X = [1, RV_daily, RV_weekly, RV_monthly] at time t
        # y = RV_daily at time t+1
        xs = []
        ys = []
        for di in range(20, ND - 1):
            rd = rv_daily[si, di]
            rw = rv_weekly[si, di]
            rm = rv_monthly[si, di]
            rv_next = rv_daily[si, di + 1]
            if np.isnan(rd) or np.isnan(rw) or np.isnan(rm) or np.isnan(rv_next):
                continue
            xs.append([1.0, rd, rw, rm])
            ys.append(rv_next)

        if len(xs) < MIN_HAR_TRAIN:
            continue

        xs = np.array(xs)
        ys = np.array(ys)

        # We need to map back to day indices
        # Build day-indexed arrays for rolling OLS
        day_indices = []
        features = []
        targets = []
        for di in range(20, ND - 1):
            rd = rv_daily[si, di]
            rw = rv_weekly[si, di]
            rm = rv_monthly[si, di]
            rv_next = rv_daily[si, di + 1]
            if np.isnan(rd) or np.isnan(rw) or np.isnan(rm) or np.isnan(rv_next):
                continue
            day_indices.append(di)
            features.append([1.0, rd, rw, rm])
            targets.append(rv_next)

        features = np.array(features)
        targets = np.array(targets)

        # Expanding window OLS
        for idx in range(MIN_HAR_TRAIN, len(day_indices)):
            di_pred = day_indices[idx]  # day index for this prediction

            # Train on all data up to idx-1, predict for day_indices[idx]
            X_train = features[:idx]
            y_train = targets[:idx]

            beta = _ols_fit(X_train, y_train)
            if beta is not None:
                x_pred = features[idx]
                pred = x_pred @ beta
                if pred > 0:
                    predicted_vol[si, di_pred] = pred

            actual_vol[si, di_pred] = targets[idx]

    n_pred = np.sum(~np.isnan(predicted_vol))
    print(f"  HAR-RV predictions: {n_pred} valid entries ({time.time()-t1:.1f}s)", flush=True)

    # Quick HAR-RV accuracy diagnostic
    pred_valid = ~np.isnan(predicted_vol) & ~np.isnan(actual_vol)
    if np.sum(pred_valid) > 100:
        errs = np.abs(predicted_vol[pred_valid] - actual_vol[pred_valid])
        mean_actual = np.mean(actual_vol[pred_valid])
        if mean_actual > 0:
            mape = np.mean(errs / actual_vol[pred_valid]) * 100
            print(f"  HAR-RV MAPE: {mape:.1f}%, mean actual RV: {mean_actual:.6f}")

    # ========================================
    # 3. VOLATILITY REGIME CLASSIFICATION
    # ========================================
    # vol_ratio = predicted_vol / RV_monthly
    vol_ratio = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            pv = predicted_vol[si, di]
            rm = rv_monthly[si, di]
            if not np.isnan(pv) and not np.isnan(rm) and rm > 0:
                vol_ratio[si, di] = pv / rm

    print(f"  Vol ratios computed ({time.time()-t0:.1f}s total HAR-RV)", flush=True)

    # ========================================
    # 4. GROUP MOMENTUM SIGNALS (from v34b)
    # ========================================
    print("\n[Signals] Computing group momentum...", flush=True)
    t2 = time.time()

    mom5 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            c_now = C[si, di]
            c_prev = C[si, di - 5]
            if not np.isnan(c_now) and not np.isnan(c_prev) and c_prev > 0:
                mom5[si, di] = (c_now - c_prev) / c_prev

    # Group momentum excluding self
    grp_mom5 = np.full((NS, ND), np.nan)
    for grp, members in group_members.items():
        for di in range(5, ND):
            for sj in members:
                ms = []
                for sk in members:
                    if sk == sj:
                        continue
                    m = mom5[sk, di]
                    if not np.isnan(m):
                        ms.append(m)
                if ms:
                    grp_mom5[sj, di] = np.mean(ms)

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

    print(f"  Group momentum + ATR done ({time.time()-t2:.1f}s)", flush=True)

    # ========================================
    # SCORING FUNCTIONS
    # ========================================

    def make_group_lag_score(mom_lag=5, min_lag=0.003, scale=10.0):
        """v34b group momentum lag scorer."""
        def score(si, di):
            own = mom5[si, di]
            grp = grp_mom5[si, di]
            if np.isnan(own) or np.isnan(grp):
                return np.nan
            divergence = grp - own
            if abs(divergence) < min_lag:
                return np.nan
            sc = np.clip(divergence * scale, -1, 1)
            if sc <= 0:
                return np.nan
            return sc
        return score

    def make_pure_vol_score(exp_thresh=1.3, cont_thresh=0.8,
                            exp_mult=0.5, cont_mult=1.0, min_score=0.3):
        """Pure vol timing signal: enter when vol contraction predicted."""
        def score(si, di):
            vr = vol_ratio[si, di]
            if np.isnan(vr):
                return np.nan
            if vr < cont_thresh:
                # Contraction: safe to trade
                # Score based on degree of contraction
                sc = np.clip((cont_thresh - vr) / cont_thresh * 5, min_score, 1.0)
                return sc
            return np.nan
        return score

    def make_vol_timed_group_score(exp_thresh=1.3, cont_thresh=0.8,
                                    exp_mult=0.5, cont_mult=1.2,
                                    mom_lag=5, min_lag=0.003, scale=10.0):
        """Combined: group momentum lag with vol-timed position sizing."""
        def score(si, di):
            # First get group lag signal
            own = mom5[si, di]
            grp = grp_mom5[si, di]
            if np.isnan(own) or np.isnan(grp):
                return np.nan
            divergence = grp - own
            if abs(divergence) < min_lag:
                return np.nan
            base_sc = np.clip(divergence * scale, -1, 1)
            if base_sc <= 0:
                return np.nan

            # Vol regime overlay
            vr = vol_ratio[si, di]
            if np.isnan(vr):
                # No vol info: neutral
                return base_sc

            if vr > exp_thresh:
                # EXPANSION: reduce or skip
                base_sc *= exp_mult
            elif vr < cont_thresh:
                # CONTRACTION: boost
                base_sc *= cont_mult
            # else NEUTRAL: keep base_sc

            return base_sc
        return score

    # ========================================
    # BACKTEST ENGINE
    # ========================================
    def run_backtest(score_fn, name, top_n=1, hold_min=2, hold_max=5,
                     trail_atr_mult=2.5, wf_split_year=None,
                     position_mult_fn=None):
        """
        Multi-position backtest with optional walk-forward.
        position_mult_fn: callable(si, di) -> float, scales lots.
                          If None, no vol-based position sizing.
        """
        cash = float(CASH0)
        trades = []
        positions = []
        last_exit = {}

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

                        # Position sizing from vol regime
                        pos_mult = 1.0
                        if position_mult_fn is not None:
                            pm = position_mult_fn(best_si, di)
                            if pm is not None:
                                pos_mult = pm

                        # Skip if position multiplier is 0
                        if pos_mult <= 0:
                            continue

                        effective_cash = cash_per_slot * pos_mult
                        lots = int(effective_cash / (notional * (1 + COMM)))
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

        return {
            'name': name, 'ann': round(ann, 1), 'n': len(trades),
            'wr': round(wr, 1), 'dd': round(max_dd, 1),
            'avg_win': round(avg_win, 2), 'avg_loss': round(avg_loss, 2),
            'avg_days': round(avg_days, 1), 'pf': round(pf, 2),
            'cash': round(cash, 0),
            'reasons': reasons, 'yearly': year_stats,
        }

    # ========================================
    # POSITION SIZING FUNCTIONS
    # ========================================
    def make_vol_position_mult(exp_thresh, cont_thresh, exp_mult, cont_mult):
        """Returns a function (si, di) -> position multiplier based on vol regime."""
        def fn(si, di):
            vr = vol_ratio[si, di]
            if np.isnan(vr):
                return 1.0  # neutral if no data
            if vr > exp_thresh:
                return exp_mult
            elif vr < cont_thresh:
                return cont_mult
            return 1.0
        return fn

    # ========================================
    # CONFIGURATION SWEEP
    # ========================================
    print("\n[Backtest] Running parameter sweep...", flush=True)
    results = []
    configs = []

    # --- A. V34b BASELINE (no vol timing) ---
    for hold_max in [3, 5, 7]:
        for trail in [2.5, 3.0]:
            configs.append((
                make_group_lag_score(), f"BASE_H{hold_max}_TR{trail:.1f}",
                1, 2, hold_max, trail, None, None
            ))

    # --- B. COMBINED: Vol-timed group momentum (score-level modulation) ---
    for exp_t in [1.2, 1.3, 1.5]:
        for cont_t in [0.7, 0.8]:
            for exp_m, cont_m in [(0.0, 1.0), (0.5, 1.0), (0.5, 1.2)]:
                for hold_max in [3, 5, 7]:
                    for trail in [2.5, 3.0]:
                        configs.append((
                            make_vol_timed_group_score(
                                exp_thresh=exp_t, cont_thresh=cont_t,
                                exp_mult=exp_m, cont_mult=cont_m,
                            ),
                            f"VG_E{exp_t}_C{cont_t}_M{exp_m:.1f}/{cont_m:.1f}_H{hold_max}_TR{trail:.1f}",
                            1, 2, hold_max, trail, None, None
                        ))

    # --- C. SEPARATE: Group mom signal + vol position sizing ---
    for exp_t in [1.2, 1.3, 1.5]:
        for cont_t in [0.7, 0.8]:
            for exp_m, cont_m in [(0.0, 1.0), (0.5, 1.0), (0.5, 1.2)]:
                for hold_max in [3, 5, 7]:
                    for trail in [2.5, 3.0]:
                        pmfn = make_vol_position_mult(exp_t, cont_t, exp_m, cont_m)
                        configs.append((
                            make_group_lag_score(),
                            f"PS_E{exp_t}_C{cont_t}_M{exp_m:.1f}/{cont_m:.1f}_H{hold_max}_TR{trail:.1f}",
                            1, 2, hold_max, trail, None, pmfn
                        ))

    # --- D. PURE VOL TIMING (no group momentum) ---
    for exp_t in [1.3]:
        for cont_t in [0.7, 0.8]:
            for hold_max in [3, 5, 7]:
                for trail in [2.5, 3.0]:
                    configs.append((
                        make_pure_vol_score(exp_thresh=exp_t, cont_thresh=cont_t),
                        f"PVOL_C{cont_t}_H{hold_max}_TR{trail:.1f}",
                        1, 2, hold_max, trail, None, None
                    ))

    # --- E. WALK-FORWARD: best combined configs on 2023-2026 ---
    for exp_t in [1.2, 1.3]:
        for cont_t in [0.7, 0.8]:
            for exp_m, cont_m in [(0.5, 1.0), (0.5, 1.2)]:
                for hold_max in [3, 5]:
                    for wf_year in [2023, 2024]:
                        configs.append((
                            make_vol_timed_group_score(
                                exp_thresh=exp_t, cont_thresh=cont_t,
                                exp_mult=exp_m, cont_mult=cont_m,
                            ),
                            f"WF{wf_year}_VG_E{exp_t}_C{cont_t}_M{exp_m:.1f}/{cont_m:.1f}_H{hold_max}",
                            1, 2, hold_max, 2.5, wf_year, None
                        ))

    # Walk-forward baselines
    for hold_max in [3, 5]:
        for wf_year in [2023, 2024]:
            configs.append((
                make_group_lag_score(),
                f"WF{wf_year}_BASE_H{hold_max}",
                1, 2, hold_max, 2.5, wf_year, None
            ))

    print(f"  {len(configs)} configurations", flush=True)

    for ci, (fn, name, tn, hmin, hmax, trail, wf, pmfn) in enumerate(configs):
        r = run_backtest(fn, name, top_n=tn, hold_min=hmin, hold_max=hmax,
                         trail_atr_mult=trail, wf_split_year=wf,
                         position_mult_fn=pmfn)
        if r and r['ann'] > -50:
            results.append(r)
            if r['ann'] > 20:
                parts = []
                for reason, stats in sorted(r['reasons'].items()):
                    wr_r = stats['w'] / stats['n'] * 100 if stats['n'] > 0 else 0
                    parts.append(f"{reason}:{stats['n']}({wr_r:.0f}%)")
                print(f"  {r['name']:55s} | Ann {r['ann']:+7.1f}% | WR {r['wr']:5.1f}% | "
                      f"N {r['n']:4d} | DD {r['dd']:6.1f}% | PF {r['pf']:4.2f} | "
                      f"AvgD {r['avg_days']:.1f}")

        if (ci + 1) % 50 == 0:
            print(f"  [{ci+1}/{len(configs)}] {len(results)} results so far", flush=True)

    # ========================================
    # RESULTS
    # ========================================
    results.sort(key=lambda x: -x['ann'])

    # Separate categories
    base_results = [r for r in results if r['name'].startswith('BASE')]
    combined_results = [r for r in results if r['name'].startswith('VG_')]
    position_results = [r for r in results if r['name'].startswith('PS_')]
    pure_vol_results = [r for r in results if r['name'].startswith('PVOL_')]
    wf_results = [r for r in results if r['name'].startswith('WF')]

    def print_top(label, res_list, count=10):
        print(f"\n  {label}")
        print(f"  {'-' * 115}")
        if not res_list:
            print("  (none)")
            return
        for r in res_list[:count]:
            parts = []
            for reason, stats in sorted(r['reasons'].items()):
                wr_r = stats['w'] / stats['n'] * 100 if stats['n'] > 0 else 0
                parts.append(f"{reason}:{stats['n']}({wr_r:.0f}%)")
            print(f"  {r['name']:55s} | Ann {r['ann']:+7.1f}% | WR {r['wr']:5.1f}% | "
                  f"N {r['n']:4d} | DD {r['dd']:6.1f}% | PF {r['pf']:4.2f} | "
                  f"AvgW {r['avg_win']:+.2f}% | AvgL {r['avg_loss']:.2f}% | "
                  f"AvgD {r['avg_days']:.1f}")
            print(f"  {'':55s} | Exits: {' | '.join(parts)}")

    print(f"\n{'=' * 120}")
    print(f"  TOP 20 OVERALL (sorted by annual return)")
    print(f"{'=' * 120}")
    print_top("ALL RESULTS", results, 20)

    print(f"\n{'=' * 120}")
    print(f"  CATEGORY BREAKDOWNS")
    print(f"{'=' * 120}")
    print_top("V34b BASELINE (no vol timing)", base_results, 5)
    print_top("COMBINED: Vol-timed Group Momentum (score modulation)", combined_results, 10)
    print_top("POSITION SIZING: Group Mom + Vol Position Mult", position_results, 10)
    print_top("PURE VOL TIMING (no group momentum)", pure_vol_results, 5)

    # Walk-forward
    if wf_results:
        wf_results.sort(key=lambda x: -x['ann'])
        print_top("WALK-FORWARD (out-of-sample 2023+)", wf_results, 15)

        # Compare WF baseline vs WF vol-timed
        wf_base = [r for r in wf_results if 'BASE' in r['name']]
        wf_vol = [r for r in wf_results if 'BASE' not in r['name']]
        if wf_base and wf_vol:
            print(f"\n  WALK-FORWARD COMPARISON:")
            print(f"  {'=' * 80}")
            best_base = max(wf_base, key=lambda x: x['ann'])
            best_vol = max(wf_vol, key=lambda x: x['ann'])
            print(f"  Best baseline:  {best_base['name']:45s} Ann={best_base['ann']:+.1f}% WR={best_base['wr']:.1f}% N={best_base['n']}")
            print(f"  Best vol-timed: {best_vol['name']:45s} Ann={best_vol['ann']:+.1f}% WR={best_vol['wr']:.1f}% N={best_vol['n']}")
            diff = best_vol['ann'] - best_base['ann']
            print(f"  Vol timing improvement: {diff:+.1f}% annual")

    # Yearly breakdown for top 5 overall
    print(f"\n{'=' * 120}")
    print(f"  YEARLY BREAKDOWN FOR TOP 5")
    print(f"{'=' * 120}")
    for r in results[:5]:
        print(f"\n  #{results.index(r)+1}: {r['name']}")
        print(f"  Ann={r['ann']:+.1f}%  WR={r['wr']:.1f}%  N={r['n']}  DD={r['dd']:.1f}%  PF={r['pf']:.2f}")
        for y in sorted(r['yearly'].keys()):
            ys = r['yearly'][y]
            wr_y = ys['w'] / ys['n'] * 100 if ys['n'] > 0 else 0
            print(f"    {y}: {ys['n']:3d}t  WR={wr_y:5.1f}%  PnL={ys['pnl']:+.1f}%")

    # HAR-RV vol regime statistics
    print(f"\n{'=' * 120}")
    print(f"  HAR-RV VOL REGIME STATISTICS")
    print(f"{'=' * 120}")
    regime_counts = {'expansion': 0, 'contraction': 0, 'neutral': 0}
    for si in range(NS):
        for di in range(MIN_TRAIN, ND):
            vr = vol_ratio[si, di]
            if np.isnan(vr):
                continue
            if vr > 1.3:
                regime_counts['expansion'] += 1
            elif vr < 0.8:
                regime_counts['contraction'] += 1
            else:
                regime_counts['neutral'] += 1
    total = sum(regime_counts.values())
    if total > 0:
        for k, v in regime_counts.items():
            print(f"    {k:15s}: {v:8d} ({v/total*100:.1f}%)")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.1f}s")
    print("=" * 120)


if __name__ == '__main__':
    main()
