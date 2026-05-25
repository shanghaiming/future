"""
Alpha Futures V37 — Kelly Criterion Adaptive Position Sizing with Cross-Commodity VaR Risk Management
=======================================================================================================
Combines Kelly criterion position sizing, KER regime adjustment, and CAViaR-X cross-asset
tail risk monitoring for Chinese commodity futures.

Key components:
  1. Rolling Kelly computation: f* = (p*b - q) / b per commodity, half-Kelly standard
  2. KER regime adjustment: trending (KER>0.3) x1.2, ranging (KER<0.15) x0.5
  3. Cross-commodity VaR: when core commodities breach VaR, reduce downstream exposure
  4. Sortino ratio for ranking: mean_return / downside_deviation
  5. Group momentum lag entry signal (proven at +86.8% from v34b)
  6. DD circuit breaker: 15%/25% thresholds

Entry: Group momentum lag (commodity lags its supply-chain group)
Position sizing: Kelly-adjusted lots with KER regime and VaR overlays
Exit: ATR trailing stop or time exit (3-7 days), no fixed stop loss

Test configs sweep Kelly fraction, KER regime, VaR monitoring, DD breaker, hold, trail.
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

# Core commodities for VaR monitoring (upstream bellwethers)
CORE_COMMODITIES = {
    'scfi': 'crude_oil',
    'ifi':  'iron_ore',
    'cufi': 'copper',
}

# Downstream mapping: core commodity -> affected downstream symbols
DOWNSTREAM = {
    'scfi': ['mafi', 'bfi', 'fufi', 'ppfi', 'vfi', 'egfi'],
    'ifi':  ['rbfi', 'hcfi'],
    'cufi': [],
}

# VaR breach lookback (10-day rolling window)
VAR_LOOKBACK = 10

# Kelly rolling window
KELLY_WINDOW = 100


def main():
    t_start = time.time()
    print("=" * 120)
    print("Alpha Futures V37 — Kelly Criterion Adaptive Position Sizing + Cross-Commodity VaR Risk Mgmt")
    print("Kelly f* = (p*b - q)/b | Half-Kelly | KER regime adj | CAViaR-X tail risk | Sortino ranking")
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

    # Map core commodity symbols to stock indices
    core_si = {}
    for sym, label in CORE_COMMODITIES.items():
        if sym in sym_to_si:
            core_si[sym] = sym_to_si[sym]

    # Map downstream symbols to stock indices
    downstream_si = {}
    for core_sym, ds_list in DOWNSTREAM.items():
        downstream_si[core_sym] = [sym_to_si[s] for s in ds_list if s in sym_to_si]

    print(f"  {NS} stocks, {ND} days, Groups: {len(group_members)}, "
          f"Core VaR monitors: {len(core_si)}, "
          f"Downstream links: {sum(len(v) for v in downstream_si.values())}")

    # ========================================
    # PRECOMPUTE ALL SIGNALS
    # ========================================
    print("\n[Signals] Computing...", flush=True)
    t0 = time.time()

    # --- Daily returns (for Kelly and VaR) ---
    daily_ret = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(1, ND):
            c_now = C[si, di]
            c_prev = C[si, di - 1]
            if not np.isnan(c_now) and not np.isnan(c_prev) and c_prev > 0:
                daily_ret[si, di] = (c_now - c_prev) / c_prev

    # --- Momentum at multiple lookbacks ---
    mom = {}
    for lag in [3, 5, 7]:
        m = np.full((NS, ND), np.nan)
        for si in range(NS):
            for di in range(lag, ND):
                c_now = C[si, di]
                c_prev = C[si, di - lag]
                if not np.isnan(c_now) and not np.isnan(c_prev) and c_prev > 0:
                    m[si, di] = (c_now - c_prev) / c_prev
        mom[lag] = m

    # --- Group momentum excluding self (for entry signal) ---
    grp_mom = {}
    for lag in [3, 5, 7]:
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

    # --- KER (Kaufman Efficiency Ratio) over 20 days ---
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

    # --- Rolling Kelly computation (per commodity, 100-day window) ---
    # Uses forward hold-period returns conditioned on signal firing (positive score).
    # For each day in the past where signal fired, compute the forward return over
    # the hold period. This gives signal-specific win rate and win/loss ratio.
    # p = fraction of positive signal-triggered trades
    # b = avg positive forward return / avg |negative forward return|
    # f_kelly = (p*b - q) / b,  q = 1-p
    kelly_raw = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(KELLY_WINDOW + 5, ND):
            # Collect signal-conditioned forward returns over the Kelly window
            fwd_rets = []
            for dd in range(di - KELLY_WINDOW, di - 5):
                # Check if signal was positive on this day
                own_m = mom[5][si, dd]
                grp_m = grp_mom[5][si, dd]
                if np.isnan(own_m) or np.isnan(grp_m):
                    continue
                divergence = grp_m - own_m
                if divergence < 0.003:   # signal did not fire
                    continue
                # Compute forward 5-day return from this signal
                c_entry = C[si, dd]
                c_exit = C[si, min(dd + 5, ND - 1)]
                if np.isnan(c_entry) or np.isnan(c_exit) or c_entry <= 0:
                    continue
                fwd_ret = (c_exit - c_entry) / c_entry
                fwd_rets.append(fwd_ret)

            if len(fwd_rets) < 20:
                continue

            fwd_arr = np.array(fwd_rets)
            pos = fwd_arr[fwd_arr > 0]
            neg = fwd_arr[fwd_arr < 0]
            if len(pos) == 0 or len(neg) == 0:
                # Edge case: all wins or all losses
                if len(neg) == 0:
                    kelly_raw[si, di] = 1.0  # all wins -> max Kelly
                continue

            p = len(pos) / len(fwd_arr)
            q = 1.0 - p
            avg_win = np.mean(pos)
            avg_loss = abs(np.mean(neg))
            if avg_loss <= 0:
                continue
            b = avg_win / avg_loss
            if b <= 0:
                continue
            f_k = (p * b - q) / b
            # Clamp to [0, 1]
            kelly_raw[si, di] = max(0.0, min(1.0, f_k))

    # --- ATR (10-day) for trailing stop ---
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

    # --- Cross-commodity VaR (10-day rolling, historical simulation) ---
    # For each core commodity, compute VaR at given confidence level
    # VaR = percentile(daily_returns_over_lookback, 100 - confidence)
    # Breach: current daily return < VaR threshold
    # We compute VaR for multiple thresholds to allow parameter sweep
    var_90 = {}   # core_sym -> array[ND] of VaR values
    var_95 = {}
    var_breach_90 = {}   # core_sym -> array[ND] bool
    var_breach_95 = {}

    for core_sym, csi in core_si.items():
        v90 = np.full(ND, np.nan)
        v95 = np.full(ND, np.nan)
        breach90 = np.full(ND, False)
        breach95 = np.full(ND, False)

        for di in range(VAR_LOOKBACK, ND):
            rets = daily_ret[csi, di - VAR_LOOKBACK:di]
            valid = rets[~np.isnan(rets)]
            if len(valid) < 5:
                continue
            v90[di] = np.percentile(valid, 10)   # 90% confidence
            v95[di] = np.percentile(valid, 5)     # 95% confidence
            cur_ret = daily_ret[csi, di]
            if not np.isnan(cur_ret):
                if cur_ret < v90[di]:
                    breach90[di] = True
                if cur_ret < v95[di]:
                    breach95[di] = True

        var_90[core_sym] = v90
        var_95[core_sym] = v95
        var_breach_90[core_sym] = breach90
        var_breach_95[core_sym] = breach95

    print(f"  Done ({time.time()-t0:.1f}s)", flush=True)

    # ========================================
    # ENTRY SIGNAL: GROUP MOMENTUM LAG
    # ========================================
    def make_group_lag_score(mom_lag=5, min_lag=0.003, scale=10.0):
        """Score = group_mom_excl - own_mom. Positive = own lags group."""
        def score(si, di):
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
        return score

    # ========================================
    # BACKTEST ENGINE WITH KELLY + VaR
    # ========================================
    def run_backtest(score_fn, name,
                     kelly_frac=0.5, use_ker=True, use_var=True,
                     var_conf=95, dd_threshold_1=15.0, dd_threshold_2=25.0,
                     hold_max=5, trail_atr_mult=2.5,
                     wf_split_year=None):
        """
        Single-position, long-only backtest with Kelly position sizing and VaR risk management.

        Parameters:
          kelly_frac: fraction of Kelly to use (0.5 = half-Kelly)
          use_ker: whether to apply KER regime adjustment
          use_var: whether to apply cross-commodity VaR monitoring
          var_conf: VaR confidence level (90 or 95)
          dd_threshold_1: DD level to halve positions (%)
          dd_threshold_2: DD level to go to cash (%)
          hold_max: max hold days
          trail_atr_mult: ATR trailing stop multiplier
          wf_split_year: walk-forward test start year (None = full period)
        """
        cash = float(CASH0)
        trades = []
        positions = []
        last_exit = {}
        dd_cooldown = 0   # days remaining with no new entries after severe DD
        equity_peak = float(CASH0)

        for di in range(MIN_TRAIN, ND):
            year = dates[di].year
            if wf_split_year is not None and year < wf_split_year:
                continue

            # --- Mark-to-market: track current equity for DD circuit breaker ---
            mtm_equity = cash
            for pos in positions:
                c_pos = C[pos['si'], di]
                if np.isnan(c_pos) or c_pos <= 0:
                    c_pos = pos['entry']
                mult = MULT.get(pos['sym'], DEF_MULT)
                mtm_equity += c_pos * mult * pos['lots']
            if mtm_equity > equity_peak:
                equity_peak = mtm_equity
            current_dd = 0.0
            if equity_peak > 0:
                current_dd = (equity_peak - mtm_equity) / equity_peak * 100

            # --- DD circuit breaker: severe -> go to cash, moderate -> reduce ---
            if dd_threshold_2 > 0 and current_dd > dd_threshold_2:
                dd_cooldown = 10
            if dd_cooldown > 0:
                dd_cooldown -= 1

            # --- Manage existing positions ---
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

                # DD circuit breaker: moderate DD -> force exit at half position
                if dd_threshold_1 > 0 and current_dd > dd_threshold_1 and days_held >= 1:
                    # Reduce by exiting if position is losing
                    if pnl_pct < 0:
                        exit_reason = 'dd_reduce'

                # Trailing stop
                if exit_reason is None and trail_atr_mult > 0 and days_held >= 2:
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

            # --- Open new position (single position, long only) ---
            if len(positions) > 0:
                continue  # already holding
            if dd_cooldown > 0:
                continue  # DD circuit breaker active

            # Score all symbols
            scored = []
            for si in range(NS):
                sc = score_fn(si, di)
                if np.isnan(sc) or sc <= 0.01:
                    continue
                sym = syms[si]
                scored.append((si, sc, sym))

            if not scored:
                continue

            scored.sort(key=lambda x: -x[1])
            best_si, best_sc, best_sym = scored[0]

            c = C[best_si, di]
            if np.isnan(c) or c <= 0:
                continue
            mult = MULT.get(best_sym, DEF_MULT)
            notional = c * mult
            if notional <= 0:
                continue

            # === KELLY POSITION SIZING ===
            k = kelly_raw[best_si, di]
            if np.isnan(k):
                f_adj = 0.3  # default moderate allocation if no Kelly history
            else:
                # Apply Kelly fraction (half-Kelly etc.)
                f_actual = k * kelly_frac
                # Clamp
                f_actual = max(0.0, min(1.0, f_actual))

                # === KER REGIME ADJUSTMENT ===
                if use_ker:
                    k_val = ker[best_si, di]
                    if not np.isnan(k_val):
                        if k_val > 0.3:
                            f_adj = f_actual * 1.2   # trending: boost
                        elif k_val < 0.15:
                            f_adj = f_actual * 0.5   # ranging: reduce
                        else:
                            f_adj = f_actual
                    else:
                        f_adj = f_actual
                else:
                    f_adj = f_actual

            # Clamp f_adj
            f_adj = max(0.0, min(1.0, f_adj))

            # If Kelly signal too weak, skip
            if f_adj < 0.1:
                continue

            # === VaR CROSS-COMMODITY CHECK ===
            var_mult = 1.0
            if use_var:
                for core_sym, ds_si_list in downstream_si.items():
                    # Check if best_si is a downstream commodity of this core
                    if best_si in ds_si_list:
                        # Check VaR breach on core
                        if var_conf == 90:
                            breached = var_breach_90[core_sym][di]
                        else:
                            breached = var_breach_95[core_sym][di]
                        if breached:
                            var_mult *= 0.5  # reduce position by 50%

            # Final position size
            effective_frac = f_adj * var_mult

            # DD circuit breaker: moderate DD -> halve new position size
            if dd_threshold_1 > 0 and current_dd > dd_threshold_1 * 0.7:
                effective_frac *= 0.5

            # Compute lots
            alloc = cash * effective_frac
            lots = int(alloc / (notional * (1 + COMM)))
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
                'kelly_f': f_adj, 'var_mult': var_mult,
            })

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
            })

        if len(trades) < 10:
            return None

        # === STATS ===
        equity = float(CASH0)
        peak = float(CASH0)
        max_dd = 0.0
        equity_curve = []
        for t in sorted(trades, key=lambda x: x['di']):
            equity += t['pnl_abs']
            equity_curve.append(equity)
            if equity > peak:
                peak = equity
            if peak > 0:
                dd = (peak - equity) / peak * 100
                if dd > max_dd:
                    max_dd = dd

        # Compute daily-equity-equivalent returns for Sortino
        trade_pnls = [t['pnl_abs'] for t in sorted(trades, key=lambda x: x['di'])]
        if len(trade_pnls) > 1:
            rets = np.diff(trade_pnls) / float(CASH0)
            mean_ret = np.mean(rets) if len(rets) > 0 else 0
            neg_rets = rets[rets < 0]
            downside_dev = np.sqrt(np.mean(neg_rets ** 2)) if len(neg_rets) > 0 else 1e-10
            sortino = mean_ret / downside_dev if downside_dev > 0 else 0
        else:
            sortino = 0

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
            'sortino': round(sortino, 3),
            'cash': round(cash, 0),
            'reasons': reasons, 'yearly': year_stats, 'grp_counts': grp_counts,
        }

    # ========================================
    # PARAMETER SWEEP
    # ========================================
    print("\n[Backtest] Running parameter sweep...", flush=True)
    results = []
    configs = []

    # --- Sweep axes ---
    kelly_fracs = [0.25, 0.5, 0.75, 1.0]
    ker_options = [True, False]
    var_options = [True, False]
    var_confs = [90, 95]
    dd_levels = [15.0, 20.0, 25.0, 0.0]   # 0 = off
    hold_periods = [3, 5]
    trail_mults = [2.5, 3.0]

    # --- Build configs ---
    # Full factorial on a reduced set first
    for kf in kelly_fracs:
        for use_ker in ker_options:
            for use_var in var_options:
                for vc in var_confs:
                    for dd1 in dd_levels:
                        for hold in hold_periods:
                            for trail in trail_mults:
                                name = (f"KF{kf:.2f}_KER{'Y' if use_ker else 'N'}"
                                        f"_VAR{'Y' if use_var else 'N'}{vc}"
                                        f"_DD{dd1:.0f}_H{hold}_T{trail:.1f}")
                                configs.append((kf, use_ker, use_var, vc, dd1, 25.0,
                                                hold, trail, None, name))

    # --- Baseline: no Kelly (full allocation), no VaR, no KER ---
    for hold in hold_periods:
        for trail in trail_mults:
            name = f"BASELINE_H{hold}_T{trail}"
            configs.append((1.0, False, False, 95, 0.0, 0.0,
                            hold, trail, None, name))

    # --- Walk-forward on best config patterns (2023-2026) ---
    for kf in [0.5, 0.75]:
        for hold in [3, 5]:
            for wf_year in [2023, 2024]:
                name = (f"KF{kf:.2f}_KERy_VARy95_DD15"
                        f"_H{hold}_T2.5_WF{wf_year}")
                configs.append((kf, True, True, 95, 15.0, 25.0,
                                hold, 2.5, wf_year, name))

    print(f"  {len(configs)} configurations", flush=True)

    for ci, (kf, uk, uv, vc, dd1, dd2, hold, trail, wf, name) in enumerate(configs):
        score_fn = make_group_lag_score(mom_lag=5, min_lag=0.003, scale=10.0)
        r = run_backtest(score_fn, name,
                         kelly_frac=kf, use_ker=uk, use_var=uv,
                         var_conf=vc, dd_threshold_1=dd1, dd_threshold_2=dd2,
                         hold_max=hold, trail_atr_mult=trail,
                         wf_split_year=wf)
        if r and r['ann'] > 0:
            results.append(r)
            if r['ann'] > 30:
                parts = []
                for reason, stats in sorted(r['reasons'].items()):
                    wr_r = stats['w'] / stats['n'] * 100 if stats['n'] > 0 else 0
                    parts.append(f"{reason}:{stats['n']}({wr_r:.0f}%)")
                print(f"  {r['name']:60s} | Ann {r['ann']:+7.1f}% | WR {r['wr']:5.1f}% | "
                      f"N {r['n']:4d} | DD {r['dd']:6.1f}% | PF {r['pf']:4.2f} | "
                      f"Sort {r['sortino']:6.3f} | AvgD {r['avg_days']:.1f}")
                print(f"  {'':60s} | Exits: {' | '.join(parts)}")

        if (ci + 1) % 100 == 0:
            print(f"  [{ci+1}/{len(configs)}] {len(results)} profitable", flush=True)

    # ========================================
    # RESULTS
    # ========================================
    results.sort(key=lambda x: -x['ann'])

    # Separate walk-forward
    wf_results = [r for r in results if '_WF' in r['name']]
    full_results = [r for r in results if '_WF' not in r['name']]

    print(f"\n{'=' * 130}")
    print(f"  TOP 20 FULL-PERIOD RESULTS (sorted by annual return)")
    print(f"{'=' * 130}")
    hdr = (f"  {'Strategy':60s} | {'Ann':>7s} | {'WR':>5s} | {'N':>4s} | {'DD':>6s} | "
           f"{'PF':>4s} | {'Sortino':>7s} | {'AvgW':>7s} | {'AvgL':>6s} | {'AvgD':>4s}")
    print(hdr)
    print(f"  {'-' * 125}")
    for r in full_results[:20]:
        print(f"  {r['name']:60s} | {r['ann']:+7.1f}% | {r['wr']:5.1f}% | "
              f"{r['n']:4d} | {r['dd']:6.1f}% | {r['pf']:4.2f} | "
              f"{r['sortino']:7.3f} | {r['avg_win']:+6.2f}% | {r['avg_loss']:5.2f}% | "
              f"{r['avg_days']:4.1f}")

    # Top by Sortino
    sortino_sorted = sorted(full_results, key=lambda x: -x['sortino'])
    print(f"\n  TOP 10 BY SORTINO RATIO:")
    print(f"  {'-' * 125}")
    for r in sortino_sorted[:10]:
        print(f"  {r['name']:60s} | Sort {r['sortino']:7.3f} | Ann {r['ann']:+7.1f}% | "
              f"WR {r['wr']:5.1f}% | DD {r['dd']:6.1f}% | PF {r['pf']:4.2f}")

    # Walk-forward results
    if wf_results:
        wf_results.sort(key=lambda x: -x['ann'])
        print(f"\n  WALK-FORWARD RESULTS (out-of-sample 2023-2026)")
        print(f"  {'-' * 125}")
        for r in wf_results[:15]:
            print(f"  {r['name']:60s} | Ann {r['ann']:+7.1f}% | WR {r['wr']:5.1f}% | "
                  f"N {r['n']:4d} | DD {r['dd']:6.1f}% | PF {r['pf']:4.2f} | Sort {r['sortino']:7.3f}")

    # Best result detail
    if full_results:
        best = full_results[0]
        print(f"\n  BEST: {best['name']}")
        print(f"  Ann={best['ann']:+.1f}%  WR={best['wr']:.1f}%  N={best['n']}  "
              f"DD={best['dd']:.1f}%  PF={best['pf']:.2f}  Sortino={best['sortino']:.3f}  "
              f"Final={best['cash']:.0f}")

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
            print(f"\n  #{idx+1}: {r['name']} (Ann={r['ann']:+.1f}%, WR={r['wr']:.1f}%, "
                  f"DD={r['dd']:.1f}%, Sortino={r['sortino']:.3f})")
            for y in sorted(r['yearly'].keys()):
                ys = r['yearly'][y]
                wr_y = ys['w'] / ys['n'] * 100 if ys['n'] > 0 else 0
                print(f"    {y}: {ys['n']:3d}t  WR={wr_y:5.1f}%  PnL={ys['pnl']:+.1f}%")

    # Comparison: Kelly vs baseline
    print(f"\n  KELLY vs BASELINE COMPARISON:")
    print(f"  {'-' * 90}")
    baselines = [r for r in full_results if r['name'].startswith('BASELINE')]
    kelly_only = [r for r in full_results if not r['name'].startswith('BASELINE')]
    if baselines and kelly_only:
        best_base = max(baselines, key=lambda x: x['ann'])
        best_kelly = max(kelly_only, key=lambda x: x['ann'])
        print(f"  Best Baseline: {best_base['name']}")
        print(f"    Ann={best_base['ann']:+.1f}%  WR={best_base['wr']:.1f}%  DD={best_base['dd']:.1f}%  "
              f"PF={best_base['pf']:.2f}")
        print(f"  Best Kelly:    {best_kelly['name']}")
        print(f"    Ann={best_kelly['ann']:+.1f}%  WR={best_kelly['wr']:.1f}%  DD={best_kelly['dd']:.1f}%  "
              f"PF={best_kelly['pf']:.2f}  Sortino={best_kelly['sortino']:.3f}")
        if best_base['ann'] != 0:
            improvement = (best_kelly['ann'] - best_base['ann']) / abs(best_base['ann']) * 100
            print(f"  Kelly improvement over baseline: {improvement:+.1f}%")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.1f}s")
    print("=" * 130)


if __name__ == '__main__':
    main()
