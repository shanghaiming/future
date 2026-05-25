"""
Alpha Futures V127 -- EXIT STRATEGY OPTIMIZATION
=================================================
Champion signal: ROC(5)>1% AND Z>1.5 AND ROC improving, rank by ROC*Z, top_n=1
Test 10 exit strategies (A-J) vs fixed hold=1.

ALL signals use NEXT-OPEN execution: signal at close di, entry at O[si, di+1].
"""
import sys, os, time, warnings
import numpy as np
import talib
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alpha_v2 import load_all_data, MIN_TRAIN, CASH0

# ============================================================
# CONSTANTS
# ============================================================
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


def annual_return(final, initial, n_days):
    if final <= 0 or initial <= 0 or n_days <= 0:
        return -100.0
    yrs = n_days / 252
    return (final / initial) ** (1.0 / yrs) * 100 - 100


def compute_atr(high, low, close, period=14):
    """ATR for a single instrument's arrays."""
    n = len(close)
    atr = np.full(n, np.nan)
    tr = np.full(n, np.nan)
    for i in range(1, n):
        if np.isnan(high[i]) or np.isnan(low[i]) or np.isnan(close[i]):
            continue
        h_l = high[i] - low[i]
        h_pc = abs(high[i] - close[i-1]) if not np.isnan(close[i-1]) else 0
        l_pc = abs(low[i] - close[i-1]) if not np.isnan(close[i-1]) else 0
        tr[i] = max(h_l, h_pc, l_pc)
    # EMA-style ATR
    valid_tr = tr[~np.isnan(tr)]
    if len(valid_tr) < period:
        return atr
    first_valid = next((i for i in range(n) if not np.isnan(tr[i])), None)
    if first_valid is None:
        return atr
    # SMA for first period
    if first_valid + period <= n:
        atr[first_valid + period - 1] = np.nanmean(tr[first_valid:first_valid + period])
        for i in range(first_valid + period, n):
            if not np.isnan(tr[i]):
                atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    return atr


def main():
    print("=" * 160)
    print("  Alpha Futures V127 -- EXIT STRATEGY OPTIMIZATION")
    print("=" * 160)
    print(f"  Champion signal: ROC(5)>1% AND Z>1.5 AND ROC improving, rank by ROC*Z, top_n=1")
    print(f"  Test 10 exit strategies (A-J) vs fixed hold=1")
    print(f"  Walk-forward by year (2020-2025)")

    # -- Load data -------------------------------------------------
    print("\n[Data] Loading...", flush=True)
    t_start = time.time()
    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)
    years = sorted(set(d.year for d in dates))
    print(f"  {NS} commodities, {ND} days, years: {years}")
    print(f"  MIN_TRAIN={MIN_TRAIN}, CASH0={CASH0:,}")

    # ================================================================
    # PRECOMPUTE INDICATORS
    # ================================================================
    print("\n[Precompute] Daily returns, ROC(5), Z-scores, ATR...", flush=True)
    t0 = time.time()

    # Daily returns
    RET = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si].astype(np.float64)
        for di in range(1, ND):
            if not np.isnan(c[di]) and not np.isnan(c[di-1]) and c[di-1] > 0:
                RET[si, di] = (c[di] / c[di-1] - 1) * 100

    # ROC(5) in percent
    ROC5 = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si].astype(np.float64)
        ROC5[si] = talib.ROC(c, timeperiod=5)

    # Z-score of daily returns (20-day rolling) -- champion style
    ZSCORE = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(21, ND):
            rets = RET[si, di-20:di]
            valid = rets[~np.isnan(rets)]
            if len(valid) < 10:
                continue
            mean_r = np.mean(valid)
            std_r = np.std(valid, ddof=1)
            if std_r > 0 and not np.isnan(RET[si, di]):
                ZSCORE[si, di] = (RET[si, di] - mean_r) / std_r

    # ROC(1) for momentum decay exit
    ROC1 = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si].astype(np.float64)
        ROC1[si] = talib.ROC(c, timeperiod=1)

    # ATR(14) for trailing stop
    ATR14 = np.full((NS, ND), np.nan)
    for si in range(NS):
        h = H[si].astype(np.float64)
        l = L[si].astype(np.float64)
        c = C[si].astype(np.float64)
        ATR14[si] = compute_atr(h, l, c, period=14)

    print(f"  All indicators computed ({time.time()-t0:.1f}s)")

    # ================================================================
    # SIGNAL GENERATION (shared by all exit strategies)
    # ================================================================
    def get_entry_signals(di, positions, top_n=1):
        """
        Champion signal: ROC(5)>1% AND Z>1.5 AND ROC improving.
        Rank by ROC*Z, top_n candidates.
        Returns list of (score, si, entry_price).
        """
        entry_di = di + 1
        if entry_di >= ND:
            return []

        candidates = []
        for s in range(NS):
            roc = ROC5[s, di]
            z = ZSCORE[s, di]

            if np.isnan(roc) or roc <= 1.0:
                continue
            if np.isnan(z) or z <= 1.5:
                continue

            # ROC improving filter
            roc_prev = ROC5[s, di-1]
            if np.isnan(roc_prev) or roc <= roc_prev:
                continue

            ep = O[s, entry_di]
            if np.isnan(ep) or ep <= 0:
                continue
            if any(p['si'] == s for p in positions):
                continue

            score = roc * z
            candidates.append((score, s, ep, roc, z))

        candidates.sort(key=lambda x: -x[0])
        return candidates[:top_n]

    # ================================================================
    # GENERIC BACKTEST ENGINE WITH EXIT STRATEGY
    # ================================================================
    def backtest(exit_cfg, start_di=MIN_TRAIN, end_di=None):
        """
        exit_cfg dict:
          exit_type: 'fixed', 'trailing_stop', 'profit_target', 'roc_exit',
                     'zscore_decay', 'partial_profit', 'intraday_reversal',
                     'momentum_decay', 'adaptive_hold', 'gap_exit'
          ... plus type-specific params
        """
        if end_di is None:
            end_di = ND

        exit_type = exit_cfg.get('exit_type', 'fixed')
        label = exit_cfg.get('label', '')

        cash = float(CASH0)
        positions = []
        trades = []
        daily_equity = []

        for di in range(start_di, end_di - 1):
            # Track daily equity
            port_val = cash
            for pos in positions:
                cp = C[pos['si'], di]
                if not np.isnan(cp) and cp > 0:
                    mult = MULT.get(pos['sym'], DEF_MULT)
                    port_val += cp * mult * pos['lots'] - cp * mult * abs(pos['lots']) * COMM
            daily_equity.append(port_val)

            # -- EXIT LOGIC (varies by strategy) --
            closed = []
            for pos in positions:
                si = pos['si']
                entry_di_pos = pos['entry_di']
                entry_price = pos['entry_price']
                mult = MULT.get(pos['sym'], DEF_MULT)
                days_held = di - entry_di_pos

                should_exit = False
                exit_reason = 'max_hold'

                if exit_type == 'fixed':
                    hold = exit_cfg.get('hold', 1)
                    if days_held >= hold:
                        should_exit = True
                        exit_reason = f'fixed_{hold}d'

                elif exit_type == 'trailing_stop':
                    n_atr = exit_cfg.get('n_atr', 2.0)
                    max_hold = exit_cfg.get('max_hold', 10)
                    # Track highest close since entry
                    highest = entry_price
                    for d in range(entry_di_pos, di + 1):
                        cc = C[si, d]
                        if not np.isnan(cc) and cc > highest:
                            highest = cc
                    atr_val = ATR14[si, di]
                    close_val = C[si, di]
                    if not np.isnan(atr_val) and atr_val > 0 and not np.isnan(close_val):
                        stop = highest - n_atr * atr_val
                        if close_val <= stop:
                            should_exit = True
                            exit_reason = f'trail_{n_atr}atr'
                    if days_held >= max_hold:
                        should_exit = True
                        exit_reason = f'trail_max{max_hold}d'

                elif exit_type == 'profit_target':
                    target_pct = exit_cfg.get('target_pct', 3.0)
                    max_hold = exit_cfg.get('max_hold', 5)
                    close_val = C[si, di]
                    if not np.isnan(close_val) and entry_price > 0:
                        unrealized = (close_val / entry_price - 1) * 100
                        if unrealized >= target_pct:
                            should_exit = True
                            exit_reason = f'target_{target_pct}%'
                    if days_held >= max_hold:
                        should_exit = True
                        exit_reason = f'pt_max{max_hold}d'

                elif exit_type == 'roc_exit':
                    max_hold = exit_cfg.get('max_hold', 5)
                    roc5_now = ROC5[si, di]
                    roc5_prev = ROC5[si, di - 1]
                    if days_held > 0 and not np.isnan(roc5_now) and not np.isnan(roc5_prev):
                        if roc5_now < roc5_prev:
                            should_exit = True
                            exit_reason = 'roc_declining'
                    if days_held >= max_hold:
                        should_exit = True
                        exit_reason = f'roc_max{max_hold}d'

                elif exit_type == 'zscore_decay':
                    max_hold = exit_cfg.get('max_hold', 5)
                    z_now = ZSCORE[si, di]
                    if days_held > 0 and not np.isnan(z_now):
                        if z_now < 0:
                            should_exit = True
                            exit_reason = 'z_below_0'
                    if days_held >= max_hold:
                        should_exit = True
                        exit_reason = f'z_max{max_hold}d'

                elif exit_type == 'partial_profit':
                    target_pct = exit_cfg.get('first_target_pct', 3.0)
                    trail_n = exit_cfg.get('trail_n_atr', 2.0)
                    max_hold = exit_cfg.get('max_hold', 10)
                    close_val = C[si, di]
                    if not np.isnan(close_val) and entry_price > 0:
                        unrealized = (close_val / entry_price - 1) * 100
                        # Phase 1: Take 50% at first target
                        if not pos.get('half_taken') and unrealized >= target_pct:
                            # Close half
                            half_lots = pos['lots'] // 2
                            if half_lots > 0:
                                mkt_val = close_val * mult * half_lots
                                cash += mkt_val - mkt_val * COMM
                                pnl = (close_val - entry_price) * mult * half_lots
                                invested = entry_price * mult * half_lots
                                pnl_pct = pnl / invested * 100 if invested > 0 else 0
                                trades.append({
                                    'pnl': pnl, 'pnl_pct': pnl_pct,
                                    'entry_di': entry_di_pos, 'exit_di': di,
                                    'sym': pos['sym'], 'reason': f'partial_{target_pct}%',
                                })
                                pos['lots'] -= half_lots
                                pos['half_taken'] = True
                                pos['trail_start'] = close_val  # trailing stop from here
                        # Phase 2: Trailing stop on remaining
                        if pos.get('half_taken') and pos.get('trail_start'):
                            highest = pos['trail_start']
                            for d in range(max(entry_di_pos, di - 5), di + 1):
                                cc = C[si, d]
                                if not np.isnan(cc) and cc > highest:
                                    highest = cc
                            atr_val = ATR14[si, di]
                            if not np.isnan(atr_val) and atr_val > 0:
                                stop = highest - trail_n * atr_val
                                if close_val <= stop:
                                    should_exit = True
                                    exit_reason = f'partial_trail_{trail_n}atr'
                    if days_held >= max_hold:
                        should_exit = True
                        exit_reason = f'partial_max{max_hold}d'

                elif exit_type == 'intraday_reversal':
                    max_hold = exit_cfg.get('max_hold', 5)
                    close_val = C[si, di]
                    open_val = O[si, di]
                    low_val = L[si, di]
                    if (not np.isnan(close_val) and not np.isnan(open_val)
                            and not np.isnan(low_val)):
                        # Intraday reversal: C < O AND L < entry_price
                        if close_val < open_val and low_val < entry_price:
                            should_exit = True
                            exit_reason = 'intraday_reversal'
                    if days_held >= max_hold:
                        should_exit = True
                        exit_reason = f'ir_max{max_hold}d'

                elif exit_type == 'momentum_decay':
                    max_hold = exit_cfg.get('max_hold', 3)
                    decay_thresh = exit_cfg.get('decay_thresh', -1.0)
                    roc1_now = ROC1[si, di]
                    if not np.isnan(roc1_now) and roc1_now < decay_thresh:
                        should_exit = True
                        exit_reason = f'roc1<{decay_thresh}%'
                    if days_held >= max_hold:
                        should_exit = True
                        exit_reason = f'md_max{max_hold}d'

                elif exit_type == 'adaptive_hold':
                    signal_score = pos.get('signal_score', 5.0)
                    if signal_score > 10:
                        hold = 1
                    elif signal_score > 5:
                        hold = 2
                    else:
                        hold = 3
                    if days_held >= hold:
                        should_exit = True
                        exit_reason = f'adaptive_{signal_score:.1f}'

                elif exit_type == 'gap_exit':
                    # Entry was at O[entry_di]. Check next day's open.
                    # This is a 1-2 day strategy.
                    # Day after entry: if O gaps down vs entry, exit; else hold through day.
                    if days_held == 0:
                        # Still on entry day, no exit check
                        pass
                    elif days_held >= 1:
                        # Check the open of the day after entry
                        next_open = O[si, entry_di_pos + 1] if entry_di_pos + 1 < ND else np.nan
                        entry_open = O[si, entry_di_pos]
                        if (not np.isnan(next_open) and not np.isnan(entry_open)
                                and next_open < entry_open):
                            # Gap down - should have exited at next_open
                            # But we're processing at di, so exit now
                            should_exit = True
                            exit_reason = 'gap_down'
                        elif days_held >= 2:
                            # Gap up or flat: held through, exit now
                            should_exit = True
                            exit_reason = 'gap_hold_done'

                if should_exit:
                    exit_price = C[si, di]
                    if np.isnan(exit_price) or exit_price <= 0:
                        exit_price = entry_price
                    mkt_val = exit_price * mult * abs(pos['lots'])
                    cash += mkt_val - mkt_val * COMM
                    pnl = (exit_price - entry_price) * mult * pos['lots']
                    invested = entry_price * mult * abs(pos['lots'])
                    pnl_pct = pnl / invested * 100 if invested > 0 else 0
                    trades.append({
                        'pnl': pnl, 'pnl_pct': pnl_pct,
                        'entry_di': entry_di_pos, 'exit_di': di,
                        'sym': pos['sym'], 'reason': exit_reason,
                    })
                    closed.append(pos)

            for pos in closed:
                positions.remove(pos)

            # Skip new entries if at capacity
            top_n = 1
            if len(positions) >= top_n:
                continue

            # -- ENTRY LOGIC (shared champion signal) --
            candidates = get_entry_signals(di, positions, top_n=top_n)
            if not candidates:
                continue

            entry_di = di + 1
            n_slots = top_n - len(positions)
            cap_per_slot = cash / max(1, n_slots)

            for sc_val, s, price, roc_val, zs_val in candidates[:n_slots]:
                sym = syms[s]
                mult = MULT.get(sym, DEF_MULT)
                contracts = max(1, int(cap_per_slot * 0.95 / (price * mult * (1 + COMM))))
                cost_in = price * mult * contracts * (1 + COMM)
                if cost_in > cash:
                    contracts = int(cash * 0.9 / (price * mult * (1 + COMM)))
                    cost_in = price * mult * contracts * (1 + COMM) if contracts > 0 else 0
                if contracts <= 0 or cost_in <= 0 or cost_in > cash:
                    continue
                cash -= cost_in
                pos = {
                    'si': s, 'entry_price': price, 'entry_di': entry_di,
                    'lots': contracts, 'dir': 1, 'sym': sym,
                    'signal_score': sc_val,
                }
                if exit_type == 'partial_profit':
                    pos['half_taken'] = False
                    pos['trail_start'] = None
                positions.append(pos)

        # Close remaining positions
        for pos in positions:
            ae = end_di - 1
            exit_price = C[pos['si'], min(ae, ND-1)]
            if np.isnan(exit_price) or exit_price <= 0:
                exit_price = pos['entry_price']
            mult = MULT.get(pos['sym'], DEF_MULT)
            mkt_val = exit_price * mult * abs(pos['lots'])
            cash += mkt_val - mkt_val * COMM
            pnl = (exit_price - pos['entry_price']) * mult * pos['lots']
            invested = pos['entry_price'] * mult * abs(pos['lots'])
            pnl_pct = pnl / invested * 100 if invested > 0 else 0
            trades.append({
                'pnl': pnl, 'pnl_pct': pnl_pct,
                'entry_di': pos['entry_di'], 'exit_di': ae,
                'sym': pos['sym'], 'reason': 'end_of_test',
            })

        n_days_test = end_di - start_di
        ann = annual_return(cash, CASH0, n_days_test)
        wr = np.mean([1 if t['pnl_pct'] > 0 else 0 for t in trades]) * 100 if trades else 0
        n_trades = len(trades)
        avg_pnl = np.mean([t['pnl_pct'] for t in trades]) if trades else 0

        # Max drawdown from daily equity
        if daily_equity:
            eq_arr = np.array(daily_equity)
            peak_arr = np.maximum.accumulate(eq_arr)
            dd_arr = (eq_arr - peak_arr) / peak_arr * 100
            mdd = np.min(dd_arr)
        else:
            mdd = 0.0

        # Sharpe ratio (annualized)
        if len(daily_equity) > 1:
            daily_rets = np.diff(daily_equity) / daily_equity[:-1]
            daily_rets = daily_rets[~np.isnan(daily_rets)]
            if len(daily_rets) > 10:
                sharpe = np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(252) if np.std(daily_rets) > 0 else 0
            else:
                sharpe = 0.0
        else:
            sharpe = 0.0

        # Average hold days
        if trades:
            avg_hold = np.mean([t['exit_di'] - t['entry_di'] for t in trades])
        else:
            avg_hold = 0.0

        return {
            'ann': ann, 'wr': wr, 'n': n_trades, 'avg_pnl': avg_pnl,
            'final_cash': cash, 'n_days': n_days_test, 'mdd': mdd,
            'sharpe': sharpe, 'avg_hold': avg_hold,
            'label': label, 'trades': trades,
        }

    # ================================================================
    # WALK-FORWARD HELPER
    # ================================================================
    wf_years = [2020, 2021, 2022, 2023, 2024, 2025]

    def walk_forward(exit_cfg):
        """Run walk-forward by year, return dict of {year: result}."""
        wf = {}
        for yr in wf_years:
            ts = te = None
            for di in range(ND):
                if dates[di].year == yr and ts is None:
                    ts = di
                if dates[di].year == yr + 1 and te is None:
                    te = di
            if ts is None:
                wf[yr] = None
                continue
            if te is None:
                te = ND
            r = backtest(exit_cfg, start_di=ts, end_di=te)
            wf[yr] = r
        return wf

    def print_wf(label, wf):
        vals = {yr: wf[yr]['ann'] if wf[yr] else 0 for yr in wf_years}
        avg = np.mean(list(vals.values()))
        pos = sum(1 for v in vals.values() if v > 0)
        mdds = [wf[yr]['mdd'] for yr in wf_years if wf[yr]]
        avg_mdd = np.mean(mdds) if mdds else 0
        sharpe_vals = [wf[yr]['sharpe'] for yr in wf_years if wf[yr]]
        avg_sharpe = np.mean(sharpe_vals) if sharpe_vals else 0
        row = f"  {label:<55} | {avg:>+8.1f}% |"
        for yr in wf_years:
            v = vals[yr]
            row += f" {v:>+8.1f}% |"
        row += f" {pos}/6 | {avg_mdd:>6.1f}% | {avg_sharpe:>6.2f}"
        print(row)
        return avg, pos, avg_sharpe

    def print_result(r):
        print(f"  {r['label']:<55} | {r['ann']:>+9.1f}% | {r['wr']:>5.1f}% | {r['n']:>5} | {r['avg_pnl']:>+7.3f}% | {r['mdd']:>+7.1f}% | {r['sharpe']:>6.2f} | {r['avg_hold']:>5.1f}d | {r['final_cash']:>11,.0f}")

    # ================================================================
    # SECTION A: FIXED HOLD SWEEP
    # ================================================================
    print(f"\n{'=' * 160}")
    print("  SECTION A: FIXED HOLD SWEEP (1,2,3,4,5,6,7,8,10,15,20 days)")
    print(f"{'=' * 160}")
    print(f"  {'Config':<55} | {'Ann':>10} | {'WR':>6} | {'N':>5} | {'AvgPnL':>8} | {'MDD':>8} | {'Sharpe':>6} | {'Hold':>5} | {'Final':>12}")
    print("-" * 160)

    hold_vals = [1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 20]
    a_results = []
    for h in hold_vals:
        cfg = {'exit_type': 'fixed', 'hold': h, 'label': f'A) Fixed hold={h}d'}
        r = backtest(cfg)
        a_results.append({**r, 'hold': h})
        print_result(r)

    best_a = max(a_results, key=lambda x: x['ann'])
    print(f"\n  BEST fixed hold: {best_a['hold']}d => {best_a['ann']:>+.1f}%, Sharpe={best_a['sharpe']:.2f}, MDD={best_a['mdd']:.1f}%")

    # Walk-forward for fixed hold
    print(f"\n  Walk-Forward: Fixed Hold Sweep")
    header = f"  {'Config':<55} | {'Avg':>8} |"
    for yr in wf_years:
        header += f" {yr:>8} |"
    header += f" {'WF+':>4} | {'AvgDD':>7} | {'Sharpe':>6}"
    print(header)
    print("-" * 170)
    for ar in a_results:
        cfg = {'exit_type': 'fixed', 'hold': ar['hold'], 'label': ar['label']}
        wf = walk_forward(cfg)
        print_wf(ar['label'], wf)

    # ================================================================
    # SECTION B: TRAILING STOP (Chandelier)
    # ================================================================
    print(f"\n{'=' * 160}")
    print("  SECTION B: TRAILING STOP (Chandelier) - N*ATR(14), Max hold sweep")
    print(f"{'=' * 160}")
    print(f"  {'Config':<55} | {'Ann':>10} | {'WR':>6} | {'N':>5} | {'AvgPnL':>8} | {'MDD':>8} | {'Sharpe':>6} | {'Hold':>5} | {'Final':>12}")
    print("-" * 160)

    b_results = []
    for n_atr in [1.5, 2.0, 2.5, 3.0, 4.0]:
        for max_h in [5, 10, 20]:
            cfg = {
                'exit_type': 'trailing_stop', 'n_atr': n_atr, 'max_hold': max_h,
                'label': f'B) Trail {n_atr}*ATR, max={max_h}d',
            }
            r = backtest(cfg)
            b_results.append({**r, 'n_atr': n_atr, 'max_hold': max_h})
            print_result(r)

    best_b = max(b_results, key=lambda x: x['ann'])
    print(f"\n  BEST trailing: {best_b['n_atr']}*ATR, max={best_b['max_hold']}d => {best_b['ann']:>+.1f}%, Sharpe={best_b['sharpe']:.2f}")

    print(f"\n  Walk-Forward: Top Trailing Stops")
    print(header)
    print("-" * 170)
    b_sorted = sorted(b_results, key=lambda x: -x['ann'])[:5]
    for br in b_sorted:
        cfg = {'exit_type': 'trailing_stop', 'n_atr': br['n_atr'], 'max_hold': br['max_hold'], 'label': br['label']}
        wf = walk_forward(cfg)
        print_wf(br['label'], wf)

    # ================================================================
    # SECTION C: PROFIT TARGET
    # ================================================================
    print(f"\n{'=' * 160}")
    print("  SECTION C: PROFIT TARGET - Exit when unrealized >= X%, max hold sweep")
    print(f"{'=' * 160}")
    print(f"  {'Config':<55} | {'Ann':>10} | {'WR':>6} | {'N':>5} | {'AvgPnL':>8} | {'MDD':>8} | {'Sharpe':>6} | {'Hold':>5} | {'Final':>12}")
    print("-" * 160)

    c_results = []
    for target in [1.0, 2.0, 3.0, 5.0, 8.0, 10.0]:
        for max_h in [5, 10]:
            cfg = {
                'exit_type': 'profit_target', 'target_pct': target, 'max_hold': max_h,
                'label': f'C) Target {target}%, max={max_h}d',
            }
            r = backtest(cfg)
            c_results.append({**r, 'target_pct': target, 'max_hold': max_h})
            print_result(r)

    best_c = max(c_results, key=lambda x: x['ann'])
    print(f"\n  BEST profit target: {best_c['target_pct']}%, max={best_c['max_hold']}d => {best_c['ann']:>+.1f}%, Sharpe={best_c['sharpe']:.2f}")

    print(f"\n  Walk-Forward: Top Profit Targets")
    print(header)
    print("-" * 170)
    c_sorted = sorted(c_results, key=lambda x: -x['ann'])[:5]
    for cr in c_sorted:
        cfg = {'exit_type': 'profit_target', 'target_pct': cr['target_pct'], 'max_hold': cr['max_hold'], 'label': cr['label']}
        wf = walk_forward(cfg)
        print_wf(cr['label'], wf)

    # ================================================================
    # SECTION D: ROC-BASED EXIT
    # ================================================================
    print(f"\n{'=' * 160}")
    print("  SECTION D: ROC-BASED EXIT - Exit when ROC(5) declines, max hold sweep")
    print(f"{'=' * 160}")
    print(f"  {'Config':<55} | {'Ann':>10} | {'WR':>6} | {'N':>5} | {'AvgPnL':>8} | {'MDD':>8} | {'Sharpe':>6} | {'Hold':>5} | {'Final':>12}")
    print("-" * 160)

    d_results = []
    for max_h in [3, 5, 7, 10]:
        cfg = {
            'exit_type': 'roc_exit', 'max_hold': max_h,
            'label': f'D) ROC declining exit, max={max_h}d',
        }
        r = backtest(cfg)
        d_results.append({**r, 'max_hold': max_h})
        print_result(r)

    best_d = max(d_results, key=lambda x: x['ann'])
    print(f"\n  BEST ROC exit: max={best_d['max_hold']}d => {best_d['ann']:>+.1f}%, Sharpe={best_d['sharpe']:.2f}")

    print(f"\n  Walk-Forward: ROC Exit")
    print(header)
    print("-" * 170)
    for dr in d_results:
        cfg = {'exit_type': 'roc_exit', 'max_hold': dr['max_hold'], 'label': dr['label']}
        wf = walk_forward(cfg)
        print_wf(dr['label'], wf)

    # ================================================================
    # SECTION E: Z-SCORE DECAY EXIT
    # ================================================================
    print(f"\n{'=' * 160}")
    print("  SECTION E: Z-SCORE DECAY EXIT - Exit when Z < 0, max hold sweep")
    print(f"{'=' * 160}")
    print(f"  {'Config':<55} | {'Ann':>10} | {'WR':>6} | {'N':>5} | {'AvgPnL':>8} | {'MDD':>8} | {'Sharpe':>6} | {'Hold':>5} | {'Final':>12}")
    print("-" * 160)

    e_results = []
    for max_h in [3, 5, 7, 10]:
        cfg = {
            'exit_type': 'zscore_decay', 'max_hold': max_h,
            'label': f'E) Z<0 exit, max={max_h}d',
        }
        r = backtest(cfg)
        e_results.append({**r, 'max_hold': max_h})
        print_result(r)

    best_e = max(e_results, key=lambda x: x['ann'])
    print(f"\n  BEST Z-decay exit: max={best_e['max_hold']}d => {best_e['ann']:>+.1f}%, Sharpe={best_e['sharpe']:.2f}")

    print(f"\n  Walk-Forward: Z-Score Decay")
    print(header)
    print("-" * 170)
    for er in e_results:
        cfg = {'exit_type': 'zscore_decay', 'max_hold': er['max_hold'], 'label': er['label']}
        wf = walk_forward(cfg)
        print_wf(er['label'], wf)

    # ================================================================
    # SECTION F: PARTIAL PROFIT TAKING
    # ================================================================
    print(f"\n{'=' * 160}")
    print("  SECTION F: PARTIAL PROFIT - 50% at +3%, rest trailing 2*ATR, max 10d")
    print(f"{'=' * 160}")
    print(f"  {'Config':<55} | {'Ann':>10} | {'WR':>6} | {'N':>5} | {'AvgPnL':>8} | {'MDD':>8} | {'Sharpe':>6} | {'Hold':>5} | {'Final':>12}")
    print("-" * 160)

    f_results = []
    for first_target in [2.0, 3.0, 5.0]:
        for trail_n in [1.5, 2.0, 3.0]:
            for max_h in [7, 10, 15]:
                cfg = {
                    'exit_type': 'partial_profit',
                    'first_target_pct': first_target,
                    'trail_n_atr': trail_n,
                    'max_hold': max_h,
                    'label': f'F) Partial {first_target}%/trail{trail_n}atr/max{max_h}d',
                }
                r = backtest(cfg)
                f_results.append({**r, 'first_target': first_target, 'trail_n': trail_n, 'max_hold': max_h})
                print_result(r)

    best_f = max(f_results, key=lambda x: x['ann'])
    print(f"\n  BEST partial: target={best_f['first_target']}%, trail={best_f['trail_n']}atr, max={best_f['max_hold']}d => {best_f['ann']:>+.1f}%")

    print(f"\n  Walk-Forward: Top 5 Partial Profit")
    print(header)
    print("-" * 170)
    f_sorted = sorted(f_results, key=lambda x: -x['ann'])[:5]
    for fr in f_sorted:
        cfg = {'exit_type': 'partial_profit', 'first_target_pct': fr['first_target'],
               'trail_n_atr': fr['trail_n'], 'max_hold': fr['max_hold'], 'label': fr['label']}
        wf = walk_forward(cfg)
        print_wf(fr['label'], wf)

    # ================================================================
    # SECTION G: INTRADAY REVERSAL EXIT
    # ================================================================
    print(f"\n{'=' * 160}")
    print("  SECTION G: INTRADAY REVERSAL EXIT - C<O AND L<entry => exit, max hold sweep")
    print(f"{'=' * 160}")
    print(f"  {'Config':<55} | {'Ann':>10} | {'WR':>6} | {'N':>5} | {'AvgPnL':>8} | {'MDD':>8} | {'Sharpe':>6} | {'Hold':>5} | {'Final':>12}")
    print("-" * 160)

    g_results = []
    for max_h in [1, 2, 3, 5]:
        cfg = {
            'exit_type': 'intraday_reversal', 'max_hold': max_h,
            'label': f'G) Intraday reversal, max={max_h}d',
        }
        r = backtest(cfg)
        g_results.append({**r, 'max_hold': max_h})
        print_result(r)

    best_g = max(g_results, key=lambda x: x['ann'])
    print(f"\n  BEST intraday reversal: max={best_g['max_hold']}d => {best_g['ann']:>+.1f}%")

    print(f"\n  Walk-Forward: Intraday Reversal")
    print(header)
    print("-" * 170)
    for gr in g_results:
        cfg = {'exit_type': 'intraday_reversal', 'max_hold': gr['max_hold'], 'label': gr['label']}
        wf = walk_forward(cfg)
        print_wf(gr['label'], wf)

    # ================================================================
    # SECTION H: MOMENTUM DECAY EXIT
    # ================================================================
    print(f"\n{'=' * 160}")
    print("  SECTION H: MOMENTUM DECAY EXIT - ROC(1)<-1% => exit, max hold sweep")
    print(f"{'=' * 160}")
    print(f"  {'Config':<55} | {'Ann':>10} | {'WR':>6} | {'N':>5} | {'AvgPnL':>8} | {'MDD':>8} | {'Sharpe':>6} | {'Hold':>5} | {'Final':>12}")
    print("-" * 160)

    h_results = []
    for thresh in [-0.5, -1.0, -1.5, -2.0]:
        for max_h in [2, 3, 5]:
            cfg = {
                'exit_type': 'momentum_decay', 'decay_thresh': thresh, 'max_hold': max_h,
                'label': f'H) ROC(1)<{thresh}%, max={max_h}d',
            }
            r = backtest(cfg)
            h_results.append({**r, 'decay_thresh': thresh, 'max_hold': max_h})
            print_result(r)

    best_h = max(h_results, key=lambda x: x['ann'])
    print(f"\n  BEST momentum decay: thresh={best_h['decay_thresh']}%, max={best_h['max_hold']}d => {best_h['ann']:>+.1f}%")

    print(f"\n  Walk-Forward: Top 3 Momentum Decay")
    print(header)
    print("-" * 170)
    h_sorted = sorted(h_results, key=lambda x: -x['ann'])[:3]
    for hr in h_sorted:
        cfg = {'exit_type': 'momentum_decay', 'decay_thresh': hr['decay_thresh'],
               'max_hold': hr['max_hold'], 'label': hr['label']}
        wf = walk_forward(cfg)
        print_wf(hr['label'], wf)

    # ================================================================
    # SECTION I: ADAPTIVE HOLD BY SIGNAL STRENGTH
    # ================================================================
    print(f"\n{'=' * 160}")
    print("  SECTION I: ADAPTIVE HOLD BY SIGNAL STRENGTH")
    print(f"  Score > 10: hold 1d | 5-10: hold 2d | 3-5: hold 3d")
    print(f"{'=' * 160}")
    print(f"  {'Config':<55} | {'Ann':>10} | {'WR':>6} | {'N':>5} | {'AvgPnL':>8} | {'MDD':>8} | {'Sharpe':>6} | {'Hold':>5} | {'Final':>12}")
    print("-" * 160)

    i_cfg = {'exit_type': 'adaptive_hold', 'label': 'I) Adaptive hold (score>10:1d, 5-10:2d, <5:3d)'}
    r_i = backtest(i_cfg)
    print_result(r_i)

    print(f"\n  Walk-Forward: Adaptive Hold")
    print(header)
    print("-" * 170)
    wf_i = walk_forward(i_cfg)
    print_wf(i_cfg['label'], wf_i)

    # ================================================================
    # SECTION J: OVERNIGHT GAP EXIT
    # ================================================================
    print(f"\n{'=' * 160}")
    print("  SECTION J: OVERNIGHT GAP EXIT")
    print(f"  Gap down next day => exit at open. Gap up/flat => hold through day, exit at close.")
    print(f"{'=' * 160}")
    print(f"  {'Config':<55} | {'Ann':>10} | {'WR':>6} | {'N':>5} | {'AvgPnL':>8} | {'MDD':>8} | {'Sharpe':>6} | {'Hold':>5} | {'Final':>12}")
    print("-" * 160)

    j_cfg = {'exit_type': 'gap_exit', 'label': 'J) Overnight gap exit'}
    r_j = backtest(j_cfg)
    print_result(r_j)

    print(f"\n  Walk-Forward: Overnight Gap Exit")
    print(header)
    print("-" * 170)
    wf_j = walk_forward(j_cfg)
    print_wf(j_cfg['label'], wf_j)

    # ================================================================
    # GRAND SUMMARY
    # ================================================================
    print(f"\n{'=' * 170}")
    print("  GRAND SUMMARY: ALL EXIT STRATEGIES RANKED BY ANNUAL RETURN")
    print(f"{'=' * 170}")
    print(f"  {'Rank':>4} | {'Config':<55} | {'Ann':>10} | {'WR':>6} | {'Sharpe':>6} | {'MDD':>8} | {'AvgHold':>8} | {'Final':>12}")
    print("-" * 170)

    all_results = []
    # Add champion baseline (fixed hold=1)
    champ_a1 = [x for x in a_results if x['hold'] == 1][0]
    all_results.append({**champ_a1, 'section': 'A (baseline)'})
    # Best from each section
    for sec, best, letter in [
        ('A', best_a, 'A'), ('B', best_b, 'B'), ('C', best_c, 'C'),
        ('D', best_d, 'D'), ('E', best_e, 'E'), ('F', best_f, 'F'),
        ('G', best_g, 'G'), ('H', best_h, 'H'),
    ]:
        all_results.append({**best, 'section': sec})
    all_results.append({**r_i, 'section': 'I'})
    all_results.append({**r_j, 'section': 'J'})

    # Also add top 3 from fixed hold sweep for reference
    for ar in sorted(a_results, key=lambda x: -x['ann'])[:3]:
        if ar['hold'] != 1:
            all_results.append({**ar, 'section': 'A (top hold)'})

    all_results.sort(key=lambda x: -x['ann'])
    for rank, r in enumerate(all_results, 1):
        marker = " *** CHAMPION BASELINE" if r.get('hold') == 1 and r.get('section', '').startswith('A') else ""
        marker = " <<< OVERALL BEST" if rank == 1 else marker
        print(f"  {rank:>4} | {r['label']:<55} | {r['ann']:>+9.1f}% | {r['wr']:>5.1f}% | {r['sharpe']:>6.2f} | {r['mdd']:>+7.1f}% | {r['avg_hold']:>7.1f}d | {r['final_cash']:>11,.0f}{marker}")

    # ================================================================
    # WALK-FORWARD COMPARISON: BEST FROM EACH SECTION
    # ================================================================
    print(f"\n{'=' * 170}")
    print("  WALK-FORWARD COMPARISON: BEST FROM EACH SECTION")
    print(f"{'=' * 170}")
    header = f"  {'Section':<55} | {'Avg':>8} |"
    for yr in wf_years:
        header += f" {yr:>8} |"
    header += f" {'WF+':>4} | {'AvgDD':>7} | {'Sharpe':>6}"
    print(header)
    print("-" * 170)

    # Build list of configs to walk-forward
    wf_configs = []
    # Best from each section
    wf_configs.append({'exit_type': 'fixed', 'hold': best_a['hold'], 'label': f"A) Best fixed={best_a['hold']}d"})
    wf_configs.append({'exit_type': 'trailing_stop', 'n_atr': best_b['n_atr'], 'max_hold': best_b['max_hold'],
                        'label': f"B) Best trail={best_b['n_atr']}*ATR,max{best_b['max_hold']}d"})
    wf_configs.append({'exit_type': 'profit_target', 'target_pct': best_c['target_pct'], 'max_hold': best_c['max_hold'],
                        'label': f"C) Best target={best_c['target_pct']}%,max{best_c['max_hold']}d"})
    wf_configs.append({'exit_type': 'roc_exit', 'max_hold': best_d['max_hold'],
                        'label': f"D) Best ROC exit,max{best_d['max_hold']}d"})
    wf_configs.append({'exit_type': 'zscore_decay', 'max_hold': best_e['max_hold'],
                        'label': f"E) Best Z-decay,max{best_e['max_hold']}d"})
    wf_configs.append({'exit_type': 'partial_profit', 'first_target_pct': best_f['first_target'],
                        'trail_n_atr': best_f['trail_n'], 'max_hold': best_f['max_hold'],
                        'label': f"F) Best partial={best_f['first_target']}%,trail{best_f['trail_n']}atr"})
    wf_configs.append({'exit_type': 'intraday_reversal', 'max_hold': best_g['max_hold'],
                        'label': f"G) Best reversal,max{best_g['max_hold']}d"})
    wf_configs.append({'exit_type': 'momentum_decay', 'decay_thresh': best_h['decay_thresh'],
                        'max_hold': best_h['max_hold'],
                        'label': f"H) Best mom.decay,ROC1<{best_h['decay_thresh']}%,max{best_h['max_hold']}d"})
    wf_configs.append(i_cfg)
    wf_configs.append(j_cfg)

    wf_summary = []
    for cfg in wf_configs:
        wf = walk_forward(cfg)
        avg, pos, avg_sharpe = print_wf(cfg['label'], wf)
        wf_summary.append({'label': cfg['label'], 'avg': avg, 'pos': pos, 'avg_sharpe': avg_sharpe, 'cfg': cfg})

    # ================================================================
    # FINAL ANSWERS
    # ================================================================
    print(f"\n{'=' * 170}")
    print("  FINAL ANSWERS")
    print(f"{'=' * 170}")

    # Q1: Is fixed hold=1 truly optimal?
    print(f"\n  Q1: Is fixed hold=1 truly optimal?")
    print(f"  {'='*60}")
    champ_ann = champ_a1['ann']
    better_holds = [(h, r) for h, r in [(x['hold'], x) for x in a_results] if r['ann'] > champ_ann and r['hold'] != 1]
    if better_holds:
        print(f"  NO - Other hold periods beat hold=1:")
        for h, r in sorted(better_holds, key=lambda x: -x[1]['ann']):
            print(f"    hold={h}d: {r['ann']:>+.1f}% vs hold=1d: {champ_ann:>+.1f}%")
    else:
        print(f"  YES - Fixed hold=1 ({champ_ann:>+.1f}%) is the best among all fixed hold periods.")
        # Find 2nd best
        a_sorted = sorted(a_results, key=lambda x: -x['ann'])
        if len(a_sorted) > 1:
            print(f"    2nd best: hold={a_sorted[1]['hold']}d ({a_sorted[1]['ann']:>+.1f}%)")

    # Q2: Does any exit strategy beat +333.5%?
    print(f"\n  Q2: Does any exit strategy beat +333.5%?")
    print(f"  {'='*60}")
    overall_best = max(all_results, key=lambda x: x['ann'])
    if overall_best['ann'] > 333.5:
        print(f"  YES - {overall_best['label']}: {overall_best['ann']:>+.1f}%")
    else:
        print(f"  NO - Best is {overall_best['label']}: {overall_best['ann']:>+.1f}%")
    # List all that beat 333.5
    beaters = [r for r in all_results if r['ann'] > 333.5]
    if beaters:
        print(f"  All strategies beating +333.5%:")
        for r in sorted(beaters, key=lambda x: -x['ann']):
            print(f"    {r['label']}: {r['ann']:>+.1f}%")

    # Q3: Best risk-adjusted returns
    print(f"\n  Q3: Best risk-adjusted returns (highest Sharpe, lowest MDD)?")
    print(f"  {'='*60}")
    best_sharpe = max(all_results, key=lambda x: x.get('sharpe', 0))
    lowest_mdd = max(all_results, key=lambda x: x.get('mdd', -999))  # least negative
    print(f"  Highest Sharpe: {best_sharpe['label']}: Sharpe={best_sharpe['sharpe']:.2f}, Ann={best_sharpe['ann']:>+.1f}%, MDD={best_sharpe['mdd']:.1f}%")
    print(f"  Lowest MDD:     {lowest_mdd['label']}: MDD={lowest_mdd['mdd']:.1f}%, Ann={lowest_mdd['ann']:>+.1f}%, Sharpe={lowest_mdd['sharpe']:.2f}")

    # Q4: Overnight gap exit
    print(f"\n  Q4: Overnight gap exit results?")
    print(f"  {'='*60}")
    print(f"  {r_j['label']}: Ann={r_j['ann']:>+.1f}%, WR={r_j['wr']:.1f}%, Sharpe={r_j['sharpe']:.2f}, MDD={r_j['mdd']:.1f}%")
    print(f"  vs Baseline hold=1: Ann={champ_a1['ann']:>+.1f}%, Sharpe={champ_a1['sharpe']:.2f}, MDD={champ_a1['mdd']:.1f}%")
    diff = r_j['ann'] - champ_a1['ann']
    print(f"  Difference: {diff:>+.1f}%")

    # Q5: Adaptive hold
    print(f"\n  Q5: Adaptive hold by signal strength results?")
    print(f"  {'='*60}")
    print(f"  {r_i['label']}: Ann={r_i['ann']:>+.1f}%, WR={r_i['wr']:.1f}%, Sharpe={r_i['sharpe']:.2f}, MDD={r_i['mdd']:.1f}%")
    print(f"  vs Baseline hold=1: Ann={champ_a1['ann']:>+.1f}%, Sharpe={champ_a1['sharpe']:.2f}, MDD={champ_a1['mdd']:.1f}%")
    diff_i = r_i['ann'] - champ_a1['ann']
    print(f"  Difference: {diff_i:>+.1f}%")

    # ================================================================
    # TRADE REASON ANALYSIS (for the best non-fixed exit)
    # ================================================================
    print(f"\n{'=' * 170}")
    print("  EXIT REASON ANALYSIS (Top exit strategies)")
    print(f"{'=' * 170}")

    # Analyze trade exit reasons for top strategies
    analysis_cfgs = []
    if best_b['ann'] > champ_a1['ann']:
        analysis_cfgs.append({
            'exit_type': 'trailing_stop', 'n_atr': best_b['n_atr'], 'max_hold': best_b['max_hold'],
            'label': f"Best Trailing: {best_b['n_atr']}*ATR,max{best_b['max_hold']}d"
        })
    if best_c['ann'] > champ_a1['ann']:
        analysis_cfgs.append({
            'exit_type': 'profit_target', 'target_pct': best_c['target_pct'], 'max_hold': best_c['max_hold'],
            'label': f"Best Target: {best_c['target_pct']}%,max{best_c['max_hold']}d"
        })
    analysis_cfgs.append({
        'exit_type': 'fixed', 'hold': 1, 'label': 'Baseline hold=1'
    })
    analysis_cfgs.append(j_cfg)
    analysis_cfgs.append(i_cfg)

    for acfg in analysis_cfgs:
        r = backtest(acfg)
        if not r['trades']:
            continue
        trades = r['trades']
        reasons = {}
        for t in trades:
            reason = t.get('reason', 'unknown')
            if reason not in reasons:
                reasons[reason] = {'count': 0, 'pnl': 0, 'win': 0}
            reasons[reason]['count'] += 1
            reasons[reason]['pnl'] += t['pnl_pct']
            if t['pnl_pct'] > 0:
                reasons[reason]['win'] += 1

        print(f"\n  {acfg['label']} ({len(trades)} trades):")
        for reason, stats in sorted(reasons.items(), key=lambda x: -x[1]['count']):
            wr = stats['win'] / stats['count'] * 100 if stats['count'] > 0 else 0
            avg = stats['pnl'] / stats['count'] if stats['count'] > 0 else 0
            print(f"    {reason:<25}: {stats['count']:>4} trades ({stats['count']/len(trades)*100:.0f}%), WR={wr:.1f}%, AvgPnL={avg:>+.3f}%")

    elapsed = time.time() - t_start
    print(f"\n  Total elapsed: {elapsed:.0f}s")
    print("=" * 170)


if __name__ == '__main__':
    main()
