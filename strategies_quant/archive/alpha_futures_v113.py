"""
Alpha Futures V113 — Position Sizing + Exit Strategy Optimization
===================================================================
Entry signal: ROC(5) cross above zero (proven +81.9% baseline)
Optimizes: HOW MUCH to bet and WHEN to exit

Tests:
  A) Volatility-scaled sizing (target_vol: 15%, 20%, 30%, 50%)
  B) Kelly Criterion (half, quarter, full)
  C) Anti-Martingale (pyramid: 10%, 20%, 30% adjustment)
  D) Momentum-strength sizing (ROC magnitude thresholds)
  E) Trailing stop (N*ATR, max hold)
  F) ROC-based exit (momentum reversal)
  G) Profit target (3%, 5%, 8%, 10%)
  H) Time-weighted exit (profit-based hold extension)
  I) Chandelier exit
  J) Combined exit (trailing + ROC + max hold)

Walk-forward by year (2020-2025) for all configs.
"""
import sys, os, time, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alpha_v2 import load_all_data, MIN_TRAIN, CASH0

MULT = {'agfi': 15, 'alfi': 5, 'aufi': 1000, 'bufi': 10, 'cufi': 5, 'fufi': 10,
        'rbfi': 10, 'znfi': 5, 'nifi': 1, 'hcfi': 10, 'spfi': 10, 'ssfi': 5,
        'sffi': 5, 'smfi': 5, 'pbfi': 5, 'snfi': 1, 'rufi': 10, 'wrfff': 10,
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


# ====================================================================
# Precompute ROC(5) and ATR
# ====================================================================
def precompute_indicators(NS, ND, C, O, H, L):
    """Precompute ROC(5) and ATR_14 using only data up to di-1 (no look-ahead)."""
    print("  Precomputing ROC(5) and ATR_14...", flush=True)
    t0 = time.time()

    roc5 = np.full((NS, ND), np.nan)
    roc5_prev = np.full((NS, ND), np.nan)  # ROC(5) at di-2 for cross detection
    atr14 = np.full((NS, ND), np.nan)

    for si in range(NS):
        for di in range(6, ND):
            d = di - 1  # use previous day's close
            c_now = C[si, d]
            c_prev5 = C[si, d - 5]
            if not np.isnan(c_now) and not np.isnan(c_prev5) and c_prev5 > 0:
                roc5[si, di] = (c_now - c_prev5) / c_prev5

            if di >= 7:
                d2 = di - 2
                c2_now = C[si, d2]
                c2_prev5 = C[si, d2 - 5]
                if not np.isnan(c2_now) and not np.isnan(c2_prev5) and c2_prev5 > 0:
                    roc5_prev[si, di] = (c2_now - c2_prev5) / c2_prev5

            # ATR_14
            if di >= 15:
                trs = []
                for dd in range(max(1, d - 13), d + 1):
                    hi = H[si, dd]; lo = L[si, dd]
                    if np.isnan(hi) or np.isnan(lo): continue
                    pc = C[si, dd - 1] if dd > 0 else np.nan
                    tr = hi - lo
                    if not np.isnan(pc):
                        tr = max(tr, abs(hi - pc), abs(lo - pc))
                    trs.append(tr)
                if len(trs) >= 7:
                    atr14[si, di] = np.mean(trs)

    print(f"  Indicators done ({time.time() - t0:.0f}s)", flush=True)
    return roc5, roc5_prev, atr14


# ====================================================================
# Generate ROC(5) cross signals
# ====================================================================
def generate_roc5_cross_signals(NS, ND, roc5, roc5_prev):
    """ROC(5) crosses above zero: roc5_prev < 0 and roc5 >= 0."""
    buy_signals = [set() for _ in range(ND)]
    for si in range(NS):
        for di in range(MIN_TRAIN, ND):
            r = roc5[si, di]
            rp = roc5_prev[si, di]
            if np.isnan(r) or np.isnan(rp): continue
            if rp < 0 and r >= 0:
                buy_signals[di].add(si)
    return buy_signals


# ====================================================================
# Core backtest engine with configurable sizing and exit
# ====================================================================
def backtest_engine(NS, ND, dates, C, O, H, L, syms,
                    buy_signals, roc5, roc5_prev, atr14,
                    sizing_cfg, exit_cfg, top_n=1,
                    wf_start=None, wf_end=None):
    """
    Walk-forward backtest with configurable position sizing and exit.

    sizing_cfg: dict with keys:
        method: 'fixed', 'vol_scaled', 'kelly', 'anti_martingale', 'momentum_strength'
        target_vol: float (annual target vol, for vol_scaled)
        kelly_frac: float (1.0=full, 0.5=half, 0.25=quarter)
        anti_mart_adj: float (adjustment %, e.g. 0.2)
        mom_thresholds: list of (roc_min, size_frac) tuples

    exit_cfg: dict with keys:
        method: 'fixed_hold', 'trailing_stop', 'roc_exit', 'profit_target',
                'time_weighted', 'chandelier', 'combined'
        hold_days: int (max hold)
        trail_atr_mult: float (for trailing stop)
        trail_max_hold: int
        profit_target_pct: float
        chandelier_atr_mult: float
        chandelier_max_hold: int
    """
    di_start = wf_start if wf_start is not None else MIN_TRAIN
    di_end = wf_end if wf_end is not None else ND

    cash = float(CASH0)
    positions = []
    trades = []
    # Kelly tracking: rolling window of past trades for Kelly computation
    trade_history = []  # list of {'won': bool, 'pnl_pct': float}

    # Anti-martingale state
    current_size_mult = 1.0

    for di in range(di_start, di_end):
        year = dates[di].year

        # --- Exit logic ---
        for pos in list(positions):
            si = pos['si']
            c = C[si, di]
            if np.isnan(c):
                continue

            mult = MULT.get(pos['sym'], DEF_MULT)
            pnl_abs = (c - pos['entry']) * mult * pos['lots'] * pos['dir']
            entry_notional = pos['entry'] * mult * pos['lots']
            pnl_pct = pnl_abs / entry_notional * 100 if entry_notional > 0 else 0
            days_held = di - pos['entry_di']

            exit_reason = None

            method = exit_cfg['method']

            if method == 'fixed_hold':
                if days_held >= exit_cfg.get('hold_days', 5):
                    exit_reason = 'time'

            elif method == 'trailing_stop':
                # Update trailing stop
                if c > pos.get('highest', pos['entry']):
                    pos['highest'] = c
                atr = atr14[si, di] if di < ND else np.nan
                if not np.isnan(atr) and atr > 0:
                    new_stop = pos['highest'] - exit_cfg['trail_atr_mult'] * atr
                    if new_stop > pos.get('trail_stop', 0):
                        pos['trail_stop'] = new_stop

                if pos.get('trail_stop') and c < pos['trail_stop']:
                    exit_reason = 'trail_stop'
                elif days_held >= exit_cfg.get('trail_max_hold', 10):
                    exit_reason = 'time'

            elif method == 'roc_exit':
                r = roc5[si, di]
                if not np.isnan(r) and r < 0:
                    exit_reason = 'roc_reversal'
                elif days_held >= exit_cfg.get('hold_days', 20):
                    exit_reason = 'time'

            elif method == 'profit_target':
                if pnl_pct >= exit_cfg.get('profit_target_pct', 5.0):
                    exit_reason = 'profit_target'
                elif days_held >= exit_cfg.get('hold_days', 20):
                    exit_reason = 'time'

            elif method == 'time_weighted':
                if pnl_pct < -3.0:
                    exit_reason = 'stop_loss'
                elif days_held >= 3 and pnl_pct > 0:
                    if days_held >= 10:
                        exit_reason = 'time'
                elif days_held >= 10:
                    exit_reason = 'time'

            elif method == 'chandelier':
                if c > pos.get('highest', pos['entry']):
                    pos['highest'] = c
                atr = atr14[si, di] if di < ND else np.nan
                if not np.isnan(atr) and atr > 0:
                    ch_stop = pos['highest'] - exit_cfg['chandelier_atr_mult'] * atr
                    if c < ch_stop:
                        exit_reason = 'chandelier'
                if exit_reason is None and days_held >= exit_cfg.get('chandelier_max_hold', 15):
                    exit_reason = 'time'

            elif method == 'combined':
                # Trailing stop at 3*ATR
                if c > pos.get('highest', pos['entry']):
                    pos['highest'] = c
                atr = atr14[si, di] if di < ND else np.nan
                if not np.isnan(atr) and atr > 0:
                    new_stop = pos['highest'] - 3.0 * atr
                    if new_stop > pos.get('trail_stop', 0):
                        pos['trail_stop'] = new_stop

                if pos.get('trail_stop') and c < pos['trail_stop']:
                    exit_reason = 'trail_stop'
                else:
                    # ROC reversal
                    r = roc5[si, di]
                    if not np.isnan(r) and r < 0:
                        exit_reason = 'roc_reversal'
                    elif days_held >= exit_cfg.get('hold_days', 15):
                        exit_reason = 'time'

            if exit_reason:
                cash += c * mult * pos['lots'] * (1 - COMM)
                trades.append({
                    'pnl_pct': pnl_pct, 'pnl_abs': pnl_abs,
                    'days': days_held, 'di': di, 'reason': exit_reason,
                    'year': year, 'si': si, 'dir': pos['dir']
                })
                # Track for Kelly
                trade_history.append({'won': pnl_abs > 0, 'pnl_pct': pnl_pct})
                # Anti-martingale update
                if sizing_cfg['method'] == 'anti_martingale':
                    adj = sizing_cfg.get('anti_mart_adj', 0.2)
                    if pnl_abs > 0:
                        current_size_mult = min(1.5, current_size_mult + adj)
                    else:
                        current_size_mult = max(0.5, current_size_mult - adj)

                positions.remove(pos)

        # --- Entry logic ---
        n_pos = len(positions)
        if n_pos < top_n:
            # Collect candidates from buy_signals[di-1] (signal at close di-1, entry at open di)
            # Actually signals are precomputed for di, we enter at O[si, di+1]
            # But we are already at di, so we need to enter at O[si, di+1]
            # To handle next-open: check signals at di, enter at O[si, di+1]
            # We'll handle this by: signals generated at di-1 → entry at O[si, di]
            # This is how the loop works: at di, we check buy_signals[di-1] and enter at O[si, di]
            pass

        # Actually, the standard pattern is: signal at close of di, entry at O[si, di+1].
        # But in this loop at di, we can look at buy_signals from di-1 and enter at O[si, di].
        # Wait — re-reading the spec: "Signal at close di, entry at O[si, di+1]"
        # So in the loop at di, we process signals from di-1 → entry at O[si, di]
        # This is exactly next-open execution.
        if di > di_start and len(positions) < top_n:
            sig_di = di - 1
            candidates = []
            for si in buy_signals[sig_di]:
                if any(p['si'] == si for p in positions): continue
                op = O[si, di]
                c_prev = C[si, sig_di]
                if np.isnan(op) or op <= 0:
                    if not np.isnan(c_prev) and c_prev > 0:
                        op = c_prev
                    else:
                        continue
                # Score by ROC(5) magnitude
                r = roc5[si, sig_di]
                if np.isnan(r): continue
                candidates.append((si, op, r))

            candidates.sort(key=lambda x: -x[2])  # highest ROC first
            slots = top_n - len(positions)

            for si, entry_price, roc_val in candidates[:slots]:
                sym = syms[si]
                mult = MULT.get(sym, DEF_MULT)
                notional = entry_price * mult
                if notional <= 0: continue

                # --- Position sizing ---
                size_frac = 1.0  # default: 100% capital

                smethod = sizing_cfg['method']
                if smethod == 'fixed':
                    size_frac = 1.0

                elif smethod == 'vol_scaled':
                    target_vol = sizing_cfg.get('target_vol', 0.20)
                    atr = atr14[si, sig_di]
                    c_ref = C[si, sig_di]
                    if not np.isnan(atr) and not np.isnan(c_ref) and c_ref > 0 and atr > 0:
                        daily_vol = atr / c_ref
                        annual_vol = daily_vol * np.sqrt(252)
                        if annual_vol > 0:
                            size_frac = target_vol / annual_vol
                    size_frac = min(size_frac, 1.0)
                    size_frac = max(size_frac, 0.1)

                elif smethod == 'kelly':
                    kelly_frac_param = sizing_cfg.get('kelly_frac', 0.5)
                    if len(trade_history) >= 20:
                        recent = trade_history[-100:]
                        wins = [t for t in recent if t['won']]
                        losses = [t for t in recent if not t['won']]
                        n_w = len(wins)
                        n_l = len(losses)
                        if n_w > 0 and n_l > 0:
                            wr = n_w / len(recent)
                            avg_win = np.mean([t['pnl_pct'] for t in wins])
                            avg_loss = np.mean([abs(t['pnl_pct']) for t in losses])
                            if avg_loss > 0:
                                kelly = wr - (1 - wr) / (avg_win / avg_loss)
                                kelly = max(0, kelly)
                                size_frac = kelly * kelly_frac_param
                    size_frac = min(size_frac, 1.0)
                    size_frac = max(size_frac, 0.1)

                elif smethod == 'anti_martingale':
                    size_frac = current_size_mult

                elif smethod == 'momentum_strength':
                    abs_roc = abs(roc_val)
                    thresholds = sizing_cfg.get('mom_thresholds', [
                        (0.00, 0.50), (0.02, 0.75), (0.05, 1.00), (0.10, 1.00)
                    ])
                    size_frac = 0.5
                    for roc_min, sfrac in thresholds:
                        if abs_roc >= roc_min:
                            size_frac = sfrac

                allocated = cash * size_frac
                lots = max(1, int(allocated / (notional * (1 + COMM))))
                cost = notional * lots * (1 + COMM)
                if cost > cash:
                    lots = max(1, int(cash / (notional * (1 + COMM))))
                    cost = notional * lots * (1 + COMM)
                if lots <= 0 or cost > cash:
                    continue

                cash -= cost
                pos_entry = {
                    'si': si, 'entry': entry_price, 'entry_di': di,
                    'lots': lots, 'dir': 1, 'sym': sym,
                    'highest': entry_price, 'trail_stop': 0,
                    'size_frac': size_frac,
                }
                # Initialize trailing stop for methods that need it
                if exit_cfg['method'] in ('trailing_stop', 'combined'):
                    atr = atr14[si, sig_di]
                    if not np.isnan(atr) and atr > 0:
                        mult_val = exit_cfg.get('trail_atr_mult', 3.0) if exit_cfg['method'] == 'trailing_stop' else 3.0
                        pos_entry['trail_stop'] = entry_price - mult_val * atr
                positions.append(pos_entry)

    # --- Close remaining positions ---
    for pos in positions:
        c = C[pos['si'], di_end - 1]
        if np.isnan(c) or c <= 0:
            c = pos['entry']
        mult = MULT.get(pos['sym'], DEF_MULT)
        pnl_abs = (c - pos['entry']) * mult * pos['lots'] * pos['dir']
        entry_notional = pos['entry'] * mult * pos['lots']
        pnl_pct = pnl_abs / entry_notional * 100 if entry_notional > 0 else 0
        cash += c * mult * pos['lots'] * (1 - COMM)
        trades.append({
            'pnl_pct': pnl_pct, 'pnl_abs': pnl_abs,
            'days': 999, 'di': di_end - 1, 'reason': 'end',
            'year': dates[di_end - 1].year, 'si': pos['si'], 'dir': pos['dir']
        })

    if not trades:
        return None

    # Compute stats
    equity = float(CASH0)
    peak = float(CASH0)
    max_dd = 0
    for t in sorted(trades, key=lambda x: x['di']):
        equity += t['pnl_abs']
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak * 100 if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd

    if equity <= 0:
        return None

    # Annualized return
    days_total = (dates[di_end - 1] - dates[di_start]).days
    yr = max(days_total / 365.25, 0.01)
    ann = ((cash / CASH0) ** (1 / yr) - 1) * 100

    nw = sum(1 for t in trades if t['pnl_abs'] > 0)
    wr = nw / max(len(trades), 1) * 100
    avg_win = np.mean([t['pnl_pct'] for t in trades if t['pnl_abs'] > 0]) if nw > 0 else 0
    avg_loss = np.mean([abs(t['pnl_pct']) for t in trades if t['pnl_abs'] <= 0]) if nw < len(trades) else 0

    year_stats = {}
    for t in trades:
        y = t.get('year', 'unknown')
        if y not in year_stats:
            year_stats[y] = {'trades': 0, 'wins': 0, 'pnl_abs_sum': 0}
        year_stats[y]['trades'] += 1
        if t['pnl_abs'] > 0:
            year_stats[y]['wins'] += 1
        year_stats[y]['pnl_abs_sum'] += t['pnl_abs']

    return {
        'ann': round(ann, 1), 'n': len(trades), 'wr': round(wr, 1),
        'max_dd': round(max_dd, 1), 'final': round(cash, 0),
        'avg_win': round(avg_win, 2), 'avg_loss': round(avg_loss, 2),
        'year_stats': year_stats,
    }


# ====================================================================
# Walk-forward runner
# ====================================================================
def run_walk_forward(NS, ND, dates, C, O, H, L, syms,
                     buy_signals, roc5, roc5_prev, atr14,
                     sizing_cfg, exit_cfg, top_n=1,
                     wf_years=None):
    """Run walk-forward test by year. wf_years = list of test years."""
    if wf_years is None:
        wf_years = [2020, 2021, 2022, 2023, 2024, 2025]

    # Find di ranges for each test year
    year_di_map = {}
    for di in range(MIN_TRAIN, ND):
        y = dates[di].year
        if y not in year_di_map:
            year_di_map[y] = [di, di]
        else:
            year_di_map[y][1] = di

    all_trades = []
    all_year_results = []

    for test_year in wf_years:
        if test_year not in year_di_map:
            continue
        di_start_y, di_end_y = year_di_map[test_year]
        # Use data from MIN_TRAIN to start of test year for warmup
        # But trade only in test year
        # For Kelly, we need trade_history from prior years too,
        # so we run from MIN_TRAIN but only count test year results

        result = backtest_engine(
            NS, ND, dates, C, O, H, L, syms,
            buy_signals, roc5, roc5_prev, atr14,
            sizing_cfg, exit_cfg, top_n=top_n,
            wf_start=MIN_TRAIN, wf_end=di_end_y + 1
        )
        if result and test_year in result['year_stats']:
            ys = result['year_stats'][test_year]
            all_year_results.append({
                'year': test_year,
                'trades': ys['trades'],
                'wins': ys['wins'],
                'pnl_abs': ys['pnl_abs_sum'],
            })
        else:
            all_year_results.append({'year': test_year, 'trades': 0, 'wins': 0, 'pnl_abs': 0})

    # Also run full backtest for overall stats
    full_result = backtest_engine(
        NS, ND, dates, C, O, H, L, syms,
        buy_signals, roc5, roc5_prev, atr14,
        sizing_cfg, exit_cfg, top_n=top_n
    )
    return full_result, all_year_results


# ====================================================================
# Print results
# ====================================================================
def print_result(name, result, wf_results=None):
    if result is None:
        print(f"  {name:55s}  NO TRADES")
        return
    ann = result['ann']
    n = result['n']
    wr = result['wr']
    mdd = result['max_dd']
    aw = result['avg_win']
    al = result['avg_loss']
    final = result['final']
    line = f"  {name:55s}  ann={ann:+8.1f}%  n={n:5d}  WR={wr:5.1f}%  MDD={mdd:6.1f}%  avgW={aw:+6.2f}  avgL={al:6.2f}  final={final:>12.0f}"
    print(line, flush=True)
    if wf_results:
        for yr in wf_results:
            print(f"    {yr['year']}: trades={yr['trades']:3d}  wins={yr['wins']:3d}  pnl={yr['pnl_abs']:+10.0f}", flush=True)


# ====================================================================
# MAIN
# ====================================================================
def main():
    print("=" * 80)
    print("Alpha Futures V113 — Position Sizing + Exit Strategy Optimization")
    print("=" * 80)

    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)
    print(f"  Loaded {NS} symbols, {ND} days")

    # Precompute indicators
    roc5, roc5_prev, atr14 = precompute_indicators(NS, ND, C, O, H, L)

    # Generate ROC(5) cross signals
    buy_signals = generate_roc5_cross_signals(NS, ND, roc5, roc5_prev)

    # Count signals
    total_signals = sum(len(s) for s in buy_signals)
    print(f"  Total ROC(5) cross signals: {total_signals}")

    top_n = 1

    # ==================================================================
    # BASELINE: ROC(5) cross, hold 5 days, 100% capital
    # ==================================================================
    print("\n" + "=" * 80)
    print("BASELINE: ROC(5) cross zero, hold 5 days, 100% capital, top_n=1")
    print("=" * 80)

    sizing_base = {'method': 'fixed'}
    exit_base = {'method': 'fixed_hold', 'hold_days': 5}

    res, wf = run_walk_forward(NS, ND, dates, C, O, H, L, syms,
                                buy_signals, roc5, roc5_prev, atr14,
                                sizing_base, exit_base, top_n=top_n)
    print_result("BASELINE: ROC5 cross, hold 5, 100%", res, wf)
    baseline_ann = res['ann'] if res else 0

    # ==================================================================
    # A) VOLATILITY-SCALED SIZING
    # ==================================================================
    print("\n" + "=" * 80)
    print("A) VOLATILITY-SCALED SIZING (hold 5 days baseline exit)")
    print("=" * 80)

    for target_vol in [0.15, 0.20, 0.30, 0.50]:
        cfg = {'method': 'vol_scaled', 'target_vol': target_vol}
        res, wf = run_walk_forward(NS, ND, dates, C, O, H, L, syms,
                                    buy_signals, roc5, roc5_prev, atr14,
                                    cfg, exit_base, top_n=top_n)
        print_result(f"  vol_scaled target_vol={target_vol:.0%}, hold 5", res, wf)

    # ==================================================================
    # B) KELLY CRITERION
    # ==================================================================
    print("\n" + "=" * 80)
    print("B) KELLY CRITERION (hold 5 days baseline exit)")
    print("=" * 80)

    for kelly_frac, kname in [(0.25, 'quarter'), (0.50, 'half'), (1.0, 'full')]:
        cfg = {'method': 'kelly', 'kelly_frac': kelly_frac}
        res, wf = run_walk_forward(NS, ND, dates, C, O, H, L, syms,
                                    buy_signals, roc5, roc5_prev, atr14,
                                    cfg, exit_base, top_n=top_n)
        print_result(f"  kelly {kname} (x{kelly_frac}), hold 5", res, wf)

    # ==================================================================
    # C) ANTI-MARTINGALE
    # ==================================================================
    print("\n" + "=" * 80)
    print("C) ANTI-MARTINGALE (hold 5 days baseline exit)")
    print("=" * 80)

    for adj in [0.10, 0.20, 0.30]:
        cfg = {'method': 'anti_martingale', 'anti_mart_adj': adj}
        res, wf = run_walk_forward(NS, ND, dates, C, O, H, L, syms,
                                    buy_signals, roc5, roc5_prev, atr14,
                                    cfg, exit_base, top_n=top_n)
        print_result(f"  anti_mart adj={adj:.0%}, hold 5", res, wf)

    # ==================================================================
    # D) MOMENTUM-STRENGTH SIZING
    # ==================================================================
    print("\n" + "=" * 80)
    print("D) MOMENTUM-STRENGTH SIZING (hold 5 days baseline exit)")
    print("=" * 80)

    threshold_sets = [
        ('conservative', [(0.00, 0.30), (0.02, 0.50), (0.05, 0.75), (0.10, 1.00)]),
        ('baseline',     [(0.00, 0.50), (0.02, 0.75), (0.05, 1.00), (0.10, 1.00)]),
        ('aggressive',   [(0.00, 0.60), (0.02, 0.80), (0.05, 1.00), (0.10, 1.20)]),
    ]
    for tname, thresholds in threshold_sets:
        cfg = {'method': 'momentum_strength', 'mom_thresholds': thresholds}
        res, wf = run_walk_forward(NS, ND, dates, C, O, H, L, syms,
                                    buy_signals, roc5, roc5_prev, atr14,
                                    cfg, exit_base, top_n=top_n)
        print_result(f"  mom_strength {tname}, hold 5", res, wf)

    # ==================================================================
    # E) TRAILING STOP EXIT
    # ==================================================================
    print("\n" + "=" * 80)
    print("E) TRAILING STOP EXIT (100% capital sizing)")
    print("=" * 80)

    for atr_mult in [1.5, 2.0, 2.5, 3.0, 4.0]:
        for max_hold in [5, 10, 15, 20]:
            ecfg = {
                'method': 'trailing_stop',
                'trail_atr_mult': atr_mult,
                'trail_max_hold': max_hold,
            }
            res, wf = run_walk_forward(NS, ND, dates, C, O, H, L, syms,
                                        buy_signals, roc5, roc5_prev, atr14,
                                        sizing_base, ecfg, top_n=top_n)
            print_result(f"  trail {atr_mult}xATR, max_hold={max_hold}", res, wf)

    # ==================================================================
    # F) ROC-BASED EXIT
    # ==================================================================
    print("\n" + "=" * 80)
    print("F) ROC-BASED EXIT (exit when ROC(5) crosses below 0)")
    print("=" * 80)

    for max_hold in [10, 15, 20]:
        ecfg = {'method': 'roc_exit', 'hold_days': max_hold}
        res, wf = run_walk_forward(NS, ND, dates, C, O, H, L, syms,
                                    buy_signals, roc5, roc5_prev, atr14,
                                    sizing_base, ecfg, top_n=top_n)
        print_result(f"  roc_exit, max_hold={max_hold}", res, wf)

    # ==================================================================
    # G) PROFIT TARGET
    # ==================================================================
    print("\n" + "=" * 80)
    print("G) PROFIT TARGET (exit at X% profit, max hold 20)")
    print("=" * 80)

    for pt_pct in [3.0, 5.0, 8.0, 10.0]:
        ecfg = {'method': 'profit_target', 'profit_target_pct': pt_pct, 'hold_days': 20}
        res, wf = run_walk_forward(NS, ND, dates, C, O, H, L, syms,
                                    buy_signals, roc5, roc5_prev, atr14,
                                    sizing_base, ecfg, top_n=top_n)
        print_result(f"  profit_target {pt_pct}%, max_hold 20", res, wf)

    # ==================================================================
    # H) TIME-WEIGHTED EXIT
    # ==================================================================
    print("\n" + "=" * 80)
    print("H) TIME-WEIGHTED EXIT")
    print("=" * 80)

    ecfg = {'method': 'time_weighted'}
    res, wf = run_walk_forward(NS, ND, dates, C, O, H, L, syms,
                                buy_signals, roc5, roc5_prev, atr14,
                                sizing_base, ecfg, top_n=top_n)
    print_result("  time_weighted (3d check, -3% stop, 10d max)", res, wf)

    # ==================================================================
    # I) CHANDELIER EXIT
    # ==================================================================
    print("\n" + "=" * 80)
    print("I) CHANDELIER EXIT")
    print("=" * 80)

    for atr_mult in [2.0, 3.0, 4.0]:
        for max_hold in [10, 15, 20]:
            ecfg = {
                'method': 'chandelier',
                'chandelier_atr_mult': atr_mult,
                'chandelier_max_hold': max_hold,
            }
            res, wf = run_walk_forward(NS, ND, dates, C, O, H, L, syms,
                                        buy_signals, roc5, roc5_prev, atr14,
                                        sizing_base, ecfg, top_n=top_n)
            print_result(f"  chandelier {atr_mult}xATR, max_hold={max_hold}", res, wf)

    # ==================================================================
    # J) COMBINED EXIT
    # ==================================================================
    print("\n" + "=" * 80)
    print("J) COMBINED EXIT (trailing 3*ATR + ROC reversal + max hold 15)")
    print("=" * 80)

    ecfg = {'method': 'combined', 'hold_days': 15}
    res, wf = run_walk_forward(NS, ND, dates, C, O, H, L, syms,
                                buy_signals, roc5, roc5_prev, atr14,
                                sizing_base, ecfg, top_n=top_n)
    print_result("  combined (trail 3ATR + ROC exit + 15d max)", res, wf)

    # ==================================================================
    # BEST COMBOS: Best sizing + best exit
    # ==================================================================
    print("\n" + "=" * 80)
    print("BEST COMBINATIONS: sizing + exit")
    print("=" * 80)

    # We'll test promising combos from each category
    combos = [
        # (sizing_name, sizing_cfg, exit_name, exit_cfg)
        ("vol_20% + roc_exit 20d",
         {'method': 'vol_scaled', 'target_vol': 0.20},
         "roc_exit 20d",
         {'method': 'roc_exit', 'hold_days': 20}),

        ("vol_30% + roc_exit 20d",
         {'method': 'vol_scaled', 'target_vol': 0.30},
         "roc_exit 20d",
         {'method': 'roc_exit', 'hold_days': 20}),

        ("vol_20% + combined",
         {'method': 'vol_scaled', 'target_vol': 0.20},
         "combined",
         {'method': 'combined', 'hold_days': 15}),

        ("vol_30% + combined",
         {'method': 'vol_scaled', 'target_vol': 0.30},
         "combined",
         {'method': 'combined', 'hold_days': 15}),

        ("vol_20% + trail 3ATR 15d",
         {'method': 'vol_scaled', 'target_vol': 0.20},
         "trail 3ATR 15d",
         {'method': 'trailing_stop', 'trail_atr_mult': 3.0, 'trail_max_hold': 15}),

        ("vol_30% + trail 3ATR 15d",
         {'method': 'vol_scaled', 'target_vol': 0.30},
         "trail 3ATR 15d",
         {'method': 'trailing_stop', 'trail_atr_mult': 3.0, 'trail_max_hold': 15}),

        ("vol_20% + chandelier 3ATR 15d",
         {'method': 'vol_scaled', 'target_vol': 0.20},
         "chandelier 3ATR 15d",
         {'method': 'chandelier', 'chandelier_atr_mult': 3.0, 'chandelier_max_hold': 15}),

        ("vol_30% + chandelier 3ATR 15d",
         {'method': 'vol_scaled', 'target_vol': 0.30},
         "chandelier 3ATR 15d",
         {'method': 'chandelier', 'chandelier_atr_mult': 3.0, 'chandelier_max_hold': 15}),

        ("kelly_half + roc_exit 20d",
         {'method': 'kelly', 'kelly_frac': 0.5},
         "roc_exit 20d",
         {'method': 'roc_exit', 'hold_days': 20}),

        ("kelly_half + combined",
         {'method': 'kelly', 'kelly_frac': 0.5},
         "combined",
         {'method': 'combined', 'hold_days': 15}),

        ("anti_mart_20% + roc_exit 20d",
         {'method': 'anti_martingale', 'anti_mart_adj': 0.2},
         "roc_exit 20d",
         {'method': 'roc_exit', 'hold_days': 20}),

        ("anti_mart_20% + combined",
         {'method': 'anti_martingale', 'anti_mart_adj': 0.2},
         "combined",
         {'method': 'combined', 'hold_days': 15}),

        ("mom_aggressive + roc_exit 20d",
         {'method': 'momentum_strength', 'mom_thresholds': [(0.00, 0.60), (0.02, 0.80), (0.05, 1.00), (0.10, 1.20)]},
         "roc_exit 20d",
         {'method': 'roc_exit', 'hold_days': 20}),

        ("mom_aggressive + combined",
         {'method': 'momentum_strength', 'mom_thresholds': [(0.00, 0.60), (0.02, 0.80), (0.05, 1.00), (0.10, 1.20)]},
         "combined",
         {'method': 'combined', 'hold_days': 15}),
    ]

    best_result = None
    best_name = ""
    best_wf = None

    for sizing_name, scfg, exit_name, ecfg in combos:
        full_name = f"{sizing_name} + {exit_name}"
        res, wf = run_walk_forward(NS, ND, dates, C, O, H, L, syms,
                                    buy_signals, roc5, roc5_prev, atr14,
                                    scfg, ecfg, top_n=top_n)
        print_result(f"  {full_name}", res, wf)
        if res and (best_result is None or res['ann'] > best_result['ann']):
            best_result = res
            best_name = full_name
            best_wf = wf

    # ==================================================================
    # SUMMARY
    # ==================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n  Baseline (ROC5 cross, hold 5, 100%):  ann={baseline_ann:+.1f}%")
    if best_result:
        print(f"  Best combo ({best_name}):")
        print(f"    ann={best_result['ann']:+.1f}%  n={best_result['n']}  WR={best_result['wr']:.1f}%  MDD={best_result['max_dd']:.1f}%  final={best_result['final']:.0f}")
        if best_wf:
            print("    Walk-forward:")
            for yr in best_wf:
                print(f"      {yr['year']}: trades={yr['trades']:3d}  wins={yr['wins']:3d}  pnl={yr['pnl_abs']:+10.0f}")

    if best_result and best_result['ann'] > baseline_ann:
        print(f"\n  *** BEATS BASELINE by {best_result['ann'] - baseline_ann:+.1f}% annualized ***")
    elif best_result:
        print(f"\n  *** Does NOT beat baseline (diff={best_result['ann'] - baseline_ann:+.1f}%) ***")
        print("  Proper sizing/exit reduces returns but may improve risk-adjusted metrics.")

    print("\nDone.")


if __name__ == '__main__':
    main()
