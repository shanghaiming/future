"""
Alpha Futures V120 — DETAILED PER-COMMODITY ANALYSIS + WALK-FORWARD STABILITY
==============================================================================
V120 FOCUS: Deep dive into the V116 champion signal:
  ROC(5) > 2% AND Z-score > 1.5, hold 3 days, top_n = 1

Tests:
  A) Full commodity profitability ranking (all 68 individually)
  B) Correlation between top commodities for diversification
  C) Sliding window walk-forward (252 train / 63 test, ~12 windows)
  D) Stability metrics (rolling 60-day, Calmar, Sortino, recovery)
  E) Parameter sensitivity surface (ROC x Z x hold)
  F) Transaction cost sensitivity
  G) 2020-2022 vs 2023-2025 alpha decay analysis

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


def main():
    print("=" * 220)
    print("  Alpha Futures V120 — DETAILED PER-COMMODITY ANALYSIS + WALK-FORWARD STABILITY")
    print("=" * 220)
    print("\n  Champion signal: ROC(5) > 2% AND Z-score > 1.5, hold 3 days, top_n = 1")
    print("  ALL signals at close di, entry at O[si, di+1] (NEXT DAY OPEN)")

    # -- Load data -------------------------------------------------
    print("\n[Data] Loading...", flush=True)
    t_start = time.time()
    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)
    years = sorted(set(d.year for d in dates))
    print(f"  {NS} commodities, {ND} days, years: {years}")

    # ================================================================
    # PRECOMPUTE INDICATORS
    # ================================================================
    print("\n[Precompute] ROC(5), Z-score...", flush=True)
    t0 = time.time()

    # -- Daily returns --
    RET = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si].astype(np.float64)
        for di in range(1, ND):
            if not np.isnan(c[di]) and not np.isnan(c[di-1]) and c[di-1] > 0:
                RET[si, di] = (c[di] / c[di-1] - 1) * 100

    # -- ROC(5) --
    ROC5 = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si].astype(np.float64)
        ROC5[si] = talib.ROC(c, timeperiod=5)

    # -- Z-score of daily returns (20-day rolling) --
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

    print(f"  All indicators computed ({time.time()-t0:.1f}s)")

    # ================================================================
    # CHAMPION SIGNAL
    # ================================================================
    sig_champ = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(25, ND):
            roc = ROC5[si, di]
            zs = ZSCORE[si, di]
            if np.isnan(roc) or np.isnan(zs):
                continue
            if roc > 2.0 and zs > 1.5:
                sig_champ[si, di] = True
    print(f"  Champion signal count: {np.sum(sig_champ)}")

    # ================================================================
    # GENERIC SINGLE-COMMODITY BACKTEST
    # ================================================================
    def backtest_single_commodity(si, sig_arr, hold_days, comm_rate=COMM,
                                  start_di=MIN_TRAIN, end_di=None,
                                  return_trades=False):
        """Backtest a single commodity. Returns annual return or (ann, trades)."""
        if end_di is None:
            end_di = ND

        cash = float(CASH0)
        positions = []
        trades = []
        daily_equity = np.full(end_di - start_di, float(CASH0))

        for di in range(start_di, end_di - 1):
            eq_idx = di - start_di

            # Mark equity before any action
            port_val = cash
            for pos in positions:
                cp = C[pos['si'], di]
                if not np.isnan(cp) and cp > 0:
                    mult = MULT.get(pos['sym'], DEF_MULT)
                    port_val += cp * mult * pos['lots'] - cp * mult * abs(pos['lots']) * comm_rate
            if eq_idx < len(daily_equity):
                daily_equity[eq_idx] = port_val

            # Close positions
            closed = []
            for pos in positions:
                days_held = di - pos['entry_di']
                if days_held >= pos['hold_days']:
                    exit_price = C[pos['si'], di]
                    if np.isnan(exit_price) or exit_price <= 0:
                        exit_price = pos['entry_price']
                    mult = MULT.get(pos['sym'], DEF_MULT)
                    mkt_val = exit_price * mult * abs(pos['lots'])
                    cash += mkt_val - mkt_val * comm_rate
                    pnl = (exit_price - pos['entry_price']) * mult * pos['lots']
                    invested = pos['entry_price'] * mult * abs(pos['lots'])
                    pnl_pct = pnl / invested * 100 if invested > 0 else 0
                    trades.append({
                        'pnl_pct': pnl_pct,
                        'entry_di': pos['entry_di'],
                        'exit_di': di,
                        'sym': pos.get('sym', ''),
                        'days_held': days_held,
                    })
                    closed.append(pos)
            for pos in closed:
                positions.remove(pos)

            if len(positions) >= 1:
                continue

            # Entry signal
            entry_di = di + 1
            if entry_di >= end_di:
                continue
            if not sig_arr[si, di]:
                continue
            ep = O[si, entry_di]
            if np.isnan(ep) or ep <= 0:
                continue

            sym = syms[si]
            mult = MULT.get(sym, DEF_MULT)
            contracts = max(1, int(cash * 0.95 / (ep * mult * (1 + comm_rate))))
            cost_in = ep * mult * contracts * (1 + comm_rate)
            if contracts <= 0 or cost_in > cash:
                continue
            cash -= cost_in
            positions.append({
                'si': si, 'entry_price': ep, 'entry_di': entry_di,
                'lots': contracts, 'dir': 1, 'sym': sym,
                'hold_days': hold_days,
            })

        # Close remaining
        for pos in positions:
            ae = end_di - 1
            exit_price = C[pos['si'], min(ae, ND-1)]
            if np.isnan(exit_price) or exit_price <= 0:
                exit_price = pos['entry_price']
            mult = MULT.get(pos['sym'], DEF_MULT)
            mkt_val = exit_price * mult * abs(pos['lots'])
            cash += mkt_val - mkt_val * comm_rate
            pnl = (exit_price - pos['entry_price']) * mult * pos['lots']
            invested = pos['entry_price'] * mult * abs(pos['lots'])
            pnl_pct = pnl / invested * 100 if invested > 0 else 0
            trades.append({
                'pnl_pct': pnl_pct,
                'entry_di': pos['entry_di'],
                'exit_di': ae,
                'sym': pos.get('sym', ''),
                'days_held': ae - pos['entry_di'],
            })

        n_days_test = end_di - start_di
        ann = annual_return(cash, CASH0, n_days_test)

        if return_trades:
            return ann, trades, daily_equity
        return ann

    # ================================================================
    # GENERIC MULTI-COMMODITY BACKTEST (for champion config)
    # ================================================================
    def backtest_multi(sig_arr, hold_days, top_n, comm_rate=COMM,
                       start_di=MIN_TRAIN, end_di=None,
                       return_trades=False):
        """Backtest with top_n positions across all commodities."""
        if end_di is None:
            end_di = ND

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
                    port_val += cp * mult * pos['lots'] - cp * mult * abs(pos['lots']) * comm_rate
            daily_equity.append(port_val)

            # Close positions
            closed = []
            for pos in positions:
                days_held = di - pos['entry_di']
                if days_held >= pos['hold_days']:
                    exit_price = C[pos['si'], di]
                    if np.isnan(exit_price) or exit_price <= 0:
                        exit_price = pos['entry_price']
                    mult = MULT.get(pos['sym'], DEF_MULT)
                    mkt_val = exit_price * mult * abs(pos['lots'])
                    cash += mkt_val - mkt_val * comm_rate
                    pnl = (exit_price - pos['entry_price']) * mult * pos['lots']
                    invested = pos['entry_price'] * mult * abs(pos['lots'])
                    pnl_pct = pnl / invested * 100 if invested > 0 else 0
                    trades.append({
                        'pnl_pct': pnl_pct,
                        'entry_di': pos['entry_di'],
                        'exit_di': di,
                        'sym': pos.get('sym', ''),
                        'days_held': days_held,
                    })
                    closed.append(pos)
            for pos in closed:
                positions.remove(pos)

            if len(positions) >= top_n:
                continue

            # Generate entry signals
            entry_di = di + 1
            if entry_di >= end_di:
                continue

            candidates = []
            for s in range(NS):
                if not sig_arr[s, di]:
                    continue
                if any(p['si'] == s for p in positions):
                    continue
                ep = O[s, entry_di]
                if np.isnan(ep) or ep <= 0:
                    continue
                sc = ROC5[s, di] if not np.isnan(ROC5[s, di]) else 0
                candidates.append((sc, s, ep))

            if not candidates:
                continue

            candidates.sort(key=lambda x: -x[0])
            n_slots = top_n - len(positions)
            cap_per_slot = cash / max(1, n_slots)

            for sc_val, s, price in candidates[:max(0, n_slots)]:
                sym = syms[s]
                mult = MULT.get(sym, DEF_MULT)
                contracts = max(1, int(cap_per_slot * 0.95 / (price * mult * (1 + comm_rate))))
                cost_in = price * mult * contracts * (1 + comm_rate)
                if cost_in > cash:
                    contracts = int(cash * 0.9 / (price * mult * (1 + comm_rate)))
                    cost_in = price * mult * contracts * (1 + comm_rate) if contracts > 0 else 0
                if contracts <= 0 or cost_in <= 0 or cost_in > cash:
                    continue
                cash -= cost_in
                positions.append({
                    'si': s, 'entry_price': price, 'entry_di': entry_di,
                    'lots': contracts, 'dir': 1, 'sym': sym,
                    'hold_days': hold_days,
                })

        # Close remaining
        for pos in positions:
            ae = end_di - 1
            exit_price = C[pos['si'], min(ae, ND-1)]
            if np.isnan(exit_price) or exit_price <= 0:
                exit_price = pos['entry_price']
            mult = MULT.get(pos['sym'], DEF_MULT)
            mkt_val = exit_price * mult * abs(pos['lots'])
            cash += mkt_val - mkt_val * comm_rate
            pnl = (exit_price - pos['entry_price']) * mult * pos['lots']
            invested = pos['entry_price'] * mult * abs(pos['lots'])
            pnl_pct = pnl / invested * 100 if invested > 0 else 0
            trades.append({
                'pnl_pct': pnl_pct,
                'entry_di': pos['entry_di'],
                'exit_di': ae,
                'sym': pos.get('sym', ''),
                'days_held': ae - pos['entry_di'],
            })

        n_days_test = end_di - start_di
        ann = annual_return(cash, CASH0, n_days_test)
        wr = np.mean([1 if t['pnl_pct'] > 0 else 0 for t in trades]) * 100 if trades else 0
        n_trades = len(trades)
        avg_pnl = np.mean([t['pnl_pct'] for t in trades]) if trades else 0

        # Max drawdown
        eq = float(CASH0)
        peak = eq
        mdd = 0.0
        for t in sorted(trades, key=lambda x: x['entry_di']):
            eq *= (1 + t['pnl_pct'] / 100)
            if eq > peak:
                peak = eq
            dd = (eq - peak) / peak * 100
            if dd < mdd:
                mdd = dd

        result = {
            'ann': ann, 'wr': wr, 'n': n_trades, 'avg_pnl': avg_pnl,
            'final_cash': cash, 'n_days': n_days_test, 'mdd': mdd,
            'trades': trades, 'daily_equity': daily_equity,
        }
        return result

    # ================================================================
    # A) FULL COMMODITY PROFITABILITY RANKING
    # ================================================================
    print(f"\n{'=' * 220}")
    print("  A) FULL COMMODITY PROFITABILITY RANKING")
    print("     Running ROC(5)>2% AND Z>1.5 strategy on EACH commodity independently")
    print(f"{'=' * 220}")

    comm_results = []
    t_a = time.time()
    for si in range(NS):
        ann = backtest_single_commodity(si, sig_champ, hold_days=3)
        # Walk-forward by calendar year
        wf_pos = 0
        wf_vals = {}
        for yr in [2020, 2021, 2022, 2023, 2024, 2025]:
            ts = None
            te = None
            for di in range(ND):
                if dates[di].year == yr and ts is None:
                    ts = di
                if dates[di].year == yr + 1 and te is None:
                    te = di
            if ts is None:
                wf_vals[yr] = 0
                continue
            if te is None:
                te = ND
            yr_ann = backtest_single_commodity(si, sig_champ, hold_days=3,
                                               start_di=ts, end_di=te)
            wf_vals[yr] = yr_ann
            if yr_ann > 0:
                wf_pos += 1
        comm_results.append({
            'si': si, 'sym': syms[si], 'ann': ann,
            'wf_pos': wf_pos, 'wf_vals': wf_vals,
        })
        if (si + 1) % 10 == 0:
            print(f"    Commodity {si+1}/{NS} done ({time.time()-t_a:.1f}s)", flush=True)

    comm_results.sort(key=lambda x: -x['ann'])

    print(f"\n  {'Rank':>4} | {'Symbol':<8} | {'Ann Return':>12} | {'WF 2020':>8} | {'WF 2021':>8} | {'WF 2022':>8} | {'WF 2023':>8} | {'WF 2024':>8} | {'WF 2025':>8} | {'WF Pos':>6} | {'Status':<20}")
    print("-" * 220)
    for i, cr in enumerate(comm_results):
        wf_strs = {yr: f"{cr['wf_vals'].get(yr, 0):>+7.1f}%" for yr in [2020, 2021, 2022, 2023, 2024, 2025]}
        status = "6/6 CONSISTENT" if cr['wf_pos'] == 6 else \
                 ("5/6 STRONG" if cr['wf_pos'] == 5 else \
                 ("4/6 MODERATE" if cr['wf_pos'] == 4 else \
                 ("MARGINAL" if cr['wf_pos'] >= 3 else "UNRELIABLE")))
        print(f"  {i+1:>4} | {cr['sym']:<8} | {cr['ann']:>+10.1f}% | {wf_strs[2020]:>8} | {wf_strs[2021]:>8} | {wf_strs[2022]:>8} | {wf_strs[2023]:>8} | {wf_strs[2024]:>8} | {wf_strs[2025]:>8} | {cr['wf_pos']}/6    | {status:<20}")

    consistent = [cr for cr in comm_results if cr['wf_pos'] == 6]
    strong = [cr for cr in comm_results if cr['wf_pos'] >= 5]
    profitable = [cr for cr in comm_results if cr['ann'] > 0]
    print(f"\n  Summary: {len(profitable)}/{NS} profitable, {len(strong)} with 5+/6 WF positive, {len(consistent)} with 6/6 WF positive")
    print(f"  Consistent 6/6 commodities: {', '.join(cr['sym'] for cr in consistent)}")

    # ================================================================
    # B) CORRELATION BETWEEN TOP COMMODITIES
    # ================================================================
    print(f"\n{'=' * 220}")
    print("  B) CORRELATION BETWEEN TOP 20 COMMODITIES")
    print(f"{'=' * 220}")

    top20 = comm_results[:20]
    # Compute daily strategy returns for each top commodity
    top_daily_rets = {}
    for cr in top20:
        si = cr['si']
        daily_ret = np.zeros(ND)
        in_pos = False
        entry_di = -1
        entry_price = 0
        hold = 3

        for di in range(MIN_TRAIN, ND - 1):
            if in_pos:
                days_held = di - entry_di
                if days_held >= hold:
                    ep = C[si, di]
                    if np.isnan(ep) or ep <= 0:
                        ep = entry_price
                    ret_val = (ep / entry_price - 1) * 100
                    # Distribute return over hold period
                    for d2 in range(max(entry_di, MIN_TRAIN), di):
                        if d2 < ND:
                            daily_ret[d2] = ret_val / max(1, di - entry_di)
                    in_pos = False

            if not in_pos and sig_champ[si, di]:
                entry_di = di + 1
                if entry_di < ND:
                    ep = O[si, entry_di]
                    if not np.isnan(ep) and ep > 0:
                        entry_price = ep
                        in_pos = True

        top_daily_rets[cr['sym']] = daily_ret

    # Pairwise correlation matrix
    top20_syms = [cr['sym'] for cr in top20]
    n_top = len(top20_syms)
    corr_matrix = np.zeros((n_top, n_top))
    for i in range(n_top):
        for j in range(n_top):
            ri = top_daily_rets[top20_syms[i]]
            rj = top_daily_rets[top20_syms[j]]
            valid = (ri != 0) | (rj != 0)
            if np.sum(valid) > 20:
                corr_matrix[i, j] = np.corrcoef(ri[valid], rj[valid])[0, 1]
            else:
                corr_matrix[i, j] = 0

    # Print correlation matrix (compact)
    print(f"\n  Pairwise correlation of daily strategy returns (top 20):")
    header = f"  {'':>8}"
    for s in top20_syms[:10]:
        header += f" {s:>7}"
    print(header)
    for i in range(min(n_top, 10)):
        row = f"  {top20_syms[i]:>8}"
        for j in range(min(n_top, 10)):
            row += f" {corr_matrix[i, j]:>+6.2f}"
        print(row)

    # Find uncorrelated groups
    print(f"\n  Low-correlation pairs (|corr| < 0.3) among top 20:")
    low_corr_pairs = []
    for i in range(n_top):
        for j in range(i+1, n_top):
            if abs(corr_matrix[i, j]) < 0.3:
                low_corr_pairs.append((top20_syms[i], top20_syms[j], corr_matrix[i, j]))
    low_corr_pairs.sort(key=lambda x: abs(x[2]))
    for s1, s2, c in low_corr_pairs[:30]:
        ann1 = next(cr['ann'] for cr in top20 if cr['sym'] == s1)
        ann2 = next(cr['ann'] for cr in top20 if cr['sym'] == s2)
        print(f"    {s1} ({ann1:>+7.1f}%) & {s2} ({ann2:>+7.1f}%) | corr = {c:>+.3f}")

    # Greedy diversification portfolio (pick top uncorrelated commodities)
    print(f"\n  GREEDY DIVERSIFICATION PORTFOLIO (select top uncorrelated set):")
    selected = [top20_syms[0]]
    for _ in range(min(9, n_top - 1)):
        best_next = None
        best_score = -999
        for s in top20_syms:
            if s in selected:
                continue
            idx_s = top20_syms.index(s)
            # Score = min correlation with selected set (lower = better diversification)
            max_corr = max(abs(corr_matrix[idx_s, top20_syms.index(sel)]) for sel in selected)
            # Secondary: prefer higher annual return
            ann_s = next(cr['ann'] for cr in top20 if cr['sym'] == s)
            score = -max_corr * 1000 + ann_s
            if score > best_score:
                best_score = score
                best_next = s
        if best_next:
            selected.append(best_next)

    for i, s in enumerate(selected):
        ann_s = next(cr['ann'] for cr in top20 if cr['sym'] == s)
        wf_s = next(cr['wf_pos'] for cr in top20 if cr['sym'] == s)
        print(f"    {i+1}. {s:<8} Ann={ann_s:>+8.1f}%  WF={wf_s}/6")

    # ================================================================
    # C) SLIDING WINDOW WALK-FORWARD (252 train / 63 test)
    # ================================================================
    print(f"\n{'=' * 220}")
    print("  C) SLIDING WINDOW WALK-FORWARD (252 train / 63 test)")
    print(f"{'=' * 220}")

    train_len = 252
    test_len = 63
    slide = 63

    wf_windows = []
    # Build windows
    start = MIN_TRAIN + train_len
    while start + test_len < ND:
        train_start = start - train_len
        train_end = start
        test_start = start
        test_end = min(start + test_len, ND)
        # Ensure we have enough data
        if test_end - test_start < 30:
            break
        wf_windows.append({
            'train_start': train_start,
            'train_end': train_end,
            'test_start': test_start,
            'test_end': test_end,
            'test_start_date': dates[test_start],
            'test_end_date': dates[min(test_end - 1, ND - 1)],
        })
        start += slide

    print(f"  Total WF windows: {len(wf_windows)}")

    # For each window, train = compute signals on train data, test = run on test data
    # Since signal is threshold-based (no fitted params), just run on test window directly
    print(f"\n  {'#':>3} | {'Test Period':<25} | {'Ann':>9} | {'WR':>6} | {'N':>5} | {'MDD':>7} | {'AvgPnL':>8}")
    print("-" * 120)

    wf_results = []
    n_pos_wf = 0
    for wi, w in enumerate(wf_windows):
        r = backtest_multi(sig_champ, hold_days=3, top_n=1,
                           start_di=w['test_start'], end_di=w['test_end'])
        period_str = f"{w['test_start_date'].strftime('%Y-%m-%d')} to {w['test_end_date'].strftime('%Y-%m-%d')}"
        print(f"  {wi+1:>3} | {period_str:<25} | {r['ann']:>+8.1f}% | {r['wr']:>5.1f}% | {r['n']:>5} | {r['mdd']:>6.1f}% | {r['avg_pnl']:>+7.3f}%")
        wf_results.append(r)
        if r['ann'] > 0:
            n_pos_wf += 1

    avg_wf_ann = np.mean([r['ann'] for r in wf_results]) if wf_results else 0
    print(f"\n  Summary: {n_pos_wf}/{len(wf_windows)} windows positive | Avg OOS ann: {avg_wf_ann:>+8.1f}%")
    print(f"  Worst window: {min(r['ann'] for r in wf_results):>+8.1f}%")
    print(f"  Best window:  {max(r['ann'] for r in wf_results):>+8.1f}%")
    print(f"  Median window: {np.median([r['ann'] for r in wf_results]):>+8.1f}%")

    # ================================================================
    # D) STABILITY METRICS
    # ================================================================
    print(f"\n{'=' * 220}")
    print("  D) STABILITY METRICS (Champion: ROC>2% + Z>1.5, H3, TN1)")
    print(f"{'=' * 220}")

    champ_result = backtest_multi(sig_champ, hold_days=3, top_n=1, return_trades=True)
    trades = champ_result['trades']
    daily_eq = champ_result['daily_equity']

    # -- Rolling 60-day return --
    if len(daily_eq) > 60:
        rolling_60 = []
        for i in range(60, len(daily_eq), 60):
            ret_60 = (daily_eq[i-1] / daily_eq[i-60] - 1) * 100 if daily_eq[i-60] > 0 else 0
            rolling_60.append(ret_60)

        pos_60 = sum(1 for r in rolling_60 if r > 0)
        pct_pos_60 = pos_60 / len(rolling_60) * 100 if rolling_60 else 0

        # Longest losing streak
        max_lose_streak = 0
        current_streak = 0
        for r in rolling_60:
            if r <= 0:
                current_streak += 1
                max_lose_streak = max(max_lose_streak, current_streak)
            else:
                current_streak = 0

        print(f"\n  Rolling 60-day returns:")
        print(f"    Total 60-day windows: {len(rolling_60)}")
        print(f"    Positive windows: {pos_60}/{len(rolling_60)} ({pct_pos_60:.1f}%)")
        print(f"    Avg 60-day return: {np.mean(rolling_60):>+8.2f}%")
        print(f"    Median 60-day return: {np.median(rolling_60):>+8.2f}%")
        print(f"    Worst 60-day return: {min(rolling_60):>+8.2f}%")
        print(f"    Best 60-day return: {max(rolling_60):>+8.2f}%")
        print(f"    Longest losing streak (consecutive negative 60-day windows): {max_lose_streak}")
    else:
        rolling_60 = []
        pos_60 = 0
        pct_pos_60 = 0
        max_lose_streak = 0
        print(f"    Not enough data for rolling 60-day analysis")

    # -- Build equity curve from daily equity --
    eq_arr = np.array(daily_eq) if daily_eq else np.array([float(CASH0)])

    # -- Max Drawdown --
    peak = np.maximum.accumulate(eq_arr)
    dd_arr = (eq_arr - peak) / peak * 100
    mdd = np.min(dd_arr)

    # -- Recovery time from DD --
    # Find DD periods and recovery times
    in_dd = False
    dd_start = 0
    recovery_times = []
    for i in range(len(eq_arr)):
        if eq_arr[i] < peak[i] * 0.99:  # in drawdown (>1% below peak)
            if not in_dd:
                in_dd = True
                dd_start = i
        else:
            if in_dd:
                recovery_times.append(i - dd_start)
                in_dd = False
    avg_recovery = np.mean(recovery_times) if recovery_times else 0
    max_recovery = max(recovery_times) if recovery_times else 0

    # -- Calmar ratio --
    calmar = champ_result['ann'] / abs(mdd) if mdd != 0 else 999

    # -- Sortino ratio --
    daily_returns = np.diff(eq_arr) / eq_arr[:-1] if len(eq_arr) > 1 else np.array([0])
    downside = daily_returns[daily_returns < 0]
    downside_std = np.std(downside, ddof=1) if len(downside) > 1 else 0.001
    sortino = (np.mean(daily_returns) * 252) / (downside_std * np.sqrt(252)) if downside_std > 0 else 999

    # -- Sharpe ratio --
    sharpe = (np.mean(daily_returns) * 252) / (np.std(daily_returns, ddof=1) * np.sqrt(252)) if len(daily_returns) > 1 else 0

    print(f"\n  Risk Metrics:")
    print(f"    Annual return:    {champ_result['ann']:>+10.1f}%")
    print(f"    Win rate:         {champ_result['wr']:>10.1f}%")
    print(f"    Total trades:     {champ_result['n']:>10d}")
    print(f"    Max drawdown:     {mdd:>10.2f}%")
    print(f"    Calmar ratio:     {calmar:>10.2f}")
    print(f"    Sharpe ratio:     {sharpe:>10.2f}")
    print(f"    Sortino ratio:    {sortino:>10.2f}")
    print(f"    Avg recovery:     {avg_recovery:>10.1f} days")
    print(f"    Max recovery:     {max_recovery:>10.1f} days")
    print(f"    Avg trade PnL:    {champ_result['avg_pnl']:>+10.3f}%")

    # ================================================================
    # E) PARAMETER SENSITIVITY SURFACE
    # ================================================================
    print(f"\n{'=' * 220}")
    print("  E) PARAMETER SENSITIVITY (ROC threshold x Z threshold x hold period)")
    print(f"{'=' * 220}")

    roc_thresholds = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
    z_thresholds = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    hold_periods = [1, 2, 3, 4, 5, 7, 10, 15, 20]

    # Precompute signals for all ROC x Z combinations
    print(f"\n  Computing signals for {len(roc_thresholds)} x {len(z_thresholds)} threshold combos...", flush=True)
    sig_cache = {}
    for roc_t in roc_thresholds:
        for z_t in z_thresholds:
            sig = np.zeros((NS, ND), dtype=bool)
            for si in range(NS):
                for di in range(25, ND):
                    roc = ROC5[si, di]
                    zs = ZSCORE[si, di]
                    if np.isnan(roc) or np.isnan(zs):
                        continue
                    if roc > roc_t and zs > z_t:
                        sig[si, di] = True
            sig_cache[(roc_t, z_t)] = sig

    # Run all combinations
    print(f"  Running {len(roc_thresholds)} x {len(z_thresholds)} x {len(hold_periods)} = {len(roc_thresholds)*len(z_thresholds)*len(hold_periods)} combos...", flush=True)
    t_e = time.time()
    param_results = []
    count = 0
    for roc_t in roc_thresholds:
        for z_t in z_thresholds:
            sig = sig_cache[(roc_t, z_t)]
            for hd in hold_periods:
                count += 1
                if count % 50 == 0:
                    print(f"    Combo {count}/{len(roc_thresholds)*len(z_thresholds)*len(hold_periods)} ({time.time()-t_e:.0f}s)", flush=True)
                r = backtest_multi(sig, hold_days=hd, top_n=1)
                param_results.append({
                    'roc_t': roc_t, 'z_t': z_t, 'hold': hd,
                    'ann': r['ann'], 'wr': r['wr'], 'n': r['n'],
                    'mdd': r['mdd'], 'avg_pnl': r['avg_pnl'],
                })

    # Find global optimum
    valid_params = [p for p in param_results if p['n'] >= 3]
    valid_params.sort(key=lambda x: -x['ann'])

    print(f"\n  TOP 20 PARAMETER COMBINATIONS (by annual return, min 3 trades):")
    print(f"  {'#':>3} | {'ROC >':>7} | {'Z >':>5} | {'Hold':>5} | {'Ann':>9} | {'WR':>6} | {'N':>5} | {'MDD':>7} | {'AvgPnL':>8}")
    print("-" * 120)
    for i, p in enumerate(valid_params[:20]):
        print(f"  {i+1:>3} | {p['roc_t']:>+6.1f}% | {p['z_t']:>+4.1f} | {p['hold']:>5} | {p['ann']:>+8.1f}% | {p['wr']:>5.1f}% | {p['n']:>5} | {p['mdd']:>6.1f}% | {p['avg_pnl']:>+7.3f}%")

    # Best for each hold period
    print(f"\n  BEST ROC/Z THRESHOLD FOR EACH HOLD PERIOD:")
    print(f"  {'Hold':>5} | {'Best ROC':>8} | {'Best Z':>6} | {'Ann':>9} | {'WR':>6} | {'N':>5} | {'MDD':>7}")
    print("-" * 90)
    for hd in hold_periods:
        sub = [p for p in valid_params if p['hold'] == hd]
        if sub:
            best = sub[0]
            print(f"  {hd:>5} | {best['roc_t']:>+7.1f}% | {best['z_t']:>+5.1f} | {best['ann']:>+8.1f}% | {best['wr']:>5.1f}% | {best['n']:>5} | {best['mdd']:>6.1f}%")

    # Surface table: ROC x Z for hold=3
    print(f"\n  SURFACE (Annual Return %, hold=3 days):")
    header = f"  {'ROC \\ Z':>8}"
    for z_t in z_thresholds:
        header += f" | {z_t:>6.1f}"
    print(header)
    print("-" * 80)
    for roc_t in roc_thresholds:
        row = f"  {roc_t:>7.1f}%"
        for z_t in z_thresholds:
            match = [p for p in param_results if p['roc_t'] == roc_t and p['z_t'] == z_t and p['hold'] == 3]
            if match:
                row += f" | {match[0]['ann']:>+6.0f}"
            else:
                row += f" |    N/A"
        print(row)

    # ================================================================
    # F) TRANSACTION COST SENSITIVITY
    # ================================================================
    print(f"\n{'=' * 220}")
    print("  F) TRANSACTION COST SENSITIVITY")
    print(f"{'=' * 220}")

    comm_rates = [0, 0.0001, 0.0003, 0.0005, 0.001, 0.0015, 0.002]
    print(f"\n  Testing champion (ROC>2% + Z>1.5, H3, TN1) at various commission rates:")
    print(f"  {'Commission':>12} | {'Ann Return':>12} | {'WR':>6} | {'N':>5} | {'MDD':>7} | {'AvgPnL':>8} | {'Status':<20}")
    print("-" * 120)

    breakeven_found = False
    for cr in comm_rates:
        r = backtest_multi(sig_champ, hold_days=3, top_n=1, comm_rate=cr)
        status = ""
        if r['ann'] <= 0 and not breakeven_found:
            status = "<-- BREAKEVEN POINT"
            breakeven_found = True
        elif r['ann'] > 0 and breakeven_found:
            status = "(still negative)"
        print(f"  {cr*100:>10.3f}% | {r['ann']:>+10.1f}% | {r['wr']:>5.1f}% | {r['n']:>5} | {r['mdd']:>6.1f}% | {r['avg_pnl']:>+7.3f}% | {status}")

    # Fine-grained breakeven search
    print(f"\n  Fine-grained breakeven search:")
    lo, hi = 0.0003, 0.001
    for _ in range(10):
        mid = (lo + hi) / 2
        r = backtest_multi(sig_champ, hold_days=3, top_n=1, comm_rate=mid)
        if r['ann'] > 0:
            lo = mid
        else:
            hi = mid
    print(f"  Approximate breakeven commission: {(lo+hi)/2*100:.4f}% ({(lo+hi)/2*10000:.1f} bps)")

    # ================================================================
    # G) 2020-2022 vs 2023-2025 SPLIT (Alpha Decay)
    # ================================================================
    print(f"\n{'=' * 220}")
    print("  G) 2020-2022 vs 2023-2025 ALPHA DECAY ANALYSIS")
    print(f"{'=' * 220}")

    # Find date boundaries
    di_2020 = None
    di_2023 = None
    di_2026 = None
    for di in range(ND):
        if dates[di].year == 2020 and di_2020 is None:
            di_2020 = di
        if dates[di].year == 2023 and di_2023 is None:
            di_2023 = di
        if dates[di].year == 2026 and di_2026 is None:
            di_2026 = di
    if di_2026 is None:
        di_2026 = ND

    # Period 1: 2020-2022
    r1 = backtest_multi(sig_champ, hold_days=3, top_n=1,
                        start_di=di_2020, end_di=di_2023)
    # Period 2: 2023-2025
    r2 = backtest_multi(sig_champ, hold_days=3, top_n=1,
                        start_di=di_2023, end_di=di_2026)

    # Per-year breakdown
    print(f"\n  Per-year breakdown:")
    print(f"  {'Year':>6} | {'Ann':>9} | {'WR':>6} | {'N':>5} | {'MDD':>7} | {'AvgPnL':>8}")
    print("-" * 80)
    for yr in range(2020, 2026):
        yr_start = None
        yr_end = None
        for di in range(ND):
            if dates[di].year == yr and yr_start is None:
                yr_start = di
            if dates[di].year == yr + 1 and yr_end is None:
                yr_end = di
        if yr_start is None:
            continue
        if yr_end is None:
            yr_end = ND
        yr_r = backtest_multi(sig_champ, hold_days=3, top_n=1,
                              start_di=yr_start, end_di=yr_end)
        print(f"  {yr:>6} | {yr_r['ann']:>+8.1f}% | {yr_r['wr']:>5.1f}% | {yr_r['n']:>5} | {yr_r['mdd']:>6.1f}% | {yr_r['avg_pnl']:>+7.3f}%")

    print(f"\n  Period comparison:")
    print(f"  {'Period':<15} | {'Ann':>9} | {'WR':>6} | {'N':>5} | {'MDD':>7} | {'AvgPnL':>8}")
    print("-" * 80)
    print(f"  {'2020-2022':<15} | {r1['ann']:>+8.1f}% | {r1['wr']:>5.1f}% | {r1['n']:>5} | {r1['mdd']:>6.1f}% | {r1['avg_pnl']:>+7.3f}%")
    print(f"  {'2023-2025':<15} | {r2['ann']:>+8.1f}% | {r2['wr']:>5.1f}% | {r2['n']:>5} | {r2['mdd']:>6.1f}% | {r2['avg_pnl']:>+7.3f}%")

    decay = r1['ann'] - r2['ann']
    if decay > 20:
        verdict_decay = "SIGNIFICANT ALPHA DECAY - first half much stronger"
    elif decay > 0:
        verdict_decay = "MILD DECAY - strategy slightly weaker recently but still viable"
    elif r2['ann'] > r1['ann']:
        verdict_decay = "NO DECAY - strategy improving over time"
    else:
        verdict_decay = "STABLE - consistent across periods"
    print(f"\n  Decay: {decay:>+.1f}% points | Verdict: {verdict_decay}")

    # Per-commodity alpha decay
    print(f"\n  Per-commodity alpha decay (2020-2022 vs 2023-2025):")
    print(f"  {'Symbol':<8} | {'2020-2022':>10} | {'2023-2025':>10} | {'Delta':>8} | {'Status':<20}")
    print("-" * 100)
    decay_data = []
    for cr in comm_results[:20]:
        si = cr['si']
        ann1 = backtest_single_commodity(si, sig_champ, hold_days=3,
                                          start_di=di_2020, end_di=di_2023)
        ann2 = backtest_single_commodity(si, sig_champ, hold_days=3,
                                          start_di=di_2023, end_di=di_2026)
        delta = ann2 - ann1
        status = "IMPROVING" if delta > 0 else ("STABLE" if delta > -20 else "DECAYING")
        decay_data.append({'sym': cr['sym'], 'ann1': ann1, 'ann2': ann2, 'delta': delta, 'status': status})
        print(f"  {cr['sym']:<8} | {ann1:>+9.1f}% | {ann2:>+9.1f}% | {delta:>+7.1f}% | {status:<20}")

    n_improving = sum(1 for d in decay_data if d['status'] == 'IMPROVING')
    n_stable = sum(1 for d in decay_data if d['status'] == 'STABLE')
    n_decaying = sum(1 for d in decay_data if d['status'] == 'DECAYING')
    print(f"\n  Among top 20: {n_improving} improving, {n_stable} stable, {n_decaying} decaying")

    # ================================================================
    # FINAL SUMMARY
    # ================================================================
    print(f"\n{'=' * 220}")
    print("  V120 FINAL SUMMARY")
    print(f"{'=' * 220}")

    print(f"\n  1. COMMODITY PROFITABILITY:")
    print(f"     {len(profitable)}/{NS} commodities profitable standalone")
    print(f"     {len(consistent)} commodities 6/6 WF consistent: {', '.join(cr['sym'] for cr in consistent)}")
    print(f"     Top 5: {', '.join(f'{cr['sym']}({cr['ann']:>+,.0f}%)' for cr in comm_results[:5])}")

    print(f"\n  2. WALK-FORWARD STABILITY:")
    print(f"     Calendar WF: see commodity table above")
    print(f"     Sliding WF: {n_pos_wf}/{len(wf_windows)} windows positive, avg {avg_wf_ann:>+,.1f}%")

    print(f"\n  3. PARAMETER OPTIMUM:")
    if valid_params:
        opt = valid_params[0]
        print(f"     Global optimum: ROC>{opt['roc_t']:.1f}% + Z>{opt['z_t']:.1f}, hold={opt['hold']}d")
        print(f"     Annual: {opt['ann']:>+,.1f}% | WR: {opt['wr']:.1f}% | N: {opt['n']} | MDD: {opt['mdd']:.1f}%")

    print(f"\n  4. TRANSACTION COST:")
    print(f"     Strategy profitable up to ~{(lo+hi)/2*100:.3f}% commission")
    print(f"     Current default (0.03%): {'PROFITABLE' if champ_result['ann'] > 0 else 'UNPROFITABLE'}")

    print(f"\n  5. ALPHA DECAY:")
    print(f"     2020-2022: {r1['ann']:>+,.1f}% | 2023-2025: {r2['ann']:>+,.1f}% | {verdict_decay}")

    print(f"\n  6. RISK METRICS:")
    print(f"     Calmar: {calmar:.2f} | Sharpe: {sharpe:.2f} | Sortino: {sortino:.2f}")
    print(f"     Max DD: {mdd:.2f}% | Avg recovery: {avg_recovery:.0f}d | Max recovery: {max_recovery:.0f}d")
    if rolling_60:
        print(f"     60d windows positive: {pos_60}/{len(rolling_60)} ({pct_pos_60:.1f}%)")
        print(f"     Longest losing streak (60d): {max_lose_streak}")

    print(f"\n  TOTAL TIME: {time.time()-t_start:.0f}s")


if __name__ == '__main__':
    main()
