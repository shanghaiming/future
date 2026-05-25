"""
Alpha Futures V121 — INDEPENDENT VERIFICATION + PARAMETER FINE-TUNING
======================================================================
V121 FOCUS:
  STEP 1: Verify the V120 champion: ROC(5)>1.0% AND Z-score>1.5, hold=1, top_n=1
  STEP 2: Parameter grid search: ROC x Z-score (35 combos)
  STEP 3: Enhancement attempts (ranking, filters)
  STEP 4: Detailed trade analysis

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
    print("=" * 140)
    print("  Alpha Futures V121 — INDEPENDENT VERIFICATION + PARAMETER FINE-TUNING")
    print("=" * 140)
    print(f"\n  Champion signal to verify: ROC(5) > 1.0% AND Z-score > 1.5, hold=1, top_n=1")
    print(f"  ALL signals at close di, entry at O[si, di+1] (NEXT DAY OPEN)")

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
    print("\n[Precompute] Daily returns, ROC(5), Z-scores, volume changes...", flush=True)
    t0 = time.time()

    # Daily returns in percent
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

    # Z-score of daily returns (20-day rolling)
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

    # Volume ratio (today vs yesterday) for filter
    VOL_RATIO = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(1, ND):
            if not np.isnan(V[si, di]) and not np.isnan(V[si, di-1]) and V[si, di-1] > 0:
                VOL_RATIO[si, di] = V[si, di] / V[si, di-1]

    print(f"  All indicators computed ({time.time()-t0:.1f}s)")

    # ================================================================
    # GENERIC MULTI-COMMODITY BACKTEST (supports custom ranking + filters)
    # ================================================================
    def backtest_multi(roc_thresh, z_thresh, hold_days=1, top_n=1,
                       rank_mode='roc', filter_mode='none',
                       start_di=MIN_TRAIN, end_di=None,
                       return_trades=False):
        """
        rank_mode: 'roc' = rank by ROC5 magnitude,
                   'roc_z' = rank by ROC5 * Z_score
        filter_mode: 'none', 'up_day' (C>O), 'roc_improving' (ROC5[di]>ROC5[di-1]),
                     'vol_increasing' (V[di]>V[di-1])
        """
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
                    port_val += cp * mult * pos['lots'] - cp * mult * abs(pos['lots']) * COMM
            daily_equity.append(port_val)

            # Close positions whose hold_days have elapsed
            closed = []
            for pos in positions:
                days_held = di - pos['entry_di']
                if days_held >= pos['hold_days']:
                    exit_price = C[pos['si'], di]
                    if np.isnan(exit_price) or exit_price <= 0:
                        exit_price = pos['entry_price']
                    mult = MULT.get(pos['sym'], DEF_MULT)
                    mkt_val = exit_price * mult * abs(pos['lots'])
                    cash += mkt_val - mkt_val * COMM
                    pnl = (exit_price - pos['entry_price']) * mult * pos['lots']
                    invested = pos['entry_price'] * mult * abs(pos['lots'])
                    pnl_pct = pnl / invested * 100 if invested > 0 else 0
                    trades.append({
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'entry_di': pos['entry_di'],
                        'exit_di': di,
                        'sym': pos['sym'],
                        'entry_price': pos['entry_price'],
                        'exit_price': exit_price,
                        'roc5': pos.get('roc5', np.nan),
                        'zscore': pos.get('zscore', np.nan),
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
                roc = ROC5[s, di]
                zs = ZSCORE[s, di]
                if np.isnan(roc) or np.isnan(zs):
                    continue
                if roc <= roc_thresh or zs <= z_thresh:
                    continue
                # Apply filter
                if filter_mode == 'up_day':
                    co = C[s, di]
                    op = O[s, di]
                    if np.isnan(co) or np.isnan(op) or co <= op:
                        continue
                elif filter_mode == 'roc_improving':
                    roc_prev = ROC5[s, di-1]
                    if np.isnan(roc_prev) or roc <= roc_prev:
                        continue
                elif filter_mode == 'vol_increasing':
                    vr = VOL_RATIO[s, di]
                    if np.isnan(vr) or vr <= 1.0:
                        continue

                ep = O[s, entry_di]
                if np.isnan(ep) or ep <= 0:
                    continue
                if any(p['si'] == s for p in positions):
                    continue

                # Compute ranking score
                if rank_mode == 'roc_z':
                    sc = roc * zs if not np.isnan(roc) and not np.isnan(zs) else 0
                else:
                    sc = roc
                candidates.append((sc, s, ep, roc, zs))

            if not candidates:
                continue

            candidates.sort(key=lambda x: -x[0])
            n_slots = top_n - len(positions)
            cap_per_slot = cash / max(1, n_slots)

            for sc_val, s, price, roc_val, zs_val in candidates[:max(0, n_slots)]:
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
                positions.append({
                    'si': s, 'entry_price': price, 'entry_di': entry_di,
                    'lots': contracts, 'dir': 1, 'sym': sym,
                    'hold_days': hold_days,
                    'roc5': roc_val, 'zscore': zs_val,
                })

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
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'entry_di': pos['entry_di'],
                'exit_di': ae,
                'sym': pos['sym'],
                'entry_price': pos['entry_price'],
                'exit_price': exit_price,
                'roc5': pos.get('roc5', np.nan),
                'zscore': pos.get('zscore', np.nan),
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

        result = {
            'ann': ann, 'wr': wr, 'n': n_trades, 'avg_pnl': avg_pnl,
            'final_cash': cash, 'n_days': n_days_test, 'mdd': mdd,
            'trades': trades, 'daily_equity': daily_equity,
        }
        return result

    # ================================================================
    # STEP 1: VERIFY CHAMPION — ROC(5) > 1.0% AND Z-score > 1.5
    # ================================================================
    print(f"\n{'=' * 140}")
    print("  STEP 1: VERIFY CHAMPION — ROC(5)>1.0% AND Z-score>1.5, hold=1, top_n=1, rank=ROC")
    print(f"{'=' * 140}")

    champ = backtest_multi(roc_thresh=1.0, z_thresh=1.5, hold_days=1, top_n=1,
                           rank_mode='roc', filter_mode='none', return_trades=True)
    print(f"\n  Champion Verification Results:")
    print(f"    Annual Return: {champ['ann']:>+.1f}%")
    print(f"    Win Rate:      {champ['wr']:.1f}%")
    print(f"    Max Drawdown:  {champ['mdd']:.1f}%")
    print(f"    # Trades:      {champ['n']}")
    print(f"    Avg PnL/trade: {champ['avg_pnl']:.3f}%")
    print(f"    Final Cash:    {champ['final_cash']:,.0f}")

    # Walk-forward by year
    print(f"\n  Walk-Forward by Year:")
    print(f"  {'Year':>6} | {'Ann Ret':>10} | {'WR':>6} | {'Trades':>6} | {'MDD':>8}")
    print(f"  {'-'*50}")
    for yr in range(2020, 2026):
        ts = te = None
        for di in range(ND):
            if dates[di].year == yr and ts is None:
                ts = di
            if dates[di].year == yr + 1 and te is None:
                te = di
        if ts is None:
            print(f"  {yr:>6} | {'N/A':>10} | {'N/A':>6} | {'N/A':>6} | {'N/A':>8}")
            continue
        if te is None:
            te = ND
        yr_res = backtest_multi(roc_thresh=1.0, z_thresh=1.5, hold_days=1, top_n=1,
                                rank_mode='roc', filter_mode='none',
                                start_di=ts, end_di=te, return_trades=True)
        print(f"  {yr:>6} | {yr_res['ann']:>+9.1f}% | {yr_res['wr']:>5.1f}% | {yr_res['n']:>6} | {yr_res['mdd']:>+7.1f}%")

    # ================================================================
    # STEP 2: PARAMETER GRID — ROC threshold x Z-score threshold
    # ================================================================
    print(f"\n{'=' * 140}")
    print("  STEP 2: PARAMETER GRID — ROC threshold x Z-score threshold (hold=1, top_n=1)")
    print(f"{'=' * 140}")

    roc_vals = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    z_vals = [1.0, 1.25, 1.5, 1.75, 2.0]

    # Header
    print(f"\n  {'ROC\\Z':>8}", end="")
    for zv in z_vals:
        print(f" | {'Z>'+str(zv):>14}", end="")
    print()
    print(f"  {'-'*100}")

    grid_results = []
    for rv in roc_vals:
        print(f"  {'ROC>'+str(rv)+'%':>8}", end="")
        for zv in z_vals:
            res = backtest_multi(roc_thresh=rv, z_thresh=zv, hold_days=1, top_n=1,
                                 rank_mode='roc', filter_mode='none', return_trades=False)
            print(f" | {res['ann']:>+10.1f}%/{res['n']:>3}", end="")
            grid_results.append({
                'roc': rv, 'z': zv, 'ann': res['ann'], 'wr': res['wr'],
                'n': res['n'], 'mdd': res['mdd'],
            })
        print()

    # Find best
    best_grid = max(grid_results, key=lambda x: x['ann'])
    print(f"\n  Best Grid Result: ROC>{best_grid['roc']}%, Z>{best_grid['z']}")
    print(f"    Annual: {best_grid['ann']:>+.1f}%, WR: {best_grid['wr']:.1f}%, Trades: {best_grid['n']}, MDD: {best_grid['mdd']:.1f}%")

    # Walk-forward for top 5 configs
    grid_sorted = sorted(grid_results, key=lambda x: -x['ann'])[:5]
    print(f"\n  Top 5 Configs — Walk-Forward by Year:")
    print(f"  {'Config':>25} | {'Ann':>8} | {'2020':>8} | {'2021':>8} | {'2022':>8} | {'2023':>8} | {'2024':>8} | {'2025':>8} | {'WF+':>4}")
    print(f"  {'-'*110}")
    for cfg in grid_sorted:
        wf_pos = 0
        wf_strs = {}
        for yr in range(2020, 2026):
            ts = te = None
            for di in range(ND):
                if dates[di].year == yr and ts is None:
                    ts = di
                if dates[di].year == yr + 1 and te is None:
                    te = di
            if ts is None:
                wf_strs[yr] = "N/A"
                continue
            if te is None:
                te = ND
            yr_res = backtest_multi(roc_thresh=cfg['roc'], z_thresh=cfg['z'],
                                    hold_days=1, top_n=1, rank_mode='roc',
                                    filter_mode='none', start_di=ts, end_di=te)
            wf_strs[yr] = f"{yr_res['ann']:>+7.1f}%"
            if yr_res['ann'] > 0:
                wf_pos += 1
        label = f"ROC>{cfg['roc']}% Z>{cfg['z']}"
        print(f"  {label:>25} | {cfg['ann']:>+7.1f}% | {wf_strs[2020]:>8} | {wf_strs[2021]:>8} | {wf_strs[2022]:>8} | {wf_strs[2023]:>8} | {wf_strs[2024]:>8} | {wf_strs[2025]:>8} | {wf_pos}/6")

    # ================================================================
    # STEP 3: ENHANCEMENT ATTEMPTS
    # ================================================================
    print(f"\n{'=' * 140}")
    print("  STEP 3: ENHANCEMENT ATTEMPTS (all use hold=1, top_n=1)")
    print(f"{'=' * 140}")

    enhancements = [
        ("Champion (baseline)", 1.0, 1.5, 'roc', 'none'),
        ("A) Rank by ROC*Z", 1.0, 1.5, 'roc_z', 'none'),
        ("B) Filter: up day (C>O)", 1.0, 1.5, 'roc', 'up_day'),
        ("C) Filter: ROC improving", 1.0, 1.5, 'roc', 'roc_improving'),
        ("D) Filter: vol increasing", 1.0, 1.5, 'roc', 'vol_increasing'),
        ("E) ROC*Z + up day", 1.0, 1.5, 'roc_z', 'up_day'),
        ("F) ROC*Z + ROC improving", 1.0, 1.5, 'roc_z', 'roc_improving'),
        ("G) ROC*Z + vol increasing", 1.0, 1.5, 'roc_z', 'vol_increasing'),
    ]

    # Also try best grid config with enhancements
    best_r, best_z = best_grid['roc'], best_grid['z']
    enhancements += [
        (f"H) Best grid ROC>{best_r} Z>{best_z} + ROC*Z", best_r, best_z, 'roc_z', 'none'),
        (f"I) Best grid ROC>{best_r} Z>{best_z} + up day", best_r, best_z, 'roc', 'up_day'),
        (f"J) Best grid ROC>{best_r} Z>{best_z} + ROC*Z + up day", best_r, best_z, 'roc_z', 'up_day'),
    ]

    print(f"\n  {'Variant':>45} | {'Ann Ret':>10} | {'WR':>6} | {'Trades':>6} | {'MDD':>8} | {'Avg PnL':>8}")
    print(f"  {'-'*100}")

    enh_results = []
    for label, rv, zv, rm, fm in enhancements:
        res = backtest_multi(roc_thresh=rv, z_thresh=zv, hold_days=1, top_n=1,
                             rank_mode=rm, filter_mode=fm, return_trades=False)
        print(f"  {label:>45} | {res['ann']:>+9.1f}% | {res['wr']:>5.1f}% | {res['n']:>6} | {res['mdd']:>+7.1f}% | {res['avg_pnl']:>+7.3f}%")
        enh_results.append({'label': label, **res})

    best_enh = max(enh_results, key=lambda x: x['ann'])
    print(f"\n  Best Enhancement: {best_enh['label']}")
    print(f"    Annual: {best_enh['ann']:>+.1f}%, WR: {best_enh['wr']:.1f}%, Trades: {best_enh['n']}, MDD: {best_enh['mdd']:.1f}%")

    # Walk-forward for the best enhancement
    # Parse params from the best enhancement label
    be_r, be_z, be_rm, be_fm = 1.0, 1.5, 'roc', 'none'
    for label, rv, zv, rm, fm in enhancements:
        if label == best_enh['label']:
            be_r, be_z, be_rm, be_fm = rv, zv, rm, fm
            break

    print(f"\n  Best Enhancement Walk-Forward:")
    print(f"  {'Year':>6} | {'Ann Ret':>10} | {'WR':>6} | {'Trades':>6} | {'MDD':>8}")
    print(f"  {'-'*50}")
    wf_pos_best = 0
    for yr in range(2020, 2026):
        ts = te = None
        for di in range(ND):
            if dates[di].year == yr and ts is None:
                ts = di
            if dates[di].year == yr + 1 and te is None:
                te = di
        if ts is None:
            print(f"  {yr:>6} | {'N/A':>10} | {'N/A':>6} | {'N/A':>6} | {'N/A':>8}")
            continue
        if te is None:
            te = ND
        yr_res = backtest_multi(roc_thresh=be_r, z_thresh=be_z, hold_days=1, top_n=1,
                                rank_mode=be_rm, filter_mode=be_fm,
                                start_di=ts, end_di=te, return_trades=True)
        print(f"  {yr:>6} | {yr_res['ann']:>+9.1f}% | {yr_res['wr']:>5.1f}% | {yr_res['n']:>6} | {yr_res['mdd']:>+7.1f}%")
        if yr_res['ann'] > 0:
            wf_pos_best += 1
    print(f"  WF positive: {wf_pos_best}/6")

    # ================================================================
    # STEP 4: DETAILED TRADE ANALYSIS (champion config)
    # ================================================================
    print(f"\n{'=' * 140}")
    print("  STEP 4: DETAILED TRADE ANALYSIS — Champion Config (ROC>1.0% Z>1.5)")
    print(f"{'=' * 140}")

    champ_trades = champ['trades']

    # Sort by PnL
    trades_sorted = sorted(champ_trades, key=lambda x: -x['pnl'])

    # Top 20 best trades
    print(f"\n  TOP 20 BEST TRADES:")
    print(f"  {'#':>3} | {'Date':>12} | {'Commodity':>10} | {'Entry':>10} | {'Exit':>10} | {'ROC5':>8} | {'Z-score':>8} | {'PnL':>14} | {'PnL%':>8}")
    print(f"  {'-'*120}")
    for i, t in enumerate(trades_sorted[:20]):
        di_entry = t['entry_di']
        dt = dates[di_entry] if di_entry < ND else 'N/A'
        roc_str = f"{t['roc5']:.2f}" if not np.isnan(t.get('roc5', np.nan)) else "N/A"
        zs_str = f"{t['zscore']:.2f}" if not np.isnan(t.get('zscore', np.nan)) else "N/A"
        print(f"  {i+1:>3} | {str(dt):>12} | {t['sym']:>10} | {t['entry_price']:>10.1f} | {t['exit_price']:>10.1f} | {roc_str:>8} | {zs_str:>8} | {t['pnl']:>+14,.0f} | {t['pnl_pct']:>+7.2f}%")

    # Bottom 20 worst trades
    print(f"\n  BOTTOM 20 WORST TRADES:")
    print(f"  {'#':>3} | {'Date':>12} | {'Commodity':>10} | {'Entry':>10} | {'Exit':>10} | {'ROC5':>8} | {'Z-score':>8} | {'PnL':>14} | {'PnL%':>8}")
    print(f"  {'-'*120}")
    for i, t in enumerate(trades_sorted[-20:]):
        di_entry = t['entry_di']
        dt = dates[di_entry] if di_entry < ND else 'N/A'
        roc_str = f"{t['roc5']:.2f}" if not np.isnan(t.get('roc5', np.nan)) else "N/A"
        zs_str = f"{t['zscore']:.2f}" if not np.isnan(t.get('zscore', np.nan)) else "N/A"
        print(f"  {i+1:>3} | {str(dt):>12} | {t['sym']:>10} | {t['entry_price']:>10.1f} | {t['exit_price']:>10.1f} | {roc_str:>8} | {zs_str:>8} | {t['pnl']:>+14,.0f} | {t['pnl_pct']:>+7.2f}%")

    # Loss clustering analysis
    print(f"\n  LOSS CLUSTERING ANALYSIS:")
    losses = [t for t in champ_trades if t['pnl'] < 0]
    wins = [t for t in champ_trades if t['pnl'] > 0]
    print(f"    Total trades: {len(champ_trades)}, Wins: {len(wins)}, Losses: {len(losses)}")
    print(f"    Win avg PnL: {np.mean([t['pnl'] for t in wins]):+,.0f}" if wins else "    No wins")
    print(f"    Loss avg PnL: {np.mean([t['pnl'] for t in losses]):+,.0f}" if losses else "    No losses")

    # Losses by commodity
    loss_by_sym = {}
    for t in losses:
        loss_by_sym[t['sym']] = loss_by_sym.get(t['sym'], 0) + t['pnl']
    if loss_by_sym:
        sym_sorted = sorted(loss_by_sym.items(), key=lambda x: x[1])
        print(f"\n    Biggest loss commodities (cumulative PnL):")
        for sym, pnl in sym_sorted[:10]:
            n_loss = sum(1 for t in losses if t['sym'] == sym)
            print(f"      {sym:<10} : {pnl:>+14,.0f}  ({n_loss} losing trades)")

    # Losses by year
    loss_by_year = {}
    for t in losses:
        di_entry = t['entry_di']
        if di_entry < ND:
            yr = dates[di_entry].year
            loss_by_year[yr] = loss_by_year.get(yr, 0) + t['pnl']
    if loss_by_year:
        print(f"\n    Losses by year:")
        for yr in sorted(loss_by_year.keys()):
            n_loss = sum(1 for t in losses if t['entry_di'] < ND and dates[t['entry_di']].year == yr)
            print(f"      {yr}: {loss_by_year[yr]:>+14,.0f}  ({n_loss} losing trades)")

    # Consecutive losses
    if champ_trades:
        trades_by_entry = sorted(champ_trades, key=lambda x: x['entry_di'])
        max_consec_loss = 0
        consec_loss = 0
        for t in trades_by_entry:
            if t['pnl'] < 0:
                consec_loss += 1
                max_consec_loss = max(max_consec_loss, consec_loss)
            else:
                consec_loss = 0
        print(f"\n    Max consecutive losses: {max_consec_loss}")

    # ================================================================
    # FINAL SUMMARY
    # ================================================================
    print(f"\n{'=' * 140}")
    print("  FINAL SUMMARY")
    print(f"{'=' * 140}")
    print(f"\n  1. VERIFICATION: Champion ROC>1.0% Z>1.5 hold=1 top_n=1")
    print(f"     Annual Return: {champ['ann']:>+.1f}%")
    print(f"     Reported:      +306.2%")
    print(f"     Match: {'YES' if abs(champ['ann'] - 306.2) < 5 else 'NO — needs investigation'}")

    print(f"\n  2. PARAMETER SENSITIVITY: Best grid config = ROC>{best_grid['roc']}% Z>{best_grid['z']}")
    print(f"     Annual: {best_grid['ann']:>+.1f}%, Trades: {best_grid['n']}")

    print(f"\n  3. BEST ENHANCEMENT: {best_enh['label']}")
    print(f"     Annual: {best_enh['ann']:>+.1f}%, WR: {best_enh['wr']:.1f}%, MDD: {best_enh['mdd']:.1f}%")

    if best_enh['ann'] > champ['ann']:
        print(f"\n  IMPROVEMENT OVER CHAMPION: {best_enh['ann'] - champ['ann']:>+.1f} percentage points")
    else:
        print(f"\n  No enhancement beat the champion baseline.")

    print(f"\n  Elapsed: {time.time()-t_start:.1f}s total")


if __name__ == '__main__':
    main()
