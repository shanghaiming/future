"""
Alpha Futures V130 — REGIME-ADAPTIVE + MULTI-SIGNAL PORTFOLIO
=============================================================
V129 proved V121 can't be improved with quality filters. Path to +600% is:
1) Regime adaptation — size up when conditions favor momentum
2) Multi-signal portfolio — combine V121 with uncorrelated signals
3) Equity curve adaptation — adapt to drawdowns/recoveries
4) Dynamic thresholds — tighten/loosen based on recent win rate

ALL signals use NEXT-OPEN execution.
"""
import sys, os, time, warnings
import numpy as np
import talib
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alpha_v2 import load_all_data, MIN_TRAIN, CASH0

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
    print("=" * 120)
    print("  Alpha Futures V130 — REGIME-ADAPTIVE + MULTI-SIGNAL PORTFOLIO")
    print("=" * 120)

    print("\n[Data] Loading...", flush=True)
    t_start = time.time()
    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)
    years = sorted(set(d.year for d in dates))
    print(f"  {NS} commodities, {ND} days")

    # ================================================================
    # PRECOMPUTE
    # ================================================================
    print("\n[Precompute]...", flush=True)
    t0 = time.time()

    RET = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si].astype(np.float64)
        for di in range(1, ND):
            if not np.isnan(c[di]) and not np.isnan(c[di-1]) and c[di-1] > 0:
                RET[si, di] = (c[di] / c[di-1] - 1) * 100

    ROC5 = np.full((NS, ND), np.nan)
    ROC20 = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si].astype(np.float64)
        ROC5[si] = talib.ROC(c, timeperiod=5)
        ROC20[si] = talib.ROC(c, timeperiod=20)

    ATR14 = np.full((NS, ND), np.nan)
    for si in range(NS):
        ATR14[si] = talib.ATR(H[si].astype(np.float64), L[si].astype(np.float64),
                               C[si].astype(np.float64), timeperiod=14)

    ZSCORE = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(21, ND):
            rets = RET[si, di-20:di]
            valid = rets[~np.isnan(rets)]
            if len(valid) < 10:
                continue
            std_r = np.std(valid, ddof=1)
            if std_r > 0 and not np.isnan(RET[si, di]):
                ZSCORE[si, di] = (RET[si, di] - np.mean(valid)) / std_r

    BODY_RATIO = np.full((NS, ND), np.nan)
    BAR_DIR = np.zeros((NS, ND), dtype=int)
    for si in range(NS):
        for di in range(ND):
            h, l, c, o = H[si, di], L[si, di], C[si, di], O[si, di]
            if any(np.isnan(x) for x in [h, l, c, o]) or h == l:
                continue
            BODY_RATIO[si, di] = abs(c - o) / (h - l)
            if c > o: BAR_DIR[si, di] = 1
            elif c < o: BAR_DIR[si, di] = -1

    # Market-wide momentum (average ROC5 across all commodities)
    MARKET_MOM = np.full(ND, np.nan)
    for di in range(ND):
        vals = ROC5[:, di]
        valid = vals[~np.isnan(vals)]
        if len(valid) > 10:
            MARKET_MOM[di] = np.mean(valid)

    # Market-wide trend strength (% of commodities with ROC5 > 0)
    MARKET_BREADTH = np.full(ND, np.nan)
    for di in range(ND):
        vals = ROC5[:, di]
        valid = vals[~np.isnan(vals)]
        if len(valid) > 10:
            MARKET_BREADTH[di] = np.sum(valid > 0) / len(valid) * 100

    # Market volatility (average ATR% across commodities)
    MARKET_VOL = np.full(ND, np.nan)
    for di in range(ND):
        vals = []
        for si in range(NS):
            if not np.isnan(ATR14[si, di]) and not np.isnan(C[si, di]) and C[si, di] > 0:
                vals.append(ATR14[si, di] / C[si, di] * 100)
        if len(vals) > 10:
            MARKET_VOL[di] = np.mean(vals)

    # OI change ratio
    OI_CHANGE = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(6, ND):
            ois = OI[si, di-5:di]
            valid = ois[~np.isnan(ois)]
            if len(valid) < 3 or np.mean(valid) == 0:
                continue
            if not np.isnan(OI[si, di]):
                OI_CHANGE[si, di] = OI[si, di] / np.mean(valid) - 1

    print(f"  All indicators computed ({time.time()-t0:.1f}s)")

    # ================================================================
    # BACKTEST ENGINE WITH REGIME ADAPTATION
    # ================================================================
    def backtest_v130(signal_func, hold_days=1, top_n=1,
                      regime_mode='none', start_di=MIN_TRAIN, end_di=None, desc=""):
        """
        regime_mode: 'none' = fixed sizing
                     'equity_curve' = reduce when equity declining
                     'market_mom' = reduce when MARKET_MOM < 0
                     'market_breadth' = reduce when < 50% bullish
                     'recent_wr' = reduce when recent 20-trade WR < 55%
                     'combined' = all of the above
        """
        if end_di is None:
            end_di = ND

        cash = float(CASH0)
        positions = []
        trades = []
        daily_equity = []
        peak_equity = float(CASH0)

        for di in range(start_di, end_di - 1):
            port_val = cash
            for pos in positions:
                cp = C[pos['si'], di]
                if not np.isnan(cp) and cp > 0:
                    mult = MULT.get(pos['sym'], DEF_MULT)
                    port_val += cp * mult * pos['lots'] - cp * mult * abs(pos['lots']) * COMM
            daily_equity.append(port_val)
            if port_val > peak_equity:
                peak_equity = port_val

            # Close positions
            closed = []
            for pos in positions:
                days_held = di - pos['entry_di']
                if days_held >= pos['hold_days']:
                    exit_price = C[pos['si'], di]
                    if np.isnan(exit_price) or exit_price <= 0:
                        exit_price = pos['entry_price']
                    mult = MULT.get(pos['sym'], DEF_MULT)
                    pnl = (exit_price - pos['entry_price']) * mult * pos['lots']
                    invested = pos['entry_price'] * mult * abs(pos['lots'])
                    pnl_pct = pnl / invested * 100 if invested > 0 else 0
                    mkt_val = exit_price * mult * abs(pos['lots'])
                    cash += mkt_val - mkt_val * COMM
                    trades.append({
                        'pnl': pnl, 'pnl_pct': pnl_pct,
                        'entry_di': pos['entry_di'], 'exit_di': di,
                        'sym': pos['sym'], 'signal_type': pos.get('signal_type', ''),
                    })
                    closed.append(pos)
            for pos in closed:
                positions.remove(pos)

            if len(positions) >= top_n:
                continue

            # Position sizing based on regime
            risk_frac = 0.95
            if regime_mode != 'none':
                if regime_mode == 'equity_curve' or regime_mode == 'combined':
                    if peak_equity > 0:
                        dd = (port_val - peak_equity) / peak_equity
                        if dd < -0.5:
                            risk_frac = min(risk_frac, 0.2)
                        elif dd < -0.3:
                            risk_frac = min(risk_frac, 0.3)
                        elif dd < -0.15:
                            risk_frac = min(risk_frac, 0.5)

                if regime_mode == 'market_mom' or regime_mode == 'combined':
                    if not np.isnan(MARKET_MOM[di]):
                        if MARKET_MOM[di] < -1.0:
                            risk_frac = min(risk_frac, 0.2)
                        elif MARKET_MOM[di] < 0:
                            risk_frac = min(risk_frac, 0.5)

                if regime_mode == 'market_breadth' or regime_mode == 'combined':
                    if not np.isnan(MARKET_BREADTH[di]):
                        if MARKET_BREADTH[di] < 40:
                            risk_frac = min(risk_frac, 0.3)
                        elif MARKET_BREADTH[di] < 50:
                            risk_frac = min(risk_frac, 0.5)

                if regime_mode == 'recent_wr' or regime_mode == 'combined':
                    if len(trades) >= 20:
                        recent = trades[-20:]
                        recent_wr = sum(1 for t in recent if t['pnl_pct'] > 0) / 20
                        if recent_wr < 0.40:
                            risk_frac = min(risk_frac, 0.3)
                        elif recent_wr < 0.50:
                            risk_frac = min(risk_frac, 0.5)

            entry_di = di + 1
            if entry_di >= end_di:
                continue

            candidates = signal_func(di)
            if not candidates:
                continue

            candidates.sort(key=lambda x: -x[0])
            n_slots = top_n - len(positions)
            cap_per_slot = cash * risk_frac / max(1, n_slots)

            for item in candidates[:max(0, n_slots)]:
                if len(item) == 3:
                    sc_val, s, price = item
                    sig_type = 'primary'
                else:
                    sc_val, s, price, sig_type = item
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
                    'hold_days': hold_days, 'score': sc_val,
                    'signal_type': sig_type,
                })

        # Close remaining
        for pos in positions:
            ae = end_di - 1
            exit_price = C[pos['si'], min(ae, ND-1)]
            if np.isnan(exit_price) or exit_price <= 0:
                exit_price = pos['entry_price']
            mult = MULT.get(pos['sym'], DEF_MULT)
            pnl = (exit_price - pos['entry_price']) * mult * pos['lots']
            invested = pos['entry_price'] * mult * abs(pos['lots'])
            pnl_pct = pnl / invested * 100 if invested > 0 else 0
            mkt_val = exit_price * mult * abs(pos['lots'])
            cash += mkt_val - mkt_val * COMM
            trades.append({
                'pnl': pnl, 'pnl_pct': pnl_pct,
                'entry_di': pos['entry_di'], 'exit_di': ae,
                'sym': pos['sym'], 'signal_type': pos.get('signal_type', ''),
            })

        n_days_test = end_di - start_di
        ann = annual_return(cash, CASH0, n_days_test)
        wr = np.mean([1 if t['pnl_pct'] > 0 else 0 for t in trades]) * 100 if trades else 0
        n_trades = len(trades)
        avg_pnl = np.mean([t['pnl_pct'] for t in trades]) if trades else 0

        if daily_equity:
            eq_arr = np.array(daily_equity)
            peak_arr = np.maximum.accumulate(eq_arr)
            dd_arr = (eq_arr - peak_arr) / peak_arr * 100
            mdd = np.min(dd_arr)
            rets = np.diff(eq_arr) / eq_arr[:-1]
            sharpe = np.mean(rets) / np.std(rets) * np.sqrt(252) if np.std(rets) > 0 else 0
            sortino_down = rets[rets < 0]
            sortino = np.mean(rets) / np.std(sortino_down) * np.sqrt(252) if len(sortino_down) > 0 and np.std(sortino_down) > 0 else 0
        else:
            mdd = 0.0
            sharpe = 0.0
            sortino = 0.0

        return {
            'ann': ann, 'wr': wr, 'n': n_trades, 'avg_pnl': avg_pnl,
            'final_cash': cash, 'n_days': n_days_test, 'mdd': mdd,
            'sharpe': sharpe, 'sortino': sortino, 'trades': trades, 'desc': desc,
        }

    def print_result(r, label=""):
        print(f"  {label:55s} | Ann={r['ann']:+8.1f}% | WR={r['wr']:5.1f}% | "
              f"N={r['n']:4d} | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f} | So={r['sortino']:5.1f}")

    def walk_forward(signal_func, hold_days=1, top_n=1, regime_mode='none', desc=""):
        wf = {}
        for yr in [2020, 2021, 2022, 2023, 2024, 2025]:
            yr_start = yr_end = None
            for di in range(ND):
                if dates[di].year == yr and yr_start is None:
                    yr_start = di
                if dates[di].year == yr:
                    yr_end = di + 1
            if yr_start is None:
                continue
            r = backtest_v130(signal_func, hold_days=hold_days, top_n=top_n,
                              regime_mode=regime_mode,
                              start_di=yr_start, end_di=yr_end, desc=f"{desc} {yr}")
            wf[yr] = r['ann']
        return wf

    # ================================================================
    # SIGNAL FUNCTIONS
    # ================================================================

    # V121 Champion signal
    def signal_v121(di):
        candidates = []
        for s in range(NS):
            roc = ROC5[s, di]
            zs = ZSCORE[s, di]
            if np.isnan(roc) or np.isnan(zs) or roc <= 1.0 or zs <= 1.5:
                continue
            roc_prev = ROC5[s, di-1] if di > 0 else np.nan
            if not np.isnan(roc_prev) and roc <= roc_prev:
                continue
            ep = O[s, di+1]
            if np.isnan(ep) or ep <= 0:
                continue
            candidates.append((roc * zs, s, ep, 'v121'))
        return candidates

    # V121 with higher Z threshold
    def signal_v121_z2(di):
        candidates = []
        for s in range(NS):
            roc = ROC5[s, di]
            zs = ZSCORE[s, di]
            if np.isnan(roc) or np.isnan(zs) or roc <= 1.0 or zs <= 2.0:
                continue
            roc_prev = ROC5[s, di-1] if di > 0 else np.nan
            if not np.isnan(roc_prev) and roc <= roc_prev:
                continue
            ep = O[s, di+1]
            if np.isnan(ep) or ep <= 0:
                continue
            candidates.append((roc * zs, s, ep, 'v121_z2'))
        return candidates

    # Final Flag Breakout (from V128)
    def signal_final_flag(di):
        candidates = []
        for s in range(NS):
            roc20 = ROC20[s, di]
            if np.isnan(roc20) or roc20 <= 5.0 or di < 6:
                continue
            highs_5 = H[s, di-4:di+1]
            lows_5 = L[s, di-4:di+1]
            if any(np.isnan(x) for x in highs_5) or any(np.isnan(x) for x in lows_5):
                continue
            range_5 = np.max(highs_5) - np.min(lows_5)
            atr = ATR14[s, di]
            if np.isnan(atr) or atr <= 0 or range_5 > atr * 3.0:
                continue
            high_4 = np.max(H[s, di-4:di])
            c = C[s, di]
            if np.isnan(c) or c <= high_4:
                continue
            ep = O[s, di+1]
            if np.isnan(ep) or ep <= 0:
                continue
            candidates.append((roc20 * (c - high_4) / atr, s, ep, 'final_flag'))
        return candidates

    # Two-leg pullback (from V128)
    def signal_pullback(di):
        candidates = []
        for s in range(NS):
            roc20 = ROC20[s, di]
            if np.isnan(roc20) or roc20 <= 3.0 or di < 3:
                continue
            pb = 0
            for k in range(1, min(di, 6) + 1):
                if BAR_DIR[s, di - k] == -1:
                    pb += 1
                else:
                    break
            if pb < 2 or BAR_DIR[s, di] != 1:
                continue
            br = BODY_RATIO[s, di]
            if np.isnan(br) or br < 0.3:
                continue
            ep = O[s, di+1]
            if np.isnan(ep) or ep <= 0:
                continue
            candidates.append((roc20 * br, s, ep, 'pullback'))
        return candidates

    # Multi-signal: V121 primary + others as secondary
    def signal_multi(di):
        # Get V121 signals
        v121_sigs = signal_v121(di)
        if v121_sigs:
            return v121_sigs  # V121 always takes priority

        # If no V121 signal, try Z>2 variant
        v121_z2_sigs = signal_v121_z2(di)
        if v121_z2_sigs:
            return v121_z2_sigs

        # Try final flag
        ff_sigs = signal_final_flag(di)
        if ff_sigs:
            return ff_sigs

        # Try pullback
        pb_sigs = signal_pullback(di)
        if pb_sigs:
            return pb_sigs

        return []

    # Multi-signal with scoring: combine all signals, pick best score
    def signal_multi_scored(di):
        all_sigs = []

        # V121 (weight 3x)
        for item in signal_v121(di):
            sc, s, ep, st = item
            all_sigs.append((sc * 3, s, ep, st))

        # Z>2 variant (weight 2x)
        for item in signal_v121_z2(di):
            sc, s, ep, st = item
            if not any(x[1] == s for x in all_sigs):
                all_sigs.append((sc * 2, s, ep, st))

        # Final flag (weight 1.5x)
        for item in signal_final_flag(di):
            sc, s, ep, st = item
            if not any(x[1] == s for x in all_sigs):
                all_sigs.append((sc * 1.5, s, ep, st))

        # Pullback (weight 1x)
        for item in signal_pullback(di):
            sc, s, ep, st = item
            if not any(x[1] == s for x in all_sigs):
                all_sigs.append((sc, s, ep, st))

        return all_sigs

    # ================================================================
    # SECTION 1: REGIME ADAPTATION FOR V121
    # ================================================================
    print("\n" + "=" * 120)
    print("  SECTION 1: REGIME ADAPTATION FOR V121 CHAMPION")
    print("=" * 120)

    regimes = ['none', 'equity_curve', 'market_mom', 'market_breadth', 'recent_wr', 'combined']
    for regime in regimes:
        r = backtest_v130(signal_v121, hold_days=1, top_n=1,
                          regime_mode=regime, desc=f"V121 + {regime}")
        print_result(r, label=f"V121 + regime={regime}")

    # ================================================================
    # SECTION 2: MULTI-SIGNAL PORTFOLIO
    # ================================================================
    print("\n" + "=" * 120)
    print("  SECTION 2: MULTI-SIGNAL PORTFOLIO")
    print("=" * 120)

    signals = [
        ("V121 only", signal_v121),
        ("V121 Z>2 only", signal_v121_z2),
        ("Final Flag only", signal_final_flag),
        ("Pullback only", signal_pullback),
        ("Multi (V121>Z2>FF>PB)", signal_multi),
        ("Multi scored", signal_multi_scored),
    ]

    for name, func in signals:
        r = backtest_v130(func, hold_days=1, top_n=1, regime_mode='none', desc=name)
        print_result(r, label=name)

    # ================================================================
    # SECTION 3: BEST COMBOS WITH TOP_N AND HOLD
    # ================================================================
    print("\n" + "=" * 120)
    print("  SECTION 3: TOP_N x HOLD for multi-signal")
    print("=" * 120)

    for name, func in [("Multi (V121>Z2>FF>PB)", signal_multi),
                       ("Multi scored", signal_multi_scored)]:
        print(f"\n  {name}:")
        for hold in [1, 2, 3]:
            for topn in [1, 2, 3]:
                r = backtest_v130(func, hold_days=hold, top_n=topn,
                                  regime_mode='none', desc=f"{name} h={hold} t={topn}")
                print(f"    hold={hold} top_n={topn}: Ann={r['ann']:+8.1f}% | "
                      f"WR={r['wr']:5.1f}% | N={r['n']:4d} | MDD={r['mdd']:6.1f}%")

    # ================================================================
    # SECTION 4: MULTI-SIGNAL + REGIME ADAPTATION
    # ================================================================
    print("\n" + "=" * 120)
    print("  SECTION 4: MULTI-SIGNAL + BEST REGIME ADAPTATION")
    print("=" * 120)

    for name, func in [("Multi (V121>Z2>FF>PB)", signal_multi),
                       ("Multi scored", signal_multi_scored)]:
        print(f"\n  {name}:")
        for regime in ['none', 'equity_curve', 'combined']:
            for topn in [1, 2, 3]:
                r = backtest_v130(func, hold_days=1, top_n=topn,
                                  regime_mode=regime, desc=f"{name} {regime} t={topn}")
                print(f"    {regime:15s} top_n={topn}: Ann={r['ann']:+8.1f}% | "
                      f"WR={r['wr']:5.1f}% | N={r['n']:4d} | MDD={r['mdd']:6.1f}%")

    # ================================================================
    # SECTION 5: WALK-FORWARD FOR TOP CONFIGURATIONS
    # ================================================================
    print("\n" + "=" * 120)
    print("  SECTION 5: WALK-FORWARD FOR TOP CONFIGURATIONS")
    print("=" * 120)

    wf_configs = [
        ("V121 baseline", signal_v121, 1, 1, 'none'),
        ("V121 + equity_curve", signal_v121, 1, 1, 'equity_curve'),
        ("V121 + combined regime", signal_v121, 1, 1, 'combined'),
        ("Multi scored + none", signal_multi_scored, 1, 1, 'none'),
        ("Multi scored + eq_curve", signal_multi_scored, 1, 1, 'equity_curve'),
        ("Multi scored + combined", signal_multi_scored, 1, 1, 'combined'),
        ("V121 top_n=2 + eq_curve", signal_v121, 1, 2, 'equity_curve'),
        ("Multi top_n=2 + eq_curve", signal_multi, 1, 2, 'equity_curve'),
    ]

    for name, func, hold, topn, regime in wf_configs:
        wf = walk_forward(func, hold_days=hold, top_n=topn, regime_mode=regime, desc=name)
        wf_str = " | ".join([f"{yr}:{ann:+.0f}%" for yr, ann in sorted(wf.items())])
        positive = sum(1 for v in wf.values() if v > 0)
        avg_wf = np.mean(list(wf.values())) if wf else 0
        print(f"  {name:40s} | {positive}/6 | Avg={avg_wf:>+7.0f}% | {wf_str}")

    # ================================================================
    # SECTION 6: SIGNAL TYPE ANALYSIS FOR MULTI
    # ================================================================
    print("\n" + "=" * 120)
    print("  SECTION 6: SIGNAL TYPE BREAKDOWN (multi-scored)")
    print("=" * 120)

    r_multi = backtest_v130(signal_multi_scored, hold_days=1, top_n=1,
                            regime_mode='none', desc="Multi scored")
    if r_multi['trades']:
        sig_types = {}
        for t in r_multi['trades']:
            st = t.get('signal_type', 'unknown')
            if st not in sig_types:
                sig_types[st] = {'count': 0, 'wins': 0, 'total_pnl': 0}
            sig_types[st]['count'] += 1
            if t['pnl_pct'] > 0:
                sig_types[st]['wins'] += 1
            sig_types[st]['total_pnl'] += t['pnl_pct']

        print(f"  {'Signal Type':20s} | {'Count':>5} | {'WR':>6} | {'AvgPnL':>8} | {'TotalPnL':>10}")
        print("-" * 70)
        for st, data in sorted(sig_types.items(), key=lambda x: -x[1]['count']):
            wr = data['wins'] / data['count'] * 100 if data['count'] > 0 else 0
            avg_pnl = data['total_pnl'] / data['count'] if data['count'] > 0 else 0
            print(f"  {st:20s} | {data['count']:>5} | {wr:>5.1f}% | {avg_pnl:>+7.3f}% | {data['total_pnl']:>+9.1f}%")

    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "=" * 120)
    print("  SUMMARY")
    print("=" * 120)

    # Collect all results from sections
    all_results = {}

    # Section 1 results
    for regime in regimes:
        r = backtest_v130(signal_v121, hold_days=1, top_n=1, regime_mode=regime)
        all_results[f"V121 + {regime}"] = r

    # Section 2 results
    for name, func in signals:
        r = backtest_v130(func, hold_days=1, top_n=1)
        all_results[name] = r

    # Best multi-signal + regime combos
    for func in [signal_multi, signal_multi_scored]:
        for regime in ['none', 'equity_curve']:
            for topn in [1, 2]:
                r = backtest_v130(func, hold_days=1, top_n=topn, regime_mode=regime)
                name = f"{func.__name__} r={regime} t={topn}"
                all_results[name] = r

    sorted_results = sorted(all_results.items(), key=lambda x: -x[1]['ann'])
    print(f"\n  Top 15 by annual return:")
    for i, (name, r) in enumerate(sorted_results[:15]):
        print(f"  #{i+1}: {name:55s} | Ann={r['ann']:+8.1f}% | WR={r['wr']:5.1f}% | "
              f"N={r['n']:4d} | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f}")

    print(f"\n  Top 15 by Sharpe ratio:")
    sorted_by_sharpe = sorted(all_results.items(), key=lambda x: -x[1]['sharpe'])
    for i, (name, r) in enumerate(sorted_by_sharpe[:15]):
        print(f"  #{i+1}: {name:55s} | Ann={r['ann']:+8.1f}% | WR={r['wr']:5.1f}% | "
              f"N={r['n']:4d} | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f}")

    print(f"\n  Top 10 by risk-adjusted (Sortino):")
    sorted_by_sortino = sorted(all_results.items(), key=lambda x: -x[1]['sortino'])
    for i, (name, r) in enumerate(sorted_by_sortino[:10]):
        print(f"  #{i+1}: {name:55s} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | "
              f"Sh={r['sharpe']:4.2f} | So={r['sortino']:5.1f}")

    print(f"\n  Total elapsed: {time.time()-t_start:.0f}s")
    print("=" * 120)


if __name__ == '__main__':
    main()
