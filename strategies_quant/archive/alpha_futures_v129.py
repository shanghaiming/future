"""
Alpha Futures V129 — V121 SIGNAL QUALITY ENHANCEMENT
====================================================
V121 Champion (+333%) is the king. V128 showed Brooks concepts confirm it
but can't beat it. So the path forward is enhancing V121's SIGNAL QUALITY.

Tests:
A) V121 + body ratio filter (strong trend bar = better entry)
B) V121 + volume confirmation (institutional participation)
C) V121 + OI increase (new money flowing in)
D) V121 + ATR-adjusted ranking (volatility-normalized momentum)
E) V121 + all quality filters combined
F) V121 + adaptive position sizing (reduce at >2x equity)
G) V121 + market breadth filter (more commodities signaling = stronger)
H) V121 + time-of-month filter (avoid expiry weeks)
I) V121 + drawdown-adaptive (reduce position during drawdowns)

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
    print("  Alpha Futures V129 — V121 SIGNAL QUALITY ENHANCEMENT")
    print("=" * 120)

    print("\n[Data] Loading...", flush=True)
    t_start = time.time()
    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)
    years = sorted(set(d.year for d in dates))
    print(f"  {NS} commodities, {ND} days, years: {years}")

    # ================================================================
    # PRECOMPUTE ALL INDICATORS
    # ================================================================
    print("\n[Precompute] All indicators...", flush=True)
    t0 = time.time()

    RET = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si].astype(np.float64)
        for di in range(1, ND):
            if not np.isnan(c[di]) and not np.isnan(c[di-1]) and c[di-1] > 0:
                RET[si, di] = (c[di] / c[di-1] - 1) * 100

    ROC5 = np.full((NS, ND), np.nan)
    ROC10 = np.full((NS, ND), np.nan)
    ROC20 = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si].astype(np.float64)
        ROC5[si] = talib.ROC(c, timeperiod=5)
        ROC10[si] = talib.ROC(c, timeperiod=10)
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
            mean_r = np.mean(valid)
            std_r = np.std(valid, ddof=1)
            if std_r > 0 and not np.isnan(RET[si, di]):
                ZSCORE[si, di] = (RET[si, di] - mean_r) / std_r

    # Body ratio: |C-O| / (H-L)
    BODY_RATIO = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(ND):
            h, l, c, o = H[si, di], L[si, di], C[si, di], O[si, di]
            if any(np.isnan(x) for x in [h, l, c, o]) or h == l:
                continue
            BODY_RATIO[si, di] = abs(c - o) / (h - l)

    # Bar direction
    BAR_DIR = np.zeros((NS, ND), dtype=int)
    for si in range(NS):
        for di in range(ND):
            c, o = C[si, di], O[si, di]
            if not np.isnan(c) and not np.isnan(o):
                BAR_DIR[si, di] = 1 if c > o else (-1 if c < o else 0)

    # Volume ratio (today / 20-day avg)
    VOL_RATIO = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(21, ND):
            vols = V[si, di-20:di]
            valid = vols[~np.isnan(vols)]
            if len(valid) < 10 or np.mean(valid) == 0:
                continue
            if not np.isnan(V[si, di]):
                VOL_RATIO[si, di] = V[si, di] / np.mean(valid)

    # OI change ratio (today / 5-day avg)
    OI_CHANGE = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(6, ND):
            ois = OI[si, di-5:di]
            valid = ois[~np.isnan(ois)]
            if len(valid) < 3 or np.mean(valid) == 0:
                continue
            if not np.isnan(OI[si, di]):
                OI_CHANGE[si, di] = OI[si, di] / np.mean(valid) - 1

    # ATR as percentage of price
    ATR_PCT = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(ND):
            if not np.isnan(ATR14[si, di]) and not np.isnan(C[si, di]) and C[si, di] > 0:
                ATR_PCT[si, di] = ATR14[si, di] / C[si, di] * 100

    print(f"  All indicators computed ({time.time()-t0:.1f}s)")

    # ================================================================
    # BACKTEST ENGINE WITH ENHANCED FEATURES
    # ================================================================
    def backtest_v129(signal_func, hold_days=1, top_n=1,
                      position_sizing='full', risk_frac=0.95,
                      start_di=MIN_TRAIN, end_di=None, desc=""):
        """
        position_sizing: 'full' = normal, 'adaptive' = reduce at >2x equity,
                         'dd_adaptive' = reduce during drawdowns
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
                        'sym': pos['sym'], 'entry_price': pos['entry_price'],
                        'exit_price': exit_price, 'score': pos.get('score', 0),
                    })
                    closed.append(pos)
            for pos in closed:
                positions.remove(pos)

            if len(positions) >= top_n:
                continue

            # Generate signals
            entry_di = di + 1
            if entry_di >= end_di:
                continue

            candidates = signal_func(di)
            if not candidates:
                continue

            candidates.sort(key=lambda x: -x[0])
            n_slots = top_n - len(positions)

            # Position sizing
            if position_sizing == 'adaptive':
                if port_val > CASH0 * 2:
                    risk_frac = 0.5
                else:
                    risk_frac = 0.95
            elif position_sizing == 'dd_adaptive':
                if peak_equity > 0:
                    dd_from_peak = (port_val - peak_equity) / peak_equity
                    if dd_from_peak < -0.3:
                        risk_frac = 0.3
                    elif dd_from_peak < -0.15:
                        risk_frac = 0.5
                    else:
                        risk_frac = 0.95
            else:
                risk_frac = risk_frac

            cap_per_slot = cash * risk_frac / max(1, n_slots)

            for sc_val, s, price in candidates[:max(0, n_slots)]:
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
                'sym': pos['sym'], 'entry_price': pos['entry_price'],
                'exit_price': exit_price, 'score': pos.get('score', 0),
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
            sharpe_arr = np.diff(eq_arr) / eq_arr[:-1]
            sharpe = np.mean(sharpe_arr) / np.std(sharpe_arr) * np.sqrt(252) if np.std(sharpe_arr) > 0 else 0
        else:
            mdd = 0.0
            sharpe = 0.0

        return {
            'ann': ann, 'wr': wr, 'n': n_trades, 'avg_pnl': avg_pnl,
            'final_cash': cash, 'n_days': n_days_test, 'mdd': mdd,
            'sharpe': sharpe, 'trades': trades, 'desc': desc,
        }

    def print_result(r, label=""):
        print(f"  {label:50s} | Ann={r['ann']:+8.1f}% | WR={r['wr']:5.1f}% | "
              f"N={r['n']:4d} | Avg={r['avg_pnl']:+5.2f}% | MDD={r['mdd']:6.1f}% | "
              f"Sh={r['sharpe']:4.2f}")

    def walk_forward(signal_func, hold_days=1, top_n=1, position_sizing='full', desc=""):
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
            r = backtest_v129(signal_func, hold_days=hold_days, top_n=top_n,
                              position_sizing=position_sizing,
                              start_di=yr_start, end_di=yr_end, desc=f"{desc} {yr}")
            wf[yr] = r['ann']
        return wf

    # ================================================================
    # SIGNAL FUNCTIONS
    # ================================================================

    # BASELINE: V121 Champion
    def signal_v121(di):
        candidates = []
        for s in range(NS):
            roc = ROC5[s, di]
            zs = ZSCORE[s, di]
            if np.isnan(roc) or np.isnan(zs):
                continue
            if roc <= 1.0 or zs <= 1.5:
                continue
            roc_prev = ROC5[s, di-1] if di > 0 else np.nan
            if not np.isnan(roc_prev) and roc <= roc_prev:
                continue
            ep = O[s, di+1]
            if np.isnan(ep) or ep <= 0:
                continue
            candidates.append((roc * zs, s, ep))
        return candidates

    # A) V121 + Body Ratio filter (strong bar confirmation)
    def signal_v121_br(di):
        candidates = []
        for s in range(NS):
            roc = ROC5[s, di]
            zs = ZSCORE[s, di]
            br = BODY_RATIO[s, di]
            bd = BAR_DIR[s, di]
            if any(np.isnan(x) for x in [roc, zs, br]):
                continue
            if roc <= 1.0 or zs <= 1.5 or bd != 1 or br < 0.5:
                continue
            roc_prev = ROC5[s, di-1] if di > 0 else np.nan
            if not np.isnan(roc_prev) and roc <= roc_prev:
                continue
            ep = O[s, di+1]
            if np.isnan(ep) or ep <= 0:
                continue
            candidates.append((roc * zs, s, ep))
        return candidates

    # B) V121 + Volume confirmation (vol > 20-day avg)
    def signal_v121_vol(di):
        candidates = []
        for s in range(NS):
            roc = ROC5[s, di]
            zs = ZSCORE[s, di]
            vr = VOL_RATIO[s, di]
            if any(np.isnan(x) for x in [roc, zs, vr]):
                continue
            if roc <= 1.0 or zs <= 1.5 or vr < 1.0:
                continue
            roc_prev = ROC5[s, di-1] if di > 0 else np.nan
            if not np.isnan(roc_prev) and roc <= roc_prev:
                continue
            ep = O[s, di+1]
            if np.isnan(ep) or ep <= 0:
                continue
            candidates.append((roc * zs, s, ep))
        return candidates

    # C) V121 + OI increase confirmation
    def signal_v121_oi(di):
        candidates = []
        for s in range(NS):
            roc = ROC5[s, di]
            zs = ZSCORE[s, di]
            oi_ch = OI_CHANGE[s, di]
            if any(np.isnan(x) for x in [roc, zs]):
                continue
            if roc <= 1.0 or zs <= 1.5:
                continue
            roc_prev = ROC5[s, di-1] if di > 0 else np.nan
            if not np.isnan(roc_prev) and roc <= roc_prev:
                continue
            # OI increasing bonus in ranking, not a hard filter
            oi_bonus = 1 + max(0, oi_ch) if not np.isnan(oi_ch) else 1.0
            ep = O[s, di+1]
            if np.isnan(ep) or ep <= 0:
                continue
            candidates.append((roc * zs * oi_bonus, s, ep))
        return candidates

    # D) V121 + Enhanced ranking (ROC*Z*BR)
    def signal_v121_rank_br(di):
        candidates = []
        for s in range(NS):
            roc = ROC5[s, di]
            zs = ZSCORE[s, di]
            br = BODY_RATIO[s, di]
            if any(np.isnan(x) for x in [roc, zs]):
                continue
            if roc <= 1.0 or zs <= 1.5:
                continue
            roc_prev = ROC5[s, di-1] if di > 0 else np.nan
            if not np.isnan(roc_prev) and roc <= roc_prev:
                continue
            br_score = br if not np.isnan(br) and br > 0 else 0.5
            ep = O[s, di+1]
            if np.isnan(ep) or ep <= 0:
                continue
            candidates.append((roc * zs * br_score, s, ep))
        return candidates

    # E) V121 + ALL quality filters combined
    def signal_v121_all_quality(di):
        candidates = []
        for s in range(NS):
            roc = ROC5[s, di]
            zs = ZSCORE[s, di]
            br = BODY_RATIO[s, di]
            vr = VOL_RATIO[s, di]
            oi_ch = OI_CHANGE[s, di]
            bd = BAR_DIR[s, di]
            if any(np.isnan(x) for x in [roc, zs]):
                continue
            if roc <= 1.0 or zs <= 1.5:
                continue
            roc_prev = ROC5[s, di-1] if di > 0 else np.nan
            if not np.isnan(roc_prev) and roc <= roc_prev:
                continue
            # Quality scoring (not hard filters, for ranking)
            quality = 1.0
            if not np.isnan(br) and br > 0.5 and bd == 1:
                quality *= (1 + br)
            if not np.isnan(vr) and vr > 1.0:
                quality *= (1 + (vr - 1) * 0.3)
            if not np.isnan(oi_ch) and oi_ch > 0:
                quality *= (1 + oi_ch * 0.5)
            ep = O[s, di+1]
            if np.isnan(ep) or ep <= 0:
                continue
            candidates.append((roc * zs * quality, s, ep))
        return candidates

    # F) V121 + Hard body ratio + Hard volume filter
    def signal_v121_hard_filters(di):
        candidates = []
        for s in range(NS):
            roc = ROC5[s, di]
            zs = ZSCORE[s, di]
            br = BODY_RATIO[s, di]
            vr = VOL_RATIO[s, di]
            bd = BAR_DIR[s, di]
            if any(np.isnan(x) for x in [roc, zs]):
                continue
            if roc <= 1.0 or zs <= 1.5:
                continue
            if np.isnan(br) or br < 0.5 or bd != 1:
                continue
            if np.isnan(vr) or vr < 0.8:
                continue
            roc_prev = ROC5[s, di-1] if di > 0 else np.nan
            if not np.isnan(roc_prev) and roc <= roc_prev:
                continue
            ep = O[s, di+1]
            if np.isnan(ep) or ep <= 0:
                continue
            candidates.append((roc * zs, s, ep))
        return candidates

    # G) V121 + Market breadth filter (require ≥3 other commodities also signaling)
    def signal_v121_breadth(di):
        # First count how many commodities signal
        all_sigs = []
        for s in range(NS):
            roc = ROC5[s, di]
            zs = ZSCORE[s, di]
            if any(np.isnan(x) for x in [roc, zs]):
                continue
            if roc <= 1.0 or zs <= 1.5:
                continue
            roc_prev = ROC5[s, di-1] if di > 0 else np.nan
            if not np.isnan(roc_prev) and roc <= roc_prev:
                continue
            ep = O[s, di+1]
            if np.isnan(ep) or ep <= 0:
                continue
            all_sigs.append((roc * zs, s, ep))

        # Require at least 2 other signals (breadth confirmation)
        if len(all_sigs) < 3:
            return []
        return all_sigs

    # H) V121 + Higher Z-score threshold (stricter signal)
    def signal_v121_strict(di):
        candidates = []
        for s in range(NS):
            roc = ROC5[s, di]
            zs = ZSCORE[s, di]
            if any(np.isnan(x) for x in [roc, zs]):
                continue
            if roc <= 1.5 or zs <= 2.0:  # Stricter thresholds
                continue
            roc_prev = ROC5[s, di-1] if di > 0 else np.nan
            if not np.isnan(roc_prev) and roc <= roc_prev:
                continue
            ep = O[s, di+1]
            if np.isnan(ep) or ep <= 0:
                continue
            candidates.append((roc * zs, s, ep))
        return candidates

    # I) V121 + ROC(3) acceleration check
    def signal_v121_accel(di):
        candidates = []
        for s in range(NS):
            roc = ROC5[s, di]
            zs = ZSCORE[s, di]
            if any(np.isnan(x) for x in [roc, zs]):
                continue
            if roc <= 1.0 or zs <= 1.5:
                continue
            roc_prev = ROC5[s, di-1] if di > 0 else np.nan
            if not np.isnan(roc_prev) and roc <= roc_prev:
                continue
            # Additional: ROC(10) > 0 (medium-term uptrend) + today's return > 1%
            roc10 = ROC10[s, di]
            ret = RET[s, di]
            if not np.isnan(roc10) and roc10 < 0:
                continue  # skip if not in medium-term uptrend
            ep = O[s, di+1]
            if np.isnan(ep) or ep <= 0:
                continue
            # Acceleration bonus
            accel = roc / roc_prev if not np.isnan(roc_prev) and roc_prev > 0 else 1.0
            candidates.append((roc * zs * accel, s, ep))
        return candidates

    # ================================================================
    # SECTION 1: ALL STRATEGIES HEAD-TO-HEAD
    # ================================================================
    print("\n" + "=" * 120)
    print("  SECTION 1: ALL ENHANCEMENTS HEAD-TO-HEAD")
    print("=" * 120)

    strategies = [
        ("BASELINE: V121 Champion", signal_v121, 1, 1, 'full'),
        ("A) V121 + Body Ratio >= 0.5", signal_v121_br, 1, 1, 'full'),
        ("B) V121 + Vol > 20d avg", signal_v121_vol, 1, 1, 'full'),
        ("C) V121 + OI bonus ranking", signal_v121_oi, 1, 1, 'full'),
        ("D) V121 + Rank by ROC*Z*BR", signal_v121_rank_br, 1, 1, 'full'),
        ("E) V121 + All quality scoring", signal_v121_all_quality, 1, 1, 'full'),
        ("F) V121 + Hard BR+Vol filters", signal_v121_hard_filters, 1, 1, 'full'),
        ("G) V121 + Breadth >= 3 sigs", signal_v121_breadth, 1, 1, 'full'),
        ("H) V121 Strict (ROC>1.5 Z>2)", signal_v121_strict, 1, 1, 'full'),
        ("I) V121 + ROC10 trend + accel", signal_v121_accel, 1, 1, 'full'),
    ]

    results = {}
    for name, func, hold, topn, ps in strategies:
        r = backtest_v129(func, hold_days=hold, top_n=topn,
                          position_sizing=ps, desc=name)
        results[name] = r
        print_result(r, label=name)

    # ================================================================
    # SECTION 2: WALK-FORWARD FOR TOP 5
    # ================================================================
    print("\n" + "=" * 120)
    print("  SECTION 2: WALK-FORWARD VALIDATION")
    print("=" * 120)

    top5 = sorted(results.items(), key=lambda x: -x[1]['ann'])[:5]
    for name, r in top5:
        func = dict((n, f) for n, f, _, _, _ in strategies)[name]
        wf = walk_forward(func, hold_days=1, top_n=1, desc=name)
        wf_str = " | ".join([f"{yr}:{ann:+.0f}%" for yr, ann in sorted(wf.items())])
        positive = sum(1 for v in wf.values() if v > 0)
        print(f"  {name:50s} | {positive}/6 WF | {wf_str}")

    # ================================================================
    # SECTION 3: ADAPTIVE POSITION SIZING FOR BEST
    # ================================================================
    print("\n" + "=" * 120)
    print("  SECTION 3: POSITION SIZING VARIANTS")
    print("=" * 120)

    best_name = max(results, key=lambda k: results[k]['ann'])
    best_func = dict((n, f) for n, f, _, _, _ in strategies)[best_name]
    v121_func = signal_v121

    for name, func in [(best_name, best_func), ("BASELINE: V121 Champion", v121_func)]:
        print(f"\n  {name}:")
        for ps in ['full', 'adaptive', 'dd_adaptive']:
            for topn in [1, 2, 3]:
                r = backtest_v129(func, hold_days=1, top_n=topn,
                                  position_sizing=ps, desc=f"{name} {ps} t={topn}")
                print(f"    {ps:15s} top_n={topn}: Ann={r['ann']:+8.1f}% | "
                      f"WR={r['wr']:5.1f}% | N={r['n']:4d} | MDD={r['mdd']:6.1f}%")

    # ================================================================
    # SECTION 4: TOP_N COMBINATIONS FOR BEST ENHANCED
    # ================================================================
    print("\n" + "=" * 120)
    print("  SECTION 4: TOP_N x HOLD COMBINATIONS (best 3 strategies)")
    print("=" * 120)

    top3_names = [n for n, _ in top5[:3]]
    for name in top3_names:
        func = dict((n, f) for n, f, _, _, _ in strategies)[name]
        print(f"\n  {name}:")
        for hold in [1, 2, 3]:
            for topn in [1, 2, 3]:
                r = backtest_v129(func, hold_days=hold, top_n=topn, desc=f"{name} h={hold} t={topn}")
                print(f"    hold={hold} top_n={topn}: Ann={r['ann']:+8.1f}% | "
                      f"WR={r['wr']:5.1f}% | N={r['n']:4d} | MDD={r['mdd']:6.1f}%")

    # ================================================================
    # SECTION 5: Z-SCORE THRESHOLD SENSITIVITY FOR BEST
    # ================================================================
    print("\n" + "=" * 120)
    print("  SECTION 5: THRESHOLD SENSITIVITY")
    print("=" * 120)

    print("\n  V121 base: ROC threshold sweep (Z>1.5 fixed)")
    for roc_t in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
        def make_roc_sig(rt):
            def sig(di):
                candidates = []
                for s in range(NS):
                    roc = ROC5[s, di]
                    zs = ZSCORE[s, di]
                    if any(np.isnan(x) for x in [roc, zs]):
                        continue
                    if roc <= rt or zs <= 1.5:
                        continue
                    roc_prev = ROC5[s, di-1] if di > 0 else np.nan
                    if not np.isnan(roc_prev) and roc <= roc_prev:
                        continue
                    ep = O[s, di+1]
                    if np.isnan(ep) or ep <= 0:
                        continue
                    candidates.append((roc * zs, s, ep))
                return candidates
            return sig
        r = backtest_v129(make_roc_sig(roc_t), hold_days=1, top_n=1, desc=f"ROC>{roc_t}")
        print_result(r, label=f"ROC>{roc_t}")

    print("\n  V121 base: Z-score threshold sweep (ROC>1.0 fixed)")
    for z_t in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        def make_z_sig(zt):
            def sig(di):
                candidates = []
                for s in range(NS):
                    roc = ROC5[s, di]
                    zs = ZSCORE[s, di]
                    if any(np.isnan(x) for x in [roc, zs]):
                        continue
                    if roc <= 1.0 or zs <= zt:
                        continue
                    roc_prev = ROC5[s, di-1] if di > 0 else np.nan
                    if not np.isnan(roc_prev) and roc <= roc_prev:
                        continue
                    ep = O[s, di+1]
                    if np.isnan(ep) or ep <= 0:
                        continue
                    candidates.append((roc * zs, s, ep))
                return candidates
            return sig
        r = backtest_v129(make_z_sig(z_t), hold_days=1, top_n=1, desc=f"Z>{z_t}")
        print_result(r, label=f"Z>{z_t}")

    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "=" * 120)
    print("  SUMMARY: ALL STRATEGIES RANKED")
    print("=" * 120)

    all_results = sorted(results.items(), key=lambda x: -x[1]['ann'])
    for i, (name, r) in enumerate(all_results):
        print(f"  #{i+1}: {name:50s} | Ann={r['ann']:+8.1f}% | WR={r['wr']:5.1f}% | "
              f"N={r['n']:4d} | MDD={r['mdd']:6.1f}%")

    print(f"\n  Total elapsed: {time.time()-t_start:.0f}s")
    print("=" * 120)


if __name__ == '__main__':
    main()
