"""
Alpha Futures V132 — VOLATILITY-ADAPTIVE LOOKBACK + OVERNIGHT-INTRADAY + OI RANKING
====================================================================================
Based on research findings from 1,458 papers:
1) Volatility-adaptive lookback: high vol → short lookback, low vol → long
2) Overnight-Intraday decomposition (from V126: +260%, WF avg +323.7%)
3) OI change as ranking factor (from V124: #1 ML feature)
4) Sortino-ratio ranking (research: more robust than raw ROC)

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
    print("  Alpha Futures V132 — VOLATILITY-ADAPTIVE LOOKBACK + OV/ID + OI RANKING")
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

    # Daily returns
    RET = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si].astype(np.float64)
        for di in range(1, ND):
            if not np.isnan(c[di]) and not np.isnan(c[di-1]) and c[di-1] > 0:
                RET[si, di] = (c[di] / c[di-1] - 1) * 100

    # ROC at multiple lookbacks
    ROC = {}
    for lb in [3, 5, 10, 20, 60]:
        ROC[lb] = np.full((NS, ND), np.nan)
        for si in range(NS):
            c = C[si].astype(np.float64)
            ROC[lb][si] = talib.ROC(c, timeperiod=lb)

    # ATR(14) and ATR as % of price
    ATR14 = np.full((NS, ND), np.nan)
    for si in range(NS):
        ATR14[si] = talib.ATR(H[si].astype(np.float64), L[si].astype(np.float64),
                               C[si].astype(np.float64), timeperiod=14)

    # Realized volatility (20-day rolling std of returns)
    RVOL20 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(21, ND):
            rets = RET[si, di-20:di]
            valid = rets[~np.isnan(rets)]
            if len(valid) >= 10:
                RVOL20[si, di] = np.std(valid, ddof=1)

    # Z-score of daily returns (20-day rolling)
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

    # Overnight gap: (open_t - close_t-1) / close_t-1 * 100
    OV_GAP = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(1, ND):
            o_t = O[si, di]; c_prev = C[si, di-1]
            if not np.isnan(o_t) and not np.isnan(c_prev) and c_prev > 0:
                OV_GAP[si, di] = (o_t - c_prev) / c_prev * 100

    # Intraday return: (close_t - open_t) / open_t * 100
    ID_RET = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(ND):
            c = C[si, di]; o = O[si, di]
            if not np.isnan(c) and not np.isnan(o) and o > 0:
                ID_RET[si, di] = (c - o) / o * 100

    # OI change ratio (today / 5-day avg - 1)
    OI_CHANGE = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(6, ND):
            ois = OI[si, di-5:di]
            valid = ois[~np.isnan(ois)]
            if len(valid) < 3 or np.mean(valid) == 0:
                continue
            if not np.isnan(OI[si, di]):
                OI_CHANGE[si, di] = OI[si, di] / np.mean(valid) - 1

    # Volatility regime: percentile of current RVOL20 vs past 60 days
    RVOL_PCT = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(81, ND):
            hist = RVOL20[si, di-60:di]
            valid = hist[~np.isnan(hist)]
            if len(valid) < 20:
                continue
            cur = RVOL20[si, di]
            if not np.isnan(cur):
                RVOL_PCT[si, di] = np.sum(valid < cur) / len(valid) * 100

    # Adaptive lookback: based on volatility regime
    # High vol (>75th pctile) → short lookback (3, 5)
    # Medium vol (25-75th) → medium lookback (5, 10)
    # Low vol (<25th) → long lookback (10, 20)
    ADAPTIVE_ROC = np.full((NS, ND), np.nan)
    ADAPTIVE_LB = np.full((NS, ND), 5)  # default lookback
    for si in range(NS):
        for di in range(81, ND):
            pct = RVOL_PCT[si, di]
            if np.isnan(pct):
                continue
            if pct > 75:
                lb = 3  # high vol → short lookback
            elif pct > 50:
                lb = 5  # medium-high vol
            elif pct > 25:
                lb = 10  # medium-low vol
            else:
                lb = 20  # low vol → long lookback
            ADAPTIVE_LB[si, di] = lb
            ADAPTIVE_ROC[si, di] = ROC[lb][si, di] if not np.isnan(ROC[lb][si, di]) else np.nan

    # Sortino-like ranking: ROC / downside_deviation
    DOWNSIDE_DEV = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(21, ND):
            rets = RET[si, di-20:di]
            valid = rets[~np.isnan(rets)]
            neg = valid[valid < 0]
            if len(neg) >= 5:
                DOWNSIDE_DEV[si, di] = np.std(neg, ddof=1)

    print(f"  All indicators computed ({time.time()-t0:.1f}s)")

    # ================================================================
    # BACKTEST ENGINE
    # ================================================================
    def backtest(signal_func, hold_days=1, top_n=1, start_di=MIN_TRAIN, end_di=None, desc=""):
        if end_di is None:
            end_di = ND
        cash = float(CASH0)
        positions = []
        trades = []
        daily_equity = []
        for di in range(start_di, end_di - 1):
            port_val = cash
            for pos in positions:
                cp = C[pos['si'], di]
                if not np.isnan(cp) and cp > 0:
                    mult = MULT.get(pos['sym'], DEF_MULT)
                    port_val += cp * mult * pos['lots'] - cp * mult * abs(pos['lots']) * COMM
            daily_equity.append(port_val)
            closed = []
            for pos in positions:
                if di - pos['entry_di'] >= pos['hold_days']:
                    exit_price = C[pos['si'], di]
                    if np.isnan(exit_price) or exit_price <= 0:
                        exit_price = pos['entry_price']
                    mult = MULT.get(pos['sym'], DEF_MULT)
                    pnl = (exit_price - pos['entry_price']) * mult * pos['lots']
                    invested = pos['entry_price'] * mult * abs(pos['lots'])
                    pnl_pct = pnl / invested * 100 if invested > 0 else 0
                    cash += exit_price * mult * abs(pos['lots']) * (1 - COMM)
                    trades.append({'pnl_pct': pnl_pct, 'signal': pos.get('signal', '')})
                    closed.append(pos)
            for pos in closed:
                positions.remove(pos)
            if len(positions) >= top_n:
                continue
            entry_di = di + 1
            if entry_di >= end_di:
                continue
            candidates = signal_func(di, entry_di)
            if not candidates:
                continue
            candidates.sort(key=lambda x: -x[0])
            n_slots = top_n - len(positions)
            cap_per_slot = cash / max(1, n_slots)
            for item in candidates[:max(0, n_slots)]:
                if len(item) == 3:
                    sc, s, price = item
                    sig = ''
                else:
                    sc, s, price, sig = item
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
                    'lots': contracts, 'dir': 1, 'sym': sym, 'hold_days': hold_days,
                    'signal': sig,
                })
        for pos in positions:
            ae = end_di - 1
            exit_price = C[pos['si'], min(ae, ND-1)]
            if np.isnan(exit_price) or exit_price <= 0:
                exit_price = pos['entry_price']
            mult = MULT.get(pos['sym'], DEF_MULT)
            cash += exit_price * mult * abs(pos['lots']) * (1 - COMM)
        n_days_test = end_di - start_di
        ann = annual_return(cash, CASH0, n_days_test)
        wr = np.mean([1 if t['pnl_pct'] > 0 else 0 for t in trades]) * 100 if trades else 0
        n_trades = len(trades)
        avg_pnl = np.mean([t['pnl_pct'] for t in trades]) if trades else 0
        if daily_equity:
            eq = np.array(daily_equity)
            pk = np.maximum.accumulate(eq)
            mdd = np.min((eq - pk) / pk * 100)
            rets = np.diff(eq) / eq[:-1]
            sharpe = np.mean(rets) / np.std(rets) * np.sqrt(252) if np.std(rets) > 0 else 0
        else:
            mdd = 0; sharpe = 0
        return {'ann': ann, 'wr': wr, 'n': n_trades, 'avg_pnl': avg_pnl,
                'mdd': mdd, 'sharpe': sharpe, 'desc': desc, 'trades': trades}

    def pr(r, label=""):
        print(f"  {label:60s} | Ann={r['ann']:+8.1f}% | WR={r['wr']:5.1f}% | "
              f"N={r['n']:4d} | Avg={r['avg_pnl']:+5.2f}% | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f}")

    def wf(signal_func, hold=1, topn=1, desc=""):
        res = {}
        for yr in [2020, 2021, 2022, 2023, 2024, 2025]:
            ys = ye = None
            for di in range(ND):
                if dates[di].year == yr and ys is None: ys = di
                if dates[di].year == yr: ye = di + 1
            if ys is None: continue
            r = backtest(signal_func, hold_days=hold, top_n=topn, start_di=ys, end_di=ye)
            res[yr] = r['ann']
        return res

    # ================================================================
    # SIGNAL DEFINITIONS
    # ================================================================

    # BASELINE: V121 Champion
    def signal_v121(di, edi):
        cands = []
        for s in range(NS):
            roc = ROC[5][s, di]; zs = ZSCORE[s, di]
            if np.isnan(roc) or np.isnan(zs) or roc <= 1.0 or zs <= 1.5:
                continue
            rp = ROC[5][s, di-1] if di > 0 else np.nan
            if not np.isnan(rp) and roc <= rp:
                continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0:
                continue
            cands.append((roc * zs, s, ep, 'v121'))
        return cands

    # A) Volatility-adaptive lookback (research: +66% Sharpe improvement)
    def signal_adaptive_lb(di, edi):
        cands = []
        for s in range(NS):
            aroc = ADAPTIVE_ROC[s, di]; zs = ZSCORE[s, di]
            if np.isnan(aroc) or np.isnan(zs) or aroc <= 1.0 or zs <= 1.5:
                continue
            # ROC improving: compare adaptive ROC with previous
            lb = int(ADAPTIVE_LB[s, di])
            rp = ROC[lb][s, di-1] if di > 0 else np.nan
            if not np.isnan(rp) and aroc <= rp:
                continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0:
                continue
            cands.append((aroc * zs, s, ep, f'adaptive_lb{lb}'))
        return cands

    # B) Overnight-Intraday decomposition (from V126: WF avg +323.7%)
    def signal_ov_id(di, edi):
        cands = []
        for s in range(NS):
            ov = OV_GAP[s, di]; id_r = ID_RET[s, di]
            roc5 = ROC[5][s, di]; zs = ZSCORE[s, di]
            if any(np.isnan(x) for x in [ov, id_r, roc5, zs]):
                continue
            # Both overnight and intraday must be positive + momentum confirmation
            if ov <= 0.3 or id_r <= 0.3 or roc5 <= 1.0:
                continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0:
                continue
            score = (ov + id_r) * roc5  # combined gap/trend strength × momentum
            cands.append((score, s, ep, 'ov_id'))
        return cands

    # B2) Overnight-Intraday + Z-score
    def signal_ov_id_z(di, edi):
        cands = []
        for s in range(NS):
            ov = OV_GAP[s, di]; id_r = ID_RET[s, di]
            roc5 = ROC[5][s, di]; zs = ZSCORE[s, di]
            if any(np.isnan(x) for x in [ov, id_r, roc5, zs]):
                continue
            if ov <= 0.2 or id_r <= 0.2 or roc5 <= 1.0 or zs <= 1.0:
                continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0:
                continue
            score = (ov + id_r) * roc5 * zs
            cands.append((score, s, ep, 'ov_id_z'))
        return cands

    # C) OI change ranking (from V124: #1 ML feature)
    def signal_oi_rank(di, edi):
        cands = []
        for s in range(NS):
            roc = ROC[5][s, di]; zs = ZSCORE[s, di]; oi_ch = OI_CHANGE[s, di]
            if np.isnan(roc) or np.isnan(zs) or roc <= 1.0 or zs <= 1.5:
                continue
            rp = ROC[5][s, di-1] if di > 0 else np.nan
            if not np.isnan(rp) and roc <= rp:
                continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0:
                continue
            # OI bonus: increasing OI = new money confirming the move
            oi_mult = 1 + max(0, oi_ch) if not np.isnan(oi_ch) else 1.0
            score = roc * zs * oi_mult
            cands.append((score, s, ep, 'oi_rank'))
        return cands

    # D) Sortino-ratio ranking (research: more robust than raw ROC)
    def signal_sortino_rank(di, edi):
        cands = []
        for s in range(NS):
            roc = ROC[5][s, di]; zs = ZSCORE[s, di]; dd = DOWNSIDE_DEV[s, di]
            if np.isnan(roc) or np.isnan(zs) or roc <= 1.0 or zs <= 1.5:
                continue
            rp = ROC[5][s, di-1] if di > 0 else np.nan
            if not np.isnan(rp) and roc <= rp:
                continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0:
                continue
            # Sortino-like: ROC / downside_dev
            sortino_proxy = roc / dd if not np.isnan(dd) and dd > 0 else roc
            score = sortino_proxy * zs
            cands.append((score, s, ep, 'sortino'))
        return cands

    # E) ADAPTIVE LB + OV/ID + OI + Z — The kitchen sink
    def signal_mega(di, edi):
        cands = []
        for s in range(NS):
            aroc = ADAPTIVE_ROC[s, di]; zs = ZSCORE[s, di]
            if np.isnan(aroc) or np.isnan(zs) or aroc <= 1.0 or zs <= 1.5:
                continue
            lb = int(ADAPTIVE_LB[s, di])
            rp = ROC[lb][s, di-1] if di > 0 else np.nan
            if not np.isnan(rp) and aroc <= rp:
                continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0:
                continue
            # Build composite score
            score = aroc * zs  # base
            # OI bonus
            oi_ch = OI_CHANGE[s, di]
            if not np.isnan(oi_ch) and oi_ch > 0:
                score *= (1 + oi_ch)
            # Overnight/intraday bonus
            ov = OV_GAP[s, di]; id_r = ID_RET[s, di]
            if not np.isnan(ov) and not np.isnan(id_r) and ov > 0 and id_r > 0:
                score *= (1 + (ov + id_r) * 0.1)
            cands.append((score, s, ep, 'mega'))
        return cands

    # F) V121 + OV/ID filter (only trade when overnight confirms)
    def signal_v121_ov_confirm(di, edi):
        cands = []
        for s in range(NS):
            roc = ROC[5][s, di]; zs = ZSCORE[s, di]
            ov = OV_GAP[s, di]
            if any(np.isnan(x) for x in [roc, zs, ov]):
                continue
            if roc <= 1.0 or zs <= 1.5 or ov <= 0:
                continue  # require positive overnight gap
            rp = ROC[5][s, di-1] if di > 0 else np.nan
            if not np.isnan(rp) and roc <= rp:
                continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0:
                continue
            cands.append((roc * zs * (1 + ov * 0.2), s, ep, 'v121_ov'))
        return cands

    # G) Adaptive LB sensitivity: test different percentile thresholds
    def make_adaptive_sig(high_pct, high_lb, med_lb, low_lb):
        def sig(di, edi):
            cands = []
            for s in range(NS):
                pct = RVOL_PCT[s, di]
                if np.isnan(pct):
                    continue
                if pct > high_pct:
                    lb = high_lb
                elif pct > 50:
                    lb = med_lb
                else:
                    lb = low_lb
                roc = ROC[lb][s, di]; zs = ZSCORE[s, di]
                if np.isnan(roc) or np.isnan(zs) or roc <= 1.0 or zs <= 1.5:
                    continue
                rp = ROC[lb][s, di-1] if di > 0 else np.nan
                if not np.isnan(rp) and roc <= rp:
                    continue
                ep = O[s, edi]
                if np.isnan(ep) or ep <= 0:
                    continue
                cands.append((roc * zs, s, ep, f'adapt_h{high_pct}_lb{lb}'))
            return cands
        return sig

    # ================================================================
    # SECTION 1: ALL STRATEGIES HEAD-TO-HEAD
    # ================================================================
    print("\n" + "=" * 120)
    print("  SECTION 1: ALL STRATEGIES HEAD-TO-HEAD")
    print("=" * 120)

    strategies = [
        ("V121 Champion (baseline)", signal_v121),
        ("A) Volatility-adaptive lookback", signal_adaptive_lb),
        ("B) Overnight-Intraday decomp", signal_ov_id),
        ("B2) OV/ID + Z-score", signal_ov_id_z),
        ("C) OI change ranking", signal_oi_rank),
        ("D) Sortino-ratio ranking", signal_sortino_rank),
        ("E) Mega (adaptive+OV/ID+OI+Z)", signal_mega),
        ("F) V121 + OV confirm filter", signal_v121_ov_confirm),
    ]

    results = {}
    for name, func in strategies:
        r = backtest(func, hold_days=1, top_n=1, desc=name)
        results[name] = r
        pr(r, label=name)

    # ================================================================
    # SECTION 2: ADAPTIVE LOOKBACK SENSITIVITY
    # ================================================================
    print("\n" + "=" * 120)
    print("  SECTION 2: ADAPTIVE LOOKBACK PARAMETER SENSITIVITY")
    print("=" * 120)

    configs = [
        ("High75 LB3/5/10", 75, 3, 5, 10),
        ("High75 LB3/5/20", 75, 3, 5, 20),
        ("High75 LB5/10/20", 75, 5, 10, 20),
        ("High66 LB3/5/10", 66, 3, 5, 10),
        ("High66 LB3/5/20", 66, 3, 5, 20),
        ("High50 LB3/5/10", 50, 3, 5, 10),
        ("High80 LB3/5/20", 80, 3, 5, 20),
        ("High80 LB5/10/20", 80, 5, 10, 20),
    ]
    for label, hp, hlb, mlb, llb in configs:
        r = backtest(make_adaptive_sig(hp, hlb, mlb, llb), desc=label)
        pr(r, label=label)

    # ================================================================
    # SECTION 3: TOP_N x HOLD for top 3
    # ================================================================
    print("\n" + "=" * 120)
    print("  SECTION 3: TOP_N x HOLD for best strategies")
    print("=" * 120)

    top3 = sorted(results.items(), key=lambda x: -x[1]['ann'])[:3]
    for name, r in top3:
        func = dict(strategies)[name]
        print(f"\n  {name}:")
        for topn in [1, 2, 3]:
            for hold in [1, 2, 3]:
                r = backtest(func, hold_days=hold, top_n=topn, desc=f"{name} t={topn} h={hold}")
                print(f"    top_n={topn} hold={hold}: Ann={r['ann']:+8.1f}% | "
                      f"WR={r['wr']:5.1f}% | N={r['n']:4d} | MDD={r['mdd']:6.1f}%")

    # ================================================================
    # SECTION 4: WALK-FORWARD
    # ================================================================
    print("\n" + "=" * 120)
    print("  SECTION 4: WALK-FORWARD")
    print("=" * 120)

    for name, func in strategies:
        w = wf(func, desc=name)
        ws = " | ".join([f"{yr}:{v:+.0f}%" for yr, v in sorted(w.items())])
        pos = sum(1 for v in w.values() if v > 0)
        avg = np.mean(list(w.values())) if w else 0
        print(f"  {name:60s} | {pos}/6 | Avg={avg:>+7.0f}% | {ws}")

    # ================================================================
    # SECTION 5: COMBINATIONS — best individual enhancements combined
    # ================================================================
    print("\n" + "=" * 120)
    print("  SECTION 5: BEST COMBINATIONS")
    print("=" * 120)

    # Adaptive LB + OI ranking
    def signal_adapt_oi(di, edi):
        cands = []
        for s in range(NS):
            aroc = ADAPTIVE_ROC[s, di]; zs = ZSCORE[s, di]; oi_ch = OI_CHANGE[s, di]
            if np.isnan(aroc) or np.isnan(zs) or aroc <= 1.0 or zs <= 1.5:
                continue
            lb = int(ADAPTIVE_LB[s, di])
            rp = ROC[lb][s, di-1] if di > 0 else np.nan
            if not np.isnan(rp) and aroc <= rp:
                continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0:
                continue
            oi_mult = 1 + max(0, oi_ch) if not np.isnan(oi_ch) else 1.0
            cands.append((aroc * zs * oi_mult, s, ep, 'adapt_oi'))
        return cands

    # OV/ID + OI + Z
    def signal_ov_id_oi(di, edi):
        cands = []
        for s in range(NS):
            ov = OV_GAP[s, di]; id_r = ID_RET[s, di]
            roc5 = ROC[5][s, di]; zs = ZSCORE[s, di]; oi_ch = OI_CHANGE[s, di]
            if any(np.isnan(x) for x in [ov, id_r, roc5, zs]):
                continue
            if roc5 <= 1.0 or zs <= 1.5:
                continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0:
                continue
            oi_mult = 1 + max(0, oi_ch) if not np.isnan(oi_ch) else 1.0
            ov_bonus = 1 + max(0, ov + id_r) * 0.05 if ov > 0 and id_r > 0 else 1.0
            rp = ROC[5][s, di-1] if di > 0 else np.nan
            if not np.isnan(rp) and roc5 <= rp:
                continue
            score = roc5 * zs * oi_mult * ov_bonus
            cands.append((score, s, ep, 'ov_oi'))
        return cands

    # V121 + OI + Sortino ranking
    def signal_v121_oi_sort(di, edi):
        cands = []
        for s in range(NS):
            roc = ROC[5][s, di]; zs = ZSCORE[s, di]; dd = DOWNSIDE_DEV[s, di]; oi_ch = OI_CHANGE[s, di]
            if np.isnan(roc) or np.isnan(zs) or roc <= 1.0 or zs <= 1.5:
                continue
            rp = ROC[5][s, di-1] if di > 0 else np.nan
            if not np.isnan(rp) and roc <= rp:
                continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0:
                continue
            oi_mult = 1 + max(0, oi_ch) if not np.isnan(oi_ch) else 1.0
            sortino_p = roc / dd if not np.isnan(dd) and dd > 0 else roc
            cands.append((sortino_p * zs * oi_mult, s, ep, 'oi_sort'))
        return cands

    combos = [
        ("Adaptive LB + OI", signal_adapt_oi),
        ("V121 + OV/ID + OI", signal_ov_id_oi),
        ("V121 + OI + Sortino", signal_v121_oi_sort),
    ]
    for name, func in combos:
        r = backtest(func, desc=name)
        pr(r, label=name)
        w = wf(func, desc=name)
        ws = " | ".join([f"{yr}:{v:+.0f}%" for yr, v in sorted(w.items())])
        pos = sum(1 for v in w.values() if v > 0)
        avg = np.mean(list(w.values())) if w else 0
        print(f"  {'WF':60s} | {pos}/6 | Avg={avg:>+7.0f}% | {ws}")

    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "=" * 120)
    print("  SUMMARY")
    print("=" * 120)

    all_res = {**results}
    for name, func in combos:
        r = backtest(func, desc=name)
        all_res[name] = r

    sorted_res = sorted(all_res.items(), key=lambda x: -x[1]['ann'])
    for i, (name, r) in enumerate(sorted_res):
        print(f"  #{i+1}: {name:60s} | Ann={r['ann']:+8.1f}% | WR={r['wr']:5.1f}% | "
              f"N={r['n']:4d} | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f}")

    print(f"\n  Total elapsed: {time.time()-t_start:.0f}s")
    print("=" * 120)


if __name__ == '__main__':
    main()
