"""
Alpha Futures V119 -- ULTIMATE SYNTHESIS: Best of V116-V118
============================================================
Combines ALL best findings from V116-V118 into the ultimate strategy.

BEST FINDINGS:
1. V116: ROC(5)>2% AND Z-score>1.5 is best single combo (+112.9%)
2. V117: Hold=1 gives +168.5%; Top 20 champion universe; Thu+Fri best; skip Monday
3. V118: Triple momentum ROC(3)>3%+ROC(5)>2%+ROC(10)>0 gives +145.5%
4. V117: Medium volatility (2-5%) commodities are best
5. V116: Hold=3 is optimal for ROC+Z-score combo

10 test configurations (A-J):
A) Triple Momentum + Champion Universe (top 20)
B) ROC+Z-score + Champion Universe (top 20)
C) ROC+Z-score + Day Filter (Thu+Fri only)
D) ROC+Z-score + Vol Filter (2-5%)
E) Triple Momentum + Z-score + Champion
F) Everything Combined (top 20 + vol + day + ROC+Z)
G) Ultimate Selective (6 filters)
H) Dynamic Hold (hold adapts to signal strength)
I) Rank-Based Selection (combined score)
J) Champion Universe Specific (top 10 only)

ALL signals use NEXT-OPEN execution: signal at close di, entry at O[si, di+1].
Walk-forward by year (2020-2025).
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
    print("  Alpha Futures V119 -- ULTIMATE SYNTHESIS: Best of V116-V118")
    print("=" * 220)
    print("\n  10 ultimate combinations (A-J), walk-forward 2020-2025")
    print("  ALL signals at close di, entry at O[si, di+1] (NEXT DAY OPEN)")
    print("  Synthesizing: ROC+Z-score, Triple Mom, Champion Universe,")
    print("  Day Filter, Vol Filter, Dynamic Hold, Rank-Based Selection")

    # -- Load data -------------------------------------------------
    print("\n[Data] Loading...", flush=True)
    t_start = time.time()
    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)
    years = sorted(set(d.year for d in dates))
    print(f"  {NS} commodities, {ND} days, years: {years}")

    # ================================================================
    # PRECOMPUTE ALL INDICATORS
    # ================================================================
    print("\n[Precompute] ROC, Z-score, ADX, SMA, Volatility...", flush=True)
    t0 = time.time()

    # -- ROC(3), ROC(5), ROC(10) --
    ROC3  = np.full((NS, ND), np.nan)
    ROC5  = np.full((NS, ND), np.nan)
    ROC10 = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si].astype(np.float64)
        ROC3[si]  = talib.ROC(c, timeperiod=3)
        ROC5[si]  = talib.ROC(c, timeperiod=5)
        ROC10[si] = talib.ROC(c, timeperiod=10)
    print(f"  ROC(3,5,10) computed ({time.time()-t0:.1f}s)")

    # -- Daily returns --
    RET = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si].astype(np.float64)
        for di in range(1, ND):
            if not np.isnan(c[di]) and not np.isnan(c[di-1]) and c[di-1] > 0:
                RET[si, di] = (c[di] / c[di-1] - 1) * 100

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
    print(f"  Z-score(20) computed ({time.time()-t0:.1f}s)")

    # -- SMA(50) --
    SMA50 = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si].astype(np.float64)
        SMA50[si] = talib.SMA(c, timeperiod=50)

    # -- ADX(14) --
    ADX14 = np.full((NS, ND), np.nan)
    for si in range(NS):
        h = H[si].astype(np.float64)
        l = L[si].astype(np.float64)
        c = C[si].astype(np.float64)
        ADX14[si] = talib.ADX(h, l, c, timeperiod=14)
    print(f"  SMA(50), ADX(14) computed ({time.time()-t0:.1f}s)")

    # -- 20-day rolling volatility (std of daily returns, in %) --
    VOL20 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(21, ND):
            rets = RET[si, di-20:di]
            valid = rets[~np.isnan(rets)]
            if len(valid) >= 10:
                VOL20[si, di] = np.std(valid, ddof=1)
    print(f"  Volatility(20) computed ({time.time()-t0:.1f}s)")

    # -- Average volatility per commodity (for static universe filter) --
    avg_volatility = np.zeros(NS)
    for si in range(NS):
        c = C[si]
        valid_mask = ~np.isnan(c) & (c > 0)
        if np.sum(valid_mask) > 20:
            rets_v = np.diff(c[valid_mask]) / c[valid_mask][:-1]
            avg_volatility[si] = np.std(rets_v) * 100

    # -- Day-of-week index arrays --
    dow_di = np.array([dates[di].weekday() for di in range(ND)])
    # Thu=3, Fri=4
    thufri_mask = np.isin(dow_di, [3, 4])
    thu_fri_set = set(np.where(thufri_mask)[0])

    print(f"  Day-of-week masks computed")
    print(f"  All indicators ready ({time.time()-t_start:.1f}s total)")

    # ================================================================
    # CHAMPION UNIVERSE: identify top 20 and top 10 by ROC(5)>2% profitability
    # ================================================================
    print("\n[Champion Universe] Per-commodity ROC(5)>2% analysis...", flush=True)

    # Simple per-commodity backtest for ranking
    comm_profit = np.zeros(NS)
    comm_ntrades = np.zeros(NS, dtype=int)
    for si in range(NS):
        cum_pnl = 0.0
        n_tr = 0
        for di in range(MIN_TRAIN, ND - 6):
            if np.isnan(ROC5[si, di]) or ROC5[si, di] <= 2.0:
                continue
            entry_di = di + 1
            if entry_di + 5 >= ND:
                continue
            ep = O[si, entry_di]
            xp = C[si, entry_di + 5]
            if np.isnan(ep) or np.isnan(xp) or ep <= 0:
                continue
            pnl_pct = (xp - ep) / ep * 100
            cum_pnl += pnl_pct
            n_tr += 1
        comm_profit[si] = cum_pnl
        comm_ntrades[si] = n_tr

    # Rank by total profit (only those with >= 5 trades)
    valid_comm = [(si, comm_profit[si], comm_ntrades[si]) for si in range(NS) if comm_ntrades[si] >= 5]
    valid_comm.sort(key=lambda x: -x[1])

    print(f"  Commodities with >=5 trades: {len(valid_comm)}")
    print(f"\n  Top 20 Champions (ROC5>2% cumulative PnL):")
    print(f"  {'#':>3} | {'Sym':<8} | {'CumPnL':>10} | {'NTrades':>7} | {'AvgPnL':>8} | {'Vol%':>6}")
    print("-" * 70)
    for i, (si, pnl, nt) in enumerate(valid_comm[:20]):
        print(f"  {i+1:>3} | {syms[si]:<8} | {pnl:>+9.1f}% | {nt:>7} | {pnl/nt:>+7.3f}% | {avg_volatility[si]:>5.2f}%")

    top20_sis = set(si for si, _, _ in valid_comm[:20])
    top10_sis = set(si for si, _, _ in valid_comm[:10])
    top30_sis = set(si for si, _, _ in valid_comm[:30])
    med_vol_sis = set(si for si in range(NS) if 2.0 <= avg_volatility[si] <= 5.0)

    print(f"\n  Universe sizes: Top10={len(top10_sis)}, Top20={len(top20_sis)}, Top30={len(top30_sis)}, MedVol={len(med_vol_sis)}")

    # ================================================================
    # SIGNAL GENERATION
    # ================================================================
    print("\n[Signals] Computing all signal arrays...", flush=True)
    t0 = time.time()

    # Base signal: ROC(5)>2% AND Z-score>1.5 (V116 best combo)
    sig_rocZ = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(25, ND):
            roc = ROC5[si, di]
            zs = ZSCORE[si, di]
            if np.isnan(roc) or np.isnan(zs):
                continue
            if roc > 2.0 and zs > 1.5:
                sig_rocZ[si, di] = True
    print(f"  ROC(5)>2% AND Z>1.5: {np.sum(sig_rocZ)} signals")

    # Triple momentum: ROC(3)>3% AND ROC(5)>2% AND ROC(10)>0
    sig_triple = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(11, ND):
            r3 = ROC3[si, di]
            r5 = ROC5[si, di]
            r10 = ROC10[si, di]
            if np.isnan(r3) or np.isnan(r5) or np.isnan(r10):
                continue
            if r3 > 3.0 and r5 > 2.0 and r10 > 0:
                sig_triple[si, di] = True
    print(f"  Triple momentum: {np.sum(sig_triple)} signals")

    # Triple + Z-score: ROC(3)>3% AND ROC(5)>2% AND ROC(10)>0 AND Z>1.5
    sig_tripleZ = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(25, ND):
            r3 = ROC3[si, di]
            r5 = ROC5[si, di]
            r10 = ROC10[si, di]
            zs = ZSCORE[si, di]
            if np.isnan(r3) or np.isnan(r5) or np.isnan(r10) or np.isnan(zs):
                continue
            if r3 > 3.0 and r5 > 2.0 and r10 > 0 and zs > 1.5:
                sig_tripleZ[si, di] = True
    print(f"  Triple+Z: {np.sum(sig_tripleZ)} signals")

    # Ultimate selective (6 filters):
    # ROC(5)>2% AND Z>1.5 AND ADX>20 AND C>SMA50 AND vol 2-5% AND (Thu or Fri)
    sig_ultimate6 = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        if si not in med_vol_sis:
            continue
        for di in range(55, ND):
            if not thufri_mask[di]:
                continue
            roc = ROC5[si, di]
            zs = ZSCORE[si, di]
            adx = ADX14[si, di]
            c_now = C[si, di]
            sma50 = SMA50[si, di]
            if np.isnan(roc) or np.isnan(zs) or np.isnan(adx) or np.isnan(c_now) or np.isnan(sma50):
                continue
            if roc > 2.0 and zs > 1.5 and adx > 20 and c_now > sma50:
                sig_ultimate6[si, di] = True
    print(f"  Ultimate 6-filter: {np.sum(sig_ultimate6)} signals")

    # Rank-based score: normalize(ROC5) + normalize(Z-score) + normalize(ADX/50)
    RANK_SCORE = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(55, ND):
            roc = ROC5[si, di]
            zs = ZSCORE[si, di]
            adx = ADX14[si, di]
            if np.isnan(roc) or np.isnan(zs) or np.isnan(adx):
                continue
            n_roc = np.clip(roc / 10.0, -1, 1)
            n_zs = np.clip(zs / 4.0, -1, 1)
            n_adx = np.clip(adx / 50.0, 0, 1)
            RANK_SCORE[si, di] = n_roc + n_zs + n_adx

    # Signal for rank-based: ROC(5)>0 (broad)
    sig_rank = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(55, ND):
            if not np.isnan(ROC5[si, di]) and ROC5[si, di] > 0:
                sig_rank[si, di] = True
    print(f"  Rank-based (ROC5>0): {np.sum(sig_rank)} signals")

    print(f"  All signals computed ({time.time()-t0:.1f}s)")

    # ================================================================
    # BACKTEST ENGINE (supports universe filter, day filter, dynamic hold)
    # ================================================================
    def run_backtest(sig_arr, hold_days, top_n, wf_test_year=None,
                     score_arr=None, rank_desc=True,
                     universe_set=None,      # set of si indices allowed
                     day_filter_set=None,    # set of di indices allowed for entry
                     dynamic_hold_fn=None,   # function(si, di) -> hold_days
                     label=""):
        """Generic backtest with next-open execution.
        dynamic_hold_fn: if provided, overrides hold_days per trade.
        """
        # Universe filter mask
        uni_mask = np.ones(NS, dtype=bool)
        if universe_set is not None:
            for si in range(NS):
                if si not in universe_set:
                    uni_mask[si] = False

        # Date boundaries
        if wf_test_year is not None:
            test_start_di = None
            test_end_di = None
            for di in range(ND):
                if dates[di].year == wf_test_year and test_start_di is None:
                    test_start_di = di
                if dates[di].year == wf_test_year + 1 and test_end_di is None:
                    test_end_di = di
            if test_start_di is None:
                return None
            if test_end_di is None:
                test_end_di = ND
            start_di = MIN_TRAIN
            end_di = test_end_di
        else:
            test_start_di = MIN_TRAIN
            start_di = MIN_TRAIN
            end_di = ND
            test_end_di = ND

        if end_di < start_di + 2:
            return None

        cash = float(CASH0)
        positions = []
        trades = []

        for di in range(start_di, end_di - 1):
            # Reset at WF window start
            if wf_test_year is not None and di == test_start_di:
                cash = float(CASH0)
                positions = []

            # -- Close positions -----------------------------------------
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
                        'pnl_pct': pnl_pct,
                        'entry_di': pos['entry_di'],
                        'exit_di': di,
                        'year': dates[di].year if di < ND else dates[-1].year,
                        'sym': pos['sym'],
                        'days_held': days_held,
                    })
                    closed.append(pos)

            for pos in closed:
                positions.remove(pos)

            if len(positions) >= top_n:
                continue

            # -- Day filter -----------------------------------------------
            if day_filter_set is not None and di not in day_filter_set:
                continue

            # -- Generate signals at day di --------------------------------
            entry_di = di + 1
            if entry_di >= end_di:
                continue

            candidates = []
            for si in range(NS):
                if not uni_mask[si]:
                    continue
                if not sig_arr[si, di]:
                    continue
                if any(p['si'] == si for p in positions):
                    continue
                ep = O[si, entry_di]
                if np.isnan(ep) or ep <= 0:
                    continue
                # Score for ranking
                if score_arr is not None:
                    sc = score_arr[si, di]
                    if np.isnan(sc):
                        sc = 0
                else:
                    sc = ROC5[si, di] if not np.isnan(ROC5[si, di]) else 0
                candidates.append((sc, si, ep))

            if not candidates:
                continue

            # Sort by score
            if rank_desc:
                candidates.sort(key=lambda x: -x[0])
            else:
                candidates.sort(key=lambda x: x[0])

            # Open positions
            n_slots = top_n - len(positions)
            cap_per_slot = cash / max(1, n_slots)
            for sc_val, si, price in candidates[:max(0, n_slots)]:
                sym = syms[si]
                mult = MULT.get(sym, DEF_MULT)
                contracts = max(1, int(cap_per_slot / (price * mult)))
                cost_in = price * mult * contracts * (1 + COMM)
                if cost_in > cash:
                    contracts = int(cash * 0.9 / (price * mult * (1 + COMM)))
                    cost_in = price * mult * contracts * (1 + COMM) if contracts > 0 else 0
                if contracts <= 0 or cost_in <= 0 or cost_in > cash:
                    continue

                # Determine hold period
                if dynamic_hold_fn is not None:
                    hd = dynamic_hold_fn(si, di)
                else:
                    hd = hold_days

                cash -= cost_in
                positions.append({
                    'si': si, 'entry_price': price, 'entry_di': entry_di,
                    'lots': contracts, 'dir': 1, 'sym': sym,
                    'hold_days': hd,
                })

        # Close remaining positions at end
        for pos in positions:
            ae = end_di - 1 if end_di < ND else ND - 1
            exit_price = C[pos['si'], ae]
            if np.isnan(exit_price) or exit_price <= 0:
                exit_price = pos['entry_price']
            mult = MULT.get(pos['sym'], DEF_MULT)
            mkt_val = exit_price * mult * abs(pos['lots'])
            cash += mkt_val - mkt_val * COMM
            pnl = (exit_price - pos['entry_price']) * mult * pos['lots']
            invested = pos['entry_price'] * mult * abs(pos['lots'])
            pnl_pct = pnl / invested * 100 if invested > 0 else 0
            trades.append({
                'pnl_pct': pnl_pct,
                'entry_di': pos['entry_di'],
                'exit_di': ae,
                'year': dates[ae].year if ae < ND else dates[-1].year,
                'sym': pos.get('sym', ''),
                'days_held': ae - pos['entry_di'],
            })

        # Results
        wf_mode = wf_test_year is not None
        n_days_test = (test_end_di - test_start_di) if wf_mode else (end_di - start_di)
        ann = annual_return(cash, CASH0, n_days_test)
        wr = np.mean([1 if t['pnl_pct'] > 0 else 0 for t in trades]) * 100 if trades else 0
        n_trades = len(trades)
        avg_pnl = np.mean([t['pnl_pct'] for t in trades]) if trades else 0
        avg_hold = np.mean([t['days_held'] for t in trades]) if trades else 0

        # Max drawdown from trade-based equity curve
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

        freq_per_yr = n_trades / (n_days_test / 252) if n_days_test > 0 else 0

        return {
            'ann': ann, 'wr': wr, 'n': n_trades, 'avg_pnl': avg_pnl,
            'final_cash': cash, 'n_days': n_days_test, 'mdd': mdd,
            'avg_hold': avg_hold, 'freq': freq_per_yr, 'label': label,
        }

    # ================================================================
    # HELPER: walk-forward for a config
    # ================================================================
    wf_years = [2020, 2021, 2022, 2023, 2024, 2025]

    def run_wf(sig_arr, hold_days, top_n, score_arr=None,
               universe_set=None, day_filter_set=None,
               dynamic_hold_fn=None, label=""):
        """Run walk-forward and return summary dict."""
        windows = {}
        mdd_dict = {}
        wr_dict = {}
        for yr in wf_years:
            r = run_backtest(sig_arr, hold_days, top_n, score_arr=score_arr,
                             universe_set=universe_set,
                             day_filter_set=day_filter_set,
                             dynamic_hold_fn=dynamic_hold_fn,
                             wf_test_year=yr, label=label)
            if r:
                windows[yr] = r['ann']
                mdd_dict[yr] = r['mdd']
                wr_dict[yr] = r['wr']
        vals = [windows.get(yr, 0) for yr in wf_years]
        avg = np.mean(vals) if vals else 0
        pos = sum(1 for v in vals if v > 0)
        avg_mdd = np.mean(list(mdd_dict.values())) if mdd_dict else 0
        avg_wr = np.mean(list(wr_dict.values())) if wr_dict else 0
        return {'windows': windows, 'avg': avg, 'pos': pos, 'vals': vals,
                'avg_mdd': avg_mdd, 'avg_wr': avg_wr}

    # ================================================================
    # DYNAMIC HOLD FUNCTION
    # ================================================================
    def dynamic_hold_fn(si, di):
        """If Z-score > 2.5: hold 1 day. If 1.5-2.5: hold 3 days. If 1.0-1.5: hold 5 days."""
        zs = ZSCORE[si, di]
        if np.isnan(zs):
            return 3
        if zs > 2.5:
            return 1
        elif zs >= 1.5:
            return 3
        else:
            return 5

    # ================================================================
    # BUILD CONFIGURATIONS
    # ================================================================
    print("\n[Sweep] Building 10 ultimate configurations (A-J)...", flush=True)
    configs = []
    cid = 0

    # A) TRIPLE MOMENTUM + CHAMPION UNIVERSE (Top 20)
    for hd in [1, 3, 5]:
        cid += 1
        configs.append({
            'id': cid, 'signal': 'A', 'label': f"A_TripleMom_Top20_H{hd}",
            'sig_arr': sig_triple, 'score_arr': None,
            'hold_days': hd, 'top_n': 1,
            'universe_set': top20_sis, 'day_filter_set': None,
            'dynamic_hold_fn': None,
        })

    # B) ROC+Z-SCORE + CHAMPION UNIVERSE (Top 20)
    for hd in [1, 3, 5]:
        cid += 1
        configs.append({
            'id': cid, 'signal': 'B', 'label': f"B_RocZ_Top20_H{hd}",
            'sig_arr': sig_rocZ, 'score_arr': None,
            'hold_days': hd, 'top_n': 1,
            'universe_set': top20_sis, 'day_filter_set': None,
            'dynamic_hold_fn': None,
        })

    # C) ROC+Z-SCORE + DAY FILTER (Thu+Fri only)
    for hd in [1, 3]:
        cid += 1
        configs.append({
            'id': cid, 'signal': 'C', 'label': f"C_RocZ_ThuFri_H{hd}",
            'sig_arr': sig_rocZ, 'score_arr': None,
            'hold_days': hd, 'top_n': 1,
            'universe_set': None, 'day_filter_set': thu_fri_set,
            'dynamic_hold_fn': None,
        })

    # D) ROC+Z-SCORE + VOL FILTER (medium vol 2-5%)
    for hd in [1, 3, 5]:
        cid += 1
        configs.append({
            'id': cid, 'signal': 'D', 'label': f"D_RocZ_MedVol_H{hd}",
            'sig_arr': sig_rocZ, 'score_arr': None,
            'hold_days': hd, 'top_n': 1,
            'universe_set': med_vol_sis, 'day_filter_set': None,
            'dynamic_hold_fn': None,
        })

    # E) TRIPLE MOMENTUM + Z-SCORE + CHAMPION (Top 20)
    for hd in [1, 3]:
        cid += 1
        configs.append({
            'id': cid, 'signal': 'E', 'label': f"E_TripleZ_Top20_H{hd}",
            'sig_arr': sig_tripleZ, 'score_arr': None,
            'hold_days': hd, 'top_n': 1,
            'universe_set': top20_sis, 'day_filter_set': None,
            'dynamic_hold_fn': None,
        })

    # F) EVERYTHING COMBINED: Top 20 + Med Vol + ROC+Z + Thu+Fri
    for hd in [1, 3]:
        cid += 1
        # Intersection of top20 and med_vol
        combo_sis = top20_sis & med_vol_sis
        configs.append({
            'id': cid, 'signal': 'F', 'label': f"F_AllCombo_Top20MedVol_ThuFri_H{hd}",
            'sig_arr': sig_rocZ, 'score_arr': None,
            'hold_days': hd, 'top_n': 1,
            'universe_set': combo_sis, 'day_filter_set': thu_fri_set,
            'dynamic_hold_fn': None,
        })

    # G) ULTIMATE SELECTIVE (6 filters)
    for hd in [1, 3]:
        cid += 1
        configs.append({
            'id': cid, 'signal': 'G', 'label': f"G_Ultimate6_H{hd}",
            'sig_arr': sig_ultimate6, 'score_arr': None,
            'hold_days': hd, 'top_n': 1,
            'universe_set': None, 'day_filter_set': None,
            'dynamic_hold_fn': None,
        })

    # H) DYNAMIC HOLD: ROC+Z-score, hold adapts to Z-score strength
    cid += 1
    configs.append({
        'id': cid, 'signal': 'H', 'label': f"H_DynamicHold_RocZ",
        'sig_arr': sig_rocZ, 'score_arr': None,
        'hold_days': 3, 'top_n': 1,  # hold_days overridden by dynamic_hold_fn
        'universe_set': None, 'day_filter_set': None,
        'dynamic_hold_fn': dynamic_hold_fn,
    })
    # H2: Dynamic hold + Top 20
    cid += 1
    configs.append({
        'id': cid, 'signal': 'H', 'label': f"H_DynamicHold_RocZ_Top20",
        'sig_arr': sig_rocZ, 'score_arr': None,
        'hold_days': 3, 'top_n': 1,
        'universe_set': top20_sis, 'day_filter_set': None,
        'dynamic_hold_fn': dynamic_hold_fn,
    })

    # I) RANK-BASED SELECTION (top 30 universe, score-based)
    cid += 1
    configs.append({
        'id': cid, 'signal': 'I', 'label': f"I_RankScore_Top30_H3",
        'sig_arr': sig_rank, 'score_arr': RANK_SCORE,
        'hold_days': 3, 'top_n': 1,
        'universe_set': top30_sis, 'day_filter_set': None,
        'dynamic_hold_fn': None,
    })
    # I2: Rank-based + ROC+Z signal (more selective)
    cid += 1
    configs.append({
        'id': cid, 'signal': 'I', 'label': f"I_RankScore_RocZ_Top30_H3",
        'sig_arr': sig_rocZ, 'score_arr': RANK_SCORE,
        'hold_days': 3, 'top_n': 1,
        'universe_set': top30_sis, 'day_filter_set': None,
        'dynamic_hold_fn': None,
    })

    # J) CHAMPION UNIVERSE SPECIFIC (Top 10 only)
    for hd in [1, 3]:
        cid += 1
        configs.append({
            'id': cid, 'signal': 'J', 'label': f"J_RocZ_Top10_H{hd}",
            'sig_arr': sig_rocZ, 'score_arr': None,
            'hold_days': hd, 'top_n': 1,
            'universe_set': top10_sis, 'day_filter_set': None,
            'dynamic_hold_fn': None,
        })

    # Additional combos
    # J3: Top 10 + Dynamic hold
    cid += 1
    configs.append({
        'id': cid, 'signal': 'J', 'label': f"J_DynamicHold_Top10",
        'sig_arr': sig_rocZ, 'score_arr': None,
        'hold_days': 3, 'top_n': 1,
        'universe_set': top10_sis, 'day_filter_set': None,
        'dynamic_hold_fn': dynamic_hold_fn,
    })
    # B4: Top 20 + Thu+Fri
    for hd in [1, 3]:
        cid += 1
        configs.append({
            'id': cid, 'signal': 'B', 'label': f"B_RocZ_Top20_ThuFri_H{hd}",
            'sig_arr': sig_rocZ, 'score_arr': None,
            'hold_days': hd, 'top_n': 1,
            'universe_set': top20_sis, 'day_filter_set': thu_fri_set,
            'dynamic_hold_fn': None,
        })
    # D4: Med vol + Thu+Fri
    for hd in [1, 3]:
        cid += 1
        configs.append({
            'id': cid, 'signal': 'D', 'label': f"D_RocZ_MedVol_ThuFri_H{hd}",
            'sig_arr': sig_rocZ, 'score_arr': None,
            'hold_days': hd, 'top_n': 1,
            'universe_set': med_vol_sis, 'day_filter_set': thu_fri_set,
            'dynamic_hold_fn': None,
        })

    total = len(configs)
    print(f"  Total configs: {total}")

    # ================================================================
    # RUN ALL CONFIGS (full backtest)
    # ================================================================
    print("\n[Backtest] Running all configs...", flush=True)
    t1 = time.time()
    results = []

    for ci, cfg in enumerate(configs):
        if ci % 5 == 0:
            print(f"  Config {ci}/{total} ({len(results)} done, {time.time()-t1:.0f}s)", flush=True)

        r = run_backtest(
            cfg['sig_arr'], cfg['hold_days'], cfg['top_n'],
            score_arr=cfg.get('score_arr'),
            universe_set=cfg.get('universe_set'),
            day_filter_set=cfg.get('day_filter_set'),
            dynamic_hold_fn=cfg.get('dynamic_hold_fn'),
            label=cfg['label'],
        )
        if r and r['n'] >= 2:
            r['config'] = cfg
            r['signal'] = cfg['signal']
            results.append(r)

    print(f"\n  Done ({time.time()-t1:.0f}s, {len(results)} configs with >= 2 trades)")
    results.sort(key=lambda x: -x['ann'])

    # ================================================================
    # PRINT TOP 30
    # ================================================================
    print(f"\n{'=' * 180}")
    print(f"  TOP 30 RESULTS (sorted by annual return)")
    print(f"{'=' * 180}")
    print(f"  {'#':>3} | {'Label':<40} | {'Ann':>9} | {'WR':>6} | {'N':>5} | {'MDD':>7} | {'AvgPnL':>8} | {'AvgHold':>7} | {'Freq':>6}")
    print("-" * 180)
    for i, r in enumerate(results[:30]):
        print(f"  {i+1:>3} | {r['label']:<40} | {r['ann']:>+8.1f}% | {r['wr']:>5.1f}% | {r['n']:>5} | {r['mdd']:>6.1f}% | {r['avg_pnl']:>+7.3f}% | {r['avg_hold']:>6.1f}d | {r['freq']:>5.1f}/yr")

    # ================================================================
    # BEST PER SIGNAL TYPE
    # ================================================================
    sig_keys = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    sig_names = {
        'A': 'A) Triple Mom + Champion Top20',
        'B': 'B) ROC+Z + Champion Top20',
        'C': 'C) ROC+Z + Thu+Fri',
        'D': 'D) ROC+Z + Med Vol (2-5%)',
        'E': 'E) Triple+Z + Champion Top20',
        'F': 'F) Everything Combined',
        'G': 'G) Ultimate Selective (6 filters)',
        'H': 'H) Dynamic Hold (Z-based)',
        'I': 'I) Rank-Based Selection',
        'J': 'J) Champion Top10',
    }

    best_per_sig = {}
    print(f"\n  BEST PER SIGNAL TYPE:")
    print(f"  {'Signal':<40} | {'Ann':>9} | {'WR':>6} | {'N':>5} | {'MDD':>7} | {'AvgPnL':>8} | {'Configs':>7}")
    print("-" * 140)
    for sig_key in sig_keys:
        sub = [r for r in results if r['signal'] == sig_key]
        if not sub:
            print(f"  {sig_names.get(sig_key, sig_key):<40} | NO RESULTS")
            continue
        best = sub[0]
        best_per_sig[sig_key] = best
        n_pos = sum(1 for r in sub if r['ann'] > 0)
        print(f"  {sig_names.get(sig_key, sig_key):<40} | {best['ann']:>+8.1f}% | {best['wr']:>5.1f}% | {best['n']:>5} | {best['mdd']:>6.1f}% | {best['avg_pnl']:>+7.3f}% | {n_pos}/{len(sub)} pos")

    # ================================================================
    # WALK-FORWARD
    # ================================================================
    # Collect top 15 + best per signal type
    wf_configs = list(results[:15])
    for sig_key in sig_keys:
        if sig_key in best_per_sig:
            r = best_per_sig[sig_key]
            if r not in wf_configs:
                wf_configs.append(r)

    print(f"\n{'=' * 240}")
    print(f"  WALK-FORWARD ({len(wf_configs)} configs, years 2020-2025)")
    print(f"{'=' * 240}")

    header = f"  {'#':>3} | {'Config':<42} | {'Avg':>8} |"
    for yr in wf_years:
        header += f" {yr:>7} |"
    header += f" {'Pos':>4} | {'MDD':>7} | {'WR':>6}"
    print(header)
    print("-" * 240)

    wf_rows = []
    for i, r in enumerate(wf_configs):
        cfg = r['config']
        wf_row = {'label': cfg['label'], 'signal': cfg['signal'],
                  'windows': {}, 'mdd': {}, 'wr': {}}

        for yr in wf_years:
            wr_result = run_backtest(
                cfg['sig_arr'], cfg['hold_days'], cfg['top_n'],
                score_arr=cfg.get('score_arr'),
                universe_set=cfg.get('universe_set'),
                day_filter_set=cfg.get('day_filter_set'),
                dynamic_hold_fn=cfg.get('dynamic_hold_fn'),
                wf_test_year=yr, label=cfg['label'],
            )
            if wr_result:
                wf_row['windows'][yr] = wr_result['ann']
                wf_row['mdd'][yr] = wr_result['mdd']
                wf_row['wr'][yr] = wr_result['wr']
        wf_rows.append(wf_row)

        vals = [wf_row['windows'].get(yr, 0) for yr in wf_years]
        avg = np.mean(vals) if vals else 0
        pos = sum(1 for v in vals if v > 0)
        avg_mdd = np.mean(list(wf_row['mdd'].values())) if wf_row['mdd'] else 0
        avg_wr = np.mean(list(wf_row['wr'].values())) if wf_row['wr'] else 0

        row_str = f"  {i+1:>3} | {wf_row['label']:<42} | {avg:>+7.1f}% |"
        for v in vals:
            row_str += f" {v:>+7.1f}% |"
        row_str += f" {pos}/6 | {avg_mdd:>6.1f}% | {avg_wr:>5.1f}%"
        print(row_str)

    # ================================================================
    # WF COMPARISON PER SIGNAL TYPE
    # ================================================================
    print(f"\n{'=' * 220}")
    print("  WALK-FORWARD COMPARISON (Best per signal type)")
    print(f"{'=' * 220}")
    header2 = f"  {'Signal':<40} | {'WF Avg':>8} |"
    for yr in wf_years:
        header2 += f" {yr:>7} |"
    header2 += f" {'Pos':>4} | Avg MDD | Avg WR"
    print(header2)
    print("-" * 220)

    for sig_key in sig_keys:
        wf_match = [w for w in wf_rows if w['signal'] == sig_key]
        if wf_match:
            wf = wf_match[0]
            vals = [wf['windows'].get(yr, 0) for yr in wf_years]
            avg = np.mean(vals) if vals else 0
            pos = sum(1 for v in vals if v > 0)
            avg_mdd = np.mean(list(wf['mdd'].values())) if wf['mdd'] else 0
            avg_wr = np.mean(list(wf['wr'].values())) if wf['wr'] else 0
            row_str = f"  {sig_names.get(sig_key, sig_key):<40} | {avg:>+7.1f}% |"
            for v in vals:
                row_str += f" {v:>+7.1f}% |"
            row_str += f" {pos}/6 | {avg_mdd:>6.1f}% | {avg_wr:>5.1f}%"
            print(row_str)
        else:
            print(f"  {sig_names.get(sig_key, sig_key):<40} | NO DATA")

    # ================================================================
    # ANSWER THE 5 KEY QUESTIONS
    # ================================================================
    print(f"\n{'=' * 220}")
    print("  ANSWERS TO 5 KEY QUESTIONS")
    print(f"{'=' * 220}")

    # Q1: Top 5 by annual return -- any breaking 200%?
    print(f"\n  1. TOP 5 CONFIGS BY ANNUAL RETURN -- Any breaking 200%?")
    over200 = [r for r in results if r['ann'] > 200]
    print(f"     Configs > 200%: {len(over200)}")
    for i, r in enumerate(results[:5]):
        wf_match = [w for w in wf_rows if w['label'] == r['label']]
        wf_info = ""
        if wf_match:
            vals = [wf_match[0]['windows'].get(yr, 0) for yr in wf_years]
            wf_pos = sum(1 for v in vals if v > 0)
            wf_avg = np.mean(vals)
            wf_info = f"WF {wf_pos}/6 pos, avg {wf_avg:>+7.1f}%"
        print(f"     {i+1}. {r['label']:<42} | {r['ann']:>+8.1f}% | WR={r['wr']:>5.1f}% | N={r['n']:>4} | MDD={r['mdd']:>6.1f}% | {wf_info}")

    # Q2: Does universe selection help with best signals?
    print(f"\n  2. DOES UNIVERSE SELECTION HELP WITH ROC+Z-SCORE?")
    # Compare B (Top20) vs all commodities
    b_all = [r for r in results if r['signal'] == 'B' and 'Top20' in r['label'] and 'ThuFri' not in r['label']]
    d_all = [r for r in results if r['signal'] == 'D' and 'MedVol' in r['label'] and 'ThuFri' not in r['label']]
    j_all = [r for r in results if r['signal'] == 'J' and 'Top10' in r['label'] and 'Dynamic' not in r['label']]

    # Baseline: ROC+Z on all 68, hold=1,3
    for hd in [1, 3]:
        base_r = run_backtest(sig_rocZ, hd, 1, label=f"BASELINE_RocZ_H{hd}")
        base_wf = run_wf(sig_rocZ, hd, 1, label=f"BASELINE_RocZ_H{hd}")
        print(f"     Baseline (all 68, H{hd}):  Ann={base_r['ann']:>+8.1f}%  WF_avg={base_wf['avg']:>+7.1f}%  WF_pos={base_wf['pos']}/6")

        for subset, name in [(b_all, "Top20"), (j_all, "Top10"), (d_all, "MedVol")]:
            match = [r for r in subset if f"H{hd}" in r['label']]
            if match:
                m = match[0]
                wf_m = [w for w in wf_rows if w['label'] == m['label']]
                if wf_m:
                    wvals = [wf_m[0]['windows'].get(yr, 0) for yr in wf_years]
                    wavg = np.mean(wvals)
                    wpos = sum(1 for v in wvals if v > 0)
                    print(f"     {name:>8} (H{hd}):  Ann={m['ann']:>+8.1f}%  WF_avg={wavg:>+7.1f}%  WF_pos={wpos}/6  {'BETTER' if m['ann'] > base_r['ann'] else 'WORSE'}")

    # Q3: Does day-of-week filter still help with ROC+Z?
    print(f"\n  3. DOES DAY-OF-WEEK FILTER (THU+FRI) HELP WITH ROC+Z-SCORE?")
    for hd in [1, 3]:
        base_r = run_backtest(sig_rocZ, hd, 1, label=f"BASELINE_RocZ_H{hd}")
        day_r = run_backtest(sig_rocZ, hd, 1, day_filter_set=thu_fri_set, label=f"DAY_RocZ_ThuFri_H{hd}")
        if day_r:
            base_wf = run_wf(sig_rocZ, hd, 1, label=f"BASELINE_RocZ_H{hd}")
            day_wf = run_wf(sig_rocZ, hd, 1, day_filter_set=thu_fri_set, label=f"DAY_RocZ_ThuFri_H{hd}")
            print(f"     Hold={hd}d: All days Ann={base_r['ann']:>+8.1f}% WF={base_wf['avg']:>+7.1f}%({base_wf['pos']}/6)  vs  Thu+Fri Ann={day_r['ann']:>+8.1f}% WF={day_wf['avg']:>+7.1f}%({day_wf['pos']}/6)  {'DAY FILTER HELPS' if day_r['ann'] > base_r['ann'] else 'DAY FILTER HURTS'}")

    # Q4: Dynamic hold vs fixed hold
    print(f"\n  4. DYNAMIC HOLD RESULTS VS FIXED HOLD:")
    h_results = [r for r in results if r['signal'] == 'H']
    for r in h_results:
        wf_m = [w for w in wf_rows if w['label'] == r['label']]
        if wf_m:
            wvals = [wf_m[0]['windows'].get(yr, 0) for yr in wf_years]
            wavg = np.mean(wvals)
            wpos = sum(1 for v in wvals if v > 0)
            print(f"     {r['label']:<42} | Ann={r['ann']:>+8.1f}% | WF_avg={wavg:>+7.1f}% | WF_pos={wpos}/6 | AvgHold={r['avg_hold']:>5.1f}d")

    # Compare with fixed hold baselines
    for hd in [1, 3, 5]:
        base_r = run_backtest(sig_rocZ, hd, 1, label=f"FIXED_RocZ_H{hd}")
        base_wf = run_wf(sig_rocZ, hd, 1, label=f"FIXED_RocZ_H{hd}")
        print(f"     Fixed H{hd}:  Ann={base_r['ann']:>+8.1f}%  WF_avg={base_wf['avg']:>+7.1f}%  WF_pos={base_wf['pos']}/6")

    # Q5: Most promising direction for 600%
    print(f"\n  5. MOST PROMISING DIRECTION FOR REACHING 600%:")
    # Analyze which combination of filters gives the highest WF avg
    print(f"\n  RANKED BY WALK-FORWARD AVERAGE:")
    wf_ranked = []
    for r in results:
        wf_m = [w for w in wf_rows if w['label'] == r['label']]
        if wf_m:
            wvals = [wf_m[0]['windows'].get(yr, 0) for yr in wf_years]
            wavg = np.mean(wvals)
            wpos = sum(1 for v in wvals if v > 0)
            wmdd = np.mean(list(wf_m[0]['mdd'].values())) if wf_m[0]['mdd'] else 0
            wf_ranked.append((r, wavg, wpos, wmdd))
    wf_ranked.sort(key=lambda x: -x[1])
    for i, (r, wavg, wpos, wmdd) in enumerate(wf_ranked[:10]):
        print(f"     {i+1:>2}. {r['label']:<42} | Ann={r['ann']:>+8.1f}% | WF_avg={wavg:>+7.1f}% | WF_pos={wpos}/6 | MDD={r['mdd']:>6.1f}% | N={r['n']}")

    # ================================================================
    # FINAL CHAMPION
    # ================================================================
    print(f"\n{'=' * 220}")
    print("  FINAL CHAMPION: V119 ULTIMATE SYNTHESIS")
    print(f"{'=' * 220}")

    if results:
        champ = results[0]
        champ_wf = [w for w in wf_rows if w['label'] == champ['label']]
        print(f"\n  CHAMPION: {champ['label']}")
        print(f"    Annual: {champ['ann']:>+8.1f}%  |  WR: {champ['wr']:>5.1f}%  |  N: {champ['n']:>4}  |  MDD: {champ['mdd']:>6.1f}%")
        print(f"    Avg PnL: {champ['avg_pnl']:>+6.3f}%  |  Avg Hold: {champ['avg_hold']:>5.1f}d  |  Freq: {champ['freq']:>5.1f}/yr")
        if champ_wf:
            cw = champ_wf[0]
            vals = [cw['windows'].get(yr, 0) for yr in wf_years]
            print(f"    Walk-forward: {[f'{v:>+7.1f}%' for v in vals]}")
            print(f"    WF avg: {np.mean(vals):>+8.1f}%  |  {sum(1 for v in vals if v > 0)}/6 positive")
            print(f"    Avg MDD: {np.mean(list(cw['mdd'].values())):>6.1f}%  |  Avg WR: {np.mean(list(cw['wr'].values())):>5.1f}%")

    # Also show best by WF avg
    if wf_ranked:
        best_wf = wf_ranked[0]
        print(f"\n  BEST WALK-FORWARD: {best_wf[0]['label']}")
        print(f"    Annual: {best_wf[0]['ann']:>+8.1f}%  |  WF_avg: {best_wf[1]:>+7.1f}%  |  WF_pos: {best_wf[2]}/6")
        bw_wf = [w for w in wf_rows if w['label'] == best_wf[0]['label']]
        if bw_wf:
            bw_vals = [bw_wf[0]['windows'].get(yr, 0) for yr in wf_years]
            print(f"    WF years: {[f'{v:>+7.1f}%' for v in bw_vals]}")

    # Top 10 commodities
    print(f"\n  TOP 10 CHAMPION COMMODITIES:")
    for i, (si, pnl, nt) in enumerate(valid_comm[:10]):
        print(f"    {i+1:>2}. {syms[si]:<8} | CumPnL={pnl:>+8.1f}% | NTrades={nt:>4} | AvgPnL={pnl/nt:>+6.3f}% | Vol={avg_volatility[si]:>5.2f}%")

    print(f"\n  TOTAL TIME: {time.time()-t_start:.0f}s")


if __name__ == '__main__':
    main()
