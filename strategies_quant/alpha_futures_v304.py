"""
V304: Pair Trading (V62) + Regime Factor (V301) Fusion
=======================================================
组合策略:
1. Pair trading: 统计套利, 市场中性, 每日交易
2. Regime factor: 方向性信号, 趋势/均值回归切换
3. 两者独立运行, 合并权益曲线
4. Walk-forward验证

V62 pair: 94.6%年化 (之前最佳)
V301 regime: 10.9%年化 WF (最稳健)
目标: 结合达到更高收益更低回撤
"""
import sys, os, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.data_loader import list_available_symbols, load_stock_data
try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False

COMMISSION = 0.0005
CASH0 = 1_000_000

# 产业链配对
PAIRS = [
    ('rbfi','ifi'), ('hcfi','ifi'), ('hcfi','rbfi'),
    ('jfi','jmfi'), ('mafi','scfi'), ('fufi','scfi'),
    ('ofi','mfi'), ('yfi','ofi'), ('pfi','yfi'),
    ('ppfi','mafi'), ('vfi','mafi'), ('egfi','mafi'),
    ('cfi','csfi'), ('tafi','mafi'), ('lfi','ppfi'),
]

SPREAD_RAW = 0; SPREAD_PCT = 1; SPREAD_LOG = 2

COMMODITY_GROUPS = {
    'BLACK': ['rbfi','hcfi','ifi','jfi','jmfi'],
    'METAL': ['cufi','alfi','znfi','nifi','snfi'],
    'PRECIOUS': ['aufi','agfi'],
    'ENERGY': ['scfi','bufi','fufi','tafi','mafi'],
    'CHEM': ['ppfi','lfi','vfi','egfi','ebfi','safi'],
    'OILCHAIN': ['mfi','yfi','ofi','pfi','rmfi'],
    'GRAIN': ['cfi','csfi','srfi','cffi'],
}


# ============================================================
# DATA LOADING
# ============================================================
def load_all_data(start='2016-01-01', end=None, min_days=500):
    print("[V304] Loading data...", flush=True)
    t0 = time.time()
    syms = list_available_symbols('daily')
    stock_data = {}
    for sym in syms:
        try:
            df = load_stock_data(sym, frequency='daily')
            if df is not None and len(df) >= min_days:
                stock_data[sym] = df
        except: pass

    vol_map = {s: df['volume'].tail(60).mean()
               for s, df in stock_data.items()
               if 'volume' in df.columns and df['volume'].tail(60).mean() > 0}
    syms = sorted([s for s, _ in sorted(vol_map.items(), key=lambda x: -x[1])[:50]])
    NS = len(syms)
    all_dates = sorted(set(d for s in syms for d in stock_data[s].index))
    i0 = next(i for i, d in enumerate(all_dates) if d >= pd.Timestamp(start))
    if end: i1 = next((i for i, d in enumerate(all_dates) if d > pd.Timestamp(end)), len(all_dates))-1
    else: i1 = len(all_dates)-1
    dates = all_dates[i0:i1+1]; ND = len(dates)
    dm = {d: i for i, d in enumerate(all_dates)}

    C = np.full((NS, len(all_dates)), np.nan)
    O = np.full((NS, len(all_dates)), np.nan)
    H = np.full((NS, len(all_dates)), np.nan)
    L = np.full((NS, len(all_dates)), np.nan)
    V = np.full((NS, len(all_dates)), np.nan)
    OI = np.full((NS, len(all_dates)), np.nan)
    for si, s in enumerate(syms):
        df = stock_data.get(s)
        if df is None: continue
        df = df[~df.index.duplicated(keep='first')]
        common = df.index[df.index.isin(dm)]
        if len(common)==0: continue
        idx = np.array([dm[d] for d in common])
        for col, arr in [('close',C),('open',O),('high',H),('low',L),('volume',V),('oi',OI)]:
            if col in df.columns: arr[si, idx] = df.loc[common, col].values.astype(float)

    C=C[:,i0:i1+1]; O=O[:,i0:i1+1]; H=H[:,i0:i1+1]
    L=L[:,i0:i1+1]; V=V[:,i0:i1+1]; OI=OI[:,i0:i1+1]
    sym_idx = {s: i for i, s in enumerate(syms)}
    print(f"  {NS} sym, {ND} days ({time.time()-t0:.1f}s)")
    return C, O, H, L, V, OI, NS, ND, dates, syms, sym_idx


# ============================================================
# PAIR TRADING ENGINE (from V62)
# ============================================================
def run_pairs(C, O, NS, ND, dates, syms, sym_idx,
              z_thresh=0.8, hold_max=1, exit_z=0.0, max_pairs=3,
              lookback=15, mode=SPREAD_LOG):
    """Pair trading: z-score mean reversion on commodity pairs."""
    trades = []

    for pd_sym, pu_sym in PAIRS:
        pdi = sym_idx.get(pd_sym)
        pui = sym_idx.get(pu_sym)
        if pdi is None or pui is None: continue

        pd_c = C[pdi]; pu_c = C[pui]

        for di in range(lookback+1, ND):
            # Compute spread
            pd_val = pd_c[di-1]; pu_val = pu_c[di-1]
            if np.isnan(pd_val) or np.isnan(pu_val) or pu_val <= 0: continue

            if mode == SPREAD_RAW: spread = pd_val - pu_val
            elif mode == SPREAD_PCT: spread = (pd_val - pu_val) / pu_val
            else: spread = np.log(pd_val) - np.log(pu_val)

            # Z-score from lookback window
            window = []
            for j in range(di-lookback, di):
                pv = pd_c[j]; uv = pu_c[j]
                if np.isnan(pv) or np.isnan(uv) or uv <= 0: continue
                if mode == SPREAD_RAW: window.append(pv - uv)
                elif mode == SPREAD_PCT: window.append((pv-uv)/uv)
                else: window.append(np.log(pv) - np.log(uv))

            if len(window) < lookback - 3: continue
            mu = np.mean(window); sig = np.std(window)
            if sig < 1e-10: continue
            z = (spread - mu) / sig

            # Entry: z-score exceeds threshold
            if abs(z) >= z_thresh:
                direction = -1 if z > 0 else 1  # mean reversion
                entry_p = O[pdi, di]  # enter at next open
                entry_u = O[pui, di]
                if np.isnan(entry_p) or np.isnan(entry_u): continue

                # Find exit day (hold_max or z reverts)
                exit_di = min(di + hold_max, ND-1)
                for edi in range(di, exit_di+1):
                    pv = pd_c[edi]; uv = pu_c[edi]
                    if np.isnan(pv) or np.isnan(uv): continue
                    if mode == SPREAD_RAW: sp = pv - uv
                    elif mode == SPREAD_PCT: sp = (pv-uv)/uv
                    else: sp = np.log(pv) - np.log(uv)
                    z_now = (sp - mu) / sig

                    if edi > di and ((direction > 0 and z_now >= exit_z) or
                                     (direction < 0 and z_now <= exit_z)):
                        exit_di = edi; break

                # Calculate PnL
                exit_p = C[pdi, exit_di]; exit_u = C[pui, exit_di]
                if np.isnan(exit_p) or np.isnan(exit_u): continue

                # Long pd/short pu if z<0, Short pd/long pu if z>0
                pnl_pd = direction * (exit_p - entry_p) / entry_p if entry_p > 0 else 0
                pnl_pu = -direction * (exit_u - entry_u) / entry_u if entry_u > 0 else 0
                pnl = (pnl_pd + pnl_pu) / 2 - COMMISSION * 2

                trades.append({
                    'pair': f"{pd_sym}/{pu_sym}", 'entry_d': dates[di],
                    'exit_d': dates[exit_di], 'z': z, 'dir': direction,
                    'pnl': pnl, 'hold': exit_di - di,
                    'pd_entry': entry_p, 'pu_entry': entry_u
                })

    return trades


# ============================================================
# REGIME FACTOR ENGINE (simplified from V301)
# ============================================================
def compute_regime_signal(C, O, H, L, V, OI, NS, ND):
    """Quick regime-aware factor signal."""
    # Compute key factors
    mom20 = np.full((NS,ND), np.nan)
    adx = np.full((NS,ND), np.nan)
    for si in range(NS):
        c = C[si]; nc = np.isnan(c)
        for di in range(20, ND):
            if not nc[di] and not nc[di-20] and c[di-20]>0:
                mom20[si,di] = c[di]/c[di-20]-1
        if HAS_TALIB:
            try:
                hs=np.where(np.isnan(H[si]),0,H[si]).astype(np.float64)
                ls=np.where(np.isnan(L[si]),0,L[si]).astype(np.float64)
                cs=np.where(nc,0,c).astype(np.float64)
                adx[si] = np.where(nc, np.nan, talib.ADX(hs,ls,cs,14))
            except: pass

    # Market regime
    mkt_adx = np.full(ND, np.nan)
    for di in range(60, ND):
        v = adx[:,di]; vld = v[~np.isnan(v)]
        if len(vld) > 5: mkt_adx[di] = np.mean(vld)

    # Signal: cross-sectional momentum rank
    signal = np.full((NS,ND), np.nan)
    for di in range(60, ND):
        vals = mom20[:,di]; vld = ~np.isnan(vals)
        if vld.sum() > 5:
            ranks = pd.Series(vals).rank(pct=True, na_option='keep').values
            # Adjust by regime
            adx_val = mkt_adx[di]
            if not np.isnan(adx_val):
                if adx_val > 30:  # trending → momentum
                    signal[:,di] = ranks * 1.5 - 0.75  # widen spread
                elif adx_val < 20:  # choppy → mean reversion
                    signal[:,di] = -ranks * 1.0 + 0.5  # invert
                else:
                    signal[:,di] = ranks - 0.5
            else:
                signal[:,di] = ranks - 0.5
    return signal


def run_factor(C, O, H, L, NS, ND, dates, syms,
               top_n=3, hold_days=5, atr_stop=3.0, leverage=2.0):
    """Factor-based directional strategy."""
    signal = compute_regime_signal(C, O, H, L, None, None, NS, ND)
    pos_alloc = leverage / top_n

    equity = CASH0; positions = []; trades = []
    for di in range(60, ND):
        # Exit
        new_pos = []
        for si, edi, ep, sp, d, a in positions:
            c = C[si,di]
            if np.isnan(c): new_pos.append((si,edi,ep,sp,d,a)); continue
            exit_r = None
            if d>0 and c<sp: exit_r='stop'
            elif di-edi>=hold_days: exit_r='hold'
            if exit_r:
                pnl = d*(c-ep)/ep - COMMISSION
                equity *= (1 + a * pnl)
                trades.append({'sym':syms[si],'entry_d':dates[edi],'exit_d':dates[di],
                    'dir':d,'pnl':pnl,'hold':di-edi,'reason':exit_r})
            else:
                new_pos.append((si,edi,ep,sp,d,a))
        positions = new_pos
        if equity <= 0: break
        if len(positions) >= top_n: continue

        held = {p[0] for p in positions}
        sig_vals = [(signal[si,di],si) for si in range(NS)
                    if not np.isnan(signal[si,di]) and si not in held
                    and not np.isnan(C[si,di]) and not np.isnan(O[si,di])
                    and signal[si,di] > 0.1]
        if not sig_vals: continue
        sig_vals.sort(key=lambda x: x[0], reverse=True)

        for score, si in sig_vals[:top_n]:
            if len(positions) >= top_n: break
            op = O[si,di]
            if np.isnan(op) or op<=0: continue
            atr_v = []
            for j in range(max(60,di-14),di):
                hh,ll,cc = H[si,j],L[si,j],C[si,j]
                if not any(np.isnan([hh,ll,cc])): atr_v.append(max(hh-ll,abs(hh-cc),abs(ll-cc)))
            if not atr_v: continue
            atr = np.mean(atr_v)
            positions.append((si, di, op, op-atr_stop*atr, 1, pos_alloc))
            held.add(si)
    return trades, equity


# ============================================================
# FUSION: Combined Portfolio
# ============================================================
def run_fusion(C, O, H, L, V, OI, NS, ND, dates, syms, sym_idx,
               pair_alloc=0.5, factor_alloc=0.5,
               pair_params=None, factor_params=None):
    """
    Run both strategies and combine at portfolio level.
    pair_alloc: fraction of capital for pair trading
    factor_alloc: fraction of capital for factor strategy
    """
    print(f"\n[V304] Fusion: pair={pair_alloc*100:.0f}% factor={factor_alloc*100:.0f}%")

    # Pair trades
    pp = pair_params or {'z_thresh': 0.8, 'hold_max': 1, 'lookback': 15}
    pair_trades = run_pairs(C, O, NS, ND, dates, syms, sym_idx, **pp)

    # Factor trades
    fp = factor_params or {'top_n': 3, 'hold_days': 5, 'atr_stop': 3.0, 'leverage': 2.0}
    factor_trades, factor_equity = run_factor(C, O, H, L, NS, ND, dates, syms, **fp)

    # Build combined equity curve
    # Pairs: daily P&L from sorted trades
    pair_equity = CASH0 * pair_alloc
    if pair_trades:
        sorted_pt = sorted(pair_trades, key=lambda t: t['entry_d'])
        for t in sorted_pt:
            pair_equity *= (1 + t['pnl'] * pair_alloc / (pair_equity / (CASH0 * pair_alloc)))

    # Factor: already tracked
    factor_eq = factor_equity

    # Combined
    combined_equity = pair_equity + factor_eq * factor_alloc

    # Analyze
    print(f"\n--- Pair Trading ({len(pair_trades)} trades) ---")
    if pair_trades:
        pnls = [t['pnl'] for t in pair_trades]
        wr = sum(1 for p in pnls if p>0)/len(pnls)*100
        cum = pair_equity / (CASH0 * pair_alloc) - 1
        yrs = (dates[-1]-dates[0]).days/365.25
        ann = (1+cum)**(1/max(yrs,.1))-1 if cum>-1 else -1
        print(f"  WR: {wr:.1f}% | Avg: {np.mean(pnls)*100:.3f}% | Cum: {cum*100:.1f}% | Ann: {ann*100:.1f}%")

    print(f"\n--- Factor Strategy ({len(factor_trades)} trades) ---")
    if factor_trades:
        pnls = [t['pnl'] for t in factor_trades]
        wr = sum(1 for p in pnls if p>0)/len(pnls)*100
        print(f"  WR: {wr:.1f}% | Avg: {np.mean(pnls)*100:.3f}% | Final equity: {factor_eq:,.0f}")

    print(f"\n--- Combined ---")
    total_ret = combined_equity / CASH0 - 1
    yrs = (dates[-1]-dates[0]).days/365.25
    ann = (1+total_ret)**(1/max(yrs,.1))-1 if total_ret>-1 else -1
    print(f"  Total: {total_ret*100:.1f}% | Ann: {ann*100:.1f}% | Final: {combined_equity:,.0f}")

    return pair_trades, factor_trades, combined_equity


# ============================================================
# WALK-FORWARD
# ============================================================
def walk_forward(C, O, H, L, V, OI, NS, ND, dates, syms, sym_idx,
                 train_years=4, test_years=1):
    print(f"\n{'='*60}")
    print(f"  WALK-FORWARD")
    print(f"{'='*60}")

    years = sorted(set(d.year for d in dates))
    all_pair_trades = []; all_factor_trades = []
    start_yr = years[0]

    while True:
        train_end = start_yr + train_years - 1
        test_yr = train_end + 1
        if test_yr > years[-1]: break

        train_m = np.array([d.year <= train_end for d in dates])
        test_m = np.array([d.year == test_yr for d in dates])
        t0 = max(0, np.where(train_m)[0][0]-60)
        sl = slice(t0, np.where(test_m)[0][-1]+1)

        C_s = C[:,sl]; O_s = O[:,sl]; H_s = H[:,sl]; L_s = L[:,sl]
        V_s = V[:,sl]; OI_s = OI[:,sl]
        d_s = dates[sl]; ND_s = len(d_s)

        pt = run_pairs(C_s, O_s, NS, ND_s, d_s, syms, sym_idx,
                       z_thresh=0.8, hold_max=1, lookback=15)
        ft, fe = run_factor(C_s, O_s, H_s, L_s, NS, ND_s, d_s, syms,
                            top_n=3, hold_days=5, atr_stop=3.0, leverage=2.0)

        pt_yr = [t for t in pt if t['entry_d'].year==test_yr]
        ft_yr = [t for t in ft if t['entry_d'].year==test_yr]
        all_pair_trades.extend(pt_yr)
        all_factor_trades.extend(ft_yr)

        if pt_yr or ft_yr:
            pwr = sum(1 for t in pt_yr if t['pnl']>0)/max(len(pt_yr),1)*100
            fwr = sum(1 for t in ft_yr if t['pnl']>0)/max(len(ft_yr),1)*100
            pavg = np.mean([t['pnl'] for t in pt_yr])*100 if pt_yr else 0
            favg = np.mean([t['pnl'] for t in ft_yr])*100 if ft_yr else 0
            print(f"  {test_yr}: pairs={len(pt_yr)}(WR={pwr:.0f}%/{pavg:.2f}%) "
                  f"factor={len(ft_yr)}(WR={fwr:.0f}%/{favg:.2f}%)")
        else:
            print(f"  {test_yr}: no trades")

        start_yr += 1

    # Combined analysis
    print(f"\n--- Walk-Forward Results ---")
    if all_pair_trades:
        pnls = [t['pnl'] for t in all_pair_trades]
        wr = sum(1 for p in pnls if p>0)/len(pnls)*100
        cum = np.prod([1+p for p in pnls])-1
        print(f"  Pairs: {len(pnls)} trades, WR={wr:.1f}%, Cum={cum*100:.1f}%")
    if all_factor_trades:
        pnls = [t['pnl'] for t in all_factor_trades]
        wr = sum(1 for p in pnls if p>0)/len(pnls)*100
        avg = np.mean(pnls)*100
        print(f"  Factor: {len(pnls)} trades, WR={wr:.1f}%, Avg={avg:.3f}%")

    return all_pair_trades, all_factor_trades


# ============================================================
# MAIN
# ============================================================
def main():
    print("="*60)
    print("  V304: PAIR TRADING + REGIME FACTOR FUSION")
    print("="*60)

    C, O, H, L, V, OI, NS, ND, dates, syms, sym_idx = load_all_data(start='2016-01-01')

    # Full backtest: sweep pair parameters
    print("\n--- Pair Parameter Sweep ---")
    best_pair = None; best_sharpe = -999
    for zt in [0.6, 0.8, 1.0, 1.2]:
        for hm in [1, 3, 5]:
            for lb in [10, 15, 20]:
                trades = run_pairs(C, O, NS, ND, dates, syms, sym_idx,
                                   z_thresh=zt, hold_max=hm, lookback=lb)
                if len(trades) < 50: continue
                pnls = [t['pnl'] for t in trades]
                wr = sum(1 for p in pnls if p>0)/len(pnls)*100
                cum = np.prod([1+p for p in pnls])-1
                yrs = (dates[-1]-dates[0]).days/365.25
                ann = (1+cum)**(1/max(yrs,.1))-1 if cum>-1 else -1
                avg = np.mean(pnls)*100
                eq_c = np.cumprod([1+p for p in pnls])
                pk = np.maximum.accumulate(eq_c)
                mdd = np.max((pk-eq_c)/pk)*100
                sh = avg/max(np.std(pnls)*100, 0.01)
                if sh > best_sharpe:
                    best_sharpe = sh
                    best_pair = {'zt':zt, 'hm':hm, 'lb':lb, 'wr':wr, 'ann':ann*100, 'sh':sh, 'n':len(trades)}
                print(f"  zt={zt} hm={hm} lb={lb}: n={len(trades)} WR={wr:.1f}% ann={ann*100:.1f}% sh={sh:.2f}")

    print(f"\n  Best pair: zt={best_pair['zt']} hm={best_pair['hm']} lb={best_pair['lb']} "
          f"(WR={best_pair['wr']:.1f}% ann={best_pair['ann']:.1f}%)")

    # Run fusion
    run_fusion(C, O, H, L, V, OI, NS, ND, dates, syms, sym_idx,
               pair_alloc=0.5, factor_alloc=0.5,
               pair_params={'z_thresh': best_pair['zt'], 'hold_max': best_pair['hm'],
                           'lookback': best_pair['lb']})

    # Walk-forward
    walk_forward(C, O, H, L, V, OI, NS, ND, dates, syms, sym_idx)

    print("\n[V304] Done.")


if __name__ == '__main__':
    main()
