"""
V302: 高杠杆 + 排名信号 + 做多做空
====================================
核心: 纯跨品种排名做多做空，无方向偏好，高杠杆复利

1. 计算15个因子 → 跨品种排名(0~1)
2. 滚动IC加权合成信号 (rank-based, 零均值)
3. 做多top N, 做空bottom N (保证对称)
4. 杠杆倍数可调 (1x~10x)
5. Day-by-day权益曲线
6. Walk-forward验证
"""
import sys, os, time, warnings
import numpy as np
import pandas as pd
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


def load_all_data(start='2016-01-01', end=None, min_days=500):
    print("[V302] Loading data...", flush=True)
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
    if end:
        i1 = next((i for i, d in enumerate(all_dates) if d > pd.Timestamp(end)), len(all_dates)) - 1
    else:
        i1 = len(all_dates) - 1
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
        if len(common) == 0: continue
        idx = np.array([dm[d] for d in common])
        for col, arr in [('close',C),('open',O),('high',H),('low',L),('volume',V),('oi',OI)]:
            if col in df.columns: arr[si, idx] = df.loc[common, col].values.astype(float)

    C=C[:,i0:i1+1]; O=O[:,i0:i1+1]; H=H[:,i0:i1+1]
    L=L[:,i0:i1+1]; V=V[:,i0:i1+1]; OI=OI[:,i0:i1+1]
    print(f"  {NS} sym, {ND} days ({time.time()-t0:.1f}s)")
    return C, O, H, L, V, OI, NS, ND, dates, syms


def compute_factors(C, O, H, L, V, OI, NS, ND):
    print("[V302] Computing factors...", flush=True)
    t0 = time.time()
    F = {}
    for si in range(NS):
        c=C[si]; o=O[si]; h=H[si]; l=L[si]; v=V[si]; oi=OI[si]
        nc = np.isnan(c)

        # Momentum
        for p, nm in [(5,'mom5'),(10,'mom10'),(20,'mom20'),(60,'mom60')]:
            if nm not in F: F[nm] = np.full((NS,ND), np.nan)
            for di in range(p, ND):
                if not nc[di] and not nc[di-p] and c[di-p]>0:
                    F[nm][si,di] = c[di]/c[di-p]-1

        # ADX
        if 'adx' not in F: F['adx'] = np.full((NS,ND), np.nan)
        if HAS_TALIB:
            try:
                hs=np.where(np.isnan(h),0,h).astype(np.float64)
                ls=np.where(np.isnan(l),0,l).astype(np.float64)
                cs=np.where(nc,0,c).astype(np.float64)
                F['adx'][si] = np.where(nc, np.nan, talib.ADX(hs,ls,cs,14))
            except: pass

        # Trend slope
        if 'slope' not in F: F['slope'] = np.full((NS,ND), np.nan)
        for di in range(20, ND):
            w=c[di-20:di]; vv=w[~np.isnan(w)]
            if len(vv)>=15 and vv[0]>0:
                x=np.arange(len(vv)); y=vv/vv[0]
                try: F['slope'][si,di] = np.polyfit(x,y,1)[0]*252
                except: pass

        # Vol 20d
        if 'vol20' not in F: F['vol20'] = np.full((NS,ND), np.nan)
        for di in range(20, ND):
            r=[]
            for j in range(max(1,di-20),di):
                if not nc[j] and not nc[j-1] and c[j-1]>0: r.append(c[j]/c[j-1]-1)
            if len(r)>=10: F['vol20'][si,di] = np.std(r)*np.sqrt(252)

        # ATR%
        if 'atrp' not in F: F['atrp'] = np.full((NS,ND), np.nan)
        if HAS_TALIB:
            try:
                hs=np.where(np.isnan(h),0,h).astype(np.float64)
                ls=np.where(np.isnan(l),0,l).astype(np.float64)
                cs=np.where(nc,0,c).astype(np.float64)
                atr=talib.ATR(hs,ls,cs,14)
                vld=~nc&(c>0)
                F['atrp'][si]=np.where(vld, atr/np.where(c>0,c,1), np.nan)
            except: pass

        # Volume anomaly
        if 'vanom' not in F: F['vanom'] = np.full((NS,ND), np.nan)
        for di in range(60, ND):
            vw=v[di-60:di]; vv=vw[~np.isnan(vw)]
            if len(vv)>=30 and not np.isnan(v[di]):
                mu,sig=np.mean(vv),np.std(vv)
                if sig>0: F['vanom'][si,di]=(v[di]-mu)/sig

        # OI change 5d
        if 'oi5' not in F: F['oi5'] = np.full((NS,ND), np.nan)
        for di in range(5, ND):
            if not np.isnan(oi[di]) and not np.isnan(oi[di-5]) and oi[di-5]>0:
                F['oi5'][si,di]=oi[di]/oi[di-5]-1

        # RSI
        if 'rsi' not in F: F['rsi'] = np.full((NS,ND), np.nan)
        if HAS_TALIB:
            try:
                cs=np.where(nc,0,c).astype(np.float64)
                F['rsi'][si]=np.where(nc, np.nan, talib.RSI(cs,14))
            except: pass

        # Z-score 20d
        if 'zscore' not in F: F['zscore'] = np.full((NS,ND), np.nan)
        for di in range(20, ND):
            w=c[di-20:di]; vv=w[~np.isnan(w)]
            if len(vv)>=15 and np.std(vv)>0 and not nc[di]:
                F['zscore'][si,di]=(c[di]-np.mean(vv))/np.std(vv)

        # Body ratio
        if 'body' not in F: F['body'] = np.full((NS,ND), np.nan)
        for di in range(1, ND):
            ohlc=[o[di],h[di],l[di],c[di]]
            if not any(np.isnan(ohlc)) and h[di]>l[di]:
                F['body'][si,di]=abs(c[di]-o[di])/(h[di]-l[di])

    # Cross-sectional ranks
    for name in ['mom20','vol20','mom5','mom60']:
        cs = np.full((NS,ND), np.nan)
        for di in range(60,ND):
            vals=F[name][:,di]; vld=~np.isnan(vals)
            if vld.sum()>5: cs[:,di]=pd.Series(vals).rank(pct=True,na_option='keep').values
        F[f'cs_{name}'] = cs

    print(f"  {len(F)} factors in {time.time()-t0:.1f}s")
    return F


def compute_signal(F, C, NS, ND, ic_window=120, fwd_period=5):
    """
    Rank-based signal with rolling IC weighting.
    Returns a cross-sectional score centered around 0.
    """
    print("[V302] Computing rank-based IC signal...", flush=True)
    t0 = time.time()

    # Use a subset of factors that are rank-based
    factor_names = ['cs_mom20','cs_mom5','cs_mom60','cs_vol20',
                    'slope','adx','vanom','oi5','rsi','zscore','body','vol20','atrp']
    # Remove cs_ prefix factors from raw, use them as-is
    # For raw factors, compute cross-sectional rank first
    F_rank = {}
    for name in factor_names:
        if name.startswith('cs_'):
            F_rank[name] = F[name].copy()
        else:
            rank_arr = np.full((NS,ND), np.nan)
            for di in range(ND):
                vals = F[name][:,di]; vld=~np.isnan(vals)
                if vld.sum()>5:
                    rank_arr[:,di] = pd.Series(vals).rank(pct=True, na_option='keep').values
            F_rank[name] = rank_arr

    # Forward returns for IC
    fwd = np.full((NS,ND), np.nan)
    for si in range(NS):
        for di in range(ND-fwd_period):
            if C[si,di]>0 and not np.isnan(C[si,di]) and not np.isnan(C[si,di+fwd_period]):
                fwd[si,di] = C[si,di+fwd_period]/C[si,di]-1

    NF = len(factor_names)
    signal = np.full((NS,ND), np.nan)

    for di in range(ic_window, ND):
        # Compute IC for each factor
        ics = np.zeros(NF)
        for fi, name in enumerate(factor_names):
            fv = F_rank[name][:, di-ic_window:di].flatten()
            rv = fwd[:, di-ic_window:di].flatten()
            vld = ~np.isnan(fv) & ~np.isnan(rv)
            if vld.sum() > 300:
                fr = fv[vld]; rr = pd.Series(rv[vld]).rank().values
                if np.std(fr)>0 and np.std(rr)>0:
                    ic = np.corrcoef(fr, rr)[0,1]
                    if not np.isnan(ic): ics[fi] = ic

        # Shrink IC toward equal weight (regularization)
        shrink = 0.5
        ics = shrink * ics + (1-shrink) * np.sign(ics) * np.mean(np.abs(ics))

        # Only use significant factors
        sig_mask = np.abs(ics) > 0.005
        if sig_mask.sum() < 3: continue

        weights = np.zeros(NF)
        weights[sig_mask] = ics[sig_mask]
        wsum = np.sum(np.abs(weights))
        if wsum > 0: weights /= wsum

        # Score each instrument (rank-based, should be symmetric)
        for si in range(NS):
            score = 0; n = 0
            for fi, name in enumerate(factor_names):
                val = F_rank[name][si, di]
                if not np.isnan(val) and sig_mask[fi]:
                    score += weights[fi] * val
                    n += 1
            if n >= 3:
                signal[si, di] = score

    # CENTER signal cross-sectionally → ensures balanced L/S
    for di in range(ic_window, ND):
        vals = signal[:, di]
        valid = vals[~np.isnan(vals)]
        if len(valid) > 5:
            median = np.median(valid)
            for si in range(NS):
                if not np.isnan(signal[si, di]):
                    signal[si, di] -= median

    print(f"  Signal done in {time.time()-t0:.1f}s")
    return signal


def backtest_daily(signal, C, O, H, L, NS, ND, dates, syms,
                   top_n=5, hold_days=5, atr_stop=2.5, leverage=1.0):
    """
    Day-by-day backtest with leverage.
    leverage=1.0 means 100% equity per position (very aggressive).
    leverage=0.2 means 20% equity per position (conservative).
    """
    max_pos = top_n * 2  # always long+short
    pos_alloc = leverage / max_pos  # fraction of equity per position

    equity = CASH0
    peak = CASH0
    equity_curve = np.full(ND, np.nan)
    equity_curve[0] = equity
    positions = []
    trades = []

    for di in range(1, ND):
        # Exit logic
        daily_pnl = 0
        new_positions = []
        for si, edi, ep, sp, d, a in positions:
            c = C[si,di]
            if np.isnan(c):
                new_positions.append((si,edi,ep,sp,d,a)); continue

            exit_r = None
            if d>0 and c<sp: exit_r='stop'
            elif d<0 and c>sp: exit_r='stop'
            elif di-edi>=hold_days: exit_r='hold'

            if exit_r:
                pnl = d*(c-ep)/ep - COMMISSION
                profit = equity * a * pnl
                daily_pnl += profit
                trades.append({
                    'sym':syms[si], 'entry_d':dates[edi], 'exit_d':dates[di],
                    'dir':d, 'entry':ep, 'exit':c, 'pnl':pnl,
                    'profit':profit, 'reason':exit_r, 'hold':di-edi
                })
            else:
                new_positions.append((si,edi,ep,sp,d,a))
        positions = new_positions

        equity += daily_pnl
        equity_curve[di] = equity
        if equity > peak: peak = equity
        if equity <= 0:
            equity_curve[di:] = 0; break

        if di < 60: continue
        held = {p[0] for p in positions}
        if len(positions) >= max_pos: continue

        # Count how many long and short slots we need to fill
        n_long = sum(1 for _,_,_,_,d,_ in positions if d > 0)
        n_short = sum(1 for _,_,_,_,d,_ in positions if d < 0)
        need_long = max(0, top_n - n_long)
        need_short = max(0, top_n - n_short)

        # Signal ranking
        sig_vals = [(signal[si,di], si) for si in range(NS)
                    if not np.isnan(signal[si,di]) and si not in held
                    and not np.isnan(C[si,di]) and not np.isnan(O[si,di])]
        if len(sig_vals) < top_n: continue
        sig_vals.sort(key=lambda x: x[0], reverse=True)

        def enter_position(si, direction):
            op = O[si,di]
            if np.isnan(op) or op<=0: return False
            atr_v = []
            for j in range(max(60,di-14),di):
                hh,ll,cc = H[si,j],L[si,j],C[si,j]
                if not any(np.isnan([hh,ll,cc])): atr_v.append(max(hh-ll,abs(hh-cc),abs(ll-cc)))
            if not atr_v: return False
            atr = np.mean(atr_v)
            if direction > 0:
                positions.append((si, di, op, op-atr_stop*atr, 1, pos_alloc))
            else:
                positions.append((si, di, op, op+atr_stop*atr, -1, pos_alloc))
            held.add(si)
            return True

        # Alternate long/short entry to maintain balance
        long_cands = [si for _,si in sig_vals[:top_n*2] if si not in held]
        short_cands = [si for _,si in sig_vals[-top_n*2:] if si not in held]
        # Use signal sign to determine direction: positive → long, negative → short
        for si in long_cands:
            if need_long <= 0 or len(positions) >= max_pos: break
            if signal[si,di] > 0 and enter_position(si, 1):
                need_long -= 1
        for si in short_cands:
            if need_short <= 0 or len(positions) >= max_pos: break
            if signal[si,di] < 0 and enter_position(si, -1):
                need_short -= 1

    # Close remaining
    for si,edi,ep,sp,d,a in positions:
        c=C[si,ND-1]
        if not np.isnan(c):
            pnl=d*(c-ep)/ep-COMMISSION
            trades.append({
                'sym':syms[si],'entry_d':dates[edi],'exit_d':dates[ND-1],
                'dir':d,'entry':ep,'exit':c,'pnl':pnl,'profit':equity*a*pnl,
                'reason':'end','hold':ND-1-edi
            })
    return trades, equity_curve


def analyze(trades, equity_curve, dates, label=""):
    if not trades:
        print(f"  [{label}] No trades"); return {}
    pnls = np.array([t['pnl'] for t in trades])
    n = len(pnls)
    wr = (pnls>0).sum()/n*100
    avg = np.mean(pnls)*100

    eq = equity_curve[~np.isnan(equity_curve)]
    if len(eq)<2: return {}
    total_ret = eq[-1]/eq[0]-1
    yrs = (dates[-1]-dates[0]).days/365.25
    ann = (1+total_ret)**(1/max(yrs,.1))-1 if total_ret>-1 else -1
    pk = np.maximum.accumulate(eq)
    dd = (pk-eq)/pk*100
    mdd = np.max(dd)

    long_t = [t for t in trades if t['dir']>0]
    short_t = [t for t in trades if t['dir']<0]
    gw = sum(p for p in pnls if p>0)
    gl = abs(sum(p for p in pnls if p<0))
    pf = gw/max(gl,1e-10)
    eq_c = np.diff(eq)/eq[:-1]
    sharpe = np.mean(eq_c)/max(np.std(eq_c),1e-10)*np.sqrt(252) if len(eq_c)>10 else 0

    print(f"\n{'='*60}")
    print(f"  [{label}]")
    print(f"{'='*60}")
    print(f"  Trades: {n} (L:{len(long_t)} S:{len(short_t)})")
    print(f"  WR: {wr:.1f}% | Avg: {avg:.3f}%")
    print(f"  Total: {total_ret*100:.1f}% | Ann: {ann*100:.1f}%")
    print(f"  MDD: {mdd:.1f}% | PF: {pf:.2f} | Sharpe: {sharpe:.2f}")
    print(f"  Final: {eq[-1]:,.0f}")

    # Per-year
    print(f"  Per Year:")
    for yr in sorted(set(t['exit_d'].year for t in trades)):
        yt = [t for t in trades if t['exit_d'].year==yr]
        ywr = sum(1 for t in yt if t['pnl']>0)/len(yt)*100
        print(f"    {yr}: {len(yt):>4} trades, WR={ywr:.1f}%")

    return {'n':n,'wr':wr,'ann':ann,'mdd':mdd,'sharpe':sharpe,'pf':pf}


def walk_forward(C, O, H, L, V, OI, NS, ND, dates, syms,
                 train_years=4, test_years=1, top_n=5, hold_days=5,
                 atr_stop=2.5, leverage=1.0):
    print(f"\n{'='*60}")
    print(f"  WALK-FORWARD train={train_years}y test={test_years}y lev={leverage}")
    print(f"{'='*60}")

    years = sorted(set(d.year for d in dates))
    all_trades = []
    start_yr = years[0]

    while True:
        train_end = start_yr + train_years - 1
        test_yr = train_end + 1
        if test_yr > years[-1]: break

        train_m = np.array([d.year <= train_end for d in dates])
        test_m = np.array([d.year == test_yr for d in dates])
        t0 = max(0, np.where(train_m)[0][0] - 60)
        sl = slice(t0, np.where(test_m)[0][-1]+1)

        C_s,O_s,H_s,L_s = C[:,sl],O[:,sl],H[:,sl],L[:,sl]
        V_s,OI_s = V[:,sl],OI[:,sl]
        d_s = dates[sl]; ND_s = len(d_s)

        F = compute_factors(C_s,O_s,H_s,L_s,V_s,OI_s,NS,ND_s)
        sig = compute_signal(F,C_s,NS,ND_s)
        trades,eq = backtest_daily(sig,C_s,O_s,H_s,L_s,NS,ND_s,d_s,syms,
                                   top_n=top_n,hold_days=hold_days,
                                   atr_stop=atr_stop,leverage=leverage)

        yr_t = [t for t in trades if t['exit_d'].year==test_yr]
        all_trades.extend(yr_t)

        if yr_t:
            pnls=[t['pnl'] for t in yr_t]
            wr=sum(1 for p in pnls if p>0)/len(pnls)*100
            print(f"  {test_yr}: {len(pnls)} trades, WR={wr:.1f}%, avg={np.mean(pnls)*100:.3f}%",
                  flush=True)
        else:
            print(f"  {test_yr}: no trades", flush=True)
        start_yr += 1

    if all_trades:
        eq_arr = np.ones(len(all_trades)+1)*CASH0
        max_pos = top_n*2
        alloc = leverage/max_pos
        for i,t in enumerate(sorted(all_trades, key=lambda x:x['exit_d'])):
            eq_arr[i+1] = eq_arr[i]*(1+t['pnl']*alloc)
        dates_wf = [t['exit_d'] for t in sorted(all_trades, key=lambda x:x['exit_d'])]
        analyze(all_trades, eq_arr, dates_wf, f"WF-lev{leverage}")

    return all_trades


def main():
    print("="*60)
    print("  V302: RANK-BASED L/S WITH LEVERAGE")
    print("="*60)

    C,O,H,L,V,OI,NS,ND,dates,syms = load_all_data(start='2016-01-01')

    # Full backtest at different leverage levels
    F = compute_factors(C,O,H,L,V,OI,NS,ND)
    sig = compute_signal(F,C,NS,ND)

    for lev in [0.5, 1.0, 2.0]:
        trades, eq = backtest_daily(sig,C,O,H,L,NS,ND,dates,syms,
                                     top_n=5,hold_days=5,atr_stop=2.5,leverage=lev)
        analyze(trades,eq,dates,f"Full-lev{lev}")

    # Sweep parameters
    print("\n" + "="*60)
    print("  PARAMETER SWEEP")
    print("="*60)
    results = []
    for top_n in [3,5]:
        for hold in [3,5,10]:
            for atr in [2.0,3.0]:
                for lev in [0.5,1.0,2.0]:
                    trades,eq = backtest_daily(sig,C,O,H,L,NS,ND,dates,syms,
                                               top_n=top_n,hold_days=hold,
                                               atr_stop=atr,leverage=lev)
                    if not trades: continue
                    pnls=np.array([t['pnl'] for t in trades])
                    eq_v=eq[~np.isnan(eq)]
                    if len(eq_v)<2: continue
                    tr=eq_v[-1]/eq_v[0]-1
                    yrs=(dates[-1]-dates[0]).days/365.25
                    ann=(1+tr)**(1/max(yrs,.1))-1 if tr>-1 else -1
                    pk=np.maximum.accumulate(eq_v)
                    mdd=np.max((pk-eq_v)/pk*100)
                    ec=np.diff(eq_v)/eq_v[:-1]
                    sh=np.mean(ec)/max(np.std(ec),1e-10)*np.sqrt(252) if len(ec)>10 else 0
                    results.append({'tn':top_n,'h':hold,'atr':atr,'lev':lev,
                                    'n':len(pnls),'wr':(pnls>0).sum()/len(pnls)*100,
                                    'ann':ann*100,'mdd':mdd,'sh':sh})

    results.sort(key=lambda x:-x['sh'])
    print(f"\n{'TN':>3} {'H':>3} {'ATR':>4} {'Lev':>5} {'Tr':>5} {'WR':>5} {'Ann%':>7} {'MDD':>6} {'Sh':>6}")
    print("-"*60)
    for r in results[:20]:
        print(f"{r['tn']:>3} {r['h']:>3} {r['atr']:>4} {r['lev']:>5.1f} "
              f"{r['n']:>5} {r['wr']:>5.1f} {r['ann']:>7.1f} {r['mdd']:>6.1f} {r['sh']:>6.2f}")

    # Walk-forward for best configs
    if results:
        for r in results[:3]:
            print(f"\n--- WF: top_n={r['tn']}, hold={r['h']}, atr={r['atr']}, lev={r['lev']} ---")
            walk_forward(C,O,H,L,V,OI,NS,ND,dates,syms,
                        top_n=r['tn'],hold_days=r['h'],
                        atr_stop=r['atr'],leverage=r['lev'])

    print("\n[V302] Done.")


if __name__ == '__main__':
    main()
