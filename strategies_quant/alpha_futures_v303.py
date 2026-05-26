"""
V303: V301 regime signal + V302 leverage backtest
Concentrated long-only with leverage 1-5x
"""
import sys, os, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from alpha_futures_v301 import (load_all_data, compute_factors, detect_regimes,
                                  generate_signals)
from alpha_futures_v302 import backtest_daily as bt_lev, analyze

COMMISSION = 0.0005
CASH0 = 1_000_000


def backtest_leveraged(signal, C, O, H, L, NS, ND, dates, syms, regime,
                       top_n=5, hold_days=5, atr_stop=2.5, leverage=1.0):
    """V301 signal with V302's leverage-enabled backtest. Long-only."""
    max_pos = top_n
    pos_alloc = leverage / max_pos

    equity = CASH0; peak = CASH0
    equity_curve = np.full(ND, np.nan); equity_curve[0] = equity
    positions = []; trades = []

    for di in range(1, ND):
        daily_pnl = 0
        new_positions = []
        for si, edi, ep, sp, d, a in positions:
            c = C[si,di]
            if np.isnan(c):
                new_positions.append((si,edi,ep,sp,d,a)); continue
            exit_r = None
            if d>0 and c<sp: exit_r='stop'
            elif di-edi>=hold_days: exit_r='hold'
            if exit_r:
                pnl = d*(c-ep)/ep - COMMISSION
                profit = equity * a * pnl
                daily_pnl += profit
                trades.append({'sym':syms[si],'entry_d':dates[edi],'exit_d':dates[di],
                    'dir':d,'entry':ep,'exit':c,'pnl':pnl,'profit':profit,
                    'reason':exit_r,'hold':di-edi,'regime':regime[edi]})
            else:
                new_positions.append((si,edi,ep,sp,d,a))
        positions = new_positions
        equity += daily_pnl; equity_curve[di] = equity
        if equity > peak: peak = equity
        if equity <= 0: equity_curve[di:] = 0; break
        if di < 60: continue

        held = {p[0] for p in positions}
        if len(positions) >= max_pos: continue

        # Regime-aware sizing
        r = regime[di]
        size_mult = {1:1.0, 0:0.8, -1:0.6, 2:0.3}.get(r, 0.5)
        alloc = pos_alloc * size_mult

        sig_vals = [(signal[si,di], si) for si in range(NS)
                    if not np.isnan(signal[si,di]) and si not in held
                    and not np.isnan(C[si,di]) and not np.isnan(O[si,di])
                    and signal[si,di] > 0.15]
        if not sig_vals: continue
        sig_vals.sort(key=lambda x: x[0], reverse=True)

        for score, si in sig_vals[:top_n]:
            if len(positions) >= max_pos: break
            if si in held: continue
            op = O[si,di]
            if np.isnan(op) or op<=0: continue
            atr_v = []
            for j in range(max(60,di-14),di):
                hh,ll,cc = H[si,j],L[si,j],C[si,j]
                if not any(np.isnan([hh,ll,cc])): atr_v.append(max(hh-ll,abs(hh-cc),abs(ll-cc)))
            if not atr_v: continue
            atr = np.mean(atr_v)
            positions.append((si, di, op, op-atr_stop*atr, 1, alloc))
            held.add(si)

    for si,edi,ep,sp,d,a in positions:
        c=C[si,ND-1]
        if not np.isnan(c):
            pnl=d*(c-ep)/ep-COMMISSION
            trades.append({'sym':syms[si],'entry_d':dates[edi],'exit_d':dates[ND-1],
                'dir':d,'entry':ep,'exit':c,'pnl':pnl,'profit':equity*a*pnl,
                'reason':'end','hold':ND-1-edi,'regime':regime[edi]})
    return trades, equity_curve


def walk_forward(C,O,H,L,V,OI,NS,ND,dates,syms,
                 train_years=4, test_years=1, top_n=5, hold_days=5,
                 atr_stop=2.5, leverage=1.0):
    years = sorted(set(d.year for d in dates))
    all_trades = []; start_yr = years[0]
    print(f"\nWF: train={train_years}y test={test_years}y tn={top_n} h={hold_days} "
          f"atr={atr_stop} lev={leverage}")

    while True:
        train_end = start_yr+train_years-1; test_yr = train_end+1
        if test_yr > years[-1]: break
        train_m = np.array([d.year<=train_end for d in dates])
        test_m = np.array([d.year==test_yr for d in dates])
        t0 = max(0, np.where(train_m)[0][0]-60)
        sl = slice(t0, np.where(test_m)[0][-1]+1)

        C_s,O_s,H_s,L_s = C[:,sl],O[:,sl],H[:,sl],L[:,sl]
        V_s,OI_s = V[:,sl],OI[:,sl]
        d_s=dates[sl]; ND_s=len(d_s)

        F=compute_factors(C_s,O_s,H_s,L_s,V_s,OI_s,NS,ND_s)
        regime=detect_regimes(F,NS,ND_s)
        signal,_,_,_=generate_signals(F,regime,C_s,NS,ND_s,syms)
        trades,eq=backtest_leveraged(signal,C_s,O_s,H_s,L_s,NS,ND_s,d_s,syms,regime,
                                      top_n=top_n,hold_days=hold_days,
                                      atr_stop=atr_stop,leverage=leverage)
        yr_t=[t for t in trades if t['exit_d'].year==test_yr]
        all_trades.extend(yr_t)
        if yr_t:
            pnls=[t['pnl'] for t in yr_t]
            wr=sum(1 for p in pnls if p>0)/len(pnls)*100
            print(f"  {test_yr}: {len(pnls)} trades, WR={wr:.1f}%, avg={np.mean(pnls)*100:.3f}%",
                  flush=True)
        else: print(f"  {test_yr}: no trades", flush=True)
        start_yr+=1

    if all_trades:
        eq_a=np.ones(len(all_trades)+1)*CASH0
        alloc=leverage/top_n
        for i,t in enumerate(sorted(all_trades,key=lambda x:x['exit_d'])):
            eq_a[i+1]=eq_a[i]*(1+t['pnl']*alloc)
        dates_wf=[t['exit_d'] for t in sorted(all_trades,key=lambda x:x['exit_d'])]
        analyze(all_trades,eq_a,dates_wf,f"WF-lev{leverage}-tn{top_n}")
    return all_trades


def main():
    print("="*60)
    print("  V303: REGIME SIGNAL + LEVERAGE (CONCENTRATED LONG)")
    print("="*60)
    C,O,H,L,V,OI,NS,ND,dates,syms = load_all_data(start='2016-01-01')
    F=compute_factors(C,O,H,L,V,OI,NS,ND)
    regime=detect_regimes(F,NS,ND)
    signal,_,_,_=generate_signals(F,regime,C,NS,ND,syms)

    # Full backtest sweep
    print("\n--- Full Backtest Sweep ---")
    results=[]
    for lev in [1.0, 2.0, 3.0, 5.0]:
        for tn in [3, 5]:
            for hold in [3, 5, 10]:
                for atr in [2.5, 3.5]:
                    trades,eq=backtest_leveraged(signal,C,O,H,L,NS,ND,dates,syms,regime,
                                                 top_n=tn,hold_days=hold,
                                                 atr_stop=atr,leverage=lev)
                    if not trades: continue
                    eq_v=eq[~np.isnan(eq)]
                    if len(eq_v)<2: continue
                    tr=eq_v[-1]/eq_v[0]-1
                    yrs=(dates[-1]-dates[0]).days/365.25
                    ann=(1+tr)**(1/max(yrs,.1))-1 if tr>-1 else -1
                    pk=np.maximum.accumulate(eq_v)
                    mdd=np.max((pk-eq_v)/pk*100)
                    ec=np.diff(eq_v)/eq_v[:-1]
                    sh=np.mean(ec)/max(np.std(ec),1e-10)*np.sqrt(252) if len(ec)>10 else 0
                    pnls=np.array([t['pnl'] for t in trades])
                    wr=(pnls>0).sum()/len(pnls)*100
                    results.append({'lev':lev,'tn':tn,'h':hold,'atr':atr,
                        'n':len(trades),'wr':wr,'ann':ann*100,'mdd':mdd,'sh':sh})

    results.sort(key=lambda x:-x['sh'])
    print(f"\n{'Lev':>4} {'TN':>3} {'H':>3} {'ATR':>4} {'Tr':>5} {'WR':>5} {'Ann%':>7} {'MDD':>6} {'Sh':>6}")
    print("-"*55)
    for r in results[:15]:
        print(f"{r['lev']:>4.1f} {r['tn']:>3} {r['h']:>3} {r['atr']:>4.1f} "
              f"{r['n']:>5} {r['wr']:>5.1f} {r['ann']:>7.1f} {r['mdd']:>6.1f} {r['sh']:>6.2f}")

    # Walk-forward for top 3 configs
    for r in results[:3]:
        walk_forward(C,O,H,L,V,OI,NS,ND,dates,syms,
                    top_n=r['tn'],hold_days=r['h'],
                    atr_stop=r['atr'],leverage=r['lev'])

    # Also WF at extreme leverage
    print("\n--- Extreme Leverage WF ---")
    for lev in [3.0, 5.0, 8.0]:
        walk_forward(C,O,H,L,V,OI,NS,ND,dates,syms,
                    top_n=3, hold_days=5, atr_stop=3.5, leverage=lev)

    print("\n[V303] Done.")


if __name__ == '__main__':
    main()
