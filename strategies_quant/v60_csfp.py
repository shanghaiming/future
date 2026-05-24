"""
V60 — Cross-Sectional Factor Portfolio (CSFP) v3
===============================================
从第一性原理: 入场信号(TIMING) + 选股排名(SELECTION)

核心洞察 (来自深度学习):
  1. LEVEL信号(MOM20>0)是描述性的, 不是预测性的 → 单独使用=追高=亏
  2. FLIP信号/共识信号是预测性的 → V53的9策略共识就是这类信号
  3. 排名方法(RANKING)决定选哪只股, 不决定何时买入

实验设计:
  A: V53共识(TIMING) + MOM20排名(SELECTION) = V53 baseline
  B: V53共识(TIMING) + 3因子百分位排名(SELECTION) = 改进选股
  C: V53共识(TIMING) + 3因子百分位排名 + 做空 = 完整系统
  D: 3因子FLIP(TIMING) + 3因子百分位排名(SELECTION) = 纯新系统

3因子 (独立家族, Section 12 Fisher正交):
  MOM20: 方向家族 → percentile
  VDP_Delta: 成交量家族 → percentile
  KER: 效率家族 → percentile
"""
import sys, os, time, pickle, warnings
import numpy as np, pandas as pd
from collections import defaultdict
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.data_loader import list_available_symbols, load_stock_data

COMMISSION = 0.0003; STAMP_DUTY = 0.001; CASH0 = 500_000
USE_STRATS = {
    'HanningFIRStrategy', 'SpikeBakeStrategy', 'QuadBBFusionStrategy',
    'IndexStrategy', 'MathAnalysisStrategy', 'EnergtStructureStrategy',
    'RegressionCandlestickStrategy', 'ConservativeMAStrategy',
    'OptimizedMASimpleStrategy',
}

print("=" * 70, flush=True)
print("  V60 — CSFP v3: Consensus TIMING + 3-Factor SELECTION", flush=True)
print("=" * 70, flush=True)

# ============================================================
# [1] 数据加载
# ============================================================
print("\n[1] Loading data...", flush=True)
t0 = time.time()

stock_data = {}
for sym in list_available_symbols('daily'):
    try:
        df = load_stock_data(sym, frequency='daily')
        if df is not None and len(df) >= 300:
            cols = [c for c in ['open','high','low','close','vol','volume','amount'] if c in df.columns]
            stock_data[sym] = df[cols].copy()
            if 'vol' in df.columns and 'volume' not in df.columns:
                stock_data[sym].rename(columns={'vol': 'volume'}, inplace=True)
    except: pass

vol_map = {s: df['volume'].tail(60).mean() for s, df in stock_data.items()
           if 'volume' in df.columns and df['volume'].tail(60).mean() > 0}
syms = sorted([s for s, _ in sorted(vol_map.items(), key=lambda x: -x[1])[:500]])
NS = len(syms)
all_dates = sorted(set(d for s in syms for d in stock_data[s].index))
i0 = next(i for i, d in enumerate(all_dates) if d >= pd.Timestamp('2016-01-01'))
i1 = next((i for i, d in enumerate(all_dates) if d > pd.Timestamp('2026-04-25')), len(all_dates)) - 1
dates = all_dates[i0:i1+1]; ND = len(dates); dm = {d: i for i, d in enumerate(all_dates)}

C = np.full((NS, len(all_dates)), np.nan)
O = np.full((NS, len(all_dates)), np.nan)
H = np.full((NS, len(all_dates)), np.nan)
L = np.full((NS, len(all_dates)), np.nan)
V = np.full((NS, len(all_dates)), np.nan)
for si, s in enumerate(syms):
    df = stock_data.get(s)
    if df is None: continue
    for d in df.index:
        if d in dm:
            di = dm[d]
            if 'close' in df.columns: C[si, di] = float(df.loc[d, 'close'])
            if 'open' in df.columns: O[si, di] = float(df.loc[d, 'open'])
            if 'high' in df.columns: H[si, di] = float(df.loc[d, 'high'])
            if 'low' in df.columns: L[si, di] = float(df.loc[d, 'low'])
            if 'volume' in df.columns: V[si, di] = float(df.loc[d, 'volume'])
C=C[:,i0:i1+1]; O=O[:,i0:i1+1]; H=H[:,i0:i1+1]; L=L[:,i0:i1+1]; V=V[:,i0:i1+1]
print(f"  {NS} stocks, {ND} days ({time.time()-t0:.1f}s)", flush=True)

# ============================================================
# [2] 因子计算
# ============================================================
print("[2] Computing factors...", flush=True)
t2 = time.time()

# MOM20 + MOM5
MOM20 = np.full_like(C, np.nan); MOM5 = np.full_like(C, np.nan)
for si in range(NS):
    for di in range(20, ND):
        if not np.isnan(C[si,di]) and not np.isnan(C[si,di-20]) and C[si,di-20]>0:
            MOM20[si,di]=(C[si,di]-C[si,di-20])/C[si,di-20]
        if not np.isnan(C[si,di]) and not np.isnan(C[si,di-5]) and C[si,di-5]>0:
            MOM5[si,di]=(C[si,di]-C[si,di-5])/C[si,di-5]

# VDP Delta: V×(2C-H-L)/(H-L), EMA(10)
EMA_P=10; a_ema=2.0/(EMA_P+1)
VDP_DELTA=np.full_like(C,np.nan)
for si in range(NS):
    ema_val=np.nan
    for di in range(1,ND):
        if np.isnan(V[si,di]) or V[si,di]<=0: continue
        if np.isnan(C[si,di]) or np.isnan(H[si,di]) or np.isnan(L[si,di]): continue
        hl=H[si,di]-L[si,di]
        if hl<=0:
            delta = V[si,di] if C[si,di]>=H[si,di] else -V[si,di] if C[si,di]<=L[si,di] else None
            if delta is None: continue
        else:
            delta=V[si,di]*(2*C[si,di]-H[si,di]-L[si,di])/hl
        ema_val=delta if np.isnan(ema_val) else a_ema*delta+(1-a_ema)*ema_val
        VDP_DELTA[si,di]=ema_val

# KER: |net|/Σ|daily|, 20-period
KER=np.full_like(C,np.nan)
for si in range(NS):
    for di in range(20,ND):
        vals=C[si,di-20:di+1]; valid=vals[~np.isnan(vals)]
        if len(valid)<20: continue
        net=abs(valid[-1]-valid[0]); total=np.sum(np.abs(np.diff(valid)))
        if total>0: KER[si,di]=net/total

print(f"  MOM20, VDP, KER done ({time.time()-t2:.1f}s)", flush=True)

# --- 百分位排名 ---
print("[3] Percentile rankings...", flush=True)
def rank_pct(arr, start=60):
    res=np.full_like(arr,np.nan)
    for di in range(start, arr.shape[1]):
        vals=arr[:,di]; mask=~np.isnan(vals)
        if mask.sum()<50: continue
        ranked=np.argsort(np.argsort(vals[mask])).astype(float)
        n=len(ranked); pct=ranked/max(n-1,1)*100.0
        idxs=np.where(mask)[0]
        for k,idx in enumerate(idxs): res[idx,di]=pct[k]
    return res

PCT_MOM=rank_pct(MOM20); PCT_VDP=rank_pct(VDP_DELTA); PCT_KER=rank_pct(KER)
print(f"  Rankings done ({time.time()-t2:.1f}s)", flush=True)

# --- 市场指标 ---
MKT_RET=np.full(ND,np.nan)
for di in range(ND):
    r=[C[si,di]/C[si,di-1]-1 for si in range(NS)
       if di>0 and not np.isnan(C[si,di]) and not np.isnan(C[si,di-1]) and C[si,di-1]>0]
    if len(r)>100: MKT_RET[di]=np.mean(r)
MKT_CUM=np.nancumsum(np.where(np.isnan(MKT_RET),0,MKT_RET))
MKT_MOM20=np.full(ND,np.nan)
for di in range(20,ND):
    v=MKT_RET[di-20:di]; v=v[~np.isnan(v)]
    if len(v)>10: MKT_MOM20[di]=np.sum(v)
MKT_MA60=np.full(ND,np.nan)
for di in range(60,ND): MKT_MA60[di]=np.mean(MKT_CUM[di-60:di])

# --- V37共识信号 ---
print("[4] Loading V37 consensus signals...", flush=True)
with open('.v15_7_signals_fixed.pkl','rb') as f: all_signals=pickle.load(f)
date_to_di={d:i for i,d in enumerate(dates)}
buy_set=defaultdict(lambda: defaultdict(int))
sell_set=defaultdict(lambda: defaultdict(int))
for sname in USE_STRATS:
    if sname not in all_signals: continue
    for sym, sigs in all_signals[sname].items():
        if sym not in syms: continue
        si=syms.index(sym)
        for ts, action, price in sigs:
            if ts in date_to_di:
                di=date_to_di[ts]
                if action=='buy': buy_set[di][si]+=1
                elif action=='sell': sell_set[di][si]+=1


# ============================================================
# [5] 核心交易逻辑
# ============================================================
def run(selection, w_mom, w_vdp, w_ker, base_sl, tp_pct, hold_max, trail_pct,
        mode='long_only', short_sl=15, short_tp=30, short_hm=12, short_trail=5,
        start_idx=0, end_idx=ND-1):
    """
    selection:
      'v53'  — V53 baseline: consensus≥3 + MOM20 ranking
      '3f'   — consensus≥3 + 3-factor percentile ranking
      '3f_ls'— consensus≥3 + 3-factor ranking + short on sell consensus
      'flip' — 3-factor flip signals (no consensus) + 3-factor ranking
    """
    cash=float(CASH0); pos=None; trades=[]; pending=None

    for di in range(max(start_idx,60), end_idx+1):
        # --- Execute pending ---
        if pending is not None:
            pt=pending[0]
            if pt=='close' and pos is not None:
                p=O[pos['si'],di]
                if np.isnan(p) or p<=0: p=C[pos['si'],di]
                if not np.isnan(p) and p>0:
                    d=pos['direction']
                    if d=='long':
                        pnl=(p-pos['entry'])/pos['entry']*100
                        cash+=pos['shares']*p*(1-COMMISSION-STAMP_DUTY)
                    else:
                        pnl=(pos['entry']-p)/pos['entry']*100
                        S=pos['shares'];E=pos['entry']
                        cash+=S*(E-p)-S*E*(COMMISSION+STAMP_DUTY)-S*p*COMMISSION
                    trades.append({'pnl':pnl,'days':(dates[di]-pos['ed']).days,
                                  'reason':pending[1],'dir':d})
                    pos=None
            elif pt=='open_long' and pos is None:
                si=pending[1]; p=O[si,di]
                if np.isnan(p) or p<=0: p=C[si,di-1] if di>0 and not np.isnan(C[si,di-1]) else np.nan
                if not np.isnan(p) and p>0 and cash>10000:
                    shares=int(cash/(1+COMMISSION)/p)
                    if shares>0:
                        cash-=shares*p*(1+COMMISSION)
                        pos={'si':si,'shares':shares,'entry':p,'direction':'long',
                             'highest':p,'ed':dates[di],'buy_di':di}
            elif pt=='open_short' and pos is None:
                si=pending[1]; p=O[si,di]
                if np.isnan(p) or p<=0: p=C[si,di-1] if di>0 and not np.isnan(C[si,di-1]) else np.nan
                if not np.isnan(p) and p>0 and cash>10000:
                    shares=int(cash/p)
                    if shares>0:
                        pos={'si':si,'shares':shares,'entry':p,'direction':'short',
                             'lowest':p,'ed':dates[di],'buy_di':di}
            pending=None

        # --- Exit ---
        if pos is not None:
            si=pos['si']; p=C[si,di]
            if np.isnan(p): continue
            if pos['direction']=='long':
                if p>pos['highest']: pos['highest']=p
                pnl=(p-pos['entry'])/pos['entry']*100
                hd=(dates[di]-pos['ed']).days
                if hd<=3: sl_eff=base_sl*1.5
                elif hd<=7: sl_eff=base_sl
                else: sl_eff=base_sl*0.7
                er=None
                if pnl<-sl_eff: er=f'sl({pnl:.1f}%,d{hd})'
                elif pnl>tp_pct: er=f'tp({pnl:.1f}%)'
                elif trail_pct>0 and pnl>5:
                    dd=(pos['highest']-p)/pos['highest']*100
                    if dd>trail_pct: er=f'trail({pnl:.1f}%,dd{dd:.1f}%)'
                elif hold_max>0 and hd>=hold_max: er=f'max({hd}d,{pnl:.1f}%)'
                elif sell_set[di].get(si,0)>=2 and buy_set[di].get(si,0)==0 and pnl>0:
                    er=f'flip({pnl:.1f}%)'
                elif sell_set[di].get(si,0)>=1 and pnl>3: er=f'signal({pnl:.1f}%)'
                elif pnl<-8 and hd>5: er=f'rev({pnl:.1f}%)'
                if er: pending=('close',er)
            elif pos['direction']=='short':
                if p<pos['lowest']: pos['lowest']=p
                pnl=(pos['entry']-p)/pos['entry']*100
                hd=(dates[di]-pos['ed']).days
                if hd<=3: sl_eff=short_sl*1.5
                elif hd<=7: sl_eff=short_sl
                else: sl_eff=short_sl*0.7
                er=None
                if pnl<-sl_eff: er=f'ssl({pnl:.1f}%,d{hd})'
                elif pnl>short_tp: er=f'stp({pnl:.1f}%)'
                elif short_trail>0 and pnl>3:
                    rb=(p-pos['lowest'])/pos['lowest']*100
                    if rb>short_trail: er=f'strail({pnl:.1f}%,rb{rb:.1f}%)'
                elif short_hm>0 and hd>=short_hm: er=f'smax({hd}d,{pnl:.1f}%)'
                elif pnl<-8 and hd>5: er=f'srev({pnl:.1f}%)'
                if er: pending=('close',er)

        # --- Entry ---
        if pos is None and pending is None:
            bull_ok=(not np.isnan(MKT_MOM20[di]) and MKT_MOM20[di]>0) or \
                    (not np.isnan(MKT_MA60[di]) and MKT_CUM[di]>MKT_MA60[di])

            # =========== SELECTION A: V53 baseline (MOM20 ranking) ===========
            if selection=='v53':
                if not bull_ok: continue
                best_si=-1;best_mom=-999;best_agree=0
                for si in range(NS):
                    agree=buy_set[di].get(si,0)
                    if agree<3: continue
                    m20=MOM20[si,di] if not np.isnan(MOM20[si,di]) else 0
                    m5=MOM5[si,di] if not np.isnan(MOM5[si,di]) else 0
                    score=m20*0.6+m5*0.4
                    if agree>best_agree or (agree==best_agree and score>best_mom):
                        best_agree=agree;best_mom=score;best_si=si
                if best_si>=0: pending=('open_long',best_si)

            # =========== SELECTION B: consensus + 3-factor percentile ===========
            elif selection=='3f':
                if not bull_ok: continue
                best_si=-1;best_score=-1;best_agree=0
                for si in range(NS):
                    agree=buy_set[di].get(si,0)
                    if agree<3: continue
                    bm=PCT_MOM[si,di]; bv=PCT_VDP[si,di]; bk=PCT_KER[si,di]
                    if np.isnan(bm) or np.isnan(bv) or np.isnan(bk): continue
                    score=bm*w_mom+bv*w_vdp+bk*w_ker
                    if agree>best_agree or (agree==best_agree and score>best_score):
                        best_agree=agree;best_score=score;best_si=si
                if best_si>=0: pending=('open_long',best_si)

            # =========== SELECTION C: consensus + 3F + short ===========
            elif selection=='3f_ls':
                if bull_ok:
                    # Long: consensus≥3 + 3F ranking
                    best_si=-1;best_score=-1;best_agree=0
                    for si in range(NS):
                        agree=buy_set[di].get(si,0)
                        if agree<3: continue
                        bm=PCT_MOM[si,di]; bv=PCT_VDP[si,di]; bk=PCT_KER[si,di]
                        if np.isnan(bm) or np.isnan(bv) or np.isnan(bk): continue
                        score=bm*w_mom+bv*w_vdp+bk*w_ker
                        if agree>best_agree or (agree==best_agree and score>best_score):
                            best_agree=agree;best_score=score;best_si=si
                    if best_si>=0: pending=('open_long',best_si)
                elif mode=='long_short':
                    # Short: sell consensus≥3 + inverted 3F ranking
                    best_si=-1;best_score=-1;best_agree=0
                    for si in range(NS):
                        agree=sell_set[di].get(si,0)
                        if agree<3: continue
                        if buy_set[di].get(si,0)>=3: continue  # 排除同时有强买入的
                        bm=PCT_MOM[si,di]; bv=PCT_VDP[si,di]; bk=PCT_KER[si,di]
                        if np.isnan(bm) or np.isnan(bv) or np.isnan(bk): continue
                        # bear_score: 低MOM + 低VDP + 高KER = 干净下跌
                        score=(100-bm)*w_mom+(100-bv)*w_vdp+bk*w_ker
                        if agree>best_agree or (agree==best_agree and score>best_score):
                            best_agree=agree;best_score=score;best_si=si
                    if best_si>=0: pending=('open_short',best_si)

            # =========== SELECTION D: 3-factor FLIP (no consensus) ===========
            elif selection=='flip':
                if not bull_ok: continue
                best_si=-1;best_score=-1
                for si in range(NS):
                    bm=PCT_MOM[si,di]; bv=PCT_VDP[si,di]; bk=PCT_KER[si,di]
                    if np.isnan(bm) or np.isnan(bv) or np.isnan(bk): continue
                    # FLIP detection: MOM5 flipped positive in last 3 days
                    mom5_now=MOM5[si,di]
                    mom5_3ago=MOM5[si,di-3] if di>=3 else np.nan
                    mom_flip=not np.isnan(mom5_now) and not np.isnan(mom5_3ago) and mom5_now>0 and mom5_3ago<=0
                    # VDP flip: delta crossed positive in last 3 days
                    vdp_now=VDP_DELTA[si,di]
                    vdp_3ago=VDP_DELTA[si,di-3] if di>=3 else np.nan
                    vdp_flip=not np.isnan(vdp_now) and not np.isnan(vdp_3ago) and vdp_now>0 and vdp_3ago<=0
                    if not (mom_flip or vdp_flip): continue  # 至少一个flip
                    score=bm*w_mom+bv*w_vdp+bk*w_ker
                    if score>best_score: best_score=score;best_si=si
                if best_si>=0: pending=('open_long',best_si)

    # Close remaining
    if pos is not None:
        p=C[pos['si'],end_idx]
        if not np.isnan(p) and p>0:
            d=pos['direction']
            if d=='long':
                pnl=(p-pos['entry'])/pos['entry']*100
                cash+=pos['shares']*p*(1-COMMISSION-STAMP_DUTY)
            else:
                pnl=(pos['entry']-p)/pos['entry']*100
                S=pos['shares'];E=pos['entry']
                cash+=S*(E-p)-S*E*(COMMISSION+STAMP_DUTY)-S*p*COMMISSION
            trades.append({'pnl':pnl,'days':999,'reason':'end','dir':d})

    if cash<=0: return None
    days=(dates[end_idx]-dates[start_idx]).days
    yr=max(days/365.25,0.01)
    ann=((cash/CASH0)**(1/yr)-1)*100
    nw=sum(1 for t in trades if t['pnl']>0)
    wr=nw/max(len(trades),1)*100
    avg_w=np.mean([t['pnl'] for t in trades if t['pnl']>0]) if nw>0 else 0
    avg_l=np.mean([t['pnl'] for t in trades if t['pnl']<=0]) if nw<len(trades) else 0
    lt=[t for t in trades if t['dir']=='long']; st=[t for t in trades if t['dir']=='short']
    return {'ann':round(ann,1),'final':round(cash,0),'n':len(trades),
            'wr':round(wr,1),'avg_w':round(avg_w,1),'avg_l':round(avg_l,1),
            'long_n':len(lt),'short_n':len(st),
            'long_wr':round(sum(1 for t in lt if t['pnl']>0)/max(len(lt),1)*100,1),
            'short_wr':round(sum(1 for t in st if t['pnl']>0)/max(len(st),1)*100,1)}

# ============================================================
# [6] 搜索
# ============================================================
print(f"\n[5] Parameter search...", flush=True)
t5=time.time()
results=[]

# A: V53 baseline
print("  A: V53 baseline...", flush=True)
r=run('v53',0,0,0,20,50,20,5)
if r: results.append({**r,'sel':'V53','w':'mom_only'})

# B: consensus + 3F ranking (long only)
print("  B: Consensus + 3F ranking...", flush=True)
for wm,wv,wk in [(0.4,0.35,0.25),(0.5,0.3,0.2),(0.3,0.4,0.3),(0.6,0.3,0.1)]:
    for sl in [15,20]:
        for hm in [15,20]:
            r=run('3f',wm,wv,wk,sl,50,hm,5)
            if r and r['ann']>0:
                results.append({**r,'sel':'3F','w':f'{wm:.1f}/{wv:.1f}/{wk:.1f}','sl':sl,'hm':hm})

# C: consensus + 3F + short
print("  C: Consensus + 3F + Short...", flush=True)
for wm,wv,wk in [(0.4,0.35,0.25),(0.5,0.3,0.2)]:
    for ssl in [15,20]:
        r=run('3f_ls',wm,wv,wk,20,50,20,5,'long_short',ssl,30,12,5)
        if r and r['ann']>0:
            results.append({**r,'sel':'3F_LS','w':f'{wm:.1f}/{wv:.1f}/{wk:.1f}','ssl':ssl})

# D: 3-factor FLIP (no consensus)
print("  D: 3-Factor FLIP...", flush=True)
for wm,wv,wk in [(0.4,0.35,0.25),(0.5,0.3,0.2)]:
    for sl in [15,20]:
        for hm in [15,20]:
            r=run('flip',wm,wv,wk,sl,50,hm,5)
            if r and r['ann']>0:
                results.append({**r,'sel':'FLIP','w':f'{wm:.1f}/{wv:.1f}/{wk:.1f}','sl':sl,'hm':hm})

print(f"  Search done ({time.time()-t5:.1f}s)", flush=True)
results.sort(key=lambda x: -x['ann'])

if results:
    print(f"\n  {len(results)} configs", flush=True)
    print(f"\n  Top 20:", flush=True)
    for r in results[:20]:
        ls=f"L={r['long_n']}t({r['long_wr']:.0f}%) S={r['short_n']}t({r['short_wr']:.0f}%)" if r.get('short_n',0)>0 else f"{r['n']}t WR={r['wr']:.0f}%"
        print(f"    {r['sel']:6s} w={r['w']:>12s}: {r['ann']:+.1f}% | {ls} "
              f"W={r['avg_w']:+.1f}% L={r['avg_l']:.1f}% | {r['final']/10000:.0f}万", flush=True)

    # V53 vs 3F comparison
    v53r=next((r for r in results if r['sel']=='V53'),None)
    best_3f=next((r for r in results if r['sel']=='3F'),None)
    if v53r and best_3f:
        print(f"\n  === V53 vs 3F Ranking ===", flush=True)
        print(f"  V53(MOM):  {v53r['ann']:+.1f}% | {v53r['n']}t WR={v53r['wr']:.0f}%", flush=True)
        print(f"  3F(PTILE): {best_3f['ann']:+.1f}% | {best_3f['n']}t WR={best_3f['wr']:.0f}%", flush=True)
        print(f"  Delta: {best_3f['ann']-v53r['ann']:+.1f}%", flush=True)

    # Best year-by-year
    best=results[0]
    if best['sel']!='V53':
        print(f"\n  === Best: {best['sel']} w={best['w']} ===", flush=True)
        for year in range(2016,2027):
            s=next((i for i,d in enumerate(dates) if d>=pd.Timestamp(f'{year}-01-01')),0)
            e=next((i for i,d in enumerate(dates) if d>pd.Timestamp(f'{year}-12-31')),ND)-1
            if year==2026: e=next((i for i,d in enumerate(dates) if d>pd.Timestamp('2026-04-25')),ND)-1
            wp=best['w'].split('/')
            wm_f,wv_f,wk_f=float(wp[0]),float(wp[1]),float(wp[2])
            if best['sel']=='3F_LS':
                r=run('3f_ls',wm_f,wv_f,wk_f,20,50,20,5,'long_short',best.get('ssl',15),30,12,5,s,e)
            elif best['sel']=='3F':
                r=run('3f',wm_f,wv_f,wk_f,best.get('sl',20),50,best.get('hm',20),5,'long_only',15,30,12,5,s,e)
            elif best['sel']=='FLIP':
                r=run('flip',wm_f,wv_f,wk_f,best.get('sl',20),50,best.get('hm',20),5,'long_only',15,30,12,5,s,e)
            else:
                r=run('v53',0,0,0,20,50,20,5,'long_only',15,30,12,5,s,e)
            if r:
                ls=f"L={r['long_n']}({r['long_wr']:.0f}%) S={r['short_n']}({r['short_wr']:.0f}%)" if r['short_n']>0 else f"{r['n']}t WR={r['wr']:.0f}%"
                print(f"    {year}: {r['ann']:+.0f}% | {ls} | {r['final']/10000:.1f}万", flush=True)

        # V53 baseline year-by-year for comparison
        print(f"\n  === V53 baseline (comparison) ===", flush=True)
        for year in range(2016,2027):
            s=next((i for i,d in enumerate(dates) if d>=pd.Timestamp(f'{year}-01-01')),0)
            e=next((i for i,d in enumerate(dates) if d>pd.Timestamp(f'{year}-12-31')),ND)-1
            if year==2026: e=next((i for i,d in enumerate(dates) if d>pd.Timestamp('2026-04-25')),ND)-1
            r=run('v53',0,0,0,20,50,20,5,'long_only',15,30,12,5,s,e)
            if r: print(f"    {year}: {r['ann']:+.0f}% | {r['n']}t WR={r['wr']:.0f}% | {r['final']/10000:.1f}万", flush=True)
else:
    print("  NO positive results!", flush=True)

print(f"\nDone! {time.time()-t0:.1f}s", flush=True)
