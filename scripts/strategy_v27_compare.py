#!/usr/bin/env python3
"""
策略创新扫描 v27 — 三种新方法
目标: 提高胜率到50%+的同时提高年化
方法:
  1. 趋势回调 (pullback): 趋势中买回调，而不是追突破
  2. 卖期权收权利金 (credit_spread): 卖OTM期权，高WR
  3. 混合 (hybrid): 期货方向仓+卖期权收权利金
"""
import os, sys, time, numpy as np, pandas as pd
from collections import defaultdict
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(__file__))
from contract_specs import get_spec

def bs_price(S, K, T, r, sigma, opt='call'):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(S-K,0) if opt=='call' else max(K-S,0)
    d1 = (np.log(S/K) + (r+0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if opt=='call': return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else: return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

def load_data(data_dir):
    data = {}
    for f in sorted(os.listdir(data_dir)):
        if not f.endswith('.csv'): continue
        symbol = f.replace('.csv','')
        df = pd.read_csv(os.path.join(data_dir, f))
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = df.sort_values('trade_date').reset_index(drop=True)
        if len(df)<100: continue
        df['return'] = df['close'].pct_change()
        df['ma5'] = df['close'].rolling(5).mean()
        df['ma10'] = df['close'].rolling(10).mean()
        df['ma20'] = df['close'].rolling(20).mean()
        df['ma60'] = df['close'].rolling(60).mean()
        df['hv_20'] = df['return'].rolling(20).std() * np.sqrt(252)
        tr1=df['high']-df['low']; tr2=abs(df['high']-df['close'].shift()); tr3=abs(df['low']-df['close'].shift())
        df['tr']=pd.concat([tr1,tr2,tr3],axis=1).max(axis=1)
        df['atr']=df['tr'].rolling(14).mean(); df['atr_pct']=df['atr']/df['close']
        delta=df['close'].diff(); gain=delta.where(delta>0,0).rolling(14).mean()
        loss_s=(-delta.where(delta<0,0)).rolling(14).mean()
        df['rsi']=100-(100/(1+gain/loss_s))
        df['mom_5']=df['close'].pct_change(5); df['mom_10']=df['close'].pct_change(10)
        df['mom_20']=df['close'].pct_change(20)
        df['trend']=np.where(df['ma20']>df['ma60'],1,-1)
        df['vol_ma20']=df['vol'].rolling(20).mean(); df['vol_ratio']=df['vol']/df['vol_ma20']
        df['oi_change']=df['oi'].pct_change(5) if 'oi' in df.columns else 0
        df=df.dropna(subset=['ma20','ma60','hv_20','mom_10','rsi'])
        if len(df)>100: data[symbol]=df
    return data

def build_date_index(data, start_date, end_date):
    dm = defaultdict(dict)
    for symbol, df in data.items():
        mask = (df['trade_date']>=start_date)&(df['trade_date']<=end_date)
        for _,row in df[mask].iterrows(): dm[row['trade_date']][symbol]=row
    return dm

# ==================== 策略1: 趋势回调 ====================
def get_pullback_signals(day_data, positions):
    """趋势回调: 趋势确立+价格回调+开始反弹"""
    signals = []
    for symbol, row in day_data.items():
        if symbol in positions: continue
        if row.get('atr_pct',0.1)>0.045: continue

        trend = row.get('trend',0)
        rsi = row.get('rsi',50)
        close = row['close']
        ma5 = row.get('ma5',close)
        ma10 = row.get('ma10',close)
        ma20 = row.get('ma20',close)
        mom_20 = row.get('mom_20',0)
        vol_ratio = row.get('vol_ratio',1.0)

        # 做多: 上升趋势 + RSI回调到30-45区间 + 价格开始反弹(close>ma5)
        if trend==1 and 25<rsi<45 and close>ma5 and mom_20>0:
            score = (45-rsi)/15*40 + min(vol_ratio,2)*20 + min(mom_20*50,1)*40
            signals.append((symbol,1,score))

        # 做空: 下降趋势 + RSI反弹到55-70区间 + 价格开始回落(close<ma5)
        elif trend==-1 and 55<rsi<75 and close<ma5 and mom_20<0:
            score = (rsi-55)/15*40 + min(vol_ratio,2)*20 + min(-mom_20*50,1)*40
            signals.append((symbol,-1,score))

    signals.sort(key=lambda x:x[2], reverse=True)
    return signals

# ==================== 策略2: 卖OTM期权 ====================
def get_sell_option_signals(day_data, positions, otm_pct=0.03):
    """卖OTM期权: 趋势方向+卖虚值期权收权利金"""
    signals = []
    for symbol, row in day_data.items():
        if symbol in positions: continue
        if row.get('atr_pct',0.1)>0.045: continue

        trend = row.get('trend',0)
        mom = row.get('mom_10',0)
        hv = row.get('hv_20',0)
        if hv<0.10 or hv>0.60: continue

        # 看涨时卖OTM看跌期权 (收权利金, 对方行权=你低价接货)
        if trend==1 and mom>0.02:
            score = abs(mom)*100
            signals.append((symbol, 1, score, 'sell_put'))

        # 看跌时卖OTM看涨期权
        elif trend==-1 and mom<-0.02:
            score = abs(mom)*100
            signals.append((symbol, -1, score, 'sell_call'))

    signals.sort(key=lambda x:x[2], reverse=True)
    return signals

# ==================== 策略3: 动量(原版) ====================
def get_momentum_signals(day_data, positions, min_mom=0.03):
    signals = []
    for symbol, row in day_data.items():
        if symbol in positions: continue
        if row.get('atr_pct',0.1)>0.045: continue
        mom=row.get('mom_10',0); trend=row.get('trend',0)
        if trend==1 and mom>min_mom: signals.append((symbol,1,abs(mom)))
        elif trend==-1 and mom<-min_mom: signals.append((symbol,-1,abs(mom)))
    signals.sort(key=lambda x:x[2], reverse=True)
    return signals

# ==================== 回测引擎 ====================
def run_backtest(data, start_date, end_date, strategy, leverage=4, hold_days=5, **kwargs):
    max_pos = 3
    r = 0.02
    date_map = build_date_index(data, start_date, end_date)
    dates = sorted(date_map.keys())

    equity=500000.0; cash=500000.0; positions={}; closed_pnls=[]; equity_curve=[]

    for date in dates:
        day_data = date_map[date]

        # 退出
        for symbol in list(positions.keys()):
            pos = positions[symbol]
            if symbol not in day_data: continue
            price = day_data[symbol]['close']
            hd = (date-pos['entry_date']).days
            if hd < pos['hold_days']: continue

            if pos['type']=='futures':
                pnl = (price-pos['entry_price'])*pos['dir']*pos['mult']*pos['lots']
                comm = price*pos['mult']*pos['lots']*0.00015*2
                net = pnl - comm
                cash += pos['margin'] + net
                closed_pnls.append(net)
                del positions[symbol]

            elif pos['type']=='sell_option':
                # 卖期权平仓: 买回期权
                hv = day_data[symbol].get('hv_20',0.25)
                rem_T = max(0.001/365, 0.001)
                buyback = bs_price(price, pos['strike'], rem_T, r, hv, pos['otype'])
                intrinsic = max(price-pos['strike'],0) if pos['otype']=='call' else max(pos['strike']-price,0)
                buyback = max(buyback, intrinsic*0.95)
                total_buyback = buyback*pos['mult']*pos['contracts']
                comm = total_buyback*0.0003
                net = pos['credit'] - total_buyback - comm - pos['comm']
                cash += pos['margin'] + net  # return margin + settle
                closed_pnls.append(net)
                del positions[symbol]

        # 入场
        if len(positions)<max_pos:
            if strategy=='pullback':
                signals = get_pullback_signals(day_data, positions)
                for symbol,direction,score in signals:
                    if len(positions)>=max_pos: break
                    row=day_data[symbol]; price=row['close']
                    mult,mr,_,_=get_spec(symbol); mpl=price*mult*mr
                    if mpl<=0: continue
                    target_n = equity*(leverage/max_pos)
                    lots = max(int(target_n/(price*mult)),1)
                    tm = sum(p['margin'] for p in positions.values())+mpl*lots
                    if tm>equity*0.85:
                        lots=max(int((equity*0.85-sum(p['margin'] for p in positions.values()))/mpl),0)
                        if lots<=0: continue
                    am=mpl*lots; comm=price*mult*lots*0.00015
                    if am+comm>cash:
                        lots=max(int((cash-comm)/mpl),0)
                        if lots<=0: continue
                        am=mpl*lots; comm=price*mult*lots*0.00015
                    cash -= am+comm
                    positions[symbol]={'type':'futures','dir':direction,'entry_price':price*(1+0.0001*direction),
                        'entry_date':date,'lots':lots,'mult':mult,'margin':am,'hold_days':hold_days,'comm':comm}

            elif strategy=='sell_option':
                otm = kwargs.get('otm_pct', 0.03)
                signals = get_sell_option_signals(day_data, positions, otm)
                for symbol,direction,score,otype in signals:
                    if len(positions)>=max_pos: break
                    row=day_data[symbol]; price=row['close']; hv=row.get('hv_20',0.25)
                    mult,mr,_,_=get_spec(symbol)
                    if direction==1: K=price*(1-otm)  # sell put OTM
                    else: K=price*(1+otm)  # sell call OTM
                    T=hold_days/365.0; sigma=hv
                    opt_type='put' if direction==1 else 'call'
                    premium=bs_price(price,K,T,r,sigma,opt_type)
                    if premium<=0: continue
                    # 卖出: 收权利金, 需要保证金
                    credit_per = premium*mult  # 每张收的权利金
                    margin_per = price*mult*mr  # 保证金(同期货)
                    risk_per = otm*price*mult  # 最大亏损 ≈ OTM距离*乘数
                    contracts = max(int(equity*0.02/risk_per),1) if risk_per>0 else 1
                    total_margin = margin_per*contracts
                    total_credit = credit_per*contracts
                    comm = total_credit*0.0003
                    if total_margin+comm>cash*0.5:
                        contracts=max(int((cash*0.5-comm)/margin_per),0)
                        if contracts<=0: continue
                        total_margin=margin_per*contracts; total_credit=credit_per*contracts; comm=total_credit*0.0003
                    cash -= total_margin - total_credit + comm  # 付保证金-收权利金+手续费
                    positions[symbol]={'type':'sell_option','dir':direction,'otype':opt_type,
                        'entry_price':price,'strike':K,'entry_date':date,'contracts':contracts,
                        'mult':mult,'margin':total_margin,'credit':total_credit,'comm':comm,
                        'hold_days':hold_days}

            elif strategy=='momentum':
                signals = get_momentum_signals(day_data, positions, kwargs.get('min_mom',0.03))
                for symbol,direction,score in signals:
                    if len(positions)>=max_pos: break
                    row=day_data[symbol]; price=row['close']
                    mult,mr,_,_=get_spec(symbol); mpl=price*mult*mr
                    if mpl<=0: continue
                    target_n=equity*(leverage/max_pos); lots=max(int(target_n/(price*mult)),1)
                    tm=sum(p['margin'] for p in positions.values())+mpl*lots
                    if tm>equity*0.85:
                        lots=max(int((equity*0.85-sum(p['margin'] for p in positions.values()))/mpl),0)
                        if lots<=0: continue
                    am=mpl*lots; comm=price*mult*lots*0.00015
                    if am+comm>cash:
                        lots=max(int((cash-comm)/mpl),0);
                        if lots<=0: continue
                        am=mpl*lots; comm=price*mult*lots*0.00015
                    cash-=am+comm
                    positions[symbol]={'type':'futures','dir':direction,'entry_price':price*(1+0.0001*direction),
                        'entry_date':date,'lots':lots,'mult':mult,'margin':am,'hold_days':hold_days,'comm':comm}

        # 权益
        unrealized=0
        for symbol,pos in positions.items():
            if symbol not in day_data: continue
            price=day_data[symbol]['close']; hv=day_data[symbol].get('hv_20',0.25)
            if pos['type']=='futures':
                unrealized+=(price-pos['entry_price'])*pos['dir']*pos['mult']*pos['lots']
            elif pos['type']=='sell_option':
                rem_T=max((pos['hold_days']-(date-pos['entry_date']).days)/365.0,0.001)
                buyback=bs_price(price,pos['strike'],rem_T,r,hv,pos['otype'])
                intrinsic=max(price-pos['strike'],0) if pos['otype']=='call' else max(pos['strike']-price,0)
                buyback=max(buyback,intrinsic*0.95)
                unrealized+=pos['credit']-buyback*pos['mult']*pos['contracts']
        equity=cash+unrealized; equity_curve.append((date,equity))
        if equity<5000: break

    if not equity_curve or equity_curve[-1][1]<=0: return None
    final=equity_curve[-1][1]; tr=(final-500000)/500000
    days=(equity_curve[-1][0]-equity_curve[0][0]).days; years=max(days/365,0.001)
    ann=float((1+tr)**(1/years)-1)
    eq=pd.DataFrame(equity_curve,columns=['date','equity'])
    eq['cummax']=eq['equity'].cummax(); eq['dd']=(eq['equity']-eq['cummax'])/eq['cummax']
    mdd=float(eq['dd'].min())
    pnls=np.array(closed_pnls); wr=float((pnls>0).mean()) if len(pnls)>0 else 0
    avg_w=float(pnls[pnls>0].mean()) if (pnls>0).any() else 0
    avg_l=float(abs(pnls[pnls<=0].mean())) if (pnls<=0).any() else 1
    pf=avg_w*(pnls>0).sum()/(avg_l*(pnls<=0).sum()) if (pnls<=0).sum()>0 and avg_l>0 else 0
    return {'strategy':strategy,'leverage':leverage,'hold_days':hold_days,'annual':ann,'wr':wr,
            'mdd':mdd,'pf':pf,'trades':len(pnls),'final':final,'total_ret':float(tr)}


def main():
    data_dir=os.path.expanduser("~/home/futures_platform/data/futures_weighted")
    print("加载数据..."); data=load_data(data_dir); print(f"加载了 {len(data)} 个品种\n")
    start_date=pd.Timestamp('2018-01-01'); end_date=pd.Timestamp('2026-05-08')

    results=[]

    # 1. 趋势回调
    print("测试趋势回调...")
    for lev in [4,5,6,7]:
        for hd in [5,7,10]:
            r=run_backtest(data,start_date,end_date,'pullback',lev,hd)
            if r: results.append(r)

    # 2. 卖期权
    print("测试卖期权...")
    for otm in [0.02,0.03,0.04,0.05]:
        for hd in [5,7,10]:
            r=run_backtest(data,start_date,end_date,'sell_option',leverage=0,hold_days=hd,otm_pct=otm)
            if r: r['otm']=otm; results.append(r)

    # 3. 动量(对照组)
    print("测试动量...")
    for lev in [4,5,6]:
        for hd in [5,7,10]:
            r=run_backtest(data,start_date,end_date,'momentum',lev,hd,min_mom=0.03)
            if r: results.append(r)

    print(f"\n{'策略':>14} {'杠杆':>4} {'持有':>4} {'OTM':>4} {'年化':>8} {'胜率':>6} {'盈亏比':>6} {'回撤':>8} {'交易':>4} {'最终':>14}")
    print("-"*85)
    results.sort(key=lambda x:x['annual'],reverse=True)
    for r in results[:40]:
        otm_str=f"{r.get('otm',0):.0%}" if 'otm' in r else "-"
        print(f"{r['strategy']:>14} {r.get('leverage',0):>4} {r['hold_days']:>4} {otm_str:>4} "
              f"{r['annual']:>8.1%} {r['wr']:>6.1%} {r['pf']:>6.2f} {r['mdd']:>8.1%} "
              f"{r['trades']:>4} {r['final']:>14,.0f}")

    # 筛选
    print("\n\n=== 胜率>=50% 且 年化最优 ===")
    good=[r for r in results if r['wr']>=0.50]
    if good:
        for r in sorted(good,key=lambda x:x['annual'],reverse=True)[:10]:
            print(f"  {r['strategy']:>14}: 年化={r['annual']:.1%}  胜率={r['wr']:.1%}  "
                  f"杠杆={r.get('leverage',0)}x  持有={r['hold_days']}天  回撤={r['mdd']:.1%}  "
                  f"权益={r['final']:,.0f}")
    else:
        print("无满足条件的组合")

    print("\n=== 胜率>=55% ===")
    good55=[r for r in results if r['wr']>=0.55]
    if good55:
        for r in sorted(good55,key=lambda x:x['annual'],reverse=True)[:10]:
            print(f"  {r['strategy']:>14}: 年化={r['annual']:.1%}  胜率={r['wr']:.1%}  "
                  f"杠杆={r.get('leverage',0)}x  持有={r['hold_days']}天  回撤={r['mdd']:.1%}")
    else:
        print("无满足条件的组合")

    # 按策略分组
    print("\n\n=== 各策略最优 ===")
    best={}
    for r in results:
        s=r['strategy']
        if s not in best or r['annual']>best[s]['annual']: best[s]=r
    for s,r in sorted(best.items(),key=lambda x:x[1]['annual'],reverse=True):
        print(f"  {s:>14}: 年化={r['annual']:.1%}  胜率={r['wr']:.1%}  盈亏比={r['pf']:.2f}  "
              f"回撤={r['mdd']:.1%}  交易={r['trades']}  权益={r['final']:,.0f}")


if __name__=='__main__':
    main()
