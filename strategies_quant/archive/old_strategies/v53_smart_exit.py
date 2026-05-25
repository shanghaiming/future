"""
V53 — V37核心 + 智能卖出系统
===============================

V37共识(a>=3)入场已经验证+43%.
但avgL=-7.2%说明亏损交易的退出可以优化.

新卖出逻辑 (概率论Section 14 极值理论):
  1. 持仓前3天: 宽止损(sl×1.5) — 给交易空间
  2. 持仓3-7天: 正常止损(sl) + 策略卖出信号
  3. 持仓7天+: 紧止损(sl×0.7) + 追踪止盈(从最高点回撤)
  4. 任何时刻: 如果consensus翻负(买入策略全部翻卖) → 立即退出

  数学基础:
  - 前3天波动最大(ATR效应) → 放宽容错
  - 7天后如果还没涨 → 大概率是错信号 → 收紧止损
  - 追踪止盈保护浮盈 — EVT优化版
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
print("  V53 — V37核心 + 智能卖出", flush=True)
print("=" * 70, flush=True)

print("\n[1] 加载...", flush=True)
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
for si, s in enumerate(syms):
    df = stock_data.get(s)
    if df is None: continue
    for d in df.index:
        if d in dm:
            di = dm[d]
            for arr, col in [(C,'close'),(O,'open')]:
                if col in df.columns: arr[si, di] = float(df.loc[d, col])
C = C[:, i0:i1+1]; O = O[:, i0:i1+1]
print(f"  {NS} stocks, {ND} days ({time.time()-t0:.1f}s", flush=True)

MOM20 = np.full_like(C, np.nan); MOM5 = np.full_like(C, np.nan)
for si in range(NS):
    for di in range(20, ND):
        if not np.isnan(C[si, di]) and not np.isnan(C[si, di-20]) and C[si, di-20] > 0:
            MOM20[si, di] = (C[si, di] - C[si, di-20]) / C[si, di-20]
        if not np.isnan(C[si, di]) and not np.isnan(C[si, di-5]) and C[si, di-5] > 0:
            MOM5[si, di] = (C[si, di] - C[si, di-5]) / C[si, di-5]

MKT_RET = np.full(ND, np.nan)
for di in range(ND):
    r = [C[si,di]/C[si,di-1]-1 for si in range(NS) if di>0 and not np.isnan(C[si,di]) and not np.isnan(C[si,di-1]) and C[si,di-1]>0]
    if len(r)>100: MKT_RET[di]=np.mean(r)
MKT_CUM = np.nancumsum(np.where(np.isnan(MKT_RET),0,MKT_RET))
MKT_MOM20 = np.full(ND, np.nan)
for di in range(20, ND):
    v=MKT_RET[di-20:di]; v=v[~np.isnan(v)]
    if len(v)>10: MKT_MOM20[di]=np.sum(v)
MKT_MA60 = np.full(ND, np.nan)
for di in range(60, ND): MKT_MA60[di]=np.mean(MKT_CUM[di-60:di])

print("[2] 策略信号...", flush=True)
with open('.v15_7_signals_fixed.pkl', 'rb') as f: all_signals = pickle.load(f)
date_to_di = {d: i for i, d in enumerate(dates)}
buy_set = defaultdict(lambda: defaultdict(int))
sell_set = defaultdict(lambda: defaultdict(int))
for sname in USE_STRATS:
    if sname not in all_signals: continue
    for sym, sigs in all_signals[sname].items():
        if sym not in syms: continue
        si = syms.index(sym)
        for ts, action, price in sigs:
            if ts in date_to_di:
                di = date_to_di[ts]
                if action == 'buy': buy_set[di][si] += 1
                elif action == 'sell': sell_set[di][si] += 1


def run(exit_mode, base_sl, tp_pct, hold_max, trail_pct=0,
        start_idx=0, end_idx=ND-1):
    """
    exit_mode:
      'v37'     — V37标准退出 (baseline)
      'adaptive'— 分时段止损
      'trail'   — 追踪止盈 + 策略卖出
      'enhanced'— 全部优化: 分时段止损 + 追踪止盈 + 共识翻负退出
    """
    cash = float(CASH0); hold = None; trades = []
    pending_sell = False
    pending_buy = None; pb_agree = 0

    for di in range(max(start_idx, 60), end_idx+1):
        if pending_sell and hold:
            p = O[hold['si'], di]
            if np.isnan(p) or p <= 0: p = C[hold['si'], di]
            if not np.isnan(p) and p > 0:
                pnl = (p - hold['entry']) / hold['entry'] * 100
                cash += hold['shares'] * p * (1 - COMMISSION - STAMP_DUTY)
                trades.append({'pnl': pnl, 'days': (dates[di]-hold['ed']).days,
                              'reason': hold.get('sr',''), 'agree': hold.get('agree',0)})
                hold = None
            pending_sell = False

        if hold is None and pending_buy is not None:
            si = pending_buy; pending_buy = None
            p = O[si, di]
            if np.isnan(p) or p <= 0:
                p = C[si, di-1] if di > 0 and not np.isnan(C[si, di-1]) else np.nan
            if not np.isnan(p) and p > 0 and cash > 10000:
                shares = int(cash / (1 + COMMISSION) / p)
                if shares > 0:
                    cash -= shares * p * (1 + COMMISSION)
                    hold = {'si': si, 'shares': shares, 'entry': p,
                            'highest': p, 'ed': dates[di], 'agree': pb_agree,
                            'buy_di': di}  # 记录买入日期
            pending_buy = None

        if hold:
            si = hold['si']; p = C[si, di]
            if np.isnan(p): continue
            if p > hold['highest']: hold['highest'] = p
            pnl = (p - hold['entry']) / hold['entry'] * 100
            hd = (dates[di] - hold['ed']).days

            if exit_mode == 'v37':
                # V37标准
                if pnl < -base_sl: pending_sell=True; hold['sr']=f'sl({pnl:.1f}%)'; continue
                if pnl > tp_pct: pending_sell=True; hold['sr']=f'tp({pnl:.1f}%)'; continue
                if hold_max > 0 and hd >= hold_max: pending_sell=True; hold['sr']=f'max({hd}d,{pnl:.1f}%)'; continue
                if sell_set[di].get(si,0)>=1 and pnl>3: pending_sell=True; hold['sr']=f'signal({pnl:.1f}%)'; continue
                if pnl<-8 and hd>5: pending_sell=True; hold['sr']=f'rev({pnl:.1f}%)'; continue

            elif exit_mode == 'adaptive':
                # 分时段止损
                if hd <= 3:
                    sl_eff = base_sl * 1.5  # 前3天: 宽止损
                elif hd <= 7:
                    sl_eff = base_sl  # 正常止损
                else:
                    sl_eff = base_sl * 0.7  # 7天后: 紧止损
                if pnl < -sl_eff: pending_sell=True; hold['sr']=f'sl({pnl:.1f}%,d{hd})'; continue
                if pnl > tp_pct: pending_sell=True; hold['sr']=f'tp({pnl:.1f}%)'; continue
                if hold_max > 0 and hd >= hold_max: pending_sell=True; hold['sr']=f'max({hd}d,{pnl:.1f}%)'; continue
                if sell_set[di].get(si,0)>=1 and pnl>3: pending_sell=True; hold['sr']=f'signal({pnl:.1f}%)'; continue
                if pnl<-8 and hd>5: pending_sell=True; hold['sr']=f'rev({pnl:.1f}%)'; continue

            elif exit_mode == 'trail':
                # 追踪止盈
                if pnl < -base_sl: pending_sell=True; hold['sr']=f'sl({pnl:.1f}%)'; continue
                if pnl > tp_pct: pending_sell=True; hold['sr']=f'tp({pnl:.1f}%)'; continue
                # 追踪止盈: 从最高点回撤>trail_pct
                if trail_pct > 0 and pnl > 5:
                    drawdown = (hold['highest'] - p) / hold['highest'] * 100
                    if drawdown > trail_pct:
                        pending_sell=True; hold['sr']=f'trail({pnl:.1f}%,dd{drawdown:.1f}%)'; continue
                if hold_max > 0 and hd >= hold_max: pending_sell=True; hold['sr']=f'max({hd}d,{pnl:.1f}%)'; continue
                if sell_set[di].get(si,0)>=1 and pnl>3: pending_sell=True; hold['sr']=f'signal({pnl:.1f}%)'; continue
                if pnl<-8 and hd>5: pending_sell=True; hold['sr']=f'rev({pnl:.1f}%)'; continue

            elif exit_mode == 'enhanced':
                # 全部优化
                # 分时段止损
                if hd <= 3: sl_eff = base_sl * 1.5
                elif hd <= 7: sl_eff = base_sl
                else: sl_eff = base_sl * 0.7
                if pnl < -sl_eff: pending_sell=True; hold['sr']=f'sl({pnl:.1f}%,d{hd})'; continue
                if pnl > tp_pct: pending_sell=True; hold['sr']=f'tp({pnl:.1f}%)'; continue
                # 追踪止盈
                if trail_pct > 0 and pnl > 5:
                    drawdown = (hold['highest'] - p) / hold['highest'] * 100
                    if drawdown > trail_pct:
                        pending_sell=True; hold['sr']=f'trail({pnl:.1f}%,dd{drawdown:.1f}%)'; continue
                if hold_max > 0 and hd >= hold_max: pending_sell=True; hold['sr']=f'max({hd}d,{pnl:.1f}%)'; continue
                # 共识翻负: 如果当天买入信号为0且卖出信号≥2
                if sell_set[di].get(si,0) >= 2 and buy_set[di].get(si,0) == 0 and pnl > 0:
                    pending_sell=True; hold['sr']=f'flip({pnl:.1f}%)'; continue
                if sell_set[di].get(si,0)>=1 and pnl>3: pending_sell=True; hold['sr']=f'signal({pnl:.1f}%)'; continue
                if pnl<-8 and hd>5: pending_sell=True; hold['sr']=f'rev({pnl:.1f}%)'; continue

        if hold is None and pending_buy is None:
            f_ok = not np.isnan(MKT_MOM20[di]) and MKT_MOM20[di]>0
            m_ok = not np.isnan(MKT_MA60[di]) and MKT_CUM[di]>MKT_MA60[di]
            if not (f_ok or m_ok): continue

            best_si, best_mom, best_agree = -1, -999, 0
            for si in range(NS):
                agree = buy_set[di].get(si, 0)
                if agree < 3: continue
                m20 = MOM20[si,di] if not np.isnan(MOM20[si,di]) else 0
                m5 = MOM5[si,di] if not np.isnan(MOM5[si,di]) else 0
                score = m20*0.6+m5*0.4
                if agree>best_agree or (agree==best_agree and score>best_mom):
                    best_agree=agree; best_mom=score; best_si=si

            if best_si >= 0:
                pending_buy = best_si; pb_agree = best_agree

    if hold:
        p = C[hold['si'], end_idx]
        if not np.isnan(p) and p > 0:
            pnl = (p-hold['entry'])/hold['entry']*100
            cash += hold['shares']*p*(1-COMMISSION-STAMP_DUTY)
            trades.append({'pnl':pnl,'days':999,'reason':'end','agree':hold.get('agree',0)})

    if cash <= 0: return None
    days = (dates[end_idx]-dates[start_idx]).days
    yr = max(days/365.25, 0.01)
    ann = ((cash/CASH0)**(1/yr)-1)*100
    nw = sum(1 for t in trades if t['pnl']>0)
    wr = nw/max(len(trades),1)*100
    avg_w = np.mean([t['pnl'] for t in trades if t['pnl']>0]) if nw>0 else 0
    avg_l = np.mean([t['pnl'] for t in trades if t['pnl']<=0]) if nw<len(trades) else 0
    # 退出原因统计
    reasons = defaultdict(int)
    for t in trades:
        r = t.get('reason','')
        if 'sl' in r: reasons['sl'] += 1
        elif 'tp' in r: reasons['tp'] += 1
        elif 'max' in r: reasons['max'] += 1
        elif 'signal' in r: reasons['signal'] += 1
        elif 'trail' in r: reasons['trail'] += 1
        elif 'flip' in r: reasons['flip'] += 1
        elif 'rev' in r: reasons['rev'] += 1
        elif 'end' in r: reasons['end'] += 1
    return {'ann':round(ann,1),'final':round(cash,0),'n':len(trades),'wr':round(wr,1),
            'avg_w':round(avg_w,1),'avg_l':round(avg_l,1),'reasons':dict(reasons)}


# ============================================================
# [3] 搜索
# ============================================================
print(f"\n[3] 卖出优化搜索...", flush=True)
t3 = time.time()
results = []

# A: V37 baseline
for sl in [15, 20]:
    for hm in [15, 20]:
        r = run('v37', sl, 50, hm)
        if r and r['ann'] > 0:
            results.append({**r, 'mode': 'v37', 'sl': sl, 'hm': hm})

# B: Adaptive stop
for sl in [15, 20]:
    for hm in [15, 20]:
        r = run('adaptive', sl, 50, hm)
        if r and r['ann'] > 0:
            results.append({**r, 'mode': 'adaptive', 'sl': sl, 'hm': hm})

# C: Trailing stop
for trail in [3, 5, 8]:
    for sl in [15, 20]:
        for hm in [15, 20]:
            r = run('trail', sl, 50, hm, trail)
            if r and r['ann'] > 0:
                results.append({**r, 'mode': f'trail{trail}', 'sl': sl, 'hm': hm})

# D: Enhanced
for trail in [3, 5, 8]:
    for sl in [15, 20]:
        for hm in [15, 20]:
            r = run('enhanced', sl, 50, hm, trail)
            if r and r['ann'] > 0:
                results.append({**r, 'mode': f'enhanced_t{trail}', 'sl': sl, 'hm': hm})

results.sort(key=lambda x: -x['ann'])
print(f"  搜索完成 ({time.time()-t3:.1f}s)", flush=True)

if results:
    print(f"\n  {len(results)} positive configs", flush=True)
    print(f"\n  Top 20:", flush=True)
    for r in results[:20]:
        print(f"    {r['mode']:18s} sl={r['sl']}% hm={r['hm']}d: "
              f"{r['ann']:+.1f}% | {r['n']}t WR={r['wr']:.0f}% "
              f"avgW={r['avg_w']:+.1f}% avgL={r['avg_l']:.1f}% | {r['final']/10000:.0f}万 "
              f"exit:{r['reasons']}", flush=True)

    print(f"\n  === 按模式分组 ===", flush=True)
    modes_done = set()
    for r in results:
        mode_base = r['mode'].split('_')[0] if 'enhanced' not in r['mode'] else 'enhanced'
        if mode_base in modes_done: continue
        modes_done.add(mode_base)
        sub = [x for x in results if x['mode'].startswith(mode_base)]
        if sub:
            best = sub[0]
            print(f"  {best['mode']:18s}: {best['ann']:+.1f}% | {best['n']}t WR={best['wr']:.0f}% "
                  f"avgW={best['avg_w']:+.1f}% avgL={best['avg_l']:.1f}% | {best['final']/10000:.0f}万", flush=True)

    # 最佳年度
    best = results[0]
    print(f"\n  === 最佳: {best['mode']} sl={best['sl']}% hm={best['hm']}d ===", flush=True)
    for year in range(2016, 2027):
        s = next((i for i, d in enumerate(dates) if d >= pd.Timestamp(f'{year}-01-01')), 0)
        e = next((i for i, d in enumerate(dates) if d > pd.Timestamp(f'{year}-12-31')), ND) - 1
        if year == 2026:
            e = next((i for i, d in enumerate(dates) if d > pd.Timestamp('2026-04-25')), ND) - 1
        mode = best['mode']
        if mode.startswith('v37') or mode.startswith('adaptive'):
            r = run(mode, best['sl'], 50, best['hm'], 0, s, e)
        elif mode.startswith('trail'):
            trail = int(mode.replace('trail',''))
            r = run('trail', best['sl'], 50, best['hm'], trail, s, e)
        else:
            trail = int(mode.split('t')[-1])
            r = run('enhanced', best['sl'], 50, best['hm'], trail, s, e)
        if r:
            print(f"    {year}: {r['final']/10000:.1f}万 | {r['ann']:+.0f}% | "
                  f"{r['n']}t WR={r['wr']:.0f}% avgW={r['avg_w']:+.1f}% avgL={r['avg_l']:.1f}%", flush=True)

print(f"\nDone! {time.time()-t0:.1f}s", flush=True)
