#!/usr/bin/env python3
"""
策略 V59 — 均值回归期货 + 期权信号确认 (纯期货, 不交易期权)

数据分析发现:
  - 所有动量信号负期望收益 (反转效应)
  - 均值回归是唯一正期望信号: RSI<25 hold10d → 51.6%WR +0.87%avg
  - 品种差异巨大: SAFI 60.8%, sifi 62.4%, SMFI 59.9% MR WR
  - 95%利润来自5%交易 (Turtle Trader启示)

策略核心:
  1. 入场: 均值回归超卖信号 (RSI<25, 连续下跌, 通道底部)
  2. 期权信号服务期货: HV百分位≈IV百分位, 偏度≈skew, IV溢价≈term structure
  3. 纯期货交易 (不买不卖期权)
  4. Turtle式ATR跟踪止损 (让利润奔跑)
  5. 信号强度排名: 最超卖的品种优先入场

目标: 年化600%, 胜率>50%, 最大持仓3
"""

import os, sys, time, json, math
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from contract_specs import get_spec

COMM = 0.00015          # 期货手续费率 (单边)
R = 0.02
INIT = 500000
TD = 252


def load(data_dir):
    """加载数据, 计算均值回归指标和期权代理信号"""
    raw = {}
    for f in sorted(os.listdir(data_dir)):
        if not f.endswith('.csv'): continue
        sym = f.replace('.csv', '')
        df = pd.read_csv(os.path.join(data_dir, f))
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = df.sort_values('trade_date').reset_index(drop=True)
        if len(df) < 300: continue
        df['ret'] = df['close'].pct_change()

        # --- 动量 (仅用于反向指标) ---
        for lag in [3, 5, 10, 20, 60]:
            df[f'm{lag}'] = df['close'].pct_change(lag)

        # --- 波动率 (期权信号代理) ---
        for w in [5, 10, 20, 40, 60]:
            df[f'hv{w}'] = df['ret'].rolling(w).std() * np.sqrt(TD)

        # IV百分位 ≈ 期权IV百分位 (高=市场恐慌=超卖确认)
        df['hv_pct'] = df['hv20'].rolling(252).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 10 else .5, raw=False)

        # IV期限结构代理: 短期vol/中期vol (高=近期恐慌=超卖确认)
        df['iv_premium'] = (df['hv5'] / df['hv20'].replace(0, np.nan) - 1).clip(-0.5, 0.5)

        # --- 趋势指标 ---
        df['ma5'] = df['close'].rolling(5).mean()
        df['ma20'] = df['close'].rolling(20).mean()
        df['ma60'] = df['close'].rolling(60).mean()

        # --- RSI ---
        d = df['close'].diff(); g = d.where(d > 0, 0).rolling(14).mean()
        l = (-d.where(d < 0, 0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + g / l))

        # --- ATR ---
        tr = pd.concat([df['high'] - df['low'],
                        abs(df['high'] - df['close'].shift()),
                        abs(df['low'] - df['close'].shift())], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        df['atr_pct'] = df['atr'] / df['close']

        # --- 通道位置 (20日高低范围中的位置) ---
        ch_hi = df['close'].rolling(20).max()
        ch_lo = df['close'].rolling(20).min()
        ch_range = (ch_hi - ch_lo).replace(0, np.nan)
        df['ch_pos'] = ((df['close'] - ch_lo) / ch_range).clip(0, 1)

        # --- 连续下跌/上涨天数 ---
        down = (df['ret'] < 0).astype(int).values
        cons_d = []
        c = 0
        for v in down:
            c = c + 1 if v else 0
            cons_d.append(c)
        df['cons_down'] = cons_d

        up = (df['ret'] > 0).astype(int).values
        cons_u = []
        c = 0
        for v in up:
            c = c + 1 if v else 0
            cons_u.append(c)
        df['cons_up'] = cons_u

        # --- 偏度代理 (期权skew ≈ 下行vol - 上行vol) ---
        up_vol = df['ret'].where(df['ret'] > 0, 0).rolling(20).std() * np.sqrt(TD)
        dn_vol = (-df['ret'].where(df['ret'] < 0, 0)).rolling(20).std() * np.sqrt(TD)
        df['skew'] = (dn_vol - up_vol) / df['hv20'].replace(0, np.nan)

        # --- 成交量比 ---
        df['vol_ma20'] = df['vol'].rolling(20).mean()
        df['vol_ratio'] = df['vol'] / df['vol_ma20'].replace(0, np.nan)

        df = df.dropna(subset=['ma20', 'ma60', 'hv20', 'rsi', 'atr', 'ch_pos'])
        if len(df) < 100: continue
        try:
            spec = get_spec(sym)
        except:
            continue

        raw[sym] = {
            'spec': spec,
            'dates': df['trade_date'].values,
            'open': df['open'].values.astype(np.float64),
            'close': df['close'].values.astype(np.float64),
            'high': df['high'].values.astype(np.float64),
            'low': df['low'].values.astype(np.float64),
            'atr': df['atr'].values.astype(np.float64),
            'atr_pct': df['atr_pct'].values.astype(np.float64),
            'hv20': df['hv20'].values.astype(np.float64),
            'hv_pct': df['hv_pct'].values.astype(np.float64),
            'iv_premium': df['iv_premium'].values.astype(np.float64),
            'rsi': df['rsi'].values.astype(np.float64),
            'trend': np.where(df['ma5'] > df['ma20'], 1., -1.).astype(np.float64),
            'ch_pos': df['ch_pos'].values.astype(np.float64),
            'cons_down': np.array(cons_d, dtype=np.int32),
            'cons_up': np.array(cons_u, dtype=np.int32),
            'skew': df['skew'].values.astype(np.float64),
            'vol_ratio': df['vol_ratio'].values.astype(np.float64),
            'm5': df['m5'].values.astype(np.float64),
        }
    return raw


def build_idx(raw, s, e):
    """构建日期索引"""
    ad = set()
    for d in raw.values():
        m = (d['dates'] >= s) & (d['dates'] <= e)
        for dt in d['dates'][m]:
            ad.add(dt)
    dates = np.array(sorted(ad))
    si = {}
    for sym, d in raw.items():
        im = {}
        m = (d['dates'] >= s) & (d['dates'] <= e)
        for dt, il in zip(d['dates'][m], np.where(m)[0]):
            im[dt] = int(il)
        si[sym] = im
    return dates, si


def signal_score(d, il, direction, rsi_lo, rsi_hi, ch_lo, ch_hi,
                 cons_min, hv_min, skew_enhance):
    """
    计算信号强度分数 (越高越强)

    direction: 1=做多(超卖回归), -1=做空(超买回归)
    返回: (score, 进入理由) 或 None
    """
    pi = il - 1  # 用昨日数据生成信号
    if pi < 0:
        return None

    rsi = d['rsi'][pi]
    ch = d['ch_pos'][pi]
    hv_pct = d['hv_pct'][pi]
    cons_d = d['cons_down'][pi]
    cons_u = d['cons_up'][pi]
    skew_v = d['skew'][pi]
    iv_prem = d['iv_premium'][pi]
    atr = d['atr'][pi]

    if np.isnan(rsi) or np.isnan(ch) or np.isnan(atr) or atr <= 0:
        return None

    # HV百分位过滤 (期权IV代理: 高IV=恐慌=超卖确认)
    if hv_min > 0 and (np.isnan(hv_pct) or hv_pct < hv_min):
        return None

    score = 0.0

    if direction > 0:  # 做多: 超卖
        if rsi >= rsi_lo:
            return None
        # RSI越低越好
        score += (rsi_lo - rsi) / rsi_lo
        # 通道越低越好
        if ch < ch_lo:
            score += (ch_lo - ch) / ch_lo
        # 连续下跌加分
        if cons_d >= cons_min:
            score += 0.5
        # Skew正值=恐慌=超卖确认
        if skew_enhance and not np.isnan(skew_v) and skew_v > 0.1:
            score += 0.3
        # IV溢价高=近期恐慌
        if not np.isnan(iv_prem) and iv_prem > 0.1:
            score += 0.3
    else:  # 做空: 超买
        if rsi <= rsi_hi:
            return None
        score += (rsi - rsi_hi) / (100 - rsi_hi)
        if ch > ch_hi:
            score += (ch - ch_hi) / (1 - ch_hi)
        if cons_u >= cons_min:
            score += 0.5
        if skew_enhance and not np.isnan(skew_v) and skew_v < -0.1:
            score += 0.3
        if not np.isnan(iv_prem) and iv_prem < -0.1:
            score += 0.3

    if score <= 0:
        return None
    return score


def bt(raw, dates, si, p):
    """均值回归期货回测

    stop_atr=0 → 无止损, 纯时间退出 (匹配数据分析方法)
    stop_atr>0 → ATR止损 + 跟踪止损
    """
    mp = p.get('max_pos', 3)
    hd = p.get('hold_days', 10)
    rsi_lo = p.get('rsi_lo', 25)
    rsi_hi = p.get('rsi_hi', 75)
    ch_lo = p.get('ch_lo', 0.4)
    ch_hi = p.get('ch_hi', 0.6)
    cons_min = p.get('cons_min', 3)
    stop_atr = p.get('stop_atr', 0)       # 0 = 无止损
    trail_atr = p.get('trail_atr', 1.5)
    trail_after_atr = p.get('trail_after', 1.0)
    nm = p.get('notional_mult', 10.0)
    risk_pct = p.get('risk_pct', 0.03)
    hv_min = p.get('hv_min', 0.0)
    mode = p.get('mode', 'both')
    skew_enhance = p.get('skew_enhance', False)
    rsi_exit = p.get('rsi_exit', False)
    score_min = p.get('score_min', 0.0)
    use_stops = stop_atr > 0

    eq = float(INIT)
    pos = {}
    pnls = []
    eqh = [float(INIT)]

    for date in dates:
        # === 退出 ===
        for sym in list(pos):
            ps = pos[sym]
            im = si.get(sym)
            if not im or date not in im:
                continue
            il = im[date]
            S = raw[sym]['close'][il]
            H = raw[sym]['high'][il]
            L = raw[sym]['low'][il]
            rsi_now = raw[sym]['rsi'][il]
            cur_atr = raw[sym]['atr'][il]
            h = int((date - ps['ed']) / np.timedelta64(1, 'D'))

            direction = ps['dir']
            should_exit = False
            exit_price = S

            # 时间止损 (主退出)
            if h >= hd:
                should_exit = True

            # RSI退出
            if rsi_exit and not np.isnan(rsi_now):
                if direction > 0 and rsi_now > 65:
                    should_exit = True
                elif direction < 0 and rsi_now < 35:
                    should_exit = True

            # 止损逻辑 (仅当use_stops=True)
            if use_stops and not should_exit:
                # 跟踪止损更新
                if not np.isnan(cur_atr) and cur_atr > 0:
                    if direction > 0:
                        if H > ps['hwm']:
                            ps['hwm'] = H
                        if ps['hwm'] >= ps['ep'] + trail_after_atr * ps['atr0']:
                            new_stop = ps['hwm'] - trail_atr * cur_atr
                            if new_stop > ps['stop']:
                                ps['stop'] = new_stop
                    else:
                        if L < ps['hwm']:
                            ps['hwm'] = L
                        if ps['hwm'] <= ps['ep'] - trail_after_atr * ps['atr0']:
                            new_stop = ps['hwm'] + trail_atr * cur_atr
                            if new_stop < ps['stop']:
                                ps['stop'] = new_stop

                # 检查止损触发
                if direction > 0 and L <= ps['stop']:
                    should_exit = True
                    exit_price = ps['stop']
                elif direction < 0 and H >= ps['stop']:
                    should_exit = True
                    exit_price = ps['stop']

            if not should_exit:
                continue

            # 计算PnL
            ml = ps['ml']
            if direction > 0:
                trade_pnl = (exit_price - ps['ep']) * ml * ps['ct']
            else:
                trade_pnl = (ps['ep'] - exit_price) * ml * ps['ct']

            notional_exit = abs(exit_price) * ml * ps['ct']
            comm = COMM * (ps['notional'] + notional_exit)

            pnl = trade_pnl - comm
            eq += pnl
            pnls.append(pnl)
            del pos[sym]

        # === 入场 ===
        if len(pos) < mp:
            sigs = []
            for sym, d in raw.items():
                if sym in pos:
                    continue
                im = si.get(sym)
                if not im or date not in im:
                    continue
                il = im[date]
                if il <= 1:
                    continue

                if mode in ('long', 'both'):
                    sc = signal_score(d, il, 1, rsi_lo, rsi_hi, ch_lo, ch_hi,
                                      cons_min, hv_min, skew_enhance)
                    if sc is not None and sc >= score_min:
                        sigs.append((sym, 1, sc, il))

                if mode in ('short', 'both'):
                    sc = signal_score(d, il, -1, rsi_lo, rsi_hi, ch_lo, ch_hi,
                                      cons_min, hv_min, skew_enhance)
                    if sc is not None and sc >= score_min:
                        sigs.append((sym, -1, sc, il))

            sigs.sort(key=lambda x: -x[2])

            for sym, direction, score, il in sigs:
                if len(pos) >= mp:
                    break

                d = raw[sym]
                entry_price = d['open'][il]
                if np.isnan(entry_price) or entry_price <= 0:
                    continue

                atr = d['atr'][il - 1] if il > 0 else d['atr'][il]
                if np.isnan(atr) or atr <= 0:
                    continue

                ml, mr, _, _ = d['spec']

                if use_stops:
                    # ATR-based sizing
                    if direction > 0:
                        stop = entry_price - stop_atr * atr
                    else:
                        stop = entry_price + stop_atr * atr
                    stop_dist = abs(entry_price - stop)
                    if stop_dist <= 0:
                        continue
                    risk = eq * risk_pct
                    contracts = risk / (stop_dist * ml)
                    contracts = max(int(contracts), 1)
                else:
                    # 无止损: 固定名义值仓位
                    stop = 0.0  # placeholder
                    notional_per = eq * nm / mp
                    contracts = int(notional_per / (entry_price * ml))
                    contracts = max(contracts, 1)

                # 名义值上限
                max_notional = eq * nm / mp
                max_ct = int(max_notional / (entry_price * ml))
                contracts = min(contracts, max(1, max_ct))

                notional = entry_price * ml * contracts

                # 保证金上限
                margin = notional * mr
                if margin > eq * 0.9:
                    contracts = max(int(eq * 0.9 / (entry_price * ml * mr)), 1)
                    notional = entry_price * ml * contracts

                pos[sym] = {
                    'dir': direction,
                    'ed': date,
                    'ep': entry_price,
                    'stop': stop,
                    'hwm': entry_price,
                    'atr0': atr,
                    'ct': contracts,
                    'ml': ml,
                    'notional': notional,
                }

        # === 权益追踪 ===
        ur = 0.
        for sym, ps in pos.items():
            im = si.get(sym)
            if im and date in im:
                S = raw[sym]['close'][im[date]]
                if ps['dir'] > 0:
                    ur += (S - ps['ep']) * ps['ml'] * ps['ct']
                else:
                    ur += (ps['ep'] - S) * ps['ml'] * ps['ct']
        ceq = eq + ur
        eqh.append(ceq)
        if ceq < 1000:
            break

    if not pnls or eq <= 0:
        return None
    tr = (eq - INIT) / INIT
    if tr <= -1:
        return None
    dys = int((dates[-1] - dates[0]) / np.timedelta64(1, 'D'))
    yrs = max(dys / 365, .001)
    ann = float((1 + tr) ** (1 / yrs) - 1)
    pa = np.array(pnls)
    wr = float((pa > 0).mean())
    aw = float(pa[pa > 0].mean()) if (pa > 0).any() else 0
    al = float(abs(pa[pa <= 0].mean())) if (pa <= 0).any() else 1
    pf = aw * (pa > 0).sum() / (al * (pa <= 0).sum()) if (pa <= 0).sum() > 0 and al > 0 else 0
    ea = np.array(eqh[1:])
    if len(ea) > 1:
        cm = np.maximum.accumulate(ea)
        dd = (ea - cm) / cm
        mdd = float(dd.min())
        rets = np.diff(ea) / ea[:-1]
        sh = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0
    else:
        mdd = 0
        sh = 0

    return {'annual': ann, 'wr': wr, 'mdd': mdd, 'pf': pf, 'trades': len(pa),
            'final': eq, 'sharpe': sh, **p}


def main():
    dd = os.path.expanduser("~/home/futures_platform/data/futures_weighted")
    print("加载数据..."); t0 = time.time()
    raw = load(dd)
    print(f"  {len(raw)}品种, {time.time()-t0:.1f}s")

    sd = pd.Timestamp('2018-01-01')
    ed = pd.Timestamp('2026-05-08')
    dates, si = build_idx(raw, sd, ed)
    print(f"  {len(dates)}交易日\n")

    pl = []

    # ====================================================================
    # A: 基线 — 无止损, 纯时间退出 (匹配数据分析方法)
    #    stop_atr=0 → 无止损, 固定名义值仓位
    # ====================================================================
    for rsi_lo in [20, 25, 30, 35]:
        for hd in [5, 7, 10, 15]:
            for nm in [3, 5, 8, 10, 15]:
                pl.append(dict(rsi_lo=rsi_lo, rsi_hi=100-rsi_lo,
                               hold_days=hd, stop_atr=0,
                               notional_mult=nm, mode='both',
                               score_min=0.0, skew_enhance=False))

    # B: 只做多 (MR买入超卖)
    for rsi_lo in [20, 25, 30]:
        for hd in [5, 7, 10]:
            for nm in [5, 8, 12]:
                pl.append(dict(rsi_lo=rsi_lo, rsi_hi=100,
                               hold_days=hd, stop_atr=0,
                               notional_mult=nm, mode='long',
                               score_min=0.0, skew_enhance=False))

    # C: HV百分位过滤 (期权IV代理: 高IV=超卖确认)
    for hv_min in [0.3, 0.4, 0.5, 0.6]:
        for rsi_lo in [25, 30]:
            pl.append(dict(rsi_lo=rsi_lo, rsi_hi=100-rsi_lo,
                           hold_days=10, stop_atr=0,
                           notional_mult=8, mode='both',
                           hv_min=hv_min, score_min=0.0))

    # D: 信号强度过滤
    for score_min in [0.0, 0.2, 0.5, 1.0]:
        pl.append(dict(rsi_lo=25, rsi_hi=75,
                       hold_days=10, stop_atr=0,
                       notional_mult=8, mode='both',
                       score_min=score_min))

    # E: 带ATR止损的版本 (对比)
    for stop_a in [2.0, 3.0, 4.0]:
        for trail_a in [1.5, 2.0]:
            pl.append(dict(rsi_lo=25, rsi_hi=75,
                           hold_days=15, stop_atr=stop_a, trail_atr=trail_a,
                           notional_mult=8, risk_pct=0.05, mode='both',
                           score_min=0.0))

    # F: 极端杠杆 (追求高年化)
    for nm in [10, 15, 20, 25, 30]:
        for rsi_lo in [25, 30]:
            pl.append(dict(rsi_lo=rsi_lo, rsi_hi=100-rsi_lo,
                           hold_days=10, stop_atr=0,
                           notional_mult=nm, mode='both',
                           score_min=0.0, skew_enhance=False))

    print(f"参数组合: {len(pl)}组")
    bt0 = time.time()
    res = []
    for i, p in enumerate(pl):
        if i % 50 == 0:
            print(f"  [{i}/{len(pl)}] {time.time()-bt0:.0f}s...")
        r = bt(raw, dates, si, p)
        if r:
            res.append(r)

    print(f"\n耗时: {time.time()-t0:.0f}s, {len(res)}组有效结果")
    res.sort(key=lambda x: x['annual'], reverse=True)

    # 打印全部结果 (按年化排序)
    print(f"\n{'Mode':>6} {'RSI':>4} {'H':>3} {'Stop':>5} {'Trail':>6} {'NM':>4} {'Risk':>5} "
          f"{'年化':>10} {'胜率':>6} {'PF':>6} {'MDD':>8} {'Sharpe':>7} {'交易':>5}")
    print("-" * 110)
    for r in res[:80]:
        mode_s = r.get('mode', 'both')[:4]
        rsi_s = f"{r.get('rsi_lo', 25)}"
        stop_s = f"{r.get('stop_atr', 2.0):.1f}"
        trail_s = f"{r.get('trail_atr', 1.5):.1f}"
        nm_s = f"{r.get('notional_mult', 8):.0f}"
        risk_s = f"{r.get('risk_pct', 0.03)*100:.0f}%"
        print(f"{mode_s:>6} {rsi_s:>4} {r['hold_days']:>3} {stop_s:>5} {trail_s:>6} {nm_s:>4} {risk_s:>5} "
              f"{r['annual']:>10.1%} {r['wr']:>6.1%} {r['pf']:>6.2f} "
              f"{r['mdd']:>8.1%} {r['sharpe']:>7.2f} {r['trades']:>5}")

    # 目标筛选
    print("\n" + "=" * 110)
    for ta, tw, lb in [(6., .50, "年化>=600% & WR>=50%"),
                       (6., .45, "年化>=600% & WR>=45%"),
                       (5., .50, "年化>=500% & WR>=50%"),
                       (4., .50, "年化>=400% & WR>=50%"),
                       (3., .50, "年化>=300% & WR>=50%"),
                       (2., .50, "年化>=200% & WR>=50%"),
                       (1., .50, "年化>=100% & WR>=50%"),
                       (1., .45, "年化>=100% & WR>=45%")]:
        print(f"\n=== {lb} ===")
        g = [r for r in res if r['annual'] >= ta and r['wr'] >= tw]
        if g:
            for r in sorted(g, key=lambda x: x['annual'], reverse=True)[:15]:
                print(f"  年化={r['annual']:>8.1%}  WR={r['wr']:>5.1%}  PF={r['pf']:>5.2f}  "
                      f"MDD={r['mdd']:>7.1%}  Sharpe={r['sharpe']:>5.2f}  "
                      f"Mode={r.get('mode','both')}  RSI={r.get('rsi_lo',25)}  "
                      f"H={r['hold_days']}  Stop={r.get('stop_atr',2):.1f}  "
                      f"Trail={r.get('trail_atr',1.5):.1f}  "
                      f"NM={r.get('notional_mult',8):.0f}  Risk={r.get('risk_pct',.03)*100:.0f}%  "
                      f"HV>={r.get('hv_min',0):.1f}  Trades={r['trades']}")
        else:
            print("  无")

    # 保存结果
    od = os.path.expanduser("~/home/futures_platform/backtest_results")
    os.makedirs(od, exist_ok=True)
    sv = [{k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
           for k, v in r.items()} for r in res[:500]]
    with open(os.path.join(od, 'backtest_v59.json'), 'w') as f:
        json.dump(sv, f, indent=2, default=str)
    print(f"\n→ backtest_results/backtest_v59.json")


if __name__ == '__main__':
    main()
