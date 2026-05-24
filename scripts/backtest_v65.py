#!/usr/bin/env python3
"""
策略 V65 — 集中仓位 + Kelly + 反马丁格尔

数学基础:
  MR+carry: 57% WR, ~0.8% avg return per trade
  Kelly fraction = 2*WR - 1 = 0.14 (14% per trade)
  Full Kelly with 30 trades/year: (1.07)^30 ≈ 7.6x = 660%

策略:
  1. 集中持仓: max_pos=1~2, 高杠杆
  2. Kelly仓位: 每笔风险 = Kelly% × 权益
  3. 反马丁格尔: 赢后加码, 亏后减码
  4. 回撤自适应: 高水位时全仓, 回撤时减仓
  5. 只做最高信心信号: carry>3% + RSI<25
"""

import os, sys, time, json, math
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from contract_specs import get_spec

COMM = 0.00015
INIT = 500000
TD = 252


def load_carry(ts_dir):
    carry = {}
    for f in os.listdir(ts_dir):
        if not f.endswith('.json'):
            continue
        with open(os.path.join(ts_dir, f)) as fh:
            d = json.load(fh)
        sym = d['symbol'].lower()
        near = d['near_price']
        far = d['far_price']
        if near > 0 and far > 0:
            c = (near - far) / near * 100
            carry[sym] = c
    return carry


def load(data_dir):
    raw = {}
    for f in sorted(os.listdir(data_dir)):
        if not f.endswith('.csv'):
            continue
        sym = f.replace('.csv', '')
        df = pd.read_csv(os.path.join(data_dir, f))
        df['trade_date'] = pd.to_datetime(df['trade_date'], format='mixed')
        df = df.sort_values('trade_date').reset_index(drop=True)
        if len(df) < 300:
            continue
        df['ret'] = df['close'].pct_change()
        for lag in [3, 5, 10, 20, 60]:
            df[f'm{lag}'] = df['close'].pct_change(lag)
        for w in [5, 10, 20, 40, 60]:
            df[f'hv{w}'] = df['ret'].rolling(w).std() * np.sqrt(TD)
        df['hv_pct'] = df['hv20'].rolling(252).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 10 else .5, raw=False)
        df['ma20'] = df['close'].rolling(20).mean()
        d = df['close'].diff()
        g = d.where(d > 0, 0).rolling(14).mean()
        l = (-d.where(d < 0, 0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + g / l))
        down = (df['ret'] < 0).astype(int).values
        cons_d = []
        c = 0
        for v in down:
            c = c + 1 if v else 0
            cons_d.append(c)
        df['cons_down'] = cons_d
        for hold in [5, 10, 15, 20]:
            df[f'fwd{hold}'] = df['close'].shift(-hold) / df['close'] - 1
        df = df.dropna(subset=['ma20', 'hv20', 'rsi'])
        if len(df) < 100:
            continue
        try:
            spec = get_spec(sym)
        except:
            continue
        raw[sym] = {
            'spec': spec,
            'dates': df['trade_date'].values,
            'open': df['open'].values.astype(np.float64),
            'close': df['close'].values.astype(np.float64),
            'rsi': df['rsi'].values.astype(np.float64),
            'cons_down': np.array(cons_d, dtype=np.int32),
            'hv_pct': df['hv_pct'].values.astype(np.float64),
            'hv20': df['hv20'].values.astype(np.float64),
            'ma20': df['ma20'].values.astype(np.float64),
            'fwd10': df['fwd10'].values.astype(np.float64),
            'fwd20': df['fwd20'].values.astype(np.float64),
        }
    return raw


def build_idx(raw, s, e):
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


def select_symbols(raw, train_end, top_pct=0.5):
    te = pd.Timestamp(train_end)
    scores = {}
    for sym, d in raw.items():
        mask = d['dates'] <= te
        idx = np.where(mask)[0]
        if len(idx) < 200:
            continue
        rsi = d['rsi'][idx]
        fwd = d['fwd10'][idx]
        sig_mask = rsi < 25
        n = sig_mask.sum()
        if n < 15:
            continue
        fwd_vals = fwd[sig_mask]
        valid = ~np.isnan(fwd_vals)
        if valid.sum() < 10:
            continue
        wr = (fwd_vals[valid] > 0).mean()
        avg = fwd_vals[valid].mean()
        scores[sym] = wr * max(avg, 0)
    if not scores:
        return []
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    cutoff = max(1, int(len(ranked) * top_pct))
    return [s for s, _ in ranked[:cutoff]]


def bt_aggressive(raw, dates, si, good_syms, carry_map, p):
    """
    集中仓位 + 自适应杠杆

    sizing_mode:
      'fixed'  - 固定NM
      'kelly'  - Kelly criterion
      'anti_martingale' - 赢后加码, 亏后减码
      'drawdown_adapt'  - 根据回撤调整
    """
    mp = p.get('max_pos', 1)
    hd = p.get('hold_days', 10)
    rsi_max = p.get('rsi_max', 25)
    base_nm = p.get('base_nm', 10)
    carry_min = p.get('carry_min', 0)
    sizing = p.get('sizing', 'fixed')
    anti_mart_mult = p.get('anti_mart_mult', 1.5)  # 赢后乘数
    dd_scale = p.get('dd_scale', True)  # 回撤自适应
    min_nm = p.get('min_nm', 3)
    max_nm = p.get('max_nm', 50)

    eq = float(INIT)
    hwm = float(INIT)  # 高水位
    streak = 0  # 连胜/连亏计数
    pos = {}
    pnls = []
    eqh = [float(INIT)]
    trade_details = []

    for date in dates:
        # === 退出 ===
        for sym in list(pos):
            ps = pos[sym]
            im = si.get(sym)
            if not im or date not in im:
                continue
            il = im[date]
            S = raw[sym]['close'][il]
            h = int((date - ps['ed']) / np.timedelta64(1, 'D'))
            if h < hd:
                continue
            ml = ps['ml']
            trade_pnl = (S - ps['ep']) * ml * ps['ct']
            notional_exit = S * ml * ps['ct']
            comm = COMM * (ps['notional'] + notional_exit)
            pnl = trade_pnl - comm
            eq += pnl
            pnls.append(pnl)

            # 更新连胜/连亏
            if pnl > 0:
                streak = max(streak + 1, 1)
            else:
                streak = min(streak - 1, -1)

            trade_details.append({
                'sym': sym, 'pnl': pnl, 'nm_used': ps['nm_used'],
                'entry': ps['ep'], 'exit': float(S), 'hold': h,
            })

            # 更新高水位
            if eq > hwm:
                hwm = eq

            del pos[sym]

        # === 计算当前杠杆 ===
        current_nm = base_nm

        if sizing == 'kelly':
            # Kelly: f = (bp - q) / b where b=win/loss ratio, p=WR, q=1-p
            if len(pnls) > 20:
                recent = pnls[-50:]
                wr = sum(1 for x in recent if x > 0) / len(recent)
                wins = [x for x in recent if x > 0]
                losses = [abs(x) for x in recent if x <= 0]
                if wins and losses:
                    avg_w = sum(wins) / len(wins)
                    avg_l = sum(losses) / len(losses)
                    if avg_l > 0:
                        b = avg_w / avg_l
                        kelly_f = (b * wr - (1 - wr)) / b
                        kelly_f = max(kelly_f, 0.02)
                        # 半Kelly更安全
                        current_nm = int(kelly_f * 100)
                        current_nm = max(min_nm, min(current_nm, max_nm))

        elif sizing == 'anti_martingale':
            if streak > 0:
                current_nm = base_nm * (anti_mart_mult ** min(streak, 5))
            elif streak < 0:
                current_nm = base_nm * (0.5 ** min(abs(streak), 3))
            current_nm = max(min_nm, min(int(current_nm), max_nm))

        # 回撤自适应: 从高水位回撤越多, 杠杆越小
        if dd_scale and hwm > 0:
            dd_from_hwm = (eq - hwm) / hwm
            if dd_from_hwm < -0.1:
                # 回撤10%+, 杠杆减半
                scale = max(0.3, 1 + dd_from_hwm)  # -10%→0.9, -30%→0.7, -50%→0.5
                current_nm = int(current_nm * scale)
                current_nm = max(min_nm, current_nm)

        # === 入场 ===
        if len(pos) < mp:
            sigs = []
            for sym, d in raw.items():
                if sym in pos:
                    continue
                if sym not in good_syms:
                    continue
                im = si.get(sym)
                if not im or date not in im:
                    continue
                il = im[date]
                if il <= 1:
                    continue
                pi = il - 1
                rsi = d['rsi'][pi]
                if np.isnan(rsi):
                    continue
                carry_val = carry_map.get(sym, 0.0)
                if carry_val < carry_min:
                    continue
                if rsi < rsi_max:
                    mr_score = (rsi_max - rsi) / rsi_max
                    carry_score = max(carry_val, 0) / 25.0
                    total_score = 0.5 * mr_score + 0.5 * carry_score
                    sigs.append((sym, total_score, il))

            sigs.sort(key=lambda x: -x[2])

            for sym, score, il in sigs:
                if len(pos) >= mp:
                    break
                d = raw[sym]
                entry_price = d['open'][il]
                if np.isnan(entry_price) or entry_price <= 0:
                    continue
                ml, mr, _, _ = d['spec']

                nm = current_nm
                notional_per = eq * nm / max(mp, 1)
                contracts = int(notional_per / (entry_price * ml))
                contracts = max(contracts, 1)
                notional = entry_price * ml * contracts
                margin = notional * mr
                if margin > eq * 0.9:
                    contracts = max(int(eq * 0.9 / (entry_price * ml * mr)), 1)
                    notional = entry_price * ml * contracts

                pos[sym] = {
                    'dir': 1, 'ed': date, 'ep': entry_price,
                    'ct': contracts, 'ml': ml, 'notional': notional,
                    'nm_used': nm,
                }

        # === 权益 ===
        ur = 0.
        for sym, ps in pos.items():
            im = si.get(sym)
            if im and date in im:
                S = raw[sym]['close'][im[date]]
                ur += (S - ps['ep']) * ps['ml'] * ps['ct']
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
        mdd = 0; sh = 0

    # 最大连胜/连亏
    max_win_streak = 0
    max_loss_streak = 0
    ws = 0
    for pnl in pnls:
        if pnl > 0:
            ws = max(ws + 1, 1)
            max_win_streak = max(max_win_streak, ws)
        else:
            ws = min(ws - 1, -1)
            max_loss_streak = max(max_loss_streak, abs(ws))

    return {
        'annual': ann, 'wr': wr, 'mdd': mdd, 'pf': pf, 'trades': len(pa),
        'final': eq, 'sharpe': sh,
        'max_win_streak': max_win_streak,
        'max_loss_streak': max_loss_streak,
        **p,
    }


def main():
    dd = os.path.expanduser("~/home/futures_platform/data/futures_weighted")
    ts_dir = os.path.expanduser("~/home/futures_platform/data/futures_term_structure")

    print("加载carry数据...")
    carry = load_carry(ts_dir)
    backwd = {s: c for s, c in carry.items() if c > 0}

    print("加载行情数据..."); t0 = time.time()
    raw = load(dd)
    print(f"  {len(raw)}品种, {time.time()-t0:.1f}s")

    train_end = '2022-01-01'
    sym_mr = select_symbols(raw, train_end, top_pct=0.5)
    sym_backwd = [s for s in raw if carry.get(s, 0) > 0]
    sym_mr_backwd = [s for s in sym_mr if carry.get(s, 0) > 0]
    sym_carry3 = [s for s in raw if carry.get(s, 0) > 3]

    print(f"MR top50%: {len(sym_mr)}")
    print(f"MR+backwd: {len(sym_mr_backwd)}")
    print(f"Backwd: {len(sym_backwd)}")
    print(f"carry>3%: {len(sym_carry3)}")

    test_s = pd.Timestamp('2022-01-01')
    test_e = pd.Timestamp('2026-05-08')
    dates, si = build_idx(raw, test_s, test_e)
    print(f"测试期: {len(dates)}交易日\n")

    res = []

    # ===================================================================
    # 1. 固定杠杆, 集中持仓 (max_pos=1, 高NM)
    # ===================================================================
    print("=== 1. 集中持仓 max_pos=1 ===")
    for sym_name, sym_set in [('backwd', sym_backwd), ('c3', sym_carry3),
                               ('mr_backwd', sym_mr_backwd)]:
        for hd in [10, 15]:
            for nm in [10, 15, 20, 30, 40, 50]:
                for carry_min in [0, 3]:
                    p = dict(strategy='concentrated', sym_set=sym_name,
                             hold_days=hd, base_nm=nm, max_pos=1,
                             rsi_max=25, carry_min=carry_min,
                             sizing='fixed', dd_scale=True)
                    r = bt_aggressive(raw, dates, si, sym_set, carry, p)
                    if r:
                        res.append(r)

    # ===================================================================
    # 2. 集中持仓 max_pos=2
    # ===================================================================
    print("=== 2. 集中持仓 max_pos=2 ===")
    for sym_name, sym_set in [('backwd', sym_backwd), ('c3', sym_carry3)]:
        for hd in [10, 15]:
            for nm in [10, 15, 20, 30]:
                p = dict(strategy='concentrated2', sym_set=sym_name,
                         hold_days=hd, base_nm=nm, max_pos=2,
                         rsi_max=25, carry_min=0,
                         sizing='fixed', dd_scale=True)
                r = bt_aggressive(raw, dates, si, sym_set, carry, p)
                if r:
                    res.append(r)

    # ===================================================================
    # 3. 反马丁格尔
    # ===================================================================
    print("=== 3. 反马丁格尔 ===")
    for sym_name, sym_set in [('backwd', sym_backwd), ('c3', sym_carry3)]:
        for hd in [10, 15]:
            for base_nm in [5, 8, 10]:
                for am_mult in [1.3, 1.5, 2.0]:
                    for mp in [1, 2]:
                        p = dict(strategy='anti_mart', sym_set=sym_name,
                                 hold_days=hd, base_nm=base_nm, max_pos=mp,
                                 rsi_max=25, carry_min=0,
                                 sizing='anti_martingale', anti_mart_mult=am_mult,
                                 dd_scale=True)
                        r = bt_aggressive(raw, dates, si, sym_set, carry, p)
                        if r:
                            res.append(r)

    # ===================================================================
    # 4. Kelly仓位
    # ===================================================================
    print("=== 4. Kelly仓位 ===")
    for sym_name, sym_set in [('backwd', sym_backwd), ('c3', sym_carry3)]:
        for hd in [10, 15]:
            for mp in [1, 2, 3]:
                for max_nm in [30, 50, 80]:
                    p = dict(strategy='kelly', sym_set=sym_name,
                             hold_days=hd, base_nm=10, max_pos=mp,
                             rsi_max=25, carry_min=0,
                             sizing='kelly', dd_scale=True, max_nm=max_nm)
                    r = bt_aggressive(raw, dates, si, sym_set, carry, p)
                    if r:
                        res.append(r)

    # ===================================================================
    # 5. 极端杠杆 max_pos=1, NM=50-100 (赌博式)
    # ===================================================================
    print("=== 5. 极端杠杆 ===")
    for sym_name, sym_set in [('backwd', sym_backwd), ('c3', sym_carry3)]:
        for hd in [10]:
            for nm in [50, 75, 100]:
                p = dict(strategy='extreme', sym_set=sym_name,
                         hold_days=hd, base_nm=nm, max_pos=1,
                         rsi_max=25, carry_min=0,
                         sizing='fixed', dd_scale=False)
                r = bt_aggressive(raw, dates, si, sym_set, carry, p)
                if r:
                    res.append(r)

    # ===================================================================
    # 6. 宽松信号 + 集中 (更多交易机会)
    # ===================================================================
    print("=== 6. 宽松信号 + 集中 ===")
    for rsi_max in [30, 35]:
        for hd in [5, 10]:
            for nm in [10, 20, 30]:
                for mp in [1, 2]:
                    p = dict(strategy='loose', sym_set='backwd',
                             hold_days=hd, base_nm=nm, max_pos=mp,
                             rsi_max=rsi_max, carry_min=0,
                             sizing='fixed', dd_scale=True)
                    r = bt_aggressive(raw, dates, si, sym_backwd, carry, p)
                    if r:
                        res.append(r)

    # ===================================================================
    # 7. 反马丁格尔 + 极端 (大起大落)
    # ===================================================================
    print("=== 7. 反马丁极端 ===")
    for sym_name, sym_set in [('backwd', sym_backwd)]:
        for hd in [10]:
            for base_nm in [10, 15, 20]:
                for am_mult in [2.0, 3.0]:
                    p = dict(strategy='anti_mart_extreme', sym_set=sym_name,
                             hold_days=hd, base_nm=base_nm, max_pos=1,
                             rsi_max=25, carry_min=0,
                             sizing='anti_martingale', anti_mart_mult=am_mult,
                             dd_scale=False, min_nm=5, max_nm=200)
                    r = bt_aggressive(raw, dates, si, sym_set, carry, p)
                    if r:
                        res.append(r)

    print(f"\n总结果: {len(res)}组有效")
    res.sort(key=lambda x: x['annual'], reverse=True)

    print(f"\n{'Strat':>18} {'Sym':>8} {'H':>3} {'NM':>4} {'MP':>3} "
          f"{'年化':>10} {'胜率':>6} {'PF':>6} {'MDD':>8} {'Sharpe':>7} {'交易':>5} {'胜连':>4} {'亏连':>4}")
    print("-" * 115)
    for r in res[:80]:
        strat = r.get('strategy', '?')
        sym = r.get('sym_set', '?')
        nm = r.get('base_nm', 0)
        if r.get('sizing') == 'anti_martingale':
            nm_s = f"AM{nm}"
        elif r.get('sizing') == 'kelly':
            nm_s = f"K{nm}"
        else:
            nm_s = f"{nm}"
        print(f"{strat:>18} {sym:>8} {r.get('hold_days','-'):>3} "
              f"{nm_s:>4} {r.get('max_pos',1):>3} "
              f"{r['annual']:>10.1%} {r['wr']:>6.1%} {r['pf']:>6.2f} "
              f"{r['mdd']:>8.1%} {r['sharpe']:>7.2f} {r['trades']:>5} "
              f"{r.get('max_win_streak',0):>4} {r.get('max_loss_streak',0):>4}")

    # 分策略
    print("\n=== 分策略最佳 ===")
    for strat_name in ['concentrated', 'concentrated2', 'anti_mart', 'kelly', 'extreme', 'loose', 'anti_mart_extreme']:
        strat_res = [r for r in res if r.get('strategy') == strat_name]
        if not strat_res:
            continue
        print(f"\n--- {strat_name} Top 5 ---")
        for r in strat_res[:5]:
            nm = r.get('base_nm', 0)
            sizing = r.get('sizing', 'fixed')
            print(f"  年化={r['annual']:>8.1%}  WR={r['wr']:>5.1%}  PF={r['pf']:>5.2f}  "
                  f"MDD={r['mdd']:>7.1%}  Sharpe={r['sharpe']:>5.2f}  "
                  f"H={r.get('hold_days',0)}  NM={nm}({sizing})  MP={r.get('max_pos',1)}  "
                  f"{r.get('sym_set','')}  Trades={r['trades']}  "
                  f"连胜={r.get('max_win_streak',0)}  连亏={r.get('max_loss_streak',0)}")

    # 目标
    print("\n" + "=" * 115)
    for ta, tw, lb in [(6., .50, "年化>=600%"),
                       (3., .50, "年化>=300%"),
                       (2., .50, "年化>=200%"),
                       (1., .50, "年化>=100%"),
                       (0.5, .50, "年化>=50%")]:
        g = [r for r in res if r['annual'] >= ta and r['wr'] >= tw]
        print(f"\n=== {lb} & WR>=50%: {len(g)}组 ===")
        if g:
            for r in sorted(g, key=lambda x: x['annual'], reverse=True)[:10]:
                nm = r.get('base_nm', 0)
                sizing = r.get('sizing', 'fixed')
                print(f"    年化={r['annual']:>8.1%}  WR={r['wr']:>5.1%}  MDD={r['mdd']:>7.1%}  "
                      f"Sharpe={r['sharpe']:>5.2f}  "
                      f"{r.get('strategy','')} H={r.get('hold_days',0)} "
                      f"NM={nm}({sizing}) MP={r.get('max_pos',1)} "
                      f"{r.get('sym_set','')}  Trades={r['trades']}  "
                      f"连胜={r.get('max_win_streak',0)}")

    od = os.path.expanduser("~/home/futures_platform/backtest_results")
    os.makedirs(od, exist_ok=True)
    sv = [{k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
           for k, v in r.items()} for r in res[:2000]]
    with open(os.path.join(od, 'backtest_v65.json'), 'w') as f:
        json.dump(sv, f, indent=2, default=str)
    print(f"\n→ backtest_results/backtest_v65.json")


if __name__ == '__main__':
    main()
