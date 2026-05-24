#!/usr/bin/env python3
"""
策略 V62 — MR信号 + 期货期限结构carry信号 双因子策略

两个独立edge来源:
  1. MR (Mean Reversion): RSI<25 → 超卖反弹, ~53% WR
  2. Carry (期限结构): Backwardation → 正carry, 学术证据5-10%年化超额

期限结构数据仅2026-05-15单日快照, 作为静态品种分类:
  - 某些品种有结构性backwardation (仓储约束、季节性生产等)
  - 这些特征持久, 可用于品种筛选

策略设计:
  - 做多: MR信号(RSI<25) + 高carry(backwardation) → 双确认
  - carry作为品种筛选: 优先交易backwardated品种
  - 对比: 纯MR vs MR+carry, 量化carry的独立贡献
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
    """加载期限结构carry数据, 返回 {symbol: carry_pct}"""
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
            # carry = (near - far) / near, 正数=backwardation=正carry
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
        df['trade_date'] = pd.to_datetime(df['trade_date'])
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

        df['ma5'] = df['close'].rolling(5).mean()
        df['ma20'] = df['close'].rolling(20).mean()
        df['ma60'] = df['close'].rolling(60).mean()

        # RSI
        d = df['close'].diff()
        g = d.where(d > 0, 0).rolling(14).mean()
        l = (-d.where(d < 0, 0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + g / l))

        # 连续下跌
        down = (df['ret'] < 0).astype(int).values
        cons_d = []
        c = 0
        for v in down:
            c = c + 1 if v else 0
            cons_d.append(c)
        df['cons_down'] = cons_d

        # 通道位置
        ch_hi = df['close'].rolling(20).max()
        ch_lo = df['close'].rolling(20).min()
        ch_range = (ch_hi - ch_lo).replace(0, np.nan)
        df['ch_pos'] = ((df['close'] - ch_lo) / ch_range).clip(0, 1)

        # 前向收益
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
            'ch_pos': df['ch_pos'].values.astype(np.float64),
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


def select_symbols(raw, train_end, signal='mr', top_pct=0.5):
    te = pd.Timestamp(train_end)
    scores = {}
    for sym, d in raw.items():
        mask = d['dates'] <= te
        idx = np.where(mask)[0]
        if len(idx) < 200:
            continue
        rsi = d['rsi'][idx]
        fwd = d['fwd10'][idx]
        if signal == 'mr':
            sig_mask = rsi < 25
        elif signal == 'mr_cons':
            sig_mask = (d['cons_down'][idx] >= 3) & (rsi < 35)
        elif signal == 'mr_tight':
            sig_mask = rsi < 20
        else:
            sig_mask = rsi < 25
        n = sig_mask.sum()
        if n < 15:
            continue
        fwd_vals = fwd[sig_mask]
        valid = ~np.isnan(fwd_vals)
        n_valid = valid.sum()
        if n_valid < 10:
            continue
        wr = (fwd_vals[valid] > 0).mean()
        avg = fwd_vals[valid].mean()
        score = wr * max(avg, 0)
        scores[sym] = {'wr': wr, 'avg': avg, 'n': n_valid, 'score': score}
    if not scores:
        return []
    ranked = sorted(scores.items(), key=lambda x: -x[1]['score'])
    cutoff = max(1, int(len(ranked) * top_pct))
    return [s for s, _ in ranked[:cutoff]]


def bt(raw, dates, si, good_syms, carry_map, p):
    """
    V62回测: MR + carry双因子

    carry_mode:
      'none'   - 不使用carry (纯MR基线)
      'filter' - 只交易backwardated品种
      'boost'  - carry作为信号加权 (高carry排名优先)
      'require'- MR信号 + carry正 双重确认
    """
    mp = p.get('max_pos', 3)
    hd = p.get('hold_days', 15)
    rsi_max = p.get('rsi_max', 25)
    cons_min = p.get('cons_min', 0)
    hv_max = p.get('hv_max', 1.0)
    nm = p.get('notional_mult', 10.0)
    use_filter = p.get('use_filter', True)
    short_side = p.get('short_side', False)
    carry_mode = p.get('carry_mode', 'none')
    carry_min = p.get('carry_min', 0.0)  # carry最低要求(%)
    carry_weight = p.get('carry_weight', 0.5)  # carry在评分中的权重

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
            h = int((date - ps['ed']) / np.timedelta64(1, 'D'))
            if h < hd:
                continue
            ml = ps['ml']
            if ps['dir'] > 0:
                trade_pnl = (S - ps['ep']) * ml * ps['ct']
            else:
                trade_pnl = (ps['ep'] - S) * ml * ps['ct']
            notional_exit = S * ml * ps['ct']
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
                if use_filter and sym not in good_syms:
                    continue
                im = si.get(sym)
                if not im or date not in im:
                    continue
                il = im[date]
                if il <= 1:
                    continue
                pi = il - 1

                rsi = d['rsi'][pi]
                cons = d['cons_down'][pi]
                hv_pct = d['hv_pct'][pi]

                if np.isnan(rsi) or np.isnan(hv_pct):
                    continue

                # 获取carry值
                carry_val = carry_map.get(sym, 0.0)

                # === 做多信号 ===
                if rsi < rsi_max:
                    if cons_min > 0 and cons < cons_min:
                        continue
                    if hv_pct > hv_max:
                        continue

                    # carry过滤逻辑
                    if carry_mode == 'filter':
                        if carry_val < carry_min:
                            continue
                    elif carry_mode == 'require':
                        if carry_val <= 0:
                            continue

                    # 信号评分
                    mr_score = (rsi_max - rsi) / rsi_max  # MR越强分越高
                    carry_score = max(carry_val, 0) / 25.0  # carry标准化 (25%=1.0)

                    if carry_mode == 'boost':
                        total_score = (1 - carry_weight) * mr_score + carry_weight * carry_score
                    elif carry_mode in ('filter', 'require'):
                        total_score = mr_score + carry_score  # carry作为额外加分
                    else:
                        total_score = mr_score

                    sigs.append((sym, 1, total_score, il))

                # === 做空信号 ===
                if short_side and rsi > (100 - rsi_max):
                    if hv_pct > hv_max:
                        continue
                    # 做空需要contango (负carry有利做空)
                    if carry_mode in ('filter', 'require') and carry_val >= 0:
                        continue
                    mr_score = (rsi - (100 - rsi_max)) / (100 - rsi_max)
                    total_score = mr_score
                    sigs.append((sym, -1, total_score, il))

            sigs.sort(key=lambda x: -x[2])

            for sym, direction, score, il in sigs:
                if len(pos) >= mp:
                    break
                d = raw[sym]
                entry_price = d['open'][il]
                if np.isnan(entry_price) or entry_price <= 0:
                    continue
                ml, mr, _, _ = d['spec']
                notional_per = eq * nm / mp
                contracts = int(notional_per / (entry_price * ml))
                contracts = max(contracts, 1)
                notional = entry_price * ml * contracts
                margin = notional * mr
                if margin > eq * 0.9:
                    contracts = max(int(eq * 0.9 / (entry_price * ml * mr)), 1)
                    notional = entry_price * ml * contracts
                pos[sym] = {
                    'dir': direction, 'ed': date, 'ep': entry_price,
                    'ct': contracts, 'ml': ml, 'notional': notional, 'hd': hd,
                }

        # === 权益 ===
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
        mdd = 0; sh = 0

    return {'annual': ann, 'wr': wr, 'mdd': mdd, 'pf': pf, 'trades': len(pa),
            'final': eq, 'sharpe': sh, 'n_sym': len(good_syms), **p}


def main():
    dd = os.path.expanduser("~/home/futures_platform/data/futures_weighted")
    ts_dir = os.path.expanduser("~/home/futures_platform/data/futures_term_structure")

    print("加载carry数据...")
    carry = load_carry(ts_dir)
    # carry: 正=backwardation(正carry), 负=contango(负carry)
    backwd = {s: c for s, c in carry.items() if c > 0}
    contg = {s: c for s, c in carry.items() if c <= 0}
    print(f"  Backwardation: {len(backwd)}品种")
    print(f"    Top10: {sorted(backwd.items(), key=lambda x: -x[1])[:10]}")
    print(f"  Contango: {len(contg)}品种")

    print("加载行情数据..."); t0 = time.time()
    raw = load(dd)
    print(f"  {len(raw)}品种, {time.time()-t0:.1f}s")

    # 只保留有carry数据的品种
    raw_with_carry = {s: d for s, d in raw.items() if s in carry}
    print(f"  有carry数据的品种: {len(raw_with_carry)}")

    train_end = '2022-01-01'

    # 走前品种选择
    sym_mr = select_symbols(raw, train_end, 'mr', top_pct=0.5)
    print(f"\nMR (RSI<25) top50%: {len(sym_mr)}品种")

    # 走前品种选择 + 只看backwardated品种
    sym_mr_carry = [s for s in sym_mr if s in backwd]
    print(f"MR top50% + backwardated: {len(sym_mr_carry)}品种")

    # 所有backwardated品种 (不过滤MR)
    sym_backwd_all = [s for s in raw if s in backwd]
    print(f"所有backwardated品种: {len(sym_backwd_all)}品种")

    # 高carry品种 (carry > 5%)
    sym_high_carry = [s for s in raw if carry.get(s, 0) > 5]
    print(f"高carry品种(>5%): {len(sym_high_carry)}品种 → {sym_high_carry}")

    # 测试期
    test_s = pd.Timestamp('2022-01-01')
    test_e = pd.Timestamp('2026-05-08')
    dates, si = build_idx(raw, test_s, test_e)
    print(f"\n测试期: {len(dates)}交易日")

    pl = []

    # ===================================================================
    # A: 纯MR基线 (不使用carry, 对比用)
    # ===================================================================
    for hd in [10, 15, 20]:
        for nm in [3, 5, 8, 10, 15, 20]:
            pl.append(dict(hold_days=hd, rsi_max=25, notional_mult=nm,
                           use_filter=True, carry_mode='none'))

    # ===================================================================
    # B: carry_filter — 只交易backwardated品种 (carry>0)
    # ===================================================================
    for hd in [10, 15, 20]:
        for nm in [3, 5, 8, 10, 15, 20]:
            pl.append(dict(hold_days=hd, rsi_max=25, notional_mult=nm,
                           use_filter=True, carry_mode='filter', carry_min=0))

    # ===================================================================
    # C: carry_filter严格 — 只交易高carry品种 (carry>3%)
    # ===================================================================
    for hd in [10, 15, 20]:
        for nm in [3, 5, 8, 10, 15, 20]:
            pl.append(dict(hold_days=hd, rsi_max=25, notional_mult=nm,
                           use_filter=True, carry_mode='filter', carry_min=3))

    # ===================================================================
    # D: carry_require — MR + carry双重确认 (必须backwardation)
    # ===================================================================
    for hd in [10, 15, 20]:
        for nm in [5, 8, 10, 15, 20]:
            pl.append(dict(hold_days=hd, rsi_max=25, notional_mult=nm,
                           use_filter=False, carry_mode='require'))

    # ===================================================================
    # E: carry_boost — carry加权排名 (MR + carry共同排名)
    # ===================================================================
    for cw in [0.3, 0.5, 0.7]:
        for hd in [10, 15, 20]:
            for nm in [5, 8, 10, 15, 20]:
                pl.append(dict(hold_days=hd, rsi_max=25, notional_mult=nm,
                               use_filter=True, carry_mode='boost', carry_weight=cw))

    # ===================================================================
    # F: 高carry品种 + 无品种过滤 + MR
    # ===================================================================
    for hd in [10, 15, 20]:
        for nm in [5, 8, 10, 15]:
            pl.append(dict(hold_days=hd, rsi_max=25, notional_mult=nm,
                           use_filter=False, carry_mode='filter', carry_min=5))

    # ===================================================================
    # G: carry + HV过滤 + 更长持有
    # ===================================================================
    for hd in [15, 20]:
        for nm in [5, 8, 10, 15]:
            pl.append(dict(hold_days=hd, rsi_max=25, hv_max=0.4,
                           notional_mult=nm, use_filter=True,
                           carry_mode='filter', carry_min=0))

    # ===================================================================
    # H: 双向 (做多backwardation+超卖, 做空contango+超买)
    # ===================================================================
    for hd in [10, 15]:
        for nm in [5, 8, 10]:
            pl.append(dict(hold_days=hd, rsi_max=25, notional_mult=nm,
                           use_filter=True, short_side=True,
                           carry_mode='filter', carry_min=0))

    # ===================================================================
    # I: 极端杠杆 (carry筛选 + 高NM)
    # ===================================================================
    for hd in [15, 20]:
        for nm in [25, 30, 40, 50]:
            pl.append(dict(hold_days=hd, rsi_max=25, notional_mult=nm,
                           use_filter=True, carry_mode='filter', carry_min=0))

    # ===================================================================
    # J: 宽松MR + carry (RSI<35)
    # ===================================================================
    for hd in [10, 15, 20]:
        for nm in [5, 10, 15]:
            pl.append(dict(hold_days=hd, rsi_max=35, notional_mult=nm,
                           use_filter=True, carry_mode='filter', carry_min=0))

    print(f"\n参数组合: {len(pl)}组")
    bt0 = time.time()
    res = []

    for i, p in enumerate(pl):
        if i % 50 == 0:
            print(f"  [{i}/{len(pl)}] {time.time()-bt0:.0f}s...")

        cm = p.get('carry_mode', 'none')

        # 根据carry模式选择品种集
        if cm == 'none':
            good = sym_mr
        elif cm in ('filter', 'boost'):
            cmin = p.get('carry_min', 0)
            # 从MR品种中筛选carry
            if p.get('use_filter', True):
                good = [s for s in sym_mr if carry.get(s, 0) >= cmin]
            else:
                good = [s for s in raw if carry.get(s, 0) >= cmin]
        elif cm == 'require':
            # 任何backwardated品种 + MR信号
            good = [s for s in raw if carry.get(s, 0) > 0]
        else:
            good = sym_mr

        if not good:
            continue

        r = bt(raw, dates, si, good, carry, p)
        if r:
            res.append(r)

    print(f"\n耗时: {time.time()-t0:.0f}s, {len(res)}组有效结果")
    res.sort(key=lambda x: x['annual'], reverse=True)

    # 输出结果
    print(f"\n{'Carry':>8} {'RSI':>4} {'H':>3} {'HV':>4} {'NM':>4} {'Cmin':>5} "
          f"{'年化':>10} {'胜率':>6} {'PF':>6} {'MDD':>8} {'Sharpe':>7} {'交易':>5}")
    print("-" * 110)
    for r in res[:80]:
        cm = r.get('carry_mode', 'none')
        cw_s = f"bw{r.get('carry_weight', 0):.1f}" if cm == 'boost' else ""
        short = "+S" if r.get('short_side') else ""
        hv_s = f"<{r.get('hv_max',1):.1f}" if r.get('hv_max', 1) < 1 else "off"
        cmin = r.get('carry_min', 0)
        filt = "Y" if r.get('use_filter', True) else "N"
        print(f"{cm:>6}{cw_s:>2} {r.get('rsi_max',25):>4} {r['hold_days']:>3} {hv_s:>4} "
              f"{r.get('notional_mult',10):>4.0f} {cmin:>4.0f}% "
              f"{r['annual']:>10.1%} {r['wr']:>6.1%} {r['pf']:>6.2f} "
              f"{r['mdd']:>8.1%} {r['sharpe']:>7.2f} {r['trades']:>5}{short}")

    # 分组统计: carry vs no-carry
    print("\n" + "=" * 110)
    print("\n=== Carry模式对比 (取每组最佳前5) ===")
    for mode in ['none', 'filter', 'require', 'boost']:
        mode_res = [r for r in res if r.get('carry_mode', 'none') == mode]
        if not mode_res:
            continue
        print(f"\n--- {mode} ---")
        for r in mode_res[:5]:
            cw_s = f" cw={r.get('carry_weight',0):.1f}" if mode == 'boost' else ""
            cmin_s = f" cmin={r.get('carry_min',0):.0f}%" if mode in ('filter',) else ""
            hv_s = f" HV<{r.get('hv_max',1):.1f}" if r.get('hv_max',1) < 1 else ""
            print(f"  年化={r['annual']:>8.1%}  WR={r['wr']:>5.1%}  PF={r['pf']:>5.2f}  "
                  f"MDD={r['mdd']:>7.1%}  Sharpe={r['sharpe']:>5.2f}  "
                  f"H={r['hold_days']}  RSI<{r.get('rsi_max',25)}  "
                  f"NM={r.get('notional_mult',10):.0f}  "
                  f"Trades={r['trades']}{cw_s}{cmin_s}{hv_s}")

    # 目标达成统计
    print("\n" + "=" * 110)
    for ta, tw, lb in [(6., .50, "年化>=600% & WR>=50%"),
                       (3., .50, "年化>=300% & WR>=50%"),
                       (2., .50, "年化>=200% & WR>=50%"),
                       (1., .50, "年化>=100% & WR>=50%"),
                       (0.5, .50, "年化>=50% & WR>=50%"),
                       (0., .50, "正收益 & WR>=50%"),
                       (0., .48, "正收益 & WR>=48%")]:
        print(f"\n=== {lb} ===")
        g = [r for r in res if r['annual'] >= ta and r['wr'] >= tw]
        if g:
            for r in sorted(g, key=lambda x: x['annual'], reverse=True)[:15]:
                cm = r.get('carry_mode', 'none')
                cw_s = f" cw={r.get('carry_weight',0):.1f}" if cm == 'boost' else ""
                cmin_s = f" cmin={r.get('carry_min',0):.0f}%" if cm == 'filter' else ""
                hv_s = f" HV<{r.get('hv_max',1):.1f}" if r.get('hv_max',1) < 1 else ""
                print(f"  年化={r['annual']:>8.1%}  WR={r['wr']:>5.1%}  PF={r['pf']:>5.2f}  "
                      f"MDD={r['mdd']:>7.1%}  Sharpe={r['sharpe']:>5.2f}  "
                      f"H={r['hold_days']}  RSI<{r.get('rsi_max',25)}  "
                      f"NM={r.get('notional_mult',10):.0f}  "
                      f"carry={cm}{cw_s}{cmin_s}{hv_s}  Trades={r['trades']}")
        else:
            print("  无")

    od = os.path.expanduser("~/home/futures_platform/backtest_results")
    os.makedirs(od, exist_ok=True)
    sv = [{k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
           for k, v in r.items()} for r in res[:800]]
    with open(os.path.join(od, 'backtest_v62.json'), 'w') as f:
        json.dump(sv, f, indent=2, default=str)
    print(f"\n→ backtest_results/backtest_v62.json")


if __name__ == '__main__':
    main()
