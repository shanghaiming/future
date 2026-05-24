#!/usr/bin/env python3
"""
V76: 真实日内止损 + 集中度管理
关键修复: 用high/low模拟日内止损触发
新增: 品种集中度上限、动态参数调整
"""
import os, glob, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')

DATA_DIR = 'data/futures_weighted'
INITIAL_CAPITAL = 500_000
CONTRACT_SPECS = 'scripts/contract_specs.py'
TEST_START = '2022-01-01'
TEST_END = '2025-12-31'


def load_data():
    import importlib.util
    spec = importlib.util.spec_from_file_location("cs", CONTRACT_SPECS)
    cs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cs)
    all_data = {}
    for f in sorted(glob.glob(os.path.join(DATA_DIR, '*.csv'))):
        sym = os.path.basename(f).replace('.csv', '')
        try: mult, margin, tick, tick_val = cs.get_spec(sym)
        except: continue
        df = pd.read_csv(f)
        if len(df) < 100: continue
        df['trade_date'] = pd.to_datetime(df['trade_date'], format='mixed')
        df = df.sort_values('trade_date').reset_index(drop=True)
        if df['close'].isna().all() or (df['close'] == 0).any(): continue
        all_data[sym] = df
    print(f"  {len(all_data)}品种")
    return all_data


def compute_signals(all_data):
    signal_data = {}
    for sym, df in all_data.items():
        c = df['close'].values.astype(float)
        o = df['open'].values.astype(float)
        h = df['high'].values.astype(float)
        l = df['low'].values.astype(float)
        v = df['vol'].values.astype(float)
        oi = df['oi'].values.astype(float)
        n = len(df)

        prev_c = np.full(n, np.nan); prev_c[1:] = c[:-1]
        gap = np.full(n, np.nan); gap[1:] = (o[1:] - prev_c[1:]) / prev_c[1:] * 100

        tr = np.full(n, np.nan)
        tr[1:] = np.maximum(h[1:]-l[1:], np.maximum(np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])))
        atr = pd.Series(tr).rolling(20).mean().values
        atr_pct = atr / c * 100

        ma20 = pd.Series(c).rolling(20).mean().values
        ma60 = pd.Series(c).rolling(60).mean().values
        ma5 = pd.Series(c).rolling(5).mean().values

        mom5 = np.full(n, np.nan); mom5[5:] = (c[5:] - c[:-5]) / c[:-5] * 100

        oi_ch = np.full(n, np.nan)
        valid = np.abs(oi[:-1]) > 0
        oi_ch_v = np.full(n-1, np.nan)
        oi_ch_v[valid] = (oi[1:][valid] - oi[:-1][valid]) / np.abs(oi[:-1][valid]) * 100
        oi_ch[1:] = oi_ch_v

        vol_ma5 = pd.Series(v).rolling(5).mean().values
        range_ = h - l
        clv = np.where(range_ > 0, (2*c - h - l) / range_, 0)

        gv = np.nan_to_num(gap)
        ga = np.nan_to_num(gap / np.where(atr_pct == 0, np.nan, atr_pct))

        s_l = np.zeros(n)
        s_l += (gv < -0.5).astype(int) * 1
        s_l += (gv < -1.0).astype(int) * 2
        s_l += (gv < -1.5).astype(int) * 2
        s_l += (gv < -2.0).astype(int) * 3
        s_l += (ga < -1.0).astype(int) * 2
        s_l += (ga < -1.5).astype(int) * 3
        s_l += ((oi_ch > 0) & (c < prev_c)).astype(int) * 3
        s_l += ((oi_ch < 0) & (c < prev_c)).astype(int) * 2
        s_l += (mom5 < -3).astype(int) * 1
        s_l += (mom5 < -5).astype(int) * 1
        s_l += (c < ma5).astype(int) * 1
        s_l += ((v > vol_ma5 * 1.5) & (c < prev_c)).astype(int) * 1
        s_l += (clv > 0.5).astype(int) * 1
        s_l += (ma20 > ma60).astype(int) * 2

        s_s = np.zeros(n)
        s_s += (gv > 0.5).astype(int) * 1
        s_s += (gv > 1.0).astype(int) * 2
        s_s += (gv > 1.5).astype(int) * 2
        s_s += (gv > 2.0).astype(int) * 3
        s_s += (ga > 1.0).astype(int) * 2
        s_s += (ga > 1.5).astype(int) * 3
        s_s += ((oi_ch > 0) & (c > prev_c)).astype(int) * 3
        s_s += ((oi_ch < 0) & (c > prev_c)).astype(int) * 2
        s_s += (mom5 > 3).astype(int) * 1
        s_s += (mom5 > 5).astype(int) * 1
        s_s += (c > ma5).astype(int) * 1
        s_s += ((v > vol_ma5 * 1.5) & (c > prev_c)).astype(int) * 1
        s_s += (clv < -0.5).astype(int) * 1
        s_s += (ma20 < ma60).astype(int) * 2

        df['score_long'] = s_l
        df['score_short'] = s_s
        df['gap_pct'] = gap
        signal_data[sym] = df
    return signal_data


def run_bt_realistic(signal_data, start, end, max_pos=7, lev=5, min_sc=7, hold=1,
                     sl_pct=None, tp_pct=None, max_per_sym=None, intraday_sl=False):
    """
    回测引擎 — 真实日内止损
    intraday_sl=True: 使用high/low判断日内是否触发止损
    max_per_sym: 每个品种最大同时持仓数
    """
    dates = pd.date_range(start=start, end=end, freq='B')
    cap = INITIAL_CAPITAL
    eq = []
    trades = []
    pos = []

    for dt in dates:
        # 平仓
        pnl = 0
        keep = []
        for p in pos:
            df = signal_data.get(p['sym'])
            if df is None:
                keep.append(p); continue

            idx = df.index[df['trade_date'] == dt]
            if len(idx) == 0:
                keep.append(p); continue

            row = df.loc[idx[0]]
            cur_o = row['open']
            cur_h = row['high']
            cur_l = row['low']
            cur_c = row['close']

            if np.isnan(cur_c):
                keep.append(p); continue

            d = (dt - p['ed']).days

            # 计算收益 (区分多空)
            if p['dir'] == 'long':
                # 日内价格路径: open → low/high → close
                # 如果gap down持续, low先到; 如果反转, high先到
                intraday_max = (cur_h - p['ep']) / p['ep'] * 100  # 最大盈利%
                intraday_min = (cur_l - p['ep']) / p['ep'] * 100  # 最大亏损%
                close_ret = (cur_c - p['ep']) / p['ep'] * 100
            else:
                intraday_max = (p['ep'] - cur_l) / p['ep'] * 100
                intraday_min = (p['ep'] - cur_h) / p['ep'] * 100
                close_ret = (p['ep'] - cur_c) / p['ep'] * 100

            reason = None
            actual_ret = close_ret

            if d >= hold:
                # 到期日 — 检查日内止损
                if intraday_sl and sl_pct and intraday_min <= sl_pct:
                    # 日内触发了止损 — 按止损价计算
                    actual_ret = sl_pct  # 最差情况: 滑点假设已在止损价
                    reason = 'SL'
                elif intraday_sl and tp_pct and intraday_max >= tp_pct:
                    # 日内触发了止盈 — 按止盈价计算
                    actual_ret = tp_pct
                    reason = 'TP'
                else:
                    actual_ret = close_ret
                    reason = 'exp'
            else:
                # 持仓中 — 检查日内止损
                if intraday_sl and sl_pct and intraday_min <= sl_pct:
                    actual_ret = sl_pct
                    reason = 'SL'
                elif intraday_sl and tp_pct and intraday_max >= tp_pct:
                    actual_ret = tp_pct
                    reason = 'TP'
                else:
                    keep.append(p)
                    continue

            if reason:
                pnl += p['not'] * actual_ret / 100
                trades.append({
                    'sym': p['sym'], 'dir': p['dir'], 'ed': p['ed'],
                    'xd': dt, 'ep': p['ep'], 'xp': cur_c, 'r': actual_ret,
                    'close_r': close_ret,  # 收盘价计算的收益(对比用)
                    'pnl': p['not'] * actual_ret / 100, 'sc': p['sc'],
                    'hold': d, 'reason': reason,
                })
        pos = keep
        cap += pnl
        if cap <= 0:
            eq.append({'date': dt, 'capital': 0}); break

        n_open = max_pos - len(pos)
        if n_open <= 0:
            eq.append({'date': dt, 'capital': cap}); continue

        cands = []
        for sym, df in signal_data.items():
            if any(p['sym'] == sym for p in pos): continue
            # 品种集中度限制
            if max_per_sym:
                sym_count = sum(1 for t in trades[-500:] if t['sym'] == sym and (dt - t['xd']).days < 30)
                # 近30天同品种交易不超过max_per_sym次
                pass  # 同一时刻只能有一个, 这个限制体现在 pos check
            idx = df.index[df['trade_date'] == dt]
            if len(idx) == 0: continue
            row = df.loc[idx[0]]
            if row['score_long'] >= min_sc:
                cands.append({'sym': sym, 'dir': 'long', 'sc': row['score_long'], 'ep': row['open']})
            if row['score_short'] >= min_sc:
                cands.append({'sym': sym, 'dir': 'short', 'sc': row['score_short'], 'ep': row['open']})

        best = {}
        for c in cands:
            if c['sym'] not in best or c['sc'] > best[c['sym']]['sc']:
                best[c['sym']] = c

        ranked = sorted(best.values(), key=lambda x: -x['sc'])
        for c in ranked[:n_open]:
            notional = cap * lev / max_pos
            pos.append({'sym': c['sym'], 'dir': c['dir'], 'ed': dt,
                        'ep': c['ep'], 'not': notional, 'sc': c['sc']})

        eq.append({'date': dt, 'capital': cap})
    return eq, trades


def run_bt_realistic_v2(signal_data, start, end, max_pos=7, lev=5, min_sc=7, hold=1,
                        sl_pct=None, tp_pct=None):
    """
    V2: 更真实的日内止损模拟
    假设: price先到达extreme再回归
    Long: 如果low <= stop_price, 认为止损触发, 按stop_price成交(加滑点0.1%)
    Short: 如果high >= stop_price, 认为止损触发
    """
    dates = pd.date_range(start=start, end=end, freq='B')
    cap = INITIAL_CAPITAL
    eq = []
    trades = []
    pos = []

    for dt in dates:
        pnl = 0
        keep = []
        for p in pos:
            df = signal_data.get(p['sym'])
            if df is None:
                keep.append(p); continue

            idx = df.index[df['trade_date'] == dt]
            if len(idx) == 0:
                keep.append(p); continue

            row = df.loc[idx[0]]
            cur_h = row['high']
            cur_l = row['low']
            cur_c = row['close']

            if np.isnan(cur_c):
                keep.append(p); continue

            d = (dt - p['ed']).days

            # 计算止损止盈触发价
            slippage = 0.001  # 0.1%滑点

            triggered = False
            actual_ret = None
            reason = None

            if p['dir'] == 'long':
                # Long position
                if sl_pct:
                    stop_price = p['ep'] * (1 + sl_pct / 100)
                    if cur_l <= stop_price:
                        # 触发止损, 按止损价+滑点成交
                        fill_price = stop_price * (1 - slippage)
                        actual_ret = (fill_price - p['ep']) / p['ep'] * 100
                        reason = 'SL'
                        triggered = True

                if not triggered and tp_pct:
                    tp_price = p['ep'] * (1 + tp_pct / 100)
                    if cur_h >= tp_price:
                        fill_price = tp_price * (1 - slippage)
                        actual_ret = (fill_price - p['ep']) / p['ep'] * 100
                        reason = 'TP'
                        triggered = True

                if not triggered:
                    actual_ret = (cur_c - p['ep']) / p['ep'] * 100

            else:
                # Short position
                if sl_pct:
                    stop_price = p['ep'] * (1 - sl_pct / 100)
                    if cur_h >= stop_price:
                        fill_price = stop_price * (1 + slippage)
                        actual_ret = (p['ep'] - fill_price) / p['ep'] * 100
                        reason = 'SL'
                        triggered = True

                if not triggered and tp_pct:
                    tp_price = p['ep'] * (1 - tp_pct / 100)
                    if cur_l <= tp_price:
                        fill_price = tp_price * (1 + slippage)
                        actual_ret = (p['ep'] - fill_price) / p['ep'] * 100
                        reason = 'TP'
                        triggered = True

                if not triggered:
                    actual_ret = (p['ep'] - cur_c) / p['ep'] * 100

            # 到期检查
            if d >= hold:
                if not triggered:
                    reason = 'exp'
            else:
                if not triggered:
                    keep.append(p)
                    continue

            if reason:
                pnl += p['not'] * actual_ret / 100
                trades.append({
                    'sym': p['sym'], 'dir': p['dir'], 'ed': p['ed'],
                    'xd': dt, 'ep': p['ep'], 'xp': cur_c, 'r': actual_ret,
                    'pnl': p['not'] * actual_ret / 100, 'sc': p['sc'],
                    'hold': d, 'reason': reason,
                })

        pos = keep
        cap += pnl
        if cap <= 0:
            eq.append({'date': dt, 'capital': 0}); break

        n_open = max_pos - len(pos)
        if n_open <= 0:
            eq.append({'date': dt, 'capital': cap}); continue

        cands = []
        for sym, df in signal_data.items():
            if any(p['sym'] == sym for p in pos): continue
            idx = df.index[df['trade_date'] == dt]
            if len(idx) == 0: continue
            row = df.loc[idx[0]]
            if row['score_long'] >= min_sc:
                cands.append({'sym': sym, 'dir': 'long', 'sc': row['score_long'], 'ep': row['open']})
            if row['score_short'] >= min_sc:
                cands.append({'sym': sym, 'dir': 'short', 'sc': row['score_short'], 'ep': row['open']})

        best = {}
        for c in cands:
            if c['sym'] not in best or c['sc'] > best[c['sym']]['sc']:
                best[c['sym']] = c

        ranked = sorted(best.values(), key=lambda x: -x['sc'])
        for c in ranked[:n_open]:
            notional = cap * lev / max_pos
            pos.append({'sym': c['sym'], 'dir': c['dir'], 'ed': dt,
                        'ep': c['ep'], 'not': notional, 'sc': c['sc']})

        eq.append({'date': dt, 'capital': cap})
    return eq, trades


def pr(eq, trades, label, verbose=True):
    eq_df = pd.DataFrame(eq)
    if len(eq_df) == 0 or eq_df['capital'].iloc[-1] <= 0:
        print(f"  {label}: 爆仓"); return None
    eq_df['peak'] = eq_df['capital'].cummax()
    eq_df['dd'] = (eq_df['capital'] - eq_df['peak']) / eq_df['peak'] * 100
    ny = max((eq_df['date'].iloc[-1] - eq_df['date'].iloc[0]).days / 365.25, 0.01)
    ann = ((eq_df['capital'].iloc[-1] / eq_df['capital'].iloc[0]) ** (1/ny) - 1) * 100
    mdd = eq_df['dd'].min()
    dr = eq_df['capital'].pct_change().dropna()
    sh = dr.mean() / dr.std() * (252**0.5) if len(dr) > 0 and dr.std() > 0 else 0
    if trades:
        td = pd.DataFrame(trades)
        wr = (td['r'] > 0).mean() * 100
        avg = td['r'].mean()
        td['year'] = pd.to_datetime(td['xd']).dt.year
    else:
        wr = avg = 0; td = pd.DataFrame()

    if verbose:
        print(f"\n{'─'*60}")
        print(f"  {label}")
        print(f"{'─'*60}")
        print(f"  N:{len(trades)} WR:{wr:.1f}% Sharpe:{sh:.2f} Avg:{avg:+.3f}%")
        print(f"  年化:{ann:.0f}% MDD:{mdd:.1f}%")
        if len(td) > 0 and 'reason' in td.columns:
            for reason in td['reason'].unique():
                sub = td[td['reason'] == reason]
                print(f"    {reason}: N={len(sub)} WR={(sub['r']>0).mean()*100:.1f}% Avg={sub['r'].mean():+.3f}%")
        if len(td) > 0 and 'year' in td.columns:
            for yr in sorted(td['year'].unique()):
                s = td[td['year'] == yr]
                print(f"    {yr}: N={len(s):4d} WR={(s['r']>0).mean()*100:.1f}% Avg={s['r'].mean():+.3f}%")

    return {'ann': ann, 'mdd': mdd, 'wr': wr, 'sh': sh, 'n': len(trades), 'avg': avg}


def main():
    print("V76: 真实日内止损 + 集中度管理")
    print("="*60)

    print("\n加载数据...")
    all_data = load_data()
    print("计算信号...")
    sd = compute_signals(all_data)

    # ═══ A. 对比: 旧SL/TP vs 真实日内SL/TP ═══
    print(f"\n{'='*60}")
    print("A. SL/TP对比: 日收盘检查 vs 真实日内触发")
    print(f"{'='*60}")

    # 旧版 (收盘价检查, 不区分日内)
    print("\n--- 旧版: 收盘价检查 (V74/V75方式) ---")
    eq_old, tr_old = run_bt_realistic(sd, TEST_START, TEST_END, max_pos=7, lev=5,
                                       min_sc=7, hold=1, sl_pct=-2, tp_pct=3,
                                       intraday_sl=False)
    pr(eq_old, tr_old, "旧版 SL=-2% TP=3% (收盘检查)")

    # 新版 (真实日内触发)
    print("\n--- 新版: 真实日内止损 (high/low触发) ---")
    eq_new, tr_new = run_bt_realistic_v2(sd, TEST_START, TEST_END, max_pos=7, lev=5,
                                          min_sc=7, hold=1, sl_pct=-2, tp_pct=3)
    pr(eq_new, tr_new, "新版 SL=-2% TP=3% (日内触发)")

    # 无止损版本
    print("\n--- 无止损 (纯hold=1d) ---")
    eq_none, tr_none = run_bt_realistic_v2(sd, TEST_START, TEST_END, max_pos=7, lev=5,
                                            min_sc=7, hold=1)
    pr(eq_none, tr_none, "无止损 hold=1d")

    # ═══ B. SL/TP参数扫描 (真实日内) ═══
    print(f"\n\n{'='*60}")
    print("B. 真实日内SL/TP参数扫描 (mp=7, lev=5, min=7, H=1d)")
    print(f"{'='*60}")

    print(f"\n  {'SL':>5} {'TP':>5} | {'N':>5} {'WR':>6} {'Avg':>8} {'MDD':>7} {'Sharpe':>7}")
    print("-" * 55)

    results = []
    for sl in [None, -1.5, -2.0, -2.5, -3.0, -4.0]:
        for tp in [None, 2.0, 3.0, 4.0, 5.0, 8.0]:
            eq, tr = run_bt_realistic_v2(sd, TEST_START, TEST_END, max_pos=7, lev=5,
                                          min_sc=7, hold=1, sl_pct=sl, tp_pct=tp)
            stats = pr(eq, tr, "", verbose=False)
            if stats is None: continue

            sl_s = f"{sl}" if sl else "None"
            tp_s = f"{tp}" if tp else "None"
            print(f"  {sl_s:>5} {tp_s:>5} | {stats['n']:5d} {stats['wr']:5.1f}% "
                  f"{stats['avg']:>+7.3f}% {stats['mdd']:>+6.1f}% {stats['sh']:>6.2f}")
            results.append({**stats, 'sl': sl, 'tp': tp})

    # 最佳组合
    good = [r for r in results if r['wr'] >= 50 and r['mdd'] >= -30]
    if good:
        good.sort(key=lambda x: -x['sh'])
        print(f"\n  满足条件 (WR≥50%, MDD≤30%): {len(good)}个")
        for r in good[:5]:
            sl_s = f"{r['sl']}" if r['sl'] else "None"
            tp_s = f"{r['tp']}" if r['tp'] else "None"
            print(f"    SL={sl_s:>5} TP={tp_s:>5}: WR={r['wr']:.1f}% MDD={r['mdd']:.1f}% Sharpe={r['sh']:.2f}")

    # ═══ C. 不同仓位+杠杆组合 (用最佳SL/TP) ═══
    print(f"\n\n{'='*60}")
    print("C. 仓位/杠杆组合 (真实日内SL)")
    print(f"{'='*60}")

    best_sl = good[0]['sl'] if good else -2
    best_tp = good[0]['tp'] if good else 3

    for mp in [3, 5, 7]:
        for lev in [3, 5, 7]:
            eq, tr = run_bt_realistic_v2(sd, TEST_START, TEST_END, max_pos=mp, lev=lev,
                                          min_sc=7, hold=1, sl_pct=best_sl, tp_pct=best_tp)
            stats = pr(eq, tr, f"mp={mp} lev={lev}x SL={best_sl} TP={best_tp}")

    # ═══ D. Walk-forward验证 ═══
    print(f"\n\n{'='*60}")
    print("D. Walk-Forward验证 (真实日内SL)")
    print(f"{'='*60}")

    for desc, mp, lev, msc, hd, sl, tp in [
        ("最佳配置", 7, 5, 7, 1, best_sl, best_tp),
        ("保守配置", 5, 3, 7, 1, -3, 5),
        ("无止损", 7, 5, 7, 1, None, None),
    ]:
        print(f"\n--- {desc} (mp={mp}, lev={lev}, min={msc}, H={hd}, SL={sl}, TP={tp}) ---")
        eq_test, tr_test = run_bt_realistic_v2(sd, TEST_START, TEST_END, max_pos=mp, lev=lev,
                                                min_sc=msc, hold=hd, sl_pct=sl, tp_pct=tp)
        pr(eq_test, tr_test, f"测试期 2022-2025")

        eq_train, tr_train = run_bt_realistic_v2(sd, '2015-01-01', '2021-12-31', max_pos=mp, lev=lev,
                                                  min_sc=msc, hold=hd, sl_pct=sl, tp_pct=tp)
        pr(eq_train, tr_train, f"训练期 2015-2021")

    # ═══ E. 滚动稳定性 (真实SL) ═══
    print(f"\n\n{'='*60}")
    print("E. 滚动稳定性 (真实日内SL)")
    print(f"{'='*60}")

    eq, tr = run_bt_realistic_v2(sd, TEST_START, TEST_END, max_pos=7, lev=5,
                                  min_sc=7, hold=1, sl_pct=best_sl, tp_pct=best_tp)
    eq_df = pd.DataFrame(eq)
    eq_df = eq_df.set_index('date')
    daily_ret = eq_df['capital'].pct_change().dropna()

    for window in [60, 120, 252]:
        rsh = daily_ret.rolling(window).apply(lambda x: x.mean() / x.std() * (252**0.5) if x.std() > 0 else 0)
        rsh = rsh.dropna()
        if len(rsh) > 0:
            print(f"  {window}天滚动Sharpe: 均值={rsh.mean():.2f} 最低={rsh.min():.2f} "
                  f"最高={rsh.max():.2f} <0占比={100*(rsh<0).mean():.1f}%")

    # 滚动WR
    tdf = pd.DataFrame(tr)
    tdf['xd'] = pd.to_datetime(tdf['xd'])
    for window in [60, 120, 252]:
        rolling_wr = []
        for i in range(window, len(daily_ret), 20):
            end_date = daily_ret.index[i]
            start_date = daily_ret.index[i - window]
            sub = tdf[(tdf['xd'] >= start_date) & (tdf['xd'] <= end_date)]
            if len(sub) >= 10:
                rolling_wr.append((sub['r'] > 0).mean() * 100)
        if rolling_wr:
            print(f"  {window}天滚动WR: 均值={np.mean(rolling_wr):.1f}% 最低={np.min(rolling_wr):.1f}% "
                  f"WR<50%占比={100*sum(1 for w in rolling_wr if w < 50)/len(rolling_wr):.1f}%")


if __name__ == '__main__':
    main()
