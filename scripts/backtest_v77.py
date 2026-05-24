#!/usr/bin/env python3
"""
V77: ATR动态止损 + Kelly仓位 + Monte Carlo鲁棒性
目标: 进一步提升策略稳健性
1. ATR动态止损: SL = entry_price * (1 - k*ATR%), TP = entry_price * (1 + k*ATR%)
2. Kelly仓位: f = (p*b - q) / b, 其中p=WR, b=avg_w/avg_l
3. Monte Carlo: 随机打乱交易顺序1000次, 检验策略稳健性
4. 蒙特卡洛VaR: 99%置信度下最大亏损
5. 连续亏损分析
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
        df['atr_pct'] = atr_pct  # ATR as % of price
        signal_data[sym] = df
    return signal_data


def run_bt(signal_data, start, end, max_pos=7, lev=5, min_sc=7, hold=1,
           sl_pct=None, tp_pct=None, atr_sl=None, atr_tp=None,
           kelly=False, kelly_frac=0.25):
    """
    回测引擎 — V76真实日内止损 + ATR动态止损 + Kelly仓位
    atr_sl/atr_tp: SL/TP as multiples of ATR% (overrides sl_pct/tp_pct)
    kelly: use Kelly criterion for position sizing
    kelly_frac: fraction of Kelly to use (0.25 = quarter Kelly)
    """
    dates = pd.date_range(start=start, end=end, freq='B')
    cap = INITIAL_CAPITAL
    eq = []
    trades = []
    pos = []
    recent_trades = []  # for Kelly calculation

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
            atr_pct = row.get('atr_pct', np.nan)

            if np.isnan(cur_c):
                keep.append(p); continue

            d = (dt - p['ed']).days
            slippage = 0.001

            # Compute actual SL/TP based on ATR if specified
            actual_sl = sl_pct
            actual_tp = tp_pct

            if atr_sl and not np.isnan(atr_pct) and atr_pct > 0:
                actual_sl = -atr_sl * atr_pct  # SL = -k * ATR%
            if atr_tp and not np.isnan(atr_pct) and atr_pct > 0:
                actual_tp = atr_tp * atr_pct   # TP = k * ATR%

            triggered = False
            actual_ret = None
            reason = None

            if p['dir'] == 'long':
                if actual_sl:
                    stop_price = p['ep'] * (1 + actual_sl / 100)
                    if cur_l <= stop_price:
                        fill = stop_price * (1 - slippage)
                        actual_ret = (fill - p['ep']) / p['ep'] * 100
                        reason = 'SL'; triggered = True

                if not triggered and actual_tp:
                    tp_price = p['ep'] * (1 + actual_tp / 100)
                    if cur_h >= tp_price:
                        fill = tp_price * (1 - slippage)
                        actual_ret = (fill - p['ep']) / p['ep'] * 100
                        reason = 'TP'; triggered = True

                if not triggered:
                    actual_ret = (cur_c - p['ep']) / p['ep'] * 100
            else:
                if actual_sl:
                    stop_price = p['ep'] * (1 - actual_sl / 100)
                    if cur_h >= stop_price:
                        fill = stop_price * (1 + slippage)
                        actual_ret = (p['ep'] - fill) / p['ep'] * 100
                        reason = 'SL'; triggered = True

                if not triggered and actual_tp:
                    tp_price = p['ep'] * (1 - actual_tp / 100)
                    if cur_l <= tp_price:
                        fill = tp_price * (1 + slippage)
                        actual_ret = (p['ep'] - fill) / p['ep'] * 100
                        reason = 'TP'; triggered = True

                if not triggered:
                    actual_ret = (p['ep'] - cur_c) / p['ep'] * 100

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
                    'hold': d, 'reason': reason, 'atr_pct': atr_pct,
                })
                recent_trades.append(actual_ret)

        pos = keep
        cap += pnl
        if cap <= 0:
            eq.append({'date': dt, 'capital': 0}); break

        # Kelly position sizing
        cur_lev = lev
        if kelly and len(recent_trades) >= 50:
            recent = recent_trades[-200:]
            wins = [r for r in recent if r > 0]
            losses = [r for r in recent if r <= 0]
            if len(wins) > 0 and len(losses) > 0:
                p_win = len(wins) / len(recent)
                avg_w = np.mean(wins)
                avg_l = abs(np.mean(losses))
                if avg_l > 0:
                    b = avg_w / avg_l
                    kelly_f = (p_win * b - (1 - p_win)) / b
                    kelly_f = max(0, min(kelly_f, 1))
                    cur_lev = lev * kelly_f * kelly_frac / 0.5  # scale
                    cur_lev = max(lev * 0.3, min(cur_lev, lev * 2))

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
        for c_ in cands:
            if c_['sym'] not in best or c_['sc'] > best[c_['sym']]['sc']:
                best[c_['sym']] = c_

        ranked = sorted(best.values(), key=lambda x: -x['sc'])
        for c_ in ranked[:n_open]:
            notional = cap * cur_lev / max_pos
            pos.append({'sym': c_['sym'], 'dir': c_['dir'], 'ed': dt,
                        'ep': c_['ep'], 'not': notional, 'sc': c_['sc']})

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
            for reason in ['SL', 'TP', 'exp']:
                sub = td[td['reason'] == reason]
                if len(sub) > 0:
                    print(f"    {reason}: N={len(sub)} WR={(sub['r']>0).mean()*100:.1f}% Avg={sub['r'].mean():+.3f}%")
        if len(td) > 0 and 'year' in td.columns:
            for yr in sorted(td['year'].unique()):
                s = td[td['year'] == yr]
                print(f"    {yr}: N={len(s):4d} WR={(s['r']>0).mean()*100:.1f}% Avg={s['r'].mean():+.3f}%")

    return {'ann': ann, 'mdd': mdd, 'wr': wr, 'sh': sh, 'n': len(trades), 'avg': avg}


def monte_carlo_test(trades, n_sim=1000, initial_capital=500_000, lev=5, max_pos=7):
    """Monte Carlo: 随机打乱交易顺序1000次"""
    tdf = pd.DataFrame(trades)
    returns = tdf['r'].values / 100  # convert to decimal

    mc_mdd = []
    mc_final = []
    mc_sharpe = []

    for _ in range(n_sim):
        np.random.shuffle(returns)
        # Simulate equity curve with same notional sizing
        n_trades = len(returns)
        trades_per_day = max_pos
        n_days = n_trades // trades_per_day + 1

        daily_pnl = np.zeros(n_days)
        for i, r in enumerate(returns):
            day = i // trades_per_day
            notional = initial_capital * lev / max_pos
            daily_pnl[day] += notional * r

        cap = initial_capital
        peak = cap
        mdd = 0
        daily_caps = []
        for dp in daily_pnl:
            cap += dp
            if cap <= 0:
                cap = 0; break
            daily_caps.append(cap)
            peak = max(peak, cap)
            dd = (cap - peak) / peak * 100
            mdd = min(mdd, dd)

        mc_mdd.append(mdd)
        mc_final.append(cap)

        if len(daily_caps) > 1:
            dc = np.array(daily_caps)
            dr = np.diff(dc) / dc[:-1]
            sh = dr.mean() / dr.std() * (252**0.5) if dr.std() > 0 else 0
            mc_sharpe.append(sh)

    return {
        'mdd_p5': np.percentile(mc_mdd, 5),
        'mdd_mean': np.mean(mc_mdd),
        'mdd_worst': np.min(mc_mdd),
        'final_p5': np.percentile(mc_final, 5),
        'final_mean': np.mean(mc_final),
        'sharpe_p5': np.percentile(mc_sharpe, 5),
        'sharpe_mean': np.mean(mc_sharpe),
    }


def analyze_consecutive(trades):
    """连续亏损分析"""
    tdf = pd.DataFrame(trades)
    returns = tdf['r'].values

    # 连续亏损
    max_consec_loss = 0
    cur_consec = 0
    max_loss_streak = []
    cur_streak = []

    for r in returns:
        if r <= 0:
            cur_consec += 1
            cur_streak.append(r)
            if cur_consec > max_consec_loss:
                max_consec_loss = cur_consec
                max_loss_streak = cur_streak.copy()
        else:
            cur_consec = 0
            cur_streak = []

    # 连续亏损累计
    cum_loss = 0
    max_cum_loss = 0
    for r in returns:
        if r <= 0:
            cum_loss += r
        else:
            cum_loss = 0
        max_cum_loss = min(max_cum_loss, cum_loss)

    # 最大回撤 (per-trade)
    peak = 0
    cur = 0
    max_dd = 0
    for r in returns:
        cur += r
        peak = max(peak, cur)
        dd = cur - peak
        max_dd = min(max_dd, dd)

    return {
        'max_consec_losses': max_consec_loss,
        'max_loss_streak': max_loss_streak,
        'max_cum_loss': max_cum_loss,
        'per_trade_mdd': max_dd,
    }


def main():
    print("V77: ATR动态止损 + Kelly仓位 + Monte Carlo")
    print("="*60)

    print("\n加载数据...")
    all_data = load_data()
    print("计算信号...")
    sd = compute_signals(all_data)

    # ═══ A. ATR动态止损 vs 固定止损 ═══
    print(f"\n{'='*60}")
    print("A. ATR动态止损 vs 固定止损")
    print(f"{'='*60}")

    print(f"\n  {'类型':>10} {'SL':>8} {'TP':>8} | {'N':>5} {'WR':>6} {'Avg':>8} {'MDD':>7} {'Sharpe':>7}")
    print("-" * 65)

    configs = [
        # (desc, sl_pct, tp_pct, atr_sl, atr_tp)
        ("固定SL", -1.5, 4.0, None, None),
        ("固定SL", -2.0, 3.0, None, None),
        ("固定SL", -2.0, 5.0, None, None),
        ("ATR*0.5", None, None, 0.5, 1.5),
        ("ATR*1.0", None, None, 1.0, 2.0),
        ("ATR*1.0", None, None, 1.0, 3.0),
        ("ATR*1.5", None, None, 1.5, 3.0),
        ("ATR*1.5", None, None, 1.5, 4.0),
        ("ATR*2.0", None, None, 2.0, 4.0),
        ("ATR*2.0", None, None, 2.0, 6.0),
        ("无止损", None, None, None, None),
    ]

    results = []
    for desc, sl, tp, asl, atp in configs:
        eq, tr = run_bt(sd, TEST_START, TEST_END, max_pos=7, lev=5, min_sc=7, hold=1,
                         sl_pct=sl, tp_pct=tp, atr_sl=asl, atr_tp=atp)
        stats = pr(eq, tr, "", verbose=False)
        if stats is None: continue

        sl_s = f"{sl}%" if sl else (f"{asl}*ATR" if asl else "None")
        tp_s = f"{tp}%" if tp else (f"{atp}*ATR" if atp else "None")
        print(f"  {desc:>10} {sl_s:>8} {tp_s:>8} | {stats['n']:5d} {stats['wr']:5.1f}% "
              f"{stats['avg']:>+7.3f}% {stats['mdd']:>+6.1f}% {stats['sh']:>6.2f}")
        results.append({**stats, 'desc': desc})

    # ═══ B. Kelly仓位 ═══
    print(f"\n\n{'='*60}")
    print("B. Kelly仓位管理")
    print(f"{'='*60}")

    for kf in [0.1, 0.25, 0.5, 1.0]:
        eq, tr = run_bt(sd, TEST_START, TEST_END, max_pos=7, lev=5, min_sc=7, hold=1,
                         sl_pct=-1.5, tp_pct=4.0, kelly=True, kelly_frac=kf)
        pr(eq, tr, f"Kelly frac={kf} (SL=-1.5 TP=4)")

    # ═══ C. Monte Carlo鲁棒性 ═══
    print(f"\n\n{'='*60}")
    print("C. Monte Carlo鲁棒性测试 (1000次模拟)")
    print(f"{'='*60}")

    # 先获取最佳配置的交易
    eq_best, tr_best = run_bt(sd, TEST_START, TEST_END, max_pos=7, lev=5, min_sc=7, hold=1,
                               sl_pct=-1.5, tp_pct=4.0)

    print("\n  运行Monte Carlo模拟...")
    mc = monte_carlo_test(tr_best, n_sim=1000)
    print(f"\n  Monte Carlo结果 (1000次):")
    print(f"    MDD: 5%分位={mc['mdd_p5']:.1f}% 均值={mc['mdd_mean']:.1f}% 最差={mc['mdd_worst']:.1f}%")
    print(f"    Sharpe: 5%分位={mc['sharpe_p5']:.2f} 均值={mc['sharpe_mean']:.2f}")
    print(f"    最终资金: 5%分位={mc['final_p5']:,.0f} 均值={mc['final_mean']:,.0f}")

    # ═══ D. 连续亏损分析 ═══
    print(f"\n\n{'='*60}")
    print("D. 连续亏损分析")
    print(f"{'='*60}")

    ca = analyze_consecutive(tr_best)
    print(f"  最大连续亏损次数: {ca['max_consec_losses']}")
    print(f"  最大累计亏损: {ca['max_cum_loss']:+.2f}%")
    print(f"  逐笔最大回撤: {ca['per_trade_mdd']:+.2f}%")

    if ca['max_loss_streak']:
        print(f"  最差连续亏损序列: {', '.join([f'{r:+.2f}%' for r in ca['max_loss_streak'][:10]])}")

    # ═══ E. 逐月/逐季稳定性 (最终配置) ═══
    print(f"\n\n{'='*60}")
    print("E. 最终推荐配置详析")
    print(f"{'='*60}")

    # 推荐配置: mp=7, lev=5, min=7, H=1d, SL=-1.5%, TP=4.0%
    print("\n  推荐配置: mp=7, lev=5, min=7, H=1d, SL=-1.5%, TP=4.0%")
    eq_final, tr_final = run_bt(sd, '2015-01-01', '2025-12-31', max_pos=7, lev=5, min_sc=7, hold=1,
                                 sl_pct=-1.5, tp_pct=4.0)
    pr(eq_final, tr_final, "全样本 2015-2025")

    # 分段分析
    print(f"\n  分段稳定性:")
    for label, start, end in [
        ("早期(2015-2017)", "2015-01-01", "2017-12-31"),
        ("中期(2018-2020)", "2018-01-01", "2020-12-31"),
        ("近期(2021-2022)", "2021-01-01", "2022-12-31"),
        ("最新(2023-2025)", "2023-01-01", "2025-12-31"),
    ]:
        eq_s, tr_s = run_bt(sd, start, end, max_pos=7, lev=5, min_sc=7, hold=1,
                             sl_pct=-1.5, tp_pct=4.0)
        stats = pr(eq_s, tr_s, "", verbose=False)
        if stats:
            print(f"    {label}: N={stats['n']:5d} WR={stats['wr']:.1f}% "
                  f"Avg={stats['avg']:+.3f}% MDD={stats['mdd']:.1f}% Sharpe={stats['sh']:.2f}")

    # ═══ F. 风险指标汇总 ═══
    print(f"\n\n{'='*60}")
    print("F. 风险指标汇总")
    print(f"{'='*60}")

    eq_test, tr_test = run_bt(sd, TEST_START, TEST_END, max_pos=7, lev=5, min_sc=7, hold=1,
                               sl_pct=-1.5, tp_pct=4.0)
    eq_df = pd.DataFrame(eq_test)
    daily_ret = eq_df['capital'].pct_change().dropna()

    tdf = pd.DataFrame(tr_test)

    # 基本指标
    wr = (tdf['r'] > 0).mean() * 100
    pf_wins = tdf[tdf['r'] > 0]['r'].sum()
    pf_losses = abs(tdf[tdf['r'] <= 0]['r'].sum())
    pf = pf_wins / pf_losses if pf_losses > 0 else 999

    avg_w = tdf[tdf['r'] > 0]['r'].mean()
    avg_l = abs(tdf[tdf['r'] <= 0]['r'].mean())

    # Calmar
    ny = max((eq_df['date'].iloc[-1] - eq_df['date'].iloc[0]).days / 365.25, 0.01)
    ann = ((eq_df['capital'].iloc[-1] / eq_df['capital'].iloc[0]) ** (1/ny) - 1) * 100
    mdd = ((eq_df['capital'] - eq_df['capital'].cummax()) / eq_df['capital'].cummax() * 100).min()
    calmar = abs(ann / mdd) if mdd != 0 else 999

    # Sortino (downside deviation)
    neg_ret = daily_ret[daily_ret < 0]
    sortino = daily_ret.mean() / neg_ret.std() * (252**0.5) if len(neg_ret) > 0 and neg_ret.std() > 0 else 0

    print(f"\n  交易统计:")
    print(f"    总交易: {len(tr_test)}")
    print(f"    胜率: {wr:.1f}%")
    print(f"    盈亏比: {pf:.2f}")
    print(f"    平均盈利: {avg_w:+.3f}%")
    print(f"    平均亏损: {avg_l:+.3f}%")
    print(f"    盈亏比率: {avg_w/avg_l:.2f}:1" if avg_l > 0 else "")

    print(f"\n  风险指标:")
    print(f"    Sharpe: {pr(eq_test, tr_test, '', verbose=False)['sh']:.2f}")
    print(f"    Sortino: {sortino:.2f}")
    print(f"    Calmar: {calmar:.1f}")
    print(f"    MDD: {mdd:.1f}%")
    print(f"    MC-MDD(5%分位): {mc['mdd_p5']:.1f}%")

    # 日收益分布
    print(f"\n  日收益分布:")
    pct_pos = (daily_ret > 0).mean() * 100
    print(f"    正收益日: {pct_pos:.1f}%")
    print(f"    日均值: {daily_ret.mean()*100:.4f}%")
    print(f"    日标准差: {daily_ret.std()*100:.4f}%")
    print(f"    日最大盈利: {daily_ret.max()*100:.3f}%")
    print(f"    日最大亏损: {daily_ret.min()*100:.3f}%")

    # VaR
    var_95 = np.percentile(daily_ret, 5) * 100
    var_99 = np.percentile(daily_ret, 1) * 100
    print(f"    VaR(95%): {var_95:.3f}%")
    print(f"    VaR(99%): {var_99:.3f}%")


if __name__ == '__main__':
    main()
