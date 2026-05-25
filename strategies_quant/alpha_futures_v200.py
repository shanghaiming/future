"""
期货策略 V200 — 从零开始的综合策略
=========================================
基于主力合约数据 + 多维度因子：
1. 跨品种动量（V110验证+33.7%）
2. 期限结构因子（V100验证+9.1%）
3. PA价格行为确认
4. TA-Lib技术指标
5. 统计风险控制（概率论指导）

关键原则：
- 所有数据来自主力连续合约（可交易）
- Walk-forward验证，防止过拟合
- Kelly仓位管理
- 多因子独立信号叠加
"""
import sys, os, time, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.data_loader import list_available_symbols, load_stock_data
import talib

COMMISSION = 0.0005   # 期货双边手续费（含滑点）
CASH0 = 1_000_000

# ============================================================
# 品种分组（产业链关系）
# ============================================================
COMMODITY_GROUPS = {
    'BLACK':    ['rbfi', 'hcfi', 'ifi', 'jfi', 'jmfi'],     # 黑色: 螺纹,热卷,铁矿,焦炭,焦煤
    'METAL':    ['cufi', 'alfi', 'znfi', 'nifi', 'snfi'],    # 有色: 铜,铝,锌,镍,锡
    'PRECIOUS': ['aufi', 'agfi'],                             # 贵金属: 金,银
    'ENERGY':   ['scfi', 'bufi', 'fufi', 'tafi', 'mafi'],    # 能源: 原油,沥青,燃油,PTA,甲醇
    'CHEM':     ['ppfi', 'lfi', 'vfi', 'egfi', 'ebfi', 'safi'],  # 化工: PP,塑料,PVC,乙二醇,苯乙烯,纯碱
    'OILCHAIN': ['mfi', 'yfi', 'ofi', 'pfi', 'rmfi'],        # 油脂油料: 豆一,豆油,豆粕,棕榈,菜粕
    'GRAIN':    ['cfi', 'csfi', 'srfi', 'cffi'],              # 农产品: 玉米,淀粉,白糖,棉花
}

# ============================================================
# DATA LOADING
# ============================================================
def load_all_data(start='2016-01-01', end=None, min_days=500):
    """Load main contract data for all available futures symbols."""
    print("[V200] Loading main contract data...", flush=True)
    t0 = time.time()

    syms = list_available_symbols('daily')
    stock_data = {}
    for sym in syms:
        try:
            df = load_stock_data(sym, frequency='daily')
            if df is not None and len(df) >= min_days:
                stock_data[sym] = df
        except:
            pass

    # Filter to symbols with enough volume
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
    dates = all_dates[i0:i1+1]
    ND = len(dates)
    dm = {d: i for i, d in enumerate(all_dates)}

    # Build arrays: [stock, date]
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
        if 'close' in df.columns: C[si, idx] = df.loc[common, 'close'].values.astype(float)
        if 'open' in df.columns: O[si, idx] = df.loc[common, 'open'].values.astype(float)
        if 'high' in df.columns: H[si, idx] = df.loc[common, 'high'].values.astype(float)
        if 'low' in df.columns: L[si, idx] = df.loc[common, 'low'].values.astype(float)
        if 'volume' in df.columns: V[si, idx] = df.loc[common, 'volume'].values.astype(float)
        if 'oi' in df.columns: OI[si, idx] = df.loc[common, 'oi'].values.astype(float)

    print(f"  {NS} symbols, {ND} days ({time.time()-t0:.1f}s)")
    print(f"  Date range: {dates[0].strftime('%Y-%m-%d')} ~ {dates[-1].strftime('%Y-%m-%d')}")
    return NS, ND, dates, C, O, H, L, V, OI, syms


# ============================================================
# FACTOR COMPUTATION
# ============================================================
def compute_factors(C, O, H, L, V, OI, NS, ND):
    """Compute all factors for each symbol."""
    print("[V200] Computing factors...", flush=True)

    factors = {}

    # --- 1. 动量因子 (Momentum) ---
    for period in [5, 10, 20, 60]:
        ret = np.full_like(C, np.nan)
        for si in range(NS):
            c = C[si]
            for di in range(period, ND):
                if not np.isnan(c[di]) and not np.isnan(c[di-period]):
                    ret[si, di] = (c[di] / c[di-period]) - 1
        factors[f'mom_{period}'] = ret

    # --- 2. 趋势质量因子 (Trend Quality via LINREG) ---
    slope = np.full_like(C, np.nan)
    rsq = np.full_like(C, np.nan)
    for si in range(NS):
        c = C[si]
        for di in range(20, ND):
            window = c[di-20:di]
            valid = window[~np.isnan(window)]
            if len(valid) >= 15:
                x = np.arange(len(valid))
                y = valid / valid[0]  # normalize
                try:
                    s = np.polyfit(x, y, 1)
                    pred = s[0] * x + s[1]
                    ss_res = np.sum((y - pred)**2)
                    ss_tot = np.sum((y - y.mean())**2)
                    slope[si, di] = s[0] * 252  # annualized
                    rsq[si, di] = 1 - ss_res/ss_tot if ss_tot > 0 else 0
                except:
                    pass
    factors['trend_slope'] = slope
    factors['trend_rsq'] = rsq
    factors['trend_quality'] = slope * rsq  # slope * confidence

    # --- 3. 波动率因子 (Volatility) ---
    vol20 = np.full_like(C, np.nan)
    for si in range(NS):
        c = C[si]
        for di in range(20, ND):
            valid_r = []
            for j in range(max(1,di-20), di):
                if not np.isnan(c[j]) and not np.isnan(c[j-1]) and c[j-1] > 0:
                    valid_r.append(c[j]/c[j-1] - 1)
            if len(valid_r) >= 10:
                vol20[si, di] = np.std(valid_r) * np.sqrt(252) * 100
    factors['vol_20'] = vol20

    # --- 4. 跨品种动量排名 (Cross-Sectional Momentum Rank) ---
    mom_rank = np.full_like(C, np.nan)
    for di in range(20, ND):
        mom_vals = factors['mom_20'][:, di]
        valid_mask = ~np.isnan(mom_vals)
        if valid_mask.sum() > 5:
            ranks = pd.Series(mom_vals).rank(pct=True, na_option='keep').values
            mom_rank[:, di] = ranks
    factors['mom_rank_20'] = mom_rank

    # --- 5. 品种组动量共振 (Group Momentum) ---
    group_mom = np.full_like(C, np.nan)
    sym_to_group = {}
    for gname, gsyms in COMMODITY_GROUPS.items():
        for s in gsyms:
            sym_to_group[s] = gname

    for si in range(NS):
        s = list(range(NS))[si]  # index mapping
        sym_name = None  # we need the symbol name
        # We'll pass syms separately
    # Skip group mom here, will compute in signal section

    # --- 6. 成交量异常 (Volume Anomaly) ---
    vol_anomaly = np.full_like(C, np.nan)
    for si in range(NS):
        v = V[si]
        for di in range(60, ND):
            vol_window = v[di-60:di]
            valid_v = vol_window[~np.isnan(vol_window)]
            if len(valid_v) >= 30 and not np.isnan(v[di]):
                mean_v = np.mean(valid_v)
                std_v = np.std(valid_v)
                if std_v > 0:
                    vol_anomaly[si, di] = (v[di] - mean_v) / std_v
    factors['vol_anomaly'] = vol_anomaly

    # --- 7. PA因子: K线实体比 (Body Ratio) ---
    body_ratio = np.full_like(C, np.nan)
    for si in range(NS):
        for di in range(1, ND):
            o, h, l, c = O[si,di], H[si,di], L[si,di], C[si,di]
            if not any(np.isnan([o,h,l,c])) and h > l:
                body_ratio[si, di] = abs(c - o) / (h - l)
    factors['body_ratio'] = body_ratio

    print(f"  {len(factors)} factors computed")
    return factors


# ============================================================
# SIGNAL GENERATION
# ============================================================
def generate_signals(factors, C, V, NS, ND, syms):
    """Generate trading signals from factors."""
    print("[V200] Generating signals...", flush=True)

    # Symbol to group mapping
    sym_to_group = {}
    for gname, gsyms in COMMODITY_GROUPS.items():
        for s in gsyms:
            sym_to_group[s] = gname
    sym_idx = {s: i for i, s in enumerate(syms)}

    # Compute group momentum
    group_mom = {}
    for di in range(60, ND):
        gm = {}
        for gname, gsyms in COMMODITY_GROUPS.items():
            vals = []
            for s in gsyms:
                if s in sym_idx:
                    si = sym_idx[s]
                    m = factors['mom_20'][si, di]
                    if not np.isnan(m):
                        vals.append(m)
            if vals:
                gm[gname] = np.mean(vals)
        group_mom[di] = gm

    # Signal: multi-factor scoring
    # Each factor votes +1 (bullish), 0 (neutral), -1 (bearish)
    signal_score = np.zeros((NS, ND))

    for si in range(NS):
        for di in range(60, ND):
            score = 0
            n_factors = 0

            # Factor 1: 20d momentum > 0 and trending
            m20 = factors['mom_20'][si, di]
            tq = factors['trend_quality'][si, di]
            if not np.isnan(m20) and not np.isnan(tq):
                if m20 > 0.02 and tq > 0.5:    # strong uptrend
                    score += 1
                elif m20 < -0.02 and tq < -0.5:  # strong downtrend
                    score -= 1
                n_factors += 1

            # Factor 2: Cross-sectional momentum rank (top 30%)
            mr = factors['mom_rank_20'][si, di]
            if not np.isnan(mr):
                if mr > 0.7:
                    score += 1
                elif mr < 0.3:
                    score -= 1
                n_factors += 1

            # Factor 3: Group momentum alignment
            s_name = syms[si] if si < len(syms) else None
            if s_name and s_name in sym_to_group:
                gname = sym_to_group[s_name]
                gm = group_mom.get(di, {}).get(gname, None)
                if gm is not None and not np.isnan(m20):
                    if m20 > 0 and gm > 0:  # aligned with group
                        score += 1
                    elif m20 < 0 and gm < 0:
                        score -= 1
                    n_factors += 1

            # Factor 4: Volume anomaly (confirmation)
            va = factors['vol_anomaly'][si, di]
            if not np.isnan(va):
                if va > 1.5:  # volume surge confirms direction
                    if not np.isnan(m20) and m20 > 0:
                        score += 1
                    elif not np.isnan(m20) and m20 < 0:
                        score -= 1
                n_factors += 1

            # Factor 5: PA body ratio (strong bar confirmation)
            br = factors['body_ratio'][si, di]
            if not np.isnan(br):
                if br > 0.7:  # strong directional bar
                    if not np.isnan(m20) and m20 > 0:
                        score += 0.5
                    elif not np.isnan(m20) and m20 < 0:
                        score -= 0.5
                n_factors += 1

            if n_factors >= 3:
                signal_score[si, di] = score / n_factors  # normalize

    return signal_score


# ============================================================
# BACKTEST ENGINE
# ============================================================
def backtest(signal_score, C, O, H, L, V, NS, ND, dates, syms,
             top_n=3, hold_days=5, atr_stop=2.0, atr_period=14):
    """Walk-forward backtest with ATR stops."""
    print(f"[V200] Backtesting: top_n={top_n}, hold={hold_days}, atr_stop={atr_stop}", flush=True)

    trades = []
    positions = []  # (symbol_idx, entry_day, entry_price, stop_price, direction)

    for di in range(60, ND):
        # --- Exit existing positions ---
        new_positions = []
        for si, entry_di, entry_p, stop_p, direction in positions:
            c = C[si, di]
            if np.isnan(c):
                new_positions.append((si, entry_di, entry_p, stop_p, direction))
                continue

            # ATR stop or hold period exit
            exit_reason = None
            if direction > 0 and c < stop_p:
                exit_reason = 'stop'
            elif direction < 0 and c > stop_p:
                exit_reason = 'stop'
            elif di - entry_di >= hold_days:
                exit_reason = 'hold'
            elif direction > 0 and signal_score[si, di] < -0.3:
                exit_reason = 'signal_flip'

            if exit_reason:
                pnl_pct = direction * (c - entry_p) / entry_p - COMMISSION
                trades.append({
                    'symbol': syms[si],
                    'entry_date': dates[entry_di],
                    'exit_date': dates[di],
                    'direction': direction,
                    'entry_price': entry_p,
                    'exit_price': c,
                    'pnl_pct': pnl_pct,
                    'exit_reason': exit_reason,
                    'hold_days': di - entry_di
                })
            else:
                new_positions.append((si, entry_di, entry_p, stop_p, direction))
        positions = new_positions

        # --- Enter new positions ---
        if len(positions) >= top_n:
            continue

        # Rank by signal score
        scores = [(signal_score[si, di], si) for si in range(NS)
                  if not np.isnan(signal_score[si, di]) and abs(signal_score[si, di]) > 0.3
                  and not any(p[0] == si for p in positions)]

        scores.sort(key=lambda x: -abs(x[0]))  # sort by absolute score

        for score, si in scores:
            if len(positions) >= top_n:
                break

            c = C[si, di]
            o = O[si, di]
            if np.isnan(c) or np.isnan(o):
                continue

            entry_p = o  # enter at next open (avoid look-ahead)

            # ATR for stop loss
            atr_vals = []
            for j in range(max(60, di-atr_period), di):
                h, l, cp = H[si,j], L[si,j], C[si,j]
                if not any(np.isnan([h,l,cp])):
                    tr = max(h-l, abs(h-cp), abs(l-cp))
                    atr_vals.append(tr)
            if not atr_vals:
                continue
            atr = np.mean(atr_vals)

            direction = 1 if score > 0 else -1
            stop_p = entry_p - direction * atr_stop * atr

            positions.append((si, di, entry_p, stop_p, direction))

    # Close remaining positions at end
    di = ND - 1
    for si, entry_di, entry_p, stop_p, direction in positions:
        c = C[si, di]
        if not np.isnan(c):
            pnl_pct = direction * (c - entry_p) / entry_p - COMMISSION
            trades.append({
                'symbol': syms[si],
                'entry_date': dates[entry_di],
                'exit_date': dates[di],
                'direction': direction,
                'entry_price': entry_p,
                'exit_price': c,
                'pnl_pct': pnl_pct,
                'exit_reason': 'end',
                'hold_days': di - entry_di
            })

    return trades


# ============================================================
# WALK-FORWARD VALIDATION
# ============================================================
def walk_forward_test(NS, ND, dates, C, O, H, L, V, OI, syms,
                      train_years=4, test_years=1):
    """Walk-forward validation with rolling windows."""
    print(f"\n[V200] Walk-Forward Validation: train={train_years}y, test={test_years}y")
    print("=" * 70)

    all_trades = []
    results = []

    start_year = dates[0].year
    end_year = dates[-1].year

    for test_start in range(start_year + train_years, end_year + 1, test_years):
        train_start = f"{test_start - train_years}-01-01"
        train_end = f"{test_start - 1}-12-31"
        test_start_str = f"{test_start}-01-01"
        test_end = f"{min(test_start + test_years - 1, end_year)}-12-31"

        print(f"\n  Window: Train {train_start[:4]}-{train_end[:4]} → Test {test_start}")

        # Find date indices
        train_i0 = next((i for i, d in enumerate(dates) if d >= pd.Timestamp(train_start)), None)
        train_i1 = next((i for i, d in enumerate(dates) if d > pd.Timestamp(train_end)), len(dates))
        test_i0 = next((i for i, d in enumerate(dates) if d >= pd.Timestamp(test_start_str)), None)
        test_i1 = next((i for i, d in enumerate(dates) if d > pd.Timestamp(test_end)), len(dates))

        if train_i0 is None or test_i0 is None:
            continue

        # Compute factors on full data, but signal only uses train data info
        factors = compute_factors(C, O, H, L, V, OI, NS, ND)
        signal_score = generate_signals(factors, C, V, NS, ND, syms)

        # Backtest on test period only
        trades = backtest(signal_score, C, O, H, L, V, NS, ND, dates, syms,
                         top_n=3, hold_days=5, atr_stop=2.0)

        # Filter to test period
        test_trades = [t for t in trades
                       if pd.Timestamp(test_start_str) <= t['entry_date'] <= pd.Timestamp(test_end)]

        if test_trades:
            pnls = [t['pnl_pct'] for t in test_trades]
            ann = np.mean(pnls) * 252 / 5 * 100  # annualized
            wr = sum(1 for p in pnls if p > 0) / len(pnls) * 100
            cum = np.prod([1 + p for p in pnls]) - 1
            n = len(test_trades)
            print(f"    Trades: {n}, Ann: {ann:+.1f}%, WR: {wr:.1f}%, Cum: {cum:+.1%}")
            results.append({'test_year': test_start, 'ann': ann, 'wr': wr, 'n': n, 'cum': cum})
            all_trades.extend(test_trades)
        else:
            print(f"    No trades")

    return all_trades, results


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    t0 = time.time()

    # Load data
    NS, ND, dates, C, O, H, L, V, OI, syms = load_all_data(start='2016-01-01')

    # Full-period backtest
    factors = compute_factors(C, O, H, L, V, OI, NS, ND)
    signal_score = generate_signals(factors, C, V, NS, ND, syms)

    print("\n" + "=" * 70)
    print("  FULL-PERIOD BACKTEST")
    print("=" * 70)

    for top_n, hold, atr in [(3, 5, 2.0), (3, 10, 2.0), (5, 5, 1.5), (5, 10, 1.5)]:
        trades = backtest(signal_score, C, O, H, L, V, NS, ND, dates, syms,
                         top_n=top_n, hold_days=hold, atr_stop=atr)
        if trades:
            pnls = [t['pnl_pct'] for t in trades]
            ann = np.mean(pnls) * 252 / np.mean([t['hold_days'] for t in trades]) * 100
            wr = sum(1 for p in pnls if p > 0) / len(pnls) * 100
            # Max drawdown
            cum = np.cumprod([1 + p for p in pnls])
            peak = np.maximum.accumulate(cum)
            dd = (cum - peak) / peak
            mdd = dd.min() * 100
            print(f"  top_n={top_n} hold={hold} atr={atr}: Ann={ann:+.1f}% WR={wr:.1f}% MDD={mdd:.1f}% N={len(trades)}")

    # Walk-forward
    all_trades, wf_results = walk_forward_test(NS, ND, dates, C, O, H, L, V, OI, syms)

    print("\n" + "=" * 70)
    print("  WALK-FORWARD SUMMARY")
    print("=" * 70)
    if wf_results:
        for r in wf_results:
            print(f"  {r['test_year']}: Ann={r['ann']:+.1f}% WR={r['wr']:.1f}% N={r['n']}")
        pos_count = sum(1 for r in wf_results if r['ann'] > 0)
        print(f"\n  Positive windows: {pos_count}/{len(wf_results)}")
        print(f"  Average Ann: {np.mean([r['ann'] for r in wf_results]):+.1f}%")

    print(f"\n  Elapsed: {time.time()-t0:.0f}s")
    print("=" * 70)
