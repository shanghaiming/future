"""
期货策略 V201 — V200 + 期限结构因子增强
=========================================
V200基础: WF 5/7正, 平均+9.3%年化
增强: 加入期限结构因子(27K条数据), 改进MDD控制

核心改进:
1. 期限结构极端信号(2σ偏离) → 回归交易
2. 基差动量 → 趋势确认/过滤
3. 跨品种相对价值 → 配对信号
4. 改进止损: 波动率自适应ATR
"""
import sys, os, time, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.data_loader import list_available_symbols, load_stock_data

COMMISSION = 0.0005
CASH0 = 1_000_000

COMMODITY_GROUPS = {
    'BLACK':    ['rbfi', 'hcfi', 'ifi', 'jfi', 'jmfi'],
    'METAL':    ['cufi', 'alfi', 'znfi', 'nifi', 'snfi'],
    'PRECIOUS': ['aufi', 'agfi'],
    'ENERGY':   ['scfi', 'bufi', 'fufi', 'tafi', 'mafi'],
    'CHEM':     ['ppfi', 'lfi', 'vfi', 'egfi', 'ebfi', 'safi'],
    'OILCHAIN': ['mfi', 'yfi', 'ofi', 'pfi', 'rmfi'],
    'GRAIN':    ['cfi', 'csfi', 'srfi', 'cffi'],
}

# ============================================================
# LOAD TERM STRUCTURE FACTORS
# ============================================================
def load_ts_factors():
    """Load pre-computed term structure factors."""
    ts_path = os.path.join(os.path.dirname(__file__), 'term_structure_factors.csv')
    if not os.path.exists(ts_path):
        print("[V201] WARNING: term_structure_factors.csv not found, skipping TS factors")
        return None
    df = pd.read_csv(ts_path)
    df['date'] = pd.to_datetime(df['date'])
    return df


# ============================================================
# DATA LOADING
# ============================================================
def load_all_data(start='2016-01-01', end=None, min_days=500):
    print("[V201] Loading main contract data...", flush=True)
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

    C = np.full((NS, len(all_dates)), np.nan)
    O = np.full((NS, len(all_dates)), np.nan)
    H = np.full((NS, len(all_dates)), np.nan)
    L = np.full((NS, len(all_dates)), np.nan)
    V = np.full((NS, len(all_dates)), np.nan)

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

    print(f"  {NS} symbols, {ND} days ({time.time()-t0:.1f}s)")
    return NS, ND, dates, C, O, H, L, V, syms


# ============================================================
# FACTOR COMPUTATION (V200 factors + TS factors)
# ============================================================
def compute_factors(C, O, H, L, V, NS, ND, dates, syms):
    print("[V201] Computing factors...", flush=True)
    factors = {}

    # --- 动量因子 ---
    for period in [5, 10, 20, 60]:
        ret = np.full_like(C, np.nan)
        for si in range(NS):
            for di in range(period, ND):
                if not np.isnan(C[si,di]) and not np.isnan(C[si,di-period]) and C[si,di-period] > 0:
                    ret[si, di] = C[si,di] / C[si,di-period] - 1
        factors[f'mom_{period}'] = ret

    # --- 趋势质量 (vectorized per symbol) ---
    slope = np.full_like(C, np.nan)
    rsq = np.full_like(C, np.nan)
    for si in range(NS):
        c = C[si]
        for di in range(20, ND):
            window = c[di-20:di]
            valid = window[~np.isnan(window)]
            if len(valid) >= 15:
                x = np.arange(len(valid))
                y = valid / valid[0]
                try:
                    s = np.polyfit(x, y, 1)
                    pred = s[0]*x + s[1]
                    ss_res = np.sum((y-pred)**2)
                    ss_tot = np.sum((y-y.mean())**2)
                    slope[si,di] = s[0] * 252
                    rsq[si,di] = 1 - ss_res/ss_tot if ss_tot > 0 else 0
                except: pass
    factors['trend_quality'] = slope * rsq

    # --- Cross-sectional rank ---
    mom_rank = np.full_like(C, np.nan)
    for di in range(20, ND):
        mv = factors['mom_20'][:, di]
        valid = ~np.isnan(mv)
        if valid.sum() > 5:
            mom_rank[:, di] = pd.Series(mv).rank(pct=True, na_option='keep').values
    factors['mom_rank_20'] = mom_rank

    # --- Volume anomaly ---
    vol_anomaly = np.full_like(C, np.nan)
    for si in range(NS):
        v = V[si]
        for di in range(60, ND):
            vw = v[di-60:di]
            vv = vw[~np.isnan(vw)]
            if len(vv) >= 30 and not np.isnan(v[di]):
                mv, sv = np.mean(vv), np.std(vv)
                if sv > 0:
                    vol_anomaly[si,di] = (v[di] - mv) / sv
    factors['vol_anomaly'] = vol_anomaly

    # --- ATR (for stop loss) ---
    atr = np.full_like(C, np.nan)
    for si in range(NS):
        for di in range(14, ND):
            trs = []
            for j in range(max(1,di-14), di):
                h,l,cp = H[si,j], L[si,j], C[si,j]
                if not any(np.isnan([h,l,cp])):
                    trs.append(max(h-l, abs(h-cp), abs(l-cp)))
            if trs:
                atr[si,di] = np.mean(trs)
    factors['atr_14'] = atr

    # --- Term Structure Factors ---
    ts_df = load_ts_factors()
    if ts_df is not None:
        # Build symbol mapping: prefer fi-name (rich data), fallback to short name
        available_syms = ts_df['symbol'].unique()
        sym_to_ts = {}
        for s in syms:
            if s in available_syms:
                sym_to_ts[s] = s  # direct match (agfi -> agfi)
            else:
                ts_sym = s.replace('fi', '').upper()
                if ts_sym in available_syms:
                    sym_to_ts[s] = ts_sym  # fallback (agfi -> AG)

        ts_basis_zscore = np.full_like(C, np.nan)
        ts_structure = np.full_like(C, np.nan)    # +1 backwardation, -1 contango
        ts_extreme = np.full_like(C, np.nan)
        ts_spread_mom = np.full_like(C, np.nan)   # 5d spread momentum

        for si, s in enumerate(syms):
            if s not in sym_to_ts:
                continue
            ts_sym = sym_to_ts[s]
            ts_sym_data = ts_df[ts_df['symbol'] == ts_sym].set_index('date')

            for di in range(60, ND):
                dt = dates[di]
                if dt in ts_sym_data.index:
                    row = ts_sym_data.loc[dt]
                    if isinstance(row, pd.DataFrame):
                        row = row.iloc[0]
                    ts_basis_zscore[si,di] = row.get('basis_zscore', np.nan)
                    ts_structure[si,di] = row.get('structure_state', np.nan)
                    ts_extreme[si,di] = row.get('extreme_signal', np.nan)
                    ts_spread_mom[si,di] = row.get('spread_momentum_5d', np.nan)

        factors['ts_basis_zscore'] = ts_basis_zscore
        factors['ts_structure'] = ts_structure
        factors['ts_extreme'] = ts_extreme
        factors['ts_spread_mom'] = ts_spread_mom

        ts_count = sum(1 for si in range(NS) for di in range(ND) if not np.isnan(ts_basis_zscore[si,di]))
        print(f"  TS factors loaded: {ts_count} non-null values")
    else:
        factors['ts_basis_zscore'] = np.full_like(C, np.nan)
        factors['ts_structure'] = np.full_like(C, np.nan)
        factors['ts_extreme'] = np.full_like(C, np.nan)
        factors['ts_spread_mom'] = np.full_like(C, np.nan)

    print(f"  {len(factors)} factors computed")
    return factors


# ============================================================
# SIGNAL GENERATION (enhanced with TS)
# ============================================================
def generate_signals(factors, C, V, NS, ND, syms):
    print("[V201] Generating signals...", flush=True)

    sym_to_group = {}
    for gname, gsyms in COMMODITY_GROUPS.items():
        for s in gsyms:
            sym_to_group[s] = gname

    # Group momentum
    sym_idx = {s: i for i, s in enumerate(syms)}
    group_mom = {}
    for di in range(60, ND):
        gm = {}
        for gname, gsyms in COMMODITY_GROUPS.items():
            vals = []
            for s in gsyms:
                if s in sym_idx:
                    m = factors['mom_20'][sym_idx[s], di]
                    if not np.isnan(m): vals.append(m)
            if vals: gm[gname] = np.mean(vals)
        group_mom[di] = gm

    signal_score = np.zeros((NS, ND))

    for si in range(NS):
        for di in range(60, ND):
            score = 0.0
            weight = 0.0
            s_name = syms[si]

            # --- 动量+趋势 (weight=1.0) ---
            m20 = factors['mom_20'][si, di]
            tq = factors['trend_quality'][si, di]
            if not np.isnan(m20) and not np.isnan(tq):
                if m20 > 0.02 and tq > 0.5:
                    score += 1.0
                elif m20 < -0.02 and tq < -0.5:
                    score -= 1.0
                weight += 1.0

            # --- Cross-sectional rank (weight=0.8) ---
            mr = factors['mom_rank_20'][si, di]
            if not np.isnan(mr):
                if mr > 0.75:
                    score += 0.8
                elif mr < 0.25:
                    score -= 0.8
                weight += 0.8

            # --- Group alignment (weight=0.6) ---
            if s_name in sym_to_group:
                gname = sym_to_group[s_name]
                gm = group_mom.get(di, {}).get(gname, None)
                if gm is not None and not np.isnan(m20):
                    if m20 > 0 and gm > 0: score += 0.6
                    elif m20 < 0 and gm < 0: score -= 0.6
                    weight += 0.6

            # --- Volume confirmation (weight=0.5) ---
            va = factors['vol_anomaly'][si, di]
            if not np.isnan(va) and va > 1.5 and not np.isnan(m20):
                if m20 > 0: score += 0.5
                elif m20 < 0: score -= 0.5
                weight += 0.5

            # === NEW: Term Structure Factors ===

            # --- TS extreme reversion (weight=0.8) ---
            # Extreme backwardation → buy signal (backwardation = supply tight)
            # Extreme contango → sell signal (contango = oversupply)
            ts_ext = factors['ts_extreme'][si, di]
            ts_bz = factors['ts_basis_zscore'][si, di]
            if not np.isnan(ts_ext) and ts_ext != 0:
                # Extreme backwardation (+1) → bullish, extreme contango (-1) → bearish
                score += ts_ext * 0.8
                weight += 0.8
            elif not np.isnan(ts_bz) and abs(ts_bz) > 1.5:
                score += np.sign(ts_bz) * 0.4  # weaker version
                weight += 0.4

            # --- TS structure confirms momentum (weight=0.6) ---
            ts_struct = factors['ts_structure'][si, di]
            if not np.isnan(ts_struct) and not np.isnan(m20):
                # Backwardation + uptrend → strong buy
                # Contango + downtrend → strong sell
                if ts_struct > 0 and m20 > 0.02:
                    score += 0.6
                elif ts_struct < 0 and m20 < -0.02:
                    score -= 0.6
                # Penalize conflicting signals
                elif ts_struct > 0 and m20 < -0.02:
                    score -= 0.3
                elif ts_struct < 0 and m20 > 0.02:
                    score -= 0.3
                weight += 0.6

            # --- TS spread momentum (weight=0.4) ---
            ts_sm = factors['ts_spread_mom'][si, di]
            if not np.isnan(ts_sm):
                # Spreading widening in backwardation → stronger bullish
                if ts_sm > 0: score += 0.2
                else: score -= 0.2
                weight += 0.4

            if weight >= 2.0:
                signal_score[si, di] = score / weight

    return signal_score


# ============================================================
# BACKTEST (with improved stop loss)
# ============================================================
def backtest(signal_score, factors, C, O, H, L, V, NS, ND, dates, syms,
             top_n=3, hold_days=5, atr_stop=2.0):
    print(f"[V201] Backtest: top_n={top_n}, hold={hold_days}, atr_stop={atr_stop}", flush=True)

    trades = []
    positions = []

    for di in range(60, ND):
        new_positions = []
        for si, entry_di, entry_p, stop_p, direction in positions:
            c = C[si, di]
            if np.isnan(c):
                new_positions.append((si, entry_di, entry_p, stop_p, direction))
                continue

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
                pnl = direction * (c - entry_p) / entry_p - COMMISSION
                trades.append({
                    'symbol': syms[si], 'entry_date': dates[entry_di],
                    'exit_date': dates[di], 'direction': direction,
                    'pnl_pct': pnl, 'exit_reason': exit_reason,
                    'hold_days': di - entry_di
                })
            else:
                new_positions.append((si, entry_di, entry_p, stop_p, direction))
        positions = new_positions

        if len(positions) >= top_n:
            continue

        scores = [(signal_score[si, di], si) for si in range(NS)
                  if not np.isnan(signal_score[si, di]) and abs(signal_score[si, di]) > 0.2
                  and not any(p[0] == si for p in positions)]
        scores.sort(key=lambda x: -abs(x[0]))

        for score, si in scores:
            if len(positions) >= top_n: break
            c, o = C[si, di], O[si, di]
            if np.isnan(c) or np.isnan(o): continue

            entry_p = o
            atr_val = factors['atr_14'][si, di]
            if np.isnan(atr_val) or atr_val <= 0: continue

            direction = 1 if score > 0 else -1
            stop_p = entry_p - direction * atr_stop * atr_val
            positions.append((si, di, entry_p, stop_p, direction))

    # Close remaining
    for si, entry_di, entry_p, stop_p, direction in positions:
        c = C[si, ND-1]
        if not np.isnan(c):
            pnl = direction * (c - entry_p) / entry_p - COMMISSION
            trades.append({
                'symbol': syms[si], 'entry_date': dates[entry_di],
                'exit_date': dates[ND-1], 'direction': direction,
                'pnl_pct': pnl, 'exit_reason': 'end',
                'hold_days': ND - 1 - entry_di
            })
    return trades


def print_stats(trades, label=""):
    if not trades: return
    pnls = [t['pnl_pct'] for t in trades]
    avg_hold = np.mean([t['hold_days'] for t in trades])
    ann = np.mean(pnls) * 252 / max(avg_hold, 1) * 100
    wr = sum(1 for p in pnls if p > 0) / len(pnls) * 100
    cum = np.cumprod([1+p for p in pnls])
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    mdd = dd.min() * 100
    print(f"  {label:40s} | Ann={ann:+6.1f}% | MDD={mdd:6.1f}% | WR={wr:.1f}% | N={len(trades)}")
    return ann, mdd, wr


# ============================================================
# WALK-FORWARD
# ============================================================
def walk_forward(NS, ND, dates, C, O, H, L, V, syms, train_years=4, test_years=1):
    print(f"\n{'='*70}")
    print("  WALK-FORWARD VALIDATION")
    print(f"{'='*70}")

    results = []
    start_year = dates[0].year
    end_year = dates[-1].year

    for test_start in range(start_year + train_years, end_year + 1, test_years):
        print(f"\n  Test {test_start}:", end=" ")

        factors = compute_factors(C, O, H, L, V, NS, ND, dates, syms)
        signals = generate_signals(factors, C, V, NS, ND, syms)

        for top_n, hold, atr in [(3, 5, 2.0), (3, 5, 1.5)]:
            trades = backtest(signals, factors, C, O, H, L, V, NS, ND, dates, syms,
                             top_n=top_n, hold_days=hold, atr_stop=atr)
            test_trades = [t for t in trades
                           if t['entry_date'].year == test_start]
            if test_trades:
                pnls = [t['pnl_pct'] for t in test_trades]
                ann = np.mean(pnls) * 252 / max(np.mean([t['hold_days'] for t in test_trades]), 1) * 100
                wr = sum(1 for p in pnls if p > 0) / len(pnls) * 100
                print(f"top{top_n}_h{hold}_a{atr}: Ann={ann:+.1f}% WR={wr:.1f}% N={len(test_trades)}", end="  ")
                if top_n == 3 and hold == 5 and atr == 2.0:
                    results.append({'year': test_start, 'ann': ann, 'wr': wr, 'n': len(test_trades)})
        print()

    if results:
        print(f"\n{'='*70}")
        print("  WALK-FORWARD SUMMARY")
        print(f"{'='*70}")
        for r in results:
            print(f"  {r['year']}: Ann={r['ann']:+.1f}%  WR={r['wr']:.1f}%  N={r['n']}")
        pos = sum(1 for r in results if r['ann'] > 0)
        print(f"\n  Positive: {pos}/{len(results)}")
        print(f"  Avg Ann: {np.mean([r['ann'] for r in results]):+.1f}%")
    return results


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    t0 = time.time()
    NS, ND, dates, C, O, H, L, V, syms = load_all_data(start='2016-01-01')

    print(f"\n{'='*70}")
    print("  V201 FULL-PERIOD BACKTEST")
    print(f"{'='*70}")

    factors = compute_factors(C, O, H, L, V, NS, ND, dates, syms)
    signals = generate_signals(factors, C, V, NS, ND, syms)

    for top_n, hold, atr in [(3,5,2.0), (3,5,1.5), (3,10,2.0), (5,5,2.0)]:
        trades = backtest(signals, factors, C, O, H, L, V, NS, ND, dates, syms,
                         top_n=top_n, hold_days=hold, atr_stop=atr)
        print_stats(trades, f"top{top_n}_h{hold}_a{atr}")

    walk_forward(NS, ND, dates, C, O, H, L, V, syms)

    print(f"\n  Elapsed: {time.time()-t0:.0f}s")
    print(f"{'='*70}")
