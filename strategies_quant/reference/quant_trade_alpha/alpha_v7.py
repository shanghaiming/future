"""
Alpha V7 — Strategy-Inspired Factor Library (No Look-Ahead)
=============================================================
从269个本地策略中提取独立信号维度，构建截面因子库。

关键原则:
  1. 所有因子只用 di-1 及更早的数据（无前瞻性偏差）
  2. 交易在 di 的开盘价执行
  3. 因子必须是连续值，可截面排名（0-100）
  4. 每个因子代表一个独立的信号维度

因子来源:
  - MOM5/MOM10/MOM20: 来自多数动量策略（ma_strategy等）
  - KINETIC_ENERGY: 来自 energt_structure.py 动能势能模型
  - BODY_RATIO: 来自K线分析（energt_structure, kline.py）
  - SHADOW_PRESSURE: 来自 energt_structure.py 压力支撑模型
  - STRUCT_TENSION: 来自 structural_tension_strategy.py 7点结构张力
  - FISHER: 来自 fisher_regime_strategy.py Fisher变换
  - DRAWDOWN_52W: 来自 metis_ladder_strategy.py 52周回撤
  - LINREG_SLOPE: 来自 stategy_lineregression.py 线性回归
  - BREAKOUT_ENERGY: 来自 energt_structure.py 突破势能
  - VDP: 来自 volume_delta_pressure_strategy.py 量能压力
  - VOL_ANOMALY: 来自 epanechnikov_confluence_strategy.py 成交量异常
  - VOLATILITY_PCT: 来自 volatility_regime_strategy.py 波动率状态
"""
import sys, os, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.data_loader import list_available_symbols, load_stock_data
from alpha_v2 import load_all_data, MIN_TRAIN

COMMISSION = 0.0003
STAMP_DUTY = 0.001
CASH0 = 500_000


def compute_all_factors(NS, ND, C, O, H, L, V):
    """Compute all factors using ONLY data up to di-1. NO look-ahead."""
    print("[Factors] Computing all NO-LOOK-AHEAD factors...", flush=True)
    t0 = time.time()
    factors = {}

    # === 1. MOM5 / MOM10 / MOM20 (from ma/trend strategies) ===
    for period, name in [(5, 'MOM5'), (10, 'MOM10'), (20, 'MOM20')]:
        arr = np.full((NS, ND), np.nan)
        for si in range(NS):
            for di in range(period + 1, ND):
                c_now = C[si, di - 1]       # yesterday's close
                c_prev = C[si, di - 1 - period]  # period days before yesterday
                if not np.isnan(c_now) and not np.isnan(c_prev) and c_prev > 0:
                    arr[si, di] = (c_now - c_prev) / c_prev
        factors[name] = arr
    print(f"  Momentum done ({time.time()-t0:.1f}s)", flush=True)

    # === 2. KINETIC_ENERGY (from energt_structure.py) ===
    # kinetic = log_return * volume — raw price movement energy
    # High kinetic = strong directional conviction with volume backing
    KINETIC = np.full((NS, ND), np.nan)
    for si in range(NS):
        ema_kinetic = np.nan
        alpha = 2.0 / 11  # 10-day EMA
        for di in range(2, ND):
            d = di - 1  # use yesterday's data
            c0, c1 = C[si, d - 1], C[si, d]
            v = V[si, d]
            if np.isnan(c0) or np.isnan(c1) or c0 <= 0 or np.isnan(v) or v <= 0:
                continue
            log_ret = np.log(c1 / c0)
            kinetic = log_ret * v  # raw energy
            if np.isnan(ema_kinetic):
                ema_kinetic = kinetic
            else:
                ema_kinetic = alpha * kinetic + (1 - alpha) * ema_kinetic
            KINETIC[si, di] = ema_kinetic
    factors['KINETIC'] = KINETIC
    print(f"  Kinetic energy done ({time.time()-t0:.1f}s)", flush=True)

    # === 3. BODY_RATIO (from K-line analysis) ===
    # (C-O)/(H-L) — measures strength of directional move
    # High body ratio = strong conviction, small shadows = no rejection
    # Use EMA to smooth
    BODY = np.full((NS, ND), np.nan)
    for si in range(NS):
        ema_body = np.nan
        alpha = 2.0 / 6  # 5-day EMA
        for di in range(2, ND):
            d = di - 1
            o, c, h, l = O[si, d], C[si, d], H[si, d], L[si, d]
            if np.isnan(o) or np.isnan(c) or np.isnan(h) or np.isnan(l):
                continue
            hl = h - l
            if hl <= 0:
                continue
            body_ratio = (c - o) / hl  # -1 to +1
            if np.isnan(ema_body):
                ema_body = body_ratio
            else:
                ema_body = alpha * body_ratio + (1 - alpha) * ema_body
            BODY[si, di] = ema_body
    factors['BODY_RATIO'] = BODY
    print(f"  Body ratio done ({time.time()-t0:.1f}s)", flush=True)

    # === 4. SHADOW_PRESSURE (from energt_structure pressure-support model) ===
    # lower_shadow * vol = buying support, upper_shadow * vol = selling pressure
    # Net = support - pressure (positive = buying pressure dominates)
    SHADOW = np.full((NS, ND), np.nan)
    for si in range(NS):
        ema_shadow = np.nan
        alpha = 2.0 / 6  # 5-day EMA
        for di in range(2, ND):
            d = di - 1
            o, c, h, l = O[si, d], C[si, d], H[si, d], L[si, d]
            v = V[si, d]
            if np.isnan(o) or np.isnan(c) or np.isnan(h) or np.isnan(l) or np.isnan(v):
                continue
            if v <= 0:
                continue
            total_range = h - l
            if total_range <= 0:
                continue
            upper_shadow = (h - max(o, c)) / total_range
            lower_shadow = (min(o, c) - l) / total_range
            # Directional weighting
            if c > o:  # bullish candle
                pressure = (lower_shadow - upper_shadow * 0.5) * v
            else:  # bearish candle
                pressure = (lower_shadow * 0.5 - upper_shadow) * v
            if np.isnan(ema_shadow):
                ema_shadow = pressure
            else:
                ema_shadow = alpha * pressure + (1 - alpha) * ema_shadow
            SHADOW[si, di] = ema_shadow
    factors['SHADOW_PRESSURE'] = SHADOW
    print(f"  Shadow pressure done ({time.time()-t0:.1f}s)", flush=True)

    # === 5. VDP (from volume_delta_pressure_strategy.py) ===
    # VDP = V * (2C - H - L) / (H - L) — net buying/selling pressure
    VDP = np.full((NS, ND), np.nan)
    for si in range(NS):
        ema_vdp = np.nan
        alpha = 2.0 / 11  # 10-day EMA
        for di in range(2, ND):
            d = di - 1
            c, h, l, v = C[si, d], H[si, d], L[si, d], V[si, d]
            if np.isnan(c) or np.isnan(h) or np.isnan(l) or np.isnan(v) or v <= 0:
                continue
            hl = h - l
            if hl <= 0:
                continue
            delta = v * (2 * c - h - l) / hl
            if np.isnan(ema_vdp):
                ema_vdp = delta
            else:
                ema_vdp = alpha * delta + (1 - alpha) * ema_vdp
            VDP[si, di] = ema_vdp
    factors['VDP'] = VDP
    print(f"  VDP done ({time.time()-t0:.1f}s)", flush=True)

    # === 6. FISHER (from fisher_regime_strategy.py) ===
    # Fisher transform: converts non-normal price to approximate Gaussian
    # F = 0.5 * ln((1+X)/(1-X)) where X is normalized price in [-1, +1]
    FISHER = np.full((NS, ND), np.nan)
    FISHER_PERIOD = 10
    for si in range(NS):
        prev_fisher = 0.0
        for di in range(FISHER_PERIOD + 1, ND):
            # Use data up to di-1
            h_win = H[si, di - 1 - FISHER_PERIOD:di]
            l_win = L[si, di - 1 - FISHER_PERIOD:di]
            c = C[si, di - 1]
            if np.any(np.isnan(h_win)) or np.any(np.isnan(l_win)) or np.isnan(c):
                prev_fisher = 0.0
                continue
            h_max = np.max(h_win)
            l_min = np.min(l_win)
            rng = h_max - l_min
            if rng <= 0:
                prev_fisher = 0.0
                continue
            x = 2.0 * (c - l_min) / rng - 1.0
            x = np.clip(x, -0.999, 0.999)
            fisher = 0.5 * np.log((1 + x) / (1 - x))
            fisher = 0.5 * fisher + 0.5 * prev_fisher  # EMA smoothing
            FISHER[si, di] = fisher
            prev_fisher = fisher
    factors['FISHER'] = FISHER
    print(f"  Fisher done ({time.time()-t0:.1f}s)", flush=True)

    # === 7. DRAWDOWN_52W (from metis_ladder_strategy.py) ===
    # Depth from 52-week high — mean reversion signal
    DRAWDOWN = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(253, ND):
            start = max(0, di - 1 - 252)
            h_win = H[si, start:di]  # 52w high up to yesterday
            h_valid = h_win[~np.isnan(h_win)]
            if len(h_valid) < 50:
                continue
            h252 = np.max(h_valid)
            c = C[si, di - 1]
            if h252 > 0 and not np.isnan(c):
                DRAWDOWN[si, di] = (c - h252) / h252  # negative = below high
    factors['DRAWDOWN_52W'] = DRAWDOWN
    print(f"  Drawdown 52w done ({time.time()-t0:.1f}s)", flush=True)

    # === 8. LINREG_SLOPE + R² (from stategy_lineregression.py) ===
    # Linear regression slope over 20 days = trend direction + strength
    # R² = trend quality (high = clean trend, low = noisy)
    LINREG = np.full((NS, ND), np.nan)
    R_SQUARED = np.full((NS, ND), np.nan)
    LR_PERIOD = 20
    for si in range(NS):
        for di in range(LR_PERIOD + 1, ND):
            # Use closes up to di-1
            c_win = C[si, di - 1 - LR_PERIOD:di]
            valid = ~np.isnan(c_win)
            if valid.sum() < LR_PERIOD:
                continue
            y = c_win[valid]
            n = len(y)
            x = np.arange(n, dtype=float)
            mx, my = np.mean(x), np.mean(y)
            ss_xx = np.sum((x - mx) ** 2)
            ss_xy = np.sum((x - mx) * (y - my))
            ss_yy = np.sum((y - my) ** 2)
            if ss_xx > 0 and ss_yy > 0 and my > 0:
                slope = ss_xy / ss_xx
                r2 = (ss_xy ** 2) / (ss_xx * ss_yy)
                LINREG[si, di] = slope / my * 100  # normalize by price level
                R_SQUARED[si, di] = r2
    factors['LINREG_SLOPE'] = LINREG
    factors['R_SQUARED'] = R_SQUARED
    print(f"  Linear regression done ({time.time()-t0:.1f}s)", flush=True)

    # === 9. BREAKOUT_ENERGY (from energt_structure.py breakout model) ===
    # Energy when price breaks through recent high/low with volume
    # breakout = 1 if C > max(C[-20:-1]) and V > V_ma20, else -1
    BREAKOUT = np.full((NS, ND), np.nan)
    for si in range(NS):
        ema_brk = np.nan
        alpha = 2.0 / 6  # 5-day EMA
        for di in range(22, ND):
            d = di - 1
            c = C[si, d]
            c20 = C[si, d - 20:d]  # 20 days before yesterday
            v = V[si, d]
            v20 = V[si, d - 20:d]
            if np.isnan(c) or len(c20[~np.isnan(c20)]) < 15:
                continue
            max20 = np.nanmax(c20)
            min20 = np.nanmin(c20)
            v_valid = v20[~np.isnan(v20) & (v20 > 0)]
            if np.isnan(v) or len(v_valid) < 10 or np.isnan(max20):
                continue
            avg_v = np.mean(v_valid)
            if avg_v <= 0 or max20 <= 0:
                continue
            log_ret = np.log(c / C[si, d - 1]) if not np.isnan(C[si, d - 1]) and C[si, d - 1] > 0 else 0
            vol_ratio = v / avg_v
            if c > max20 and vol_ratio > 1.0:
                energy = abs(log_ret) * v * 2  # double energy for breakout
            elif c < min20 and vol_ratio > 1.0:
                energy = -abs(log_ret) * v * 2  # downward breakout
            else:
                energy = log_ret * v
            if np.isnan(ema_brk):
                ema_brk = energy
            else:
                ema_brk = alpha * energy + (1 - alpha) * ema_brk
            BREAKOUT[si, di] = ema_brk
    factors['BREAKOUT'] = BREAKOUT
    print(f"  Breakout energy done ({time.time()-t0:.1f}s)", flush=True)

    # === 10. VOL_ANOMALY (from epanechnikov strategy) ===
    # Volume / 20-day avg volume — measures participation intensity
    VOL_ANOM = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(22, ND):
            d = di - 1
            v = V[si, d]
            v20 = V[si, d - 20:d]
            v_valid = v20[~np.isnan(v20) & (v20 > 0)]
            if np.isnan(v) or len(v_valid) < 10:
                continue
            avg_v = np.mean(v_valid)
            if avg_v > 0:
                VOL_ANOM[si, di] = v / avg_v
    factors['VOL_ANOMALY'] = VOL_ANOM
    print(f"  Volume anomaly done ({time.time()-t0:.1f}s)", flush=True)

    # === 11. VOLATILITY_PCT (from volatility_regime_strategy.py) ===
    # ATR percentile — vectorized computation
    VOL_PCT = np.full((NS, ND), np.nan)
    ATR_P = 14
    for si in range(NS):
        # Compute True Range vectorized
        h, l, c = H[si], L[si], C[si]
        tr = np.full(ND, np.nan)
        for d in range(1, ND):
            if np.isnan(h[d]) or np.isnan(l[d]):
                continue
            val = h[d] - l[d]
            if not np.isnan(c[d - 1]):
                val = max(val, abs(h[d] - c[d - 1]), abs(l[d] - c[d - 1]))
            tr[d] = val
        # Rolling ATR using cumsum trick
        valid_tr = ~np.isnan(tr)
        cumsum = np.nancumsum(tr)
        atr_arr = np.full(ND, np.nan)
        for d in range(ATR_P, ND):
            window = tr[d - ATR_P + 1:d + 1]
            v = window[~np.isnan(window)]
            if len(v) >= ATR_P - 2:
                atr_arr[d] = np.mean(v)
        # Percentile of current ATR in last 50 ATRs
        for di in range(61, ND):
            d = di - 1  # use yesterday's data
            if np.isnan(atr_arr[d]):
                continue
            atr_win = atr_arr[max(ATR_P, d - 50):d + 1]
            atr_valid = atr_win[~np.isnan(atr_win)]
            if len(atr_valid) < 10:
                continue
            pct = np.sum(atr_valid < atr_arr[d]) / max(len(atr_valid) - 1, 1) * 100
            VOL_PCT[si, di] = pct
    factors['VOLATILITY_PCT'] = VOL_PCT
    print(f"  Volatility percentile done ({time.time()-t0:.1f}s)", flush=True)

    # === 12. STRUCT_TENSION (from structural_tension_strategy.py) ===
    # Price displacement from key structural points normalized by ATR
    # Simplified: displacement from recent swing high/low and mid
    TENSION = np.full((NS, ND), np.nan)
    PIVOT_LEN = 5
    for si in range(NS):
        for di in range(PIVOT_LEN * 2 + 5, ND):
            # Find pivots in data up to di-1
            h_arr = H[si, di - 1 - PIVOT_LEN * 2:di]
            l_arr = L[si, di - 1 - PIVOT_LEN * 2:di]
            c_arr = C[si, di - 1 - PIVOT_LEN * 2:di]
            n = len(h_arr)
            if n < PIVOT_LEN * 2:
                continue

            # Find recent swing high/low
            swing_h, swing_l = np.nanmax(h_arr[-10:]), np.nanmin(l_arr[-10:])
            mid = (swing_h + swing_l) / 2.0
            atr = np.nanmean(h_arr[-14:]) - np.nanmean(l_arr[-14:]) if n >= 14 else 0

            c = C[si, di - 1]
            if np.isnan(c) or np.isnan(swing_h) or np.isnan(swing_l):
                continue
            if atr <= 0:
                # Use simple range as proxy
                atr = swing_h - swing_l
            if atr <= 0:
                continue

            # Tension = displacement from 3 reference points
            d_h = (c - swing_h) / atr
            d_l = (c - swing_l) / atr
            d_m = (c - mid) / atr
            TENSION[si, di] = np.mean([d_h, d_l, d_m])
    factors['TENSION'] = TENSION
    print(f"  Structural tension done ({time.time()-t0:.1f}s)", flush=True)

    # === 13. PRICE_PERCENTILE (price position in 60-day range) ===
    PCT = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(61, ND):
            vals = C[si, di - 1 - 60:di]
            valid = vals[~np.isnan(vals)]
            if len(valid) < 30:
                continue
            cur = C[si, di - 1]
            if np.isnan(cur):
                continue
            PCT[si, di] = np.sum(valid < cur) / max(len(valid) - 1, 1) * 100
    factors['PRICE_PCT'] = PCT
    print(f"  Price percentile done ({time.time()-t0:.1f}s)", flush=True)

    # === Cross-sectional rank normalization ===
    def rank_pct(arr, start=60):
        """Rank normalize to 0-100 across stocks for each day."""
        res = np.full_like(arr, np.nan)
        for di in range(start, arr.shape[1]):
            vals = arr[:, di]
            mask = ~np.isnan(vals)
            if mask.sum() < 50:
                continue
            ranked = np.argsort(np.argsort(vals[mask])).astype(float)
            n = len(ranked)
            pct = ranked / max(n - 1, 1) * 100
            for k, idx in enumerate(np.where(mask)[0]):
                res[idx, di] = pct[k]
        return res

    # Rank normalize all factors
    ranked_factors = {}
    for name, arr in factors.items():
        ranked_factors[f'R_{name}'] = rank_pct(arr)
        print(f"  Ranked {name} done ({time.time()-t0:.1f}s)", flush=True)

    # === Delta transforms (change in rank over N days) ===
    def delta_rank(arr, lag=3):
        res = np.full_like(arr, np.nan)
        for di in range(lag, arr.shape[1]):
            for si in range(arr.shape[0]):
                if not np.isnan(arr[si, di]) and not np.isnan(arr[si, di - lag]):
                    res[si, di] = arr[si, di] - arr[si, di - lag]
        return res

    delta_names = list(ranked_factors.keys())
    for name in delta_names:
        arr = ranked_factors[name]
        ranked_factors[f'D_{name[2:]}_3'] = delta_rank(arr, 3)

    print(f"\n  All factors done! Total: {len(ranked_factors)} factors ({time.time()-t0:.1f}s)", flush=True)
    return ranked_factors


def backtest_v7(factor_weights, factors, NS, ND, dates, C, O, H, L, V,
                top_n=10, rebalance_days=5):
    """No-look-ahead backtest. Factors use di-1 data, trade at di open."""
    factor_names = list(factor_weights.keys())
    weights = np.array([factor_weights[f] for f in factor_names])

    cash = float(CASH0)
    holdings = []
    trades = []
    last_rebalance = -999
    year_stats = {}

    for di in range(MIN_TRAIN, ND):
        year = dates[di].year

        if di - last_rebalance >= rebalance_days:
            # Composite score using factors at di (computed from di-1 data)
            composite = np.zeros(NS)
            count = np.zeros(NS)
            for fname, w in zip(factor_names, weights):
                if fname not in factors:
                    continue
                arr = factors[fname]
                vals = arr[:, di]
                valid = ~np.isnan(vals)
                if valid.sum() < 50:
                    continue
                composite[valid] += w * vals[valid]
                count[valid] += abs(w)

            mask = count > 0
            if mask.sum() < top_n * 2:
                continue
            composite[mask] /= count[mask]
            composite[~mask] = -9999

            top_indices = set(np.argsort(-composite)[:top_n])
            current_indices = set(h['si'] for h in holdings)

            # Sell at open price
            to_sell = current_indices - top_indices
            for pos in list(holdings):
                if pos['si'] in to_sell:
                    p = O[pos['si'], di]
                    if np.isnan(p) or p <= 0:
                        p = C[pos['si'], di]
                    if not np.isnan(p) and p > 0:
                        pnl = (p - pos['entry']) / pos['entry'] * 100
                        cash += pos['shares'] * p * (1 - COMMISSION - STAMP_DUTY)
                        trades.append({
                            'pnl': pnl, 'days': (dates[di] - pos['ed']).days,
                            'di': di, 'reason': 'rebalance', 'year': year
                        })
                        holdings.remove(pos)

            # Buy at open price
            current_indices = set(h['si'] for h in holdings)
            to_buy = top_indices - current_indices
            n_to_buy = len(to_buy)
            if n_to_buy > 0 and cash > 10000:
                alloc = cash / n_to_buy
                for si in to_buy:
                    p = O[si, di]
                    if np.isnan(p) or p <= 0:
                        p = C[si, di]
                    if not np.isnan(p) and p > 0:
                        shares = int(alloc / (1 + COMMISSION) / p)
                        if shares > 0:
                            cost = shares * p * (1 + COMMISSION)
                            if cost <= cash:
                                cash -= cost
                                holdings.append({
                                    'si': si, 'shares': shares, 'entry': p,
                                    'ed': dates[di]
                                })
            last_rebalance = di

    # Close remaining
    for pos in holdings:
        p = C[pos['si'], ND - 1]
        if not np.isnan(p) and p > 0:
            pnl = (p - pos['entry']) / pos['entry'] * 100
            cash += pos['shares'] * p * (1 - COMMISSION - STAMP_DUTY)
            trades.append({'pnl': pnl, 'days': 999, 'di': ND - 1, 'reason': 'end',
                           'year': dates[ND - 1].year})

    if cash <= 0 or not trades:
        return None

    days_total = (dates[ND - 1] - dates[MIN_TRAIN]).days
    yr = max(days_total / 365.25, 0.01)
    ann = ((cash / CASH0) ** (1 / yr) - 1) * 100
    nw = sum(1 for t in trades if t['pnl'] > 0)
    wr = nw / max(len(trades), 1) * 100
    avg_w = np.mean([t['pnl'] for t in trades if t['pnl'] > 0]) if nw > 0 else 0
    avg_l = np.mean([abs(t['pnl']) for t in trades if t['pnl'] <= 0]) if nw < len(trades) else 0

    for t in trades:
        y = t.get('year', 'unknown')
        if y not in year_stats:
            year_stats[y] = {'trades': 0, 'wins': 0, 'total_pnl': 0}
        year_stats[y]['trades'] += 1
        if t['pnl'] > 0:
            year_stats[y]['wins'] += 1
        year_stats[y]['total_pnl'] += t['pnl']

    equity = float(CASH0)
    peak = float(CASH0)
    max_dd = 0
    for t in sorted(trades, key=lambda x: x['di']):
        equity *= (1 + t['pnl'] / 100)
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak * 100
        if dd > max_dd:
            max_dd = dd

    return {
        'ann': round(ann, 1), 'n': len(trades), 'wr': round(wr, 1),
        'avg_w': round(avg_w, 1), 'avg_l': round(avg_l, 1),
        'edge': round((nw / max(len(trades), 1)) * avg_w - (1 - nw / max(len(trades), 1)) * avg_l, 2),
        'max_dd': round(max_dd, 1), 'tpy': round(len(trades) / yr, 1),
        'final': round(cash, 0), 'year_stats': year_stats,
    }


if __name__ == '__main__':
    print("=" * 70, flush=True)
    print("  Alpha V7 — Strategy-Inspired Factor Library (No Look-Ahead)", flush=True)
    print("  All factors use di-1 data, trade at di open", flush=True)
    print("=" * 70, flush=True)

    NS, ND, dates, C, O, H, L, V, syms, sym_set = load_all_data()
    factors = compute_all_factors(NS, ND, C, O, H, L, V)

    # Single factor tests
    single_results = []
    factor_names = [k for k in factors.keys() if k.startswith('R_') and not k.startswith('D_')]
    print(f"\n{'='*80}", flush=True)
    print(f"  SINGLE FACTOR TESTS (Top=10, Reb=5)", flush=True)
    print(f"{'='*80}", flush=True)
    for fname in factor_names:
        r = backtest_v7({fname: 1.0}, factors, NS, ND, dates, C, O, H, L, V,
                        top_n=10, rebalance_days=5)
        if r:
            single_results.append({'factor': fname, **r})
            print(f"  {fname:<25s} | Ann={r['ann']:+7.1f}% WR={r['wr']:5.1f}% "
                  f"DD={r['max_dd']:5.1f}% Edge={r['edge']:+.2f}%", flush=True)

    single_results.sort(key=lambda x: -x['ann'])
    print(f"\n  Best single factors:", flush=True)
    for r in single_results[:10]:
        print(f"    {r['factor']:<25s} → {r['ann']:+.1f}% (WR={r['wr']:.0f}%, DD={r['max_dd']:.1f}%)", flush=True)

    # Multi-factor combinations
    portfolios = {
        'All_Equal': {f: 1.0 / len(factor_names) for f in factor_names},
        'Momentum': {'R_MOM5': 0.3, 'R_MOM10': 0.2, 'R_MOM20': 0.2,
                     'R_LINREG_SLOPE': 0.3},
        'Energy': {'R_KINETIC': 0.35, 'R_BREAKOUT': 0.35, 'R_SHADOW_PRESSURE': 0.3},
        'Volume': {'R_VDP': 0.4, 'R_VOL_ANOMALY': 0.3, 'R_SHADOW_PRESSURE': 0.3},
        'Structure': {'R_TENSION': 0.3, 'R_PRICE_PCT': 0.3, 'R_FISHER': 0.2,
                      'R_LINREG_SLOPE': 0.2},
        'MomEnergy': {'R_MOM5': 0.2, 'R_KINETIC': 0.25, 'R_BREAKOUT': 0.25,
                      'R_VOL_ANOMALY': 0.15, 'R_BODY_RATIO': 0.15},
        'MomStruct': {'R_MOM5': 0.25, 'R_TENSION': 0.25, 'R_FISHER': 0.25,
                      'R_LINREG_SLOPE': 0.25},
        'Full_V7': {'R_MOM5': 0.08, 'R_KINETIC': 0.08, 'R_VDP': 0.08,
                    'R_BREAKOUT': 0.08, 'R_FISHER': 0.08, 'R_TENSION': 0.08,
                    'R_BODY_RATIO': 0.08, 'R_SHADOW_PRESSURE': 0.08,
                    'R_VOL_ANOMALY': 0.08, 'R_LINREG_SLOPE': 0.08,
                    'R_PRICE_PCT': 0.08, 'R_VOLATILITY_PCT': 0.04,
                    'R_DRAWDOWN_52W': 0.04, 'R_R_SQUARED': 0.04},
        'Reversion': {'R_DRAWDOWN_52W': 0.4, 'R_FISHER': 0.3, 'R_VOLATILITY_PCT': 0.3},
        'Best5': {},  # Will be filled dynamically
    }

    # Fill Best5 with top 5 single factors
    if len(single_results) >= 5:
        best5_names = [r['factor'] for r in single_results[:5]]
        portfolios['Best5'] = {f: 0.2 for f in best5_names}

    results = []
    for pname, weights in portfolios.items():
        if not weights:
            continue
        for top_n in [5, 10, 15, 20]:
            for rebal in [3, 5, 10, 20]:
                r = backtest_v7(weights, factors, NS, ND, dates, C, O, H, L, V,
                                top_n=top_n, rebalance_days=rebal)
                if r:
                    results.append({
                        'portfolio': pname,
                        'top_n': top_n,
                        'rebal': rebal,
                        **r
                    })
        print(f"  {pname} done", flush=True)

    results.sort(key=lambda x: -x['ann'])

    print(f"\n{'='*110}", flush=True)
    print(f"  TOP 30 MULTI-FACTOR RESULTS", flush=True)
    print(f"  {'Portfolio':<15s} {'Top':>3s} {'Reb':>3s} | {'Ann':>7s} {'N':>5s} {'TPY':>4s} {'WR':>5s} "
          f"{'Edge':>6s} {'DD':>5s}", flush=True)
    print(f"  {'-'*100}", flush=True)
    for r in results[:30]:
        print(f"  {r['portfolio']:<15s} {r['top_n']:3d} {r['rebal']:3d} | {r['ann']:+7.1f}% {r['n']:5d} "
              f"{r['tpy']:4.0f} {r['wr']:5.1f}% {r['edge']:+6.2f}% {r['max_dd']:5.1f}%", flush=True)

    # Best per portfolio
    best_per = {}
    for r in results:
        p = r['portfolio']
        if p not in best_per or r['ann'] > best_per[p]['ann']:
            best_per[p] = r
    print(f"\n  Best per portfolio:", flush=True)
    for r in sorted(best_per.values(), key=lambda x: -x['ann']):
        print(f"    {r['portfolio']:<15s} → {r['ann']:+.1f}% (Top={r['top_n']}, Reb={r['rebal']}, "
              f"WR={r['wr']:.0f}%, DD={r['max_dd']:.1f}%, TPY={r['tpy']:.0f})", flush=True)

    # Year-by-year for best
    if best_per:
        best = sorted(best_per.values(), key=lambda x: -x['ann'])[0]
        print(f"\n  Year-by-year: {best['portfolio']} (Ann={best['ann']:+.1f}%)", flush=True)
        for y in sorted(best.get('year_stats', {}).keys()):
            s = best['year_stats'][y]
            wr = s['wins'] / max(s['trades'], 1) * 100
            print(f"    {y}: {s['trades']:4d} trades, WR={wr:.0f}%, pnl={s['total_pnl']:+.0f}%", flush=True)

    print(f"\n{'='*70}", flush=True)
    print(f"  ALPHA V7 COMPLETE — NO LOOK-AHEAD", flush=True)
    print(f"{'='*70}", flush=True)
