"""
Alpha Futures V12 — VDP+OI 日内策略 (无杠杆, 无Gap)
====================================================
核心: 每天用VDP+OI+K线形态决定做多/做空方向
      在当天open买入, close平仓 — 日内交易, 无隔夜风险
      不使用任何gap/隔夜跳空信号

信号来源 (全部用前一日数据):
  A. VDP方向: 前一日量压方向 + 前一日K线方向 → 当天顺势
  B. OI趋势: OI连续增加 + VDP确认 → 趋势延续
  C. K线形态: 锤子线/吞没 + VDP + OI → 反转
  D. 综合评分: 多因子排名, 选最强品种
"""
import sys, os, time, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alpha_v2 import load_all_data, MIN_TRAIN, COMMISSION, STAMP_DUTY, CASH0

MULT = {
    'agfi': 15, 'alfi': 5, 'aufi': 1000, 'bufi': 10, 'cufi': 5,
    'fufi': 10, 'rbfi': 10, 'znfi': 5, 'nifi': 1, 'hcfi': 10,
    'spfi': 10, 'ssfi': 5, 'sffi': 5, 'smfi': 5, 'pbfi': 5,
    'snfi': 1, 'rufi': 10, 'wrffi': 10,
    'afi': 10, 'bfi': 10, 'bbfi': 500, 'cffi': 5, 'cfi': 10,
    'csfi': 10, 'ebfi': 5, 'egfi': 10, 'fbfi': 500,
    'ifi': 100, 'jfi': 100, 'jmfi': 60, 'lfi': 5, 'mfi': 10,
    'pgfi': 20, 'ppfi': 5, 'vfi': 5, 'yfi': 10, 'pfi': 10,
    'jdfi': 5, 'lhfi': 16, 'pkfi': 5, 'rrfi': 20, 'lrfi': 20,
    'jrfi': 20, 'pmfi': 20, 'whfi': 20, 'rsfi': 20, 'cjfi': 10,
    'mafi': 10, 'apfi': 10, 'cyfi': 5, 'fgfi': 20, 'oifi': 10,
    'pfifi': 5, 'rmfi': 10, 'srfi': 10, 'tafi': 5, 'safi': 20,
    'urfi': 20, 'scfi': 1000, 'lufi': 10, 'bcfi': 5, 'nrfi': 1,
    'lgfi': 20, 'brfi': 5, 'lcfi': 1, 'sifi': 5,
    'ni': 1, 'tai': 5,
}
DEF_MULT = 10
COMM_RATE = 0.0003


def daily_backtest(NS, ND, dates, C, O, H, L, V, OI, syms,
                    signal_func, max_pos=1, slippage=0.001,
                    oi_mom5=None, vol_ratio=None, vdp_arr=None,
                    vdp_ema=None, atr_arr=None):
    """
    日内策略回测
    每天用signal_func选出品种, 当天open进close出
    signal_func(si, di) → (direction, score) or None
    """
    cash = float(CASH0)
    trades = []
    year_stats = {}

    for di in range(MIN_TRAIN, ND):
        year = dates[di].year
        candidates = []

        for si in range(NS):
            result = signal_func(si, di, C, O, H, L, V, OI,
                                  oi_mom5, vol_ratio, vdp_arr, vdp_ema, atr_arr)
            if result is not None:
                direction, score = result
                candidates.append((si, direction, score))

        if not candidates:
            continue

        candidates.sort(key=lambda x: x[2], reverse=True)
        remaining = cash

        for si, direction, score in candidates[:max_pos]:
            open_p = O[si, di]
            close_p = C[si, di]
            if np.isnan(open_p) or np.isnan(close_p) or open_p <= 0:
                continue

            mult = MULT.get(syms[si], DEF_MULT)
            notional = open_p * mult
            if notional <= 0: continue

            alloc = remaining / max(1, max_pos - candidates[:max_pos].index((si, direction, score)))
            lots = int(alloc / notional)
            if lots <= 0: continue
            while lots * notional > remaining and lots > 0:
                lots -= 1
            if lots <= 0: continue

            # Apply slippage
            if direction == 1:
                entry = open_p * (1 + slippage)
                exit_p = close_p * (1 - slippage)
            else:
                entry = open_p * (1 - slippage)
                exit_p = close_p * (1 + slippage)

            pnl = (exit_p - entry) * mult * lots * direction
            comm = notional * lots * COMM_RATE * 2
            pnl -= comm

            notional_used = notional * lots
            remaining -= notional_used
            cash += pnl

            pnl_pct = pnl / notional_used * 100 if notional_used > 0 else 0
            trades.append({
                'pnl_pct': pnl_pct, 'pnl_abs': pnl,
                'di': di, 'year': year, 'si': si, 'dir': direction,
            })

    if not trades: return None

    equity = float(CASH0); peak = float(CASH0); max_dd = 0
    for t in sorted(trades, key=lambda x: x['di']):
        equity += t['pnl_abs']
        if equity > peak: peak = equity
        dd = (peak - equity) / peak * 100 if peak > 0 else 0
        if dd > max_dd: max_dd = dd

    final_cash = cash
    if final_cash <= 0: return None

    days_total = (dates[ND-1] - dates[MIN_TRAIN]).days
    yr = max(days_total / 365.25, 0.01)
    ann = ((final_cash / CASH0) ** (1 / yr) - 1) * 100

    nw = sum(1 for t in trades if t['pnl_abs'] > 0)
    wr = nw / max(len(trades), 1) * 100
    avg_win = np.mean([t['pnl_pct'] for t in trades if t['pnl_abs'] > 0]) if nw > 0 else 0
    avg_loss = np.mean([abs(t['pnl_pct']) for t in trades if t['pnl_abs'] <= 0]) if nw < len(trades) else 0

    # Sharpe
    daily_rets = {}
    for t in trades:
        key = dates[t['di']].strftime('%Y-%m-%d')
        daily_rets[key] = daily_rets.get(key, 0) + t['pnl_pct'] / 100
    rets = list(daily_rets.values())
    sharpe = np.mean(rets) / np.std(rets) * np.sqrt(252) if len(rets) > 30 and np.std(rets) > 0 else 0

    for t in trades:
        y = t.get('year', 'unknown')
        if y not in year_stats:
            year_stats[y] = {'trades': 0, 'wins': 0, 'total_pnl': 0, 'pnl_abs_sum': 0}
        year_stats[y]['trades'] += 1
        if t['pnl_abs'] > 0: year_stats[y]['wins'] += 1
        year_stats[y]['total_pnl'] += t['pnl_pct']
        year_stats[y]['pnl_abs_sum'] += t['pnl_abs']

    return {
        'ann': round(ann, 1), 'n': len(trades), 'wr': round(wr, 1),
        'max_dd': round(max_dd, 1), 'final': round(final_cash, 0),
        'avg_win': round(avg_win, 2), 'avg_loss': round(avg_loss, 2),
        'sharpe': round(sharpe, 2),
        'year_stats': year_stats,
    }


# ====================================================================
# Signal Functions (ALL use di-1 data only, NO look-ahead, NO gap)
# ====================================================================

def make_signal_vdp_momentum(si, di, C, O, H, L, V, OI,
                              oi_mom5, vol_ratio, vdp_arr, vdp_ema, atr_arr):
    """A. VDP动量: 昨日VDP>0 + 昨日阳线 → 今日做多"""
    d = di - 1
    c = C[si, d]; o = O[si, d]
    vdp_v = vdp_ema[si, di] if vdp_ema is not None else np.nan
    if np.isnan(c) or np.isnan(o) or np.isnan(vdp_v): return None

    # 昨日K线方向
    candle_bull = c > o * 1.001  # 阳线
    candle_bear = c < o * 0.999  # 阴线

    # VDP确认
    if candle_bull and vdp_v > 0:
        return (1, abs(vdp_v))
    if candle_bear and vdp_v < 0:
        return (-1, abs(vdp_v))
    return None


def make_signal_vdp_oi(si, di, C, O, H, L, V, OI,
                        oi_mom5, vol_ratio, vdp_arr, vdp_ema, atr_arr):
    """B. VDP+OI: VDP方向 + OI增加 → 强信号"""
    d = di - 1
    c = C[si, d]; o = O[si, d]
    vdp_v = vdp_ema[si, di] if vdp_ema is not None else np.nan
    oi_m = oi_mom5[si, di] if oi_mom5 is not None else np.nan
    vr = vol_ratio[si, di] if vol_ratio is not None else np.nan

    if np.isnan(c) or np.isnan(o) or np.isnan(vdp_v): return None

    candle_bull = c > o * 1.001
    candle_bear = c < o * 0.999

    score = abs(vdp_v)

    # OI增加 = 新资金进入
    if not np.isnan(oi_m) and oi_m > 0:
        score *= (1 + oi_m * 5)

    # 量放大
    if not np.isnan(vr) and vr > 1.0:
        score *= vr

    if candle_bull and vdp_v > 0 and (np.isnan(oi_m) or oi_m > 0):
        return (1, score)
    if candle_bear and vdp_v < 0 and (np.isnan(oi_m) or oi_m > 0):
        return (-1, score)
    return None


def make_signal_candle_vdp(si, di, C, O, H, L, V, OI,
                            oi_mom5, vol_ratio, vdp_arr, vdp_ema, atr_arr):
    """C. K线形态 + VDP: 锤子线/吞没 + VDP确认"""
    d = di - 1
    c = C[si, d]; o = O[si, d]; h = H[si, d]; l = L[si, d]
    vdp_v = vdp_ema[si, di] if vdp_ema is not None else np.nan

    if any(np.isnan(x) for x in [c, o, h, l]) or np.isnan(vdp_v): return None
    if h == l: return None

    hl = h - l
    body = abs(c - o)
    upper_shadow = h - max(c, o)
    lower_shadow = min(c, o) - l

    score = abs(vdp_v)

    # Bullish hammer: small body at top, long lower shadow
    if lower_shadow > body * 2 and lower_shadow > upper_shadow * 2 and vdp_v > 0:
        return (1, score * 2)

    # Bearish shooting star: small body at bottom, long upper shadow
    if upper_shadow > body * 2 and upper_shadow > lower_shadow * 2 and vdp_v < 0:
        return (-1, score * 2)

    # Bullish engulfing: today's body engulfs yesterday's
    if di >= 2:
        c_prev = C[si, d-1]; o_prev = O[si, d-1]
        if not np.isnan(c_prev) and not np.isnan(o_prev):
            if c > o and o_prev > c_prev and c > o_prev and o < c_prev and vdp_v > 0:
                return (1, score * 1.5)
            if c < o and o_prev < c_prev and c < o_prev and o > c_prev and vdp_v < 0:
                return (-1, score * 1.5)

    return None


def make_signal_oi_trend(si, di, C, O, H, L, V, OI,
                          oi_mom5, vol_ratio, vdp_arr, vdp_ema, atr_arr):
    """D. OI趋势延续: OI连续增加 + 价格趋势 + VDP确认"""
    d = di - 1
    c = C[si, d]; o = O[si, d]
    oi_m = oi_mom5[si, di] if oi_mom5 is not None else np.nan
    vdp_v = vdp_ema[si, di] if vdp_ema is not None else np.nan

    if np.isnan(c) or np.isnan(oi_m): return None

    # OI increasing significantly
    if oi_m < 0.01: return None

    # Price trend
    c5 = C[si, max(0, d-4):d+1]
    c5v = c5[~np.isnan(c5)]
    if len(c5v) < 3: return None
    price_trend = (c5v[-1] - c5v[0]) / c5v[0]

    score = oi_m * 10

    if price_trend > 0.01:  # Uptrend
        if np.isnan(vdp_v) or vdp_v > 0:  # VDP confirms (or no data)
            return (1, score * (1 + price_trend * 10))
    elif price_trend < -0.01:  # Downtrend
        if np.isnan(vdp_v) or vdp_v < 0:
            return (-1, score * (1 + abs(price_trend) * 10))

    return None


def make_signal_composite(si, di, C, O, H, L, V, OI,
                           oi_mom5, vol_ratio, vdp_arr, vdp_ema, atr_arr):
    """E. 综合评分: VDP 30% + OI 30% + K线 20% + 量 20%"""
    d = di - 1
    c = C[si, d]; o = O[si, d]
    vdp_v = vdp_ema[si, di] if vdp_ema is not None else np.nan
    oi_m = oi_mom5[si, di] if oi_mom5 is not None else np.nan
    vr = vol_ratio[si, di] if vol_ratio is not None else np.nan

    if np.isnan(c) or np.isnan(o): return None

    candle_dir = 1 if c > o else -1

    score = 0
    components = 0

    # VDP component (30%)
    if not np.isnan(vdp_v):
        vdp_dir = 1 if vdp_v > 0 else -1
        score += 0.30 * vdp_dir * min(abs(vdp_v) / 1e7, 1.0)
        components += 1

    # OI component (30%)
    if not np.isnan(oi_m):
        oi_dir = 1 if oi_m > 0 else -1
        score += 0.30 * oi_dir * min(abs(oi_m) * 5, 1.0)
        components += 1

    # Candle component (20%)
    body_ratio = abs(c - o) / max(H[si, d] - L[si, d], 0.01) if not np.isnan(H[si, d]) else 0
    score += 0.20 * candle_dir * body_ratio
    components += 1

    # Volume component (20%)
    if not np.isnan(vr):
        vol_dir = 1 if vr > 1.0 else -1
        score += 0.20 * vol_dir * min(abs(vr - 1), 1.0)
        components += 1

    if components < 3: return None
    if abs(score) < 0.15: return None  # Minimum threshold

    direction = 1 if score > 0 else -1
    return (direction, abs(score))


def make_signal_volume_pressure(si, di, C, O, H, L, V, OI,
                                 oi_mom5, vol_ratio, vdp_arr, vdp_ema, atr_arr):
    """F. 纯量压: Cumulative Delta方向 + 放量"""
    d = di - 1
    c = C[si, d]; o = O[si, d]; h = H[si, d]; l = L[si, d]
    v = V[si, d]
    vr = vol_ratio[si, di] if vol_ratio is not None else np.nan

    if any(np.isnan(x) for x in [c, o, h, l, v]): return None
    if h == l: return None

    # Volume Delta Pressure
    vdp_raw = v * (2 * c - h - l) / (h - l)

    # Need volume confirmation
    if not np.isnan(vr) and vr < 1.0: return None  # Below average volume

    score = abs(vdp_raw)
    if np.isnan(vr): score *= 0.5  # Penalize if no vol ratio

    if vdp_raw > 0:
        return (1, score)
    elif vdp_raw < 0:
        return (-1, score)
    return None


if __name__ == '__main__':
    print("=" * 80, flush=True)
    print("  Alpha Futures V12 — VDP+OI 日内策略 (无杠杆, 无Gap)", flush=True)
    print("=" * 80, flush=True)

    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)

    # Precompute factors
    print("\n  预计算因子...", flush=True)
    t0 = time.time()

    oi_mom5 = np.full((NS, ND), np.nan)
    vol_ratio = np.full((NS, ND), np.nan)
    vdp_arr = np.full((NS, ND), np.nan)
    vdp_ema = np.full((NS, ND), np.nan)
    atr_arr = np.full((NS, ND), np.nan)

    for si in range(NS):
        vdp_ema_val = 0
        for di in range(20, ND):
            d = di - 1
            # OI momentum
            oi_now = OI[si, d]
            if not np.isnan(oi_now) and oi_now > 0:
                oi5 = OI[si, max(0, d-4):d+1]
                oi5v = oi5[~np.isnan(oi5)]
                if len(oi5v) >= 3:
                    oi_mom5[si, di] = (oi_now - oi5v[0]) / oi5v[0]

            # Volume ratio
            v_now = V[si, d]
            if not np.isnan(v_now):
                v20 = V[si, max(0, d-19):d+1]
                v20v = v20[~np.isnan(v20)]
                if len(v20v) >= 10:
                    vol_ratio[si, di] = v_now / np.mean(v20v)

            # VDP
            hl = H[si, d] - L[si, d]
            if not np.isnan(hl) and hl > 0:
                c_d = C[si, d]; h_d = H[si, d]; l_d = L[si, d]; v_d = V[si, d]
                if not any(np.isnan([c_d, h_d, l_d, v_d])):
                    vdp_val = v_d * (2 * c_d - h_d - l_d) / hl
                    vdp_arr[si, di] = vdp_val
                    alpha = 2.0 / 15
                    vdp_ema_val = alpha * vdp_val + (1 - alpha) * vdp_ema_val
                    vdp_ema[si, di] = vdp_ema_val

            # ATR
            if di >= 11:
                trs = []
                for dd in range(max(1, d-9), d+1):
                    hi = H[si, dd]; lo = L[si, dd]; pc = C[si, dd-1]
                    if np.isnan(hi) or np.isnan(lo): continue
                    tr = hi - lo
                    if not np.isnan(pc): tr = max(tr, abs(hi-pc), abs(lo-pc))
                    trs.append(tr)
                if trs: atr_arr[si, di] = np.mean(trs)

    print(f"  因子完成 ({time.time()-t0:.0f}s)", flush=True)

    results = []

    strategies = [
        ('VDP_MOM', make_signal_vdp_momentum),
        ('VDP_OI', make_signal_vdp_oi),
        ('CANDLE', make_signal_candle_vdp),
        ('OI_TREND', make_signal_oi_trend),
        ('COMPOSITE', make_signal_composite),
        ('VOL_PRESS', make_signal_volume_pressure),
    ]

    for name, func in strategies:
        print(f"\n  [{name}]", flush=True)
        for mp in [1, 2, 3]:
            for slip in [0.0, 0.001, 0.002]:
                r = daily_backtest(NS, ND, dates, C, O, H, L, V, OI, syms,
                                    signal_func=func, max_pos=mp, slippage=slip,
                                    oi_mom5=oi_mom5, vol_ratio=vol_ratio,
                                    vdp_arr=vdp_arr, vdp_ema=vdp_ema, atr_arr=atr_arr)
                if r and r['ann'] > 10:
                    r['desc'] = name
                    r['mp'] = mp
                    r['slip'] = slip
                    results.append(r)
        print(f"    done ({len(results)})", flush=True)

    print(f"\n  完成 ({time.time()-t0:.0f}s, {len(results)} >10%)", flush=True)

    results.sort(key=lambda x: -x['ann'])
    print(f"\n{'='*80}", flush=True)
    print(f"  TOP 40", flush=True)
    print(f"  {'策略':<14s} {'MP':>2s} {'Slip':>6s} | {'Ann':>8s} {'N':>5s} {'WR':>5s} {'AvgW':>6s} {'AvgL':>5s} {'DD':>6s} {'Sharpe':>7s}", flush=True)
    for r in results[:40]:
        print(f"  {r['desc']:<14s} P{r['mp']:>1d} {r['slip']:.2%}  | "
              f"{r['ann']:+8.1f}% {r['n']:5d} {r['wr']:5.1f}% "
              f"{r['avg_win']:+6.2f}% {r['avg_loss']:5.2f}% {r['max_dd']:6.1f}% {r['sharpe']:7.2f}", flush=True)

    for i, r in enumerate(results[:5]):
        print(f"\n  #{i+1}: {r['desc']} P{r['mp']} slip={r['slip']:.2%} "
              f"(Ann={r['ann']:+.1f}%, DD={r['max_dd']:.1f}%, WR={r['wr']:.0f}%, Sharpe={r['sharpe']:.2f})", flush=True)
        for y in sorted(r.get('year_stats', {}).keys()):
            s = r['year_stats'][y]
            wr = s['wins'] / max(s['trades'], 1) * 100
            print(f"    {y}: {s['trades']:4d} t, WR={wr:.0f}%, pnl={s['total_pnl']:+.0f}%, abs={s['pnl_abs_sum']:+,.0f}", flush=True)

    # 按策略汇总
    print(f"\n  --- 按策略类型 ---", flush=True)
    seen = set()
    for r in results:
        if r['desc'] not in seen:
            seen.add(r['desc'])
            sub = [x for x in results if x['desc'] == r['desc']]
            best = sub[0]
            print(f"  {best['desc']:<14s}: Best={best['ann']:+.1f}% DD={best['max_dd']:.1f}% "
                  f"WR={best['wr']:.0f}% Sharpe={best['sharpe']:.2f}", flush=True)

    if results:
        print(f"\n  Best: {results[0]['ann']:+.1f}% DD={results[0]['max_dd']:.1f}% Sharpe={results[0]['sharpe']:.2f}", flush=True)
    print(f"  目标: 年化600%+ WR50%+ 无杠杆 无Gap", flush=True)
    print(f"{'='*80}", flush=True)
