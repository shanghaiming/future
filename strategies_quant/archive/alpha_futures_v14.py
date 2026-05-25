"""
Alpha Futures V14 — Daily Rotation Suite (无杠杆, 纯日线)
=========================================================
核心思路: 每天满仓轮动 (持有1天), 复利250次/年
数学: 600%年化 = (1+0.78%)^250, 需要0.78%日均收益

策略组:
  1. MOM3:     3日动量排名 — 趋势延续
  2. MOM5:     5日动量排名 — 中期趋势
  3. OI_SURGE: OI异动排名 — 资金涌入
  4. VDP_DIR:  VDP方向排名 — 买卖压力
  5. BODY_R:   K线实体占比 — 趋势强度
  6. COMPOSITE: 多因子加权组合
  7. CONFLUENCE: 信号共振 (3+信号一致)
  8. REVERSAL3: 3日反转 — 跌最多抄底
  9. OI_MOM:   OI+价格方向共振
  10. MOM_OI_VDP: 三因子融合

约束: P1集中 (全仓单品种), 1天持有, 不做gap, 不做日内, 无杠杆
"""
import sys, os, time, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alpha_v2 import load_all_data, MIN_TRAIN, CASH0

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
COMM = 0.0003  # 手续费


def run_daily_rotation(NS, ND, dates, C, O, H, L, V, OI, syms,
                       score_fn, name, allow_short=False, hold_days=1,
                       top_n=1, min_trades=50):
    """Generic daily rotation backtest.
    score_fn(si, di) -> float score. Higher = more bullish.
    If allow_short, also consider negative scores for shorting.
    """
    cash = float(CASH0)
    trades = []
    pos = None  # {'si', 'entry', 'entry_di', 'lots', 'dir', 'sym'}

    for di in range(MIN_TRAIN, ND - hold_days):
        year = dates[di].year

        # === EXIT: sell current position ===
        if pos is not None and di - pos['entry_di'] >= hold_days:
            c = C[pos['si'], di]
            if np.isnan(c) or c <= 0:
                c = pos['entry']
            mult = MULT.get(pos['sym'], DEF_MULT)
            pnl = (c - pos['entry']) * mult * pos['lots'] * pos['dir']
            cost_out = c * mult * pos['lots'] * COMM
            cash += c * mult * pos['lots'] - cost_out
            pnl_pct = pnl / (pos['entry'] * mult * pos['lots']) * 100 if pos['entry'] > 0 else 0
            trades.append({
                'pnl_pct': pnl_pct, 'pnl_abs': pnl,
                'days': di - pos['entry_di'], 'di': di,
                'year': year, 'sym': pos['sym'], 'dir': pos['dir']
            })
            pos = None

        # === ENTRY: find best candidate ===
        if pos is None:
            candidates = []
            for si in range(NS):
                score = score_fn(si, di)
                if np.isnan(score): continue

                c = C[si, di]
                if np.isnan(c) or c <= 0: continue

                # Long: positive score
                if score > 0:
                    candidates.append((si, score, 1, c))
                # Short: negative score (if allowed)
                if allow_short and score < 0:
                    candidates.append((si, -score, -1, c))

            if not candidates:
                continue

            # Pick top_n by absolute score
            candidates.sort(key=lambda x: -x[1])
            best_si, best_sc, best_dir, best_c = candidates[0]

            sym = syms[best_si]
            mult = MULT.get(sym, DEF_MULT)
            notional = best_c * mult
            if notional <= 0: continue

            lots = int(cash / notional)
            if lots <= 0: continue

            cost_in = notional * lots * (1 + COMM)
            if cost_in > cash: continue

            cash -= cost_in
            pos = {
                'si': best_si, 'entry': best_c, 'entry_di': di,
                'lots': lots, 'dir': best_dir, 'sym': sym
            }

    # Close remaining position
    if pos is not None:
        c = C[pos['si'], min(ND-1, pos['entry_di'] + hold_days)]
        if np.isnan(c) or c <= 0: c = pos['entry']
        mult = MULT.get(pos['sym'], DEF_MULT)
        pnl = (c - pos['entry']) * mult * pos['lots'] * pos['dir']
        cash += c * mult * pos['lots'] * (1 - COMM)
        trades.append({
            'pnl_pct': pnl / (pos['entry'] * mult * pos['lots']) * 100,
            'pnl_abs': pnl, 'days': hold_days,
            'di': ND-1, 'year': dates[ND-1].year,
            'sym': pos['sym'], 'dir': pos['dir']
        })

    if len(trades) < min_trades:
        return None

    # Compute stats
    equity = float(CASH0)
    peak = float(CASH0)
    max_dd = 0
    for t in sorted(trades, key=lambda x: x['di']):
        equity += t['pnl_abs']
        if equity > peak: peak = equity
        dd = (peak - equity) / peak * 100 if peak > 0 else 0
        if dd > max_dd: max_dd = dd

    days_total = (dates[ND-1] - dates[MIN_TRAIN]).days
    yr = max(days_total / 365.25, 0.01)
    ann = ((cash / CASH0) ** (1 / yr) - 1) * 100

    nw = sum(1 for t in trades if t['pnl_abs'] > 0)
    wr = nw / len(trades) * 100
    avg_pnl = np.mean([t['pnl_pct'] for t in trades])

    # Yearly breakdown
    year_stats = {}
    for t in trades:
        y = t['year']
        if y not in year_stats:
            year_stats[y] = {'n': 0, 'w': 0, 'pnl': 0.0}
        year_stats[y]['n'] += 1
        if t['pnl_abs'] > 0: year_stats[y]['w'] += 1
        year_stats[y]['pnl'] += t['pnl_pct']

    return {
        'name': name, 'ann': round(ann, 1), 'n': len(trades),
        'wr': round(wr, 1), 'dd': round(max_dd, 1),
        'avg_pnl': round(avg_pnl, 3),
        'final': round(cash, 0), 'cash0': CASH0,
        'years': year_stats
    }


if __name__ == '__main__':
    print("=" * 90, flush=True)
    print("  Alpha Futures V14 — Daily Rotation Suite (纯日线, 无杠杆, 1天持有)", flush=True)
    print("=" * 90, flush=True)

    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)
    print(f"  数据: {NS}品种, {ND}天", flush=True)

    # ============================================================
    # Pre-compute factors
    # ============================================================
    print("\n  预计算因子...", flush=True)
    t0 = time.time()

    # Price momentum: 3d, 5d, 10d
    mom3 = np.full((NS, ND), np.nan)
    mom5 = np.full((NS, ND), np.nan)
    mom10 = np.full((NS, ND), np.nan)

    # OI momentum: 3d, 5d
    oi_mom3 = np.full((NS, ND), np.nan)
    oi_mom5 = np.full((NS, ND), np.nan)

    # VDP EMA (15-day)
    vdp_ema = np.full((NS, ND), np.nan)

    # Body ratio: (C-O)/(H-L)
    body_r = np.full((NS, ND), np.nan)

    # Volume ratio: V / V_20
    vol_ratio = np.full((NS, ND), np.nan)

    # ATR 10-day (for volatility targeting)
    atr10 = np.full((NS, ND), np.nan)

    for si in range(NS):
        vdp_e = 0.0
        for di in range(20, ND):
            d = di - 1
            c_now = C[si, d]
            if np.isnan(c_now) or c_now <= 0:
                continue

            # Momentum
            for lag, arr in [(3, mom3), (5, mom5), (10, mom10)]:
                c_prev = C[si, max(0, d - lag)]
                if not np.isnan(c_prev) and c_prev > 0:
                    arr[si, di] = (c_now - c_prev) / c_prev

            # OI momentum
            oi_now = OI[si, d]
            if not np.isnan(oi_now) and oi_now > 0:
                for lag, arr in [(3, oi_mom3), (5, oi_mom5)]:
                    oi_prev = OI[si, max(0, d - lag)]
                    if not np.isnan(oi_prev) and oi_prev > 0:
                        arr[si, di] = (oi_now - oi_prev) / oi_prev

            # VDP
            hl = H[si, d] - L[si, d]
            if not np.isnan(hl) and hl > 0:
                cd = C[si, d]; hd = H[si, d]; ld = L[si, d]; vd = V[si, d]
                if not any(np.isnan([cd, hd, ld, vd])):
                    vdp_val = vd * (2*cd - hd - ld) / hl
                    alpha = 2.0 / 15
                    vdp_e = alpha * vdp_val + (1 - alpha) * vdp_e
                    vdp_ema[si, di] = vdp_e

            # Body ratio
            co = c_now - O[si, d]
            if not np.isnan(co) and hl > 0:
                body_r[si, di] = co / hl

            # Volume ratio
            v_now = V[si, d]
            if not np.isnan(v_now) and v_now > 0:
                v20 = V[si, max(0, d-19):d+1]
                v20v = v20[~np.isnan(v20)]
                if len(v20v) >= 10:
                    vol_ratio[si, di] = v_now / np.mean(v20v)

            # ATR
            trs = []
            for dd in range(max(1, d-9), d+1):
                hi = H[si, dd]; lo = L[si, dd]; pc = C[si, dd-1]
                if np.isnan(hi) or np.isnan(lo): continue
                tr = hi - lo
                if not np.isnan(pc):
                    tr = max(tr, abs(hi-pc), abs(lo-pc))
                trs.append(tr)
            if trs:
                atr10[si, di] = np.mean(trs)

    print(f"  因子完成 ({time.time()-t0:.0f}s)", flush=True)

    # ============================================================
    # Define scoring functions
    # ============================================================

    # 1. MOM3: 3-day momentum ranking
    def score_mom3(si, di):
        v = mom3[si, di]
        return v if not np.isnan(v) else np.nan

    # 2. MOM5: 5-day momentum ranking
    def score_mom5(si, di):
        v = mom5[si, di]
        return v if not np.isnan(v) else np.nan

    # 3. OI_SURGE: OI 5-day growth
    def score_oi_surge(si, di):
        v = oi_mom5[si, di]
        return v if not np.isnan(v) else np.nan

    # 4. VDP_DIR: VDP EMA direction
    def score_vdp(si, di):
        v = vdp_ema[si, di]
        return v * 1e-6 if not np.isnan(v) else np.nan

    # 5. BODY_R: Candle body ratio (trend conviction)
    def score_body(si, di):
        v = body_r[si, di]
        return v if not np.isnan(v) else np.nan

    # 6. COMPOSITE: Weighted combination
    def make_composite(w_mom=0.30, w_oi=0.30, w_vdp=0.25, w_body=0.15):
        def score(si, di):
            vals = []
            weights = []
            for arr, w in [(mom3, w_mom), (oi_mom5, w_oi), (vdp_ema, w_vdp), (body_r, w_body)]:
                v = arr[si, di]
                if not np.isnan(v):
                    # Normalize each factor to [-1, 1] range
                    if arr is vdp_ema:
                        v = np.sign(v) * min(abs(v) / 1e7, 1.0)
                    elif arr is oi_mom5:
                        v = np.clip(v * 5, -1, 1)
                    elif arr is mom3:
                        v = np.clip(v * 10, -1, 1)
                    # body_r already in [-1, 1]
                    vals.append(v)
                    weights.append(w)
            if not vals:
                return np.nan
            return sum(v * w for v, w in zip(vals, weights)) / sum(weights)
        return score

    # 7. CONFLUENCE: Count of agreeing signals
    def score_confluence(si, di):
        signals = []
        for arr in [mom3, oi_mom5, vdp_ema, body_r]:
            v = arr[si, di]
            if not np.isnan(v):
                if arr is vdp_ema:
                    signals.append(1 if v > 0 else -1)
                else:
                    signals.append(1 if v > 0 else -1)
        if not signals:
            return np.nan
        # Sum of signs: ranges from -4 to +4
        # Return 0 if not enough agreement
        total = sum(signals)
        if abs(total) < 2:  # Need at least 2/4 agreement
            return 0
        # Scale to [-1, 1]
        return total / len(signals)

    # 8. REVERSAL3: Buy biggest 3-day loser, sell biggest 3-day winner
    def score_reversal3(si, di):
        v = mom3[si, di]
        if np.isnan(v): return np.nan
        return -v  # Reverse: buy losers, sell winners

    # 9. OI_MOM: OI + price direction confluence
    def score_oi_mom(si, di):
        om = oi_mom5[si, di]
        m3 = mom3[si, di]
        if np.isnan(om) or np.isnan(m3):
            return np.nan
        # Both positive = bullish, both negative = bearish
        if om > 0 and m3 > 0: return (om + m3) / 2
        if om < 0 and m3 < 0: return (om + m3) / 2
        return 0  # Divergence = no signal

    # 10. MOM_OI_VDP: Three-factor fusion
    def score_mom_oi_vdp(si, di):
        m3 = mom3[si, di]
        om = oi_mom3[si, di]
        vd = vdp_ema[si, di]
        vals = []
        if not np.isnan(m3): vals.append(np.clip(m3 * 10, -1, 1))
        if not np.isnan(om): vals.append(np.clip(om * 5, -1, 1))
        if not np.isnan(vd): vals.append(np.sign(vd) * min(abs(vd) / 1e7, 1.0))
        if len(vals) < 2:
            return np.nan
        return np.mean(vals)

    # ============================================================
    # Run all strategies
    # ============================================================
    strategies = [
        ("MOM3_long", score_mom3, False, 1),
        ("MOM3_LS", score_mom3, True, 1),
        ("MOM5_long", score_mom5, False, 1),
        ("MOM5_LS", score_mom5, True, 1),
        ("OI_SURGE_long", score_oi_surge, False, 1),
        ("OI_SURGE_LS", score_oi_surge, True, 1),
        ("VDP_long", score_vdp, False, 1),
        ("VDP_LS", score_vdp, True, 1),
        ("BODY_long", score_body, False, 1),
        ("BODY_LS", score_body, True, 1),
        ("COMPOSITE_v1", make_composite(0.30, 0.30, 0.25, 0.15), True, 1),
        ("COMPOSITE_v2", make_composite(0.40, 0.30, 0.20, 0.10), True, 1),
        ("COMPOSITE_v3", make_composite(0.20, 0.40, 0.25, 0.15), True, 1),
        ("CONFLUENCE", score_confluence, True, 1),
        ("REVERSAL3_long", score_reversal3, False, 1),
        ("REVERSAL3_LS", score_reversal3, True, 1),
        ("OI_MOM_long", score_oi_mom, False, 1),
        ("OI_MOM_LS", score_oi_mom, True, 1),
        ("MOM_OI_VDP_long", score_mom_oi_vdp, False, 1),
        ("MOM_OI_VDP_LS", score_mom_oi_vdp, True, 1),
    ]

    # Also test 2-day hold for best strategies
    strategies_2d = [
        ("MOM3_LS_H2", score_mom3, True, 2),
        ("COMPOSITE_v1_H2", make_composite(0.30, 0.30, 0.25, 0.15), True, 2),
        ("CONFLUENCE_H2", score_confluence, True, 2),
        ("OI_MOM_LS_H2", score_oi_mom, True, 2),
        ("MOM_OI_VDP_LS_H2", score_mom_oi_vdp, True, 2),
    ]

    all_strategies = strategies + strategies_2d

    print(f"\n  运行 {len(all_strategies)} 个策略...", flush=True)
    results = []

    for name, score_fn, allow_short, hold in all_strategies:
        r = run_daily_rotation(NS, ND, dates, C, O, H, L, V, OI, syms,
                               score_fn, name, allow_short=allow_short,
                               hold_days=hold)
        if r is not None:
            results.append(r)
            tag = f"{'SHORT' if allow_short else 'LONG'}"
            print(f"  {name:25s} [{tag}] H{hold}d | Ann={r['ann']:+8.1f}%  "
                  f"WR={r['wr']:5.1f}%  N={r['n']:4d}  DD={r['dd']:5.1f}%  "
                  f"AvgPnl={r['avg_pnl']:+.3f}%", flush=True)
        else:
            print(f"  {name:25s} — too few trades", flush=True)

    # Sort by annual return
    results.sort(key=lambda x: -x['ann'])

    print(f"\n{'='*90}", flush=True)
    print(f"  RANKING (by Annual Return)", flush=True)
    print(f"{'='*90}", flush=True)
    print(f"  {'Strategy':25s} | {'Ann':>8s} {'WR':>5s} {'N':>5s} {'DD':>6s} {'AvgPnl':>7s} | {'Final':>12s}", flush=True)
    print(f"  {'-'*25}-+-{'-'*8}-{'-'*5}-{'-'*5}-{'-'*6}-{'-'*7}-+-{'-'*12}", flush=True)

    for r in results:
        print(f"  {r['name']:25s} | {r['ann']:+8.1f}% {r['wr']:5.1f}% {r['n']:5d} "
              f"{r['dd']:6.1f}% {r['avg_pnl']:+6.3f}% | {r['final']:>12,.0f}", flush=True)

    # Show yearly breakdown for top 5
    for i, r in enumerate(results[:5]):
        print(f"\n  #{i+1}: {r['name']} (Ann={r['ann']:+.1f}%, WR={r['wr']:.0f}%, DD={r['dd']:.1f}%)", flush=True)
        for y in sorted(r['years'].keys()):
            s = r['years'][y]
            wr = s['w'] / max(s['n'], 1) * 100
            print(f"    {y}: {s['n']:4d}t WR={wr:.0f}% pnl={s['pnl']:+.1f}%", flush=True)

    print(f"\n  目标: 年化600%+ WR50%+ 无杠杆 纯日线", flush=True)
    if results and results[0]['ann'] >= 600:
        print(f"  >>> TARGET ACHIEVED: {results[0]['name']} = {results[0]['ann']:+.1f}% <<<", flush=True)
    elif results:
        print(f"  Best: {results[0]['ann']:+.1f}% — gap to 600%: {600-results[0]['ann']:.0f}%", flush=True)
    print(f"{'='*90}", flush=True)
