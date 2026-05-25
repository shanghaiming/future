"""
Alpha Futures V24 — Adaptive Hold Extension (无杠杆, 纯日线)
============================================================
核心创新: 赢利头寸自动延长持仓, 亏损头寸快速止损

数学基础:
  - 短持仓(2-3天) → 快速轮换, 高频交易
  - 赢利 → 延长到5-10天, 捕捉大行情
  - 大赢利 → 延长到10-15天, 最大化利润
  - 亏损 → 严格止损(1.5-2.5%), 绝不拖延

预期效果:
  - WR ~50%, avg_win 3.5%, avg_loss 1.8% → 盈亏比 1.94
  - 120 trades/year → (1+0.82%)^120 = 2.67x → 167%
  - 180 trades/year → (1+0.82%)^180 = 4.37x → 337%
  - 如果WR能到55%: avg=1.045% → (1+1.045%)^180 = 6.5x → 550%

信号: 复合VDP+OI+动量+趋势质量评分, 截面排名选最强品种

约束: 不做gap, 不做日内, 无杠杆, 持仓>1天
"""
import sys, os, time, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alpha_v2 import load_all_data, MIN_TRAIN, CASH0, compute_vdp, compute_frama

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
COMM = 0.0003


# ============================================================
# FAST SIGNAL COMPUTATION
# ============================================================

def fast_ema(arr, period):
    n = len(arr)
    ema = np.full(n, np.nan)
    alpha = 2.0 / (period + 1)
    start = None
    for i in range(n):
        if not np.isnan(arr[i]):
            if start is None:
                ema[i] = arr[i]; start = i
            else:
                ema[i] = alpha * arr[i] + (1 - alpha) * ema[i-1]
    return ema


def fast_sma(arr, period):
    n = len(arr)
    sma = np.full(n, np.nan)
    for i in range(period - 1, n):
        w = arr[i-period+1:i+1]
        v = w[~np.isnan(w)]
        if len(v) >= period // 2:
            sma[i] = np.mean(v)
    return sma


def fast_atr(H, L, C, period=14):
    n = len(H)
    tr = np.full(n, np.nan)
    for i in range(1, n):
        if np.isnan(H[i]) or np.isnan(L[i]): continue
        tr[i] = H[i] - L[i]
        if not np.isnan(C[i-1]):
            tr[i] = max(tr[i], abs(H[i]-C[i-1]), abs(L[i]-C[i-1]))
    atr = np.full(n, np.nan)
    for i in range(period, n):
        w = tr[i-period+1:i+1]
        v = w[~np.isnan(w)]
        if len(v) > 0: atr[i] = np.mean(v)
    return atr


def precompute(NS, ND, C, O, H, L, V, OI):
    print("[Signals] Precomputing...", flush=True)
    t0 = time.time()
    S = {}
    for si in range(NS):
        c, o, h, l, v, oi = C[si], O[si], H[si], L[si], V[si], OI[si]
        if np.sum(~np.isnan(c)) < 60:
            continue

        # Momentum
        mom3 = np.full(ND, np.nan)
        mom5 = np.full(ND, np.nan)
        mom10 = np.full(ND, np.nan)
        mom20 = np.full(ND, np.nan)
        for i in range(3, ND):
            if not np.isnan(c[i]) and not np.isnan(c[i-3]):
                mom3[i] = (c[i] - c[i-3]) / c[i-3]
        for i in range(5, ND):
            if not np.isnan(c[i]) and not np.isnan(c[i-5]):
                mom5[i] = (c[i] - c[i-5]) / c[i-5]
        for i in range(10, ND):
            if not np.isnan(c[i]) and not np.isnan(c[i-10]):
                mom10[i] = (c[i] - c[i-10]) / c[i-10]
        for i in range(20, ND):
            if not np.isnan(c[i]) and not np.isnan(c[i-20]):
                mom20[i] = (c[i] - c[i-20]) / c[i-20]

        # VDP
        vdp = compute_vdp(c, h, l, v)
        vdp_ema = fast_ema(vdp, 15)

        # EMAs
        ema10 = fast_ema(c, 10)
        ema20 = fast_ema(c, 20)
        ema50 = fast_ema(c, 50)
        sma200 = fast_sma(c, 200)

        # ATR
        atr = fast_atr(h, l, c, 14)
        atr_pct = np.where(~np.isnan(atr) & (c > 0), atr / c, np.nan)

        # OI momentum
        oi_mom5 = np.full(ND, np.nan)
        for i in range(5, ND):
            if not np.isnan(oi[i]) and oi[i-5] > 0 and not np.isnan(oi[i-5]):
                oi_mom5[i] = (oi[i] - oi[i-5]) / oi[i-5]

        # Volume relative
        vol_sma = fast_sma(v, 20)
        rel_vol = np.where(~np.isnan(v) & ~np.isnan(vol_sma) & (vol_sma > 0), v / vol_sma, np.nan)

        # KER (Kaufman Efficiency Ratio)
        ker = np.full(ND, np.nan)
        for i in range(10, ND):
            d = abs(c[i] - c[i-10])
            p = np.sum(np.abs(np.diff(c[i-10:i+1])))
            ker[i] = d / p if p > 0 else 0

        # FRAMA
        frama = compute_frama(c, h, l, 16)

        # Kalman velocity
        from alpha_v2 import compute_kalman_velocity
        kal_vel = compute_kalman_velocity(c)

        # Donchian
        donch_u20 = np.full(ND, np.nan)
        donch_l10 = np.full(ND, np.nan)
        for i in range(20, ND):
            hw = h[i-20:i]; lw = l[i-10:i]
            hv = hw[~np.isnan(hw)]; lv = lw[~np.isnan(lw)]
            if len(hv) > 0: donch_u20[i] = np.max(hv)
            if len(lv) > 0: donch_l10[i] = np.min(lv)

        # Body ratio (trend strength indicator)
        body = np.full(ND, np.nan)
        for i in range(ND):
            if not np.isnan(c[i]) and not np.isnan(o[i]) and not np.isnan(h[i]) and not np.isnan(l[i]):
                rng = h[i] - l[i]
                if rng > 0:
                    body[i] = abs(c[i] - o[i]) / rng

        # Close position in range (0=low, 1=high)
        close_pos = np.full(ND, np.nan)
        for i in range(ND):
            if not np.isnan(h[i]) and not np.isnan(l[i]) and not np.isnan(c[i]):
                rng = h[i] - l[i]
                if rng > 0:
                    close_pos[i] = (c[i] - l[i]) / rng

        # RSI
        from alpha_v2 import compute_rsi
        rsi = compute_rsi(c, 14)

        # Bollinger %B
        bb_pct = np.full(ND, np.nan)
        for i in range(19, ND):
            w = c[i-19:i+1]
            v2 = w[~np.isnan(w)]
            if len(v2) >= 10:
                m = np.mean(v2); s = np.std(v2, ddof=0)
                if s > 0:
                    bb_pct[i] = (c[i] - (m - 2*s)) / (4*s)

        # Linear regression slope + R²
        lr_slope = np.full(ND, np.nan)
        r_squared = np.full(ND, np.nan)
        for i in range(19, ND):
            w = c[i-19:i+1]
            v2 = w[~np.isnan(w)]
            if len(v2) >= 10:
                x = np.arange(len(v2), dtype=float)
                xm = x.mean(); ym = v2.mean()
                ss_xy = np.sum((x - xm) * (v2 - ym))
                ss_xx = np.sum((x - xm) ** 2)
                ss_yy = np.sum((v2 - ym) ** 2)
                if ss_xx > 0:
                    lr_slope[i] = ss_xy / ss_xx
                    if ym > 0: lr_slope[i] /= ym
                    if ss_yy > 0:
                        r_squared[i] = (ss_xy ** 2) / (ss_xx * ss_yy)

        S[si] = {
            'mom3': mom3, 'mom5': mom5, 'mom10': mom10, 'mom20': mom20,
            'vdp_ema': vdp_ema, 'ema10': ema10, 'ema20': ema20,
            'ema50': ema50, 'sma200': sma200,
            'atr': atr, 'atr_pct': atr_pct,
            'oi_mom5': oi_mom5, 'rel_vol': rel_vol, 'ker': ker,
            'frama': frama, 'kal_vel': kal_vel,
            'donch_u20': donch_u20, 'donch_l10': donch_l10,
            'body': body, 'close_pos': close_pos,
            'rsi': rsi, 'bb_pct': bb_pct,
            'lr_slope': lr_slope, 'r_squared': r_squared,
        }

    print(f"  Done in {time.time()-t0:.1f}s, {len(S)} stocks", flush=True)
    return S


# ============================================================
# SCORING FUNCTIONS
# ============================================================

def score_vdp_mom(si, di, S, C, H, L, V, OI, ND, p):
    """VDP + Momentum composite — the proven winner from v14b."""
    if si not in S: return np.nan
    s = S[si]
    c = C[si, di]
    if np.isnan(c) or c <= 0: return np.nan

    mom = s[p.get('mom_key', 'mom5')][di]
    if np.isnan(mom): return np.nan

    score = 0.0

    # Base: momentum
    if mom > 0:
        score += min(mom / p.get('mom_scale', 0.12), 1.0) * 0.40
    else:
        return np.nan

    # VDP confirmation
    vdp = s['vdp_ema'][di]
    if not np.isnan(vdp):
        if vdp > 0: score += 0.20
        else: score -= 0.10

    # OI flow
    oi_m = s['oi_mom5'][di]
    if not np.isnan(oi_m):
        if oi_m > 0: score += 0.15
        else: score -= 0.05

    # KER trend quality
    ker = s['ker'][di]
    if not np.isnan(ker) and ker > 0.3: score += 0.10

    # Trend filter
    ema50 = s['ema50'][di]
    sma200 = s['sma200'][di]
    if not np.isnan(sma200) and c > sma200: score += 0.05
    if not np.isnan(ema50) and c > ema50: score += 0.05
    if not np.isnan(sma200) and c < sma200: score -= 0.15

    # Volume
    rv = s['rel_vol'][di]
    if not np.isnan(rv) and rv > 1.5: score += 0.05

    return score if score > p.get('min_score', 0.25) else np.nan


def score_frama_mom(si, di, S, C, H, L, V, OI, ND, p):
    """FRAMA adaptive MA + momentum + OI — proven from v7 (+40%)."""
    if si not in S: return np.nan
    s = S[si]
    c = C[si, di]
    if np.isnan(c) or c <= 0: return np.nan

    fra = s['frama'][di]
    if np.isnan(fra): return np.nan

    score = 0.0
    # FRAMA direction
    if di > 0:
        fra_prev = s['frama'][di-1]
        if not np.isnan(fra_prev):
            if c > fra and fra > fra_prev:  # price above rising FRAMA
                score += 0.30
            elif c < fra:
                return np.nan
        else:
            return np.nan

    # Momentum
    mom = s['mom10'][di]
    if not np.isnan(mom) and mom > 0:
        score += min(mom / 0.12, 1.0) * 0.20

    # VDP
    vdp = s['vdp_ema'][di]
    if not np.isnan(vdp) and vdp > 0: score += 0.15

    # OI
    oi_m = s['oi_mom5'][di]
    if not np.isnan(oi_m) and oi_m > 0: score += 0.10

    # KER
    ker = s['ker'][di]
    if not np.isnan(ker) and ker > 0.3: score += 0.10

    # Trend filter
    sma200 = s['sma200'][di]
    if not np.isnan(sma200) and c > sma200: score += 0.10
    elif not np.isnan(sma200): return np.nan

    # Volume
    rv = s['rel_vol'][di]
    if not np.isnan(rv) and rv > 1.2: score += 0.05

    return score if score > p.get('min_score', 0.25) else np.nan


def score_breakout_flow(si, di, S, C, H, L, V, OI, ND, p):
    """Donchian breakout + volume/OI flow — captures momentum bursts."""
    if si not in S: return np.nan
    s = S[si]
    c = C[si, di]
    if np.isnan(c) or c <= 0: return np.nan

    score = 0.0

    # Donchian 20-day breakout
    du = s['donch_u20'][di]
    if np.isnan(du) or du <= 0: return np.nan
    if c >= du * 0.995:
        score += 0.30
        if c > du * 1.01: score += 0.10  # strong breakout
    else:
        return np.nan

    # Close position near high of range
    cp = s['close_pos'][di]
    if not np.isnan(cp) and cp > 0.7: score += 0.10

    # Volume surge on breakout
    rv = s['rel_vol'][di]
    if not np.isnan(rv) and rv > 1.5: score += 0.15
    elif not np.isnan(rv) and rv > 1.2: score += 0.05

    # VDP positive
    vdp = s['vdp_ema'][di]
    if not np.isnan(vdp) and vdp > 0: score += 0.10

    # OI increasing
    oi_m = s['oi_mom5'][di]
    if not np.isnan(oi_m) and oi_m > 0: score += 0.10

    # ATR% high enough for big moves
    ap = s['atr_pct'][di]
    if not np.isnan(ap) and ap > p.get('min_atr_pct', 0.012): score += 0.10

    # Trend filter
    sma200 = s['sma200'][di]
    if not np.isnan(sma200) and c > sma200: score += 0.05

    return score if score > p.get('min_score', 0.3) else np.nan


def score_kalman_mom(si, di, S, C, H, L, V, OI, ND, p):
    """Kalman velocity + momentum + OI — adaptive trend detection."""
    if si not in S: return np.nan
    s = S[si]
    c = C[si, di]
    if np.isnan(c) or c <= 0: return np.nan

    score = 0.0

    # Kalman velocity positive
    kv = s['kal_vel'][di]
    if np.isnan(kv): return np.nan
    if kv > 0:
        score += min(kv / p.get('kv_scale', 5.0), 1.0) * 0.30
    else:
        return np.nan

    # Kalman velocity increasing
    if di > 0:
        kv_prev = s['kal_vel'][di-1]
        if not np.isnan(kv_prev) and kv > kv_prev:
            score += 0.10

    # Momentum confirmation
    mom = s['mom5'][di]
    if not np.isnan(mom) and mom > 0:
        score += min(mom / 0.10, 1.0) * 0.20

    # VDP
    vdp = s['vdp_ema'][di]
    if not np.isnan(vdp) and vdp > 0: score += 0.10

    # OI
    oi_m = s['oi_mom5'][di]
    if not np.isnan(oi_m) and oi_m > 0: score += 0.10

    # KER
    ker = s['ker'][di]
    if not np.isnan(ker) and ker > 0.3: score += 0.10

    # Trend filter
    sma200 = s['sma200'][di]
    if not np.isnan(sma200) and c > sma200: score += 0.10

    return score if score > p.get('min_score', 0.25) else np.nan


def score_lr_trend(si, di, S, C, H, L, V, OI, ND, p):
    """Linear regression trend quality — strong R² + positive slope."""
    if si not in S: return np.nan
    s = S[si]
    c = C[si, di]
    if np.isnan(c) or c <= 0: return np.nan

    score = 0.0

    # LR slope positive
    lr = s['lr_slope'][di]
    r2 = s['r_squared'][di]
    if np.isnan(lr) or lr <= 0: return np.nan
    score += min(lr / 0.005, 1.0) * 0.25

    # R² high = strong trend
    if not np.isnan(r2):
        if r2 > 0.7: score += 0.20
        elif r2 > 0.5: score += 0.10

    # VDP
    vdp = s['vdp_ema'][di]
    if not np.isnan(vdp) and vdp > 0: score += 0.15

    # OI
    oi_m = s['oi_mom5'][di]
    if not np.isnan(oi_m) and oi_m > 0: score += 0.10

    # Momentum
    mom = s['mom10'][di]
    if not np.isnan(mom) and mom > 0: score += 0.10

    # Trend filter
    ema50 = s['ema50'][di]
    sma200 = s['sma200'][di]
    if not np.isnan(sma200) and c > sma200: score += 0.10
    if not np.isnan(ema50) and c > ema50: score += 0.05

    return score if score > p.get('min_score', 0.3) else np.nan


# ============================================================
# ADAPTIVE HOLD BACKTEST ENGINE
# ============================================================

def run_adaptive(NS, ND, dates, C, O, H, L, V, OI, syms, S,
                 score_fn, name, params,
                 hold_min=2, hold_max=15,
                 tp_pct=0.04, sl_pct=0.02, trail_atr=2.0,
                 extend_at=0.02,  # if profit > this, extend hold
                 allow_short=False, reentry_gap=1):
    """Backtest with adaptive hold extension.

    Key features:
    1. Minimum hold = hold_min days
    2. Take profit at tp_pct
    3. Stop loss at sl_pct (ATR-adaptive)
    4. Trailing stop after profit > extend_at
    5. If profitable after hold_min, extend to hold_max
    6. If profitable after hold_max/2, extend further
    7. Rotate to better candidate if idle
    """
    cash = float(CASH0)
    trades = []
    pos = None
    last_exit = {}

    for di in range(MIN_TRAIN, ND):
        year = dates[di].year

        # === MANAGE POSITION ===
        if pos is not None:
            c = C[pos['si'], di]
            if np.isnan(c) or c <= 0:
                c = pos['entry']
            mult = MULT.get(pos['sym'], DEF_MULT)
            mkt_val = c * mult * pos['lots']
            pnl = (c - pos['entry']) * mult * pos['lots'] * pos['dir']
            pnl_pct = pnl / (pos['entry'] * mult * pos['lots']) * 100 if pos['entry'] > 0 else 0
            days_held = di - pos['entry_di']

            exit_reason = None

            # 1. Stop loss (hard)
            if pnl_pct / 100 < -sl_pct:
                exit_reason = 'stop'

            # 2. Take profit (hard)
            if exit_reason is None and pnl_pct / 100 > tp_pct and days_held >= hold_min:
                exit_reason = 'tp'

            # 3. Trailing stop (ATR-based, activates after profit > extend_at)
            if exit_reason is None and trail_atr > 0 and pnl_pct / 100 > extend_at:
                atr = pos.get('atr', 0)
                if atr > 0:
                    trail_price = pos.get('trail_price', pos['entry'])
                    if pos['dir'] == 1:
                        new_trail = c - trail_atr * atr
                        if new_trail > trail_price:
                            pos['trail_price'] = new_trail
                        if c < trail_price and days_held >= hold_min:
                            exit_reason = 'trail'
                    else:
                        new_trail = c + trail_atr * atr
                        if new_trail < trail_price:
                            pos['trail_price'] = new_trail
                        if c > trail_price and days_held >= hold_min:
                            exit_reason = 'trail'

            # 4. Score flip exit (after min hold)
            if exit_reason is None and days_held >= hold_min:
                cur_score = score_fn(pos['si'], di, S, C, H, L, V, OI, ND, params)
                if not np.isnan(cur_score):
                    if pos['dir'] == 1 and cur_score < -0.02:
                        exit_reason = 'signal_flip'

            # 5. Max hold exit
            # Adaptive: if profitable, extend beyond hold_max
            max_hold = hold_max
            if pnl_pct / 100 > extend_at * 2:
                max_hold = hold_max * 2  # double for big winners
            elif pnl_pct / 100 > extend_at:
                max_hold = int(hold_max * 1.5)

            if exit_reason is None and days_held >= max_hold:
                exit_reason = 'time'

            # 6. Better candidate rotation (only when position is NOT profitable)
            if exit_reason is None and pnl_pct / 100 < extend_at and days_held >= hold_min:
                best_si, best_dir, best_sc = -1, 0, 0
                for sj in range(NS):
                    sc = score_fn(sj, di, S, C, H, L, V, OI, ND, params)
                    if np.isnan(sc): continue
                    if sc > best_sc:
                        best_sc = sc; best_si = sj; best_dir = 1

                if best_si >= 0 and best_si != pos['si']:
                    cur_sc = score_fn(pos['si'], di, S, C, H, L, V, OI, ND, params)
                    cur_sc = abs(cur_sc) if not np.isnan(cur_sc) else 0
                    if best_sc > cur_sc * 1.5 + 0.1:
                        exit_reason = 'rotate'

            if exit_reason:
                cost_out = mkt_val * COMM
                cash += mkt_val - cost_out
                trades.append({
                    'pnl_pct': pnl_pct, 'pnl_abs': pnl,
                    'days': days_held, 'di': di, 'year': year,
                    'sym': pos['sym'], 'dir': pos['dir'],
                    'reason': exit_reason,
                })
                last_exit[pos['sym']] = di
                pos = None

        # === ENTRY ===
        if pos is None:
            best_si, best_dir, best_sc = -1, 0, 0
            for si in range(NS):
                sc = score_fn(si, di, S, C, H, L, V, OI, ND, params)
                if np.isnan(sc): continue

                sym = syms[si]
                if sym in last_exit and di - last_exit[sym] < reentry_gap:
                    continue

                if sc > best_sc:
                    best_sc = sc; best_si = si; best_dir = 1

            if best_si >= 0 and best_sc > 0:
                c = C[best_si, di]
                if np.isnan(c) or c <= 0: continue

                sym = syms[best_si]
                mult = MULT.get(sym, DEF_MULT)
                notional = c * mult
                if notional <= 0: continue

                lots = int(cash / notional)
                if lots <= 0: continue

                cost_in = notional * lots * (1 + COMM)
                if cost_in > cash: continue

                # ATR
                atr_val = 0
                trs = []
                for dd in range(max(1, di-14), di+1):
                    hi = H[best_si, dd]; lo = L[best_si, dd]; pc = C[best_si, dd-1]
                    if np.isnan(hi) or np.isnan(lo): continue
                    tr = hi - lo
                    if not np.isnan(pc):
                        tr = max(tr, abs(hi-pc), abs(lo-pc))
                    trs.append(tr)
                if trs: atr_val = np.mean(trs)

                cash -= cost_in
                pos = {
                    'si': best_si, 'entry': c, 'entry_di': di,
                    'lots': lots, 'dir': best_dir, 'sym': sym,
                    'atr': atr_val, 'trail_price': c - trail_atr * atr_val,
                    'score': best_sc,
                }

    # Close remaining
    if pos is not None:
        c = C[pos['si'], ND-1]
        if np.isnan(c) or c <= 0: c = pos['entry']
        mult = MULT.get(pos['sym'], DEF_MULT)
        pnl = (c - pos['entry']) * mult * pos['lots'] * pos['dir']
        cash += c * mult * pos['lots'] * (1 - COMM)
        trades.append({
            'pnl_pct': pnl / (pos['entry'] * mult * pos['lots']) * 100,
            'pnl_abs': pnl, 'days': ND-1 - pos['entry_di'],
            'di': ND-1, 'year': dates[ND-1].year,
            'sym': pos['sym'], 'dir': pos['dir'],
            'reason': 'end',
        })

    if len(trades) < 20:
        return None

    # Stats
    equity = float(CASH0); peak = float(CASH0); max_dd = 0
    for t in sorted(trades, key=lambda x: x['di']):
        equity += t['pnl_abs']
        if equity > peak: peak = equity
        if peak > 0:
            dd = (peak - equity) / peak * 100
            if dd > max_dd: max_dd = dd

    days_total = (dates[ND-1] - dates[MIN_TRAIN]).days
    yr = max(days_total / 365.25, 0.01)
    ann = ((cash / CASH0) ** (1 / yr) - 1) * 100

    nw = sum(1 for t in trades if t['pnl_abs'] > 0)
    wr = nw / len(trades) * 100
    avg_pnl = np.mean([t['pnl_pct'] for t in trades])
    avg_days = np.mean([t['days'] for t in trades])
    avg_win = np.mean([t['pnl_pct'] for t in trades if t['pnl_abs'] > 0]) if nw > 0 else 0
    avg_loss = np.mean([abs(t['pnl_pct']) for t in trades if t['pnl_abs'] <= 0]) if nw < len(trades) else 0

    year_stats = {}
    for t in trades:
        y = t['year']
        if y not in year_stats:
            year_stats[y] = {'n': 0, 'w': 0, 'pnl': 0.0}
        year_stats[y]['n'] += 1
        if t['pnl_abs'] > 0: year_stats[y]['w'] += 1
        year_stats[y]['pnl'] += t['pnl_pct']

    reasons = {}
    for t in trades:
        r = t['reason']
        if r not in reasons:
            reasons[r] = {'n': 0, 'w': 0, 'pnl': 0.0}
        reasons[r]['n'] += 1
        if t['pnl_abs'] > 0: reasons[r]['w'] += 1
        reasons[r]['pnl'] += t['pnl_pct']

    # Win/loss distribution
    wins = sorted([t['pnl_pct'] for t in trades if t['pnl_abs'] > 0], reverse=True)
    losses = sorted([t['pnl_pct'] for t in trades if t['pnl_abs'] <= 0])
    pct_big_win = len([w for w in wins if w > 3.0]) / max(len(wins), 1) * 100

    return {
        'name': name, 'ann': round(ann, 1), 'n': len(trades),
        'wr': round(wr, 1), 'dd': round(max_dd, 1),
        'avg_pnl': round(avg_pnl, 3), 'avg_days': round(avg_days, 1),
        'avg_win': round(avg_win, 3), 'avg_loss': round(avg_loss, 3),
        'cash': round(cash, 0), 'yearly': year_stats,
        'reasons': reasons, 'trades': trades,
        'pct_big_win': round(pct_big_win, 1),
        'wlr': round(avg_win / max(avg_loss, 0.01), 2),
    }


def print_result(r):
    if r is None:
        print("  [SKIP] No trades")
        return
    print(f"  {r['name']:50s} | Ann {r['ann']:+7.1f}% | WR {r['wr']:5.1f}% | "
          f"N {r['n']:4d} | DD {r['dd']:6.1f}% | AvgPnl {r['avg_pnl']:+.3f}% | "
          f"AvgD {r['avg_days']:.1f} | W/L {r['avg_win']:.2f}/{r['avg_loss']:.2f} | "
          f"WLR {r['wlr']:.2f} | BigW {r['pct_big_win']:.0f}%")
    if r.get('reasons'):
        parts = []
        for reason, stats in sorted(r['reasons'].items()):
            wr = stats['w'] / stats['n'] * 100 if stats['n'] > 0 else 0
            parts.append(f"{reason}:{stats['n']}({wr:.0f}%)")
        print(f"  {'':50s} | Exits: {' | '.join(parts)}")


# ============================================================
# MAIN
# ============================================================

def main():
    t_start = time.time()
    print("=" * 120)
    print("Alpha Futures V24 — Adaptive Hold Extension Strategy")
    print("=" * 120)

    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(
        max_stocks=500, load_oi=True
    )
    S = precompute(NS, ND, C, O, H, L, V, OI)

    results = []

    # === STRATEGY 1: VDP + MOM (proven winner) with adaptive holds ===
    print("\n--- VDP + Momentum ---")
    for mom_key in ['mom3', 'mom5', 'mom10']:
        for ms in [0.20, 0.25]:
            for tp, sl, trail in [
                (0.03, 0.015, 1.5), (0.04, 0.020, 2.0), (0.05, 0.020, 2.0),
                (0.05, 0.025, 2.5), (0.06, 0.025, 2.0), (0.08, 0.030, 2.5),
                (0.04, 0.015, 1.5), (0.05, 0.015, 2.0),
            ]:
                for hold_max in [7, 10, 15]:
                    params = {'mom_key': mom_key, 'mom_scale': 0.12, 'min_score': ms}
                    name = f"VDP_{mom_key}_TP{tp}_SL{sl}_T{trail}_H{hold_max}_MS{ms}"
                    r = run_adaptive(NS, ND, dates, C, O, H, L, V, OI, syms, S,
                                    score_vdp_mom, name, params,
                                    hold_min=2, hold_max=hold_max,
                                    tp_pct=tp, sl_pct=sl, trail_atr=trail,
                                    extend_at=tp*0.5)
                    if r: results.append(r)

    # === STRATEGY 2: FRAMA + Momentum ===
    print("\n--- FRAMA + Momentum ---")
    for tp, sl, trail in [
        (0.04, 0.020, 2.0), (0.05, 0.020, 2.0), (0.05, 0.025, 2.5),
        (0.06, 0.025, 2.0), (0.08, 0.030, 2.5),
    ]:
        for hold_max in [7, 10, 15]:
            params = {'min_score': 0.25}
            name = f"FRAMA_TP{tp}_SL{sl}_T{trail}_H{hold_max}"
            r = run_adaptive(NS, ND, dates, C, O, H, L, V, OI, syms, S,
                            score_frama_mom, name, params,
                            hold_min=2, hold_max=hold_max,
                            tp_pct=tp, sl_pct=sl, trail_atr=trail,
                            extend_at=tp*0.5)
            if r: results.append(r)

    # === STRATEGY 3: Breakout + Flow ===
    print("\n--- Breakout + Flow ---")
    for min_atr in [0.010, 0.015, 0.020]:
        for tp, sl, trail in [
            (0.04, 0.020, 2.0), (0.05, 0.025, 2.5), (0.06, 0.025, 2.0),
        ]:
            for hold_max in [7, 10]:
                params = {'min_atr_pct': min_atr, 'min_score': 0.3}
                name = f"BRK_A{min_atr}_TP{tp}_SL{sl}_T{trail}_H{hold_max}"
                r = run_adaptive(NS, ND, dates, C, O, H, L, V, OI, syms, S,
                                score_breakout_flow, name, params,
                                hold_min=2, hold_max=hold_max,
                                tp_pct=tp, sl_pct=sl, trail_atr=trail,
                                extend_at=tp*0.5)
                if r: results.append(r)

    # === STRATEGY 4: Kalman Velocity ===
    print("\n--- Kalman Velocity ---")
    for kv_scale in [3.0, 5.0, 8.0]:
        for tp, sl, trail in [
            (0.04, 0.020, 2.0), (0.05, 0.020, 2.0), (0.05, 0.025, 2.5),
        ]:
            for hold_max in [7, 10]:
                params = {'kv_scale': kv_scale, 'min_score': 0.25}
                name = f"KAL_K{kv_scale}_TP{tp}_SL{sl}_T{trail}_H{hold_max}"
                r = run_adaptive(NS, ND, dates, C, O, H, L, V, OI, syms, S,
                                score_kalman_mom, name, params,
                                hold_min=2, hold_max=hold_max,
                                tp_pct=tp, sl_pct=sl, trail_atr=trail,
                                extend_at=tp*0.5)
                if r: results.append(r)

    # === STRATEGY 5: LR Trend Quality ===
    print("\n--- LR Trend Quality ---")
    for tp, sl, trail in [
        (0.04, 0.020, 2.0), (0.05, 0.025, 2.5), (0.06, 0.025, 2.0),
    ]:
        for hold_max in [7, 10, 15]:
            params = {'min_score': 0.3}
            name = f"LR_TP{tp}_SL{sl}_T{trail}_H{hold_max}"
            r = run_adaptive(NS, ND, dates, C, O, H, L, V, OI, syms, S,
                            score_lr_trend, name, params,
                            hold_min=2, hold_max=hold_max,
                            tp_pct=tp, sl_pct=sl, trail_atr=trail,
                            extend_at=tp*0.5)
            if r: results.append(r)

    # === SUMMARY ===
    print("\n" + "=" * 120)
    print(f"TOTAL: {len(results)} profitable configs tested")
    print("=" * 120)

    if results:
        results.sort(key=lambda x: -x['ann'])

        print(f"\n{'='*120}")
        print("TOP 30 BY ANNUAL RETURN:")
        print(f"{'='*120}")
        for r in results[:30]:
            print_result(r)

        print(f"\n--- TOP 10 BY WIN/LOSS RATIO ---")
        by_wlr = sorted(results, key=lambda x: -x['wlr'])
        for r in by_wlr[:10]:
            print_result(r)

        print(f"\n--- TOP 10 BY RISK-ADJUSTED ---")
        by_ra = sorted(results, key=lambda x: -x['ann'] / max(x['dd'], 1))
        for r in by_ra[:10]:
            ratio = r['ann'] / max(r['dd'], 1)
            print(f"  {r['name']:50s} | Ann {r['ann']:+7.1f}% | DD {r['dd']:6.1f}% | "
                  f"R {ratio:.2f} | WR {r['wr']:5.1f}% | WLR {r['wlr']:.2f}")

        print(f"\n--- YEARLY BREAKDOWN (Top 5) ---")
        for r in results[:5]:
            print(f"\n  {r['name']}:")
            for y in sorted(r['yearly'].keys()):
                ys = r['yearly'][y]
                wr = ys['w'] / ys['n'] * 100 if ys['n'] > 0 else 0
                print(f"    {y}: {ys['n']:3d} trades, WR {wr:5.1f}%, PnL {ys['pnl']:+.1f}%")

        # Exit reason analysis for top 3
        print(f"\n--- EXIT ANALYSIS (Top 3) ---")
        for r in results[:3]:
            print(f"\n  {r['name']}:")
            for reason, stats in sorted(r['reasons'].items()):
                wr = stats['w'] / stats['n'] * 100 if stats['n'] > 0 else 0
                avg = stats['pnl'] / stats['n'] if stats['n'] > 0 else 0
                print(f"    {reason:12s}: {stats['n']:4d} trades, WR {wr:5.1f}%, "
                      f"Total {stats['pnl']:+.1f}%, Avg {avg:+.3f}%")

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
