"""
Alpha Futures V25 — Fast Rotation 2-Day (无杠杆, 纯日线)
========================================================
核心: 最少持仓2天(满足用户要求), 但每日轮换到最强信号
无硬止损, 只用trailing stop和score flip退出

v14 1天: +88.7% WR 47.1% (用户说太短)
v14b 3天: +73.0% WR 49.4%
v25目标: 2天最少持仓, 期望接近v14的水平

数学: 250天/2天 = 125次轮换/年, 需要2.3%/次 = 600%
      如果WR=50%, avg_win=4.6%, avg_loss=0% → 需要完美择时
      实际: WR 50%, avg_win 3%, avg_loss 2% → net 0.5%/次 = 87%

关键优化:
  1. 每日排名68品种, 持仓2天后可轮换到更强品种
  2. 无硬止损 → 用trailing stop保护盈利
  3. 无TP → 让盈利持续奔跑
  4. Score flip → 信号翻转时立即退出
  5. 多种评分函数取最佳

约束: 不做gap, 不做日内, 无杠杆, 持仓>1天
"""
import sys, os, time, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alpha_v2 import load_all_data, MIN_TRAIN, CASH0, compute_vdp, compute_frama, compute_kalman_velocity

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


def precompute(NS, ND, C, O, H, L, V, OI):
    print("[Signals] Precomputing...", flush=True)
    t0 = time.time()
    S = {}
    for si in range(NS):
        c, o, h, l, v, oi = C[si], O[si], H[si], L[si], V[si], OI[si]
        if np.sum(~np.isnan(c)) < 60: continue

        # Momentums
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

        # VDP + EMA
        vdp = compute_vdp(c, h, l, v)
        vdp_ema = np.full(ND, np.nan)
        vdp_ema[0] = vdp[0]
        a = 2.0 / 16
        for i in range(1, ND):
            if not np.isnan(vdp[i]):
                vdp_ema[i] = a * vdp[i] + (1-a) * (vdp_ema[i-1] if not np.isnan(vdp_ema[i-1]) else vdp[i])

        # EMAs
        def ema(arr, period):
            e = np.full(ND, np.nan)
            al = 2.0 / (period + 1)
            start = None
            for i in range(ND):
                if not np.isnan(arr[i]):
                    if start is None: e[i] = arr[i]; start = i
                    else: e[i] = al * arr[i] + (1-al) * e[i-1]
            return e

        ema10 = ema(c, 10)
        ema20 = ema(c, 20)
        ema50 = ema(c, 50)
        sma200 = np.full(ND, np.nan)
        for i in range(199, ND):
            w = c[i-199:i+1]; v2 = w[~np.isnan(w)]
            if len(v2) >= 100: sma200[i] = np.mean(v2)

        # ATR
        atr = np.full(ND, np.nan)
        for i in range(14, ND):
            trs = []
            for dd in range(i-13, i+1):
                if np.isnan(h[dd]) or np.isnan(l[dd]): continue
                tr = h[dd] - l[dd]
                if dd > 0 and not np.isnan(c[dd-1]):
                    tr = max(tr, abs(h[dd]-c[dd-1]), abs(l[dd]-c[dd-1]))
                trs.append(tr)
            if trs: atr[i] = np.mean(trs)

        # OI momentum
        oi_mom5 = np.full(ND, np.nan)
        for i in range(5, ND):
            if not np.isnan(oi[i]) and oi[i-5] > 0 and not np.isnan(oi[i-5]):
                oi_mom5[i] = (oi[i] - oi[i-5]) / oi[i-5]

        # Volume relative
        vol_sma = np.full(ND, np.nan)
        for i in range(19, ND):
            w = v[i-19:i+1]; v2 = w[~np.isnan(w)]
            if len(v2) >= 10: vol_sma[i] = np.mean(v2)
        rel_vol = np.where(~np.isnan(v) & ~np.isnan(vol_sma) & (vol_sma > 0), v / vol_sma, np.nan)

        # KER
        ker = np.full(ND, np.nan)
        for i in range(10, ND):
            d = abs(c[i] - c[i-10])
            p = np.sum(np.abs(np.diff(c[i-10:i+1])))
            ker[i] = d / p if p > 0 else 0

        # FRAMA
        frama = compute_frama(c, h, l, 16)

        # Kalman velocity
        kal_vel = compute_kalman_velocity(c)

        # Linreg slope + R²
        lr_slope = np.full(ND, np.nan)
        r_sq = np.full(ND, np.nan)
        for i in range(19, ND):
            w = c[i-19:i+1]; v2 = w[~np.isnan(w)]
            if len(v2) >= 10:
                x = np.arange(len(v2), dtype=float)
                xm, ym = x.mean(), v2.mean()
                sxy = np.sum((x-xm)*(v2-ym)); sxx = np.sum((x-xm)**2); syy = np.sum((v2-ym)**2)
                if sxx > 0:
                    lr_slope[i] = sxy/sxx / ym if ym > 0 else 0
                    if syy > 0: r_sq[i] = sxy**2 / (sxx*syy)

        S[si] = {
            'mom3': mom3, 'mom5': mom5, 'mom10': mom10, 'mom20': mom20,
            'vdp_ema': vdp_ema, 'ema10': ema10, 'ema20': ema20,
            'ema50': ema50, 'sma200': sma200,
            'atr': atr, 'oi_mom5': oi_mom5, 'rel_vol': rel_vol,
            'ker': ker, 'frama': frama, 'kal_vel': kal_vel,
            'lr_slope': lr_slope, 'r_sq': r_sq,
        }

    print(f"  Done in {time.time()-t0:.1f}s, {len(S)} stocks", flush=True)
    return S


# ============================================================
# SCORING FUNCTIONS
# ============================================================

def score_vdp_mom(si, di, S, C, H, L, V, OI, ND, p):
    """VDP + Momentum — proven winner."""
    if si not in S: return np.nan
    s = S[si]; c = C[si, di]
    if np.isnan(c) or c <= 0: return np.nan

    mk = p.get('mom_key', 'mom5')
    mom = s[mk][di]
    if np.isnan(mom): return np.nan

    score = 0.0
    if mom > 0:
        score += min(mom / p.get('mom_scale', 0.12), 1.0) * 0.40
    else:
        return np.nan

    vdp = s['vdp_ema'][di]
    if not np.isnan(vdp):
        if vdp > 0: score += 0.20
        else: score -= 0.10

    oi_m = s['oi_mom5'][di]
    if not np.isnan(oi_m):
        if oi_m > 0: score += 0.15
        else: score -= 0.05

    ker = s['ker'][di]
    if not np.isnan(ker) and ker > 0.3: score += 0.10

    ema50 = s['ema50'][di]
    sma200 = s['sma200'][di]
    if not np.isnan(sma200) and c > sma200: score += 0.05
    if not np.isnan(ema50) and c > ema50: score += 0.05
    if not np.isnan(sma200) and c < sma200: score -= 0.15

    rv = s['rel_vol'][di]
    if not np.isnan(rv) and rv > 1.5: score += 0.05

    return score if score > p.get('min_score', 0.15) else np.nan


def score_mom_only(si, di, S, C, H, L, V, OI, ND, p):
    """Pure momentum ranking — simplest signal."""
    if si not in S: return np.nan
    s = S[si]; c = C[si, di]
    if np.isnan(c) or c <= 0: return np.nan

    mk = p.get('mom_key', 'mom5')
    mom = s[mk][di]
    if np.isnan(mom) or mom <= 0: return np.nan

    # Just return momentum as score
    return mom


def score_mom_oi(si, di, S, C, H, L, V, OI, ND, p):
    """Momentum + OI confirmation."""
    if si not in S: return np.nan
    s = S[si]; c = C[si, di]
    if np.isnan(c) or c <= 0: return np.nan

    mk = p.get('mom_key', 'mom5')
    mom = s[mk][di]
    if np.isnan(mom) or mom <= 0: return np.nan

    score = mom * 5  # base momentum

    oi_m = s['oi_mom5'][di]
    if not np.isnan(oi_m) and oi_m > 0:
        score *= 1.5  # boost by 50% if OI increasing

    vdp = s['vdp_ema'][di]
    if not np.isnan(vdp) and vdp > 0:
        score *= 1.3  # boost by 30% if VDP positive

    ker = s['ker'][di]
    if not np.isnan(ker) and ker > 0.3:
        score *= 1.2  # boost by 20% if KER high

    return score


def score_composite_v25(si, di, S, C, H, L, V, OI, ND, p):
    """Composite score with weighted factors."""
    if si not in S: return np.nan
    s = S[si]; c = C[si, di]
    if np.isnan(c) or c <= 0: return np.nan

    score = 0.0

    # 1. Momentum (40% weight)
    mk = p.get('mom_key', 'mom5')
    mom = s[mk][di]
    if np.isnan(mom) or mom <= 0: return np.nan
    score += min(mom / 0.12, 1.0) * 0.40

    # 2. VDP (20%)
    vdp = s['vdp_ema'][di]
    if not np.isnan(vdp):
        if vdp > 0: score += 0.20
        else: score -= 0.10

    # 3. OI flow (15%)
    oi_m = s['oi_mom5'][di]
    if not np.isnan(oi_m):
        if oi_m > 0: score += 0.15
        else: score -= 0.05

    # 4. FRAMA trend (10%)
    fra = s['frama'][di]
    if not np.isnan(fra) and di > 0:
        fra_prev = s['frama'][di-1]
        if not np.isnan(fra_prev):
            if c > fra and fra > fra_prev: score += 0.10
            elif c < fra: score -= 0.10

    # 5. Kalman velocity (10%)
    kv = s['kal_vel'][di]
    if not np.isnan(kv):
        if kv > 0: score += 0.05
        else: score -= 0.05

    # 6. Trend filter (5%)
    sma200 = s['sma200'][di]
    if not np.isnan(sma200):
        if c > sma200: score += 0.05
        else: score -= 0.15

    return score if score > p.get('min_score', 0.20) else np.nan


def score_kalman_mom(si, di, S, C, H, L, V, OI, ND, p):
    """Kalman velocity + momentum."""
    if si not in S: return np.nan
    s = S[si]; c = C[si, di]
    if np.isnan(c) or c <= 0: return np.nan

    kv = s['kal_vel'][di]
    if np.isnan(kv) or kv <= 0: return np.nan

    score = min(kv / 5.0, 1.0) * 0.40

    mom = s['mom5'][di]
    if not np.isnan(mom) and mom > 0:
        score += min(mom / 0.10, 1.0) * 0.25

    vdp = s['vdp_ema'][di]
    if not np.isnan(vdp) and vdp > 0: score += 0.15

    oi_m = s['oi_mom5'][di]
    if not np.isnan(oi_m) and oi_m > 0: score += 0.10

    ker = s['ker'][di]
    if not np.isnan(ker) and ker > 0.3: score += 0.10

    return score if score > 0.20 else np.nan


def score_lr_mom(si, di, S, C, H, L, V, OI, ND, p):
    """Linear regression slope + momentum."""
    if si not in S: return np.nan
    s = S[si]; c = C[si, di]
    if np.isnan(c) or c <= 0: return np.nan

    lr = s['lr_slope'][di]
    r2 = s['r_sq'][di]
    if np.isnan(lr) or lr <= 0: return np.nan

    score = min(lr / 0.005, 1.0) * 0.30

    if not np.isnan(r2) and r2 > 0.5: score += 0.15

    mom = s['mom5'][di]
    if not np.isnan(mom) and mom > 0:
        score += min(mom / 0.10, 1.0) * 0.20

    vdp = s['vdp_ema'][di]
    if not np.isnan(vdp) and vdp > 0: score += 0.15

    oi_m = s['oi_mom5'][di]
    if not np.isnan(oi_m) and oi_m > 0: score += 0.10

    sma200 = s['sma200'][di]
    if not np.isnan(sma200) and c > sma200: score += 0.10

    return score if score > 0.25 else np.nan


# ============================================================
# FAST ROTATION BACKTEST ENGINE
# ============================================================

def run_rotation(NS, ND, dates, C, O, H, L, V, OI, syms, S,
                 score_fn, name, params,
                 hold_min=2, hold_max=10, trail_atr=2.0,
                 use_stop=False, stop_pct=0.05,
                 reentry_gap=0, rotate_boost=0.05):
    """Fast rotation backtest.

    Key design:
    - hold_min: minimum days before rotation allowed
    - hold_max: maximum days, then forced exit
    - trail_atr: trailing stop (only for winning positions)
    - use_stop: whether to use hard stop loss
    - No TP — let winners run via trailing stop
    - Score flip exit after hold_min
    - Rotation to stronger signal after hold_min
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
            if np.isnan(c) or c <= 0: c = pos['entry']
            mult = MULT.get(pos['sym'], DEF_MULT)
            mkt_val = c * mult * pos['lots']
            pnl = (c - pos['entry']) * mult * pos['lots'] * pos['dir']
            pnl_pct = pnl / (pos['entry'] * mult * pos['lots']) * 100 if pos['entry'] > 0 else 0
            days_held = di - pos['entry_di']

            exit_reason = None

            # 1. Hard stop (optional)
            if use_stop and pnl_pct / 100 < -stop_pct:
                exit_reason = 'stop'

            # 2. Trailing stop (ATR-based, for winning positions only)
            if exit_reason is None and trail_atr > 0 and pnl_pct > 0:
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

            # 3. Score flip (after min hold)
            if exit_reason is None and days_held >= hold_min:
                cur_score = score_fn(pos['si'], di, S, C, H, L, V, OI, ND, params)
                if np.isnan(cur_score):
                    exit_reason = 'signal_gone'
                elif cur_score < -0.02:
                    exit_reason = 'signal_flip'

            # 4. Max hold
            if exit_reason is None and days_held >= hold_max:
                exit_reason = 'time'

            # 5. Rotation to better candidate (after min hold)
            if exit_reason is None and days_held >= hold_min:
                best_si, best_sc = -1, 0
                for sj in range(NS):
                    sc = score_fn(sj, di, S, C, H, L, V, OI, ND, params)
                    if np.isnan(sc): continue
                    if sc > best_sc:
                        best_sc = sc; best_si = sj

                if best_si >= 0 and best_si != pos['si']:
                    cur_sc = score_fn(pos['si'], di, S, C, H, L, V, OI, ND, params)
                    cur_sc = abs(cur_sc) if not np.isnan(cur_sc) else 0
                    # Rotate if new candidate is significantly stronger
                    if best_sc > cur_sc + rotate_boost:
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
            best_si, best_sc = -1, 0
            for si in range(NS):
                sc = score_fn(si, di, S, C, H, L, V, OI, ND, params)
                if np.isnan(sc): continue

                sym = syms[si]
                if sym in last_exit and di - last_exit[sym] < reentry_gap:
                    continue

                if sc > best_sc:
                    best_sc = sc; best_si = si

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
                    if not np.isnan(pc): tr = max(tr, abs(hi-pc), abs(lo-pc))
                    trs.append(tr)
                if trs: atr_val = np.mean(trs)

                cash -= cost_in
                pos = {
                    'si': best_si, 'entry': c, 'entry_di': di,
                    'lots': lots, 'dir': 1, 'sym': sym,
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
            'sym': pos['sym'], 'dir': pos['dir'], 'reason': 'end',
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

    return {
        'name': name, 'ann': round(ann, 1), 'n': len(trades),
        'wr': round(wr, 1), 'dd': round(max_dd, 1),
        'avg_pnl': round(avg_pnl, 3), 'avg_days': round(avg_days, 1),
        'avg_win': round(avg_win, 3), 'avg_loss': round(avg_loss, 3),
        'cash': round(cash, 0), 'yearly': year_stats, 'reasons': reasons,
        'wlr': round(avg_win / max(avg_loss, 0.01), 2),
    }


def print_result(r):
    if r is None:
        print("  [SKIP]")
        return
    print(f"  {r['name']:55s} | Ann {r['ann']:+7.1f}% | WR {r['wr']:5.1f}% | "
          f"N {r['n']:4d} | DD {r['dd']:6.1f}% | AvgPnl {r['avg_pnl']:+.3f}% | "
          f"AvgD {r['avg_days']:.1f} | W/L {r['avg_win']:.2f}/{r['avg_loss']:.2f}")
    if r.get('reasons'):
        parts = []
        for reason, stats in sorted(r['reasons'].items()):
            wr = stats['w'] / stats['n'] * 100 if stats['n'] > 0 else 0
            parts.append(f"{reason}:{stats['n']}({wr:.0f}%)")
        print(f"  {'':55s} | Exits: {' | '.join(parts)}")


def main():
    t_start = time.time()
    print("=" * 130)
    print("Alpha Futures V25 — Fast Rotation 2-Day Hold")
    print("=" * 130)

    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(
        max_stocks=500, load_oi=True
    )
    S = precompute(NS, ND, C, O, H, L, V, OI)

    results = []

    # === Test all scoring functions × hold parameters ===
    configs = [
        # (score_fn, score_name, params_list, hold_combos)
    ]

    # 1. VDP_MOM with various momentum keys
    for mk in ['mom3', 'mom5', 'mom10']:
        for ms in [0.10, 0.15, 0.20]:
            for rb in [0.03, 0.05, 0.10]:
                for hm, hmax, trail in [(2, 5, 1.5), (2, 7, 2.0), (2, 10, 2.5),
                                         (2, 5, 2.0), (2, 7, 1.5), (2, 10, 2.0),
                                         (3, 5, 2.0), (3, 7, 2.5), (3, 10, 3.0)]:
                    p = {'mom_key': mk, 'mom_scale': 0.12, 'min_score': ms, 'rotate_boost': rb}
                    name = f"VDP_{mk}_MS{ms}_RB{rb}_H{hm}_{hmax}_T{trail}"
                    r = run_rotation(NS, ND, dates, C, O, H, L, V, OI, syms, S,
                                    score_vdp_mom, name, p,
                                    hold_min=hm, hold_max=hmax, trail_atr=trail)
                    if r: results.append(r)

    # 2. Pure momentum
    for mk in ['mom3', 'mom5', 'mom10', 'mom20']:
        for hm, hmax, trail in [(2, 5, 1.5), (2, 7, 2.0), (2, 10, 2.5),
                                 (3, 5, 2.0), (3, 7, 2.5)]:
            p = {'mom_key': mk}
            name = f"MOM_{mk}_H{hm}_{hmax}_T{trail}"
            r = run_rotation(NS, ND, dates, C, O, H, L, V, OI, syms, S,
                            score_mom_only, name, p,
                            hold_min=hm, hold_max=hmax, trail_atr=trail)
            if r: results.append(r)

    # 3. Mom + OI
    for mk in ['mom3', 'mom5', 'mom10']:
        for hm, hmax, trail in [(2, 5, 2.0), (2, 7, 2.5), (2, 10, 2.0),
                                 (3, 5, 2.0), (3, 7, 2.5)]:
            p = {'mom_key': mk}
            name = f"MOI_{mk}_H{hm}_{hmax}_T{trail}"
            r = run_rotation(NS, ND, dates, C, O, H, L, V, OI, syms, S,
                            score_mom_oi, name, p,
                            hold_min=hm, hold_max=hmax, trail_atr=trail)
            if r: results.append(r)

    # 4. Composite
    for mk in ['mom3', 'mom5', 'mom10']:
        for ms in [0.15, 0.20, 0.25]:
            for hm, hmax, trail in [(2, 5, 2.0), (2, 7, 2.5), (2, 10, 2.0),
                                     (3, 5, 2.0), (3, 7, 2.5)]:
                p = {'mom_key': mk, 'min_score': ms, 'rotate_boost': 0.05}
                name = f"COMP_{mk}_MS{ms}_H{hm}_{hmax}_T{trail}"
                r = run_rotation(NS, ND, dates, C, O, H, L, V, OI, syms, S,
                                score_composite_v25, name, p,
                                hold_min=hm, hold_max=hmax, trail_atr=trail)
                if r: results.append(r)

    # 5. Kalman + Mom
    for hm, hmax, trail in [(2, 5, 2.0), (2, 7, 2.5), (2, 10, 2.0),
                             (3, 5, 2.0), (3, 7, 2.5)]:
        name = f"KAL_H{hm}_{hmax}_T{trail}"
        r = run_rotation(NS, ND, dates, C, O, H, L, V, OI, syms, S,
                        score_kalman_mom, name, {},
                        hold_min=hm, hold_max=hmax, trail_atr=trail)
        if r: results.append(r)

    # 6. LR + Mom
    for hm, hmax, trail in [(2, 5, 2.0), (2, 7, 2.5), (2, 10, 2.0),
                             (3, 5, 2.0), (3, 7, 2.5)]:
        name = f"LR_H{hm}_{hmax}_T{trail}"
        r = run_rotation(NS, ND, dates, C, O, H, L, V, OI, syms, S,
                        score_lr_mom, name, {},
                        hold_min=hm, hold_max=hmax, trail_atr=trail)
        if r: results.append(r)

    # === SUMMARY ===
    print(f"\n{'='*130}")
    print(f"TOTAL: {len(results)} profitable configs")
    print(f"{'='*130}")

    if results:
        results.sort(key=lambda x: -x['ann'])

        print(f"\nTOP 30 BY ANNUAL RETURN:")
        for r in results[:30]:
            print_result(r)

        print(f"\n--- TOP 10 BY WIN RATE ---")
        by_wr = sorted(results, key=lambda x: -x['wr'])
        for r in by_wr[:10]:
            print_result(r)

        print(f"\n--- TOP 10 BY RISK-ADJUSTED ---")
        by_ra = sorted(results, key=lambda x: -x['ann'] / max(x['dd'], 1))
        for r in by_ra[:10]:
            ratio = r['ann'] / max(r['dd'], 1)
            print(f"  {r['name']:55s} | Ann {r['ann']:+7.1f}% | DD {r['dd']:6.1f}% | "
                  f"R {ratio:.2f} | WR {r['wr']:5.1f}%")

        print(f"\n--- YEARLY BREAKDOWN (Top 5) ---")
        for r in results[:5]:
            print(f"\n  {r['name']}:")
            for y in sorted(r['yearly'].keys()):
                ys = r['yearly'][y]
                wr = ys['w'] / ys['n'] * 100 if ys['n'] > 0 else 0
                print(f"    {y}: {ys['n']:3d} trades, WR {wr:5.1f}%, PnL {ys['pnl']:+.1f}%")

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
