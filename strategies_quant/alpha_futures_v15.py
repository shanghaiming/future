"""
Alpha Futures V15 — Advanced Swing with Pyramiding + Lead-Lag + Drawdown Control
================================================================================
核心创新:
  1. 金字塔加仓: 1/3仓入场 → 盈利+2天后加1/3 → 再盈利+2天加1/3
     → 输的仓位1/3, 赢的仓位100%, 自然3:1盈亏比
  2. 跨品种滞后: 铜→其他金属, 原油→化工, 豆粕→豆系
     → 领涨品种动了, 跟涨品种还没动, 买入跟涨
  3. 回撤控制: >5%回撤减半, >15%回撤1/4仓, >25%停止交易
  4. 多周期对齐: 3d+10d+趋势方向一致 → 信号加强
  5. VDP_MOM最优评分 (来自v14b验证)

约束: 不做gap, 不做日内, 无杠杆, 2-7天持仓
"""
import sys, os, time, warnings, json
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
COMM = 0.0003

# Cross-commodity groups and leaders
GROUPS = {
    'black': {'leader': 'rbfi', 'followers': ['hcfi', 'ifi', 'jfi', 'jmfi']},
    'metal': {'leader': 'cufi', 'followers': ['alfi', 'znfi', 'aufi', 'agfi', 'nifi', 'sffi', 'ssfi', 'pbfi', 'snfi']},
    'energy': {'leader': 'scfi', 'followers': ['mafi', 'tafi', 'bfi', 'fufi', 'egfi', 'pgfi', 'ebfi', 'bufi', 'oifi']},
    'agri': {'leader': 'mfi', 'followers': ['afi', 'yfi', 'cfi', 'srfi', 'pfi', 'csfi', 'rrfi', 'rsfi', 'whfi', 'pkfi']},
    'chem': {'leader': 'ppfi', 'followers': ['vfi', 'lfi', 'egfi', 'mafi', 'tafi']},
}


def load_term_structure(syms, dates, sym_set):
    """Load term structure carry signal from JSON files."""
    ts_dir = '/Users/chengming/home/futures_platform/data/futures_term_structure'
    if not os.path.exists(ts_dir):
        return None

    dm = {d: i for i, d in enumerate(dates)}
    NS = len(syms)
    ND = len(dates)
    carry = np.full((NS, ND), np.nan)  # carry signal: negative=backwardation(bullish), positive=contango(bearish)

    loaded = 0
    for si, sym in enumerate(syms):
        # Try lowercase symbol for filenames
        prefix = sym.replace('fi', '').replace('fi', '')
        files = [f for f in os.listdir(ts_dir) if f.startswith(sym + '_') and f.endswith('.json')]
        if not files:
            # Try without 'fi'
            base = sym.replace('fi', '')
            files = [f for f in os.listdir(ts_dir) if f.startswith(base + '_') and f.endswith('.json')]
        if not files:
            continue

        for fname in files:
            try:
                with open(os.path.join(ts_dir, fname)) as f:
                    data = json.load(f)
                date_str = data.get('date', '')
                if not date_str: continue
                d = np.datetime64(date_str)
                if d not in dm: continue
                di = dm[d]
                # Spread: negative = backwardation, positive = contango
                sp = data.get('total_spread_pct', np.nan)
                if sp is not None and not np.isnan(sp):
                    carry[si, di] = sp
                    loaded += 1
            except:
                pass

    print(f"  期限结构: {loaded} data points loaded", flush=True)
    return carry


if __name__ == '__main__':
    print("=" * 95, flush=True)
    print("  Alpha Futures V15 — Advanced Swing (金字塔+跨品种+回撤控制)", flush=True)
    print("=" * 95, flush=True)

    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)
    dm = {d: i for i, d in enumerate(dates)}

    # Load term structure
    carry = load_term_structure(syms, dates, sym_set)
    has_carry = carry is not None and not np.all(np.isnan(carry))

    print("\n  预计算因子...", flush=True)
    t0 = time.time()

    # Pre-compute factors
    mom3 = np.full((NS, ND), np.nan)
    mom5 = np.full((NS, ND), np.nan)
    mom10 = np.full((NS, ND), np.nan)
    oi_mom5 = np.full((NS, ND), np.nan)
    vdp_ema = np.full((NS, ND), np.nan)
    vol_ratio = np.full((NS, ND), np.nan)
    atr10 = np.full((NS, ND), np.nan)
    body_r = np.full((NS, ND), np.nan)
    donch_pos = np.full((NS, ND), np.nan)
    # Lead-lag bonus
    lead_bonus = np.full((NS, ND), 0.0)
    # Carry smooth (5-day EMA)
    carry_ema = np.full((NS, ND), np.nan)
    # Multi-TF alignment
    mtf_align = np.full((NS, ND), np.nan)

    # Compute lead-lag index
    leader_si = {}
    follower_to_leader = {}
    for gname, gdata in GROUPS.items():
        ldr = gdata['leader']
        if ldr in sym_set:
            leader_si[gname] = syms.index(ldr)
            for f in gdata['followers']:
                if f in sym_set:
                    follower_to_leader[f] = (gname, syms.index(f))

    for si in range(NS):
        vdp_e = 0.0
        ce = 0.0  # carry EMA
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
                oi5 = OI[si, max(0, d-4)]
                if not np.isnan(oi5) and oi5 > 0:
                    oi_mom5[si, di] = (oi_now - oi5) / oi5

            # VDP EMA
            hl = H[si, d] - L[si, d]
            if not np.isnan(hl) and hl > 0:
                cd = C[si, d]; hd = H[si, d]; ld = L[si, d]; vd = V[si, d]
                if not any(np.isnan([cd, hd, ld, vd])):
                    vdp_val = vd * (2*cd - hd - ld) / hl
                    alpha = 2.0 / 15
                    vdp_e = alpha * vdp_val + (1 - alpha) * vdp_e
                    vdp_ema[si, di] = vdp_e

            # Body ratio
            if not np.isnan(hl) and hl > 0:
                co = c_now - O[si, d]
                if not np.isnan(co):
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
            if trs: atr10[si, di] = np.mean(trs)

            # Donchian position
            if di >= 20:
                h20 = H[si, max(0, d-19):d+1]
                l20 = L[si, max(0, d-19):d+1]
                h20v = h20[~np.isnan(h20)]
                l20v = l20[~np.isnan(l20)]
                if len(h20v) > 0 and len(l20v) > 0:
                    hh = np.max(h20v); ll = np.min(l20v)
                    rng = hh - ll
                    if rng > 0:
                        donch_pos[si, di] = (c_now - ll) / rng

            # Carry EMA
            if has_carry and not np.isnan(carry[si, di]):
                cv = carry[si, di]
                a = 2.0 / 6  # 5-day EMA
                ce = a * cv + (1 - a) * ce if not np.isnan(ce) else cv
                carry_ema[si, di] = ce

            # Multi-TF alignment
            m3 = mom3[si, di]
            m10 = mom10[si, di]
            if not np.isnan(m3) and not np.isnan(m10):
                if m3 > 0 and m10 > 0:
                    mtf_align[si, di] = min(abs(m3) + abs(m10), 0.2)
                elif m3 < 0 and m10 < 0:
                    mtf_align[si, di] = -min(abs(m3) + abs(m10), 0.2)

    # Lead-lag bonus
    for gname, gdata in GROUPS.items():
        ldr = gdata['leader']
        if ldr not in sym_set: continue
        lsi = syms.index(ldr)
        for di in range(25, ND):
            # Leader's 3-day momentum
            lm = mom3[lsi, di]
            if np.isnan(lm): continue
            for f in gdata['followers']:
                if f not in sym_set: continue
                fsi = syms.index(f)
                fm = mom3[fsi, di]
                if np.isnan(fm): continue
                # If leader moved but follower hasn't → bonus
                if lm > 0.02 and fm < lm * 0.5:
                    lead_bonus[fsi, di] += 0.3 * lm
                elif lm < -0.02 and fm > lm * 0.5:
                    lead_bonus[fsi, di] += 0.3 * lm  # Negative = bearish

    print(f"  因子完成 ({time.time()-t0:.0f}s)", flush=True)

    # ============================================================
    # Scoring functions
    # ============================================================

    def make_score_v15(w_mom=0.30, w_oi=0.15, w_vdp=0.20, w_lead=0.15,
                       w_mtf=0.10, w_carry=0.10, use_carry=True,
                       min_mom=0.0, require_mtf=False):
        def score(si, di):
            vals = []; ws = []
            # Momentum
            m5 = mom5[si, di]
            if np.isnan(m5): return np.nan
            if abs(m5) < min_mom: return 0
            vals.append(np.clip(m5 * 8, -1, 1)); ws.append(w_mom)

            # OI momentum
            om = oi_mom5[si, di]
            if not np.isnan(om):
                oi_sc = np.clip(om * 5, -1, 1)
                vals.append(oi_sc); ws.append(w_oi)

            # VDP
            vd = vdp_ema[si, di]
            if not np.isnan(vd):
                vdp_sc = np.sign(vd) * min(abs(vd) / 5e6, 1.0)
                # VDP-MOM interaction: boost when aligned
                m5v = m5 * 8
                if (vdp_sc > 0 and m5v > 0) or (vdp_sc < 0 and m5v < 0):
                    vdp_sc *= 1.3
                else:
                    vdp_sc *= 0.5
                vals.append(vdp_sc); ws.append(w_vdp)

            # Lead-lag bonus
            lb = lead_bonus[si, di]
            if abs(lb) > 0.01:
                vals.append(np.clip(lb * 2, -1, 1)); ws.append(w_lead)

            # Multi-TF alignment
            mtf = mtf_align[si, di]
            if not np.isnan(mtf):
                vals.append(np.clip(mtf * 5, -1, 1)); ws.append(w_mtf)
                if require_mtf and abs(mtf) < 0.02:
                    return 0  # Skip if no multi-TF alignment

            # Carry (term structure)
            if use_carry and has_carry:
                ce = carry_ema[si, di]
                if not np.isnan(ce):
                    # Backwardation (negative carry) + momentum up = bullish
                    carry_sc = 0
                    if m5 > 0 and ce < -0.1: carry_sc = 0.5  # Backwardation + up
                    elif m5 < 0 and ce > 0.1: carry_sc = -0.5  # Contango + down
                    vals.append(carry_sc); ws.append(w_carry)

            if not vals: return np.nan
            return sum(v*w for v,w in zip(vals, ws)) / sum(ws)
        return score

    # ============================================================
    # Backtest with pyramiding
    # ============================================================

    def run_v15(score_fn, name, hold_max=7, trail_atr=2.0, stop_loss=0.05,
                allow_short=True, pyramid=True, use_dd_ctrl=True):
        cash = float(CASH0)
        trades = []
        pos = None  # {'si', 'entry', 'entry_di', 'lots', 'dir', 'sym', 'atr', 'trail_price', 'scale_level'}
        peak_equity = float(CASH0)

        for di in range(MIN_TRAIN, ND):
            year = dates[di].year

            # Track peak equity for drawdown control
            if cash > peak_equity:
                peak_equity = cash

            # Drawdown-based position sizing
            size_mult = 1.0
            if use_dd_ctrl:
                dd = (peak_equity - cash) / peak_equity if peak_equity > 0 else 0
                if dd > 0.25:
                    size_mult = 0  # Stop trading
                elif dd > 0.15:
                    size_mult = 0.25
                elif dd > 0.05:
                    size_mult = 0.5

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

                # Fixed stop loss
                if pnl_pct / 100 < -stop_loss:
                    exit_reason = 'stop'

                # Trailing stop
                if exit_reason is None and trail_atr > 0 and days_held >= 2:
                    atr = pos.get('atr', 0)
                    trail_price = pos.get('trail_price', pos['entry'])
                    if atr > 0:
                        if pos['dir'] == 1:
                            new_trail = c - trail_atr * atr
                            if new_trail > trail_price:
                                pos['trail_price'] = new_trail
                            if c < trail_price:
                                exit_reason = 'trail'
                        else:
                            new_trail = c + trail_atr * atr
                            if new_trail < trail_price:
                                pos['trail_price'] = new_trail
                            if c > trail_price:
                                exit_reason = 'trail'

                # Signal flip exit
                if exit_reason is None and days_held >= 2:
                    cur_score = score_fn(pos['si'], di)
                    if not np.isnan(cur_score):
                        if pos['dir'] == 1 and cur_score < -0.15:
                            exit_reason = 'signal_flip'
                        elif pos['dir'] == -1 and cur_score > 0.15:
                            exit_reason = 'signal_flip'

                # Time exit
                if exit_reason is None and days_held >= hold_max:
                    exit_reason = 'time'

                # Pyramid: add to winner on days 2 and 4
                if exit_reason is None and pyramid and size_mult > 0:
                    scale = pos.get('scale_level', 0)
                    if scale == 0 and days_held >= 2 and pnl_pct > 0:
                        # Add 1/3 more
                        add_lots = int(cash * 0.33 / (c * mult)) if c * mult > 0 else 0
                        if add_lots > 0:
                            cost = add_lots * mult * c * (1 + COMM)
                            if cost <= cash:
                                cash -= cost
                                pos['lots'] += add_lots
                                pos['scale_level'] = 1
                    elif scale == 1 and days_held >= 4 and pnl_pct > 0:
                        add_lots = int(cash * 0.5 / (c * mult)) if c * mult > 0 else 0
                        if add_lots > 0:
                            cost = add_lots * mult * c * (1 + COMM)
                            if cost <= cash:
                                cash -= cost
                                pos['lots'] += add_lots
                                pos['scale_level'] = 2

                if exit_reason:
                    cash += mkt_val * (1 - COMM)
                    trades.append({
                        'pnl_pct': pnl_pct, 'pnl_abs': pnl,
                        'days': days_held, 'di': di, 'year': year,
                        'sym': pos['sym'], 'dir': pos['dir'],
                        'reason': exit_reason,
                        'scale': pos.get('scale_level', 0)
                    })
                    pos = None

            # === ENTRY ===
            if pos is None and size_mult > 0:
                best_si, best_dir, best_sc = -1, 0, 0
                for si in range(NS):
                    sc = score_fn(si, di)
                    if np.isnan(sc): continue
                    if sc > best_sc:
                        best_sc = sc; best_si = si; best_dir = 1
                    if allow_short and -sc > best_sc:
                        best_sc = -sc; best_si = si; best_dir = -1

                if best_si >= 0 and best_sc > 0.05:
                    c = C[best_si, di]
                    if np.isnan(c) or c <= 0: continue

                    sym = syms[best_si]
                    mult = MULT.get(sym, DEF_MULT)
                    notional = c * mult
                    if notional <= 0: continue

                    # Start with 1/3 of (sized) capital
                    alloc = cash * 0.33 * size_mult
                    lots = int(alloc / notional)
                    if lots <= 0: continue

                    cost_in = notional * lots * (1 + COMM)
                    if cost_in > cash: continue

                    # ATR
                    atr_val = 0
                    trs = []
                    for dd in range(max(1, di-10), di+1):
                        hi = H[best_si, dd]; lo = L[best_si, dd]; pc = C[best_si, dd-1]
                        if np.isnan(hi) or np.isnan(lo): continue
                        tr = hi - lo
                        if not np.isnan(pc):
                            tr = max(tr, abs(hi-pc), abs(lo-pc))
                        trs.append(tr)
                    if trs: atr_val = np.mean(trs)

                    cash -= cost_in
                    trail_price = c - trail_atr * atr_val if best_dir == 1 else c + trail_atr * atr_val
                    pos = {
                        'si': best_si, 'entry': c, 'entry_di': di,
                        'lots': lots, 'dir': best_dir, 'sym': sym,
                        'atr': atr_val, 'trail_price': trail_price,
                        'scale_level': 0
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
                'scale': pos.get('scale_level', 0)
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
            'avg_win': round(avg_win, 2), 'avg_loss': round(avg_loss, 2),
            'final': round(cash, 0), 'years': year_stats, 'reasons': reasons
        }

    # ============================================================
    # Parameter configurations
    # ============================================================
    configs = []

    for w_mom in [0.30, 0.40]:
        for w_oi in [0.10, 0.15]:
            for w_vdp in [0.20, 0.25]:
                for w_lead in [0.10, 0.15]:
                    for use_carry in ([True, False] if has_carry else [False]):
                        for hold in [5, 7]:
                            for trail in [2.0, 3.0]:
                                for sl in [0.04, 0.06]:
                                    for pyramid in [True, False]:
                                        w_mtf = 0.10
                                        w_carry = 0.10 if use_carry else 0
                                        sname = f"M{w_mom:.0f}O{w_oi:.0f}V{w_vdp:.0f}L{w_lead:.0f}C{w_carry:.0f}_H{hold}_T{trail}_S{sl}_{'P' if pyramid else 'N'}"
                                        configs.append((sname, make_score_v15(
                                            w_mom=w_mom, w_oi=w_oi, w_vdp=w_vdp,
                                            w_lead=w_lead, w_carry=w_carry,
                                            use_carry=use_carry
                                        ), hold, trail, sl, True, pyramid, True))

    print(f"  共 {len(configs)} 个配置", flush=True)

    results = []
    for ci, (sname, sfn, hold, trail, sl, short, pyramid, dd_ctrl) in enumerate(configs):
        if ci % 50 == 0:
            print(f"  配置 {ci}/{len(configs)} ({len(results)} profitable)", flush=True)

        r = run_v15(sfn, sname, hold_max=hold, trail_atr=trail, stop_loss=sl,
                     allow_short=short, pyramid=pyramid, use_dd_ctrl=dd_ctrl)
        if r and r['ann'] > 5:
            results.append(r)

    print(f"\n  完成 ({time.time()-t0:.0f}s, {len(results)} >5%)", flush=True)
    results.sort(key=lambda x: -x['ann'])

    print(f"\n{'='*95}", flush=True)
    print(f"  TOP 30", flush=True)
    print(f"  {'Strategy':55s} | {'Ann':>8s} {'WR':>5s} {'N':>4s} {'DD':>6s} {'AvgW':>6s} {'AvgL':>6s} {'AvgD':>5s}", flush=True)
    for r in results[:30]:
        print(f"  {r['name']:55s} | {r['ann']:+8.1f}% {r['wr']:5.1f}% {r['n']:4d} "
              f"{r['dd']:6.1f}% {r['avg_win']:+6.2f}% {r['avg_loss']:6.2f}% {r['avg_days']:5.1f}d", flush=True)

    for i, r in enumerate(results[:5]):
        print(f"\n  #{i+1}: {r['name']} (Ann={r['ann']:+.1f}%, WR={r['wr']:.0f}%, DD={r['dd']:.1f}%)", flush=True)
        print(f"    AvgWin={r['avg_win']:+.2f}% AvgLoss={r['avg_loss']:.2f}% AvgDays={r['avg_days']:.1f}", flush=True)
        for reason, s in sorted(r['reasons'].items(), key=lambda x: -x[1]['n']):
            rwr = s['w'] / max(s['n'], 1) * 100
            print(f"    {reason:12s}: {s['n']:4d}t WR={rwr:.0f}% pnl={s['pnl']:+.1f}%", flush=True)
        for y in sorted(r['years'].keys()):
            s = r['years'][y]
            wr = s['w'] / max(s['n'], 1) * 100
            print(f"    {y}: {s['n']:3d}t WR={wr:.0f}% pnl={s['pnl']:+.1f}%", flush=True)

    print(f"\n  目标: 年化600%+ WR50%+ 无杠杆 纯日线", flush=True)
    if results and results[0]['ann'] >= 600:
        print(f"  >>> TARGET ACHIEVED <<<", flush=True)
    elif results:
        print(f"  Best: {results[0]['ann']:+.1f}% — gap to 600%: {600-results[0]['ann']:.0f}%", flush=True)
    print(f"{'='*95}", flush=True)
