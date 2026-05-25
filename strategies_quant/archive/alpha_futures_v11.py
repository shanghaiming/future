"""
Alpha Futures V11 — OI+VDP+仓位分析 (无杠杆, 无Gap)
====================================================
策略来源: 仓位分析18规则 + VDP量压 + Kalman速度 + 熵过滤
全部使用OHLCV+OI数据, 不用隔夜gap信号

策略:
  A. 仓位分析矩阵 — Price+OI+Volume 18规则 + VDP确认
  B. OI动量翻转 — OI从负转正 + 价格动量确认
  C. VDP假突破反转 — Donchian突破但VDP反向 → 反转交易
  D. 每日轮动 — 综合评分排名, 持最强品种
  E. 仓位势能 — 多维能量模型 (OI权重40%)
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


# ====================================================================
# Factor Precomputation
# ====================================================================
def precompute_factors(NS, ND, C, O, H, L, V, OI):
    """预计算所有因子 — 只用di-1及之前数据, 无look-ahead"""
    print("  预计算因子...", flush=True)
    t0 = time.time()

    # OI 5日动量
    oi_mom5 = np.full((NS, ND), np.nan)
    # OI 10日动量
    oi_mom10 = np.full((NS, ND), np.nan)
    # Volume ratio (vs 20日均量)
    vol_ratio = np.full((NS, ND), np.nan)
    # VDP: Volume Delta Pressure (用前一日)
    vdp = np.full((NS, ND), np.nan)
    # VDP EMA14
    vdp_ema = np.full((NS, ND), np.nan)
    # ATR 10日
    atr10 = np.full((NS, ND), np.nan)
    # 价格动量 5日, 10日, 20日
    mom5 = np.full((NS, ND), np.nan)
    mom10 = np.full((NS, ND), np.nan)
    mom20 = np.full((NS, ND), np.nan)
    # Donchian channel highs/lows
    donch_high5 = np.full((NS, ND), np.nan)
    donch_low5 = np.full((NS, ND), np.nan)
    donch_high10 = np.full((NS, ND), np.nan)
    donch_low10 = np.full((NS, ND), np.nan)
    # Shannon entropy (20日)
    entropy20 = np.full((NS, ND), np.nan)
    # 布林带宽度
    bb_width = np.full((NS, ND), np.nan)
    # RSI 14日
    rsi14 = np.full((NS, ND), np.nan)

    for si in range(NS):
        vdp_ema_val = 0
        for di in range(20, ND):
            d = di - 1  # use previous day

            # --- OI动量 ---
            oi_now = OI[si, d]
            if not np.isnan(oi_now) and oi_now > 0:
                oi5 = OI[si, max(0, d-4):d+1]
                oi5v = oi5[~np.isnan(oi5)]
                if len(oi5v) >= 3:
                    oi_mom5[si, di] = (oi_now - oi5v[0]) / oi5v[0]
                oi10 = OI[si, max(0, d-9):d+1]
                oi10v = oi10[~np.isnan(oi10)]
                if len(oi10v) >= 5:
                    oi_mom10[si, di] = (oi_now - oi10v[0]) / oi10v[0]

            # --- Volume ratio ---
            v_now = V[si, d]
            if not np.isnan(v_now):
                v20 = V[si, max(0, d-19):d+1]
                v20v = v20[~np.isnan(v20)]
                if len(v20v) >= 10:
                    vol_ratio[si, di] = v_now / np.mean(v20v)

            # --- VDP ---
            hl = H[si, d] - L[si, d]
            if not np.isnan(hl) and hl > 0:
                c_d = C[si, d]; h_d = H[si, d]; l_d = L[si, d]; v_d = V[si, d]
                if not any(np.isnan([c_d, h_d, l_d, v_d])):
                    vdp_val = v_d * (2 * c_d - h_d - l_d) / hl
                    vdp[si, di] = vdp_val
                    alpha = 2.0 / 15  # EMA14
                    vdp_ema_val = alpha * vdp_val + (1 - alpha) * vdp_ema_val
                    vdp_ema[si, di] = vdp_ema_val

            # --- ATR ---
            if di >= 11:
                trs = []
                for dd in range(max(1, d-9), d+1):
                    hi = H[si, dd]; lo = L[si, dd]; pc = C[si, dd-1]
                    if np.isnan(hi) or np.isnan(lo): continue
                    tr = hi - lo
                    if not np.isnan(pc): tr = max(tr, abs(hi-pc), abs(lo-pc))
                    trs.append(tr)
                if trs: atr10[si, di] = np.mean(trs)

            # --- 价格动量 ---
            c_now = C[si, d]
            if not np.isnan(c_now) and c_now > 0:
                for lookback, arr in [(5, mom5), (10, mom10), (20, mom20)]:
                    c_prev = C[si, max(0, d - lookback)]
                    if not np.isnan(c_prev) and c_prev > 0:
                        arr[si, di] = (c_now - c_prev) / c_prev

            # --- Donchian ---
            for chan, h_arr, l_arr in [(5, donch_high5, donch_low5), (10, donch_high10, donch_low10)]:
                ch = C[si, max(0, d-chan+1):d+1]
                vc = ch[~np.isnan(ch)]
                if len(vc) >= max(1, chan // 2):
                    h_arr[si, di] = np.max(vc)
                    l_arr[si, di] = np.min(vc)

            # --- Shannon Entropy (returns distribution) ---
            rets = []
            for dd in range(max(1, d-19), d+1):
                if not np.isnan(C[si, dd]) and not np.isnan(C[si, dd-1]) and C[si, dd-1] > 0:
                    rets.append((C[si, dd] - C[si, dd-1]) / C[si, dd-1])
            if len(rets) >= 10:
                rets = np.array(rets)
                # Bin into 5 buckets
                try:
                    hist, _ = np.histogram(rets, bins=5, density=True)
                    probs = hist * np.diff(_)
                    probs = probs[probs > 0]
                    if len(probs) > 0:
                        entropy20[si, di] = -np.sum(probs * np.log2(probs))
                except:
                    pass

            # --- BB width ---
            c20 = C[si, max(0, d-19):d+1]
            c20v = c20[~np.isnan(c20)]
            if len(c20v) >= 10:
                sma = np.mean(c20v)
                std = np.std(c20v)
                if sma > 0:
                    bb_width[si, di] = 4 * std / sma  # (upper-lower)/mid ≈ 4*std/sma

            # --- RSI 14 ---
            if di >= 15:
                gains = []; losses = []
                for dd in range(max(1, d-13), d+1):
                    delta = C[si, dd] - C[si, dd-1]
                    if np.isnan(delta): continue
                    if delta > 0: gains.append(delta)
                    else: losses.append(abs(delta))
                avg_gain = np.mean(gains) if gains else 0
                avg_loss = np.mean(losses) if losses else 0.0001
                rs = avg_gain / avg_loss
                rsi14[si, di] = 100 - 100 / (1 + rs)

    print(f"  因子完成 ({time.time()-t0:.0f}s)", flush=True)
    return {
        'oi_mom5': oi_mom5, 'oi_mom10': oi_mom10,
        'vol_ratio': vol_ratio,
        'vdp': vdp, 'vdp_ema': vdp_ema,
        'atr10': atr10,
        'mom5': mom5, 'mom10': mom10, 'mom20': mom20,
        'donch_high5': donch_high5, 'donch_low5': donch_low5,
        'donch_high10': donch_high10, 'donch_low10': donch_low10,
        'entropy20': entropy20, 'bb_width': bb_width, 'rsi14': rsi14,
    }


# ====================================================================
# Swing Backtest Engine
# ====================================================================
def swing_backtest(buy_d, sell_d, short_d, cover_d,
                    NS, ND, dates, C, O, H, L, V, OI, syms,
                    max_positions=1, sl_pct=0.05, hold_max=15,
                    ranking='mom5', factors=None):
    cash = float(CASH0)
    positions = []
    trades = []
    year_stats = {}

    for di in range(MIN_TRAIN, ND):
        year = dates[di].year

        # 平仓
        for pos in list(positions):
            si = pos['si']
            c = C[si, di]
            if np.isnan(c): continue
            mult = MULT.get(pos['sym'], DEF_MULT)
            pnl = (c - pos['entry']) * mult * pos['lots'] * pos['dir']
            pnl_pct = pnl / (pos['entry'] * mult * pos['lots']) * 100

            # ATR追踪止损
            if pos.get('trail_stop'):
                if pos['dir'] == 1 and c < pos['trail_stop']:
                    cash += c * mult * pos['lots'] * (1 - COMM_RATE)
                    trades.append({'pnl_pct': pnl_pct, 'pnl_abs': pnl,
                                   'days': di - pos['entry_di'], 'di': di,
                                   'reason': 'trail', 'year': year, 'si': si, 'dir': pos['dir']})
                    positions.remove(pos)
                    continue
                elif pos['dir'] == -1 and c > pos['trail_stop']:
                    cash += c * mult * pos['lots'] * (1 - COMM_RATE)
                    trades.append({'pnl_pct': pnl_pct, 'pnl_abs': pnl,
                                   'days': di - pos['entry_di'], 'di': di,
                                   'reason': 'trail', 'year': year, 'si': si, 'dir': pos['dir']})
                    positions.remove(pos)
                    continue
                # Update trail
                atr = factors['atr10'][si, di] if factors else 0
                if not np.isnan(atr) and atr > 0:
                    if pos['dir'] == 1:
                        new_stop = c - 2.0 * atr
                        if new_stop > pos['trail_stop']:
                            pos['trail_stop'] = new_stop
                    else:
                        new_stop = c + 2.0 * atr
                        if not np.isnan(pos['trail_stop']) and new_stop < pos['trail_stop']:
                            pos['trail_stop'] = new_stop

            # 固定止损
            if pnl_pct / 100 < -sl_pct:
                cash += c * mult * pos['lots'] * (1 - COMM_RATE)
                trades.append({'pnl_pct': pnl_pct, 'pnl_abs': pnl,
                               'days': di - pos['entry_di'], 'di': di,
                               'reason': 'stop', 'year': year, 'si': si, 'dir': pos['dir']})
                positions.remove(pos)
                continue

            # 信号平仓
            if pos['dir'] == 1 and di in sell_d.get(si, set()):
                cash += c * mult * pos['lots'] * (1 - COMM_RATE)
                trades.append({'pnl_pct': pnl_pct, 'pnl_abs': pnl,
                               'days': di - pos['entry_di'], 'di': di,
                               'reason': 'signal', 'year': year, 'si': si, 'dir': pos['dir']})
                positions.remove(pos)
                continue
            if pos['dir'] == -1 and di in cover_d.get(si, set()):
                cash += c * mult * pos['lots'] * (1 - COMM_RATE)
                trades.append({'pnl_pct': pnl_pct, 'pnl_abs': pnl,
                               'days': di - pos['entry_di'], 'di': di,
                               'reason': 'signal', 'year': year, 'si': si, 'dir': pos['dir']})
                positions.remove(pos)
                continue

            # 超时
            if di - pos['entry_di'] >= hold_max:
                cash += c * mult * pos['lots'] * (1 - COMM_RATE)
                trades.append({'pnl_pct': pnl_pct, 'pnl_abs': pnl,
                               'days': di - pos['entry_di'], 'di': di,
                               'reason': 'time', 'year': year, 'si': si, 'dir': pos['dir']})
                positions.remove(pos)

        # 开仓
        if len(positions) < max_positions:
            candidates = []
            for si in range(NS):
                if any(p['si'] == si for p in positions): continue
                c = C[si, di]
                if np.isnan(c) or c <= 0: continue

                is_buy = di in buy_d.get(si, set())
                is_short = di in short_d.get(si, set())
                if not is_buy and not is_short:
                    continue

                direction = 1 if is_buy else -1

                # Ranking score
                score = 0
                if ranking == 'mom5':
                    m5 = factors['mom5'][si, di]
                    score = m5 * direction if not np.isnan(m5) else 0
                elif ranking == 'mom10':
                    m10 = factors['mom10'][si, di]
                    score = m10 * direction if not np.isnan(m10) else 0
                elif ranking == 'oi_mom':
                    om = factors['oi_mom5'][si, di]
                    score = om * direction if not np.isnan(om) else 0
                elif ranking == 'vdp':
                    v = factors['vdp_ema'][si, di]
                    score = v * direction if not np.isnan(v) else 0
                elif ranking == 'composite':
                    m5 = factors['mom5'][si, di]
                    om = factors['oi_mom5'][si, di]
                    vr = factors['vol_ratio'][si, di]
                    s1 = m5 * direction if not np.isnan(m5) else 0
                    s2 = om * direction if not np.isnan(om) else 0
                    s3 = (vr - 1) * direction if not np.isnan(vr) else 0
                    score = 0.3 * s1 + 0.4 * s2 + 0.3 * s3

                candidates.append((si, c, direction, score, syms[si]))

            candidates.sort(key=lambda x: x[3], reverse=True)
            slots = max_positions - len(positions)

            for si, price, direction, score, sym in candidates[:slots]:
                mult = MULT.get(sym, DEF_MULT)
                notional = price * mult
                if notional <= 0: continue
                lots = int(cash / notional)
                if lots <= 0: continue
                cost = notional * lots * (1 + COMM_RATE)
                if cost > cash: continue

                cash -= cost
                trail = 0
                atr = factors['atr10'][si, di] if factors else np.nan
                if not np.isnan(atr) and atr > 0:
                    if direction == 1:
                        trail = price - 2.0 * atr
                    else:
                        trail = price + 2.0 * atr
                positions.append({
                    'si': si, 'entry': price, 'entry_di': di,
                    'lots': lots, 'dir': direction, 'sym': sym,
                    'trail_stop': trail,
                })

    # 清仓
    for pos in positions:
        c = C[pos['si'], ND-1]
        if np.isnan(c) or c <= 0: c = pos['entry']
        mult = MULT.get(pos['sym'], DEF_MULT)
        pnl = (c - pos['entry']) * mult * pos['lots'] * pos['dir']
        pnl_pct = pnl / (pos['entry'] * mult * pos['lots']) * 100
        cash += c * mult * pos['lots'] * (1 - COMM_RATE)
        trades.append({'pnl_pct': pnl_pct, 'pnl_abs': pnl,
                       'days': 999, 'di': ND-1, 'reason': 'end',
                       'year': dates[ND-1].year, 'si': pos['si'], 'dir': pos['dir']})

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
        'year_stats': year_stats,
    }


# ====================================================================
# Daily Rotation Backtest
# ====================================================================
def rotation_backtest(NS, ND, dates, C, O, H, L, V, OI, syms,
                       rebalance_days, hold_min, sl_pct, factors):
    """每日/每N日轮动: 按综合评分持仓最强品种"""
    cash = float(CASH0)
    position = None  # single position
    trades = []
    year_stats = {}
    last_rebalance = -999

    for di in range(MIN_TRAIN, ND):
        year = dates[di].year

        # 平仓检查
        if position is not None:
            si = position['si']
            c = C[si, di]
            if np.isnan(c):
                pass  # skip
            else:
                mult = MULT.get(position['sym'], DEF_MULT)
                pnl = (c - position['entry']) * mult * position['lots'] * position['dir']
                pnl_pct = pnl / (position['entry'] * mult * position['lots']) * 100

                # 止损
                if pnl_pct / 100 < -sl_pct:
                    cash += c * mult * position['lots'] * (1 - COMM_RATE)
                    trades.append({'pnl_pct': pnl_pct, 'pnl_abs': pnl,
                                   'days': di - position['entry_di'], 'di': di,
                                   'reason': 'stop', 'year': year, 'si': si, 'dir': position['dir']})
                    position = None

                # 持仓至少hold_min天后才允许换仓
                elif di - position['entry_di'] >= hold_min and di - last_rebalance >= rebalance_days:
                    # 计算当前持仓评分
                    cur_score = _score_symbol(si, di, position['dir'], factors)
                    # 找最强品种
                    best_si, best_dir, best_score = -1, 0, -999
                    for sj in range(NS):
                        for d in [1, -1]:
                            sc = _score_symbol(sj, di, d, factors)
                            if sc > best_score:
                                best_score = sc
                                best_si = sj
                                best_dir = d

                    if best_si >= 0 and (best_si != si or best_dir != position['dir']) and best_score > cur_score * 1.2:
                        # 换仓
                        cash += c * mult * position['lots'] * (1 - COMM_RATE)
                        trades.append({'pnl_pct': pnl_pct, 'pnl_abs': pnl,
                                       'days': di - position['entry_di'], 'di': di,
                                       'reason': 'rotate', 'year': year, 'si': si, 'dir': position['dir']})
                        position = None
                        last_rebalance = di

        # 开仓
        if position is None:
            best_si, best_dir, best_score = -1, 0, -999
            for si in range(NS):
                for d in [1, -1]:
                    sc = _score_symbol(si, di, d, factors)
                    if sc > best_score:
                        best_score = sc
                        best_si = si
                        best_dir = d

            if best_si >= 0 and best_score > 0.02:  # minimum score threshold
                c = C[best_si, di]
                if np.isnan(c) or c <= 0: continue
                sym = syms[best_si]
                mult = MULT.get(sym, DEF_MULT)
                notional = c * mult
                if notional <= 0: continue
                lots = int(cash / notional)
                if lots <= 0: continue
                cost = notional * lots * (1 + COMM_RATE)
                if cost > cash: continue
                cash -= cost
                position = {
                    'si': best_si, 'entry': c, 'entry_di': di,
                    'lots': lots, 'dir': best_dir, 'sym': sym,
                }
                last_rebalance = di

    # 清仓
    if position is not None:
        c = C[position['si'], ND-1]
        if np.isnan(c) or c <= 0: c = position['entry']
        mult = MULT.get(position['sym'], DEF_MULT)
        pnl = (c - position['entry']) * mult * position['lots'] * position['dir']
        pnl_pct = pnl / (position['entry'] * mult * position['lots']) * 100
        cash += c * mult * position['lots'] * (1 - COMM_RATE)
        trades.append({'pnl_pct': pnl_pct, 'pnl_abs': pnl,
                       'days': 999, 'di': ND-1, 'reason': 'end',
                       'year': dates[ND-1].year, 'si': position['si'], 'dir': position['dir']})

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
        'avg_win': 0, 'avg_loss': 0,
        'year_stats': year_stats,
    }


def _score_symbol(si, di, direction, factors):
    """综合评分: OI动量40% + 价格动量30% + VDP 30%"""
    oi_m = factors['oi_mom5'][si, di]
    m5 = factors['mom5'][si, di]
    vdp_v = factors['vdp_ema'][si, di]

    score = 0
    cnt = 0
    if not np.isnan(oi_m):
        score += 0.4 * oi_m * direction
        cnt += 1
    if not np.isnan(m5):
        score += 0.3 * m5 * direction
        cnt += 1
    if not np.isnan(vdp_v):
        # Normalize VDP by dividing by some scale
        score += 0.3 * np.sign(vdp_v) * direction * min(abs(vdp_v) / 1e8, 0.1)
        cnt += 1

    return score if cnt >= 2 else -999


# ====================================================================
# Main
# ====================================================================
if __name__ == '__main__':
    print("=" * 80, flush=True)
    print("  Alpha Futures V11 — OI+VDP+仓位分析 (无杠杆, 无Gap)", flush=True)
    print("=" * 80, flush=True)

    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)
    F = precompute_factors(NS, ND, C, O, H, L, V, OI)
    t0 = time.time()
    results = []

    # ===== 策略A: 仓位分析矩阵 =====
    # 18规则: price_up/down + oi_up/down + vol_expanding/normal/contracting
    # VDP确认: 做多需VDP>0, 做空需VDP<0
    print("\n  [A] 仓位分析矩阵 (Price+OI+Volume+VDP)", flush=True)
    for min_signal in [1, 2]:  # 1=中等信号, 2=强信号
        for use_vdp in [True, False]:
            buy_d = {si: set() for si in range(NS)}
            sell_d = {si: set() for si in range(NS)}
            short_d = {si: set() for si in range(NS)}
            cover_d = {si: set() for si in range(NS)}

            for si in range(NS):
                for di in range(MIN_TRAIN + 1, ND):
                    d = di - 1
                    c_now = C[si, d]; c_prev = C[si, d-1]
                    if np.isnan(c_now) or np.isnan(c_prev) or c_prev <= 0:
                        continue

                    price_chg = (c_now - c_prev) / c_prev
                    price_up = price_chg > 0.002
                    price_down = price_chg < -0.002

                    oi_m = F['oi_mom5'][si, di]
                    oi_up = not np.isnan(oi_m) and oi_m > 0.01
                    oi_down = not np.isnan(oi_m) and oi_m < -0.01

                    vr = F['vol_ratio'][si, di]
                    vol_exp = not np.isnan(vr) and vr > 1.3
                    vol_norm = not np.isnan(vr) and vr > 0.7

                    vdp_v = F['vdp_ema'][si, di]

                    # 18-rule scoring
                    signal = 0
                    if price_up and oi_up and vol_exp: signal = 2   # Strong long
                    elif price_up and oi_up and vol_norm: signal = 1
                    elif price_up and oi_up: signal = 1
                    elif price_down and oi_up and vol_exp: signal = -2  # Strong short
                    elif price_down and oi_up and vol_norm: signal = -1
                    elif price_down and oi_up: signal = -1
                    elif price_up and oi_down: signal = -1   # Short covering → bearish
                    elif price_down and oi_down: signal = 1   # Long covering → bullish

                    if abs(signal) < min_signal:
                        continue

                    # VDP confirmation
                    if use_vdp and not np.isnan(vdp_v):
                        if signal > 0 and vdp_v <= 0: continue  # Need buy pressure
                        if signal < 0 and vdp_v >= 0: continue  # Need sell pressure

                    if signal > 0:
                        buy_d[si].add(di)
                    elif signal < 0:
                        short_d[si].add(di)

                    # Exit: opposite signal or signal weakens
                    if signal > 0:
                        # Exit long when price down + oi down
                        sell_d[si].add(di)
                    elif signal < 0:
                        cover_d[si].add(di)

            for mp in [1]:
                for sl in [0.03, 0.05, 0.10]:
                    for hm in [5, 7, 10, 15]:
                        for rank in ['composite', 'oi_mom', 'mom5']:
                            r = swing_backtest(buy_d, sell_d, short_d, cover_d,
                                              NS, ND, dates, C, O, H, L, V, OI, syms,
                                              max_positions=mp, sl_pct=sl, hold_max=hm,
                                              ranking=rank, factors=F)
                            if r and r['ann'] > 0:
                                r['desc'] = f"POS_A_sig{min_signal}_{'vdp' if use_vdp else 'novdp'}"
                                r['mp'] = mp; r['sl'] = sl; r['hm'] = hm; r['rank'] = rank
                                results.append(r)
            print(f"    sig{min_signal}_{'vdp' if use_vdp else 'novdp'} done ({len(results)})", flush=True)

    # ===== 策略B: OI动量翻转 =====
    # OI从负转正 + 价格动量确认 → 做多
    # OI从正转负 + 价格动量确认 → 做空
    print("  [B] OI动量翻转", flush=True)
    buy_d = {si: set() for si in range(NS)}
    sell_d = {si: set() for si in range(NS)}
    short_d = {si: set() for si in range(NS)}
    cover_d = {si: set() for si in range(NS)}

    for si in range(NS):
        for di in range(MIN_TRAIN + 1, ND):
            om = F['oi_mom5'][si, di]
            om_prev = F['oi_mom5'][si, di-1]
            if np.isnan(om) or np.isnan(om_prev): continue

            m5 = F['mom5'][si, di]
            vdp_v = F['vdp_ema'][si, di]

            # OI flip: negative → positive (new money entering)
            if om > 0 and om_prev <= 0:
                if not np.isnan(m5) and m5 > 0:  # Price momentum confirms
                    if not np.isnan(vdp_v) and vdp_v > 0:  # VDP confirms buying
                        buy_d[si].add(di)
                    elif np.isnan(vdp_v):  # No VDP data, still ok
                        buy_d[si].add(di)

            # OI flip: positive → negative (money leaving)
            if om < 0 and om_prev >= 0:
                if not np.isnan(m5) and m5 < 0:
                    if not np.isnan(vdp_v) and vdp_v < 0:
                        short_d[si].add(di)
                    elif np.isnan(vdp_v):
                        short_d[si].add(di)

            # Exit on momentum reversal
            if om < 0 and m5 < 0: sell_d[si].add(di)
            if om > 0 and m5 > 0: cover_d[si].add(di)

    for mp in [1]:
        for sl in [0.03, 0.05, 0.10]:
            for hm in [5, 7, 10, 15]:
                for rank in ['composite', 'oi_mom']:
                    r = swing_backtest(buy_d, sell_d, short_d, cover_d,
                                      NS, ND, dates, C, O, H, L, V, OI, syms,
                                      max_positions=mp, sl_pct=sl, hold_max=hm,
                                      ranking=rank, factors=F)
                    if r and r['ann'] > 0:
                        r['desc'] = "OI_FLIP"
                        r['mp'] = mp; r['sl'] = sl; r['hm'] = hm; r['rank'] = rank
                        results.append(r)
    print(f"    done ({len(results)})", flush=True)

    # ===== 策略C: VDP假突破反转 =====
    # 价格突破Donchian但VDP反向 → 假突破 → 反向交易
    print("  [C] VDP假突破反转", flush=True)
    for chan in [5, 10]:
        for use_oi in [True, False]:
            buy_d = {si: set() for si in range(NS)}
            sell_d = {si: set() for si in range(NS)}
            short_d = {si: set() for si in range(NS)}
            cover_d = {si: set() for si in range(NS)}

            for si in range(NS):
                for di in range(MIN_TRAIN + 1, ND):
                    d = di - 1
                    c = C[si, d]
                    if np.isnan(c): continue

                    h_key = f'donch_high{chan}'
                    l_key = f'donch_low{chan}'
                    dh = F[h_key][si, di]
                    dl = F[l_key][si, di]
                    if np.isnan(dh) or np.isnan(dl): continue

                    vdp_v = F['vdp_ema'][si, di]
                    if np.isnan(vdp_v): continue

                    # Price breaks above channel BUT VDP negative → fake breakout → SHORT
                    if c > dh and vdp_v < 0:
                        if use_oi:
                            om = F['oi_mom5'][si, di]
                            if np.isnan(om) or om > 0: continue  # Need OI declining
                        short_d[si].add(di)

                    # Price breaks below channel BUT VDP positive → fake breakdown → LONG
                    if c < dl and vdp_v > 0:
                        if use_oi:
                            om = F['oi_mom5'][si, di]
                            if np.isnan(om) or om < 0: continue  # Need OI increasing
                        buy_d[si].add(di)

                    # Exit: Donchian mid-cross or VDP reversal
                    mid = (dh + dl) / 2
                    if c > mid: cover_d[si].add(di)  # Cover shorts
                    if c < mid: sell_d[si].add(di)   # Sell longs

            for mp in [1]:
                for sl in [0.03, 0.05]:
                    for hm in [3, 5, 7, 10]:
                        r = swing_backtest(buy_d, sell_d, short_d, cover_d,
                                          NS, ND, dates, C, O, H, L, V, OI, syms,
                                          max_positions=mp, sl_pct=sl, hold_max=hm,
                                          ranking='vdp', factors=F)
                        if r and r['ann'] > 0:
                            r['desc'] = f"FAKE_c{chan}_{'oi' if use_oi else 'nooi'}"
                            r['mp'] = mp; r['sl'] = sl; r['hm'] = hm; r['rank'] = 'vdp'
                            results.append(r)
            print(f"    c{chan}_{'oi' if use_oi else 'nooi'} done ({len(results)})", flush=True)

    # ===== 策略D: 每日轮动 =====
    print("  [D] 每日轮动 (综合评分)", flush=True)
    for reb in [1, 2, 3, 5]:
        for hold_min in [1, 2, 3]:
            for sl in [0.03, 0.05, 0.10]:
                r = rotation_backtest(NS, ND, dates, C, O, H, L, V, OI, syms,
                                      rebalance_days=reb, hold_min=hold_min,
                                      sl_pct=sl, factors=F)
                if r and r['ann'] > 0:
                    r['desc'] = f"ROTATE_r{reb}_h{hold_min}"
                    r['mp'] = 1; r['sl'] = sl; r['hm'] = hold_min; r['rank'] = 'composite'
                    results.append(r)
    print(f"    done ({len(results)})", flush=True)

    # ===== 策略E: 熵突破 (从chaos到order) =====
    print("  [E] 熵突破 + OI确认", flush=True)
    buy_d = {si: set() for si in range(NS)}
    sell_d = {si: set() for si in range(NS)}
    short_d = {si: set() for si in range(NS)}
    cover_d = {si: set() for si in range(NS)}

    for si in range(NS):
        for di in range(MIN_TRAIN + 1, ND):
            ent = F['entropy20'][si, di]
            ent_prev = F['entropy20'][si, di-1]
            if np.isnan(ent) or np.isnan(ent_prev): continue

            # Entropy drop (chaos → order)
            delta_h = ent - ent_prev

            m5 = F['mom5'][si, di]
            om = F['oi_mom5'][si, di]
            vdp_v = F['vdp_ema'][si, di]

            # Entropy dropping + momentum + OI + VDP alignment
            if delta_h < -0.05 and ent < 2.0:  # Significant entropy drop
                if not np.isnan(m5) and not np.isnan(om):
                    if m5 > 0 and om > 0:  # Bullish alignment
                        if np.isnan(vdp_v) or vdp_v > 0:
                            buy_d[si].add(di)
                    elif m5 < 0 and om < 0:  # Bearish alignment
                        if np.isnan(vdp_v) or vdp_v < 0:
                            short_d[si].add(di)

            # Exit when entropy rises (back to chaos)
            if ent > 2.2:
                sell_d[si].add(di)
                cover_d[si].add(di)

    for mp in [1]:
        for sl in [0.03, 0.05, 0.10]:
            for hm in [5, 7, 10, 15]:
                r = swing_backtest(buy_d, sell_d, short_d, cover_d,
                                  NS, ND, dates, C, O, H, L, V, OI, syms,
                                  max_positions=mp, sl_pct=sl, hold_max=hm,
                                  ranking='composite', factors=F)
                if r and r['ann'] > 0:
                    r['desc'] = "ENTROPY"
                    r['mp'] = mp; r['sl'] = sl; r['hm'] = hm; r['rank'] = 'composite'
                    results.append(r)
    print(f"    done ({len(results)})", flush=True)

    # ===== 策略F: 仓位势能 (多维度能量) =====
    # 简化版: OI能量40% + 趋势能量30% + 突破能量30%
    print("  [F] 仓位势能 (多维能量)", flush=True)
    buy_d = {si: set() for si in range(NS)}
    sell_d = {si: set() for si in range(NS)}
    short_d = {si: set() for si in range(NS)}
    cover_d = {si: set() for si in range(NS)}

    for si in range(NS):
        for di in range(MIN_TRAIN + 1, ND):
            om = F['oi_mom5'][si, di]
            m5 = F['mom5'][si, di]
            m10 = F['mom10'][si, di]
            vr = F['vol_ratio'][si, di]
            vdp_v = F['vdp_ema'][si, di]

            # Skip if missing data
            if np.isnan(om) or np.isnan(m5): continue

            # Energy components (normalized to [-1, 1])
            oi_energy = np.clip(om * 5, -1, 1)  # Scale OI momentum
            trend_energy = np.clip(m5 * 10, -1, 1)  # Scale price momentum

            # Breakout energy: price near Donchian high/low
            breakout_energy = 0
            dh5 = F['donch_high5'][si, di]
            dl5 = F['donch_low5'][si, di]
            if not np.isnan(dh5) and not np.isnan(dl5):
                c = C[si, di-1]
                if not np.isnan(c) and (dh5 - dl5) > 0:
                    pos_in_range = (c - dl5) / (dh5 - dl5)  # 0 to 1
                    if pos_in_range > 0.95: breakout_energy = 1  # Near high → bullish breakout
                    elif pos_in_range < 0.05: breakout_energy = -1  # Near low → bearish breakout

            # VDP energy
            vdp_energy = 0
            if not np.isnan(vdp_v):
                vdp_energy = np.clip(vdp_v / abs(vdp_v) * 0.5, -1, 1) if abs(vdp_v) > 0 else 0

            # Weighted composite
            total_energy = 0.40 * oi_energy + 0.25 * trend_energy + 0.15 * breakout_energy + 0.20 * vdp_energy

            if total_energy > 0.3:  # Strong positive energy
                buy_d[si].add(di)
            elif total_energy < -0.3:  # Strong negative energy
                short_d[si].add(di)

            # Exit when energy reverses
            if total_energy < -0.1: sell_d[si].add(di)
            if total_energy > 0.1: cover_d[si].add(di)

    for mp in [1]:
        for sl in [0.03, 0.05, 0.10]:
            for hm in [5, 7, 10, 15]:
                r = swing_backtest(buy_d, sell_d, short_d, cover_d,
                                  NS, ND, dates, C, O, H, L, V, OI, syms,
                                  max_positions=mp, sl_pct=sl, hold_max=hm,
                                  ranking='composite', factors=F)
                if r and r['ann'] > 0:
                    r['desc'] = "ENERGY"
                    r['mp'] = mp; r['sl'] = sl; r['hm'] = hm; r['rank'] = 'composite'
                    results.append(r)
    print(f"    done ({len(results)})", flush=True)

    # ====================================================================
    # Results
    # ====================================================================
    print(f"\n  完成 ({time.time()-t0:.0f}s, {len(results)} configs)", flush=True)
    results.sort(key=lambda x: -x['ann'])

    print(f"\n{'='*80}", flush=True)
    print(f"  TOP 40", flush=True)
    print(f"  {'策略':<28s} {'SL':>4s} {'HM':>3s} {'Rank':>10s} | {'Ann':>8s} {'N':>5s} {'WR':>5s} {'AvgW':>6s} {'AvgL':>5s} {'DD':>6s}", flush=True)
    for r in results[:40]:
        print(f"  {r['desc']:<28s} SL{r['sl']:.0%} H{r['hm']:>2d} {r['rank']:>10s} | "
              f"{r['ann']:+8.1f}% {r['n']:5d} {r['wr']:5.1f}% {r['avg_win']:+6.2f}% {r['avg_loss']:5.2f}% {r['max_dd']:6.1f}%", flush=True)

    for i, r in enumerate(results[:5]):
        print(f"\n  #{i+1}: {r['desc']} SL{r['sl']:.0%} H{r['hm']} {r['rank']} "
              f"(Ann={r['ann']:+.1f}%, DD={r['max_dd']:.1f}%, WR={r['wr']:.0f}%)", flush=True)
        for y in sorted(r.get('year_stats', {}).keys()):
            s = r['year_stats'][y]
            wr = s['wins'] / max(s['trades'], 1) * 100
            print(f"    {y}: {s['trades']:4d} t, WR={wr:.0f}%, pnl={s['total_pnl']:+.0f}%, abs={s['pnl_abs_sum']:+,.0f}", flush=True)

    # 按策略类型汇总
    print(f"\n  --- 按策略类型 ---", flush=True)
    seen = set()
    for r in results:
        prefix = r['desc'].split('_')[0]
        if prefix not in seen:
            seen.add(prefix)
            sub = [x for x in results if x['desc'].startswith(prefix)]
            best = sub[0]
            print(f"  {prefix:<12s}: Best={best['ann']:+.1f}% DD={best['max_dd']:.1f}% "
                  f"WR={best['wr']:.0f}% ({best['desc']} SL{best['sl']:.0%} H{best['hm']})", flush=True)

    if results:
        print(f"\n  Best: {results[0]['ann']:+.1f}% DD={results[0]['max_dd']:.1f}%", flush=True)
    print(f"  目标: 年化600%+ | 基线: +69.4% (V8f Donchian)", flush=True)
    print(f"{'='*80}", flush=True)
