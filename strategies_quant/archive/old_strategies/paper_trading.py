"""
期货模拟盘 — 多策略独立运行
============================
每个策略独立跟踪权益、持仓、盈亏。
模拟盘 ≠ 回测。从今天开始，空仓起步，逐日推进。

工作流:
  1. 创建策略: python paper_trading.py --create V121
  2. 每日运行: python paper_trading.py --run V121
     → 处理新交易日，平到期持仓，生成信号开新仓
  3. 运行全部: python paper_trading.py --run-all
  4. 查看信号: python paper_trading.py --signal V121
  5. 查看状态: python paper_trading.py --status V121
  6. 列出策略: python paper_trading.py --list

数据: futures_weighted (信号) + futures_daily (最新日)
"""
import sys, os, time, json, argparse, warnings
import itertools
import numpy as np
import pandas as pd
import talib
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alpha_v2 import load_all_data, CASH0
from contract_specs import get_multiplier, get_margin_rate

COMM = 0.0003

NAME_MAP = {
    'afi': '豆一', 'agfi': '沪银', 'alfi': '沪铝', 'aofi': '氧化铝', 'apfi': '苹果',
    'aufi': '沪金', 'bfi': '沥青', 'bcfi': '国际铜', 'brfi': '丁二烯橡胶',
    'bufi': '乙二醇', 'cffi': '玉米', 'cfi': '棉花', 'cjfi': '红枣', 'csfi': '淀粉',
    'cufi': '沪铜', 'cyfi': '棉纱', 'ebfi': '苯乙烯', 'ecfi': '集运指数', 'egfi': '乙二醇',
    'fgfi': '玻璃', 'fufi': '燃料油', 'hcfi': '热卷', 'ifi': '铁矿石',
    'jdfi': '鸡蛋', 'jfi': '焦炭', 'jmfi': '焦煤', 'jrfi': '粳稻', 'lfi': '塑料',
    'lcfi': '碳酸锂', 'lhfi': '生猪', 'lrfi': '晚籼稻', 'mafi': '甲醇', 'mfi': '豆粕',
    'nifi': '沪镍', 'lufi': '低硫燃料油', 'nrfi': '20号胶', 'oifi': '菜油', 'pfi': '棕榈油', 'pbfi': '沪铅',
    'pffi': '短纤', 'pgfi': 'LPG', 'pkfi': '花生', 'ppfi': '聚丙烯', 'rbfi': '螺纹钢',
    'rmfi': '菜粕', 'rrfi': '早籼稻', 'rufi': '橡胶', 'safi': '硅铁', 'scfi': '原油',
    'shfi': '烧碱', 'sifi': '工业硅', 'snfi': '沪锡', 'spfi': '纸浆', 'srfi': '白糖',
    'ssfi': '不锈钢', 'tafi': 'PTA', 'urfi': '尿素', 'vfi': 'PVC', 'whfi': '强麦',
    'yfi': '豆油', 'zcfi': '动煤', 'znfi': '沪锌',
}

FI_TO_DAILY = {
    'afi': 'A', 'agfi': 'AG', 'alfi': 'AL', 'aofi': 'AO', 'apfi': 'AP',
    'aufi': 'AU', 'bfi': 'BU', 'bcfi': 'BC', 'brfi': 'BR', 'bufi': 'B',
    'cffi': 'C', 'cfi': 'CF', 'cjfi': 'CJ', 'csfi': 'CS', 'cufi': 'CU',
    'cyfi': 'CY', 'ebfi': 'EB', 'ecfi': 'EC', 'egfi': 'EG', 'fgfi': 'FG',
    'fufi': 'FU', 'hcfi': 'HC', 'ifi': 'I', 'jdfi': 'JD', 'jfi': 'J',
    'jmfi': 'JM', 'jrfi': 'JR', 'lfi': 'L', 'lcfi': 'LC', 'lhfi': 'LH',
    'lrfi': 'LR', 'mafi': 'MA', 'mfi': 'M', 'nifi': 'NI', 'lufi': 'LU', 'nrfi': 'NR',
    'oifi': 'OI', 'pfi': 'P', 'pbfi': 'PB', 'pffi': 'PF', 'pgfi': 'PG',
    'pkfi': 'PK', 'ppfi': 'PP', 'rbfi': 'RB', 'rmfi': 'RM', 'rrfi': 'RR',
    'rufi': 'RU', 'safi': 'SA', 'scfi': 'SC', 'shfi': 'SH', 'sifi': 'SI',
    'smfi': 'SM', 'snfi': 'SN', 'spfi': 'SP', 'srfi': 'SR', 'ssfi': 'SS',
    'tafi': 'TA', 'urfi': 'UR', 'vfi': 'V', 'whfi': 'WH', 'yfi': 'Y',
    'zcfi': 'ZC', 'znfi': 'ZN',
}

FUTURES_DAILY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'futures_daily')
STATE_DIR = os.path.dirname(os.path.abspath(__file__))

# ===================== 策略注册表 =====================
# 每个策略定义: id, name, signal_mode, short_mode, params
STRATEGIES = {
    'V121': {
        'name': 'V121 Long-only',
        'description': 'ROC(5)>1% + Z>1.5 + ROC improving, long only',
        'signal_mode': 'v121',
        'short_mode': 'long_only',
        'atr_norm_max': 10.0,
        'max_corr': 0.5,
        'top_n': 3,
        'hold': 1,
        'initial_cash': float(CASH0),
    },
    'V121_DUAL': {
        'name': 'V121 Dual-Side',
        'description': 'V121 long + short mirror, best R/M=11.41',
        'signal_mode': 'v121',
        'short_mode': 'short_mirror',
        'atr_norm_max': 10.0,
        'max_corr': 0.5,
        'top_n': 3,
        'hold': 1,
        'initial_cash': float(CASH0),
    },
    'V121_MTF': {
        'name': 'V121 Multi-TF',
        'description': 'ROC(5)+ROC(10) confirmation, dual-side',
        'signal_mode': 'v121_mtf',
        'short_mode': 'short_mirror',
        'atr_norm_max': 10.0,
        'max_corr': 0.5,
        'top_n': 3,
        'hold': 1,
        'initial_cash': float(CASH0),
    },
}

DD_TIERS = [(0, 1.00), (0.10, 0.90), (0.20, 0.70), (0.30, 0.50)]


def state_path(strategy_id):
    return os.path.join(STATE_DIR, f'paper_trading_state_{strategy_id}.json')


def load_state(strategy_id):
    p = state_path(strategy_id)
    if os.path.exists(p):
        with open(p, 'r') as f:
            return json.load(f)
    return None


def save_state(strategy_id, state):
    with open(state_path(strategy_id), 'w') as f:
        json.dump(state, f, indent=2, ensure_ascii=False, default=float)


def list_strategies():
    """列出所有策略及其状态"""
    print(f"\n{'='*70}")
    print(f"  模拟盘策略列表")
    print(f"{'='*70}")
    for sid, cfg in STRATEGIES.items():
        state = load_state(sid)
        status = "未创建" if state is None else f"运行中 (权益={state.get('cash', 0):>10,.0f})"
        print(f"  {sid:15s} | {cfg['name']:20s} | {status}")
        print(f"  {'':15s} | {cfg['description']}")
    print(f"{'='*70}")


# ===================== 数据加载 =====================
def load_extended_data():
    """加载加权数据 + 用daily追加最新日"""
    NS, ND, dates, C, O, H, L, V, OI, syms, _ = load_all_data(load_oi=True)
    sym_to_si = {s: i for i, s in enumerate(syms)}

    try:
        rb_file = os.path.join(FUTURES_DAILY_DIR, 'RB0.csv')
        df = pd.read_csv(rb_file, nrows=2)
        daily_latest = pd.to_datetime(df['trade_date'].iloc[0], format='%Y%m%d')
        weighted_latest = dates[-1]

        if daily_latest > weighted_latest:
            all_daily_dates = set()
            for fi_sym, daily_sym in FI_TO_DAILY.items():
                df_path = os.path.join(FUTURES_DAILY_DIR, f'{daily_sym}0.csv')
                if os.path.exists(df_path):
                    d = pd.read_csv(df_path, usecols=['trade_date'])
                    d['trade_date'] = pd.to_datetime(d['trade_date'], format='%Y%m%d')
                    all_daily_dates.update(d['trade_date'].tolist())

            new_dates = sorted([d for d in all_daily_dates if d > weighted_latest])
            if new_dates:
                n_new = len(new_dates)
                print(f"  [扩展] 从futures_daily追加 {n_new} 天: {new_dates[0].date()} ~ {new_dates[-1].date()}")

                C_new = np.full((NS, ND + n_new), np.nan); C_new[:, :ND] = C
                O_new = np.full((NS, ND + n_new), np.nan); O_new[:, :ND] = O
                H_new = np.full((NS, ND + n_new), np.nan); H_new[:, :ND] = H
                L_new = np.full((NS, ND + n_new), np.nan); L_new[:, :ND] = L
                V_new = np.full((NS, ND + n_new), np.nan); V_new[:, :ND] = V
                OI_new = np.full((NS, ND + n_new), np.nan); OI_new[:, :ND] = OI

                dates_ext = list(dates) + new_dates
                dm = {d: i for i, d in enumerate(dates_ext)}

                for fi_sym, daily_sym in FI_TO_DAILY.items():
                    if fi_sym not in sym_to_si: continue
                    si = sym_to_si[fi_sym]
                    df_path = os.path.join(FUTURES_DAILY_DIR, f'{daily_sym}0.csv')
                    if not os.path.exists(df_path): continue
                    try:
                        df = pd.read_csv(df_path)
                        df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
                        for _, row in df.iterrows():
                            dt = row['trade_date']
                            if dt in dm and dm[dt] >= ND:
                                di = dm[dt]
                                if not pd.isna(row.get('close')): C_new[si, di] = float(row['close'])
                                if not pd.isna(row.get('open')): O_new[si, di] = float(row['open'])
                                if not pd.isna(row.get('high')): H_new[si, di] = float(row['high'])
                                if not pd.isna(row.get('low')): L_new[si, di] = float(row['low'])
                                if 'vol' in row and not pd.isna(row.get('vol')): V_new[si, di] = float(row['vol'])
                                if 'oi' in row and not pd.isna(row.get('oi')): OI_new[si, di] = float(row['oi'])
                    except: continue

                return NS, ND + n_new, pd.DatetimeIndex(dates_ext), C_new, O_new, H_new, L_new, V_new, OI_new, syms
    except: pass

    return NS, ND, dates, C, O, H, L, V, OI, syms


def compute_indicators(NS, ND, C, O, H, L):
    t0 = time.time()
    print("[指标] 计算...", flush=True)
    RET = np.full((NS, ND), np.nan)
    ROC5 = np.full((NS, ND), np.nan)
    ROC10 = np.full((NS, ND), np.nan)
    ROC20 = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si].astype(np.float64)
        for di in range(1, ND):
            if not np.isnan(c[di]) and not np.isnan(c[di-1]) and c[di-1] > 0:
                RET[si, di] = (c[di] / c[di-1] - 1) * 100
        ROC5[si] = talib.ROC(c, timeperiod=5)
        ROC10[si] = talib.ROC(c, timeperiod=10)
        ROC20[si] = talib.ROC(c, timeperiod=20)

    ATR14 = np.full((NS, ND), np.nan)
    ATR_NORM = np.full((NS, ND), np.nan)
    for si in range(NS):
        ATR14[si] = talib.ATR(H[si].astype(np.float64), L[si].astype(np.float64),
                               C[si].astype(np.float64), timeperiod=14)
        for di in range(ND):
            atr = ATR14[si, di]; cp = C[si, di]
            if not np.isnan(atr) and not np.isnan(cp) and cp > 0:
                ATR_NORM[si, di] = atr / cp * 100

    ZSCORE = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(21, ND):
            rets = RET[si, di-20:di]
            v = rets[~np.isnan(rets)]
            if len(v) < 10: continue
            s = np.std(v, ddof=1)
            if s > 0 and not np.isnan(RET[si, di]):
                ZSCORE[si, di] = (RET[si, di] - np.mean(v)) / s

    OV_GAP = np.full((NS, ND), np.nan)
    ID_RET = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(ND):
            o, c = O[si, di], C[si, di]
            if not np.isnan(o) and not np.isnan(c):
                if di > 0 and not np.isnan(C[si, di-1]) and C[si, di-1] > 0:
                    OV_GAP[si, di] = (o - C[si, di-1]) / C[si, di-1] * 100
                if o > 0: ID_RET[si, di] = (c - o) / o * 100

    print(f"  完成 ({time.time()-t0:.1f}s)")
    return {'RET': RET, 'ROC5': ROC5, 'ROC10': ROC10, 'ROC20': ROC20,
            'ATR14': ATR14, 'ATR_NORM': ATR_NORM, 'ZSCORE': ZSCORE,
            'OV_GAP': OV_GAP, 'ID_RET': ID_RET}


# ===================== 信号 =====================
def get_signals(di, edi, NS, ind, O, H, L, C, strategy_cfg):
    """根据策略配置生成信号"""
    ROC5 = ind['ROC5']; ROC10 = ind['ROC10']; ZSCORE = ind['ZSCORE']
    ATR_NORM = ind['ATR_NORM']; ROC20 = ind['ROC20']
    ATR14 = ind['ATR14']; OV_GAP = ind['OV_GAP']; ID_RET = ind['ID_RET']

    signal_mode = strategy_cfg['signal_mode']
    short_mode = strategy_cfg['short_mode']
    atr_max = strategy_cfg['atr_norm_max']

    # === Long signals ===
    def v121_long():
        cands = []
        for s in range(NS):
            roc = ROC5[s, di]; zs = ZSCORE[s, di]
            if np.isnan(roc) or np.isnan(zs) or roc <= 1.0 or zs <= 1.5: continue
            rp = ROC5[s, di-1] if di > 0 else np.nan
            if not np.isnan(rp) and roc <= rp: continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            cands.append((roc * zs, s, ep, 'v121'))
        return cands

    def v121_mtf_long():
        cands = []
        for s in range(NS):
            roc5 = ROC5[s, di]; roc10 = ROC10[s, di]; zs = ZSCORE[s, di]
            if any(np.isnan(x) for x in [roc5, roc10, zs]): continue
            if roc5 <= 1.0 or roc10 <= 2.0 or zs <= 1.5: continue
            rp = ROC5[s, di-1] if di > 0 else np.nan
            if not np.isnan(rp) and roc5 <= rp: continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            cands.append((roc5 * roc10 * zs / 10.0, s, ep, 'v121_mtf'))
        return cands

    def ov_id():
        cands = []
        for s in range(NS):
            ov = OV_GAP[s, di]; idr = ID_RET[s, di]; roc = ROC5[s, di]
            if any(np.isnan(x) for x in [ov, idr, roc]): continue
            if ov <= 0.3 or idr <= 0.3 or roc <= 1.0: continue
            zs = ZSCORE[s, di]; ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            zb = zs if not np.isnan(zs) and zs > 1.0 else 1.0
            cands.append(((ov + idr) * roc * zb * 2, s, ep, 'ov_id'))
        return cands

    def final_flag():
        cands = []
        for s in range(NS):
            r20 = ROC20[s, di]
            if np.isnan(r20) or r20 <= 5.0 or di < 6: continue
            h5 = H[s, di-4:di+1]; l5 = L[s, di-4:di+1]
            if any(np.isnan(x) for x in h5) or any(np.isnan(x) for x in l5): continue
            r5 = np.max(h5) - np.min(l5)
            atr = ATR14[s, di]
            if np.isnan(atr) or atr <= 0 or r5 > atr * 3.0: continue
            h4 = np.max(H[s, di-4:di])
            cp = C[s, di]
            if np.isnan(cp) or cp <= h4: continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            cands.append((r20 * (cp - h4) / atr, s, ep, 'ff'))
        return cands

    def union():
        a = {}
        for sc, s, ep, st in v121_long():
            if s not in a: a[s] = [0, ep, []]
            a[s][0] += sc * 3; a[s][2].append('v121')
        for sc, s, ep, st in ov_id():
            if s not in a: a[s] = [0, ep, []]
            a[s][0] += sc * 2; a[s][2].append('ov_id')
        for sc, s, ep, st in final_flag():
            if s not in a: a[s] = [0, ep, []]
            a[s][0] += sc; a[s][2].append('ff')
        return [(sc, s, ep, '+'.join(sigs)) for s, (sc, ep, sigs) in a.items()]

    # Choose long signal based on signal_mode
    if signal_mode == 'v121_mtf':
        long_fn = v121_mtf_long
    else:
        long_fn = v121_long

    long_cands = [c for c in long_fn() if not np.isnan(ATR_NORM[c[1], di]) and ATR_NORM[c[1], di] < atr_max]
    union_cands = [c for c in union() if not np.isnan(ATR_NORM[c[1], di]) and ATR_NORM[c[1], di] < atr_max]
    long_cands.sort(key=lambda x: -x[0])
    union_cands.sort(key=lambda x: -x[0])

    # === Short signals ===
    short_cands = []
    if short_mode != 'long_only':
        if signal_mode == 'v121_mtf':
            for s in range(NS):
                roc5 = ROC5[s, di]; roc10 = ROC10[s, di]; zs = ZSCORE[s, di]
                if any(np.isnan(x) for x in [roc5, roc10, zs]): continue
                if roc5 >= -1.0 or roc10 >= -2.0 or zs >= -1.5: continue
                rp = ROC5[s, di-1] if di > 0 else np.nan
                if not np.isnan(rp) and roc5 >= rp: continue
                ep = O[s, edi]
                if np.isnan(ep) or ep <= 0: continue
                short_cands.append((abs(roc5 * roc10 * zs) / 10.0, s, ep, 'v121_short_mtf'))
        else:
            for s in range(NS):
                roc = ROC5[s, di]; zs = ZSCORE[s, di]
                if np.isnan(roc) or np.isnan(zs) or roc >= -1.0 or zs >= -1.5: continue
                rp = ROC5[s, di-1] if di > 0 else np.nan
                if not np.isnan(rp) and roc >= rp: continue
                ep = O[s, edi]
                if np.isnan(ep) or ep <= 0: continue
                short_cands.append((abs(roc * zs), s, ep, 'v121_short'))

        short_cands = [c for c in short_cands if not np.isnan(ATR_NORM[c[1], di]) and ATR_NORM[c[1], di] < atr_max]
        short_cands.sort(key=lambda x: -x[0])

    return long_cands, union_cands, short_cands


def get_corr(sa, sb, di, RET, w=20):
    s = max(0, di - w)
    a = RET[sa, s:di]; b = RET[sb, s:di]
    v = ~(np.isnan(a) | np.isnan(b))
    if np.sum(v) < 8: return 0.5
    a = a[v]; b = b[v]
    if np.std(a) == 0 or np.std(b) == 0: return 0.5
    c = np.corrcoef(a, b)[0, 1]
    return c if not np.isnan(c) else 0.5


def dd_size(pv, hw):
    if hw <= 0: return 1.0
    dd = (pv - hw) / hw
    for t, f in DD_TIERS:
        if dd >= -t: return f
    return DD_TIERS[-1][1]


def pos_value(positions, C, di):
    pv = 0
    for p in positions:
        cp = C[p['si'], di]
        if not np.isnan(cp) and cp > 0:
            m = get_multiplier(p['sym'])
            d = p.get('dir', 1)
            unrealized = (cp - p['entry_price']) * m * p['lots'] * d
            pv += p['entry_price'] * m * abs(p['lots']) + unrealized - cp * m * abs(p['lots']) * COMM
    return pv


# ===================== 每日处理 =====================
def process_new_days(state, NS, ND, dates, C, O, H, L, syms, ind, strategy_cfg):
    """处理新交易日，返回日志"""
    sym_to_si = {s: i for i, s in enumerate(syms)}
    RET = ind['RET']

    cash = float(state['cash'])
    hw = float(state['high_water'])
    positions = list(state.get('open_positions', []))
    trades = list(state.get('trades', []))
    daily_log = list(state.get('daily_log', []))

    # 恢复持仓中的si索引
    held = []
    for p in positions:
        if p['sym'] in sym_to_si:
            p['si'] = sym_to_si[p['sym']]
            held.append(p)

    last_date = pd.Timestamp(state.get('last_date', state['start_date']))
    new_dis = [di for di in range(ND) if dates[di] > last_date]

    if not new_dis:
        print(f"  [{state['strategy_id']}] 无新数据")
        return state, []

    top_n = strategy_cfg['top_n']
    hold = strategy_cfg['hold']
    max_corr = strategy_cfg['max_corr']
    short_mode = strategy_cfg['short_mode']
    initial_cash = strategy_cfg['initial_cash']

    today_journal = []

    for di in new_dis:
        dt = str(dates[di].date())
        close_px = C[:, di]

        # 估值
        pv = cash + pos_value(held, C, di)
        if pv > hw: hw = pv

        # 平仓: hold期满
        closed_today = []
        for p in list(held):
            if di >= p['entry_di'] + p['hold_days']:
                ep = close_px[p['si']]
                if np.isnan(ep) or ep <= 0: ep = p['entry_price']
                m = get_multiplier(p['sym'])
                mr = get_margin_rate(p['sym'])
                d = p.get('dir', 1)
                pnl = (ep - p['entry_price']) * m * p['lots'] * d
                inv = p['entry_price'] * m * abs(p['lots'])
                margin_used = inv * mr
                pp = pnl / inv * 100 if inv > 0 else 0
                if d == 1:  # long: sell
                    cash += ep * m * abs(p['lots']) * (1 - COMM)
                else:  # short: buy back
                    cash += p['entry_price'] * m * abs(p['lots']) + pnl - ep * m * abs(p['lots']) * COMM
                name = NAME_MAP.get(p['sym'], p['sym'])
                trade = {
                    'close_date': dt, 'sym': p['sym'], 'name': name,
                    'entry_date': p['entry_date'],
                    'entry_price': round(p['entry_price'], 2),
                    'exit_price': round(float(ep), 2),
                    'pnl_pct': round(pp, 2), 'pnl_amount': round(float(pnl), 0),
                    'lots': p['lots'], 'sig': p['sig'], 'dir': d,
                    'margin': round(margin_used, 0),
                }
                trades.append(trade)
                closed_today.append(trade)
                held.remove(p)

        # 生成信号 & 开仓
        edi = di + 1
        opened_today = []
        if edi < ND and len(held) < top_n:
            long_c, union_c, short_c = get_signals(di, edi, NS, ind, O, H, L, C, strategy_cfg)

            best_v = next((c for c in long_c if c[1] not in {p['si'] for p in held}), None)
            best_u = next((c for c in union_c if c[1] not in {p['si'] for p in held}), None)

            entries = []  # (score, si, price, sig, pos_mult, dir)
            if best_v and best_u:
                if best_v[1] == best_u[1]:
                    entries.append((best_v[0], best_v[1], best_v[2], 'v121+union', 1.5, 1))
                else:
                    corr = get_corr(best_v[1], best_u[1], di, RET)
                    if corr < max_corr:
                        entries.append((best_v[0], best_v[1], best_v[2], 'v121', 1.0, 1))
                        entries.append((best_u[0], best_u[1], best_u[2], 'union', 1.0, 1))
                    else:
                        b = best_v if best_v[0] >= best_u[0] else best_u
                        entries.append((b[0], b[1], b[2], 'best', 1.0, 1))
            elif best_v:
                entries.append((best_v[0], best_v[1], best_v[2], 'v121', 1.0, 1))
            elif best_u:
                entries.append((best_u[0], best_u[1], best_u[2], 'union', 1.0, 1))

            # Short entries
            if short_mode != 'long_only' and len(held) + len(entries) < top_n:
                held_si = {p['si'] for p in held} | {e[1] for e in entries}
                best_short = next((c for c in short_c if c[1] not in held_si), None)
                if best_short:
                    entries.append((best_short[0], best_short[1], best_short[2], 'short', 1.0, -1))

            # Sizing
            dsz = dd_size(pv, hw)
            pos_size = max(0.05, min(0.99, dsz))

            held_si = {p['si'] for p in held}
            for sc, s, pr, sig, mult, d in entries:
                if s in held_si or len(held) >= top_n: continue
                cap = pv * pos_size * mult
                sym = syms[s]; m = get_multiplier(sym); mr = get_margin_rate(sym)
                ct = max(1, int(cap / (pr * m * (1 + COMM))))
                ci = pr * m * ct * (1 + COMM)
                if ci > cash:
                    ct = int(cash * 0.9 / (pr * m * (1 + COMM)))
                    ci = pr * m * ct * (1 + COMM) if ct > 0 else 0
                if ct <= 0 or ci <= 0 or ci > cash: continue
                cash -= ci
                name = NAME_MAP.get(sym, sym)
                entry = {
                    'si': s, 'sym': sym, 'name': name,
                    'entry_price': float(pr), 'entry_di': edi,
                    'entry_date': str(dates[edi].date()),
                    'lots': ct, 'hold_days': hold,
                    'sig': sig, 'score': float(sc), 'dir': d,
                }
                held.append(entry)
                held_si.add(s)
                margin_used = pr * m * mr * ct
                opened_today.append({
                    'open_date': str(dates[edi].date()), 'sym': sym, 'name': name,
                    'entry_price': round(float(pr), 2), 'lots': ct, 'sig': sig,
                    'dir': d, 'score': round(float(sc), 2),
                    'margin': round(margin_used, 0),
                })

        # 记录当日
        pv_end = cash + pos_value(held, C, di)
        day_pnl = sum(t['pnl_amount'] for t in closed_today)
        daily_log.append({
            'date': dt,
            'equity': round(pv_end, 2),
            'day_pnl': round(day_pnl, 0),
            'closed': len(closed_today),
            'opened': len(opened_today),
            'open_positions': len(held),
        })

        today_journal.append({
            'date': dt,
            'closed': closed_today,
            'opened': opened_today,
            'equity': round(pv_end, 2),
            'day_pnl': round(day_pnl, 0),
        })

    # 保存持仓 (去掉si)
    positions_save = []
    for p in held:
        positions_save.append({
            'sym': p['sym'], 'name': p.get('name', p['sym']),
            'entry_price': p['entry_price'], 'entry_di': p['entry_di'],
            'entry_date': p['entry_date'], 'lots': p['lots'],
            'hold_days': p['hold_days'], 'sig': p['sig'],
            'score': p.get('score', 0), 'dir': p.get('dir', 1),
            'margin_rate': get_margin_rate(p['sym']),
        })

    last_dt = str(dates[new_dis[-1]].date())
    state.update({
        'last_date': last_dt,
        'cash': round(cash, 2),
        'high_water': round(hw, 2),
        'open_positions': positions_save,
        'trades': trades,
        'daily_log': daily_log,
    })
    return state, today_journal


# ===================== API接口 =====================
def get_all_status():
    """返回所有策略状态 (供API调用)"""
    results = {}
    for sid, cfg in STRATEGIES.items():
        state = load_state(sid)
        if state:
            results[sid] = {
                'id': sid,
                'name': cfg['name'],
                'description': cfg['description'],
                'start_date': state.get('start_date'),
                'last_date': state.get('last_date'),
                'cash': state.get('cash', 0),
                'high_water': state.get('high_water', 0),
                'open_positions': state.get('open_positions', []),
                'trades': state.get('trades', []),
                'daily_log': state.get('daily_log', []),
                'initial_cash': cfg['initial_cash'],
                'n_trades': len(state.get('trades', [])),
            }
    return results


# ===================== CLI =====================
def main():
    parser = argparse.ArgumentParser(description='期货模拟盘 — 多策略独立运行')
    parser.add_argument('--list', action='store_true', help='列出所有策略')
    parser.add_argument('--create', metavar='ID', help='创建策略 (从今天开始)')
    parser.add_argument('--run', metavar='ID', help='运行指定策略')
    parser.add_argument('--run-all', action='store_true', help='运行所有已创建的策略')
    parser.add_argument('--signal', metavar='ID', help='显示最新信号')
    parser.add_argument('--status', metavar='ID', help='显示策略状态')
    parser.add_argument('--report', metavar='ID', help='详细报告')
    parser.add_argument('--delete', metavar='ID', help='删除策略')
    args = parser.parse_args()

    # 列出策略
    if args.list:
        list_strategies()
        return

    # 创建策略 (从今天开始，空仓)
    if args.create:
        sid = args.create.upper()
        if sid not in STRATEGIES:
            print(f"[错误] 未知策略: {sid}")
            print(f"  可用策略: {', '.join(STRATEGIES.keys())}")
            return

        state = load_state(sid)
        if state is not None:
            print(f"[错误] 策略 {sid} 已存在。先 --delete {sid} 再重新创建")
            return

        # 找到最新交易日作为起始日
        NS, ND, dates, C, O, H, L, V, OI, syms = load_extended_data()
        start_date = str(dates[-1].date())
        cfg = STRATEGIES[sid]

        state = {
            'strategy_id': sid,
            'strategy_name': cfg['name'],
            'start_date': start_date,
            'last_date': start_date,
            'cash': cfg['initial_cash'],
            'high_water': cfg['initial_cash'],
            'open_positions': [],
            'trades': [],
            'daily_log': [],
        }
        save_state(sid, state)
        print(f"[创建] 策略 {sid} ({cfg['name']})")
        print(f"  起始日: {start_date}")
        print(f"  初始资金: {cfg['initial_cash']:,.0f}")
        print(f"  信号模式: {cfg['signal_mode']}")
        print(f"  做空模式: {cfg['short_mode']}")
        print(f"  下一步: python paper_trading.py --run {sid}")
        return

    # 删除策略
    if args.delete:
        sid = args.delete.upper()
        p = state_path(sid)
        if os.path.exists(p):
            os.remove(p)
            print(f"[删除] 策略 {sid} 已删除")
        else:
            print(f"[错误] 策略 {sid} 不存在")
        return

    # 加载数据 (运行/信号/报告都需要)
    need_data = any([args.run, args.run_all, args.signal, args.status, args.report])
    NS = ND = dates = C = O = H = L = syms = ind = None
    if need_data:
        NS, ND, dates, C, O, H, L, V, OI, syms = load_extended_data()
        ind = compute_indicators(NS, ND, C, O, H, L)

    # 显示信号
    if args.signal:
        sid = args.signal.upper()
        if sid not in STRATEGIES:
            print(f"[错误] 未知策略: {sid}"); return
        cfg = STRATEGIES[sid]
        di = ND - 2; edi = ND - 1
        if di < 30: print("[数据不足]"); return
        print(f"\n  策略: {cfg['name']} | 信号日: {dates[di].date()} -> 执行日: {dates[edi].date()}")
        long_c, union_c, short_c = get_signals(di, edi, NS, ind, O, H, L, C, cfg)
        if long_c:
            print(f"\n  做多候选:")
            for sc, s, ep, sig in long_c[:5]:
                sym = syms[s]; name = NAME_MAP.get(sym, sym)
                print(f"    {name:8s} | 得分={sc:.2f} | ROC5={ind['ROC5'][s,di]:+.2f}% | Z={ind['ZSCORE'][s,di]:.2f}")
        else:
            print(f"\n  做多: (无信号)")
        if short_c:
            print(f"\n  做空候选:")
            for sc, s, ep, sig in short_c[:5]:
                sym = syms[s]; name = NAME_MAP.get(sym, sym)
                print(f"    {name:8s} | 得分={sc:.2f} | ROC5={ind['ROC5'][s,di]:+.2f}% | Z={ind['ZSCORE'][s,di]:.2f}")
        else:
            print(f"\n  做空: (无信号)")
        return

    # 运行策略
    if args.run:
        sid = args.run.upper()
        state = load_state(sid)
        if state is None:
            print(f"[错误] 策略 {sid} 未创建。先: python paper_trading.py --create {sid}")
            return
        cfg = STRATEGIES[sid]
        print(f"\n[运行] 策略 {sid} ({cfg['name']})")
        state, journal = process_new_days(state, NS, ND, dates, C, O, H, L, syms, ind, cfg)
        save_state(sid, state)
        if journal:
            for day in journal[-3:]:
                print(f"  {day['date']}  权益: {day['equity']:>12,.0f}  盈亏: {day['day_pnl']:>+10,.0f}")
                for t in day.get('closed', []):
                    d_str = '多' if t.get('dir', 1) == 1 else '空'
                    print(f"    平{d_str} {t['name']:6s} {t['pnl_pct']:>+6.1f}% ({t['pnl_amount']:>+8,.0f}元)")
                for t in day.get('opened', []):
                    d_str = '多' if t.get('dir', 1) == 1 else '空'
                    print(f"    开{d_str} {t['name']:6s} 价格={t['entry_price']:>8.1f} {t['lots']}手")
        print(f"  现金: {state['cash']:>12,.0f} | 持仓: {len(state.get('open_positions', []))}个")
        return

    # 运行全部
    if args.run_all:
        for sid, cfg in STRATEGIES.items():
            state = load_state(sid)
            if state is None: continue
            print(f"\n[运行] {sid} ({cfg['name']})")
            state, journal = process_new_days(state, NS, ND, dates, C, O, H, L, syms, ind, cfg)
            save_state(sid, state)
            if journal:
                for day in journal[-3:]:
                    print(f"  {day['date']}  权益: {day['equity']:>12,.0f}  盈亏: {day['day_pnl']:>+10,.0f}")
            print(f"  现金: {state['cash']:>12,.0f}")
        return

    # 显示状态
    if args.status:
        sid = args.status.upper()
        state = load_state(sid)
        if state is None:
            print(f"[错误] 策略 {sid} 未创建"); return
        cfg = STRATEGIES[sid]
        initial = cfg['initial_cash']
        ret = (state['cash'] / initial - 1) * 100
        print(f"\n  策略: {cfg['name']} ({sid})")
        print(f"  起始: {state['start_date']} | 最新: {state.get('last_date', 'N/A')}")
        print(f"  现金: {state['cash']:>12,.0f} | 收益: {ret:>+.1f}%")
        positions = state.get('open_positions', [])
        if positions:
            print(f"\n  持仓 ({len(positions)}个):")
            for p in positions:
                d_str = '多' if p.get('dir', 1) == 1 else '空'
                print(f"    {d_str} {p.get('name', p['sym']):6s} | {p['entry_price']:>8.1f} | {p['lots']}手 | {p['sig']}")
        trades = state.get('trades', [])
        if trades:
            wr = len([t for t in trades if t['pnl_pct'] > 0]) / len(trades) * 100
            print(f"\n  {len(trades)}笔 | WR={wr:.1f}%")
        return

    # 详细报告
    if args.report:
        sid = args.report.upper()
        state = load_state(sid)
        if state is None:
            print(f"[错误] 策略 {sid} 未创建"); return
        cfg = STRATEGIES[sid]
        initial = cfg['initial_cash']
        trades = state.get('trades', [])
        daily_log = state.get('daily_log', [])

        print(f"\n{'='*70}")
        print(f"  策略报告: {cfg['name']} ({sid})")
        print(f"{'='*70}")
        print(f"  区间: {state['start_date']} ~ {state.get('last_date', 'N/A')}")
        print(f"  初始: {initial:,.0f} | 现金: {state['cash']:,.0f} | 收益: {(state['cash']/initial-1)*100:>+.1f}%")

        if daily_log:
            eq = [d['equity'] for d in daily_log]
            peaks = list(itertools.accumulate(eq, max))
            mdd = min((e - p) / p * 100 for e, p in zip(eq, peaks))
            print(f"  MDD: {mdd:>+.1f}%")

        if trades:
            wins = [t for t in trades if t['pnl_pct'] > 0]
            wr = len(wins) / len(trades) * 100
            print(f"  {len(trades)}笔 | WR={wr:.1f}%")

            print(f"\n  最近10笔:")
            for t in reversed(trades[-10:]):
                d_str = '多' if t.get('dir', 1) == 1 else '空'
                print(f"  {t['close_date']} | {d_str}{t['name']:6s} | {t['entry_price']:>8.1f}->{t['exit_price']:>8.1f} | {t['pnl_pct']:>+6.1f}%")

        print(f"{'='*70}")
        return

    # 无参数 → 显示帮助
    parser.print_help()


if __name__ == '__main__':
    main()
