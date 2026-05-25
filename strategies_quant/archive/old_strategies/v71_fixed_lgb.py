"""
V71 — Bug-Fixed LightGBM Ranking
==================================
修复:
  1. 策略去重: 59→43个唯一策略 (移除15个重复+1对重复)
  2. 训练leakage: train_end = train_di - FWD_DAYS - 1
  3. 内置bug check: random baseline, 年度分解
  4. 无rev退出条件
"""
import sys, os, time, warnings, pickle
import numpy as np, pandas as pd
import lightgbm as lgb
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from core.data_loader import list_available_symbols, load_stock_data

COMMISSION = 0.0003; STAMP_DUTY = 0.001; CASH0 = 500_000

# ============================================================
# [0] 策略去重 — 只保留唯一信号
# ============================================================
print("=" * 70, flush=True)
print("  V71 — Bug-Fixed LightGBM Ranking", flush=True)
print("=" * 70, flush=True)

# 先加载信号做去重
print("\n[0] Deduplicating strategies...", flush=True)
with open('.v15_7_signals_fixed.pkl','rb') as f: all_signals=pickle.load(f)

USE_STRATS_RAW = {
    'AggressiveMAStrategy', 'BalancedMAStrategy', 'CSVAutoSelectAdapter',
    'CSVPriceActionAdapter', 'ClusterV2Strategy', 'ClusterV3Strategy',
    'CompGStrategy', 'ConservativeMAStrategy', 'EnergtStructureStrategy',
    'EnsembleKERGatedStrategy', 'EnsembleNWVTStrategy',
    'EnsembleNWVolumeStrategy', 'EnsemblePSTVStrategy',
    'EnsembleRegressionVolumeStrategy', 'EnsembleStructureVolumeStrategy',
    'EnsembleSTVHurstStrategy', 'FramaStrategy', 'HanningFIRStrategy',
    'IndexStrategy', 'IRLSStrategy', 'KlineStrategy',
    'MACDMomentumStrategy', 'MaCompensatStrategy',
    'MathAnalysisStrategy', 'LineRegressionBandStrategy',
    'MeanReversionMAStrategy', 'PeakExStrategy', 'PeakStrategy',
    'PriceVolIntStrategy', 'ProportionalVolumeSplitStrategy',
    'PriceVolIntV2Strategy', 'RSIDivergenceStrategy', 'RectangleStrategy',
    'ReflectWaveStrategy', 'RegressionCandlestickStrategy',
    'SimpleMovingAverageStrategy', 'SpikeBakeStrategy',
    'StdgStrategy', 'TradingViewStrategy', 'TrendFollowingMAStrategy',
    'VarStrategy', 'VisualStrategy', 'VolatilityTerrainStrategy',
    'VolumeDeltaPressureStrategy', 'WaveClusterStrategy',
    'WaveCoxStrategy', 'WaveDtwStrategy', 'WaveletStrategy',
    'XgboostStrategy', 'ZonePivotStrategy', 'ClusterStrategy',
    'FutureFilterStrategy', 'FutureFiltetV2Strategy', 'MaStrategyAdapterStrategy',
    'MultiMethodWaveStrategy',
    'StategyLineregressionStrategy', 'StategyMomentumStrategy',
    'StategyRectangleStrategy', 'ThreeDStrategy',
}

avail_all = [s for s in USE_STRATS_RAW if s in all_signals and len(all_signals[s]) >= 100]

# 去重: 基于信号hash
from collections import defaultdict
strat_hash = {}
for name in avail_all:
    h = []
    for sym in sorted(all_signals[name].keys())[:10]:
        for t, a, p in all_signals[name][sym][:5]:
            h.append((str(t)[:10], a, round(p,2)))
    strat_hash[name] = tuple(h)

seen_hashes = set()
USE_STRATS = []
deduped_out = []
for name in avail_all:
    h = strat_hash[name]
    if h not in seen_hashes:
        seen_hashes.add(h)
        USE_STRATS.append(name)
    else:
        deduped_out.append(name)

print(f"  Raw: {len(avail_all)} strategies", flush=True)
print(f"  After dedup: {len(USE_STRATS)} unique strategies", flush=True)
print(f"  Removed {len(deduped_out)} duplicates: {deduped_out[:5]}...", flush=True)

# ============================================================
# [1] 数据加载
# ============================================================
print("\n[1] Loading data...", flush=True)
t0 = time.time()

stock_data = {}
for sym in list_available_symbols('daily'):
    try:
        df = load_stock_data(sym, frequency='daily')
        if df is not None and len(df) >= 300:
            cols = [c for c in ['open','high','low','close','vol','volume','amount'] if c in df.columns]
            stock_data[sym] = df[cols].copy()
            if 'vol' in df.columns and 'volume' not in df.columns:
                stock_data[sym].rename(columns={'vol': 'volume'}, inplace=True)
    except: pass

# Use cutoff date for volume calculation to ensure stable stock universe
CUTOFF_DATE = pd.Timestamp('2026-01-19')
vol_map = {}
for s, df in stock_data.items():
    if 'volume' not in df.columns: continue
    df_before = df[df.index <= CUTOFF_DATE]
    if len(df_before) < 60: continue
    avg_vol = df_before['volume'].tail(60).mean()
    if avg_vol > 0: vol_map[s] = avg_vol
syms = sorted([s for s, _ in sorted(vol_map.items(), key=lambda x: -x[1])[:500]])
NS = len(syms)
all_dates = sorted(set(d for s in syms for d in stock_data[s].index))
i0 = next(i for i, d in enumerate(all_dates) if d >= pd.Timestamp('2016-01-01'))
i1 = next((i for i, d in enumerate(all_dates) if d > CUTOFF_DATE), len(all_dates)) - 1
dates = all_dates[i0:i1+1]; ND = len(dates); dm = {d: i for i, d in enumerate(all_dates)}

C = np.full((NS, len(all_dates)), np.nan)
O = np.full((NS, len(all_dates)), np.nan)
H = np.full((NS, len(all_dates)), np.nan)
L = np.full((NS, len(all_dates)), np.nan)
V = np.full((NS, len(all_dates)), np.nan)
for si, s in enumerate(syms):
    df = stock_data.get(s)
    if df is None: continue
    for d in df.index:
        if d in dm:
            di = dm[d]
            if 'close' in df.columns: C[si, di] = float(df.loc[d, 'close'])
            if 'open' in df.columns: O[si, di] = float(df.loc[d, 'open'])
            if 'high' in df.columns: H[si, di] = float(df.loc[d, 'high'])
            if 'low' in df.columns: L[si, di] = float(df.loc[d, 'low'])
            if 'volume' in df.columns: V[si, di] = float(df.loc[d, 'volume'])
C=C[:,i0:i1+1]; O=O[:,i0:i1+1]; H=H[:,i0:i1+1]; L=L[:,i0:i1+1]; V=V[:,i0:i1+1]
print(f"  {NS} stocks, {ND} days ({time.time()-t0:.1f}s)", flush=True)

# [2] 策略信号
print("[2] Loading strategy signals (deduped)...", flush=True)
N_STRAT = len(USE_STRATS)
date_to_di = {d: i for i, d in enumerate(dates)}
STRAT_BUY = np.zeros((NS, ND, N_STRAT), dtype=np.int8)
STRAT_SELL = np.zeros((NS, ND, N_STRAT), dtype=np.int8)
for ki, sname in enumerate(USE_STRATS):
    for sym, sig_list in all_signals[sname].items():
        if sym not in syms: continue
        si = syms.index(sym)
        for ts, action, price in sig_list:
            if ts in date_to_di:
                di = date_to_di[ts]
                if action == 'buy': STRAT_BUY[si, di, ki] = 1
                elif action == 'sell': STRAT_SELL[si, di, ki] = 1
print(f"  {N_STRAT} unique strategies loaded", flush=True)

# [3] 特征计算
print("[3] Computing features...", flush=True)
t2 = time.time()

MOM5 = np.full_like(C, np.nan); MOM10 = np.full_like(C, np.nan); MOM20 = np.full_like(C, np.nan)
for si in range(NS):
    for di in range(20, ND):
        if np.isnan(C[si,di]): continue
        if not np.isnan(C[si,di-5]) and C[si,di-5]>0: MOM5[si,di] = (C[si,di]-C[si,di-5])/C[si,di-5]
        if not np.isnan(C[si,di-10]) and C[si,di-10]>0: MOM10[si,di] = (C[si,di]-C[si,di-10])/C[si,di-10]
        if not np.isnan(C[si,di-20]) and C[si,di-20]>0: MOM20[si,di] = (C[si,di]-C[si,di-20])/C[si,di-20]

PRICE_PCT = np.full_like(C, np.nan)
for si in range(NS):
    for di in range(60, ND):
        vals = C[si, di-60:di+1]; valid = vals[~np.isnan(vals)]
        if len(valid) < 30: continue
        cur = C[si,di]
        if np.isnan(cur): continue
        PRICE_PCT[si,di] = np.sum(valid < cur) / max(len(valid)-1, 1) * 100

EMA_P = 10; a_ema = 2.0/(EMA_P+1)
VDP_DELTA = np.full_like(C, np.nan)
for si in range(NS):
    ema_val = np.nan
    for di in range(1, ND):
        if np.isnan(V[si,di]) or V[si,di]<=0: continue
        if np.isnan(C[si,di]) or np.isnan(H[si,di]) or np.isnan(L[si,di]): continue
        hl = H[si,di]-L[si,di]
        if hl <= 0:
            delta = V[si,di] if C[si,di]>=H[si,di] else -V[si,di] if C[si,di]<=L[si,di] else None
            if delta is None: continue
        else:
            delta = V[si,di]*(2*C[si,di]-H[si,di]-L[si,di])/hl
        ema_val = delta if np.isnan(ema_val) else a_ema*delta+(1-a_ema)*ema_val
        VDP_DELTA[si,di] = ema_val

REL_VOL = np.full_like(C, np.nan)
for si in range(NS):
    for di in range(20, ND):
        if np.isnan(V[si,di]) or V[si,di]<=0: continue
        v20 = V[si, di-20:di]; v20v = v20[~np.isnan(v20)]
        if len(v20v) < 10: continue
        avg_v = np.mean(v20v)
        if avg_v > 0: REL_VOL[si,di] = V[si,di] / avg_v

BB_WIDTH = np.full_like(C, np.nan); ATR_PCT = np.full_like(C, np.nan)
for si in range(NS):
    for di in range(20, ND):
        c20 = C[si, di-20:di+1]; valid = c20[~np.isnan(c20)]
        if len(valid) < 15: continue
        ma = np.mean(valid); std = np.std(valid)
        if ma > 0 and std > 0: BB_WIDTH[si,di] = (4*std)/ma * 100
        if di < 2: continue
        atr_vals = []
        for dd in range(max(di-14,1), di+1):
            if not np.isnan(H[si,dd]) and not np.isnan(L[si,dd]):
                tr = H[si,dd]-L[si,dd]
                if not np.isnan(C[si,dd-1]):
                    tr = max(tr, abs(H[si,dd]-C[si,dd-1]), abs(L[si,dd]-C[si,dd-1]))
                atr_vals.append(tr)
        if len(atr_vals) >= 5 and not np.isnan(C[si,di]) and C[si,di] > 0:
            ATR_PCT[si,di] = np.mean(atr_vals) / C[si,di] * 100

def rank_pct(arr, start=60):
    res = np.full_like(arr, np.nan)
    for di in range(start, arr.shape[1]):
        vals = arr[:,di]; mask = ~np.isnan(vals)
        if mask.sum() < 50: continue
        ranked = np.argsort(np.argsort(vals[mask])).astype(float)
        n = len(ranked); pct = ranked/max(n-1,1)*100
        for k, idx in enumerate(np.where(mask)[0]): res[idx,di] = pct[k]
    return res

R_MOM5 = rank_pct(MOM5); R_MOM10 = rank_pct(MOM10); R_MOM20 = rank_pct(MOM20)
R_PRICE = rank_pct(PRICE_PCT); R_VDP = rank_pct(VDP_DELTA)
R_REL_VOL = rank_pct(REL_VOL); R_BB = rank_pct(BB_WIDTH); R_ATR = rank_pct(ATR_PCT)

def delta_rank(arr, lag=3):
    res = np.full_like(arr, np.nan)
    for di in range(lag, arr.shape[1]):
        for si in range(arr.shape[0]):
            if not np.isnan(arr[si,di]) and not np.isnan(arr[si,di-lag]):
                res[si,di] = arr[si,di] - arr[si,di-lag]
    return res

D_MOM5_3 = delta_rank(R_MOM5, 3); D_MOM10_5 = delta_rank(R_MOM10, 5)
D_MOM20_10 = delta_rank(R_MOM20, 10); D_PRICE_5 = delta_rank(R_PRICE, 5)
D_VDP_5 = delta_rank(R_VDP, 5); D_REL_VOL_5 = delta_rank(R_REL_VOL, 5)
D_BB_5 = delta_rank(R_BB, 5); D_ATR_5 = delta_rank(R_ATR, 5)

MKT_BREADTH = np.full(ND, np.nan); MKT_MOM20_VAL = np.full(ND, np.nan)
for di in range(20, ND):
    above = sum(1 for si in range(NS)
                if not np.isnan(C[si,di]) and not np.isnan(C[si,di-20]) and C[si,di-20]>0
                and (C[si,di]-C[si,di-20])/C[si,di-20] > 0)
    total = sum(1 for si in range(NS) if not np.isnan(C[si,di]) and not np.isnan(C[si,di-20]) and C[si,di-20]>0)
    if total > 100: MKT_BREADTH[di] = above / total * 100
    r20 = [C[si,di]/C[si,di-20]-1 for si in range(NS)
           if not np.isnan(C[si,di]) and not np.isnan(C[si,di-20]) and C[si,di-20]>0]
    if len(r20) > 100: MKT_MOM20_VAL[di] = np.mean(r20) * 100

print(f"  Features done ({time.time()-t2:.1f}s)", flush=True)

# [4] 特征矩阵
N_HAND = 20
N_FEAT = N_HAND + 2 * N_STRAT
FEAT = np.full((NS, ND, N_FEAT), np.nan)
FWD_RET = np.full((NS, ND), np.nan)
for si in range(NS):
    for di in range(60, ND):
        fi = 0
        for feat in [D_MOM5_3, D_MOM10_5, D_MOM20_10, D_PRICE_5, D_VDP_5,
                     D_REL_VOL_5, D_BB_5, D_ATR_5,
                     R_MOM5, R_MOM10, R_MOM20, R_PRICE, R_VDP, R_REL_VOL, R_BB, R_ATR]:
            FEAT[si,di,fi] = feat[si,di]; fi += 1
        BUY_CON = STRAT_BUY[si,di].sum(); SELL_CON = STRAT_SELL[si,di].sum()
        FEAT[si,di,fi] = BUY_CON; fi += 1
        FEAT[si,di,fi] = SELL_CON; fi += 1
        if not np.isnan(MKT_BREADTH[di]): FEAT[si,di,fi] = MKT_BREADTH[di]
        fi += 1
        if not np.isnan(MKT_MOM20_VAL[di]): FEAT[si,di,fi] = MKT_MOM20_VAL[di]
        fi += 1
        for ki in range(N_STRAT):
            FEAT[si,di,fi] = STRAT_BUY[si,di,ki]; fi += 1
        for ki in range(N_STRAT):
            FEAT[si,di,fi] = STRAT_SELL[si,di,ki]; fi += 1
        if di < ND - 5 and not np.isnan(C[si,di]) and not np.isnan(C[si,di+5]) and C[si,di]>0:
            FWD_RET[si,di] = (C[si,di+5]-C[si,di])/C[si,di]*100

print(f"  Feature matrix: ({NS}, {ND}, {N_FEAT}) — was {N_HAND+2*59}, now deduped", flush=True)

# ============================================================
# [5] 训练 — 修复leakage: train_end = train_di - FWD_DAYS - 1
# ============================================================
TRAIN_WINDOW = 252 * 3; RETRAIN_FREQ = 42; MIN_TRAIN = 252 * 2
FWD_DAYS = 5

print(f"\n[5] Training (leakage-fixed)...", flush=True)
t5 = time.time()

MODELS = [
    ('ultra_conserv', {
        'objective': 'lambdarank', 'metric': 'ndcg',
        'learning_rate': 0.05, 'num_leaves': 7,
        'feature_fraction': 0.4, 'bagging_fraction': 0.7,
        'bagging_freq': 3, 'min_data_in_leaf': 100,
        'label_gain': [1, 2, 3, 4, 5], 'verbose': -1, 'n_jobs': -1,
    }, 300),
    ('ultra_aggressive', {
        'objective': 'lambdarank', 'metric': 'ndcg',
        'learning_rate': 0.1, 'num_leaves': 127,
        'feature_fraction': 0.9, 'bagging_fraction': 0.9,
        'bagging_freq': 3, 'min_data_in_leaf': 20,
        'label_gain': [1, 2, 3, 4, 5], 'verbose': -1, 'n_jobs': -1,
    }, 100),
]

all_scores = {}
for m_name, params, n_rounds in MODELS:
    print(f"\n  Training: {m_name}...", flush=True)
    SCORE = np.full((NS, ND), np.nan)
    n_trains = 0
    for train_di in range(MIN_TRAIN, ND, RETRAIN_FREQ):
        # FIX: train_end = train_di - FWD_DAYS - 1 (was train_di - FWD_DAYS)
        train_end = train_di - FWD_DAYS - 1
        train_start = max(MIN_TRAIN - 100, train_end - TRAIN_WINDOW)
        if train_start >= train_end: continue

        train_X = []; train_y = []; train_group = []
        for di in range(train_start, train_end + 1):
            day_X = []; day_y = []
            for si in range(NS):
                f = FEAT[si, di]
                if np.any(np.isnan(f)): continue
                if np.isnan(FWD_RET[si, di]): continue
                day_X.append(f); day_y.append(FWD_RET[si, di])
            if len(day_X) >= 50:
                train_X.extend(day_X); train_y.extend(day_y)
                train_group.append(len(day_X))

        if len(train_X) < 2000: continue
        tX = np.array(train_X, dtype=np.float32)
        ty = np.array(train_y, dtype=np.float32)
        rank_y = np.zeros(len(ty), dtype=np.int32)
        offset = 0
        for g in train_group:
            grp_ret = ty[offset:offset+g]
            order = np.argsort(grp_ret)
            for bucket, idx in enumerate(order):
                rank_y[offset + idx] = min(int(bucket * 5 / g), 4)
            offset += g
        train_data = lgb.Dataset(tX, label=rank_y, group=train_group)
        model = lgb.train(params, train_data, num_boost_round=n_rounds)
        n_trains += 1

        pred_end = min(train_di + RETRAIN_FREQ, ND)
        for di in range(train_di, pred_end):
            pred_list = []; pred_si = []
            for si in range(NS):
                f = FEAT[si, di]
                if np.any(np.isnan(f)): continue
                if np.isnan(C[si,di]) or C[si,di]<=0: continue
                pred_list.append(f); pred_si.append(si)
            if pred_list:
                scores = model.predict(np.array(pred_list, dtype=np.float32))
                for k, si in enumerate(pred_si):
                    SCORE[si, di] = scores[k]

        if n_trains % 10 == 0:
            print(f"    #{n_trains}: di={train_di}", flush=True)
    print(f"    Done: {n_trains} trainings", flush=True)
    all_scores[m_name] = SCORE

# Feature importance (ultra_conserv last model)
print(f"\n  Feature importance (last ultra_conserv model):", flush=True)
importance = model.feature_importance(importance_type='gain')
feat_names = (
    ['D_MOM5_3','D_MOM10_5','D_MOM20_10','D_PRICE_5','D_VDP_5',
     'D_REL_VOL_5','D_BB_5','D_ATR_5',
     'R_MOM5','R_MOM10','R_MOM20','R_PRICE','R_VDP','R_REL_VOL','R_BB','R_ATR',
     'BUY_CON','SELL_CON','MKT_BREADTH','MKT_MOM20'] +
    [f'BUY_{s[:12]}' for s in USE_STRATS] +
    [f'SELL_{s[:12]}' for s in USE_STRATS]
)
top_idx = np.argsort(importance)[::-1][:15]
for i in top_idx:
    name = feat_names[i] if i < len(feat_names) else f'feat_{i}'
    print(f"    {name:20s}: {importance[i]:.1f}", flush=True)

print(f"\n  All models trained ({time.time()-t5:.1f}s)", flush=True)

# ============================================================
# [6] Backtest engine
# ============================================================
def run_backtest(score_arr, sl_pct, tp_pct, hold_max, trail_pct,
                 trail_start=5, score_thresh=-999):
    cash = float(CASH0); pos = None; trades = []; pending = None
    for di in range(MIN_TRAIN, ND):
        if pending is not None:
            pt = pending[0]
            if pt == 'close' and pos is not None:
                p = O[pos['si'], di]
                if np.isnan(p) or p <= 0: p = C[pos['si'], di]
                if not np.isnan(p) and p > 0:
                    pnl = (p - pos['entry']) / pos['entry'] * 100
                    cash += pos['shares'] * p * (1 - COMMISSION - STAMP_DUTY)
                    trades.append({'pnl': pnl, 'days': (dates[di]-pos['ed']).days,
                                  'reason': pending[1], 'di': di})
                    pos = None
            elif pt == 'open_long' and pos is None:
                si = pending[1]; p = O[si, di]
                if np.isnan(p) or p <= 0:
                    p = C[si, di-1] if di > 0 and not np.isnan(C[si, di-1]) else np.nan
                if not np.isnan(p) and p > 0 and cash > 10000:
                    shares = int(cash / (1 + COMMISSION) / p)
                    if shares > 0:
                        cash -= shares * p * (1 + COMMISSION)
                        pos = {'si': si, 'shares': shares, 'entry': p,
                               'highest': p, 'ed': dates[di]}
            pending = None

        if pos is not None:
            si = pos['si']; p = C[si, di]
            if not np.isnan(p):
                if p > pos['highest']: pos['highest'] = p
                pnl = (p - pos['entry']) / pos['entry'] * 100
                hd = (dates[di] - pos['ed']).days
                if hd <= 2: sl_eff = sl_pct * 1.3
                elif hd <= 5: sl_eff = sl_pct
                else: sl_eff = sl_pct * 0.7
                er = None
                if pnl < -sl_eff: er = f'sl({pnl:.1f}%)'
                elif pnl > tp_pct: er = f'tp({pnl:.1f}%)'
                elif trail_pct > 0 and pnl > trail_start:
                    dd = (pos['highest'] - p) / pos['highest'] * 100
                    if dd > trail_pct: er = f'trail({pnl:.1f}%)'
                elif hold_max > 0 and hd >= hold_max: er = f'max({hd}d)'
                if er: pending = ('close', er)

        if pos is None and pending is None:
            best_si = -1; best_score = -1e9
            for si in range(NS):
                s = score_arr[si, di]
                if np.isnan(s): continue
                if s > best_score: best_score = s; best_si = si
            if best_si >= 0 and best_score > score_thresh:
                pending = ('open_long', best_si)

    if pos is not None:
        p = C[pos['si'], ND-1]
        if not np.isnan(p) and p > 0:
            pnl = (p - pos['entry']) / pos['entry'] * 100
            cash += pos['shares'] * p * (1 - COMMISSION - STAMP_DUTY)
            trades.append({'pnl': pnl, 'days': 999, 'reason': 'end', 'di': ND-1})
    if cash <= 0 or not trades: return None
    days = (dates[ND-1] - dates[MIN_TRAIN]).days; yr = max(days/365.25, 0.01)
    ann = ((cash/CASH0)**(1/yr)-1)*100
    nw = sum(1 for t in trades if t['pnl'] > 0); wr = nw/max(len(trades),1)*100
    avg_w = np.mean([t['pnl'] for t in trades if t['pnl'] > 0]) if nw > 0 else 0
    avg_l = np.mean([abs(t['pnl']) for t in trades if t['pnl'] <= 0]) if nw < len(trades) else 0
    edge = (nw/max(len(trades),1)) * avg_w - (1-nw/max(len(trades),1)) * avg_l
    # Max drawdown
    eq = [CASH0]
    for t in trades: eq.append(eq[-1] * (1 + t['pnl']/100))
    peak = eq[0]; max_dd = 0
    for v in eq:
        if v > peak: peak = v
        dd = (peak - v) / peak * 100
        if dd > max_dd: max_dd = dd
    return {'ann': round(ann,1), 'n': len(trades), 'wr': round(wr,1),
            'avg_w': round(avg_w,1), 'avg_l': round(avg_l,1),
            'edge': round(edge,2), 'tpy': round(len(trades)/yr,1),
            'max_dd': round(max_dd,1), 'final': round(cash,0), 'trades': trades}

# ============================================================
# [7] 搜索
# ============================================================
print(f"\n[7] Parameter search...", flush=True)
results = {}

for ens_name, ens_score in all_scores.items():
    model_results = []
    for hm in [5, 6, 7, 8, 10, 12]:
        for sl in [4, 5, 6, 7, 8, 10, 12]:
            for trail in [0, 1, 2, 3]:
                for ts in [3, 4, 5, 6]:
                    if trail == 0 and ts > 2: continue
                    r = run_backtest(ens_score, sl_pct=sl, tp_pct=50, hold_max=hm,
                                    trail_pct=trail, trail_start=ts)
                    if r:
                        t = r.pop('trades')
                        model_results.append({**r, 'hm': hm, 'sl': sl, 'trail': trail, 'ts': ts})
    model_results.sort(key=lambda x: -x['ann'])
    results[ens_name] = model_results

# ============================================================
# [8] Random baseline (bug check)
# ============================================================
print(f"\n[8] Bug check: random baseline...", flush=True)
np.random.seed(42)
SCORE_RAND = np.full_like(ens_score, np.nan)
mask = ~np.isnan(all_scores['ultra_conserv'])
SCORE_RAND[mask] = np.random.randn(mask.sum())

r_rand = run_backtest(SCORE_RAND, sl_pct=7, tp_pct=50, hold_max=8,
                      trail_pct=1, trail_start=4)
if r_rand:
    t = r_rand.pop('trades')
    print(f"  Random baseline: {r_rand['ann']:+.1f}% (should be negative)", flush=True)
else:
    print(f"  Random baseline: blew up (good)", flush=True)

# Reversed
SCORE_REV = -all_scores['ultra_conserv']
r_rev = run_backtest(SCORE_REV, sl_pct=7, tp_pct=50, hold_max=8,
                     trail_pct=1, trail_start=4)
if r_rev:
    t = r_rev.pop('trades')
    print(f"  Reversed:        {r_rev['ann']:+.1f}% (should be very negative)", flush=True)
else:
    print(f"  Reversed:        blew up (good)", flush=True)

# ============================================================
# [9] Results
# ============================================================
print(f"\n[9] RESULTS", flush=True)

for ens_name in all_scores:
    mr = results[ens_name]
    print(f"\n  === {ens_name} (top 10) ===", flush=True)
    print(f"  {'HM':>3s} {'SL':>3s} {'Tr':>3s} {'TS':>3s} | "
          f"{'Ann':>7s} {'N':>4s} {'WR':>5s} {'W':>6s} {'L':>6s} {'Edge':>6s} {'TPY':>5s} {'DD':>5s}", flush=True)
    print(f"  {'-'*75}", flush=True)
    for r in mr[:10]:
        print(f"  {r['hm']:3d} {r['sl']:3d} {r['trail']:3d} {r['ts']:3d} | "
              f"{r['ann']:+7.1f}% {r['n']:4d} {r['wr']:5.1f}% {r['avg_w']:+6.1f}% "
              f"{r['avg_l']:6.1f}% {r['edge']:+6.2f}% {r['tpy']:5.1f} {r['max_dd']:4.1f}%", flush=True)

# ============================================================
# [10] Year-by-year (best config)
# ============================================================
print(f"\n[10] Year-by-year breakdown...", flush=True)
best_overall = None
for ens_name, mr in results.items():
    if mr and (best_overall is None or mr[0]['ann'] > best_overall['ann']):
        best_overall = {**mr[0], 'ens': ens_name}

if best_overall:
    print(f"  Best config: {best_overall['ens']} HM={best_overall['hm']} SL={best_overall['sl']} "
          f"Tr={best_overall['trail']} TS={best_overall['ts']} → {best_overall['ann']:+.1f}%", flush=True)

    # Re-run with trade tracking for year-by-year
    ens_score = all_scores[best_overall['ens']]
    r_full = run_backtest(ens_score, sl_pct=best_overall['sl'], tp_pct=50,
                          hold_max=best_overall['hm'], trail_pct=best_overall['trail'],
                          trail_start=best_overall['ts'])
    if r_full:
        trades = r_full['trades']
        dates_ts = pd.DatetimeIndex(dates)
        print(f"\n  {'Year':>6s} | {'Trades':>6s} {'WR':>5s} {'avgPnL':>7s} {'Total':>8s}", flush=True)
        print(f"  {'-'*40}", flush=True)
        for year in sorted(set(d.year for d in dates_ts[MIN_TRAIN:])):
            yr_trades = [t for t in trades
                        if dates_ts[t['di']].year == year and t['reason'] != 'end']
            if not yr_trades: continue
            nw = sum(1 for t in yr_trades if t['pnl'] > 0)
            wr = nw/max(len(yr_trades),1)*100
            avg_pnl = np.mean([t['pnl'] for t in yr_trades])
            # Compute cumulative return for this year
            cum = 1.0
            for t in yr_trades: cum *= (1 + t['pnl']/100)
            total = (cum - 1) * 100
            print(f"  {year:>6d} | {len(yr_trades):6d} {wr:5.0f}% {avg_pnl:+7.2f}% {total:+8.1f}%", flush=True)

# Final
print(f"\n{'='*70}", flush=True)
if best_overall:
    print(f"  FINAL: {best_overall['ens']} → {best_overall['ann']:+.1f}% | "
          f"WR={best_overall['wr']:.0f}% Edge={best_overall['edge']:+.2f}% "
          f"TPY={best_overall['tpy']:.0f} DD={best_overall['max_dd']:.0f}%", flush=True)
    above_300 = sum(1 for mr in results.values() for r in mr if r['ann'] >= 300)
    above_250 = sum(1 for mr in results.values() for r in mr if r['ann'] >= 250)
    print(f"  ≥300%: {above_300} | ≥250%: {above_250}", flush=True)
    print(f"  Bugs fixed: dedup strategies, train_end-1 leakage fix", flush=True)
print(f"{'='*70}", flush=True)

print(f"\nDone! {time.time()-t0:.1f}s", flush=True)
