"""
V100 — Multi-Horizon × Multi-Seed Ensemble (9 models)
======================================================
Combines V98's multi-horizon training with V94's multi-seed diversity:
  3 horizons (3d/5d/10d) × 3 seeds = 9 LGB LambdaRank models
  Each model rank-normalized per day, then averaged
  FIX: each config now uses its own horizon as the ranking label
"""
import sys, os, time, warnings, pickle
import numpy as np, pandas as pd
import lightgbm as lgb
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.data_loader import list_available_symbols, load_stock_data

COMMISSION = 0.0003; STAMP_DUTY = 0.001; CASH0 = 500_000

print("=" * 70, flush=True)
print("  V100 — Multi-Horizon × Multi-Seed Ensemble (9 models)", flush=True)
print("=" * 70, flush=True)

# [0] Strategy dedup
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
strat_hash = {}
for name in avail_all:
    h = []
    for sym in sorted(all_signals[name].keys())[:10]:
        for t, a, p in all_signals[name][sym][:5]:
            h.append((str(t)[:10], a, round(p,2)))
    strat_hash[name] = tuple(h)
seen_hashes = set(); USE_STRATS = []
for name in avail_all:
    h = strat_hash[name]
    if h not in seen_hashes:
        seen_hashes.add(h); USE_STRATS.append(name)

N_STRAT = len(USE_STRATS)
print(f"  {N_STRAT} unique strategies", flush=True)

# [1] Data loading
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

vol_map = {s: df['volume'].tail(60).mean() for s, df in stock_data.items()
           if 'volume' in df.columns and df['volume'].tail(60).mean() > 0}
syms = sorted([s for s, _ in sorted(vol_map.items(), key=lambda x: -x[1])[:500]])
NS = len(syms)
all_dates = sorted(set(d for s in syms for d in stock_data[s].index))
i0 = next(i for i, d in enumerate(all_dates) if d >= pd.Timestamp('2016-01-01'))
i1 = next((i for i, d in enumerate(all_dates) if d > pd.Timestamp('2026-04-25')), len(all_dates)) - 1
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

# [2] Strategy signals
print("[2] Loading strategy signals...", flush=True)
date_to_di = {d: i for i, d in enumerate(dates)}
int_date_to_di = {}
for d, i in date_to_di.items():
    int_date_to_di[int(d.strftime('%Y%m%d'))] = i
    int_date_to_di[str(d)[:10]] = i
    int_date_to_di[d] = i
STRAT_BUY = np.zeros((NS, ND, N_STRAT), dtype=np.int8)
STRAT_SELL = np.zeros((NS, ND, N_STRAT), dtype=np.int8)
for ki, sname in enumerate(USE_STRATS):
    for sym, sig_list in all_signals[sname].items():
        if sym not in syms: continue
        si = syms.index(sym)
        for ts, action, price in sig_list:
            if isinstance(ts, int):
                di = int_date_to_di.get(ts)
            else:
                di = date_to_di.get(ts)
            if di is not None:
                is_buy = (action == 'buy') if isinstance(action, str) else (action == 0)
                is_sell = (action == 'sell') if isinstance(action, str) else (action == 1)
                if is_buy: STRAT_BUY[si, di, ki] = 1
                elif is_sell: STRAT_SELL[si, di, ki] = 1

# [3] V71 features
print("[3] Computing V71 features...", flush=True)
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

# [4] Feature matrix
N_HAND = 20; N_FEAT = N_HAND + 2 * N_STRAT
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

# Multi-horizon forward returns
FWD_RET_3 = np.full((NS, ND), np.nan)
FWD_RET_10 = np.full((NS, ND), np.nan)
for si in range(NS):
    for di in range(60, ND):
        if np.isnan(C[si,di]) or C[si,di]<=0: continue
        if di < ND - 3 and not np.isnan(C[si,di+3]):
            FWD_RET_3[si,di] = (C[si,di+3]-C[si,di])/C[si,di]*100
        if di < ND - 10 and not np.isnan(C[si,di+10]):
            FWD_RET_10[si,di] = (C[si,di+10]-C[si,di])/C[si,di]*100
print(f"  Multi-horizon fwd ret computed (3d/5d/10d)", flush=True)

print(f"  Feature matrix: ({NS}, {ND}, {N_FEAT})", flush=True)

# ====================================================================
# [5] MULTI-HORIZON × MULTI-SEED ENSEMBLE (3 horizons × 3 seeds = 9)
# ====================================================================
TRAIN_WINDOW = 252 * 3; RETRAIN_FREQ = 42; MIN_TRAIN = 252 * 2; FWD_DAYS = 5

ENSEMBLE_CONFIGS = [
    {'num_leaves': 7,  'feature_fraction': 0.4, 'learning_rate': 0.05,
     'bagging_fraction': 0.7, 'min_data_in_leaf': 100},   # Config A
    {'num_leaves': 15, 'feature_fraction': 0.6, 'learning_rate': 0.03,
     'bagging_fraction': 0.8, 'min_data_in_leaf': 150},   # Config B
    {'num_leaves': 5,  'feature_fraction': 0.3, 'learning_rate': 0.07,
     'bagging_fraction': 0.6, 'min_data_in_leaf': 80},    # Config C
]
N_CFG = len(ENSEMBLE_CONFIGS)
SEED_SETS = [42, 123, 256]
N_SEEDS = len(SEED_SETS)
N_ENS = N_CFG * N_SEEDS  # 9 total

LGB_BASE = {
    'objective': 'lambdarank', 'metric': 'ndcg',
    'bagging_freq': 3, 'label_gain': [1, 2, 3, 4, 5],
    'verbose': -1, 'n_jobs': -1,
}

# Each config uses its own horizon as the ranking label (FIX from V98)
FWD_RET_MAP = [FWD_RET_3, FWD_RET, FWD_RET_10]  # 3d, 5d, 10d
FWD_DAYS_MAP = [3, 5, 10]

def make_train_data(train_start, train_end, fwd_arr):
    """Train data using fwd_arr for both filtering AND ranking labels."""
    train_X = []; train_y = []; train_group = []
    for di in range(train_start, train_end + 1):
        day_X = []; day_y = []
        for si in range(NS):
            f = FEAT[si, di]
            if np.any(np.isnan(f)): continue
            if np.isnan(fwd_arr[si, di]): continue
            day_X.append(f); day_y.append(fwd_arr[si, di])  # USE OWN HORIZON
        if len(day_X) >= 50:
            train_X.extend(day_X); train_y.extend(day_y)
            train_group.append(len(day_X))
    if len(train_X) < 2000: return None, None, None
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
    return tX, rank_y, train_group

print(f"\n[5] Training MULTI-HORIZON × MULTI-SEED ENSEMBLE "
      f"({N_ENS} models: {N_CFG} horizons × {N_SEEDS} seeds)...", flush=True)
t5 = time.time()

SCORE_ALL = [np.full((NS, ND), np.nan) for _ in range(N_ENS)]
retrain_points = []

for train_di in range(MIN_TRAIN, ND, RETRAIN_FREQ):
    pred_end = min(train_di + RETRAIN_FREQ, ND)
    retrain_points.append((train_di, pred_end))

    for ci, cfg in enumerate(ENSEMBLE_CONFIGS):
        fwd_d = FWD_DAYS_MAP[ci]
        train_end = train_di - fwd_d - 1
        train_start = max(MIN_TRAIN - 100, train_end - TRAIN_WINDOW)
        if train_start >= train_end: continue

        tX, rank_y, train_group = make_train_data(train_start, train_end, FWD_RET_MAP[ci])
        if tX is None: continue

        for si_idx, seed in enumerate(SEED_SETS):
            model_idx = si_idx * N_CFG + ci
            params = dict(LGB_BASE)
            params.update(cfg)
            params['seed'] = seed
            train_data = lgb.Dataset(tX, label=rank_y, group=train_group)
            model = lgb.train(params, train_data, num_boost_round=300)

            for di in range(train_di, pred_end):
                pred_list = []; pred_si = []
                for si in range(NS):
                    f = FEAT[si, di]
                    if np.any(np.isnan(f)): continue
                    if np.isnan(C[si,di]) or C[si,di]<=0: continue
                    pred_list.append(f); pred_si.append(si)
                if pred_list:
                    pX = np.array(pred_list, dtype=np.float32)
                    scores = model.predict(pX)
                    for k, si in enumerate(pred_si):
                        SCORE_ALL[model_idx][si, di] = scores[k]

    if len(retrain_points) % 5 == 0:
        print(f"    #{len(retrain_points)}: di={train_di}", flush=True)

# Rank-normalize each model per day, then average
print(f"  Rank-normalizing {N_ENS} models...", flush=True)
SCORE_NORM = [np.full((NS, ND), np.nan) for _ in range(N_ENS)]
for mi in range(N_ENS):
    for di in range(MIN_TRAIN, ND):
        vals = SCORE_ALL[mi][:, di]
        mask = ~np.isnan(vals)
        if mask.sum() < 50: continue
        ranked = np.argsort(np.argsort(vals[mask])).astype(float)
        n = len(ranked)
        for k, idx in enumerate(np.where(mask)[0]):
            SCORE_NORM[mi][idx, di] = ranked[k] / max(n-1, 1) * 100

SCORE_PRI = np.nanmean(np.stack(SCORE_NORM, axis=0), axis=0)
print(f"    Done: {len(retrain_points)} trainings × {N_ENS} models ({time.time()-t5:.1f}s)", flush=True)

# ====================================================================
# [6] Leak-free walk-forward meta model
# ====================================================================
print(f"\n[6] Leak-free walk-forward meta model...", flush=True)
t6 = time.time()

N_META_FEAT = N_FEAT + 4

SCORE_RANK = np.full((NS, ND), np.nan)
for di in range(MIN_TRAIN, ND):
    vals = SCORE_PRI[:, di]
    mask = ~np.isnan(vals)
    if mask.sum() < 50: continue
    ranked = np.argsort(np.argsort(vals[mask])).astype(float)
    n = len(ranked); pct = ranked / max(n-1, 1) * 100
    for k, idx in enumerate(np.where(mask)[0]):
        SCORE_RANK[idx, di] = pct[k]

META_PARAMS = {
    'objective': 'binary', 'metric': 'auc',
    'learning_rate': 0.05, 'num_leaves': 15,
    'feature_fraction': 0.5, 'bagging_fraction': 0.7,
    'bagging_freq': 3, 'min_data_in_leaf': 200,
    'verbose': -1, 'n_jobs': -1,
}
META_ROUNDS = 100

def make_meta_feat(si, di):
    f = FEAT[si, di]
    if np.any(np.isnan(f)): return None
    ext = np.zeros(N_META_FEAT, dtype=np.float32)
    ext[:N_FEAT] = f
    ext[N_FEAT] = SCORE_PRI[si, di] if not np.isnan(SCORE_PRI[si, di]) else 0
    ext[N_FEAT+1] = SCORE_RANK[si, di] if not np.isnan(SCORE_RANK[si, di]) else 50
    ext[N_FEAT+2] = MKT_BREADTH[di] if not np.isnan(MKT_BREADTH[di]) else 50
    ext[N_FEAT+3] = MKT_MOM20_VAL[di] if not np.isnan(MKT_MOM20_VAL[di]) else 0
    return ext

SCORE_META = np.full((NS, ND), np.nan)
n_meta_trains = 0
MIN_META_SAMPLES = 200

for rp_idx, (train_di, pred_end) in enumerate(retrain_points):
    meta_train_end = train_di - FWD_DAYS - 1
    meta_train_start = MIN_TRAIN
    if meta_train_end <= meta_train_start: continue

    meta_X_list = []; meta_y_list = []
    for di in range(meta_train_start, meta_train_end + 1):
        vals = SCORE_PRI[:, di]
        mask = ~np.isnan(vals)
        if mask.sum() < 50: continue
        top_indices = np.argsort(vals[mask])[::-1][:5]
        actual_indices = np.where(mask)[0][top_indices]

        for si in actual_indices:
            if np.isnan(FWD_RET[si, di]): continue
            ext = make_meta_feat(si, di)
            if ext is None: continue
            meta_X_list.append(ext)
            meta_y_list.append(1.0 if FWD_RET[si, di] > 0 else 0.0)

    if len(meta_y_list) < MIN_META_SAMPLES: continue

    mX = np.array(meta_X_list, dtype=np.float32)
    my = np.array(meta_y_list, dtype=np.float32)

    try:
        meta_train_data = lgb.Dataset(mX, label=my)
        meta_model = lgb.train(META_PARAMS, meta_train_data, num_boost_round=META_ROUNDS)
    except Exception:
        continue

    n_meta_trains += 1

    for di in range(train_di, pred_end):
        pred_list = []; pred_si = []
        for si in range(NS):
            if np.isnan(C[si,di]) or C[si,di]<=0: continue
            if np.isnan(SCORE_PRI[si, di]): continue
            ext = make_meta_feat(si, di)
            if ext is None: continue
            pred_list.append(ext); pred_si.append(si)
        if pred_list:
            pX = np.array(pred_list, dtype=np.float32)
            probs = meta_model.predict(pX)
            for k, si in enumerate(pred_si):
                SCORE_META[si, di] = probs[k]

    if n_meta_trains % 5 == 0:
        n_pos = my.sum()
        print(f"    meta #{n_meta_trains}: di={train_di}, samples={len(my)}, "
              f"pos={int(n_pos)}, neg={len(my)-int(n_pos)}", flush=True)

print(f"  Walk-forward meta: {n_meta_trains} trainings ({time.time()-t6:.1f}s)", flush=True)

# ====================================================================
# [7] Backtest engine
# ====================================================================
def run_backtest(score_arr, meta_arr, meta_threshold, sl_pct, tp_pct,
                 hold_max, trail_pct, trail_start=5,
                 atr_adaptive=False, atr_scale_min=0.6, atr_scale_range=0.8):
    cash = float(CASH0); pos = None; trades = []; pending = None
    n_filtered = 0; n_total = 0
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
                        if atr_adaptive and not np.isnan(ATR_PCT[si, di]):
                            atr_p = ATR_PCT[si, di]
                            atr_vals = ATR_PCT[:, di]
                            atr_valid = atr_vals[~np.isnan(atr_vals)]
                            if len(atr_valid) > 50:
                                pct = np.sum(atr_valid < atr_p) / max(len(atr_valid)-1,1)
                                pos['atr_scale'] = atr_scale_min + atr_scale_range * pct
                                pos['hm_scale'] = 0.7 + 0.6 * pct
                            else:
                                pos['atr_scale'] = 1.0; pos['hm_scale'] = 1.0
                        else:
                            pos['atr_scale'] = 1.0; pos['hm_scale'] = 1.0
            pending = None

        if pos is not None:
            si = pos['si']; p = C[si, di]
            if not np.isnan(p):
                if p > pos['highest']: pos['highest'] = p
                pnl = (p - pos['entry']) / pos['entry'] * 100
                hd = (dates[di] - pos['ed']).days
                base_sl = sl_pct * pos['atr_scale']
                if hd <= 2: sl_eff = base_sl * 1.3
                elif hd <= 5: sl_eff = base_sl
                else: sl_eff = base_sl * 0.7
                eff_hm = int(hold_max * pos['hm_scale'])
                er = None
                if pnl < -sl_eff: er = f'sl({pnl:.1f}%)'
                elif pnl > tp_pct: er = f'tp({pnl:.1f}%)'
                elif trail_pct > 0 and pnl > trail_start:
                    dd = (pos['highest'] - p) / pos['highest'] * 100
                    if dd > trail_pct: er = f'trail({pnl:.1f}%)'
                elif hold_max > 0 and hd >= max(eff_hm, 2): er = f'max({hd}d)'
                if er: pending = ('close', er)

        if pos is None and pending is None:
            best_si = -1; best_score = -1e9
            for si in range(NS):
                s = score_arr[si, di]
                if np.isnan(s): continue
                if s > best_score: best_score = s; best_si = si
            if best_si >= 0:
                n_total += 1
                if meta_threshold > 0 and meta_arr is not None:
                    meta_prob = meta_arr[best_si, di]
                    if np.isnan(meta_prob) or meta_prob < meta_threshold:
                        n_filtered += 1
                        continue
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
            'max_dd': round(max_dd,1), 'final': round(cash,0), 'trades': trades,
            'n_filtered': n_filtered, 'n_total': n_total}

# ====================================================================
# [8] Parameter search
# ====================================================================
print(f"\n[8] Parameter search...", flush=True)
t8 = time.time()

all_results = []

# No meta baseline
for hm in [5, 6, 7, 8]:
    for sl in [4, 5, 6, 7]:
        for ts in [3, 4, 5]:
            r = run_backtest(SCORE_PRI, None, 0, sl, 50, hm, 1, ts)
            if r:
                t = r.pop('trades')
                all_results.append({**r, 'hm': hm, 'sl': sl, 'ts': ts,
                                   'meta_th': 0, 'mode': 'fixed', 'label': 'no_meta_fixed'})

# No meta + ATR
for hm in [6, 7, 8]:
    for sl_base in [4, 5, 6]:
        for ts in [4, 5]:
            for amin in [0.5, 0.6]:
                for arng in [0.8, 1.0]:
                    r = run_backtest(SCORE_PRI, None, 0, sl_base, 50, hm, 1, ts,
                                    atr_adaptive=True, atr_scale_min=amin, atr_scale_range=arng)
                    if r:
                        t = r.pop('trades')
                        all_results.append({**r, 'hm': hm, 'sl': sl_base, 'ts': ts,
                                           'meta_th': 0, 'amin': amin, 'arng': arng,
                                           'mode': 'atr', 'label': 'no_meta_atr'})

# Walk-forward meta + fixed SL
for th in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
    for hm in [5, 6, 7, 8]:
        for sl in [4, 5, 6, 7]:
            for ts in [3, 4, 5]:
                r = run_backtest(SCORE_PRI, SCORE_META, th, sl, 50, hm, 1, ts)
                if r:
                    t = r.pop('trades')
                    all_results.append({**r, 'hm': hm, 'sl': sl, 'ts': ts,
                                       'meta_th': th, 'mode': 'fixed',
                                       'label': f'wf_meta{th:.2f}_fixed'})

# Walk-forward meta + ATR
for th in [0.50, 0.55, 0.60, 0.65, 0.70]:
    for hm in [6, 7, 8]:
        for sl_base in [4, 5, 6]:
            for ts in [4, 5]:
                for amin in [0.5, 0.6]:
                    for arng in [0.8, 1.0]:
                        r = run_backtest(SCORE_PRI, SCORE_META, th, sl_base, 50, hm, 1, ts,
                                        atr_adaptive=True, atr_scale_min=amin, atr_scale_range=arng)
                        if r:
                            t = r.pop('trades')
                            all_results.append({**r, 'hm': hm, 'sl': sl_base, 'ts': ts,
                                               'meta_th': th, 'amin': amin, 'arng': arng,
                                               'mode': 'atr', 'label': f'wf_meta{th:.2f}_atr'})

all_results.sort(key=lambda x: -x['ann'])
print(f"  {len(all_results)} configs tested ({time.time()-t8:.1f}s)", flush=True)

# [9] Bug check
print(f"\n[9] Bug check...", flush=True)
np.random.seed(42)
SCORE_RAND = np.full_like(SCORE_PRI, np.nan)
mask = ~np.isnan(SCORE_PRI)
SCORE_RAND[mask] = np.random.randn(mask.sum())

r_rand = run_backtest(SCORE_RAND, None, 0, 6, 50, 6, 1, 4,
                      atr_adaptive=True, atr_scale_min=0.6, atr_scale_range=0.8)
if r_rand:
    t = r_rand.pop('trades')
    print(f"  Random: {r_rand['ann']:+.1f}%", flush=True)

SCORE_REV = -SCORE_PRI
r_rev = run_backtest(SCORE_REV, None, 0, 6, 50, 6, 1, 4,
                     atr_adaptive=True, atr_scale_min=0.6, atr_scale_range=0.8)
if r_rev:
    t = r_rev.pop('trades')
    print(f"  Reversed: {r_rev['ann']:+.1f}%", flush=True)

# [10] Results
print(f"\n[10] RESULTS (top 30)", flush=True)
print(f"  {'Label':>22s} {'HM':>3s} {'SL':>3s} {'TS':>3s} | "
      f"{'Ann':>7s} {'N':>4s} {'WR':>5s} {'W':>6s} "
      f"{'L':>6s} {'Edge':>6s} {'TPY':>5s} {'DD':>5s}", flush=True)
print(f"  {'-'*105}", flush=True)
for r in all_results[:30]:
    print(f"  {r['label']:>22s} {r['hm']:3d} {r['sl']:3d} {r['ts']:3d} | "
          f"{r['ann']:+7.1f}% {r['n']:4d} {r['wr']:5.1f}% "
          f"{r['avg_w']:+6.1f}% {r['avg_l']:6.1f}% {r['edge']:+6.2f}% "
          f"{r['tpy']:5.1f} {r['max_dd']:4.1f}%",
          flush=True)

# Best per threshold
print(f"\n  Best per threshold (walk-forward meta):", flush=True)
for th in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
    sub = [r for r in all_results if abs(r['meta_th'] - th) < 0.01 and r['meta_th'] > 0]
    if sub:
        b = sub[0]
        print(f"    th={th:.2f}: {b['ann']:+.1f}% WR={b['wr']:.0f}% "
              f"Edge={b['edge']:+.2f}% TPY={b['tpy']:.0f} DD={b['max_dd']:.0f}% "
              f"filt={b['n_filtered']}/{b['n_total']}", flush=True)

sub_no = [r for r in all_results if r['meta_th'] == 0]
if sub_no:
    b = sub_no[0]
    print(f"\n  No-meta best: {b['ann']:+.1f}% WR={b['wr']:.0f}% "
          f"Edge={b['edge']:+.2f}% TPY={b['tpy']:.0f} DD={b['max_dd']:.0f}%", flush=True)

above_300 = sum(1 for r in all_results if r['ann'] >= 300)
above_400 = sum(1 for r in all_results if r['ann'] >= 400)
above_500 = sum(1 for r in all_results if r['ann'] >= 500)
above_800 = sum(1 for r in all_results if r['ann'] >= 800)
print(f"\n  >=300%: {above_300} | >=400%: {above_400} | >=500%: {above_500} | >=800%: {above_800}",
      flush=True)

# [11] Year-by-year for best
print(f"\n[11] Year-by-year...", flush=True)
best = all_results[0]
print(f"  Best: {best['label']} HM={best['hm']} SL={best['sl']} "
      f"TS={best['ts']} Th={best['meta_th']} -> {best['ann']:+.1f}%", flush=True)

atr_kw = {}
if best.get('mode') == 'atr':
    atr_kw = {'atr_adaptive': True,
              'atr_scale_min': best.get('amin', 0.6),
              'atr_scale_range': best.get('arng', 0.8)}

r_full = run_backtest(SCORE_PRI, SCORE_META if best['meta_th'] > 0 else None,
                      best['meta_th'],
                      sl_pct=best['sl'], tp_pct=50, hold_max=best['hm'],
                      trail_pct=1, trail_start=best['ts'], **atr_kw)

if r_full:
    trades = r_full['trades']
    dates_ts = pd.DatetimeIndex(dates)
    print(f"\n  {'Year':>6s} | {'Trades':>6s} {'WR':>5s} {'avgPnL':>7s} {'Total':>8s}", flush=True)
    print(f"  {'-'*40}", flush=True)
    for year in sorted(set(d.year for d in dates_ts[MIN_TRAIN:])):
        yr_trades = [t for t in trades if dates_ts[t['di']].year == year and t['reason'] != 'end']
        if not yr_trades: continue
        nw = sum(1 for t in yr_trades if t['pnl'] > 0)
        wr = nw/max(len(yr_trades),1)*100
        avg_pnl = np.mean([t['pnl'] for t in yr_trades])
        cum = 1.0
        for t in yr_trades: cum *= (1 + t['pnl']/100)
        total = (cum - 1) * 100
        print(f"  {year:>6d} | {len(yr_trades):6d} {wr:5.0f}% {avg_pnl:+7.2f}% {total:+8.1f}%",
              flush=True)

print(f"\n{'='*70}", flush=True)
print(f"  FINAL: {best['label']} -> {best['ann']:+.1f}% | WR={best['wr']:.0f}% "
      f"Edge={best['edge']:+.2f}% TPY={best['tpy']:.0f} DD={best['max_dd']:.0f}%", flush=True)
print(f"  Multi-horizon × multi-seed ({N_ENS} models: {N_CFG} horizons × {N_SEEDS} seeds)",
      flush=True)
print(f"{'='*70}", flush=True)
print(f"\nDone! {time.time()-t0:.1f}s", flush=True)
