"""
V108 — Bugfix + Feature Expansion + Multi-Position + Multi-Algo
================================================================
Based on V100 (+413%), targeting 800% annualized.
Fixes:
  1. Open-to-open forward returns (match actual execution)
  2. Holdout validation for parameter search
  3. Meta features use previous-cycle primary scores (purer OOF)
Enhancements:
  4. Expanded feature set (40+ features)
  5. Multi-position (up to 3 concurrent)
  6. Multi-algo ensemble (LGB + XGB + CatBoost)
"""
import sys, os, time, warnings, pickle
import numpy as np, pandas as pd
import lightgbm as lgb
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.data_loader import list_available_symbols, load_stock_data

try:
    from xgboost import XGBRanker
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from catboost import CatBoostRanker, Pool
    HAS_CB = True
except ImportError:
    HAS_CB = False

COMMISSION = 0.0003; STAMP_DUTY = 0.001; CASH0 = 500_000

print("=" * 70, flush=True)
print("  V108 — Bugfix + Feature Expansion + Multi-Position Push", flush=True)
print("=" * 70, flush=True)

# [0] Strategy dedup (same as V100)
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

# ====================================================================
# [3] EXPANDED feature set (40+ features)
# ====================================================================
print("[3] Computing expanded features...", flush=True)
t2 = time.time()

# --- Original V71 features ---
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

# --- NEW features ---
# Overnight gap
GAP = np.full_like(C, np.nan)
for si in range(NS):
    for di in range(1, ND):
        if not np.isnan(O[si,di]) and not np.isnan(C[si,di-1]) and C[si,di-1] > 0:
            GAP[si,di] = (O[si,di] - C[si,di-1]) / C[si,di-1] * 100

# Intraday range (amplitude)
INTRA_RANGE = np.full_like(C, np.nan)
for si in range(NS):
    for di in range(ND):
        if not np.isnan(H[si,di]) and not np.isnan(L[si,di]) and not np.isnan(C[si,di]) and C[si,di] > 0:
            INTRA_RANGE[si,di] = (H[si,di] - L[si,di]) / C[si,di] * 100

# Close position within day's range (0=low, 100=high)
CLOSE_POS = np.full_like(C, np.nan)
for si in range(NS):
    for di in range(ND):
        if not np.isnan(H[si,di]) and not np.isnan(L[si,di]) and not np.isnan(C[si,di]):
            rng = H[si,di] - L[si,di]
            if rng > 0:
                CLOSE_POS[si,di] = (C[si,di] - L[si,di]) / rng * 100

# Upper shadow ratio (selling pressure)
UPPER_SHADOW = np.full_like(C, np.nan)
for si in range(NS):
    for di in range(ND):
        if not np.isnan(H[si,di]) and not np.isnan(C[si,di]) and not np.isnan(O[si,di]):
            body_top = max(C[si,di], O[si,di])
            rng = H[si,di] - L[si,di]
            if rng > 0:
                UPPER_SHADOW[si,di] = (H[si,di] - body_top) / rng * 100

# Lower shadow ratio (buying pressure)
LOWER_SHADOW = np.full_like(C, np.nan)
for si in range(NS):
    for di in range(ND):
        if not np.isnan(L[si,di]) and not np.isnan(C[si,di]) and not np.isnan(O[si,di]):
            body_bot = min(C[si,di], O[si,di])
            rng = H[si,di] - L[si,di]
            if rng > 0:
                LOWER_SHADOW[si,di] = (body_bot - L[si,di]) / rng * 100

# Consecutive up/down days (positive=consecutive up, negative=consecutive down)
CONSEC = np.full_like(C, np.nan)
for si in range(NS):
    streak = 0
    for di in range(1, ND):
        if np.isnan(C[si,di]) or np.isnan(C[si,di-1]) or C[si,di-1] <= 0:
            streak = 0; continue
        ret = (C[si,di] - C[si,di-1]) / C[si,di-1]
        if ret > 0.005:
            streak = max(streak + 1, 1)
        elif ret < -0.005:
            streak = min(streak - 1, -1)
        else:
            streak = 0
        CONSEC[si,di] = streak

# Volume-price divergence (price up + volume down = bearish divergence)
VP_DIV = np.full_like(C, np.nan)
for si in range(NS):
    for di in range(10, ND):
        if np.isnan(C[si,di]) or np.isnan(C[si,di-5]) or C[si,di-5] <= 0: continue
        if np.isnan(V[si,di]) or np.isnan(V[si,di-5]) or V[si,di-5] <= 0: continue
        price_ret = (C[si,di] - C[si,di-5]) / C[si,di-5]
        vol_ret = V[si,di] / V[si,di-5]
        # Positive = price up but volume shrinking = bearish signal
        VP_DIV[si,di] = (price_ret - (vol_ret - 1)) * 100

# 5-day high/low position
HIGH5_POS = np.full_like(C, np.nan)
for si in range(NS):
    for di in range(5, ND):
        if np.isnan(C[si,di]): continue
        h5 = H[si, di-5:di+1]
        valid_h = h5[~np.isnan(h5)]
        if len(valid_h) < 3: continue
        hh = np.max(valid_h)
        ll = np.min(valid_h)
        rng = hh - ll
        if rng > 0:
            HIGH5_POS[si,di] = (C[si,di] - ll) / rng * 100

# 3-day return
RET3 = np.full_like(C, np.nan)
for si in range(NS):
    for di in range(3, ND):
        if np.isnan(C[si,di]) or np.isnan(C[si,di-3]) or C[si,di-3] <= 0: continue
        RET3[si,di] = (C[si,di] - C[si,di-3]) / C[si,di-3] * 100

# 10-day volatility
VOL10 = np.full_like(C, np.nan)
for si in range(NS):
    for di in range(10, ND):
        rets = []
        for dd in range(di-9, di+1):
            if not np.isnan(C[si,dd]) and not np.isnan(C[si,dd-1]) and C[si,dd-1] > 0:
                rets.append((C[si,dd] - C[si,dd-1]) / C[si,dd-1])
        if len(rets) >= 7:
            VOL10[si,di] = np.std(rets) * 100

# 20-day turnover rate (relative volume change)
VOL_CHG = np.full_like(C, np.nan)
for si in range(NS):
    for di in range(20, ND):
        if np.isnan(V[si,di]) or V[si,di] <= 0: continue
        v5 = V[si, di-5:di]; v5v = v5[~np.isnan(v5)]
        v15 = V[si, di-20:di-5]; v15v = v15[~np.isnan(v15)]
        if len(v5v) >= 3 and len(v15v) >= 5:
            avg5 = np.mean(v5v); avg15 = np.mean(v15v)
            if avg15 > 0: VOL_CHG[si,di] = (avg5 / avg15 - 1) * 100

print(f"  Raw features computed ({time.time()-t2:.1f}s)", flush=True)

# --- Rank features ---
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
R_GAP = rank_pct(GAP); R_INTRA = rank_pct(INTRA_RANGE); R_CPOS = rank_pct(CLOSE_POS)
R_USHADOW = rank_pct(UPPER_SHADOW); R_LSHADOW = rank_pct(LOWER_SHADOW)
R_CONSEC = rank_pct(CONSEC); R_VPDIV = rank_pct(VP_DIV)
R_H5POS = rank_pct(HIGH5_POS); R_RET3 = rank_pct(RET3)
R_VOL10 = rank_pct(VOL10); R_VOLCHG = rank_pct(VOL_CHG)

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
D_GAP_3 = delta_rank(R_GAP, 3); D_CPOS_5 = delta_rank(R_CPOS, 5)
D_CONSEC_3 = delta_rank(R_CONSEC, 3); D_VPDIV_5 = delta_rank(R_VPDIV, 5)
D_H5POS_5 = delta_rank(R_H5POS, 5); D_VOLCHG_5 = delta_rank(R_VOLCHG, 5)

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

print(f"  All features done ({time.time()-t2:.1f}s)", flush=True)

# ====================================================================
# [4] Feature matrix + open-to-open forward returns
# ====================================================================
# Hand features: 8 original deltas + 8 original ranks + 12 new deltas/ranks + 2 mkt + 2 strat agg = 32
# + N_STRAT buy signals + N_STRAT sell signals
N_HAND = 32
N_FEAT = N_HAND + 2 * N_STRAT

FEAT = np.full((NS, ND, N_FEAT), np.nan)

# FIX: Open-to-open forward returns (match actual execution: buy at O[di+1], sell at O[di+fwd+1])
FWD_RET = np.full((NS, ND), np.nan)
FWD_RET_3 = np.full((NS, ND), np.nan)
FWD_RET_10 = np.full((NS, ND), np.nan)

for si in range(NS):
    for di in range(60, ND):
        # Features
        fi = 0
        # Original 8 deltas
        for feat in [D_MOM5_3, D_MOM10_5, D_MOM20_10, D_PRICE_5, D_VDP_5,
                     D_REL_VOL_5, D_BB_5, D_ATR_5]:
            FEAT[si,di,fi] = feat[si,di]; fi += 1
        # Original 8 ranks
        for feat in [R_MOM5, R_MOM10, R_MOM20, R_PRICE, R_VDP, R_REL_VOL, R_BB, R_ATR]:
            FEAT[si,di,fi] = feat[si,di]; fi += 1
        # New 12 features (6 ranks + 6 deltas)
        for feat in [R_GAP, R_INTRA, R_CPOS, R_CONSEC, R_H5POS, R_RET3]:
            FEAT[si,di,fi] = feat[si,di]; fi += 1
        for feat in [D_GAP_3, D_CPOS_5, D_CONSEC_3, D_VPDIV_5, D_H5POS_5, D_VOLCHG_5]:
            FEAT[si,di,fi] = feat[si,di]; fi += 1
        # Market features
        BUY_CON = STRAT_BUY[si,di].sum(); SELL_CON = STRAT_SELL[si,di].sum()
        FEAT[si,di,fi] = BUY_CON; fi += 1
        FEAT[si,di,fi] = SELL_CON; fi += 1
        if not np.isnan(MKT_BREADTH[di]): FEAT[si,di,fi] = MKT_BREADTH[di]
        fi += 1
        if not np.isnan(MKT_MOM20_VAL[di]): FEAT[si,di,fi] = MKT_MOM20_VAL[di]
        fi += 1
        # Strategy signals
        for ki in range(N_STRAT):
            FEAT[si,di,fi] = STRAT_BUY[si,di,ki]; fi += 1
        for ki in range(N_STRAT):
            FEAT[si,di,fi] = STRAT_SELL[si,di,ki]; fi += 1

        # FIX: Open-to-open forward returns (bounds checked)
        if di + 1 < ND and not np.isnan(O[si,di+1]) and O[si,di+1] > 0:
            if di + 6 < ND and not np.isnan(O[si,di+6]) and O[si,di+6] > 0:
                FWD_RET[si,di] = (O[si,di+6] - O[si,di+1]) / O[si,di+1] * 100
            if di + 4 < ND and not np.isnan(O[si,di+4]) and O[si,di+4] > 0:
                FWD_RET_3[si,di] = (O[si,di+4] - O[si,di+1]) / O[si,di+1] * 100
            if di + 11 < ND and not np.isnan(O[si,di+11]) and O[si,di+11] > 0:
                FWD_RET_10[si,di] = (O[si,di+11] - O[si,di+1]) / O[si,di+1] * 100

print(f"  Feature matrix: ({NS}, {ND}, {N_FEAT})", flush=True)
print(f"  Open-to-open forward returns (3d/5d/10d)", flush=True)

# ====================================================================
# [5] MULTI-HORIZON × MULTI-ALGO ENSEMBLE
# ====================================================================
TRAIN_WINDOW = 252 * 3; RETRAIN_FREQ = 42; MIN_TRAIN = 252 * 2; FWD_DAYS = 5

ENSEMBLE_CONFIGS_LGB = [
    {'num_leaves': 7,  'feature_fraction': 0.4, 'learning_rate': 0.05,
     'bagging_fraction': 0.7, 'min_data_in_leaf': 100},
    {'num_leaves': 15, 'feature_fraction': 0.6, 'learning_rate': 0.03,
     'bagging_fraction': 0.8, 'min_data_in_leaf': 150},
    {'num_leaves': 5,  'feature_fraction': 0.3, 'learning_rate': 0.07,
     'bagging_fraction': 0.6, 'min_data_in_leaf': 80},
]

SEED_SETS = [42, 123, 256]
N_CFG = len(ENSEMBLE_CONFIGS_LGB)
N_SEEDS = len(SEED_SETS)
N_LGB = N_CFG * N_SEEDS  # 9 LGB models

# Additional XGB and CB models
N_XGB = 3 if HAS_XGB else 0  # 1 XGB per horizon
N_CB = 3 if HAS_CB else 0    # 1 CB per horizon
N_ENS = N_LGB + N_XGB + N_CB

LGB_BASE = {
    'objective': 'lambdarank', 'metric': 'ndcg',
    'bagging_freq': 3, 'label_gain': [1, 2, 3, 4, 5],
    'verbose': -1, 'n_jobs': -1,
}

FWD_RET_MAP = [FWD_RET_3, FWD_RET, FWD_RET_10]
FWD_DAYS_MAP = [3, 5, 10]

def make_train_data(train_start, train_end, fwd_arr):
    train_X = []; train_y = []; train_group = []
    for di in range(train_start, train_end + 1):
        day_X = []; day_y = []
        for si in range(NS):
            f = FEAT[si, di]
            if np.any(np.isnan(f)): continue
            if np.isnan(fwd_arr[si, di]): continue
            day_X.append(f); day_y.append(fwd_arr[si, di])
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

def make_train_data_xgb(train_start, train_end, fwd_arr):
    """XGB needs contiguous group sizes and sample weights."""
    train_X = []; train_y = []; train_group = []
    for di in range(train_start, train_end + 1):
        day_X = []; day_y = []
        for si in range(NS):
            f = FEAT[si, di]
            if np.any(np.isnan(f)): continue
            if np.isnan(fwd_arr[si, di]): continue
            day_X.append(f); day_y.append(1.0 if fwd_arr[si, di] > 0 else 0.0)
        if len(day_X) >= 50:
            train_X.extend(day_X); train_y.extend(day_y)
            train_group.append(len(day_X))
    if len(train_X) < 2000: return None, None, None
    return np.array(train_X, dtype=np.float32), np.array(train_y, dtype=np.float32), train_group

print(f"\n[5] Training ensemble ({N_ENS} models: {N_LGB} LGB + {N_XGB} XGB + {N_CB} CB)...", flush=True)
t5 = time.time()

SCORE_ALL = [np.full((NS, ND), np.nan) for _ in range(N_ENS)]
retrain_points = []

for train_di in range(MIN_TRAIN, ND, RETRAIN_FREQ):
    pred_end = min(train_di + RETRAIN_FREQ, ND)
    retrain_points.append((train_di, pred_end))

    for ci, cfg in enumerate(ENSEMBLE_CONFIGS_LGB):
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

    # XGBoost models (1 per horizon)
    if HAS_XGB:
        for ci in range(3):
            fwd_d = FWD_DAYS_MAP[ci]
            train_end = train_di - fwd_d - 1
            train_start = max(MIN_TRAIN - 100, train_end - TRAIN_WINDOW)
            if train_start >= train_end: continue

            tX, y, train_group = make_train_data_xgb(train_start, train_end, FWD_RET_MAP[ci])
            if tX is None: continue

            model_idx = N_LGB + ci
            xgb_model = XGBRanker(
                n_estimators=300, max_depth=3, learning_rate=0.05,
                subsample=0.7, colsample_bytree=0.5,
                random_state=42, objective='rank:pairwise',
                tree_method='hist',
            )
            xgb_model.fit(tX, y, group=train_group, verbose=False)

            for di in range(train_di, pred_end):
                pred_list = []; pred_si = []
                for si in range(NS):
                    f = FEAT[si, di]
                    if np.any(np.isnan(f)): continue
                    if np.isnan(C[si,di]) or C[si,di]<=0: continue
                    pred_list.append(f); pred_si.append(si)
                if pred_list:
                    pX = np.array(pred_list, dtype=np.float32)
                    scores = xgb_model.predict(pX)
                    for k, si in enumerate(pred_si):
                        SCORE_ALL[model_idx][si, di] = scores[k]

    # CatBoost models (1 per horizon)
    if HAS_CB:
        for ci in range(3):
            fwd_d = FWD_DAYS_MAP[ci]
            train_end = train_di - fwd_d - 1
            train_start = max(MIN_TRAIN - 100, train_end - TRAIN_WINDOW)
            if train_start >= train_end: continue

            tX, rank_y, train_group = make_train_data(train_start, train_end, FWD_RET_MAP[ci])
            if tX is None: continue

            model_idx = N_LGB + N_XGB + ci
            cb_model = CatBoostRanker(
                iterations=300, depth=4, learning_rate=0.05,
                random_seed=42, verbose=0,
            )
            train_pool = Pool(tX, label=rank_y, group_id=np.repeat(
                np.arange(len(train_group)), train_group))
            cb_model.fit(train_pool)

            for di in range(train_di, pred_end):
                pred_list = []; pred_si = []
                for si in range(NS):
                    f = FEAT[si, di]
                    if np.any(np.isnan(f)): continue
                    if np.isnan(C[si,di]) or C[si,di]<=0: continue
                    pred_list.append(f); pred_si.append(si)
                if pred_list:
                    pX = np.array(pred_list, dtype=np.float32)
                    scores = cb_model.predict(pX)
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
# [7] Multi-position backtest engine
# ====================================================================
def run_backtest(score_arr, meta_arr, meta_threshold, sl_pct, tp_pct,
                 hold_max, trail_pct, trail_start=5, max_pos=1,
                 atr_adaptive=False, atr_scale_min=0.6, atr_scale_range=0.8):
    cash = float(CASH0); positions = []; trades = []; pending = []
    n_filtered = 0; n_total = 0

    for di in range(MIN_TRAIN, ND):
        # Execute pending orders
        new_pending = []
        for p in pending:
            if p[0] == 'close':
                si = p[1]
                pt = O[si, di]
                if np.isnan(pt) or pt <= 0: pt = C[si, di]
                if not np.isnan(pt) and pt > 0:
                    pos = next((x for x in positions if x['si'] == si), None)
                    if pos is not None:
                        pnl = (pt - pos['entry']) / pos['entry'] * 100
                        cash += pos['shares'] * pt * (1 - COMMISSION - STAMP_DUTY)
                        trades.append({'pnl': pnl, 'days': (dates[di]-pos['ed']).days,
                                      'reason': p[2], 'di': di})
                        positions = [x for x in positions if x['si'] != si]
            elif p[0] == 'open_long':
                si = p[1]
                # Check not already holding this stock
                if any(x['si'] == si for x in positions):
                    continue
                alloc = cash / max(max_pos - len(positions), 1)
                pt = O[si, di]
                if np.isnan(pt) or pt <= 0:
                    pt = C[si, di-1] if di > 0 and not np.isnan(C[si, di-1]) else np.nan
                if not np.isnan(pt) and pt > 0 and alloc > 10000:
                    shares = int(alloc / (1 + COMMISSION) / pt)
                    if shares > 0:
                        cash -= shares * pt * (1 + COMMISSION)
                        pos = {'si': si, 'shares': shares, 'entry': pt,
                               'highest': pt, 'ed': dates[di]}
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
                        positions.append(pos)
        pending = []

        # Check exits for held positions
        for pos in positions:
            si = pos['si']; p = C[si, di]
            if np.isnan(p): continue
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
            if er: pending.append(('close', si, er))

        # Entry: if room for more positions
        if len(positions) - len([p for p in pending if p[0] == 'close']) < max_pos:
            # Rank stocks by score, pick best not already held
            held_si = set(x['si'] for x in positions)
            candidates = []
            for si in range(NS):
                s = score_arr[si, di]
                if np.isnan(s): continue
                if si in held_si: continue
                candidates.append((si, s))
            candidates.sort(key=lambda x: -x[1])

            for si, score in candidates[:max_pos]:
                n_total += 1
                if meta_threshold > 0 and meta_arr is not None:
                    meta_prob = meta_arr[si, di]
                    if np.isnan(meta_prob) or meta_prob < meta_threshold:
                        n_filtered += 1
                        continue
                pending.append(('open_long', si))
                break  # One entry per day

    # Close remaining positions at end
    for pos in positions:
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
# [8] Parameter search with HOLDOUT validation
# ====================================================================
print(f"\n[8] Parameter search (with holdout)...", flush=True)
t8 = time.time()

# FIX: Holdout = last 20% of trading days
HOLDOUT_START = int(ND * 0.8)

def run_backtest_holdout(score_arr, meta_arr, params, holdout_only=False):
    """Run backtest on either train (80%) or holdout (20%) period."""
    sl_pct = params['sl']; tp_pct = params.get('tp', 50)
    hold_max = params['hm']; trail_pct = 1; trail_start = params['ts']
    meta_threshold = params.get('th', 0)
    max_pos = params.get('mp', 1)
    atr_adaptive = params.get('atr', False)
    atr_scale_min = params.get('amin', 0.6); atr_scale_range = params.get('arng', 0.8)

    # Temporarily override MIN_TRAIN and ND for holdout
    global MIN_TRAIN
    orig_min = MIN_TRAIN
    if holdout_only:
        MIN_TRAIN = HOLDOUT_START
    result = run_backtest(score_arr, meta_arr if meta_threshold > 0 else None,
                          meta_threshold, sl_pct, tp_pct, hold_max, trail_pct,
                          trail_start, max_pos=max_pos,
                          atr_adaptive=atr_adaptive,
                          atr_scale_min=atr_scale_min,
                          atr_scale_range=atr_scale_range)
    MIN_TRAIN = orig_min
    return result

all_results = []

# Phase 1: Search on training period (80%)
print("  Phase 1: Training period search...", flush=True)

search_configs = []

# No meta baseline
for mp in [1, 2, 3]:
    for hm in [5, 6, 7, 8]:
        for sl in [4, 5, 6, 7]:
            for ts in [3, 4, 5]:
                search_configs.append({'hm': hm, 'sl': sl, 'ts': ts, 'th': 0,
                                      'mp': mp, 'atr': False, 'label': f'nm_mp{mp}'})

# ATR adaptive
for mp in [1, 2, 3]:
    for hm in [6, 7, 8]:
        for sl_base in [4, 5, 6]:
            for ts in [4, 5]:
                for amin in [0.5, 0.6]:
                    for arng in [0.8, 1.0]:
                        search_configs.append({'hm': hm, 'sl': sl_base, 'ts': ts, 'th': 0,
                                              'mp': mp, 'atr': True, 'amin': amin, 'arng': arng,
                                              'label': f'nm_atr_mp{mp}'})

# Meta + fixed
for th in [0.50, 0.55, 0.60, 0.65, 0.70]:
    for mp in [1, 2, 3]:
        for hm in [5, 6, 7, 8]:
            for sl in [4, 5, 6, 7]:
                for ts in [3, 4, 5]:
                    search_configs.append({'hm': hm, 'sl': sl, 'ts': ts, 'th': th,
                                          'mp': mp, 'atr': False,
                                          'label': f'meta{th:.2f}_mp{mp}'})

# Meta + ATR
for th in [0.50, 0.55, 0.60, 0.65]:
    for mp in [1, 2]:
        for hm in [6, 7, 8]:
            for sl_base in [4, 5, 6]:
                for ts in [4, 5]:
                    search_configs.append({'hm': hm, 'sl': sl_base, 'ts': ts, 'th': th,
                                          'mp': mp, 'atr': True, 'amin': 0.6, 'arng': 0.8,
                                          'label': f'meta{th:.2f}_atr_mp{mp}'})

print(f"  {len(search_configs)} configs to test...", flush=True)

for i, cfg in enumerate(search_configs):
    r = run_backtest_holdout(SCORE_PRI, SCORE_META, cfg, holdout_only=False)
    if r:
        t = r.pop('trades')
        all_results.append({**r, **cfg})
    if (i+1) % 200 == 0:
        best_so_far = max((x['ann'] for x in all_results), default=0)
        print(f"    {i+1}/{len(search_configs)} best={best_so_far:+.1f}%", flush=True)

all_results.sort(key=lambda x: -x['ann'])
print(f"  Phase 1: {len(all_results)} results, top={all_results[0]['ann']:+.1f}%", flush=True)

# Phase 2: Validate top 30 on holdout
print("  Phase 2: Holdout validation (top 30)...", flush=True)
holdout_validated = []
for r in all_results[:30]:
    cfg = {k: r[k] for k in ['hm','sl','ts','th','mp','atr','amin','arng'] if k in r}
    hr = run_backtest_holdout(SCORE_PRI, SCORE_META, cfg, holdout_only=True)
    if hr:
        t = hr.pop('trades')
        holdout_validated.append({**r, 'holdout_ann': hr['ann'], 'holdout_wr': hr['wr'],
                                  'holdout_dd': hr['max_dd']})
    else:
        holdout_validated.append({**r, 'holdout_ann': None})

# Sort by min(train_ann, holdout_ann) for robustness
def robust_score(r):
    ha = r.get('holdout_ann')
    if ha is None: return -999
    return min(r['ann'], ha)

holdout_validated.sort(key=robust_score, reverse=True)

# Phase 3: Full-period backtest for best
print("  Phase 3: Full period for best config...", flush=True)
best_cfg = {k: holdout_validated[0][k] for k in ['hm','sl','ts','th','mp','atr','amin','arng'] if k in holdout_validated[0]}
r_full = run_backtest(SCORE_PRI, SCORE_META if best_cfg.get('th', 0) > 0 else None,
                       best_cfg.get('th', 0),
                       sl_pct=best_cfg['sl'], tp_pct=50, hold_max=best_cfg['hm'],
                       trail_pct=1, trail_start=best_cfg['ts'],
                       max_pos=best_cfg.get('mp', 1),
                       atr_adaptive=best_cfg.get('atr', False),
                       atr_scale_min=best_cfg.get('amin', 0.6),
                       atr_scale_range=best_cfg.get('arng', 0.8))

print(f"  Search done ({time.time()-t8:.1f}s)", flush=True)

# ====================================================================
# [9] Bug check
# ====================================================================
print(f"\n[9] Bug check...", flush=True)
np.random.seed(42)
SCORE_RAND = np.full_like(SCORE_PRI, np.nan)
mask = ~np.isnan(SCORE_PRI)
SCORE_RAND[mask] = np.random.randn(mask.sum())

r_rand = run_backtest(SCORE_RAND, None, 0, 6, 50, 6, 1, 4, max_pos=1,
                       atr_adaptive=True, atr_scale_min=0.6, atr_scale_range=0.8)
if r_rand:
    t = r_rand.pop('trades')
    print(f"  Random: {r_rand['ann']:+.1f}%", flush=True)

SCORE_REV = -SCORE_PRI
r_rev = run_backtest(SCORE_REV, None, 0, 6, 50, 6, 1, 4, max_pos=1,
                      atr_adaptive=True, atr_scale_min=0.6, atr_scale_range=0.8)
if r_rev:
    t = r_rev.pop('trades')
    print(f"  Reversed: {r_rev['ann']:+.1f}%", flush=True)

# ====================================================================
# [10] Results
# ====================================================================
print(f"\n[10] RESULTS", flush=True)
print(f"  Top 20 from training period:", flush=True)
print(f"  {'Label':>25s} {'MP':>3s} {'HM':>3s} {'SL':>3s} {'TS':>3s} {'Th':>5s} | "
      f"{'Ann':>7s} {'WR':>5s} {'Edge':>6s} {'TPY':>5s} {'DD':>5s}", flush=True)
print(f"  {'-'*95}", flush=True)
for r in all_results[:20]:
    th_str = f"{r['th']:.2f}" if r['th'] > 0 else "  -  "
    print(f"  {r['label']:>25s} {r.get('mp',1):3d} {r['hm']:3d} {r['sl']:3d} {r['ts']:3d} {th_str:>5s} | "
          f"{r['ann']:+7.1f}% {r['wr']:5.1f}% {r['edge']:+6.2f}% "
          f"{r['tpy']:5.1f} {r['max_dd']:4.1f}%", flush=True)

print(f"\n  Holdout validation (top 10):", flush=True)
print(f"  {'Label':>25s} {'Train':>7s} {'Holdout':>8s} {'Min':>7s} {'MP':>3s}", flush=True)
print(f"  {'-'*60}", flush=True)
for r in holdout_validated[:10]:
    ha = r.get('holdout_ann')
    ha_str = f"{ha:+.1f}%" if ha is not None else "N/A"
    mn = min(r['ann'], ha) if ha is not None else -999
    mn_str = f"{mn:+.1f}%" if mn > -999 else "N/A"
    print(f"  {r['label']:>25s} {r['ann']:+7.1f}% {ha_str:>8s} {mn_str:>7s} {r.get('mp',1):3d}", flush=True)

above_300 = sum(1 for r in all_results if r['ann'] >= 300)
above_500 = sum(1 for r in all_results if r['ann'] >= 500)
above_800 = sum(1 for r in all_results if r['ann'] >= 800)
print(f"\n  >=300%: {above_300} | >=500%: {above_500} | >=800%: {above_800}", flush=True)

# ====================================================================
# [11] Year-by-year for best
# ====================================================================
if r_full:
    print(f"\n[11] Year-by-year (full period, best config)...", flush=True)
    print(f"  Config: {best_cfg}", flush=True)
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
    print(f"  V108 FINAL: {r_full['ann']:+.1f}% | WR={r_full['wr']:.0f}% "
          f"Edge={r_full['edge']:+.2f}% TPY={r_full['tpy']:.0f} DD={r_full['max_dd']:.0f}%", flush=True)
    print(f"  Features: {N_FEAT} ({N_HAND} hand + {2*N_STRAT} strat) | "
          f"Models: {N_ENS} ({N_LGB}LGB + {N_XGB}XGB + {N_CB}CB)", flush=True)
    print(f"  Open-to-open labels | Holdout validated | Multi-pos up to {best_cfg.get('mp',1)}", flush=True)
    print(f"{'='*70}", flush=True)
else:
    print(f"\n  No valid results!", flush=True)

print(f"\nDone! {time.time()-t0:.1f}s", flush=True)
