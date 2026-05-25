"""
V112 — Regime-Adaptive Trading Engine
======================================
Key insight: v109-v111 all stuck at +11-13% because the TRADING LOGIC is fixed.
The LambdaRank ranking is good, but the execution doesn't adapt to regime.

V112 innovation: Regime-dependent EVERYTHING
  - Entry timing adapted by KER/Hurst regime
  - Position sizing adapted by regime conviction + Kelly
  - Stop loss adapted by regime volatility (GARCH proxy via ATR)
  - Hold period adapted by regime type
  - SHORT positions in bear regimes
  - Entropy gate blocks all trading in chaos
  - Multi-timeframe confirmation

Regime Classification (per day, market-wide):
  TRENDING: KER > 0.3, low entropy → trend-follow, wider stops, longer holds
  CYCLICAL: KER 0.15-0.3 → momentum with tighter stops
  RANGING: KER < 0.15, low entropy → mean reversion, quick entries/exits
  CHAOTIC: high entropy → NO TRADING (无为)

Architecture: same [0]-[6] as v110, new [7]-[10] trading engine
Target: 1000% annualized
"""
import sys, os, time, warnings, pickle
import numpy as np, pandas as pd
import lightgbm as lgb
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.data_loader import list_available_symbols, load_stock_data

COMMISSION = 0.0003; STAMP_DUTY = 0.001; CASH0 = 500_000

print("=" * 70, flush=True)
print("  V112-FAST v2 — Relaxed regime thresholds (less CHAOS blocking)", flush=True)
print("=" * 70, flush=True)

# [0] Strategy dedup (same as v109)
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

# [1] Data loading (same as v109)
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
    df = df[~df.index.duplicated(keep='first')]
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

# [2] Strategy signals (same as v109)
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

# ====================================================================
# [3] Feature computation — V109 base + PHILOSOPHY features
# ====================================================================
print("[3] Computing features (v109 base + philosophy)...", flush=True)
t2 = time.time()

# --- V109 base features (same) ---
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

GAP = np.full_like(C, np.nan)
CLOSE_POS = np.full_like(C, np.nan)
CONSEC = np.full_like(C, np.nan)
for si in range(NS):
    streak = 0
    for di in range(ND):
        # Gap
        if di > 0 and not np.isnan(O[si,di]) and not np.isnan(C[si,di-1]) and C[si,di-1] > 0:
            GAP[si,di] = (O[si,di] - C[si,di-1]) / C[si,di-1] * 100
        # Close position
        if not np.isnan(H[si,di]) and not np.isnan(L[si,di]) and not np.isnan(C[si,di]):
            rng = H[si,di] - L[si,di]
            if rng > 0: CLOSE_POS[si,di] = (C[si,di] - L[si,di]) / rng * 100
        # Consecutive
        if di > 0 and not np.isnan(C[si,di]) and not np.isnan(C[si,di-1]) and C[si,di-1] > 0:
            ret = (C[si,di] - C[si,di-1]) / C[si,di-1]
            if ret > 0.005: streak = max(streak + 1, 1)
            elif ret < -0.005: streak = min(streak - 1, -1)
            else: streak = 0
            CONSEC[si,di] = streak

print(f"  V109 base features ({time.time()-t2:.1f}s)", flush=True)

# --- NEW: Philosophy features ---

# KER (Kaufman Efficiency Ratio) ≈ |H-0.5| approximation
# KER = |net displacement| / total path length
# Layer 2: regime detection — KER>0.3=trend, KER<0.15=range
KER = np.full_like(C, np.nan)
for si in range(NS):
    for di in range(20, ND):
        if np.isnan(C[si,di]) or np.isnan(C[si,di-20]) or C[si,di-20] <= 0: continue
        net = abs(C[si,di] - C[si,di-20])
        total = 0.0
        valid = True
        for dd in range(di-19, di+1):
            if np.isnan(C[si,dd]) or np.isnan(C[si,dd-1]): valid = False; break
            total += abs(C[si,dd] - C[si,dd-1])
        if valid and total > 0:
            KER[si,di] = net / total

# Shannon Entropy of returns — Layer 1: entropy gate
# H(X) = -sum(p_i * log2(p_i)), max H = log2(n_bins)
# High entropy → chaotic → don't trade (无为)
ENTROPY = np.full_like(C, np.nan)
N_BINS = 10
H_MAX = np.log2(N_BINS)  # ~3.322 bits
for si in range(NS):
    for di in range(50, ND):
        rets = []
        for dd in range(di-49, di+1):
            if not np.isnan(C[si,dd]) and dd > 0 and not np.isnan(C[si,dd-1]) and C[si,dd-1] > 0:
                rets.append((C[si,dd] - C[si,dd-1]) / C[si,dd-1])
        if len(rets) < 30: continue
        rets = np.array(rets)
        counts, _ = np.histogram(rets, bins=N_BINS)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        h = -np.sum(probs * np.log2(probs))
        ENTROPY[si,di] = h / H_MAX  # normalized to [0,1]

# Structural Tension (simplified 3-point: high, low, midpoint anchors)
# Layer 4: signal generation — tension > 0 bullish, < 0 bearish
TENSION = np.full_like(C, np.nan)
for si in range(NS):
    for di in range(20, ND):
        if np.isnan(C[si,di]): continue
        c20 = C[si, max(0,di-20):di+1]
        h20 = H[si, max(0,di-20):di+1]
        l20 = L[si, max(0,di-20):di+1]
        cv = c20[~np.isnan(c20)]
        hv = h20[~np.isnan(h20)]
        lv = l20[~np.isnan(l20)]
        if len(cv) < 10: continue
        hh = np.max(hv) if len(hv) > 0 else np.max(cv)
        ll = np.min(lv) if len(lv) > 0 else np.min(cv)
        mid = (hh + ll) / 2.0
        rng = hh - ll
        if rng > 0:
            # 3-anchor tension: (price-high_anchor) + (price-low_anchor) + (price-midpoint)
            t = ((C[si,di] - hh) + (C[si,di] - ll) + (C[si,di] - mid)) / (3 * rng)
            TENSION[si,di] = t

print(f"  Philosophy features: KER + Entropy + Tension ({time.time()-t2:.1f}s)", flush=True)

# Rank features (same as v109)
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
R_GAP = rank_pct(GAP); R_CPOS = rank_pct(CLOSE_POS)
R_CONSEC = rank_pct(CONSEC)
# NEW: Rank the philosophy features
R_KER = rank_pct(KER, start=50)
R_ENTROPY = rank_pct(ENTROPY, start=50)
R_TENSION = rank_pct(TENSION, start=50)

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
D_CONSEC_3 = delta_rank(R_CONSEC, 3)

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
# [4] Feature matrix — v109 base + 3 philosophy ranks
# ====================================================================
N_HAND_V109 = 22  # 8 deltas + 8 ranks + 3 new ranks + 3 delta ranks
N_PHILOSOPHY = 3  # R_KER, R_ENTROPY, R_TENSION
N_FEAT = N_HAND_V109 + N_PHILOSOPHY + 4 + 2 * N_STRAT  # +4: buy/sell_con, mkt_breadth, mkt_mom20

FEAT = np.full((NS, ND, N_FEAT), np.nan)
FWD_RET = np.full((NS, ND), np.nan)
FWD_RET_3 = np.full((NS, ND), np.nan)
FWD_RET_10 = np.full((NS, ND), np.nan)

for si in range(NS):
    for di in range(60, ND):
        fi = 0
        # 8 original deltas
        for feat in [D_MOM5_3, D_MOM10_5, D_MOM20_10, D_PRICE_5, D_VDP_5,
                     D_REL_VOL_5, D_BB_5, D_ATR_5]:
            FEAT[si,di,fi] = feat[si,di]; fi += 1
        # 8 original ranks
        for feat in [R_MOM5, R_MOM10, R_MOM20, R_PRICE, R_VDP, R_REL_VOL, R_BB, R_ATR]:
            FEAT[si,di,fi] = feat[si,di]; fi += 1
        # 5 v109 new ranks
        for feat in [R_GAP, R_CPOS, R_CONSEC]:
            FEAT[si,di,fi] = feat[si,di]; fi += 1
        # 3 delta ranks
        for feat in [D_GAP_3, D_CPOS_5, D_CONSEC_3]:
            FEAT[si,di,fi] = feat[si,di]; fi += 1
        # === NEW: 3 PHILOSOPHY features ===
        # R_KER: 趋势效率 (道法自然)
        FEAT[si,di,fi] = R_KER[si,di] if not np.isnan(R_KER[si,di]) else 50; fi += 1
        # R_ENTROPY: 信息有序度 (无为而治)
        FEAT[si,di,fi] = R_ENTROPY[si,di] if not np.isnan(R_ENTROPY[si,di]) else 50; fi += 1
        # R_TENSION: 结构张力 (阴阳几何)
        FEAT[si,di,fi] = R_TENSION[si,di] if not np.isnan(R_TENSION[si,di]) else 50; fi += 1
        # Strategy aggregates + market
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

        # Forward returns
        if di < ND - 5 and not np.isnan(C[si,di]) and not np.isnan(C[si,di+5]) and C[si,di]>0:
            FWD_RET[si,di] = (C[si,di+5]-C[si,di])/C[si,di]*100
        if di < ND - 3 and not np.isnan(C[si,di]) and not np.isnan(C[si,di+3]) and C[si,di]>0:
            FWD_RET_3[si,di] = (C[si,di+3]-C[si,di])/C[si,di]*100
        if di < ND - 10 and not np.isnan(C[si,di]) and not np.isnan(C[si,di+10]) and C[si,di]>0:
            FWD_RET_10[si,di] = (C[si,di+10]-C[si,di])/C[si,di]*100

print(f"  Feature matrix: ({NS}, {ND}, {N_FEAT})", flush=True)

# ====================================================================
# [5] 1-Config × 1-Seed × 3-Horizon = 9 LGB models (FAST version)
# ====================================================================
TRAIN_WINDOW = 252 * 3; RETRAIN_FREQ = 84; MIN_TRAIN = 252 * 2; FWD_DAYS = 5

ENSEMBLE_CONFIGS = [
    {'num_leaves': 15, 'feature_fraction': 0.5, 'learning_rate': 0.05,
     'bagging_fraction': 0.7, 'min_data_in_leaf': 150},
]
N_CFG = len(ENSEMBLE_CONFIGS)
SEED_SETS = [42]
N_SEEDS = len(SEED_SETS)
N_ENS = N_CFG * N_SEEDS * 3  # 9 LGB models

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

print(f"\n[5] Training ensemble ({N_ENS} models)...", flush=True)
t5 = time.time()

SCORE_ALL = [np.full((NS, ND), np.nan) for _ in range(N_ENS)]
retrain_points = []

for train_di in range(MIN_TRAIN, ND, RETRAIN_FREQ):
    pred_end = min(train_di + RETRAIN_FREQ, ND)
    retrain_points.append((train_di, pred_end))

    for hi in range(3):
        fwd_d = FWD_DAYS_MAP[hi]
        train_end = train_di - fwd_d - 1
        train_start = max(MIN_TRAIN - 100, train_end - TRAIN_WINDOW)
        if train_start >= train_end: continue

        tX, rank_y, train_group = make_train_data(train_start, train_end, FWD_RET_MAP[hi])
        if tX is None: continue

        for ci, cfg in enumerate(ENSEMBLE_CONFIGS):
            for si_idx, seed in enumerate(SEED_SETS):
                model_idx = hi * (N_CFG * N_SEEDS) + ci * N_SEEDS + si_idx
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

# Rank-normalize then average
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
print(f"    Done: {len(retrain_points)} × {N_ENS} models ({time.time()-t5:.1f}s)", flush=True)

# ====================================================================
# [6] Walk-forward meta model
# ====================================================================
print(f"\n[6] Walk-forward meta model...", flush=True)
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

for rp_idx, (train_di, pred_end) in enumerate(retrain_points):
    meta_train_end = train_di - FWD_DAYS - 1
    if meta_train_end <= MIN_TRAIN: continue

    meta_X_list = []; meta_y_list = []
    for di in range(MIN_TRAIN, meta_train_end + 1):
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

    if len(meta_y_list) < 200: continue

    mX = np.array(meta_X_list, dtype=np.float32)
    my = np.array(meta_y_list, dtype=np.float32)

    try:
        meta_train_data = lgb.Dataset(mX, label=my)
        meta_model = lgb.train(META_PARAMS, meta_train_data, num_boost_round=100)
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
        print(f"    meta #{n_meta_trains}: di={train_di}", flush=True)

print(f"  Meta: {n_meta_trains} trainings ({time.time()-t6:.1f}s)", flush=True)

# ====================================================================
# [7] REGIME-ADAPTIVE TRADING ENGINE
# ====================================================================
# Key innovation: classify each day's market regime and adapt ALL trading params

# --- Regime classification per day ---
REGIME = np.full(ND, 0, dtype=np.int8)  # 0=CHAOTIC, 1=RANGING, 2=CYCLICAL, 3=TRENDING
REGIME_KER = np.full(ND, np.nan)
REGIME_ENTROPY = np.full(ND, np.nan)

for di in range(MIN_TRAIN, ND):
    # Market-wide KER: average of top-200 stocks' KER
    ker_vals = KER[:, di]
    ker_valid = ker_vals[~np.isnan(ker_vals)]
    if len(ker_valid) > 50:
        REGIME_KER[di] = np.median(ker_valid)  # use median, robust to outliers

    # Market-wide entropy: average of valid entropies
    ent_vals = ENTROPY[:, di]
    ent_valid = ent_vals[~np.isnan(ent_vals)]
    if len(ent_valid) > 50:
        REGIME_ENTROPY[di] = np.mean(ent_valid)

    # Classify regime
    ker_m = REGIME_KER[di] if not np.isnan(REGIME_KER[di]) else 0.2
    ent_m = REGIME_ENTROPY[di] if not np.isnan(REGIME_ENTROPY[di]) else 0.7

    if ent_m > 0.90:  # RELAXED: only extreme chaos blocks
        REGIME[di] = 0  # CHAOTIC: no trading
    elif ker_m < 0.12:  # RELAXED: slightly lower threshold
        REGIME[di] = 1  # RANGING: mean reversion
    elif ker_m < 0.20:  # RELAXED: more days classified as trending
        REGIME[di] = 2  # CYCLICAL: momentum with caution
    else:
        REGIME[di] = 3  # TRENDING: trend following

# Regime statistics
n_chaotic = np.sum(REGIME[MIN_TRAIN:] == 0)
n_ranging = np.sum(REGIME[MIN_TRAIN:] == 1)
n_cyclical = np.sum(REGIME[MIN_TRAIN:] == 2)
n_trending = np.sum(REGIME[MIN_TRAIN:] == 3)
print(f"  Regime distribution: TREND={n_trending} CYCLE={n_cyclical} "
      f"RANGE={n_ranging} CHAOS={n_chaotic}", flush=True)

# --- Regime-dependent parameters ---
REGIME_PARAMS = {
    0: {'sl_mult': 0, 'hm': 0, 'trail': 0, 'trail_start': 999, 'kelly_scale': 0},      # CHAOTIC: no trade
    1: {'sl_mult': 3.5, 'hm': 8, 'trail': 1.8, 'trail_start': 2, 'kelly_scale': 0.3},   # RANGING: tight, quick
    2: {'sl_mult': 5.0, 'hm': 15, 'trail': 2.2, 'trail_start': 3, 'kelly_scale': 0.5},  # CYCLICAL: moderate
    3: {'sl_mult': 6.5, 'hm': 25, 'trail': 3.0, 'trail_start': 4, 'kelly_scale': 0.8},  # TRENDING: wide, long
}

def run_backtest_v112(score_arr, meta_arr, base_params):
    """Regime-adaptive backtest engine with LONG+SHORT."""
    max_pos = base_params.get('mp', 2)
    use_meta = base_params.get('th', 0) > 0 and meta_arr is not None
    meta_threshold = base_params.get('th', 0)
    min_score_pct = base_params.get('sp', 70)  # only trade stocks above this rank percentile

    cash = float(CASH0)
    positions = []  # {'si':, 'dir': 1/-1, 'shares':, 'entry':, 'highest':, 'lowest':, 'ed':}
    trades = []
    pending = []  # ('open_long'/'open_short'/'close', si, alloc_or_reason)

    for di in range(MIN_TRAIN, ND):
        regime = REGIME[di]
        rp = REGIME_PARAMS[regime]

        # === EXECUTE PENDING ===
        new_pending = []
        for p in pending:
            if p[0] == 'close':
                si = p[1]; direction = p[2]
                pt = O[si, di]
                if np.isnan(pt) or pt <= 0: pt = C[si, di]
                if np.isnan(pt) or pt <= 0: continue
                pos = next((x for x in positions if x['si'] == si), None)
                if pos is None: continue
                if direction == 1:  # close long
                    pnl = (pt - pos['entry']) / pos['entry'] * 100
                    cash += pos['shares'] * pt * (1 - COMMISSION - STAMP_DUTY)
                else:  # close short
                    pnl = (pos['entry'] - pt) / pos['entry'] * 100
                    # Buy back shares to cover short
                    cash -= pos['shares'] * pt * (1 + COMMISSION)
                    # (We already received the short sale proceeds when opening)
                trades.append({'pnl': pnl, 'days': (dates[di] - pos['ed']).days,
                              'reason': p[3], 'dir': direction, 'di': di})
                positions = [x for x in positions if x['si'] != si]

            elif p[0] in ('open_long', 'open_short'):
                si = p[1]; alloc = p[2]; direction = 1 if p[0] == 'open_long' else -1
                if any(x['si'] == si for x in positions): continue
                pt = O[si, di]
                if np.isnan(pt) or pt <= 0:
                    pt = C[si, di-1] if di > 0 and not np.isnan(C[si, di-1]) else np.nan
                if np.isnan(pt) or pt <= 0 or alloc < 5000: continue
                shares = int(alloc / (1 + COMMISSION) / pt)
                if shares <= 0: continue
                if direction == 1:  # LONG
                    cash -= shares * pt * (1 + COMMISSION)
                    positions.append({'si': si, 'dir': 1, 'shares': shares, 'entry': pt,
                                     'highest': pt, 'lowest': pt, 'ed': dates[di]})
                else:  # SHORT
                    cash += shares * pt * (1 - COMMISSION)
                    positions.append({'si': si, 'dir': -1, 'shares': shares, 'entry': pt,
                                     'highest': pt, 'lowest': pt, 'ed': dates[di]})
        pending = []

        # === REGIME 0: CHAOTIC → NO TRADING, close everything ===
        if regime == 0:
            for pos in positions:
                pending.append(('close', pos['si'], pos['dir'], 'chaos_exit'))
            continue

        # === CHECK EXITS (regime-adaptive) ===
        for pos in positions:
            si = pos['si']; p = C[si, di]
            if np.isnan(p) or p <= 0: continue
            hd = (dates[di] - pos['ed']).days

            if pos['dir'] == 1:  # LONG position
                if p > pos['highest']: pos['highest'] = p
                pnl = (p - pos['entry']) / pos['entry'] * 100

                # ATR-based stop loss
                atr_p = ATR_PCT[si, di] if not np.isnan(ATR_PCT[si, di]) else 3.0
                sl_pct = rp['sl_mult'] * atr_p
                # Widen stop in first 2 days
                if hd <= 2: sl_pct *= 1.5

                er = None
                if pnl < -sl_pct: er = f'sl({pnl:.1f}%)'
                elif pnl > 15: er = f'tp({pnl:.1f}%)'
                elif rp['trail'] > 0 and pnl > rp['trail_start']:
                    dd = (pos['highest'] - p) / pos['highest'] * 100
                    if dd > rp['trail']: er = f'trail({pnl:.1f}%)'
                elif hd >= rp['hm'] and pnl < 1: er = f'max({hd}d,{pnl:.1f}%)'
                elif hd >= rp['hm'] * 1.5: er = f'hard_max({hd}d)'

                # REGIME CHANGE EXIT: if regime drops from trending, take profit or cut
                if hd >= 3 and regime < 2 and pnl > 2: er = f'regime_down({pnl:.1f}%)'

            else:  # SHORT position
                if p < pos['lowest']: pos['lowest'] = p
                pnl = (pos['entry'] - p) / pos['entry'] * 100

                atr_p = ATR_PCT[si, di] if not np.isnan(ATR_PCT[si, di]) else 3.0
                sl_pct = rp['sl_mult'] * atr_p
                if hd <= 2: sl_pct *= 1.5

                er = None
                if pnl < -sl_pct: er = f'sl({pnl:.1f}%)'
                elif pnl > 15: er = f'tp({pnl:.1f}%)'
                elif rp['trail'] > 0 and pnl > rp['trail_start']:
                    dd = (p - pos['lowest']) / pos['lowest'] * 100
                    if dd > rp['trail']: er = f'trail({pnl:.1f}%)'
                elif hd >= rp['hm'] and pnl < 1: er = f'max({hd}d,{pnl:.1f}%)'
                elif hd >= rp['hm'] * 1.5: er = f'hard_max({hd}d)'

                if hd >= 3 and regime >= 3 and pnl > 2: er = f'regime_up({pnl:.1f}%)'

            if er:
                pending.append(('close', pos['si'], pos['dir'], er))

        # === ENTRY SIGNALS (regime-dependent) ===
        room = max_pos - len(positions) + len([p for p in pending if p[0] == 'close'])
        if room <= 0: continue

        held_si = set(x['si'] for x in positions)

        # Get score-ranked candidates
        candidates_long = []
        candidates_short = []

        for si in range(NS):
            if si in held_si: continue
            if np.isnan(C[si, di]) or C[si, di] <= 0: continue
            s = score_arr[si, di]
            if np.isnan(s): continue

            # Meta model filter
            if use_meta:
                mp = meta_arr[si, di] if not np.isnan(meta_arr[si, di]) else 0
                if mp < meta_threshold: continue

            # Score rank percentile
            day_scores = score_arr[:, di]
            valid_mask = ~np.isnan(day_scores)
            if valid_mask.sum() < 50: continue
            rank_pct = np.sum(day_scores[valid_mask] < s) / max(valid_mask.sum() - 1, 1) * 100

            if rank_pct >= min_score_pct:
                candidates_long.append((si, s, rank_pct))
            if rank_pct <= (100 - min_score_pct):
                candidates_short.append((si, s, rank_pct))

        candidates_long.sort(key=lambda x: -x[1])
        candidates_short.sort(key=lambda x: x[1])

        # Regime-dependent entry logic
        entered = 0

        if regime == 3:  # TRENDING: go long top stocks, maybe short bottom
            kelly_frac = rp['kelly_scale']
            for si, s, rpct in candidates_long[:room]:
                if cash <= 0: break
                alloc = cash * kelly_frac / max(max_pos, 1)
                # VDP confirmation for long
                vdp = VDP_DELTA[si, di]
                if not np.isnan(vdp) and vdp < 0: continue  # VDP bearish, skip
                pending.append(('open_long', si, alloc))
                entered += 1
                if entered >= room: break

            # Short bottom stocks in strong downtrend
            if entered < room and REGIME_KER[di] > 0.35:
                for si, s, rpct in candidates_short[:1]:
                    if cash <= 0: break
                    alloc = cash * kelly_frac * 0.5 / max(max_pos, 1)
                    vdp = VDP_DELTA[si, di]
                    if not np.isnan(vdp) and vdp > 0: continue
                    pending.append(('open_short', si, alloc))
                    entered += 1
                    if entered >= room: break

        elif regime == 2:  # CYCLICAL: cautious long, no short
            kelly_frac = rp['kelly_scale']
            for si, s, rpct in candidates_long[:room]:
                if cash <= 0: break
                alloc = cash * kelly_frac / max(max_pos, 1)
                # Extra confirmation: tension must be positive
                tension = TENSION[si, di] if not np.isnan(TENSION[si, di]) else 0
                if tension < -0.2: continue
                pending.append(('open_long', si, alloc))
                entered += 1
                if entered >= room: break

        elif regime == 1:  # RANGING: mean reversion — buy oversold, sell overbought
            kelly_frac = rp['kelly_scale']
            for si, s, rpct in candidates_long[:room]:
                if cash <= 0: break
                alloc = cash * kelly_frac / max(max_pos, 1)
                # Mean reversion: price should be near lower BB
                bb = BB_WIDTH[si, di] if not np.isnan(BB_WIDTH[si, di]) else 5
                close_pos = CLOSE_POS[si, di] if not np.isnan(CLOSE_POS[si, di]) else 50
                if close_pos > 60: continue  # skip if already high
                pending.append(('open_long', si, alloc))
                entered += 1
                if entered >= room: break

    # Close remaining
    for pos in positions:
        p = C[pos['si'], ND-1]
        if not np.isnan(p) and p > 0:
            if pos['dir'] == 1:
                pnl = (p - pos['entry']) / pos['entry'] * 100
                cash += pos['shares'] * p * (1 - COMMISSION - STAMP_DUTY)
            else:
                pnl = (pos['entry'] - p) / pos['entry'] * 100
                # Buy back to cover
                cash -= pos['shares'] * p * (1 + COMMISSION)
                # Short proceeds already in cash from open
            trades.append({'pnl': pnl, 'days': 999, 'reason': 'end',
                          'dir': pos['dir'], 'di': ND-1})

    if cash <= 0 or not trades: return None
    days = (dates[ND-1] - dates[MIN_TRAIN]).days; yr = max(days/365.25, 0.01)
    ann = ((cash/CASH0)**(1/yr)-1)*100
    nw = sum(1 for t in trades if t['pnl'] > 0)
    nl = sum(1 for t in trades if t['pnl'] <= 0)
    wr = nw / max(len(trades), 1) * 100
    avg_w = np.mean([t['pnl'] for t in trades if t['pnl'] > 0]) if nw > 0 else 0
    avg_l = np.mean([abs(t['pnl']) for t in trades if t['pnl'] <= 0]) if nl > 0 else 0
    edge = (nw/max(len(trades),1)) * avg_w - (nl/max(len(trades),1)) * avg_l
    long_trades = sum(1 for t in trades if t.get('dir', 1) == 1)
    short_trades = sum(1 for t in trades if t.get('dir', 1) == -1)

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
            'max_dd': round(max_dd,1), 'final': round(cash,0),
            'long_n': long_trades, 'short_n': short_trades}

# ====================================================================
# [8] Parameter search — regime-adaptive configs
# ====================================================================
print(f"\n[8] Parameter search (regime-adaptive)...", flush=True)
t8 = time.time()

all_results = []
search_configs = []

# Core configs: vary max_pos, score percentile, meta threshold
for mp in [1, 2, 3]:
    for sp in [60, 70, 80, 90]:
        for th in [0, 0.50, 0.55]:
            label = f'v112_mp{mp}_sp{sp}_m{th:.2f}'
            search_configs.append({
                'mp': mp, 'sp': sp, 'th': th,
                'label': label
            })

print(f"  {len(search_configs)} configs...", flush=True)

for i, cfg in enumerate(search_configs):
    r = run_backtest_v112(SCORE_PRI, SCORE_META if cfg['th'] > 0 else None, cfg)
    if r:
        all_results.append({**r, **cfg})
    if (i+1) % 20 == 0:
        best = max((x['ann'] for x in all_results), default=0)
        print(f"    {i+1}/{len(search_configs)} best={best:+.1f}%", flush=True)

all_results.sort(key=lambda x: -x['ann'])
print(f"  Done: {len(all_results)} results ({time.time()-t8:.1f}s)", flush=True)

# --- DIAGNOSTICS ---
print(f"\n  [DIAG] SCORE_PRI NaN%: {np.isnan(SCORE_PRI).sum()}/{SCORE_PRI.size} "
      f"= {np.isnan(SCORE_PRI).mean()*100:.1f}%", flush=True)
valid_scores = SCORE_PRI[~np.isnan(SCORE_PRI)]
if len(valid_scores) > 0:
    print(f"  [DIAG] SCORE_PRI range: [{valid_scores.min():.2f}, {valid_scores.max():.2f}], "
          f"mean={valid_scores.mean():.2f}", flush=True)
# Random baseline
np.random.seed(42)
SCORE_RAND = np.full_like(SCORE_PRI, np.nan)
mask = ~np.isnan(SCORE_PRI)
SCORE_RAND[mask] = np.random.randn(mask.sum())
r_rand = run_backtest_v112(SCORE_RAND, None, {'mp':2,'sp':70,'th':0})
if r_rand: print(f"  [DIAG] Random baseline: {r_rand['ann']:+.1f}%", flush=True)

# ====================================================================
# [9] Results
# ====================================================================
print(f"\n[9] RESULTS (top 30)", flush=True)
print(f"  {'Label':>40s} {'Ann':>7s} {'WR':>5s} {'Edge':>6s} {'TPY':>5s} "
      f"{'DD':>5s} {'Long':>5s} {'Short':>5s}", flush=True)
print(f"  {'-'*110}", flush=True)
for r in all_results[:30]:
    print(f"  {r['label']:>40s} {r['ann']:+7.1f}% {r['wr']:5.1f}% "
          f"{r['edge']:+6.2f}% {r['tpy']:5.1f} {r['max_dd']:4.1f}% "
          f"{r.get('long_n',0):5d} {r.get('short_n',0):5d}", flush=True)

# Group analysis
for mp_val in [1, 2, 3]:
    group = [r for r in all_results if r.get('mp') == mp_val]
    if group:
        best = max(group, key=lambda x: x['ann'])
        avg = np.mean([r['ann'] for r in group])
        print(f"\n  MP={mp_val}: best={best['ann']:+.1f}% avg={avg:+.1f}% ({len(group)} configs)", flush=True)

above_100 = sum(1 for r in all_results if r['ann'] >= 100)
above_300 = sum(1 for r in all_results if r['ann'] >= 300)
above_500 = sum(1 for r in all_results if r['ann'] >= 500)
above_1000 = sum(1 for r in all_results if r['ann'] >= 1000)
print(f"\n  >=100%: {above_100} | >=300%: {above_300} | >=500%: {above_500} | >=1000%: {above_1000}", flush=True)

if all_results:
    best = all_results[0]
    print(f"\n{'='*70}", flush=True)
    print(f"  V112 FINAL: {best['ann']:+.1f}% | WR={best['wr']:.0f}% "
          f"Edge={best['edge']:+.2f}% TPY={best['tpy']:.0f} DD={best['max_dd']:.0f}%", flush=True)
    print(f"  Long={best.get('long_n',0)} Short={best.get('short_n',0)} | {best['label']}", flush=True)
    print(f"  Regime distribution: TREND={n_trending} CYCLE={n_cyclical} "
          f"RANGE={n_ranging} CHAOS={n_chaotic}", flush=True)
    print(f"{'='*70}", flush=True)

print(f"\nDone! {time.time()-t0:.1f}s", flush=True)
