"""
V110 — Philosophy-Driven LambdaRank Pipeline (NO LOOK-AHEAD)
==============================================================
Based on V109 (+413%), enhanced with:
  1. Philosophy features: KER (趋势效率), Shannon Entropy (无为门), Structural Tension (几何)
  2. Top-2 stock selection (集中兵力, max 2 stocks)
  3. Kelly criterion position sizing (中庸之道)
  4. 7-layer architecture from market_philosophy.md
  5. Target: 600% annualized

NO LOOK-AHEAD FIX: All features use di-1 data (yesterday's close).
  FEAT[si, di] is populated from raw features at di-1.
  FWD_RET[si, di] = return from di to di+5 (label unchanged).
  Model learns: yesterday's data → predict today-to-5d return.
  Trade executes at O[si, di] (today's open) — correct.

Philosophy layers:
  Layer 1: Entropy Gate (无为) — H(X) > 0.8*H_max → block
  Layer 2: KER Regime (体制) — trend/range routing
  Layer 3: LambdaRank Top-2 (集中兵力)
  Layer 4: Signal Generation (阴阳)
  Layer 5: VDP Confirmation (虚实)
  Layer 6: Kelly Sizing (中庸)
  Layer 7: EVT ATR Stop (无常)
"""
import sys, os, time, warnings, pickle
import numpy as np, pandas as pd
import lightgbm as lgb
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.data_loader import list_available_symbols, load_stock_data

COMMISSION = 0.0003; STAMP_DUTY = 0.001; CASH0 = 500_000

print("=" * 70, flush=True)
print("  V110 — Philosophy-Driven LambdaRank (Top-2 + Kelly + 7 Layers)", flush=True)
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
for di in range(21, ND):
    # Use di-1 data: no look-ahead (market breadth known at yesterday's close)
    d = di - 1
    above = sum(1 for si in range(NS)
                if not np.isnan(C[si,d]) and not np.isnan(C[si,d-20]) and C[si,d-20]>0
                and (C[si,d]-C[si,d-20])/C[si,d-20] > 0)
    total = sum(1 for si in range(NS) if not np.isnan(C[si,d]) and not np.isnan(C[si,d-20]) and C[si,d-20]>0)
    if total > 100: MKT_BREADTH[di] = above / total * 100
    r20 = [C[si,d]/C[si,d-20]-1 for si in range(NS)
           if not np.isnan(C[si,d]) and not np.isnan(C[si,d-20]) and C[si,d-20]>0]
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
    for di in range(61, ND):  # Start from 61 (need di-1 >= 60 for all features)
        fi = 0
        pdi = di - 1  # Previous day: no look-ahead
        # 8 original deltas — use di-1
        for feat in [D_MOM5_3, D_MOM10_5, D_MOM20_10, D_PRICE_5, D_VDP_5,
                     D_REL_VOL_5, D_BB_5, D_ATR_5]:
            FEAT[si,di,fi] = feat[si,pdi]; fi += 1
        # 8 original ranks — use di-1
        for feat in [R_MOM5, R_MOM10, R_MOM20, R_PRICE, R_VDP, R_REL_VOL, R_BB, R_ATR]:
            FEAT[si,di,fi] = feat[si,pdi]; fi += 1
        # 5 v109 new ranks — use di-1
        for feat in [R_GAP, R_CPOS, R_CONSEC]:
            FEAT[si,di,fi] = feat[si,pdi]; fi += 1
        # 3 delta ranks — use di-1
        for feat in [D_GAP_3, D_CPOS_5, D_CONSEC_3]:
            FEAT[si,di,fi] = feat[si,pdi]; fi += 1
        # === NEW: 3 PHILOSOPHY features — use di-1 ===
        # R_KER: 趋势效率 (道法自然)
        FEAT[si,di,fi] = R_KER[si,pdi] if not np.isnan(R_KER[si,pdi]) else 50; fi += 1
        # R_ENTROPY: 信息有序度 (无为而治)
        FEAT[si,di,fi] = R_ENTROPY[si,pdi] if not np.isnan(R_ENTROPY[si,pdi]) else 50; fi += 1
        # R_TENSION: 结构张力 (阴阳几何)
        FEAT[si,di,fi] = R_TENSION[si,pdi] if not np.isnan(R_TENSION[si,pdi]) else 50; fi += 1
        # Strategy aggregates + market — use di-1 signals
        BUY_CON = STRAT_BUY[si,pdi].sum(); SELL_CON = STRAT_SELL[si,pdi].sum()
        FEAT[si,di,fi] = BUY_CON; fi += 1
        FEAT[si,di,fi] = SELL_CON; fi += 1
        if not np.isnan(MKT_BREADTH[di]): FEAT[si,di,fi] = MKT_BREADTH[di]
        fi += 1
        if not np.isnan(MKT_MOM20_VAL[di]): FEAT[si,di,fi] = MKT_MOM20_VAL[di]
        fi += 1
        for ki in range(N_STRAT):
            FEAT[si,di,fi] = STRAT_BUY[si,pdi,ki]; fi += 1
        for ki in range(N_STRAT):
            FEAT[si,di,fi] = STRAT_SELL[si,pdi,ki]; fi += 1

        # Forward returns
        if di < ND - 5 and not np.isnan(C[si,di]) and not np.isnan(C[si,di+5]) and C[si,di]>0:
            FWD_RET[si,di] = (C[si,di+5]-C[si,di])/C[si,di]*100
        if di < ND - 3 and not np.isnan(C[si,di]) and not np.isnan(C[si,di+3]) and C[si,di]>0:
            FWD_RET_3[si,di] = (C[si,di+3]-C[si,di])/C[si,di]*100
        if di < ND - 10 and not np.isnan(C[si,di]) and not np.isnan(C[si,di+10]) and C[si,di]>0:
            FWD_RET_10[si,di] = (C[si,di+10]-C[si,di])/C[si,di]*100

print(f"  Feature matrix: ({NS}, {ND}, {N_FEAT})", flush=True)

# ====================================================================
# [5] 3-Config × 3-Seed × 3-Horizon = 27 LGB models
# ====================================================================
TRAIN_WINDOW = 252 * 3; RETRAIN_FREQ = 42; MIN_TRAIN = 252 * 2; FWD_DAYS = 5

ENSEMBLE_CONFIGS = [
    {'num_leaves': 7,  'feature_fraction': 0.4, 'learning_rate': 0.05,
     'bagging_fraction': 0.7, 'min_data_in_leaf': 100},
    {'num_leaves': 15, 'feature_fraction': 0.6, 'learning_rate': 0.03,
     'bagging_fraction': 0.8, 'min_data_in_leaf': 150},
    {'num_leaves': 5,  'feature_fraction': 0.3, 'learning_rate': 0.07,
     'bagging_fraction': 0.6, 'min_data_in_leaf': 80},
]
N_CFG = len(ENSEMBLE_CONFIGS)
SEED_SETS = [42, 123, 256]
N_SEEDS = len(SEED_SETS)
N_ENS = N_CFG * N_SEEDS * 3  # 27 LGB models

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
# [7] Backtest with Top-2 + Kelly + 7-Layer Architecture
# ====================================================================
def run_backtest(score_arr, meta_arr, params):
    sl_pct = params['sl']; tp_pct = params.get('tp', 50)
    hold_max = params['hm']; trail_pct = 1; trail_start = params['ts']
    meta_threshold = params.get('th', 0)
    max_pos = params.get('mp', 2)  # DEFAULT TOP-2
    atr_adaptive = params.get('atr', False)
    atr_min = params.get('amin', 0.6); atr_rng = params.get('arng', 0.8)
    use_entropy_gate = params.get('entropy_gate', False)
    use_kelly = params.get('kelly', False)
    kelly_fraction = params.get('kf', 0.5)  # half-Kelly default

    cash = float(CASH0); positions = []; trades = []; pending = []

    for di in range(MIN_TRAIN, ND):
        # Execute pending
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
                si = p[1]; alloc = p[2]  # allocation amount from Kelly
                if any(x['si'] == si for x in positions): continue
                pt = O[si, di]
                if np.isnan(pt) or pt <= 0:
                    pt = C[si, di-1] if di > 0 and not np.isnan(C[si, di-1]) else np.nan
                if not np.isnan(pt) and pt > 0 and alloc > 10000:
                    shares = int(alloc / (1 + COMMISSION) / pt)
                    if shares > 0:
                        cash -= shares * pt * (1 + COMMISSION)
                        pos = {'si': si, 'shares': shares, 'entry': pt,
                               'highest': pt, 'ed': dates[di], 'alloc': alloc}
                        if atr_adaptive and not np.isnan(ATR_PCT[si, di]):
                            atr_p = ATR_PCT[si, di]
                            atr_vals = ATR_PCT[:, di]
                            atr_valid = atr_vals[~np.isnan(atr_vals)]
                            if len(atr_valid) > 50:
                                pct = np.sum(atr_valid < atr_p) / max(len(atr_valid)-1,1)
                                pos['atr_scale'] = atr_min + atr_rng * pct
                                pos['hm_scale'] = 0.7 + 0.6 * pct
                            else:
                                pos['atr_scale'] = 1.0; pos['hm_scale'] = 1.0
                        else:
                            pos['atr_scale'] = 1.0; pos['hm_scale'] = 1.0
                        positions.append(pos)
        pending = []

        # Check exits
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

        # === LAYER 1: Entropy Gate (无为) ===
        # Market-wide entropy — skip trading if too chaotic
        if use_entropy_gate:
            ent_vals = ENTROPY[:, di]
            ent_valid = ent_vals[~np.isnan(ent_vals)]
            if len(ent_valid) > 50:
                avg_entropy = np.mean(ent_valid)
                if avg_entropy > 0.8:  # H > 0.8*H_max → too chaotic
                    continue

        # === LAYER 2+3+6: Top-K entry with Kelly sizing ===
        room = max_pos - len(positions) + len([p for p in pending if p[0]=='close'])
        if room > 0:
            held_si = set(x['si'] for x in positions)
            candidates = []
            for si in range(NS):
                s = score_arr[si, di]
                if np.isnan(s): continue
                if si in held_si: continue
                candidates.append((si, s))
            candidates.sort(key=lambda x: -x[1])

            entered = 0
            for si, score in candidates[:room*3]:
                if meta_threshold > 0 and meta_arr is not None:
                    meta_prob = meta_arr[si, di]
                    if np.isnan(meta_prob) or meta_prob < meta_threshold: continue

                # === LAYER 6: Kelly Sizing (中庸) ===
                if use_kelly:
                    # Estimate win probability from meta model
                    if not np.isnan(meta_arr[si, di]) if meta_arr is not None else False:
                        p = meta_arr[si, di]
                    else:
                        p = 0.55
                    # Kelly: f* = (bp - q) / b, b=avgW/avgL≈1.5
                    b = 1.5; q = 1 - p
                    f_star = (b * p - q) / b
                    f_actual = max(0.1, min(0.5, f_star * kelly_fraction))  # half-Kelly
                    alloc = cash * f_actual
                else:
                    alloc = cash / max(max_pos - len(positions) + 1, 1)

                pending.append(('open_long', si, alloc))
                entered += 1
                if entered >= room: break

    # Close remaining
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
            'max_dd': round(max_dd,1), 'final': round(cash,0)}

# ====================================================================
# [8] Parameter search — Top-2 + Kelly focus
# ====================================================================
print(f"\n[8] Parameter search (Top-2 + Kelly)...", flush=True)
t8 = time.time()

all_results = []
search_configs = []

# Baseline: Top-1 (compare with v109)
for hm in [5, 6, 7]:
    for sl in [4, 5, 6]:
        search_configs.append({'hm':hm,'sl':sl,'ts':4,'th':0,'mp':1,'atr':False,
                              'kelly':False,'entropy_gate':False,
                              'label':f'v109_mp1_h{hm}s{sl}'})

# Top-2 baseline
for hm in [5, 6, 7]:
    for sl in [4, 5, 6]:
        search_configs.append({'hm':hm,'sl':sl,'ts':4,'th':0,'mp':2,'atr':False,
                              'kelly':False,'entropy_gate':False,
                              'label':f'mp2_h{hm}s{sl}'})

# Top-2 + meta
for th in [0.50, 0.55, 0.60]:
    for hm in [5, 6, 7]:
        for sl in [4, 5, 6]:
            search_configs.append({'hm':hm,'sl':sl,'ts':4,'th':th,'mp':2,'atr':False,
                                  'kelly':False,'entropy_gate':False,
                                  'label':f'mp2_m{th:.2f}_h{hm}s{sl}'})

# Top-2 + Kelly (THE KEY INNOVATION)
for th in [0.50, 0.55, 0.60]:
    for hm in [5, 6, 7]:
        for sl in [4, 5, 6]:
            search_configs.append({'hm':hm,'sl':sl,'ts':4,'th':th,'mp':2,'atr':False,
                                  'kelly':True,'kf':0.5,'entropy_gate':False,
                                  'label':f'kelly_m{th:.2f}_h{hm}s{sl}'})

# Top-2 + Kelly + Entropy Gate (7-LAYER)
for th in [0.55, 0.60]:
    for hm in [5, 6, 7]:
        for sl in [4, 5, 6]:
            search_configs.append({'hm':hm,'sl':sl,'ts':4,'th':th,'mp':2,'atr':False,
                                  'kelly':True,'kf':0.5,'entropy_gate':True,
                                  'label':f'7layer_m{th:.2f}_h{hm}s{sl}'})

# Top-2 + ATR + Kelly (full combo)
for th in [0.55, 0.60]:
    for hm in [6, 7]:
        for sl in [4, 5]:
            search_configs.append({'hm':hm,'sl':sl,'ts':4,'th':th,'mp':2,'atr':True,
                                  'amin':0.6,'arng':0.8,'kelly':True,'kf':0.5,
                                  'entropy_gate':True,
                                  'label':f'full_m{th:.2f}_h{hm}s{sl}'})

print(f"  {len(search_configs)} configs...", flush=True)

for i, cfg in enumerate(search_configs):
    r = run_backtest(SCORE_PRI, SCORE_META if cfg['th'] > 0 else None, cfg)
    if r:
        all_results.append({**r, **cfg})
    if (i+1) % 100 == 0:
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
else:
    print(f"  [DIAG] SCORE_PRI is ALL NaN!", flush=True)
# Check how many days have valid scores
days_with_scores = sum(1 for di in range(MIN_TRAIN, ND) if (~np.isnan(SCORE_PRI[:, di])).sum() > 0)
print(f"  [DIAG] Days with scores: {days_with_scores}/{ND-MIN_TRAIN}", flush=True)
# Check meta
if SCORE_META is not None:
    meta_valid = (~np.isnan(SCORE_META)).sum()
    print(f"  [DIAG] SCORE_META valid: {meta_valid}/{SCORE_META.size}", flush=True)
# Test first config directly
test_cfg = {'hm':6,'sl':5,'ts':4,'th':0,'mp':1,'atr':False,'kelly':False,'entropy_gate':False}
r_test = run_backtest(SCORE_PRI, None, test_cfg)
print(f"  [DIAG] Test mp1 baseline: {r_test}", flush=True)

# ====================================================================
# [9] Bug check
# ====================================================================
print(f"\n[9] Bug check...", flush=True)
np.random.seed(42)
SCORE_RAND = np.full_like(SCORE_PRI, np.nan)
mask = ~np.isnan(SCORE_PRI)
SCORE_RAND[mask] = np.random.randn(mask.sum())
r_rand = run_backtest(SCORE_RAND, None, {'hm':6,'sl':6,'ts':4,'th':0,'mp':2,'atr':False,'kelly':False,'entropy_gate':False})
if r_rand: print(f"  Random: {r_rand['ann']:+.1f}%", flush=True)

SCORE_REV = -SCORE_PRI
r_rev = run_backtest(SCORE_REV, None, {'hm':6,'sl':6,'ts':4,'th':0,'mp':2,'atr':False,'kelly':False,'entropy_gate':False})
if r_rev: print(f"  Reversed: {r_rev['ann']:+.1f}%", flush=True)

# ====================================================================
# [10] Results
# ====================================================================
print(f"\n[10] RESULTS (top 30)", flush=True)
print(f"  {'Label':>35s} {'MP':>3s} {'HM':>3s} {'SL':>3s} | "
      f"{'Ann':>7s} {'WR':>5s} {'Edge':>6s} {'TPY':>5s} {'DD':>5s}", flush=True)
print(f"  {'-'*100}", flush=True)
for r in all_results[:30]:
    print(f"  {r['label']:>35s} {r['mp']:3d} {r['hm']:3d} {r['sl']:3d} | "
          f"{r['ann']:+7.1f}% {r['wr']:5.1f}% {r['edge']:+6.2f}% "
          f"{r['tpy']:5.1f} {r['max_dd']:4.1f}%", flush=True)

# Compare groups
v109_like = [r for r in all_results if r.get('mp') == 1 and not r.get('kelly')]
mp2_base = [r for r in all_results if r.get('mp') == 2 and not r.get('kelly') and r.get('th',0) == 0]
mp2_meta = [r for r in all_results if r.get('mp') == 2 and not r.get('kelly') and r.get('th',0) > 0]
kelly = [r for r in all_results if r.get('kelly')]
full_7layer = [r for r in all_results if r.get('kelly') and r.get('entropy_gate')]

def group_stats(group, name):
    if not group: return
    best = max(group, key=lambda x: x['ann'])
    avg = np.mean([r['ann'] for r in group])
    print(f"\n  {name}: best={best['ann']:+.1f}% avg={avg:+.1f}% ({len(group)} configs)", flush=True)

group_stats(v109_like, "V109-baseline (mp=1)")
group_stats(mp2_base, "Top-2 baseline")
group_stats(mp2_meta, "Top-2 + Meta")
group_stats(kelly, "Top-2 + Kelly (中庸)")
group_stats(full_7layer, "7-Layer (Kelly+Entropy)")

above_300 = sum(1 for r in all_results if r['ann'] >= 300)
above_500 = sum(1 for r in all_results if r['ann'] >= 500)
above_800 = sum(1 for r in all_results if r['ann'] >= 800)
print(f"\n  >=300%: {above_300} | >=500%: {above_500} | >=800%: {above_800}", flush=True)

if all_results:
    best = all_results[0]
    print(f"\n{'='*70}", flush=True)
    print(f"  V110 FINAL: {best['ann']:+.1f}% | WR={best['wr']:.0f}% "
          f"Edge={best['edge']:+.2f}% TPY={best['tpy']:.0f} DD={best['max_dd']:.0f}%", flush=True)
    print(f"  {N_ENS} models | {N_FEAT} features (+3 philosophy) | {best['label']}", flush=True)
    print(f"  Architecture: Top-{best['mp']} Kelly={'Yes' if best.get('kelly') else 'No'} "
          f"EntropyGate={'Yes' if best.get('entropy_gate') else 'No'}", flush=True)
    print(f"{'='*70}", flush=True)

print(f"\nDone! {time.time()-t0:.1f}s", flush=True)
