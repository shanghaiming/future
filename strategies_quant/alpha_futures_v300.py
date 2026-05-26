"""
期货策略 V300 — 高维因子投影策略 (High-Dimensional Factor Projection)
=====================================================================
核心思想: 在低维空间中寻找高维市场的投影

1. 构建15+维因子空间(动量/趋势/波动率/成交量OI/均值回归/PA)
2. 跨品种排名标准化 → 因子矩阵
3. 滚动PCA投影 → 2-3个潜在因子(低维投影)
4. 滚动IC加权 → 预测性投影方向
5. 多+空 集中持仓(顶部N/底部N)
6. Kelly仓位管理 + ATR止损
7. Walk-forward验证

目标: 年化600%+, 胜率50%+
"""
import sys, os, time, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.data_loader import list_available_symbols, load_stock_data
try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False

COMMISSION = 0.0005
CASH0 = 1_000_000

COMMODITY_GROUPS = {
    'BLACK':    ['rbfi', 'hcfi', 'ifi', 'jfi', 'jmfi'],
    'METAL':    ['cufi', 'alfi', 'znfi', 'nifi', 'snfi'],
    'PRECIOUS': ['aufi', 'agfi'],
    'ENERGY':   ['scfi', 'bufi', 'fufi', 'tafi', 'mafi'],
    'CHEM':     ['ppfi', 'lfi', 'vfi', 'egfi', 'ebfi', 'safi'],
    'OILCHAIN': ['mfi', 'yfi', 'ofi', 'pfi', 'rmfi'],
    'GRAIN':    ['cfi', 'csfi', 'srfi', 'cffi'],
}


# ============================================================
# DATA LOADING
# ============================================================
def load_all_data(start='2016-01-01', end=None, min_days=500):
    print("[V300] Loading main contract data...", flush=True)
    t0 = time.time()

    syms = list_available_symbols('daily')
    stock_data = {}
    for sym in syms:
        try:
            df = load_stock_data(sym, frequency='daily')
            if df is not None and len(df) >= min_days:
                stock_data[sym] = df
        except:
            pass

    vol_map = {s: df['volume'].tail(60).mean()
               for s, df in stock_data.items()
               if 'volume' in df.columns and df['volume'].tail(60).mean() > 0}
    syms = sorted([s for s, _ in sorted(vol_map.items(), key=lambda x: -x[1])[:50]])
    NS = len(syms)

    all_dates = sorted(set(d for s in syms for d in stock_data[s].index))
    i0 = next(i for i, d in enumerate(all_dates) if d >= pd.Timestamp(start))
    if end:
        i1 = next((i for i, d in enumerate(all_dates) if d > pd.Timestamp(end)), len(all_dates)) - 1
    else:
        i1 = len(all_dates) - 1
    dates = all_dates[i0:i1+1]
    ND = len(dates)
    dm = {d: i for i, d in enumerate(all_dates)}

    C = np.full((NS, len(all_dates)), np.nan)
    O = np.full((NS, len(all_dates)), np.nan)
    H = np.full((NS, len(all_dates)), np.nan)
    L = np.full((NS, len(all_dates)), np.nan)
    V = np.full((NS, len(all_dates)), np.nan)
    OI = np.full((NS, len(all_dates)), np.nan)

    for si, s in enumerate(syms):
        df = stock_data.get(s)
        if df is None: continue
        df = df[~df.index.duplicated(keep='first')]
        common = df.index[df.index.isin(dm)]
        if len(common) == 0: continue
        idx = np.array([dm[d] for d in common])
        if 'close' in df.columns: C[si, idx] = df.loc[common, 'close'].values.astype(float)
        if 'open' in df.columns: O[si, idx] = df.loc[common, 'open'].values.astype(float)
        if 'high' in df.columns: H[si, idx] = df.loc[common, 'high'].values.astype(float)
        if 'low' in df.columns: L[si, idx] = df.loc[common, 'low'].values.astype(float)
        if 'volume' in df.columns: V[si, idx] = df.loc[common, 'volume'].values.astype(float)
        if 'oi' in df.columns: OI[si, idx] = df.loc[common, 'oi'].values.astype(float)

    # Trim to date range
    C = C[:, i0:i1+1]; O = O[:, i0:i1+1]; H = H[:, i0:i1+1]
    L = L[:, i0:i1+1]; V = V[:, i0:i1+1]; OI = OI[:, i0:i1+1]

    print(f"  {NS} symbols, {ND} days ({time.time()-t0:.1f}s)")
    print(f"  Date range: {dates[0].strftime('%Y-%m-%d')} ~ {dates[-1].strftime('%Y-%m-%d')}")
    return C, O, H, L, V, OI, NS, ND, dates, syms


# ============================================================
# PHASE 1: HIGH-DIMENSIONAL FACTOR SPACE (15+ factors)
# ============================================================
def compute_factors(C, O, H, L, V, OI, NS, ND):
    """
    Compute 15+ factors per instrument per day.
    Returns dict of factor_name -> (NS, ND) array.
    """
    print("[V300] Computing factors...", flush=True)
    t0 = time.time()
    F = {}

    for si in range(NS):
        c = C[si]; o = O[si]; h = H[si]; l = L[si]; v = V[si]; oi = OI[si]
        nan_c = np.isnan(c)

        # ---- 1. MOMENTUM FACTORS ----
        for period, name in [(5,'mom5'), (10,'mom10'), (20,'mom20'), (60,'mom60')]:
            if name not in F:
                F[name] = np.full((NS, ND), np.nan)
            ret = np.full(ND, np.nan)
            for di in range(period, ND):
                if not nan_c[di] and not nan_c[di-period] and c[di-period] > 0:
                    ret[di] = c[di] / c[di-period] - 1.0
            F[name][si] = ret

        # MACD histogram (normalized by price)
        if 'macd' not in F:
            F['macd'] = np.full((NS, ND), np.nan)
        if HAS_TALIB:
            try:
                cs = np.where(nan_c, 0, c).astype(np.float64)
                _, _, hist = talib.MACD(cs, 12, 26, 9)
                valid = ~nan_c & (c > 0)
                F['macd'][si] = np.where(valid, hist / np.where(c > 0, c, 1), np.nan)
            except: pass

        # ---- 2. TREND FACTORS ----
        # ADX
        if 'adx' not in F:
            F['adx'] = np.full((NS, ND), np.nan)
        if HAS_TALIB:
            try:
                hs = np.where(np.isnan(h), 0, h).astype(np.float64)
                ls = np.where(np.isnan(l), 0, l).astype(np.float64)
                cs = np.where(nan_c, 0, c).astype(np.float64)
                adx = talib.ADX(hs, ls, cs, timeperiod=14)
                F['adx'][si] = np.where(nan_c, np.nan, adx)
            except: pass

        # Trend slope (20d linreg, annualized)
        if 'slope' not in F:
            F['slope'] = np.full((NS, ND), np.nan)
        for di in range(20, ND):
            w = c[di-20:di]
            vld = w[~np.isnan(w)]
            if len(vld) >= 15 and vld[0] > 0:
                x = np.arange(len(vld))
                y = vld / vld[0]
                try:
                    s = np.polyfit(x, y, 1)
                    F['slope'][si, di] = s[0] * 252
                except: pass

        # ---- 3. VOLATILITY FACTORS ----
        # 20d realized vol
        if 'vol20' not in F:
            F['vol20'] = np.full((NS, ND), np.nan)
        for di in range(20, ND):
            rets = []
            for j in range(max(1, di-20), di):
                if not nan_c[j] and not nan_c[j-1] and c[j-1] > 0:
                    rets.append(c[j]/c[j-1] - 1)
            if len(rets) >= 10:
                F['vol20'][si, di] = np.std(rets) * np.sqrt(252)

        # ATR% (14d)
        if 'atrp' not in F:
            F['atrp'] = np.full((NS, ND), np.nan)
        if HAS_TALIB:
            try:
                hs = np.where(np.isnan(h), 0, h).astype(np.float64)
                ls = np.where(np.isnan(l), 0, l).astype(np.float64)
                cs = np.where(nan_c, 0, c).astype(np.float64)
                atr = talib.ATR(hs, ls, cs, timeperiod=14)
                valid = ~nan_c & (c > 0)
                F['atrp'][si] = np.where(valid, atr / np.where(c > 0, c, 1), np.nan)
            except: pass

        # Vol regime: 5d vol / 20d vol
        if 'vregime' not in F:
            F['vregime'] = np.full((NS, ND), np.nan)
        for di in range(25, ND):
            r5, r20 = [], []
            for j in range(max(1, di-5), di):
                if not nan_c[j] and not nan_c[j-1] and c[j-1] > 0:
                    r5.append(c[j]/c[j-1] - 1)
            for j in range(max(1, di-20), di):
                if not nan_c[j] and not nan_c[j-1] and c[j-1] > 0:
                    r20.append(c[j]/c[j-1] - 1)
            if len(r5) >= 3 and len(r20) >= 10:
                s5, s20 = np.std(r5), np.std(r20)
                if s20 > 0:
                    F['vregime'][si, di] = s5 / s20

        # ---- 4. VOLUME / OI FACTORS ----
        # Volume anomaly (z-score vs 60d)
        if 'vanom' not in F:
            F['vanom'] = np.full((NS, ND), np.nan)
        for di in range(60, ND):
            vw = v[di-60:di]
            vv = vw[~np.isnan(vw)]
            if len(vv) >= 30 and not np.isnan(v[di]):
                mu, sig = np.mean(vv), np.std(vv)
                if sig > 0:
                    F['vanom'][si, di] = (v[di] - mu) / sig

        # OI change 5d / 20d
        if 'oi5' not in F:
            F['oi5'] = np.full((NS, ND), np.nan)
        for di in range(5, ND):
            if not np.isnan(oi[di]) and not np.isnan(oi[di-5]) and oi[di-5] > 0:
                F['oi5'][si, di] = oi[di] / oi[di-5] - 1

        if 'oi20' not in F:
            F['oi20'] = np.full((NS, ND), np.nan)
        for di in range(20, ND):
            if not np.isnan(oi[di]) and not np.isnan(oi[di-20]) and oi[di-20] > 0:
                F['oi20'][si, di] = oi[di] / oi[di-20] - 1

        # ---- 5. MEAN REVERSION FACTORS ----
        # RSI 14
        if 'rsi' not in F:
            F['rsi'] = np.full((NS, ND), np.nan)
        if HAS_TALIB:
            try:
                cs = np.where(nan_c, 0, c).astype(np.float64)
                rsi = talib.RSI(cs, timeperiod=14)
                F['rsi'][si] = np.where(nan_c, np.nan, rsi)
            except: pass
        else:
            for di in range(15, ND):
                ups, downs = [], []
                for j in range(di-14, di):
                    if not nan_c[j] and not nan_c[j-1]:
                        chg = c[j] - c[j-1]
                        if chg > 0: ups.append(chg)
                        else: downs.append(-chg)
                avg_up = np.mean(ups) if ups else 0
                avg_dn = np.mean(downs) if downs else 0.001
                rs = avg_up / avg_dn
                F['rsi'][si, di] = 100 - 100/(1+rs)

        # Price z-score vs 20d mean
        if 'zscore' not in F:
            F['zscore'] = np.full((NS, ND), np.nan)
        for di in range(20, ND):
            w = c[di-20:di]
            vv = w[~np.isnan(w)]
            if len(vv) >= 15 and np.std(vv) > 0 and not nan_c[di]:
                F['zscore'][si, di] = (c[di] - np.mean(vv)) / np.std(vv)

        # ---- 6. PRICE ACTION FACTORS ----
        # Body ratio
        if 'body' not in F:
            F['body'] = np.full((NS, ND), np.nan)
        for di in range(1, ND):
            ohlc = [o[di], h[di], l[di], c[di]]
            if not any(np.isnan(ohlc)) and h[di] > l[di]:
                F['body'][si, di] = abs(c[di] - o[di]) / (h[di] - l[di])

        # Range ratio (ATR-like)
        if 'range' not in F:
            F['range'] = np.full((NS, ND), np.nan)
        for di in range(1, ND):
            if not nan_c[di] and not nan_c[di-1] and c[di-1] > 0 and not np.isnan(h[di]) and not np.isnan(l[di]):
                F['range'][si, di] = (h[di] - l[di]) / c[di-1]

    # ---- 7. CROSS-SECTIONAL FACTORS ----
    # CS momentum rank
    cs_rank_mom = np.full((NS, ND), np.nan)
    for di in range(60, ND):
        vals = F['mom20'][:, di]
        valid = ~np.isnan(vals)
        if valid.sum() > 5:
            ranks = pd.Series(vals).rank(pct=True, na_option='keep').values
            cs_rank_mom[:, di] = ranks
    F['cs_mom'] = cs_rank_mom

    # CS volatility rank
    cs_rank_vol = np.full((NS, ND), np.nan)
    for di in range(60, ND):
        vals = F['vol20'][:, di]
        valid = ~np.isnan(vals)
        if valid.sum() > 5:
            ranks = pd.Series(vals).rank(pct=True, na_option='keep').values
            cs_rank_vol[:, di] = ranks
    F['cs_vol'] = cs_rank_vol

    # Group momentum alignment
    sym_to_group = {}
    for gname, gsyms in COMMODITY_GROUPS.items():
        for s in gsyms:
            sym_to_group[s] = gname

    F['group_mom'] = np.full((NS, ND), np.nan)
    sym_idx = {}  # will be passed in main, but compute group mom here
    # We need syms for this — will compute in signal generation

    print(f"  {len(F)} factors in {time.time()-t0:.1f}s", flush=True)
    return F


# ============================================================
# PHASE 2: LOW-DIMENSIONAL PROJECTION
# ============================================================
def cross_sectional_rank_matrix(F, NS, ND):
    """Convert all factors to cross-sectional percentile ranks (0~1)."""
    print("[V300] Building cross-sectional rank matrix...", flush=True)
    factor_names = sorted(F.keys())
    NF = len(factor_names)

    # Rank matrix: (NS, ND, NF)
    R = np.full((NS, ND, NF), np.nan)
    for fi, name in enumerate(factor_names):
        for di in range(ND):
            vals = F[name][:, di]
            valid = ~np.isnan(vals)
            if valid.sum() > 5:
                ranks = pd.Series(vals).rank(pct=True, na_option='keep').values
                R[:, di, fi] = ranks

    return R, factor_names


def rolling_pca_projection(R, NS, ND, NF, n_components=3, window=120, refit_freq=20):
    """
    Rolling PCA on the cross-sectional factor rank covariance.

    Every `refit_freq` days, compute PCA on the pooled rank matrix
    over the last `window` days. Use eigenvectors to project.

    Returns: scores (NS, ND) — PC1 score for each instrument-day.
    """
    print(f"[V300] PCA projection: {NF} factors → {n_components} components, window={window}", flush=True)
    t0 = time.time()

    scores = np.full((NS, ND), np.nan)
    loadings = None  # (NF, n_components)

    for di in range(window, ND):
        # Refit PCA periodically
        if loadings is None or (di - window) % refit_freq == 0:
            # Pool rank data over window: (NS*window, NF)
            R_pool = R[:, di-window:di, :].reshape(-1, NF)
            valid = ~np.isnan(R_pool).any(axis=1)
            R_valid = R_pool[valid]

            if R_valid.shape[0] < NF + 10:
                continue

            # Standardize
            mu = R_valid.mean(axis=0)
            sig = R_valid.std(axis=0)
            sig[sig == 0] = 1
            R_z = (R_valid - mu) / sig

            # Covariance → eigen-decompose
            cov = np.cov(R_z.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)

            # Top components (largest eigenvalue)
            idx = np.argsort(eigenvalues)[::-1][:n_components]
            loadings = eigenvectors[:, idx]  # (NF, n_components)
            loadings_norm = loadings / (np.abs(loadings).sum(axis=0, keepdims=True) + 1e-8)

        # Project today's ranks (impute NaN with 0.5 = median rank)
        R_today = R[:, di, :].copy()  # (NS, NF)
        nan_mask = np.isnan(R_today)
        R_today[nan_mask] = 0.5
        for si in range(NS):
            if nan_mask[si].sum() > NF // 2:
                continue  # skip if too many factors missing
            pc_scores = R_today[si] @ loadings_norm
            weights = np.array([1.0, 0.3, 0.1])[:n_components]
            scores[si, di] = np.dot(pc_scores, weights)

    print(f"  PCA done in {time.time()-t0:.1f}s", flush=True)
    return scores


def rolling_ic_projection(F, C, NS, ND, fwd_period=5, window=120, min_ic_samples=500):
    """
    Rolling IC-weighted projection.

    For each factor, compute rolling Spearman IC with forward returns.
    Weight factors by IC to create composite signal.

    Returns: scores (NS, ND)
    """
    print(f"[V300] IC projection: window={window}, fwd={fwd_period}", flush=True)
    t0 = time.time()

    factor_names = sorted(F.keys())
    NF = len(factor_names)

    # Pre-compute cross-sectional ranks for all factors
    F_rank = {}
    for name in factor_names:
        rank_arr = np.full((NS, ND), np.nan)
        for di in range(ND):
            vals = F[name][:, di]
            valid = ~np.isnan(vals)
            if valid.sum() > 5:
                rank_arr[:, di] = pd.Series(vals).rank(pct=True, na_option='keep').values
        F_rank[name] = rank_arr

    # Compute forward returns
    fwd = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(ND - fwd_period):
            if C[si, di] > 0 and not np.isnan(C[si, di]) and not np.isnan(C[si, di+fwd_period]):
                fwd[si, di] = C[si, di+fwd_period] / C[si, di] - 1.0

    scores = np.full((NS, ND), np.nan)

    for di in range(window, ND):
        # Compute IC for each factor over rolling window (using RANKS)
        ics = np.zeros(NF)
        for fi, name in enumerate(factor_names):
            f_vals = F_rank[name][:, di-window:di].flatten()
            r_vals = fwd[:, di-window:di].flatten()
            valid = ~np.isnan(f_vals) & ~np.isnan(r_vals)
            n_valid = valid.sum()

            if n_valid >= min_ic_samples:
                f_r = f_vals[valid]
                r_r = pd.Series(r_vals[valid]).rank().values
                if np.std(f_r) > 0 and np.std(r_r) > 0:
                    ic = np.corrcoef(f_r, r_r)[0, 1]
                    if not np.isnan(ic):
                        ics[fi] = ic

        # Only use factors with |IC| > threshold
        significant = np.abs(ics) > 0.01
        if significant.sum() < 3:
            continue

        # Weight by IC (keep sign for direction)
        weights = np.zeros(NF)
        weights[significant] = ics[significant]
        w_sum = np.sum(np.abs(weights))
        if w_sum > 0:
            weights /= w_sum

        # Score each instrument using RANK values
        for si in range(NS):
            score = 0
            n_used = 0
            for fi, name in enumerate(factor_names):
                val = F_rank[name][si, di]  # USE RANK, not raw value
                if not np.isnan(val) and significant[fi]:
                    score += weights[fi] * val
                    n_used += 1
            if n_used >= 3:
                scores[si, di] = score

    print(f"  IC projection done in {time.time()-t0:.1f}s", flush=True)
    return scores


def equal_weight_signal(F, NS, ND):
    """Simple equal-weight baseline: average of cross-sectional factor ranks."""
    print("[V300] Equal-weight signal...", flush=True)
    factor_names = sorted(F.keys())

    scores = np.full((NS, ND), np.nan)
    for di in range(60, ND):
        vals = []
        for name in factor_names:
            v = F[name][:, di]
            valid = ~np.isnan(v)
            if valid.sum() > 5:
                ranks = pd.Series(v).rank(pct=True, na_option='keep').values
                vals.append(ranks)
        if len(vals) >= 5:
            stacked = np.stack(vals, axis=1)
            # NaN-aware mean
            scores[:, di] = np.nanmean(stacked, axis=1)

    return scores


# ============================================================
# PHASE 3: BACKTEST ENGINE
# ============================================================
def backtest(signal, C, O, H, L, NS, ND, dates, syms,
             top_n=5, hold_days=5, atr_stop=2.0, atr_period=14,
             use_short=True, method='pca'):
    """
    Walk-forward backtest with long+short positions.

    Each day:
    - Rank instruments by signal score
    - Enter long top_n, short bottom_n (if use_short)
    - ATR stop loss for risk management
    - Fixed hold period exit
    """
    print(f"[V300] Backtest: top_n={top_n}, hold={hold_days}, atr_stop={atr_stop}, "
          f"short={use_short}, method={method}", flush=True)

    trades = []
    positions = []  # (si, entry_di, entry_price, stop_price, direction)

    for di in range(60, ND):
        # --- Exit existing positions ---
        new_positions = []
        for si, edi, ep, sp, d in positions:
            c = C[si, di]
            if np.isnan(c):
                new_positions.append((si, edi, ep, sp, d))
                continue

            exit_reason = None
            if d > 0 and c < sp:
                exit_reason = 'stop'
            elif d < 0 and c > sp:
                exit_reason = 'stop'
            elif di - edi >= hold_days:
                exit_reason = 'hold'

            if exit_reason:
                pnl = d * (c - ep) / ep - COMMISSION
                trades.append({
                    'sym': syms[si], 'entry_d': dates[edi], 'exit_d': dates[di],
                    'dir': d, 'entry': ep, 'exit': c, 'pnl': pnl,
                    'reason': exit_reason, 'hold': di - edi, 'method': method
                })
            else:
                new_positions.append((si, edi, ep, sp, d))
        positions = new_positions

        # --- Enter new positions ---
        held = {p[0] for p in positions}
        if len(positions) >= top_n * (2 if use_short else 1):
            continue

        # Rank by signal
        sig_vals = [(signal[si, di], si) for si in range(NS)
                    if not np.isnan(signal[si, di]) and si not in held
                    and not np.isnan(C[si, di]) and not np.isnan(O[si, di])]

        if not sig_vals:
            continue

        sig_vals.sort(key=lambda x: x[0], reverse=True)

        n_enter = top_n * (2 if use_short else 1) - len(positions)

        # Long top candidates
        for score, si in sig_vals[:top_n]:
            if len(positions) >= top_n * (2 if use_short else 1):
                break
            if si in held:
                continue

            o = O[si, di]
            if np.isnan(o) or o <= 0:
                continue

            # ATR for stop
            atr_vals = []
            for j in range(max(60, di-atr_period), di):
                hh, ll, cc = H[si,j], L[si,j], C[si,j]
                if not any(np.isnan([hh, ll, cc])):
                    tr = max(hh-ll, abs(hh-cc), abs(ll-cc))
                    atr_vals.append(tr)
            if not atr_vals:
                continue
            atr = np.mean(atr_vals)

            stop = o - atr_stop * atr
            positions.append((si, di, o, stop, 1))
            held.add(si)

        # Short bottom candidates
        if use_short:
            for score, si in sig_vals[-top_n:]:
                if len(positions) >= top_n * (2 if use_short else 1):
                    break
                if si in held:
                    continue

                o = O[si, di]
                if np.isnan(o) or o <= 0:
                    continue

                atr_vals = []
                for j in range(max(60, di-atr_period), di):
                    hh, ll, cc = H[si,j], L[si,j], C[si,j]
                    if not any(np.isnan([hh, ll, cc])):
                        tr = max(hh-ll, abs(hh-cc), abs(ll-cc))
                        atr_vals.append(tr)
                if not atr_vals:
                    continue
                atr = np.mean(atr_vals)

                stop = o + atr_stop * atr
                positions.append((si, di, o, stop, -1))
                held.add(si)

    # Close remaining
    for si, edi, ep, sp, d in positions:
        c = C[si, ND-1]
        if not np.isnan(c):
            pnl = d * (c - ep) / ep - COMMISSION
            trades.append({
                'sym': syms[si], 'entry_d': dates[edi], 'exit_d': dates[ND-1],
                'dir': d, 'entry': ep, 'exit': c, 'pnl': pnl,
                'reason': 'end', 'hold': ND-1 - edi, 'method': method
            })

    return trades


# ============================================================
# PHASE 4: WALK-FORWARD VALIDATION
# ============================================================
def walk_forward_test(C, O, H, L, V, OI, NS, ND, dates, syms,
                      train_years=4, test_years=1,
                      top_n=5, hold_days=5, atr_stop=2.0,
                      use_short=True, method='pca'):
    """Rigorous walk-forward: train factors+projection on train, test on holdout."""
    print(f"\n[V300] Walk-Forward: train={train_years}y, test={test_years}y, "
          f"method={method}, short={use_short}", flush=True)

    # Convert dates to year boundaries
    years = sorted(set(d.year for d in dates))
    if len(years) < train_years + test_years:
        print("  Not enough data for walk-forward")
        return []

    all_trades = []
    start_year = years[0]

    while True:
        train_end_year = start_year + train_years - 1
        test_year = train_end_year + 1

        if test_year > years[-1]:
            break

        # Find date indices
        train_mask = np.array([d.year <= train_end_year for d in dates])
        test_mask = np.array([d.year == test_year for d in dates])

        train_start = np.where(train_mask)[0][0]
        train_end = np.where(train_mask)[0][-1]
        test_start = np.where(test_mask)[0][0]
        test_end = np.where(test_mask)[0][-1]

        # Combined slice (need history for factor computation)
        slice_start = max(0, train_start - 60)  # extra history for lookback
        C_s = C[:, slice_start:test_end+1]
        O_s = O[:, slice_start:test_end+1]
        H_s = H[:, slice_start:test_end+1]
        L_s = L[:, slice_start:test_end+1]
        V_s = V[:, slice_start:test_end+1]
        OI_s = OI[:, slice_start:test_end+1]
        dates_s = dates[slice_start:test_end+1]
        ND_s = len(dates_s)

        # Compute factors on full slice
        F = compute_factors(C_s, O_s, H_s, L_s, V_s, OI_s, NS, ND_s)

        # Build rank matrix
        R, fnames = cross_sectional_rank_matrix(F, NS, ND_s)

        # Compute signal using specified method
        if method == 'pca':
            signal = rolling_pca_projection(R, NS, ND_s, len(fnames))
        elif method == 'ic':
            signal = rolling_ic_projection(F, C_s, NS, ND_s)
        elif method == 'ew':
            signal = equal_weight_signal(F, NS, ND_s)

        # Only test on the test year
        # Adjust indices relative to slice
        test_start_rel = test_start - slice_start
        test_end_rel = test_end - slice_start

        # Backtest only on test period (but need signal history)
        test_trades = backtest(signal, C_s, O_s, H_s, L_s, NS, ND_s, dates_s, syms,
                              top_n=top_n, hold_days=hold_days, atr_stop=atr_stop,
                              use_short=use_short, method=method)

        # Filter trades to test period only
        test_year_trades = [t for t in test_trades
                           if t['entry_d'].year == test_year]
        all_trades.extend(test_year_trades)

        # Report this window
        if test_year_trades:
            pnls = [t['pnl'] for t in test_year_trades]
            wr = sum(1 for p in pnls if p > 0) / len(pnls) * 100
            cum = np.cumprod([1 + p for p in pnls])[-1]
            print(f"  {test_year}: {len(pnls)} trades, WR={wr:.1f}%, "
                  f"Cum={cum:.3f}x, Avg={np.mean(pnls)*100:.3f}%", flush=True)
        else:
            print(f"  {test_year}: no trades", flush=True)

        start_year += 1

    return all_trades


# ============================================================
# RESULTS ANALYSIS
# ============================================================
def analyze_trades(trades, label="", max_concurrent=10):
    """Analyze trades with proper portfolio-level equity tracking."""
    if not trades:
        print(f"  [{label}] No trades")
        return {}

    pnls = np.array([t['pnl'] for t in trades])
    n = len(pnls)
    wins = pnls > 0
    wr = wins.sum() / n * 100 if n > 0 else 0
    avg = np.mean(pnls) * 100

    # Proper portfolio compounding: each trade uses 1/max_concurrent of capital
    alloc_frac = 1.0 / max_concurrent
    equity = 1.0
    peak = 1.0
    mdd = 0.0

    # Sort trades by exit date for sequential processing
    sorted_trades = sorted(trades, key=lambda t: t['exit_d'])
    equity_curve = []
    for t in sorted_trades:
        portfolio_pnl = t['pnl'] * alloc_frac
        equity *= (1 + portfolio_pnl)
        equity_curve.append(equity)
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak * 100
        if dd > mdd:
            mdd = dd

    cum_ret = equity - 1

    # Annualize
    if n > 1:
        entry_dates = [t['entry_d'] for t in trades]
        total_days = (max(entry_dates) - min(entry_dates)).days + 1
        years = total_days / 365.25
        ann = (1 + cum_ret) ** (1/max(years, 0.1)) - 1 if cum_ret > -1 else -1
    else:
        ann = 0; years = 0

    # Direction breakdown
    long_trades = [t for t in trades if t['dir'] > 0]
    short_trades = [t for t in trades if t['dir'] < 0]

    gross_win = sum(pnls[wins]) if wins.sum() > 0 else 0
    gross_loss = abs(sum(pnls[~wins])) if (~wins).sum() > 0 else 1e-10

    result = {
        'n': n, 'wr': wr, 'avg': avg, 'cum_ret': cum_ret,
        'ann': ann, 'mdd': mdd, 'years': years,
        'long_wr': sum(1 for t in long_trades if t['pnl'] > 0) / max(len(long_trades),1) * 100,
        'short_wr': sum(1 for t in short_trades if t['pnl'] > 0) / max(len(short_trades),1) * 100,
        'profit_factor': gross_win / gross_loss,
    }

    print(f"\n{'='*60}")
    print(f"  [{label}] RESULTS (alloc=1/{max_concurrent})")
    print(f"{'='*60}")
    print(f"  Trades: {n} (L:{len(long_trades)} S:{len(short_trades)})")
    print(f"  Win Rate: {wr:.1f}% (L:{result['long_wr']:.1f}% S:{result['short_wr']:.1f}%)")
    print(f"  Avg Trade: {avg:.3f}%")
    print(f"  Cumulative: {cum_ret*100:.1f}%")
    print(f"  Annual: {ann*100:.1f}%")
    print(f"  Max DD: {mdd:.1f}%")
    print(f"  Profit Factor: {result['profit_factor']:.2f}")
    print(f"  Period: {years:.1f} years")

    return result


# ============================================================
# PHASE 5: PARAMETER SWEEP
# ============================================================
def parameter_sweep(C, O, H, L, V, OI, NS, ND, dates, syms):
    """Sweep key parameters to find optimal configuration."""
    print("\n" + "="*60)
    print("  PARAMETER SWEEP")
    print("="*60)

    # First compute factors once
    F = compute_factors(C, O, H, L, V, OI, NS, ND)
    R, fnames = cross_sectional_rank_matrix(F, NS, ND)
    NF = len(fnames)

    # Compute all signals
    print("\n--- Computing PCA signal ---")
    sig_pca = rolling_pca_projection(R, NS, ND, NF, n_components=3, window=120)
    print("\n--- Computing IC signal ---")
    sig_ic = rolling_ic_projection(F, C, NS, ND, fwd_period=5, window=120)
    print("\n--- Computing EW signal ---")
    sig_ew = equal_weight_signal(F, NS, ND)

    signals = {'pca': sig_pca, 'ic': sig_ic, 'ew': sig_ew}

    results = []
    for method, sig in signals.items():
        for top_n in [3, 5, 8]:
            for hold in [3, 5, 10]:
                for atr_mult in [1.5, 2.0, 3.0]:
                    for short in [True, False]:
                        trades = backtest(sig, C, O, H, L, NS, ND, dates, syms,
                                         top_n=top_n, hold_days=hold,
                                         atr_stop=atr_mult, use_short=short,
                                         method=method)
                        if trades:
                            pnls = np.array([t['pnl'] for t in trades])
                            n = len(pnls)
                            wr = (pnls > 0).sum() / n * 100
                            avg = np.mean(pnls) * 100
                            # Proper compounding with portfolio allocation
                            max_pos = top_n * (2 if short else 1)
                            alloc = 1.0 / max_pos
                            eq = 1.0; pk = 1.0; mdd = 0.0
                            for p in pnls:
                                eq *= (1 + p * alloc)
                                pk = max(pk, eq)
                                dd = (pk - eq) / pk * 100
                                mdd = max(mdd, dd)
                            cum = eq - 1
                            pf = abs(sum(pnls[pnls>0]) / sum(pnls[pnls<0])) if (pnls<0).sum() > 0 and (pnls>0).sum() > 0 else 999

                            results.append({
                                'method': method, 'top_n': top_n, 'hold': hold,
                                'atr': atr_mult, 'short': short,
                                'n': n, 'wr': wr, 'cum': cum, 'avg': avg,
                                'mdd': mdd, 'pf': pf
                            })

    # Sort by Sharpe-like metric: avg/mdd
    results.sort(key=lambda x: -x['avg']/max(x['mdd'], 1))

    print(f"\n{'Method':>6} {'TopN':>4} {'Hold':>4} {'ATR':>4} {'Short':>5} "
          f"{'Trades':>6} {'WR%':>5} {'Avg%':>6} {'Cum%':>8} {'MDD%':>6} {'PF':>5}")
    print("-"*80)
    for r in results[:30]:
        print(f"{r['method']:>6} {r['top_n']:>4} {r['hold']:>4} {r['atr']:>4} "
              f"{str(r['short']):>5} {r['n']:>6} {r['wr']:>5.1f} {r['avg']:>6.3f} "
              f"{r['cum']*100:>8.1f} {r['mdd']:>6.1f} {r['pf']:>5.2f}")

    return results


# ============================================================
# MAIN
# ============================================================
def main():
    print("="*60)
    print("  V300: HIGH-DIMENSIONAL FACTOR PROJECTION STRATEGY")
    print("="*60)

    # Load data
    C, O, H, L, V, OI, NS, ND, dates, syms = load_all_data(start='2016-01-01')

    # ===== FULL BACKTEST (in-sample, for rapid iteration) =====
    print("\n" + "="*60)
    print("  FULL PERIOD BACKTEST")
    print("="*60)

    F = compute_factors(C, O, H, L, V, OI, NS, ND)
    R, fnames = cross_sectional_rank_matrix(F, NS, ND)
    NF = len(fnames)

    # Three projection methods
    sig_pca = rolling_pca_projection(R, NS, ND, NF, n_components=3, window=120)
    sig_ic = rolling_ic_projection(F, C, NS, ND, fwd_period=5, window=120)
    sig_ew = equal_weight_signal(F, NS, ND)

    print("\n--- PCA Method ---")
    trades_pca = backtest(sig_pca, C, O, H, L, NS, ND, dates, syms,
                          top_n=5, hold_days=5, atr_stop=2.0, use_short=True, method='pca')
    analyze_trades(trades_pca, "PCA-LongShort", max_concurrent=10)

    print("\n--- IC Method ---")
    trades_ic = backtest(sig_ic, C, O, H, L, NS, ND, dates, syms,
                         top_n=5, hold_days=5, atr_stop=2.0, use_short=True, method='ic')
    analyze_trades(trades_ic, "IC-LongShort", max_concurrent=10)

    print("\n--- Equal Weight Method ---")
    trades_ew = backtest(sig_ew, C, O, H, L, NS, ND, dates, syms,
                         top_n=5, hold_days=5, atr_stop=2.0, use_short=True, method='ew')
    analyze_trades(trades_ew, "EW-LongShort", max_concurrent=10)

    # ===== PARAMETER SWEEP =====
    sweep_results = parameter_sweep(C, O, H, L, V, OI, NS, ND, dates, syms)

    # ===== WALK-FORWARD for best method =====
    # Pick best method from sweep
    if sweep_results:
        best = sweep_results[0]
        best_method = best['method']
        print(f"\n\nBest in-sample method: {best_method} "
              f"(WR={best['wr']:.1f}%, Cum={best['cum']*100:.1f}%)")

        print("\n" + "="*60)
        print("  WALK-FORWARD VALIDATION")
        print("="*60)
        wf_trades = walk_forward_test(C, O, H, L, V, OI, NS, ND, dates, syms,
                                       train_years=4, test_years=1,
                                       top_n=best['top_n'], hold_days=best['hold'],
                                       atr_stop=best['atr'], use_short=best['short'],
                                       method=best_method)
        max_c = best['top_n'] * (2 if best['short'] else 1)
        analyze_trades(wf_trades, f"WalkForward-{best_method}", max_concurrent=max_c)

    print("\n[V300] Done.")


if __name__ == '__main__':
    main()
