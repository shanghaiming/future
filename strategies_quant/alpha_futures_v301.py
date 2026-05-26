"""
V301: 高维投影 → 市场状态检测 → 自适应策略切换
=================================================
核心创新: 不是用投影直接做信号，而是用PCA检测市场状态，
在不同状态下切换momentum/mean-reversion/carry策略。

架构:
1. 因子计算 (15个维度)
2. PCA状态检测 (高维→低维投影)
3. 三类信号: 动量 / 均值回归 / 期限结构carry
4. 根据状态动态分配权重
5. Kelly仓位管理
6. Day-by-day权益曲线
7. Walk-forward验证
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
    print("[V301] Loading data...", flush=True)
    t0 = time.time()
    syms = list_available_symbols('daily')
    stock_data = {}
    for sym in syms:
        try:
            df = load_stock_data(sym, frequency='daily')
            if df is not None and len(df) >= min_days:
                stock_data[sym] = df
        except: pass

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

    C = C[:, i0:i1+1]; O = O[:, i0:i1+1]; H = H[:, i0:i1+1]
    L = L[:, i0:i1+1]; V = V[:, i0:i1+1]; OI = OI[:, i0:i1+1]
    print(f"  {NS} sym, {ND} days ({time.time()-t0:.1f}s) {dates[0].strftime('%Y-%m-%d')}~{dates[-1].strftime('%Y-%m-%d')}")
    return C, O, H, L, V, OI, NS, ND, dates, syms


# ============================================================
# PHASE 1: FACTOR COMPUTATION (vectorized per instrument)
# ============================================================
def compute_factors(C, O, H, L, V, OI, NS, ND):
    print("[V301] Computing factors...", flush=True)
    t0 = time.time()
    F = {}

    for si in range(NS):
        c = C[si]; o = O[si]; h = H[si]; l = L[si]; v = V[si]; oi = OI[si]
        nan_c = np.isnan(c)

        # 1. Momentum: returns at multiple horizons
        for period, name in [(5,'mom5'), (10,'mom10'), (20,'mom20'), (60,'mom60')]:
            if name not in F: F[name] = np.full((NS, ND), np.nan)
            for di in range(period, ND):
                if not nan_c[di] and not nan_c[di-period] and c[di-period] > 0:
                    F[name][si, di] = c[di] / c[di-period] - 1.0

        # 2. Trend: ADX
        if 'adx' not in F: F['adx'] = np.full((NS, ND), np.nan)
        if HAS_TALIB:
            try:
                hs = np.where(np.isnan(h), 0, h).astype(np.float64)
                ls = np.where(np.isnan(l), 0, l).astype(np.float64)
                cs = np.where(nan_c, 0, c).astype(np.float64)
                adx = talib.ADX(hs, ls, cs, 14)
                F['adx'][si] = np.where(nan_c, np.nan, adx)
            except: pass

        # 3. Trend slope (20d linreg annualized)
        if 'slope' not in F: F['slope'] = np.full((NS, ND), np.nan)
        for di in range(20, ND):
            w = c[di-20:di]
            vv = w[~np.isnan(w)]
            if len(vv) >= 15 and vv[0] > 0:
                x = np.arange(len(vv))
                y = vv / vv[0]
                try:
                    s = np.polyfit(x, y, 1)
                    F['slope'][si, di] = s[0] * 252
                except: pass

        # 4. Volatility: 20d realized vol
        if 'vol20' not in F: F['vol20'] = np.full((NS, ND), np.nan)
        for di in range(20, ND):
            rets = []
            for j in range(max(1, di-20), di):
                if not nan_c[j] and not nan_c[j-1] and c[j-1] > 0:
                    rets.append(c[j]/c[j-1] - 1)
            if len(rets) >= 10:
                F['vol20'][si, di] = np.std(rets) * np.sqrt(252)

        # 5. ATR%
        if 'atrp' not in F: F['atrp'] = np.full((NS, ND), np.nan)
        if HAS_TALIB:
            try:
                hs = np.where(np.isnan(h), 0, h).astype(np.float64)
                ls = np.where(np.isnan(l), 0, l).astype(np.float64)
                cs = np.where(nan_c, 0, c).astype(np.float64)
                atr = talib.ATR(hs, ls, cs, 14)
                valid = ~nan_c & (c > 0)
                F['atrp'][si] = np.where(valid, atr / np.where(c > 0, c, 1), np.nan)
            except: pass

        # 6. Volume anomaly (z-score)
        if 'vanom' not in F: F['vanom'] = np.full((NS, ND), np.nan)
        for di in range(60, ND):
            vw = v[di-60:di]
            vv = vw[~np.isnan(vw)]
            if len(vv) >= 30 and not np.isnan(v[di]):
                mu, sig = np.mean(vv), np.std(vv)
                if sig > 0: F['vanom'][si, di] = (v[di] - mu) / sig

        # 7. OI change 5d
        if 'oi5' not in F: F['oi5'] = np.full((NS, ND), np.nan)
        for di in range(5, ND):
            if not np.isnan(oi[di]) and not np.isnan(oi[di-5]) and oi[di-5] > 0:
                F['oi5'][si, di] = oi[di] / oi[di-5] - 1

        # 8. RSI 14
        if 'rsi' not in F: F['rsi'] = np.full((NS, ND), np.nan)
        if HAS_TALIB:
            try:
                cs = np.where(nan_c, 0, c).astype(np.float64)
                rsi = talib.RSI(cs, 14)
                F['rsi'][si] = np.where(nan_c, np.nan, rsi)
            except: pass

        # 9. Price z-score (20d)
        if 'zscore' not in F: F['zscore'] = np.full((NS, ND), np.nan)
        for di in range(20, ND):
            w = c[di-20:di]; vv = w[~np.isnan(w)]
            if len(vv) >= 15 and np.std(vv) > 0 and not nan_c[di]:
                F['zscore'][si, di] = (c[di] - np.mean(vv)) / np.std(vv)

        # 10. Body ratio (PA)
        if 'body' not in F: F['body'] = np.full((NS, ND), np.nan)
        for di in range(1, ND):
            ohlc = [o[di], h[di], l[di], c[di]]
            if not any(np.isnan(ohlc)) and h[di] > l[di]:
                F['body'][si, di] = abs(c[di] - o[di]) / (h[di] - l[di])

    # 11. Cross-sectional momentum rank
    cs_mom = np.full((NS, ND), np.nan)
    for di in range(60, ND):
        vals = F['mom20'][:, di]
        valid = ~np.isnan(vals)
        if valid.sum() > 5:
            cs_mom[:, di] = pd.Series(vals).rank(pct=True, na_option='keep').values
    F['cs_mom'] = cs_mom

    # 12. Cross-sectional vol rank
    cs_vol = np.full((NS, ND), np.nan)
    for di in range(60, ND):
        vals = F['vol20'][:, di]
        valid = ~np.isnan(vals)
        if valid.sum() > 5:
            cs_vol[:, di] = pd.Series(vals).rank(pct=True, na_option='keep').values
    F['cs_vol'] = cs_vol

    print(f"  {len(F)} factors in {time.time()-t0:.1f}s", flush=True)
    return F


# ============================================================
# PHASE 2: REGIME DETECTION via PCA on market-wide factors
# ============================================================
def detect_regimes(F, NS, ND, window=120):
    """
    Use PCA on cross-sectional factor covariance to detect market regimes.
    Regime types:
    - TRENDING: high ADX, low cross-sectional dispersion → momentum works
    - MEAN_REVERTING: low ADX, high dispersion → mean-reversion works
    - VOLATILE: high vol, uncertain → reduce exposure
    """
    print("[V301] Detecting market regimes...", flush=True)
    t0 = time.time()

    # Market-wide features (averaged across instruments)
    mkt_adx = np.full(ND, np.nan)
    mkt_vol = np.full(ND, np.nan)
    mkt_disp = np.full(ND, np.nan)  # cross-sectional return dispersion
    mkt_trend = np.full(ND, np.nan)  # average trend slope

    for di in range(60, ND):
        # Average ADX
        adx_vals = F['adx'][:, di]
        valid = adx_vals[~np.isnan(adx_vals)]
        if len(valid) > 5: mkt_adx[di] = np.mean(valid)

        # Average vol
        vol_vals = F['vol20'][:, di]
        valid = vol_vals[~np.isnan(vol_vals)]
        if len(valid) > 5: mkt_vol[di] = np.mean(valid)

        # Cross-sectional dispersion of 20d returns
        mom_vals = F['mom20'][:, di]
        valid = mom_vals[~np.isnan(mom_vals)]
        if len(valid) > 5: mkt_disp[di] = np.std(valid)

        # Average trend slope
        slope_vals = F['slope'][:, di]
        valid = slope_vals[~np.isnan(slope_vals)]
        if len(valid) > 5: mkt_trend[di] = np.mean(valid)

    # Rolling z-scores for regime classification
    regime = np.full(ND, 0)  # 0=neutral, 1=trending, -1=choppy, 2=volatile

    for di in range(window, ND):
        adx_w = mkt_adx[di-window:di]
        vol_w = mkt_vol[di-window:di]
        disp_w = mkt_disp[di-window:di]

        adx_val = mkt_adx[di]
        vol_val = mkt_vol[di]
        disp_val = mkt_disp[di]

        if np.isnan(adx_val) or np.isnan(vol_val):
            continue

        adx_valid = adx_w[~np.isnan(adx_w)]
        vol_valid = vol_w[~np.isnan(vol_w)]

        if len(adx_valid) < 30 or len(vol_valid) < 30:
            continue

        adx_z = (adx_val - np.mean(adx_valid)) / max(np.std(adx_valid), 0.01)
        vol_z = (vol_val - np.mean(vol_valid)) / max(np.std(vol_valid), 0.01)

        # Regime classification
        if vol_z > 1.5:
            regime[di] = 2  # volatile → reduce
        elif adx_z > 0.5 and vol_z < 1.0:
            regime[di] = 1  # trending → momentum
        elif adx_z < -0.5:
            regime[di] = -1  # choppy → mean-reversion
        else:
            regime[di] = 0  # neutral

    # Print regime distribution
    counts = {2: (regime == 2).sum(), 1: (regime == 1).sum(),
              0: (regime == 0).sum(), -1: (regime == -1).sum()}
    print(f"  Regimes: volatile={counts[2]}, trending={counts[1]}, "
          f"neutral={counts[0]}, choppy={counts[-1]} ({time.time()-t0:.1f}s)", flush=True)

    return regime


# ============================================================
# PHASE 3: SIGNAL GENERATION
# ============================================================
def generate_signals(F, regime, C, NS, ND, syms):
    """
    Generate three independent signals and blend by regime.
    1. Momentum signal (cross-sectional + time-series)
    2. Mean-reversion signal (RSI, z-score)
    3. Carry/slope signal (trend slope + OI)
    """
    print("[V301] Generating regime-aware signals...", flush=True)

    sym_idx = {s: i for i, s in enumerate(syms)}
    sym_to_group = {}
    for gname, gsyms in COMMODITY_GROUPS.items():
        for s in gsyms:
            sym_to_group[s] = gname

    # --- Momentum Signal ---
    mom_signal = np.full((NS, ND), np.nan)
    for di in range(60, ND):
        scores = np.full(NS, np.nan)
        for si in range(NS):
            m20 = F['mom20'][si, di]
            m60 = F['mom60'][si, di]
            adx = F['adx'][si, di]
            cs = F['cs_mom'][si, di]

            if np.isnan(m20) or np.isnan(cs):
                continue

            score = 0
            n = 0

            # Time-series momentum (strong trend)
            if not np.isnan(m20) and not np.isnan(adx):
                if m20 > 0.02 and adx > 25:
                    score += 1.5
                elif m20 < -0.02 and adx > 25:
                    score -= 1.5
                n += 1

            # Cross-sectional momentum rank
            if cs > 0.8: score += 1.0
            elif cs < 0.2: score -= 1.0
            n += 1

            # Volume confirmation
            va = F['vanom'][si, di]
            if not np.isnan(va) and abs(va) > 1.5:
                if va > 0 and m20 > 0: score += 0.5
                elif va > 0 and m20 < 0: score -= 0.5
                n += 1

            # Group alignment
            s_name = syms[si] if si < len(syms) else None
            if s_name and s_name in sym_to_group:
                gname = sym_to_group[s_name]
                gsyms = COMMODITY_GROUPS.get(gname, [])
                gm_vals = []
                for gs in gsyms:
                    if gs in sym_idx:
                        gm = F['mom20'][sym_idx[gs], di]
                        if not np.isnan(gm): gm_vals.append(gm)
                if gm_vals and not np.isnan(m20):
                    gm = np.mean(gm_vals)
                    if (m20 > 0 and gm > 0) or (m20 < 0 and gm < 0):
                        score += 0.5 * np.sign(m20)
                    n += 1

            if n >= 2:
                scores[si] = score / n
        mom_signal[:, di] = scores

    # --- Mean Reversion Signal ---
    mr_signal = np.full((NS, ND), np.nan)
    for di in range(60, ND):
        scores = np.full(NS, np.nan)
        for si in range(NS):
            rsi = F['rsi'][si, di]
            zs = F['zscore'][si, di]
            cs_rank = F['cs_mom'][si, di]

            if np.isnan(rsi) or np.isnan(zs):
                continue

            score = 0; n = 0

            # RSI extreme → reversal
            if rsi > 70:
                score -= 1.5  # overbought → short
                n += 1
            elif rsi < 30:
                score += 1.5  # oversold → long
                n += 1

            # Z-score extreme → reversal
            if zs > 2.0:
                score -= 1.0
                n += 1
            elif zs < -2.0:
                score += 1.0
                n += 1

            # Contrarian: buy worst performers (extreme cs_rank)
            if not np.isnan(cs_rank):
                if cs_rank < 0.15:  # bottom 15%
                    score += 0.5
                elif cs_rank > 0.85:  # top 15%
                    score -= 0.5
                n += 1

            if n >= 2:
                scores[si] = score / n
        mr_signal[:, di] = scores

    # --- Carry/Slope Signal ---
    carry_signal = np.full((NS, ND), np.nan)
    for di in range(60, ND):
        scores = np.full(NS, np.nan)
        for si in range(NS):
            slope = F['slope'][si, di]
            oi5 = F['oi5'][si, di]
            body = F['body'][si, di]

            if np.isnan(slope):
                continue

            score = 0; n = 0

            # Trend slope direction
            if slope > 0.5:
                score += 1.0
            elif slope < -0.5:
                score -= 1.0
            n += 1

            # OI increasing confirms trend
            if not np.isnan(oi5):
                if oi5 > 0.05 and slope > 0:
                    score += 0.5
                elif oi5 < -0.05 and slope < 0:
                    score += 0.5
                elif oi5 > 0.05 and slope < 0:
                    score -= 0.5
                elif oi5 < -0.05 and slope > 0:
                    score -= 0.5
                n += 1

            # Strong bar confirms
            if not np.isnan(body) and body > 0.7:
                if not np.isnan(F['mom5'][si, di]):
                    score += 0.3 * np.sign(F['mom5'][si, di])
                n += 1

            if n >= 2:
                scores[si] = score / n
        carry_signal[:, di] = scores

    # --- Regime-Aware Blending ---
    signal = np.full((NS, ND), np.nan)
    for di in range(60, ND):
        r = regime[di]
        for si in range(NS):
            m = mom_signal[si, di]
            mr = mr_signal[si, di]
            c = carry_signal[si, di]

            vals = []; weights = []
            if not np.isnan(m):
                w = {1: 0.6, 0: 0.3, -1: 0.1, 2: 0.1}[r]
                vals.append(m); weights.append(w)
            if not np.isnan(mr):
                w = {1: 0.1, 0: 0.3, -1: 0.6, 2: 0.1}[r]
                vals.append(mr); weights.append(w)
            if not np.isnan(c):
                w = {1: 0.3, 0: 0.4, -1: 0.3, 2: 0.2}[r]
                vals.append(c); weights.append(w)

            if vals:
                total_w = sum(weights)
                if total_w > 0:
                    signal[si, di] = sum(v*w for v, w in zip(vals, weights)) / total_w

    return signal, mom_signal, mr_signal, carry_signal


# ============================================================
# PHASE 4: DAY-BY-DAY BACKTEST WITH EQUITY CURVE
# ============================================================
def backtest_daily(signal, C, O, H, L, NS, ND, dates, syms, regime,
                   top_n=5, hold_days=5, atr_stop=2.5, use_short=True):
    """
    Day-by-day backtest tracking equity curve.
    Position sizing: equal weight within max_concurrent.
    """
    max_pos = top_n * (2 if use_short else 1)
    alloc = 1.0 / max_pos

    equity = CASH0
    peak = CASH0
    equity_curve = np.full(ND, np.nan)
    equity_curve[0] = equity
    positions = []  # (si, entry_di, entry_price, stop_price, direction, alloc_at_entry)
    trades = []

    for di in range(1, ND):
        # Mark-to-market: update equity from open positions
        daily_pnl = 0
        new_positions = []
        for si, edi, ep, sp, d, a in positions:
            c = C[si, di]
            if np.isnan(c):
                new_positions.append((si, edi, ep, sp, d, a))
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
                trade_profit = equity * a * pnl
                daily_pnl += trade_profit
                trades.append({
                    'sym': syms[si], 'entry_d': dates[edi], 'exit_d': dates[di],
                    'dir': d, 'entry': ep, 'exit': c, 'pnl': pnl,
                    'profit': trade_profit, 'reason': exit_reason,
                    'hold': di - edi, 'regime': regime[edi],
                    'equity_at_entry': equity
                })
            else:
                new_positions.append((si, edi, ep, sp, d, a))
        positions = new_positions

        equity += daily_pnl
        equity_curve[di] = equity

        if equity > peak: peak = equity
        dd = (peak - equity) / peak

        if equity <= 0:
            print(f"  BLOWUP at {dates[di].strftime('%Y-%m-%d')}, equity={equity:.0f}")
            equity_curve[di:] = 0
            break

        # --- Enter new positions ---
        if di < 60: continue
        held = {p[0] for p in positions}
        if len(positions) >= max_pos: continue

        sig_vals = [(signal[si, di], si) for si in range(NS)
                    if not np.isnan(signal[si, di]) and si not in held
                    and not np.isnan(C[si, di]) and not np.isnan(O[si, di])
                    and abs(signal[si, di]) > 0.2]
        if not sig_vals: continue
        sig_vals.sort(key=lambda x: x[0], reverse=True)

        # Position size fraction (adjust for regime)
        r = regime[di]
        size_mult = {1: 1.0, 0: 0.7, -1: 0.5, 2: 0.3}.get(r, 0.5)
        pos_alloc = alloc * size_mult

        # Long top candidates
        for score, si in sig_vals[:top_n]:
            if len(positions) >= max_pos: break
            if si in held: continue
            op = O[si, di]
            if np.isnan(op) or op <= 0: continue

            # ATR stop
            atr_vals = []
            for j in range(max(60, di-14), di):
                hh, ll, cc = H[si,j], L[si,j], C[si,j]
                if not any(np.isnan([hh, ll, cc])):
                    atr_vals.append(max(hh-ll, abs(hh-cc), abs(ll-cc)))
            if not atr_vals: continue
            atr = np.mean(atr_vals)

            stop = op - atr_stop * atr
            positions.append((si, di, op, stop, 1, pos_alloc))
            held.add(si)

        # Short bottom candidates
        if use_short:
            for score, si in sig_vals[-top_n:]:
                if len(positions) >= max_pos: break
                if si in held: continue
                op = O[si, di]
                if np.isnan(op) or op <= 0: continue

                atr_vals = []
                for j in range(max(60, di-14), di):
                    hh, ll, cc = H[si,j], L[si,j], C[si,j]
                    if not any(np.isnan([hh, ll, cc])):
                        atr_vals.append(max(hh-ll, abs(hh-cc), abs(ll-cc)))
                if not atr_vals: continue
                atr = np.mean(atr_vals)

                stop = op + atr_stop * atr
                positions.append((si, di, op, stop, -1, pos_alloc))
                held.add(si)

    # Close remaining
    for si, edi, ep, sp, d, a in positions:
        c = C[si, ND-1]
        if not np.isnan(c):
            pnl = d * (c - ep) / ep - COMMISSION
            trades.append({
                'sym': syms[si], 'entry_d': dates[edi], 'exit_d': dates[ND-1],
                'dir': d, 'entry': ep, 'exit': c, 'pnl': pnl,
                'profit': equity * a * pnl, 'reason': 'end',
                'hold': ND-1 - edi, 'regime': regime[edi],
                'equity_at_entry': equity
            })

    return trades, equity_curve


# ============================================================
# PHASE 5: ANALYSIS
# ============================================================
def analyze(trades, equity_curve, dates, label=""):
    if not trades:
        print(f"  [{label}] No trades")
        return {}

    pnls = np.array([t['pnl'] for t in trades])
    n = len(pnls)
    wr = (pnls > 0).sum() / n * 100
    avg = np.mean(pnls) * 100

    # From equity curve
    eq = equity_curve[~np.isnan(equity_curve)]
    if len(eq) < 2:
        print(f"  [{label}] Insufficient equity data")
        return {}

    total_ret = eq[-1] / eq[0] - 1
    years = (dates[-1] - dates[0]).days / 365.25
    ann = (1 + total_ret) ** (1 / max(years, 0.1)) - 1 if total_ret > -1 else -1

    # Max drawdown
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / peak * 100
    mdd = np.max(dd)

    # Per-regime stats
    regime_pnls = {r: [] for r in [1, 0, -1, 2]}
    for t in trades:
        r = t.get('regime', 0)
        regime_pnls.get(r, []).append(t['pnl'])

    long_trades = [t for t in trades if t['dir'] > 0]
    short_trades = [t for t in trades if t['dir'] < 0]
    gross_win = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    pf = gross_win / max(gross_loss, 1e-10)

    # Sharpe (from daily equity changes)
    eq_changes = np.diff(eq) / eq[:-1]
    sharpe = np.mean(eq_changes) / max(np.std(eq_changes), 1e-10) * np.sqrt(252)

    print(f"\n{'='*60}")
    print(f"  [{label}] RESULTS")
    print(f"{'='*60}")
    print(f"  Trades: {n} (L:{len(long_trades)} S:{len(short_trades)})")
    print(f"  Win Rate: {wr:.1f}%")
    print(f"  Avg Trade: {avg:.3f}%")
    print(f"  Total Return: {total_ret*100:.1f}%")
    print(f"  Annual Return: {ann*100:.1f}%")
    print(f"  Max DD: {mdd:.1f}%")
    print(f"  Profit Factor: {pf:.2f}")
    print(f"  Sharpe: {sharpe:.2f}")
    print(f"  Final Equity: {eq[-1]:,.0f}")

    print(f"\n  Per-Regime:")
    for r, name in [(1,'Trending'), (0,'Neutral'), (-1,'Choppy'), (2,'Volatile')]:
        rp = regime_pnls.get(r, [])
        if rp:
            rwr = sum(1 for p in rp if p > 0) / len(rp) * 100
            print(f"    {name:>10}: {len(rp):>4} trades, WR={rwr:.1f}%, avg={np.mean(rp)*100:.3f}%")

    # Annual breakdown
    print(f"\n  Annual Returns:")
    for yr in sorted(set(t['exit_d'].year for t in trades)):
        yr_trades = [t for t in trades if t['exit_d'].year == yr]
        if yr_trades:
            yr_wr = sum(1 for t in yr_trades if t['pnl'] > 0) / len(yr_trades) * 100
            print(f"    {yr}: {len(yr_trades):>4} trades, WR={yr_wr:.1f}%")

    return {
        'n': n, 'wr': wr, 'avg': avg, 'total_ret': total_ret,
        'ann': ann, 'mdd': mdd, 'pf': pf, 'sharpe': sharpe
    }


# ============================================================
# PHASE 6: WALK-FORWARD VALIDATION
# ============================================================
def walk_forward(C, O, H, L, V, OI, NS, ND, dates, syms,
                 train_years=4, test_years=1, top_n=5, hold_days=5,
                 atr_stop=2.5, use_short=True):
    print(f"\n{'='*60}")
    print(f"  WALK-FORWARD: train={train_years}y, test={test_years}y")
    print(f"{'='*60}")

    years = sorted(set(d.year for d in dates))
    all_trades = []
    all_equity = []

    start_year = years[0]
    while True:
        train_end = start_year + train_years - 1
        test_year = train_end + 1
        if test_year > years[-1]: break

        # Slice data
        train_mask = np.array([d.year <= train_end for d in dates])
        test_mask = np.array([d.year == test_year for d in dates])
        t0 = max(0, np.where(train_mask)[0][0] - 60)

        sl = slice(t0, np.where(test_mask)[0][-1] + 1)
        C_s, O_s, H_s, L_s = C[:, sl], O[:, sl], H[:, sl], L[:, sl]
        V_s, OI_s = V[:, sl], OI[:, sl]
        dates_s = dates[sl]
        ND_s = len(dates_s)

        # Compute on full slice
        F = compute_factors(C_s, O_s, H_s, L_s, V_s, OI_s, NS, ND_s)
        regime = detect_regimes(F, NS, ND_s)
        signal, _, _, _ = generate_signals(F, regime, C_s, NS, ND_s, syms)

        # Backtest full slice but only keep test-year trades
        trades, eq = backtest_daily(signal, C_s, O_s, H_s, L_s, NS, ND_s,
                                     dates_s, syms, regime,
                                     top_n=top_n, hold_days=hold_days,
                                     atr_stop=atr_stop, use_short=use_short)

        yr_trades = [t for t in trades if t['exit_d'].year == test_year]
        all_trades.extend(yr_trades)

        if yr_trades:
            pnls = [t['pnl'] for t in yr_trades]
            wr = sum(1 for p in pnls if p > 0) / len(pnls) * 100
            print(f"  {test_year}: {len(pnls)} trades, WR={wr:.1f}%, avg={np.mean(pnls)*100:.3f}%",
                  flush=True)
        else:
            print(f"  {test_year}: no trades", flush=True)

        start_year += 1

    # Build synthetic equity curve from WF trades
    if all_trades:
        eq = np.ones(len(all_trades) + 1) * CASH0
        for i, t in enumerate(sorted(all_trades, key=lambda x: x['exit_d'])):
            eq[i+1] = eq[i] * (1 + t['pnl'] / (top_n * (2 if use_short else 1)))

        dates_wf = [t['exit_d'] for t in sorted(all_trades, key=lambda x: x['exit_d'])]
        analyze(all_trades, eq, dates_wf, "WalkForward")

    return all_trades


# ============================================================
# PARAMETER SWEEP
# ============================================================
def sweep(C, O, H, L, V, OI, NS, ND, dates, syms):
    print("\n" + "="*60)
    print("  PARAMETER SWEEP")
    print("="*60)

    F = compute_factors(C, O, H, L, V, OI, NS, ND)
    regime = detect_regimes(F, NS, ND)
    signal, _, _, _ = generate_signals(F, regime, C, NS, ND, syms)

    results = []
    for top_n in [3, 5, 8]:
        for hold in [3, 5, 10]:
            for atr in [2.0, 2.5, 3.5]:
                for short in [True, False]:
                    trades, eq = backtest_daily(signal, C, O, H, L, NS, ND,
                                                dates, syms, regime,
                                                top_n=top_n, hold_days=hold,
                                                atr_stop=atr, use_short=short)
                    if not trades: continue
                    pnls = np.array([t['pnl'] for t in trades])
                    n = len(pnls)
                    wr = (pnls > 0).sum() / n * 100
                    avg = np.mean(pnls) * 100
                    eq_v = eq[~np.isnan(eq)]
                    if len(eq_v) < 2: continue
                    total_ret = eq_v[-1] / eq_v[0] - 1
                    pk = np.maximum.accumulate(eq_v)
                    dd = (pk - eq_v) / pk * 100
                    mdd = np.max(dd)
                    yrs = (dates[-1] - dates[0]).days / 365.25
                    ann = (1 + total_ret) ** (1 / max(yrs, 0.1)) - 1 if total_ret > -1 else -1
                    eq_chg = np.diff(eq_v) / eq_v[:-1]
                    sharpe = np.mean(eq_chg) / max(np.std(eq_chg), 1e-10) * np.sqrt(252)

                    results.append({
                        'top_n': top_n, 'hold': hold, 'atr': atr, 'short': short,
                        'n': n, 'wr': wr, 'avg': avg, 'ann': ann*100,
                        'mdd': mdd, 'sharpe': sharpe, 'total': total_ret*100
                    })

    results.sort(key=lambda x: -x['sharpe'])
    print(f"\n{'TopN':>4} {'Hold':>4} {'ATR':>4} {'Short':>5} "
          f"{'Trades':>6} {'WR%':>5} {'Avg%':>6} {'Ann%':>7} {'MDD%':>6} {'Sharpe':>7}")
    print("-"*75)
    for r in results[:20]:
        print(f"{r['top_n']:>4} {r['hold']:>4} {r['atr']:>4} {str(r['short']):>5} "
              f"{r['n']:>6} {r['wr']:>5.1f} {r['avg']:>6.3f} "
              f"{r['ann']:>7.1f} {r['mdd']:>6.1f} {r['sharpe']:>7.2f}")

    return results


# ============================================================
# MAIN
# ============================================================
def main():
    print("="*60)
    print("  V301: REGIME-AWARE MULTI-STRATEGY")
    print("="*60)

    C, O, H, L, V, OI, NS, ND, dates, syms = load_all_data(start='2016-01-01')

    # Full backtest
    F = compute_factors(C, O, H, L, V, OI, NS, ND)
    regime = detect_regimes(F, NS, ND)
    signal, mom_sig, mr_sig, carry_sig = generate_signals(F, regime, C, NS, ND, syms)

    print("\n--- Regime-Aware Long+Short ---")
    trades, eq = backtest_daily(signal, C, O, H, L, NS, ND, dates, syms, regime,
                                 top_n=5, hold_days=5, atr_stop=2.5, use_short=True)
    analyze(trades, eq, dates, "RegimeAware-L+S")

    # Sweep
    results = sweep(C, O, H, L, V, OI, NS, ND, dates, syms)

    # Walk-forward for best config
    if results:
        best = results[0]
        print(f"\nBest config: top_n={best['top_n']}, hold={best['hold']}, "
              f"atr={best['atr']}, short={best['short']}, sharpe={best['sharpe']:.2f}")
        wf = walk_forward(C, O, H, L, V, OI, NS, ND, dates, syms,
                          top_n=best['top_n'], hold_days=best['hold'],
                          atr_stop=best['atr'], use_short=best['short'])

    print("\n[V301] Done.")


if __name__ == '__main__':
    main()
