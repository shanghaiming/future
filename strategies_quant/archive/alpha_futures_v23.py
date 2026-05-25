"""
Alpha Futures V23 — Multi-Signal Composite Strategy (无杠杆, 纯日线)
===================================================================
综合340+现有策略研究的精华，构建多因子复合信号系统。

核心思路:
  1. 波动率压缩→扩张识别 (squeeze breakout) — 捕捉大行情启动
  2. Donchian通道突破 + 趋势过滤 — 经典海龟交易法
  3. VDP+OI资金流确认 — 验证突破真实性
  4. KER趋势效率过滤 — 只在趋势明确时交易
  5. 供需区回调入场 — 更好的入场价格
  6. 自适应持仓 — 趋势中持仓更长，震荡中快速退出
  7. 跨品种截面排名 — 选最强品种

数学目标:
  - 持仓5-15天, 捕捉3-8%的价格波动
  - 年交易60-120次, WR>50%, 平均盈亏比>2:1
  - 理论: 80次 × 2.3% = 600%年化

约束: 不做gap, 不做日内, 无杠杆
"""
import sys, os, time, warnings
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

# 品种分组 (产业链关系)
GROUPS = {
    'black':  ['rbfi', 'hcfi', 'ifi', 'jfi', 'jmfi'],
    'metal':  ['cufi', 'alfi', 'znfi', 'aufi', 'agfi', 'nifi', 'snfi', 'pbfi', 'sffi', 'ssfi', 'ni'],
    'energy': ['scfi', 'mafi', 'tafi', 'bfi', 'fufi', 'ebfi', 'pgfi', 'egfi', 'fgfi', 'oifi', 'lufi', 'brfi', 'spfi'],
    'agri':   ['afi', 'mfi', 'yfi', 'cfi', 'srfi', 'pfi', 'rmfi', 'rrfi', 'lrfi', 'whfi', 'rsfi', 'pmfi', 'cjfi', 'apfi', 'csfi', 'cyfi', 'jrfi', 'pkfi', 'rrfi'],
    'chem':   ['ppfi', 'vfi', 'lfi', 'ebfi', 'egfi', 'tafi', 'mafi', 'fgfi', 'safi', 'urfi'],
}


# ============================================================
# SIGNAL COMPUTATION
# ============================================================

def compute_atr(H, L, C, period=14):
    """ATR computation for a single series."""
    n = len(H)
    tr = np.full(n, np.nan)
    for i in range(1, n):
        if np.isnan(H[i]) or np.isnan(L[i]):
            continue
        tr[i] = H[i] - L[i]
        if not np.isnan(C[i-1]):
            tr[i] = max(tr[i], abs(H[i] - C[i-1]), abs(L[i] - C[i-1]))
    atr = np.full(n, np.nan)
    for i in range(period, n):
        window = tr[i-period+1:i+1]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            atr[i] = np.mean(valid)
    return atr


def compute_ema(arr, period):
    """EMA for a single series."""
    n = len(arr)
    ema = np.full(n, np.nan)
    alpha = 2.0 / (period + 1)
    start = None
    for i in range(n):
        if not np.isnan(arr[i]):
            if start is None:
                ema[i] = arr[i]
                start = i
            else:
                ema[i] = alpha * arr[i] + (1 - alpha) * ema[i-1]
    return ema


def compute_sma(arr, period):
    """SMA for a single series."""
    n = len(arr)
    sma = np.full(n, np.nan)
    for i in range(period - 1, n):
        window = arr[i-period+1:i+1]
        valid = window[~np.isnan(window)]
        if len(valid) >= period // 2:
            sma[i] = np.mean(valid)
    return sma


def compute_donchian(H, L, period=20):
    """Donchian channel: upper = highest high, lower = lowest low."""
    n = len(H)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    for i in range(period, n):
        h_win = H[i-period:i]
        l_win = L[i-period:i]
        h_valid = h_win[~np.isnan(h_win)]
        l_valid = l_win[~np.isnan(l_win)]
        if len(h_valid) > 0:
            upper[i] = np.max(h_valid)
        if len(l_valid) > 0:
            lower[i] = np.min(l_valid)
    return upper, lower


def compute_vdp(C, H, L, V):
    """Volume Delta Pressure."""
    n = len(C)
    vdp = np.zeros(n)
    for i in range(n):
        rng = H[i] - L[i]
        if rng > 0 and V[i] > 0 and not np.isnan(C[i]):
            vdp[i] = V[i] * (2 * C[i] - H[i] - L[i]) / rng
    return vdp


def compute_ker(C, period=10):
    """Kaufman Efficiency Ratio."""
    n = len(C)
    ker = np.full(n, np.nan)
    for i in range(period, n):
        direction = abs(C[i] - C[i - period])
        path = np.sum(np.abs(np.diff(C[i-period:i+1])))
        ker[i] = direction / path if path > 0 else 0
    return ker


def compute_rsi(C, period=14):
    """RSI."""
    n = len(C)
    rsi = np.full(n, np.nan)
    if n < period + 1:
        return rsi
    delta = np.diff(C, prepend=C[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = np.mean(gain[1:period+1])
    avg_l = np.mean(loss[1:period+1])
    for i in range(period, n):
        avg_g = (avg_g * (period - 1) + gain[i]) / period
        avg_l = (avg_l * (period - 1) + loss[i]) / period
        if avg_l > 0:
            rsi[i] = 100 - 100 / (1 + avg_g / avg_l)
        else:
            rsi[i] = 100
    return rsi


def compute_bollinger_pct(C, period=20, num_std=2.0):
    """Bollinger %B: position within bands (0=lower, 1=upper)."""
    n = len(C)
    pct = np.full(n, np.nan)
    for i in range(period - 1, n):
        window = C[i-period+1:i+1]
        valid = window[~np.isnan(window)]
        if len(valid) >= period // 2:
            m = np.mean(valid)
            s = np.std(valid, ddof=0)
            if s > 0:
                pct[i] = (C[i] - (m - num_std * s)) / (2 * num_std * s)
    return pct


def compute_linreg_slope(C, period=20):
    """Linear regression slope (normalized by price)."""
    n = len(C)
    slope = np.full(n, np.nan)
    x = np.arange(period, dtype=float)
    x_mean = x.mean()
    denom = np.sum((x - x_mean) ** 2)
    for i in range(period - 1, n):
        window = C[i-period+1:i+1]
        valid_idx = ~np.isnan(window)
        if np.sum(valid_idx) < period // 2:
            continue
        y = window[valid_idx]
        if len(y) < period // 2:
            continue
        xv = np.arange(len(y), dtype=float)
        xm = xv.mean()
        d = np.sum((xv - xm) ** 2)
        if d > 0:
            slope[i] = np.sum((xv - xm) * (y - y.mean())) / d
            if y.mean() > 0:
                slope[i] = slope[i] / y.mean()  # normalize
    return slope


# ============================================================
# PRECOMPUTE ALL SIGNALS
# ============================================================

def precompute_signals(NS, ND, C, O, H, L, V, OI):
    """Precompute all signals for all stocks. Returns dict of arrays."""
    print("[Signals] Precomputing...", flush=True)
    t0 = time.time()
    sigs = {}

    # Per-stock signals
    for si in range(NS):
        c, o, h, l, v, oi = C[si], O[si], H[si], L[si], V[si], OI[si]
        valid = ~np.isnan(c)
        if np.sum(valid) < 60:
            continue

        sym_key = si

        # ATR
        atr = compute_atr(h, l, c, 14)
        atr20 = compute_atr(h, l, c, 20)

        # Moving averages
        ema10 = compute_ema(c, 10)
        ema20 = compute_ema(c, 20)
        ema50 = compute_ema(c, 50)
        sma200 = compute_sma(c, 200)

        # Momentum
        mom5 = np.full(ND, np.nan)
        mom10 = np.full(ND, np.nan)
        mom20 = np.full(ND, np.nan)
        for i in range(5, ND):
            if not np.isnan(c[i]) and not np.isnan(c[i-5]):
                mom5[i] = (c[i] - c[i-5]) / c[i-5]
        for i in range(10, ND):
            if not np.isnan(c[i]) and not np.isnan(c[i-10]):
                mom10[i] = (c[i] - c[i-10]) / c[i-10]
        for i in range(20, ND):
            if not np.isnan(c[i]) and not np.isnan(c[i-20]):
                mom20[i] = (c[i] - c[i-20]) / c[i-20]

        # VDP and EMA of VDP
        vdp = compute_vdp(c, h, l, v)
        vdp_ema15 = compute_ema(vdp, 15)

        # KER (Kaufman Efficiency Ratio)
        ker = compute_ker(c, 10)
        ker20 = compute_ker(c, 20)

        # RSI
        rsi = compute_rsi(c, 14)

        # Bollinger %B
        bb_pct = compute_bollinger_pct(c, 20, 2.0)

        # Linear regression slope
        lr_slope = compute_linreg_slope(c, 20)

        # Donchian channels
        donch_upper20, donch_lower20 = compute_donchian(h, l, 20)
        donch_upper10, donch_lower10 = compute_donchian(h, l, 10)

        # OI momentum
        oi_mom5 = np.full(ND, np.nan)
        oi_mom10 = np.full(ND, np.nan)
        for i in range(5, ND):
            if not np.isnan(oi[i]) and oi[i-5] > 0 and not np.isnan(oi[i-5]):
                oi_mom5[i] = (oi[i] - oi[i-5]) / oi[i-5]
        for i in range(10, ND):
            if not np.isnan(oi[i]) and oi[i-10] > 0 and not np.isnan(oi[i-10]):
                oi_mom10[i] = (oi[i] - oi[i-10]) / oi[i-10]

        # Volatility metrics
        atr_pct = np.full(ND, np.nan)
        for i in range(ND):
            if not np.isnan(atr[i]) and c[i] > 0:
                atr_pct[i] = atr[i] / c[i]

        # Squeeze detection: fast ATR / slow ATR ratio
        atr_ratio = np.full(ND, np.nan)
        atr_fast = compute_atr(h, l, c, 7)
        atr_slow = compute_atr(h, l, c, 30)
        for i in range(ND):
            if not np.isnan(atr_fast[i]) and not np.isnan(atr_slow[i]) and atr_slow[i] > 0:
                atr_ratio[i] = atr_fast[i] / atr_slow[i]

        # ATR compression: current ATR vs 60-day average ATR
        atr_compression = np.full(ND, np.nan)
        for i in range(60, ND):
            window = atr[i-60:i]
            valid_w = window[~np.isnan(window)]
            if len(valid_w) > 20 and not np.isnan(atr[i]):
                avg_atr = np.mean(valid_w)
                if avg_atr > 0:
                    atr_compression[i] = atr[i] / avg_atr

        # Volume metrics
        vol_avg20 = compute_sma(v, 20)
        rel_vol = np.full(ND, np.nan)
        for i in range(ND):
            if not np.isnan(v[i]) and not np.isnan(vol_avg20[i]) and vol_avg20[i] > 0:
                rel_vol[i] = v[i] / vol_avg20[i]

        # Intraday range (normalized)
        intraday_range = np.full(ND, np.nan)
        for i in range(ND):
            if not np.isnan(h[i]) and not np.isnan(l[i]) and not np.isnan(o[i]) and o[i] > 0:
                intraday_range[i] = (h[i] - l[i]) / o[i]

        # Body ratio (trend strength)
        body_ratio = np.full(ND, np.nan)
        for i in range(ND):
            if not np.isnan(c[i]) and not np.isnan(o[i]) and not np.isnan(h[i]) and not np.isnan(l[i]):
                rng = h[i] - l[i]
                if rng > 0:
                    body_ratio[i] = abs(c[i] - o[i]) / rng

        # Store all signals
        sigs[si] = {
            'atr': atr, 'atr20': atr20, 'atr_pct': atr_pct,
            'ema10': ema10, 'ema20': ema20, 'ema50': ema50, 'sma200': sma200,
            'mom5': mom5, 'mom10': mom10, 'mom20': mom20,
            'vdp': vdp, 'vdp_ema15': vdp_ema15,
            'ker': ker, 'ker20': ker20,
            'rsi': rsi, 'bb_pct': bb_pct,
            'lr_slope': lr_slope,
            'donch_upper20': donch_upper20, 'donch_lower20': donch_lower20,
            'donch_upper10': donch_upper10, 'donch_lower10': donch_lower10,
            'oi_mom5': oi_mom5, 'oi_mom10': oi_mom10,
            'atr_ratio': atr_ratio, 'atr_compression': atr_compression,
            'rel_vol': rel_vol, 'intraday_range': intraday_range,
            'body_ratio': body_ratio,
        }

    print(f"  Done in {time.time()-t0:.1f}s, {len(sigs)} stocks", flush=True)
    return sigs


# ============================================================
# SCORING FUNCTIONS
# ============================================================

def score_squeeze_breakout(si, di, sigs, C, H, L, V, OI, ND, params):
    """Squeeze Breakout: ATR compression → expansion + Donchian breakout.

    Key insight: commodities emerging from volatility squeeze tend to
    make large directional moves (5-10% in 5-15 days).

    Score = weighted sum of:
      - Donchian 20-day breakout (close above upper channel)
      - ATR expanding after compression (atr_compression < 0.8 → now > 1.0)
      - KER trend efficiency > 0.3
      - VDP positive (buying pressure)
      - OI increasing (money flowing in)
      - Volume confirmation
    """
    if si not in sigs:
        return np.nan
    s = sigs[si]
    c = C[si, di]
    if np.isnan(c) or c <= 0:
        return np.nan

    score = 0.0

    # 1. Donchian breakout (weight: 30%)
    donch_u = s['donch_upper20'][di] if not np.isnan(s['donch_upper20'][di]) else 0
    if donch_u > 0 and c >= donch_u * 0.998:  # within 0.2% of breakout
        score += 0.30
        # Strong breakout = close significantly above channel
        if c > donch_u * 1.01:
            score += 0.10

    # 2. ATR compression/expansion (weight: 20%)
    comp = s['atr_compression'][di]
    if not np.isnan(comp):
        if comp < params.get('squeeze_thresh', 0.85):  # still compressed
            score += 0.10
        elif comp > 1.0:  # expanding after compression
            score += 0.20

    # 3. KER trend efficiency (weight: 15%)
    ker = s['ker'][di]
    if not np.isnan(ker) and ker > params.get('ker_thresh', 0.3):
        score += 0.15 * min(ker / 0.5, 1.0)

    # 4. VDP direction (weight: 15%)
    vdp = s['vdp_ema15'][di]
    if not np.isnan(vdp) and vdp > 0:
        score += 0.15

    # 5. OI momentum (weight: 10%)
    oi_m = s['oi_mom5'][di]
    if not np.isnan(oi_m) and oi_m > 0:
        score += 0.10

    # 6. Volume confirmation (weight: 10%)
    rv = s['rel_vol'][di]
    if not np.isnan(rv) and rv > 1.2:
        score += 0.10

    # 7. Trend filter: price above EMA50
    ema50 = s['ema50'][di]
    if not np.isnan(ema50) and c > ema50:
        score += 0.05
    else:
        score -= 0.15  # penalize counter-trend

    # 8. Momentum confirmation
    mom = s['mom10'][di]
    if not np.isnan(mom) and mom > 0:
        score += 0.05

    return score if score > params.get('min_score', 0.3) else np.nan


def score_turtle_trend(si, di, sigs, C, H, L, V, OI, ND, params):
    """Turtle-style trend following with multi-factor confirmation.

    Classic Donchian breakout with:
    - EMA200 trend filter (only long above 200MA)
    - KER efficiency confirmation
    - VDP + OI money flow confirmation
    - Intended for 5-15 day holds
    """
    if si not in sigs:
        return np.nan
    s = sigs[si]
    c = C[si, di]
    if np.isnan(c) or c <= 0:
        return np.nan

    score = 0.0

    # 1. Donchian 20-day breakout
    donch_u = s['donch_upper20'][di]
    if np.isnan(donch_u) or donch_u <= 0:
        return np.nan
    if c > donch_u:
        score += 0.30

    # 2. Price above EMA200 (major trend filter)
    sma200 = s['sma200'][di]
    if np.isnan(sma200):
        return np.nan
    if c < sma200:
        return np.nan  # hard filter: no counter-trend
    score += 0.10

    # 3. EMA10 > EMA50 > EMA200 (stacked MAs)
    ema10 = s['ema10'][di]
    ema50 = s['ema50'][di]
    if not np.isnan(ema10) and not np.isnan(ema50):
        if ema10 > ema50 > sma200:
            score += 0.15

    # 4. KER > threshold
    ker = s['ker'][di]
    if not np.isnan(ker) and ker > params.get('ker_thresh', 0.25):
        score += 0.10

    # 5. Linear regression slope positive
    lr = s['lr_slope'][di]
    if not np.isnan(lr) and lr > 0:
        score += 0.10

    # 6. VDP positive
    vdp = s['vdp_ema15'][di]
    if not np.isnan(vdp) and vdp > 0:
        score += 0.10

    # 7. OI increasing
    oi_m = s['oi_mom5'][di]
    if not np.isnan(oi_m) and oi_m > 0:
        score += 0.05

    # 8. High ATR% = bigger moves available
    atr_pct = s['atr_pct'][di]
    if not np.isnan(atr_pct) and atr_pct > params.get('min_atr_pct', 0.015):
        score += 0.10

    return score if score > params.get('min_score', 0.3) else np.nan


def score_momentum_flow(si, di, sigs, C, H, L, V, OI, ND, params):
    """Momentum + Money Flow composite.

    Combines momentum ranking with VDP and OI flow confirmation.
    This was the best performer in v14b (+73%) but extended to longer holds.
    """
    if si not in sigs:
        return np.nan
    s = sigs[si]
    c = C[si, di]
    if np.isnan(c) or c <= 0:
        return np.nan

    score = 0.0
    mom = s[params.get('mom_key', 'mom10')][di]
    if np.isnan(mom) or mom <= 0:
        return np.nan

    # Base: momentum strength (0-0.4)
    score += min(mom / 0.15, 1.0) * 0.40

    # VDP direction confirmation
    vdp = s['vdp_ema15'][di]
    if not np.isnan(vdp):
        if vdp > 0:
            score += 0.20
        else:
            score -= 0.10  # VDP divergence = warning

    # OI flow
    oi_m = s['oi_mom5'][di]
    if not np.isnan(oi_m):
        if oi_m > 0:
            score += 0.15
        else:
            score -= 0.05

    # KER efficiency
    ker = s['ker'][di]
    if not np.isnan(ker) and ker > 0.3:
        score += 0.10

    # Trend filter
    ema50 = s['ema50'][di]
    if not np.isnan(ema50):
        if c > ema50:
            score += 0.10
        else:
            score -= 0.15

    # Volume surge
    rv = s['rel_vol'][di]
    if not np.isnan(rv) and rv > 1.5:
        score += 0.05

    return score if score > params.get('min_score', 0.3) else np.nan


def score_pullback_entry(si, di, sigs, C, H, L, V, OI, ND, params):
    """Pullback in uptrend - uses body_ratio from precomputed signals instead of O array."""
    """Pullback in uptrend — buy the dip in a trend.

    Entry logic:
    1. Overall uptrend (EMA10 > EMA50, price above EMA200)
    2. Short-term pullback (close below EMA10 or RSI < 40)
    3. VDP still positive (buying pressure on pullback = accumulation)
    4. OI still increasing (money not leaving)
    5. Reversal candle (body_ratio > 0.6, close > open)
    """
    if si not in sigs:
        return np.nan
    s = sigs[si]
    c = C[si, di]
    # Use body_ratio from precomputed signals to avoid needing O array
    body = s['body_ratio'][di]
    if np.isnan(c) or c <= 0:
        return np.nan

    score = 0.0

    # 1. Uptrend: EMA10 > EMA50 > price
    ema10 = s['ema10'][di]
    ema50 = s['ema50'][di]
    sma200 = s['sma200'][di]
    if np.isnan(ema10) or np.isnan(ema50) or np.isnan(sma200):
        return np.nan
    if not (ema10 > ema50 and c > sma200):
        return np.nan
    score += 0.20

    # 2. Pullback: price near or below EMA10
    if c < ema10 * 1.02:  # within 2% of EMA10 (slight pullback)
        score += 0.15
    if c < ema10:  # actual pullback below EMA10
        score += 0.10

    # 3. RSI not overbought
    rsi = s['rsi'][di]
    if not np.isnan(rsi):
        if rsi < 50:  # room to run
            score += 0.10
        elif rsi > 70:  # overbought, skip
            return np.nan

    # 4. VDP positive on pullback (accumulation)
    vdp = s['vdp_ema15'][di]
    if not np.isnan(vdp) and vdp > 0:
        score += 0.15

    # 5. Reversal candle: strong body (close near high of range)
    if not np.isnan(body) and body > 0.5:
        score += 0.15

    # 6. OI still increasing
    oi_m = s['oi_mom5'][di]
    if not np.isnan(oi_m) and oi_m > 0:
        score += 0.10

    # 7. KER trend quality
    ker = s['ker'][di]
    if not np.isnan(ker) and ker > 0.2:
        score += 0.05

    return score if score > params.get('min_score', 0.4) else np.nan


def score_composite(si, di, sigs, C, H, L, V, OI, ND, params):
    """Composite: combines squeeze, trend, momentum, and flow.

    The most selective signal — requires multiple confirmations.
    Only triggers when conditions are exceptional.
    """
    if si not in sigs:
        return np.nan
    s = sigs[si]
    c = C[si, di]
    if np.isnan(c) or c <= 0:
        return np.nan

    score = 0.0
    confirmations = 0

    # A: Trend filter (required)
    ema50 = s['ema50'][di]
    sma200 = s['sma200'][di]
    if np.isnan(sma200) or c < sma200:
        return np.nan
    if not np.isnan(ema50) and c > ema50:
        confirmations += 1

    # B: Momentum positive
    mom = s['mom10'][di]
    if not np.isnan(mom) and mom > 0:
        confirmations += 1
        score += min(mom / 0.10, 1.0) * 0.15

    # C: KER efficiency
    ker = s['ker'][di]
    if not np.isnan(ker) and ker > 0.3:
        confirmations += 1
        score += 0.10

    # D: VDP positive
    vdp = s['vdp_ema15'][di]
    if not np.isnan(vdp) and vdp > 0:
        confirmations += 1
        score += 0.10

    # E: OI increasing
    oi_m = s['oi_mom5'][di]
    if not np.isnan(oi_m) and oi_m > 0:
        confirmations += 1
        score += 0.05

    # F: Volume surge
    rv = s['rel_vol'][di]
    if not np.isnan(rv) and rv > 1.3:
        confirmations += 1
        score += 0.05

    # G: Squeeze detection
    comp = s['atr_compression'][di]
    if not np.isnan(comp):
        if comp < 0.8:
            confirmations += 1
            score += 0.10
        elif comp > 1.0:
            confirmations += 1
            score += 0.15

    # H: Donchian breakout
    donch_u = s['donch_upper20'][di]
    if not np.isnan(donch_u) and donch_u > 0 and c >= donch_u * 0.99:
        confirmations += 1
        score += 0.10

    # Need minimum confirmations
    min_conf = params.get('min_confirmations', 4)
    if confirmations < min_conf:
        return np.nan

    # Bonus for many confirmations
    if confirmations >= 7:
        score += 0.15
    elif confirmations >= 6:
        score += 0.10

    return score if score > params.get('min_score', 0.3) else np.nan


def score_highvol_momentum(si, di, sigs, C, H, L, V, OI, ND, params):
    """High-volatility momentum: only trade commodities with ATR/Price > threshold.

    Insight from v19: high-vol commodities give bigger moves (5.2% avg win vs 2.8%).
    Combined with squeeze detection and flow confirmation.
    """
    if si not in sigs:
        return np.nan
    s = sigs[si]
    c = C[si, di]
    if np.isnan(c) or c <= 0:
        return np.nan

    # Hard filter: high ATR%
    atr_pct = s['atr_pct'][di]
    if np.isnan(atr_pct) or atr_pct < params.get('min_atr_pct', 0.02):
        return np.nan

    score = 0.0

    # Momentum
    mom = s['mom10'][di]
    if np.isnan(mom) or mom <= 0:
        return np.nan
    score += min(mom / 0.15, 1.0) * 0.35

    # Trend filter
    ema50 = s['ema50'][di]
    sma200 = s['sma200'][di]
    if not np.isnan(sma200) and c > sma200:
        score += 0.15
    if not np.isnan(ema50) and c > ema50:
        score += 0.10

    # VDP
    vdp = s['vdp_ema15'][di]
    if not np.isnan(vdp) and vdp > 0:
        score += 0.15

    # KER
    ker = s['ker'][di]
    if not np.isnan(ker) and ker > 0.25:
        score += 0.10

    # OI
    oi_m = s['oi_mom5'][di]
    if not np.isnan(oi_m) and oi_m > 0:
        score += 0.10

    # Volume
    rv = s['rel_vol'][di]
    if not np.isnan(rv) and rv > 1.2:
        score += 0.05

    return score if score > params.get('min_score', 0.3) else np.nan


def score_weinstein_stage2(si, di, sigs, C, H, L, V, OI, ND, params):
    """Weinstein Stage 2 (Advancing) identification.

    Stage 2 = price above rising 30-week MA (150-day), making higher highs.
    Combined with breakout and momentum confirmation.
    """
    if si not in sigs:
        return np.nan
    s = sigs[si]
    c = C[si, di]
    if np.isnan(c) or c <= 0:
        return np.nan

    score = 0.0

    # 1. Price above long-term MA (use precomputed sma200 as proxy for 150-day)
    sma200 = s['sma200'][di]
    if np.isnan(sma200):
        return np.nan
    if c < sma200:
        return np.nan
    score += 0.15

    # 2. Long-term MA rising (compare with 5 days ago via ema50 slope)
    ema50 = s['ema50'][di]
    if not np.isnan(ema50) and di >= 5:
        ema50_5ago = s['ema50'][di - 5]
        if not np.isnan(ema50_5ago) and ema50 > ema50_5ago:
            score += 0.15

    # 3. EMA20 above EMA50 (short-term trend up)
    ema20 = s['ema20'][di]
    ema50 = s['ema50'][di]
    if not np.isnan(ema20) and not np.isnan(ema50) and ema20 > ema50:
        score += 0.15

    # 4. Recent breakout (close near 20-day high)
    donch_u = s['donch_upper20'][di]
    if not np.isnan(donch_u) and donch_u > 0:
        if c >= donch_u * 0.98:
            score += 0.15
        if c >= donch_u:
            score += 0.10

    # 5. Momentum
    mom = s['mom20'][di]
    if not np.isnan(mom) and mom > 0:
        score += 0.10

    # 6. VDP + OI
    vdp = s['vdp_ema15'][di]
    if not np.isnan(vdp) and vdp > 0:
        score += 0.10

    oi_m = s['oi_mom5'][di]
    if not np.isnan(oi_m) and oi_m > 0:
        score += 0.10

    return score if score > params.get('min_score', 0.3) else np.nan


# ============================================================
# BACKTEST ENGINE
# ============================================================

def run_backtest(NS, ND, dates, C, O, H, L, V, OI, syms, sigs,
                score_fn, name, params,
                hold_max=10, trail_atr=2.0, stop_loss=0.04,
                allow_short=False, reentry_gap=2):
    """Generic backtest engine with trailing stop and multiple exit conditions.

    Enhanced features:
    - ATR-adaptive trailing stop
    - Score-based rotation (switch to better candidate)
    - Signal flip exit
    - Time exit with adaptive holding
    - Reentry cooldown
    """
    cash = float(CASH0)
    trades = []
    pos = None
    last_exit = {}  # sym -> exit_di

    for di in range(MIN_TRAIN, ND):
        year = dates[di].year

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

            # 1. Fixed stop loss
            if pnl_pct / 100 < -stop_loss:
                exit_reason = 'stop'

            # 2. Trailing stop (ATR-based)
            if exit_reason is None and trail_atr > 0:
                atr = pos.get('atr', 0)
                if atr > 0:
                    trail_price = pos.get('trail_price', pos['entry'])
                    if pos['dir'] == 1:
                        new_trail = c - trail_atr * atr
                        if new_trail > trail_price:
                            pos['trail_price'] = new_trail
                        if c < trail_price and days_held >= 2:
                            exit_reason = 'trail'
                    else:
                        new_trail = c + trail_atr * atr
                        if new_trail < trail_price:
                            pos['trail_price'] = new_trail
                        if c > trail_price and days_held >= 2:
                            exit_reason = 'trail'

            # 3. Signal flip exit
            if exit_reason is None and days_held >= 3:
                cur_score = score_fn(pos['si'], di, sigs, C, H, L, V, OI, ND, params)
                if not np.isnan(cur_score):
                    if pos['dir'] == 1 and cur_score < -0.02:
                        exit_reason = 'signal_flip'
                    elif pos['dir'] == -1 and cur_score > 0.02:
                        exit_reason = 'signal_flip'

            # 4. Time exit
            if exit_reason is None and days_held >= hold_max:
                exit_reason = 'time'

            # 5. Better candidate rotation (after minimum hold)
            if exit_reason is None and days_held >= max(3, hold_max // 2):
                best_si, best_dir, best_sc = -1, 0, 0
                for sj in range(NS):
                    sc = score_fn(sj, di, sigs, C, H, L, V, OI, ND, params)
                    if np.isnan(sc): continue
                    if sc > best_sc:
                        best_sc = sc; best_si = sj; best_dir = 1
                    if allow_short and -sc > best_sc:
                        best_sc = -sc; best_si = sj; best_dir = -1

                cur_sc = score_fn(pos['si'], di, sigs, C, H, L, V, OI, ND, params)
                cur_sc = abs(cur_sc) if not np.isnan(cur_sc) else 0
                if best_sc > cur_sc * 1.5 + 0.1 and best_si != pos['si']:
                    exit_reason = 'rotate'

            if exit_reason:
                cost_out = mkt_val * COMM
                cash += mkt_val - cost_out
                trades.append({
                    'pnl_pct': pnl_pct, 'pnl_abs': pnl,
                    'days': days_held, 'di': di, 'year': year,
                    'sym': pos['sym'], 'dir': pos['dir'],
                    'reason': exit_reason, 'entry_date': pos['entry_di'],
                })
                last_exit[pos['sym']] = di
                pos = None

        # === ENTRY ===
        if pos is None:
            best_si, best_dir, best_sc = -1, 0, 0
            for si in range(NS):
                sc = score_fn(si, di, sigs, C, H, L, V, OI, ND, params)
                if np.isnan(sc): continue

                sym = syms[si]
                if sym in last_exit and di - last_exit[sym] < reentry_gap:
                    continue

                if sc > best_sc:
                    best_sc = sc; best_si = si; best_dir = 1
                if allow_short and -sc > best_sc:
                    best_sc = -sc; best_si = si; best_dir = -1

            if best_si >= 0 and best_sc > 0:
                c = C[best_si, di]
                if np.isnan(c) or c <= 0: continue

                sym = syms[best_si]
                mult = MULT.get(sym, DEF_MULT)
                notional = c * mult
                if notional <= 0: continue

                lots = int(cash / notional)
                if lots <= 0: continue

                cost_in = notional * lots * (1 + COMM)
                if cost_in > cash: continue

                # Get ATR for trailing stop
                atr_val = 0
                trs = []
                for dd in range(max(1, di-14), di+1):
                    hi = H[best_si, dd]; lo = L[best_si, dd]; pc = C[best_si, dd-1]
                    if np.isnan(hi) or np.isnan(lo): continue
                    tr = hi - lo
                    if not np.isnan(pc):
                        tr = max(tr, abs(hi-pc), abs(lo-pc))
                    trs.append(tr)
                if trs:
                    atr_val = np.mean(trs)

                cash -= cost_in
                trail_price = c - trail_atr * atr_val if best_dir == 1 else c + trail_atr * atr_val
                pos = {
                    'si': best_si, 'entry': c, 'entry_di': di,
                    'lots': lots, 'dir': best_dir, 'sym': sym,
                    'atr': atr_val, 'trail_price': trail_price,
                    'score': best_sc,
                }

    # Close remaining position
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
            'sym': pos['sym'], 'dir': pos['dir'],
            'reason': 'end', 'entry_date': pos['entry_di'],
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
        'avg_win': round(avg_win, 3), 'avg_loss': round(avg_loss, 3),
        'cash': round(cash, 0), 'yearly': year_stats,
        'reasons': reasons, 'trades': trades,
    }


def print_result(r):
    """Print a result summary."""
    if r is None:
        print("  [SKIP] No trades")
        return
    print(f"  {r['name']:45s} | Ann {r['ann']:+7.1f}% | WR {r['wr']:5.1f}% | "
          f"N {r['n']:4d} | DD {r['dd']:6.1f}% | AvgPnl {r['avg_pnl']:+.3f}% | "
          f"AvgDays {r['avg_days']:.1f} | W/L {r['avg_win']:.2f}/{r['avg_loss']:.2f}")
    if r.get('reasons'):
        parts = []
        for reason, stats in sorted(r['reasons'].items()):
            wr = stats['w'] / stats['n'] * 100 if stats['n'] > 0 else 0
            parts.append(f"{reason}:{stats['n']}({wr:.0f}%)")
        print(f"  {'':45s} | Exits: {' | '.join(parts)}")


# ============================================================
# MAIN: RUN ALL CONFIGURATIONS
# ============================================================

def main():
    t_start = time.time()
    print("=" * 100)
    print("Alpha Futures V23 — Multi-Signal Composite Strategy")
    print("=" * 100)

    # Load data
    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(
        max_stocks=500, load_oi=True
    )

    # Precompute signals
    sigs = precompute_signals(NS, ND, C, O, H, L, V, OI)

    results = []

    # === STRATEGY 1: Squeeze Breakout (multiple configs) ===
    print("\n--- Squeeze Breakout ---")
    for sq_thresh in [0.80, 0.85, 0.90]:
        for ker_t in [0.25, 0.35]:
            for hold in [5, 8, 12]:
                for trail in [1.5, 2.0, 2.5]:
                    for sl in [0.03, 0.05]:
                        params = {
                            'squeeze_thresh': sq_thresh,
                            'ker_thresh': ker_t,
                            'min_score': 0.3,
                        }
                        name = f"SQUEEZE_SQ{sq_thresh}_K{ker_t}_H{hold}_T{trail}_SL{sl}"
                        r = run_backtest(NS, ND, dates, C, O, H, L, V, OI, syms,
                                        sigs, score_squeeze_breakout, name, params,
                                        hold_max=hold, trail_atr=trail, stop_loss=sl)
                        if r:
                            results.append(r)

    # === STRATEGY 2: Turtle Trend (multiple configs) ===
    print("\n--- Turtle Trend ---")
    for ker_t in [0.20, 0.30]:
        for min_atr in [0.010, 0.015, 0.020]:
            for hold in [7, 10, 15]:
                for trail in [2.0, 3.0]:
                    for sl in [0.04, 0.06]:
                        params = {
                            'ker_thresh': ker_t,
                            'min_atr_pct': min_atr,
                            'min_score': 0.3,
                        }
                        name = f"TURTLE_K{ker_t}_A{min_atr}_H{hold}_T{trail}_SL{sl}"
                        r = run_backtest(NS, ND, dates, C, O, H, L, V, OI, syms,
                                        sigs, score_turtle_trend, name, params,
                                        hold_max=hold, trail_atr=trail, stop_loss=sl)
                        if r:
                            results.append(r)

    # === STRATEGY 3: Momentum Flow (extended hold) ===
    print("\n--- Momentum Flow ---")
    for mom_key in ['mom5', 'mom10', 'mom20']:
        for hold in [5, 8, 12]:
            for trail in [1.5, 2.0, 2.5]:
                for sl in [0.03, 0.05]:
                    params = {
                        'mom_key': mom_key,
                        'min_score': 0.3,
                    }
                    name = f"MFLOW_{mom_key}_H{hold}_T{trail}_SL{sl}"
                    r = run_backtest(NS, ND, dates, C, O, H, L, V, OI, syms,
                                    sigs, score_momentum_flow, name, params,
                                    hold_max=hold, trail_atr=trail, stop_loss=sl)
                    if r:
                        results.append(r)

    # === STRATEGY 4: Pullback Entry ===
    print("\n--- Pullback Entry ---")
    for hold in [7, 10, 15]:
        for trail in [2.0, 2.5, 3.0]:
            for sl in [0.04, 0.06]:
                for ms in [0.4, 0.5]:
                    params = {'min_score': ms}
                    name = f"PULLBACK_H{hold}_T{trail}_SL{sl}_MS{ms}"
                    r = run_backtest(NS, ND, dates, C, O, H, L, V, OI, syms,
                                    sigs, score_pullback_entry, name, params,
                                    hold_max=hold, trail_atr=trail, stop_loss=sl)
                    if r:
                        results.append(r)

    # === STRATEGY 5: Composite (multi-factor) ===
    print("\n--- Composite ---")
    for min_conf in [4, 5, 6]:
        for hold in [7, 10, 15]:
            for trail in [2.0, 2.5]:
                for sl in [0.04, 0.06]:
                    params = {
                        'min_confirmations': min_conf,
                        'min_score': 0.3,
                    }
                    name = f"COMP_C{min_conf}_H{hold}_T{trail}_SL{sl}"
                    r = run_backtest(NS, ND, dates, C, O, H, L, V, OI, syms,
                                    sigs, score_composite, name, params,
                                    hold_max=hold, trail_atr=trail, stop_loss=sl)
                    if r:
                        results.append(r)

    # === STRATEGY 6: High-Vol Momentum ===
    print("\n--- High-Vol Momentum ---")
    for min_atr in [0.015, 0.020, 0.025]:
        for hold in [5, 8, 10]:
            for trail in [1.5, 2.0, 2.5]:
                for sl in [0.03, 0.05]:
                    params = {
                        'min_atr_pct': min_atr,
                        'min_score': 0.3,
                    }
                    name = f"HIGHVOL_A{min_atr}_H{hold}_T{trail}_SL{sl}"
                    r = run_backtest(NS, ND, dates, C, O, H, L, V, OI, syms,
                                    sigs, score_highvol_momentum, name, params,
                                    hold_max=hold, trail_atr=trail, stop_loss=sl)
                    if r:
                        results.append(r)

    # === STRATEGY 7: Weinstein Stage 2 ===
    print("\n--- Weinstein Stage 2 ---")
    for hold in [7, 10, 15]:
        for trail in [2.0, 2.5, 3.0]:
            for sl in [0.04, 0.06]:
                params = {'min_score': 0.3}
                name = f"WEINSTEIN_H{hold}_T{trail}_SL{sl}"
                r = run_backtest(NS, ND, dates, C, O, H, L, V, OI, syms,
                                sigs, score_weinstein_stage2, name, params,
                                hold_max=hold, trail_atr=trail, stop_loss=sl)
                if r:
                    results.append(r)

    # === SUMMARY ===
    print("\n" + "=" * 100)
    print(f"TOTAL CONFIGS TESTED: {len(results)} profitable out of 7 strategy types")
    print("=" * 100)

    if results:
        # Sort by annualized return
        results.sort(key=lambda x: -x['ann'])

        print(f"\n{'='*100}")
        print(f"TOP 30 RESULTS (by annual return):")
        print(f"{'='*100}")
        for r in results[:30]:
            print_result(r)

        print(f"\n--- YEARLY BREAKDOWN (Top 10) ---")
        for r in results[:10]:
            print(f"\n  {r['name']}:")
            for y in sorted(r['yearly'].keys()):
                ys = r['yearly'][y]
                wr = ys['w'] / ys['n'] * 100 if ys['n'] > 0 else 0
                print(f"    {y}: {ys['n']:3d} trades, WR {wr:5.1f}%, PnL {ys['pnl']:+.1f}%")

        # Best by WR
        by_wr = sorted(results, key=lambda x: -x['wr'])
        print(f"\n--- TOP 10 BY WIN RATE ---")
        for r in by_wr[:10]:
            print_result(r)

        # Best by Sharpe-like metric (ann / max_dd)
        by_sharpe = sorted(results, key=lambda x: -x['ann'] / max(x['dd'], 1))
        print(f"\n--- TOP 10 BY RISK-ADJUSTED (Ann/DD) ---")
        for r in by_sharpe[:10]:
            ratio = r['ann'] / max(r['dd'], 1)
            print(f"  {r['name']:45s} | Ann {r['ann']:+7.1f}% | DD {r['dd']:6.1f}% | "
                  f"Ratio {ratio:.2f} | WR {r['wr']:5.1f}% | N {r['n']:4d}")

        # Best by avg win/loss ratio
        by_wr2 = sorted(results, key=lambda x: -x['avg_win'] / max(x['avg_loss'], 0.01))
        print(f"\n--- TOP 10 BY WIN/LOSS RATIO ---")
        for r in by_wr2[:10]:
            wlr = r['avg_win'] / max(r['avg_loss'], 0.01)
            print(f"  {r['name']:45s} | W/L {wlr:.2f} | AvgW {r['avg_win']:.3f}% | "
                  f"AvgL {r['avg_loss']:.3f}% | WR {r['wr']:5.1f}% | Ann {r['ann']:+7.1f}%")

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
