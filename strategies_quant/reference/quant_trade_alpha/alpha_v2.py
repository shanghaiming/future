"""
Alpha V2 — Simple Strategies with Real Signal Quality
=====================================================
Based on deep research of 211 existing strategies:
- 6/18 momentum strategies are just SMA crossover variants → build genuinely independent signals
- FRAMA, energy model, K-means code exists but isn't used → USE them
- VDP is the most important formula → build around it
- Only 3/18 mean-reversion strategies have trend filters → always include trend filter
- A-share specific alpha: T+1, 10% limits, retail overreaction

Each strategy is self-contained and testable. No ML models.
"""
import sys, os, time, warnings, pickle
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.data_loader import list_available_symbols, load_stock_data

COMMISSION = 0.0003
STAMP_DUTY = 0.001
CASH0 = 500_000

# ============================================================
# DATA LOADING (shared by all strategies)
# ============================================================
def load_all_data(max_stocks=500, min_days=300, start='2016-01-01', end=None):
    """Load OHLCV data for top stocks by volume."""
    print("[Data] Loading...", flush=True)
    t0 = time.time()
    stock_data = {}
    for sym in list_available_symbols('daily'):
        try:
            df = load_stock_data(sym, frequency='daily')
            if df is not None and len(df) >= min_days:
                cols = [c for c in ['open','high','low','close','vol','volume','amount'] if c in df.columns]
                stock_data[sym] = df[cols].copy()
                if 'vol' in df.columns and 'volume' not in df.columns:
                    stock_data[sym].rename(columns={'vol': 'volume'}, inplace=True)
        except:
            pass

    vol_map = {s: df['volume'].tail(60).mean() for s, df in stock_data.items()
               if 'volume' in df.columns and df['volume'].tail(60).mean() > 0}
    syms = sorted([s for s, _ in sorted(vol_map.items(), key=lambda x: -x[1])[:max_stocks]])
    NS = len(syms)
    sym_set = set(syms)
    all_dates = sorted(set(d for s in syms for d in stock_data[s].index))

    i0 = next(i for i, d in enumerate(all_dates) if d >= pd.Timestamp(start))
    if end:
        i1 = next((i for i, d in enumerate(all_dates) if d > pd.Timestamp(end)), len(all_dates)) - 1
    else:
        i1 = len(all_dates) - 1
    dates = all_dates[i0:i1+1]
    ND = len(dates)
    dm = {d: i for i, d in enumerate(all_dates)}

    # Build arrays: [stock, date]
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

    C = C[:, i0:i1+1]
    O = O[:, i0:i1+1]
    H = H[:, i0:i1+1]
    L = L[:, i0:i1+1]
    V = V[:, i0:i1+1]
    print(f"  {NS} stocks, {ND} days ({time.time()-t0:.1f}s)", flush=True)
    return NS, ND, dates, C, O, H, L, V, syms, sym_set


# ============================================================
# SIGNAL COMPUTATION FUNCTIONS
# ============================================================

def compute_frama(close, high, low, period=16, FC=1, SC=200):
    """Fractal Adaptive Moving Average — adapts speed based on fractal dimension.
    Fast (FC) in trends, slow (SC) in ranges."""
    n = len(close)
    frama = np.full(n, np.nan)
    if n < period:
        return frama
    frama[period-1] = np.mean(close[:period])
    for i in range(period, n):
        H1 = np.max(high[i-period:i])
        L1 = np.min(low[i-period:i])
        N1 = (H1 - L1) / period
        half = period // 2
        H2 = np.max(high[i-half:i])
        L2 = np.min(low[i-half:i])
        N2 = (H2 - L2) / half
        H3 = np.max(high[i-period:i-half])
        L3 = np.min(low[i-period:i-half])
        N3 = (H3 - L3) / half
        if N1 > 0 and N2 > 0 and N3 > 0:
            D = (np.log(N1 + N2) - np.log(N3)) / np.log(2)
        else:
            D = 1.5
        D = max(1, min(2, D))
        alpha = np.exp(-4.6 * (D - 1))
        alpha = max(2/(SC+1), min(2/(FC+1)-0.01, alpha))
        frama[i] = alpha * close[i] + (1 - alpha) * frama[i-1]
    return frama


def compute_vdp(close, high, low, volume):
    """Volume Delta Pressure = V * (2C - H - L) / (H - L).
    Measures net buying/selling pressure per bar.
    +V = buying pressure, -V = selling pressure."""
    n = len(close)
    vdp = np.zeros(n)
    for i in range(n):
        rng = high[i] - low[i]
        if rng > 0 and volume[i] > 0:
            vdp[i] = volume[i] * (2 * close[i] - high[i] - low[i]) / rng
    return vdp


def compute_rsi(close, period=14):
    """RSI computation."""
    n = len(close)
    rsi = np.full(n, np.nan)
    if n < period + 1:
        return rsi
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.mean(gain[1:period+1])
    avg_loss = np.mean(loss[1:period+1])
    for i in range(period, n):
        avg_gain = (avg_gain * (period - 1) + gain[i]) / period
        avg_loss = (avg_loss * (period - 1) + loss[i]) / period
        if avg_loss > 0:
            rs = avg_gain / avg_loss
            rsi[i] = 100 - 100 / (1 + rs)
        else:
            rsi[i] = 100
    return rsi


def compute_ker(close, period=10):
    """Kaufman Efficiency Ratio = |direction| / volatility.
    KER ≈ 1 in strong trends, ≈ 0 in choppy markets."""
    n = len(close)
    ker = np.full(n, np.nan)
    if n < period + 1:
        return ker
    for i in range(period, n):
        direction = abs(close[i] - close[i - period])
        volatility = np.sum(np.abs(np.diff(close[i-period:i+1])))
        ker[i] = direction / volatility if volatility > 0 else 0
    return ker


def compute_bollinger(close, period=20, num_std=2.0):
    """Bollinger Bands."""
    n = len(close)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    mid = np.full(n, np.nan)
    pct = np.full(n, np.nan)  # %B: position within bands
    for i in range(period - 1, n):
        window = close[i-period+1:i+1]
        m = np.mean(window)
        s = np.std(window, ddof=0)
        upper[i] = m + num_std * s
        lower[i] = m - num_std * s
        mid[i] = m
        if s > 0:
            pct[i] = (close[i] - lower[i]) / (upper[i] - lower[i])
    return upper, lower, mid, pct


def compute_kalman_velocity(close, Q=0.01, R=0.1):
    """Simplified Kalman filter: estimates price level + velocity.
    Returns velocity array. When velocity changes sign = momentum shift."""
    n = len(close)
    vel = np.zeros(n)
    est = close[0]
    vel_est = 0.0
    P = np.array([[1.0, 0.0], [0.0, 1.0]])
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H_mat = np.array([[1.0, 0.0]])
    Q_mat = np.array([[Q, 0.0], [0.0, Q]])
    R_mat = np.array([[R]])
    for i in range(n):
        if np.isnan(close[i]):
            vel[i] = vel_est
            continue
        # Predict
        x_pred = F @ np.array([est, vel_est])
        P_pred = F @ P @ F.T + Q_mat
        # Update
        y = close[i] - H_mat @ x_pred
        S = H_mat @ P_pred @ H_mat.T + R_mat
        K = P_pred @ H_mat.T / S[0, 0]
        x_upd = x_pred + K.flatten() * y[0]
        P = (np.eye(2) - K @ H_mat) @ P_pred
        est = x_upd[0]
        vel_est = x_upd[1]
        vel[i] = vel_est
    return vel


# ============================================================
# STRATEGY SIGNAL GENERATORS
# Each returns: buy_days[si] = set of day-indices, sell_days[si] = set of day-indices
# ============================================================

MIN_TRAIN = 252  # warmup period

def strategy_1_frama(NS, ND, C, O, H, L, V):
    """S1: FRAMA Crossover — actually uses fractal adaptive MA.
    Buy when FRAMA turns up (adaptive alpha increases), sell when it turns down."""
    name = "S1_FRAMA_Cross"
    buy_days = [set() for _ in range(NS)]
    sell_days = [set() for _ in range(NS)]

    for si in range(NS):
        c, h, l = C[si], H[si], L[si]
        valid = ~np.isnan(c)
        if np.sum(valid) < 60:
            continue
        frama = compute_frama(c, h, l, period=16)
        frama_slope = np.full(ND, np.nan)
        for i in range(1, ND):
            if not np.isnan(frama[i]) and not np.isnan(frama[i-1]):
                frama_slope[i] = frama[i] - frama[i-1]

        for di in range(MIN_TRAIN, ND):
            if np.isnan(frama_slope[di]) or np.isnan(frama_slope[di-1]):
                continue
            # FRAMA slope crosses zero: buy when turning up, sell when turning down
            if frama_slope[di] > 0 and frama_slope[di-1] <= 0:
                buy_days[si].add(di)
            elif frama_slope[di] < 0 and frama_slope[di-1] >= 0:
                sell_days[si].add(di)
    return name, buy_days, sell_days


def strategy_2_vdp_reversal(NS, ND, C, O, H, L, V):
    """S2: VDP Reversal — buy when selling pressure was extreme and reverses.
    VDP < -2sigma = extreme selling → VDP turns positive = smart money buying."""
    name = "S2_VDP_Reversal"
    buy_days = [set() for _ in range(NS)]
    sell_days = [set() for _ in range(NS)]

    for si in range(NS):
        c, h, l, v = C[si], H[si], L[si], V[si]
        valid = ~np.isnan(c)
        if np.sum(valid) < 60:
            continue
        vdp = compute_vdp(c, h, l, v)
        # EMA of VDP (20-day)
        ema_vdp = np.full(ND, np.nan)
        ema_vdp[0] = vdp[0]
        alpha = 2.0 / 21
        for i in range(1, ND):
            if not np.isnan(vdp[i]):
                ema_vdp[i] = alpha * vdp[i] + (1 - alpha) * (ema_vdp[i-1] if not np.isnan(ema_vdp[i-1]) else vdp[i])

        # Rolling std of VDP (20-day)
        vdp_std = np.full(ND, np.nan)
        for i in range(20, ND):
            window = vdp[i-20:i]
            valid_w = window[~np.isnan(window)]
            if len(valid_w) > 10:
                vdp_std[i] = np.std(valid_w)

        # Signal: VDP was below -2σ (extreme selling) and now turns positive
        for di in range(MIN_TRAIN, ND):
            if np.isnan(ema_vdp[di]) or np.isnan(vdp_std[di]) or vdp_std[di] == 0:
                continue
            # Extreme selling pressure in last 5 days
            extreme_selling = False
            for j in range(max(20, di-5), di):
                if not np.isnan(ema_vdp[j]) and ema_vdp[j] < -2 * vdp_std[di]:
                    extreme_selling = True
                    break
            if extreme_selling and ema_vdp[di] > 0:
                buy_days[si].add(di)
            # Sell when VDP goes very positive then reverses
            if di > 5 and not np.isnan(ema_vdp[di-1]) and not np.isnan(ema_vdp[di]):
                if ema_vdp[di-1] > 2 * vdp_std[di] and ema_vdp[di] < ema_vdp[di-1]:
                    sell_days[si].add(di)
    return name, buy_days, sell_days


def strategy_3_oversold_trend(NS, ND, C, O, H, L, V):
    """S3: Oversold + Trend Filter — THE key A-share strategy.
    Buy when RSI < 30 AND price above 50MA (trend filter).
    Sell when RSI > 70 or price drops below 50MA."""
    name = "S3_Oversold_Trend"
    buy_days = [set() for _ in range(NS)]
    sell_days = [set() for _ in range(NS)]

    for si in range(NS):
        c, v = C[si], V[si]
        valid = ~np.isnan(c)
        if np.sum(valid) < 60:
            continue
        rsi = compute_rsi(c, 14)
        ma50 = np.full(ND, np.nan)
        for i in range(49, ND):
            window = c[i-49:i+1]
            valid_w = window[~np.isnan(window)]
            if len(valid_w) >= 30:
                ma50[i] = np.mean(valid_w)

        for di in range(MIN_TRAIN, ND):
            if np.isnan(rsi[di]) or np.isnan(ma50[di]) or np.isnan(c[di]):
                continue
            # Buy: oversold (RSI < 30) + above 50MA (uptrend)
            if rsi[di] < 30 and c[di] > ma50[di]:
                buy_days[si].add(di)
            # Sell: overbought or trend breaks
            elif rsi[di] > 70 or c[di] < ma50[di]:
                sell_days[si].add(di)
    return name, buy_days, sell_days


def strategy_4_ker_momentum(NS, ND, C, O, H, L, V):
    """S4: Kaufman Efficiency + Momentum — only trade when market is trending.
    KER > 0.4 = trending → follow momentum direction.
    KER < 0.2 = choppy → stay out."""
    name = "S4_KER_Momentum"
    buy_days = [set() for _ in range(NS)]
    sell_days = [set() for _ in range(NS)]

    for si in range(NS):
        c = C[si]
        valid = ~np.isnan(c)
        if np.sum(valid) < 60:
            continue
        ker = compute_ker(c, 10)
        mom5 = np.full(ND, np.nan)
        for i in range(5, ND):
            if not np.isnan(c[i]) and not np.isnan(c[i-5]):
                mom5[i] = (c[i] - c[i-5]) / c[i-5]

        for di in range(MIN_TRAIN, ND):
            if np.isnan(ker[di]) or np.isnan(mom5[di]) or np.isnan(c[di]):
                continue
            # Strong trend + positive momentum = buy
            if ker[di] > 0.4 and mom5[di] > 0.02:
                buy_days[si].add(di)
            # Strong trend + negative momentum = sell
            elif ker[di] > 0.4 and mom5[di] < -0.02:
                sell_days[si].add(di)
    return name, buy_days, sell_days


def strategy_5_volume_breakout(NS, ND, C, O, H, L, V):
    """S5: Volume Surge + Price Breakout — institutional accumulation signal.
    Volume > 2x average + close > 20-day high = confirmed breakout."""
    name = "S5_Vol_Breakout"
    buy_days = [set() for _ in range(NS)]
    sell_days = [set() for _ in range(NS)]

    for si in range(NS):
        c, v = C[si], V[si]
        valid = ~np.isnan(c)
        if np.sum(valid) < 60:
            continue
        vol_ma20 = np.full(ND, np.nan)
        high20 = np.full(ND, np.nan)
        for i in range(19, ND):
            vw = v[i-19:i+1]
            cw = c[i-19:i+1]
            vv = vw[~np.isnan(vw)]
            cc = cw[~np.isnan(cw)]
            if len(vv) >= 10:
                vol_ma20[i] = np.mean(vv)
            if len(cc) >= 10:
                high20[i] = np.max(cc[:-1])  # exclude current day

        for di in range(MIN_TRAIN, ND):
            if np.isnan(vol_ma20[di]) or np.isnan(high20[di]) or np.isnan(c[di]) or np.isnan(v[di]):
                continue
            # Buy: volume > 2x average + close > 20-day high
            if v[di] > 2 * vol_ma20[di] and c[di] > high20[di]:
                buy_days[si].add(di)
            # Sell: volume surge + close < 20-day low
            low20 = np.min(c[di-19:di][~np.isnan(c[di-19:di])]) if di >= 20 else np.nan
            if not np.isnan(low20) and v[di] > 1.5 * vol_ma20[di] and c[di] < low20:
                sell_days[si].add(di)
    return name, buy_days, sell_days


def strategy_6_bb_squeeze(NS, ND, C, O, H, L, V):
    """S6: Bollinger Squeeze Breakout — volatility compression → expansion.
    BB width at 120-day low → breakout direction = entry."""
    name = "S6_BB_Squeeze"
    buy_days = [set() for _ in range(NS)]
    sell_days = [set() for _ in range(NS)]

    for si in range(NS):
        c = C[si]
        valid = ~np.isnan(c)
        if np.sum(valid) < 150:
            continue
        bb_up, bb_lo, bb_mid, bb_pct = compute_bollinger(c, 20, 2.0)
        bb_width = np.full(ND, np.nan)
        for i in range(ND):
            if not np.isnan(bb_up[i]) and not np.isnan(bb_lo[i]) and bb_mid[i] > 0:
                bb_width[i] = (bb_up[i] - bb_lo[i]) / bb_mid[i]

        for di in range(MIN_TRAIN + 120, ND):
            if np.isnan(bb_width[di]) or np.isnan(bb_pct[di]) or np.isnan(c[di]):
                continue
            # BB width at 120-day low = squeeze
            window = bb_width[di-119:di+1]
            valid_w = window[~np.isnan(window)]
            if len(valid_w) < 60:
                continue
            if bb_width[di] <= np.min(valid_w):
                # Breakout direction
                if bb_pct[di] > 0.8:
                    buy_days[si].add(di)
                elif bb_pct[di] < 0.2:
                    sell_days[si].add(di)
    return name, buy_days, sell_days


def strategy_7_calendar(NS, ND, dates, C, O, H, L, V):
    """S7: A-Share Calendar Effects — month-end, pre-holiday, turn-of-month.
    Free alpha from structural market patterns."""
    name = "S7_Calendar_Ashare"
    buy_days = [set() for _ in range(NS)]
    sell_days = [set() for _ in range(NS)]

    for di in range(MIN_TRAIN, ND):
        d = dates[di]
        # Turn-of-month effect: last 3 trading days + first 3 trading days
        is_turn_of_month = False
        # Last 3 days of month
        if di + 3 < ND:
            future_months = [dates[di+k].month for k in range(1, 4)]
            if d.month not in future_months:  # next 3 days are in a different month
                is_turn_of_month = True
        # First 3 days of month
        if di >= 3:
            past_months = [dates[di-k].month for k in range(1, 4)]
            if d.month not in past_months:
                is_turn_of_month = True

        # Pre-holiday effect (Chinese holidays: Jan 1, Spring Festival ~Jan/Feb, May 1, Oct 1)
        is_pre_holiday = False
        if di + 1 < ND and di + 2 < ND:
            gap = (dates[di+1] - d).days
            gap2 = (dates[di+2] - dates[di+1]).days
            if gap > 2 or gap2 > 2:  # upcoming holiday (market closed > 2 days)
                is_pre_holiday = True

        if is_turn_of_month or is_pre_holiday:
            for si in range(NS):
                if not np.isnan(C[si, di]) and C[si, di] > 0:
                    buy_days[si].add(di)
                    # Sell 3 days later
                    sell_di = min(di + 3, ND - 1)
                    sell_days[si].add(sell_di)
    return name, buy_days, sell_days


def strategy_8_panic_reversal(NS, ND, C, O, H, L, V):
    """S8: Panic Reversal — price down > 7% + volume > 3x average.
    Retail panic selling creates opportunity. Works because of 10% daily limit."""
    name = "S8_Panic_Reversal"
    buy_days = [set() for _ in range(NS)]
    sell_days = [set() for _ in range(NS)]

    for si in range(NS):
        c, v = C[si], V[si]
        valid = ~np.isnan(c)
        if np.sum(valid) < 60:
            continue
        # Daily returns
        ret = np.full(ND, np.nan)
        for i in range(1, ND):
            if not np.isnan(c[i]) and not np.isnan(c[i-1]) and c[i-1] > 0:
                ret[i] = (c[i] - c[i-1]) / c[i-1]

        vol_ma20 = np.full(ND, np.nan)
        for i in range(19, ND):
            vw = v[i-19:i+1]
            vv = vw[~np.isnan(vw)]
            if len(vv) >= 10:
                vol_ma20[i] = np.mean(vv)

        rsi = compute_rsi(c, 6)  # short-term RSI

        for di in range(MIN_TRAIN, ND):
            if np.isnan(ret[di]) or np.isnan(vol_ma20[di]) or np.isnan(rsi[di]):
                continue
            # Panic: price down > 7% + volume > 3x average + RSI oversold
            if ret[di] < -0.07 and v[di] > 3 * vol_ma20[di] and rsi[di] < 25:
                buy_days[si].add(di)
            # Sell after 3 days or RSI > 60
            elif not np.isnan(rsi[di]) and rsi[di] > 60:
                sell_days[si].add(di)
    return name, buy_days, sell_days


def strategy_9_kalman_vel(NS, ND, C, O, H, L, V):
    """S9: Kalman Velocity — momentum from optimal state estimation.
    Buy when velocity turns positive, sell when negative.
    Faster and less noisy than MA crossover."""
    name = "S9_Kalman_Vel"
    buy_days = [set() for _ in range(NS)]
    sell_days = [set() for _ in range(NS)]

    for si in range(NS):
        c = C[si]
        valid = ~np.isnan(c)
        if np.sum(valid) < 60:
            continue
        vel = compute_kalman_velocity(np.nan_to_num(c, nan=np.nanmean(c)))

        for di in range(MIN_TRAIN, ND):
            if np.isnan(vel[di]) or np.isnan(vel[di-1]):
                continue
            if np.isnan(c[di]):
                continue
            # Velocity crosses zero
            if vel[di] > 0 and vel[di-1] <= 0:
                buy_days[si].add(di)
            elif vel[di] < 0 and vel[di-1] >= 0:
                sell_days[si].add(di)
    return name, buy_days, sell_days


def strategy_10_multi_oversold(NS, ND, C, O, H, L, V):
    """S10: Multi-Factor Oversold — requires 3+ conditions:
    RSI < 30, BB < 20%, Stochastic < 20, VDP reversal, above 50MA.
    Multi-dimensional confirmation = higher confidence bottom."""
    name = "S10_Multi_Oversold"
    buy_days = [set() for _ in range(NS)]
    sell_days = [set() for _ in range(NS)]

    for si in range(NS):
        c, h, l, v = C[si], H[si], L[si], V[si]
        valid = ~np.isnan(c)
        if np.sum(valid) < 60:
            continue
        rsi14 = compute_rsi(c, 14)
        rsi6 = compute_rsi(c, 6)
        bb_up, bb_lo, bb_mid, bb_pct = compute_bollinger(c, 20, 2.0)
        vdp = compute_vdp(c, h, l, v)
        ema_vdp = np.full(ND, np.nan)
        ema_vdp[0] = vdp[0]
        alpha_ema = 2.0 / 11
        for i in range(1, ND):
            if not np.isnan(vdp[i]):
                ema_vdp[i] = alpha_ema * vdp[i] + (1 - alpha_ema) * (ema_vdp[i-1] if not np.isnan(ema_vdp[i-1]) else vdp[i])
        ma50 = np.full(ND, np.nan)
        for i in range(49, ND):
            window = c[i-49:i+1]
            valid_w = window[~np.isnan(window)]
            if len(valid_w) >= 30:
                ma50[i] = np.mean(valid_w)

        for di in range(MIN_TRAIN, ND):
            if np.isnan(c[di]):
                continue
            score = 0
            if not np.isnan(rsi14[di]) and rsi14[di] < 30:
                score += 1
            if not np.isnan(rsi6[di]) and rsi6[di] < 25:
                score += 1
            if not np.isnan(bb_pct[di]) and bb_pct[di] < 0.15:
                score += 1
            if not np.isnan(ema_vdp[di]) and ema_vdp[di] > 0:
                score += 1  # VDP turning positive after selling
            # Stochastic (simplified)
            if di >= 14:
                low14 = np.min(l[di-13:di+1][~np.isnan(l[di-13:di+1])])
                high14 = np.max(h[di-13:di+1][~np.isnan(h[di-13:di+1])])
                if high14 > low14 and (c[di] - low14) / (high14 - low14) < 0.2:
                    score += 1

            # Buy when score >= 3 AND above 50MA (trend filter)
            if score >= 3 and (np.isnan(ma50[di]) or c[di] > ma50[di]):
                buy_days[si].add(di)
            # Sell: RSI > 65 or BB% > 0.85
            if (not np.isnan(rsi14[di]) and rsi14[di] > 65) or \
               (not np.isnan(bb_pct[di]) and bb_pct[di] > 0.85):
                sell_days[si].add(di)
    return name, buy_days, sell_days


# ============================================================
# BACKTEST ENGINE
# ============================================================

def backtest_strategy(name, buy_days, sell_days, NS, ND, dates, C, O,
                      hold_days=5, sl_pct=5.0, tp_pct=30.0, max_stocks=2):
    """Backtest a single strategy. Holds up to max_stocks positions simultaneously."""
    # Build day-indexed reverse maps for fast lookup
    buy_by_day = [set() for _ in range(ND)]
    sell_by_day = [set() for _ in range(ND)]
    for si in range(NS):
        for di in buy_days[si]:
            buy_by_day[di].add(si)
        for di in sell_days[si]:
            sell_by_day[di].add(si)

    cash = float(CASH0)
    positions = []  # list of {'si', 'shares', 'entry', 'highest', 'ed'}
    trades = []
    pending = []  # list of ('open'/'close', si)

    for di in range(MIN_TRAIN, ND):
        # Execute pending orders
        new_pending = []
        for ptype, psi in pending:
            if ptype == 'close':
                for pos in positions:
                    if pos['si'] == psi:
                        p = O[psi, di]
                        if np.isnan(p) or p <= 0:
                            p = C[psi, di]
                        if not np.isnan(p) and p > 0:
                            pnl = (p - pos['entry']) / pos['entry'] * 100
                            cash += pos['shares'] * p * (1 - COMMISSION - STAMP_DUTY)
                            trades.append({
                                'pnl': pnl,
                                'days': (dates[di] - pos['ed']).days,
                                'di': di,
                                'reason': 'sell_signal'
                            })
                            positions.remove(pos)
                        break
            elif ptype == 'open' and len(positions) < max_stocks:
                p = O[psi, di]
                if np.isnan(p) or p <= 0:
                    p = C[psi, di-1] if di > 0 and not np.isnan(C[psi, di-1]) else np.nan
                if not np.isnan(p) and p > 0 and cash > 10000:
                    alloc = cash / max(1, max_stocks - len(positions))
                    shares = int(alloc / (1 + COMMISSION) / p)
                    if shares > 0:
                        cost = shares * p * (1 + COMMISSION)
                        if cost <= cash:
                            cash -= cost
                            positions.append({
                                'si': psi, 'shares': shares, 'entry': p,
                                'highest': p, 'ed': dates[di]
                            })
        pending = []

        # Check exits for current positions
        for pos in positions:
            si_p = pos['si']
            p = C[si_p, di]
            if np.isnan(p):
                continue
            if p > pos['highest']:
                pos['highest'] = p
            pnl = (p - pos['entry']) / pos['entry'] * 100
            hd = (dates[di] - pos['ed']).days

            if si_p in sell_by_day[di]:
                pending.append(('close', si_p))
            elif pnl < -sl_pct:
                pending.append(('close', si_p))
            elif pnl > tp_pct:
                pending.append(('close', si_p))
            elif hd >= hold_days:
                pending.append(('close', si_p))

        # Entry: if room for new positions, check buy signals
        n_open_slots = max_stocks - len(positions) - sum(1 for pt, _ in pending if pt == 'open')
        if n_open_slots > 0:
            # Collect all buy candidates for this day
            candidates = []
            for si in buy_by_day[di]:
                if not np.isnan(C[si, di]) and C[si, di] > 0:
                    # Already holding this stock? Skip
                    if any(pos['si'] == si for pos in positions):
                        continue
                    # Score by volume (prefer liquid stocks)
                    vol_score = V[si, di] if not np.isnan(V[si, di]) else 0
                    candidates.append((si, vol_score))
            # Sort by volume, take top N
            candidates.sort(key=lambda x: -x[1])
            for si, _ in candidates[:n_open_slots]:
                pending.append(('open', si))

    # Close remaining positions
    for pos in positions:
        p = C[pos['si'], ND-1]
        if not np.isnan(p) and p > 0:
            pnl = (p - pos['entry']) / pos['entry'] * 100
            cash += pos['shares'] * p * (1 - COMMISSION - STAMP_DUTY)
            trades.append({'pnl': pnl, 'days': 999, 'di': ND-1, 'reason': 'end'})

    if cash <= 0 or not trades:
        return None

    days_total = (dates[ND-1] - dates[MIN_TRAIN]).days
    yr = max(days_total / 365.25, 0.01)
    ann = ((cash / CASH0) ** (1/yr) - 1) * 100
    nw = sum(1 for t in trades if t['pnl'] > 0)
    wr = nw / max(len(trades), 1) * 100
    avg_w = np.mean([t['pnl'] for t in trades if t['pnl'] > 0]) if nw > 0 else 0
    avg_l = np.mean([abs(t['pnl']) for t in trades if t['pnl'] <= 0]) if nw < len(trades) else 0
    edge = (nw / max(len(trades), 1)) * avg_w - (1 - nw / max(len(trades), 1)) * avg_l
    max_dd = 0
    equity = CASH0
    peak = CASH0
    for t in sorted(trades, key=lambda x: x['di']):
        equity *= (1 + t['pnl'] / 100)
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak * 100
        if dd > max_dd:
            max_dd = dd

    return {
        'name': name,
        'ann': round(ann, 1),
        'n': len(trades),
        'wr': round(wr, 1),
        'avg_w': round(avg_w, 1),
        'avg_l': round(avg_l, 1),
        'edge': round(edge, 2),
        'max_dd': round(max_dd, 1),
        'tpy': round(len(trades) / yr, 1),
        'final': round(cash, 0),
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("=" * 70, flush=True)
    print("  Alpha V2 — Simple Strategies with Real Signal Quality", flush=True)
    print("=" * 70, flush=True)

    NS, ND, dates, C, O, H, L, V, syms, sym_set = load_all_data()

    # Generate all strategy signals
    print(f"\n[Signals] Computing strategy signals...", flush=True)
    t1 = time.time()
    strategies = []

    strategies.append(strategy_1_frama(NS, ND, C, O, H, L, V))
    print(f"  S1 FRAMA done ({time.time()-t1:.1f}s)", flush=True)

    strategies.append(strategy_2_vdp_reversal(NS, ND, C, O, H, L, V))
    print(f"  S2 VDP Reversal done ({time.time()-t1:.1f}s)", flush=True)

    strategies.append(strategy_3_oversold_trend(NS, ND, C, O, H, L, V))
    print(f"  S3 Oversold+Trend done ({time.time()-t1:.1f}s)", flush=True)

    strategies.append(strategy_4_ker_momentum(NS, ND, C, O, H, L, V))
    print(f"  S4 KER Momentum done ({time.time()-t1:.1f}s)", flush=True)

    strategies.append(strategy_5_volume_breakout(NS, ND, C, O, H, L, V))
    print(f"  S5 Vol Breakout done ({time.time()-t1:.1f}s)", flush=True)

    strategies.append(strategy_6_bb_squeeze(NS, ND, C, O, H, L, V))
    print(f"  S6 BB Squeeze done ({time.time()-t1:.1f}s)", flush=True)

    strategies.append(strategy_7_calendar(NS, ND, dates, C, O, H, L, V))
    print(f"  S7 Calendar done ({time.time()-t1:.1f}s)", flush=True)

    strategies.append(strategy_8_panic_reversal(NS, ND, C, O, H, L, V))
    print(f"  S8 Panic Reversal done ({time.time()-t1:.1f}s)", flush=True)

    strategies.append(strategy_9_kalman_vel(NS, ND, C, O, H, L, V))
    print(f"  S9 Kalman Vel done ({time.time()-t1:.1f}s)", flush=True)

    strategies.append(strategy_10_multi_oversold(NS, ND, C, O, H, L, V))
    print(f"  S10 Multi Oversold done ({time.time()-t1:.1f}s)", flush=True)

    print(f"  All signals computed ({time.time()-t1:.1f}s)", flush=True)

    # Backtest each strategy with different parameters
    print(f"\n[Backtest] Testing each strategy...", flush=True)
    results = []
    for name, buy_days, sell_days in strategies:
        for hm in [3, 5, 7, 10]:
            for sl in [4, 6, 8]:
                r = backtest_strategy(name, buy_days, sell_days, NS, ND, dates,
                                      C, O, hold_days=hm, sl_pct=sl, tp_pct=30, max_stocks=2)
                if r:
                    results.append(r)
        print(f"  {name} tested", flush=True)

    # Sort by annualized return
    results.sort(key=lambda x: -x['ann'])

    # Print results
    print(f"\n{'='*90}", flush=True)
    print(f"  TOP 30 RESULTS", flush=True)
    print(f"  {'Strategy':<25s} {'Ann':>7s} {'N':>4s} {'WR':>5s} "
          f"{'W':>6s} {'L':>6s} {'Edge':>6s} {'DD':>5s} {'TPY':>5s}", flush=True)
    print(f"  {'-'*85}", flush=True)
    for r in results[:30]:
        print(f"  {r['name']:<25s} {r['ann']:+7.1f}% {r['n']:4d} "
              f"{r['wr']:5.1f}% {r['avg_w']:+6.1f}% {r['avg_l']:6.1f}% "
              f"{r['edge']:+6.2f}% {r['max_dd']:5.1f}% {r['tpy']:5.1f}", flush=True)

    # Best per strategy
    print(f"\n  Best per strategy:", flush=True)
    best_per = {}
    for r in results:
        s = r['name']
        if s not in best_per or r['ann'] > best_per[s]['ann']:
            best_per[s] = r
    for r in sorted(best_per.values(), key=lambda x: -x['ann']):
        print(f"    {r['name']:<25s} → {r['ann']:+.1f}% (N={r['n']}, WR={r['wr']:.0f}%, "
              f"Edge={r['edge']:+.2f}%, DD={r['max_dd']:.1f}%)", flush=True)

    print(f"\n{'='*70}", flush=True)
    above_0 = sum(1 for r in best_per.values() if r['ann'] > 0)
    above_20 = sum(1 for r in best_per.values() if r['ann'] > 20)
    above_50 = sum(1 for r in best_per.values() if r['ann'] > 50)
    print(f"  Strategies > 0%: {above_0}/{len(best_per)}", flush=True)
    print(f"  Strategies > 20%: {above_20}/{len(best_per)}", flush=True)
    print(f"  Strategies > 50%: {above_50}/{len(best_per)}", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"\nDone! Total time: {time.time()-t1:.1f}s", flush=True)
