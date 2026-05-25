"""
Alpha Futures V111 -- MATHEMATICAL/STATISTICAL INDICATORS
=========================================================
Current best: ROC(5) cross +81.9%, 6/6 WF.

V111 FOCUS: Go beyond standard TA-Lib. Build indicators from mathematical principles.
Test 12 custom mathematical/statistical indicators:

A) HURST EXPONENT (trend persistence via R/S analysis)
B) FRACTAL DIMENSION (complexity via box-counting)
C) SHANNON ENTROPY (information content of return signs)
D) LINEAR REGRESSION STRENGTH (R^2 of trend)
E) Z-SCORE OF RETURNS (statistical extreme)
F) AUTOCORRELATION (serial dependence at lag 1)
G) ROLLING SKEWNESS (asymmetry of returns)
H) KURTOSIS FILTER (tail risk regime)
I) PRICE ACCELERATION (second derivative)
J) PARTIAL AUTOCORRELATION (PACF at lag 1)
K) DOUBLE MOMENTUM (ROC cross)
L) ROLLING BETA TO MARKET (low-beta strength)

ALL signals at close di, entry at O[si, di+1] (NEXT-OPEN execution).
Exit at C[si, di+hold] (close price hold days later).
Walk-forward by year (2020-2025).
"""
import sys, os, time, warnings
import numpy as np
from scipy import stats as sp_stats
from numpy.lib.stride_tricks import as_strided
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alpha_v2 import load_all_data, MIN_TRAIN, CASH0

# ============================================================
# CONSTANTS
# ============================================================
MULT = {'agfi': 15, 'alfi': 5, 'aufi': 1000, 'bufi': 10, 'cufi': 5, 'fufi': 10,
        'rbfi': 10, 'znfi': 5, 'nifi': 1, 'hcfi': 10, 'spfi': 10, 'ssfi': 5,
        'sffi': 5, 'smfi': 5, 'pbfi': 5, 'snfi': 1, 'rufi': 10, 'wrffi': 10,
        'afi': 10, 'bfi': 10, 'bbfi': 500, 'cffi': 5, 'cfi': 10, 'csfi': 10,
        'ebfi': 5, 'egfi': 10, 'fbfi': 500, 'ifi': 100, 'jfi': 100, 'jmfi': 60,
        'lfi': 5, 'mfi': 10, 'pgfi': 20, 'ppfi': 5, 'vfi': 5, 'yfi': 10,
        'pfi': 10, 'jdfi': 5, 'lhfi': 16, 'pkfi': 5, 'rrfi': 20, 'lrfi': 20,
        'jrfi': 20, 'pmfi': 20, 'whfi': 20, 'rsfi': 20, 'cjfi': 10, 'mafi': 10,
        'apfi': 10, 'cyfi': 5, 'fgfi': 20, 'oifi': 10, 'pfifi': 5, 'rmfi': 10,
        'srfi': 10, 'tafi': 5, 'safi': 20, 'urfi': 20, 'scfi': 1000, 'lufi': 10,
        'bcfi': 5, 'nrfi': 1, 'lgfi': 20, 'brfi': 5, 'lcfi': 1, 'sifi': 5,
        'ni': 1, 'tai': 5}
DEF_MULT = 10
COMM = 0.0003


def annual_return(final, initial, n_days):
    if final <= 0 or initial <= 0 or n_days <= 0:
        return -100.0
    yrs = n_days / 252
    return (final / initial) ** (1.0 / yrs) * 100 - 100


# ============================================================
# MATHEMATICAL INDICATOR COMPUTATION FUNCTIONS
# ============================================================

def compute_hurst_rs(series, window=50):
    """Rolling Hurst exponent via R/S analysis.
    H > 0.5 = trending (persistent), H < 0.5 = mean-reverting."""
    n = len(series)
    hurst = np.full(n, np.nan)
    for i in range(window, n):
        x = series[i - window:i]
        valid = ~np.isnan(x)
        xv = x[valid]
        if len(xv) < window:
            continue
        # R/S analysis on sub-periods
        mean_x = np.mean(xv)
        cumdev = np.cumsum(xv - mean_x)
        R = np.max(cumdev) - np.min(cumdev)
        S = np.std(xv, ddof=1)
        if S <= 0 or R <= 0:
            continue
        rs = R / S
        # Approximate Hurst: H ~ log(R/S) / log(n/2)
        hurst[i] = np.log(rs) / np.log(len(xv) / 2.0)
    return hurst


def compute_fractal_dimension(prices, window=30):
    """Estimate fractal dimension via simplified box-counting.
    Low FD = smooth trend, high FD = noisy/choppy."""
    n = len(prices)
    fd = np.full(n, np.nan)
    for i in range(window, n):
        x = prices[i - window:i]
        valid = ~np.isnan(x)
        xv = x[valid]
        if len(xv) < 10:
            continue
        # Normalized path length method
        xmin, xmax = np.min(xv), np.max(xv)
        if xmax == xmin:
            continue
        norm = (xv - xmin) / (xmax - xmin)
        # Total path length
        path_len = np.sum(np.abs(np.diff(norm)))
        # Straight-line distance = 1.0
        if path_len <= 0:
            continue
        # FD approximation: D = 1 + log(path_len) / log(N-1)
        fd[i] = 1.0 + np.log(path_len) / np.log(len(xv) - 1)
    return fd


def compute_shannon_entropy(returns, window=20):
    """Rolling Shannon entropy of return sign sequence (+/-).
    Low entropy = predictable pattern (trending)."""
    n = len(returns)
    entropy = np.full(n, np.nan)
    for i in range(window, n):
        r = returns[i - window:i]
        valid = ~np.isnan(r)
        rv = r[valid]
        if len(rv) < window:
            continue
        signs = np.sign(rv)
        n_pos = np.sum(signs > 0)
        n_neg = np.sum(signs < 0)
        n_zero = np.sum(signs == 0)
        total = n_pos + n_neg + n_zero
        if total == 0:
            continue
        ent = 0.0
        for count in [n_pos, n_neg, n_zero]:
            if count > 0:
                p = count / total
                ent -= p * np.log2(p)
        entropy[i] = ent
    return entropy


def compute_rsq_trend(prices, window=20):
    """R-squared and slope from linear regression on prices."""
    n = len(prices)
    rsq = np.full(n, np.nan)
    slope = np.full(n, np.nan)
    for i in range(window, n):
        y = prices[i - window:i]
        valid = ~np.isnan(y)
        yv = y[valid]
        if len(yv) < window:
            continue
        x = np.arange(len(yv), dtype=np.float64)
        slope_v, intercept, r_value, p_value, std_err = sp_stats.linregress(x, yv)
        rsq[i] = r_value ** 2
        slope[i] = slope_v
    return rsq, slope


def compute_zscore_returns(returns, window=20):
    """Z-score of today's return relative to past window distribution."""
    n = len(returns)
    zscore = np.full(n, np.nan)
    for i in range(window + 1, n):
        hist = returns[i - window:i]
        valid = ~np.isnan(hist)
        hv = hist[valid]
        if len(hv) < window:
            continue
        mu = np.mean(hv)
        sigma = np.std(hv, ddof=1)
        if sigma <= 0 or np.isnan(returns[i]):
            continue
        zscore[i] = (returns[i] - mu) / sigma
    return zscore


def compute_autocorrelation(returns, window=20, lag=1):
    """Rolling autocorrelation of returns at given lag."""
    n = len(returns)
    autocorr = np.full(n, np.nan)
    for i in range(window + lag, n):
        r = returns[i - window:i]
        valid = ~np.isnan(r)
        rv = r[valid]
        if len(rv) < window:
            continue
        mean_r = np.mean(rv)
        demeaned = rv - mean_r
        var = np.sum(demeaned ** 2)
        if var <= 0:
            continue
        cov = np.sum(demeaned[lag:] * demeaned[:-lag])
        autocorr[i] = cov / var
    return autocorr


def compute_rolling_skewness(returns, window=20):
    """Rolling skewness of returns."""
    n = len(returns)
    skew = np.full(n, np.nan)
    for i in range(window, n):
        r = returns[i - window:i]
        valid = ~np.isnan(r)
        rv = r[valid]
        if len(rv) < window:
            continue
        skew[i] = sp_stats.skew(rv, bias=False)
    return skew


def compute_rolling_kurtosis(returns, window=20):
    """Rolling kurtosis (excess) of returns."""
    n = len(returns)
    kurt = np.full(n, np.nan)
    for i in range(window, n):
        r = returns[i - window:i]
        valid = ~np.isnan(r)
        rv = r[valid]
        if len(rv) < window:
            continue
        kurt[i] = sp_stats.kurtosis(rv, bias=False)  # excess kurtosis
    return kurt


def compute_pacf_lag1(returns, window=20):
    """Simplified PACF at lag 1 via Durbin-Levinson / partial autocorrelation.
    PACF(1) = autocorrelation at lag 1 (for lag 1, PACF = ACF)."""
    n = len(returns)
    pacf = np.full(n, np.nan)
    for i in range(window, n):
        r = returns[i - window:i]
        valid = ~np.isnan(r)
        rv = r[valid]
        if len(rv) < window:
            continue
        mean_r = np.mean(rv)
        demeaned = rv - mean_r
        var = np.sum(demeaned ** 2)
        if var <= 0:
            continue
        # For lag 1, PACF(1) = ACF(1)
        cov = np.sum(demeaned[1:] * demeaned[:-1])
        pacf[i] = cov / var
    return pacf


def main():
    print("=" * 200)
    print("Alpha Futures V111 -- MATHEMATICAL/STATISTICAL INDICATORS")
    print("=" * 200)
    print("\n  12 mathematical indicators: Hurst, Fractal Dim, Shannon Entropy, R^2,")
    print("  Z-score, Autocorrelation, Skewness, Kurtosis, Acceleration, PACF,")
    print("  Double Momentum, Rolling Beta.")
    print("  ALL signals at close di, entry at O[si, di+1] (NEXT DAY OPEN)")

    # -- Load data -------------------------------------------------
    print("\n[Data] Loading...", flush=True)
    t_start = time.time()
    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)
    years = sorted(set(d.year for d in dates))
    print(f"  {NS} commodities, {ND} days, years: {years}")

    # ================================================================
    # PRECOMPUTE ALL MATHEMATICAL INDICATORS
    # ================================================================
    print("\n[Math] Computing all mathematical indicators...", flush=True)
    t0 = time.time()

    # Allocate arrays
    ROC5 = np.full((NS, ND), np.nan)
    ROC3 = np.full((NS, ND), np.nan)
    ROC10 = np.full((NS, ND), np.nan)
    SMA20 = np.full((NS, ND), np.nan)
    ATR = np.full((NS, ND), np.nan)
    RETURNS = np.full((NS, ND), np.nan)
    HURST = np.full((NS, ND), np.nan)
    FRACDIM = np.full((NS, ND), np.nan)
    ENTROPY = np.full((NS, ND), np.nan)
    RSQ = np.full((NS, ND), np.nan)
    SLOPE = np.full((NS, ND), np.nan)
    ZSCORE = np.full((NS, ND), np.nan)
    AUTOCORR = np.full((NS, ND), np.nan)
    SKEWNESS = np.full((NS, ND), np.nan)
    KURTOSIS = np.full((NS, ND), np.nan)
    VELOCITY = np.full((NS, ND), np.nan)
    ACCELERATION = np.full((NS, ND), np.nan)
    PACF1 = np.full((NS, ND), np.nan)
    BETA = np.full((NS, ND), np.nan)
    MARKET_RET = np.full(ND, np.nan)
    RANGE_RATIO = np.full((NS, ND), np.nan)  # range / ATR

    # Market average return (cross-commodity)
    for di in range(1, ND):
        rets = []
        for si in range(NS):
            if not np.isnan(C[si, di]) and not np.isnan(C[si, di - 1]) and C[si, di - 1] > 0:
                rets.append((C[si, di] - C[si, di - 1]) / C[si, di - 1])
        if rets:
            MARKET_RET[di] = np.mean(rets)

    for si in range(NS):
        c = C[si]
        h = H[si]
        l = L[si]
        valid = ~np.isnan(c)
        if np.sum(valid) < 100:
            continue

        # Returns
        ret = np.full(ND, np.nan)
        for di in range(1, ND):
            if not np.isnan(c[di]) and not np.isnan(c[di - 1]) and c[di - 1] > 0:
                ret[di] = (c[di] - c[di - 1]) / c[di - 1]
        RETURNS[si] = ret

        # ROC
        for di in range(5, ND):
            if not np.isnan(c[di]) and not np.isnan(c[di - 5]) and c[di - 5] > 0:
                ROC5[si, di] = (c[di] - c[di - 5]) / c[di - 5]
        for di in range(3, ND):
            if not np.isnan(c[di]) and not np.isnan(c[di - 3]) and c[di - 3] > 0:
                ROC3[si, di] = (c[di] - c[di - 3]) / c[di - 3]
        for di in range(10, ND):
            if not np.isnan(c[di]) and not np.isnan(c[di - 10]) and c[di - 10] > 0:
                ROC10[si, di] = (c[di] - c[di - 10]) / c[di - 10]

        # SMA20
        for di in range(20, ND):
            cw = c[di - 20:di + 1]
            vw = cw[~np.isnan(cw)]
            if len(vw) >= 10:
                SMA20[si, di] = np.mean(vw)

        # ATR (14-day)
        tr = np.full(ND, np.nan)
        for di in range(1, ND):
            if np.isnan(h[di]) or np.isnan(l[di]) or np.isnan(c[di]):
                continue
            hl = h[di] - l[di]
            hc = abs(h[di] - c[di - 1]) if not np.isnan(c[di - 1]) else 0
            lc = abs(l[di] - c[di - 1]) if not np.isnan(c[di - 1]) else 0
            tr[di] = max(hl, hc, lc)
        for di in range(14, ND):
            tw = tr[di - 13:di + 1]
            vw = tw[~np.isnan(tw)]
            if len(vw) >= 7:
                ATR[si, di] = np.mean(vw)

        # Range ratio (today's range / ATR)
        for di in range(14, ND):
            if not np.isnan(h[di]) and not np.isnan(l[di]) and not np.isnan(ATR[si, di]) and ATR[si, di] > 0:
                RANGE_RATIO[si, di] = (h[di] - l[di]) / ATR[si, di]

        # A) Hurst Exponent (50-day window)
        HURST[si] = compute_hurst_rs(c, window=50)

        # B) Fractal Dimension (30-day window)
        FRACDIM[si] = compute_fractal_dimension(c, window=30)

        # C) Shannon Entropy (20-day window)
        ENTROPY[si] = compute_shannon_entropy(ret, window=20)

        # D) Linear Regression R^2 (20-day window)
        r2, sl = compute_rsq_trend(c, window=20)
        RSQ[si] = r2
        SLOPE[si] = sl

        # E) Z-score of returns (20-day window)
        ZSCORE[si] = compute_zscore_returns(ret, window=20)

        # F) Autocorrelation (20-day window, lag 1)
        AUTOCORR[si] = compute_autocorrelation(ret, window=20, lag=1)

        # G) Rolling Skewness (20-day)
        SKEWNESS[si] = compute_rolling_skewness(ret, window=20)

        # H) Rolling Kurtosis (20-day)
        KURTOSIS[si] = compute_rolling_kurtosis(ret, window=20)

        # I) Price Acceleration (velocity=ROC5, acceleration=velocity change)
        VELOCITY[si] = ROC5[si]
        for di in range(10, ND):
            if not np.isnan(ROC5[si, di]) and not np.isnan(ROC5[si, di - 5]):
                ACCELERATION[si, di] = ROC5[si, di] - ROC5[si, di - 5]

        # J) PACF at lag 1 (20-day window)
        PACF1[si] = compute_pacf_lag1(ret, window=20)

        # L) Rolling Beta to market (20-day)
        for di in range(20, ND):
            mr = MARKET_RET[di - 19:di + 1]
            sr = ret[di - 19:di + 1]
            valid_mask = ~np.isnan(mr) & ~np.isnan(sr)
            mv = mr[valid_mask]
            sv = sr[valid_mask]
            if len(mv) < 10:
                continue
            var_m = np.var(mv, ddof=1)
            if var_m <= 0:
                continue
            cov_ms = np.cov(mv, sv, ddof=1)[0, 1]
            BETA[si, di] = cov_ms / var_m

        if (si + 1) % 5 == 0 or si == NS - 1:
            print(f"  ... {si+1}/{NS} commodities done ({time.time()-t0:.1f}s)", flush=True)

    print(f"  All mathematical indicators computed ({time.time()-t0:.1f}s)")

    # ================================================================
    # SIGNAL GENERATION
    # ================================================================
    print("\n[Signals] Computing all 12 signal types...", flush=True)

    sig_labels = {
        'A': 'Hurst>0.55+ROC5>0',
        'B': 'FD<1.3+C>SMA20',
        'C': 'Entropy<0.85+ROC5>0',
        'D': 'Rsq>0.6+Slope>0',
        'E': 'Zscore>2.0',
        'F': 'Autocorr>0.2+ROC5>0',
        'G': 'Skew>0.5+ROC5>0',
        'H': 'Kurt>5.0+Range>1.5ATR',
        'I': 'Vel>0+Accel>0',
        'J': 'PACF>0.3+ROC5>0',
        'K': 'ROC3_X_ROC10',
        'L': 'Beta<0.5+ROC5>0-MktDown',
    }

    # Build signal arrays
    sig_A = np.zeros((NS, ND), dtype=bool)
    sig_B = np.zeros((NS, ND), dtype=bool)
    sig_C = np.zeros((NS, ND), dtype=bool)
    sig_D = np.zeros((NS, ND), dtype=bool)
    sig_E = np.zeros((NS, ND), dtype=bool)
    sig_F = np.zeros((NS, ND), dtype=bool)
    sig_G = np.zeros((NS, ND), dtype=bool)
    sig_H = np.zeros((NS, ND), dtype=bool)
    sig_I = np.zeros((NS, ND), dtype=bool)
    sig_J = np.zeros((NS, ND), dtype=bool)
    sig_K = np.zeros((NS, ND), dtype=bool)
    sig_L = np.zeros((NS, ND), dtype=bool)

    # Score arrays for ranking
    score_arr = np.zeros((NS, ND), dtype=np.float64)

    for si in range(NS):
        for di in range(MIN_TRAIN, ND):
            if np.isnan(C[si, di]):
                continue

            roc5 = ROC5[si, di]
            roc3 = ROC3[si, di]
            roc10 = ROC10[si, di]

            # A) HURST: H > 0.55 AND ROC5 > 0
            h_v = HURST[si, di]
            if not np.isnan(h_v) and not np.isnan(roc5):
                if h_v > 0.55 and roc5 > 0:
                    sig_A[si, di] = True

            # B) FRACTAL DIMENSION: FD < 1.3 AND C > SMA20
            fd_v = FRACDIM[si, di]
            sma20_v = SMA20[si, di]
            if not np.isnan(fd_v) and not np.isnan(sma20_v):
                if fd_v < 1.3 and C[si, di] > sma20_v:
                    sig_B[si, di] = True

            # C) SHANNON ENTROPY: Entropy < 0.85 AND ROC5 > 0
            ent_v = ENTROPY[si, di]
            if not np.isnan(ent_v) and not np.isnan(roc5):
                if ent_v < 0.85 and roc5 > 0:
                    sig_C[si, di] = True

            # D) R^2 TREND: R^2 > 0.6 AND slope > 0
            rsq_v = RSQ[si, di]
            sl_v = SLOPE[si, di]
            if not np.isnan(rsq_v) and not np.isnan(sl_v):
                if rsq_v > 0.6 and sl_v > 0:
                    sig_D[si, di] = True

            # E) Z-SCORE: z > 2.0
            z_v = ZSCORE[si, di]
            if not np.isnan(z_v):
                if z_v > 2.0:
                    sig_E[si, di] = True

            # F) AUTOCORRELATION: autocorr > 0.2 AND ROC5 > 0
            ac_v = AUTOCORR[si, di]
            if not np.isnan(ac_v) and not np.isnan(roc5):
                if ac_v > 0.2 and roc5 > 0:
                    sig_F[si, di] = True

            # G) SKEWNESS: skew > 0.5 AND ROC5 > 0
            sk_v = SKEWNESS[si, di]
            if not np.isnan(sk_v) and not np.isnan(roc5):
                if sk_v > 0.5 and roc5 > 0:
                    sig_G[si, di] = True

            # H) KURTOSIS: kurtosis > 5.0 AND range > 1.5*ATR
            ku_v = KURTOSIS[si, di]
            rr_v = RANGE_RATIO[si, di]
            if not np.isnan(ku_v) and not np.isnan(rr_v):
                if ku_v > 5.0 and rr_v > 1.5:
                    sig_H[si, di] = True

            # I) ACCELERATION: velocity > 0 AND acceleration > 0
            vel_v = VELOCITY[si, di]
            acc_v = ACCELERATION[si, di]
            if not np.isnan(vel_v) and not np.isnan(acc_v):
                if vel_v > 0 and acc_v > 0:
                    sig_I[si, di] = True

            # J) PACF: PACF > 0.3 AND ROC5 > 0
            pacf_v = PACF1[si, di]
            if not np.isnan(pacf_v) and not np.isnan(roc5):
                if pacf_v > 0.3 and roc5 > 0:
                    sig_J[si, di] = True

            # K) DOUBLE MOMENTUM: ROC3 crosses above ROC10
            if di > 0 and not np.isnan(roc3) and not np.isnan(roc10):
                roc3_prev = ROC3[si, di - 1]
                roc10_prev = ROC10[si, di - 1]
                if not np.isnan(roc3_prev) and not np.isnan(roc10_prev):
                    if roc3 > roc10 and roc3_prev <= roc10_prev:
                        sig_K[si, di] = True

            # L) ROLLING BETA: beta < 0.5 AND ROC5 > 0 AND market ROC5 < 0
            beta_v = BETA[si, di]
            mkt_roc5 = np.nan
            if di >= 5:
                # Market ROC5: average market return over last 5 days
                mr5 = MARKET_RET[di - 4:di + 1]
                valid_mr = mr5[~np.isnan(mr5)]
                if len(valid_mr) > 0:
                    mkt_roc5 = np.sum(valid_mr)  # cumulative return
            if not np.isnan(beta_v) and not np.isnan(roc5) and not np.isnan(mkt_roc5):
                if beta_v < 0.5 and roc5 > 0 and mkt_roc5 < 0:
                    sig_L[si, di] = True

            # Score: use ROC5 as primary score
            score_arr[si, di] = roc5 if not np.isnan(roc5) else 0.0

    sig_map = {
        'A': sig_A, 'B': sig_B, 'C': sig_C, 'D': sig_D,
        'E': sig_E, 'F': sig_F, 'G': sig_G, 'H': sig_H,
        'I': sig_I, 'J': sig_J, 'K': sig_K, 'L': sig_L,
    }

    for key in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']:
        n_sig = int(np.sum(sig_map[key]))
        print(f"  {key}) {sig_labels[key]}: {n_sig} signals")

    print(f"\n  All signals computed ({time.time()-t_start:.1f}s total)")

    # ================================================================
    # BACKTEST ENGINE
    # ================================================================
    def run_backtest(config, wf_test_year=None):
        sig_type = config['signal']
        hold_days = config['hold_days']
        top_n = config['top_n']
        comm = config.get('comm', COMM)

        sig_arr = sig_map[sig_type]

        # Date boundaries
        if wf_test_year is not None:
            test_start_di = None
            test_end_di = None
            for di in range(ND):
                if dates[di].year == wf_test_year and test_start_di is None:
                    test_start_di = di
                if dates[di].year == wf_test_year + 1 and test_end_di is None:
                    test_end_di = di
            if test_start_di is None:
                return None
            if test_end_di is None:
                test_end_di = ND
            start_di = MIN_TRAIN
            end_di = test_end_di
        else:
            test_start_di = MIN_TRAIN
            start_di = MIN_TRAIN
            end_di = ND
            test_end_di = ND

        if end_di < start_di + hold_days + 2:
            return None

        cash = float(CASH0)
        positions = []
        trades = []

        for di in range(start_di, end_di - 1):
            # Reset cash at test window start (WF mode)
            if wf_test_year is not None and di == test_start_di:
                cash = float(CASH0)
                positions = []

            # -- Close positions -----------------------------------------
            closed = []
            for pos in positions:
                days_held = di - pos['entry_di']
                if days_held >= pos['hold_days']:
                    exit_price = C[pos['si'], di]
                    if np.isnan(exit_price) or exit_price <= 0:
                        exit_price = pos['entry_price']
                    mult = MULT.get(pos['sym'], DEF_MULT)
                    mkt_val = exit_price * mult * abs(pos['lots'])
                    cash += mkt_val - mkt_val * comm
                    pnl = (exit_price - pos['entry_price']) * mult * pos['lots']
                    invested = pos['entry_price'] * mult * abs(pos['lots'])
                    pnl_pct = pnl / invested * 100 if invested > 0 else 0
                    trades.append({
                        'pnl_pct': pnl_pct,
                        'entry_di': pos['entry_di'],
                        'exit_di': di,
                        'year': dates[di].year if di < ND else dates[-1].year,
                        'sym': pos.get('sym', ''),
                        'days_held': days_held,
                    })
                    closed.append(pos)

            for pos in closed:
                positions.remove(pos)

            if len(positions) >= top_n:
                continue

            # -- Generate signals at day di --------------------------------
            entry_di = di + 1
            if entry_di >= end_di:
                continue

            candidates = []
            for si in range(NS):
                if not sig_arr[si, di]:
                    continue
                if any(p['si'] == si for p in positions):
                    continue
                ep = O[si, entry_di]
                if np.isnan(ep) or ep <= 0:
                    continue
                sc = score_arr[si, di] if not np.isnan(score_arr[si, di]) else 0
                candidates.append((sc, {
                    'si': si, 'sym': syms[si], 'entry_price': ep,
                }))

            if not candidates:
                continue

            # Sort by score descending
            candidates.sort(key=lambda x: -x[0])

            # Open positions
            n_slots = top_n - len(positions)
            for sc_val, info in candidates[:max(0, n_slots)]:
                si = info['si']
                sym = info['sym']
                price = info['entry_price']
                mult = MULT.get(sym, DEF_MULT)
                contracts = max(1, int(cash / (price * mult)))
                cost_in = price * mult * contracts * (1 + comm)
                if cost_in > cash:
                    contracts = int(cash * 0.9 / (price * mult * (1 + comm)))
                    cost_in = price * mult * contracts * (1 + comm) if contracts > 0 else 0
                if contracts <= 0 or cost_in <= 0 or cost_in > cash:
                    continue

                cash -= cost_in
                positions.append({
                    'si': si, 'entry_price': price, 'entry_di': entry_di,
                    'lots': contracts, 'dir': 1, 'sym': sym,
                    'hold_days': hold_days,
                })

        # Close remaining positions at end
        for pos in positions:
            ae = end_di - 1 if end_di < ND else ND - 1
            exit_price = C[pos['si'], ae]
            if np.isnan(exit_price) or exit_price <= 0:
                exit_price = pos['entry_price']
            mult = MULT.get(pos['sym'], DEF_MULT)
            mkt_val = exit_price * mult * abs(pos['lots'])
            cash += mkt_val - mkt_val * comm
            pnl = (exit_price - pos['entry_price']) * mult * pos['lots']
            invested = pos['entry_price'] * mult * abs(pos['lots'])
            pnl_pct = pnl / invested * 100 if invested > 0 else 0
            trades.append({
                'pnl_pct': pnl_pct,
                'entry_di': pos['entry_di'],
                'exit_di': ae,
                'year': dates[ae].year if ae < ND else dates[-1].year,
                'sym': pos.get('sym', ''),
                'days_held': ae - pos['entry_di'],
            })

        # Results
        wf_mode = wf_test_year is not None
        n_days_test = (test_end_di - test_start_di) if wf_mode else (end_di - start_di)
        ann = annual_return(cash, CASH0, n_days_test)
        wr = np.mean([1 if t['pnl_pct'] > 0 else 0 for t in trades]) * 100 if trades else 0
        n_trades = len(trades)
        avg_pnl = np.mean([t['pnl_pct'] for t in trades]) if trades else 0
        avg_hold = np.mean([t['days_held'] for t in trades]) if trades else 0

        # Max drawdown from equity curve
        eq = float(CASH0)
        peak = eq
        mdd = 0.0
        for t in trades:
            eq *= (1 + t['pnl_pct'] / 100)
            if eq > peak:
                peak = eq
            dd = (eq - peak) / peak * 100
            if dd < mdd:
                mdd = dd

        freq_per_yr = n_trades / (n_days_test / 252) if n_days_test > 0 else 0

        return {
            'ann': ann, 'wr': wr, 'n': n_trades, 'avg_pnl': avg_pnl,
            'final_cash': cash, 'n_days': n_days_test, 'mdd': mdd,
            'avg_hold': avg_hold, 'freq': freq_per_yr,
        }

    # ================================================================
    # BUILD CONFIGURATIONS
    # ================================================================
    print("\n[Sweep] Building configurations...", flush=True)
    configs = []
    cid = 0

    # Hold days per signal type (as specified)
    hold_map = {
        'A': [5, 10], 'B': [5, 10], 'C': [5, 10],
        'D': [5, 10, 20], 'E': [3, 5, 10],
        'F': [5, 10], 'G': [5, 10], 'H': [5, 10],
        'I': [5, 10], 'J': [5, 10], 'K': [5, 10],
        'L': [5, 10],
    }
    top_n_map = {
        'A': [1, 3], 'B': [1, 3], 'C': [1, 3],
        'D': [1, 3], 'E': [1, 3], 'F': [1, 3],
        'G': [1, 3], 'H': [1, 3], 'I': [1, 3],
        'J': [1, 3], 'K': [1, 3], 'L': [1, 3],
    }

    for sig_key in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']:
        for hd in hold_map[sig_key]:
            for tn in top_n_map[sig_key]:
                cid += 1
                configs.append({
                    'id': cid, 'signal': sig_key,
                    'hold_days': hd, 'top_n': tn, 'comm': COMM,
                    'label': f"{sig_key}_{sig_labels[sig_key]}_H{hd}_TN{tn}",
                })

    print(f"  Total configs: {len(configs)}")

    # ================================================================
    # RUN FULL-PERIOD BACKTEST
    # ================================================================
    print("\n[Backtest] Running full-period sweep...", flush=True)
    results = []
    for i, cfg in enumerate(configs):
        r = run_backtest(cfg)
        if r:
            r['config'] = cfg
            r['label'] = cfg['label']
            results.append(r)
        if (i + 1) % 10 == 0 or i == len(configs) - 1:
            print(f"  ... {i+1}/{len(configs)} done ({time.time()-t_start:.0f}s)", flush=True)

    results.sort(key=lambda x: -x['ann'])

    # ================================================================
    # FULL-PERIOD RESULTS
    # ================================================================
    print(f"\n{'=' * 200}")
    print("  FULL-PERIOD RESULTS (All configs) -- NEXT-OPEN EXECUTION, MATHEMATICAL INDICATORS")
    print(f"{'=' * 200}")
    print(f"  {'#':>3} | {'Label':<42} | {'Ann':>9} | {'WR':>6} | {'N':>5} | {'AvgPnL':>7} | {'MDD':>7} | {'AvgHold':>7} | {'Freq/Yr':>7} | {'Final':>14}")
    print("-" * 180)
    for i, r in enumerate(results):
        print(f"  {i+1:>3} | {r['label']:<42} | {r['ann']:>+8.1f}% | {r['wr']:>5.1f}% | {r['n']:>5} | {r['avg_pnl']:>+6.3f}% | {r['mdd']:>6.1f}% | {r['avg_hold']:>6.1f}d | {r['freq']:>6.1f}/yr | {r['final_cash']:>13,.0f}")

    # ================================================================
    # BEST PER SIGNAL TYPE
    # ================================================================
    sig_names = {
        'A': 'A) HURST EXPONENT (trend persistence)',
        'B': 'B) FRACTAL DIMENSION (complexity)',
        'C': 'C) SHANNON ENTROPY (info content)',
        'D': 'D) R^2 TREND STRENGTH (linear regression)',
        'E': 'E) Z-SCORE EXTREME RETURNS',
        'F': 'F) AUTOCORRELATION (serial dependence)',
        'G': 'G) ROLLING SKEWNESS (asymmetry)',
        'H': 'H) KURTOSIS FILTER (tail risk regime)',
        'I': 'I) PRICE ACCELERATION (2nd derivative)',
        'J': 'J) PARTIAL AUTOCORRELATION (AR structure)',
        'K': 'K) DOUBLE MOMENTUM (ROC crossover)',
        'L': 'L) ROLLING BETA (low-beta strength)',
    }

    print(f"\n{'=' * 200}")
    print("  BEST PER SIGNAL TYPE (Full Period)")
    print(f"{'=' * 200}")
    print(f"  {'Signal':<48} | {'Ann':>9} | {'WR':>6} | {'N':>5} | {'AvgPnL':>7} | {'MDD':>7} | {'AvgHold':>7} | {'Freq/Yr':>7} | Best Config")
    print("-" * 200)

    best_per_sig = {}
    for r in results:
        key = r['config']['signal']
        if key not in best_per_sig:
            best_per_sig[key] = r

    for sig_key in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']:
        if sig_key in best_per_sig:
            b = best_per_sig[sig_key]
            print(f"  {sig_names.get(sig_key, sig_key):<48} | {b['ann']:>+8.1f}% | {b['wr']:>5.1f}% | {b['n']:>5} | {b['avg_pnl']:>+6.3f}% | {b['mdd']:>6.1f}% | {b['avg_hold']:>6.1f}d | {b['freq']:>6.1f}/yr | {b['label']}")

    # ================================================================
    # SIGNAL TYPE SUMMARY
    # ================================================================
    print(f"\n{'=' * 200}")
    print("  SIGNAL TYPE SUMMARY (Average of all configs per type)")
    print(f"{'=' * 200}")
    print(f"  {'Signal':<48} | {'Avg Ann':>9} | {'Avg WR':>7} | {'Avg N':>7} | {'Avg PnL':>8} | {'Avg MDD':>8} | {'#Positive':>9}")
    print("-" * 170)

    for sig_key in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']:
        sub = [r for r in results if r['config']['signal'] == sig_key]
        if not sub:
            continue
        avg_ann = np.mean([r['ann'] for r in sub])
        avg_wr = np.mean([r['wr'] for r in sub])
        avg_n = np.mean([r['n'] for r in sub])
        avg_pnl = np.mean([r['avg_pnl'] for r in sub])
        avg_mdd = np.mean([r['mdd'] for r in sub])
        n_pos = sum(1 for r in sub if r['ann'] > 0)
        print(f"  {sig_names.get(sig_key, sig_key):<48} | {avg_ann:>+8.1f}% | {avg_wr:>6.1f}% | {avg_n:>7.0f} | {avg_pnl:>+7.3f}% | {avg_mdd:>7.1f}% | {n_pos:>5}/{len(sub)}")

    # ================================================================
    # WALK-FORWARD (Top 15 configs + best per signal)
    # ================================================================
    wf_years = [2020, 2021, 2022, 2023, 2024, 2025]

    # Collect top 15 overall + best per signal type
    wf_configs = list(results[:15])
    for sig_key in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']:
        if sig_key in best_per_sig:
            r = best_per_sig[sig_key]
            if r['config'] not in [w['config'] for w in wf_configs]:
                wf_configs.append(r)

    print(f"\n{'=' * 220}")
    print(f"  WALK-FORWARD ({len(wf_configs)} configs)")
    print(f"{'=' * 220}")

    header = f"  {'#':>3} | {'Config':<42} | {'Avg':>8} |"
    for yr in wf_years:
        header += f" {yr:>7} |"
    header += f" {'Pos':>4} | {'MDD':>7} | {'WR':>6}"
    print(header)
    print("-" * 220)

    wf_rows = []
    for i, r in enumerate(wf_configs):
        cfg = r['config']
        wf_row = {'label': cfg['label'], 'signal': cfg['signal'],
                  'entry': 'next_open', 'windows': {}, 'mdd': {}, 'wr': {}}
        for yr in wf_years:
            wr = run_backtest(cfg, wf_test_year=yr)
            if wr:
                wf_row['windows'][yr] = wr['ann']
                wf_row['mdd'][yr] = wr['mdd']
                wf_row['wr'][yr] = wr['wr']
        wf_rows.append(wf_row)

        vals = [wf_row['windows'].get(yr, 0) for yr in wf_years]
        avg = np.mean(vals) if vals else 0
        pos = sum(1 for v in vals if v > 0)
        avg_mdd = np.mean(list(wf_row['mdd'].values())) if wf_row['mdd'] else 0
        avg_wr = np.mean(list(wf_row['wr'].values())) if wf_row['wr'] else 0

        row_str = f"  {i+1:>3} | {wf_row['label']:<42} | {avg:>+7.1f}% |"
        for v in vals:
            row_str += f" {v:>+7.1f}% |"
        row_str += f" {pos}/6 | {avg_mdd:>6.1f}% | {avg_wr:>5.1f}%"
        print(row_str)

    # ================================================================
    # WF COMPARISON PER SIGNAL
    # ================================================================
    print(f"\n{'=' * 200}")
    print("  WALK-FORWARD COMPARISON (Best per signal type)")
    print(f"{'=' * 200}")
    header2 = f"  {'Signal':<48} | {'WF Avg':>8} |"
    for yr in wf_years:
        header2 += f" {yr:>7} |"
    header2 += f" {'Pos':>4} | Avg MDD | Avg WR"
    print(header2)
    print("-" * 200)

    for sig_key in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']:
        wf_match = [w for w in wf_rows if w['signal'] == sig_key]
        if wf_match:
            wf = wf_match[0]
            vals = [wf['windows'].get(yr, 0) for yr in wf_years]
            avg = np.mean(vals) if vals else 0
            pos = sum(1 for v in vals if v > 0)
            avg_mdd = np.mean(list(wf['mdd'].values())) if wf['mdd'] else 0
            avg_wr = np.mean(list(wf['wr'].values())) if wf['wr'] else 0
            row_str = f"  {sig_names.get(sig_key, sig_key):<48} | {avg:>+7.1f}% |"
            for v in vals:
                row_str += f" {v:>+7.1f}% |"
            row_str += f" {pos}/6 | {avg_mdd:>6.1f}% | {avg_wr:>5.1f}%"
            print(row_str)

    # ================================================================
    # FINAL VERDICT
    # ================================================================
    print(f"\n{'=' * 200}")
    print("  FINAL VERDICT: MATHEMATICAL/STATISTICAL INDICATORS")
    print(f"{'=' * 200}")
    print()
    print("  KEY QUESTION: Which mathematical indicators provide genuine alpha")
    print("  with next-open execution? What underlying market structure do they capture?")
    print()

    insights = {
        'A': 'Captures TREND PERSISTENCE via fractal geometry. Hurst > 0.55 means price changes are not independent -- positive autocorrelation indicates trending regime.',
        'B': 'Captures PRICE SMOOTHNESS via fractal dimension. Low FD means the price path is close to a straight line (clear directional move), filtering out noise.',
        'C': 'Captures PREDICTABILITY via information theory. Low entropy means return signs are imbalanced (more consecutive + or -), indicating trending behavior.',
        'D': 'Captures TREND LINEARITY via regression fit. High R^2 means price follows a near-linear trajectory, indicating orderly trending rather than choppy action.',
        'E': 'Captures STATISTICAL EXTREMES. z > 2 means returns 2+ standard deviations above the rolling mean, testing whether extreme days have follow-through momentum.',
        'F': 'Captures SERIAL CORRELATION. Positive autocorrelation at lag 1 means today\'s return predicts tomorrow\'s -- the fundamental precondition for momentum strategies.',
        'G': 'Captures RETURN ASYMMETRY. Positive skewness means the return distribution has a fat right tail -- the market tends to have larger upward moves than downward.',
        'H': 'Captures TAIL RISK REGIME. High kurtosis + expanding range = fat-tailed distribution with volatility expansion, often preceding breakouts.',
        'I': 'Captures MOMENTUM ACCELERATION. Not just positive velocity (momentum) but accelerating momentum -- like a ball rolling downhill getting faster.',
        'J': 'Captures AR(1) STRUCTURE. High PACF at lag 1 means strong autoregressive structure, validating trend-following as the correct strategy for this regime.',
        'K': 'Captures MOMENTUM CROSSOVER. Fast ROC crossing above slow ROC = short-term momentum accelerating relative to medium-term, classic timing signal.',
        'L': 'Captures IDIOSYNCRATIC STRENGTH. Low-beta commodity rising while market falls = genuine commodity-specific demand, not just beta-driven movement.',
    }

    for sig_key in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']:
        sub = [r for r in results if r['config']['signal'] == sig_key]
        if not sub:
            continue
        best = sub[0]
        n_pos = sum(1 for r in sub if r['ann'] > 0)

        wf_match = [w for w in wf_rows if w['signal'] == sig_key]
        wf_pos = 0
        wf_avg = 0
        if wf_match:
            wf = wf_match[0]
            vals = [wf['windows'].get(yr, 0) for yr in wf_years]
            wf_pos = sum(1 for v in vals if v > 0)
            wf_avg = np.mean(vals)

        verdict = "POSITIVE" if best['ann'] > 0 else "NEGATIVE"
        genuine = "GENUINE ALPHA" if wf_pos >= 4 and best['ann'] > 0 else ("MARGINAL" if wf_pos >= 3 and best['ann'] > 0 else "NO ALPHA")
        beats = "BEATS +81.9%" if best['ann'] > 81.9 else ("CLOSE" if best['ann'] > 50 else "INSUFFICIENT")

        print(f"  {sig_names.get(sig_key, sig_key)}")
        print(f"    Best annual: {best['ann']:>+8.1f}%  |  {n_pos}/{len(sub)} positive configs")
        print(f"    Walk-forward: {wf_pos}/6 positive  |  WF avg: {wf_avg:>+8.1f}%")
        print(f"    Trade freq: {best['freq']:>5.1f}/yr  |  Avg hold: {best['avg_hold']:>5.1f}d  |  Avg PnL: {best['avg_pnl']:>+6.3f}%")
        print(f"    VERDICT: {verdict}  -->  {genuine}  -->  {beats}")
        print(f"    INSIGHT: {insights.get(sig_key, '')}")
        print()

    # Top 5 configs
    print(f"\n  TOP 5 CONFIGS BY ANNUAL RETURN:")
    print(f"  {'#':>3} | {'Label':<42} | {'Ann':>9} | {'WR':>6} | {'N':>5} | {'MDD':>7} | WF_Avg | WF_Pos")
    print("-" * 140)
    for i, r in enumerate(results[:5]):
        wf_match = [w for w in wf_rows if w['label'] == r['label']]
        if wf_match:
            wf = wf_match[0]
            vals = [wf['windows'].get(yr, 0) for yr in wf_years]
            wf_avg = np.mean(vals)
            wf_pos = sum(1 for v in vals if v > 0)
        else:
            wf_avg = 0
            wf_pos = 0
        print(f"  {i+1:>3} | {r['label']:<42} | {r['ann']:>+8.1f}% | {r['wr']:>5.1f}% | {r['n']:>5} | {r['mdd']:>6.1f}% | {wf_avg:>+7.1f}% | {wf_pos}/6")

    # Configs beating +81.9%
    beating = [r for r in results if r['ann'] > 81.9]
    print(f"\n  CONFIGS BEATING +81.9% (current best): {len(beating)}")
    for r in beating[:20]:
        wf_match = [w for w in wf_rows if w['label'] == r['label']]
        wf_pos = 0
        wf_avg = 0
        if wf_match:
            wf = wf_match[0]
            vals = [wf['windows'].get(yr, 0) for yr in wf_years]
            wf_pos = sum(1 for v in vals if v > 0)
            wf_avg = np.mean(vals)
        print(f"    {r['label']:<42} | Ann={r['ann']:>+8.1f}% | WF Avg={wf_avg:>+7.1f}% | WF Pos={wf_pos}/6")

    # Absolute best
    if results:
        champ = results[0]
        print(f"\n  {'='*70}")
        print(f"  CHAMPION: {champ['label']}")
        print(f"    Annual: {champ['ann']:>+8.1f}%  |  WR: {champ['wr']:>5.1f}%  |  N: {champ['n']:>4}  |  MDD: {champ['mdd']:>6.1f}%")
        print(f"    Avg PnL/trade: {champ['avg_pnl']:>+6.3f}%  |  Avg Hold: {champ['avg_hold']:>5.1f}d  |  Freq: {champ['freq']:>5.1f}/yr")
        champ_wf = [w for w in wf_rows if w['label'] == champ['label']]
        if champ_wf:
            cw = champ_wf[0]
            vals = [cw['windows'].get(yr, 0) for yr in wf_years]
            print(f"    WF: {[f'{v:>+7.1f}%' for v in vals]}  |  {sum(1 for v in vals if v > 0)}/6 positive")
        print(f"  {'='*70}")

    print(f"\n  Total runtime: {time.time()-t_start:.0f}s")


if __name__ == '__main__':
    main()
