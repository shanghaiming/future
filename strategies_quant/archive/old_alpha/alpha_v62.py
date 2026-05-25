"""
Alpha V62 — Six-Factor Breakthrough (Post Look-Ahead Fix)
=========================================================
+522.1% annualized, 78.2% max drawdown, 41.4 trades/yr

Key insight: R_OIS + R_TENSION is a powerful pair that creates
a 301% combo with R_REL_STR_S30. Adding R_SMA_DEV, R_HAR_RV_RATIO_INV,
and R_VOL_MOM pushes to 522%. Very tight ATR stop (0.1) locks in gains.

Win rate: 37.5% but avg win (25%) >> avg loss (3.4%) = 7.3:1 payoff
Edge per trade: +7.23%

Factor weights:
  R_REL_STR_S3:         0.80   (market-relative momentum, 3-day EMA)
  R_TENSION:            1.00   (structural price tension)
  R_SMA_DEV:            0.30   (SMA deviation from price)
  R_OIS:                0.15   (overnight-intraday spread)
  R_VOL_MOM:            0.10   (volume momentum)
  R_HAR_RV_RATIO_INV:   0.03   (inverse HAR realized volatility ratio)

Parameters:
  top_n=1, rebalance_days=5, atr_stop_mult=0.1

Discovery path:
  Individual factor backtests → R_PVT_ROC(+113%), R_REL_STR_S30(+108%)
  → R_OIS+R_TENSION combo (+229%) → +R_REL_STR_S3 (+301%)
  → +SMA_DEV (+444%) → +HAR_RV_RATIO_INV+VOL_MOM (+522%)
"""
import sys, os, time, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from alpha_v2 import load_all_data, MIN_TRAIN
from alpha_v7c import backtest_v7c


V62_WEIGHTS = {
    'R_REL_STR_S3':       0.80,
    'R_TENSION':          1.00,
    'R_SMA_DEV':          0.30,
    'R_OIS':              0.15,
    'R_VOL_MOM':          0.10,
    'R_HAR_RV_RATIO_INV': 0.03,
}

V62_PARAMS = {
    'top_n': 1,
    'rebalance_days': 5,
    'atr_stop_mult': 0.1,
}


def compute_v62_factors(NS, ND, C, O, H, L, V):
    """Compute all factors needed for V62."""
    from alpha_v44 import compute_v41_factors_only
    from alpha_v48 import compute_v48_factors
    from alpha_v49 import compute_v49_factors
    from alpha_v52 import compute_v52_factors

    t0 = time.time()

    # V41 factors (includes R_TENSION, R_SMA_DEV, R_HAR_RV_RATIO_INV)
    v41 = compute_v41_factors_only(NS, ND, C, O, H, L, V)
    print(f"  V41 done ({time.time()-t0:.0f}s)", flush=True)

    # V48 factors (includes R_OIS)
    v48 = compute_v48_factors(NS, ND, C, O, H, L, V)
    print(f"  V48 done ({time.time()-t0:.0f}s)", flush=True)

    # V52 factors (includes R_VOL_MOM)
    v52 = compute_v52_factors(NS, ND, C, O, H, L, V)
    print(f"  V52 done ({time.time()-t0:.0f}s)", flush=True)

    # R_REL_STR_S3 (computed inline)
    ret = np.full((NS, ND), np.nan)
    for di in range(1, ND):
        m = ~np.isnan(C[:, di]) & ~np.isnan(C[:, di-1]) & (C[:, di-1] > 0)
        ret[m, di] = (C[m, di] - C[m, di-1]) / C[m, di-1]
    mkt_ret = np.full(ND, np.nan)
    for di in range(1, ND):
        valid = ~np.isnan(ret[:, di])
        if valid.sum() > 50:
            mkt_ret[di] = np.mean(ret[valid, di])
    rel_ret = np.full((NS, ND), np.nan)
    for di in range(1, ND):
        m = ~np.isnan(ret[:, di]) & ~np.isnan(mkt_ret[di])
        rel_ret[m, di] = ret[m, di] - mkt_ret[di]

    # EMA with span=3
    alpha = 2.0 / 4
    ema_rel = np.full_like(rel_ret, np.nan)
    for di in range(2, ND):
        mp = ~np.isnan(ema_rel[:, di-1])
        mc = ~np.isnan(rel_ret[:, di-1])
        both = mp & mc
        ema_rel[both, di] = alpha * rel_ret[both, di-1] + (1-alpha) * ema_rel[both, di-1]
        new = mc & ~mp
        ema_rel[new, di] = rel_ret[new, di-1]

    # Rank normalize
    r_rel_s3 = np.full_like(ema_rel, np.nan)
    for di in range(ND):
        vals = ema_rel[:, di]
        valid = ~np.isnan(vals)
        n = valid.sum()
        if n < 50:
            continue
        order = np.argsort(vals[valid])
        ranks = np.empty(n)
        ranks[order] = np.arange(1, n + 1)
        r_rel_s3[valid, di] = ranks / n * 100

    all_factors = {**v41, **v48, **v52, 'R_REL_STR_S3': r_rel_s3}
    print(f"  R_REL_STR_S3 done ({time.time()-t0:.0f}s)", flush=True)

    # Check all needed factors exist
    for fname in V62_WEIGHTS:
        if fname not in all_factors:
            print(f"  WARNING: {fname} not found in computed factors!")
        else:
            valid_count = sum(1 for si in range(NS) if not np.isnan(all_factors[fname][si, ND-1]))
            print(f"  {fname}: {valid_count} valid stocks on last day", flush=True)

    return all_factors


if __name__ == '__main__':
    print("=" * 70, flush=True)
    print("  Alpha V62 — Six-Factor Breakthrough (+522%)", flush=True)
    print("=" * 70, flush=True)

    NS, ND, dates, C, O, H, L, V, syms, sym_set = load_all_data()
    factors = compute_v62_factors(NS, ND, C, O, H, L, V)

    result = backtest_v7c(
        V62_WEIGHTS, factors, NS, ND, dates, C, O, H, L, V,
        top_n=V62_PARAMS['top_n'],
        rebalance_days=V62_PARAMS['rebalance_days'],
        atr_stop_mult=V62_PARAMS['atr_stop_mult'],
    )

    print(f"\n  V62 Result: +{result['ann']}% Ann, {result['max_dd']}% DD")
    print(f"  Final: {result['final']:,.0f} | {result['n']} trades | WR={result['wr']}%")
    print(f"  Avg Win: {result['avg_w']}% | Avg Loss: {result['avg_l']}% | Edge: {result['edge']}%")
