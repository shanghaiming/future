"""
Backtest Individual Alpha Factors (Post Look-Ahead Bias Fix)
=============================================================
Computes factors from v44, v48, v49, v52, v55, and dashboard REL_STRENGTH,
then runs a single-factor backtest for EACH factor independently.

Standard parameters: top_n=1, rebalance_days=5, atr_stop_mult=0.8
Reports: annualized return, max drawdown, win rate, number of trades.

Factor cache is saved to .sim_cache/ (same as sim_run.py) so re-runs
are fast -- only the backtest loop needs to execute.
"""
import sys, os, time, json, warnings
import numpy as np
import datetime
from pathlib import Path

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from alpha_v2 import load_all_data, MIN_TRAIN
from alpha_v7c import backtest_v7c


# ---------------------------------------------------------------------------
# Helper functions (local copies to avoid importing heavy modules)
# ---------------------------------------------------------------------------

def _rank_normalize(factor_2d, min_stocks=50):
    """Rank-normalize a (NS, ND) array cross-sectionally to [1, 100]."""
    NS, ND = factor_2d.shape
    ranked = np.full_like(factor_2d, np.nan)
    for di in range(ND):
        vals = factor_2d[:, di]
        valid = ~np.isnan(vals)
        n = valid.sum()
        if n < min_stocks:
            continue
        order = np.argsort(vals[valid])
        ranks = np.empty(n)
        ranks[order] = np.arange(1, n + 1)
        ranked[valid, di] = ranks / n * 100
    return ranked


def _ema(arr, span):
    """EMA along axis=1. No look-ahead: uses arr[:, di-1]."""
    NS, ND = arr.shape
    alpha = 2.0 / (span + 1)
    out = np.full_like(arr, np.nan)
    for di in range(2, ND):
        mask_prev = ~np.isnan(out[:, di - 1])
        mask_curr = ~np.isnan(arr[:, di - 1])
        both = mask_prev & mask_curr
        out[both, di] = alpha * arr[both, di - 1] + (1 - alpha) * out[both, di - 1]
        new_only = mask_curr & ~mask_prev
        out[new_only, di] = arr[new_only, di - 1]
    return out


def _rolling_mean(arr, window, min_valid=None):
    """Rolling mean along axis=1. Handles NaN."""
    if min_valid is None:
        min_valid = window // 2
    NS, ND = arr.shape
    out = np.full_like(arr, np.nan)
    cumsum = np.nancumsum(arr, axis=1)
    cumcount = np.cumsum(~np.isnan(arr), axis=1)
    for di in range(window, ND):
        s = cumsum[:, di - 1] - (cumsum[:, di - window - 1] if di > window else 0)
        c = cumcount[:, di - 1] - (cumcount[:, di - window - 1] if di > window else 0)
        valid = c >= min_valid
        out[valid, di] = s[valid] / c[valid]
    return out


# ---------------------------------------------------------------------------
# Factor cache (same location and format as sim_run.py)
# ---------------------------------------------------------------------------

CACHE_DIR = Path(__file__).resolve().parent / '.sim_cache'


def _find_latest_data_date(ND, dates, C, NS):
    """Find the latest trading date in the data."""
    latest_di = ND - 1
    while latest_di > MIN_TRAIN:
        valid = sum(1 for si in range(NS) if not np.isnan(C[si, latest_di]))
        if valid > 50:
            break
        latest_di -= 1
    raw_date = dates[latest_di]
    return str(raw_date.date() if hasattr(raw_date, 'date') else raw_date)


def load_or_compute_factors(NS, ND, C, O, H, L, V, dates, force_recompute=False):
    """Load cached factors or compute from scratch. Same cache format as sim_run.py."""
    from alpha_v44 import compute_v41_factors_only
    from alpha_v48 import compute_v48_factors
    from alpha_v49 import compute_v49_factors
    from alpha_v52 import compute_v52_factors

    cache_key = _find_latest_data_date(ND, dates, C, NS)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f'factors_{cache_key}.npz'
    meta_file = CACHE_DIR / f'factors_{cache_key}_meta.json'

    # Check cache -- load if available
    if not force_recompute and cache_file.exists() and meta_file.exists():
        print(f"  Loading cached factors for {cache_key}...", flush=True)
        t1 = time.time()
        loaded = np.load(cache_file, allow_pickle=False)
        with open(meta_file, 'r') as f:
            meta = json.load(f)
        all_factors = {name: loaded[name] for name in meta['factor_names']}
        print(f"  Cache loaded: {len(all_factors)} factors ({time.time()-t1:.1f}s)", flush=True)
        return all_factors

    # Compute from scratch
    print(f"  Computing factors for {cache_key}...", flush=True)
    t1 = time.time()

    # V41 factors (~62 min, compute once)
    print("\n  [1/5] Computing V41 factors (slow, ~62min)...", flush=True)
    v41 = compute_v41_factors_only(NS, ND, C, O, H, L, V)
    print(f"  V41 done ({time.time()-t1:.1f}s)", flush=True)

    # V48 factors
    print("\n  [2/5] Computing V48 factors...", flush=True)
    v48 = compute_v48_factors(NS, ND, C, O, H, L, V)
    print(f"  V48 done ({time.time()-t1:.1f}s)", flush=True)

    # V49 factors
    print("\n  [3/5] Computing V49 factors...", flush=True)
    v49 = compute_v49_factors(NS, ND, C, O, H, L, V)
    print(f"  V49 done ({time.time()-t1:.1f}s)", flush=True)

    # V52 factors
    print("\n  [4/5] Computing V52 factors...", flush=True)
    v52 = compute_v52_factors(NS, ND, C, O, H, L, V)
    print(f"  V52 done ({time.time()-t1:.1f}s)", flush=True)

    # R_TREND_ACC (computed directly to avoid full v55 ~40+ min)
    print("\n  [5/5] Computing R_TREND_ACC + REL_STRENGTH...", flush=True)
    ema5 = np.full((NS, ND), np.nan)
    ema60 = np.full((NS, ND), np.nan)
    a5 = 2.0 / 6
    a60 = 2.0 / 61
    for di in range(2, ND):
        m = ~np.isnan(C[:, di - 1])
        for arr, alpha in [(ema5, a5), (ema60, a60)]:
            prev = ~np.isnan(arr[:, di - 1])
            both = prev & m
            arr[both, di] = alpha * C[both, di - 1] + (1 - alpha) * arr[both, di - 1]
            new = m & ~prev
            arr[new, di] = C[new, di - 1]
    trend_str = np.full((NS, ND), np.nan)
    mask = ~np.isnan(ema5) & ~np.isnan(ema60) & (ema60 > 0)
    trend_str[mask] = (ema5[mask] - ema60[mask]) / ema60[mask]
    trend_acc = np.full((NS, ND), np.nan)
    for di in range(6, ND):
        m = ~np.isnan(trend_str[:, di]) & ~np.isnan(trend_str[:, di - 5])
        trend_acc[m, di] = trend_str[m, di] - trend_str[m, di - 5]
    r_trend_acc = _rank_normalize(trend_acc)
    print(f"  R_TREND_ACC done ({time.time()-t1:.1f}s)", flush=True)

    # REL_STRENGTH variants (from alpha_v61.py logic)
    ret = np.full((NS, ND), np.nan)
    for di in range(1, ND):
        m = ~np.isnan(C[:, di]) & ~np.isnan(C[:, di - 1]) & (C[:, di - 1] > 0)
        ret[m, di] = (C[m, di] - C[m, di - 1]) / C[m, di - 1]

    mkt_ret = np.full(ND, np.nan)
    for di in range(1, ND):
        valid = ~np.isnan(ret[:, di])
        if valid.sum() > 50:
            mkt_ret[di] = np.mean(ret[valid, di])

    rel_ret = np.full((NS, ND), np.nan)
    for di in range(1, ND):
        m = ~np.isnan(ret[:, di]) & ~np.isnan(mkt_ret[di])
        rel_ret[m, di] = ret[m, di] - mkt_ret[di]

    rel_factors = {}
    for span in [3, 5, 10, 30]:
        name = f'R_REL_STR_S{span}'
        smoothed = _ema(rel_ret, span)
        rel_factors[name] = _rank_normalize(smoothed)
    print(f"  REL_STRENGTH done ({time.time()-t1:.1f}s)", flush=True)

    # Combine all factors
    all_factors = {
        **v41, **v48, **v49, **v52,
        'R_TREND_ACC': r_trend_acc,
        **rel_factors,
    }
    print(f"\n  All factors computed: {len(all_factors)} total ({time.time()-t1:.1f}s)", flush=True)

    # Save to cache (same format as sim_run.py)
    np.savez_compressed(cache_file, **all_factors)
    with open(meta_file, 'w') as f:
        json.dump({
            'factor_names': list(all_factors.keys()),
            'data_date': cache_key,
            'computed_at': datetime.datetime.now().isoformat(),
        }, f)
    print(f"  Factors cached to {cache_file.name}", flush=True)

    return all_factors


# ---------------------------------------------------------------------------
# Select factors to backtest
# ---------------------------------------------------------------------------

def select_factors(all_factors):
    """Select the specific factors to backtest individually.

    Returns a list of (name, source_module) tuples.
    """
    factor_spec = []

    # From alpha_v44 (compute_v41_factors_only): core structural factors
    factor_spec.extend([
        ('R_BWP_BNW', 'v44'),
        ('R_TENSION', 'v44'),
        ('R_R_SQUARED', 'v44'),
        ('R_SMA_DEV', 'v44'),
        ('R_HAR_RV_RATIO_INV', 'v44'),
    ])

    # From alpha_v48: BVR and other notable factors
    factor_spec.extend([
        ('R_BVR', 'v48'),
        ('R_OIS', 'v48'),
        ('R_PCS', 'v48'),
        ('R_VCM', 'v48'),
        ('R_TER', 'v48'),
        ('R_ISKEW', 'v48'),
        ('R_AMIHUD', 'v48'),
        ('R_ONRET', 'v48'),
    ])

    # From alpha_v49: VWCM, notable factors
    factor_spec.extend([
        ('R_BUY_FRAC', 'v49'),
        ('R_VWCM', 'v49'),
        ('R_WQ1', 'v49'),
        ('R_WQ6', 'v49'),
        ('R_PV_CORR', 'v49'),
        ('R_KW_MOM', 'v49'),
        ('R_RET_ASYM', 'v49'),
        ('R_RANGE_POS', 'v49'),
        ('R_IV_TREND', 'v49'),
        ('R_GAP_MOM', 'v49'),
        ('R_PVT_ROC', 'v49'),
    ])

    # From alpha_v52: VPIN, microstructure factors
    factor_spec.extend([
        ('R_VPIN', 'v52'),
        ('R_BUY_FRAC', 'v52'),
        ('R_KYLE', 'v52'),
        ('R_CS_SPREAD', 'v52'),
        ('R_ROLL', 'v52'),
        ('R_SMI', 'v52'),
        ('R_OFI', 'v52'),
        ('R_DEPTH_IMB', 'v52'),
        ('R_VOL_MOM', 'v52'),
        ('R_LIQ_MOM', 'v52'),
    ])

    # From v55: R_TREND_ACC
    factor_spec.extend([
        ('R_TREND_ACC', 'v55'),
    ])

    # From dashboard REL_STRENGTH
    factor_spec.extend([
        ('R_REL_STR_S30', 'dashboard'),
        ('R_REL_STR_S10', 'dashboard'),
        ('R_REL_STR_S5', 'dashboard'),
        ('R_REL_STR_S3', 'dashboard'),
    ])

    # Filter to only those actually present in all_factors, deduplicating names
    seen = set()
    selected = []
    for name, source in factor_spec:
        if name in all_factors and name not in seen:
            selected.append((name, source))
            seen.add(name)

    # Check for any factors that exist in all_factors but weren't listed
    # (e.g., if v48/v49/v52 produce additional factors not listed above)
    listed = {name for name, _ in selected}
    for name in sorted(all_factors.keys()):
        if name not in listed:
            selected.append((name, 'other'))

    return selected


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 80, flush=True)
    print("  Individual Factor Backtest (Post Look-Ahead Bias Fix)")
    print("  Parameters: top_n=1, rebalance_days=5, atr_stop_mult=0.8", flush=True)
    print("=" * 80)

    t_total = time.time()

    # Load data
    print("\n  Loading data...", flush=True)
    NS, ND, dates, C, O, H, L, V, syms, sym_set = load_all_data()
    print(f"  Data loaded: {NS} stocks, {ND} days ({time.time()-t_total:.1f}s)", flush=True)

    # Compute or load cached factors
    print("\n  --- Factor Computation ---", flush=True)
    all_factors = load_or_compute_factors(NS, ND, C, O, H, L, V, dates)
    print(f"  Available factors: {len(all_factors)}", flush=True)

    # Select factors to test
    factor_list = select_factors(all_factors)
    print(f"\n  Factors to backtest: {len(factor_list)}", flush=True)

    # Run backtests
    print(f"\n  --- Running Single-Factor Backtests ---", flush=True)
    print(f"  {'Factor':<25s} {'Src':<10s} | {'Ann':>8s} {'N':>5s} {'WR':>6s} {'Edge':>7s} {'DD':>6s} {'Final':>12s}", flush=True)
    print(f"  {'-'*90}", flush=True)

    results = []
    for idx, (fname, source) in enumerate(factor_list):
        t0 = time.time()
        r = backtest_v7c(
            {fname: 1.0}, all_factors, NS, ND, dates, C, O, H, L, V,
            top_n=1, rebalance_days=5, atr_stop_mult=0.8,
        )
        elapsed = time.time() - t0
        if r:
            r['factor'] = fname
            r['source'] = source
            results.append(r)
            print(f"  {fname:<25s} {source:<10s} | {r['ann']:+7.1f}% {r['n']:5d} {r['wr']:5.1f}% "
                  f"{r['edge']:+6.2f}% {r['max_dd']:5.1f}% {r['final']:>10,.0f}   ({elapsed:.1f}s)",
                  flush=True)
        else:
            print(f"  {fname:<25s} {source:<10s} | NO TRADES ({elapsed:.1f}s)", flush=True)

    # Sort by annualized return (descending)
    results.sort(key=lambda x: -x['ann'])

    # Print sorted results table
    print(f"\n{'='*110}", flush=True)
    print(f"  SORTED RESULTS — Individual Factor Backtest (top_n=1, rebal=5, atr=0.8)", flush=True)
    print(f"  {'#':>3s}  {'Factor':<25s} {'Src':<10s} | {'Ann':>8s} {'N':>5s} {'WR':>6s} {'Edge':>7s} {'DD':>6s} {'Final':>12s}", flush=True)
    print(f"  {'-'*105}", flush=True)
    for i, r in enumerate(results):
        print(f"  {i+1:3d}  {r['factor']:<25s} {r['source']:<10s} | "
              f"{r['ann']:+7.1f}% {r['n']:5d} {r['wr']:5.1f}% "
              f"{r['edge']:+6.2f}% {r['max_dd']:5.1f}% {r['final']:>10,.0f}", flush=True)

    # Print grouped by source module
    print(f"\n{'='*110}", flush=True)
    print(f"  RESULTS GROUPED BY SOURCE MODULE", flush=True)
    sources = sorted(set(r['source'] for r in results))
    for src in sources:
        src_results = sorted([r for r in results if r['source'] == src], key=lambda x: -x['ann'])
        print(f"\n  --- {src} ---", flush=True)
        print(f"  {'Factor':<25s} | {'Ann':>8s} {'N':>5s} {'WR':>6s} {'Edge':>7s} {'DD':>6s}", flush=True)
        print(f"  {'-'*70}", flush=True)
        for r in src_results:
            print(f"  {r['factor']:<25s} | {r['ann']:+7.1f}% {r['n']:5d} {r['wr']:5.1f}% "
                  f"{r['edge']:+6.2f}% {r['max_dd']:5.1f}%", flush=True)

    # Summary statistics
    if results:
        print(f"\n{'='*110}", flush=True)
        print(f"  SUMMARY", flush=True)
        print(f"  Total factors tested: {len(results)}", flush=True)
        print(f"  Best factor:  {results[0]['factor']} ({results[0]['source']}) = {results[0]['ann']:+.1f}% Ann, {results[0]['max_dd']:.1f}% DD", flush=True)
        worst = results[-1]
        print(f"  Worst factor: {worst['factor']} ({worst['source']}) = {worst['ann']:+.1f}% Ann, {worst['max_dd']:.1f}% DD", flush=True)
        positive = [r for r in results if r['ann'] > 0]
        print(f"  Positive returns: {len(positive)}/{len(results)} ({len(positive)/len(results)*100:.0f}%)", flush=True)
        avg_ann = np.mean([r['ann'] for r in results])
        avg_dd = np.mean([r['max_dd'] for r in results])
        print(f"  Average Ann Return: {avg_ann:+.1f}%", flush=True)
        print(f"  Average Max Drawdown: {avg_dd:.1f}%", flush=True)

    print(f"\n  Total runtime: {time.time()-t_total:.0f}s", flush=True)
    print(f"{'='*80}", flush=True)


if __name__ == '__main__':
    main()
