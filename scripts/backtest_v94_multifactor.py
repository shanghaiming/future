#!/usr/bin/env python3
"""
Backtest V94: Multi-Factor Commodity Ranking Strategy (Optimized)
=================================================================
Weekly-rebalanced, cross-sectional commodity futures ranking strategy.

Key factors discovered:
  - POI (Price-OI confirmation): strongest signal (Sharpe ~0.90 standalone)
  - Momentum-20: trend following
  - Momentum-60: intermediate trend
  - Volatility: low-vol preference (risk adjustment)
  - Carry: weak/negative in Chinese commodities (structural long bias)

Best configurations found:
  poi_vol_mom  L5/S0 lev1x:  Sharpe=1.23, AnnRet=34.21%, MDD=-35.94%
  poi_only     L3/S0 lev1x:  Sharpe=1.30, AnnRet=44.03%, MDD=-40.42%
  poi_only     L5/S0 lev1x:  Sharpe=1.07, AnnRet=27.89%, MDD=-33.37%

Data:
  - Term structure: data/futures_term_structure/  (JSON files)
  - Daily OHLCV+OI: data/futures_weighted/        (CSV files)
"""

import os
import json
import glob
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TS_DIR = os.path.join(BASE_DIR, "data", "futures_term_structure")
DAILY_DIR = os.path.join(BASE_DIR, "data", "futures_weighted")

# ---------------------------------------------------------------------------
# Factor weights (optimized from exploration)
# ---------------------------------------------------------------------------
WEIGHT_PRESETS = {
    # Top performers
    "poi_vol_mom":  {"carry": 0, "mom20": 1.0, "mom60": 0.5, "mr5": 0, "vol": 0.5, "oi": 0, "poi": 2.5},
    "poi_only":     {"carry": 0, "mom20": 0, "mom60": 0, "mr5": 0, "vol": 0, "oi": 0, "poi": 1.0},
    "poi_mom_oi":   {"carry": 0, "mom20": 1.0, "mom60": 0.5, "mr5": 0, "vol": 0, "oi": 0.5, "poi": 3.0},
    "all_star":     {"carry": 0, "mom20": 1.0, "mom60": 0.5, "mr5": 0, "vol": 0.3, "oi": 0.5, "poi": 3.0},
    "poi_mom20":    {"carry": 0, "mom20": 1.5, "mom60": 0, "mr5": 0, "vol": 0, "oi": 0, "poi": 2.5},
    # Original comparisons
    "smart_momentum": {"carry": 1.0, "mom20": 2.5, "mom60": 1.5, "mr5": -0.5, "vol": 0.5, "oi": 1.0, "poi": 1.0},
    "momentum":     {"carry": 0.5, "mom20": 2.0, "mom60": 1.5, "mr5": 0.0, "vol": 0.5, "oi": 0.5, "poi": 0.5},
    "carry_heavy":  {"carry": 3.0, "mom20": 1.0, "mom60": 0.5, "mr5": 0.0, "vol": 0.5, "oi": 0.5, "poi": 0.0},
    "equal":        {"carry": 1.0, "mom20": 1.0, "mom60": 0.0, "mr5": 0.5, "vol": 0.5, "oi": 0.5, "poi": 0.0},
    "balanced":     {"carry": 1.5, "mom20": 1.5, "mom60": 1.0, "mr5": 0.5, "vol": 1.0, "oi": 1.0, "poi": 0.5},
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_term_structure():
    """Load all term structure JSON files into a DataFrame."""
    records = []
    for fp in glob.glob(os.path.join(TS_DIR, "*.json")):
        with open(fp, "r") as f:
            d = json.load(f)
        records.append({
            "symbol": d["symbol"],
            "date": d["date"],
            "structure": d.get("structure", ""),
            "near_price": d.get("near_price", np.nan),
            "total_spread_pct": d.get("total_spread_pct", np.nan),
        })
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values(["symbol", "date"], inplace=True)
    df.drop_duplicates(subset=["symbol", "date"], keep="last", inplace=True)
    return df


def load_daily_data():
    """Load all daily CSV files into a single DataFrame."""
    frames = []
    for fp in glob.glob(os.path.join(DAILY_DIR, "*.csv")):
        sym = os.path.basename(fp).replace(".csv", "")
        df = pd.read_csv(fp)
        df["symbol"] = sym
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["trade_date"] = pd.to_datetime(df["trade_date"], format="mixed")
    df.rename(columns={"trade_date": "date"}, inplace=True)
    df.sort_values(["symbol", "date"], inplace=True)
    df.drop_duplicates(subset=["symbol", "date"], keep="last", inplace=True)
    return df


# ---------------------------------------------------------------------------
# Factor computation
# ---------------------------------------------------------------------------

def compute_daily_factors(daily_df):
    """Compute price-based factors from daily data."""
    dfs = []
    for sym, grp in daily_df.groupby("symbol"):
        g = grp.set_index("date").sort_index()
        g = g[~g.index.duplicated(keep="last")]

        if len(g) < 120:
            continue

        close = g["close"].astype(float)
        vol = g["vol"].astype(float)
        oi = g["oi"].astype(float)
        valid = close > 0

        # Momentum factors
        mom_20 = close.pct_change(20)
        mom_60 = close.pct_change(60)
        ret_5 = close.pct_change(5)

        # Volatility
        daily_ret = close.pct_change()
        vol_20 = daily_ret.rolling(20).std() * np.sqrt(252)

        # OI trend
        oi_chg_20 = oi.pct_change(20)

        # Price-OI confirmation signal (the alpha factor)
        price_chg_5 = close.pct_change(5)
        oi_chg_5 = oi.pct_change(5)
        poi_signal = np.sign(price_chg_5) * np.sign(oi_chg_5) * np.abs(price_chg_5)

        out = pd.DataFrame({
            "symbol": sym,
            "close": close,
            "vol_raw": vol,
            "oi_raw": oi,
            "valid": valid,
            "mom20_score": mom_20,
            "mom60_score": mom_60,
            "mr5_score": -ret_5,
            "vol_score": -vol_20,
            "oi_score": oi_chg_20,
            "poi_score": poi_signal,
        }, index=g.index)

        dfs.append(out)

    result = pd.concat(dfs, axis=0) if dfs else pd.DataFrame()
    if len(result):
        result.index.name = "date"
        result.reset_index(inplace=True)
    return result


def compute_carry_scores(ts_df):
    """Compute carry score from term structure data."""
    ts_df = ts_df.copy()
    ts_df["carry_score"] = -ts_df["total_spread_pct"]
    return ts_df[["symbol", "date", "carry_score"]]


# ---------------------------------------------------------------------------
# Cross-sectional z-score with winsorization
# ---------------------------------------------------------------------------

def cross_sectional_zscore(df, factor_col):
    """Z-score a factor column cross-sectionally with winsorization."""
    result_col = factor_col + "_z"
    df = df.copy()

    def zscore_vals(vals):
        mask = np.isfinite(vals)
        if mask.sum() < 5:
            return np.full_like(vals, np.nan)
        mu = np.nanmean(vals)
        sigma = np.nanstd(vals)
        if sigma < 1e-12:
            return np.zeros_like(vals)
        return (vals - mu) / sigma

    # Winsorize then z-score
    def winsorize_vals(vals):
        mask = np.isfinite(vals)
        if mask.sum() < 10:
            return vals
        v = vals.copy()
        low = np.nanpercentile(v, 2)
        high = np.nanpercentile(v, 98)
        v[mask] = np.clip(v[mask], low, high)
        return v

    df["_tmp"] = df.groupby("date")[factor_col].transform(winsorize_vals)
    df[result_col] = df.groupby("date")["_tmp"].transform(zscore_vals)
    df.drop(columns=["_tmp"], inplace=True)
    return df


# ---------------------------------------------------------------------------
# Weekly rebalance dates
# ---------------------------------------------------------------------------

def get_monday_dates(dates_sorted):
    """Get all Monday dates from sorted date array."""
    dates = pd.Series(dates_sorted).sort_values().unique()
    return sorted([d for d in dates if pd.Timestamp(d).weekday() == 0])


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------

def run_backtest(factor_df, n_long, n_short, leverage, weight_dict,
                 start_date=None, end_date=None):
    """Run weekly-rebalanced long-short backtest."""
    factor_keys = ["carry", "mom20", "mom60", "mr5", "vol", "oi", "poi"]
    factor_z_cols = ["carry_score_z", "mom20_score_z", "mom60_score_z",
                     "mr5_score_z", "vol_score_z", "oi_score_z", "poi_score_z"]

    factor_df = factor_df.copy()

    # Build composite score
    composite_parts = []
    for fk, fn in zip(factor_keys, factor_z_cols):
        w = weight_dict.get(fk, 0.0)
        if w != 0 and fn in factor_df.columns:
            composite_parts.append(w * factor_df[fn].fillna(0))
    factor_df["composite"] = sum(composite_parts) if composite_parts else 0

    # Liquidity filters
    if "oi_raw" in factor_df.columns:
        factor_df.loc[factor_df["oi_raw"] < 1000, "composite"] = np.nan
    if "vol_raw" in factor_df.columns:
        factor_df.loc[factor_df["vol_raw"] < 100, "composite"] = np.nan
    if "valid" in factor_df.columns:
        factor_df.loc[~factor_df["valid"], "composite"] = np.nan

    # Date filter
    if start_date:
        factor_df = factor_df[factor_df["date"] >= pd.Timestamp(start_date)]
    if end_date:
        factor_df = factor_df[factor_df["date"] <= pd.Timestamp(end_date)]

    # Weekly rebalance (Mondays)
    all_dates = sorted(factor_df["date"].unique())
    mondays = get_monday_dates(all_dates)
    if len(mondays) < 2:
        return None

    portfolio_returns = []

    for i in range(len(mondays) - 1):
        rebal_date = mondays[i]
        next_rebal_date = mondays[i + 1]

        mask = factor_df["date"] == pd.Timestamp(rebal_date)
        day_data = factor_df[mask].dropna(subset=["composite"]).copy()

        min_needed = max(n_long, n_short, 1)
        if len(day_data) < min_needed:
            continue

        day_data.sort_values("composite", ascending=False, inplace=True)

        long_syms = day_data["symbol"].head(n_long).tolist() if n_long > 0 else []
        short_syms = day_data["symbol"].tail(n_short).tolist() if n_short > 0 else []

        next_mask = factor_df["date"] == pd.Timestamp(next_rebal_date)
        next_data = factor_df[next_mask].set_index("symbol")["close"]
        curr_data = day_data.set_index("symbol")["close"]

        long_rets = []
        for sym in long_syms:
            if sym in curr_data.index and sym in next_data.index:
                c0, c1 = curr_data[sym], next_data[sym]
                if np.isfinite(c0) and np.isfinite(c1) and c0 > 0:
                    long_rets.append((c1 - c0) / c0)

        short_rets = []
        for sym in short_syms:
            if sym in curr_data.index and sym in next_data.index:
                c0, c1 = curr_data[sym], next_data[sym]
                if np.isfinite(c0) and np.isfinite(c1) and c0 > 0:
                    short_rets.append(-(c1 - c0) / c0)

        if not long_rets and not short_rets:
            continue

        avg_long = np.mean(long_rets) if long_rets else 0.0
        avg_short = np.mean(short_rets) if short_rets else 0.0

        if n_short == 0:
            port_ret = leverage * avg_long
        elif n_long == 0:
            port_ret = leverage * avg_short
        else:
            port_ret = leverage * (avg_long + avg_short) / 2.0

        portfolio_returns.append({
            "date": rebal_date,
            "next_date": next_rebal_date,
            "long_avg": avg_long,
            "short_avg": avg_short,
            "port_ret": port_ret,
            "n_long": len(long_rets),
            "n_short": len(short_rets),
            "long_syms": long_syms,
            "short_syms": short_syms,
        })

    return pd.DataFrame(portfolio_returns) if portfolio_returns else None


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def compute_metrics(rets_df, label=""):
    """Compute strategy performance metrics from weekly returns."""
    if rets_df is None or len(rets_df) == 0:
        return None

    rets = rets_df["port_ret"].values
    cum = np.cumprod(1 + rets)
    total_return = cum[-1] - 1.0

    n_weeks = len(rets)
    n_years = n_weeks / 52.0
    if n_years < 0.01:
        return None
    annual_return = (1 + total_return) ** (1 / n_years) - 1

    mean_ret = np.mean(rets)
    std_ret = np.std(rets, ddof=1) if len(rets) > 1 else 1e-10
    sharpe = (mean_ret / std_ret) * np.sqrt(52) if std_ret > 1e-12 else 0.0
    wr = np.mean(rets > 0) * 100
    avg_ret = mean_ret * 100

    running_max = np.maximum.accumulate(cum)
    drawdowns = (cum - running_max) / running_max
    mdd = np.min(drawdowns) * 100

    calmar = annual_return / abs(mdd / 100) if abs(mdd) > 0.01 else 0.0

    downside = rets[rets < 0]
    downside_std = np.std(downside, ddof=1) if len(downside) > 1 else 1e-10
    sortino = (mean_ret / downside_std) * np.sqrt(52) if downside_std > 1e-12 else 0.0

    has_long = rets_df["n_long"].sum() > 0
    has_short = rets_df["n_short"].sum() > 0
    long_wr = np.mean(rets_df["long_avg"].values > 0) * 100 if has_long else 0
    short_wr = np.mean(rets_df["short_avg"].values > 0) * 100 if has_short else 0
    avg_long_ret = np.mean(rets_df["long_avg"].values) * 100 if has_long else 0
    avg_short_ret = np.mean(rets_df["short_avg"].values) * 100 if has_short else 0

    return {
        "label": label,
        "n_weeks": n_weeks,
        "n_years": round(n_years, 2),
        "total_return_pct": round(total_return * 100, 2),
        "annual_return_pct": round(annual_return * 100, 2),
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "calmar": round(calmar, 2),
        "win_rate_pct": round(wr, 2),
        "avg_weekly_ret_pct": round(avg_ret, 4),
        "mdd_pct": round(mdd, 2),
        "long_wr_pct": round(long_wr, 2),
        "short_wr_pct": round(short_wr, 2),
        "avg_long_ret_pct": round(avg_long_ret, 4),
        "avg_short_ret_pct": round(avg_short_ret, 4),
    }


def print_metrics(m):
    if m is None:
        print("  (no data)")
        return
    print(f"  Label:             {m['label']}")
    print(f"  Period:            {m['n_weeks']} weeks ({m['n_years']} years)")
    print(f"  Total Return:      {m['total_return_pct']:.2f}%")
    print(f"  Annual Return:     {m['annual_return_pct']:.2f}%")
    print(f"  Sharpe Ratio:      {m['sharpe']:.2f}")
    print(f"  Sortino Ratio:     {m['sortino']:.2f}")
    print(f"  Calmar Ratio:      {m['calmar']:.2f}")
    print(f"  Win Rate:          {m['win_rate_pct']:.1f}%")
    print(f"  Avg Weekly Ret:    {m['avg_weekly_ret_pct']:.4f}%")
    print(f"  Max Drawdown:      {m['mdd_pct']:.2f}%")
    if m["long_wr_pct"] > 0:
        print(f"  Long WR:           {m['long_wr_pct']:.1f}%  (avg ret: {m['avg_long_ret_pct']:.4f}%)")
    if m["short_wr_pct"] > 0:
        print(f"  Short WR:          {m['short_wr_pct']:.1f}%  (avg ret: {m['avg_short_ret_pct']:.4f}%)")
    print()


def _calc_mdd(rets):
    cum = np.cumprod(1 + np.array(rets))
    running_max = np.maximum.accumulate(cum)
    dd = (cum - running_max) / running_max
    return np.min(dd)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 85)
    print("BACKTEST V94: Multi-Factor Commodity Ranking Strategy (Optimized)")
    print("=" * 85)

    # ── Load data ──
    print("\n[1] Loading term structure data...")
    ts_df = load_term_structure()
    print(f"    {len(ts_df):,} records, {ts_df['symbol'].nunique()} symbols, "
          f"{ts_df['date'].min().date()} ~ {ts_df['date'].max().date()}")

    print("[2] Loading daily OHLCV data...")
    daily_df = load_daily_data()
    print(f"    {len(daily_df):,} records, {daily_df['symbol'].nunique()} symbols, "
          f"{daily_df['date'].min().date()} ~ {daily_df['date'].max().date()}")

    # ── Compute factors ──
    print("[3] Computing daily factors...")
    daily_factors = compute_daily_factors(daily_df)
    print(f"    {len(daily_factors):,} rows, {daily_factors['symbol'].nunique()} symbols")

    print("[4] Computing carry scores...")
    carry_df = compute_carry_scores(ts_df)

    # ── Merge ──
    print("[5] Merging factors...")
    carry_idx = carry_df.set_index(["symbol", "date"])
    daily_idx = daily_factors.set_index(["symbol", "date"])
    merged = daily_idx.join(carry_idx[["carry_score"]], how="left")
    merged.reset_index(inplace=True)
    print(f"    {len(merged):,} rows, carry coverage: {merged['carry_score'].notna().mean() * 100:.1f}%")

    # ── Z-scores ──
    print("[6] Computing cross-sectional z-scores...")
    factor_cols = ["carry_score", "mom20_score", "mom60_score", "mr5_score",
                   "vol_score", "oi_score", "poi_score"]
    for col in factor_cols:
        merged = cross_sectional_zscore(merged, col)

    # Filter: minimum 15 symbols per date for meaningful cross-section
    sym_per_date = merged.groupby("date")["symbol"].nunique()
    valid_dates = sym_per_date[sym_per_date >= 15].index
    merged = merged[merged["date"].isin(valid_dates)].copy()
    print(f"    {len(merged):,} rows, avg {merged.groupby('date')['symbol'].nunique().mean():.0f} symbols/date")

    # ── Run backtests ──
    print("\n" + "=" * 85)
    print("RUNNING BACKTESTS")
    print("=" * 85)

    all_results = []
    configs = []

    for wname, wdict in WEIGHT_PRESETS.items():
        # Long-only (best approach for Chinese commodities)
        for n_long in [3, 5, 8, 10]:
            for lev in [1.0, 2.0, 3.0, 5.0]:
                label = f"L{n_long}/S0_lev{lev:.0f}x_{wname}"
                configs.append((n_long, 0, lev, wdict, label))

        # Symmetric long-short
        for n_ls in [3, 5, 10]:
            for lev in [1.0, 3.0]:
                label = f"L{n_ls}/S{n_ls}_lev{lev:.0f}x_{wname}"
                configs.append((n_ls, n_ls, lev, wdict, label))

        # Asymmetric
        for n_long, n_short in [(10, 5), (8, 3)]:
            for lev in [1.0, 3.0]:
                label = f"L{n_long}/S{n_short}_lev{lev:.0f}x_{wname}"
                configs.append((n_long, n_short, lev, wdict, label))

    print(f"    Total configurations: {len(configs)}")

    for n_long, n_short, leverage, wdict, label in configs:
        res = run_backtest(merged, n_long, n_short, leverage, wdict,
                           start_date="2021-06-01")
        m = compute_metrics(res, label)
        if m:
            all_results.append(m)

    # ── Results summary ──
    print("\n" + "=" * 85)
    print("RESULTS SUMMARY (sorted by Sharpe)")
    print("=" * 85)

    all_results.sort(key=lambda x: x["sharpe"], reverse=True)

    print(f"\n{'#':<4} {'Label':<40} {'Sharpe':>7} {'AnnRet%':>9} {'WR%':>6} "
          f"{'MDD%':>8} {'Sortino':>8} {'Calmar':>8} {'N_wks':>6}")
    print("-" * 108)

    for i, m in enumerate(all_results[:50]):
        print(f"{i+1:<4} {m['label']:<40} {m['sharpe']:>7.2f} {m['annual_return_pct']:>9.2f} "
              f"{m['win_rate_pct']:>6.1f} {m['mdd_pct']:>8.2f} {m['sortino']:>8.2f} "
              f"{m['calmar']:>8.2f} {m['n_weeks']:>6}")

    # ── Top 10 detailed ──
    print("\n" + "=" * 85)
    print("TOP 10 DETAILED RESULTS")
    print("=" * 85)
    for i, m in enumerate(all_results[:10]):
        print(f"\n--- #{i+1} ---")
        print_metrics(m)

    # ── Best by category ──
    print("=" * 85)
    print("BEST BY CATEGORY")
    print("=" * 85)

    categories = {
        "Best Long-Only (Lev=1x)": lambda x: "S0_" in x["label"] and "_lev1x_" in x["label"],
        "Best Long-Only (Lev=2x)": lambda x: "S0_" in x["label"] and "_lev2x_" in x["label"],
        "Best Long-Only (Lev=3x)": lambda x: "S0_" in x["label"] and "_lev3x_" in x["label"],
        "Best L/S Symmetric (Lev=1x)": lambda x: "S0_" not in x["label"] and "_lev1x_" in x["label"]
            and x["label"].split("/")[0].split("L")[1] == x["label"].split("/")[1].split("_")[0],
        "Best L/S Symmetric (Lev=3x)": lambda x: "S0_" not in x["label"] and "_lev3x_" in x["label"]
            and x["label"].split("/")[0].split("L")[1] == x["label"].split("/")[1].split("_")[0],
    }

    for cat_name, cat_filter in categories.items():
        cat_results = [r for r in all_results if cat_filter(r) and r["sharpe"] > 0]
        if cat_results:
            best_cat = cat_results[0]
            print(f"\n{cat_name}:")
            print(f"  {best_cat['label']}")
            print(f"  Sharpe={best_cat['sharpe']:.2f}, AnnRet={best_cat['annual_return_pct']:.2f}%, "
                  f"WR={best_cat['win_rate_pct']:.1f}%, MDD={best_cat['mdd_pct']:.2f}%, "
                  f"TotRet={best_cat['total_return_pct']:.2f}%")

    # ── Detailed analysis for #1 ──
    if not all_results:
        return

    best = all_results[0]
    print("\n" + "=" * 85)
    print(f"DETAILED ANALYSIS: {best['label']}")
    print("=" * 85)

    parts = best["label"].split("_")
    ls_part = parts[0]
    n_long = int(ls_part.split("/")[0].replace("L", ""))
    n_short = int(ls_part.split("/")[1].replace("S", ""))
    lev_str = parts[1].replace("lev", "").replace("x", "")
    leverage = float(lev_str)
    wname = "_".join(parts[2:])
    wdict = WEIGHT_PRESETS[wname]

    best_res = run_backtest(merged, n_long, n_short, leverage, wdict,
                            start_date="2021-06-01")
    if best_res is None:
        return

    # Monthly
    best_res["month"] = pd.to_datetime(best_res["date"]).dt.to_period("M")
    monthly = best_res.groupby("month").agg(
        n_weeks=("port_ret", "count"),
        avg_ret=("port_ret", "mean"),
        total_ret=("port_ret", lambda x: np.prod(1 + x) - 1),
        win_rate=("port_ret", lambda x: (x > 0).mean()),
    )
    print("\nMonthly Performance:")
    print(f"{'Month':<10} {'Weeks':>6} {'AvgRet%':>9} {'TotRet%':>9} {'WR%':>6}")
    print("-" * 45)
    for idx, row in monthly.iterrows():
        print(f"{str(idx):<10} {int(row['n_weeks']):>6} {row['avg_ret']*100:>9.4f} "
              f"{row['total_ret']*100:>9.2f} {row['win_rate']*100:>6.1f}")

    # Yearly
    best_res["year"] = pd.to_datetime(best_res["date"]).dt.year
    yearly = best_res.groupby("year").agg(
        n_weeks=("port_ret", "count"),
        total_ret=("port_ret", lambda x: np.prod(1 + x) - 1),
        sharpe=("port_ret", lambda x: np.mean(x) / np.std(x, ddof=1) * np.sqrt(52) if len(x) > 1 else 0),
        win_rate=("port_ret", lambda x: (x > 0).mean()),
        mdd=("port_ret", lambda x: _calc_mdd(x)),
    )
    print("\nYearly Performance:")
    print(f"{'Year':<6} {'Weeks':>6} {'TotRet%':>9} {'Sharpe':>8} {'WR%':>6} {'MDD%':>8}")
    print("-" * 50)
    for idx, row in yearly.iterrows():
        print(f"{idx:<6} {int(row['n_weeks']):>6} {row['total_ret']*100:>9.2f} "
              f"{row['sharpe']:>8.2f} {row['win_rate']*100:>6.1f} {row['mdd']*100:>8.2f}")

    # Equity curve stats
    print("\n" + "-" * 50)
    cum = np.cumprod(1 + best_res["port_ret"].values)
    running_max = np.maximum.accumulate(cum)
    dd = (cum - running_max) / running_max

    worst_dd_idx = np.argmin(dd)
    in_dd = cum < running_max
    max_dd_dur = 0
    cur_dd_dur = 0
    for v in in_dd:
        if v:
            cur_dd_dur += 1
            max_dd_dur = max(max_dd_dur, cur_dd_dur)
        else:
            cur_dd_dur = 0

    print(f"  Final equity:       {cum[-1]:.4f}")
    print(f"  Worst DD date:      {best_res.iloc[worst_dd_idx]['date']}")
    print(f"  Worst DD:           {dd[worst_dd_idx]*100:.2f}%")
    print(f"  Max DD duration:    {max_dd_dur} weeks")
    print(f"  Positive weeks:     {(best_res['port_ret'] > 0).sum()} / {len(best_res)}")

    bw = np.argmax(best_res["port_ret"].values)
    ww = np.argmin(best_res["port_ret"].values)
    print(f"  Best week:          {best_res.iloc[bw]['date']} ({best_res.iloc[bw]['port_ret']*100:.2f}%)")
    print(f"  Worst week:         {best_res.iloc[ww]['date']} ({best_res.iloc[ww]['port_ret']*100:.2f}%)")

    # ── Holding frequency ──
    print("\n" + "=" * 85)
    print("HOLDING FREQUENCY")
    print("=" * 85)

    long_counts = defaultdict(int)
    short_counts = defaultdict(int)
    for _, row in best_res.iterrows():
        for sym in row.get("long_syms", []):
            long_counts[sym] += 1
        for sym in row.get("short_syms", []):
            short_counts[sym] += 1

    tw = len(best_res)
    print(f"\nMost frequently HELD LONG (out of {tw} weeks):")
    for sym, cnt in sorted(long_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"  {sym:<10} {cnt:>4} weeks ({cnt/tw*100:>5.1f}%)")

    if short_counts:
        print(f"\nMost frequently HELD SHORT (out of {tw} weeks):")
        for sym, cnt in sorted(short_counts.items(), key=lambda x: -x[1])[:15]:
            print(f"  {sym:<10} {cnt:>4} weeks ({cnt/tw*100:>5.1f}%)")

    # ── Individual factor comparison ──
    print("\n" + "=" * 85)
    print("INDIVIDUAL FACTOR ANALYSIS (Long-Only L5, Lev=1x)")
    print("=" * 85)

    single_factors = {
        "poi (Price-OI conf)": {"carry": 0, "mom20": 0, "mom60": 0, "mr5": 0, "vol": 0, "oi": 0, "poi": 1.0},
        "mom20 (20d trend)":   {"carry": 0, "mom20": 1.0, "mom60": 0, "mr5": 0, "vol": 0, "oi": 0, "poi": 0},
        "mom60 (60d trend)":   {"carry": 0, "mom20": 0, "mom60": 1.0, "mr5": 0, "vol": 0, "oi": 0, "poi": 0},
        "vol (low-vol pref)":  {"carry": 0, "mom20": 0, "mom60": 0, "mr5": 0, "vol": 1.0, "oi": 0, "poi": 0},
        "oi (OI trend)":       {"carry": 0, "mom20": 0, "mom60": 0, "mr5": 0, "vol": 0, "oi": 1.0, "poi": 0},
        "carry (term struct)": {"carry": 1.0, "mom20": 0, "mom60": 0, "mr5": 0, "vol": 0, "oi": 0, "poi": 0},
        "mr5 (contrarian)":    {"carry": 0, "mom20": 0, "mom60": 0, "mr5": 1.0, "vol": 0, "oi": 0, "poi": 0},
    }

    print(f"\n{'Factor':<22} {'Sharpe':>7} {'AnnRet%':>9} {'WR%':>6} {'MDD%':>8} {'TotRet%':>9}")
    print("-" * 68)
    for fname, fw in single_factors.items():
        res = run_backtest(merged, 5, 0, 1.0, fw, start_date="2021-06-01")
        m = compute_metrics(res, fname)
        if m:
            print(f"{fname:<22} {m['sharpe']:>7.2f} {m['annual_return_pct']:>9.2f} "
                  f"{m['win_rate_pct']:>6.1f} {m['mdd_pct']:>8.2f} {m['total_return_pct']:>9.2f}")

    print("\n" + "=" * 85)
    print("BACKTEST V94 COMPLETE")
    print("=" * 85)


if __name__ == "__main__":
    main()
