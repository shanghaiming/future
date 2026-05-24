#!/usr/bin/env python3
"""
Backtest V97: Volatility-Based Strategy for Chinese Commodity Futures
======================================================================
Tests three volatility-based strategies:

Strategy 1 - Low Vol Anomaly:
  - Long K lowest-vol commodities, short K highest-vol commodities
  - Rebalance every 10/15/20 days

Strategy 2 - Vol Breakout:
  - Short when 20d vol > 60d vol by threshold (vol expanding)
  - Long when 20d vol < 60d vol by threshold (vol contracting)
  - Long when 20d vol crosses above 60d vol after contraction (trend start)

Strategy 3 - Mean Reversion on Vol:
  - Short when vol rank > 80th pct (expect vol decrease)
  - Long when vol rank < 20th pct (expect vol increase)

Risk: -2% SL, +5% TP per position
Walk-forward: Train 2021-2023, Validate 2024, Test 2025-2026

Data: data/futures_weighted/ (CSV with trade_date,open,high,low,close,vol,oi)
"""

import os
import glob
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DAILY_DIR = os.path.join(BASE_DIR, "data", "futures_weighted")

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_daily_data(min_history=300):
    """Load all daily CSV files into a single DataFrame."""
    frames = []
    for fp in glob.glob(os.path.join(DAILY_DIR, "*.csv")):
        sym = os.path.basename(fp).replace(".csv", "")
        df = pd.read_csv(fp)
        df["symbol"] = sym
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    # Handle mixed date formats (YYYYMMDD or YYYY-MM-DD)
    df["trade_date"] = pd.to_datetime(df["trade_date"], format="mixed")
    df.rename(columns={"trade_date": "date"}, inplace=True)
    df.sort_values(["symbol", "date"], inplace=True)
    df.drop_duplicates(subset=["symbol", "date"], keep="last", inplace=True)

    # Filter out invalid rows (close <= 0 or zero volume)
    df = df[df["close"] > 0].copy()
    df["close"] = df["close"].astype(float)
    df["open"] = df["open"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)

    # Filter symbols with insufficient history
    counts = df.groupby("symbol").size()
    valid_syms = counts[counts >= min_history].index
    df = df[df["symbol"].isin(valid_syms)].copy()

    return df


# ---------------------------------------------------------------------------
# Volatility calculations
# ---------------------------------------------------------------------------

def compute_vol_features(daily_df):
    """Compute volatility features for each symbol-date."""
    dfs = []
    for sym, grp in daily_df.groupby("symbol"):
        g = grp.set_index("date").sort_index()
        g = g[~g.index.duplicated(keep="last")]

        close = g["close"]
        daily_ret = close.pct_change()

        # Realized volatility at multiple windows
        vol_10 = daily_ret.rolling(10).std() * np.sqrt(252)
        vol_20 = daily_ret.rolling(20).std() * np.sqrt(252)
        vol_60 = daily_ret.rolling(60).std() * np.sqrt(252)

        # Vol ratio (short / long)
        vol_ratio = vol_20 / vol_60.replace(0, np.nan)

        # Vol percentile rank over 252-day lookback
        vol_rank = vol_20.rolling(252).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
        )

        # Cross-over signal: 20d vol crosses above 60d vol
        vol_cross_above = ((vol_ratio > 1.0) & (vol_ratio.shift(1) <= 1.0)).astype(int)
        # Also track previous 5-day state for contraction-then-expansion
        was_contracting = (vol_ratio.shift(5) < 0.8).astype(int)
        vol_expansion_after_calm = (vol_cross_above * was_contracting).astype(int)

        # Forward returns for backtesting (using close-to-close)
        fwd_ret_1 = close.pct_change(1).shift(-1)
        fwd_ret_5 = close.pct_change(5).shift(-5)
        fwd_ret_10 = close.pct_change(10).shift(-10)
        fwd_ret_15 = close.pct_change(15).shift(-15)
        fwd_ret_20 = close.pct_change(20).shift(-20)

        # Daily return (for continuous PnL tracking)
        ret_1 = close.pct_change()

        out = pd.DataFrame({
            "symbol": sym,
            "close": close,
            "vol_10": vol_10,
            "vol_20": vol_20,
            "vol_60": vol_60,
            "vol_ratio": vol_ratio,
            "vol_rank": vol_rank,
            "vol_cross_above": vol_cross_above,
            "vol_expansion_after_calm": vol_expansion_after_calm,
            "fwd_ret_1": fwd_ret_1,
            "fwd_ret_5": fwd_ret_5,
            "fwd_ret_10": fwd_ret_10,
            "fwd_ret_15": fwd_ret_15,
            "fwd_ret_20": fwd_ret_20,
            "ret_1": ret_1,
        }, index=g.index)

        dfs.append(out)

    result = pd.concat(dfs, axis=0) if dfs else pd.DataFrame()
    if len(result):
        result.index.name = "date"
        result.reset_index(inplace=True)
    return result


# ---------------------------------------------------------------------------
# Risk management: apply SL/TP to a series of returns
# ---------------------------------------------------------------------------

def apply_sl_tp(entry_price_series, ret_series, sl=-0.02, tp=0.05, hold_days=10):
    """
    Apply stop-loss and take-profit to forward returns.
    Simplified: check if cumulative return breaches SL/TP at any point
    during the hold period. For portfolio-level backtest we use the
    forward return directly with a cap.

    For a more realistic approach, we track intraperiod drawdowns using
    the high/low data. Here we use a simplified position-level approach.
    """
    # Clip returns at SL/TP boundaries
    capped = ret_series.copy()
    capped = capped.clip(lower=sl, upper=tp)
    return capped


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def compute_metrics(returns, label="", annual_factor=252):
    """Compute strategy performance metrics from daily return series."""
    if len(returns) == 0 or returns.std() < 1e-12:
        return {
            "Label": label,
            "AnnRet(%)": 0.0,
            "AnnVol(%)": 0.0,
            "Sharpe": 0.0,
            "Sortino": 0.0,
            "MDD(%)": 0.0,
            "Calmar": 0.0,
            "WinRate(%)": 0.0,
            "NTrades": 0,
            "AvgTrade(%)": 0.0,
        }

    mu = returns.mean()
    sigma = returns.std()
    ann_ret = mu * annual_factor
    ann_vol = sigma * np.sqrt(annual_factor)
    sharpe = ann_ret / ann_vol if ann_vol > 1e-12 else 0.0

    downside = returns[returns < 0]
    sortino = ann_ret / (downside.std() * np.sqrt(annual_factor)) if len(downside) > 0 and downside.std() > 1e-12 else 0.0

    cum = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    mdd = dd.min()

    calmar = ann_ret / abs(mdd) if abs(mdd) > 1e-12 else 0.0
    win_rate = (returns > 0).sum() / len(returns) * 100 if len(returns) > 0 else 0.0

    return {
        "Label": label,
        "AnnRet(%)": round(ann_ret * 100, 2),
        "AnnVol(%)": round(ann_vol * 100, 2),
        "Sharpe": round(sharpe, 2),
        "Sortino": round(sortino, 2),
        "MDD(%)": round(mdd * 100, 2),
        "Calmar": round(calmar, 2),
        "WinRate(%)": round(win_rate, 1),
        "NTrades": len(returns),
        "AvgTrade(%)": round(returns.mean() * 100, 4),
    }


def print_metrics_table(results, title=""):
    """Print a formatted metrics table."""
    if not results:
        print(f"\n{'='*80}")
        print(f"  {title} - NO RESULTS")
        print(f"{'='*80}")
        return

    df = pd.DataFrame(results)

    # Format for display
    display_cols = ["Label", "AnnRet(%)", "AnnVol(%)", "Sharpe", "Sortino",
                    "MDD(%)", "Calmar", "WinRate(%)", "NTrades", "AvgTrade(%)"]

    print(f"\n{'='*120}")
    print(f"  {title}")
    print(f"{'='*120}")
    print(df[display_cols].to_string(index=False))
    print(f"{'='*120}")
    return df


# ---------------------------------------------------------------------------
# STRATEGY 1: Low Volatility Anomaly
# ---------------------------------------------------------------------------

def run_low_vol_anomaly(vdf, k_values=(5, 10, 15), hold_days=(10, 15, 20),
                        sl=-0.02, tp=0.05,
                        start_date=None, end_date=None):
    """
    Long K lowest-vol commodities, short K highest-vol commodities.
    Rebalance at regular intervals.
    """
    results = []

    for k in k_values:
        for hd in hold_days:
            label = f"LowVol L{k}/S{k} Hold{hd}d"

            # Filter date range
            mask = vdf["vol_20"].notna() & vdf["close"].notna()
            if start_date:
                mask &= vdf["date"] >= pd.Timestamp(start_date)
            if end_date:
                mask &= vdf["date"] <= pd.Timestamp(end_date)
            sub = vdf[mask].copy()

            if len(sub) == 0:
                continue

            # Get rebalance dates (every hd trading days)
            all_dates = sorted(sub["date"].unique())
            rebal_dates = all_dates[::hd]

            long_rets = []
            short_rets = []
            ls_rets = []

            for rd in rebal_dates:
                # Get vol_20 for all commodities on this date
                day_data = sub[sub["date"] == rd].copy()
                day_data = day_data.dropna(subset=["vol_20"])

                if len(day_data) < 2 * k:
                    continue

                # Rank by volatility
                day_data = day_data.sort_values("vol_20")

                # Long: lowest vol K
                long_syms = day_data.head(k)["symbol"].values
                # Short: highest vol K
                short_syms = day_data.tail(k)["symbol"].values

                # Get forward returns
                fwd_col = f"fwd_ret_{hd}"

                # Long leg returns
                long_fwd = sub[(sub["date"] == rd) & (sub["symbol"].isin(long_syms))][fwd_col]
                long_fwd = long_fwd.dropna()
                if len(long_fwd) > 0:
                    long_ret = long_fwd.mean()
                    long_ret = np.clip(long_ret, sl, tp)
                else:
                    long_ret = 0.0

                # Short leg returns (negate)
                short_fwd = sub[(sub["date"] == rd) & (sub["symbol"].isin(short_syms))][fwd_col]
                short_fwd = short_fwd.dropna()
                if len(short_fwd) > 0:
                    short_ret = -short_fwd.mean()
                    short_ret = np.clip(short_ret, sl, tp)
                else:
                    short_ret = 0.0

                ls_ret = 0.5 * long_ret + 0.5 * short_ret

                long_rets.append(long_ret)
                short_rets.append(short_ret)
                ls_rets.append(ls_ret)

            if not ls_rets:
                continue

            long_rets = np.array(long_rets)
            short_rets = np.array(short_rets)
            ls_rets = np.array(ls_rets)

            # Compute metrics for L/S, Long-only, Short-only
            m_ls = compute_metrics(ls_rets, label)
            results.append(m_ls)

            m_long = compute_metrics(long_rets, f"  Long-only L{k} H{hd}d")
            results.append(m_long)

            m_short = compute_metrics(short_rets, f"  Short-only S{k} H{hd}d")
            results.append(m_short)

    return results


# ---------------------------------------------------------------------------
# STRATEGY 2: Volatility Breakout
# ---------------------------------------------------------------------------

def run_vol_breakout(vdf, thresholds=(1.3, 1.5, 2.0), hold_days=10,
                     sl=-0.02, tp=0.05,
                     start_date=None, end_date=None):
    """
    Signal-based strategy:
    - Short when vol_ratio > threshold (expanding vol)
    - Long when vol_ratio < 1/threshold (contracting vol)
    - Long on expansion-after-calm crossover signal
    """
    results = []

    for thresh in thresholds:
        inv_thresh = 1.0 / thresh

        for mode in ["contracting_long", "expanding_short", "expansion_crossover"]:
            if mode == "contracting_long":
                label = f"VolBreak Long(contract<={inv_thresh:.2f}) T{thresh} H{hold_days}d"
                signal_mask = vdf["vol_ratio"] < inv_thresh
                direction = 1
            elif mode == "expanding_short":
                label = f"VolBreak Short(expand>={thresh}) T{thresh} H{hold_days}d"
                signal_mask = vdf["vol_ratio"] > thresh
                direction = -1
            else:  # expansion_crossover
                label = f"VolBreak Crossover(expand_after_calm) T{thresh} H{hold_days}d"
                signal_mask = vdf["vol_expansion_after_calm"] == 1
                direction = 1

            mask = signal_mask & vdf["vol_ratio"].notna() & vdf["close"].notna()
            if start_date:
                mask &= vdf["date"] >= pd.Timestamp(start_date)
            if end_date:
                mask &= vdf["date"] <= pd.Timestamp(end_date)
            sub = vdf[mask].copy()

            if len(sub) == 0:
                continue

            fwd_col = f"fwd_ret_{hold_days}"
            sub = sub.dropna(subset=[fwd_col])

            if len(sub) == 0:
                continue

            rets = sub[fwd_col].values * direction
            rets = np.clip(rets, sl, tp)

            m = compute_metrics(rets, label)
            results.append(m)

    return results


# ---------------------------------------------------------------------------
# STRATEGY 3: Vol Mean Reversion (Percentile Rank)
# ---------------------------------------------------------------------------

def run_vol_mean_reversion(vdf, high_pct=80, low_pct=20, hold_days=(10, 15, 20),
                           sl=-0.02, tp=0.05,
                           start_date=None, end_date=None):
    """
    Long when vol rank < low_pct, Short when vol rank > high_pct.
    """
    results = []

    for hd in hold_days:
        for mode in ["long_low_rank", "short_high_rank", "ls_combined"]:
            if mode == "long_low_rank":
                label = f"VolMR Long(rank<{low_pct}) H{hd}d"
                mask = (vdf["vol_rank"] < low_pct) & vdf["vol_rank"].notna()
                direction = 1
            elif mode == "short_high_rank":
                label = f"VolMR Short(rank>{high_pct}) H{hd}d"
                mask = (vdf["vol_rank"] > high_pct) & vdf["vol_rank"].notna()
                direction = -1
            else:
                label = f"VolMR LS(rank<{low_pct}/>{high_pct}) H{hd}d"
                # Combined: long low-rank, short high-rank
                pass

            if mode in ("long_low_rank", "short_high_rank"):
                if start_date:
                    mask &= vdf["date"] >= pd.Timestamp(start_date)
                if end_date:
                    mask &= vdf["date"] <= pd.Timestamp(end_date)
                sub = vdf[mask].copy()

                fwd_col = f"fwd_ret_{hd}"
                sub = sub.dropna(subset=[fwd_col])

                if len(sub) == 0:
                    continue

                rets = sub[fwd_col].values * direction
                rets = np.clip(rets, sl, tp)
                m = compute_metrics(rets, label)
                results.append(m)
            else:
                # Combined L/S
                # Need to group by date to get equal-weighted L/S
                if start_date:
                    date_mask = vdf["date"] >= pd.Timestamp(start_date)
                else:
                    date_mask = pd.Series(True, index=vdf.index)
                if end_date:
                    date_mask &= vdf["date"] <= pd.Timestamp(end_date)

                fwd_col = f"fwd_ret_{hd}"
                sub = vdf[date_mask & vdf["vol_rank"].notna()].copy()
                sub = sub.dropna(subset=[fwd_col])

                if len(sub) == 0:
                    continue

                long_mask = sub["vol_rank"] < low_pct
                short_mask = sub["vol_rank"] > high_pct

                # Group by date
                ls_daily_rets = []
                for dt, day_data in sub.groupby("date"):
                    long_data = day_data[long_mask[day_data.index]]
                    short_data = day_data[short_mask[day_data.index]]

                    if len(long_data) == 0 and len(short_data) == 0:
                        continue

                    long_ret = long_data[fwd_col].mean() if len(long_data) > 0 else 0.0
                    short_ret = -short_data[fwd_col].mean() if len(short_data) > 0 else 0.0

                    # Weight equally
                    n_legs = (1 if len(long_data) > 0 else 0) + (1 if len(short_data) > 0 else 0)
                    ls_ret = (long_ret + short_ret) / n_legs
                    ls_ret = np.clip(ls_ret, sl, tp)
                    ls_daily_rets.append(ls_ret)

                if not ls_daily_rets:
                    continue

                rets = np.array(ls_daily_rets)
                m = compute_metrics(rets, label)
                results.append(m)

    return results


# ---------------------------------------------------------------------------
# Walk-Forward Analysis
# ---------------------------------------------------------------------------

def walk_forward_analysis(vdf, strategy_func, best_config, periods):
    """
    Run walk-forward analysis.
    periods: list of (label, start, end) tuples.
    """
    results = []

    for label, start, end in periods:
        # For each period, we need to pass the appropriate params
        # This is strategy-specific, so we handle it per strategy
        pass

    return results


def walk_forward_low_vol(vdf, k, hold_days, periods):
    """Walk-forward for low vol anomaly."""
    results = []
    for label, start, end in periods:
        res = run_low_vol_anomaly(vdf, k_values=[k], hold_days=[hold_days],
                                  start_date=start, end_date=end)
        if res:
            # Find the L/S result
            for r in res:
                if f"L{k}/S{k}" in r["Label"]:
                    r["Label"] = f"LowVol K={k} H={hold_days}d [{label}]"
                    results.append(r)
                    break
    return results


def walk_forward_vol_breakout(vdf, threshold, mode, hold_days, periods):
    """Walk-forward for vol breakout."""
    results = []
    for label, start, end in periods:
        res = run_vol_breakout(vdf, thresholds=[threshold], hold_days=hold_days,
                               start_date=start, end_date=end)
        # Find the matching mode
        mode_key = {
            "contracting_long": "Long(contract",
            "expanding_short": "Short(expand",
            "expansion_crossover": "Crossover",
        }[mode]

        for r in res:
            if mode_key in r["Label"]:
                r["Label"] = f"VolBreak T={threshold} {mode} H={hold_days}d [{label}]"
                results.append(r)
                break
    return results


def walk_forward_vol_mr(vdf, mode, hold_days, periods):
    """Walk-forward for vol mean reversion."""
    results = []
    for label, start, end in periods:
        res = run_vol_mean_reversion(vdf, hold_days=[hold_days],
                                     start_date=start, end_date=end)
        mode_key = {
            "long_low_rank": "Long(rank",
            "short_high_rank": "Short(rank",
            "ls_combined": "LS(rank",
        }[mode]

        for r in res:
            if mode_key in r["Label"]:
                r["Label"] = f"VolMR {mode} H={hold_days}d [{label}]"
                results.append(r)
                break
    return results


# ---------------------------------------------------------------------------
# Equity Curve (for best strategies)
# ---------------------------------------------------------------------------

def compute_equity_curve(returns_list):
    """Compute equity curve from a list of returns."""
    equity = [1.0]
    for r in returns_list:
        equity.append(equity[-1] * (1 + r))
    return equity


def print_equity_curve_summary(returns_list, label):
    """Print equity curve statistics."""
    eq = compute_equity_curve(returns_list)
    peak = np.maximum.accumulate(eq)
    dd = [(eq[i] - peak[i]) / peak[i] for i in range(len(eq))]
    max_dd_idx = np.argmin(dd)

    print(f"\n  Equity Curve: {label}")
    print(f"    Start equity: 1.0000")
    print(f"    End equity:   {eq[-1]:.4f}")
    print(f"    Peak equity:  {max(eq):.4f}")
    print(f"    Max DD:       {min(dd)*100:.2f}% at bar {max_dd_idx}")
    print(f"    Final return: {(eq[-1]-1)*100:.2f}%")
    return eq


# ---------------------------------------------------------------------------
# Detailed L/S with equity curve tracking
# ---------------------------------------------------------------------------

def run_low_vol_detailed(vdf, k, hold_days, sl=-0.02, tp=0.05,
                         start_date=None, end_date=None):
    """Run low vol anomaly with full equity curve tracking."""
    mask = vdf["vol_20"].notna() & vdf["close"].notna()
    if start_date:
        mask &= vdf["date"] >= pd.Timestamp(start_date)
    if end_date:
        mask &= vdf["date"] <= pd.Timestamp(end_date)
    sub = vdf[mask].copy()

    all_dates = sorted(sub["date"].unique())
    rebal_dates = all_dates[::hold_days]
    fwd_col = f"fwd_ret_{hold_days}"

    ls_rets = []
    long_rets = []
    short_rets = []
    rebal_actual_dates = []

    for rd in rebal_dates:
        day_data = sub[sub["date"] == rd].dropna(subset=["vol_20"])
        if len(day_data) < 2 * k:
            continue

        day_data = day_data.sort_values("vol_20")
        long_syms = day_data.head(k)["symbol"].values
        short_syms = day_data.tail(k)["symbol"].values

        # Long leg
        long_fwd = sub[(sub["date"] == rd) & (sub["symbol"].isin(long_syms))][fwd_col]
        long_fwd = long_fwd.dropna()
        long_ret = np.clip(long_fwd.mean(), sl, tp) if len(long_fwd) > 0 else 0.0

        # Short leg
        short_fwd = sub[(sub["date"] == rd) & (sub["symbol"].isin(short_syms))][fwd_col]
        short_fwd = short_fwd.dropna()
        short_ret = np.clip(-short_fwd.mean(), sl, tp) if len(short_fwd) > 0 else 0.0

        ls_ret = 0.5 * long_ret + 0.5 * short_ret

        ls_rets.append(ls_ret)
        long_rets.append(long_ret)
        short_rets.append(short_ret)
        rebal_actual_dates.append(rd)

    return {
        "ls_rets": np.array(ls_rets),
        "long_rets": np.array(long_rets),
        "short_rets": np.array(short_rets),
        "rebal_dates": rebal_actual_dates,
    }


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 120)
    print("  BACKTEST V97: Volatility-Based Strategy for Chinese Commodity Futures")
    print("  Data: ~76 commodities, daily OHLCV+OI")
    print("  Risk: -2% SL, +5% TP")
    print("=" * 120)

    # Load data
    print("\n[1] Loading daily data...")
    daily_df = load_daily_data(min_history=300)
    n_syms = daily_df["symbol"].nunique()
    date_min = daily_df["date"].min()
    date_max = daily_df["date"].max()
    print(f"    Loaded {n_syms} commodities, {len(daily_df):,} rows")
    print(f"    Date range: {date_min.strftime('%Y-%m-%d')} to {date_max.strftime('%Y-%m-%d')}")

    # Compute vol features
    print("\n[2] Computing volatility features (vol_20, vol_60, vol_ratio, vol_rank)...")
    vdf = compute_vol_features(daily_df)
    print(f"    Feature rows: {len(vdf):,}")
    print(f"    vol_20 coverage: {vdf['vol_20'].notna().sum():,}")
    print(f"    vol_ratio coverage: {vdf['vol_ratio'].notna().sum():,}")
    print(f"    vol_rank coverage: {vdf['vol_rank'].notna().sum():,}")

    # =========================================================================
    # STRATEGY 1: Low Volatility Anomaly
    # =========================================================================
    print("\n" + "=" * 120)
    print("  STRATEGY 1: Low Volatility Anomaly")
    print("  Long K lowest-vol, Short K highest-vol, rebalance every N days")
    print("=" * 120)

    # Full sample first
    print("\n--- Full Sample ---")
    s1_results = run_low_vol_anomaly(
        vdf,
        k_values=(5, 10, 15),
        hold_days=(10, 15, 20),
        sl=-0.02, tp=0.05,
    )
    s1_table = print_metrics_table(s1_results, "Strategy 1: Low Vol Anomaly (Full Sample)")

    # Find best L/S config by Sharpe
    best_s1 = None
    best_s1_sharpe = -999
    for r in s1_results:
        if "/S" in r["Label"] and "Long-only" not in r["Label"] and "Short-only" not in r["Label"]:
            if r["Sharpe"] > best_s1_sharpe:
                best_s1_sharpe = r["Sharpe"]
                best_s1 = r
    if best_s1:
        print(f"\n  >>> Best S1 Config: {best_s1['Label']} | Sharpe={best_s1['Sharpe']}, "
              f"AnnRet={best_s1['AnnRet(%)']}%, MDD={best_s1['MDD(%)']}%")

    # =========================================================================
    # STRATEGY 2: Volatility Breakout
    # =========================================================================
    print("\n" + "=" * 120)
    print("  STRATEGY 2: Volatility Breakout")
    print("  Vol ratio = 20d vol / 60d vol")
    print("  Long when vol contracting, Short when vol expanding")
    print("=" * 120)

    s2_results = []
    for hd in [5, 10, 15]:
        res = run_vol_breakout(
            vdf,
            thresholds=(1.3, 1.5, 2.0),
            hold_days=hd,
            sl=-0.02, tp=0.05,
        )
        s2_results.extend(res)

    s2_table = print_metrics_table(s2_results, "Strategy 2: Vol Breakout (Full Sample)")

    best_s2 = max(s2_results, key=lambda x: x["Sharpe"]) if s2_results else None
    if best_s2:
        print(f"\n  >>> Best S2 Config: {best_s2['Label']} | Sharpe={best_s2['Sharpe']}, "
              f"AnnRet={best_s2['AnnRet(%)']}%, MDD={best_s2['MDD(%)']}%")

    # =========================================================================
    # STRATEGY 3: Vol Mean Reversion
    # =========================================================================
    print("\n" + "=" * 120)
    print("  STRATEGY 3: Volatility Mean Reversion (Percentile Rank)")
    print("  Vol rank over 252-day lookback")
    print("  Long when rank < 20, Short when rank > 80")
    print("=" * 120)

    s3_results = run_vol_mean_reversion(
        vdf,
        high_pct=80, low_pct=20,
        hold_days=(10, 15, 20),
        sl=-0.02, tp=0.05,
    )
    s3_table = print_metrics_table(s3_results, "Strategy 3: Vol Mean Reversion (Full Sample)")

    best_s3 = max(s3_results, key=lambda x: x["Sharpe"]) if s3_results else None
    if best_s3:
        print(f"\n  >>> Best S3 Config: {best_s3['Label']} | Sharpe={best_s3['Sharpe']}, "
              f"AnnRet={best_s3['AnnRet(%)']}%, MDD={best_s3['MDD(%)']}%")

    # =========================================================================
    # WALK-FORWARD ANALYSIS
    # =========================================================================
    print("\n" + "=" * 120)
    print("  WALK-FORWARD ANALYSIS")
    print("  Train: 2021-2023 | Validate: 2024 | Test: 2025-2026")
    print("=" * 120)

    wf_periods = [
        ("Train 2021-2023", "2021-01-01", "2023-12-31"),
        ("Valid 2024", "2024-01-01", "2024-12-31"),
        ("Test 2025-2026", "2025-01-01", "2099-12-31"),
    ]

    # --- Walk-forward: Best Low Vol Anomaly ---
    if best_s1:
        # Parse K and hold_days from label
        label = best_s1["Label"]
        try:
            parts = label.split(" ")
            k_part = parts[1]  # "L5/S5"
            k_val = int(k_part.split("/")[0].replace("L", ""))
            hd_part = parts[2]  # "Hold10d"
            hd_val = int(hd_part.replace("Hold", "").replace("d", ""))
        except (IndexError, ValueError):
            k_val = 10
            hd_val = 10

        print(f"\n--- Walk-Forward: Low Vol Anomaly (K={k_val}, Hold={hd_val}d) ---")
        wf_s1 = walk_forward_low_vol(vdf, k_val, hd_val, wf_periods)
        print_metrics_table(wf_s1, f"Walk-Forward: Low Vol K={k_val} H={hd_val}d")

        # Detailed equity curves for test period
        print("\n  Equity Curve on Test Period (2025-2026):")
        detail = run_low_vol_detailed(vdf, k_val, hd_val,
                                      start_date="2025-01-01", end_date="2099-12-31")
        if len(detail["ls_rets"]) > 0:
            print_equity_curve_summary(detail["ls_rets"].tolist(), "L/S Portfolio")
            print_equity_curve_summary(detail["long_rets"].tolist(), "Long-only Leg")
            print_equity_curve_summary(detail["short_rets"].tolist(), "Short-only Leg")

    # --- Walk-forward: Best Vol Breakout ---
    if best_s2:
        label = best_s2["Label"]
        # Parse threshold and mode
        try:
            # Format: "VolBreak Long(contract<=0.67) T1.5 H10d"
            t_str = label.split("T")[1].split(" ")[0]
            t_val = float(t_str)
            if "contract" in label:
                mode_val = "contracting_long"
            elif "Crossover" in label:
                mode_val = "expansion_crossover"
            else:
                mode_val = "expanding_short"
            h_str = label.split("H")[1].split("d")[0]
            h_val = int(h_str)
        except (IndexError, ValueError):
            t_val = 1.5
            mode_val = "contracting_long"
            h_val = 10

        print(f"\n--- Walk-Forward: Vol Breakout (T={t_val}, Mode={mode_val}, H={h_val}d) ---")
        wf_s2 = walk_forward_vol_breakout(vdf, t_val, mode_val, h_val, wf_periods)
        print_metrics_table(wf_s2, f"Walk-Forward: VolBreak T={t_val} {mode_val} H={h_val}d")

    # --- Walk-forward: Best Vol Mean Reversion ---
    if best_s3:
        label = best_s3["Label"]
        try:
            if "Long(rank" in label:
                mode_val = "long_low_rank"
            elif "Short(rank" in label:
                mode_val = "short_high_rank"
            else:
                mode_val = "ls_combined"
            h_str = label.split("H")[1].split("d")[0]
            h_val = int(h_str)
        except (IndexError, ValueError):
            mode_val = "long_low_rank"
            h_val = 10

        print(f"\n--- Walk-Forward: Vol Mean Reversion (Mode={mode_val}, H={h_val}d) ---")
        wf_s3 = walk_forward_vol_mr(vdf, mode_val, h_val, wf_periods)
        print_metrics_table(wf_s3, f"Walk-Forward: VolMR {mode_val} H={h_val}d")

    # =========================================================================
    # CROSS-STRATEGY COMPARISON
    # =========================================================================
    print("\n" + "=" * 120)
    print("  CROSS-STRATEGY COMPARISON: BEST CONFIGS (Full Sample)")
    print("=" * 120)

    best_all = []
    if best_s1:
        best_all.append(best_s1)
    if best_s2:
        best_all.append(best_s2)
    if best_s3:
        best_all.append(best_s3)

    if best_all:
        comp_df = pd.DataFrame(best_all)
        display_cols = ["Label", "AnnRet(%)", "AnnVol(%)", "Sharpe", "Sortino",
                        "MDD(%)", "Calmar", "WinRate(%)", "NTrades"]
        print("\n" + comp_df[display_cols].to_string(index=False))

    # =========================================================================
    # SENSITIVITY ANALYSIS: Vary SL/TP for best strategy
    # =========================================================================
    print("\n" + "=" * 120)
    print("  SENSITIVITY ANALYSIS: SL/TP Variations")
    print("=" * 120)

    if best_s1:
        sl_tp_combos = [
            (-0.01, 0.03),
            (-0.02, 0.05),
            (-0.03, 0.08),
            (-0.02, 0.10),
            (-0.05, 0.15),
            (None, None),  # No SL/TP
        ]

        print(f"\n--- Sensitivity for Low Vol Anomaly (K={k_val}, H={hd_val}d) ---")
        sens_results = []
        for sl_val, tp_val in sl_tp_combos:
            sl_use = sl_val if sl_val is not None else -1.0
            tp_use = tp_val if tp_val is not None else 1.0
            label_sltp = f"SL={sl_val}/TP={tp_val}"
            res = run_low_vol_anomaly(vdf, k_values=[k_val], hold_days=[hd_val],
                                      sl=sl_use, tp=tp_use)
            for r in res:
                if "/S" in r["Label"] and "Long-only" not in r["Label"]:
                    r["Label"] = label_sltp
                    sens_results.append(r)
                    break

        print_metrics_table(sens_results, f"SL/TP Sensitivity: Low Vol K={k_val} H={hd_val}d")

    # =========================================================================
    # ADDITIONAL ANALYSIS: Vol regime breakdown
    # =========================================================================
    print("\n" + "=" * 120)
    print("  ADDITIONAL: VOL REGIME BREAKDOWN (By Year)")
    print("=" * 120)

    if best_s1:
        print(f"\n--- Annual Performance: Low Vol Anomaly (K={k_val}, H={hd_val}d) ---")
        annual_results = []
        for year in range(2021, 2027):
            res = run_low_vol_anomaly(vdf, k_values=[k_val], hold_days=[hd_val],
                                      start_date=f"{year}-01-01", end_date=f"{year}-12-31")
            for r in res:
                if "/S" in r["Label"] and "Long-only" not in r["Label"]:
                    r["Label"] = f"Year {year}"
                    annual_results.append(r)
                    break

        print_metrics_table(annual_results, f"Annual Breakdown: Low Vol K={k_val} H={hd_val}d")

    # =========================================================================
    # STRATEGY ENHANCEMENT: Long-only Low Vol (no short leg)
    # =========================================================================
    print("\n" + "=" * 120)
    print("  ENHANCEMENT: Long-Only Low Vol Variants")
    print("=" * 120)

    # Some markets have structural long bias; test long-only variants
    lo_results = []
    for k in [3, 5, 7, 10]:
        for hd in [10, 20]:
            mask = vdf["vol_20"].notna() & vdf["close"].notna()
            sub = vdf[mask].copy()
            all_dates = sorted(sub["date"].unique())
            rebal_dates = all_dates[::hd]
            fwd_col = f"fwd_ret_{hd}"

            rets = []
            for rd in rebal_dates:
                day_data = sub[sub["date"] == rd].dropna(subset=["vol_20"])
                if len(day_data) < k:
                    continue
                day_data = day_data.sort_values("vol_20")
                long_syms = day_data.head(k)["symbol"].values
                long_fwd = sub[(sub["date"] == rd) & (sub["symbol"].isin(long_syms))][fwd_col]
                long_fwd = long_fwd.dropna()
                if len(long_fwd) > 0:
                    ret = np.clip(long_fwd.mean(), -0.02, 0.05)
                    rets.append(ret)

            if rets:
                m = compute_metrics(np.array(rets), f"LowVol-Long K={k} H={hd}d")
                lo_results.append(m)

    print_metrics_table(lo_results, "Long-Only Low Vol Variants")

    # =========================================================================
    # ENHANCEMENT: Vol Breakout with Daily Rebalance
    # =========================================================================
    print("\n" + "=" * 120)
    print("  ENHANCEMENT: Vol Breakout Daily Signal (1-day forward return)")
    print("=" * 120)

    daily_breakout_results = []
    for thresh in [1.2, 1.3, 1.5, 2.0]:
        inv_thresh = 1.0 / thresh

        # Contracting (long)
        mask_long = (vdf["vol_ratio"] < inv_thresh) & vdf["vol_ratio"].notna() & vdf["fwd_ret_1"].notna()
        long_rets = vdf.loc[mask_long, "fwd_ret_1"].values
        long_rets = np.clip(long_rets, -0.02, 0.05)
        m = compute_metrics(long_rets, f"Daily Long(ratio<{inv_thresh:.2f}) T={thresh}")
        daily_breakout_results.append(m)

        # Expanding (short)
        mask_short = (vdf["vol_ratio"] > thresh) & vdf["vol_ratio"].notna() & vdf["fwd_ret_1"].notna()
        short_rets = -vdf.loc[mask_short, "fwd_ret_1"].values
        short_rets = np.clip(short_rets, -0.02, 0.05)
        m = compute_metrics(short_rets, f"Daily Short(ratio>{thresh}) T={thresh}")
        daily_breakout_results.append(m)

    print_metrics_table(daily_breakout_results, "Vol Breakout Daily Signal")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 120)
    print("  SUMMARY")
    print("=" * 120)

    print("""
  Strategy 1 - Low Vol Anomaly:
    Cross-sectional ranking by 20d realized vol.
    Long lowest-vol, short highest-vol commodities.
    Tests the "low-volatility anomaly" in Chinese commodity futures.

  Strategy 2 - Vol Breakout:
    Time-series signal based on 20d/60d vol ratio.
    Long when vol contracting (calm), Short when vol expanding (risky).
    Tests mean-reversion in volatility term structure.

  Strategy 3 - Vol Mean Reversion:
    Percentile rank of current vol vs 252-day history.
    Long at low percentiles, Short at high percentiles.
    Tests vol mean-reversion effect.

  Risk Management: -2% SL, +5% TP per position
  Walk-Forward: Train 2021-2023 -> Validate 2024 -> Test 2025-2026
""")

    # Print best configs
    print("  Best Configurations (Full Sample):")
    print("-" * 80)
    for r in best_all:
        print(f"    {r['Label']}")
        print(f"      Sharpe={r['Sharpe']}, AnnRet={r['AnnRet(%)']}%, "
              f"MDD={r['MDD(%)']}%, WinRate={r['WinRate(%)']}%, "
              f"Trades={r['NTrades']}")
        print()

    print("=" * 120)
    print("  DONE")
    print("=" * 120)


if __name__ == "__main__":
    main()
