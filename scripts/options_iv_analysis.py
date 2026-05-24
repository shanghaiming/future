#!/usr/bin/env python3
"""
Options IV Surface Analysis for Chinese Commodity Futures
==========================================================
Analyses IV smile/skew construction, cross-sectional ranking,
relationship with future returns, and time-series evolution.

Data sources:
  - data/options/ : 88 JSON files across 6 date snapshots
  - data/futures_weighted/ : daily futures price data (CSV)
"""

import json
import os
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
OPTIONS_DIR = BASE_DIR / "data" / "options"
FUTURES_DIR = BASE_DIR / "data" / "futures_weighted"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


# ==============================================================================
# 1. LOAD ALL OPTIONS JSON FILES -> UNIFIED DATAFRAME
# ==============================================================================

def load_all_options():
    """Load all JSON files, parse both formats into a unified DataFrame."""
    records = []

    for fname in sorted(os.listdir(OPTIONS_DIR)):
        if not fname.endswith(".json"):
            continue

        filepath = OPTIONS_DIR / fname
        with open(filepath) as f:
            data = json.load(f)

        # Parse filename -> symbol and date
        parts = fname.replace(".json", "").rsplit("_", 1)
        symbol = parts[0]
        date_str = parts[1]  # e.g. "20260508"

        if isinstance(data, dict) and "surface" in data:
            # FORMAT A: surface dict
            underlying_price = data.get("underlying_price", np.nan)
            hv_20 = data.get("hv_20", np.nan)
            hv_60 = data.get("hv_60", np.nan)
            date_val = data.get("date", date_str)

            for row in data["surface"]:
                rec = {
                    "symbol": symbol,
                    "date": date_str,
                    "underlying_price": underlying_price,
                    "hv_20": hv_20,
                    "hv_60": hv_60,
                    "moneyness": row.get("moneyness", np.nan),
                    "expiry_days": row.get("expiry_days", np.nan),
                    "flag": "call" if row["flag"] == "call" else "put",
                    "iv": row.get("iv", np.nan),
                    "delta": row.get("delta", np.nan),
                    "gamma": row.get("gamma", np.nan),
                    "theta": row.get("theta", np.nan),
                    "vega": row.get("vega", np.nan),
                    "rho": row.get("rho", np.nan),
                }
                records.append(rec)

        elif isinstance(data, list):
            # FORMAT B: array of records
            # Map expiry in years to approximate days
            expiry_map = {0.08: 30, 0.17: 60, 0.42: 90}
            # Map approximate moneyness to standard levels
            def round_moneyness(m):
                standard = [0.80, 0.84, 0.88, 0.92, 0.96, 1.00, 1.04, 1.08, 1.12, 1.16, 1.20]
                return min(standard, key=lambda x: abs(x - m))

            for row in data:
                expiry_years = row.get("expiry", np.nan)
                expiry_days = expiry_map.get(expiry_years, round(expiry_years * 365))
                moneyness_raw = row.get("moneyness", np.nan)
                moneyness = round_moneyness(moneyness_raw) if not np.isnan(moneyness_raw) else np.nan

                flag_raw = row.get("flag", "")
                flag = "call" if flag_raw == "c" else "put"

                rec = {
                    "symbol": symbol,
                    "date": date_str,
                    "underlying_price": np.nan,  # Not in format B
                    "hv_20": np.nan,
                    "hv_60": np.nan,
                    "moneyness": moneyness,
                    "expiry_days": expiry_days,
                    "flag": flag,
                    "iv": row.get("implied_vol", np.nan),
                    "delta": row.get("delta", np.nan),
                    "gamma": row.get("gamma", np.nan),
                    "theta": row.get("theta", np.nan),
                    "vega": row.get("vega", np.nan),
                    "rho": row.get("rho", np.nan),
                }
                records.append(rec)

    df = pd.DataFrame(records)
    # Round moneyness for format A too (clean up float noise)
    df["moneyness"] = df["moneyness"].round(2)
    return df


# ==============================================================================
# 2. IV SKEW / SMILE METRICS
# ==============================================================================

def compute_iv_metrics(df):
    """
    For each symbol+date, compute IV surface metrics.
    Returns a DataFrame with one row per (symbol, date).
    """
    results = []

    for (sym, dt), grp in df.groupby(["symbol", "date"]):
        grp_c = grp[grp["flag"] == "call"]
        grp_p = grp[grp["flag"] == "put"]

        underlying = grp["underlying_price"].dropna().iloc[0] if len(grp["underlying_price"].dropna()) > 0 else np.nan
        hv_20 = grp["hv_20"].dropna().iloc[0] if len(grp["hv_20"].dropna()) > 0 else np.nan
        hv_60 = grp["hv_60"].dropna().iloc[0] if len(grp["hv_60"].dropna()) > 0 else np.nan

        # --- ATM IV at 30d expiry ---
        atm_30 = grp[(grp["moneyness"] == 1.00) & (grp["expiry_days"] == 30)]
        atm_iv_30 = atm_30["iv"].mean() if len(atm_30) > 0 else np.nan

        # --- ATM IV at 60d and 90d ---
        atm_60 = grp[(grp["moneyness"] == 1.00) & (grp["expiry_days"] == 60)]
        atm_iv_60 = atm_60["iv"].mean() if len(atm_60) > 0 else np.nan
        atm_90 = grp[(grp["moneyness"] == 1.00) & (grp["expiry_days"] == 90)]
        atm_iv_90 = atm_90["iv"].mean() if len(atm_90) > 0 else np.nan

        # --- OTM Put IV (moneyness ~0.90 or nearest available) ---
        otm_put_targets = [0.90, 0.88, 0.92, 0.84]
        otm_put_iv_30 = np.nan
        for target_m in otm_put_targets:
            candidates = grp_p[(grp_p["moneyness"] == target_m) & (grp_p["expiry_days"] == 30)]
            if len(candidates) > 0:
                otm_put_iv_30 = candidates["iv"].mean()
                break

        # --- OTM Call IV (moneyness ~1.10 or nearest) ---
        otm_call_targets = [1.10, 1.12, 1.08, 1.16]
        otm_call_iv_30 = np.nan
        for target_m in otm_call_targets:
            candidates = grp_c[(grp_c["moneyness"] == target_m) & (grp_c["expiry_days"] == 30)]
            if len(candidates) > 0:
                otm_call_iv_30 = candidates["iv"].mean()
                break

        # --- ITM Put IV (moneyness ~1.10, which is ITM for a put) ---
        itm_put_targets = [1.10, 1.12, 1.08, 1.16]
        itm_put_iv_30 = np.nan
        for target_m in itm_put_targets:
            candidates = grp_p[(grp_p["moneyness"] == target_m) & (grp_p["expiry_days"] == 30)]
            if len(candidates) > 0:
                itm_put_iv_30 = candidates["iv"].mean()
                break

        # --- Deep OTM Put IV (moneyness ~0.84) ---
        deep_otm_put_targets = [0.84, 0.80, 0.88]
        deep_otm_put_iv_30 = np.nan
        for target_m in deep_otm_put_targets:
            candidates = grp_p[(grp_p["moneyness"] == target_m) & (grp_p["expiry_days"] == 30)]
            if len(candidates) > 0:
                deep_otm_put_iv_30 = candidates["iv"].mean()
                break

        # === Derived metrics ===

        # Skew = IV(OTM_put, K=0.9) - IV(OTM_call, K=1.1)
        skew = otm_put_iv_30 - otm_call_iv_30 if not (np.isnan(otm_put_iv_30) or np.isnan(otm_call_iv_30)) else np.nan

        # Skew_ratio = IV(OTM_put) / IV(ATM)
        skew_ratio = otm_put_iv_30 / atm_iv_30 if not (np.isnan(otm_put_iv_30) or np.isnan(atm_iv_30)) and atm_iv_30 > 0 else np.nan

        # IV term spread = IV_30d - IV_90d (at ATM)
        iv_term_spread = atm_iv_30 - atm_iv_90 if not (np.isnan(atm_iv_30) or np.isnan(atm_iv_90)) else np.nan

        # Kurtosis proxy = IV(OTM) - 2*IV(ATM) + IV(ITM)
        # Using put OTM, ATM, put ITM
        kurtosis_proxy = np.nan
        if not (np.isnan(otm_put_iv_30) or np.isnan(atm_iv_30) or np.isnan(itm_put_iv_30)):
            kurtosis_proxy = otm_put_iv_30 - 2 * atm_iv_30 + itm_put_iv_30

        # IV-HV ratio
        iv_hv_ratio = np.nan
        if not np.isnan(atm_iv_30) and not np.isnan(hv_20) and hv_20 > 0:
            iv_hv_ratio = atm_iv_30 / hv_20
        elif not np.isnan(atm_iv_30) and not np.isnan(hv_60) and hv_60 > 0:
            iv_hv_ratio = atm_iv_30 / hv_60

        # Deep skew (more extreme)
        deep_skew = deep_otm_put_iv_30 - otm_call_iv_30 if not (np.isnan(deep_otm_put_iv_30) or np.isnan(otm_call_iv_30)) else np.nan

        # Full smile for stats: std of IV across all moneyness at 30d
        iv_30d = grp[(grp["expiry_days"] == 30)]
        iv_smile_std = iv_30d.groupby("moneyness")["iv"].mean().std() if len(iv_30d) > 0 else np.nan

        results.append({
            "symbol": sym,
            "date": dt,
            "underlying_price": underlying,
            "hv_20": hv_20,
            "hv_60": hv_60,
            "atm_iv_30": atm_iv_30,
            "atm_iv_60": atm_iv_60,
            "atm_iv_90": atm_iv_90,
            "otm_put_iv_30": otm_put_iv_30,
            "otm_call_iv_30": otm_call_iv_30,
            "deep_otm_put_iv_30": deep_otm_put_iv_30,
            "itm_put_iv_30": itm_put_iv_30,
            "skew": skew,
            "skew_ratio": skew_ratio,
            "iv_term_spread": iv_term_spread,
            "kurtosis_proxy": kurtosis_proxy,
            "iv_hv_ratio": iv_hv_ratio,
            "deep_skew": deep_skew,
            "iv_smile_std": iv_smile_std,
        })

    return pd.DataFrame(results)


# ==============================================================================
# 3. LOAD FUTURES DAILY DATA & COMPUTE FORWARD RETURNS
# ==============================================================================

def load_futures_daily():
    """Load all futures daily CSV files into one DataFrame."""
    frames = []
    seen_keys = set()
    for fname in sorted(os.listdir(FUTURES_DIR)):
        if not fname.endswith(".csv"):
            continue
        fpath = FUTURES_DIR / fname
        try:
            df = pd.read_csv(fpath)
            if len(df) == 0:
                continue
            # Normalize ts_code to lowercase for matching with options symbols
            df["ts_code"] = df["ts_code"].str.lower()
            # Deduplicate: same ts_code may appear in multiple files (case variants)
            key = (df["ts_code"].iloc[0], fname.lower())
            if key in seen_keys:
                continue
            seen_keys.add(key)
            frames.append(df)
        except Exception:
            continue
    df = pd.concat(frames, ignore_index=True)
    df["trade_date"] = df["trade_date"].astype(str)
    # Deduplicate rows by (ts_code, trade_date) - keep first
    df = df.drop_duplicates(subset=["ts_code", "trade_date"], keep="first")
    df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    return df


def compute_forward_returns(fut_df, iv_metrics_df):
    """
    For each (symbol, date) in iv_metrics, compute forward returns.
    Options dates may not be actual trading dates (e.g., holidays).
    Find the nearest actual trading date on or before the options date.
    """
    # Pre-build per-symbol sorted date index for fast lookup
    sym_date_map = {}
    for sym, grp in fut_df.groupby("ts_code"):
        grp_sorted = grp.sort_values("trade_date").reset_index(drop=True)
        sym_date_map[sym] = grp_sorted

    fwd_periods = [1, 3, 5, 10]
    result_rows = []

    for _, row in iv_metrics_df.iterrows():
        sym = row["symbol"].lower()
        dt = row["date"]

        if sym not in sym_date_map:
            continue

        sym_prices = sym_date_map[sym]

        # Find the nearest trading date <= dt
        valid_dates = sym_prices[sym_prices["trade_date"] <= dt]
        if len(valid_dates) == 0:
            continue

        # Use the most recent date on or before the options date
        best_idx = valid_dates.index[-1]
        base_price = sym_prices.loc[best_idx, "close"]
        actual_date = sym_prices.loc[best_idx, "trade_date"]

        fwd = {"symbol": row["symbol"], "date": dt, "actual_trade_date": actual_date, "base_price": base_price}

        for n in fwd_periods:
            fwd_idx = best_idx + n
            if fwd_idx < len(sym_prices):
                fwd_price = sym_prices.loc[fwd_idx, "close"]
                fwd[f"fwd_ret_{n}d"] = (fwd_price / base_price) - 1
            else:
                fwd[f"fwd_ret_{n}d"] = np.nan

        result_rows.append(fwd)

    return pd.DataFrame(result_rows)


# ==============================================================================
# 4. CROSS-SECTIONAL ANALYSIS
# ==============================================================================

def cross_sectional_analysis(iv_metrics_df):
    """Rank symbols by IV metrics at each date."""
    print("=" * 100)
    print("CROSS-SECTIONAL ANALYSIS: IV RANKINGS BY DATE")
    print("=" * 100)

    for dt in sorted(iv_metrics_df["date"].unique()):
        sub = iv_metrics_df[iv_metrics_df["date"] == dt].dropna(subset=["atm_iv_30"])
        if len(sub) == 0:
            continue
        print(f"\n--- Date: {dt} ({len(sub)} symbols) ---")

        # Top/Bottom by ATM IV
        print(f"\n  TOP 10 HIGHEST ATM IV (30d):")
        top_iv = sub.nlargest(10, "atm_iv_30")[["symbol", "atm_iv_30", "skew", "iv_hv_ratio"]]
        print(top_iv.to_string(index=False))

        print(f"\n  TOP 10 LOWEST ATM IV (30d):")
        bot_iv = sub.nsmallest(10, "atm_iv_30")[["symbol", "atm_iv_30", "skew", "iv_hv_ratio"]]
        print(bot_iv.to_string(index=False))

        # Most negative skew
        skew_sub = sub.dropna(subset=["skew"])
        if len(skew_sub) > 0:
            print(f"\n  TOP 10 MOST NEGATIVE SKEW (downside fear):")
            neg_skew = skew_sub.nsmallest(10, "skew")[["symbol", "skew", "skew_ratio", "atm_iv_30"]]
            print(neg_skew.to_string(index=False))

            print(f"\n  TOP 10 MOST POSITIVE SKEW:")
            pos_skew = skew_sub.nlargest(10, "skew")[["symbol", "skew", "skew_ratio", "atm_iv_30"]]
            print(pos_skew.to_string(index=False))

        # IV term spread
        ts_sub = sub.dropna(subset=["iv_term_spread"])
        if len(ts_sub) > 0:
            print(f"\n  TOP 10 MOST INVERTED TERM SPREAD (30d - 90d, most negative):")
            inv_ts = ts_sub.nsmallest(10, "iv_term_spread")[["symbol", "iv_term_spread", "atm_iv_30"]]
            print(inv_ts.to_string(index=False))

            print(f"\n  TOP 10 STEEPEST TERM SPREAD (30d - 90d, most positive):")
            steep_ts = ts_sub.nlargest(10, "iv_term_spread")[["symbol", "iv_term_spread", "atm_iv_30"]]
            print(steep_ts.to_string(index=False))


# ==============================================================================
# 5. RELATIONSHIP WITH FUTURE RETURNS
# ==============================================================================

def analyze_iv_return_relationship(iv_metrics_df, fwd_returns_df):
    """Test relationships between IV factors and forward returns."""
    print("\n" + "=" * 100)
    print("RELATIONSHIP: IV FACTORS vs FORWARD RETURNS")
    print("=" * 100)

    # Merge metrics with forward returns
    merged = iv_metrics_df.merge(fwd_returns_df, on=["symbol", "date"], how="inner")
    n_obs = len(merged)
    print(f"\n  Total merged observations: {n_obs}")

    factors = ["atm_iv_30", "skew", "skew_ratio", "iv_term_spread", "iv_hv_ratio", "kurtosis_proxy", "deep_skew"]
    ret_cols = ["fwd_ret_1d", "fwd_ret_3d", "fwd_ret_5d", "fwd_ret_10d"]

    # a) Correlation table
    print("\n  --- Correlation Matrix: IV Factors vs Forward Returns ---")
    corr_data = []
    for factor in factors:
        row = {"factor": factor}
        for ret in ret_cols:
            valid = merged[[factor, ret]].dropna()
            if len(valid) >= 5:
                r, p = stats.pearsonr(valid[factor], valid[ret])
                row[f"{ret}_r"] = r
                row[f"{ret}_p"] = p
                row[f"{ret}_n"] = len(valid)
            else:
                row[f"{ret}_r"] = np.nan
                row[f"{ret}_p"] = np.nan
                row[f"{ret}_n"] = 0
        corr_data.append(row)

    corr_df = pd.DataFrame(corr_data)
    print("\n  Factor                | fwd_ret_1d_r  (p)     | fwd_ret_3d_r  (p)     | fwd_ret_5d_r  (p)     | fwd_ret_10d_r (p)")
    print("  " + "-" * 120)
    for _, row in corr_df.iterrows():
        line = f"  {row['factor']:<22s}"
        for ret in ret_cols:
            r = row[f"{ret}_r"]
            p = row[f"{ret}_p"]
            n = row[f"{ret}_n"]
            if np.isnan(r):
                line += f" | {'N/A':>20s}"
            else:
                sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
                line += f" | {r:>7.4f} ({p:.3f}){sig:<4s} n={int(n)}"
        print(line)

    # b) Quintile analysis for ATM IV
    print("\n  --- Quintile Analysis: ATM IV -> Forward Returns ---")
    for ret in ret_cols:
        valid = merged[["atm_iv_30", ret]].dropna()
        if len(valid) < 10:
            continue
        valid = valid.copy()
        valid["iv_quintile"] = pd.qcut(valid["atm_iv_30"], 5, labels=["Q1(Low)", "Q2", "Q3", "Q4", "Q5(High)"], duplicates="drop")
        qstats = valid.groupby("iv_quintile", observed=True)[ret].agg(["mean", "median", "count"])
        print(f"\n  {ret}:")
        print(qstats.to_string())

    # c) Quintile analysis for Skew
    print("\n  --- Quintile Analysis: SKEW -> Forward Returns ---")
    for ret in ret_cols:
        valid = merged[["skew", ret]].dropna()
        if len(valid) < 10:
            continue
        valid = valid.copy()
        # Use median split for skew since many values are identical
        skew_median = valid["skew"].median()
        valid["skew_group"] = np.where(valid["skew"] <= skew_median, "Low Skew", "High Skew")
        qstats = valid.groupby("skew_group", observed=True)[ret].agg(["mean", "median", "count"])
        print(f"\n  {ret} (median split at skew={skew_median:.6f}):")
        print(qstats.to_string())

    # d) IV-HV ratio analysis
    print("\n  --- IV-HV Ratio (Volatility Risk Premium) -> Forward Returns ---")
    for ret in ret_cols:
        valid = merged[["iv_hv_ratio", ret]].dropna()
        if len(valid) < 10:
            continue
        valid = valid.copy()
        median_ratio = valid["iv_hv_ratio"].median()
        valid["vrp_signal"] = np.where(valid["iv_hv_ratio"] > median_ratio, "High(Overpriced)", "Low(Underpriced)")
        vrp_stats = valid.groupby("vrp_signal")[ret].agg(["mean", "median", "std", "count"])
        print(f"\n  {ret} (median IV/HV ratio = {median_ratio:.4f}):")
        print(vrp_stats.to_string())

    # e) Long-Short portfolios based on IV and Skew
    print("\n  --- Simple Long-Short Signals ---")
    for dt in sorted(merged["date"].unique()):
        sub = merged[merged["date"] == dt].dropna(subset=["atm_iv_30", "skew", "fwd_ret_5d"])
        if len(sub) < 10:
            continue
        sub = sub.copy()

        # High IV -> short, Low IV -> long
        iv_median = sub["atm_iv_30"].median()
        low_iv_ret = sub[sub["atm_iv_30"] <= iv_median]["fwd_ret_5d"].mean()
        high_iv_ret = sub[sub["atm_iv_30"] > iv_median]["fwd_ret_5d"].mean()
        ls_iv = low_iv_ret - high_iv_ret

        # Negative skew -> short, Positive skew -> long
        skew_median = sub["skew"].median()
        pos_skew_ret = sub[sub["skew"] >= skew_median]["fwd_ret_5d"].mean()
        neg_skew_ret = sub[sub["skew"] < skew_median]["fwd_ret_5d"].mean()
        ls_skew = pos_skew_ret - neg_skew_ret

        print(f"  {dt}: LS(IV)={ls_iv:+.4f}  LS(Skew)={ls_skew:+.4f}  (n={len(sub)})")

    return corr_df


# ==============================================================================
# 6. TIME-SERIES ANALYSIS
# ==============================================================================

def time_series_analysis(iv_metrics_df):
    """Track IV and skew evolution across dates for each symbol."""
    print("\n" + "=" * 100)
    print("TIME-SERIES ANALYSIS: IV EVOLUTION ACROSS DATES")
    print("=" * 100)

    # Symbols with multiple dates
    sym_counts = iv_metrics_df.groupby("symbol")["date"].nunique()
    multi_date_syms = sym_counts[sym_counts > 1].index.tolist()
    print(f"\n  Symbols with data at multiple dates: {len(multi_date_syms)}")

    if len(multi_date_syms) == 0:
        print("  No symbols with multiple dates found. Skipping time-series analysis.")
        return

    # Biggest IV changes
    print("\n  --- Biggest ATM IV Changes (first -> last date) ---")
    ts_changes = []
    for sym in multi_date_syms:
        sub = iv_metrics_df[iv_metrics_df["symbol"] == sym].sort_values("date")
        if len(sub) < 2:
            continue
        first_iv = sub.iloc[0]["atm_iv_30"]
        last_iv = sub.iloc[-1]["atm_iv_30"]
        first_skew = sub.iloc[0]["skew"]
        last_skew = sub.iloc[-1]["skew"]
        if np.isnan(first_iv) or np.isnan(last_iv):
            continue
        ts_changes.append({
            "symbol": sym,
            "first_date": sub.iloc[0]["date"],
            "last_date": sub.iloc[-1]["date"],
            "first_atm_iv": first_iv,
            "last_atm_iv": last_iv,
            "iv_change": last_iv - first_iv,
            "iv_pct_change": (last_iv - first_iv) / first_iv if first_iv != 0 else np.nan,
            "first_skew": first_skew,
            "last_skew": last_skew,
            "skew_change": (last_skew - first_skew) if not (np.isnan(first_skew) or np.isnan(last_skew)) else np.nan,
            "n_dates": len(sub),
        })

    if not ts_changes:
        print("  No valid time-series changes found.")
        return

    ts_df = pd.DataFrame(ts_changes)

    print("\n  Top 10 IV INCREASES:")
    print(ts_df.nlargest(10, "iv_change")[["symbol", "first_date", "last_date", "first_atm_iv", "last_atm_iv", "iv_change"]].to_string(index=False))

    print("\n  Top 10 IV DECREASES:")
    print(ts_df.nsmallest(10, "iv_change")[["symbol", "first_date", "last_date", "first_atm_iv", "last_atm_iv", "iv_change"]].to_string(index=False))

    print("\n  Top 10 SKEW INCREASES (becoming more fearful):")
    valid_skew = ts_df.dropna(subset=["skew_change"])
    if len(valid_skew) > 0:
        print(valid_skew.nlargest(10, "skew_change")[["symbol", "first_date", "last_date", "first_skew", "last_skew", "skew_change"]].to_string(index=False))

        print("\n  Top 10 SKEW DECREASES (becoming less fearful / more complacent):")
        print(valid_skew.nsmallest(10, "skew_change")[["symbol", "first_date", "last_date", "first_skew", "last_skew", "skew_change"]].to_string(index=False))

    # Evolution table for all symbols across dates
    print("\n  --- Full IV Evolution Table ---")
    pivot_iv = iv_metrics_df.pivot_table(index="symbol", columns="date", values="atm_iv_30")
    pivot_iv = pivot_iv.dropna(how="all")
    print(f"\n  ATM IV (30d) across dates ({len(pivot_iv)} symbols):")
    with pd.option_context("display.max_rows", 200, "display.width", 200, "display.float_format", "{:.4f}".format):
        print(pivot_iv.to_string())

    print("\n  --- Full Skew Evolution Table ---")
    pivot_skew = iv_metrics_df.pivot_table(index="symbol", columns="date", values="skew")
    pivot_skew = pivot_skew.dropna(how="all")
    print(f"\n  Skew across dates ({len(pivot_skew)} symbols):")
    with pd.option_context("display.max_rows", 200, "display.width", 200, "display.float_format", "{:.4f}".format):
        print(pivot_skew.to_string())


# ==============================================================================
# 7. SUMMARY TABLES
# ==============================================================================

def print_summary_tables(iv_metrics_df, corr_df):
    """Print all requested summary tables."""
    print("\n" + "=" * 100)
    print("SUMMARY TABLES")
    print("=" * 100)

    # a) Per-symbol summary
    print("\n  --- Per-Symbol Summary (averaged across dates) ---")
    sym_summary = iv_metrics_df.groupby("symbol").agg(
        avg_atm_iv=("atm_iv_30", "mean"),
        std_atm_iv=("atm_iv_30", "std"),
        avg_skew=("skew", "mean"),
        avg_skew_ratio=("skew_ratio", "mean"),
        avg_iv_hv_ratio=("iv_hv_ratio", "mean"),
        avg_iv_term_spread=("iv_term_spread", "mean"),
        avg_kurtosis_proxy=("kurtosis_proxy", "mean"),
        n_dates=("date", "nunique"),
    ).dropna(subset=["avg_atm_iv"])

    sym_summary = sym_summary.sort_values("avg_atm_iv", ascending=False)
    print(f"\n  ({len(sym_summary)} symbols)")
    with pd.option_context("display.max_rows", 200, "display.width", 200, "display.float_format", "{:.4f}".format):
        print(sym_summary.to_string())

    # b) Top 10 most negative skew (highest fear)
    print("\n  --- TOP 10 SYMBOLS WITH MOST NEGATIVE SKEW (Highest Downside Fear) ---")
    skew_summary = iv_metrics_df.groupby("symbol").agg(
        avg_skew=("skew", "mean"),
        avg_atm_iv=("atm_iv_30", "mean"),
        avg_skew_ratio=("skew_ratio", "mean"),
        n_dates=("date", "nunique"),
    ).dropna(subset=["avg_skew"])
    top_fear = skew_summary.nsmallest(10, "avg_skew")
    print(top_fear.to_string())

    # c) Top 10 highest IV
    print("\n  --- TOP 10 SYMBOLS WITH HIGHEST IV (Most Volatile) ---")
    top_iv = sym_summary.nlargest(10, "avg_atm_iv")[["avg_atm_iv", "std_atm_iv", "avg_skew", "avg_iv_hv_ratio"]]
    print(top_iv.to_string())

    # d) Top 10 lowest IV
    print("\n  --- TOP 10 SYMBOLS WITH LOWEST IV (Least Volatile) ---")
    low_iv = sym_summary.nsmallest(10, "avg_atm_iv")[["avg_atm_iv", "std_atm_iv", "avg_skew", "avg_iv_hv_ratio"]]
    print(low_iv.to_string())

    # e) IC-like correlation results
    print("\n  --- IC-Like Test: Correlation Between IV Factors and Forward Returns ---")
    print("  (See detailed correlation table above for per-horizon results)")
    if corr_df is not None and len(corr_df) > 0:
        print("\n  Summary of strongest relationships:")
        for _, row in corr_df.iterrows():
            for ret in ["fwd_ret_1d", "fwd_ret_3d", "fwd_ret_5d", "fwd_ret_10d"]:
                r = row[f"{ret}_r"]
                p = row[f"{ret}_p"]
                if not np.isnan(r) and abs(r) > 0.15:
                    sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
                    direction = "positive" if r > 0 else "negative"
                    print(f"    {row['factor']:<22s} -> {ret:>13s}: r={r:+.4f} p={p:.4f} {sig} ({direction})")


# ==============================================================================
# 8. IV SMILE PROFILES FOR SELECTED SYMBOLS
# ==============================================================================

def print_iv_smile_profiles(df):
    """Print IV smile data for a few representative symbols."""
    print("\n" + "=" * 100)
    print("IV SMILE PROFILES (30d expiry, selected symbols)")
    print("=" * 100)

    # Pick a few representative symbols
    dates = sorted(df["date"].unique())
    latest_date = dates[-1]
    sub = df[(df["date"] == latest_date) & (df["expiry_days"] == 30)]

    # Pick symbols with different characteristics
    symbols_to_show = []
    if "agfi" in sub["symbol"].values:
        symbols_to_show.append("agfi")
    if "aufi" in sub["symbol"].values:
        symbols_to_show.append("aufi")
    if "cufi" in sub["symbol"].values:
        symbols_to_show.append("cufi")
    if "rbfi" in sub["symbol"].values:
        symbols_to_show.append("rbfi")
    if "mafi" in sub["symbol"].values:
        symbols_to_show.append("mafi")
    if "iffi" in sub["symbol"].values:
        symbols_to_show.append("iffi")

    # Fill to 6 if needed
    avail = sub["symbol"].unique().tolist()
    for s in avail:
        if len(symbols_to_show) >= 6:
            break
        if s not in symbols_to_show:
            symbols_to_show.append(s)

    for sym in symbols_to_show[:6]:
        sym_data = sub[sub["symbol"] == sym].sort_values(["moneyness", "flag"])
        if len(sym_data) == 0:
            continue
        up = sym_data["underlying_price"].iloc[0]
        print(f"\n  --- {sym} (date={latest_date}, underlying={up}) ---")
        print(f"  {'Moneyness':>10s} | {'Call IV':>10s} | {'Put IV':>10s} | {'C-P Diff':>10s} | {'Call Delta':>10s} | {'Put Delta':>10s}")
        print("  " + "-" * 75)
        for m in sorted(sym_data["moneyness"].unique()):
            c_iv = sym_data[(sym_data["moneyness"] == m) & (sym_data["flag"] == "call")]["iv"]
            p_iv = sym_data[(sym_data["moneyness"] == m) & (sym_data["flag"] == "put")]["iv"]
            c_delta = sym_data[(sym_data["moneyness"] == m) & (sym_data["flag"] == "call")]["delta"]
            p_delta = sym_data[(sym_data["moneyness"] == m) & (sym_data["flag"] == "put")]["delta"]

            c_iv_val = c_iv.values[0] if len(c_iv) > 0 else np.nan
            p_iv_val = p_iv.values[0] if len(p_iv) > 0 else np.nan
            c_d_val = c_delta.values[0] if len(c_delta) > 0 else np.nan
            p_d_val = p_delta.values[0] if len(p_delta) > 0 else np.nan
            diff = c_iv_val - p_iv_val if not (np.isnan(c_iv_val) or np.isnan(p_iv_val)) else np.nan

            print(f"  {m:>10.2f} | {c_iv_val:>10.4f} | {p_iv_val:>10.4f} | {diff:>+10.4f} | {c_d_val:>+10.4f} | {p_d_val:>+10.4f}")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("=" * 100)
    print("OPTIONS IV SURFACE ANALYSIS FOR CHINESE COMMODITY FUTURES")
    print("=" * 100)

    # 1. Load all options data
    print("\n[1] Loading options data...")
    df = load_all_options()
    n_format_a = len(df[df["hv_20"].notna()])
    n_format_b = len(df[df["hv_20"].isna()])
    print(f"    Total records: {len(df)}")
    print(f"    Format A (surface): {n_format_a} records")
    print(f"    Format B (array):   {n_format_b} records")
    print(f"    Unique symbols: {df['symbol'].nunique()}")
    print(f"    Unique dates: {sorted(df['date'].unique())}")
    print(f"    Moneyness levels: {sorted(df['moneyness'].dropna().unique())}")
    print(f"    Expiry days: {sorted(df['expiry_days'].dropna().unique())}")

    # 2. Compute IV metrics
    print("\n[2] Computing IV surface metrics...")
    iv_metrics = compute_iv_metrics(df)
    print(f"    Metric rows: {len(iv_metrics)}")
    print(f"    Symbols with ATM IV: {iv_metrics['atm_iv_30'].notna().sum()}")
    print(f"    Symbols with skew: {iv_metrics['skew'].notna().sum()}")
    print(f"    Symbols with IV-HV ratio: {iv_metrics['iv_hv_ratio'].notna().sum()}")

    # 3. Load futures daily data
    print("\n[3] Loading futures daily data...")
    fut_df = load_futures_daily()
    print(f"    Total futures rows: {len(fut_df)}")
    print(f"    Unique symbols: {fut_df['ts_code'].nunique()}")
    print(f"    Date range: {fut_df['trade_date'].min()} to {fut_df['trade_date'].max()}")

    # 4. Compute forward returns
    print("\n[4] Computing forward returns...")
    fwd_returns = compute_forward_returns(fut_df, iv_metrics)
    for n in [1, 3, 5, 10]:
        col = f"fwd_ret_{n}d"
        valid = fwd_returns[col].notna().sum()
        print(f"    {col}: {valid} observations")

    # 5. IV Smile Profiles
    print_iv_smile_profiles(df)

    # 6. Cross-sectional analysis
    cross_sectional_analysis(iv_metrics)

    # 7. IV-return relationship
    corr_df = analyze_iv_return_relationship(iv_metrics, fwd_returns)

    # 8. Time-series analysis
    time_series_analysis(iv_metrics)

    # 9. Summary tables
    print_summary_tables(iv_metrics, corr_df)

    # 10. Save full data table
    print("\n[10] Saving output files...")
    full_table = iv_metrics.merge(fwd_returns, on=["symbol", "date"], how="left")
    output_path = OUTPUT_DIR / "iv_analysis_full_table.csv"
    full_table.to_csv(output_path, index=False)
    print(f"    Full data table saved to: {output_path}")

    sym_summary = iv_metrics.groupby("symbol").agg(
        avg_atm_iv=("atm_iv_30", "mean"),
        std_atm_iv=("atm_iv_30", "std"),
        avg_skew=("skew", "mean"),
        avg_skew_ratio=("skew_ratio", "mean"),
        avg_iv_hv_ratio=("iv_hv_ratio", "mean"),
        avg_iv_term_spread=("iv_term_spread", "mean"),
        avg_kurtosis_proxy=("kurtosis_proxy", "mean"),
        n_dates=("date", "nunique"),
    ).dropna(subset=["avg_atm_iv"]).sort_values("avg_atm_iv", ascending=False)

    summary_path = OUTPUT_DIR / "iv_analysis_symbol_summary.csv"
    sym_summary.to_csv(summary_path)
    print(f"    Symbol summary saved to: {summary_path}")

    print("\n" + "=" * 100)
    print("ANALYSIS COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()
