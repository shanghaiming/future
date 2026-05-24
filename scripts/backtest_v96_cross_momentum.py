#!/usr/bin/env python3
"""
Backtest V96: Cross-Sectional Momentum Strategy for Chinese Commodity Futures
==============================================================================
Weekly-rebalanced, cross-sectional ranking strategy based on past N-day returns.

Variants tested:
  1. Plain momentum: rank by past N-day return
  2. Volume-weighted momentum: rank by return * volume change ratio
  3. OI-confirmed momentum: rank by return, but only trade when OI confirms direction

Parameters grid:
  - Lookback N: 5, 10, 20 days
  - Top/Bottom K: 5, 10 commodities
  - Holding period H: 5, 10, 15 days
  - Stop loss: -2%, Take profit: +5%

Walk-forward:
  - Train: 2021-01-01 to 2023-12-31
  - Validate: 2024-01-01 to 2024-12-31
  - Test: 2025-01-01 to 2026-05-20

Data: data/futures_weighted/  (CSV: ts_code, trade_date, open, high, low, close, vol, amount, oi)
"""

import sys
import os
import glob
import warnings
from itertools import product

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DAILY_DIR = os.path.join(BASE_DIR, "data", "futures_weighted")

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
LOOKBACKS = [5, 10, 20]
TOP_KS = [5, 10]
HOLD_PERIODS = [5, 10, 15]
STOP_LOSS_PCT = -0.02
TAKE_PROFIT_PCT = 0.05

WALK_FORWARD_DATES = {
    "train": ("2021-01-01", "2023-12-31"),
    "validate": ("2024-01-01", "2024-12-31"),
    "test": ("2025-01-01", "2026-05-20"),
}


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_all_futures():
    """Load all commodity CSV files into a single DataFrame."""
    frames = []
    for fp in glob.glob(os.path.join(DAILY_DIR, "*.csv")):
        symbol = os.path.basename(fp).replace(".csv", "")
        try:
            df = pd.read_csv(fp)
        except Exception:
            continue
        if df.empty or len(df) < 100:
            continue

        df.columns = [c.strip().lower() for c in df.columns]
        if "trade_date" not in df.columns or "close" not in df.columns:
            continue

        df["symbol"] = symbol
        df["trade_date"] = pd.to_datetime(df["trade_date"], format="mixed", errors="coerce")
        df.dropna(subset=["trade_date", "close"], inplace=True)
        df.sort_values("trade_date", inplace=True)
        df.drop_duplicates(subset=["trade_date"], keep="last", inplace=True)

        df = df[df["trade_date"] >= "2020-01-01"].copy()
        if len(df) < 60:
            continue

        frames.append(df[["symbol", "trade_date", "open", "high", "low", "close", "vol", "oi"]])

    if not frames:
        raise RuntimeError("No data loaded!")

    data = pd.concat(frames, ignore_index=True)
    data.sort_values(["symbol", "trade_date"], inplace=True)
    data.reset_index(drop=True, inplace=True)
    return data


# ---------------------------------------------------------------------------
# Pre-compute features + forward returns with stops (vectorized)
# ---------------------------------------------------------------------------
def compute_features(data):
    """Compute all per-symbol features and forward returns."""
    data = data.copy()
    data.sort_values(["symbol", "trade_date"], inplace=True)

    # Past N-day returns (momentum signal)
    for n in LOOKBACKS:
        data[f"mom_{n}"] = data.groupby("symbol")["close"].pct_change(n)

    # Volume change ratios
    for n in LOOKBACKS:
        vol_avg = data.groupby("symbol")["vol"].transform(
            lambda x: x.rolling(n, min_periods=n).mean()
        )
        data[f"vol_ratio_{n}"] = data["vol"] / vol_avg.replace(0, np.nan)

    # OI change
    for n in LOOKBACKS:
        data[f"oi_chg_{n}"] = data.groupby("symbol")["oi"].pct_change(n)

    # Forward returns (simple close-to-close for each holding period)
    for h in HOLD_PERIODS:
        data[f"fwd_ret_{h}"] = data.groupby("symbol")["close"].pct_change(h).shift(-h)

    # Forward returns with stop-loss / take-profit (vectorized per holding period)
    # Precompute: for each row, the entry price is close, and we scan forward H days
    # checking low/high against stops.
    # We'll build an index map: for each row, find the rows H steps ahead in same symbol.
    for h in HOLD_PERIODS:
        data[f"fwd_ret_sl_{h}"] = _compute_fwd_ret_with_stops(data, h)

    return data


def _compute_fwd_ret_with_stops(data, hold_period):
    """
    Vectorized computation of forward returns with stop-loss/take-profit.
    For each row, check forward `hold_period` days within same symbol.
    If any day's low <= entry*(1+SL) or high >= entry*(1+TP), cap the return.
    """
    grp = data.groupby("symbol")

    # Build arrays for close, high, low within each symbol
    close = data["close"].values.astype(np.float64)
    high = data["high"].values.astype(np.float64)
    low = data["low"].values.astype(np.float64)

    result = np.full(len(data), np.nan)

    for sym, idx in grp.groups.items():
        idx_arr = idx.values
        n = len(idx_arr)

        for i in range(n - hold_period):
            entry = close[idx_arr[i]]
            if np.isnan(entry) or entry == 0:
                continue

            ret = np.nan  # will be set below

            # Check each day in holding period for stops
            stopped = False
            for d in range(1, hold_period + 1):
                j = idx_arr[i + d]

                # Stop loss: intraday low breached
                day_low_ret = (low[j] - entry) / entry
                if day_low_ret <= STOP_LOSS_PCT:
                    ret = STOP_LOSS_PCT
                    stopped = True
                    break

                # Take profit: intraday high breached
                day_high_ret = (high[j] - entry) / entry
                if day_high_ret >= TAKE_PROFIT_PCT:
                    ret = TAKE_PROFIT_PCT
                    stopped = True
                    break

            if not stopped:
                exit_price = close[idx_arr[i + hold_period]]
                ret = (exit_price - entry) / entry

            result[idx_arr[i]] = ret

    return result


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------
def generate_signal_plain(data, lookback):
    """Plain momentum: rank by past N-day return."""
    return data[f"mom_{lookback}"]


def generate_signal_vol_weighted(data, lookback):
    """Volume-weighted momentum: rank by return * volume ratio."""
    return data[f"mom_{lookback}"] * data[f"vol_ratio_{lookback}"]


def generate_signal_oi_confirmed(data, lookback):
    """OI-confirmed momentum: return, zeroed when OI doesn't confirm direction."""
    mom = data[f"mom_{lookback}"].copy()
    oi_chg = data[f"oi_chg_{lookback}"]
    confirmed = ((mom > 0) & (oi_chg > 0)) | ((mom < 0) & (oi_chg < 0))
    return mom.where(confirmed, 0.0)


def generate_signal_plain_reversed(data, lookback):
    """Reversed momentum (mean reversion): long losers, short winners."""
    return -data[f"mom_{lookback}"]


def generate_signal_vol_weighted_reversed(data, lookback):
    """Reversed vol-weighted momentum (mean reversion)."""
    return -(data[f"mom_{lookback}"] * data[f"vol_ratio_{lookback}"])


def generate_signal_oi_confirmed_reversed(data, lookback):
    """Reversed OI-confirmed: mean reversion with OI filter."""
    mom = -data[f"mom_{lookback}"].copy()
    oi_chg = data[f"oi_chg_{lookback}"]
    # For reversed: we still want OI confirmation of the *new* direction
    confirmed = ((mom > 0) & (oi_chg > 0)) | ((mom < 0) & (oi_chg < 0))
    return mom.where(confirmed, 0.0)


SIGNAL_GENERATORS = {
    "plain": generate_signal_plain,
    "vol_weighted": generate_signal_vol_weighted,
    "oi_confirmed": generate_signal_oi_confirmed,
    "plain_reversed": generate_signal_plain_reversed,
    "vol_weighted_rev": generate_signal_vol_weighted_reversed,
    "oi_confirmed_rev": generate_signal_oi_confirmed_reversed,
}


# ---------------------------------------------------------------------------
# Backtest engine (fast: uses pre-computed forward returns)
# ---------------------------------------------------------------------------
def run_backtest(data, signal_type, lookback, top_k, hold_period,
                 start_date, end_date, use_stops=True):
    """
    Fast cross-sectional momentum backtest using pre-computed features.

    On each rebalance day (every `hold_period` trading days):
      1. Rank all commodities by signal
      2. Go long top K, short bottom K (equal weight)
      3. Use pre-computed forward returns (with or without stops)
    """
    mask = (data["trade_date"] >= start_date) & (data["trade_date"] <= end_date)
    subset = data[mask].copy()

    all_dates = sorted(subset["trade_date"].unique())
    if len(all_dates) < 20:
        return None

    rebalance_dates = all_dates[::hold_period]
    signal_func = SIGNAL_GENERATORS[signal_type]
    fwd_col = f"fwd_ret_sl_{hold_period}" if use_stops else f"fwd_ret_{hold_period}"

    # Use a dict for fast date lookup
    date_set = set(all_dates)
    date_to_idx = {d: i for i, d in enumerate(all_dates)}

    period_rets = []
    long_rets_list = []
    short_rets_list = []

    for rebal_date in rebalance_dates:
        cs = subset[subset["trade_date"] == rebal_date].copy()
        if len(cs) < 2 * top_k + 5:
            continue

        signals = signal_func(cs, lookback)
        cs = cs.copy()
        cs["signal"] = signals
        cs.dropna(subset=["signal", fwd_col], inplace=True)

        if len(cs) < 2 * top_k + 5:
            continue

        cs.sort_values("signal", ascending=False, inplace=True)

        longs = cs.head(top_k)
        shorts = cs.tail(top_k)

        if longs["signal"].abs().sum() == 0 and shorts["signal"].abs().sum() == 0:
            continue

        long_fwd = longs[fwd_col].values
        short_fwd = shorts[fwd_col].values

        # Drop NaN forward returns
        long_fwd = long_fwd[~np.isnan(long_fwd)]
        short_fwd = short_fwd[~np.isnan(short_fwd)]

        if len(long_fwd) == 0 and len(short_fwd) == 0:
            continue

        # Short side: flip sign
        short_fwd_neg = -short_fwd

        avg_long = np.mean(long_fwd) if len(long_fwd) > 0 else 0.0
        avg_short = np.mean(short_fwd_neg) if len(short_fwd_neg) > 0 else 0.0

        # Equal-weight portfolio: 1/(2K) per position
        weight = 1.0 / (2 * top_k)
        period_ret = weight * (len(long_fwd) * avg_long + len(short_fwd_neg) * avg_short)

        period_rets.append(period_ret)
        long_rets_list.extend(long_fwd.tolist())
        short_rets_list.extend(short_fwd_neg.tolist())

    if not period_rets:
        return None

    n_periods = len(period_rets)
    returns = np.array(period_rets)

    # --- Compute metrics from period returns ---
    n_days_total = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days
    n_years = n_days_total / 365.25
    trading_days = n_periods * hold_period  # approximate

    # Total return from compounding period returns
    cum = np.cumprod(1 + returns)
    total_ret = cum[-1] - 1

    # Annualize based on actual calendar time
    if n_years > 0:
        ann_ret = (1 + total_ret) ** (1 / n_years) - 1
    else:
        ann_ret = total_ret

    # Volatility: annualize the std of period returns
    ann_vol = np.std(returns) * np.sqrt(n_periods / n_years) if n_years > 0 and n_periods > 1 else 0

    sharpe = (ann_ret - 0.02) / ann_vol if ann_vol > 0 else 0

    # Max drawdown
    running_max = np.maximum.accumulate(cum)
    drawdowns = (cum - running_max) / running_max
    mdd = np.min(drawdowns)

    win_rate = np.mean(returns > 0)
    calmar = ann_ret / abs(mdd) if mdd != 0 else 0

    avg_long = np.mean(long_rets_list) if long_rets_list else 0
    avg_short = np.mean(short_rets_list) if short_rets_list else 0

    return {
        "total_return": total_ret,
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "calmar": calmar,
        "win_rate": win_rate,
        "n_periods": n_periods,
        "avg_long_ret": avg_long,
        "avg_short_ret": avg_short,
        "period_returns": returns,
    }


# ---------------------------------------------------------------------------
# Walk-forward optimization
# ---------------------------------------------------------------------------
def walk_forward_optimization(data, results_no_stops=None):
    """Walk-forward: train -> validate -> test. Tests both with/without stops."""

    log("=" * 100)
    log("WALK-FORWARD OPTIMIZATION (NO STOPS)")
    log("=" * 100)

    # --- Phase 1: Grid search on TRAINING (no stops, to find best signal) ---
    log(f"\n[TRAIN] {WALK_FORWARD_DATES['train'][0]} to {WALK_FORWARD_DATES['train'][1]}")
    log("-" * 100)

    train_results = []
    total = len(list(product(SIGNAL_GENERATORS.keys(), LOOKBACKS, TOP_KS, HOLD_PERIODS)))
    done = 0
    for signal_type, lookback, top_k, hold_period in product(
        SIGNAL_GENERATORS.keys(), LOOKBACKS, TOP_KS, HOLD_PERIODS
    ):
        done += 1
        res = run_backtest(
            data, signal_type, lookback, top_k, hold_period,
            WALK_FORWARD_DATES["train"][0], WALK_FORWARD_DATES["train"][1],
            use_stops=False,
        )
        if res is None:
            continue
        train_results.append({
            "signal_type": signal_type,
            "lookback": lookback,
            "top_k": top_k,
            "hold_period": hold_period,
            **{k: v for k, v in res.items() if k != "period_returns"},
        })
        if done % 18 == 0:
            log(f"  Train progress: {done}/{total}")

    train_df = pd.DataFrame(train_results)
    if train_df.empty:
        log("No valid train results!")
        return None

    train_df.sort_values("sharpe", ascending=False, inplace=True)
    log(f"\nTop 15 configurations by Sharpe (train, no stops):")
    log(train_df.head(15)[
        ["signal_type", "lookback", "top_k", "hold_period",
         "sharpe", "ann_return", "max_drawdown", "calmar", "win_rate"]
    ].to_string(index=False))

    # --- Phase 2: Validate top 15 (both with and without stops) ---
    log(f"\n[VALIDATE] {WALK_FORWARD_DATES['validate'][0]} to {WALK_FORWARD_DATES['validate'][1]}")
    log("-" * 100)

    top_n = min(15, len(train_df))
    validate_results = []
    for _, row in train_df.head(top_n).iterrows():
        for stops_label, use_stops in [("no_stops", False), ("with_stops", True)]:
            res = run_backtest(
                data, row["signal_type"], int(row["lookback"]),
                int(row["top_k"]), int(row["hold_period"]),
                WALK_FORWARD_DATES["validate"][0], WALK_FORWARD_DATES["validate"][1],
                use_stops=use_stops,
            )
            if res is None:
                continue
            validate_results.append({
                "signal_type": row["signal_type"],
                "lookback": int(row["lookback"]),
                "top_k": int(row["top_k"]),
                "hold_period": int(row["hold_period"]),
                "stops": stops_label,
                "train_sharpe": row["sharpe"],
                "val_sharpe": res["sharpe"],
                "val_ann_return": res["ann_return"],
                "val_max_drawdown": res["max_drawdown"],
                "val_calmar": res["calmar"],
                "val_win_rate": res["win_rate"],
            })

    val_df = pd.DataFrame(validate_results)
    if val_df.empty:
        log("No valid validation results!")
        return None

    val_df.sort_values("val_sharpe", ascending=False, inplace=True)
    log(f"\nValidation results (top {top_n} from training, both stop variants):")
    log(val_df[[
        "signal_type", "lookback", "top_k", "hold_period", "stops",
        "train_sharpe", "val_sharpe", "val_ann_return", "val_max_drawdown", "val_calmar"
    ]].to_string(index=False))

    best = val_df.iloc[0]
    use_stops_best = best["stops"] == "with_stops"
    log(f"\n>>> BEST CONFIG (by validation Sharpe):")
    log(f"    Signal: {best['signal_type']}, Lookback: {int(best['lookback'])}, "
        f"K: {int(best['top_k'])}, Hold: {int(best['hold_period'])}, "
        f"Stops: {best['stops']}")
    log(f"    Train Sharpe: {best['train_sharpe']:.3f}, Val Sharpe: {best['val_sharpe']:.3f}")

    # --- Phase 3: Test ---
    log(f"\n[TEST] {WALK_FORWARD_DATES['test'][0]} to {WALK_FORWARD_DATES['test'][1]}")
    log("-" * 100)

    test_res = run_backtest(
        data, best["signal_type"], int(best["lookback"]),
        int(best["top_k"]), int(best["hold_period"]),
        WALK_FORWARD_DATES["test"][0], WALK_FORWARD_DATES["test"][1],
        use_stops=use_stops_best,
    )

    if test_res is None:
        log("Test produced no results!")
        return best, val_df, None

    log(f"\n>>> OUT-OF-SAMPLE TEST RESULTS ({best['stops']}):")
    log(f"    Total Return:   {test_res['total_return']:>8.2%}")
    log(f"    Annual Return:  {test_res['ann_return']:>8.2%}")
    log(f"    Annual Vol:     {test_res['ann_vol']:>8.2%}")
    log(f"    Sharpe Ratio:   {test_res['sharpe']:>8.3f}")
    log(f"    Max Drawdown:   {test_res['max_drawdown']:>8.2%}")
    log(f"    Calmar Ratio:   {test_res['calmar']:>8.3f}")
    log(f"    Win Rate:       {test_res['win_rate']:>8.2%}")
    log(f"    Avg Long Ret:   {test_res['avg_long_ret']:>8.4f}")
    log(f"    Avg Short Ret:  {test_res['avg_short_ret']:>8.4f}")
    log(f"    N Rebalances:   {test_res['n_periods']:>8d}")

    return best, val_df, test_res


# ---------------------------------------------------------------------------
# Full grid search
# ---------------------------------------------------------------------------
def full_grid_search(data, use_stops=True, label="WITH STOPS"):
    """Grid search across full 2021-2026 period."""

    log("\n" + "=" * 100)
    log(f"FULL PERIOD GRID SEARCH -- {label} (2021-01-01 to 2026-05-20)")
    log("=" * 100)

    configs = list(product(SIGNAL_GENERATORS.keys(), LOOKBACKS, TOP_KS, HOLD_PERIODS))
    total = len(configs)
    results = []

    for done, (signal_type, lookback, top_k, hold_period) in enumerate(configs, 1):
        res = run_backtest(
            data, signal_type, lookback, top_k, hold_period,
            "2021-01-01", "2026-05-20",
            use_stops=use_stops,
        )
        if res is None:
            continue
        results.append({
            "signal_type": signal_type,
            "lookback": lookback,
            "top_k": top_k,
            "hold_period": hold_period,
            **{k: v for k, v in res.items() if k != "period_returns"},
        })
        if done % 18 == 0:
            log(f"  [{label}] Progress: {done}/{total} configs tested...")

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df.sort_values("sharpe", ascending=False, inplace=True)
    return results_df


# ---------------------------------------------------------------------------
# No-stops comparison
# ---------------------------------------------------------------------------
def compare_stops(data, best_config):
    """Compare with-stops vs no-stops for best config."""
    log("\n" + "=" * 100)
    log("STOP-LOSS / TAKE-PROFIT IMPACT")
    log("=" * 100)

    st, lb, tk, hp = best_config
    for label, use_stops in [("With stops (SL=-2%, TP=+5%)", True), ("No stops", False)]:
        res = run_backtest(data, st, lb, tk, hp, "2021-01-01", "2026-05-20", use_stops=use_stops)
        if res is None:
            continue
        log(f"\n  {label}:")
        log(f"    Sharpe={res['sharpe']:.3f}, AnnRet={res['ann_return']:.2%}, "
            f"MDD={res['max_drawdown']:.2%}, WR={res['win_rate']:.2%}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    log("Loading futures data...")
    data = load_all_futures()
    n_symbols = data["symbol"].nunique()
    date_range = f"{data['trade_date'].min().date()} to {data['trade_date'].max().date()}"
    log(f"  Loaded {n_symbols} commodities, {len(data)} rows, {date_range}")

    log("\nComputing features and forward returns (with stops)...")
    data = compute_features(data)
    log(f"  Done. {len(data)} rows with all features.")

    # --- Section 1: Full grid search WITH stops ---
    results_df = full_grid_search(data)

    # --- Section 1b: Full grid search WITHOUT stops ---
    results_no_stops = full_grid_search(data, use_stops=False, label="NO-STOPS")

    if not results_df.empty:
        log(f"\n{'=' * 100}")
        log("WITH STOPS (SL=-2%, TP=+5%) -- TOP 30 BY SHARPE (FULL PERIOD 2021-2026)")
        log(f"{'=' * 100}")
        display_cols = [
            "signal_type", "lookback", "top_k", "hold_period",
            "sharpe", "calmar", "ann_return", "ann_vol", "max_drawdown", "win_rate"
        ]
        log(results_df[display_cols].head(30).to_string(index=False))

    if not results_no_stops.empty:
        log(f"\n{'=' * 100}")
        log("NO STOPS -- TOP 30 BY SHARPE (FULL PERIOD 2021-2026)")
        log(f"{'=' * 100}")
        log(results_no_stops[display_cols].head(30).to_string(index=False))

    # Best per signal type (combined: with and without stops)
    log(f"\n{'=' * 100}")
    log("BEST PER SIGNAL TYPE (FULL PERIOD, WITH STOPS)")
    log(f"{'=' * 100}")
    for st in SIGNAL_GENERATORS:
        sub = results_df[results_df["signal_type"] == st]
        if sub.empty:
            continue
        b = sub.iloc[0]
        log(f"\n  {st}:")
        log(f"    Lookback={int(b['lookback'])}, K={int(b['top_k'])}, "
            f"Hold={int(b['hold_period'])}")
        log(f"    Sharpe={b['sharpe']:.3f}, AnnRet={b['ann_return']:.2%}, "
            f"MDD={b['max_drawdown']:.2%}, Calmar={b['calmar']:.3f}, "
            f"WR={b['win_rate']:.2%}")

    log(f"\n{'=' * 100}")
    log("BEST PER SIGNAL TYPE (FULL PERIOD, NO STOPS)")
    log(f"{'=' * 100}")
    for st in SIGNAL_GENERATORS:
        sub = results_no_stops[results_no_stops["signal_type"] == st]
        if sub.empty:
            continue
        b = sub.iloc[0]
        log(f"\n  {st}:")
        log(f"    Lookback={int(b['lookback'])}, K={int(b['top_k'])}, "
            f"Hold={int(b['hold_period'])}")
        log(f"    Sharpe={b['sharpe']:.3f}, AnnRet={b['ann_return']:.2%}, "
            f"MDD={b['max_drawdown']:.2%}, Calmar={b['calmar']:.3f}, "
            f"WR={b['win_rate']:.2%}")

    # Pick the absolute best overall (with or without stops)
    best_with = results_df.iloc[0] if not results_df.empty else None
    best_without = results_no_stops.iloc[0] if not results_no_stops.empty else None
    if best_with is not None and best_without is not None:
        if best_without["sharpe"] > best_with["sharpe"]:
            overall_best = best_without
            best_label = "no stops"
        else:
            overall_best = best_with
            best_label = "with stops"
    elif best_without is not None:
        overall_best = best_without
        best_label = "no stops"
    else:
        overall_best = best_with
        best_label = "with stops"

    # --- Section 2: Walk-forward (using best overall config and also grid) ---
    wf = walk_forward_optimization(data, results_no_stops)

    # --- Section 3: Long vs Short decomposition ---
    log(f"\n{'=' * 100}")
    log("LONG vs SHORT DECOMPOSITION (OVERALL BEST FULL-PERIOD CONFIG)")
    log(f"{'=' * 100}")
    if overall_best is not None:
        b = overall_best
        log(f"\n  Config: {b['signal_type']}/L{int(b['lookback'])}/"
            f"K{int(b['top_k'])}/H{int(b['hold_period'])} ({best_label})")
        log(f"  Avg long-side ret:  {b['avg_long_ret']:.4f}")
        log(f"  Avg short-side ret: {b['avg_short_ret']:.4f}")
        if abs(b['avg_short_ret']) > 1e-6:
            ratio = abs(b['avg_long_ret']) / abs(b['avg_short_ret'])
            log(f"  Long/Short contribution ratio: {ratio:.2f}")

    # --- Section 4: Stops comparison for top config ---
    if overall_best is not None:
        compare_stops(data, (overall_best["signal_type"], int(overall_best["lookback"]),
                             int(overall_best["top_k"]), int(overall_best["hold_period"])))

    # --- Section 5: Summary ---
    log(f"\n{'=' * 100}")
    log("STRATEGY SUMMARY")
    log(f"{'=' * 100}")

    if overall_best is not None:
        b = overall_best
        log(f"\n  Overall full-period best ({best_label}): {b['signal_type']}/"
            f"L{int(b['lookback'])}/K{int(b['top_k'])}/H{int(b['hold_period'])}")
        log(f"    Sharpe={b['sharpe']:.3f}, AnnRet={b['ann_return']:.2%}, "
            f"MDD={b['max_drawdown']:.2%}, Calmar={b['calmar']:.3f}")

    if wf and len(wf) >= 3 and wf[2] is not None:
        best_val, val_df, test_res = wf
        log(f"\n  Walk-forward OOS test: {best_val['signal_type']}/"
            f"L{int(best_val['lookback'])}/K{int(best_val['top_k'])}/H{int(best_val['hold_period'])}")
        log(f"    Train Sharpe={best_val['train_sharpe']:.3f}, "
            f"Val Sharpe={best_val['val_sharpe']:.3f}")
        log(f"    Test Sharpe={test_res['sharpe']:.3f}, "
            f"Test AnnRet={test_res['ann_return']:.2%}, "
            f"Test MDD={test_res['max_drawdown']:.2%}")

    log("\nDone.")


if __name__ == "__main__":
    main()
