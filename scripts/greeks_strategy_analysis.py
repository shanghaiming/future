#!/usr/bin/env python3
"""
Greeks-Based Options Strategy Framework for Chinese Commodity Futures
=====================================================================
Comprehensive analysis of options Greeks across 54 products to identify:
1. Delta-neutral volatility trading opportunities (theta/gamma ratios)
2. Greeks surface analysis for top liquid products
3. Optimal strategy per product based on Greeks profile
4. Risk metrics (gamma exposure, theta decay, vega sensitivity, IV smile)
5. Concrete trade recommendations with actual strikes and prices

KEY DATA NOTE:
  All put options in the dataset have degenerate Greeks (delta=-1, gamma=0, vega=0).
  This is because the IV/Greeks calculator only produced valid results for calls.
  Strategy:
    - Use CALL Greeks for all quantitative analysis (delta, gamma, theta, vega)
    - Use put-call symmetry for straddle analysis: at the same strike,
      gamma_put = gamma_call, vega_put = vega_call, theta_put ~ theta_call
    - Use actual put PRICES for trade construction where puts are needed

Data: ~/home/futures_platform/data/options_calculated/all_options_with_iv.json
"""

import json
import os
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_PATH = os.path.expanduser(
    "~/home/futures_platform/data/options_calculated/all_options_with_iv.json"
)
REPORT_DIR = os.path.expanduser("~/home/futures_platform/output")
os.makedirs(REPORT_DIR, exist_ok=True)

# Chinese commodity futures product names
PRODUCT_NAMES = {
    "AP": "Apple (ZCE)", "CF": "Cotton (ZCE)", "CJ": "Cashew (ZCE)",
    "FG": "Flat Glass (ZCE)", "MA": "Methanol (ZCE)", "OI": "Oleic Acid (ZCE)",
    "PF": "Polyester Filament (ZCE)", "PK": "Palm Kernel (DCE)",
    "PL": "PP (GFEX)", "PR": "Propane (DCE)", "PX": "Paraxylene (GFEX)",
    "RM": "Rapeseed Meal (ZCE)", "SA": "Soda Ash (ZCE)",
    "SF": "Silicon Fe (ZCE)", "SH": "Caustic Soda (ZCE)",
    "SM": "Silicomanganese (DCE)", "SR": "Sugar (ZCE)", "TA": "PTA (ZCE)",
    "UR": "Urea (ZCE)", "a": "Soybean-1 (DCE)", "ad": "Adipic Acid (ZCE)",
    "ao": "Alumina (SHFE)", "au": "Gold (SHFE)", "b": "Soybean-2 (DCE)",
    "bc": "Baltic Dry (INE)", "br": "Butadiene Rubber (SHFE)",
    "bu": "Bitumen (SHFE)", "bz": "Benzene (ZCE)", "c": "Corn (DCE)",
    "cs": "Corn Starch (DCE)", "cu": "Copper (SHFE)", "eb": "EB (DCE)",
    "eg": "Ethylene Glycol (DCE)", "fu": "Fuel Oil (SHFE)",
    "jm": "Coking Coal (DCE)", "l": "Linear PE (DCE)", "m": "Soybean Meal (DCE)",
    "ni": "Nickel (SHFE)", "nr": "NR Rubber (SHFE)", "op": "Options (GFEX)",
    "pb": "Lead (SHFE)", "pd": "Palladium (GFEX)", "pp": "PP (DCE)",
    "ps": "Polystyrene (GFEX)", "pt": "Bottled Water (GFEX)",
    "rb": "Rebar (SHFE)", "ru": "Rubber (SHFE)", "sc": "Crude Oil (INE)",
    "si": "Silicon Steel (GFEX)", "sn": "Tin (SHFE)", "sp": "Pulp (SHFE)",
    "v": "PVC (DCE)", "y": "Soybean Oil (DCE)", "zn": "Zinc (SHFE)",
}


# ---------------------------------------------------------------------------
# Data Loading & Enrichment
# ---------------------------------------------------------------------------
def load_data():
    with open(DATA_PATH) as f:
        raw = json.load(f)
    df = pd.DataFrame(raw)

    # All valid Greeks come from CALLs only.
    # Puts have degenerate Greeks (delta=-1, gamma=0, vega=0).
    df["has_real_greeks"] = (
        (df["gamma"] > 0) & (df["vega"] > 0) & (df["implied_vol"] > 0)
    )

    # Build lookup: for each (product, expiry, strike), find call Greeks
    # so we can synthesize put Greeks via put-call symmetry
    calls = df[(df["option_type"] == "CALL") & df["has_real_greeks"]].copy()
    call_greeks_map = {}
    for _, row in calls.iterrows():
        key = (row["product"], row["expiry_date"], row["strike"])
        call_greeks_map[key] = {
            "call_delta": row["delta"],
            "call_gamma": row["gamma"],
            "call_theta": row["theta"],
            "call_vega": row["vega"],
            "call_iv": row["implied_vol"],
            "call_price": row["market_price"],
        }

    # Enrich puts with synthesized Greeks from call at same strike
    # By put-call symmetry at same K:
    #   gamma_put = gamma_call
    #   vega_put  = vega_call
    #   theta_put ~ theta_call  (approximately, differs by r*K*e^{-rT} term)
    #   delta_put = delta_call - 1
    put_mask = df["option_type"] == "PUT"
    for idx in df[put_mask].index:
        key = (df.at[idx, "product"], df.at[idx, "expiry_date"], df.at[idx, "strike"])
        if key in call_greeks_map:
            cg = call_greeks_map[key]
            df.at[idx, "gamma"] = cg["call_gamma"]
            df.at[idx, "vega"] = cg["call_vega"]
            df.at[idx, "theta"] = cg["call_theta"]  # approximate
            df.at[idx, "delta"] = cg["call_delta"] - 1.0
            df.at[idx, "implied_vol"] = cg["call_iv"]
            df.at[idx, "has_real_greeks"] = True

    return df


# ===========================================================================
# SECTION 1: Delta-Neutral Volatility Trading Analysis
# ===========================================================================
def analyze_delta_neutral(df):
    """
    For each product, construct a theoretical ATM straddle using
    call + synthesized put Greeks. Compute theta/gamma ratio.
    """
    print("\n" + "=" * 80)
    print("SECTION 1: DELTA-NEUTRAL VOLATILITY TRADING ANALYSIS")
    print("=" * 80)

    valid = df[df["has_real_greeks"]].copy()
    # Use only CALL options for primary analysis
    calls = valid[valid["option_type"] == "CALL"].copy()

    results = []
    for product, grp in calls.groupby("product"):
        if len(grp) < 2:
            continue

        # For each expiry, find ATM call
        for dte, exp_grp in grp.groupby("days_to_expiry"):
            # Find ATM call (moneyness closest to 1.0)
            atm_idx = (exp_grp["moneyness"] - 1.0).abs().idxmin()
            atm_call = exp_grp.loc[atm_idx]

            underlying = atm_call["underlying_price"]
            atm_strike = atm_call["strike"]

            # Synthesized straddle metrics:
            # ATM call delta ~0.5, ATM put delta ~-0.5 => straddle delta ~0
            # Gamma doubles: straddle_gamma = 2 * call_gamma
            # Theta doubles: straddle_theta = 2 * call_theta
            # Vega doubles:  straddle_vega = 2 * call_vega
            straddle_gamma = 2.0 * atm_call["gamma"]
            straddle_theta = 2.0 * atm_call["theta"]  # both negative
            straddle_vega = 2.0 * atm_call["vega"]
            straddle_delta = atm_call["delta"] + (atm_call["delta"] - 1.0)  # call + put

            # Straddle price: use call price + actual put price if available,
            # otherwise estimate via put-call parity: P = C - S + K*e^{-rT}
            # Simplified: P ~ C - (S - K), but for ATM P ~ C
            put_data = valid[
                (valid["product"] == product) &
                (valid["days_to_expiry"] == dte) &
                (valid["option_type"] == "PUT") &
                (valid["strike"] == atm_strike)
            ]
            if not put_data.empty:
                put_price = put_data.iloc[0]["market_price"]
            else:
                # Estimate: for ATM, put ~ call (small difference for low rates)
                put_price = atm_call["market_price"]

            straddle_price = atm_call["market_price"] + put_price

            if straddle_gamma <= 0:
                continue

            theta_gamma_ratio = abs(straddle_theta) / straddle_gamma
            vega_gamma_ratio = straddle_vega / straddle_gamma
            theta_pct_premium = abs(straddle_theta) / straddle_price * 100 if straddle_price > 0 else 0
            vega_pct_premium = straddle_vega / straddle_price * 100 if straddle_price > 0 else 0

            results.append({
                "product": product,
                "name": PRODUCT_NAMES.get(product, product),
                "expiry": atm_call["expiry_date"],
                "dte": dte,
                "underlying": underlying,
                "atm_strike": atm_strike,
                "call_price": atm_call["market_price"],
                "put_price": put_price,
                "straddle_price": straddle_price,
                "straddle_delta": straddle_delta,
                "straddle_gamma": straddle_gamma,
                "straddle_theta": straddle_theta,
                "straddle_vega": straddle_vega,
                "theta_gamma_ratio": theta_gamma_ratio,
                "vega_gamma_ratio": vega_gamma_ratio,
                "theta_pct_premium": theta_pct_premium,
                "vega_pct_premium": vega_pct_premium,
                "implied_vol": atm_call["implied_vol"],
            })

    if not results:
        print("No valid products found for delta-neutral analysis.")
        return pd.DataFrame()

    results_df = pd.DataFrame(results)

    # For each product, keep the nearest-expiry result as the primary signal
    results_df = results_df.sort_values(["product", "dte"]).drop_duplicates("product")
    results_df = results_df.sort_values("theta_gamma_ratio")

    # Classification by terciles
    n = len(results_df)
    if n >= 6:
        low_thresh = results_df["theta_gamma_ratio"].quantile(0.33)
        high_thresh = results_df["theta_gamma_ratio"].quantile(0.67)
    else:
        low_thresh = results_df["theta_gamma_ratio"].median()
        high_thresh = results_df["theta_gamma_ratio"].median()

    results_df["strategy"] = results_df["theta_gamma_ratio"].apply(
        lambda x: "BUY_OPTIONS" if x <= low_thresh
        else ("SELL_OPTIONS" if x >= high_thresh else "NEUTRAL")
    )

    # Print summary
    print(f"\nAnalyzed {len(results_df)} products (nearest-expiry ATM straddle per product)")
    print(f"Theta/Gamma ratio thresholds: BUY <= {low_thresh:.1f}, SELL >= {high_thresh:.1f}")

    print("\n--- TOP GAMMA SCALPING TARGETS (Low Theta/Gamma = Cheap Gamma) ---")
    buy_df = results_df[results_df["strategy"] == "BUY_OPTIONS"].head(15)
    if not buy_df.empty:
        print(buy_df[["product", "name", "dte", "straddle_price", "straddle_gamma",
                      "straddle_theta", "theta_gamma_ratio", "implied_vol"]].to_string(index=False))

    print("\n--- TOP THETA SELLING TARGETS (High Theta/Gamma = Expensive Gamma) ---")
    sell_df = results_df[results_df["strategy"] == "SELL_OPTIONS"].head(15)
    if not sell_df.empty:
        print(sell_df[["product", "name", "dte", "straddle_price", "straddle_gamma",
                       "straddle_theta", "theta_gamma_ratio", "implied_vol"]].to_string(index=False))

    print("\n--- FULL PRODUCT RANKING BY THETA/GAMMA RATIO ---")
    pd.set_option("display.max_rows", 100)
    print(results_df[["product", "name", "strategy", "dte", "straddle_price",
                      "theta_gamma_ratio", "implied_vol"]].to_string(index=False))

    return results_df


# ===========================================================================
# SECTION 2: Greeks Surface Analysis (Top 10 Products)
# ===========================================================================
def analyze_greeks_surface(df, top_n=10):
    """
    For the top N products by record count, analyze how Greeks vary
    across strikes and expiries.
    """
    print("\n" + "=" * 80)
    print("SECTION 2: GREEKS SURFACE ANALYSIS (Top Products)")
    print("=" * 80)

    valid = df[df["has_real_greeks"] & (df["option_type"] == "CALL")].copy()
    product_counts = valid.groupby("product").size().nlargest(top_n)
    top_products = product_counts.index.tolist()

    print(f"\nTop {top_n} products by valid call record count:")
    for p, c in product_counts.items():
        print(f"  {p} ({PRODUCT_NAMES.get(p, p)}): {c} records")

    surface_analysis = {}

    for product in top_products:
        prod_data = valid[valid["product"] == product].copy()
        underlying = prod_data["underlying_price"].iloc[0]
        print(f"\n{'=' * 60}")
        print(f"PRODUCT: {product} ({PRODUCT_NAMES.get(product, product)})")
        print(f"  Underlying: {underlying:.2f}")
        print(f"  IV range: {prod_data['implied_vol'].min():.4f} - {prod_data['implied_vol'].max():.4f}")
        print(f"  DTE range: {prod_data['days_to_expiry'].min()} - {prod_data['days_to_expiry'].max()}")
        print(f"  Strikes: {prod_data['strike'].min():.0f} - {prod_data['strike'].max():.0f}")

        # For each expiry, show full surface
        expiry_stats = []
        for dte in sorted(prod_data["days_to_expiry"].unique()):
            exp_grp = prod_data[prod_data["days_to_expiry"] == dte].sort_values("strike")

            # Find ATM call
            atm_idx = (exp_grp["moneyness"] - 1.0).abs().idxmin()
            atm_c = exp_grp.loc[atm_idx]

            # Gamma peak
            gamma_peak = exp_grp.loc[exp_grp["gamma"].idxmax()]
            is_atm_gamma = 0.95 <= gamma_peak["moneyness"] <= 1.05

            # Stats
            delta_range = f"{exp_grp['delta'].min():.3f} to {exp_grp['delta'].max():.3f}"

            # Most negative theta (highest time decay)
            max_theta_row = exp_grp.loc[exp_grp["theta"].idxmin()]
            min_theta_row = exp_grp.loc[exp_grp["theta"].idxmax()]
            theta_info = (
                f"min={max_theta_row['theta']:.4f} (K={max_theta_row['strike']:.0f})"
                f", max={min_theta_row['theta']:.4f} (K={min_theta_row['strike']:.0f})"
            )

            # Max vega
            max_vega_row = exp_grp.loc[exp_grp["vega"].idxmax()]
            vega_info = (
                f"max={max_vega_row['vega']:.4f} at K={max_vega_row['strike']:.0f}"
                f" (m={max_vega_row['moneyness']:.3f})"
            )

            expiry_stats.append({
                "dte": dte,
                "n_strikes": len(exp_grp),
                "atm_iv": atm_c["implied_vol"],
                "atm_delta": atm_c["delta"],
                "gamma_peak_strike": gamma_peak["strike"],
                "gamma_peak_moneyness": gamma_peak["moneyness"],
                "delta_range": delta_range,
            })

            print(f"\n  Expiry DTE={dte} ({len(exp_grp)} calls):")
            print(f"    ATM: IV={atm_c['implied_vol']:.4f}, delta={atm_c['delta']:.4f}, "
                  f"gamma={atm_c['gamma']:.6f}, theta={atm_c['theta']:.4f}, vega={atm_c['vega']:.4f}")
            print(f"    Gamma peak: K={gamma_peak['strike']:.0f} (m={gamma_peak['moneyness']:.3f})"
                  f" {'[ATM confirmed]' if is_atm_gamma else '[OFF-CENTER]'}")
            print(f"    Delta range: {delta_range}")
            print(f"    Theta: {theta_info}")
            print(f"    Vega: {vega_info}")

        # Detailed IV smile for nearest expiry
        nearest_dte = sorted(prod_data["days_to_expiry"].unique())[0]
        near_data = prod_data[prod_data["days_to_expiry"] == nearest_dte].sort_values("strike")
        print(f"\n  DETAILED STRIKE-BY-STRIKE (DTE={nearest_dte}):")
        print(f"    {'K':>10s} {'Money':>6s} {'Price':>8s} {'IV':>7s} "
              f"{'Delta':>7s} {'Gamma':>10s} {'Theta':>8s} {'Vega':>8s}")
        for _, row in near_data.iterrows():
            marker = " << ATM" if abs(row["moneyness"] - 1.0) < 0.02 else ""
            print(f"    {row['strike']:>10.0f} {row['moneyness']:>6.3f} {row['market_price']:>8.2f} "
                  f"{row['implied_vol']:>7.4f} {row['delta']:>+7.4f} {row['gamma']:>10.6f} "
                  f"{row['theta']:>+8.4f} {row['vega']:>8.4f}{marker}")

        surface_analysis[product] = expiry_stats

    return surface_analysis, top_products


# ===========================================================================
# SECTION 3: Optimal Strategy Per Product
# ===========================================================================
def recommend_strategy_per_product(df, delta_neutral_results):
    """
    Based on Greeks profile, recommend optimal strategy per product.
    """
    print("\n" + "=" * 80)
    print("SECTION 3: OPTIMAL STRATEGY PER PRODUCT")
    print("=" * 80)

    valid = df[df["has_real_greeks"] & (df["option_type"] == "CALL")].copy()
    recommendations = []

    for product, grp in valid.groupby("product"):
        underlying = grp["underlying_price"].iloc[0]

        avg_iv = grp["implied_vol"].mean()
        max_gamma_row = grp.loc[grp["gamma"].idxmax()]
        max_vega_row = grp.loc[grp["vega"].idxmax()]

        dn_row = delta_neutral_results[delta_neutral_results["product"] == product]
        if dn_row.empty:
            continue
        dn = dn_row.iloc[0]
        tg_ratio = dn["theta_gamma_ratio"]
        strategy_class = dn["strategy"]
        straddle_price = dn["straddle_price"]

        # Determine specific strategy
        cross_iv_median = valid.groupby("product")["implied_vol"].mean().median()

        if strategy_class == "BUY_OPTIONS":
            if avg_iv < cross_iv_median:
                specific = "LONG VEGA + GAMMA SCALPING"
                desc = (
                    f"Low IV ({avg_iv:.1%} vs median {cross_iv_median:.1%}) with cheap "
                    f"theta/gamma ({tg_ratio:.1f}). Buy ATM straddle, delta-hedge daily. "
                    f"Benefits from IV expansion + realized vol > implied."
                )
            else:
                specific = "GAMMA SCALPING"
                desc = (
                    f"Low theta/gamma ({tg_ratio:.1f}) means cheap time decay for convexity. "
                    f"Buy ATM straddle at K={dn['atm_strike']:.0f}, delta-hedge intraday. "
                    f"Collect rebalancing P&L from realized vol > implied vol."
                )
        elif strategy_class == "SELL_OPTIONS":
            if avg_iv > cross_iv_median:
                specific = "SHORT VOLATILITY (IRON CONDOR)"
                desc = (
                    f"High IV ({avg_iv:.1%}) + expensive theta/gamma ({tg_ratio:.1f}). "
                    f"Sell OTM strangle, buy wings. Collect rich premium with bounded gamma."
                )
            else:
                specific = "THETA COLLECTION (CREDIT SPREAD)"
                desc = (
                    f"Expensive theta/gamma ({tg_ratio:.1f}). Sell near-ATM call spread. "
                    f"Avoid selling at K={max_gamma_row['strike']:.0f} (max gamma). "
                    f"Credit spread caps risk."
                )
        else:
            specific = "NEUTRAL / CALENDAR SPREAD"
            desc = (
                "Balanced Greeks. Consider calendar spreads (sell near-term, buy far-term) "
                "to exploit term structure."
            )

        # Per-product risk stats
        atm_grp = grp[(grp["moneyness"] >= 0.97) & (grp["moneyness"] <= 1.03)]
        if not atm_grp.empty:
            avg_theta_pct = (atm_grp["theta"].abs() / atm_grp["market_price"] * 100).mean()
            avg_vega_pct = (atm_grp["vega"] / atm_grp["market_price"] * 100).mean()
        else:
            avg_theta_pct = avg_vega_pct = 0.0

        recommendations.append({
            "product": product,
            "name": PRODUCT_NAMES.get(product, product),
            "strategy_class": strategy_class,
            "specific_strategy": specific,
            "description": desc,
            "avg_iv": avg_iv,
            "theta_gamma_ratio": tg_ratio,
            "straddle_price": straddle_price,
            "max_gamma_strike": max_gamma_row["strike"],
            "max_vega_strike": max_vega_row["strike"],
            "avg_theta_pct": avg_theta_pct,
            "avg_vega_pct": avg_vega_pct,
        })

    rec_df = pd.DataFrame(recommendations)

    for strat in ["BUY_OPTIONS", "SELL_OPTIONS", "NEUTRAL"]:
        subset = rec_df[rec_df["strategy_class"] == strat]
        if subset.empty:
            continue
        label = {
            "BUY_OPTIONS": "BUY OPTIONS (Gamma Scalping / Long Vol)",
            "SELL_OPTIONS": "SELL OPTIONS (Theta Collection / Short Vol)",
            "NEUTRAL": "NEUTRAL",
        }[strat]
        print(f"\n--- {label} ({len(subset)} products) ---")
        for _, row in subset.iterrows():
            print(f"\n  {row['product']} ({row['name']})")
            print(f"    Strategy: {row['specific_strategy']}")
            print(f"    {row['description']}")
            print(f"    Avg IV: {row['avg_iv']:.2%}, Theta/Gamma ratio: {row['theta_gamma_ratio']:.1f}")
            print(f"    ATM straddle cost: {row['straddle_price']:.2f}")
            print(f"    Max gamma at K={row['max_gamma_strike']:.0f}, Max vega at K={row['max_vega_strike']:.0f}")

    return rec_df


# ===========================================================================
# SECTION 4: Risk Metrics
# ===========================================================================
def calculate_risk_metrics(df):
    """
    Per-product risk metrics:
    - Max gamma exposure (at what strike/expiry)
    - Theta decay per day as % of premium
    - Vega per 1% IV change as % of premium
    - Implied probability distribution (Breeden-Litzenberger)
    """
    print("\n" + "=" * 80)
    print("SECTION 4: RISK METRICS")
    print("=" * 80)

    valid = df[df["has_real_greeks"] & (df["option_type"] == "CALL")].copy()
    risk_data = []

    for product, grp in valid.groupby("product"):
        underlying = grp["underlying_price"].iloc[0]

        # Max gamma
        max_gamma_row = grp.loc[grp["gamma"].idxmax()]
        max_gamma = grp["gamma"].max()

        # ATM options for theta/vega % metrics
        atm_options = grp[(grp["moneyness"] >= 0.97) & (grp["moneyness"] <= 1.03)]
        if atm_options.empty:
            # Widen to 0.90-1.10
            atm_options = grp[(grp["moneyness"] >= 0.90) & (grp["moneyness"] <= 1.10)]
        if atm_options.empty:
            continue

        theta_decay_pct = (atm_options["theta"].abs() / atm_options["market_price"] * 100).mean()
        vega_pct = (atm_options["vega"] / atm_options["market_price"] * 100).mean()

        # IV stats
        iv_mean = grp["implied_vol"].mean()
        iv_min = grp["implied_vol"].min()
        iv_max = grp["implied_vol"].max()
        iv_range = iv_max - iv_min

        # Breeden-Litzenberger implied distribution
        nearest_dte = sorted(grp["days_to_expiry"].unique())[0]
        near_calls = grp[grp["days_to_expiry"] == nearest_dte].sort_values("strike")
        bl_result = _breeden_litzenberger(near_calls, underlying)

        risk_data.append({
            "product": product,
            "name": PRODUCT_NAMES.get(product, product),
            "underlying": underlying,
            "max_gamma": max_gamma,
            "max_gamma_strike": max_gamma_row["strike"],
            "max_gamma_dte": max_gamma_row["days_to_expiry"],
            "max_gamma_expiry": max_gamma_row["expiry_date"],
            "theta_decay_pct": theta_decay_pct,
            "vega_pct": vega_pct,
            "iv_mean": iv_mean,
            "iv_min": iv_min,
            "iv_max": iv_max,
            "iv_range": iv_range,
            "bl_mode": bl_result["mode"],
            "bl_mode_pct": bl_result["mode_pct"],
            "bl_std": bl_result["std"],
            "bl_skew": bl_result["skew"],
        })

    risk_df = pd.DataFrame(risk_data)

    print(f"\nRisk metrics computed for {len(risk_df)} products")

    print("\n--- TOP 15: HIGHEST GAMMA EXPOSURE ---")
    print(risk_df.nlargest(15, "max_gamma")[
        ["product", "name", "max_gamma", "max_gamma_strike", "max_gamma_dte", "underlying"]
    ].to_string(index=False))

    print("\n--- TOP 15: FASTEST THETA DECAY (% of premium/day) ---")
    print(risk_df.nlargest(15, "theta_decay_pct")[
        ["product", "name", "theta_decay_pct", "vega_pct", "iv_mean"]
    ].to_string(index=False))

    print("\n--- TOP 15: HIGHEST VEGA SENSITIVITY (% of premium per 1% IV move) ---")
    print(risk_df.nlargest(15, "vega_pct")[
        ["product", "name", "vega_pct", "iv_mean", "iv_range"]
    ].to_string(index=False))

    print("\n--- IMPLIED DISTRIBUTION (Breeden-Litzenberger) ---")
    print(risk_df[["product", "name", "bl_mode", "bl_mode_pct", "bl_std", "bl_skew"]].to_string(index=False))

    return risk_df


def _breeden_litzenberger(calls_df, underlying_price):
    """
    Extract risk-neutral probability density from call option prices.
    f(K) ~ e^{rT} * C''(K). With r~0, f(K) ~ C''(K).
    """
    result = {"mode": None, "mode_pct": None, "std": None, "skew": None}
    if len(calls_df) < 5:
        return result

    strikes = calls_df["strike"].values.astype(float)
    prices = calls_df["market_price"].values.astype(float)

    dk = np.diff(strikes)
    if len(dk) == 0 or dk.min() <= 0:
        return result

    prob_density = []
    prob_strikes = []
    for i in range(1, len(strikes) - 1):
        dk_prev = strikes[i] - strikes[i - 1]
        dk_next = strikes[i + 1] - strikes[i]
        if dk_prev <= 0 or dk_next <= 0:
            continue
        d2c = (prices[i + 1] - 2 * prices[i] + prices[i - 1]) / ((dk_prev + dk_next) / 2) ** 2
        if d2c > 0:
            prob_density.append(d2c)
            prob_strikes.append(strikes[i])

    if len(prob_density) < 3:
        return result

    prob_density = np.array(prob_density)
    prob_strikes = np.array(prob_strikes)

    dk_mean = np.mean(np.diff(prob_strikes))
    total_prob = np.sum(prob_density * dk_mean)
    if total_prob <= 0:
        return result
    prob_norm = prob_density / total_prob

    mode_idx = np.argmax(prob_norm)
    mode = prob_strikes[mode_idx]
    mode_pct = (mode - underlying_price) / underlying_price * 100

    mean_val = np.sum(prob_strikes * prob_norm * dk_mean)
    variance = np.sum((prob_strikes - mean_val) ** 2 * prob_norm * dk_mean)
    std_val = np.sqrt(variance) if variance > 0 else 0

    skewness = 0
    if std_val > 0:
        skewness = np.sum((prob_strikes - mean_val) ** 3 * prob_norm * dk_mean) / std_val ** 3

    return {
        "mode": round(mode, 2),
        "mode_pct": round(mode_pct, 2),
        "std": round(std_val, 2),
        "skew": round(skewness, 3),
    }


# ===========================================================================
# SECTION 5: Concrete Trade Recommendations
# ===========================================================================
def generate_trade_recommendations(df, delta_neutral_results, risk_df):
    """
    Generate specific, actionable trade ideas with actual strikes and prices.
    All Greeks come from calls; put prices are actual market prices.
    """
    print("\n" + "=" * 80)
    print("SECTION 5: CONCRETE TRADE RECOMMENDATIONS")
    print("=" * 80)

    valid_calls = df[df["has_real_greeks"] & (df["option_type"] == "CALL")].copy()
    all_puts = df[df["option_type"] == "PUT"].copy()

    trades = []

    for product, grp in valid_calls.groupby("product"):
        underlying = grp["underlying_price"].iloc[0]
        dn_row = delta_neutral_results[delta_neutral_results["product"] == product]
        if dn_row.empty:
            continue
        dn = dn_row.iloc[0]
        strat = dn["strategy"]
        iv = dn["implied_vol"]

        # Nearest expiry
        nearest_dte = sorted(grp["days_to_expiry"].unique())[0]
        near_calls = grp[grp["days_to_expiry"] == nearest_dte].sort_values("strike")
        near_puts = all_puts[
            (all_puts["product"] == product) &
            (all_puts["days_to_expiry"] == nearest_dte)
        ].sort_values("strike")

        if len(near_calls) < 3:
            continue

        # Find ATM call
        atm_idx = (near_calls["moneyness"] - 1.0).abs().idxmin()
        atm_call = near_calls.loc[atm_idx]
        atm_strike = atm_call["strike"]

        # Helper: find closest put by strike
        def find_put(strike):
            if near_puts.empty:
                return None
            diffs = (near_puts["strike"] - strike).abs()
            return near_puts.loc[diffs.idxmin()]

        # --- SELL OPTIONS strategies ---
        if strat == "SELL_OPTIONS":
            # 1) Bear Call Spread: sell ATM call, buy OTM call
            otm_calls = near_calls[near_calls["strike"] > atm_strike]
            if len(otm_calls) >= 1:
                buy_call = otm_calls.iloc[0]
                sell_call = atm_call
                net_credit = sell_call["market_price"] - buy_call["market_price"]
                spread_width = buy_call["strike"] - sell_call["strike"]
                max_loss = spread_width - net_credit
                if net_credit > 0 and max_loss > 0:
                    trades.append({
                        "product": product,
                        "name": PRODUCT_NAMES.get(product, product),
                        "strategy": "BEAR CALL SPREAD",
                        "dte": nearest_dte,
                        "expiry": sell_call["expiry_date"],
                        "legs": (
                            f"Sell {sell_call['strike']:.0f}C @ {sell_call['market_price']:.2f}, "
                            f"Buy {buy_call['strike']:.0f}C @ {buy_call['market_price']:.2f}"
                        ),
                        "net_premium": net_credit,
                        "max_loss": max_loss,
                        "risk_reward": round(net_credit / max_loss, 3),
                        "delta": sell_call["delta"] - buy_call["delta"],
                        "theta": sell_call["theta"] - buy_call["theta"],
                        "vega": sell_call["vega"] - buy_call["vega"],
                        "iv": iv,
                        "underlying": underlying,
                        "rationale": (
                            f"High theta/gamma ({dn['theta_gamma_ratio']:.1f}). "
                            f"Sell expensive time decay. IV={iv:.1%}. "
                            f"Defined risk credit spread."
                        ),
                    })

            # 2) Bull Put Spread: sell OTM put (above), buy further OTM put
            if len(near_puts) >= 2:
                # Find puts that are slightly OTM (strike < underlying)
                otm_puts = near_puts[(near_puts["strike"] < underlying)].sort_values("strike", ascending=False)
                if len(otm_puts) >= 2:
                    sell_put = otm_puts.iloc[0]
                    buy_put = otm_puts.iloc[1]
                    net_credit = sell_put["market_price"] - buy_put["market_price"]
                    spread_width = sell_put["strike"] - buy_put["strike"]
                    max_loss = spread_width - net_credit
                    if net_credit > 0 and max_loss > 0:
                        trades.append({
                            "product": product,
                            "name": PRODUCT_NAMES.get(product, product),
                            "strategy": "BULL PUT SPREAD",
                            "dte": nearest_dte,
                            "expiry": sell_put["expiry_date"],
                            "legs": (
                                f"Sell {sell_put['strike']:.0f}P @ {sell_put['market_price']:.2f}, "
                                f"Buy {buy_put['strike']:.0f}P @ {buy_put['market_price']:.2f}"
                            ),
                            "net_premium": net_credit,
                            "max_loss": max_loss,
                            "risk_reward": round(net_credit / max_loss, 3),
                            "delta": None,
                            "theta": None,
                            "vega": None,
                            "iv": iv,
                            "underlying": underlying,
                            "rationale": (
                                f"Theta/gamma={dn['theta_gamma_ratio']:.1f}. "
                                f"Credit put spread, bullish bias. "
                                f"Profit if underlying stays above {sell_put['strike']:.0f}."
                            ),
                        })

            # 3) Iron Condor: call spread + put spread
            if len(otm_calls) >= 2 and len(near_puts) >= 4:
                deep_otm_calls = otm_calls.iloc[1:]  # skip first (used for call spread wing)
                # For call side: sell first OTM, buy second OTM
                cs_sell = otm_calls.iloc[0]
                cs_buy = otm_calls.iloc[1] if len(otm_calls) >= 2 else None

                # For put side: find 2 OTM puts (strike < underlying)
                otm_puts_for_ic = near_puts[near_puts["strike"] < underlying].sort_values(
                    "strike", ascending=False
                )
                if len(otm_puts_for_ic) >= 2 and cs_buy is not None:
                    ps_sell = otm_puts_for_ic.iloc[0]
                    ps_buy = otm_puts_for_ic.iloc[1]

                    call_cr = cs_sell["market_price"] - cs_buy["market_price"]
                    put_cr = ps_sell["market_price"] - ps_buy["market_price"]
                    net_credit_ic = call_cr + put_cr
                    call_width = cs_buy["strike"] - cs_sell["strike"]
                    put_width = ps_sell["strike"] - ps_buy["strike"]
                    max_loss_ic = max(call_width, put_width) - net_credit_ic

                    if net_credit_ic > 0 and max_loss_ic > 0:
                        trades.append({
                            "product": product,
                            "name": PRODUCT_NAMES.get(product, product),
                            "strategy": "IRON CONDOR",
                            "dte": nearest_dte,
                            "expiry": cs_sell["expiry_date"],
                            "legs": (
                                f"Sell {cs_sell['strike']:.0f}C @ {cs_sell['market_price']:.2f}, "
                                f"Buy {cs_buy['strike']:.0f}C @ {cs_buy['market_price']:.2f}, "
                                f"Sell {ps_sell['strike']:.0f}P @ {ps_sell['market_price']:.2f}, "
                                f"Buy {ps_buy['strike']:.0f}P @ {ps_buy['market_price']:.2f}"
                            ),
                            "net_premium": net_credit_ic,
                            "max_loss": max_loss_ic,
                            "risk_reward": round(net_credit_ic / max_loss_ic, 3),
                            "delta": (cs_sell["delta"] - cs_buy["delta"]),
                            "theta": (cs_sell["theta"] - cs_buy["theta"]),
                            "vega": (cs_sell["vega"] - cs_buy["vega"]),
                            "iv": iv,
                            "underlying": underlying,
                            "rationale": (
                                f"Iron condor profits from time decay in both directions. "
                                f"IV={iv:.1%}, TG={dn['theta_gamma_ratio']:.1f}. "
                                f"Profit zone: {ps_sell['strike']:.0f} - {cs_sell['strike']:.0f}."
                            ),
                        })

        # --- BUY OPTIONS strategies ---
        elif strat == "BUY_OPTIONS":
            # 1) ATM Straddle: buy call + put at ATM strike
            put_atm = find_put(atm_strike)
            call_price = atm_call["market_price"]
            put_price = put_atm["market_price"] if put_atm is not None else call_price
            straddle_cost = call_price + put_price

            # Greeks (using call Greeks * 2 for straddle)
            straddle_gamma = 2.0 * atm_call["gamma"]
            straddle_theta = 2.0 * atm_call["theta"]
            straddle_vega = 2.0 * atm_call["vega"]
            straddle_delta = atm_call["delta"] + (atm_call["delta"] - 1.0)
            be_pct = straddle_cost / underlying * 100

            trades.append({
                "product": product,
                "name": PRODUCT_NAMES.get(product, product),
                "strategy": "LONG ATM STRADDLE",
                "dte": nearest_dte,
                "expiry": atm_call["expiry_date"],
                "legs": (
                    f"Buy {atm_call['strike']:.0f}C @ {call_price:.2f}, "
                    f"Buy {atm_call['strike']:.0f}P @ {put_price:.2f}"
                ),
                "net_premium": -straddle_cost,
                "max_loss": straddle_cost,
                "risk_reward": None,
                "delta": straddle_delta,
                "theta": straddle_theta,
                "vega": straddle_vega,
                "iv": iv,
                "underlying": underlying,
                "rationale": (
                    f"Low theta/gamma ({dn['theta_gamma_ratio']:.1f}) = cheap convexity. "
                    f"Breakeven: +/-{be_pct:.1f}% from {underlying:.0f}. "
                    f"Gamma scalp by delta-hedging daily."
                ),
            })

            # 2) Bull Call Spread (ITM call buy, OTM call sell)
            itm_calls = near_calls[near_calls["strike"] < atm_strike]
            otm_calls = near_calls[near_calls["strike"] > atm_strike]
            if len(itm_calls) >= 1 and len(otm_calls) >= 1:
                long_call = itm_calls.iloc[-1]  # deepest ITM (just below ATM)
                short_call = otm_calls.iloc[0]   # just OTM
                net_debit = long_call["market_price"] - short_call["market_price"]
                max_profit = short_call["strike"] - long_call["strike"] - net_debit
                if net_debit > 0 and max_profit > 0:
                    trades.append({
                        "product": product,
                        "name": PRODUCT_NAMES.get(product, product),
                        "strategy": "BULL CALL SPREAD",
                        "dte": nearest_dte,
                        "expiry": long_call["expiry_date"],
                        "legs": (
                            f"Buy {long_call['strike']:.0f}C @ {long_call['market_price']:.2f}, "
                            f"Sell {short_call['strike']:.0f}C @ {short_call['market_price']:.2f}"
                        ),
                        "net_premium": -net_debit,
                        "max_loss": net_debit,
                        "risk_reward": round(max_profit / net_debit, 3),
                        "delta": long_call["delta"] - short_call["delta"],
                        "theta": long_call["theta"] - short_call["theta"],
                        "vega": long_call["vega"] - short_call["vega"],
                        "iv": iv,
                        "underlying": underlying,
                        "rationale": (
                            f"Low IV ({iv:.1%}) environment. Bullish spread, defined risk. "
                            f"TG={dn['theta_gamma_ratio']:.1f}, cheap to hold."
                        ),
                    })

            # 3) Calendar spread if multiple expiries available
            all_dtes = sorted(grp["days_to_expiry"].unique())
            if len(all_dtes) >= 2:
                near_dte = all_dtes[0]
                far_dte = all_dtes[1] if len(all_dtes) > 1 else None
                if far_dte is not None:
                    near_atm = grp[
                        (grp["days_to_expiry"] == near_dte) &
                        (grp["option_type"] == "CALL")
                    ]
                    far_atm = grp[
                        (grp["days_to_expiry"] == far_dte) &
                        (grp["option_type"] == "CALL")
                    ]
                    if len(near_atm) > 0 and len(far_atm) > 0:
                        # Find ATM in each
                        n_idx = (near_atm["moneyness"] - 1.0).abs().idxmin()
                        f_idx = (far_atm["moneyness"] - 1.0).abs().idxmin()
                        n_call = near_atm.loc[n_idx]
                        f_call = far_atm.loc[f_idx]
                        net_debit_cal = f_call["market_price"] - n_call["market_price"]
                        if net_debit_cal > 0:
                            trades.append({
                                "product": product,
                                "name": PRODUCT_NAMES.get(product, product),
                                "strategy": "CALL CALENDAR SPREAD",
                                "dte": f"{near_dte}/{far_dte}",
                                "expiry": f"{n_call['expiry_date']}/{f_call['expiry_date']}",
                                "legs": (
                                    f"Sell {n_call['strike']:.0f}C (DTE={near_dte}) @ {n_call['market_price']:.2f}, "
                                    f"Buy {f_call['strike']:.0f}C (DTE={far_dte}) @ {f_call['market_price']:.2f}"
                                ),
                                "net_premium": -net_debit_cal,
                                "max_loss": net_debit_cal,
                                "risk_reward": None,
                                "delta": f_call["delta"] - n_call["delta"],
                                "theta": f_call["theta"] - n_call["theta"],
                                "vega": f_call["vega"] - n_call["vega"],
                                "iv": iv,
                                "underlying": underlying,
                                "rationale": (
                                    f"Sell front-month ATM call, buy back-month. "
                                    f"Profits from near-term theta decay > far-term. "
                                    f"Low TG ({dn['theta_gamma_ratio']:.1f}) supports holding."
                                ),
                            })

    # --- Sort and display ---
    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        print("\nNo trade recommendations generated.")
        return trades_df

    sell_trades = trades_df[trades_df["net_premium"] > 0].sort_values("risk_reward", ascending=False)
    buy_trades = trades_df[trades_df["net_premium"] < 0].sort_values("net_premium", ascending=False)

    print("\n" + "-" * 70)
    print("INCOME TRADES (Option Selling / Credit Spreads)")
    print("-" * 70)
    for i, (_, t) in enumerate(sell_trades.head(20).iterrows(), 1):
        rr_str = f"{t['risk_reward']:.3f}" if pd.notna(t["risk_reward"]) else "N/A"
        delta_str = f"{t['delta']:.4f}" if pd.notna(t["delta"]) else "N/A"
        theta_str = f"{t['theta']:.4f}" if pd.notna(t["theta"]) else "N/A"
        vega_str = f"{t['vega']:.4f}" if pd.notna(t["vega"]) else "N/A"
        print(f"\n  Trade #{i}: {t['product']} ({t['name']}) - {t['strategy']}")
        print(f"    Expiry: {t['expiry']} (DTE={t['dte']})")
        print(f"    Legs: {t['legs']}")
        print(f"    Net Credit: {t['net_premium']:.2f}  |  Max Loss: {t['max_loss']:.2f}  |  R/R: {rr_str}")
        print(f"    Net Delta: {delta_str}, Net Theta: {theta_str}, Net Vega: {vega_str}")
        print(f"    Rationale: {t['rationale']}")

    print("\n" + "-" * 70)
    print("DIRECTIONAL / VOLATILITY TRADES (Option Buying)")
    print("-" * 70)
    for i, (_, t) in enumerate(buy_trades.head(15).iterrows(), 1):
        rr_str = f"{t['risk_reward']:.3f}" if pd.notna(t["risk_reward"]) else "unlimited"
        delta_str = f"{t['delta']:.4f}" if pd.notna(t["delta"]) else "N/A"
        theta_str = f"{t['theta']:.4f}" if pd.notna(t["theta"]) else "N/A"
        vega_str = f"{t['vega']:.4f}" if pd.notna(t["vega"]) else "N/A"
        print(f"\n  Trade #{i}: {t['product']} ({t['name']}) - {t['strategy']}")
        print(f"    Expiry: {t['expiry']} (DTE={t['dte']})")
        print(f"    Legs: {t['legs']}")
        print(f"    Cost: {abs(t['net_premium']):.2f}  |  Max Loss: {t['max_loss']:.2f}  |  R/R: {rr_str}")
        print(f"    Net Delta: {delta_str}, Net Theta: {theta_str}, Net Vega: {vega_str}")
        print(f"    Rationale: {t['rationale']}")

    return trades_df


# ===========================================================================
# SECTION 6: Summary Table
# ===========================================================================
def generate_summary_table(delta_neutral_results, risk_df, rec_df):
    """Compact summary table for dashboard consumption."""
    print("\n" + "=" * 80)
    print("SECTION 6: SUMMARY TABLE FOR DASHBOARD")
    print("=" * 80)

    if delta_neutral_results.empty or risk_df.empty:
        print("Insufficient data for summary.")
        return None

    merged = delta_neutral_results.merge(risk_df, on="product", how="inner", suffixes=("", "_risk"))
    if not rec_df.empty:
        merged = merged.merge(rec_df[["product", "specific_strategy"]], on="product", how="left")

    cols = [
        "product", "name", "strategy", "specific_strategy",
        "underlying", "implied_vol", "theta_gamma_ratio",
        "straddle_price", "theta_decay_pct", "vega_pct",
        "max_gamma_strike", "iv_range", "bl_skew",
    ]
    # Only keep columns that exist
    cols = [c for c in cols if c in merged.columns]
    summary = merged[cols].sort_values("theta_gamma_ratio")

    csv_path = os.path.join(REPORT_DIR, "greeks_strategy_summary.csv")
    summary.to_csv(csv_path, index=False)
    print(f"\nSummary saved to: {csv_path}")
    print(summary.to_string(index=False))

    return summary


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    print("=" * 80)
    print("GREEKS-BASED OPTIONS STRATEGY FRAMEWORK")
    print("Chinese Commodity Futures Analysis")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 80)

    df = load_data()
    total = len(df)
    valid = df["has_real_greeks"].sum()
    calls_valid = ((df["option_type"] == "CALL") & df["has_real_greeks"]).sum()
    puts_synthesized = ((df["option_type"] == "PUT") & df["has_real_greeks"]).sum()
    print(f"\nLoaded {total} option records across {df['product'].nunique()} products")
    print(f"  CALLs with valid Greeks: {calls_valid}")
    print(f"  PUTs with synthesized Greeks (from calls): {puts_synthesized}")
    print(f"  Total records with Greeks: {valid}")

    dn_results = analyze_delta_neutral(df)
    surface_analysis, top_products = analyze_greeks_surface(df, top_n=10)
    rec_df = recommend_strategy_per_product(df, dn_results)
    risk_df = calculate_risk_metrics(df)
    trades_df = generate_trade_recommendations(df, dn_results, risk_df)
    summary = generate_summary_table(dn_results, risk_df, rec_df)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nProducts analyzed: {df['product'].nunique()}")
    print(f"Valid Greeks records: {valid}")
    print(f"Trade recommendations: {len(trades_df)}")
    print(f"\nOutput files:")
    print(f"  {os.path.join(REPORT_DIR, 'greeks_strategy_summary.csv')}")
    if not trades_df.empty:
        trade_path = os.path.join(REPORT_DIR, "greeks_trade_recommendations.csv")
        trades_df.to_csv(trade_path, index=False)
        print(f"  {trade_path}")

    return {
        "delta_neutral": dn_results,
        "surface_analysis": surface_analysis,
        "recommendations": rec_df,
        "risk_metrics": risk_df,
        "trades": trades_df,
        "summary": summary,
    }


if __name__ == "__main__":
    results = main()
