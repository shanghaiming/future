"""V104: Crisis Filter + Cross-Market Correlation Regime Detection.
On top of V96 (NW+BMA+Vol-Adaptive). During crises correlations spike
to 1.0 -- factor signals unreliable, high drawdown risk.
Layer 1: cross-sectional corr crisis detector -> reduce size.
Layer 2: factor IC instability detector -> reduce size.
Both compound. Walk-forward 2019-2026. No leverage.
"""
import sys
import os
import time
import warnings
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from alpha_futures_data import load_all_data
from alpha_futures_v96 import (
    compute_raw_factors,
    compute_rolling_ic,
    compute_bma_weights,
    compute_nw_predicted_returns_with_bma,
    compute_ker,
    compute_portfolio_volatility,
    build_sector_lookup,
    get_dynamic_mode,
    get_mode_threshold,
    compute_atr_at,
    FACTOR_NAMES,
    N_FACTORS,
    SECTOR_MAP,
)

CASH0 = 1_000_000
COMM = 0.0005
LEVERAGE = 1.0


# =====================================================================
# INNOVATION: Two-Layer Crisis Detection
# =====================================================================

def compute_market_crisis(
    C: np.ndarray, NS: int, ND: int, corr_window: int = 20,
) -> np.ndarray:
    """Layer 1: Cross-sectional correlation crisis detector.

    Compute rolling pairwise return correlation across all commodities.
    When median correlation is high, the market is in "crisis mode" --
    all commodities move together, factor signals unreliable.

    Returns (ND,) array of median pairwise correlation values.
    """
    t0 = time.time()
    print(f"[V104] Computing market crisis (corr_window={corr_window})...",
          flush=True)

    # Precompute daily returns for all instruments
    daily_rets = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(1, ND):
            if (not np.isnan(C[si, di])
                    and not np.isnan(C[si, di - 1])
                    and C[si, di - 1] > 0):
                daily_rets[si, di] = C[si, di] / C[si, di - 1] - 1.0

    median_corr = np.full(ND, np.nan)

    for di in range(corr_window + 1, ND):
        # Collect returns matrix for the window: (NS, corr_window)
        ret_window = daily_rets[:, di - corr_window:di]
        # Find instruments with enough data
        valid_instruments = []
        for si in range(NS):
            valid_count = np.sum(~np.isnan(ret_window[si]))
            if valid_count >= corr_window * 0.7:
                valid_instruments.append(si)

        if len(valid_instruments) < 5:
            continue

        # Compute pairwise correlations (sample for speed if > 30)
        if len(valid_instruments) > 30:
            rng = np.random.RandomState(42)
            sampled = rng.choice(
                valid_instruments, size=30, replace=False)
        else:
            sampled = np.array(valid_instruments)

        pair_corrs = []
        n_sampled = len(sampled)
        for i in range(n_sampled):
            for j in range(i + 1, n_sampled):
                ri = ret_window[sampled[i]]
                rj = ret_window[sampled[j]]
                valid_mask = (~np.isnan(ri)) & (~np.isnan(rj))
                n_valid = np.sum(valid_mask)
                if n_valid < corr_window * 0.5:
                    continue
                ri_v = ri[valid_mask]
                rj_v = rj[valid_mask]
                if np.std(ri_v) < 1e-12 or np.std(rj_v) < 1e-12:
                    continue
                corr = np.corrcoef(ri_v, rj_v)[0, 1]
                if not np.isnan(corr):
                    pair_corrs.append(corr)

        if len(pair_corrs) >= 10:
            median_corr[di] = np.median(pair_corrs)

    n_crisis = np.sum(
        median_corr[~np.isnan(median_corr)] > 0.4)
    print(
        f"  Market crisis done: {time.time() - t0:.1f}s, "
        f"crisis_days(>0.4)={n_crisis}", flush=True)
    return median_corr


def compute_factor_instability(
    raw_factors: Dict[str, np.ndarray],
    NS: int, ND: int, ic_window: int = 20, min_pairs: int = 10,
) -> np.ndarray:
    """Layer 2: Factor instability detector.

    Track rolling IC standard deviation for each factor.
    When max IC_std is high, factors are unreliable -- signals noisy.

    Returns (ND,) array of max IC standard deviation across factors.
    """
    t0 = time.time()
    print(f"[V104] Computing factor instability (ic_window={ic_window})...",
          flush=True)

    fwd_ret = raw_factors["fwd_ret_5d"]
    ic_std_max = np.full(ND, np.nan)

    for di in range(ic_window + 10, ND):
        factor_ic_stds = []
        for fname in FACTOR_NAMES:
            factor = raw_factors[fname]
            ic_vals = []
            for tdi in range(di - ic_window, di):
                f_day = factor[:, tdi]
                r_day = fwd_ret[:, tdi]
                valid_mask = (~np.isnan(f_day)) & (~np.isnan(r_day))
                f_valid = f_day[valid_mask]
                r_valid = r_day[valid_mask]
                if len(f_valid) >= min_pairs:
                    f_rank = pd.Series(f_valid).rank().values
                    r_rank = pd.Series(r_valid).rank().values
                    corr = np.corrcoef(f_rank, r_rank)[0, 1]
                    if not np.isnan(corr):
                        ic_vals.append(corr)

            if len(ic_vals) >= 5:
                factor_ic_stds.append(np.std(ic_vals))

        if factor_ic_stds:
            ic_std_max[di] = np.max(factor_ic_stds)

    valid_vals = ic_std_max[~np.isnan(ic_std_max)]
    if len(valid_vals) > 0:
        p75 = np.percentile(valid_vals, 75)
        n_unstable = np.sum(valid_vals > p75)
        print(
            f"  Factor instability done: {time.time() - t0:.1f}s, "
            f"p75={p75:.3f} unstable_days={n_unstable}", flush=True)
    return ic_std_max


def get_crisis_multiplier(
    median_corr: float, ic_std_max: float,
    crisis_threshold: float, instability_threshold: float,
    crisis_size_mult: float, instab_size_mult: float,
) -> float:
    """Compute combined crisis multiplier from both layers.

    Both layers compound: crisis AND unstable -> size * mult1 * mult2.
    """
    mult = 1.0
    if not np.isnan(median_corr) and median_corr > crisis_threshold:
        mult *= crisis_size_mult
    if not np.isnan(ic_std_max) and ic_std_max > instability_threshold:
        mult *= instab_size_mult
    return mult


# =====================================================================
# Backtest with crisis filter
# =====================================================================

def backtest_v104(
    C: np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
    NS: int, ND: int, dates: np.ndarray, syms: List[str],
    predicted: np.ndarray, ker_regime: np.ndarray,
    port_vol: np.ndarray, median_corr: np.ndarray,
    ic_std_max: np.ndarray, sector_lookup: Dict[int, str],
    top_n: int = 2, max_per_sector: int = 2, hold_days: int = 5,
    win_threshold: float = 0.60, normal_threshold: float = 0.80,
    lose_threshold: float = 0.90, win_rate_window: int = 15,
    atr_stop: float = 3.0, vol_lookback: int = 20,
    vol_high_mult: float = 2.0, vol_low_mult: float = 0.5,
    size_reduce: float = 0.5, size_boost: float = 1.3,
    crisis_threshold: float = 0.4, instability_threshold: float = 0.3,
    crisis_size_mult: float = 0.3, instab_size_mult: float = 0.5,
    start_di: int = 60, end_di: Optional[int] = None,
) -> Tuple[List[dict], float, float]:
    """Backtest V104: V96 NW+BMA+Vol + crisis filter."""
    if end_di is None:
        end_di = ND - 1

    vol_data = port_vol[max(start_di, vol_lookback + 1):end_di]
    vol_data_valid = vol_data[~np.isnan(vol_data)]
    vol_median = (
        np.median(vol_data_valid) if len(vol_data_valid) > 10 else 1e-6)

    equity = CASH0
    peak = equity
    max_dd = 0.0
    positions: List[Tuple[int, int, float, float, float]] = []
    trades: List[dict] = []
    recent_trades_win: List[int] = []
    crisis_days = 0
    unstable_days = 0

    for di in range(max(start_di, 1), end_di):
        d = dates[di]
        daily_pnl = 0.0
        new_positions: List[Tuple[int, int, float, float, float]] = []

        mode = get_dynamic_mode(
            recent_trades_win, win_threshold, win_rate_window)
        current_threshold = get_mode_threshold(
            mode, win_threshold, normal_threshold, lose_threshold)

        # V96 vol-adaptive sizing
        vol_mult = 1.0
        pv = port_vol[di]
        if not np.isnan(pv) and not np.isnan(vol_median) and vol_median > 1e-12:
            ratio = pv / vol_median
            if ratio > vol_high_mult:
                vol_mult = size_reduce
            elif ratio < vol_low_mult:
                vol_mult = size_boost

        # V104 crisis filter
        crisis_mult = get_crisis_multiplier(
            median_corr[di], ic_std_max[di],
            crisis_threshold, instability_threshold,
            crisis_size_mult, instab_size_mult)

        if (not np.isnan(median_corr[di])
                and median_corr[di] > crisis_threshold):
            crisis_days += 1
        if (not np.isnan(ic_std_max[di])
                and ic_std_max[di] > instability_threshold):
            unstable_days += 1

        # Combined size multiplier: vol-adaptive * crisis filter
        combined_mult = vol_mult * crisis_mult

        # Exit logic
        pos_by_si: Dict[int, List] = defaultdict(list)
        for si, edi, ep, sp, alloc in positions:
            pos_by_si[si].append((edi, ep, sp, alloc))

        for si, pos_list in pos_by_si.items():
            c = C[si, di]
            if np.isnan(c):
                for edi, ep, sp, alloc in pos_list:
                    new_positions.append((si, edi, ep, sp, alloc))
                continue

            earliest_edi = min(p[0] for p in pos_list)
            hold = di - earliest_edi
            stopped = any(c < sp for _, _, sp, _ in pos_list)

            if stopped:
                for edi, ep, sp, alloc in pos_list:
                    pnl = (c - ep) / ep - COMM
                    profit = equity * alloc * pnl
                    daily_pnl += profit
                    is_win = pnl > 0
                    trades.append({
                        "pnl_abs": profit, "pnl_pct": pnl * 100,
                        "days": di - edi + 1, "di": di,
                        "year": d.year, "sym": syms[si],
                        "sector": sector_lookup.get(si, 'OTHER'),
                        "reason": "stop", "mode": mode[:1].upper(),
                    })
                    recent_trades_win.append(1 if is_win else 0)
            elif hold >= hold_days:
                for edi, ep, sp, alloc in pos_list:
                    pnl = (c - ep) / ep - COMM
                    profit = equity * alloc * pnl
                    daily_pnl += profit
                    is_win = pnl > 0
                    trades.append({
                        "pnl_abs": profit, "pnl_pct": pnl * 100,
                        "days": di - edi + 1, "di": di,
                        "year": d.year, "sym": syms[si],
                        "sector": sector_lookup.get(si, 'OTHER'),
                        "reason": "hold", "mode": mode[:1].upper(),
                    })
                    recent_trades_win.append(1 if is_win else 0)
            else:
                for edi, ep, sp, alloc in pos_list:
                    new_positions.append((si, edi, ep, sp, alloc))

        positions = new_positions
        equity += daily_pnl
        if equity > peak:
            peak = equity
        if peak > 0:
            dd = (peak - equity) / peak * 100
            if dd > max_dd:
                max_dd = dd
        if equity <= 0:
            break

        # Entry: select top_n by predicted return
        held = {p[0] for p in positions}
        if len(held) >= top_n:
            continue

        candidates = []
        for si in range(NS):
            if si in held:
                continue
            pred = predicted[si, di]
            if np.isnan(pred):
                continue
            if di + 1 >= ND or np.isnan(O[si, di + 1]):
                continue
            if ker_regime[si, di] < 0:
                continue
            candidates.append((pred, si))

        if not candidates:
            continue

        candidates.sort(key=lambda x: -x[0])

        n_to_take = top_n
        if mode == "winning":
            n_to_take = min(top_n + 1, top_n * 2)
        elif mode == "losing":
            n_to_take = max(1, top_n - 1)

        sector_counts: Dict[str, int] = defaultdict(int)
        for si_held in held:
            sector_counts[sector_lookup.get(si_held, 'OTHER')] += 1

        new_entries = []
        for pred_val, si in candidates:
            if len(held) + len(new_entries) >= n_to_take:
                break
            if si in held:
                continue
            sym_sector = sector_lookup.get(si, 'OTHER')
            if sector_counts[sym_sector] >= max_per_sector:
                continue
            if pred_val <= 0:
                continue
            new_entries.append((pred_val, si, sym_sector))
            sector_counts[sym_sector] += 1

        if not new_entries:
            continue

        num_total = len(positions) + len(new_entries)
        # Apply combined multiplier: vol-adaptive * crisis filter
        alloc_per_pos = LEVERAGE / num_total * combined_mult

        updated_positions = []
        for si, edi, ep, sp, old_alloc in positions:
            updated_positions.append((si, edi, ep, sp, alloc_per_pos))

        for pred_val, si, sym_sector in new_entries:
            ep = O[si, di + 1]
            if np.isnan(ep) or ep <= 0:
                continue
            atr = compute_atr_at(H, L, C, si, di, start_di)
            if atr is None:
                continue
            updated_positions.append(
                (si, di + 1, ep, ep - atr_stop * atr, alloc_per_pos))

        positions = updated_positions

    # Close remaining positions
    for si, edi, ep, sp, alloc in positions:
        c = C[si, ND - 1]
        if not np.isnan(c) and c > 0:
            pnl = (c - ep) / ep - COMM
            equity += equity * alloc * pnl

    if trades:
        trades[0]["crisis_info"] = (
            f"crisis_days={crisis_days} unstable_days={unstable_days} "
            f"of {end_di - start_di} total")
    return trades, equity, max_dd


def analyze(
    trades: List[dict], equity: float, max_dd: float, label: str = "",
) -> Optional[dict]:
    if not trades:
        print(f"  {label}: no trades")
        return None

    nw = sum(1 for t in trades if t["pnl_pct"] > 0)
    wr = nw / len(trades) * 100
    n_days = max(1, trades[-1]["di"] - trades[0]["di"])
    ann = ((equity / CASH0) ** (1 / max(1.0, n_days / 252)) - 1) * 100
    ap = [t["pnl_abs"] for t in sorted(trades, key=lambda x: x["di"])]
    rets = np.array(ap) / CASH0
    sh = (np.mean(rets) / np.std(rets) * np.sqrt(252)
          if np.std(rets) > 0 else 0)

    n_stop = sum(1 for t in trades if t["reason"] == "stop")
    n_hold = sum(1 for t in trades if t["reason"] == "hold")
    sector_counts: Dict[str, int] = defaultdict(int)
    for t in trades:
        sector_counts[t.get("sector", "OTHER")] += 1
    sector_str = " ".join(
        f"{k}:{v}" for k, v in sorted(sector_counts.items()))
    avg_pnl = np.mean([t["pnl_pct"] for t in trades])
    avg_win = np.mean(
        [t["pnl_pct"] for t in trades if t["pnl_pct"] > 0])
    avg_loss = np.mean(
        [t["pnl_pct"] for t in trades if t["pnl_pct"] <= 0])

    print(
        f"  {label}: {len(trades)}t (stop:{n_stop} hold:{n_hold}) "
        f"WR={wr:.1f}% ann={ann:+.1f}% DD={max_dd:.1f}% "
        f"Sh={sh:.2f} eq={equity:,.0f}")
    print(
        f"    avg_pnl={avg_pnl:+.3f}% avg_win={avg_win:+.3f}% "
        f"avg_loss={avg_loss:+.3f}%")
    print(f"    sectors: {sector_str}")


    return {
        "n": len(trades), "wr": wr, "dd": max_dd,
        "ann": ann, "sh": sh, "eq": equity,
    }


def walk_forward(
    C: np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
    NS: int, ND: int, dates: np.ndarray, syms: List[str],
    predicted: np.ndarray, ker_regime: np.ndarray,
    port_vol: np.ndarray, median_corr: np.ndarray,
    ic_std_max: np.ndarray, sector_lookup: Dict[int, str],
    top_n: int = 2, max_per_sector: int = 2, hold_days: int = 5,
    vol_high_mult: float = 2.0, vol_low_mult: float = 0.5,
    size_reduce: float = 0.5, size_boost: float = 1.3,
    vol_lookback: int = 20,
    crisis_threshold: float = 0.4, instability_threshold: float = 0.3,
    crisis_size_mult: float = 0.3, instab_size_mult: float = 0.5,
    label: str = "",
) -> List[dict]:
    cfg_str = (
        f"tn={top_n} mps={max_per_sector} vhm={vol_high_mult:.1f} "
        f"vlm={vol_low_mult:.1f} sr={size_reduce:.1f} sb={size_boost:.1f} "
        f"ct={crisis_threshold:.1f} it={instability_threshold:.1f} "
        f"csm={crisis_size_mult:.1f} ism={instab_size_mult:.1f}")
    print(f"\n{'=' * 70}")
    print(f"  WALK-FORWARD V104 {label}")
    print(f"  {cfg_str}")
    print(f"{'=' * 70}")

    years = sorted(set(d.year for d in dates))
    all_trades: List[dict] = []

    for test_year in range(2019, years[-1] + 1):
        test_start = None
        test_end_idx = None
        for i, d in enumerate(dates):
            if d.year == test_year and test_start is None:
                test_start = i
            if d.year == test_year:
                test_end_idx = i
        if test_start is None:
            continue

        trades, _, _ = backtest_v104(
            C, O, H, L, NS, ND, dates, syms,
            predicted, ker_regime, port_vol,
            median_corr, ic_std_max,
            sector_lookup=sector_lookup,
            top_n=top_n, max_per_sector=max_per_sector,
            hold_days=hold_days,
            vol_high_mult=vol_high_mult,
            vol_low_mult=vol_low_mult,
            size_reduce=size_reduce, size_boost=size_boost,
            vol_lookback=vol_lookback,
            crisis_threshold=crisis_threshold,
            instability_threshold=instability_threshold,
            crisis_size_mult=crisis_size_mult,
            instab_size_mult=instab_size_mult,
            start_di=test_start, end_di=test_end_idx + 1,
        )

        test_trades = [t for t in trades
                       if dates[t["di"]].year == test_year]
        all_trades.extend(test_trades)

        if test_trades:
            n = len(test_trades)
            nw = sum(1 for t in test_trades if t["pnl_pct"] > 0)
            wr_val = nw / n * 100
            avg = np.mean([t["pnl_pct"] for t in test_trades])
            yr_sectors: Dict[str, int] = defaultdict(int)
            for t in test_trades:
                yr_sectors[t.get("sector", "OTHER")] += 1
            sec_str = " ".join(
                f"{k}:{v}" for k, v in sorted(yr_sectors.items()))
            print(
                f"  {test_year}: {n}t WR={wr_val:.1f}% avg={avg:+.2f}% "
                f"sectors=[{sec_str}]", flush=True)
        else:
            print(f"  {test_year}: no trades", flush=True)

    if all_trades:
        nw = sum(1 for t in all_trades if t["pnl_pct"] > 0)
        wr_val = nw / len(all_trades) * 100
        avg = np.mean([t["pnl_pct"] for t in all_trades])
        cum = np.prod(
            [1 + t["pnl_pct"] / 100 for t in all_trades]) - 1
        agg_sectors: Dict[str, int] = defaultdict(int)
        for t in all_trades:
            agg_sectors[t.get("sector", "OTHER")] += 1
        sec_str = " ".join(
            f"{k}:{v}" for k, v in sorted(agg_sectors.items()))
        print(
            f"\n  WF TOTAL: {len(all_trades)}t "
            f"WR={wr_val:.1f}% avg={avg:+.2f}% cum={cum:+.1%}")
        print(f"  WF SECTORS: {sec_str}")
        return all_trades
    return []


def main() -> None:
    t0 = time.time()
    print("=" * 70)
    print('  V104: "危邦不入" Crisis Filter + Cross-Market Correlation')
    print("  Layer 1: Cross-sectional correlation crisis detector")
    print("  Layer 2: Factor instability detector")
    print("  Base: V96 NW+BMA+Vol-Adaptive framework")
    print("  Walk-forward 2019-2026. No leverage.")
    print("=" * 70)

    C, O, H, L, V, OI, NS, ND, dates, syms = load_all_data(
        start="2016-01-01")
    print(
        f"  {NS} sym, {ND} days, "
        f"{dates[0].strftime('%Y-%m-%d')} to "
        f"{dates[-1].strftime('%Y-%m-%d')}")

    sector_lookup = build_sector_lookup(syms)
    sector_dist: Dict[str, int] = defaultdict(int)
    for sec in sector_lookup.values():
        sector_dist[sec] += 1
    print(f"  Sector distribution: {dict(sector_dist)}")

    bt_2019 = None
    for i, d in enumerate(dates):
        if d >= pd.Timestamp("2019-01-01"):
            bt_2019 = i
            break

    # === 1. Compute V96 signals (reused from V96) ===
    raw_factors = compute_raw_factors(C, O, H, L, V, OI, NS, ND)
    ker_regime = compute_ker(C, NS, ND)

    # Use best V96 config: ic_window=60, prior_strength=5
    ic_array = compute_rolling_ic(raw_factors, NS, ND, ic_window=60)
    bma_weights = compute_bma_weights(ic_array, ND, prior_strength=5.0)

    print("\n--- Computing NW+BMA predictions ---")
    predicted = compute_nw_predicted_returns_with_bma(
        raw_factors, bma_weights, NS, ND,
        training_window=40, kernel_bandwidth=1.0,
    )

    # === 2. Compute V96 vol-adaptive baseline ===
    vol_cache: Dict[int, np.ndarray] = {}
    for vlb in [15]:
        vol_cache[vlb] = compute_portfolio_volatility(C, NS, ND, vlb)

    # === 3. V104 Innovation: Crisis detection layers ===
    crisis_cache: Dict[int, np.ndarray] = {}
    for cw in [15, 20, 30]:
        crisis_cache[cw] = compute_market_crisis(C, NS, ND, cw)

    instab_cache: Dict[int, np.ndarray] = {}
    for iw in [15, 20, 30]:
        instab_cache[iw] = compute_factor_instability(
            raw_factors, NS, ND, ic_window=iw)

    # === 4. Parameter sweep ===
    print("\n" + "=" * 70)
    print("  PARAMETER SWEEP V104 (2019-2026)")
    print("  NO LEVERAGE. NW+BMA+Vol+Crisis filter.")
    print("=" * 70)

    results: List[dict] = []
    sweep_count = 0

    for corr_w in [15, 20, 30]:
        mc = crisis_cache[corr_w]
        for ic_w in [15, 20, 30]:
            icm = instab_cache[ic_w]
            for crisis_threshold in [0.3, 0.4, 0.5]:
                for instability_threshold in [0.25, 0.30, 0.35]:
                    for crisis_size_mult in [0.2, 0.3, 0.5]:
                        for instab_size_mult in [0.3, 0.5, 0.7]:
                            for mps in [2, 3]:
                                sweep_count += 1
                                trades, eq, dd = backtest_v104(
                                    C, O, H, L, NS, ND,
                                    dates, syms,
                                    predicted, ker_regime,
                                    vol_cache[15],
                                    mc, icm,
                                    sector_lookup=sector_lookup,
                                    top_n=2,
                                    max_per_sector=mps,
                                    hold_days=5,
                                    vol_high_mult=2.0,
                                    vol_low_mult=0.5,
                                    size_reduce=0.5,
                                    size_boost=1.3,
                                    vol_lookback=15,
                                    crisis_threshold=crisis_threshold,
                                    instability_threshold=instability_threshold,
                                    crisis_size_mult=crisis_size_mult,
                                    instab_size_mult=instab_size_mult,
                                    start_di=bt_2019,
                                )

                                if len(trades) < 10:
                                    continue

                                nw = sum(
                                    1 for t in trades
                                    if t["pnl_pct"] > 0)
                                wr = nw / len(trades) * 100
                                n_days = max(
                                    1,
                                    trades[-1]["di"] - trades[0]["di"])
                                ann = ((eq / CASH0) ** (
                                    1 / max(
                                        1.0, n_days / 252)) - 1) * 100
                                ap = [t["pnl_abs"]
                                      for t in sorted(
                                          trades,
                                          key=lambda x: x["di"])]
                                rets_arr = np.array(ap) / CASH0
                                sh_val = (
                                    np.mean(rets_arr)
                                    / np.std(rets_arr) * np.sqrt(252)
                                    if np.std(rets_arr) > 0 else 0)

                                results.append({
                                    "cw": corr_w, "icw": ic_w,
                                    "ct": crisis_threshold,
                                    "it": instability_threshold,
                                    "csm": crisis_size_mult,
                                    "ism": instab_size_mult,
                                    "mps": mps,
                                    "n": len(trades), "wr": wr,
                                    "ann": ann, "dd": dd,
                                    "sharpe": sh_val, "eq": eq,
                                })

    # === V96 baseline for comparison ===
    trades_v96, eq_v96, dd_v96 = backtest_v104(
        C, O, H, L, NS, ND, dates, syms,
        predicted, ker_regime, vol_cache[15],
        crisis_cache[20], instab_cache[20],
        sector_lookup=sector_lookup,
        top_n=2, max_per_sector=2, hold_days=5,
        vol_high_mult=2.0, vol_low_mult=0.5,
        size_reduce=0.5, size_boost=1.3,
        vol_lookback=15,
        crisis_threshold=99.0,  # effectively disable crisis filter
        instability_threshold=99.0,
        crisis_size_mult=1.0,
        instab_size_mult=1.0,
        start_di=bt_2019,
    )

    results.sort(key=lambda x: -x["ann"])
    print(
        f"\n  Evaluated {sweep_count} configs, "
        f"{len(results)} with 10+ trades")

    print(
        f"\n{'CW':>3} {'IW':>3} {'CT':>4} {'IT':>4} "
        f"{'CSM':>4} {'ISM':>4} {'MPS':>3} "
        f"{'N':>5} {'WR':>6} {'Ann':>9} {'DD':>7} {'Sh':>7}")
    print("-" * 85)
    for r in results[:15]:
        print(
            f"{r['cw']:>3} {r['icw']:>3} {r['ct']:>4.1f} "
            f"{r['it']:>4.2f} "
            f"{r['csm']:>4.1f} {r['ism']:>4.1f} {r['mps']:>3} "
            f"{r['n']:>5} {r['wr']:>5.1f}% {r['ann']:>+8.1f}% "
            f"{r['dd']:>6.1f}% {r['sharpe']:>6.2f}")

    if not results:
        print("  No configs with 10+ trades. Exiting.")
        return

    # === 5. Walk-forward for best configs ===
    best_by_sharpe = max(results, key=lambda x: x["sharpe"])
    best_by_ann = results[0]
    best_risk_adj = max(
        results, key=lambda x: x["ann"] / max(x["dd"], 1.0))

    for label, best in [
        ("BEST-ANN", best_by_ann),
        ("BEST-SHARPE", best_by_sharpe),
        ("BEST-RISK-ADJ", best_risk_adj),
    ]:
        walk_forward(
            C, O, H, L, NS, ND, dates, syms,
            predicted, ker_regime,
            vol_cache[15],
            crisis_cache[best["cw"]],
            instab_cache[best["icw"]],
            sector_lookup=sector_lookup,
            top_n=2, max_per_sector=best["mps"],
            hold_days=5,
            vol_high_mult=2.0, vol_low_mult=0.5,
            size_reduce=0.5, size_boost=1.3,
            vol_lookback=15,
            crisis_threshold=best["ct"],
            instability_threshold=best["it"],
            crisis_size_mult=best["csm"],
            instab_size_mult=best["ism"],
            label=label,
        )

    # === 6. Compare V104 vs V96 baseline ===
    print("\n" + "=" * 70)
    print('  COMPARISON: V104 (Crisis Filter) vs V96 (No Crisis Filter)')
    print("  (2019-2026 OOS)")
    print("=" * 70)

    # V104 best
    best = best_by_ann
    trades_v104, eq_v104, dd_v104 = backtest_v104(
        C, O, H, L, NS, ND, dates, syms,
        predicted, ker_regime, vol_cache[15],
        crisis_cache[best["cw"]],
        instab_cache[best["icw"]],
        sector_lookup=sector_lookup,
        top_n=2, max_per_sector=best["mps"],
        hold_days=5,
        vol_high_mult=2.0, vol_low_mult=0.5,
        size_reduce=0.5, size_boost=1.3,
        vol_lookback=15,
        crisis_threshold=best["ct"],
        instability_threshold=best["it"],
        crisis_size_mult=best["csm"],
        instab_size_mult=best["ism"],
        start_di=bt_2019,
    )

    print(f"\n  V104 BEST-ANN (NW+BMA+Vol+Crisis):")
    analyze(trades_v104, eq_v104, dd_v104, "V104-Crisis")
    print(f"\n  V96 BASELINE (NW+BMA+Vol, no crisis filter):")
    analyze(trades_v96, eq_v96, dd_v96, "V96-baseline")

    if trades_v104 and trades_v96:
        print(
            f"\n  Delta: eq={eq_v104 - eq_v96:+,.0f} "
            f"dd={dd_v104 - dd_v96:+.1f}% "
            f"trades={len(trades_v104) - len(trades_v96):+d}")

    print(f"\n[V104] Done. {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
