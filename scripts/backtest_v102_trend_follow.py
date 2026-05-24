#!/usr/bin/env python3
"""
Trend Following Strategy Backtest v102
=======================================
Tests 5 trend-following variants with multiple filter combinations:
  1. Dual Moving Average (SMA crossover)
  2. Channel Breakout (Donchian)
  3. ADX Filter (+DI/-DI with ADX threshold)
  4. Trend + OI Confirmation
  5. Trend + Term Structure (backwardation/contango filter)

Walk-forward: Train 2021-2023, Validate 2024, Test 2025-2026.

Position sizing: equal-weight across top-N ranked signals per side.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict
from itertools import product

# ──────────────────────────── PATHS ────────────────────────────
DATA_DIR = os.path.expanduser('~/home/futures_platform/data/futures_weighted/')
TS_DIR = os.path.expanduser('~/home/futures_platform/data/futures_term_structure/')

# ──────────────────────────── PARAMETERS ───────────────────────
INITIAL_CAPITAL = 5_000_000
COMMISSION_BPS = 0.5          # 0.05% round-trip commission
SLIPPAGE_BPS = 1.0            # 0.1% slippage per side

WALK_FORWARD = {
    'train': ('2021-01-01', '2023-12-31'),
    'validate': ('2024-01-01', '2024-12-31'),
    'test': ('2025-01-01', '2026-05-21'),
}

SMA_PAIRS = [(10, 30), (20, 60), (50, 120)]
DONCHIAN_PERIODS = [10, 20, 40]
ADX_THRESHOLD = 25
ADX_PERIOD = 14

POSITION_COUNTS = [5, 10, 15]
HOLDING_PERIODS = ['signal', 10, 20]   # signal = hold until reversal
SL_TP_PAIRS = [(-0.02, 0.05), (-0.03, 0.08), (None, None)]  # None = no SL/TP


# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════
def load_futures_data(min_days=500):
    """Load all commodity daily CSVs into a dict of DataFrames."""
    data = {}
    for f in sorted(os.listdir(DATA_DIR)):
        if not f.endswith('.csv'):
            continue
        symbol = f.replace('.csv', '')
        df = pd.read_csv(os.path.join(DATA_DIR, f))
        df['trade_date'] = pd.to_datetime(df['trade_date'], format='mixed')
        df = df.sort_values('trade_date').reset_index(drop=True)
        # Filter: need at least min_days of data and data after 2020
        if len(df) < min_days:
            continue
        if df['trade_date'].max() < pd.Timestamp('2021-01-01'):
            continue
        if df['close'].isna().all():
            continue
        data[symbol] = df
    return data


def load_term_structure_index():
    """Build a DataFrame of term structure: symbol x date -> structure."""
    records = []
    files = [f for f in os.listdir(TS_DIR) if f.endswith('.json')]
    for i, f in enumerate(files):
        if i % 20000 == 0:
            print(f"  Loading term structure... {i}/{len(files)}", end='\r')
        try:
            with open(os.path.join(TS_DIR, f)) as fp:
                item = json.load(fp)
            records.append({
                'symbol': item['symbol'],
                'date': pd.to_datetime(item['date']),
                'structure': item.get('structure', 'unknown'),
            })
        except Exception:
            continue
    print(f"  Loaded {len(records)} term structure records.          ")
    if not records:
        return pd.DataFrame()
    ts_df = pd.DataFrame(records)
    ts_df = ts_df.drop_duplicates(subset=['symbol', 'date'], keep='last')
    return ts_df


# ═══════════════════════════════════════════════════════════════
# TECHNICAL INDICATORS
# ═══════════════════════════════════════════════════════════════
def compute_indicators(df):
    """Add all required technical indicators to a single commodity DataFrame."""
    df = df.copy()
    c = df['close']
    h = df['high']
    l = df['low']

    # Returns
    df['ret'] = c.pct_change()

    # SMAs for dual MA
    for w in [10, 20, 30, 50, 60, 120]:
        df[f'sma_{w}'] = c.rolling(w, min_periods=w).mean()

    # Donchian channels
    for p in [10, 20, 40]:
        df[f'don_high_{p}'] = h.rolling(p, min_periods=p).max().shift(1)
        df[f'don_low_{p}'] = l.rolling(p, min_periods=p).min().shift(1)

    # ATR (14-period)
    tr1 = h - l
    tr2 = (h - c.shift()).abs()
    tr3 = (l - c.shift()).abs()
    df['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr14'] = df['tr'].rolling(14, min_periods=14).mean()

    # ADX system
    plus_dm = (h - h.shift()).clip(lower=0)
    minus_dm = (l.shift() - l).clip(lower=0)
    # Zero out when the other DM is larger
    plus_dm = plus_dm.where(plus_dm > minus_dm, 0)
    minus_dm = minus_dm.where(minus_dm > plus_dm, 0)
    atr_s = df['tr'].rolling(ADX_PERIOD, min_periods=ADX_PERIOD).mean()
    plus_di = 100 * (plus_dm.rolling(ADX_PERIOD, min_periods=ADX_PERIOD).mean() / atr_s)
    minus_di = 100 * (minus_dm.rolling(ADX_PERIOD, min_periods=ADX_PERIOD).mean() / atr_s)
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    df['plus_di'] = plus_di
    df['minus_di'] = minus_di
    df['adx'] = dx.rolling(ADX_PERIOD, min_periods=ADX_PERIOD).mean()

    # OI change (5-day rate of change)
    df['oi_ma5'] = df['oi'].rolling(5, min_periods=5).mean()
    df['oi_ma20'] = df['oi'].rolling(20, min_periods=20).mean()
    df['oi_increasing'] = (df['oi_ma5'] > df['oi_ma20']).astype(int)

    return df


# ═══════════════════════════════════════════════════════════════
# SIGNAL GENERATORS
# ═══════════════════════════════════════════════════════════════
def signal_dual_ma(df, fast, slow):
    """SMA crossover: +1 long, -1 short, 0 flat."""
    s = pd.Series(0, index=df.index, dtype=int)
    s[df[f'sma_{fast}'] > df[f'sma_{slow}']] = 1
    s[df[f'sma_{fast}'] < df[f'sma_{slow}']] = -1
    return s


def signal_donchian(df, period):
    """Donchian breakout: long above high, short below low."""
    s = pd.Series(0, index=df.index, dtype=int)
    s[df['close'] > df[f'don_high_{period}']] = 1
    s[df['close'] < df[f'don_low_{period}']] = -1
    return s


def signal_adx(df):
    """ADX-filtered DI crossover."""
    s = pd.Series(0, index=df.index, dtype=int)
    trending = df['adx'] > ADX_THRESHOLD
    s[(df['plus_di'] > df['minus_di']) & trending] = 1
    s[(df['minus_di'] > df['plus_di']) & trending] = -1
    return s


def signal_trend_oi(df, base_signal):
    """Filter base trend signal: only when OI is increasing."""
    return base_signal.where(df['oi_increasing'] == 1, 0)


def build_ts_series_for_symbol(symbol, dates, ts_lookup):
    """Build a pandas Series of term structure for a symbol aligned to its dates."""
    # Extract only this symbol's entries from ts_lookup
    struct_map = {}
    for (sym, dt), struct in ts_lookup.items():
        if sym == symbol:
            struct_map[dt] = struct
    return pd.Series([struct_map.get(d, 'unknown') for d in dates], index=dates)


def apply_term_structure_filter(base_signal, df, ts_lookup):
    """
    Vectorized term structure filter.
    - long only when backwardated
    - short only when contango
    """
    if ts_lookup is None or len(ts_lookup) == 0:
        return base_signal

    symbol = df['ts_code'].iloc[0] if 'ts_code' in df.columns else None
    if symbol is None:
        return pd.Series(0, index=df.index, dtype=int)

    # Build structure series aligned to df dates
    dates = df['trade_date'].values
    structs = np.array(['unknown'] * len(dates), dtype='U20')
    for i, d in enumerate(dates):
        s = ts_lookup.get((symbol, pd.Timestamp(d)), None)
        if s is not None:
            structs[i] = s

    result = base_signal.values.copy()
    # Zero out longs where not backwardation
    mask_long = (result == 1) & (structs != 'backwardation')
    mask_short = (result == -1) & (structs != 'contango')
    mask_unk = (structs == 'unknown')
    result[mask_long | mask_short | mask_unk] = 0
    return pd.Series(result, index=df.index, dtype=int)


# ═══════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════
def run_backtest(all_data, signals_by_symbol, pos_count, holding_period,
                 sl_pct, tp_pct, start_date, end_date, ts_lookup=None):
    """
    Run a multi-commodity portfolio backtest.

    Parameters
    ----------
    all_data : dict[str, DataFrame]   - indicator-enriched price data
    signals_by_symbol : dict[str, Series]  - raw signal per symbol
    pos_count : int        - max positions per side (long/short)
    holding_period : str or int  - 'signal' or int days
    sl_pct : float or None  - stop loss as negative fraction (e.g. -0.02)
    tp_pct : float or None  - take profit as positive fraction (e.g. 0.05)
    start_date, end_date : str
    ts_lookup : dict or None  - (symbol, date) -> structure for term structure filter

    Returns dict with performance metrics.
    """
    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date)

    # Pre-build date-indexed lookups: price_map[sym][date] = close, atr_map[sym][date] = atr
    sig_map = {}
    price_map = {}
    atr_map = {}

    for sym, sig in signals_by_symbol.items():
        df = all_data[sym]
        sig_map[sym] = dict(zip(df['trade_date'], sig))
        price_map[sym] = dict(zip(df['trade_date'], df['close']))
        atr_map[sym] = dict(zip(df['trade_date'], df['atr14'])) if 'atr14' in df.columns else {}

    # Gather all dates in range
    all_dates = set()
    for sym, df in all_data.items():
        mask = (df['trade_date'] >= start_dt) & (df['trade_date'] <= end_dt)
        all_dates.update(df.loc[mask, 'trade_date'].tolist())
    dates = sorted(all_dates)
    if not dates:
        return None

    # Only include symbols that have signals and are in all_data
    active_syms = [s for s in signals_by_symbol if s in all_data]

    positions = {}
    closed_trades = []
    equity = INITIAL_CAPITAL
    equity_curve = []
    cost_rate = COMMISSION_BPS / 10000 * 2 + SLIPPAGE_BPS / 10000 * 2

    for di, date in enumerate(dates):
        # ── Check exits ──
        to_remove = []
        for sym, pos in list(positions.items()):
            price = price_map.get(sym, {}).get(date, None)
            if price is None:
                continue

            pnl_pct = (price - pos['entry_price']) / pos['entry_price'] * pos['direction']
            exit_reason = None

            if sl_pct is not None and pnl_pct <= sl_pct:
                exit_reason = 'SL'
            elif tp_pct is not None and pnl_pct >= tp_pct:
                exit_reason = 'TP'

            if holding_period != 'signal':
                days_held = (date - pos['entry_date']).days
                if days_held >= holding_period:
                    exit_reason = 'hold_period'

            if holding_period == 'signal' and exit_reason is None:
                cur_sig = sig_map.get(sym, {}).get(date, 0)
                if pos['direction'] == 1 and cur_sig <= 0:
                    exit_reason = 'signal_rev'
                elif pos['direction'] == -1 and cur_sig >= 0:
                    exit_reason = 'signal_rev'

            if exit_reason:
                pnl = pnl_pct - cost_rate
                dollar_pnl = pnl * pos['notional']
                equity += dollar_pnl
                closed_trades.append({
                    'symbol': sym, 'direction': pos['direction'],
                    'entry_date': pos['entry_date'], 'exit_date': date,
                    'entry_price': pos['entry_price'], 'exit_price': price,
                    'pnl_pct': pnl, 'pnl_dollar': dollar_pnl,
                    'exit_reason': exit_reason,
                })
                to_remove.append(sym)

        for sym in to_remove:
            del positions[sym]

        # ── Build candidates for new entries ──
        long_cands = []
        short_cands = []
        for sym in active_syms:
            if sym in positions:
                continue
            sm = sig_map.get(sym, {})
            pm = price_map.get(sym, {})
            am = atr_map.get(sym, {})
            cur_sig = sm.get(date, 0)
            price = pm.get(date, None)
            if price is None or price <= 0 or cur_sig == 0:
                continue
            atr = am.get(date, np.nan)
            strength = (atr / price) if (not pd.isna(atr) and atr > 0) else 0
            if cur_sig == 1:
                long_cands.append((sym, strength, price))
            elif cur_sig == -1:
                short_cands.append((sym, strength, price))

        long_cands.sort(key=lambda x: x[1])
        short_cands.sort(key=lambda x: x[1])

        cur_longs = sum(1 for p in positions.values() if p['direction'] == 1)
        cur_shorts = sum(1 for p in positions.values() if p['direction'] == -1)
        slots_long = max(0, pos_count - cur_longs)
        slots_short = max(0, pos_count - cur_shorts)

        notional_per_pos = equity / max(pos_count * 2, 1)

        for sym, strength, price in long_cands[:slots_long]:
            if equity <= 0:
                break
            positions[sym] = {
                'direction': 1, 'entry_price': price,
                'entry_date': date, 'notional': notional_per_pos,
            }

        for sym, strength, price in short_cands[:slots_short]:
            if equity <= 0:
                break
            positions[sym] = {
                'direction': -1, 'entry_price': price,
                'entry_date': date, 'notional': notional_per_pos,
            }

        # Mark-to-market equity
        mtm = 0
        for sym, pos in positions.items():
            price = price_map.get(sym, {}).get(date, None)
            if price is not None:
                mtm += (price - pos['entry_price']) / pos['entry_price'] * pos['direction'] * pos['notional']
        equity_curve.append({'date': date, 'equity': equity + mtm})

    # Close any remaining positions at end date
    # Use the last date from the loop
    last_date = dates[-1] if dates else end_dt
    for sym, pos in list(positions.items()):
        price = price_map.get(sym, {}).get(last_date, None)
        if price is not None:
            pnl_pct = (price - pos['entry_price']) / pos['entry_price'] * pos['direction']
            pnl_pct -= cost_rate
            dollar_pnl = pnl_pct * pos['notional']
            equity += dollar_pnl
            closed_trades.append({
                'symbol': sym, 'direction': pos['direction'],
                'entry_date': pos['entry_date'], 'exit_date': last_date,
                'entry_price': pos['entry_price'], 'exit_price': price,
                'pnl_pct': pnl_pct, 'pnl_dollar': dollar_pnl,
                'exit_reason': 'end',
            })

    if not closed_trades:
        return None

    return _compute_metrics(closed_trades, equity_curve)


def _compute_metrics(trades, equity_curve):
    """Compute standard performance metrics from trades and equity curve."""
    if not trades:
        return None

    tdf = pd.DataFrame(trades)
    total_trades = len(tdf)
    win_trades = (tdf['pnl_pct'] > 0).sum()
    win_rate = win_trades / total_trades if total_trades > 0 else 0

    avg_pnl = tdf['pnl_pct'].mean()
    total_pnl_dollar = tdf['pnl_dollar'].sum()

    # Sharpe (from trade returns, annualized ~252 trading days)
    if len(tdf) > 1:
        sharpe = tdf['pnl_pct'].mean() / tdf['pnl_pct'].std() * np.sqrt(252 / max(avg_bars_per_trade(tdf), 1))
    else:
        sharpe = 0

    # Max Drawdown from equity curve
    if equity_curve:
        eq = pd.DataFrame(equity_curve)
        eq['peak'] = eq['equity'].cummax()
        eq['dd'] = (eq['equity'] - eq['peak']) / eq['peak']
        max_dd = eq['dd'].min()
        final_equity = eq['equity'].iloc[-1]
        total_return = (final_equity - INITIAL_CAPITAL) / INITIAL_CAPITAL
    else:
        max_dd = 0
        total_return = 0
        final_equity = INITIAL_CAPITAL

    # Profit factor
    gross_profit = tdf.loc[tdf['pnl_dollar'] > 0, 'pnl_dollar'].sum()
    gross_loss = abs(tdf.loc[tdf['pnl_dollar'] < 0, 'pnl_dollar'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Calmar
    n_years = max((tdf['exit_date'].max() - tdf['entry_date'].min()).days / 365, 0.1)
    annual_return = (1 + total_return) ** (1 / n_years) - 1 if total_return > -1 else -1
    calmar = annual_return / abs(max_dd) if max_dd != 0 else 0

    # Avg hold period
    tdf['hold_days'] = (tdf['exit_date'] - tdf['entry_date']).dt.days
    avg_hold = tdf['hold_days'].mean()

    # Long / short breakdown
    long_trades = tdf[tdf['direction'] == 1]
    short_trades = tdf[tdf['direction'] == -1]

    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'avg_pnl_pct': avg_pnl * 100,
        'total_return_pct': total_return * 100,
        'annual_return_pct': annual_return * 100,
        'sharpe': round(sharpe, 2),
        'max_dd_pct': max_dd * 100,
        'calmar': round(calmar, 2),
        'profit_factor': round(profit_factor, 2),
        'avg_hold_days': round(avg_hold, 1),
        'final_equity': round(final_equity, 0),
        'long_wins': f"{(long_trades['pnl_pct'] > 0).mean() * 100:.1f}%" if len(long_trades) > 0 else "N/A",
        'short_wins': f"{(short_trades['pnl_pct'] > 0).mean() * 100:.1f}%" if len(short_trades) > 0 else "N/A",
        'long_count': len(long_trades),
        'short_count': len(short_trades),
    }


def avg_bars_per_trade(tdf):
    """Estimate average calendar days per trade for Sharpe annualization."""
    if 'hold_days' in tdf.columns:
        return tdf['hold_days'].mean()
    return (tdf['exit_date'] - tdf['entry_date']).dt.days.mean()


# ═══════════════════════════════════════════════════════════════
# STRATEGY VARIANT RUNNER
# ═══════════════════════════════════════════════════════════════
def generate_signals_for_variant(all_data, variant, params, ts_lookup=None):
    """Generate signals for a specific variant + params combo.

    Returns dict[symbol -> Series of signals].
    """
    signals = {}
    for sym, df in all_data.items():
        sig = None
        if variant == 'dual_ma':
            fast, slow = params
            sig = signal_dual_ma(df, fast, slow)
        elif variant == 'donchian':
            period = params
            sig = signal_donchian(df, period)
        elif variant == 'adx':
            sig = signal_adx(df)
        elif variant == 'trend_oi':
            # Use SMA(20,60) as base trend, filter with OI
            base = signal_dual_ma(df, 20, 60)
            sig = signal_trend_oi(df, base)
        elif variant == 'trend_ts':
            # Use SMA(20,60) as base trend, filter with term structure
            base = signal_dual_ma(df, 20, 60)
            if ts_lookup is not None:
                sig = apply_term_structure_filter(base, df, ts_lookup)
            else:
                sig = base
        else:
            sig = pd.Series(0, index=df.index, dtype=int)
        signals[sym] = sig
    return signals


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    t0 = time.time()

    print("=" * 80)
    print("TREND FOLLOWING STRATEGY BACKTEST v102")
    print("=" * 80)

    # ── Load data ──
    print("\n[1/5] Loading futures daily data...")
    raw_data = load_futures_data(min_days=500)
    print(f"  Loaded {len(raw_data)} commodities")

    print("\n[2/5] Loading term structure data...")
    ts_df = load_term_structure_index()
    # Build lookup dict: (symbol, date) -> structure
    ts_lookup = None
    if not ts_df.empty:
        ts_lookup = {}
        for _, row in ts_df.iterrows():
            ts_lookup[(row['symbol'], row['date'])] = row['structure']
        print(f"  Term structure lookup: {len(ts_lookup)} entries")
    else:
        print("  WARNING: No term structure data loaded.")

    # ── Compute indicators ──
    print("\n[3/5] Computing technical indicators...")
    all_data = {}
    for sym, df in raw_data.items():
        try:
            enriched = compute_indicators(df)
            enriched['ts_code'] = sym
            all_data[sym] = enriched
        except Exception as e:
            print(f"  Skipping {sym}: {e}")
    print(f"  Enriched {len(all_data)} commodities")

    # ── Define all variants ──
    print("\n[4/5] Running strategy variants...")
    variants = [
        ('dual_ma', (10, 30), 'DMA(10,30)'),
        ('dual_ma', (20, 60), 'DMA(20,60)'),
        ('dual_ma', (50, 120), 'DMA(50,120)'),
        ('donchian', 10, 'Donchian(10)'),
        ('donchian', 20, 'Donchian(20)'),
        ('donchian', 40, 'Donchian(40)'),
        ('adx', None, 'ADX(14)>25'),
        ('trend_oi', None, 'Trend+OI'),
        ('trend_ts', None, 'Trend+TermStruct'),
    ]

    # Generate signals for each variant
    variant_signals = {}
    for variant, params, label in variants:
        sigs = generate_signals_for_variant(all_data, variant, params, ts_lookup)
        variant_signals[label] = sigs
        # Count non-zero signals across all symbols for train period
        cnt = 0
        for sym, s in sigs.items():
            df = all_data[sym]
            mask = (df['trade_date'] >= WALK_FORWARD['train'][0]) & \
                   (df['trade_date'] <= WALK_FORWARD['train'][1])
            cnt += (s[mask] != 0).sum()
        print(f"  {label}: {cnt} non-zero signal-days in train period")

    # ── Grid search: train period ──
    print("\n[5/5] Grid search on training period (2021-2023)...")
    print(f"  Variants: {len(variants)}")
    print(f"  Position counts: {POSITION_COUNTS}")
    print(f"  Holding periods: {HOLDING_PERIODS}")
    print(f"  SL/TP pairs: {len(SL_TP_PAIRS)}")
    total_combos = len(variants) * len(POSITION_COUNTS) * len(HOLDING_PERIODS) * len(SL_TP_PAIRS)
    print(f"  Total combinations: {total_combos}")

    results = []
    combo_idx = 0
    for variant, params, label in variants:
        sigs = variant_signals[label]
        for pos_count in POSITION_COUNTS:
            for hold_period in HOLDING_PERIODS:
                for sl, tp in SL_TP_PAIRS:
                    combo_idx += 1
                    if combo_idx % 50 == 0:
                        print(f"  Progress: {combo_idx}/{total_combos}...", end='\r')
                    try:
                        res = run_backtest(
                            all_data, sigs, pos_count, hold_period, sl, tp,
                            WALK_FORWARD['train'][0], WALK_FORWARD['train'][1],
                            ts_lookup=None,
                        )
                    except Exception as e:
                        res = None
                    if res:
                        res['variant'] = label
                        res['pos_count'] = pos_count
                        res['hold'] = str(hold_period)
                        sl_str = f"{sl*100:.0f}%" if sl else "None"
                        tp_str = f"{tp*100:.0f}%" if tp else "None"
                        res['sl_tp'] = f"{sl_str}/{tp_str}"
                        results.append(res)

    print(f"  Completed {combo_idx} combinations.                ")

    if not results:
        print("ERROR: No valid results. Check data and parameters.")
        return

    # ── Print training results ──
    rdf = pd.DataFrame(results)
    rdf = rdf.sort_values('sharpe', ascending=False)

    print("\n" + "=" * 120)
    print("TRAINING RESULTS (2021-2023) - Top 30 by Sharpe")
    print("=" * 120)
    cols = ['variant', 'pos_count', 'hold', 'sl_tp', 'total_trades', 'win_rate',
            'avg_pnl_pct', 'total_return_pct', 'annual_return_pct', 'sharpe',
            'max_dd_pct', 'calmar', 'profit_factor', 'avg_hold_days']
    print(rdf[cols].head(30).to_string(index=False))

    # ── Select best by variant ──
    print("\n" + "=" * 120)
    print("BEST PER VARIANT (Training)")
    print("=" * 120)
    best_per_variant = []
    for variant, params, label in variants:
        mask = rdf['variant'] == label
        if mask.any():
            best = rdf[mask].iloc[0]
            best_per_variant.append(best)
    bpv_df = pd.DataFrame(best_per_variant).sort_values('sharpe', ascending=False)
    print(bpv_df[cols].to_string(index=False))

    # ── Walk-forward validation ──
    print("\n" + "=" * 120)
    print("WALK-FORWARD VALIDATION (2024)")
    print("=" * 120)

    wf_results = []
    # Take top-5 from training, validate
    top5 = rdf.head(5)
    for _, row in top5.iterrows():
        label = row['variant']
        sigs = variant_signals[label]
        hold = int(row['hold']) if row['hold'] != 'signal' else 'signal'
        sl_raw = row['sl_tp']
        # Parse sl_tp
        parts = sl_raw.split('/')
        sl_val = float(parts[0].replace('%', '')) / 100 if parts[0] != 'None' else None
        tp_val = float(parts[1].replace('%', '')) / 100 if parts[1] != 'None' else None

        try:
            val_res = run_backtest(
                all_data, sigs, int(row['pos_count']), hold, sl_val, tp_val,
                WALK_FORWARD['validate'][0], WALK_FORWARD['validate'][1]
            )
        except Exception:
            val_res = None
        if val_res:
            val_res['variant'] = label
            val_res['pos_count'] = row['pos_count']
            val_res['hold'] = row['hold']
            val_res['sl_tp'] = row['sl_tp']
            val_res['period'] = 'validate'
            wf_results.append(val_res)

    if wf_results:
        wf_df = pd.DataFrame(wf_results).sort_values('sharpe', ascending=False)
        print(wf_df[cols].to_string(index=False))
    else:
        print("  No valid validation results.")

    # ── Walk-forward test on best ──
    print("\n" + "=" * 120)
    print("WALK-FORWARD TEST (2025-2026)")
    print("=" * 120)

    if wf_results:
        # Pick best from validation by Sharpe
        best_wf = wf_df.iloc[0]
        label = best_wf['variant']
        sigs = variant_signals[label]
        hold = int(best_wf['hold']) if best_wf['hold'] != 'signal' else 'signal'
        sl_raw = best_wf['sl_tp']
        parts = sl_raw.split('/')
        sl_val = float(parts[0].replace('%', '')) / 100 if parts[0] != 'None' else None
        tp_val = float(parts[1].replace('%', '')) / 100 if parts[1] != 'None' else None

        test_res = run_backtest(
            all_data, sigs, int(best_wf['pos_count']), hold, sl_val, tp_val,
            WALK_FORWARD['test'][0], WALK_FORWARD['test'][1]
        )
        if test_res:
            test_res['variant'] = label
            test_res['pos_count'] = best_wf['pos_count']
            test_res['hold'] = best_wf['hold']
            test_res['sl_tp'] = best_wf['sl_tp']
            test_res['period'] = 'test'
            test_df = pd.DataFrame([test_res])
            print(test_df[cols].to_string(index=False))

            # ── Full walk-forward summary for best ──
            print("\n" + "=" * 120)
            print(f"WALK-FORWARD SUMMARY: {label} | pos={best_wf['pos_count']} | hold={best_wf['hold']} | SL/TP={best_wf['sl_tp']}")
            print("=" * 120)
            # Re-run all three periods for comparison
            for period_name, (sd, ed) in WALK_FORWARD.items():
                res = run_backtest(
                    all_data, sigs, int(best_wf['pos_count']), hold, sl_val, tp_val, sd, ed
                )
                if res:
                    print(f"\n  {period_name.upper()} ({sd} to {ed}):")
                    print(f"    Trades: {res['total_trades']}  |  Win Rate: {res['win_rate']:.1%}")
                    print(f"    Total Return: {res['total_return_pct']:.2f}%  |  Annual: {res['annual_return_pct']:.2f}%")
                    print(f"    Sharpe: {res['sharpe']}  |  MaxDD: {res['max_dd_pct']:.2f}%  |  Calmar: {res['calmar']}")
                    print(f"    Profit Factor: {res['profit_factor']}  |  Avg Hold: {res['avg_hold_days']} days")
                    print(f"    Long: {res['long_count']} trades ({res['long_wins']} win)  |  Short: {res['short_count']} trades ({res['short_wins']} win)")
        else:
            print("  No valid test results.")

    # ── Also test all best-per-variant on 2025-2026 ──
    print("\n" + "=" * 120)
    print("ALL VARIANT BEST - OOS TEST (2025-2026)")
    print("=" * 120)

    oos_results = []
    for _, row in bpv_df.iterrows():
        label = row['variant']
        sigs = variant_signals[label]
        hold = int(row['hold']) if row['hold'] != 'signal' else 'signal'
        sl_raw = row['sl_tp']
        parts = sl_raw.split('/')
        sl_val = float(parts[0].replace('%', '')) / 100 if parts[0] != 'None' else None
        tp_val = float(parts[1].replace('%', '')) / 100 if parts[1] != 'None' else None

        try:
            res = run_backtest(
                all_data, sigs, int(row['pos_count']), hold, sl_val, tp_val,
                WALK_FORWARD['test'][0], WALK_FORWARD['test'][1]
            )
        except Exception:
            res = None
        if res:
            res['variant'] = label
            res['pos_count'] = row['pos_count']
            res['hold'] = row['hold']
            res['sl_tp'] = row['sl_tp']
            oos_results.append(res)

    if oos_results:
        oos_df = pd.DataFrame(oos_results).sort_values('sharpe', ascending=False)
        print(oos_df[cols].to_string(index=False))
    else:
        print("  No OOS results.")

    # ── Detailed analysis: equity curve for best ──
    print("\n" + "=" * 120)
    print("DETAILED EQUITY CURVE ANALYSIS (Best Strategy - Full Period 2021-2026)")
    print("=" * 120)

    if wf_results:
        best_label = wf_df.iloc[0]['variant']
        best_sigs = variant_signals[best_label]
        best_hold = int(wf_df.iloc[0]['hold']) if wf_df.iloc[0]['hold'] != 'signal' else 'signal'
        best_sl_raw = wf_df.iloc[0]['sl_tp']
        bparts = best_sl_raw.split('/')
        best_sl = float(bparts[0].replace('%', '')) / 100 if bparts[0] != 'None' else None
        best_tp = float(bparts[1].replace('%', '')) / 100 if bparts[1] != 'None' else None
        best_pos = int(wf_df.iloc[0]['pos_count'])

        # Run full period with equity curve output
        full_res = run_backtest(
            all_data, best_sigs, best_pos, best_hold, best_sl, best_tp,
            '2021-01-01', '2026-05-21'
        )
        if full_res:
            print(f"\n  Strategy: {best_label} | Positions: {best_pos} | Hold: {best_hold} | SL/TP: {best_sl_raw}")
            print(f"  Full Period Performance:")
            print(f"    Total Trades: {full_res['total_trades']}")
            print(f"    Win Rate: {full_res['win_rate']:.1%}")
            print(f"    Total Return: {full_res['total_return_pct']:.2f}%")
            print(f"    Annual Return: {full_res['annual_return_pct']:.2f}%")
            print(f"    Sharpe: {full_res['sharpe']}")
            print(f"    Max Drawdown: {full_res['max_dd_pct']:.2f}%")
            print(f"    Calmar: {full_res['calmar']}")
            print(f"    Profit Factor: {full_res['profit_factor']}")
            print(f"    Avg Hold Days: {full_res['avg_hold_days']}")
            print(f"    Final Equity: {full_res['final_equity']:,.0f}")
            print(f"    Long: {full_res['long_count']} ({full_res['long_wins']} win)")
            print(f"    Short: {full_res['short_count']} ({full_res['short_wins']} win)")

    elapsed = time.time() - t0
    print(f"\nTotal execution time: {elapsed:.1f}s")
    print("=" * 80)


if __name__ == '__main__':
    main()
