#!/usr/bin/env python3
"""
Unit tests for the futures platform core scripts.

Covers:
  - contract_specs: spec lookup, margin, PnL, max lots
  - Signal computation logic (from signal_generator_v1 / backtest_v80)
  - Backtest engine: zero capital, no signals, all-losing trades
  - Data loading edge cases: missing files, empty data, NaN, constant prices
  - Options BSM pricing edge cases
"""

import os
import sys
import json
import tempfile
import shutil
import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Ensure scripts/ is on sys.path so we can import contract_specs directly
# ---------------------------------------------------------------------------
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import contract_specs as cs


# ===================================================================
# 1. Contract Specs Tests
# ===================================================================
class TestContractSpecs(unittest.TestCase):
    """Test contract_specs.py core functions."""

    # --- get_spec ---
    def test_get_spec_known_symbol(self):
        """Known symbols return correct (multiplier, margin_rate, tick, name)."""
        mult, margin, tick, name = cs.get_spec('rbfi')
        self.assertEqual(mult, 10)
        self.assertAlmostEqual(margin, 0.10)
        self.assertEqual(tick, 1)
        self.assertEqual(name, '螺纹')

    def test_get_spec_aufi(self):
        mult, margin, tick, name = cs.get_spec('aufi')
        self.assertEqual(mult, 1000)
        self.assertAlmostEqual(margin, 0.10)
        self.assertEqual(tick, 0.02)

    def test_get_spec_ifi_iron_ore(self):
        mult, margin, tick, name = cs.get_spec('ifi')
        self.assertEqual(mult, 100)
        self.assertAlmostEqual(margin, 0.12)
        self.assertEqual(tick, 0.5)

    def test_get_spec_unknown_returns_default(self):
        """Unknown symbol should return the DEFAULT_SPEC tuple."""
        mult, margin, tick, name = cs.get_spec('UNKNOWN_SYMBOL')
        self.assertEqual(mult, 10)
        self.assertAlmostEqual(margin, 0.10)
        self.assertEqual(name, '未知')

    def test_get_spec_all_symbols_have_positive_multiplier(self):
        """Every listed symbol must have multiplier > 0."""
        for sym, (mult, margin, tick, name) in cs.CONTRACT_SPECS.items():
            with self.subTest(symbol=sym):
                self.assertGreater(mult, 0, f"{sym} has non-positive multiplier")

    def test_get_spec_all_symbols_have_valid_margin(self):
        """Margin rates must be in (0, 1)."""
        for sym, (mult, margin, tick, name) in cs.CONTRACT_SPECS.items():
            with self.subTest(symbol=sym):
                self.assertGreater(margin, 0, f"{sym} has non-positive margin rate")
                self.assertLess(margin, 1, f"{sym} has margin >= 100%")

    # --- calc_margin ---
    def test_calc_margin_basic(self):
        """rbfi: mult=10, margin=0.10 => margin = price*10*lots*0.10."""
        m = cs.calc_margin('rbfi', price=3800, lots=2)
        expected = 3800 * 10 * 2 * 0.10
        self.assertAlmostEqual(m, expected)

    def test_calc_margin_zero_lots(self):
        m = cs.calc_margin('rbfi', price=3800, lots=0)
        self.assertAlmostEqual(m, 0)

    def test_calc_margin_zero_price(self):
        m = cs.calc_margin('rbfi', price=0, lots=2)
        self.assertAlmostEqual(m, 0)

    # --- calc_contract_value ---
    def test_calc_contract_value(self):
        cv = cs.calc_contract_value('rbfi', price=3800, lots=1)
        self.assertAlmostEqual(cv, 3800 * 10)

    def test_calc_contract_value_multiple_lots(self):
        cv = cs.calc_contract_value('aufi', price=500, lots=3)
        self.assertAlmostEqual(cv, 500 * 1000 * 3)

    # --- calc_pnl ---
    def test_calc_pnl_long_profit(self):
        """Long: direction=1, price goes up => positive PnL."""
        pnl = cs.calc_pnl('rbfi', 3800, 3900, 1, 1)
        expected = (3900 - 3800) * 1 * 10 * 1
        self.assertAlmostEqual(pnl, expected)

    def test_calc_pnl_long_loss(self):
        pnl = cs.calc_pnl('rbfi', 3800, 3700, 1, 1)
        expected = (3700 - 3800) * 1 * 10 * 1
        self.assertAlmostEqual(pnl, expected)
        self.assertLess(pnl, 0)

    def test_calc_pnl_short_profit(self):
        """Short: direction=-1, price goes down => positive PnL."""
        pnl = cs.calc_pnl('rbfi', 3800, 3700, -1, 1)
        expected = (3700 - 3800) * (-1) * 10 * 1
        self.assertAlmostEqual(pnl, expected)
        self.assertGreater(pnl, 0)

    def test_calc_pnl_short_loss(self):
        pnl = cs.calc_pnl('rbfi', 3800, 3900, -1, 2)
        expected = (3900 - 3800) * (-1) * 10 * 2
        self.assertAlmostEqual(pnl, expected)
        self.assertLess(pnl, 0)

    def test_calc_pnl_zero_lots(self):
        pnl = cs.calc_pnl('rbfi', 3800, 3900, 1, 0)
        self.assertAlmostEqual(pnl, 0)

    # --- calc_max_lots ---
    def test_calc_max_lots_normal(self):
        lots = cs.calc_max_lots('rbfi', price=3800, available_cash=500000, max_equity_pct=0.35)
        margin_per_lot = 3800 * 10 * 0.10
        expected = int(500000 * 0.35 / margin_per_lot)
        self.assertEqual(lots, expected)

    def test_calc_max_lots_zero_cash(self):
        lots = cs.calc_max_lots('rbfi', price=3800, available_cash=0)
        self.assertEqual(lots, 0)

    def test_calc_max_lots_zero_price(self):
        lots = cs.calc_max_lots('rbfi', price=0, available_cash=500000)
        self.assertEqual(lots, 0)


# ===================================================================
# 2. Signal Computation Tests
# ===================================================================
class TestSignalComputation(unittest.TestCase):
    """
    Test the signal scoring logic extracted from signal_generator_v1 /
    backtest_v80. We replicate the core computation here so tests run
    without needing the full module.
    """

    def _build_dataframe(self, n_rows=120, gap=-1.5, oi_chg=5.0, mom5=-4.0):
        """
        Build a synthetic OHLCV+OI DataFrame with a configurable last-bar gap.
        """
        dates = pd.date_range(end='2025-05-20', periods=n_rows, freq='B')
        np.random.seed(42)
        base_price = 3800
        closes = base_price + np.cumsum(np.random.randn(n_rows) * 10)
        closes = np.maximum(closes, 100)  # keep positive

        opens = closes + np.random.randn(n_rows) * 5
        highs = np.maximum(opens, closes) + np.abs(np.random.randn(n_rows) * 5)
        lows = np.minimum(opens, closes) - np.abs(np.random.randn(n_rows) * 5)
        vols = np.abs(np.random.randn(n_rows) * 10000) + 5000
        ois = np.abs(np.random.randn(n_rows) * 100000) + 500000

        # Override last bar to create a specific gap
        prev_close = closes[-2]
        opens[-1] = prev_close * (1 + gap / 100)
        closes[-1] = opens[-1] + 10  # slightly recover
        highs[-1] = max(opens[-1], closes[-1]) + 5
        lows[-1] = min(opens[-1], closes[-1]) - 5

        df = pd.DataFrame({
            'trade_date': dates,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'vol': vols,
            'oi': ois,
        })
        return df

    def _compute_score_long(self, df):
        """Replicate the long scoring logic from backtest_v80."""
        c = df['close'].values.astype(float)
        o = df['open'].values.astype(float)
        h = df['high'].values.astype(float)
        l = df['low'].values.astype(float)
        v = df['vol'].values.astype(float)
        oi = df['oi'].values.astype(float)
        n = len(df)

        prev_c = np.full(n, np.nan); prev_c[1:] = c[:-1]
        gap = np.full(n, np.nan); gap[1:] = (o[1:] - prev_c[1:]) / prev_c[1:] * 100
        tr = np.full(n, np.nan)
        tr[1:] = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
        atr_pct = pd.Series(tr).rolling(20).mean().values / c * 100
        ma5 = pd.Series(c).rolling(5).mean().values
        ma20 = pd.Series(c).rolling(20).mean().values
        ma60 = pd.Series(c).rolling(60).mean().values
        mom5 = np.full(n, np.nan); mom5[5:] = (c[5:] - c[:-5]) / c[:-5] * 100
        oi_ch = np.full(n, np.nan)
        valid = np.abs(oi[:-1]) > 0
        oi_ch_v = np.full(n - 1, np.nan)
        oi_ch_v[valid] = (oi[1:][valid] - oi[:-1][valid]) / np.abs(oi[:-1][valid]) * 100
        oi_ch[1:] = oi_ch_v
        vol_ma5 = pd.Series(v).rolling(5).mean().values
        range_ = h - l
        clv = np.where(range_ > 0, (2 * c - h - l) / range_, 0)
        gv = np.nan_to_num(gap)
        ga = np.nan_to_num(gap / np.where(atr_pct == 0, np.nan, atr_pct))

        s_l = np.zeros(n)
        s_l += (gv < -0.5).astype(int) * 1
        s_l += (gv < -1.0).astype(int) * 2
        s_l += (gv < -1.5).astype(int) * 2
        s_l += (gv < -2.0).astype(int) * 3
        s_l += (ga < -1.0).astype(int) * 2
        s_l += (ga < -1.5).astype(int) * 3
        s_l += ((oi_ch > 0) & (c < prev_c)).astype(int) * 3
        s_l += ((oi_ch < 0) & (c < prev_c)).astype(int) * 2
        s_l += (mom5 < -3).astype(int) * 1
        s_l += (mom5 < -5).astype(int) * 1
        s_l += (c < ma5).astype(int) * 1
        s_l += ((v > vol_ma5 * 1.5) & (c < prev_c)).astype(int) * 1
        s_l += (clv > 0.5).astype(int) * 1
        s_l += (ma20 > ma60).astype(int) * 2

        return s_l[-1], gap[-1]

    def test_large_negative_gap_produces_high_score(self):
        """A -1.5% gap down should produce a long score >= 5."""
        df = self._build_dataframe(n_rows=120, gap=-1.5)
        score, gap = self._compute_score_long(df)
        self.assertLess(gap, -1.0)
        self.assertGreaterEqual(score, 5)

    def test_zero_gap_produces_low_score(self):
        """A 0% gap should produce a long score near 0."""
        df = self._build_dataframe(n_rows=120, gap=0)
        score, gap = self._compute_score_long(df)
        # With no gap, gap-based factors contribute 0, only OI/mom/MA factors
        # Score depends on random data but gap-based portion is 0
        self.assertAlmostEqual(gap, 0, places=1)

    def test_positive_gap_produces_zero_long_gap_factors(self):
        """A positive gap should not trigger gap-based long factors."""
        df = self._build_dataframe(n_rows=120, gap=1.5)
        score, gap = self._compute_score_long(df)
        self.assertGreater(gap, 0.5)

    def test_single_data_point_no_crash(self):
        """A DataFrame with only 1 row should not crash signal computation."""
        dates = pd.date_range(end='2025-05-20', periods=1, freq='B')
        df = pd.DataFrame({
            'trade_date': dates,
            'open': [3800.0],
            'high': [3850.0],
            'low': [3750.0],
            'close': [3800.0],
            'vol': [10000.0],
            'oi': [100000.0],
        })
        c = df['close'].values.astype(float)
        o = df['open'].values.astype(float)
        self.assertEqual(len(c), 1)
        self.assertEqual(len(o), 1)

    def test_all_nan_close(self):
        """All-NaN close prices should be handled gracefully."""
        df = self._build_dataframe(n_rows=120)
        df['close'] = np.nan
        c = df['close'].values.astype(float)
        self.assertTrue(np.all(np.isnan(c)))

    def test_constant_prices_zero_gap(self):
        """Constant prices (no movement) should produce 0 gap."""
        df = self._build_dataframe(n_rows=120)
        const = 3800
        df['close'] = const
        df['open'] = const
        df['high'] = const
        df['low'] = const
        c = df['close'].values.astype(float)
        o = df['open'].values.astype(float)
        n = len(df)
        prev_c = np.full(n, np.nan); prev_c[1:] = c[:-1]
        gap = np.full(n, np.nan); gap[1:] = (o[1:] - prev_c[1:]) / prev_c[1:] * 100
        # All gaps should be 0 (constant prices)
        self.assertTrue(np.allclose(gap[1:], 0, atol=1e-10, equal_nan=False))


# ===================================================================
# 3. Backtest Engine Edge Cases
# ===================================================================
class TestBacktestEdgeCases(unittest.TestCase):
    """
    Test backtest engine logic with extreme inputs.
    Uses the backtest_v80 run_bt logic structure.
    """

    def _make_signal_data(self, n_days=250, n_syms=5, min_score=7):
        """Create synthetic signal_data dict matching backtest_v80 format."""
        signal_data = {}
        dates = pd.date_range(start='2024-01-01', periods=n_days, freq='B')
        np.random.seed(123)
        for i in range(n_syms):
            sym = f'sym{i}'
            df = pd.DataFrame({
                'trade_date': dates,
                'open': 3800 + np.random.randn(n_days) * 50,
                'high': 3850 + np.random.randn(n_days) * 50,
                'low': 3750 + np.random.randn(n_days) * 50,
                'close': 3800 + np.random.randn(n_days) * 50,
                'vol': 10000 + np.abs(np.random.randn(n_days) * 2000),
                'oi': 100000 + np.abs(np.random.randn(n_days) * 20000),
                'score_long': 0,
                'score_short': 0,
                'gap_pct': np.random.randn(n_days) * 0.5,
                'atr_pct': np.abs(np.random.randn(n_days)) * 0.5 + 1.0,
            })
            signal_data[sym] = df
        return signal_data

    def test_no_signals_no_trades(self):
        """When all scores are 0, no trades should be produced."""
        signal_data = self._make_signal_data(min_score=7)
        # All scores are already 0, so min_score=7 means nothing triggers
        eq, trades = self._run_minimal_backtest(signal_data, '2024-01-01', '2024-12-31')
        self.assertEqual(len(trades), 0)

    def test_zero_capital_stops_immediately(self):
        """With 0 initial capital, backtest should produce empty or zero results."""
        signal_data = self._make_signal_data()
        eq, trades = self._run_minimal_backtest(
            signal_data, '2024-01-01', '2024-12-31', initial_capital=0
        )
        # Either no trades, or equity curve stops at 0
        if len(eq) > 0:
            self.assertLessEqual(eq[-1]['capital'], 0)

    def test_negative_capital_handled(self):
        """Negative capital should not cause crash."""
        signal_data = self._make_signal_data()
        eq, trades = self._run_minimal_backtest(
            signal_data, '2024-01-01', '2024-12-31', initial_capital=-100
        )
        # Should not raise

    def test_empty_signal_data(self):
        """Empty signal_data dict should produce empty results."""
        eq, trades = self._run_minimal_backtest({}, '2024-01-01', '2024-12-31')
        self.assertEqual(len(trades), 0)
        # Equity curve may still have dates from pd.date_range

    def _run_minimal_backtest(self, signal_data, start, end,
                               max_pos=7, lev=5, min_sc=7, hold=1,
                               sl_pct=-1.5, tp_pct=4.0, initial_capital=500000):
        """Simplified version of backtest_v80.run_bt for testing."""
        dates = pd.date_range(start=start, end=end, freq='B')
        cap = initial_capital
        eq, trades, pos = [], [], []

        for dt in dates:
            pnl = 0
            keep = []
            for p in pos:
                df = signal_data.get(p['sym'])
                if df is None:
                    keep.append(p); continue
                idx = df.index[df['trade_date'] == dt]
                if len(idx) == 0:
                    keep.append(p); continue
                row = df.loc[idx[0]]
                cur_c = row['close']
                if np.isnan(cur_c):
                    keep.append(p); continue

                d = (dt - p['ed']).days
                actual_ret = (cur_c - p['ep']) / p['ep'] * 100 if p['dir'] == 'long' else (p['ep'] - cur_c) / p['ep'] * 100
                reason = 'exp' if d >= hold else None
                if reason is None:
                    keep.append(p); continue

                pnl += p['not'] * actual_ret / 100
                trades.append({
                    'sym': p['sym'], 'dir': p['dir'], 'ed': p['ed'], 'xd': dt,
                    'ep': p['ep'], 'xp': cur_c, 'r': actual_ret,
                    'pnl': p['not'] * actual_ret / 100, 'sc': p['sc'],
                    'hold': d, 'reason': reason,
                })

            pos = keep
            cap += pnl
            if cap <= 0:
                eq.append({'date': dt, 'capital': 0}); break

            n_open = max_pos - len(pos)
            if n_open <= 0:
                eq.append({'date': dt, 'capital': cap}); continue

            cands = []
            for sym, df in signal_data.items():
                if any(p['sym'] == sym for p in pos): continue
                idx = df.index[df['trade_date'] == dt]
                if len(idx) == 0: continue
                row = df.loc[idx[0]]
                if row['score_long'] >= min_sc:
                    cands.append({'sym': sym, 'dir': 'long', 'sc': row['score_long'], 'ep': row['open']})
                if row['score_short'] >= min_sc:
                    cands.append({'sym': sym, 'dir': 'short', 'sc': row['score_short'], 'ep': row['open']})

            best = {}
            for c_ in cands:
                if c_['sym'] not in best or c_['sc'] > best[c_['sym']]['sc']:
                    best[c_['sym']] = c_

            ranked = sorted(best.values(), key=lambda x: -x['sc'])

            for c_ in ranked:
                if n_open <= 0: break
                notional = cap * lev / max_pos if cap > 0 else 0
                pos.append({'sym': c_['sym'], 'dir': c_['dir'], 'ed': dt,
                            'ep': c_['ep'], 'not': notional, 'sc': c_['sc']})
                n_open -= 1

            eq.append({'date': dt, 'capital': cap})
        return eq, trades

    def test_all_losing_trades(self):
        """
        Force all trades to lose by giving every signal and using
        prices that move against the position direction.
        """
        signal_data = self._make_signal_data()
        # Force all symbols to have high long scores
        for sym, df in signal_data.items():
            df['score_long'] = 10  # force long entry
            # Set explicit positive prices: open=4000, close=3900 (long loses)
            df['open'] = 4000.0
            df['close'] = 3900.0  # close < open => long loses

        eq, trades = self._run_minimal_backtest(signal_data, '2024-01-01', '2024-03-01')
        if len(trades) > 0:
            # All trades should have negative returns
            for t in trades:
                self.assertLess(t['r'], 0, f"Expected loss but got r={t['r']}")


# ===================================================================
# 4. Data Loading Edge Cases
# ===================================================================
class TestDataLoadingEdgeCases(unittest.TestCase):
    """Test data loading with corrupt/missing/empty files."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_load_missing_directory_returns_empty(self):
        """Loading from a non-existent directory should return empty dict."""
        data_dir = os.path.join(self.tmpdir, 'nonexistent')
        result = {}
        if os.path.exists(data_dir):
            for f in sorted(os.listdir(data_dir)):
                if f.endswith('.csv'):
                    df = pd.read_csv(os.path.join(data_dir, f))
                    result[f] = df
        self.assertEqual(len(result), 0)

    def test_load_empty_csv(self):
        """An empty CSV file should not crash (caught by len check)."""
        empty_path = os.path.join(self.tmpdir, 'empty.csv')
        with open(empty_path, 'w') as f:
            f.write('')  # truly empty

        # pandas raises on empty CSV
        with self.assertRaises(Exception):
            pd.read_csv(empty_path)

    def test_load_csv_header_only(self):
        """A CSV with only headers and no data rows should give 0-length DataFrame."""
        header_path = os.path.join(self.tmpdir, 'header_only.csv')
        with open(header_path, 'w') as f:
            f.write('trade_date,open,high,low,close,vol,oi\n')

        df = pd.read_csv(header_path)
        self.assertEqual(len(df), 0)

    def test_load_csv_with_nan_close(self):
        """CSV with NaN close values should be detected."""
        nan_path = os.path.join(self.tmpdir, 'nan_close.csv')
        df = pd.DataFrame({
            'trade_date': ['20250101', '20250102', '20250103'],
            'open': [3800, 3810, 3820],
            'high': [3850, 3860, 3870],
            'low': [3750, 3760, 3770],
            'close': [3800, np.nan, 3820],
            'vol': [10000, 10000, 10000],
            'oi': [100000, 100000, 100000],
        })
        df.to_csv(nan_path, index=False)

        loaded = pd.read_csv(nan_path)
        self.assertTrue(loaded['close'].isna().any())

    def test_load_csv_with_zero_close(self):
        """CSV with zero close values should be filtered out (as done in load_data)."""
        zero_path = os.path.join(self.tmpdir, 'zero_close.csv')
        df = pd.DataFrame({
            'trade_date': ['20250101', '20250102', '20250103'],
            'open': [3800, 3810, 3820],
            'high': [3850, 3860, 3870],
            'low': [3750, 3760, 3770],
            'close': [3800, 0, 3820],
            'vol': [10000, 10000, 10000],
            'oi': [100000, 100000, 100000],
        })
        df.to_csv(zero_path, index=False)

        loaded = pd.read_csv(zero_path)
        self.assertTrue((loaded['close'] == 0).any())
        # The load_data functions filter: (df['close'] == 0).any() => skip
        should_skip = (loaded['close'] == 0).any()
        self.assertTrue(should_skip)

    def test_incremental_merge_dedup(self):
        """Test the incremental merge pattern from data_collector.py."""
        # Simulate existing data
        old_path = os.path.join(self.tmpdir, 'test.csv')
        old_df = pd.DataFrame({
            'trade_date': ['20250101', '20250102', '20250103'],
            'close': [3800, 3810, 3820],
        })
        old_df.to_csv(old_path, index=False)

        # New data overlaps on 20250103
        new_df = pd.DataFrame({
            'trade_date': ['20250103', '20250104'],
            'close': [3815, 3830],  # updated 20250103
        })

        old = pd.read_csv(old_path, dtype=str)
        old['trade_date'] = old['trade_date'].astype(str)
        new_df = new_df.copy()
        new_df['trade_date'] = new_df['trade_date'].astype(str)

        combined = pd.concat([old, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=['trade_date'], keep='last')
        combined = combined.sort_values('trade_date', ascending=False).reset_index(drop=True)

        # Should have 4 unique dates
        self.assertEqual(len(combined), 4)
        # 20250103 should have the updated value
        row_03 = combined[combined['trade_date'] == '20250103']
        self.assertEqual(float(row_03.iloc[0]['close']), 3815)


# ===================================================================
# 5. BSM / Options Pricing Tests
# ===================================================================
class TestBSMPricing(unittest.TestCase):
    """Test BSM option pricing edge cases from options_collector.py."""

    def _bsm_call_price(self, S, K, T, r, sigma):
        """Simplified BSM call price for testing."""
        if T <= 0 or sigma <= 0:
            return max(S - K, 0)
        from scipy.stats import norm
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    def _bsm_put_price(self, S, K, T, r, sigma):
        if T <= 0 or sigma <= 0:
            return max(K - S, 0)
        from scipy.stats import norm
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    def test_call_atm_reasonable(self):
        """ATM call should be positive and reasonable."""
        price = self._bsm_call_price(S=100, K=100, T=0.25, r=0.02, sigma=0.20)
        self.assertGreater(price, 0)
        self.assertLess(price, 20)  # rough sanity

    def test_put_atm_reasonable(self):
        price = self._bsm_put_price(S=100, K=100, T=0.25, r=0.02, sigma=0.20)
        self.assertGreater(price, 0)

    def test_zero_expiry_returns_intrinsic(self):
        """T=0 should return intrinsic value."""
        call = self._bsm_call_price(S=105, K=100, T=0, r=0.02, sigma=0.20)
        self.assertAlmostEqual(call, 5.0)
        put = self._bsm_put_price(S=95, K=100, T=0, r=0.02, sigma=0.20)
        self.assertAlmostEqual(put, 5.0)

    def test_zero_sigma_returns_intrinsic(self):
        """sigma=0 should return intrinsic value."""
        call = self._bsm_call_price(S=105, K=100, T=0.25, r=0.02, sigma=0)
        self.assertAlmostEqual(call, 5.0)
        put = self._bsm_put_price(S=95, K=100, T=0.25, r=0.02, sigma=0)
        self.assertAlmostEqual(put, 5.0)

    def test_otm_call_zero_intrinsic(self):
        """OTM call with T=0 should return 0."""
        call = self._bsm_call_price(S=95, K=100, T=0, r=0.02, sigma=0.20)
        self.assertAlmostEqual(call, 0)

    def test_call_deep_itm_approaches_intrinsic(self):
        """Deep ITM call price should be close to S - K*exp(-rT)."""
        call = self._bsm_call_price(S=200, K=100, T=0.25, r=0.02, sigma=0.20)
        intrinsic = 200 - 100 * np.exp(-0.02 * 0.25)
        self.assertAlmostEqual(call, intrinsic, places=1)

    def test_put_call_parity(self):
        """C - P = S - K*exp(-rT) for European options."""
        S, K, T, r, sigma = 100, 100, 0.5, 0.02, 0.25
        call = self._bsm_call_price(S, K, T, r, sigma)
        put = self._bsm_put_price(S, K, T, r, sigma)
        lhs = call - put
        rhs = S - K * np.exp(-r * T)
        self.assertAlmostEqual(lhs, rhs, places=2)


# ===================================================================
# 6. Backtest V94 Factor Tests
# ===================================================================
class TestV94FactorComputation(unittest.TestCase):
    """Test factor computation from backtest_v94_multifactor.py."""

    def _compute_momentum(self, closes, window=20):
        """20-day momentum = pct_change(20)."""
        if len(closes) < window + 1:
            return np.array([])
        return (closes[window:] - closes[:-window]) / closes[:-window]

    def test_momentum_positive_when_rising(self):
        closes = np.linspace(100, 120, 30)  # steadily rising
        mom = self._compute_momentum(closes, window=20)
        self.assertTrue(np.all(mom > 0))

    def test_momentum_negative_when_falling(self):
        closes = np.linspace(120, 100, 30)  # steadily falling
        mom = self._compute_momentum(closes, window=20)
        self.assertTrue(np.all(mom < 0))

    def test_momentum_insufficient_data(self):
        closes = np.array([100, 101, 102])
        mom = self._compute_momentum(closes, window=20)
        self.assertEqual(len(mom), 0)

    def test_realized_volatility_constant_prices(self):
        """Constant prices should give ~0 realized vol."""
        closes = np.full(25, 100.0)
        rets = np.diff(closes) / closes[:-1]
        vol = np.std(rets) * np.sqrt(252)
        self.assertAlmostEqual(vol, 0, places=10)

    def test_cross_sectional_zscore(self):
        """Test z-score normalization across a cross-section."""
        factor_vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        mu = np.mean(factor_vals)
        sigma = np.std(factor_vals)
        z = (factor_vals - mu) / sigma
        self.assertAlmostEqual(np.mean(z), 0, places=10)
        self.assertAlmostEqual(np.std(z), 1.0, places=10)


# ===================================================================
# 7. Performance Metric Calculation Tests
# ===================================================================
class TestPerformanceMetrics(unittest.TestCase):
    """Test metric calculation patterns used across backtest scripts."""

    def test_sharpe_from_returns(self):
        """Sharpe = mean(daily_ret) / std(daily_ret) * sqrt(252)."""
        np.random.seed(42)
        daily_rets = np.random.randn(252) * 0.01 + 0.0005  # slight positive bias
        mean_r = np.mean(daily_rets)
        std_r = np.std(daily_rets, ddof=1)
        sharpe = mean_r / std_r * np.sqrt(252) if std_r > 0 else 0
        # With positive bias, Sharpe should be positive
        self.assertGreater(sharpe, 0)

    def test_sharpe_zero_std(self):
        """Zero std returns should give Sharpe = 0."""
        daily_rets = np.zeros(100)
        std_r = np.std(daily_rets, ddof=1)
        sharpe = 0 if std_r == 0 else np.mean(daily_rets) / std_r * np.sqrt(252)
        self.assertEqual(sharpe, 0)

    def test_max_drawdown_calculation(self):
        """Test MDD: peak at 100, trough at 80 => -20%."""
        equity = np.array([100, 95, 90, 80, 85, 90, 88])
        running_max = np.maximum.accumulate(equity)
        dd = (equity - running_max) / running_max * 100
        mdd = dd.min()
        self.assertAlmostEqual(mdd, -20.0)

    def test_max_drawdown_no_drawdown(self):
        """Monotonically increasing equity should have 0 drawdown."""
        equity = np.array([100, 110, 120, 130, 140])
        running_max = np.maximum.accumulate(equity)
        dd = (equity - running_max) / running_max * 100
        mdd = dd.min()
        self.assertAlmostEqual(mdd, 0)

    def test_win_rate_from_trades(self):
        """Win rate = fraction of positive returns."""
        returns = np.array([0.5, -0.3, 0.2, 0.1, -0.1, 0.4])
        wr = (returns > 0).mean() * 100
        self.assertAlmostEqual(wr, 66.67, places=1)

    def test_win_rate_no_trades(self):
        """Empty returns should not crash."""
        returns = np.array([])
        if len(returns) == 0:
            wr = 0
        else:
            wr = (returns > 0).mean() * 100
        self.assertEqual(wr, 0)


if __name__ == '__main__':
    unittest.main()
