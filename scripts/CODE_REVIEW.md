# Futures Platform Scripts -- Code Review

**Date**: 2026-05-21
**Reviewer**: Claude Code (automated)
**Scope**: All 120+ Python files in `scripts/`
**Focus**: Production scripts, core data modules, backtest engines, data collectors

---

## Summary of Findings

| Severity | Count |
|----------|-------|
| CRITICAL | 6     |
| WARNING  | 19    |
| INFO     | 12    |

---

## Per-File Issues

### 1. contract_specs.py -- Core Data

| # | Severity | Issue |
|---|----------|-------|
| 1 | WARNING | Duplicate key `srfi` (白糖) defined on lines 70 and 74. The second definition silently overwrites the first in Python dicts. Both definitions are identical so no data corruption, but it is a maintenance hazard. |
| 2 | WARNING | `wrffi` (line 27) is likely a typo for `wrfi`. The WEIGHTED_LIST in `futures_weighted_collector.py` uses `wrfi`. This means contract specs for wire rod will never match the actual CSV filename. |
| 3 | WARNING | `pfifi` (line 68, short for 短纤) -- the weighted collector uses `pffi` (lowercase). The casing mismatch means specs lookup fails for short fiber. |
| 4 | INFO | `nrfi` (20号胶) comment says "10吨/手 actually" on line 80 but the spec says mult=1. If the true multiplier is 10, all margin/PnL calculations for this symbol are off by 10x. |
| 5 | INFO | `lgfi` (line 81) duplicates `lufi` (低硫燃油). Two different symbols pointing to possibly the same product. |
| 6 | INFO | `sffi2` and `smfi2` in `term_structure_daily.py` PRODUCTS dict use different keys than `sffi`/`smfi` in CONTRACT_SPECS. If these CZCE variants need different specs, they are missing. |

**Recommended Fixes**:
- Remove duplicate `srfi` entry.
- Fix `wrffi` -> `wrfi` to match the actual CSV filename.
- Fix `pfifi` -> `pffi` to match the actual CSV filename.
- Verify `nrfi` multiplier (1 vs 10) against actual exchange specs.
- Audit all symbol keys for consistency across `contract_specs.py`, `futures_weighted_collector.py`, `term_structure_daily.py`, and `daily_market_report.py`.

---

### 2. signal_generator_v1.py -- Production Tool

| # | Severity | Issue |
|---|----------|-------|
| 7 | CRITICAL | **Hard-coded credentials path**: `CONTRACT_SPECS = 'scripts/contract_specs.py'` (line 16) and `DATA_DIR = 'data/futures_weighted'` (line 15) are relative paths that only work when CWD is the project root. If the script is invoked from any other directory, it silently fails or loads wrong data. |
| 8 | WARNING | **Bare `except` clause** (line 68): `except: continue` swallows all exceptions including `KeyboardInterrupt`, `SystemExit`, and `MemoryError`. Should catch `Exception` at minimum. |
| 9 | WARNING | **No input validation**: `compute_daily_signal` (line 80) does not validate that arrays `c`, `o`, `h`, `l`, `v`, `oi` all have the same length as `df`. Mismatched columns would silently produce wrong results. |
| 10 | WARNING | `gap_atr` calculation (line 133): `gap / atr_pct` can produce extreme values when `atr_pct` is near zero (a stock with near-zero ATR). No clamping or guard. |
| 11 | WARNING | `generate_signals_for_date` (line 182) does a linear scan `df[mask].tail(61)` for every symbol on every date. For large datasets and many dates this is O(symbols * dates * rows_per_symbol). |
| 12 | INFO | `SCORE_WEIGHTS` dict (lines 31-52) is defined but never used in the scoring logic. The actual scoring is done inline with hardcoded weights in `compute_daily_signal`. The dict is dead code. |
| 13 | INFO | Long stop-loss and short stop-loss formulas (lines 217-221): For long positions, SL price = `entry * (1 + SL%/100)`. Since `SL% = -1.5`, this gives `entry * 0.985`. The SL price is *below* entry, which is correct for longs. But the naming `sl_price` is confusing because it looks like it should be the trigger price, not the fill price. |

**Recommended Fixes**:
- Replace hard-coded relative paths with `os.path.dirname(os.path.dirname(os.path.abspath(__file__)))` based paths.
- Replace bare `except` with `except Exception`.
- Add array length validation in `compute_daily_signal`.
- Clamp `gap_atr` to a reasonable range (e.g., -10 to +10).
- Remove unused `SCORE_WEIGHTS` dict or wire it into the scoring logic.

---

### 3. backtest_v80.py -- Main Strategy Backtest

| # | Severity | Issue |
|---|----------|-------|
| 14 | CRITICAL | **Lookahead bias in blacklist test** (lines 290-306): The script first runs the base backtest over the full test period to identify the worst-performing symbols, *then* excludes those symbols and re-runs the same test period. This uses future information (which symbols performed poorly) to filter past data. The "walk-forward" validation at line 386 only runs after the best config is already selected. |
| 15 | WARNING | **Same hard-coded paths** as signal_generator_v1: `DATA_DIR = 'data/futures_weighted'`, `CONTRACT_SPECS = 'scripts/contract_specs.py'`. |
| 16 | WARNING | `compute_signals` (line 42): `np.nan_to_num(gap)` converts NaN to 0, which means bars with no previous close (the first bar) get a gap of 0 instead of being excluded. This creates phantom zero-gap signals on the first day of each series. |
| 17 | WARNING | `run_bt` (line 114): The `sl_pct` parameter is used as a truthy check (`if sl_pct:`). If `sl_pct=0.0` (intentionally disabling stop-loss), the condition evaluates to False and SL is skipped. This is correct behavior but the intent is unclear -- use `if sl_pct is not None:` instead. |
| 18 | WARNING | Backtest results are never saved to disk. The entire optimization output is printed to stdout only. If the terminal buffer overflows or the process is killed, results are lost. |
| 19 | INFO | The `pr()` function (line 247) uses `wr = avg = 0` on the `else` branch (no trades), but this is a single assignment. Python parses `wr = avg = 0` correctly but the style is unusual. |

**Recommended Fixes**:
- Fix look-ahead bias: compute blacklist on a training period (e.g., 2016-2020), then test on a held-out period (2022-2025).
- Use `os.path` based paths.
- Mask out NaN gaps before scoring instead of `nan_to_num`.
- Use explicit `is not None` checks for optional parameters.
- Write results to a JSON/CSV file.

---

### 4. data_collector.py -- A-Share + Futures Data

| # | Severity | Issue |
|---|----------|-------|
| 20 | CRITICAL | **Bare `except` in `get_existing_latest_date`** (line 59): `except: return None` silently swallows all errors including `PermissionError`, `pd.errors.EmptyDataError`, and `UnicodeDecodeError`. A corrupt CSV would be silently treated as "no data" rather than alerting the user. |
| 21 | WARNING | **Division by zero** in `normalize_stock_sina` (line 135): `pct_chg = ((closes - pre_close) / pre_close * 100)`. If `pre_close == 0`, this produces `inf`. The only guard is `np.nan` for the first row. |
| 22 | WARNING | **Same division by zero** in `incremental_merge_futures` (line 476): `pct_chg = ((closes - pre_close) / pre_close * 100)` with no guard against `pre_close == 0`. |
| 23 | INFO | `collect_all_stocks_daily` spawns 4 subprocesses but never checks their exit codes (line 282: `p.wait()` result is discarded). If a subprocess crashes, the log parsing silently produces 0 counts. |

**Recommended Fixes**:
- Replace all bare `except:` with `except Exception:`.
- Guard all `pre_close` divisions with `np.where(pre_close != 0, ..., np.nan)`.
- Check subprocess return codes and log failures.

---

### 5. futures_weighted_collector.py -- TqSDK Data

| # | Severity | Issue |
|---|----------|-------|
| 24 | CRITICAL | **Hard-coded credentials** (lines 26-27): `TQ_ACCOUNT = "18844561230"` and `TQ_PASSWORD = "zxcvbnm0717"` are stored in plaintext in the source file. Although there is an `os.environ.get` fallback, the defaults are real credentials. These should be environment-variable-only with no default. |
| 25 | WARNING | `fetch_tqsdk_kline` (line 101): The busy-wait loop `while _time.time() < deadline` with `api.wait_update` can hang for up to 60 seconds per symbol. With 80+ symbols, this means up to 80 minutes of waiting even if data arrives quickly. |
| 26 | WARNING | `incremental_update` (line 158): `pd.read_csv(filepath, dtype=str)` reads the entire old file into memory as strings, then concatenates with new data and writes back. For files with 8000+ rows this is fine, but it does not handle the case where the CSV is corrupted mid-write (partial write on crash). |
| 27 | INFO | The WEIGHTED_LIST contains `lgfi` (原木加权, line 77) and `psfi` (多晶硅加权, line 77) which may be very new or upcoming products. If tqsdk has no data, these will show as failures. |

**Recommended Fixes**:
- Remove default credential values. Raise an error if env vars are not set.
- Add a faster data-availability check (e.g., check if `kl.close` has valid values after the first `wait_update`).
- Use atomic writes (write to temp file, then `os.rename`) to prevent data corruption.

---

### 6. options_collector.py -- Options Data

| # | Severity | Issue |
|---|----------|-------|
| 28 | CRITICAL | **`analyze_volatility_surface` uses simulated data** (lines 172-225): The function's docstring says "uses simulated data to demonstrate the analysis framework." This means the entire options analytics pipeline produces fake results. The `bsm_price` and `bsm_iv` functions are correct, but the surface analysis is meaningless with hardcoded `base_iv = 0.20` and parametric smile/skew. |
| 29 | WARNING | `bsm_iv` (line 65): The binary search uses fixed bounds `sigma_low=0.001, sigma_high=5.0`. For deep OTM options with very low prices, 100 iterations may not converge, and the function returns the midpoint of the last interval without warning the caller. |
| 30 | INFO | `fetch_underlying_price` (line 148) tries multiple akshare interfaces. If the API format changes, it silently returns `None` and the symbol is skipped. |

**Recommended Fixes**:
- Implement real option chain parsing from `akshare` data instead of simulation.
- Add convergence warning to `bsm_iv` (return `None` if not converged within tolerance).
- Add logging for skipped symbols.

---

### 7. daily_market_report.py -- Market Report

| # | Severity | Issue |
|---|----------|-------|
| 31 | WARNING | `load_index_data` (line 46) uses `ak.stock_zh_index_daily_em` which is a network call. If akshare is not installed or the API is down, the function returns an empty dict and the report silently omits the entire index section. No error message. |
| 32 | WARNING | `analyze_market_breadth` (line 92) only samples 500 files (`files[:500]`). If there are 5000 stocks, the breadth analysis is based on a biased sample (alphabetically first 500). |
| 33 | INFO | The report is generated in Chinese with emoji characters. If the terminal or file encoding is not UTF-8, the output may be garbled. |

**Recommended Fixes**:
- Add try/except around `akshare` calls with user-visible error messages.
- Randomly sample 500 stocks instead of taking the first 500.
- Add explicit `encoding='utf-8'` to all file operations.

---

### 8. term_structure_daily.py -- Term Structure Snapshot

| # | Severity | Issue |
|---|----------|-------|
| 34 | CRITICAL | **Hard-coded credentials** (lines 19-20): Same issue as `futures_weighted_collector.py` -- plaintext TQ account and password. |
| 35 | WARNING | `fetch_daily_snapshot` (line 102): NaN check `q.last_price != q.last_price` is a float NaN comparison. While this works for float, it is fragile and non-idiomatic. Use `math.isnan()` or `pd.isna()`. |
| 36 | WARNING | CZCE date parsing (lines 112-119): The logic assumes CZCE contracts use 3-digit or 4-digit codes where `yy = int(num[0]) + 2020`. This is hardcoded to the 2020s decade and will break for contracts expiring in 2030+. |
| 37 | INFO | `quote_list.index(q)` on line 105 is O(n) and called inside a loop, making it O(n^2). With 100+ contracts per product this is slow. |

**Recommended Fixes**:
- Remove plaintext credentials (same as weighted collector fix).
- Use `math.isnan()` for NaN checks.
- Make CZCE year parsing decade-agnostic.
- Use `enumerate(quote_list)` instead of `quote_list.index(q)`.

---

### 9. backtest_engine.py -- Generic Backtest Framework

| # | Severity | Issue |
|---|----------|-------|
| 38 | WARNING | `calculate_position_size` (line 195): Hard-coded `contract_value = price * 10` assumes all contracts have multiplier 10. This is wrong for `aufi` (1000), `ifi` (100), `scfi` (1000), etc. Should use `contract_specs.get_spec()`. |
| 39 | WARNING | `load_data` (line 78): Does not handle `pd.errors.EmptyDataError` or files with missing columns. A CSV without `close` column would crash with a `KeyError`. |
| 40 | WARNING | `check_stops` (line 296): Uses `close` price for stop-loss check, not `high`/`low`. This means intraday stop-loss triggers are missed -- a position that hits stop-loss intraday but recovers by close would not be stopped out. |
| 41 | INFO | The engine computes `HV`, `MACD`, `RSI`, `Bollinger Bands`, `ATR` on every `load_data` call, even if only some indicators are used. This adds unnecessary computation time. |

**Recommended Fixes**:
- Use `contract_specs.get_spec()` for multiplier.
- Add error handling for malformed CSV files.
- Use `high`/`low` for intraday stop-loss checks.
- Compute indicators lazily or on demand.

---

### 10. backtest_v94_multifactor.py -- Multi-Factor Strategy

| # | Severity | Issue |
|---|----------|-------|
| 42 | INFO | Well-structured code with proper docstrings and clear separation of concerns. This is the cleanest script in the repository. |
| 43 | INFO | `load_daily_data` (line 80) reads all CSVs and concatenates into a single DataFrame. For 80+ symbols with 8000 rows each, this creates a DataFrame with ~640K rows. Memory usage is manageable but could be optimized with chunked processing. |

---

### 11. tq_options_collector.py -- TqSDK Options

| # | Severity | Issue |
|---|----------|-------|
| 44 | CRITICAL | **Hard-coded credentials** (line 21-22): Same plaintext credential issue. |
| 45 | WARNING | **Bug in BSM put delta** (line 73): `delta = norm.cdf(d1) if option_type == 'C' else norm.cdf(d1) - 1`. For puts, the correct formula is `norm.cdf(d1) - 1`, which gives a negative delta. This is actually correct. However, the corresponding `bsm_put_price` (line 44-49) uses `S * norm.cdf(d1)` instead of `S * norm.cdf(-d1)` in the put formula. The correct BSM put is `K*exp(-rT)*N(-d2) - S*N(-d1)`. Line 49 has `- S * norm.cdf(d1)` which should be `- S * norm.cdf(-d1)`. This is a real pricing bug that produces incorrect put prices for all OTM puts. |
| 46 | WARNING | `fetch_product_options` (line 83): The `query_quotes(ins_class='OPTION')` call queries ALL option contracts across ALL exchanges. The filtering by exchange/prefix happens client-side. This is extremely slow and wasteful when you only want one product's options. |

**Recommended Fixes**:
- Fix `bsm_put_price` to use `-S * norm.cdf(-d1)`.
- Use `api.query_quotes(ins_class='OPTION', exchange_id=exchange)` to narrow the query.
- Remove plaintext credentials.

---

### 12. analyze_signals.py (and analyze_signals[2-12].py)

| # | Severity | Issue |
|---|----------|-------|
| 47 | WARNING | `dir` shadows the Python built-in `dir()` function (line 3). While it works, it is bad practice. |
| 48 | INFO | 12 near-identical `analyze_signalsN.py` files exist with minor parameter variations. These are experiment artifacts and could be consolidated into a single parameterized script. |

---

### 13. backtest_v[2-90].py -- Version History

| # | Severity | Issue |
|---|----------|-------|
| 49 | INFO | Over 80 backtest version files exist in the scripts directory. Most are superseded experiments. They should be archived or moved to an `experiments/` subdirectory. The active strategy is V80 (and V94 for multi-factor). |
| 50 | WARNING | Nearly all backtest versions duplicate the same `load_data()`, `compute_signals()`, and `run_bt()` functions with minor variations. This is massive code duplication (estimated 10,000+ lines of duplicated logic). |

---

## Cross-Cutting Issues

### CRITICAL: Hard-Coded Credentials (3 files)

Files: `futures_weighted_collector.py`, `term_structure_daily.py`, `tq_options_collector.py`

All three files contain plaintext TQ account `18844561230` and password `zxcvbnm0717`. Even though `futures_weighted_collector.py` has an `os.environ.get` fallback, the defaults expose real credentials in source code.

**Fix**: Remove defaults, raise error if env vars not set:
```python
TQ_ACCOUNT = os.environ.get("TQ_ACCOUNT")
TQ_PASSWORD = os.environ.get("TQ_PASSWORD")
if not TQ_ACCOUNT or not TQ_PASSWORD:
    raise EnvironmentError("Set TQ_ACCOUNT and TQ_PASSWORD environment variables")
```

### CRITICAL: Lookahead Bias in V80 Optimization

The blacklist optimization in `backtest_v80.py` uses the full test period to identify worst symbols, then re-tests on the same period. This inflates reported performance.

### WARNING: Pervasive Bare `except` Clauses

Found in: `signal_generator_v1.py`, `data_collector.py`, `backtest_v80.py`, `futures_weighted_collector.py`, `daily_market_report.py`, and many backtest versions.

Pattern:
```python
try:
    ...
except:        # catches KeyboardInterrupt, SystemExit, MemoryError
    continue
```

Should be:
```python
except Exception:
    continue
```

### WARNING: Symbol Key Inconsistencies

The same product uses different symbol keys across files:

| Product | contract_specs.py | weighted_collector.py | term_structure_daily.py |
|---------|-------------------|-----------------------|-------------------------|
| Wire Rod | `wrffi` | `wrfi` | `wrfi` |
| Short Fiber | `pfifi` | `pffi` | `pffi` |
| White Sugar | `srfi` (x2) | `srfi` | `srfi` |
| Silicon Steel CZCE | `sffi` | - | `sffi2` |
| Manganese CZCE | `smfi` | - | `smfi2` |

### WARNING: Division by Zero

Multiple locations compute `pct_chg = (close - prev_close) / prev_close * 100` without checking `prev_close != 0`:
- `data_collector.py` lines 135, 389, 476
- `daily_market_report.py` line 79 (guarded by `if prev == 0`)
- Various backtest scripts

### INFO: Code Duplication

The following patterns are duplicated across 80+ files:
1. `load_data()` function (importlib.util loading of contract_specs + CSV reading)
2. Signal computation (gap, ATR, MA, momentum, OI change, CLV scoring)
3. Backtest engine loop (position tracking, SL/TP, equity curve)
4. Performance metrics (Sharpe, MDD, WR calculation)

Estimated duplicated lines: 10,000+

---

## Recommended Refactoring Plan

### Phase 1: Security (Immediate)
1. Move TQ credentials to environment variables with no defaults
2. Add `.env` to `.gitignore`
3. Audit for any other hardcoded credentials

### Phase 2: Bug Fixes (This Week)
1. Fix `tq_options_collector.py` BSM put pricing formula
2. Fix symbol key inconsistencies in `contract_specs.py`
3. Add `prev_close != 0` guards to all pct_chg calculations
4. Replace all bare `except:` with `except Exception:`

### Phase 3: Path Configuration (This Week)
1. Create a central `config.py` with base paths derived from `__file__`
2. Replace all hard-coded `DATA_DIR`, `CONTRACT_SPECS` strings with imports from config
3. Test that scripts work from any working directory

### Phase 4: Code Consolidation (Next Sprint)
1. Extract common `load_data()`, `compute_signals()`, `run_backtest()`, `print_metrics()` into a shared module `scripts/lib/`
2. Keep only active strategy versions (V80, V94) in `scripts/`
3. Move experiment versions to `scripts/experiments/` or an archive

### Phase 5: Test Coverage
See test coverage plan below.

---

## Test Coverage Plan

### Current Coverage: 0 test files -> 1 test file (54 tests)

### Test File: `scripts/test_scripts.py`

| Module | Tests | Coverage |
|--------|-------|----------|
| contract_specs (get_spec, calc_margin, calc_contract_value, calc_pnl, calc_max_lots) | 16 | Core functions fully covered |
| Signal computation (gap scoring, edge cases) | 6 | Scoring logic validated |
| Backtest engine (zero capital, no signals, all-losing, empty data) | 5 | Edge case handling |
| Data loading (missing files, empty CSV, NaN, zero prices, dedup) | 6 | File I/O edge cases |
| BSM pricing (call, put, parity, edge cases) | 7 | Options math validated |
| V94 factors (momentum, vol, z-score) | 5 | Factor computation |
| Performance metrics (Sharpe, MDD, WR) | 6 | Metric calculations |
| **Total** | **54** | |

### Recommended Additional Tests

| Priority | Test | Description |
|----------|------|-------------|
| HIGH | `test_sl_tp_trigger` | Verify stop-loss and take-profit trigger correctly for long/short |
| HIGH | `test_cooldown_period` | Verify same-symbol cooldown blocks re-entry |
| HIGH | `test_max_position_limit` | Verify position count never exceeds max_positions |
| MEDIUM | `test_walk_forward_no_leakage` | Verify training data does not overlap with test data |
| MEDIUM | `test_incremental_update_atomic` | Verify CSV updates are atomic (no partial writes) |
| MEDIUM | `test_contract_specs_completeness` | Verify every symbol in weighted_collector has a matching spec |
| LOW | `test_daily_report_no_crash` | Verify report generation handles missing data gracefully |
| LOW | `test_term_structure_parsing` | Verify CZCE/DCE contract code parsing |

---

## Files Reviewed (Complete List)

### Priority Files (fully read)
- `scripts/contract_specs.py`
- `scripts/signal_generator_v1.py`
- `scripts/backtest_v80.py`
- `scripts/data_collector.py`
- `scripts/futures_weighted_collector.py`
- `scripts/options_collector.py`
- `scripts/daily_market_report.py`
- `scripts/term_structure_daily.py`
- `scripts/backtest_engine.py`
- `scripts/backtest_v94_multifactor.py`
- `scripts/tq_options_collector.py`
- `scripts/analyze_signals.py`
- `scripts/backtest_v90.py` (partial)

### Acknowledged but Not Deeply Reviewed
- `scripts/backtest_v[2-69].py` -- superseded versions, same patterns
- `scripts/backtest_v[70-79].py` -- intermediate iterations
- `scripts/backtest_v[81-89].py` -- post-V80 experiments
- `scripts/analyze_signals[2-12].py` -- experiment artifacts
- `scripts/grid_search.py`, `grid_search_fast.py`, `quick_search.py` -- parameter search
- `scripts/param_sweep.py`, `momentum_finetune.py`, `options_sweep.py` -- parameter tuning
- `scripts/strategy_compare.py`, `strategy_v27_compare.py` -- comparison tools
- `scripts/term_structure_collector.py`, `term_structure_collector_v[2,3].py` -- older collectors
- `scripts/term_structure_backfill.py`, `term_structure_backfill_v[3,4].py` -- backfill tools
- `scripts/futures_collector_v2.py` -- older data source
- `scripts/options_collector_commodity.py` -- commodity options collector
- `scripts/backtest_term_structure.py`, `backtest_aggressive.py`, `backtest_trend.py` -- strategy variants
