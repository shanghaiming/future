#!/usr/bin/env python3
"""
批量回测 quant_trade 策略在期货数据上的表现
"""
import os, sys, importlib, traceback, time
sys.stdout.reconfigure(line_buffering=True)
import pandas as pd
import numpy as np

# 确保能找到 core 和 strategies
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, 'core'))
sys.path.insert(0, os.path.join(BASE, 'strategies'))

from core.data_loader import load_stock_data, list_available_symbols
from core.backtest_engine import BacktestEngine
from core.base_strategy import BaseStrategy

# 选几个代表品种测试
TEST_SYMBOLS = ['rbfi', 'safi', 'ifi', 'mfi', 'aufi']

# 需要跳过的文件 (非策略或会报错的)
SKIP = {
    '__init__', 'base_strategy', 'backtest_engine', 'data_loader',
    'performance', 'runner', 'market_regime', 'csv_auto_select',
    'csv_auto_select_adapter', 'csv_price_action_adapter', 'fix_imports',
    'check_reason', 'structure_test', 'structure_validator', 'test',
    'simple_test', 'import_test', 'test_single_strategy_signals',
    'batch_test_signals', 'simple_batch_test', 'simple_import_test',
    'index', 'kline', 'visual', 'var', 'limit_up_board', 'rush_buy',
    'spike_bake', 'stdg', 'reflect_wave', 'peak', 'peak-ex', 'rectangle',
    'future_filter', 'future_filtet_v2', 'stock_filter', 'stock_filter_v2',
    'analyze_signal_quality', 'analyze_strategy_selectivity',
    'position_sizing_adjuster', 'performance_evaluator', 'exit_strategy_optimizer',
    'common_errors_avoidance_system', 'continuous_improvement_system',
    'psychological_discipline_manager', 'psychological_training_system',
    'trading_log_analyzer', 'trading_plan_creator', 'trading_system_integrator',
    'system_summary_and_improvement', 'trend_channel_analyzer',
    'market_structure_deep_analyzer', 'market_structure_identifier',
    'multi_timeframe_coordinator', 'case_study_analyzer',
    # price_action 子模块
    'price_action_ranges_advanced_entry_techniques', 'price_action_ranges_advanced_risk_management_system',
    'price_action_ranges_case_study_analyzer', 'price_action_ranges_common_errors_avoidance_system',
    'price_action_ranges_continuous_improvement_system', 'price_action_ranges_exit_strategy_optimizer',
    'price_action_ranges_market_structure_deep_analyzer', 'price_action_ranges_market_structure_identifier',
    'price_action_ranges_multi_timeframe_coordinator', 'price_action_ranges_performance_evaluator',
    'price_action_ranges_position_sizing_adjuster', 'price_action_ranges_price_action_strategy_adapter',
    'price_action_ranges_psychological_discipline_manager', 'price_action_ranges_psychological_training_system',
    'price_action_ranges_system_summary_and_improvement', 'price_action_ranges_trading_log_analyzer',
    'price_action_ranges_trading_plan_creator', 'price_action_ranges_trading_system_integrator',
    'price_action_ranges_trend_channel_analyzer',
    'price_action_reversals_multi_timeframe_reversal_system', 'price_action_reversals_reversal_case_studies',
    'price_action_reversals_reversal_confirmation_system', 'price_action_reversals_reversal_pattern_recognition',
    'price_action_reversals_reversal_position_management', 'price_action_reversals_reversal_risk_management',
    'price_action_reversals_reversal_system_integration', 'price_action_reversals_reversal_timing_system',
    'price_action_reversals_reversal_trading_basics', 'price_action_reversals_reversal_trading_psychology',
    'price_action_strategy_adapter',
    # v* 和 alpha* 中非策略的
    'alpha_factor_cache',
    'mindgo_api', 'paper_trade', 'cli', 'manage', 'diagnose',
    'backtest_factors', 'benchmark_strategies', 'rotation_backtest',
    'stock_select', 'parse_news', 'signal_analyzer',
    'regenerate_signals', 'regenerate_signals_all', 'regenerate_signals_final',
    'regenerate_signals_merge', 'regenerate_signals_missing',
    'regenerate_signals_original', 'regenerate_signals_parallel',
    'regenerate_signals_v71', 'strict_test', 'test_all', 'test_platform',
    'test_runner_fix', 'test_strategy_fixed', 'test_strategy_import',
    'check_strategy_inheritance', 'simple_strategy_backtest',
    'backtest_integration_test', 'extracted_v15_6yr_backtest',
    'extracted_v15_7_fixed_300',
}


def discover_strategies():
    """发现所有可运行的策略文件"""
    strategies = []
    for f in sorted(os.listdir(BASE)):
        if not f.endswith('.py'):
            continue
        name = f.replace('.py', '')
        if name in SKIP:
            continue
        if name.startswith('price_action_ranges_') or name.startswith('price_action_reversals_'):
            continue
        strategies.append(name)
    return strategies


def find_strategy_class(module):
    """在模块中找到 BaseStrategy 子类"""
    import core.base_strategy as bs
    for attr_name in dir(module):
        try:
            attr = getattr(module, attr_name)
        except Exception:
            continue
        if (isinstance(attr, type)
            and issubclass(attr, bs.BaseStrategy)
            and attr is not bs.BaseStrategy
            and attr.__module__ == module.__name__):
            return attr
    return None


def run_single_strategy(name, data):
    """运行单个策略, 静默所有策略内部print"""
    import io
    import contextlib

    try:
        module = importlib.import_module(name)
    except Exception as e:
        return None, f'import error: {str(e)[:80]}'

    cls = find_strategy_class(module)
    if cls is None:
        return None, 'no BaseStrategy subclass found'

    try:
        # 静默所有print输出
        f = io.StringIO()
        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            engine = BacktestEngine(data, cls, initial_cash=1e6, commission=0.0003)
            result = engine.run_backtest({})
        if result:
            summary = result.get('summary', {})
            risk = result.get('risk_return', {})
            trades_info = result.get('trades', {})
            dd = result.get('drawdown', {})
            return {
                'name': name,
                'total_return': summary.get('total_return', 0),
                'annualized_return': summary.get('annualized_return', 0),
                'sharpe_ratio': risk.get('sharpe_ratio', 0),
                'max_drawdown': dd.get('max_drawdown', 0),
                'win_rate': trades_info.get('win_rate', 0),
                'total_trades': summary.get('trade_count', 0),
            }, None
        return None, 'no result'
    except Exception as e:
        return None, f'run error: {str(e)[:80]}'


def main():
    print("=" * 80)
    print("批量回测 quant_trade 策略 on 期货数据")
    print("=" * 80)

    # 加载数据
    symbols = list_available_symbols('daily')
    print(f"\n可用品种: {len(symbols)}")

    # 用 rbfi (螺纹钢) 作为主要测试品种
    data = load_stock_data('rbfi')
    data = data[data['close'].notna() & (data['close'] > 0)]
    print(f"测试数据: rbfi, {len(data)} rows")
    if 'symbol' not in data.columns:
        data['symbol'] = 'rbfi'

    strategies = discover_strategies()
    print(f"\n发现 {len(strategies)} 个策略文件")
    print("-" * 80)

    results = []
    errors = []

    t0 = time.time()
    for i, name in enumerate(strategies):
        result, err = run_single_strategy(name, data)
        if result:
            results.append(result)
            ret = result['total_return']
            sharpe = result['sharpe_ratio']
            trades = result['total_trades']
            print(f"  [{i+1:3d}/{len(strategies)}] {name:<45s} ret={ret:>+8.2%}  sharpe={sharpe:>6.2f}  trades={trades:>4}")
        elif err:
            errors.append((name, err))

    elapsed = time.time() - t0

    # 排序输出
    print(f"\n{'='*80}")
    print(f"回测完成: {len(results)} 成功, {len(errors)} 失败, {elapsed:.1f}s")
    print(f"{'='*80}")

    if results:
        results.sort(key=lambda x: x.get('sharpe_ratio', 0) or 0, reverse=True)
        print(f"\n{'排名':<4} {'策略':<45} {'总收益':>8} {'年化':>8} {'Sharpe':>7} {'回撤':>8} {'胜率':>6} {'交易数':>6}")
        print("-" * 100)
        for i, r in enumerate(results[:50]):
            print(f"{i+1:<4} {r['name']:<45} {r['total_return']:>+7.1%} "
                  f"{r['annualized_return']:>+7.1%} {r['sharpe_ratio']:>7.2f} "
                  f"{r['max_drawdown']:>7.1%} {r['win_rate']:>5.1%} {r['total_trades']:>6}")

        # 保存结果
        df = pd.DataFrame(results)
        out_path = os.path.join(BASE, '..', 'backtest_results', 'quant_strategies_results.csv')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"\n结果已保存: {out_path}")

    if errors:
        print(f"\n失败策略 ({len(errors)}):")
        for name, err in errors[:20]:
            print(f"  {name}: {err}")


if __name__ == '__main__':
    main()
