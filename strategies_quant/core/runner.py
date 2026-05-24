"""
统一回测运行器 - 策略发现、加载、执行
"""
import sys
import os
import pandas as pd
import importlib
import io

from .base_strategy import BaseStrategy
from .backtest_engine import BacktestEngine


class BacktestRunner:
    def __init__(self, load_all=False):
        """初始化回测运行器
        Args:
            load_all: 是否立即加载所有策略（默认False，按需加载）
        """
        self.strategies = {}
        self.strategies_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'strategies')

        if load_all:
            self.strategies = self._discover_strategies()

    def _discover_strategies(self, strategy_name_param=None):
        """发现策略
        Args:
            strategy_name_param: 可选，指定要加载的策略名称
        Returns:
            策略字典 {策略名: 策略类}
        """
        strategies = {}

        if not os.path.exists(self.strategies_dir):
            return strategies

        # 确定要处理的文件列表
        if strategy_name_param:
            filename = f"{strategy_name_param}.py"
            if not os.path.exists(os.path.join(self.strategies_dir, filename)):
                print(f"策略文件不存在: {filename}")
                return strategies
            file_list = [filename]
        else:
            file_list = []
            for filename in os.listdir(self.strategies_dir):
                if filename.endswith('.py') and filename not in ['__init__.py', 'base_strategy.py']:
                    exclude_patterns = [
                        'test_', 'fix_', 'check_', 'analyzer', 'case_study',
                        'price_action_talib_backtest_system',
                        'price_action_reversals_reversal_',
                        'price_action_ranges_market_structure_deep_analyzer',
                        'batch_', 'debug_', 'sample_', 'example_', 'tool_', 'util_', 'helper_'
                    ]
                    should_exclude = False
                    for pattern in exclude_patterns:
                        if pattern in filename:
                            should_exclude = True
                            break

                    if not should_exclude:
                        file_list.append(filename)

        for filename in file_list:
            strategy_name = filename[:-3]

            try:
                module_path = f'strategies.{strategy_name}'
                spec = importlib.util.spec_from_file_location(
                    module_path,
                    os.path.join(self.strategies_dir, filename)
                )
                if spec is None:
                    continue

                module = importlib.util.module_from_spec(spec)

                # 保存原始文件描述符，防止被模块覆盖
                real_stdout = sys.stdout
                real_stderr = sys.stderr
                sys.stdout = io.StringIO()
                sys.stderr = io.StringIO()

                try:
                    spec.loader.exec_module(module)
                finally:
                    # 强制恢复，不信任模块可能篡改的 sys.__stdout__
                    sys.stdout = real_stdout
                    sys.stderr = real_stderr

                # 查找策略类
                found_class = None
                for attr_name in dir(module):
                    if attr_name.startswith('_'):
                        continue
                    try:
                        attr = getattr(module, attr_name)
                        if isinstance(attr, type) and attr != BaseStrategy:
                            try:
                                if issubclass(attr, BaseStrategy):
                                    found_class = attr
                                    break
                            except TypeError:
                                continue
                    except Exception:
                        continue

                if found_class is None:
                    for attr_name, attr_value in module.__dict__.items():
                        if attr_name.startswith('_'):
                            continue
                        if isinstance(attr_value, type) and attr_value != BaseStrategy:
                            try:
                                if issubclass(attr_value, BaseStrategy):
                                    found_class = attr_value
                                    break
                            except TypeError:
                                continue

                if found_class is None:
                    common_class_names = [
                        f"{strategy_name.replace('_', ' ').title().replace(' ', '')}Strategy",
                        f"{strategy_name}Strategy",
                        f"{strategy_name}".title().replace('_', ''),
                    ]
                    for class_name in common_class_names:
                        if hasattr(module, class_name):
                            attr = getattr(module, class_name)
                            if isinstance(attr, type) and attr != BaseStrategy:
                                try:
                                    if issubclass(attr, BaseStrategy):
                                        found_class = attr
                                        break
                                except TypeError:
                                    continue

                if found_class is not None:
                    strategies[filename[:-3]] = found_class
                    if strategy_name_param is not None:
                        print(f"加载策略: {filename[:-3]}")
                else:
                    print(f"策略 {filename[:-3]} 未找到继承BaseStrategy的类")

            except BaseException as e:
                # 捕获 SystemExit 等非 Exception 异常
                sys.stdout = real_stdout if 'real_stdout' in dir() else sys.__stdout__
                sys.stderr = real_stderr if 'real_stderr' in dir() else sys.__stderr__
                error_msg = str(e)
                if "No module named" in error_msg:
                    print(f"策略 {filename[:-3]} 依赖缺失: {error_msg}")
                else:
                    print(f"策略 {filename[:-3]} 加载失败: {error_msg}")

        return strategies

    def load_strategy(self, strategy_name):
        """加载单个策略"""
        if strategy_name in self.strategies:
            return self.strategies[strategy_name]

        strategies = self._discover_strategies(strategy_name_param=strategy_name)
        if strategy_name in strategies:
            self.strategies[strategy_name] = strategies[strategy_name]
            return self.strategies[strategy_name]
        else:
            print(f"策略加载失败: {strategy_name}")
            return None

    def load_all_strategies(self):
        """加载所有策略"""
        self.strategies = self._discover_strategies()
        print(f"已加载 {len(self.strategies)} 个策略")
        return self.strategies

    def list_strategies(self):
        """列出所有可用策略"""
        if not self.strategies:
            self.load_all_strategies()
        return list(self.strategies.keys())

    def run(self, strategy_name, data, params=None):
        """运行策略回测"""
        strategy_cls = self.load_strategy(strategy_name)
        if strategy_cls is None:
            raise ValueError(f"策略不存在或加载失败: {strategy_name}")

        params = params or {}

        initial_cash = params.get('initial_cash', 100000)

        try:
            print(f"使用{strategy_name}策略: 数据 {len(data)} 行")

            engine = BacktestEngine(data, strategy_cls, initial_cash=initial_cash)
            results = engine.run_backtest(params)

            results['strategy_name'] = strategy_name
            results['initial_cash'] = initial_cash

            return results

        except Exception as e:
            print(f"回测引擎执行失败: {e}")
            import traceback
            traceback.print_exc()

            return {
                'strategy_name': strategy_name,
                'initial_cash': initial_cash,
                'signals': [],
                'error': str(e)
            }


if __name__ == "__main__":
    runner = BacktestRunner()
    strategies = runner.list_strategies()
    print(f"发现 {len(strategies)} 个策略:")
    for s in strategies:
        print(f"  - {s}")
