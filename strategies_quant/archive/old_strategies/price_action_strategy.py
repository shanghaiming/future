try:
    from core.base_strategy import BaseStrategy
except ImportError:
    from core.base_strategy import BaseStrategy
import pandas as pd
import numpy as np
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# 添加价格行为集成路径
workspace_path = "/Users/chengming/.openclaw/workspace"
sys.path.append(workspace_path)

try:
    from strategies.price_action_integration.optimized_integration_engine import OptimizedPriceActionIntegrationEngine
    from strategies.price_action_integration.price_action_rules_integrator import PriceActionRulesIntegrator
    PRICE_ACTION_AVAILABLE = True
    print("✅ 成功导入价格行为策略模块")
except ImportError as e:
    PRICE_ACTION_AVAILABLE = False
    print(f"⚠️  价格行为模块导入失败: {e}")
    print(f"⚠️  将使用简化替代实现")

class PriceActionStrategy(BaseStrategy):
    def __init__(self, data, params):
        super().__init__(data, params)
        
        # 默认参数
        self.engine_config = params.get('engine_config', {})
        self.min_data_points = params.get('min_data_points', 100)
        self.use_multi_stock = params.get('use_multi_stock', False)
        
        # 设置可用性标志
        self.price_action_available = PRICE_ACTION_AVAILABLE
        
        # 初始化引擎
        self.engine = None
        self.rules_integrator = None
        
        if self.price_action_available:
            try:
                self.engine = OptimizedPriceActionIntegrationEngine(self.engine_config)
                self.rules_integrator = PriceActionRulesIntegrator()
                print("✅ 价格行为引擎初始化成功")
            except Exception as e:
                print(f"⚠️  价格行为引擎初始化失败: {e}")
                self.price_action_available = False
        else:
            print("⚠️  价格行为引擎不可用，使用简化模式")
        
    def generate_signals(self):
        """统一的多股票信号生成入口"""
        data = self.data.copy()
        
        # 确保数据有symbol列
        if 'symbol' not in data.columns:
            # 如果没有symbol列，假设是单股票数据，添加默认symbol
            data['symbol'] = 'DEFAULT'
        
        self.signals = []
        current_holding = None  # 当前持有的股票
        
        # 按时间遍历
        unique_times = data.index.unique()
        
        for i, current_time in enumerate(unique_times):
            current_bars = data.loc[current_time]
            
            # 如果只有一只股票，确保格式统一
            if isinstance(current_bars, pd.Series):
                current_bars = pd.DataFrame([current_bars])
            
            # 如果没有持仓，选择最优股票
            if current_holding is None:
                best_stock = self._select_best_stock(current_bars, current_time, data)
                if best_stock:
                    self.signals.append({
                        'timestamp': current_time,
                        'action': 'buy',
                        'symbol': best_stock
                    })
                    current_holding = best_stock
                    print(f"买入 {best_stock}")
            
            # 如果有持仓，检查是否需要卖出
            else:
                should_sell = self._check_sell_signal(current_holding, current_time, data)
                if should_sell:
                    self.signals.append({
                        'timestamp': current_time,
                        'action': 'sell',
                        'symbol': current_holding
                    })
                    print(f"卖出 {current_holding}")
                    current_holding = None
        print(*self.signals, sep='\n')
        return self.signals
    
    def _select_best_stock(self, current_bars, current_time, full_data):
        """
        选择最优股票
        返回评分最高的股票代码，如果没有符合条件的则返回None
        """
        best_score = -float('inf')
        best_stock = None
        
        for _, bar in current_bars.iterrows():
            symbol = bar['symbol']
            
            # 获取该股票的历史数据（当前时间之前）
            symbol_data = full_data[full_data['symbol'] == symbol]
            symbol_data = symbol_data[symbol_data.index <= current_time]
            
            # 计算该股票的评分
            score, should_buy = self._calculate_stock_score(symbol_data, symbol, current_time)
            
            # 更新最优股票
            if should_buy and score > best_score:
                best_score = score
                best_stock = symbol
        
        return best_stock
    
    def _check_sell_signal(self, symbol, current_time, full_data):
        """
        检查持仓股票是否需要卖出
        返回布尔值：True表示需要卖出，False表示继续持有
        """
        # 获取该股票的历史数据（当前时间之前）
        symbol_data = full_data[full_data['symbol'] == symbol]
        symbol_data = symbol_data[symbol_data.index <= current_time]
        
        # 计算该股票的卖出信号
        _, should_sell = self._calculate_stock_score(symbol_data, symbol, current_time, check_sell=True)
        
        return should_sell
    
    def _calculate_stock_score(self, symbol_data, symbol, current_time, check_sell=False):
        """
        计算单只股票的评分和交易信号
        使用价格行为分析引擎
        
        返回: (score, should_trade)
        - score: 股票评分（越高越好）
        - should_trade: 
            - 如果是选股模式(check_sell=False): True表示可以买入
            - 如果是卖出检查模式(check_sell=True): True表示需要卖出
        """
        # 确保有足够的数据
        if len(symbol_data) < self.min_data_points:
            return 0, False
        
        # 复制数据避免修改原数据
        data = symbol_data.copy()
        
        if not self.price_action_available or self.engine is None:
            # 使用简化替代方案（补偿移动平均线）
            return self._simple_cma_score(data, check_sell)
        
        try:
            # 使用价格行为引擎
            self.engine.load_data(data)
            results = self.engine.run_analysis()
            
            # 获取补偿移动平均线结果
            cma_results = results.get('compensated_ma', {})
            if 'cma_values' not in cma_results:
                return self._simple_cma_score(data, check_sell)
            
            cma_values = cma_results['cma_values']
            
            # 获取当前和前一时刻的指标
            if len(data) < 2:
                return 0, False
            
            current_price = data['close'].iloc[-1]
            prev_price = data['close'].iloc[-2]
            current_cma = cma_values[-1]
            prev_cma = cma_values[-2] if len(cma_values) >= 2 else cma_values[-1]
            
            # 计算评分（基于价格与CMA的关系）
            price_above_cma = current_price > current_cma
            cma_trend = current_cma > prev_cma
            
            score = 0
            if price_above_cma:
                score += 10
            else:
                score -= 10
            
            if cma_trend:
                score += 5
            else:
                score -= 5
            
            # 判断交易信号
            if check_sell:
                # 卖出信号：价格下穿CMA或CMA转跌
                should_sell = (prev_price >= prev_cma and current_price < current_cma) or (not cma_trend)
                return score, should_sell
            else:
                # 买入信号：价格上穿CMA且CMA转升
                should_buy = (prev_price <= prev_cma and current_price > current_cma) and cma_trend
                return score, should_buy
                
        except Exception as e:
            print(f"⚠️  价格行为分析失败 {symbol}: {e}")
            # 失败时使用简化方案
            return self._simple_cma_score(data, check_sell)
    
    def _simple_cma_score(self, data, check_sell=False):
        """
        简化版补偿移动平均线评分（备用方案）
        """
        if len(data) < 20:
            return 0, False
        
        # 计算补偿移动平均线（简化版）
        window = 20
        data['cma'] = data['close'].rolling(window).mean()
        
        # 添加趋势调整
        trend = data['close'].rolling(window).std() / data['close'].rolling(window).mean()
        data['cma'] = data['cma'] * (1 + trend * 0.1)
        
        if len(data) < 2:
            return 0, False
        
        current_price = data['close'].iloc[-1]
        prev_price = data['close'].iloc[-2]
        current_cma = data['cma'].iloc[-1]
        prev_cma = data['cma'].iloc[-2]
        
        # 计算评分
        score = (current_price - current_cma) / current_cma * 100
        
        if check_sell:
            # 卖出：价格下穿CMA
            should_sell = prev_price >= prev_cma and current_price < current_cma
            return score, should_sell
        else:
            # 买入：价格上穿CMA
            should_buy = prev_price <= prev_cma and current_price > current_cma
            return score, should_buy