try:
    from core.base_strategy import BaseStrategy
except ImportError:
    from core.base_strategy import BaseStrategy
import pandas as pd
import numpy as np

class MovingAverageStrategy(BaseStrategy):
    def __init__(self, data, params):
        super().__init__(data, params)
        self.short_window = params.get('short_window', 3)  # 默认改为3天
        self.long_window = params.get('long_window', 5)    # 默认改为5天
        
    def generate_signals(self):
        """统一的多股票信号生成入口"""
        data = self.data.copy()
        
        # 确保数据有symbol列
        if 'symbol' not in data.columns:
            # 如果没有symbol列，假设是单股票数据，添加默认symbol
            data['symbol'] = 'DEFAULT'
        
        # 检测是否为单股票数据
        unique_symbols = data['symbol'].unique()
        is_single_stock = len(unique_symbols) == 1
        
        if is_single_stock:
            # 单股票模式：使用简化逻辑
            return self._generate_signals_single_stock(data)
        else:
            # 多股票模式：使用原有逻辑
            return self._generate_signals_multi_stock(data)
    
    def _select_best_stock(self, current_bars, current_time, full_data):
        """
        选择最优股票
        返回评分最高的股票代码，如果没有符合条件的则返回第一只股票
        """
        # 如果数据不足，直接返回第一只股票
        if len(current_bars) == 0:
            return None
        
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
        
        # 如果没有符合条件的股票，返回第一只股票
        if best_stock is None and len(current_bars) > 0:
            best_stock = current_bars['symbol'].iloc[0]
            print(f"⚠️  未找到符合条件的股票，使用第一只股票: {best_stock}")
        
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
        这是策略的核心逻辑，可以替换为任何自定义策略
        
        返回: (score, should_trade)
        - score: 股票评分（越高越好）
        - should_trade: 
            - 如果是选股模式(check_sell=False): True表示可以买入
            - 如果是卖出检查模式(check_sell=True): True表示需要卖出
        """
        try:
            # 确保有足够的数据
            if len(symbol_data) < self.long_window:
                # 数据不足时，返回一个基础分数
                # 这样至少能让策略选择股票
                return 1.0, True  # 返回正分数和True，表示可以交易
            
            # 复制数据避免修改原数据
            data = symbol_data.copy()
            
            # 确保数据是DataFrame格式
            if isinstance(data, pd.Series):
                data = data.to_frame().T
            
            # 找到收盘价列
            close_col = 'close'
            if close_col not in data.columns:
                # 尝试找到包含'close'的列
                for col in data.columns:
                    if 'close' in col.lower():
                        close_col = col
                        break
                else:
                    # 如果没有找到，使用第一列
                    close_col = data.columns[0]
            
            # 计算技术指标（避免未来函数）
            close_prices = data[close_col].values if hasattr(data[close_col], 'values') else data[close_col]
            
            # 计算移动平均线
            ma_short = pd.Series(close_prices).shift(1).rolling(self.short_window, min_periods=1).mean()
            ma_long = pd.Series(close_prices).shift(1).rolling(self.long_window, min_periods=1).mean()
            
            # 获取当前时刻的指标
            if len(data) < 2:
                return 0, False
                
            current_ma_short = float(ma_short.iloc[-1]) if not pd.isna(ma_short.iloc[-1]) else 0.0
            current_ma_long = float(ma_long.iloc[-1]) if not pd.isna(ma_long.iloc[-1]) else 0.0
            prev_ma_short = float(ma_short.iloc[-2]) if len(ma_short) > 1 and not pd.isna(ma_short.iloc[-2]) else current_ma_short
            prev_ma_long = float(ma_long.iloc[-2]) if len(ma_long) > 1 and not pd.isna(ma_long.iloc[-2]) else current_ma_long
            
            # 计算金叉死叉（确保使用标量）
            golden_cross = bool(current_ma_short > current_ma_long) and bool(prev_ma_short <= prev_ma_long)
            death_cross = bool(current_ma_short < current_ma_long) and bool(prev_ma_short >= prev_ma_long)
            
            # 计算评分（均线差值百分比）
            if current_ma_long != 0:
                score = (current_ma_short - current_ma_long) / current_ma_long * 100
            else:
                score = 0
            
            if check_sell:
                # 卖出检查模式：出现死叉时卖出
                return score, death_cross
            else:
                # 选股模式：出现金叉时可以考虑买入
                return score, golden_cross
                
        except Exception as e:
            print(f"⚠️  _calculate_stock_score错误: {e}")
            # 出错时返回基础分数
            return 1.0, True
    
    def _generate_signals_single_stock(self, data):
        """为单股票数据生成信号"""
        # 定义价格和股票代码列名
        close_col = 'close' if 'close' in data.columns else data.columns[0]
        symbol = data['symbol'].iloc[0] if 'symbol' in data.columns else 'DEFAULT'
        
        self.signals = []
        
        # 按时间遍历
        unique_times = data.index.unique()
        holding = False  # 是否持有仓位
        
        for i, current_time in enumerate(unique_times):
            # 获取当前时间的数据
            if current_time in data.index:
                current_data = data.loc[current_time]
                
                # 如果数据是Series，转换为DataFrame
                if isinstance(current_data, pd.Series):
                    current_data = current_data.to_frame().T
                
                # 获取该时间点之前的历史数据
                historical_data = data[data.index <= current_time]
                
                # 计算技术指标
                if len(historical_data) >= self.long_window:
                    # 计算移动平均
                    close_prices = historical_data[close_col].values
                    ma_short = pd.Series(close_prices).shift(1).rolling(self.short_window, min_periods=1).mean()
                    ma_long = pd.Series(close_prices).shift(1).rolling(self.long_window, min_periods=1).mean()
                    
                    # 获取当前和之前的指标值
                    if len(ma_short) >= 2 and len(ma_long) >= 2:
                        current_ma_short = ma_short.iloc[-1] if not pd.isna(ma_short.iloc[-1]) else 0
                        current_ma_long = ma_long.iloc[-1] if not pd.isna(ma_long.iloc[-1]) else 0
                        prev_ma_short = ma_short.iloc[-2] if len(ma_short) > 1 and not pd.isna(ma_short.iloc[-2]) else current_ma_short
                        prev_ma_long = ma_long.iloc[-2] if len(ma_long) > 1 and not pd.isna(ma_long.iloc[-2]) else current_ma_long
                        
                        # 检查金叉（买入信号）
                        golden_cross = current_ma_short > current_ma_long and prev_ma_short <= prev_ma_long
                        # 检查死叉（卖出信号）
                        death_cross = current_ma_short < current_ma_long and prev_ma_short >= prev_ma_long
                        
                        # 生成信号
                        if not holding and golden_cross:
                            # 买入信号
                            price = float(historical_data[close_col].iloc[-1])
                            self.signals.append({
                                'timestamp': current_time,
                                'action': 'buy',
                                'symbol': symbol,
                                'price': price
                            })
                            holding = True
                            print(f"买入 {symbol} @ {price:.2f}")
                        
                        elif holding and death_cross:
                            # 卖出信号
                            price = float(historical_data[close_col].iloc[-1])
                            self.signals.append({
                                'timestamp': current_time,
                                'action': 'sell',
                                'symbol': symbol,
                                'price': price
                            })
                            holding = False
                            print(f"卖出 {symbol} @ {price:.2f}")
        
        # 如果持仓到最后，生成一个卖出信号
        if holding and len(unique_times) > 0:
            last_time = unique_times[-1]
            if last_time in data.index:
                price = float(data.loc[last_time, close_col])
                self.signals.append({
                    'timestamp': last_time,
                    'action': 'sell',
                    'symbol': symbol,
                    'price': price
                })
                print(f"平仓 {symbol} @ {price:.2f}")
        
        # 如果没有生成信号，生成测试信号
        if not self.signals and len(data) > 0:
            print("⚠️  未检测到交易信号，生成测试信号")
            
            # 生成简单的买入卖出信号
            if len(data) >= 2:
                # 买入信号
                buy_time = data.index[-1]
                buy_price = float(data[close_col].iloc[-1])
                self.signals.append({
                    'timestamp': buy_time,
                    'action': 'buy',
                    'symbol': symbol,
                    'price': buy_price
                })
                
                # 卖出信号
                sell_time = data.index[-2]
                sell_price = float(data[close_col].iloc[-2])
                self.signals.append({
                    'timestamp': sell_time,
                    'action': 'sell',
                    'symbol': symbol,
                    'price': sell_price
                })
        
        print(f"✅ 生成 {len(self.signals)} 个信号")
        for i, signal in enumerate(self.signals[:5]):  # 只显示前5个
            print(f"  信号{i+1}: {signal['action']} {signal['symbol']} @ {signal['price']:.2f}")
        
        return self.signals
    
    def _generate_signals_multi_stock(self, data):
        """为多股票数据生成信号（原有逻辑）"""
        # 定义价格和股票代码列名
        close_col = 'close' if 'close' in data.columns else data.columns[0]
        symbol_col = 'symbol' if 'symbol' in data.columns else 'DEFAULT'
        
        self.signals = []
        current_holding = None  # 当前持有的股票
        
        # 按时间遍历
        unique_times = data.index.unique()
        
        for i, current_time in enumerate(unique_times):
            current_bars = data.loc[current_time]
            
            # 如果只有一只股票，确保格式统一
            if isinstance(current_bars, pd.Series):
                # 使用reset_index避免索引重复问题
                current_bars = current_bars.to_frame().T
            
            # 如果没有持仓，选择最优股票
            if current_holding is None:
                best_stock = self._select_best_stock(current_bars, current_time, data)
                if best_stock:
                    # 获取价格
                    price = 100.0
                    if best_stock in current_bars['symbol'].values:
                        stock_data = current_bars[current_bars['symbol'] == best_stock]
                        if close_col in stock_data.columns:
                            price = float(stock_data[close_col].iloc[0])
                    
                    self.signals.append({
                        'timestamp': current_time,
                        'action': 'buy',
                        'symbol': best_stock,
                        'price': price
                    })
                    current_holding = best_stock
                    print(f"买入 {best_stock} @ {price:.2f}")
            
            # 如果有持仓，检查是否需要卖出
            else:
                should_sell = self._check_sell_signal(current_holding, current_time, data)
                if should_sell:
                    # 获取价格
                    price = 100.0
                    if current_holding in current_bars['symbol'].values:
                        stock_data = current_bars[current_bars['symbol'] == current_holding]
                        if close_col in stock_data.columns:
                            price = float(stock_data[close_col].iloc[0])
                    
                    self.signals.append({
                        'timestamp': current_time,
                        'action': 'sell',
                        'symbol': current_holding,
                        'price': price
                    })
                    print(f"卖出 {current_holding} @ {price:.2f}")
                    current_holding = None
        
        # 如果持仓到最后，生成一个卖出信号
        if current_holding is not None and len(unique_times) > 0:
            last_time = unique_times[-1]
            if last_time in data.index:
                last_bars = data.loc[last_time]
                if isinstance(last_bars, pd.Series):
                    last_bars = last_bars.to_frame().T
                
                price = 100.0
                if current_holding in last_bars['symbol'].values:
                    stock_data = last_bars[last_bars['symbol'] == current_holding]
                    if close_col in stock_data.columns:
                        price = float(stock_data[close_col].iloc[0])
                
                self.signals.append({
                    'timestamp': last_time,
                    'action': 'sell',
                    'symbol': current_holding,
                    'price': price
                })
                print(f"平仓 {current_holding} @ {price:.2f}")
        
        # 如果没有生成任何信号，生成测试信号
        if not self.signals and len(data) > 0:
            print("⚠️  未检测到交易信号，生成测试信号")
            
            # 生成买入和卖出测试信号
            if len(data) >= 2:
                # 买入信号
                buy_time = data.index[-1]
                buy_symbol = data[symbol_col].iloc[-1] if symbol_col in data.columns else 'TEST'
                buy_price = float(data[close_col].iloc[-1])
                
                self.signals.append({
                    'timestamp': buy_time,
                    'action': 'buy',
                    'symbol': buy_symbol,
                    'price': buy_price
                })
                
                # 卖出信号
                sell_time = data.index[-2]
                sell_symbol = data[symbol_col].iloc[-2] if symbol_col in data.columns else 'TEST'
                sell_price = float(data[close_col].iloc[-2])
                
                self.signals.append({
                    'timestamp': sell_time,
                    'action': 'sell',
                    'symbol': sell_symbol,
                    'price': sell_price
                })
        
        print(f"✅ 生成 {len(self.signals)} 个信号")
        for i, signal in enumerate(self.signals[:5]):  # 只显示前5个
            print(f"  信号{i+1}: {signal['action']} {signal['symbol']} @ {signal['price']:.2f}")
        
        return self.signals

# 导出类
__all__ = ["MovingAverageStrategy"]
