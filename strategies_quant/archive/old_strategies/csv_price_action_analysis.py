import pandas as pd
import numpy as np
from pathlib import Path
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# 策略改造: 添加BaseStrategy导入
try:
    from core.base_strategy import BaseStrategy
except ImportError:
    from core.base_strategy import BaseStrategy

# ========== 核心信号定义接口 ==========
class SignalDefinition(ABC):
    """信号定义抽象基类 - 用户可以继承并实现自己的信号逻辑"""
    
    @abstractmethod
    def generate_signals(self, df_window: pd.DataFrame) -> List[Dict]:
        """
        生成信号的核心方法
        
        参数:
            df_window: 当前窗口的DataFrame，包含OHLCV数据
            
        返回:
            信号列表，每个信号至少包含:
            - type: 信号类型（字符串）
            - idx: 信号在窗口中的位置（整数）
            - price: 信号价格（浮点数）
            - features: 特征字典（用于后续分析）
        """
        pass
    
    def get_signal_name(self) -> str:
        """返回信号定义的名称"""
        return self.__class__.__name__


class MACDDivergenceSignal(SignalDefinition):
    """MACD顶底背离信号检测（全局检测+延迟确认版本）"""
    
    def __init__(self, fast_period=12, slow_period=26, signal_period=9, delay_bars=2):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.delay_bars = delay_bars  # 确认延迟的K线数量
    
    def get_signal_name(self) -> str:
        return "MACD_Divergence_Global"
    
    def calculate_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算MACD指标"""
        df = df.copy()
        df['ema_fast'] = df['close'].ewm(span=self.fast_period, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=self.slow_period, adjust=False).mean()
        df['dif'] = df['ema_fast'] - df['ema_slow']
        df['dea'] = df['dif'].ewm(span=self.signal_period, adjust=False).mean()
        df['macd_hist'] = (df['dif'] - df['dea']) * 2
        return df
    
    def find_extreme_points(self, price_series: pd.Series, macd_series: pd.Series) -> tuple:
        """寻找价格和MACD的极值点"""
        n = len(price_series)
        price_peaks = []
        price_valleys = []
        macd_peaks = []
        macd_valleys = []
        
        # 寻找价格极值点
        for i in range(1, n-1):
            # 价格峰值
            if price_series.iloc[i] > price_series.iloc[i-1] and price_series.iloc[i] > price_series.iloc[i+1]:
                price_peaks.append(i)
            # 价格谷值
            elif price_series.iloc[i] < price_series.iloc[i-1] and price_series.iloc[i] < price_series.iloc[i+1]:
                price_valleys.append(i)
        
        # 寻找MACD极值点
        for i in range(1, n-1):
            # MACD峰值
            if macd_series.iloc[i] > macd_series.iloc[i-1] and macd_series.iloc[i] > macd_series.iloc[i+1]:
                macd_peaks.append(i)
            # MACD谷值
            elif macd_series.iloc[i] < macd_series.iloc[i-1] and macd_series.iloc[i] < macd_series.iloc[i+1]:
                macd_valleys.append(i)
        
        return price_peaks, price_valleys, macd_peaks, macd_valleys
    
    def detect_divergence_global(self, df_with_macd: pd.DataFrame) -> List[Dict]:
        """全局检测背离信号"""
        signals = []
        price_series = df_with_macd['close']
        macd_series = df_with_macd['dif']
        
        # 寻找极值点
        price_peaks, price_valleys, macd_peaks, macd_valleys = self.find_extreme_points(price_series, macd_series)
        
        # 检测顶背离（价格创新高，MACD未创新高）
        for i in range(1, len(price_peaks)):
            curr_peak = price_peaks[i]
            prev_peak = price_peaks[i-1]
            
            if (price_series.iloc[curr_peak] > price_series.iloc[prev_peak] and
                self.has_corresponding_macd_peak(macd_peaks, curr_peak, prev_peak, macd_series, lower=True)):
                
                # 计算延迟确认的位置和价格
                confirm_idx = curr_peak + self.delay_bars
                if confirm_idx < len(df_with_macd):
                    signals.append({
                        'type': 'top_divergence',
                        'divergence_idx': curr_peak,  # 背离发生的位置
                        'idx': confirm_idx,  # 实际信号位置（延迟后）
                        'price': price_series.iloc[confirm_idx],  # 实际信号价格
                        'features': {
                            'divergence_type': 'top_divergence',
                            'divergence_price': price_series.iloc[curr_peak],  # 背离点的价格
                            'prev_price': price_series.iloc[prev_peak],
                            'curr_price': price_series.iloc[curr_peak],
                            'prev_macd': macd_series.iloc[prev_peak],
                            'curr_macd': macd_series.iloc[curr_peak],
                            'delay_bars': self.delay_bars,
                            'strength': self.calculate_divergence_strength(
                                price_series.iloc[prev_peak], price_series.iloc[curr_peak],
                                macd_series.iloc[prev_peak], macd_series.iloc[curr_peak]
                            )
                        }
                    })
        
        # 检测底背离（价格创新低，MACD未创新低）
        for i in range(1, len(price_valleys)):
            curr_valley = price_valleys[i]
            prev_valley = price_valleys[i-1]
            
            if (price_series.iloc[curr_valley] < price_series.iloc[prev_valley] and
                self.has_corresponding_macd_valley(macd_valleys, curr_valley, prev_valley, macd_series, higher=True)):
                
                # 计算延迟确认的位置和价格
                confirm_idx = curr_valley + self.delay_bars
                if confirm_idx < len(df_with_macd):
                    signals.append({
                        'type': 'bottom_divergence',
                        'divergence_idx': curr_valley,  # 背离发生的位置
                        'idx': confirm_idx,  # 实际信号位置（延迟后）
                        'price': price_series.iloc[confirm_idx],  # 实际信号价格
                        'features': {
                            'divergence_type': 'bottom_divergence',
                            'divergence_price': price_series.iloc[curr_valley],  # 背离点的价格
                            'prev_price': price_series.iloc[prev_valley],
                            'curr_price': price_series.iloc[curr_valley],
                            'prev_macd': macd_series.iloc[prev_valley],
                            'curr_macd': macd_series.iloc[curr_valley],
                            'delay_bars': self.delay_bars,
                            'strength': self.calculate_divergence_strength(
                                price_series.iloc[prev_valley], price_series.iloc[curr_valley],
                                macd_series.iloc[prev_valley], macd_series.iloc[curr_valley]
                            )
                        }
                    })
        
        return signals
    
    def has_corresponding_macd_peak(self, macd_peaks: List[int], curr_price_peak: int, 
                                   prev_price_peak: int, macd_series: pd.Series, lower: bool = False) -> bool:
        """检查是否有对应的MACD峰值，并满足条件"""
        # 找到与价格峰值对应的MACD峰值
        curr_macd_peak = self.find_nearest_extreme(macd_peaks, curr_price_peak)
        prev_macd_peak = self.find_nearest_extreme(macd_peaks, prev_price_peak)
        
        if curr_macd_peak is None or prev_macd_peak is None:
            return False
        
        # 检查MACD是否满足条件（对于顶背离，当前MACD应该低于前一个）
        if lower:
            return macd_series.iloc[curr_macd_peak] < macd_series.iloc[prev_macd_peak]
        else:
            return macd_series.iloc[curr_macd_peak] > macd_series.iloc[prev_macd_peak]
    
    def has_corresponding_macd_valley(self, macd_valleys: List[int], curr_price_valley: int, 
                                     prev_price_valley: int, macd_series: pd.Series, higher: bool = False) -> bool:
        """检查是否有对应的MACD谷值，并满足条件"""
        # 找到与价格谷值对应的MACD谷值
        curr_macd_valley = self.find_nearest_extreme(macd_valleys, curr_price_valley)
        prev_macd_valley = self.find_nearest_extreme(macd_valleys, prev_price_valley)
        
        if curr_macd_valley is None or prev_macd_valley is None:
            return False
        
        # 检查MACD是否满足条件（对于底背离，当前MACD应该高于前一个）
        if higher:
            return macd_series.iloc[curr_macd_valley] > macd_series.iloc[prev_macd_valley]
        else:
            return macd_series.iloc[curr_macd_valley] < macd_series.iloc[prev_macd_valley]
    
    def find_nearest_extreme(self, extremes: List[int], target_idx: int, max_distance: int = 5) -> int:
        """在极值点列表中找到最接近目标索引的极值点"""
        if not extremes:
            return None
        
        # 找到最接近的极值点
        nearest = min(extremes, key=lambda x: abs(x - target_idx))
        
        # 如果距离太远，返回None
        if abs(nearest - target_idx) > max_distance:
            return None
        
        return nearest
    
    def calculate_divergence_strength(self, prev_price: float, curr_price: float, 
                                    prev_macd: float, curr_macd: float) -> float:
        """计算背离强度"""
        price_change_pct = (curr_price - prev_price) / prev_price * 100
        macd_change_pct = (curr_macd - prev_macd) / abs(prev_macd) * 100 if prev_macd != 0 else 0
        
        # 背离强度 = 价格变化百分比与MACD变化百分比的差异绝对值
        strength = abs(price_change_pct - macd_change_pct)
        
        return strength
    
    def generate_signals(self, df_window: pd.DataFrame) -> List[Dict]:
        """生成MACD背离信号"""
        if len(df_window) < self.slow_period + 10:
            return []
        
        try:
            # 计算MACD
            df_with_macd = self.calculate_macd(df_window)
            
            # 全局检测背离
            signals = self.detect_divergence_global(df_with_macd)
            
            # 转换索引回原始索引
            for signal in signals:
                original_idx = df_with_macd.index[signal['idx']]
                signal['idx'] = original_idx
                
                # 如果需要，也可以转换divergence_idx
                divergence_original_idx = df_with_macd.index[signal['divergence_idx']]
                signal['features']['divergence_original_idx'] = divergence_original_idx
            
            return signals
            
        except Exception as e:
            print(f"MACD背离信号生成错误: {e}")
            return []

class BreakoutSignal(SignalDefinition):
    """价格突破信号"""
    
    def __init__(self, lookback_period: int = 20, volatility_threshold: float = 1.5):
        self.lookback_period = lookback_period
        self.volatility_threshold = volatility_threshold
    
    def generate_signals(self, df_window: pd.DataFrame) -> List[Dict]:
        signals = []
        
        if len(df_window) < self.lookback_period + 1:
            return signals
        
        for i in range(self.lookback_period, len(df_window)):
            # 计算前期价格区间
            lookback_data = df_window.iloc[i-self.lookback_period:i]
            high_max = lookback_data['high'].max()
            low_min = lookback_data['low'].min()
            avg_volume = lookback_data['volume'].mean()
            current_volume = df_window['volume'].iloc[i]
            
            current_high = df_window['high'].iloc[i]
            current_low = df_window['low'].iloc[i]
            current_close = df_window['close'].iloc[i]
            
            # 向上突破
            if current_close > high_max and current_volume > avg_volume * self.volatility_threshold:
                signals.append({
                    'type': 'breakout_high',
                    'idx': i,
                    'price': current_close,
                    'features': {
                        'breakout_level': high_max,
                        'breakout_percent': (current_close - high_max) / high_max,
                        'volume_ratio': current_volume / avg_volume,
                        'lookback_period': self.lookback_period
                    }
                })
            
            # 向下突破
            elif current_close < low_min and current_volume > avg_volume * self.volatility_threshold:
                signals.append({
                    'type': 'breakout_low',
                    'idx': i,
                    'price': current_close,
                    'features': {
                        'breakout_level': low_min,
                        'breakout_percent': (low_min - current_close) / low_min,
                        'volume_ratio': current_volume / avg_volume,
                        'lookback_period': self.lookback_period
                    }
                })
        
        return signals

# ========== 信号分析引擎 ==========
class SignalAnalysisEngine:
    """通用的信号分析引擎"""
    
    def __init__(self, signal_definitions: List[SignalDefinition]):
        """
        初始化分析引擎
        
        参数:
            signal_definitions: 信号定义列表
        """
        self.signal_definitions = signal_definitions
    
    def analyze_window(self, df_window: pd.DataFrame) -> List[Dict]:
        """分析单个窗口，生成所有信号"""
        all_signals = []
        
        for signal_definition in self.signal_definitions:
            try:
                signals = signal_definition.generate_signals(df_window)
                
                # 添加信号定义信息
                for signal in signals:
                    signal['signal_definition'] = signal_definition.get_signal_name()
                
                all_signals.extend(signals)
            except Exception as e:
                print(f"信号定义 {signal_definition.get_signal_name()} 生成失败: {str(e)}")
        
        return all_signals

# ========== 信号评估框架 ==========
class SignalEvaluator:
    """信号评估器"""
    
    @staticmethod
    def calculate_future_returns(
        signals: List[Dict], 
        df_full: pd.DataFrame, 
        window_start_idx: int, 
        future_bars: int = 5
    ) -> List[Dict]:
        """计算信号在未来N根K线的收益率"""
        valid_signals = []
        
        for signal in signals:
            relative_idx = signal['idx']
            absolute_entry_idx = window_start_idx + relative_idx
            entry_price = signal['price']
            
            # 检查未来数据是否足够
            if absolute_entry_idx + future_bars < len(df_full):
                exit_price = df_full.iloc[absolute_entry_idx + future_bars]['close']
                
                # 计算收益率（多空统一处理）
                if signal['type'].startswith(('ma_golden_cross', 'rsi_oversold_bounce', 'breakout_high')):
                    # 做多信号
                    returns = (exit_price - entry_price) / entry_price
                else:
                    # 做空信号
                    returns = (entry_price - exit_price) / entry_price
                
                signal['future_returns'] = returns
                signal['future_bars'] = future_bars
                signal['absolute_idx'] = absolute_entry_idx
                valid_signals.append(signal)
        
        return valid_signals

# ========== 统计分析框架 ==========
class SignalStatistics:
    """信号统计分析"""
    
    @staticmethod
    def analyze_signals(df_signals: pd.DataFrame, future_bars: int) -> pd.DataFrame:
        """分析信号统计"""
        if df_signals.empty:
            return pd.DataFrame()
        
        results = []
        
        # 按不同维度分组分析
        grouping_levels = [
            ['signal_type'],  # 按信号类型
            ['signal_type', 'signal_definition'],  # 按信号类型和定义
            ['signal_definition'],  # 按信号定义
        ]
        
        for group_cols in grouping_levels:
            if not all(col in df_signals.columns for col in group_cols):
                continue
                
            grouped = df_signals.groupby(group_cols)
            
            for group_name, group_data in grouped:
                if len(group_data) < 5:  # 最小样本要求
                    continue
                
                stats = SignalStatistics._calculate_group_stats(group_data, group_name, group_cols, future_bars)
                results.append(stats)
        
        # 总体统计
        overall_stats = SignalStatistics._calculate_group_stats(df_signals, 'OVERALL', ['OVERALL'], future_bars)
        results.append(overall_stats)
        
        return pd.DataFrame(results)
    
    @staticmethod
    def _calculate_group_stats(group: pd.DataFrame, group_name: Any, group_cols: List[str], future_bars: int) -> Dict:
        """计算分组统计指标"""
        returns = group['return']
        total_count = len(returns)
        
        if total_count == 0:
            return {}
        
        # 基础统计
        win_rate = (returns > 0).mean()
        avg_return = returns.mean()
        std_return = returns.std()
        
        # 盈亏统计
        winning_trades = returns[returns > 0]
        losing_trades = returns[returns < 0]
        
        avg_win = winning_trades.mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades.mean() if len(losing_trades) > 0 else 0
        win_loss_ratio = len(winning_trades) / len(losing_trades) if len(losing_trades) > 0 else float('inf')
        
        # 风险回报比
        risk_reward_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        # 期望值
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * abs(avg_loss))
        
        # 最大回撤和盈利因子
        cumulative_returns = (1 + returns).cumprod()
        max_drawdown = (cumulative_returns / cumulative_returns.expanding().max() - 1).min()
        profit_factor = abs(winning_trades.sum() / losing_trades.sum()) if losing_trades.sum() != 0 else float('inf')
        
        # 创建结果字典
        result = {
            'signal_group': str(group_name),
            'grouping_columns': '_'.join(group_cols),
            'sample_count': total_count,
            'win_rate': win_rate,
            'avg_return': avg_return,
            'std_return': std_return,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'win_loss_ratio': win_loss_ratio,
            'risk_reward_ratio': risk_reward_ratio,
            'expectancy': expectancy,
            'max_drawdown': max_drawdown,
            'profit_factor': profit_factor,
            'future_bars': future_bars
        }
        
        # 添加分组信息
        if isinstance(group_name, tuple):
            for i, col in enumerate(group_cols):
                result[col] = group_name[i]
        elif group_cols[0] != 'OVERALL':
            result[group_cols[0]] = group_name
        
        return result

# ========== 主批处理框架 ==========
def analyze_signal_probabilities(
    input_dir: str,
    output_dir: str,
    signal_definitions: List[SignalDefinition],
    window_size: int = 100,
    step: int = 5,
    future_bars: int = 5
):
    """
    通用的信号概率分析主函数
    
    参数:
        input_dir: 输入CSV文件目录
        output_dir: 输出目录
        signal_definitions: 信号定义列表
        window_size: 分析窗口大小
        step: 窗口滑动步长
        future_bars: 未来收益计算周期
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 初始化分析引擎
    analysis_engine = SignalAnalysisEngine(signal_definitions)
    
    csv_files = list(input_path.glob("*.csv"))  # 改为匹配所有CSV文件
    if not csv_files:
        print(f"在 {input_dir} 中未找到CSV文件")
        return
    
    print(f"找到 {len(csv_files)} 个CSV文件，开始信号概率分析...")
    print(f"使用的信号定义: {[sd.get_signal_name() for sd in analysis_engine.signal_definitions]}")
    
    all_signals = []
    
    for csv_file in sorted(csv_files):
        print(f"\n处理文件: {csv_file.name}")
        stock_name = csv_file.stem
        
        try:
            # 读取数据（假设包含OHLCV列）
            df_full = pd.read_csv(csv_file)
            if 'trade_date' in df_full.columns:
                df_full = df_full.sort_values('trade_date')
            else:
                df_full = df_full.reset_index(drop=True)
            
            # 检查必要的列
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df_full.columns]
            if missing_columns:
                print(f"  跳过: 缺少必要列 {missing_columns}")
                continue
            
            if len(df_full) < window_size:
                print(f"  跳过: 数据不足 {window_size} 行")
                continue
            
            # 滑动窗口分析
            window_count = 0
            signal_count = 0
            
            for start in range(0, len(df_full) - window_size + 1, step):
                end = start + window_size
                df_window = df_full.iloc[start:end].copy().reset_index(drop=True)
                
                try:
                    # 生成信号
                    signals = analysis_engine.analyze_window(df_window)
                    
                    # 计算未来收益
                    valid_signals = SignalEvaluator.calculate_future_returns(
                        signals, df_full, start, future_bars
                    )
                    
                    # 添加元数据
                    for signal in valid_signals:
                        signal.update({
                            'stock': stock_name,
                            'window_start': start,
                            'window_end': end - 1
                        })
                    
                    all_signals.extend(valid_signals)
                    signal_count += len(valid_signals)
                    window_count += 1
                    
                except Exception as e:
                    if window_count % 50 == 0:  # 减少错误输出频率
                        print(f"    窗口 {start}-{end-1} 处理失败: {str(e)}")
            
            print(f"  ✅ 完成: {window_count} 个窗口, {signal_count} 个信号")
            
        except Exception as e:
            print(f"  ❌ 文件处理失败: {str(e)}")
    
    # 保存和统计分析
    if all_signals:
        _save_and_analyze_results(all_signals, output_path, future_bars)
    else:
        print("\n⚠️  没有找到任何有效信号")

def _save_and_analyze_results(all_signals: List[Dict], output_path: Path, future_bars: int):
    """保存结果并进行统计分析"""
    # 转换为DataFrame
    signal_records = []
    for signal in all_signals:
        record = {
            'stock': signal['stock'],
            'window_start': signal['window_start'],
            'window_end': signal['window_end'],
            'signal_type': signal['type'],
            'signal_definition': signal.get('signal_definition', 'Unknown'),
            'entry_price': signal['price'],
            'return': signal['future_returns'],
            'future_bars': signal['future_bars'],
            'absolute_index': signal.get('absolute_idx', -1)
        }
        
        # 添加特征
        features = signal.get('features', {})
        for feature_name, feature_value in features.items():
            record[f'feature_{feature_name}'] = feature_value
        
        signal_records.append(record)
    
    df_all_signals = pd.DataFrame(signal_records)
    
    # 保存原始信号
    signals_output = output_path / "all_signals_raw.csv"
    df_all_signals.to_csv(signals_output, index=False, float_format='%.6f')
    print(f"\n✅ 所有原始信号数据已保存至: {signals_output}")
    
    # 统计分析
    df_stats = SignalStatistics.analyze_signals(df_all_signals, future_bars)
    if not df_stats.empty:
        stats_output = output_path / "signal_probability_statistics.csv"
        df_stats.to_csv(stats_output, index=False, float_format='%.4f')
        _print_statistics_summary(df_stats, future_bars)
    else:
        print("\n⚠️  没有足够的信号进行统计分析")

def _print_statistics_summary(df_stats: pd.DataFrame, future_bars: int):
    """打印统计摘要"""
    print(f"\n{'='*60}")
    print(f"信号概率统计分析结果 (未来{future_bars}根K线)")
    print(f"{'='*60}")
    
    # 按分组列排序，总体统计放在最后
    df_sorted = df_stats.sort_values('grouping_columns')
    
    for _, row in df_sorted.iterrows():
        group_desc = row['signal_group']
        if row['grouping_columns'] == 'OVERALL':
            print(f"\n📊 总体统计:")
        else:
            print(f"\n📈 {group_desc}:")
        
        print(f"   样本数量: {row['sample_count']:>6d}")
        print(f"   胜率:     {row['win_rate']:>8.2%}")
        print(f"   平均收益: {row['avg_return']:>8.2%}")
        print(f"   盈亏比:   {row['risk_reward_ratio']:>8.2f}")
        print(f"   期望值:   {row['expectancy']:>8.2%}")



# ========== 主程序 ==========
if __name__ == "__main__":
    # 配置路径
    input_dir = r"E:\stock\csv_version\analysis_results"  # 修改为你的输入目录
    output_dir = r"E:\stock\csv_version\signal_probabilities"
    
    # 定义要使用的信号
    signal_definitions = [
        #BreakoutSignal(lookback_period=20, volatility_threshold=1.5),
        MACDDivergenceSignal()
    ]
    
    # 运行信号分析
    analyze_signal_probabilities(
        input_dir=input_dir,
        output_dir=output_dir,
        signal_definitions=signal_definitions,
        window_size=100,
        step=10,  # 可以调整步长以减少计算量
        future_bars=5
    )


# ============================================================================
# 策略改造: 添加CSVPriceActionAnalysisStrategy类
# 将CSV价格行为分析系统转换为交易策略
# ============================================================================

class CSVPriceActionAnalysisStrategy(BaseStrategy):
    """CSV价格行为分析策略"""
    
    def __init__(self, data: pd.DataFrame, params: dict):
        """
        初始化策略
        
        参数:
            data: 价格数据
            params: 策略参数
        """
        super().__init__(data, params)
        
        # 从params提取参数
        signal_type = params.get('signal_type', 'MACDDivergence')
        
        # 创建信号定义实例
        if signal_type == 'MACDDivergence':
            self.signal_def = MACDDivergenceSignal()
        else:
            # 默认信号定义
            self.signal_def = MACDDivergenceSignal()
    
    def generate_signals(self):
        """
        生成交易信号
        
        基于CSV价格行为分析生成交易信号
        """
        # 使用信号定义生成信号
        signals = self.signal_def.generate_signals(self.data)
        
        if signals:
            # 取第一个信号
            first_signal = signals[0]
            signal_type = first_signal.get('type', '')
            
            if 'buy' in signal_type.lower() or 'bottom' in signal_type.lower():
                self._record_signal(
                    timestamp=self.data.index[-1],
                    action='buy',
                    price=self.data['close'].iloc[-1]
                )
            elif 'sell' in signal_type.lower() or 'top' in signal_type.lower():
                self._record_signal(
                    timestamp=self.data.index[-1],
                    action='sell',
                    price=self.data['close'].iloc[-1]
                )
            else:
                # 未知信号类型，hold
                self._record_signal(
                    timestamp=self.data.index[-1],
                    action='hold',
                    price=self.data['close'].iloc[-1]
                )
        else:
            # 无信号，hold
            self._record_signal(
                timestamp=self.data.index[-1],
                action='hold',
                price=self.data['close'].iloc[-1]
            )
        
        return self.signals


# ============================================================================
# 策略改造完成
# ============================================================================