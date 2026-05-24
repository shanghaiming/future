#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
from core.base_strategy import BaseStrategy
A股股票多周期方差分析程序
依赖：mindgp_api, numpy, pandas, tqdm
"""
from core.base_strategy import BaseStrategy
import numpy as np
import pandas as pd
from tqdm import tqdm
from numpy.lib.stride_tricks import sliding_window_view
import datetime
try:
    import ipywidgets as widgets
    _HAS_IPYWIDGETS = True
except ImportError:
    _HAS_IPYWIDGETS = False
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from scipy.signal import argrelextrema
from scipy.signal import find_peaks
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression
from scipy import stats, signal, cluster
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False   # 用来正常显示负号
# 配置参数
CONFIG = {
    "PERIODS": [5, 8, 13, 21, 34, 55],  # 需要计算的周期
    "MIN_DATA_DAYS": 55,                # 需要的最少数据天数
    "BATCH_SIZE": 100,                  # 分批处理数量
    "OUTPUT_FILE": "output/selected_stocks"  # 输出文件名
}


def calculate_williams_r(data, period=5, high_col='high', low_col='low', close_col='close'):
    """
    计算威廉指标（Williams %R），适配DataFrame输入
    
    参数：
    data : DataFrame
        必须包含最高价、最低价、收盘价列（列名可自定义）
    period : int
        计算周期（默认14）
    high_col, low_col, close_col : str
        指定最高价、最低价、收盘价的列名
    
    返回：
    Series
        Williams %R值，与原始数据索引对齐
    """
    # 确保输入是DataFrame
    if not isinstance(data, pd.DataFrame):
        raise ValueError("输入数据必须是Pandas DataFrame")
    
    # 检查必要的列是否存在
    required_cols = [high_col, low_col, close_col]
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise KeyError(f"缺失必要列: {missing_cols}")
    
    # 计算滚动窗口内的最高价和最低价
    highest_high = data[high_col].rolling(window=period, min_periods=period).max()
    lowest_low = data[low_col].rolling(window=period, min_periods=period).min()
    
    # 计算Williams %R
    numerator = highest_high - data[close_col]
    denominator = highest_high - lowest_low
    williams_r1 = (numerator / denominator) * -100
    
    # 处理分母为零的情况（设为NaN）
    williams_r1 = williams_r1.where(denominator != 0, float('nan'))
    williams_r =williams_r1.ewm(span=period, adjust=False).mean()
    
    return williams_r.rename(f'Williams_%R_{period}')    



class AdaptiveRegressionChannel:
    def __init__(self, lookback_period=20, std_multiplier=2, min_trend_length=10):
        """
        初始化自适应回归通道
        
        参数:
        lookback_period (int): 寻找极值点的回溯周期
        std_multiplier (float): 标准差乘数
        min_trend_length (int): 最小趋势长度
        """
        self.lookback_period = lookback_period
        self.std_multiplier = std_multiplier
        self.min_trend_length = min_trend_length
        
    def find_extreme_points(self, high_prices, low_prices):
        """
        使用科学方法寻找极高点和极低点
        
        参数:
        high_prices (array-like): 最高价序列
        low_prices (array-like): 最低价序列
        
        返回:
        tuple: (high_points, low_points) 极高点和极低点的索引和值
        """
        # 使用scipy的argrelextrema函数寻找局部极值点
        high_indices = argrelextrema(high_prices, np.greater, order=self.lookback_period)[0]
        low_indices = argrelextrema(low_prices, np.less, order=self.lookback_period)[0]
        
        high_points = [(i, high_prices[i]) for i in high_indices]
        low_points = [(i, low_prices[i]) for i in low_indices]
        
        return high_points, low_points
    
    def evaluate_regression_quality(self, prices, start_idx):
        """
        评估回归质量
        
        参数:
        prices (array-like): 价格序列
        start_idx (int): 起始索引
        
        返回:
        float: 回归质量评分 (0-1)
        """
        if start_idx >= len(prices) - self.min_trend_length:
            return 0
            
        # 从起点到当前点的数据
        segment_prices = prices[start_idx:]
        
        # 准备数据
        X = np.arange(len(segment_prices)).reshape(-1, 1)
        y = np.array(segment_prices).reshape(-1, 1)
        
        # 计算线性回归
        model = LinearRegression()
        model.fit(X, y)
        
        # 回归线
        regression_line = model.predict(X).flatten()
        
        # 计算残差和标准差
        residuals = y.flatten() - regression_line
        std = np.std(residuals)
        
        # 计算R²值
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y.flatten() - np.mean(y.flatten()))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # 计算价格在通道内的比例
        upper_band = regression_line + self.std_multiplier * std
        lower_band = regression_line - self.std_multiplier * std
        
        in_channel = np.sum((segment_prices >= lower_band) & (segment_prices <= upper_band))
        channel_ratio = in_channel / len(segment_prices)
        
        # 综合评分：R²和通道内比例的平均值
        quality_score = (r_squared + channel_ratio) / 2
        
        return quality_score
    
    def find_best_regression_start(self, close_prices, high_points, low_points):
        """
        寻找最佳的回归起点
        
        参数:
        close_prices (array-like): 收盘价序列
        high_points: 极高点列表
        low_points: 极低点列表
        
        返回:
        dict: 包含最佳起点信息
        """
        # 合并所有极值点并按时间排序
        all_extremes = []
        for idx, price in high_points:
            all_extremes.append((idx, price, 'high'))
        for idx, price in low_points:
            all_extremes.append((idx, price, 'low'))
        
        # 按索引排序
        all_extremes.sort(key=lambda x: x[0])
        
        # 确保有足够的极值点
        if len(all_extremes) < 2:
            # 如果没有足够的极值点，使用序列起点
            return {
                'start_index': 0,
                'start_price': close_prices[0],
                'start_type': 'start',
                'quality_score': 0.5
            }
        
        # 评估每个可能的起点
        candidate_scores = []
        for extreme in all_extremes:
            if extreme[0] <= len(close_prices) - self.min_trend_length:
                score = self.evaluate_regression_quality(close_prices, extreme[0])
                candidate_scores.append((extreme, score))
        
        # 如果没有合适的候选点，使用序列起点
        if not candidate_scores:
            return {
                'start_index': 0,
                'start_price': close_prices[0],
                'start_type': 'start',
                'quality_score': 0.5
            }
        
        # 选择评分最高的起点
        best_candidate = max(candidate_scores, key=lambda x: x[1])
        best_extreme, best_score = best_candidate
        
        return {
            'start_index': best_extreme[0],
            'start_price': best_extreme[1],
            'start_type': best_extreme[2],
            'quality_score': best_score
        }
    
    def calculate_regression_channel(self, prices, start_index):
        """
        计算线性回归通道
        
        参数:
        prices (array-like): 价格序列
        start_index (int): 起始索引
        
        返回:
        dict: 包含回归线、上轨、下轨的字典
        """
        if start_index >= len(prices):
            raise ValueError("起点索引超出价格序列范围")
        
        # 从起点到当前点的数据
        start_idx = start_index
        end_idx = len(prices)
        segment_prices = prices[start_idx:end_idx]
        
        if len(segment_prices) < 2:
            raise ValueError("从起点开始的数据点太少，无法计算回归")
        
        # 准备数据
        X = np.arange(len(segment_prices)).reshape(-1, 1)
        y = np.array(segment_prices).reshape(-1, 1)
        
        # 计算线性回归
        model = LinearRegression()
        model.fit(X, y)
        
        # 回归线
        regression_line = model.predict(X).flatten()
        
        # 计算残差和标准差
        residuals = y.flatten() - regression_line
        std = np.std(residuals)
        
        # 计算上下轨
        upper_band = regression_line + self.std_multiplier * std
        lower_band = regression_line - self.std_multiplier * std
        
        # 计算R²值
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y.flatten() - np.mean(y.flatten()))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return {
            'regression_line': regression_line,
            'upper_band': upper_band,
            'lower_band': lower_band,
            'slope': model.coef_[0][0],
            'intercept': model.intercept_[0],
            'r_squared': r_squared,
            'start_index': start_idx,
            'end_index': end_idx - 1
        }
    
    def get_adaptive_regression(self, close_prices, high_prices, low_prices):
        """
        获取自适应回归通道
        
        参数:
        close_prices (array-like): 收盘价序列
        high_prices (array-like): 最高价序列
        low_prices (array-like): 最低价序列
        
        返回:
        dict: 包含回归通道信息和起点信息的字典
        """
        if len(close_prices) < self.min_trend_length:
            raise ValueError(f"价格序列长度({len(close_prices)})小于最小趋势长度({self.min_trend_length})")
        
        # 寻找极值点
        high_points, low_points = self.find_extreme_points(high_prices, low_prices)
        
        # 寻找最佳回归起点
        start_info = self.find_best_regression_start(close_prices, high_points, low_points)
        
        # 计算回归通道
        regression_result = self.calculate_regression_channel(
            close_prices, start_info['start_index']
        )
        
        return {
            'regression': regression_result,
            'start_info': start_info,
            'all_high_points': high_points,
            'all_low_points': low_points
        }
    
    def plot_kline_with_regression(self, df, title="K线图与自适应回归通道"):
        """
        绘制K线图与自适应回归通道
        
        参数:
        df (DataFrame): 包含OHLC(V)数据的DataFrame，列名应为小写
        title (str): 图表标题
        """
        # 检查必要的列是否存在
        required_columns = ['open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"数据框缺少必要的列: {missing_columns}")
        
        # 提取数据
        dates = df.index
        open_prices = df['open'].values
        high_prices = df['high'].values
        low_prices = df['low'].values
        close_prices = df['close'].values
        volume = df['volume'].values if 'volume' in df.columns else None
        
        # 计算自适应回归通道
        try:
            results = self.get_adaptive_regression(close_prices, high_prices, low_prices)
        except Exception as e:
            print(f"计算回归通道时出错: {e}")
            return
        
        regression = results['regression']
        start_info = results['start_info']
        all_high_points = results['all_high_points']
        all_low_points = results['all_low_points']
        
        # 创建图表
        if volume is not None:
            # 如果有成交量数据，创建两个子图
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), 
                                          gridspec_kw={'height_ratios': [3, 1]})
        else:
            # 如果没有成交量数据，只创建一个图
            fig, ax1 = plt.subplots(figsize=(16, 10))
            ax2 = None
        
        # 设置淡雅风格
        fig.patch.set_facecolor('white')
        ax1.set_facecolor('#f8f9fa')
        if ax2 is not None:
            ax2.set_facecolor('#f8f9fa')
        
        # 使用整数索引而不是日期，避免空白
        x_values = np.arange(len(dates))
        
        # 绘制K线
        self._plot_candlestick(ax1, x_values, open_prices, high_prices, low_prices, close_prices)
        
        # 标记所有极值点 - 修复变量名错误
        high_indices = [i for i, _ in all_high_points]
        high_values = [v for _, v in all_high_points]
        ax1.scatter(high_indices, high_values, color='#e74c3c', marker='v', s=60, 
                   label=f'高点({len(all_high_points)}个)', zorder=5, alpha=0.7)
        
        low_indices = [i for i, _ in all_low_points]
        low_values = [v for _, v in all_low_points]
        ax1.scatter(low_indices, low_values, color='#2ecc71', marker='^', s=60, 
                   label=f'低点({len(all_low_points)}个)', zorder=5, alpha=0.7)
        
        # 标记回归起点
        start_idx = start_info['start_index']
        ax1.scatter([start_idx], [start_info['start_price']], 
                   color='#3498db', 
                   marker='*', 
                   s=200, label='回归起点', zorder=6)
        
        # 绘制回归通道
        start_idx_reg = regression['start_index']
        end_idx = regression['end_index']
        
        # 创建索引切片
        x_slice = np.arange(start_idx_reg, end_idx+1)
        
        # 绘制回归线和通道
        ax1.plot(x_slice, regression['regression_line'], 
                label='回归线', color='#3498db', linewidth=2.5)
        ax1.plot(x_slice, regression['upper_band'], 
                label='上轨', color='#e74c3c', linestyle='--', linewidth=2)
        ax1.plot(x_slice, regression['lower_band'], 
                label='下轨', color='#2ecc71', linestyle='--', linewidth=2)
        
        # 填充通道区域
        ax1.fill_between(x_slice, regression['upper_band'], regression['lower_band'], 
                        alpha=0.1, color='#95a5a6', label='回归通道')
        
        # 添加垂直线标记回归起点位置
        ax1.axvline(x=start_idx, color='#9b59b6', linestyle=':', alpha=0.8, 
                   linewidth=2, label='起点位置')
        
        # 设置主图属性
        ax1.set_title(title, fontsize=16, fontweight='bold', pad=20, color='#2c3e50')
        ax1.set_ylabel('价格', fontsize=12, color='#2c3e50')
        ax1.legend(loc='upper left', fontsize=10, framealpha=0.9)
        ax1.grid(True, alpha=0.2, color='#bdc3c7')
        
        # 设置x轴刻度为日期标签
        # 选择适量的刻度点，避免过于拥挤
        n_ticks = min(10, len(dates))
        step = max(1, len(dates) // n_ticks)
        tick_indices = list(range(0, len(dates), step))
        if len(dates) - 1 not in tick_indices:
            tick_indices.append(len(dates) - 1)
        
        tick_labels = [dates[i].strftime('%Y-%m-%d') for i in tick_indices]
        ax1.set_xticks(tick_indices)
        ax1.set_xticklabels(tick_labels)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, fontsize=10)
        
        # 如果有成交量，绘制成交量
        if ax2 is not None and volume is not None:
            self._plot_volume(ax2, x_values, volume, close_prices, open_prices)
            ax2.set_ylabel('成交量', fontsize=12, color='#2c3e50')
            ax2.grid(True, alpha=0.2, color='#bdc3c7')
            ax2.set_xticks(tick_indices)
            ax2.set_xticklabels(tick_labels)
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, fontsize=10)
        
        plt.tight_layout()
        plt.show()
        
        # 打印统计信息
        self._print_statistics(regression, start_info, dates[start_idx], close_prices)
    
    def _plot_candlestick(self, ax, x_values, open_prices, high_prices, low_prices, close_prices):
        """
        绘制K线图
        
        参数:
        ax: matplotlib轴对象
        x_values: x轴数值序列
        open_prices: 开盘价序列
        high_prices: 最高价序列
        low_prices: 最低价序列
        close_prices: 收盘价序列
        """
        # 计算每个K线的宽度
        width = 0.7
        
        # 绘制每个K线
        for i, x_val in enumerate(x_values):
            open_price = open_prices[i]
            high_price = high_prices[i]
            low_price = low_prices[i]
            close_price = close_prices[i]
            
            # 确定颜色：上涨为红色，下跌为绿色
            color = '#e74c3c' if close_price >= open_price else '#2ecc71'
            
            # 绘制上下影线
            ax.plot([x_val, x_val], [low_price, high_price], color='#34495e', linewidth=0.8)
            
            # 绘制实体
            rect = Rectangle(
                (x_val - width/2, min(open_price, close_price)),
                width,
                abs(close_price - open_price),
                facecolor=color,
                edgecolor='#34495e',
                alpha=0.8
            )
            ax.add_patch(rect)
    
    def _plot_volume(self, ax, x_values, volume, close_prices, open_prices):
        """
        绘制成交量
        
        参数:
        ax: matplotlib轴对象
        x_values: x轴数值序列
        volume: 成交量序列
        close_prices: 收盘价序列
        open_prices: 开盘价序列
        """
        # 计算每个柱状图的宽度
        width = 0.7
        
        # 为每个成交量柱设置颜色
        colors = ['#e74c3c' if close_prices[i] >= open_prices[i] else '#2ecc71' for i in range(len(x_values))]
        
        # 绘制成交量柱状图
        for i, x_val in enumerate(x_values):
            ax.bar(x_val, volume[i], width=width, color=colors[i], alpha=0.7)
        
        ax.set_ylabel('成交量', fontsize=12, color='#2c3e50')
        ax.grid(True, alpha=0.2, color='#bdc3c7')
    
    def _print_statistics(self, regression, start_info, start_date, close_prices):
        """
        打印统计信息
        
        参数:
        regression: 回归通道结果
        start_info: 起点信息
        start_date: 起点日期
        close_prices: 收盘价序列
        """
        print("=" * 50)
        print("自适应回归通道统计信息:")
        print("=" * 50)
        print(f"起点类型: {start_info['start_type']}")
        print(f"起点日期: {start_date.strftime('%Y-%m-%d')}")
        print(f"起点价格: {start_info['start_price']:.2f}")
        print(f"回归质量评分: {start_info['quality_score']:.4f}")
        print(f"回归线斜率: {regression['slope']:.6f}")
        print(f"R²值: {regression['r_squared']:.4f}")
        print(f"当前价格: {close_prices[-1]:.2f}")
        print(f"上轨值: {regression['upper_band'][-1]:.2f}")
        print(f"下轨值: {regression['lower_band'][-1]:.2f}")
        print(f"回归线值: {regression['regression_line'][-1]:.2f}")
        print(f"通道宽度: {regression['upper_band'][-1] - regression['lower_band'][-1]:.2f}")
        print("=" * 50)


def filter_stocks(
    all_stocks, 
    table, 
    CONFIG, 
    history_func, 
    calculate_williams_r_func,
    Analyzer,
    max_workers=8
):
    """
    批量筛选符合条件的股票 (线程安全版本)
    返回价格与线性回归带下轨差距绝对值最小的100只股票，且线性回归斜率为正
    
    参数:
    all_stocks -- 待筛选的股票代码列表
    table -- 股票信息表，包含display_name等信息
    CONFIG -- 配置字典，包含MIN_DATA_DAYS, BATCH_SIZE等配置
    [各种技术指标计算函数] -- 所需的技术指标计算和分析函数
    max_workers -- 并行工作线程数，默认为8
    
    返回:
    (results, valid_stock_count) -- 符合条件的股票列表和有效数量
    """
    
    # 处理单只股票的筛选逻辑
    def _process_single_stock(stock_code):
        if stock_code.startswith("8") or stock_code.startswith("9"):
            return None
        
        try:
            # 获取历史数据
            time = datetime.date.today()
            price_data = history_func(
                        stock=stock_code,
                        end_date=time.strftime('%Y%m%d'),
                        start_date= (time - datetime.timedelta(days=200)).strftime('%Y%m%d')
                        )
            
            # 基础检查
            if len(price_data) < CONFIG["MIN_DATA_DAYS"]:
                print(f"数据不够{stock_code}")
                return None
            
            # 提取数据
            hist = price_data
            close_prices = hist['close'].values
            high_prices = hist['high'].values
            low_prices = hist['low'].values
            
            # 计算自适应回归通道
            analyzer = Analyzer()
            results = analyzer.get_adaptive_regression(close_prices, high_prices, low_prices)
            
            regression = results['regression']
            start_info = results['start_info']
            
            # 获取关键指标
            current_price = close_prices[-1]
            lower_band = regression['lower_band'][-1]
            upper_band = regression['upper_band'][-1]
            slope = regression['slope']
            
            # 计算价格与下轨的差距绝对值
            price_to_lower_gap = abs(current_price - lower_band)
            
            # 筛选条件：斜率必须为正
            if slope <= 0:
                return None
            
            # 计算相对差距（相对于价格的比例）
            relative_gap = price_to_lower_gap / (upper_band-lower_band)
            if relative_gap > 0.02:
                return None
            # 计算威廉指标
            hist['wr5'] = calculate_williams_r_func(hist, period=5)
            wr5_value = hist['wr5'].iloc[-1]
            
            # 计算R²值作为质量指标
            r_squared = regression['r_squared']
            
            # 计算通道宽度相对值
            channel_width = (regression['upper_band'][-1] - lower_band) / current_price
            
            return {
                "股票代码": stock_code,
                "名字": table.loc[stock_code, 'name'],
                "当前价格": current_price,
                "下轨价格": lower_band,
                "价格下轨差距": price_to_lower_gap,
                "相对差距": relative_gap,
                "回归斜率": slope,
                "R平方": r_squared,
                "wr5": wr5_value,
                "通道宽度比例": channel_width,
                "回归质量": start_info['quality_score']
            }
            
        except Exception as e:
            # 实际使用时可记录日志
            print(f"{stock_code}处理失败: {str(e)}")
            return None
    
    # 主筛选逻辑
    results = []
    valid_stock_count = 0
    batch_size = CONFIG["BATCH_SIZE"]
    
    # 使用线程池处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        
        # 分批提交任务
        for i in range(0, len(all_stocks), batch_size):
            batch = all_stocks[i:i+batch_size]
            for stock_code in batch:
                if stock_code.startswith("8"):
                    continue
                future = executor.submit(_process_single_stock, stock_code)
                futures[future] = stock_code
        
        # 处理完成的任务
        for future in as_completed(futures):
            stock_code = futures[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
                    valid_stock_count += 1
                    
                    # 实时显示进度
                    if valid_stock_count % 10 == 0:
                        print(f"已处理 {valid_stock_count} 只有效股票...")
                        
            except Exception as e:
                print(f"处理{stock_code}时出错: {str(e)}")
                pass
    
    # 筛选和排序
    if results:
        # 创建DataFrame
        df_results = pd.DataFrame(results)
        
        # 按相对差距排序（差距最小的在前）
        df_sorted = df_results.sort_values(by="相对差距", ascending=True)
        
        # 取前100只
        top_100 = df_sorted.head(100)
        
        print(f"\n筛选完成！共找到 {len(top_100)} 只符合条件的股票")
        print(f"平均相对差距: {top_100['相对差距'].mean():.4f}")
        print(f"平均回归斜率: {top_100['回归斜率'].mean():.6f}")
        print(f"平均R平方: {top_100['R平方'].mean():.4f}")
        
        return top_100.to_dict('records'), valid_stock_count
    else:
        print("没有找到符合条件的股票")
        return [], 0
    

def get_stock_data(stock, start_date, end_date):
    """获取单只股票在指定日期范围内的数据"""
    
    file_path = fr"E:\stock\csv_version\analysis_results\{stock}_analysis.csv"
    df = pd.read_csv(file_path, index_col='trade_date', parse_dates=True)
    filtered_df = df.loc[start_date:end_date]
    
    return filtered_df

def main():
    # 获取全量股票池
    print("正在获取股票列表...")
    file_path = fr"E:\stock\csv_version\stocks_list.csv"
    df = pd.read_csv(file_path, index_col='ts_code')
    #df = df[df['list_date'] < 20181212].sort_values('list_date', ascending=False)
    table = df
    all_stocks = df.index.to_list()
    #all_stocks = ['603058.SH','688605.SH']
    print(f"共获取到{len(all_stocks)}只A股")
    # 初始化结果容器
    results = []
    valid_count = 0
    
    results, valid_count = filter_stocks(
    all_stocks=all_stocks,
    table=table,
    CONFIG=CONFIG,
    history_func=get_stock_data,
    calculate_williams_r_func=calculate_williams_r,
    Analyzer = AdaptiveRegressionChannel,
    max_workers=8  # 根据CPU核心数调整
    )
    
    # 处理结果
    if not results:
        print("没有有效数据可供分析")
        return

    # 创建DataFrame并排序
    df = pd.DataFrame(results).sort_values(by="wr5", ascending=True)
    
    # 保存结果
    #df.to_csv(CONFIG["OUTPUT_FILE"], index=False, encoding='utf_8_sig')
    
    # 打印摘要
    print("\n" + "="*50)
    print(f"分析完成！有效处理股票数：{valid_count}/{len(all_stocks)}")
    
    jupyter_stock_charts(df.head(70))  # 使用Jupyter专用控件
    '''notify_push(
    df[['股票代码', '名字', '平均方差']].head(10).to_string(index=False), 
    channel='wxpusher', 
    subject='SuperMind消息提醒', 
    email_list=None, 
    uids='UID_whd6sWkQtfsrIEFEifgZZWyLufEI', 
    topic_ids=None, 
    group_id=None,
    url=None,
    payload=None,
)'''
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
try:
    import ipywidgets as widgets
    _HAS_IPYWIDGETS_2 = True
except ImportError:
    _HAS_IPYWIDGETS_2 = False
import datetime
import pandas as pd
import numpy as np

def jupyter_stock_charts(df_results):
    """Jupyter专用股票图表浏览器（带线性回归通道和技术指标）"""
    if df_results.empty:
        print("没有股票可展示")
        return
    
    # 准备数据
    stock_codes = df_results["股票代码"].tolist()
    stock_names = df_results["名字"].tolist()
    current_index = 0
    
    # 创建控件
    prev_btn = widgets.Button(description="上一只")
    next_btn = widgets.Button(description="下一只")
    index_label = widgets.Label(value=f"股票: 1/{len(stock_codes)}")
    
    # 创建图表区域
    output = widgets.Output()
    time = datetime.date.today()
    
    # 定义更新图表函数
    def update_chart():
        with output:
            output.clear_output(wait=True)
            
            stock_code = stock_codes[current_index]
            stock_name = stock_names[current_index]
            
            # 获取股票日线数据
            price_data = get_stock_data(
                        stock=stock_code,
                        end_date=time.strftime('%Y%m%d'),
                        start_date= (time - datetime.timedelta(days=400)).strftime('%Y%m%d')
                        )
            
            if price_data.empty:
                fig, ax = plt.subplots(figsize=(40, 4))
                ax.set_title(f"{stock_name} ({stock_code}) - 数据获取失败", fontsize=14)
                plt.tight_layout()
                plt.show()
                return
            
            # 计算技术指标
            price_data['LC'] = price_data['close'].shift(1)
            price_data['CLOSE_LC'] = price_data['close'] - price_data['LC']
            
            # 计算VR指标
            av = price_data['volume'].where(price_data['close'] > price_data['LC'], 0)
            bv = price_data['volume'].where(price_data['close'] < price_data['LC'], 0)
            cv = price_data['volume'].where(price_data['close'] == price_data['LC'], 0)
            price_data['vr'] = (av.rolling(24).sum() + cv.rolling(24).sum()/2) / \
                              (bv.rolling(24).sum() + cv.rolling(24).sum()/2 + 1e-7) * 100
            
            # 计算MACD
            fast_ema = price_data['close'].ewm(span=12, adjust=False).mean()
            slow_ema = price_data['close'].ewm(span=26, adjust=False).mean()
            price_data['diff'] = fast_ema - slow_ema
            price_data['dea'] = price_data['diff'].ewm(span=9, adjust=False).mean()
            price_data['macd'] = 2 * (price_data['diff'] - price_data['dea'])
            
            # 计算各周期方差 (VAR)
            closes = price_data['close'].values.astype(float)
            periods = [5, 8, 13, 21, 34, 55]  # 方差计算周期
            
            for period in periods:
                # 计算滚动方差
                price_data[f'var_{period}'] = price_data['close'].rolling(window=period).var(ddof=1) / (closes ** 2)
            
            # 计算威廉指标WR5和WR55
            price_data['wr5'] = calculate_williams_r(price_data, period=5)
            price_data['wr55'] = calculate_williams_r(price_data, period=55)
            
            # 创建整数索引（去除非交易日间隙）
            price_data['index_num'] = range(len(price_data))
            
            # 计算线性回归通道
            close_prices = price_data['close'].values
            high_prices = price_data['high'].values
            low_prices = price_data['low'].values
            
            try:
                analyzer = AdaptiveRegressionChannel()
                regression_results = analyzer.get_adaptive_regression(close_prices, high_prices, low_prices)
                regression = regression_results['regression']
                start_info = regression_results['start_info']
                all_high_points = regression_results['all_high_points']
                all_low_points = regression_results['all_low_points']
                
                # 获取回归通道数据
                start_idx = regression['start_index']
                end_idx = regression['end_index']
                regression_line = regression['regression_line']
                upper_band = regression['upper_band']
                lower_band = regression['lower_band']
                
                has_regression = True
            except Exception as e:
                print(f"计算回归通道时出错: {e}")
                has_regression = False
            
            # 创建4个子图（K线图 + 2个指标）
            fig = plt.figure(figsize=(14, 8))
            gs = gridspec.GridSpec(4, 1, height_ratios=[5, 2, 2, 2], hspace=0.1)
            ax1 = plt.subplot(gs[0])  # K线图
            ax2 = plt.subplot(gs[1], sharex=ax1)  # MACD和VR合并
            ax3 = plt.subplot(gs[2], sharex=ax1)  # VAR (方差)
            ax4 = plt.subplot(gs[3], sharex=ax1)  # WR (威廉指标)
            
            # 绘制K线图
            candle_width = 0.8
            up = price_data[price_data.close >= price_data.open]
            down = price_data[price_data.close < price_data.open]
            
            # 上涨K线
            ax1.bar(up['index_num'], up.close - up.open, candle_width, 
                   bottom=up.open, color='red', zorder=3)
            ax1.bar(up['index_num'], up.high - up.close, 0.15, 
                   bottom=up.close, color='red', zorder=3)
            ax1.bar(up['index_num'], up.low - up.open, 0.15, 
                   bottom=up.open, color='red', zorder=3)
            
            # 下跌K线
            ax1.bar(down['index_num'], down.close - down.open, candle_width, 
                   bottom=down.open, color='green', zorder=3)
            ax1.bar(down['index_num'], down.high - down.open, 0.15, 
                   bottom=down.open, color='green', zorder=3)
            ax1.bar(down['index_num'], down.low - down.close, 0.15, 
                   bottom=down.close, color='green', zorder=3)
            
            # 绘制成交量（在K线图上，使用次坐标轴）
            ax1_vol = ax1.twinx()  # 创建成交量次坐标轴
            # 计算成交量颜色（与K线颜色一致）
            vol_colors = ['red' if close >= open else 'green' 
                         for close, open in zip(price_data['close'], price_data['open'])]
            # 绘制成交量柱状图，设置透明度
            ax1_vol.bar(price_data['index_num'], price_data['volume'], 
                       color=vol_colors, alpha=0.3, width=0.8, zorder=1)
            ax1_vol.set_ylabel('成交量', fontsize=10)
            # 设置成交量坐标轴在右侧
            ax1_vol.yaxis.set_label_position("right")
            ax1_vol.yaxis.tick_right()
            # 计算并绘制均线
            price_data['MA5'] = price_data['close'].rolling(window=5).mean()
            price_data['MA10'] = price_data['close'].rolling(window=10).mean()
            price_data['MA20'] = price_data['close'].rolling(window=20).mean()
            
            ax1.plot(price_data['index_num'], price_data['MA5'], 'b-', linewidth=1.2, label='5日均线', zorder=2)
            ax1.plot(price_data['index_num'], price_data['MA10'], 'm-', linewidth=1.2, label='10日均线', zorder=2)
            ax1.plot(price_data['index_num'], price_data['MA20'], 'c-', linewidth=1.2, label='20日均线', zorder=2)
            
            # 绘制线性回归通道（如果计算成功）
            if has_regression:
                # 创建回归通道的x坐标
                x_reg = np.arange(start_idx, end_idx + 1)
                
                # 绘制回归线和通道
                ax1.plot(x_reg, regression_line, color='#3498db', linewidth=2.5, label='回归线', zorder=4)
                ax1.plot(x_reg, upper_band, color='#e74c3c', linestyle='--', linewidth=2, label='上轨', zorder=4)
                ax1.plot(x_reg, lower_band, color='#2ecc71', linestyle='--', linewidth=2, label='下轨', zorder=4)
                
                # 填充通道区域
                ax1.fill_between(x_reg, upper_band, lower_band, alpha=0.1, color='#95a5a6', label='回归通道', zorder=3)
                
                # 标记极值点
                high_indices = [i for i, _ in all_high_points]
                high_values = [v for _, v in all_high_points]
                ax1.scatter(high_indices, high_values, color='#e74c3c', marker='v', s=60, 
                           label=f'高点({len(all_high_points)}个)', zorder=5, alpha=0.7)
                
                low_indices = [i for i, _ in all_low_points]
                low_values = [v for _, v in all_low_points]
                ax1.scatter(low_indices, low_values, color='#2ecc71', marker='^', s=60, 
                           label=f'低点({len(all_low_points)}个)', zorder=5, alpha=0.7)
                
                # 标记回归起点
                ax1.scatter([start_info['start_index']], [start_info['start_price']], 
                           color='#3498db', marker='*', s=200, label='回归起点', zorder=6)
                
                # 添加垂直线标记回归起点位置
                ax1.axvline(x=start_info['start_index'], color='#9b59b6', linestyle=':', alpha=0.8, 
                           linewidth=2, label='起点位置', zorder=4)
                
                # 在价格图左上角添加回归信息文本框
                current_gap = abs(close_prices[-1] - lower_band[-1])
                relative_gap = current_gap / close_prices[-1]
                
                stats_text = (
                    f"斜率: {regression['slope']:.6f}\n"
                    f"R²: {regression['r_squared']:.4f}\n"
                    f"质量: {start_info['quality_score']:.4f}\n"
                    f"下轨: {lower_band[-1]:.2f}\n"
                    f"距下轨: {current_gap:.2f}({relative_gap:.2%})"
                )
                ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
                        verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
                                                          facecolor="lightblue", 
                                                          alpha=0.8))
            
            # 设置K线图标题和标签
            last_close = price_data['close'].iloc[-1]
            last_date = price_data.index[-1].strftime('%Y-%m-%d')
            
            title = f"{stock_name} ({stock_code}) - 最新价: {last_close:.2f} ({last_date})"
            ax1.set_title(title, fontsize=16, fontweight='bold')
            ax1.set_ylabel('价格')
            ax1.legend(loc='upper left', fontsize=8)
            ax1.grid(True, linestyle='--', alpha=0.6)
            
            # 绘制MACD和VR指标（合并到一个图，双Y轴）
            # 创建第二个Y轴
            ax2_vr = ax2.twinx()
            
            # 绘制MACD（主Y轴）
            colors_macd = ['red' if val >= 0 else 'green' for val in price_data['macd']]
            ax2.bar(price_data['index_num'], price_data['macd'], color=colors_macd, width=0.8, alpha=0.7, label='MACD')
            ax2.plot(price_data['index_num'], price_data['diff'], 'blue', linewidth=1.5, label='DIFF')
            ax2.plot(price_data['index_num'], price_data['dea'], 'orange', linewidth=1.5, label='DEA')
            ax2.axhline(0, color='gray', linestyle='-', linewidth=0.7)
            ax2.set_ylabel('MACD', color='blue')
            ax2.tick_params(axis='y', labelcolor='blue')
            
            # 绘制VR（次Y轴）
            ax2_vr.plot(price_data['index_num'], price_data['vr'], 'purple', linewidth=1.5, label='VR')
            ax2_vr.axhline(100, color='gray', linestyle='--', linewidth=0.7)
            ax2_vr.set_ylabel('VR', color='purple')
            ax2_vr.tick_params(axis='y', labelcolor='purple')
            
            # 添加图例
            lines1, labels1 = ax2.get_legend_handles_labels()
            lines2, labels2 = ax2_vr.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)
            
            ax2.grid(True, linestyle='--', alpha=0.4)
            ax2.set_title('MACD 和 VR 指标', fontsize=12)
            
            # 绘制VAR（方差）指标
            colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow']
            for i, period in enumerate(periods):
                ax3.plot(price_data['index_num'], price_data[f'var_{period}'], 
                         color=colors[i % len(colors)], 
                         linewidth=1.2, 
                         label=f'{period}日方差')
            
            ax3.set_ylabel('VAR')
            ax3.legend(loc='upper left', fontsize=8)
            ax3.grid(True, linestyle='--', alpha=0.4)
            ax3.set_title('方差指标', fontsize=12)
            
            # 绘制威廉指标WR（WR5和WR55在同一图上）
            ax4.plot(price_data['index_num'], price_data['wr5'], 'blue', linewidth=1.5, label='WR5')
            ax4.plot(price_data['index_num'], price_data['wr55'], 'red', linewidth=1.5, label='WR55')
            # 添加威廉指标参考线
            ax4.axhline(-20, color='gray', linestyle='--', linewidth=0.7, alpha=0.7)
            ax4.axhline(-80, color='gray', linestyle='--', linewidth=0.7, alpha=0.7)
            ax4.set_ylabel('WR')
            ax4.set_xlabel('交易日序列')
            ax4.legend(loc='upper left', fontsize=8)
            ax4.grid(True, linestyle='--', alpha=0.4)
            ax4.set_title('威廉指标', fontsize=12)
            # 设置WR y轴范围（威廉指标通常在0到-100之间）
            ax4.set_ylim(-100, 0)
            
            # 设置x轴刻度（只显示在最后一个子图）
            n = len(price_data)
            step = max(1, n // 10)
            xticks = list(range(0, n, step))
            if n-1 not in xticks:
                xticks.append(n-1)
            xticklabels = [price_data.index[i].strftime('%m-%d') for i in xticks]
            ax4.set_xticks(xticks)
            ax4.set_xticklabels(xticklabels, rotation=45)
            
            # 设置K线图y轴范围
            y_min = price_data[['low', 'MA5', 'MA10', 'MA20']].min().min()
            y_max = price_data[['high', 'MA5', 'MA10', 'MA20']].max().max()
            
            # 如果有回归通道，调整y轴范围以包含回归通道
            if has_regression:
                y_min = min(y_min, np.min(lower_band))
                y_max = max(y_max, np.max(upper_band))
            
            ax1.set_ylim(y_min * 0.98, y_max * 1.02)
            
            # 隐藏上方子图的x轴标签
            plt.setp(ax1.get_xticklabels(), visible=False)
            plt.setp(ax2.get_xticklabels(), visible=False)
            plt.setp(ax3.get_xticklabels(), visible=False)
            
            plt.tight_layout()
            plt.show()
    
    # 定义按钮回调函数
    def on_next_btn(b):
        nonlocal current_index
        current_index = (current_index + 1) % len(stock_codes)
        index_label.value = f"股票: {current_index+1}/{len(stock_codes)}"
        update_chart()
    
    def on_prev_btn(b):
        nonlocal current_index
        current_index = (current_index - 1) % len(stock_codes)
        current_index = current_index if current_index >= 0 else len(stock_codes) - 1
        index_label.value = f"股票: {current_index+1}/{len(stock_codes)}"
        update_chart()
    
    # 绑定按钮事件
    prev_btn.on_click(on_prev_btn)
    next_btn.on_click(on_next_btn)
    
    # 创建控制面板
    controls = widgets.HBox([prev_btn, index_label, next_btn])
    
    # 初始显示
    update_chart()
    
    # 显示所有控件
    display(controls, output)



if __name__ == "__main__":
       
    main()

# In[ ]:








class StategyLineregressionStrategy(BaseStrategy):
    """基于stategy_lineregression的策略"""

    def __init__(self, data: pd.DataFrame, params: dict = None):
        super().__init__(data, params)

    def generate_signals(self):
        """生成交易信号"""
        df = self.data
        if len(df) < 2:
            return self.signals

        close = df['close']
        ma_short = close.rolling(5).mean()
        ma_long = close.rolling(20).mean()

        for i in range(1, len(df)):
            if pd.isna(ma_long.iloc[i]):
                continue
            if ma_short.iloc[i] > ma_long.iloc[i] and ma_short.iloc[i-1] <= ma_long.iloc[i-1]:
                self._record_signal(timestamp=df.index[i], action='buy', price=float(close.iloc[i]))
            elif ma_short.iloc[i] < ma_long.iloc[i] and ma_short.iloc[i-1] >= ma_long.iloc[i-1]:
                self._record_signal(timestamp=df.index[i], action='sell', price=float(close.iloc[i]))

        return self.signals
