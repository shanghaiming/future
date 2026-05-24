#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from core.base_strategy import BaseStrategy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from scipy.signal import argrelextrema
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和图表样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class SpatiotemporalBoxDetector:
    """
    时空矩形箱体检测器 - 同时考虑空间位置和时间连续性
    """
    
    def __init__(self, data, lookback_period=100):
        """
        初始化
        
        参数:
        data: K线数据 (DataFrame，包含'open', 'high', 'low', 'close'列)
        lookback_period: 回看周期，只分析最近这段时间的数据
        """
        self.data = data.copy()
        self.lookback_period = lookback_period
        self.boxes = []
        
        # 确保数据按时间排序
        self.data = self.data.sort_index()
        
        # 只保留最近的数据
        if len(self.data) > lookback_period:
            self.recent_data = self.data.iloc[-lookback_period:]
        else:
            self.recent_data = self.data
    
    def find_pivot_points(self, window=5):
        """
        寻找关键高低点(pivot points)
        """
        data = self.recent_data
        
        # 寻找局部高点
        high_indices = argrelextrema(data['high'].values, np.greater, order=window)[0]
        high_pivots = [(data.index[i], data['high'].iloc[i]) for i in high_indices]
        
        # 寻找局部低点
        low_indices = argrelextrema(data['low'].values, np.less, order=window)[0]
        low_pivots = [(data.index[i], data['low'].iloc[i]) for i in low_indices]
        
        return high_pivots, low_pivots
    
    def cluster_prices_with_dbscan(self, price_points, eps_percent=0.02, min_samples=2):
        """
        使用DBSCAN聚类算法识别价格水平
        """
        if len(price_points) < min_samples:
            return []
        
        # 提取价格
        prices = np.array([p[1] for p in price_points]).reshape(-1, 1)
        
        # 计算eps（基于价格百分比）
        avg_price = np.mean(prices)
        eps = avg_price * eps_percent
        
        # 使用DBSCAN聚类
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(prices)
        
        # 找出每个簇的中心点
        clusters = {}
        for i, label in enumerate(labels):
            if label == -1:  # 噪声点（突出的高低点）
                continue
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(price_points[i])
        
        # 计算每个簇的平均价格和强度
        cluster_levels = []
        for label, points in clusters.items():
            cluster_prices = [p[1] for p in points]
            cluster_mean = np.mean(cluster_prices)
            cluster_std = np.std(cluster_prices)
            
            # 计算簇的强度（点数）
            strength = len(points)
            
            cluster_levels.append({
                'price': cluster_mean,
                'points': points,
                'strength': strength,
                'std': cluster_std
            })
        
        # 按强度排序
        cluster_levels.sort(key=lambda x: x['strength'], reverse=True)
        return cluster_levels
    
    def find_continuous_box(self, min_cluster_strength=2, max_height_percent=0.15, include_last_n=5):
        """
        寻找时间上连续的矩形箱体
        """
        self.boxes = []
        data = self.recent_data
        
        # 寻找高低点
        high_pivots, low_pivots = self.find_pivot_points(window=5)
        
        # 聚类高低点
        resistance_levels = self.cluster_prices_with_dbscan(high_pivots)
        support_levels = self.cluster_prices_with_dbscan(low_pivots)
        
        # 过滤强度不足的簇
        resistance_levels = [r for r in resistance_levels if r['strength'] >= min_cluster_strength]
        support_levels = [s for s in support_levels if s['strength'] >= min_cluster_strength]
        
        # 如果没有足够的簇，直接返回
        if not resistance_levels or not support_levels:
            return self.boxes
        
        # 寻找可能的箱体组合
        for resistance in resistance_levels[:3]:  # 取前3个最强的阻力位
            for support in support_levels[:3]:    # 取前3个最强的支撑位
                # 计算箱体高度
                height = resistance['price'] - support['price']
                avg_price = (resistance['price'] + support['price']) / 2
                height_percent = height / avg_price
                
                # 检查高度是否合理
                if height_percent > max_height_percent:
                    continue
                
                # 确定箱体的时间范围
                resistance_dates = [p[0] for p in resistance['points']]
                support_dates = [p[0] for p in support['points']]
                all_dates = resistance_dates + support_dates
                
                start_date = min(all_dates)
                end_date = max(all_dates)
                
                # 确保包含最新数据
                if end_date < data.index[-1]:
                    end_date = data.index[-1]
                
                # 检查最后几根K线是否在箱体内
                last_n_data = data.iloc[-include_last_n:]
                all_in_box = all(
                    (support['price'] <= last_n_data['low'].iloc[i]) and 
                    (last_n_data['high'].iloc[i] <= resistance['price'])
                    for i in range(len(last_n_data))
                )
                
                if not all_in_box:
                    continue
                
                # 检查时间连续性 - 是否有明显的中断
                if not self._check_time_continuity(data, start_date, end_date, resistance['price'], support['price']):
                    continue
                
                # 检查是否有突出的高低点突破箱体
                if self._has_breakouts(data, start_date, end_date, resistance['price'], support['price']):
                    continue
                
                # 计算质量评分
                quality = self._calculate_box_quality(resistance, support, height_percent)
                
                self.boxes.append({
                    'start': start_date,
                    'end': end_date,
                    'resistance': resistance['price'],
                    'support': support['price'],
                    'height': height,
                    'height_percent': height_percent,
                    'quality': quality,
                    'resistance_strength': resistance['strength'],
                    'support_strength': support['strength'],
                    'resistance_std': resistance['std'],
                    'support_std': support['std']
                })
        
        # 按质量排序
        self.boxes.sort(key=lambda x: x['quality'], reverse=True)
        
        # 只保留最佳结果
        self.boxes = self.boxes[:1]
        
        return self.boxes
    
    def _check_time_continuity(self, data, start_date, end_date, resistance, support):
        """
        检查时间连续性 - 确保箱体在时间上是连续的
        """
        # 获取箱体时间范围内的数据
        box_data = data.loc[start_date:end_date]
        
        # 计算在箱体内的K线比例
        in_box_count = 0
        for i in range(len(box_data)):
            high = box_data['high'].iloc[i]
            low = box_data['low'].iloc[i]
            
            # 检查K线是否在箱体内
            if support <= low and high <= resistance:
                in_box_count += 1
        
        # 计算在箱体内的比例
        in_box_ratio = in_box_count / len(box_data)
        
        # 如果大部分K线都在箱体内，说明连续性良好
        return in_box_ratio >= 0.7  # 70%的K线在箱体内
    
    def _has_breakouts(self, data, start_date, end_date, resistance, support):
        """
        检查是否有突出的高低点突破箱体
        """
        # 获取箱体时间范围内的数据
        box_data = data.loc[start_date:end_date]
        
        # 检查是否有明显的高点突破
        max_high = box_data['high'].max()
        if max_high > resistance * 1.02:  # 超过阻力线2%
            return True
        
        # 检查是否有明显的低点突破
        min_low = box_data['low'].min()
        if min_low < support * 0.98:  # 低于支撑线2%
            return True
        
        # 检查是否有连续多个K线突破箱体
        breakout_count = 0
        for i in range(len(box_data)):
            high = box_data['high'].iloc[i]
            low = box_data['low'].iloc[i]
            
            # 检查是否有突破
            if high > resistance * 1.01 or low < support * 0.99:
                breakout_count += 1
            else:
                breakout_count = 0
            
            # 如果有连续3个K线突破，认为是明显中断
            if breakout_count >= 3:
                return True
        
        return False
    
    def _calculate_box_quality(self, resistance, support, height_percent):
        """
        计算箱体质量评分
        """
        # 簇强度越高，质量越高
        resistance_strength_score = min(resistance['strength'] / 5, 1.0)
        support_strength_score = min(support['strength'] / 5, 1.0)
        
        # 簇标准差越小，质量越高（价格更集中）
        resistance_std_score = 1.0 - min(resistance['std'] / (resistance['price'] * 0.01), 1.0)
        support_std_score = 1.0 - min(support['std'] / (support['price'] * 0.01), 1.0)
        
        # 高度百分比适中（1%-15%）
        height_score = 1.0 - abs(height_percent - 0.08) / 0.08
        
        # 综合评分
        quality = (
            resistance_strength_score * 0.25 +
            support_strength_score * 0.25 +
            resistance_std_score * 0.15 +
            support_std_score * 0.15 +
            height_score * 0.2
        )
        
        return min(max(quality, 0), 1.0)
    
    def get_latest_box(self):
        """
        获取最新的箱体
        """
        if not self.boxes:
            return None
        return self.boxes[0]
    
    def plot_with_boxes(self, title="K线图与时空矩形箱体识别"):
        """
        绘制K线图并标记矩形箱体区域
        """
        if not self.boxes:
            print("未检测到符合条件的时空连续矩形箱体")
            return
        
        # 创建图表
        fig, ax = plt.subplots(1, 1, figsize=(15, 8))
        
        # 准备数据
        data = self.recent_data
        dates = data.index
        opens = data['open']
        highs = data['high']
        lows = data['low']
        closes = data['close']
        
        # 计算涨跌颜色
        colors = ['red' if close >= open else 'green' 
                 for open, close in zip(opens, closes)]
        
        # 绘制K线
        for i, (date, open_price, high, low, close, color) in enumerate(
                zip(dates, opens, highs, lows, closes, colors)):
            
            # 绘制影线
            ax.plot([date, date], [low, high], color='black', linewidth=1)
            
            # 绘制K线实体
            body_height = abs(close - open_price)
            if body_height > 0:
                ax.bar(date, body_height, bottom=min(open_price, close), 
                       color=color, width=0.8, alpha=0.7)
        
        # 绘制矩形箱体
        for i, box in enumerate(self.boxes):
            start_date = box['start']
            end_date = box['end']
            resistance = box['resistance']
            support = box['support']
            
            # 绘制矩形
            rect = plt.Rectangle(
                (mdates.date2num(start_date), support),
                mdates.date2num(end_date) - mdates.date2num(start_date),
                box['height'],
                linewidth=2, edgecolor='purple', facecolor='yellow', alpha=0.2
            )
            ax.add_patch(rect)
            
            # 绘制支撑线和阻力线
            ax.hlines(resistance, start_date, end_date, colors='red', 
                      linestyles='-', linewidth=2, label='阻力线')
            ax.hlines(support, start_date, end_date, colors='blue', 
                      linestyles='-', linewidth=2, label='支撑线')
            
            # 标注信息
            mid_date = start_date + (end_date - start_date) / 2
            ax.text(mid_date, resistance, 
                    f'阻力: {resistance:.2f}\n强度: {box["resistance_strength"]}', 
                    ha='center', va='bottom', fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
            
            ax.text(mid_date, support, 
                    f'支撑: {support:.2f}\n强度: {box["support_strength"]}', 
                    ha='center', va='top', fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
            
            # 标记最后几根K线
            last_n = 5
            if len(data) > last_n:
                last_dates = data.index[-last_n:]
                for date in last_dates:
                    ax.axvline(x=date, color='orange', linestyle=':', alpha=0.7, linewidth=1)
        
        # 设置图表属性
        ax.set_title(f'{title} (最近{self.lookback_period}期数据)', fontsize=16)
        ax.set_ylabel('价格', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 自动调整x轴日期显示
        plt.gcf().autofmt_xdate()
        plt.tight_layout()
        plt.show()
    
    def print_detection_results(self):
        """
        打印检测结果
        """
        if not self.boxes:
            print("未检测到符合条件的时空连续矩形箱体")
            return
        
        print(f"在最近{self.lookback_period}期数据中检测到时空连续矩形箱体:")
        print("=" * 60)
        
        for i, box in enumerate(self.boxes):
            print(f"时空连续矩形箱体 (质量评分: {box['quality']:.2f}):")
            print(f"  时间范围: {box['start']} 到 {box['end']}")
            print(f"  支撑位: {box['support']:.4f} (强度: {box['support_strength']}, 标准差: {box['support_std']:.4f})")
            print(f"  阻力位: {box['resistance']:.4f} (强度: {box['resistance_strength']}, 标准差: {box['resistance_std']:.4f})")
            print(f"  箱体高度: {box['height']:.4f} ({box['height_percent']*100:.2f}%)")
            print()

# 使用示例
if __name__ == "__main__":
    # 数据加载
    ts_code = "002876.SZ"
    csv_path = fr'E:\stock\csv_version\analysis_results\{ts_code}_analysis.csv'    
    df = pd.read_csv(csv_path, index_col='trade_date', parse_dates=['trade_date']).sort_index(ascending=True).tail(200)
    
    # 创建检测器实例
    detector = SpatiotemporalBoxDetector(df, lookback_period=100)
    
    # 检测时空连续矩形箱体
    boxes = detector.find_continuous_box(
        min_cluster_strength=2,    # 最小簇强度
        max_height_percent=0.15,   # 最大箱体高度百分比
        include_last_n=5           # 必须包含最后5根K线
    )
    
    # 打印结果
    detector.print_detection_results()
    
    # 获取最新箱体
    latest = detector.get_latest_box()
    if latest:
        print(f"最佳时空连续矩形箱体结束于: {latest['end']}")
    
    # 绘制图表
    detector.plot_with_boxes("时空连续矩形箱体识别结果")

# In[ ]:






class RectangleStrategy(BaseStrategy):
    """基于rectangle的策略"""
    
    def __init__(self, data, params=None):
        super().__init__(data, params)
        # 初始化代码
        self.name = "RectangleStrategy"
        self.description = "基于rectangle的策略"
        
    def calculate_signals(self):
        """计算交易信号"""
        # 策略逻辑
        return df
        
    def generate_signals(self):
        """矩形整理策略生成交易信号"""
        import numpy as np
        df = self.data
        window = 20
        for i in range(window, len(df)):
            recent = df['high'].iloc[i-window:i]
            resistance = recent.max()
            support = df['low'].iloc[i-window:i].min()
            price = float(df['close'].iloc[i])
            sym = df['symbol'].iloc[i] if 'symbol' in df.columns else 'DEFAULT'
            rng = resistance - support
            if rng < 1e-10:
                continue
            # 突破阻力
            if price > resistance * 1.01:
                self._record_signal(df.index[i], 'buy', sym, price)
            # 跌破支撑
            elif price < support * 0.99:
                self._record_signal(df.index[i], 'sell', sym, price)
        return self.signals
