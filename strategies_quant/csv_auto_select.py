import errno
# from h11 import ERROR  # 注释掉可能缺失的依赖
import pandas as pd
import numpy as np

# 策略改造: 添加BaseStrategy导入
try:
    from core.base_strategy import BaseStrategy
except ImportError:
    from core.base_strategy import BaseStrategy
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import json
import shutil
import os
import warnings
from tqdm import tqdm
from sklearn.cluster import KMeans
from scipy.signal import argrelextrema

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
            logging.FileHandler('app.log', encoding='UTF-8'),
            logging.StreamHandler()  # 同时输出到控制台
        ]
)
logger = logging.getLogger(__name__)
# ================= 配置系统 =================
def load_config():
    """加载系统配置 - 修正数据路径指向实际data目录"""
    import os
    from pathlib import Path
    
    # 项目根目录
    strategies_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(strategies_dir)  # quant_trade-main目录
    
    return {
        "data_paths": {
            # 原始数据目录（也作为主要数据源）
            "daily_raw": Path(os.path.join(project_root, 'data', 'daily_data2')),
            "weekly_raw": Path(os.path.join(project_root, 'data', 'week_data2')),
            # 分析结果目录（用于读取） - 修改为指向原始数据目录
            "daily": Path(os.path.join(project_root, 'data', 'daily_data2')),
            "output": Path(os.path.join(project_root, 'dashboard', 'output'))
        },
        "target": Path(os.path.join(project_root, 'dashboard', 'output', 'selected_stocks')),
        "file_suffix": ".csv",  # 修改为.csv以匹配原始数据文件
        "system_params": {
            "max_workers": 8,
            "chunk_size": 200
        }
    }

# ================= 数据加载系统 =================
def load_single_stock_data(config, code):
    """加载单只股票数据 - 支持多个数据源"""
    # 尝试多个数据源
    data_sources = [
        config["data_paths"]["daily"] / f"{code}{config['file_suffix']}",  # 分析结果目录
        config["data_paths"]["daily_raw"] / f"{code}.csv",  # 原始日线数据
    ]
    
    for file_path in data_sources:
        if not file_path.exists():
            continue
            
        try:
            df = pd.read_csv(file_path, parse_dates=['trade_date'], index_col='trade_date')
            
            # 确保必要的列存在
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in df.columns:
                    # 尝试从其他列名映射
                    if '开盘价' in df.columns and col == 'open':
                        df[col] = df['开盘价']
                    elif 'open' in df.columns and col == 'open':  # 英文列名已存在
                        pass
                    elif '最高价' in df.columns and col == 'high':
                        df[col] = df['最高价']
                    elif 'high' in df.columns and col == 'high':  # 英文列名已存在
                        pass
                    elif '最低价' in df.columns and col == 'low':
                        df[col] = df['最低价']
                    elif 'low' in df.columns and col == 'low':  # 英文列名已存在
                        pass
                    elif '收盘价' in df.columns and col == 'close':
                        df[col] = df['收盘价']
                    elif 'close' in df.columns and col == 'close':  # 英文列名已存在
                        pass
                    elif '成交量' in df.columns and col == 'volume':
                        df[col] = df['成交量']
                    elif 'vol' in df.columns and col == 'volume':  # 原始数据使用vol
                        df[col] = df['vol']
                    elif 'amount' in df.columns and col == 'volume':  # 金额列备用
                        df[col] = df['amount']
                    else:
                        df[col] = np.nan
            
            df = df.dropna(subset=required_cols)
            if not df.empty:
                logger.info(f"成功从 {file_path.name} 加载数据: {code}")
                return df
                
        except Exception as e:
            logger.warning(f"从 {file_path} 加载失败: {e}")
            continue
    
    logger.warning(f"所有数据源都无法加载股票: {code}")
    return pd.DataFrame()


class KLineCenterAnalyzer:
    def __init__(self, window=5, shadow_ratio=0.7, min_zone_separation=0.02, 
                 max_zone_width_ratio=0.1, entity_size_threshold=0.05):
        self.window = window
        self.shadow_ratio = shadow_ratio
        self.min_zone_separation = min_zone_separation
        self.max_zone_width_ratio = max_zone_width_ratio
        self.entity_size_threshold = entity_size_threshold
    
    def is_bullish(self, open_price, close_price):
        """判断是否为阳线"""
        return close_price >= open_price
    
    def calculate_shadow_ratio(self, open_price, high_price, low_price, close_price):
        """计算影线比例"""
        body_size = abs(close_price - open_price)
        upper_shadow = high_price - max(open_price, close_price)
        lower_shadow = min(open_price, close_price) - low_price
        
        if body_size == 0:
            return upper_shadow, lower_shadow, 0
        
        return upper_shadow, lower_shadow, max(upper_shadow, lower_shadow) / body_size
    
    def find_reversal_points(self, df):
        """寻找K线反转点"""
        reversal_points = []
        opens = df['open'].values
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        
        for i in range(1, len(opens)-1):
            # 检查颜色反转
            prev_bullish = self.is_bullish(opens[i-1], closes[i-1])
            curr_bullish = self.is_bullish(opens[i], closes[i])
            next_bullish = self.is_bullish(opens[i+1], closes[i+1])
            
            # 颜色反转点
            if (prev_bullish != curr_bullish) or (curr_bullish != next_bullish):
                reversal_points.append(('color_reversal', i, closes[i]))
            
            # 检查影线反转
            _, _, ratio_curr = self.calculate_shadow_ratio(
                opens[i], highs[i], lows[i], closes[i])
            
            # 影线比例超过阈值，可能是反转点
            if ratio_curr > self.shadow_ratio:
                upper_shadow, lower_shadow, _ = self.calculate_shadow_ratio(
                    opens[i], highs[i], lows[i], closes[i])
                
                if upper_shadow > lower_shadow:
                    reversal_points.append(('upper_shadow', i, highs[i]))
                else:
                    reversal_points.append(('lower_shadow', i, lows[i]))
        
        return reversal_points
    
    def find_volume_price_points(self, df):
        """寻找成交量极值点对应的K线价格"""
        volumes = df['volume'].values
        closes = df['close'].values
        
        # 寻找成交量局部高点
        volume_high_idx = argrelextrema(volumes, np.greater, order=self.window)[0]
        
        volume_points = []
        for idx in volume_high_idx:
            volume_points.append(('volume_price', idx, closes[idx]))
        
        return volume_points
    
    def cluster_prices(self, points, n_clusters=3):
        """对价格点进行聚类，返回聚类中心"""
        if not points:
            return []
            
        prices = np.array([point[2] for point in points]).reshape(-1, 1)
        
        if len(prices) <= n_clusters:
            return sorted([p[0] for p in prices])
        
        kmeans = KMeans(n_clusters=min(n_clusters, len(prices)), random_state=0).fit(prices)
        centers = sorted([center[0] for center in kmeans.cluster_centers_])
        return centers
    
    def find_centers(self, df):
        """找出中枢水平线"""
        reversal_points = self.find_reversal_points(df)
        volume_points = self.find_volume_price_points(df)
        
        all_points = reversal_points + volume_points
        
        if not all_points:
            return [], [], []
        
        center_lines = self.cluster_prices(all_points, 3)
        
        return center_lines, reversal_points, volume_points
    
    def calculate_center_zones(self, df, center_lines):
        """计算中枢区域，确保不重叠"""
        center_zones = []
        opens = df['open'].values
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        
        price_range = np.max(highs) - np.min(lows)
        min_separation = price_range * self.min_zone_separation
        max_zone_width = price_range * self.max_zone_width_ratio
        
        entity_sizes = np.abs(closes - opens)
        entity_size_threshold = np.percentile(entity_sizes, 95)
        
        for center_line in center_lines:
            crossing_k_lines = []
            for i in range(len(df)):
                if lows[i] <= center_line <= highs[i]:
                    entity_high = max(opens[i], closes[i])
                    entity_low = min(opens[i], closes[i])
                    entity_size = entity_high - entity_low
                    
                    if entity_size <= entity_size_threshold:
                        crossing_k_lines.append((entity_high, entity_low))
            
            if crossing_k_lines:
                entity_highs = [h for h, l in crossing_k_lines]
                entity_lows = [l for h, l in crossing_k_lines]
                
                zone_high = np.mean(entity_highs)
                zone_low = np.mean(entity_lows)
                
                min_zone_height = price_range * 0.005
                if zone_high - zone_low < min_zone_height:
                    mid_point = (zone_high + zone_low) / 2
                    zone_high = mid_point + min_zone_height / 2
                    zone_low = mid_point - min_zone_height / 2
                
                if zone_high - zone_low > max_zone_width:
                    mid_point = (zone_high + zone_low) / 2
                    zone_high = mid_point + max_zone_width / 2
                    zone_low = mid_point - max_zone_width / 2
                
                overlap_found = False
                for i, (existing_center, existing_low, existing_high, existing_count) in enumerate(center_zones):
                    if not (zone_high < existing_low or zone_low > existing_high):
                        # 发现重叠，合并区域
                        merged_low = min(zone_low, existing_low)
                        merged_high = max(zone_high, existing_high)
                        
                        if merged_high - merged_low > max_zone_width:
                            mid_point = (merged_low + merged_high) / 2
                            merged_low = mid_point - max_zone_width / 2
                            merged_high = mid_point + max_zone_width / 2
                        
                        merged_center = (center_line + existing_center) / 2
                        merged_count = len(crossing_k_lines) + existing_count
                        
                        center_zones[i] = (merged_center, merged_low, merged_high, merged_count)
                        overlap_found = True
                        break
                
                if not overlap_found:
                    center_zones.append((center_line, zone_low, zone_high, len(crossing_k_lines)))
        
        # 确保区域之间有最小分离距离，防止重叠
        center_zones.sort(key=lambda x: x[0])
        
        non_overlapping_zones = []
        for zone in center_zones:
            center_line, zone_low, zone_high, count = zone
            
            if not non_overlapping_zones:
                non_overlapping_zones.append(zone)
                continue
                
            # 检查是否与已有区域重叠
            overlaps = False
            for existing_zone in non_overlapping_zones:
                existing_center, existing_low, existing_high, existing_count = existing_zone
                
                if not (zone_high < existing_low or zone_low > existing_high):
                    # 有重叠，跳过这个区域
                    overlaps = True
                    break
            
            if not overlaps:
                non_overlapping_zones.append(zone)
        
        # 按中心线排序
        non_overlapping_zones.sort(key=lambda x: x[0])
        
        return non_overlapping_zones


class MomentumEnergyAnalyzer:
    def __init__(self, df, symbol="未知标的"):
        self.df = df.copy()
        self.symbol = symbol
        self.center_analyzer = KLineCenterAnalyzer()
        self.calculate_all_indicators()
    
    def calculate_all_indicators(self):
        """一次性计算所有指标"""
        # 1. 计算基础技术指标
        self.calculate_basic_indicators()
        
        # 2. 计算中枢
        self.calculate_centers()
        
        # 3. 计算各种势能
        self.calculate_energy_components()
        
        # 4. 生成信号
        self.generate_signals()
    
    def calculate_basic_indicators(self):
        """计算基础技术指标"""
        # 价格变化
        self.df['price_change'] = self.df['close'].pct_change()
        self.df['log_return'] = np.log(self.df['close'] / self.df['close'].shift(1))
        
        # K线特征
        self.df['body_size'] = abs(self.df['close'] - self.df['open']) / self.df['open']
        self.df['total_range'] = (self.df['high'] - self.df['low']) / self.df['open']
        self.df['upper_shadow'] = (self.df['high'] - np.maximum(self.df['open'], self.df['close'])) / self.df['open']
        self.df['lower_shadow'] = (np.minimum(self.df['open'], self.df['close']) - self.df['low']) / self.df['open']
        self.df['price_position'] = (self.df['close'] - self.df['low']) / (self.df['high'] - self.df['low'])
        
        # 移动平均
        self.df['ma5'] = self.df['close'].rolling(5).mean()
        self.df['ma20'] = self.df['close'].rolling(20).mean()
        self.df['volume_ma5'] = self.df['volume'].rolling(5).mean()
    
    def calculate_centers(self):
        """计算中枢区域"""
        # 使用整个数据集计算中枢
        center_lines, reversal_points, volume_points = self.center_analyzer.find_centers(self.df)
        self.center_zones = self.center_analyzer.calculate_center_zones(self.df, center_lines)
        
        '''print(f"检测到 {len(self.center_zones)} 个中枢区域:")
        for i, (center, low, high, count) in enumerate(self.center_zones):
            print(f"  中枢{i+1}: {center:.3f} [{low:.3f} - {high:.3f}], 穿越K线: {count}")'''
    
    def calculate_center_energy(self):
        """计算基于中枢的势能 - 修正极性"""
        center_energy = []
        
        for i in range(len(self.df)):
            price = self.df['close'].iloc[i]
            total_energy = 0
            
            for center, zone_low, zone_high, strength in self.center_zones:
                # 计算强度权重
                zone_strength = np.log1p(strength)
                
                if price > zone_high:
                    # 在上方 - 正势能（上涨动力）- 修正极性
                    distance_ratio = (price - center) / (zone_high - center) if (zone_high - center) > 0 else 0
                    energy = distance_ratio * zone_strength  # 改为正数
                elif price < zone_low:
                    # 在下方 - 负势能（下跌压力）- 修正极性
                    distance_ratio = (center - price) / (center - zone_low) if (center - zone_low) > 0 else 0
                    energy = -distance_ratio * zone_strength  # 改为负数
                else:
                    # 在中枢内 - 根据位置决定
                    position = (price - zone_low) / (zone_high - zone_low) if (zone_high - zone_low) > 0 else 0.5
                    # 中枢上半部：轻微正势能；下半部：轻微负势能
                    energy = (position - 0.5) * 0.5 * zone_strength
                
                total_energy += energy
            
            center_energy.append(total_energy)
        
        return center_energy
    
    def calculate_breakout_energy(self):
        """计算突破势能"""
        breakout_energy = [0] * len(self.df)
        
        for i in range(1, len(self.df)):
            current = self.df.iloc[i]
            prev = self.df.iloc[i-1]
            
            # 突破前高
            if current['close'] > prev['high'] and current['volume'] > prev['volume']:
                breakout_energy[i] = current['log_return'] * current['volume'] * 2
            # 跌破前低
            elif current['close'] < prev['low'] and current['volume'] > prev['volume']:
                breakout_energy[i] = current['log_return'] * current['volume'] * 2
            # 普通情况
            else:
                breakout_energy[i] = current['log_return'] * current['volume']
        
        return breakout_energy
    
    def calculate_energy_components(self):
        """计算各种势能成分"""
        
        # 1. 基础动能势能
        self.df['kinetic_energy'] = self.df['log_return'] * self.df['volume']
        self.df['potential_energy'] = (self.df['price_position'] - 0.5) * self.df['volume'] * 2
        self.df['mechanical_energy'] = self.df['kinetic_energy'] + self.df['potential_energy']
        self.df['cumulative_mechanical'] = self.df['mechanical_energy'].cumsum()
        
        # 2. 压力支撑势能
        def pressure_support_energy(row):
            pressure = row['upper_shadow'] * row['volume']
            support = row['lower_shadow'] * row['volume']
            if row['close'] > row['open']:
                return support - pressure * 0.5
            else:
                return pressure - support * 0.5
        
        self.df['pressure_support_energy'] = self.df.apply(pressure_support_energy, axis=1)
        self.df['cumulative_pressure_support'] = self.df['pressure_support_energy'].cumsum()
        
        # 3. 趋势势能
        self.df['trend_strength'] = (self.df['ma5'] - self.df['ma20']) / self.df['ma20'] * 100
        self.df['trend_energy'] = self.df['trend_strength'] * self.df['volume'] / self.df['volume_ma5']
        self.df['cumulative_trend'] = self.df['trend_energy'].cumsum()
        
        # 4. 中枢势能（新增）
        self.df['center_energy'] = self.calculate_center_energy()
        self.df['cumulative_center'] = self.df['center_energy'].cumsum()
        
        # 5. 突破势能（保留）
        self.df['breakout_energy'] = self.calculate_breakout_energy()
        self.df['cumulative_breakout'] = self.df['breakout_energy'].cumsum()
        
        # 6. 综合势能（加权组合）
        # 标准化各成分
        energy_components = [
            'cumulative_mechanical', 
            'cumulative_pressure_support', 
            'cumulative_trend', 
            'cumulative_center',
            'cumulative_breakout'
        ]
        
        for col in energy_components:
            if self.df[col].std() > 0:
                self.df[f'normalized_{col}'] = (self.df[col] - self.df[col].mean()) / self.df[col].std()
            else:
                self.df[f'normalized_{col}'] = 0
        
        # 权重分配（中枢势能和突破势能占重要地位）
        weights = {
            'mechanical': 0.3,
            'pressure_support': 0.25, 
            'trend': 0.25,
            'center': 0.0,      # 中枢势能权重最大
            'breakout': 0.2     # 突破势能重要地位
        }
        
        self.df['comprehensive_energy'] = (
            weights['mechanical'] * self.df['normalized_cumulative_mechanical'] +
            weights['pressure_support'] * self.df['normalized_cumulative_pressure_support'] +
            weights['trend'] * self.df['normalized_cumulative_trend'] +
            weights['center'] * self.df['normalized_cumulative_center'] +
            weights['breakout'] * self.df['normalized_cumulative_breakout']
        )
        
        # 势能动量
        self.df['energy_momentum'] = self.df['comprehensive_energy'].diff(3)
    
    def generate_signals(self):
        """生成交易信号"""
        # 势能水平
        self.df['energy_level'] = pd.cut(
            self.df['comprehensive_energy'], 
            bins=[-np.inf, -1, -0.5, 0.5, 1, np.inf],
            labels=['极低', '较低', '中性', '较高', '极高']
        )
        
        # 势能趋势
        self.df['energy_trend'] = np.where(
            self.df['energy_momentum'] > 0.1, '加速上升',
            np.where(self.df['energy_momentum'] < -0.1, '加速下降', '平稳')
        )
        
        # 交易信号（修正逻辑：高势能看跌，低势能看涨）
        conditions = [
            (self.df['energy_level'].isin(['极高', '较高'])) & (self.df['energy_trend'] == '加速上升'),
            (self.df['energy_level'].isin(['极低', '较低'])) & (self.df['energy_trend'] == '加速下降'),
            (self.df['energy_level'].isin(['较高'])) & (self.df['energy_trend'] == '平稳'),
            (self.df['energy_level'].isin(['较低'])) & (self.df['energy_trend'] == '平稳')
        ]
        choices = ['强烈看跌', '强烈看涨', '温和看跌', '温和看涨']
        self.df['trading_signal'] = np.select(conditions, choices, default='中性')
    


# ================= 策略评估（基于最新数据实时判断） =================
def evaluate_buy_conditions(analyzer_df):
    """
    基于最新一根K线的技术指标做买入判断（非统计历史信号）
    返回: (是否满足买入条件, 满足的条件列表, 不满足的原因列表)
    """
    if analyzer_df.empty or len(analyzer_df) < 20:
        return False, [], ["数据不足(需至少20根K线)"]

    latest = analyzer_df.iloc[-1]
    conditions_met = []
    reasons = []

    # --- 条件1: 均线多头排列 (MA5 > MA20 且收盘价在MA5上方) ---
    ma5 = latest.get('ma5')
    ma20 = latest.get('ma20')
    close = latest['close']
    if ma5 is not None and ma20 is not None and not (pd.isna(ma5) or pd.isna(ma20)):
        if close > ma5 > ma20:
            conditions_met.append("均线多头排列")
        else:
            reasons.append("均线未多头排列")

    # --- 条件2: 成交量放大 (当日成交量 > 5日均量) ---
    vol = latest['volume']
    vol_ma5 = latest.get('volume_ma5')
    if vol_ma5 is not None and not pd.isna(vol_ma5) and vol_ma5 > 0:
        if vol > vol_ma5:
            conditions_met.append("成交量放大")
        else:
            reasons.append("成交量未放大")

    # --- 条件3: 综合势能为正且动量上升 ---
    comp_energy = latest.get('comprehensive_energy')
    energy_mom = latest.get('energy_momentum')
    if comp_energy is not None and energy_mom is not None:
        if pd.notna(comp_energy) and pd.notna(energy_mom):
            if comp_energy > 0 and energy_mom > 0:
                conditions_met.append("势能正向且上升")
            else:
                reasons.append("势能条件不满足")

    # --- 条件4: 价格位置偏上 (price_position > 0.6, 即靠近当日高点) ---
    price_pos = latest.get('price_position')
    if price_pos is not None and pd.notna(price_pos):
        if price_pos > 0.6:
            conditions_met.append("价格偏强")
        else:
            reasons.append("价格偏弱")

    # --- 条件5: 趋势势能为正 ---
    trend_energy = latest.get('trend_energy')
    if trend_energy is not None and pd.notna(trend_energy):
        if trend_energy > 0:
            conditions_met.append("趋势势能正")
        else:
            reasons.append("趋势势能负")

    # 组合条件: 至少满足3个条件视为买入信号
    buy_signal = len(conditions_met) >= 3

    return buy_signal, conditions_met, reasons


def evaluate_sell_conditions(analyzer_df):
    """
    基于最新一根K线的技术指标做卖出判断（非统计历史信号）
    返回: (是否满足卖出条件, 满足的条件列表, 不满足的原因列表)
    """
    if analyzer_df.empty or len(analyzer_df) < 20:
        return False, [], ["数据不足(需至少20根K线)"]

    latest = analyzer_df.iloc[-1]
    conditions_met = []
    reasons = []

    # --- 条件1: 均线空头排列 (MA5 < MA20 且收盘价在MA5下方) ---
    ma5 = latest.get('ma5')
    ma20 = latest.get('ma20')
    close = latest['close']
    if ma5 is not None and ma20 is not None and not (pd.isna(ma5) or pd.isna(ma20)):
        if close < ma5 < ma20:
            conditions_met.append("均线空头排列")
        else:
            reasons.append("均线未空头排列")

    # --- 条件2: 成交量放大 (放量下跌) ---
    vol = latest['volume']
    vol_ma5 = latest.get('volume_ma5')
    if vol_ma5 is not None and not pd.isna(vol_ma5) and vol_ma5 > 0:
        if vol > vol_ma5 and close < latest['open']:
            conditions_met.append("放量下跌")
        else:
            reasons.append("非放量下跌")

    # --- 条件3: 综合势能为负且动量下降 ---
    comp_energy = latest.get('comprehensive_energy')
    energy_mom = latest.get('energy_momentum')
    if comp_energy is not None and energy_mom is not None:
        if pd.notna(comp_energy) and pd.notna(energy_mom):
            if comp_energy < 0 and energy_mom < 0:
                conditions_met.append("势能负向且下降")
            else:
                reasons.append("势能条件不满足")

    # --- 条件4: 价格位置偏下 (price_position < 0.4, 即靠近当日低点) ---
    price_pos = latest.get('price_position')
    if price_pos is not None and pd.notna(price_pos):
        if price_pos < 0.4:
            conditions_met.append("价格偏弱")
        else:
            reasons.append("价格仍偏强")

    # --- 条件5: 趋势势能为负 ---
    trend_energy = latest.get('trend_energy')
    if trend_energy is not None and pd.notna(trend_energy):
        if trend_energy < 0:
            conditions_met.append("趋势势能负")
        else:
            reasons.append("趋势势能正")

    # 组合条件: 至少满足3个条件视为卖出信号
    sell_signal = len(conditions_met) >= 3

    return sell_signal, conditions_met, reasons


def evaluate_strategy(daily_data):
    """基于最新数据实时评估股票策略"""
    # 运行分析，计算所有技术指标
    analyzer = MomentumEnergyAnalyzer(daily_data)
    df = analyzer.df

    # 直接基于最新数据做买卖判断
    buy_signal, buy_conditions, buy_reasons = evaluate_buy_conditions(df)
    sell_signal, sell_conditions, sell_reasons = evaluate_sell_conditions(df)

    if buy_signal:
        return "BUY", buy_conditions
    elif sell_signal:
        return "SELL", sell_conditions
    else:
        reasons = buy_reasons + sell_reasons
        return "HOLD", reasons

# ================= 并行处理系统 =================
def process_stock_chunk(config, codes):
    """处理一批股票"""
    results = []
    for code in codes:
        try:
            daily_data = load_single_stock_data(config, code)
            
            if daily_data.empty:
                results.append((code, "ERROR", "数据加载失败"))
                continue
            
            decision, reasons = evaluate_strategy(daily_data)
            results.append((code, decision, reasons))
        except Exception as e:
            results.append((code, "ERROR", str(e)))
    return results

def run_parallel_processing(config, all_codes):
    """并行处理所有股票"""
    chunk_size = config["system_params"]["chunk_size"]
    chunks = [all_codes[i:i+chunk_size] 
              for i in range(0, len(all_codes), chunk_size)]
    
    results = []
    with ProcessPoolExecutor(max_workers=config["system_params"]["max_workers"]) as executor:
        futures = [executor.submit(process_stock_chunk, config, chunk) for chunk in chunks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="处理进度"):
            results.extend(future.result())
    return results

# ================= 报告生成 =================
def generate_report(config, results):
    """生成分析报告"""
    output_dir = config["data_paths"]["output"]
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建结果DataFrame
    df = pd.DataFrame(results, columns=["代码", "决策", "原因/条件"])
    
    # 保存筛选结果
    results_path = output_dir / "策略决策结果.csv"
    df.to_csv(results_path, index=False, encoding='utf-8-sig')
    
    # 生成统计报告
    buy_count = df[df["决策"] == "BUY"].shape[0]
    sell_count = df[df["决策"] == "SELL"].shape[0]
    hold_count = df[df["决策"] == "HOLD"].shape[0]
    error_count = df[df["决策"] == "ERROR"].shape[0]
    total_count = df.shape[0]


    hold_df = df[df["决策"] == "HOLD"].explode("原因/条件")
    

    report = {
        "总股票数": total_count,
        "买入信号": buy_count,
        "卖出信号": sell_count,
        "持有信号": hold_count,
        "错误数量": error_count,
        "买入比例": f"{buy_count/total_count:.2%}",
        "卖出比例": f"{sell_count/total_count:.2%}",
        "常见持有原因": hold_df["原因/条件"].value_counts().head(5).to_dict()
    }
    
    with open(output_dir / "策略摘要报告.json", "w", encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    buy_stocks = df[df["决策"] == "BUY"]["代码"].tolist()
    # 复制文件
    return df
    for stock_code in buy_stocks:
        # 构建源文件路径和目标文件路径
        source_file = config["data_paths"]["daily"] / f"{stock_code}{config['file_suffix']}"  # 假设文件扩展名是.csv
        target_file = config["target"] / f"{stock_code}{config['file_suffix']}"
        
        # 检查源文件是否存在，然后复制
        if os.path.exists(source_file):
            shutil.copy2(source_file, target_file)  # copy2会保留文件元数据
            print(f"已复制: {stock_code}")
        else:
            print(f"文件不存在: {source_file}")
    

# ================= 主程序 =================
def auto_select(strategy_id=1):
    """主执行函数"""
    # 根据策略ID调整配置
    if strategy_id == 1:
        # 策略1的特定配置
        logging.info("执行策略1: 趋势跟踪策略")
    elif strategy_id == 2:
        # 策略2的特定配置
        logging.info("执行策略2: 均值回归策略")
    elif strategy_id == 3:
        # 策略3的特定配置
        logging.info("执行策略3: 动量策略")
    elif strategy_id == 4:
        # 策略4的特定配置
        logging.info("执行策略4: 价值投资策略")
    elif strategy_id == 5:
        # 策略5的特定配置
        logging.info("执行策略5: 成长股策略")
    else:
        logging.warning(f"未知策略ID: {strategy_id}，使用默认策略1")
        strategy_id = 1
    config = load_config()
    
    # 确保目录存在
    for path in config["data_paths"].values():
        path.mkdir(parents=True, exist_ok=True)
    
    # 获取股票列表
    daily_files = list(config["data_paths"]["daily"].glob(f"*{config['file_suffix']}"))
    if not daily_files:
        logging.error(f"未找到日线数据文件，请检查路径: {config['data_paths']['daily']}")
        return
    
    all_codes = list(set(f.stem.split("_")[0] for f in daily_files))
    logging.info(f"发现 {len(all_codes)} 只待处理股票")
    
    # 执行策略评估
    results = run_parallel_processing(config, all_codes)
    
    # 生成报告
    report_df = generate_report(config, results)
    
    # 打印摘要

    buy_stocks = report_df[report_df["决策"] == "BUY"]["代码"].tolist()
    sell_stocks = report_df[report_df["决策"] == "SELL"]["代码"].tolist()
    
    logger.info("\n" + "="*50)
    logger.info(f"策略评估完成!")
    logger.info(f"买入信号: {len(buy_stocks)} 只股票")
    logger.info(f"卖出信号: {len(sell_stocks)} 只股票")
    logger.info(f"详细结果已保存到: {config['data_paths']['output']}")
    logger.info("="*50)

    if strategy_id == 1:
        code = buy_stocks
    elif strategy_id == 2:
        # 策略2的特定配置
        code = sell_stocks


    return code


# ============================================================================
# 策略改造: 添加CsvAutoSelectStrategy类
# 将CSV自动选择系统转换为交易策略
# ============================================================================

class CsvAutoSelectStrategy(BaseStrategy):
    """CSV自动选择策略"""
    
    def __init__(self, data: pd.DataFrame, params: dict):
        """
        初始化策略
        
        参数:
            data: 价格数据
            params: 策略参数
        """
        super().__init__(data, params)
        
        # 从params提取参数
        self.strategy_id = params.get('strategy_id', 1)
        
    def generate_signals(self):
        """
        生成交易信号
        
        基于CSV自动选择生成交易信号
        """
        # 简化版本的auto_select逻辑
        # 实际实现应该调用auto_select函数，但这里简化处理
        
        # 根据策略ID选择信号类型
        if self.strategy_id == 1:  # 趋势跟踪策略
            # 检查是否有买入信号
            if self.data['close'].iloc[-1] > self.data['close'].rolling(20).mean().iloc[-1]:
                self._record_signal(
                    timestamp=self.data.index[-1],
                    action='buy',
                    price=self.data['close'].iloc[-1]
                )
            else:
                self._record_signal(
                    timestamp=self.data.index[-1],
                    action='hold',
                    price=self.data['close'].iloc[-1]
                )
        elif self.strategy_id == 2:  # 均值回归策略
            # 检查是否有卖出信号
            if self.data['close'].iloc[-1] < self.data['close'].rolling(20).mean().iloc[-1]:
                self._record_signal(
                    timestamp=self.data.index[-1],
                    action='sell',
                    price=self.data['close'].iloc[-1]
                )
            else:
                self._record_signal(
                    timestamp=self.data.index[-1],
                    action='hold',
                    price=self.data['close'].iloc[-1]
                )
        else:
            # 其他策略，默认hold
            self._record_signal(
                timestamp=self.data.index[-1],
                action='hold',
                price=self.data['close'].iloc[-1]
            )
        
        return self.signals


# ============================================================================
# 策略改造完成
# ============================================================================

if __name__ == "__main__":
    auto_select(strategy_id=1)