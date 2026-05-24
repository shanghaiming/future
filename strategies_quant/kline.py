#!/usr/bin/env python
# coding: utf-8

# In[1]:


from core.base_strategy import BaseStrategy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from datetime import datetime

# 定义K线方向常量
BULLISH = 1  # 阳线
BEARISH = 2  # 阴线
DOJI = 3     # 十字星

# 定义K线关系常量
CONTAINING = 1
CONTAINED = 2
UPWARD_GAP = 3
DOWNWARD_GAP = 4
OVERLAP = 5
SEPARATE = 6
NO_RELATIONSHIP = 7

# 定义K线数据结构
class KLine:
    def __init__(self, open, close, high, low, timestamp=None):
        self.open = open
        self.close = close
        self.high = high
        self.low = low
        self.timestamp = timestamp
        
        # 确定K线方向
        if self.close > self.open:
            self.direction = BULLISH
        elif self.close < self.open:
            self.direction = BEARISH
        else:
            self.direction = DOJI
    
    def get_effective_high(self):
        upper_shadow = self.high - max(self.open, self.close)
        body_length = abs(self.close - self.open)
        return self.high if upper_shadow > (body_length * 0.33) else max(self.open, self.close)
    
    def get_effective_low(self):
        lower_shadow = min(self.open, self.close) - self.low
        body_length = abs(self.close - self.open)
        return self.low if lower_shadow > (body_length * 0.33) else min(self.open, self.close)
    
    def get_body_length(self):
        return abs(self.close - self.open)
    
    def get_total_length(self):
        return self.high - self.low

# 从DataFrame创建K线列表
def create_klines_from_df(df):
    klines = []
    for idx, row in df.iterrows():
        kline = KLine(
            open=row['open'],
            close=row['close'],
            high=row['high'],
            low=row['low'],
            timestamp=idx if isinstance(idx, datetime) else None
        )
        klines.append(kline)
    return klines

# 确定两根K线的关系
def determine_relationship(kline1, kline2):
    k1_high, k1_low = kline1.get_effective_high(), kline1.get_effective_low()
    k2_high, k2_low = kline2.get_effective_high(), kline2.get_effective_low()
    
    if k2_low >= k1_low and k2_high <= k1_high:
        return CONTAINED
    if k1_low >= k2_low and k1_high <= k2_high:
        return CONTAINING
    if k2_low > k1_high:
        return UPWARD_GAP
    if k2_high < k1_low:
        return DOWNWARD_GAP
    if k2_low > k1_high or k2_high < k1_low:
        return SEPARATE
    if k2_low < k1_high and k2_high > k1_low:
        return OVERLAP
    return NO_RELATIONSHIP

# 计算缺口
def calculate_gap(kline1, kline2):
    upward_gap = max(0, kline2.get_effective_low() - kline1.get_effective_high())
    downward_gap = max(0, kline1.get_effective_low() - kline2.get_effective_high())
    
    if upward_gap > 0:
        gap_type = "upward"
    elif downward_gap > 0:
        gap_type = "downward"
    else:
        gap_type = "none"
        
    return {
        "type": gap_type, 
        "upward_gap": upward_gap, 
        "downward_gap": downward_gap, 
        "has_gap": gap_type != "none"
    }

# 关系名称映射
RELATIONSHIP_NAMES = {
    CONTAINING: "CONTAINING",
    CONTAINED: "CONTAINED",
    UPWARD_GAP: "UPWARD_GAP",
    DOWNWARD_GAP: "DOWNWARD_GAP",
    OVERLAP: "OVERLAP",
    SEPARATE: "SEPARATE",
    NO_RELATIONSHIP: "NO_RELATIONSHIP"
}

# 表征两根K线
def characterize_two_klines(kline1, kline2):
    relationship = determine_relationship(kline1, kline2)
    return {
        "relationship": RELATIONSHIP_NAMES[relationship],
        "gap_info": calculate_gap(kline1, kline2),
        "kline1": {"open": kline1.open, "close": kline1.close, "high": kline1.high, "low": kline1.low},
        "kline2": {"open": kline2.open, "close": kline2.close, "high": kline2.high, "low": kline2.low}
    }

# 递归表征所有K线
def recursive_characterization(klines):
    results = []
    for i in range(1, len(klines)):
        result = characterize_two_klines(klines[i-1], klines[i])
        result["index"] = i
        results.append(result)
    return results

# 绘制K线图并标注表征结果
def plot_klines_with_characterization(df, characterization_results):
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # 绘制K线
    for i, (idx, row) in enumerate(df.iterrows()):
        color = 'red' if row['close'] > row['open'] else 'green'
        ax.plot([i, i], [row['low'], row['high']], color=color, linewidth=1)
        rect = Rectangle((i-0.3, min(row['open'], row['close'])), 0.6, abs(row['close']-row['open']), 
                        facecolor=color, alpha=0.7)
        ax.add_patch(rect)
    
    # 标注表征结果
    color_map = {
        "CONTAINING": "blue", "CONTAINED": "purple", "UPWARD_GAP": "orange",
        "DOWNWARD_GAP": "brown", "OVERLAP": "gray", "SEPARATE": "cyan", "NO_RELATIONSHIP": "black"
    }
    
    for result in characterization_results:
        i = result["index"]
        relationship = result["relationship"]
        mid_x = i - 0.5
        mid_y = (df.iloc[i-1]['high'] + df.iloc[i]['low']) / 2
        color = color_map.get(relationship, "black")
        
        ax.annotate(relationship, xy=(mid_x, mid_y), xytext=(mid_x, mid_y+0.5),
                   arrowprops=dict(arrowstyle="->", color=color), fontsize=8, color=color, ha='center')
    
    # 设置图表属性
    ax.set_xlabel('Index')
    ax.set_ylabel('Price')
    ax.set_title('K-line Chart with Characterization')
    ax.grid(True)
    ax.set_xticks(range(len(df)))
    
    # 设置x轴标签
    if len(df) < 20:
        labels = [d.strftime('%Y-%m-%d') if isinstance(d, datetime) else str(d) for d in df.index]
        ax.set_xticklabels(labels)
    else:
        labels = [d.strftime('%Y-%m-%d') if isinstance(d, datetime) and i%5==0 else '' for i, d in enumerate(df.index)]
        ax.set_xticklabels(labels)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    

# 在Jupyter中运行
if __name__ == "__main__":
# MindGo数据获取（需在平台环境中运行）
    # 方法一：使用平台内置数据接口
    from mindgo_api import *
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='Get stock data by code.')
    parser.add_argument(
        '--code',                # 参数名
        type=str,
        default='002791.SZ',     # 默认证券代码
        help='Stock code (default: 300032.SZ)'
    )
    
    # 解析参数时忽略未知参数
    args, unknown = parser.parse_known_args()  # 关键修改
    df = get_price(
        securities=args.code,  # 注意参数名是复数但支持单个代码
        #end_date=datetime.date.today().strftime('%Y%m%d'),  # 结束日期设为今天
        end_date='20250920',  # 结束日期设为今天
        fre_step='1d',           # 日线频率
        fields=['open','high','low','close','volume'],
        fq='pre',                # 前复权
        bar_count=100,           # 获取250根K线
        skip_paused=True         # 跳过停牌日
    ).sort_index()  # 清除证券代码索引层级
   
    # 检查数据样例
    print(f"最新数据日期：{df.index[-1].strftime('%Y-%m-%d')}")
    print(df.tail())
    # 创建K线列表
    klines = create_klines_from_df(df)
    
    # 进行表征
    characterization_results = recursive_characterization(klines)
    
    # 打印结果
    print("K线表征结果:")
    for i, result in enumerate(characterization_results):
        print(f"K线 {i} 和 {i+1}: {result['relationship']}")
        if result['gap_info']['has_gap']:
            gap_size = result['gap_info']['upward_gap'] or result['gap_info']['downward_gap']
            print(f"  缺口类型: {result['gap_info']['type']}, 大小: {gap_size}")
    
    # 绘制图表
    plot_klines_with_characterization(df, characterization_results)



class KlineStrategy(BaseStrategy):
    """基于kline的策略"""
    
    def __init__(self, data, params=None):
        super().__init__(data, params)
        # 初始化代码
        self.name = "KlineStrategy"
        self.description = "基于kline的策略"
        
    def calculate_signals(self):
        """计算交易信号"""
        # 策略逻辑
        return df
        
    def generate_signals(self):
        """K线形态策略生成交易信号"""
        import numpy as np
        df = self.data
        o, h, l, c = df['open'], df['high'], df['low'], df['close']
        body = abs(c - o)
        total_range = h - l
        upper_shadow = h - np.maximum(o, c)
        lower_shadow = np.minimum(o, c) - l

        for i in range(1, len(df)):
            sym = df['symbol'].iloc[i] if 'symbol' in df.columns else 'DEFAULT'
            price = float(df['close'].iloc[i])
            tr = total_range.iloc[i]
            if tr < 1e-10:
                continue
            # 锤子线(看涨)
            if lower_shadow.iloc[i] > 2 * body.iloc[i] and upper_shadow.iloc[i] < 0.1 * tr:
                self._record_signal(df.index[i], 'buy', sym, price)
            # 流星线(看跌)
            elif upper_shadow.iloc[i] > 2 * body.iloc[i] and lower_shadow.iloc[i] < 0.1 * tr:
                self._record_signal(df.index[i], 'sell', sym, price)
            # 看涨吞没
            elif i > 0 and c.iloc[i-1] < o.iloc[i-1] and c.iloc[i] > o.iloc[i] and c.iloc[i] > o.iloc[i-1] and o.iloc[i] < c.iloc[i-1]:
                self._record_signal(df.index[i], 'buy', sym, price)
            # 看跌吞没
            elif i > 0 and c.iloc[i-1] > o.iloc[i-1] and c.iloc[i] < o.iloc[i] and c.iloc[i] < o.iloc[i-1] and o.iloc[i] > c.iloc[i-1]:
                self._record_signal(df.index[i], 'sell', sym, price)
        return self.signals
