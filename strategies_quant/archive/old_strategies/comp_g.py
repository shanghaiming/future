#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 完整技术分析看盘程序 v1.0

from core.base_strategy import BaseStrategy
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import gridspec
import argparse
import datetime
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
from matplotlib.dates import DateFormatter
from matplotlib.patches import Rectangle
from scipy.signal import find_peaks
from scipy.stats import linregress
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates



# 标准EMA计算
def calculate_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

# 补偿EMA计算
def calculate_compensated_ema(series, span, beta=0.15):
    ema = series.ewm(span=span, adjust=False).mean()
    comp_ema = np.full(len(series), np.nan)
    prev_ema = 0.0
    data_queue = []
    
    for i, value in enumerate(series):
        data_queue.append(value)
        
        if len(data_queue) > span:
            removed = data_queue.pop(0)
            
            # 仅在高值移出时补偿
            if removed > prev_ema:
                compensation = beta * (removed - prev_ema)
            else:
                compensation = 0
        else:
            compensation = 0
        
        comp_ema[i] = ema.iloc[i] + compensation
        prev_ema = comp_ema[i]
    
    return pd.Series(comp_ema, index=series.index)

# 标准MACD计算
def calculate_macd(close, fast=12, slow=26, signal=9):
    ema_fast = calculate_ema(close, fast)
    ema_slow = calculate_ema(close, slow)
    dif = ema_fast - ema_slow
    dea = calculate_ema(dif, signal)
    macd_hist = (dif - dea) * 2
    return dif, dea, macd_hist

# 补偿MACD计算
def calculate_compensated_macd(close, fast=12, slow=26, signal=9, beta=0.15):
    ema_fast = calculate_compensated_ema(close, fast, beta)
    ema_slow = calculate_compensated_ema(close, slow, beta)
    dif = ema_fast - ema_slow
    dea = calculate_compensated_ema(dif, signal, beta)
    macd_hist = (dif - dea) * 2
    return dif, dea, macd_hist

# 修正的标准效率比率(ER)计算
def calculate_er(close, period=10):
    er = np.zeros(len(close))
    
    for i in range(period, len(close)):
        # 计算价格变动净值
        net_change = close.iloc[i] - close.iloc[i-period]
        
        # 计算价格变动绝对值总和
        abs_change_sum = 0
        for j in range(i-period+1, i+1):
            abs_change_sum += abs(close.iloc[j] - close.iloc[j-1])
        
        # 避免除零错误
        if abs_change_sum == 0:
            er[i] = 0
        else:
            er[i] = abs(net_change) / abs_change_sum
    
    return pd.Series(er, index=close.index)

# 修正的补偿效率比率(ER)计算
def calculate_compensated_er(close, period=10, beta=0.2):
    er = np.zeros(len(close))
    prev_er = 0.0
    
    for i in range(period, len(close)):
        # 计算价格变动净值
        net_change = close.iloc[i] - close.iloc[i-period]
        
        # 计算价格变动绝对值总和
        abs_change_sum = 0
        for j in range(i-period+1, i+1):
            abs_change_sum += abs(close.iloc[j] - close.iloc[j-1])
        
        # 计算标准ER
        if abs_change_sum == 0:
            std_er = 0
        else:
            std_er = abs(net_change) / abs_change_sum
        
        # 计算补偿量
        if i > period:
            er_change = std_er - prev_er
            compensation = beta * er_change
        else:
            compensation = 0
        
        # 应用补偿
        er[i] = std_er + compensation
        prev_er = er[i]
    
    return pd.Series(er, index=close.index)

def plot_compensated_vs_standard_indicators(df, window=20, beta=0.25, title='补偿 vs 标准指标对比'):
    """
    绘制K线图与补偿/标准指标对比
    
    参数:
        df (pd.DataFrame): 包含OHLC数据的DataFrame
        window (int): 均线窗口大小
        beta (float): 补偿系数
        title (str): 图表标题
    """
    # 确保列名是小写
    df = df.rename(columns={
        'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close',
        'OPEN': 'open', 'HIGH': 'high', 'LOW': 'low', 'CLOSE': 'close'
    })
    
    # 计算均线
    df['sma'] = df['close'].rolling(window=window).mean()
    df['cma'] = calculate_compensated_ema(df['close'], window, beta)
    
    # 计算MACD - 标准和补偿
    df['macd_dif'], df['macd_dea'], df['macd_hist'] = calculate_macd(df['close'])
    df['comp_macd_dif'], df['comp_macd_dea'], df['comp_macd_hist'] = calculate_compensated_macd(df['close'], beta=beta)
    
    # 计算ER - 标准和补偿
    df['er'] = calculate_er(df['close'], period=10)
    df['comp_er'] = calculate_compensated_er(df['close'], period=10, beta=beta*1.5)
    
    # 创建图表 - 纵向布局
    plt.figure(figsize=(16, 20), dpi=100)
    gs = gridspec.GridSpec(5, 1, height_ratios=[3, 2, 2, 1.5, 1.5])
    
    # 子图1: K线和均线
    ax1 = plt.subplot(gs[0])
    
    # 子图2: 标准MACD (DIF, DEA和柱状图)
    ax2 = plt.subplot(gs[1])
    
    # 子图3: 补偿MACD (DIF, DEA和柱状图)
    ax3 = plt.subplot(gs[2])
    
    # 子图4: 标准ER
    ax4 = plt.subplot(gs[3])
    
    # 子图5: 补偿ER
    ax5 = plt.subplot(gs[4])
    
    # 设置日期格式
    date_fmt = mdates.DateFormatter('%Y-%m-%d')
    for ax in [ax1, ax2, ax3, ax4, ax5]:
        ax.xaxis.set_major_formatter(date_fmt)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # ====================== K线图和均线 ======================
    # 绘制K线
    for i in range(len(df)):
        date = df.index[i]
        open_price = df['open'].iloc[i]
        high_price = df['high'].iloc[i]
        low_price = df['low'].iloc[i]
        close_price = df['close'].iloc[i]
        
        # 确定K线颜色
        color = '#2ecc71' if close_price >= open_price else '#e74c3c'
        
        # 绘制K线实体
        ax1.plot([date, date], [open_price, close_price], 
                color=color, linewidth=4, solid_capstyle='round')
        
        # 绘制上下影线
        ax1.plot([date, date], [low_price, high_price], 
                color=color, linewidth=1)
    
    # 绘制均线
    ax1.plot(df.index, df['sma'], 'b-', linewidth=1.5, label=f'标准均线 ({window}日)', alpha=0.7)
    ax1.plot(df.index, df['cma'], '#8e44ad', linewidth=2.0, label=f'补偿均线 (β={beta})')
    
    # 添加均线对比标注
    if len(df) > window + 20:
        # 标注最大差异点
        diff_max_idx = (df['cma'] - df['sma']).idxmax()
        ax1.annotate(f'最大补偿效果: {df["cma"][diff_max_idx]-df["sma"][diff_max_idx]:.2f}', 
                    xy=(diff_max_idx, df['cma'][diff_max_idx]),
                    xytext=(diff_max_idx - pd.Timedelta(days=20), df['cma'][diff_max_idx] * 1.05),
                    arrowprops=dict(facecolor='#8e44ad', shrink=0.05),
                    fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.5', fc='#f1c40f', alpha=0.8))
    
    # 添加标题和标签
    ax1.set_title(f'{title} (窗口={window}, β={beta})', fontsize=16)
    ax1.set_ylabel('价格')
    ax1.legend(loc='best')
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # ====================== 标准MACD ======================
    # 绘制MACD柱
    macd_pos_std = df['macd_hist'] > 0
    macd_neg_std = df['macd_hist'] < 0
    ax2.bar(df.index[macd_pos_std], df['macd_hist'][macd_pos_std], 
           color='#3498db', alpha=0.6, width=0.8, label='正柱')
    ax2.bar(df.index[macd_neg_std], df['macd_hist'][macd_neg_std], 
           color='#e74c3c', alpha=0.6, width=0.8, label='负柱')
    
    # 绘制DIF和DEA线
    ax2.plot(df.index, df['macd_dif'], 'b-', linewidth=1.8, label='DIF')
    ax2.plot(df.index, df['macd_dea'], 'g-', linewidth=1.8, label='DEA')
    
    # 添加零线
    ax2.axhline(0, color='gray', linestyle='--', alpha=0.7)
    
    # 添加标题和图例
    ax2.set_title('标准MACD (DIF, DEA和柱状图)', fontsize=14)
    ax2.set_ylabel('MACD值')
    ax2.legend(loc='upper left')
    ax2.grid(True, linestyle='--', alpha=0.2)
    
    # ====================== 补偿MACD ======================
    # 绘制补偿MACD柱
    macd_pos_comp = df['comp_macd_hist'] > 0
    macd_neg_comp = df['comp_macd_hist'] < 0
    ax3.bar(df.index[macd_pos_comp], df['comp_macd_hist'][macd_pos_comp], 
           color='#2980b9', alpha=0.7, width=0.8, label='补偿正柱')
    ax3.bar(df.index[macd_neg_comp], df['comp_macd_hist'][macd_neg_comp], 
           color='#c0392b', alpha=0.7, width=0.8, label='补偿负柱')
    
    # 绘制补偿DIF和DEA线
    ax3.plot(df.index, df['comp_macd_dif'], '#3498db', linewidth=2.0, label='补偿DIF')
    ax3.plot(df.index, df['comp_macd_dea'], '#2ecc71', linewidth=2.0, label='补偿DEA')
    
    # 添加零线
    ax3.axhline(0, color='gray', linestyle='--', alpha=0.7)
    
    # 添加标题和图例
    ax3.set_title('补偿MACD (DIF, DEA和柱状图)', fontsize=14)
    ax3.set_ylabel('MACD值')
    ax3.legend(loc='upper left')
    ax3.grid(True, linestyle='--', alpha=0.2)
    
    # 标注补偿效果
    if len(df) > 100:
        # 找到MACD柱差异最大的点
        max_diff_idx = (df['comp_macd_hist'] - df['macd_hist']).abs().idxmax()
        ax3.annotate('补偿MACD信号\n更清晰稳定', 
                    xy=(max_diff_idx, df['comp_macd_hist'][max_diff_idx]),
                    xytext=(max_diff_idx - pd.Timedelta(days=15), 
                           df['comp_macd_hist'][max_diff_idx] * 1.5),
                    arrowprops=dict(facecolor='#2980b9', shrink=0.05),
                    fontsize=12,
                    bbox=dict(boxstyle='round,pad=0.5', fc='#f1c40f', alpha=0.8))
    
    # ====================== 标准ER ======================
    ax4.plot(df.index, df['er'], '#9b59b6', linewidth=2.0, label='标准ER')
    # 添加关键水平线
    ax4.axhline(0.8, color='#e74c3c', linestyle='--', alpha=0.7, label='超买区')
    ax4.axhline(0.5, color='gray', linestyle='--', alpha=0.7)
    ax4.axhline(0.2, color='#2ecc71', linestyle='--', alpha=0.7, label='超卖区')
    # 填充ER区域
    ax4.fill_between(df.index, df['er'], 0, color='#9b59b6', alpha=0.2)
    ax4.set_ylabel('效率比率(ER)')
    ax4.set_ylim(0, 1)
    ax4.set_title('标准效率比率(ER)', fontsize=14)
    ax4.legend(loc='upper left')
    ax4.grid(True, linestyle='--', alpha=0.2)
    
    # ====================== 补偿ER ======================
    ax5.plot(df.index, df['comp_er'], '#8e44ad', linewidth=2.5, label='补偿ER')
    # 添加关键水平线
    ax5.axhline(0.8, color='#e74c3c', linestyle='--', alpha=0.7, label='超买区')
    ax5.axhline(0.5, color='gray', linestyle='--', alpha=0.7)
    ax5.axhline(0.2, color='#2ecc71', linestyle='--', alpha=0.7, label='超卖区')
    # 填充ER区域
    ax5.fill_between(df.index, df['comp_er'], 0, color='#8e44ad', alpha=0.25)
    ax5.set_ylabel('效率比率(ER)')
    ax5.set_ylim(0, 1)
    ax5.set_title('补偿效率比率(ER)', fontsize=14)
    ax5.legend(loc='upper left')
    ax5.grid(True, linestyle='--', alpha=0.2)
    
    # 标注ER补偿效果
    if len(df) > 100:
        # 找到ER差异最大的点
        max_er_idx = (df['comp_er'] - df['er']).abs().idxmax()
        ax5.annotate('补偿ER更平滑\n减少锯齿波动', 
                    xy=(max_er_idx, df['comp_er'][max_er_idx]),
                    xytext=(max_er_idx - pd.Timedelta(days=20), 
                           df['comp_er'][max_er_idx] * 1.2),
                    arrowprops=dict(facecolor='#8e44ad', shrink=0.05),
                    fontsize=12,
                    bbox=dict(boxstyle='round,pad=0.5', fc='#f1c40f', alpha=0.8))
    
    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)  # 增加子图间距
    plt.show()



# ================= 主程序 =================
if __name__ == "__main__":
    # MindGo数据获取（需在平台环境中运行）
    # 方法一：使用平台内置数据接口
    from mindgo_api import *
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='Get stock data by code.')
    parser.add_argument(
        '--code',                # 参数名
        type=str,
        default='301053.SZ',     # 默认证券代码
        help='Stock code (default: 300032.SZ)'
    )
    
    # 解析参数时忽略未知参数
    args, unknown = parser.parse_known_args()  # 关键修改
    df = get_price(
        securities=args.code,  # 注意参数名是复数但支持单个代码
        #end_date=datetime.date.today().strftime('%Y%m%d'),  # 结束日期设为今天
        end_date='20230711',  # 结束日期设为今天
        fre_step='1d',           # 日线频率
        fields=['open','high','low','close','volume'],
        fq='pre',                # 前复权
        bar_count=100,           # 获取250根K线
        skip_paused=True         # 跳过停牌日
    ).sort_index()  # 清除证券代码索引层级
   
    # 检查数据样例
    print(f"最新数据日期：{df.index[-1].strftime('%Y-%m-%d')}")
    print(df.tail())
    # 绘制补偿与标准指标对比
    plot_compensated_vs_standard_indicators(
        df, 
        window=5, 
        beta=0.3, 
        title='股票: 补偿指标 vs 标准指标'
    )
    

# In[ ]:






class CompGStrategy(BaseStrategy):
    """基于comp_g的策略"""
    
    def __init__(self, data, params=None):
        super().__init__(data, params)
        # 初始化代码
        self.name = "CompGStrategy"
        self.description = "基于comp_g的策略"
        
    def calculate_signals(self):
        """计算交易信号"""
        # 策略逻辑
        return df
        
    def generate_signals(self):
        """Momentum state machine - only signal when momentum crosses threshold from opposite side."""
        import numpy as np
        df = self.data
        # Compute blended momentum
        mom5 = df['close'].pct_change(5)
        mom10 = df['close'].pct_change(10)
        energy = mom5 * 0.6 + mom10 * 0.4

        threshold = 0.02
        state = 'flat'  # 'flat', 'long', 'short'

        for i in range(20, len(df)):
            sym = df['symbol'].iloc[i] if 'symbol' in df.columns else 'DEFAULT'
            price = float(df['close'].iloc[i])
            e = energy.iloc[i]
            if pd.isna(e):
                continue

            if e > threshold and state != 'long':
                self._record_signal(df.index[i], 'buy', sym, price)
                state = 'long'
            elif e < -threshold and state != 'short':
                self._record_signal(df.index[i], 'sell', sym, price)
                state = 'short'
            elif abs(e) <= threshold:
                state = 'flat'

        return self.signals
