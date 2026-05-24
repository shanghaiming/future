def get_concept_relate(*a, **kw):
    import pandas as pd
    return pd.DataFrame(columns=["concept_thscode"])

#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 导入必要库
from core.base_strategy import BaseStrategy
import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from core.data_loader import *
from tqdm import tqdm  # 新增进度条库
def tsma_fast(data, n):
    """极速TSMA实现（参数n为周期数）"""
    data = np.asarray(data)
    length = len(data)
    if n <= 0 or n > length:
        return np.full_like(data, np.nan)
    
    # 生成滑动窗口视图（倒序）
    windows = sliding_window_view(data, n)[:, ::-1]
    m = len(windows)
    
    # 预计算绝对时间索引矩阵
    i_indices = np.arange(n-1, n-1 + m)  # 窗口结束的原始索引
    x = i_indices[:, None] - np.arange(n)
    
    # 批量计算核心项
    y_sum = windows.sum(axis=1)
    x_sum = x.sum(axis=1)
    xx_sum = (x ** 2).sum(axis=1)
    xy_sum = (x * windows).sum(axis=1)
    
    denominator = xx_sum - (x_sum ** 2) / n
    numerator = xy_sum - (y_sum * x_sum) / n
    k = np.divide(numerator, denominator, where=denominator != 0)
    b = (y_sum / n) - k * (x_sum / n)
    
    tsma = k * i_indices + b + k
    
    result = np.full_like(data, np.nan)
    result[n-1 : n-1 + m] = np.where(denominator != 0, tsma, np.nan)
    return result



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
    williams_r = (numerator / denominator) * -100
    
    # 处理分母为零的情况（设为NaN）
    williams_r = williams_r.where(denominator != 0, float('nan'))
    
    return williams_r.rename(f'Williams_%R_{period}')
    
# ================= 指标计算 =================
def calculate_indicators(df, M=55, N=34, 
                        window_llv=10, window_hhv=25,
                        ema_period=4):
    """集成所有指标计算"""
    # 基础数据准备
    df['LC'] = df['close'].shift(1)
    df['CLOSE_LC'] = df['close'] - df['LC']
    
    # 计算动力线
    df['var2'] = df['low'].rolling(window_llv, min_periods=1).min()
    df['var3'] = df['high'].rolling(window_hhv, min_periods=1).max()
    denominator = df['var3'] - df['var2'] + 1e-5
    df['temp'] = (df['close'] - df['var2']) / denominator * 4
    df['temp'] = df['temp'].clip(0, 4)
    df['动力线'] = df['temp'].ewm(ema_period, adjust=False).mean()
    
    # 庄家/散户线
    # 计算M周期极值
    df['HHV_HIGH'] = df['high'].rolling(M, min_periods=1).max()
    df['LLV_LOW'] = df['low'].rolling(M, min_periods=1).min()
    
    # 处理分母为零的情况
    denominator = (df['HHV_HIGH'] - df['LLV_LOW']).replace(0, np.nan).fillna(1e-7)
    df['散户线'] = 100 * (df['HHV_HIGH'] - df['close']) / denominator
    df['散户线'] = df['散户线'].clip(0, 100)  # 限制在0-100范围

    # ========= 庄家线计算 =========
    # 计算N周期极值
    df['N_LLV'] = df['low'].rolling(N, min_periods=1).min()
    df['N_HHV'] = df['high'].rolling(N, min_periods=1).max()
    
    # RSV计算（带分母保护）
    rsv_denominator = (df['N_HHV'] - df['N_LLV']).replace(0, np.nan).fillna(1)
    df['RSV'] = (df['close'] - df['N_LLV']) / rsv_denominator * 100
    
    # SMA计算函数（通达信算法）
    def sma(series, window, alpha=1):
        sma_vals = []
        prev = series[0]
        for val in series:
            prev = (alpha*val + (window - alpha)*prev)/window
            sma_vals.append(prev)
        return pd.Series(sma_vals, index=series.index)
    
    # 计算KDJ指标
    df['K'] = sma(df['RSV'], window=3, alpha=1)  # 通达信SMA参数
    df['D'] = sma(df['K'], window=3, alpha=1)
    df['J'] = 3*df['K'] - 2*df['D']
    df['tsma5'] = tsma_fast(df['close'],5)
    df['tsma8'] = tsma_fast(df['close'],8)
    df['tsma13'] = tsma_fast(df['close'],13)
    
    # 最终EMA平滑
    df['庄家线'] = df['J'].ewm(span=6, adjust=False).mean().clip(0, 100)
    
    # VR指标
    av = df['volume'].where(df['close'] > df['LC'], 0)
    bv = df['volume'].where(df['close'] < df['LC'], 0)
    cv = df['volume'].where(df['close'] == df['LC'], 0)
    df['vr'] = (av.rolling(24).sum() + cv.rolling(24).sum()/2) / \
              (bv.rolling(24).sum() + cv.rolling(24).sum()/2 + 1e-7) * 100
    
    # MACD
    fast_ema = df['close'].ewm(span=5, adjust=False).mean()
    slow_ema = df['close'].ewm(span=13, adjust=False).mean()
    df['diff'] = fast_ema - slow_ema
    df['dea'] = df['diff'].ewm(span=8, adjust=False).mean()
    df['macd'] = 2 * (df['diff'] - df['dea'])
    
    # ER效率比率
    change = df['close'].diff(14).abs()
    volatility = df['close'].diff().abs().rolling(14).sum()
    df['er'] = change / (volatility + 1e-7)

    #威廉指标
    df['wr5'] = calculate_williams_r(df, period=5)
    df['wr55'] = calculate_williams_r(df, period=55)

    #神奇九转
    # 计算连续上涨条件
    df['A1'] = df['close'] > df['close'].shift(4)
    df['NT'] = df['A1'].astype(int).groupby((~df['A1']).cumsum()).cumcount()
    
    # 计算连续下跌条件
    df['B1'] = df['close'] < df['close'].shift(4)
    df['NT0'] = df['B1'].astype(int).groupby((~df['B1']).cumsum()).cumcount()
    
    # 初始化标记列
    df['up_mark'] = 0
    df['down_mark'] = 0

    # 处理上涨标记
    for i in range(len(df)):
        # 九转结构成立
        if df['NT'].iloc[i] == 9:
            start = max(0, i - 8)
            for j in range(start, i + 1):
                df['up_mark'].iloc[j] = j - start + 1
        
        # 最后一个K线且未完成结构
        if i == len(df) - 1 and 5 <= df['NT'].iloc[i] <= 8:
            nt = df['NT'].iloc[i]
            start = max(0, i - nt + 1)
            for j in range(start, i + 1):
                df['up_mark'].iloc[j] = j - start + 1

    # 处理下跌标记
    for i in range(len(df)):
        # 九转结构成立
        if df['NT0'].iloc[i] == 9:
            start = max(0, i - 8)
            for j in range(start, i + 1):
                df['down_mark'].iloc[j] = j - start + 1
        
        # 最后一个K线且未完成结构
        if i == len(df) - 1 and 5 <= df['NT0'].iloc[i] <= 8:
            nt = df['NT0'].iloc[i]
            start = max(0, i - nt + 1)
            for j in range(start, i + 1):
                df['down_mark'].iloc[j] = j - start + 1

    # 清理中间列
      
       
    return df.drop(['var2','var3','temp','LC','CLOSE_LC', 'A1', 'B1', 'NT', 'NT0'], axis=1)
# 获取同花顺概念指数数据（带清洗）
concept_df = get_concept_relate(date='20230801', fields='concept_thscode')
# 关键修复1：正确处理DataFrame结构
# 清洗数据并创建代码-名称映射
concept_clean = (
    concept_df
    .dropna(subset=['concept_thscode'])  # 移除空代码
    .reset_index()  # 将概念名称转为普通列
    .rename(columns={'index': 'concept_name'})
)

# 创建代码到名称的映射字典
code_to_name = dict(zip(concept_clean['concept_thscode'], concept_clean['concept_name']))

# 获取有效证券代码列表
valid_codes = concept_clean['concept_thscode'].tolist()

# 存储符合条件的指数
selected_indices = []

# 新增导入
from concurrent.futures import ThreadPoolExecutor, as_completed

# 修改后的并行处理部分
def process_code(thscode):
    try:
        df = get_price(
            securities=thscode,
            end_date='20250516',
            fre_step='1d',
            fields=['open','high','low','close','volume'],
            fq='pre',
            bar_count=250,
            skip_paused=True
        )
        
        if df.empty or len(df) < 55:
            return None
            
        df = df.sort_index(ascending=True)
        df = calculate_indicators(df)
        latest_wr5 = df['wr5'].iloc[-1]
        
        if pd.notnull(latest_wr5) and latest_wr5 < -80:
            return {
                'code': thscode,
                'name': code_to_name.get(thscode, '未知指数'),
                'willr': latest_wr5
            }
        return None
        
    except Exception as e:
        tqdm.write(f"⚠️ 处理代码 {thscode} 时发生错误：{str(e)}")
        return None

# 修改后的主处理逻辑
selected_indices = []
with ThreadPoolExecutor(max_workers=4) as executor:  # 根据API限制调整线程数
    futures = {executor.submit(process_code, code): code for code in valid_codes}
    
    for future in tqdm(as_completed(futures), total=len(futures), desc="并行处理中"):
        result = future.result()
        if result:
            selected_indices.append(result)


# 结果输出
if selected_indices:
    # 按WR值从小到大排序
    sorted_indices = sorted(selected_indices, key=lambda x: x['willr'])
    
    print(f"满足威廉指标WR5<-80的指数（共{len(sorted_indices)}个，按WR5升序排列）:")
    for idx, item in enumerate(sorted_indices, 1):
        print(f"{idx}. {item['name']}（{item['code']}）: WR5={item['willr']:.2f}")
else:
    print("当前没有符合条件的指数")

# In[ ]:






class IndexStrategy(BaseStrategy):
    """基于index的策略"""
    
    def __init__(self, data, params=None):
        super().__init__(data, params)
        # 初始化代码
        self.name = "IndexStrategy"
        self.description = "基于index的策略"
        
    def generate_signals(self):
        """RSI(14) with Bollinger Band confirmation. Buy RSI<30 + lower BB, sell RSI>70 + upper BB."""
        df = self.data

        if len(df) < 30:
            return self.signals

        close = df['close']
        # RSI(14)
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        rs = gain / loss.replace(0, float('nan'))
        rsi = 100 - (100 / (1 + rs))

        # Bollinger Bands (20, 2)
        bb_mid = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std

        for i in range(20, len(df)):
            price = float(close.iloc[i])
            r = rsi.iloc[i]
            if pd.isna(r):
                continue
            # Buy: RSI crosses below 30 AND price touches/crosses lower BB
            if (r < 30 and price <= bb_lower.iloc[i]):
                self._record_signal(df.index[i], 'buy', price=price)
            # Sell: RSI crosses above 70 AND price touches/crosses upper BB
            elif (r > 70 and price >= bb_upper.iloc[i]):
                self._record_signal(df.index[i], 'sell', price=price)

        return self.signals
