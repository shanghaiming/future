def get_money_flow_step(*args, **kwargs):
    return {}

#!/usr/bin/env python
# coding: utf-8

# In[2]:


# 获取平安银行, 贵州茅台2023年2月1日前3天的特大单买入数据
from core.base_strategy import BaseStrategy
value = get_money_flow_step(
    security_list=['300765.SZ'],
    start_date=None,
    end_date='20251204',
    fre_step='1d',
    fields=['act_buy_xl', 'pas_buy_xl', 
                'act_buy_l', 'pas_buy_l',
                'act_buy_m', 'pas_buy_m', 
                'act_sell_xl', 'pas_sell_xl', 
                'act_sell_l', 'pas_sell_l',
                'act_sell_m', 'pas_sell_m',
                'buy_l', 'sell_l',
                'dde_l', 'net_flow_rate','l_net_value'],
    count=1,
    is_panel=0
)
# 打印主动买入特大单金额数据
print(value) 

# In[ ]:






class UntitledStrategy(BaseStrategy):
    """基于Untitled的策略"""
    
    def __init__(self, data, params=None):
        super().__init__(data, params)
        # 初始化代码
        self.name = "UntitledStrategy"
        self.description = "基于Untitled的策略"
        
    def generate_signals(self):
        """MA(5)/MA(20) crossover with MACD confirmation."""
        df = self.data

        if len(df) < 30:
            return self.signals

        ma5 = df['close'].rolling(5).mean()
        ma20 = df['close'].rolling(20).mean()
        # MACD
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        dif = ema12 - ema26
        dea = dif.ewm(span=9, adjust=False).mean()
        macd_hist = 2 * (dif - dea)

        for i in range(26, len(df)):
            # MA crossover: buy when MA5 crosses above MA20
            if (ma5.iloc[i] > ma20.iloc[i] and ma5.iloc[i - 1] <= ma20.iloc[i - 1]
                    and macd_hist.iloc[i] > 0):
                self._record_signal(df.index[i], 'buy', price=float(df['close'].iloc[i]))
            # MA crossover: sell when MA5 crosses below MA20
            elif (ma5.iloc[i] < ma20.iloc[i] and ma5.iloc[i - 1] >= ma20.iloc[i - 1]
                    and macd_hist.iloc[i] < 0):
                self._record_signal(df.index[i], 'sell', price=float(df['close'].iloc[i]))

        return self.signals
