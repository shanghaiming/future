import pandas as pd, os, numpy as np

dir = os.path.expanduser('~/home/futures_platform/data/futures_weighted')

# 深入分析RSI突破60 + VOL>1.5
for hold in [3, 5, 10, 15, 20]:
    for vol_th in [1.2, 1.5, 2.0]:
        signals = []
        for f in sorted(os.listdir(dir))[:40]:
            if not f.endswith('.csv'): continue
            df = pd.read_csv(os.path.join(dir, f))
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df = df.sort_values('trade_date').reset_index(drop=True)
            
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            df['rsi'] = 100 - (100 / (1 + gain / loss))
            
            df['vol_ma20'] = df['vol'].rolling(20).mean()
            df['vol_ratio'] = df['vol'] / df['vol_ma20']
            
            mask = (df['rsi'] > 60) & (df['rsi'].shift(1) <= 60) & (df['vol_ratio'] > vol_th)
            for i in range(60, len(df)-hold):
                if mask.iloc[i]:
                    future_ret = (df['close'].iloc[i+hold] - df['close'].iloc[i]) / df['close'].iloc[i]
                    signals.append(future_ret)
        
        if signals:
            r = pd.Series(signals)
            print(f'RSI突破60 VOL>{vol_th} 持有{hold}天: 信号={len(r)} 胜率={(r>0).mean():.1%} 平均={r.mean():.2%}')
