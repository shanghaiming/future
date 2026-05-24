import pandas as pd, os, numpy as np

dir = os.path.expanduser('~/home/futures_platform/data/futures_weighted')

# 测试更宽松的条件
for rsi_th in [52, 55, 58]:
    for vol_th in [0.8, 1.0, 1.2]:
        signals = []
        total_days = 0
        for f in sorted(os.listdir(dir)):
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
            
            mask = (df['rsi'] > rsi_th) & (df['rsi'].shift(1) <= rsi_th) & (df['vol_ratio'] > vol_th)
            for i in range(60, len(df)-15):
                if mask.iloc[i]:
                    future_ret = (df['close'].iloc[i+15] - df['close'].iloc[i]) / df['close'].iloc[i]
                    signals.append(future_ret)
            total_days += len(df) - 60
        
        if signals:
            r = pd.Series(signals)
            freq = len(signals) / total_days * 100
            print(f'RSI突破{rsi_th} VOL>{vol_th}: 信号={len(r)} 胜率={(r>0).mean():.1%} 平均={r.mean():.2%} 频率={freq:.2f}%')
