import pandas as pd, os, numpy as np

dir = os.path.expanduser('~/home/futures_platform/data/futures_weighted')

# 测试不同偏离程度下的收益
for dev_th in [0.005, 0.01, 0.015, 0.02, 0.03]:
    signals = []
    for f in sorted(os.listdir(dir))[:40]:
        if not f.endswith('.csv'): continue
        df = pd.read_csv(os.path.join(dir, f))
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = df.sort_values('trade_date').reset_index(drop=True)
        
        for p in [5, 10, 20, 60]:
            df[f'ma_{p}'] = df['close'].rolling(p).mean()
        df['trend_score'] = sum((df['close'] > df[f'ma_{p}']).astype(int) for p in [5, 10, 20, 60])
        df['ma5_dev'] = (df['close'] - df['ma_5']) / df['ma_5']
        
        mask = (df['trend_score'] >= 2) & (df['close'] < df['ma_5']) & (df['ma5_dev'] < -dev_th)
        for i in range(60, len(df)-5):
            if mask.iloc[i]:
                future_ret = (df['close'].iloc[i+5] - df['close'].iloc[i]) / df['close'].iloc[i]
                signals.append(future_ret)
    
    if signals:
        r = pd.Series(signals)
        print(f'偏离>{dev_th*100:.1f}%: 信号={len(r)} 胜率={(r>0).mean():.1%} 平均={r.mean():.2%} 中位数={r.median():.2%}')
