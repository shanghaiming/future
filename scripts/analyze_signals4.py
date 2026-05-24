import pandas as pd, os, numpy as np

dir = os.path.expanduser('~/home/futures_platform/data/futures_weighted')

# 测试更宽松的入场条件
conditions = [
    ('趋势>=2 + 价格<MA5', lambda df: (df['trend_score'] >= 2) & (df['close'] < df['ma_5'])),
    ('趋势>=1 + 价格<MA5', lambda df: (df['trend_score'] >= 1) & (df['close'] < df['ma_5'])),
    ('趋势>=2 + 价格<MA10', lambda df: (df['trend_score'] >= 2) & (df['close'] < df['ma_10'])),
    ('趋势>=1 + 价格<MA10', lambda df: (df['trend_score'] >= 1) & (df['close'] < df['ma_10'])),
    ('趋势>=2 + 价格<MA5 + HV>0.05', lambda df: (df['trend_score'] >= 2) & (df['close'] < df['ma_5']) & (df['hv_20'] > 0.05)),
    ('趋势>=1 + 价格<MA5 + HV>0.05', lambda df: (df['trend_score'] >= 1) & (df['close'] < df['ma_5']) & (df['hv_20'] > 0.05)),
]

for name, cond in conditions:
    signals = []
    for f in sorted(os.listdir(dir))[:40]:
        if not f.endswith('.csv'): continue
        df = pd.read_csv(os.path.join(dir, f))
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = df.sort_values('trade_date').reset_index(drop=True)
        
        for p in [5, 10, 20, 60]:
            df[f'ma_{p}'] = df['close'].rolling(p).mean()
        df['trend_score'] = sum((df['close'] > df[f'ma_{p}']).astype(int) for p in [5, 10, 20, 60])
        
        df['return'] = df['close'].pct_change()
        df['hv_20'] = df['return'].rolling(20).std() * np.sqrt(252)
        
        mask = cond(df)
        for i in range(60, len(df)-5):
            if mask.iloc[i]:
                future_ret = (df['close'].iloc[i+5] - df['close'].iloc[i]) / df['close'].iloc[i]
                signals.append(future_ret)
    
    if signals:
        r = pd.Series(signals)
        print(f'{name}: 信号={len(r)} 胜率={(r>0).mean():.1%} 平均={r.mean():.2%}')
    else:
        print(f'{name}: 无信号')
