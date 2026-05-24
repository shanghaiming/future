import pandas as pd, os, numpy as np

dir = os.path.expanduser('~/home/futures_platform/data/futures_weighted')

# 模拟v8的信号生成，测试不同阈值下的胜率
for threshold in [2, 3, 4, 5]:
    signals = []
    for f in sorted(os.listdir(dir))[:40]:
        if not f.endswith('.csv'): continue
        df = pd.read_csv(os.path.join(dir, f))
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = df.sort_values('trade_date').reset_index(drop=True)
        
        df['return'] = df['close'].pct_change()
        for p in [5, 10, 20, 30, 60]:
            df[f'ma_{p}'] = df['close'].rolling(p).mean()
        
        df['ma_score'] = 0
        for short, long in [(5,10), (10,20), (20,30), (30,60)]:
            df.loc[df[f'ma_{short}'] > df[f'ma_{long}'], 'ma_score'] += 1
            df.loc[df[f'ma_{short}'] < df[f'ma_{long}'], 'ma_score'] -= 1
        
        for p in [10, 20, 60]:
            df[f'mom_{p}'] = (df['close'] - df['close'].shift(p)) / df['close'].shift(p)
        
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + gain / loss))
        
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        df['trend_score'] = 0.0
        df.loc[df['ma_score'] >= 3, 'trend_score'] += 2
        df.loc[df['ma_score'] == 2, 'trend_score'] += 1
        df.loc[df['ma_score'] <= -3, 'trend_score'] -= 2
        df.loc[df['ma_score'] == -2, 'trend_score'] -= 1
        df.loc[df['mom_20'] > 0.05, 'trend_score'] += 1
        df.loc[df['mom_20'] < -0.05, 'trend_score'] -= 1
        df.loc[df['macd'] > df['macd_signal'], 'trend_score'] += 1
        df.loc[df['macd'] < df['macd_signal'], 'trend_score'] -= 1
        df.loc[df['rsi'] > 60, 'trend_score'] += 0.5
        df.loc[df['rsi'] < 40, 'trend_score'] -= 0.5
        
        df['hv_20'] = df['return'].rolling(20).std() * np.sqrt(252)
        df['vol_ma20'] = df['vol'].rolling(20).mean()
        df['vol_ratio'] = df['vol'] / df['vol_ma20']
        
        for i in range(80, len(df)-20):
            latest = df.iloc[i]
            prev = df.iloc[i-1]
            
            if latest['hv_20'] < 0.12 or latest['hv_20'] > 0.8:
                continue
            if latest['vol_ratio'] < 0.7:
                continue
            
            if latest['trend_score'] >= threshold:
                if latest['close'] > latest['ma_20'] and latest['mom_10'] > 0 and latest['macd_hist'] > prev['macd_hist']:
                    future_ret = (df['close'].iloc[i+20] - df['close'].iloc[i]) / df['close'].iloc[i]
                    signals.append({'ret': future_ret, 'score': latest['trend_score']})
            elif latest['trend_score'] <= -threshold:
                if latest['close'] < latest['ma_20'] and latest['mom_10'] < 0 and latest['macd_hist'] < prev['macd_hist']:
                    future_ret = -(df['close'].iloc[i+20] - df['close'].iloc[i]) / df['close'].iloc[i]
                    signals.append({'ret': future_ret, 'score': abs(latest['trend_score'])})
    
    if signals:
        r = pd.DataFrame(signals)
        print(f'阈值={threshold}: 信号={len(r)} 胜率={(r.ret>0).mean():.1%} 平均={r.ret.mean():.2%}')
        # 按分数分层
        for score in sorted(r['score'].unique()):
            sub = r[r['score'] == score]
            if len(sub) > 10:
                print(f'  分数={score}: 次数={len(sub)} 胜率={(sub.ret>0).mean():.1%} 平均={sub.ret.mean():.2%}')
