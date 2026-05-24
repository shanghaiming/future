#!/usr/bin/env python3
"""
商品期权分析（基于期货波动率推算）
- 用期货历史波动率作为标的价格波动率输入
- 模拟不同行权价的期权价格
- 计算IV、Greeks、波动率曲面
- 保存到 ~/home/futures_platform/data/options/

注意：这是模拟分析框架，实际期权价格需从交易所获取
"""

import os
for k in list(os.environ.keys()):
    if 'proxy' in k.lower():
        del os.environ[k]

import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats import norm

BASE_DIR = os.path.expanduser("~/home/futures_platform")
DATA_DIR = os.path.join(BASE_DIR, "data")
FUTURES_DIR = os.path.join(DATA_DIR, "futures_weighted")
OPTIONS_DIR = os.path.join(DATA_DIR, "options")
LOG_DIR = os.path.join(BASE_DIR, "logs")

os.makedirs(OPTIONS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# 商品期权品种 - 直接从数据目录读取所有品种
FUTURES_DIR = os.path.join(DATA_DIR, "futures_weighted")
COMMODITY_OPTIONS = {}
for f in os.listdir(FUTURES_DIR):
    if f.endswith('.csv'):
        symbol = f.replace('.csv', '')
        COMMODITY_OPTIONS[symbol] = symbol  # 用代码作为名称


def log(msg):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_msg = f"[{timestamp}] {msg}"
    print(log_msg)


def load_futures_data(symbol):
    """加载期货加权数据"""
    path = os.path.join(FUTURES_DIR, f"{symbol.lower()}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
    df = df.sort_values('trade_date')
    return df


def calc_hv(prices, window=20):
    """计算历史波动率"""
    log_ret = np.log(prices / prices.shift(1))
    return log_ret.rolling(window).std().iloc[-1] * np.sqrt(252)


def bsm_price(S, K, T, r, sigma, flag='c'):
    """BSM定价（期货期权q=0）"""
    if T <= 0 or sigma <= 0:
        return max(0, S - K) if flag == 'c' else max(0, K - S)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if flag == 'c':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bsm_iv(S, K, T, r, price, flag='c'):
    """二分法IV"""
    if price <= 0:
        return None
    sigma_low, sigma_high = 0.001, 5.0
    for _ in range(100):
        sigma_mid = (sigma_low + sigma_high) / 2
        price_mid = bsm_price(S, K, T, r, sigma_mid, flag)
        if abs(price_mid - price) < 1e-6:
            return sigma_mid
        if price_mid < price:
            sigma_low = sigma_mid
        else:
            sigma_high = sigma_mid
    return sigma_mid


def bsm_greeks(S, K, T, r, sigma, flag='c'):
    """BSM Greeks"""
    if T <= 0 or sigma <= 0:
        return None
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    nd1 = norm.pdf(d1)
    if flag == 'c':
        delta = norm.cdf(d1)
        theta = (-S * nd1 * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2))
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    else:
        delta = -norm.cdf(-d1)
        theta = (-S * nd1 * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2))
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
    gamma = nd1 / (S * sigma * np.sqrt(T))
    vega = S * nd1 * np.sqrt(T)
    return {
        'delta': delta, 'gamma': gamma,
        'theta': theta / 365, 'vega': vega / 100, 'rho': rho / 100
    }


def analyze_commodity_option(symbol, name):
    """分析单个商品期权"""
    df = load_futures_data(symbol)
    if df is None or len(df) < 60:
        return None
    
    S = df['close'].iloc[-1]
    hv_20 = calc_hv(df['close'], 20)
    hv_60 = calc_hv(df['close'], 60)
    
    # 用HV作为IV基准，构建波动率曲面
    # 模拟不同到期日和行权价
    expiries = [30/365, 60/365, 90/365]  # 1/2/3个月
    
    # 行权价范围：80%到120%
    strikes = [S * (0.8 + 0.04 * i) for i in range(11)]
    
    surface = []
    for T in expiries:
        for K in strikes:
            moneyness = K / S
            
            # 波动率微笑：OTM期权IV更高
            smile = 0.05 * (moneyness - 1) ** 2
            skew = -0.15 * (moneyness - 1)  # 商品期权put skew更明显
            iv = hv_20 + smile + skew
            iv = max(0.05, iv)  # 最低5%
            
            # 看涨和看跌
            for flag in ['c', 'p']:
                price = bsm_price(S, K, T, 0.025, iv, flag)
                greeks = bsm_greeks(S, K, T, 0.025, iv, flag)
                
                surface.append({
                    'symbol': symbol,
                    'name': name,
                    'underlying_price': S,
                    'strike': K,
                    'moneyness': moneyness,
                    'expiry_days': int(T * 365),
                    'flag': 'call' if flag == 'c' else 'put',
                    'iv': iv,
                    'price': price,
                    'delta': greeks['delta'],
                    'gamma': greeks['gamma'],
                    'theta': greeks['theta'],
                    'vega': greeks['vega'],
                    'rho': greeks['rho'],
                })
    
    return {
        'symbol': symbol,
        'name': name,
        'date': df['trade_date'].iloc[-1].strftime('%Y-%m-%d'),
        'underlying_price': S,
        'hv_20': hv_20,
        'hv_60': hv_60,
        'surface': surface,
    }


def main():
    log("=" * 60)
    log("开始商品期权分析")
    log("=" * 60)
    
    results = []
    for symbol, name in COMMODITY_OPTIONS.items():
        result = analyze_commodity_option(symbol, name)
        if result:
            # 保存
            date_str = result['date'].replace('-', '')
            filepath = os.path.join(OPTIONS_DIR, f"{symbol}_{date_str}.json")
            
            # 转换numpy类型为Python原生类型
            result_serializable = {
                'symbol': result['symbol'],
                'name': result['name'],
                'date': result['date'],
                'underlying_price': float(result['underlying_price']),
                'hv_20': float(result['hv_20']),
                'hv_60': float(result['hv_60']),
                'surface': []
            }
            for s in result['surface']:
                result_serializable['surface'].append({
                    k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                    for k, v in s.items()
                })
            
            with open(filepath, 'w') as f:
                json.dump(result_serializable, f, indent=2, ensure_ascii=False)
            
            # CSV
            df = pd.DataFrame(result['surface'])
            csv_path = os.path.join(OPTIONS_DIR, f"{symbol}_{date_str}.csv")
            df.to_csv(csv_path, index=False)
            
            log(f"✓ {symbol} ({name}): spot={result['underlying_price']:.2f}, HV20={result['hv_20']:.1%}")
            results.append(result)
        else:
            log(f"✗ {symbol} ({name}): 数据不足")
    
    log(f"\n完成: 共{len(results)}个品种")
    
    # 汇总
    print("\n" + "=" * 60)
    print("商品期权波动率汇总")
    print("=" * 60)
    print(f"\n{'品种':<8} {'名称':<8} {'价格':<10} {'HV20':<8} {'HV60':<8}")
    print("-" * 50)
    for r in sorted(results, key=lambda x: x['hv_20'], reverse=True)[:15]:
        print(f"{r['symbol']:<8} {r['name']:<8} {r['underlying_price']:<10.2f} {r['hv_20']:<8.1%} {r['hv_60']:<8.1%}")
    
    return results


if __name__ == '__main__':
    main()
