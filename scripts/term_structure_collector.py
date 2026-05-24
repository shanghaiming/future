#!/usr/bin/env python3
"""
期货期限结构数据采集
- 从akshare新浪接口采集多合约数据
- 计算期限结构曲线、contango/backwardation
- 保存到 ~/home/futures_platform/data/futures_term_structure/
"""

# 必须在任何其他导入之前清除代理环境变量
import os
for k in list(os.environ.keys()):
    if 'proxy' in k.lower():
        del os.environ[k]

import sys
import time
import json
import pandas as pd
import numpy as np
from datetime import datetime

# 现在导入akshare
import akshare as ak

BASE_DIR = os.path.expanduser("~/home/futures_platform")
DATA_DIR = os.path.join(BASE_DIR, "data")
TERM_DIR = os.path.join(DATA_DIR, "futures_term_structure")
LOG_DIR = os.path.join(BASE_DIR, "logs")

os.makedirs(TERM_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# 品种映射
FUTURES_CONTRACTS = {
    'CU': '沪铜', 'AL': '沪铝', 'ZN': '沪锌', 'PB': '沪铅',
    'NI': '沪镍', 'SN': '沪锡', 'AU': '沪金', 'AG': '沪银',
    'RB': '螺纹', 'HC': '热卷', 'I': '铁矿', 'J': '焦炭',
    'JM': '焦煤', 'FG': '玻璃', 'SC': '原油', 'BU': '沥青',
    'TA': 'PTA', 'MA': '甲醇', 'PP': '聚丙烯', 'L': '塑料',
    'V': 'PVC', 'EG': '乙二醇', 'EB': '苯乙烯', 'SA': '纯碱',
    'UR': '尿素', 'P': '棕榈', 'Y': '豆油', 'M': '豆粕',
    'RM': '菜粕', 'SR': '白糖', 'CF': '棉花', 'AP': '苹果',
    'CJ': '红枣', 'LH': '生猪', 'JD': '鸡蛋',
    'IF': 'IF', 'IC': 'IC', 'IH': 'IH', 'IM': 'IM',
    'T': 'T', 'TF': 'TF', 'TS': 'TS',
}


def log(msg):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_msg = f"[{timestamp}] {msg}"
    print(log_msg)
    log_file = os.path.join(LOG_DIR, f"term_structure_{datetime.now().strftime('%Y%m%d')}.log")
    with open(log_file, 'a') as f:
        f.write(log_msg + '\n')


def fetch_contract(symbol):
    """获取单个合约最新数据"""
    try:
        df = ak.futures_zh_daily_sina(symbol=symbol)
        if df is None or df.empty:
            return None
        latest = df.iloc[-1]
        return {
            'symbol': symbol,
            'count': len(df),
            'date': str(latest['date']),
            'close': float(latest['close']),
            'open': float(latest['open']),
            'high': float(latest['high']),
            'low': float(latest['low']),
            'volume': int(latest['volume']),
            'hold': int(latest['hold']),
        }
    except Exception as e:
        return None


def collect_symbol(prefix, name):
    """采集单个品种的期限结构"""
    contracts = []
    for m in [6, 7, 8, 9, 12]:
        symbol = f"{prefix}26{m:02d}"
        data = fetch_contract(symbol)
        if data:
            contracts.append(data)
        time.sleep(0.5)
    
    if len(contracts) < 2:
        return None
    
    # 排序
    contracts.sort(key=lambda x: int(x['symbol'][-2:]))
    
    near = contracts[0]
    far = contracts[-1]
    spread = far['close'] - near['close']
    spread_pct = spread / near['close'] * 100
    structure = 'contango' if spread > 0 else 'backwardation'
    
    result = {
        'symbol': prefix,
        'name': name,
        'date': near['date'],
        'structure': structure,
        'curve': contracts,
        'near_contract': near['symbol'],
        'near_price': near['close'],
        'far_contract': far['symbol'],
        'far_price': far['close'],
        'total_spread': spread,
        'total_spread_pct': spread_pct,
    }
    
    # 保存
    date_str = near['date'].replace('-', '')
    filepath = os.path.join(TERM_DIR, f"{prefix}_{date_str}.json")
    with open(filepath, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    return result


def main():
    log("=" * 60)
    log("开始采集期货期限结构数据")
    log("=" * 60)
    
    results = []
    for prefix, name in FUTURES_CONTRACTS.items():
        log(f"采集 {prefix} ({name})...")
        result = collect_symbol(prefix, name)
        if result:
            log(f"  ✓ {prefix}: {len(result['curve'])}个合约, 结构={result['structure']}, 价差={result['total_spread_pct']:.2f}%")
            results.append(result)
        else:
            log(f"  ✗ {prefix}: 有效合约不足")
    
    log(f"\n采集完成: 共{len(results)}个品种")
    
    # 汇总
    contango = [r for r in results if r['structure'] == 'contango']
    backwardation = [r for r in results if r['structure'] == 'backwardation']
    
    print("\n" + "=" * 60)
    print("期限结构汇总")
    print("=" * 60)
    print(f"\nContango (远月>近月): {len(contango)}个")
    for r in sorted(contango, key=lambda x: x['total_spread_pct'], reverse=True)[:5]:
        print(f"  {r['symbol']} ({r['name']}): 价差={r['total_spread_pct']:.2f}%")
    
    print(f"\nBackwardation (近月>远月): {len(backwardation)}个")
    for r in sorted(backwardation, key=lambda x: x['total_spread_pct'])[:5]:
        print(f"  {r['symbol']} ({r['name']}): 价差={r['total_spread_pct']:.2f}%")
    
    return results


if __name__ == '__main__':
    main()
