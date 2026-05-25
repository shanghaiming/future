#!/usr/bin/env python3
"""
期货数据采集脚本 v2（新平台版）
- 采集加权指数 + 主力连续 + 多合约期限结构数据
- 支持东财API（等解封）+ akshare备用
- 自动保存到 ~/home/futures_platform/data/
"""

import os
import sys
import time
import json
import requests
import pandas as pd
from datetime import datetime, timedelta

BASE_DIR = os.path.expanduser("~/home/futures_platform")
DATA_DIR = os.path.join(BASE_DIR, "data")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# 东财API配置
EASTMONEY_API = "https://push2his.eastmoney.com/api/qt/stock/kline/get"

# 期货品种映射（东财代码）
FUTURES_SYMBOLS = {
    # 有色金属
    'cufi': '沪铜', 'alfi': '沪铝', 'znfi': '沪锌', 'pbfi': '沪铅',
    'nifi': '沪镍', 'snfi': '沪锡', 'aufi': '沪金', 'agfi': '沪银',
    # 黑色系
    'rbfi': '螺纹', 'hcfi': '热卷', 'ifi': '铁矿', 'jfi': '焦炭',
    'jmfi': '焦煤', 'fgfi': '玻璃', 'sifi': '硅铁', 'smfi': '锰硅',
    # 能源化工
    'scfi': '原油', 'bufi': '沥青', 'lufi': '低硫燃油', 'fufi': '燃油',
    'tafi': 'PTA', 'mafi': '甲醇', 'ppfi': '聚丙烯', 'lfi': '塑料',
    'vfi': 'PVC', 'egfi': '乙二醇', 'ebfi': '苯乙烯', 'urfi': '尿素',
    'safi': '纯碱', 'pfi': '棕榈', 'yfi': '豆油', 'ofi': '豆粕',
    'rmfi': '菜粕', 'mfi': '豆一', 'csfi': '玉米淀粉', 'cfi': '玉米',
    # 农产品
    'srfi': '白糖', 'cffi': '棉花', 'apfi': '苹果', 'cjfi': '红枣',
    'rrfi': '粳米', 'lhfi': '生猪', 'jdfi': '鸡蛋',
    # 金融期货
    'iffi': 'IF', 'icfi': 'IC', 'ihfi': 'IH', 'imfi': 'IM',
    'tsfi': 'TS', 'tfi': 'TF', 'tfffi': 'T',
}


def log(msg):
    """记录日志"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_msg = f"[{timestamp}] {msg}"
    print(log_msg)
    
    log_file = os.path.join(LOG_DIR, f"collector_{datetime.now().strftime('%Y%m%d')}.log")
    with open(log_file, 'a') as f:
        f.write(log_msg + '\n')


def fetch_eastmoney_kline(symbol, market='159', klt='101', lmt=500):
    """
    从东财获取K线数据
    market: 159=期货
    klt: 101=日K
    """
    params = {
        'secid': f'{market}.{symbol}',
        'klt': klt,
        'fqt': '1',
        'lmt': lmt,
        'end': '20500000',
        'fields1': 'f1,f2,f3,f4,f5,f6,f7,f8',
        'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64',
        'ut': '7eea3edcaed734bea9cbfc24409ed989',
        'forcect': '1'
    }
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }
    
    try:
        resp = requests.get(EASTMONEY_API, params=params, headers=headers, timeout=30)
        if resp.status_code != 200:
            log(f"{symbol}: HTTP {resp.status_code}")
            return None
        
        data = resp.json()
        if data.get('data') is None or data['data'].get('klines') is None:
            log(f"{symbol}: 无数据返回")
            return None
        
        klines = data['data']['klines']
        records = []
        for line in klines:
            parts = line.split(',')
            if len(parts) >= 6:
                records.append({
                    'trade_date': parts[0].replace('-', ''),
                    'open': float(parts[1]),
                    'close': float(parts[2]),
                    'high': float(parts[3]),
                    'low': float(parts[4]),
                    'vol': float(parts[5]),
                    'amount': float(parts[6]) if len(parts) > 6 else 0,
                    'amplitude': float(parts[7]) if len(parts) > 7 else 0,
                    'pct_change': float(parts[8]) if len(parts) > 8 else 0,
                    'change': float(parts[9]) if len(parts) > 9 else 0,
                    'turnover': float(parts[10]) if len(parts) > 10 else 0,
                })
        
        df = pd.DataFrame(records)
        return df
        
    except Exception as e:
        log(f"{symbol}: 异常 {str(e)}")
        return None


def save_data(df, symbol, data_type='futures_daily'):
    """保存数据到对应目录
    注意: 东财数据保存到 futures_daily，不要写 futures_weighted（tqsdk专属）
    """
    save_dir = os.path.join(DATA_DIR, data_type)
    os.makedirs(save_dir, exist_ok=True)

    filepath = os.path.join(save_dir, f"{symbol.lower()}.csv")

    # 统一日期格式为 YYYYMMDD
    df['trade_date'] = df['trade_date'].astype(str).str.replace('-', '')

    # 如果已有数据，合并去重
    if os.path.exists(filepath):
        existing = pd.read_csv(filepath, dtype=str)
        existing['trade_date'] = existing['trade_date'].astype(str).str.replace('-', '')
        combined = pd.concat([existing, df], ignore_index=True)
        combined = combined.drop_duplicates(subset=['trade_date'], keep='last')
        combined = combined.sort_values('trade_date', ascending=False)
        combined.to_csv(filepath, index=False)
        new_rows = len(combined) - len(existing)
    else:
        df.to_csv(filepath, index=False)
        new_rows = len(df)

    return new_rows


def collect_weighted_data():
    """采集加权指数数据"""
    log("=" * 60)
    log("开始采集期货加权指数数据")
    log("=" * 60)
    
    success = 0
    fail = 0
    fail_symbols = []
    
    for symbol, name in FUTURES_SYMBOLS.items():
        log(f"采集 {symbol} ({name})...")
        df = fetch_eastmoney_kline(symbol)
        
        if df is not None and not df.empty:
            new_rows = save_data(df, symbol, 'futures_daily')
            log(f"  ✓ {symbol}: {len(df)}条, 新增{new_rows}条")
            success += 1
        else:
            log(f"  ✗ {symbol}: 失败")
            fail += 1
            fail_symbols.append(symbol)
        
        time.sleep(0.5)  # 避免请求过快
    
    log(f"\n采集完成: 成功{success}, 失败{fail}")
    if fail > 5:
        log(f"!!! 失败过多({fail}个): {', '.join(fail_symbols[:10])}")
    
    return success, fail, fail_symbols


def collect_main_contract_data():
    """采集主力连续合约数据（格式不同）"""
    # 主力合约代码格式可能不同，需要确认
    # 暂时用同样的代码，实际可能需要调整
    log("主力合约采集（与加权共用代码）")
    return collect_weighted_data()


def check_api_status():
    """检查东财API状态"""
    test_df = fetch_eastmoney_kline('cufi', lmt=1)
    if test_df is not None:
        log("✓ 东财API正常")
        return True
    else:
        log("✗ 东财API异常（可能被拉黑）")
        return False


if __name__ == '__main__':
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # 检查API状态
    if not check_api_status():
        log("API不可用，退出")
        sys.exit(1)
    
    # 采集数据
    success, fail, fails = collect_weighted_data()
    
    # 保存状态
    status = {
        'timestamp': datetime.now().isoformat(),
        'success': success,
        'fail': fail,
        'fail_symbols': fails,
        'total_symbols': len(FUTURES_SYMBOLS)
    }
    
    status_file = os.path.join(BASE_DIR, 'config', 'last_collect_status.json')
    os.makedirs(os.path.dirname(status_file), exist_ok=True)
    with open(status_file, 'w') as f:
        json.dump(status, f, indent=2)
    
    log(f"状态已保存到 {status_file}")
