#!/usr/bin/env python3
"""
每日市场复盘报告
数据源: 本地CSV/JSON (futures_platform + quant_trade-main)
推送: 飞书
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

FUTURES_DIR = os.path.expanduser("~/home/futures_platform/data")
TERM_DIR = os.path.join(FUTURES_DIR, "futures_term_structure")
WEIGHTED_DIR = os.path.join(FUTURES_DIR, "futures_weighted")
OPTIONS_DIR = os.path.join(FUTURES_DIR, "tq_options")
STOCK_DIR = os.path.expanduser("~/home/quant_trade-main/data/daily_data2")


def load_futures_daily(symbol, days=60):
    """加载期货加权日线"""
    path = os.path.join(WEIGHTED_DIR, f"{symbol}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df['trade_date'] = df['trade_date'].astype(str)
    df = df.sort_values('trade_date', ascending=False).head(days)
    return df


def load_term_structure_today(symbol):
    """加载今日期限结构"""
    today = datetime.now().strftime('%Y%m%d')
    path = os.path.join(TERM_DIR, f"{symbol}_{today}.json")
    if not os.path.exists(path):
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
        path = os.path.join(TERM_DIR, f"{symbol}_{yesterday}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_index_data(days=30):
    """加载主要指数 - 用akshare实时接口"""
    import akshare as ak
    indices = {
        'sh000001': '上证指数',
        'sz399001': '深证成指',
        'sz399006': '创业板指',
        'sh000300': '沪深300',
        'sh000905': '中证500',
        'sh000852': '中证1000',
    }
    result = {}
    for code, name in indices.items():
        try:
            df = ak.stock_zh_index_daily_em(symbol=code)
            if df is not None and len(df) > 0:
                df = df.rename(columns={'date': 'trade_date'})
                df['trade_date'] = df['trade_date'].astype(str).str.replace('-', '')
                df = df.sort_values('trade_date', ascending=False).head(days)
                if len(df) > 0:
                    result[name] = df
        except:
            pass
    return result


def calc_change(df, n=1):
    """计算n日涨跌幅"""
    if df is None or len(df) < n + 1:
        return None
    latest = df.iloc[0]['close']
    prev = df.iloc[n]['close']
    if prev == 0:
        return None
    return (latest / prev - 1) * 100


def calc_volatility(df, window=20):
    """计算历史波动率"""
    if df is None or len(df) < window + 1:
        return None
    df_sorted = df.sort_values('trade_date', ascending=True).tail(window + 1)
    returns = df_sorted['close'].pct_change().dropna()
    return returns.std() * np.sqrt(252) * 100


def analyze_market_breadth():
    """市场宽度: 涨跌家数"""
    if not os.path.exists(STOCK_DIR):
        return None
    files = [f for f in os.listdir(STOCK_DIR) if f.endswith('.csv')]
    up = down = flat = 0
    dates_found = set()
    for f in files[:500]:  # 采样500只
        try:
            df = pd.read_csv(os.path.join(STOCK_DIR, f))
            df['trade_date'] = df['trade_date'].astype(str)
            df = df.sort_values('trade_date', ascending=False).head(2)
            if len(df) < 2:
                continue
            dates_found.add(df.iloc[0]['trade_date'])
            chg = df.iloc[0].get('pct_chg', 0)
            if pd.isna(chg):
                close0 = df.iloc[0]['close']
                close1 = df.iloc[1]['close']
                if close1 > 0:
                    chg = (close0 / close1 - 1) * 100
                else:
                    continue
            if chg > 0.5:
                up += 1
            elif chg < -0.5:
                down += 1
            else:
                flat += 1
        except:
            continue
    if not dates_found:
        return None
    return {'up': up, 'down': down, 'flat': flat, 'date': max(dates_found), 'total': up + down + flat}


def analyze_futures_momentum():
    """期货品种动量排名"""
    # 主要品种列表
    products = {
        'rbfi': '螺纹', 'hcfi': '热卷', 'ifi': '铁矿', 'jfi': '焦炭', 'jmfi': '焦煤',
        'cufi': '沪铜', 'alfi': '沪铝', 'znfi': '沪锌', 'nifi': '沪镍',
        'aufi': '黄金', 'agfi': '白银',
        'rufi': '橡胶', 'bufi': '沥青', 'spfi': '纸浆',
        'mfi': '豆粕', 'yfi': '豆油', 'pfi': '棕榈油', 'cfi': '玉米',
        'safi': '纯碱', 'fgfi': '玻璃', 'tafi': 'PTA', 'mafi': '甲醇',
        'egfi': '乙二醇', 'ppfi': '聚丙烯', 'lfi': '塑料', 'vfi': 'PVC',
        'oifi': '菜油', 'rmfi': '菜粕', 'srfi': '白糖', 'cffi': '棉花',
        'scfi': '原油', 'apfi': '苹果', 'jdfi': '鸡蛋',
    }

    results = []
    for sym, name in products.items():
        df = load_futures_daily(sym, 30)
        if df is None or len(df) < 2:
            continue
        chg1 = calc_change(df, 1)
        chg5 = calc_change(df, 5)
        chg20 = calc_change(df, 20) if len(df) >= 21 else None
        vol = calc_volatility(df, 20)

        # 期限结构
        ts = load_term_structure_today(sym)
        structure = ts.get('structure', '') if ts else ''

        results.append({
            'name': name, 'symbol': sym,
            'close': df.iloc[0]['close'],
            'chg1': chg1, 'chg5': chg5, 'chg20': chg20,
            'vol': vol, 'structure': structure,
        })

    return results


def analyze_term_structure_changes():
    """期限结构异动: 从contango变backwardation或反之"""
    products = {
        'rbfi': '螺纹', 'hcfi': '热卷', 'ifi': '铁矿', 'cufi': '沪铜', 'alfi': '沪铝',
        'aufi': '黄金', 'agfi': '白银', 'mfi': '豆粕', 'yfi': '豆油', 'safi': '纯碱',
        'fgfi': '玻璃', 'scfi': '原油', 'ppfi': '聚丙烯', 'oifi': '菜油',
    }

    changes = []
    for sym, name in products.items():
        today_ts = load_term_structure_today(sym)
        if not today_ts:
            continue

        # 找5天前的
        for d in range(1, 10):
            prev_date = (datetime.now() - timedelta(days=d)).strftime('%Y%m%d')
            prev_path = os.path.join(TERM_DIR, f"{sym}_{prev_date}.json")
            if os.path.exists(prev_path):
                with open(prev_path) as f:
                    prev_ts = json.load(f)
                if prev_ts.get('structure', '') != today_ts.get('structure', ''):
                    changes.append({
                        'name': name, 'symbol': sym,
                        'prev': prev_ts.get('structure', ''),
                        'now': today_ts.get('structure', ''),
                        'spread_pct': today_ts.get('total_spread_pct', 0),
                    })
                break

    return changes


def generate_report():
    """生成完整复盘报告"""
    today = datetime.now().strftime('%Y-%m-%d')
    lines = []
    lines.append(f"📊 每日市场复盘 {today}")
    lines.append("=" * 40)

    # 1. 指数概览
    indices = load_index_data(30)
    if indices:
        lines.append("\n【指数概览】")
        for name, df in indices.items():
            chg1 = calc_change(df, 1)
            chg5 = calc_change(df, 5)
            chg20 = calc_change(df, 20)
            vol = calc_volatility(df, 20)
            close = df.iloc[0]['close']
            date = df.iloc[0]['trade_date']

            chg1_str = f"{chg1:+.2f}%" if chg1 is not None else "N/A"
            chg5_str = f"{chg5:+.2f}%" if chg5 is not None else "N/A"
            chg20_str = f"{chg20:+.2f}%" if chg20 is not None else "N/A"
            vol_str = f"波动率{vol:.1f}%" if vol is not None else ""

            lines.append(f"  {name}: {close:.2f} 日{chg1_str} 周{chg5_str} 月{chg20_str} {vol_str}")

    # 2. 市场宽度
    breadth = analyze_market_breadth()
    if breadth:
        total = breadth['total']
        if total > 0:
            up_pct = breadth['up'] / total * 100
            down_pct = breadth['down'] / total * 100
            lines.append(f"\n【市场宽度】({breadth['date']})")
            lines.append(f"  上涨 {breadth['up']}只({up_pct:.0f}%) 下跌 {breadth['down']}只({down_pct:.0f}%) 平盘 {breadth['flat']}只")

    # 3. 期货动量排名
    futures = analyze_futures_momentum()
    if futures:
        lines.append("\n【期货动量】日涨幅TOP10")
        sorted_f = sorted(futures, key=lambda x: x.get('chg1') or -999, reverse=True)
        for f in sorted_f[:10]:
            chg1 = f"{f['chg1']:+.2f}%" if f['chg1'] is not None else "N/A"
            struct = f" [{f['structure'][:3]}]" if f['structure'] else ""
            lines.append(f"  {f['name']}: {f['close']:.0f} {chg1}{struct}")

        lines.append("\n【期货动量】日跌幅TOP10")
        for f in sorted_f[-10:]:
            chg1 = f"{f['chg1']:+.2f}%" if f['chg1'] is not None else "N/A"
            struct = f" [{f['structure'][:3]}]" if f['structure'] else ""
            lines.append(f"  {f['name']}: {f['close']:.0f} {chg1}{struct}")

    # 4. 期限结构异动
    ts_changes = analyze_term_structure_changes()
    if ts_changes:
        lines.append("\n【期限结构异动】")
        for c in ts_changes:
            lines.append(f"  {c['name']}: {c['prev'][:3]}→{c['now'][:3]} 展期{c['spread_pct']:+.2f}%")

    # 5. 高波动品种
    if futures:
        lines.append("\n【高波动品种】20日历史波动率TOP10")
        sorted_vol = sorted(futures, key=lambda x: x.get('vol') or 0, reverse=True)
        for f in sorted_vol[:10]:
            vol_str = f"{f['vol']:.1f}%" if f['vol'] is not None else "N/A"
            lines.append(f"  {f['name']}: {vol_str}")

    # 6. 5日/20日动量排名
    if futures:
        lines.append("\n【周动量TOP5】")
        sorted_w = sorted(futures, key=lambda x: x.get('chg5') or -999, reverse=True)
        for f in sorted_w[:5]:
            chg5 = f"{f['chg5']:+.2f}%" if f['chg5'] is not None else "N/A"
            lines.append(f"  {f['name']}: {chg5}")

        lines.append("\n【月动量TOP5】")
        sorted_m = sorted(futures, key=lambda x: x.get('chg20') or -999, reverse=True)
        for f in sorted_m[:5]:
            chg20 = f"{f['chg20']:+.2f}%" if f['chg20'] is not None else "N/A"
            lines.append(f"  {f['name']}: {chg20}")

    return "\n".join(lines)


if __name__ == '__main__':
    report = generate_report()
    print(report)

    # 保存到文件
    today = datetime.now().strftime('%Y%m%d')
    report_dir = os.path.join(FUTURES_DIR, "daily_reports")
    os.makedirs(report_dir, exist_ok=True)
    with open(os.path.join(report_dir, f"report_{today}.txt"), 'w', encoding='utf-8') as f:
        f.write(report)
