#!/usr/bin/env python3
"""
美股日线数据采集脚本 (新浪数据源)

数据源: akshare stock_us_daily (新浪财经)
采集内容: 美股核心股票日线 OHLCV
存储格式: CSV, 新数据在上面 (date降序)

用法:
  python3 us_stock_collector.py               # 增量更新
  python3 us_stock_collector.py --fresh        # 全量重采 (需确认)
  python3 us_stock_collector.py --list         # 查看已采集股票
"""

import os
import sys
import time
import argparse
import subprocess
import warnings
from datetime import datetime

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# ==================== 路径配置 ====================
DATA_DIR = os.path.expanduser("~/home/futures_platform/data/us_stock_daily")

# ==================== 核心股票池 ====================
# 按用户需求可以扩展，这里放主流大盘股+中概股+期货相关
US_STOCKS = {
    # ---- 科技巨头 (Magnificent 7) ----
    "AAPL":  "Apple",
    "MSFT":  "Microsoft",
    "GOOGL": "Alphabet",
    "AMZN":  "Amazon",
    "META":  "Meta",
    "NVDA":  "NVIDIA",
    "TSLA":  "Tesla",

    # ---- 科技二线 ----
    "NFLX":  "Netflix",
    "AMD":   "AMD",
    "INTC":  "Intel",
    "AVGO":  "Broadcom",
    "ORCL":  "Oracle",
    "CRM":   "Salesforce",
    "ADBE":  "Adobe",
    "PYPL":  "PayPal",
    "UBER":  "Uber",
    "SHOP":  "Shopify",
    "SQ":    "Block",
    "SNAP":  "Snap",

    # ---- 金融 ----
    "JPM":   "JPMorgan",
    "BAC":   "Bank of America",
    "GS":    "Goldman Sachs",
    "MS":    "Morgan Stanley",
    "V":     "Visa",
    "MA":    "Mastercard",

    # ---- 能源/期货相关 ----
    "XOM":   "ExxonMobil",
    "CVX":   "Chevron",
    "COP":   "ConocoPhillips",
    "SLB":   "Schlumberger",
    "NEM":   "Newmont (Gold)",
    "FCX":   "Freeport (Copper)",
    "AA":    "Alcoa",
    "GOLD":  "Barrick Gold",
    "CLF":   "Cleveland-Cliffs (Steel)",

    # ---- 工业/制造 ----
    "CAT":   "Caterpillar",
    "DE":    "John Deere",
    "BA":    "Boeing",
    "GE":    "GE Aerospace",
    "HON":   "Honeywell",
    "UNP":   "Union Pacific",

    # ---- 消费 ----
    "WMT":   "Walmart",
    "COST":  "Costco",
    "KO":    "Coca-Cola",
    "PEP":   "PepsiCo",
    "MCD":   "McDonald's",
    "SBUX":  "Starbucks",
    "NKE":   "Nike",

    # ---- 医药 ----
    "JNJ":   "Johnson&Johnson",
    "UNH":   "UnitedHealth",
    "PFE":   "Pfizer",
    "LLY":   "Eli Lilly",
    "MRK":   "Merck",
    "ABBV":  "AbbVie",

    # ---- 指数ETF ----
    "SPY":   "S&P 500 ETF",
    "QQQ":   "Nasdaq 100 ETF",
    "IWM":   "Russell 2000 ETF",
    "DIA":   "Dow Jones ETF",
    "GLD":   "Gold ETF",
    "SLV":   "Silver ETF",
    "USO":   "Oil ETF",
    "TLT":   "20yr Treasury ETF",
    "HYG":   "High Yield Corp ETF",
    "UVXY":  "VIX 2x ETF",

    # ---- 中概股 ----
    "BABA":  "Alibaba",
    "JD":    "JD.com",
    "PDD":   "PDD (Temu)",
    "BIDU":  "Baidu",
    "NIO":   "NIO",
    "LI":    "Li Auto",
    "XPEV":  "XPeng",
    "BILI":  "Bilibili",
    "TME":   "Tencent Music",
    "FUTU":  "Futu",
    "TCEHY": "Tencent ADR",
}

# 美股指数 (新浪特殊代码)
US_INDICES = {
    ".DJI":  "Dow Jones",
    ".IXIC": "Nasdaq Composite",
    ".INX":  "S&P 500",
    ".NDX":  "Nasdaq 100",
}


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def get_existing_latest_date(filepath):
    """读取已有CSV的最新日期 (第一行)"""
    if not os.path.exists(filepath):
        return None
    try:
        df = pd.read_csv(filepath, nrows=1, usecols=["date"], dtype=str)
        if len(df) == 0:
            return None
        return df["date"].iloc[0]
    except Exception:
        return None


def fetch_one_stock(symbol):
    """子进程: 采集单只股票日线"""
    import akshare as ak
    import traceback

    try:
        if symbol.startswith("."):
            # 指数用 index_us_stock_sina
            df = ak.index_us_stock_sina(symbol=symbol)
        else:
            # 先试前复权, 失败则降级到不复权
            try:
                df = ak.stock_us_daily(symbol=symbol, adjust="qfq")
            except Exception:
                df = ak.stock_us_daily(symbol=symbol, adjust="")

        if df is None or len(df) == 0:
            return symbol, None, "empty response"

        # 统一列名
        df = df.rename(columns={"date": "date", "open": "open", "high": "high",
                                "low": "low", "close": "close", "volume": "volume"})
        df = df[["date", "open", "high", "low", "close", "volume"]].copy()
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df = df.sort_values("date", ascending=False).reset_index(drop=True)
        df["open"] = df["open"].astype(float).round(4)
        df["high"] = df["high"].astype(float).round(4)
        df["low"] = df["low"].astype(float).round(4)
        df["close"] = df["close"].astype(float).round(4)
        df["volume"] = df["volume"].astype(float).astype(int)

        return symbol, df, None

    except Exception as e:
        return symbol, None, traceback.format_exc()


def incremental_merge(old_path, new_df, symbol):
    """增量合并: 保留旧数据, 追加新数据, 去重, 降序"""
    if not os.path.exists(old_path):
        new_df.to_csv(old_path, index=False)
        return len(new_df)

    old_df = pd.read_csv(old_path, dtype=str)
    new_df = new_df.copy()
    new_df["date"] = new_df["date"].astype(str)
    old_df["date"] = old_df["date"].astype(str)

    combined = pd.concat([old_df, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["date"], keep="last")
    combined = combined.sort_values("date", ascending=False).reset_index(drop=True)

    n_new = len(new_df)
    combined.to_csv(old_path, index=False)
    return n_new


def collect_all(fresh=False):
    """采集所有股票"""
    ensure_dir(DATA_DIR)
    all_symbols = {**US_STOCKS, **US_INDICES}
    total = len(all_symbols)

    print(f"[US Stock] 共 {total} 只股票/指数待采集")
    print(f"[US Stock] 数据路径: {DATA_DIR}")
    print(f"[US Stock] 模式: {'全量重采' if fresh else '增量更新'}")
    print()

    success = 0
    failed = 0
    skipped = 0
    t0 = time.time()

    for i, (symbol, name) in enumerate(all_symbols.items(), 1):
        filepath = os.path.join(DATA_DIR, f"{symbol.replace('.', '')}.csv")

        # 增量模式: 检查是否需要更新
        if not fresh:
            latest = get_existing_latest_date(filepath)
            if latest:
                # 新浪返回的是完整历史, 所以如果最新日期是今天或昨天(周末情况)就跳过
                today = datetime.now().strftime("%Y-%m-%d")
                # 美股T+1, 数据到北京时间今天凌晨
                yesterday = pd.Timestamp.now() - pd.Timedelta(days=1)
                # 如果是周末, 往前找周五
                while yesterday.dayofweek >= 5:
                    yesterday -= pd.Timedelta(days=1)
                yesterday_str = yesterday.strftime("%Y-%m-%d")

                if latest >= yesterday_str:
                    skipped += 1
                    continue

        # 用子进程避免akshare V8引擎内存泄漏
        result = subprocess.run(
            [sys.executable, "-c",
             f"import sys; sys.path.insert(0, '{os.path.dirname(os.path.abspath(__file__))}'); "
             f"from us_stock_collector import fetch_one_stock; "
             f"import json; "
             f"sym, df, err = fetch_one_stock('{symbol}'); "
             f"r = {{'symbol': sym, 'error': err}}; "
             f"print(json.dumps(r))"],
            capture_output=True, text=True, timeout=60
        )

        if result.returncode != 0:
            print(f"  [{i:3d}/{total}] {symbol:8s} ({name:20s}) FAILED: {result.stderr[:100]}")
            failed += 1
            continue

        import json
        try:
            info = json.loads(result.stdout.strip().split('\n')[-1])
        except Exception:
            print(f"  [{i:3d}/{total}] {symbol:8s} ({name:20s}) PARSE ERROR")
            failed += 1
            continue

        if info.get("error"):
            err = info["error"][:80]
            print(f"  [{i:3d}/{total}] {symbol:8s} ({name:20s}) ERROR: {err}")
            failed += 1
            continue

        # 数据在stdout里没法直接传DataFrame, 改用直接调用的方式
        # 子进程只返回状态, 实际拉数据在这里
        sym_ret, df, err = fetch_one_stock(symbol)
        if err or df is None:
            print(f"  [{i:3d}/{total}] {symbol:8s} ({name:20s}) ERROR: {(err or 'empty')[:80]}")
            failed += 1
            continue

        if fresh:
            df.to_csv(filepath, index=False)
            n = len(df)
        else:
            n = incremental_merge(filepath, df, symbol)

        date_range = f"{df['date'].iloc[-1]}~{df['date'].iloc[0]}" if len(df) > 0 else "?"
        print(f"  [{i:3d}/{total}] {symbol:8s} ({name:20s}) {len(df):6d}条 {date_range}  +{n}")
        success += 1

        # 新浪限频: 间隔0.5s
        time.sleep(0.5)

    elapsed = time.time() - t0
    print(f"\n[US Stock] 完成: {success}成功, {failed}失败, {skipped}跳过, 耗时{elapsed:.0f}s")


def show_status():
    """查看已采集数据状态"""
    if not os.path.exists(DATA_DIR):
        print(f"目录不存在: {DATA_DIR}")
        return

    files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".csv")])
    if not files:
        print("无数据文件")
        return

    print(f"{'代码':10s} {'名称':20s} {'条数':>8s} {'最新日期':12s} {'最早日期':12s}")
    print("-" * 70)

    all_map = {**US_STOCKS, **US_INDICES}
    for f in files:
        symbol = f.replace(".csv", "")
        name = all_map.get(symbol, all_map.get(f".{symbol}", "?"))
        filepath = os.path.join(DATA_DIR, f)
        try:
            df = pd.read_csv(filepath, dtype=str, usecols=["date"])
            n = len(df)
            latest = df["date"].iloc[0]
            oldest = df["date"].iloc[-1]
            print(f"{symbol:10s} {name:20s} {n:8d} {latest:12s} {oldest:12s}")
        except Exception:
            print(f"{symbol:10s} {name:20s} ERROR reading")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="美股日线采集")
    parser.add_argument("--fresh", action="store_true", help="全量重采")
    parser.add_argument("--list", action="store_true", help="查看已采集状态")
    args = parser.parse_args()

    if args.list:
        show_status()
    else:
        collect_all(fresh=args.fresh)
