#!/usr/bin/env python3
"""
金融数据采集脚本 - A股(日线+周线) + 期货(日线)
数据源: akshare/新浪(A股+期货)
排序: 新数据在上面（trade_date降序）

单位说明:
  A股 新浪原始: volume=股, amount=元
  保存格式: vol=手(=volume/100), amount=千元(=amount/1000)，与原tushare CSV兼容
  期货 vol=手, oi=手 (新浪原生)

用法:
  python3 data_collector.py stock_daily          # A股日线增量更新
  python3 data_collector.py stock_daily --fresh   # A股日线全量重采
  python3 data_collector.py stock_weekly          # A股周线增量更新
  python3 data_collector.py stock_weekly --fresh  # A股周线全量重采
  python3 data_collector.py futures               # 期货主力连续日线
  python3 data_collector.py futures --fresh
  python3 data_collector.py all                   # 全部采集
"""

import os
import sys
import time
import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import akshare as ak

warnings.filterwarnings("ignore")

# ==================== 路径配置 ====================
STOCK_DATA_DIR = os.path.expanduser("~/home/quant_trade-main/data")
FUTURES_DATA_DIR = os.path.expanduser("~/home/futures_platform/data")
STOCK_DAILY_DIR = os.path.join(STOCK_DATA_DIR, "daily_data2")
STOCK_WEEKLY_DIR = os.path.join(STOCK_DATA_DIR, "weekly_data")
FUTURES_DAILY_DIR = os.path.join(FUTURES_DATA_DIR, "futures_daily")

# ==================== 工具函数 ====================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def get_existing_latest_date(filepath):
    """读取已有CSV的最新日期（第一行，因为新数据在上面）"""
    if not os.path.exists(filepath):
        return None
    try:
        df = pd.read_csv(filepath, nrows=1, usecols=["trade_date"], dtype=str)
        if len(df) == 0:
            return None
        return df["trade_date"].iloc[0]
    except:
        return None


def incremental_merge(old_path, new_df, ts_code):
    """增量合并：新数据与旧CSV合并，去重，降序排列"""
    if not os.path.exists(old_path):
        new_df.to_csv(old_path, index=False)
        return len(new_df)

    old_df = pd.read_csv(old_path, dtype=str)
    old_df["trade_date"] = old_df["trade_date"].astype(str).str.replace("-", "")
    new_df = new_df.copy()
    if "trade_date" in new_df.columns:
        new_df["trade_date"] = new_df["trade_date"].astype(str).str.replace("-", "")

    combined = pd.concat([old_df, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["trade_date"], keep="last")
    combined["trade_date"] = combined["trade_date"].astype(str)
    combined = combined.sort_values("trade_date", ascending=False).reset_index(drop=True)

    n_new = len(new_df)
    combined.to_csv(old_path, index=False)
    return n_new


# ==================== A股采集 (akshare/新浪) ====================

# ts_code → 新浪symbol映射: 000001.SZ → sz000001, 600000.SH → sh600000
def ts_code_to_sina(ts_code):
    """ts_code(000001.SZ) → 新浪symbol(sz000001)"""
    code, exchange = ts_code.split(".")
    prefix = "sh" if exchange == "SH" else "sz"
    return f"{prefix}{code}"


def get_stock_list():
    """获取A股股票列表（从本地已有CSV或新浪接口）"""
    print("[Stock] 获取A股股票列表...")
    # 优先从本地CSV文件名获取
    if os.path.exists(STOCK_DAILY_DIR):
        existing = []
        for f in sorted(os.listdir(STOCK_DAILY_DIR)):
            if f.endswith(".csv"):
                ts_code = f.replace(".csv", "")
                existing.append((ts_code, ""))
        if existing:
            print(f"[Stock] 本地共 {len(existing)} 只A股")
            return existing
    return []


def normalize_stock_sina(df, ts_code):
    """
    将新浪A股日线转为标准格式
    新浪原始: date,open,high,low,close,volume(股),amount(元)
    保存格式: ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol(手),amount(千元)
    """
    df = df.rename(columns={"date": "trade_date"})
    df["trade_date"] = df["trade_date"].astype(str).str.replace("-", "")
    df["ts_code"] = ts_code

    for col in ["open", "high", "low", "close", "volume", "amount"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 单位换算: volume(股)→vol(手=股/100), amount(元)→千元(元/1000)
    df["vol"] = (df["volume"] / 100).round(2)
    df["amount"] = (df["amount"] / 1000).round(3)

    # 计算 pre_close / change / pct_chg
    df = df.sort_values("trade_date", ascending=True).reset_index(drop=True)
    closes = df["close"].astype(float)
    pre_close = closes.shift(1)
    df["pre_close"] = pre_close.round(2)
    df["change"] = (closes - pre_close).round(2)
    df["pct_chg"] = ((closes - pre_close) / pre_close * 100).round(2)
    df.loc[0, ["pre_close", "change", "pct_chg"]] = np.nan

    target = ["ts_code", "trade_date", "open", "high", "low", "close",
              "pre_close", "change", "pct_chg", "vol", "amount"]
    for col in target:
        if col not in df.columns:
            df[col] = ""

    df = df[target]
    df = df.sort_values("trade_date", ascending=False).reset_index(drop=True)
    return df


def collect_stock_daily(ts_code, fresh=False):
    """采集单只A股日线 (新浪)
    新浪单位: volume=股, amount=元 → 保存: vol=手, amount=千元
    """
    filepath = os.path.join(STOCK_DAILY_DIR, f"{ts_code}.csv")
    sina_symbol = ts_code_to_sina(ts_code)

    if not fresh and os.path.exists(filepath):
        latest = get_existing_latest_date(filepath)
        if latest:
            start = (datetime.strptime(latest, "%Y%m%d") + timedelta(days=1)).strftime("%Y%m%d")
            if start > datetime.now().strftime("%Y%m%d"):
                return 0  # 已最新
        else:
            start = "20100101"
    else:
        start = "20100101"

    end = datetime.now().strftime("%Y%m%d")

    for attempt in range(3):
        try:
            df = ak.stock_zh_a_daily(symbol=sina_symbol, start_date=start, end_date=end)
            if df is None or len(df) == 0:
                return 0

            df = normalize_stock_sina(df, ts_code)

            if not fresh and os.path.exists(filepath):
                return incremental_merge(filepath, df, ts_code)
            else:
                df.to_csv(filepath, index=False)
                return len(df)

        except Exception as e:
            if attempt < 2:
                time.sleep(2)
            else:
                return -1
    return 0


def collect_stock_weekly(ts_code, fresh=False):
    """采集单只A股周线 (新浪)
    新浪无周线接口，用日线聚合
    """
    filepath = os.path.join(STOCK_WEEKLY_DIR, f"{ts_code}.csv")
    sina_symbol = ts_code_to_sina(ts_code)

    if not fresh and os.path.exists(filepath):
        latest = get_existing_latest_date(filepath)
        if latest:
            start = (datetime.strptime(latest, "%Y%m%d") - timedelta(days=30)).strftime("%Y%m%d")
            if start > datetime.now().strftime("%Y%m%d"):
                return 0
        else:
            start = "20100101"
    else:
        start = "20100101"

    end = datetime.now().strftime("%Y%m%d")

    for attempt in range(3):
        try:
            df = ak.stock_zh_a_daily(symbol=sina_symbol, start_date=start, end_date=end)
            if df is None or len(df) == 0:
                return 0

            # 日线聚合成周线
            df = df.rename(columns={"date": "trade_date"})
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            df = df.sort_values("trade_date").set_index("trade_date")
            weekly = df.resample("W").agg({
                "open": "first", "high": "max", "low": "min", "close": "last",
                "volume": "sum", "amount": "sum"
            }).dropna()
            weekly = weekly.reset_index()
            weekly["trade_date"] = weekly["trade_date"].dt.strftime("%Y%m%d")

            df_norm = normalize_stock_sina(weekly, ts_code)

            if not fresh and os.path.exists(filepath):
                return incremental_merge(filepath, df_norm, ts_code)
            else:
                df_norm.to_csv(filepath, index=False)
                return len(df_norm)

        except Exception as e:
            if attempt < 2:
                time.sleep(2)
            else:
                return -1
    return 0


def collect_all_stocks_daily(fresh=False):
    """采集全部A股日线 (4进程并行)"""
    ensure_dir(STOCK_DAILY_DIR)
    stock_list = get_stock_list()
    total = len(stock_list)

    print(f"\n[Stock-Daily] 开始采集 {total} 只A股日线 ({'全量' if fresh else '增量'})")
    print(f"[Stock-Daily] 数据源: akshare/新浪 (4进程)")
    print(f"[Stock-Daily] 路径: {STOCK_DAILY_DIR}\n")

    # 分4批，每批单独python进程跑，避免mini_racer多线程崩溃
    import subprocess
    n_workers = 4
    batches = [[] for _ in range(n_workers)]
    for i, item in enumerate(stock_list):
        batches[i % n_workers].append(item)

    batch_logs = [f"/tmp/stock_daily_batch{b}.log" for b in range(n_workers)]
    procs = []
    script_path = os.path.abspath(__file__)

    for b in range(n_workers):
        # 写批次股票列表到临时文件
        batch_file = f"/tmp/stock_daily_batch{b}.txt"
        with open(batch_file, "w") as f:
            for tc, nm in batches[b]:
                f.write(f"{tc}\n")
        cmd = [
            sys.executable, "-u", script_path,
            "_batch_stock_daily", batch_file,
            "--fresh" if fresh else ""
        ]
        cmd = [c for c in cmd if c]
        p = subprocess.Popen(cmd, stdout=open(batch_logs[b], "w"), stderr=subprocess.STDOUT)
        procs.append(p)

    # 等待全部完成
    for p in procs:
        p.wait()

    # 汇总结果
    success = fail = skip = new_rows = 0
    for b in range(n_workers):
        log = batch_logs[b]
        if os.path.exists(log):
            for line in open(log):
                line = line.strip()
                if line.startswith("BATCH_RESULT:"):
                    parts = line.split(":")
                    success += int(parts[1])
                    skip += int(parts[2])
                    fail += int(parts[3])
                    new_rows += int(parts[4])

    print(f"\n[Stock-Daily] === 完成 === 总:{total} 成功:{success} 跳过:{skip} 失败:{fail} 新增:{new_rows}行")


def collect_all_stocks_weekly(fresh=False):
    """采集全部A股周线"""
    ensure_dir(STOCK_WEEKLY_DIR)
    stock_list = get_stock_list()
    total = len(stock_list)
    success = fail = skip = new_rows = 0

    print(f"\n[Stock-Weekly] 开始采集 {total} 只A股周线 ({'全量' if fresh else '增量'})")
    print(f"[Stock-Weekly] 数据源: akshare/新浪")
    print(f"[Stock-Weekly] 路径: {STOCK_WEEKLY_DIR}\n")

    for i, (ts_code, name) in enumerate(stock_list):
        n = collect_stock_weekly(ts_code, fresh)
        if n > 0:
            success += 1
            new_rows += n
        elif n == 0:
            skip += 1
        else:
            fail += 1

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{total}] 成功:{success} 跳过:{skip} 失败:{fail} 新增:{new_rows}行")
            time.sleep(3)
        elif (i + 1) % 10 == 0:
            time.sleep(0.5)

    print(f"\n[Stock-Weekly] === 完成 === 总:{total} 成功:{success} 跳过:{skip} 失败:{fail} 新增:{new_rows}行")


# ==================== 期货采集 (akshare/新浪) ====================

FUTURES_LIST = [
    # === 黑色系 ===
    ("RB0", "螺纹钢"), ("HC0", "热卷"), ("I0", "铁矿石"), ("J0", "焦炭"),
    ("JM0", "焦煤"), ("ZC0", "动力煤"), ("SF0", "硅铁"), ("SM0", "锰硅"),
    ("FG0", "玻璃"), ("SA0", "纯碱"), ("WR0", "线材"),
    # === 有色金属 ===
    ("CU0", "沪铜"), ("AL0", "沪铝"), ("ZN0", "沪锌"), ("PB0", "沪铅"),
    ("NI0", "沪镍"), ("SN0", "沪锡"), ("AO0", "氧化铝"), ("BC0", "国际铜"),
    ("SI0", "工业硅"), ("LC0", "碳酸锂"),
    # === 贵金属 ===
    ("AU0", "黄金"), ("AG0", "白银"),
    # === 能源化工 ===
    ("SC0", "原油"), ("FU0", "燃油"), ("BU0", "沥青"), ("RU0", "橡胶"),
    ("TA0", "PTA"), ("MA0", "甲醇"), ("EG0", "乙二醇"), ("PF0", "短纤"),
    ("PG0", "LPG"), ("PP0", "聚丙烯"), ("L0", "塑料"), ("V0", "PVC"),
    ("EB0", "苯乙烯"), ("UR0", "尿素"), ("BR0", "丁二烯橡胶"),
    ("SP0", "纸浆"), ("SH0", "烧碱"), ("EC0", "集运指数"),
    # === 农产品 ===
    ("C0", "玉米"), ("CS0", "淀粉"), ("A0", "豆一"), ("B0", "豆二"),
    ("M0", "豆粕"), ("Y0", "豆油"), ("P0", "棕榈油"), ("CF0", "棉花"),
    ("SR0", "白糖"), ("AP0", "苹果"), ("CJ0", "红枣"), ("PK0", "花生"),
    ("OI0", "菜油"), ("RM0", "菜粕"), ("RS0", "菜籽"), ("WH0", "强麦"),
    ("JR0", "粳稻"), ("LR0", "晚稻"), ("PM0", "普麦"), ("RR0", "粳米"),
    ("CY0", "棉纱"), ("JD0", "鸡蛋"), ("LH0", "生猪"),
    ("FB0", "纤维板"), ("BB0", "胶合板"),
    # === 股指期货 ===
    ("IF0", "沪深300"), ("IC0", "中证500"), ("IM0", "中证1000"), ("IH0", "上证50"),
    # === 国债期货 ===
    ("T0", "十年国债"), ("TF0", "五年国债"), ("TS0", "二年国债"), ("TL0", "三十年国债"),
]


def normalize_futures_and_save(df, filepath, ts_code, sort_desc=True):
    """
    统一保存期货数据为标准格式:
    ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount,oi
    新浪期货单位: volume=手, hold=手(持仓量), 无amount字段(用close*vol估算)
    """
    df = df.rename(columns={"date": "trade_date", "volume": "vol", "hold": "oi"})
    df["trade_date"] = df["trade_date"].astype(str).str.replace("-", "")
    df["ts_code"] = ts_code

    for col in ["open", "high", "low", "close", "vol", "oi"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 计算 amount (成交额，元) = close * vol * 乘数(简化为close*vol)
    if "amount" not in df.columns:
        df["amount"] = (df["close"] * df["vol"]).round(2)

    # 计算 pre_close / change / pct_chg
    df = df.sort_values("trade_date", ascending=True).reset_index(drop=True)
    closes = df["close"].astype(float)
    pre_close = closes.shift(1)
    df["pre_close"] = pre_close.round(2)
    df["change"] = (closes - pre_close).round(2)
    df["pct_chg"] = ((closes - pre_close) / pre_close * 100).round(2)

    df.loc[0, ["pre_close", "change", "pct_chg"]] = np.nan

    target = ["ts_code", "trade_date", "open", "high", "low", "close",
              "pre_close", "change", "pct_chg", "vol", "amount", "oi"]
    for col in target:
        if col not in df.columns:
            df[col] = ""

    df = df[target]

    if sort_desc:
        df = df.sort_values("trade_date", ascending=False).reset_index(drop=True)

    df["open"] = pd.to_numeric(df["open"], errors="coerce").round(2)
    df["high"] = pd.to_numeric(df["high"], errors="coerce").round(2)
    df["low"] = pd.to_numeric(df["low"], errors="coerce").round(2)
    df["close"] = pd.to_numeric(df["close"], errors="coerce").round(2)
    df["vol"] = pd.to_numeric(df["vol"], errors="coerce").round(0)
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").round(2)
    df["oi"] = pd.to_numeric(df["oi"], errors="coerce").round(0)

    df.to_csv(filepath, index=False)
    return len(df)


def collect_futures(symbol, fresh=False):
    """采集单个期货品种主力连续日线 (新浪)
    新浪单位: volume=手, hold=手
    """
    filepath = os.path.join(FUTURES_DAILY_DIR, f"{symbol}.csv")

    for attempt in range(3):
        try:
            df = ak.futures_zh_daily_sina(symbol=symbol)
            if len(df) == 0:
                return 0

            if not fresh and os.path.exists(filepath):
                return incremental_merge_futures(filepath, df, symbol)
            else:
                return normalize_futures_and_save(df, filepath, symbol)

        except Exception as e:
            if attempt < 2:
                time.sleep(2)
            else:
                print(f"[Futures] {symbol} 失败: {e}")
                return -1
    return 0


def incremental_merge_futures(old_path, new_df, ts_code):
    """增量合并期货数据（含oi持仓量）"""
    if not os.path.exists(old_path):
        return normalize_futures_and_save(new_df, old_path, ts_code)

    old_df = pd.read_csv(old_path, dtype=str)
    old_df["trade_date"] = old_df["trade_date"].astype(str).str.replace("-", "")

    new_df = new_df.copy()
    new_df = new_df.rename(columns={"date": "trade_date", "volume": "vol", "hold": "oi"})
    new_df["trade_date"] = new_df["trade_date"].astype(str).str.replace("-", "")

    # 只取新日期的数据
    existing_dates = set(old_df["trade_date"].tolist())
    new_df = new_df[~new_df["trade_date"].isin(existing_dates)]

    if len(new_df) == 0:
        return 0

    # 转换新数据并合并
    new_df["ts_code"] = ts_code
    for col in ["open", "high", "low", "close", "vol", "oi"]:
        if col in new_df.columns:
            new_df[col] = pd.to_numeric(new_df[col], errors="coerce")

    if "amount" not in new_df.columns:
        new_df["amount"] = (new_df["close"] * new_df["vol"]).round(2)

    # 计算涨跌
    new_df = new_df.sort_values("trade_date", ascending=True).reset_index(drop=True)
    closes = new_df["close"].astype(float)
    pre_close = closes.shift(1)
    new_df["pre_close"] = pre_close.round(2)
    new_df["change"] = (closes - pre_close).round(2)
    new_df["pct_chg"] = ((closes - pre_close) / pre_close * 100).round(2)

    target = ["ts_code", "trade_date", "open", "high", "low", "close",
              "pre_close", "change", "pct_chg", "vol", "amount", "oi"]
    for col in target:
        if col not in new_df.columns:
            new_df[col] = ""

    new_df = new_df[target]

    # 合并
    combined = pd.concat([old_df, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["trade_date"], keep="last")
    combined = combined.sort_values("trade_date", ascending=False).reset_index(drop=True)
    combined.to_csv(old_path, index=False)

    return len(new_df)


def collect_all_futures(fresh=False):
    """采集全部期货日线"""
    ensure_dir(FUTURES_DAILY_DIR)
    total = len(FUTURES_LIST)
    success = fail = new_rows = 0

    print(f"\n[Futures] 开始采集 {total} 个期货品种 ({'全量' if fresh else '增量'})")
    print(f"[Futures] 数据源: akshare/新浪")
    print(f"[Futures] 路径: {FUTURES_DAILY_DIR}\n")

    for i, (symbol, name) in enumerate(FUTURES_LIST):
        n = collect_futures(symbol, fresh)
        if n > 0:
            success += 1
            new_rows += n
            print(f"  [{i+1}/{total}] {symbol} {name}: +{n}行")
        elif n == 0:
            fp = os.path.join(FUTURES_DAILY_DIR, f"{symbol}.csv")
            if os.path.exists(fp):
                cnt = len(pd.read_csv(fp))
                print(f"  [{i+1}/{total}] {symbol} {name}: 已最新 ({cnt}行)")
            else:
                print(f"  [{i+1}/{total}] {symbol} {name}: 无数据")
        else:
            fail += 1
        time.sleep(1)

    print(f"\n[Futures] === 完成 === 总:{total} 成功:{success} 失败:{fail} 新增:{new_rows}行")


# ==================== 修复已有期货CSV排序 ====================

def fix_futures_sort_order():
    """修复已有期货CSV：改成新数据在上面（降序），与A股格式一致"""
    if not os.path.exists(FUTURES_DAILY_DIR):
        return

    print("[Fix] 修复期货CSV排序（改为新数据在上面）...")
    fixed = 0
    for f in os.listdir(FUTURES_DAILY_DIR):
        if not f.endswith(".csv"):
            continue
        filepath = os.path.join(FUTURES_DAILY_DIR, f)
        try:
            df = pd.read_csv(filepath, dtype=str)
            if len(df) < 2:
                continue
            dates = df["trade_date"].astype(str)
            if dates.iloc[0] < dates.iloc[-1]:
                df = df.iloc[::-1].reset_index(drop=True)
                df.to_csv(filepath, index=False)
                fixed += 1
        except:
            pass
    print(f"[Fix] 修复了 {fixed} 个文件\n")


# ==================== 主函数 ====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="金融数据采集 (A股+期货)")
    parser.add_argument("mode",
        choices=["stock_daily", "stock_weekly", "futures", "all", "fix", "_batch_stock_daily"],
        help="stock_daily | stock_weekly | futures | all | fix(仅修复排序)")
    parser.add_argument("batch_file", nargs="?", default=None, help="批次文件(内部用)")
    parser.add_argument("--fresh", action="store_true", help="全量重采（默认增量）")
    args = parser.parse_args()

    # 子进程模式：处理一批股票
    if args.mode == "_batch_stock_daily":
        if not args.batch_file or not os.path.exists(args.batch_file):
            print("BATCH_RESULT:0:0:0:0")
            sys.exit(1)
        codes = [l.strip() for l in open(args.batch_file) if l.strip()]
        success = fail = skip = new_rows = 0
        for i, ts_code in enumerate(codes):
            n = collect_stock_daily(ts_code, args.fresh)
            if n > 0:
                success += 1
                new_rows += n
            elif n == 0:
                skip += 1
            else:
                fail += 1
            if (i + 1) % 100 == 0:
                print(f"  [{i+1}/{len(codes)}] 成功:{success} 跳过:{skip} 失败:{fail} 新增:{new_rows}行")
                sys.stdout.flush()
        print(f"BATCH_RESULT:{success}:{skip}:{fail}:{new_rows}")
        sys.exit(0)

    print("=" * 60)
    print(f"金融数据采集工具 v4 (akshare/新浪)")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"模式: {args.mode} | {'全量' if args.fresh else '增量'}")
    print("=" * 60)

    # 先修复期货排序
    if args.mode in ("futures", "all", "fix"):
        fix_futures_sort_order()

    if args.mode in ("stock_daily", "all"):
        collect_all_stocks_daily(args.fresh)

    if args.mode in ("stock_weekly", "all"):
        collect_all_stocks_weekly(args.fresh)

    if args.mode in ("futures", "all"):
        collect_all_futures(args.fresh)

    print("\n全部完成！")
