#!/usr/bin/env python3
"""
期货加权指数日线数据采集 - 天勤(tqsdk)数据源
加权合约按成交量加权，不存在换月跳空问题，适合量化回测。

数据源: tqsdk (天勤量化)
单位: vol=手, amount=close*vol(估算元), oi=手(持仓量)

注意: tqsdk免费账户每分钟有请求限制，品种间需适当延时
"""

import os
import sys
import time
import signal
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tqsdk import TqApi, TqAuth

# ==================== 配置 ====================
BASE_DATA_DIR = os.path.expanduser("~/home/futures_platform/data")
FUTURES_WEIGHTED_DIR = os.path.join(BASE_DATA_DIR, "futures_weighted")

TQ_ACCOUNT = os.environ.get("TQ_ACCOUNT", "18844561230")
TQ_PASSWORD = os.environ.get("TQ_PASSWORD", "zxcvbnm0717")

# 品种列表: (tqsdk加权代码, CSV文件名, 品种名称)
# tqsdk加权合约代码: KQ.m@EXCHANGE.symbol
WEIGHTED_LIST = [
    # === 黑色系 ===
    ("KQ.m@SHFE.rb", "rbfi", "螺纹钢加权"), ("KQ.m@SHFE.hc", "hcfi", "热卷加权"),
    ("KQ.m@SHFE.i", "ifi", "铁矿石加权"), ("KQ.m@DCE.j", "jfi", "焦炭加权"),
    ("KQ.m@DCE.jm", "jmfi", "焦煤加权"), ("KQ.m@CZCE.ZC", "ZCFI", "动力煤加权"),
    ("KQ.m@CZCE.SF", "SFFI", "硅铁加权"), ("KQ.m@CZCE.SM", "SMFI", "锰硅加权"),
    ("KQ.m@CZCE.FG", "FGFI", "玻璃加权"), ("KQ.m@CZCE.SA", "SAFI", "纯碱加权"),
    ("KQ.m@SHFE.wr", "wrfi", "线材加权"),
    # === 有色金属 ===
    ("KQ.m@SHFE.cu", "cufi", "沪铜加权"), ("KQ.m@SHFE.al", "alfi", "沪铝加权"),
    ("KQ.m@SHFE.zn", "znfi", "沪锌加权"), ("KQ.m@SHFE.pb", "pbfi", "沪铅加权"),
    ("KQ.m@SHFE.ni", "nifi", "沪镍加权"), ("KQ.m@SHFE.sn", "snfi", "沪锡加权"),
    ("KQ.m@SHFE.ao", "aofi", "氧化铝加权"), ("KQ.m@INE.bc", "bcfi", "国际铜加权"),
    ("KQ.m@GFEX.si", "sifi", "工业硅加权"), ("KQ.m@GFEX.lc", "lcfi", "碳酸锂加权"),
    ("KQ.m@SHFE.ss", "ssfi", "不锈钢加权"),
    # === 贵金属 ===
    ("KQ.m@SHFE.au", "aufi", "沪金加权"), ("KQ.m@SHFE.ag", "agfi", "沪银加权"),
    # === 能源化工 ===
    ("KQ.m@INE.sc", "scfi", "原油加权"), ("KQ.m@SHFE.fu", "fufi", "燃油加权"),
    ("KQ.m@INE.lu", "lufi", "低硫燃油加权"), ("KQ.m@SHFE.bu", "bufi", "沥青加权"),
    ("KQ.m@SHFE.ru", "rufi", "橡胶加权"), ("KQ.m@INE.nr", "nrfi", "20号胶加权"),
    ("KQ.m@CZCE.TA", "TAFI", "PTA加权"), ("KQ.m@CZCE.MA", "MAFI", "甲醇加权"),
    ("KQ.m@DCE.eg", "egfi", "乙二醇加权"), ("KQ.m@CZCE.PF", "PFFI", "短纤加权"),
    ("KQ.m@DCE.pg", "pgfi", "LPG加权"), ("KQ.m@DCE.pp", "ppfi", "聚丙烯加权"),
    ("KQ.m@DCE.l", "lfi", "塑料加权"), ("KQ.m@DCE.v", "vfi", "PVC加权"),
    ("KQ.m@DCE.eb", "ebfi", "苯乙烯加权"), ("KQ.m@CZCE.UR", "URFI", "尿素加权"),
    ("KQ.m@SHFE.br", "brfi", "丁二烯胶加权"), ("KQ.m@SHFE.sp", "spfi", "纸浆加权"),
    ("KQ.m@CZCE.SH", "SHFI", "烧碱加权"), ("KQ.m@INE.ec", "ecfi", "欧线集运加权"),
    ("KQ.m@CZCE.PX", "PXFI", "对二甲苯加权"), ("KQ.m@INE.bz", "bzfi", "纯苯加权"),
    ("KQ.m@DCE.pl", "PLFI", "丙烯加权"), ("KQ.m@CZCE.PR", "PRFI", "瓶片加权"),
    # === 农产品 ===
    ("KQ.m@DCE.c", "cfi", "玉米加权"), ("KQ.m@DCE.cs", "csfi", "淀粉加权"),
    ("KQ.m@DCE.a", "afi", "豆一加权"), ("KQ.m@DCE.b", "bfi", "豆二加权"),
    ("KQ.m@DCE.m", "mfi", "豆粕加权"), ("KQ.m@DCE.y", "yfi", "豆油加权"),
    ("KQ.m@DCE.p", "pfi", "棕榈油加权"), ("KQ.m@CZCE.CF", "CFFI", "棉花加权"),
    ("KQ.m@CZCE.SR", "SRFI", "白糖加权"), ("KQ.m@CZCE.AP", "APFI", "苹果加权"),
    ("KQ.m@CZCE.CJ", "CJFI", "红枣加权"), ("KQ.m@CZCE.PK", "PKFI", "花生加权"),
    ("KQ.m@CZCE.OI", "OIFI", "菜油加权"), ("KQ.m@CZCE.RM", "RMFI", "菜粕加权"),
    ("KQ.m@CZCE.RS", "RSFI", "菜籽加权"), ("KQ.m@CZCE.WH", "WHFI", "强麦加权"),
    ("KQ.m@CZCE.JR", "JRFI", "粳稻加权"), ("KQ.m@CZCE.LR", "LRFI", "晚籼稻加权"),
    ("KQ.m@CZCE.PM", "PMFI", "普麦加权"), ("KQ.m@CZCE.RI", "RIFI", "早籼稻加权"),
    ("KQ.m@DCE.rr", "rrfi", "粳米加权"), ("KQ.m@CZCE.CY", "CYFI", "棉纱加权"),
    ("KQ.m@CZCE.op", "opfi", "胶版纸加权"), ("KQ.m@DCE.jd", "jdfi", "鸡蛋加权"),
    ("KQ.m@DCE.lh", "lhfi", "生猪加权"), ("KQ.m@DCE.fb", "fbfi", "纤维板加权"),
    ("KQ.m@DCE.bb", "bbfi", "胶合板加权"),
    # === 新品种 ===
    ("KQ.m@GFEX.ps", "psfi", "多晶硅加权"), ("KQ.m@DCE.lg", "lgfi", "原木加权"),
]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def get_existing_latest_date(filepath):
    """读取CSV第一行（最新日期）"""
    if not os.path.exists(filepath):
        return None
    try:
        df = pd.read_csv(filepath, nrows=1, dtype=str)
        if len(df) == 0:
            return None
        val = df["trade_date"].iloc[0]
        if val == "nan" or pd.isna(val):
            return None
        return str(val)
    except:
        return None


def fetch_tqsdk_kline(api, symbol, data_length=8000):
    """
    从tqsdk获取期货加权指数日线K线
    tqsdk K线字段: datetime, open, high, low, close, volume, close_oi, open_oi
    注意: K线无amount字段，用 close*vol 估算成交额
    volume=手, close_oi=手(收盘持仓)

    返回: DataFrame with trade_date,open,high,low,close,vol,amount,oi
    """
    try:
        kl = api.get_kline_serial(symbol, 86400, data_length)

        # 盘后wait_update可能卡住，限时等待数据到达
        import time as _time
        deadline = _time.time() + 60
        while _time.time() < deadline:
            try:
                api.wait_update(deadline=_time.time() + 5)
            except:
                break
            # 检查是否有有效数据（close>0）
            if api.is_changing(kl):
                valid = kl[kl["close"] > 0]
                if len(valid) > 100:
                    break

        df = kl.copy()
        # 过滤NaT
        df = df[df["datetime"] != "NaT"]
        df = df.dropna(subset=["datetime"])

        if len(df) == 0:
            return pd.DataFrame()

        # 转换日期: tqsdk返回纳秒时间戳
        df["trade_date"] = pd.to_datetime(df["datetime"]).dt.strftime("%Y%m%d")

        # tqsdk K线无amount字段，用close*vol估算
        result = pd.DataFrame()
        result["trade_date"] = df["trade_date"]
        result["open"] = df["open"].astype(float).round(2)
        result["high"] = df["high"].astype(float).round(2)
        result["low"] = df["low"].astype(float).round(2)
        result["close"] = df["close"].astype(float).round(2)
        result["vol"] = df["volume"].astype(float).round(0)
        result["amount"] = (df["close"].astype(float) * df["volume"].astype(float)).round(2)
        result["oi"] = df["close_oi"].astype(float).round(0)

        # 降序
        result = result.sort_values("trade_date", ascending=False).reset_index(drop=True)
        return result

    except Exception as e:
        print(f"  [ERROR] {symbol}: {e}")
        return pd.DataFrame()


def incremental_update(filepath, new_df, symbol):
    """增量更新CSV：合并去重，保持降序"""
    if not os.path.exists(filepath):
        new_df = new_df.copy()
        new_df.insert(0, "ts_code", symbol)
        new_df.to_csv(filepath, index=False)
        return len(new_df)

    old_df = pd.read_csv(filepath, dtype=str)
    old_df["trade_date"] = old_df["trade_date"].astype(str).str.replace("-", "")

    new_df = new_df.copy()
    new_df["trade_date"] = new_df["trade_date"].astype(str).str.replace("-", "")

    combined = pd.concat([old_df, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["trade_date"], keep="last")

    for col in ["open", "high", "low", "close", "vol", "amount", "oi"]:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")

    combined = combined.sort_values("trade_date", ascending=False).reset_index(drop=True)
    combined["ts_code"] = symbol

    cols = ["ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount", "oi"]
    for c in cols:
        if c not in combined.columns:
            combined[c] = ""
    combined = combined[cols]

    combined.to_csv(filepath, index=False)

    # 返回新增行数
    old_dates = set(old_df["trade_date"].tolist())
    new_dates = set(new_df["trade_date"].tolist())
    return len(new_dates - old_dates)


def collect_all(fresh=False):
    """采集全部期货加权指数日线"""
    ensure_dir(FUTURES_WEIGHTED_DIR)
    total = len(WEIGHTED_LIST)
    success = fail = skip = new_rows = 0

    print("=" * 60)
    print(f"期货加权指数日线采集 (tqsdk)")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"品种数: {total} | 模式: {'全量' if fresh else '增量'}")
    print(f"路径: {FUTURES_WEIGHTED_DIR}")
    print(f"单位: vol=手, amount=close*vol(估算元), oi=手")
    print("=" * 60)

    try:
        api = None
        for i, (tq_symbol, csv_name, display_name) in enumerate(WEIGHTED_LIST):
            # 每10个品种重建tqsdk连接，防连接断开
            if i % 10 == 0:
                if api is not None:
                    try: api.close()
                    except: pass
                api = TqApi(auth=TqAuth(TQ_ACCOUNT, TQ_PASSWORD))

            csv_path = os.path.join(FUTURES_WEIGHTED_DIR, f"{csv_name.lower()}.csv")

            # 增量模式: 已有数据则只取最近部分
            if not fresh and os.path.exists(csv_path):
                latest = get_existing_latest_date(csv_path)
                if latest and latest >= datetime.now().strftime("%Y%m%d"):
                    skip += 1
                    if (i + 1) % 10 == 0:
                        print(f"  [{i+1}/{total}] {csv_name} {display_name}: 已最新")
                    time.sleep(0.1)
                    continue

            # 拉取数据
            df = fetch_tqsdk_kline(api, tq_symbol, data_length=8000)

            if len(df) == 0:
                if os.path.exists(csv_path):
                    skip += 1
                else:
                    fail += 1
                    print(f"  [{i+1}/{total}] {csv_name} {display_name}: 无数据")
            else:
                n = incremental_update(csv_path, df, csv_name)
                if n > 0:
                    success += 1
                    new_rows += n
                    print(f"  [{i+1}/{total}] {csv_name} {display_name}: +{n}行")
                else:
                    skip += 1

            # tqsdk免费账户限流: 每次请求间隔
            time.sleep(0.5)

        if api is not None:
            try: api.close()
            except: pass

    except Exception as e:
        print(f"\n[ERROR] tqsdk连接失败: {e}")
        print("请检查天勤账号密码是否正确")
        return

    print()
    print(f"=== 完成 === 总:{total} 成功:{success} 跳过:{skip} 失败:{fail} 新增:{new_rows}行")

    if fail > 5:
        print(f"\n  !! 严重警告: {fail}个品种数据拉取失败!")
        print(f"  !! 可能原因: tqsdk账号权限/品种代码变更/网络问题")
    elif fail > 0:
        print(f"\n  [!] 注意: {fail}个品种失败(可能是已退市或暂停交易)")


if __name__ == "__main__":
    import argparse, subprocess
    parser = argparse.ArgumentParser(description="期货加权指数日线采集 (tqsdk)")
    parser.add_argument("mode", nargs="?", default="all", help="all | _fetch_one(内部)")
    parser.add_argument("--fresh", action="store_true", help="全量重采")
    parser.add_argument("--symbol", default=None, help="tqsdk品种代码(内部)")
    parser.add_argument("--csv-name", default=None, help="csv文件名(内部)")
    parser.add_argument("--display-name", default=None, help="品种名(内部)")
    args = parser.parse_args()

    if args.mode == "_fetch_one":
        # 单品种采集：独立tqsdk连接
        csv_path = os.path.join(FUTURES_WEIGHTED_DIR, f"{args.csv_name.lower()}.csv")
        try:
            api = TqApi(auth=TqAuth(TQ_ACCOUNT, TQ_PASSWORD))
            df = fetch_tqsdk_kline(api, args.symbol, data_length=8000)
            if len(df) > 0:
                n = incremental_update(csv_path, df, args.csv_name)
                print(f"OK:{args.csv_name}:{n}")
            else:
                print(f"EMPTY:{args.csv_name}")
            api.close()
        except Exception as e:
            print(f"FAIL:{args.csv_name}:{str(e)[:80]}")
        sys.exit(0)

    # 主进程：逐品种spawn子进程
    ensure_dir(FUTURES_WEIGHTED_DIR)
    total = len(WEIGHTED_LIST)
    success = fail = skip = new_rows = 0

    print("=" * 60)
    print(f"期货加权指数日线采集 (tqsdk, 逐品种独立连接)")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"品种数: {total} | 模式: {'全量' if args.fresh else '增量'}")
    print(f"路径: {FUTURES_WEIGHTED_DIR}")
    print(f"单位: vol=手, amount=close*vol(估算元), oi=手")
    print("=" * 60)

    for i, (tq_symbol, csv_name, display_name) in enumerate(WEIGHTED_LIST):
        csv_path = os.path.join(FUTURES_WEIGHTED_DIR, f"{csv_name.lower()}.csv")

        if not args.fresh and os.path.exists(csv_path):
            latest = get_existing_latest_date(csv_path)
            if latest and latest >= datetime.now().strftime("%Y%m%d"):
                skip += 1
                continue

        cmd = [sys.executable, "-u", os.path.abspath(__file__),
               "_fetch_one", "--fresh" if args.fresh else "",
               "--symbol", tq_symbol,
               "--csv-name", csv_name,
               "--display-name", display_name]
        cmd = [c for c in cmd if c]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
            output = (result.stdout or "").strip().split("\n")[-1]
            if output.startswith("OK:"):
                parts = output.split(":")
                n = int(parts[2]) if len(parts) > 2 else 0
                success += 1
                new_rows += n
                print(f"  [{i+1}/{total}] {csv_name} {display_name}: +{n}行")
            elif output.startswith("EMPTY:"):
                skip += 1
                print(f"  [{i+1}/{total}] {csv_name} {display_name}: 无数据")
            else:
                fail += 1
                err = output if output else result.stderr.strip().split("\n")[-1] if result.stderr else "unknown"
                print(f"  [{i+1}/{total}] {csv_name} {display_name}: FAIL {err[:60]}")
        except subprocess.TimeoutExpired:
            fail += 1
            print(f"  [{i+1}/{total}] {csv_name} {display_name}: 超时")
        except Exception as e:
            fail += 1
            print(f"  [{i+1}/{total}] {csv_name} {display_name}: 异常 {str(e)[:60]}")

        time.sleep(0.3)

    print()
    print(f"=== 完成 === 总:{total} 成功:{success} 跳过:{skip} 失败:{fail} 新增:{new_rows}行")
    if fail > 5:
        print(f"\n  !! 严重警告: {fail}个品种数据拉取失败!")
    elif fail > 0:
        print(f"\n  [!] 注意: {fail}个品种失败(可能是已退市或暂停交易)")
