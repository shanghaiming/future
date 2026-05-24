#!/usr/bin/env python3
"""期货分析平台 API 后端 - 为React看板提供数据"""
import os, sys, json, glob
import pandas as pd
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from datetime import datetime

BASE_DIR = os.path.expanduser("~/home/futures_platform")
DATA_DIR = os.path.join(BASE_DIR, "data")
TS_DIR = os.path.join(DATA_DIR, "futures_term_structure")
OPT_DIR = os.path.join(DATA_DIR, "options")
OPT_CALC_DIR = os.path.join(DATA_DIR, "options_calculated")
FUT_DIR = os.path.join(DATA_DIR, "futures_weighted")

app = Flask(__name__)
CORS(app)

# ============ 缓存 ============
_ts_cache = None
_opt_cache = None
_fut_cache = None
_opt_calc_cache = None
_opt_summary_cache = None


_ts_index_cache = None  # 品种→日期列表的索引


def build_ts_index():
    """构建期限结构文件索引（不读内容，只扫描文件名）"""
    global _ts_index_cache
    if _ts_index_cache is not None:
        return _ts_index_cache
    _ts_index_cache = {}  # symbol -> [(date_str, filepath), ...]
    if not os.path.isdir(TS_DIR):
        return _ts_index_cache
    for fname in os.listdir(TS_DIR):
        if not fname.endswith('.json') and not fname.endswith('.json.gz'):
            continue
        is_gz = fname.endswith('.gz')
        base = fname.replace('.gz', '') if is_gz else fname
        parts = base.rsplit('_', 1)
        if len(parts) != 2:
            continue
        sym = parts[0]
        date_str = parts[1].replace('.json', '')
        fpath = os.path.join(TS_DIR, fname)
        if sym not in _ts_index_cache:
            _ts_index_cache[sym] = []
        _ts_index_cache[sym].append((date_str, fpath, is_gz))
    # 每个品种按日期排序
    for sym in _ts_index_cache:
        _ts_index_cache[sym].sort(key=lambda x: x[0])
    return _ts_index_cache


def _read_json_file(fpath, is_gz=False):
    """读取JSON文件，支持gzip"""
    import gzip
    if is_gz:
        with gzip.open(fpath, 'rt', encoding='utf-8') as fp:
            return json.load(fp)
    else:
        with open(fpath) as fp:
            return json.load(fp)


def load_ts_data():
    """加载所有期限结构数据 - 用索引加速"""
    global _ts_cache
    if _ts_cache is not None:
        return _ts_cache
    records = []
    index = build_ts_index()
    for sym, file_list in index.items():
        # 只加载最新一天的数据用于概览
        # 历史数据按需加载（ts_history接口）
        if not file_list:
            continue
        # 加载最新一天的
        latest_date, latest_path, is_gz = file_list[-1]
        try:
            d = _read_json_file(latest_path, is_gz)
            curve = d.get('curve', [])
            rec = {
                'symbol': d.get('symbol', sym),
                'date': d.get('date', latest_date),
                'structure': d.get('structure', ''),
                'near_contract': d.get('near_contract', ''),
                'near_price': d.get('near_price', 0),
                'far_contract': d.get('far_contract', ''),
                'far_price': d.get('far_price', 0),
                'total_spread': d.get('total_spread', 0),
                'total_spread_pct': d.get('total_spread_pct', 0),
                'curve': curve,
            }
            records.append(rec)
        except:
            continue
    _ts_cache = pd.DataFrame(records)
    if len(_ts_cache) > 0:
        _ts_cache['date'] = pd.to_datetime(_ts_cache['date'])
    return _ts_cache


def load_opt_data():
    global _opt_cache
    if _opt_cache is not None:
        return _opt_cache
    records = []
    for f in glob.glob(os.path.join(OPT_DIR, '*.json')):
        try:
            with open(f) as fp:
                d = json.load(fp)
            if isinstance(d, list):
                for item in d:
                    item['_source'] = os.path.basename(f)
                    records.append(item)
            else:
                d['_source'] = os.path.basename(f)
                surface = d.get('surface', [])
                for s in surface:
                    s['symbol'] = d.get('symbol', '')
                    s['date'] = d.get('date', '')
                    s['underlying_price'] = d.get('underlying_price', 0)
                    s['hv_20'] = d.get('hv_20', 0)
                    s['hv_60'] = d.get('hv_60', 0)
                    records.append(s)
        except:
            continue
    _opt_cache = pd.DataFrame(records)
    return _opt_cache


def load_fut_data():
    global _fut_cache
    if _fut_cache is not None:
        return _fut_cache
    all_data = {}
    for f in sorted(glob.glob(os.path.join(FUT_DIR, '*.csv'))):
        sym = os.path.basename(f).replace('.csv', '')
        df = pd.read_csv(f)
        if len(df) < 10: continue
        df['trade_date'] = pd.to_datetime(df['trade_date'], format='mixed')
        df = df.sort_values('trade_date').reset_index(drop=True)
        all_data[sym] = df
    _fut_cache = all_data
    return _fut_cache


def load_opt_calc_data():
    """加载计算后的期权数据(真实IV和Greeks)"""
    global _opt_calc_cache, _opt_summary_cache
    if _opt_calc_cache is not None:
        return _opt_calc_cache, _opt_summary_cache

    # Load summary
    summary_path = os.path.join(OPT_CALC_DIR, "iv_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            _opt_summary_cache = json.load(f)
    else:
        _opt_summary_cache = []

    # Load all options with calculated IV
    all_path = os.path.join(OPT_CALC_DIR, "all_options_with_iv.json")
    if os.path.exists(all_path):
        with open(all_path) as f:
            _opt_calc_cache = json.load(f)
    else:
        # Load individual files
        records = []
        for fpath in sorted(glob.glob(os.path.join(OPT_CALC_DIR, '*.json'))):
            fname = os.path.basename(fpath)
            if fname in ('iv_summary.json', 'all_options_with_iv.json'):
                continue
            try:
                with open(fpath) as f:
                    data = json.load(f)
                if isinstance(data, list):
                    records.extend(data)
            except:
                continue
        _opt_calc_cache = records

    return _opt_calc_cache, _opt_summary_cache


# ============ API 路由 ============

@app.route('/api/overview')
def overview():
    """总览数据"""
    ts = load_ts_data()
    opt = load_opt_data()
    fut = load_fut_data()

    ts_symbols = ts['symbol'].nunique() if len(ts) > 0 else 0
    ts_dates = len(ts)
    opt_symbols = opt['symbol'].nunique() if 'symbol' in opt.columns and len(opt) > 0 else 0
    opt_contracts = len(opt)
    fut_symbols = len(fut)

    latest_ts = ts['date'].max().strftime('%Y-%m-%d') if len(ts) > 0 else 'N/A'
    if len(ts) > 0:
        latest = ts[ts['date'] == ts['date'].max()]
        back_count = len(latest[latest['structure'] == 'backwardation'])
        cont_count = len(latest[latest['structure'] == 'contango'])
    else:
        back_count = cont_count = 0

    return jsonify({
        'ts_symbols': ts_symbols,
        'ts_records': ts_dates,
        'ts_latest_date': latest_ts,
        'opt_symbols': opt_symbols,
        'opt_contracts': opt_contracts,
        'fut_symbols': fut_symbols,
        'backwardation_count': int(back_count),
        'contango_count': int(cont_count),
    })


@app.route('/api/ts/structure')
def ts_structure():
    """期限结构概览 - 当前各品种状态"""
    ts = load_ts_data()
    if len(ts) == 0:
        return jsonify([])

    latest_date = ts['date'].max()
    latest = ts[ts['date'] == latest_date].copy()
    latest = latest.sort_values('total_spread_pct')

    result = []
    for _, row in latest.iterrows():
        curve_data = row.get('curve', [])
        if isinstance(curve_data, str):
            try: curve_data = json.loads(curve_data)
            except: curve_data = []
        result.append({
            'symbol': row['symbol'],
            'date': row['date'].strftime('%Y-%m-%d'),
            'structure': row['structure'],
            'near_price': float(row['near_price']) if row['near_price'] else 0,
            'far_price': float(row['far_price']) if row['far_price'] else 0,
            'total_spread_pct': float(row['total_spread_pct']) if row['total_spread_pct'] else 0,
            'curve': curve_data,
        })
    return jsonify(result)


@app.route('/api/ts/history/<symbol>')
def ts_history(symbol):
    """单个品种期限结构历史 - 按需加载"""
    index = build_ts_index()
    if symbol not in index:
        return jsonify([])

    result = []
    for date_str, fpath, is_gz in index[symbol]:
        try:
            d = _read_json_file(fpath, is_gz)
            result.append({
                'date': d.get('date', date_str.replace('', '-').replace('--', '-')),
                'spread_pct': float(d.get('total_spread_pct', 0)) if d.get('total_spread_pct') is not None else None,
                'structure': d.get('structure', ''),
                'near_price': float(d.get('near_price', 0)) if d.get('near_price') else 0,
                'far_price': float(d.get('far_price', 0)) if d.get('far_price') else 0,
            })
        except:
            continue
    return jsonify(result)


@app.route('/api/ts/curve/<symbol>/<date>')
def ts_curve(symbol, date):
    """某个品种某天的完整曲线"""
    ts = load_ts_data()
    if len(ts) == 0:
        return jsonify({})

    mask = (ts['symbol'] == symbol) & (ts['date'] == pd.Timestamp(date))
    rows = ts[mask]
    if len(rows) == 0:
        return jsonify({})

    row = rows.iloc[0]
    curve = row.get('curve', [])
    if isinstance(curve, str):
        try: curve = json.loads(curve)
        except: curve = []

    return jsonify({
        'symbol': symbol,
        'date': date,
        'structure': row['structure'],
        'curve': curve,
    })


@app.route('/api/options/surface')
def options_surface():
    """期权波动率曲面数据 - 优先使用计算后的真实IV数据"""
    symbol = request.args.get('symbol', '')
    date = request.args.get('date', '')

    calc_data, _ = load_opt_calc_data()

    if calc_data:
        result = []
        for rec in calc_data:
            match = True
            if symbol and rec.get('product', '').lower() != symbol.lower():
                match = False
            if date and rec.get('date', '') != date:
                match = False
            if match:
                result.append(rec)
        return jsonify(result)

    # Fallback to synthetic data
    opt = load_opt_data()
    if len(opt) == 0:
        return jsonify([])

    df = opt.copy()
    if 'symbol' in df.columns and symbol:
        df = df[df['symbol'] == symbol]
    if 'date' in df.columns and date:
        df = df[df['date'] == date]

    if len(df) == 0:
        return jsonify([])

    result = []
    for _, row in df.iterrows():
        rec = {}
        for col in ['symbol', 'date', 'moneyness', 'expiry_days', 'flag',
                     'iv', 'delta', 'gamma', 'theta', 'vega', 'rho',
                     'strike', 'price', 'underlying_price', 'hv_20', 'hv_60']:
            if col in df.columns:
                val = row.get(col)
                if pd.isna(val): rec[col] = None
                elif isinstance(val, (np.integer,)): rec[col] = int(val)
                elif isinstance(val, (np.floating,)): rec[col] = float(val)
                else: rec[col] = val
        result.append(rec)
    return jsonify(result)


@app.route('/api/options/symbols')
def options_symbols():
    """可用期权品种列表 - 优先使用计算后的数据"""
    calc_data, calc_summary = load_opt_calc_data()

    if calc_data:
        symbols = sorted(set(r.get('product', '') for r in calc_data if r.get('product')))
        dates = sorted(set(r.get('date', '') for r in calc_data if r.get('date')))
        return jsonify({'symbols': symbols, 'dates': dates})

    # Fallback
    opt = load_opt_data()
    if len(opt) == 0 or 'symbol' not in opt.columns:
        return jsonify([])

    symbols = opt['symbol'].dropna().unique().tolist()
    dates = opt['date'].dropna().unique().tolist() if 'date' in opt.columns else []
    return jsonify({'symbols': symbols, 'dates': [str(d) for d in dates]})


@app.route('/api/options/iv_summary')
def options_iv_summary():
    """各品种IV汇总统计 - 优先使用计算后的真实IV数据"""
    calc_data, calc_summary = load_opt_calc_data()

    if calc_summary:
        return jsonify(calc_summary)

    # Fallback to synthetic data
    opt = load_opt_data()
    if len(opt) == 0:
        return jsonify([])

    results = []
    if 'symbol' not in opt.columns:
        return jsonify([])

    for sym in opt['symbol'].dropna().unique():
        sym_df = opt[opt['symbol'] == sym]
        if len(sym_df) == 0: continue

        if 'moneyness' in sym_df.columns and 'iv' in sym_df.columns:
            atm = sym_df[(sym_df['moneyness'] >= 0.96) & (sym_df['moneyness'] <= 1.04)]
            atm_iv = atm['iv'].mean() if len(atm) > 0 else None

            puts_low = sym_df[(sym_df['flag'] == 'p') & (sym_df['moneyness'] <= 0.88)]
            calls_high = sym_df[(sym_df['flag'] == 'c') & (sym_df['moneyness'] >= 1.12)]
            put_iv = puts_low['iv'].mean() if len(puts_low) > 0 else None
            call_iv = calls_high['iv'].mean() if len(calls_high) > 0 else None
            skew = (put_iv - call_iv) if put_iv and call_iv else None
        else:
            atm_iv = None
            skew = None

        hv20 = sym_df['hv_20'].iloc[0] if 'hv_20' in sym_df.columns and len(sym_df) > 0 else None
        hv60 = sym_df['hv_60'].iloc[0] if 'hv_60' in sym_df.columns and len(sym_df) > 0 else None
        underlying = sym_df['underlying_price'].iloc[0] if 'underlying_price' in sym_df.columns else None

        iv_hv_ratio = (atm_iv / hv20) if atm_iv and hv20 and hv20 > 0 else None

        results.append({
            'symbol': sym,
            'underlying_price': float(underlying) if underlying else None,
            'atm_iv': float(atm_iv) if atm_iv else None,
            'skew': float(skew) if skew else None,
            'hv_20': float(hv20) if hv20 else None,
            'hv_60': float(hv60) if hv60 else None,
            'iv_hv_ratio': float(iv_hv_ratio) if iv_hv_ratio else None,
            'n_contracts': len(sym_df),
        })

    results.sort(key=lambda x: x.get('atm_iv') or 0, reverse=True)
    return jsonify(results)


@app.route('/api/futures/symbols')
def futures_symbols():
    """期货品种列表"""
    fut = load_fut_data()
    result = []
    for sym, df in fut.items():
        latest = df.iloc[-1]
        ret_5d = (df['close'].iloc[-1] / df['close'].iloc[-6] - 1) * 100 if len(df) > 5 else 0
        ret_20d = (df['close'].iloc[-1] / df['close'].iloc[-21] - 1) * 100 if len(df) > 20 else 0
        vol_20d = df['close'].pct_change().tail(20).std() * np.sqrt(252) * 100 if len(df) > 20 else 0
        result.append({
            'symbol': sym,
            'close': float(latest['close']),
            'date': latest['trade_date'].strftime('%Y-%m-%d'),
            'ret_5d': round(float(ret_5d), 2),
            'ret_20d': round(float(ret_20d), 2),
            'vol_20d': round(float(vol_20d), 2),
            'volume': float(latest.get('vol', 0)),
            'oi': float(latest.get('oi', 0)),
            'n_days': len(df),
        })
    return jsonify(result)


@app.route('/api/futures/price/<symbol>')
def futures_price(symbol):
    """期货价格历史"""
    fut = load_fut_data()
    if symbol not in fut:
        return jsonify([])
    df = fut[symbol]
    days = min(len(df), 250)
    sub = df.tail(days)
    result = []
    for _, row in sub.iterrows():
        result.append({
            'date': row['trade_date'].strftime('%Y-%m-%d'),
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close']),
            'volume': float(row.get('vol', 0)),
            'oi': float(row.get('oi', 0)),
        })
    return jsonify(result)


if __name__ == '__main__':
    print("预加载数据...")
    load_ts_data()
    load_opt_data()
    load_fut_data()
    load_opt_calc_data()
    print("启动API服务: http://localhost:5001")
    app.run(host='0.0.0.0', port=5001, debug=False)
