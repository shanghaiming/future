"""
V15.8 — 最终验证: 因果Wavelet + 买卖都用次日open
"""
import sys, os, time, pickle, importlib, warnings, signal as sig_module
import numpy as np, pandas as pd
from collections import defaultdict
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.data_loader import list_available_symbols, load_stock_data

COMMISSION = 0.0003
STAMP_DUTY = 0.001
INITIAL_CASH = 500_000

print("=" * 70, flush=True)
print("  V15.8 — 因果Wavelet + 买卖均次日open", flush=True)
print("=" * 70, flush=True)

# 加载股票数据
print("\n[Step 1] 加载股票数据...", flush=True)
all_stock_data = {}
symbols = list_available_symbols('daily')
for sym in symbols:
    try:
        df = load_stock_data(sym, frequency='daily')
        if df is not None and len(df) >= 500:
            cols = [c for c in ['open', 'high', 'low', 'close', 'vol', 'volume', 'amount'] if c in df.columns]
            all_stock_data[sym] = df[cols].copy()
            if 'vol' in all_stock_data[sym].columns and 'volume' not in all_stock_data[sym].columns:
                all_stock_data[sym] = all_stock_data[sym].rename(columns={'vol': 'volume'})
    except: pass
sym_volumes = {}
for sym, df in all_stock_data.items():
    if 'volume' in df.columns:
        vol = df['volume'].tail(60).mean()
        if not np.isnan(vol) and vol > 0:
            sym_volumes[sym] = vol
target_syms = sorted([s for s, _ in sorted(sym_volumes.items(), key=lambda x: -x[1])[:200]])
print(f"  {len(target_syms)} stocks", flush=True)


# ============================================================
# 重新生成因果WaveletStrategy信号
# ============================================================
print("\n[Step 2] 生成因果WaveletStrategy信号...", flush=True)

STRATEGIES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'strategies')

def load_strategy_class(class_name):
    for fname in os.listdir(STRATEGIES_DIR):
        if not fname.endswith('.py') or fname.startswith('_'): continue
        fpath = os.path.join(STRATEGIES_DIR, fname)
        try:
            spec = importlib.util.spec_from_file_location(f"s_{fname[:-3]}", fpath)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if hasattr(mod, class_name):
                return getattr(mod, class_name)
        except: continue
    return None

def timeout_handler(signum, frame):
    raise TimeoutError()

WaveletCls = load_strategy_class('WaveletStrategy')
wavelet_sigs = {}
for sym in target_syms:
    df = all_stock_data.get(sym)
    if df is None or len(df) < 100: continue
    try:
        old_handler = sig_module.signal(sig_module.SIGALRM, timeout_handler)
        sig_module.alarm(30)
        try:
            s = WaveletCls(df, {})
            signals = s.generate_signals()
            sig_module.alarm(0)
            if signals:
                clean = []
                for sd in signals:
                    ts = sd.get('timestamp')
                    action = sd.get('action')
                    price = sd.get('price', 0)
                    if ts and action in ('buy', 'sell'):
                        clean.append((pd.Timestamp(ts), action, float(price) if price else 0))
                if clean:
                    wavelet_sigs[sym] = clean
        except TimeoutError:
            sig_module.alarm(0)
        except:
            sig_module.alarm(0)
    except: pass

n_buys = sum(1 for sigs in wavelet_sigs.values() for _, a, _ in sigs if a == 'buy')
print(f"  {n_buys} buy signals, {len(wavelet_sigs)} stocks", flush=True)

# 验证无 look-ahead
diffs = []
for sym, sigs in list(wavelet_sigs.items())[:20]:
    df = all_stock_data.get(sym)
    if df is None: continue
    for ts, action, price in sigs:
        if ts in df.index:
            close = float(df.loc[ts, 'close'])
            diffs.append((price - close) / close * 100)
if diffs:
    print(f"  信号价 vs close: mean={np.mean(diffs):.4f}% max={np.max(np.abs(diffs)):.4f}%", flush=True)
    print(f"  {'OK - 无价格偏差' if np.max(np.abs(diffs)) < 0.01 else 'WARNING!'}", flush=True)


# ============================================================
# 回测引擎 — 买卖均用次日open
# ============================================================
def simulate_causal(strat_sigs, price_data, config):
    """全因果回测: 买卖信号均在次日open执行"""
    wide_sl = config.get('wide_sl', 10)
    profit_lock = config.get('profit_lock', 15)
    tight_ts = config.get('tight_ts', 10)
    max_pos = config.get('max_positions', 1)
    start_date = pd.Timestamp(config.get('start_date', '2016-01-01'))
    end_date = pd.Timestamp(config.get('end_date', '2026-01-19'))

    # 聚合信号
    buy_sigs = defaultdict(list)
    sell_sigs = defaultdict(list)
    dates_set = set()
    for sym, sigs in strat_sigs.items():
        for ts, action, price in sigs:
            if start_date <= ts <= end_date:
                dates_set.add(ts)
                if action == 'buy': buy_sigs[ts].append(sym)
                elif action == 'sell': sell_sigs[ts].append(sym)

    all_dates = sorted(dates_set)
    if not all_dates: return None

    def get_close(sym, date):
        df = price_data.get(sym)
        if df is None: return None
        mask = df.index <= date
        if mask.sum() == 0: return None
        return float(df['close'].loc[mask].iloc[-1])

    def get_next_open(sym, date):
        df = price_data.get(sym)
        if df is None: return None
        future = df[df.index > date]
        if len(future) == 0: return None
        return float(future.iloc[0]['open'])

    cash = float(INITIAL_CASH)
    holdings = {}
    pending_buys = []   # (sym, signal_date) — 次日执行
    pending_sells = []  # (sym, reason, signal_date) — 次日执行
    trades = []

    for date in all_dates:
        # === 先执行昨天的待执行交易 (用今天的open) ===
        for sym in list(pending_sells):
            if sym not in holdings:
                pending_sells.remove(sym)
                continue
            sell_price = get_next_open(sym, holdings[sym].get('last_signal_date', date))
            # 用今天的open卖出
            sell_price = None
            df = price_data.get(sym)
            if df is not None and date in df.index:
                sell_price = float(df.loc[date, 'open'])
            if sell_price and sell_price > 0:
                info = holdings[sym]
                pnl_pct = (sell_price - info['entry_price']) / info['entry_price'] * 100
                amt = info['shares'] * sell_price
                cash += amt - amt * COMMISSION - amt * STAMP_DUTY
                trades.append({'action': 'sell', 'symbol': sym, 'date': date,
                              'price': sell_price, 'shares': info['shares'],
                              'entry_price': info['entry_price'], 'pnl_pct': pnl_pct})
                del holdings[sym]
            pending_sells.remove(sym)

        for sym in list(pending_buys):
            if sym in holdings or len(holdings) >= max_pos:
                pending_buys.remove(sym)
                continue
            buy_price = None
            df = price_data.get(sym)
            if df is not None and date in df.index:
                buy_price = float(df.loc[date, 'open'])
            if buy_price and buy_price > 0:
                pv = cash
                for s, info in holdings.items():
                    p = get_close(s, date)
                    if p: pv += info['shares'] * p
                per_stock = pv / (len(holdings) + 1) / (1 + COMMISSION + STAMP_DUTY)
                shares = int(per_stock / buy_price)
                if shares > 0 and cash >= shares * buy_price * (1 + COMMISSION):
                    cash -= shares * buy_price * (1 + COMMISSION)
                    holdings[sym] = {'shares': shares, 'entry_price': buy_price,
                                    'highest': buy_price, 'entry_date': date}
                    trades.append({'action': 'buy', 'symbol': sym, 'date': date,
                                  'price': buy_price, 'shares': shares})
            pending_buys.remove(sym)

        # === 止损/止盈检查 (用当天close判断, 次日open执行) ===
        for sym, info in list(holdings.items()):
            p = get_close(sym, date)
            if p is None: continue
            entry = info['entry_price']
            highest = info.get('highest', p)
            if p > highest: info['highest'] = p; highest = p
            pnl_pct = (p - entry) / entry * 100
            if pnl_pct < -wide_sl:
                if sym not in pending_sells:
                    pending_sells.append(sym)
            elif pnl_pct >= profit_lock and p < highest:
                if (highest - p) / highest * 100 > tight_ts:
                    if sym not in pending_sells:
                        pending_sells.append(sym)

        # === 今天的信号 → 明天执行 ===
        for sym in sell_sigs[date]:
            if sym in holdings and sym not in pending_sells:
                pending_sells.append(sym)

        for sym in buy_sigs[date]:
            if sym not in holdings and sym not in pending_buys and len(holdings) < max_pos:
                pending_buys.append(sym)

    # 清仓
    if holdings:
        last_date = all_dates[-1]
        for sym, info in list(holdings.items()):
            p = get_close(sym, last_date)
            if p and p > 0:
                cash += info['shares'] * p * (1 - COMMISSION - STAMP_DUTY)
                trades.append({'action': 'sell', 'symbol': sym, 'date': last_date,
                              'price': p, 'shares': info['shares'],
                              'entry_price': info['entry_price'],
                              'pnl_pct': (p - info['entry_price']) / info['entry_price'] * 100})

    if cash <= 0: return None
    final = cash
    days = (all_dates[-1] - all_dates[0]).days
    yr = max(days / 365.25, 0.01)
    ann = ((final / INITIAL_CASH) ** (1 / yr) - 1) * 100

    yearly = defaultdict(lambda: {'pnl': 0, 'trades': 0, 'wins': 0})
    for t in trades:
        if t['action'] == 'sell':
            yk = pd.Timestamp(t['date']).year
            yearly[yk]['trades'] += 1
            pnl = (t['price'] - t['entry_price']) * t['shares']
            pnl -= t['price'] * t['shares'] * (COMMISSION + STAMP_DUTY)
            pnl -= t['entry_price'] * t['shares'] * COMMISSION
            yearly[yk]['pnl'] += pnl
            if pnl > 0: yearly[yk]['wins'] += 1

    sells = [t for t in trades if t['action'] == 'sell']
    nw = sum(1 for t in sells if t['pnl_pct'] > 0)
    wr = nw / max(len(sells), 1) * 100
    return {'annualized': round(ann, 1), 'final_value': round(final, 0),
            'n_trades': len(sells), 'win_rate': round(wr, 1),
            'yearly': dict(yearly), 'config': config}


# ============================================================
# 跑优化
# ============================================================
print("\n[Step 3] 因果回测 (买卖均次日open)...", flush=True)

configs = []
for wide_sl in [5, 8, 10, 15, 20]:
    for profit_lock in [8, 10, 15, 20, 30]:
        for tight_ts in [5, 8, 10, 15]:
            configs.append({
                'max_positions': 1,
                'wide_sl': wide_sl, 'profit_lock': profit_lock, 'tight_ts': tight_ts,
                'start_date': '2016-01-01', 'end_date': '2026-01-19',
            })

print(f"  {len(configs)} configs", flush=True)
results = []
t0 = time.time()
for i, cfg in enumerate(configs):
    r = simulate_causal(wavelet_sigs, all_stock_data, cfg)
    if r and r['annualized'] > 0:
        results.append(r)
    if (i+1) % 50 == 0:
        best = max(r['annualized'] for r in results) if results else 0
        print(f"  [{i+1}/{len(configs)}] best={best:+.1f}%", flush=True)

results.sort(key=lambda x: -x['annualized'])
print(f"  Done in {time.time()-t0:.0f}s", flush=True)

print("\n  Top 5:", flush=True)
for r in results[:5]:
    c = r['config']
    print(f"    w{c['wide_sl']}/l{c['profit_lock']}/t{c['tight_ts']}: "
          f"{r['annualized']:+.1f}% ann | {r['n_trades']}t WR={r['win_rate']:.0f}% | "
          f"终值={r['final_value']/10000:.0f}万", flush=True)

if results:
    best = results[0]
    c = best['config']
    print(f"\n  最佳年度 (w{c['wide_sl']}/l{c['profit_lock']}/t{c['tight_ts']}):", flush=True)
    for yr in sorted(best['yearly'].keys()):
        yd = best['yearly'][yr]
        wr = yd['wins'] / max(yd['trades'], 1) * 100
        print(f"    {yr}: {yd['pnl']/10000:+.1f}万 ({yd['trades']}t, WR={wr:.0f}%)", flush=True)

# 每年独立50万测试
print("\n[Step 4] 每年独立50万...", flush=True)
best_cfg = results[0]['config'] if results else {'wide_sl': 8, 'profit_lock': 15, 'tight_ts': 10}
for year in range(2016, 2027):
    start = pd.Timestamp(f'{year}-01-01')
    end = pd.Timestamp(f'{year}-12-31')
    if year == 2026: end = pd.Timestamp('2026-04-25')
    r = simulate_causal(wavelet_sigs, all_stock_data, {**best_cfg, 'start_date': str(start.date()), 'end_date': str(end.date())})
    if r:
        print(f"  {year}: 终值={r['final_value']/10000:.1f}万 | +{((r['final_value']/INITIAL_CASH)-1)*100:.0f}% | "
              f"{r['n_trades']}t WR={r['win_rate']:.0f}%", flush=True)

print("\nDone!", flush=True)
