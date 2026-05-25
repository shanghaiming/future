"""
Alpha Futures V17 — Cross-Commodity Relative Value (Pair Trading)
=================================================================
Market-neutral strategy: long the underperformer, short the overperformer
within correlated commodity pairs. Profits from spread mean-reversion.

KEY INSIGHT for futures pair trading:
  - Use LOG(price_A / price_B) as the spread for better stationarity
  - Z-score computed on log-spread over a rolling window
  - When z > threshold: A is expensive vs B → short A, long B
  - When z < -threshold: A is cheap vs B → long A, short B
  - Exit when spread reverts to mean (z crosses zero)
  - Futures PnL = direction * (exit_price - entry_price) * lots * multiplier

VARIANTS:
  ZSCORE_2  : Enter at |z|=2.0, exit at z=0
  ZSCORE_15 : Enter at |z|=1.5, exit at |z|=0.5
  ZSCORE_25 : Enter at |z|=2.5, exit at |z|=0.5
  MOM_RATIO : Use 10-day return ratio instead of price ratio for spread
  OI_WEIGHTED: Weight allocation by OI change (stronger OI leg gets more)

Each variant tested with hold_max=[5,10,15] and stop_loss=[0.03,0.05].
"""
import sys, os, time, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alpha_v2 import load_all_data, MIN_TRAIN

# ── constants ──────────────────────────────────────────────────
CASH0 = 500_000

MULT = {
    'agfi': 15, 'alfi': 5, 'aufi': 1000, 'bufi': 10, 'cufi': 5,
    'fufi': 10, 'rbfi': 10, 'znfi': 5, 'nifi': 1, 'hcfi': 10,
    'spfi': 10, 'ssfi': 5, 'sffi': 5, 'smfi': 5, 'pbfi': 5,
    'snfi': 1, 'rufi': 10, 'wrfff': 10, 'afi': 10, 'bfi': 10,
    'bbfi': 500, 'cffi': 5, 'cfi': 10, 'csfi': 10, 'ebfi': 5,
    'egfi': 10, 'fbfi': 500, 'ifi': 100, 'jfi': 100, 'jmfi': 60,
    'lfi': 5, 'mfi': 10, 'pgfi': 20, 'ppfi': 5, 'vfi': 5,
    'yfi': 10, 'pfi': 10, 'jdfi': 5, 'lhfi': 16, 'pkfi': 5,
    'rrfi': 20, 'lrfi': 20, 'jrfi': 20, 'pmfi': 20, 'whfi': 20,
    'rsfi': 20, 'cjfi': 10, 'mafi': 10, 'apfi': 10, 'cyfi': 5,
    'fgfi': 20, 'oifi': 10, 'pfifi': 5, 'rmfi': 10, 'srfi': 10,
    'tafi': 5, 'safi': 20, 'urfi': 20, 'scfi': 1000, 'lufi': 10,
    'bcfi': 5, 'nrfi': 1, 'lgfi': 20, 'brfi': 5, 'lcfi': 1,
    'sifi': 5, 'ni': 1, 'tai': 5,
}
DEF_MULT = 10
COMM = 0.0003

# Supply-chain correlated pairs
PAIRS = [
    ('rbfi', 'ifi',  'Steel vs IronOre'),
    ('rbfi', 'jfi',  'Steel vs Coke'),
    ('jfi',  'jmfi', 'Coke vs Coal'),
    ('afi',  'mfi',  'Soybean vs Meal'),
    ('mfi',  'yfi',  'Meal vs Oil'),
    ('cufi', 'alfi', 'Copper vs Aluminum'),
    ('scfi', 'mafi', 'Oil vs Methanol'),
    ('scfi', 'tafi', 'Oil vs PTA'),
    ('ppfi', 'vfi',  'PP vs PVC'),
    ('ppfi', 'lfi',  'PP vs PE'),
]

ZSCORE_WINDOW = 20


# ── data loading ───────────────────────────────────────────────
def load_pair_data():
    """Load data only for symbols involved in pairs."""
    print("[Data] Loading pair symbols ...", flush=True)
    t0 = time.time()

    needed = set()
    for a, b, _ in PAIRS:
        needed.add(a)
        needed.add(b)

    ret = load_all_data(max_stocks=500, min_days=60, start='2013-01-01',
                        load_oi=True)
    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = ret

    sym_idx = {s: i for i, s in enumerate(syms)}
    avail = {s: sym_idx[s] for s in needed if s in sym_idx}
    missing = needed - set(avail.keys())
    if missing:
        print(f"  WARNING: missing symbols: {missing}", flush=True)

    print(f"  {len(avail)} pair symbols found, {ND} days ({time.time()-t0:.1f}s)",
          flush=True)
    return NS, ND, dates, C, O, H, L, V, OI, syms, sym_set, avail


# ── spread / signal helpers ────────────────────────────────────
def rolling_zscore_fast(arr, window):
    """Vectorized rolling z-score with NaN handling."""
    n = len(arr)
    z = np.full(n, np.nan)
    if n < window:
        return z
    # cumsum approach for speed
    valid_mask = ~np.isnan(arr)
    arr_filled = np.where(valid_mask, arr, 0.0)
    count = np.cumsum(valid_mask).astype(float)
    cumsum = np.cumsum(arr_filled)
    cumsum2 = np.cumsum(arr_filled ** 2)

    for i in range(window - 1, n):
        c_end = count[i]
        c_start = count[i - window] if i >= window else 0
        n_valid = c_end - c_start
        if n_valid < window // 2:
            continue
        s_end = cumsum[i]
        s_start = cumsum[i - window] if i >= window else 0
        s2_end = cumsum2[i]
        s2_start = cumsum2[i - window] if i >= window else 0
        s = s_end - s_start
        s2 = s2_end - s2_start
        m = s / n_valid
        var = s2 / n_valid - m * m
        if var > 1e-20 and valid_mask[i]:
            z[i] = (arr[i] - m) / np.sqrt(var)
    return z


def compute_spread(c_a, c_b, nd, variant='ZSCORE_2'):
    """
    Compute the spread series for z-scoring.
    For most variants: log(price_A / price_B).
    For MOM_RATIO: 10-day return of A minus 10-day return of B.
    """
    if variant == 'MOM_RATIO':
        spread = np.full(nd, np.nan)
        mw = 10
        for i in range(mw, nd):
            if (np.isnan(c_a[i]) or np.isnan(c_a[i - mw]) or
                    np.isnan(c_b[i]) or np.isnan(c_b[i - mw])):
                continue
            if c_a[i - mw] > 0 and c_b[i - mw] > 0:
                ret_a = np.log(c_a[i] / c_a[i - mw])
                ret_b = np.log(c_b[i] / c_b[i - mw])
                spread[i] = ret_a - ret_b
        return spread
    else:
        spread = np.full(nd, np.nan)
        for i in range(nd):
            if not np.isnan(c_a[i]) and not np.isnan(c_b[i]) and c_a[i] > 0 and c_b[i] > 0:
                spread[i] = np.log(c_a[i] / c_b[i])
        return spread


def compute_oi_change(oi_arr, nd, window=5):
    """OI percentage change over window."""
    chg = np.full(nd, np.nan)
    for i in range(window, nd):
        if not np.isnan(oi_arr[i]) and not np.isnan(oi_arr[i - window]) \
                and oi_arr[i - window] > 0:
            chg[i] = (oi_arr[i] - oi_arr[i - window]) / oi_arr[i - window]
    return chg


# ── pair backtest engine ───────────────────────────────────────
def backtest_pair(
    sym_a, sym_b, pair_name,
    idx_a, idx_b,
    NS, ND, dates, C, O, H, L, V, OI,
    variant='ZSCORE_2',
    hold_max=10,
    stop_loss=0.05,
):
    """
    Backtest a single pair with given variant/params.

    Futures model:
      - We track equity as cash + unrealized PnL
      - When opening: deduct notional margin from cash, track position
      - PnL per leg = direction * (current_price - entry_price) * lots * mult
      - When closing: add realized PnL back to cash

    Returns dict of metrics or None.
    """
    c_a = C[idx_a]
    c_b = C[idx_b]
    o_a = O[idx_a]
    o_b = O[idx_b]
    oi_a = OI[idx_a]
    oi_b = OI[idx_b]

    ma = MULT.get(sym_a, DEF_MULT)
    mb = MULT.get(sym_b, DEF_MULT)

    # compute signal series using CLOSE prices
    spread = compute_spread(c_a, c_b, ND, variant)
    z = rolling_zscore_fast(spread, ZSCORE_WINDOW)

    oi_chg_a = compute_oi_change(oi_a, ND, 5) if variant == 'OI_WEIGHTED' else None
    oi_chg_b = compute_oi_change(oi_b, ND, 5) if variant == 'OI_WEIGHTED' else None

    entry_z, exit_z = {
        'ZSCORE_2':  (2.0, 0.0),
        'ZSCORE_15': (1.5, 0.5),
        'ZSCORE_25': (2.5, 0.5),
        'MOM_RATIO': (2.0, 0.0),
        'OI_WEIGHTED': (2.0, 0.0),
    }.get(variant, (2.0, 0.0))

    # simulation
    cash = float(CASH0)
    pos = None
    trades = []

    for di in range(MIN_TRAIN, ND):
        # execution prices
        ep_a = o_a[di] if not np.isnan(o_a[di]) and o_a[di] > 0 else c_a[di]
        ep_b = o_b[di] if not np.isnan(o_b[di]) and o_b[di] > 0 else c_b[di]
        if np.isnan(ep_a) or ep_a <= 0 or np.isnan(ep_b) or ep_b <= 0:
            continue

        # ── exit check ──
        if pos is not None:
            d = pos['dir']
            # mark-to-market with close prices
            ca = c_a[di]
            cb = c_b[di]
            if np.isnan(ca) or np.isnan(cb):
                continue

            pnl_a = d * (ca - pos['entry_a']) * ma * pos['lots_a']
            pnl_b = -d * (cb - pos['entry_b']) * mb * pos['lots_b']
            mtm_pnl = pnl_a + pnl_b

            hold_days = di - pos['entry_day']
            cap = pos['capital']

            # stop-loss: 3% of notional
            sl_hit = cap > 0 and (mtm_pnl / cap) < -stop_loss

            # exit signal from z-score
            exit_signal = False
            zz = z[di]
            if not np.isnan(zz):
                if d == 1 and zz <= exit_z:
                    exit_signal = True
                elif d == -1 and zz >= -exit_z:
                    exit_signal = True

            if exit_signal or sl_hit or hold_days >= hold_max:
                # close at execution price (open of this bar)
                realized_a = d * (ep_a - pos['entry_a']) * ma * pos['lots_a']
                realized_b = -d * (ep_b - pos['entry_b']) * mb * pos['lots_b']
                gross = realized_a + realized_b
                comm_cost = (pos['lots_a'] * ep_a * ma + pos['lots_b'] * ep_b * mb) * COMM
                net_pnl = gross - comm_cost

                cash += cap + net_pnl  # return capital + profit

                reason = 'sl' if sl_hit else ('hold' if hold_days >= hold_max else 'signal')
                trades.append({
                    'pnl_pct': net_pnl / cap * 100 if cap > 0 else 0,
                    'pnl_abs': net_pnl,
                    'days': hold_days,
                    'di': di,
                    'reason': reason,
                    'dir': d,
                    'year': dates[di].year,
                })
                pos = None

        # ── entry check ──
        if pos is None:
            zz = z[di]
            if np.isnan(zz):
                continue
            direction = 0
            if zz > entry_z:
                direction = -1   # short A, long B (A expensive)
            elif zz < -entry_z:
                direction = 1    # long A, short B (A cheap)

            if direction != 0:
                # allocate: 50% of cash per leg
                alloc_per_leg = cash * 0.5

                lots_a = max(1, int(alloc_per_leg / (ep_a * ma)))
                lots_b = max(1, int(alloc_per_leg / (ep_b * mb)))

                # OI_WEIGHTED: tilt
                if variant == 'OI_WEIGHTED' and oi_chg_a is not None and oi_chg_b is not None:
                    oca = oi_chg_a[di] if not np.isnan(oi_chg_a[di]) else 0
                    ocb = oi_chg_b[di] if not np.isnan(oi_chg_b[di]) else 0
                    total_oi = abs(oca) + abs(ocb)
                    if total_oi > 1e-8:
                        frac_a = abs(oca) / total_oi
                        frac_b = abs(ocb) / total_oi
                    else:
                        frac_a = frac_b = 0.5
                    lots_a = max(1, int(cash * frac_a / (ep_a * ma)))
                    lots_b = max(1, int(cash * frac_b / (ep_b * mb)))

                notional = lots_a * ep_a * ma + lots_b * ep_b * mb

                if notional > cash:
                    scale = cash * 0.95 / notional
                    lots_a = max(1, int(lots_a * scale))
                    lots_b = max(1, int(lots_b * scale))
                    notional = lots_a * ep_a * ma + lots_b * ep_b * mb

                if lots_a > 0 and lots_b > 0 and notional <= cash:
                    cash -= notional
                    pos = {
                        'dir': direction,
                        'entry_a': ep_a,
                        'entry_b': ep_b,
                        'lots_a': lots_a,
                        'lots_b': lots_b,
                        'entry_day': di,
                        'capital': notional,
                    }

    # close any open position at end
    if pos is not None:
        ca = c_a[ND - 1]
        cb = c_b[ND - 1]
        if not np.isnan(ca) and ca > 0 and not np.isnan(cb) and cb > 0:
            realized_a = pos['dir'] * (ca - pos['entry_a']) * ma * pos['lots_a']
            realized_b = -pos['dir'] * (cb - pos['entry_b']) * mb * pos['lots_b']
            gross = realized_a + realized_b
            comm_cost = (pos['lots_a'] * ca * ma + pos['lots_b'] * cb * mb) * COMM
            net_pnl = gross - comm_cost
            cash += pos['capital'] + net_pnl
            trades.append({
                'pnl_pct': net_pnl / pos['capital'] * 100 if pos['capital'] > 0 else 0,
                'pnl_abs': net_pnl,
                'days': ND - 1 - pos['entry_day'],
                'di': ND - 1,
                'reason': 'end',
                'dir': pos['dir'],
                'year': dates[ND - 1].year,
            })

    if not trades:
        return None

    final_equity = cash
    # add back unrealized if pos still open (shouldn't be, but safety)
    if final_equity <= 0:
        return None

    days_total = (dates[ND - 1] - dates[MIN_TRAIN]).days
    yr = max(days_total / 365.25, 0.01)
    ann_ret = ((final_equity / CASH0) ** (1 / yr) - 1) * 100

    n_trades = len(trades)
    nw = sum(1 for t in trades if t['pnl_pct'] > 0)
    wr = nw / max(n_trades, 1) * 100
    avg_w = np.mean([t['pnl_pct'] for t in trades if t['pnl_pct'] > 0]) if nw > 0 else 0
    nl = n_trades - nw
    avg_l = np.mean([abs(t['pnl_pct']) for t in trades if t['pnl_pct'] <= 0]) if nl > 0 else 0
    edge = (nw / max(n_trades, 1)) * avg_w - (nl / max(n_trades, 1)) * avg_l
    tpy = n_trades / yr

    # max drawdown from equity curve
    eq = CASH0
    peak = CASH0
    max_dd = 0.0
    for t in sorted(trades, key=lambda x: x['di']):
        eq += t['pnl_abs']
        if eq > peak:
            peak = eq
        if peak > 0:
            dd = (peak - eq) / peak * 100
            if dd > max_dd:
                max_dd = dd

    return {
        'name': f"{pair_name}|{variant}|h{hold_max}|sl{stop_loss}",
        'pair': pair_name,
        'variant': variant,
        'hold': hold_max,
        'sl': stop_loss,
        'ann': round(ann_ret, 1),
        'n': n_trades,
        'wr': round(wr, 1),
        'avg_w': round(avg_w, 2),
        'avg_l': round(avg_l, 2),
        'edge': round(edge, 2),
        'max_dd': round(max_dd, 1),
        'tpy': round(tpy, 1),
        'final': round(final_equity, 0),
        'trades': trades,
        'dates': dates,
        'yr': yr,
    }


# ── portfolio backtest: run all pairs together ─────────────────
def backtest_portfolio(
    NS, ND, dates, C, O, H, L, V, OI, avail,
    variant='ZSCORE_2',
    hold_max=10,
    stop_loss=0.05,
    max_positions=5,
):
    """
    Run all pairs simultaneously with shared capital.
    Each pair can open at most 1 position. Max max_positions open at once.
    """
    # precompute signals for each pair
    pair_signals = []
    for sym_a, sym_b, label in PAIRS:
        if sym_a not in avail or sym_b not in avail:
            continue
        idx_a = avail[sym_a]
        idx_b = avail[sym_b]
        c_a, c_b = C[idx_a], C[idx_b]
        o_a, o_b = O[idx_a], O[idx_b]
        oi_a, oi_b = OI[idx_a], OI[idx_b]
        ma = MULT.get(sym_a, DEF_MULT)
        mb = MULT.get(sym_b, DEF_MULT)

        spread = compute_spread(c_a, c_b, ND, variant)
        z = rolling_zscore_fast(spread, ZSCORE_WINDOW)

        entry_z, exit_z = {
            'ZSCORE_2':  (2.0, 0.0),
            'ZSCORE_15': (1.5, 0.5),
            'ZSCORE_25': (2.5, 0.5),
            'MOM_RATIO': (2.0, 0.0),
            'OI_WEIGHTED': (2.0, 0.0),
        }.get(variant, (2.0, 0.0))

        pair_signals.append({
            'label': label, 'sym_a': sym_a, 'sym_b': sym_b,
            'idx_a': idx_a, 'idx_b': idx_b,
            'c_a': c_a, 'c_b': c_b, 'o_a': o_a, 'o_b': o_b,
            'ma': ma, 'mb': mb,
            'z': z, 'entry_z': entry_z, 'exit_z': exit_z,
        })

    if not pair_signals:
        return None

    # simulate
    cash = float(CASH0)
    positions = []
    trades = []

    for di in range(MIN_TRAIN, ND):
        # ── exit existing positions ──
        new_positions = []
        for pos in positions:
            sig = pair_signals[pos['pair_idx']]
            d = pos['dir']
            ca = sig['c_a'][di]
            cb = sig['c_b'][di]
            if np.isnan(ca) or np.isnan(cb):
                new_positions.append(pos)
                continue

            ep_a = sig['o_a'][di] if not np.isnan(sig['o_a'][di]) and sig['o_a'][di] > 0 else ca
            ep_b = sig['o_b'][di] if not np.isnan(sig['o_b'][di]) and sig['o_b'][di] > 0 else cb
            if ep_a <= 0 or ep_b <= 0:
                new_positions.append(pos)
                continue

            pnl_a = d * (ca - pos['entry_a']) * sig['ma'] * pos['lots_a']
            pnl_b = -d * (cb - pos['entry_b']) * sig['mb'] * pos['lots_b']
            mtm_pnl = pnl_a + pnl_b
            hold_days = di - pos['entry_day']
            cap = pos['capital']

            sl_hit = cap > 0 and (mtm_pnl / cap) < -stop_loss
            exit_signal = False
            zz = sig['z'][di]
            if not np.isnan(zz):
                if d == 1 and zz <= sig['exit_z']:
                    exit_signal = True
                elif d == -1 and zz >= -sig['exit_z']:
                    exit_signal = True

            if exit_signal or sl_hit or hold_days >= hold_max:
                realized_a = d * (ep_a - pos['entry_a']) * sig['ma'] * pos['lots_a']
                realized_b = -d * (ep_b - pos['entry_b']) * sig['mb'] * pos['lots_b']
                gross = realized_a + realized_b
                comm_cost = (pos['lots_a'] * ep_a * sig['ma'] +
                             pos['lots_b'] * ep_b * sig['mb']) * COMM
                net_pnl = gross - comm_cost
                cash += cap + net_pnl

                reason = 'sl' if sl_hit else ('hold' if hold_days >= hold_max else 'signal')
                trades.append({
                    'pnl_pct': net_pnl / cap * 100 if cap > 0 else 0,
                    'pnl_abs': net_pnl,
                    'days': hold_days,
                    'di': di,
                    'reason': reason,
                    'dir': d,
                    'year': dates[di].year,
                    'pair': sig['label'],
                })
            else:
                new_positions.append(pos)
        positions = new_positions

        # ── entry: check all pairs for new signals ──
        open_pairs = {pos['pair_idx'] for pos in positions}
        n_open = len(positions)
        if n_open < max_positions:
            candidates = []
            for pi, sig in enumerate(pair_signals):
                if pi in open_pairs:
                    continue
                zz = sig['z'][di]
                if np.isnan(zz):
                    continue
                direction = 0
                if zz > sig['entry_z']:
                    direction = -1
                elif zz < -sig['entry_z']:
                    direction = 1
                if direction != 0:
                    candidates.append((pi, direction, abs(zz)))
            candidates.sort(key=lambda x: -x[2])
            slots = max_positions - n_open
            for pi, direction, _ in candidates[:slots]:
                sig = pair_signals[pi]
                ep_a = sig['o_a'][di] if not np.isnan(sig['o_a'][di]) and sig['o_a'][di] > 0 else sig['c_a'][di]
                ep_b = sig['o_b'][di] if not np.isnan(sig['o_b'][di]) and sig['o_b'][di] > 0 else sig['c_b'][di]
                if np.isnan(ep_a) or ep_a <= 0 or np.isnan(ep_b) or ep_b <= 0:
                    continue

                alloc_per_leg = cash / max_positions * 0.5
                lots_a = max(1, int(alloc_per_leg / (ep_a * sig['ma'])))
                lots_b = max(1, int(alloc_per_leg / (ep_b * sig['mb'])))
                notional = lots_a * ep_a * sig['ma'] + lots_b * ep_b * sig['mb']
                if notional > cash / max_positions:
                    scale = cash / max_positions * 0.95 / notional
                    lots_a = max(1, int(lots_a * scale))
                    lots_b = max(1, int(lots_b * scale))
                    notional = lots_a * ep_a * sig['ma'] + lots_b * ep_b * sig['mb']

                if lots_a > 0 and lots_b > 0 and notional <= cash:
                    cash -= notional
                    positions.append({
                        'pair_idx': pi,
                        'dir': direction,
                        'entry_a': ep_a,
                        'entry_b': ep_b,
                        'lots_a': lots_a,
                        'lots_b': lots_b,
                        'entry_day': di,
                        'capital': notional,
                    })

    # close remaining
    for pos in positions:
        sig = pair_signals[pos['pair_idx']]
        ca = sig['c_a'][ND - 1]
        cb = sig['c_b'][ND - 1]
        if not np.isnan(ca) and ca > 0 and not np.isnan(cb) and cb > 0:
            realized_a = pos['dir'] * (ca - pos['entry_a']) * sig['ma'] * pos['lots_a']
            realized_b = -pos['dir'] * (cb - pos['entry_b']) * sig['mb'] * pos['lots_b']
            gross = realized_a + realized_b
            comm_cost = (pos['lots_a'] * ca * sig['ma'] +
                         pos['lots_b'] * cb * sig['mb']) * COMM
            net_pnl = gross - comm_cost
            cash += pos['capital'] + net_pnl
            trades.append({
                'pnl_pct': net_pnl / pos['capital'] * 100 if pos['capital'] > 0 else 0,
                'pnl_abs': net_pnl,
                'days': ND - 1 - pos['entry_day'],
                'di': ND - 1,
                'reason': 'end',
                'dir': pos['dir'],
                'year': dates[ND - 1].year,
                'pair': sig['label'],
            })

    if not trades or cash <= 0:
        return None

    days_total = (dates[ND - 1] - dates[MIN_TRAIN]).days
    yr = max(days_total / 365.25, 0.01)
    ann_ret = ((cash / CASH0) ** (1 / yr) - 1) * 100

    n_trades = len(trades)
    nw = sum(1 for t in trades if t['pnl_pct'] > 0)
    wr = nw / max(n_trades, 1) * 100
    avg_w = np.mean([t['pnl_pct'] for t in trades if t['pnl_pct'] > 0]) if nw > 0 else 0
    nl = n_trades - nw
    avg_l = np.mean([abs(t['pnl_pct']) for t in trades if t['pnl_pct'] <= 0]) if nl > 0 else 0
    edge = (nw / max(n_trades, 1)) * avg_w - (nl / max(n_trades, 1)) * avg_l
    tpy = n_trades / yr

    peak = CASH0
    max_dd = 0.0
    eq = CASH0
    for t in sorted(trades, key=lambda x: x['di']):
        eq += t['pnl_abs']
        if eq > peak:
            peak = eq
        if peak > 0:
            dd = (peak - eq) / peak * 100
            if dd > max_dd:
                max_dd = dd

    return {
        'name': f"PORTFOLIO|{variant}|h{hold_max}|sl{stop_loss}|max{max_positions}",
        'pair': 'PORTFOLIO',
        'variant': variant,
        'hold': hold_max,
        'sl': stop_loss,
        'ann': round(ann_ret, 1),
        'n': n_trades,
        'wr': round(wr, 1),
        'avg_w': round(avg_w, 2),
        'avg_l': round(avg_l, 2),
        'edge': round(edge, 2),
        'max_dd': round(max_dd, 1),
        'tpy': round(tpy, 1),
        'final': round(cash, 0),
        'trades': trades,
        'dates': dates,
        'yr': yr,
    }


# ── yearly breakdown ───────────────────────────────────────────
def yearly_breakdown(result):
    """Print yearly PnL breakdown for a result."""
    trades = result['trades']
    dates = result['dates']
    if not trades:
        return

    by_year = {}
    for t in trades:
        yr = t.get('year', dates[t['di']].year)
        by_year.setdefault(yr, []).append(t)

    print(f"\n  Yearly breakdown: {result['name']}", flush=True)
    print(f"  {'Year':>6s}  {'N':>4s}  {'WR':>6s}  {'AvgPnL':>8s}  "
          f"{'TotPnL':>12s}", flush=True)
    print(f"  {'-'*50}", flush=True)
    for yr in sorted(by_year):
        yt = by_year[yr]
        n = len(yt)
        nw = sum(1 for t in yt if t['pnl_pct'] > 0)
        wr = nw / max(n, 1) * 100
        avg = np.mean([t['pnl_pct'] for t in yt])
        tot = sum(t['pnl_pct'] for t in yt)
        print(f"  {yr:>6d}  {n:>4d}  {wr:>5.1f}%  {avg:>+7.2f}%  "
              f"{tot:>+11.2f}%", flush=True)


# ── main ───────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 80, flush=True)
    print("  Alpha Futures V17 — Cross-Commodity Pair Trading", flush=True)
    print("=" * 80, flush=True)

    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set, avail = load_pair_data()

    variants = ['ZSCORE_2', 'ZSCORE_15', 'ZSCORE_25', 'MOM_RATIO', 'OI_WEIGHTED']
    hold_maxes = [5, 10, 15]
    stop_losses = [0.03, 0.05]

    # ═══════════════════════════════════════════════════════════
    # PHASE 1: Individual pair tests
    # ═══════════════════════════════════════════════════════════
    print("\n[Phase 1] Testing individual pairs ...", flush=True)
    results = []
    total_combos = len(PAIRS) * len(variants) * len(hold_maxes) * len(stop_losses)
    done = 0
    t_start = time.time()

    for sym_a, sym_b, pair_label in PAIRS:
        if sym_a not in avail or sym_b not in avail:
            print(f"  SKIP {pair_label}: missing data", flush=True)
            continue
        idx_a = avail[sym_a]
        idx_b = avail[sym_b]

        for variant in variants:
            for hm in hold_maxes:
                for sl in stop_losses:
                    r = backtest_pair(
                        sym_a, sym_b, pair_label,
                        idx_a, idx_b,
                        NS, ND, dates, C, O, H, L, V, OI,
                        variant=variant,
                        hold_max=hm,
                        stop_loss=sl,
                    )
                    if r:
                        results.append(r)
                    done += 1
                    if done % 40 == 0:
                        elapsed = time.time() - t_start
                        print(f"  ... {done}/{total_combos} combos "
                              f"({elapsed:.1f}s)", flush=True)

    print(f"\n  Phase 1 done: {done} combos tested in "
          f"{time.time()-t_start:.1f}s", flush=True)

    results.sort(key=lambda x: -x['ann'])

    # ── TOP 20 table ──
    print(f"\n{'='*110}", flush=True)
    print(f"  TOP 20 INDIVIDUAL PAIR RESULTS", flush=True)
    print(f"  {'Pair':<20s} {'Variant':<12s} {'H':>2s} {'SL':>4s} │ "
          f"{'Ann':>8s} {'N':>4s} {'WR':>6s} {'AvgW':>7s} {'AvgL':>7s} "
          f"{'Edge':>7s} {'DD':>6s} {'TPY':>5s}", flush=True)
    print(f"  {'-'*105}", flush=True)
    for r in results[:20]:
        print(f"  {r['pair']:<20s} {r['variant']:<12s} {r['hold']:>2d} "
              f"{r['sl']:>4.2f} │ {r['ann']:>+7.1f}% {r['n']:>4d} "
              f"{r['wr']:>5.1f}% {r['avg_w']:>+6.2f}% {r['avg_l']:>6.2f}% "
              f"{r['edge']:>+6.2f}% {r['max_dd']:>5.1f}% {r['tpy']:>5.1f}",
              flush=True)

    # ── best per pair ──
    print(f"\n  Best variant per pair:", flush=True)
    best_per_pair = {}
    for r in results:
        p = r['pair']
        if p not in best_per_pair or r['ann'] > best_per_pair[p]['ann']:
            best_per_pair[p] = r
    for r in sorted(best_per_pair.values(), key=lambda x: -x['ann']):
        print(f"    {r['pair']:<20s} {r['variant']:<12s} h{r['hold']} sl{r['sl']:.2f}"
              f" → Ann {r['ann']:+.1f}%  WR {r['wr']:.1f}%  "
              f"Edge {r['edge']:+.2f}%  DD {r['max_dd']:.1f}%  N={r['n']}",
              flush=True)

    # ── best per variant ──
    print(f"\n  Best pair per variant:", flush=True)
    best_per_var = {}
    for r in results:
        v = r['variant']
        if v not in best_per_var or r['ann'] > best_per_var[v]['ann']:
            best_per_var[v] = r
    for r in sorted(best_per_var.values(), key=lambda x: -x['ann']):
        print(f"    {r['variant']:<12s} {r['pair']:<20s} h{r['hold']} sl{r['sl']:.2f}"
              f" → Ann {r['ann']:+.1f}%  WR {r['wr']:.1f}%  "
              f"Edge {r['edge']:+.2f}%  DD {r['max_dd']:.1f}%  N={r['n']}",
              flush=True)

    # ═══════════════════════════════════════════════════════════
    # PHASE 2: Portfolio mode (all pairs together)
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*80}", flush=True)
    print(f"  PHASE 2: PORTFOLIO MODE (all pairs, shared capital)", flush=True)
    print(f"{'='*80}", flush=True)

    portfolio_results = []
    for variant in variants:
        for hm in hold_maxes:
            for sl in stop_losses:
                for mp in [3, 5]:
                    r = backtest_portfolio(
                        NS, ND, dates, C, O, H, L, V, OI, avail,
                        variant=variant, hold_max=hm, stop_loss=sl,
                        max_positions=mp,
                    )
                    if r:
                        portfolio_results.append(r)

    portfolio_results.sort(key=lambda x: -x['ann'])

    if portfolio_results:
        print(f"\n  TOP 10 PORTFOLIO RESULTS:", flush=True)
        print(f"  {'Config':<50s} │ {'Ann':>8s} {'N':>4s} {'WR':>6s} "
              f"{'AvgW':>7s} {'AvgL':>7s} {'Edge':>7s} {'DD':>6s}", flush=True)
        print(f"  {'-'*100}", flush=True)
        for r in portfolio_results[:10]:
            print(f"  {r['name']:<50s} │ {r['ann']:>+7.1f}% {r['n']:>4d} "
                  f"{r['wr']:>5.1f}% {r['avg_w']:>+6.2f}% {r['avg_l']:>6.2f}% "
                  f"{r['edge']:>+6.2f}% {r['max_dd']:>5.1f}%", flush=True)

    # ═══════════════════════════════════════════════════════════
    # PHASE 3: Yearly breakdown for top 3
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*80}", flush=True)
    print(f"  YEARLY BREAKDOWN — TOP 3 (Individual Pairs)", flush=True)
    print(f"{'='*80}", flush=True)
    for r in results[:3]:
        yearly_breakdown(r)

    if portfolio_results:
        print(f"\n  YEARLY BREAKDOWN — TOP 3 (Portfolio)", flush=True)
        print(f"{'='*80}", flush=True)
        for r in portfolio_results[:3]:
            yearly_breakdown(r)

    # ═══════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*80}", flush=True)
    print(f"  SUMMARY", flush=True)
    print(f"{'='*80}", flush=True)
    if results:
        wrs = [r['wr'] for r in results]
        anns = [r['ann'] for r in results]
        print(f"  Individual pairs:", flush=True)
        print(f"    Configs tested      : {len(results)}", flush=True)
        print(f"    Ann return range     : {min(anns):+.1f}% to {max(anns):+.1f}%",
              flush=True)
        print(f"    WR range             : {min(wrs):.1f}% to {max(wrs):.1f}%",
              flush=True)
        above_60 = sum(1 for r in results if r['wr'] >= 60)
        above_50 = sum(1 for r in results if r['wr'] >= 50)
        print(f"    WR >= 60%            : {above_60}/{len(results)}", flush=True)
        print(f"    WR >= 50%            : {above_50}/{len(results)}", flush=True)
        avg_wr = np.mean(wrs)
        print(f"    Avg WR               : {avg_wr:.1f}% (directional baseline ~47-50%)",
              flush=True)

    if portfolio_results:
        p_wrs = [r['wr'] for r in portfolio_results]
        p_anns = [r['ann'] for r in portfolio_results]
        print(f"\n  Portfolio mode:", flush=True)
        print(f"    Configs tested      : {len(portfolio_results)}", flush=True)
        print(f"    Ann return range     : {min(p_anns):+.1f}% to {max(p_anns):+.1f}%",
              flush=True)
        print(f"    WR range             : {min(p_wrs):.1f}% to {max(p_wrs):.1f}%",
              flush=True)
        p_above_60 = sum(1 for r in portfolio_results if r['wr'] >= 60)
        p_above_50 = sum(1 for r in portfolio_results if r['wr'] >= 50)
        print(f"    WR >= 60%            : {p_above_60}/{len(portfolio_results)}",
              flush=True)
        print(f"    WR >= 50%            : {p_above_50}/{len(portfolio_results)}",
              flush=True)
        avg_pwr = np.mean(p_wrs)
        print(f"    Avg WR               : {avg_pwr:.1f}%", flush=True)

    print(f"\n{'='*80}", flush=True)
    print(f"  Done! Total time: {time.time()-t_start:.1f}s", flush=True)
