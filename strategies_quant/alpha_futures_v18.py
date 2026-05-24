"""
Alpha Futures V18 — Term Structure Carry + Momentum
====================================================
Core: Use backwardation/contango as carry signal, combine with price momentum.
Academic research shows carry is one of the strongest commodity factors.

Data: Daily OHLCV+OI from alpha_v2 + term structure JSON files
Period: 2021-01 ~ 2026-05 (term structure data availability)

6 strategy variants with parameter sweep:
  PURE_CARRY / CARRY_MOM5 / CARRY_MOM10 / CARRY_OI / CARRY_ALL / CARRY_MOM_ROTATION
"""
import sys, os, time, warnings, json, glob
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alpha_v2 import load_all_data, MIN_TRAIN, CASH0

# ============================================================
# CONSTANTS
# ============================================================
MULT = {
    'agfi': 15, 'alfi': 5, 'aufi': 1000, 'bufi': 10, 'cufi': 5,
    'fufi': 10, 'rbfi': 10, 'znfi': 5, 'nifi': 1, 'hcfi': 10,
    'spfi': 10, 'ssfi': 5, 'sffi': 5, 'smfi': 5, 'pbfi': 5,
    'snfi': 1, 'rufi': 10, 'wrffi': 10,
    'afi': 10, 'bfi': 10, 'bbfi': 500, 'cffi': 5, 'cfi': 10,
    'csfi': 10, 'ebfi': 5, 'egfi': 10, 'fbfi': 500,
    'ifi': 100, 'jfi': 100, 'jmfi': 60, 'lfi': 5, 'mfi': 10,
    'pgfi': 20, 'ppfi': 5, 'vfi': 5, 'yfi': 10, 'pfi': 10,
    'jdfi': 5, 'lhfi': 16, 'pkfi': 5, 'rrfi': 20, 'lrfi': 20,
    'jrfi': 20, 'pmfi': 20, 'whfi': 20, 'rsfi': 20, 'cjfi': 10,
    'mafi': 10, 'apfi': 10, 'cyfi': 5, 'fgfi': 20, 'oifi': 10,
    'pfifi': 5, 'rmfi': 10, 'srfi': 10, 'tafi': 5, 'safi': 20,
    'urfi': 20, 'scfi': 1000, 'lufi': 10, 'bcfi': 5, 'nrfi': 1,
    'lgfi': 20, 'brfi': 5, 'lcfi': 1, 'sifi': 5,
    'ni': 1, 'tai': 5,
}
DEF_MULT = 10
COMM = 0.0003
TERM_DIR = '/Users/chengming/home/futures_platform/data/futures_term_structure'


# ============================================================
# LOAD TERM STRUCTURE DATA
# ============================================================
def load_term_structure(syms):
    """Load all term structure JSON files, return dict keyed by (symbol, date_str).
    Only loads files for symbols that exist in our price data.
    """
    sym_set = set(syms)
    records = {}
    pattern = os.path.join(TERM_DIR, '*.json')
    files = sorted(glob.glob(pattern))
    print(f"  [TermStruct] Loading {len(files)} files...", flush=True)

    loaded = 0
    skipped = 0
    for fp in files:
        fname = os.path.basename(fp)
        # symbol is everything before the last _YYYYMMDD
        parts = fname.replace('.json', '').rsplit('_', 1)
        if len(parts) != 2:
            skipped += 1
            continue
        sym, date_str = parts
        if sym not in sym_set:
            skipped += 1
            continue
        try:
            with open(fp, 'r') as f:
                data = json.load(f)
            records[(sym, date_str)] = {
                'structure': data.get('structure', 'unknown'),
                'spread_pct': data.get('total_spread_pct', 0.0),
                'near_price': data.get('near_price', np.nan),
                'far_price': data.get('far_price', np.nan),
            }
            loaded += 1
        except Exception:
            skipped += 1

    print(f"  [TermStruct] Loaded {loaded} records, skipped {skipped}", flush=True)
    return records


def build_carry_arrays(NS, ND, dates, syms, term_records):
    """Build carry signal arrays aligned to the price data dates.

    Returns per-symbol per-day arrays:
      spread_pct[si, di]  — total_spread_pct from term structure
      structure[si, di]   — 1=backwardation, -1=contango, 0=unknown
      carry_5d[si, di]    — 5-day change in spread_pct (carry momentum)
    """
    spread_pct = np.zeros((NS, ND), dtype=np.float64)
    structure_arr = np.zeros((NS, ND), dtype=np.float64)

    # Build date string -> date index mapping
    date_str_map = {}
    for di, d in enumerate(dates):
        date_str_map[d.strftime('%Y%m%d')] = di

    # Fill arrays
    for si, sym in enumerate(syms):
        for dstr, di in date_str_map.items():
            key = (sym, dstr)
            if key in term_records:
                rec = term_records[key]
                spread_pct[si, di] = rec['spread_pct']
                if rec['structure'] == 'backwardation':
                    structure_arr[si, di] = 1.0
                elif rec['structure'] == 'contango':
                    structure_arr[si, di] = -1.0

    # Carry momentum: 5-day change in spread_pct
    carry_5d = np.full((NS, ND), np.nan, dtype=np.float64)
    for si in range(NS):
        for di in range(5, ND):
            prev = spread_pct[si, di - 5]
            curr = spread_pct[si, di]
            if prev != 0:
                carry_5d[si, di] = curr - prev

    return spread_pct, structure_arr, carry_5d


# ============================================================
# FACTOR PRE-COMPUTATION
# ============================================================
def compute_factors(NS, ND, C, H, L, V, OI, spread_pct, structure_arr, carry_5d):
    """Pre-compute all factors needed by the strategy variants."""
    print("  Computing factors...", flush=True)
    t0 = time.time()

    # Price momentum: 5-day and 10-day returns
    mom5 = np.full((NS, ND), np.nan, dtype=np.float64)
    mom10 = np.full((NS, ND), np.nan, dtype=np.float64)
    for si in range(NS):
        for di in range(10, ND):
            c0 = C[si, di]
            c5 = C[si, di - 5]
            c10 = C[si, di - 10]
            if not np.isnan(c0) and not np.isnan(c5) and c5 > 0:
                mom5[si, di] = (c0 - c5) / c5
            if not np.isnan(c0) and not np.isnan(c10) and c10 > 0:
                mom10[si, di] = (c0 - c10) / c10

    # OI momentum: 5-day OI change
    oi_mom5 = np.full((NS, ND), np.nan, dtype=np.float64)
    for si in range(NS):
        for di in range(5, ND):
            oi_now = OI[si, di]
            oi_prev = OI[si, di - 5]
            if (not np.isnan(oi_now) and not np.isnan(oi_prev)
                    and oi_prev > 0):
                oi_mom5[si, di] = (oi_now - oi_prev) / oi_prev

    # Volume ratio: current volume / 20-day average
    vol_ratio = np.full((NS, ND), np.nan, dtype=np.float64)
    for si in range(NS):
        for di in range(20, ND):
            v_now = V[si, di]
            if np.isnan(v_now) or v_now <= 0:
                continue
            v20 = V[si, di - 20:di]
            vv = v20[~np.isnan(v20)]
            if len(vv) >= 10:
                vol_ratio[si, di] = v_now / np.mean(vv)

    # ATR(10) for trailing stops
    atr10 = np.full((NS, ND), np.nan, dtype=np.float64)
    for si in range(NS):
        for di in range(11, ND):
            trs = []
            for dd in range(di - 10, di):
                hi = H[si, dd]
                lo = L[si, dd]
                pc = C[si, dd - 1]
                if np.isnan(hi) or np.isnan(lo):
                    continue
                tr = hi - lo
                if not np.isnan(pc):
                    tr = max(tr, abs(hi - pc), abs(lo - pc))
                trs.append(tr)
            if trs:
                atr10[si, di] = np.mean(trs)

    # VDP: Volume-Derived Price (from alpha_v2 research)
    # Simplified: cumulative volume-weighted price direction over 10 days
    vdp = np.full((NS, ND), np.nan, dtype=np.float64)
    for si in range(NS):
        for di in range(10, ND):
            seg_c = C[si, di-10:di]
            seg_v = V[si, di-10:di]
            valid = ~(np.isnan(seg_c) | np.isnan(seg_v))
            vc = seg_c[valid]
            vv = seg_v[valid]
            if len(vc) >= 5 and np.sum(vv) > 0:
                vdp[si, di] = np.sum(vc * vv) / np.sum(vv)

    print(f"  Factors done ({time.time()-t0:.1f}s)", flush=True)
    return mom5, mom10, oi_mom5, vol_ratio, atr10, vdp


# ============================================================
# SIGNAL GENERATION — 6 STRATEGY VARIANTS
# ============================================================
def generate_signals(variant, NS, ND, C, O, H, L, V, OI,
                     spread_pct, structure_arr, carry_5d,
                     mom5, mom10, oi_mom5, vol_ratio, atr10, vdp):
    """Generate buy/sell/short/cover signal sets for a given strategy variant.

    Returns: buy_d, sell_d, short_d, cover_d — dicts of {si: set(di)}
    """
    buy_d = {si: set() for si in range(NS)}
    sell_d = {si: set() for si in range(NS)}
    short_d = {si: set() for si in range(NS)}
    cover_d = {si: set() for si in range(NS)}

    SPREAD_THRESH = 0.1  # minimum absolute spread to qualify

    for si in range(NS):
        for di in range(20, ND):
            c = C[si, di]
            if np.isnan(c) or c <= 0:
                continue

            spread = spread_pct[si, di]
            struct = structure_arr[si, di]
            c5 = carry_5d[si, di]
            m5 = mom5[si, di]
            m10 = mom10[si, di]
            om5 = oi_mom5[si, di]
            vr = vol_ratio[si, di]

            if variant == 'PURE_CARRY':
                # Pure carry: backwardated = long, contango = short
                # No momentum filter
                if spread < -SPREAD_THRESH:
                    buy_d[si].add(di)
                if spread > SPREAD_THRESH:
                    short_d[si].add(di)

            elif variant == 'CARRY_MOM5':
                # Carry + 5-day momentum alignment
                if spread < -SPREAD_THRESH and not np.isnan(m5) and m5 > 0:
                    buy_d[si].add(di)
                if spread > SPREAD_THRESH and not np.isnan(m5) and m5 < 0:
                    short_d[si].add(di)

            elif variant == 'CARRY_MOM10':
                # Carry + 10-day momentum alignment
                if spread < -SPREAD_THRESH and not np.isnan(m10) and m10 > 0:
                    buy_d[si].add(di)
                if spread > SPREAD_THRESH and not np.isnan(m10) and m10 < 0:
                    short_d[si].add(di)

            elif variant == 'CARRY_OI':
                # Carry + OI increasing confirmation
                oi_ok = not np.isnan(om5) and om5 > 0
                if spread < -SPREAD_THRESH and oi_ok:
                    buy_d[si].add(di)
                if spread > SPREAD_THRESH and oi_ok:
                    short_d[si].add(di)

            elif variant == 'CARRY_ALL':
                # Carry + momentum + OI + VDP confirmation
                mom_ok = not np.isnan(m5) and m5 > 0
                oi_ok = not np.isnan(om5) and om5 > 0
                # VDP direction confirmation
                vdp_ok = bool(not np.isnan(vdp) and vdp > 0)
                score_long = sum([bool(mom_ok), bool(oi_ok), bool(vdp_ok)])
                mom_ok_s = not np.isnan(m5) and m5 < 0
                oi_ok_s = not np.isnan(om5) and om5 < 0
                vdp_ok_s = bool(not np.isnan(vdp) and vdp < 0)
                score_short = sum([bool(mom_ok_s), bool(oi_ok_s), bool(vdp_ok_s)])

                if spread < -SPREAD_THRESH and score_long >= 2:
                    buy_d[si].add(di)
                if spread > SPREAD_THRESH and score_short >= 2:
                    short_d[si].add(di)

            elif variant == 'CARRY_MOM_ROTATION':
                # This variant is handled separately — signals per day for top N
                # We mark ALL eligible commodities here; the backtester picks the best
                if spread < -SPREAD_THRESH and not np.isnan(m5) and m5 > 0:
                    buy_d[si].add(di)
                if spread > SPREAD_THRESH and not np.isnan(m5) and m5 < 0:
                    short_d[si].add(di)

            # Exit signals: carry reversal or spread crosses zero
            # For hold-based exits, we rely on hold_max in backtest
            # Signal-based exits: structure flip
            if struct == 1.0 and spread > 0:
                # Was backwardation, now contango -> exit long
                sell_d[si].add(di)
            if struct == -1.0 and spread < 0:
                # Was contango, now backwardation -> exit short
                cover_d[si].add(di)

    return buy_d, sell_d, short_d, cover_d


# ============================================================
# BACKTEST ENGINE (shared)
# ============================================================
def backtest(buy_d, sell_d, short_d, cover_d,
             NS, ND, dates, C, O, H, L, V, OI, syms,
             max_positions=1, sl_pct=0.05, hold_max=7,
             trail_mult=2.0, start_di=0,
             ranking='carry_strength', variant='PURE_CARRY',
             spread_pct=None, mom5=None):
    """Run backtest with trailing stop and hold limit.

    Args:
        start_di: index to start backtest from (e.g., where 2021 data begins)
        ranking: how to rank candidates when multiple signals fire
        spread_pct: carry signal array for ranking
        mom5: 5d momentum for ranking
    """
    cash = float(CASH0)
    positions = []
    trades = []
    year_stats = {}

    for di in range(start_di, ND):
        year = dates[di].year

        # --- Exit logic ---
        for pos in list(positions):
            si = pos['si']
            c = C[si, di]
            if np.isnan(c):
                continue
            mult = MULT.get(pos['sym'], DEF_MULT)
            if pos['dir'] == 1:
                pnl = (c - pos['entry']) * mult * pos['lots']
            else:
                pnl = (pos['entry'] - c) * mult * pos['lots']
            pnl_pct = pnl / (pos['entry'] * mult * pos['lots']) * 100

            # Trailing stop exit
            if trail_mult > 0 and pos.get('trail_stop') and not np.isnan(pos['trail_stop']):
                stopped = False
                if pos['dir'] == 1 and c < pos['trail_stop']:
                    stopped = True
                elif pos['dir'] == -1 and c > pos['trail_stop']:
                    stopped = True
                if stopped:
                    cash += c * mult * pos['lots'] * (1 - COMM)
                    trades.append({
                        'pnl_pct': pnl_pct, 'pnl_abs': pnl,
                        'days': di - pos['entry_di'], 'di': di,
                        'reason': 'trail', 'year': year, 'si': si,
                        'dir': pos['dir'], 'sym': pos['sym'],
                    })
                    positions.remove(pos)
                    continue

                # Update trailing stop
                atr = pos.get('atr', 0)
                if atr > 0 and not np.isnan(atr):
                    if pos['dir'] == 1:
                        new_stop = c - trail_mult * atr
                        if pos['trail_stop'] is np.nan or new_stop > pos['trail_stop']:
                            pos['trail_stop'] = new_stop
                    else:
                        new_stop = c + trail_mult * atr
                        if pos['trail_stop'] is np.nan or new_stop < pos['trail_stop']:
                            pos['trail_stop'] = new_stop

            # Fixed stop-loss
            if pnl_pct / 100 < -sl_pct:
                cash += c * mult * pos['lots'] * (1 - COMM)
                trades.append({
                    'pnl_pct': pnl_pct, 'pnl_abs': pnl,
                    'days': di - pos['entry_di'], 'di': di,
                    'reason': 'stop', 'year': year, 'si': si,
                    'dir': pos['dir'], 'sym': pos['sym'],
                })
                positions.remove(pos)
                continue

            # Signal-based exit
            if pos['dir'] == 1 and di in sell_d.get(si, set()):
                cash += c * mult * pos['lots'] * (1 - COMM)
                trades.append({
                    'pnl_pct': pnl_pct, 'pnl_abs': pnl,
                    'days': di - pos['entry_di'], 'di': di,
                    'reason': 'signal', 'year': year, 'si': si,
                    'dir': pos['dir'], 'sym': pos['sym'],
                })
                positions.remove(pos)
                continue
            if pos['dir'] == -1 and di in cover_d.get(si, set()):
                cash += c * mult * pos['lots'] * (1 - COMM)
                trades.append({
                    'pnl_pct': pnl_pct, 'pnl_abs': pnl,
                    'days': di - pos['entry_di'], 'di': di,
                    'reason': 'signal', 'year': year, 'si': si,
                    'dir': pos['dir'], 'sym': pos['sym'],
                })
                positions.remove(pos)
                continue

            # Hold timeout
            if di - pos['entry_di'] >= hold_max:
                cash += c * mult * pos['lots'] * (1 - COMM)
                trades.append({
                    'pnl_pct': pnl_pct, 'pnl_abs': pnl,
                    'days': di - pos['entry_di'], 'di': di,
                    'reason': 'time', 'year': year, 'si': si,
                    'dir': pos['dir'], 'sym': pos['sym'],
                })
                positions.remove(pos)

        # --- Entry logic ---
        if len(positions) < max_positions:
            candidates = []

            for si in range(NS):
                if any(p['si'] == si for p in positions):
                    continue
                c = C[si, di]
                if np.isnan(c) or c <= 0:
                    continue

                if di in buy_d.get(si, set()):
                    candidates.append((si, c, 1, syms[si]))
                if di in short_d.get(si, set()):
                    candidates.append((si, c, -1, syms[si]))

            if candidates:
                # Rank candidates
                def _rank(x):
                    si, _, d, _ = x
                    if ranking == 'carry_strength':
                        # Stronger spread = better carry
                        sp = spread_pct[si, di] if spread_pct is not None else 0
                        return abs(sp) * d
                    elif ranking == 'mom5':
                        m = mom5[si, di] if mom5 is not None else np.nan
                        if np.isnan(m):
                            return 0
                        return m * d
                    elif ranking == 'carry_mom':
                        sp = spread_pct[si, di] if spread_pct is not None else 0
                        m = mom5[si, di] if mom5 is not None else np.nan
                        if np.isnan(m):
                            return abs(sp) * d
                        return (abs(sp) + abs(m)) * d
                    return 0

                candidates.sort(key=_rank, reverse=True)

                slots = max_positions - len(positions)
                for si, price, direction, sym in candidates[:slots]:
                    mult = MULT.get(sym, DEF_MULT)
                    notional_per_lot = price * mult
                    if notional_per_lot <= 0:
                        continue
                    alloc = cash / max(1, max_positions - len(positions))
                    lots = int(alloc / notional_per_lot)
                    if lots > 0:
                        cost = notional_per_lot * lots * (1 + COMM)
                        if cost <= cash:
                            cash -= cost
                            # Compute ATR at entry for trailing stop
                            atr = 0.0
                            if di >= 11:
                                trs = []
                                for dd in range(max(1, di - 10), di):
                                    hi = H[si, dd]
                                    lo = L[si, dd]
                                    pc = C[si, dd - 1]
                                    if np.isnan(hi) or np.isnan(lo):
                                        continue
                                    tr = hi - lo
                                    if not np.isnan(pc):
                                        tr = max(tr, abs(hi - pc), abs(lo - pc))
                                    trs.append(tr)
                                if trs:
                                    atr = np.mean(trs)

                            trail_stop = np.nan
                            if atr > 0 and trail_mult > 0:
                                if direction == 1:
                                    trail_stop = price - trail_mult * atr
                                else:
                                    trail_stop = price + trail_mult * atr

                            positions.append({
                                'si': si, 'entry': price, 'entry_di': di,
                                'lots': lots, 'dir': direction, 'sym': sym,
                                'atr': atr, 'trail_stop': trail_stop,
                            })

    # --- Close remaining positions ---
    for pos in positions:
        c = C[pos['si'], ND - 1]
        if np.isnan(c) or c <= 0:
            c = pos['entry']
        mult = MULT.get(pos['sym'], DEF_MULT)
        pnl = (c - pos['entry']) * mult * pos['lots'] * pos['dir']
        pnl_pct = pnl / (pos['entry'] * mult * pos['lots']) * 100
        cash += c * mult * pos['lots'] * (1 - COMM)
        trades.append({
            'pnl_pct': pnl_pct, 'pnl_abs': pnl,
            'days': 999, 'di': ND - 1, 'reason': 'end',
            'year': dates[ND - 1].year, 'si': pos['si'],
            'dir': pos['dir'], 'sym': pos['sym'],
        })

    if not trades:
        return None

    # --- Compute statistics ---
    equity = float(CASH0)
    peak = float(CASH0)
    max_dd = 0.0
    total_pnl = 0.0
    for t in sorted(trades, key=lambda x: x['di']):
        equity += t['pnl_abs']
        total_pnl += t['pnl_abs']
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak * 100 if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd

    final_cash = cash
    if final_cash <= 0:
        return None

    days_total = (dates[ND - 1] - dates[start_di]).days
    yr = max(days_total / 365.25, 0.01)
    ann = ((final_cash / CASH0) ** (1 / yr) - 1) * 100

    nw = sum(1 for t in trades if t['pnl_abs'] > 0)
    wr = nw / max(len(trades), 1) * 100
    avg_win = np.mean([t['pnl_pct'] for t in trades if t['pnl_abs'] > 0]) if nw > 0 else 0
    avg_loss = np.mean([abs(t['pnl_pct']) for t in trades if t['pnl_abs'] <= 0]) if nw < len(trades) else 0

    for t in trades:
        y = t.get('year', 'unknown')
        if y not in year_stats:
            year_stats[y] = {'trades': 0, 'wins': 0, 'total_pnl': 0, 'pnl_abs_sum': 0}
        year_stats[y]['trades'] += 1
        if t['pnl_abs'] > 0:
            year_stats[y]['wins'] += 1
        year_stats[y]['total_pnl'] += t['pnl_pct']
        year_stats[y]['pnl_abs_sum'] += t['pnl_abs']

    return {
        'ann': round(ann, 1), 'n': len(trades), 'wr': round(wr, 1),
        'max_dd': round(max_dd, 1), 'final': round(final_cash, 0),
        'avg_win': round(avg_win, 1), 'avg_loss': round(avg_loss, 1),
        'year_stats': year_stats, 'total_pnl': round(total_pnl, 0),
    }


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    print("=" * 90, flush=True)
    print("  Alpha Futures V18 — Term Structure Carry + Momentum", flush=True)
    print("=" * 90, flush=True)

    # Load price data
    print("\n[1] Loading price data (OHLCV + OI)...", flush=True)
    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)
    print(f"  {NS} symbols, {ND} trading days", flush=True)

    # Load term structure data
    print("\n[2] Loading term structure data...", flush=True)
    term_records = load_term_structure(syms)

    # Build carry arrays
    print("\n[3] Building carry signal arrays...", flush=True)
    spread_pct, structure_arr, carry_5d = build_carry_arrays(
        NS, ND, dates, syms, term_records)

    # Find where 2021 starts in the dates array
    start_di_2021 = 0
    for i, d in enumerate(dates):
        if d.year >= 2021:
            start_di_2021 = i
            break
    # Ensure at least 20 days warmup for factors
    start_di = max(start_di_2021, MIN_TRAIN)
    start_date = dates[start_di]
    print(f"  Backtest starts at di={start_di} ({start_date.strftime('%Y-%m-%d')})", flush=True)

    # Count how many symbols have term structure data
    sym_with_ts = set()
    for si, sym in enumerate(syms):
        for di in range(start_di, ND):
            if spread_pct[si, di] != 0:
                sym_with_ts.add(sym)
                break
    print(f"  {len(sym_with_ts)} symbols have term structure data", flush=True)

    # Compute factors
    print("\n[4] Computing factors...", flush=True)
    t0_all = time.time()
    mom5, mom10, oi_mom5, vol_ratio, atr10, vdp = compute_factors(
        NS, ND, C, H, L, V, OI, spread_pct, structure_arr, carry_5d)

    # ========================================================
    # STRATEGY VARIANTS
    # ========================================================
    variants = [
        'PURE_CARRY',
        'CARRY_MOM5',
        'CARRY_MOM10',
        'CARRY_OI',
        'CARRY_ALL',
        'CARRY_MOM_ROTATION',
    ]

    # Parameter sweep
    hold_vals = [3, 5, 7]
    trail_vals = [1.5, 2.0, 3.0]
    stop_vals = [0.03, 0.05]
    max_pos_vals = [1, 2]
    ranking_options = {
        'PURE_CARRY': ['carry_strength'],
        'CARRY_MOM5': ['carry_mom', 'carry_strength'],
        'CARRY_MOM10': ['carry_mom', 'carry_strength'],
        'CARRY_OI': ['carry_strength'],
        'CARRY_ALL': ['carry_mom', 'carry_strength'],
        'CARRY_MOM_ROTATION': ['carry_mom'],
    }

    all_results = []

    for variant in variants:
        print(f"\n  [{variant}] Generating signals...", flush=True)
        t0_var = time.time()

        buy_d, sell_d, short_d, cover_d = generate_signals(
            variant, NS, ND, C, O, H, L, V, OI,
            spread_pct, structure_arr, carry_5d,
            mom5, mom10, oi_mom5, vol_ratio, atr10, vdp)

        # Count signals
        n_buy = sum(len(v) for v in buy_d.values())
        n_short = sum(len(v) for v in short_d.values())
        print(f"    Buy signals: {n_buy}, Short signals: {n_short}", flush=True)

        rankings = ranking_options.get(variant, ['carry_strength'])
        n_configs = (len(hold_vals) * len(trail_vals) * len(stop_vals)
                     * len(max_pos_vals) * len(rankings))
        print(f"    Sweeping {n_configs} parameter combinations...", flush=True)

        for hold in hold_vals:
            for trail in trail_vals:
                for stop in stop_vals:
                    for mp in max_pos_vals:
                        for rank in rankings:
                            r = backtest(
                                buy_d, sell_d, short_d, cover_d,
                                NS, ND, dates, C, O, H, L, V, OI, syms,
                                max_positions=mp, sl_pct=stop, hold_max=hold,
                                trail_mult=trail, start_di=start_di,
                                ranking=rank, variant=variant,
                                spread_pct=spread_pct, mom5=mom5)
                            if r is not None:
                                r['variant'] = variant
                                r['hold'] = hold
                                r['trail'] = trail
                                r['stop'] = stop
                                r['mp'] = mp
                                r['rank'] = rank
                                all_results.append(r)

        elapsed = time.time() - t0_var
        n_res = sum(1 for r in all_results if r['variant'] == variant)
        print(f"    Done ({elapsed:.1f}s, {n_res} results)", flush=True)

    # ========================================================
    # RESULTS
    # ========================================================
    print(f"\n{'=' * 90}", flush=True)
    print(f"  TOTAL RESULTS: {len(all_results)}", flush=True)
    print(f"{'=' * 90}", flush=True)

    if not all_results:
        print("  No results found!", flush=True)
    else:
        # Filter to reasonable results
        all_results.sort(key=lambda x: -x['ann'])

        # Show TOP 20
        print(f"\n  TOP 20 (by annualized return):", flush=True)
        header = (f"  {'#':>3s} {'Variant':<22s} {'MP':>2s} {'Hold':>4s} "
                  f"{'Trail':>5s} {'Stop':>5s} {'Rank':<16s} | "
                  f"{'Ann':>8s} {'N':>5s} {'WR':>5s} {'AvgW':>6s} "
                  f"{'AvgL':>6s} {'DD':>6s} {'Final':>10s}")
        print(header, flush=True)
        print("  " + "-" * 108, flush=True)

        for i, r in enumerate(all_results[:20]):
            line = (f"  {i+1:>3d} {r['variant']:<22s} P{r['mp']:<1d} "
                    f"H{r['hold']:>2d}  "
                    f"T{r['trail']:>3.1f}  "
                    f"S{r['stop']:>4.2f} "
                    f"{r['rank']:<16s} | "
                    f"{r['ann']:>+7.1f}% "
                    f"{r['n']:5d} "
                    f"{r['wr']:>5.1f}% "
                    f"{r['avg_win']:>+5.1f}% "
                    f"{r['avg_loss']:>5.1f}% "
                    f"{r['max_dd']:>5.1f}% "
                    f"{r['final']:>10,.0f}")
            print(line, flush=True)

        # Yearly breakdown for top 3
        print(f"\n{'=' * 90}", flush=True)
        print(f"  YEARLY BREAKDOWN — TOP 3", flush=True)
        print(f"{'=' * 90}", flush=True)

        for i, r in enumerate(all_results[:3]):
            label = (f"#{i+1}: {r['variant']} P{r['mp']} H{r['hold']} "
                     f"T{r['trail']} S{r['stop']} {r['rank']}")
            print(f"\n  {label}", flush=True)
            print(f"  Ann={r['ann']:+.1f}%  DD={r['max_dd']:.1f}%  "
                  f"WR={r['wr']:.0f}%  N={r['n']}  Final={r['final']:,.0f}", flush=True)
            print(f"  {'Year':>6s} {'Trades':>7s} {'WR':>6s} {'PnL%':>8s} {'PnL$':>12s}", flush=True)
            print(f"  {'-'*42}", flush=True)
            for y in sorted(r.get('year_stats', {}).keys()):
                s = r['year_stats'][y]
                wr_y = s['wins'] / max(s['trades'], 1) * 100
                print(f"  {y:>6d} {s['trades']:>7d} {wr_y:>5.0f}% "
                      f"{s['total_pnl']:>+7.0f}% {s['pnl_abs_sum']:>+12,.0f}", flush=True)

        # Summary by variant
        print(f"\n{'=' * 90}", flush=True)
        print(f"  BEST PER VARIANT", flush=True)
        print(f"{'=' * 90}", flush=True)
        seen = set()
        for r in all_results:
            v = r['variant']
            if v in seen:
                continue
            seen.add(v)
            label = (f"  {v:<22s} | Ann={r['ann']:+7.1f}%  N={r['n']:5d}  "
                     f"WR={r['wr']:5.1f}%  DD={r['max_dd']:5.1f}%  "
                     f"H{r['hold']} T{r['trail']} S{r['stop']} P{r['mp']} {r['rank']}")
            print(label, flush=True)

    elapsed_total = time.time() - t0_all
    print(f"\n  Total compute time: {elapsed_total:.0f}s", flush=True)
    print(f"{'=' * 90}", flush=True)
