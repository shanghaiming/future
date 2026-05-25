"""
Alpha Futures V112 -- OI (Open Interest) + Term Structure DEEP DIVE
=====================================================================
Futures-unique signals that stock strategies don't have:
- OI momentum, surge, divergence, capitulation, volume/OI ratio, OI-weighted ROC
- Term structure: backwardation, spread compression, combos with OI

ALL signals computed at close of day di using data up to and including di.
Entry at O[si, di+1] (NEXT DAY OPEN).
Exit at C[si, di+hold] (close price hold days later).

10 OI signals (A-F with variants) + 4 term structure signals (G-J) = ~60 configs.
Walk-forward by year (2020-2025).
"""
import sys, os, time, warnings, json
import numpy as np
import talib
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alpha_v2 import load_all_data, MIN_TRAIN, CASH0

# ============================================================
# CONSTANTS
# ============================================================
MULT = {'agfi': 15, 'alfi': 5, 'aufi': 1000, 'bufi': 10, 'cufi': 5, 'fufi': 10,
        'rbfi': 10, 'znfi': 5, 'nifi': 1, 'hcfi': 10, 'spfi': 10, 'ssfi': 5,
        'sffi': 5, 'smfi': 5, 'pbfi': 5, 'snfi': 1, 'rufi': 10, 'wrffi': 10,
        'afi': 10, 'bfi': 10, 'bbfi': 500, 'cffi': 5, 'cfi': 10, 'csfi': 10,
        'ebfi': 5, 'egfi': 10, 'fbfi': 500, 'ifi': 100, 'jfi': 100, 'jmfi': 60,
        'lfi': 5, 'mfi': 10, 'pgfi': 20, 'ppfi': 5, 'vfi': 5, 'yfi': 10,
        'pfi': 10, 'jdfi': 5, 'lhfi': 16, 'pkfi': 5, 'rrfi': 20, 'lrfi': 20,
        'jrfi': 20, 'pmfi': 20, 'whfi': 20, 'rsfi': 20, 'cjfi': 10, 'mafi': 10,
        'apfi': 10, 'cyfi': 5, 'fgfi': 20, 'oifi': 10, 'pfifi': 5, 'rmfi': 10,
        'srfi': 10, 'tafi': 5, 'safi': 20, 'urfi': 20, 'scfi': 1000, 'lufi': 10,
        'bcfi': 5, 'nrfi': 1, 'lgfi': 20, 'brfi': 5, 'lcfi': 1, 'sifi': 5,
        'ni': 1, 'tai': 5}
DEF_MULT = 10
COMM = 0.0003

TS_DIR = '/Users/chengming/home/futures_platform/data/futures_term_structure'


def annual_return(final, initial, n_days):
    if final <= 0 or initial <= 0 or n_days <= 0:
        return -100.0
    yrs = n_days / 252
    return (final / initial) ** (1.0 / yrs) * 100 - 100


def load_term_structure_all(syms, dates):
    """Pre-load all term structure data into a dict indexed by (symbol, date_str).
    Returns: dict[(sym, date_str)] = {'structure': ..., 'near_price': ..., 'far_price': ..., 'total_spread': ..., 'total_spread_pct': ...}
    """
    print("  [TS] Pre-loading term structure data...", flush=True)
    t0 = time.time()
    ts_data = {}
    sym_set = set(syms)

    # Build date string index for fast lookup
    date_str_set = set()
    for d in dates:
        date_str_set.add(d.strftime('%Y%m%d'))

    # Scan files and load
    loaded = 0
    skipped = 0
    for fname in os.listdir(TS_DIR):
        if not fname.endswith('.json'):
            continue
        parts = fname.replace('.json', '').split('_')
        if len(parts) < 2:
            continue
        sym = parts[0]
        date_str = parts[1]
        if sym not in sym_set or date_str not in date_str_set:
            skipped += 1
            continue
        path = os.path.join(TS_DIR, fname)
        try:
            with open(path) as f:
                data = json.load(f)
            ts_data[(sym, date_str)] = {
                'structure': data.get('structure'),
                'near_price': data.get('near_price'),
                'far_price': data.get('far_price'),
                'total_spread': data.get('total_spread'),
                'total_spread_pct': data.get('total_spread_pct'),
            }
            loaded += 1
        except:
            pass

    print(f"  [TS] Loaded {loaded} term structure entries ({skipped} skipped) in {time.time()-t0:.1f}s")
    return ts_data


def main():
    print("=" * 200)
    print("Alpha Futures V112 -- OI + Term Structure DEEP DIVE")
    print("=" * 200)
    print("\n  Futures-unique signals: OI momentum, surge, divergence, capitulation, V/OI ratio, OI-weighted ROC")
    print("  Term structure: backwardation, spread compression, ROC combos, OI+TS+ROC triple")
    print("  ALL signals at close di, entry at O[si, di+1] (NEXT DAY OPEN)")

    # -- Load data -------------------------------------------------
    print("\n[Data] Loading...", flush=True)
    t_start = time.time()
    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)
    years = sorted(set(d.year for d in dates))
    print(f"  {NS} commodities, {ND} days, years: {years}")

    # Check OI availability
    oi_valid = np.sum(~np.isnan(OI))
    oi_total = OI.size
    print(f"  OI coverage: {oi_valid}/{oi_total} cells ({oi_valid/oi_total*100:.1f}%)")

    # -- Pre-load term structure -----------------------------------
    ts_data = load_term_structure_all(syms, dates)

    # ================================================================
    # PRECOMPUTE INDICATORS
    # ================================================================
    print("\n[Indicators] Computing TA-Lib and derived indicators...", flush=True)
    t0 = time.time()

    ROC5 = np.full((NS, ND), np.nan)
    OI_SMA20 = np.full((NS, ND), np.nan)
    OI_SMA5 = np.full((NS, ND), np.nan)
    VOL_SMA20 = np.full((NS, ND), np.nan)

    # Derived OI arrays
    OI_5d_change_pct = np.full((NS, ND), np.nan)  # (OI[di] - OI[di-5]) / OI[di-5]
    OI_10d_change_pct = np.full((NS, ND), np.nan)

    for si in range(NS):
        c = C[si].astype(np.float64)
        oi = OI[si].astype(np.float64)
        v = V[si].astype(np.float64)

        ROC5[si] = talib.ROC(c, timeperiod=5)
        OI_SMA20[si] = talib.SMA(oi, timeperiod=20)
        OI_SMA5[si] = talib.SMA(oi, timeperiod=5)
        VOL_SMA20[si] = talib.SMA(v, timeperiod=20)

        # OI 5-day change rate
        for di in range(5, ND):
            if not np.isnan(oi[di]) and not np.isnan(oi[di-5]) and oi[di-5] > 0:
                OI_5d_change_pct[si, di] = (oi[di] - oi[di-5]) / oi[di-5]

        # OI 10-day change rate
        for di in range(10, ND):
            if not np.isnan(oi[di]) and not np.isnan(oi[di-10]) and oi[di-10] > 0:
                OI_10d_change_pct[si, di] = (oi[di] - oi[di-10]) / oi[di-10]

        if (si + 1) % 10 == 0 or si == NS - 1:
            print(f"  ... {si+1}/{NS} commodities done ({time.time()-t0:.1f}s)", flush=True)

    print(f"  All indicators computed ({time.time()-t0:.1f}s)")

    # ================================================================
    # BUILD TERM STRUCTURE ARRAYS
    # ================================================================
    print("\n[TermStructure] Building arrays from pre-loaded data...", flush=True)
    t0 = time.time()

    # Arrays for term structure signals
    TS_STRUCT = np.full((NS, ND), 0, dtype=np.int8)  # 0=unknown, 1=contango, 2=backwardation
    TS_SPREAD_PCT = np.full((NS, ND), np.nan)
    TS_NEAR_PRICE = np.full((NS, ND), np.nan)
    TS_FAR_PRICE = np.full((NS, ND), np.nan)

    for si in range(NS):
        sym = syms[si]
        for di in range(ND):
            ds = dates[di].strftime('%Y%m%d')
            ts = ts_data.get((sym, ds))
            if ts is None:
                continue
            if ts['structure'] == 'contango':
                TS_STRUCT[si, di] = 1
            elif ts['structure'] == 'backwardation':
                TS_STRUCT[si, di] = 2
            if ts['total_spread_pct'] is not None:
                TS_SPREAD_PCT[si, di] = ts['total_spread_pct']
            if ts['near_price'] is not None:
                TS_NEAR_PRICE[si, di] = ts['near_price']
            if ts['far_price'] is not None:
                TS_FAR_PRICE[si, di] = ts['far_price']

    n_ts_cells = np.sum(TS_STRUCT > 0)
    n_backwardation = np.sum(TS_STRUCT == 2)
    n_contango = np.sum(TS_STRUCT == 1)
    print(f"  TS coverage: {n_ts_cells} cells ({n_contango} contango, {n_backwardation} backwardation)")
    print(f"  TS arrays built ({time.time()-t0:.1f}s)")

    # ================================================================
    # SIGNAL GENERATION
    # ================================================================
    print("\n[Signals] Computing all OI + Term Structure signals...", flush=True)

    # -- A) OI MOMENTUM --
    # OI increasing by > 3% AND ROC(5) > 0 (new money + positive price = strong trend)
    sig_A = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(25, ND):
            oi_chg = OI_5d_change_pct[si, di]
            roc = ROC5[si, di]
            if np.isnan(oi_chg) or np.isnan(roc):
                continue
            if oi_chg > 0.03 and roc > 0:
                sig_A[si, di] = True
    print(f"  A) OI_MOMENTUM (OI+3% & ROC5>0): {np.sum(sig_A)} signals")

    # -- B) OI SURGE --
    # OI > 1.5 * SMA(OI,20) AND C > O (bullish close)
    sig_B = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(25, ND):
            oi_v = OI[si, di]
            oi_sma = OI_SMA20[si, di]
            c_v = C[si, di]
            o_v = O[si, di]
            if np.isnan(oi_v) or np.isnan(oi_sma) or np.isnan(c_v) or np.isnan(o_v):
                continue
            if oi_sma > 0 and oi_v > 1.5 * oi_sma and c_v > o_v:
                sig_B[si, di] = True
    print(f"  B) OI_SURGE (OI>1.5xSMA20 & C>O): {np.sum(sig_B)} signals")

    # -- C) OI-PRICE DIVERGENCE (smart money) --
    # Price up + OI up = new longs entering (strong) -> BUY
    # ROC(10) > 0 AND OI[di] > OI[di-5]
    sig_C = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(25, ND):
            roc10 = OI_10d_change_pct[si, di]
            oi_chg5 = OI_5d_change_pct[si, di]
            c_v = C[si, di]
            c_v10 = C[si, di-10] if di >= 10 else np.nan
            if np.isnan(roc10) or np.isnan(oi_chg5) or np.isnan(c_v) or np.isnan(c_v10):
                continue
            # Price up over 10 days AND OI up over 5 days
            price_up = (c_v - c_v10) / c_v10 > 0 if c_v10 > 0 else False
            oi_up = oi_chg5 > 0
            if price_up and oi_up:
                sig_C[si, di] = True
    print(f"  C) OI_PRICE_DIV (Price+OI both up): {np.sum(sig_C)} signals")

    # -- D) OI CAPITULATION --
    # OI drops > 5% in 5 days AND price drops > 3% in 5 days -> reversal
    sig_D = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(25, ND):
            oi_chg = OI_5d_change_pct[si, di]
            roc = ROC5[si, di]
            if np.isnan(oi_chg) or np.isnan(roc):
                continue
            if oi_chg < -0.05 and roc < -3.0:
                sig_D[si, di] = True
    print(f"  D) OI_CAPITULATION (OI-5% & ROC5<-3%): {np.sum(sig_D)} signals")

    # -- E) VOLUME/OI RATIO --
    # V[di] / OI[di] > 2x 20-day average AND C > O
    sig_E = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(25, ND):
            v_v = V[si, di]
            oi_v = OI[si, di]
            c_v = C[si, di]
            o_v = O[si, di]
            if np.isnan(v_v) or np.isnan(oi_v) or np.isnan(c_v) or np.isnan(o_v):
                continue
            if oi_v <= 0:
                continue
            turnover = v_v / oi_v
            # Compute 20-day average turnover
            turnovers = []
            for dd in range(max(0, di-19), di+1):
                vv = V[si, dd]
                ov = OI[si, dd]
                if not np.isnan(vv) and not np.isnan(ov) and ov > 0:
                    turnovers.append(vv / ov)
            if len(turnovers) >= 10:
                avg_to = np.mean(turnovers)
                if avg_to > 0 and turnover > 2.0 * avg_to and c_v > o_v:
                    sig_E[si, di] = True
    print(f"  E) VOL_OI_RATIO (turnover>2x avg & C>O): {np.sum(sig_E)} signals")

    # -- F) OI-WEIGHTED ROC --
    # Score = ROC(5) * sign(OI_5d_change) * abs(OI_5d_change_pct)
    # High score = strong momentum confirmed by new money
    sig_F = np.zeros((NS, ND), dtype=bool)
    score_F = np.zeros((NS, ND), dtype=np.float64)
    for si in range(NS):
        for di in range(25, ND):
            roc = ROC5[si, di]
            oi_chg = OI_5d_change_pct[si, di]
            if np.isnan(roc) or np.isnan(oi_chg):
                continue
            score = roc * np.sign(oi_chg) * abs(oi_chg)
            score_F[si, di] = score
            # Signal if score is positive (momentum confirmed by OI)
            if score > 0:
                sig_F[si, di] = True
    print(f"  F) OI_WEIGHTED_ROC (score>0): {np.sum(sig_F)} signals")

    # -- G) BACKWARDATION SIGNAL --
    # structure[di] == backwardation AND structure[di-5] == contango (flip)
    sig_G = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(25, ND):
            curr = TS_STRUCT[si, di]
            prev5 = TS_STRUCT[si, di-5]
            if curr == 2 and prev5 == 1:
                sig_G[si, di] = True
    print(f"  G) BACKWARDATION_FLIP (contango->backwardation): {np.sum(sig_G)} signals")

    # -- H) SPREAD COMPRESSION --
    # Spread moving toward 0 AND near_price rising
    sig_H = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(25, ND):
            sp = TS_SPREAD_PCT[si, di]
            sp5 = TS_SPREAD_PCT[si, di-5]
            np_v = TS_NEAR_PRICE[si, di]
            np5 = TS_NEAR_PRICE[si, di-5]
            if np.isnan(sp) or np.isnan(sp5) or np.isnan(np_v) or np.isnan(np5):
                continue
            if np5 <= 0:
                continue
            # Spread narrowing (absolute value decreasing)
            spread_narrowing = abs(sp) < abs(sp5)
            # Near price rising
            near_rising = np_v > np5
            if spread_narrowing and near_rising:
                sig_H[si, di] = True
    print(f"  H) SPREAD_COMPRESSION (narrowing & near up): {np.sum(sig_H)} signals")

    # -- I) TERM STRUCTURE + ROC COMBO --
    # Backwardation AND ROC(5) > 0
    sig_I = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(25, ND):
            ts = TS_STRUCT[si, di]
            roc = ROC5[si, di]
            if np.isnan(roc):
                continue
            if ts == 2 and roc > 0:
                sig_I[si, di] = True
    print(f"  I) TS_BACKWARDATION+ROC5>0: {np.sum(sig_I)} signals")

    # -- J) OI + TERM STRUCTURE + ROC TRIPLE --
    # OI increasing AND backwardation AND ROC(5) > 0
    sig_J = np.zeros((NS, ND), dtype=bool)
    score_J = np.zeros((NS, ND), dtype=np.float64)
    for si in range(NS):
        for di in range(25, ND):
            oi_chg = OI_5d_change_pct[si, di]
            ts = TS_STRUCT[si, di]
            roc = ROC5[si, di]
            if np.isnan(oi_chg) or np.isnan(roc):
                continue
            if oi_chg > 0 and ts == 2 and roc > 0:
                sig_J[si, di] = True
                score_J[si, di] = roc * oi_chg  # momentum * OI change strength
    print(f"  J) OI+TS_BACKWARDATION+ROC5 (triple): {np.sum(sig_J)} signals")

    # -- K) OI MOMENTUM STRONGER (OI > 5% increase) --
    # Variant of A with stricter OI threshold
    sig_K = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(25, ND):
            oi_chg = OI_5d_change_pct[si, di]
            roc = ROC5[si, di]
            if np.isnan(oi_chg) or np.isnan(roc):
                continue
            if oi_chg > 0.05 and roc > 0:
                sig_K[si, di] = True
    print(f"  K) OI_MOMENTUM_STRONG (OI+5% & ROC5>0): {np.sum(sig_K)} signals")

    # -- L) OI SURGE + ROC COMBO --
    # OI > 1.3 * SMA(OI,20) AND C > O AND ROC(5) > 0
    sig_L = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(25, ND):
            oi_v = OI[si, di]
            oi_sma = OI_SMA20[si, di]
            c_v = C[si, di]
            o_v = O[si, di]
            roc = ROC5[si, di]
            if np.isnan(oi_v) or np.isnan(oi_sma) or np.isnan(c_v) or np.isnan(o_v) or np.isnan(roc):
                continue
            if oi_sma > 0 and oi_v > 1.3 * oi_sma and c_v > o_v and roc > 0:
                sig_L[si, di] = True
    print(f"  L) OI_SURGE+ROC (OI>1.3xSMA20 & C>O & ROC5>0): {np.sum(sig_L)} signals")

    print(f"\n  All signals computed ({time.time()-t_start:.1f}s total)")

    # ================================================================
    # BACKTEST ENGINE
    # ================================================================
    def run_backtest(config, wf_test_year=None):
        sig_type = config['signal']
        hold_days = config['hold_days']
        top_n = config['top_n']
        use_score = config.get('use_score', None)
        comm = config.get('comm', COMM)

        # Map signal type to signal array and score array
        sig_map = {
            'A': sig_A, 'B': sig_B, 'C': sig_C, 'D': sig_D,
            'E': sig_E, 'F': sig_F, 'G': sig_G, 'H': sig_H,
            'I': sig_I, 'J': sig_J, 'K': sig_K, 'L': sig_L,
        }
        score_map = {
            'A': score_F, 'B': score_F, 'C': score_F, 'D': score_F,
            'E': score_F, 'F': score_F, 'G': score_F, 'H': score_F,
            'I': score_F, 'J': score_J, 'K': score_F, 'L': score_F,
        }
        sig_arr = sig_map[sig_type]
        sc_arr = score_map[sig_type]

        # Date boundaries
        if wf_test_year is not None:
            test_start_di = None
            test_end_di = None
            for di in range(ND):
                if dates[di].year == wf_test_year and test_start_di is None:
                    test_start_di = di
                if dates[di].year == wf_test_year + 1 and test_end_di is None:
                    test_end_di = di
            if test_start_di is None:
                return None
            if test_end_di is None:
                test_end_di = ND
            start_di = MIN_TRAIN
            end_di = test_end_di
        else:
            test_start_di = MIN_TRAIN
            start_di = MIN_TRAIN
            end_di = ND
            test_end_di = ND

        if end_di < start_di + hold_days + 2:
            return None

        cash = float(CASH0)
        positions = []
        trades = []

        for di in range(start_di, end_di - 1):
            # Reset cash at test window start (WF mode)
            if wf_test_year is not None and di == test_start_di:
                cash = float(CASH0)
                positions = []

            # -- Close positions -----------------------------------------
            closed = []
            for pos in positions:
                days_held = di - pos['entry_di']
                if days_held >= pos['hold_days']:
                    exit_price = C[pos['si'], di]
                    if np.isnan(exit_price) or exit_price <= 0:
                        exit_price = pos['entry_price']
                    mult = MULT.get(pos['sym'], DEF_MULT)
                    mkt_val = exit_price * mult * abs(pos['lots'])
                    cash += mkt_val - mkt_val * comm
                    pnl = (exit_price - pos['entry_price']) * mult * pos['lots']
                    invested = pos['entry_price'] * mult * abs(pos['lots'])
                    pnl_pct = pnl / invested * 100 if invested > 0 else 0
                    trades.append({
                        'pnl_pct': pnl_pct,
                        'entry_di': pos['entry_di'],
                        'exit_di': di,
                        'year': dates[di].year if di < ND else dates[-1].year,
                        'sym': pos.get('sym', ''),
                        'days_held': days_held,
                    })
                    closed.append(pos)

            for pos in closed:
                positions.remove(pos)

            if len(positions) >= top_n:
                continue

            # -- Generate signals at day di --------------------------------
            entry_di = di + 1
            if entry_di >= end_di:
                continue

            candidates = []
            for si in range(NS):
                if not sig_arr[si, di]:
                    continue
                if any(p['si'] == si for p in positions):
                    continue
                ep = O[si, entry_di]
                if np.isnan(ep) or ep <= 0:
                    continue
                sc = sc_arr[si, di] if not np.isnan(sc_arr[si, di]) else 0
                candidates.append((sc, {
                    'si': si, 'sym': syms[si], 'entry_price': ep,
                }))

            if not candidates:
                continue

            # Sort by score descending
            candidates.sort(key=lambda x: -x[0])

            # Open positions
            n_slots = top_n - len(positions)
            for sc_val, info in candidates[:max(0, n_slots)]:
                si = info['si']
                sym = info['sym']
                price = info['entry_price']
                mult = MULT.get(sym, DEF_MULT)
                contracts = max(1, int(cash / (price * mult)))
                cost_in = price * mult * contracts * (1 + comm)
                if cost_in > cash:
                    contracts = int(cash * 0.9 / (price * mult * (1 + comm)))
                    cost_in = price * mult * contracts * (1 + comm) if contracts > 0 else 0
                if contracts <= 0 or cost_in <= 0 or cost_in > cash:
                    continue

                cash -= cost_in
                positions.append({
                    'si': si, 'entry_price': price, 'entry_di': entry_di,
                    'lots': contracts, 'dir': 1, 'sym': sym,
                    'hold_days': hold_days,
                })

        # Close remaining positions at end
        for pos in positions:
            ae = end_di - 1 if end_di < ND else ND - 1
            exit_price = C[pos['si'], ae]
            if np.isnan(exit_price) or exit_price <= 0:
                exit_price = pos['entry_price']
            mult = MULT.get(pos['sym'], DEF_MULT)
            mkt_val = exit_price * mult * abs(pos['lots'])
            cash += mkt_val - mkt_val * comm
            pnl = (exit_price - pos['entry_price']) * mult * pos['lots']
            invested = pos['entry_price'] * mult * abs(pos['lots'])
            pnl_pct = pnl / invested * 100 if invested > 0 else 0
            trades.append({
                'pnl_pct': pnl_pct,
                'entry_di': pos['entry_di'],
                'exit_di': ae,
                'year': dates[ae].year if ae < ND else dates[-1].year,
                'sym': pos.get('sym', ''),
                'days_held': ae - pos['entry_di'],
            })

        # Results
        wf_mode = wf_test_year is not None
        n_days_test = (test_end_di - test_start_di) if wf_mode else (end_di - start_di)
        ann = annual_return(cash, CASH0, n_days_test)
        wr = np.mean([1 if t['pnl_pct'] > 0 else 0 for t in trades]) * 100 if trades else 0
        n_trades = len(trades)
        avg_pnl = np.mean([t['pnl_pct'] for t in trades]) if trades else 0
        avg_hold = np.mean([t['days_held'] for t in trades]) if trades else 0

        # Max drawdown from equity curve
        eq = float(CASH0)
        peak = eq
        mdd = 0.0
        for t in trades:
            eq *= (1 + t['pnl_pct'] / 100)
            if eq > peak:
                peak = eq
            dd = (eq - peak) / peak * 100
            if dd < mdd:
                mdd = dd

        freq_per_yr = n_trades / (n_days_test / 252) if n_days_test > 0 else 0

        return {
            'ann': ann, 'wr': wr, 'n': n_trades, 'avg_pnl': avg_pnl,
            'final_cash': cash, 'n_days': n_days_test, 'mdd': mdd,
            'avg_hold': avg_hold, 'freq': freq_per_yr,
        }

    # ================================================================
    # BUILD CONFIGURATIONS
    # ================================================================
    print("\n[Sweep] Building configurations...", flush=True)
    configs = []
    cid = 0

    sig_labels = {
        'A': 'OI_Momentum',
        'B': 'OI_Surge',
        'C': 'OI_PriceDiv',
        'D': 'OI_Capitul',
        'E': 'Vol_OI_Ratio',
        'F': 'OI_Wt_ROC',
        'G': 'TS_BackFlip',
        'H': 'TS_SpreadCmp',
        'I': 'TS_Back+ROC',
        'J': 'OI+TS+ROC',
        'K': 'OI_MomStrong',
        'L': 'OI_Surge+ROC',
    }

    for sig_key in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']:
        for hd in [5, 10]:
            for tn in [1, 3]:
                cid += 1
                configs.append({
                    'id': cid, 'signal': sig_key,
                    'hold_days': hd, 'top_n': tn, 'comm': COMM,
                    'label': f"{sig_key}_{sig_labels[sig_key]}_H{hd}_TN{tn}",
                })

    print(f"  Total configs: {len(configs)}")

    # ================================================================
    # RUN FULL-PERIOD BACKTEST
    # ================================================================
    print("\n[Backtest] Running full-period sweep...", flush=True)
    results = []
    for i, cfg in enumerate(configs):
        r = run_backtest(cfg)
        if r:
            r['config'] = cfg
            r['label'] = cfg['label']
            results.append(r)
        if (i + 1) % 10 == 0 or i == len(configs) - 1:
            print(f"  ... {i+1}/{len(configs)} done ({time.time()-t_start:.0f}s)", flush=True)

    results.sort(key=lambda x: -x['ann'])

    # ================================================================
    # FULL-PERIOD RESULTS
    # ================================================================
    print(f"\n{'=' * 200}")
    print("  FULL-PERIOD RESULTS (All configs) -- NEXT-OPEN EXECUTION, OI + TERM STRUCTURE SIGNALS")
    print(f"{'=' * 200}")
    print(f"  {'#':>3} | {'Label':<36} | {'Ann':>9} | {'WR':>6} | {'N':>5} | {'AvgPnL':>7} | {'MDD':>7} | {'AvgHold':>7} | {'Freq/Yr':>7} | {'Final':>14}")
    print("-" * 180)
    for i, r in enumerate(results):
        print(f"  {i+1:>3} | {r['label']:<36} | {r['ann']:>+8.1f}% | {r['wr']:>5.1f}% | {r['n']:>5} | {r['avg_pnl']:>+6.3f}% | {r['mdd']:>6.1f}% | {r['avg_hold']:>6.1f}d | {r['freq']:>6.1f}/yr | {r['final_cash']:>13,.0f}")

    # ================================================================
    # BEST PER SIGNAL TYPE
    # ================================================================
    sig_names = {
        'A': 'A) OI MOMENTUM (OI+3% & ROC5>0)',
        'B': 'B) OI SURGE (OI>1.5xSMA20 & C>O)',
        'C': 'C) OI-PRICE DIVERGENCE (Price+OI both up)',
        'D': 'D) OI CAPITULATION (OI-5% & ROC5<-3%)',
        'E': 'E) VOLUME/OI RATIO (turnover>2x avg)',
        'F': 'F) OI-WEIGHTED ROC (score>0)',
        'G': 'G) BACKWARDATION FLIP (contango->back)',
        'H': 'H) SPREAD COMPRESSION (narrowing & near up)',
        'I': 'I) TERM STRUCTURE + ROC (back+ROC5>0)',
        'J': 'J) OI + TS + ROC TRIPLE',
        'K': 'K) OI MOMENTUM STRONG (OI+5% & ROC5>0)',
        'L': 'L) OI SURGE + ROC (OI>1.3xSMA20 & C>O & ROC5>0)',
    }

    print(f"\n{'=' * 200}")
    print("  BEST PER SIGNAL TYPE (Full Period)")
    print(f"{'=' * 200}")
    print(f"  {'Signal':<55} | {'Ann':>9} | {'WR':>6} | {'N':>5} | {'AvgPnL':>7} | {'MDD':>7} | {'AvgHold':>7} | {'Freq/Yr':>7} | Best Config")
    print("-" * 200)

    best_per_sig = {}
    for r in results:
        key = r['config']['signal']
        if key not in best_per_sig:
            best_per_sig[key] = r

    for sig_key in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']:
        if sig_key in best_per_sig:
            b = best_per_sig[sig_key]
            print(f"  {sig_names.get(sig_key, sig_key):<55} | {b['ann']:>+8.1f}% | {b['wr']:>5.1f}% | {b['n']:>5} | {b['avg_pnl']:>+6.3f}% | {b['mdd']:>6.1f}% | {b['avg_hold']:>6.1f}d | {b['freq']:>6.1f}/yr | {b['label']}")

    # ================================================================
    # SIGNAL TYPE SUMMARY
    # ================================================================
    print(f"\n{'=' * 200}")
    print("  SIGNAL TYPE SUMMARY (Average of all configs per type)")
    print(f"{'=' * 200}")
    print(f"  {'Signal':<55} | {'Avg Ann':>9} | {'Avg WR':>7} | {'Avg N':>7} | {'Avg PnL':>8} | {'Avg MDD':>8} | {'#Positive':>9}")
    print("-" * 170)

    for sig_key in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']:
        sub = [r for r in results if r['config']['signal'] == sig_key]
        if not sub:
            continue
        avg_ann = np.mean([r['ann'] for r in sub])
        avg_wr = np.mean([r['wr'] for r in sub])
        avg_n = np.mean([r['n'] for r in sub])
        avg_pnl = np.mean([r['avg_pnl'] for r in sub])
        avg_mdd = np.mean([r['mdd'] for r in sub])
        n_pos = sum(1 for r in sub if r['ann'] > 0)
        print(f"  {sig_names.get(sig_key, sig_key):<55} | {avg_ann:>+8.1f}% | {avg_wr:>6.1f}% | {avg_n:>7.0f} | {avg_pnl:>+7.3f}% | {avg_mdd:>7.1f}% | {n_pos:>5}/{len(sub)}")

    # ================================================================
    # OI vs TERM STRUCTURE COMPARISON
    # ================================================================
    print(f"\n{'=' * 200}")
    print("  CATEGORY COMPARISON: OI SIGNALS vs TERM STRUCTURE SIGNALS vs COMBINED")
    print(f"{'=' * 200}")
    oi_sigs = ['A', 'B', 'C', 'D', 'E', 'F', 'K', 'L']
    ts_sigs = ['G', 'H', 'I']
    combo_sigs = ['J']

    for cat_name, cat_keys in [('OI SIGNALS (A,B,C,D,E,F,K,L)', oi_sigs),
                                ('TERM STRUCTURE (G,H,I)', ts_sigs),
                                ('COMBO OI+TS (J)', combo_sigs)]:
        sub = [r for r in results if r['config']['signal'] in cat_keys]
        if not sub:
            continue
        avg_ann = np.mean([r['ann'] for r in sub])
        best_ann = max(r['ann'] for r in sub)
        avg_wr = np.mean([r['wr'] for r in sub])
        n_pos = sum(1 for r in sub if r['ann'] > 0)
        print(f"  {cat_name:<55} | Avg Ann: {avg_ann:>+8.1f}% | Best: {best_ann:>+8.1f}% | WR: {avg_wr:>5.1f}% | {n_pos}/{len(sub)} positive")

    # ================================================================
    # WALK-FORWARD (Top 15 configs + best per signal type)
    # ================================================================
    wf_years = [2020, 2021, 2022, 2023, 2024, 2025]

    # Collect top 15 overall + best per signal type
    wf_configs = list(results[:15])
    for sig_key in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']:
        if sig_key in best_per_sig:
            r = best_per_sig[sig_key]
            if r['config'] not in [w['config'] for w in wf_configs]:
                wf_configs.append(r)

    print(f"\n{'=' * 220}")
    print(f"  WALK-FORWARD ({len(wf_configs)} configs)")
    print(f"{'=' * 220}")

    header = f"  {'#':>3} | {'Config':<36} | {'Avg':>8} |"
    for yr in wf_years:
        header += f" {yr:>7} |"
    header += f" {'Pos':>4} | {'MDD':>7} | {'WR':>6}"
    print(header)
    print("-" * 220)

    wf_rows = []
    for i, r in enumerate(wf_configs):
        cfg = r['config']
        wf_row = {'label': cfg['label'], 'signal': cfg['signal'],
                  'entry': 'next_open', 'windows': {}, 'mdd': {}, 'wr': {}}
        for yr in wf_years:
            wr = run_backtest(cfg, wf_test_year=yr)
            if wr:
                wf_row['windows'][yr] = wr['ann']
                wf_row['mdd'][yr] = wr['mdd']
                wf_row['wr'][yr] = wr['wr']
        wf_rows.append(wf_row)

        vals = [wf_row['windows'].get(yr, 0) for yr in wf_years]
        avg = np.mean(vals) if vals else 0
        pos = sum(1 for v in vals if v > 0)
        avg_mdd = np.mean(list(wf_row['mdd'].values())) if wf_row['mdd'] else 0
        avg_wr = np.mean(list(wf_row['wr'].values())) if wf_row['wr'] else 0

        row_str = f"  {i+1:>3} | {wf_row['label']:<36} | {avg:>+7.1f}% |"
        for v in vals:
            row_str += f" {v:>+7.1f}% |"
        row_str += f" {pos}/6 | {avg_mdd:>6.1f}% | {avg_wr:>5.1f}%"
        print(row_str)

    # ================================================================
    # WF COMPARISON PER SIGNAL
    # ================================================================
    print(f"\n{'=' * 200}")
    print("  WALK-FORWARD COMPARISON (Best per signal type)")
    print(f"{'=' * 200}")
    header2 = f"  {'Signal':<55} | {'WF Avg':>8} |"
    for yr in wf_years:
        header2 += f" {yr:>7} |"
    header2 += f" {'Pos':>4} | Avg MDD | Avg WR"
    print(header2)
    print("-" * 200)

    for sig_key in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']:
        wf_match = [w for w in wf_rows if w['signal'] == sig_key]
        if wf_match:
            wf = wf_match[0]
            vals = [wf['windows'].get(yr, 0) for yr in wf_years]
            avg = np.mean(vals) if vals else 0
            pos = sum(1 for v in vals if v > 0)
            avg_mdd = np.mean(list(wf['mdd'].values())) if wf['mdd'] else 0
            avg_wr = np.mean(list(wf['wr'].values())) if wf['wr'] else 0
            row_str = f"  {sig_names.get(sig_key, sig_key):<55} | {avg:>+7.1f}% |"
            for v in vals:
                row_str += f" {v:>+7.1f}% |"
            row_str += f" {pos}/6 | {avg_mdd:>6.1f}% | {avg_wr:>5.1f}%"
            print(row_str)

    # ================================================================
    # FINAL VERDICT
    # ================================================================
    print(f"\n{'=' * 200}")
    print("  FINAL VERDICT: OI + TERM STRUCTURE DEEP DIVE")
    print(f"{'=' * 200}")
    print()
    print("  KEY QUESTIONS:")
    print("  1. Which OI signals work with next-open execution?")
    print("  2. Term structure signal viability?")
    print("  3. Best configs by annual return?")
    print("  4. Any config beating +81.9%?")
    print()

    for sig_key in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']:
        sub = [r for r in results if r['config']['signal'] == sig_key]
        if not sub:
            continue
        best = sub[0]
        n_pos = sum(1 for r in sub if r['ann'] > 0)

        wf_match = [w for w in wf_rows if w['signal'] == sig_key]
        wf_pos = 0
        wf_avg = 0
        if wf_match:
            wf = wf_match[0]
            vals = [wf['windows'].get(yr, 0) for yr in wf_years]
            wf_pos = sum(1 for v in vals if v > 0)
            wf_avg = np.mean(vals)

        verdict = "POSITIVE" if best['ann'] > 0 else "NEGATIVE"
        genuine = "GENUINE ALPHA" if wf_pos >= 4 and best['ann'] > 0 else ("MARGINAL" if wf_pos >= 3 and best['ann'] > 0 else "NO ALPHA")
        beats = "BEATS +81.9%" if best['ann'] > 81.9 else ("CLOSE" if best['ann'] > 50 else "INSUFFICIENT")

        category = "OI" if sig_key in oi_sigs else ("TS" if sig_key in ts_sigs else "COMBO")
        print(f"  [{category}] {sig_names.get(sig_key, sig_key)}")
        print(f"    Best annual: {best['ann']:>+8.1f}%  |  {n_pos}/{len(sub)} positive configs")
        print(f"    Walk-forward: {wf_pos}/6 positive  |  WF avg: {wf_avg:>+8.1f}%")
        print(f"    Trade freq: {best['freq']:>5.1f}/yr  |  Avg hold: {best['avg_hold']:>5.1f}d  |  Avg PnL: {best['avg_pnl']:>+6.3f}%")
        print(f"    VERDICT: {verdict}  -->  {genuine}  -->  {beats}")
        print()

    # Absolute best
    if results:
        champ = results[0]
        print(f"  {'='*70}")
        print(f"  CHAMPION: {champ['label']}")
        print(f"    Annual: {champ['ann']:>+8.1f}%  |  WR: {champ['wr']:>5.1f}%  |  N: {champ['n']:>4}  |  MDD: {champ['mdd']:>6.1f}%")
        print(f"    Avg PnL/trade: {champ['avg_pnl']:>+6.3f}%  |  Avg Hold: {champ['avg_hold']:>5.1f}d  |  Freq: {champ['freq']:>5.1f}/yr")
        champ_wf = [w for w in wf_rows if w['label'] == champ['label']]
        if champ_wf:
            cw = champ_wf[0]
            vals = [cw['windows'].get(yr, 0) for yr in wf_years]
            print(f"    WF: {[f'{v:>+7.1f}%' for v in vals]}  |  {sum(1 for v in vals if v > 0)}/6 positive")
        print(f"  {'='*70}")

    # Top 5 summary
    print(f"\n  TOP 5 CONFIGS:")
    print(f"  {'#':>3} | {'Label':<36} | {'Ann':>9} | {'WR':>6} | {'N':>5} | {'MDD':>7} | WF_Avg | WF_Pos")
    print("-" * 130)
    for i, r in enumerate(results[:5]):
        wf_match = [w for w in wf_rows if w['label'] == r['label']]
        if wf_match:
            wf = wf_match[0]
            vals = [wf['windows'].get(yr, 0) for yr in wf_years]
            wf_avg = np.mean(vals)
            wf_pos = sum(1 for v in vals if v > 0)
        else:
            wf_avg = 0
            wf_pos = 0
        print(f"  {i+1:>3} | {r['label']:<36} | {r['ann']:>+8.1f}% | {r['wr']:>5.1f}% | {r['n']:>5} | {r['mdd']:>6.1f}% | {wf_avg:>+7.1f}% | {wf_pos}/6")

    # Beating +81.9%?
    beating = [r for r in results if r['ann'] > 81.9]
    if beating:
        print(f"\n  CONFIGS BEATING +81.9%: {len(beating)}")
        for r in beating[:10]:
            wf_match = [w for w in wf_rows if w['label'] == r['label']]
            wf_pos = 0
            wf_avg = 0
            if wf_match:
                wf = wf_match[0]
                vals = [wf['windows'].get(yr, 0) for yr in wf_years]
                wf_pos = sum(1 for v in vals if v > 0)
                wf_avg = np.mean(vals)
            print(f"    {r['label']:<36} | Ann: {r['ann']:>+8.1f}% | WF: {wf_avg:>+7.1f}% | WF_pos: {wf_pos}/6")
    else:
        print(f"\n  No configs beating +81.9%")

    print(f"\n  Total runtime: {time.time()-t_start:.0f}s")


if __name__ == '__main__':
    main()
