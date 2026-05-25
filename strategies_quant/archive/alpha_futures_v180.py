"""
Alpha Futures V180 -- Dual-Side Pair Trading Strategy
==============================================================================
Combines pair trading (V62) with dual-side long+short approach (V177).

Core idea:
  1. Find correlated commodity pairs with divergent spreads
  2. Go LONG the underperformer AND SHORT the overperformer (natural hedge)
  3. Add V121 momentum filter for timing confirmation

Signal Logic:
  - Track log price ratio with 20-day z-score
  - When z > threshold: pair stretched -> SHORT A, LONG B (expect convergence)
  - When z < -threshold: pair compressed -> LONG A, SHORT B (expect convergence)
  - V121 momentum filter: require at least one side has |ROC(5)|>1%

Supply Chain Pairs (from available data):
  Ferrous: rbfi/hcfi, jfi/jmfi, rbfi/jfi, hcfi/jfi, rbfi/jmfi
  Metals:  cufi/alfi, cufi/znfi, alfi/znfi, nifi/snfi, agfi/aufi
  Energy:  scfi/fufi, mafi/scfi, egfi/mafi, ppfi/mafi, vfi/mafi,
           ppfi/egfi, ppfi/vfi, pgfi/scfi, ebfi/scfi
  Chem:    fgfi/vfi, pffi/ppfi
  Agri:    aofi/safi

Test matrix:
  1. BASELINE: V177 dual-side (for comparison)
  2. PAIR_ONLY: Pure pair trading with z>2 threshold
  3. PAIR_MOMENTUM: Pair trading + V121 momentum filter
  4. PAIR_LOOSE: z>1.5 threshold (more trades)
  5. PAIR_TIGHT: z>2.5 threshold (fewer, higher quality trades)

Walk-forward validation across 2020-2025.
"""
import sys, os, time, warnings
import numpy as np
import talib
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alpha_v2 import load_all_data, MIN_TRAIN, CASH0

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
        'wrfi': 10, 'pffi': 5, 'nrfi': 1, 'prfi': 5, 'shfi': 5,
        'ni': 1, 'tai': 5}
DEF_MULT = 10
COMM = 0.0003

# Comprehensive supply chain / structural pairs
ALL_PAIRS = [
    # Ferrous chain
    ('rbfi', 'hcfi'),   # rebar / HRC
    ('jfi', 'jmfi'),    # coke / coking coal
    ('rbfi', 'jfi'),    # rebar / coke
    ('hcfi', 'jfi'),    # HRC / coke
    ('rbfi', 'jmfi'),   # rebar / coking coal
    # Metals
    ('cufi', 'alfi'),   # copper / aluminum
    ('cufi', 'znfi'),   # copper / zinc
    ('alfi', 'znfi'),   # aluminum / zinc
    ('nifi', 'snfi'),   # nickel / tin
    ('agfi', 'aufi'),   # silver / gold
    # Energy/petrochemical
    ('scfi', 'fufi'),   # crude / fuel oil
    ('mafi', 'scfi'),   # methanol / crude
    ('egfi', 'mafi'),   # EG / methanol
    ('ppfi', 'mafi'),   # PP / methanol
    ('vfi', 'mafi'),    # PVC / methanol
    ('ppfi', 'egfi'),   # PP / EG
    ('ppfi', 'vfi'),    # PP / PVC
    ('pgfi', 'scfi'),   # LPG / crude
    # Chemicals
    ('fgfi', 'vfi'),    # glass / PVC
    ('pffi', 'ppfi'),   # short fiber / PP
    # Agriculture
    ('aofi', 'safi'),   # soy / sugar
]

PAIR_LABELS = {
    ('rbfi', 'hcfi'): 'rebar/HRC',     ('jfi', 'jmfi'): 'coke/coal',
    ('rbfi', 'jfi'): 'rebar/coke',     ('hcfi', 'jfi'): 'HRC/coke',
    ('rbfi', 'jmfi'): 'rebar/coal',    ('cufi', 'alfi'): 'copper/aluminum',
    ('cufi', 'znfi'): 'copper/zinc',   ('alfi', 'znfi'): 'aluminum/zinc',
    ('nifi', 'snfi'): 'nickel/tin',    ('agfi', 'aufi'): 'silver/gold',
    ('scfi', 'fufi'): 'crude/fueloil', ('mafi', 'scfi'): 'methanol/crude',
    ('egfi', 'mafi'): 'EG/methanol',   ('ppfi', 'mafi'): 'PP/methanol',
    ('vfi', 'mafi'): 'PVC/methanol',   ('ppfi', 'egfi'): 'PP/EG',
    ('ppfi', 'vfi'): 'PP/PVC',         ('pgfi', 'scfi'): 'LPG/crude',
    ('fgfi', 'vfi'): 'glass/PVC',      ('pffi', 'ppfi'): 'fiber/PP',
    ('aofi', 'safi'): 'soy/sugar',
}

# Walk-forward windows
WF_WINDOWS = [
    (2019, 2020), (2020, 2021), (2021, 2022),
    (2022, 2023), (2023, 2024), (2024, 2025),
]


def annual_return(final, initial, n_days):
    if final <= 0 or initial <= 0 or n_days <= 0:
        return -100.0
    return (final / initial) ** (1.0 / (n_days / 252)) * 100 - 100


def main():
    print("=" * 140)
    print("  V180 -- Dual-Side Pair Trading Strategy")
    print("  Combine pair trading (V62) + dual-side long/short (V177) + V121 momentum filter")
    print("=" * 140)
    t_start = time.time()

    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)
    sym_to_si = {syms[si]: si for si in range(NS)}
    print(f"  {NS} commodities, {ND} days")

    # Year boundaries
    year_start_di = {}
    year_end_di = {}
    for di in range(ND):
        y = dates[di].year
        if y not in year_start_di:
            year_start_di[y] = di
        year_end_di[y] = di

    # ===================== PRECOMPUTE SINGLE-SYMBOL INDICATORS =====================
    print("\n[Precompute] Single-symbol indicators...", flush=True)
    t0 = time.time()

    RET = np.full((NS, ND), np.nan)
    ROC5 = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si].astype(np.float64)
        for di in range(1, ND):
            if not np.isnan(c[di]) and not np.isnan(c[di - 1]) and c[di - 1] > 0:
                RET[si, di] = (c[di] / c[di - 1] - 1) * 100
        ROC5[si] = talib.ROC(c, timeperiod=5)

    ATR14 = np.full((NS, ND), np.nan)
    for si in range(NS):
        ATR14[si] = talib.ATR(H[si].astype(np.float64), L[si].astype(np.float64),
                               C[si].astype(np.float64), timeperiod=14)

    ATR_NORM = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(ND):
            atr = ATR14[si, di]
            cp = C[si, di]
            if not np.isnan(atr) and not np.isnan(cp) and cp > 0:
                ATR_NORM[si, di] = atr / cp * 100

    ZSCORE = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(21, ND):
            rets = RET[si, di - 20:di]
            v = rets[~np.isnan(rets)]
            if len(v) < 10:
                continue
            s = np.std(v, ddof=1)
            if s > 0 and not np.isnan(RET[si, di]):
                ZSCORE[si, di] = (RET[si, di] - np.mean(v)) / s

    # Regime
    BREADTH = np.full(ND, np.nan)
    for di in range(5, ND):
        pos_count = 0
        total = 0
        for si in range(NS):
            r = ROC5[si, di]
            if not np.isnan(r):
                total += 1
                if r > 0:
                    pos_count += 1
        if total > 0:
            BREADTH[di] = pos_count / total

    MKT_RET = np.full(ND, np.nan)
    for di in range(ND):
        rets_day = RET[:, di]
        valid = rets_day[~np.isnan(rets_day)]
        if len(valid) > 10:
            MKT_RET[di] = np.mean(valid)

    MKT_VOL = np.full(ND, np.nan)
    for di in range(20, ND):
        window = MKT_RET[di - 20:di]
        valid = window[~np.isnan(window)]
        if len(valid) >= 10:
            MKT_VOL[di] = np.std(valid, ddof=1)

    valid_vols = MKT_VOL[~np.isnan(MKT_VOL)]
    VOL_MEDIAN = np.median(valid_vols) if len(valid_vols) > 0 else 1.0

    print(f"  Single-symbol done ({time.time() - t0:.1f}s)")

    # ===================== PRECOMPUTE PAIR INDICATORS =====================
    print("\n[Precompute] Pair indicators...", flush=True)
    t1 = time.time()

    pair_indices = []
    for sym_a, sym_b in ALL_PAIRS:
        si_a = sym_to_si.get(sym_a, -1)
        si_b = sym_to_si.get(sym_b, -1)
        if si_a >= 0 and si_b >= 0:
            pair_indices.append((si_a, si_b, sym_a, sym_b))
        else:
            missing = sym_a if si_a < 0 else sym_b
            print(f"  WARNING: {missing} not found, skipping pair ({sym_a}, {sym_b})")
    print(f"  Valid pairs: {len(pair_indices)}")

    Z_WINDOW = 20
    pair_zscore = {}
    pair_corr = {}
    # Also compute spread return (daily change in log ratio)
    pair_spread_ret = {}

    for si_a, si_b, sym_a, sym_b in pair_indices:
        key = (si_a, si_b)
        ca = C[si_a].astype(np.float64)
        cb = C[si_b].astype(np.float64)

        log_ratio = np.full(ND, np.nan)
        for di in range(ND):
            if not np.isnan(ca[di]) and not np.isnan(cb[di]) and ca[di] > 0 and cb[di] > 0:
                log_ratio[di] = np.log(ca[di]) - np.log(cb[di])

        # Rolling z-score
        z = np.full(ND, np.nan)
        for di in range(Z_WINDOW, ND):
            window = log_ratio[di - Z_WINDOW:di]
            valid = window[~np.isnan(window)]
            if len(valid) >= 10:
                m = np.mean(valid)
                s = np.std(valid, ddof=1)
                if s > 1e-10 and not np.isnan(log_ratio[di]):
                    z[di] = (log_ratio[di] - m) / s
        pair_zscore[key] = z

        # Rolling correlation
        corr_arr = np.full(ND, np.nan)
        for di in range(Z_WINDOW, ND):
            ret_a = RET[si_a, di - Z_WINDOW:di]
            ret_b = RET[si_b, di - Z_WINDOW:di]
            valid = ~(np.isnan(ret_a) | np.isnan(ret_b))
            n_valid = np.sum(valid)
            if n_valid >= 8:
                ra = ret_a[valid]
                rb = ret_b[valid]
                if np.std(ra) > 0 and np.std(rb) > 0:
                    c = np.corrcoef(ra, rb)[0, 1]
                    if not np.isnan(c):
                        corr_arr[di] = c
        pair_corr[key] = corr_arr

        # Spread daily return (for forward-looking signal quality)
        spread_ret = np.full(ND, np.nan)
        for di in range(1, ND):
            if not np.isnan(log_ratio[di]) and not np.isnan(log_ratio[di - 1]):
                spread_ret[di] = log_ratio[di] - log_ratio[di - 1]
        pair_spread_ret[key] = spread_ret

    print(f"  Pair indicators done ({time.time() - t1:.1f}s)")

    # ===================== HELPERS =====================
    def compute_composite(di, daily_eq, high_water):
        scores = []
        bth = BREADTH[di]
        if not np.isnan(bth):
            scores.append(np.clip((bth - 0.4) / (0.7 - 0.4), 0, 1))
        vol = MKT_VOL[di]
        if not np.isnan(vol) and VOL_MEDIAN > 0:
            vol_ratio = vol / VOL_MEDIAN
            scores.append(np.clip((1.5 - vol_ratio) / (1.5 - 0.8), 0, 1))
        if len(daily_eq) >= 20:
            eq_window = np.array(daily_eq[-20:])
            x = np.arange(20)
            try:
                slope = np.polyfit(x, eq_window, 1)[0]
                eq_mean = np.mean(eq_window)
                norm_slope = slope / eq_mean * 100 if eq_mean > 0 else 0
                eq_rets = np.diff(eq_window) / eq_window[:-1] * 100
                eq_rets = eq_rets[np.isfinite(eq_rets)]
                eq_std = np.std(eq_rets) if len(eq_rets) > 5 else 1.0
                z = norm_slope / eq_std if eq_std > 0 else 0
                scores.append(np.clip((z + 1.0) / 2.0, 0, 1))
            except Exception:
                pass
        if high_water > 0:
            cur_dd = (daily_eq[-1] - high_water) / high_water
        else:
            cur_dd = 0
        scores.append(np.clip(1.0 + cur_dd / 0.3, 0, 1))
        return np.mean(scores) if scores else 0.5

    def dd_size(pv, high_water, tiers):
        if high_water <= 0:
            return tiers[0][1]
        dd = (pv - high_water) / high_water
        for dd_thresh, size_frac in tiers:
            if dd >= -dd_thresh:
                return size_frac
        return tiers[-1][1]

    # ===================== BACKTEST: V177 BASELINE =====================
    def backtest_v177(start_di=MIN_TRAIN, end_di=None,
                      atr_norm_max=12.0, max_corr=0.7,
                      dd_tiers=None, regime_lo=0.5, regime_hi=1.5,
                      top_n=3, hold=1, short_mode='short_mirror'):
        if end_di is None:
            end_di = ND
        if dd_tiers is None:
            dd_tiers = [(0, 1.00), (0.10, 0.90), (0.20, 0.70), (0.30, 0.50)]

        cash = float(CASH0)
        positions = []
        trades = []
        daily_eq = []
        high_water = float(CASH0)

        for di in range(start_di, end_di - 1):
            pv = cash
            for p in positions:
                cp = C[p['si'], di]
                if not np.isnan(cp) and cp > 0:
                    m = MULT.get(p['sym'], DEF_MULT)
                    d = p.get('dir', 1)
                    unrealized = (cp - p['entry_price']) * m * p['lots'] * d
                    pv += p['entry_price'] * m * abs(p['lots']) + unrealized - cp * m * abs(p['lots']) * COMM
            daily_eq.append(pv)
            if pv > high_water:
                high_water = pv

            cl = []
            for p in positions:
                if di - p['entry_di'] >= p['hold_days']:
                    ep = C[p['si'], di]
                    if np.isnan(ep) or ep <= 0:
                        ep = p['entry_price']
                    m = MULT.get(p['sym'], DEF_MULT)
                    d = p.get('dir', 1)
                    pnl = (ep - p['entry_price']) * m * p['lots'] * d
                    inv = p['entry_price'] * m * abs(p['lots'])
                    pp = pnl / inv * 100 if inv > 0 else 0
                    if d == 1:
                        cash += ep * m * abs(p['lots']) * (1 - COMM)
                    else:
                        margin = p['entry_price'] * m * abs(p['lots'])
                        cash += margin + pnl - ep * m * abs(p['lots']) * COMM
                    trades.append(pp)
                    cl.append(p)
            for p in cl:
                positions.remove(p)

            dd_sz = dd_size(pv, high_water, dd_tiers)
            composite = compute_composite(di, daily_eq, high_water)
            regime_mult = regime_lo + composite * (regime_hi - regime_lo)
            pos_size = max(0.05, min(0.99, dd_sz * regime_mult))

            if len(positions) >= top_n:
                continue
            edi = di + 1
            if edi >= end_di:
                continue

            held_si = set(p['si'] for p in positions)
            entries = []

            # Long signals
            for s in range(NS):
                roc = ROC5[s, di]
                zs = ZSCORE[s, di]
                if np.isnan(roc) or np.isnan(zs) or roc <= 1.0 or zs <= 1.5:
                    continue
                rp = ROC5[s, di - 1] if di > 0 else np.nan
                if not np.isnan(rp) and roc <= rp:
                    continue
                ep = O[s, edi]
                if np.isnan(ep) or ep <= 0:
                    continue
                if not np.isnan(ATR_NORM[s, di]) and ATR_NORM[s, di] >= atr_norm_max:
                    continue
                if s not in held_si:
                    entries.append((roc * zs, s, ep, 'v121_long', pos_size, 1))
                    break

            # Short signals
            if short_mode != 'long_only' and len(positions) + len(entries) < top_n:
                held_si2 = held_si | set(e[1] for e in entries)
                for s in range(NS):
                    roc = ROC5[s, di]
                    zs = ZSCORE[s, di]
                    if np.isnan(roc) or np.isnan(zs) or roc >= -1.0 or zs >= -1.5:
                        continue
                    rp = ROC5[s, di - 1] if di > 0 else np.nan
                    if not np.isnan(rp) and roc >= rp:
                        continue
                    ep = O[s, edi]
                    if np.isnan(ep) or ep <= 0:
                        continue
                    if not np.isnan(ATR_NORM[s, di]) and ATR_NORM[s, di] >= atr_norm_max:
                        continue
                    if s not in held_si2:
                        entries.append((abs(roc * zs), s, ep, 'v121_short', pos_size, -1))
                        break

            cash_snapshot = cash
            n_planned = len(entries)
            for sc, s, pr, sig_str, pct, d in entries:
                if s in set(p['si'] for p in positions):
                    continue
                if len(positions) >= top_n:
                    break
                cap = cash_snapshot * pct / max(n_planned, 1)
                sym = syms[s]
                m = MULT.get(sym, DEF_MULT)
                ct = max(1, int(cap / (pr * m * (1 + COMM))))
                ci = pr * m * ct * (1 + COMM)
                if ci > cash:
                    ct = int(cash * 0.9 / (pr * m * (1 + COMM)))
                    ci = pr * m * ct * (1 + COMM) if ct > 0 else 0
                if ct <= 0 or ci <= 0 or ci > cash:
                    continue
                cash -= ci
                positions.append({'si': s, 'entry_price': pr, 'entry_di': edi,
                                  'lots': ct, 'dir': d, 'sym': sym, 'hold_days': hold,
                                  'sig': sig_str, 'score': sc})

        for p in positions:
            ae = end_di - 1
            ep = C[p['si'], min(ae, ND - 1)]
            if np.isnan(ep) or ep <= 0:
                ep = p['entry_price']
            m = MULT.get(p['sym'], DEF_MULT)
            d = p.get('dir', 1)
            if d == 1:
                cash += ep * m * abs(p['lots']) * (1 - COMM)
            else:
                pnl = (ep - p['entry_price']) * m * p['lots'] * d
                margin = p['entry_price'] * m * abs(p['lots'])
                cash += margin + pnl - ep * m * abs(p['lots']) * COMM

        nd = end_di - start_di
        ann = annual_return(cash, CASH0, nd)
        wr = np.mean([1 if t > 0 else 0 for t in trades]) * 100 if trades else 0
        nt = len(trades)
        if daily_eq:
            eq = np.array(daily_eq)
            pk = np.maximum.accumulate(eq)
            mdd = np.min((eq - pk) / pk * 100)
            r = np.diff(eq) / eq[:-1]
            r = np.where(np.isfinite(r), r, 0)
            sh = np.mean(r) / np.std(r) * np.sqrt(252) if np.std(r) > 0 else 0
        else:
            mdd = 0; sh = 0
        return {'ann': ann, 'wr': wr, 'n': nt, 'mdd': mdd, 'sharpe': sh, 'final': cash}

    # ===================== BACKTEST: PAIR TRADING =====================
    def backtest_pair(start_di=MIN_TRAIN, end_di=None,
                      z_thresh=2.0, max_pairs=3, max_hold=5,
                      momentum_filter=False,
                      dd_tiers=None, regime_lo=0.5, regime_hi=1.5,
                      min_corr=0.3, corr_filter=False,
                      pair_set=None, exit_z=0.3):
        """
        Pair trading with V62-style cash management.

        When z > z_thresh: SHORT A, LONG B (expect ratio to revert down)
        When z < -z_thresh: LONG A, SHORT B (expect ratio to revert up)
        Exit when z crosses exit_z or max_hold days reached.

        Cash management:
          - Each pair trade is a self-contained position
          - Lock margin (entry cost for both legs)
          - On exit: return margin + realized PnL from both legs
        """
        if end_di is None:
            end_di = ND
        if dd_tiers is None:
            dd_tiers = [(0, 1.00), (0.10, 0.90), (0.20, 0.70), (0.30, 0.50)]
        if pair_set is None:
            pair_set = pair_indices

        cash = float(CASH0)
        pair_positions = []
        trades = []
        daily_eq = []
        high_water = float(CASH0)

        for di in range(start_di, end_di):
            # Mark-to-market: cash + unrealized PnL of open positions
            pv = cash
            for pos in pair_positions:
                c_a = C[pos['si_a'], di]
                c_b = C[pos['si_b'], di]
                if not np.isnan(c_a) and c_a > 0 and not np.isnan(c_b) and c_b > 0:
                    m_a = MULT.get(pos['sym_a'], DEF_MULT)
                    m_b = MULT.get(pos['sym_b'], DEF_MULT)
                    d = pos['dir']  # +1=long A/short B, -1=short A/long B
                    # PnL from leg A: long A = (c_a - entry_a), short A = (entry_a - c_a)
                    pnl_a = (c_a - pos['entry_a']) * m_a * pos['lots_a'] * d
                    # PnL from leg B: short B = (entry_b - c_b), long B = (c_b - entry_b)
                    pnl_b = (pos['entry_b'] - c_b) * m_b * pos['lots_b'] * d
                    pv += pnl_a + pnl_b
            daily_eq.append(pv)
            if pv > high_water:
                high_water = pv

            # --- Manage existing positions ---
            new_positions = []
            for pos in pair_positions:
                key = (pos['si_a'], pos['si_b'])
                z_arr = pair_zscore.get(key)
                if z_arr is None:
                    new_positions.append(pos)
                    continue

                z_now = z_arr[di] if di < len(z_arr) else np.nan
                days_held = di - pos['entry_di']
                exit_reason = None

                # Convergence exit: z returned toward 0
                if not np.isnan(z_now):
                    if pos['dir'] == 1 and z_now >= -exit_z:
                        exit_reason = 'converge'
                    elif pos['dir'] == -1 and z_now <= exit_z:
                        exit_reason = 'converge'
                    # Stop loss: z moved further against us
                    if exit_reason is None:
                        if pos['dir'] == 1 and z_now < pos['entry_z'] - 2.0:
                            exit_reason = 'stop_loss'
                        elif pos['dir'] == -1 and z_now > pos['entry_z'] + 2.0:
                            exit_reason = 'stop_loss'

                if exit_reason is None and days_held >= max_hold:
                    exit_reason = 'time'

                if exit_reason:
                    c_a = C[pos['si_a'], di]
                    c_b = C[pos['si_b'], di]
                    if np.isnan(c_a) or c_a <= 0:
                        c_a = pos['entry_a']
                    if np.isnan(c_b) or c_b <= 0:
                        c_b = pos['entry_b']

                    m_a = MULT.get(pos['sym_a'], DEF_MULT)
                    m_b = MULT.get(pos['sym_b'], DEF_MULT)
                    d = pos['dir']

                    # PnL calculation
                    pnl_a = (c_a - pos['entry_a']) * m_a * pos['lots_a'] * d
                    pnl_b = (pos['entry_b'] - c_b) * m_b * pos['lots_b'] * d

                    # Exit costs
                    exit_cost = (c_a * m_a * pos['lots_a'] + c_b * m_b * pos['lots_b']) * COMM
                    entry_cost = pos.get('entry_comm', 0)

                    total_pnl = pnl_a + pnl_b - exit_cost - entry_cost
                    invested = pos['cash_invested']
                    pnl_pct = total_pnl / invested * 100 if invested > 0 else 0

                    # Cash settlement
                    # Leg A: if dir=+1 (long), sell at c_a; if dir=-1 (short), buy back at c_a
                    # Leg B: if dir=+1 (short B), buy back at c_b; if dir=-1 (long B), sell at c_b
                    if d == 1:
                        # Long A, Short B -> sell A, buy back B
                        cash += c_a * m_a * pos['lots_a'] * (1 - COMM)  # sell A
                        cash -= c_b * m_b * pos['lots_b'] * (1 + COMM)  # buy B
                        # But we already locked margin, so we need to account for that
                        # Actually, simpler: return locked cash + settle PnL
                    else:
                        # Short A, Long B -> buy back A, sell B
                        cash -= c_a * m_a * pos['lots_a'] * (1 + COMM)  # buy A
                        cash += c_b * m_b * pos['lots_b'] * (1 - COMM)  # sell B

                    # The above double-counts. Let's use the V62 model:
                    # On entry: cash -= total_cost (margin locked)
                    # On exit: cash += margin_return + settlement
                    # For pair trade, settlement is:
                    #   If dir=+1 (long A, short B):
                    #     return from A = c_a * m_a * lots_a (sold)
                    #     return from B = -c_b * m_b * lots_b (bought back, net cost)
                    #     cash gets: margin_locked + c_a*m_a*lots_a - c_b*m_b*lots_b - costs
                    #   If dir=-1 (short A, long B):
                    #     return from A = -c_a * m_a * lots_a (bought back)
                    #     return from B = c_b * m_b * lots_b (sold)
                    # Actually let me just redo cash management cleanly.

                    # UNDO the above double-counting. Reset cash to pre-exit.
                    # We already did the + and - above, undo:
                    if d == 1:
                        cash -= c_a * m_a * pos['lots_a'] * (1 - COMM)
                        cash += c_b * m_b * pos['lots_b'] * (1 + COMM)
                    else:
                        cash += c_a * m_a * pos['lots_a'] * (1 + COMM)
                        cash -= c_b * m_b * pos['lots_b'] * (1 - COMM)

                    # Correct model: return locked margin + PnL from settlement
                    # The position was entered with total_cost = cost_a + cost_b locked from cash
                    # On exit, we receive:
                    #   Leg A value: +c_a * m_a * lots_a if long (dir=+1), -c_a * m_a * lots_a if short
                    #   Leg B value: -c_b * m_b * lots_b if short (dir=+1), +c_b * m_b * lots_b if long
                    # Plus the locked margin goes back into cash
                    if d == 1:
                        settlement = (c_a * m_a * pos['lots_a'] - c_b * m_b * pos['lots_b'])
                    else:
                        settlement = (-c_a * m_a * pos['lots_a'] + c_b * m_b * pos['lots_b'])
                    settlement -= exit_cost  # exit commissions

                    # cash was reduced by total_cost on entry
                    # now we get back the total_cost + settlement
                    # But entry cost was already taken from cash. So we just add:
                    #   cash += total_cost + settlement
                    # But wait - the total_cost included entry commissions, which we already
                    # accounted for when cash was reduced. The settlement is the net PnL.
                    # So the correct formula is:
                    #   cash was reduced by: cost_a + cost_b = entry_a*m_a*lots_a*(1+COMM) + entry_b*m_b*lots_b*(1+COMM)
                    #   On exit, we should add: entry_a*m_a*lots_a + entry_b*m_b*lots_b + pnl
                    #   where pnl = settlement - entry_val - exit_cost
                    # Simplification: cash += (pos['cash_invested'] + total_pnl)
                    cash += invested + total_pnl

                    trades.append(pnl_pct)
                else:
                    new_positions.append(pos)

            pair_positions = new_positions

            occupied = set()
            for pos in pair_positions:
                occupied.add(pos['si_a'])
                occupied.add(pos['si_b'])

            # --- Sizing ---
            dd_sz = dd_size(pv, high_water, dd_tiers)
            composite = compute_composite(di, daily_eq, high_water)
            regime_mult = regime_lo + composite * (regime_hi - regime_lo)
            pos_size = max(0.05, min(0.99, dd_sz * regime_mult))

            # --- Open new positions ---
            n_can_open = max_pairs - len(pair_positions)
            if n_can_open <= 0:
                continue

            candidates = []
            for si_a, si_b, sym_a, sym_b in pair_set:
                if si_a in occupied or si_b in occupied:
                    continue
                key = (si_a, si_b)
                z_arr = pair_zscore.get(key)
                if z_arr is None:
                    continue
                z_val = z_arr[di] if di < len(z_arr) else np.nan
                if np.isnan(z_val):
                    continue
                if abs(z_val) < z_thresh:
                    continue

                # Correlation filter
                if corr_filter:
                    corr_arr = pair_corr.get(key)
                    if corr_arr is not None:
                        corr_val = corr_arr[di] if di < len(corr_arr) else np.nan
                        if np.isnan(corr_val) or corr_val < min_corr:
                            continue

                # Momentum filter
                if momentum_filter:
                    roc_a = ROC5[si_a, di]
                    roc_b = ROC5[si_b, di]
                    if np.isnan(roc_a) or np.isnan(roc_b):
                        continue
                    # At least one side has meaningful momentum
                    if abs(roc_a) < 1.0 and abs(roc_b) < 1.0:
                        continue
                    # Direction sanity: don't short a strongly rising asset to go long a falling one
                    if z_val > 0:  # short A, long B
                        if roc_a > 3.0 and roc_b < -3.0:
                            continue
                    else:  # long A, short B
                        if roc_a < -3.0 and roc_b > 3.0:
                            continue

                candidates.append((abs(z_val), si_a, si_b, sym_a, sym_b, z_val))

            if not candidates:
                continue
            candidates.sort(key=lambda x: -x[0])

            opened = 0
            for _, si_a, si_b, sym_a, sym_b, z_val in candidates:
                if opened >= n_can_open:
                    break
                if si_a in occupied or si_b in occupied:
                    continue

                c_a = C[si_a, di]
                c_b = C[si_b, di]
                if np.isnan(c_a) or c_a <= 0 or np.isnan(c_b) or c_b <= 0:
                    continue

                m_a = MULT.get(sym_a, DEF_MULT)
                m_b = MULT.get(sym_b, DEF_MULT)

                if z_val > 0:
                    pair_dir = -1  # short A, long B
                else:
                    pair_dir = 1   # long A, short B

                # Capital allocation: use fixed fraction of EQUITY (pv)
                # pos_size is the regime/DD-adjusted sizing (0-1)
                # Each pair gets: pv * pos_size / max_pairs
                # But only use a fraction of that for each leg to leave cash buffer
                pair_cap = pv * pos_size / max(max_pairs, 1)
                cash_per_leg = pair_cap / 2

                lots_a = max(1, int(cash_per_leg / (c_a * m_a * (1 + COMM))))
                lots_b = max(1, int(cash_per_leg / (c_b * m_b * (1 + COMM))))
                cost_a = c_a * m_a * lots_a * (1 + COMM)
                cost_b = c_b * m_b * lots_b * (1 + COMM)
                total_cost = cost_a + cost_b

                if total_cost > cash:
                    scale = cash * 0.85 / total_cost
                    lots_a = max(1, int(lots_a * scale))
                    lots_b = max(1, int(lots_b * scale))
                    cost_a = c_a * m_a * lots_a * (1 + COMM)
                    cost_b = c_b * m_b * lots_b * (1 + COMM)
                    total_cost = cost_a + cost_b
                    if total_cost > cash:
                        continue

                entry_comm = (c_a * m_a * lots_a + c_b * m_b * lots_b) * COMM
                cash -= total_cost
                pair_positions.append({
                    'si_a': si_a, 'si_b': si_b,
                    'sym_a': sym_a, 'sym_b': sym_b,
                    'entry_a': c_a, 'entry_b': c_b,
                    'lots_a': lots_a, 'lots_b': lots_b,
                    'entry_di': di,
                    'entry_z': z_val,
                    'dir': pair_dir,
                    'cash_invested': total_cost,
                    'entry_comm': entry_comm,
                })
                occupied.add(si_a)
                occupied.add(si_b)
                opened += 1

        # Close remaining
        for pos in pair_positions:
            ae = end_di - 1
            c_a = C[pos['si_a'], min(ae, ND - 1)]
            c_b = C[pos['si_b'], min(ae, ND - 1)]
            if np.isnan(c_a) or c_a <= 0:
                c_a = pos['entry_a']
            if np.isnan(c_b) or c_b <= 0:
                c_b = pos['entry_b']

            m_a = MULT.get(pos['sym_a'], DEF_MULT)
            m_b = MULT.get(pos['sym_b'], DEF_MULT)
            d = pos['dir']

            pnl_a = (c_a - pos['entry_a']) * m_a * pos['lots_a'] * d
            pnl_b = (pos['entry_b'] - c_b) * m_b * pos['lots_b'] * d
            exit_cost = (c_a * m_a * pos['lots_a'] + c_b * m_b * pos['lots_b']) * COMM
            total_pnl = pnl_a + pnl_b - exit_cost - pos.get('entry_comm', 0)
            invested = pos['cash_invested']
            pnl_pct = total_pnl / invested * 100 if invested > 0 else 0
            cash += invested + total_pnl
            trades.append(pnl_pct)

        nd = end_di - start_di
        ann = annual_return(cash, CASH0, nd)
        wr = np.mean([1 if t > 0 else 0 for t in trades]) * 100 if trades else 0
        nt = len(trades)
        if daily_eq:
            eq = np.array(daily_eq)
            pk = np.maximum.accumulate(eq)
            mdd = np.min((eq - pk) / pk * 100)
            r = np.diff(eq) / eq[:-1]
            r = np.where(np.isfinite(r), r, 0)
            sh = np.mean(r) / np.std(r) * np.sqrt(252) if np.std(r) > 0 else 0
        else:
            mdd = 0; sh = 0
        return {'ann': ann, 'wr': wr, 'n': nt, 'mdd': mdd, 'sharpe': sh, 'final': cash}

    # ===================== PRINTING HELPERS =====================
    def pr(r, label=""):
        ratio = abs(r['ann'] / r['mdd']) if r['mdd'] != 0 else 0
        print(f"  {label:70s} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | "
              f"Sh={r['sharpe']:5.2f} | R/M={ratio:5.2f} | WR={r['wr']:5.1f}% | N={r['n']:4d}")

    def walk_forward(bt_func, label="", **kwargs):
        res = {}
        for yr in [2020, 2021, 2022, 2023, 2024, 2025]:
            ys = ye = None
            for di in range(ND):
                if dates[di].year == yr and ys is None:
                    ys = di
                if dates[di].year == yr:
                    ye = di + 1
            if ys is None:
                continue
            r = bt_func(start_di=ys, end_di=ye, **kwargs)
            res[yr] = r
        return res

    def print_wf(wf_res, label=""):
        pos = sum(1 for r in wf_res.values() if r['ann'] > 0)
        avg_ann = np.mean([r['ann'] for r in wf_res.values()])
        worst_mdd = min(r['mdd'] for r in wf_res.values())
        avg_rm = np.mean([abs(r['ann'] / r['mdd']) if r['mdd'] != 0 else 0
                          for r in wf_res.values()])
        ws = " | ".join([f"{yr}:{r['ann']:+.0f}%/{r['mdd']:.0f}%"
                         for yr, r in sorted(wf_res.items())])
        print(f"    {label}")
        print(f"      {pos}/6 pos | Avg={avg_ann:>+7.0f}% | AvgR/M={avg_rm:.2f} | WorstWfMDD={worst_mdd:>5.0f}%")
        print(f"      {ws}")

    # ===================== CONFIG =====================
    DD_AGGR100 = [(0, 1.00), (0.10, 0.90), (0.20, 0.70), (0.30, 0.50)]
    all_results = []

    # ===================== TEST 1: BASELINE V177 =====================
    print("\n" + "=" * 140)
    print("  TEST 1: BASELINE V177 (dual-side long+short)")
    print("=" * 140)

    r_base = backtest_v177(atr_norm_max=12.0, max_corr=0.5,
                           dd_tiers=DD_AGGR100, top_n=3, hold=1,
                           short_mode='short_mirror')
    pr(r_base, "V177 BASELINE: short_mirror, atr<12%, corr=0.5, top_n=3")
    all_results.append({**r_base, 'label': 'V177_baseline', 'test': 'BASELINE'})

    # ===================== TEST 2: PAIR_ONLY (z>2) =====================
    print("\n" + "=" * 140)
    print("  TEST 2: PAIR_ONLY (pure pair trading, z>2)")
    print("=" * 140)

    for mp in [1, 2, 3]:
        for mh in [3, 5, 7]:
            for ez in [0.0, 0.3]:
                r = backtest_pair(z_thresh=2.0, max_pairs=mp, max_hold=mh,
                                  momentum_filter=False, exit_z=ez,
                                  dd_tiers=DD_AGGR100)
                label = f"PAIR_ONLY z>2, mp={mp}, h={mh}, ez={ez}"
                pr(r, label)
                all_results.append({**r, 'label': label, 'test': 'PAIR_ONLY'})

    # ===================== TEST 3: PAIR_MOMENTUM (z>2 + V121 filter) =====================
    print("\n" + "=" * 140)
    print("  TEST 3: PAIR_MOMENTUM (z>2 + V121 momentum filter)")
    print("=" * 140)

    for mp in [1, 2, 3]:
        for mh in [3, 5, 7]:
            for ez in [0.0, 0.3]:
                r = backtest_pair(z_thresh=2.0, max_pairs=mp, max_hold=mh,
                                  momentum_filter=True, exit_z=ez,
                                  dd_tiers=DD_AGGR100)
                label = f"PAIR_MOM z>2+mom, mp={mp}, h={mh}, ez={ez}"
                pr(r, label)
                all_results.append({**r, 'label': label, 'test': 'PAIR_MOMENTUM'})

    # ===================== TEST 4: PAIR_LOOSE (z>1.5) =====================
    print("\n" + "=" * 140)
    print("  TEST 4: PAIR_LOOSE (z>1.5, more trades)")
    print("=" * 140)

    for mom in [False, True]:
        for mp in [2, 3]:
            for mh in [3, 5]:
                r = backtest_pair(z_thresh=1.5, max_pairs=mp, max_hold=mh,
                                  momentum_filter=mom, exit_z=0.3,
                                  dd_tiers=DD_AGGR100)
                mom_tag = "+mom" if mom else ""
                label = f"PAIR_LOOSE z>1.5{mom_tag}, mp={mp}, h={mh}"
                pr(r, label)
                all_results.append({**r, 'label': label, 'test': 'PAIR_LOOSE'})

    # ===================== TEST 5: PAIR_TIGHT (z>2.5) =====================
    print("\n" + "=" * 140)
    print("  TEST 5: PAIR_TIGHT (z>2.5, fewer higher quality trades)")
    print("=" * 140)

    for mom in [False, True]:
        for mp in [2, 3]:
            for mh in [5, 7, 10]:
                r = backtest_pair(z_thresh=2.5, max_pairs=mp, max_hold=mh,
                                  momentum_filter=mom, exit_z=0.3,
                                  dd_tiers=DD_AGGR100)
                mom_tag = "+mom" if mom else ""
                label = f"PAIR_TIGHT z>2.5{mom_tag}, mp={mp}, h={mh}"
                pr(r, label)
                all_results.append({**r, 'label': label, 'test': 'PAIR_TIGHT'})

    # ===================== TEST 6: Correlation filter =====================
    print("\n" + "=" * 140)
    print("  TEST 6: PAIR + CORRELATION FILTER (corr > 0.5)")
    print("=" * 140)

    for zt in [1.5, 2.0, 2.5]:
        for mp in [2, 3]:
            r = backtest_pair(z_thresh=zt, max_pairs=mp, max_hold=5,
                              momentum_filter=True, exit_z=0.3,
                              corr_filter=True, min_corr=0.5,
                              dd_tiers=DD_AGGR100)
            label = f"PAIR_CORR z>{zt}+mom+corr>0.5, mp={mp}"
            pr(r, label)
            all_results.append({**r, 'label': label, 'test': 'PAIR_CORR'})

    # ===================== RANKING =====================
    print("\n" + "=" * 140)
    print("  RANKING: Top 30 by annual return")
    print("=" * 140)

    ranked = sorted(all_results, key=lambda x: -x['ann'])

    print(f"\n  {'#':>3s} | {'Test':20s} | {'Label':50s} | {'Ann':>8s} | {'MDD':>6s} | "
          f"{'R/M':>5s} | {'Sh':>5s} | {'WR':>5s} | {'N':>4s}")
    print(f"  {'-' * 130}")
    for i, r in enumerate(ranked[:30]):
        rm = abs(r['ann'] / r['mdd']) if r['mdd'] != 0 else 0
        print(f"  {i + 1:3d} | {r.get('test', '?'):20s} | {r['label']:50s} | "
              f"{r['ann']:+7.0f}% | {r['mdd']:5.0f}% | {rm:5.2f} | "
              f"{r['sharpe']:4.2f} | {r['wr']:4.1f}% | {r['n']:4d}")

    # ===================== BEST PER TEST =====================
    print("\n" + "=" * 140)
    print("  BEST PER TEST GROUP")
    print("=" * 140)

    test_names = sorted(set(r.get('test', '') for r in all_results))
    best_per_test = {}
    for tn in test_names:
        subset = [r for r in all_results if r.get('test') == tn]
        if subset:
            best = max(subset, key=lambda x: x['ann'])
            best_per_test[tn] = best
            rm = abs(best['ann'] / best['mdd']) if best['mdd'] != 0 else 0
            print(f"  {tn:20s}: Ann={best['ann']:+7.0f}%  MDD={best['mdd']:5.0f}%  "
                  f"R/M={rm:5.2f}  Sh={best['sharpe']:4.2f}  N={best['n']:4d}  "
                  f"| {best['label']}")

    # ===================== WALK-FORWARD: TOP 5 + BASELINE =====================
    print("\n" + "=" * 140)
    print("  WALK-FORWARD VALIDATION")
    print("=" * 140)

    # WF baseline
    print(f"\n  V177 Baseline Walk-Forward:")
    wf_base = walk_forward(backtest_v177, label="V177_baseline",
                            atr_norm_max=12.0, max_corr=0.5,
                            dd_tiers=DD_AGGR100, top_n=3, hold=1,
                            short_mode='short_mirror')
    print_wf(wf_base, "V177 baseline")

    # WF top 5 pair configs
    top5_pair = [r for r in ranked if r.get('test') != 'BASELINE'][:5]
    wf_all = {'V177_baseline': wf_base}

    for rank, cfg in enumerate(top5_pair):
        label = cfg['label']
        print(f"\n  Walk-forward #{rank + 1}: {label}")

        # Parse config from label
        z_thresh = 2.0
        momentum_filter = False
        max_pairs = 3
        max_hold = 5
        exit_z = 0.3
        corr_filter = False
        min_corr = 0.3

        # Parse z threshold
        for zt_str, zt_val in [('z>1.5', 1.5), ('z>2.0', 2.0), ('z>2.5', 2.5)]:
            if zt_str in label:
                z_thresh = zt_val
                break

        if '+mom' in label:
            momentum_filter = True
        if 'corr>' in label:
            corr_filter = True
            min_corr = 0.5

        for mp_str, mp_val in [('mp=1', 1), ('mp=2', 2), ('mp=3', 3)]:
            if mp_str in label:
                max_pairs = mp_val
                break

        for mh_str, mh_val in [('h=3', 3), ('h=5', 5), ('h=7', 7), ('h=10', 10)]:
            if mh_str in label:
                max_hold = mh_val
                break

        for ez_str, ez_val in [('ez=0.0', 0.0), ('ez=0.3', 0.3)]:
            if ez_str in label:
                exit_z = ez_val
                break

        wf_res = walk_forward(backtest_pair, label=label,
                               z_thresh=z_thresh, max_pairs=max_pairs,
                               max_hold=max_hold, momentum_filter=momentum_filter,
                               exit_z=exit_z, corr_filter=corr_filter,
                               min_corr=min_corr, dd_tiers=DD_AGGR100)
        print_wf(wf_res, label)
        wf_all[label] = wf_res

    # ===================== WALK-FORWARD AGGREGATE =====================
    print("\n" + "=" * 140)
    print("  WALK-FORWARD AGGREGATE")
    print("=" * 140)

    print(f"\n  {'Config':65s} | {'Pos/6':>5s} | {'Avg Ann':>8s} | {'Med Ann':>8s} | "
          f"{'Avg R/M':>7s} | {'Worst':>8s}")
    print(f"  {'-' * 120}")

    wf_summary = []
    for label, wf_res in wf_all.items():
        anns = [r['ann'] for r in wf_res.values()]
        mdds = [r['mdd'] for r in wf_res.values()]
        n_pos = sum(1 for a in anns if a > 0)
        avg_ann = np.mean(anns)
        med_ann = np.median(anns)
        avg_rm = np.mean([abs(a / m) if m != 0 else 0 for a, m in zip(anns, mdds)])
        worst_mdd = min(mdds)
        wf_summary.append({
            'label': label, 'n_pos': n_pos, 'avg_ann': avg_ann,
            'med_ann': med_ann, 'avg_rm': avg_rm, 'worst_mdd': worst_mdd,
        })
        print(f"  {label:65s} | {n_pos:5d} | {avg_ann:>+7.0f}% | {med_ann:>+7.0f}% | "
              f"{avg_rm:>6.2f} | {worst_mdd:>7.0f}%")

    wf_summary.sort(key=lambda x: -x['avg_ann'])
    if wf_summary:
        best_wf = wf_summary[0]
        print(f"\n  Best WF config: {best_wf['label']}")
        print(f"    Avg Ann={best_wf['avg_ann']:+.0f}%  Med Ann={best_wf['med_ann']:+.0f}%  "
              f"Worst MDD={best_wf['worst_mdd']:.0f}%  Pos/6={best_wf['n_pos']}")

    # ===================== FINAL SUMMARY =====================
    print("\n" + "=" * 140)
    print("  V180 FINAL SUMMARY")
    print("=" * 140)

    base_ann = r_base['ann']
    base_mdd = r_base['mdd']
    base_rm = abs(base_ann / base_mdd) if base_mdd != 0 else 0
    print(f"\n  V177 Baseline: Ann={base_ann:+.0f}%  MDD={base_mdd:.0f}%  R/M={base_rm:.2f}")

    best_pair = None
    for r in ranked:
        if r.get('test') != 'BASELINE':
            best_pair = r
            break
    if best_pair:
        bp_rm = abs(best_pair['ann'] / best_pair['mdd']) if best_pair['mdd'] != 0 else 0
        print(f"  Best Pair:     Ann={best_pair['ann']:+.0f}%  MDD={best_pair['mdd']:.0f}%  "
              f"R/M={bp_rm:.2f}  | {best_pair['label']}")

    if wf_summary:
        best_wf_pair = None
        for ws in wf_summary:
            if ws['label'] != 'V177_baseline':
                best_wf_pair = ws
                break
        if best_wf_pair:
            print(f"  Best WF Pair:  Avg Ann={best_wf_pair['avg_ann']:+.0f}%  "
                  f"Med Ann={best_wf_pair['med_ann']:+.0f}%  "
                  f"Pos/6={best_wf_pair['n_pos']}  "
                  f"| {best_wf_pair['label']}")

    print(f"\n  Test Group Comparison:")
    for tn in test_names:
        best = best_per_test.get(tn)
        if best:
            rm = abs(best['ann'] / best['mdd']) if best['mdd'] != 0 else 0
            delta = rm - base_rm
            print(f"    {tn:20s}: Ann={best['ann']:+7.0f}%  R/M={rm:5.2f}  Delta_R/M={delta:+.2f}")

    print(f"\n  Elapsed: {time.time() - t_start:.0f}s")
    print("=" * 140)


if __name__ == '__main__':
    main()
