"""
Alpha Futures V138 — SUPPLY CHAIN PAIR MOMENTUM + CROSS-COMMODITY SIGNALS
=============================================================================
Tests cross-group momentum and supply chain transmission signals:
  A) Group Leader + Laggard: leader ROC>3% triggers laggard catch-up buy
  B) Group Breadth: 80%+ positive => buy strongest member with Z>1.5
  C) Pair Spread Momentum: spread expanding + strong member ROC>1%
  D) Upstream->Downstream: iron ore ROC>3% => buy rebar if ROC>1%
  E) Whole Chain Confirmation: full chain bullish => strongest member
  F) Cross-Chain Diversification: best chain + best member + Z>1.5
All signals use NEXT-OPEN execution: signal at close di, entry at O[si, di+1].
"""
import sys, os, time, warnings
import numpy as np
import talib
from itertools import combinations
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
        'ni': 1, 'tai': 5, 'hci': 10}
DEF_MULT = 10
COMM = 0.0003

# Supply chain groups — note: 'mfi' appears in energy (methanol) and oilseed (soybean meal)
# 'cfi' appears in oilseed (corn) and soft (cotton). Context-specific.
GROUPS = {
    'steel':      ['rbfi', 'hci', 'ifi', 'jfi', 'jmfi'],
    'base_metal': ['cufi', 'alfi', 'znfi', 'nifi', 'sffi', 'pbfi'],
    'precious':   ['aufi', 'agfi'],
    'energy':     ['scfi', 'mfi', 'fufi', 'bfi', 'egfi', 'pgfi'],
    'oilseed':    ['afi', 'mfi', 'yfi', 'pfi', 'cfi'],
    'soft':       ['srfi', 'cfi', 'apfi', 'cjfi', 'whfi'],
}

# Supply chain relationships for upstream->downstream
# Upstream raw materials -> downstream products (1-2 day lead)
CHAIN_LINKS = {
    'steel': [('ifi', 'rbfi'), ('jfi', 'rbfi'), ('jmfi', 'rbfi'),
              ('ifi', 'hci'), ('jfi', 'hci')],
    'base_metal': [('cufi', 'znfi'), ('cufi', 'alfi')],
}


def annual_return(final, initial, n_days):
    if final <= 0 or initial <= 0 or n_days <= 0: return -100.0
    return (final / initial) ** (1.0 / (n_days / 252)) * 100 - 100


def main():
    print("=" * 120)
    print("  V138 — SUPPLY CHAIN PAIR MOMENTUM + CROSS-COMMODITY SIGNALS")
    print("=" * 120)
    t_start = time.time()
    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)
    print(f"  {NS} commodities, {ND} days")

    # Build sym_to_idx dict for fast lookup
    sym_to_idx = {sym: idx for idx, sym in enumerate(syms)}

    # Build group_idx: group_name -> list of si indices (only those present in data)
    group_idx = {}
    for gname, members in GROUPS.items():
        idxs = []
        for m in members:
            if m in sym_to_idx:
                idxs.append(sym_to_idx[m])
        group_idx[gname] = idxs

    print("\n  Group membership (resolved):")
    for gname, idxs in group_idx.items():
        names = [syms[si] for si in idxs]
        print(f"    {gname:12s}: {names}")

    print("\n[Precompute]...", flush=True)
    t0 = time.time()
    RET = np.full((NS, ND), np.nan)
    ROC5 = np.full((NS, ND), np.nan)
    ROC20 = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si].astype(np.float64)
        for di in range(1, ND):
            if not np.isnan(c[di]) and not np.isnan(c[di-1]) and c[di-1] > 0:
                RET[si, di] = (c[di] / c[di-1] - 1) * 100
        ROC5[si] = talib.ROC(c, timeperiod=5)
        ROC20[si] = talib.ROC(c, timeperiod=20)

    ZSCORE = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(21, ND):
            rets = RET[si, di-20:di]
            v = rets[~np.isnan(rets)]
            if len(v) < 10: continue
            s = np.std(v, ddof=1)
            if s > 0 and not np.isnan(RET[si, di]):
                ZSCORE[si, di] = (RET[si, di] - np.mean(v)) / s

    # Precompute log prices for spread calculation
    LOG_C = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(ND):
            if not np.isnan(C[si, di]) and C[si, di] > 0:
                LOG_C[si, di] = np.log(C[si, di])

    print(f"  Done ({time.time()-t0:.1f}s)")

    # ======================== BACKTEST ========================
    def backtest(signal_func, hold=1, top_n=1, start_di=MIN_TRAIN, end_di=None, desc=""):
        if end_di is None: end_di = ND
        cash = float(CASH0); positions = []; trades = []; daily_eq = []
        for di in range(start_di, end_di - 1):
            pv = cash
            for p in positions:
                cp = C[p['si'], di]
                if not np.isnan(cp) and cp > 0:
                    m = MULT.get(p['sym'], DEF_MULT)
                    pv += cp * m * p['lots'] - cp * m * abs(p['lots']) * COMM
            daily_eq.append(pv)
            cl = []
            for p in positions:
                if di - p['entry_di'] >= p['hold_days']:
                    ep = C[p['si'], di]
                    if np.isnan(ep) or ep <= 0: ep = p['entry_price']
                    m = MULT.get(p['sym'], DEF_MULT)
                    pnl = (ep - p['entry_price']) * m * p['lots']
                    inv = p['entry_price'] * m * abs(p['lots'])
                    pp = pnl / inv * 100 if inv > 0 else 0
                    cash += ep * m * abs(p['lots']) * (1 - COMM)
                    trades.append({'pnl_pct': pp, 'sig': p.get('sig', '')})
                    cl.append(p)
            for p in cl: positions.remove(p)
            if len(positions) >= top_n: continue
            edi = di + 1
            if edi >= end_di: continue
            cands = signal_func(di, edi)
            if not cands: continue
            cands.sort(key=lambda x: -x[0])
            ns = top_n - len(positions)
            cap = cash / max(1, ns)
            for item in cands[:ns]:
                if len(item) == 3: sc, s, pr = item; sig = ''
                else: sc, s, pr, sig = item
                sym = syms[s]; m = MULT.get(sym, DEF_MULT)
                ct = max(1, int(cap * 0.95 / (pr * m * (1 + COMM))))
                ci = pr * m * ct * (1 + COMM)
                if ci > cash:
                    ct = int(cash * 0.9 / (pr * m * (1 + COMM)))
                    ci = pr * m * ct * (1 + COMM) if ct > 0 else 0
                if ct <= 0 or ci <= 0 or ci > cash: continue
                cash -= ci
                positions.append({'si': s, 'entry_price': pr, 'entry_di': edi,
                                  'lots': ct, 'dir': 1, 'sym': sym, 'hold_days': hold, 'sig': sig})
        for p in positions:
            ae = end_di - 1
            ep = C[p['si'], min(ae, ND-1)]
            if np.isnan(ep) or ep <= 0: ep = p['entry_price']
            m = MULT.get(p['sym'], DEF_MULT)
            cash += ep * m * abs(p['lots']) * (1 - COMM)
        nd = end_di - start_di
        ann = annual_return(cash, CASH0, nd)
        wr = np.mean([1 if t['pnl_pct'] > 0 else 0 for t in trades]) * 100 if trades else 0
        nt = len(trades)
        ap = np.mean([t['pnl_pct'] for t in trades]) if trades else 0
        if daily_eq:
            eq = np.array(daily_eq); pk = np.maximum.accumulate(eq)
            mdd = np.min((eq - pk) / pk * 100)
            r = np.diff(eq) / eq[:-1]
            sh = np.mean(r) / np.std(r) * np.sqrt(252) if np.std(r) > 0 else 0
        else: mdd = 0; sh = 0
        sig_breakdown = {}
        for t in trades:
            s = t.get('sig', '?')
            if s not in sig_breakdown: sig_breakdown[s] = {'n': 0, 'w': 0, 'pnl': 0}
            sig_breakdown[s]['n'] += 1
            if t['pnl_pct'] > 0: sig_breakdown[s]['w'] += 1
            sig_breakdown[s]['pnl'] += t['pnl_pct']
        return {'ann': ann, 'wr': wr, 'n': nt, 'avg_pnl': ap, 'mdd': mdd, 'sharpe': sh,
                'desc': desc, 'sig_breakdown': sig_breakdown}

    def pr(r, label=""):
        print(f"  {label:60s} | Ann={r['ann']:+8.1f}% | WR={r['wr']:5.1f}% | "
              f"N={r['n']:4d} | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f}")

    def wf(func, hold=1, topn=1):
        res = {}
        for yr in [2020, 2021, 2022, 2023, 2024, 2025]:
            ys = ye = None
            for di in range(ND):
                if dates[di].year == yr and ys is None: ys = di
                if dates[di].year == yr: ye = di + 1
            if ys: res[yr] = backtest(func, hold=hold, top_n=topn, start_di=ys, end_di=ye)['ann']
        return res

    # ============ SIGNAL FUNCTIONS ============

    # A) Group Leader + Laggard
    # Leader = highest ROC(5) in group; Laggard = lowest ROC(5) but still positive
    # Buy laggard when leader ROC(5) > 3% (catch-up trade)
    # Score: leader_ROC * laggard_ROC * Z-score_laggard
    def sig_leader_laggard(di, edi):
        cands = []
        for gname, idxs in group_idx.items():
            if len(idxs) < 2: continue
            members = []
            for si in idxs:
                roc = ROC5[si, di]
                if np.isnan(roc): continue
                members.append((si, roc))
            if len(members) < 2: continue
            members.sort(key=lambda x: -x[1])
            leader_si, leader_roc = members[0]
            if leader_roc <= 3.0: continue  # leader must have ROC > 3%
            # Find laggard: lowest ROC but still positive
            laggard = None
            for si, roc in reversed(members):
                if roc > 0:
                    laggard = (si, roc)
                    break
            if laggard is None: continue
            lagg_si, lagg_roc = laggard
            if lagg_si == leader_si: continue  # different commodity
            zs = ZSCORE[lagg_si, di]
            if np.isnan(zs): continue
            ep = O[lagg_si, edi]
            if np.isnan(ep) or ep <= 0: continue
            score = leader_roc * lagg_roc * zs
            if score <= 0: continue
            cands.append((score, lagg_si, ep, f'A_leader_laggard_{gname}'))
        return cands

    # B) Group Breadth
    # 80%+ of group members positive ROC(5) => buy strongest with Z > 1.5
    def sig_group_breadth(di, edi):
        cands = []
        for gname, idxs in group_idx.items():
            if len(idxs) < 2: continue
            n_pos = 0; members_data = []
            for si in idxs:
                roc = ROC5[si, di]; zs = ZSCORE[si, di]
                if np.isnan(roc): continue
                if roc > 0: n_pos += 1
                members_data.append((si, roc, zs))
            total = len(members_data)
            if total < 2 or n_pos / total < 0.8: continue
            # Buy the strongest member with Z > 1.5
            for si, roc, zs in sorted(members_data, key=lambda x: -x[1]):
                if roc <= 1.0: continue
                if np.isnan(zs) or zs <= 1.5: continue
                ep = O[si, edi]
                if np.isnan(ep) or ep <= 0: continue
                score = roc * zs * (n_pos / total)
                cands.append((score, si, ep, f'B_breadth_{gname}'))
                break  # one per group
        return cands

    # C) Pair Spread Momentum
    # For each pair within a group: spread = log(C[a]/C[b])
    # When spread ROC(5) > 2% AND stronger member ROC(5) > 1% => buy stronger
    # Score: ROC(5) * Z * spread_ROC
    def sig_pair_spread(di, edi):
        cands = []
        for gname, idxs in group_idx.items():
            if len(idxs) < 2: continue
            # Filter valid members at this di
            valid = []
            for si in idxs:
                roc = ROC5[si, di]
                if np.isnan(roc): continue
                if np.isnan(LOG_C[si, di]): continue
                valid.append(si)
            if len(valid) < 2: continue
            for a_si, b_si in combinations(valid, 2):
                roc_a = ROC5[a_si, di]; roc_b = ROC5[b_si, di]
                if np.isnan(roc_a) or np.isnan(roc_b): continue
                # Compute spread = log(C[a]/C[b])
                spread_now = LOG_C[a_si, di] - LOG_C[b_si, di]
                spread_5ago = LOG_C[a_si, di-5] - LOG_C[b_si, di-5] if di >= 5 else np.nan
                if np.isnan(spread_5ago): continue
                if spread_5ago == 0: continue
                spread_roc = (spread_now - spread_5ago) / abs(spread_5ago) * 100
                if spread_roc <= 2.0: continue  # spread must be expanding
                # Stronger member is the one with higher ROC(5)
                if roc_a >= roc_b:
                    strong_si, strong_roc = a_si, roc_a
                else:
                    strong_si, strong_roc = b_si, roc_b
                if strong_roc <= 1.0: continue  # stronger must have ROC > 1%
                zs = ZSCORE[strong_si, di]
                if np.isnan(zs): zs = 1.0
                ep = O[strong_si, edi]
                if np.isnan(ep) or ep <= 0: continue
                score = strong_roc * max(zs, 0.5) * spread_roc
                if score <= 0: continue
                cands.append((score, strong_si, ep, f'C_pair_spread_{gname}'))
        return cands

    # D) Upstream -> Downstream Transmission
    # Steel chain: ifi(iron ore) leads rbfi(rebar) by 1-2 days
    # If upstream ROC(5) > 3% => buy downstream if downstream ROC(5) > 1%
    def sig_upstream_downstream(di, edi):
        cands = []
        for chain_name, links in CHAIN_LINKS.items():
            for up_sym, down_sym in links:
                if up_sym not in sym_to_idx or down_sym not in sym_to_idx: continue
                up_si = sym_to_idx[up_sym]
                down_si = sym_to_idx[down_sym]
                up_roc = ROC5[up_si, di]
                down_roc = ROC5[down_si, di]
                if np.isnan(up_roc) or np.isnan(down_roc): continue
                if up_roc <= 3.0: continue  # upstream must be surging
                if down_roc <= 1.0: continue  # downstream already positive
                zs = ZSCORE[down_si, di]
                if np.isnan(zs): zs = 1.0
                ep = O[down_si, edi]
                if np.isnan(ep) or ep <= 0: continue
                score = up_roc * down_roc * max(zs, 0.5)
                cands.append((score, down_si, ep, f'D_upstream_{up_sym}->{down_sym}'))
        return cands

    # E) Whole Chain Confirmation
    # Buy when ENTIRE supply chain is bullish: all members have ROC(5) > 1%
    # Score: min(ROC5 across chain) * Z * number_positive
    def sig_whole_chain(di, edi):
        cands = []
        for gname, idxs in group_idx.items():
            if len(idxs) < 3: continue  # need at least 3 members for "chain"
            valid_rocs = []
            for si in idxs:
                roc = ROC5[si, di]
                if np.isnan(roc): continue
                valid_rocs.append((si, roc))
            if len(valid_rocs) < 3: continue
            # All must have ROC > 1%
            all_positive = all(roc > 1.0 for _, roc in valid_rocs)
            if not all_positive: continue
            # Buy the strongest member with best Z
            best = None
            for si, roc in sorted(valid_rocs, key=lambda x: -x[1]):
                zs = ZSCORE[si, di]
                if np.isnan(zs): continue
                ep = O[si, edi]
                if np.isnan(ep) or ep <= 0: continue
                min_roc = min(r for _, r in valid_rocs)
                n_pos = len(valid_rocs)
                score = min_roc * zs * n_pos
                if score <= 0: continue
                best = (score, si, ep, f'E_whole_chain_{gname}')
                break
            if best:
                cands.append(best)
        return cands

    # F) Cross-Chain Diversification
    # Find strongest chain (highest avg ROC(5)), then strongest member with Z > 1.5
    def sig_cross_chain(di, edi):
        cands = []
        best_chain = None
        best_avg_roc = -999
        best_chain_members = []
        for gname, idxs in group_idx.items():
            if len(idxs) < 2: continue
            rocs = []
            members = []
            for si in idxs:
                roc = ROC5[si, di]
                if np.isnan(roc): continue
                rocs.append(roc)
                members.append((si, roc))
            if len(rocs) < 2: continue
            avg_roc = np.mean(rocs)
            if avg_roc > best_avg_roc:
                best_avg_roc = avg_roc
                best_chain = gname
                best_chain_members = members
        if best_chain is None or best_avg_roc <= 1.0: return []
        # Find strongest member with Z > 1.5
        for si, roc in sorted(best_chain_members, key=lambda x: -x[1]):
            zs = ZSCORE[si, di]
            if np.isnan(zs) or zs <= 1.5: continue
            if roc <= 1.0: continue
            ep = O[si, edi]
            if np.isnan(ep) or ep <= 0: continue
            score = roc * zs * best_avg_roc
            if score <= 0: continue
            cands.append((score, si, ep, f'F_cross_chain_{best_chain}'))
            break  # best from best chain
        return cands

    # V121 baseline for comparison
    def sig_v121(di, edi):
        c = []
        for s in range(NS):
            roc = ROC5[s, di]; zs = ZSCORE[s, di]
            if np.isnan(roc) or np.isnan(zs) or roc <= 1.0 or zs <= 1.5: continue
            rp = ROC5[s, di-1] if di > 0 else np.nan
            if not np.isnan(rp) and roc <= rp: continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            c.append((roc * zs, s, ep, 'v121'))
        return c

    # ============ COMBINED SIGNALS ============

    # Union of all supply chain signals, competing by score
    def sig_union_all(di, edi):
        all_sigs = {}
        for item in sig_leader_laggard(di, edi):
            sc, s, ep, sg = item
            if s not in all_sigs or sc > all_sigs[s][0]:
                all_sigs[s] = (sc, s, ep, sg)
        for item in sig_group_breadth(di, edi):
            sc, s, ep, sg = item
            if s not in all_sigs:
                all_sigs[s] = (sc, s, ep, sg)
            else:
                # Sum scores if same commodity
                old = all_sigs[s]
                all_sigs[s] = (old[0] + sc, s, ep, f'{old[3]}+{sg}')
        for item in sig_pair_spread(di, edi):
            sc, s, ep, sg = item
            if s not in all_sigs:
                all_sigs[s] = (sc, s, ep, sg)
            else:
                old = all_sigs[s]
                all_sigs[s] = (old[0] + sc, s, ep, old[3])
        for item in sig_upstream_downstream(di, edi):
            sc, s, ep, sg = item
            if s not in all_sigs:
                all_sigs[s] = (sc, s, ep, sg)
            else:
                old = all_sigs[s]
                all_sigs[s] = (old[0] + sc, s, ep, old[3])
        for item in sig_whole_chain(di, edi):
            sc, s, ep, sg = item
            if s not in all_sigs:
                all_sigs[s] = (sc, s, ep, sg)
            else:
                old = all_sigs[s]
                all_sigs[s] = (old[0] + sc, s, ep, old[3])
        for item in sig_cross_chain(di, edi):
            sc, s, ep, sg = item
            if s not in all_sigs:
                all_sigs[s] = (sc, s, ep, sg)
            else:
                old = all_sigs[s]
                all_sigs[s] = (old[0] + sc, s, ep, old[3])
        return list(all_sigs.values())

    # Cascade: try each signal type in order of strength
    def sig_cascade(di, edi):
        for sig_func in [sig_whole_chain, sig_upstream_downstream, sig_leader_laggard,
                         sig_group_breadth, sig_pair_spread, sig_cross_chain]:
            cands = sig_func(di, edi)
            if cands: return cands
        return []

    # V121 + best chain signal combined
    def sig_v121_plus_chain(di, edi):
        all_sigs = {}
        for item in sig_v121(di, edi):
            sc, s, ep, sg = item
            all_sigs[s] = (sc * 2, s, ep, sg)  # V121 gets 2x weight
        for item in sig_cross_chain(di, edi):
            sc, s, ep, sg = item
            if s not in all_sigs:
                all_sigs[s] = (sc, s, ep, sg)
            else:
                old = all_sigs[s]
                all_sigs[s] = (old[0] + sc, s, ep, f'{old[3]}+{sg}')
        for item in sig_whole_chain(di, edi):
            sc, s, ep, sg = item
            if s not in all_sigs:
                all_sigs[s] = (sc, s, ep, sg)
            else:
                old = all_sigs[s]
                all_sigs[s] = (old[0] + sc * 1.5, s, ep, old[3])
        return list(all_sigs.values())

    # ============ SECTION 1: ALL INDIVIDUAL SIGNALS ============
    print("\n" + "=" * 120)
    print("  SECTION 1: INDIVIDUAL SUPPLY CHAIN SIGNALS")
    print("=" * 120)

    configs = [
        ("V121 baseline (comparison)", sig_v121, 1, 1),
        ("A) Leader + Laggard", sig_leader_laggard, 1, 1),
        ("B) Group Breadth (80%+)", sig_group_breadth, 1, 1),
        ("C) Pair Spread Momentum", sig_pair_spread, 1, 1),
        ("D) Upstream -> Downstream", sig_upstream_downstream, 1, 1),
        ("E) Whole Chain Confirmation", sig_whole_chain, 1, 1),
        ("F) Cross-Chain Diversification", sig_cross_chain, 1, 1),
    ]

    results = {}
    for name, func, hold, topn in configs:
        r = backtest(func, hold=hold, top_n=topn, desc=name)
        results[name] = r
        pr(r, label=name)

    # ============ SECTION 2: COMBINED SIGNALS ============
    print("\n" + "=" * 120)
    print("  SECTION 2: COMBINED / ENSEMBLE SIGNALS")
    print("=" * 120)

    combo_configs = [
        ("Union All SC signals", sig_union_all, 1, 1),
        ("Cascade (E>D>A>B>C>F)", sig_cascade, 1, 1),
        ("V121 + Chain combined", sig_v121_plus_chain, 1, 1),
        ("Union All t=2", sig_union_all, 1, 2),
        ("Cascade t=2", sig_cascade, 1, 2),
        ("V121 + Chain t=2", sig_v121_plus_chain, 1, 2),
        ("Union All t=3", sig_union_all, 1, 3),
        ("V121 + Chain t=2 h=2", sig_v121_plus_chain, 2, 2),
        ("Union All t=2 h=2", sig_union_all, 2, 2),
    ]

    for name, func, hold, topn in combo_configs:
        r = backtest(func, hold=hold, top_n=topn, desc=name)
        results[name] = r
        pr(r, label=name)

    # ============ SECTION 3: TOP_N x HOLD for best configs ============
    print("\n" + "=" * 120)
    print("  SECTION 3: TOP_N x HOLD for top individual signals")
    print("=" * 120)

    best3 = sorted([(n, r) for n, r in results.items()
                     if n in {cn for cn, _, _, _ in configs}],
                    key=lambda x: -x[1]['ann'])[:3]
    func_map = {n: f for n, f, _, _ in configs}
    for name, r in best3:
        func = func_map[name]
        print(f"\n  {name}:")
        for topn in [1, 2]:
            for hold in [1, 2, 3]:
                r = backtest(func, hold=hold, top_n=topn, desc=f"{name} t={topn} h={hold}")
                print(f"    top_n={topn} hold={hold}: Ann={r['ann']:+8.1f}% | "
                      f"WR={r['wr']:5.1f}% | N={r['n']:4d} | MDD={r['mdd']:6.1f}%")

    # ============ SECTION 4: WALK-FORWARD ============
    print("\n" + "=" * 120)
    print("  SECTION 4: WALK-FORWARD 2020-2025")
    print("=" * 120)

    all_configs = configs + combo_configs
    for name, func, hold, topn in all_configs:
        w = wf(func, hold=hold, topn=topn)
        ws = " | ".join([f"{yr}:{v:+.0f}%" for yr, v in sorted(w.items())])
        pos = sum(1 for v in w.values() if v > 0)
        avg = np.mean(list(w.values())) if w else 0
        print(f"  {name:60s} | {pos}/6 | Avg={avg:>+7.0f}% | {ws}")

    # ============ SECTION 5: SIGNAL BREAKDOWN ============
    print("\n" + "=" * 120)
    print("  SECTION 5: SIGNAL TYPE BREAKDOWN")
    print("=" * 120)

    for name in ["Union All SC signals", "Cascade (E>D>A>B>C>F)", "V121 + Chain combined"]:
        if name in results:
            r = results[name]
            bd = r.get('sig_breakdown', {})
            print(f"\n  {name}:")
            for sig, data in sorted(bd.items(), key=lambda x: -x[1]['n']):
                wr = data['w'] / data['n'] * 100 if data['n'] > 0 else 0
                ap = data['pnl'] / data['n'] if data['n'] > 0 else 0
                print(f"    {sig:40s}: N={data['n']:4d} | WR={wr:5.1f}% | AvgPnL={ap:+.2f}%")

    # ============ SUMMARY ============
    print("\n" + "=" * 120)
    print("  SUMMARY: TOP 15 BY ANNUAL RETURN")
    print("=" * 120)

    sorted_r = sorted(results.items(), key=lambda x: -x[1]['ann'])
    for i, (name, r) in enumerate(sorted_r[:15]):
        print(f"  #{i+1}: {name:60s} | Ann={r['ann']:+8.1f}% | WR={r['wr']:5.1f}% | "
              f"N={r['n']:4d} | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f}")

    sorted_sh = sorted(results.items(), key=lambda x: -x[1]['sharpe'])
    print(f"\n  TOP 10 BY SHARPE:")
    for i, (name, r) in enumerate(sorted_sh[:10]):
        print(f"  #{i+1}: {name:60s} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f}")

    print(f"\n  Elapsed: {time.time()-t_start:.0f}s")
    print("=" * 120)


if __name__ == '__main__':
    main()
