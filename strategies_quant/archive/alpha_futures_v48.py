"""
Alpha Futures V48 — Advanced Supply Chain Lead-Lag with Multi-Hop Propagation
==============================================================================
Core idea: V34b's group momentum lag (+86.8%) only looked at 1 hop (commodity
vs its group). But supply chains have multi-hop relationships:
  iron ore -> rebar -> hot coil -> steel products
If iron ore moves today, rebar moves tomorrow, hot coil the day after.
Trade the multi-hop propagation.

Supply chain map:
  scfi (crude) -> mafi (methanol) -> ppfi (PP), vfi (PVC), egfi (EG)
  scfi (crude) -> bfi (bitumen), fufi (fuel oil)
  jmfi (coal) -> jfi (coke) -> rbfi (rebar) -> hcfi (hot coil)
  ifi (iron ore) -> rbfi (rebar) -> hcfi (hot coil)
  afi (soybean) -> mfi (meal), yfi (oil) -> pfi (palm oil)

5 signals:
  1. Direct upstream lag (1-hop) — upstream moved yesterday, self hasn't caught up
  2. 2-hop upstream lag — 2 hops away moved 2 days ago, self hasn't caught up
  3. Full chain momentum — average of entire supply chain weighted by hop distance
  4. Upstream velocity change — upstream is ACCELERATING, about to pull downstream
  5. Combined group + chain — V34b's group lag + chain-specific signal

Configs (~200): signal x lookback x lag_days x threshold x hold x trail
Walk-forward: 2023, 2024.
Print top 20 full-period, top 10 walk-forward, per-chain breakdown.
"""
import sys, os, time, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alpha_v2 import load_all_data, MIN_TRAIN, CASH0

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

GROUP_MAP = {
    'rbfi': 'ferrous', 'hcfi': 'ferrous', 'ifi': 'ferrous', 'jfi': 'ferrous', 'jmfi': 'ferrous',
    'cufi': 'nonferrous', 'alfi': 'nonferrous', 'znfi': 'nonferrous', 'nifi': 'nonferrous',
    'afi': 'oils', 'mfi': 'oils', 'yfi': 'oils', 'pfi': 'oils', 'cfi': 'oils',
    'scfi': 'energy', 'mafi': 'energy', 'bfi': 'energy', 'fufi': 'energy',
    'ppfi': 'chemical', 'vfi': 'chemical', 'egfi': 'chemical', 'pgfi': 'chemical',
}

# Direct upstream links (1-hop)
UPSTREAM_1 = {
    'rbfi': 'ifi', 'hcfi': 'rbfi', 'jfi': 'jmfi',
    'mafi': 'scfi', 'bfi': 'scfi', 'fufi': 'scfi',
    'mfi': 'afi', 'yfi': 'afi', 'pfi': 'yfi',
    'ppfi': 'mafi', 'vfi': 'mafi', 'egfi': 'mafi',
}

# 2-hop upstream links
UPSTREAM_2 = {
    'hcfi': 'ifi',      # hcfi <- rbfi <- ifi
    'ppfi': 'scfi',     # ppfi <- mafi <- scfi
    'vfi': 'scfi',      # vfi <- mafi <- scfi
    'egfi': 'scfi',     # egfi <- mafi <- scfi
    'pfi': 'afi',       # pfi <- yfi <- afi
}

# Full supply chain paths (for chain momentum signal)
# Each path: list of (symbol, hop_distance_from_head)
CHAIN_PATHS = {
    'crude_chem': ['scfi', 'mafi', 'ppfi'],
    'crude_chem_v': ['scfi', 'mafi', 'vfi'],
    'crude_chem_eg': ['scfi', 'mafi', 'egfi'],
    'crude_bitumen': ['scfi', 'bfi'],
    'crude_fuel': ['scfi', 'fufi'],
    'coal_steel': ['jmfi', 'jfi', 'rbfi', 'hcfi'],
    'ore_steel': ['ifi', 'rbfi', 'hcfi'],
    'soy_crush': ['afi', 'mfi'],
    'soy_oil': ['afi', 'yfi', 'pfi'],
}

# Reverse map: symbol -> list of chain names it belongs to
SYM_CHAINS = {}
for chain_name, chain_syms in CHAIN_PATHS.items():
    for sym in chain_syms:
        if sym not in SYM_CHAINS:
            SYM_CHAINS[sym] = []
        SYM_CHAINS[sym].append(chain_name)

# Chain name for display per upstream relationship
CHAIN_LABELS = {
    'rbfi': 'ore->rebar', 'hcfi': 'ore->rebar->hc', 'jfi': 'coal->coke',
    'mafi': 'crude->methanol', 'bfi': 'crude->bitumen', 'fufi': 'crude->fuel',
    'mfi': 'bean->meal', 'yfi': 'bean->oil', 'pfi': 'bean->oil->palm',
    'ppfi': 'crude->methanol->PP', 'vfi': 'crude->methanol->PVC',
    'egfi': 'crude->methanol->EG',
}


def main():
    t_start = time.time()
    print("=" * 120)
    print("Alpha Futures V48 — Advanced Supply Chain Lead-Lag with Multi-Hop Propagation")
    print("Core: multi-hop supply chain momentum propagation. Upstream moves first,")
    print("      downstream catches up with 1-2 day lag per hop.")
    print("=" * 120)

    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)

    sym_to_si = {syms[si]: si for si in range(NS)}

    # Build group membership
    group_members = {}
    for si in range(NS):
        grp = GROUP_MAP.get(syms[si])
        if grp is None:
            continue
        if grp not in group_members:
            group_members[grp] = []
        group_members[grp].append(si)

    # Build 1-hop upstream index
    upstream1_si = {}
    for si in range(NS):
        up_sym = UPSTREAM_1.get(syms[si])
        if up_sym and up_sym in sym_to_si:
            upstream1_si[si] = sym_to_si[up_sym]
        else:
            upstream1_si[si] = -1

    # Build 2-hop upstream index
    upstream2_si = {}
    for si in range(NS):
        up_sym = UPSTREAM_2.get(syms[si])
        if up_sym and up_sym in sym_to_si:
            upstream2_si[si] = sym_to_si[up_sym]
        else:
            upstream2_si[si] = -1

    # Build chain paths as si indices with hop offsets
    chain_si_paths = {}
    for chain_name, chain_syms in CHAIN_PATHS.items():
        path = []
        for hop, sym in enumerate(chain_syms):
            if sym in sym_to_si:
                path.append((sym_to_si[sym], hop, sym))
        if len(path) >= 2:
            chain_si_paths[chain_name] = path

    # Which chains each si belongs to
    si_chains = {}
    for si in range(NS):
        chains = SYM_CHAINS.get(syms[si], [])
        if chains:
            si_chains[si] = [(cn, chain_si_paths[cn]) for cn in chains if cn in chain_si_paths]

    n_with_up1 = sum(1 for v in upstream1_si.values() if v >= 0)
    n_with_up2 = sum(1 for v in upstream2_si.values() if v >= 0)
    n_with_chain = sum(1 for si in range(NS) if si in si_chains)
    print(f"  {NS} commodities, {ND} days, {len(group_members)} groups")
    print(f"  1-hop upstream: {n_with_up1} commodities, 2-hop: {n_with_up2}, chains: {n_with_chain}")

    # ========================================
    # PRECOMPUTE MOMENTUM AT ALL LOOKBACKS
    # ========================================
    print("\n[Signals] Computing momentum, group momentum, ATR...", flush=True)
    t0 = time.time()

    mom = {}
    for lag in [3, 5, 7]:
        m = np.full((NS, ND), np.nan)
        for si in range(NS):
            for di in range(lag, ND):
                c_now = C[si, di]
                c_prev = C[si, di - lag]
                if not np.isnan(c_now) and not np.isnan(c_prev) and c_prev > 0:
                    m[si, di] = (c_now - c_prev) / c_prev
        mom[lag] = m

    # Group momentum (excluding self)
    grp_mom = {}
    for lag in [3, 5, 7]:
        gm = np.full((NS, ND), np.nan)
        for grp, members in group_members.items():
            for di in range(lag, ND):
                for sj in members:
                    ms = []
                    for sk in members:
                        if sk == sj:
                            continue
                        m = mom[lag][sk, di]
                        if not np.isnan(m):
                            ms.append(m)
                    if ms:
                        gm[sj, di] = np.mean(ms)
        grp_mom[lag] = gm

    # ATR for trailing stops
    atr10 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(11, ND):
            trs = []
            for dd in range(di - 10, di):
                hi, lo, pc = H[si, dd], L[si, dd], C[si, dd - 1]
                if np.isnan(hi) or np.isnan(lo):
                    continue
                tr = hi - lo
                if not np.isnan(pc):
                    tr = max(tr, abs(hi - pc), abs(lo - pc))
                trs.append(tr)
            if trs:
                atr10[si, di] = np.mean(trs)

    print(f"  Done ({time.time() - t0:.1f}s)", flush=True)

    # ========================================
    # SCORING FUNCTIONS (5 signals)
    # ========================================

    def make_signal1_direct_upstream(lookback=5, lag_days=1, threshold=0.005, scale=10.0):
        """Signal 1: Direct upstream lag (1-hop).
        Upstream moved {lag_days} ago, self hasn't caught up yet.
        LONG when divergence = upstream_lagged_mom - own_mom > threshold.
        """
        def score(si, di):
            usi = upstream1_si.get(si, -1)
            if usi < 0:
                return np.nan
            own = mom[lookback][si, di]
            if np.isnan(own):
                return np.nan
            src_di = di - lag_days
            if src_di < lookback:
                return np.nan
            up_m = mom[lookback][usi, src_di]
            if np.isnan(up_m):
                return np.nan
            divergence = up_m - own
            if divergence < threshold:
                return np.nan
            return np.clip(divergence * scale, 0, 1)
        return score

    def make_signal2_twohop_upstream(lookback=5, lag_days=2, threshold=0.005, scale=10.0):
        """Signal 2: 2-hop upstream lag.
        2-hop upstream moved {lag_days} ago, self hasn't caught up.
        E.g. iron ore moved 2 days ago -> hot coil should catch up now.
        """
        def score(si, di):
            u2si = upstream2_si.get(si, -1)
            if u2si < 0:
                return np.nan
            own = mom[lookback][si, di]
            if np.isnan(own):
                return np.nan
            src_di = di - lag_days
            if src_di < lookback:
                return np.nan
            up2_m = mom[lookback][u2si, src_di]
            if np.isnan(up2_m):
                return np.nan
            divergence = up2_m - own
            if divergence < threshold:
                return np.nan
            return np.clip(divergence * scale, 0, 1)
        return score

    def make_signal3_chain_momentum(lookback=5, threshold=0.005, scale=8.0):
        """Signal 3: Full chain momentum.
        Average momentum of entire supply chain (all hops), weighted by recency.
        If chain is trending up but current commodity hasn't moved -> LONG.
        Each hop's momentum is lagged by its hop distance from self.
        """
        def score(si, di):
            chains = si_chains.get(si)
            if not chains:
                return np.nan
            own = mom[lookback][si, di]
            if np.isnan(own):
                return np.nan

            best_div = -999
            for chain_name, path in chains:
                # Find own position in this chain
                own_hop = None
                for psi, hop, psym in path:
                    if psi == si:
                        own_hop = hop
                        break
                if own_hop is None:
                    continue

                # Collect upstream momentums, each lagged by hop distance
                chain_moms = []
                for psi, hop, psym in path:
                    # Lag by the number of hops from self to this node
                    dist = own_hop - hop
                    if dist <= 0:
                        continue  # self or downstream
                    src_di = di - dist
                    if src_di < lookback:
                        continue
                    m = mom[lookback][psi, src_di]
                    if np.isnan(m):
                        continue
                    # Weight by inverse distance (closer hops matter more)
                    weight = 1.0 / dist
                    chain_moms.append((m, weight))

                if not chain_moms:
                    continue
                total_w = sum(w for _, w in chain_moms)
                if total_w <= 0:
                    continue
                weighted_avg = sum(m * w for m, w in chain_moms) / total_w
                div = weighted_avg - own
                if div > best_div:
                    best_div = div

            if best_div < threshold:
                return np.nan
            return np.clip(best_div * scale, 0, 1)
        return score

    def make_signal4_upstream_velocity(lookback=5, scale=10.0):
        """Signal 4: Upstream velocity change.
        Not just upstream level, but whether upstream is ACCELERATING.
        up_vel = mom3[upstream] - mom5[upstream] (short-term vs mid-term).
        If up_vel > 0 AND mom5[upstream] > 0: upstream accelerating upward,
        downstream about to follow.
        """
        short_lb = max(lookback - 2, 3)
        def score(si, di):
            usi = upstream1_si.get(si, -1)
            if usi < 0:
                return np.nan
            m_short = mom[short_lb][usi, di]
            m_mid = mom[lookback][usi, di]
            if np.isnan(m_short) or np.isnan(m_mid):
                return np.nan
            # Must be positive direction
            if m_mid <= 0:
                return np.nan
            up_vel = m_short - m_mid
            if up_vel <= 0:
                return np.nan
            # Score = velocity * level
            raw = up_vel * m_mid * scale * 100
            return np.clip(raw, 0, 1)
        return score

    def make_signal5_combined(lookback=5, lag_days=1, threshold=0.003, scale=10.0):
        """Signal 5: Combined group + chain.
        Average of V34b's group lag signal and chain-specific upstream signal.
        0.5 * group_score + 0.5 * chain_score.
        """
        def score(si, di):
            own = mom[lookback][si, di]
            if np.isnan(own):
                return np.nan

            # Group component
            grp = grp_mom[lookback][si, di]
            if np.isnan(grp):
                return np.nan
            group_div = grp - own

            # Chain component
            usi = upstream1_si.get(si, -1)
            if usi >= 0:
                src_di = di - lag_days
                if src_di < lookback:
                    return np.nan
                up_m = mom[lookback][usi, src_di]
                if np.isnan(up_m):
                    chain_div = group_div  # fallback to group only
                else:
                    chain_div = up_m - own
            else:
                chain_div = group_div  # no upstream, use group

            combined = 0.5 * group_div + 0.5 * chain_div
            if combined < threshold:
                return np.nan
            return np.clip(combined * scale, 0, 1)
        return score

    # ========================================
    # BACKTEST ENGINE (same as V34b/V45)
    # ========================================
    def run_backtest(score_fn, name, top_n=1, hold_min=2, hold_max=3,
                     trail_atr_mult=2.5, wf_split_year=None):
        """Single position, long only, lots = cash/(price*mult)."""
        cash = float(CASH0)
        trades = []
        positions = []

        for di in range(MIN_TRAIN, ND):
            year = dates[di].year
            if wf_split_year is not None and year < wf_split_year:
                continue

            # Manage existing positions
            new_positions = []
            for pos in positions:
                c = C[pos['si'], di]
                if np.isnan(c) or c <= 0:
                    c = pos['entry']
                mult = MULT.get(pos['sym'], DEF_MULT)
                mkt_val = c * mult * pos['lots']
                pnl = (c - pos['entry']) * mult * pos['lots'] * pos['dir']
                pnl_pct = pnl / (pos['entry'] * mult * pos['lots']) * 100 if pos['entry'] > 0 else 0
                days_held = di - pos['entry_di']

                exit_reason = None

                # Trailing stop
                if trail_atr_mult > 0 and days_held >= 2:
                    atr = pos.get('atr', 0)
                    if atr > 0 and pos['dir'] == 1:
                        new_trail = c - trail_atr_mult * atr
                        if new_trail > pos.get('trail_price', pos['entry']):
                            pos['trail_price'] = new_trail
                        if c < pos['trail_price']:
                            exit_reason = 'trail'

                # Signal flip (after min hold)
                if exit_reason is None and days_held >= hold_min:
                    cur_score = score_fn(pos['si'], di)
                    if not np.isnan(cur_score) and cur_score < -0.01:
                        exit_reason = 'signal_flip'

                # Time exit
                if exit_reason is None and days_held >= hold_max:
                    exit_reason = 'time'

                if exit_reason:
                    cost_out = mkt_val * COMM
                    cash += mkt_val - cost_out
                    trades.append({
                        'pnl_pct': pnl_pct, 'pnl_abs': pnl,
                        'days': days_held, 'di': di, 'year': year,
                        'sym': pos['sym'], 'dir': pos['dir'], 'reason': exit_reason,
                    })
                else:
                    new_positions.append(pos)

            positions = new_positions

            # Open new positions
            n_open = len(positions)
            if n_open < top_n:
                slots = top_n - n_open
                scored = []
                for si in range(NS):
                    sc = score_fn(si, di)
                    if np.isnan(sc) or sc <= 0.01:
                        continue
                    sym = syms[si]
                    if any(p['sym'] == sym for p in positions):
                        continue
                    scored.append((si, sc, sym))

                if scored:
                    scored.sort(key=lambda x: -x[1])
                    cash_per_slot = cash / slots if slots > 0 else cash

                    for best_si, best_sc, best_sym in scored[:slots]:
                        c = C[best_si, di]
                        if np.isnan(c) or c <= 0:
                            continue
                        mult = MULT.get(best_sym, DEF_MULT)
                        notional = c * mult
                        if notional <= 0:
                            continue
                        lots = int(cash_per_slot / (notional * (1 + COMM)))
                        if lots <= 0:
                            continue
                        cost_in = notional * lots * (1 + COMM)
                        if cost_in > cash:
                            lots = int(cash / (notional * (1 + COMM)))
                            if lots <= 0:
                                continue
                            cost_in = notional * lots * (1 + COMM)

                        atr_val = atr10[best_si, di] if not np.isnan(atr10[best_si, di]) else 0
                        cash -= cost_in
                        trail_price = c - trail_atr_mult * atr_val
                        positions.append({
                            'si': best_si, 'entry': c, 'entry_di': di,
                            'lots': lots, 'dir': 1, 'sym': best_sym,
                            'atr': atr_val, 'trail_price': trail_price,
                        })

        # Close remaining
        for pos in positions:
            c = C[pos['si'], ND - 1]
            if np.isnan(c) or c <= 0:
                c = pos['entry']
            mult = MULT.get(pos['sym'], DEF_MULT)
            pnl = (c - pos['entry']) * mult * pos['lots'] * pos['dir']
            cash += c * mult * pos['lots'] * (1 - COMM)
            trades.append({
                'pnl_pct': pnl / (pos['entry'] * mult * pos['lots']) * 100,
                'pnl_abs': pnl, 'days': ND - 1 - pos['entry_di'],
                'di': ND - 1, 'year': dates[ND - 1].year,
                'sym': pos['sym'], 'dir': pos['dir'], 'reason': 'end',
            })

        if len(trades) < 10:
            return None

        # Stats
        equity = float(CASH0)
        peak = float(CASH0)
        max_dd = 0
        for t in sorted(trades, key=lambda x: x['di']):
            equity += t['pnl_abs']
            if equity > peak:
                peak = equity
            if peak > 0:
                dd = (peak - equity) / peak * 100
                if dd > max_dd:
                    max_dd = dd

        days_total = (dates[ND - 1] - dates[MIN_TRAIN]).days
        yr = max(days_total / 365.25, 0.01)
        if wf_split_year:
            first_test_di = None
            for d in range(MIN_TRAIN, ND):
                if dates[d].year >= wf_split_year:
                    first_test_di = d
                    break
            if first_test_di:
                days_total = (dates[ND - 1] - dates[first_test_di]).days
                yr = max(days_total / 365.25, 0.01)

        ann = ((cash / CASH0) ** (1 / yr) - 1) * 100
        nw = sum(1 for t in trades if t['pnl_abs'] > 0)
        wr = nw / len(trades) * 100
        avg_win = np.mean([t['pnl_pct'] for t in trades if t['pnl_abs'] > 0]) if nw > 0 else 0
        avg_loss = np.mean([abs(t['pnl_pct']) for t in trades if t['pnl_abs'] <= 0]) if nw < len(trades) else 0
        avg_days = np.mean([t['days'] for t in trades])
        pf = (sum(t['pnl_abs'] for t in trades if t['pnl_abs'] > 0) /
              max(abs(sum(t['pnl_abs'] for t in trades if t['pnl_abs'] < 0)), 1))

        reasons = {}
        for t in trades:
            r = t['reason']
            if r not in reasons:
                reasons[r] = {'n': 0, 'w': 0, 'pnl': 0.0}
            reasons[r]['n'] += 1
            if t['pnl_abs'] > 0:
                reasons[r]['w'] += 1
            reasons[r]['pnl'] += t['pnl_pct']

        year_stats = {}
        for t in trades:
            y = t['year']
            if y not in year_stats:
                year_stats[y] = {'n': 0, 'w': 0, 'pnl': 0.0}
            year_stats[y]['n'] += 1
            if t['pnl_abs'] > 0:
                year_stats[y]['w'] += 1
            year_stats[y]['pnl'] += t['pnl_pct']

        # Chain breakdown (using UPSTREAM_1 relationship)
        chain_counts = {}
        for t in trades:
            chain_lbl = CHAIN_LABELS.get(t['sym'], 'other')
            if chain_lbl not in chain_counts:
                chain_counts[chain_lbl] = {'n': 0, 'w': 0, 'pnl': 0.0}
            chain_counts[chain_lbl]['n'] += 1
            if t['pnl_abs'] > 0:
                chain_counts[chain_lbl]['w'] += 1
            chain_counts[chain_lbl]['pnl'] += t['pnl_abs']

        # Group breakdown
        grp_counts = {}
        for t in trades:
            g = GROUP_MAP.get(t['sym'], 'other')
            if g not in grp_counts:
                grp_counts[g] = {'n': 0, 'w': 0, 'pnl': 0.0}
            grp_counts[g]['n'] += 1
            if t['pnl_abs'] > 0:
                grp_counts[g]['w'] += 1
            grp_counts[g]['pnl'] += t['pnl_abs']

        return {
            'name': name, 'ann': round(ann, 1), 'n': len(trades),
            'wr': round(wr, 1), 'dd': round(max_dd, 1),
            'avg_win': round(avg_win, 2), 'avg_loss': round(avg_loss, 2),
            'avg_days': round(avg_days, 1), 'pf': round(pf, 2),
            'cash': round(cash, 0),
            'reasons': reasons, 'yearly': year_stats,
            'grp_counts': grp_counts, 'chain_counts': chain_counts,
        }

    # ========================================
    # PARAMETER SWEEP
    # ========================================
    print("\n[Backtest] Building configurations...", flush=True)
    configs = []

    # Signal 1: Direct upstream lag (1-hop)
    for lookback in [3, 5, 7]:
        for lag_days in [1, 2, 3]:
            for threshold in [0.003, 0.005, 0.01]:
                for hold in [3, 5]:
                    for trail in [2.5, 3.0]:
                        configs.append((
                            make_signal1_direct_upstream(lookback=lookback, lag_days=lag_days,
                                                         threshold=threshold),
                            f"S1_LB{lookback}_LAG{lag_days}_TH{threshold*1000:.0f}_H{hold}_TR{trail*10:.0f}",
                            1, 2, hold, trail, None
                        ))

    # Signal 2: 2-hop upstream lag
    for lookback in [3, 5, 7]:
        for lag_days in [2, 3]:
            for threshold in [0.003, 0.005, 0.01]:
                for hold in [3, 5]:
                    for trail in [2.5, 3.0]:
                        configs.append((
                            make_signal2_twohop_upstream(lookback=lookback, lag_days=lag_days,
                                                         threshold=threshold),
                            f"S2_LB{lookback}_LAG{lag_days}_TH{threshold*1000:.0f}_H{hold}_TR{trail*10:.0f}",
                            1, 2, hold, trail, None
                        ))

    # Signal 3: Full chain momentum
    for lookback in [3, 5, 7]:
        for threshold in [0.003, 0.005, 0.01]:
            for hold in [3, 5]:
                for trail in [2.5, 3.0]:
                    configs.append((
                        make_signal3_chain_momentum(lookback=lookback, threshold=threshold),
                        f"S3_LB{lookback}_TH{threshold*1000:.0f}_H{hold}_TR{trail*10:.0f}",
                        1, 2, hold, trail, None
                    ))

    # Signal 4: Upstream velocity change
    for lookback in [5, 7]:
        for hold in [3, 5]:
            for trail in [2.5, 3.0]:
                configs.append((
                    make_signal4_upstream_velocity(lookback=lookback),
                    f"S4_LB{lookback}_H{hold}_TR{trail*10:.0f}",
                    1, 2, hold, trail, None
                ))

    # Signal 5: Combined group + chain
    for lookback in [3, 5, 7]:
        for lag_days in [1, 2]:
            for threshold in [0.003, 0.005]:
                for hold in [3, 5]:
                    for trail in [2.5, 3.0]:
                        configs.append((
                            make_signal5_combined(lookback=lookback, lag_days=lag_days,
                                                  threshold=threshold),
                            f"S5_LB{lookback}_LAG{lag_days}_TH{threshold*1000:.0f}_H{hold}_TR{trail*10:.0f}",
                            1, 2, hold, trail, None
                        ))

    # Walk-forward configs for best parameter combos
    wf_configs = []
    for sig in [1, 2, 3, 4, 5]:
        for wf_year in [2023, 2024]:
            if sig == 1:
                for lookback in [3, 5]:
                    for lag_days in [1, 2]:
                        for hold in [3, 5]:
                            for trail in [2.5, 3.0]:
                                wf_configs.append((
                                    make_signal1_direct_upstream(lookback=lookback, lag_days=lag_days,
                                                                 threshold=0.005),
                                    f"S1_LB{lookback}_LAG{lag_days}_TH5_H{hold}_TR{trail*10:.0f}_WF{wf_year}",
                                    1, 2, hold, trail, wf_year
                                ))
            elif sig == 2:
                for lookback in [3, 5]:
                    for lag_days in [2, 3]:
                        for hold in [3, 5]:
                            wf_configs.append((
                                make_signal2_twohop_upstream(lookback=lookback, lag_days=lag_days,
                                                             threshold=0.005),
                                f"S2_LB{lookback}_LAG{lag_days}_TH5_H{hold}_TR25_WF{wf_year}",
                                1, 2, hold, 2.5, wf_year
                            ))
            elif sig == 3:
                for lookback in [3, 5]:
                    for hold in [3, 5]:
                        wf_configs.append((
                            make_signal3_chain_momentum(lookback=lookback, threshold=0.005),
                            f"S3_LB{lookback}_TH5_H{hold}_TR25_WF{wf_year}",
                            1, 2, hold, 2.5, wf_year
                        ))
            elif sig == 4:
                for lookback in [5]:
                    for hold in [3, 5]:
                        wf_configs.append((
                            make_signal4_upstream_velocity(lookback=lookback),
                            f"S4_LB{lookback}_H{hold}_TR25_WF{wf_year}",
                            1, 2, hold, 2.5, wf_year
                        ))
            elif sig == 5:
                for lookback in [3, 5]:
                    for lag_days in [1, 2]:
                        for hold in [3, 5]:
                            wf_configs.append((
                                make_signal5_combined(lookback=lookback, lag_days=lag_days,
                                                      threshold=0.003),
                                f"S5_LB{lookback}_LAG{lag_days}_TH3_H{hold}_TR25_WF{wf_year}",
                                1, 2, hold, 2.5, wf_year
                            ))

    all_configs = configs + wf_configs
    print(f"  {len(configs)} full-period + {len(wf_configs)} walk-forward = {len(all_configs)} total", flush=True)

    # ========================================
    # RUN BACKTESTS
    # ========================================
    print("\n[Backtest] Running...", flush=True)
    results = []
    t_backtest_start = time.time()

    for ci, (fn, name, tn, hmin, hmax, trail, wf) in enumerate(all_configs):
        r = run_backtest(fn, name, top_n=tn, hold_min=hmin, hold_max=hmax,
                         trail_atr_mult=trail, wf_split_year=wf)
        if r and r['ann'] > 0:
            results.append(r)
            if r['ann'] > 50:
                parts = []
                for reason, stats in sorted(r['reasons'].items()):
                    wr_r = stats['w'] / stats['n'] * 100 if stats['n'] > 0 else 0
                    parts.append(f"{reason}:{stats['n']}({wr_r:.0f}%)")
                print(f"  {r['name']:55s} | Ann {r['ann']:+7.1f}% | WR {r['wr']:5.1f}% | "
                      f"N {r['n']:4d} | DD {r['dd']:6.1f}% | PF {r['pf']:4.2f} | "
                      f"AvgW {r['avg_win']:+.2f}% | AvgL {r['avg_loss']:.2f}% | AvgD {r['avg_days']:.1f}")
                print(f"  {'':55s} | Exits: {' | '.join(parts)}")

        if (ci + 1) % 50 == 0:
            elapsed = time.time() - t_backtest_start
            rate = (ci + 1) / elapsed
            eta = (len(all_configs) - ci - 1) / rate
            print(f"  [{ci + 1}/{len(all_configs)}] {len(results)} profitable "
                  f"({elapsed:.0f}s elapsed, ETA {eta:.0f}s)", flush=True)

    print(f"  Backtests done ({time.time() - t_backtest_start:.1f}s)", flush=True)

    # ========================================
    # RESULTS
    # ========================================
    results.sort(key=lambda x: -x['ann'])

    wf_results = [r for r in results if '_WF' in r['name']]
    full_results = [r for r in results if '_WF' not in r['name']]

    # --- Top 20 full-period ---
    print(f"\n{'=' * 130}")
    print(f"  TOP 20 FULL-PERIOD RESULTS")
    print(f"{'=' * 130}")
    print(f"  {'Strategy':55s} | {'Ann':>7s} | {'WR':>5s} | {'N':>4s} | {'DD':>6s} | "
          f"{'PF':>4s} | {'AvgW':>7s} | {'AvgL':>6s} | {'AvgD':>4s}")
    print(f"  {'-' * 130}")
    for r in full_results[:20]:
        print(f"  {r['name']:55s} | {r['ann']:+7.1f}% | {r['wr']:5.1f}% | "
              f"{r['n']:4d} | {r['dd']:6.1f}% | {r['pf']:4.2f} | "
              f"{r['avg_win']:+6.2f}% | {r['avg_loss']:5.2f}% | "
              f"{r['avg_days']:4.1f}")

    # --- Top 10 walk-forward ---
    if wf_results:
        print(f"\n  TOP 10 WALK-FORWARD RESULTS (out-of-sample)")
        print(f"  {'-' * 130}")
        wf_results.sort(key=lambda x: -x['ann'])
        for r in wf_results[:10]:
            print(f"  {r['name']:55s} | {r['ann']:+7.1f}% | {r['wr']:5.1f}% | "
                  f"{r['n']:4d} | {r['dd']:6.1f}% | {r['pf']:4.2f}")

    # --- Best per signal ---
    print(f"\n  BEST PER SIGNAL:")
    for sig_num in [1, 2, 3, 4, 5]:
        sig_results = [r for r in full_results if r['name'].startswith(f'S{sig_num}_')]
        if sig_results:
            best = sig_results[0]
            print(f"    Signal {sig_num}: {best['name']:55s} | "
                  f"Ann {best['ann']:+7.1f}% | WR {best['wr']:5.1f}% | "
                  f"N {best['n']:4d} | DD {best['dd']:6.1f}% | PF {best['pf']:4.2f}")

    # --- Per-chain breakdown for best config ---
    if full_results:
        best = full_results[0]
        print(f"\n{'=' * 130}")
        print(f"  BEST CONFIG DETAIL")
        print(f"  {best['name']}")
        print(f"  Ann={best['ann']:+.1f}%  WR={best['wr']:.1f}%  N={best['n']}  "
              f"DD={best['dd']:.1f}%  PF={best['pf']:.2f}")
        print(f"  AvgWin={best['avg_win']:+.2f}%  AvgLoss={best['avg_loss']:.2f}%  "
              f"AvgDays={best['avg_days']:.1f}  Final={best['cash']:.0f}")
        print(f"{'=' * 130}")

        print(f"\n  EXIT REASON BREAKDOWN:")
        for reason, s in sorted(best['reasons'].items(), key=lambda x: -x[1]['n']):
            rwr = s['w'] / max(s['n'], 1) * 100
            print(f"    {reason:12s}: {s['n']:4d} trades  WR={rwr:5.1f}%  PnL={s['pnl']:+.1f}%")

        print(f"\n  YEARLY BREAKDOWN:")
        for y in sorted(best['yearly'].keys()):
            s = best['yearly'][y]
            wr_y = s['w'] / max(s['n'], 1) * 100
            print(f"    {y}: {s['n']:3d} trades  WR={wr_y:5.1f}%  PnL={s['pnl']:+.1f}%")

        print(f"\n  SUPPLY CHAIN BREAKDOWN:")
        if best.get('chain_counts'):
            for ch in sorted(best['chain_counts'].keys(),
                             key=lambda x: -best['chain_counts'][x]['n']):
                cs = best['chain_counts'][ch]
                wr_c = cs['w'] / max(cs['n'], 1) * 100
                print(f"    {ch:30s}: {cs['n']:3d}t  WR={wr_c:5.1f}%  Abs={cs['pnl']:+.0f}")

        print(f"\n  GROUP BREAKDOWN:")
        for g in sorted(best['grp_counts'].keys(), key=lambda x: -best['grp_counts'][x]['n']):
            gs = best['grp_counts'][g]
            wr_g = gs['w'] / max(gs['n'], 1) * 100
            print(f"    {g:15s}: {gs['n']:3d}t  WR={wr_g:5.1f}%  Abs={gs['pnl']:+.0f}")

    # --- Yearly for top 5 ---
    if len(full_results) >= 2:
        print(f"\n  YEARLY BREAKDOWN FOR TOP 5:")
        for idx, r in enumerate(full_results[:5]):
            print(f"\n  #{idx + 1}: {r['name']} (Ann={r['ann']:+.1f}%, WR={r['wr']:.1f}%, DD={r['dd']:.1f}%)")
            for y in sorted(r['yearly'].keys()):
                ys = r['yearly'][y]
                wr_y = ys['w'] / ys['n'] * 100 if ys['n'] > 0 else 0
                print(f"    {y}: {ys['n']:3d}t  WR={wr_y:5.1f}%  PnL={ys['pnl']:+.1f}%")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.1f}s")
    print("=" * 130)


if __name__ == '__main__':
    main()
