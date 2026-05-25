"""
Alpha Futures V33 — Supply Chain Momentum Lag Strategy
======================================================
Cross-commodity supply chain linkages create predictable lags.
When upstream moves, downstream follows. Exploit this lag.

Groups:
  黑色 (Ferrous):  rbfi(螺纹), hcfi(热卷), ifi(铁矿), jfi(焦炭), jmfi(焦煤)
  有色 (NonFerrous): cufi(铜), alfi(铝), znfi(锌), nifi(镍)
  油脂 (Oils):     afi(豆一), mfi(豆粕), yfi(豆油), pfi(棕榈油), cfi(玉米)
  能源 (Energy):    scfi(原油), mafi(甲醇), bfi(沥青), fufi(燃油)
  化工 (Chemical):  ppfi(PP), vfi(PVC), egfi(乙二醇), pgfi(纸浆)

Signal:
  Lead: upstream momentum positive + own momentum negative = lag opportunity
  Confirm: VDP EMA direction, OI rising (money flowing in)
  Score = lead_signal_strength * (1 + corr_strength) * VDP_confirm * OI_confirm

Exit: hold 3-5 days, time exit, signal flip
Single position, P1 concentrated, no leverage, COMM=0.0003
"""
import sys, os, time, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alpha_v2 import load_all_data, MIN_TRAIN, CASH0, compute_vdp

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

# ============================================================
# SUPPLY CHAIN DEFINITIONS
# ============================================================
# Group definitions: commodity -> group name
GROUP_MAP = {
    'rbfi': 'ferrous', 'hcfi': 'ferrous', 'ifi': 'ferrous', 'jfi': 'ferrous', 'jmfi': 'ferrous',
    'cufi': 'nonferrous', 'alfi': 'nonferrous', 'znfi': 'nonferrous', 'nifi': 'nonferrous',
    'afi': 'oils', 'mfi': 'oils', 'yfi': 'oils', 'pfi': 'oils', 'cfi': 'oils',
    'scfi': 'energy', 'mafi': 'energy', 'bfi': 'energy', 'fufi': 'energy',
    'ppfi': 'chemical', 'vfi': 'chemical', 'egfi': 'chemical', 'pgfi': 'chemical',
}

# Upstream leaders for each commodity (key = downstream, value = upstream leader)
UPSTREAM = {
    # 黑色: ifi(铁矿) -> rbfi(螺纹) -> hcfi(热卷); jmfi(焦煤) -> jfi(焦炭) -> rbfi
    'rbfi': 'ifi',      # iron ore -> rebar
    'hcfi': 'rbfi',     # rebar -> hot coil (co-integrated)
    'jfi': 'jmfi',      # coking coal -> coke
    # 能源: scfi(原油) -> downstream
    'mafi': 'scfi',     # crude -> methanol
    'bfi': 'scfi',      # crude -> asphalt
    'fufi': 'scfi',     # crude -> fuel oil
    # 油脂: afi(豆一) -> mfi(豆粕), yfi(豆油)
    'mfi': 'afi',       # soybean -> meal
    'yfi': 'afi',       # soybean -> oil
    'pfi': 'yfi',       # soy oil -> palm oil (substitute)
    # 化工: scfi/mafi -> pp/v/eg; pgfi is paper (standalone)
    'ppfi': 'mafi',     # methanol -> PP
    'vfi': 'mafi',      # methanol -> PVC
    'egfi': 'mafi',     # methanol -> EG
}

# Also define reverse: downstream for each upstream (for the "upstream lead" concept)
# If a commodity has no explicit upstream, it is a primary commodity and we use
# group momentum as the leader signal.


def main():
    t_start = time.time()
    print("=" * 110)
    print("Alpha Futures V33 — Supply Chain Momentum Lag Strategy")
    print("Core: upstream momentum leads downstream by 1-3 days")
    print("=" * 110)

    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)

    # ========================================
    # BUILD GROUP / UPSTREAM INDEX MAPS
    # ========================================
    sym_to_si = {syms[si]: si for si in range(NS)}

    # group_members[group_name] = list of si
    group_members = {}
    for si in range(NS):
        grp = GROUP_MAP.get(syms[si])
        if grp is None:
            continue
        if grp not in group_members:
            group_members[grp] = []
        group_members[grp].append(si)

    # upstream_si[si] = si of upstream leader (or -1)
    upstream_si = {}
    for si in range(NS):
        up_sym = UPSTREAM.get(syms[si])
        if up_sym and up_sym in sym_to_si:
            upstream_si[si] = sym_to_si[up_sym]
        else:
            upstream_si[si] = -1

    print(f"  Group members: {', '.join(f'{k}:{len(v)}' for k, v in group_members.items())}")
    print(f"  Upstream links: {sum(1 for v in upstream_si.values() if v >= 0)} commodities have leaders")

    # ========================================
    # PRECOMPUTE ALL SIGNALS
    # ========================================
    print("[Signals] Computing all signals...", flush=True)
    t0 = time.time()

    # --- 1. 5-day momentum for each commodity ---
    mom5 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            c_now = C[si, di]
            c_prev = C[si, di - 5]
            if not np.isnan(c_now) and not np.isnan(c_prev) and c_prev > 0:
                mom5[si, di] = (c_now - c_prev) / c_prev

    # --- 2. 10-day momentum (secondary) ---
    mom10 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(10, ND):
            c_now = C[si, di]
            c_prev = C[si, di - 10]
            if not np.isnan(c_now) and not np.isnan(c_prev) and c_prev > 0:
                mom10[si, di] = (c_now - c_prev) / c_prev

    # --- 3. Group momentum (average mom5 of other members in same group) ---
    group_mom5 = np.full((NS, ND), np.nan)
    for grp, members in group_members.items():
        for di in range(5, ND):
            moms = []
            for sj in members:
                m = mom5[sj, di]
                if not np.isnan(m):
                    moms.append(m)
            if moms:
                avg_mom = np.mean(moms)
                for sj in members:
                    group_mom5[sj, di] = avg_mom

    # --- 4. Group momentum excluding self ---
    group_mom5_excl = np.full((NS, ND), np.nan)
    for grp, members in group_members.items():
        for di in range(5, ND):
            for sj in members:
                moms = []
                for sk in members:
                    if sk == sj:
                        continue
                    m = mom5[sk, di]
                    if not np.isnan(m):
                        moms.append(m)
                if moms:
                    group_mom5_excl[sj, di] = np.mean(moms)

    # --- 5. Upstream leader momentum ---
    leader_mom5 = np.full((NS, ND), np.nan)
    for si in range(NS):
        usi = upstream_si[si]
        if usi < 0:
            # No upstream: use group momentum as proxy leader
            leader_mom5[si, :] = group_mom5_excl[si, :]
        else:
            # Use upstream's momentum, shifted by 1 day (to capture the lag)
            for di in range(1, ND):
                lm = mom5[usi, di - 1]  # yesterday's upstream momentum
                if not np.isnan(lm):
                    leader_mom5[si, di] = lm

    # --- 6. Rolling 20-day correlation with upstream leader ---
    corr_strength = np.full((NS, ND), np.nan)
    for si in range(NS):
        usi = upstream_si[si]
        if usi < 0:
            # No upstream: use correlation with group average
            for di in range(25, ND):
                own_vals = []
                lead_vals = []
                for dd in range(di - 20, di):
                    ov = mom5[si, dd]
                    gv = group_mom5_excl[si, dd]
                    if not np.isnan(ov) and not np.isnan(gv):
                        own_vals.append(ov)
                        lead_vals.append(gv)
                if len(own_vals) >= 10:
                    own_arr = np.array(own_vals)
                    lead_arr = np.array(lead_vals)
                    own_std = np.std(own_arr)
                    lead_std = np.std(lead_arr)
                    if own_std > 0 and lead_std > 0:
                        corr_strength[si, di] = np.corrcoef(own_arr, lead_arr)[0, 1]
        else:
            for di in range(25, ND):
                own_vals = []
                lead_vals = []
                for dd in range(di - 20, di):
                    ov = mom5[si, dd]
                    lv = mom5[usi, dd]
                    if not np.isnan(ov) and not np.isnan(lv):
                        own_vals.append(ov)
                        lead_vals.append(lv)
                if len(own_vals) >= 10:
                    own_arr = np.array(own_vals)
                    lead_arr = np.array(lead_vals)
                    own_std = np.std(own_arr)
                    lead_std = np.std(lead_arr)
                    if own_std > 0 and lead_std > 0:
                        corr_strength[si, di] = np.corrcoef(own_arr, lead_arr)[0, 1]

    # --- 7. VDP EMA (10-day) ---
    vdp_ema = np.full((NS, ND), np.nan)
    for si in range(NS):
        c, h, l, v = C[si], H[si], L[si], V[si]
        vdp_e = 0.0
        alpha = 2.0 / 11
        for di in range(1, ND):
            d = di - 1
            cd, hd, ld, vd = c[d], h[d], l[d], v[d]
            if np.isnan(cd) or np.isnan(hd) or np.isnan(ld) or np.isnan(vd):
                continue
            rng = hd - ld
            if rng <= 0:
                continue
            vdp_val = vd * (2 * cd - hd - ld) / rng
            vdp_e = alpha * vdp_val + (1 - alpha) * vdp_e
            vdp_ema[si, di] = vdp_e

    # --- 8. OI momentum 5-day ---
    oi_mom5 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(6, ND):
            d = di - 1
            oi_now = OI[si, d]
            if np.isnan(oi_now) or oi_now <= 0:
                continue
            oi_prev = OI[si, d - 5]
            if not np.isnan(oi_prev) and oi_prev > 0:
                oi_mom5[si, di] = (oi_now - oi_prev) / oi_prev

    # --- 9. OI EMA 5-day (smoothed OI trend) ---
    oi_ema = np.full((NS, ND), np.nan)
    for si in range(NS):
        oe = 0.0
        alpha_oi = 2.0 / 6
        for di in range(1, ND):
            oi_val = OI[si, di]
            if np.isnan(oi_val):
                continue
            oe = alpha_oi * oi_val + (1 - alpha_oi) * oe
            oi_ema[si, di] = oe

    # OI rising: current OI EMA > 5 days ago OI EMA
    oi_rising = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(6, ND):
            cur = oi_ema[si, di]
            prev = oi_ema[si, di - 5]
            if not np.isnan(cur) and not np.isnan(prev) and prev > 0:
                oi_rising[si, di] = (cur - prev) / prev

    # --- 10. ATR for trailing stop ---
    atr10 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(11, ND):
            trs = []
            for dd in range(di - 10, di):
                hi, lo = H[si, dd], L[si, dd]
                pc = C[si, dd - 1]
                if np.isnan(hi) or np.isnan(lo):
                    continue
                tr = hi - lo
                if not np.isnan(pc):
                    tr = max(tr, abs(hi - pc), abs(lo - pc))
                trs.append(tr)
            if trs:
                atr10[si, di] = np.mean(trs)

    print(f"  Done ({time.time()-t0:.1f}s)", flush=True)

    # ========================================
    # SCORING FUNCTIONS
    # ========================================

    def score_supply_chain_lag(si, di):
        """
        Core strategy: exploit upstream-downstream momentum lag.

        LONG setup: upstream leader mom5 > 0, own mom5 < 0 (lagging behind)
        SHORT setup: upstream leader mom5 < 0, own mom5 > 0 (lagging behind)

        Score = lead_signal * (1 + corr_strength) * VDP_confirm * OI_confirm
        """
        if di < 25:
            return np.nan

        own = mom5[si, di]
        leader = leader_mom5[si, di]
        corr = corr_strength[si, di]
        vdp = vdp_ema[si, di]
        oi_r = oi_rising[si, di]

        if np.isnan(own) or np.isnan(leader):
            return np.nan

        # --- Lead signal strength ---
        # The bigger the divergence between leader and own, the stronger the signal
        divergence = leader - own  # positive = leader ahead (long opportunity)
        lead_strength = np.clip(divergence * 10, -1, 1)

        # Only trade when there's meaningful divergence (> 0.5% spread)
        if abs(divergence) < 0.005:
            return np.nan

        # --- Correlation strength multiplier ---
        corr_mult = 1.0
        if not np.isnan(corr):
            # High correlation = more reliable lag signal
            # Map corr from [-1,1] to [0.3, 2.0]
            corr_mult = max(0.3, 1.0 + corr)

        # --- VDP confirmation ---
        vdp_mult = 1.0
        if not np.isnan(vdp):
            if (lead_strength > 0 and vdp > 0) or (lead_strength < 0 and vdp < 0):
                vdp_mult = 1.4  # VDP confirms direction
            else:
                vdp_mult = 0.3  # VDP contradicts

        # --- OI confirmation ---
        oi_mult = 1.0
        if not np.isnan(oi_r):
            if oi_r > 0.02:  # OI rising significantly
                oi_mult = 1.3
            elif oi_r > 0:
                oi_mult = 1.1
            elif oi_r < -0.02:  # OI falling
                oi_mult = 0.6
            else:
                oi_mult = 0.9

        score = lead_strength * corr_mult * vdp_mult * oi_mult
        return np.clip(score, -3, 3)

    def score_supply_chain_lag_aggressive(si, di):
        """
        More aggressive: also uses group momentum as secondary confirmation.
        Allows longer lags (leader 2-day delayed momentum).
        """
        if di < 25:
            return np.nan

        own = mom5[si, di]
        leader = leader_mom5[si, di]
        grp = group_mom5_excl[si, di]
        corr = corr_strength[si, di]
        vdp = vdp_ema[si, di]
        oi_r = oi_rising[si, di]

        if np.isnan(own) or np.isnan(leader):
            return np.nan

        divergence = leader - own
        lead_strength = np.clip(divergence * 12, -1, 1)

        if abs(divergence) < 0.003:
            return np.nan

        # Correlation
        corr_mult = 1.0
        if not np.isnan(corr):
            corr_mult = max(0.3, 1.0 + corr * 0.8)

        # Group momentum confirmation
        grp_mult = 1.0
        if not np.isnan(grp):
            if (lead_strength > 0 and grp > 0) or (lead_strength < 0 and grp < 0):
                grp_mult = 1.3
            else:
                grp_mult = 0.5

        # VDP
        vdp_mult = 1.0
        if not np.isnan(vdp):
            if (lead_strength > 0 and vdp > 0) or (lead_strength < 0 and vdp < 0):
                vdp_mult = 1.4
            else:
                vdp_mult = 0.3

        # OI
        oi_mult = 1.0
        if not np.isnan(oi_r):
            if oi_r > 0.02:
                oi_mult = 1.3
            elif oi_r > 0:
                oi_mult = 1.1
            elif oi_r < -0.02:
                oi_mult = 0.6

        score = lead_strength * corr_mult * grp_mult * vdp_mult * oi_mult
        return np.clip(score, -3, 3)

    def score_upstream_breakout(si, di):
        """
        Upstream breakout + downstream hasn't moved yet.
        Uses mom10 for upstream to catch sustained moves.
        """
        if di < 25:
            return np.nan

        own_m5 = mom5[si, di]
        own_m10 = mom10[si, di]
        corr = corr_strength[si, di]
        vdp = vdp_ema[si, di]
        oi_r = oi_rising[si, di]

        if np.isnan(own_m5):
            return np.nan

        # Get upstream momentum (10-day, for sustained move)
        usi = upstream_si[si]
        if usi >= 0:
            up_m5 = mom5[usi, di]
            up_m10 = mom10[usi, di]
            if np.isnan(up_m5) or np.isnan(up_m10):
                return np.nan

            # Upstream in strong trend (both m5 and m10 same direction)
            up_direction = (up_m5 + up_m10) / 2
            if abs(up_m5) < 0.005 and abs(up_m10) < 0.005:
                return np.nan
        else:
            # No upstream: use group momentum
            up_m5 = group_mom5_excl[si, di]
            if np.isnan(up_m5):
                return np.nan
            up_direction = up_m5
            if abs(up_m5) < 0.005:
                return np.nan

        # Lead: upstream direction vs own
        divergence = up_direction - own_m5
        lead_strength = np.clip(divergence * 10, -1, 1)

        if abs(divergence) < 0.004:
            return np.nan

        corr_mult = 1.0
        if not np.isnan(corr):
            corr_mult = max(0.3, 1.0 + corr)

        vdp_mult = 1.0
        if not np.isnan(vdp):
            if (lead_strength > 0 and vdp > 0) or (lead_strength < 0 and vdp < 0):
                vdp_mult = 1.5
            else:
                vdp_mult = 0.2

        oi_mult = 1.0
        if not np.isnan(oi_r):
            if oi_r > 0.02:
                oi_mult = 1.3
            elif oi_r > 0:
                oi_mult = 1.1
            elif oi_r < -0.02:
                oi_mult = 0.5

        score = lead_strength * corr_mult * vdp_mult * oi_mult
        return np.clip(score, -3, 3)

    def score_group_rotation(si, di):
        """
        Pure group momentum rotation: own lagging group = buy if group strong.
        No upstream concept, just group-relative momentum.
        """
        if di < 10:
            return np.nan

        own = mom5[si, di]
        grp_excl = group_mom5_excl[si, di]
        vdp = vdp_ema[si, di]
        oi_r = oi_rising[si, di]

        if np.isnan(own) or np.isnan(grp_excl):
            return np.nan

        # Group is moving but this commodity is lagging
        if abs(grp_excl) < 0.003:
            return np.nan

        divergence = grp_excl - own
        lead_strength = np.clip(divergence * 10, -1, 1)

        # Must have meaningful lag
        if abs(divergence) < 0.003:
            return np.nan

        vdp_mult = 1.0
        if not np.isnan(vdp):
            if (lead_strength > 0 and vdp > 0) or (lead_strength < 0 and vdp < 0):
                vdp_mult = 1.3
            else:
                vdp_mult = 0.4

        oi_mult = 1.0
        if not np.isnan(oi_r):
            if oi_r > 0.01:
                oi_mult = 1.2
            elif oi_r < -0.02:
                oi_mult = 0.6

        score = lead_strength * vdp_mult * oi_mult
        return np.clip(score, -2, 2)

    def score_corr_weighted_lead(si, di):
        """
        Only trade when correlation is HIGH — the lag is most exploitable
        when the historical link is strong.
        """
        if di < 25:
            return np.nan

        own = mom5[si, di]
        leader = leader_mom5[si, di]
        corr = corr_strength[si, di]
        vdp = vdp_ema[si, di]
        oi_r = oi_rising[si, di]

        if np.isnan(own) or np.isnan(leader):
            return np.nan

        # REQUIRE high correlation (> 0.3)
        if np.isnan(corr) or corr < 0.3:
            return np.nan

        divergence = leader - own
        lead_strength = np.clip(divergence * 12, -1, 1)

        if abs(divergence) < 0.004:
            return np.nan

        # Strong correlation multiplier (0.3 -> 1.0, 1.0 -> 2.0)
        corr_mult = 1.0 + corr

        vdp_mult = 1.0
        if not np.isnan(vdp):
            if (lead_strength > 0 and vdp > 0) or (lead_strength < 0 and vdp < 0):
                vdp_mult = 1.5
            else:
                vdp_mult = 0.2

        oi_mult = 1.0
        if not np.isnan(oi_r):
            if oi_r > 0.02:
                oi_mult = 1.4
            elif oi_r > 0:
                oi_mult = 1.1
            elif oi_r < -0.02:
                oi_mult = 0.5

        score = lead_strength * corr_mult * vdp_mult * oi_mult
        return np.clip(score, -3, 3)

    # ========================================
    # BACKTEST ENGINE
    # ========================================
    def run_backtest(score_fn, name, hold_min=3, hold_max=5, trail_atr_mult=2.5,
                     stop_loss_pct=0.05, allow_short=True):
        """Single position, P1 concentrated, no leverage."""
        cash = float(CASH0)
        trades = []
        pos = None  # single position dict
        last_exit_di = -999

        for di in range(MIN_TRAIN, ND):
            year = dates[di].year

            # --- Manage existing position ---
            if pos is not None:
                c = C[pos['si'], di]
                if np.isnan(c) or c <= 0:
                    c = pos['entry']
                mult = MULT.get(pos['sym'], DEF_MULT)
                mkt_val = c * mult * pos['lots']
                pnl = (c - pos['entry']) * mult * pos['lots'] * pos['dir']
                pnl_pct = pnl / (pos['entry'] * mult * pos['lots']) * 100 if pos['entry'] > 0 else 0
                days_held = di - pos['entry_di']

                exit_reason = None

                # Stop loss
                if pnl_pct / 100 < -stop_loss_pct:
                    exit_reason = 'stop'

                # Trailing stop
                if exit_reason is None and trail_atr_mult > 0 and days_held >= 2:
                    atr = pos.get('atr', 0)
                    if atr > 0:
                        if pos['dir'] == 1:
                            new_trail = c - trail_atr_mult * atr
                            if new_trail > pos.get('trail_price', pos['entry']):
                                pos['trail_price'] = new_trail
                            if c < pos['trail_price']:
                                exit_reason = 'trail'
                        else:
                            new_trail = c + trail_atr_mult * atr
                            if new_trail < pos.get('trail_price', pos['entry']):
                                pos['trail_price'] = new_trail
                            if c > pos['trail_price']:
                                exit_reason = 'trail'

                # Signal flip exit (after minimum hold)
                if exit_reason is None and days_held >= hold_min:
                    cur_score = score_fn(pos['si'], di)
                    if not np.isnan(cur_score):
                        if pos['dir'] == 1 and cur_score < -0.02:
                            exit_reason = 'signal_flip'
                        elif pos['dir'] == -1 and cur_score > 0.02:
                            exit_reason = 'signal_flip'

                # Rotation: switch to better opportunity (after minimum hold)
                if exit_reason is None and days_held >= hold_min:
                    best_si, best_dir, best_sc = -1, 0, 0
                    for sj in range(NS):
                        sc = score_fn(sj, di)
                        if np.isnan(sc):
                            continue
                        if sc > best_sc:
                            best_sc = sc
                            best_si = sj
                            best_dir = 1
                        if allow_short and -sc > best_sc:
                            best_sc = -sc
                            best_si = sj
                            best_dir = -1
                    cur_sc = score_fn(pos['si'], di)
                    if np.isnan(cur_sc):
                        cur_sc = 0
                    cur_sc = abs(cur_sc)
                    if best_sc > cur_sc * 1.5 + 0.05 and best_si != pos['si']:
                        exit_reason = 'rotate'

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
                    last_exit_di = di
                    pos = None

            # --- Open new position ---
            if pos is None:
                # Score all symbols
                scored = []
                for si in range(NS):
                    sc = score_fn(si, di)
                    if np.isnan(sc):
                        continue
                    sym = syms[si]
                    if sc > 0:
                        scored.append((si, 1, sc, sym))
                    if allow_short and -sc > 0:
                        scored.append((si, -1, -sc, sym))

                if not scored:
                    continue

                scored.sort(key=lambda x: -x[2])

                for best_si, best_dir, best_sc, best_sym in scored:
                    c = C[best_si, di]
                    if np.isnan(c) or c <= 0:
                        continue
                    mult = MULT.get(best_sym, DEF_MULT)
                    notional = c * mult
                    if notional <= 0:
                        continue

                    lots = int(cash / (notional * (1 + COMM)))
                    if lots <= 0:
                        continue
                    cost_in = notional * lots * (1 + COMM)
                    if cost_in > cash:
                        lots = int(cash / (notional * (1 + COMM)))
                        if lots <= 0:
                            continue
                    cost_in = notional * lots * (1 + COMM)

                    atr_val = 0
                    trs = []
                    for dd in range(max(1, di - 10), di + 1):
                        hi, lo, pc = H[best_si, dd], L[best_si, dd], C[best_si, dd - 1]
                        if np.isnan(hi) or np.isnan(lo):
                            continue
                        tr = hi - lo
                        if not np.isnan(pc):
                            tr = max(tr, abs(hi - pc), abs(lo - pc))
                        trs.append(tr)
                    if trs:
                        atr_val = np.mean(trs)

                    cash -= cost_in
                    if best_dir == 1:
                        trail_price = c - trail_atr_mult * atr_val
                    else:
                        trail_price = c + trail_atr_mult * atr_val
                    pos = {
                        'si': best_si, 'entry': c, 'entry_di': di,
                        'lots': lots, 'dir': best_dir, 'sym': best_sym,
                        'atr': atr_val, 'trail_price': trail_price,
                    }
                    break  # single position

        # Close remaining position
        if pos is not None:
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

        # Equity curve for MDD
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
        ann = ((cash / CASH0) ** (1 / yr) - 1) * 100

        nw = sum(1 for t in trades if t['pnl_abs'] > 0)
        wr = nw / len(trades) * 100
        avg_pnl = np.mean([t['pnl_pct'] for t in trades])
        avg_days = np.mean([t['days'] for t in trades])
        avg_win = np.mean([t['pnl_pct'] for t in trades if t['pnl_abs'] > 0]) if nw > 0 else 0
        avg_loss = np.mean([abs(t['pnl_pct']) for t in trades if t['pnl_abs'] <= 0]) if nw < len(trades) else 0
        profit_factor = (sum(t['pnl_abs'] for t in trades if t['pnl_abs'] > 0) /
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
                year_stats[y] = {'n': 0, 'w': 0, 'pnl': 0.0, 'abs': 0.0}
            year_stats[y]['n'] += 1
            if t['pnl_abs'] > 0:
                year_stats[y]['w'] += 1
            year_stats[y]['pnl'] += t['pnl_pct']
            year_stats[y]['abs'] += t['pnl_abs']

        # Direction breakdown
        long_trades = [t for t in trades if t['dir'] == 1]
        short_trades = [t for t in trades if t['dir'] == -1]
        long_wr = sum(1 for t in long_trades if t['pnl_abs'] > 0) / max(len(long_trades), 1) * 100
        short_wr = sum(1 for t in short_trades if t['pnl_abs'] > 0) / max(len(short_trades), 1) * 100

        # Symbol frequency
        sym_counts = {}
        for t in trades:
            s = t['sym']
            if s not in sym_counts:
                sym_counts[s] = 0
            sym_counts[s] += 1

        # Group frequency
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
            'avg_pnl': round(avg_pnl, 3), 'avg_days': round(avg_days, 1),
            'avg_win': round(avg_win, 2), 'avg_loss': round(avg_loss, 2),
            'cash': round(cash, 0), 'reasons': reasons, 'yearly': year_stats,
            'pf': round(profit_factor, 2),
            'long_n': len(long_trades), 'short_n': len(short_trades),
            'long_wr': round(long_wr, 1), 'short_wr': round(short_wr, 1),
            'sym_counts': sym_counts, 'grp_counts': grp_counts,
        }

    # ========================================
    # RUN ALL CONFIGS
    # ========================================
    print("\n[Backtest] Running all configurations...", flush=True)
    results = []

    configs = [
        # (score_fn, name, hold_min, hold_max, trail_atr, stop_loss, allow_short)
        # --- Core supply chain lag ---
        (score_supply_chain_lag, "SC_LAG_H3T25_SL5", 3, 5, 2.5, 0.05, True),
        (score_supply_chain_lag, "SC_LAG_H3T30_SL5", 3, 5, 3.0, 0.05, True),
        (score_supply_chain_lag, "SC_LAG_H3T25_SL4", 3, 5, 2.5, 0.04, True),
        (score_supply_chain_lag, "SC_LAG_H4T25_SL5", 4, 7, 2.5, 0.05, True),
        (score_supply_chain_lag, "SC_LAG_H3T25_SL5_L", 3, 5, 2.5, 0.05, False),

        # --- Aggressive (group confirmation) ---
        (score_supply_chain_lag_aggressive, "SC_AGR_H3T25_SL5", 3, 5, 2.5, 0.05, True),
        (score_supply_chain_lag_aggressive, "SC_AGR_H3T30_SL5", 3, 5, 3.0, 0.05, True),
        (score_supply_chain_lag_aggressive, "SC_AGR_H4T25_SL5", 4, 7, 2.5, 0.05, True),

        # --- Upstream breakout (sustained moves) ---
        (score_upstream_breakout, "UP_BRK_H3T25_SL5", 3, 5, 2.5, 0.05, True),
        (score_upstream_breakout, "UP_BRK_H4T30_SL5", 4, 7, 3.0, 0.05, True),
        (score_upstream_breakout, "UP_BRK_H3T25_SL5_L", 3, 5, 2.5, 0.05, False),

        # --- Group rotation ---
        (score_group_rotation, "GRP_ROT_H3T25_SL5", 3, 5, 2.5, 0.05, True),
        (score_group_rotation, "GRP_ROT_H3T30_SL5", 3, 5, 3.0, 0.05, True),

        # --- Correlation-weighted (high corr only) ---
        (score_corr_weighted_lead, "CORR_H3T25_SL5", 3, 5, 2.5, 0.05, True),
        (score_corr_weighted_lead, "CORR_H3T30_SL5", 3, 5, 3.0, 0.05, True),
        (score_corr_weighted_lead, "CORR_H4T25_SL5", 4, 7, 2.5, 0.05, True),
    ]

    for fn, name, hm, hx, ta, sl, ashort in configs:
        r = run_backtest(fn, name, hold_min=hm, hold_max=hx,
                         trail_atr_mult=ta, stop_loss_pct=sl, allow_short=ashort)
        if r:
            results.append(r)
            print(f"  {r['name']:30s} | Ann {r['ann']:+7.1f}% | WR {r['wr']:5.1f}% | "
                  f"N {r['n']:4d} | DD {r['dd']:6.1f}% | PF {r['pf']:4.2f} | "
                  f"AvgW {r['avg_win']:+.2f}% | AvgL {r['avg_loss']:.2f}% | "
                  f"AvgD {r['avg_days']:.1f} | L{r['long_n']}/S{r['short_n']}")
            parts = []
            for reason, stats in sorted(r['reasons'].items()):
                wr_r = stats['w'] / stats['n'] * 100 if stats['n'] > 0 else 0
                parts.append(f"{reason}:{stats['n']}({wr_r:.0f}%)")
            print(f"  {'':30s} | Exits: {' | '.join(parts)}")

    # ========================================
    # RESULTS SUMMARY
    # ========================================
    results.sort(key=lambda x: -x['ann'])

    print(f"\n{'=' * 110}")
    print(f"  RESULTS TABLE (sorted by Annual Return)")
    print(f"{'=' * 110}")
    print(f"  {'Name':30s} | {'Ann':>7s} | {'WR':>5s} | {'N':>4s} | {'DD':>6s} | "
          f"{'PF':>4s} | {'AvgW':>7s} | {'AvgL':>6s} | {'AvgD':>4s} | {'L/S':>7s}")
    print(f"  {'-' * 105}")
    for r in results:
        print(f"  {r['name']:30s} | {r['ann']:+7.1f}% | {r['wr']:5.1f}% | "
              f"{r['n']:4d} | {r['dd']:6.1f}% | {r['pf']:4.2f} | "
              f"{r['avg_win']:+6.2f}% | {r['avg_loss']:5.2f}% | "
              f"{r['avg_days']:4.1f} | {r['long_n']:3d}/{r['short_n']:<3d}")

    # Yearly breakdown for top 5
    if results:
        print(f"\n--- YEARLY BREAKDOWN (Top 5) ---")
        for r in results[:5]:
            print(f"\n  {r['name']} (Ann {r['ann']:+.1f}%, WR {r['wr']:.1f}%, DD {r['dd']:.1f}%):")
            for y in sorted(r['yearly'].keys()):
                ys = r['yearly'][y]
                wr_y = ys['w'] / ys['n'] * 100 if ys['n'] > 0 else 0
                print(f"    {y}: {ys['n']:3d}t  WR {wr_y:5.1f}%  PnL {ys['pnl']:+.1f}%  Abs {ys['abs']:+.0f}")

    # Group breakdown for top 3
    if len(results) >= 1:
        print(f"\n--- GROUP BREAKDOWN (Top 3) ---")
        for r in results[:3]:
            print(f"\n  {r['name']}:")
            for g in sorted(r['grp_counts'].keys(), key=lambda x: -r['grp_counts'][x]['n']):
                gs = r['grp_counts'][g]
                wr_g = gs['w'] / gs['n'] * 100 if gs['n'] > 0 else 0
                print(f"    {g:15s}: {gs['n']:3d}t  WR {wr_g:5.1f}%  Abs {gs['pnl']:+.0f}")

    # Direction breakdown for top 3
    if len(results) >= 1:
        print(f"\n--- DIRECTION BREAKDOWN (Top 3) ---")
        for r in results[:3]:
            print(f"  {r['name']}: Long {r['long_n']}t WR {r['long_wr']:.1f}% | "
                  f"Short {r['short_n']}t WR {r['short_wr']:.1f}%")

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"{'=' * 110}")


if __name__ == '__main__':
    main()
