"""
Factor Cache Utility
====================
Computes all factors once and saves to pickle.
Future backtests can load the cache in ~10 seconds instead of 45+ minutes.

Usage:
  python alpha_factor_cache.py          # Compute and save cache
  # In backtest scripts:
  from alpha_factor_cache import load_cached_factors
  all_factors, NS, ND, dates, C, O, H, L, V, syms, sym_set = load_cached_factors()
"""
import sys, os, time, warnings, pickle, hashlib
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alpha_v2 import load_all_data, MIN_TRAIN
from alpha_v7 import compute_all_factors
from alpha_v7b import compute_interaction_factors
from alpha_v7d import compute_extra_factors
from alpha_v7e import compute_v7e_factors
from alpha_v7f import compute_advanced_interactions
from alpha_v8 import compute_v8_factors, compute_v8_interactions
from alpha_v9 import compute_v9_factors, compute_v9_interactions
from alpha_v10 import compute_v10_factors, compute_v10_interactions
from alpha_v11 import compute_v11_factors, compute_v11_interactions
from alpha_v14 import compute_v14_factors, compute_v14_interactions

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'factor_cache')
CACHE_FILE = os.path.join(CACHE_DIR, 'all_factors.pkl')


def compute_all_cached_factors():
    """Compute all factors and save to cache."""
    os.makedirs(CACHE_DIR, exist_ok=True)

    t0 = time.time()
    NS, ND, dates, C, O, H, L, V, syms, sym_set = load_all_data()
    t_data = time.time() - t0
    print(f"[Cache] Data loaded: {t_data:.0f}s", flush=True)

    t1 = time.time()
    base = compute_all_factors(NS, ND, C, O, H, L, V)
    t_base = time.time() - t1
    print(f"[Cache] Base factors: {len(base)} in {t_base:.0f}s", flush=True)

    inter = compute_interaction_factors(base, NS, ND, C, O, H, L, V)
    extra = compute_extra_factors(NS, ND, C, O, H, L, V)
    v7e = compute_v7e_factors(NS, ND, C, O, H, L, V)
    adv = compute_advanced_interactions({**base, **inter, **extra, **v7e}, NS, ND)

    v8f = compute_v8_factors(NS, ND, C, O, H, L, V)
    v8_all = {**base, **inter, **extra, **v7e, **adv, **v8f}
    v8_inter = compute_v8_interactions(v8_all, NS, ND)
    v8_all.update(v8_inter)

    v9f = compute_v9_factors(NS, ND, C, O, H, L, V)
    v9_all = {**v8_all, **v9f}
    v9_inter = compute_v9_interactions(v9_all, NS, ND)
    v9_all.update(v9_inter)

    v10f = compute_v10_factors(NS, ND, C, O, H, L, V)
    v10_all = {**v9_all, **v10f}
    v10_inter = compute_v10_interactions(v10_all, NS, ND)
    v10_all.update(v10_inter)

    v11f = compute_v11_factors(NS, ND, C, O, H, L, V)
    v11_all = {**v10_all, **v11f}
    v11_inter = compute_v11_interactions(v11_all, NS, ND)
    v11_all.update(v11_inter)

    v14f = compute_v14_factors(NS, ND, C, O, H, L, V)
    all_factors = {**v11_all, **v14f}
    v14_inter = compute_v14_interactions(all_factors, NS, ND)
    all_factors.update(v14_inter)

    t_factors = time.time() - t1
    print(f"[Cache] All {len(all_factors)} factors computed in {t_factors:.0f}s", flush=True)

    # Save to pickle
    t3 = time.time()
    cache_data = {
        'factors': all_factors,
        'NS': NS, 'ND': ND, 'dates': dates,
        'C': C, 'O': O, 'H': H, 'L': L, 'V': V,
        'syms': syms, 'sym_set': sym_set,
    }
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    cache_size = os.path.getsize(CACHE_FILE) / 1024 / 1024
    t_save = time.time() - t3
    print(f"[Cache] Saved {cache_size:.0f} MB in {t_save:.0f}s → {CACHE_FILE}", flush=True)
    print(f"[Cache] Total time: {time.time() - t0:.0f}s", flush=True)
    return all_factors, NS, ND, dates, C, O, H, L, V, syms, sym_set


def load_cached_factors(force_recompute=False):
    """Load factors from cache, computing if needed.

    Returns: (all_factors, NS, ND, dates, C, O, H, L, V, syms, sym_set)
    """
    if not force_recompute and os.path.exists(CACHE_FILE):
        print(f"[Cache] Loading from {CACHE_FILE}...", flush=True)
        t0 = time.time()
        with open(CACHE_FILE, 'rb') as f:
            data = pickle.load(f)
        t_load = time.time() - t0
        print(f"[Cache] Loaded {len(data['factors'])} factors in {t_load:.1f}s", flush=True)
        return (data['factors'], data['NS'], data['ND'], data['dates'],
                data['C'], data['O'], data['H'], data['L'], data['V'],
                data['syms'], data['sym_set'])
    else:
        if force_recompute:
            print("[Cache] Force recomputing...", flush=True)
        else:
            print("[Cache] No cache found, computing...", flush=True)
        return compute_all_cached_factors()


if __name__ == '__main__':
    compute_all_cached_factors()
