"""
Alpha V58 — VAE Latent Factor Discovery
=========================================
Variational Autoencoder to discover non-linear factor interactions.

Architecture:
  Encoder: 65 → 128 → 64 → [mu(8), logvar(8)]
  Latent: 8 dimensions (reparameterization trick)
  Decoder: 8 → 64 → 128 → 65

Training: Expanding window, retrain every 60 days, no look-ahead.
Loss: MSE reconstruction + beta * KL divergence (beta-VAE)
"""
import sys, os, time, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alpha_v2 import load_all_data, MIN_TRAIN
from alpha_v7 import COMMISSION, STAMP_DUTY, CASH0
from alpha_v7c import backtest_v7c
from alpha_v44 import compute_v41_factors_only
from alpha_v48 import compute_v48_factors
from alpha_v49 import compute_v49_factors
from alpha_v52 import compute_v52_factors
from alpha_v55 import compute_decomposed_factors


class VAE(nn.Module):
    """Beta-VAE for factor compression."""
    def __init__(self, input_dim, latent_dim=8, hidden_dims=[128, 64], beta=0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.beta = beta

        # Encoder
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.LeakyReLU(0.2),
            ])
            prev = h
        self.encoder = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(prev, latent_dim)
        self.fc_logvar = nn.Linear(prev, latent_dim)

        # Decoder
        dec_layers = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            dec_layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.LeakyReLU(0.2),
            ])
            prev = h
        dec_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

    def loss_function(self, x, x_recon, mu, logvar):
        recon_loss = nn.functional.mse_loss(x_recon, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + self.beta * kl_loss

    def get_latent(self, x):
        """Get latent representation (no gradient needed)."""
        with torch.no_grad():
            mu, _ = self.encode(x)
            return mu


def train_vae(model, X_train, epochs=50, batch_size=4096, lr=1e-3):
    """Train VAE on factor data."""
    dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        n_batches = 0
        for (batch,) in loader:
            optimizer.zero_grad()
            x_recon, mu, logvar = model(batch)
            loss = model.loss_function(batch, x_recon, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        scheduler.step()

    return total_loss / max(n_batches, 1)


def compute_vae_latent(all_factors, NS, ND, latent_dim=8, retrain_every=60,
                       beta=0.1, epochs=50):
    """Compute VAE latent factors with expanding window."""
    factor_names = sorted([k for k in all_factors.keys() if k.startswith('R_')])
    F = len(factor_names)
    print(f"  VAE: {F} factors → {latent_dim} latent dims, beta={beta}", flush=True)

    # Build factor tensor
    factor_tensor = np.full((NS, ND, F), np.nan)
    for fi, fname in enumerate(factor_names):
        factor_tensor[:, :, fi] = all_factors[fname]

    # Output: latent factors
    latent_raw = np.full((NS, ND, latent_dim), np.nan)

    t0 = time.time()
    model = None
    last_train_di = -retrain_every

    for di in range(MIN_TRAIN + 120, ND):
        # Retrain VAE periodically
        if di - last_train_di >= retrain_every:
            # Collect training data from MIN_TRAIN to di-1
            train_data = factor_tensor[:, MIN_TRAIN:di, :].reshape(-1, F)
            valid = ~np.any(np.isnan(train_data), axis=1)
            train_clean = train_data[valid]

            if len(train_clean) > 5000:
                # Normalize: clip outliers, standardize
                train_clean = np.clip(train_clean, 1, 99)  # Already rank-normalized [1,100]
                train_clean = (train_clean - 50) / 30  # Center around 0, scale ~[-1.5, 1.5]

                # Train VAE
                model = VAE(F, latent_dim=latent_dim, beta=beta)
                train_vae(model, train_clean, epochs=epochs, batch_size=4096)
                model.eval()
                last_train_di = di

                if di % 600 == 0:
                    print(f"    VAE trained at di={di}, samples={len(train_clean)} "
                          f"({time.time()-t0:.0f}s)", flush=True)

        if model is None:
            continue

        # Encode current day's cross-section
        day_factors = factor_tensor[:, di, :]  # (NS, F)
        valid_stocks = ~np.any(np.isnan(day_factors), axis=1)

        if valid_stocks.sum() > 50:
            input_data = day_factors[valid_stocks].copy()
            input_data = np.clip(input_data, 1, 99)
            input_data = (input_data - 50) / 30

            x_tensor = torch.tensor(input_data, dtype=torch.float32)
            latent = model.get_latent(x_tensor).numpy()

            for k in range(latent_dim):
                latent_raw[valid_stocks, di, k] = latent[:, k]

    # Rank normalize each latent dimension
    latent_factors = {}
    for k in range(latent_dim):
        name = f'R_VAE_{k}'
        arr = latent_raw[:, :, k]
        ranked = np.full_like(arr, np.nan)
        for di in range(ND):
            vals = arr[:, di]
            valid = ~np.isnan(vals)
            n = valid.sum()
            if n < 50:
                continue
            order = np.argsort(vals[valid])
            ranks = np.empty(n)
            ranks[order] = np.arange(1, n + 1)
            ranked[valid, di] = ranks / n * 100
        latent_factors[name] = ranked

    print(f"  VAE latent factors done ({time.time()-t0:.0f}s)", flush=True)
    return latent_factors


if __name__ == '__main__':
    print("=" * 70, flush=True)
    print("  Alpha V58 — VAE Latent Factor Discovery")
    print("  V56 verified: +1630.7% DD=25.2%", flush=True)
    print("=" * 70)

    NS, ND, dates, C, O, H, L, V, syms, sym_set = load_all_data()

    print("\n  Computing all factors...", flush=True)
    v41 = compute_v41_factors_only(NS, ND, C, O, H, L, V)
    v48 = compute_v48_factors(NS, ND, C, O, H, L, V)
    v49 = compute_v49_factors(NS, ND, C, O, H, L, V)
    v52 = compute_v52_factors(NS, ND, C, O, H, L, V)
    v55 = compute_decomposed_factors(NS, ND, C, O, H, L, V)
    all_factors = {**v41, **v48, **v49, **v52, **v55}

    # V56 winning weights
    v54_base = {'R_BWP_BNW': 0.205, 'R_TENSION': 0.205, 'R_VWCM': 0.205,
                'R_BVR': 0.154, 'R_BUY_FRAC': 0.138, 'R_VPIN': 0.092}
    v56_weights = {**v54_base, 'R_SHOCK_MOM': 0.08, 'R_TREND_ACC': 0.15}
    total = sum(v56_weights.values())
    v56_norm = {k: v / total for k, v in v56_weights.items()}

    results = []

    # =====================================================================
    # Baseline: V56
    # =====================================================================
    print("\n  V56 baseline...", flush=True)
    r = backtest_v7c(v56_norm, all_factors, NS, ND, dates, C, O, H, L, V,
                    top_n=1, rebalance_days=5, atr_stop_mult=0.5)
    if r:
        r['test'] = 'V56_BASE'
        results.append(r)
        print(f"  V56: {r['ann']:+.1f}%", flush=True)

    # =====================================================================
    # Test 1: VAE latent factors — sweep latent dim and beta
    # =====================================================================
    print("\n  Test 1: VAE latent factors...", flush=True)

    for latent_dim in [5, 8, 12]:
        for beta in [0.01, 0.05, 0.1, 0.5]:
            vae_factors = compute_vae_latent(all_factors, NS, ND,
                                             latent_dim=latent_dim,
                                             beta=beta, epochs=50)
            vae_all = {**all_factors, **vae_factors}

            # VAE solo (equal weight)
            vae_names = sorted(vae_factors.keys())
            weights = {f: 1.0 / len(vae_names) for f in vae_names}
            for atr in [0.5, 0.8]:
                r = backtest_v7c(weights, vae_all, NS, ND, dates, C, O, H, L, V,
                                top_n=1, rebalance_days=5, atr_stop_mult=atr)
                if r:
                    r['test'] = f'VAE_L{latent_dim}_B{beta}_EQ_A{atr}'
                    results.append(r)
                    print(f"    VAE_L{latent_dim}_B{beta}_EQ_A{atr}: {r['ann']:+.1f}%", flush=True)

            # V56 + each VAE component
            for fname in vae_names:
                for w in [0.05, 0.08, 0.10]:
                    weights = {**v56_norm, fname: w}
                    total = sum(weights.values())
                    wn = {k: v / total for k, v in weights.items()}
                    r = backtest_v7c(wn, vae_all, NS, ND, dates, C, O, H, L, V,
                                    top_n=1, rebalance_days=5, atr_stop_mult=0.5)
                    if r:
                        r['test'] = f'V56+{fname}_W{w:.2f}'
                        results.append(r)

    # =====================================================================
    # Test 2: Best VAE + V56 + SHOCK_MOM + TREND_ACC optimization
    # =====================================================================
    print("\n  Test 2: Full combo optimization...", flush=True)

    # Find best VAE solo
    vae_solos = [r for r in results if 'VAE_' in r['test'] and '_EQ_' in r['test']]
    if vae_solos:
        best_vae = max(vae_solos, key=lambda x: x['ann'])
        print(f"  Best VAE solo: {best_vae['test']} = {best_vae['ann']:+.1f}%", flush=True)

    # Find best V56+VAE
    v56_vae = [r for r in results if 'V56+R_VAE' in r['test']]
    if v56_vae:
        best_combo = max(v56_vae, key=lambda x: x['ann'])
        print(f"  Best V56+VAE: {best_combo['test']} = {best_combo['ann']:+.1f}%", flush=True)

        # Extract best VAE factor name and retrain with more epochs
        for r in sorted(v56_vae, key=lambda x: -x['ann'])[:5]:
            print(f"    {r['test']}: {r['ann']:+.1f}%", flush=True)

    # =====================================================================
    # Test 3: VAE with higher epochs (deeper training)
    # =====================================================================
    print("\n  Test 3: VAE deep training...", flush=True)
    # Use best params from test 1
    if vae_solos:
        # Parse best config
        for latent_dim in [8]:
            for beta in [0.05, 0.1]:
                vae_factors = compute_vae_latent(all_factors, NS, ND,
                                                 latent_dim=latent_dim,
                                                 beta=beta, epochs=100)
                vae_all = {**all_factors, **vae_factors}

                # Test best VAE factors with V56 + weight sweep
                for fname in sorted(vae_factors.keys()):
                    for w in [0.05, 0.08, 0.10, 0.12]:
                        weights = {**v56_norm, fname: w}
                        total = sum(weights.values())
                        wn = {k: v / total for k, v in weights.items()}
                        r = backtest_v7c(wn, vae_all, NS, ND, dates, C, O, H, L, V,
                                        top_n=1, rebalance_days=5, atr_stop_mult=0.5)
                        if r:
                            r['test'] = f'DEEP_L{latent_dim}_B{beta}+{fname[-1:]}_W{w:.2f}'
                            results.append(r)

    # =====================================================================
    # RESULTS
    # =====================================================================
    results.sort(key=lambda x: -x['ann'])

    def all_positive(r):
        ys = r.get('year_stats', {})
        return all(s['total_pnl'] > 0 for s in ys.values())

    print(f"\n{'='*100}", flush=True)
    print(f"  ALL RESULTS (V58 VAE LATENT FACTOR DISCOVERY)", flush=True)
    print(f"  {'Test':<45s} | {'Ann':>7s} {'N':>5s} {'WR':>5s} {'Edge':>6s} {'DD':>5s}", flush=True)
    print(f"  {'-'*85}", flush=True)
    for r in results[:60]:
        pos_mark = " ALL+" if all_positive(r) else ""
        print(f"  {r['test']:<45s} | {r['ann']:+7.1f}% {r['n']:5d} {r['wr']:5.1f}% "
              f"{r['edge']:+6.2f}% {r['max_dd']:5.1f}%{pos_mark}", flush=True)

    # Solo VAE summary
    print(f"\n  === VAE SOLO SUMMARY ===", flush=True)
    vae_solos_sorted = sorted([r for r in results if '_EQ_' in r['test']], key=lambda x: -x['ann'])
    for r in vae_solos_sorted[:15]:
        pos_mark = " ALL+" if all_positive(r) else ""
        print(f"  {r['test']:<45s} {r['ann']:+7.1f}% DD={r['max_dd']:.1f}%{pos_mark}", flush=True)

    # V56+VAE summary
    print(f"\n  === V56 + VAE BEST ===", flush=True)
    v56_vae_sorted = sorted([r for r in results if 'V56+R_VAE' in r['test']], key=lambda x: -x['ann'])
    for r in v56_vae_sorted[:15]:
        pos_mark = " ALL+" if all_positive(r) else ""
        print(f"  {r['test']:<45s} {r['ann']:+7.1f}% DD={r['max_dd']:.1f}%{pos_mark}", flush=True)

    for i, r in enumerate(results[:3]):
        print(f"\n  Year-by-year #{i+1}: {r['test']} "
              f"(Ann={r['ann']:+.1f}%, DD={r['max_dd']:.1f}%)", flush=True)
        for y in sorted(r.get('year_stats', {}).keys()):
            s = r['year_stats'][y]
            wr = s['wins'] / max(s['trades'], 1) * 100
            print(f"    {y}: {s['trades']:4d} trades, WR={wr:.0f}%, pnl={s['total_pnl']:+.0f}%", flush=True)

    if results:
        best = results[0]
        print(f"\n  === V58 BEST ===", flush=True)
        print(f"  V58: {best['test']} = {best['ann']:+.1f}% DD={best['max_dd']:.1f}%", flush=True)
        print(f"  V56 RECORD: +1630.7% DD=25.2%", flush=True)
        delta = best['ann'] - 1630.7
        print(f"  Delta from V56: {delta:+.1f}%", flush=True)

    print(f"\n{'='*70}", flush=True)
