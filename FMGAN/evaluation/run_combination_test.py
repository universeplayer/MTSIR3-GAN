"""
Phase 0.4: Quick combination test.
Tests whether MOMENT coarse imputation + R3GAN refinement > either alone.

This is a lightweight prototype to validate the core hypothesis before
committing to full model development.

Usage:
    python evaluation/run_combination_test.py --dataset AirQuality
"""

import argparse
import json
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn

# Ensure FMGAN root is on the path regardless of working directory
_FMGAN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _FMGAN_ROOT not in sys.path:
    sys.path.insert(0, _FMGAN_ROOT)

from foundation_model.moment_wrapper import MOMENTImputer
from data.unified_loader import TimeSeriesImputationDataset, fit_standard_scaler
from evaluation.metrics import compute_all_metrics
from protocol import make_dataloader_generator, seed_dataloader_worker, seed_everything


class SimpleGANRefiner(nn.Module):
    """
    Lightweight GAN refiner for quick hypothesis validation.
    Not the full R3GAN — just a simple ConvNet to test if
    refining MOMENT's output improves quality.

    Full R3GAN integration will be done in Phase 1 after validation.
    """

    def __init__(self, n_features, seq_len, noise_dim=32):
        super().__init__()
        self.noise_dim = noise_dim
        # Input: [X_observed, mask, X_coarse, noise] → 3*n_features + noise_dim channels
        in_channels = 3 * n_features + noise_dim

        self.generator = nn.Sequential(
            # Treat (seq_len,) as 1D spatial dimension
            nn.Conv1d(in_channels, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(256, n_features, kernel_size=3, padding=1),
        )

        self.discriminator = nn.Sequential(
            nn.Conv1d(n_features, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, 1),
        )

    def generate(self, X_observed, mask, X_coarse):
        """Generate refined imputation."""
        B, T, F = X_observed.shape
        noise = torch.randn(B, T, self.noise_dim, device=X_observed.device)

        # Concatenate all inputs: (B, T, 3F + noise_dim)
        x = torch.cat([X_observed, mask, X_coarse, noise], dim=-1)
        x = x.permute(0, 2, 1)  # (B, C, T) for Conv1d

        residual = self.generator(x)  # (B, F, T)
        residual = residual.permute(0, 2, 1)  # (B, T, F)

        # Residual learning: output = X_coarse + residual
        X_refined = X_coarse + residual

        # Keep observed values unchanged
        X_final = X_observed * mask + X_refined * (1 - mask)
        return X_final

    def discriminate(self, x):
        """Discriminate real vs fake."""
        x = x.permute(0, 2, 1)  # (B, F, T)
        return self.discriminator(x)


def train_refiner(refiner, train_data, n_features, epochs=50, lr=1e-4,
                  device='cuda', seed=42):
    """Quick training loop for the refiner."""
    opt_g = torch.optim.Adam(refiner.generator.parameters(), lr=lr, betas=(0.0, 0.9))
    opt_d = torch.optim.Adam(refiner.discriminator.parameters(), lr=lr, betas=(0.0, 0.9))

    X_obs, mask, X_coarse, X_intact = [torch.from_numpy(x).float().to(device) for x in train_data]

    dataset = torch.utils.data.TensorDataset(X_obs, mask, X_coarse, X_intact)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=True, drop_last=True,
        generator=make_dataloader_generator(seed),
        worker_init_fn=seed_dataloader_worker,
    )

    refiner.train()
    for epoch in range(epochs):
        total_g_loss = 0
        total_d_loss = 0
        n_batches = 0

        for batch_obs, batch_mask, batch_coarse, batch_real in loader:
            # ---- Train Discriminator ----
            opt_d.zero_grad()
            with torch.no_grad():
                fake = refiner.generate(batch_obs, batch_mask, batch_coarse)
            d_real = refiner.discriminate(batch_real)
            d_fake = refiner.discriminate(fake)

            # Relativistic loss (matching R3GAN)
            d_loss = (nn.functional.softplus(-(d_real - d_fake))).mean()
            d_loss.backward()
            opt_d.step()

            # ---- Train Generator ----
            opt_g.zero_grad()
            fake = refiner.generate(batch_obs, batch_mask, batch_coarse)
            d_real = refiner.discriminate(batch_real).detach()
            d_fake = refiner.discriminate(fake)

            # Adversarial loss
            g_adv = (nn.functional.softplus(-(d_fake - d_real))).mean()
            # Reconstruction loss on observed positions
            g_recon = nn.functional.l1_loss(fake * batch_mask, batch_real * batch_mask)
            g_loss = g_adv + 10.0 * g_recon

            g_loss.backward()
            opt_g.step()

            total_g_loss += g_loss.item()
            total_d_loss += d_loss.item()
            n_batches += 1

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: G_loss={total_g_loss/n_batches:.4f}, "
                  f"D_loss={total_d_loss/n_batches:.4f}")

    return refiner


def run_combination_test(dataset_name, missing_rate=0.25, seq_len=96, seed=42):
    """Run the full combination test pipeline."""
    seed_everything(seed, deterministic=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # 1. Load data
    print(f"\n[1/5] Loading {dataset_name} dataset...")
    data_path = os.path.join(
        os.path.dirname(__file__), '..', '..', 'datasets', dataset_name, 'data.npz'
    )
    loaded = np.load(data_path)
    X_full = loaded['X']
    n = len(X_full)

    # Train/test split
    X_train_raw = X_full[:int(0.7 * n)]
    X_test_raw = X_full[int(0.85 * n):]
    scaler = fit_standard_scaler(X_train_raw)

    train_ds = TimeSeriesImputationDataset(X_train_raw, seq_len=seq_len,
                                            missing_rate=missing_rate, seed=seed,
                                            scaler=scaler)
    test_ds = TimeSeriesImputationDataset(X_test_raw, seq_len=seq_len,
                                           missing_rate=missing_rate, seed=seed + 1,
                                           scaler=scaler)

    # Collect numpy arrays
    def collect(ds):
        X_intact, X_obs, masks, indicating = [], [], [], []
        for i in range(len(ds)):
            item = ds[i]
            X_intact.append(item['X_intact'].numpy())
            X_obs.append(item['X_observed'].numpy())
            masks.append(item['mask'].numpy())
            indicating.append(item['indicating_mask'].numpy())
        return (np.stack(X_intact), np.stack(X_obs),
                np.stack(masks), np.stack(indicating))

    train_intact, train_obs, train_mask, _ = collect(train_ds)
    test_intact, test_obs, test_mask, test_indicating = collect(test_ds)
    n_features = test_intact.shape[-1]

    # 2. MOMENT coarse imputation
    print("\n[2/5] Running MOMENT coarse imputation...")
    moment = MOMENTImputer(frozen=True, device=device)
    moment.to(device)

    train_coarse = moment.impute_numpy(train_obs, train_mask)
    test_coarse = moment.impute_numpy(test_obs, test_mask)

    moment_metrics = compute_all_metrics(test_coarse, test_intact, test_indicating)
    print(f"  MOMENT alone: MAE={moment_metrics['MAE']:.6f}, MSE={moment_metrics['MSE']:.6f}")

    # 3. Train GAN refiner
    print("\n[3/5] Training GAN refiner (50 epochs)...")
    refiner = SimpleGANRefiner(n_features, seq_len).to(device)
    train_data = (train_obs, train_mask, train_coarse, train_intact)
    refiner = train_refiner(
        refiner, train_data, n_features, epochs=50, device=device, seed=seed,
    )

    # 4. Evaluate combination
    print("\n[4/5] Evaluating MOMENT + GAN combination...")
    refiner.eval()
    test_combined = []
    bs = 32
    for i in range(0, len(test_obs), bs):
        batch_obs = torch.from_numpy(test_obs[i:i+bs]).float().to(device)
        batch_mask = torch.from_numpy(test_mask[i:i+bs]).float().to(device)
        batch_coarse = torch.from_numpy(test_coarse[i:i+bs]).float().to(device)
        with torch.no_grad():
            refined = refiner.generate(batch_obs, batch_mask, batch_coarse)
        test_combined.append(refined.cpu().numpy())

    test_combined = np.concatenate(test_combined, axis=0)
    combo_metrics = compute_all_metrics(test_combined, test_intact, test_indicating)

    # 5. Also evaluate "no MOMENT" (GAN with linear interpolation as coarse)
    print("\n[5/5] Evaluating GAN with linear interpolation (no MOMENT)...")
    # Simple linear interpolation baseline
    from scipy import interpolate

    def linear_interp(X_obs_np, mask_np):
        result = X_obs_np.copy()
        for i in range(len(X_obs_np)):
            for j in range(X_obs_np.shape[-1]):
                observed_idx = np.where(mask_np[i, :, j] == 1)[0]
                if len(observed_idx) > 1:
                    f = interpolate.interp1d(observed_idx, X_obs_np[i, observed_idx, j],
                                            kind='linear', fill_value='extrapolate')
                    missing_idx = np.where(mask_np[i, :, j] == 0)[0]
                    result[i, missing_idx, j] = f(missing_idx)
        return result

    test_linear = linear_interp(test_obs, test_mask)
    linear_metrics = compute_all_metrics(test_linear, test_intact, test_indicating)

    # Summary
    print(f"\n{'='*70}")
    print(f"COMBINATION TEST RESULTS - {dataset_name} (rate={missing_rate})")
    print(f"{'='*70}")
    print(f"{'Method':<30} {'MAE':<10} {'MSE':<10} {'RMSE':<10}")
    print(f"{'-'*70}")
    print(f"{'Linear Interpolation':<30} {linear_metrics['MAE']:<10.6f} {linear_metrics['MSE']:<10.6f} {linear_metrics['RMSE']:<10.6f}")
    print(f"{'MOMENT alone':<30} {moment_metrics['MAE']:<10.6f} {moment_metrics['MSE']:<10.6f} {moment_metrics['RMSE']:<10.6f}")
    print(f"{'MOMENT + GAN (Ours)':<30} {combo_metrics['MAE']:<10.6f} {combo_metrics['MSE']:<10.6f} {combo_metrics['RMSE']:<10.6f}")
    print(f"{'='*70}")

    improvement_over_moment = (moment_metrics['MAE'] - combo_metrics['MAE']) / moment_metrics['MAE'] * 100
    print(f"\nGAN refinement improvement over MOMENT: {improvement_over_moment:+.2f}% MAE")

    if improvement_over_moment > 0:
        print(">>> POSITIVE SIGNAL: GAN refinement adds value! Proceed to Phase 1.")
    elif improvement_over_moment > -2:
        print(">>> NEUTRAL: Marginal difference. May need frequency-domain loss or better architecture.")
    else:
        print(">>> NEGATIVE: GAN hurts performance. Investigate training stability or rethink approach.")

    results = {
        'dataset': dataset_name,
        'missing_rate': missing_rate,
        'seed': seed,
        'linear_interp': linear_metrics,
        'moment_alone': moment_metrics,
        'moment_plus_gan': combo_metrics,
        'improvement_pct': round(improvement_over_moment, 2),
    }
    return results


def main():
    parser = argparse.ArgumentParser(description='Phase 0.4: Combination test')
    parser.add_argument('--dataset', type=str, default='AirQuality')
    parser.add_argument('--missing_rate', type=float, default=0.25)
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default='results/phase0_combination.json')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    results = run_combination_test(args.dataset, args.missing_rate, args.seq_len, args.seed)

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
