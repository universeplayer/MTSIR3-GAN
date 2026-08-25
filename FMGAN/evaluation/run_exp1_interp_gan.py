"""
Experiment 1: Compare coarse imputation sources + GAN refinement on AirQuality.
Also test MOMENT on Weather (longer sequences).
"""
import os, sys
import numpy as np
import torch
import torch.nn as nn
from scipy import interpolate

_FMGAN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _FMGAN_ROOT not in sys.path:
    sys.path.insert(0, _FMGAN_ROOT)

from data.unified_loader import TimeSeriesImputationDataset, fit_standard_scaler
from foundation_model.moment_wrapper import MOMENTImputer
from evaluation.metrics import compute_all_metrics
from evaluation.run_combination_test import SimpleGANRefiner, train_refiner
from protocol import seed_everything


def linear_interp(X_obs_np, mask_np):
    result = X_obs_np.copy()
    for i in range(len(X_obs_np)):
        for j in range(X_obs_np.shape[-1]):
            observed_idx = np.where(mask_np[i, :, j] == 1)[0]
            if len(observed_idx) > 1:
                f = interpolate.interp1d(observed_idx, X_obs_np[i, observed_idx, j],
                                         kind='linear', fill_value='extrapolate')
                missing_idx = np.where(mask_np[i, :, j] == 0)[0]
                if len(missing_idx) > 0:
                    result[i, missing_idx, j] = f(missing_idx)
    return result


def collect(ds):
    xi, xo, m, ind = [], [], [], []
    for i in range(len(ds)):
        item = ds[i]
        xi.append(item['X_intact'].numpy())
        xo.append(item['X_observed'].numpy())
        m.append(item['mask'].numpy())
        ind.append(item['indicating_mask'].numpy())
    return np.stack(xi), np.stack(xo), np.stack(m), np.stack(ind)


def eval_gan_refiner(train_obs, train_mask, train_coarse, train_intact,
                     test_obs, test_mask, test_coarse, test_intact, test_indicating,
                     n_features, seq_len, epochs=50, device='cuda', seed=42):
    seed_everything(seed, deterministic=True)
    refiner = SimpleGANRefiner(n_features, seq_len).to(device)
    train_data = (train_obs, train_mask, train_coarse, train_intact)
    refiner = train_refiner(
        refiner, train_data, n_features, epochs=epochs, device=device, seed=seed,
    )

    refiner.eval()
    preds = []
    bs = 32
    for i in range(0, len(test_obs), bs):
        b_obs = torch.from_numpy(test_obs[i:i+bs]).float().to(device)
        b_mask = torch.from_numpy(test_mask[i:i+bs]).float().to(device)
        b_coarse = torch.from_numpy(test_coarse[i:i+bs]).float().to(device)
        with torch.no_grad():
            refined = refiner.generate(b_obs, b_mask, b_coarse)
        preds.append(refined.cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    return compute_all_metrics(preds, test_intact, test_indicating)


def run_dataset_experiment(dataset_name, data_path, seq_len, missing_rate=0.25,
                           seed=42):
    seed_everything(seed, deterministic=True)
    print()
    print('=' * 70)
    print(f'Dataset: {dataset_name} (seq_len={seq_len}, rate={missing_rate})')
    print('=' * 70)

    data = np.load(data_path)
    X_full = data['X']
    n = len(X_full)
    X_train_raw = X_full[:int(0.7 * n)]
    X_test_raw = X_full[int(0.85 * n):]
    scaler = fit_standard_scaler(X_train_raw)

    train_ds = TimeSeriesImputationDataset(
        X_train_raw, seq_len=seq_len, missing_rate=missing_rate, seed=seed,
        scaler=scaler,
    )
    test_ds = TimeSeriesImputationDataset(
        X_test_raw, seq_len=seq_len, missing_rate=missing_rate, seed=seed + 1,
        scaler=scaler,
    )

    train_intact, train_obs, train_mask, _ = collect(train_ds)
    test_intact, test_obs, test_mask, test_indicating = collect(test_ds)
    n_features = test_intact.shape[-1]

    print(f'  Train: {train_obs.shape}, Test: {test_obs.shape}')

    device = ('cuda' if torch.cuda.is_available()
              else 'mps' if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available()
              else 'cpu')
    results = {}

    # 1. Linear Interpolation alone
    print('  [1/6] Linear Interpolation...')
    test_li = linear_interp(test_obs, test_mask)
    results['Linear Interp'] = compute_all_metrics(test_li, test_intact, test_indicating)

    # 2. Linear Interp + GAN
    print('  [2/6] Linear Interp + GAN...')
    train_li = linear_interp(train_obs, train_mask)
    results['Linear Interp + GAN'] = eval_gan_refiner(
        train_obs, train_mask, train_li, train_intact,
        test_obs, test_mask, test_li, test_intact, test_indicating,
        n_features, seq_len, device=device, seed=seed,
    )

    # 3. MOMENT alone
    print('  [3/6] MOMENT alone...')
    imputer = MOMENTImputer(frozen=True, device=device)
    imputer.to(device)
    train_moment = imputer.impute_numpy(train_obs, train_mask, batch_size=4)
    test_moment = imputer.impute_numpy(test_obs, test_mask, batch_size=4)
    results['MOMENT alone'] = compute_all_metrics(test_moment, test_intact, test_indicating)

    # 4. MOMENT + GAN
    print('  [4/6] MOMENT + GAN...')
    results['MOMENT + GAN'] = eval_gan_refiner(
        train_obs, train_mask, train_moment, train_intact,
        test_obs, test_mask, test_moment, test_intact, test_indicating,
        n_features, seq_len, device=device, seed=seed,
    )

    # 5. Mean fill alone (simplest baseline)
    print('  [5/6] Mean fill...')
    # Per-feature mean from training set
    train_means = np.nanmean(train_intact, axis=(0, 1))
    test_mean_fill = test_obs.copy()
    for j in range(n_features):
        missing_positions = test_mask[:, :, j] == 0
        test_mean_fill[:, :, j][missing_positions] = train_means[j]
    results['Mean Fill'] = compute_all_metrics(test_mean_fill, test_intact, test_indicating)

    # 6. Mean fill + GAN
    print('  [6/6] Mean Fill + GAN...')
    train_mean_fill = train_obs.copy()
    for j in range(n_features):
        missing_positions = train_mask[:, :, j] == 0
        train_mean_fill[:, :, j][missing_positions] = train_means[j]
    results['Mean Fill + GAN'] = eval_gan_refiner(
        train_obs, train_mask, train_mean_fill, train_intact,
        test_obs, test_mask, test_mean_fill, test_intact, test_indicating,
        n_features, seq_len, device=device, seed=seed,
    )

    # Print results
    print()
    print('=' * 70)
    header = f"{'Method':<25} {'MAE':<10} {'MSE':<10} {'RMSE':<10}"
    print(header)
    print('-' * 70)
    for name, m in results.items():
        row = f"{name:<25} {m['MAE']:<10.6f} {m['MSE']:<10.6f} {m['RMSE']:<10.6f}"
        print(row)
    print('=' * 70)

    # Key comparisons
    for coarse_name in ['Linear Interp', 'MOMENT alone', 'Mean Fill']:
        gan_name = coarse_name.replace(' alone', '') + ' + GAN'
        if coarse_name + ' + GAN' in results:
            gan_name = coarse_name + ' + GAN'
        if gan_name in results:
            imp = (results[coarse_name]['MAE'] - results[gan_name]['MAE']) / results[coarse_name]['MAE'] * 100
            print(f"  GAN improvement over {coarse_name}: {imp:+.2f}% MAE")

    return results


if __name__ == '__main__':
    dataset_dir = os.path.join(_FMGAN_ROOT, '..', 'datasets')

    # Experiment 1: AirQuality (short sequences)
    run_dataset_experiment(
        'AirQuality',
        os.path.join(dataset_dir, 'AirQuality', 'data.npz'),
        seq_len=36, missing_rate=0.25,
    )

    # Experiment 2: Weather (longer sequences, better for MOMENT)
    weather_path = os.path.join(dataset_dir, 'Weather', 'data.npz')
    if os.path.exists(weather_path):
        run_dataset_experiment(
            'Weather',
            weather_path,
            seq_len=96, missing_rate=0.25,
        )
    else:
        print('\nWeather dataset not found, skipping.')
