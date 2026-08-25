"""
Unified data loader for FMGAN experiments.
Supports multiple datasets, missing patterns, and missing rates.
Uses PyGrinder for standardized missing value generation.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


def _validate_fitted_scaler(scaler, n_features):
    """Fail closed when a supplied scaler cannot safely transform a split."""
    required = ('mean_', 'scale_', 'transform')
    missing = [name for name in required if not hasattr(scaler, name)]
    if missing:
        raise ValueError(
            'scaler must already be fitted on the train split; missing '
            + ', '.join(missing)
        )

    mean = np.asarray(scaler.mean_)
    scale = np.asarray(scaler.scale_)
    if mean.size != n_features or scale.size != n_features:
        raise ValueError(
            'scaler feature count does not match data: '
            f'{mean.size} vs {n_features}'
        )
    if not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise ValueError(
            'train-fitted scaler contains non-finite statistics; this usually '
            'means at least one training feature is entirely NaN'
        )
    if np.any(scale <= 0):
        raise ValueError('train-fitted scaler contains a non-positive scale')


def fit_standard_scaler(X_train):
    """Fit the experiment scaler on training timesteps only.

    Keeping this operation outside ``TimeSeriesImputationDataset`` makes the
    train/validation/test relationship explicit: fit once here, then pass the
    returned object unchanged to every split.
    """
    X_train = np.asarray(X_train)
    if X_train.ndim != 2:
        raise ValueError(
            f'expected 2D time series (timesteps, features), got {X_train.shape}'
        )
    if X_train.shape[0] == 0 or X_train.shape[1] == 0:
        raise ValueError('cannot fit train-only scaler on an empty array')
    if np.isinf(X_train).any():
        raise ValueError('train-only scaler input contains infinite values')

    all_nan_features = np.flatnonzero(np.isnan(X_train).all(axis=0))
    if all_nan_features.size:
        indices = ', '.join(str(int(index)) for index in all_nan_features)
        raise ValueError(
            'train-only scaling requires at least one observed value per '
            f'feature; all-NaN training feature indices: {indices}'
        )

    scaler = StandardScaler().fit(X_train)
    _validate_fitted_scaler(scaler, X_train.shape[1])
    return scaler


class MissingPatternGenerator:
    """Generate missing value masks with different patterns."""

    @staticmethod
    def point_missing(shape, rate, rng=None):
        """MCAR (Missing Completely At Random) - random point missing."""
        rng = rng or np.random.default_rng()
        mask = rng.random(shape) > rate  # True = observed
        return mask.astype(np.float32)

    @staticmethod
    def subsequence_missing(shape, rate, min_len=5, max_len=20, rng=None):
        """Missing subsequences in individual features."""
        rng = rng or np.random.default_rng()
        n_samples, seq_len, n_features = shape
        mask = np.ones(shape, dtype=np.float32)

        for i in range(n_samples):
            for j in range(n_features):
                total_missing = 0
                target_missing = int(seq_len * rate)
                while total_missing < target_missing:
                    start = rng.integers(0, seq_len)
                    length = rng.integers(min_len, min(max_len, seq_len - start) + 1)
                    mask[i, start:start + length, j] = 0.0
                    total_missing += length
        return mask

    @staticmethod
    def block_missing(shape, rate, rng=None):
        """Block missing - rectangular regions missing across multiple features."""
        rng = rng or np.random.default_rng()
        n_samples, seq_len, n_features = shape
        mask = np.ones(shape, dtype=np.float32)

        for i in range(n_samples):
            total_elements = seq_len * n_features
            target_missing = int(total_elements * rate)
            current_missing = 0

            while current_missing < target_missing:
                t_start = rng.integers(0, seq_len)
                t_len = rng.integers(1, max(2, seq_len // 5))
                f_start = rng.integers(0, n_features)
                f_len = rng.integers(1, max(2, n_features // 3))

                t_end = min(t_start + t_len, seq_len)
                f_end = min(f_start + f_len, n_features)
                mask[i, t_start:t_end, f_start:f_end] = 0.0
                current_missing += (t_end - t_start) * (f_end - f_start)

        return mask


class TimeSeriesImputationDataset(Dataset):
    """
    Unified dataset for time series imputation.

    Returns:
        X_intact: ground truth (complete data)
        X_observed: observed data (with missing values zeroed out)
        mask: binary mask (1 = observed, 0 = missing)
        indicating_mask: mask for evaluation (only artificially masked positions)
    """

    def __init__(self, X, seq_len=96, missing_rate=0.25, missing_pattern='point',
                 stride=None, seed=42, scaler=None):
        """
        Args:
            X: numpy array of shape (total_len, n_features) - the full time series
            seq_len: length of each sample window
            missing_rate: fraction of values to mask
            missing_pattern: 'point', 'subsequence', or 'block'
            stride: step size between windows (default: seq_len // 2)
            seed: random seed for reproducibility
            scaler: optional fitted ``StandardScaler``. If supplied, it is
                used only for ``transform`` and is never refit. Omitting it
                preserves the legacy standalone behavior of fitting on ``X``.
        """
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(
                f'expected 2D time series (timesteps, features), got {X.shape}'
            )
        self.seq_len = seq_len
        self.missing_rate = missing_rate
        self.missing_pattern = missing_pattern
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)

        # Normalize. Formal experiments pass the train-fitted scaler to every
        # split. The fallback retains compatibility for standalone callers.
        if scaler is None:
            self.scaler = fit_standard_scaler(X)
            self.scaler_source = 'local_legacy_fit'
        else:
            _validate_fitted_scaler(scaler, X.shape[1])
            self.scaler = scaler
            self.scaler_source = 'external_train_fit'
        X_scaled = self.scaler.transform(X)
        introduced_nan = np.isnan(X_scaled) & ~np.isnan(X)
        if introduced_nan.any():
            raise ValueError(
                'train-fitted scaler introduced NaN values at originally '
                'observed positions; refusing to alter the evaluation mask'
            )

        # Create sliding windows
        stride = stride or seq_len // 2
        self.stride = int(stride)
        self.samples = []
        for start in range(0, len(X_scaled) - seq_len + 1, stride):
            window = X_scaled[start:start + seq_len]
            if not np.isnan(window).all():
                self.samples.append(window.astype(np.float32))

        self.samples = np.stack(self.samples)  # (N, seq_len, n_features)

        # Handle originally missing values (NaN)
        self.original_mask = (~np.isnan(self.samples)).astype(np.float32)
        self.samples = np.nan_to_num(self.samples, nan=0.0)

        # Generate artificial missing masks
        gen = MissingPatternGenerator()
        gen_func = {
            'point': gen.point_missing,
            'subsequence': gen.subsequence_missing,
            'block': gen.block_missing,
        }[missing_pattern]

        self.artificial_mask = gen_func(self.samples.shape, missing_rate, self.rng)
        # Combined mask: observed only if both originally present AND not artificially masked
        self.combined_mask = self.original_mask * self.artificial_mask
        # Indicating mask: positions that are artificially masked (for evaluation)
        self.indicating_mask = self.original_mask * (1 - self.artificial_mask)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        X_intact = torch.from_numpy(self.samples[idx])       # (seq_len, features)
        mask = torch.from_numpy(self.combined_mask[idx])      # (seq_len, features)
        indicating = torch.from_numpy(self.indicating_mask[idx])
        X_observed = X_intact * mask                          # Zero out missing

        return {
            'X_intact': X_intact,
            'X_observed': X_observed,
            'mask': mask,
            'indicating_mask': indicating,
        }


def load_dataset(name, seq_len=96, missing_rate=0.25, missing_pattern='point',
                 batch_size=64, seed=42, split='train', num_workers=4,
                 scaler=None):
    """
    Load a dataset by name and return train/val/test DataLoaders.

    Args:
        name: dataset name (AirQuality, ETTh1, ETTh2, PhysioNet2012, etc.)
        seq_len: sample window length
        missing_rate: fraction of values to artificially mask
        missing_pattern: 'point', 'subsequence', or 'block'
        batch_size: batch size
        seed: random seed
        split: 'train', 'val', or 'test'
        num_workers: dataloader workers
        scaler: optional scaler previously fitted on the training split

    Returns:
        DataLoader for the specified split
    """
    import pypots.data
    import pygrinder

    # Use PyPOTS unified loading where available
    pypots_datasets = {
        'PhysioNet2012': 'physionet_2012',
        'ETTh1': 'ETTh1',
        'ETTh2': 'ETTh2',
    }

    if name in pypots_datasets:
        data = pypots.data.load_specific_dataset(pypots_datasets[name])
        X = data['train_X'] if split == 'train' else data.get('test_X', data['train_X'])
    else:
        import os
        data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'datasets', name)
        npz_path = os.path.join(data_dir, 'data.npz')
        if os.path.exists(npz_path):
            loaded = np.load(npz_path)
            X_full = loaded['X']
            # Simple 70/15/15 split
            n = len(X_full)
            if split == 'train':
                X = X_full[:int(0.7 * n)]
            elif split == 'val':
                X = X_full[int(0.7 * n):int(0.85 * n)]
            else:
                X = X_full[int(0.85 * n):]
        else:
            raise FileNotFoundError(f"Dataset {name} not found at {data_dir}")

    # If X is 3D (N, T, F), flatten to 2D (N*T, F) for windowing
    if X.ndim == 3:
        n_samples, t, f = X.shape
        X = X.reshape(-1, f)

    dataset = TimeSeriesImputationDataset(
        X, seq_len=seq_len, missing_rate=missing_rate,
        missing_pattern=missing_pattern, seed=seed, scaler=scaler,
    )

    try:
        from protocol import make_dataloader_generator, seed_dataloader_worker
    except ModuleNotFoundError as exc:
        if exc.name != 'protocol':
            raise
        # Support callers importing this module as
        # ``FMGAN.data.unified_loader`` from the repository root.
        from ..protocol import make_dataloader_generator, seed_dataloader_worker

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=(split == 'train'),
        num_workers=num_workers, pin_memory=True, drop_last=(split == 'train'),
        generator=make_dataloader_generator(seed),
        worker_init_fn=seed_dataloader_worker,
    )
    return loader
