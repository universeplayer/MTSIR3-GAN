"""Regression tests for split, missingness, import, and seed hardening."""

import contextlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np


FMGAN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(FMGAN_ROOT)
if FMGAN_ROOT not in sys.path:
    sys.path.insert(0, FMGAN_ROOT)

from evaluation.metrics import compute_all_metrics  # noqa: E402
from evaluation.run_baselines import (  # noqa: E402
    apply_artificial_missing,
    masked_metric_inputs,
    scale_splits_train_only,
    split_raw_data,
    window_timeline,
)
import evaluation.run_exp1_interp_gan as exp1  # noqa: E402


class BaselineSplitTests(unittest.TestCase):
    def test_continuous_timeline_is_split_before_windowing(self):
        timeline = np.arange(100, dtype=np.float32).reshape(-1, 1)
        raw_splits, needs_windowing = split_raw_data(timeline)

        self.assertTrue(needs_windowing)
        self.assertLess(raw_splits['train'].max(), raw_splits['val'].min())
        self.assertLess(raw_splits['val'].max(), raw_splits['test'].min())

        train_windows = window_timeline(
            raw_splits['train'], seq_len=10, stride=5, split_name='train',
        )
        val_windows = window_timeline(
            raw_splits['val'], seq_len=10, stride=5, split_name='val',
        )
        self.assertLess(train_windows.max(), val_windows.min())

    def test_scaler_counts_raw_training_timesteps_not_overlapping_windows(self):
        timeline = np.arange(200, dtype=np.float32).reshape(100, 2)
        raw_splits, _ = split_raw_data(timeline)
        _, scaler = scale_splits_train_only(raw_splits)

        self.assertEqual(int(scaler.n_samples_seen_), len(raw_splits['train']))
        np.testing.assert_allclose(
            scaler.mean_, raw_splits['train'].mean(axis=0),
        )


class NaturalMissingnessTests(unittest.TestCase):
    def test_original_nan_is_never_an_evaluation_target(self):
        values = np.array([[[1.0], [np.nan], [3.0], [4.0]]], dtype=np.float32)
        artificial = np.array([[[1.0], [0.0], [0.0], [1.0]]], dtype=np.float32)
        intact, missing, combined, indicating = apply_artificial_missing(
            values, artificial,
        )

        self.assertEqual(float(indicating[0, 1, 0]), 0.0)
        self.assertEqual(float(indicating[0, 2, 0]), 1.0)
        self.assertTrue(np.isnan(missing[0, 1, 0]))
        self.assertEqual(float(combined[0, 1, 0]), 0.0)

        prediction = np.array([[[1.0], [np.nan], [2.5], [4.0]]], dtype=np.float32)
        pred_eval, target_eval = masked_metric_inputs(
            prediction, intact, indicating,
        )
        metrics = compute_all_metrics(pred_eval, target_eval, indicating)
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
        self.assertAlmostEqual(metrics['MAE'], 0.5, places=6)

    def test_nonfinite_prediction_at_evaluation_position_fails_closed(self):
        prediction = np.array([[[np.nan]]], dtype=np.float32)
        target = np.array([[[1.0]]], dtype=np.float32)
        indicating = np.ones_like(target)
        with self.assertRaisesRegex(ValueError, 'predictions are non-finite'):
            masked_metric_inputs(prediction, target, indicating)


class CompatibilityAndSeedTests(unittest.TestCase):
    def test_namespace_package_import_resolves_protocol_helper(self):
        program = """
import sys
import types
import numpy as np

pypots = types.ModuleType('pypots')
pypots_data = types.ModuleType('pypots.data')
pypots_data.load_specific_dataset = lambda _name: {
    'train_X': np.arange(40, dtype=np.float32).reshape(40, 1),
}
pypots.data = pypots_data
sys.modules['pypots'] = pypots
sys.modules['pypots.data'] = pypots_data
sys.modules['pygrinder'] = types.ModuleType('pygrinder')

from FMGAN.data.unified_loader import load_dataset
loader = load_dataset(
    'PhysioNet2012', seq_len=4, batch_size=2, num_workers=0,
)
print(type(loader).__name__)
"""
        completed = subprocess.run(
            [sys.executable, '-c', program],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), 'DataLoader')

    def test_exp1_forwards_outer_seed_to_every_refiner_run(self):
        class DummyMoment:
            def __init__(self, *args, **kwargs):
                pass

            def to(self, _device):
                return self

            def impute_numpy(self, observed, _mask, batch_size=4):
                return observed.copy()

        seen_seeds = []

        def fake_eval(*args, **kwargs):
            seen_seeds.append(kwargs['seed'])
            return {'MAE': 1.0, 'MSE': 1.0, 'RMSE': 1.0, 'MRE': 1.0}

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'data.npz')
            x = np.linspace(0.0, 1.0, 160, dtype=np.float32).reshape(80, 2)
            np.savez(path, X=x)

            with mock.patch.object(exp1, 'MOMENTImputer', DummyMoment), \
                    mock.patch.object(exp1, 'eval_gan_refiner', fake_eval), \
                    contextlib.redirect_stdout(io.StringIO()):
                exp1.run_dataset_experiment(
                    'tiny', path, seq_len=4, missing_rate=0.25, seed=123,
                )

        self.assertEqual(seen_seeds, [123, 123, 123])


if __name__ == '__main__':
    unittest.main()
