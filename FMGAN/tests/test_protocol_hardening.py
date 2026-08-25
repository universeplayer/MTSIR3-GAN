"""Focused offline tests for FMGAN scaling and reproducibility metadata."""

import json
import os
import random
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch


FMGAN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if FMGAN_ROOT not in sys.path:
    sys.path.insert(0, FMGAN_ROOT)

from data.unified_loader import (  # noqa: E402
    TimeSeriesImputationDataset,
    fit_standard_scaler,
)
from protocol import (  # noqa: E402
    SOURCE_CODE_ALLOWLIST,
    build_evaluation_evidence,
    build_protocol_manifest,
    make_dataloader_generator,
    seed_dataloader_worker,
    seed_everything,
    sha256_file,
    source_code_record,
    write_json_atomic,
)


class TrainOnlyScalingTests(unittest.TestCase):
    def test_validation_uses_training_statistics_without_refit(self):
        train = np.array([[0.0], [2.0], [4.0], [6.0]], dtype=np.float32)
        validation = np.array(
            [[100.0], [102.0], [104.0], [106.0]], dtype=np.float32,
        )
        scaler = fit_standard_scaler(train)
        original_mean = scaler.mean_.copy()

        train_ds = TimeSeriesImputationDataset(
            train, seq_len=2, stride=2, missing_rate=0.0,
            seed=1, scaler=scaler,
        )
        val_ds = TimeSeriesImputationDataset(
            validation, seq_len=2, stride=2, missing_rate=0.0,
            seed=2, scaler=scaler,
        )

        np.testing.assert_array_equal(scaler.mean_, original_mean)
        self.assertIs(train_ds.scaler, scaler)
        self.assertIs(val_ds.scaler, scaler)
        self.assertEqual(train_ds.scaler_source, 'external_train_fit')
        self.assertEqual(val_ds.scaler_source, 'external_train_fit')
        self.assertAlmostEqual(float(train_ds.samples.mean()), 0.0, places=6)
        # A validation-fitted scaler would center this near zero. A large value
        # demonstrates that only training statistics were applied.
        self.assertGreater(float(val_ds.samples.mean()), 40.0)

    def test_omitted_scaler_keeps_legacy_standalone_behavior(self):
        values = np.arange(8, dtype=np.float32).reshape(-1, 1)
        dataset = TimeSeriesImputationDataset(
            values, seq_len=2, stride=2, missing_rate=0.0, seed=3,
        )
        self.assertEqual(dataset.scaler_source, 'local_legacy_fit')
        self.assertAlmostEqual(float(dataset.samples.mean()), 0.0, places=6)

    def test_all_nan_training_feature_fails_closed(self):
        train = np.array([
            [np.nan, 1.0],
            [np.nan, 2.0],
            [np.nan, 3.0],
        ], dtype=np.float32)

        with self.assertRaisesRegex(
            ValueError, 'all-NaN training feature indices: 0',
        ):
            fit_standard_scaler(train)

    def test_nonfinite_external_scaler_fails_before_mask_creation(self):
        train = np.array([
            [np.nan, 1.0],
            [np.nan, 2.0],
        ], dtype=np.float32)
        # Fit directly to emulate an old caller bypassing the hardened helper.
        from sklearn.preprocessing import StandardScaler
        with np.errstate(invalid='ignore', divide='ignore'):
            scaler = StandardScaler().fit(train)

        validation = np.array([[10.0, 3.0], [11.0, 4.0]], dtype=np.float32)
        with self.assertRaisesRegex(ValueError, 'non-finite statistics'):
            TimeSeriesImputationDataset(
                validation, seq_len=2, stride=2, missing_rate=0.0,
                seed=4, scaler=scaler,
            )


class DeterminismTests(unittest.TestCase):
    def test_legacy_warn_only_and_formal_fail_closed_modes_are_explicit(self):
        with mock.patch('protocol.torch.use_deterministic_algorithms') as setter:
            legacy = seed_everything(10, deterministic=True)
        setter.assert_called_once_with(True, warn_only=True)
        self.assertTrue(legacy['deterministic_warn_only'])

        with mock.patch('protocol.torch.use_deterministic_algorithms') as setter:
            formal = seed_everything(
                10, deterministic=True, deterministic_warn_only=False,
            )
        setter.assert_called_once_with(True, warn_only=False)
        self.assertFalse(formal['deterministic_warn_only'])

    def test_same_mask_seed_is_identical_and_different_seed_changes_mask(self):
        values = np.arange(80, dtype=np.float32).reshape(40, 2)
        scaler = fit_standard_scaler(values[:24])

        first = TimeSeriesImputationDataset(
            values, seq_len=4, stride=4, missing_rate=0.4,
            seed=9, scaler=scaler,
        )
        repeated = TimeSeriesImputationDataset(
            values, seq_len=4, stride=4, missing_rate=0.4,
            seed=9, scaler=scaler,
        )
        changed = TimeSeriesImputationDataset(
            values, seq_len=4, stride=4, missing_rate=0.4,
            seed=10, scaler=scaler,
        )

        np.testing.assert_array_equal(first.combined_mask, repeated.combined_mask)
        self.assertFalse(np.array_equal(first.combined_mask, changed.combined_mask))

    def test_python_numpy_torch_and_dataloader_repeat(self):
        def sample(seed):
            seed_everything(seed, deterministic=True)
            python_value = random.random()
            numpy_value = np.random.random()
            torch_value = torch.rand(1).item()
            loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(torch.arange(12)),
                batch_size=3, shuffle=True,
                generator=make_dataloader_generator(seed),
            )
            order = torch.cat([batch[0] for batch in loader]).tolist()
            return python_value, numpy_value, torch_value, order

        self.assertEqual(sample(1234), sample(1234))
        self.assertNotEqual(sample(1234), sample(1235))

    def test_worker_seed_repeats_python_and_numpy_streams(self):
        def worker_sample(seed):
            torch.manual_seed(seed)
            seed_dataloader_worker(0)
            return random.random(), np.random.random()

        self.assertEqual(worker_sample(77), worker_sample(77))
        self.assertNotEqual(worker_sample(77), worker_sample(78))

    def test_repeated_seeding_does_not_relabel_runtime_env_as_startup_state(self):
        first = seed_everything(2468, deterministic=True)
        repeated = seed_everything(2468, deterministic=True)
        self.assertEqual(
            first['python_hash_seed_at_process_start'],
            repeated['python_hash_seed_at_process_start'],
        )
        self.assertEqual(
            first['python_hash_seed_matches_run'],
            repeated['python_hash_seed_matches_run'],
        )


class ManifestTests(unittest.TestCase):
    def test_manifest_hash_is_stable_and_binds_config_and_masks(self):
        values = np.arange(48, dtype=np.float32).reshape(24, 2)
        train = values[:12]
        validation = values[12:18]
        test = values[18:]
        scaler = fit_standard_scaler(train)
        split_seeds = {'train': 5, 'val': 6, 'test': 7}

        def make_datasets(test_seed=7):
            return {
                'train': TimeSeriesImputationDataset(
                    train, seq_len=3, stride=3, missing_rate=0.25,
                    seed=5, scaler=scaler,
                ),
                'val': TimeSeriesImputationDataset(
                    validation, seq_len=3, stride=3, missing_rate=0.25,
                    seed=6, scaler=scaler,
                ),
                'test': TimeSeriesImputationDataset(
                    test, seq_len=3, stride=3, missing_rate=0.25,
                    seed=test_seed, scaler=scaler,
                ),
            }

        raw_splits = {
            'train': (0, 12, train),
            'val': (12, 18, validation),
            'test': (18, 24, test),
        }
        seed_record = {
            'base_seed': 5,
            'python_random_seed': 5,
            'numpy_seed': 5,
            'torch_seed': 5,
            'dataloader_seed': 5,
            'deterministic_algorithms': True,
        }

        with tempfile.TemporaryDirectory() as directory:
            dataset_path = os.path.join(directory, 'data.npz')
            np.savez(dataset_path, X=values)

            first = build_protocol_manifest(
                dataset_path, raw_splits, make_datasets(), scaler,
                {'epochs': 1, 'dataset': 'tiny'}, seed_record, split_seeds,
            )
            reordered = build_protocol_manifest(
                dataset_path, raw_splits, make_datasets(), scaler,
                {'dataset': 'tiny', 'epochs': 1}, seed_record, split_seeds,
            )
            changed_config = build_protocol_manifest(
                dataset_path, raw_splits, make_datasets(), scaler,
                {'epochs': 2, 'dataset': 'tiny'}, seed_record, split_seeds,
            )
            changed_masks = build_protocol_manifest(
                dataset_path, raw_splits, make_datasets(test_seed=8), scaler,
                {'epochs': 1, 'dataset': 'tiny'}, seed_record,
                {'train': 5, 'val': 6, 'test': 8},
            )

        self.assertEqual(first['manifest_sha256'], reordered['manifest_sha256'])
        self.assertNotEqual(
            first['manifest_sha256'], changed_config['manifest_sha256'],
        )
        self.assertNotEqual(
            first['manifest_sha256'], changed_masks['manifest_sha256'],
        )
        self.assertEqual(first['scaler']['fitted_on'], 'train_only')
        self.assertEqual(first['splits']['test']['mask_seed'], 7)
        self.assertEqual(
            frozenset(first['source_code']['files']),
            frozenset(SOURCE_CODE_ALLOWLIST),
        )
        for relative_path in SOURCE_CODE_ALLOWLIST:
            expected_path = os.path.join(FMGAN_ROOT, relative_path)
            self.assertEqual(
                first['source_code']['files'][relative_path]['sha256'],
                sha256_file(expected_path),
            )

        with tempfile.TemporaryDirectory() as directory:
            output = os.path.join(directory, 'protocol_manifest.json')
            first_file_hash = write_json_atomic(output, first)
            second_file_hash = write_json_atomic(output, first)
            with open(output, encoding='utf-8') as handle:
                on_disk = json.load(handle)
        self.assertEqual(first_file_hash, second_file_hash)
        self.assertEqual(on_disk['manifest_sha256'], first['manifest_sha256'])

    def test_source_allowlist_fails_closed_when_root_is_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'missing or symlinked'):
                source_code_record(directory)


class EvaluationEvidenceTests(unittest.TestCase):
    def test_records_exact_hashes_shapes_and_finite_score_counts(self):
        shape = (2, 3, 1)
        coarse = np.zeros(shape, dtype=np.float32)
        prediction = np.full(shape, 0.5, dtype=np.float32)
        target = np.ones(shape, dtype=np.float32)
        mask = np.array(
            [[[1.0], [0.0], [1.0]], [[0.0], [1.0], [0.0]]],
            dtype=np.float32,
        )

        evidence = build_evaluation_evidence(
            coarse, prediction, target, mask,
        )

        self.assertEqual(evidence['score_count'], 3)
        self.assertEqual(
            set(evidence['arrays']),
            {'coarse', 'prediction', 'target', 'indicating_mask'},
        )
        for record in evidence['arrays'].values():
            self.assertEqual(record['shape'], [2, 3, 1])
            self.assertEqual(record['element_count'], 6)
            self.assertEqual(record['finite_count'], 6)
            self.assertEqual(record['score_count'], 3)
            self.assertEqual(record['finite_score_count'], 3)
            self.assertEqual(len(record['sha256']), 64)

    def test_nonfinite_prediction_fails_before_result_is_written(self):
        shape = (1, 2, 1)
        coarse = np.zeros(shape, dtype=np.float32)
        prediction = np.zeros(shape, dtype=np.float32)
        prediction[0, 1, 0] = np.nan
        target = np.ones(shape, dtype=np.float32)
        mask = np.array([[[1.0], [0.0]]], dtype=np.float32)

        with self.assertRaisesRegex(ValueError, 'prediction contains non-finite'):
            build_evaluation_evidence(coarse, prediction, target, mask)

    def test_nonbinary_or_empty_score_mask_fails_closed(self):
        values = np.ones((1, 2, 1), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, 'must be binary'):
            build_evaluation_evidence(
                values, values, values,
                np.array([[[1.0], [0.5]]], dtype=np.float32),
            )
        with self.assertRaisesRegex(ValueError, 'zero scored positions'):
            build_evaluation_evidence(
                values, values, values, np.zeros_like(values),
            )


if __name__ == '__main__':
    unittest.main()
