"""Offline tests for the fail-closed paired-run acceptance gate."""

import json
import math
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch


FMGAN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VALIDATOR_PATH = os.path.join(
    FMGAN_ROOT, 'evaluation', 'validate_paired_runs.py',
)
if FMGAN_ROOT not in sys.path:
    sys.path.insert(0, FMGAN_ROOT)

from evaluation.validate_paired_runs import (  # noqa: E402
    PairedRunValidationError,
    validate_paired_runs,
)
from protocol import (  # noqa: E402
    PROTOCOL_MANIFEST_VERSION,
    SOURCE_CODE_ALLOWLIST,
    SOURCE_CODE_ALLOWLIST_VERSION,
    build_evaluation_evidence,
    canonical_json,
    derive_paired_random_stream_seeds,
    sha256_bytes,
    sha256_file,
)


DIGEST_A = 'a' * 64
DIGEST_B = 'b' * 64
DIGEST_C = 'c' * 64
DIGEST_D = 'd' * 64


def write_json(path, payload):
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write('\n')


def make_run(root, mode, config_overrides=None, test_cardinality=3,
             source_digest=DIGEST_A, deterministic_warn_only=False):
    run_dir = os.path.join(root, mode)
    os.makedirs(run_dir)
    config = {
        'dataset': 'Tiny',
        'seq_len': 4,
        'missing_rate': 0.25,
        'coarse': 'linear',
        'noise_dim': 2,
        'width': 8,
        'n_stages': 1,
        'n_blocks': 1,
        'cardinality': 2,
        'freq_branch': mode == 'adversarial',
        'epochs': 2,
        'batch_size': 2,
        'seed': 42,
        'num_workers': 0,
        'deterministic': True,
        'lr': 0.0002,
        'gamma': 0.05 if mode == 'adversarial' else 999.0,
        'lambda_recon': 10.0,
        'lambda_freq': 1.0,
        'training_mode': mode,
        'ema': True,
        'augment': False,
        'stride_divisor': 2,
        'outdir': run_dir,
        'log_every': 1,
        'device': 'cpu',
        'resolved_device': 'cpu',
        'resolved_training_mode': mode,
    }
    if config_overrides:
        config.update(config_overrides)
    epochs = config['epochs']
    streams = derive_paired_random_stream_seeds(config['seed'])
    seeds = {
        'base_seed': config['seed'],
        'python_random_seed': config['seed'],
        'numpy_seed': config['seed'],
        'torch_seed': config['seed'],
        'cuda_seed': None,
        'dataloader_seed': config['seed'],
        'deterministic_algorithms': True,
        'deterministic_warn_only': deterministic_warn_only,
        'python_hash_seed_at_process_start': None,
        'python_hash_seed_matches_run': False,
        'paired_random_stream_version': 1,
        'paired_random_stream_implementation': (
            'cpu_torch_generator_then_device_transfer'
        ),
        'torch_version': str(torch.__version__),
        'paired_random_streams': streams,
        'fixed_evaluation_noise': {
            'validation_shape': [2, config['seq_len'], config['noise_dim']],
            'validation_sha256': DIGEST_C,
            'test_shape': [
                test_cardinality, config['seq_len'], config['noise_dim'],
            ],
            'test_sha256': DIGEST_D,
        },
    }

    def split(start, stop, count, raw_digest, mask_digest, mask_seed):
        return {
            'start': start,
            'stop': stop,
            'raw_shape': [stop - start, 1],
            'raw_sha256': raw_digest,
            'window_count': count,
            'seq_len': config['seq_len'],
            'stride': config['seq_len'],
            'missing_rate': config['missing_rate'],
            'missing_pattern': 'point',
            'mask_seed': mask_seed,
            'original_mask_sha256': DIGEST_A,
            'combined_mask_sha256': mask_digest,
            'indicating_mask_sha256': DIGEST_D,
        }

    manifest = {
        'protocol_manifest_version': PROTOCOL_MANIFEST_VERSION,
        'dataset_source': {
            'filename': 'data.npz',
            'size_bytes': 128,
            'sha256': DIGEST_A,
        },
        'splits': {
            'train': split(0, 8, 4, DIGEST_A, DIGEST_B, config['seed']),
            'val': split(8, 12, 2, DIGEST_B, DIGEST_C, config['seed'] + 1),
            'test': split(
                12, 18, test_cardinality, DIGEST_C, DIGEST_D,
                config['seed'] + 2,
            ),
        },
        'scaler': {
            'class': 'StandardScaler',
            'fitted_on': 'train_only',
            'state_sha256': DIGEST_B,
            'mean': [0.0],
            'scale': [1.0],
            'var': [1.0],
            'n_features_in': 1,
            'n_samples_seen': 8,
        },
        'seeds': seeds,
        'source_code': {
            'allowlist_version': SOURCE_CODE_ALLOWLIST_VERSION,
            'files': {
                path: {'size_bytes': 128, 'sha256': source_digest}
                for path in SOURCE_CODE_ALLOWLIST
            },
        },
        'config': config,
        'config_sha256': sha256_bytes(
            canonical_json(config).encode('ascii'),
        ),
    }
    manifest['manifest_sha256'] = sha256_bytes(
        canonical_json(manifest).encode('ascii'),
    )
    manifest_path = os.path.join(run_dir, 'protocol_manifest.json')
    write_json(manifest_path, manifest)

    coarse_metrics = {'MAE': 1.0, 'MSE': 4.0, 'RMSE': 2.0, 'MRE': 0.1}
    refined_mae = 0.8 if mode == 'adversarial' else 0.7
    refined_metrics = {
        'MAE': refined_mae,
        'MSE': refined_mae ** 2,
        'RMSE': refined_mae,
        'MRE': refined_mae / 10,
    }
    raw_arg_names = set(config) - {'resolved_device', 'resolved_training_mode'}
    shape = (test_cardinality, config['seq_len'], 1)
    indicating_mask = np.zeros(shape, dtype=np.float32)
    indicating_mask[:, 0, 0] = 1.0
    target = np.ones(shape, dtype=np.float32)
    coarse = np.zeros(shape, dtype=np.float32)
    prediction = np.full(
        shape, 0.8 if mode == 'adversarial' else 0.7,
        dtype=np.float32,
    )
    result = {
        'dataset': config['dataset'],
        'coarse_method': config['coarse'],
        'coarse_metrics': coarse_metrics,
        'refined_metrics': refined_metrics,
        'improvement_pct': round((1.0 - refined_mae) * 100, 2),
        'best_val_mae': 0.25,
        'evaluation_evidence': build_evaluation_evidence(
            coarse, prediction, target, indicating_mask,
        ),
        'paired_random_streams': streams,
        'protocol_manifest': {
            'path': 'protocol_manifest.json',
            'manifest_sha256': manifest['manifest_sha256'],
            'file_sha256': sha256_file(manifest_path),
        },
        'args': {name: config[name] for name in raw_arg_names},
    }
    write_json(os.path.join(run_dir, 'results.json'), result)

    log = []
    for epoch in range(1, epochs + 1):
        entry = {
            'g_adv': 0.2 if mode == 'adversarial' else 0.0,
            'g_recon': 0.4,
            'g_freq': 0.1,
            'val_MAE': 0.5 if epoch < epochs else 0.25,
            'val_MSE': 0.25,
            'val_RMSE': 0.5,
            'val_MRE': 0.2,
            'epoch': epoch,
        }
        if mode == 'adversarial':
            entry.update({'d_adv': 0.3, 'd_r1': 0.01, 'd_r2': 0.02})
        log.append(entry)
    write_json(os.path.join(run_dir, 'training_log.json'), log)

    checkpoint = {
        'G': {'weight': torch.tensor([1.0])},
        'G_ema': {'weight': torch.tensor([1.0])},
        'D': ({'weight': torch.tensor([2.0])}
              if mode == 'adversarial' else None),
        'training_mode': mode,
        'seed': config['seed'],
        'paired_random_streams': streams,
        'protocol_manifest_sha256': manifest['manifest_sha256'],
        'epoch': epochs - 1,
        'val_mae': 0.25,
    }
    torch.save(checkpoint, os.path.join(run_dir, 'best_model.pt'))
    return run_dir


class PairedRunValidatorTests(unittest.TestCase):
    def test_cli_returns_machine_readable_valid_report(self):
        with tempfile.TemporaryDirectory() as directory:
            adversarial = make_run(directory, 'adversarial')
            reconstruction = make_run(directory, 'reconstruction_only')
            completed = subprocess.run(
                [sys.executable, VALIDATOR_PATH, adversarial, reconstruction],
                check=False, capture_output=True, text=True,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        report = json.loads(completed.stdout)
        self.assertTrue(report['valid'])
        self.assertEqual(report['test_cardinality'], 3)

    def test_valid_pair_reports_paired_delta_and_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            adversarial = make_run(directory, 'adversarial')
            reconstruction = make_run(directory, 'reconstruction_only')
            report = validate_paired_runs(adversarial, reconstruction)

        self.assertTrue(report['valid'])
        self.assertEqual(report['validation_report_version'], 2)
        self.assertEqual(report['protocol_manifest_version'], 3)
        self.assertEqual(len(report['validator_sha256']), 64)
        self.assertEqual(report['epochs'], 2)
        self.assertEqual(report['device'], 'cpu')
        self.assertEqual(report['test_cardinality'], 3)
        self.assertAlmostEqual(
            report['delta_MAE_adversarial_minus_reconstruction_only'],
            0.1,
        )
        self.assertEqual(
            len(report['runs']['adversarial']['checkpoint_sha256']), 64,
        )

    def test_stale_checkpoint_manifest_binding_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            adversarial = make_run(directory, 'adversarial')
            reconstruction = make_run(directory, 'reconstruction_only')
            checkpoint_path = os.path.join(adversarial, 'best_model.pt')
            checkpoint = torch.load(
                checkpoint_path, map_location='cpu', weights_only=True,
            )
            checkpoint['protocol_manifest_sha256'] = DIGEST_D
            torch.save(checkpoint, checkpoint_path)

            with self.assertRaisesRegex(
                    PairedRunValidationError, 'checkpoint-to-manifest binding'):
                validate_paired_runs(adversarial, reconstruction)

    def test_nonfinite_result_metric_is_rejected_at_json_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            adversarial = make_run(directory, 'adversarial')
            reconstruction = make_run(directory, 'reconstruction_only')
            result_path = os.path.join(reconstruction, 'results.json')
            with open(result_path, encoding='utf-8') as handle:
                result = json.load(handle)
            result['refined_metrics']['MAE'] = math.nan
            write_json(result_path, result)

            with self.assertRaisesRegex(
                    PairedRunValidationError, 'non-finite JSON constant'):
                validate_paired_runs(adversarial, reconstruction)

    def test_truncated_result_args_cannot_weaken_manifest_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            adversarial = make_run(directory, 'adversarial')
            reconstruction = make_run(directory, 'reconstruction_only')
            result_path = os.path.join(reconstruction, 'results.json')
            with open(result_path, encoding='utf-8') as handle:
                result = json.load(handle)
            result['args'] = {'training_mode': 'reconstruction_only'}
            write_json(result_path, result)

            with self.assertRaisesRegex(
                    PairedRunValidationError, 'args-to-manifest binding'):
                validate_paired_runs(adversarial, reconstruction)

    def test_warn_only_determinism_record_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            adversarial = make_run(
                directory, 'adversarial', deterministic_warn_only=True,
            )
            reconstruction = make_run(directory, 'reconstruction_only')

            with self.assertRaisesRegex(
                    PairedRunValidationError, 'deterministic_warn_only must be false'):
                validate_paired_runs(adversarial, reconstruction)

    def test_source_allowlist_hash_drift_is_rejected_between_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            adversarial = make_run(directory, 'adversarial')
            reconstruction = make_run(
                directory, 'reconstruction_only', source_digest=DIGEST_B,
            )

            with self.assertRaisesRegex(
                    PairedRunValidationError, 'shared protocol manifest fields differ'):
                validate_paired_runs(adversarial, reconstruction)

    def test_shared_non_train_scaler_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            adversarial = make_run(directory, 'adversarial')
            reconstruction = make_run(directory, 'reconstruction_only')
            for run_dir in (adversarial, reconstruction):
                manifest_path = os.path.join(run_dir, 'protocol_manifest.json')
                with open(manifest_path, encoding='utf-8') as handle:
                    manifest = json.load(handle)
                manifest['scaler']['fitted_on'] = 'full_dataset'
                manifest_without_hash = dict(manifest)
                manifest_without_hash.pop('manifest_sha256')
                manifest['manifest_sha256'] = sha256_bytes(
                    canonical_json(manifest_without_hash).encode('ascii'),
                )
                write_json(manifest_path, manifest)

            with self.assertRaisesRegex(
                    PairedRunValidationError, 'fitted_on mismatch'):
                validate_paired_runs(adversarial, reconstruction)

    def test_shared_wrong_mask_seed_derivation_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            adversarial = make_run(directory, 'adversarial')
            reconstruction = make_run(directory, 'reconstruction_only')
            for run_dir in (adversarial, reconstruction):
                manifest_path = os.path.join(run_dir, 'protocol_manifest.json')
                with open(manifest_path, encoding='utf-8') as handle:
                    manifest = json.load(handle)
                manifest['splits']['test']['mask_seed'] = 999
                manifest_without_hash = dict(manifest)
                manifest_without_hash.pop('manifest_sha256')
                manifest['manifest_sha256'] = sha256_bytes(
                    canonical_json(manifest_without_hash).encode('ascii'),
                )
                write_json(manifest_path, manifest)

            with self.assertRaisesRegex(
                    PairedRunValidationError, 'mask_seed derivation'):
                validate_paired_runs(adversarial, reconstruction)

    def test_shared_noncontiguous_raw_splits_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            adversarial = make_run(directory, 'adversarial')
            reconstruction = make_run(directory, 'reconstruction_only')
            for run_dir in (adversarial, reconstruction):
                manifest_path = os.path.join(run_dir, 'protocol_manifest.json')
                with open(manifest_path, encoding='utf-8') as handle:
                    manifest = json.load(handle)
                manifest['splits']['val']['start'] = 9
                manifest['splits']['val']['raw_shape'][0] = 3
                manifest_without_hash = dict(manifest)
                manifest_without_hash.pop('manifest_sha256')
                manifest['manifest_sha256'] = sha256_bytes(
                    canonical_json(manifest_without_hash).encode('ascii'),
                )
                write_json(manifest_path, manifest)

            with self.assertRaisesRegex(
                    PairedRunValidationError, 'train/validation split boundary'):
                validate_paired_runs(adversarial, reconstruction)

    def test_nonfinite_prediction_count_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            adversarial = make_run(directory, 'adversarial')
            reconstruction = make_run(directory, 'reconstruction_only')
            result_path = os.path.join(adversarial, 'results.json')
            with open(result_path, encoding='utf-8') as handle:
                result = json.load(handle)
            prediction = result['evaluation_evidence']['arrays']['prediction']
            prediction['finite_count'] -= 1
            write_json(result_path, result)

            with self.assertRaisesRegex(
                    PairedRunValidationError, 'full-array finite count'):
                validate_paired_runs(adversarial, reconstruction)

    def test_shared_target_mask_and_coarse_evidence_must_match(self):
        with tempfile.TemporaryDirectory() as directory:
            adversarial = make_run(directory, 'adversarial')
            reconstruction = make_run(directory, 'reconstruction_only')
            result_path = os.path.join(reconstruction, 'results.json')
            with open(result_path, encoding='utf-8') as handle:
                result = json.load(handle)
            result['evaluation_evidence']['arrays']['coarse']['sha256'] = DIGEST_B
            write_json(result_path, result)

            with self.assertRaisesRegex(
                    PairedRunValidationError, 'shared test coarse evidence'):
                validate_paired_runs(adversarial, reconstruction)

    def test_actual_epoch_count_must_match_and_be_paired(self):
        with tempfile.TemporaryDirectory() as directory:
            adversarial = make_run(directory, 'adversarial')
            reconstruction = make_run(
                directory, 'reconstruction_only', {'epochs': 3},
            )

            with self.assertRaisesRegex(PairedRunValidationError, 'epochs mismatch'):
                validate_paired_runs(adversarial, reconstruction)

    def test_test_cardinality_must_match(self):
        with tempfile.TemporaryDirectory() as directory:
            adversarial = make_run(directory, 'adversarial')
            reconstruction = make_run(
                directory, 'reconstruction_only', test_cardinality=4,
            )

            with self.assertRaisesRegex(
                    PairedRunValidationError, 'test cardinality mismatch'):
                validate_paired_runs(adversarial, reconstruction)

    def test_reconstruction_checkpoint_cannot_contain_discriminator(self):
        with tempfile.TemporaryDirectory() as directory:
            adversarial = make_run(directory, 'adversarial')
            reconstruction = make_run(directory, 'reconstruction_only')
            checkpoint_path = os.path.join(reconstruction, 'best_model.pt')
            checkpoint = torch.load(
                checkpoint_path, map_location='cpu', weights_only=True,
            )
            checkpoint['D'] = {'weight': torch.tensor([2.0])}
            torch.save(checkpoint, checkpoint_path)

            with self.assertRaisesRegex(
                    PairedRunValidationError, 'must be null'):
                validate_paired_runs(adversarial, reconstruction)

    def test_shared_protocol_field_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            adversarial = make_run(directory, 'adversarial')
            reconstruction = make_run(
                directory, 'reconstruction_only', {'lambda_recon': 11.0},
            )

            with self.assertRaisesRegex(
                    PairedRunValidationError, 'shared protocol manifest fields differ'):
                validate_paired_runs(adversarial, reconstruction)

    def test_short_training_log_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            adversarial = make_run(directory, 'adversarial')
            reconstruction = make_run(directory, 'reconstruction_only')
            log_path = os.path.join(adversarial, 'training_log.json')
            with open(log_path, encoding='utf-8') as handle:
                log = json.load(handle)
            write_json(log_path, log[:-1])

            with self.assertRaisesRegex(
                    PairedRunValidationError, 'completed epochs mismatch'):
                validate_paired_runs(adversarial, reconstruction)


if __name__ == '__main__':
    unittest.main()
