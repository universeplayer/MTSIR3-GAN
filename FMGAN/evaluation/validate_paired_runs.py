#!/usr/bin/env python3
"""Fail-closed validation for one FMGAN paired experiment.

The validator accepts exactly one adversarial run directory and one
reconstruction-only run directory.  It does not run either model.  Instead it
checks that the saved protocol manifest, result, training log, and best
checkpoint form a self-consistent record and that the two records differ only
where the treatment requires them to differ.

Usage:
    python FMGAN/evaluation/validate_paired_runs.py \
        FMGAN/results/paired_v1/Weather/seed_42/adversarial \
        FMGAN/results/paired_v1/Weather/seed_42/reconstruction_only
"""

import argparse
import copy
import json
import math
import os
import sys
from collections.abc import Mapping

import torch


_FMGAN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _FMGAN_ROOT not in sys.path:
    sys.path.insert(0, _FMGAN_ROOT)

from protocol import (  # noqa: E402
    EVALUATION_EVIDENCE_VERSION,
    PAIRED_RANDOM_STREAM_OFFSETS,
    PROTOCOL_MANIFEST_VERSION,
    SOURCE_CODE_ALLOWLIST,
    SOURCE_CODE_ALLOWLIST_VERSION,
    canonical_json,
    derive_seed,
    derive_paired_random_stream_seeds,
    sha256_bytes,
    sha256_file,
)


EXPECTED_MODES = ('adversarial', 'reconstruction_only')
VALIDATION_REPORT_VERSION = 2
EXPECTED_METRICS = frozenset({'MAE', 'MSE', 'RMSE', 'MRE'})
EXPECTED_EVALUATION_ARRAYS = frozenset({
    'coarse', 'prediction', 'target', 'indicating_mask',
})
MODE_CONFIG_FIELDS = frozenset({
    'outdir',
    'resolved_training_mode',
    'training_mode',
})
RESOLVED_CONFIG_FIELDS = frozenset({
    'resolved_device',
    'resolved_training_mode',
})
# These settings affect only a discriminator that reconstruction-only mode does
# not instantiate.  Keeping this allowlist explicit makes new or misspelled
# fields fail closed rather than being silently discarded.
DISCRIMINATOR_ONLY_CONFIG_FIELDS = frozenset({'freq_branch', 'gamma'})
PAIR_EXEMPT_CONFIG_FIELDS = MODE_CONFIG_FIELDS | DISCRIMINATOR_ONLY_CONFIG_FIELDS
DISCRIMINATOR_LOG_FIELDS = frozenset({'d_adv', 'd_r1', 'd_r2'})
HEX_DIGITS = frozenset('0123456789abcdef')


class PairedRunValidationError(RuntimeError):
    """Raised when a run pair cannot prove that it is protocol matched."""


def _fail(message):
    raise PairedRunValidationError(message)


def _strict_json(path, label):
    if os.path.islink(path):
        _fail(f'{label} must not be a symlink: {path}')
    if not os.path.isfile(path):
        _fail(f'missing {label}: {path}')

    def reject_constant(value):
        _fail(f'{label} contains non-finite JSON constant {value!r}')

    try:
        with open(path, encoding='utf-8') as handle:
            payload = json.load(handle, parse_constant=reject_constant)
    except PairedRunValidationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        _fail(f'cannot read strict {label}: {exc}')
    if not isinstance(payload, Mapping):
        _fail(f'{label} must be a JSON object')
    return dict(payload)


def _mapping(value, label):
    if not isinstance(value, Mapping):
        _fail(f'{label} must be an object')
    return value


def _finite_number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f'{label} must be a finite number')
    if not math.isfinite(float(value)):
        _fail(f'{label} must be finite')
    return float(value)


def _positive_int(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        _fail(f'{label} must be a positive integer')
    return value


def _nonnegative_int(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _fail(f'{label} must be a non-negative integer')
    return value


def _sha256_hex(value, label):
    if (not isinstance(value, str) or len(value) != 64 or
            any(character not in HEX_DIGITS for character in value)):
        _fail(f'{label} must be a lowercase SHA-256 digest')
    return value


def _equal(left, right, label):
    if left != right:
        _fail(f'{label} mismatch: {left!r} != {right!r}')


def _validate_metric_dict(value, label):
    metrics = _mapping(value, label)
    if frozenset(metrics) != EXPECTED_METRICS:
        _fail(
            f'{label} metric definitions mismatch: '
            f'{sorted(metrics)} != {sorted(EXPECTED_METRICS)}'
        )
    validated = {
        name: _finite_number(metrics[name], f'{label}.{name}')
        for name in sorted(EXPECTED_METRICS)
    }
    if validated['MSE'] < 0:
        _fail(f'{label}.MSE must be non-negative')
    if not math.isclose(
            validated['RMSE'], math.sqrt(validated['MSE']),
            rel_tol=1e-6, abs_tol=1e-8):
        _fail(f'{label}.RMSE is inconsistent with sqrt(MSE)')
    return validated


def _finite_vector(value, expected_size, label, *, positive=False,
                   nonnegative=False):
    if not isinstance(value, list) or len(value) != expected_size:
        _fail(f'{label} must contain exactly {expected_size} values')
    validated = [
        _finite_number(item, f'{label}[{index}]')
        for index, item in enumerate(value)
    ]
    if positive and any(item <= 0 for item in validated):
        _fail(f'{label} values must be positive')
    if nonnegative and any(item < 0 for item in validated):
        _fail(f'{label} values must be non-negative')
    return validated


def _validate_scaler(value, train_split, label):
    scaler = _mapping(value, label)
    expected_fields = frozenset({
        'class', 'fitted_on', 'state_sha256', 'mean', 'scale', 'var',
        'n_features_in', 'n_samples_seen',
    })
    if frozenset(scaler) != expected_fields:
        _fail(f'{label} has an unexpected schema')
    _equal(scaler.get('class'), 'StandardScaler', f'{label}.class')
    _equal(scaler.get('fitted_on'), 'train_only', f'{label}.fitted_on')
    _sha256_hex(scaler.get('state_sha256'), f'{label}.state_sha256')

    n_features = _positive_int(
        scaler.get('n_features_in'), f'{label}.n_features_in',
    )
    _equal(
        n_features, train_split['raw_shape'][1],
        f'{label} feature count versus raw training split',
    )
    _finite_vector(scaler.get('mean'), n_features, f'{label}.mean')
    _finite_vector(
        scaler.get('scale'), n_features, f'{label}.scale', positive=True,
    )
    _finite_vector(
        scaler.get('var'), n_features, f'{label}.var', nonnegative=True,
    )

    observed_counts = scaler.get('n_samples_seen')
    if isinstance(observed_counts, int) and not isinstance(observed_counts, bool):
        observed_counts = [observed_counts] * n_features
    elif isinstance(observed_counts, list):
        if len(observed_counts) != n_features:
            _fail(f'{label}.n_samples_seen must match the feature count')
    else:
        _fail(f'{label}.n_samples_seen must be an integer or per-feature list')
    train_timesteps = train_split['raw_shape'][0]
    for index, count in enumerate(observed_counts):
        if (isinstance(count, bool) or not isinstance(count, int) or
                not 1 <= count <= train_timesteps):
            _fail(
                f'{label}.n_samples_seen[{index}] must be within the raw '
                'training-timestep count'
            )
    return dict(scaler)


def _validate_numeric_log_entry(entry, label):
    entry = _mapping(entry, label)
    for name, value in entry.items():
        _finite_number(value, f'{label}.{name}')
    return entry


def _validate_state_dict(value, label, allow_none=False):
    if value is None and allow_none:
        return
    state = _mapping(value, label)
    if not state:
        _fail(f'{label} must not be empty')
    for name, tensor in state.items():
        if not isinstance(name, str):
            _fail(f'{label} contains a non-string parameter name')
        if isinstance(tensor, torch.Tensor):
            if not bool(torch.isfinite(tensor).all().item()):
                _fail(f'{label}.{name} contains non-finite checkpoint values')
        elif isinstance(tensor, (int, float)) and not isinstance(tensor, bool):
            _finite_number(tensor, f'{label}.{name}')
        else:
            _fail(f'{label}.{name} has unsupported checkpoint value type')


def _load_checkpoint(path, label):
    if os.path.islink(path):
        _fail(f'{label} must not be a symlink: {path}')
    if not os.path.isfile(path):
        _fail(f'missing {label}: {path}')
    try:
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    except Exception as exc:
        _fail(f'cannot safely load {label}: {exc}')
    if not isinstance(checkpoint, Mapping):
        _fail(f'{label} must contain a metadata dictionary')
    return dict(checkpoint)


def _validate_manifest(manifest, manifest_path, label):
    _equal(
        manifest.get('protocol_manifest_version'), PROTOCOL_MANIFEST_VERSION,
        f'{label}.protocol_manifest_version',
    )
    stated_hash = _sha256_hex(
        manifest.get('manifest_sha256'), f'{label}.manifest_sha256',
    )
    unhashed = copy.deepcopy(manifest)
    unhashed.pop('manifest_sha256', None)
    computed_hash = sha256_bytes(canonical_json(unhashed).encode('ascii'))
    _equal(stated_hash, computed_hash, f'{label} internal manifest hash')

    config = _mapping(manifest.get('config'), f'{label}.config')
    stated_config_hash = _sha256_hex(
        manifest.get('config_sha256'), f'{label}.config_sha256',
    )
    computed_config_hash = sha256_bytes(canonical_json(config).encode('ascii'))
    _equal(stated_config_hash, computed_config_hash, f'{label} config hash')

    dataset_source = _mapping(
        manifest.get('dataset_source'), f'{label}.dataset_source',
    )
    _sha256_hex(
        dataset_source.get('sha256'), f'{label}.dataset_source.sha256',
    )
    _positive_int(
        dataset_source.get('size_bytes'), f'{label}.dataset_source.size_bytes',
    )
    if not isinstance(dataset_source.get('filename'), str):
        _fail(f'{label}.dataset_source.filename must be a string')

    source_code = _mapping(
        manifest.get('source_code'), f'{label}.source_code',
    )
    _equal(
        source_code.get('allowlist_version'), SOURCE_CODE_ALLOWLIST_VERSION,
        f'{label}.source_code.allowlist_version',
    )
    source_files = _mapping(
        source_code.get('files'), f'{label}.source_code.files',
    )
    if frozenset(source_files) != frozenset(SOURCE_CODE_ALLOWLIST):
        _fail(
            f'{label}.source_code.files must match the critical source allowlist'
        )
    for relative_path in SOURCE_CODE_ALLOWLIST:
        record = _mapping(
            source_files[relative_path],
            f'{label}.source_code.files.{relative_path}',
        )
        if frozenset(record) != frozenset({'size_bytes', 'sha256'}):
            _fail(f'{label} source record has an unexpected schema: {relative_path}')
        _positive_int(
            record.get('size_bytes'),
            f'{label}.source_code.files.{relative_path}.size_bytes',
        )
        _sha256_hex(
            record.get('sha256'),
            f'{label}.source_code.files.{relative_path}.sha256',
        )

    splits = _mapping(manifest.get('splits'), f'{label}.splits')
    if frozenset(splits) != frozenset({'train', 'val', 'test'}):
        _fail(f'{label}.splits must contain exactly train, val, and test')
    config_seq_len = _positive_int(
        config.get('seq_len'), f'{label}.config.seq_len',
    )
    config_missing_rate = _finite_number(
        config.get('missing_rate'), f'{label}.config.missing_rate',
    )
    if not 0 <= config_missing_rate < 1:
        _fail(f'{label}.config.missing_rate must be in [0, 1)')
    split_records = {}
    for split_name in ('train', 'val', 'test'):
        split = _mapping(splits[split_name], f'{label}.splits.{split_name}')
        expected_split_fields = frozenset({
            'start', 'stop', 'raw_shape', 'raw_sha256', 'window_count',
            'seq_len', 'stride', 'missing_rate', 'missing_pattern',
            'mask_seed', 'original_mask_sha256', 'combined_mask_sha256',
            'indicating_mask_sha256',
        })
        if frozenset(split) != expected_split_fields:
            _fail(f'{label}.splits.{split_name} has an unexpected schema')
        start = _nonnegative_int(
            split.get('start'), f'{label}.splits.{split_name}.start',
        )
        stop = _positive_int(
            split.get('stop'), f'{label}.splits.{split_name}.stop',
        )
        if stop <= start:
            _fail(f'{label}.splits.{split_name} stop must exceed start')
        raw_shape = split.get('raw_shape')
        if (not isinstance(raw_shape, list) or len(raw_shape) != 2 or
                any(isinstance(item, bool) or not isinstance(item, int) or
                    item <= 0 for item in raw_shape)):
            _fail(
                f'{label}.splits.{split_name}.raw_shape must contain two '
                'positive integers'
            )
        _equal(
            raw_shape[0], stop - start,
            f'{label}.splits.{split_name} raw interval length',
        )
        _positive_int(
            split.get('window_count'),
            f'{label}.splits.{split_name}.window_count',
        )
        _equal(
            _positive_int(
                split.get('seq_len'), f'{label}.splits.{split_name}.seq_len',
            ),
            config_seq_len,
            f'{label}.splits.{split_name} seq_len versus config',
        )
        _positive_int(
            split.get('stride'), f'{label}.splits.{split_name}.stride',
        )
        _equal(
            _finite_number(
                split.get('missing_rate'),
                f'{label}.splits.{split_name}.missing_rate',
            ),
            config_missing_rate,
            f'{label}.splits.{split_name} missing_rate versus config',
        )
        _equal(
            split.get('missing_pattern'), 'point',
            f'{label}.splits.{split_name}.missing_pattern',
        )
        for hash_name in (
                'raw_sha256', 'original_mask_sha256',
                'combined_mask_sha256', 'indicating_mask_sha256'):
            _sha256_hex(
                split.get(hash_name),
                f'{label}.splits.{split_name}.{hash_name}',
            )
        split_records[split_name] = dict(split)

    if split_records['train']['start'] != 0:
        _fail(f'{label}.splits.train.start must be zero')
    _equal(
        split_records['train']['stop'], split_records['val']['start'],
        f'{label} train/validation split boundary',
    )
    _equal(
        split_records['val']['stop'], split_records['test']['start'],
        f'{label} validation/test split boundary',
    )
    raw_feature_counts = {
        split['raw_shape'][1] for split in split_records.values()
    }
    if len(raw_feature_counts) != 1:
        _fail(f'{label} raw split feature counts differ')

    scaler = _validate_scaler(
        manifest.get('scaler'), split_records['train'], f'{label}.scaler',
    )

    seeds = _mapping(manifest.get('seeds'), f'{label}.seeds')
    base_seed = seeds.get('base_seed')
    if isinstance(base_seed, bool) or not isinstance(base_seed, int):
        _fail(f'{label}.seeds.base_seed must be an integer')
    _equal(
        seeds.get('paired_random_stream_version'), 1,
        f'{label}.seeds.paired_random_stream_version',
    )
    if seeds.get('deterministic_algorithms') is not True:
        _fail(f'{label}.seeds.deterministic_algorithms must be true')
    if seeds.get('deterministic_warn_only') is not False:
        _fail(f'{label}.seeds.deterministic_warn_only must be false')
    _equal(
        seeds.get('paired_random_stream_implementation'),
        'cpu_torch_generator_then_device_transfer',
        f'{label}.seeds.paired_random_stream_implementation',
    )
    streams = _mapping(
        seeds.get('paired_random_streams'),
        f'{label}.seeds.paired_random_streams',
    )
    if frozenset(streams) != frozenset(PAIRED_RANDOM_STREAM_OFFSETS):
        _fail(f'{label} has an incomplete or unknown paired random stream set')
    _equal(
        dict(streams), derive_paired_random_stream_seeds(base_seed),
        f'{label} paired random stream derivation',
    )
    for offset, split_name in enumerate(('train', 'val', 'test')):
        _equal(
            split_records[split_name].get('mask_seed'),
            derive_seed(base_seed, offset),
            f'{label}.splits.{split_name}.mask_seed derivation',
        )

    fixed_noise = _mapping(
        seeds.get('fixed_evaluation_noise'),
        f'{label}.seeds.fixed_evaluation_noise',
    )
    for split_name, manifest_split in (
            ('validation', splits['val']), ('test', splits['test'])):
        shape = fixed_noise.get(f'{split_name}_shape')
        expected_shape = [
            manifest_split['window_count'], manifest_split['seq_len'],
            config.get('noise_dim'),
        ]
        _equal(shape, expected_shape, f'{label} fixed {split_name} noise shape')
        _sha256_hex(
            fixed_noise.get(f'{split_name}_sha256'),
            f'{label}.seeds.fixed_evaluation_noise.{split_name}_sha256',
        )

    return {
        'manifest_sha256': stated_hash,
        'file_sha256': sha256_file(manifest_path),
        'config': dict(config),
        'splits': split_records,
        'scaler': scaler,
        'seeds': seeds,
        'source_code': source_code,
    }


def _validate_evaluation_evidence(value, manifest_record, label):
    evidence = _mapping(value, label)
    _equal(
        evidence.get('evaluation_evidence_version'),
        EVALUATION_EVIDENCE_VERSION,
        f'{label}.evaluation_evidence_version',
    )
    score_count = _positive_int(
        evidence.get('score_count'), f'{label}.score_count',
    )
    arrays = _mapping(evidence.get('arrays'), f'{label}.arrays')
    if frozenset(arrays) != EXPECTED_EVALUATION_ARRAYS:
        _fail(f'{label}.arrays must contain exactly the four evaluation arrays')

    expected_prefix = [
        manifest_record['splits']['test']['window_count'],
        manifest_record['splits']['test']['seq_len'],
    ]
    shared_shape = None
    records = {}
    expected_record_fields = frozenset({
        'shape', 'dtype', 'sha256', 'element_count', 'finite_count',
        'score_count', 'finite_score_count',
    })
    for name in sorted(EXPECTED_EVALUATION_ARRAYS):
        record = _mapping(arrays[name], f'{label}.arrays.{name}')
        if frozenset(record) != expected_record_fields:
            _fail(f'{label}.arrays.{name} has an unexpected schema')
        shape = record.get('shape')
        if (not isinstance(shape, list) or len(shape) != 3 or
                any(isinstance(item, bool) or not isinstance(item, int) or
                    item <= 0 for item in shape)):
            _fail(f'{label}.arrays.{name}.shape must contain three positive integers')
        _equal(shape[:2], expected_prefix, f'{label}.arrays.{name} test shape')
        if shared_shape is None:
            shared_shape = shape
        else:
            _equal(shape, shared_shape, f'{label}.arrays.{name} shared shape')
        dtype = record.get('dtype')
        if not isinstance(dtype, str) or not dtype:
            _fail(f'{label}.arrays.{name}.dtype must be a non-empty string')
        _sha256_hex(record.get('sha256'), f'{label}.arrays.{name}.sha256')
        element_count = _positive_int(
            record.get('element_count'),
            f'{label}.arrays.{name}.element_count',
        )
        _equal(
            element_count, math.prod(shape),
            f'{label}.arrays.{name}.element_count',
        )
        finite_count = _nonnegative_int(
            record.get('finite_count'),
            f'{label}.arrays.{name}.finite_count',
        )
        _equal(
            finite_count, element_count,
            f'{label}.arrays.{name} full-array finite count',
        )
        _equal(
            record.get('score_count'), score_count,
            f'{label}.arrays.{name}.score_count',
        )
        _equal(
            record.get('finite_score_count'), score_count,
            f'{label}.arrays.{name}.finite_score_count',
        )
        if score_count > element_count:
            _fail(f'{label}.arrays.{name}.score_count exceeds element count')
        records[name] = dict(record)
    return {
        'evaluation_evidence_version': EVALUATION_EVIDENCE_VERSION,
        'score_count': score_count,
        'arrays': records,
    }


def _validate_result(result, manifest_record, expected_mode, label):
    config = manifest_record['config']
    result_manifest = _mapping(
        result.get('protocol_manifest'), f'{label}.protocol_manifest',
    )
    _equal(
        result_manifest.get('path'), 'protocol_manifest.json',
        f'{label}.protocol_manifest.path',
    )
    _equal(
        result_manifest.get('manifest_sha256'),
        manifest_record['manifest_sha256'],
        f'{label} result-to-manifest hash binding',
    )
    _equal(
        result_manifest.get('file_sha256'), manifest_record['file_sha256'],
        f'{label} result-to-manifest file binding',
    )

    args = _mapping(result.get('args'), f'{label}.args')
    expected_args = {
        name: value for name, value in config.items()
        if name not in RESOLVED_CONFIG_FIELDS
    }
    _equal(dict(args), expected_args, f'{label} args-to-manifest binding')
    _equal(args.get('training_mode'), expected_mode, f'{label}.args.training_mode')
    _equal(result.get('dataset'), config.get('dataset'), f'{label}.dataset')
    _equal(
        result.get('coarse_method'), config.get('coarse'),
        f'{label}.coarse_method',
    )
    _equal(
        result.get('paired_random_streams'),
        manifest_record['seeds']['paired_random_streams'],
        f'{label} result paired random streams',
    )

    coarse_metrics = _validate_metric_dict(
        result.get('coarse_metrics'), f'{label}.coarse_metrics',
    )
    refined_metrics = _validate_metric_dict(
        result.get('refined_metrics'), f'{label}.refined_metrics',
    )
    improvement = _finite_number(
        result.get('improvement_pct'), f'{label}.improvement_pct',
    )
    best_val_mae = _finite_number(
        result.get('best_val_mae'), f'{label}.best_val_mae',
    )
    evaluation_evidence = _validate_evaluation_evidence(
        result.get('evaluation_evidence'), manifest_record,
        f'{label}.evaluation_evidence',
    )
    if coarse_metrics['MAE'] == 0:
        _fail(f'{label}.coarse_metrics.MAE must be non-zero')
    expected_improvement = round(
        (coarse_metrics['MAE'] - refined_metrics['MAE']) /
        coarse_metrics['MAE'] * 100,
        2,
    )
    if not math.isclose(improvement, expected_improvement, abs_tol=1e-9):
        _fail(f'{label}.improvement_pct is inconsistent with the MAE values')
    return {
        'args': dict(args),
        'coarse_metrics': coarse_metrics,
        'refined_metrics': refined_metrics,
        'best_val_mae': best_val_mae,
        'improvement_pct': improvement,
        'evaluation_evidence': evaluation_evidence,
    }


def _validate_log(log_path, epochs, expected_mode, label):
    if os.path.islink(log_path):
        _fail(f'{label} must not be a symlink: {log_path}')
    if not os.path.isfile(log_path):
        _fail(f'missing {label}: {log_path}')

    def reject_constant(value):
        _fail(f'{label} contains non-finite JSON constant {value!r}')

    try:
        with open(log_path, encoding='utf-8') as handle:
            log = json.load(handle, parse_constant=reject_constant)
    except PairedRunValidationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        _fail(f'cannot read strict {label}: {exc}')
    if not isinstance(log, list):
        _fail(f'{label} must be a JSON array')
    if len(log) != epochs:
        _fail(f'{label} completed epochs mismatch: {len(log)} != {epochs}')
    for index, raw_entry in enumerate(log):
        entry = _validate_numeric_log_entry(raw_entry, f'{label}[{index}]')
        _equal(entry.get('epoch'), index + 1, f'{label}[{index}].epoch')
        d_fields = DISCRIMINATOR_LOG_FIELDS.intersection(entry)
        if expected_mode == 'adversarial' and d_fields != DISCRIMINATOR_LOG_FIELDS:
            _fail(f'{label}[{index}] lacks discriminator metrics')
        if expected_mode == 'reconstruction_only' and d_fields:
            _fail(f'{label}[{index}] contains discriminator-only metrics')
        if 'val_MAE' not in entry:
            _fail(f'{label}[{index}] lacks val_MAE')
    return log


def _validate_checkpoint(checkpoint, manifest_record, result_record,
                         expected_mode, epochs, log, label):
    _equal(
        checkpoint.get('protocol_manifest_sha256'),
        manifest_record['manifest_sha256'],
        f'{label} checkpoint-to-manifest binding',
    )
    _equal(
        checkpoint.get('training_mode'), expected_mode,
        f'{label}.training_mode',
    )
    _equal(
        checkpoint.get('seed'), manifest_record['seeds']['base_seed'],
        f'{label}.seed',
    )
    _equal(
        checkpoint.get('paired_random_streams'),
        manifest_record['seeds']['paired_random_streams'],
        f'{label}.paired_random_streams',
    )
    epoch = checkpoint.get('epoch')
    if (isinstance(epoch, bool) or not isinstance(epoch, int) or
            not 0 <= epoch < epochs):
        _fail(f'{label}.epoch must be in [0, {epochs})')
    val_mae = _finite_number(checkpoint.get('val_mae'), f'{label}.val_mae')
    _equal(val_mae, result_record['best_val_mae'], f'{label} best validation MAE')
    _equal(val_mae, log[epoch]['val_MAE'], f'{label} checkpoint-to-log val_MAE')

    _validate_state_dict(checkpoint.get('G'), f'{label}.G')
    if manifest_record['config'].get('ema'):
        _validate_state_dict(checkpoint.get('G_ema'), f'{label}.G_ema')
    elif checkpoint.get('G_ema') is not None:
        _fail(f'{label}.G_ema must be absent when EMA is disabled')
    discriminator = checkpoint.get('D')
    if expected_mode == 'adversarial':
        _validate_state_dict(discriminator, f'{label}.D')
    elif discriminator is not None:
        _fail(f'{label}.D must be null for reconstruction_only')


def _inspect_run(run_dir, expected_mode):
    if expected_mode not in EXPECTED_MODES:
        _fail(f'unknown expected mode: {expected_mode}')
    run_dir = os.path.abspath(run_dir)
    if os.path.islink(run_dir):
        _fail(f'{expected_mode} run directory must not be a symlink')
    if not os.path.isdir(run_dir):
        _fail(f'missing {expected_mode} run directory: {run_dir}')

    manifest_path = os.path.join(run_dir, 'protocol_manifest.json')
    result_path = os.path.join(run_dir, 'results.json')
    checkpoint_path = os.path.join(run_dir, 'best_model.pt')
    log_path = os.path.join(run_dir, 'training_log.json')
    manifest = _strict_json(manifest_path, f'{expected_mode} manifest')
    result = _strict_json(result_path, f'{expected_mode} result')
    checkpoint = _load_checkpoint(
        checkpoint_path, f'{expected_mode} checkpoint',
    )

    manifest_record = _validate_manifest(
        manifest, manifest_path, f'{expected_mode} manifest',
    )
    config = manifest_record['config']
    _equal(
        config.get('training_mode'), expected_mode,
        f'{expected_mode} manifest.config.training_mode',
    )
    _equal(
        config.get('resolved_training_mode'), expected_mode,
        f'{expected_mode} manifest.config.resolved_training_mode',
    )
    _equal(
        config.get('seed'), manifest_record['seeds']['base_seed'],
        f'{expected_mode} manifest.config.seed',
    )
    if config.get('deterministic') is not True:
        _fail(f'{expected_mode} manifest.config.deterministic must be true')
    epochs = _positive_int(
        config.get('epochs'), f'{expected_mode} manifest.config.epochs',
    )
    _positive_int(
        config.get('cardinality'),
        f'{expected_mode} manifest.config.cardinality',
    )
    _positive_int(
        config.get('noise_dim'),
        f'{expected_mode} manifest.config.noise_dim',
    )
    device = config.get('resolved_device')
    if not isinstance(device, str) or not device:
        _fail(f'{expected_mode} manifest.config.resolved_device is required')

    result_record = _validate_result(
        result, manifest_record, expected_mode, f'{expected_mode} result',
    )
    log = _validate_log(
        log_path, epochs, expected_mode, f'{expected_mode} training log',
    )
    _validate_checkpoint(
        checkpoint, manifest_record, result_record, expected_mode,
        epochs, log, f'{expected_mode} checkpoint',
    )
    return {
        'run_dir': run_dir,
        'manifest': manifest,
        'manifest_record': manifest_record,
        'result_record': result_record,
        'epochs': epochs,
        'device': device,
        'test_cardinality': manifest_record['splits']['test']['window_count'],
        'model_cardinality': config['cardinality'],
        'checkpoint_sha256': sha256_file(checkpoint_path),
        'training_log_sha256': sha256_file(log_path),
        'result_sha256': sha256_file(result_path),
    }


def _shared_manifest_view(manifest):
    shared = copy.deepcopy(manifest)
    shared.pop('manifest_sha256', None)
    shared.pop('config_sha256', None)
    config = _mapping(shared.get('config'), 'manifest.config')
    for field in PAIR_EXEMPT_CONFIG_FIELDS:
        config.pop(field, None)
    return shared


def validate_paired_runs(adversarial_run, reconstruction_run):
    """Validate and summarize a matched two-mode run pair.

    Raises:
        PairedRunValidationError: if any required evidence is absent,
            non-finite, internally inconsistent, or not pairwise matched.
    """
    adversarial_run = os.path.abspath(adversarial_run)
    reconstruction_run = os.path.abspath(reconstruction_run)
    if adversarial_run == reconstruction_run:
        _fail('adversarial and reconstruction-only run directories must differ')

    adversarial = _inspect_run(adversarial_run, 'adversarial')
    reconstruction = _inspect_run(
        reconstruction_run, 'reconstruction_only',
    )

    for field, label in (
            ('epochs', 'epochs'),
            ('device', 'resolved device'),
            ('test_cardinality', 'test cardinality'),
            ('model_cardinality', 'model cardinality')):
        _equal(adversarial[field], reconstruction[field], label)

    adversarial_shared = _shared_manifest_view(adversarial['manifest'])
    reconstruction_shared = _shared_manifest_view(reconstruction['manifest'])
    if canonical_json(adversarial_shared) != canonical_json(reconstruction_shared):
        _fail(
            'shared protocol manifest fields differ outside the explicit '
            f'exemptions: {sorted(PAIR_EXEMPT_CONFIG_FIELDS)}'
        )

    _equal(
        adversarial['result_record']['coarse_metrics'],
        reconstruction['result_record']['coarse_metrics'],
        'coarse metrics',
    )
    for array_name in ('coarse', 'target', 'indicating_mask'):
        _equal(
            adversarial['result_record']['evaluation_evidence']['arrays'][array_name],
            reconstruction['result_record']['evaluation_evidence']['arrays'][array_name],
            f'shared test {array_name} evidence',
        )

    adv_mae = adversarial['result_record']['refined_metrics']['MAE']
    rec_mae = reconstruction['result_record']['refined_metrics']['MAE']
    config = adversarial['manifest_record']['config']
    return {
        'valid': True,
        'validation_report_version': VALIDATION_REPORT_VERSION,
        'validator_sha256': sha256_file(os.path.abspath(__file__)),
        'protocol_manifest_version': PROTOCOL_MANIFEST_VERSION,
        'dataset': config['dataset'],
        'seed': adversarial['manifest_record']['seeds']['base_seed'],
        'epochs': adversarial['epochs'],
        'device': adversarial['device'],
        'test_cardinality': adversarial['test_cardinality'],
        'model_cardinality': adversarial['model_cardinality'],
        'delta_MAE_adversarial_minus_reconstruction_only': adv_mae - rec_mae,
        'runs': {
            'adversarial': {
                'directory': adversarial['run_dir'],
                'manifest_sha256': adversarial['manifest_record']['manifest_sha256'],
                'result_sha256': adversarial['result_sha256'],
                'checkpoint_sha256': adversarial['checkpoint_sha256'],
                'training_log_sha256': adversarial['training_log_sha256'],
                'refined_metrics': adversarial['result_record']['refined_metrics'],
            },
            'reconstruction_only': {
                'directory': reconstruction['run_dir'],
                'manifest_sha256': reconstruction['manifest_record']['manifest_sha256'],
                'result_sha256': reconstruction['result_sha256'],
                'checkpoint_sha256': reconstruction['checkpoint_sha256'],
                'training_log_sha256': reconstruction['training_log_sha256'],
                'refined_metrics': reconstruction['result_record']['refined_metrics'],
            },
        },
    }


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description='Fail-closed validator for one paired FMGAN run',
    )
    parser.add_argument('adversarial_run')
    parser.add_argument('reconstruction_only_run')
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    try:
        report = validate_paired_runs(
            args.adversarial_run, args.reconstruction_only_run,
        )
    except PairedRunValidationError as exc:
        print(f'INVALID: {exc}', file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
