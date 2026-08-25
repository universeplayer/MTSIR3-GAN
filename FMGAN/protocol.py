"""Reproducibility and provenance helpers for FMGAN experiments.

The functions in this module are intentionally independent of model code.  A
run can therefore lock its random streams and write a compact protocol
manifest before any expensive optimization starts.
"""

import hashlib
import json
import os
import random
import tempfile
from contextlib import contextmanager

import numpy as np
import torch


PROTOCOL_MANIFEST_VERSION = 3
SOURCE_CODE_ALLOWLIST_VERSION = 1
EVALUATION_EVIDENCE_VERSION = 1
UINT32_MODULUS = 2 ** 32
# Capture once. Reading os.environ after the first seed_everything call would
# mistake our own runtime mutation for the interpreter's startup hash seed.
PYTHON_HASH_SEED_AT_MODULE_IMPORT = os.environ.get('PYTHONHASHSEED')

# These named streams are deliberately stable protocol, rather than incidental
# offsets scattered through the training loop.  In particular, the generator
# update stream exists in both training modes while the discriminator stream is
# consumed only by adversarial runs.  Consuming the latter therefore cannot
# advance the former.
PAIRED_RANDOM_STREAM_OFFSETS = {
    # Offset zero preserves the historical default adversarial G
    # initialization for the common linear/mean/zero coarse paths.
    'generator_init': 0,
    'discriminator_init': 101,
    'augmentation': 201,
    'generator_update_noise': 202,
    'discriminator_noise': 203,
    'validation_noise': 301,
    'test_noise': 302,
}

# Only files that can change the formal refiner's data, model, optimization,
# coarse-imputation, or metric behavior are bound.  This deliberately avoids a
# dirty-repository tree hash, generated artifacts, papers, and absolute paths.
SOURCE_CODE_ALLOWLIST = (
    'train_refiner.py',
    'protocol.py',
    'models/r3gan_1d.py',
    'data/unified_loader.py',
    'evaluation/metrics.py',
    'foundation_model/moment_wrapper.py',
)


def derive_seed(base_seed, offset=0):
    """Return a deterministic NumPy/DataLoader-compatible uint32 seed."""
    return (int(base_seed) + int(offset)) % UINT32_MODULUS


def derive_paired_random_stream_seeds(base_seed):
    """Return the named seeds for a strict paired two-mode comparison."""
    return {
        name: derive_seed(base_seed, offset)
        for name, offset in PAIRED_RANDOM_STREAM_OFFSETS.items()
    }


def make_torch_generator(seed):
    """Create an explicitly seeded CPU generator for protocol randomness.

    Drawing protocol noise on CPU and then transferring it to the selected
    accelerator avoids falling back to a device-global RNG.  It also keeps the
    random stream definition identical on CPU, CUDA, and MPS.
    """
    generator = torch.Generator(device='cpu')
    generator.manual_seed(derive_seed(seed))
    return generator


def draw_standard_normal(shape, generator, device='cpu', dtype=torch.float32):
    """Draw from a private CPU stream and transfer without further randomness."""
    sample = torch.randn(
        tuple(int(size) for size in shape), generator=generator,
        device='cpu', dtype=dtype,
    )
    return sample.to(device=device)


@contextmanager
def isolated_torch_initialization(seed):
    """Seed CPU module initialization without advancing the global RNG.

    FMGAN modules initialize their parameters on CPU before ``.to(device)``.
    ``fork_rng`` restores the caller's CPU state on exit, while the explicit
    state below makes construction independent of all earlier random draws.
    """
    private_state = make_torch_generator(seed).get_state()
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(private_state)
        yield


def seed_everything(seed, deterministic=True, deterministic_warn_only=True):
    """Seed Python, NumPy, Torch, CUDA, and deterministic backend controls.

    ``PYTHONHASHSEED`` only takes effect when the interpreter starts.  We set
    it for child processes and record whether the current interpreter already
    had the requested value instead of claiming retroactive hash determinism.

    ``deterministic_warn_only=True`` preserves the historical helper behavior
    for legacy callers.  Formal ``train_refiner.py`` runs explicitly pass
    ``False`` so an unavailable deterministic kernel aborts instead of merely
    warning and continuing with an unverifiable run.
    """
    seed = derive_seed(seed)
    hash_seed_at_start = PYTHON_HASH_SEED_AT_MODULE_IMPORT
    os.environ['PYTHONHASHSEED'] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    deterministic_warn_only = bool(deterministic and deterministic_warn_only)
    torch.use_deterministic_algorithms(
        bool(deterministic), warn_only=deterministic_warn_only,
    )
    if getattr(torch.backends, 'cudnn', None) is not None:
        torch.backends.cudnn.deterministic = bool(deterministic)
        if deterministic:
            torch.backends.cudnn.benchmark = False

    return {
        'base_seed': seed,
        'python_random_seed': seed,
        'numpy_seed': seed,
        'torch_seed': seed,
        'cuda_seed': seed if torch.cuda.is_available() else None,
        'dataloader_seed': seed,
        'deterministic_algorithms': bool(deterministic),
        'deterministic_warn_only': deterministic_warn_only,
        'python_hash_seed_at_process_start': hash_seed_at_start,
        'python_hash_seed_matches_run': hash_seed_at_start == str(seed),
    }


def seed_dataloader_worker(_worker_id):
    """Seed Python and NumPy from Torch's deterministic per-worker seed."""
    worker_seed = torch.initial_seed() % UINT32_MODULUS
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_dataloader_generator(seed):
    """Create the explicit generator used for deterministic shuffling."""
    generator = torch.Generator()
    generator.manual_seed(derive_seed(seed))
    return generator


def canonical_json(payload):
    """Serialize JSON data deterministically for hashing and on-disk review."""
    return json.dumps(
        _json_safe(payload), sort_keys=True, separators=(',', ':'),
        ensure_ascii=True, allow_nan=False,
    )


def sha256_bytes(payload):
    """Return the SHA-256 hex digest for bytes."""
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path, chunk_size=1024 * 1024):
    """Hash a file without reading it all into memory."""
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array):
    """Hash array metadata and C-order bytes, not a lossy JSON conversion."""
    value = np.asarray(array)
    contiguous = np.ascontiguousarray(value)
    header = canonical_json({
        'dtype': contiguous.dtype.str,
        'shape': list(contiguous.shape),
    }).encode('ascii')
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b'\0')
    digest.update(memoryview(contiguous).cast('B'))
    return digest.hexdigest()


def source_code_record(source_root=None):
    """Hash the fixed allowlist of source files using relative path keys.

    Missing files, symlinks, or paths escaping ``source_root`` abort manifest
    construction.  The allowlist is code-owned rather than caller-supplied so
    a run cannot silently weaken its own provenance record.
    """
    if source_root is None:
        source_root = os.path.dirname(os.path.abspath(__file__))
    source_root = os.path.abspath(source_root)
    if os.path.islink(source_root) or not os.path.isdir(source_root):
        raise ValueError(f'invalid source root: {source_root}')

    files = {}
    for relative_path in SOURCE_CODE_ALLOWLIST:
        normalized = os.path.normpath(relative_path)
        if (os.path.isabs(relative_path) or normalized != relative_path or
                normalized == '..' or normalized.startswith('..' + os.sep)):
            raise ValueError(
                f'unsafe source-code allowlist path: {relative_path}'
            )
        path = os.path.abspath(os.path.join(source_root, relative_path))
        try:
            inside_root = os.path.commonpath([source_root, path]) == source_root
        except ValueError:
            inside_root = False
        if not inside_root:
            raise ValueError(
                f'source-code allowlist path escapes root: {relative_path}'
            )
        if os.path.islink(path) or not os.path.isfile(path):
            raise ValueError(
                f'missing or symlinked source-code allowlist file: '
                f'{relative_path}'
            )
        files[relative_path] = {
            'size_bytes': os.path.getsize(path),
            'sha256': sha256_file(path),
        }
    return {
        'allowlist_version': SOURCE_CODE_ALLOWLIST_VERSION,
        'files': files,
    }


def array_evidence_record(array, score_mask):
    """Return exact bytes, shape, and finiteness counts for one test array."""
    value = np.asarray(array)
    mask = np.asarray(score_mask)
    if value.shape != mask.shape:
        raise ValueError(
            f'evaluation array shape {value.shape} does not match score mask '
            f'{mask.shape}'
        )
    scored = mask != 0
    finite = np.isfinite(value)
    return {
        'shape': list(value.shape),
        'dtype': value.dtype.str,
        'sha256': sha256_array(value),
        'element_count': int(value.size),
        'finite_count': int(np.count_nonzero(finite)),
        'score_count': int(np.count_nonzero(scored)),
        'finite_score_count': int(np.count_nonzero(finite & scored)),
    }


def build_evaluation_evidence(coarse, prediction, target, indicating_mask):
    """Build fail-closed, non-raw evidence for a formal test evaluation."""
    arrays = {
        'coarse': np.asarray(coarse),
        'prediction': np.asarray(prediction),
        'target': np.asarray(target),
        'indicating_mask': np.asarray(indicating_mask),
    }
    shape = arrays['indicating_mask'].shape
    if not shape or any(value.shape != shape for value in arrays.values()):
        raise ValueError('all formal evaluation arrays must share one shape')
    mask = arrays['indicating_mask']
    if not np.isfinite(mask).all():
        raise ValueError('formal indicating mask contains non-finite values')
    if not np.logical_or(mask == 0, mask == 1).all():
        raise ValueError('formal indicating mask must be binary')
    score_count = int(np.count_nonzero(mask))
    if score_count <= 0:
        raise ValueError('formal evaluation has zero scored positions')

    records = {
        name: array_evidence_record(value, mask)
        for name, value in arrays.items()
    }
    for name, record in records.items():
        if record['finite_count'] != record['element_count']:
            raise ValueError(
                f'formal evaluation array {name} contains non-finite values'
            )
        if record['finite_score_count'] != score_count:
            raise ValueError(
                f'formal evaluation array {name} is non-finite at a scored '
                'position'
            )
    return {
        'evaluation_evidence_version': EVALUATION_EVIDENCE_VERSION,
        'score_count': score_count,
        'arrays': records,
    }


def scaler_record(scaler):
    """Return a stable, auditable record for a fitted StandardScaler."""
    state = {
        'class': type(scaler).__name__,
        'mean': np.asarray(scaler.mean_),
        'scale': np.asarray(scaler.scale_),
        'var': np.asarray(scaler.var_),
        'n_features_in': int(scaler.n_features_in_),
        'n_samples_seen': np.asarray(scaler.n_samples_seen_),
    }
    hash_payload = {
        key: (sha256_array(value) if isinstance(value, np.ndarray) else value)
        for key, value in state.items()
    }
    return {
        'fitted_on': 'train_only',
        'state_sha256': sha256_bytes(canonical_json(hash_payload).encode('ascii')),
        'mean': _json_safe(state['mean']),
        'scale': _json_safe(state['scale']),
        'var': _json_safe(state['var']),
        'n_features_in': state['n_features_in'],
        'n_samples_seen': _json_safe(state['n_samples_seen']),
    }


def build_protocol_manifest(dataset_path, raw_splits, datasets, scaler,
                            config, seed_record, split_seeds,
                            source_root=None):
    """Build a deterministic split/mask/scaler/config provenance manifest.

    Args:
        dataset_path: source ``.npz`` path.
        raw_splits: mapping of split name to ``(start, stop, raw_array)``.
        datasets: mapping of split name to ``TimeSeriesImputationDataset``.
        scaler: the one scaler fitted on the training split.
        config: run configuration (normally ``vars(args)``).
        seed_record: output from :func:`seed_everything`.
        split_seeds: artificial-mask seed for each split.
    """
    split_records = {}
    for name in sorted(raw_splits):
        start, stop, raw = raw_splits[name]
        dataset = datasets[name]
        split_records[name] = {
            'start': int(start),
            'stop': int(stop),
            'raw_shape': list(np.asarray(raw).shape),
            'raw_sha256': sha256_array(raw),
            'window_count': len(dataset),
            'seq_len': int(dataset.seq_len),
            'stride': int(dataset.stride),
            'missing_rate': float(dataset.missing_rate),
            'missing_pattern': dataset.missing_pattern,
            'mask_seed': int(split_seeds[name]),
            'original_mask_sha256': sha256_array(dataset.original_mask),
            'combined_mask_sha256': sha256_array(dataset.combined_mask),
            'indicating_mask_sha256': sha256_array(dataset.indicating_mask),
        }

    config = _json_safe(config)
    manifest = {
        'protocol_manifest_version': PROTOCOL_MANIFEST_VERSION,
        # Avoid machine-specific absolute paths while binding the source bytes.
        'dataset_source': {
            'filename': os.path.basename(dataset_path),
            'size_bytes': os.path.getsize(dataset_path),
            'sha256': sha256_file(dataset_path),
        },
        'splits': split_records,
        'scaler': scaler_record(scaler),
        'seeds': _json_safe(seed_record),
        'source_code': source_code_record(source_root=source_root),
        'config': config,
        'config_sha256': sha256_bytes(canonical_json(config).encode('ascii')),
    }
    manifest['manifest_sha256'] = sha256_bytes(
        canonical_json(manifest).encode('ascii')
    )
    return manifest


def write_json_atomic(path, payload):
    """Atomically write stable, indented JSON and return its file hash."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(
        prefix='.protocol-', suffix='.json.tmp', dir=directory,
    )
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            json.dump(_json_safe(payload), handle, indent=2, sort_keys=True,
                      ensure_ascii=True, allow_nan=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
        raise
    return sha256_file(path)


def _json_safe(value):
    """Convert common scientific Python values to strict JSON values."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, torch.device):
        return str(value)
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not np.isfinite(value):
            # Preserve unexpected scientific sentinel values explicitly while
            # keeping the JSON syntax strict. Train scalers fail closed before
            # an all-NaN feature can reach this serializer.
            if np.isnan(value):
                return 'NaN'
            return 'Infinity' if value > 0 else '-Infinity'
        return value
    return str(value)
