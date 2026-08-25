"""
Phase 1: Train the full R3GAN-1D refiner for time series imputation.

Usage:
    python train_refiner.py --dataset AirQuality --coarse linear --epochs 200
    python train_refiner.py --dataset Weather --coarse moment --epochs 200
    python train_refiner.py --dataset Weather --coarse linear \
        --training-mode reconstruction_only --lambda_freq 0 --epochs 200

Interrupted-run resume is intentionally unsupported. Start a new output
directory for each formal run; final evaluation accepts only a checkpoint
written by the current run whose protocol hash and training mode match.
"""

import argparse
import json
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from scipy import interpolate as scipy_interp

_FMGAN_ROOT = os.path.dirname(os.path.abspath(__file__))
if _FMGAN_ROOT not in sys.path:
    sys.path.insert(0, _FMGAN_ROOT)

from models.r3gan_1d import RefinerGenerator, RefinerDiscriminator, R3GANTrainer
from data.unified_loader import TimeSeriesImputationDataset, fit_standard_scaler
from evaluation.metrics import compute_all_metrics
from protocol import (
    build_evaluation_evidence,
    build_protocol_manifest,
    derive_paired_random_stream_seeds,
    derive_seed,
    draw_standard_normal,
    isolated_torch_initialization,
    make_dataloader_generator,
    make_torch_generator,
    seed_dataloader_worker,
    seed_everything,
    sha256_array,
    write_json_atomic,
)


TRAINING_MODES = ('adversarial', 'reconstruction_only')


def linear_interp(X_obs_np, mask_np):
    """Linear interpolation coarse imputation."""
    result = X_obs_np.copy()
    for i in range(len(X_obs_np)):
        for j in range(X_obs_np.shape[-1]):
            obs_idx = np.where(mask_np[i, :, j] == 1)[0]
            mis_idx = np.where(mask_np[i, :, j] == 0)[0]
            if len(obs_idx) > 1 and len(mis_idx) > 0:
                f = scipy_interp.interp1d(obs_idx, X_obs_np[i, obs_idx, j],
                                          kind='linear', fill_value='extrapolate')
                result[i, mis_idx, j] = f(mis_idx)
    return result


def get_coarse_imputation(method, X_obs, mask, device='cuda'):
    """Get coarse imputation using specified method."""
    if method == 'linear':
        return linear_interp(X_obs, mask)
    elif method == 'mean':
        result = X_obs.copy()
        means = np.nanmean(X_obs, axis=(0, 1))
        means = np.nan_to_num(means, nan=0.0)
        for j in range(X_obs.shape[-1]):
            missing = mask[:, :, j] == 0
            result[:, :, j][missing] = means[j]
        return result
    elif method == 'moment':
        from foundation_model.moment_wrapper import MOMENTImputer
        imputer = MOMENTImputer(frozen=True, device=device)
        imputer.to(device)
        return imputer.impute_numpy(X_obs, mask, batch_size=4)
    elif method == 'zero':
        return X_obs.copy()  # Missing positions already zeroed
    else:
        raise ValueError(f"Unknown coarse method: {method}")


def cosine_decay(step, total_steps, base_value, final_value):
    """Cosine annealing schedule."""
    progress = min(step / max(total_steps, 1), 1.0)
    decay = 0.5 * (1 + np.cos(np.pi * progress))
    return final_value + (base_value - final_value) * decay


def pick_device(preferred=None):
    """Select compute device, supporting CUDA, Apple MPS, and CPU.

    Pass ``preferred`` (e.g. from ``--device``) to force a choice; otherwise
    auto-detect in order of preference: CUDA > MPS > CPU.
    """
    if preferred:
        return preferred
    if torch.cuda.is_available():
        return 'cuda'
    if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def build_models(args, n_features, device):
    """Build paired models without consuming the caller's global Torch RNG."""
    training_mode = getattr(args, 'training_mode', 'adversarial')
    if training_mode not in TRAINING_MODES:
        raise ValueError(
            f"Unknown training mode: {training_mode}; expected one of {TRAINING_MODES}"
        )

    width = [args.width] * args.n_stages
    blocks = [args.n_blocks] * args.n_stages
    cardinality = [args.cardinality] * args.n_stages
    stream_seeds = derive_paired_random_stream_seeds(
        int(getattr(args, 'seed', 42)),
    )

    with isolated_torch_initialization(stream_seeds['generator_init']):
        generator = RefinerGenerator(
            n_features, noise_dim=args.noise_dim,
            width=width, blocks=blocks, cardinality=cardinality,
            expansion=2, kernel_size=3,
        )
    generator = generator.to(device)

    discriminator = None
    if training_mode == 'adversarial':
        with isolated_torch_initialization(stream_seeds['discriminator_init']):
            discriminator = RefinerDiscriminator(
                n_features, width=width, blocks=blocks,
                cardinality=cardinality, expansion=2, kernel_size=3,
                freq_branch=args.freq_branch,
            )
        discriminator = discriminator.to(device)

    return generator, discriminator


def validate_checkpoint_metadata(checkpoint, expected_manifest_sha256,
                                 expected_training_mode):
    """Reject stale or legacy checkpoints that cannot prove protocol identity."""
    if not isinstance(checkpoint, dict):
        raise RuntimeError('best checkpoint is not a metadata dictionary')

    manifest_sha256 = checkpoint.get('protocol_manifest_sha256')
    if manifest_sha256 is None:
        raise RuntimeError(
            'best checkpoint lacks protocol_manifest_sha256; refusing an '
            'unverifiable legacy or stale checkpoint'
        )
    if manifest_sha256 != expected_manifest_sha256:
        raise RuntimeError(
            'best checkpoint protocol mismatch: '
            f'{manifest_sha256} != {expected_manifest_sha256}'
        )

    training_mode = checkpoint.get('training_mode')
    if training_mode is None:
        raise RuntimeError(
            'best checkpoint lacks training_mode; refusing an unverifiable '
            'legacy or stale checkpoint'
        )
    if training_mode != expected_training_mode:
        raise RuntimeError(
            'best checkpoint training-mode mismatch: '
            f'{training_mode} != {expected_training_mode}'
        )
    return checkpoint


def load_verified_checkpoint(path, device, expected_manifest_sha256,
                             expected_training_mode):
    """Load a best checkpoint and validate its protocol binding."""
    if not os.path.isfile(path):
        raise RuntimeError(f'current run produced no best checkpoint: {path}')
    checkpoint = torch.load(path, map_location=device)
    return validate_checkpoint_metadata(
        checkpoint, expected_manifest_sha256, expected_training_mode,
    )


def resolve_runtime_options(args):
    """Resolve defaults added after the original programmatic ``train`` API."""
    return {
        'seed': int(getattr(args, 'seed', 42)),
        'deterministic': bool(getattr(args, 'deterministic', True)),
        'num_workers': int(getattr(args, 'num_workers', 0)),
        'training_mode': getattr(args, 'training_mode', 'adversarial'),
    }


def seed_formal_run(seed, deterministic):
    """Seed a formal run with hard deterministic-kernel enforcement."""
    return seed_everything(
        seed, deterministic=deterministic, deterministic_warn_only=False,
    )


def train(args):
    runtime = resolve_runtime_options(args)
    seed = runtime['seed']
    deterministic = runtime['deterministic']
    num_workers = runtime['num_workers']
    training_mode = runtime['training_mode']
    # Formal runs fail rather than continue after a nondeterministic-kernel
    # warning.  The helper's warn-only default remains for legacy callers.
    seed_record = seed_formal_run(seed, deterministic)
    stream_seeds = derive_paired_random_stream_seeds(seed)
    seed_record['paired_random_stream_version'] = 1
    seed_record['paired_random_stream_implementation'] = (
        'cpu_torch_generator_then_device_transfer'
    )
    seed_record['torch_version'] = str(torch.__version__)
    seed_record['paired_random_streams'] = stream_seeds
    device = pick_device(getattr(args, 'device', None))
    print(f"Device: {device}")
    print(f"Training mode: {training_mode}")
    print(f"Seed: {seed} (deterministic={deterministic})")

    # Load data
    dataset_dir = os.path.join(_FMGAN_ROOT, '..', 'datasets')
    data_path = os.path.join(dataset_dir, args.dataset, 'data.npz')
    data = np.load(data_path)
    X_full = data['X']
    n = len(X_full)
    train_stop = int(0.7 * n)
    val_stop = int(0.85 * n)
    X_train_raw = X_full[:train_stop]
    X_val_raw = X_full[train_stop:val_stop]
    X_test_raw = X_full[val_stop:]

    # Fit normalization statistics once on training timesteps. Validation and
    # test data are transformed by this exact object without refitting.
    scaler = fit_standard_scaler(X_train_raw)
    split_seeds = {
        'train': derive_seed(seed, 0),
        'val': derive_seed(seed, 1),
        'test': derive_seed(seed, 2),
    }

    # Use smaller stride for training to get more samples (fight overfitting)
    train_stride = max(1, args.seq_len // args.stride_divisor)
    train_ds = TimeSeriesImputationDataset(
        X_train_raw, seq_len=args.seq_len, missing_rate=args.missing_rate,
        seed=split_seeds['train'], stride=train_stride, scaler=scaler,
    )
    val_ds = TimeSeriesImputationDataset(
        X_val_raw, seq_len=args.seq_len, missing_rate=args.missing_rate,
        seed=split_seeds['val'], scaler=scaler,
    )
    test_ds = TimeSeriesImputationDataset(
        X_test_raw, seq_len=args.seq_len, missing_rate=args.missing_rate,
        seed=split_seeds['test'], scaler=scaler,
    )

    # Generate evaluation noise once, before the manifest is written, so its
    # exact bytes can be bound into the run provenance as well as reused at
    # every validation epoch and final test evaluation.
    val_noise = draw_standard_normal(
        (len(val_ds), args.seq_len, args.noise_dim),
        make_torch_generator(stream_seeds['validation_noise']),
        device='cpu', dtype=torch.float32,
    )
    test_noise = draw_standard_normal(
        (len(test_ds), args.seq_len, args.noise_dim),
        make_torch_generator(stream_seeds['test_noise']),
        device='cpu', dtype=torch.float32,
    )
    seed_record['fixed_evaluation_noise'] = {
        'validation_shape': list(val_noise.shape),
        'validation_sha256': sha256_array(val_noise.numpy()),
        'test_shape': list(test_noise.shape),
        'test_sha256': sha256_array(test_noise.numpy()),
    }

    os.makedirs(args.outdir, exist_ok=True)
    protocol_config = dict(vars(args))
    protocol_config.update(runtime)
    protocol_config['resolved_device'] = str(device)
    protocol_config['resolved_training_mode'] = training_mode
    protocol_manifest = build_protocol_manifest(
        dataset_path=data_path,
        raw_splits={
            'train': (0, train_stop, X_train_raw),
            'val': (train_stop, val_stop, X_val_raw),
            'test': (val_stop, n, X_test_raw),
        },
        datasets={'train': train_ds, 'val': val_ds, 'test': test_ds},
        scaler=scaler,
        config=protocol_config,
        seed_record=seed_record,
        split_seeds=split_seeds,
    )
    protocol_manifest_path = os.path.join(args.outdir, 'protocol_manifest.json')
    protocol_manifest_file_sha256 = write_json_atomic(
        protocol_manifest_path, protocol_manifest,
    )
    print(
        f"Protocol manifest: {protocol_manifest['manifest_sha256']} "
        f"({protocol_manifest_path})"
    )

    def collect(ds):
        xi, xo, m, ind = [], [], [], []
        for i in range(len(ds)):
            item = ds[i]
            xi.append(item['X_intact'].numpy())
            xo.append(item['X_observed'].numpy())
            m.append(item['mask'].numpy())
            ind.append(item['indicating_mask'].numpy())
        return np.stack(xi), np.stack(xo), np.stack(m), np.stack(ind)

    train_intact, train_obs, train_mask, _ = collect(train_ds)
    val_intact, val_obs, val_mask, val_indicating = collect(val_ds)
    test_intact, test_obs, test_mask, test_indicating = collect(test_ds)

    n_features = train_intact.shape[-1]
    print(f"Dataset: {args.dataset}, features={n_features}, seq_len={args.seq_len}")
    print(f"Train: {train_obs.shape}, Val: {val_obs.shape}, Test: {test_obs.shape}")

    # Compute coarse imputations
    print(f"Computing coarse imputation ({args.coarse})...")
    train_coarse = get_coarse_imputation(args.coarse, train_obs, train_mask, device)
    val_coarse = get_coarse_imputation(args.coarse, val_obs, val_mask, device)
    test_coarse = get_coarse_imputation(args.coarse, test_obs, test_mask, device)

    coarse_metrics = compute_all_metrics(test_coarse, test_intact, test_indicating)
    print(f"Coarse ({args.coarse}) MAE: {coarse_metrics['MAE']:.6f}")

    # Build models. Reconstruction-only runs do not instantiate D at all.
    G, D = build_models(args, n_features, device)

    g_params = sum(p.numel() for p in G.parameters())
    print(f"Generator: {g_params:,} params")
    if D is not None:
        d_params = sum(p.numel() for p in D.parameters())
        print(f"Discriminator: {d_params:,} params")
    else:
        print("Discriminator: disabled (not instantiated)")

    # Optimizers
    opt_g = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.0, 0.9), eps=1e-8)
    opt_d = None
    if D is not None:
        opt_d = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.0, 0.9), eps=1e-8)

    # EMA generator
    G_ema = None
    if args.ema:
        import copy
        G_ema = copy.deepcopy(G)
        G_ema.eval()

    # Trainer
    trainer = R3GANTrainer(
        G, D, lambda_recon=args.lambda_recon, lambda_freq=args.lambda_freq,
        adversarial=(training_mode == 'adversarial'),
    )

    # DataLoader
    train_tensors = torch.utils.data.TensorDataset(
        torch.from_numpy(train_obs).float(),
        torch.from_numpy(train_mask).float(),
        torch.from_numpy(train_coarse).float(),
        torch.from_numpy(train_intact).float(),
    )
    loader = torch.utils.data.DataLoader(
        train_tensors, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=num_workers,
        generator=make_dataloader_generator(seed),
        worker_init_fn=seed_dataloader_worker,
    )

    # All protocol randomness after model construction is isolated by role.
    # The adversarial-only D stream can be consumed without perturbing either
    # the common G-update stream or the fixed validation/test tensors.
    augmentation_generator = make_torch_generator(
        stream_seeds['augmentation'],
    )
    generator_noise_generator = make_torch_generator(
        stream_seeds['generator_update_noise'],
    )
    discriminator_noise_generator = make_torch_generator(
        stream_seeds['discriminator_noise'],
    )

    # Training loop
    best_val_mae = float('inf')
    total_steps = args.epochs * len(loader)
    global_step = 0
    log = []
    best_checkpoint_written = False

    print(f"\nTraining for {args.epochs} epochs ({total_steps} steps)...")
    print(f"  lr={args.lr}, gamma={args.gamma}, lambda_recon={args.lambda_recon}, lambda_freq={args.lambda_freq}")

    for epoch in range(args.epochs):
        G.train()
        if D is not None:
            D.train()
        epoch_metrics = {'g_adv': 0, 'g_recon': 0, 'g_freq': 0}
        if D is not None:
            epoch_metrics.update({'d_adv': 0, 'd_r1': 0, 'd_r2': 0})
        n_batches = 0

        for batch_obs, batch_mask, batch_coarse, batch_real in loader:
            batch_obs = batch_obs.to(device)
            batch_mask = batch_mask.to(device)
            batch_coarse = batch_coarse.to(device)
            batch_real = batch_real.to(device)

            # Data augmentation (applied to all tensors consistently)
            if args.augment:
                # 1. Gaussian noise on observed values (small jitter)
                jitter = draw_standard_normal(
                    batch_real.shape, augmentation_generator,
                    device=device, dtype=batch_real.dtype,
                ) * 0.01
                batch_real = batch_real + jitter
                batch_obs = batch_real * batch_mask  # Re-derive observed from jittered real
                # Re-derive coarse from jittered observed (fast linear interp approx)
                batch_coarse = batch_coarse + jitter * batch_mask

                # 2. Random temporal flip (50% chance)
                if torch.rand(1, generator=augmentation_generator).item() > 0.5:
                    batch_obs = batch_obs.flip(1)
                    batch_mask = batch_mask.flip(1)
                    batch_coarse = batch_coarse.flip(1)
                    batch_real = batch_real.flip(1)

            # Schedule
            lr = cosine_decay(global_step, total_steps, args.lr, args.lr * 0.25)
            gamma = cosine_decay(global_step, total_steps, args.gamma, args.gamma * 0.1)
            for pg in opt_g.param_groups:
                pg['lr'] = lr
            if opt_d is not None:
                for pg in opt_d.param_groups:
                    pg['lr'] = lr

            # ---- Discriminator step (adversarial mode only) ----
            d_info = {}
            if opt_d is not None:
                d_noise = draw_standard_normal(
                    (batch_obs.shape[0], batch_obs.shape[1], args.noise_dim),
                    discriminator_noise_generator,
                    device=device, dtype=batch_obs.dtype,
                )
                opt_d.zero_grad()
                d_loss, d_info = trainer.discriminator_step(
                    batch_obs, batch_mask, batch_coarse, batch_real,
                    gamma=gamma, noise=d_noise,
                )
                d_loss.backward()
                opt_d.step()

            # ---- Generator step ----
            g_noise = draw_standard_normal(
                (batch_obs.shape[0], batch_obs.shape[1], args.noise_dim),
                generator_noise_generator,
                device=device, dtype=batch_obs.dtype,
            )
            opt_g.zero_grad()
            g_loss, g_info = trainer.generator_step(
                batch_obs, batch_mask, batch_coarse, batch_real,
                noise=g_noise,
            )
            g_loss.backward()
            opt_g.step()

            # EMA update
            if G_ema is not None:
                ema_decay = min(0.999, 1 - 1 / (global_step + 1))
                with torch.no_grad():
                    for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                        p_ema.data.lerp_(p.data, 1 - ema_decay)

            for k in epoch_metrics:
                src = g_info if k.startswith('g_') else d_info
                epoch_metrics[k] += src.get(k, 0)
            n_batches += 1
            global_step += 1

        # Epoch summary
        for k in epoch_metrics:
            epoch_metrics[k] /= max(n_batches, 1)

        # Validation
        eval_G = G_ema if G_ema is not None else G
        eval_G.eval()
        val_preds = []
        bs = 32
        for i in range(0, len(val_obs), bs):
            b_obs = torch.from_numpy(val_obs[i:i+bs]).float().to(device)
            b_mask = torch.from_numpy(val_mask[i:i+bs]).float().to(device)
            b_coarse = torch.from_numpy(val_coarse[i:i+bs]).float().to(device)
            b_noise = val_noise[i:i+bs].to(device)
            with torch.no_grad():
                pred = eval_G(b_obs, b_mask, b_coarse, noise=b_noise)
            val_preds.append(pred.cpu().numpy())
        val_preds = np.concatenate(val_preds, axis=0)
        val_metrics = compute_all_metrics(val_preds, val_intact, val_indicating)

        improved = val_metrics['MAE'] < best_val_mae
        if improved:
            best_val_mae = val_metrics['MAE']
            torch.save({
                'G': G.state_dict(),
                'G_ema': G_ema.state_dict() if G_ema else None,
                'D': D.state_dict() if D is not None else None,
                'training_mode': training_mode,
                'seed': seed,
                'paired_random_streams': stream_seeds,
                'protocol_manifest_sha256': protocol_manifest['manifest_sha256'],
                'epoch': epoch,
                'val_mae': best_val_mae,
            }, os.path.join(args.outdir, 'best_model.pt'))
            best_checkpoint_written = True

        if (epoch + 1) % args.log_every == 0 or improved:
            star = ' *BEST*' if improved else ''
            if D is not None:
                print(f"  Epoch {epoch+1}/{args.epochs}: "
                      f"g_adv={epoch_metrics['g_adv']:.4f} g_recon={epoch_metrics['g_recon']:.4f} "
                      f"g_freq={epoch_metrics['g_freq']:.4f} | "
                      f"d_adv={epoch_metrics['d_adv']:.4f} r1={epoch_metrics['d_r1']:.4f} | "
                      f"val_MAE={val_metrics['MAE']:.6f}{star}")
            else:
                print(f"  Epoch {epoch+1}/{args.epochs}: "
                      f"g_recon={epoch_metrics['g_recon']:.4f} "
                      f"g_freq={epoch_metrics['g_freq']:.4f} | "
                      f"val_MAE={val_metrics['MAE']:.6f}{star}")

        log.append({**epoch_metrics, **{f'val_{k}': v for k, v in val_metrics.items()}, 'epoch': epoch + 1})

    # Final evaluation on test set
    print("\n" + "=" * 70)
    print("Final evaluation on test set")
    print("=" * 70)

    if not best_checkpoint_written:
        raise RuntimeError(
            'current run did not produce a finite improving validation '
            'checkpoint; refusing to reuse any pre-existing best_model.pt. '
            'Interrupted-run resume is not supported.'
        )
    ckpt = load_verified_checkpoint(
        os.path.join(args.outdir, 'best_model.pt'), device,
        protocol_manifest['manifest_sha256'], training_mode,
    )
    eval_G = G_ema if G_ema else G
    if G_ema and ckpt['G_ema']:
        eval_G.load_state_dict(ckpt['G_ema'])
    else:
        G.load_state_dict(ckpt['G'])
        eval_G = G
    eval_G.eval()

    test_preds = []
    for i in range(0, len(test_obs), bs):
        b_obs = torch.from_numpy(test_obs[i:i+bs]).float().to(device)
        b_mask = torch.from_numpy(test_mask[i:i+bs]).float().to(device)
        b_coarse = torch.from_numpy(test_coarse[i:i+bs]).float().to(device)
        b_noise = test_noise[i:i+bs].to(device)
        with torch.no_grad():
            pred = eval_G(b_obs, b_mask, b_coarse, noise=b_noise)
        test_preds.append(pred.cpu().numpy())
    test_preds = np.concatenate(test_preds, axis=0)
    evaluation_evidence = build_evaluation_evidence(
        coarse=test_coarse,
        prediction=test_preds,
        target=test_intact,
        indicating_mask=test_indicating,
    )
    test_metrics = compute_all_metrics(test_preds, test_intact, test_indicating)

    print(f"Coarse ({args.coarse}):  MAE={coarse_metrics['MAE']:.6f}  MSE={coarse_metrics['MSE']:.6f}")
    model_label = ('R3GAN-1D refined' if training_mode == 'adversarial'
                   else 'Reconstruction-only refined')
    print(f"{model_label}:  MAE={test_metrics['MAE']:.6f}  MSE={test_metrics['MSE']:.6f}")
    imp = (coarse_metrics['MAE'] - test_metrics['MAE']) / coarse_metrics['MAE'] * 100
    print(f"Improvement: {imp:+.2f}% MAE")

    # Save results
    results = {
        'dataset': args.dataset,
        'coarse_method': args.coarse,
        'coarse_metrics': coarse_metrics,
        'refined_metrics': test_metrics,
        'improvement_pct': round(imp, 2),
        'best_val_mae': best_val_mae,
        'evaluation_evidence': evaluation_evidence,
        'paired_random_streams': stream_seeds,
        'protocol_manifest': {
            'path': 'protocol_manifest.json',
            'manifest_sha256': protocol_manifest['manifest_sha256'],
            'file_sha256': protocol_manifest_file_sha256,
        },
        'args': vars(args),
    }
    with open(os.path.join(args.outdir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    with open(os.path.join(args.outdir, 'training_log.json'), 'w') as f:
        json.dump(log, f, indent=2)

    print(f"\nResults saved to {args.outdir}/")


def build_arg_parser():
    """Create the CLI parser so mode validation is reusable in tests."""
    parser = argparse.ArgumentParser(description='Train R3GAN-1D refiner')
    # Data
    parser.add_argument('--dataset', type=str, default='AirQuality')
    parser.add_argument('--seq_len', type=int, default=36)
    parser.add_argument('--missing_rate', type=float, default=0.25)
    parser.add_argument('--coarse', type=str, default='linear',
                        choices=['linear', 'mean', 'moment', 'zero'])
    # Architecture
    parser.add_argument('--noise_dim', type=int, default=32)
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--n_stages', type=int, default=4)
    parser.add_argument('--n_blocks', type=int, default=2)
    parser.add_argument('--cardinality', type=int, default=32)
    parser.add_argument('--freq_branch', type=bool, default=True)
    # Training
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', '--num-workers', type=int, default=0)
    parser.add_argument(
        '--deterministic', action=argparse.BooleanOptionalAction, default=True,
        help='Use deterministic Torch algorithms where available',
    )
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--gamma', type=float, default=0.05)
    parser.add_argument('--lambda_recon', type=float, default=10.0)
    parser.add_argument('--lambda_freq', type=float, default=1.0)
    parser.add_argument(
        '--training_mode', '--training-mode',
        choices=TRAINING_MODES, default='adversarial',
        help=(
            "'adversarial' keeps the existing R3GAN objective; "
            "'reconstruction_only' does not instantiate or optimize a "
            "discriminator and trains the same generator with the masked L1 "
            "and configured spectral reconstruction losses only"
        ),
    )
    parser.add_argument('--ema', action='store_true', default=True)
    parser.add_argument('--augment', action='store_true', default=False,
                        help='Enable data augmentation (jitter + temporal flip)')
    parser.add_argument('--stride_divisor', type=int, default=2,
                        help='Train stride = seq_len // stride_divisor (higher = more overlap = more samples)')
    # Output
    parser.add_argument('--outdir', type=str, default='results/phase1')
    parser.add_argument('--log_every', type=int, default=10)
    parser.add_argument('--device', type=str, default=None,
                        choices=['cuda', 'mps', 'cpu'],
                        help='Force compute device; default auto-detects CUDA > MPS > CPU')
    return parser


if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    train(args)
