# Paired adversarial-versus-reconstruction experiment plan

This is the empirical gate that remains after the offline protocol hardening.
It is a plan, not evidence: no real-dataset training run has been executed in
this checkout because `datasets/<name>/data.npz` is absent.

## Question and estimand

Does the discriminator improve imputation after a competent linear coarse
imputer when the generator, data split, scaler, masks, initialization,
augmentation, optimization noise, and evaluation noise are held pairwise
fixed?

The primary paired estimand is, for each dataset and seed,

`delta_MAE = MAE_adversarial - MAE_reconstruction_only`.

Negative values favor adversarial training. Report the paired mean and median,
all seed-level deltas, a paired bootstrap 95% interval, and the fraction of
seeds favoring each mode. Do not treat runs as independent configurations or
claim significance from a single seed.

## Frozen initial matrix

- Datasets: `AirQuality`, `Weather`, and `Electricity` only after each source
  file and conversion step has an explicit provenance hash.
- Coarse method: `linear`.
- Artificial missingness: 25% point-MCAR.
- Seeds: `42, 43, 44, 45, 46` as a minimum; increase to ten paired seeds if the
  five-seed interval straddles a practically meaningful effect.
- Modes: `adversarial` and `reconstruction_only`.
- Objective in both modes: the same masked L1 reconstruction and configured
  spectral reconstruction terms; the discriminator is the only intended
  treatment difference. A literal L1-only experiment (`--lambda_freq 0`) is a
  separate ablation and must not be mixed into this comparison.
- Determinism: enabled; `--num-workers 0` is the verified safe local default.
- Output directory: unique per dataset, seed, and mode. Never reuse an output
  directory or a checkpoint from another run.

## Pre-run gates

1. Materialize each dataset from an authoritative source without overwriting
   existing files. Record the source URL or Git object, source SHA-256,
   conversion command, resulting `data.npz` SHA-256, shape, dtype, and natural
   missing-value count.
2. Run all `FMGAN/tests` and require every test to pass.
3. Run a one-epoch CPU smoke pair in disposable output directories. Verify the
   two manifests have the same dataset, split, scaler, masks, generator-init
   stream, augmentation stream, generator-update stream, validation noise, and
   test noise. Only the training mode and discriminator-specific fields may
   differ.
4. Freeze the exact CLI arguments and code-tree diff hash before any formal
   run. Any code/config change invalidates the matrix and requires a clean
   restart.
5. Check available GPU/runtime budget. Do not silently shorten one mode, change
   batch size, or substitute a different device for only part of a pair.

The formal runner now writes protocol-manifest schema v3: it hard-fails on a
nondeterministic Torch operation, binds an allowlist of six critical source
files, and records exact coarse/prediction/target/score-mask array evidence.
These fields are mandatory; a legacy result without them is not eligible for
the paired matrix.

## Command template

Run the two commands for a seed from the same frozen checkout. Replace the
bracketed values; do not point two runs at the same directory.

```bash
.venv/bin/python FMGAN/train_refiner.py \
  --dataset [DATASET] --seq_len [SEQ_LEN] --missing_rate 0.25 \
  --coarse linear --training-mode reconstruction_only \
  --epochs 200 --batch_size 32 --seed [SEED] \
  --num-workers 0 --deterministic \
  --outdir FMGAN/results/paired_v1/[DATASET]/seed_[SEED]/reconstruction_only

.venv/bin/python FMGAN/train_refiner.py \
  --dataset [DATASET] --seq_len [SEQ_LEN] --missing_rate 0.25 \
  --coarse linear --training-mode adversarial \
  --epochs 200 --batch_size 32 --seed [SEED] \
  --num-workers 0 --deterministic \
  --outdir FMGAN/results/paired_v1/[DATASET]/seed_[SEED]/adversarial
```

Use the dataset-specific sequence lengths already declared by the project
(`36` for AirQuality and `96` for Weather/Electricity) unless a new protocol is
predeclared before running either mode.

## Pair acceptance checks

For every seed pair, fail closed unless:

- both runs completed the same epoch count on the same device class;
- each checkpoint is bound to its own current manifest and training mode;
- the manifest-bound dataset, scaler, split indices, artificial masks, coarse
  imputations, generator initialization, augmentation, generator-update noise,
  and validation/test noise hashes match across modes;
- all test predictions and scored targets are finite;
- both result files expose the same metric definitions and test cardinality;
- neither run loaded a stale checkpoint or resumed an interrupted optimizer/RNG
  state.

Retain failed pairs in an audit log, but exclude them only by a predeclared
technical rule applied symmetrically. Never drop a seed because its effect is
unfavorable.

Run the fail-closed validator for every completed pair:

```bash
.venv/bin/python FMGAN/evaluation/validate_paired_runs.py \
  FMGAN/results/paired_v1/[DATASET]/seed_[SEED]/adversarial \
  FMGAN/results/paired_v1/[DATASET]/seed_[SEED]/reconstruction_only
```

Archive the validator's machine-readable JSON with the pair. An exit status
other than zero, a legacy/missing schema field, or any shared-protocol mismatch
invalidates that pair until it is rerun from a clean output directory.

## Reporting boundary

The first report should show the full seed-level table, paired deltas, coverage,
runtime, and manifest hashes. It may conclude that the discriminator helps,
hurts, or is unresolved. It must not claim causality before the pair checks pass,
and it must not generalize beyond the tested datasets, 25% point-MCAR, linear
coarse starts, and the frozen architecture. Block missingness, MNAR, natural
missingness, downstream utility, and stronger coarse imputers remain separate
future experiments.
