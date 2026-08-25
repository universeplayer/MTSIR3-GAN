# FMGAN protocol blockers — 2026-08-18

This note records the boundary between the offline protocol hardening that is
implemented and the evidence still required before interpreting a matched
adversarial-versus-reconstruction-only experiment causally.

## Resolved in the current working tree

- Continuous 2D timelines are split before sliding-window construction, so a
  raw timestep cannot appear on both sides of a train/validation/test boundary.
- The scaler is fitted on raw training observations only and reused unchanged.
- Natural missing values are excluded from artificial-missing evaluation masks.
- Train, validation, and test artificial masks use explicit independent seeds.
- A best checkpoint is accepted only when its protocol-manifest hash and
  training mode match the current run. Interrupted-run resume is not supported.
- A train feature with no observed training value fails before scaling rather
  than silently turning valid validation/test values into NaNs.

## Strict paired stochastic control resolved in the current working tree

The protocol manifest is now version 3 and records a stable set of named random
streams. For the same base seed, the two training modes now have:

1. byte-identical generator initialization from an isolated initialization
   stream; optional discriminator initialization uses a separate isolated
   stream and neither construction advances the caller's global Torch RNG;
2. one explicit generator-update noise draw per batch from a stream that the
   discriminator step cannot consume; adversarial-only fake generation uses a
   separate discriminator-noise stream;
3. a separate explicit augmentation stream, keeping jitter and temporal flips
   paired even when one mode performs an additional forward pass; and
4. fixed validation and test noise tensors generated once from split-specific
   streams and reused for every evaluation. Their shapes and exact byte hashes,
   plus the Torch version and stream implementation, are bound into the
   protocol manifest.

Formal `train_refiner.py` runs now request deterministic Torch algorithms with
`warn_only=False`; an unsupported nondeterministic kernel aborts the run. The
shared helper retains its historical warn-only default solely for legacy
callers, and the manifest records which boundary was used. Version 3 also
binds the SHA-256 and size of a fixed six-file source-code allowlist, using only
relative paths, and the paired validator requires the two modes to carry the
same source record.

Final `results.json` files now retain non-raw evidence for the exact test
coarse input, prediction, target, and indicating mask: array SHA-256, dtype,
shape, element count, finite count, score count, and finite scored count. The
formal runner refuses non-finite or unscored evaluations before writing a
result. The validator requires shared target/mask/coarse records, finite
mode-specific predictions, complete epoch logs, and consistent
manifest/result/checkpoint bindings.

The paired acceptance gate additionally verifies that both otherwise-matched
manifests describe contiguous raw train/validation/test intervals, that every
split's sequence length and missing rate agree with the run configuration,
that the recorded scaler is a finite train-only `StandardScaler` whose feature
and observed-sample counts fit the raw training split, and that artificial-mask
seeds are exactly `seed`, `seed+1`, and `seed+2`. This rejects a pair of
identically malformed manifests rather than treating cross-mode equality alone
as sufficient evidence. Validation-report schema v2 also records the validator
file hash, so an archived acceptance report identifies the exact gate that
produced it.

Formal training therefore no longer relies on implicit generator noise. The
legacy ``noise=None`` model/trainer path remains available for API compatibility
but is not used by ``train_refiner.py``. Focused regression tests check matched
initial state, preservation of the global RNG, isolation of D and G-update
streams, explicit trainer routing, and fixed split-specific evaluation noise.

## Remaining evidence gate

This resolves the stochastic-control implementation blocker; it does not create
experimental evidence by itself. A causal empirical claim still requires
predeclared matched multi-seed adversarial and reconstruction-only runs, paired
reporting with uncertainty, and review of the resulting manifests/checkpoints.

## Live re-verification — 2026-08-24

- The original 48-test schema-v3 suite still passed before this audit change.
  After adding three malformed-pair regressions for scaler provenance, mask-seed
  derivation, and contiguous raw splits, the complete suite passes **51/51**.
- `python -m py_compile` passes for the formal protocol, trainer, and paired
  validator; `git diff --check` plus an explicit trailing-whitespace scan pass.
- Protocol schema code remains SHA-256
  `039b063159bc79f704caaeae8eece21dfde8d0b908a8df77e8e6019020994a09`;
  the hardened paired validator is
  `0522810b358db670a2555565027744935c0113f2034e6db36c8108a5e5807676`.
- No repository-owned `datasets/*/data.npz` and no schema-v3
  `protocol_manifest.json` exist in this checkout. Historical unpaired phase-1
  result/checkpoint files remain present but are not eligible evidence for the
  planned comparison.
