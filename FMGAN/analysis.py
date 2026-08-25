"""FMGAN results aggregator — reproduce the MiLeTS 2026 result tables.

Walks results/results/phase1_*/results.json and baseline_*.json and reconstructs:
  Table 1: R3GAN-1D refinement effectiveness (dataset x coarse method -> before/after MAE, Delta%)
  Table 2: comparison vs established baselines (BRITS / SAITS) on Weather @ rate 0.25

The script retains every saved run in the detailed table.  A named provenance
registry excludes one legacy AirQuality run from the linear-start aggregate:
that run configured a non-zero reconstruction weight but logged exactly zero
reconstruction loss for all 200 epochs.  Keeping the exclusion explicit avoids
silently mixing a known logging/training anomaly into the headline summary.

Stdlib only; offline. Run from the repository root:
    python3 FMGAN/analysis.py
"""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results" / "results"

# canonical coarse-method order for the table
COARSE_ORDER = ["zero", "mean", "linear", "linear_v2", "augmented"]

# This is deliberately a named, reviewable registry rather than a heuristic.
# The saved ``results.json`` remains in the detailed table, while the aggregate
# omits it for the evidence-backed reason recorded here and in the README.
AGGREGATE_EXCLUSIONS = {
    "phase1_air_linear": (
        "legacy log anomaly: lambda_recon=10, but g_recon was 0.0 in all "
        "200 logged epochs"
    ),
}


def load(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None


def collect_phase1(results_root=RES):
    rows = []
    for d in sorted(Path(results_root).glob("phase1_*")):
        rj = d / "results.json"
        if not rj.exists():
            continue
        j = load(rj)
        if not j or "coarse_metrics" not in j or "refined_metrics" not in j:
            continue
        rows.append({
            "dir": d.name,
            "dataset": j.get("dataset", "?"),
            "coarse": j.get("coarse_method", "?"),
            "before_mae": j["coarse_metrics"].get("MAE"),
            "after_mae": j["refined_metrics"].get("MAE"),
            "delta_pct": j.get("improvement_pct"),
            "aggregate_exclusion": AGGREGATE_EXCLUSIONS.get(d.name),
        })
    return rows


def collect_baselines():
    out = []
    for f in sorted(RES.glob("baseline_*.json")):
        j = load(f)
        if isinstance(j, list):
            out.extend(j)
        elif isinstance(j, dict):
            out.append(j)
    return out


def fmt(x, n=3):
    return f"{x:.{n}f}" if isinstance(x, (int, float)) else str(x)


def summarize(rows, predicate, exclude_flagged=True):
    """Return a finite-delta summary for rows selected by ``predicate``."""
    selected = [r for r in rows if predicate(r)]
    if exclude_flagged:
        selected = [r for r in selected if not r["aggregate_exclusion"]]
    deltas = [
        float(r["delta_pct"])
        for r in selected
        if isinstance(r["delta_pct"], (int, float))
        and math.isfinite(r["delta_pct"])
    ]
    if not deltas:
        return None
    return {
        "count": len(deltas),
        "minimum": min(deltas),
        "maximum": max(deltas),
        "mean": sum(deltas) / len(deltas),
    }


def validate_saved_deltas(rows, tolerance=0.0051):
    """Check that each saved Delta% agrees with its before/after MAE values.

    ``improvement_pct`` is stored to two decimals, hence the half-centesimal
    tolerance.  Invalid or inconsistent public evidence should fail loudly.
    """
    errors = []
    for row in rows:
        before = row["before_mae"]
        after = row["after_mae"]
        saved = row["delta_pct"]
        values = (before, after, saved)
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
            errors.append(f"{row['dir']}: non-finite or missing MAE/Delta%")
            continue
        if before <= 0:
            errors.append(f"{row['dir']}: before_mae must be positive")
            continue
        recomputed = (before - after) / before * 100.0
        if abs(saved - recomputed) > tolerance:
            errors.append(
                f"{row['dir']}: saved Delta%={saved} but MAEs imply "
                f"{recomputed:.6f}"
            )
    if errors:
        raise ValueError("invalid phase-one results:\n  " + "\n  ".join(errors))


def main():
    rows = collect_phase1()
    validate_saved_deltas(rows)
    print(f"# FMGAN — reproduced from {len(rows)} phase1 results.json\n")

    print("## Table 1 — R3GAN-1D refinement effectiveness (MAE, lower=better)")
    print(
        f"{'Dataset':12s} {'Coarse':12s} {'Before':>8s} {'After':>8s} "
        f"{'Delta%':>8s}  Aggregate status"
    )
    def key(r):
        try:
            ci = COARSE_ORDER.index(r["coarse"])
        except ValueError:
            ci = len(COARSE_ORDER)
        return (r["dataset"], ci)
    for r in sorted(rows, key=key):
        dpct = f"{r['delta_pct']:+.1f}" if r["delta_pct"] is not None else "?"
        status = (
            f"EXCLUDED — {r['aggregate_exclusion']}"
            if r["aggregate_exclusion"] else "included"
        )
        print(f"{r['dataset']:12s} {r['coarse']:12s} "
              f"{fmt(r['before_mae']):>8s} {fmt(r['after_mae']):>8s} "
              f"{dpct:>8s}  {status}")

    # Headline finding: large saved-run changes for zero/mean starts, but no
    # consistent gain for the eight eligible linear-start configurations.
    print("\n## Descriptive finding — large changes occur only for zero/mean starts")
    for cat, preds in [("weak fills (zero/mean)", lambda c: c in ("zero", "mean")),
                       ("strong fill (linear*)", lambda c: c.startswith("linear"))]:
        summary = summarize(rows, lambda r: preds(r["coarse"]))
        if summary:
            print(
                f"  {cat:24s}: Delta% range "
                f"[{summary['minimum']:+.1f}, {summary['maximum']:+.1f}], "
                f"mean {summary['mean']:+.1f}  (n={summary['count']})"
            )

    exclusions = [r for r in rows if r["aggregate_exclusion"]]
    if exclusions:
        print("\n## Aggregate exclusions (still shown in Table 1)")
        for row in exclusions:
            print(
                f"  {row['dir']}: Delta%={row['delta_pct']:+.1f}; "
                f"{row['aggregate_exclusion']}"
            )

    print("\n## Table 2 — vs established baselines (raw baseline_*.json)")
    for b in collect_baselines():
        mae = b.get("MAE")
        mse = b.get("MSE")
        valid = all(
            isinstance(value, (int, float)) and math.isfinite(value)
            for value in (mae, mse)
        )
        status = "valid" if valid else "INVALID/UNUSABLE"
        print(f"  {b.get('method','?'):8s} {b.get('dataset','?'):12s} "
              f"rate={b.get('missing_rate','?')}  MAE={fmt(mae)}  "
              f"MSE={fmt(mse)}  [{status}]")
    print("\n  (Saved Weather endpoints: R3GAN-1D standalone MAE ~0.228 vs BRITS ~0.039;")
    print("   linear-start refinement 0.067 -> 0.067. These heterogeneous runs are")
    print("   descriptive endpoints, not a matched causal comparison.)")


if __name__ == "__main__":
    main()
