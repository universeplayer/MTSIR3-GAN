"""Regression tests for the public phase-one result summary."""

import json
import os
import re
import sys
import unittest
from pathlib import Path


FMGAN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if FMGAN_ROOT not in sys.path:
    sys.path.insert(0, FMGAN_ROOT)

from analysis import (  # noqa: E402
    collect_phase1,
    summarize,
    validate_saved_deltas,
)


class PhaseOneAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = collect_phase1()

    def test_public_snapshot_has_all_14_saved_runs(self):
        self.assertEqual(len(self.rows), 14)
        self.assertEqual(len({row["dir"] for row in self.rows}), 14)

    def test_saved_deltas_recompute_from_mae(self):
        # Raises with the exact offending run when a public result drifts.
        validate_saved_deltas(self.rows)

    def test_only_documented_legacy_run_is_excluded(self):
        excluded = {
            row["dir"]: row["aggregate_exclusion"]
            for row in self.rows
            if row["aggregate_exclusion"]
        }
        self.assertEqual(set(excluded), {"phase1_air_linear"})
        self.assertIn("g_recon was 0.0", excluded["phase1_air_linear"])

    def test_local_legacy_log_supports_exclusion_when_available(self):
        """Verify the provenance claim in checkouts retaining ignored logs."""
        run_dir = (
            Path(FMGAN_ROOT) / "results" / "results" / "phase1_air_linear"
        )
        log_path = run_dir / "training_log.json"
        if not log_path.exists():
            self.skipTest("legacy training log is intentionally not published")

        result = json.loads((run_dir / "results.json").read_text())
        log = json.loads(log_path.read_text())
        self.assertEqual(result["args"]["lambda_recon"], 10.0)
        self.assertEqual(len(log), 200)
        self.assertTrue(all(entry.get("g_recon") == 0.0 for entry in log))

    def test_eligible_linear_summary_matches_readme(self):
        summary = summarize(
            self.rows,
            lambda row: row["coarse"].startswith("linear"),
        )
        self.assertEqual(summary["count"], 8)
        self.assertAlmostEqual(summary["mean"], -0.72, places=12)
        self.assertAlmostEqual(summary["minimum"], -3.02, places=12)
        self.assertAlmostEqual(summary["maximum"], 1.13, places=12)

    def test_unfiltered_linear_snapshot_retains_legacy_provenance(self):
        summary = summarize(
            self.rows,
            lambda row: row["coarse"].startswith("linear"),
            exclude_flagged=False,
        )
        self.assertEqual(summary["count"], 9)
        self.assertAlmostEqual(summary["mean"], -3.0677777777777777)
        self.assertAlmostEqual(summary["minimum"], -21.85)
        self.assertAlmostEqual(summary["maximum"], 1.13)

    def test_weak_fill_summary_matches_readme(self):
        summary = summarize(
            self.rows,
            lambda row: row["coarse"] in {"zero", "mean"},
        )
        self.assertEqual(summary["count"], 5)
        self.assertAlmostEqual(summary["minimum"], 48.41)
        self.assertAlmostEqual(summary["maximum"], 70.23)

    def test_english_and_chinese_readmes_state_same_aggregate(self):
        repo_root = Path(FMGAN_ROOT).parent
        english = (repo_root / "README.md").read_text()
        chinese = (repo_root / "README_CN.md").read_text()
        for text in (english, chinese):
            self.assertIn("14", text)
            self.assertIn("−0.7%", text)
            self.assertIn("−3.0%", text)
            self.assertIn("+1.1%", text)
            self.assertIn("−21.9%", text)

    def test_readme_table_rounds_electricity_zero_result_correctly(self):
        repo_root = Path(FMGAN_ROOT).parent
        result = next(
            row for row in self.rows if row["dir"] == "phase1_elec_zero"
        )
        self.assertEqual(f"{result['after_mae']:.3f}", "0.426")
        row_pattern = re.compile(
            r"\|\s*Electricity\s*\|\s*Zero fill\s*\|\s*0\.832\s*"
            r"\|\s*0\.426\s*\|"
        )
        for name in ("README.md", "README_CN.md"):
            self.assertRegex((repo_root / name).read_text(), row_pattern)


if __name__ == "__main__":
    unittest.main()
