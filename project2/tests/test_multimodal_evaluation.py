from __future__ import annotations

import sys
import unittest
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from multimodal_evaluation import (
    FieldCounts,
    aggregate_rejection_metrics,
    contains_match,
    percentile,
    predict_rejection,
    score_field,
    update_counts,
)


class MultimodalMetricTests(unittest.TestCase):
    def test_wrong_value_counts_as_false_positive_and_false_negative(self) -> None:
        counts = FieldCounts()
        outcome = score_field(
            actual="708-2L-99999",
            expected_values=["708-2L-00300"],
            visibility="readable",
        )
        update_counts(counts, outcome)

        self.assertEqual(outcome, "fn_and_fp")
        self.assertEqual(counts.fn, 1)
        self.assertEqual(counts.fp, 1)

    def test_unreadable_field_penalizes_hallucination(self) -> None:
        outcome = score_field(
            actual="PC200-8",
            expected_values=[],
            visibility="unreadable",
        )
        self.assertEqual(outcome, "fp")

    def test_unreviewed_field_is_skipped(self) -> None:
        outcome = score_field(
            actual="KOMATSU",
            expected_values=[],
            visibility="unreviewed",
        )
        self.assertEqual(outcome, "skipped")

    def test_alias_contains_match(self) -> None:
        self.assertTrue(
            contains_match("KOMATSU hydraulic pump", ["hydraulic pump", "主泵"])
        )

    def test_rejection_confusion_matrix(self) -> None:
        metrics = aggregate_rejection_metrics(
            [
                {"expected_reject": True, "predicted_reject": True},
                {"expected_reject": True, "predicted_reject": False},
                {"expected_reject": False, "predicted_reject": True},
                {"expected_reject": False, "predicted_reject": False},
            ]
        )
        self.assertEqual(metrics["tp"], 1)
        self.assertEqual(metrics["fp"], 1)
        self.assertEqual(metrics["fn"], 1)
        self.assertEqual(metrics["tn"], 1)
        self.assertEqual(metrics["accuracy"], 0.5)

    def test_percentile_uses_linear_interpolation(self) -> None:
        self.assertEqual(percentile([10, 20, 30, 40], 0.5), 25.0)
        self.assertEqual(percentile([10, 20, 30, 40], 0.95), 38.5)

    def test_damage_evidence_is_not_rejected_when_auto_merge_is_false(self) -> None:
        self.assertFalse(
            predict_rejection(
                {
                    "image_type": "part",
                    "image_quality": "fair",
                    "confidence": 0.6,
                    "visible_damage": ["裂纹", "漏油"],
                    "safe_for_auto_merge": False,
                }
            )
        )

    def test_irrelevant_or_unusable_image_is_rejected(self) -> None:
        self.assertTrue(
            predict_rejection(
                {
                    "image_type": "irrelevant",
                    "image_quality": "good",
                    "confidence": 0.8,
                }
            )
        )
        self.assertTrue(
            predict_rejection(
                {
                    "image_type": "nameplate",
                    "image_quality": "unusable",
                    "confidence": 0.3,
                }
            )
        )


if __name__ == "__main__":
    unittest.main()
