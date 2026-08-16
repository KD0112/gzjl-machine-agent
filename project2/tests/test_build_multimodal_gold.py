from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from tests.build_multimodal_gold import build_gold


FIELDS = [
    "id",
    "challenge_tags",
    "reviewer_a",
    "reviewer_b",
    "adjudicator",
    "license_manual_verified",
    "privacy_approved",
    "expected_image_types",
    "brand_visibility",
    "expected_brand_any",
    "machine_model_visibility",
    "expected_machine_model_any",
    "part_name_candidate_visibility",
    "expected_part_name_any",
    "part_number_visibility",
    "expected_part_number_any",
    "expected_damage_keywords",
    "should_reject",
    "annotation_notes",
    "conflict_fields",
    "adjudication_reason",
]


class BuildGoldTests(unittest.TestCase):
    def _paths(self, directory: Path) -> tuple[Path, Path, Path]:
        return (
            directory / "manifest.jsonl",
            directory / "annotations.csv",
            directory / "gold.jsonl",
        )

    def _write_manifest(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "id": "real-001",
                    "image": "tests/example.jpg",
                    "data_origin": "public_open_license",
                    "annotation_version": 1,
                }
            )
            + "\n",
            encoding="utf-8",
        )

    def _valid_annotation(self) -> dict[str, str]:
        return {
            "id": "real-001",
            "challenge_tags": "glare|old_worn",
            "reviewer_a": "reviewer-a",
            "reviewer_b": "reviewer-b",
            "adjudicator": "lead",
            "license_manual_verified": "true",
            "privacy_approved": "true",
            "expected_image_types": "nameplate|part_label",
            "brand_visibility": "readable",
            "expected_brand_any": "KOMATSU|小松",
            "machine_model_visibility": "not_present",
            "expected_machine_model_any": "",
            "part_name_candidate_visibility": "not_present",
            "expected_part_name_any": "",
            "part_number_visibility": "unreadable",
            "expected_part_number_any": "",
            "expected_damage_keywords": "",
            "should_reject": "true",
            "annotation_notes": "glare obscures the serial number",
        }

    def _write_annotations(self, path: Path, row: dict[str, str]) -> None:
        with path.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerow(row)

    def test_builds_gold_only_after_all_gates_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest, annotations, output = self._paths(
                Path(temporary_directory)
            )
            self._write_manifest(manifest)
            self._write_annotations(annotations, self._valid_annotation())

            rows = build_gold(manifest, annotations, output)

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["evaluation_status"], "gold")
            self.assertEqual(rows[0]["license_review_status"], "manual_verified")
            self.assertEqual(rows[0]["expected_brand_any"], ["KOMATSU", "小松"])
            self.assertTrue(rows[0]["should_reject"])
            self.assertTrue(output.exists())

    def test_rejects_same_reviewer_and_missing_readable_value(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest, annotations, output = self._paths(
                Path(temporary_directory)
            )
            self._write_manifest(manifest)
            row = self._valid_annotation()
            row["reviewer_b"] = "reviewer-a"
            row["expected_brand_any"] = ""
            self._write_annotations(annotations, row)

            with self.assertRaisesRegex(ValueError, "must be different"):
                build_gold(manifest, annotations, output)

            row = self._valid_annotation()
            row["adjudicator"] = "reviewer-a"
            self._write_annotations(annotations, row)
            with self.assertRaisesRegex(ValueError, "adjudicator must be different"):
                build_gold(manifest, annotations, output)

            row = self._valid_annotation()
            row["conflict_fields"] = "brand_visibility"
            row["adjudication_reason"] = ""
            self._write_annotations(annotations, row)
            with self.assertRaisesRegex(ValueError, "adjudication_reason"):
                build_gold(manifest, annotations, output)


if __name__ == "__main__":
    unittest.main()
