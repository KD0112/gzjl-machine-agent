from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from tests.prepare_multimodal_double_review import (  # noqa: E402
    merge_reviews,
    prepare_packets,
)


TEMPLATE_FIELDS = [
    "id",
    "image",
    "description",
    "scenario",
    "challenge_tags",
    "landing_url",
    "license",
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
]


class MultimodalDoubleReviewTests(unittest.TestCase):
    def _write_template(self, path: Path) -> None:
        with path.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=TEMPLATE_FIELDS)
            writer.writeheader()
            writer.writerow(
                {
                    "id": "real-001",
                    "image": "tests/example.jpg",
                    "description": "test plate",
                    "scenario": "nameplate",
                    "challenge_tags": "glare|old_worn",
                    "landing_url": "https://example.com/source",
                    "license": "CC BY 4.0",
                }
            )

    @staticmethod
    def _read(path: Path) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            return list(csv.DictReader(file))

    @staticmethod
    def _write(path: Path, rows: list[dict[str, str]]) -> None:
        with path.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _complete(row: dict[str, str]) -> None:
        row.update(
            {
                "license_manual_verified": "true",
                "privacy_approved": "true",
                "expected_image_types": "nameplate",
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
            }
        )

    def test_prepare_packets_are_separate_and_blinded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template = root / "template.csv"
            packet_a = root / "a.csv"
            packet_b = root / "b.csv"
            self._write_template(template)

            count = prepare_packets(
                template,
                packet_a,
                packet_b,
                reviewer_a="alice",
                reviewer_b="bob",
            )

            self.assertEqual(count, 1)
            self.assertEqual(self._read(packet_a)[0]["reviewer"], "alice")
            self.assertEqual(self._read(packet_b)[0]["reviewer"], "bob")
            self.assertNotIn("model_brand", self._read(packet_a)[0])

    def test_merge_prefills_independent_agreements(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template = root / "template.csv"
            packet_a = root / "a.csv"
            packet_b = root / "b.csv"
            output = root / "adjudication.csv"
            self._write_template(template)
            prepare_packets(
                template,
                packet_a,
                packet_b,
                reviewer_a="alice",
                reviewer_b="bob",
            )
            rows_a = self._read(packet_a)
            rows_b = self._read(packet_b)
            self._complete(rows_a[0])
            self._complete(rows_b[0])
            rows_b[0]["expected_brand_any"] = "小松|KOMATSU"
            self._write(packet_a, rows_a)
            self._write(packet_b, rows_b)

            merged = merge_reviews(packet_a, packet_b, output)

            self.assertEqual(merged[0]["expected_brand_any"], "KOMATSU|小松")
            self.assertEqual(merged[0]["conflict_fields"], "")
            self.assertEqual(merged[0]["incomplete_fields"], "")
            self.assertEqual(merged[0]["reviewer_a"], "alice")
            self.assertEqual(merged[0]["reviewer_b"], "bob")

    def test_merge_blanks_conflicting_final_value_for_adjudication(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template = root / "template.csv"
            packet_a = root / "a.csv"
            packet_b = root / "b.csv"
            output = root / "adjudication.csv"
            self._write_template(template)
            prepare_packets(
                template,
                packet_a,
                packet_b,
                reviewer_a="alice",
                reviewer_b="bob",
            )
            rows_a = self._read(packet_a)
            rows_b = self._read(packet_b)
            self._complete(rows_a[0])
            self._complete(rows_b[0])
            rows_b[0]["part_number_visibility"] = "not_present"
            self._write(packet_a, rows_a)
            self._write(packet_b, rows_b)

            merged = merge_reviews(packet_a, packet_b, output)

            self.assertEqual(merged[0]["part_number_visibility"], "")
            self.assertIn("part_number_visibility", merged[0]["conflict_fields"])
            self.assertEqual(
                merged[0]["reviewer_a_part_number_visibility"],
                "unreadable",
            )
            self.assertEqual(
                merged[0]["reviewer_b_part_number_visibility"],
                "not_present",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
