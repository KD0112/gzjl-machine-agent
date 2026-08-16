from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = BASE_DIR / "tests" / "multimodal_real_candidates.jsonl"
DEFAULT_ANNOTATIONS = (
    BASE_DIR / "tests" / "multimodal_real_annotation_template.csv"
)
DEFAULT_OUTPUT = BASE_DIR / "tests" / "multimodal_real_gold.jsonl"
VISIBILITY_VALUES = {"readable", "unreadable", "not_present"}
IMAGE_TYPES = {
    "nameplate",
    "part_label",
    "part",
    "damage",
    "document",
    "irrelevant",
    "unknown",
}
FIELD_COLUMNS = {
    "brand": ("brand_visibility", "expected_brand_any"),
    "machine_model": (
        "machine_model_visibility",
        "expected_machine_model_any",
    ),
    "part_name_candidate": (
        "part_name_candidate_visibility",
        "expected_part_name_any",
    ),
    "part_number": ("part_number_visibility", "expected_part_number_any"),
}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _split(value: str) -> list[str]:
    return [
        item.strip()
        for item in str(value or "").replace("；", "|").replace(";", "|").split("|")
        if item.strip()
    ]


def _boolean(value: str) -> bool | None:
    normalized = str(value or "").strip().casefold()
    if normalized in {"true", "1", "yes", "y", "是"}:
        return True
    if normalized in {"false", "0", "no", "n", "否"}:
        return False
    return None


def _validate_annotation(row: dict[str, str]) -> list[str]:
    errors = []
    reviewer_a = row.get("reviewer_a", "").strip()
    reviewer_b = row.get("reviewer_b", "").strip()
    adjudicator = row.get("adjudicator", "").strip()
    if not reviewer_a or not reviewer_b:
        errors.append("reviewer_a and reviewer_b are required")
    elif reviewer_a.casefold() == reviewer_b.casefold():
        errors.append("reviewer_a and reviewer_b must be different")
    if not adjudicator:
        errors.append("adjudicator is required")
    elif adjudicator.casefold() in {
        reviewer_a.casefold(),
        reviewer_b.casefold(),
    }:
        errors.append("adjudicator must be different from both reviewers")
    if _boolean(row.get("license_manual_verified", "")) is not True:
        errors.append("license_manual_verified must be true")
    if _boolean(row.get("privacy_approved", "")) is not True:
        errors.append("privacy_approved must be true")

    expected_types = _split(row.get("expected_image_types", ""))
    if not expected_types:
        errors.append("expected_image_types is required")
    unsupported_types = sorted(set(expected_types) - IMAGE_TYPES)
    if unsupported_types:
        errors.append(f"unsupported image types: {', '.join(unsupported_types)}")

    for field, (visibility_column, expected_column) in FIELD_COLUMNS.items():
        visibility = row.get(visibility_column, "").strip().casefold()
        expected_values = _split(row.get(expected_column, ""))
        if visibility not in VISIBILITY_VALUES:
            errors.append(
                f"{field} visibility must be readable/unreadable/not_present"
            )
        elif visibility == "readable" and not expected_values:
            errors.append(f"{field} readable requires an expected value")
        elif visibility != "readable" and expected_values:
            errors.append(f"{field} expected values require readable visibility")

    if _boolean(row.get("should_reject", "")) is None:
        errors.append("should_reject must be true or false")
    if row.get("conflict_fields", "").strip() and not row.get(
        "adjudication_reason",
        "",
    ).strip():
        errors.append("conflicting reviews require adjudication_reason")
    return errors


def build_gold(
    manifest_path: Path,
    annotations_path: Path,
    output_path: Path,
) -> list[dict[str, Any]]:
    manifest_rows = _load_jsonl(manifest_path)
    annotation_rows = _load_csv(annotations_path)
    manifest_by_id = {row["id"]: row for row in manifest_rows}
    annotation_by_id = {row.get("id", ""): row for row in annotation_rows}
    errors = []
    gold_rows = []

    missing_annotations = sorted(set(manifest_by_id) - set(annotation_by_id))
    extra_annotations = sorted(set(annotation_by_id) - set(manifest_by_id) - {""})
    if missing_annotations:
        errors.append(
            f"missing annotation rows: {', '.join(missing_annotations[:10])}"
        )
    if extra_annotations:
        errors.append(f"unknown annotation rows: {', '.join(extra_annotations[:10])}")

    for case_id, case in manifest_by_id.items():
        annotation = annotation_by_id.get(case_id)
        if not annotation:
            continue
        row_errors = _validate_annotation(annotation)
        if row_errors:
            errors.append(f"{case_id}: {', '.join(row_errors)}")
            continue

        field_visibility = {
            field: annotation[visibility_column].strip().casefold()
            for field, (visibility_column, _) in FIELD_COLUMNS.items()
        }
        gold_rows.append(
            {
                **case,
                "evaluation_status": "gold",
                "license_review_status": "manual_verified",
                "privacy_review_status": "approved",
                "annotation_version": int(case.get("annotation_version", 1)) + 1,
                "annotator_a": annotation["reviewer_a"].strip(),
                "annotator_b": annotation["reviewer_b"].strip(),
                "adjudicated_by": annotation["adjudicator"].strip(),
                "challenge_tags": _split(annotation["challenge_tags"]),
                "expected_image_types": _split(
                    annotation["expected_image_types"]
                ),
                "expected_brand_any": _split(annotation["expected_brand_any"]),
                "expected_machine_model_any": _split(
                    annotation["expected_machine_model_any"]
                ),
                "expected_part_name_any": _split(
                    annotation["expected_part_name_any"]
                ),
                "expected_part_number_any": _split(
                    annotation["expected_part_number_any"]
                ),
                "expected_damage_keywords": _split(
                    annotation["expected_damage_keywords"]
                ),
                "field_visibility": field_visibility,
                "should_reject": _boolean(annotation["should_reject"]),
                "annotation_notes": annotation.get("annotation_notes", "").strip(),
            }
        )

    if errors:
        message = "\n".join(f"- {error}" for error in errors[:60])
        raise ValueError(f"Gold dataset validation failed:\n{message}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for row in gold_rows:
            file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return gold_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        rows = build_gold(args.manifest, args.annotations, args.output)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"Built {len(rows)} gold cases.")
    print(f"Gold dataset: {args.output}")


if __name__ == "__main__":
    main()
