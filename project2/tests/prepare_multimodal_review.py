from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = BASE_DIR / "tests" / "multimodal_real_candidates.jsonl"
DEFAULT_OUTPUT = BASE_DIR / "reports" / "multimodal_real_review_workbook.csv"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _list_text(value: Any) -> str:
    if not value:
        return ""
    if isinstance(value, list):
        return "|".join(str(item) for item in value)
    return str(value)


def build_review_workbook(
    manifest_path: Path,
    predictions_path: Path,
    output_path: Path,
) -> int:
    candidates = _load_jsonl(manifest_path)
    predictions = {
        str(row["id"]): row for row in _load_jsonl(predictions_path)
    }
    missing = [
        str(case["id"])
        for case in candidates
        if str(case["id"]) not in predictions
    ]
    if missing:
        raise ValueError(f"missing prediction rows: {', '.join(missing[:10])}")

    fields = [
        "id",
        "image",
        "description",
        "scenario",
        "challenge_tags",
        "landing_url",
        "license",
        "model_image_type",
        "model_brand",
        "model_machine_model",
        "model_part_name_candidate",
        "model_part_number",
        "model_visible_damage",
        "model_image_quality",
        "model_confidence",
        "model_safe_for_auto_merge",
        "model_predicted_reject",
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for case in candidates:
            prediction = predictions[str(case["id"])]
            if (
                prediction.get("image_sha256")
                and prediction["image_sha256"] != case.get("sha256")
            ):
                raise ValueError(f"image hash mismatch: {case['id']}")
            writer.writerow(
                {
                    "id": case["id"],
                    "image": case["image"],
                    "description": case["description"],
                    "scenario": case["scenario"],
                    "challenge_tags": _list_text(case["challenge_tags"]),
                    "landing_url": case["landing_url"],
                    "license": case["license"],
                    "model_image_type": prediction.get("image_type", ""),
                    "model_brand": prediction.get("brand") or "",
                    "model_machine_model": prediction.get("machine_model") or "",
                    "model_part_name_candidate": (
                        prediction.get("part_name_candidate") or ""
                    ),
                    "model_part_number": prediction.get("part_number") or "",
                    "model_visible_damage": _list_text(
                        prediction.get("visible_damage")
                    ),
                    "model_image_quality": prediction.get("image_quality", ""),
                    "model_confidence": prediction.get("confidence", ""),
                    "model_safe_for_auto_merge": prediction.get(
                        "safe_for_auto_merge",
                        "",
                    ),
                    "model_predicted_reject": prediction.get(
                        "predicted_reject",
                        "",
                    ),
                    "brand_visibility": "unreviewed",
                    "machine_model_visibility": "unreviewed",
                    "part_name_candidate_visibility": "unreviewed",
                    "part_number_visibility": "unreviewed",
                }
            )
    return len(candidates)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        count = build_review_workbook(
            args.manifest,
            args.predictions,
            args.output,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"Prepared {count} review rows.")
    print(f"Review workbook: {args.output}")
    print("Model columns are suggestions only; two independent reviewers are required.")


if __name__ == "__main__":
    main()
