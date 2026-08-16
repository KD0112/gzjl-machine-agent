from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE = BASE_DIR / "tests" / "multimodal_real_annotation_template.csv"
DEFAULT_REVIEWER_A = BASE_DIR / "reports" / "multimodal_reviewer_a.csv"
DEFAULT_REVIEWER_B = BASE_DIR / "reports" / "multimodal_reviewer_b.csv"
DEFAULT_ADJUDICATION = (
    BASE_DIR / "reports" / "multimodal_adjudication_workbook.csv"
)

SOURCE_FIELDS = [
    "id",
    "image",
    "description",
    "scenario",
    "challenge_tags",
    "landing_url",
    "license",
]
DECISION_FIELDS = [
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
]
MULTI_VALUE_FIELDS = {
    "expected_image_types",
    "expected_brand_any",
    "expected_machine_model_any",
    "expected_part_name_any",
    "expected_part_number_any",
    "expected_damage_keywords",
}
BOOLEAN_FIELDS = {
    "license_manual_verified",
    "privacy_approved",
    "should_reject",
}
REQUIRED_DECISION_FIELDS = {
    "license_manual_verified",
    "privacy_approved",
    "expected_image_types",
    "brand_visibility",
    "machine_model_visibility",
    "part_name_candidate_visibility",
    "part_number_visibility",
    "should_reject",
}
PACKET_FIELDS = [
    "reviewer",
    *SOURCE_FIELDS,
    *DECISION_FIELDS,
    "reviewer_notes",
]
ADJUDICATION_FIELDS = [
    *SOURCE_FIELDS,
    "reviewer_a",
    "reviewer_b",
    "adjudicator",
    *DECISION_FIELDS,
    "annotation_notes",
    "conflict_fields",
    "incomplete_fields",
    "adjudication_reason",
    *[f"reviewer_a_{field}" for field in DECISION_FIELDS],
    "reviewer_a_notes",
    *[f"reviewer_b_{field}" for field in DECISION_FIELDS],
    "reviewer_b_notes",
]


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _split(value: Any) -> list[str]:
    return [
        item.strip()
        for item in _clean(value).replace("；", "|").replace(";", "|").split("|")
        if item.strip()
    ]


def _boolean(value: Any) -> bool | None:
    normalized = _clean(value).casefold()
    if normalized in {"true", "1", "yes", "y", "是"}:
        return True
    if normalized in {"false", "0", "no", "n", "否"}:
        return False
    return None


def _normalized(field: str, value: Any) -> Any:
    if field in MULTI_VALUE_FIELDS:
        return tuple(sorted(item.casefold() for item in _split(value)))
    if field in BOOLEAN_FIELDS:
        return _boolean(value)
    return _clean(value).casefold()


def _canonical_agreement(field: str, value: Any) -> str:
    if field in BOOLEAN_FIELDS:
        parsed = _boolean(value)
        if parsed is None:
            return ""
        return "true" if parsed else "false"
    if field in MULTI_VALUE_FIELDS:
        return "|".join(_split(value))
    return _clean(value)


def prepare_packets(
    template_path: Path,
    reviewer_a_path: Path,
    reviewer_b_path: Path,
    *,
    reviewer_a: str,
    reviewer_b: str,
) -> int:
    name_a = _clean(reviewer_a)
    name_b = _clean(reviewer_b)
    if not name_a or not name_b:
        raise ValueError("reviewer names are required")
    if name_a.casefold() == name_b.casefold():
        raise ValueError("reviewer names must be different")

    template_rows = _load_csv(template_path)

    def build_rows(reviewer: str) -> list[dict[str, str]]:
        packets = []
        for source in template_rows:
            row = {"reviewer": reviewer}
            row.update({field: _clean(source.get(field, "")) for field in SOURCE_FIELDS})
            row.update({field: "" for field in DECISION_FIELDS})
            for field in DECISION_FIELDS:
                if field.endswith("_visibility"):
                    row[field] = "unreviewed"
            row["reviewer_notes"] = ""
            packets.append(row)
        return packets

    _write_csv(reviewer_a_path, build_rows(name_a), PACKET_FIELDS)
    _write_csv(reviewer_b_path, build_rows(name_b), PACKET_FIELDS)
    return len(template_rows)


def _index_reviews(
    rows: list[dict[str, str]],
    path: Path,
) -> tuple[str, dict[str, dict[str, str]]]:
    reviewers = {_clean(row.get("reviewer", "")) for row in rows}
    reviewers.discard("")
    if len(reviewers) != 1:
        raise ValueError(f"{path}: exactly one reviewer identity is required")
    by_id = {_clean(row.get("id", "")): row for row in rows}
    if "" in by_id or len(by_id) != len(rows):
        raise ValueError(f"{path}: ids must be non-empty and unique")
    return next(iter(reviewers)), by_id


def merge_reviews(
    reviewer_a_path: Path,
    reviewer_b_path: Path,
    output_path: Path,
) -> list[dict[str, str]]:
    name_a, rows_a = _index_reviews(_load_csv(reviewer_a_path), reviewer_a_path)
    name_b, rows_b = _index_reviews(_load_csv(reviewer_b_path), reviewer_b_path)
    if name_a.casefold() == name_b.casefold():
        raise ValueError("reviewer identities must be different")
    if set(rows_a) != set(rows_b):
        missing_a = sorted(set(rows_b) - set(rows_a))
        missing_b = sorted(set(rows_a) - set(rows_b))
        raise ValueError(
            "review packets contain different ids: "
            f"missing_a={missing_a[:5]}, missing_b={missing_b[:5]}"
        )

    output_rows = []
    for case_id, row_a in rows_a.items():
        row_b = rows_b[case_id]
        for source_field in SOURCE_FIELDS:
            if _clean(row_a.get(source_field)) != _clean(row_b.get(source_field)):
                raise ValueError(f"{case_id}: source field mismatch: {source_field}")

        merged = {
            field: _clean(row_a.get(field, ""))
            for field in SOURCE_FIELDS
        }
        merged.update(
            {
                "reviewer_a": name_a,
                "reviewer_b": name_b,
                "adjudicator": "",
                "annotation_notes": "",
                "adjudication_reason": "",
            }
        )
        conflicts = []
        incomplete = []
        for field in DECISION_FIELDS:
            value_a = _clean(row_a.get(field, ""))
            value_b = _clean(row_b.get(field, ""))
            normalized_a = _normalized(field, value_a)
            normalized_b = _normalized(field, value_b)
            is_incomplete = field in REQUIRED_DECISION_FIELDS and (
                normalized_a in {"", (), None}
                or normalized_b in {"", (), None}
                or normalized_a == "unreviewed"
                or normalized_b == "unreviewed"
            )
            if is_incomplete:
                incomplete.append(field)
            if normalized_a == normalized_b and not is_incomplete:
                merged[field] = _canonical_agreement(field, value_a)
            else:
                merged[field] = ""
                if normalized_a != normalized_b:
                    conflicts.append(field)
            merged[f"reviewer_a_{field}"] = value_a
            merged[f"reviewer_b_{field}"] = value_b

        merged["reviewer_a_notes"] = _clean(row_a.get("reviewer_notes", ""))
        merged["reviewer_b_notes"] = _clean(row_b.get("reviewer_notes", ""))
        merged["conflict_fields"] = "|".join(conflicts)
        merged["incomplete_fields"] = "|".join(incomplete)
        output_rows.append(merged)

    _write_csv(output_path, output_rows, ADJUDICATION_FIELDS)
    return output_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    packets_parser = subparsers.add_parser("packets")
    packets_parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    packets_parser.add_argument("--reviewer-a-output", type=Path, default=DEFAULT_REVIEWER_A)
    packets_parser.add_argument("--reviewer-b-output", type=Path, default=DEFAULT_REVIEWER_B)
    packets_parser.add_argument("--reviewer-a", required=True)
    packets_parser.add_argument("--reviewer-b", required=True)

    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("--reviewer-a-input", type=Path, default=DEFAULT_REVIEWER_A)
    merge_parser.add_argument("--reviewer-b-input", type=Path, default=DEFAULT_REVIEWER_B)
    merge_parser.add_argument("--output", type=Path, default=DEFAULT_ADJUDICATION)

    args = parser.parse_args()
    try:
        if args.command == "packets":
            count = prepare_packets(
                args.template,
                args.reviewer_a_output,
                args.reviewer_b_output,
                reviewer_a=args.reviewer_a,
                reviewer_b=args.reviewer_b,
            )
            print(f"Prepared {count} blinded rows for each reviewer.")
            print(f"Reviewer A packet: {args.reviewer_a_output}")
            print(f"Reviewer B packet: {args.reviewer_b_output}")
        else:
            rows = merge_reviews(
                args.reviewer_a_input,
                args.reviewer_b_input,
                args.output,
            )
            conflict_count = sum(bool(row["conflict_fields"]) for row in rows)
            incomplete_count = sum(bool(row["incomplete_fields"]) for row in rows)
            print(f"Merged {len(rows)} rows.")
            print(f"Rows with conflicts: {conflict_count}")
            print(f"Rows with incomplete fields: {incomplete_count}")
            print(f"Adjudication workbook: {args.output}")
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
