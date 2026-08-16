from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable


UNSCORED_VISIBILITY = {"unreviewed", "unscored", "unknown", ""}


def normalize_text(value: Any) -> str:
    return "".join(str(value or "").strip().casefold().split())


def exact_match(actual: Any, expected_values: Iterable[str]) -> bool:
    actual_normalized = normalize_text(actual)
    return bool(actual_normalized) and any(
        actual_normalized == normalize_text(expected)
        for expected in expected_values
        if normalize_text(expected)
    )


def contains_match(actual: Any, expected_values: Iterable[str]) -> bool:
    actual_normalized = normalize_text(actual)
    return bool(actual_normalized) and any(
        normalize_text(expected) in actual_normalized
        for expected in expected_values
        if normalize_text(expected)
    )


@dataclass
class FieldCounts:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    skipped: int = 0

    @property
    def scored(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    @property
    def correct(self) -> int:
        return self.tp + self.tn

    def metrics(self) -> dict[str, int | float | None]:
        precision_denominator = self.tp + self.fp
        recall_denominator = self.tp + self.fn
        f1_denominator = 2 * self.tp + self.fp + self.fn
        return {
            **asdict(self),
            "scored": self.scored,
            "accuracy": _ratio(self.correct, self.scored),
            "precision": _ratio(self.tp, precision_denominator),
            "recall": _ratio(self.tp, recall_denominator),
            "f1": _ratio(2 * self.tp, f1_denominator),
        }


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return round(numerator / denominator, 4)


def score_field(
    *,
    actual: Any,
    expected_values: Iterable[str],
    visibility: str,
    matcher: Callable[[Any, Iterable[str]], bool] = exact_match,
) -> str:
    normalized_visibility = str(visibility or "").strip().casefold()
    if normalized_visibility in UNSCORED_VISIBILITY:
        return "skipped"

    expected = [value for value in expected_values if normalize_text(value)]
    expected_present = normalized_visibility == "readable"
    actual_present = bool(normalize_text(actual))

    if expected_present:
        if not expected:
            raise ValueError("readable fields require at least one expected value")
        if matcher(actual, expected):
            return "tp"
        return "fn_and_fp" if actual_present else "fn"

    if normalized_visibility not in {"not_present", "unreadable"}:
        raise ValueError(f"unsupported field visibility: {visibility}")
    return "fp" if actual_present else "tn"


def update_counts(counts: FieldCounts, outcome: str) -> None:
    if outcome == "fn_and_fp":
        counts.fn += 1
        counts.fp += 1
        return
    if not hasattr(counts, outcome):
        raise ValueError(f"unsupported field outcome: {outcome}")
    setattr(counts, outcome, getattr(counts, outcome) + 1)


def aggregate_field_metrics(
    rows: list[dict[str, Any]],
    field_specs: dict[str, dict[str, Any]],
) -> dict[str, dict[str, int | float | None]]:
    counts = {field: FieldCounts() for field in field_specs}
    for row in rows:
        for field, spec in field_specs.items():
            outcome = score_field(
                actual=row.get(spec["actual_key"]),
                expected_values=row.get(spec["expected_key"]) or [],
                visibility=row.get(spec["visibility_key"], "unreviewed"),
                matcher=spec.get("matcher", exact_match),
            )
            update_counts(counts[field], outcome)
    return {field: value.metrics() for field, value in counts.items()}


def aggregate_rejection_metrics(
    rows: list[dict[str, Any]],
) -> dict[str, int | float | None]:
    counts = FieldCounts()
    for row in rows:
        expected_reject = bool(row.get("expected_reject"))
        predicted_reject = bool(row.get("predicted_reject"))
        if expected_reject and predicted_reject:
            counts.tp += 1
        elif not expected_reject and predicted_reject:
            counts.fp += 1
        elif expected_reject and not predicted_reject:
            counts.fn += 1
        else:
            counts.tn += 1
    return counts.metrics()


def predict_rejection(inspection: dict[str, Any]) -> bool:
    if not inspection:
        return True
    if inspection.get("image_quality") in {"poor", "unusable"}:
        return True
    if inspection.get("image_type") in {"document", "irrelevant", "unknown"}:
        return True

    usable_evidence = any(
        [
            inspection.get("brand"),
            inspection.get("machine_model"),
            inspection.get("part_name_candidate"),
            inspection.get("part_number"),
            inspection.get("visible_damage"),
            inspection.get("extracted_text"),
        ]
    )
    confidence = float(inspection.get("confidence") or 0)
    return confidence < 0.5 and not usable_evidence


def metrics_by_group(
    rows: list[dict[str, Any]],
    key: str,
) -> dict[str, dict[str, int | float | None]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        values = row.get(key) or ["unclassified"]
        if isinstance(values, str):
            values = [values]
        for value in values:
            grouped[str(value)].append(row)

    result = {}
    for value, group_rows in sorted(grouped.items()):
        passed = sum(bool(row.get("passed")) for row in group_rows)
        expected_reject = sum(bool(row.get("expected_reject")) for row in group_rows)
        predicted_reject = sum(bool(row.get("predicted_reject")) for row in group_rows)
        result[value] = {
            "cases": len(group_rows),
            "passed": passed,
            "pass_rate": _ratio(passed, len(group_rows)),
            "expected_reject": expected_reject,
            "predicted_reject": predicted_reject,
        }
    return result


def percentile(values: Iterable[float], percentile_value: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return round(ordered[0], 2)
    position = (len(ordered) - 1) * percentile_value
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return round(
        ordered[lower] + (ordered[upper] - ordered[lower]) * fraction,
        2,
    )
