from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from agent_harness import ModelInvocationError
from image_evidence import validate_image_upload
from multimodal_evaluation import (
    aggregate_field_metrics,
    aggregate_rejection_metrics,
    contains_match,
    exact_match,
    metrics_by_group,
    percentile,
    predict_rejection,
    score_field,
)
from vision_service import VISION_PROMPT, inspect_image


DEFAULT_CASES = BASE_DIR / "tests" / "multimodal_cases.jsonl"
REPORT_DIR = BASE_DIR / "reports"
EVALUATOR_VERSION = "2026-07-27.2"
GOLD_STATUSES = {"gold", "synthetic_gold"}
FIELD_SPECS = {
    "brand": {
        "actual_key": "brand",
        "expected_key": "expected_brand_any",
        "visibility_key": "brand_visibility",
        "matcher": contains_match,
    },
    "machine_model": {
        "actual_key": "machine_model",
        "expected_key": "expected_machine_model_any",
        "visibility_key": "machine_model_visibility",
        "matcher": exact_match,
    },
    "part_name": {
        "actual_key": "part_name_candidate",
        "expected_key": "expected_part_name_any",
        "visibility_key": "part_name_candidate_visibility",
        "matcher": contains_match,
    },
    "part_number": {
        "actual_key": "part_number",
        "expected_key": "expected_part_number_any",
        "visibility_key": "part_number_visibility",
        "matcher": exact_match,
    },
}


def _load_cases(path: Path) -> list[dict[str, Any]]:
    cases = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            case = json.loads(line)
            case["_line_number"] = line_number
            cases.append(case)
    return cases


def _mime_type(path: Path) -> str:
    suffix = path.suffix.lower()
    mapping = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }
    if suffix not in mapping:
        raise ValueError(f"unsupported image extension: {suffix}")
    return mapping[suffix]


def _gold_errors(case: dict[str, Any]) -> list[str]:
    status = str(case.get("evaluation_status") or "").strip()
    origin = str(case.get("data_origin") or "").strip()
    errors = []
    if status not in GOLD_STATUSES:
        errors.append("evaluation_status is not gold")
    if origin == "public_open_license":
        if case.get("license_review_status") != "manual_verified":
            errors.append("license is not manually verified")
        if case.get("privacy_review_status") != "approved":
            errors.append("privacy review is not approved")
    if status == "gold":
        if not case.get("annotator_a") or not case.get("annotator_b"):
            errors.append("two annotators are required")
        if not case.get("adjudicated_by"):
            errors.append("adjudication is required")
        if not isinstance(case.get("should_reject"), bool):
            errors.append("should_reject must be boolean")
    return errors


def _expected_values(case: dict[str, Any], list_key: str, legacy_key: str) -> list[str]:
    values = case.get(list_key)
    if values is not None:
        return [str(value) for value in values if str(value).strip()]
    legacy_value = case.get(legacy_key)
    return [str(legacy_value)] if legacy_value else []


def _visibility(
    case: dict[str, Any],
    field: str,
    expected_values: list[str],
) -> str:
    explicit = (case.get("field_visibility") or {}).get(field)
    if explicit:
        return str(explicit)
    if case.get("evaluation_status") == "candidate":
        return "unreviewed"
    return "readable" if expected_values else "not_present"


def _damage_match(visible_damage: list[str], keywords: list[str]) -> bool:
    if not keywords:
        return True
    combined = " ".join(visible_damage).casefold()
    return any(keyword.casefold() in combined for keyword in keywords)


def _runtime_latency_ms(runtime: dict[str, Any]) -> float | None:
    events = runtime.get("events") or []
    if not events:
        return None
    total_values = [
        float(event["total_latency_ms"])
        for event in events
        if event.get("total_latency_ms") is not None
    ]
    if total_values:
        return max(total_values)
    latency_values = [
        float(event["latency_ms"])
        for event in events
        if event.get("latency_ms") is not None
    ]
    return round(sum(latency_values), 2) if latency_values else None


def _evaluate_case(case: dict[str, Any]) -> dict[str, Any]:
    image_path = BASE_DIR / case["image"]
    content = image_path.read_bytes()
    validated = validate_image_upload(
        content,
        filename=image_path.name,
        claimed_mime_type=_mime_type(image_path),
    )
    try:
        result = inspect_image(
            content=validated.content,
            mime_type=validated.mime_type,
            evidence_id=case["id"],
            local_quality=validated.local_quality,
            quality_signals=list(validated.quality_signals),
            request_id=f"eval-{case['id']}",
            thread_id=f"eval-{case['id']}",
        )
        inspection = result["inspection"]
        runtime = result["model_runtime"]
        error = ""
        error_type = ""
    except ModelInvocationError as exc:
        inspection = {}
        runtime = exc.snapshot
        error = str(exc)
        error_type = exc.error_type
    except Exception as exc:
        inspection = {}
        runtime = {}
        error = str(exc)
        error_type = type(exc).__name__

    expected_brand = _expected_values(case, "expected_brand_any", "expected_brand")
    expected_model = _expected_values(
        case,
        "expected_machine_model_any",
        "expected_machine_model",
    )
    expected_part_name = _expected_values(
        case,
        "expected_part_name_any",
        "expected_part_name",
    )
    expected_part_number = _expected_values(
        case,
        "expected_part_number_any",
        "expected_part_number",
    )
    visibility = {
        "brand": _visibility(case, "brand", expected_brand),
        "machine_model": _visibility(case, "machine_model", expected_model),
        "part_name_candidate": _visibility(
            case,
            "part_name_candidate",
            expected_part_name,
        ),
        "part_number": _visibility(case, "part_number", expected_part_number),
    }
    field_outcomes = {
        "brand": score_field(
            actual=inspection.get("brand"),
            expected_values=expected_brand,
            visibility=visibility["brand"],
            matcher=contains_match,
        ),
        "machine_model": score_field(
            actual=inspection.get("machine_model"),
            expected_values=expected_model,
            visibility=visibility["machine_model"],
        ),
        "part_name": score_field(
            actual=inspection.get("part_name_candidate"),
            expected_values=expected_part_name,
            visibility=visibility["part_name_candidate"],
            matcher=contains_match,
        ),
        "part_number": score_field(
            actual=inspection.get("part_number"),
            expected_values=expected_part_number,
            visibility=visibility["part_number"],
        ),
    }
    expected_types = case.get("expected_image_types") or []
    type_pass = (
        inspection.get("image_type") in expected_types if expected_types else True
    )
    damage_pass = _damage_match(
        inspection.get("visible_damage") or [],
        case.get("expected_damage_keywords") or [],
    )
    expected_reject = case.get("should_reject")
    predicted_reject = predict_rejection(inspection)
    rejection_pass = (
        True
        if not isinstance(expected_reject, bool)
        else predicted_reject == expected_reject
    )
    hallucinated_part_number = (
        visibility["part_number"] in {"not_present", "unreadable"}
        and bool(inspection.get("part_number"))
    )

    checks = {
        "schema_pass": bool(inspection),
        "type_pass": type_pass,
        "brand_pass": field_outcomes["brand"] in {"tp", "tn", "skipped"},
        "machine_model_pass": field_outcomes["machine_model"]
        in {"tp", "tn", "skipped"},
        "part_name_pass": field_outcomes["part_name"] in {"tp", "tn", "skipped"},
        "part_number_pass": field_outcomes["part_number"]
        in {"tp", "tn", "skipped"},
        "damage_pass": damage_pass,
        "rejection_pass": rejection_pass,
    }
    passed = not error and all(checks.values())
    route = runtime.get("route", {})
    return {
        "id": case["id"],
        "description": case.get("description", ""),
        "image": case["image"],
        "data_origin": case.get("data_origin", "synthetic"),
        "domain_match": case.get("domain_match", "synthetic"),
        "scenario": case.get("scenario", "unclassified"),
        "challenge_tags": case.get("challenge_tags") or ["unclassified"],
        "evaluation_status": case.get("evaluation_status", "legacy_synthetic"),
        "evaluator_version": EVALUATOR_VERSION,
        "image_sha256": hashlib.sha256(content).hexdigest(),
        "prompt_sha256": hashlib.sha256(
            VISION_PROMPT.encode("utf-8")
        ).hexdigest(),
        "passed": passed,
        **checks,
        "brand_outcome": field_outcomes["brand"],
        "machine_model_outcome": field_outcomes["machine_model"],
        "part_name_outcome": field_outcomes["part_name"],
        "part_number_outcome": field_outcomes["part_number"],
        "hallucinated_part_number": hallucinated_part_number,
        "expected_reject": expected_reject,
        "predicted_reject": predicted_reject,
        "provider": route.get("provider", ""),
        "model": route.get("model", ""),
        "attempts": runtime.get("attempts", 0),
        "estimated_input_tokens": runtime.get("estimated_input_tokens", 0),
        "latency_ms": _runtime_latency_ms(runtime),
        "image_type": inspection.get("image_type", ""),
        "brand": inspection.get("brand"),
        "machine_model": inspection.get("machine_model"),
        "part_name_candidate": inspection.get("part_name_candidate"),
        "part_number": inspection.get("part_number"),
        "visible_damage": inspection.get("visible_damage", []),
        "image_quality": inspection.get("image_quality", ""),
        "confidence": inspection.get("confidence"),
        "safe_for_auto_merge": inspection.get("safe_for_auto_merge"),
        "warnings": inspection.get("warnings", []),
        "required_followups": inspection.get("required_followups", []),
        "expected_brand_any": expected_brand,
        "expected_machine_model_any": expected_model,
        "expected_part_name_any": expected_part_name,
        "expected_part_number_any": expected_part_number,
        "brand_visibility": visibility["brand"],
        "machine_model_visibility": visibility["machine_model"],
        "part_name_candidate_visibility": visibility["part_name_candidate"],
        "part_number_visibility": visibility["part_number"],
        "error_type": error_type,
        "error": error,
    }


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, ensure_ascii=False)
                        if isinstance(value, (list, dict))
                        else value
                    )
                    for key, value in row.items()
                }
            )


def _write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _load_resume_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows = _load_cases(path)
    for row in rows:
        row.pop("_line_number", None)
    return {str(row["id"]): row for row in rows if row.get("passed")}


def _resume_row_matches(case: dict[str, Any], row: dict[str, Any]) -> bool:
    image_path = BASE_DIR / case["image"]
    image_sha256 = hashlib.sha256(image_path.read_bytes()).hexdigest()
    prompt_sha256 = hashlib.sha256(VISION_PROMPT.encode("utf-8")).hexdigest()
    return (
        row.get("image_sha256") == image_sha256
        and row.get("prompt_sha256") == prompt_sha256
    )


def _rescore_resume_row(
    case: dict[str, Any],
    row: dict[str, Any],
) -> dict[str, Any]:
    rescored = dict(row)
    expected_reject = case.get("should_reject")
    predicted_reject = predict_rejection(rescored)
    rescored["expected_reject"] = expected_reject
    rescored["predicted_reject"] = predicted_reject
    rescored["rejection_pass"] = (
        True
        if not isinstance(expected_reject, bool)
        else predicted_reject == expected_reject
    )
    check_keys = [
        "schema_pass",
        "type_pass",
        "brand_pass",
        "machine_model_pass",
        "part_name_pass",
        "part_number_pass",
        "damage_pass",
        "rejection_pass",
    ]
    rescored["passed"] = not rescored.get("error") and all(
        bool(rescored.get(key)) for key in check_keys
    )
    rescored["evaluator_version"] = EVALUATOR_VERSION
    return rescored


def _metric(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.2%}"
    return str(value)


def _escape(value: Any) -> str:
    return str(value if value not in {None, ""} else "-").replace("|", "/")


def _write_summary(rows: list[dict[str, Any]], path: Path, cases_path: Path) -> None:
    total = len(rows)
    passed = sum(bool(row["passed"]) for row in rows)
    candidate_run = any(
        row.get("evaluation_status") == "candidate" for row in rows
    )
    hallucinations = sum(bool(row["hallucinated_part_number"]) for row in rows)
    errors = sum(bool(row["error"]) for row in rows)
    provider = next((row["provider"] for row in rows if row["provider"]), "")
    model = next((row["model"] for row in rows if row["model"]), "")
    field_metrics = aggregate_field_metrics(rows, FIELD_SPECS)
    rejection_rows = [
        row for row in rows if isinstance(row.get("expected_reject"), bool)
    ]
    rejection_metrics = aggregate_rejection_metrics(rejection_rows)
    latency_values = [
        float(row["latency_ms"])
        for row in rows
        if row.get("latency_ms") is not None
    ]
    origins = sorted({str(row["data_origin"]) for row in rows})
    lines = [
        "# 多模态图片评测报告",
        "",
        f"生成时间：{datetime.now().isoformat(timespec='seconds')}",
        "",
        f"- 用例文件：`{cases_path}`",
        f"- 数据来源：{', '.join(origins) or '-'}",
        f"- Provider：`{provider or '未取得'}`",
        f"- 模型：`{model or '未取得'}`",
        f"- 总用例：{total}",
        (
            f"- 候选预跑完成：{passed}/{total}（不是准确率）"
            if candidate_run
            else f"- 整案通过：{passed}/{total}"
        ),
        f"- API/解析错误：{errors}",
        f"- 不可读或不存在时仍输出零件号：{hallucinations}",
        f"- 延迟 P50 / P95：{percentile(latency_values, 0.5) or '-'} / "
        f"{percentile(latency_values, 0.95) or '-'} ms",
        "",
        "## 字段级指标",
        "",
        "| 字段 | Scored | TP | FP | FN | TN | Accuracy | Precision | Recall | F1 | Skipped |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for field, metrics in field_metrics.items():
        lines.append(
            f"| {field} | {metrics['scored']} | {metrics['tp']} | "
            f"{metrics['fp']} | {metrics['fn']} | {metrics['tn']} | "
            f"{_metric(metrics['accuracy'])} | {_metric(metrics['precision'])} | "
            f"{_metric(metrics['recall'])} | {_metric(metrics['f1'])} | "
            f"{metrics['skipped']} |"
        )
    lines.extend(
        [
            "",
            "口径：错误但非空的字段同时计一次 FP 和一次 FN；`unreviewed` 字段跳过，"
            "不会被当作正确。",
            "",
            "## 拒识指标",
            "",
            f"- 已标注拒识用例：{len(rejection_rows)}",
            f"- TP / FP / FN / TN：{rejection_metrics['tp']} / "
            f"{rejection_metrics['fp']} / {rejection_metrics['fn']} / "
            f"{rejection_metrics['tn']}",
            f"- Precision / Recall / F1：{_metric(rejection_metrics['precision'])} / "
            f"{_metric(rejection_metrics['recall'])} / "
            f"{_metric(rejection_metrics['f1'])}",
            "",
            "## 场景分层",
            "",
            "| 标签 | 用例 | 通过 | 通过率 | 应拒识 | 实际拒识 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for tag, metrics in metrics_by_group(rows, "challenge_tags").items():
        lines.append(
            f"| {_escape(tag)} | {metrics['cases']} | {metrics['passed']} | "
            f"{_metric(metrics['pass_rate'])} | {metrics['expected_reject']} | "
            f"{metrics['predicted_reject']} |"
        )
    lines.extend(
        [
            "",
            "## 用例明细",
            "",
            "| ID | 场景 | 结果 | 类型 | 品牌 | 机型 | 零件号 | 质量 | 拒识 | 延迟 ms | 错误 |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | --- |",
        ]
    )
    for row in rows:
        result_label = (
            "预跑完成"
            if candidate_run and row["passed"]
            else ("通过" if row["passed"] else "失败")
        )
        lines.append(
            f"| {_escape(row['id'])} | {_escape(row['scenario'])} | "
            f"{result_label} | {_escape(row['image_type'])} | "
            f"{_escape(row['brand'])} | {_escape(row['machine_model'])} | "
            f"{_escape(row['part_number'])} | {_escape(row['image_quality'])} | "
            f"{'是' if row['predicted_reject'] else '否'} | "
            f"{_escape(row['latency_ms'])} | {_escape(row['error_type'])} |"
        )
    lines.extend(
        [
            "",
            "## 结论边界",
            "",
            "- `candidate`、`unreviewed` 和单人预标注样本不得用于宣称准确率。",
            "- 公开许可机械图片只能说明迁移与拒识表现，不能替代真实挖机配件业务集。",
            "- 对外结论必须同时写明样本数、数据来源、场景分布和字段可读性。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--ids",
        default="",
        help="Comma-separated case IDs for targeted badcase reruns.",
    )
    parser.add_argument(
        "--include-candidates",
        action="store_true",
        help="Run candidate images for pre-labeling; do not treat output as accuracy.",
    )
    parser.add_argument(
        "--require-gold",
        action="store_true",
        help="Fail before model calls unless every selected case passes gold gates.",
    )
    parser.add_argument(
        "--resume-jsonl",
        type=Path,
        help="Reuse successful rows when image and prompt hashes still match.",
    )
    args = parser.parse_args()

    cases_path = args.cases if args.cases.is_absolute() else BASE_DIR / args.cases
    cases = _load_cases(cases_path)
    selected_ids = {
        value.strip() for value in args.ids.split(",") if value.strip()
    }
    if selected_ids:
        cases = [case for case in cases if case.get("id") in selected_ids]
        missing_ids = selected_ids - {str(case.get("id")) for case in cases}
        if missing_ids:
            raise SystemExit(f"Unknown case IDs: {', '.join(sorted(missing_ids))}")
    if not args.include_candidates:
        cases = [
            case
            for case in cases
            if case.get("evaluation_status") != "candidate"
        ]
    if args.limit > 0:
        cases = cases[: args.limit]
    if not cases:
        raise SystemExit(
            "No eligible cases. Candidate datasets require --include-candidates; "
            "formal metrics require gold labels."
        )
    if args.require_gold:
        invalid = [
            (case["id"], _gold_errors(case))
            for case in cases
            if _gold_errors(case)
        ]
        if invalid:
            details = "; ".join(
                f"{case_id}: {', '.join(errors)}"
                for case_id, errors in invalid[:10]
            )
            raise SystemExit(f"Gold gate failed before model calls: {details}")

    resume_rows = (
        _load_resume_rows(args.resume_jsonl) if args.resume_jsonl else {}
    )
    rows = []
    reused = 0
    for index, case in enumerate(cases, start=1):
        cached = resume_rows.get(str(case["id"]))
        if cached and _resume_row_matches(case, cached):
            rows.append(_rescore_resume_row(case, cached))
            reused += 1
            print(f"[{index}/{len(cases)}] {case['id']}: reused")
            continue
        print(f"[{index}/{len(cases)}] {case['id']}: evaluating")
        rows.append(_evaluate_case(case))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = REPORT_DIR / f"multimodal_evaluation_{timestamp}.csv"
    jsonl_path = REPORT_DIR / f"multimodal_evaluation_{timestamp}.jsonl"
    summary_name = (
        "multimodal_evaluation_summary.md"
        if cases_path.resolve() == DEFAULT_CASES.resolve()
        else f"multimodal_evaluation_summary_{cases_path.stem}.md"
    )
    summary_path = REPORT_DIR / summary_name
    _write_csv(rows, csv_path)
    _write_jsonl(rows, jsonl_path)
    _write_summary(rows, summary_path, cases_path)

    failed = [row for row in rows if not row["passed"]]
    print(f"Evaluated {len(rows)} multimodal cases.")
    print(f"Passed: {len(rows) - len(failed)}/{len(rows)}")
    print(f"CSV report: {csv_path}")
    print(f"JSONL report: {jsonl_path}")
    print(f"Summary report: {summary_path}")
    print(f"Reused successful cases: {reused}")
    for row in failed:
        print(
            f"- {row['id']}: error={row['error_type'] or '-'} "
            f"type={row['image_type'] or '-'} confidence={row['confidence']}"
        )
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
