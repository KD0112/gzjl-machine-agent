from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


LOG_PATH = Path(__file__).resolve().parent / "logs" / "agent_runs.csv"

FIELDNAMES = [
    "run_id",
    "timestamp",
    "execution_mode",
    "status",
    "request_id",
    "thread_id",
    "customer_id",
    "approval_mode",
    "question",
    "intents",
    "slots",
    "parse_source",
    "parse_confidence",
    "model_runtime",
    "evidence_ids",
    "vision_status",
    "vision_results",
    "vision_model_runtime",
    "image_confirmation_decisions",
    "confirmed_visual_slots",
    "missing_fields",
    "called_tools",
    "skipped_tools",
    "unsupported_tools",
    "approval_decisions",
    "tool_errors",
    "tool_execution_keys",
    "handoff_id",
    "handoff_status",
    "handoff_priority",
    "handoff_reason",
    "assigned_agent",
    "human_reply",
    "tool_arguments",
    "tool_results",
    "execution_trace",
    "customer_reply",
    "error_message",
]

JSON_FIELDS = {
    "intents",
    "slots",
    "missing_fields",
    "called_tools",
    "skipped_tools",
    "unsupported_tools",
    "approval_decisions",
    "tool_errors",
    "tool_execution_keys",
    "handoff_reason",
    "tool_arguments",
    "tool_results",
    "execution_trace",
    "model_runtime",
    "evidence_ids",
    "vision_results",
    "vision_model_runtime",
    "image_confirmation_decisions",
    "confirmed_visual_slots",
}


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def append_agent_run(
    result: dict[str, Any],
    execution_mode: str,
    tool_arguments: dict[str, Any] | None = None,
    error_message: str = "",
    log_path: Path = LOG_PATH,
) -> str:
    """Append one complete Agent turn to the CSV log and return its run id."""
    if log_path.exists():
        _refresh_header_if_needed(log_path)

    parse_result = result.get("parse_result", {})
    run_id = uuid4().hex[:12]
    row = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "execution_mode": execution_mode,
        "status": result.get("status", ""),
        "request_id": result.get("request_id", ""),
        "thread_id": result.get("thread_id", ""),
        "customer_id": result.get("customer_id", ""),
        "approval_mode": result.get("approval_mode", ""),
        "question": parse_result.get("raw_question", ""),
        "intents": _json_dumps(parse_result.get("intents", [])),
        "slots": _json_dumps(parse_result.get("slots", {})),
        "parse_source": parse_result.get("parse_source", "rules"),
        "parse_confidence": parse_result.get("confidence", ""),
        "model_runtime": _json_dumps(
            result.get("model_runtime")
            or parse_result.get("debug", {}).get("model_runtime", {})
        ),
        "evidence_ids": _json_dumps(
            [
                item.get("evidence_id")
                for item in result.get("attachments", [])
                if item.get("evidence_id")
            ]
        ),
        "vision_status": result.get("vision_status", ""),
        "vision_results": _json_dumps(result.get("vision_results", [])),
        "vision_model_runtime": _json_dumps(
            result.get("vision_model_runtime", [])
        ),
        "image_confirmation_decisions": _json_dumps(
            result.get("image_confirmation_decisions", [])
        ),
        "confirmed_visual_slots": _json_dumps(
            result.get("confirmed_visual_slots", {})
        ),
        "missing_fields": _json_dumps(parse_result.get("missing_fields", [])),
        "called_tools": _json_dumps(result.get("called_tools", [])),
        "skipped_tools": _json_dumps(result.get("skipped_tools", [])),
        "unsupported_tools": _json_dumps(result.get("unsupported_tools", [])),
        "approval_decisions": _json_dumps(result.get("approval_decisions", [])),
        "tool_errors": _json_dumps(result.get("tool_errors", {})),
        "tool_execution_keys": _json_dumps(result.get("tool_execution_keys", {})),
        "handoff_id": result.get("handoff_id", ""),
        "handoff_status": result.get("handoff_status", ""),
        "handoff_priority": result.get("handoff_priority", ""),
        "handoff_reason": _json_dumps(result.get("handoff_reason", {})),
        "assigned_agent": result.get("assigned_agent", ""),
        "human_reply": result.get("human_reply", ""),
        "tool_arguments": _json_dumps(tool_arguments or result.get("tool_arguments", {})),
        "tool_results": _json_dumps(result.get("tool_results", {})),
        "execution_trace": _json_dumps(result.get("execution_trace", [])),
        "customer_reply": result.get("customer_reply", ""),
        "error_message": error_message,
    }

    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = log_path.exists()
    with log_path.open("a", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    return run_id


def _refresh_header_if_needed(log_path: Path) -> None:
    with log_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames == FIELDNAMES:
            return
        rows = [{field: row.get(field, "") for field in FIELDNAMES} for row in reader]

    with log_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _parse_json_field(value: str) -> Any:
    if not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def read_agent_runs(limit: int = 100, log_path: Path = LOG_PATH) -> list[dict[str, Any]]:
    """Read recent Agent run logs, newest first."""
    if not log_path.exists():
        return []

    with log_path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    recent_rows = rows[-limit:][::-1]
    parsed_rows: list[dict[str, Any]] = []
    for row in recent_rows:
        parsed = dict(row)
        for field in JSON_FIELDS:
            parsed[field] = _parse_json_field(parsed.get(field, ""))
        parsed_rows.append(parsed)
    return parsed_rows
