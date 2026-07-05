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
    "question",
    "intents",
    "slots",
    "missing_fields",
    "called_tools",
    "unsupported_tools",
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
    "unsupported_tools",
    "tool_arguments",
    "tool_results",
    "execution_trace",
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
        "question": parse_result.get("raw_question", ""),
        "intents": _json_dumps(parse_result.get("intents", [])),
        "slots": _json_dumps(parse_result.get("slots", {})),
        "missing_fields": _json_dumps(parse_result.get("missing_fields", [])),
        "called_tools": _json_dumps(result.get("called_tools", [])),
        "unsupported_tools": _json_dumps(result.get("unsupported_tools", [])),
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
