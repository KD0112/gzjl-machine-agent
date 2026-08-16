from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Any

from handoff_repository import HandoffRepository


def _elapsed_seconds(start: str, end: str) -> float | None:
    if not start or not end:
        return None
    try:
        return max(0.0, (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds())
    except ValueError:
        return None


def summarize_handoffs(repository: HandoffRepository) -> dict[str, Any]:
    cases = repository.list_cases("all", limit=10000)
    status_counts = Counter(item["status"] for item in cases)
    reason_counts = Counter(item["reason_code"] for item in cases)
    priority_counts = Counter(item["priority"] for item in cases)
    resolution_times = [
        seconds
        for item in cases
        if (seconds := _elapsed_seconds(item["created_at"], item["resolved_at"])) is not None
    ]
    resolved = [item for item in cases if item["status"] == "resolved"]
    suggestion_adoptions = sum(
        bool(item.get("human_reply"))
        and item["human_reply"].strip()
        == str(item.get("context", {}).get("agent_suggested_reply", "")).strip()
        for item in resolved
    )
    return {
        "total": len(cases),
        "status_counts": dict(status_counts),
        "reason_counts": dict(reason_counts),
        "priority_counts": dict(priority_counts),
        "resolution_rate": round(len(resolved) / len(cases), 4) if cases else 0.0,
        "average_resolution_seconds": (
            round(sum(resolution_times) / len(resolution_times), 1)
            if resolution_times
            else None
        ),
        "suggestion_adoption_rate": (
            round(suggestion_adoptions / len(resolved), 4) if resolved else 0.0
        ),
    }
