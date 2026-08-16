from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path
from typing import Any


PROJECT2_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT2_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT2_ROOT))

import agent_graph  # noqa: E402
from handoff_repository import HandoffRepository  # noqa: E402


CASES_PATH = Path(__file__).with_name("handoff_cases.jsonl")
REPORT_DIR = PROJECT2_ROOT / "reports"


def load_cases(path: Path = CASES_PATH) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def evaluate() -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        checkpointer = agent_graph.create_sqlite_checkpointer(root / "checkpoints.sqlite3")
        repository = HandoffRepository(root / "handoff.sqlite3")
        graph = agent_graph.build_graph(checkpointer, repository)
        try:
            for case in load_cases():
                result = agent_graph.start_graph_agent(
                    case["question"],
                    thread_id=case["id"],
                    approval_mode="auto",
                    handoff_mode="manual",
                    parser_mode="rules",
                    knowledge_mode=False,
                    clarification_count=case.get("clarification_count", 0),
                    graph=graph,
                )
                actual_handoff = result["status"] == "waiting_human"
                actual_reason = (result.get("handoff_reason") or {}).get("reason_code", "")
                checks = {
                    "status": result["status"] == case["expected_status"],
                    "handoff": actual_handoff == case["expected_handoff"],
                    "reason": actual_reason == case["expected_reason_code"],
                    "called_tools": set(result["called_tools"])
                    == set(case["expected_called_tools"]),
                    "case_persisted": bool(repository.get_case_by_thread(case["id"]))
                    == case["expected_handoff"],
                }
                results.append(
                    {
                        "id": case["id"],
                        "question": case["question"],
                        "passed": all(checks.values()),
                        "checks": checks,
                        "actual_status": result["status"],
                        "actual_handoff": actual_handoff,
                        "actual_reason_code": actual_reason,
                    }
                )
        finally:
            checkpointer.conn.close()
    return results


def write_reports(results: list[dict[str, Any]]) -> tuple[Path, Path]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = REPORT_DIR / "handoff_evaluation.csv"
    md_path = REPORT_DIR / "handoff_evaluation_summary.md"
    check_names = ["status", "handoff", "reason", "called_tools", "case_persisted"]

    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "id",
                "passed",
                *check_names,
                "actual_status",
                "actual_reason_code",
                "question",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "id": result["id"],
                    "passed": result["passed"],
                    **result["checks"],
                    "actual_status": result["actual_status"],
                    "actual_reason_code": result["actual_reason_code"],
                    "question": result["question"],
                }
            )

    total = len(results)
    passed = sum(item["passed"] for item in results)
    lines = [
        "# 人工接管策略评测",
        "",
        f"- 总用例数：{total}",
        f"- 通过用例数：{passed}",
        f"- 通过率：{passed / total * 100:.1f}%" if total else "- 通过率：0.0%",
        "",
        "## 分项",
        "",
    ]
    for name in check_names:
        count = sum(item["checks"][name] for item in results)
        lines.append(f"- {name}: {count}/{total}")
    failed = [item for item in results if not item["passed"]]
    lines.extend(["", "## 失败用例", ""])
    lines.extend(
        [f"- {item['id']}: {item['checks']}" for item in failed]
        or ["暂无失败用例。"]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, md_path


def main() -> None:
    results = evaluate()
    csv_path, md_path = write_reports(results)
    total = len(results)
    passed = sum(item["passed"] for item in results)
    print(f"Total: {total}")
    print(f"Passed: {passed}")
    print(f"Pass rate: {passed / total * 100:.1f}%" if total else "Pass rate: 0.0%")
    print(f"CSV report: {csv_path}")
    print(f"Summary report: {md_path}")
    if passed != total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
