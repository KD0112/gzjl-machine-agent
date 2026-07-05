from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


PROJECT2_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT2_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT2_ROOT))

from agent_workflow import run_agent  # noqa: E402
from agent_graph import run_graph_agent  # noqa: E402


DEFAULT_CASES_PATH = Path(__file__).with_name("agent_cases.jsonl")
DEFAULT_REPORT_DIR = PROJECT2_ROOT / "reports"
FORBIDDEN_CUSTOMER_WORDS = [
    "tool_results",
    "tool_arguments",
    "inventory_tool",
    "quote_tool",
    "logistics_tool",
    "ticket_tool",
    "unsupported_tools",
    "模拟库存",
    "quote_status",
    "draft",
]


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc
    return cases


def _as_set(value: list[str] | None) -> set[str]:
    return set(value or [])


def _slot_matches(actual_slots: dict[str, Any], expected_slots: dict[str, Any]) -> bool:
    for key, expected_value in expected_slots.items():
        if actual_slots.get(key) != expected_value:
            return False
    return True


def _contains_all(text: str, keywords: list[str]) -> bool:
    return all(keyword in text for keyword in keywords)


def _trace_step_names(trace: list[dict[str, Any]]) -> list[str]:
    return [str(item.get("step", "")) for item in trace]


def _trace_tool_names(trace: list[dict[str, Any]]) -> list[str]:
    return [str(item.get("tool_name")) for item in trace if item.get("tool_name")]


def _optional_equals(case: dict[str, Any], key: str, actual_value: Any) -> bool:
    return key not in case or actual_value == case[key]


def _contains_all_values(actual_values: list[str], expected_values: list[str]) -> bool:
    return all(value in actual_values for value in expected_values)


def evaluate_case(case: dict[str, Any], mode: str) -> dict[str, Any]:
    runner = run_graph_agent if mode == "graph" else run_agent
    result = runner(case["question"])
    parse_result = result["parse_result"]
    reply = result["customer_reply"]
    trace = result.get("execution_trace", [])
    trace_step_names = _trace_step_names(trace)
    trace_tool_names = _trace_tool_names(trace)

    checks = {
        "status": _optional_equals(case, "expected_status", result["status"]),
        "intents": _as_set(parse_result["intents"]) == _as_set(case.get("expected_intents")),
        "called_tools": _as_set(result["called_tools"]) == _as_set(case.get("expected_called_tools")),
        "unsupported_tools": _as_set(result["unsupported_tools"]) == _as_set(case.get("expected_unsupported_tools")),
        "missing_fields": _as_set(parse_result["missing_fields"]) == _as_set(case.get("expected_missing_fields")),
        "slots": _slot_matches(parse_result["slots"], case.get("expected_slots", {})),
        "ready_for_tools": _optional_equals(case, "expected_ready_for_tools", parse_result["ready_for_tools"]),
        "reply_keywords": _contains_all(reply, case.get("expected_reply_keywords", [])),
        "no_internal_words": not any(word in reply for word in FORBIDDEN_CUSTOMER_WORDS),
        "trace_present": bool(trace) and trace_step_names[0] == "parse" and trace_step_names[-1] == "build_response",
        "trace_tools_match": trace_tool_names == result["called_tools"],
        "trace_expected_steps": _contains_all_values(trace_step_names, case.get("expected_trace_steps", [])),
        "trace_expected_tools": case.get("expected_trace_tools") is None
        or trace_tool_names == case.get("expected_trace_tools", []),
    }

    return {
        "id": case["id"],
        "question": case["question"],
        "passed": all(checks.values()),
        "checks": checks,
        "actual_intents": parse_result["intents"],
        "actual_called_tools": result["called_tools"],
        "actual_unsupported_tools": result["unsupported_tools"],
        "actual_missing_fields": parse_result["missing_fields"],
        "actual_slots": parse_result["slots"],
        "actual_status": result["status"],
        "actual_ready_for_tools": parse_result["ready_for_tools"],
        "actual_trace_steps": trace_step_names,
        "actual_trace_tools": trace_tool_names,
        "customer_reply": reply,
    }


def _is_default_cases(path: Path) -> bool:
    try:
        return path.resolve() == DEFAULT_CASES_PATH.resolve()
    except FileNotFoundError:
        return path == DEFAULT_CASES_PATH


def report_paths(report_dir: Path, mode: str, cases_path: Path) -> tuple[Path, Path]:
    if _is_default_cases(cases_path):
        return (
            report_dir / f"agent_evaluation_{mode}.csv",
            report_dir / f"agent_evaluation_summary_{mode}.md",
        )
    stem = cases_path.stem
    return (
        report_dir / f"agent_evaluation_{stem}_{mode}.csv",
        report_dir / f"agent_evaluation_summary_{stem}_{mode}.md",
    )


def write_reports(results: list[dict[str, Any]], report_dir: Path, mode: str, cases_path: Path) -> tuple[Path, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path, md_path = report_paths(report_dir, mode, cases_path)

    check_names = [
        "status",
        "intents",
        "called_tools",
        "unsupported_tools",
        "missing_fields",
        "slots",
        "ready_for_tools",
        "reply_keywords",
        "no_internal_words",
        "trace_present",
        "trace_tools_match",
        "trace_expected_steps",
        "trace_expected_tools",
    ]

    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["id", "passed", *check_names, "question"])
        writer.writeheader()
        for item in results:
            writer.writerow(
                {
                    "id": item["id"],
                    "passed": item["passed"],
                    **item["checks"],
                    "question": item["question"],
                }
            )

    total = len(results)
    passed = sum(1 for item in results if item["passed"])
    lines = [
        f"# 多工具 Agent 评估报告（{mode}）",
        "",
        f"- 用例文件：`{cases_path}`",
        f"- 总用例数：{total}",
        f"- 通过用例数：{passed}",
        f"- 通过率：{passed / total * 100:.1f}%" if total else "- 通过率：0.0%",
        "",
        "## 分项通过率",
        "",
    ]

    for check_name in check_names:
        check_passed = sum(1 for item in results if item["checks"][check_name])
        lines.append(f"- {check_name}: {check_passed}/{total} = {check_passed / total * 100:.1f}%")

    failed = [item for item in results if not item["passed"]]
    lines.extend(["", "## 失败用例", ""])
    if not failed:
        lines.append("暂无失败用例。")
    else:
        for item in failed:
            failed_checks = [name for name, ok in item["checks"].items() if not ok]
            lines.append(f"- {item['id']}：{', '.join(failed_checks)}")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate project2 multi-tool agent.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--mode", choices=["workflow", "graph"], default="workflow")
    args = parser.parse_args()

    cases = load_cases(args.cases)
    results = [evaluate_case(case, args.mode) for case in cases]
    csv_path, md_path = write_reports(results, args.report_dir, args.mode, args.cases)

    total = len(results)
    passed = sum(1 for item in results if item["passed"])
    print(f"Total: {total}")
    print(f"Mode: {args.mode}")
    print(f"Passed: {passed}")
    print(f"Pass rate: {passed / total * 100:.1f}%" if total else "Pass rate: 0.0%")
    print(f"CSV report: {csv_path}")
    print(f"Summary report: {md_path}")

    failed = [item for item in results if not item["passed"]]
    if failed:
        print("Failed cases:")
        for item in failed:
            failed_checks = [name for name, ok in item["checks"].items() if not ok]
            print(f"- {item['id']}: {', '.join(failed_checks)}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
