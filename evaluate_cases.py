import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from rag_chat import answer_with_metadata, classify_question, generate_ticket_draft, retrieve


DEFAULT_CASES_PATH = Path("tests/customer_service_cases.jsonl")
REPORTS_DIR = Path("reports")


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            case = json.loads(line)
            required = {"id", "question", "expected_type", "should_follow_up"}
            missing = required - set(case)
            if missing:
                raise ValueError(f"Line {line_no} missing fields: {sorted(missing)}")
            cases.append(case)
    return cases


def docs_to_text(docs: list[Any]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def contains_any_keyword(text: str, keywords: list[str]) -> bool:
    if not keywords:
        return True
    return any(keyword in text for keyword in keywords)


def evaluate_case(case: dict[str, Any], k: int, with_llm: bool) -> dict[str, Any]:
    question = case["question"]
    if with_llm:
        result = answer_with_metadata(question, k=k)
        classification = result["classification"]
        docs = result["docs"]
        ticket = result["ticket_draft"]
        answer_text = result["answer"]
    else:
        classification = classify_question(question)
        docs = retrieve(question, k=k)
        ticket = generate_ticket_draft(question, classification, docs)
        answer_text = ""

    retrieved_text = docs_to_text(docs)
    retrieval_keywords = case.get("retrieval_keywords", [])
    actual_type = classification["type"]
    expected_type = case["expected_type"]
    actual_follow_up = classification["needs_follow_up"]
    expected_follow_up = bool(case["should_follow_up"])
    actual_owner = ticket["建议处理人"]
    expected_owner = case.get("expected_owner", actual_owner)

    classification_ok = actual_type == expected_type
    follow_up_ok = actual_follow_up == expected_follow_up
    owner_ok = actual_owner == expected_owner
    retrieval_hit = contains_any_keyword(retrieved_text, retrieval_keywords)

    issues = []
    if not classification_ok:
        issues.append("classification_wrong")
    if not follow_up_ok:
        issues.append("follow_up_wrong")
    if not owner_ok:
        issues.append("ticket_owner_wrong")
    if not retrieval_hit:
        issues.append("retrieval_miss")

    return {
        "id": case["id"],
        "question": question,
        "expected_type": expected_type,
        "actual_type": actual_type,
        "classification_ok": classification_ok,
        "expected_follow_up": expected_follow_up,
        "actual_follow_up": actual_follow_up,
        "follow_up_ok": follow_up_ok,
        "expected_owner": expected_owner,
        "actual_owner": actual_owner,
        "owner_ok": owner_ok,
        "retrieval_keywords": "、".join(retrieval_keywords),
        "retrieval_hit": retrieval_hit,
        "top_source": docs[0].metadata.get("source", "unknown") if docs else "none",
        "missing_fields": "、".join(classification["missing_fields"]),
        "suggested_questions": "；".join(classification["follow_up_questions"]),
        "ticket_priority": ticket["优先级"],
        "ticket_next_action": ticket["下一步动作"],
        "issues": "、".join(issues) if issues else "pass",
        "answer_preview": answer_text[:120].replace("\n", " ") if answer_text else "",
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def percent(ok_count: int, total: int) -> str:
    return f"{ok_count / total * 100:.1f}%"


def write_markdown(rows: list[dict[str, Any]], path: Path, k: int, with_llm: bool) -> None:
    total = len(rows)
    cls_ok = sum(row["classification_ok"] for row in rows)
    follow_ok = sum(row["follow_up_ok"] for row in rows)
    owner_ok = sum(row["owner_ok"] for row in rows)
    retrieval_ok = sum(row["retrieval_hit"] for row in rows)
    failed_rows = [row for row in rows if row["issues"] != "pass"]

    lines = [
        "# 客服 Agent 批量测试报告",
        "",
        f"- 测试时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 测试条数：{total}",
        f"- Top-K：{k}",
        f"- 是否调用大模型生成回答：{'是' if with_llm else '否'}",
        "",
        "## 汇总指标",
        "",
        f"- 问题分类准确率：{cls_ok}/{total} = {percent(cls_ok, total)}",
        f"- 主动追问判断准确率：{follow_ok}/{total} = {percent(follow_ok, total)}",
        f"- 工单处理人准确率：{owner_ok}/{total} = {percent(owner_ok, total)}",
        f"- 检索关键词命中率：{retrieval_ok}/{total} = {percent(retrieval_ok, total)}",
        "",
        "## Badcase 明细",
        "",
    ]

    if not failed_rows:
        lines.append("本轮没有发现自动规则层面的 badcase。")
    else:
        lines.extend(
            [
                "| ID | 问题 | 期望类型 | 实际类型 | 问题标签 |",
                "|---|---|---|---|---|",
            ]
        )
        for row in failed_rows:
            lines.append(
                f"| {row['id']} | {row['question']} | {row['expected_type']} | {row['actual_type']} | {row['issues']} |"
            )

    lines.extend(
        [
            "",
            "## 说明",
            "",
            "- 默认测试不调用 DeepSeek，因此不会逐条消耗 API token。",
            "- 这里的检索命中率是轻量检查：只看 Top-K 片段中是否出现预设关键词，不等同于最终回答准确率。",
            "- 如果要测试最终回答质量，可以加 `--with-llm`，但会调用 DeepSeek 30 次。",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate customer-service Agent cases.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--with-llm", action="store_true")
    args = parser.parse_args()

    REPORTS_DIR.mkdir(exist_ok=True)
    cases = load_cases(args.cases)
    rows = [evaluate_case(case, args.k, args.with_llm) for case in cases]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = REPORTS_DIR / f"evaluation_{timestamp}.csv"
    md_path = REPORTS_DIR / "evaluation_summary.md"
    write_csv(rows, csv_path)
    write_markdown(rows, md_path, args.k, args.with_llm)

    total = len(rows)
    failed = [row for row in rows if row["issues"] != "pass"]
    print(f"Evaluated {total} cases.")
    print(f"CSV report: {csv_path}")
    print(f"Markdown summary: {md_path}")
    print(f"Failed cases: {len(failed)}")
    for row in failed[:10]:
        print(f"- {row['id']} {row['issues']}: {row['question']}")


if __name__ == "__main__":
    main()
