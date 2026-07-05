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


def contains_all_keywords(text: str, keywords: list[str]) -> bool:
    if not keywords:
        return True
    return all(keyword in text for keyword in keywords)


def doc_source_text(docs: list[Any]) -> str:
    return "\n".join(str(doc.metadata.get("source", "")) for doc in docs)


def doc_metadata_values(docs: list[Any], key: str) -> list[str]:
    values = []
    for doc in docs:
        value = doc.metadata.get(key)
        if value and str(value) not in values:
            values.append(str(value))
    return values


def check_expected_metadata(docs: list[Any], key: str, expected_values: list[str]) -> bool:
    if not expected_values:
        return True
    actual_values = doc_metadata_values(docs, key)
    return all(expected in actual_values for expected in expected_values)


def check_expected_source(docs: list[Any], expected_keywords: list[str]) -> bool:
    if not expected_keywords:
        return True
    source_text = doc_source_text(docs)
    return contains_all_keywords(source_text, expected_keywords)


def case_list(case: dict[str, Any], key: str) -> list[str]:
    value = case.get(key, [])
    if isinstance(value, str):
        return [value]
    return list(value)


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
    expected_source_keywords = case_list(case, "expected_source_keywords")
    expected_categories = case_list(case, "expected_categories")
    expected_risk_levels = case_list(case, "expected_risk_levels")
    expected_missing_keywords = case_list(case, "expected_missing_keywords")
    expected_ticket_keywords = case_list(case, "expected_ticket_keywords")

    classification_ok = actual_type == expected_type
    follow_up_ok = actual_follow_up == expected_follow_up
    owner_ok = actual_owner == expected_owner
    retrieval_hit = contains_any_keyword(retrieved_text, retrieval_keywords)
    source_hit = check_expected_source(docs, expected_source_keywords)
    category_hit = check_expected_metadata(docs, "category", expected_categories)
    risk_level_hit = check_expected_metadata(docs, "risk_level", expected_risk_levels)
    missing_fields_text = "、".join(classification["missing_fields"])
    missing_fields_hit = contains_all_keywords(missing_fields_text, expected_missing_keywords)
    ticket_text = json.dumps(ticket, ensure_ascii=False)
    ticket_keywords_hit = contains_all_keywords(ticket_text, expected_ticket_keywords)

    issues = []
    if not classification_ok:
        issues.append("classification_wrong")
    if not follow_up_ok:
        issues.append("follow_up_wrong")
    if not owner_ok:
        issues.append("ticket_owner_wrong")
    if not retrieval_hit:
        issues.append("retrieval_miss")
    if not source_hit:
        issues.append("source_miss")
    if not category_hit:
        issues.append("category_miss")
    if not risk_level_hit:
        issues.append("risk_level_miss")
    if not missing_fields_hit:
        issues.append("missing_fields_miss")
    if not ticket_keywords_hit:
        issues.append("ticket_keywords_miss")

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
        "expected_source_keywords": "、".join(expected_source_keywords),
        "source_check_applicable": bool(expected_source_keywords),
        "source_hit": source_hit,
        "expected_categories": "、".join(expected_categories),
        "actual_categories": "、".join(doc_metadata_values(docs, "category")),
        "category_check_applicable": bool(expected_categories),
        "category_hit": category_hit,
        "expected_risk_levels": "、".join(expected_risk_levels),
        "actual_risk_levels": "、".join(doc_metadata_values(docs, "risk_level")),
        "risk_level_check_applicable": bool(expected_risk_levels),
        "risk_level_hit": risk_level_hit,
        "expected_missing_keywords": "、".join(expected_missing_keywords),
        "missing_fields_check_applicable": bool(expected_missing_keywords),
        "missing_fields_hit": missing_fields_hit,
        "expected_ticket_keywords": "、".join(expected_ticket_keywords),
        "ticket_keywords_check_applicable": bool(expected_ticket_keywords),
        "ticket_keywords_hit": ticket_keywords_hit,
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


def checked_metric(rows: list[dict[str, Any]], ok_key: str, applicable_key: str) -> tuple[int, int]:
    checked_rows = [row for row in rows if row[applicable_key]]
    return sum(row[ok_key] for row in checked_rows), len(checked_rows)


def write_markdown(rows: list[dict[str, Any]], path: Path, k: int, with_llm: bool, cases_path: Path) -> None:
    total = len(rows)
    cls_ok = sum(row["classification_ok"] for row in rows)
    follow_ok = sum(row["follow_up_ok"] for row in rows)
    owner_ok = sum(row["owner_ok"] for row in rows)
    retrieval_ok = sum(row["retrieval_hit"] for row in rows)
    source_ok, source_total = checked_metric(rows, "source_hit", "source_check_applicable")
    category_ok, category_total = checked_metric(rows, "category_hit", "category_check_applicable")
    risk_ok, risk_total = checked_metric(rows, "risk_level_hit", "risk_level_check_applicable")
    missing_ok, missing_total = checked_metric(rows, "missing_fields_hit", "missing_fields_check_applicable")
    ticket_keywords_ok, ticket_keywords_total = checked_metric(
        rows,
        "ticket_keywords_hit",
        "ticket_keywords_check_applicable",
    )
    failed_rows = [row for row in rows if row["issues"] != "pass"]

    lines = [
        "# 客服 Agent 批量测试报告",
        "",
        f"- 测试时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 用例文件：`{cases_path}`",
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
    ]

    enhanced_lines = []
    if missing_total:
        enhanced_lines.append(f"- 缺失字段关键词命中率：{missing_ok}/{missing_total} = {percent(missing_ok, missing_total)}")
    if ticket_keywords_total:
        enhanced_lines.append(
            f"- 工单关键词命中率：{ticket_keywords_ok}/{ticket_keywords_total} = {percent(ticket_keywords_ok, ticket_keywords_total)}"
        )
    if source_total:
        enhanced_lines.append(f"- 检索来源命中率：{source_ok}/{source_total} = {percent(source_ok, source_total)}")
    if category_total:
        enhanced_lines.append(
            f"- metadata category 命中率：{category_ok}/{category_total} = {percent(category_ok, category_total)}"
        )
    if risk_total:
        enhanced_lines.append(
            f"- metadata risk_level 命中率：{risk_ok}/{risk_total} = {percent(risk_ok, risk_total)}"
        )

    if enhanced_lines:
        lines.extend(["## 增强 RAG 检查", "", *enhanced_lines, ""])

    lines.extend(
        [
        "## Badcase 明细",
        "",
        ]
    )

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
            "- 检索关键词命中率是轻量检查：只看 Top-K 片段中是否出现预设关键词，不等同于最终回答准确率。",
            "- 增强 RAG 检查会额外验证命中文档、metadata、缺失字段和工单关键词，适合内部回归和面试展示。",
            f"- 如果要测试最终回答质量，可以加 `--with-llm`，但会调用 DeepSeek {total} 次。",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def is_default_cases_path(path: Path) -> bool:
    try:
        return path.resolve() == DEFAULT_CASES_PATH.resolve()
    except FileNotFoundError:
        return path == DEFAULT_CASES_PATH


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
    is_default_cases = is_default_cases_path(args.cases)
    report_prefix = "evaluation" if is_default_cases else f"evaluation_{args.cases.stem}"
    csv_path = REPORTS_DIR / f"{report_prefix}_{timestamp}.csv"
    md_path = REPORTS_DIR / ("evaluation_summary.md" if is_default_cases else f"evaluation_summary_{args.cases.stem}.md")
    write_csv(rows, csv_path)
    write_markdown(rows, md_path, args.k, args.with_llm, args.cases)

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
