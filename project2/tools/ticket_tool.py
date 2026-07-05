from __future__ import annotations

import argparse
import json
import re
from typing import Any


ISSUE_KEYWORDS = {
    "退货": ["退货", "买错", "不要了"],
    "换货": ["换货", "换一个", "换件"],
    "退款": ["退款", "退钱"],
    "质保": ["质保", "保修", "保内"],
}


def _detect_issue_type(text: str) -> str:
    normalized = text.strip().lower()
    for issue_type, keywords in ISSUE_KEYWORDS.items():
        if any(keyword in normalized for keyword in keywords):
            return issue_type
    return "售后咨询"


def _ticket_id(order_id: str) -> str:
    safe_order_id = re.sub(r"[^A-Za-z0-9]", "", order_id.upper())
    suffix = safe_order_id[-10:] if safe_order_id else "UNKNOWN"
    return f"AS-{suffix}"


def create_after_sales_ticket(
    order_id: str,
    raw_question: str,
    issue_type: str | None = None,
    evidence: str | None = None,
) -> dict[str, Any]:
    """Create a deterministic after-sales ticket draft for manual follow-up."""
    issue = issue_type or _detect_issue_type(raw_question)
    missing_items: list[str] = []
    if not evidence:
        missing_items.append("问题照片或视频")

    return {
        "ticket_id": _ticket_id(order_id),
        "ticket_status": "draft",
        "owner": "售后专员",
        "order_id": order_id,
        "issue_type": issue,
        "customer_request": raw_question,
        "missing_items": missing_items,
        "next_action": "人工客服核对订单、购买时间、配件状态和售后政策后再给出处理结论。",
        "need_manual_confirm": True,
        "risk_notice": "售后结论、退款、退货和换货必须由人工根据订单与证据确认，Agent 只生成工单草稿。",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a simulated after-sales ticket draft.")
    parser.add_argument("--order-id", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--issue-type")
    parser.add_argument("--evidence")
    args = parser.parse_args()

    result = create_after_sales_ticket(
        order_id=args.order_id,
        raw_question=args.question,
        issue_type=args.issue_type,
        evidence=args.evidence,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
