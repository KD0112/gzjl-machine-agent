from __future__ import annotations

from typing import Any

from schemas import HandoffDecision


HUMAN_REQUEST_KEYWORDS = (
    "转人工",
    "人工客服",
    "真人客服",
    "找客服",
    "找人工",
    "人工处理",
    "人工回复",
    "投诉",
)

EXPERT_REVIEW_INTENTS = {"compatibility", "diagnosis"}


def evaluate_handoff(state: dict[str, Any]) -> HandoffDecision:
    """Apply deterministic business rules for customer-service escalation."""
    if state.get("handoff_mode", "off") == "off" or state.get("human_reply"):
        return HandoffDecision(required=False)

    question = str(state.get("question", ""))
    if any(keyword in question for keyword in HUMAN_REQUEST_KEYWORDS):
        return HandoffDecision(
            required=True,
            reason_code="explicit_human_request",
            reason_text="客户明确要求人工客服接入。",
            priority="高",
        )

    vision_status = str(state.get("vision_status", ""))
    if vision_status == "failed":
        return HandoffDecision(
            required=True,
            reason_code="vision_provider_failure",
            reason_text="图片识别服务失败，已保留图片证据，需要人工客服核对。",
            priority="中",
        )
    if vision_status == "needs_better_image":
        return HandoffDecision(
            required=True,
            reason_code="vision_low_quality",
            reason_text="图片模糊、内容无关或没有可靠关键字段，需要补拍或人工核对。",
            priority="中",
        )
    if vision_status == "needs_human":
        return HandoffDecision(
            required=True,
            reason_code="vision_human_requested",
            reason_text="客户要求人工确认图片中的配件信息。",
            priority="高",
        )
    if vision_status == "rejected":
        return HandoffDecision(
            required=True,
            reason_code="vision_customer_rejected",
            reason_text="客户否认图片识别候选结果，需要人工重新核对证据。",
            priority="中",
        )

    tool_errors = state.get("tool_errors") or {}
    if tool_errors:
        failed_tools = "、".join(tool_errors)
        return HandoffDecision(
            required=True,
            reason_code="tool_failure",
            reason_text=f"工具重试后仍未成功：{failed_tools}。",
            priority="高",
        )

    unsupported_tools = state.get("unsupported_tools") or []
    if unsupported_tools:
        return HandoffDecision(
            required=True,
            reason_code="unsupported_capability",
            reason_text=f"当前系统尚未接入：{'、'.join(unsupported_tools)}。",
            priority="中",
        )

    tool_results = state.get("tool_results") or {}
    unmatched_tools = [
        name
        for name, result in tool_results.items()
        if isinstance(result, dict)
        and (
            result.get("matched") is False
            or result.get("retrieval_status") in {"no_docs", "low_confidence", "error"}
            or result.get("needs_handoff") is True
        )
    ]
    if unmatched_tools:
        return HandoffDecision(
            required=True,
            reason_code="no_reliable_result",
            reason_text=f"没有取得足够可靠的结果：{'、'.join(unmatched_tools)}。",
            priority="中",
        )

    parse_result = state.get("parse_result") or {}
    intents = set(parse_result.get("intents") or [])
    if "after_sales" in intents:
        return HandoffDecision(
            required=True,
            reason_code="after_sales_review",
            reason_text="售后结论涉及订单、证据和政策，需要人工客服处理。",
            priority="高",
        )

    expert_intents = sorted(intents & EXPERT_REVIEW_INTENTS)
    if expert_intents:
        return HandoffDecision(
            required=True,
            reason_code="expert_review",
            reason_text="适配或故障诊断需要配件顾问或技术支持确认。",
            priority="高",
        )

    missing_fields = parse_result.get("missing_fields") or []
    clarification_count = int(state.get("clarification_count", 0))
    if missing_fields and clarification_count >= 2:
        return HandoffDecision(
            required=True,
            reason_code="repeated_missing_information",
            reason_text="连续追问后关键信息仍不完整，需要人工协助收集。",
            priority="中",
        )

    confidence = float(parse_result.get("confidence", 1.0))
    if not intents or confidence < 0.45:
        return HandoffDecision(
            required=True,
            reason_code="low_parse_confidence",
            reason_text="系统无法稳定识别客户诉求，需要人工判断。",
            priority="中",
        )

    return HandoffDecision(required=False)
