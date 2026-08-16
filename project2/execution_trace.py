from __future__ import annotations

from typing import Any


TOOL_TITLES = {
    "inventory_tool": "查询库存",
    "quote_tool": "生成报价草稿",
    "logistics_tool": "估算物流",
    "ticket_tool": "生成售后工单草稿",
    "knowledge_tool": "查询企业知识库",
}


def parse_step(parse_result: dict[str, Any]) -> dict[str, Any]:
    missing_fields = parse_result.get("missing_fields") or []
    intents = parse_result.get("intents") or []
    return {
        "step": "parse",
        "title": "解析客户问题",
        "status": "need_more_info" if missing_fields else "completed",
        "summary": f"识别到 {len(intents)} 个意图，抽取槽位并检查缺失字段。",
        "data": {
            "intents": intents,
            "slots": parse_result.get("slots", {}),
            "missing_fields": missing_fields,
            "matched_keywords": parse_result.get("debug", {}).get("matched_keywords", {}),
            "candidate_tools": parse_result.get("candidate_tools", []),
            "ready_for_tools": parse_result.get("ready_for_tools", False),
            "parse_source": parse_result.get("parse_source", "rules"),
            "confidence": parse_result.get("confidence"),
            "langchain_error": parse_result.get("debug", {}).get("langchain_error", ""),
        },
    }


def model_runtime_step(model_runtime: dict[str, Any]) -> dict[str, Any]:
    route = model_runtime.get("route", {})
    successes = int(model_runtime.get("successes", 0))
    failures = int(model_runtime.get("failures", 0))
    status = "completed" if successes else "fallback"
    provider = route.get("provider") or "unconfigured"
    model = route.get("model") or "unknown"
    return {
        "step": "model_harness",
        "title": "执行模型 Harness",
        "status": status,
        "summary": (
            f"{provider}/{model}，调用 {model_runtime.get('calls', 0)} 次，"
            f"尝试 {model_runtime.get('attempts', 0)} 次，失败 {failures} 次。"
        ),
        "data": {
            "trace_id": model_runtime.get("trace_id"),
            "route": route,
            "calls": model_runtime.get("calls", 0),
            "attempts": model_runtime.get("attempts", 0),
            "successes": successes,
            "failures": failures,
            "estimated_input_tokens": model_runtime.get("estimated_input_tokens", 0),
            "reserved_output_tokens": model_runtime.get("reserved_output_tokens", 0),
            "estimated_cost_cny": model_runtime.get("estimated_cost_cny", 0),
            "last_error_type": model_runtime.get("last_error_type", ""),
            "policy": model_runtime.get("policy", {}),
        },
    }


def image_inspection_step(
    vision_results: list[dict[str, Any]],
    *,
    vision_status: str,
    error: str = "",
) -> dict[str, Any]:
    inspections = [
        item.get("inspection", {})
        for item in vision_results
        if item.get("inspection")
    ]
    return {
        "step": "inspect_image",
        "title": "提取图片证据",
        "status": vision_status,
        "summary": (
            f"处理 {len(vision_results)} 张图片，"
            f"取得 {len(inspections)} 份结构化证据。"
        ),
        "data": {
            "evidence_ids": [item.get("evidence_id") for item in vision_results],
            "image_types": [item.get("image_type") for item in inspections],
            "confidence": [item.get("confidence") for item in inspections],
            "image_quality": [item.get("image_quality") for item in inspections],
            "safe_for_auto_merge": [
                item.get("safe_for_auto_merge") for item in inspections
            ],
            "error": error,
        },
    }


def image_confirmation_step(
    *,
    decision: str,
    confirmed_fields: dict[str, Any],
    comment: str = "",
) -> dict[str, Any]:
    return {
        "step": "confirm_image_evidence",
        "title": "确认图片证据",
        "status": decision,
        "summary": (
            f"客户对图片候选信息选择 {decision}，"
            f"确认 {len(confirmed_fields)} 个业务字段。"
        ),
        "data": {
            "decision": decision,
            "confirmed_fields": confirmed_fields,
            "comment": comment,
        },
    }


def context_step(
    context_snapshot: dict[str, Any],
    *,
    turn_count: int,
    conflicts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    conflicts = conflicts or []
    return {
        "step": "build_context",
        "title": "构建受控上下文",
        "status": "conflict_review" if conflicts else "completed",
        "summary": (
            f"第 {turn_count} 轮上下文已按优先级装配，"
            f"估算 {context_snapshot.get('estimated_tokens', 0)}/"
            f"{context_snapshot.get('max_tokens', 0)} tokens。"
        ),
        "data": {
            "policy_version": context_snapshot.get("policy_version"),
            "sections": context_snapshot.get("sections", []),
            "dropped_sections": context_snapshot.get("dropped_sections", []),
            "dropped_messages": context_snapshot.get("dropped_messages", 0),
            "injection_signals": context_snapshot.get("injection_signals", []),
            "conflicts": conflicts,
        },
    }


def memory_step(writes: list[dict[str, Any]], customer_id: str) -> dict[str, Any]:
    accepted = [item for item in writes if item.get("status") != "rejected"]
    rejected = [item for item in writes if item.get("status") == "rejected"]
    return {
        "step": "persist_memory",
        "title": "更新客户长期记忆",
        "status": "completed" if not rejected else "partially_rejected",
        "summary": (
            f"为客户 {customer_id} 更新 {len(accepted)} 条受治理的资料，"
            f"策略拒绝 {len(rejected)} 条。"
        ),
        "data": {
            "customer_id": customer_id,
            "accepted_fact_types": [item.get("fact_type") for item in accepted],
            "rejected": rejected,
        },
    }


def missing_fields_step(parse_result: dict[str, Any]) -> dict[str, Any]:
    missing_fields = parse_result.get("missing_fields") or []
    return {
        "step": "guard_missing_fields",
        "title": "缺失字段拦截",
        "status": "need_more_info",
        "summary": "关键信息不足，停止工具调用并生成追问。",
        "data": {
            "missing_fields": missing_fields,
            "follow_up": parse_result.get("follow_up"),
        },
    }


def _tool_summary(tool_name: str, result: dict[str, Any]) -> str:
    if tool_name == "inventory_tool":
        if result.get("in_stock"):
            return f"命中库存，{result.get('warehouse')} 可用 {result.get('stock_count')} 件。"
        if result.get("matched"):
            return "命中配件记录，但当前库存不足，需要人工确认调货。"
        return "未命中库存记录，需要人工核对型号、配件和品质档位。"

    if tool_name == "quote_tool":
        if result.get("matched"):
            return f"生成参考报价区间 {result.get('total_price_range')} 元，正式价格需人工确认。"
        return "未命中自动报价规则，需要人工核价。"

    if tool_name == "logistics_tool":
        if result.get("matched"):
            return f"估算发往 {result.get('city')}，时效 {result.get('estimated_days')}，运费 {result.get('freight_range')} 元。"
        return "未命中物流规则，需要人工确认专线、运费和时效。"

    if tool_name == "ticket_tool":
        return f"生成售后工单草稿 {result.get('ticket_id')}，售后结论需人工确认。"

    if tool_name == "knowledge_tool":
        if result.get("matched"):
            return (
                f"知识库检索成功，最佳距离 {result.get('top_distance')}，"
                f"返回 {len(result.get('sources', []))} 个来源。"
            )
        return "知识库证据不足，需要人工结合客户信息进一步确认。"

    return "完成工具调用。"


def tool_step(
    tool_name: str,
    arguments: dict[str, Any],
    result: dict[str, Any],
    *,
    attempt: int = 1,
    idempotency_key: str = "",
) -> dict[str, Any]:
    return {
        "step": "call_tool",
        "title": TOOL_TITLES.get(tool_name, tool_name),
        "status": "completed",
        "tool_name": tool_name,
        "summary": _tool_summary(tool_name, result),
        "data": {
            "arguments": arguments,
            "result": result,
            "attempt": attempt,
            "idempotency_key": idempotency_key,
        },
    }


def tool_route_step(tool_name: str, arguments: dict[str, Any], approval_required: bool) -> dict[str, Any]:
    return {
        "step": "route_tool",
        "title": "选择下一工具",
        "status": "waiting_approval" if approval_required else "ready",
        "summary": (
            f"准备执行 {TOOL_TITLES.get(tool_name, tool_name)}，执行前需要人工审批。"
            if approval_required
            else f"准备执行 {TOOL_TITLES.get(tool_name, tool_name)}。"
        ),
        "data": {
            "target_tool": tool_name,
            "arguments": arguments,
            "approval_required": approval_required,
        },
    }


def approval_decision_step(
    tool_name: str,
    decision: str,
    arguments: dict[str, Any],
    comment: str = "",
) -> dict[str, Any]:
    labels = {"approve": "批准", "edit": "修改后批准", "reject": "拒绝"}
    return {
        "step": "human_approval",
        "title": "人工审批",
        "status": decision,
        "summary": f"人工对 {TOOL_TITLES.get(tool_name, tool_name)} 作出“{labels.get(decision, decision)}”决定。",
        "data": {
            "target_tool": tool_name,
            "decision": decision,
            "arguments": arguments,
            "comment": comment,
        },
    }


def tool_skipped_step(tool_name: str, reason: str) -> dict[str, Any]:
    return {
        "step": "skip_tool",
        "title": "跳过工具调用",
        "status": "rejected",
        "summary": f"{TOOL_TITLES.get(tool_name, tool_name)}未执行：{reason}",
        "data": {
            "target_tool": tool_name,
            "reason": reason,
        },
    }


def tool_error_step(
    tool_name: str,
    error_type: str,
    message: str,
    attempts: int,
) -> dict[str, Any]:
    return {
        "step": "tool_error",
        "title": "工具失败兜底",
        "status": "failed",
        "summary": f"{TOOL_TITLES.get(tool_name, tool_name)}执行失败，已转为可解释兜底。",
        "data": {
            "target_tool": tool_name,
            "error_type": error_type,
            "message": message,
            "attempts": attempts,
        },
    }


def idempotent_reuse_step(tool_name: str, idempotency_key: str) -> dict[str, Any]:
    return {
        "step": "idempotent_reuse",
        "title": "复用已有工具结果",
        "status": "completed",
        "summary": f"{TOOL_TITLES.get(tool_name, tool_name)}命中幂等记录，没有重复执行。",
        "data": {
            "target_tool": tool_name,
            "idempotency_key": idempotency_key,
        },
    }


def handoff_evaluation_step(decision: dict[str, Any]) -> dict[str, Any]:
    required = bool(decision.get("required"))
    return {
        "step": "evaluate_handoff",
        "title": "评估人工接管",
        "status": "handoff_required" if required else "continue",
        "summary": (
            decision.get("reason_text") or "当前结果可以由 Agent 继续回复。"
        ),
        "data": decision,
    }


def handoff_created_step(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "step": "create_handoff",
        "title": "创建人工服务单",
        "status": "waiting_human",
        "summary": (
            f"已创建人工服务单 {case.get('handoff_id')}，"
            f"优先级为{case.get('priority', '普通')}。"
        ),
        "data": {
            "handoff_id": case.get("handoff_id"),
            "reason_code": case.get("reason_code"),
            "reason_text": case.get("reason_text"),
            "priority": case.get("priority"),
            "status": case.get("status"),
        },
    }


def human_response_step(
    handoff_id: str,
    agent_name: str,
    message: str,
) -> dict[str, Any]:
    return {
        "step": "human_response",
        "title": "人工客服回复",
        "status": "completed",
        "summary": f"{agent_name}已处理人工服务单 {handoff_id} 并恢复流程。",
        "data": {
            "handoff_id": handoff_id,
            "agent_name": agent_name,
            "reply_preview": message[:180].replace("\n", " "),
        },
    }


def unsupported_tools_step(unsupported_tools: list[str]) -> dict[str, Any]:
    return {
        "step": "mark_unsupported_tools",
        "title": "记录未接入工具",
        "status": "manual_follow_up",
        "summary": "识别到暂未接入的工具，需要转人工或后续扩展工具层。",
        "data": {
            "unsupported_tools": unsupported_tools,
        },
    }


def response_step(status: str, customer_reply: str) -> dict[str, Any]:
    return {
        "step": "build_response",
        "title": "生成客户侧回复",
        "status": status,
        "summary": "把追问或工具结果整理成客户可读的客服回复。",
        "data": {
            "reply_preview": customer_reply[:180].replace("\n", " "),
        },
    }


def append_step(trace: list[dict[str, Any]], step: dict[str, Any]) -> list[dict[str, Any]]:
    return [*trace, step]
