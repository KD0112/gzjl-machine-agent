from __future__ import annotations

from typing import Any


TOOL_TITLES = {
    "inventory_tool": "查询库存",
    "quote_tool": "生成报价草稿",
    "logistics_tool": "估算物流",
    "ticket_tool": "生成售后工单草稿",
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

    return "完成工具调用。"


def tool_step(tool_name: str, arguments: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    return {
        "step": "call_tool",
        "title": TOOL_TITLES.get(tool_name, tool_name),
        "status": "completed",
        "tool_name": tool_name,
        "summary": _tool_summary(tool_name, result),
        "data": {
            "arguments": arguments,
            "result": result,
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
