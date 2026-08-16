from __future__ import annotations

from typing import Any

from langchain_tools import get_langchain_tool_map
from schemas import (
    InventoryToolArgs,
    KnowledgeToolArgs,
    LogisticsToolArgs,
    QuoteToolArgs,
    TicketToolArgs,
    dump_args,
)


TOOL_REGISTRY = get_langchain_tool_map()
SUPPORTED_TOOLS = set(TOOL_REGISTRY)
TOOL_ARG_MODELS = {
    tool_name: tool.args_schema
    for tool_name, tool in TOOL_REGISTRY.items()
}


def build_tool_args(tool_name: str, parse_result: dict[str, Any]) -> dict[str, Any]:
    slots = parse_result["slots"]

    if tool_name == "inventory_tool":
        return dump_args(
            InventoryToolArgs(
                brand=slots.get("brand"),
                machine_model=slots["machine_model"],
                part_name=slots["part_name"],
                quality_level=slots.get("quality_level"),
            )
        )

    if tool_name == "quote_tool":
        return dump_args(
            QuoteToolArgs(
                brand=slots.get("brand"),
                machine_model=slots["machine_model"],
                part_name=slots["part_name"],
                quality_level=slots["quality_level"],
                quantity=slots.get("quantity") or 1,
            )
        )

    if tool_name == "logistics_tool":
        return dump_args(
            LogisticsToolArgs(
                city=slots["city"],
                part_name=slots["part_name"],
                urgent=slots.get("urgent"),
            )
        )

    if tool_name == "ticket_tool":
        return dump_args(
            TicketToolArgs(
                order_id=slots["order_id"],
                raw_question=parse_result["raw_question"],
            )
        )

    if tool_name == "knowledge_tool":
        return dump_args(
            KnowledgeToolArgs(
                question=parse_result["raw_question"],
                top_k=5,
            )
        )

    raise ValueError(f"Unsupported tool: {tool_name}")


def call_tool(tool_name: str, parse_result: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    args = build_tool_args(tool_name, parse_result)
    return args, execute_tool_with_args(tool_name, args)


def validate_tool_args(tool_name: str, raw_args: dict[str, Any]) -> dict[str, Any]:
    model_class = TOOL_ARG_MODELS.get(tool_name)
    if model_class is None:
        raise ValueError(f"Unsupported tool: {tool_name}")
    return dump_args(model_class.model_validate(raw_args))


def execute_tool_with_args(tool_name: str, raw_args: dict[str, Any]) -> dict[str, Any]:
    tool = TOOL_REGISTRY.get(tool_name)
    if tool is None:
        raise ValueError(f"Unsupported tool: {tool_name}")
    args = validate_tool_args(tool_name, raw_args)
    result = tool.invoke(
        args,
        config={
            "run_name": f"business_tool:{tool_name}",
            "tags": ["project2", "business_tool"],
            "metadata": {"tool_name": tool_name},
        },
    )
    if not isinstance(result, dict):
        raise TypeError(f"Tool {tool_name} must return a dictionary")
    return result
