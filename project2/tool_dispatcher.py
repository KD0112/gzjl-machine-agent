from __future__ import annotations

from typing import Any

from schemas import (
    InventoryToolArgs,
    LogisticsToolArgs,
    QuoteToolArgs,
    TicketToolArgs,
    dump_args,
)
from tools.inventory_tool import query_inventory
from tools.logistics_tool import estimate_logistics
from tools.quote_tool import generate_quote
from tools.ticket_tool import create_after_sales_ticket


SUPPORTED_TOOLS = {"inventory_tool", "quote_tool", "logistics_tool", "ticket_tool"}


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

    raise ValueError(f"Unsupported tool: {tool_name}")


def call_tool(tool_name: str, parse_result: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    args = build_tool_args(tool_name, parse_result)

    if tool_name == "inventory_tool":
        return args, query_inventory(**args)
    if tool_name == "quote_tool":
        return args, generate_quote(**args)
    if tool_name == "logistics_tool":
        return args, estimate_logistics(**args)
    if tool_name == "ticket_tool":
        return args, create_after_sales_ticket(**args)

    raise ValueError(f"Unsupported tool: {tool_name}")
