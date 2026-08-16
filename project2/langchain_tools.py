from __future__ import annotations

from functools import lru_cache

from langchain_core.tools import StructuredTool

from schemas import (
    InventoryToolArgs,
    KnowledgeToolArgs,
    LogisticsToolArgs,
    QuoteToolArgs,
    TicketToolArgs,
)
from tools.inventory_tool import query_inventory
from tools.knowledge_tool import query_knowledge
from tools.logistics_tool import estimate_logistics
from tools.quote_tool import generate_quote
from tools.ticket_tool import create_after_sales_ticket


@lru_cache(maxsize=1)
def get_langchain_tool_map() -> dict[str, StructuredTool]:
    """Create the canonical production tool registry."""
    tools = [
        StructuredTool.from_function(
            func=query_inventory,
            name="inventory_tool",
            description="查询指定品牌、机型、配件和品质档位的模拟库存。",
            args_schema=InventoryToolArgs,
        ),
        StructuredTool.from_function(
            func=generate_quote,
            name="quote_tool",
            description="根据机型、配件、品质档位和数量生成报价草稿。",
            args_schema=QuoteToolArgs,
        ),
        StructuredTool.from_function(
            func=estimate_logistics,
            name="logistics_tool",
            description="估算配件发往指定城市的时效和基础运费。",
            args_schema=LogisticsToolArgs,
        ),
        StructuredTool.from_function(
            func=create_after_sales_ticket,
            name="ticket_tool",
            description="根据订单号和客户问题生成售后工单草稿。",
            args_schema=TicketToolArgs,
        ),
        StructuredTool.from_function(
            func=query_knowledge,
            name="knowledge_tool",
            description="查询企业知识库，回答适配、故障、服务流程和综合咨询。",
            args_schema=KnowledgeToolArgs,
        ),
    ]
    return {tool.name: tool for tool in tools}


def get_langchain_tools() -> list[StructuredTool]:
    """Expose deterministic business functions through LangChain tool schemas."""
    return list(get_langchain_tool_map().values())
