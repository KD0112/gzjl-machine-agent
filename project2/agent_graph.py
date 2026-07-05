from __future__ import annotations

import argparse
import json
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from agent_parser import parse_customer_question
from execution_trace import (
    append_step,
    missing_fields_step,
    parse_step,
    response_step,
    tool_step,
    unsupported_tools_step,
)
from response_builder import build_customer_reply
from tool_dispatcher import SUPPORTED_TOOLS, call_tool


class AgentState(TypedDict, total=False):
    question: str
    parse_result: dict[str, Any]
    tool_results: dict[str, Any]
    tool_arguments: dict[str, Any]
    called_tools: list[str]
    unsupported_tools: list[str]
    customer_reply: str
    status: str
    execution_mode: str
    execution_trace: list[dict[str, Any]]


def parse_node(state: AgentState) -> AgentState:
    parse_result = parse_customer_question(state["question"])
    return {
        "parse_result": parse_result,
        "tool_results": {},
        "tool_arguments": {},
        "called_tools": [],
        "unsupported_tools": [],
        "execution_mode": "langgraph",
        "execution_trace": [parse_step(parse_result)],
    }


def route_after_parse(state: AgentState) -> str:
    if state["parse_result"]["missing_fields"]:
        return "build_response"
    return "call_inventory"


def call_inventory_node(state: AgentState) -> AgentState:
    parse_result = state["parse_result"]
    if "inventory_tool" not in parse_result["candidate_tools"]:
        return {}

    args, result = call_tool("inventory_tool", parse_result)

    tool_results = dict(state.get("tool_results", {}))
    tool_results["inventory_tool"] = result
    tool_arguments = dict(state.get("tool_arguments", {}))
    tool_arguments["inventory_tool"] = args
    called_tools = [*state.get("called_tools", []), "inventory_tool"]
    execution_trace = append_step(
        state.get("execution_trace", []),
        tool_step("inventory_tool", args, result),
    )
    return {
        "tool_results": tool_results,
        "tool_arguments": tool_arguments,
        "called_tools": called_tools,
        "execution_trace": execution_trace,
    }


def call_quote_node(state: AgentState) -> AgentState:
    parse_result = state["parse_result"]
    if "quote_tool" not in parse_result["candidate_tools"]:
        return {}

    args, result = call_tool("quote_tool", parse_result)

    tool_results = dict(state.get("tool_results", {}))
    tool_results["quote_tool"] = result
    tool_arguments = dict(state.get("tool_arguments", {}))
    tool_arguments["quote_tool"] = args
    called_tools = [*state.get("called_tools", []), "quote_tool"]
    execution_trace = append_step(
        state.get("execution_trace", []),
        tool_step("quote_tool", args, result),
    )
    return {
        "tool_results": tool_results,
        "tool_arguments": tool_arguments,
        "called_tools": called_tools,
        "execution_trace": execution_trace,
    }


def call_logistics_node(state: AgentState) -> AgentState:
    parse_result = state["parse_result"]
    if "logistics_tool" not in parse_result["candidate_tools"]:
        return {}

    args, result = call_tool("logistics_tool", parse_result)

    tool_results = dict(state.get("tool_results", {}))
    tool_results["logistics_tool"] = result
    tool_arguments = dict(state.get("tool_arguments", {}))
    tool_arguments["logistics_tool"] = args
    called_tools = [*state.get("called_tools", []), "logistics_tool"]
    execution_trace = append_step(
        state.get("execution_trace", []),
        tool_step("logistics_tool", args, result),
    )
    return {
        "tool_results": tool_results,
        "tool_arguments": tool_arguments,
        "called_tools": called_tools,
        "execution_trace": execution_trace,
    }


def call_ticket_node(state: AgentState) -> AgentState:
    parse_result = state["parse_result"]
    if "ticket_tool" not in parse_result["candidate_tools"]:
        return {}

    args, result = call_tool("ticket_tool", parse_result)

    tool_results = dict(state.get("tool_results", {}))
    tool_results["ticket_tool"] = result
    tool_arguments = dict(state.get("tool_arguments", {}))
    tool_arguments["ticket_tool"] = args
    called_tools = [*state.get("called_tools", []), "ticket_tool"]
    execution_trace = append_step(
        state.get("execution_trace", []),
        tool_step("ticket_tool", args, result),
    )
    return {
        "tool_results": tool_results,
        "tool_arguments": tool_arguments,
        "called_tools": called_tools,
        "execution_trace": execution_trace,
    }


def mark_unsupported_tools_node(state: AgentState) -> AgentState:
    parse_result = state["parse_result"]
    unsupported_tools = [
        tool_name
        for tool_name in parse_result["candidate_tools"]
        if tool_name not in SUPPORTED_TOOLS and tool_name not in state.get("called_tools", [])
    ]
    if not unsupported_tools:
        return {"unsupported_tools": unsupported_tools}
    return {
        "unsupported_tools": unsupported_tools,
        "execution_trace": append_step(
            state.get("execution_trace", []),
            unsupported_tools_step(unsupported_tools),
        ),
    }


def build_response_node(state: AgentState) -> AgentState:
    parse_result = state["parse_result"]
    tool_results = state.get("tool_results", {})
    unsupported_tools = state.get("unsupported_tools", [])
    status = "need_more_info" if parse_result["missing_fields"] else "completed"
    customer_reply = build_customer_reply(parse_result, tool_results, unsupported_tools)
    execution_trace = list(state.get("execution_trace", []))
    if parse_result["missing_fields"]:
        execution_trace = append_step(execution_trace, missing_fields_step(parse_result))
    execution_trace = append_step(execution_trace, response_step(status, customer_reply))
    return {
        "status": status,
        "customer_reply": customer_reply,
        "execution_trace": execution_trace,
    }


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("parse", parse_node)
    graph.add_node("call_inventory", call_inventory_node)
    graph.add_node("call_quote", call_quote_node)
    graph.add_node("call_logistics", call_logistics_node)
    graph.add_node("call_ticket", call_ticket_node)
    graph.add_node("mark_unsupported_tools", mark_unsupported_tools_node)
    graph.add_node("build_response", build_response_node)

    graph.add_edge(START, "parse")
    graph.add_conditional_edges(
        "parse",
        route_after_parse,
        {
            "call_inventory": "call_inventory",
            "build_response": "build_response",
        },
    )
    graph.add_edge("call_inventory", "call_quote")
    graph.add_edge("call_quote", "call_logistics")
    graph.add_edge("call_logistics", "call_ticket")
    graph.add_edge("call_ticket", "mark_unsupported_tools")
    graph.add_edge("mark_unsupported_tools", "build_response")
    graph.add_edge("build_response", END)
    return graph.compile()


COMPILED_GRAPH = build_graph()


def run_graph_agent(question: str) -> dict[str, Any]:
    result = COMPILED_GRAPH.invoke({"question": question})
    return {
        "status": result["status"],
        "customer_reply": result["customer_reply"],
        "parse_result": result["parse_result"],
        "tool_results": result.get("tool_results", {}),
        "tool_arguments": result.get("tool_arguments", {}),
        "called_tools": result.get("called_tools", []),
        "unsupported_tools": result.get("unsupported_tools", []),
        "execution_mode": result.get("execution_mode", "langgraph"),
        "execution_trace": result.get("execution_trace", []),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run project2 LangGraph agent workflow.")
    parser.add_argument("question")
    args = parser.parse_args()

    result = run_graph_agent(args.question)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
