from __future__ import annotations

import argparse
import json
from typing import Any

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


def run_agent(question: str) -> dict[str, Any]:
    parse_result = parse_customer_question(question)
    execution_trace = [parse_step(parse_result)]

    if parse_result["follow_up"]:
        customer_reply = build_customer_reply(parse_result, {}, [])
        execution_trace = append_step(execution_trace, missing_fields_step(parse_result))
        execution_trace = append_step(execution_trace, response_step("need_more_info", customer_reply))
        return {
            "status": "need_more_info",
            "customer_reply": customer_reply,
            "parse_result": parse_result,
            "tool_results": {},
            "tool_arguments": {},
            "called_tools": [],
            "unsupported_tools": [],
            "execution_trace": execution_trace,
        }

    tool_results: dict[str, Any] = {}
    tool_arguments: dict[str, Any] = {}
    called_tools: list[str] = []
    unsupported_tools: list[str] = []

    for tool_name in parse_result["candidate_tools"]:
        if tool_name not in SUPPORTED_TOOLS:
            unsupported_tools.append(tool_name)
            continue

        args, result = call_tool(tool_name, parse_result)
        tool_arguments[tool_name] = args
        tool_results[tool_name] = result
        called_tools.append(tool_name)
        execution_trace = append_step(
            execution_trace,
            tool_step(tool_name, args, result),
        )

    if unsupported_tools:
        execution_trace = append_step(execution_trace, unsupported_tools_step(unsupported_tools))

    customer_reply = build_customer_reply(parse_result, tool_results, unsupported_tools)
    execution_trace = append_step(execution_trace, response_step("completed", customer_reply))

    return {
        "status": "completed",
        "customer_reply": customer_reply,
        "parse_result": parse_result,
        "tool_results": tool_results,
        "tool_arguments": tool_arguments,
        "called_tools": called_tools,
        "unsupported_tools": unsupported_tools,
        "execution_trace": execution_trace,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run project2 multi-tool agent workflow.")
    parser.add_argument("question")
    args = parser.parse_args()

    result = run_agent(args.question)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
