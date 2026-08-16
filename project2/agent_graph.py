from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import uuid
from functools import partial
from pathlib import Path
from typing import Any, Callable, Literal, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.errors import NodeError
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command, RetryPolicy, interrupt

from agent_parser import INTENT_TOOLS, build_follow_up, find_missing_fields
from context_manager import (
    DEFAULT_CONTEXT_POLICY,
    ContextPolicy,
    build_context_snapshot,
    compact_messages,
    guard_knowledge_result,
    make_message,
    merge_conversation_context,
)
from langchain_adapter import parse_customer_question_with_langchain
from execution_trace import (
    append_step,
    approval_decision_step,
    context_step,
    handoff_created_step,
    handoff_evaluation_step,
    human_response_step,
    idempotent_reuse_step,
    image_confirmation_step,
    image_inspection_step,
    memory_step,
    missing_fields_step,
    model_runtime_step,
    parse_step,
    response_step,
    tool_error_step,
    tool_route_step,
    tool_skipped_step,
    tool_step,
    unsupported_tools_step,
)
from handoff_policy import evaluate_handoff
from handoff_repository import DEFAULT_HANDOFF_REPOSITORY, HandoffRepository
from image_evidence import (
    DEFAULT_IMAGE_EVIDENCE_REPOSITORY,
    ImageEvidenceRepository,
)
from memory_repository import DEFAULT_MEMORY_REPOSITORY, MemoryRepository
from semantic_memory import SemanticMemoryStore
from response_builder import build_customer_reply
from schemas import HumanReply, ImageConfirmation
from tool_dispatcher import (
    SUPPORTED_TOOLS,
    build_tool_args,
    execute_tool_with_args,
    validate_tool_args,
)
from vision_service import inspect_image


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT_DB_PATH = BASE_DIR / "logs" / "langgraph_checkpoints.sqlite3"
APPROVAL_REQUIRED_TOOLS = {"quote_tool", "ticket_tool"}
APPROVAL_DECISIONS = {"approve", "edit", "reject"}
TOOL_APPROVAL_REASONS = {
    "quote_tool": "报价会影响客户预期，演示模式要求人工确认参数后再生成报价草稿。",
    "ticket_tool": "售后请求涉及退货、退款或换货边界，演示模式要求人工确认后再生成工单草稿。",
}
VISUAL_SLOT_MAP = {
    "brand": "brand",
    "machine_model": "machine_model",
    "part_name_candidate": "part_name",
    "part_number": "part_number",
}
IMAGE_INSPECTION_MARKERS = (
    "识别",
    "提取",
    "读取",
    "图片上",
    "图上",
    "标签上",
    "铭牌上",
    "看图",
    "看看图片",
)
IMAGE_BUSINESS_ACTION_MARKERS = (
    "现货",
    "有货",
    "库存",
    "价格",
    "报价",
    "多少钱",
    "发货",
    "物流",
    "运费",
    "多久",
    "适配",
    "匹配",
    "能不能用",
    "通用",
    "故障",
    "漏油",
    "异响",
    "维修",
    "损坏",
    "退货",
    "换货",
    "售后",
)
NEGATION_MARKERS = ("不要", "不用", "无需", "不需要", "先别", "暂不", "别")
IMAGE_CONFIRMATION_DECISIONS = {"confirm", "edit", "reject", "human"}


class AgentState(TypedDict, total=False):
    question: str
    request_id: str
    thread_id: str
    session_id: str
    approval_mode: Literal["auto", "manual"]
    handoff_mode: Literal["off", "manual"]
    parser_mode: Literal["rules", "hybrid", "llm"]
    knowledge_mode: bool
    memory_mode: Literal["off", "profile"]
    clarification_count: int
    turn_count: int
    channel: str
    customer_id: str
    thread_customer_id: str
    messages: list[dict[str, Any]]
    conversation_summary: str
    conversation_slots: dict[str, Any]
    long_term_memories: list[dict[str, Any]]
    episodic_memories: list[dict[str, Any]]
    memory_job_id: str
    memory_writes: list[dict[str, Any]]
    context_snapshot: dict[str, Any]
    context_dropped_messages: int
    model_runtime: dict[str, Any]
    attachments: list[dict[str, Any]]
    vision_results: list[dict[str, Any]]
    vision_status: str
    vision_error: str
    vision_model_runtime: list[dict[str, Any]]
    image_confirmation_request: dict[str, Any] | None
    image_confirmation_decisions: list[dict[str, Any]]
    confirmed_visual_slots: dict[str, Any]
    parse_result: dict[str, Any]
    tool_queue: list[str]
    current_tool: str | None
    pending_tool_arguments: dict[str, Any]
    tool_results: dict[str, Any]
    tool_arguments: dict[str, Any]
    called_tools: list[str]
    skipped_tools: list[str]
    unsupported_tools: list[str]
    tool_errors: dict[str, dict[str, Any]]
    tool_execution_keys: dict[str, str]
    approval_request: dict[str, Any] | None
    approval_decisions: list[dict[str, Any]]
    handoff_required: bool
    handoff_reason: dict[str, Any]
    handoff_id: str
    handoff_status: str
    handoff_priority: str
    assigned_agent: str
    human_reply: str
    handoff_context: dict[str, Any]
    customer_reply: str
    status: str
    execution_mode: str
    execution_trace: list[dict[str, Any]]


def _checkpoint_path() -> Path:
    configured = os.getenv("LANGGRAPH_CHECKPOINT_DB", "").strip()
    if not configured:
        return DEFAULT_CHECKPOINT_DB_PATH
    path = Path(configured)
    return path if path.is_absolute() else BASE_DIR / path


def create_sqlite_checkpointer(path: Path | None = None) -> SqliteSaver:
    checkpoint_path = path or _checkpoint_path()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(checkpoint_path), check_same_thread=False, timeout=30)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA busy_timeout=30000")
    return SqliteSaver(connection)


def _thread_config(thread_id: str) -> dict[str, dict[str, str]]:
    return {"configurable": {"thread_id": thread_id}}


def _idempotency_key(request_id: str, tool_name: str, arguments: dict[str, Any]) -> str:
    payload = json.dumps(
        {
            "request_id": request_id,
            "tool_name": tool_name,
            "arguments": arguments,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    return f"{request_id}:{tool_name}:{digest}"


def load_context_node(
    state: AgentState,
    *,
    repository: MemoryRepository,
    context_policy: ContextPolicy,
) -> AgentState:
    request_id = state.get("request_id") or uuid.uuid4().hex
    thread_id = state.get("thread_id") or request_id
    session_id = state.get("session_id") or thread_id
    turn_count = int(state.get("turn_count", 0)) + 1
    customer_id = state.get("customer_id", "")
    bound_customer_id = state.get("thread_customer_id", "")
    if bound_customer_id and customer_id and bound_customer_id != customer_id:
        raise ValueError("同一 thread_id 不能切换 customer_id，请新建会话。")
    customer_id = customer_id or bound_customer_id
    semantic_store = SemanticMemoryStore(repository.path)
    if customer_id and state.get("question", ""):
        semantic_store.append_message(
            customer_id=customer_id,
            thread_id=thread_id,
            turn_id=turn_count,
            role="user",
            content=state.get("question", ""),
            request_id=request_id,
        )
    episodic_memories = (
        semantic_store.search(
            customer_id=customer_id,
            query=state.get("question", ""),
            limit=max(1, context_policy.max_memory_items // 2),
        )
        if customer_id and state.get("question", "")
        else []
    )
    memories = (
        repository.list_active(customer_id, limit=context_policy.max_memory_items)
        if state.get("memory_mode", "profile") == "profile"
        else []
    )
    messages = list(state.get("messages", []))
    if not messages or messages[-1].get("request_id") != request_id:
        messages.append(
            make_message(
                "user",
                state.get("question", ""),
                turn_index=turn_count,
                request_id=request_id,
            )
        )
    messages, summary, dropped = compact_messages(
        messages,
        state.get("conversation_summary", ""),
        context_policy,
    )
    dropped_total = int(state.get("context_dropped_messages", 0)) + dropped
    context_snapshot = build_context_snapshot(
        question=state.get("question", ""),
        messages=messages,
        conversation_summary=summary,
        conversation_slots=state.get("conversation_slots", {}),
        long_term_memories=memories,
        episodic_memories=episodic_memories,
        policy=context_policy,
        dropped_messages=dropped_total,
    )
    return {
        "request_id": request_id,
        "thread_id": thread_id,
        "session_id": session_id,
        "turn_count": turn_count,
        "customer_id": customer_id,
        "thread_customer_id": customer_id,
        "messages": messages,
        "conversation_summary": summary,
        "long_term_memories": memories,
        "episodic_memories": episodic_memories,
        "memory_writes": [],
        "context_snapshot": context_snapshot,
        "context_dropped_messages": dropped_total,
    }


def route_after_context(state: AgentState) -> str:
    return "inspect_image" if state.get("attachments") else "parse"


def _visual_candidates(vision_results: list[dict[str, Any]]) -> dict[str, Any]:
    candidates: dict[str, Any] = {}
    for item in vision_results:
        inspection = item.get("inspection") or {}
        for source_field, slot_field in VISUAL_SLOT_MAP.items():
            value = inspection.get(source_field)
            if value and slot_field not in candidates:
                candidates[slot_field] = value
    return candidates


def inspect_image_node(
    state: AgentState,
    *,
    repository: ImageEvidenceRepository,
    inspector: Callable[..., dict[str, Any]] = inspect_image,
) -> AgentState:
    attachments = list(state.get("attachments", []))[:3]
    vision_results: list[dict[str, Any]] = []
    runtimes: list[dict[str, Any]] = []
    errors: list[str] = []

    for attachment in attachments:
        evidence_id = str(attachment.get("evidence_id", ""))
        try:
            content, metadata = repository.read_content(
                evidence_id,
                customer_id=state.get("customer_id", ""),
            )
            result = inspector(
                content=content,
                mime_type=metadata["mime_type"],
                evidence_id=evidence_id,
                local_quality=metadata.get("local_quality", "fair"),
                quality_signals=metadata.get("quality_signals", []),
                request_id=state.get("request_id", ""),
                thread_id=state.get("thread_id", ""),
            )
            vision_results.append(result)
            if result.get("model_runtime"):
                runtimes.append(result["model_runtime"])
        except Exception as exc:
            errors.append(f"{evidence_id}: {exc}")
            vision_results.append(
                {
                    "evidence_id": evidence_id,
                    "inspection": {},
                    "error": str(exc),
                }
            )

    valid_inspections = [
        item.get("inspection", {})
        for item in vision_results
        if item.get("inspection")
    ]
    candidates = _visual_candidates(vision_results)
    acceptable = [
        item
        for item in valid_inspections
        if item.get("image_quality") not in {"poor", "unusable"}
        and float(item.get("confidence", 0)) >= 0.55
        and item.get("image_type") not in {"irrelevant", "unknown"}
    ]
    if errors and not valid_inspections:
        vision_status = "failed"
    elif not acceptable or not candidates:
        vision_status = "needs_better_image"
    else:
        vision_status = "waiting_confirmation"

    confirmation_request = None
    if vision_status == "waiting_confirmation":
        confirmation_request = {
            "kind": "image_evidence_confirmation",
            "thread_id": state.get("thread_id", ""),
            "request_id": state.get("request_id", ""),
            "evidence_ids": [item.get("evidence_id") for item in vision_results],
            "candidate_fields": candidates,
            "inspections": valid_inspections,
            "allowed_decisions": ["confirm", "edit", "reject", "human"],
            "reason": "图片识别结果只是候选证据，关键字段需要客户确认后才能进入业务槽位。",
        }

    trace = append_step(
        state.get("execution_trace", []),
        image_inspection_step(
            vision_results,
            vision_status=vision_status,
            error="；".join(errors),
        ),
    )
    return {
        "vision_results": vision_results,
        "vision_status": vision_status,
        "vision_error": "；".join(errors),
        "vision_model_runtime": runtimes,
        "image_confirmation_request": confirmation_request,
        "image_confirmation_decisions": [],
        "confirmed_visual_slots": {},
        "execution_trace": trace,
    }


def route_after_image_inspection(state: AgentState) -> str:
    if state.get("image_confirmation_request"):
        return "confirm_image"
    return "parse"


def _normalize_visual_slots(raw_fields: dict[str, Any]) -> dict[str, Any]:
    allowed = {"brand", "machine_model", "part_name", "part_number"}
    normalized: dict[str, Any] = {}
    for field, value in raw_fields.items():
        if field not in allowed or value is None:
            continue
        text = str(value).strip()
        if text:
            normalized[field] = text[:160]
    return normalized


def confirm_image_node(state: AgentState) -> AgentState:
    request = state.get("image_confirmation_request")
    if not request:
        return {}

    while True:
        response = interrupt(request)
        if isinstance(response, str):
            raw_confirmation = {"decision": response}
        else:
            raw_confirmation = response
        try:
            confirmation = ImageConfirmation.model_validate(raw_confirmation)
            break
        except Exception as exc:
            request = {
                **request,
                "validation_error": f"图片确认格式无效：{exc}",
            }

    decision = confirmation.decision
    candidate_fields = _normalize_visual_slots(
        dict(request.get("candidate_fields", {}))
    )
    if decision == "confirm":
        confirmed_fields = candidate_fields
        vision_status = "confirmed"
    elif decision == "edit":
        confirmed_fields = _normalize_visual_slots(confirmation.confirmed_fields)
        if not confirmed_fields:
            decision = "reject"
            vision_status = "rejected"
        else:
            vision_status = "confirmed"
    elif decision == "human":
        confirmed_fields = {}
        vision_status = "needs_human"
    else:
        confirmed_fields = {}
        vision_status = "rejected"

    decision_record = {
        "decision": decision,
        "confirmed_fields": confirmed_fields,
        "comment": confirmation.comment,
    }
    return {
        "confirmed_visual_slots": confirmed_fields,
        "vision_status": vision_status,
        "image_confirmation_request": None,
        "image_confirmation_decisions": [
            *state.get("image_confirmation_decisions", []),
            decision_record,
        ],
        "execution_trace": append_step(
            state.get("execution_trace", []),
            image_confirmation_step(
                decision=decision,
                confirmed_fields=confirmed_fields,
                comment=confirmation.comment,
            ),
        ),
    }


def _is_image_inspection_only(state: AgentState) -> bool:
    if not state.get("confirmed_visual_slots"):
        return False
    question = str(state.get("question", "")).replace(" ", "")
    if not any(marker in question for marker in IMAGE_INSPECTION_MARKERS):
        return False

    for marker in IMAGE_BUSINESS_ACTION_MARKERS:
        start = 0
        while True:
            index = question.find(marker, start)
            if index < 0:
                break
            prefix = question[max(0, index - 10) : index]
            if not any(negation in prefix for negation in NEGATION_MARKERS):
                return False
            start = index + len(marker)
    return True


def parse_node(
    state: AgentState,
    *,
    context_policy: ContextPolicy = DEFAULT_CONTEXT_POLICY,
) -> AgentState:
    previous_parse_result = dict(state.get("parse_result", {}))
    parse_result = parse_customer_question_with_langchain(
        state["question"],
        mode=state.get("parser_mode", "rules"),
        context=(state.get("context_snapshot") or {}).get("rendered_context", ""),
        request_id=state.get("request_id", ""),
        thread_id=state.get("thread_id", ""),
    )
    merged = merge_conversation_context(
        question=state["question"],
        current_intents=list(parse_result.get("intents") or []),
        current_slots=dict(parse_result.get("slots") or {}),
        previous_parse_result=previous_parse_result,
        conversation_slots=state.get("conversation_slots", {}),
        long_term_memories=state.get("long_term_memories", []),
    )
    visual_conflicts: list[dict[str, Any]] = []
    for field, value in state.get("confirmed_visual_slots", {}).items():
        existing = merged["slots"].get(field)
        if existing is None:
            merged["slots"][field] = value
            merged["slot_sources"][field] = "confirmed_image"
            merged["conversation_slots"][field] = value
        elif str(existing).strip().casefold() != str(value).strip().casefold():
            visual_conflicts.append(
                {
                    "field": field,
                    "kept_value": existing,
                    "image_value": value,
                    "resolution": "current_question_wins",
                }
            )
    parse_result["intents"] = merged["intents"]
    parse_result["slots"] = merged["slots"]
    parse_result["slot_sources"] = merged["slot_sources"]
    parse_result["context_conflicts"] = [
        *merged["conflicts"],
        *visual_conflicts,
    ]
    parse_result["visual_evidence"] = {
        "status": state.get("vision_status", ""),
        "evidence_ids": [
            item.get("evidence_id") for item in state.get("vision_results", [])
        ],
        "confirmed_slots": state.get("confirmed_visual_slots", {}),
    }
    if _is_image_inspection_only(state):
        parse_result["intents"] = ["image_inspection"]
        parse_result["confidence"] = max(
            0.95,
            float(parse_result.get("confidence") or 0),
        )
        parse_result.setdefault("debug", {})["image_intent_override"] = (
            "confirmed_visual_evidence_only"
        )
    parse_result["missing_fields"] = find_missing_fields(
        parse_result["intents"],
        parse_result["slots"],
    )
    parse_result["candidate_tools"] = [
        INTENT_TOOLS[intent]
        for intent in parse_result["intents"]
        if intent in INTENT_TOOLS
    ]
    parse_result["follow_up"] = build_follow_up(parse_result["missing_fields"])
    parse_result["ready_for_tools"] = (
        bool(parse_result["candidate_tools"]) and not parse_result["missing_fields"]
    )
    candidate_tools = list(dict.fromkeys(parse_result["candidate_tools"]))
    if state.get("knowledge_mode") and any(
        intent in {"compatibility", "diagnosis", "general_consulting"}
        for intent in parse_result.get("intents", [])
    ):
        candidate_tools.append("knowledge_tool")
        candidate_tools = list(dict.fromkeys(candidate_tools))
        parse_result["candidate_tools"] = candidate_tools
        parse_result["ready_for_tools"] = not parse_result.get("missing_fields")
    tool_queue = [name for name in candidate_tools if name in SUPPORTED_TOOLS]
    unsupported_tools = [name for name in candidate_tools if name not in SUPPORTED_TOOLS]
    request_id = state.get("request_id") or uuid.uuid4().hex
    thread_id = state.get("thread_id") or request_id
    previous_missing = previous_parse_result.get("missing_fields") or []
    if parse_result["missing_fields"] and previous_missing:
        clarification_count = int(state.get("clarification_count", 0)) + 1
    elif parse_result["missing_fields"]:
        clarification_count = int(state.get("clarification_count", 0))
    else:
        clarification_count = 0
    context_snapshot = build_context_snapshot(
        question=state["question"],
        messages=state.get("messages", []),
        conversation_summary=state.get("conversation_summary", ""),
        conversation_slots=merged["conversation_slots"],
        long_term_memories=state.get("long_term_memories", []),
        episodic_memories=state.get("episodic_memories", []),
        policy=context_policy,
        dropped_messages=state.get("context_dropped_messages", 0),
    )
    model_runtime = parse_result.get("debug", {}).get("model_runtime", {})
    previous_trace = list(state.get("execution_trace", []))
    keep_visual_trace = bool(
        previous_trace
        and previous_trace[-1].get("step")
        in {"inspect_image", "confirm_image_evidence"}
    )
    execution_trace = previous_trace if keep_visual_trace else []
    execution_trace.append(parse_step(parse_result))
    if model_runtime:
        execution_trace.append(model_runtime_step(model_runtime))
    execution_trace.append(
        context_step(
            context_snapshot,
            turn_count=int(state.get("turn_count", 1)),
            conflicts=parse_result["context_conflicts"],
        )
    )

    return {
        "request_id": request_id,
        "thread_id": thread_id,
        "session_id": state.get("session_id", thread_id),
        "approval_mode": state.get("approval_mode", "auto"),
        "handoff_mode": state.get("handoff_mode", "off"),
        "parser_mode": state.get("parser_mode", "rules"),
        "knowledge_mode": bool(state.get("knowledge_mode", False)),
        "memory_mode": state.get("memory_mode", "profile"),
        "clarification_count": clarification_count,
        "turn_count": int(state.get("turn_count", 1)),
        "channel": state.get("channel", "web"),
        "customer_id": state.get("customer_id", ""),
        "conversation_slots": merged["conversation_slots"],
        "context_snapshot": context_snapshot,
        "model_runtime": model_runtime,
        "attachments": state.get("attachments", []),
        "vision_results": state.get("vision_results", []),
        "vision_status": state.get("vision_status", ""),
        "vision_error": state.get("vision_error", ""),
        "vision_model_runtime": state.get("vision_model_runtime", []),
        "image_confirmation_request": state.get("image_confirmation_request"),
        "image_confirmation_decisions": state.get(
            "image_confirmation_decisions",
            [],
        ),
        "confirmed_visual_slots": state.get("confirmed_visual_slots", {}),
        "parse_result": parse_result,
        "tool_queue": tool_queue,
        "current_tool": None,
        "pending_tool_arguments": {},
        "tool_results": {},
        "tool_arguments": {},
        "called_tools": [],
        "skipped_tools": [],
        "unsupported_tools": unsupported_tools,
        "tool_errors": {},
        "tool_execution_keys": {},
        "approval_request": None,
        "approval_decisions": [],
        "handoff_required": False,
        "handoff_reason": {},
        "handoff_id": "",
        "handoff_status": "",
        "handoff_priority": "",
        "assigned_agent": "",
        "human_reply": "",
        "handoff_context": {},
        "customer_reply": "",
        "status": "running",
        "execution_mode": "langgraph",
        "execution_trace": execution_trace,
    }


def route_after_parse(state: AgentState) -> str:
    if state.get("vision_status") in {
        "failed",
        "needs_better_image",
        "needs_human",
        "rejected",
    }:
        return "evaluate_handoff"
    if state["parse_result"]["missing_fields"]:
        return "evaluate_handoff"
    if state.get("tool_queue"):
        return "prepare_tool"
    if state.get("unsupported_tools"):
        return "mark_unsupported_tools"
    return "evaluate_handoff"


def prepare_tool_node(state: AgentState) -> AgentState:
    queue = list(state.get("tool_queue", []))
    if not queue:
        return {"current_tool": None, "pending_tool_arguments": {}, "approval_request": None}

    tool_name = queue.pop(0)
    arguments = build_tool_args(tool_name, state["parse_result"])
    approval_required = (
        state.get("approval_mode") == "manual" and tool_name in APPROVAL_REQUIRED_TOOLS
    )
    approval_request = None
    if approval_required:
        approval_request = {
            "kind": "tool_approval",
            "tool_name": tool_name,
            "arguments": arguments,
            "reason": TOOL_APPROVAL_REASONS[tool_name],
            "allowed_decisions": ["approve", "edit", "reject"],
            "thread_id": state["thread_id"],
            "request_id": state["request_id"],
        }

    execution_trace = append_step(
        state.get("execution_trace", []),
        tool_route_step(tool_name, arguments, approval_required),
    )
    return {
        "tool_queue": queue,
        "current_tool": tool_name,
        "pending_tool_arguments": arguments,
        "approval_request": approval_request,
        "status": "waiting_approval" if approval_required else "running",
        "execution_trace": execution_trace,
    }


def route_after_prepare(state: AgentState) -> str:
    if state.get("approval_request"):
        return "approval"
    return "execute_tool"


def _normalize_approval_response(
    response: Any,
    current_arguments: dict[str, Any],
) -> tuple[str, dict[str, Any], str]:
    if isinstance(response, str):
        decision = response
        edited_arguments = current_arguments
        comment = ""
    elif isinstance(response, dict):
        decision = str(response.get("decision", "")).strip().lower()
        edited_arguments = response.get("edited_arguments") or current_arguments
        comment = str(response.get("comment", "")).strip()
    else:
        decision = ""
        edited_arguments = current_arguments
        comment = ""

    if decision not in APPROVAL_DECISIONS:
        return "reject", current_arguments, "审批结果无效，已按拒绝处理。"
    return decision, edited_arguments, comment


def approval_node(state: AgentState) -> AgentState:
    request = state["approval_request"]
    if not request:
        return {"status": "running"}

    response = interrupt(request)
    tool_name = str(request["tool_name"])
    current_arguments = dict(state.get("pending_tool_arguments", {}))
    decision, edited_arguments, comment = _normalize_approval_response(response, current_arguments)

    if decision == "edit":
        try:
            edited_arguments = validate_tool_args(tool_name, edited_arguments)
        except Exception as exc:
            decision = "reject"
            edited_arguments = current_arguments
            comment = f"修改后的参数未通过校验：{exc}"

    approval_record = {
        "tool_name": tool_name,
        "decision": decision,
        "arguments": edited_arguments,
        "comment": comment,
    }
    approval_decisions = [*state.get("approval_decisions", []), approval_record]
    execution_trace = append_step(
        state.get("execution_trace", []),
        approval_decision_step(tool_name, decision, edited_arguments, comment),
    )
    return {
        "pending_tool_arguments": edited_arguments,
        "approval_request": None,
        "approval_decisions": approval_decisions,
        "status": "running",
        "execution_trace": execution_trace,
    }


def route_after_approval(state: AgentState) -> str:
    decisions = state.get("approval_decisions", [])
    if decisions and decisions[-1]["decision"] in {"approve", "edit"}:
        return "execute_tool"
    return "skip_tool"


def _retryable_tool_error(exc: Exception) -> bool:
    return isinstance(exc, (ConnectionError, TimeoutError, OSError))


def execute_tool_node(
    state: AgentState,
    runtime: Runtime | None = None,
) -> AgentState:
    tool_name = state["current_tool"]
    if not tool_name:
        return {}

    arguments = validate_tool_args(tool_name, state.get("pending_tool_arguments", {}))
    idempotency_key = _idempotency_key(state["request_id"], tool_name, arguments)
    existing_keys = dict(state.get("tool_execution_keys", {}))
    existing_results = dict(state.get("tool_results", {}))
    if existing_keys.get(tool_name) == idempotency_key and tool_name in existing_results:
        return {
            "execution_trace": append_step(
                state.get("execution_trace", []),
                idempotent_reuse_step(tool_name, idempotency_key),
            )
        }

    result = execute_tool_with_args(tool_name, arguments)
    if tool_name == "knowledge_tool" and isinstance(result, dict):
        result = guard_knowledge_result(result)
    tool_results = dict(existing_results)
    tool_results[tool_name] = result
    tool_arguments = dict(state.get("tool_arguments", {}))
    tool_arguments[tool_name] = arguments
    called_tools = list(state.get("called_tools", []))
    if tool_name not in called_tools:
        called_tools.append(tool_name)
    existing_keys[tool_name] = idempotency_key

    attempt = 1
    if runtime is not None:
        attempt = int(getattr(runtime.execution_info, "node_attempt", 1))
    execution_trace = append_step(
        state.get("execution_trace", []),
        tool_step(
            tool_name,
            arguments,
            result,
            attempt=attempt,
            idempotency_key=idempotency_key,
        ),
    )
    return {
        "tool_results": tool_results,
        "tool_arguments": tool_arguments,
        "called_tools": called_tools,
        "tool_execution_keys": existing_keys,
        "status": "running",
        "execution_trace": execution_trace,
    }


def tool_error_handler(state: AgentState, error: NodeError) -> Command:
    tool_name = state.get("current_tool") or "unknown_tool"
    exc = error.error
    attempts = 3 if _retryable_tool_error(exc) else 1
    tool_errors = dict(state.get("tool_errors", {}))
    tool_errors[tool_name] = {
        "error_type": type(exc).__name__,
        "message": str(exc),
        "attempts": attempts,
        "retryable": _retryable_tool_error(exc),
    }
    execution_trace = append_step(
        state.get("execution_trace", []),
        tool_error_step(tool_name, type(exc).__name__, str(exc), attempts),
    )
    return Command(
        update={
            "tool_errors": tool_errors,
            "status": "tool_error",
            "execution_trace": execution_trace,
        },
        goto="advance_after_tool",
    )


def skip_tool_node(state: AgentState) -> AgentState:
    tool_name = state.get("current_tool") or "unknown_tool"
    decisions = state.get("approval_decisions", [])
    reason = decisions[-1].get("comment") if decisions else ""
    reason = reason or "人工拒绝了本次工具调用。"
    skipped_tools = list(state.get("skipped_tools", []))
    if tool_name not in skipped_tools:
        skipped_tools.append(tool_name)
    return {
        "skipped_tools": skipped_tools,
        "execution_trace": append_step(
            state.get("execution_trace", []),
            tool_skipped_step(tool_name, reason),
        ),
    }


def advance_after_tool_node(state: AgentState) -> AgentState:
    return {
        "current_tool": None,
        "pending_tool_arguments": {},
        "approval_request": None,
    }


def route_after_tool(state: AgentState) -> str:
    if state.get("tool_queue"):
        return "prepare_tool"
    if state.get("unsupported_tools"):
        return "mark_unsupported_tools"
    return "evaluate_handoff"


def mark_unsupported_tools_node(state: AgentState) -> AgentState:
    unsupported_tools = state.get("unsupported_tools", [])
    if not unsupported_tools:
        return {}
    return {
        "execution_trace": append_step(
            state.get("execution_trace", []),
            unsupported_tools_step(unsupported_tools),
        )
    }


def evaluate_handoff_node(state: AgentState) -> AgentState:
    decision = evaluate_handoff(dict(state)).model_dump()
    return {
        "handoff_required": bool(decision["required"]),
        "handoff_reason": decision,
        "handoff_priority": decision.get("priority", "普通"),
        "execution_trace": append_step(
            state.get("execution_trace", []),
            handoff_evaluation_step(decision),
        ),
    }


def route_after_handoff_evaluation(state: AgentState) -> str:
    return "create_handoff" if state.get("handoff_required") else "build_response"


def _build_handoff_context(state: AgentState) -> dict[str, Any]:
    parse_result = state.get("parse_result", {})
    suggested_reply = build_customer_reply(
        parse_result,
        state.get("tool_results", {}),
        state.get("unsupported_tools", []),
        skipped_tools=state.get("skipped_tools", []),
        tool_errors=state.get("tool_errors", {}),
    )
    return {
        "question": state.get("question", ""),
        "request_id": state.get("request_id", ""),
        "thread_id": state.get("thread_id", ""),
        "parse_result": state.get("parse_result", {}),
        "called_tools": state.get("called_tools", []),
        "skipped_tools": state.get("skipped_tools", []),
        "unsupported_tools": state.get("unsupported_tools", []),
        "tool_arguments": state.get("tool_arguments", {}),
        "tool_results": state.get("tool_results", {}),
        "tool_errors": state.get("tool_errors", {}),
        "approval_decisions": state.get("approval_decisions", []),
        "execution_trace": state.get("execution_trace", []),
        "agent_suggested_reply": suggested_reply,
        "channel": state.get("channel", "web"),
        "customer_id": state.get("customer_id", ""),
        "session_id": state.get("session_id", ""),
        "turn_count": state.get("turn_count", 0),
        "messages": state.get("messages", []),
        "conversation_summary": state.get("conversation_summary", ""),
        "conversation_slots": state.get("conversation_slots", {}),
        "long_term_memories": state.get("long_term_memories", []),
        "context_snapshot": state.get("context_snapshot", {}),
        "attachments": state.get("attachments", []),
        "vision_results": state.get("vision_results", []),
        "vision_status": state.get("vision_status", ""),
        "vision_error": state.get("vision_error", ""),
        "image_confirmation_decisions": state.get(
            "image_confirmation_decisions",
            [],
        ),
        "confirmed_visual_slots": state.get("confirmed_visual_slots", {}),
    }


def create_handoff_node(
    state: AgentState,
    *,
    repository: HandoffRepository,
) -> AgentState:
    decision = state.get("handoff_reason") or {}
    context = _build_handoff_context(state)
    case = repository.create_case(
        thread_id=state["thread_id"],
        request_id=state["request_id"],
        reason_code=str(decision.get("reason_code", "manual_review")),
        reason_text=str(decision.get("reason_text", "需要人工客服继续处理。")),
        priority=str(decision.get("priority", "普通")),
        question=state["question"],
        context=context,
        channel=state.get("channel", "web"),
        customer_id=state.get("customer_id", ""),
    )
    handoff_id = str(case["handoff_id"])
    customer_reply = (
        f"这个问题已转给人工客服继续处理，服务单号 {handoff_id}。"
        "客服会看到本轮问题、已查询结果和转接原因，不需要您重复描述。"
    )
    return {
        "handoff_id": handoff_id,
        "handoff_status": str(case["status"]),
        "handoff_priority": str(case["priority"]),
        "handoff_context": context,
        "status": "waiting_human",
        "customer_reply": customer_reply,
        "execution_trace": append_step(
            state.get("execution_trace", []),
            handoff_created_step(case),
        ),
    }


def wait_for_human_node(
    state: AgentState,
    *,
    repository: HandoffRepository,
) -> AgentState:
    request = {
        "kind": "human_response",
        "handoff_id": state["handoff_id"],
        "thread_id": state["thread_id"],
        "reason": state.get("handoff_reason", {}),
        "priority": state.get("handoff_priority", "普通"),
        "context": state.get("handoff_context", {}),
        "allowed_decisions": ["respond"],
    }

    while True:
        response = interrupt(request)
        raw_reply = (
            {"message": response, "agent_name": "人工客服"}
            if isinstance(response, str)
            else response
        )
        try:
            human = HumanReply.model_validate(raw_reply)
            break
        except Exception as exc:
            request = {
                **request,
                "validation_error": f"人工回复格式无效：{exc}",
            }

    case = repository.resolve_case(
        state["handoff_id"],
        human_reply=human.message,
        agent_name=human.agent_name,
    )
    return {
        "handoff_status": str(case["status"]),
        "assigned_agent": human.agent_name,
        "human_reply": human.message,
        "customer_reply": human.message,
        "status": "human_replied",
        "execution_trace": append_step(
            state.get("execution_trace", []),
            human_response_step(state["handoff_id"], human.agent_name, human.message),
        ),
    }


def persist_memory_node(
    state: AgentState,
    *,
    repository: MemoryRepository,
) -> AgentState:
    customer_id = state.get("customer_id", "").strip()
    if state.get("memory_mode", "profile") == "off" or not customer_id:
        return {"memory_writes": []}

    parse_result = state.get("parse_result", {})
    slot_sources = parse_result.get("slot_sources", {})
    current_slots = {
        field: value
        for field, value in (parse_result.get("slots") or {}).items()
        if value is not None and slot_sources.get(field) == "current_question"
    }
    writes = repository.remember_profile_slots(
        customer_id=customer_id,
        slots=current_slots,
        source=f"confirmed_customer_input:{state.get('request_id', '')}",
        confidence=float(parse_result.get("confidence", 0.0)),
    )
    memories = repository.list_active(customer_id)
    if not writes:
        return {
            "memory_writes": [],
            "long_term_memories": memories,
        }
    return {
        "memory_writes": writes,
        "long_term_memories": memories,
        "execution_trace": append_step(
            state.get("execution_trace", []),
            memory_step(writes, customer_id),
        ),
    }


def build_response_node(
    state: AgentState,
    *,
    context_policy: ContextPolicy = DEFAULT_CONTEXT_POLICY,
    repository: MemoryRepository = DEFAULT_MEMORY_REPOSITORY,
) -> AgentState:
    parse_result = state["parse_result"]
    tool_results = state.get("tool_results", {})
    unsupported_tools = state.get("unsupported_tools", [])
    skipped_tools = state.get("skipped_tools", [])
    tool_errors = state.get("tool_errors", {})

    human_reply = state.get("human_reply", "").strip()
    vision_status = state.get("vision_status", "")
    image_inspection_only = "image_inspection" in parse_result.get("intents", [])
    if human_reply:
        status = "completed"
    elif vision_status in {
        "failed",
        "needs_better_image",
        "needs_human",
        "rejected",
    }:
        status = "need_better_image"
    elif image_inspection_only:
        status = "completed"
    elif parse_result["missing_fields"]:
        status = "need_more_info"
    elif tool_errors:
        status = "completed_with_errors"
    else:
        status = "completed"

    if human_reply:
        customer_reply = human_reply
    elif image_inspection_only:
        slots = parse_result.get("slots", {})
        confirmed_items = [
            f"{label}：{slots.get(field)}"
            for field, label in (
                ("brand", "品牌"),
                ("machine_model", "设备型号"),
                ("part_name", "配件名称"),
                ("part_number", "零件号"),
            )
            if slots.get(field)
        ]
        customer_reply = (
            "已按您的确认保留图片中的候选信息："
            + "；".join(confirmed_items)
            + "。图片识别结果只作为字段证据，不代表已确认适配关系、价格、库存或故障结论。"
        )
    elif status == "need_better_image":
        followups: list[str] = []
        for item in state.get("vision_results", []):
            inspection = item.get("inspection") or {}
            followups.extend(inspection.get("required_followups") or [])
        followup_text = "、".join(list(dict.fromkeys(followups))[:3])
        if vision_status == "failed":
            customer_reply = (
                "图片识别服务本轮没有成功，系统没有根据图片猜测配件信息。"
                "您可以补充设备品牌、机型、配件名或零件号，也可以转人工客服核对。"
            )
        elif vision_status == "rejected":
            customer_reply = (
                "已按您的确认撤回本次图片候选结果，不会把它写入机型或配件信息。"
                "请补拍铭牌、旧件标签和配件整体照片，或转人工客服核对。"
            )
        else:
            customer_reply = (
                "这张图片目前不足以可靠确认配件信息，系统没有自动合并候选字段。"
                f"{followup_text or '请补拍清晰、正对、无遮挡的铭牌或旧件标签照片。'}"
            )
    else:
        customer_reply = build_customer_reply(
            parse_result,
            tool_results,
            unsupported_tools,
            skipped_tools=skipped_tools,
            tool_errors=tool_errors,
        )
    messages = list(state.get("messages", []))
    messages.append(
        make_message(
            "human_agent" if human_reply else "assistant",
            customer_reply,
            turn_index=int(state.get("turn_count", 1)),
            request_id=state.get("request_id", ""),
        )
    )
    customer_id = str(state.get("customer_id", "")).strip()
    thread_id = str(state.get("thread_id", "")).strip()
    turn_count = int(state.get("turn_count", 1))
    memory_job_id = ""
    if customer_id and thread_id:
        semantic_store = SemanticMemoryStore(repository.path)
        semantic_store.append_message(
            customer_id=customer_id,
            thread_id=thread_id,
            turn_id=turn_count,
            role="human_agent" if human_reply else "assistant",
            content=customer_reply,
            request_id=state.get("request_id", ""),
        )
        consolidate_every = max(
            1, int(os.getenv("AGENT_MEMORY_CONSOLIDATE_EVERY_TURNS", "4"))
        )
        if turn_count % consolidate_every == 0 or int(state.get("context_dropped_messages", 0)) > 0:
            memory_job_id, _ = semantic_store.enqueue(
                customer_id=customer_id,
                thread_id=thread_id,
                upto_turn=turn_count,
                reason="periodic_turn_boundary",
            )
    messages, conversation_summary, dropped = compact_messages(
        messages,
        state.get("conversation_summary", ""),
        context_policy,
    )
    dropped_total = int(state.get("context_dropped_messages", 0)) + dropped
    context_snapshot = build_context_snapshot(
        question=state.get("question", ""),
        messages=messages,
        conversation_summary=conversation_summary,
        conversation_slots=state.get("conversation_slots", {}),
        long_term_memories=state.get("long_term_memories", []),
        episodic_memories=state.get("episodic_memories", []),
        tool_results=tool_results,
        policy=context_policy,
        dropped_messages=dropped_total,
    )
    execution_trace = list(state.get("execution_trace", []))
    if parse_result["missing_fields"]:
        execution_trace = append_step(execution_trace, missing_fields_step(parse_result))
    execution_trace = append_step(execution_trace, response_step(status, customer_reply))
    return {
        "status": status,
        "customer_reply": customer_reply,
        "messages": messages,
        "conversation_summary": conversation_summary,
        "context_snapshot": context_snapshot,
        "context_dropped_messages": dropped_total,
        "memory_job_id": memory_job_id,
        "execution_trace": execution_trace,
    }


def build_graph(
    checkpointer: Any | None = None,
    handoff_repository: HandoffRepository | None = None,
    memory_repository: MemoryRepository | None = None,
    context_policy: ContextPolicy | None = None,
    image_repository: ImageEvidenceRepository | None = None,
    vision_inspector: Callable[..., dict[str, Any]] | None = None,
):
    repository = handoff_repository or DEFAULT_HANDOFF_REPOSITORY
    memories = memory_repository or DEFAULT_MEMORY_REPOSITORY
    images = image_repository or DEFAULT_IMAGE_EVIDENCE_REPOSITORY
    active_vision_inspector = vision_inspector or inspect_image
    active_context_policy = context_policy or DEFAULT_CONTEXT_POLICY
    graph = StateGraph(AgentState)
    graph.add_node(
        "load_context",
        partial(
            load_context_node,
            repository=memories,
            context_policy=active_context_policy,
        ),
    )
    graph.add_node(
        "parse",
        partial(parse_node, context_policy=active_context_policy),
    )
    graph.add_node(
        "inspect_image",
        partial(
            inspect_image_node,
            repository=images,
            inspector=active_vision_inspector,
        ),
    )
    graph.add_node("confirm_image", confirm_image_node)
    graph.add_node("prepare_tool", prepare_tool_node)
    graph.add_node("approval", approval_node)
    graph.add_node(
        "execute_tool",
        execute_tool_node,
        retry_policy=RetryPolicy(
            max_attempts=3,
            initial_interval=0.1,
            max_interval=1.0,
            jitter=False,
            retry_on=_retryable_tool_error,
        ),
        error_handler=tool_error_handler,
        destinations=("advance_after_tool",),
    )
    graph.add_node("skip_tool", skip_tool_node)
    graph.add_node("advance_after_tool", advance_after_tool_node)
    graph.add_node("mark_unsupported_tools", mark_unsupported_tools_node)
    graph.add_node("evaluate_handoff", evaluate_handoff_node)
    graph.add_node(
        "create_handoff",
        partial(create_handoff_node, repository=repository),
    )
    graph.add_node(
        "wait_for_human",
        partial(wait_for_human_node, repository=repository),
    )
    graph.add_node(
        "persist_memory",
        partial(persist_memory_node, repository=memories),
    )
    graph.add_node(
        "build_response",
        partial(
            build_response_node,
            context_policy=active_context_policy,
            repository=memories,
        ),
    )

    graph.add_edge(START, "load_context")
    graph.add_conditional_edges(
        "load_context",
        route_after_context,
        {
            "inspect_image": "inspect_image",
            "parse": "parse",
        },
    )
    graph.add_conditional_edges(
        "inspect_image",
        route_after_image_inspection,
        {
            "confirm_image": "confirm_image",
            "parse": "parse",
        },
    )
    graph.add_edge("confirm_image", "parse")
    graph.add_conditional_edges(
        "parse",
        route_after_parse,
        {
            "prepare_tool": "prepare_tool",
            "mark_unsupported_tools": "mark_unsupported_tools",
            "evaluate_handoff": "evaluate_handoff",
        },
    )
    graph.add_conditional_edges(
        "prepare_tool",
        route_after_prepare,
        {
            "approval": "approval",
            "execute_tool": "execute_tool",
        },
    )
    graph.add_conditional_edges(
        "approval",
        route_after_approval,
        {
            "execute_tool": "execute_tool",
            "skip_tool": "skip_tool",
        },
    )
    graph.add_edge("execute_tool", "advance_after_tool")
    graph.add_edge("skip_tool", "advance_after_tool")
    graph.add_conditional_edges(
        "advance_after_tool",
        route_after_tool,
        {
            "prepare_tool": "prepare_tool",
            "mark_unsupported_tools": "mark_unsupported_tools",
            "evaluate_handoff": "evaluate_handoff",
        },
    )
    graph.add_edge("mark_unsupported_tools", "evaluate_handoff")
    graph.add_conditional_edges(
        "evaluate_handoff",
        route_after_handoff_evaluation,
        {
            "create_handoff": "create_handoff",
            "build_response": "persist_memory",
        },
    )
    graph.add_edge("create_handoff", "wait_for_human")
    graph.add_edge("wait_for_human", "persist_memory")
    graph.add_edge("persist_memory", "build_response")
    graph.add_edge("build_response", END)
    return graph.compile(checkpointer=checkpointer)


CHECKPOINTER = create_sqlite_checkpointer()
COMPILED_GRAPH = build_graph(CHECKPOINTER)


def _interrupt_payloads(result: dict[str, Any]) -> list[Any]:
    payloads = []
    for item in result.get("__interrupt__", ()) or ():
        payloads.append(getattr(item, "value", item))
    return payloads


def _format_graph_result(result: dict[str, Any], thread_id: str) -> dict[str, Any]:
    interrupt_payloads = _interrupt_payloads(result)
    formatted = {key: value for key, value in result.items() if key != "__interrupt__"}
    if interrupt_payloads:
        pending_request = interrupt_payloads[0]
        if isinstance(pending_request, dict) and pending_request.get("kind") == "human_response":
            formatted["status"] = "waiting_human"
            formatted["handoff_id"] = pending_request.get(
                "handoff_id",
                formatted.get("handoff_id", ""),
            )
            formatted["handoff_status"] = "queued"
            formatted.setdefault(
                "customer_reply",
                "当前问题已转交人工客服，人工回复后会从已保存状态继续。",
            )
        elif (
            isinstance(pending_request, dict)
            and pending_request.get("kind") == "image_evidence_confirmation"
        ):
            formatted["status"] = "waiting_image_confirmation"
            formatted["customer_reply"] = (
                "我已经从图片中提取出候选信息，请先确认品牌、机型、配件名和零件号，"
                "确认后再继续查询。"
            )
            formatted["image_confirmation_request"] = pending_request
        else:
            formatted["status"] = "waiting_approval"
            formatted["customer_reply"] = "当前操作正在等待人工审批，审批后会从已保存的节点继续执行。"
            formatted["approval_request"] = pending_request
    formatted["interrupts"] = interrupt_payloads
    formatted["thread_id"] = thread_id
    formatted.setdefault("request_id", thread_id)
    formatted.setdefault("session_id", thread_id)
    formatted.setdefault("parse_result", {})
    formatted["parse_result"].setdefault("raw_question", formatted.get("question", ""))
    formatted["parse_result"].setdefault("intents", [])
    formatted["parse_result"].setdefault("slots", {})
    formatted["parse_result"].setdefault("missing_fields", [])
    formatted["parse_result"].setdefault("candidate_tools", [])
    formatted["parse_result"].setdefault("ready_for_tools", False)
    formatted["parse_result"].setdefault("parse_source", "not_started")
    formatted["parse_result"].setdefault("confidence", 0)
    formatted["parse_result"].setdefault("debug", {})
    formatted.setdefault("tool_results", {})
    formatted.setdefault("tool_arguments", {})
    formatted.setdefault("called_tools", [])
    formatted.setdefault("skipped_tools", [])
    formatted.setdefault("unsupported_tools", [])
    formatted.setdefault("tool_errors", {})
    formatted.setdefault("tool_execution_keys", {})
    formatted.setdefault("approval_decisions", [])
    formatted.setdefault("handoff_mode", "off")
    formatted.setdefault("parser_mode", "rules")
    formatted.setdefault("knowledge_mode", False)
    formatted.setdefault("memory_mode", "profile")
    formatted.setdefault("clarification_count", 0)
    formatted.setdefault("turn_count", 0)
    formatted.setdefault("channel", "web")
    formatted.setdefault("customer_id", "")
    formatted.setdefault("thread_customer_id", formatted.get("customer_id", ""))
    formatted.setdefault("messages", [])
    formatted.setdefault("conversation_summary", "")
    formatted.setdefault("conversation_slots", {})
    formatted.setdefault("long_term_memories", [])
    formatted.setdefault("memory_writes", [])
    formatted.setdefault("context_snapshot", {})
    formatted.setdefault("context_dropped_messages", 0)
    formatted.setdefault("attachments", [])
    formatted.setdefault("vision_results", [])
    formatted.setdefault("vision_status", "")
    formatted.setdefault("vision_error", "")
    formatted.setdefault("vision_model_runtime", [])
    formatted.setdefault("image_confirmation_request", None)
    formatted.setdefault("image_confirmation_decisions", [])
    formatted.setdefault("confirmed_visual_slots", {})
    formatted.setdefault("handoff_required", False)
    formatted.setdefault("handoff_reason", {})
    formatted.setdefault("handoff_id", "")
    formatted.setdefault("handoff_status", "")
    formatted.setdefault("handoff_priority", "")
    formatted.setdefault("assigned_agent", "")
    formatted.setdefault("human_reply", "")
    formatted.setdefault("handoff_context", {})
    formatted.setdefault("execution_mode", "langgraph")
    formatted.setdefault("execution_trace", [])
    return formatted


def start_graph_agent(
    question: str,
    *,
    thread_id: str | None = None,
    request_id: str | None = None,
    approval_mode: Literal["auto", "manual"] = "manual",
    handoff_mode: Literal["off", "manual"] = "off",
    parser_mode: Literal["rules", "hybrid", "llm"] = "rules",
    knowledge_mode: bool = False,
    memory_mode: Literal["off", "profile"] = "profile",
    clarification_count: int | None = None,
    channel: str = "web",
    customer_id: str = "",
    session_id: str = "",
    attachments: list[dict[str, Any]] | None = None,
    graph: Any | None = None,
) -> dict[str, Any]:
    active_graph = graph or COMPILED_GRAPH
    active_thread_id = thread_id or uuid.uuid4().hex
    active_request_id = request_id or uuid.uuid4().hex
    graph_input: dict[str, Any] = {
        "question": question,
        "thread_id": active_thread_id,
        "request_id": active_request_id,
        "approval_mode": approval_mode,
        "handoff_mode": handoff_mode,
        "parser_mode": parser_mode,
        "knowledge_mode": knowledge_mode,
        "memory_mode": memory_mode,
        "channel": channel,
        "attachments": list(attachments or []),
    }
    if clarification_count is not None:
        graph_input["clarification_count"] = clarification_count
    if customer_id.strip():
        graph_input["customer_id"] = customer_id.strip()
    if session_id.strip():
        graph_input["session_id"] = session_id.strip()
    result = active_graph.invoke(
        graph_input,
        config=_thread_config(active_thread_id),
    )
    return _format_graph_result(result, active_thread_id)


def resume_graph_agent(
    thread_id: str,
    decision: Literal["approve", "edit", "reject"],
    *,
    edited_arguments: dict[str, Any] | None = None,
    comment: str = "",
    graph: Any | None = None,
) -> dict[str, Any]:
    active_graph = graph or COMPILED_GRAPH
    pending_payloads = _pending_snapshot_interrupts(active_graph, thread_id)
    if pending_payloads:
        kind = pending_payloads[0].get("kind") if isinstance(pending_payloads[0], dict) else ""
        if kind == "human_response":
            raise ValueError("当前线程等待人工客服回复，请使用 resume_handoff_agent。")
        if kind == "image_evidence_confirmation":
            raise ValueError("当前线程等待客户确认图片证据，请使用 resume_image_confirmation。")
    payload = {
        "decision": decision,
        "edited_arguments": edited_arguments,
        "comment": comment,
    }
    result = active_graph.invoke(
        Command(resume=payload),
        config=_thread_config(thread_id),
    )
    return _format_graph_result(result, thread_id)


def resume_image_confirmation(
    thread_id: str,
    decision: Literal["confirm", "edit", "reject", "human"],
    *,
    confirmed_fields: dict[str, Any] | None = None,
    comment: str = "",
    graph: Any | None = None,
) -> dict[str, Any]:
    active_graph = graph or COMPILED_GRAPH
    pending_payloads = _pending_snapshot_interrupts(active_graph, thread_id)
    if not pending_payloads:
        raise ValueError("当前线程没有等待图片证据确认。")
    kind = pending_payloads[0].get("kind") if isinstance(pending_payloads[0], dict) else ""
    if kind != "image_evidence_confirmation":
        raise ValueError("当前线程等待的不是图片证据确认。")
    confirmation = ImageConfirmation(
        decision=decision,
        confirmed_fields=confirmed_fields or {},
        comment=comment,
    )
    result = active_graph.invoke(
        Command(resume=confirmation.model_dump()),
        config=_thread_config(thread_id),
    )
    return _format_graph_result(result, thread_id)


def resume_handoff_agent(
    thread_id: str,
    message: str,
    *,
    agent_name: str = "人工客服",
    graph: Any | None = None,
) -> dict[str, Any]:
    active_graph = graph or COMPILED_GRAPH
    pending_payloads = _pending_snapshot_interrupts(active_graph, thread_id)
    if not pending_payloads:
        raise ValueError("当前线程没有等待人工客服回复。")
    kind = pending_payloads[0].get("kind") if isinstance(pending_payloads[0], dict) else ""
    if kind != "human_response":
        raise ValueError("当前线程等待的不是人工客服回复。")
    human = HumanReply(message=message, agent_name=agent_name)
    result = active_graph.invoke(
        Command(resume=human.model_dump()),
        config=_thread_config(thread_id),
    )
    return _format_graph_result(result, thread_id)


def resume_graph_run(thread_id: str, *, graph: Any | None = None) -> dict[str, Any]:
    active_graph = graph or COMPILED_GRAPH
    result = active_graph.invoke(None, config=_thread_config(thread_id))
    return _format_graph_result(result, thread_id)


def run_graph_agent(
    question: str,
    *,
    thread_id: str | None = None,
    approval_mode: Literal["auto", "manual"] = "auto",
    handoff_mode: Literal["off", "manual"] = "off",
    parser_mode: Literal["rules", "hybrid", "llm"] = "rules",
    knowledge_mode: bool = False,
    memory_mode: Literal["off", "profile"] = "profile",
    channel: str = "web",
    customer_id: str = "",
    session_id: str = "",
    attachments: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run the graph with automatic approval for API and regression compatibility."""
    return start_graph_agent(
        question,
        thread_id=thread_id,
        approval_mode=approval_mode,
        handoff_mode=handoff_mode,
        parser_mode=parser_mode,
        knowledge_mode=knowledge_mode,
        memory_mode=memory_mode,
        channel=channel,
        customer_id=customer_id,
        session_id=session_id,
        attachments=attachments,
    )


def _task_summary(task: Any) -> dict[str, Any]:
    interrupts = [
        getattr(item, "value", item)
        for item in (getattr(task, "interrupts", ()) or ())
    ]
    error = getattr(task, "error", None)
    return {
        "id": str(getattr(task, "id", "")),
        "name": str(getattr(task, "name", "")),
        "error": str(error) if error else "",
        "interrupts": interrupts,
    }


def _pending_snapshot_interrupts(active_graph: Any, thread_id: str) -> list[Any]:
    snapshot = active_graph.get_state(_thread_config(thread_id))
    payloads: list[Any] = []
    for task in snapshot.tasks:
        for item in (getattr(task, "interrupts", ()) or ()):
            payloads.append(getattr(item, "value", item))
    return payloads


def get_graph_state(thread_id: str, *, graph: Any | None = None) -> dict[str, Any]:
    active_graph = graph or COMPILED_GRAPH
    snapshot = active_graph.get_state(_thread_config(thread_id))
    return {
        "thread_id": thread_id,
        "values": dict(snapshot.values),
        "next": list(snapshot.next),
        "config": dict(snapshot.config),
        "metadata": dict(snapshot.metadata or {}),
        "tasks": [_task_summary(task) for task in snapshot.tasks],
    }


def load_graph_thread(
    thread_id: str,
    *,
    customer_id: str = "",
    graph: Any | None = None,
) -> dict[str, Any]:
    """Load the latest thread state and enforce customer ownership."""
    snapshot = get_graph_state(thread_id, graph=graph)
    values = snapshot["values"]
    if not values:
        raise KeyError(f"会话 checkpoint 不存在：{thread_id}")
    bound_customer_id = str(
        values.get("thread_customer_id") or values.get("customer_id") or ""
    ).strip()
    requested_customer_id = customer_id.strip()
    if (
        requested_customer_id
        and bound_customer_id
        and requested_customer_id != bound_customer_id
    ):
        raise PermissionError("无权恢复其他客户的会话")
    if requested_customer_id and not bound_customer_id:
        raise PermissionError("旧会话缺少客户归属，不能从网页直接恢复")
    return _format_graph_result(values, thread_id)


def get_graph_history(
    thread_id: str,
    *,
    limit: int = 20,
    graph: Any | None = None,
) -> list[dict[str, Any]]:
    active_graph = graph or COMPILED_GRAPH
    history = []
    for snapshot in active_graph.get_state_history(_thread_config(thread_id)):
        history.append(
            {
                "status": snapshot.values.get("status", ""),
                "current_tool": snapshot.values.get("current_tool"),
                "next": list(snapshot.next),
                "config": dict(snapshot.config),
                "metadata": dict(snapshot.metadata or {}),
                "tasks": [_task_summary(task) for task in snapshot.tasks],
            }
        )
        if len(history) >= limit:
            break
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description="Run project2 LangGraph agent workflow.")
    parser.add_argument("question", nargs="?")
    parser.add_argument("--thread-id")
    parser.add_argument("--manual-approval", action="store_true")
    parser.add_argument("--enable-handoff", action="store_true")
    parser.add_argument("--parser-mode", choices=["rules", "hybrid", "llm"], default="rules")
    parser.add_argument("--enable-knowledge", action="store_true")
    parser.add_argument("--customer-id", default="")
    parser.add_argument("--disable-memory", action="store_true")
    parser.add_argument("--resume", choices=sorted(APPROVAL_DECISIONS))
    parser.add_argument("--human-reply", default="")
    parser.add_argument("--agent-name", default="人工客服")
    parser.add_argument("--edited-arguments", default="")
    parser.add_argument("--comment", default="")
    parser.add_argument("--recover", action="store_true")
    args = parser.parse_args()

    if args.human_reply:
        if not args.thread_id:
            parser.error("--human-reply requires --thread-id")
        result = resume_handoff_agent(
            args.thread_id,
            args.human_reply,
            agent_name=args.agent_name,
        )
    elif args.recover:
        if not args.thread_id:
            parser.error("--recover requires --thread-id")
        result = resume_graph_run(args.thread_id)
    elif args.resume:
        if not args.thread_id:
            parser.error("--resume requires --thread-id")
        edited_arguments = json.loads(args.edited_arguments) if args.edited_arguments else None
        result = resume_graph_agent(
            args.thread_id,
            args.resume,
            edited_arguments=edited_arguments,
            comment=args.comment,
        )
    else:
        if not args.question:
            parser.error("question is required when starting a new run")
        result = start_graph_agent(
            args.question,
            thread_id=args.thread_id,
            approval_mode="manual" if args.manual_approval else "auto",
            handoff_mode="manual" if args.enable_handoff else "off",
            parser_mode=args.parser_mode,
            knowledge_mode=args.enable_knowledge,
            memory_mode="off" if args.disable_memory else "profile",
            customer_id=args.customer_id,
            session_id=args.thread_id or "",
        )

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
