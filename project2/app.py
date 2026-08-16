from __future__ import annotations

import json
import os
import uuid

import streamlit as st

from agent_graph import (
    get_graph_history,
    get_graph_state,
    load_graph_thread,
    resume_image_confirmation,
    resume_graph_agent,
    resume_handoff_agent,
    run_graph_agent,
    start_graph_agent,
)
from agent_workflow import run_agent
from conversation_repository import DEFAULT_CONVERSATION_REPOSITORY
from handoff_metrics import summarize_handoffs
from handoff_repository import DEFAULT_HANDOFF_REPOSITORY
from image_evidence import (
    DEFAULT_IMAGE_EVIDENCE_REPOSITORY,
    ImageValidationError,
    validate_image_upload,
)
from memory_repository import DEFAULT_MEMORY_REPOSITORY
from model_router import ModelConfigurationError, ModelRouter
from tool_call_logger import LOG_PATH, append_agent_run, read_agent_runs


EXAMPLE_QUESTIONS = [
    "小松PC200原厂液压泵要1件，有没有现货，多少钱，发到贵阳要多久？",
    "卡特320D原厂液压泵要1件，有没有库存，报价多少？",
    "小松 PC200 的液压泵有没有现货？多少钱？发到贵阳要多久？",
    "三一SY215副厂液压泵要2件，发到长沙多少钱多久？",
    "订单号 A20260616001，买错了能不能退货？",
]

APP_STATE_VERSION = "langgraph_multisession_v8"
DEPLOY_BUILD_LABEL = "project2-multisession-v2026-07-28"
GIT_COMMIT = os.getenv("GIT_COMMIT", "").strip()[:7]
DEPLOY_LABEL = (
    f"{DEPLOY_BUILD_LABEL} | commit {GIT_COMMIT}"
    if GIT_COMMIT
    else DEPLOY_BUILD_LABEL
)
RUNNERS = {
    "手写 workflow": run_agent,
    "LangGraph": run_graph_agent,
}
APPROVAL_OPTIONS = {
    "批准": "approve",
    "修改后批准": "edit",
    "拒绝": "reject",
}
PARSER_OPTIONS = {
    "规则解析": "rules",
    "混合解析": "hybrid",
    "LangChain 解析": "llm",
}
CONVERSATION_BACKFILL_KEY = "agent_run_backfill_v1"


def record_conversation_result(
    result: dict,
    *,
    customer_id: str = "",
    question: str = "",
    execution_mode: str = "LangGraph",
) -> None:
    thread_id = str(result.get("thread_id", "")).strip()
    owner_id = str(
        customer_id
        or result.get("thread_customer_id")
        or result.get("customer_id")
        or ""
    ).strip()
    if not thread_id or not owner_id:
        return
    DEFAULT_CONVERSATION_REPOSITORY.record_result(
        result,
        customer_id=owner_id,
        question=question,
        channel=str(result.get("channel", "web") or "web"),
        execution_mode=execution_mode,
    )


def backfill_conversation_directory() -> int:
    """Import existing logged LangGraph threads through the public state API once."""
    if DEFAULT_CONVERSATION_REPOSITORY.get_metadata(CONVERSATION_BACKFILL_KEY):
        return 0

    grouped: dict[str, dict] = {}
    for row in reversed(read_agent_runs(limit=1000)):
        thread_id = str(row.get("thread_id", "")).strip()
        execution_mode = str(row.get("execution_mode", ""))
        if not thread_id or "LangGraph" not in execution_mode:
            continue
        item = grouped.setdefault(
            thread_id,
            {
                "first_question": str(row.get("question", "")),
                "latest_question": str(row.get("question", "")),
                "execution_mode": execution_mode,
            },
        )
        item["latest_question"] = str(row.get("question", ""))
        item["execution_mode"] = execution_mode

    imported = 0
    for thread_id, item in grouped.items():
        try:
            result = load_graph_thread(thread_id)
            customer_id = str(
                result.get("thread_customer_id") or result.get("customer_id") or ""
            ).strip()
            if not customer_id:
                continue
            record_conversation_result(
                result,
                customer_id=customer_id,
                question=item["first_question"],
                execution_mode=item["execution_mode"],
            )
            if item["latest_question"] != item["first_question"]:
                record_conversation_result(
                    result,
                    customer_id=customer_id,
                    question=item["latest_question"],
                    execution_mode=item["execution_mode"],
                )
            imported += 1
        except (KeyError, PermissionError, ValueError):
            continue

    DEFAULT_CONVERSATION_REPOSITORY.set_metadata(
        CONVERSATION_BACKFILL_KEY,
        f"completed:{imported}",
    )
    return imported


def activate_conversation(thread_id: str, customer_id: str) -> dict:
    DEFAULT_CONVERSATION_REPOSITORY.get_thread(
        thread_id,
        customer_id=customer_id,
    )
    result = load_graph_thread(thread_id, customer_id=customer_id)
    st.session_state["active_thread_id"] = thread_id
    st.session_state["last_result"] = result
    st.session_state["customer_id"] = customer_id
    st.session_state["conversation_just_opened"] = True
    st.session_state["execution_mode"] = "LangGraph"
    st.session_state["manual_approval"] = result.get("approval_mode", "manual") == "manual"
    st.session_state["handoff_mode"] = result.get("handoff_mode", "off")
    st.session_state["parser_mode"] = result.get("parser_mode", "hybrid")
    st.session_state["knowledge_mode"] = bool(result.get("knowledge_mode", False))
    latest_question = (result.get("parse_result") or {}).get("raw_question", "")
    if latest_question:
        st.session_state["question"] = latest_question
    return result


def conversation_option_label(item: dict) -> str:
    archived = " | 已归档" if item.get("archived") else ""
    status = item.get("status", "new")
    return f"{item.get('title', '新会话')} | {status}{archived}"


def render_handoff_workbench() -> None:
    st.subheader("人工客服工作台")
    st.caption("领取 Agent 转交的服务单，查看完整上下文，并用同一 thread_id 恢复流程。")

    filter_label = st.selectbox(
        "服务单状态",
        ["待领取", "处理中", "已完成", "全部"],
    )
    status_map = {
        "待领取": "queued",
        "处理中": "claimed",
        "已完成": "resolved",
        "全部": "all",
    }
    cases = DEFAULT_HANDOFF_REPOSITORY.list_cases(status_map[filter_label], limit=100)
    all_cases = DEFAULT_HANDOFF_REPOSITORY.list_cases("all", limit=500)
    pending_outbox = DEFAULT_HANDOFF_REPOSITORY.list_outbox("pending", limit=100)
    handoff_metrics = summarize_handoffs(DEFAULT_HANDOFF_REPOSITORY)

    metric_cols = st.columns(4)
    metric_cols[0].metric("待领取", sum(item["status"] == "queued" for item in all_cases))
    metric_cols[1].metric("处理中", sum(item["status"] == "claimed" for item in all_cases))
    metric_cols[2].metric("已完成", sum(item["status"] == "resolved" for item in all_cases))
    metric_cols[3].metric("待发送", len(pending_outbox))
    with st.expander("接管指标", expanded=False):
        st.json(handoff_metrics)

    if not cases:
        st.info("当前筛选条件下没有人工服务单。")
    else:
        st.dataframe(
            [
                {
                    "服务单": item["handoff_id"],
                    "优先级": item["priority"],
                    "状态": item["status"],
                    "转接原因": item["reason_text"],
                    "负责人": item["assigned_to"],
                    "渠道": item["channel"],
                    "创建时间": item["created_at"],
                }
                for item in cases
            ],
            width="stretch",
            hide_index=True,
        )
        selected_handoff_id = st.selectbox(
            "查看服务单",
            [item["handoff_id"] for item in cases],
            format_func=lambda value: next(
                f"{item['priority']} | {value} | {item['question'][:32]}"
                for item in cases
                if item["handoff_id"] == value
            ),
        )
        selected_case = next(
            item for item in cases if item["handoff_id"] == selected_handoff_id
        )

        detail_cols = st.columns([1, 1])
        with detail_cols[0]:
            st.markdown("**客户问题**")
            st.info(selected_case["question"])
            st.markdown("**转接信息**")
            st.json(
                {
                    "handoff_id": selected_case["handoff_id"],
                    "thread_id": selected_case["thread_id"],
                    "status": selected_case["status"],
                    "priority": selected_case["priority"],
                    "reason_code": selected_case["reason_code"],
                    "reason_text": selected_case["reason_text"],
                    "channel": selected_case["channel"],
                    "customer_id": selected_case["customer_id"],
                    "assigned_to": selected_case["assigned_to"],
                }
            )
        with detail_cols[1]:
            st.markdown("**Agent 上下文**")
            st.json(selected_case["context"])

        if selected_case["status"] != "resolved":
            default_agent = selected_case["assigned_to"] or "客服小王"
            agent_name = st.text_input("客服名称", value=default_agent)
            if selected_case["status"] == "queued":
                if st.button("领取服务单", type="secondary"):
                    DEFAULT_HANDOFF_REPOSITORY.claim_case(
                        selected_case["handoff_id"],
                        agent_name,
                    )
                    st.rerun()

            with st.form("human_reply_form"):
                human_reply = st.text_area(
                    "人工回复",
                    height=150,
                    placeholder="输入将直接回复给客户的内容。",
                )
                human_submitted = st.form_submit_button(
                    "回复客户并完成",
                    type="primary",
                )

            if human_submitted:
                try:
                    if selected_case["status"] == "queued":
                        DEFAULT_HANDOFF_REPOSITORY.claim_case(
                            selected_case["handoff_id"],
                            agent_name,
                        )
                    resumed = resume_handoff_agent(
                        selected_case["thread_id"],
                        human_reply,
                        agent_name=agent_name,
                    )
                    resumed["log_run_id"] = append_agent_run(
                        result=resumed,
                        execution_mode="LangGraph human handoff resume",
                    )
                    record_conversation_result(
                        resumed,
                        customer_id=selected_case["customer_id"],
                        execution_mode="LangGraph human handoff resume",
                    )
                    st.session_state["active_thread_id"] = selected_case["thread_id"]
                    st.session_state["customer_id"] = selected_case["customer_id"]
                    st.session_state["last_result"] = resumed
                    st.success("人工回复已写入图状态，服务单已完成。")
                    st.rerun()
                except Exception as exc:
                    st.error(f"人工回复提交失败：{exc}")
        else:
            st.success(
                f"{selected_case['assigned_to']} 已完成处理："
                f"{selected_case['human_reply']}"
            )

    if pending_outbox:
        st.subheader("异步消息 Outbox")
        st.caption("非网页渠道的人工回复会先进入待发送队列，由真实渠道适配器负责投递。")
        st.dataframe(pending_outbox, width="stretch", hide_index=True)


st.set_page_config(
    page_title="多工具销售 Agent 调试台",
    layout="wide",
)

st.title("挖机配件多工具销售 Agent 调试台")
st.caption(
    "用于验证意图识别、槽位抽取、缺失信息追问、工具调用和运行日志。"
    f" 部署标识：{DEPLOY_LABEL}"
)

if st.session_state.get("app_state_version") != APP_STATE_VERSION:
    st.session_state["app_state_version"] = APP_STATE_VERSION
    st.session_state.pop("last_result", None)
    st.session_state.pop("active_thread_id", None)
    st.session_state.pop("selected_conversation_id", None)
    st.session_state.pop("customer_id", None)

try:
    backfill_conversation_directory()
except Exception as exc:
    st.sidebar.warning(f"历史会话目录回填失败：{exc}")

workspace_mode = st.sidebar.radio(
    "工作区",
    ["Agent 调试台", "人工客服工作台"],
)
if workspace_mode == "人工客服工作台":
    render_handoff_workbench()
    st.stop()

runner_labels = list(RUNNERS)
stored_execution_mode = st.session_state.get("execution_mode", runner_labels[0])
stored_parser_mode = (
    st.session_state.get("parser_mode", "hybrid")
    if stored_execution_mode == "LangGraph"
    else "hybrid"
)
stored_manual_handoff = (
    st.session_state.get("handoff_mode", "manual") == "manual"
    if stored_execution_mode == "LangGraph"
    else True
)
stored_knowledge_mode = (
    bool(st.session_state.get("knowledge_mode", True))
    if stored_execution_mode == "LangGraph"
    else True
)
stored_parser_label = next(
    (
        label
        for label, parser_mode in PARSER_OPTIONS.items()
        if parser_mode == stored_parser_mode
    ),
    list(PARSER_OPTIONS)[1],
)

with st.sidebar:
    st.caption(f"部署标识：{DEPLOY_LABEL}")
    st.caption("入口文件：project2/app.py")
    st.divider()
    st.subheader("会话")
    customer_id = st.text_input(
        "客户 ID",
        value=st.session_state.get("customer_id", "demo-customer-001"),
    ).strip()
    if st.button("新建会话", width="stretch"):
        if not customer_id:
            st.error("请先填写客户 ID，再新建会话。")
        else:
            new_thread_id = uuid.uuid4().hex
            DEFAULT_CONVERSATION_REPOSITORY.create_thread(
                thread_id=new_thread_id,
                customer_id=customer_id,
            )
            st.session_state["active_thread_id"] = new_thread_id
            st.session_state["selected_conversation_id"] = new_thread_id
            st.session_state["customer_id"] = customer_id
            st.session_state.pop("last_result", None)
            st.rerun()

    show_archived = st.toggle("显示已归档会话", value=False)
    conversations = DEFAULT_CONVERSATION_REPOSITORY.list_threads(
        customer_id,
        include_archived=show_archived,
        limit=50,
    )
    if conversations:
        conversation_map = {item["thread_id"]: item for item in conversations}
        conversation_ids = list(conversation_map)
        active_thread_id = st.session_state.get("active_thread_id", "")
        stored_conversation_id = st.session_state.get(
            "selected_conversation_id",
            "",
        )
        if stored_conversation_id not in conversation_ids:
            st.session_state["selected_conversation_id"] = (
                active_thread_id
                if active_thread_id in conversation_ids
                else conversation_ids[0]
            )
        selected_thread_id = st.selectbox(
            "最近会话",
            conversation_ids,
            key="selected_conversation_id",
            format_func=lambda value: conversation_option_label(
                conversation_map[value]
            ),
        )
        selected_conversation = conversation_map[selected_thread_id]
        st.caption(
            f"{selected_thread_id[:12]} | "
            f"{selected_conversation.get('updated_at', '')[:19]}"
        )
        open_col, archive_col = st.columns(2)
        if open_col.button(
            "打开",
            width="stretch",
            disabled=bool(selected_conversation.get("archived")),
        ):
            try:
                activate_conversation(selected_thread_id, customer_id)
                st.rerun()
            except (KeyError, PermissionError, ValueError) as exc:
                st.error(f"会话恢复失败：{exc}")
        if selected_conversation.get("archived"):
            if archive_col.button("恢复归档", width="stretch"):
                DEFAULT_CONVERSATION_REPOSITORY.restore_thread(
                    selected_thread_id,
                    customer_id=customer_id,
                )
                st.rerun()
        elif archive_col.button("归档", width="stretch"):
            DEFAULT_CONVERSATION_REPOSITORY.archive_thread(
                selected_thread_id,
                customer_id=customer_id,
            )
            if st.session_state.get("active_thread_id") == selected_thread_id:
                st.session_state.pop("active_thread_id", None)
                st.session_state.pop("last_result", None)
            st.rerun()

        with st.expander("重命名会话", expanded=False):
            renamed_title = st.text_input(
                "会话标题",
                value=selected_conversation["title"],
                key=f"conversation_title_{selected_thread_id}",
                max_chars=80,
            )
            if st.button(
                "保存标题",
                key=f"rename_conversation_{selected_thread_id}",
                width="stretch",
            ):
                try:
                    DEFAULT_CONVERSATION_REPOSITORY.rename_thread(
                        selected_thread_id,
                        customer_id=customer_id,
                        title=renamed_title,
                    )
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))
    else:
        st.caption("该客户暂无可恢复会话。")
    st.divider()
    st.subheader("执行模式")
    execution_mode = st.radio(
        "选择本轮 Agent 调度方式",
        runner_labels,
        index=(
            runner_labels.index(stored_execution_mode)
            if stored_execution_mode in runner_labels
            else 0
        ),
        horizontal=False,
    )
    manual_approval = st.toggle(
        "启用人工审批演示",
        value=bool(st.session_state.get("manual_approval", True)),
        disabled=execution_mode != "LangGraph",
        help="启用后，报价和售后工单会在执行前暂停，等待批准、修改或拒绝。",
    )
    manual_handoff = st.toggle(
        "启用人工客服接管",
        value=stored_manual_handoff,
        disabled=execution_mode != "LangGraph",
        help="启用后，工具失败、无可靠结果、高风险问题或客户明确要求人工时会创建服务单。",
    )
    parser_label = st.selectbox(
        "语义解析模式",
        list(PARSER_OPTIONS),
        index=list(PARSER_OPTIONS).index(stored_parser_label),
        disabled=execution_mode != "LangGraph",
        help="混合解析会优先使用稳定规则，只在规则置信度不足时调用 LangChain 模型。",
    )
    knowledge_mode = st.toggle(
        "启用 RAG knowledge_tool",
        value=stored_knowledge_mode,
        disabled=execution_mode != "LangGraph",
        help="适配、故障和综合咨询可复用项目一知识库；证据不足时自动转人工。",
    )
    st.divider()
    st.subheader("当前工具")
    st.write("inventory_tool：查询模拟库存")
    st.write("quote_tool：生成报价草稿")
    st.write("logistics_tool：估算物流时效")
    st.write("ticket_tool：生成售后工单草稿")
    st.write("knowledge_tool：查询项目一 RAG")
    st.divider()
    st.subheader("调试重点")
    st.write("看 `called_tools` 是否符合用户意图。")
    st.write("看 `missing_fields` 是否阻止了错误工具调用。")
    st.write("看 `tool_results` 是否来自确定性工具。")
    st.write("看 `execution_trace` 是否能解释每一步决策。")
    st.write("看历史记录是否能回放参数和结果。")
    st.write("看 checkpoint 是否能保存并恢复待审批任务。")
    st.write("看人工服务单是否携带完整上下文并能恢复原线程。")

active_handoff_mode = "manual" if execution_mode == "LangGraph" and manual_handoff else "off"
active_parser_mode = PARSER_OPTIONS[parser_label] if execution_mode == "LangGraph" else "rules"
active_knowledge_mode = execution_mode == "LangGraph" and knowledge_mode
customer_changed = (
    "customer_id" in st.session_state
    and st.session_state.get("customer_id") != customer_id
)
if customer_changed:
    st.session_state["active_thread_id"] = uuid.uuid4().hex
    st.session_state.pop("last_result", None)

default_question = st.session_state.get("question", EXAMPLE_QUESTIONS[0])
question_options = (
    EXAMPLE_QUESTIONS
    if default_question in EXAMPLE_QUESTIONS
    else [default_question, *EXAMPLE_QUESTIONS]
)

selected_example = st.selectbox(
    "示例问题",
    question_options,
    index=question_options.index(default_question),
)

with st.form("agent_debug_form"):
    question = st.text_area(
        "客户问题",
        value=selected_example,
        height=110,
        placeholder="例如：小松PC200原厂液压泵要1件，有没有现货，多少钱，发到贵阳要多久？",
    )
    uploaded_images = st.file_uploader(
        "上传配件图片",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        disabled=execution_mode != "LangGraph",
        help="最多 3 张。建议上传设备铭牌、旧件标签、配件整体或损坏部位。",
    )
    submitted = st.form_submit_button("运行 Agent")

conversation_just_opened = bool(
    st.session_state.pop("conversation_just_opened", False)
)
execution_config_changed = (
    not conversation_just_opened
    and "last_result" in st.session_state
    and (
        st.session_state.get("execution_mode") != execution_mode
        or st.session_state.get("manual_approval") != manual_approval
        or st.session_state.get("handoff_mode") != active_handoff_mode
        or st.session_state.get("parser_mode") != active_parser_mode
        or st.session_state.get("knowledge_mode") != active_knowledge_mode
        or customer_changed
    )
)
if execution_config_changed:
    st.session_state["active_thread_id"] = uuid.uuid4().hex
    st.session_state.pop("last_result", None)

st.session_state["execution_mode"] = execution_mode
st.session_state["manual_approval"] = manual_approval
st.session_state["handoff_mode"] = active_handoff_mode
st.session_state["parser_mode"] = active_parser_mode
st.session_state["knowledge_mode"] = active_knowledge_mode
st.session_state["customer_id"] = customer_id

if submitted:
    st.session_state["question"] = question
    try:
        if execution_mode == "LangGraph":
            if not customer_id:
                raise ValueError("LangGraph 多会话模式要求填写客户 ID。")
            thread_id = st.session_state.get("active_thread_id") or uuid.uuid4().hex
            attachments = []
            if submitted and uploaded_images:
                if len(uploaded_images) > 3:
                    raise ImageValidationError("每轮最多上传 3 张图片。")
                DEFAULT_IMAGE_EVIDENCE_REPOSITORY.delete_expired()
                for uploaded_image in uploaded_images:
                    validated = validate_image_upload(
                        uploaded_image.getvalue(),
                        filename=uploaded_image.name,
                        claimed_mime_type=uploaded_image.type or "",
                    )
                    attachments.append(
                        DEFAULT_IMAGE_EVIDENCE_REPOSITORY.store(
                            validated,
                            customer_id=customer_id,
                            session_id=thread_id,
                        )
                    )
            result = start_graph_agent(
                question,
                thread_id=thread_id,
                approval_mode="manual" if manual_approval else "auto",
                handoff_mode=active_handoff_mode,
                parser_mode=active_parser_mode,
                knowledge_mode=active_knowledge_mode,
                customer_id=customer_id,
                session_id=thread_id,
                attachments=attachments,
            )
            st.session_state["active_thread_id"] = thread_id
        else:
            result = run_agent(question)
            st.session_state.pop("active_thread_id", None)
    except Exception as exc:
        st.error(f"Agent 执行失败：{exc}")
        st.stop()
    result["log_run_id"] = append_agent_run(
        result=result,
        execution_mode=execution_mode,
        tool_arguments=result.get("tool_arguments", {}),
    )
    if execution_mode == "LangGraph":
        try:
            record_conversation_result(
                result,
                customer_id=customer_id,
                question=question,
                execution_mode=execution_mode,
            )
        except Exception as exc:
            st.warning(f"Agent 已完成，但会话目录更新失败：{exc}")
    st.session_state["last_result"] = result

if "last_result" not in st.session_state:
    st.info("请选择执行模式、填写客户问题，必要时上传图片，然后点击“运行 Agent”。")
    st.stop()

result = st.session_state["last_result"]
parse_result = result["parse_result"]

if result.get("status") == "waiting_image_confirmation":
    confirmation_request = result.get("image_confirmation_request") or {}
    st.warning("图片识别已完成，但候选字段尚未获得客户确认。")
    st.caption(f"thread_id：{result.get('thread_id')}")
    st.write(confirmation_request.get("reason", "请确认图片识别候选字段。"))

    evidence_ids = confirmation_request.get("evidence_ids", [])
    image_columns = st.columns(max(1, min(3, len(evidence_ids))))
    for index, evidence_id in enumerate(evidence_ids):
        try:
            image_content, metadata = DEFAULT_IMAGE_EVIDENCE_REPOSITORY.read_content(
                evidence_id,
                customer_id=customer_id,
            )
            with image_columns[index % len(image_columns)]:
                st.image(image_content, caption=metadata.get("original_filename", evidence_id))
        except Exception as exc:
            st.warning(f"图片证据 {evidence_id} 无法显示：{exc}")

    image_decision_options = {
        "确认候选字段": "confirm",
        "修改后确认": "edit",
        "识别结果不对": "reject",
        "转人工核对": "human",
    }
    with st.form("image_confirmation_form"):
        image_decision_label = st.radio(
            "客户确认",
            list(image_decision_options),
            horizontal=True,
        )
        confirmed_fields_text = st.text_area(
            "候选字段",
            value=json.dumps(
                confirmation_request.get("candidate_fields", {}),
                ensure_ascii=False,
                indent=2,
            ),
            height=170,
            help="修改后确认时可编辑 brand、machine_model、part_name、part_number。",
        )
        image_confirmation_comment = st.text_input(
            "确认备注",
            placeholder="例如：零件号已与旧件标签逐字符核对",
        )
        image_confirmation_submitted = st.form_submit_button("确认并继续")

    if image_confirmation_submitted:
        decision = image_decision_options[image_decision_label]
        confirmed_fields = {}
        if decision == "edit":
            try:
                confirmed_fields = json.loads(confirmed_fields_text)
                if not isinstance(confirmed_fields, dict):
                    raise ValueError("候选字段必须是 JSON 对象")
            except (json.JSONDecodeError, ValueError) as exc:
                st.error(f"候选字段格式不正确：{exc}")
                st.stop()
        try:
            resumed_result = resume_image_confirmation(
                result["thread_id"],
                decision,
                confirmed_fields=confirmed_fields,
                comment=image_confirmation_comment,
            )
        except Exception as exc:
            st.error(f"图片确认恢复失败：{exc}")
            st.stop()
        resumed_result["log_run_id"] = append_agent_run(
            result=resumed_result,
            execution_mode="LangGraph image confirmation",
            tool_arguments=resumed_result.get("tool_arguments", {}),
        )
        try:
            record_conversation_result(
                resumed_result,
                customer_id=customer_id,
                execution_mode="LangGraph image confirmation",
            )
        except Exception as exc:
            st.warning(f"图片确认已完成，但会话目录更新失败：{exc}")
        st.session_state["last_result"] = resumed_result
        st.rerun()

if result.get("status") == "waiting_approval":
    approval_request = result.get("approval_request") or {}
    st.warning(
        f"当前线程已暂停，等待人工审批：{approval_request.get('tool_name', 'unknown_tool')}"
    )
    st.caption(f"thread_id：{result.get('thread_id')}")
    st.write(approval_request.get("reason", "请检查工具和参数后作出决定。"))

    with st.form("human_approval_form"):
        approval_label = st.radio(
            "审批决定",
            list(APPROVAL_OPTIONS),
            horizontal=True,
        )
        edited_arguments_text = st.text_area(
            "工具参数",
            value=json.dumps(
                approval_request.get("arguments", {}),
                ensure_ascii=False,
                indent=2,
            ),
            height=180,
            help="选择“修改后批准”时，可以调整这里的 JSON 参数；系统会再次经过 Pydantic 校验。",
        )
        approval_comment = st.text_input("审批备注", placeholder="例如：数量已与客户确认")
        approval_submitted = st.form_submit_button("提交审批并继续执行")

    if approval_submitted:
        decision = APPROVAL_OPTIONS[approval_label]
        edited_arguments = None
        if decision == "edit":
            try:
                edited_arguments = json.loads(edited_arguments_text)
                if not isinstance(edited_arguments, dict):
                    raise ValueError("工具参数必须是 JSON 对象")
            except (json.JSONDecodeError, ValueError) as exc:
                st.error(f"工具参数格式不正确：{exc}")
                st.stop()

        try:
            resumed_result = resume_graph_agent(
                result["thread_id"],
                decision,
                edited_arguments=edited_arguments,
                comment=approval_comment,
            )
        except Exception as exc:
            st.error(f"恢复执行失败：{exc}")
            st.stop()

        resumed_result["log_run_id"] = append_agent_run(
            result=resumed_result,
            execution_mode="LangGraph HITL resume",
            tool_arguments=resumed_result.get("tool_arguments", {}),
        )
        try:
            record_conversation_result(
                resumed_result,
                customer_id=customer_id,
                execution_mode="LangGraph HITL resume",
            )
        except Exception as exc:
            st.warning(f"审批已完成，但会话目录更新失败：{exc}")
        st.session_state["last_result"] = resumed_result
        st.rerun()

if result.get("status") == "waiting_human":
    st.warning(
        f"当前线程已转人工，服务单：{result.get('handoff_id', 'unknown')}"
    )
    st.caption(f"thread_id：{result.get('thread_id')}")
    st.write(
        (result.get("handoff_reason") or {}).get(
            "reason_text",
            "Agent 无法可靠完成本轮问题，需要人工客服继续处理。",
        )
    )
    st.info("请在左侧切换到“人工客服工作台”，领取服务单并提交真实回复。")

left, right = st.columns([1.2, 1])

with left:
    st.subheader("客户侧回复")
    st.info(result["customer_reply"])

    st.subheader("流程状态")
    status_label = {
        "completed": "完成",
        "waiting_approval": "待审批",
        "waiting_human": "待人工",
        "waiting_image_confirmation": "待图片确认",
        "need_better_image": "需补拍图片",
        "human_replied": "已回复",
        "need_more_info": "需补信息",
        "completed_with_errors": "部分失败",
    }.get(result["status"], result["status"])
    status_cols = st.columns(6)
    status_cols[0].metric("状态", status_label)
    status_cols[1].metric("意图数", len(parse_result["intents"]))
    status_cols[2].metric("已调用工具", len(result["called_tools"]))
    status_cols[3].metric("缺失字段", len(parse_result["missing_fields"]))
    status_cols[4].metric("轨迹步骤", len(result.get("execution_trace", [])))
    status_cols[5].metric(
        "待处理",
        (
            "是"
            if result.get("status")
            in {"waiting_approval", "waiting_human", "waiting_image_confirmation"}
            else "否"
        ),
    )
    st.caption(f"当前执行模式：{execution_mode}")
    st.caption(
        f"解析：{result.get('parser_mode', active_parser_mode)} | "
        f"来源：{parse_result.get('parse_source', 'rules')} | "
        f"置信度：{parse_result.get('confidence', '未记录')}"
    )
    if result.get("thread_id"):
        st.caption(f"LangGraph thread_id：{result['thread_id']}")
    if result.get("log_run_id"):
        st.caption(f"本次运行已写入日志：{result['log_run_id']}")

with right:
    st.subheader("本轮工具调用")
    if result["called_tools"]:
        for tool_name in result["called_tools"]:
            st.success(tool_name)
    else:
        st.warning("本轮没有调用工具，通常是因为信息不足。")

    if result["unsupported_tools"]:
        st.subheader("未接入工具")
        for tool_name in result["unsupported_tools"]:
            st.warning(tool_name)

    if result.get("handoff_id"):
        st.subheader("人工接管")
        st.warning(
            f"{result['handoff_id']} | "
            f"{result.get('handoff_priority', '普通')} | "
            f"{result.get('handoff_status', result.get('status', ''))}"
        )

tabs = st.tabs(
    [
        "执行轨迹",
        "解析结果",
        "工具参数",
        "工具结果",
        "历史记录",
        "上下文 / 记忆",
        "图片证据",
        "Checkpoint / 审批",
        "完整 JSON",
    ]
)

with tabs[0]:
    trace = result.get("execution_trace") or []
    if not trace:
        st.write("暂无执行轨迹。")
    else:
        st.dataframe(
            [
                {
                    "步骤": index,
                    "节点": item.get("step"),
                    "名称": item.get("title"),
                    "状态": item.get("status"),
                    "工具": item.get("tool_name") or item.get("data", {}).get("target_tool", ""),
                    "说明": item.get("summary", ""),
                }
                for index, item in enumerate(trace, start=1)
            ],
            width="stretch",
            hide_index=True,
        )
        for index, item in enumerate(trace, start=1):
            label = f"{index}. {item.get('title')} | {item.get('status')}"
            trace_tool = item.get("tool_name") or item.get("data", {}).get("target_tool")
            if trace_tool:
                label += f" | {trace_tool}"
            with st.expander(label, expanded=index == 1):
                st.json(item)

with tabs[1]:
    st.json(parse_result)
    st.subheader("模型 Harness")
    model_runtime = result.get("model_runtime") or parse_result.get("debug", {}).get(
        "model_runtime",
        {},
    )
    if model_runtime:
        runtime_cols = st.columns(5)
        route = model_runtime.get("route", {})
        runtime_cols[0].metric("Provider", route.get("provider") or "未配置")
        runtime_cols[1].metric("模型", route.get("model") or "未调用")
        runtime_cols[2].metric("调用 / 尝试", f"{model_runtime.get('calls', 0)}/{model_runtime.get('attempts', 0)}")
        runtime_cols[3].metric(
            "输入 Token",
            model_runtime.get("estimated_input_tokens", 0),
        )
        runtime_cols[4].metric(
            "估算费用",
            f"¥{float(model_runtime.get('estimated_cost_cny', 0)):.6f}",
        )
        st.json(model_runtime)
    else:
        st.info("本轮由规则直接完成解析，没有调用模型。")

    with st.expander("当前模型路由配置", expanded=False):
        try:
            st.json(ModelRouter.from_env().describe())
        except ModelConfigurationError as exc:
            st.error(str(exc))

with tabs[2]:
    if result.get("tool_arguments"):
        st.json(result["tool_arguments"])
    else:
        st.write("暂无工具参数。")

with tabs[3]:
    if result["tool_results"]:
        st.json(result["tool_results"])
    else:
        st.write("暂无工具结果。")

with tabs[4]:
    history = read_agent_runs(limit=50)
    if not history:
        st.write("暂无历史记录。点击“运行 Agent”后会写入日志。")
    else:
        table_rows = [
            {
                "时间": row["timestamp"],
                "模式": row["execution_mode"],
                "状态": row["status"],
                "Thread": row.get("thread_id", "")[:12],
                "问题": row["question"],
                "意图": "、".join(row.get("intents") or []),
                "工具": "、".join(row.get("called_tools") or []),
                "缺失字段": "、".join(row.get("missing_fields") or []),
                "轨迹步骤": len(row.get("execution_trace") or []),
            }
            for row in history
        ]
        st.dataframe(table_rows, width="stretch", hide_index=True)

        selected_run_id = st.selectbox(
            "查看单条记录详情",
            [row["run_id"] for row in history],
            format_func=lambda run_id: next(
                f"{row['timestamp']} | {row['execution_mode']} | {row['question'][:36]}"
                for row in history
                if row["run_id"] == run_id
            ),
        )
        selected_row = next(row for row in history if row["run_id"] == selected_run_id)
        st.json(selected_row)

        if LOG_PATH.exists():
            st.download_button(
                "下载 CSV 日志",
                LOG_PATH.read_bytes(),
                file_name="agent_runs.csv",
                mime="text/csv",
            )

with tabs[5]:
    context_snapshot = result.get("context_snapshot", {})
    context_cols = st.columns(5)
    context_cols[0].metric("会话轮数", result.get("turn_count", 0))
    context_cols[1].metric("近期消息", len(result.get("messages", [])))
    context_cols[2].metric("长期记忆", len(result.get("long_term_memories", [])))
    context_cols[3].metric(
        "上下文 Token",
        (
            f"{context_snapshot.get('estimated_tokens', 0)}/"
            f"{context_snapshot.get('max_tokens', 0)}"
        ),
    )
    context_cols[4].metric(
        "注入信号",
        len(context_snapshot.get("injection_signals", [])),
    )

    st.subheader("近期多轮消息")
    st.json(result.get("messages", []))
    st.subheader("较早对话摘要")
    st.code(result.get("conversation_summary", "") or "暂无摘要。")
    st.subheader("上下文装配结果")
    st.json(context_snapshot)
    st.subheader("客户长期记忆")
    active_memories = (
        DEFAULT_MEMORY_REPOSITORY.list_active(customer_id)
        if customer_id
        else []
    )
    st.json(
        {
            "loaded": active_memories,
            "writes_this_turn": result.get("memory_writes", []),
            "slot_sources": parse_result.get("slot_sources", {}),
            "conflicts": parse_result.get("context_conflicts", []),
        }
    )

with tabs[6]:
    st.json(
        {
            "attachments": result.get("attachments", []),
            "vision_status": result.get("vision_status", ""),
            "vision_error": result.get("vision_error", ""),
            "vision_results": result.get("vision_results", []),
            "vision_model_runtime": result.get("vision_model_runtime", []),
            "image_confirmation_request": result.get("image_confirmation_request"),
            "image_confirmation_decisions": result.get(
                "image_confirmation_decisions",
                [],
            ),
            "confirmed_visual_slots": result.get("confirmed_visual_slots", {}),
        }
    )

with tabs[7]:
    if execution_mode != "LangGraph" or not result.get("thread_id"):
        st.info("切换到 LangGraph 模式后，可以查看 checkpoint 和人工审批状态。")
    else:
        checkpoint_state = get_graph_state(result["thread_id"])
        checkpoint_history = get_graph_history(result["thread_id"], limit=20)

        checkpoint_cols = st.columns(4)
        checkpoint_cols[0].metric("Checkpoint 数量", len(checkpoint_history))
        checkpoint_cols[1].metric("下一节点", "、".join(checkpoint_state["next"]) or "END")
        checkpoint_cols[2].metric("审批记录", len(result.get("approval_decisions", [])))
        checkpoint_cols[3].metric("人工服务单", "有" if result.get("handoff_id") else "无")

        st.subheader("审批状态")
        st.json(
            {
                "approval_request": result.get("approval_request"),
                "approval_decisions": result.get("approval_decisions", []),
                "skipped_tools": result.get("skipped_tools", []),
                "tool_errors": result.get("tool_errors", {}),
                "tool_execution_keys": result.get("tool_execution_keys", {}),
                "handoff_id": result.get("handoff_id"),
                "handoff_status": result.get("handoff_status"),
                "handoff_reason": result.get("handoff_reason", {}),
                "assigned_agent": result.get("assigned_agent"),
                "human_reply": result.get("human_reply"),
            }
        )

        st.subheader("最新 StateSnapshot")
        st.json(checkpoint_state)

        st.subheader("Checkpoint 历史")
        st.dataframe(
            [
                {
                    "序号": index,
                    "状态": item.get("status"),
                    "当前工具": item.get("current_tool") or "",
                    "下一节点": "、".join(item.get("next") or []) or "END",
                    "步骤": item.get("metadata", {}).get("step", ""),
                    "来源": item.get("metadata", {}).get("source", ""),
                }
                for index, item in enumerate(checkpoint_history, start=1)
            ],
            width="stretch",
            hide_index=True,
        )

with tabs[8]:
    st.code(json.dumps(result, ensure_ascii=False, indent=2), language="json")
