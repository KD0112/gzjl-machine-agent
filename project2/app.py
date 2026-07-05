from __future__ import annotations

import json

import streamlit as st

from agent_graph import run_graph_agent
from agent_workflow import run_agent
from tool_call_logger import LOG_PATH, append_agent_run, read_agent_runs


EXAMPLE_QUESTIONS = [
    "小松PC200原厂液压泵要1件，有没有现货，多少钱，发到贵阳要多久？",
    "卡特320D原厂液压泵要1件，有没有库存，报价多少？",
    "小松 PC200 的液压泵有没有现货？多少钱？发到贵阳要多久？",
    "三一SY215副厂液压泵要2件，发到长沙多少钱多久？",
    "订单号 A20260616001，买错了能不能退货？",
]

APP_STATE_VERSION = "tool_history_v2"
RUNNERS = {
    "手写 workflow": run_agent,
    "LangGraph": run_graph_agent,
}


st.set_page_config(
    page_title="多工具销售 Agent 调试台",
    layout="wide",
)

st.title("挖机配件多工具销售 Agent 调试台")
st.caption("用于验证意图识别、槽位抽取、缺失信息追问、工具调用和运行日志。")

if st.session_state.get("app_state_version") != APP_STATE_VERSION:
    st.session_state["app_state_version"] = APP_STATE_VERSION
    st.session_state.pop("last_result", None)

with st.sidebar:
    st.subheader("执行模式")
    execution_mode = st.radio(
        "选择本轮 Agent 调度方式",
        list(RUNNERS),
        horizontal=False,
    )
    st.divider()
    st.subheader("当前工具")
    st.write("inventory_tool：查询模拟库存")
    st.write("quote_tool：生成报价草稿")
    st.write("logistics_tool：估算物流时效")
    st.write("ticket_tool：生成售后工单草稿")
    st.divider()
    st.subheader("调试重点")
    st.write("看 `called_tools` 是否符合用户意图。")
    st.write("看 `missing_fields` 是否阻止了错误工具调用。")
    st.write("看 `tool_results` 是否来自确定性工具。")
    st.write("看 `execution_trace` 是否能解释每一步决策。")
    st.write("看历史记录是否能回放参数和结果。")

default_question = st.session_state.get("question", EXAMPLE_QUESTIONS[0])

selected_example = st.selectbox(
    "示例问题",
    EXAMPLE_QUESTIONS,
    index=EXAMPLE_QUESTIONS.index(default_question)
    if default_question in EXAMPLE_QUESTIONS
    else 0,
)

with st.form("agent_debug_form"):
    question = st.text_area(
        "客户问题",
        value=selected_example,
        height=110,
        placeholder="例如：小松PC200原厂液压泵要1件，有没有现货，多少钱，发到贵阳要多久？",
    )
    submitted = st.form_submit_button("运行 Agent")

should_rerun_agent = (
    submitted
    or "last_result" not in st.session_state
    or st.session_state.get("question") != question
    or st.session_state.get("execution_mode") != execution_mode
)

if should_rerun_agent:
    st.session_state["question"] = question
    st.session_state["execution_mode"] = execution_mode
    result = RUNNERS[execution_mode](question)
    if submitted:
        result["log_run_id"] = append_agent_run(
            result=result,
            execution_mode=execution_mode,
            tool_arguments=result.get("tool_arguments", {}),
        )
    st.session_state["last_result"] = result

result = st.session_state["last_result"]
parse_result = result["parse_result"]

left, right = st.columns([1.2, 1])

with left:
    st.subheader("客户侧回复")
    st.info(result["customer_reply"])

    st.subheader("流程状态")
    status_cols = st.columns(5)
    status_cols[0].metric("状态", result["status"])
    status_cols[1].metric("意图数", len(parse_result["intents"]))
    status_cols[2].metric("已调用工具", len(result["called_tools"]))
    status_cols[3].metric("缺失字段", len(parse_result["missing_fields"]))
    status_cols[4].metric("轨迹步骤", len(result.get("execution_trace", [])))
    st.caption(f"当前执行模式：{execution_mode}")
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

tabs = st.tabs(["执行轨迹", "解析结果", "工具参数", "工具结果", "历史记录", "完整 JSON"])

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
                    "工具": item.get("tool_name", ""),
                    "说明": item.get("summary", ""),
                }
                for index, item in enumerate(trace, start=1)
            ],
            width="stretch",
            hide_index=True,
        )
        for index, item in enumerate(trace, start=1):
            label = f"{index}. {item.get('title')} | {item.get('status')}"
            if item.get("tool_name"):
                label += f" | {item.get('tool_name')}"
            with st.expander(label, expanded=index == 1):
                st.json(item)

with tabs[1]:
    st.json(parse_result)

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
    st.code(json.dumps(result, ensure_ascii=False, indent=2), language="json")
