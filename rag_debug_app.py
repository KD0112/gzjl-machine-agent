from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st

import rag_chat


DEMO_CASES: list[dict[str, Any]] = [
    {
        "question": "电器件通电了还能换货吗？",
        "scene": "售后边界与风险控制",
        "expected_type": "售后质保 / after_sales",
        "core_capability": "分类、追问、售后转人工",
        "talk_track": "这条用来展示高风险售后问题。系统不是直接承诺能换，而是先识别为售后质保，再从售后政策文档里找到电器件通电后的处理边界，最后生成给售后专员的工单草稿。",
        "watch_points": [
            "问题类型应为售后质保，且需要主动追问。",
            "Top-K 应命中《售后质保与退换货政策》，metadata 里 risk_level 为高。",
            "工单草稿应包含订单号、购买时间、照片/视频、是否安装使用等缺失信息。",
            "Prompt 中应提醒模型不能直接承诺退款、退货、换货或赔付。",
        ],
    },
    {
        "question": "液压泵异响怎么排查？",
        "scene": "故障诊断与维修协同",
        "expected_type": "故障诊断 / fault_diagnosis",
        "core_capability": "故障 FAQ、证据收集、维修转接",
        "talk_track": "这条用来展示故障诊断场景。系统会把问题识别为故障类咨询，从故障诊断 FAQ 中检索液压泵异响的排查步骤，并把需要补充的油液、滤芯、视频和维修师傅判断整理进工单。",
        "watch_points": [
            "问题类型应为故障诊断，处理人倾向维修或技术人员。",
            "Top-K 应命中《故障诊断FAQ》或液压件相关知识。",
            "回答不应直接判断配件质量问题，而应先引导排查油液、滤芯、管路和压力。",
            "工单里应保留风险提示，避免客服越权做维修结论。",
        ],
    },
    {
        "question": "没有零件号能不能配液压泵？",
        "scene": "型号适配与缺信息追问",
        "expected_type": "型号适配 / part_match",
        "core_capability": "适配规则、主动追问、防错配",
        "talk_track": "这条用来展示 RAG 和业务规则结合。液压泵不能只按机型粗略匹配，系统会命中型号适配规则，提醒需要旧件铭牌、整机铭牌、总成照片或零件号，再进入人工确认。",
        "watch_points": [
            "问题类型应为型号适配，且需要主动追问。",
            "Top-K 应命中《型号适配与件号确认规则》。",
            "缺失信息应包含机型、零件号、铭牌照片或旧件照片。",
            "话术重点是降低错配风险，而不是为了成交直接报价。",
        ],
    },
    {
        "question": "你们能不能给三档方案？",
        "scene": "报价档位与销售策略",
        "expected_type": "报价咨询 / part_quote",
        "core_capability": "多档报价、客户预算匹配",
        "talk_track": "这条用来展示销售咨询。系统会检索报价档位说明或客户 FAQ，把方案拆成经济型、稳定型、原厂型，并提醒客服结合工况、预算和质保要求确认。",
        "watch_points": [
            "问题类型应偏向报价咨询或配件报价。",
            "Top-K 应命中《报价与品质档位说明》或客户常见问题。",
            "回答应强调三档方案的使用场景，而不是只给一个价格。",
            "如果缺少具体配件型号，系统应继续追问机型、配件名和预算。",
        ],
    },
    {
        "question": "PC200液压泵有没有现货？",
        "scene": "库存咨询与采购跟进",
        "expected_type": "库存采购 / inventory_procurement",
        "core_capability": "库存意图、锁货提醒、采购转接",
        "talk_track": "这条用来展示库存类咨询。系统会识别客户在问现货，但不会编造库存数量，而是提示库存变化快，需要确认具体件号、品质档位和收货地，再由销售或采购跟进。",
        "watch_points": [
            "问题类型应偏向库存或采购咨询。",
            "Top-K 可命中库存、报价、型号适配相关片段。",
            "回答不能编造现货数量或发货承诺。",
            "工单草稿应把具体机型、配件和收货地列为待补充信息。",
        ],
    },
]

EXAMPLE_QUESTIONS = [case["question"] for case in DEMO_CASES]

METADATA_KEYS = [
    "title",
    "category",
    "risk_level",
    "company",
    "document_type",
    "version",
    "updated_date",
    "source",
]


def _source_name(metadata: dict[str, Any]) -> str:
    source = metadata.get("source", "")
    return Path(str(source)).name if source else "unknown"


def _doc_rows(docs: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, doc in enumerate(docs, start=1):
        metadata = doc.metadata or {}
        rows.append(
            {
                "rank": index,
                "source": _source_name(metadata),
                "title": metadata.get("title", ""),
                "category": metadata.get("category", ""),
                "risk_level": metadata.get("risk_level", ""),
                "preview": doc.page_content[:120].replace("\n", " "),
            }
        )
    return rows


def _context_text(docs: list[Any]) -> str:
    return "\n\n".join(
        f"[Source {index}: {_source_name(doc.metadata)}]\n{doc.page_content}"
        for index, doc in enumerate(docs, start=1)
    )


def _demo_case_for(question: str) -> dict[str, Any] | None:
    normalized_question = question.strip()
    for case in DEMO_CASES:
        if case["question"] == normalized_question:
            return case
    return None


def run_debug(question: str, k: int, with_llm: bool) -> dict[str, Any]:
    classification = rag_chat.classify_question(question)
    docs = rag_chat.retrieve(question, k=k)
    ticket_draft = rag_chat.generate_ticket_draft(question, classification, docs)
    context = _context_text(docs)
    prompt = rag_chat.build_prompt(question, context, classification)

    answer = ""
    answer_error = ""
    if with_llm:
        try:
            result = rag_chat.answer_with_metadata(question, k=k)
            answer = result["answer"]
        except Exception as exc:
            answer_error = str(exc)

    return {
        "question": question,
        "top_k": k,
        "classification": classification,
        "docs": docs,
        "doc_rows": _doc_rows(docs),
        "ticket_draft": ticket_draft,
        "context": context,
        "prompt": prompt,
        "answer": answer,
        "answer_error": answer_error,
    }


def render_doc_card(index: int, doc: Any) -> None:
    metadata = doc.metadata or {}
    title = metadata.get("title") or _source_name(metadata)
    category = metadata.get("category") or "未标注"
    risk_level = metadata.get("risk_level") or "未标注"

    with st.expander(f"Top {index} | {title} | {category} | 风险：{risk_level}", expanded=index == 1):
        meta_view = {key: metadata.get(key) for key in METADATA_KEYS if metadata.get(key)}
        st.json(meta_view or metadata)
        st.markdown(doc.page_content)


def render_demo_case(case: dict[str, Any] | None, result: dict[str, Any]) -> None:
    if not case:
        st.info("当前是自定义问题，暂无预设演示讲法。可以先看分类、Top-K、工单草稿和 Prompt，再按实际命中结果组织讲解。")
        return

    st.subheader(case["scene"])

    cols = st.columns(3)
    cols[0].metric("预期类型", case["expected_type"])
    cols[1].metric("核心能力", case["core_capability"])
    cols[2].metric("实际类型", result["classification"]["label"])

    st.markdown("**讲解稿**")
    st.write(case["talk_track"])

    st.markdown("**现场观察点**")
    for point in case["watch_points"]:
        st.markdown(f"- {point}")

    st.markdown("**建议展示顺序**")
    st.markdown(
        "\n".join(
            [
                "- 先看顶部指标：确认分类、是否追问、命中片段数。",
                "- 再看 Top-K 概览：说明答案来自哪类业务文档和风险等级。",
                "- 展开检索片段：证明不是凭空回答，而是命中了具体知识。",
                "- 打开工单草稿：说明系统能把客户咨询转成内部可处理流程。",
                "- 最后看 Prompt：说明生成回答前已经注入业务边界和追问策略。",
            ]
        )
    )


def main() -> None:
    st.set_page_config(page_title="内部 RAG 调试台", layout="wide")

    st.title("企业客服 RAG 内部调试台")
    st.caption("用于查看问题分类、Top-K 检索片段、metadata、工单草稿和可选的大模型回答。")

    with st.sidebar:
        st.subheader("调试参数")
        k = st.slider("Top-K", min_value=1, max_value=8, value=3)
        with_llm = st.checkbox("调用 DeepSeek 生成回答", value=False)
        st.divider()
        st.subheader("演示问题")
        selected_question = st.selectbox("选择问题", EXAMPLE_QUESTIONS)

    question = st.text_area(
        "用户问题",
        value=selected_question,
        height=90,
        placeholder="输入一个客服问题，例如：液压泵异响怎么排查？",
    )
    submitted = st.button("运行 RAG 调试", type="primary")

    should_run = submitted or "rag_debug_result" not in st.session_state
    if should_run:
        st.session_state["rag_debug_result"] = run_debug(question, k, with_llm)

    result = st.session_state["rag_debug_result"]
    classification = result["classification"]

    cols = st.columns(5)
    cols[0].metric("问题类型", classification["label"])
    cols[1].metric("类型代码", classification["type"])
    cols[2].metric("是否追问", "是" if classification["needs_follow_up"] else "否")
    cols[3].metric("Top-K", result["top_k"])
    cols[4].metric("命中片段", len(result["docs"]))

    if classification["follow_up_questions"]:
        st.warning("\n".join(f"- {item}" for item in classification["follow_up_questions"]))

    left, right = st.columns([1.05, 1])
    with left:
        st.subheader("Top-K 检索概览")
        st.dataframe(result["doc_rows"], width="stretch", hide_index=True)

    with right:
        st.subheader("分类结果")
        st.json(classification)

    tabs = st.tabs(["演示讲法", "检索片段", "工单草稿", "回答 / Prompt", "完整 JSON"])

    with tabs[0]:
        render_demo_case(_demo_case_for(result["question"]), result)

    with tabs[1]:
        for index, doc in enumerate(result["docs"], start=1):
            render_doc_card(index, doc)

    with tabs[2]:
        st.json(result["ticket_draft"])

    with tabs[3]:
        if result["answer"]:
            st.subheader("DeepSeek 回答")
            st.markdown(result["answer"])
        elif result["answer_error"]:
            st.error(result["answer_error"])
        else:
            st.info("当前未调用 DeepSeek。勾选侧边栏选项后重新运行可生成最终回答。")

        st.subheader("构造后的 Prompt")
        st.code(result["prompt"], language="text")

    with tabs[4]:
        serializable = {
            "question": result["question"],
            "top_k": result["top_k"],
            "classification": result["classification"],
            "doc_rows": result["doc_rows"],
            "ticket_draft": result["ticket_draft"],
            "answer": result["answer"],
            "answer_error": result["answer_error"],
        }
        st.code(json.dumps(serializable, ensure_ascii=False, indent=2), language="json")


if __name__ == "__main__":
    main()
