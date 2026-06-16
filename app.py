import html
import re

import streamlit as st

import rag_chat


DEFAULT_TOP_K = 3
SERVICE_PHONE = "18750528881"
MANUAL_SERVICE_PHONE = "13608517353"
MANUAL_SERVICE_NOTICE = (
    f"如 AI 助手暂未完全解决您的问题，欢迎致电 {MANUAL_SERVICE_PHONE} 联系人工客服，"
    "我们将进一步为您核实配件型号、库存、报价与售后方案。"
)

PART_IMAGES = [
    {
        "name": "液压泵 / 主泵",
        "desc": "适用于挖机液压系统询价与型号核对",
        "url": "https://www.hydpumpparts.com/photo/pc159016159-jcb_parts_1_stage_333_g5393_hydraulic_gear_pump_for_backhoe_loader.jpg",
    },
    {
        "name": "斗齿 / 齿座",
        "desc": "支持按机型、尺寸、旧件照片辅助匹配",
        "url": "https://www.bluediamondattachments.com/uploads/product-spec/_mediumSize/210005_1-1.jpg",
    },
    {
        "name": "滤芯 / 保养件",
        "desc": "机油滤、柴油滤、液压滤等常用件沟通",
        "url": "https://iprorwxhllkplp5p-static.micyjz.com/cloud/liBpoKlmljSRmjioqonrip/1.jpg",
    },
]

QUICK_QUESTIONS = [
    "PC200液压泵有没有现货？",
    "液压泵异响怎么排查？",
    "买错了能不能退货？",
]


st.set_page_config(
    page_title="劲龙机械配件智能助手",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #0d2421;
        --muted: #62716b;
        --soft: #f5f7ef;
        --panel: #eef3e7;
        --line: #d5ddcf;
        --green: #245f47;
        --green-2: #174633;
        --orange: #f59b22;
        --orange-2: #ffb24a;
        --red: #ff5962;
        --shadow: 0 18px 42px rgba(20, 48, 38, 0.1);
    }

    .stApp {
        background:
            radial-gradient(circle at 10% 0%, rgba(245, 155, 34, 0.08), transparent 32%),
            linear-gradient(180deg, #fbfcf7 0%, #f5f7ef 46%, #eef3e7 100%);
        color: var(--ink);
    }

    header[data-testid="stHeader"],
    div[data-testid="stDecoration"],
    button[kind="header"] {
        display: none;
    }

    .block-container {
        max-width: 1120px;
        padding: 30px 24px 32px;
    }

    .jl-topbar {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 20px;
        margin-bottom: 16px;
    }

    .jl-brand {
        display: flex;
        align-items: center;
        gap: 14px;
    }

    .jl-logo {
        width: 52px;
        height: 52px;
        border-radius: 15px;
        display: grid;
        place-items: center;
        background: linear-gradient(145deg, var(--orange), #d96f00);
        color: #111a17;
        font-size: 28px;
        font-weight: 900;
        box-shadow: 0 14px 28px rgba(245, 155, 34, 0.28);
    }

    .jl-title {
        margin: 0;
        font-size: 38px;
        line-height: 1.12;
        font-weight: 850;
        letter-spacing: 0;
        color: var(--ink);
    }

    .jl-subtitle {
        margin-top: 8px;
        color: var(--muted);
        font-size: 15px;
    }

    .jl-phone {
        background: var(--green);
        color: #fff;
        border-radius: 8px;
        padding: 14px 18px;
        font-weight: 800;
        font-size: 14px;
        box-shadow: 0 16px 32px rgba(36, 95, 71, 0.23);
        white-space: nowrap;
    }

    .jl-chips {
        display: flex;
        gap: 9px;
        flex-wrap: wrap;
        margin: 10px 0 18px;
    }

    .jl-chip {
        border: 1px solid #cbd6c4;
        background: rgba(255, 255, 255, 0.76);
        color: var(--green);
        border-radius: 7px;
        padding: 7px 12px;
        font-size: 13px;
        font-weight: 650;
    }

    .jl-rule {
        height: 1px;
        background: var(--line);
        margin: 16px 0 28px;
    }

    .jl-showcase {
        border-radius: 10px;
        border: 1px solid rgba(213, 221, 207, 0.85);
        background:
            linear-gradient(180deg, rgba(250, 252, 244, 0.98), rgba(224, 232, 214, 0.94)),
            var(--panel);
        min-height: 322px;
        padding: 22px 44px 26px;
        position: relative;
        overflow: hidden;
        box-shadow: var(--shadow);
        margin-bottom: 26px;
    }

    .jl-showcase:after {
        content: "";
        position: absolute;
        left: 0;
        right: 0;
        bottom: 54px;
        height: 28px;
        background: rgba(188, 201, 178, 0.55);
        border-top: 1px solid rgba(157, 174, 146, 0.35);
        border-bottom: 1px solid rgba(157, 174, 146, 0.35);
    }

    .jl-showcase-top {
        display: grid;
        grid-template-columns: 64px 1fr;
        align-items: center;
        gap: 18px;
        position: relative;
        z-index: 1;
        margin-bottom: 12px;
    }

    .jl-bot {
        width: 58px;
        height: 58px;
        border-radius: 50%;
        background:
            radial-gradient(circle at 32% 42%, #79d8ff 0 8%, transparent 9%),
            radial-gradient(circle at 68% 42%, #79d8ff 0 8%, transparent 9%),
            radial-gradient(circle at 50% 45%, #ffffff 0 32%, #1d2c38 33% 100%);
        border: 2px solid rgba(20, 45, 60, 0.18);
        box-shadow: 0 12px 22px rgba(20, 44, 30, 0.13);
    }

    .jl-search {
        min-height: 44px;
        border-radius: 9px;
        border: 2px solid var(--green);
        background: var(--green);
        color: #ffffff;
        display: flex;
        align-items: center;
        padding: 0 18px;
        font-size: 14px;
        font-weight: 780;
        box-shadow: inset 0 -1px 0 rgba(255, 255, 255, 0.14);
    }

    .jl-caption {
        display: flex;
        justify-content: space-between;
        gap: 14px;
        margin: 0 0 14px 82px;
        color: #2d4039;
        font-size: 13px;
        font-weight: 720;
        position: relative;
        z-index: 1;
    }

    .jl-products {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 34px;
        position: relative;
        z-index: 1;
    }

    .jl-product {
        border-radius: 10px;
        background: #ffffff;
        border: 1px solid #bbc7b4;
        padding: 14px 14px 12px;
        text-align: center;
        box-shadow: 0 14px 26px rgba(31, 48, 38, 0.08);
        min-height: 200px;
    }

    .jl-product img {
        width: 100%;
        height: 132px;
        object-fit: contain;
        display: block;
        margin-bottom: 10px;
    }

    .jl-product-name {
        color: var(--ink);
        font-weight: 820;
        font-size: 15px;
        margin-bottom: 4px;
    }

    .jl-product-desc {
        color: var(--muted);
        font-size: 12px;
        line-height: 1.45;
    }

    .jl-service-row {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 12px;
        margin: 4px 0 22px;
    }

    .jl-service-card {
        border: 1px solid #d5ddcf;
        border-radius: 9px;
        background: rgba(255, 255, 255, 0.75);
        padding: 13px 14px;
        color: var(--ink);
    }

    .jl-service-title {
        font-weight: 820;
        font-size: 14px;
        margin-bottom: 5px;
    }

    .jl-service-text {
        color: var(--muted);
        font-size: 12px;
        line-height: 1.5;
    }

    .jl-section-title {
        margin: 26px 0 14px;
        font-size: 18px;
        font-weight: 840;
        color: var(--ink);
    }

    div.stButton > button {
        border-radius: 8px;
        border: 1px solid #cfd8ca;
        background: #ffffff;
        color: var(--green);
        min-height: 42px;
        font-size: 14px;
        font-weight: 650;
        box-shadow: 0 6px 16px rgba(20, 48, 38, 0.04);
    }

    div.stButton > button:hover {
        border-color: var(--green);
        color: var(--green);
        background: #f8fbf4;
    }

    section[data-testid="stChatMessage"] {
        background: transparent;
        border: none;
        box-shadow: none;
        padding: 4px 0;
    }

    section[data-testid="stChatMessage"] p,
    section[data-testid="stChatMessage"] li {
        color: var(--ink);
        line-height: 1.76;
        font-size: 15px;
    }

    [data-testid="chatAvatarIcon-assistant"] {
        background: var(--orange);
    }

    [data-testid="chatAvatarIcon-user"] {
        background: var(--red);
    }

    div[data-testid="stChatInput"] {
        background: rgba(245, 247, 239, 0.96);
        border-top: 1px solid var(--line);
    }

    div[data-testid="stChatInput"] textarea {
        background: #ffffff;
        color: var(--ink);
        border: 1px solid #cfd8ca;
        border-radius: 9px;
    }

    .jl-footer-note {
        margin: 18px 0 76px;
        color: #7a8580;
        font-size: 12px;
        text-align: center;
    }

    @media (max-width: 860px) {
        .block-container {
            padding: 20px 16px 28px;
        }

        .jl-topbar {
            display: block;
        }

        .jl-brand {
            align-items: flex-start;
        }

        .jl-title {
            font-size: 30px;
        }

        .jl-phone {
            display: inline-block;
            margin-top: 16px;
        }

        .jl-showcase {
            padding: 18px;
        }

        .jl-showcase-top {
            grid-template-columns: 52px 1fr;
        }

        .jl-bot {
            width: 50px;
            height: 50px;
        }

        .jl-caption {
            margin-left: 0;
            display: block;
        }

        .jl-products,
        .jl-service-row {
            grid-template-columns: 1fr;
            gap: 12px;
        }

        .jl-showcase:after {
            display: none;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_header() -> None:
    st.markdown(
        f"""
        <div class="jl-topbar">
            <div class="jl-brand">
                <div class="jl-logo">JL</div>
                <div>
                    <h1 class="jl-title">劲龙机械配件助手</h1>
                    <div class="jl-subtitle">挖机配件咨询、型号匹配、故障排查、售后处理和发货沟通</div>
                </div>
            </div>
            <div class="jl-phone">联系电话：{SERVICE_PHONE}</div>
        </div>
        <div class="jl-chips">
            <span class="jl-chip">液压泵</span>
            <span class="jl-chip">主控阀</span>
            <span class="jl-chip">行走马达</span>
            <span class="jl-chip">斗齿滤芯</span>
            <span class="jl-chip">售后质保</span>
            <span class="jl-chip">物流发货</span>
        </div>
        <div class="jl-rule"></div>
        """,
        unsafe_allow_html=True,
    )


def render_showcase() -> None:
    product_cards = "".join(
        f"""
        <div class="jl-product">
            <img src="{html.escape(item['url'])}" alt="{html.escape(item['name'])}">
            <div class="jl-product-name">{html.escape(item['name'])}</div>
            <div class="jl-product-desc">{html.escape(item['desc'])}</div>
        </div>
        """
        for item in PART_IMAGES
    )
    st.markdown(
        f"""
        <div class="jl-showcase">
            <div class="jl-showcase-top">
                <div class="jl-bot"></div>
                <div class="jl-search">输入设备型号、配件名称或故障现象，助手会先判断需要补充的信息</div>
            </div>
            <div class="jl-caption">
                <span>常用配件：泵阀 / 斗齿 / 滤芯</span>
                <span>先核型号，再谈报价和发货</span>
            </div>
            <div class="jl-products">{product_cards}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_service_cards() -> None:
    st.markdown(
        """
        <div class="jl-service-row">
            <div class="jl-service-card">
                <div class="jl-service-title">型号先确认</div>
                <div class="jl-service-text">优先核对品牌、机型、铭牌、旧件号和安装位置。</div>
            </div>
            <div class="jl-service-card">
                <div class="jl-service-title">报价更稳妥</div>
                <div class="jl-service-text">库存、品质档位、是否急用确认后，再进入正式报价。</div>
            </div>
            <div class="jl-service-card">
                <div class="jl-service-title">故障可追问</div>
                <div class="jl-service-text">遇到异响、漏油、动作慢，会先给排查顺序和补充信息。</div>
            </div>
            <div class="jl-service-card">
                <div class="jl-service-title">人工可接手</div>
                <div class="jl-service-text">复杂配件、售后和急件可转人工继续核实处理。</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def init_chat_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "您好，我是劲龙机械配件助手。您可以告诉我设备品牌、完整型号、配件名称、"
                    "旧件照片信息或故障现象，我会先帮您判断下一步需要确认什么。"
                ),
            }
        ]


def render_quick_questions() -> None:
    st.markdown('<div class="jl-section-title">常见问题</div>', unsafe_allow_html=True)
    cols = st.columns(len(QUICK_QUESTIONS))
    for col, example in zip(cols, QUICK_QUESTIONS):
        if col.button(example, use_container_width=True):
            st.session_state.pending_question = example
            st.rerun()


def render_messages() -> None:
    st.markdown('<div class="jl-section-title">智能对话</div>', unsafe_allow_html=True)
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def clean_customer_answer(answer: str) -> str:
    internal_prefix_patterns = [
        r"(?im)^\s*根据(企业)?知识库[，,:：]*\s*",
        r"(?im)^\s*根据(给定)?资料[，,:：]*\s*",
        r"(?im)^\s*根据现有资料[，,:：]*\s*",
        r"(?im)^\s*目前资料显示[，,:：]*\s*",
        r"(?im)^\s*资料中显示[，,:：]*\s*",
    ]
    for pattern in internal_prefix_patterns:
        answer = re.sub(pattern, "", answer)

    replacements = {
        "知识库中没有找到依据": "暂未查询到明确记录",
        "知识库没有找到依据": "暂未查询到明确记录",
        "根据知识库": "",
        "根据给定资料": "",
        "根据现有资料": "",
        "目前资料显示": "",
        "给定资料": "现有信息",
        "资料中仅提到": "目前仅能确认",
        "资料中": "目前信息中",
        "RAG": "",
        "向量库": "",
        "Source": "来源",
    }
    for old, new in replacements.items():
        answer = answer.replace(old, new)

    answer = re.sub(r"(?im)^\s*(引用来源|来源)\s*[:：].*$", "", answer)
    answer = re.sub(r"(?im)^\s*来源\s*\d+.*$", "", answer)
    answer = re.sub(r"\n{3,}", "\n\n", answer)
    return answer.strip()


def append_manual_service_notice(answer: str) -> str:
    if MANUAL_SERVICE_PHONE in answer:
        return answer
    return f"{answer}\n\n{MANUAL_SERVICE_NOTICE}"


def ask_assistant(question: str) -> str:
    result = rag_chat.answer_with_metadata(question, k=DEFAULT_TOP_K)
    answer = clean_customer_answer(result["answer"].strip())
    classification = result.get("classification", {})
    follow_up_questions = classification.get("follow_up_questions", [])

    if follow_up_questions:
        escaped_questions = "\n".join(f"- {html.escape(item)}" for item in follow_up_questions)
        answer = f"{answer}\n\n为了确认更准确，建议补充：\n{escaped_questions}"

    return append_manual_service_notice(answer)


def render_footer() -> None:
    st.markdown(
        """
        <div class="jl-footer-note">
            AI 助手回复仅用于售前沟通和初步排查，正式价格、库存、适配关系和售后方案以人工核实结果为准。
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    render_header()
    render_showcase()
    render_quick_questions()
    render_service_cards()
    init_chat_state()
    render_messages()

    pending_question = st.session_state.pop("pending_question", None)
    chat_question = st.chat_input("请输入您的配件或售后问题，例如：小松PC200液压泵多少钱？")
    question = pending_question or chat_question

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("正在为您查询配件资料..."):
                try:
                    answer = ask_assistant(question)
                except Exception:
                    answer = (
                        "抱歉，当前查询暂时没有成功。您可以稍后再试，"
                        f"也可以直接拨打 {MANUAL_SERVICE_PHONE} 联系人工客服。"
                    )
                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()

    render_footer()


if __name__ == "__main__":
    main()
