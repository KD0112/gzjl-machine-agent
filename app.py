import base64
import html
import re
from pathlib import Path

import streamlit as st

import rag_chat
from rag_history import RagHistoryRepository
from settings import RAG_STREAM_ENABLED


DEFAULT_TOP_K = 3
SERVICE_PHONE = "18750528881"
MANUAL_SERVICE_PHONE = "13608517353"
ASSET_DIR = Path(__file__).resolve().parent / "assets"
MANUAL_SERVICE_NOTICE = (
    f"如 AI 助手暂未完全解决您的问题，欢迎致电 {MANUAL_SERVICE_PHONE} 联系人工客服，"
    "我们将进一步为您核实底盘件型号、尺寸、库存、报价与售后方案。"
)

PART_IMAGES = [
    {
        "name": "履带链条 / 链轨总成",
        "desc": "核对机型、链节数、链节距、链板孔距与使用工况",
        "asset": "undercarriage_track_chain.webp",
    },
    {
        "name": "引导轮 / 张紧装置",
        "desc": "区分引导轮磨损、张紧油缸泄漏、链条拉长与掉链问题",
        "asset": "undercarriage_front_idler.webp",
    },
    {
        "name": "履带板 / 链板螺栓",
        "desc": "确认板宽、孔数、孔距、螺栓直径、牙距与强度等级",
        "asset": "undercarriage_track_shoes.webp",
    },
]

QUICK_QUESTIONS = [
    "托轮和支重轮有什么区别？",
    "链条总成询价需要提供什么？",
    "引导轮反复掉链怎么排查？",
]


st.set_page_config(
    page_title="劲龙机械底盘件智能助手",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded",
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
        background: #f4f5f1;
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
        border-radius: 8px;
        display: grid;
        place-items: center;
        background: var(--orange);
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
        border-radius: 8px;
        border: 1px solid rgba(213, 221, 207, 0.85);
        background: #e8ece5;
        min-height: 322px;
        padding: 22px 28px 26px;
        position: relative;
        overflow: hidden;
        box-shadow: var(--shadow);
        margin-bottom: 26px;
    }

    .jl-showcase:after {
        display: none;
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
        gap: 18px;
        position: relative;
        z-index: 1;
    }

    .jl-product {
        border-radius: 7px;
        background: #ffffff;
        border: 1px solid #bbc7b4;
        padding: 14px 14px 12px;
        text-align: left;
        box-shadow: 0 14px 26px rgba(31, 48, 38, 0.08);
        min-height: 226px;
    }

    .jl-product img {
        width: 100%;
        height: 150px;
        object-fit: contain;
        display: block;
        margin-bottom: 10px;
        border: 1px solid #e0e4de;
        border-radius: 6px;
        background: #f2f3f1;
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


def _asset_data_uri(filename: str) -> str:
    asset_root = ASSET_DIR.resolve()
    asset_path = (asset_root / filename).resolve()
    if asset_path.parent != asset_root or not asset_path.is_file():
        raise FileNotFoundError(f"页面图片不存在：{filename}")
    encoded = base64.b64encode(asset_path.read_bytes()).decode("ascii")
    return f"data:image/webp;base64,{encoded}"


def render_header() -> None:
    st.markdown(
        f"""
        <div class="jl-topbar">
            <div class="jl-brand">
                <div class="jl-logo">JL</div>
                <div>
                    <h1 class="jl-title">挖机底盘件智能助手</h1>
                    <div class="jl-subtitle">贵州劲龙机械 · 链条、四轮一带、销轴与紧固件咨询</div>
                </div>
            </div>
            <div class="jl-phone">联系电话：{SERVICE_PHONE}</div>
        </div>
        <div class="jl-chips">
            <span class="jl-chip">履带链条</span>
            <span class="jl-chip">支重轮 / 托链轮</span>
            <span class="jl-chip">引导轮</span>
            <span class="jl-chip">驱动齿</span>
            <span class="jl-chip">履带板</span>
            <span class="jl-chip">销轴与螺栓</span>
        </div>
        <div class="jl-rule"></div>
        """,
        unsafe_allow_html=True,
    )


def render_showcase() -> None:
    product_cards = "".join(
        f"""
        <div class="jl-product">
            <img src="{_asset_data_uri(item['asset'])}" alt="{html.escape(item['name'])}">
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
                <div class="jl-search">输入机型、底盘件名称、尺寸或磨损现象，助手会先核对适配信息</div>
            </div>
            <div class="jl-caption">
                <span>知识主题：底盘件与销轴紧固件 V1.0</span>
                <span>先核机型与尺寸，再确认适配</span>
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
                <div class="jl-service-title">机型尺寸先确认</div>
                <div class="jl-service-text">核对品牌、完整机型、链节距、孔距、板宽和安装位置。</div>
            </div>
            <div class="jl-service-card">
                <div class="jl-service-title">磨损需要联动看</div>
                <div class="jl-service-text">链条、驱动齿、引导轮和支重轮磨损可能相互影响。</div>
            </div>
            <div class="jl-service-card">
                <div class="jl-service-title">螺栓不能凭外观</div>
                <div class="jl-service-text">链板、支重轮和斗轴螺栓需确认直径、牙距、长度和等级。</div>
            </div>
            <div class="jl-service-card">
                <div class="jl-service-title">照片人工可复核</div>
                <div class="jl-service-text">型号不清、偏磨、掉链和复杂适配问题可转人工继续核实。</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def init_chat_state() -> None:
    if "current_user_id" not in st.session_state:
        st.session_state.current_user_id = "local_user"
    if "current_conversation_id" not in st.session_state:
        st.session_state.current_conversation_id = None


def render_conversation_sidebar(
    repository: RagHistoryRepository,
) -> tuple[str, str]:
    init_chat_state()
    with st.sidebar:
        st.subheader("我的会话")
        user_id = st.text_input(
            "用户标识",
            key="current_user_id",
            help="本地演示使用；正式登录后可替换为认证系统的用户 ID。",
        ).strip()
        if not user_id:
            st.warning("请输入用户标识。")
            st.stop()

        conversations = repository.list_conversations(user_id)
        known_ids = {item["conversation_id"] for item in conversations}
        current_id = st.session_state.current_conversation_id
        if current_id not in known_ids:
            if conversations:
                current_id = conversations[0]["conversation_id"]
            else:
                current_id = repository.create_conversation(user_id)["conversation_id"]
            st.session_state.current_conversation_id = current_id
            conversations = repository.list_conversations(user_id)

        if st.button("新建会话", use_container_width=True):
            created = repository.create_conversation(user_id)
            st.session_state.current_conversation_id = created["conversation_id"]
            st.rerun()

        option_ids = [item["conversation_id"] for item in conversations]
        title_by_id = {
            item["conversation_id"]: item["title"]
            for item in conversations
        }
        selected_id = st.selectbox(
            "切换会话",
            option_ids,
            index=option_ids.index(st.session_state.current_conversation_id),
            format_func=lambda value: title_by_id.get(value, "新会话"),
        )
        st.session_state.current_conversation_id = selected_id
        st.caption("消息保存在本地 SQLite，刷新或重启页面后仍可恢复。")
    return user_id, selected_id


def render_quick_questions() -> None:
    st.markdown('<div class="jl-section-title">常见问题</div>', unsafe_allow_html=True)
    cols = st.columns(len(QUICK_QUESTIONS))
    for col, example in zip(cols, QUICK_QUESTIONS):
        if col.button(example, use_container_width=True):
            st.session_state.pending_question = example
            st.rerun()


def _display_customer_answer(answer: str) -> str:
    return append_manual_service_notice(clean_customer_answer(answer))


def render_customer_citations(citations: list[dict[str, object]]) -> None:
    if not citations:
        return
    with st.expander("参考依据"):
        for citation in citations:
            location = citation.get("page_or_sheet") or citation.get("section") or ""
            suffix = f"（{location}）" if location else ""
            st.markdown(f"- {citation.get('document_name', 'unknown')}{suffix}")


def render_messages(
    repository: RagHistoryRepository,
    user_id: str,
    conversation_id: str,
) -> None:
    st.markdown('<div class="jl-section-title">智能对话</div>', unsafe_allow_html=True)
    messages = repository.list_messages(user_id, conversation_id)
    if not messages:
        with st.chat_message("assistant"):
            st.markdown(
                "您好，我是劲龙机械底盘件助手。您可以告诉我设备品牌、完整型号、"
                "链条或四轮一带名称、旧件尺寸和磨损现象，我会先帮您判断还需确认什么。"
            )
        return

    for message in messages:
        with st.chat_message(message["role"]):
            content = message["content"]
            if message["role"] == "assistant":
                content = _display_customer_answer(content)
            st.markdown(content)
            if message["role"] == "assistant":
                if message.get("cache_hit"):
                    st.caption("cache_hit：是")
                citations = repository.get_message_citations(
                    user_id,
                    conversation_id,
                    message["message_id"],
                )
                render_customer_citations(citations)


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


def ask_assistant(
    question: str,
    *,
    user_id: str,
    conversation_id: str,
) -> dict[str, object]:
    return rag_chat.answer_with_metadata(
        question,
        k=DEFAULT_TOP_K,
        user_id=user_id,
        conversation_id=conversation_id,
    )


def stream_assistant(
    question: str,
    *,
    user_id: str,
    conversation_id: str,
):
    return rag_chat.stream_answer_with_metadata(
        question,
        k=DEFAULT_TOP_K,
        user_id=user_id,
        conversation_id=conversation_id,
    )


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
    repository = RagHistoryRepository()
    user_id, conversation_id = render_conversation_sidebar(repository)
    render_header()
    render_showcase()
    render_quick_questions()
    render_service_cards()
    render_messages(repository, user_id, conversation_id)

    pending_question = st.session_state.pop("pending_question", None)
    chat_question = st.chat_input("请输入底盘件问题，例如：托轮和支重轮有什么区别？")
    question = pending_question or chat_question

    if question and not st.session_state.get("processing_request"):
        st.session_state.processing_request = True
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("正在为您查询底盘件资料..."):
                placeholder = st.empty()
                try:
                    final_result = None
                    if RAG_STREAM_ENABLED:
                        accumulated = ""
                        for event in stream_assistant(
                            question,
                            user_id=user_id,
                            conversation_id=conversation_id,
                        ):
                            if event["type"] == "delta":
                                accumulated += event["text"]
                                placeholder.markdown(
                                    f"{clean_customer_answer(accumulated)}▌"
                                )
                            elif event["type"] == "final":
                                final_result = event["result"]
                            elif event["type"] == "error":
                                placeholder.error(event["message"])
                    else:
                        final_result = ask_assistant(
                            question,
                            user_id=user_id,
                            conversation_id=conversation_id,
                        )

                    if final_result:
                        answer = _display_customer_answer(
                            str(final_result["answer"])
                        )
                        placeholder.markdown(answer)
                        if final_result.get("cache_hit"):
                            st.caption("cache_hit：是")
                        render_customer_citations(
                            list(final_result.get("citations") or [])
                        )
                except Exception:
                    placeholder.error(
                        "抱歉，当前查询暂时没有成功。您可以稍后再试，"
                        f"也可以直接拨打 {MANUAL_SERVICE_PHONE} 联系人工客服。"
                    )
                finally:
                    st.session_state.processing_request = False
        st.rerun()

    render_footer()


if __name__ == "__main__":
    main()
