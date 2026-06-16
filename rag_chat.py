import re
from functools import lru_cache
from typing import Any

from openai import OpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from settings import (
    CHROMA_DB_DIR,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    EMBEDDING_MODEL,
    require_deepseek_key,
)


QUESTION_TYPES = {
    "part_quote": {
        "label": "配件询价",
        "style": "先判断是否能给报价依据，再说明缺少哪些询价信息，避免直接编价格。",
    },
    "part_match": {
        "label": "配件匹配",
        "style": "重点围绕设备型号、旧件照片、零件号、安装位置来确认是否适配。",
    },
    "fault_diagnosis": {
        "label": "故障诊断",
        "style": "给出可能原因和排查顺序，不把初步判断说成确定结论。",
    },
    "after_sales": {
        "label": "售后质保",
        "style": "先确认订单、购买时间、使用情况和问题证据，再说明可处理路径。",
    },
    "logistics": {
        "label": "物流交付",
        "style": "围绕收货地、是否急件、发货方式、包装和时效进行回答。",
    },
    "inventory_procurement": {
        "label": "库存采购",
        "style": "说明是否需要查现货、替代件、采购周期和客户可接受方案。",
    },
    "general": {
        "label": "综合咨询",
        "style": "基于知识库直接回答；如果资料不足，明确说明并追问。",
    },
}


TYPE_KEYWORDS = {
    "part_quote": (
        "多少钱",
        "价格",
        "报价",
        "询价",
        "采购",
        "购买",
        "买",
        "下单",
        "费用",
        "便宜",
    ),
    "part_match": (
        "配件",
        "零件",
        "型号",
        "件号",
        "零件号",
        "匹配",
        "适配",
        "通用",
        "替换",
        "安装位置",
        "尺寸",
        "铭牌",
        "照片",
        "发动机号",
        "能不能配",
    ),
    "fault_diagnosis": (
        "故障",
        "坏了",
        "异响",
        "漏油",
        "报警",
        "不动",
        "没力",
        "无力",
        "缺力",
        "动作慢",
        "黑烟",
        "跑偏",
        "油温高",
        "发热",
        "烧",
        "抖动",
        "卡滞",
        "维修",
        "诊断",
        "原因",
    ),
    "after_sales": (
        "售后",
        "退换",
        "退货",
        "换货",
        "退款",
        "质保",
        "保修",
        "三包",
        "质量问题",
    ),
    "logistics": (
        "发货",
        "物流",
        "快递",
        "到货",
        "多久",
        "运费",
        "送到",
        "地址",
        "包装",
        "时效",
    ),
    "inventory_procurement": (
        "库存",
        "现货",
        "缺货",
        "采购周期",
        "订货",
        "调货",
        "备货",
        "常备",
        "预定",
        "到仓",
        "供应商",
    ),
}


MACHINE_KEYWORDS = (
    "挖机",
    "挖掘机",
    "装载机",
    "推土机",
    "小松",
    "卡特",
    "日立",
    "三一",
    "徐工",
    "斗山",
    "神钢",
    "沃尔沃",
    "柳工",
    "临工",
    "山河",
    "现代",
    "龙工",
    "铭牌",
    "型号",
)

PART_KEYWORDS = (
    "液压泵",
    "主控阀",
    "行走马达",
    "回转马达",
    "中心接头",
    "回转支承",
    "液压油滤芯",
    "空气滤芯",
    "柴油滤芯",
    "破碎锤",
    "喷油器",
    "喷油泵",
    "电脑板",
    "显示屏",
    "泵",
    "阀",
    "马达",
    "油缸",
    "履带",
    "斗齿",
    "滤芯",
    "发动机",
    "电磁阀",
    "传感器",
    "销轴",
    "轴承",
    "齿轮",
    "密封",
    "油管",
    "链条",
    "喷油嘴",
    "配件",
    "零件",
    "件号",
    "零件号",
)

URGENCY_KEYWORDS = ("急", "今天", "明天", "尽快", "现货", "加急", "多久", "时效")
LOCATION_PATTERN = re.compile(r"[\u4e00-\u9fa5]{2,}(省|市|区|县|镇|街道)")
MODEL_PATTERN = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]{1,5}[- ]?\d{2,6}[A-Za-z0-9-]*(?![A-Za-z0-9])")


TICKET_OWNER_BY_TYPE = {
    "part_quote": "销售/配件顾问",
    "part_match": "销售/配件顾问",
    "fault_diagnosis": "技术支持/维修工程师",
    "after_sales": "售后专员",
    "logistics": "仓储物流",
    "inventory_procurement": "采购/仓库",
    "general": "客服一线",
}

NEXT_ACTION_BY_TYPE = {
    "part_quote": "补齐型号、零件号、照片、急用程度和配件档位后，再进入正式报价。",
    "part_match": "补齐设备型号、旧件照片、零件号或安装位置后，再确认是否适配。",
    "fault_diagnosis": "补充故障视频、设备型号和维修历史后，转技术支持做进一步判断。",
    "after_sales": "补充订单号、购买时间和问题证据后，按售后规则判断处理方式。",
    "logistics": "确认收货地、是否急件、包装和大件物流责任后，再给出交付方案。",
    "inventory_procurement": "确认配件型号、数量和客户可接受周期后，查询现货或安排采购。",
    "general": "先基于知识库回复；若资料不足，转人工补充业务信息。",
}

RISK_NOTICE_BY_TYPE = {
    "part_quote": "不能在型号、件号和库存未确认前承诺最终价格。",
    "part_match": "不能只凭一个机型判断适配，高风险件需以旧件号或铭牌为准。",
    "fault_diagnosis": "线上诊断只能给排查方向，不能替代现场检测或维修结论。",
    "after_sales": "售后处理需要订单和证据，已安装、通电、污染或损坏件风险更高。",
    "logistics": "大件和易损件需要提前确认包装、签收检查和破损责任。",
    "inventory_procurement": "现货和采购周期变化快，正式承诺前需要二次确认。",
    "general": "资料不足时需要明确说明，避免把经验判断说成企业承诺。",
}


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache(maxsize=1)
def get_deepseek_client() -> OpenAI:
    return OpenAI(
        api_key=require_deepseek_key(),
        base_url=DEEPSEEK_BASE_URL,
    )


@lru_cache(maxsize=1)
def load_vector_db() -> Chroma:
    if not CHROMA_DB_DIR.exists() or not any(CHROMA_DB_DIR.iterdir()):
        from build_index import build_index

        build_index()

    return Chroma(
        persist_directory=str(CHROMA_DB_DIR),
        embedding_function=get_embeddings(),
    )


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword.lower() in text.lower() for keyword in keywords)


def _score_question_type(question: str) -> dict[str, int]:
    scores: dict[str, int] = {}
    for question_type, keywords in TYPE_KEYWORDS.items():
        scores[question_type] = sum(1 for keyword in keywords if keyword in question)
    return scores


def _has_machine_info(question: str) -> bool:
    return _contains_any(question, MACHINE_KEYWORDS) or MODEL_PATTERN.search(question) is not None


def _has_part_info(question: str) -> bool:
    return _contains_any(question, PART_KEYWORDS)


def _has_location_info(question: str) -> bool:
    return LOCATION_PATTERN.search(question) is not None


def extract_machine_model(question: str) -> str:
    model_match = MODEL_PATTERN.search(question)
    if model_match:
        return model_match.group(0).upper().replace(" ", "")
    for keyword in MACHINE_KEYWORDS:
        if keyword in question and keyword not in {"铭牌", "型号"}:
            return keyword
    return "待补充"


def extract_parts(question: str) -> list[str]:
    parts = [keyword for keyword in PART_KEYWORDS if keyword in question]
    unique_parts = list(dict.fromkeys(parts))
    filtered_parts = [
        part
        for part in unique_parts
        if not any(part != other and part in other for other in unique_parts)
    ]
    return filtered_parts or ["待补充"]


def extract_location(question: str) -> str:
    match = LOCATION_PATTERN.search(question)
    return match.group(0) if match else "待补充"


def judge_priority(question: str, classification: dict[str, Any]) -> str:
    urgent_words = ("急", "今天", "明天", "尽快", "马上", "加急", "停工")
    if _contains_any(question, urgent_words):
        return "高"
    if classification["type"] in {"fault_diagnosis", "after_sales"}:
        return "中"
    return "普通"


def generate_ticket_draft(
    question: str,
    classification: dict[str, Any],
    docs: list[Any] | None = None,
) -> dict[str, Any]:
    """Create a structured customer-service ticket draft.

    This is not a persisted order/ticket system yet. It is the Agent's
    structured handoff output for a human salesperson, technician, warehouse
    worker, or after-sales specialist.
    """
    question_type = classification["type"]
    source_paths = []
    for doc in docs or []:
        source = doc.metadata.get("source", "unknown")
        if source not in source_paths:
            source_paths.append(source)

    missing_fields = classification["missing_fields"]
    risk_flags = [RISK_NOTICE_BY_TYPE[question_type]]
    if missing_fields:
        risk_flags.append("当前客户信息不完整，需要先追问，避免错误报价或错误判断。")

    return {
        "客户诉求": question,
        "问题类型": classification["label"],
        "问题类型代码": question_type,
        "建议处理人": TICKET_OWNER_BY_TYPE[question_type],
        "优先级": judge_priority(question, classification),
        "设备/机型": extract_machine_model(question),
        "涉及配件": extract_parts(question),
        "收货/服务地点": extract_location(question),
        "已知信息": {
            "分类依据": classification["reason"],
            "检索来源数量": len(source_paths),
        },
        "缺失信息": missing_fields or ["暂无明显缺失信息"],
        "建议追问": classification["follow_up_questions"] or ["暂无必须追问项；如知识库证据不足，需要转人工确认。"],
        "下一步动作": NEXT_ACTION_BY_TYPE[question_type],
        "风险提示": risk_flags,
        "证据来源": source_paths or ["暂无"],
    }


def classify_question(question: str) -> dict[str, Any]:
    """Classify a customer question before retrieval and generation.

    This rule-based classifier is intentionally simple: it is fast, stable,
    and easy to explain in an interview. Later it can be replaced by a model
    classifier or a fine-tuned intent detector.
    """
    scores = _score_question_type(question)
    best_type = max(scores, key=scores.get)
    if scores[best_type] == 0:
        best_type = "general"

    # Strong business intents should win over weak words like "买".
    if scores.get("after_sales", 0) > 0:
        best_type = "after_sales"
    elif scores.get("fault_diagnosis", 0) > 0:
        best_type = "fault_diagnosis"
    elif scores.get("logistics", 0) > 0:
        best_type = "logistics"
    elif scores.get("part_quote", 0) > 0:
        best_type = "part_quote"

    missing_fields: list[str] = []
    follow_up_questions: list[str] = []
    has_machine_info = _has_machine_info(question)
    has_part_info = _has_part_info(question)
    has_urgency_info = _contains_any(question, URGENCY_KEYWORDS)
    has_location_info = _has_location_info(question)

    if best_type in {"part_quote", "part_match", "fault_diagnosis"} and not has_machine_info:
        missing_fields.append("设备品牌、完整型号或铭牌照片")
        follow_up_questions.append("方便提供设备品牌、完整型号，或者直接发铭牌照片吗？")

    if best_type in {"part_quote", "part_match"} and not has_part_info:
        missing_fields.append("配件名称、旧件照片或零件号")
        follow_up_questions.append("需要确认的是哪个配件？最好提供旧件照片、零件号或安装位置。")

    if best_type == "part_quote" and not has_urgency_info:
        missing_fields.append("是否急用、是否要求原厂件/副厂件/经济型方案")
        follow_up_questions.append("这个件是否急用？更倾向原厂件、稳定替代件，还是经济型方案？")

    if best_type == "fault_diagnosis":
        missing_fields.append("故障现象照片/视频、出现时间、是否刚维修或更换过配件")
        follow_up_questions.append("故障是一直存在还是偶发？能否补充照片/视频，以及最近是否维修过相关部位？")

    if best_type == "after_sales":
        missing_fields.append("订单号、购买时间、问题照片/视频、是否安装使用")
        follow_up_questions.append("请补充订单号或购买时间，并提供问题照片/视频，方便判断售后处理方式。")

    if best_type == "logistics" and not has_location_info:
        missing_fields.append("收货城市、是否急件、货物大致体积/重量")
        follow_up_questions.append("请补充收货城市和是否急用，我再按物流时效给你更准确的建议。")

    if best_type == "inventory_procurement" and not has_part_info:
        missing_fields.append("配件名称、型号、零件号或照片")
        follow_up_questions.append("请补充要查库存的配件名称、型号、零件号或照片。")

    question_meta = QUESTION_TYPES[best_type]
    reason = "命中关键词：" + "、".join(
        keyword for keyword in TYPE_KEYWORDS.get(best_type, ()) if keyword in question
    )
    if best_type == "general":
        reason = "没有命中明确业务关键词，按综合咨询处理。"

    return {
        "type": best_type,
        "label": question_meta["label"],
        "reason": reason,
        "answer_style": question_meta["style"],
        "missing_fields": missing_fields,
        "follow_up_questions": follow_up_questions,
        "needs_follow_up": bool(missing_fields),
        "scores": scores,
    }


def retrieve(question: str, k: int = 3):
    """This is the retrieval step in RAG."""
    db = load_vector_db()
    return db.similarity_search(question, k=k)


def build_prompt(question: str, context: str, classification: dict[str, Any]) -> str:
    missing_text = "、".join(classification["missing_fields"]) or "暂无明显缺失信息"
    follow_up_text = "\n".join(f"- {item}" for item in classification["follow_up_questions"])
    if not follow_up_text:
        follow_up_text = "- 暂无必须追问项；如知识库证据不足，需要明确说明。"

    return f"""
你是工程机械配件企业的智能客服 Agent，负责面向客户回答挖机配件咨询问题。

当前问题类型：{classification["label"]}（{classification["type"]}）
分类原因：{classification["reason"]}
回答策略：{classification["answer_style"]}
当前缺失信息：{missing_text}
建议追问：
{follow_up_text}

请严格遵守：
1. 只能根据“给定资料”回答，不要编造知识库中没有的价格、库存、承诺或政策。
2. 如果资料中没有答案，要用客户听得懂的话说“暂未查询到明确记录”，然后给出下一步需要客户补充的信息。
3. 如果信息不足，不要直接拒绝；先给出基于资料能判断的部分，再主动追问。
4. 回答尽量像真实客服：简洁、可执行、能推动下一步沟通。
5. 不要向客户暴露“知识库”“给定资料”“Source”“RAG”“向量库”等技术词。
6. 不要使用“根据现有资料”“资料中显示”这类内部表达，直接给客户结论和下一步建议。

给定资料：
{context}

用户问题：
{question}

请用中文回答，建议结构：
- 初步判断
- 依据或处理建议
- 需要补充的信息
"""


def answer_with_metadata(question: str, k: int = 3) -> dict[str, Any]:
    classification = classify_question(question)
    docs = retrieve(question, k=k)
    context = "\n\n".join(
        f"[Source {i + 1}: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for i, doc in enumerate(docs)
    )
    prompt = build_prompt(question, context, classification)

    response = get_deepseek_client().chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {
                "role": "system",
                "content": "你是一个严谨、不会编造的企业智能客服 Agent。",
            },
            {"role": "user", "content": prompt},
        ],
    )

    return {
        "answer": response.choices[0].message.content,
        "docs": docs,
        "classification": classification,
        "ticket_draft": generate_ticket_draft(question, classification, docs),
        "context": context,
    }


def answer(question: str, k: int = 3):
    """Backward-compatible wrapper for the Day 1 app and tests."""
    result = answer_with_metadata(question, k=k)
    return result["answer"], result["docs"]


def main() -> None:
    print("RAG chatbot is ready. Type q to quit.")
    while True:
        question = input("\n请输入问题：").strip()
        if question.lower() in {"q", "quit", "exit"}:
            break
        result = answer_with_metadata(question)
        print(f"\n问题类型：{result['classification']['label']}")
        if result["classification"]["needs_follow_up"]:
            print("建议追问：")
            for item in result["classification"]["follow_up_questions"]:
                print("-", item)
        print("\n工单草稿：")
        for key, value in result["ticket_draft"].items():
            print(f"{key}: {value}")
        print("\n回答：")
        print(result["answer"])
        print("\n检索到的来源：")
        for doc in result["docs"]:
            print("-", doc.metadata.get("source", "unknown"))


if __name__ == "__main__":
    main()
