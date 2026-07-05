from __future__ import annotations

import argparse
import json
import re
from typing import Any


BRAND_ALIASES = {
    "小松": ["小松", "komatsu"],
    "卡特": ["卡特", "卡特彼勒", "cat", "caterpillar"],
    "三一": ["三一", "sany"],
    "日立": ["日立", "hitachi"],
}

PART_ALIASES = {
    "液压泵": ["液压泵", "主泵", "泵总成"],
    "喷油嘴": ["喷油嘴", "油嘴"],
    "斗齿": ["斗齿", "齿尖"],
    "回油滤芯": ["回油滤芯", "滤芯", "液压滤芯"],
    "行走马达": ["行走马达", "行走电机"],
    "主控阀": ["主控阀", "多路阀"],
}

QUALITY_ALIASES = {
    "原厂": ["原厂", "原装", "正厂"],
    "副厂": ["副厂", "替代", "国产"],
    "经济型": ["经济型", "便宜点", "便宜的", "实惠点"],
}

CITY_ALIASES = [
    "贵阳",
    "遵义",
    "六盘水",
    "安顺",
    "毕节",
    "铜仁",
    "广州",
    "深圳",
    "长沙",
    "成都",
    "重庆",
    "昆明",
]

INTENT_KEYWORDS = {
    "inventory": ["现货", "有货", "库存", "仓库", "缺货", "到货"],
    "quote": ["多少钱", "价格", "报价", "什么价", "几钱", "价位", "费用"],
    "logistics": ["发货", "发到", "物流", "快递", "运费", "多久", "几天", "时效"],
    "after_sales": ["退货", "换货", "退款", "质保", "保修", "售后", "买错", "订单"],
    "compatibility": ["适配", "匹配", "能不能配", "能配", "通用", "发动机号", "零件号"],
    "diagnosis": ["异响", "漏油", "故障", "排查", "不工作", "没力", "高温", "抖动"],
}

INTENT_TOOLS = {
    "inventory": "inventory_tool",
    "quote": "quote_tool",
    "logistics": "logistics_tool",
    "after_sales": "ticket_tool",
}

REQUIRED_FIELDS_BY_INTENT = {
    "inventory": ["machine_model", "part_name", "quality_level"],
    "quote": ["machine_model", "part_name", "quality_level", "quantity"],
    "logistics": ["city", "part_name"],
    "after_sales": ["order_id"],
    "compatibility": ["machine_model", "part_name"],
    "diagnosis": ["machine_model"],
}

FIELD_LABELS = {
    "machine_model": "设备型号",
    "part_name": "配件名称",
    "quality_level": "品质档位",
    "quantity": "数量",
    "city": "收货城市",
    "order_id": "订单号",
}

CHINESE_NUMBERS = {
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}


def _normalize_text(text: str) -> str:
    return text.strip().lower().replace(" ", "")


def _find_by_alias(text: str, aliases: dict[str, list[str]]) -> str | None:
    normalized = _normalize_text(text)
    for value, words in aliases.items():
        if any(_normalize_text(word) in normalized for word in words):
            return value
    return None


def _extract_machine_model(text: str) -> str | None:
    match = re.search(r"\b([A-Za-z]{1,5}\s?-?\d{2,4}[A-Za-z]?)\b", text)
    if match:
        return match.group(1).replace(" ", "").upper()

    # Handle common Chinese-adjacent model strings such as "PC200液压泵".
    match = re.search(r"([A-Za-z]{1,5}\d{2,4}[A-Za-z]?)(?=[\u4e00-\u9fff])", text)
    if match:
        return match.group(1).upper()

    # Handle brand-adjacent numeric models such as "卡特320D".
    match = re.search(r"(小松|卡特|卡特彼勒|三一|日立)\s*(\d{2,4}[A-Za-z]?)", text)
    if match:
        return match.group(2).upper()
    return None


def _extract_quantity(text: str) -> int | None:
    match = re.search(r"(\d+)\s*(个|件|套|只|支|台)", text)
    if match:
        return int(match.group(1))

    match = re.search(r"([一二两三四五六七八九十])\s*(个|件|套|只|支|台)", text)
    if match:
        return CHINESE_NUMBERS.get(match.group(1))
    return None


def _extract_city(text: str) -> str | None:
    normalized = _normalize_text(text)
    for city in CITY_ALIASES:
        if _normalize_text(city) in normalized or _normalize_text(f"{city}市") in normalized:
            return city

    match = re.search(r"发到([\u4e00-\u9fff]{2,6})市?", text)
    if match:
        return match.group(1)
    return None


def _extract_order_id(text: str) -> str | None:
    match = re.search(r"(订单号|订单|单号)[:：\s]*([A-Za-z0-9-]{4,})", text)
    if match:
        return match.group(2).upper()
    return None


def _extract_urgent(text: str) -> bool | None:
    normalized = _normalize_text(text)
    if any(word in normalized for word in ["急用", "加急", "今天要", "马上要", "尽快"]):
        return True
    if any(word in normalized for word in ["不急", "不着急"]):
        return False
    return None


def classify_intents(text: str) -> tuple[list[str], dict[str, list[str]]]:
    normalized = _normalize_text(text)
    matched_keywords: dict[str, list[str]] = {}

    for intent, keywords in INTENT_KEYWORDS.items():
        hits = [word for word in keywords if _normalize_text(word) in normalized]
        if hits:
            matched_keywords[intent] = hits

    if not matched_keywords and _find_by_alias(text, PART_ALIASES):
        matched_keywords["general_consulting"] = ["配件咨询"]

    ordered_intents = [
        intent
        for intent in [
            "after_sales",
            "diagnosis",
            "compatibility",
            "inventory",
            "quote",
            "logistics",
            "general_consulting",
        ]
        if intent in matched_keywords
    ]
    return ordered_intents, matched_keywords


def extract_slots(text: str) -> dict[str, Any]:
    return {
        "brand": _find_by_alias(text, BRAND_ALIASES),
        "machine_model": _extract_machine_model(text),
        "part_name": _find_by_alias(text, PART_ALIASES),
        "quality_level": _find_by_alias(text, QUALITY_ALIASES),
        "quantity": _extract_quantity(text),
        "city": _extract_city(text),
        "urgent": _extract_urgent(text),
        "order_id": _extract_order_id(text),
    }


def find_missing_fields(intents: list[str], slots: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    for intent in intents:
        for field in REQUIRED_FIELDS_BY_INTENT.get(intent, []):
            if not slots.get(field) and field not in missing:
                missing.append(field)
    return missing


def build_follow_up(missing_fields: list[str]) -> str | None:
    if not missing_fields:
        return None

    parts: list[str] = []
    if "machine_model" in missing_fields:
        parts.append("设备型号")
    if "part_name" in missing_fields:
        parts.append("具体配件名称")
    if "quality_level" in missing_fields:
        parts.append("需要原厂、副厂还是经济型")
    if "quantity" in missing_fields:
        parts.append("数量大概几件")
    if "city" in missing_fields:
        parts.append("收货城市")
    if "order_id" in missing_fields:
        parts.append("订单号")

    if not parts:
        readable = "、".join(FIELD_LABELS.get(field, field) for field in missing_fields)
        return f"为了继续处理，请您补充：{readable}。"

    return "为了帮您准确查询，请您补充：" + "、".join(parts) + "。"


def parse_customer_question(text: str) -> dict[str, Any]:
    intents, matched_keywords = classify_intents(text)
    slots = extract_slots(text)
    missing_fields = find_missing_fields(intents, slots)
    candidate_tools = [INTENT_TOOLS[intent] for intent in intents if intent in INTENT_TOOLS]

    return {
        "raw_question": text,
        "intents": intents,
        "slots": slots,
        "missing_fields": missing_fields,
        "ready_for_tools": bool(candidate_tools) and not missing_fields,
        "candidate_tools": candidate_tools,
        "follow_up": build_follow_up(missing_fields),
        "debug": {
            "parser": "rule_based_v1",
            "matched_keywords": matched_keywords,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse customer question into intents and slots.")
    parser.add_argument("question")
    args = parser.parse_args()

    result = parse_customer_question(args.question)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
