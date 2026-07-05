from __future__ import annotations

from typing import Any

from brand_voice import ASSISTANT_NAME


FIELD_PROMPTS = {
    "machine_model": "设备型号",
    "part_name": "具体配件名称",
    "quality_level": "需要原厂、副厂还是经济型",
    "quantity": "数量大概几件",
    "city": "收货城市",
    "order_id": "订单号",
}


def _subject(slots: dict[str, Any]) -> str:
    quality_part = "".join(
        str(part) for part in [slots.get("quality_level"), slots.get("part_name")] if part
    )
    parts = [slots.get("brand"), slots.get("machine_model"), quality_part]
    return " ".join(str(part) for part in parts if part)


def _build_follow_up(parse_result: dict[str, Any]) -> str:
    missing_fields = parse_result["missing_fields"]
    missing_labels = [FIELD_PROMPTS.get(field, field) for field in missing_fields]
    if not missing_labels:
        return "我需要再确认一点信息，才能继续帮您查询。"

    return (
        "我可以继续帮您查库存、价格和发货时效。"
        f"为了避免型号或报价不准确，麻烦您再补充一下：{'、'.join(missing_labels)}。"
    )


def _inventory_sentence(inventory: dict[str, Any]) -> str:
    if inventory.get("in_stock"):
        return (
            f"库存这边先帮您看了一下，{inventory.get('warehouse')}目前有 "
            f"{inventory.get('stock_count')} 件可用库存。"
        )
    if inventory.get("matched"):
        return "库存这边找到了对应配件记录，但当前可用库存不足，需要人工再确认调货周期。"
    return "库存这边暂时没有查到完全匹配的记录，建议再核对一下型号、配件名称或品质档位。"


def _quote_sentence(quote: dict[str, Any]) -> str:
    if quote.get("matched"):
        return (
            f"价格先给您一个参考区间：{quote.get('total_price_range')} 元。"
            "这类配件价格会受库存批次、品质档位和发货方式影响，最终报价需要人工确认。"
        )
    return "价格这边暂时没有匹配到自动报价规则，需要人工根据实际库存和品质档位给您核价。"


def _logistics_sentence(logistics: dict[str, Any]) -> str:
    if logistics.get("matched"):
        return (
            f"发货方面，预计可以从{logistics.get('dispatch_warehouse')}发往"
            f"{logistics.get('city')}，一般走{logistics.get('delivery_method')}，"
            f"时效大约 {logistics.get('estimated_days')}，基础运费参考 "
            f"{logistics.get('freight_range')} 元。"
        )
    return "物流这边暂时没有匹配到该城市的自动规则，需要人工确认专线、运费和时效。"


def _ticket_sentence(ticket: dict[str, Any]) -> str:
    missing_items = ticket.get("missing_items") or []
    missing_text = "、".join(missing_items)
    extra = f"还需要补充{missing_text}。" if missing_text else "当前信息已可以先转售后核对。"
    return (
        f"售后工单已先帮您整理好，工单号 {ticket.get('ticket_id')}，"
        f"问题类型是{ticket.get('issue_type')}。{extra}"
        "人工客服会继续核对订单、配件状态和具体售后政策。"
    )


def _no_tool_guidance(parse_result: dict[str, Any]) -> str:
    intents = set(parse_result["intents"])
    slots = parse_result["slots"]
    subject = _subject(slots) or "这个配件"

    if "compatibility" in intents:
        return (
            f"{subject}能不能适配，建议按设备铭牌、旧件照片、零件号和安装位置再复核。"
            "这类判断我可以先帮您整理信息，但最终适配结论需要配件顾问人工确认。"
        )

    if "diagnosis" in intents:
        return (
            "故障排查我可以先帮您判断方向，但线上不能直接替代现场检测。"
            "建议补充故障视频、出现时间、是否刚维修或更换过配件，再转技术支持人工确认。"
        )

    if "general_consulting" in intents:
        return "我可以先帮您整理配件咨询信息；如果涉及准确型号、价格或库存，还需要人工进一步确认。"

    return "这个问题需要人工进一步核实，我可以先帮您整理需要补充的信息。"


def build_customer_reply(
    parse_result: dict[str, Any],
    tool_results: dict[str, Any],
    unsupported_tools: list[str],
) -> str:
    """Build a customer-facing reply without exposing internal tool/debug wording."""
    if parse_result["missing_fields"]:
        return _build_follow_up(parse_result)

    slots = parse_result["slots"]
    subject = _subject(slots)
    reply_parts: list[str] = []

    if subject:
        reply_parts.append(f"您好，这里是{ASSISTANT_NAME}，我先帮您查一下{subject}的情况。")
    else:
        reply_parts.append(f"您好，这里是{ASSISTANT_NAME}，我先帮您查一下相关配件信息。")

    inventory = tool_results.get("inventory_tool")
    if inventory:
        reply_parts.append(_inventory_sentence(inventory))

    quote = tool_results.get("quote_tool")
    if quote:
        reply_parts.append(_quote_sentence(quote))

    logistics = tool_results.get("logistics_tool")
    if logistics:
        reply_parts.append(_logistics_sentence(logistics))

    ticket = tool_results.get("ticket_tool")
    if ticket:
        reply_parts.append(_ticket_sentence(ticket))

    if "ticket_tool" in unsupported_tools:
        reply_parts.append("售后问题我建议转人工客服继续处理，这样可以核对订单、照片和具体售后政策。")

    if not tool_results:
        reply_parts.append(_no_tool_guidance(parse_result))

    if ticket and not any([inventory, quote, logistics]):
        reply_parts.append("如果您现在比较急，可以让人工客服优先核对这张售后工单。")
    else:
        reply_parts.append("如果您现在比较急，可以让人工客服帮您锁库存、确认最终价格和发货方式。")
    return "\n\n".join(reply_parts)
