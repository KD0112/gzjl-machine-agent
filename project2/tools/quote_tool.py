from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "inventory.csv"


def _load_rows(data_path: Path = DATA_PATH) -> list[dict[str, str]]:
    with data_path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _same_text(left: str | None, right: str | None) -> bool:
    if not right:
        return True
    return (left or "").strip().lower() == right.strip().lower()


def _match_part(row_value: str, query_value: str) -> bool:
    row_value = row_value.strip().lower()
    query_value = query_value.strip().lower()
    return row_value == query_value or row_value in query_value or query_value in row_value


def generate_quote(
    machine_model: str,
    part_name: str,
    quality_level: str,
    quantity: int = 1,
    brand: str | None = None,
    data_path: Path = DATA_PATH,
) -> dict[str, Any]:
    """Generate a draft quote from simulated price bands."""
    if quantity <= 0:
        return {
            "matched": False,
            "quote_status": "invalid",
            "message": "数量必须大于 0。",
            "need_manual_confirm": True,
        }

    rows = _load_rows(data_path)
    matches = [
        row
        for row in rows
        if _same_text(row.get("machine_model"), machine_model)
        and _match_part(row.get("part_name", ""), part_name)
        and _same_text(row.get("quality_level"), quality_level)
        and _same_text(row.get("brand"), brand)
    ]

    if not matches:
        return {
            "matched": False,
            "quote_status": "need_manual_quote",
            "price_range": None,
            "currency": "CNY",
            "need_manual_confirm": True,
            "message": "暂未找到匹配报价规则，建议人工根据型号、品质档位和库存情况报价。",
        }

    row = matches[0]
    unit_min = int(row["price_min"])
    unit_max = int(row["price_max"])
    total_min = unit_min * quantity
    total_max = unit_max * quantity
    in_stock = row.get("status") == "available" and int(row.get("stock_count", 0)) >= quantity

    return {
        "matched": True,
        "part_id": row["part_id"],
        "brand": row["brand"],
        "machine_model": row["machine_model"],
        "part_name": row["part_name"],
        "quality_level": row["quality_level"],
        "quantity": quantity,
        "unit_price_range": f"{unit_min}-{unit_max}",
        "total_price_range": f"{total_min}-{total_max}",
        "currency": "CNY",
        "quote_status": "draft",
        "in_stock_for_quantity": in_stock,
        "need_manual_confirm": True,
        "risk_notice": "该报价为模拟草稿，正式价格需人工确认库存、品质档位、税费和物流费用。",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a simulated excavator part quote.")
    parser.add_argument("--machine-model", required=True)
    parser.add_argument("--part-name", required=True)
    parser.add_argument("--quality-level", required=True)
    parser.add_argument("--quantity", type=int, default=1)
    parser.add_argument("--brand")
    args = parser.parse_args()

    result = generate_quote(
        machine_model=args.machine_model,
        part_name=args.part_name,
        quality_level=args.quality_level,
        quantity=args.quantity,
        brand=args.brand,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
