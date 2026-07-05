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


def query_inventory(
    machine_model: str,
    part_name: str,
    quality_level: str | None = None,
    brand: str | None = None,
    data_path: Path = DATA_PATH,
) -> dict[str, Any]:
    """Query simulated inventory for excavator parts."""
    rows = _load_rows(data_path)
    candidates = [
        row
        for row in rows
        if _same_text(row.get("machine_model"), machine_model)
        and _match_part(row.get("part_name", ""), part_name)
        and _same_text(row.get("brand"), brand)
        and _same_text(row.get("quality_level"), quality_level)
    ]

    if not candidates:
        return {
            "matched": False,
            "in_stock": False,
            "stock_count": 0,
            "warehouse": None,
            "need_manual_confirm": True,
            "message": "暂未查询到匹配库存，建议转人工核实型号、配件名称和品质档位。",
            "options": [],
        }

    available = [row for row in candidates if row.get("status") == "available" and int(row.get("stock_count", 0)) > 0]
    options = [
        {
            "part_id": row["part_id"],
            "brand": row["brand"],
            "machine_model": row["machine_model"],
            "part_name": row["part_name"],
            "quality_level": row["quality_level"],
            "stock_count": int(row["stock_count"]),
            "warehouse": row["warehouse"],
            "status": row["status"],
        }
        for row in candidates
    ]

    if not available:
        return {
            "matched": True,
            "in_stock": False,
            "stock_count": 0,
            "warehouse": candidates[0].get("warehouse"),
            "need_manual_confirm": True,
            "message": "已找到匹配配件，但当前模拟库存为缺货，建议人工确认调货周期。",
            "options": options,
        }

    best = max(available, key=lambda row: int(row.get("stock_count", 0)))
    return {
        "matched": True,
        "in_stock": True,
        "stock_count": int(best["stock_count"]),
        "warehouse": best["warehouse"],
        "need_manual_confirm": True,
        "message": "查询到模拟库存，正式库存仍需人工确认。",
        "options": options,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Query simulated excavator part inventory.")
    parser.add_argument("--machine-model", required=True)
    parser.add_argument("--part-name", required=True)
    parser.add_argument("--quality-level")
    parser.add_argument("--brand")
    args = parser.parse_args()

    result = query_inventory(
        machine_model=args.machine_model,
        part_name=args.part_name,
        quality_level=args.quality_level,
        brand=args.brand,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
