from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "logistics_rules.csv"


def _load_rows(data_path: Path = DATA_PATH) -> list[dict[str, str]]:
    with data_path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _normalize_city(city: str) -> str:
    return city.strip().removesuffix("市")


def estimate_logistics(
    city: str,
    part_name: str,
    urgent: bool | None = None,
    data_path: Path = DATA_PATH,
) -> dict[str, Any]:
    """Estimate delivery method, time range, and freight for a target city."""
    normalized_city = _normalize_city(city)
    rows = _load_rows(data_path)
    match = next((row for row in rows if _normalize_city(row["city"]) == normalized_city), None)

    if not match:
        return {
            "matched": False,
            "city": city,
            "part_name": part_name,
            "delivery_method": "待人工确认",
            "estimated_days": None,
            "freight_range": None,
            "need_packaging_confirm": True,
            "need_manual_confirm": True,
            "message": "暂未匹配到该城市的物流规则，建议人工确认专线、运费和时效。",
        }

    estimated_days = match["urgent_days"] if urgent else match["normal_days"]
    return {
        "matched": True,
        "city": match["city"],
        "region": match["region"],
        "part_name": part_name,
        "dispatch_warehouse": match["dispatch_warehouse"],
        "delivery_method": match["delivery_method"],
        "estimated_days": estimated_days,
        "freight_range": f"{match['base_fee_min']}-{match['base_fee_max']}",
        "currency": "CNY",
        "urgent": bool(urgent),
        "need_packaging_confirm": True,
        "need_manual_confirm": True,
        "packaging_note": match["packaging_note"],
        "risk_notice": "物流时效和运费为模拟估算，正式发货需人工确认库存仓库、包装方式和收货地址。",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate simulated logistics for excavator parts.")
    parser.add_argument("--city", required=True)
    parser.add_argument("--part-name", required=True)
    parser.add_argument("--urgent", action="store_true")
    args = parser.parse_args()

    result = estimate_logistics(
        city=args.city,
        part_name=args.part_name,
        urgent=args.urgent,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
