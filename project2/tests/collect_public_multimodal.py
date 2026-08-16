from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps


BASE_DIR = Path(__file__).resolve().parents[1]
FIXTURE_DIR = BASE_DIR / "tests" / "fixtures" / "multimodal_real"
SANITIZED_DIR = FIXTURE_DIR / "sanitized"
MANIFEST_PATH = BASE_DIR / "tests" / "multimodal_real_candidates.jsonl"
ATTRIBUTION_PATH = BASE_DIR / "tests" / "multimodal_real_attribution.csv"
ANNOTATION_TEMPLATE_PATH = (
    BASE_DIR / "tests" / "multimodal_real_annotation_template.csv"
)
CONTACT_SHEET_PATH = BASE_DIR / "reports" / "multimodal_real_candidates_contact_sheet.jpg"
USER_AGENT = "ExcavatorAgentEvaluation/1.0 (educational portfolio project)"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
ALLOWED_LICENSE_PREFIXES = (
    "cc0",
    "cc by",
    "public domain",
    "pdm",
)

CURATED_SOURCES = [
    ("nameplate", "File:Maker's nameplate on Goliath - geograph.org.uk - 1552318.jpg"),
    ("nameplate", "File:Firefighter and Ladder Factory nameplate, 2017 Fehérgyarmat.jpg"),
    ("nameplate", "File:Ex-Cell-O Corp brass nameplate.jpg"),
    ("old_plate", "File:Worksplate (3161220528).jpg"),
    ("old_plate", "File:GNR Worksplate (12416942653).jpg"),
    ("old_plate", "File:Vulcan Foundry Worksplate (12418528813).jpg"),
    ("old_plate", "File:LMS worksplate (21866170936).jpg"),
    ("old_plate", "File:AEG Worksplate (28314359018).jpg"),
    ("old_plate", "File:Linke-Hoffmann Worksplate (38247105644).jpg"),
    ("old_plate", "File:Vulcan Foundry Worksplate (49580054303).jpg"),
    ("old_plate", "File:Neilson worksplate at Matjiesfontein.jpg"),
    ("old_plate", "File:Bagnall worksplate on No.6 Dunlop.jpg"),
    ("old_plate", "File:Fletcher Jennings worksplate - 55123480000.jpg"),
    ("serial_plate", "File:Rollin - serial plate 5919.jpg"),
    ("serial_plate", "File:RX-7 spirit R serial plate.jpg"),
    ("serial_plate", "File:Serial plate of an Tohatsu MFS9.8A3 outboard motor.jpg"),
    ("serial_plate", "File:Boeing MAX 737-8200 serial plate.tif"),
    (
        "serial_plate",
        "File:Silk City Diner serial plate 6027 - Airport Diner Kutztown PA.jpg",
    ),
    (
        "serial_plate",
        "File:Silk City Diner serial plate 5904 - Route 30 Diner Ronks PA.jpg",
    ),
    (
        "serial_plate",
        "File:Silk City Diner Serial Plate, Roadside Diner, Wall, NJ.jpg",
    ),
    ("serial_plate", "File:The Scratch Kitchen Serial Plate 5607.jpg"),
    ("manufacturer_plate", "File:KD7-534 manufacturer plate.JPG"),
    ("manufacturer_plate", "File:22120 WDP 3A - Manufacturer plate.jpg"),
    (
        "manufacturer_plate",
        "File:VEB Brauerei Cottbus (old electrical cabinets, manufacturer plate).png",
    ),
    (
        "manufacturer_plate",
        "File:VEB Brauerei Cottbus (old electrical cabinets, manufacturer plate 2).png",
    ),
    (
        "manufacturer_plate",
        "File:TransSibirianRail RossijaExpress manufacturer plate in coach 1981.png",
    ),
    (
        "manufacturer_plate",
        "File:Custom Coaches manufacturer plate on former Harris Park Transport "
        "mo7491 December 2024.jpg",
    ),
    (
        "manufacturer_plate",
        "File:Manufacturer plate on the 10 inch refractor telescope at Old "
        "Observatory, Leiden, on 14 september 2025.jpg",
    ),
    (
        "manufacturer_plate",
        "File:Silk City Diner manufacturer plate - Airport Diner Kutztown PA.jpg",
    ),
    (
        "hydraulic_part",
        "File:Hydraulic pump, Cragside - geograph.org.uk - 3592568.jpg",
    ),
    (
        "hydraulic_part",
        "File:Hydraulic pump for Cross Keys Swing Bridge - geograph.org.uk - "
        "4359788.jpg",
    ),
    (
        "hydraulic_part",
        "File:Smaller hydraulic pump, Avesta Cyclops works - geograph.org.uk - "
        "1689042.jpg",
    ),
    ("hydraulic_part", "File:Hydraulic pump.png"),
    ("damage", "File:Crompton Parkinson rusty nameplate.jpg"),
    ("damage", "File:N1944A Oil Leak.jpg"),
    ("damage", "File:Leaking Oil Drum Talpiot 01.jpg"),
    (
        "damage",
        "File:Leaking Oil Drum Talpiot 02.jpg",
    ),
    (
        "damage",
        "File:Oil Leak in the Snow - geograph.org.uk - 6736926.jpg",
    ),
    (
        "damage",
        "File:23 0051581 Convair Negative Image - Little Joe II hydraulic "
        "control system accumulator damaged o-ring 01-24-1964 (53889741694).jpg",
    ),
    ("damage", "File:Crimped hose ends.JPG"),
]

CHALLENGE_TAG_OVERRIDES = {
    "old_plate": ["old_worn", "metal_plate"],
    "serial_plate": ["serial_text", "metal_plate"],
    "manufacturer_plate": ["manufacturer_text", "mixed_distance"],
    "hydraulic_part": ["clutter", "no_guaranteed_readable_label"],
    "damage": ["damage_evidence", "ambiguous_severity"],
}

TITLE_TAG_OVERRIDES = {
    "File:RX-7 spirit R serial plate.jpg": ["glare", "reflection"],
    "File:Crompton Parkinson rusty nameplate.jpg": ["corrosion", "low_contrast"],
    "File:Leaking Oil Drum Talpiot 01.jpg": ["oil_leak"],
    "File:Leaking Oil Drum Talpiot 02.jpg": ["oil_leak"],
    "File:Oil Leak in the Snow - geograph.org.uk - 6736926.jpg": [
        "oil_leak",
        "low_contrast",
    ],
    "File:23 0051581 Convair Negative Image - Little Joe II hydraulic "
    "control system accumulator damaged o-ring 01-24-1964 (53889741694).jpg": [
        "damaged_seal",
        "grayscale",
    ],
    "File:Crimped hose ends.JPG": ["close_up", "ambiguous_damage"],
}


def _request_json(params: dict[str, Any], *, retries: int = 4) -> dict[str, Any]:
    query = urllib.parse.urlencode(params)
    request = urllib.request.Request(
        f"{COMMONS_API}?{query}",
        headers={"User-Agent": USER_AGENT},
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                return json.load(response)
        except Exception:
            if attempt + 1 == retries:
                raise
            time.sleep(1.5 * (2**attempt))
    raise RuntimeError("unreachable")


def _clean_html(value: str | None) -> str:
    if not value:
        return ""
    return html.unescape(re.sub(r"<[^>]+>", "", value)).strip()


def _safe_text(metadata: dict[str, Any], key: str) -> str:
    entry = metadata.get(key) or {}
    return _clean_html(str(entry.get("value") or ""))


def _page_to_candidate(
    page: dict[str, Any],
    *,
    scenario: str,
) -> dict[str, Any] | None:
    info = (page.get("imageinfo") or [{}])[0]
    metadata = info.get("extmetadata") or {}
    license_name = _safe_text(metadata, "LicenseShortName")
    mime_type = str(info.get("mime") or "")
    if not mime_type.startswith("image/"):
        return None
    if not license_name.casefold().startswith(ALLOWED_LICENSE_PREFIXES):
        return None
    return {
        "scenario": scenario,
        "query": f"curated:{page.get('title', '')}",
        "title": page.get("title", ""),
        "source_url": info.get("url", ""),
        "download_url": info.get("thumburl") or info.get("url", ""),
        "landing_url": info.get("descriptionurl", ""),
        "source_width": info.get("width"),
        "source_height": info.get("height"),
        "source_mime_type": mime_type,
        "creator": _safe_text(metadata, "Artist"),
        "credit": _safe_text(metadata, "Credit"),
        "license": license_name,
        "license_url": _safe_text(metadata, "LicenseUrl"),
        "attribution_required": license_name.casefold().startswith("cc by"),
    }


def _query_curated_sources() -> list[dict[str, Any]]:
    scenario_by_title = {
        title.casefold(): scenario for scenario, title in CURATED_SOURCES
    }
    rows_by_title: dict[str, dict[str, Any]] = {}
    batch_size = 20
    for offset in range(0, len(CURATED_SOURCES), batch_size):
        batch = CURATED_SOURCES[offset : offset + batch_size]
        payload = _request_json(
            {
                "action": "query",
                "titles": "|".join(title for _, title in batch),
                "prop": "imageinfo",
                "iiprop": "url|size|mime|extmetadata",
                "iiurlwidth": 1600,
                "format": "json",
                "formatversion": 2,
                "origin": "*",
            }
        )
        for page in payload.get("query", {}).get("pages", []):
            title_key = str(page.get("title") or "").casefold()
            scenario = scenario_by_title.get(title_key)
            if not scenario:
                continue
            row = _page_to_candidate(page, scenario=scenario)
            if row:
                rows_by_title[title_key] = row
        time.sleep(0.8)

    rows = []
    for scenario, title in CURATED_SOURCES:
        row = rows_by_title.get(title.casefold())
        if row:
            rows.append(row)
    return rows


def _download(url: str, *, retries: int = 4) -> bytes:
    for attempt in range(retries):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except Exception:
            if attempt + 1 == retries:
                raise
            time.sleep(1.5 * (2**attempt))
    raise RuntimeError("unreachable")


def _sanitize_image(content: bytes, output_path: Path) -> dict[str, Any]:
    with Image.open(io.BytesIO(content)) as source:
        image = ImageOps.exif_transpose(source)
        if image.mode not in {"RGB", "L"}:
            background = Image.new("RGB", image.size, "white")
            if "A" in image.getbands():
                background.paste(image, mask=image.getchannel("A"))
            else:
                background.paste(image.convert("RGB"))
            image = background
        else:
            image = image.convert("RGB")
        image.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(
            output_path,
            format="JPEG",
            quality=90,
            optimize=True,
            progressive=True,
        )
        width, height = image.size
    output_bytes = output_path.read_bytes()
    return {
        "sha256": hashlib.sha256(output_bytes).hexdigest(),
        "width": width,
        "height": height,
        "bytes": len(output_bytes),
        "exif_removed": True,
    }


def _existing_image_metadata(output_path: Path) -> dict[str, Any]:
    content = output_path.read_bytes()
    with Image.open(io.BytesIO(content)) as image:
        width, height = image.size
    return {
        "sha256": hashlib.sha256(content).hexdigest(),
        "width": width,
        "height": height,
        "bytes": len(content),
        "exif_removed": True,
    }


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return cleaned[:45] or "image"


def _candidate_id(index: int, row: dict[str, Any]) -> str:
    digest = hashlib.sha1(row["landing_url"].encode("utf-8")).hexdigest()[:8]
    return f"real-{index:03d}-{_slug(row['scenario'])}-{digest}"


def _write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_attribution(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "id",
        "title",
        "creator",
        "license",
        "license_url",
        "landing_url",
        "source_url",
        "sha256",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def _write_annotation_template(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "id",
        "image",
        "description",
        "scenario",
        "challenge_tags",
        "landing_url",
        "license",
        "reviewer_a",
        "reviewer_b",
        "adjudicator",
        "license_manual_verified",
        "privacy_approved",
        "expected_image_types",
        "brand_visibility",
        "expected_brand_any",
        "machine_model_visibility",
        "expected_machine_model_any",
        "part_name_candidate_visibility",
        "expected_part_name_any",
        "part_number_visibility",
        "expected_part_number_any",
        "expected_damage_keywords",
        "should_reject",
        "annotation_notes",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "id": row["id"],
                    "image": row["image"],
                    "description": row["description"],
                    "scenario": row["scenario"],
                    "challenge_tags": "|".join(row["challenge_tags"]),
                    "landing_url": row["landing_url"],
                    "license": row["license"],
                    "brand_visibility": "unreviewed",
                    "machine_model_visibility": "unreviewed",
                    "part_name_candidate_visibility": "unreviewed",
                    "part_number_visibility": "unreviewed",
                }
            )


def _make_contact_sheet(rows: list[dict[str, Any]], output_path: Path) -> None:
    tile_width = 320
    tile_height = 250
    columns = 4
    rows_count = (len(rows) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_width, rows_count * tile_height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, row in enumerate(rows):
        x = (index % columns) * tile_width
        y = (index // columns) * tile_height
        image_path = BASE_DIR / row["image"]
        with Image.open(image_path) as source:
            image = source.convert("RGB")
            image.thumbnail((tile_width - 16, 190), Image.Resampling.LANCZOS)
            image_x = x + (tile_width - image.width) // 2
            sheet.paste(image, (image_x, y + 8))
        label = f"{row['id']}\n{row['scenario']} | {row['license']}"
        draw.multiline_text((x + 8, y + 202), label, fill="black", font=font, spacing=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format="JPEG", quality=88, optimize=True)


def collect(target: int) -> list[dict[str, Any]]:
    candidates = _query_curated_sources()
    if len(candidates) < target:
        raise RuntimeError(
            f"Only {len(candidates)} unique candidates were found; target was {target}."
        )

    collected = []
    for row in candidates:
        index = len(collected) + 1
        candidate_id = _candidate_id(index, row)
        output_path = SANITIZED_DIR / f"{candidate_id}.jpg"
        try:
            if output_path.exists():
                image_metadata = _existing_image_metadata(output_path)
            else:
                content = _download(row["download_url"])
                image_metadata = _sanitize_image(content, output_path)
        except Exception as exc:
            print(f"Skipped download failure: {row['title']} ({type(exc).__name__})")
            continue
        challenge_tags = [
            row["scenario"],
            "real_photo",
            *CHALLENGE_TAG_OVERRIDES.get(row["scenario"], []),
            *TITLE_TAG_OVERRIDES.get(row["title"], []),
        ]
        collected.append(
            {
                "id": candidate_id,
                "description": row["title"].removeprefix("File:"),
                "image": output_path.relative_to(BASE_DIR).as_posix(),
                "data_origin": "public_open_license",
                "domain_match": "machinery_transfer",
                "scenario": row["scenario"],
                "challenge_tags": list(dict.fromkeys(challenge_tags)),
                "evaluation_status": "candidate",
                "license_review_status": "api_metadata_verified",
                "privacy_review_status": "exif_removed_manual_review_pending",
                "annotation_version": 1,
                "annotator_a": "",
                "annotator_b": "",
                "adjudicated_by": "",
                "field_visibility": {
                    "brand": "unreviewed",
                    "machine_model": "unreviewed",
                    "part_name_candidate": "unreviewed",
                    "part_number": "unreviewed",
                },
                "expected_image_types": [],
                "expected_brand_any": [],
                "expected_machine_model_any": [],
                "expected_part_name_any": [],
                "expected_part_number_any": [],
                "expected_damage_keywords": [],
                "should_reject": None,
                "source_query": row["query"],
                "title": row["title"],
                "creator": row["creator"],
                "credit": row["credit"],
                "license": row["license"],
                "license_url": row["license_url"],
                "attribution_required": row["attribution_required"],
                "landing_url": row["landing_url"],
                "source_url": row["source_url"],
                "source_width": row["source_width"],
                "source_height": row["source_height"],
                "source_mime_type": row["source_mime_type"],
                "collected_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                **image_metadata,
            }
        )
        if len(collected) >= target:
            break
    if len(collected) < target:
        raise RuntimeError(
            f"Only {len(collected)} candidates downloaded successfully; target was {target}."
        )
    return collected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=40)
    args = parser.parse_args()
    if args.target < 30 or args.target > 50:
        raise SystemExit("--target must be between 30 and 50")

    rows = collect(args.target)
    _write_jsonl(rows, MANIFEST_PATH)
    _write_attribution(rows, ATTRIBUTION_PATH)
    _write_annotation_template(rows, ANNOTATION_TEMPLATE_PATH)
    _make_contact_sheet(rows, CONTACT_SHEET_PATH)
    print(f"Collected {len(rows)} public real-image candidates.")
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Attribution: {ATTRIBUTION_PATH}")
    print(f"Annotation template: {ANNOTATION_TEMPLATE_PATH}")
    print(f"Contact sheet: {CONTACT_SHEET_PATH}")
    print("Status: candidate only; manual privacy review and two-person labels required.")


if __name__ == "__main__":
    main()
