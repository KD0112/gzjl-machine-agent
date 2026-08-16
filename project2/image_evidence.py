from __future__ import annotations

import hashlib
import io
import os
import re
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

from PIL import Image, ImageFilter, ImageOps, ImageStat, UnidentifiedImageError


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_EVIDENCE_DIR = BASE_DIR / "logs" / "image_evidence"
DEFAULT_EVIDENCE_DB = BASE_DIR / "logs" / "image_evidence.sqlite3"
ALLOWED_FORMATS = {
    "JPEG": (".jpg", ".jpeg", "image/jpeg"),
    "PNG": (".png", "image/png"),
    "WEBP": (".webp", "image/webp"),
}
MIME_TO_FORMAT = {
    "image/jpeg": "JPEG",
    "image/png": "PNG",
    "image/webp": "WEBP",
}
SAFE_ID_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")


class ImageValidationError(ValueError):
    """Raised when uploaded bytes are not an allowed, safe decodable image."""


def _env_int(name: str, default: int, minimum: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return max(minimum, int(raw))
    except ValueError:
        return default


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _safe_filename(filename: str) -> str:
    normalized = SAFE_ID_PATTERN.sub("_", Path(filename).name).strip("._")
    return normalized[:120] or "uploaded-image"


def _customer_partition(customer_id: str) -> str:
    raw = customer_id.strip() or "anonymous"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


@dataclass(frozen=True)
class ImagePolicy:
    max_bytes: int = 8 * 1024 * 1024
    max_pixels: int = 20_000_000
    min_dimension: int = 96
    retention_hours: int = 24

    @classmethod
    def from_env(cls) -> "ImagePolicy":
        return cls(
            max_bytes=_env_int("AGENT_IMAGE_MAX_BYTES", 8 * 1024 * 1024, 1024),
            max_pixels=_env_int("AGENT_IMAGE_MAX_PIXELS", 20_000_000, 10_000),
            min_dimension=_env_int("AGENT_IMAGE_MIN_DIMENSION", 96, 32),
            retention_hours=_env_int("AGENT_IMAGE_RETENTION_HOURS", 24, 1),
        )


@dataclass(frozen=True)
class ValidatedImage:
    original_filename: str
    safe_filename: str
    image_format: str
    mime_type: str
    width: int
    height: int
    size_bytes: int
    sha256: str
    content: bytes
    local_quality: str
    quality_signals: tuple[str, ...]

    def public_metadata(self) -> dict[str, Any]:
        return {
            "original_filename": self.original_filename,
            "safe_filename": self.safe_filename,
            "image_format": self.image_format,
            "mime_type": self.mime_type,
            "width": self.width,
            "height": self.height,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "local_quality": self.local_quality,
            "quality_signals": list(self.quality_signals),
        }


def _quality_signals(image: Image.Image) -> tuple[str, tuple[str, ...]]:
    gray = ImageOps.grayscale(image)
    stats = ImageStat.Stat(gray)
    brightness = float(stats.mean[0])
    contrast = float(stats.stddev[0])
    edge_stats = ImageStat.Stat(gray.filter(ImageFilter.FIND_EDGES))
    edge_variance = float(edge_stats.var[0])

    signals: list[str] = []
    if min(image.size) < 480:
        signals.append("resolution_low")
    if brightness < 35:
        signals.append("too_dark")
    elif brightness > 225:
        signals.append("overexposed")
    if contrast < 18:
        signals.append("low_contrast")
    if edge_variance < 90:
        signals.append("possibly_blurry")

    severe = {"too_dark", "overexposed", "low_contrast"}
    if len(severe & set(signals)) >= 2:
        quality = "poor"
    elif signals:
        quality = "fair"
    else:
        quality = "good"
    return quality, tuple(signals)


def validate_image_upload(
    content: bytes,
    *,
    filename: str,
    claimed_mime_type: str = "",
    policy: ImagePolicy | None = None,
) -> ValidatedImage:
    active_policy = policy or ImagePolicy.from_env()
    if not content:
        raise ImageValidationError("图片内容为空。")
    if len(content) > active_policy.max_bytes:
        raise ImageValidationError(
            f"图片超过 {active_policy.max_bytes // (1024 * 1024)} MB 限制。"
        )

    safe_filename = _safe_filename(filename)
    extension = Path(safe_filename).suffix.lower()
    if extension not in {".jpg", ".jpeg", ".png", ".webp"}:
        raise ImageValidationError("只允许 JPG、PNG 或 WebP 文件。")
    normalized_claimed_mime = claimed_mime_type.lower().strip()
    if normalized_claimed_mime and normalized_claimed_mime not in MIME_TO_FORMAT:
        raise ImageValidationError("声明的 MIME 类型不是 JPG、PNG 或 WebP。")

    try:
        with Image.open(io.BytesIO(content)) as opened:
            detected_format = str(opened.format or "").upper()
            if detected_format not in ALLOWED_FORMATS:
                raise ImageValidationError("文件内容不是受支持的 JPG、PNG 或 WebP。")
            if normalized_claimed_mime and MIME_TO_FORMAT[normalized_claimed_mime] != detected_format:
                raise ImageValidationError("文件内容与声明的 MIME 类型不一致。")
            allowed_values = ALLOWED_FORMATS[detected_format]
            if extension not in allowed_values:
                raise ImageValidationError("文件扩展名与实际图片格式不一致。")
            if getattr(opened, "is_animated", False) or getattr(opened, "n_frames", 1) > 1:
                raise ImageValidationError("当前不接受动画图片。")
            width, height = opened.size
            if width < active_policy.min_dimension or height < active_policy.min_dimension:
                raise ImageValidationError(
                    f"图片尺寸过小，宽高至少为 {active_policy.min_dimension}px。"
                )
            if width * height > active_policy.max_pixels:
                raise ImageValidationError("图片总像素超过安全限制。")
            opened.load()
            normalized = ImageOps.exif_transpose(opened)
            if normalized.mode not in {"RGB", "RGBA"}:
                normalized = normalized.convert("RGBA" if "A" in normalized.getbands() else "RGB")
            quality, quality_signals = _quality_signals(normalized)

            output = io.BytesIO()
            if detected_format == "JPEG":
                normalized.convert("RGB").save(
                    output,
                    format="JPEG",
                    quality=92,
                    optimize=True,
                )
                mime_type = "image/jpeg"
            elif detected_format == "PNG":
                normalized.save(output, format="PNG", optimize=True)
                mime_type = "image/png"
            else:
                normalized.save(output, format="WEBP", quality=92, method=4)
                mime_type = "image/webp"
    except ImageValidationError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageValidationError("图片无法安全解码，文件可能损坏或格式伪装。") from exc

    sanitized = output.getvalue()
    return ValidatedImage(
        original_filename=filename,
        safe_filename=safe_filename,
        image_format=detected_format,
        mime_type=mime_type,
        width=width,
        height=height,
        size_bytes=len(sanitized),
        sha256=hashlib.sha256(sanitized).hexdigest(),
        content=sanitized,
        local_quality=quality,
        quality_signals=quality_signals,
    )


class ImageEvidenceRepository:
    """Store sanitized image bytes outside LangGraph State with customer isolation."""

    def __init__(
        self,
        db_path: Path | None = None,
        evidence_dir: Path | None = None,
        policy: ImagePolicy | None = None,
    ) -> None:
        self.db_path = db_path or DEFAULT_EVIDENCE_DB
        self.evidence_dir = evidence_dir or DEFAULT_EVIDENCE_DIR
        self.policy = policy or ImagePolicy.from_env()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(str(self.db_path), timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=30000")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS image_evidence (
                    evidence_id TEXT PRIMARY KEY,
                    customer_partition TEXT NOT NULL,
                    customer_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    original_filename TEXT NOT NULL,
                    storage_path TEXT NOT NULL,
                    mime_type TEXT NOT NULL,
                    image_format TEXT NOT NULL,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    sha256 TEXT NOT NULL,
                    local_quality TEXT NOT NULL,
                    quality_signals_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL
                )
                """
            )

    def store(
        self,
        image: ValidatedImage,
        *,
        customer_id: str,
        session_id: str,
    ) -> dict[str, Any]:
        import json

        evidence_id = f"img_{uuid.uuid4().hex}"
        partition = _customer_partition(customer_id)
        suffix = {
            "JPEG": ".jpg",
            "PNG": ".png",
            "WEBP": ".webp",
        }[image.image_format]
        partition_dir = self.evidence_dir / partition
        partition_dir.mkdir(parents=True, exist_ok=True)
        storage_path = partition_dir / f"{evidence_id}{suffix}"
        storage_path.write_bytes(image.content)
        created_at = _utc_now()
        expires_at = created_at + timedelta(hours=self.policy.retention_hours)

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO image_evidence (
                    evidence_id, customer_partition, customer_id, session_id,
                    original_filename, storage_path, mime_type, image_format,
                    width, height, size_bytes, sha256, local_quality,
                    quality_signals_json, status, created_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    evidence_id,
                    partition,
                    customer_id,
                    session_id,
                    image.original_filename,
                    str(storage_path),
                    image.mime_type,
                    image.image_format,
                    image.width,
                    image.height,
                    image.size_bytes,
                    image.sha256,
                    image.local_quality,
                    json.dumps(list(image.quality_signals), ensure_ascii=False),
                    "active",
                    created_at.isoformat(timespec="seconds"),
                    expires_at.isoformat(timespec="seconds"),
                ),
            )
        return self.get_metadata(evidence_id, customer_id=customer_id)

    def get_metadata(self, evidence_id: str, *, customer_id: str) -> dict[str, Any]:
        import json

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM image_evidence
                WHERE evidence_id = ? AND customer_partition = ? AND status = 'active'
                """,
                (evidence_id, _customer_partition(customer_id)),
            ).fetchone()
        if row is None:
            raise KeyError("图片证据不存在、已过期或不属于当前客户。")
        result = dict(row)
        result["quality_signals"] = json.loads(result.pop("quality_signals_json"))
        result.pop("customer_partition", None)
        result.pop("storage_path", None)
        return result

    def read_content(self, evidence_id: str, *, customer_id: str) -> tuple[bytes, dict[str, Any]]:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM image_evidence
                WHERE evidence_id = ? AND customer_partition = ? AND status = 'active'
                """,
                (evidence_id, _customer_partition(customer_id)),
            ).fetchone()
        if row is None:
            raise KeyError("图片证据不存在、已过期或不属于当前客户。")
        expires_at = datetime.fromisoformat(row["expires_at"])
        if expires_at <= _utc_now():
            self.delete(evidence_id, customer_id=customer_id)
            raise KeyError("图片证据已经过期。")
        path = Path(row["storage_path"])
        if not path.exists():
            raise KeyError("图片证据文件不存在。")
        return path.read_bytes(), self.get_metadata(evidence_id, customer_id=customer_id)

    def delete(self, evidence_id: str, *, customer_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT storage_path FROM image_evidence
                WHERE evidence_id = ? AND customer_partition = ?
                """,
                (evidence_id, _customer_partition(customer_id)),
            ).fetchone()
            if row is None:
                return False
            connection.execute(
                """
                UPDATE image_evidence SET status = 'deleted'
                WHERE evidence_id = ? AND customer_partition = ?
                """,
                (evidence_id, _customer_partition(customer_id)),
            )
        path = Path(row["storage_path"])
        if path.exists():
            path.unlink()
        return True

    def delete_expired(self) -> int:
        now = _utc_now().isoformat(timespec="seconds")
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT evidence_id, customer_id, storage_path
                FROM image_evidence
                WHERE status = 'active' AND expires_at <= ?
                """,
                (now,),
            ).fetchall()
            connection.execute(
                """
                UPDATE image_evidence SET status = 'expired'
                WHERE status = 'active' AND expires_at <= ?
                """,
                (now,),
            )
        for row in rows:
            path = Path(row["storage_path"])
            if path.exists():
                path.unlink()
        return len(rows)


DEFAULT_IMAGE_EVIDENCE_REPOSITORY = ImageEvidenceRepository()
