import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent


def _streamlit_secret(name: str) -> str:
    try:
        import streamlit as st

        if name in st.secrets:
            return str(st.secrets[name]).strip()
    except Exception:
        return ""
    return ""


def _setting(name: str, default: str = "") -> str:
    return (os.getenv(name, "").strip() or _streamlit_secret(name) or default).strip()


def _int_setting(name: str, default: int) -> int:
    raw_value = _setting(name, str(default))
    try:
        return int(raw_value)
    except ValueError:
        return default


def _float_setting(name: str, default: float) -> float:
    raw_value = _setting(name, str(default))
    try:
        return float(raw_value)
    except ValueError:
        return default


def _bool_setting(name: str, default: bool) -> bool:
    raw_value = _setting(name, str(default)).lower()
    if raw_value in {"1", "true", "yes", "on"}:
        return True
    if raw_value in {"0", "false", "no", "off"}:
        return False
    return default


def _extensions_setting(name: str, default: str) -> tuple[str, ...]:
    raw_value = _setting(name, default)
    extensions = []
    for item in raw_value.split(","):
        normalized = item.strip().lower()
        if not normalized:
            continue
        if not normalized.startswith("."):
            normalized = f".{normalized}"
        if normalized not in extensions:
            extensions.append(normalized)
    return tuple(extensions)


def _csv_setting(name: str, default: str) -> tuple[str, ...]:
    values = []
    for item in _setting(name, default).split(","):
        normalized = item.strip()
        if normalized and normalized not in values:
            values.append(normalized)
    return tuple(values)


DEEPSEEK_API_KEY = _setting("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = _setting("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = _setting("DEEPSEEK_MODEL", "deepseek-v4-flash")

EMBEDDING_PROVIDER = _setting("EMBEDDING_PROVIDER", "huggingface").lower()
EMBEDDING_MODEL = _setting("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
EMBEDDING_DEVICE = _setting("EMBEDDING_DEVICE", "cpu")
EMBEDDING_NORMALIZE = _bool_setting("EMBEDDING_NORMALIZE", True)
VECTOR_DB_PROVIDER = _setting("VECTOR_DB_PROVIDER", "chroma").lower()
VECTOR_DB_COLLECTION = _setting("VECTOR_DB_COLLECTION", "langchain")
VECTOR_DB_DISTANCE_METRIC = _setting("VECTOR_DB_DISTANCE_METRIC", "l2").lower()
DOCS_DIR = BASE_DIR / _setting("DOCS_DIR", "docs")
CHROMA_DB_DIR = BASE_DIR / _setting("CHROMA_DB_DIR", "chroma_db")
RAG_CHUNK_SIZE = _int_setting("RAG_CHUNK_SIZE", 500)
RAG_CHUNK_OVERLAP = _int_setting("RAG_CHUNK_OVERLAP", 80)
RAG_MAX_DISTANCE = _float_setting("RAG_MAX_DISTANCE", 1.0)
RAG_STREAM_ENABLED = _bool_setting("RAG_STREAM_ENABLED", True)
RAG_SEMANTIC_CACHE_ENABLED = _bool_setting(
    "RAG_SEMANTIC_CACHE_ENABLED",
    True,
)
RAG_SEMANTIC_CACHE_THRESHOLD = _float_setting(
    "RAG_SEMANTIC_CACHE_THRESHOLD",
    0.25,
)
RAG_SEMANTIC_CACHE_TTL_SECONDS = _int_setting(
    "RAG_SEMANTIC_CACHE_TTL_SECONDS",
    24 * 60 * 60,
)
RAG_SEMANTIC_CACHE_COLLECTION = _setting(
    "RAG_SEMANTIC_CACHE_COLLECTION",
    "rag_answer_cache",
)
RAG_SEMANTIC_CACHE_DIR = BASE_DIR / _setting(
    "RAG_SEMANTIC_CACHE_DIR",
    "data/rag_semantic_cache",
)
RAG_PROMPT_VERSION = _setting("RAG_PROMPT_VERSION", "rag_prompt_v3")
RAG_SEMANTIC_CACHE_ALLOWED_CATEGORIES = _csv_setting(
    "RAG_SEMANTIC_CACHE_ALLOWED_CATEGORIES",
    "general",
)
RAG_UPLOAD_DIR = BASE_DIR / _setting("RAG_UPLOAD_DIR", "data/rag_uploads")
RAG_STATE_DB_PATH = BASE_DIR / _setting(
    "RAG_STATE_DB_PATH",
    "data/rag_documents.sqlite3",
)
RAG_ALLOWED_EXTENSIONS = _extensions_setting(
    "RAG_ALLOWED_EXTENSIONS",
    ".pdf,.docx,.xlsx,.txt,.md,.png,.jpg,.jpeg,.webp",
)
RAG_DOCUMENT_MAX_BYTES = _int_setting(
    "RAG_DOCUMENT_MAX_BYTES",
    20 * 1024 * 1024,
)
RAG_IMAGE_MAX_BYTES = _int_setting(
    "RAG_IMAGE_MAX_BYTES",
    10 * 1024 * 1024,
)
RAG_BATCH_MAX_BYTES = _int_setting(
    "RAG_BATCH_MAX_BYTES",
    50 * 1024 * 1024,
)
RAG_OCR_ENABLED = _bool_setting("RAG_OCR_ENABLED", True)
RAG_OCR_BACKEND = _setting(
    "RAG_OCR_BACKEND",
    "rapidocr_onnxruntime",
).lower()


def require_deepseek_key() -> str:
    if not DEEPSEEK_API_KEY:
        raise RuntimeError(
            "DEEPSEEK_API_KEY is empty. Set it in local .env or Streamlit Cloud Secrets."
        )
    return DEEPSEEK_API_KEY
