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


DEEPSEEK_API_KEY = _setting("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = _setting("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = _setting("DEEPSEEK_MODEL", "deepseek-v4-flash")

EMBEDDING_MODEL = _setting("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
DOCS_DIR = BASE_DIR / _setting("DOCS_DIR", "docs")
CHROMA_DB_DIR = BASE_DIR / _setting("CHROMA_DB_DIR", "chroma_db")


def require_deepseek_key() -> str:
    if not DEEPSEEK_API_KEY:
        raise RuntimeError(
            "DEEPSEEK_API_KEY is empty. Set it in local .env or Streamlit Cloud Secrets."
        )
    return DEEPSEEK_API_KEY
