from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


PROJECT2_ROOT = Path(__file__).resolve().parents[1]
DAY1_ROOT = PROJECT2_ROOT.parent


def _load_rag_api() -> Any:
    if str(DAY1_ROOT) not in sys.path:
        sys.path.insert(0, str(DAY1_ROOT))
    import rag_chat

    return rag_chat


def _source_record(doc: Any) -> dict[str, Any]:
    metadata = dict(getattr(doc, "metadata", {}) or {})
    source = str(metadata.get("source", "unknown"))
    return {
        "source": source,
        "source_name": Path(source).name,
        "rank": metadata.get("retrieval_rank"),
        "distance": metadata.get("retrieval_distance"),
        "provider": metadata.get("retrieval_provider"),
        "preview": str(getattr(doc, "page_content", ""))[:260],
    }


def query_knowledge(question: str, top_k: int = 5) -> dict[str, Any]:
    """Reuse project one's RAG pipeline as a structured project-two tool."""
    rag = _load_rag_api()
    if getattr(rag, "DEEPSEEK_API_KEY", ""):
        result = rag.answer_with_metadata(question, k=top_k)
        retrieval = result["retrieval"]
        answer = result["answer"]
        answer_source = result["answer_source"]
        docs = result["docs"]
    else:
        classification = rag.classify_question(question)
        retrieval = rag.retrieve_with_metadata(question, k=top_k)
        docs = retrieval["docs"]
        if retrieval["status"] in {"no_docs", "low_confidence"}:
            answer = rag.build_low_confidence_answer(classification)
        else:
            answer = (
                "已找到相关知识片段，但当前未配置生成模型。"
                "为避免错误解释，建议由人工客服结合来源内容确认。"
            )
        answer_source = "retrieval_only"

    sources = [_source_record(doc) for doc in docs]
    status = str(retrieval.get("status", "error"))
    needs_handoff = status != "ok" or answer_source != "llm"
    return {
        "matched": status == "ok",
        "answer": answer,
        "answer_source": answer_source,
        "retrieval_status": status,
        "retrieval_reason": retrieval.get("reason", ""),
        "top_distance": retrieval.get("top_distance"),
        "max_distance": retrieval.get("max_distance"),
        "retriever": retrieval.get("retriever", {}),
        "sources": sources,
        "needs_handoff": needs_handoff,
        "need_manual_confirm": needs_handoff,
    }
