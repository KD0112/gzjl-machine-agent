import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
import re
import shutil

from document_service import DocumentService
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from rag_components import (
    build_index_fingerprint,
    create_embeddings,
    create_vector_store,
    fingerprint_changes,
)
from settings import BASE_DIR, CHROMA_DB_DIR, DOCS_DIR, EMBEDDING_MODEL, RAG_CHUNK_OVERLAP, RAG_CHUNK_SIZE


MANIFEST_PATH = CHROMA_DB_DIR / "index_manifest.json"


def _apply_markdown_front_matter(document):
    content = document.page_content
    match = re.match(r"^---\r?\n(.*?)\r?\n---\r?\n", content, flags=re.DOTALL)
    if not match:
        return document

    front_matter = match.group(1).strip()
    body = content[match.end() :].lstrip()

    for line in front_matter.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value:
            document.metadata[key] = value

    document.page_content = body
    return document


def _reset_chroma_dir() -> None:
    resolved_db_dir = CHROMA_DB_DIR.resolve()
    resolved_base_dir = BASE_DIR.resolve()
    if resolved_db_dir == resolved_base_dir or resolved_base_dir not in resolved_db_dir.parents:
        raise RuntimeError(f"Refusing to delete unsafe Chroma directory: {resolved_db_dir}")
    if CHROMA_DB_DIR.exists():
        shutil.rmtree(CHROMA_DB_DIR)


def _source_for_display(path: str) -> str:
    try:
        return str(Path(path).resolve().relative_to(BASE_DIR.resolve()))
    except ValueError:
        return path


def load_documents(document_service: DocumentService | None = None):
    documents = []
    for path in DOCS_DIR.glob("*"):
        suffix = path.suffix.lower()
        if suffix in {".txt", ".md"}:
            loaded_docs = [
                Document(
                    page_content=path.read_text(encoding="utf-8"),
                    metadata={"source": str(path)},
                )
            ]
            if suffix == ".md":
                loaded_docs = [_apply_markdown_front_matter(doc) for doc in loaded_docs]
            documents.extend(loaded_docs)
        elif suffix == ".pdf":
            reader = PdfReader(str(path))
            documents.extend(
                Document(
                    page_content=page.extract_text() or "",
                    metadata={"source": str(path), "page": page_index},
                )
                for page_index, page in enumerate(reader.pages)
            )
    active_service = document_service or DocumentService()
    documents.extend(active_service.load_active_documents())
    return documents


def _chunk_identity_payload(chunk, chunk_index: int, chunk_size: int, chunk_overlap: int) -> dict[str, object]:
    metadata = chunk.metadata or {}
    payload = {
        "source": metadata.get("source", ""),
        "title": metadata.get("title", ""),
        "category": metadata.get("category", ""),
        "risk_level": metadata.get("risk_level", ""),
        "chunk_index": chunk_index,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "content": chunk.page_content,
    }
    if metadata.get("version_id"):
        payload.update(
            {
                "document_id": metadata.get("document_id", ""),
                "version_id": metadata.get("version_id", ""),
                "version_number": metadata.get("version_number", ""),
                "original_name": metadata.get("original_name", ""),
                "parser_name": metadata.get("parser_name", ""),
                "page": metadata.get("page", ""),
                "sheet": metadata.get("sheet", ""),
                "row_start": metadata.get("row_start", ""),
                "row_end": metadata.get("row_end", ""),
            }
        )
    return payload


def _assign_chunk_metadata(chunks, chunk_size: int, chunk_overlap: int) -> list[str]:
    ids: list[str] = []
    for chunk_index, chunk in enumerate(chunks):
        metadata = dict(chunk.metadata or {})
        payload = _chunk_identity_payload(chunk, chunk_index, chunk_size, chunk_overlap)
        digest = hashlib.sha1(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
        source_name = Path(str(metadata.get("source", "unknown"))).stem or "unknown"
        chunk_id = f"{source_name}-{chunk_index:04d}-{digest}"
        metadata.update(
            {
                "chunk_id": chunk_id,
                "chunk_index": chunk_index,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
            }
        )
        chunk.metadata = metadata
        ids.append(chunk_id)
    return ids


def _load_manifest() -> dict[str, object]:
    if not MANIFEST_PATH.exists():
        return {}
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _write_manifest(
    chunks,
    ids: list[str],
    chunk_size: int,
    chunk_overlap: int,
    index_fingerprint: dict[str, object],
) -> None:
    docs = []
    seen_sources = set()
    for chunk in chunks:
        source = str((chunk.metadata or {}).get("source", "unknown"))
        if source in seen_sources:
            continue
        seen_sources.add(source)
        docs.append(_source_for_display(source))

    payload = {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "embedding_model": EMBEDDING_MODEL,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "document_count": len(docs),
        "chunk_count": len(ids),
        "documents": docs,
        "chunk_ids": ids,
        "index_fingerprint": index_fingerprint,
    }
    MANIFEST_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _split_documents(documents, chunk_size: int, chunk_overlap: int):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(documents)
    ids = _assign_chunk_metadata(chunks, chunk_size, chunk_overlap)
    return chunks, ids


def _get_embeddings():
    """Backward-compatible wrapper around the shared embedding factory."""
    return create_embeddings()


def _full_rebuild(
    documents,
    chunk_size: int,
    chunk_overlap: int,
    *,
    embeddings,
    index_fingerprint: dict[str, object],
) -> tuple[int, list[Document]]:
    _reset_chroma_dir()
    CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
    chunks, ids = _split_documents(documents, chunk_size, chunk_overlap)
    db = create_vector_store(embeddings, persist_directory=CHROMA_DB_DIR)
    db.add_documents(chunks, ids=ids)
    _write_manifest(chunks, ids, chunk_size, chunk_overlap, index_fingerprint)
    return len(chunks), chunks


def _incremental_update(
    documents,
    chunk_size: int,
    chunk_overlap: int,
    *,
    embeddings,
    index_fingerprint: dict[str, object],
) -> tuple[int, int, int, list[Document]]:
    manifest = _load_manifest()
    if not manifest:
        written_chunks, chunks = _full_rebuild(
            documents,
            chunk_size,
            chunk_overlap,
            embeddings=embeddings,
            index_fingerprint=index_fingerprint,
        )
        return written_chunks, 0, 0, chunks

    old_ids = set(manifest.get("chunk_ids", []))
    chunks, ids = _split_documents(documents, chunk_size, chunk_overlap)
    new_ids = set(ids)
    ids_to_delete = sorted(old_ids - new_ids)
    chunks_to_add = [chunk for chunk, chunk_id in zip(chunks, ids) if chunk_id not in old_ids]
    ids_to_add = [chunk_id for chunk_id in ids if chunk_id not in old_ids]

    db = create_vector_store(embeddings, persist_directory=CHROMA_DB_DIR)
    if ids_to_delete:
        db.delete(ids=ids_to_delete)
    if chunks_to_add:
        db.add_documents(chunks_to_add, ids=ids_to_add)

    _write_manifest(chunks, ids, chunk_size, chunk_overlap, index_fingerprint)
    return len(chunks), len(ids_to_add), len(ids_to_delete), chunks


def _uploaded_chunk_counts(
    chunks: list[Document],
    active_version_ids: list[str],
) -> dict[str, int]:
    counts = Counter(
        str((chunk.metadata or {}).get("version_id"))
        for chunk in chunks
        if (chunk.metadata or {}).get("version_id")
    )
    return {
        version_id: int(counts.get(version_id, 0))
        for version_id in active_version_ids
    }


def index_configuration_changes(
    *,
    embeddings=None,
    chunk_size: int = RAG_CHUNK_SIZE,
    chunk_overlap: int = RAG_CHUNK_OVERLAP,
) -> dict[str, dict[str, object]]:
    active_embeddings = embeddings or _get_embeddings()
    current_fingerprint = build_index_fingerprint(
        active_embeddings,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    previous_fingerprint = _load_manifest().get("index_fingerprint")
    return fingerprint_changes(previous_fingerprint, current_fingerprint)


def build_index(
    chunk_size: int = RAG_CHUNK_SIZE,
    chunk_overlap: int = RAG_CHUNK_OVERLAP,
    incremental: bool = False,
    document_service: DocumentService | None = None,
) -> None:
    if not DOCS_DIR.exists():
        raise RuntimeError(f"Docs directory not found: {DOCS_DIR}")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")

    active_service = document_service or DocumentService()
    documents = load_documents(active_service)
    if not documents:
        raise RuntimeError(f"No .txt, .md, or .pdf documents found in: {DOCS_DIR}")

    active_version_ids = active_service.active_version_ids()
    try:
        embeddings = _get_embeddings()
        index_fingerprint = build_index_fingerprint(
            embeddings,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        if incremental and CHROMA_DB_DIR.exists() and any(CHROMA_DB_DIR.iterdir()):
            changes = fingerprint_changes(
                _load_manifest().get("index_fingerprint"),
                index_fingerprint,
            )
            if changes:
                print("Index configuration changed; performing a safe full rebuild.")
                print(json.dumps(changes, ensure_ascii=False, indent=2))
                total_chunks, indexed_chunks = _full_rebuild(
                    documents,
                    chunk_size,
                    chunk_overlap,
                    embeddings=embeddings,
                    index_fingerprint=index_fingerprint,
                )
                print(f"Loaded documents: {len(documents)}")
                print(f"Written chunks: {total_chunks}")
            else:
                (
                    total_chunks,
                    added_chunks,
                    deleted_chunks,
                    indexed_chunks,
                ) = _incremental_update(
                    documents,
                    chunk_size,
                    chunk_overlap,
                    embeddings=embeddings,
                    index_fingerprint=index_fingerprint,
                )
                print(f"Loaded documents: {len(documents)}")
                print(f"Total chunks: {total_chunks}")
                print(f"Added chunks: {added_chunks}")
                print(f"Deleted chunks: {deleted_chunks}")
        else:
            total_chunks, indexed_chunks = _full_rebuild(
                documents,
                chunk_size,
                chunk_overlap,
                embeddings=embeddings,
                index_fingerprint=index_fingerprint,
            )
            print(f"Loaded documents: {len(documents)}")
            print(f"Written chunks: {total_chunks}")

        active_service.mark_indexed(
            _uploaded_chunk_counts(indexed_chunks, active_version_ids)
        )
        active_service.mark_inactive_removed()
    except Exception as exc:
        active_service.mark_index_failed(active_version_ids, exc)
        raise

    print(f"Chunk size: {chunk_size}")
    print(f"Chunk overlap: {chunk_overlap}")
    print(f"Vector database: {CHROMA_DB_DIR}")
    print(f"Manifest: {MANIFEST_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or update the local Chroma knowledge base.")
    parser.add_argument("--chunk-size", type=int, default=RAG_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=RAG_CHUNK_OVERLAP)
    parser.add_argument("--incremental", action="store_true")
    args = parser.parse_args()

    build_index(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        incremental=args.incremental,
    )


if __name__ == "__main__":
    main()
