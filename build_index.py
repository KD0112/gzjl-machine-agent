from pathlib import Path
import re
import shutil

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from settings import BASE_DIR, CHROMA_DB_DIR, DOCS_DIR, EMBEDDING_MODEL


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


def load_documents():
    documents = []
    for path in DOCS_DIR.glob("*"):
        suffix = path.suffix.lower()
        if suffix in {".txt", ".md"}:
            loaded_docs = TextLoader(str(path), encoding="utf-8").load()
            if suffix == ".md":
                loaded_docs = [_apply_markdown_front_matter(doc) for doc in loaded_docs]
            documents.extend(loaded_docs)
        elif suffix == ".pdf":
            documents.extend(PyPDFLoader(str(path)).load())
    return documents


def build_index() -> None:
    if not DOCS_DIR.exists():
        raise RuntimeError(f"Docs directory not found: {DOCS_DIR}")

    documents = load_documents()
    if not documents:
        raise RuntimeError(f"No .txt, .md, or .pdf documents found in: {DOCS_DIR}")

    _reset_chroma_dir()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DB_DIR),
    )

    print(f"Loaded documents: {len(documents)}")
    print(f"Written chunks: {len(chunks)}")
    print(f"Vector database: {CHROMA_DB_DIR}")


if __name__ == "__main__":
    build_index()
