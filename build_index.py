from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from settings import CHROMA_DB_DIR, DOCS_DIR, EMBEDDING_MODEL


def load_documents():
    documents = []
    for path in DOCS_DIR.glob("*"):
        suffix = path.suffix.lower()
        if suffix in {".txt", ".md"}:
            documents.extend(TextLoader(str(path), encoding="utf-8").load())
        elif suffix == ".pdf":
            documents.extend(PyPDFLoader(str(path)).load())
    return documents


def build_index() -> None:
    if not DOCS_DIR.exists():
        raise RuntimeError(f"Docs directory not found: {DOCS_DIR}")

    documents = load_documents()
    if not documents:
        raise RuntimeError(f"No .txt, .md, or .pdf documents found in: {DOCS_DIR}")

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
