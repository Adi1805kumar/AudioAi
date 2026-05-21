"""
rag_engine.py — PDF Ingestion & Vector Retrieval (RAG)
────────────────────────────────────────────────────────
Provides two public functions:
  • ingest_pdf(path)   — parse → chunk → embed → store in ChromaDB
  • search(query)      — embed query → top-K similarity search → return context

This module is called by:
  • tools.py::search_documents()  (Gemini function call)
  • main.py  (on startup, to auto-ingest docs/ folder)

Design choices:
  • sentence-transformers: runs fully offline, no API key needed for embeddings
  • ChromaDB: persistent local vector store, zero configuration
  • Chunk size 500 / overlap 100 chars: good balance for voice-length answers
"""

import os
from pathlib import Path
from loguru import logger

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions

from app.config import CHROMA_DB_PATH, EMBEDDING_MODEL, RAG_TOP_K

# ── Singleton ChromaDB client & collection ────────────────────────────────────
_client = None
_collection = None

_COLLECTION_NAME = "desktop_assistant_docs"
_CHUNK_SIZE = 500
_CHUNK_OVERLAP = 100


def _get_collection():
    global _client, _collection
    if _collection is not None:
        return _collection

    Path(CHROMA_DB_PATH).mkdir(parents=True, exist_ok=True)
    _client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    _collection = _client.get_or_create_collection(
        name=_COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info("[RAG] ChromaDB collection '{}' ready ({} docs).",
                _COLLECTION_NAME, _collection.count())
    return _collection


# ── Public API ─────────────────────────────────────────────────────────────────

def ingest_pdf(file_path: str) -> int:
    """
    Parse a PDF, split into overlapping chunks, embed and store in ChromaDB.
    Returns number of new chunks added.
    Skips if the file has already been ingested (checked by doc_id prefix).
    """
    collection = _get_collection()
    file_path = str(Path(file_path).resolve())
    doc_id_prefix = Path(file_path).stem

    # Check if already ingested (avoid duplicates on restart)
    existing = collection.get(where={"source": file_path}, limit=1)
    if existing["ids"]:
        logger.info("[RAG] '{}' already ingested — skipping.", file_path)
        return 0

    logger.info("[RAG] Ingesting: {}", file_path)
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=_CHUNK_SIZE,
        chunk_overlap=_CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = splitter.split_documents(pages)

    if not chunks:
        logger.warning("[RAG] No text extracted from '{}'.", file_path)
        return 0

    ids = [f"{doc_id_prefix}_{i}" for i in range(len(chunks))]
    texts = [c.page_content for c in chunks]
    metadatas = [{"source": file_path, "page": c.metadata.get("page", 0)} for c in chunks]

    collection.add(ids=ids, documents=texts, metadatas=metadatas)
    logger.success("[RAG] Added {} chunks from '{}'.", len(chunks), Path(file_path).name)
    return len(chunks)


def ingest_folder(folder_path: str = "./docs") -> int:
    """Ingest all PDFs found in a folder (recursive)."""
    folder = Path(folder_path)
    if not folder.exists():
        logger.warning("[RAG] Docs folder '{}' does not exist.", folder_path)
        return 0

    total = 0
    for pdf in folder.rglob("*.pdf"):
        total += ingest_pdf(str(pdf))
    logger.info("[RAG] Folder ingestion complete. {} total chunks added.", total)
    return total


def search(query: str, top_k: int = RAG_TOP_K) -> str:
    """
    Similarity search over ingested documents.
    Returns a formatted string of the top-K results ready to inject into Gemini context.
    """
    collection = _get_collection()

    if collection.count() == 0:
        return "No documents have been ingested yet. Ask the user to drop PDFs into the docs/ folder."

    results = collection.query(
        query_texts=[query],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    if not docs:
        return "No relevant content found in local documents."

    parts = []
    for doc, meta, dist in zip(docs, metas, distances):
        source = Path(meta.get("source", "unknown")).name
        page = meta.get("page", "?")
        relevance = round((1 - dist) * 100, 1)
        parts.append(
            f"[Source: {source}, Page {page}, Relevance: {relevance}%]\n{doc}"
        )

    return "\n\n---\n\n".join(parts)