"""Splits each policy into chunks of ~800 characters, then builds the vector index (Chroma) and the
keyword index (BM25) over them. The same chunks are searched and sent to the LLM.

Run: uv run python -m app.ingest.chunk_index
"""

import pickle
import re
import shutil
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app import config
from app.policy_store import load_procedures
from app.retriever import get_embedder, tokenize

PAGE_MARKER = re.compile(r"^=== Page (\d+) ===\s*$", re.MULTILINE)


def load_policy_pages(proc: dict) -> list[Document]:
    """Load every file of one policy as one Document per page (page numbers start at 1)."""
    docs = []
    for name in proc["files"]:
        path = config.POLICY_DIR / name
        if not path.exists():
            raise FileNotFoundError(f"Policy file missing: {path} (see README: downloading the CMS policies)")
        meta = {"policy_id": proc["policy_id"], "policy_key": proc["policy_key"], "source": name}
        if path.suffix.lower() == ".pdf":
            from langchain_community.document_loaders import PyPDFLoader
            for d in PyPDFLoader(str(path)).load():
                docs.append(Document(page_content=d.page_content, metadata={**meta, "page": int(d.metadata["page"]) + 1}))
        else:
            # Sample policies are text files with "=== Page N ===" markers instead of PDF pages.
            parts = PAGE_MARKER.split(path.read_text())
            for page, text in zip(parts[1::2], parts[2::2]):
                docs.append(Document(page_content=text.strip(), metadata={**meta, "page": int(page)}))
    return docs


def make_chunks(pages: list[Document]) -> list[dict]:
    """Return chunks [{id, text, metadata}]; each chunk stays on one page so its page number is exact."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
    chunks = []
    for page_doc in pages:
        m = page_doc.metadata
        for i, text in enumerate(splitter.split_text(page_doc.page_content)):
            chunks.append({"id": f"{m['policy_id']}|{m['source']}|p{m['page']}|{i}", "text": text, "metadata": m})
    return chunks


def build_indexes(chunks: list[dict], index_dir: Path) -> None:
    import chromadb
    from chromadb.config import Settings

    index_dir.mkdir(parents=True, exist_ok=True)
    chroma_dir = index_dir / "chroma"
    if chroma_dir.exists():
        shutil.rmtree(chroma_dir)  # rebuild from scratch so removed policies don't linger
    client = chromadb.PersistentClient(path=str(chroma_dir), settings=Settings(anonymized_telemetry=False))
    col = client.create_collection("policy_chunks", metadata={"hnsw:space": "cosine"})
    embeddings = [e.tolist() for e in get_embedder().passage_embed([c["text"] for c in chunks])]
    col.add(ids=[c["id"] for c in chunks], documents=[c["text"] for c in chunks],
            embeddings=embeddings, metadatas=[c["metadata"] for c in chunks])

    with open(index_dir / "bm25.pkl", "wb") as f:
        pickle.dump({"chunks": chunks, "tokens": [tokenize(c["text"]) for c in chunks]}, f)


def main() -> None:
    all_chunks = []
    for proc in load_procedures():
        chunks = make_chunks(load_policy_pages(proc))
        all_chunks.extend(chunks)
        print(f"{proc['policy_id']}: {len(chunks)} chunks")
    build_indexes(all_chunks, config.INDEX_DIR)
    print(f"Indexes written to {config.INDEX_DIR}")


if __name__ == "__main__":
    main()
