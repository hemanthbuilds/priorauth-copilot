"""Indexes every policy document in data/policies/ (.pdf, or .txt with "=== Page N ===" markers): splits each
into chunks of ~800 characters and builds the vector index (Chroma) and the keyword index (BM25) over them.
The same chunks are searched and sent to the LLM. To add or update a policy, drop its PDF in the folder and
run this again. Files listed in procedures.yaml are tagged with their policy ID so a question about that
procedure searches only its policy; any other file is tagged with its file name and searched when no
procedure is mentioned.

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


def policy_files() -> list[tuple[Path, str]]:
    """Every policy document in the folder, with the policy ID it belongs to."""
    owner = {f: p["policy_id"] for p in load_procedures() for f in p["files"]}
    files = sorted(f for f in config.POLICY_DIR.iterdir() if f.suffix.lower() in (".pdf", ".txt"))
    return [(f, owner.get(f.name, f.stem)) for f in files]


def load_policy_pages(proc: dict) -> list[Document]:
    """Load every file of one policy as one Document per page (page numbers start at 1)."""
    return [d for name in proc["files"] for d in load_file(config.POLICY_DIR / name, proc["policy_id"])]


def load_file(path: Path, policy_id: str) -> list[Document]:
    """One Document per page of a .pdf, or per "=== Page N ===" section of a .txt (used by the test fixtures)."""
    if not path.exists():
        raise FileNotFoundError(f"Policy file missing: {path}")
    meta = {"policy_id": policy_id, "source": path.name}
    if path.suffix.lower() == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader
        return [Document(page_content=d.page_content, metadata={**meta, "page": int(d.metadata["page"]) + 1})
                for d in PyPDFLoader(str(path)).load()]
    parts = PAGE_MARKER.split(path.read_text())
    return [Document(page_content=text.strip(), metadata={**meta, "page": int(page)})
            for page, text in zip(parts[1::2], parts[2::2])]


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
    for path, policy_id in policy_files():
        chunks = make_chunks(load_file(path, policy_id))
        all_chunks.extend(chunks)
        print(f"{path.name} ({policy_id}): {len(chunks)} chunks")
    build_indexes(all_chunks, config.INDEX_DIR)
    print(f"Indexes written to {config.INDEX_DIR}")


if __name__ == "__main__":
    main()
