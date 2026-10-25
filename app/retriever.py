"""Hybrid search over policy chunks: BM25 (keywords) + vector (meaning), fused with Reciprocal Rank
Fusion, then reranked by a cross-encoder. The top 3 chunks are what the LLM reads.

Search is filtered to the one policy that covers the procedure, so another procedure's policy
can never leak into the evidence. Models and indexes are loaded once and reused.
"""

import pickle
import re
import threading
from functools import lru_cache
from pathlib import Path

from app import config
from app.models import Hit, RetrievalTrace

STOPWORDS = set("a an the and or of to in on for with at by is are be as that this from it its was were has have "
                "had not no any all".split())


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


def rrf_fuse(ranked_lists: list[list[Hit]], k: int = config.RRF_K) -> list[Hit]:
    """Reciprocal Rank Fusion: each list adds 1 / (k + rank) for every chunk it returned (rank starts at 1).

    Chunks found high up by both searches win. Only ranks matter, so BM25 and cosine scores
    (which are on different scales) never need to be compared directly.
    """
    scores: dict[str, float] = {}
    first_seen: dict[str, Hit] = {}
    for hits in ranked_lists:
        for rank, hit in enumerate(hits, start=1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (k + rank)
            first_seen.setdefault(hit.chunk_id, hit)
    ordered = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    return [first_seen[cid].model_copy(update={"score": round(scores[cid], 6)}) for cid in ordered]


@lru_cache(maxsize=1)
def get_embedder():
    from fastembed import TextEmbedding
    return TextEmbedding(config.EMBED_MODEL, cache_dir=str(config.MODEL_CACHE))


@lru_cache(maxsize=1)
def get_reranker():
    from fastembed.rerank.cross_encoder import TextCrossEncoder
    return TextCrossEncoder(config.RERANK_MODEL, cache_dir=str(config.MODEL_CACHE))


class Retriever:
    def __init__(self, index_dir: Path = config.INDEX_DIR):
        import chromadb
        from chromadb.config import Settings
        from rank_bm25 import BM25Okapi

        if not (index_dir / "bm25.pkl").exists():
            raise FileNotFoundError(f"No index in {index_dir}. Run: uv run python -m app.ingest.chunk_index")
        client = chromadb.PersistentClient(path=str(index_dir / "chroma"), settings=Settings(anonymized_telemetry=False))
        self.collection = client.get_collection("policy_chunks")
        with open(index_dir / "bm25.pkl", "rb") as f:
            data = pickle.load(f)
        self.chunks: list[dict] = data["chunks"]
        self.bm25 = BM25Okapi(data["tokens"])
        self.embedder = get_embedder()
        self.reranker = get_reranker()
        self._lock = threading.Lock()  # the ONNX models are shared between threads

    @staticmethod
    def _hit(chunk_id: str, text: str, meta: dict, score: float) -> Hit:
        return Hit(chunk_id=chunk_id, source=meta["source"], page=int(meta["page"]), text=text,
                   score=round(float(score), 4))

    def bm25_search(self, query: str, policy_id: str | None, k: int = config.SEARCH_K) -> list[Hit]:
        scores = self.bm25.get_scores(tokenize(query))
        idx = [i for i, c in enumerate(self.chunks) if policy_id is None or c["metadata"]["policy_id"] == policy_id]
        idx.sort(key=lambda i: scores[i], reverse=True)
        return [self._hit(self.chunks[i]["id"], self.chunks[i]["text"], self.chunks[i]["metadata"], scores[i])
                for i in idx[:k] if scores[i] > 0]

    def vector_search(self, query: str, policy_id: str | None, k: int = config.SEARCH_K) -> list[Hit]:
        with self._lock:
            emb = next(iter(self.embedder.query_embed(query))).tolist()
        res = self.collection.query(query_embeddings=[emb], n_results=k,
                                    where={"policy_id": policy_id} if policy_id else None)
        return [self._hit(cid, doc, meta, 1 - dist)  # cosine distance → similarity
                for cid, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0])]

    def rerank(self, query: str, hits: list[Hit], top: int = config.RERANK_TOP) -> list[Hit]:
        """Cross-encoder reads the query and each chunk together, which is slower but more accurate."""
        if not hits:
            return []
        with self._lock:
            scores = list(self.reranker.rerank(query, [h.text for h in hits]))
        ranked = sorted(zip(hits, scores), key=lambda p: p[1], reverse=True)[:top]
        return [h.model_copy(update={"score": round(float(s), 4)}) for h, s in ranked]

    def retrieve(self, query: str, policy_id: str | None, mode: str = "full") -> RetrievalTrace:
        """mode: full (hybrid, this policy only) | all_policies (hybrid over every policy) | vector_only (this policy only)."""
        if mode == "vector_only":
            vector = self.vector_search(query, policy_id)
            top = vector[:config.RERANK_TOP]
            return RetrievalTrace(vector=vector, reranked=top)
        pid = None if mode == "all_policies" else policy_id
        bm25 = self.bm25_search(query, pid)
        vector = self.vector_search(query, pid)
        fused = rrf_fuse([bm25, vector])
        reranked = self.rerank(query, fused)
        return RetrievalTrace(bm25=bm25, vector=vector, fused=fused, reranked=reranked)


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    return Retriever(config.INDEX_DIR)
