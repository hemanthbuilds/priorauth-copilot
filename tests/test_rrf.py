"""Reciprocal Rank Fusion math, tokenizer, and the real hybrid retriever's policy filter on sample data."""

import pytest

from app import config
from app.models import Hit
from app.retriever import rrf_fuse, tokenize


def hit(cid: str) -> Hit:
    return Hit(chunk_id=cid, source="s", page=1, text=cid, score=0.0)


def test_rrf_scores_match_formula():
    fused = rrf_fuse([[hit("a"), hit("b")], [hit("b"), hit("c")]], k=60)
    scores = {h.chunk_id: h.score for h in fused}
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61, abs=1e-6)  # rank 2 in BM25 + rank 1 in vector
    assert scores["a"] == pytest.approx(1 / 61, abs=1e-6)
    assert scores["c"] == pytest.approx(1 / 62, abs=1e-6)
    assert [h.chunk_id for h in fused] == ["b", "a", "c"]  # found by both lists → first


def test_rrf_handles_empty_lists():
    assert rrf_fuse([[], []]) == []
    assert [h.chunk_id for h in rrf_fuse([[hit("x")], []])] == ["x"]


def test_tokenize_drops_stopwords():
    assert tokenize("The MRI of the Lumbar spine, 6 weeks!") == ["mri", "lumbar", "spine", "6", "weeks"]


def test_real_retriever_stays_inside_the_procedures_policy(tmp_path):
    # Builds a real index of the practice policies (local models, no internet once they are downloaded).
    from app.ingest.chunk_index import build_indexes, load_policy_pages, make_chunks
    from app.policy_store import load_procedures
    from app.retriever import Retriever
    chunks = [c for p in load_procedures() for c in make_chunks(load_policy_pages(p))]
    build_indexes(chunks, tmp_path)
    trace = Retriever(tmp_path).retrieve("at least 12 weeks of non-surgical treatment", "SAMPLE-TKA-001", "full")
    assert trace.reranked and len(trace.reranked) <= config.RERANK_TOP
    assert {h.source for h in trace.bm25 + trace.vector + trace.reranked} == {"total_knee.txt"}
    assert all(h.text for h in trace.reranked)
