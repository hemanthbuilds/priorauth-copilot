"""Reciprocal Rank Fusion math, tokenizer, and the real hybrid retriever's policy filter on sample data."""

import pytest

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


def test_real_retriever_stays_inside_the_graph_policy():
    # Uses the real index built by chunk_index (skipped if it hasn't been built yet).
    from app import config
    if not (config.INDEX_DIR / "bm25.pkl").exists():
        pytest.skip("index not built")
    from app.retriever import get_retriever
    trace = get_retriever().retrieve("at least 12 weeks of non-surgical treatment", "SAMPLE-TKA-001", "full")
    assert trace.reranked and len(trace.reranked) <= 3
    assert {h.source for h in trace.bm25 + trace.vector + trace.reranked} == {"total_knee.txt"}
    assert all(h.text for h in trace.reranked)
