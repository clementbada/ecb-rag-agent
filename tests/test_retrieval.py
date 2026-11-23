from ecbrag.retrieval import Retriever, rrf_fuse
from conftest import HashEmbedder


def test_rrf_deck_case():
    fused = dict(rrf_fuse(["A", "B", "C"], ["B", "C", "x4", "x5", "x6", "x7", "x8", "x9", "A"]))
    assert abs(fused["B"] - 0.0325) < 1e-4
    assert sorted(fused, key=lambda k: -fused[k])[:3] == ["B", "C", "A"]


def test_bm25_and_hybrid_search(small_index):
    for mode, llm in [("bm25", None), ("hybrid", HashEmbedder()), ("dense", HashEmbedder())]:
        hits = Retriever(small_index, llm, mode).search("cash day-to-day transactions digital euro", k=3)
        assert hits and hits[0]["doc_id"] == "ecb-002", mode


def test_metadata_filters(small_index):
    r = Retriever(small_index, None, "bm25")
    assert all(h["speakers"] == "Frank Elderson" for h in r.search("risks banks", k=5, speaker="elderson"))
    assert all(h["date"] >= "2026-01-01" for h in r.search("inflation cash nature", k=5, date_from="2026-01-01"))
    assert r.search("inflation", k=5, speaker="nobody") == []
