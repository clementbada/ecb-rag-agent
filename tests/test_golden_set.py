from collections import Counter

from ecbrag.corpus import load_corpus, load_golden


def test_corpus_shape():
    docs = load_corpus()
    assert len(docs) == 500
    assert len({d["doc_id"] for d in docs}) == 500


def test_golden_evidence_is_verbatim_in_corpus():
    docs = {d["doc_id"]: d for d in load_corpus()}
    golden = load_golden()
    assert len(golden) >= 40
    for q in golden:
        for ev in q["evidence"]:
            assert ev["text"] in docs[ev["doc_id"]]["text"], q["id"]
        assert q["split"] in ("dev", "test") and q["must"]
    splits = Counter(q["split"] for q in golden)
    assert splits["test"] >= 15, "the held-out split must be large enough to report on"
