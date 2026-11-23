from ecbrag.metrics import gold_chunk_ids, is_correct, mrr, ndcg_at_k, precision_at_k, recall_at_k


def test_deck_mini_case():
    ranked, gold = ["x", "g1", "g2", "x2", "x3"], {"g1", "g2"}
    assert recall_at_k(ranked, gold, 5) == 1.0
    assert precision_at_k(ranked, gold, 5) == 0.4
    assert mrr(ranked, gold) == 0.5
    assert abs(ndcg_at_k(ranked, gold, 5) - 0.69) < 0.005


def test_is_correct_numbers_and_text():
    assert is_correct("Cash fell to 40% in 2025 [ecb-445#c3]", [["40"]])
    assert not is_correct("Cash fell to 45%", [["40"]])
    assert is_correct("Inflation was -0.2% in August", [["-0.2"]])
    assert is_correct("Eight members at launch, 141 in 2024", [["eight", "8"], ["141"]])
    assert not is_correct("141 members in 2024", [["eight", "8"], ["141"]])
    assert is_correct("This is not in the knowledge base.", [["not", "no information"]])
    assert not is_correct("", [["1"]])


def test_gold_mapping_uses_span_overlap(small_index, docs):
    q = {"evidence": [{"doc_id": "ecb-003", "text": "1.7 times faster than ecosystems"}]}
    gold = gold_chunk_ids(q, small_index.chunks, {d["doc_id"]: d for d in docs})
    assert gold and all(g.startswith("ecb-003#") for g in gold)


def test_integer_targets_match_exactly():
    assert not is_correct("142 members", [["141"]])
    assert is_correct("141 members", [["141"]])
    assert is_correct("about 43.48%", [["43.5"]])
