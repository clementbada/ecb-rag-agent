from ecbrag.metrics import (gold_chunk_ids, is_correct, mrr, ndcg_at_k, precision_at_k, recall_at_k,
                            score_answer)


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


def test_citation_ids_never_satisfy_a_numeric_target():
    # chunk ids carry digits (ecb-492#c5); citing c5 is not answering "5%"
    assert not is_correct("I could not find it in the knowledge base [ecb-492#c5].", [["5"]])
    assert not is_correct("No estimate given (ecb-025#c10)", [["10"]])
    assert not is_correct("See ecb-317 for details.", [["317"]])
    # real figures next to a citation still count
    assert is_correct("Europe hosts about 5% of global AI compute [ecb-492#c5].", [["5"]])
    assert is_correct("It fell from 68% to 40% [ecb-445#c3][ecb-445#c4]", [["68"], ["40"]])


def test_unanswerable_needs_an_explicit_refusal():
    q = {"type": "unanswerable", "must": [["not", "no information"]]}
    assert score_answer("The knowledge base does not contain any speech on this.", q)
    assert score_answer("No information on the December 2026 meeting is available.", q)
    assert score_answer("I could not find a 2021 speech by Vujčić on this topic.", q)
    # the output guard's replacement message is a refusal too
    from ecbrag.guardrails import BLOCKED_ANSWER
    assert score_answer(BLOCKED_ANSWER, q)
    # hallucinations that merely contain "not" used to pass
    assert not score_answer("The Governing Council cut rates by 25 bp; inflation was not a concern.", q)
    assert not score_answer("Vujčić said the limit of 3,000 euros is not too restrictive.", q)


def test_score_answer_uses_must_for_other_types():
    assert score_answer("3.2% in May [ecb-470#c2]", {"type": "fact", "must": [["3.2"]]})
    assert not score_answer("[ecb-470#c3]", {"type": "fact", "must": [["3"]]})
