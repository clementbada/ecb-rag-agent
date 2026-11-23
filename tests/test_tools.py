import json

from ecbrag.agent.tools import REGISTRY_PARAMS, ToolBox, safe_eval, validate_args
from ecbrag.retrieval import Retriever


def test_validate_args_cases():
    p = REGISTRY_PARAMS["search_speeches"]
    assert validate_args(p, '{"query": "wages"}') == (True, {"query": "wages"}, "")
    ok, _, err = validate_args(p, "{}")
    assert not ok and "query" in err
    ok, _, err = validate_args(p, "not json")
    assert not ok and "JSON" in err
    ok, _, err = validate_args(p, '{"query": "x", "date_from": "March 2024"}')
    assert not ok and "YYYY-MM-DD" in err
    ok, _, err = validate_args(p, '{"query": "x", "k": 50}')
    assert not ok and "between" in err
    ok, args, _ = validate_args(p, '{"query": "x", "junk": 1}')
    assert ok and args == {"query": "x"}
    ok, _, err = validate_args(REGISTRY_PARAMS["final_answer"], '{"answer": "a", "sources": [1]}')
    assert not ok


def test_calculator_is_safe():
    assert safe_eval("(3.3 - 2.3) / 2.3 * 100") == (3.3 - 2.3) / 2.3 * 100
    assert safe_eval("2^3") == 8
    for bad in ["__import__('os')", "open('x')", "2 ** 1000"]:
        try:
            safe_eval(bad)
            raise AssertionError(f"accepted {bad}")
        except (ValueError, SyntaxError):
            pass


def test_dispatch_never_raises_and_validates_citations(small_index, docs):
    tb = ToolBox(Retriever(small_index, None, "bm25"), docs)
    obs, status, _ = tb.dispatch("serach_speeches", '{"query": "x"}')
    assert status == "unknown" and "search_speeches" in obs
    obs, status, _ = tb.dispatch("calculator", '{"expression": "1/0"}')
    assert status == "error" and obs.startswith("Error")
    obs, status, _ = tb.dispatch("final_answer", json.dumps({"answer": "a", "sources": ["ecb-777#c1"]}))
    assert status == "error" and "not returned" in obs
    obs, status, _ = tb.dispatch("search_speeches", '{"query": "cash transactions"}')
    assert status == "ok" and "<document" in obs
    cid = next(iter(tb.contexts))
    obs, status, _ = tb.dispatch("final_answer", json.dumps({"answer": f"40% [{cid}]", "sources": [cid]}))
    assert status == "ok" and tb.final["sources"] == [cid]


def test_list_speeches_counts(small_index, docs):
    tb = ToolBox(Retriever(small_index, None, "bm25"), docs)
    out = tb.list_speeches(date_from="2026-01-01", group_by_speaker=True)
    assert "2 speech(es)" in out and "Piero Cipollone: 1" in out
