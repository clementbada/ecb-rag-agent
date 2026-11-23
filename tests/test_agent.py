import json

from ecbrag.agent.graph import Agent
from ecbrag.agent.tools import ToolBox
from ecbrag.retrieval import Retriever
from conftest import ScriptedLLM


def make_agent(small_index, docs, script, **kw):
    llm = ScriptedLLM(script)
    return Agent(llm, ToolBox(Retriever(small_index, None, "bm25"), docs), **kw), llm


def test_happy_path_terminates_with_final_answer(small_index, docs):
    agent, llm = make_agent(small_index, docs, [
        {"tool_calls": [("search_speeches", '{"query": "cash share transactions 2025", "speaker": "Cipollone"}')]},
    ])
    # second step: cite a chunk that the search actually returned
    first = agent.tools.retriever.search("cash share transactions 2025", speaker="Cipollone")[0]["chunk_id"]
    llm.script.append({"tool_calls": [("final_answer", json.dumps({"answer": f"40% in 2025 [{first}]",
                                                                    "sources": [first]}))]})
    run = agent.run("What share of transactions were in cash in 2025?")
    assert run["terminated_by"] == "final_answer"
    assert run["answer"].startswith("40%") and run["sources"] == [first]
    assert run["iterations"] == 2 and run["tool_calls"] == 2 and run["hallucinated_calls"] == 0
    assert all(m["role"] != "tool" or m["content"] for m in llm.calls[-1])


def test_hallucinated_tool_is_an_observation_and_agent_recovers(small_index, docs):
    agent, _ = make_agent(small_index, docs, [
        {"tool_calls": [("web_search", '{"q": "ecb"}')]},
        {"tool_calls": [("search_speeches", '{"query": "nature resources"}')]},
        {"content": "We use resources 1.7 times faster than they regenerate."},
    ])
    run = agent.run("How fast are natural resources used?")
    assert run["hallucinated_calls"] == 1 and run["tool_errors"] == 1
    assert run["terminated_by"] == "text" and "1.7" in run["answer"]


def test_max_iterations_triggers_fallback(small_index, docs):
    looping = [{"tool_calls": [("search_speeches", json.dumps({"query": f"wages {i}"}))]} for i in range(3)]
    agent, _ = make_agent(small_index, docs, looping + [{"content": "best effort answer"}], max_iterations=3)
    run = agent.run("Loop forever")
    assert run["terminated_by"] == "max_iterations" and run["iterations"] == 3
    assert run["answer"] == "best effort answer"


def test_repeated_call_is_detected(small_index, docs):
    same = {"tool_calls": [("search_speeches", '{"query": "inflation"}')]}
    agent, _ = make_agent(small_index, docs, [same, same, {"content": "done"}])
    run = agent.run("inflation?")
    assert run["repeated_calls"] == 1
