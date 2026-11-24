"""End-to-end evaluation of the agent on the golden set (held-out TEST split by default).

Reports, per question and in aggregate: answer correctness, RAGAS-style faithfulness,
evidence recall (did the agent retrieve a gold passage?), iterations, tool calls,
hallucinated-tool-call rate, tool errors, termination reason, cost and latency.
Optionally compares with the single-shot RAG baseline (--baseline).

    python scripts/eval_agent.py --split test
    python scripts/eval_agent.py --split test --baseline
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ecbrag import config  # noqa: E402
from ecbrag.agent.graph import build_agent  # noqa: E402
from ecbrag.corpus import load_corpus, load_golden, write_jsonl  # noqa: E402
from ecbrag.generation import rag_answer  # noqa: E402
from ecbrag.metrics import faithfulness, gold_chunk_ids, score_answer  # noqa: E402
from ecbrag.reporting import mean, pct, save_table  # noqa: E402
from ecbrag.tracing import flush, traced  # noqa: E402


def score(llm, q, run, chunks, docs_by_id, judge: bool) -> dict:
    gold = gold_chunk_ids(q, chunks, docs_by_id) if q["evidence"] else set()
    retrieved = set(run["contexts"])
    faith = None
    if judge and q["type"] not in ("unanswerable", "metadata") and run["contexts"]:
        faith = faithfulness(llm, run["answer"], list(run["contexts"].values()))
    return {
        "correct": float(score_answer(run["answer"], q)),
        "faithfulness": faith,
        "evidence_recall": (len(retrieved & gold) > 0) if gold else None,
        "cited": bool(run.get("sources")),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["dev", "test", "all"])
    ap.add_argument("--baseline", action="store_true", help="also run single-shot RAG for comparison")
    ap.add_argument("--no-judge", action="store_true", help="skip the LLM faithfulness judge")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    agent = build_agent()
    llm = agent.llm
    chunks = agent.tools.retriever.index.chunks
    docs_by_id = {d["doc_id"]: d for d in load_corpus()}
    questions = load_golden(args.split)[: args.limit]
    cfg = config.best_config()
    print(f"Agent on {len(questions)} {args.split} questions | index={cfg['variant']} retrieval={cfg['mode']}")

    run_agent = traced(agent.run, "agent-run")
    rows, runs = [], []
    for q in questions:
        run = run_agent(q["question"])
        s = score(llm, q, run, chunks, docs_by_id, not args.no_judge)
        rows.append({"system": "agent", "id": q["id"], "type": q["type"], **s,
                     "iterations": run["iterations"], "tool_calls": run["tool_calls"],
                     "hallucinated_calls": run["hallucinated_calls"], "tool_errors": run["tool_errors"],
                     "repeated_calls": run["repeated_calls"], "terminated_by": run["terminated_by"],
                     "guard": run.get("guard", "off"),
                     "nudged": run["nudged"], "invalid_citations": run["invalid_citations"],
                     "tokens": run["tokens_in"] + run["tokens_out"], "cost_usd": run["cost_usd"],
                     "latency_ms": run["latency_ms"], "answer": run["answer"][:300]})
        runs.append({"id": q["id"], **run})
        r = rows[-1]
        print(f"  {q['id']} [{q['type']:12s}] correct={r['correct']:.0f} iters={r['iterations']} "
              f"tools={r['tool_calls']} halluc={r['hallucinated_calls']} {r['terminated_by']} "
              f"guard={r['guard']} ${r['cost_usd']:.4f} {r['latency_ms'] / 1000:.1f}s")

        if args.baseline:
            t0 = time.perf_counter()
            base = rag_answer(llm, agent.tools.retriever, q["question"])
            sb = score(llm, q, base, chunks, docs_by_id, not args.no_judge)
            rows.append({"system": "rag_baseline", "id": q["id"], "type": q["type"], **sb,
                         "iterations": 1, "tool_calls": 0, "hallucinated_calls": 0, "tool_errors": 0,
                         "repeated_calls": 0, "terminated_by": "single_shot", "guard": "n/a",
                         "nudged": False, "invalid_citations": 0,
                         "tokens": base["tokens_in"] + base["tokens_out"], "cost_usd": base["cost_usd"],
                         "latency_ms": (time.perf_counter() - t0) * 1000, "answer": base["answer"][:300]})
    flush()

    df = pd.DataFrame(rows)
    summary = []
    for system, g in df.groupby("system", sort=False):
        calls = g["tool_calls"].sum()
        summary.append({
            "system": system, "n": len(g),
            "success_rate": g["correct"].mean(),
            "faithfulness": mean(g["faithfulness"].tolist()),
            "evidence_recall": mean([float(x) for x in g["evidence_recall"] if x is not None]),
            "mean_iterations": g["iterations"].mean(),
            "mean_tool_calls": g["tool_calls"].mean(),
            "hallucinated_call_rate": (g["hallucinated_calls"].sum() / calls) if calls else 0.0,
            "tool_error_rate": (g["tool_errors"].sum() / calls) if calls else 0.0,
            "guard_blocked": int(g["guard"].astype(str).str.startswith("blocked").sum()),
            # share of runs that did not end through the validated final_answer tool
            "non_final_answer_rate": float((~g["terminated_by"].isin(["final_answer", "single_shot"])).mean()),
            "invalid_citations": int(g["invalid_citations"].sum()),
            "cost_per_answer_usd": g["cost_usd"].mean(),
            "latency_p50_s": pct(g["latency_ms"], 50) / 1000,
            "latency_p95_s": pct(g["latency_ms"], 95) / 1000,
        })
    sdf = pd.DataFrame(summary)
    by_type = (df.groupby(["system", "type"])["correct"].mean().unstack(0).reset_index()
               .rename(columns={"type": "question type"}))
    term = Counter(df[df.system == "agent"]["terminated_by"])
    note = (f"Split: **{args.split}** (held-out unless dev). Index: `{cfg['variant']}`, retrieval: `{cfg['mode']}`, "
            f"model: `{config.CHAT_MODEL}`. Termination: {dict(term)}. Cost includes agent calls, reranking and "
            "query embeddings. Faithfulness = share of answer claims supported by retrieved passages (LLM judge).")
    save_table(sdf, config.RESULTS_DIR / f"agent_{args.split}", "Agent evaluation", note)
    save_table(by_type, config.RESULTS_DIR / f"agent_{args.split}_by_type", "Success rate by question type")
    save_table(df, config.RESULTS_DIR / f"agent_{args.split}_per_question", "Agent per question")
    config.RUNS_DIR.mkdir(exist_ok=True)
    write_jsonl(config.RUNS_DIR / f"agent_{args.split}_{time.strftime('%Y%m%d_%H%M%S')}.jsonl", runs)
    print(sdf.to_string(index=False))
    print(f"Total spend this evaluation: ${llm.meter.usd():.4f}")


if __name__ == "__main__":
    main()
