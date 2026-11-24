"""Ask the agent one question from the terminal and see the full trace.

    python scripts/ask.py "How did Lagarde describe the inflation outlook in 2026?"
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ecbrag import config  # noqa: E402
from ecbrag.agent.graph import build_agent  # noqa: E402
from ecbrag.tracing import flush, traced  # noqa: E402


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    question = " ".join(sys.argv[1:])
    agent = build_agent()
    run = traced(agent.run, "agent-run")(question)
    flush()

    print("\n" + "=" * 80 + f"\nQ: {question}\n" + "-" * 80)
    print(run["answer"])
    print("-" * 80 + "\nSources:")
    chunks = {c["chunk_id"]: c for c in agent.tools.retriever.index.chunks}
    docs = {d["doc_id"]: d for d in agent.tools.docs}
    for s in run["sources"]:
        meta = chunks.get(s) or docs.get(s) or {}
        print(f"  [{s}] {meta.get('speakers', '?')} | {meta.get('date', '?')} | {meta.get('title', '?')}")
    print("-" * 80 + "\nTrace:")
    for step in run["trace"]:
        if step["type"] == "llm":
            print(f"  LLM  iter={step['iteration']} in={step['tokens_in']} out={step['tokens_out']} "
                  f"{step['latency_ms']:.0f}ms tool_calls={step['n_tool_calls']}")
        elif step["type"] == "nudge":
            print(f"  NUDGE iter={step['iteration']} plain-text answer -> asked for final_answer")
        else:
            print(f"  TOOL {step['name']}({step['arguments'][:70]}) -> {step['status']} {step['latency_ms']:.0f}ms")
    print("-" * 80)
    print(f"terminated_by={run['terminated_by']} iterations={run['iterations']} tool_calls={run['tool_calls']} "
          f"hallucinated={run['hallucinated_calls']} tokens={run['tokens_in'] + run['tokens_out']} "
          f"cost=${run['cost_usd']:.4f} latency={run['latency_ms'] / 1000:.1f}s")
    config.RUNS_DIR.mkdir(exist_ok=True)
    out = config.RUNS_DIR / f"ask_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(run, indent=2, ensure_ascii=False))
    print(f"Run saved to {out}")


if __name__ == "__main__":
    main()
