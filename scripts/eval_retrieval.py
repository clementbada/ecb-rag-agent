"""Retrieval scorecard: chunking variants x retrieval modes, scored on the golden set.

On the DEV split, the best configuration (highest MRR, then nDCG@5) is written to
results/best_config.json and used by the agent. The TEST split is held out: run it once,
at the end, to report the numbers.

    python scripts/eval_retrieval.py --split dev
    python scripts/eval_retrieval.py --split test
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ecbrag import config  # noqa: E402
from ecbrag.corpus import load_corpus, load_golden  # noqa: E402
from ecbrag.index import HybridIndex  # noqa: E402
from ecbrag.llm import LLM  # noqa: E402
from ecbrag.metrics import gold_chunk_ids, hit_at_k, mrr, ndcg_at_k, precision_at_k, recall_at_k  # noqa: E402
from ecbrag.reporting import mean, pct, save_table  # noqa: E402
from ecbrag.retrieval import Retriever  # noqa: E402

K = 5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    ap.add_argument("--variants", nargs="+", default=list(config.CHUNK_VARIANTS))
    ap.add_argument("--modes", nargs="+", default=config.RETRIEVAL_MODES)
    args = ap.parse_args()

    docs_by_id = {d["doc_id"]: d for d in load_corpus()}
    questions = [q for q in load_golden(args.split) if q["evidence"]]  # metadata / unanswerable excluded
    print(f"{len(questions)} {args.split} questions with gold evidence")
    llm = LLM()
    per_q, summary = [], []

    for variant in args.variants:
        index = HybridIndex.load(variant)
        for mode in args.modes:
            retriever = Retriever(index, llm, mode)
            usd0, lat = llm.meter.usd(), []
            rows = []
            for q in questions:
                gold = gold_chunk_ids(q, index.chunks, docs_by_id)
                t0 = time.perf_counter()
                hits = retriever.search(q["question"], k=K)
                lat.append((time.perf_counter() - t0) * 1000)
                ranked = [h["chunk_id"] for h in hits]
                row = {"variant": variant, "mode": mode, "id": q["id"], "type": q["type"],
                       f"recall@{K}": recall_at_k(ranked, gold, K),
                       f"precision@{K}": precision_at_k(ranked, gold, K),
                       "mrr": mrr(ranked, gold), f"ndcg@{K}": ndcg_at_k(ranked, gold, K),
                       f"doc_hit@{K}": hit_at_k([h["doc_id"] for h in hits], set(q["gold_docs"]), K),
                       "n_gold_chunks": len(gold), "latency_ms": lat[-1], "retrieved": " ".join(ranked)}
                rows.append(row)
            per_q.extend(rows)
            n = len(rows)
            summary.append({
                "variant": variant, "mode": mode,
                f"recall@{K}": mean(r[f"recall@{K}"] for r in rows),
                f"precision@{K}": mean(r[f"precision@{K}"] for r in rows),
                "MRR": mean(r["mrr"] for r in rows),
                f"nDCG@{K}": mean(r[f"ndcg@{K}"] for r in rows),
                f"doc_hit@{K}": mean(r[f"doc_hit@{K}"] for r in rows),
                "latency_p50_ms": pct(lat, 50), "cost_per_query_usd": (llm.meter.usd() - usd0) / max(n, 1),
            })
            s = summary[-1]
            print(f"{variant:22s} {mode:14s} R@5={s['recall@5']:.3f} P@5={s['precision@5']:.3f} "
                  f"MRR={s['MRR']:.3f} nDCG={s['nDCG@5']:.3f} p50={s['latency_p50_ms']:.0f}ms "
                  f"${s['cost_per_query_usd']:.5f}/q")

    df = pd.DataFrame(summary)
    note = (f"Split: **{args.split}** ({len(questions)} questions with gold evidence). k = {K}. "
            "Gold chunks = chunks covering at least 50% of a hand-verified evidence span, so variants with "
            "different chunk boundaries are scored on the same facts. Cost includes query embedding and rerank calls.")
    save_table(df, config.RESULTS_DIR / f"retrieval_{args.split}", "Retrieval scorecard", note)
    save_table(pd.DataFrame(per_q), config.RESULTS_DIR / f"retrieval_{args.split}_per_question",
               "Retrieval per question")
    if args.split == "dev":
        best = df.sort_values(["MRR", f"nDCG@{K}", f"precision@{K}"], ascending=False).iloc[0]
        cfg = {"variant": best["variant"], "mode": best["mode"], "selected_on": "dev",
               "MRR": round(float(best["MRR"]), 4), f"nDCG@{K}": round(float(best[f"nDCG@{K}"]), 4)}
        config.BEST_CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
        print("Best configuration on dev ->", cfg)
    print(f"Total spend: ${llm.meter.usd():.4f}. Tables written to {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
