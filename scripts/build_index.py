"""Chunk + embed the corpus and persist one FAISS/BM25 index per chunking variant.

    python scripts/build_index.py                 # all variants (A, B, C)
    python scripts/build_index.py --variant C_sentence250_header
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ecbrag import config  # noqa: E402
from ecbrag.index import HybridIndex  # noqa: E402
from ecbrag.llm import LLM  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="all", choices=["all", *config.CHUNK_VARIANTS])
    args = ap.parse_args()
    variants = list(config.CHUNK_VARIANTS) if args.variant == "all" else [args.variant]
    llm = LLM()
    for v in variants:
        t0 = time.time()
        before = llm.meter.usd()
        idx = HybridIndex.build(v, llm)
        print(f"[{v}] {len(idx.chunks)} chunks indexed in {time.time() - t0:.0f}s, "
              f"embedding cost ${llm.meter.usd() - before:.4f} ({llm.meter.embed_in:,} tokens so far)")
    print("Done. Indexes are in", config.INDEX_DIR)


if __name__ == "__main__":
    main()
