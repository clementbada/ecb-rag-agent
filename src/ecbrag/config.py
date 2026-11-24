"""Central configuration: paths, models, prices and chunking variants."""
from __future__ import annotations

import json
import os
from pathlib import Path

try:  # load .env if python-dotenv is installed (never commit the .env file)
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
INDEX_DIR = ROOT / "indexes"
RESULTS_DIR = ROOT / "results"
RUNS_DIR = ROOT / "runs"

CORPUS_PATH = DATA_DIR / "ecb_speeches_500.jsonl"
GOLDEN_PATH = DATA_DIR / "golden_set.jsonl"
SECURITY_PATH = DATA_DIR / "security_set.jsonl"
BEST_CONFIG_PATH = RESULTS_DIR / "best_config.json"

# --- Models (served through OpenRouter, same stack as the course labs) -------------
BASE_URL = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
CHAT_MODEL = os.getenv("CHAT_MODEL", "mistralai/mistral-small-2603")  # Mistral Small 4
EMBED_MODEL = os.getenv("EMBED_MODEL", "mistralai/mistral-embed-2312")  # 1024 dims
EMBED_DIM = 1024
# Pin the Mistral provider on OpenRouter so the price table below stays exact.
PROVIDER_ROUTING = {"provider": {"order": ["Mistral"]}}

# USD per 1M tokens
PRICES = {"chat_in": 0.15, "chat_out": 0.60, "embed_in": 0.10}

# --- Chunking variants compared in the retrieval evaluation -------------------------
# A: the course-lab baseline (fixed 200-word windows, 50-word overlap)
# B: same windows + a contextual header (speaker, title, date) prepended before indexing
# C: sentence-aware packing (~250 words, 1-sentence overlap) + contextual header
CHUNK_VARIANTS = {
    "A_fixed200": {"method": "fixed", "size": 200, "overlap": 50, "header": False},
    "B_fixed200_header": {"method": "fixed", "size": 200, "overlap": 50, "header": True},
    "C_sentence250_header": {"method": "sentence", "size": 250, "overlap": 1, "header": True},
}

RETRIEVAL_MODES = ["dense", "bm25", "hybrid", "hybrid_rerank"]

# --- Retrieval / agent defaults ------------------------------------------------------
TOP_K = 5
CANDIDATE_POOL = 20  # retrieve wide (BM25 + dense), then rerank to TOP_K
RERANK_MAX_CHARS = 2000  # per candidate; chunks are ~1.3-1.5k chars, so the reranker reads them whole
RRF_K = 60
MAX_ITERATIONS = 8
MAX_RUN_TOKENS = 60_000  # hard token budget per agent run


def best_config() -> dict:
    """Configuration selected on the DEV split (written by scripts/eval_retrieval.py)."""
    default = {"variant": "C_sentence250_header", "mode": "hybrid_rerank"}
    if BEST_CONFIG_PATH.exists():
        try:
            cfg = json.loads(BEST_CONFIG_PATH.read_text())
            return {"variant": cfg.get("variant", default["variant"]),
                    "mode": cfg.get("mode", default["mode"])}
        except (OSError, ValueError):
            return default
    return default
