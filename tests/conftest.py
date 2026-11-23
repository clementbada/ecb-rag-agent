import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ecbrag.index import HybridIndex, tokenize  # noqa: E402
from ecbrag.chunking import chunk_corpus  # noqa: E402
from ecbrag.llm import ChatResult, Meter, ToolCall, normalise  # noqa: E402


class HashEmbedder:
    """Deterministic offline embedder (bag of hashed words) — tests only."""

    embed_model = "hash-test"

    def __init__(self, dim: int = 128):
        self.dim = dim
        self.meter = Meter()

    def embed(self, texts):
        m = np.zeros((len(texts), self.dim), dtype="float32")
        for i, t in enumerate(texts):
            for w in tokenize(t):
                m[i, hash(w) % self.dim] += 1.0
        return normalise(m)


class ScriptedLLM(HashEmbedder):
    """Fake chat model: replays a list of scripted responses (tool calls or text)."""

    def __init__(self, script):
        super().__init__()
        self.script = list(script)
        self.calls = []

    def chat(self, messages, tools=None, **_):
        self.calls.append(messages)
        step = self.script.pop(0) if self.script else {"content": "fallback answer"}
        calls = [ToolCall(id=f"call_{len(self.calls)}_{j}", name=n, arguments=a)
                 for j, (n, a) in enumerate(step.get("tool_calls", []))]
        return ChatResult(content=step.get("content", ""), tool_calls=calls,
                          tokens_in=100, tokens_out=20, latency_ms=1.0)


DOCS = [
    {"doc_id": "ecb-001", "date": "2025-03-01", "speakers": "Christine Lagarde", "title": "Inflation and policy",
     "subtitle": "", "text": "Inflation stood at 2.4% in February. The Governing Council lowered rates. "
                              "Wage growth is moderating across the euro area. " * 5},
    {"doc_id": "ecb-002", "date": "2026-04-01", "speakers": "Piero Cipollone", "title": "The digital euro",
     "subtitle": "", "text": "Cash use declined from 68% of day-to-day transactions in 2019 to 40% in 2025. "
                              "A digital euro would preserve monetary sovereignty. " * 5},
    {"doc_id": "ecb-003", "date": "2026-03-09", "speakers": "Frank Elderson", "title": "Nature in decline",
     "subtitle": "", "text": "We are using natural resources 1.7 times faster than ecosystems can regenerate them. "
                              "Banks must manage nature-related risks. " * 5},
]


@pytest.fixture
def docs():
    return [dict(d) for d in DOCS]


@pytest.fixture
def small_index(docs):
    chunks = chunk_corpus(docs, {"method": "fixed", "size": 30, "overlap": 10, "header": True})
    emb = HashEmbedder()
    return HybridIndex("test", chunks, emb.embed([c["index_text"] for c in chunks]))
