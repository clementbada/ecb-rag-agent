"""The hybrid index: FAISS (dense, inner product on unit vectors) + BM25 (sparse),
built once offline per chunking variant and persisted to indexes/<variant>/."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Optional, Protocol

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from . import config
from .chunking import chunk_corpus
from .corpus import fold, load_corpus, read_jsonl, write_jsonl

TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = set("""a an the of and or to in on for by with at from as is are was were be been it its this that
these those which what who whom how when where why does do did has have had not no but if than then so such
into over under about after before between during per our their his her we you they i he she them us""".split())


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN.findall(fold(text)) if t not in STOPWORDS]


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


def embed_all(embedder: Embedder, texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Batched embedding; halves the batch on payload/limit errors."""
    out, i, bs = [], 0, batch_size
    t0 = time.time()
    while i < len(texts):
        batch = texts[i:i + bs]
        try:
            out.append(embedder.embed(batch))
            i += len(batch)
            if len(out) % 20 == 0:
                print(f"  embedded {i}/{len(texts)} chunks ({time.time() - t0:.0f}s)")
        except Exception as exc:  # noqa: BLE001
            if bs == 1:
                raise
            bs = max(1, bs // 2)
            print(f"  embedding batch failed ({type(exc).__name__}); retrying with batch size {bs}")
    return np.vstack(out).astype("float32")


class HybridIndex:
    def __init__(self, variant: str, chunks: list[dict], vectors: np.ndarray):
        self.variant = variant
        self.chunks = chunks
        self.by_id = {c["chunk_id"]: i for i, c in enumerate(chunks)}
        self.vectors = vectors
        self.faiss = faiss.IndexFlatIP(vectors.shape[1])  # exact inner product == cosine
        self.faiss.add(vectors)
        self.bm25 = BM25Okapi([tokenize(c["index_text"]) for c in chunks], k1=1.5, b=0.75)
        self._speaker_folded = [fold(c["speakers"]) for c in chunks]

    # -- persistence -------------------------------------------------------------------
    @staticmethod
    def path(variant: str) -> Path:
        return config.INDEX_DIR / variant

    @classmethod
    def build(cls, variant: str, embedder: Embedder, docs: Optional[list[dict]] = None) -> "HybridIndex":
        cfg = config.CHUNK_VARIANTS[variant]
        docs = docs if docs is not None else load_corpus()
        chunks = chunk_corpus(docs, cfg)
        print(f"[{variant}] {len(docs)} documents -> {len(chunks)} chunks")
        vectors = embed_all(embedder, [c["index_text"] for c in chunks])
        idx = cls(variant, chunks, vectors)
        idx.save()
        return idx

    def save(self) -> None:
        p = self.path(self.variant)
        p.mkdir(parents=True, exist_ok=True)
        write_jsonl(p / "chunks.jsonl", self.chunks)
        np.save(p / "vectors.npy", self.vectors)
        (p / "meta.json").write_text(json.dumps({"variant": self.variant, "n_chunks": len(self.chunks),
                                                 "dim": int(self.vectors.shape[1]),
                                                 "embed_model": config.EMBED_MODEL}, indent=2))

    @classmethod
    def load(cls, variant: str) -> "HybridIndex":
        p = cls.path(variant)
        if not (p / "vectors.npy").exists():
            raise FileNotFoundError(f"No index for variant {variant}. Run: python scripts/build_index.py --variant {variant}")
        return cls(variant, read_jsonl(p / "chunks.jsonl"), np.load(p / "vectors.npy"))

    # -- filtering -----------------------------------------------------------------------
    def allowed(self, speaker: Optional[str] = None, date_from: Optional[str] = None,
                date_to: Optional[str] = None) -> Optional[np.ndarray]:
        """Row ids matching the metadata filters, or None when no filter is set."""
        if not (speaker or date_from or date_to):
            return None
        sp = fold(speaker) if speaker else None
        ids = [i for i, c in enumerate(self.chunks)
               if (not sp or sp in self._speaker_folded[i])
               and (not date_from or c["date"] >= date_from)
               and (not date_to or c["date"] <= date_to)]
        return np.array(ids, dtype="int64")

    # -- the two retrievers ------------------------------------------------------------
    def dense(self, query_vec: np.ndarray, k: int, allowed: Optional[np.ndarray] = None) -> list[tuple[int, float]]:
        q = query_vec.reshape(1, -1).astype("float32")
        if allowed is not None:
            if len(allowed) == 0:
                return []
            k = min(k, len(allowed))
            params = faiss.SearchParameters(sel=faiss.IDSelectorBatch(allowed))
            scores, ids = self.faiss.search(q, k, params=params)
        else:
            scores, ids = self.faiss.search(q, min(k, len(self.chunks)))
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]

    def sparse(self, query: str, k: int, allowed: Optional[np.ndarray] = None) -> list[tuple[int, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        if allowed is not None:
            if len(allowed) == 0:
                return []
            mask = np.full(len(scores), -np.inf)
            mask[allowed] = scores[allowed]
            scores = mask
        top = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in top if np.isfinite(scores[i]) and scores[i] > 0]
