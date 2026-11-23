"""Evaluation metrics: retrieval (Recall@k, precision@k, MRR, nDCG@k), answer correctness,
and RAGAS-style faithfulness (Es et al., 2023) computed with an LLM judge."""
from __future__ import annotations

import math
import re
from typing import Iterable

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------------------
# Gold mapping: evidence spans -> gold chunks (works for any chunking variant)
# ---------------------------------------------------------------------------------------


def evidence_spans(question: dict, docs_by_id: dict) -> list[tuple[str, int, int]]:
    spans = []
    for ev in question.get("evidence", []):
        text = docs_by_id[ev["doc_id"]]["text"]
        start = text.find(ev["text"])
        if start < 0:
            raise ValueError(f"evidence not found in {ev['doc_id']}: {ev['text'][:60]}")
        spans.append((ev["doc_id"], start, start + len(ev["text"])))
    return spans


def gold_chunk_ids(question: dict, chunks: Iterable[dict], docs_by_id: dict, min_cover: float = 0.5) -> set[str]:
    """A chunk is gold if it covers at least `min_cover` of an evidence span.
    With overlapping windows, a fact usually lives in two adjacent chunks: both are gold."""
    spans = evidence_spans(question, docs_by_id)
    gold = set()
    for c in chunks:
        for doc_id, a, b in spans:
            if c["doc_id"] != doc_id:
                continue
            overlap = min(b, c["end"]) - max(a, c["start"])
            if overlap > 0 and overlap / (b - a) >= min_cover:
                gold.add(c["chunk_id"])
    return gold


# ---------------------------------------------------------------------------------------
# Retrieval metrics (binary relevance) — identical to the course formulas
# ---------------------------------------------------------------------------------------


def recall_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    return len(set(ranked[:k]) & gold) / len(gold) if gold else 0.0


def precision_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    return len(set(ranked[:k]) & gold) / k


def mrr(ranked: list[str], gold: set[str]) -> float:
    for i, cid in enumerate(ranked, start=1):
        if cid in gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 1) for i, cid in enumerate(ranked[:k], start=1) if cid in gold)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    return dcg / idcg if idcg else 0.0


def hit_at_k(ranked_docs: list[str], gold_docs: set[str], k: int) -> float:
    """Document-level: did any chunk of a gold document make the top-k?"""
    return float(bool(set(ranked_docs[:k]) & gold_docs))


# ---------------------------------------------------------------------------------------
# Answer correctness against the golden "must" groups
# ---------------------------------------------------------------------------------------

NUM = re.compile(r"[-+−]?\d[\d,]*(?:\.\d+)?")


def _numbers(text: str) -> list[float]:
    vals = []
    for m in NUM.findall(text or ""):
        try:
            vals.append(float(m.replace(",", "").replace("−", "-")))
        except ValueError:
            pass
    return vals


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def is_correct(answer: str, must: list[list[str]], tol: float = 0.01) -> bool:
    """Correct if EVERY group is satisfied; a group is satisfied if ANY alternative appears.
    Integer alternatives (counts, years) must match exactly; decimal alternatives match any
    number in the answer within `tol` (relative);
    text alternatives match as case-insensitive substrings."""
    if not answer:
        return False
    low = answer.lower()
    nums = _numbers(answer)
    for group in must:
        ok = False
        for alt in group:
            if _is_number(alt):
                target = float(alt)
                eps = 1e-9 if "." not in alt else max(abs(target) * tol, 1e-9)
                ok = any(abs(n - target) <= eps for n in nums)
            else:
                ok = alt.lower() in low
            if ok:
                break
        if not ok:
            return False
    return True


# ---------------------------------------------------------------------------------------
# Faithfulness (RAGAS definition): share of the answer's claims supported by the context
# ---------------------------------------------------------------------------------------


class Claims(BaseModel):
    claims: list[str] = Field(default_factory=list)


class Verdicts(BaseModel):
    verdicts: list[bool] = Field(default_factory=list)


def faithfulness(llm, answer: str, contexts: list[str]) -> float | None:
    """F = |supported claims| / |claims|. Returns None when the answer makes no claim
    (e.g. a correct 'not in the knowledge base' refusal)."""
    if not answer or not contexts:
        return None
    claims = llm.chat_parse([
        {"role": "system", "content": "Decompose the answer into simple, standalone factual claims. "
                                      "Ignore statements that only say information is unavailable. "
                                      "Return the claims via the schema (empty list if none)."},
        {"role": "user", "content": answer}], Claims).claims
    if not claims:
        return None
    ctx = "\n\n".join(f"[{n}] {c[:1500]}" for n, c in enumerate(contexts, start=1))
    listed = "\n".join(f"{i}. {c}" for i, c in enumerate(claims, start=1))
    verdicts = llm.chat_parse([
        {"role": "system", "content": "For each numbered claim, answer true only if it is FULLY supported "
                                      "by the passages (arithmetic derived from passage numbers counts as "
                                      "supported), false otherwise. One verdict per claim, same order."},
        {"role": "user", "content": f"Passages:\n{ctx}\n\nClaims:\n{listed}"}], Verdicts).verdicts
    verdicts = (list(verdicts) + [False] * len(claims))[:len(claims)]
    return sum(verdicts) / len(claims)
