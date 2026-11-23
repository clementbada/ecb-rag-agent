"""Chunking strategies. Every chunk keeps its character span in the source document,
so gold evidence can be mapped onto chunks of ANY variant (span overlap), and the
metadata (speaker, title, date) rides along into the citations."""
from __future__ import annotations

import re

WORD = re.compile(r"\S+")
# Sentence boundary: end punctuation, whitespace, then an uppercase letter / quote / digit.
SENT_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"“‘(0-9])")


def fixed_windows(text: str, size: int = 200, overlap: int = 50) -> list[tuple[int, int]]:
    """Fixed word windows with overlap. Window i covers words [i*(size-overlap), +size)."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    words = [m.span() for m in WORD.finditer(text)]
    if not words:
        return []
    step = size - overlap
    spans = []
    for start in range(0, len(words), step):
        window = words[start:start + size]
        spans.append((window[0][0], window[-1][1]))
        if start + size >= len(words):
            break
    return spans


def sentence_spans(text: str) -> list[tuple[int, int]]:
    spans, start = [], 0
    for m in SENT_END.finditer(text):
        spans.append((start, m.start()))
        start = m.end()
    if start < len(text):
        spans.append((start, len(text)))
    return [(a, b) for a, b in spans if text[a:b].strip()]


def sentence_windows(text: str, size: int = 250, overlap: int = 1) -> list[tuple[int, int]]:
    """Structure-aware packing: whole sentences up to ~size words, `overlap` sentences
    repeated between consecutive chunks. Never cuts a sentence in half (an over-long
    sentence becomes its own chunk)."""
    sents = sentence_spans(text)
    if not sents:
        return []
    lengths = [len(text[a:b].split()) for a, b in sents]
    spans, i = [], 0
    while i < len(sents):
        j, n = i, 0
        while j < len(sents) and (n + lengths[j] <= size or j == i):
            n += lengths[j]
            j += 1
        spans.append((sents[i][0], sents[j - 1][1]))
        if j >= len(sents):
            break
        i = max(j - overlap, i + 1)  # always make progress
    return spans


def header_for(doc: dict) -> str:
    return f"{doc['speakers']} — {doc['title']} ({doc['date']})"


def chunk_document(doc: dict, method: str, size: int, overlap: int, header: bool) -> list[dict]:
    text = doc["text"]
    spans = fixed_windows(text, size, overlap) if method == "fixed" else sentence_windows(text, size, overlap)
    chunks = []
    for n, (a, b) in enumerate(spans):
        body = text[a:b]
        chunks.append({
            "chunk_id": f"{doc['doc_id']}#c{n}",
            "doc_id": doc["doc_id"],
            "date": doc["date"],
            "speakers": doc["speakers"],
            "title": doc["title"],
            "start": a,
            "end": b,
            "text": body,
            # what is embedded / BM25-indexed (the contextual header is a design choice)
            "index_text": f"{header_for(doc)}\n{body}" if header else body,
        })
    return chunks


def chunk_corpus(docs: list[dict], variant_cfg: dict) -> list[dict]:
    out = []
    for d in docs:
        out.extend(chunk_document(d, variant_cfg["method"], variant_cfg["size"],
                                  variant_cfg["overlap"], variant_cfg["header"]))
    return out
