"""Loading the corpus and the golden set."""
from __future__ import annotations

import json
import unicodedata
from pathlib import Path

from . import config


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_corpus(path: Path = config.CORPUS_PATH) -> list[dict]:
    return read_jsonl(path)


def load_golden(split: str | None = None, path: Path = config.GOLDEN_PATH) -> list[dict]:
    rows = read_jsonl(path)
    if split and split != "all":
        rows = [r for r in rows if r["split"] == split]
    return rows


def fold(text: str) -> str:
    """Lowercase + strip accents, for forgiving speaker matching ('vujcic' == 'Vujčić')."""
    nfkd = unicodedata.normalize("NFKD", text or "")
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower()
