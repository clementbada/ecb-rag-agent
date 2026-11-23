"""Rebuild data/ecb_speeches_500.jsonl from the official ECB speeches dataset.

Source: https://www.ecb.europa.eu/press/key/shared/data/all_ECB_speeches.csv (pipe-separated,
published and updated by the ECB). Selection rule: the 500 most recent speeches with at least
300 words of text, one per title. The processed file is committed, so this script is only
needed to reproduce or refresh the corpus.

    python scripts/download_corpus.py
"""
from __future__ import annotations

import io
import sys
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ecbrag import config  # noqa: E402
from ecbrag.corpus import write_jsonl  # noqa: E402

URL = "https://www.ecb.europa.eu/press/key/shared/data/all_ECB_speeches.csv"
N_DOCS = 500
MIN_WORDS = 300


def main() -> None:
    print(f"Downloading {URL} ...")
    raw = urllib.request.urlopen(URL, timeout=120).read().decode("utf-8")
    df = pd.read_csv(io.StringIO(raw), sep="|", dtype=str).fillna("")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["contents"] = df["contents"].str.replace(r"\s+", " ", regex=True).str.strip()
    df["n_words"] = df["contents"].str.split().str.len()
    keep = (df[df["n_words"] >= MIN_WORDS]
            .sort_values("date", ascending=False)
            .drop_duplicates("title")
            .head(N_DOCS)
            .sort_values("date")
            .reset_index(drop=True))
    rows = [{"doc_id": f"ecb-{i:03d}", "date": r.date.strftime("%Y-%m-%d"), "speakers": r.speakers,
             "title": r.title, "subtitle": r.subtitle, "n_words": int(r.n_words), "text": r.contents}
            for i, r in keep.iterrows()]
    write_jsonl(config.CORPUS_PATH, rows)
    print(f"{len(rows)} speeches, {keep['date'].min().date()} -> {keep['date'].max().date()}, "
          f"{int(keep['n_words'].sum()):,} words -> {config.CORPUS_PATH}")
    print("Note: if the ECB has published new speeches since, the selection (and doc ids) will shift; "
          "the golden set refers to the committed version of the corpus.")


if __name__ == "__main__":
    main()
