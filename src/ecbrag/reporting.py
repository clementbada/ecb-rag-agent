"""Small helpers to write evaluation tables (CSV + Markdown) into results/."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def save_table(df: pd.DataFrame, path_stem: Path, title: str, note: str = "") -> None:
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path_stem.with_suffix(".csv"), index=False)
    md = f"# {title}\n\n"
    if note:
        md += note.strip() + "\n\n"
    md += df.to_markdown(index=False, floatfmt=".3f") + "\n"
    path_stem.with_suffix(".md").write_text(md, encoding="utf-8")


def _clean(values) -> list[float]:
    """Drop None and NaN (pandas turns None into NaN in numeric columns)."""
    out = []
    for v in values:
        if v is None:
            continue
        v = float(v)
        if not np.isnan(v):
            out.append(v)
    return out


def pct(values, q: float) -> float:
    values = _clean(values)
    return float(np.percentile(values, q)) if values else float("nan")


def mean(values) -> float:
    values = _clean(values)
    return float(np.mean(values)) if values else float("nan")
