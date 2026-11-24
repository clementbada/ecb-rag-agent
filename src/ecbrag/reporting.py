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


def bootstrap_ci(values, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean. With ~20 test questions one question moves a
    rate by ~5 points, so every headline number should travel with its interval."""
    arr = np.array(_clean(values), dtype=float)
    if len(arr) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_bootstrap_diff(a, b, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> dict:
    """Mean of (a - b) over the SAME questions, with a bootstrap CI. If the interval
    contains 0, the difference between the two systems is not established."""
    pairs = np.array([(x, y) for x, y in zip(a, b) if x is not None and y is not None], dtype=float)
    pairs = pairs[~np.isnan(pairs).any(axis=1)] if len(pairs) else pairs
    if len(pairs) == 0:
        return {"diff": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    d = pairs[:, 0] - pairs[:, 1]
    lo, hi = bootstrap_ci(d, n_boot, alpha, seed)
    return {"diff": float(d.mean()), "ci_low": lo, "ci_high": hi, "n": int(len(d))}


def fmt_ci(ci: tuple[float, float]) -> str:
    return f"[{ci[0]:.2f}, {ci[1]:.2f}]"
