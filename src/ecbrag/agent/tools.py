"""The agent's tools: registry (schemas), argument validation and dispatch.

Contract (session 6): a tool problem is an OBSERVATION for the model, never an exception.
dispatch() always returns a string and a status: ok | unknown | invalid | error.
"""
from __future__ import annotations

import ast
import json
import operator as op
import re
import time
from collections import Counter
from typing import Any, Optional

from ..corpus import fold
from ..guardrails import wrap_document

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
# A source id as cited inline: ecb-445#c3 (chunk) or ecb-445 (whole speech).
SOURCE_ID_RE = re.compile(r"\becb-\d+(?:#c\d+)?\b")


def cited_ids(text: str) -> list[str]:
    """Source ids cited inline in an answer, in order of first appearance."""
    seen: dict[str, None] = {}
    for m in SOURCE_ID_RE.findall(text or ""):
        seen.setdefault(m, None)
    return list(seen)

TOOL_SCHEMAS: list[dict] = [
    {"type": "function", "function": {
        "name": "search_speeches",
        "description": (
            "Hybrid semantic + keyword search over 500 ECB speeches (Feb 2020 - Oct 2026). "
            "Returns the most relevant passages with their ids, speaker, date and title. "
            "Use the optional speaker / date filters whenever the question names a person or a period. "
            "Call it several times with different queries for multi-part questions."),
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "What to look for, in English."},
            "speaker": {"type": "string", "description": "Optional speaker surname, e.g. 'Lagarde'."},
            "date_from": {"type": "string", "description": "Optional lower date bound, YYYY-MM-DD."},
            "date_to": {"type": "string", "description": "Optional upper date bound, YYYY-MM-DD."},
            "k": {"type": "integer", "description": "Number of passages (1-8, default 5).", "minimum": 1, "maximum": 8},
        }, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "list_speeches",
        "description": (
            "Look up the speech catalogue (metadata only: date, speaker, title, doc id). Use it to count "
            "speeches, find the latest speech of someone, or find a speech by its title. "
            "Set group_by_speaker=true to get the number of speeches per speaker. "
            "Cite catalogue results as [catalogue] or by their doc id."),
        "parameters": {"type": "object", "properties": {
            "speaker": {"type": "string", "description": "Optional speaker surname."},
            "date_from": {"type": "string", "description": "Optional lower date bound, YYYY-MM-DD."},
            "date_to": {"type": "string", "description": "Optional upper date bound, YYYY-MM-DD."},
            "title_contains": {"type": "string", "description": "Optional words that must appear in the title."},
            "group_by_speaker": {"type": "boolean", "description": "Return counts per speaker."},
            "limit": {"type": "integer", "description": "Max rows listed (1-30, default 10).", "minimum": 1, "maximum": 30},
        }, "required": []}}},
    {"type": "function", "function": {
        "name": "calculator",
        "description": "Exact arithmetic. Supports + - * / ** % and parentheses. Never compute by hand.",
        "parameters": {"type": "object", "properties": {
            "expression": {"type": "string", "description": "e.g. (3.3 - 2.3) / 2.3 * 100"},
        }, "required": ["expression"]}}},
    {"type": "function", "function": {
        "name": "final_answer",
        "description": (
            "Give the final answer to the user. Call it exactly once, at the end. The answer must cite "
            "its sources inline as [chunk_id] and list them in `sources`. Only cite ids returned by "
            "tools during this conversation. If the knowledge base does not contain the answer, say so "
            "and leave sources empty."),
        "parameters": {"type": "object", "properties": {
            "answer": {"type": "string", "description": "Concise answer with inline [chunk_id] citations."},
            "sources": {"type": "array", "items": {"type": "string"},
                        "description": "The chunk ids (or doc ids) supporting the answer."},
        }, "required": ["answer", "sources"]}}},
]

REGISTRY_PARAMS = {t["function"]["name"]: t["function"]["parameters"] for t in TOOL_SCHEMAS}

_TYPES = {"string": str, "integer": int, "boolean": bool, "array": list}


def validate_args(parameters: dict, arguments_json: str) -> tuple[bool, Optional[dict], str]:
    """Check raw JSON arguments against a (subset of) JSON Schema. Never raises."""
    try:
        raw = json.loads(arguments_json or "{}")
    except json.JSONDecodeError as exc:
        return False, None, f"arguments must be valid JSON ({exc.msg})"
    if not isinstance(raw, dict):
        return False, None, "arguments must be a JSON object"
    props = parameters.get("properties", {})
    missing = [f for f in parameters.get("required", []) if f not in raw]
    if missing:
        return False, None, f"missing required field(s): {', '.join(missing)}"
    args: dict[str, Any] = {}
    for name, value in raw.items():
        if name not in props or value is None:
            continue  # unknown fields are dropped, not fatal
        spec = props[name]
        expected = spec.get("type")
        if expected == "integer" and isinstance(value, float) and value.is_integer():
            value = int(value)
        py = _TYPES.get(expected)
        if py and (not isinstance(value, py) or (py is int and isinstance(value, bool))):
            return False, None, f"field '{name}' must be of type {expected}"
        if expected == "array" and spec.get("items", {}).get("type") == "string":
            if not all(isinstance(v, str) for v in value):
                return False, None, f"field '{name}' must be a list of strings"
        if expected == "integer":
            lo, hi = spec.get("minimum"), spec.get("maximum")
            if (lo is not None and value < lo) or (hi is not None and value > hi):
                return False, None, f"field '{name}' must be between {lo} and {hi}"
        if name.startswith("date_") and isinstance(value, str) and value and not DATE_RE.match(value):
            return False, None, f"field '{name}' must use the format YYYY-MM-DD"
        if isinstance(value, str) and name != "answer" and not value.strip() and name in parameters.get("required", []):
            return False, None, f"field '{name}' must not be empty"
        args[name] = value
    return True, args, ""


# -- safe calculator -------------------------------------------------------------------------
_BIN = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
        ast.Pow: op.pow, ast.Mod: op.mod, ast.FloorDiv: op.floordiv}
_UN = {ast.UAdd: op.pos, ast.USub: op.neg}


def safe_eval(expression: str) -> float:
    expression = expression.replace("^", "**").replace("×", "*").replace("−", "-")

    def ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
            right = ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("exponent too large")
            return _BIN[type(node.op)](ev(node.left), right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UN:
            return _UN[type(node.op)](ev(node.operand))
        raise ValueError("only numbers and + - * / ** % ( ) are allowed")

    return ev(ast.parse(expression, mode="eval").body)


class ToolBox:
    """Tools bound to one retriever + the speech catalogue. Keeps per-run state:
    which sources were returned (for citation validation) and the retrieved contexts."""

    def __init__(self, retriever, docs: list[dict]):
        self.retriever = retriever
        self.docs = sorted(docs, key=lambda d: d["date"])
        self.reset()

    def reset(self) -> None:
        self.seen_ids: set[str] = set()
        self.contexts: dict[str, str] = {}  # chunk_id -> text (for faithfulness)
        self.final: Optional[dict] = None

    # -- tools -----------------------------------------------------------------------------
    def search_speeches(self, query: str, speaker: str | None = None, date_from: str | None = None,
                        date_to: str | None = None, k: int = 5) -> str:
        hits = self.retriever.search(query, k=k, speaker=speaker, date_from=date_from, date_to=date_to)
        if not hits:
            filt = ", ".join(f"{n}={v}" for n, v in [("speaker", speaker), ("date_from", date_from),
                                                     ("date_to", date_to)] if v)
            return ("No passage matched" + (f" with filters ({filt})" if filt else "") +
                    ". Try broader keywords, remove or widen a filter, or check the speaker with list_speeches.")
        for h in hits:
            self.seen_ids.update({h["chunk_id"], h["doc_id"]})
            # metadata travels with the text so the faithfulness judge can verify "who / when"
            self.contexts[h["chunk_id"]] = f"{h['speakers']} — {h['title']} ({h['date']})\n{h['text']}"
        return "\n\n".join(wrap_document(h) for h in hits)

    def list_speeches(self, speaker: str | None = None, date_from: str | None = None,
                      date_to: str | None = None, title_contains: str | None = None,
                      group_by_speaker: bool = False, limit: int = 10) -> str:
        sp = fold(speaker) if speaker else None
        words = fold(title_contains).split() if title_contains else []
        rows = [d for d in self.docs
                if (not sp or sp in fold(d["speakers"]))
                and (not date_from or d["date"] >= date_from)
                and (not date_to or d["date"] <= date_to)
                and all(w in fold(d["title"]) for w in words)]
        if not rows:
            return "No speech in the catalogue matches these filters."
        head = f"{len(rows)} speech(es) match (catalogue covers {self.docs[0]['date']} to {self.docs[-1]['date']})."
        if group_by_speaker:
            counts = Counter(s.strip() for d in rows for s in d["speakers"].split(","))
            self.seen_ids.add("catalogue")
            out = head + "\nSpeeches per speaker:\n" + "\n".join(f"- {s}: {n}" for s, n in counts.most_common())
            self.contexts[f"catalogue:{len(self.contexts)}"] = out
            return out
        rows = sorted(rows, key=lambda d: d["date"], reverse=True)[:limit]
        for d in rows:
            self.seen_ids.add(d["doc_id"])
        listing = "\n".join(f"- {d['doc_id']} | {d['date']} | {d['speakers']} | {d['title']}" for d in rows)
        out = f"{head} Most recent first:\n{listing}"
        self.contexts[f"catalogue:{len(self.contexts)}"] = out
        return out

    def calculator(self, expression: str) -> str:
        value = safe_eval(expression)
        if isinstance(value, float):
            value = round(value, 10)
            if value.is_integer():
                value = int(value)
        self.contexts[f"calc:{len(self.contexts)}"] = f"Calculator result: {expression} = {value}"
        return str(value)

    def final_answer(self, answer: str, sources: list[str]) -> str:
        # Check the `sources` list AND every id cited inline in the answer text: a fabricated
        # [ecb-999#c7] written in the answer must be rejected even if `sources` is clean.
        inline = cited_ids(answer)
        unknown = [s for s in dict.fromkeys(list(sources) + inline) if s not in self.seen_ids]
        if unknown:
            raise ValueError("these sources were not returned by any tool in this run: "
                             + ", ".join(unknown[:5]) + ". Cite only ids you actually retrieved.")
        # sources = what the model listed + anything it cited inline but forgot to list
        self.final = {"answer": answer, "sources": list(dict.fromkeys(list(sources) + inline))}
        return "Final answer recorded."

    def verified_citations(self, text: str) -> tuple[list[str], list[str]]:
        """Split the ids cited in `text` into (returned by a tool this run, never returned)."""
        ids = cited_ids(text)
        return [i for i in ids if i in self.seen_ids], [i for i in ids if i not in self.seen_ids]

    # -- dispatch ----------------------------------------------------------------------------
    def classify(self, name: str, arguments_json: str) -> str:
        if name not in REGISTRY_PARAMS:
            return "unknown"
        ok, _, _ = validate_args(REGISTRY_PARAMS[name], arguments_json)
        return "ok" if ok else "invalid"

    def dispatch(self, name: str, arguments_json: str) -> tuple[str, str, float]:
        """Returns (observation, status, latency_ms). Never raises."""
        t0 = time.perf_counter()
        if name not in REGISTRY_PARAMS:
            obs, status = (f"Error: unknown tool '{name}'. Available tools: "
                           + ", ".join(sorted(REGISTRY_PARAMS)), "unknown")
        else:
            ok, args, err = validate_args(REGISTRY_PARAMS[name], arguments_json)
            if not ok:
                obs, status = f"Error: invalid arguments for tool '{name}': {err}", "invalid"
            else:
                try:
                    obs, status = getattr(self, name)(**args), "ok"
                except Exception as exc:  # noqa: BLE001 - errors are observations
                    obs, status = f"Error: tool '{name}' failed: {exc}", "error"
        return obs, status, (time.perf_counter() - t0) * 1000
