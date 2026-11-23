"""LLM access layer: one metered client for chat, structured outputs and embeddings.

Every call records tokens, cost and latency in a shared Meter, so cost per answer and
latency are always available. If LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are set, the
OpenAI client is swapped for Langfuse's drop-in wrapper and every call is traced there too.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Type

import numpy as np
from pydantic import BaseModel

from . import config

FATAL_STATUS = {400, 401, 402, 403, 404, 422}


def langfuse_enabled() -> bool:
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


def _openai_class():
    if langfuse_enabled():
        try:
            from langfuse.openai import OpenAI  # drop-in, traces every call

            return OpenAI
        except ImportError:
            pass
    from openai import OpenAI

    return OpenAI


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON string, exactly as produced by the model


@dataclass
class ChatResult:
    content: str
    tool_calls: list[ToolCall]
    tokens_in: int
    tokens_out: int
    latency_ms: float

    @property
    def cost_usd(self) -> float:
        return usd_chat(self.tokens_in, self.tokens_out)

    def as_message(self) -> dict:
        """The assistant message to append to the conversation."""
        msg: dict[str, Any] = {"role": "assistant", "content": self.content or ""}
        if self.tool_calls:
            msg["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.name, "arguments": tc.arguments}}
                for tc in self.tool_calls
            ]
        return msg


def usd_chat(tokens_in: int, tokens_out: int) -> float:
    return (tokens_in * config.PRICES["chat_in"] + tokens_out * config.PRICES["chat_out"]) / 1e6


def usd_embed(tokens_in: int) -> float:
    return tokens_in * config.PRICES["embed_in"] / 1e6


@dataclass
class Meter:
    """Cumulative usage counters (the course's cost discipline)."""

    chat_calls: int = 0
    chat_in: int = 0
    chat_out: int = 0
    embed_calls: int = 0
    embed_in: int = 0
    events: list = field(default_factory=list)

    def usd(self) -> float:
        return usd_chat(self.chat_in, self.chat_out) + usd_embed(self.embed_in)

    def snapshot(self) -> dict:
        return {"chat_calls": self.chat_calls, "chat_in": self.chat_in, "chat_out": self.chat_out,
                "embed_calls": self.embed_calls, "embed_in": self.embed_in,
                "usd": round(self.usd(), 6)}


class LLM:
    """Thin wrapper over an OpenAI-compatible endpoint (OpenRouter by default)."""

    def __init__(self, api_key: Optional[str] = None, base_url: str = config.BASE_URL,
                 chat_model: str = config.CHAT_MODEL, embed_model: str = config.EMBED_MODEL,
                 max_attempts: int = 5, backoff_s: float = 2.0):
        api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set. Copy .env.example to .env and fill it in.")
        OpenAI = _openai_class()
        self.client = OpenAI(api_key=api_key, base_url=base_url, max_retries=0,
                             default_headers={"X-Title": "ECB speeches RAG agent"})
        self.chat_model = chat_model
        self.embed_model = embed_model
        self.max_attempts = max_attempts
        self.backoff_s = backoff_s
        self.meter = Meter()

    # -- retry policy ----------------------------------------------------------------
    def _with_retries(self, fn, what: str):
        for attempt in range(1, self.max_attempts + 1):
            try:
                return fn()
            except Exception as exc:  # noqa: BLE001 - we classify below
                status = getattr(exc, "status_code", None)
                if status in FATAL_STATUS or attempt == self.max_attempts:
                    raise
                wait = self.backoff_s ** attempt
                print(f"  [{what}] {type(exc).__name__} (status={status}) - retry {attempt} in {wait:.0f}s")
                time.sleep(wait)
        raise RuntimeError("unreachable")

    # -- chat --------------------------------------------------------------------------
    def chat(self, messages: list[dict], tools: Optional[list[dict]] = None,
             temperature: float = 0.0, max_tokens: int = 1200) -> ChatResult:
        kwargs: dict[str, Any] = dict(model=self.chat_model, messages=messages,
                                      temperature=temperature, max_tokens=max_tokens,
                                      extra_body=config.PROVIDER_ROUTING)
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        t0 = time.perf_counter()
        resp = self._with_retries(lambda: self.client.chat.completions.create(**kwargs), "chat")
        latency = (time.perf_counter() - t0) * 1000
        msg = resp.choices[0].message
        calls = [ToolCall(id=tc.id, name=tc.function.name, arguments=tc.function.arguments or "")
                 for tc in (msg.tool_calls or [])]
        tin = getattr(resp.usage, "prompt_tokens", 0) or 0
        tout = getattr(resp.usage, "completion_tokens", 0) or 0
        self._record_chat(tin, tout, latency, "chat")
        return ChatResult(content=msg.content or "", tool_calls=calls,
                          tokens_in=tin, tokens_out=tout, latency_ms=latency)

    def chat_parse(self, messages: list[dict], schema: Type[BaseModel],
                   temperature: float = 0.0) -> BaseModel:
        """Structured output: the response is constrained to the Pydantic schema."""
        parse = getattr(self.client.chat.completions, "parse", None)
        if parse is None:  # older SDKs
            parse = self.client.beta.chat.completions.parse
        t0 = time.perf_counter()
        try:
            resp = self._with_retries(
                lambda: parse(model=self.chat_model, messages=messages, response_format=schema,
                              temperature=temperature, extra_body=config.PROVIDER_ROUTING),
                "parse")
            parsed = resp.choices[0].message.parsed
            usage = resp.usage
        except Exception:  # noqa: BLE001 - fall back to JSON mode + validation
            resp = self._with_retries(
                lambda: self.client.chat.completions.create(
                    model=self.chat_model, temperature=temperature,
                    response_format={"type": "json_object"}, extra_body=config.PROVIDER_ROUTING,
                    messages=messages + [{"role": "system", "content":
                                          "Reply with JSON matching this schema: "
                                          + json.dumps(schema.model_json_schema())}]),
                "json")
            parsed = schema.model_validate_json(resp.choices[0].message.content or "{}")
            usage = resp.usage
        latency = (time.perf_counter() - t0) * 1000
        self._record_chat(getattr(usage, "prompt_tokens", 0) or 0,
                          getattr(usage, "completion_tokens", 0) or 0, latency, "parse")
        return parsed

    def _record_chat(self, tin: int, tout: int, latency_ms: float, kind: str) -> None:
        self.meter.chat_calls += 1
        self.meter.chat_in += tin
        self.meter.chat_out += tout
        self.meter.events.append({"kind": kind, "tokens_in": tin, "tokens_out": tout,
                                  "latency_ms": round(latency_ms, 1),
                                  "usd": usd_chat(tin, tout)})

    # -- embeddings ----------------------------------------------------------------------
    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed one batch; returns an L2-normalised float32 matrix (cosine == inner product)."""
        resp = self._with_retries(
            lambda: self.client.embeddings.create(model=self.embed_model, input=texts,
                                                  encoding_format="float"),
            "embed")
        tokens = getattr(resp.usage, "prompt_tokens", None) or getattr(resp.usage, "total_tokens", 0) or 0
        self.meter.embed_calls += 1
        self.meter.embed_in += tokens
        mat = np.array([d.embedding for d in sorted(resp.data, key=lambda d: d.index)], dtype="float32")
        return normalise(mat)


def normalise(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (mat / norms).astype("float32")
