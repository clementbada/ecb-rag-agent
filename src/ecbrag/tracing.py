"""Optional Langfuse tracing. Without keys, everything still runs and is logged locally."""
from __future__ import annotations

from .llm import langfuse_enabled


def traced(fn, name: str):
    """Wrap a function with Langfuse's @observe when Langfuse is configured."""
    if not langfuse_enabled():
        return fn
    try:
        from langfuse import observe  # langfuse >= 3
    except ImportError:
        try:
            from langfuse.decorators import observe  # langfuse 2.x
        except ImportError:
            return fn
    return observe(name=name)(fn)


def flush() -> None:
    if not langfuse_enabled():
        return
    try:
        from langfuse import get_client

        get_client().flush()
    except Exception:  # noqa: BLE001
        try:
            from langfuse.decorators import langfuse_context

            langfuse_context.flush()
        except Exception:  # noqa: BLE001
            pass
