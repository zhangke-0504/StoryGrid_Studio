"""Shared event emitter for streaming sub-agent activity back to callers.

The supervisor SSE endpoint installs an emitter via :func:`set_event_emitter`
inside a `contextvars.ContextVar`. Sub-agents can then call
:func:`emit_event` to push their tool calls / tool results into the same
stream without coupling to the HTTP layer.
"""

from __future__ import annotations

import contextvars
from typing import Any, Awaitable, Callable, Optional

EventEmitter = Callable[[dict[str, Any]], Awaitable[None]]

_current_event_emitter: contextvars.ContextVar[Optional[EventEmitter]] = contextvars.ContextVar(
    "current_event_emitter",
    default=None,
)


def set_event_emitter(emitter: Optional[EventEmitter]) -> contextvars.Token:
    return _current_event_emitter.set(emitter)


def reset_event_emitter(token: contextvars.Token) -> None:
    _current_event_emitter.reset(token)


def get_event_emitter() -> Optional[EventEmitter]:
    return _current_event_emitter.get()


async def emit_event(event: dict[str, Any]) -> None:
    emitter = _current_event_emitter.get()
    if emitter is None:
        return
    await emitter(event)
