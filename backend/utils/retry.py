"""Async retry helper for tool-level resilience against transient network errors.

Wraps an async callable; retries on connection / timeout / 5xx style errors
raised by the openai SDK or the underlying httpx stack. Does not retry on
auth errors, validation errors, or any other deterministic failure.
"""
from __future__ import annotations

import asyncio
import functools
import logging
import random
from typing import Awaitable, Callable, Tuple, Type, TypeVar

import httpx
from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError

logger = logging.getLogger(__name__)

T = TypeVar("T")

DEFAULT_RETRY_EXCEPTIONS: Tuple[Type[BaseException], ...] = (
	APIConnectionError,
	APITimeoutError,
	RateLimitError,
	InternalServerError,
	httpx.ConnectError,
	httpx.ReadError,
	httpx.WriteError,
	httpx.RemoteProtocolError,
	httpx.PoolTimeout,
	httpx.ConnectTimeout,
	httpx.ReadTimeout,
)


def async_retry(
	max_attempts: int = 3,
	base_delay: float = 1.0,
	max_delay: float = 8.0,
	exceptions: Tuple[Type[BaseException], ...] = DEFAULT_RETRY_EXCEPTIONS,
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
	"""Retry the decorated async function up to ``max_attempts`` times.

	Uses exponential backoff with jitter: delay = min(max_delay, base_delay * 2**i) + jitter.
	"""

	def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
		@functools.wraps(func)
		async def wrapper(*args, **kwargs) -> T:
			last_exc: BaseException | None = None
			for attempt in range(1, max_attempts + 1):
				try:
					return await func(*args, **kwargs)
				except exceptions as exc:
					last_exc = exc
					if attempt >= max_attempts:
						logger.error(
							"%s failed after %d attempts: %s",
							func.__qualname__,
							attempt,
							exc,
						)
						raise
					delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
					delay += random.uniform(0, 0.5)
					logger.warning(
						"%s attempt %d/%d failed (%s); retrying in %.2fs",
						func.__qualname__,
						attempt,
						max_attempts,
						exc.__class__.__name__,
						delay,
					)
					await asyncio.sleep(delay)
			assert last_exc is not None
			raise last_exc

		return wrapper

	return decorator
