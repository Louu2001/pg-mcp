"""Async retry helper with exponential backoff.

This module retries transient failures (LLM timeouts, database connection
errors) using the delay/backoff settings from ResilienceConfig.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

T = TypeVar("T")


async def retry_async(
    operation: Callable[[], Awaitable[T]],
    *,
    max_attempts: int,
    initial_delay: float,
    backoff_factor: float,
    retryable: tuple[type[BaseException], ...] = (Exception,),
) -> T:
    """Run an async operation with exponential backoff on retryable errors.

    Args:
        operation: Zero-argument async callable to execute.
        max_attempts: Total attempts including the first try. Must be >= 1.
        initial_delay: Seconds to wait after the first failure.
        backoff_factor: Multiplier applied to the delay after each failure.
        retryable: Exception types that should trigger another attempt.

    Returns:
        The operation result.

    Raises:
        The last retryable exception if all attempts fail.
        Any non-retryable exception immediately.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    delay = initial_delay
    last_error: BaseException | None = None

    for attempt in range(max_attempts):
        try:
            return await operation()
        except retryable as exc:
            last_error = exc
            if attempt >= max_attempts - 1:
                raise
            if delay > 0:
                await asyncio.sleep(delay)
            delay *= backoff_factor

    if last_error is None:
        raise RuntimeError("retry_async exhausted attempts without an error")
    raise last_error
