import time
from collections.abc import Callable
from typing import TypeVar

import requests
from langgraph.types import RetryPolicy

T = TypeVar("T")

_TRANSIENT_TYPES: tuple[type[BaseException], ...] = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
)

try:
    import openai

    _TRANSIENT_TYPES += (
        openai.APIConnectionError,
        openai.APITimeoutError,
        openai.RateLimitError,
        openai.InternalServerError,
    )
except ImportError:  # pragma: no cover
    pass

try:
    from tavily.errors import TimeoutError as TavilyTimeoutError

    _TRANSIENT_TYPES += (TavilyTimeoutError,)
except ImportError:  # pragma: no cover
    pass


def is_transient_error(exc: BaseException) -> bool:
    """Return True for errors that are worth retrying.

    Tools wrap low-level errors in RuntimeError, so the whole exception
    chain (``raise ... from exc``) is inspected. Programming errors and
    validation errors are never retried.
    """

    seen = 0

    while exc is not None and seen < 6:
        if isinstance(exc, _TRANSIENT_TYPES):
            return True

        if isinstance(exc, requests.exceptions.HTTPError):
            status = getattr(exc.response, "status_code", None)

            if status is not None and (status == 429 or status >= 500):
                return True

        exc = exc.__cause__ or exc.__context__
        seen += 1

    return False


# Used by LangGraph for nodes whose failure should stop the workflow.
RETRY_POLICY = RetryPolicy(
    max_attempts=3,
    initial_interval=1.0,
    backoff_factor=2.0,
    retry_on=is_transient_error,
)


def call_with_retries(
    fn: Callable[..., T],
    *args,
    attempts: int = 3,
    base_delay: float = 1.0,
    sleep: Callable[[float], None] | None = None,
    **kwargs,
) -> T:
    """Call ``fn`` and retry transient failures with exponential backoff.

    Used inside nodes that degrade gracefully instead of failing the graph.
    """

    for attempt in range(1, attempts + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            if attempt == attempts or not is_transient_error(exc):
                raise

            (sleep or time.sleep)(base_delay * 2 ** (attempt - 1))

    raise RuntimeError("unreachable")  # pragma: no cover
