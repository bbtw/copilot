import inspect
import logging
from collections.abc import Awaitable, Callable
from functools import wraps
from time import perf_counter
from typing import Any, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")

FinishAttrs = Callable[[Any], dict[str, Any]]


def timed(
    name: str,
    *,
    finish_attrs: FinishAttrs | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Wrap a callable with start/finish/failure logs.

    Emits three log shapes via the wrapped function's module logger:
      - `<name> start`
      - `<name> finish elapsed=<seconds>s [k=v ...]` on success
      - `<name> failed elapsed=<seconds>s` on exception (re-raised)

    Pass `finish_attrs=lambda result: {...}` to log result-derived fields
    on success. Sync and async callables are both supported.
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        logger = logging.getLogger(func.__module__)

        if inspect.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
                logger.info("%s start", name)
                started = perf_counter()
                try:
                    result = await func(*args, **kwargs)  # type: ignore[misc]
                except Exception:
                    logger.exception("%s failed elapsed=%.2fs", name, perf_counter() - started)
                    raise
                logger.info(
                    "%s finish elapsed=%.2fs%s",
                    name,
                    perf_counter() - started,
                    _format_attrs(finish_attrs, result),
                )
                return result

            return async_wrapped  # type: ignore[return-value]

        @wraps(func)
        def sync_wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
            logger.info("%s start", name)
            started = perf_counter()
            try:
                result = func(*args, **kwargs)
            except Exception:
                logger.exception("%s failed elapsed=%.2fs", name, perf_counter() - started)
                raise
            logger.info(
                "%s finish elapsed=%.2fs%s",
                name,
                perf_counter() - started,
                _format_attrs(finish_attrs, result),
            )
            return result

        return sync_wrapped

    return decorator


def _format_attrs(extractor: FinishAttrs | None, result: Any) -> str:
    if extractor is None:
        return ""
    attrs = extractor(result)
    if not attrs:
        return ""
    return " " + " ".join(f"{k}={v}" for k, v in attrs.items())
