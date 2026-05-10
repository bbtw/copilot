import logging
import inspect
from collections.abc import Awaitable, Callable
from time import perf_counter
from typing import Any

from lang_graph_state.domain.state import RetirementPlanState

logger = logging.getLogger(__name__)


def instrument_node(
    name: str,
    node: Callable[[RetirementPlanState], dict[str, Any]] | Callable[[RetirementPlanState], Awaitable[dict[str, Any]]],
) -> Callable[[RetirementPlanState], dict[str, Any]] | Callable[[RetirementPlanState], Awaitable[dict[str, Any]]]:
    """Wrap a node with lightweight logs while leaving LangGraph tracing intact."""

    if inspect.iscoroutinefunction(node):
        async def wrapped_async(state: RetirementPlanState) -> dict[str, Any]:
            logger.info("Starting node=%s", name)
            started = perf_counter()
            try:
                result = await node(state)
            except Exception:
                logger.exception("Failed node=%s elapsed=%.2fs", name, perf_counter() - started)
                raise
            logger.info(
                "Finished node=%s elapsed=%.2fs writes=%s",
                name,
                perf_counter() - started,
                sorted(result),
            )
            return result

        return wrapped_async

    def wrapped(state: RetirementPlanState) -> dict[str, Any]:
        logger.info("Starting node=%s", name)
        started = perf_counter()
        try:
            result = node(state)
        except Exception:
            logger.exception("Failed node=%s elapsed=%.2fs", name, perf_counter() - started)
            raise
        logger.info(
            "Finished node=%s elapsed=%.2fs writes=%s",
            name,
            perf_counter() - started,
            sorted(result),
        )
        return result

    return wrapped
