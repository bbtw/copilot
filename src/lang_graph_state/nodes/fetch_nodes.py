from typing import Callable, Awaitable
import httpx

from lang_graph_state.errors import SourceFetchError
from lang_graph_state.state import (
    GraphState,
    CustomerProfileResult,
    InsightsResult,
    OptimizerResult,
    MonteCarloResult,
)
from lang_graph_state.token_manager import TokenManager

Node = Callable[[GraphState], Awaitable[dict]]


def make_customer_profile_node(client: httpx.AsyncClient, token_manager: TokenManager) -> Node:
    async def node(state: GraphState) -> dict:
        try:
            token = await token_manager.get_token()
            # TODO: replace with real endpoint and parsing
            _ = token
            return {"customer_profile": CustomerProfileResult()}
        except Exception as e:
            raise SourceFetchError("customer_profile", e) from e
    return node


def make_insights_node(client: httpx.AsyncClient, token_manager: TokenManager) -> Node:
    async def node(state: GraphState) -> dict:
        try:
            token = await token_manager.get_token()
            _ = token
            return {"insights": InsightsResult()}
        except Exception as e:
            raise SourceFetchError("insights", e) from e
    return node


def make_optimizer_node(client: httpx.AsyncClient, token_manager: TokenManager) -> Node:
    async def node(state: GraphState) -> dict:
        try:
            token = await token_manager.get_token()
            _ = token
            return {"optimizer": OptimizerResult()}
        except Exception as e:
            raise SourceFetchError("optimizer", e) from e
    return node


def make_monte_carlo_node(client: httpx.AsyncClient, token_manager: TokenManager) -> Node:
    async def node(state: GraphState) -> dict:
        try:
            token = await token_manager.get_token()
            _ = token
            return {"monte_carlo": MonteCarloResult()}
        except Exception as e:
            raise SourceFetchError("monte_carlo", e) from e
    return node
