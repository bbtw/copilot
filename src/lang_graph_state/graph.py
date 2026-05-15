import asyncio
import httpx

from langgraph.graph import StateGraph, START, END

from lang_graph_state.checkpointer import make_checkpointer
from lang_graph_state.settings import Settings
from lang_graph_state.state import GraphState
from lang_graph_state.token_manager import TokenManager
from lang_graph_state.nodes.fetch_nodes import (
    make_customer_profile_node,
    make_insights_node,
    make_optimizer_node,
    make_monte_carlo_node,
)
from lang_graph_state.nodes.synthesis_node import make_synthesis_node


async def build_graph(settings: Settings | None = None):
    s = settings or Settings()

    managers = {
        "customer_profile": TokenManager(s.oauth_token_url, s.customer_profile_user, s.customer_profile_pass),
        "insights": TokenManager(s.oauth_token_url, s.insights_user, s.insights_pass),
        "optimizer": TokenManager(s.oauth_token_url, s.optimizer_user, s.optimizer_pass),
        "monte_carlo": TokenManager(s.oauth_token_url, s.monte_carlo_user, s.monte_carlo_pass),
        "llm_gateway": TokenManager(s.oauth_token_url, s.llm_gateway_user, s.llm_gateway_pass),
    }

    await asyncio.gather(*[m.initialize() for m in managers.values()])

    client = httpx.AsyncClient()

    builder = StateGraph(GraphState)

    builder.add_node("customer_profile", make_customer_profile_node(client, managers["customer_profile"]))
    builder.add_node("insights", make_insights_node(client, managers["insights"]))
    builder.add_node("optimizer", make_optimizer_node(client, managers["optimizer"]))
    builder.add_node("monte_carlo", make_monte_carlo_node(client, managers["monte_carlo"]))
    builder.add_node("synthesis", make_synthesis_node(
        client,
        managers["llm_gateway"],
        s.llm_model_provider,
        s.llm_model_id,
    ))

    builder.add_edge(START, "customer_profile")
    builder.add_edge(START, "insights")
    builder.add_edge(START, "optimizer")
    builder.add_edge(START, "monte_carlo")

    builder.add_edge("customer_profile", "synthesis")
    builder.add_edge("insights", "synthesis")
    builder.add_edge("optimizer", "synthesis")
    builder.add_edge("monte_carlo", "synthesis")

    builder.add_edge("synthesis", END)

    checkpointer = make_checkpointer(s)
    return builder.compile(checkpointer=checkpointer)
