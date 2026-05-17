from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator

import httpx
from langgraph.graph import StateGraph, START, END

from lang_graph_state.checkpointer import make_checkpointer
from lang_graph_state.nodes.fetch_nodes import make_fetch_node
from lang_graph_state.nodes.synthesis_node import make_synthesis_node
from lang_graph_state.settings import Settings
from lang_graph_state.sources import ALL_SOURCES
from lang_graph_state.state import GraphState
from lang_graph_state.synthesis.parsing import parse_synthesis_response
from lang_graph_state.synthesis.prompt import build_synthesis_request
from lang_graph_state.oauth_identity import OAuthIdentity


@asynccontextmanager
async def build_graph(
    settings: Settings | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> AsyncIterator:
    s = settings or Settings()

    oauth_client = httpx.AsyncClient(transport=transport)
    source_clients = {
        "customer_profile": httpx.AsyncClient(base_url=s.customer_profile_url, transport=transport),
        "insights": httpx.AsyncClient(base_url=s.insights_url, transport=transport),
        "optimizer": httpx.AsyncClient(base_url=s.optimizer_url, transport=transport),
        "monte_carlo": httpx.AsyncClient(base_url=s.monte_carlo_url, transport=transport),
    }
    llm_client = httpx.AsyncClient(base_url=s.llm_gateway_url, transport=transport)

    identities = {
        "customer_profile": OAuthIdentity(s.oauth_token_url, s.customer_profile_user, s.customer_profile_pass, oauth_client),
        "insights": OAuthIdentity(s.oauth_token_url, s.insights_user, s.insights_pass, oauth_client),
        "optimizer": OAuthIdentity(s.oauth_token_url, s.optimizer_user, s.optimizer_pass, oauth_client),
        "monte_carlo": OAuthIdentity(s.oauth_token_url, s.monte_carlo_user, s.monte_carlo_pass, oauth_client),
        "llm_gateway": OAuthIdentity(s.oauth_token_url, s.llm_gateway_user, s.llm_gateway_pass, oauth_client),
    }

    try:
        await asyncio.gather(*(i.initialize() for i in identities.values()))

        builder = StateGraph(GraphState)
        for source in ALL_SOURCES:
            builder.add_node(
                source.name,
                make_fetch_node(source, source_clients[source.name], identities[source.name]),
            )
        builder.add_node(
            "synthesis",
            make_synthesis_node(
                llm_client,
                identities["llm_gateway"],
                s.llm_model_provider,
                s.llm_model_id,
                build_synthesis_request,
                parse_synthesis_response,
            ),
        )

        for source in ALL_SOURCES:
            builder.add_edge(START, source.name)
            builder.add_edge(source.name, "synthesis")
        builder.add_edge("synthesis", END)

        checkpointer = make_checkpointer(s)
        yield builder.compile(checkpointer=checkpointer)
    finally:
        await asyncio.gather(
            oauth_client.aclose(),
            llm_client.aclose(),
            *(c.aclose() for c in source_clients.values()),
            return_exceptions=True,
        )


async def run(graph, fs_req_id: str) -> dict:
    return await graph.ainvoke(
        {"fs_req_id": fs_req_id},
        config={"configurable": {"thread_id": fs_req_id}},
    )
