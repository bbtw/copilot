from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import httpx
from pydantic import BaseModel

if TYPE_CHECKING:
    from lang_graph_state.state import GraphState


@dataclass(frozen=True)
class RequestSpec:
    method: str
    path: str
    json: dict[str, Any] | None = None
    params: dict[str, Any] | None = None
    headers: dict[str, str] | None = None


@dataclass(frozen=True)
class SourceDefinition:
    name: str
    result_model: type[BaseModel]
    build_request: Callable[[GraphState], RequestSpec]
    parse_response: Callable[[httpx.Response], BaseModel]
