from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

import httpx
from pydantic import BaseModel


class RequestState(Protocol):
    fs_req_id: str


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
    build_request: Callable[[RequestState], RequestSpec]
    parse_response: Callable[[httpx.Response], BaseModel]
