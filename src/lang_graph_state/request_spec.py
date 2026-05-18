from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class RequestState(Protocol):
    fs_req_id: str


@dataclass(frozen=True)
class RequestSpec:
    method: str
    path: str
    json: dict[str, Any] | None = None
    params: dict[str, Any] | None = None
    headers: dict[str, str] | None = None
