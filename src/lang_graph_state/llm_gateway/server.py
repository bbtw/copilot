import logging
import os
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

_UPSTREAM_TIMEOUT_SECONDS = 300.0


@dataclass(frozen=True)
class GatewaySettings:
    upstream_base_url: str = "http://localhost:11434/v1"
    upstream_api_key: str = "ollama"
    host: str = "127.0.0.1"
    port: int = 8001

    @classmethod
    def from_env(cls) -> "GatewaySettings":
        return cls(
            upstream_base_url=os.environ.get("LOCAL_LLM_UPSTREAM_BASE_URL", cls.upstream_base_url),
            upstream_api_key=os.environ.get("LOCAL_LLM_UPSTREAM_API_KEY", cls.upstream_api_key),
            host=os.environ.get("LOCAL_LLM_GATEWAY_HOST", cls.host),
            port=int(os.environ.get("LOCAL_LLM_GATEWAY_PORT", str(cls.port))),
        )

    @property
    def upstream_chat_completions_url(self) -> str:
        return f"{self.upstream_base_url.rstrip('/')}/chat/completions"


def create_app(
    settings: GatewaySettings | None = None,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    settings = settings or GatewaySettings.from_env()
    app = FastAPI(title="Local LLM Gateway")

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {
            "status": "ok",
            "gateway": {
                "host": settings.host,
                "port": settings.port,
            },
            "upstream": {
                "base_url": settings.upstream_base_url,
                "auth_configured": bool(settings.upstream_api_key),
            },
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(body: dict[str, Any]) -> Any:
        model = str(body.get("model") or "")
        started = perf_counter()
        status_code = 500
        logger.info("Local LLM gateway request start model=%s", model or "<missing>")

        try:
            if body.get("stream") is True:
                raise HTTPException(status_code=400, detail="stream=true is not supported by the local LLM gateway")

            async with httpx.AsyncClient(
                transport=transport,
                timeout=_UPSTREAM_TIMEOUT_SECONDS,
            ) as client:
                response = await client.post(
                    settings.upstream_chat_completions_url,
                    json=body,
                    headers={"Authorization": f"Bearer {settings.upstream_api_key}"},
                )
                status_code = response.status_code
                response.raise_for_status()
                return JSONResponse(content=response.json(), status_code=response.status_code)
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code
            raise HTTPException(status_code=status_code, detail=_upstream_error_detail(exc.response)) from exc
        except httpx.RequestError as exc:
            status_code = 502
            raise HTTPException(status_code=502, detail="Upstream LLM gateway is unavailable") from exc
        except HTTPException as exc:
            status_code = exc.status_code
            raise
        finally:
            logger.info(
                "Local LLM gateway request finish model=%s status=%s elapsed=%.2fs",
                model or "<missing>",
                status_code,
                perf_counter() - started,
            )

    return app


def _upstream_error_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        text = response.text.strip()
        return text[:500] or f"Upstream LLM gateway returned HTTP {response.status_code}"

    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if message:
                return str(message)
        detail = payload.get("detail")
        if detail:
            return str(detail)
        message = payload.get("message")
        if message:
            return str(message)

    return f"Upstream LLM gateway returned HTTP {response.status_code}"
