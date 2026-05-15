import asyncio
import time
import httpx

TOKEN_TTL_SECONDS = 900  # 15 minutes


class TokenManager:
    def __init__(self, token_url: str, username: str, password: str) -> None:
        self._token_url = token_url
        self._username = username
        self._password = password
        self._token: str | None = None
        self._expires_at: float = 0.0
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        await self._refresh()

    async def get_token(self) -> str:
        if time.monotonic() >= self._expires_at:
            async with self._lock:
                if time.monotonic() >= self._expires_at:
                    await self._refresh()
        return self._token  # type: ignore[return-value]

    async def _refresh(self) -> None:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self._token_url,
                data={"username": self._username, "password": self._password},
            )
            response.raise_for_status()
            self._token = response.json()["access_token"]
            self._expires_at = time.monotonic() + TOKEN_TTL_SECONDS
