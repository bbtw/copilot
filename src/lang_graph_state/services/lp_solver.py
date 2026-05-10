import asyncio

from lang_graph_state.domain.models import LPResult, Profile


async def solve_lp(profile: Profile) -> LPResult:
    await asyncio.sleep(0)
    return LPResult()
