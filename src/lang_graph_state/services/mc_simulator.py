import asyncio

from lang_graph_state.domain.models import LPResult, MCResult, Profile


async def simulate(profile: Profile, lp_result: LPResult) -> MCResult:
    await asyncio.sleep(0)
    return MCResult()
