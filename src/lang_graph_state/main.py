import asyncio

from lang_graph_state.graph import build_graph, run


async def amain() -> None:
    async with build_graph() as graph:
        result = await run(graph, fs_req_id="test-001")
        print(result)


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()
