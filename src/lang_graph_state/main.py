import asyncio
from lang_graph_state.graph import build_graph


async def amain() -> None:
    graph = await build_graph()
    result = await graph.ainvoke(
        {"fs_req_id": "test-001"},
        config={"configurable": {"thread_id": "test-001"}},
    )
    print(result)


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()
