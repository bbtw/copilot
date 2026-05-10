import asyncio
import logging

import pytest

from lang_graph_state.instrumentation.timing import timed


def test_timed_emits_start_and_finish_logs(caplog):
    @timed("widget_build")
    def build():
        return {"a": 1, "b": 2}

    with caplog.at_level(logging.INFO):
        build()

    messages = [record.message for record in caplog.records]
    assert any("widget_build start" in m for m in messages)
    assert any("widget_build finish elapsed=" in m for m in messages)


def test_timed_includes_finish_attrs_in_finish_log(caplog):
    @timed("widget_build", finish_attrs=lambda result: {"writes": sorted(result)})
    def build():
        return {"b": 2, "a": 1}

    with caplog.at_level(logging.INFO):
        build()

    finish = next(m for m in (r.message for r in caplog.records) if "finish" in m)
    assert "writes=['a', 'b']" in finish


def test_timed_logs_failure_and_reraises(caplog):
    @timed("widget_build")
    def build():
        raise ValueError("boom")

    with caplog.at_level(logging.INFO):
        with pytest.raises(ValueError, match="boom"):
            build()

    messages = [record.message for record in caplog.records]
    assert any("widget_build start" in m for m in messages)
    assert any("widget_build failed elapsed=" in m for m in messages)
    assert not any("finish" in m for m in messages)


def test_timed_supports_async_callables(caplog):
    @timed("widget_build_async", finish_attrs=lambda result: {"output_chars": len(result)})
    async def build():
        await asyncio.sleep(0)
        return "hello"

    with caplog.at_level(logging.INFO):
        result = asyncio.run(build())

    assert result == "hello"
    finish = next(m for m in (r.message for r in caplog.records) if "finish" in m)
    assert "widget_build_async finish elapsed=" in finish
    assert "output_chars=5" in finish


def test_timed_logs_under_wrapped_function_module():
    @timed("widget_build")
    def build():
        return None

    assert build.__wrapped__.__module__ == __name__
