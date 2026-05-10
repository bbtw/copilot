import io
import logging

from lang_graph_state.instrumentation.logging import configure_logging


def test_configure_logging_formats_plain_human_readable_lines(monkeypatch):
    stream = io.StringIO()
    monkeypatch.setenv("LOG_COLOR", "never")

    configure_logging(level="INFO", stream=stream)
    logging.getLogger("lang_graph_state.nodes.load_profile").info("Loaded profile")

    output = stream.getvalue()
    assert "info" in output
    assert "nodes.load_profile" in output
    assert "Loaded profile" in output
    assert "\033[" not in output


def test_configure_logging_can_force_color(monkeypatch):
    stream = io.StringIO()
    monkeypatch.setenv("LOG_COLOR", "always")

    configure_logging(level="INFO", stream=stream)
    logging.getLogger("lang_graph_state.main").warning("Gateway unavailable")

    output = stream.getvalue()
    assert "\033[" in output
    assert "warning" in output
    assert "Gateway unavailable" in output


def test_configure_logging_honors_level(monkeypatch):
    stream = io.StringIO()
    monkeypatch.setenv("LOG_COLOR", "never")

    configure_logging(level="WARNING", stream=stream)
    logger = logging.getLogger("lang_graph_state.main")
    logger.info("hidden")
    logger.warning("visible")

    output = stream.getvalue()
    assert "hidden" not in output
    assert "visible" in output
