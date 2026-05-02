from main import route_by_confidence_band


def test_route_by_confidence_band_uses_monte_carlo_band():
    assert route_by_confidence_band({"confidence_band": "low"}) == "low"
    assert route_by_confidence_band({"confidence_band": "medium"}) == "medium"
    assert route_by_confidence_band({"confidence_band": "high"}) == "high"
