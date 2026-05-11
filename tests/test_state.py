from lang_graph_state.domain.models import AnalysisSection, merge_analysis_sections


def test_merge_deduplicates_by_kind():
    current = [AnalysisSection(kind="standard_explanation", content="old")]
    updates = [
        AnalysisSection(kind="contribution_explanation", content="b"),
        AnalysisSection(kind="standard_explanation", content="new"),
    ]
    merged = merge_analysis_sections(current, updates)
    assert [(s.kind, s.content) for s in merged] == [
        ("standard_explanation", "new"),
        ("contribution_explanation", "b"),
    ]


def test_merge_preserves_canonical_order():
    sections = [
        AnalysisSection(kind="withdrawal_explanation", content="c"),
        AnalysisSection(kind="standard_explanation", content="a"),
        AnalysisSection(kind="contribution_explanation", content="b"),
    ]
    merged = merge_analysis_sections([], sections)
    assert [s.kind for s in merged] == [
        "standard_explanation",
        "contribution_explanation",
        "withdrawal_explanation",
    ]


def test_merge_handles_empty_inputs():
    assert merge_analysis_sections(None, None) == []
    assert merge_analysis_sections([], []) == []
