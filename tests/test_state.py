from lang_graph_state.domain.models import AnalysisSection, merge_analysis_sections


def test_merge_deduplicates_by_kind():
    current = [AnalysisSection(kind="section_a", content="old")]
    updates = [
        AnalysisSection(kind="section_b", content="b"),
        AnalysisSection(kind="section_a", content="new"),
    ]
    merged = merge_analysis_sections(current, updates)
    assert [(s.kind, s.content) for s in merged] == [
        ("section_a", "new"),
        ("section_b", "b"),
    ]


def test_merge_preserves_canonical_order():
    sections = [
        AnalysisSection(kind="section_c", content="c"),
        AnalysisSection(kind="section_a", content="a"),
        AnalysisSection(kind="section_b", content="b"),
    ]
    merged = merge_analysis_sections([], sections)
    assert [s.kind for s in merged] == ["section_a", "section_b", "section_c"]


def test_merge_handles_empty_inputs():
    assert merge_analysis_sections(None, None) == []
    assert merge_analysis_sections([], []) == []
