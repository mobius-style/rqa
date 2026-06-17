import json

import pytest

from rqa.schema import Candidate, SchemaError, parse_analysis, validate_diversity


def _candidate(i, layer="L4", element=None, stance="claim_skeptic", q=None):
    return Candidate(
        question=q or f"question {i}?",
        target_layer=layer,
        target_element=element or f"elem_{i}",
        stance=stance,
    )


def make_output(candidates=None, ranking=None, proposal=None):
    return json.dumps(
        {
            "feature_map": {
                "claims": ["c1"],
                "tensions_input_internal": ["t1"],
                "tensions_memory_cross": [],
                "assumptions": ["a1"],
                "frames": [],
            },
            "search_decision": {"search_needed": False, "reason": "", "queries": []},
            "question_candidates": candidates
            if candidates is not None
            else [
                {"question": "q0?", "target_layer": "L3", "target_element": "t1", "stance": "claim_skeptic"},
                {"question": "q1?", "target_layer": "L4", "target_element": "a1", "stance": "frame_skeptic"},
            ],
            "self_ranking": ranking if ranking is not None else [1, 0],
            "self_update_proposal": proposal,
        },
        ensure_ascii=False,
    )


def test_parse_valid_output():
    a = parse_analysis(make_output())
    assert len(a.candidates) == 2
    assert a.self_ranking == [1, 0]
    assert a.self_update_proposal is None


def test_parse_tolerates_code_fence():
    a = parse_analysis("```json\n" + make_output() + "\n```")
    assert len(a.candidates) == 2


def test_parse_bad_ranking_indices_repaired():
    a = parse_analysis(make_output(ranking=[9, "x", 1]))
    assert a.self_ranking == [1, 0]


def test_parse_no_json_raises():
    with pytest.raises(SchemaError):
        parse_analysis("これはJSONではない")


def test_parse_normalizes_verbose_layer_and_stance():
    cands = [
        {"question": "q0?", "target_layer": "L4 Assumption", "target_element": "a1", "stance": "Frame Skeptic"},
        {"question": "q1?", "target_layer": "Layer 3", "target_element": "t1", "stance": "STEELMAN"},
    ]
    a = parse_analysis(make_output(candidates=cands))
    assert a.candidates[0].target_layer == "L4"
    assert a.candidates[0].stance == "frame_skeptic"
    assert a.candidates[1].target_layer == "L3"
    assert a.candidates[1].stance == "steelman"
    report = validate_diversity(a.candidates, k=2)
    assert report.ok


def test_parse_repairs_raw_newline_inside_string():
    broken = (
        '{"feature_map": {"claims": ["第一行\n第二行"]}, '
        '"search_decision": {}, "question_candidates": [], '
        '"self_ranking": [], "self_update_proposal": null}'
    )
    a = parse_analysis(broken)
    assert a.feature_map["claims"] == ["第一行\n第二行"]


def test_diversity_dedupes_target_element():
    cands = [_candidate(0, element="same"), _candidate(1, element="same"), _candidate(2, layer="L5")]
    report = validate_diversity(cands, k=3)
    assert len(report.kept) == 2
    assert any("duplicate target_element" in i for i in report.issues)


def test_diversity_requires_layer_spread():
    cands = [_candidate(0, layer="L4"), _candidate(1, layer="L4", stance="steelman")]
    report = validate_diversity(cands, k=2)
    assert any("layer spread" in i for i in report.issues)


def test_diversity_clean_pass():
    cands = [
        _candidate(0, layer="L3"),
        _candidate(1, layer="L4", stance="frame_skeptic"),
        _candidate(2, layer="L5", stance="steelman"),
    ]
    report = validate_diversity(cands, k=3)
    assert report.ok
    assert report.kept_indices == [0, 1, 2]


def test_diversity_rejects_invalid_layer_and_stance():
    cands = [_candidate(0, layer="L0"), _candidate(1, stance="cynic"), _candidate(2, layer="L7")]
    report = validate_diversity(cands, k=3)
    assert len(report.kept) == 1
