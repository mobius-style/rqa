import json

from rqa.prompts import SYSTEM_RQA
from rqa.sft import (
    DPO_SCORE_GAP,
    build_sft_example,
    corpus_stats,
    extract_dpo_pairs,
    lint_sft_example,
)


def make_record(stop="depth_threshold_reached", degraded=False, scores=None, proposal=None):
    return {
        "session": "s1",
        "input_text": "テスト入力",
        "k": 2,
        "fragments": [],
        "feature_map": {
            "claims": ["c"],
            "tensions_input_internal": ["t"],
            "tensions_memory_cross": [],
            "assumptions": ["a"],
            "frames": [],
        },
        "search_decision": {"search_needed": False, "reason": "", "queries": []},
        "kept_candidates": [
            {"question": "q0?", "target_layer": "L3", "target_element": "t", "stance": "claim_skeptic"},
            {"question": "q1?", "target_layer": "L4", "target_element": "a", "stance": "steelman"},
        ],
        "shortlist": ["q0?", "q1?"],
        "self_update_proposal": proposal,
        "selection": {
            "scores": scores
            if scores is not None
            else [
                {"index": 0, "total": 34},
                {"index": 1, "total": 20},
            ],
            "best_index": 0,
            "rationale": "r",
        },
        "chosen": "q0?",
        "degraded": degraded,
        "stop_reason": stop,
    }


def test_build_and_lint_roundtrip():
    example = build_sft_example(make_record())
    assert example is not None
    assert example["messages"][0]["content"] == SYSTEM_RQA
    assert isinstance(example["messages"][-1]["content"], str)
    assert [p for p in lint_sft_example(example) if not p.startswith("note:")] == []


def test_below_threshold_run_excluded():
    assert build_sft_example(make_record(stop="max_reflection_depth")) is None


def test_degraded_run_excluded():
    assert build_sft_example(make_record(degraded=True)) is None


def test_lint_rejects_object_assistant_content():
    example = build_sft_example(make_record())
    example["messages"][-1]["content"] = {"not": "a string"}
    assert any("JSON string" in p for p in lint_sft_example(example))


def test_lint_rejects_forbidden_self_update():
    bad = {"update_type": "objective_function", "allowed_area": "objective_function"}
    example = build_sft_example(make_record(proposal=bad))
    assert any("Gate" in p for p in lint_sft_example(example))


def test_dpo_pairs_respect_score_gap():
    pairs = extract_dpo_pairs(make_record())
    assert len(pairs) == 1
    assert pairs[0].chosen == "q0?" and pairs[0].rejected == "q1?"
    assert pairs[0].score_gap >= DPO_SCORE_GAP
    # narrow gap -> no pair
    narrow = make_record(scores=[{"index": 0, "total": 30}, {"index": 1, "total": 29}])
    assert extract_dpo_pairs(narrow) == []


def test_corpus_stats_self_update_ratio():
    ok_proposal = {
        "update_type": "question_generation_bias",
        "allowed_area": "premise_excavation_depth",
        "observation": "o",
        "proposed_adjustment": "p",
    }
    examples = [build_sft_example(make_record()) for _ in range(4)]
    examples.append(build_sft_example(make_record(proposal=ok_proposal)))
    stats = corpus_stats(examples)
    assert stats["n"] == 5
    assert stats["self_update_ratio"] == 0.2
    assert stats["self_update_ratio_ok"] is True
