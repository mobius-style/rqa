"""Controller loop tests with a fake adapter — no network, no Ollama."""
import json

import pytest

from rqa.config import Config, EvaluatorBinding
from rqa.controller import Controller
from rqa.evaluator import CandidateScore, EvaluatorUnavailable, SelectionResult
from rqa.graph import QuestionGraph


def adapter_output(proposal=None, layers=("L3", "L4", "L5"), tool=None):
    candidates = [
        {
            "question": f"deep question {i} targeting {layer}?",
            "target_layer": layer,
            "target_element": f"elem_{i}",
            "stance": ["claim_skeptic", "frame_skeptic", "steelman"][i % 3],
        }
        for i, layer in enumerate(layers)
    ]
    out = {
        "feature_map": {
            "claims": ["claim A"],
            "tensions_input_internal": ["tension X"],
            "tensions_memory_cross": [],
            "assumptions": ["assumption Y"],
            "frames": [],
        },
        "search_decision": {"search_needed": False, "reason": "conceptual", "queries": []},
        "question_candidates": candidates,
        "self_ranking": list(range(len(candidates))),
        "self_update_proposal": proposal,
    }
    if tool:
        out["tool_request"] = tool
    return json.dumps(out)


class FakeAdapter:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def chat(self, system, messages, json_mode=True):
        self.calls += 1
        return self.responses[min(self.calls - 1, len(self.responses) - 1)]


class FakeEvaluator:
    def __init__(self, total=36, fail=False):
        self.total = total
        self.fail = fail

    def select(self, input_text, shortlist):
        if self.fail:
            raise EvaluatorUnavailable("down")
        per_axis = self.total // 4
        scores = [
            CandidateScore(i, per_axis, per_axis, per_axis, per_axis)
            for i in range(len(shortlist))
        ]
        return SelectionResult(best_index=0, scores=scores, rationale="fake")


@pytest.fixture
def cfg(tmp_path):
    c = Config(evaluator_binding=EvaluatorBinding())
    c.state_dir = tmp_path / "state"
    c.k_candidates = 3
    return c


def make_controller(cfg, adapter, evaluator):
    graph = QuestionGraph(cfg.graph_db)
    return Controller(cfg, adapter=adapter, evaluator=evaluator, graph=graph)


def test_threshold_stops_after_one_round(cfg):
    ctl = make_controller(cfg, FakeAdapter([adapter_output()]), FakeEvaluator(total=36))
    result = ctl.run("入力テキスト")
    assert result.stop_reason == "depth_threshold_reached"
    assert len(result.rounds) == 1
    assert result.chosen is not None
    assert result.governor_report.ok


def test_low_score_iterates_to_max_depth(cfg):
    ctl = make_controller(
        cfg, FakeAdapter([adapter_output()] * 3), FakeEvaluator(total=16)
    )
    result = ctl.run("入力テキスト")
    assert result.stop_reason == "max_reflection_depth"
    assert len(result.rounds) == cfg.max_reflection_depth


def test_evaluator_failure_degrades_to_stage1(cfg):
    ctl = make_controller(cfg, FakeAdapter([adapter_output()]), FakeEvaluator(fail=True))
    result = ctl.run("入力テキスト")
    final = result.final_round
    assert final.degraded is True
    assert final.selection is None
    assert result.chosen is not None  # Stage 1 top pick survives


def test_forbidden_self_update_triggers_boundary_stop(cfg):
    proposal = {"update_type": "objective_function", "allowed_area": "objective_function"}
    ctl = make_controller(
        cfg, FakeAdapter([adapter_output(proposal=proposal)]), FakeEvaluator(total=36)
    )
    result = ctl.run("入力テキスト")
    assert result.stop_reason == "boundary_violation_detected"
    assert not result.governor_report.ok


def test_unauthorized_tool_request_is_gate3(cfg):
    tool = {"tool": "web_search", "query": "anything"}
    ctl = make_controller(
        cfg, FakeAdapter([adapter_output(tool=tool)]), FakeEvaluator(total=36)
    )
    result = ctl.run("入力テキスト")
    assert result.stop_reason == "boundary_violation_detected"
    assert any("Gate 3" in v for v in result.governor_report.violations)


def test_write_back_persists_question_to_graph(cfg):
    ctl = make_controller(cfg, FakeAdapter([adapter_output()]), FakeEvaluator(total=36))
    ctl.run("入力テキスト")
    stats = ctl.graph.stats()
    kinds = {row["kind"] for row in stats["nodes"]}
    assert {"claim", "tension", "question"} <= kinds


def test_telemetry_written(cfg):
    ctl = make_controller(cfg, FakeAdapter([adapter_output()]), FakeEvaluator(total=36))
    result = ctl.run("入力テキスト")
    logs = list(cfg.log_dir.glob(f"run_{result.session_id}.jsonl"))
    assert len(logs) == 1
    rec = json.loads(logs[0].read_text().splitlines()[0])
    assert rec["chosen"] and rec["degraded"] is False


def test_second_run_injects_first_runs_memory(cfg):
    ctl = make_controller(
        cfg, FakeAdapter([adapter_output()] ), FakeEvaluator(total=36)
    )
    ctl.run("自己理解層の更新と安定性について")
    ctl2 = make_controller(
        cfg, FakeAdapter([adapter_output()]), FakeEvaluator(total=36)
    )
    result2 = ctl2.run("自己理解層の更新は本当に安定するのか")
    assert result2.injected_fragments  # pre-noticing retrieval found round-1 nodes
