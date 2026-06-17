"""v0.3 tests: memory-ref provenance, conversational surface logic."""
import json
import time
from types import SimpleNamespace

import pytest

from rqa.__main__ import _apply_budget, _instrument_budget, _run_with_progress, cmd_ask
from rqa import governor
from rqa.chat import (
    canned_light_reply,
    canned_missing_target_reply,
    input_weight,
    rgc_route,
    should_surface,
)
from rqa.config import Config, EvaluatorBinding
from rqa.controller import Controller
from rqa.evaluator import CandidateScore, SelectionResult
from rqa.graph import QuestionGraph
from rqa.render import render_brief

from test_controller import FakeAdapter, FakeEvaluator, adapter_output


# -- memory-ref provenance (§5.4 / Gate fix) --------------------------------

def test_sanitize_strips_unknown_refs():
    fm = {
        "tensions_memory_cross": [
            {"tension": "real", "memory_ref": "54"},
            {"tension": "real2", "memory_ref": "node 12"},
            {"tension": "fabricated", "memory_ref": "2025-01-28 14:37:09 [K=6]"},
            {"tension": "fabricated2", "memory_ref": "99"},
        ]
    }
    clean, stripped = governor.sanitize_memory_refs(fm, allowed_node_ids={54, 12})
    assert [t["tension"] for t in clean["tensions_memory_cross"]] == ["real", "real2"]
    assert len(stripped) == 2


def test_sanitize_empty_allowed_strips_all():
    fm = {"tensions_memory_cross": [{"tension": "x", "memory_ref": "3"}]}
    clean, stripped = governor.sanitize_memory_refs(fm, allowed_node_ids=set())
    assert clean["tensions_memory_cross"] == []
    assert stripped == ["3"]


def fabricated_memory_output():
    out = json.loads(adapter_output())
    out["feature_map"]["tensions_memory_cross"] = [
        {"tension": "捏造", "memory_ref": "2025-01-28 14:37:09", "confidence": "high"}
    ]
    return json.dumps(out, ensure_ascii=False)


def test_controller_strips_fabricated_refs(tmp_path):
    cfg = Config(evaluator_binding=EvaluatorBinding())
    cfg.state_dir = tmp_path / "state"
    cfg.k_candidates = 3
    ctl = Controller(
        cfg,
        adapter=FakeAdapter([fabricated_memory_output()]),
        evaluator=FakeEvaluator(total=36),
        graph=QuestionGraph(cfg.graph_db),
    )
    result = ctl.run("入力")  # empty graph -> nothing injected -> all refs fabricated
    assert result.memory_refs_stripped == ["2025-01-28 14:37:09"]
    assert result.final_round.analysis.feature_map["tensions_memory_cross"] == []


# -- conversational surface --------------------------------------------------

def test_input_weight_light_cases():
    for text in (
        "こんにちは",
        "ありがとう!",
        "OK",
        "了解です",
        "hi",
        "うん",
        "あなたは何者ですか",
        "あなたにとっての幸せとは何か",
        "何ができますか",
    ):
        assert input_weight(text) == "light", text


def test_input_weight_normal_cases():
    for text in (
        "評価器を凍結すれば自己更新は自動化してよいのではないか",
        "What is the difference between phonetics and phonology?",
        "MOBIUS-RQAとして、前回の仕様との関係であなたは何者ですか",
        "あなたにとっての幸せとは何かを論文として分析して",
    ):
        assert input_weight(text) == "normal", text


def test_canned_light_identity_reply_is_product_face():
    reply = canned_light_reply("あなたは何者ですか")
    assert "MOBIUS-RQA" in reply
    assert "Gemma" in reply
    assert "単なるモデル名ではなく" in reply


def test_canned_light_reply_skips_deep_context():
    assert canned_light_reply("MOBIUS-RQAとして、前回の仕様との関係であなたは何者ですか") is None


def test_canned_light_happiness_reply_is_non_anthropomorphic():
    reply = canned_light_reply("あなたにとっての幸せとは何か")
    assert "主観的な幸福" in reply
    assert "よい状態" in reply


def test_canned_light_greeting_reply():
    assert canned_light_reply("こんにちは") == "こんにちは。今日はどこから始めましょうか。"


def test_missing_target_reply_blocks_graph_guessing():
    reply = canned_missing_target_reply("この仕様の前提を見て")
    assert "対象がまだこちらに渡っていません" in reply
    assert "bin/rqa review" in reply
    assert "対象がまだこちらに渡っていません" in canned_missing_target_reply("この案の前提を見て")


# Gate-D recalibration (2026-06-13): an open conceptual question is a deepening
# OPPORTUNITY, so it reaches the loop (L2 answer-first + sidecar). The system
# decides whether to volunteer a question; depth is NOT withheld until the user
# explicitly asks for reflection (that tuning stranded the adapter — 40/47
# strong questions diverted; see docs/GATE_D_VERDICT.md). Language-symmetric.
def test_rgc_route_open_conceptual_question_reaches_loop_ja():
    route = rgc_route("音声学と音韻論の違いは？")
    assert route.level == 2
    assert route.sidecar is True


def test_rgc_route_open_conceptual_question_reaches_loop_en():
    route = rgc_route("How does phonetics differ from phonology?")
    assert route.level == 2
    assert route.sidecar is True


def test_rgc_route_closed_lookup_stays_direct():
    # closed factual lookups correctly stay L1 (no loop)
    for q in ("ナイロビはどの国にある?", "What country is Nairobi in?", "炭素の原子番号は?"):
        assert rgc_route(q).level == 1, q


def test_rgc_route_guided_for_soft_reflection():
    route = rgc_route("RQAの応答設計の前提を軽く見て")
    assert route.level == 2
    assert route.mode == "guided"
    assert route.sidecar is True


def test_rgc_route_instrument_for_file_review():
    route = rgc_route("docs/SPEC_v0_2.mdをレビューして")
    assert route.level == 3
    assert route.mode == "instrument"


def test_instrument_budget_quick_standard_full():
    assert _instrument_budget("こんにちは").name == "quick"
    assert _instrument_budget("RQAの応答設計の前提を軽く見て").name == "standard"
    assert _instrument_budget("docs/SPEC_v0_2.mdをレビューして").name == "full"
    assert _instrument_budget("短文", review=True).name == "full"


def test_apply_budget_respects_explicit_k_depth():
    cfg = Config(evaluator_binding=EvaluatorBinding())
    cfg.k_candidates = 5
    cfg.max_reflection_depth = 2
    budget = _instrument_budget("こんにちは")
    args = SimpleNamespace(k=5, depth=2)
    _apply_budget(cfg, budget, args)
    assert cfg.k_candidates == 5
    assert cfg.max_reflection_depth == 2
    assert cfg.max_memory_fragments == 2
    assert cfg.max_regen_per_round == 0


def test_ask_light_input_fast_paths_to_conversation(capsys):
    rc = cmd_ask(
        SimpleNamespace(
            text="あなたは何者ですか",
            k=None,
            depth=None,
            model=None,
            no_evaluator=False,
            brief=False,
            instrument=False,
        )
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "MOBIUS-RQA" in out
    assert "# RQA 出力" not in out


def test_ask_greeting_fast_paths_to_conversation(capsys):
    rc = cmd_ask(
        SimpleNamespace(
            text="こんにちは",
            k=None,
            depth=None,
            model=None,
            no_evaluator=False,
            brief=False,
            instrument=False,
        )
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "今日はどこから" in out
    assert "# RQA 出力" not in out


def test_ask_missing_target_fast_paths_without_graph(capsys):
    rc = cmd_ask(
        SimpleNamespace(
            text="この仕様の前提を見て",
            k=None,
            depth=None,
            model=None,
            no_evaluator=False,
            brief=False,
            instrument=False,
        )
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "対象がまだこちらに渡っていません" in out
    assert "# RQA 出力" not in out


def test_ask_direct_route_uses_chat_surface(monkeypatch):
    called = {}

    def fake_run_once(
        cfg,
        text,
        sidecar_enabled=True,
        sidecar_timeout=150,
        stream_answer=True,
        show_sidecar_status=True,
    ):
        called["text"] = text
        called["sidecar"] = sidecar_enabled
        called["timeout"] = sidecar_timeout
        called["stream"] = stream_answer
        called["status"] = show_sidecar_status
        return 0

    monkeypatch.setattr("rqa.chat.run_once", fake_run_once)
    # closed factual lookup -> L1 direct (no loop), per Gate-D recalibration
    rc = cmd_ask(
        SimpleNamespace(
            text="ナイロビはどの国にある？",
            k=None,
            depth=None,
            model=None,
            no_evaluator=False,
            brief=False,
            instrument=False,
        )
    )
    assert rc == 0
    assert called == {
        "text": "ナイロビはどの国にある？",
        "sidecar": False,
        "timeout": 150,
        "stream": True,
        "status": True,
    }


def test_ask_guided_route_uses_sidecar(monkeypatch, capsys):
    called = {}

    def fake_run_once(
        cfg,
        text,
        sidecar_enabled=True,
        sidecar_timeout=150,
        stream_answer=True,
        show_sidecar_status=True,
    ):
        called["text"] = text
        called["sidecar"] = sidecar_enabled
        called["timeout"] = sidecar_timeout
        called["stream"] = stream_answer
        called["status"] = show_sidecar_status
        return 0

    monkeypatch.setattr("rqa.chat.run_once", fake_run_once)
    rc = cmd_ask(
        SimpleNamespace(
            text="RQAの応答設計の前提を軽く見て",
            k=None,
            depth=None,
            model=None,
            no_evaluator=False,
            brief=False,
            instrument=False,
        )
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "まず自然に答え" in out
    assert called == {
        "text": "RQAの応答設計の前提を軽く見て",
        "sidecar": True,
        "timeout": 30,
        "stream": False,
        "status": True,
    }


def test_l3_progress_updates_are_printed(capsys):
    def slow_runner():
        time.sleep(0.03)
        return "done"

    assert _run_with_progress(slow_runner, interval=0.01) == "done"
    out = capsys.readouterr().out
    assert "中間観測:" in out
    assert "まだ結論ではありません" in out


def _result_with_score(total, degraded=False, tmp_path=None):
    cfg = Config(evaluator_binding=EvaluatorBinding())
    cfg.state_dir = tmp_path / "state"
    cfg.k_candidates = 3
    ctl = Controller(
        cfg,
        adapter=FakeAdapter([adapter_output()]),
        evaluator=FakeEvaluator(total=total, fail=degraded),
        graph=QuestionGraph(cfg.graph_db),
    )
    return ctl.run("入力テキスト")


def test_surface_above_threshold(tmp_path):
    result = _result_with_score(36, tmp_path=tmp_path)
    assert should_surface(result, threshold=30) is not None


def test_silence_below_threshold(tmp_path):
    result = _result_with_score(16, tmp_path=tmp_path)
    assert should_surface(result, threshold=30) is None


def test_silence_when_chat_actionability_low(tmp_path):
    result = _result_with_score(36, tmp_path=tmp_path)
    result.final_round.selection = SelectionResult(
        best_index=0,
        scores=[CandidateScore(0, depth=9, sharpness=8, novelty=8, actionability=6)],
        rationale="deep but not useful now",
    )
    assert should_surface(result, threshold=34, min_depth=8, min_sharpness=8, min_actionability=7) is None


def test_silence_when_degraded(tmp_path):
    result = _result_with_score(36, degraded=True, tmp_path=tmp_path)
    assert should_surface(result, threshold=30) is None


def test_silence_on_none():
    assert should_surface(None, threshold=30) is None


def test_render_brief(tmp_path):
    result = _result_with_score(36, tmp_path=tmp_path)
    text = render_brief(result)
    assert text.startswith("Q: ")
    assert "/40" in text
