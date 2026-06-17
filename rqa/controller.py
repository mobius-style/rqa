"""Agent Shell / Controller — the bounded reflection loop (SPEC_v0_2.md §15.3)."""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from . import governor
from .config import Config
from .evaluator import EvaluatorUnavailable, GroqEvaluator, SelectionResult
from .graph import QuestionGraph
from .llm import AdapterError, OllamaAdapter
from .prompts import SYSTEM_RQA, build_user_prompt
from .schema import Analysis, Candidate, SchemaError, parse_analysis, validate_diversity


@dataclass
class RoundRecord:
    round_index: int
    analysis: Analysis
    kept_candidates: list[Candidate]
    diversity_issues: list[str]
    shortlist: list[Candidate]
    selection: SelectionResult | None
    degraded: bool
    chosen: Candidate | None


@dataclass
class RunResult:
    session_id: str
    input_text: str
    rounds: list[RoundRecord] = field(default_factory=list)
    injected_fragments: list[dict] = field(default_factory=list)
    fragments_filtered: int = 0
    memory_refs_stripped: list[str] = field(default_factory=list)
    governor_report: governor.GovernorReport | None = None
    stop_reason: str = ""

    @property
    def final_round(self) -> RoundRecord | None:
        return self.rounds[-1] if self.rounds else None

    @property
    def chosen(self) -> Candidate | None:
        return self.final_round.chosen if self.final_round else None


class Controller:
    def __init__(self, cfg: Config, adapter: OllamaAdapter | None = None,
                 evaluator: GroqEvaluator | None = None, graph: QuestionGraph | None = None):
        self.cfg = cfg
        self.adapter = adapter or OllamaAdapter(
            cfg.adapter_model, cfg.ollama_url, cfg.num_ctx, cfg.temperature
        )
        self.evaluator = evaluator or GroqEvaluator(cfg.evaluator_binding)
        self.graph = graph or QuestionGraph(cfg.graph_db)

    # -- main loop ---------------------------------------------------------

    def run(self, input_text: str) -> RunResult:
        session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:6]
        result = RunResult(session_id=session_id, input_text=input_text)

        # 読み出し1: pre-noticing retrieval, Essentials-filtered (§5.4)
        raw_fragments = [f.as_dict() for f in self.graph.search(input_text, self.cfg.max_memory_fragments * 2)]
        kept, dropped = governor.filter_fragments(raw_fragments)
        result.injected_fragments = kept[: self.cfg.max_memory_fragments]
        result.fragments_filtered = dropped

        messages: list[dict] = [
            {"role": "user", "content": build_user_prompt(input_text, result.injected_fragments, self.cfg.k_candidates)}
        ]

        allowed_node_ids = {f["node_id"] for f in result.injected_fragments}

        for step in range(self.cfg.max_reflection_depth):
            record = self._run_round(step, input_text, messages)
            # provenance discipline: the model may only cite injected nodes (§5.4)
            sanitized, stripped = governor.sanitize_memory_refs(
                record.analysis.feature_map, allowed_node_ids
            )
            record.analysis.feature_map = sanitized
            result.memory_refs_stripped.extend(stripped)
            result.rounds.append(record)

            # Governor boundary check on raw analysis (Gates 1/3/4)
            report = governor.final_boundary_check(record.analysis, self.cfg.enabled_tools)
            if not report.ok:
                result.governor_report = report
                result.stop_reason = "boundary_violation_detected"
                break

            if record.chosen is None:
                result.stop_reason = "no_valid_candidate"
                break

            score = record.selection.score_of_best_total() if record.selection else None
            if record.selection and score is not None and score >= self.cfg.depth_threshold:
                result.stop_reason = "depth_threshold_reached"
                break
            if step == self.cfg.max_reflection_depth - 1:
                result.stop_reason = "max_reflection_depth"
                break

            # not deep enough — feed selection verdict back and iterate
            feedback = self._round_feedback(record)
            messages.append({"role": "assistant", "content": json.dumps(record.analysis.raw, ensure_ascii=False)})
            messages.append({"role": "user", "content": build_user_prompt(
                input_text, [], self.cfg.k_candidates, round_feedback=feedback)})

        if result.governor_report is None and result.final_round is not None:
            result.governor_report = governor.final_boundary_check(
                result.final_round.analysis, self.cfg.enabled_tools
            )

        self._write_back(result)
        self._write_telemetry(result)
        self._write_run_record(result)
        return result

    # -- single round ------------------------------------------------------

    def _run_round(self, step: int, input_text: str, messages: list[dict]) -> RoundRecord:
        try:
            analysis = self._call_adapter(messages)
        except (SchemaError, AdapterError) as exc:
            # unparseable or empty after retry — fail the round, never the process
            empty = Analysis(feature_map={}, search_decision={}, candidates=[],
                             self_ranking=[], self_update_proposal=None)
            return RoundRecord(
                round_index=step, analysis=empty, kept_candidates=[],
                diversity_issues=[f"adapter failed: {exc}"],
                shortlist=[], selection=None, degraded=False, chosen=None,
            )
        diversity = validate_diversity(analysis.candidates, self.cfg.k_candidates)

        # one corrective regeneration if diversity constraints failed (§5.3)
        if not diversity.ok and self.cfg.max_regen_per_round > 0:
            regen_msgs = messages + [
                {"role": "assistant", "content": json.dumps(analysis.raw, ensure_ascii=False)},
                {"role": "user", "content": (
                    "Diversity constraints failed:\n- " + "\n- ".join(diversity.issues)
                    + f"\nRegenerate ALL {self.cfg.k_candidates} question_candidates fixing these issues. "
                      "Keep feature_map and other fields. Same JSON schema."
                )},
            ]
            try:
                analysis2 = self._call_adapter(regen_msgs)
                diversity2 = validate_diversity(analysis2.candidates, self.cfg.k_candidates)
                if len(diversity2.kept) > len(diversity.kept):
                    analysis, diversity = analysis2, diversity2
            except (SchemaError, Exception):  # noqa: BLE001 — regen is best-effort
                pass

        # Stage 1: self-ranking over kept candidates
        kept_set = set(diversity.kept_indices)
        ranked = [i for i in analysis.self_ranking if i in kept_set]
        shortlist_idx = ranked[: self.cfg.shortlist_s]
        shortlist = [analysis.candidates[i] for i in shortlist_idx]

        # Stage 2: pinned evaluator; degrade to Stage 1 on failure (§5.2)
        selection: SelectionResult | None = None
        degraded = False
        chosen: Candidate | None = shortlist[0] if shortlist else None
        if shortlist and self.cfg.evaluator_enabled:
            try:
                selection = self.evaluator.select(input_text, shortlist)
                chosen = shortlist[selection.best_index]
            except EvaluatorUnavailable:
                degraded = True
        elif shortlist:
            degraded = True

        return RoundRecord(
            round_index=step,
            analysis=analysis,
            kept_candidates=diversity.kept,
            diversity_issues=diversity.issues,
            shortlist=shortlist,
            selection=selection,
            degraded=degraded,
            chosen=chosen,
        )

    def _call_adapter(self, messages: list[dict]) -> Analysis:
        try:
            text = self.adapter.chat(SYSTEM_RQA, messages)
        except AdapterError:
            text = self.adapter.chat(SYSTEM_RQA, messages)  # one retry for transient empties
        try:
            analysis = parse_analysis(text)
        except SchemaError:
            retry = messages + [
                {"role": "assistant", "content": text},
                {"role": "user", "content": "Output was not valid JSON per the schema. Respond again with ONE valid JSON object only."},
            ]
            text = self.adapter.chat(SYSTEM_RQA, retry)
            try:
                analysis = parse_analysis(text)
            except SchemaError:
                self._dump_raw(text)
                raise
        if not analysis.candidates:
            self._dump_raw(text)
        return analysis

    def _write_run_record(self, result: RunResult) -> None:
        """Full reconstruction record — the raw material for SFT/DPO export (§12)."""
        final = result.final_round
        if final is None:
            return
        runs_dir = self.cfg.state_dir / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "session": result.session_id,
            "input_text": result.input_text,
            "k": self.cfg.k_candidates,
            "fragments": result.injected_fragments,
            "feature_map": final.analysis.feature_map,
            "search_decision": final.analysis.search_decision,
            "kept_candidates": [
                {
                    "question": c.question,
                    "target_layer": c.target_layer,
                    "target_element": c.target_element,
                    "stance": c.stance,
                }
                for c in final.kept_candidates
            ],
            "shortlist": [c.question for c in final.shortlist],
            "self_update_proposal": final.analysis.self_update_proposal,
            "selection": {
                "scores": [s.as_dict() for s in final.selection.scores],
                "best_index": final.selection.best_index,
                "rationale": final.selection.rationale,
            }
            if final.selection
            else None,
            "chosen": final.chosen.question if final.chosen else None,
            "degraded": final.degraded,
            "stop_reason": result.stop_reason,
            "rounds": len(result.rounds),
        }
        (runs_dir / f"{result.session_id}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8"
        )

    def _dump_raw(self, text: str) -> None:
        """Persist unusable adapter output for offline diagnosis."""
        self.cfg.log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        (self.cfg.log_dir / f"raw_failure_{ts}_{uuid.uuid4().hex[:6]}.txt").write_text(
            text, encoding="utf-8"
        )

    @staticmethod
    def _round_feedback(record: RoundRecord) -> str:
        if record.selection is None or record.chosen is None:
            return "The previous candidates were too shallow. Target deeper layers (L4/L5)."
        s = record.selection.score_of(record.selection.best_index)
        return (
            f"Best candidate so far scored {s.total if s else '?'}/40 "
            f"(rationale: {record.selection.rationale}). "
            "Go one layer deeper: excavate the assumption BEHIND your best candidate, "
            "or shift the frame it presupposes."
        )

    # -- 蓄積の書き込み (§5.1 bottom) ---------------------------------------

    def _write_back(self, result: RunResult) -> None:
        final = result.final_round
        if final is None:
            return
        sid = result.session_id
        fm = final.analysis.feature_map
        self.graph.add_node(
            "note", result.input_text[:500], provenance="user", session=sid,
            meta={"role": "input"},
        )
        for claim in (fm.get("claims") or [])[:5]:
            if isinstance(claim, str) and claim.strip():
                self.graph.add_node("claim", claim, provenance="user", session=sid)
        for tension in (fm.get("tensions_input_internal") or [])[:5]:
            if isinstance(tension, str) and tension.strip():
                self.graph.add_node("tension", tension, provenance="self", session=sid)
        if final.chosen is not None:
            self.graph.add_node(
                "question",
                final.chosen.question,
                provenance="self",
                session=sid,
                meta={
                    "target_layer": final.chosen.target_layer,
                    "stance": final.chosen.stance,
                    "stop_reason": result.stop_reason,
                },
            )

    def _write_telemetry(self, result: RunResult) -> None:
        self.cfg.log_dir.mkdir(parents=True, exist_ok=True)
        path = self.cfg.log_dir / f"run_{result.session_id}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for rec in result.rounds:
                fh.write(json.dumps({
                    "session": result.session_id,
                    "round": rec.round_index,
                    "n_candidates": len(rec.analysis.candidates),
                    "n_kept": len(rec.kept_candidates),
                    "diversity_issues": rec.diversity_issues,
                    "shortlist": [c.question for c in rec.shortlist],
                    "scores": [s.as_dict() for s in rec.selection.scores] if rec.selection else None,
                    "degraded": rec.degraded,
                    "chosen": rec.chosen.question if rec.chosen else None,
                    "stop_reason": result.stop_reason,
                    "fragments_injected": len(result.injected_fragments),
                    "fragments_filtered": result.fragments_filtered,
                    "memory_refs_stripped": result.memory_refs_stripped,
                }, ensure_ascii=False) + "\n")
