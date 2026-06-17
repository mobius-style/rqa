"""SFT/DPO data tooling (SPEC_v0_2.md §12).

- build_sft_example: run record -> training messages (assistant content = JSON string)
- lint_sft_example: data quality gate. Reuses the SAME schema parser and Governor
  checks that run in production, so nothing trainable can violate what the runtime
  would reject (§12.1 構成要件 enforced at data level).
- extract_dpo_pairs: run record selection scores -> preference pairs
  (score-gap threshold per §12.2; selector-log pairs are tagged with their source
  so the 50% cap can be enforced at corpus assembly time).
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from . import governor
from .prompts import SYSTEM_RQA, build_user_prompt
from .schema import SchemaError, parse_analysis, validate_diversity

DPO_SCORE_GAP = 2  # §12.2: pairs below this gap are noise
SELF_UPDATE_MAX_RATIO = 0.20  # §12.1: proposals in at most 20% of examples


def training_output_from_record(record: dict) -> dict:
    """Rebuild a clean §11.1 output object from a run record (drops model quirks)."""
    candidates = record.get("kept_candidates") or []
    return {
        "feature_map": record.get("feature_map") or {},
        "search_decision": record.get("search_decision") or {},
        "question_candidates": candidates,
        "self_ranking": list(range(len(candidates))),
        "self_update_proposal": record.get("self_update_proposal"),
    }


def build_sft_example(record: dict, require_threshold: bool = True) -> dict | None:
    """Run record -> SFT messages example. Returns None if the run does not qualify.

    Qualification (§12.1 審査): selection succeeded and, when require_threshold,
    the run stopped because the Evaluator score crossed the depth threshold.
    """
    if record.get("degraded"):
        return None
    if require_threshold and record.get("stop_reason") != "depth_threshold_reached":
        return None
    if not record.get("kept_candidates"):
        return None
    output = training_output_from_record(record)
    user = build_user_prompt(
        record["input_text"], record.get("fragments") or [], record.get("k", 6)
    )
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_RQA},
            {"role": "user", "content": user},
            {"role": "assistant", "content": json.dumps(output, ensure_ascii=False)},
        ],
        "meta": {
            "source": "run_record",
            "session": record.get("session"),
            "evaluator_best_total": (record.get("selection") or {}).get("scores", [{}])[0].get("total")
            if record.get("selection")
            else None,
        },
    }


def lint_sft_example(example: dict, k_expected: int | None = None) -> list[str]:
    """Data quality gate. Empty list = pass."""
    problems: list[str] = []
    msgs = example.get("messages") or []
    roles = [m.get("role") for m in msgs]
    if roles[:1] != ["system"] or "user" not in roles or roles[-1] != "assistant":
        problems.append("messages must be [system, ..., user, assistant]")
        return problems

    assistant = msgs[-1].get("content")
    if not isinstance(assistant, str):
        problems.append("assistant content must be a JSON string, not an object (§12.1-5)")
        return problems

    try:
        analysis = parse_analysis(assistant)
    except SchemaError as exc:
        problems.append(f"assistant content unparseable: {exc}")
        return problems

    k = k_expected or len(analysis.candidates)
    report = validate_diversity(analysis.candidates, k)
    problems += [f"diversity: {i}" for i in report.issues]

    problems += governor.check_self_update(analysis.self_update_proposal)
    problems += governor.check_tool_request(analysis.tool_request, enabled_tools=())

    user = next(m["content"] for m in msgs if m["role"] == "user")
    if "[memory_context]" in user:
        fm = analysis.feature_map
        if not (fm.get("tensions_memory_cross") or []):
            # not fatal for every example, but the graph-conditioned subset must
            # reference memory (§12.1-2); surface it for corpus assembly
            problems.append("note: memory_context present but no memory-cross tension")
    return [p for p in problems if not p.startswith("note:")] + [
        p for p in problems if p.startswith("note:")
    ]


def corpus_stats(examples: list[dict]) -> dict:
    """§12.1 構成要件 checks that only make sense corpus-wide."""
    n = len(examples)
    with_proposal = 0
    with_memory = 0
    for ex in examples:
        assistant = ex["messages"][-1]["content"]
        try:
            analysis = parse_analysis(assistant)
        except SchemaError:
            continue
        if analysis.self_update_proposal is not None:
            with_proposal += 1
        user = next(m["content"] for m in ex["messages"] if m["role"] == "user")
        if "[memory_context]" in user:
            with_memory += 1
    return {
        "n": n,
        "self_update_ratio": round(with_proposal / n, 3) if n else 0.0,
        "self_update_ratio_ok": (with_proposal / n) <= SELF_UPDATE_MAX_RATIO if n else True,
        "memory_context_ratio": round(with_memory / n, 3) if n else 0.0,
        "memory_context_target": 0.30,
    }


@dataclass
class DpoPair:
    prompt: str
    chosen: str
    rejected: str
    score_gap: int
    source: str  # "selector_log" (§12.2 供給源2) etc.

    def as_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
            "score_gap": self.score_gap,
            "source": self.source,
        }


def extract_dpo_pairs(record: dict) -> list[DpoPair]:
    """Selector-log preference pairs from one run record (§12.2 供給源2)."""
    selection = record.get("selection")
    shortlist = record.get("shortlist") or []
    if not selection or len(shortlist) < 2:
        return []
    scores = {s["index"]: s["total"] for s in selection.get("scores", []) if "index" in s}
    best = selection.get("best_index")
    if best not in scores:
        return []
    prompt = record["input_text"]
    pairs = []
    for idx, total in scores.items():
        if idx == best or idx >= len(shortlist):
            continue
        gap = scores[best] - total
        if gap >= DPO_SCORE_GAP:
            pairs.append(
                DpoPair(
                    prompt=prompt,
                    chosen=shortlist[best],
                    rejected=shortlist[idx],
                    score_gap=gap,
                    source="selector_log",
                )
            )
    return pairs
