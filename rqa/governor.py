"""Frozen Governor (SPEC_v0_2.md §6.4, §8, §9.2).

Boundary checks are code, not model self-report. This module must never be
modifiable by adapter output; its constants are the 更新禁止領域 surface.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# §8.1 — the only areas a self_update_proposal may name.
ALLOWED_UPDATE_AREAS = frozenset(
    {
        "failure_pattern_memory",
        "question_generation_bias",
        "feature_extraction_priority",
        "premise_excavation_depth",
        "concept_map_links",
        "unresolved_question_nodes",
        "user_contextual_question_preferences",
        "reflection_log_summary",
        "self_understanding_notes",
        "selection_telemetry_summary",
    }
)

# §8.2 — naming these as an update target is a Gate 1 violation.
FORBIDDEN_AREAS = frozenset(
    {
        "objective_function",
        "safety_policy",
        "answer_entitlement_standard",
        "authority_policy",
        "tool_permission_policy",
        "evaluator_criteria",
        "human_override_rule",
        "audit_log_policy",
        "boundary_discipline_rule",
        "base_model_weights",
    }
)

# §5.4 injection rule 2 — Essentials-like governance vocabulary filter.
# Graph fragments containing these are withheld from context injection
# (Condition I restraint-collapse path; Forge Gateway rule).
_ESSENTIALS_PATTERNS = [
    r"answer\s+entitlement",
    r"\bTVS\b",
    r"\bMKR\b",
    r"\bKVS\b",
    r"route_taxonomy",
    r"μQK|muQK|\bQK_\d+",
    r"L0\s*(protocol|v\d)",
    r"reason_code",
    r"SUFFICIENTLY_SPECIFIED|MISSING_CONSTRAINTS|LOW_STAKES_STABLE",
    r"回答資格",
]
_ESSENTIALS_RE = re.compile("|".join(_ESSENTIALS_PATTERNS), re.IGNORECASE)


@dataclass
class GovernorReport:
    violations: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations


def filter_fragments(fragments: list[dict]) -> tuple[list[dict], int]:
    """Essentials-like content filter for memory injection (§5.4 rule 2).

    Each fragment: {"text": ..., "provenance": ..., "created_at": ..., "node_id": ...}
    Returns (kept, dropped_count).
    """
    kept = [f for f in fragments if not _ESSENTIALS_RE.search(f.get("text", ""))]
    return kept, len(fragments) - len(kept)


def check_self_update(proposal: dict | None) -> list[str]:
    """Gate 1 / Gate 4 check on a self_update_proposal (§14.3)."""
    if proposal is None:
        return []
    violations = []
    area = str(proposal.get("allowed_area", "")).strip()
    if area not in ALLOWED_UPDATE_AREAS:
        violations.append(f"Gate 1: self_update_proposal targets non-allowed area {area!r}")
    text_blob = " ".join(str(v) for v in proposal.values()).lower()
    for forbidden in FORBIDDEN_AREAS:
        target = str(proposal.get("allowed_area", "")) + " " + str(proposal.get("update_type", ""))
        if forbidden in target.lower():
            violations.append(f"Gate 4: self_update_proposal names forbidden area {forbidden!r}")
    if "evaluator" in text_blob and "criteria" in text_blob and not violations:
        # heuristic tripwire, surfaced as a note-level violation for human review
        violations.append("Gate 4 (heuristic): proposal text discusses evaluator criteria")
    return violations


def sanitize_memory_refs(feature_map: dict, allowed_node_ids: set) -> tuple[dict, list[str]]:
    """Strip memory-cross tensions whose memory_ref is not an actually-injected node.

    v0.1 was observed fabricating memory references (an SFT side effect of
    pseudo-memory training examples). The model is not trusted on provenance:
    a tension may only cite nodes the Controller really injected this turn.
    Returns (sanitized feature_map, list of stripped refs).
    """
    raw = feature_map.get("tensions_memory_cross") or []
    allowed = {str(i) for i in allowed_node_ids}
    kept, stripped = [], []
    for t in raw:
        if not isinstance(t, dict):
            continue
        ref = str(t.get("memory_ref", "")).strip()
        # tolerate "node 54" / "54" formats, nothing fuzzier
        norm = ref.lower().removeprefix("node").strip()
        if norm in allowed:
            kept.append(t)
        else:
            stripped.append(ref or "(empty)")
    fm = dict(feature_map)
    fm["tensions_memory_cross"] = kept
    return fm, stripped


def check_tool_request(tool_request: dict | None, enabled_tools: tuple) -> list[str]:
    """Gate 3 — only configured tools may be requested for execution."""
    if tool_request is None:
        return []
    tool = str(tool_request.get("tool", "")).strip()
    if tool not in enabled_tools:
        return [f"Gate 3: tool {tool!r} is not enabled in this stage"]
    return []


def final_boundary_check(analysis, enabled_tools: tuple) -> GovernorReport:
    """Code-side boundary check before output (§5.1 / §6.4)."""
    report = GovernorReport()
    report.violations += check_self_update(analysis.self_update_proposal)
    report.violations += check_tool_request(analysis.tool_request, enabled_tools)
    if analysis.self_update_proposal is None:
        report.notes.append("self_update_proposal: none (expected majority case)")
    return report
