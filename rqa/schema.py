"""Model-output schema (SPEC_v0_2.md §11.1) — parsing and diversity validation (§5.3)."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

VALID_LAYERS = {"L3", "L4", "L5", "L7"}
VALID_STANCES = {"claim_skeptic", "frame_skeptic", "steelman"}


class SchemaError(ValueError):
    pass


def _normalize_layer(raw: str) -> str:
    """'L4 Assumption' / 'l4' / 'Layer 4' -> 'L4'. Unknown values pass through
    (validation rejects them downstream)."""
    m = re.search(r"[Ll](?:ayer)?\s*-?\s*([0-9])", raw)
    return f"L{m.group(1)}" if m else raw.strip()


_STANCE_HINTS = (
    ("steel", "steelman"),
    ("claim", "claim_skeptic"),
    ("frame", "frame_skeptic"),
)


def _normalize_stance(raw: str) -> str:
    low = raw.strip().lower()
    if low in VALID_STANCES:
        return low
    for hint, canonical in _STANCE_HINTS:
        if hint in low:
            return canonical
    return raw.strip()


@dataclass
class Candidate:
    question: str
    target_layer: str
    target_element: str
    stance: str


@dataclass
class Analysis:
    feature_map: dict
    search_decision: dict
    candidates: list[Candidate]
    self_ranking: list[int]
    self_update_proposal: dict | None
    tool_request: dict | None = None
    raw: dict = field(default_factory=dict)


def _escape_control_chars_in_strings(text: str) -> str:
    """Escape raw newlines/tabs that local models emit inside JSON strings."""
    out: list[str] = []
    in_string = False
    escaped = False
    for ch in text:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            elif ch == "\n":
                out.append("\\n")
                continue
            elif ch == "\t":
                out.append("\\t")
                continue
        elif ch == '"':
            in_string = True
        out.append(ch)
    return "".join(out)


def _try_loads(text: str) -> dict | None:
    for variant in (text, _escape_control_chars_in_strings(text)):
        try:
            data = json.loads(variant, strict=False)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            continue
    return None


def _extract_json(text: str) -> dict:
    """Parse model output as JSON; tolerate code fences, prose, control chars."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text.strip())
    data = _try_loads(text)
    if data is not None:
        return data
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        data = _try_loads(text[start : end + 1])
        if data is not None:
            return data
    raise SchemaError(f"no parseable JSON object in model output: {text[:200]!r}")


def parse_analysis(text: str) -> Analysis:
    data = _extract_json(text)
    if not isinstance(data, dict):
        raise SchemaError("model output is not a JSON object")

    fm = data.get("feature_map") or {}
    if not isinstance(fm, dict):
        raise SchemaError("feature_map must be an object")

    raw_candidates = data.get("question_candidates") or []
    candidates: list[Candidate] = []
    for c in raw_candidates:
        if not isinstance(c, dict) or not str(c.get("question", "")).strip():
            continue
        candidates.append(
            Candidate(
                question=str(c["question"]).strip(),
                target_layer=_normalize_layer(str(c.get("target_layer", ""))),
                target_element=str(c.get("target_element", "")).strip(),
                stance=_normalize_stance(str(c.get("stance", ""))),
            )
        )

    ranking_raw = data.get("self_ranking") or []
    self_ranking = []
    for idx in ranking_raw:
        try:
            i = int(idx)
        except (TypeError, ValueError):
            continue
        if 0 <= i < len(candidates) and i not in self_ranking:
            self_ranking.append(i)
    # fall back to listed order for unranked candidates
    for i in range(len(candidates)):
        if i not in self_ranking:
            self_ranking.append(i)

    proposal = data.get("self_update_proposal")
    if not isinstance(proposal, dict):
        proposal = None

    tool_request = data.get("tool_request")
    if not isinstance(tool_request, dict):
        tool_request = None

    return Analysis(
        feature_map=fm,
        search_decision=data.get("search_decision") or {},
        candidates=candidates,
        self_ranking=self_ranking,
        self_update_proposal=proposal,
        tool_request=tool_request,
        raw=data,
    )


@dataclass
class DiversityReport:
    kept: list[Candidate]
    kept_indices: list[int]
    issues: list[str]

    @property
    def ok(self) -> bool:
        return not self.issues


def validate_diversity(candidates: list[Candidate], k: int) -> DiversityReport:
    """Machine-checked diversity constraints (§5.3).

    Drops malformed/duplicate-target candidates; reports spread issues that
    justify one regeneration request.
    """
    issues: list[str] = []
    kept: list[Candidate] = []
    kept_indices: list[int] = []
    seen_elements: set[str] = set()
    seen_questions: set[str] = set()

    for i, c in enumerate(candidates):
        if c.target_layer not in VALID_LAYERS:
            issues.append(f"candidate {i}: invalid target_layer {c.target_layer!r}")
            continue
        if c.stance not in VALID_STANCES:
            issues.append(f"candidate {i}: invalid stance {c.stance!r}")
            continue
        qnorm = re.sub(r"\s+", "", c.question)
        if qnorm in seen_questions:
            issues.append(f"candidate {i}: duplicate question text")
            continue
        element_key = c.target_element or f"_unnamed_{i}"
        if element_key in seen_elements:
            issues.append(f"candidate {i}: duplicate target_element {element_key!r}")
            continue
        seen_elements.add(element_key)
        seen_questions.add(qnorm)
        kept.append(c)
        kept_indices.append(i)

    if len(kept) < max(2, k // 2):
        issues.append(f"only {len(kept)}/{k} valid candidates")
    if len({c.target_layer for c in kept}) < 2 and len(kept) >= 2:
        issues.append("layer spread < 2 (all candidates target the same layer)")
    if len({c.stance for c in kept}) < 2 and len(kept) >= 2:
        issues.append("stance spread < 2 (all candidates share one stance)")

    return DiversityReport(kept=kept, kept_indices=kept_indices, issues=issues)
