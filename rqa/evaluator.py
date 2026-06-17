"""External Evaluator — runtime selector, Stage 2 (SPEC_v0_2.md §5.2, §6.5 role A).

Pinned External Evaluator with Local Degradation: any failure here raises
EvaluatorUnavailable and the controller degrades to Stage 1. It never halts the run.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import requests

from .config import EvaluatorBinding
from .prompts import EVALUATOR_RUBRIC
from .schema import Candidate


class EvaluatorUnavailable(RuntimeError):
    pass


@dataclass
class CandidateScore:
    index: int
    depth: int
    sharpness: int
    novelty: int
    actionability: int

    @property
    def total(self) -> int:
        return self.depth + self.sharpness + self.novelty + self.actionability

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "depth": self.depth,
            "sharpness": self.sharpness,
            "novelty": self.novelty,
            "actionability": self.actionability,
            "total": self.total,
        }


@dataclass
class SelectionResult:
    best_index: int
    scores: list[CandidateScore]
    rationale: str

    def score_of(self, index: int) -> CandidateScore | None:
        return next((s for s in self.scores if s.index == index), None)

    def score_of_best_total(self) -> int | None:
        s = self.score_of(self.best_index)
        return s.total if s else None


class GroqEvaluator:
    def __init__(self, binding: EvaluatorBinding):
        self.binding = binding
        self._key = binding.api_key()

    def available(self) -> bool:
        return bool(self._key)

    def select(self, input_text: str, shortlist: list[Candidate]) -> SelectionResult:
        if not self._key:
            raise EvaluatorUnavailable(f"{self.binding.api_key_env} not set")
        listing = "\n".join(
            f"[{i}] (layer={c.target_layer}, stance={c.stance}, target={c.target_element})\n{c.question}"
            for i, c in enumerate(shortlist)
        )
        user = (
            f"Original input:\n{input_text[:4000]}\n\nCandidate questions:\n{listing}"
        )
        try:
            resp = requests.post(
                f"{self.binding.endpoint}/chat/completions",
                headers={"Authorization": f"Bearer {self._key}"},
                json={
                    "model": self.binding.model,
                    "messages": [
                        {"role": "system", "content": EVALUATOR_RUBRIC},
                        {"role": "user", "content": user},
                    ],
                    "response_format": {"type": "json_object"},
                    "temperature": 0.2,
                },
                timeout=60,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            data = json.loads(content)
        except (requests.RequestException, KeyError, IndexError, json.JSONDecodeError) as exc:
            raise EvaluatorUnavailable(f"evaluator call failed: {exc}") from exc

        scores: list[CandidateScore] = []
        for s in data.get("scores", []):
            try:
                idx = int(s["index"])
                if not 0 <= idx < len(shortlist):
                    continue
                scores.append(
                    CandidateScore(
                        index=idx,
                        depth=int(s.get("depth", 0)),
                        sharpness=int(s.get("sharpness", 0)),
                        novelty=int(s.get("novelty", 0)),
                        actionability=int(s.get("actionability", 0)),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
        if not scores:
            raise EvaluatorUnavailable("evaluator returned no usable scores")

        best = data.get("best_index")
        try:
            best = int(best)
        except (TypeError, ValueError):
            best = -1
        if not 0 <= best < len(shortlist):
            best = max(scores, key=lambda s: s.total).index

        return SelectionResult(
            best_index=best,
            scores=scores,
            rationale=str(data.get("rationale", ""))[:500],
        )
