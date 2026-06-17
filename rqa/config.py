"""RQA configuration — defaults follow SPEC_v0_2.md §13.2 / §15.1."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MMV_ROOT = PROJECT_ROOT.parent / "MOBIUS_MMV"

EVALUATOR_BINDING_PATH = PROJECT_ROOT / "config" / "evaluator_binding.yaml"


def _load_env_fallback(key: str) -> str | None:
    """env var -> mobius_rqa/.env -> MOBIUS_MMV/.env (pinned-evaluator credential)."""
    if os.environ.get(key):
        return os.environ[key]
    for env_file in (PROJECT_ROOT / ".env", MMV_ROOT / ".env"):
        if env_file.is_file():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith(f"{key}="):
                    val = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if val:
                        return val
    return None


@dataclass
class EvaluatorBinding:
    release: str = "MMV-L-RC3.3"
    endpoint: str = "https://api.groq.com/openai/v1"
    model: str = "openai/gpt-oss-120b"
    api_key_env: str = "GROQ_API_KEY"

    @classmethod
    def load(cls, path: Path = EVALUATOR_BINDING_PATH) -> "EvaluatorBinding":
        if path.is_file():
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            return cls(
                release=data.get("release", cls.release),
                endpoint=data.get("endpoint", cls.endpoint),
                model=data.get("model", cls.model),
                api_key_env=data.get("api_key_env", cls.api_key_env),
            )
        return cls()

    def api_key(self) -> str | None:
        return _load_env_fallback(self.api_key_env)


@dataclass
class Config:
    # adapter (generator)
    # Gate D APPROVED by owner T (2026-06-13): the SFT adapter is the standing
    # default. Base fallback if the adapter is unavailable: "gemma4:12b"
    # (revert = set this back to "gemma4:12b"; fully reversible).
    adapter_model: str = "rqa-gemma4:v0.1"
    ollama_url: str = "http://localhost:11434"
    # num_ctx >= 16384: Ollama's 4096 default silently truncates long inputs
    # (OPERATE local-arm confound, 2026-06)
    num_ctx: int = 16384
    temperature: float = 0.7

    # bounded reflection (§15.1)
    k_candidates: int = 6
    shortlist_s: int = 3
    max_reflection_depth: int = 3
    max_regen_per_round: int = 1
    max_memory_fragments: int = 8
    depth_threshold: int = 30  # of 40 (4 rubric axes x 10)

    # selection stage 2
    evaluator_enabled: bool = True
    evaluator_binding: EvaluatorBinding = field(default_factory=EvaluatorBinding.load)

    # Stage A/B: tools the controller will actually execute.
    # concept_memory_lookup runs implicitly as pre-noticing retrieval.
    enabled_tools: tuple = ()

    # storage
    state_dir: Path = PROJECT_ROOT / "state"

    @property
    def graph_db(self) -> Path:
        return self.state_dir / "graph.db"

    @property
    def log_dir(self) -> Path:
        return self.state_dir / "logs"
