# MOBIUS-RQA — memory-echo governance for assistants with memory

**Keep an assistant's own past guesses from coming back as if they were facts.**

Assistants with long-term memory share a failure mode this project calls
*memory echo*: an interpretation the model produced on an earlier turn is
stored, retrieved later, and then treated as independent, externally-grounded
evidence. Left unchecked it yields a self-reinforcing echo chamber and
fabricated self-citations — *"as I established earlier, X"* — where X was never
actually established.

This repository is a working, model-independent, near-deterministic mitigation,
with an empirical pilot behind it:

- **Provenance-tagged memory.** Every stored item records its source class
  (user / assistant-generated / external / system), and the tag is carried into
  the assembled context, so an assistant-generated item is visibly not
  independent evidence. ([`rqa/graph.py`](rqa/graph.py), [`rqa/prompts.py`](rqa/prompts.py))
- **Output-time self-citation verification.** Before a response is emitted,
  every memory reference it makes is checked against the set of nodes actually
  injected this turn; references to anything not injected are fabricated
  self-citations and are stripped. The model is *not* trusted on provenance —
  the check is deterministic code, not a model self-report.
  (`sanitize_memory_refs` in [`rqa/governor.py`](rqa/governor.py))
- **Measured.** In a pilot ([`docs/EMPIRICAL_COMPANION_FINDINGS.md`](docs/EMPIRICAL_COMPANION_FINDINGS.md)),
  ungoverned raw memory collapses answer-space plurality to ~1/7 of baseline
  while governed memory recovers it; the output-time verifier removes fabricated
  self-citations deterministically, and the effect reproduces across **six local
  models in three families** (Gemma-4, Llama-3.1, Qwen-3.5/3.6).

If you are building a memory or personalization layer for a chat assistant or
agent, those two primitives — provenance tagging and the output-time citation
check — are the reusable, liftable part.

> Scope: this is a research pilot (small n), not a hardened library. The
> measurements establish the *mechanism*; production use wants larger n and your
> own integration. See [docs/EMPIRICAL_COMPANION_FINDINGS.md](docs/EMPIRICAL_COMPANION_FINDINGS.md)
> for the limitations.

## The system it lives in: bounded reflective questioning

MOBIUS-RQA (Reflective Questioning Adapter / BRSA) is also a complete reflective
questioning assistant — a sibling of the MOBIUS MMV system. It does not
primarily answer; it structures tensions, excavates assumptions, and produces
**deeper questions** under explicit governance. The memory governance above is
what keeps that reflection honest: its questions stay grounded in what was
actually said, not in the assistant's own earlier guesses.

Spec: [docs/SPEC_v0_4.md](docs/SPEC_v0_4.md) (full integrated spec).
Earlier records remain in [docs/SPEC_v0_2.md](docs/SPEC_v0_2.md) and
[docs/SPEC_v0_3_DELTA.md](docs/SPEC_v0_3_DELTA.md).
Current implementation: **v0.4 RGC conversation/instrument surface + trained adapter path**.
The code default remains `gemma4:12b`; the adapter is available through the
launcher/Ollama binding pending owner approval.

## Architecture (implemented subset)

```
input
 → Question Graph pre-noticing retrieval   (graph.py, Essentials filter in governor.py)
 → frozen base LLM via Ollama (gemma4:12b) (llm.py, prompts.py)
 → K diverse question candidates           (machine-checked, schema.py)
 → Stage 1: local self-ranking
 → Stage 2: Pinned External Evaluator      (MMV-L gpt-oss-120B via Groq, evaluator.py)
            with Local Degradation         (outage → Stage 1 result, never halts)
 → Governor boundary check (code, not model self-report)
 → output + Graph write-back + selection telemetry
```

Boundary discipline, the Essentials-like injection filter, append-only graph
with audit log, and bounded reflection limits follow the spec exactly.

## Usage

```bash
PY=../venv313/bin/python          # shared MOBIUS venv  (or: bin/rqa <verb>)

$PY -m rqa chat                   # conversational REPL: streaming answer +
                                  #   async reflective sidecar (SPEC v0.3)
$PY -m rqa chat "あなたは何者ですか" # one-shot conversation; light inputs skip sidecar
$PY -m rqa check                  # environment health
$PY -m rqa ask "自己理解の層だけを更新すれば、自己更新AIは安定するのではないか?"
                                  # RGC-routed: direct/guided/instrument by weight
$PY -m rqa ask "あなたは何者ですか" # light opener: fast conversation reply
$PY -m rqa ask --instrument "あなたは何者ですか"
                                  # force dashboard for a light opener
$PY -m rqa ask --brief "..."      # question-only view
$PY -m rqa review path/to/doc.md  # document review mode (dogfood target)
$PY -m rqa graph stats
$PY -m rqa graph search "自己理解層"
$PY -m rqa ask --no-evaluator "..."   # force Stage 1 only
```

Evaluator credentials: `GROQ_API_KEY` from the environment or `./.env`.
The evaluator binding itself is pinned
in [config/evaluator_binding.yaml](config/evaluator_binding.yaml) — changing
it is an evaluator swap and requires human approval (spec §6.5).

## Tests

```bash
../venv313/bin/python -m pytest -q tests/
```

Unit tests are network-free (fake adapter/evaluator). Latest local run:
`64 passed`. The live path is exercised via `rqa check` + a real `rqa ask`.

## Training (Stage C / SPEC §13)

Dedicated venv (`.venv-train` — keeps the shared MOBIUS venv frozen):

```bash
PYT=.venv-train/bin/python
$PYT scripts/prepare_training_data.py        # {{SYSTEM_RQA}} -> real prompt, train/val split
$PYT scripts/train_sft.py --smoke            # 4-step VRAM/stack verification
$PYT scripts/train_sft.py                    # full QLoRA SFT -> models/rqa_adapter_v0_1/
$PYT scripts/eval_adapter.py                 # base vs adapter on human-reviewed holdout
```

Base: google/gemma-4-12B-it (unified multimodal; LoRA targets the language
tower only). 4-bit NF4 + rank 16 + gradient checkpointing fits a 16GB RTX
5070 Ti. Corpus: `data/sft_phase1_train.jsonl` (426; see
docs/PHASE1_CORPUS_REPORT.md). `sft_phase1_holdout.jsonl` and
`gold_seed.jsonl` are evaluation-only — never train on them.

## Not implemented yet (by design — spec staging)

- 読み出し2 cross-contradiction check / 読み出し3 graph curation (Stage B/C)
- web_search / local_RAG / document_fetch tools (Stage B)
- ME5 embedding retrieval (Stage B; current backend is char-trigram)

## License

- Code & figures: **AGPL-3.0-or-later** (see [LICENSE](LICENSE)).
- Documents under `docs/` that carry their own license header (the companion
  paper drafts) are **CC BY-NC-SA 4.0**, as stated in those files.

## Related — the Möbius program

Part of the [MOBIUS](https://github.com/mobius-style) program — local-first, AGPL:

- [mmv](https://github.com/mobius-style/mmv) — answer-entitlement runtime: decides *whether* answering is warranted
- [rqa](https://github.com/mobius-style/rqa) — reflective questioning adapter: deepens *the question* when it is not
- [rcgov](https://github.com/mobius-style/rcgov) — reflective context governor: governs *what a model may read*
- [infinity](https://github.com/mobius-style/infinity) — composite capstone (MMV × RQA) with an OpenAI-compatible API
