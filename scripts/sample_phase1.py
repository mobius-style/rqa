"""Sample 500 seeds for Phase 1 annotation with generation directives (SPEC §12.1).

- category-balanced draw from data/seed_inputs.jsonl
- per-seed directives: K (4-6), include_memory (~32%), include_self_update (~10%)
- splits into chunks of 25 -> data/phase1_chunks/chunk_NN.jsonl

Deterministic (seeded RNG) so the batch is reproducible.

Usage: ../venv313/bin/python scripts/sample_phase1.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SEEDS = PROJECT_ROOT / "data" / "seed_inputs.jsonl"
CHUNK_DIR = PROJECT_ROOT / "data" / "phase1_chunks"
BATCH = PROJECT_ROOT / "data" / "phase1_batch.jsonl"

# category -> (quota, default deepening posture for the generator)
PLAN = {
    "conceptual_explain": (110, "deepen"),
    "factual_inquiry": (90, "rule6"),
    "self_reference": (50, "moderate"),
    "correction": (60, "moderate"),
    "casual_engagement": (30, "rule6"),
    "volatile_current": (60, "anchor_search"),
    "stable_control": (35, "rule6"),
    "stale_premise_trap": (25, "anchor_search"),
    "date_boundary": (15, "anchor_search"),
    "query_neutrality": (15, "moderate"),
    "ambiguous_time_frame": (10, "anchor_search"),
}
CHUNK_SIZE = 25
MEMORY_RATIO = 0.32
SELF_UPDATE_RATIO = 0.10


def main() -> int:
    rng = random.Random(20260612)
    pools: dict[str, list[dict]] = {}
    for line in SEEDS.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        pools.setdefault(row["category"], []).append(row)

    sampled: list[dict] = []
    for cat, (quota, posture) in PLAN.items():
        pool = pools.get(cat, [])
        rng.shuffle(pool)
        for row in pool[:quota]:
            row["directives"] = {"posture": posture}
            sampled.append(row)

    rng.shuffle(sampled)
    n = len(sampled)
    mem_idx = set(rng.sample(range(n), int(n * MEMORY_RATIO)))
    # self_update only on deepening-capable items, never on rule6
    deepen_idx = [i for i, r in enumerate(sampled) if r["directives"]["posture"] in ("deepen", "moderate")]
    su_idx = set(rng.sample(deepen_idx, min(int(n * SELF_UPDATE_RATIO), len(deepen_idx))))
    for i, row in enumerate(sampled):
        row["directives"]["k"] = rng.choice([4, 4, 5, 6])
        row["directives"]["include_memory"] = i in mem_idx
        row["directives"]["include_self_update"] = i in su_idx

    BATCH.parent.mkdir(parents=True, exist_ok=True)
    with BATCH.open("w", encoding="utf-8") as fh:
        for row in sampled:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    for old in CHUNK_DIR.glob("chunk_*.jsonl"):
        old.unlink()
    for ci in range(0, n, CHUNK_SIZE):
        chunk = sampled[ci : ci + CHUNK_SIZE]
        path = CHUNK_DIR / f"chunk_{ci // CHUNK_SIZE:02d}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for row in chunk:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    by_cat: dict[str, int] = {}
    for r in sampled:
        by_cat[r["category"]] = by_cat.get(r["category"], 0) + 1
    print(f"sampled {n} seeds -> {BATCH}, {len(list(CHUNK_DIR.glob('chunk_*.jsonl')))} chunks")
    print(f"memory={sum(1 for r in sampled if r['directives']['include_memory'])}, "
          f"self_update={sum(1 for r in sampled if r['directives']['include_self_update'])}")
    for cat, c in sorted(by_cat.items(), key=lambda kv: -kv[1]):
        print(f"  {cat:25s} {c}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
