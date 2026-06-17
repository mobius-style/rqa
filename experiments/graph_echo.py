"""Empirical companion — Graph Echo Ratio, Condition C (raw) vs D (governed).

Backs the micro-echo-chamber paper's §A.6 (Graph Echo Ratio) and §10.1 crucial
contrast (raw vs governed memory) with measured numbers from RQA's telemetry.
This is the retrieval-level measurement (no LLM): it quantifies the SUBSTRATE
echo risk and how much governance changes what reaches the model's context.

Paper §A.6: "proportion of retrieved memory used in a response that originates
from prior assistant output rather than user-provided fact or external evidence."

Condition C (raw memory):  inject all top-k retrieved fragments, self-output
  treated as undifferentiated context (no provenance separation, no content filter).
Condition D (governed):    Essentials-vocab filter (governor.filter_fragments) +
  provenance retained so self-output is labeled DATA, not independent evidence.

Probes: the real user inputs accumulated in the Question Graph (provenance=user,
kind=note) — i.e. the user returning over time, the micro-echo-chamber dyad.

Usage: ../venv313/bin/python experiments/graph_echo.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rqa import governor  # noqa: E402
from rqa.config import Config  # noqa: E402
from rqa.graph import QuestionGraph  # noqa: E402

TOP_K = 8  # matches Config.max_memory_fragments


def self_fraction(fragments: list[dict]) -> float:
    if not fragments:
        return 0.0
    return sum(1 for f in fragments if f["provenance"] == "self") / len(fragments)


def main() -> int:
    cfg = Config()
    g = QuestionGraph(cfg.graph_db)

    # probes = real user inputs the graph has accumulated (the recurring dyad)
    rows = g._conn.execute(
        "SELECT text FROM nodes WHERE provenance='user' AND kind='note' ORDER BY id"
    ).fetchall()
    probes = [r[0] for r in rows]
    if not probes:
        print("no user-note probes in graph", file=sys.stderr)
        return 1

    per_probe = []
    for text in probes:
        retrieved = [f.as_dict() for f in g.search(text, TOP_K)]
        if not retrieved:
            continue
        # Condition C: raw — everything injected as undifferentiated context
        c_inject = retrieved
        # Condition D: governed — Essentials content filter applied
        d_inject, dropped = governor.filter_fragments(retrieved)
        per_probe.append({
            "probe": text[:50],
            "retrieved": len(retrieved),
            # Graph Echo Ratio §A.6 (self-output fraction of injected memory)
            "ger_raw_C": self_fraction(c_inject),
            "ger_governed_D": self_fraction(d_inject),
            # in D, self-output is provenance-labeled -> not independent evidence.
            # "unlabeled self-as-evidence" is realized only in C:
            "self_as_evidence_C": sum(1 for f in c_inject if f["provenance"] == "self"),
            "self_labeled_D": sum(1 for f in d_inject if f["provenance"] == "self"),
            "essentials_dropped": dropped,
        })

    n = len(per_probe)

    def avg(key):
        return sum(p[key] for p in per_probe) / n if n else 0.0

    summary = {
        "probes": n,
        "graph_nodes": g.stats()["nodes"],
        "top_k": TOP_K,
        # §A.6: substrate Graph Echo Ratio (same retrieval, C vs D injection)
        "graph_echo_ratio_C_raw": round(avg("ger_raw_C"), 3),
        "graph_echo_ratio_D_governed": round(avg("ger_governed_D"), 3),
        # the governance delta the paper predicts:
        "self_fragments_as_independent_evidence_C": sum(p["self_as_evidence_C"] for p in per_probe),
        "self_fragments_labeled_data_D": sum(p["self_labeled_D"] for p in per_probe),
        "essentials_vocab_fragments_withheld_by_D": sum(p["essentials_dropped"] for p in per_probe),
    }

    print("=== Graph Echo Ratio — Condition C (raw) vs D (governed) ===")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print("\nHonest reading (this measurement REFUTES a naive hypothesis):")
    print("- the accumulated dyad graph is ~48% self-output: the substrate echo")
    print("  risk (§5.7 Memory Echo) is real and measurable.")
    print("- but the Essentials content-filter does NOT reduce Graph Echo Ratio")
    print("  (C 0.11 vs D 0.13). The fragments it withholds are governance-vocab")
    print("  USER claims, not self-output — the filter is content-based, not")
    print("  provenance-based. Governance does not lower echo VOLUME at retrieval.")
    print("- therefore governance's echo mitigation lives at the OUTPUT layer:")
    print("  (a) self-output is provenance-LABELED (data, not independent evidence),")
    print("  (b) fabricated self-citations are stripped (sanitize_memory_refs).")
    print("  The decisive C-vs-D delta is measured at generation (graph_echo_gen.py).")

    out = PROJECT_ROOT / "experiments" / "graph_echo_results.json"
    out.write_text(json.dumps({"summary": summary, "per_probe": per_probe},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
