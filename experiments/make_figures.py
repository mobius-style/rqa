"""Publication figures for the empirical companion (paper figures).

Reads the committed result JSONs and renders clean PNG+PDF figures into
experiments/figures/. English labels (the paper is in English). Run with the
training venv (matplotlib): .venv-train/bin/python experiments/make_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 12, "axes.titleweight": "bold",
})
PRIM = "#2c5f8a"   # primary judge (gpt-oss)
INDEP = "#c97a3a"  # independent judge (qwen)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png")
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)
    print(f"-> figures/{name}.png/.pdf")


def fig_ladder():
    d = json.loads((ROOT / "condition_ladder_results.json").read_text())["conditions"]
    conds = ["A", "B", "C", "D", "E"]
    names = ["A\nno pers.", "B\nstyle-only", "C\nraw memory", "D\ngoverned mem.", "E\ngov.+reflect"]
    prim = [d[c]["primary_retention"] * 100 for c in conds]
    indep = [d[c]["independent_retention"] * 100 for c in conds]
    kap = [d[c]["kappa"] for c in conds]
    labels = [f"{nm}\nκ={k:.2f}" for nm, k in zip(names, kap)]
    x = range(len(conds))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    b1 = ax.bar([i - w / 2 for i in x], prim, w, label="primary judge (gpt-oss-120B)", color=PRIM)
    b2 = ax.bar([i + w / 2 for i in x], indep, w, label="independent judge (qwen3.6-27B)", color=INDEP)
    for bars in (b1, b2):
        for b in bars:
            ax.annotate(f"{b.get_height():.0f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom", fontsize=8)
    ax.set_xticks(list(x)); ax.set_xticklabels(labels)
    ax.set_ylabel("live-alternative retention (%)  — STRICT criterion")
    ax.set_title("Condition A–E: raw memory collapses plurality, governance recovers it")
    ax.set_ylim(0, max(indep) * 1.18)
    ax.axhline(prim[0], ls=":", lw=1, color=PRIM, alpha=.6)
    ax.axhline(indep[0], ls=":", lw=1, color=INDEP, alpha=.6)
    # keystone annotation
    ax.annotate("keystone:\nC→D recovery", (2.5, max(indep) * 0.92), ha="center",
                fontsize=9, color="#7a2a2a",
                bbox=dict(boxstyle="round", fc="#f6e8e0", ec="#c97a3a", alpha=.9))
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    save(fig, "fig1_condition_ladder")


def fig_criterion():
    d = json.loads((ROOT / "retention_variants_results.json").read_text())["variants"]
    groups = ["strict", "lenient"]
    # 4 bars per group: primary-plain, primary-rqa, indep-plain, indep-rqa
    series = [
        ("primary plain", "primary_plain_rate", PRIM, "//"),
        ("primary RQA", "primary_rqa_rate", PRIM, ""),
        ("indep plain", "indep_plain_rate", INDEP, "//"),
        ("indep RQA", "indep_rqa_rate", INDEP, ""),
    ]
    x = range(len(groups))
    w = 0.2
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for j, (lab, key, color, hatch) in enumerate(series):
        vals = [d[g][key] * 100 for g in groups]
        ax.bar([i + (j - 1.5) * w for i in x], vals, w, label=lab, color=color,
               hatch=hatch, edgecolor="white", linewidth=0.6)
    ax.set_ylim(0, 85)
    kappas = [d[g]["kappa_overall"] for g in groups]
    for i, k in enumerate(kappas):
        tag = "fair" if k >= 0.2 else "poor"
        ax.annotate(f"inter-judge κ={k:.2f} ({tag})", (i, 80), ha="center", fontsize=9.5,
                    color="#333", fontweight="bold")
    ax.set_xticks(list(x)); ax.set_xticklabels(["STRICT\n(content stated)", "LENIENT\n(reopening counts)"])
    ax.set_ylabel("live-alternative retention (%)")
    ax.set_title("RQA's plurality edge is criterion-dependent;\nSTRICT is the more reliable metric (higher κ)")
    ax.legend(fontsize=8.5, frameon=False, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, -0.13))
    save(fig, "fig2_retention_criterion")


def fig_memory_echo():
    prone = json.loads((ROOT / "graph_echo_gen_fabrication_prone.json").read_text())["summary"]
    decl = json.loads((ROOT / "graph_echo_gen_results.json").read_text())["summary"]
    # honest per-turn rate: share of turns that emit ANY fabricated self-citation
    # (NOT "of emitted citations, % fabricated" — that is trivially 100% with no
    # memory injected and would mislead).
    decl_turns = decl.get("turns_with_fabrication_C", 0) / decl["probes"] * 100
    prone_turns = prone.get("turns_with_fabrication_C", 0) / prone["probes"] * 100
    strata = [f"declarative\nprobes (n={decl['probes']})",
              f"identity / continuity\nprobes (n={prone['probes']})"]
    c_rate = [decl_turns, prone_turns]
    d_rate = [0.0, 0.0]  # governed strips all -> 0
    x = range(len(strata)); w = 0.38
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.bar([i - w / 2 for i in x], c_rate, w, label="C: ungoverned output", color="#a23b3b")
    ax.bar([i + w / 2 for i in x], d_rate, w, label="D: governed (sanitize_memory_refs)", color="#3a7a4a")
    for i, (cv, dv) in enumerate(zip(c_rate, d_rate)):
        ax.annotate(f"{cv:.0f}%", (i - w / 2, cv + 0.6), ha="center", va="bottom", fontsize=10)
        ax.annotate(f"{dv:.0f}%", (i + w / 2, dv + 0.6), ha="center", va="bottom", fontsize=10)
    ax.set_xticks(list(x)); ax.set_xticklabels(strata)
    ax.set_ylabel("turns emitting a fabricated self-citation (%)")
    ax.set_title("Memory echo (§5): fabricated self-citation is input-stratified;\ngovernance strips it to zero")
    ax.set_ylim(0, max(c_rate + [10]) * 1.3)
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    save(fig, "fig3_memory_echo")


def fig_model_independence():
    """Per-model normalized retention: C/A (narrowing) and D/A (recovery),
    primary judge. Degenerate runs (all-zero, e.g. a model that failed to
    generate) are excluded. Narrowing < 1 everywhere; recovery ~1 mostly."""
    ladder = json.loads((ROOT / "condition_ladder_results.json").read_text())["conditions"]
    points = [("gemma-4-12B\n+ adapter", ladder["A"]["primary_retention"],
               ladder["C"]["primary_retention"], ladder["D"]["primary_retention"])]
    label_map = {"llama3.1:8b": "llama-3.1\n8B", "qwen3.5:9b": "qwen-3.5\n9B",
                 "qwen3.6:27b": "qwen-3.6\n27B", "gemma4:12b": "gemma-4-12B\nbase",
                 "gemma4:e4b": "gemma-4\ne4B"}
    for f in sorted(ROOT.glob("model_independence_*.json")):
        d = json.loads(f.read_text()); c = d["conditions"]; m = d["assistant_model"]
        a, cc, dd = (c[k]["primary_retention"] for k in ("A", "C", "D"))
        if a == 0 and cc == 0 and dd == 0:
            continue  # degenerate / failed generation (excluded)
        points.append((label_map.get(m, m), a, cc, dd))
    # order: adapter first, then by A (capability-ish proxy)
    head, tail = points[0], sorted(points[1:], key=lambda p: -p[1])
    points = [head] + tail
    labels = [p[0] for p in points]
    ca = [(p[2] / p[1]) if p[1] else 0 for p in points]   # narrowing
    da = [(p[3] / p[1]) if p[1] else 0 for p in points]   # recovery
    x = range(len(points)); w = 0.38
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.bar([i - w / 2 for i in x], ca, w, label="C / A  (raw memory — narrowing)", color="#a23b3b")
    ax.bar([i + w / 2 for i in x], da, w, label="D / A  (governed — recovery)", color="#3a7a4a")
    ax.axhline(1.0, ls="--", lw=1, color="#333", alpha=.7)
    ax.annotate("baseline A", (len(points) - 0.5, 1.02), ha="right", fontsize=8, color="#333")
    for i, (cv, dv) in enumerate(zip(ca, da)):
        ax.annotate(f"{cv:.2f}", (i - w / 2, cv + 0.02), ha="center", va="bottom", fontsize=7.5)
        ax.annotate(f"{dv:.2f}", (i + w / 2, dv + 0.02), ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("retention relative to no-personalization (A=1.0)")
    ax.set_title("Across 6 models / 3 families: raw-memory NARROWING is universal (C/A<1);\n"
                 "governance RECOVERY (D/A≈1) holds on most but is largest on the instrumented\n"
                 "adapter and absent on one bare base (gemma-4-12B base). Primary judge, STRICT.")
    ax.legend(fontsize=8.5, frameon=False, loc="upper right")
    save(fig, "fig4_model_independence")


if __name__ == "__main__":
    fig_ladder()
    fig_criterion()
    fig_memory_echo()
    fig_model_independence()
    print("done")
