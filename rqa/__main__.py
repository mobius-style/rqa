"""MOBIUS-RQA CLI.

  python -m rqa chat                 conversational mode (streaming + sidecar)
  python -m rqa ask "..."            run the reflection loop on a text
  python -m rqa review path.md       run on a document file
  python -m rqa graph stats          Question Graph statistics
  python -m rqa graph search "..."   raw retrieval (post-filter view)
  python -m rqa check                environment / binding health check
"""
from __future__ import annotations

import argparse
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

from . import governor
from .config import Config
from .controller import Controller
from .evaluator import GroqEvaluator
from .graph import QuestionGraph
from .llm import OllamaAdapter
from .render import render


_L3_PROGRESS_MESSAGES = (
    "中間観測: 入力の主張と隠れた前提を分けています。まだ結論ではありません。",
    "中間観測: 候補を選別しています。採点前の問いはまだ表に出しません。",
    "中間観測: 境界条件と記憶参照を確認しています。固まり次第まとめます。",
)


@dataclass(frozen=True)
class InstrumentBudget:
    name: str
    k_candidates: int
    max_reflection_depth: int
    max_memory_fragments: int
    max_regen_per_round: int
    progress_interval: float
    reason: str


_BUDGETS = {
    "quick": InstrumentBudget(
        name="quick",
        k_candidates=3,
        max_reflection_depth=1,
        max_memory_fragments=2,
        max_regen_per_round=0,
        progress_interval=10,
        reason="short or light instrument request",
    ),
    "standard": InstrumentBudget(
        name="standard",
        k_candidates=4,
        max_reflection_depth=2,
        max_memory_fragments=4,
        max_regen_per_round=1,
        progress_interval=12,
        reason="moderate instrument request",
    ),
    "full": InstrumentBudget(
        name="full",
        k_candidates=6,
        max_reflection_depth=3,
        max_memory_fragments=8,
        max_regen_per_round=1,
        progress_interval=12,
        reason="document, audit, or high-stakes instrument request",
    ),
}


def _instrument_budget(text: str, *, review: bool = False) -> InstrumentBudget:
    if review:
        return _BUDGETS["full"]
    t = text.strip()
    heavy = (
        len(t) > 500
        or "\n" in t
        or any(marker in t for marker in ("徹底", "監査", "公開", "論文", "特許", "評価", "検証"))
        or any(marker in t for marker in (".md", ".py", "docs/", "tests/", "rqa/", "/"))
    )
    if heavy:
        return _BUDGETS["full"]
    moderate = (
        len(t) > 120
        or any(marker in t for marker in ("設計", "仕様", "前提", "論点", "レビュー", "比較", "方針"))
    )
    if moderate:
        return _BUDGETS["standard"]
    return _BUDGETS["quick"]


def _apply_budget(cfg: Config, budget: InstrumentBudget, args) -> None:
    if not getattr(args, "k", None):
        cfg.k_candidates = budget.k_candidates
    if not getattr(args, "depth", None):
        cfg.max_reflection_depth = budget.max_reflection_depth
    cfg.max_memory_fragments = budget.max_memory_fragments
    cfg.max_regen_per_round = budget.max_regen_per_round
    cfg.shortlist_s = min(cfg.shortlist_s, cfg.k_candidates)


def _print_prelude(level: int) -> None:
    if level == 2:
        print("短く言うと、まず自然に答え、そのうえで必要なら前提を一段だけ確認します。\n", flush=True)
    elif level == 3:
        print("短く言うと、これは計器モードで深く見る入力です。結果が固まったらまとめて出します。\n", flush=True)


def _print_budget(budget: InstrumentBudget) -> None:
    print(
        f"反射予算: {budget.name} "
        f"(k={budget.k_candidates}, depth={budget.max_reflection_depth}, memory={budget.max_memory_fragments})\n",
        flush=True,
    )


def _run_with_progress(runner, *, interval: float = 12):
    holder = {}

    def target() -> None:
        try:
            holder["result"] = runner()
        except BaseException as exc:  # noqa: BLE001 — re-raise in the main thread
            holder["error"] = exc

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    index = 0
    while worker.is_alive():
        worker.join(timeout=interval)
        if worker.is_alive():
            msg = _L3_PROGRESS_MESSAGES[min(index, len(_L3_PROGRESS_MESSAGES) - 1)]
            print(msg, flush=True)
            index += 1
    if "error" in holder:
        raise holder["error"]
    return holder.get("result")


def _make_config(args) -> Config:
    cfg = Config()
    if getattr(args, "k", None):
        cfg.k_candidates = args.k
    if getattr(args, "depth", None):
        cfg.max_reflection_depth = args.depth
    if getattr(args, "model", None):
        cfg.adapter_model = args.model
    if getattr(args, "no_evaluator", False):
        cfg.evaluator_enabled = False
    return cfg


def cmd_ask(args) -> int:
    cfg = _make_config(args)
    text = args.text
    if text == "-":
        text = sys.stdin.read()
    prelude_level = 3 if getattr(args, "instrument", False) and not getattr(args, "brief", False) else None
    if not any(
        (
            getattr(args, "instrument", False),
            getattr(args, "brief", False),
            getattr(args, "k", None),
            getattr(args, "depth", None),
            getattr(args, "no_evaluator", False),
        )
    ):
        from .chat import canned_light_reply, canned_missing_target_reply, rgc_route, run_once

        reply = canned_light_reply(text) or canned_missing_target_reply(text)
        if reply is not None:
            print(reply)
            return 0
        route = rgc_route(text)
        if route.mode == "direct":
            return run_once(cfg, text, sidecar_enabled=False)
        if route.mode == "guided":
            _print_prelude(2)
            # answer streams first; the sidecar gets a real (not 6s) window so a
            # warranted question can actually land — one reflection round is
            # ~15-30s, and 6s guaranteed a timeout (question never surfaced).
            return run_once(
                cfg,
                text,
                sidecar_enabled=True,
                sidecar_timeout=30,
                stream_answer=False,
                show_sidecar_status=True,
            )
        if route.mode == "instrument":
            prelude_level = 3
    if prelude_level is not None:
        _print_prelude(prelude_level)
    budget = _instrument_budget(text)
    _apply_budget(cfg, budget, args)
    if prelude_level == 3:
        _print_budget(budget)
        result = _run_with_progress(lambda: Controller(cfg).run(text), interval=budget.progress_interval)
    else:
        controller = Controller(cfg)
        result = controller.run(text)
    if getattr(args, "brief", False):
        from .render import render_brief

        print(render_brief(result))
    else:
        print(render(result))
    return 0


def cmd_chat(args) -> int:
    from .chat import run_once, run_repl

    cfg = _make_config(args)
    if getattr(args, "text", None):
        return run_once(cfg, " ".join(args.text), sidecar_enabled=not getattr(args, "no_sidecar", False))
    return run_repl(cfg, sidecar_enabled=not getattr(args, "no_sidecar", False))


def cmd_review(args) -> int:
    cfg = _make_config(args)
    path = Path(args.path)
    if not path.is_file():
        print(f"file not found: {path}", file=sys.stderr)
        return 1
    body = path.read_text(encoding="utf-8")
    preamble = (
        f"以下は文書『{path.name}』である。レビュー対象として読み、"
        "主張・違和感・暗黙の前提・未検証の仮定を抽出し、"
        "この文書の作成者が答えるべき最も深い問いを生成せよ。\n\n"
    )
    budget = _instrument_budget(body, review=True)
    _apply_budget(cfg, budget, args)
    _print_prelude(3)
    _print_budget(budget)
    result = _run_with_progress(lambda: Controller(cfg).run(preamble + body), interval=budget.progress_interval)
    print(render(result))
    return 0


def cmd_graph(args) -> int:
    cfg = Config()
    graph = QuestionGraph(cfg.graph_db)
    if args.graph_cmd == "stats":
        stats = graph.stats()
        print(f"db: {stats['db_path']}")
        print(f"audit entries: {stats['audit_entries']}")
        for row in stats["nodes"]:
            print(f"  {row['kind']:10s} {row['status']:10s} {row['count']}")
    elif args.graph_cmd == "search":
        fragments = [f.as_dict() for f in graph.search(args.query, top_k=10)]
        kept, dropped = governor.filter_fragments(fragments)
        for f in kept:
            print(f"[{f['score']:.3f}] (node {f['node_id']}, {f['kind']}, {f['status']}) {f['text'][:120]}")
        if dropped:
            print(f"({dropped} fragment(s) withheld by Essentials filter)")
    return 0


def cmd_check(args) -> int:
    cfg = Config()
    adapter = OllamaAdapter(cfg.adapter_model, cfg.ollama_url, cfg.num_ctx, cfg.temperature)
    print(f"adapter model     : {cfg.adapter_model}")
    print(f"  ollama reachable: {'ok' if adapter.health() else 'NG (model missing or ollama down)'}")
    ev = GroqEvaluator(cfg.evaluator_binding)
    print(f"evaluator binding : {cfg.evaluator_binding.release} ({cfg.evaluator_binding.model})")
    print(f"  api key         : {'ok' if ev.available() else 'NOT FOUND -> runs degrade to Stage 1'}")
    print(f"graph db          : {cfg.graph_db}")
    print(f"num_ctx           : {cfg.num_ctx}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="rqa", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ask = sub.add_parser("ask", help="run reflection loop on a text")
    p_ask.add_argument("text", help="input text, or '-' for stdin")
    p_ask.add_argument("--k", type=int, help="candidates per round (default 6)")
    p_ask.add_argument("--depth", type=int, help="max reflection rounds (default 3)")
    p_ask.add_argument("--model", help="override adapter model")
    p_ask.add_argument("--no-evaluator", action="store_true", help="force Stage 1 only")
    p_ask.add_argument("--brief", action="store_true", help="print only the selected question(s)")
    p_ask.add_argument("--instrument", action="store_true", help="force full dashboard even for light inputs")
    p_ask.set_defaults(func=cmd_ask)

    p_chat = sub.add_parser("chat", help="conversational mode (streaming + reflective sidecar)")
    p_chat.add_argument("--model", help="override model")
    p_chat.add_argument("--no-sidecar", action="store_true", help="plain chat, no reflection")
    p_chat.add_argument("--no-evaluator", action="store_true")
    p_chat.add_argument("text", nargs="*", help="optional one-shot message; omit for REPL")
    p_chat.set_defaults(func=cmd_chat)

    p_rev = sub.add_parser("review", help="run on a document file")
    p_rev.add_argument("path")
    p_rev.add_argument("--k", type=int)
    p_rev.add_argument("--depth", type=int)
    p_rev.add_argument("--model")
    p_rev.add_argument("--no-evaluator", action="store_true")
    p_rev.set_defaults(func=cmd_review)

    p_graph = sub.add_parser("graph", help="Question Graph operations")
    graph_sub = p_graph.add_subparsers(dest="graph_cmd", required=True)
    graph_sub.add_parser("stats")
    p_search = graph_sub.add_parser("search")
    p_search.add_argument("query")
    p_graph.set_defaults(func=cmd_graph)

    p_check = sub.add_parser("check", help="environment health check")
    p_check.set_defaults(func=cmd_check)

    args = parser.parse_args()
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001 — readable error, not a traceback
        print(f"rqa error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
