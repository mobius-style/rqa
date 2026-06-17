"""Human-facing output (SPEC_v0_2.md §11.2)."""
from __future__ import annotations

from .controller import RunResult


def _format_memory_ref(ref) -> str:
    text = str(ref or "?").strip() or "?"
    if text.lower().startswith("node "):
        return text
    return f"node {text}"


def render_brief(result: RunResult) -> str:
    """Question-only view: the selected question, runner-ups, one status line."""
    final = result.final_round
    if final is None or final.chosen is None:
        return "(有効な問いが得られませんでした)"
    lines = [f"Q: {final.chosen.question}"]
    for c in final.shortlist:
        if c.question != final.chosen.question:
            lines.append(f"   次点: {c.question}")
    score = final.selection.score_of_best_total() if final.selection else None
    status = f"({score}/40" if score is not None else "(縮退選別"
    status += f", {len(result.rounds)}ラウンド"
    if result.memory_refs_stripped:
        status += f", 捏造参照{len(result.memory_refs_stripped)}件破棄"
    lines.append(status + ")")
    return "\n".join(lines)


def render(result: RunResult) -> str:
    lines: list[str] = []
    final = result.final_round
    if final is None:
        return "実行が完了しませんでした(ラウンド0)。"
    fm = final.analysis.feature_map

    def section(title: str) -> None:
        lines.append("")
        lines.append(f"## {title}")

    def bullets(items, empty="(なし)") -> None:
        items = [i for i in (items or []) if isinstance(i, str) and i.strip()]
        if items:
            lines.extend(f"- {i}" for i in items)
        else:
            lines.append(empty)

    lines.append(f"# RQA 出力 (session {result.session_id})")

    section("1. 入力の主張")
    bullets(fm.get("claims"))

    section("2. 検出された違和感")
    lines.append("**入力内在:**")
    bullets(fm.get("tensions_input_internal"))
    mc = fm.get("tensions_memory_cross") or []
    if mc:
        lines.append("**過去の記録との緊張:**")
        for t in mc:
            if isinstance(t, dict) and t.get("tension"):
                lines.append(
                    f"- {t['tension']} ({_format_memory_ref(t.get('memory_ref'))}, "
                    f"confidence: {t.get('confidence', '?')})"
                )

    section("3. 隠れた前提")
    bullets(fm.get("assumptions"))

    if fm.get("frames"):
        section("3b. 認識枠")
        bullets(fm.get("frames"))

    section("4. 検索判断")
    sd = final.analysis.search_decision
    if sd.get("search_needed"):
        lines.append(f"検索が必要: {sd.get('reason', '')}")
        bullets(sd.get("queries"))
    else:
        lines.append(f"検索不要。{sd.get('reason', '')}")

    section("5. より深い問い(選別済み)")
    if final.chosen:
        lines.append(f"**{final.chosen.question}**")
        lines.append(
            f"  (layer={final.chosen.target_layer}, stance={final.chosen.stance}, "
            f"target={final.chosen.target_element})"
        )
        if final.selection:
            s = final.selection.score_of(final.selection.best_index)
            if s:
                lines.append(
                    f"  Evaluator採点: {s.total}/40 "
                    f"(depth {s.depth} / sharpness {s.sharpness} / "
                    f"novelty {s.novelty} / actionability {s.actionability})"
                )
            if final.selection.rationale:
                lines.append(f"  選別理由: {final.selection.rationale}")
        if final.degraded:
            lines.append("  ※ Stage 2縮退 — ローカル自己順位付けのみで選別(§5.2)")
    else:
        lines.append("(有効な候補が得られませんでした)")

    runners = [c for c in final.shortlist if final.chosen and c.question != final.chosen.question]
    if runners:
        section("5b. 次点候補")
        for c in runners:
            lines.append(f"- {c.question} ({c.target_layer}/{c.stance})")

    if final.analysis.self_update_proposal:
        section("6. 自己理解層への更新候補")
        p = final.analysis.self_update_proposal
        lines.append(f"- 領域: {p.get('allowed_area', '?')}")
        lines.append(f"- 観察: {p.get('observation', '')}")
        lines.append(f"- 調整案: {p.get('proposed_adjustment', '')}")

    section("7. 境界チェック(Governor発行)")
    gr = result.governor_report
    if gr is None:
        lines.append("(未実施)")
    elif gr.ok:
        lines.append("違反なし。")
    else:
        lines.extend(f"- 違反: {v}" for v in gr.violations)
    if result.fragments_filtered:
        lines.append(
            f"- 注入フィルタ: Essentialsライク断片 {result.fragments_filtered} 件を遮断(§5.4)"
        )
    if result.memory_refs_stripped:
        lines.append(
            f"- 記憶参照検証: 根拠のない記憶参照 {len(result.memory_refs_stripped)} 件を破棄"
            f"({', '.join(result.memory_refs_stripped[:3])})"
        )
    if result.injected_fragments:
        lines.append(f"- 記憶注入: {len(result.injected_fragments)} 断片")
    lines.append(f"- 反省ラウンド: {len(result.rounds)} / 停止理由: {result.stop_reason}")

    return "\n".join(lines)
