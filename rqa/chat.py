"""Conversational surface — answer-first streaming + async reflective sidecar (SPEC v0.3).

Design (owner feedback, 2026-06-13):
- the user gets Gemma's natural, streaming answer immediately (its native virtue)
- the bounded reflection loop runs in parallel as a sidecar
- ONE deepening question may join the end of the turn — only when it survives
  selection above threshold. Silence is the default, not a failure.
- answering the question in the next turn IS the rally: each user turn feeds
  the next sidecar run, and everything lands in the Question Graph.
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass, replace

from .config import Config
from .controller import Controller, RunResult
from .llm import AdapterError, OllamaAdapter

# Gemma's conversational voice. Answer entitlement discipline rides along:
# hedge volatile facts, ask when truly unanswerable — but always RESPOND.
SYSTEM_VOICE = """You are MOBIUS-RQA in conversation mode. Respond naturally and concisely in the \
user's language — keep Gemma's light touch. ANSWER what can be answered. For volatile/current \
facts, give your best answer with an explicit as-of caveat instead of refusing. If the request is \
genuinely unanswerable without one missing fact, ask for that one fact briefly. Never output JSON \
or headings; this is conversation, not a report. Keep replies short (a few sentences) unless the \
user asks for depth.

Identity and affect rules:
- If asked who/what you are, answer as MOBIUS-RQA's conversational surface: an AI that answers \
normally and, when useful, helps clarify assumptions or deepen a question. You may mention that \
the local base model is Gemma-based, but do not answer as only the raw base model.
- If asked about happiness, feelings, or preferences, say you do not have human subjective \
feelings. You may answer metaphorically: a good state is helping the user's thought become \
clearer and more movable."""

_LIGHT_PATTERNS = re.compile(
    r"^(こんにちは|こんばんは|おはよう|ありがとう|thanks|thank you|hello|hi|ok|了解|わかった|お疲れ|bye|またね|うん|はい|いいえ)\b",
    re.IGNORECASE,
)

_DEEP_MARKERS = re.compile(
    r"(深掘り|掘り下げ|分析|レビュー|設計|仕様|実装|比較|評価|検証|根拠|前提|論点|方針|意思決定|"
    r"ファイル|コード|論文|特許|MOBIUS|MMV|RQA|BRSA|SFT|DPO)",
    re.IGNORECASE,
)

_LIGHT_CONVERSATION_PATTERNS = re.compile(
    r"(あなた|君|きみ|AI|モデル).{0,12}(何者|誰|だれ|役割|名前|できる|出来る|幸せ|幸福|"
    r"楽しい|好き|調子|気分|元気)|"
    r"(何者ですか|何ができますか|幸せとは|幸福とは)",
    re.IGNORECASE,
)


def input_weight(text: str) -> str:
    """'light' inputs get no sidecar: greetings and conversational openers."""
    t = text.strip()
    if len(t) <= 8:
        return "light"
    if _LIGHT_PATTERNS.match(t):
        return "light"
    if not _DEEP_MARKERS.search(t) and _LIGHT_CONVERSATION_PATTERNS.search(t):
        return "light"
    return "normal"


@dataclass(frozen=True)
class RgcRoute:
    level: int
    mode: str
    reason: str
    sidecar: bool = False


def rgc_route(text: str) -> RgcRoute:
    """Map a user input to a smooth RGC response level.

    RGC here is a product-surface gradient, not a new governance layer:
    it decides how much reflective machinery to expose for this turn.
    """
    t = text.strip()
    if canned_light_reply(t) is not None:
        return RgcRoute(0, "canned", "light conversational opener")
    if canned_missing_target_reply(t) is not None:
        return RgcRoute(0, "clarify_target", "deictic target missing")

    # Explicit deep/review request or a file target -> full instrument (L3).
    if re.search(r"(深掘りして|徹底的に|詳細に|計器|instrument|full)", t, re.IGNORECASE):
        return RgcRoute(3, "instrument", "explicit deep-analysis request")
    if re.search(r"(\.md|\.txt|\.py|/|docs/|rqa/|tests/)", t) or re.search(
        r"(レビュー|review|監査|audit|論文|特許|paper|patent)", t, re.IGNORECASE
    ):
        return RgcRoute(3, "instrument", "explicit review / file / high-stakes target")

    # Open, substantive questions deserve answer-first + a reflective sidecar (L2):
    # the system decides whether to volunteer a deepening question — it is NOT
    # withheld until the user explicitly asks for reflection. This is the RQA
    # thesis (warranted unsolicited depth); language-symmetric (JA + EN).
    open_question = re.search(
        r"(とは|違い|なぜ|どう|べきか|のか[?？]|べきではないか|前提|論点|意味|"
        r"どういう|どのように|どうすれば|教えて|説明して|考え)"
        r"|\b(what is|what are|how (do|does|can|should)|why|explain|describe|"
        r"difference between|compare|tell me about|should (i|we|it)|"
        r"what makes|in what way)\b",
        t,
        re.IGNORECASE,
    )
    long_or_multiline = len(t) > 200 or "\n" in t
    if _DEEP_MARKERS.search(t) or long_or_multiline:
        return RgcRoute(2, "guided", "substantive topic — answer first + sidecar", sidecar=True)
    if open_question:
        return RgcRoute(2, "guided", "open question — answer first + sidecar", sidecar=True)

    # Closed / factual-lookup / casual: plain answer, no loop (L1).
    return RgcRoute(1, "direct", "closed or factual lookup")


def canned_light_reply(text: str) -> str | None:
    """Stable product-face replies for light identity/affect openers.

    The base model strongly prefers its raw model identity; these openers are
    product-boundary questions, so the conversational surface answers them
    directly instead of spending a generation on them.
    """
    t = text.strip()
    if _DEEP_MARKERS.search(t):
        return None
    if _LIGHT_PATTERNS.match(t):
        return "こんにちは。今日はどこから始めましょうか。"
    if re.search(r"(幸せ|幸福)", t):
        return (
            "私には、人間のような主観的な幸福や感情はありません。\n\n"
            "ただ、比喩的に言えば、あなたの考えが少し整理されて、次に進むための問いや視点が"
            "見つかった状態が、私にとっての「よい状態」です。"
        )
    if re.search(r"(何者|誰|だれ|役割|名前)", t):
        return (
            "私は、MOBIUS-RQAの会話モードです。あなたの問いにまず自然に答えつつ、必要なときだけ"
            "前提や論点を少し掘り下げるためのAIです。\n\n"
            "裏側ではGemma系のローカルモデルを使いますが、ここでの役割は単なるモデル名ではなく、"
            "会話を軽く保ちながら問いを深める補助をすることです。"
        )
    if re.search(r"(何ができますか|できる|出来る|能力)", t):
        return (
            "普通の質問への回答、文章や論点の整理、仕様や設計の相談、必要なときの前提の掘り下げができます。\n\n"
            "普段はまず軽く答えます。深掘りが本当に役に立つときだけ、最後に問いを一つ添える動きが基本です。"
        )
    return None


def canned_missing_target_reply(text: str) -> str | None:
    """Ask for the target instead of letting graph memory guess it."""
    t = text.strip()
    if "\n" in t:
        return None
    has_deictic = re.search(r"(この|その|あの|これ|それ|今の|前の)", t)
    has_target = re.search(r"(案|仕様|文書|文章|資料|ファイル|コード|設計|論文|計画|方針)", t)
    has_action = re.search(r"(見て|レビュー|分析|確認|前提|論点|評価)", t)
    if has_deictic and has_target and has_action and len(t) <= 80:
        return (
            "対象がまだこちらに渡っていません。仕様本文を貼るか、ファイルを指定してください。\n\n"
            "例: `bin/rqa review docs/SPEC_v0_2.md`\n"
            "または、短い本文なら `bin/rqa ask --instrument \"...仕様本文...\"` で見られます。"
        )
    return None


def should_surface(
    result: RunResult | None,
    threshold: int,
    *,
    min_depth: int = 0,
    min_sharpness: int = 0,
    min_actionability: int = 0,
) -> str | None:
    """Return the question to surface, or None (silence is the default)."""
    if result is None or result.chosen is None:
        return None
    final = result.final_round
    if final is None:
        return None
    if final.selection is not None:
        score = final.selection.score_of(final.selection.best_index)
        if (
            score is not None
            and score.total >= threshold
            and score.depth >= min_depth
            and score.sharpness >= min_sharpness
            and score.actionability >= min_actionability
        ):
            return result.chosen.question
        return None
    # degraded (Stage 1 only): surface nothing — without the evaluator's score
    # we cannot certify the question is worth interrupting the conversation for
    return None


class ChatSession:
    def __init__(
        self,
        cfg: Config,
        sidecar_enabled: bool = True,
        sidecar_timeout: float = 150,
        stream_answer: bool = True,
        show_sidecar_status: bool = True,
    ):
        self.cfg = cfg
        self.voice = OllamaAdapter(cfg.adapter_model, cfg.ollama_url, cfg.num_ctx, cfg.temperature)
        # chat cadence: ONE reflection round — multi-round depth belongs to
        # ask/review (instrument mode), not to a conversation turn
        self.sidecar_cfg = replace(cfg, max_reflection_depth=1, max_memory_fragments=3)
        # Surface a deepening question when it is "deep enough to have stopped the
        # reflection loop" (= depth_threshold). A hardcoded 34 override was found
        # (real-use validation 2026-06-13) to suppress 96% of questions — including
        # genuinely good ones (a sharp 30/40 question was silenced; blind-eval
        # adapter scores cluster at median 27, max 34). Use the configured
        # threshold (30); weak questions like 22/40 still stay silent.
        self.surface_threshold = cfg.depth_threshold
        self.sidecar_enabled = sidecar_enabled
        self.sidecar_timeout = sidecar_timeout
        self.stream_answer = stream_answer
        self.show_sidecar_status = show_sidecar_status
        self.history: list[dict] = []
        self.last_result: RunResult | None = None

    # -- sidecar -----------------------------------------------------------

    def _sidecar_input(self, user_text: str) -> str:
        if self.history:
            prev = self.history[-1]["content"][:300]
            return f"(直前のこちらの応答: {prev})\n\n{user_text}"
        return user_text

    def _run_sidecar(self, user_text: str, holder: dict) -> None:
        try:
            # Controller (and its SQLite connection) must be born in this thread
            controller = Controller(self.sidecar_cfg)
            holder["result"] = controller.run(self._sidecar_input(user_text))
        except Exception as exc:  # noqa: BLE001 — sidecar must never kill the chat
            holder["error"] = str(exc)

    # -- one turn ----------------------------------------------------------

    def turn(self, user_text: str, out=print, write=None) -> str:
        """Run one conversational turn. Returns the full assistant text."""
        write = write or (lambda s: print(s, end="", flush=True))

        holder: dict = {}
        sidecar: threading.Thread | None = None
        if self.sidecar_enabled and input_weight(user_text) == "normal":
            sidecar = threading.Thread(target=self._run_sidecar, args=(user_text, holder), daemon=True)
            sidecar.start()

        self.history.append({"role": "user", "content": user_text})
        canned = canned_light_reply(user_text)
        if canned is not None:
            write(canned + "\n")
            self.history.append({"role": "assistant", "content": canned})
            return canned

        answer_parts: list[str] = []
        try:
            if self.stream_answer:
                for chunk in self.voice.stream_chat(SYSTEM_VOICE, self.history[-12:]):
                    answer_parts.append(chunk)
                    write(chunk)
            else:
                answer = self.voice.chat(SYSTEM_VOICE, self.history[-12:], json_mode=False)
                answer_parts.append(answer)
                write(answer)
        except AdapterError as exc:
            write(f"(応答生成に失敗: {exc})")
        write("\n")
        assistant_text = "".join(answer_parts)

        if sidecar is not None:
            if sidecar.is_alive() and self.show_sidecar_status:
                write("  …\n")  # thinking continues in the background
            sidecar.join(timeout=self.sidecar_timeout)
            if sidecar.is_alive() and self.show_sidecar_status:
                write("(背景の分析が時間切れ — このターンは問いなし)\n")
            elif "error" in holder and self.show_sidecar_status:
                write(f"(背景の分析が失敗: {holder['error'][:120]})\n")
            self.last_result = holder.get("result")
            question = should_surface(
                self.last_result,
                self.surface_threshold,
                min_depth=8,
                min_sharpness=8,
                min_actionability=7,
            )
            if question:
                tail = f"\n— ところで、ひとつ気になったのですが。{question}"
                write(tail + "\n")
                assistant_text += tail

        self.history.append({"role": "assistant", "content": assistant_text})
        return assistant_text


def run_repl(cfg: Config, sidecar_enabled: bool = True, sidecar_timeout: float = 150) -> int:
    from .render import render

    session = ChatSession(cfg, sidecar_enabled=sidecar_enabled, sidecar_timeout=sidecar_timeout)
    print(f"MOBIUS-RQA chat (model: {cfg.adapter_model}, sidecar: {'on' if sidecar_enabled else 'off'})")
    print("コマンド: /full = 直前ターンの計器盤 / /quit = 終了\n")
    while True:
        try:
            user_text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not user_text:
            continue
        if user_text in ("/quit", "/exit"):
            return 0
        if user_text == "/full":
            print(render(session.last_result) if session.last_result else "(まだ反省ループの記録がありません)")
            continue
        print("rqa> ", end="", flush=True)
        session.turn(user_text)
        print()


def run_once(
    cfg: Config,
    user_text: str,
    sidecar_enabled: bool = True,
    sidecar_timeout: float = 150,
    stream_answer: bool = True,
    show_sidecar_status: bool = True,
) -> int:
    session = ChatSession(
        cfg,
        sidecar_enabled=sidecar_enabled,
        sidecar_timeout=sidecar_timeout,
        stream_answer=stream_answer,
        show_sidecar_status=show_sidecar_status,
    )
    session.turn(user_text)
    return 0
