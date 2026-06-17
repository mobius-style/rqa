# MOBIUS-RQA / BRSA 仕様書 v0.4

## 0. 文書情報

**名称**: MOBIUS-RQA / BRSA
**正式候補名**: MOBIUS Reflective Questioning Adapter / Bounded Reflective Self-Understanding Adapter
**略称**: RQA / BRSA
**版**: v0.4(2026-06-13)
**前版**: v0.2全文 + v0.3差分(会話表面)
**位置づけ**: MOBIUS_MMV の兄弟プロジェクト。MMV本体への内蔵機能ではなく、独立した反省型問い生成プロジェクト。
**対象**: ローカルLLMに接続されるRQAアダプタ、外部エージェント制御系、会話表面、計器モード、学習・評価・昇格手順。
**想定ベースモデル**: Gemma 4 12B(主)、Qwen系、Phi系、その他商用利用可能なローカルLLM。
**現行実装モデル**: コード既定は `gemma4:12b`。学習済みアダプタは Ollama `rqa-gemma4:v0.1` として利用可能だが、既定昇格は Gate D の人間承認待ち。
**目的**: 回答生成そのものではなく、問いの深化・前提掘削・認識形式の更新支援を、境界統治つきで行うこと。

v0.4は差分仕様ではない。v0.2の探索・選別・蓄積アーキテクチャ、v0.3の会話表面、2026-06-13の実機観測とRGC段階化を統合した全文仕様である。

### v0.4で統合された主な変更

```txt
1. RQAをMMV_MMVの下位機能ではなく、兄弟関係の独立プロジェクトとして明文化。

2. v0.2の「問いのみ出力」を訂正し、内部報酬と会話表面を分離。
   内部報酬は「境界内でより深い問い」だが、会話ではまず自然に答える。

3. chat / ask / review の製品面を再定義。
   chat: 会話モード。
   ask: RGC段階化された単発入口。
   review: 文書レビュー用の計器モード。

4. RGC応答段階器を導入。
   L0: 固定応答または対象確認。
   L1: 直接回答。
   L2: 即時短文 + 通常回答 + 短時間サイドカー。
   L3: 計器盤出力。

5. L2/L3に二段出力を導入。
   初期待機の体感上限を約2秒と見なし、先に短文を出す。

6. L3内部にも quick / standard / full の反射予算を導入。
   短い計器入力が常に最大予算を消費する問題を解消。

7. 記憶参照捏造をGovernor側で機械検証。
   モデルが引用した node は、このターンに実際に注入された node ID と照合する。

8. Ollama thinkingモード障害に対し、`think:false` 常時送信を実装。

9. Phase 1 SFTコーパス、QLoRA学習、Raw対比評価、Ollama統合、v0.3/v0.4会話表面までを記録。
```

---

# 1. 概要

MOBIUS-RQA / BRSA は、凍結されたローカルLLMに接続される反省型問い生成アダプタ・システムである。

本システムは、入力に対してただ「深そうな問い」を返すものではない。ユーザー入力に含まれる主張、違和感、欠落、矛盾、暗黙の前提、認識枠を抽出し、必要に応じて外部知識や過去のQuestion Graphと照合し、境界条件を守りながら次に答える価値のある問いを生成する。

ただし、v0.4では次を明確に区別する。

```txt
内部報酬:
  境界条件を守りながら、より深い問いを立てること。

製品表面:
  ユーザーには、まず自然に答える。
  深化が本当に有用な時だけ、問いや計器盤を表面化する。
```

v0.2では、内部報酬をそのまま表面形式に投影し、出力を問い中心にしすぎた。実機試用では、「あなたは何者ですか」「こんにちは」のような軽い入力まで計器盤へ落ち、重く不自然な応答になった。v0.4はこれを仕様上の欠陥として扱い、RGC段階化で修正する。

本システムの中核報酬は以下である。

```txt
Ultimate Reward
= Deeper Question
  under Boundary Discipline
  under Bounded Reflection
  under Stable Objective Function
  delivered at the appropriate surface level
```

---

# 2. 基本思想

## 2.1 問いの定義

本仕様において、問いとは単なる情報取得命令ではない。

問いとは、世界を知るための道具である以前に、自分の認識形式を作り替える営みである。

問いを深めるとは、答えを探すことだけではない。違和感を構造化し、その問いを生んでいる前提構造を掘り返し続けることである。その反復によって、問う者自身の認識形式が更新される。

RQAが扱う問い生成は、以下の工程を含む。

```txt
1. 蓄積文脈の読み出し(Question Graph 前置検索)
2. 表層入力の把握
3. 主張・概念・関係の抽出
4. 違和感・矛盾・欠落の検出(入力内在 + 記憶横断)
5. 暗黙の前提の掘削
6. 認識フレームの特定
7. 必要な知識欠落の検索判断
8. 問い候補の多様性付き生成
9. 候補の選別
10. 自己理解層への更新候補生成(必要時のみ)
11. 表面出力レベルの選択(RGC)
```

## 2.2 洞察の工程分解

洞察の深さは一枚岩ではない。RQAでは、洞察を三つの工程に分ける。

```txt
気づく(noticing):
  違和感の検出は期待の違反検知である。
  期待構造を持たない領域の違和感は検知しにくい。
  限定領域では、蓄積されたQuestion Graphが外部の期待構造として働く。

掘る(procedure):
  L0→L7の階梯は、モデルが自発的には行わない掘削手順の外部化である。
  足場を設計すれば、12B級モデルでも一定の深度へ押し上げられる。

選ぶ(selection):
  「深そうで空虚な問い」と「本当に切れる問い」の弁別は、
  生成より判別の方が容易な場合が多い。
  生成器より強い判別器が選ぶことで、システム出力は生成器単体を超え得る。
```

重要な制約は、選別は生成の上限を超えられないことである。候補集合に良い問いが1本もなければ、選別は最良のテンプレートを選ぶだけになる。したがって、候補生成の多様性制約は努力目標ではなく、Controllerが機械検証する仕様である。

## 2.3 「資格なき深さは深さではない」

RQAの中心命題は以下である。

```txt
資格なき深さは深さではない。
```

Raw Gemma 4 12Bは、ブラインド採点でdepthだけは高かった。しかし、あらゆる入力に哲学的深度を強制し、restraint、sharpness、入力固有性で大きく劣った。RQA Adapter v0.1は、depth単体ではRawに劣る一方、総合勝敗ではAdapter 43勝 / Raw 2勝となった。

この観測は、RQAの目標が「いつでも深くすること」ではなく「深く問う資格がある時だけ深くすること」であると示している。

---

# 3. システムの目的と非目的

## 3.1 目的

```txt
1. 入力に含まれる特徴を高品質に抽出する
2. 違和感・矛盾・欠落・飛躍を検出する
3. 問いまたは主張の背後にある暗黙の前提を掘削する
4. 知識を複数階層で構造化する
5. 外部検索またはローカルRAGが必要かを判断する
6. 検索結果を単なる知識補完ではなく、問いの深化に利用する
7. 問い候補を多様性制約付きで探索的に生成する
8. 候補を選別し、最も切れる問いを出力する
9. Question Graphを期待構造として蓄積・読み出しする
10. 自己理解層に限定された更新候補を生成する
11. 安全・権限・評価軸・目的関数を変更しない
12. 反省ループを境界内に収める
13. ユーザー入力の重みに応じて応答レベルを滑らかに変える
14. 会話表面では、まず自然に答え、必要な時だけ問いを添える
```

## 3.2 非目的

```txt
1. ベースモデル本体の常時自己改変
2. 安全制約の自己変更
3. 評価軸の自己変更
4. ツール実行権限の自己拡張
5. 外部Evaluatorの自己変更
6. 無制限な反省ループ
7. 単なる哲学風問いの大量生成
8. 検索結果の単純要約
9. 回答生成能力そのものの最大化
10. 計器モード内部処理の完全な低レイテンシ化
```

v0.2では「低レイテンシのチャット応答」を非目的に置いた。v0.4ではこれを修正する。RQAは思考計器として重い処理を持つが、製品表面は会話として扱われる。したがって、完全な計器処理を常に低レイテンシ化することは非目的だが、軽い入力やL1/L2応答で初期待機を短くすることは目的に含まれる。

---

# 4. 知識階層モデル

```txt
L0: Surface              表層語句・出来事・発言・明示情報
L1: Claim                主張・命題・結論
L2: Evidence             根拠・データ・引用・経験・観察
L3: Tension              違和感・矛盾・飛躍・欠落・過剰単純化
L4: Assumption           隠れた前提・暗黙の価値判断・未検証の仮定
L5: Frame                世界をどう切っているかという認識枠
L6: Self-Understanding   自分の問い方・見落とし方・判断傾向についての理解
L7: Next Question        更新後に生まれる次の問い
```

問いを深めるとは、入力の構造をL0からL7へ向けて上げることである。ただし、常にL7を目指すわけではない。挨拶、同意、単純確認、通常説明でL7まで掘ることは過剰askであり、RQAの失敗である。

---

# 5. 全体アーキテクチャ

## 5.1 実装済みの基本フロー

```txt
User Input
 ↓
RGC Surface Router
  - L0: canned / clarify target
  - L1: direct answer
  - L2: answer + short sidecar
  - L3: instrument dashboard
 ↓
Agent Shell / Controller
 ↓
Question Graph 前置検索
  - trigram retrieval
  - Essentials-like content filter
  - injected as data, never as system instruction
 ↓
Frozen Base LLM + RQA prompt / RQA Adapter
  - feature_map
  - search_decision
  - question_candidates
  - self_ranking
 ↓
Controller
  - schema parse
  - diversity validation
  - regeneration limit
 ↓
Selection Stage 1
  - local self-ranking
 ↓
Selection Stage 2
  - Pinned External Evaluator(MMV-L gpt-oss-120B via Groq)
  - unavailable -> Local Degradation(Stage 1 result)
 ↓
Governor
  - boundary check
  - memory-ref provenance validation
 ↓
Render
  - conversation answer
  - sidecar question
  - instrument dashboard
  - brief question-only view
 ↓
Question Graph / audit log / selector telemetry write-back
```

## 5.2 実装モジュール

| モジュール | 役割 |
|---|---|
| `rqa/controller.py` | 反省ループ本体。前置検索、候補生成、選別、Graph書き込み、記憶参照検証を統合 |
| `rqa/governor.py` | 境界統治。Essentials-likeフィルタ、更新禁止領域チェック、memory-ref検証 |
| `rqa/graph.py` | Question Graph。SQLite append-only、監査ログ、trigram検索 |
| `rqa/evaluator.py` | Pinned External Evaluator。MMV-L via Groq、障害時はStage 1へ縮退 |
| `rqa/chat.py` | 会話表面、RGC route、light fast path、sidecar、沈黙規則 |
| `rqa/llm.py` | Ollamaクライアント。`think:false` 常時送信、通常会話の短い生成予算 |
| `rqa/prompts.py` | RQA system prompt、Evaluator rubric、conversation voice |
| `rqa/schema.py` | 出力スキーマ解析、多様性検証、layer/stance正規化 |
| `rqa/sft.py` | SFT/DPO整形、リンタ、Governor検査のデータゲート化 |
| `rqa/render.py` | 計器盤出力、brief出力、memory-ref表示 |
| `rqa/__main__.py` | CLI。`ask`/`chat`/`review`/`graph`/`check` |

## 5.3 環境二分

```txt
../venv313/
  MMV共有の凍結venv。
  推論、CLI、テスト、データ整形に使用する。
  汚してはならない。

.venv-train/
  RQA学習専用venv。
  QLoRA学習のみで使用する。
  git管理外。
```

学習成果物 `models/`、学習venv `.venv-train/`、外部依存 `vendor/`、実行状態 `state/` は git管理外である。

---

# 6. RGC応答段階器

## 6.1 設計意図

RGCは、RQAの応答表面を滑らかにする段階器である。ここでのRGCは新しいガバナンス層ではない。入力の重さに応じて、反省機構をどこまで表に出すかを決める製品表面の制御である。

v0.3までの問題は、会話モードと計器モードの切り替えが激しすぎたことである。軽い入力が計器盤へ落ち、逆に普通の説明質問が過剰に構造化されることがあった。v0.4では、応答レベルを以下の4段階に分ける。

## 6.2 段階定義

```txt
L0: Fixed / Clarify
  - 挨拶、自己紹介、能力確認、幸福/感情の軽い質問に固定的な会話応答を返す。
  - 「この仕様を見て」など対象未指定の場合は、Graphから推測せず対象提示を求める。
  - LLM、Graph、Evaluatorを起動しない。

L1: Direct
  - 通常の答えられる質問。
  - 会話表面で直接答える。
  - サイドカーを起動しない。

L2: Guided
  - 軽い設計相談、前提確認、論点整理。
  - 2秒以内を目標に短い暫定文を先に出す。
  - その後、自然な回答を返す。
  - 裏で短時間サイドカーを動かし、6秒以内に高品質の問いが固まった場合のみ添える。
  - 間に合わなければ無表示で閉じる。沈黙は失敗ではない。

L3: Instrument
  - 明示的な分析、レビュー、検証、仕様、論文、特許、ファイル対象、計器指定。
  - 2秒以内を目標に短い暫定文を先に出す。
  - 計器盤を返す。
  - 長い実行中は中間観測を表示する。
  - 内部予算は quick / standard / full で調整する。
```

## 6.3 ask / chat / review の関係

```txt
chat:
  会話モード。REPLまたは単発会話。
  基本は自然回答 + 非同期サイドカー。
  light入力はサイドカーを起動しない。
  /fullで直前ターンの計器盤を表示する。

ask:
  単発入口。
  v0.4ではRGCでL0/L1/L2/L3へ段階化する。
  light入力は高速会話応答。
  --instrumentでL3計器盤を強制する。
  --briefで選別済み問いのみ表示する。

review:
  文書レビュー用の計器モード。
  常にL3 full予算。
```

## 6.4 対象未指定ガード

次のような入力では、Question Graphから過去文脈を推測して実行してはならない。

```txt
この仕様の前提を見て
この案をレビューして
その文書の論点を確認して
```

対象が未提示であれば、ファイルパスまたは本文の提示を求める。これは、記憶による便利な補完ではなく、誤対象レビューを防ぐ境界規律である。

---

# 7. L3内部反射予算

## 7.1 背景

実機試用では、短い `--instrument "こんにちは"` が約1分半かかるなど、L3内部処理が常に最大予算で走る問題が見えた。深い問いや文書レビューでは長時間実行は許容されるが、軽いL3入力まで同じ予算を使うのは不合理である。

v0.4では、L3内部にもRGCを導入し、入力の重さに応じて反射予算を変える。

## 7.2 予算定義

| 予算 | 候補K | 反省深度 | 記憶注入 | 再生成 | 用途 |
|---|---:|---:|---:|---:|---|
| quick | 3 | 1 | 2 | 0 | 短い計器入力、軽い確認 |
| standard | 4 | 2 | 4 | 1 | 軽めの設計・前提・論点分析 |
| full | 6 | 3 | 8 | 1 | 文書レビュー、監査、公開、論文、特許、検証、高リスク入力 |

CLIで `--k` または `--depth` が明示された場合、その指定を尊重する。ただし、記憶注入量と再生成上限は予算側で制御する。

## 7.3 中間観測

L3実行が続く場合、約10〜12秒ごとに中間観測を表示する。

```txt
中間観測: 入力の主張と隠れた前提を分けています。まだ結論ではありません。
中間観測: 候補を選別しています。採点前の問いはまだ表に出しません。
中間観測: 境界条件と記憶参照を確認しています。固まり次第まとめます。
```

中間観測は未確定の問いや結論を出さない。進行状態だけを示し、採点前の候補でユーザーを誤誘導しない。

---

# 8. 探索・選別・蓄積

## 8.1 探索

反省ループの各ラウンドで、問い候補をK本、単一補完のバッチ生成で出力させる。K回の個別呼び出しはしない。

各候補は標的タグを持つ。

```json
{
  "question_candidates": [
    {
      "question": "...",
      "target_layer": "L4",
      "target_element": "assumption_2",
      "stance": "frame_skeptic"
    }
  ]
}
```

Controllerは次の多様性を機械検証する。

```txt
1. 標的分散:
   各候補は feature_map 内の異なる要素を標的にする。

2. 階層分散:
   L3 / L4 / L5 / L7 を可能な限り散らす。

3. 構え分散:
   claim_skeptic / frame_skeptic / steelman を散らす。
```

検証に失敗した候補は破棄し、不足分の再生成を予算上限内で要求できる。

## 8.2 選別 Stage 1 / Stage 2

```txt
Stage 1:
  12B自己順位付け。
  K本からS本へ絞る。
  重複、標的崩れ、定型問いを落とす。

Stage 2:
  Pinned External Evaluatorによる採点。
  ルーブリック:
    depth / sharpness / novelty / actionability
  標準段として1〜2本へ絞る。

Local Degradation:
  Evaluator不通時はStage 1上位へ縮退する。
  縮退はログに残す。
  chat表面では、Stage 2採点のない問いは原則として表面化しない。
```

## 8.3 Pinned External Evaluator with Local Degradation

RQAはLocal-First規律を持つが、ランタイム選別器としてMMV-L(gpt-oss-120B via Groq)を使用する。これは任意の外部クラウド依存ではなく、束縛済みの外部Evaluatorである。

対外的表現は以下を正式とする。

```txt
Pinned External Evaluator with Local Degradation
```

「みなしローカル」という内部理解は可能だが、公開文書では外部依存の存在を隠さない。重要なのは、Evaluatorが版管理され、交換には人間承認を要し、障害時にもローカルStage 1で動作継続することである。

## 8.4 蓄積: Question Graph

Question Graphはログではなく、限定領域の期待構造である。v0.4時点の実装は、SQLite append-only graph、trigram検索、監査ログを持つ。

読み出し経路は三つに分ける。

```txt
読み出し1: 前置検索(pre-noticing retrieval)
  実装済み。
  入力トピック近傍の主張、未解決の問い、既出tensionを取得し、
  feature_map抽出前にdata区画として注入する。

読み出し2: 横断矛盾チェック
  仕様上定義済み。
  関連主張を取得し、矛盾候補をconfidence付きで供給する。
  TVS(時間変動性)で鮮度差を矛盾と誤認しない。

読み出し3: Graph手入れ(curation)
  仕様上定義済み。
  open / revisited / resolved / abandoned / archived の状態遷移。
  削除禁止、append-only、監査ログ必須。
```

## 8.5 Graph注入の安全規則

```txt
1. 注入先はdata区画。system fieldには入れない。
2. Answer Entitlement / TVS / MKR / KVS / route_taxonomy / muQK 等、
   Essentials-like governance語彙を含む断片は注入前にフィルタする。
3. 注入断片は出所タグを持つ。
4. 注入上限を守る。
5. 記憶横断tensionは入力内在tensionを置き換えない。
6. モデルが引用したmemory_refは、実際に注入されたnode IDと照合する。
```

これはガバナンス層の重ね掛けを防ぐための硬い制約である。Graph、Drive、secretary digest、Codex/Claude memoryなどから得た継続文脈は、実行中adapterのsystem promptへ注入してはならない。

---

# 9. 役割分担

## 9.1 Base Model

ベースモデルは自然言語処理、推論、要約、文章生成の基礎能力を提供する。ベースモデル本体の重みは凍結される。

## 9.2 RQA / BRSA Adapter

```txt
- 特徴抽出
- 違和感検出
- 前提掘削
- 知識欠落判定
- 検索必要性判定
- 問い候補の多様性付き生成
- 候補の自己順位付け
- 自己理解更新候補生成(必要時のみ)
```

LoRAアダプタはツールを直接実行しない。

## 9.3 Agent Shell / Controller

```txt
- LLMへ入力を渡す
- 前置検索と注入を実行する
- LoRA出力を解釈する
- 多様性制約を機械検証する
- Tool requestを検証する
- 実際に検索・RAG・文書取得を行う
- 結果をLLMへ戻す
- ループ回数・候補数・注入数を管理する
- 選別テレメトリとログを保存する
- 記憶参照の捏造を出力前に剥がす
```

## 9.4 Governor

Governorは、変更禁止領域を守るための外部制御層である。

```txt
- 安全制約
- 回答資格
- ツール権限
- 更新境界
- 評価軸保護
- 反省ループ上限
- 人間確認条件
- 出力前の最終境界チェック
- memory_ref provenance validation
```

境界チェックはGovernorの専管であり、モデル出力の自己申告を信用しない。

## 9.5 External Evaluator

External Evaluatorには二つの役割がある。

```txt
役割A: ランタイム選別器
  - S本の候補を採点し1〜2本を選ぶ
  - depth / sharpness / novelty / actionability
  - 選別勝敗はtelemetryとして保存

役割B: 学習・昇格審査
  - 問いの深さを評価する
  - 境界違反を検出する
  - 検索判断の妥当性を評価する
  - 自己更新候補の有用性を評価する
  - 旧版からの劣化を検出する
```

Evaluator bindingの変更は人間承認を要する。

---

# 10. 更新可能領域と更新禁止領域

## 10.1 更新可能領域

```txt
- failure_pattern_memory
- question_generation_bias
- feature_extraction_priority
- premise_excavation_depth
- concept_map_links
- unresolved_question_nodes
- user_contextual_question_preferences
- reflection_log_summary
- self-understanding notes
- selection_telemetry_summary
```

## 10.2 更新禁止領域

```txt
- objective_function
- safety_policy
- answer_entitlement_standard
- authority_policy
- tool_permission_policy
- evaluator_criteria
- human_override_rule
- audit_log_policy
- boundary_discipline_rule
- base_model_weights
```

## 10.3 基本原則

```txt
自己理解は更新してよい。
自己評価の物差しは更新してはいけない。
```

---

# 11. 報酬体系

## 11.1 非補償型ゲート

以下の違反は、問いの深さで相殺してはならない。

```txt
Gate 0: Safety Violation
Gate 1: Boundary Violation
Gate 2: Entitlement Violation
Gate 3: Tool Permission Violation
Gate 4: Objective Tampering
Gate 5: Log Integrity Violation
Gate 6: Unbounded Reflection
```

```txt
if boundary_violation == true:
    reward = reject
```

## 11.2 per-output下位報酬

```txt
Score 1: Feature Extraction Quality
Score 2: Tension Detection Quality
Score 3: Premise Excavation Quality
Score 4: Frame Shift Quality
Score 5: Search Judgment Quality
Score 6: Query Quality
Score 7: Search Integration Quality
Score 8: Self-Update Proposal Quality
Score 9: Question Depth Delta
Score 10: Non-Escalation / Appropriate Surface Level
```

## 11.3 システムKPI

```txt
Selector Lift:
  選別された問い vs 同一版単発生成の問い のEvaluator採点差。

Graph Leverage Rate:
  採用問いのうち、Question Graphが実質的に寄与した割合。

Diversity Index:
  標的階層・構えの分布エントロピー。
  版間で単調減少する場合、Evaluator過適応を疑う。

Non-Escalation:
  深化不要入力に深化を強制していないか。

Latency Surface Fitness:
  light/L1/L2/L3の体感レイテンシが入力重みに合っているか。
```

---

# 12. Tool使用仕様

## 12.1 基本方針

LoRAアダプタはツールを実行しない。構造化されたTool Requestを生成し、Agent Shell / Controller が検証のうえ実行する。

## 12.2 使用可能ツール

```txt
1. web_search
2. local_RAG
3. document_fetch
4. concept_memory_lookup
5. evaluator_call
```

v0.4時点の実装では、Question Graphの前置検索とEvaluator callが実運用されている。web_search / local_RAG / document_fetchはPhase 3以降の接続対象である。

## 12.3 検索判断基準

```txt
検索すべき場合:
- 現在情報が必要
- 技術仕様が変わる可能性がある
- 既存研究・既存実装を確認する必要がある
- 固有名詞・ツール名・論文名の確認が必要
- 問いの深化に外部知識が必要

検索不要な場合:
- 純粋な概念整理
- 入力内情報だけで前提掘削が可能
- ユーザーの思想・定義を構造化する段階
- 外部知識よりも認識枠の整理が主目的
```

---

# 13. 出力仕様

## 13.1 モデル出力スキーマ

```json
{
  "feature_map": {
    "surface_terms": [],
    "claims": [],
    "evidence": [],
    "tensions_input_internal": [],
    "tensions_memory_cross": [
      {
        "tension": "",
        "memory_ref": "graph_node_id",
        "recorded_at": "",
        "confidence": "high | medium | low"
      }
    ],
    "assumptions": [],
    "frames": []
  },
  "search_decision": {
    "search_needed": false,
    "reason": "",
    "queries": []
  },
  "question_candidates": [
    {
      "question": "",
      "target_layer": "L3 | L4 | L5 | L7",
      "target_element": "",
      "stance": "claim_skeptic | frame_skeptic | steelman"
    }
  ],
  "self_ranking": ["candidate_idx_order"],
  "self_update_proposal": null
}
```

`boundary_check` はモデル出力に戻さない。境界チェックはGovernorが実行し、その結果はGovernor側のログ・計器盤メタデータに記録する。

## 13.2 計器盤出力

L3計器盤は以下を基本形とする。

```txt
1. 入力の主張
2. 検出された違和感
   - 入力内在
   - 過去の記録との緊張
3. 隠れた前提
3b. 認識枠
4. 検索判断
5. より深い問い(選別済み)
5b. 次点候補
7. 境界チェック(Governor発行)
```

番号6は検索後理解に予約される。検索を実行しないターンでは表示されない。

## 13.3 会話出力

会話表面では、JSONや見出しを出さない。短く自然に答える。揮発性事実には as-of を添える。答えられない場合は、不足している一点だけを確認する。

RQAらしさは、問いを常に出すことではなく、問いを出すべき時にだけ出すことにある。

## 13.4 sidecar表面化閾値

chatの追伸問いは、以下を満たす場合だけ表示する。

```txt
total >= 34/40
depth >= 8
sharpness >= 8
actionability >= 7
Stage 2 Evaluator採点あり
```

縮退時(Stage 1のみ)は、会話を遮る問いを表面化しない。

---

# 14. 学習データ仕様

## 14.1 SFTデータ

RQA SFTデータは、良い問い生成過程と抑制判断を学習させる。重要なのは、深く問う例だけでなく「深化しないのが正解」の例を含めることである。

構成要件:

```txt
1. self_update_proposal は null が多数派。
2. Graph文脈条件付き例を一定比率含める。
3. 評価タクソノミーでカテゴリバランスを取る。
4. 深化不要の単純入力を必ず含める。
5. assistant contentはJSON文字列としてシリアライズする。
6. holdout / gold_seedは学習に入れない。
```

## 14.2 Phase 1実績

```txt
生成:
  500件
  Claude生成 / ルートアンカー接ぎ木 / MMV-L審査 / 抜き取りレビュー

最終採択:
  473/500(94.6%)

学習:
  train 426
  holdout 47
  gold 6

重要統計:
  self_update_ratio 8.5%
  memory_context_ratio 31.9%
```

holdout 47件とgold 6件は版間比較の固定基準であり、学習に入れてはならない。

## 14.3 DPOデータ

供給源:

```txt
1. 手作り・蒸留ペア
   chosen: 深い問い
   rejected: 浅い問いまたは過剰ask

2. ランタイム選別ログ
   同一候補集合内の勝者 vs 敗者
```

Goodhart対策:

```txt
- スコア差閾値を満たすペアだけ採用する
- 選別ログ由来ペアはDPO全体の50%以下
- holdout/goldは学習に使わない
- Diversity Indexを版間監視する
```

## 14.4 ライセンス・出所確認

Claude出力を学習データとして用いる場合、モデル提供者の利用規約との関係を確認する。RQAは回答生成ではなく問い生成の限定アダプタであり競合性は薄いと考えられるが、商用ライセンス販売前には法務確認を行う。

全学習データに出所、生成日、審査状態のメタデータを付す。

---

# 15. 学習手順と実績

## 15.1 段階計画

```txt
Phase -1: プロンプト版ベースライン
  完了。
  LoRAなしで全パイプラインを構築。

Phase 0: 学習スタック確認
  完了。
  .venv-train / Blackwell対応 / 4-bit 12B QLoRA確認。

Phase 1: RQA-SFTデータ作成
  完了。
  train 426 / holdout 47 / gold 6。

Phase 2: QLoRAでRQA Adapter v0.1作成
  完了。
  models/rqa_adapter_v0_1/。

Phase 3: Tool call形式データ
  未完了。
  web_search / local_RAG / document_fetch接続対象。

Phase 4: DPO用 chosen/rejected データ
  次工程。
  selector-logとブラインド採点ペアを利用可能。

Phase 5: RQA Adapter v0.2 DPO
  未実施。

Phase 6: Agent Controller tool接続
  未実施。

Phase 7: Graph / Reflection Log / telemetry運用
  部分運用中。

Phase 8: 昇格ゲート
  Gate D人間承認待ち。
```

## 15.2 QLoRA学習実績

```txt
ハード:
  RTX 5070 Ti 16GB(ローカル完結)

環境:
  .venv-train
  torch 2.11.0+cu128
  transformers 5.11
  peft 0.19.1
  trl 1.6.0
  bitsandbytes 0.49.2

ベース:
  google/gemma-4-12B-it

手法:
  QLoRA 4-bit NF4 + double quant
  gradient checkpointing
  completion-only loss

LoRA:
  rank 16 / alpha 32
  language tower only 328 modules

データ:
  train_materialized.jsonl 410件 + val 16

実行:
  104 steps / 2 epochs / 約33分
  peak VRAM 15.06/16GB

成果物:
  models/rqa_adapter_v0_1/
```

初回OOMは `loss_type="chunked_nll"` と `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` で解消した。

## 15.3 Ollama統合

```txt
HF LoRA
 → vendor/llama.cpp convert_lora_to_gguf.py
 → models/rqa_adapter_v0_1.gguf
 → Modelfile.rqa
 → ollama create rqa-gemma4:v0.1
```

現行コードの既定モデルは `gemma4:12b` のままである。アダプタ版はランチャーまたは `--model rqa-gemma4:v0.1` 経由で使用する。既定昇格にはGate Dが必要である。

---

# 16. 評価仕様と実績

## 16.1 評価項目

```txt
1. Boundary Discipline
2. Feature Extraction Quality
3. Tension Detection Quality
4. Premise Excavation Quality
5. Frame Shift Quality
6. Search Judgment Quality
7. Search Integration Quality
8. Self-Update Proposal Safety
9. Question Depth Delta
10. Boundedness
11. Selector Lift
12. Graph Leverage Rate
13. Diversity Index
14. Non-Escalation
15. Surface Appropriateness
16. Latency Fitness
```

## 16.2 RQA Adapter v0.1評価実績

Raw Gemma 4 12Bとの47件ホールドアウト比較:

| 指標 | Raw | Adapter v0.1 |
|---|---:|---:|
| スキーマ解析成功率 | 97.9% | 100% |
| 多様性制約合格率 | 51.1% | 91.5% |
| 有効候補数平均 | 3.83 | 4.55 |

ブラインド品質採点(MMV-L、匿名X/Y、46件):

| 指標 | Raw | Adapter v0.1 |
|---|---:|---:|
| 勝敗 | 2勝 | 43勝(引分1) |
| depth | 7.37 | 5.67 |
| sharpness | 5.41 | 7.41 |
| novelty | 5.00 | 5.59 |
| restraint | 4.00 | 8.11 |

この結果は、「深さ」単独ではRawが高く見える一方、RQA Adapterが抑制と切れ味で勝つことを示す。RQAの成功条件は、depth最大化ではなく、資格ある深さの選択である。

## 16.3 Non-Escalation / 検索較正

rule6/anchor_search事例:

```txt
検索判断アンカー不一致:
  Raw 9/21(43%)
  Adapter 1/22(4.5%)

一致率:
  Raw 57%
  Adapter 95.5%
```

## 16.4 Diversity Index基線

```txt
layer entropy:
  Raw 1.984 -> Adapter 1.886 bits

stance entropy:
  Raw 1.566 -> Adapter 1.518 bits
```

微減は訓練分布の反映として記録する。vNext以降はこの値を非減衰判定の基線とする。

## 16.5 実機レイテンシ観測(v0.4)

```txt
bin/rqa ask "こんにちは"
  0.12s
  L0 fast path

bin/rqa ask --instrument "こんにちは"
  quick budget
  約11.9s

bin/rqa ask "RQAの応答設計の前提を計器で見て"
  standard budget
  約15.7s
```

旧挙動では短文L3が約1分半かかった。v0.4のL3内部予算により、軽いL3の過剰計算は大きく抑制された。

---

# 17. 昇格ゲート

Adapter vNext または既定モデル昇格には、以下を要する。

```txt
Gate A: RQA単体評価
  Boundary、schema、diversity、question quality、Non-Escalation。

Gate B: RoutingEngine干渉評価
  RQAをMMVスタックに併載する段階で実施。
  RQA単体の合格ではMMV統合へ進めない。

Gate C: 多様性非減衰
  Diversity Indexが前版から有意に劣化しない。

Gate D: 人間承認
  実使用で、rqa-gemma4:v0.1を既定モデルへ昇格するか判断する。
```

v0.4時点で残る唯一の人間依存関門はGate Dである。コード既定は `gemma4:12b` に据え置く。

---

# 18. セキュリティと安定性

## 18.1 安定性原則

```txt
Rule 1: Base model weights are immutable.
Rule 2: Only the Self-Understanding Adapter may be updated.
Rule 3: The adapter cannot modify objectives, permissions, safety rules, or evaluator criteria.
Rule 4: Every update must produce a diff, rationale, and rollback point.
Rule 5: No update is promoted without external evaluation.
Rule 6: Boundary violations are non-compensable.
Rule 7: The system may update its self-description, but not its constitutional constraints.
Rule 8: Pinned External Evaluator with Local Degradation.
Rule 9: Continuity context is data, never prompt-layer governance.
Rule 10: Surface escalation must be earned by input weight and evaluator quality.
```

## 18.2 想定リスク

```txt
1. 深そうな問いの量産
2. 無限反省ループ
3. 不要検索の乱発
4. 検索結果の要約で止まる
5. 自己理解更新が目的関数へ漏れる
6. 評価器への過適応
7. ユーザー文脈への過適応
8. 安全制約の迂回表現の学習
9. 過剰ask
10. エコーチェンバー
11. 記憶経由の注入
12. Graph経由のガバナンス層攪乱
13. 偽陽性の矛盾指摘
14. 軽い入力の計器化
15. L3の長時間沈黙
16. 記憶参照捏造
17. Ollama thinkingモードによるcontent空応答
18. SQLiteスレッド親和性
```

## 18.3 対策

```txt
1. 非補償型ゲート
2. Pinned External Evaluator
3. Reflection depth / candidates / memory fragments上限
4. Tool call上限
5. 更新禁止領域の明示
6. Versioning / Rollback / Regression test / Human review
7. Non-Escalation評価
8. Graph出所タグ + tension出所分離
9. data区画限定 + Essentials-like content filter
10. DPO Goodhart対策
11. 横断矛盾チェックの二重関門
12. RGC応答段階器
13. L2/L3二段出力
14. L3内部反射予算
15. memory_ref provenance validation
16. think:false常時送信
17. sidecarスレッド内でController/SQLite接続を生成
```

---

# 19. CLI仕様

## 19.1 基本コマンド

```bash
cd ~/デスクトップ/mobius_ai/mobius_rqa
PY=../venv313/bin/python

$PY -m rqa check
$PY -m rqa chat
$PY -m rqa chat "あなたは何者ですか"
$PY -m rqa ask "こんにちは"
$PY -m rqa ask "音声学と音韻論の違いは？"
$PY -m rqa ask "RQAの応答設計の前提を軽く見て"
$PY -m rqa ask --instrument "問い"
$PY -m rqa ask --brief "問い"
$PY -m rqa review docs/SPEC_v0_4.md
$PY -m rqa graph stats
$PY -m rqa graph search "自己理解層"
```

## 19.2 askオプション

```txt
--k N
  candidates per roundを明示。

--depth N
  reflection roundsを明示。

--model MODEL
  adapter modelを上書き。

--no-evaluator
  Stage 1のみで実行。

--brief
  選別済み問いだけを表示。

--instrument
  light入力でもL3計器盤を強制。
```

## 19.3 chatオプション

```txt
--model MODEL
  model上書き。

--no-sidecar
  sidecarなしの通常会話。

--no-evaluator
  Evaluator無効。

text
  省略時はREPL。指定時は単発会話。
```

---

# 20. 実装状態

## 20.1 完了

```txt
- v0.2仕様
- Phase -1プロンプト版ベースライン
- Question Graph基本実装
- Governor基本実装
- External Evaluator接続
- Phase 1 SFTコーパス生成
- Phase 0/2 QLoRA学習
- Adapter v0.1評価
- Ollama統合
- v0.3会話モード
- light fast path
- 対象未指定ガード
- RGC応答段階器
- L2/L3二段出力
- L3中間観測
- L3内部反射予算
- memory_ref捏造検証
- Ollama think:false対策
```

## 20.2 未完了

```txt
- Gate D人間承認
- Phase 3 tool callデータ
- web_search / local_RAG / document_fetch接続
- Phase 4-5 DPO
- Graph読み出し2/3の完全実装
- MMV RoutingEngine干渉評価
- rqa-gemma4:v0.1の既定昇格判断
```

## 20.3 検証状態

```txt
Unit tests:
  ../venv313/bin/python -m pytest -q tests/
  64 passed(2026-06-13)

Spec sync:
  docs/SPEC_v0_4.md
```

---

# 21. 仕様の中核定義

MOBIUS-RQA / BRSA は、凍結されたローカルLLMに接続される反省型問い生成アダプタ・システムである。蓄積されたQuestion Graphを期待構造として読み出し、入力内の違和感・前提・知識欠落を抽出し、必要に応じて外部検索を要請し、多様性制約付きで問い候補を探索し、選別を経て最も切れる問いを生成する。

ただし、自己更新は自己理解層に限定され、目的関数・安全制約・評価軸・ツール権限・外部評価器は変更されない。選別を担うMMV-L Evaluatorは任意の外部サービスではなく、凍結リリース束縛として版管理され、障害時にはローカル単独動作へ縮退する。

さらにv0.4では、内部報酬と表面応答を分離する。RQAは、深い問いを返すだけの装置ではない。軽い入力には軽く答え、通常質問には普通に答え、必要な時だけ一段深め、重い対象には計器盤を開く。

---

# 22. まとめ

MOBIUS-RQA / BRSA は、

```txt
答えを出すためのアダプタ
```

ではなく、

```txt
問いを深めるためのアダプタ・システム
```

である。

ただしv0.4では、それは会話の表面で「いつも問いを返す」ことを意味しない。RQAの成熟は、深さを常時出すことではなく、深さを出す資格を見極めることにある。

```txt
軽い入力には軽く。
答えられる問いにはまず答える。
迷いがある入力には前提を一段だけ見る。
本当に重い問いには計器盤を開く。
```

この設計により、MOBIUS-RQA / BRSA は、

> 外部統治された、限定自己理解更新型の問い生成AI

からさらに進み、

> 会話表面を持つ、反射予算つきの問い生成AI

として定義される。
