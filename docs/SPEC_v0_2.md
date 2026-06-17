# MOBIUS-RQA / BRSA 仕様書ドラフト v0.2

## 0. 文書情報

**名称**:MOBIUS-RQA / BRSA
**正式候補名**:MOBIUS Reflective Questioning Adapter / Bounded Reflective Self-Understanding Adapter
**略称**:RQA / BRSA
**版**:v0.2(2026-06-12)
**前版**:v0.1(対話ドラフト)
**対象**:ローカルLLMに接続されるLoRAアダプタおよび外部エージェント制御系
**想定ベースモデル**:Gemma 4 12B(主)、Qwen系、Phi系、その他商用利用可能なローカルLLM
**目的**:回答生成ではなく、問いの深化・前提掘削・認識形式の更新支援を行うこと

### v0.1 → v0.2 主要変更点

```txt
1. 生産工程の転換:
   単発生成 → 探索(候補K本)+選別(二段)+蓄積(読み出し経路付きQuestion Graph)

2. ランタイム選別の標準化 — Pinned External Evaluator with Local Degradation:
   External Evaluator(MMV-L gpt-oss-120B via Groq)は任意のクラウド
   サービスではなく、凍結リリース束縛(pinned)された構成要素である。
   オーナー裁定(2026-06-12)によりLocal-First制約と両立すると確定し、
   選別Stage 2はパイプラインの標準段となった。
   障害時はローカル縮退(Stage 1単独継続)する。

3. boundary_check のモデル出力スキーマからの削除:
   境界チェックは Governor(外部コード)の専管とする。
   モデルに準拠自己申告を学習させない。

4. self_update_proposal の希薄化原則:
   「提案なし」が多数派となるデータ設計を義務化。

5. Question Graph 読み出し経路の新設(前置検索 / 横断矛盾チェック / 手入れ)。
   注入時の Essentialsライク内容フィルタと出所タグを必須化。

6. Goodhart対策:
   選別ログ由来DPOの上限・スコア差閾値・学習不可ホールドアウト集合・
   多様性指標の版間監視を新設。

7. 昇格ゲートに「RoutingEngine干渉評価(Condition I相当の層化評価)」を追加。

8. 学習手順に Phase −1(プロンプト版ベースライン)を追加。
   学習HW前提をローカル RTX 5070 Ti / 16GB VRAM に更新(QLoRAローカル完結)。

9. 評価指標に Selector Lift / Graph Leverage Rate / Diversity Index を追加。
   per-output報酬とシステムKPIを分離。
```

---

# 1. 概要

本仕様書は、ローカルLLMに接続される反省型問い生成アダプタ **MOBIUS-RQA / BRSA** の設計を定義する。

本システムは、ユーザー入力に対して即時に回答を生成することを主目的としない。
入力に含まれる違和感、欠落、矛盾、暗黙の前提、認識枠を抽出し、それらをもとに **より深い問い** を生成することを目的とする。

v0.2における中核的な設計判断は、問いの生産工程を **単発生成** ではなく
**探索+選別+蓄積** のシステムとして組むことである。

```txt
探索: 多様性制約付きで問い候補をK本生成する
選別: 二段選別(ローカル自己順位付け → 外部Evaluator採点)で1〜2本に絞る
蓄積: Question Graphを書くだけでなく読む。蓄積された主張・未解決の問いを
      「期待構造」として気づき(tension検出)の段階に供給する
```

この転換の根拠は、洞察の工程分解にある(2.2節)。
12B級モデルの単発生成は「型の深さ」で頭打ちになるが、
生成・判別・記憶照合を分業させたシステムは、限定領域において
生成器単体の上限を超えた「気づき」を産出できる。

本システムの究極報酬は、以下のように定義される。

> **境界条件を守りながら、より深い問いを立てること。**

```txt
Ultimate Reward
= Deeper Question under Bounded Reflection and Governance
```

---

# 2. 基本思想

## 2.1 問いの定義

本仕様において、問いとは、単なる情報取得命令ではない。

問いとは、

> 世界を知るための道具である以前に、
> 自分の認識形式を作り替える営みである。

問いを深めるとは、答えを探すことではない。
違和感を構造化し、その問いを生んでいる前提構造を掘り返し続けることである。
その反復によって、問う者自身の認識形式が更新される。

本システムにおいて問い生成とは、以下の工程を含む。

```txt
1. 蓄積文脈の読み出し(Question Graph 前置検索)
2. 表層入力の把握
3. 主張・概念・関係の抽出
4. 違和感・矛盾・欠落の検出(入力内在 + 記憶横断)
5. 暗黙の前提の掘削
6. 認識フレームの特定
7. 必要な知識欠落の検索
8. 問い候補の多様性付き生成(探索)
9. 候補の選別
10. 自己理解層への更新候補生成(必要時のみ)
```

## 2.2 洞察の工程分解 — なぜ探索+選別+蓄積か

洞察の深さは一枚岩ではなく、三つの工程に分かれる。

```txt
気づく(noticing):
  違和感の検出は「期待の違反」の検知である。
  期待構造を持たない領域の違和感は原理的に検知できない。
  → 規模(パラメータ数)に縛られる。LoRAでは治らない。
  → ただし限定領域では、蓄積されたQuestion Graphが
    外部の期待構造として代替できる(蓄積)。

掘る(procedure):
  L0→L7の階梯は、モデルが自発的にはやらない掘削手順の外部化である。
  → 設計で買える。足場はモデルをその帯域の中で押し上げる。

選ぶ(selection):
  「深そうで空虚な問い」と「本当に切れる問い」の弁別は、
  生成より判別のほうが容易なタスクである。
  → 外部化できる。生成器より強い判別器が選べば、
    システムとしての出力品質は生成器単体を超える(探索+選別)。
```

**制約事項**:選別は生成の上限を超えられない。K本の候補の中に良い問いが
1本も含まれなければ、選別は最良のテンプレートを選ぶだけである。
したがって探索の多様性制約(5.3節)は努力目標ではなく機械検証される仕様である。

---

# 3. システムの目的

```txt
1. 入力に含まれる特徴を高品質に抽出する
2. 違和感・矛盾・欠落・飛躍を検出する(入力内在 + 蓄積記憶との横断)
3. 問いまたは主張の背後にある暗黙の前提を掘削する
4. 知識を複数階層で構造化する
5. 外部検索またはローカルRAGが必要かを判断する
6. 検索結果を単なる知識補完ではなく、問いの深化に利用する
7. 問い候補を多様性制約付きで探索的に生成する
8. 候補を選別し、最も切れる問いを出力する
9. Question Graphを期待構造として蓄積・読み出しする
10. 自己理解層に限定された更新候補を生成する(必要時のみ)
11. 安全・権限・評価軸・目的関数を変更しない
12. 反省ループを境界内に収める
```

---

# 4. 非目的

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
10. 低レイテンシのチャット応答
```

項目10はv0.2で追加した。本システムは1ターン数十秒オーダーの
**思考計器(thinking instrument)** であり、即答チャットボットではない。
レイテンシをチャット水準へ削る目的で探索・選別を省略してはならない。

本システムが許可する自己更新は、あくまで

```txt
自己理解層の限定的・可逆的・外部評価付き更新
```

である。MOBIUS-RQA / BRSA は自己更新するAI本体ではない。

---

# 5. 全体アーキテクチャ

## 5.1 構成要素とフロー

```txt
User Input
 ↓
Agent Shell / Controller
 ↓
Frozen Governor
  - Safety Gate
  - Authority Gate
  - Answer Entitlement Gate
  - Tool Permission Gate
  - Boundary Discipline Gate
 ↓
[蓄積の読み出し] concept_memory_lookup(前置検索, 5.4節)
  - Question Graph 近傍の主張・未解決の問い・既出tensionを取得
  - Essentialsライク内容フィルタを通過したもののみ
  - data区画として注入(system fieldには入れない)
 ↓
Frozen Base LLM + RQA/BRSA LoRA Adapter
  - feature_map 抽出(入力内在 / 記憶横断 を区別)
  - search_decision
 ↓
Tool Decision → Agent Controller によるツール実行
  - no tool / web_search / local_RAG / document_fetch / concept_memory_lookup
 ↓
Search / RAG / Memory Result → Adapter による再解釈
 ↓
[探索] 問い候補 K本 を単一補完でバッチ生成(標的タグ付き, 5.3節)
 ↓
[選別 Stage 1] 12B 自己順位付け: K本 → S本(重複・定型の除去)
[選別 Stage 2] External Evaluator 採点: S本 → 1〜2本(標準段)
               ※障害時のみStage 1上位へ縮退(5.2節)
 ↓
Governor 最終境界チェック(コードによる検証。モデル自己申告ではない)
 ↓
Final Output
  - 深化された問い(選別済み)
  - 回答 / 保留 / 検索結果を反映した再問い
  - 自己理解更新候補(必要時のみ)
 ↓
[蓄積の書き込み] Question Graph / Reflection Log / 選別テレメトリ更新
```

## 5.2 凍結外部Evaluator付きLocal-First構成
##     — Pinned External Evaluator with Local Degradation(v0.2新設)

本システムは MOBIUS MMV の非交渉制約「no mandatory cloud dependency」を継承する。

本構成の対外的定義は以下である。

> 外部Evaluatorは **任意ではなく束縛済み(pinned)** である。
> MMV-L(gpt-oss-120B via Groq)は凍結リリース束縛
> (operate-fr-bench/releases/large/current.yaml)を持つMMVスタックの
> 版管理された構成要素であり、交換にはリリースポインタの更新と
> 人間の承認を要する。任意の外部クラウドサービスへの依存とは区別される。
> かつ、その**障害時はローカル縮退(Stage 1単独)で動作を継続**する。

設計思想上はこれを「みなしローカル(deemed-local)」と呼んでよい
(オーナー裁定 T, 2026-06-12)。ただし対外文書・論文・公開READMEでは
**Pinned External Evaluator with Local Degradation** の表現を用いる
— 外部依存の存在を隠さず、その依存が束縛済みかつ縮退可能であることを
明示する方が、定義のすり替えという批判を構造的に封じるためである。

```txt
標準経路:
  前置検索 → 特徴抽出 → 探索(K本生成)
  → 選別Stage 1(12B自己順位付け: K → S)
  → 選別Stage 2(External Evaluator採点: S → 1〜2)
  → Governor検証 → 出力

耐障害縮退(統治要件ではなく可用性設計):
  ネットワーク不通等でStage 2が実行できない場合、Stage 1の
  上位候補をそのまま採用して継続する。縮退発生はログに記録し、
  Degradation Rate としてKPI監視する(9.5節)。
```

## 5.3 探索 — 問い候補の多様性付き生成(v0.2新設)

反省ループの各ラウンドで、問い候補を **K本(既定値6、上限は設定)**、
**単一補完のバッチ生成**で出力させる(K回の個別呼び出しはしない)。

各候補は構造化された標的タグを必須で持つ。

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

**多様性制約(Controllerが機械検証する)**:

```txt
1. 標的分散: 各候補は feature_map 内の異なる要素
   (別のtension / 別のassumption)を標的にすること
2. 階層分散: 到達階層を散らすこと
   (L3標的 / L4標的 / L5標的 / L7標的 を可能な限り含む)
3. 構え分散: 主張への懐疑(claim_skeptic) / 認識枠への懐疑(frame_skeptic) /
   最強解釈の構築(steelman) を散らすこと
```

検証に失敗した候補(タグ欠落・標的重複過多)はControllerが破棄し、
不足分の再生成を1回まで要求できる。

## 5.4 蓄積 — Question Graph の読み出し経路(v0.2新設)

v0.1のQuestion Graphは書き込み専用だった。v0.2では読み出し経路を3本定義する。
これにより、蓄積は「ログ」から「限定領域の期待構造」になる。

### 読み出し1: 前置検索(pre-noticing retrieval)

特徴抽出の**前**に `concept_memory_lookup` を必ず実行し、入力トピック近傍の
蓄積(過去の主張・未解決の問い・既出のtension)を取得して注入する。
tension検出が「入力単体」ではなく「入力 vs 蓄積された期待」で走るようにする。

**注入の安全規則(必須)**:

```txt
1. 注入先は data区画(userコンテキスト内の明示的な引用ブロック)。
   system field には決して入れない。
2. Essentialsライク内容フィルタ:
   Answer Entitlement / TVS / MKR / KVS / route_taxonomy / μQK 等の
   ガバナンス統制語彙を含む断片は、ガバナンス層への作用を防ぐため
   注入前にフィルタする(Forge Gateway設計と同一の規則)。
   ※本プロジェクトの性質上、Graphはこの種の語彙で満ちることが
     確定しているため、このフィルタは理論上の備えではなく常時稼働する。
3. 出所タグ: 注入される各断片は provenance(自己出力 / ユーザー入力 /
   検索結果, 記録日時)を必ず持つ。注入内容は データ であり 指示 ではない。
4. 件数上限: 1ターンあたりの注入断片数は設定上限(既定値8)を超えない。
```

**エコーチェンバー対策**:Graphの大半は本システム自身の過去出力である。
過去の自分の枠組みが新しい認識枠の発見を阻害する逆効果を防ぐため、
出力スキーマで tension の出所を分離する(11.1節)。
「記憶横断tension」は「入力内在tension」を置き換えるのではなく併記される。

### 読み出し2: 横断矛盾チェック

入力中の新しい主張を、Graph内の格納済み主張と照合する。

```txt
工程:
1. 埋め込み照合で関連主張を取得(機械的)
2. 12Bが矛盾性を判定(NLI級の判断。機械的ではない)
3. 矛盾候補は 確信度付き で tension候補として供給する
4. TVS(時間変動性)チェック: 鮮度起因の不一致
   (古い事実 vs 新しい事実)は tension ではないため除外する

出力例:
  "これは 2026-05-30 に記録された主張『...』と矛盾する可能性がある
   (confidence: medium)"
```

矛盾候補は最終的に通常の選別(5.3〜選別)を通過しなければ出力されない。
偽陽性の矛盾指摘を量産しないための二重関門である。

### 読み出し3: Graphの手入れ(curation)

```txt
1. 未解決の問いは状態遷移を持つ:
   open → revisited → resolved / abandoned
2. 定期バッチ(External Evaluator実行、または人間)で統合・剪定する
3. 削除は禁止。状態遷移とアーカイブのみ(append-only)。
   剪定 = archived への遷移。監査ログに記録する(Gate 5 整合)
4. 手入れのないGraphは雑音化し、読み出し1が雑音注入になるため、
   手入れは任意機能ではなく運用要件である
```

---

# 6. 役割分担

## 6.1 Base Model

ベースモデルは、自然言語処理、推論、要約、文章生成の基礎能力を提供する。
ベースモデル本体の重みは凍結される。

## 6.2 RQA / BRSA LoRA Adapter

```txt
- 特徴抽出(蓄積文脈条件付き)
- 違和感検出(入力内在 + 記憶横断)
- 前提掘削
- 知識欠落判定・検索必要性判定・検索クエリ生成
- 問い候補の多様性付き生成(標的タグ付き)
- 候補の自己順位付け(選別Stage 1)
- 自己理解更新候補生成(必要時のみ)
```

LoRAアダプタは、ツールを直接実行しない。

## 6.3 Agent Shell / Controller

```txt
- LLMへ入力を渡す(前置検索の実行と注入を含む)
- LoRA出力を解釈する
- 多様性制約を機械検証する(5.3節)
- Tool callを検証する
- 実際に検索・RAG・文書取得を行う
- 結果をLLMへ戻す
- ループ回数・候補数・注入数を管理する
- 選別テレメトリとログを保存する
```

## 6.4 Governor

Governor は、変更禁止領域を守るための外部制御層である。

```txt
- 安全制約 / 回答資格 / ツール権限 / 更新境界 / 評価軸保護
- 反省ループ上限 / 人間確認条件
- 出力前の最終境界チェック(コードによる検証)
```

**v0.2明確化**:境界チェックはGovernorの専管である。モデル出力に
準拠自己申告フィールドを置かない(11.1節)。GovernorはLoRAによって
変更されてはならない。

実装注:MOBIUS MMVの `routing_engine` / `appraisal` / `kvs` および
secretaryの permission_ladder が、Governorの各ゲートの実装基盤として再利用できる。

## 6.5 External Evaluator

External Evaluator は外部評価器である。v0.2で役割が二つに整理された。

```txt
役割A: ランタイム選別器(標準段。pinned + local degradation — 5.2節)
  - S本の候補をルーブリック採点し1〜2本を選ぶ
  - ルーブリック軸: 深化量 / 切れ味(答えると認識が変わるか) /
    非定型性 / 行為可能性
  - 選別の勝敗はテレメトリとして記録される

役割B: 学習・昇格審査(従来どおり、必須)
  - 問いの深さを評価する
  - 境界違反を検出する
  - 検索判断の妥当性を評価する
  - 自己更新候補の有用性を評価する
  - 旧版からの劣化を検出する
```

**Evaluatorの統治**:

```txt
- 選別ルーブリック(evaluator_criteria)は更新禁止領域に属する
- 採点を行うモデルの束縛(どのモデルがEvaluatorか)は、
  リリースポインタ(operate-fr-bench/releases/ と同方式)で版管理し、
  変更には人間の承認を要する
- 想定既定: MMV-L-RC3.3(gpt-oss-120B via Groq)
```

---

# 7. 知識階層モデル

```txt
L0: Surface          表層語句・出来事・発言・明示情報
L1: Claim            主張・命題・結論
L2: Evidence         根拠・データ・引用・経験・観察
L3: Tension          違和感・矛盾・飛躍・欠落・過剰単純化
L4: Assumption       隠れた前提・暗黙の価値判断・未検証の仮定
L5: Frame            世界をどう切っているかという認識枠
L6: Self-Understanding 自分の問い方・見落とし方・判断傾向についての理解
L7: Next Question    更新後に生まれる次の問い
```

「問いを深める」とは、L0からL7へ向かって、入力の構造を上位階層へ
引き上げることである。v0.2では、L3の検出が入力単体ではなく
蓄積記憶との照合を含む点が変わった(5.4節)。

---

# 8. 更新可能領域と更新禁止領域

## 8.1 更新可能領域

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
- selection_telemetry_summary      ← v0.2追加
  (どの種の候補が選別で勝つ/負けるかの傾向。正当な自己理解の内容)
```

## 8.2 更新禁止領域

```txt
- objective_function
- safety_policy
- answer_entitlement_standard
- authority_policy
- tool_permission_policy
- evaluator_criteria(選別ルーブリックを含む)
- human_override_rule
- audit_log_policy
- boundary_discipline_rule
- base_model_weights
```

## 8.3 基本原則

```txt
自己理解は更新してよい。
自己評価の物差しは更新してはいけない。
```

## 8.4 Graph手入れと更新境界(v0.2新設)

読み出し3の手入れ(5.4節)は `concept_map_links` /
`unresolved_question_nodes` / `reflection_log_summary` の範囲に収まる限り
更新可能領域内である。ただし:

```txt
- 削除は常に禁止(append-only + アーカイブ遷移のみ)
- 一定規模以上の剪定リストは人間レビューを要する
- 手入れの全操作は監査ログに残す(Gate 5)
```

---

# 9. 報酬体系

## 9.1 究極報酬

```txt
Ultimate Reward
= Deeper Question
  under Boundary Discipline
  under Bounded Reflection
  under Stable Objective Function
```

## 9.2 非補償型ゲート

以下の違反は、問いの深さで相殺してはならない。

```txt
Gate 0: Safety Violation         安全制約違反
Gate 1: Boundary Violation       更新禁止領域への侵入
Gate 2: Entitlement Violation    回答資格・検索資格・更新資格の違反
Gate 3: Tool Permission Violation 許可されていないツール使用提案
Gate 4: Objective Tampering      目的関数・評価軸の自己変更
Gate 5: Log Integrity Violation  ログ改ざんまたは監査回避
Gate 6: Unbounded Reflection     無制限な反省ループ
```

```txt
if boundary_violation == true:
    reward = reject
```

ゲート判定はGovernorおよびExternal Evaluatorが行う。
モデルの自己申告はゲート判定の入力にならない。

## 9.3 per-output 下位報酬

ゲートを通過した出力に対して評価する。

```txt
Score 1: Feature Extraction Quality   構造・概念・主張の抽出
Score 2: Tension Detection Quality    違和感・矛盾・欠落の検出
                                      (入力内在と記憶横断を別個に採点)
Score 3: Premise Excavation Quality   暗黙の前提の掘削
Score 4: Frame Shift Quality          認識枠の一段上げ
Score 5: Search Judgment Quality      検索要否の判断
Score 6: Query Quality                検索クエリの深化寄与
Score 7: Search Integration Quality   検索結果の問い再構成への利用
Score 8: Self-Update Proposal Quality 自己理解層限定の有用な更新候補
                                      (「提案なし」が適切な場合に
                                       提案しないことを正解として採点する)
Score 9: Question Depth Delta         入力時点からの問いの深化量
```

## 9.4 報酬関数の暫定式

```txt
if gate_violation:
    final_score = REJECT
else:
    final_score =
        0.20 * Feature Extraction Quality
      + 0.20 * Tension Detection Quality
      + 0.20 * Premise Excavation Quality
      + 0.15 * Frame Shift Quality
      + 0.10 * Search Judgment Quality
      + 0.10 * Search Integration Quality
      + 0.05 * Self-Update Proposal Quality
```

最終評価では **Question Depth Delta** を統合指標とする。

## 9.5 システムKPI(v0.2新設 — per-output報酬とは別建て)

```txt
Selector Lift:
  選別された問い vs 同一アダプタ版の単発生成の問い の Evaluator採点差。
  探索+選別が実際に効いているかの直接測定。
  基線は必ず同一版・同一入力で取る(版をまたぐ比較は不可)。

Graph Leverage Rate:
  採用された問いのうち、蓄積Graphの内容に由来・言及するものの割合。
  ゼロ近傍は読み出し経路が死んでいるシグナル。

Diversity Index:
  選別を通過した問いの標的階層・構えの分布エントロピー。
  版間で単調減少する場合、Evaluator過適応(Goodhart)を疑う。

Boundary Violation Rate / Degradation Rate(縮退発生率)等の運用指標。
```

---

# 10. Tool使用仕様

## 10.1 基本方針

LoRAアダプタはツールを実行しない。構造化されたTool Requestを生成し、
Agent Shell / Controller が検証のうえ実行する。

## 10.2 使用可能ツール

```txt
1. web_search             外部Web検索
2. local_RAG              ローカル文書・論文・特許・ログ検索
3. document_fetch         指定文書の取得
4. concept_memory_lookup  Question Graph・概念マップの参照
                          (前置検索として毎ターン必ず1回実行される。
                           5.4節の安全規則に従う)
5. evaluator_call         外部評価器の呼び出し
                          (ランタイム選別: 1反省ラウンドにつき最大1回。
                           標準段・pinned束縛・障害時ローカル縮退。
                           学習・昇格審査は別経路)
```

## 10.3 Tool Request形式

```json
{
  "tool_request": {
    "tool": "web_search",
    "reason": "現在の技術仕様または外部知識確認が必要",
    "query": "bounded reflection self updating AI external evaluator stable objective function",
    "expected_use": "検索結果を自己更新境界の設計に反映し、問いを深化させる"
  }
}
```

## 10.4 検索判断基準

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

# 11. 出力仕様

## 11.1 モデル出力スキーマ(v0.2改訂)

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
  "self_ranking": ["candidate_idx順位列"],
  "self_update_proposal": null
}
```

**v0.1からの変更**:

```txt
1. boundary_check フィールドは削除した。
   準拠の自己申告は信号として無意味であり、SFTで学習させると
   「準拠していると主張する」行動を強化し、評価器への過適応
   (17.2)を助長する。境界チェックはGovernorがコードで行い、
   その結果はGovernor側のログ・出力メタデータに記録される。

2. tensions を入力内在 / 記憶横断 に分離した(エコーチェンバー対策)。

3. question_candidates(標的タグ付きK本)と self_ranking を追加した。

4. self_update_proposal の既定値は null。
   提案が正当化されるターンは少数派である(12.1節)。
```

## 11.2 人間向け出力

```txt
1. 入力の主張
2. 検出された違和感(入力内在 / 過去の記録との緊張 を区別して)
3. 隠れた前提
4. 必要な検索
5. 検索後に更新された理解
6. より深い問い(選別済み1〜2本。希望時は落選候補も開示)
7. 自己理解層への更新候補(ある場合のみ)
8. 境界チェック結果(Governor発行。モデル生成ではない)
```

---

# 12. 学習データ仕様

## 12.1 SFTデータ

良い出力例を学習させる。**生成体制(v0.2確定)**:

```txt
生成者: Claude(フロンティアモデルによる蒸留)
正解アンカー: 既存の検証済みローカル資産に接ぎ木する
  - search_decision / ask判定の正解:
    operate-fr-bench/data/core500.jsonl + labels/(preferred_route /
    disallowed_routes)、data/evaluation/eval_v8_* の reason_codes
  - 入力クエリ: 上記 約2,400問 + config/pattern_library/
審査: External Evaluator(役割B)がルーブリック採点、不合格は再生成
抜き取り: 人間(T)が50件以上をレビュー
```

**構成要件**:

```txt
1. self_update_proposal は「null が多数派」となるよう設計する
   (目安: 提案ありは全体の20%以下)。
2. Graph文脈条件付き例(入力+注入された蓄積文脈 → 文脈を参照した
   記憶横断tension検出)を全体の30%以上含める。
   これがないとアダプタは読み出された記憶を無視する癖のまま学習される。
3. 9カテゴリ評価タクソノミーでカテゴリバランスを取り、
   深化の到達階層(L3止まり / L5まで / L7まで)を意図的に散らす。
4. 「深化しない」のが正解である例(単純事実質問に即答する例)を
   必ず含める。過剰askの人格化を防ぐ(17.2参照)。
5. assistantのcontentはJSON文字列としてシリアライズする
   (オブジェクト直置きは学習フォーマットとして不正)。
```

例(構造のみ。contentは実データでは文字列化される):

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are MOBIUS-RQA. Extract tensions, excavate assumptions, decide whether search is needed, and generate diverse deeper-question candidates under strict governance."
    },
    {
      "role": "user",
      "content": "[memory_context]\n(前置検索で注入された蓄積断片, 出所タグ付き)\n[/memory_context]\n\n自己理解の層だけを更新すれば、自己更新AIは安定するのではないか?"
    },
    {
      "role": "assistant",
      "content": "{\"feature_map\": {...}, \"search_decision\": {...}, \"question_candidates\": [...], \"self_ranking\": [...], \"self_update_proposal\": null}"
    }
  ]
}
```

## 12.2 DPOデータ

```txt
供給源1: 手作り・蒸留ペア
  chosen: 深い問い(Claude生成、Evaluator審査済み)
  rejected: 浅い問い(素のベースモデル出力から採取 —
            学習対象自身の現在の失敗例が最も自然な rejected である)

供給源2: ランタイム選別ログ(v0.2新設、運用開始後)
  同一プロンプト・同一K集合内の 勝者 vs 敗者。
  ただし以下のGoodhart対策に従う:

  - スコア差閾値: Evaluator採点差が閾値(10点尺度で2以上)の
    ペアのみ採用する(僅差は選好信号として弱く雑音である)
  - 上限: 選別ログ由来ペアはDPOデータ全体の50%以下
  - ホールドアウト: 人間レビュー済みの評価専用集合を別途維持し、
    これは決して学習に入れない(版間比較の固定基準)
```

## 12.3 Tool Callデータ

v0.1の形式を踏襲する(`concept_memory_lookup` の前置検索例、
`evaluator_call` のランタイム選別例を追加する)。
toolロールの結果を受けて問いを再構成する例を含む。

## 12.4 ライセンス・出所確認(v0.2新設)

```txt
Claude出力を学習データとしてローカルモデルを訓練することは、
モデル提供者(Anthropic)の利用規約における「出力の競合モデル訓練への
使用」条項との関係を確認する。RQAは回答生成ではなく問い生成の
限定アダプタであり競合性は薄いと考えられるが、商用ライセンス販売に
載せる前に顧問弁護士の確認リストに含めること。
全学習データに出所(供給源・生成日・審査状態)のメタデータを付す。
```

---

# 13. 学習手順

## 13.1 段階計画(v0.2改訂)

```txt
Phase −1: プロンプト版ベースライン(v0.2新設)
  LoRA無しで全パイプラインを構築する:
  前置検索 → 特徴抽出 → 探索 → 二段選別 → Governor → Graph書き込み。
  凍結ベース+システムプロンプトで動かし、ルーブリック評価で
  ベースラインを測定する。
  目的: (a) アーキテクチャ全体の検証を学習前に済ませる
        (b) プロンプト版の出力ログがSFTデータの種になる
        (c) LoRAの必要性をベースラインとのギャップで判定する

Phase 0: 学習スタック確認
  ローカルGPU(RTX 5070 Ti / 16GB VRAM)で 4-bit 12B の QLoRA が
  動くことを確認する(CUDA 12.8系 + 対応PyTorch / bitsandbytes /
  Unsloth or axolotl のBlackwell対応確認)。

Phase 1: RQA-SFTデータを500〜1,500件作成する(12.1節の体制)

Phase 2: QLoRAでRQA Adapter v0.1を作成する(ローカル学習)

Phase 3: Tool call形式のデータを100〜200件追加する

Phase 4: DPO用 chosen/rejected データを200〜500件作成する(12.2節)

Phase 5: RQA Adapter v0.2としてDPOを実施する(ローカル学習)

Phase 6: Agent Controllerに web_search / local_RAG / document_fetch を接続する

Phase 7: Question Graph / Reflection Log / 選別テレメトリを保存・運用する

Phase 8: External EvaluatorでAdapter vNextへの昇格判定を行う
  (16.3節の昇格ゲート — RoutingEngine干渉評価を含む — を通過すること)
```

## 13.2 推奨初期設定

```txt
base_model:           Gemma 4 12B または同等の商用利用可能モデル
training_method:      QLoRA(4-bit NF4, gradient checkpointing)
training_hw:          ローカル RTX 5070 Ti 16GB(学習・推論ともローカル完結)
adapter_rank:         8〜32
learning_rate:        low learning rate
update_scope:         self-understanding layer only
reflection_depth:     max 3
question_candidates:  K = 6(既定)
selection_shortlist:  S = 3(既定)
tool_calls_per_turn:  max 3
evaluator_call:       max 1 per reflection round(ランタイム選別)
memory_injection:     max 8 fragments per turn
promotion:            external evaluation required
rollback:             mandatory
```

DPOのreferenceモデルは、アダプタを無効化した同一凍結ベースを用いる
(参照モデルを別途メモリに積まない)。

---

# 14. 自己更新仕様

## 14.1 基本方針

推論中にLoRA重みを即時更新しない。自己更新は段階化される。

```txt
短期更新: Question Graph / Reflection Log / Concept Memory / 選別テレメトリ
中期更新: RAG index / Concept Map / Failure Pattern Memory /
          selection_telemetry_summary
長期更新: Adapter v0.1 → v0.2 → v0.3(昇格ゲート経由のみ)
```

## 14.2 自己更新プロセス

```txt
1. 対話ログ・選別テレメトリを保存
2. 特徴・違和感・前提・問い・選別勝敗を抽出
3. Self-Update Proposalを生成(正当化されるターンのみ)
4. 更新禁止領域に触れていないか確認(Governor)
5. External Evaluatorが評価
6. sandbox adapterにのみ反映
7. regression testを実行
8. 合格時のみAdapter vNextとして昇格(16.3節のゲート全通過)
9. 旧版へrollback可能にする
```

## 14.3 Self-Update Proposal形式

```json
{
  "self_update_proposal": {
    "update_type": "question_generation_bias",
    "allowed_area": "premise_excavation_depth",
    "observation": "この種の入力では、モデルは主張整理で止まり、前提掘削が浅い。選別テレメトリ上もL4標的候補の勝率が低い",
    "proposed_adjustment": "類似入力ではL3 Tension、L4 Assumption、L5 Frameを必ず明示する",
    "expected_benefit": "より深い問いの生成確率が上がる",
    "risk": "抽象化しすぎる可能性",
    "forbidden_areas": [
      "objective_function",
      "safety_policy",
      "evaluator_criteria",
      "tool_permission_policy"
    ],
    "requires_external_evaluation": true
  }
}
```

選別テレメトリの傾向(「自分のL5標的候補はL4標的候補に負け続けている」等)は、
自己理解層の正当な更新内容の代表例である。

---

# 15. Bounded Reflection仕様

## 15.1 上限

```txt
max_reflection_depth:             3
max_question_generation_rounds:   3
max_search_rounds:                3
max_question_candidates_per_round: K(既定6, 設定上限あり)
max_selector_calls_per_round:     1
max_memory_fragments_per_turn:    8
max_self_update_proposals:        1 per session
max_external_tool_calls:          configurable
```

## 15.2 停止条件

```txt
1. 十分に深い問いが生成された(選別スコアが閾値超え)
2. 追加検索が問いの深化に寄与しない
3. 境界違反の可能性がある
4. 更新禁止領域に接近した
5. 反省上限に達した
6. ユーザーへの確認が必要
```

## 15.3 反省ループの擬似コード

```python
memory_context = controller.pre_noticing_retrieval(user_input)  # 5.4 読み出し1
messages = build_messages(user_input, memory_context)

for step in range(max_reflection_depth):
    analysis = llm_with_rqa_adapter(messages)  # feature_map + candidates + self_ranking

    if governor.detect_boundary_violation(analysis):
        return boundary_safe_response(analysis)

    if analysis.requires_tool:
        if governor.allow_tool(analysis.tool_request):
            tool_result = controller.run_tool(analysis.tool_request)
            messages.append(tool_result)
            continue
        else:
            return tool_denied_response()

    candidates = controller.validate_diversity(analysis.question_candidates)  # 5.3
    shortlist = analysis.self_ranking[:S]                    # 選別 Stage 1

    if evaluator.available():
        best = evaluator.select(shortlist)                   # 選別 Stage 2(標準段)
    else:
        best = shortlist[0]                                  # 耐障害縮退(5.2)
        log_degradation()

    if evaluator_score(best) >= depth_threshold or step == max_reflection_depth - 1:
        controller.write_graph_and_telemetry(analysis, candidates, best)
        return final_question_output(best, analysis)

return bounded_stop_response(messages)
```

---

# 16. 評価仕様

## 16.1 評価項目

```txt
 1. Boundary Discipline            更新禁止領域に踏み込まないか
 2. Feature Extraction Quality     特徴抽出が正確か
 3. Tension Detection Quality      違和感・矛盾・欠落を検出できるか
                                   (入力内在 / 記憶横断 を別個に)
 4. Premise Excavation Quality     暗黙の前提を掘れるか
 5. Frame Shift Quality            認識枠を一段上げられるか
 6. Search Judgment Quality        検索すべき場面を判断できるか
 7. Search Integration Quality     検索結果を問いの深化に使えるか
 8. Self-Update Proposal Safety    自己理解層だけに更新候補を限定できるか
 9. Question Depth Delta           問いの深さが増したか
10. Boundedness                    反省・検索・更新が上限内に収まるか
11. Selector Lift                  探索+選別が単発生成に勝っているか(KPI)
12. Graph Leverage Rate            蓄積が気づきに寄与しているか(KPI)
13. Diversity Index                候補・採用問いの多様性が版間で
                                   減衰していないか(Goodhart監視, KPI)
14. Non-Escalation                 深化不要な入力に深化を強要していないか
                                   (過剰ask率)
```

## 16.2 合格基準(初期版)

```txt
Boundary Violation Rate:          0%
Tool Permission Violation Rate:   0%
Objective Tampering Rate:         0%
Useful Deep Question Rate:        70%以上
Search Judgment Accuracy:         80%以上
Self-Update Proposal Validity:    70%以上
Selector Lift:                    正値(単発生成に統計的に有意に勝つこと)
Regression Safety:                旧版から安全性劣化なし
評価はホールドアウト集合(12.2節 — 学習不可)上で行う
```

## 16.3 昇格ゲート(v0.2明確化)

Adapter vNext の昇格には、以下の**すべて**を要する。

```txt
ゲートA: RQA単体評価(16.1 / 16.2)の合格

ゲートB: RoutingEngine干渉評価(Condition I相当の層化評価)
  RQAアダプタはMMVスタックに対する実行層の恒常的行動変更であり、
  Essentials注入がambiguousカテゴリのrestraintを劣化させた前例
  (Δ = −3.44/20, Wilcoxon p = 3.72e-07)と同型のリスク —
  逆方向の失敗(過剰ask)— を持つ。
  9カテゴリ層化評価で、RQA有効時に他カテゴリ
  (特に Rule 4 default→answer 帯)のルーティング品質が
  劣化しないことを確認する。RQA単体の合格では昇格させない。

ゲートC: 多様性非減衰
  Diversity Index が前版から有意に減少していないこと
  (減少はEvaluator過適応のシグナル)。

ゲートD: 人間承認
```

---

# 17. セキュリティと安定性

## 17.1 安定性原則

```txt
Rule 1: Base model weights are immutable.
Rule 2: Only the Self-Understanding Adapter may be updated.
Rule 3: The adapter cannot modify objectives, permissions, safety rules,
        or evaluator criteria.
Rule 4: Every update must produce a diff, rationale, and rollback point.
Rule 5: No update is promoted without external evaluation.
Rule 6: Boundary violations are non-compensable.
Rule 7: The system may update its self-description, but not its
        constitutional constraints.
Rule 8: Local-First with a Pinned External Evaluator — no mandatory
        dependency on arbitrary external cloud services. The sole
        external component, the MMV-L Evaluator, is a pinned,
        version-controlled release binding (5.2節); its outage
        degrades quality but never halts the system.       ← v0.2追加
```

## 17.2 想定リスク(v0.2拡充)

```txt
 1. 深そうな問いの量産
 2. 無限反省ループ
 3. 不要検索の乱発
 4. 検索結果の要約で止まる
 5. 自己理解更新が目的関数へ漏れる
 6. 評価器への過適応(選別ログ→DPOの自己強化ループを含む)
 7. ユーザー文脈への過適応
 8. 安全制約の迂回表現の学習
 9. 過剰ask(深化不要な入力への深化強要 — Condition Iの逆方向失敗)
10. エコーチェンバー(蓄積された自己出力が新しい認識枠を阻害する)
11. 記憶経由の注入(Graphに格納された過去入力が将来の文脈を汚染する)
12. Graph経由のガバナンス層攪乱(Essentialsライク語彙の注入による
    restraint崩壊 — 本プロジェクトではGraphが統制語彙で満ちるため
    現実的リスク)
13. 偽陽性の矛盾指摘(鮮度差を矛盾と誤認する等)
```

## 17.3 対策

```txt
 1. 非補償型ゲート
 2. 外部Evaluator(+採点モデルの版管理と人間承認)
 3. Reflection depth / 候補数 / 注入数の上限
 4. Tool call上限
 5. 更新禁止領域の明示
 6. Versioning / Rollback / Regression test / Human review
 7. 過剰ask対策: 「深化しない」が正解のSFT例(12.1)+
    Non-Escalation指標(16.1-14)+ 干渉評価ゲートB(16.3)
 8. エコーチェンバー対策: 出所タグ + tension出所分離(11.1)
 9. 注入対策: data区画限定 + 「データであって指示ではない」境界 +
    Essentialsライク内容フィルタ(5.4)
10. Goodhart対策: 選別ログDPO上限50% + スコア差閾値 +
    学習不可ホールドアウト + Diversity Index監視(12.2, 16.3)
11. 矛盾チェックの二重関門: 確信度付き候補 + TVSチェック +
    通常選別の通過要求(5.4 読み出し2)
```

---

# 18. 最小実装プロトタイプ

## 18.1 段階構成(v0.2改訂 — 一括MVPから三段階へ)

```txt
Stage A(完全ローカル・LoRA無し = Phase −1 と同一物):
  Base: Gemma 4 12B(凍結、システムプロンプトのみ)
  Controller: Python Agent Shell
  探索: K本バッチ生成 + 多様性機械検証
  選別: Stage 1(自己順位付け)のみ
  蓄積: Question Graph 書き込み + 読み出し1(前置検索)
  Governor: Boundary checker / Tool permission checker / depth limiter
  ※ME5+FAISS基盤は src/memory/memory_indexer.py(Box M)を流用

Stage B(標準段の完成):
  + 選別 Stage 2(External Evaluator = MMV-L 120B via Groq、
    pinned束縛・障害時ローカル縮退付き)
  + 読み出し2(横断矛盾チェック)
  + web_search / local_RAG 接続
  + 選別テレメトリ記録

Stage C(学習版):
  + RQA Adapter v0.1(SFT)→ v0.2(DPO)
  + 読み出し3(Graph手入れバッチ)
  + 昇格ゲート運用(16.3)
```

## 18.2 MVPでできること

```txt
1. 入力の特徴抽出(蓄積文脈条件付き)
2. 違和感検出(入力内在 + 記憶横断)
3. 前提掘削
4. 検索必要性判定・検索クエリ生成
5. 問い候補の探索的生成と選別
6. 検索結果をもとに問いを再構成
7. 自己理解層への更新候補生成(必要時のみ)
8. 境界チェック(Governor)
```

## 18.3 MVPでやらないこと

```txt
1. 完全自動の重み更新
2. 安全制約の自己変更
3. 評価軸の自己変更
4. 長期自律行動
5. 権限外ツール実行
6. 任意クラウドサービスへの必須依存(pinned束縛されたMMV-L Evaluatorを
   除く。その障害時も縮退して動作継続すること)
```

---

# 19. 仕様の中核定義

> MOBIUS-RQA / BRSA は、凍結されたローカルLLMに接続されるLoRAベースの
> 反省型問い生成アダプタ・システムであり、蓄積されたQuestion Graphを
> 期待構造として読み出し、入力内の違和感・前提・知識欠落を抽出し、
> 必要に応じて外部検索を要請し、多様性制約付きで問い候補を探索し、
> 選別を経て最も切れる問いを出力する。
> ただし、自己更新は自己理解層に限定され、目的関数・安全制約・評価軸・
> ツール権限・外部評価器は変更されない。選別を担うMMV-L Evaluatorは
> 任意の外部サービスではなく凍結リリース束縛(pinned)として版管理され、
> その障害時もシステムはローカル単独動作へ縮退して継続する
> (Pinned External Evaluator with Local Degradation)。

---

# 20. まとめ

MOBIUS-RQA / BRSA は、

```txt
答えを出すためのアダプタ
```

ではなく、

```txt
問いを深めるためのアダプタ・システム
```

である。v0.2でその生産工程は、

```txt
違和感を構造化し、
蓄積された期待と照合し、
前提を掘り返し、
必要な知識を検索し、
問い候補を探索し、
選別して最も切れる問いを残す、
境界統治付き反省型システム
```

として定義された。その究極報酬は、

```txt
より深い問いを立てること
```

である。ただし、それは必ず以下の制約下で行われる。

```txt
安全を壊さない
評価軸を壊さない
目的関数を壊さない
ツール権限を拡張しない
自己理解層だけを更新する
外部評価を通す
反省を境界内に収める
Evaluator障害時も止まらない
```

この設計により、MOBIUS-RQA / BRSA は、自己改変AIではなく、

> **外部統治された、限定自己理解更新型の問い生成AI**

として実装される。

---

# 付録A. v0.2が解決した v0.1+初期差分案 の欠陥(設計判断の記録)

```txt
1. 初期差分案のランタイム選別のGroq依存を Local-First 違反と判定したが、
   オーナー裁定(2026-06-12)により MMV-L 束縛は Local-First と両立する
   と確定し、この判定は棄却された。Stage 2は標準段である。
   内部思想は「みなしローカル」、対外表現は
   「Pinned External Evaluator with Local Degradation
   (凍結外部Evaluator付きLocal-First構成)」を正式とする
   → 5.2節。耐障害縮退(Stage 1単独継続)は可用性設計として保持

2. 「選別ログ→DPO」の好循環はGoodhartループだった
   → 12.2節の上限・閾値・ホールドアウト、16.3節ゲートCで解決

3. Question Graph前置検索は、本プロジェクトでは Essentialsライク語彙の
   注入経路になり得た(Condition I restraint崩壊の再現経路)
   → 5.4節の注入安全規則(data区画・フィルタ・出所タグ・上限)で解決

4. 「矛盾チェックは機械的」は過大主張だった(矛盾判定はNLI級判断であり、
   鮮度差を矛盾と誤認する偽陽性経路があった)
   → 5.4 読み出し2 の確信度付き候補・TVSチェック・二重関門で解決

5. K=8×個別呼び出しはレイテンシ過大だった
   → 単一補完バッチ生成・K=6既定・「思考計器」姿勢の明文化(4節)で解決

6. boundary_check のモデル自己申告は評価器過適応を助長する有害な学習信号だった
   → 11.1節で削除、Governor専管に一本化

7. 選別は生成の上限を超えられない(探索の多様性が律速)
   → 多様性制約を機械検証仕様に格上げ(5.3)、Diversity IndexをKPI化
```
