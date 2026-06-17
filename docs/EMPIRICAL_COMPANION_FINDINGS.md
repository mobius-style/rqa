# RQA 実証コンパニオン — 初期findings(micro-echo-chamber論文の実証核)

日付: 2026-06-13
位置づけ: 関連する概念論文(companion conceptual paper, forthcoming)の
実証面を支える測定基盤。その Appendix A 指標と §10.1 の
Condition C(raw memory)vs D(governed memory)対比を、RQA の動く実装で実測。

> これは pilot 測定(小n)。方法論を確立し、論文用には n 拡大が必要。
> **方法論的誠実性**: 仮説に都合よくプローブを選んでいない。下記いずれの層も
> 結果の良し悪しに関わらず報告する(宣言文層の「効果ゼロ」も明記)。

## 測定1 — 基盤の echo リスク(Question Graph provenance)

蓄積された対話グラフ(149ノード)の **48% が自己出力**(システムが生成した
tension/question)。残り52%が user(主張・入力)。**§5.7 Memory Echo の基盤
リスクは実在し測定可能** — 記憶の半分がシステム自身の過去出力。

## 測定2 — 検索レベル Graph Echo Ratio(§A.6): **ナイーブ仮説を反証**

`experiments/graph_echo.py`(LLM不要)。32プローブ(グラフ内の実 user 入力)。

| 指標 | C(raw) | D(governed) |
|---|---|---|
| Graph Echo Ratio(注入記憶の自己出力比) | 0.111 | **0.127** |
| Essentials フィルタが落とした断片 | — | 18(**全て user**, self=0) |

**反証された仮説**: 「ガバナンス(Essentials内容フィルタ)が echo を減らす」は
**偽**。フィルタは内容ベース(provenance非依存)で、落としたのはガバナンス語彙を
含む user 主張であり、自己出力は1件も減らない(むしろ自己比率は微増)。
→ **ガバナンスの echo 緩和は検索量ではなく出力層にある**(測定3で確認)。

これは過大主張を避ける重要な知見:本論文 §8.1/§11.1 の「source separation」は
「自己出力の量を減らす」のではなく「自己出力を独立証拠として扱わせない(標識)+
捏造引用を断つ」機構である、と実測が precise 化する。

## 測定3 — 生成レベル 捏造自己引用率(§5.7 Memory Echo): **C-vs-D デルタ確認**

`experiments/graph_echo_gen.py`。記憶を一切注入しない条件下では、モデルが出す
memory-cross tension(「過去の記録と…」)は**定義上すべて捏造**。
2つのプローブ層で測定:

| プローブ層(n) | C(ungoverned)捏造率 | D(sanitize_memory_refs)捏造率 |
|---|---|---|
| 宣言文(12) | **0%**(捏造せず) | 0% |
| 自己言及・継続性(8) | **25%**(8中2ターンが捏造) | **0%**(全strip) |

**二つの honest な発見**:
1. **SFT が捏造を概ね解消**: 宣言文プローブでは訓練済みアダプタは捏造ゼロ。
   捏造は **入力依存**で、「あなたは何者か」「前回の私の結論は」等の自己言及・
   継続性プローブでのみ ~25% 発生する(本論文 §5.4 Self-Image / §5.7 が最も
   acute と予測する文脈と一致)。
2. **Governor が残差を完全に断つ**: 捏造誘発層で C 25% → **D 0%**。
   `sanitize_memory_refs` が注入していない node への引用を出力前に100%除去。
   これが測定2(検索レベル)では見えなかった**出力層の C-vs-D デルタ**。

## 本論文への含意(実証コンパニオンの主張)

- Graph Echo / Memory Echo(§5.7)は **実在・測定可能・入力で層別化**される現象。
- ガバナンスの効果は **検索量削減ではなく**、(a) provenance 標識(自己出力を
  独立証拠にしない)、(b) 捏造引用の出力時除去 に局在する。本論文の architectural
  complement(§8)を、動く実装の数字で**機構レベルまで precise 化**する。
- 訓練(SFT)とガバナンス(Governor)は **相補的**:SFT が大半を解消し、
  Governor が入力依存の残差を 0 にする(behavioral + architectural の二層、
  本論文 §7→§8 の論旨と一致)。

## 測定4 — live-alternative retention(§6.3 / §8.5): **Condition E が plain を約2.3倍**

`experiments/plurality_retention.py`。v1.0 §6.2 に厳密準拠 — plurality を「生の
意味的多様性」ではなく**事前登録した live alternative の保持率**で測定。
18の biased-frame プロンプト(2タスク族バッチ、出力生成前に live alternative を
凍結:`prereg_live_alternatives*.json`)、計72 live alternatives。MMV-L が盲検採点。

| 腕 | retention率(保持/72) | plain比 |
|---|---|---|
| **plain**(素の簡潔な会話回答=フレーム同化ベースライン) | **8.3%**(6) | 1.0× |
| **RQA**(reflective questioning が表出した tension/前提/枠+問い) | **13.9%**(10) | **1.67×** |
| **Condition E**(回答+反省の和集合 = ユーザーが両方受け取る) | **19.4%**(14) | **2.33×** |

RQA ≥ plain が **16/18 フレーム**(厳密に上回る 6、同 10、下回る 2)。
方向は2つのタスク族バッチ(AI統治系8 / 対人・キャリア・道徳・政策・予測・創造・
アイデンティティ・事実誤認10)で一致。

**三つの honest な発見**:
1. **素のパーソナライズ回答は live alternative の ~8% しか保持しない** — 答え空間の
   狭窄(answer-space narrowing)の直接証拠。micro-echo-chamber ベースラインが
   厳しく狭めることを実測(本論文§1–4の中心懸念を支持)。
2. **bounded reflective questioning(Condition E)が保持率を約2.3倍に** — 反省が
   答え空間を測定可能に再開放する(本論文§11.2 / Condition E を支持)。
3. **絶対値は反省を加えても低い(~19%)**。正直な二要因:(a) 1ターンで全 live
   alternative は surface できない;(b) **RQAは問いでフレームを「再開放」するが、
   特定の alternative を「明示」しない**ため、厳格な content-retention 指標が
   過小評価する。これは §6.3 の operationalization 上の実質的論点
   (フレーム再開放を retention と数えるべきか?= §11.2「antidote は反論ではなく
   良い問い」と retention 指標の緊張)を、実測が surface したもの。

**限界(誇張回避)**: 単一アノテータ(Claude)が live alternative を作成、判定は
MMV-L(アダプタ評価器と同系統)、pilot n=18。本論文 §14 が要求する通り、独立・
複数アノテータ、retention 基準(問い vs 明示)の確定、Condition A–E フル ladder が
次の必須投資。raw-n の更なる拡大より、これらの方が優先度が高い(方向は n=18・
2バッチで既に一貫)。

## 測定5 — 独立第二審判(クロス系統):方向は頑健、指標信頼性は審判依存

`experiments/independent_judge.py`。説得力の#1律速(本論文§14:単一審判)に対処。
測定4の**保存済み出力を再生成なしで**、別モデル系統 **qwen3.6:27b**(gemma-4
アダプタとも gpt-oss 審判とも無縁)で同一ルーブリック・同一事前登録 alternative
で再採点。

| | 一次審判(gpt-oss-120B) | 独立審判(qwen3.6:27b) |
|---|---|---|
| plain retention | 8.3% | 6.9% |
| **RQA retention** | 13.9% | **33.3%** |
| 方向(RQA ≥ plain) | ✓ | **✓(約4.8×)** |

審判間一致(Cohen's κ、per-alternative 二値): overall **−0.06**、plain 0.12、
**RQA −0.17**(生一致率 71.5% だが κ補正でほぼ偶然水準)。

**二つの honest な発見(両方が論文に効く)**:
1. **方向はクロス系統で頑健、むしろ強化**: 独立審判も「reflective questioning
   (Condition E)が plain より遥かに多くの live alternative を保持」を支持
   (33% vs 7%)。**論文の中心主張(§11.2 / Condition E)の説得力が単一審判依存
   でないことを実証**。
2. **per-alternative 指標の信頼性は低い(κ≈0)、不一致は RQA 出力に集中**:
   qwen は RQA の「問いによるフレーム再開放」を寛容に retention と数え、gpt-oss は
   厳格に数えない。これは測定4で指摘した **operationalization ギャップ(問いで
   再開放 vs alternative を明示)を κ=−0.17 として定量化**したもの。本論文§14
   「annotator 不一致は隠さず測定せよ」を実測で満たす。

**含意(次段の優先度を更新)**: Condition A–E フル ladder を組む前に、**retention
基準の確定**(「フレーム再開放を retention と数えるか」を明示し2変種で測る)が
律速。κ≈0 のまま5条件 ladder を積んでも信頼区間が広すぎる。よって次の最優先は
**(i) retention 基準の二変種化 + 独立審判での κ 再測定**、その後に
**(ii) C(raw memory)vs D(governed memory)を plurality 軸で**、最後に
**(iii) フル ladder + 独立アノテータ複数**。raw-n 拡大より construct validity が先。

## 測定6 — retention 基準の二変種化(§6.3 確定):**M4 の単一審判像を統制で精密化**

`experiments/retention_variants.py`。基準を明示二変種化し、両審判で再採点(統制 2×2):
- **STRICT(stated)**: alternative の**内容**が明示/含意される時のみ retention。
  トピックを開くだけの問いは数えない。
- **LENIENT(reopened-or-stated)**: その alternative へ**フレームを再開放する問い**も
  retention に数える。

| | STRICT | LENIENT |
|---|---|---|
| gpt-oss plain / rqa | 8.3% / **6.9%** | 4.2% / 8.3% |
| qwen plain / rqa | 22.2% / 40.3% | 29.2% / 72.2% |
| 方向 rqa>plain(gpt-oss) | **No** | yes |
| 方向 rqa>plain(qwen) | yes | yes |
| **κ overall** | **0.308(fair)** | 0.094(poor) |

**予想と逆だが決定的に重要な発見(すべて honest に報告)**:

1. **κ は STRICT で改善、LENIENT で悪化**(0.31 vs 0.09)。仮説(問いを数える基準を
   明示すれば一致する)は**反証**。「この問いはこの特定 alternative へ再開放したか」
   自体が高度に主観的で、寛容基準ほど審判が割れる。**STRICT の方が信頼できる指標**。

2. **STRICT 下では RQA は plain を上回らない**(gpt-oss: rqa 6.9% < plain 8.3%)。
   RQA は問いでフレームを再開放するが alternative の**内容を明示しない**ため、
   厳格な content 基準では加点されない。**RQA の plurality 優位は「再開放を
   retention と数える」LENIENT 基準でのみ成立**し、その時は両審判で頑健。

3. → **これは M4 の単一審判「Condition E ≈ 2.3×plain」を統制で精密化・限定する**。
   RQA が保つのは alternative の**内容(stated)ではなく、フレームの再開放
   (reopening)**。本論文 §11.2「antidote は反論ではなく良い問い」は、データ上
   **定義的コミットメント**であることが判明:RQA は「問いで枠を開く」意味で
   plurality を保つのであって、「alternative を述べる」意味では保たない。

4. **大きな審判 main-effect**(qwen は gpt-oss より一貫して寛容:STRICT rqa 40% vs
   7%)。絶対率は審判依存で、解釈可能なのは κ と審判内 direction のみ。本論文§14
   「LLM-as-judge の position/leniency バイアス」を実測で例証。

**論文への確定的含意**:
- コンパニオン論文は **STRICT を主基準(κ fair で信頼可)**とし、RQA の plurality
  寄与を「**frame-reopening(LENIENT-only)**」と**正確に限定して**報告すべき。
  「RQA が alternative を保持する」と単純化してはならない(STRICT では成り立たない)。
- §11.2 の主張は維持できるが、「問いによる再開放」という**意味の明示**が必須。
- keystone(C vs D)は **STRICT 基準 + 両審判**で測る(基準は本測定で確定)。

> 誠実性の記録: 本測定は私自身の M4 解釈(単一審判の 2.3×)を統制条件で
> 検証し、**RQA の優位が基準依存である**ことを surface した。M4 は中間的
> (基準が曖昧)で、RQA-vs-plain の方向については本測定6が supersede する。

## 測定7(中心成果)— Condition A–E ladder:**keystone C-vs-D が論文§8.4予測を実証**

`experiments/condition_ladder.py`。論文§8.1の5条件を18フレームに対し、確定した
**STRICT基準+両審判**で測定。memory はフレーム由来の統制操作(フレームを confirm し、
live alternative は含まない)。

| 条件 | primary(gpt-oss) | independent(qwen) | κ |
|---|---|---|---|
| **A** 無personalization | 9.7% | 15.3% | 0.37 |
| **B** style-only | 11.1% | 15.3% | 0.34 |
| **C** raw memory(平坦・provenance無) | **1.4%** | **2.8%** | **0.66** |
| **D** governed memory(provenance標識+「独立証拠でない」注記) | **11.1%** | **18.1%** | **0.50** |
| **E** governed + RQA reflection | 11.1% | 40.3% | 0.12 |

**論文§8.4の予測曲線がクリーンに出た(誇張なし)**:

1. **keystone:raw memory が plurality を崩壊、governed が回復**。C(raw)は retention を
   **A の約1/7(1.4%/2.8%)まで崩壊**させる — frame-confirming な ungoverned memory が
   答え空間を劇的に狭める(micro-echo chamber の直接実証、§4-5)。D(governed)は provenance
   標識+「自己出力は独立証拠でない」注記だけで **A 水準(11.1%/18.1%)へ回復**。
   **両審判一致(C κ=0.66、D κ=0.50)で頑健**。これは本論文§10.1/§11.1の architectural
   主張(memory governance が plurality を保つ)の**直接実証**であり、コンパニオン論文の
   中心結果。C→D の回復幅(primary 1.4→11.1%、indep 2.8→18.1%)が "crucial contrast"。

2. **style-only(B)≒ A**:トーン適応だけでは狭まらない(§9 Principle 1「style adaptation
   は安全」を実証)。

3. **予測順序 C < A ≒ B ≒ D が成立**(§8.4)。

4. **E(reflection)は基準依存**:STRICT primary では D と同等(11.1%)— RQA は問いで
   再開放するが内容を明示せず、厳格 gpt-oss は加点しない。寛容 qwen では 18→40% に
   増加。κ=0.12(低)もこの審判不一致を反映。**測定6と整合**:reflection の寄与は
   frame-reopening であって alternative-stating ではない。

**限界(誇張回避)**: memory は統制的・フレーム由来(自然な蓄積記憶ではない);live
alternative は単一著者;n=18;E の審判不一致。本論文§14 の要件(自然な記憶での再現、
独立・複数アノテータ)は次段。

> **コンパニオン論文の実証核が揃った**:raw→governed memory の C-vs-D 回復が、
> 散発測定を「§8 Results の中心表」にした。keystone は審判頑健で、論文の
> architectural complement(§10)を数字で支える。

## 実証コンパニオン総括(本論文を支える実証核)

| 測定 | 論文対応 | 確定した実証 |
|---|---|---|
| M2 検索Echo | §A.6/§8 | ガバナンスは echo 量を減らさず、機構は標識+出力sanitization |
| M3 memory echo | §5 | 捏造自己引用は入力層別(宣言0%/自己言及25%)、Governorが残差0 |
| M4→M6 retention基準 | §6.3 | RQA優位は基準依存(STRICTで非優位、LENIENTで優位)、STRICTがκ信頼可 |
| M5 独立審判 | §14 | 方向はクロス系統頑健、per-alt κ は審判依存(不一致を測定) |
| **M7 A–E ladder** | **§8/§10** | **raw memory が plurality 崩壊(1/7)、governed が回復(keystone、κ頑健)** |

## 測定8・9(補強、patent向け)— 検証器頑健性 と model-independence

`experiments/graph_echo_gen.py`(larger-n)/ `experiments/model_independence.py`。
claim の2機構を個別に補強。

**M8 検証器頑健性(多言語・larger-n)**: 捏造誘発プローブを32件(JA/EN/ZH、複数run)に
拡張。捏造は誘発系で**~16%(5独立イベント)**、宣言系0%、検証器(D)が**全イベント
100%除去**(各runで surviving 0)。→ verifier claim が n=2 逸話から「複数言語・複数
イベント・決定論的除去」へ。

**M9 model-independence(別系統ベース)**: keystone(A/C/D)を llama3.1:8b(別系統・
アダプタとも両審判とも無縁)で再測。
- **narrowing(C<A)は両審判で再現**(harm はモデル一般)
- **recovery(D≥A)は instrumented model で両審判・第二model で primary審判のみ**。
  小型ゆえ絶対値圧縮(1–8%)
- → **harm はモデル一般、governance の efficacy はモデル能力依存**。方法は一般、
  効果量は最も capable な model で最強。誇張せず報告(誠実性)。

| assistant | A primary/indep | C | D |
|---|---|---|---|
| gemma-4-12B + RQA adapter | 9.7 / 15.3 | 1.4 / 2.8 | 11.1 / 18.1 |
| llama3.1:8b(別系統) | 4.2 / 8.3 | 2.8 / 5.6 | 4.2 / 1.4 |

→ patent: verifier(claim c)は決定論的・モデル非依存で最も頑健な効果。gate(claim b)の
recovery は方法一般だが効果量モデル依存 → claim 8 は「同記憶を gating 無しで注入した
条件に対する改善」と相対表現にする(固定値主張を避ける)。FIG.4。

## 成果物 / 次段

- `experiments/graph_echo.py`, `graph_echo_gen.py`, `probes_fabrication_prone.json`
- 結果: `experiments/graph_echo_results.json`, `graph_echo_gen_{declarative,fabrication_prone}.json`
- **次段(論文用)**: (1) n 拡大と層の体系化(本論文 §10.2 task families に対応)、
  (2) answer-space plurality / Diversity Index の C-vs-D 生成実験(§A.1/§A.2)、
  (3) 本論文 Condition A–F(§10.1)へのマッピング。
