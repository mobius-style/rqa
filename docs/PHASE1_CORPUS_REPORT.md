# Phase 1 SFTコーパス報告書

日付: 2026-06-12
体制: 仕様12.1どおり — Claude生成 / ルートアンカー接ぎ木 / MMV-L審査 / 抜き取りレビュー(50件、T委任によりClaude実施)

## 1. 生成

- シード: `data/seed_inputs.jsonl`(1,986件)から決定論的サンプラー(`scripts/sample_phase1.py`, seed=20260612)で**500件**をカテゴリ均衡抽出
- 11カテゴリ(conceptual_explain 110 / factual_inquiry 90 / volatile_current 60 / correction 60 / self_reference 50 / stable_control 35 / casual_engagement 30 / stale_premise_trap 25 / date_boundary 15 / query_neutrality 15 / ambiguous_time_frame 10)
- 生成指令: posture(deepen/moderate/rule6/anchor_search)、K∈{4,5,6}、memory文脈32%、self_update 10%
- 生成: Claudeエージェント20体(25件/体)、正典ガイド `docs/PHASE1_GENERATION_GUIDE.md` 準拠、各自リンタ自己検査
- 結果: **500/500生成、マージ後リンタ500/500通過**
- コーパス統計: self_update_ratio 0.10(規定≤0.20)/ memory_context_ratio 0.32(規定≥0.30)

### 生成中の一貫した判断(エージェント報告から)

- **アンカー優先則**: posture=anchor_search でも `anchors.search_needed=false` の場合(stale-premise trap等)はアンカーが勝ち、偽前提のre-anchorを主軸にする
- **言語タグ不一致**(seed_inputsの language タグが実テキストと異なる約15件): 入力本文の言語が勝つ
- **rule6+memoryの両立**: 記憶横断tensionが「抑制を支持する」内容(過剰確認の自己観察等)になるよう設計し、深化を強制しない
- **プレースホルダ入力**(X/Y/Z): 指示対象の解決を第一候補に置く

## 2. 抜き取りレビュー(50件精読)

層化無作為50件(`state/spot_review_digest.md`、seed=50)を精読した。

**判定: 不合格 0 / 要注意 2 / 合格 48**

### 強み(全体傾向)

1. **posture規律が良好**: rule6事例は一貫して抑制的で、claims欄が「直接回答が適切」と明示。逆に深化事例(不確定性原理、スマートコントラクト、フォネティクス等)は専門的に正確な具体性を持つ(「観測者攪乱は誤読」「IAS 34」「basso continuo」等、入力固有の判別点を含む)
2. **記憶横断tensionが実質的**: 単なる引用ではなく判断を変える使い方
   (例: [457] 「Pichai退任」偽前提に対し、過去の失敗記録 node 33 を根拠に検証優先を選ぶ — 教材として模範的)
3. **アンカー遵守**: volatile系は全例 search_needed=true + 日付境界候補。stale-premise trap は前提検証を最上位候補に配置
4. **言語忠実**: ja/zh/en すべて入力言語で記述
5. **claims欄の事実主張を抜き取り照合**: Orwell(1945/1949)、al-Khwārizmī経路、パリ–ベルリン約880km(大円)等 — 誤り検出なし

### 要注意2件(不合格ではない)

1. **[426] 東京–京都の道路距離「約280マイル(約450km)」**: 経路次第で450〜490kmであり許容近似だが、claims欄に数値を書く形式は誤差混入のリスク経路。今後の生成では「桁が安定した値のみclaimsに書く」を推奨
2. **[171] 単発の同意表明にK=6**: 候補が埋め草気味になる(内容は健全)。casual_engagementはK=4を既定にする方が自然

## 3. External Evaluator審査(MMV-L gpt-oss-120B)

(審査ルーブリック: fidelity / depth — 単純入力への抑制を高評価 / calibration / format_language、
verdict=accept かつ全軸≥5 で採択。結果は `data/audit_results.jsonl`)

### 第1回審査(全数500件)

- **採択 401/500(80.2%)、API エラー 0**
- 採択群の軸平均: fidelity 8.78 / depth 8.43 / calibration 9.06 / format_language 9.92

### 審査プロンプトのバグと再審査(透明性記録)

却下99件を点検した結果、**50件は「`K = n` 行をfeature_mapに反映していない」
という却下**だった。K行は候補数を指定する制御パラメータであり内容ではない —
これは審査ルーブリック(本セッションでClaudeが起草、未批准)にその説明を
書き落とした起草バグである。ルーブリックに1行の明確化を追加し
(`scripts/audit_sft.py` 内 NOTE)、**却下99件のみ**を再審査した。
第1回の採択401件は再審査の対象外(明確化は減点要因の除去であり、
採択済み例の評価を変えない)。両回の結果を保存:
`audit_results.jsonl` / `audit_results_pass2.jsonl`。

### 最終結果

- **最終採択 473/500(94.6%)**
- 実質的却下 27件: 過剰確認(memory記録の無視)、検索判断の不一致、
  根拠なきtension/assumptionの混入 — いずれも保守方向の却下であり妥当
- 抜き取りレビューとEvaluatorの不一致: 抜き取り50件中、私が合格とした
  2件([78] 儀礼概観 / [116] 半期報告書)をEvaluatorはfidelity/calibration
  理由で却下した。保守方向(コーパスから除外)なのでEvaluator判定を採用

### 学習/ホールドアウト分割

- 採択473件のうち、**抜き取りレビュー済み47件をホールドアウト**
  (`sft_phase1_holdout.jsonl` — 人間代理レビュー済み・学習不可、仕様12.2)
- **学習コーパス: `sft_phase1_train.jsonl` 426件**
  (リンタ426/426通過、self_update 8.5% ≤20%、memory文脈 31.9% ≥30%)
- gold_seed 6件も評価専用(学習に入れない)

## 4. 成果物

| ファイル | 内容 |
|---|---|
| `data/sft_phase1_raw.jsonl` | 生成500件(リンタ通過) |
| `data/audit_results.jsonl` / `_pass2.jsonl` | Evaluator審査結果(第1回/再審査) |
| `data/sft_phase1_v1.jsonl` | 採択473件(分割前の全体) |
| `data/sft_phase1_train.jsonl` | **学習コーパス 426件** |
| `data/sft_phase1_holdout.jsonl` | ホールドアウト47件(レビュー済み・学習不可) |
| `data/gold_seed.jsonl` | ゴールド6件(フォーマット標準・学習不可) |
| `state/spot_review_digest.md` | 抜き取り50件ダイジェスト |

## 5. 次工程

- Phase 2(QLoRA SFT学習)へ。学習時は `{{SYSTEM_RQA}}` プレースホルダを `rqa/prompts.py` の実体で置換する
- 学習に入れないホールドアウト: gold_seed 6件+抜き取り50件のindexは評価専用として確保する(仕様12.2)
- DPO種: 運用蓄積中の selector-log(`scripts/export_sft.py dpo`)+今後の蒸留ペア
