# DPO(Phase 4-5)status — コーパス完成・パイプライン検証済・本学習は16GB境界で保留

日付: 2026-06-13

## 結論

**DPOコーパスと学習パイプラインは完成・動作検証済み。ただし v0.2 本学習は
ローカル16GB(RTX 5070 Ti)の境界に達したため、意図的に保留する。**
切り詰めて vacuous(空転)学習で v0.2 を出すことはしない(ゲートDの restraint
規律・退行を出さない方針に反するため)。

## できたもの

| 成果物 | 内容 |
|---|---|
| `data/dpo_corpus.jsonl` | **361 選好ペア**(`scripts/build_dpo_corpus.py`、新規LLM呼び出しなし) |
| `scripts/build_dpo_corpus.py` | 既存データからDPOコーパスを組成 |
| `scripts/train_dpo.py` | QLoRA-DPO(v0.1から継続、PEFT暗黙参照) |

コーパス内訳(§12.2 の selector-log ≤50% 遵守、実績8%):
- **fabrication_negative 290**: 実SFT出力(chosen)に「注入していないnodeを引用する
  記憶横断tension」を注入した負例(rejected)。**記憶捏造欠陥を直接狙う**合成負例。
  差分が feature_map 冒頭に出るため、系列を切り詰めても選好信号は保たれる
- **blind_adapter_win 43**: MMV-Lブラインド採点でアダプタが素Gemmaに勝った対
- **selector_log 28**: アダプタ自身のK候補内、評価器勝者 vs 敗者(スコア差≥2)

## OOM対処 = 特定対象のみ切り詰め(T方針 2026-06-13)

T方針「OOMは原因対象を特定しそれのみ切り詰める。全体は切り詰めない」を適用。
**安全な(内容・正しさを変えない)targetトリムを全て適用済み**:

1. `precompute_ref_log_probs=True` — 参照logprobを事前計算し、学習中の同時
   logit実体化を4→2に半減(特定対象=ref-pass、内容不変)
2. entropy メトリクス無効化 — trl 1.6 が全logitを reshape する**ログ専用**
   メトリクスを zeros 化(損失・勾配・学習結果に無影響)
3. gradient checkpointing 実効化 — `enable_input_require_grads()` +
   `use_reentrant=False`(PEFT併用時の無効化罠を回避、活性化を解放。重み不変)

→ これで backward は通過し、OOMは**単一の特定対象**に絞られた:
   **Gemma-4 の `logits = logits / final_logit_softcapping` による
   `[seq × 256,256語彙]` 全テンソル実体化(forward時、約1.46GB、空き約564MB)**。

**max_length の一律削減は方針違反なので採らない**(completionを削り信号を消す
=「全体切り詰め」)。スクリプト既定は全長1536のまま。

### 残る特定対象トリム(精密に特定済み・要検証のため本セッションでは未実施)

最後の対象 = softcapping全logit。正しいtargetトリム = **completion位置のみlogit
計算**(`logits_to_keep`)。これで loss は厳密に同一(prompt位置のlogpsは元々
mask=0で捨てられる)。**ただし注意**: trl DPO は `input_ids` を**右パディング**
(`dpo_trainer.py:191`)するため配列は `[prompt | completion | PAD]` で
completionは末尾ではない。素朴な「末尾K」はPADを拾いcompletionを外す=logps汚染。
**正しい窓 = K = seq_len − (バッチ内の最早completion開始位置)**(completion_mask
の最初の1の位置の最小値から算出)。窓内のPADは completion_mask=0 でmask、窓外の
prompt接頭辞は元々捨てられるので**右パディングでも厳密に正しい**。

実施手順(次回・GPU余裕がない時のみ必要):
1. `_compute_loss` を monkeypatch し、completion_mask から K を算出
2. `model(**kwargs, logits_to_keep=K)` で末尾K位置のlogitのみ計算
3. realign: 窓内の off-by-one を厳密に合わせる
4. **検証必須**: 両方収まる max_len で「stock全logit経路」と「本パッチ」の
   per-sequence logps が一致(tol 1e-4)することを確認してから信頼する
   (誤れば静かに v0.2 を汚すため、一致確認なしに採用しない)

→ クラウドA100ではこのトリム自体が不要(全長で収まる)。最も素直。

## 16GB境界(正確な記録)

パイプラインは動く(`--smoke --max-len 640` で4ステップ完走、ピーク13.89GB)。
しかし**完全忠実なDPOは収まらない**。理由:

```
SYSTEM_RQA = 853 トークン、chosen 中央値489 / p90 680 トークン
→ prompt+completion に中央値約1342・p90約1533 トークン必要
DPOは chosen+rejected の logprob を要し、Gemma-4 の logit softcapping が
[seq × 256,256語彙] 全体を実体化する(SFTの chunked_nll が使えない)
→ 約1024〜1152 トークン超で OOM。収まる長さでは completion がほぼ消え、
  logps/rewards が全てゼロ(=学習信号なし)
```

適用済みの緩和(いずれも正攻法、効果はあったが不足):
1. `precompute_ref_log_probs=True` — 参照logprobを事前計算し、学習中の同時
   logit実体化を4→2に半減(ref-pass通過)
2. entropy メトリクスの無効化 — trl 1.6 が全logitを reshape する**ログ専用**
   メトリクスを zeros 化(損失・勾配に無影響、学習結果は同一)。forward通過
3. max_len 掃引(1600→1280)— backward で約1〜1.5GB不足が残る

## 前進の道(本学習を実行する条件)

1. **クラウドGPU(A100 40/80GB)**: 無改造で余裕。コーパス・スクリプトは
   そのまま使える(最も素直)
2. **コンパクトDPOプロンプト**: 853トークンのSYSTEM_RQAを短縮した訓練用
   プロンプト。ただし訓練/推論のプロンプト不一致は品質上の妥協で、要評価
3. **chunked-logprob DPO**: completion部分のみ logit を計算する実装
   (`logits_to_keep` 相当)。transformers/trl 内部の改修が必要

## v0.2 を出す際の必須手順(退行防止)

本学習後、**必ずホールドアウト47件で v0.1 と再対比**し、ブラインド採点の
43勝2敗・多様性91.5%・restraint・検索較正が**退行していないこと**を確認して
から v0.2 を名乗る。DPOは少数・偏ったペアでモデルを壊し得るため、
fabrication_negative 偏重(80%)のコーパスが正当な記憶使用まで抑制して
いないか(memory_context あり事例での記憶横断tension発火率)も要確認。

## 現時点の妥当な判断

記憶捏造は **コード(`sanitize_memory_refs`)で100%抑制済み**でユーザーには
無害。よってDPOによる根治は「あれば良い」であって緊急ではない。本学習は
クラウドGPUが使える時、または上記2/3の準備ができた時に実施する。
