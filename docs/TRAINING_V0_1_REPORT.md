# RQA Adapter v0.1 学習報告(Phase 0/2 完了)

日付: 2026-06-13
ハード: RTX 5070 Ti 16GB(ローカル完結 — 学習・推論ともクラウド不使用)

## 環境(Phase 0)

- 専用venv `.venv-train`(**MMV共有のvenv313は不変** — 凍結環境保護)
- torch 2.11.0+cu128(Blackwell対応)/ transformers 5.11 / peft 0.19.1 / trl 1.6.0 / bitsandbytes 0.49.2
- ベース: `google/gemma-4-12B-it`(23.9GB、unified multimodal)— HF_HOME共有キャッシュへ取得

## 学習(Phase 2)

| 項目 | 値 |
|---|---|
| 手法 | QLoRA(4-bit NF4 + double quant、gradient checkpointing、completion-only loss) |
| LoRA | rank 16 / alpha 32、**言語タワーのみ328モジュール**(vision/audio不変 — update_scope規律) |
| データ | `train_materialized.jsonl` 410件(+val 16)、max_length 3072 |
| 実行 | 104ステップ(2 epoch)、約33分、ピークVRAM **15.06/16GB** |
| 損失 | train 1.08 / **eval 0.43(ep1)→ 0.40(ep2)**、eval token精度 91.1% |
| 成果物 | `models/rqa_adapter_v0_1/`(adapter 126MB、git管理外) |

**途中の障害と対処**: 初回はステップ10でOOM(256kボキャブラリのlogitsが
seq 3072で約1.6GB)。`loss_type="chunked_nll"` + `expandable_segments:True`
で解消し完走。

## 検証(ホールドアウト10件、ベース vs アダプタ、同一プロンプト・同条件)

| 指標 | ベース(素のGemma 4 12B) | **+RQA Adapter v0.1** |
|---|---|---|
| スキーマ解析成功率 | 100% | 100% |
| **多様性制約合格率** | 60% | **100%** |
| 有効候補数(平均) | 3.7本 | **4.4本** |
| 記憶横断tension使用 | 4/10 | 4/10(=memory付き事例数と一致、過不足なし) |
| search_needed発火 | 4/10 | 2/10(より抑制的) |

質的サンプル(同一入力「What objective are you meant to achieve?」への第1候補):

- ベース: 「システムの主機能が『構造抽出』なら『目的』の提示はそれを満たすのか…」(自己言及的で回りくどい)
- アダプタ: 「**他人に説明する用の建前の目的か、あなたのタスクで実際に示す挙動か — 両者は乖離し得るが、どちらを問うているのか?**」(鋭く、相手に向いている)

## 結論

- Phase −1で特定した**SFTの主目標「機械的信頼性」が達成**された:
  多様性制約合格率 60%→100%(タグ規律・標的分散・構え分散の体得)
- 記憶使用の較正が正確(memory文脈がある時だけ記憶横断tensionを出す)
- 完全ローカルで学習33分 — 反復実験が現実的な速度で回る

## 残作業(次フェーズ)

1. **Ollama統合**: adapter→GGUF変換(llama.cpp)で `rqa` CLIの実行系に接続
   (現状のCLIはプロンプト版のまま。HF推論経路では動作確認済み)
2. Phase 3(tool callデータ)/ Phase 4-5(DPO — selector-logが蓄積中)
3. **昇格ゲート**(仕様16.3): RQA単体評価の拡大 + RoutingEngine干渉評価
   (Condition I相当)+ Diversity Index 監視 + 人間承認 — v0.1はまだ
   sandbox段階であり、昇格判定は未実施
