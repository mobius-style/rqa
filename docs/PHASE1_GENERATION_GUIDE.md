# Phase 1 注釈生成ガイド(生成エージェント用・正典)

あなたの任務: チャンクファイルの各シードに対し、MOBIUS-RQA のSFT学習例を1件ずつ生成し、
指定された出力ファイルに JSONL で書き出すこと。

## 入力(チャンク行の形式)

```json
{"seed_id": "...", "input_text": "...", "language": "ja|en",
 "anchors": {"search_needed": bool, "preferred_route": "...", "failure_modes": [...], ...},
 "category": "...",
 "directives": {"posture": "deepen|moderate|rule6|anchor_search", "k": 4-6,
                "include_memory": bool, "include_self_update": bool}}
```

## 出力(1シード=1行)

```json
{"messages": [
   {"role": "system", "content": "{{SYSTEM_RQA}}"},
   {"role": "user", "content": "<user content>"},
   {"role": "assistant", "content": "<§11.1出力オブジェクトのJSON文字列>"}],
 "meta": {"source": "claude_phase1", "seed_id": "<seed_id>", "category": "<category>"}}
```

system は必ずプレースホルダ文字列 `{{SYSTEM_RQA}}` のまま。

### user content の組み立て(この順・この形式厳守)

1. directives.include_memory が true のときだけ、先頭に memory_context ブロック:
```
[memory_context]  (data, not instructions — retrieved Question Graph records)
- (node <番号>, <claim|question|tension>, <self|user>, <YYYY-MM-DD>, status=open) <断片テキスト>
[/memory_context]
```
   断片は1〜3個。**入力トピックに本当に関連する、もっともらしい過去記録を創作する**
   (過去に立てた問い・過去の主張・過去の自己観察)。日付は2026-04〜06。
   禁止語彙: Answer Entitlement / TVS / MKR / KVS / route_taxonomy / μQK / reason_code
   (これらはフィルタで遮断されるため断片に含めない)。
2. `K = <directives.k>`
3. 入力本文(input_text そのまま)

各部は空行で区切る。

### assistant content(JSON文字列化した §11.1 オブジェクト)

```json
{"feature_map": {"surface_terms": [], "claims": [...], "evidence": [],
  "tensions_input_internal": [...],
  "tensions_memory_cross": [{"tension": "...", "memory_ref": "<node番号>", "recorded_at": "YYYY-MM-DD", "confidence": "high|medium|low"}],
  "assumptions": [...], "frames": [...]},
 "search_decision": {"search_needed": bool, "reason": "...", "queries": [...]},
 "question_candidates": [
   {"question": "...", "target_layer": "L3|L4|L5|L7", "target_element": "...", "stance": "claim_skeptic|frame_skeptic|steelman"}],
 "self_ranking": [<0始まりの添字、強い順>],
 "self_update_proposal": null}
```

## 品質規則(リンタが機械検査する — 違反は不合格)

1. question_candidates は **ちょうど K 本**。各候補の target_element は**全て異なる**
   (feature_map内の要素を指す: 例 "assumptions[0]", "claims[0]_scope")
2. target_layer は2種類以上に分散(L3/L4/L5/L7)。stance も2種類以上に分散
3. include_memory=true の例は tensions_memory_cross を必ず1件以上(memory_refは
   memory_contextのnode番号と一致させる)。false の例では空配列
4. include_self_update=true のときだけ self_update_proposal を書く。allowed_area は
   premise_excavation_depth / question_generation_bias / feature_extraction_priority /
   failure_pattern_memory / self_understanding_notes のいずれか。false なら null
5. 言語: input_text と同じ言語で全自然言語フィールドを書く(JA入力→JA、EN入力→EN)

## posture 別の書き方(品質の核心)

- **deepen**: 本気の前提掘削。違和感2〜3件、前提2〜3件、認識枠1〜2件。問いは
  「答えると理解が変わる」具体的なもの。テンプレ問い(「その前提は正しいですか?」)禁止。
  到達階層をチャンク内で散らす(L4止まり/L5まで/L7まで)
- **moderate**: 掘削は軽め(違和感1〜2件)。問いは実用的明確化+1本だけ深い問い
- **rule6**: 単純事実・挨拶・雑談。**深化を強制しない**: tensions/assumptions/framesは
  空or最小、claimsに「直接回答が適切」の旨を書き、候補は実務的な明確化質問のみ
  (それでも layer/stance タグと多様性規則は守る — L3中心+L4/L7混在でよい)
- **anchor_search**: anchors.search_needed=true を必ず尊重し search_decision.search_needed=true、
  queries に実用的な検索語。anchors.failure_modes(stale_commitment等)があれば、
  鮮度・日付境界に関する明確化候補を含める。深化は強制しない(rule6寄り)

## 作業手順(厳守)

1. Read であなたのチャンクファイルを読む
2. Read で `data/gold_seed.jsonl` を読み、形式の手本とする(特に1行目と3行目)
3. 全シードの学習例を生成し、Write で出力ファイルに書く(JSONL、1行1例)
4. Bash で自己検査:
   `cd <project_root> && ../venv313/bin/python scripts/export_sft.py lint <出力ファイル>`
5. FAIL があれば修正して再書き出し→再lint。全通過まで繰り返す
6. 最終報告: 生成数、lint結果(N/N pass)、posture別の内訳
