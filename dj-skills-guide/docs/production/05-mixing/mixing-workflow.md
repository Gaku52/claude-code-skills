# Mixing Workflow

完全なミキシング手順の集大成。10フェーズワークフロー・チェックリスト・時間配分を完全マスターします。

## この章で学ぶこと

- 完全な10フェーズワークフロー
- 各フェーズ時間配分
- チェックリスト
- トラブルシューティング
- Break推奨タイミング
- 完成判断基準
- プロのTips

---

## なぜWorkflowが重要なのか

**効率と品質:**

```
Workflowなし:

問題:
行き当たりばったり
何度も戻る
時間無駄

結果:
8時間+ かかる
疲労
品質低下

Workflowあり:

メリット:
計画的
一方通行
効率的

結果:
4時間で完成
疲労少ない
品質高い

プロの真実:

Workflow:
確立している
毎回同じ

理由:
効率最大化
品質安定

使用頻度:
100%
全プロジェクト
```

---

## 完全10フェーズWorkflow

**4時間で完成:**

### Phase 0: 準備 (10分)

```
タスク:

1. プロジェクト保存
   名前: "Track Name - Mix v1"

2. Backup作成
   Time Machine等

3. 不要トラック削除:
   使ってない音色
   古いテイク

4. グループ化:
   Drums
   Music (Lead・Pad)
   FX

5. Color分け:
   Drums: 赤
   Music: 青
   FX: 緑

6. トラック名整理:
   "Kick"
   "Bass"
   明確に

7. Reference Track配置

8. LUFS Meter挿入

9. Spectrum挿入

10. 休憩・水分

完成条件:
整理整頓完了
準備OK
```

### Phase 1: Gain Staging (20分)

```
タスク:

1. 全トラック再生
   Master何dB?

2. 計算:
   目標 -6 dB
   差分計算

3. 全トラック Utility:
   Gain調整
   または
   Fader調整

4. Kick Solo:
   Master -6 dB調整

5. Bass追加:
   バランス

6. Drums追加:
   Snare・Hi-Hat

7. Music追加:
   Lead・Pad

8. Vocal追加 (あれば)

9. 全体再生:
   Master -6 dB確認

10. 全セクション確認:
    Intro・Verse・Drop

完成条件:
Master -6 dB (全セクション)
各トラック相対バランス良い

休憩: 5分
```

### Phase 2: Low End (30分)

```
タスク:

1. Kick EQ:
   High Pass 30 Hz
   Low Shelf +3 dB @ 60 Hz
   Peak -3 dB @ 250 Hz
   High Cut 10 kHz

2. Kick Compressor:
   Threshold -12 dB
   Ratio 4:1
   Attack 10 ms
   Release 80 ms

3. Bass EQ:
   High Pass 40 Hz
   Low Shelf +2 dB @ 80 Hz
   Peak -2 dB @ 250 Hz
   High Cut 5 kHz

4. Bass Compressor:
   Threshold -15 dB
   Ratio 6:1
   Attack 30 ms
   Release 100 ms

5. Sidechain:
   Kick → Bass
   Ratio 8:1
   Attack 0.1 ms
   Release 100 ms

6. 全トラック High Pass:
   Snare 200 Hz
   Lead 200 Hz
   Pad 300 Hz
   Vocal 80 Hz
   等

7. Spectrum確認:
   20-120 Hz
   Kick・Bass明確？

8. Mono確認:
   全トラック Bass Mono: On (120 Hz)

9. A/B比較:
   High Pass前後

10. Reference比較:
    低域同じくらい？

完成条件:
低域クリア
Kick・Bass分離
全トラック High Pass済み

休憩: 5分
```

### Phase 3: Mid Range (30分)

```
タスク:

1. Spectrum確認:
   250-500 Hz盛り上がりすぎ？

2. 全トラック処理:
   Peak -2〜-4 dB @ 300-500 Hz

   Snare: -2 dB @ 400 Hz
   Lead: -3 dB @ 300 Hz
   Vocal: -3 dB @ 300 Hz
   Pad: -4 dB @ 500 Hz

3. Lead・Vocal明瞭度:
   Peak +3 dB @ 3 kHz
   Q: 1.5

4. Snare スナップ:
   Peak +2 dB @ 3 kHz

5. De-ess (Vocal):
   Peak -4 dB @ 7 kHz
   Q: 3.0

6. Spectrum再確認:
   250-500 Hz スッキリ？

7. Reference比較:
   Mid同じバランス？

8. Solo確認:
   各トラック聴こえる？

9. A/B比較:
   EQ前後

10. Master確認:
    -6 dB維持？

完成条件:
Low-Mid クリア
明瞭度向上
分離良い

休憩: 10分
```

### Phase 4: High End (20分)

```
タスク:

1. Hi-Hat EQ:
   High Pass 6000 Hz

2. 全体 High Shelf:
   Master EQ Eight
   +1 dB @ 10 kHz
   Q: 1.0

3. Spectrum確認:
   10-20 kHz 適度？

4. De-ess再確認:
   Vocal刺さらない？

5. Reference比較:
   高域同じくらい？

6. 明るさ調整:
   必要なら+0.5 dB

7. 刺さりチェック:
   長時間聴いて疲れない？

8. A/B比較

9. 複数デバイス確認:
   ヘッドホン・スピーカー

10. Master確認

完成条件:
高域適度
刺さらない
空気感ある

休憩なし
```

### Phase 5: Stereo Image (20分)

```
タスク:

1. 全トラック Bass Mono:
   Utility挿入
   Bass Mono: On
   Freq: 120 Hz

2. Width設定:
   Kick: 0%
   Bass: 0%
   Snare: 0-10%
   Lead: 20-30%
   Pad: 80-100%
   FX: 100-120%

3. Panning:
   Hi-Hat: L/R
   Percussion: L/R
   FX: L/R極端

4. Master Width:
   Utility
   Width: 105-110%
   Bass Mono: On (120 Hz)

5. Mono確認:
   Width 0%で再生
   問題なし？

6. Goniometer確認:
   やや縦長楕円？

7. Reference比較:
   ステレオ幅同じ？

8. バランス確認:
   Center 60%, Wide 40%

9. 位相チェック:
   Correlation Meter

10. 最終確認

完成条件:
120 Hz以下Mono
ステレオ広い
Mono互換性OK

休憩: 10分 (重要)
```

### Phase 6: Depth & Space (30分)

```
タスク:

1. Return A作成:
   Reverb (Hall)
   Decay: 2.5 s
   Size: 70%
   Pre-Delay: 25 ms

2. Return A Post EQ:
   High Pass: 400 Hz
   High Cut: 10 kHz

3. Return B作成:
   Reverb (Room)
   Decay: 1.0 s
   Pre-Delay: 15 ms
   High Pass: 400 Hz

4. Return C作成:
   Delay (1/8)
   Feedback: 40%
   Ping Pong: On
   High Pass: 500 Hz
   Low Pass: 6 kHz

5. Return D作成:
   Delay (1/4 Dotted)

6. Send量調整:
   Kick: 0%
   Bass: 0%
   Snare: 30% (A)
   Vocal: 25% (A), 20% (C)
   Lead: 25% (A), 25% (C)
   Pad: 50% (A)
   Hi-Hat: 10% (B)

7. Pre-Delay確認:
   全Return設定済み？

8. 低域濁りチェック:
   Return EQ High Pass OK？

9. Reference比較:
   奥行き同じ？

10. バランス最終調整

完成条件:
奥行き明確
低域クリア
空間適切

休憩: 5分
```

### Phase 7: Dynamics (30分)

```
タスク:

1. Kick・Bass:
   Compressor設定済み (Phase 2)

2. Vocal Compressor 1:
   Threshold: -18 dB
   Ratio: 3:1
   Attack: 5 ms
   Release: 50 ms

3. Vocal Compressor 2:
   Threshold: -12 dB
   Ratio: 2:1

4. Drum Bus作成:
   Kick・Snare・Hi-Hat
   → Group

5. Drum Bus処理:
   Glue Compressor
   Threshold: -15 dB
   Ratio: 2:1
   Attack: 10 ms
   Release: Auto
   GR: -3 dB

6. Music Bus作成:
   Lead・Pad
   → Group

7. Music Bus処理:
   Compressor
   Ratio: 2:1
   GR: -2 dB

8. 各Compressor確認:
   GR -3〜-6 dB？

9. ダイナミクス維持:
   呼吸ある？

10. Reference比較

完成条件:
安定
パンチ維持
過剰圧縮なし

休憩: 10分
```

### Phase 8: Automation (30分)

```
タスク:

1. Volume Automation:
   Intro Fade
   Breakdown削除
   Drop復帰

2. Filter Automation:
   Pad Auto Filter
   Buildup: 300 Hz → 8000 Hz
   Resonance: 20% → 70%

3. Send Automation:
   Buildup Reverb増加
   Drop戻す

4. FX Automation:
   Riser Volume
   White Noise

5. Pan Automation:
   FX動き

6. Automation確認:
   全て描画OK？

7. Buildup効果確認:
   劇的？

8. Drop効果確認:
   インパクト？

9. 各セクション再生

10. Reference比較:
    展開同じくらい劇的？

完成条件:
Buildup完璧
Drop劇的
展開自然

休憩なし
```

### Phase 9: Reference & Final (30分)

```
タスク:

1. Reference Track:
   Volume Matching

2. LUFS比較:
   目標 -18 LUFS (Mix)
   Reference -14 LUFS

3. Spectrum比較:
   全帯域バランス

4. A/B切り替え:
   10回以上

5. 問題特定:
   何が違う？

6. 最終調整:
   特定した問題修正

7. 全セクション再生:
   Intro・Verse・Buildup・Drop・Outro

8. Mono確認:
   再度

9. 複数デバイス:
   ヘッドホン・スピーカー・車

10. 休憩後再確認:
    耳リセット

完成条件:
Referenceレベル
全セクションOK
複数デバイスOK

休憩: 15分 (重要)
```

### Phase 10: 書き出し準備 (10分)

```
タスク:

1. Master確認:
   -6 dB以上ヘッドルーム

2. クリッピングチェック:
   全セクション

3. Reference Track:
   Mute (忘れずに!)

4. 不要トラック:
   Mute

5. プロジェクト保存:
   "Track Name - Mix FINAL"

6. Freeze:
   CPU重いトラック

7. Bounce設定確認:
   WAV, 24-bit, 44.1 kHz

8. 書き出し範囲:
   正確？

9. Normalize: Off

10. 最終確認

完成:
Mix完了
マスタリングへ

合計時間: 約4時間
```

---

## チェックリスト

**各フェーズ完了確認:**

### 開始前

```
□ プロジェクト Backup
□ 不要トラック削除
□ グループ化・整理
□ Reference配置
□ LUFS・Spectrum挿入
```

### Phase 1-2: 低域

```
□ Master -6 dB
□ Kick・Bass EQ・Comp
□ Sidechain設定
□ 全トラック High Pass
□ 120 Hz以下Mono
```

### Phase 3-4: Mid・High

```
□ Low-Mid -2〜-4 dB (全トラック)
□ 明瞭度 +3 dB @ 3 kHz
□ De-ess (Vocal)
□ High Shelf +1 dB @ 10 kHz
□ Spectrum バランス良い
```

### Phase 5: Stereo

```
□ 全トラック Bass Mono: On
□ Width設定 (Kick 0%, Pad 100%)
□ Panning (Hi-Hat L/R)
□ Mono確認OK
□ Correlation +0.4-0.7
```

### Phase 6: Depth

```
□ Return Track 4つ
□ Return EQ High Pass 400 Hz
□ Pre-Delay 20-30 ms
□ Send量適切
□ 低域濁りなし
```

### Phase 7: Dynamics

```
□ Kick・Bass・Vocal Comp
□ Drum Bus Comp (GR -3 dB)
□ Music Bus Comp (GR -2 dB)
□ ダイナミクス維持
□ 過剰圧縮なし
```

### Phase 8: Automation

```
□ Buildup Filter Automation
□ Volume Automation
□ Send Automation
□ Drop劇的
□ 展開自然
```

### Phase 9: Reference

```
□ Volume Matching
□ LUFS比較
□ Spectrum比較
□ A/B切り替え 10回+
□ 問題修正済み
```

### 最終確認

```
□ Master -6 dB以上
□ クリッピングなし
□ Mono互換性OK
□ 複数デバイス確認
□ Reference Track Mute
□ プロジェクト保存
```

---

## Break推奨タイミング

**耳の疲労防止:**

```
Phase 2完了後: 5分
Phase 3完了後: 10分 (重要)
Phase 5完了後: 10分 (重要)
Phase 7完了後: 10分
Phase 9完了後: 15分 (最重要)

理由:

耳疲労:
1時間で判断力低下

Break効果:
リフレッシュ
客観性回復

推奨:

Phase 9後:
30分休憩
耳完全リセット
最終判断

合計Break時間: 55分
作業時間: 4時間
総時間: 約5時間
```

---

## トラブルシューティング

**よくある問題:**

### 問題1: Master 0 dB超え

```
原因:
ヘッドルーム不足

解決:

全トラック Utility:
Gain -3 dB

または:

Master Utility:
Gain -3 dB

確認:
Master -6 dB
```

### 問題2: 濁る

```
原因:
Low-Mid処理不足

解決:

Phase 3戻る:
全トラック -2〜-4 dB @ 300-500 Hz

確認:
Spectrum
```

### 問題3: 音が遠い

```
原因:
Reverb過剰

解決:

Send量:
全て半分に

確認:
近くなった？
```

### 問題4: Mono で消える

```
原因:
120 Hz以下Stereo

解決:

全トラック:
Bass Mono: On (120 Hz)

確認:
Mono再生
```

---

## プロのTips

**効率化:**

```
Tip 1: Template使用

保存:
"Mix Template"

含む:
- Return Track 4つ
- LUFS Meter
- Spectrum
- Utility (全トラック)

メリット:
30分短縮

Tip 2: Macro作成

Buildup Macro:
- Filter Cutoff
- Resonance
- Reverb Send

1つのAutomation:
3つ同時制御

Tip 3: Group活用

Drums Group:
一括処理

Music Group:
一括処理

メリット:
効率的
CPU軽い

Tip 4: Snapshot

各Phase完了後:
プロジェクト保存
"Mix v1-1", "Mix v1-2"

理由:
戻れる
安心

Tip 5: 翌日確認

完成後:
一晩置く

翌日:
新鮮な耳で確認

効果:
客観的
問題発見
```

---

## 完成判断基準

**いつ完成？**

```
基準:

1. Reference比較:
   同レベル達成

2. チェックリスト:
   全て✓

3. 複数デバイス:
   全て良い音

4. 時間:
   4-5時間経過
   (これ以上は逆効果)

5. 直感:
   「良い」と感じる

6. 翌日確認:
   問題なし

完成:
6つ全て満たす

未完成:
1つでも問題

対処:

小さい問題:
次回修正

大きい問題:
今日修正
ただし1時間以内

ルール:

完璧主義:
NG
60-70点で次へ

理由:
経験値
次作で向上
```

---

## まとめ

### 10フェーズWorkflow

```
Phase 0: 準備 (10分)
Phase 1: Gain Staging (20分)
Phase 2: Low End (30分)
Phase 3: Mid Range (30分)
Phase 4: High End (20分)
Phase 5: Stereo (20分)
Phase 6: Depth (30分)
Phase 7: Dynamics (30分)
Phase 8: Automation (30分)
Phase 9: Reference (30分)
Phase 10: 書き出し (10分)

合計: 4時間 + Break 55分 = 約5時間
```

### 重要原則

```
□ 計画的に進める
□ 各Phase完成後次へ
□ 戻らない (原則)
□ Break取る (必須)
□ Reference比較 (頻繁)
□ チェックリスト使用
□ 完璧主義避ける
```

### 次のステップ

```
Mix完成後:
- 一晩置く
- 翌日確認
- マスタリングへ
```

---

**Mixing完全マスター！次はproduction/06-arrangement（楽曲構成）へ進みましょう。**
