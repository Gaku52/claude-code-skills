# Transitions（トランジション）

セクション間を滑らかに繋ぎます。4小節ルール・Filter・Volume・Sendで自然な流れを作ります。

## この章で学ぶこと

- 4小節ルール
- Filter変化
- Volume変化
- Send変化
- Fill とRiser
- 自然な流れの作り方
- セクション別トランジション

---

## なぜTransitionsが重要なのか

**自然な流れ:**

```
トランジションなし:

結果:
唐突
不自然
ぎこちない

トランジションあり:

結果:
滑らか
自然
プロフェッショナル

プロとアマの差:

アマ:
セクション突然変化
不自然

プロ:
4小節前から予告
自然な流れ

真実:

「プロの楽曲」=
完璧なトランジション

方法:
4小節ルール
Filter・Volume・Send変化
Fill・Riser

重要性:

良いトランジション:
聴き手気づかない
自然すぎて

悪いトランジション:
違和感
素人感

使用頻度:

100%
全セクション境界
```

---

## 4小節ルール

**最も重要な原則:**

### 基本概念

```
ルール:

セクション変化の
4小節前から
変化を予告

例:

Bar 61-64: Verse終わり
→ 変化開始

Bar 65: Buildup開始
→ 明確な変化

理由1: 予測可能性

人間の脳:
予測好き

4小節前:
変化予告
→ 期待

Bar 65:
予想通り変化
→ 満足

理由2: 自然さ

急な変化:
違和感

4小節かけて:
徐々に変化
自然

理由3: 音楽理論

4小節:
フレーズ単位
自然な区切り

8小節:
セクション単位

4小節 = 半セクション
変化に最適
```

### 適用箇所

```
全セクション境界:

Intro → Verse:
Bar 29-32で変化開始

Verse → Buildup:
Bar 61-64で変化開始

Buildup → Drop:
Bar 69-72で変化開始
(Buildup全体が変化)

Drop → Breakdown:
Bar 101-104で変化開始

Breakdown → Buildup:
Bar 117-120で変化開始

Drop → Outro:
Bar 141-144で変化開始

例外:

Drop瞬間:
予告なし
一瞬の沈黙

理由:
最大のインパクト
```

---

## Filter変化

**最も効果的:**

### Cutoff上昇

```
用途:

エネルギー上昇:
Verse → Buildup
Breakdown → Buildup

設定:

Auto Filter挿入:
該当トラック
(Pad・Lead)

Cutoff Automation:

Bar 61: 500 Hz (暗い)
Bar 62: 1000 Hz
Bar 63: 2000 Hz
Bar 64: 4000 Hz (明るく)
Bar 65: 8000 Hz (Buildup開始)

直線的:
徐々に上昇

Resonance:

Bar 61: 10%
Bar 64: 40%

同時に上昇:
効果的

効果:

明るくなる:
期待感
エネルギー上昇

聴き手:
変化察知
期待
```

### Cutoff下降

```
用途:

エネルギー下降:
Drop → Breakdown
Drop → Outro

設定:

Cutoff Automation:

Bar 101: 8000 Hz (明るい)
Bar 102: 4000 Hz
Bar 103: 2000 Hz
Bar 104: 1000 Hz (暗く)
Bar 105: 500 Hz (Breakdown開始)

効果:

暗くなる:
落ち着く
静寂へ
```

### Lowpass vs Highpass

```
Lowpass (推奨):

Cutoff上昇:
暗 → 明
効果的

用途:
Buildup
ほとんどの場合

Highpass:

Cutoff上昇:
低域削除

用途:
特殊効果
まれ

推奨:
Lowpass使用
```

---

## Volume変化

**シンプルで効果的:**

### Fade In

```
用途:

要素追加:
Intro → Verse
新しいトラック登場

設定:

Track Volume Automation:

Bar 29: -∞ dB (無音)
Bar 32: -12 dB (通常音量)

3-4小節かけて:
徐々に大きく

または:

Utility Gain:

Bar 29: -∞ dB
Bar 32: 0 dB

効果:

自然:
要素登場
違和感なし
```

### Fade Out

```
用途:

要素削除:
Drop → Breakdown
Drop → Outro

設定:

Track Volume Automation:

Bar 101: -12 dB (通常)
Bar 104: -∞ dB (無音)

3-4小節かけて:
徐々に小さく

効果:

自然:
要素退場
```

### Volume Swell

```
用途:

強調:
Buildup終盤
Drop直前

設定:

全トラック Volume:

Bar 63-64:
-12 dB → -6 dB (徐々に大きく)

Bar 64.4:
-6 dB → -∞ dB (瞬間的に消える)

Bar 65.1:
-6 dB (復帰)

効果:

最大のインパクト:
一瞬の沈黙
→ Drop爆発
```

---

## Send変化

**空間の変化:**

### Reverb Send上昇

```
用途:

Buildup:
広がり・浮遊感

設定:

Snare Send A (Reverb):

Bar 61: 25% (通常)
Bar 64: 60% (増加)
Bar 65: 25% (戻す)

または

Bar 65: 80% (さらに広げる)

効果:

広がり:
空間拡大
期待感

推奨トラック:

Snare: 最も効果的
Lead: 効果的
Vocal: 効果的
Kick・Bass: NG
```

### Reverb Send下降

```
用途:

Drop → Breakdown:
タイト → 広い

設定:

全トラック Send A:

Bar 101-104:
25% → 5% (下降)

Bar 105 (Breakdown):
Vocal Send: 50% (広げる)

効果:

対比:
タイト (Drop)
広い (Breakdown)
```

### Delay Send

```
用途:

Buildup:
リズミカル装飾

設定:

Lead Send C (Delay 1/8):

Bar 61: 0%
Bar 62: 10%
Bar 63: 20%
Bar 64: 35%
Bar 65: 15% (戻す)

効果:

リズム:
複雑
動き
```

---

## Fill（フィル）

**リズミカル装飾:**

### Snare Fill

```
用途:

セクション境界:
Bar 8 (最後)
Bar 16 (最後)
変化予告

パターン1: 4分音符

Bar 64:
拍 1: Snare
拍 2: Snare
拍 3: Snare
拍 4: Snare

シンプル:
効果的

パターン2: 8分音符

Bar 64:
1・1.5・2・2.5・3・3.5・4・4.5
全てSnare

より密:
効果的

パターン3: 16分音符 (Roll)

Bar 64:
16分音符全て
Snare Roll

最も密:
Buildup最適

設定:

Snare Track:

Bar 64のみ:
MIDI Note追加

または

Drum Rack:
別Snareサンプル
Roll専用

Velocity:

徐々に大きく:
64 → 100 → 127

効果:
加速感
```

### Tom Fill

```
用途:

Rock・Pop風:
ダイナミック

パターン:

Bar 64:
High Tom → Mid Tom → Low Tom
下降パターン

効果:
ダイナミック
展開
```

### Cymbal Fill

```
用途:

Crash:
セクション開始
強調

設定:

Bar 65.1 (Buildup開始):
Crash Cymbal

または

Bar 73.1 (Drop開始):
Crash Cymbal

効果:
強調
明確な変化
```

---

## Riser（ライザー）

**上昇効果音:**

### White Noise Riser

```
用途:

Buildup:
緊張感・期待感

作成方法:

Step 1: White Noise

Simpler:
White Noiseサンプル
または
Analog Noise

Step 2: Filter Automation

Auto Filter:
Cutoff上昇

Bar 61: 200 Hz
Bar 64: 8000 Hz

Resonance: 40%

Step 3: Volume Automation

Bar 61: -∞ dB
Bar 62: -24 dB
Bar 63: -18 dB
Bar 64: -12 dB
Bar 65: -∞ dB (消える)

徐々に大きく:
緊張感

Step 4: Reverb

Send A: 40%
広がり

効果:

最強のBuildup:
期待感最大
```

### Impact Riser

```
サンプル:

Splice・Loopcloud:
"Impact Riser"
"Tension Riser"

使用:

Bar 61-64:
配置

Volume:
徐々に大きく

Bar 65.1:
Impact消える
Drop開始

簡単:
サンプル使用推奨
```

### Pitch Riser

```
作成:

Lead・Bassトラック:

Transpose Automation:

Bar 61: 0 semitones
Bar 64: +12 semitones (1オクターブ上)

徐々に上昇:
期待感

Bar 65: 0 (戻す)

効果:
緊張感
```

---

## セクション別トランジション

**具体例:**

### Intro → Verse

```
Bar 29-32:

Filter:
Pad Cutoff上昇
500 → 2000 Hz

Volume:
Lead Fade In
-∞ → -12 dB

Send:
なし (まだ控えめ)

Fill:
なし (まだ早い)

Bar 33 (Verse開始):

Lead:
明確に登場

効果:
自然な移行
```

### Verse → Buildup

```
Bar 61-64 (最重要):

Filter:
Pad・Lead Cutoff上昇
300 → 8000 Hz

Resonance:
20% → 70%

Volume:
White Noise Fade In
-∞ → -12 dB

Send:
Snare Reverb上昇
25% → 60%

Fill:
Bar 63-64: Snare 8分音符
Bar 64: Snare Roll 16分音符

Bar 65 (Buildup開始):

明確な変化:
全て全開

効果:
最高の期待感
```

### Buildup → Drop

```
Bar 72 (Buildup最後):

拍 1-3:
Snare Roll 32分音符
最密

拍 4 (Bar 72.4):
全て停止 (0.25拍)
完全な沈黙

または:

Lowpass Filter 0 Hz
一瞬

Bar 73.1 (Drop開始):

全て復帰:
Kick・Bass・Lead
最大音量

Crash:
強調

効果:
最大のインパクト
```

### Drop → Breakdown

```
Bar 101-104:

Filter:
全トラック Cutoff下降
8000 → 500 Hz

Volume:
Kick・Bass Fade Out
-6 dB → -∞ dB

Send:
Reverb減少
30% → 5%

Bar 105 (Breakdown開始):

Vocal・Padのみ:
静寂

Reverb:
Vocal Send 50%
広い空間

効果:
劇的な対比
10/10 → 3/10
```

### Breakdown → Buildup 2

```
Bar 117-120:

Volume:
Percussion Fade In
-∞ → -12 dB

Hi-Hat:
復帰

Filter:
Pad Cutoff上昇開始
500 → 2000 Hz

Bar 121 (Buildup 2開始):

White Noise:
登場

Filter:
急上昇開始

効果:
再びエネルギー上昇
```

### Drop 2 → Outro

```
Bar 141-144:

Volume:
Lead・Vocal Fade Out
-6 dB → -∞ dB

Filter:
Pad Cutoff下降
8000 → 2000 Hz

Send:
Reverb減少
30% → 15%

Bar 145 (Outro開始):

Lead・Vocal:
完全に消える

残る:
Kick・Bass・Percussion

効果:
徐々に終了へ
```

---

## 一瞬の沈黙

**最大のインパクト:**

### Drop直前

```
タイミング:

Bar 72.4 (拍4):
0.25拍
完全な沈黙

方法1: 全トラック Mute

Automation:
Track Activator Off

Bar 72.4: Off
Bar 73.1: On

方法2: Master Utility

Utility (Master):

Gain Automation:
Bar 72.4: -∞ dB
Bar 73.1: 0 dB

方法3: Lowpass Filter

Master Track:
Auto Filter

Cutoff Automation:
Bar 72.4: 0 Hz (全て遮断)
Bar 73.1: 21000 Hz (全開)

推奨:
方法3 (Lowpass)
最も自然

効果:

最大のインパクト:
沈黙 → 爆発
劇的

重要:

長すぎNG:
0.25拍のみ
0.5拍 = 長すぎ
```

---

## トランジションのタイミング

**精密さが鍵:**

### Grid設定

```
Ableton:

Grid: 1/16
必須

理由:
精密な配置
16分音符単位

トランジション:

開始: Bar X.1
終了: Bar Y.1

必ずGrid上:
ずれNG
```

### Automation精度

```
描画:

Draw Mode: On (B)

ブレークポイント:

必ずGrid上:
拍の頭
拍の裏

NG:

中途半端な位置:
Bar 64.3.2
不自然

OK:

Bar 64.1 (拍1)
Bar 64.3 (拍3)
明確
```

---

## よくある失敗

### 1. 4小節ルール無視

```
問題:
Bar 64で突然変化
予告なし
唐突

解決:

Bar 61-64:
4小節かけて変化

効果:
自然
```

### 2. Filter変化なし

```
問題:
Volume変化のみ
単調

解決:

Filter:
必ず使用
Cutoff・Resonance

効果:
ダイナミック
```

### 3. Fill多すぎ

```
問題:
全セクション境界Fill
うるさい

解決:

Fillは厳選:
重要な境界のみ
Verse → Buildup
Buildup → Drop

頻度:
2-3回/曲
```

### 4. 一瞬の沈黙長すぎ

```
問題:
1拍沈黙
長すぎ
違和感

解決:

0.25拍のみ:
16分音符1つ分

測定:
Grid 1/16確認
```

---

## Ableton Liveでの実践

**効率的な作業:**

### Automationワークフロー

```
Step 1: Show Automation

A: Automation表示

Step 2: Device選択

Device Chooser:
Auto Filter選択

Step 3: Parameter選択

Parameter Chooser:
Frequency (Cutoff)選択

Step 4: 描画

Draw Mode: On (B)

Bar 61-64:
直線描画
上昇

Step 5: 確認

再生:
聴いて確認

調整:
ブレークポイント移動
```

### Macro Knob活用

```
効率化:

Audio Effect Rack:

Macro 1 "Transition":

Map:
- Filter Cutoff
- Filter Resonance
- Reverb Send

1つのAutomation:
3つ同時変化

Bar 61: 0%
Bar 64: 100%

簡単:
推奨
```

---

## トランジション分析

**Reference学習:**

### 5曲分析

```
Step 1: 曲選択

好きな曲 5曲:
Techno・House

Step 2: セクション境界特定

Locator:
各セクション境界

Step 3: トランジション確認

Bar X-4 〜 Bar X:
何が起きている？

Filter?
Volume?
Send?
Fill?

Step 4: メモ

表作成:

曲 | 境界 | Filter | Volume | Send | Fill
----|------|--------|--------|------|-----

Step 5: 共通点発見

全曲:
4小節ルール？
Filter上昇？

Step 6: 応用

自分の曲:
同じ技法使用
```

---

## まとめ

### Transitions

```
□ 4小節ルール厳守
□ Filter Cutoff変化必須
□ Volume Fade In/Out
□ Send変化 (Reverb・Delay)
□ Fill厳選使用
□ Riser (White Noise)
```

### 重要タイミング

```
Verse → Buildup: Bar 61-64最重要
Buildup → Drop: Bar 72.4一瞬の沈黙
Drop → Breakdown: Bar 101-104
```

### 重要原則

```
□ 4小節前から予告
□ Grid 1/16厳守
□ Automation精密
□ 一瞬の沈黙 0.25拍のみ
□ Reference分析5曲
```

---

**次は:** [Arrangement Techniques](./arrangement-techniques.md) - アレンジメント技法の完全ガイド
