# Creative Effects

Filter・LFO・特殊効果で個性を出します。実験的エフェクトを完全マスターし、他とは違う音を作ります。

## この章で学ぶこと

- Auto Filter（自動Cutoff変化）
- Vocoder（ボーカル加工）
- Resonators（共鳴フィルター）
- Corpus（物理モデリング）
- Grain Delay（粒状ディレイ）
- Frequency Shifter（周波数シフト）
- Beat Repeat（リピート効果）

---

## なぜCreative Effectsが重要なのか

**個性と差別化:**

```
標準的エフェクト:

EQ・Comp・Reverb:
必須
誰でも使う

Creative Effects:

Auto Filter・Vocoder等:
個性
差別化

使用頻度:

Auto Filter: 60%
Techno/House必須

Vocoder: 10%
特定用途

Resonators: 15%
実験的

結果:

標準エフェクト:
基礎

Creative:
個性・プロっぽさ

真実:
「独自の音」=
Creative Effectsの使い方
```

---

## Auto Filter 完全ガイド

**最重要Creative Effect:**

### 基本原理

```
機能:

Filter:
Cutoff・Resonance

LFO:
自動的にCutoff変化

Envelope Follower:
Input信号でCutoff変化

結果:
動的フィルター
Techno/House必須

用途:

Bass: グルーヴ
Lead: 動き
Drums: 質感
```

### インターフェイス

```
┌─────────────────────────────────┐
│  Auto Filter                    │
├─────────────────────────────────┤
│  Filter Type: [Lowpass ▼]       │
│  Cutoff: 1000 Hz                │
│  Resonance: 40%                 │
│                                 │
│  [LFO] [Envelope Follower]      │
│                                 │
│  LFO:                           │
│    Rate: [1/8 ▼] Sync: On       │
│    Amount: 50%                  │
│    Shape: [Sine ▼]              │
│                                 │
│  Envelope:                      │
│    Amount: 30%                  │
│    Attack: 100 ms               │
│    Release: 200 ms              │
│                                 │
│  Output: 0 dB                   │
└─────────────────────────────────┘

2つのモード:
- LFO (周期的変化)
- Envelope (信号追従)
```

### Filter Type

```
Lowpass:

特徴:
高域カット

用途:
Bass・Lead
最も使う

設定:
Cutoff: 500-2000 Hz
Resonance: 20-60%

Highpass:

特徴:
低域カット

用途:
Hi-Hat・FX

設定:
Cutoff: 2000-8000 Hz

Bandpass:

特徴:
特定帯域のみ

用途:
Vocal・Lead
実験的

設定:
Cutoff: 1000-3000 Hz
Resonance: 50%+

Notch:

特徴:
特定帯域カット

用途:
実験的

Morph (Dual):

特徴:
複数Filterブレンド

用途:
複雑な変化
```

### LFOモード

```
パラメーター:

1. Rate:

Sync: On (推奨)

設定:

1/16: 速い
1/8: 標準 (Techno/House)
1/4: ゆっくり
1/2: 非常にゆっくり

推奨:
1/8 (BPM 128)

2. Amount:

機能:
変化量

設定:

20%: わずか
50%: 標準
80%: 強烈

推奨:
30-60%

3. Shape:

Sine:
滑らか
標準

Triangle:
リニア

Square:
急激
On/Off的

Random (S&H):
ランダム
実験的

推奨:
Sine (標準)

4. Phase:

L/R位相差

180°: ステレオ広い

5. Offset:

LFO開始位置

0°: 標準
```

### Envelope Followerモード

```
機能:

Input信号:
音量大きい

Cutoff:
開く

音量小さい:
閉じる

結果:
ダイナミックFilter

パラメーター:

Amount:

+50%:
音大 → Open

-50%:
音大 → Close (逆)

Attack:

速い (10 ms):
即座に反応

遅い (200 ms):
ゆっくり

Release:

速い (50 ms):
タイト

遅い (500 ms):
滑らか

推奨設定:

Bass (ダイナミック):

Amount: +60%
Attack: 50 ms
Release: 150 ms

結果:
ベロシティでCutoff変化
```

---

## Auto Filter実践例

**トラック別設定:**

### Techno Bass (LFO)

```
目標:
グルーヴィーなCutoff変化

設定:

Filter: Lowpass
Cutoff: 800 Hz
Resonance: 45%

LFO:
Rate: 1/8
Sync: On
Amount: 50%
Shape: Sine

結果:
8分音符でCutoff開閉
グルーヴ

Chain:

Auto Filter → Saturator
温かみ追加
```

### Acid Bass (Envelope + LFO)

```
目標:
TB-303的

設定:

Filter: Lowpass
Cutoff: 600 Hz
Resonance: 65% (高め)

LFO:
Rate: 1/16 (速い)
Amount: 40%
Shape: Triangle

Envelope:
Amount: +50%
Attack: 10 ms (速い)
Release: 100 ms

結果:
クラシックAcid
```

### Hi-Hat (Highpass)

```
目標:
質感追加

設定:

Filter: Highpass
Cutoff: 5000 Hz
Resonance: 30%

LFO:
Rate: 1/8
Amount: 25% (控えめ)
Shape: Sine

結果:
わずかな動き
```

### Lead (Bandpass)

```
目標:
実験的

設定:

Filter: Bandpass
Cutoff: 2000 Hz
Resonance: 70% (高め)

LFO:
Rate: 1/4
Amount: 60%
Shape: Random

Automation:
Cutoff: 1000 Hz → 4000 Hz
16小節

結果:
複雑な動き
```

---

## Vocoder 完全ガイド

**ボーカル加工:**

### 基本原理

```
原理:

Carrier:
シンセ音

Modulator:
Vocal

結果:
「歌うシンセ」

有名曲:
Daft Punk - Harder, Better, Faster, Stronger
Kraftwerk全般
```

### インターフェイス

```
┌─────────────────────────────────┐
│  Vocoder                        │
├─────────────────────────────────┤
│  Carrier: [Internal ▼]          │
│  Modulator: [External ▼]        │
│                                 │
│  Bands: 20                      │
│  Range: Low-High                │
│                                 │
│  Formant Shift: 0 st            │
│  Noise: 20%                     │
│                                 │
│  Unvoiced: 30%                  │
│  Enhance: 50%                   │
└─────────────────────────────────┘

重要:
- Carrier (音色)
- Modulator (Vocal)
```

### 設定方法

```
Step 1: トラック構成

Track 1 (Vocoder):

Vocoder挿入
Carrier: Wavetableまたは内蔵

Track 2 (Vocal):

Audio Track
Vocalサンプル

Output: Track 1 (Sidechain)

Step 2: Vocoder設定

Modulator: External
Audio From: Track 2

Carrier: Internal Synth
または
External Instrument

Bands: 20-40
多いほど明瞭

Step 3: Carrier調整

Pitch:
Vocalメロディーに合わせ

Chord:
コード演奏

結果:
「歌うシンセ」
```

### パラメーター

```
Bands:

10-20: 太い、Lo-Fi
20-30: 標準
30-40: 明瞭

Formant Shift:

-12 st: 低く、暗い
0 st: 標準
+12 st: 高く、明るい

Noise:

0%: クリーン
20-40%: 自然
60%+: ノイジー

Unvoiced:

子音明瞭度

30-50%: 標準

Enhance:

高域強調

50%: 標準
```

---

## Vocoder実践例

**設定例:**

### Daft Punk風

```
Vocoder設定:

Bands: 20
Formant Shift: +3 st (やや高く)
Noise: 25%
Unvoiced: 40%
Enhance: 60%

Carrier:

Wavetable:
Basic Saw
Unison: 3 Voices
Detune: 15%

Chord: Em

結果:
ロボット的Vocal
```

### Kraftwerk風

```
Vocoder設定:

Bands: 16 (少なめ、Lo-Fi)
Formant Shift: 0 st
Noise: 15%
Enhance: 40%

Carrier:

Operator:
Simple Saw Wave

Chord: Minor

結果:
ビンテージロボット
```

---

## Resonators 完全ガイド

**共鳴フィルター:**

### 基本原理

```
機能:

5つのResonator:
特定周波数を強調

音程:
Resonator = 音程

MIDI:
音程演奏可能

結果:
メロディック共鳴

用途:
Drums → メロディー
Noise → トーン
```

### インターフェイス

```
┌─────────────────────────────────┐
│  Resonators                     │
├─────────────────────────────────┤
│  Mode: [Mode A ▼]               │
│                                 │
│  Resonator I:                   │
│    Pitch: C3                    │
│    Gain: 0 dB                   │
│    Width: 50%                   │
│                                 │
│  [I][II][III][IV][V]            │
│                                 │
│  Color: 50%                     │
│  Decay: 500 ms                  │
│                                 │
│  Dry/Wet: 50%                   │
└─────────────────────────────────┘

5つのResonator:
それぞれ音程
```

### パラメーター

```
Mode:

A-E:
異なるルーティング

Mode A:
並列
推奨

Pitch:

MIDI Note
C3 = 中央ハ

Gain:

各Resonator音量

Width:

Q値
狭い = 鋭い

Decay:

残響時間

100 ms: 短い
1000 ms: 長い

Color:

フィルター

0%: Dark
100%: Bright
```

---

## Resonators実践例

**用途別設定:**

### Kick → Bass

```
目標:
Kickにメロディー

設定:

Mode: A
Resonator I:
  Pitch: C2 (Root)
  Gain: 0 dB
  Width: 40%

Resonator II:
  Pitch: C3 (Octave)
  Gain: -3 dB

Decay: 400 ms

Dry/Wet: 70%

Input:
Kick

Result:
音程のあるKick
```

### Snare → Tone

```
目標:
メロディックSnare

設定:

Resonator I: E3
Resonator II: G3
Resonator III: B3

Chord: Em

Decay: 300 ms
Dry/Wet: 50%

Result:
コード的Snare
```

### Noise → Pad

```
目標:
NoiseからPad

Input:
White Noise

Resonators:
C3, E3, G3, C4, E4
(Cmajor)

Decay: 800 ms (長め)
Dry/Wet: 100%

Result:
Noise Pad
```

---

## Corpus 完全ガイド

**物理モデリング:**

### 基本原理

```
機能:

物理モデリング:
弦・板・パイプ等

Input信号:
励振

結果:
物理的共鳴

用途:
Drums → 楽器的
実験的音色
```

### Material Types

```
Beam (梁):

木材・金属棒

特徴:
金属的
ベル的

Marimba:

木琴

特徴:
温かい
木質

String (弦):

ギター・ハープ

特徴:
弦的
リンギング

Membrane (膜):

ドラム皮

特徴:
タイト
ドラム的

Plate (板):

金属板

特徴:
明るい
金属的

Pipe (パイプ):

管楽器

特徴:
倍音豊か

Tube:

中空管

特徴:
深い共鳴
```

---

## Corpus実践例

**設定例:**

### Snare → Marimba

```
Material: Marimba

Tune: C3
Decay: 0.5 s
Material: Wood

Dry/Wet: 60%

Result:
木琴的Snare
```

### Kick → Tube

```
Material: Tube

Tune: C1 (低い)
Decay: 0.8 s

Dry/Wet: 50%

Result:
深い共鳴Kick
```

---

## Grain Delay 完全ガイド

**粒状ディレイ:**

### 基本原理

```
機能:

Input:
小さい粒(Grain)に分割

処理:
各Grain独立

結果:
粒状・グリッチ的

用途:
実験的
Ambient
Glitch
```

### パラメーター

```
Spray:

ランダム化

0%: クリーン
50%: 適度
100%: カオス

Frequency:

Grain速度

1 Hz: 遅い
100 Hz: 速い

Time:

ディレイ時間

Feedback:

繰り返し

推奨設定:

Ambient:

Spray: 30%
Frequency: 10 Hz
Time: 500 ms
Feedback: 40%

Glitch:

Spray: 70%
Frequency: 50 Hz
Time: 100 ms
```

---

## Frequency Shifter

**周波数シフト:**

### 基本原理

```
Pitch Shifterとの違い:

Pitch Shifter:
倍音関係維持

Frequency Shifter:
全周波数を同じHz移動

結果:
非調和
金属的・ベル的

用途:
実験的
特殊効果
```

### パラメーター

```
Coarse:

大きなシフト
±5000 Hz

Fine:

微調整
±500 Hz

推奨:

わずか: ±50 Hz
デチューン的

中間: ±200 Hz
金属的

強烈: ±1000 Hz
非調和
```

---

## Beat Repeat

**リピート効果:**

### 基本原理

```
機能:

特定部分を繰り返し

Chance:
確率

Result:
Glitch的リピート

用途:
Glitch Hop
実験的
ビルドアップ
```

### パラメーター

```
Repeat:

繰り返し回数

2-8回

Chance:

発生確率

10%: 稀に
50%: 頻繁

Grid:

繰り返し単位

1/16, 1/8, 1/4

Gate:

繰り返し長さ

推奨設定:

Glitch:

Repeat: 4
Chance: 30%
Grid: 1/16
```

---

## よくある質問

### Q1: どれを最初に学ぶべき？

```
優先順位:

1. Auto Filter (必須)
   使用頻度: 60%
   Techno/House必須

2. Resonators (推奨)
   創造的
   15%

3. Vocoder (特定用途)
   ロボットVocal
   10%

4. その他 (実験的)
   5%

推奨:
Auto Filterから
```

### Q2: Auto FilterとEQ Eightの違い

```
EQ Eight:

静的:
固定周波数

用途:
ミックス基礎

Auto Filter:

動的:
Cutoff変化

用途:
グルーヴ・動き

使い分け:

EQ: 全トラック
Auto Filter: 一部
```

---

## まとめ

### Auto Filter

```
□ 最重要Creative Effect
□ LFO: 1/8 (Techno/House)
□ Envelope: ダイナミック
□ Resonance: 30-60%
□ Bass・Leadに必須
```

### Vocoder

```
□ Carrier + Modulator
□ Bands: 20-30
□ Formant Shift: 音程調整
□ ロボットVocal
```

### Resonators

```
□ Drums → メロディー
□ 5つのResonator
□ MIDI演奏可能
□ 実験的
```

### 重要原則

```
□ Auto Filter最優先
□ 実験的に使う
□ 控えめから始める
□ A/B比較
□ 個性を出す
```

---

**次は:** [Mastering Chain](./mastering-chain.md) - 最終仕上げのマスタリング実践
