# Genre Sounds（ジャンル別サウンド）

Techno、House、Dubstep、Trance、Hip Hop、Drum & Bassの代表的サウンドを完全再現。各ジャンル特有の音作りテクニックをStep-by-Stepで習得します。

## この章で学ぶこと

- Techno: Acid Bass、Industrial Lead
- House: Deep Bass、Pluck、Soulful Pad
- Dubstep: Wobble Bass、Growl
- Trance: Pluck、Supersaw
- Hip Hop: サンプルマニピュレーション
- Drum & Bass: Reese Bass
- 各ジャンル特有の技術
- 実践的な練習方法


## 前提知識

このガイドを読む前に、以下の知識があると理解が深まります:

- 基本的なプログラミングの知識
- 関連する基礎概念の理解
- [FM Sound Design（FM合成サウンドデザイン）](./fm-sound-design.md) の内容を理解していること

---

## なぜGenre Soundsが重要なのか

**ジャンル理解の深化:**

```
ジャンル知識なし:

状況:
何となく音作り
ジャンルの特徴不明
レーベルに送れない

結果:
ジャンル不明
A&Rに却下
リリース不可

ジャンル知識あり:

状況:
各ジャンルの定番サウンド理解
特徴を再現
プロと同じ音

結果:
ジャンル明確
A&R通過
リリース可能

プロの知識:

Techno プロデューサー:
TB-303風 Acid Bass 必須
工業的サウンド理解

House プロデューサー:
Deep Bass、Soulful Pad必須
グルーヴ重視

Dubstep プロデューサー:
Wobble Bass、Growl必須
LFO使いこなす
```

**ジャンル別の市場:**

```
市場規模（Beatport販売数）:

Techno: 35%（最大）
House: 30%
Dubstep: 10%
Trance: 8%
Drum & Bass: 7%
Hip Hop: 5%
その他: 5%

必要なサウンド:

Techno:
Acid Bass 必須度 ★★★★★
Industrial Lead 必須度 ★★★★☆

House:
Deep Bass 必須度 ★★★★★
Pluck 必須度 ★★★★☆

Dubstep:
Wobble Bass 必須度 ★★★★★
Growl 必須度 ★★★★★

結論:
ジャンル特化 = 成功への近道
```

---

## Techno（テクノ）

### 特徴

```
BPM: 125-135
Key: マイナー多い（Am、Dm、Em）
構成: ミニマル、反復
重要な音: Kick、Acid Bass、Industrial Lead

サウンドの特徴:
- 工業的、機械的
- Acid Bass（TB-303風）
- ダーク、ハードな音色
- Filter動き重要
```

### 実践1: Acid Bass（TB-303風）

```
目標: TB-303風のアシッドベースを作る

楽器: Wavetable

OSC 1:
Wavetable: Basic Shapes > Saw
Octave: 0
Volume: 100%

OSC 2: Off
SUB: 0%
UNISON: 1 voice

Filter:
Type: Low Pass 24 dB
Cutoff: 450 Hz（開始点、オートメーションで動かす）
Resonance: 72%（TB-303の特徴）
Drive: 8%

Filter Envelope:
A: 0 ms
D: 160 ms
S: 18%
R: 35 ms
Amount: +48（重要）

Amp Envelope:
A: 0 ms
D: 90 ms
S: 82%
R: 18 ms

エフェクト:

1. EQ Eight:
   High Pass: 35 Hz
   Peak: 80 Hz、+2 dB、Q 1.2

2. Saturator:
   Drive: 7 dB
   Curve: A Bit Warmer
   Dry/Wet: 85%

3. Delay:
   Time: 1/8 Dotted
   Feedback: 22%
   Dry/Wet: 14%
   Filter: LP 2500 Hz

4. Utility:
   Width: 0%（Mono必須）
   Gain: -2 dB

Macro設定:

Macro 1 - Cutoff:
- Filter Cutoff: 180 Hz - 2800 Hz

Macro 2 - Resonance:
- Resonance: 45% - 85%

Macro 3 - Env Amount:
- Filter Envelope Amount: +15 - +60

Macro 4 - Delay:
- Delay Dry/Wet: 0% - 30%

MIDIパターン（16小節）:

Bar 1-2:
A1 ──── ──── C2 ── E2 ──
Velocity: 105-125（Accent）

Bar 3-4:
A1 ── C2 ── E2 ── G2 ──
Velocity: 85-115

Bar 5-6:
D2 ──── F2 ── A2 ── C3 ──
Velocity: 95-127

Bar 7-8:
A1 ── C2 ── E2 ── A1 ──
Velocity: 100-120

Bar 9-16: バリエーション

オートメーション:
Bar 1-4: Cutoff 400 Hz → 800 Hz
Bar 5-8: Cutoff 800 Hz → 1600 Hz
Bar 9-12: Cutoff 1600 Hz → 2200 Hz
Bar 13-16: Cutoff 2200 Hz → 600 Hz（Drop）

確認:
□ TB-303風の「ピコピコ」音
□ Resonance 70%前後
□ Filter Envelope動き
□ Cutoffオートメーション
□ Acidサウンド

所要時間: 60分
```

### 実践2: Industrial Lead

```
目標: 工業的、金属的なリードを作る

楽器: Operator（FM合成）

Algorithm: 7

OSC A（Carrier）:
Waveform: Sine
Coarse: 1.00
Fine: 0 cent
Level: 100%

Envelope:
A: 5 ms
D: 150 ms
S: 65%
R: 200 ms

OSC B（Modulator 1）:
Waveform: Sine
Coarse: 5.00
Fine: +18 cent（非整数比率）
Level: 78%

Envelope:
A: 0 ms
D: 100 ms
S: 50%
R: 150 ms

OSC C（Modulator 2）:
Waveform: Sine
Coarse: 7.00
Fine: +25 cent
Level: 55%

Envelope:
A: 0 ms
D: 80 ms
S: 40%
R: 100 ms

OSC D（Noise）:
Waveform: Sine
Coarse: 11.00
Fixed: On、Frequency 3500 Hz
Level: 25%

Envelope:
A: 0 ms
D: 30 ms
S: 0%
R: 20 ms

Filter:
Type: Low Pass 12 dB
Cutoff: 4500 Hz
Resonance: 12%

エフェクト:

1. EQ Eight:
   High Pass: 250 Hz
   Peak: 2500 Hz、+3 dB、Q 1.5

2. Saturator:
   Drive: 9 dB
   Curve: Hard Curve

3. Erosion:
   Mode: Wide Noise
   Frequency: 5 kHz
   Amount: 15%

4. Reverb:
   Type: Small Room
   Decay: 1.2s
   Dry/Wet: 18%

5. Delay:
   Time: 1/16
   Feedback: 30%
   Dry/Wet: 12%

6. Utility:
   Width: 40%（狭め）
   Gain: -3 dB

確認:
□ 金属的な音色
□ 工業的な質感
□ Erosion効果
□ Technoらしい

所要時間: 75分
```

---

## House（ハウス）

### 特徴

```
BPM: 120-128
Key: メジャー・マイナー両方
構成: 4つ打ち、グルーヴ重視
重要な音: Deep Bass、Pluck、Soulful Pad、Vocal

サウンドの特徴:
- グルーヴィー
- Deep Bass（温かい）
- Soulful Pad（広がり）
- ボーカル重視
```

### 実践1: Deep House Bass

```
目標: 温かく深いDeep Bassを作る

楽器: Wavetable

OSC 1:
Wavetable: Basic Shapes > Triangle
Octave: 0
Volume: 100%

OSC 2:
Wavetable: Basic Shapes > Sine
Octave: -1
Volume: 35%

SUB: 0%
UNISON: 1 voice

Filter:
Type: Low Pass 12 dB（24 dBより柔らかい）
Cutoff: 380 Hz
Resonance: 8%（控えめ）
Drive: 3%

Filter Envelope:
A: 5 ms
D: 250 ms
S: 40%
R: 100 ms
Amount: +22

Amp Envelope:
A: 0 ms
D: 150 ms
S: 90%
R: 80 ms

エフェクト:

1. EQ Eight:
   High Pass: 32 Hz
   Peak: 100 Hz、+3 dB、Q 1.0

2. Saturator:
   Drive: 4 dB
   Curve: Warm Tube

3. Compressor:
   Ratio: 3:1
   Threshold: -10 dB
   Attack: 5 ms
   Release: 80 ms
   Gain: +3 dB

4. Chorus:
   Rate: 0.35 Hz
   Amount: 12%
   Dry/Wet: 18%

5. Reverb:
   Type: Small Room
   Decay: 0.8s
   Dry/Wet: 8%

6. Utility:
   Width: 0%（Mono）
   Gain: -1 dB

MIDIパターン:

Bar 1-2:
Am7（A1、C2、E2、G2）
タイミング: オフビート

Bar 3-4:
Fmaj7（F1、A1、C2、E2）

Bar 5-8: 繰り返し

確認:
□ 温かい音色
□ 深い低域
□ グルーヴィー
□ Deep House感

所要時間: 55分
```

### 実践2: Soulful Pad

```
目標: Soulfulで広がりのあるPadを作る

楽器: Wavetable

OSC 1:
Wavetable: FM > Harmonic Flight
Position: 38%
Octave: 0
Volume: 100%

OSC 2:
Wavetable: Additive Resonant > Formant Vowels
Position: 52%
Octave: +1
Detune: +14 cent
Volume: 62%

SUB: 0%
UNISON: 7 voices、Detune 26%

Filter:
Type: Low Pass 12 dB
Cutoff: 1600 Hz
Resonance: 6%

Amp Envelope:
A: 900 ms（ゆっくり）
D: 1600 ms
S: 75%
R: 3200 ms（長い）

LFO 1:
Destination: Filter Cutoff
Waveform: Sine
Rate: 0.22 Hz
Depth: 16%

LFO 2:
Destination: Pan
Waveform: Triangle
Rate: 0.18 Hz
Depth: 50%

エフェクト:

1. EQ Eight:
   High Pass: 300 Hz
   Low Pass: 9000 Hz

2. Chorus:
   Rate: 0.28 Hz
   Amount: 45%
   Dry/Wet: 42%

3. Phaser:
   Rate: 0.15 Hz
   Amount: 28%
   Dry/Wet: 22%

4. Reverb:
   Type: Large Hall
   Decay: 5.2s
   Damping: 3500 Hz
   Dry/Wet: 50%

5. Delay:
   Time: 1/4 Dotted
   Feedback: 35%
   Dry/Wet: 20%

6. Utility:
   Width: 160%
   Gain: -4 dB

確認:
□ Soulful感
□ 広がり大
□ ゆっくり立ち上がる
□ 温かい

所要時間: 70分
```

---

## Dubstep（ダブステップ）

### 特徴

```
BPM: 140（ハーフタイム感）
Key: マイナー多い
構成: Drop重視、ビルドアップ
重要な音: Wobble Bass、Growl、Sub Bass

サウンドの特徴:
- Wobble Bass（LFO → Filter）
- Growl（複雑な波形）
- 重低音（Sub Bass）
- ハーフタイムドラム
```

### 実践1: Wobble Bass

```
目標: 代表的なWobble Bassを作る

楽器: Wavetable

OSC 1:
Wavetable: Modern Shapes > Formant Square
Position: 45%
Octave: 0
Volume: 100%

OSC 2:
Wavetable: Basic Shapes > Saw
Octave: 0
Detune: +11 cent
Volume: 75%

SUB: 25%（低域補強）
UNISON: 4 voices、Detune 18%

Filter:
Type: Band Pass（重要）
Cutoff: 800 Hz（開始点）
Resonance: 55%
Drive: 12%

Filter Envelope:
A: 0 ms
D: 50 ms
S: 0%
R: 20 ms
Amount: +15

Amp Envelope:
A: 0 ms
D: 100 ms
S: 100%
R: 50 ms

LFO 1（Wobble）:
Destination: Filter Cutoff
Waveform: Saw Down（重要）
Rate: 1/4（テンポ同期、重要）
Depth: 65%
Retrigger: On

LFO 2:
Destination: Resonance
Waveform: Sine
Rate: 1/8
Depth: 20%

エフェクト:

1. EQ Eight:
   High Pass: 30 Hz
   Peak: 60 Hz、+4 dB、Q 1.5

2. Saturator:
   Drive: 10 dB
   Curve: Hard Curve

3. Erosion:
   Mode: Wide Noise
   Frequency: 4 kHz
   Amount: 18%

4. Auto Filter（2個目）:
   LFO → Cutoff
   さらなるWobble

5. Compressor:
   Ratio: 8:1
   Threshold: -8 dB
   Attack: 0 ms
   Release: 40 ms
   Gain: +6 dB

6. Limiter:
   Ceiling: -0.3 dB

7. Utility:
   Width: 15%（ほぼMono）
   Gain: 調整

Macro設定:

Macro 1 - Wobble Speed:
- LFO 1 Rate: 1/8 - 1/2

Macro 2 - Wobble Depth:
- LFO 1 Depth: 30% - 85%

Macro 3 - Resonance:
- Resonance: 40% - 70%

Macro 4 - Grit:
- Erosion Amount: 0% - 30%

MIDIパターン:

Bar 1-2:
E1 ──────────────────────
（ロングノート、4小節）

Velocity: 127

オートメーション:
Macro 1（Wobble Speed）:
Bar 1-2: 1/4
Bar 3-4: 1/8
Bar 5-6: 1/16（速くなる）

確認:
□ Wobble効果（LFO → Filter）
□ Band Pass Filter
□ Saw Down波形
□ 1/4 同期
□ Dubstep Wobble

所要時間: 70分
```

### 実践2: Growl Bass

```
目標: 複雑なGrowl Bassを作る

楽器: Wavetable

OSC 1:
Wavetable: Complex > Dystopia
Position: 62%
Octave: 0
Volume: 100%

OSC 2:
Wavetable: Digital > Glitch Step
Position: 45%
Octave: 0
Detune: +9 cent
Volume: 80%

SUB: 30%
UNISON: 6 voices、Detune 22%

Filter 1:
Type: Low Pass 24 dB
Cutoff: 1200 Hz
Resonance: 48%

Filter 2:
Type: Band Pass
Cutoff: 600 Hz
Resonance: 35%

LFO 1:
Destination: Filter 1 Cutoff
Waveform: Random S&H
Rate: 1/16
Depth: 55%

LFO 2:
Destination: OSC 1 Position
Waveform: Triangle
Rate: 1/8
Depth: 40%

LFO 3:
Destination: Unison Detune
Waveform: Sine
Rate: 0.5 Hz
Depth: 30%

エフェクト:

1. Saturator:
   Drive: 12 dB
   Curve: Digital Clip

2. Erosion:
   Amount: 25%

3. Redux:
   Bit Depth: 10 bit
   Sample Rate: 8000 Hz

4. Auto Filter:
   LFO複数

5. Utility:
   Width: 20%

確認:
□ Growl音
□ 複雑な倍音
□ ランダムLFO
□ デジタル感

所要時間: 85分
```

---

## Trance（トランス）

### 特徴

```
BPM: 130-140
Key: メジャー・マイナー
構成: ビルドアップ、Drop、ブレイクダウン
重要な音: Pluck、Supersaw、Pad

サウンドの特徴:
- Pluck（短いアタック）
- Supersaw（多重Saw波）
- メロディ重視
- エモーショナル
```

### 実践1: Trance Pluck

```
目標: 代表的なPluck音を作る

楽器: Wavetable

OSC 1:
Wavetable: Basic Shapes > Saw
Octave: 0
Volume: 100%

OSC 2:
Wavetable: Basic Shapes > Square
Octave: +1
Detune: +9 cent
Volume: 50%

SUB: 0%
UNISON: 3 voices、Detune 14%

Filter:
Type: Low Pass 24 dB
Cutoff: 3500 Hz
Resonance: 18%

Filter Envelope:
A: 0 ms
D: 350 ms（重要）
S: 0%
R: 120 ms
Amount: +35

Amp Envelope:
A: 0 ms
D: 400 ms
S: 0%（重要、Pluck特性）
R: 150 ms

Pitch Envelope:
A: 0 ms
D: 40 ms
S: 0%
R: 0 ms
Amount: +5 semitones

エフェクト:

1. EQ Eight:
   High Pass: 200 Hz

2. Compressor:
   Ratio: 4:1
   Threshold: -12 dB
   Attack: 1 ms
   Release: 100 ms
   Gain: +4 dB

3. Chorus:
   Rate: 0.6 Hz
   Amount: 30%
   Dry/Wet: 35%

4. Reverb:
   Type: Plate
   Decay: 2.5s
   Dry/Wet: 28%

5. Delay:
   Time: 1/8
   Feedback: 25%
   Dry/Wet: 18%

6. Utility:
   Width: 130%
   Gain: -2 dB

MIDIパターン:

Bar 1-2:
C3 ─ E3 ─ G3 ─ C4 ─ E4 ─ G4 ─ C5 ─ ─
16分音符アルペジオ

確認:
□ 短いPluck音
□ Decay 350 ms
□ Sustain 0%
□ Filter Envelope動き
□ Tranceらしい

所要時間: 65分
```

### 実践2: Supersaw

```
目標: 厚いSupersaw Leadを作る

方法: 5-7層のSaw波をレイヤリング

Layer 1（Main）:
OSC: Saw
Detune: 0 cent
Volume: 100%

Layer 2:
OSC: Saw
Detune: +7 cent
Volume: 90%

Layer 3:
OSC: Saw
Detune: -7 cent
Volume: 90%

Layer 4:
OSC: Saw
Detune: +14 cent
Volume: 75%

Layer 5:
OSC: Saw
Detune: -14 cent
Volume: 75%

Layer 6:
OSC: Saw
Octave: +1
Detune: +5 cent
Volume: 40%

Layer 7:
OSC: Saw
Octave: -1
Volume: 25%

または Wavetable UNISON使用:

UNISON:
Amount: 8 voices
Detune: 35%

Filter:
Cutoff: 3500 Hz

エフェクト:
Chorus、Reverb、Delay

確認:
□ 非常に厚い
□ 広がり150%以上
□ Supersaw感

所要時間: 90分（レイヤリング）、60分（UNISON）
```

---

## Hip Hop（ヒップホップ）

### 特徴

```
BPM: 70-100
Key: 様々
構成: ループベース、サンプリング重視
重要な音: サンプル加工、808 Bass、ボーカルチョップ

サウンドの特徴:
- サンプリング（レコード）
- 808 Bass（ロング）
- Lo-Fi質感
- ボーカルチョップ
```

### 実践1: 808 Bass

```
目標: ロングな808 Bassを作る

楽器: Wavetable

OSC 1:
Wavetable: Basic Shapes > Sine
Octave: 0
Volume: 100%

OSC 2: Off
SUB: 0%
UNISON: 1 voice

Filter:
Type: Low Pass 12 dB
Cutoff: 180 Hz
Resonance: 0%

Pitch Envelope:
A: 0 ms
D: 60 ms
S: 0%
R: 0 ms
Amount: +14 semitones

Amp Envelope:
A: 0 ms
D: 400 ms（長い）
S: 70%
R: 200 ms

エフェクト:

1. EQ Eight:
   High Pass: 28 Hz
   Peak: 55 Hz、+4 dB、Q 1.8

2. Saturator:
   Drive: 5 dB
   Curve: Warm

3. Compressor:
   Ratio: 6:1
   Threshold: -10 dB
   Attack: 0 ms
   Release: 100 ms
   Gain: +4 dB

4. Utility:
   Width: 0%
   Gain: +1 dB

MIDIパターン:

Bar 1:
C1 ──────────── ─ F1 ─
ロングノート

確認:
□ Sine波
□ Pitch Envelope下降
□ Decay 400 ms
□ 808感

所要時間: 45分
```

### 実践2: ボーカルチョップ（Lo-Fi）

```
参照: sampling-techniques.md

追加エフェクト（Lo-Fi化）:

1. Redux:
   Bit Depth: 8-12 bit
   Sample Rate: 16000 Hz

2. Vinyl Distortion:
   Crackle、Noise追加

3. EQ Eight:
   High Pass: 300 Hz
   Low Pass: 6000 Hz
   → 帯域制限

4. Reverb:
   Small Room
   Dry/Wet: 25%

確認:
□ Lo-Fi質感
□ ビンテージ感
□ Hip Hop感

所要時間: 60分
```

---

## Drum & Bass（ドラムンベース）

### 特徴

```
BPM: 170-180
Key: マイナー多い
構成: 高速ドラム、Reese Bass
重要な音: Reese Bass、アーメンブレイク

サウンドの特徴:
- Reese Bass（デチューン）
- 高速ドラム（170 BPM）
- ダーク
- エネルギッシュ
```

### 実践1: Reese Bass

```
目標: 代表的なReese Bassを作る

楽器: Wavetable

OSC 1:
Wavetable: Basic Shapes > Saw
Octave: 0
Detune: 0 cent
Volume: 100%

OSC 2:
Wavetable: Basic Shapes > Saw
Octave: 0
Detune: +18 cent（重要、Reese特性）
Volume: 100%

OSC 3（LayerまたはSimpler追加）:
Wavetable: Basic Shapes > Saw
Octave: 0
Detune: -18 cent
Volume: 100%

SUB: 0%
UNISON: 2 voices、Detune 8%

Filter:
Type: Low Pass 24 dB
Cutoff: 500 Hz
Resonance: 22%

Amp Envelope:
A: 5 ms
D: 100 ms
S: 85%
R: 80 ms

LFO 1:
Destination: OSC 2 Detune
Waveform: Sine
Rate: 0.8 Hz
Depth: 15%
→ Detuneが揺れる、Reese特性

エフェクト:

1. EQ Eight:
   High Pass: 32 Hz
   Peak: 80 Hz、+3 dB、Q 1.2

2. Saturator:
   Drive: 8 dB
   Curve: A Bit Warmer

3. Chorus:
   Rate: 0.4 Hz
   Amount: 35%
   Dry/Wet: 25%

4. Compressor:
   Ratio: 4:1
   Threshold: -10 dB

5. Utility:
   Width: 0%
   Gain: -2 dB

確認:
□ 2-3 Saw波、Detune ±18 cent
□ Filter LP 500 Hz
□ LFO → Detune
□ うねるような音
□ Reese感

所要時間: 70分
```

---

## ジャンル横断テクニック

### サンプリング vs シンセシス

```
ジャンル別使用率:

Techno:
シンセシス: 80%
サンプリング: 20%

House:
シンセシス: 60%
サンプリング: 40%

Dubstep:
シンセシス: 90%
サンプリング: 10%

Trance:
シンセシス: 85%
サンプリング: 15%

Hip Hop:
シンセシス: 20%
サンプリング: 80%

D&B:
シンセシス: 70%
サンプリング: 30%
```

### BPM別のGroove

```
遅いBPM（70-100、Hip Hop）:
グルーヴ: Swing 60-70%
ドラム: ゆったり
ベース: ロング（Decay 400 ms）

中速BPM（120-128、House）:
グルーヴ: Swing 55-65%
ドラム: 4つ打ち
ベース: Medium（Decay 150 ms）

速いBPM（130-140、Techno、Dubstep、Trance）:
グルーヴ: Swing 50-60%
ドラム: タイト
ベース: 短め（Decay 100 ms）

超高速BPM（170-180、D&B）:
グルーヴ: Swing 50-55%
ドラム: 非常にタイト
ベース: 短い（Decay 80 ms）
```

---

## よくある質問（FAQ）

**Q1: どのジャンルから始めるべきですか？**

```
A: 好きなジャンルから

推奨順序:

1. 好きなジャンル（モチベーション最重要）

2. 初心者向け:
   - House（シンプル、グルーヴ）
   - Lo-Fi Hip Hop（サンプリング）

3. 中級者向け:
   - Techno（Acid Bass）
   - Trance（メロディ）

4. 上級者向け:
   - Dubstep（複雑なLFO）
   - D&B（高速、複雑）

理由:
好きなジャンル = 継続できる
```

**Q2: 複数ジャンルを作るべきですか？**

```
A: 最初は1ジャンル特化

戦略:

Year 1:
1ジャンル特化
完全マスター

Year 2:
2ジャンル目追加
クロスオーバー

Year 3:
3-4ジャンル
多様性

理由:
1ジャンル特化 = リリース確率高い
複数ジャンル = リスク分散

プロの例:
Adam Beyer: Techno 100%
Deadmau5: Progressive House 80%、他 20%
Skrillex: Dubstep → 多様化
```

**Q3: ジャンルの特徴を掴むコツは？**

```
A: Top 10分析

手順:

1. Beatport Top 10選択:
   自分のジャンル

2. 10曲をAbletonに読み込み:
   スペクトラム分析

3. 共通点を探す:
   - BPM
   - Key
   - ベース周波数
   - ドラムパターン
   - サウンド

4. 再現を試みる:
   50%似たら成功

5. 自分の曲と比較:
   差分を埋める

結果:
ジャンルの特徴理解
プロと同じ音
```

**Q4: Acid Bassがうまく作れません**

```
A: Resonance と Filter Envelopeが鍵

チェックリスト:

□ Saw波使用
□ Filter LP 24 dB
□ Resonance 60-80%
□ Filter Envelope Amount +40以上
□ Decay 150-200 ms
□ Sustain 10-30%
□ Cutoffオートメーション

調整:
Resonance 70% → 80%
→ より「ピコピコ」

Envelope Amount +40 → +55
→ より音色変化

確認:
参考曲（TB-303）と比較
```

**Q5: Wobble Bassの "Wobble" が弱いです**

```
A: LFO設定を見直す

チェックリスト:

□ Filter Type: Band Pass（重要）
□ LFO Waveform: Saw Down
□ LFO Rate: 1/4（テンポ同期）
□ LFO Depth: 60%以上
□ LFO Retrigger: On
□ Resonance: 50%以上

調整:
Depth 60% → 75%
→ より強いWobble

Rate 1/4 → 1/8
→ 速いWobble

確認:
Skrillex等の参考曲と比較
```

**Q6: Supersawが薄いです**

```
A: レイヤー数とDetune

解決策:

方法1: レイヤリング
5-7層のSaw波
Detune: 0, ±7, ±14 cent

方法2: UNISON
Amount: 8 voices
Detune: 35%

方法3: 両方組み合わせ
2層のWavetable（各 UNISON 6 voices）
+ Detune

エフェクト:
Chorus 40%
Stereo Width 150%

結果:
非常に厚い Supersaw
```

---

## 練習方法

### Week 1: Techno完全マスター

```
Day 1-3: Acid Bass
1. TB-303風 Acid Bass作成
2. 5種類のバリエーション
3. オートメーション練習

Day 4-5: Industrial Lead
1. Operator使用
2. 金属的なリード3種類
3. エフェクト実験

Day 6-7: Techno楽曲制作
1. Acid Bass + Industrial Lead
2. ドラム（Kick、Hi-Hat）
3. 8小節 → 完成

目標:
Techno特有のサウンド習得
```

### Week 2: House & Dubstep

```
Day 1-2: Deep Bass
1. House用Deep Bass作成
2. 温かい音色
3. グルーヴ練習

Day 3-4: Wobble Bass
1. Dubstep用Wobble作成
2. LFO → Filter
3. Macro設定

Day 5-6: Soulful Pad
1. House用Pad作成
2. 広がり重視
3. エモーショナル

Day 7: 2ジャンル楽曲
1. House楽曲完成
2. Dubstep楽曲完成

目標:
2ジャンル習得
```

### Week 3: Trance & Hip Hop

```
Day 1-2: Trance Pluck
1. Pluck音作成
2. アルペジオパターン
3. 5種類

Day 3-4: 808 Bass
1. Hip Hop用808作成
2. Pitch Envelope
3. ロングDecay

Day 5-6: ボーカルチョップ
1. サンプリング
2. Lo-Fi化
3. Hip Hopトラック

Day 7: 2ジャンル楽曲
1. Trance楽曲完成
2. Hip Hop楽曲完成

目標:
合計4ジャンル習得
```

### Week 4: D&B & 総合練習

```
Day 1-2: Reese Bass
1. D&B用Reese作成
2. Detune ±18 cent
3. LFO → Detune

Day 3-4: 全ジャンル復習
1. 6ジャンル × 1サウンド作成
2. 時間短縮（30分/サウンド）

Day 5-6: クロスオーバー
1. Techno + Dubstep
2. House + Hip Hop
3. 実験的楽曲

Day 7: ポートフォリオ作成
1. 6ジャンル楽曲完成
2. SoundCloudにアップ
3. レーベルに送る

目標:
全ジャンル習得、リリース可能
```

---

## まとめ

ジャンル別サウンドは、プロへの道の最終ステップです。

**重要ポイント:**

1. **Techno**: Acid Bass（Resonance 70%）、Industrial Lead
2. **House**: Deep Bass（温かい）、Soulful Pad
3. **Dubstep**: Wobble Bass（LFO 1/4）、Growl
4. **Trance**: Pluck（Sustain 0%）、Supersaw
5. **Hip Hop**: 808 Bass、ボーカルチョップ
6. **D&B**: Reese Bass（Detune ±18 cent）

**学習順序:**
1. 好きなジャンル1つ（Month 1-3）
2. 関連ジャンル追加（Month 4-6）
3. 多様化（Month 7-12）

**プロへの道:**
- 1ジャンル特化 → リリース確率高い
- Top 10分析 → 特徴理解
- 50%再現 → 十分プロレベル

**次のステップ:** サウンドデザイン完了、ミキシング・マスタリングへ

---

## 関連ファイル

- **[Wavetable Sound Design](./wavetable-sound-design.md)** - Wavetable活用
- **[FM Sound Design](./fm-sound-design.md)** - Operator活用
- **[Sampling Techniques](./sampling-techniques.md)** - サンプリング技術
- **[Layering](./layering.md)** - レイヤリング技術
- **[Modulation Techniques](./modulation-techniques.md)** - モジュレーション

---

**ジャンル別サウンドで、プロと同じ音を作りましょう！**

---

## 発展編: ジャンル別キック・ベース・リードの周波数特性マップ

各ジャンルの三大要素（キック、ベース、リード）について、周波数帯域ごとの特性を詳細にマッピングします。ミックスの際に各要素が衝突しないよう、ジャンルごとの「住み分け」を理解することが極めて重要です。

### Technoの周波数マップ

```
Techno Kick:
  Sub（20-60 Hz）: ★★★★★ 非常に強い
  Low（60-200 Hz）: ★★★★☆ 強い
  Low-Mid（200-500 Hz）: ★★☆☆☆ 控えめ
  Mid（500-2000 Hz）: ★★★☆☆ クリック感
  High-Mid（2-5 kHz）: ★★★★☆ アタック
  High（5-20 kHz）: ★★☆☆☆ 軽い空気感

  特徴:
  - 50 Hz付近に強烈なピーク
  - 3-4 kHz付近にアタックのクリック
  - Decayは短め（200-350 ms）
  - サイドチェインコンプで他のパートをダッキング
  - ピッチ: E1-G1（約41-49 Hz）が定番

Techno Acid Bass:
  Sub（20-60 Hz）: ★★★☆☆ 中程度
  Low（60-200 Hz）: ★★★★☆ メイン帯域
  Low-Mid（200-500 Hz）: ★★★☆☆ フィルター動きで変化
  Mid（500-2000 Hz）: ★★★★★ フィルターのスイープ領域
  High-Mid（2-5 kHz）: ★★★★☆ レゾナンスのピーク
  High（5-20 kHz）: ★★★☆☆ ディストーションの倍音

  特徴:
  - 80-150 Hz: ベースのファンダメンタル
  - 800-2500 Hz: フィルターカットオフのスイープ範囲
  - レゾナンスにより2-4 kHz付近にシャープなピーク
  - Saturator/Distortionで高域の倍音を追加

Techno Industrial Lead:
  Sub（20-60 Hz）: ☆☆☆☆☆ なし
  Low（60-200 Hz）: ★☆☆☆☆ 最小限
  Low-Mid（200-500 Hz）: ★★★☆☆ ボディ
  Mid（500-2000 Hz）: ★★★★★ メイン帯域
  High-Mid（2-5 kHz）: ★★★★★ 金属感
  High（5-20 kHz）: ★★★★☆ エッジ、ノイズ

  特徴:
  - 1-3 kHz: FM合成による金属的な倍音集中
  - 4-6 kHz: Erosionによるノイジーな質感
  - ハイパスフィルターで250 Hz以下をカット
  - ステレオ幅は狭め（40%）でモノ互換性確保
```

### Houseの周波数マップ

```
House Kick:
  Sub（20-60 Hz）: ★★★★☆ 強い
  Low（60-200 Hz）: ★★★★★ 太い
  Low-Mid（200-500 Hz）: ★★★☆☆ 温かさ
  Mid（500-2000 Hz）: ★★☆☆☆ 控えめ
  High-Mid（2-5 kHz）: ★★★☆☆ 軽いアタック
  High（5-20 kHz）: ★☆☆☆☆ 最小限

  特徴:
  - Technoより丸い音色
  - 80-120 Hz付近にピーク
  - アタックはTechnoほど鋭くない
  - Decayはやや長め（250-400 ms）
  - 4つ打ちの土台としてドライブ感を与える

House Deep Bass:
  Sub（20-60 Hz）: ★★★★★ 非常に強い
  Low（60-200 Hz）: ★★★★★ ファンダメンタル
  Low-Mid（200-500 Hz）: ★★★★☆ 温かさの源
  Mid（500-2000 Hz）: ★★☆☆☆ 控えめ
  High-Mid（2-5 kHz）: ★☆☆☆☆ ほぼなし
  High（5-20 kHz）: ☆☆☆☆☆ なし

  特徴:
  - Triangle波ベースで温かい倍音構成
  - 55-110 Hz: ファンダメンタルが支配的
  - ローパスフィルター380 Hzで高域をカット
  - Saturatorで軽いハーモニクスを追加
  - コーラスで微かな揺れを付与

House Soulful Pad:
  Sub（20-60 Hz）: ☆☆☆☆☆ なし
  Low（60-200 Hz）: ☆☆☆☆☆ なし
  Low-Mid（200-500 Hz）: ★★★☆☆ 温かさ
  Mid（500-2000 Hz）: ★★★★★ メイン帯域
  High-Mid（2-5 kHz）: ★★★★☆ 輝き
  High（5-20 kHz）: ★★★☆☆ 空気感

  特徴:
  - ハイパス300 Hz / ローパス9000 Hzでバンド制限
  - Unisonの7ボイスで厚みを出す
  - コーラス + フェイザーで有機的な動き
  - リバーブ（Large Hall 5.2s）で空間的な広がり
  - ステレオ幅160%で包み込む感覚
```

### Dubstepの周波数マップ

```
Dubstep Kick:
  Sub（20-60 Hz）: ★★★★★ 極めて強い
  Low（60-200 Hz）: ★★★★★ 重い
  Low-Mid（200-500 Hz）: ★★★☆☆ 存在感
  Mid（500-2000 Hz）: ★★★☆☆ クリック
  High-Mid（2-5 kHz）: ★★★★☆ 鋭いアタック
  High（5-20 kHz）: ★★☆☆☆ 軽い

  特徴:
  - 30-50 Hz付近に強烈なサブの塊
  - ハーフタイムフィールのため間隔が広い
  - Dropセクションではキックよりベースが主役
  - サイドチェインはかなり深め（-10 dB以上）

Dubstep Wobble Bass:
  Sub（20-60 Hz）: ★★★★☆ Sub層で補強
  Low（60-200 Hz）: ★★★★★ ファンダメンタル
  Low-Mid（200-500 Hz）: ★★★★★ フィルター中心帯域
  Mid（500-2000 Hz）: ★★★★★ LFOスイープの核心
  High-Mid（2-5 kHz）: ★★★★☆ エロージョンの質感
  High（5-20 kHz）: ★★★☆☆ ディストーション倍音

  特徴:
  - Band Passフィルターで400-2000 Hzをスイープ
  - LFO（Saw Down, 1/4同期）によるリズミカルな動き
  - Saturator + Erosionで攻撃的な質感
  - コンプレッサーで8:1の強い圧縮
  - リミッターで-0.3 dBにピーク制御

Dubstep Growl Bass:
  Sub（20-60 Hz）: ★★★★☆ Sub層
  Low（60-200 Hz）: ★★★★★ 重い
  Low-Mid（200-500 Hz）: ★★★★★ 複雑な倍音
  Mid（500-2000 Hz）: ★★★★★ グロウルの核心
  High-Mid（2-5 kHz）: ★★★★★ デジタルノイズ
  High（5-20 kHz）: ★★★★☆ Redux/ビット破壊

  特徴:
  - ランダムS&H LFOによる予測不能な動き
  - Wavetable Position変調で波形自体が変化
  - Redux（10 bit / 8000 Hz）でデジタルグリッチ
  - デュアルフィルター構成（LP + BP）
  - 6ボイスユニゾンで分厚い壁のような音
```

### Tranceの周波数マップ

```
Trance Kick:
  Sub（20-60 Hz）: ★★★★★ パンチのある低域
  Low（60-200 Hz）: ★★★★☆ しっかり
  Low-Mid（200-500 Hz）: ★★☆☆☆ 控えめ
  Mid（500-2000 Hz）: ★★★☆☆ ミッドパンチ
  High-Mid（2-5 kHz）: ★★★★★ 鋭いクリック
  High（5-20 kHz）: ★★★☆☆ 空気感

  特徴:
  - 55-65 Hz付近にサブのピーク
  - 3.5-5 kHzに非常に鋭いアタッククリック
  - Technoよりもクリックが強調される傾向
  - テール（リリース）は中程度
  - レイヤー構成: Sub層 + Body層 + Click層

Trance Pluck:
  Sub（20-60 Hz）: ☆☆☆☆☆ なし
  Low（60-200 Hz）: ☆☆☆☆☆ なし（HPF 200 Hz）
  Low-Mid（200-500 Hz）: ★★★☆☆ ボディ
  Mid（500-2000 Hz）: ★★★★★ メイン帯域
  High-Mid（2-5 kHz）: ★★★★☆ 明るさ
  High（5-20 kHz）: ★★★☆☆ シャイン

  特徴:
  - Sustain 0%のPluck特性
  - Filter Envelope Decay 350 msで倍音が急速に変化
  - Pitch Envelope +5 semitonesで打撃感
  - リバーブ（Plate 2.5s）で残響を付与
  - ステレオ幅130%でワイド感

Trance Supersaw Lead:
  Sub（20-60 Hz）: ☆☆☆☆☆ なし
  Low（60-200 Hz）: ★★☆☆☆ ルート音の下端
  Low-Mid（200-500 Hz）: ★★★★☆ 厚み
  Mid（500-2000 Hz）: ★★★★★ Sawの密度
  High-Mid（2-5 kHz）: ★★★★★ ブライトネス
  High（5-20 kHz）: ★★★★☆ エア、シマー

  特徴:
  - 5-7レイヤーまたは8ボイスユニゾン
  - Detune幅: ±7, ±14 centの階層的配置
  - コーラス + リバーブで空間的厚み
  - ステレオ幅150%以上が基本
  - ダイナミクスはコンプレッサーで制御
```

### D&Bの周波数マップ

```
D&B Kick:
  Sub（20-60 Hz）: ★★★★☆ 強い
  Low（60-200 Hz）: ★★★★★ パンチ
  Low-Mid（200-500 Hz）: ★★★☆☆ 中程度
  Mid（500-2000 Hz）: ★★★★☆ スナップ感
  High-Mid（2-5 kHz）: ★★★★★ 非常に鋭いアタック
  High（5-20 kHz）: ★★★☆☆ クリスプ

  特徴:
  - 170-180 BPMの高速テンポに対応する短いDecay
  - アタックが非常に鋭く、タイト
  - スネアとの組み合わせでブレイクビートパターン
  - 60-100 Hz付近のファンダメンタルが明瞭
  - レイヤー: Sub + Punch + Click の3層構成

D&B Reese Bass:
  Sub（20-60 Hz）: ★★★★★ 深い
  Low（60-200 Hz）: ★★★★★ うねり
  Low-Mid（200-500 Hz）: ★★★★☆ デチューンの倍音
  Mid（500-2000 Hz）: ★★★☆☆ フィルターで制御
  High-Mid（2-5 kHz）: ★★☆☆☆ 控えめ
  High（5-20 kHz）: ★☆☆☆☆ 最小限

  特徴:
  - 2-3本のSaw波をDetune ±18 centで重ねる
  - LFOでDetuneを揺らしてうねり（Reese特性）を生成
  - LP 24 dB / 500 Hzで高域をカット
  - コーラスで微妙な揺れを追加
  - モノ（Width 0%）で位相の崩れを防ぐ

D&B Snare:
  Sub（20-60 Hz）: ★☆☆☆☆ 最小限
  Low（60-200 Hz）: ★★★☆☆ ボディ
  Low-Mid（200-500 Hz）: ★★★★☆ パンチ
  Mid（500-2000 Hz）: ★★★★★ スナッピー
  High-Mid（2-5 kHz）: ★★★★★ クラック
  High（5-20 kHz）: ★★★★☆ ノイズテール

  特徴:
  - D&Bのスネアは非常に特徴的
  - レイヤー: Body（200-500 Hz）+ Snap（1-3 kHz）+ Noise（5+ kHz）
  - サンプルのピッチシフト、リバースが多用される
  - アーメンブレイクからの抽出・加工が定番
  - トランジェントシェイパーでアタックを強調
```

### Hip Hopの周波数マップ

```
Hip Hop Kick（808）:
  Sub（20-60 Hz）: ★★★★★ 極めて強い（メイン）
  Low（60-200 Hz）: ★★★★★ Pitch Envelopeの動き
  Low-Mid（200-500 Hz）: ★★☆☆☆ 軽いボディ
  Mid（500-2000 Hz）: ★★☆☆☆ クリック
  High-Mid（2-5 kHz）: ★★☆☆☆ 軽いアタック
  High（5-20 kHz）: ★☆☆☆☆ 最小限

  特徴:
  - Sine波がベース
  - Pitch Envelope: +14 semitones → 0（60 msで下降）
  - 非常にロングなDecay（400 ms以上）
  - 55 Hz付近に+4 dBのピーク
  - ローパス180 Hzで上をカット
  - モノ必須（サブウーファー再生のため）

Hip Hop 808 Bass（ロングベース）:
  Sub（20-60 Hz）: ★★★★★ メイン
  Low（60-200 Hz）: ★★★★★ サステイン部分
  Low-Mid（200-500 Hz）: ★☆☆☆☆ ほぼなし
  Mid（500-2000 Hz）: ★☆☆☆☆ Distortionで僅かに
  High-Mid（2-5 kHz）: ☆☆☆☆☆ なし
  High（5-20 kHz）: ☆☆☆☆☆ なし

  特徴:
  - Sustain 70%でロングに伸びるベース
  - グライド（ポルタメント）でピッチスライド
  - コンプレッサー6:1で安定感
  - Saturatorで倍音を足し小型スピーカーでも聞こえるように
  - サブベースとしての役割が最重要
```

---

## シンセプリセットレシピ集

各ジャンルの代表的なサウンドを素早く再現するためのレシピを紹介します。DAWの新規プロジェクトを開いたら、これらの設定を入力するだけで即座にジャンルらしいサウンドが得られます。

### レシピ1: Minimal Techno Stab

```
概要: ミニマルテクノで使われる短いスタブ音

楽器: Wavetable または Analog

OSC設定:
  OSC 1: Saw波、Octave 0
  OSC 2: Square波、Octave +1、Detune +3 cent
  Volume比率: OSC 1 = 100%、OSC 2 = 40%

フィルター:
  Type: Low Pass 24 dB
  Cutoff: 1800 Hz
  Resonance: 25%

エンベロープ:
  Filter Env:
    A: 0 ms / D: 80 ms / S: 0% / R: 40 ms
    Amount: +30

  Amp Env:
    A: 0 ms / D: 120 ms / S: 0% / R: 60 ms

エフェクト:
  1. Saturator: Drive 6 dB, Warm Tube
  2. EQ Eight: HPF 150 Hz, Peak 1.2 kHz +2 dB
  3. Reverb: Small Room, Decay 0.6s, Dry/Wet 15%
  4. Delay: 1/16, Feedback 15%, Dry/Wet 10%

ポイント:
  - Sustain 0%が「Stab」感のカギ
  - Decayは80-120 msの短い範囲
  - リバーブは控えめ（空間を汚さない）
  - ベロシティでFilter Cutoffを制御すると表現力UP
```

### レシピ2: Future Bass Chord（フューチャーベース・コード）

```
概要: EDMのFuture Bassで使われる柔らかいコードサウンド

楽器: Wavetable

OSC設定:
  OSC 1: Wavetable "FM > Soft FM"、Position 30%
  OSC 2: Wavetable "Basic Shapes > Saw"
  Volume比率: OSC 1 = 100%、OSC 2 = 55%
  UNISON: 5 voices、Detune 20%

フィルター:
  Type: Low Pass 12 dB
  Cutoff: 2200 Hz
  Resonance: 12%

エンベロープ:
  Filter Env:
    A: 15 ms / D: 600 ms / S: 35% / R: 300 ms
    Amount: +28

  Amp Env:
    A: 10 ms / D: 800 ms / S: 60% / R: 400 ms

LFO:
  LFO 1 → Volume（サイドチェイン風）:
    Waveform: Custom（急速下降→ゆっくり上昇）
    Rate: 1/4 同期
    Depth: 40%

エフェクト:
  1. OTT（Multiband Dynamics）: Amount 40%
  2. Chorus: Rate 0.5 Hz, Amount 25%, Dry/Wet 30%
  3. Reverb: Hall, Decay 3.0s, Dry/Wet 35%
  4. Stereo Width: 140%

ポイント:
  - OTT（Over The Top）コンプレッションがFuture Bassの核
  - コードはMaj7、Min7、Sus2を多用
  - LFOでポンピング効果を擬似再現
  - 高域のシマー感が重要
```

### レシピ3: Lo-Fi Hip Hop Keys

```
概要: Lo-Fiヒップホップの定番ピアノ/ローズ音

楽器: Simpler（サンプルベース）またはElectric

ベースサウンド:
  Electric Piano（Fender Rhodes系）

加工チェイン:
  1. Vinyl Distortion:
     Tracing Model: On
     Crackle: 15%
     Pinch: 5%

  2. Auto Filter:
     Type: Low Pass
     Cutoff: 3500 Hz
     LFO Rate: 0.3 Hz
     LFO Depth: 8%

  3. Redux:
     Bit Depth: 12 bit
     Sample Rate: 22050 Hz

  4. EQ Eight:
     HPF: 200 Hz
     LPF: 5000 Hz
     Peak: 800 Hz, +2 dB

  5. Chorus:
     Rate: 0.4 Hz
     Amount: 20%
     Dry/Wet: 25%

  6. Reverb:
     Type: Small Room
     Decay: 1.5s
     Dry/Wet: 22%

  7. Utility:
     Width: 110%
     Gain: -3 dB

ポイント:
  - Vinyl Distortionでレコードの質感を出す
  - Reduxのビットデプスは12 bit以上（8 bitだとやりすぎ）
  - ローパスフィルターで高域を丸める
  - スウィング65%でグルーヴ感を出す
  - ベロシティのバラツキ（80-110）で人間味を追加
```

### レシピ4: Progressive House Arp

```
概要: プログレッシブハウスのアルペジオシンセ

楽器: Wavetable

OSC設定:
  OSC 1: Wavetable "Additive Resonant > Bright Harmonics"
  Position: 25%
  OSC 2: Basic Shapes > Saw、Octave +1
  Volume比率: OSC 1 = 100%、OSC 2 = 30%
  UNISON: 3 voices、Detune 12%

フィルター:
  Type: Low Pass 24 dB
  Cutoff: 2800 Hz
  Resonance: 15%

エンベロープ:
  Filter Env:
    A: 0 ms / D: 300 ms / S: 20% / R: 150 ms
    Amount: +32

  Amp Env:
    A: 0 ms / D: 350 ms / S: 10% / R: 200 ms

Arpeggiatorデバイス:
  Style: Up
  Rate: 1/16
  Gate: 60%
  Steps: 8

エフェクト:
  1. Compressor: Ratio 3:1, Threshold -10 dB
  2. Ping Pong Delay: Time 3/16, Feedback 40%, Dry/Wet 25%
  3. Reverb: Plate, Decay 2.8s, Dry/Wet 30%
  4. Utility: Width 130%, Gain -2 dB

ポイント:
  - Arpeggiator + Ping Pong Delayの組み合わせが鉄板
  - 3/16のDelay Timeで独特のリズム感
  - コードは1-2音のシンプルなものが効果的
  - フィルターオートメーションでビルドアップに活用
```

### レシピ5: Ambient Techno Pad

```
概要: アンビエントテクノで使われる浮遊感のあるパッド

楽器: Wavetable

OSC設定:
  OSC 1: Wavetable "Noise > Breathe"
  Position: 45%、Octave 0
  OSC 2: Wavetable "FM > Distant Bell"
  Position: 60%、Octave +1
  Volume比率: OSC 1 = 100%、OSC 2 = 45%
  UNISON: 4 voices、Detune 30%

フィルター:
  Type: Low Pass 12 dB
  Cutoff: 1200 Hz
  Resonance: 5%

エンベロープ:
  Amp Env:
    A: 3000 ms / D: 2000 ms / S: 70% / R: 5000 ms

LFO:
  LFO 1 → Filter Cutoff:
    Waveform: Sine
    Rate: 0.08 Hz（非常にゆっくり）
    Depth: 20%

  LFO 2 → OSC 1 Position:
    Waveform: Triangle
    Rate: 0.12 Hz
    Depth: 35%

エフェクト:
  1. Grain Delay: Pitch -5, Spray 80 ms, Dry/Wet 20%
  2. Reverb: Large Hall, Decay 8.0s, Dry/Wet 60%
  3. Delay: 1/4 Dotted, Feedback 50%, Dry/Wet 25%
  4. Auto Pan: Rate 0.15 Hz, Amount 40%
  5. Utility: Width 180%, Gain -5 dB

ポイント:
  - アタック3秒以上でゆっくり立ち上がる
  - リリース5秒以上で長く残響
  - Grain Delayでテクスチャーを追加
  - 非常に遅いLFOで微かな動きを与える
  - リバーブのDry/Wetは50%以上で空間的
```

---

## リファレンストラック分析

プロの楽曲を分析し、各ジャンルのサウンドデザインの実例を学びます。

### Techno リファレンス

```
分析対象曲の傾向（Beatport Techno Top 10の共通特性）:

キック:
  - ピッチ: E1-G1（41-49 Hz）
  - アタックの長さ: 1-3 ms
  - ディケイ: 200-350 ms
  - 3-5 kHzにクリック成分
  - サイドチェインは4-6 dB程度のリダクション

ベース:
  - 多くがモノラル
  - Acid Bassの場合: レゾナンス60-80%
  - フィルターオートメーションが楽曲展開の軸
  - サブベースは55-80 Hz帯に集中
  - Kickとの棲み分け: サイドチェインまたはEQで住み分け

リード/シンセ:
  - FM合成系が多い（金属的、デジタル）
  - ディレイ（1/16、1/8 Dotted）で奥行き
  - ステレオ幅は控えめ（40-80%）
  - オートメーションによる変化が楽曲の推進力

構成:
  - イントロ: 32-64小節（Kickのみから始まる）
  - ブレイク: 16-32小節（Kickなし、Pad/FX）
  - ドロップ: 32-64小節（全パート復帰）
  - アウトロ: 16-32小節（徐々にパートを抜く）

ミックス特性:
  - マスターラウドネス: -6 to -8 LUFS
  - ダイナミックレンジ: 6-10 LU
  - ローエンドのモノ互換性が極めて重要
  - ハイパスフィルター: キック以外は全て30-80 Hzでカット
```

### House リファレンス

```
分析対象曲の傾向（Beatport House Top 10の共通特性）:

キック:
  - Technoより丸く温かい
  - 80-120 Hz付近が中心
  - アタックはソフトめ
  - バウンス感のあるGroove
  - ゴーストキックを使う場合あり

ベース:
  - Deep Bassは温かいTriangle/Sine系
  - オフビートに配置されることが多い
  - グライド/ポルタメントで滑らかな動き
  - コードに沿ったベースライン（ルート+5th）
  - Saturatorで軽くウォームアップ

パッド/コード:
  - 7thコード（Maj7、Min7）が頻出
  - 広いステレオイメージ（140-180%）
  - ゆっくりしたフィルターLFO
  - リバーブが深め（Decay 3-6s）
  - ボーカルサンプルとの相性を考慮

ボーカル:
  - House楽曲の核となる要素
  - 男性/女性ボーカルどちらも
  - コーラス/ハーモニーの多層レイヤー
  - ディレイ（1/4 Dotted）で空間演出
  - De-esserで歯擦音を制御

ミックス特性:
  - マスターラウドネス: -7 to -9 LUFS
  - Technoよりダイナミックレンジが広い
  - グルーヴの微細なタイミングが命
  - ミッドレンジの温かさが重要
```

### Dubstep リファレンス

```
分析対象曲の傾向（Beatport Dubstep Top 10の共通特性）:

イントロ/ビルドアップ:
  - メロディックな要素（ピアノ、ボーカル）
  - 徐々にテンション上昇
  - ホワイトノイズスウィープ
  - リバースシンバル
  - BPM: 140（ハーフタイム感覚で70 BPMに感じる）

ドロップ:
  - Wobble BassまたはGrowl Bassが主役
  - ハーフタイムドラム（スネアが3拍目）
  - サブベースとミッドベースの2層構造
  - 極端なサイドチェイン（-12 dB以上）
  - エフェクト: ビットクラッシャー、フランジャー

ベース構造:
  - Sub Bass層: 30-80 Hz（Sine波、クリーン）
  - Mid Bass層: 80-5000 Hz（Wobble/Growl）
  - 2つの層をマルチバンド処理で分離
  - Mid Bass層にのみディストーション適用
  - Sub Bass層はクリーンなまま維持

サウンドデザインの特徴:
  - 自動化（オートメーション）の量が非常に多い
  - 1曲あたり50-100トラックのオートメーション
  - LFO Rate、Filter Cutoff、Wavetable Positionが主な対象
  - リサンプリング技法（ベースを一度録音→再加工）が多用
  - FM合成 + ウェーブテーブル + サンプリングの併用

ミックス特性:
  - マスターラウドネス: -5 to -7 LUFS（最も音圧が高い）
  - ダイナミックレンジは狭い
  - マルチバンドコンプで帯域ごとに制御
  - サブベースのヘッドルームが極めて重要
```

---

## サウンドの時代変遷

各ジャンルのサウンドは時代とともに進化してきました。歴史を知ることで、現在のサウンドがなぜそのように作られているのかを理解できます。

### Technoの進化

```
1980年代後半（Detroit Techno誕生期）:
  代表アーティスト: Juan Atkins、Derrick May、Kevin Saunderson
  使用機材: Roland TR-808、TR-909、TB-303、Jupiter-8、DX7
  サウンド特性:
  - アナログシンセのウォームさ
  - TB-303のAcid Bass（偶然の発見）
  - TR-909のキック・ハイハット
  - コードとメロディが比較的多い
  - ソウル/ファンクの影響が残る

1990年代前半（ヨーロッパでの発展）:
  代表アーティスト: Jeff Mills、Richie Hawtin、Dave Clarke
  サウンド特性:
  - よりミニマルな方向へ
  - 反復の美学が確立
  - ハードテクノの台頭
  - サンプラーの導入（Akai MPC）
  - エフェクトの積極活用

1990年代後半-2000年代（Minimal Techno）:
  代表アーティスト: Ricardo Villalobos、Richie Hawtin（Plastikman）
  サウンド特性:
  - 極端なミニマリズム
  - マイクロサウンド（微細な音の変化）
  - クリック&カット
  - デジタルプロセッシング
  - ダブの影響（ディレイ、リバーブ）

2010年代（Industrial Techno復興）:
  代表アーティスト: Ben Klock、Marcel Dettmann、Blawan
  サウンド特性:
  - Industrial/EBMの影響回帰
  - 重くダークなキック
  - ディストーションの多用
  - ノイズ/テクスチャーの重視
  - モジュラーシンセの流行

2020年代（現在）:
  代表アーティスト: ANNA、Amelie Lens、I Hate Models
  サウンド特性:
  - ハードテクノの大衆化
  - 高速BPM（140-150）の台頭
  - Rave/Acid Technoの復興
  - ブレイクビーツとの融合
  - AIツールの活用が始まる
```

### Houseの進化

```
1980年代（Chicago House誕生期）:
  代表アーティスト: Frankie Knuckles、Marshall Jefferson、Larry Heard
  使用機材: TR-808、TR-909、Juno-106、DX7
  サウンド特性:
  - ディスコの進化形
  - 4つ打ちキックの確立
  - ゴスペル/ソウルの影響
  - アナログシンセの温かさ
  - ボーカルサンプルの多用

1990年代（多様化の時代）:
  サブジャンル分岐:
  - Deep House: Larry Heard、Kerri Chandler
    → 温かいパッド、ジャズ的コード
  - Progressive House: Sasha、John Digweed
    → ロングブレイク、壮大な展開
  - Tech House: Mr. G、Terry Francis
    → TechnoとHouseの融合
  - Garage: Todd Edwards、MJ Cole
    → 2ステップリズム、UKサウンド

2000年代（Electro House/Minimal）:
  代表アーティスト: Deadmau5、Eric Prydz、Dubfire
  サウンド特性:
  - エレクトロハウスの台頭
  - サイドチェインコンプのポンピング効果
  - フィルターハウスの発展
  - デジタルプロダクションの標準化
  - Ableton Liveの普及

2010年代（EDM/Deep House復興）:
  代表アーティスト: Disclosure、Duke Dumont、Tchami
  サウンド特性:
  - Future Houseの誕生
  - ポップとの融合
  - Deep Houseのメインストリーム化
  - ベースハウスの台頭
  - ストリーミング時代への適応

2020年代（現在）:
  代表アーティスト: Fred again..、Skrillex（House転向）、John Summit
  サウンド特性:
  - テックハウスの大衆化
  - ブレイクビーツハウスの流行
  - UKガラージの再評価
  - ライブ感のある制作スタイル
  - ソーシャルメディアとの連動
```

### Dubstepの進化

```
2000年代前半（UK Dubstep誕生期）:
  代表アーティスト: Skream、Benga、Digital Mystikz
  サウンド特性:
  - ダブ/レゲエの影響
  - 深いサブベース
  - ミニマルなアプローチ
  - ハーフタイムリズム
  - スペースと残響の美学
  - 非常にダークで内省的

2008-2012年（Brostep / 商業化）:
  代表アーティスト: Skrillex、Excision、Flux Pavilion
  サウンド特性:
  - Wobble Bassの過激化
  - Growl Bassの発展
  - FM合成の多用
  - 音圧競争（ラウドネスウォー）
  - ドロップ中心の構成
  - 映画/ゲームとのタイアップ

2013-2018年（多様化と成熟）:
  代表アーティスト: Virtual Riot、MUST DIE!、Zomboy
  サウンド特性:
  - リサンプリング技法の高度化
  - Serum シンセサイザーの標準化
  - サウンドデザインの複雑化
  - Riddimサブジャンルの台頭
  - Colour Bassの誕生
  - メロディック・ダブステップの発展

2020年代（現在）:
  代表アーティスト: Subtronics、SVDDEN DEATH、Chime
  サウンド特性:
  - Riddim Dubstepの主流化
  - ウェーブテーブル合成の高度活用
  - AIアシストによるサウンドデザイン
  - Tearout/Briddimなど更なる細分化
  - ライブパフォーマンス重視
  - コラボレーション文化の発展
```

---

## クロスジャンル・サウンドデザイン

異なるジャンルの要素を組み合わせて、オリジナリティのあるサウンドを作る手法を解説します。

### Techno x Dubstep（Industrial Bass Music）

```
コンセプト:
  Technoの反復性・ミニマリズム + Dubstepのベースデザイン
  BPM: 140-150
  代表アーティスト: MUST DIE!、I Hate Models、REMNANT.exe

ベースの作り方:
  楽器: Wavetable

  OSC 1:
    Wavetable: Complex > Dystopia
    Position: 50%
    Octave: 0
    Volume: 100%

  OSC 2:
    Wavetable: Basic Shapes > Saw
    Octave: 0
    Detune: +12 cent
    Volume: 70%

  UNISON: 4 voices、Detune 15%

  フィルター:
    Type: Band Pass
    Cutoff: 700 Hz
    Resonance: 40%

  LFO 1:
    Destination: Filter Cutoff
    Waveform: Saw Down
    Rate: 1/8（テンポ同期）
    Depth: 50%

  エフェクト:
    1. Saturator: Drive 8 dB、Hard Curve
    2. EQ Eight: HPF 40 Hz、Peak 800 Hz +3 dB
    3. Compressor: Ratio 6:1、Threshold -8 dB
    4. Utility: Width 20%、Mono寄り

  Techno要素:
    - 反復するパターン（16小節ループ）
    - フィルターの緩やかなオートメーション
    - ミニマルな展開

  Dubstep要素:
    - Band Passフィルターの使用
    - LFOによるWobble的な動き
    - ディストーションの強さ
```

### House x Trance（Progressive/Melodic House）

```
コンセプト:
  Houseのグルーヴ + Tranceのメロディ・エモーション
  BPM: 124-130
  代表アーティスト: Anyma、Artbat、Tale Of Us

リード/Pluckの作り方:
  楽器: Wavetable

  OSC 1:
    Wavetable: Additive Resonant > Bright Harmonics
    Position: 35%
    Octave: 0

  OSC 2:
    Wavetable: Basic Shapes > Saw
    Octave: +1
    Detune: +6 cent
    Volume: 45%

  UNISON: 4 voices、Detune 16%

  フィルター:
    Type: Low Pass 24 dB
    Cutoff: 3000 Hz
    Resonance: 15%

  Filter Envelope:
    A: 0 ms / D: 400 ms / S: 15% / R: 200 ms
    Amount: +30

  Amp Envelope:
    A: 5 ms / D: 500 ms / S: 20% / R: 300 ms

  エフェクト:
    1. Compressor: Ratio 3:1
    2. Ping Pong Delay: 3/16、Feedback 35%、Dry/Wet 22%
    3. Reverb: Plate、Decay 3.5s、Dry/Wet 32%
    4. Utility: Width 140%

  House要素:
    - 4つ打ちキック
    - グルーヴィーなベースライン
    - 控えめなSwing

  Trance要素:
    - エモーショナルなメロディ
    - Pluck音のアルペジオ
    - ビルドアップ→ドロップの構成
    - Supersawの厚み
```

### D&B x Techno（Techstep/Neurofunk）

```
コンセプト:
  D&Bの高速リズム + Technoのダークさ・工業感
  BPM: 174
  代表アーティスト: Noisia、Current Value、Mefjus

Neurofunk Bassの作り方:
  楽器: Wavetable + リサンプリング

  Step 1 - 基本波形作成:
    OSC 1: Wavetable "Digital > Glitch Step"、Position 55%
    OSC 2: Wavetable "Complex > Dystopia"、Position 40%
    UNISON: 3 voices、Detune 10%

  Step 2 - フィルター変調:
    Filter: Band Pass、Cutoff 900 Hz、Resonance 50%
    LFO 1 → Cutoff: Rate 1/16、Depth 60%
    LFO 2 → OSC 1 Position: Rate 1/8、Depth 45%

  Step 3 - ディストーション:
    Saturator: Drive 10 dB、Digital Clip
    Erosion: Amount 20%

  Step 4 - リサンプリング:
    1. 上記の音を4小節録音
    2. 録音をSimplerに読み込み
    3. ワープモードで再加工
    4. スライス → 並べ替え

  Step 5 - 最終処理:
    EQ Eight: HPF 35 Hz、Peak 100 Hz +3 dB
    Compressor: Ratio 4:1
    Utility: Width 0%（モノ）

  D&B要素:
    - 174 BPMの高速テンポ
    - ブレイクビートパターン
    - Reese Bassの要素

  Techno要素:
    - 工業的なディストーション
    - FM合成由来の金属的倍音
    - ミニマルな反復
```

### Hip Hop x House（G-House / Bass House）

```
コンセプト:
  Hip Hopのグルーヴ・808 + Houseの4つ打ち
  BPM: 124-128
  代表アーティスト: Malaa、Tchami、DJ Snake

Bass Houseベースの作り方:
  楽器: Wavetable

  OSC 1:
    Wavetable: Basic Shapes > Saw
    Octave: -1
    Volume: 100%

  OSC 2:
    Wavetable: Basic Shapes > Square
    Octave: -1
    Detune: +4 cent
    Volume: 65%

  SUB: 20%（Sine波）
  UNISON: 2 voices、Detune 5%

  フィルター:
    Type: Low Pass 24 dB
    Cutoff: 600 Hz
    Resonance: 30%
    Drive: 10%

  Filter Envelope:
    A: 0 ms / D: 180 ms / S: 25% / R: 80 ms
    Amount: +35

  Amp Envelope:
    A: 0 ms / D: 200 ms / S: 80% / R: 60 ms

  エフェクト:
    1. Saturator: Drive 7 dB、A Bit Warmer
    2. OTT: Amount 30%
    3. EQ Eight: HPF 30 Hz、Peak 80 Hz +3 dB
    4. Compressor: Ratio 4:1、Threshold -10 dB
    5. Utility: Width 0%

  Hip Hop要素:
    - 808的なサブベースの太さ
    - グルーヴィーなベースライン
    - ボーカルチョップ
    - トラップ風のハイハット

  House要素:
    - 4つ打ちキック
    - BPM 124-128
    - サイドチェインコンプ
    - ビルドアップ構成
```

---

## 実践演習: ジャンル再現チャレンジ

### 演習1: 3ジャンル・ベース比較制作

```
目的: 同じルートノート（A1）で3ジャンルのベースを作り分ける

課題:
  1. Techno Acid Bass（A1）
  2. House Deep Bass（A1）
  3. Dubstep Wobble Bass（A1）

手順:
  1. 3つのMIDIトラックを作成
  2. 全てA1のロングノート（8小節）を打ち込む
  3. 各トラックにジャンル別のシンセ設定を適用
  4. スペクトラムアナライザーで周波数特性を比較

評価基準:
  □ 同じノートなのに全く異なるキャラクター
  □ Techno: フィルターの動きが明確
  □ House: 温かさ、丸み
  □ Dubstep: LFOによるリズミカルな変化
  □ 各ベースが自分のジャンルらしいと感じる

分析ポイント:
  - 周波数の分布はどう違うか？
  - フィルターの設定はどう影響しているか？
  - エンベロープの違いが音色にどう反映されるか？
  - エフェクトチェインの違いは？

所要時間: 90分
```

### 演習2: ジャンル変換チャレンジ

```
目的: 1つのループを異なるジャンルに変換する

課題:
  House 8小節ループを作成 → Techno版、D&B版に変換

Step 1 - オリジナルHouseループ作成（BPM 124）:
  - 4つ打ちキック
  - Deep Bass（Am → Fmaj7）
  - Soulful Pad
  - パーカッション

Step 2 - Techno変換:
  BPM変更: 124 → 130
  キック: より硬く、クリック追加
  ベース: Deep Bass → Acid Bass
  パッド: Soulful → Industrial Lead
  パーカッション: より機械的に
  エフェクト: ディストーション追加、リバーブ削減

Step 3 - D&B変換:
  BPM変更: 124 → 174
  キック: ブレイクビートパターンに変更
  ベース: Deep Bass → Reese Bass
  パッド: 削除またはアンビエント化
  ドラム: スネア追加、ハイハット高速化
  エフェクト: コンプレッション強化

評価基準:
  □ 各バージョンがそのジャンルらしく聞こえる
  □ BPM変更に伴うグルーヴの調整ができている
  □ サウンドの置き換えが適切
  □ ミックスバランスが各ジャンルに適している

所要時間: 120分
```

### 演習3: ブラインドジャンル判定テスト

```
目的: 自分のサウンドが本当にジャンルらしいか客観的に評価する

手順:
  1. 6ジャンル（Techno、House、Dubstep、Trance、Hip Hop、D&B）
     のそれぞれで8小節のデモを作成

  2. 全デモをファイル名をランダムに変更して保存
     例: demo_A.wav、demo_B.wav ... demo_F.wav

  3. 友人/知人に聞かせてジャンルを当ててもらう

  4. 正答率を記録:
     6/6: エクセレント（ジャンル特性を完全に理解）
     5/6: グッド（概ね良い、1つ改善点あり）
     4/6: 平均（2つのジャンルの区別が曖昧）
     3/6以下: 要練習（ジャンルの核となるサウンド要素の復習が必要）

  5. 間違えられたジャンルの分析:
     - 何と間違えられたか？
     - どの要素が不足していたか？
     - リファレンストラックとの差は何か？

改善サイクル:
  テスト → 分析 → 修正 → 再テスト
  月1回実施を推奨

所要時間: 各デモ30分 × 6 = 180分 + テスト時間
```

### 演習4: リファレンスマッチング

```
目的: プロの楽曲のサウンドをできるだけ正確に再現する

手順:
  1. Beatportから各ジャンルのTop 1曲をダウンロード
  2. Abletonに読み込み、ループ再生
  3. 隣のトラックで同じサウンドを再現
  4. A/B比較で差を埋める

対象サウンド（各ジャンル1つ）:
  Techno: Acid Bassを1フレーズ再現
  House: Deep Bassを1フレーズ再現
  Dubstep: Wobble Bassのドロップ1小節を再現
  Trance: Supersawのコード進行を再現
  Hip Hop: 808 Bassのパターンを再現
  D&B: Reese Bassを1フレーズ再現

評価方法:
  スペクトラムアナライザーで比較:
  - 周波数バランスの一致度
  - ダイナミクスの一致度
  - ステレオイメージの一致度

  聴覚テスト:
  - 目を閉じてA/B比較
  - 70%以上似ていれば合格
  - プロの質感に近づけば成功

  具体的な数値目標:
  - RMS差: ±2 dB以内
  - ピーク周波数差: ±50 Hz以内
  - ステレオ相関: ±10%以内

所要時間: 各サウンド60分 × 6 = 360分（数日に分けて実施）
```

### 演習5: オリジナルジャンル創造

```
目的: 既存ジャンルの知識を活かし、2つのジャンルを融合した
     オリジナルサウンドを開発する

Step 1 - ジャンル選択:
  以下から2つ選択して融合:
  A: Techno + Trance = "Tech Trance"
  B: House + D&B = "Liquid Funk"
  C: Dubstep + Hip Hop = "Trap"
  D: Trance + Dubstep = "Melodic Bass"
  E: Techno + D&B = "Crossbreed"

Step 2 - サウンドパレット定義:
  ジャンル1から取り入れる要素（3つ）:
  1. _______________
  2. _______________
  3. _______________

  ジャンル2から取り入れる要素（3つ）:
  1. _______________
  2. _______________
  3. _______________

Step 3 - BPM / Key / 構成の決定:
  BPM: ジャンル1とジャンル2の中間値を基準に調整
  Key: 両ジャンルに適したキーを選択
  構成: 両ジャンルの構成の良いところを組み合わせ

Step 4 - サウンドデザイン:
  キック: どちらのジャンルのキックを基にするか
  ベース: 両ジャンルのベース要素をどうブレンドするか
  リード/パッド: どちらの特性を持たせるか
  ドラム: リズムパターンの融合方法

Step 5 - 4小節デモ制作:
  融合サウンドで4小節のデモを制作
  A/B比較: 各ジャンル単体との違いを確認

Step 6 - フィードバック:
  SoundCloudにアップして反応を確認
  コメントでジャンルを聞く → どう認識されるか

評価基準:
  □ 2つのジャンルの要素が共存している
  □ 新鮮さがある（既存のどちらとも完全には一致しない）
  □ まとまりがある（バラバラに聞こえない）
  □ 踊れる / 体が動く

所要時間: 240分（1日かけて実施）
```

---

## ジャンル別マスタリング指針

各ジャンルのマスタリングにはそれぞれ特有の基準があります。

```
Techno マスタリング:
  ターゲットラウドネス: -7 LUFS
  ダイナミックレンジ: 7-9 LU
  ローエンド: 30 Hz以下をカット、50-60 Hzにわずかなブースト
  ステレオ: 100 Hz以下をモノ化
  リミッター: Ceiling -0.5 dB、Release 中速
  注意: DJミックスを前提とした設計（イントロ/アウトロの音量）

House マスタリング:
  ターゲットラウドネス: -8 LUFS
  ダイナミックレンジ: 8-10 LU
  ローエンド: 35 Hz以下をカット、80-100 Hzをやや強調
  ステレオ: 120 Hz以下をモノ化
  リミッター: Ceiling -0.5 dB、Release 中速～やや遅め
  注意: グルーヴ感を潰さないよう控えめなリミッティング

Dubstep マスタリング:
  ターゲットラウドネス: -6 LUFS
  ダイナミックレンジ: 5-7 LU
  ローエンド: 25 Hz以下をカット、40-60 Hzを強調
  ステレオ: 80 Hz以下をモノ化
  マルチバンドコンプ: 必須（特にLow/Low-Mid帯域）
  リミッター: Ceiling -0.3 dB、Release 速め
  注意: ドロップの音圧確保が最優先

Trance マスタリング:
  ターゲットラウドネス: -7 LUFS
  ダイナミックレンジ: 8-10 LU
  ローエンド: 30 Hz以下をカット
  ステレオ: 100 Hz以下をモノ化
  リミッター: Ceiling -0.5 dB
  注意: ブレイクダウンとドロップのダイナミクスを保つ

Hip Hop マスタリング:
  ターゲットラウドネス: -9 LUFS
  ダイナミックレンジ: 6-8 LU
  ローエンド: 28 Hz以下をカット、50-80 Hzを強調
  ステレオ: 100 Hz以下をモノ化
  リミッター: Ceiling -0.5 dB
  注意: ボーカルとの帯域バランス、808の存在感

D&B マスタリング:
  ターゲットラウドネス: -7 LUFS
  ダイナミックレンジ: 7-9 LU
  ローエンド: 30 Hz以下をカット
  ステレオ: 100 Hz以下をモノ化
  リミッター: Ceiling -0.5 dB、Release 速め
  注意: 高速BPMでのトランジェント保持
```

---

## ジャンル別推奨プラグイン・ツール

```
Techno向け:
  シンセ:
  - Diva（u-he）: アナログエミュレーション最高峰
  - Repro-5（u-he）: Prophet-5エミュレーション
  - TAL-BassLine-101: TB-303/SH-101クローン
  - Serum（Xfer）: ウェーブテーブル
  エフェクト:
  - Valhalla VintageVerb: ビンテージリバーブ
  - Soundtoys Decapitator: サチュレーション
  - FabFilter Pro-Q 3: 精密EQ

House向け:
  シンセ:
  - Juno-106エミュレーション（TAL-U-NO-LX）
  - Arturia Mini V: Minimoogエミュレーション
  - Massive X（Native Instruments）
  エフェクト:
  - Valhalla Room: ナチュラルリバーブ
  - RC-20 Retro Color: ビンテージ質感
  - Cableguys ShaperBox: LFOツール

Dubstep向け:
  シンセ:
  - Serum（Xfer）: ★★★★★ 必須中の必須
  - Phase Plant（Kilohearts）: モジュラーシンセ
  - Vital: 無料ウェーブテーブルシンセ
  エフェクト:
  - OTT（Xfer、無料）: マルチバンドコンプ
  - Trash 2（iZotope）: マルチバンドディストーション
  - Disperser（Kilohearts）: 位相操作

Trance向け:
  シンセ:
  - Sylenth1（LennarDigital）: Supersawの定番
  - Spire（Reveal Sound）: 多機能シンセ
  - Nexus（reFX）: プリセットが豊富
  エフェクト:
  - Valhalla Supermassive（無料）: 壮大なリバーブ
  - Soundtoys EchoBoy: 高品質ディレイ
  - Ozone（iZotope）: マスタリングスイート

Hip Hop向け:
  シンセ/サンプラー:
  - Omnisphere（Spectrasonics）: 万能シンセ
  - Kontakt（Native Instruments）: サンプラー
  - RC-20 Retro Color: Lo-Fi必携
  エフェクト:
  - iZotope Vinyl（無料）: ビニール質感
  - Soundtoys Little AlterBoy: ボーカル変調
  - Waves CLA-76: クラシックコンプ

D&B向け:
  シンセ:
  - Serum（Xfer）: ベースデザインに最適
  - Massive（Native Instruments）: Reeseの定番
  - FM8（Native Instruments）: FM合成
  エフェクト:
  - FabFilter Saturn 2: マルチバンドディストーション
  - Transient Master（Native Instruments）
  - Pro-L 2（FabFilter）: リミッター
```

---

## トラブルシューティング: ジャンルサウンドの問題解決

### よくある問題と解決策

```
問題1: Technoのキックが軽い
  原因: サブ帯域の不足、Decayが短すぎる
  解決:
  - 50 Hz付近を+3-4 dBブースト
  - Decay を250-350 msに調整
  - Saturatorで軽い倍音を追加
  - サブレイヤー（Sine波 50 Hz）を追加

問題2: Houseのベースが冷たい
  原因: 波形選択、エフェクト不足
  解決:
  - Saw波 → Triangle波に変更
  - Saturator（Warm Tube）を追加
  - コーラスで微かな揺れ
  - レゾナンスを下げる（10%以下）
  - フィルターカットオフを下げる

問題3: Dubstepのベースがスカスカ
  原因: レイヤー不足、コンプレッション不十分
  解決:
  - Sub Bass層を別トラックで追加
  - Mid Bass層のユニゾンボイスを増やす
  - OTTコンプレッションを追加
  - Saturator + Erosionで倍音を埋める
  - コンプレッサーの比率を上げる（8:1以上）

問題4: Tranceのメロディが平坦
  原因: ダイナミクス不足、空間処理不足
  解決:
  - ベロシティの強弱をつける
  - Pitch Envelopeで微かなピッチ変化を追加
  - Ping Pong Delayでリズム感を付与
  - リバーブ（Plate 2-3s）で空間的奥行き
  - ステレオ幅を広げる（130-150%）

問題5: Hip Hopの808が小さいスピーカーで聞こえない
  原因: サブ帯域のみで倍音がない
  解決:
  - Saturatorで倍音を追加（Drive 5-8 dB）
  - EQで80-120 Hz付近をやや強調
  - マルチバンドコンプでLow帯域を持ち上げる
  - リファレンスとの比較を車・イヤホンで実施

問題6: D&BのReese Bassがうねらない
  原因: Detune幅が不足、LFO設定
  解決:
  - Detune幅を±18-25 centに拡大
  - LFO → Detuneのdepthを15-25%に
  - コーラスを追加（Rate 0.4 Hz、Amount 35%）
  - 2つ以上のSaw波を使う
  - フィルターカットオフを微調整（400-600 Hz）
```

---

## 補足: サウンドデザインの効率化ワークフロー

```
効率的な制作フロー:

Phase 1 - テンプレート準備（初回のみ、120分）:
  1. 6ジャンル分のプロジェクトテンプレートを作成
  2. 各テンプレートに基本的なシンセ設定を保存
  3. エフェクトチェインをラック化してプリセット保存
  4. MIDIパターンのクリップライブラリを作成

Phase 2 - 日常の制作フロー（毎回、60分）:
  1. テンプレートから新規プロジェクト作成（2分）
  2. プリセットをロード（3分）
  3. MIDIパターンを打ち込みまたは選択（10分）
  4. 音色を微調整（20分）
  5. エフェクト調整（15分）
  6. ミックスバランス（10分）

Phase 3 - プリセット管理:
  命名規則:
    [ジャンル]_[カテゴリ]_[特徴]_[バージョン]
    例: Techno_Bass_Acid_v2
    例: House_Pad_Soulful_v1
    例: Dubstep_Bass_Wobble_v3

  フォルダ構成:
    Presets/
    ├── Techno/
    │   ├── Bass/
    │   ├── Lead/
    │   └── Pad/
    ├── House/
    │   ├── Bass/
    │   ├── Chord/
    │   └── Pad/
    ├── Dubstep/
    │   ├── Bass/
    │   └── FX/
    ├── Trance/
    │   ├── Lead/
    │   ├── Pluck/
    │   └── Pad/
    ├── HipHop/
    │   ├── 808/
    │   └── Keys/
    └── DnB/
        ├── Bass/
        └── Drums/

  バックアップ:
  - Google Drive / Dropboxに同期
  - バージョン管理（v1, v2 ...）
  - 月1回のバックアップチェック
```

---

## 次に読むべきガイド

- [Layering（レイヤリング）](./layering.md) - 次のトピックへ進む

---

## 参考文献

- [MDN Web Docs](https://developer.mozilla.org/) - Web技術のリファレンス
- [Wikipedia](https://ja.wikipedia.org/) - 技術概念の概要
