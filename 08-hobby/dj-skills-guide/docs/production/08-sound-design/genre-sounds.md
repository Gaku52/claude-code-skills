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
