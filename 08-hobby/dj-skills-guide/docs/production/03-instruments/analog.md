# Analog

アナログシンセの温かみ。クラシックな減算式合成で、ビンテージサウンドとモダンなTechnoサウンドを両立します。

## この章で学ぶこと

- アナログ合成の原理
- Analogの構造
- 2つのオシレーター活用
- Moogスタイルフィルター
- ノイズジェネレーター
- Unisonとデチューン
- アナログパッドの作り方
- ビンテージリードの作り方

---

## なぜAnalogが重要なのか

**温かみのある音色:**

```
音色の特徴:

Wavetable:
モダン
デジタル
クリア

Operator:
明るい
金属的

Analog:
温かい
アナログ的
ビンテージ

用途:

Analog推奨:

パッド:
温かみ
広がり

リード:
ビンテージ感
Moog的

ベース:
太い
アナログ感

使用率:

Wavetable: 70%
Operator: 15%
Analog: 10%
その他: 5%

理由:
特定の用途
温かみが欲しい時

歴史:

アナログシンセ:
1970-80年代
Moog, Prophet-5, Juno-60

Analogのモデル:
これらをエミュレート
```

---

## 減算式合成とは

**Subtractive Synthesis:**

### 基本原理

```
シンプルな流れ:

Oscillator:
豊富な倍音を生成
Saw, Square波

↓

Filter:
不要な周波数をカット
削って音色作る

↓

Amplifier:
音量調整

= 減算式

Wavetableとの違い:

Wavetable:
複雑な波形
デジタル処理

Analog:
シンプルな波形
フィルターで削る
アナログ的

メリット:

シンプル:
理解しやすい

温かい:
アナログ感

クラシック:
ビンテージサウンド
```

---

## Analogの構造

**2オシレーターシステム:**

### 全体像

```
┌────────────────────────────┐
│ Oscillator 1               │
│ Saw, Square, Triangle      │
└──────────┬─────────────────┘
           │
┌──────────▼─────────────────┐
│ Oscillator 2               │
│ Saw, Square, Triangle      │
└──────────┬─────────────────┘
           │
┌──────────▼─────────────────┐
│ Noise Generator            │
│ White, Pink                │
└──────────┬─────────────────┘
           │ Mix
           ▼
┌────────────────────────────┐
│ Filter (Moogスタイル)       │
│ 24dB Lowpass               │
└──────────┬─────────────────┘
           │
┌──────────▼─────────────────┐
│ Amplifier                  │
└──────────┬─────────────────┘
           │
┌──────────▼─────────────────┐
│ Effects                    │
│ Chorus等                   │
└────────────────────────────┘

制御:

Filter Envelope:
時間変化

Amp Envelope:
音量変化

LFO:
周期的変化
```

### インターフェイス

```
左側:
┌────────────────────────────┐
│ OSC 1, OSC 2               │
│ 波形選択、Tune             │
└────────────────────────────┘

中央:
┌────────────────────────────┐
│ FILTER                     │
│ Cutoff, Resonance, Type    │
└────────────────────────────┘

右側:
┌────────────────────────────┐
│ AMP, GLOBAL                │
│ Volume, Unison             │
└────────────────────────────┘

下部:
┌────────────────────────────┐
│ Envelopes, LFO             │
│ Filter Env, Amp Env        │
└────────────────────────────┘
```

---

## Oscillator（オシレーター）

**2つのオシレーター:**

### Wave（波形）

```
3種類の基本波形:

Saw (ノコギリ波):
明るい
豊富な倍音
ベース、リード向き

Square (矩形波):
中間的
ホロー
クラリネット的

Triangle (三角波):
暗い
シンプル
フルート的

パルス幅 (Square波):

Width調整:
細い = 明るい
太い = 暗い

PWM (Pulse Width Modulation):
LFOで自動変調
動きのある音

推奨:

ベース:
Saw + Square

パッド:
Saw + Triangle

リード:
Saw + Saw (Detune)
```

### Tune

```
Semi (半音):
-24 〜 +24 st

Detune:
-50 〜 +50 cent

OSC 2設定:

パッド:
Semi: +12 (1オクターブ上)
Detune: +7 cent

リード:
Semi: 0
Detune: +10 cent
= 厚み

ベース:
Semi: -12 (1オクターブ下)
Sub Bass層
```

### Level

```
OSC 1:
メイン
100%

OSC 2:
サブ
50-80%

バランス:

両方同じ:
厚い

OSC1優先:
クリア

推奨:

OSC 1: 100%
OSC 2: 70%
```

---

## Filter（フィルター）

**Moogスタイル:**

### Filter Type

```
24dB Lowpass:
Moog風
温かい
太い

12dB Lowpass:
軽い
明るい

Bandpass:
中域のみ
電話的

Highpass:
低域カット

推奨:

ほぼ全て:
24dB Lowpass

理由:
Analogの特徴
温かみ
```

### Cutoff & Resonance

```
Cutoff:

パッド:
800-1500 Hz
柔らかい

リード:
2000-4000 Hz
明るい

ベース:
500-1000 Hz
太い

Resonance:

0-30%:
自然

30-60%:
強調
Moog的

60-100%:
自己発振
特殊効果

推奨値:

パッド: 20%
リード: 40%
ベース: 25%
```

### Drive

```
機能:
アナログ歪み
温かみ

範囲:
0-100%

効果:
倍音付加
サチュレーション

推奨:

パッド: 10-20%
リード: 30-40%
ベース: 20-30%

注意:
上げすぎると歪む
```

---

## Noise Generator

**ノイズ追加:**

### Noise Type

```
White Noise:
全周波数均等
「シャー」

Pink Noise:
低域多い
自然

用途:

パッド:
Pink Noise 5-10%
温かみ

リード:
White Noise 2-5%
存在感

Hi-Hat:
White Noise 100%
ノイズ楽器

設定:

Level:
0-15%
控えめ

Color:
White / Pink

Filter:
ノイズもフィルター通る
```

---

## Unison & Detune

**厚みの秘密:**

### Unison

```
機能:
音を複製
わずかにデチューン

Voices:
1 (Off)
2-8 (複製数)

Detune:
0-100%
ズレ具合

推奨設定:

パッド:
Voices: 4-6
Detune: 30-50%
非常に厚い

リード:
Voices: 2-3
Detune: 20-30%
適度に

ベース:
Voices: 2
Detune: 10-15%
わずかに

注意:

CPU負荷:
Voices多い = 重い

モノラル:
Unisonで広がり
```

---

## Envelope

**時間変化:**

### Filter Envelope

```
Amount:
-100% 〜 +100%

パッド設定:

Attack: 1000 ms
ゆっくり開く

Decay: 1500 ms

Sustain: 60%

Release: 2000 ms
長い余韻

Amount: +40%
適度に

リード設定:

Attack: 50 ms
速い

Decay: 400 ms

Sustain: 40%

Release: 300 ms

Amount: +60%
しっかり
```

### Amp Envelope

```
パッド:

Attack: 1500 ms
ふわっと

Decay: 1000 ms

Sustain: 80%

Release: 2500 ms
長い

リード:

Attack: 5 ms
即座

Decay: 300 ms

Sustain: 70%

Release: 200 ms
```

---

## 実践: Analog Padの作り方

**12分で完成:**

### Step 1: 初期化 (1分)

```
1. 新規MIDIトラック

2. Analog挿入:
   Browser > Instruments > Analog

3. Init

4. 確認
```

### Step 2: Oscillator (4分)

```
OSC 1:

Wave: Saw
Semi: 0
Detune: 0
Level: 100%

OSC 2:

Wave: Saw
Semi: +12 (1オクターブ上)
Detune: +7 cent
Level: 80%

OSC 1+2 Mix:

Sub:
Level: 10%
低域補強

Noise:
Type: Pink
Level: 8%
温かみ
```

### Step 3: Filter (3分)

```
Type:
24dB Lowpass

Cutoff:
1200 Hz
柔らかい

Resonance:
22%

Drive:
18%
わずかに温かく
```

### Step 4: Unison (2分)

```
Voices:
6
非常に厚い

Detune:
45%
広がり

Amount:
100%
```

### Step 5: Envelope (2分)

```
Filter Envelope:

Attack: 1200 ms
Decay: 1800 ms
Sustain: 65%
Release: 2500 ms
Amount: +35%

Amp Envelope:

Attack: 1500 ms
Decay: 1200 ms
Sustain: 85%
Release: 3000 ms

効果:
ゆっくり包み込む
```

### Step 6: 仕上げ

```
Global:

Volume: -6 dB

確認:
Cメジャーコード (C4, E4, G4)
温かく広がる

プリセット保存:
"My Warm Pad"

完成:
美しいアナログパッド
```

---

## 実践: Vintage Leadの作り方

**10分で完成:**

### Step 1: Oscillator (4分)

```
OSC 1:

Wave: Saw
Semi: 0
Level: 100%

OSC 2:

Wave: Saw
Semi: 0
Detune: +12 cent
Level: 90%

効果:
厚み、デチューン

Sub:
Level: 15%

Noise:
Level: 3%
わずかに
```

### Step 2: Filter (3分)

```
Type:
24dB Lowpass

Cutoff:
3000 Hz
明るめ

Resonance:
38%
Moog的

Drive:
32%
歪み
```

### Step 3: Envelope (2分)

```
Filter Envelope:

Attack: 60 ms
Decay: 500 ms
Sustain: 45%
Release: 350 ms
Amount: +55%

Amp Envelope:

Attack: 8 ms
Decay: 350 ms
Sustain: 75%
Release: 250 ms
```

### Step 4: Unison (1分)

```
Voices: 3

Detune: 25%

効果:
適度な厚み
```

### Step 5: 仕上げ

```
確認:
C4 - C5 メロディ
ビンテージ感

保存:
"My Vintage Lead"
```

---

## よくある質問

### Q1: WavetableとAnalogどっち？

**A:** 用途次第

```
Wavetable推奨:

モダンな音:
Techno Bass
EDM Lead

クリア:
はっきり

Analog推奨:

温かい音:
パッド
ビンテージLead

アナログ感:
Moog的

推奨:

ベース → Wavetable
パッド → Analog
リード → どちらでも
```

### Q2: Unisonは常にOn？

**A:** 音色次第

```
Unison On推奨:

パッド:
Voices 4-6
必須

厚いリード:
Voices 2-3

Unison Off:

クリーンなベース:
単音
Unisonなし

CPU節約:
重い時Off

推奨:

パッド: 必須On
リード: お好みで
ベース: 基本Off
```

### Q3: 24dB vs 12dB Filter？

**A:** ほぼ24dB

```
24dB:
急峻
Moog的
温かい
推奨

12dB:
緩やか
明るい

使い分け:

99%: 24dB
特殊: 12dB

理由:
24dBがAnalogの特徴
```

---

## まとめ

### Analog基礎

```
□ 減算式合成
□ 2オシレーター + Noise
□ Moogスタイルフィルター
□ Unison で厚み
□ 温かみのある音色
```

### 音色別設定

```
パッド:
Saw + Saw, Cutoff 1200 Hz, Unison 6

リード:
Saw + Saw, Cutoff 3000 Hz, Unison 2-3

ベース:
Saw + Square, Cutoff 800 Hz, Unison Off
```

### 重要ポイント

```
□ OSC2 をDetune
□ Unison でパワー
□ 24dB Lowpass推奨
□ Noise わずかに
□ Drive で温かみ
```

---

**次は:** [Sampler](./sampler.md) - サンプルを自在に操る
