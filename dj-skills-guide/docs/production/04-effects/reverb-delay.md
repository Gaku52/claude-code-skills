# Reverb・Delay

空間系エフェクトで深みと奥行きを作ります。Reverb・Delayを完全マスターし、プロの立体的なミックスを実現します。

## この章で学ぶこと

- Reverbの種類（Hall・Room・Plate）
- Decay Time・Pre-Delay調整
- Delay Time・Feedback設定
- Ping Pong Delay
- Return Track活用
- ジャンル別設定（Techno/House）
- 低域処理・EQ

---

## なぜ空間系が重要なのか

**深みと奥行き:**

```
空間なしミックス:

特徴:
平面的
狭い
近い

空間ありミックス:

特徴:
立体的
広い
奥行き

使用頻度:

Reverb: 70%
全トラックに何らかの空間

Delay: 60%
装飾・リズム

プロの真実:

「良いミックス」=
空間使い分け

近い: Vocal・Lead (Dry)
中間: Snare・Pad
遠い: FX・Ambience

結果:
立体的・プロの音
```

---

## Reverb 完全ガイド

**残響エフェクト:**

### インターフェイス

```
┌─────────────────────────────────┐
│  Reverb                         │
├─────────────────────────────────┤
│  Type: [Hall ▼]                 │
│                                 │
│  Pre-Delay: 20.0 ms             │
│  Decay Time: 2.50 s             │
│  Size: 70%                      │
│                                 │
│  Diffusion: 80%                 │
│  Damping: 6000 Hz               │
│                                 │
│  Dry/Wet: 30%                   │
│                                 │
│  [Early Reflections] [Global]   │
└─────────────────────────────────┘

重要パラメーター:
- Type (空間の種類)
- Decay Time (残響時間)
- Size (空間サイズ)
- Pre-Delay (初期反射遅延)
```

### Reverbタイプ

```
Hall (ホール):

特徴:
大きい空間
長い残響
豊か

用途:
Pad・Strings
壮大な雰囲気

設定:

Decay: 2.5-4.0 s
Size: 60-80%
Pre-Delay: 20-40 ms

ジャンル:
Ambient・Trance

Room (部屋):

特徴:
小さい空間
短い残響
自然

用途:
Drums・Vocal
リアル

設定:

Decay: 0.8-1.5 s
Size: 30-50%
Pre-Delay: 10-20 ms

ジャンル:
全ジャンル

Plate (プレート):

特徴:
金属板残響
明るい
密度高い

用途:
Snare・Vocal
クラシック

設定:

Decay: 1.5-2.5 s
Size: 50%
Pre-Delay: 10-30 ms

ジャンル:
Rock・Pop・Techno

Chamber (チェンバー):

特徴:
反射多い
複雑
ビンテージ

用途:
Vocal・Drums
60年代風

設定:

Decay: 1.0-2.0 s
Size: 40-60%

Spring (スプリング):

特徴:
バネ残響
ギターアンプ的

用途:
実験的

設定:
特殊用途
```

### パラメーター詳細

```
1. Pre-Delay:

機能:
Dry音からReverb開始までの遅延

単位:
ms (ミリ秒)

設定:

0 ms:
すぐReverb
密着

20-40 ms:
わずかに遅延
分離良い

60+ ms:
明確に遅延
スラップバック的

推奨:

Vocal: 20-30 ms (分離)
Snare: 15-25 ms
Pad: 10-20 ms
Lead: 25-40 ms

理由:
Dry音明確
Reverbと分離

2. Decay Time:

機能:
残響時間

単位:
s (秒)

設定:

短い (0.5-1.0 s):
タイト
近い

中間 (1.5-2.5 s):
標準
自然

長い (3.0-5.0 s):
広大
壮大

推奨:

Techno Snare: 1.8-2.2 s
House Vocal: 2.0-2.5 s
Pad: 3.0-4.0 s
Drums (Room): 0.8-1.2 s

テンポ連動:

BPM 128:
Decay: 1.8-2.2 s
ちょうど良い

BPM 140:
Decay: 1.5-1.8 s
短め

理由:
テンポ速い
→ Decay短く

3. Size:

機能:
空間サイズ

単位:
% (0-100%)

設定:

小さい (20-40%):
タイト
近い

中間 (50-70%):
標準

大きい (80-100%):
広大
壮大

推奨:

Drums: 30-50%
Vocal: 50-70%
Pad: 70-90%

4. Diffusion:

機能:
反射密度

設定:

低い (30-50%):
反射が個別に聴こえる
クリア

高い (70-100%):
滑らか
密

推奨:

Drums: 60-70%
Vocal: 80-90% (滑らか)

5. Damping:

機能:
高域減衰周波数

単位:
Hz

設定:

低い (2000-4000 Hz):
暗いReverb
自然

高い (8000+ Hz):
明るいReverb
人工的

推奨:

Techno: 4000-6000 Hz (暗め)
House: 6000-8000 Hz
Vocal: 6000-7000 Hz

理由:
現実空間は高域減衰
→ 暗めが自然

6. Dry/Wet:

機能:
ミックス比率

設定:

Individual Track:
10-30%
控えめ

Return Track:
100% (Wet Only)
Sendで調整

推奨:
Return Track使用
```

---

## Reverb実践例

**トラック別設定:**

### Techno Snare/Clap

```
目標:
広い、長い残響

Type: Hall

Pre-Delay: 20 ms
Decay Time: 2.0 s
Size: 70%
Diffusion: 75%
Damping: 5000 Hz

Return Track:

EQ Eight (Post):
High Pass: 400 Hz (低域カット)
Peak: -3 dB @ 800 Hz (濁り除去)
High Cut: 10 kHz (暗く)

Send量:
25-35%

結果:
広く、深いTechno Clap
```

### House Vocal

```
目標:
自然、前に出る

Type: Plate

Pre-Delay: 25 ms (分離)
Decay Time: 2.3 s
Size: 60%
Diffusion: 85% (滑らか)
Damping: 6500 Hz

Return Track:

EQ Eight:
High Pass: 300 Hz
Peak: +2 dB @ 4 kHz (明瞭)

Compressor:
Ratio: 3:1
Threshold: -15 dB
(Reverb制御)

Send量:
20-30%

結果:
明瞭、深いVocal
```

### Pad (Ambient)

```
目標:
壮大、後ろに

Type: Hall

Pre-Delay: 15 ms
Decay Time: 3.5 s (長い)
Size: 85%
Diffusion: 90%
Damping: 4000 Hz (暗い)

Return Track:

EQ Eight:
High Pass: 500 Hz (低域大幅カット)
High Cut: 8 kHz (暗く)

Send量:
40-60% (多め)

結果:
後ろの広大な空間
```

### Drums (Room)

```
目標:
自然な空間

Type: Room

Pre-Delay: 10 ms
Decay Time: 1.0 s (短い)
Size: 35%
Diffusion: 65%
Damping: 7000 Hz

Send量:
Kick: 0%
Snare: 15%
HH: 8%
Perc: 20%

結果:
自然、タイトなドラム
```

---

## Delay 完全ガイド

**やまびこエフェクト:**

### Simple Delay

```
┌─────────────────────────────────┐
│  Simple Delay                   │
├─────────────────────────────────┤
│  Time: [1/8 ▼] Sync: On         │
│  Feedback: 40%                  │
│  Dry/Wet: 20%                   │
│                                 │
│  [Filter] [Ping Pong]           │
└─────────────────────────────────┘

基本パラメーター:
- Time (遅延時間)
- Feedback (繰り返し)
- Dry/Wet (ミックス)
```

### パラメーター詳細

```
1. Time:

機能:
遅延時間

設定:

Sync: On (推奨)
音符単位

1/4: 4分音符
1/8: 8分音符
1/16: 16分音符
1/4 Dotted: 付点4分
1/8 Dotted: 付点8分

Sync: Off
ms単位
自由設定

推奨:

Techno/House:
1/8 または 1/16
リズミカル

Ambient:
1/4 Dotted
広がり

理由:
テンポ同期
グルーヴ維持

2. Feedback:

機能:
繰り返し回数

単位:
% (0-100%)

設定:

低い (20-30%):
1-2回繰り返し
控えめ

中間 (40-60%):
3-5回
標準

高い (70-90%):
多数回
実験的

推奨:

Vocal: 30-40%
Lead: 40-50%
FX: 60-80%

注意:
90%+: 無限ループ
発振

3. Dry/Wet:

機能:
ミックス比率

設定:

Individual: 10-20%
Return: 100% (Wet)

推奨:
Return Track

4. Stereo Mode:

Ping Pong:

機能:
左右交互

設定:

Offset: 100%
完全に左右

Offset: 50%
やや左右

用途:
広がり
House的

Stereo:

機能:
両側同時

用途:
Techno
タイト
```

### Filter Delay

```
┌─────────────────────────────────┐
│  Filter Delay                   │
├─────────────────────────────────┤
│  Time L: 1/8                    │
│  Time R: 1/4                    │
│  Feedback: 50%                  │
│                                 │
│  Filter Cutoff: 3000 Hz         │
│  Resonance: 20%                 │
│                                 │
│  Dry/Wet: 25%                   │
└─────────────────────────────────┘

特徴:
フィルター内蔵
Delay徐々に暗く

用途:
Techno・創造的
```

---

## Delay実践例

**用途別設定:**

### Rhythmic Delay (Techno/House)

```
目標:
リズミカル、装飾

設定:

Type: Simple Delay
Time: 1/8
Sync: On
Feedback: 40%
Ping Pong: On
Offset: 80%

Return Track:

Filter:
High Pass: 500 Hz (低域カット)
Low Pass: 6000 Hz (暗く)

Compressor:
Ratio: 4:1
Threshold: -12 dB

Dry/Wet: 100% (Return)

Send量:
Vocal: 15-20%
Lead: 20-25%
HH: 10%

結果:
リズミカル、広がり
```

### Dotted Delay (創造的)

```
目標:
The Edge (U2) スタイル

設定:

Time: 1/4 Dotted
Sync: On
Feedback: 45%
Ping Pong: Off

Return Track:

Reverb (追加):
Plate, Decay 1.5s
並列処理

Send量:
Lead: 25-30%

結果:
複雑なリズム
広がり
```

### Dub Delay

```
目標:
長いフィードバック

設定:

Time: 1/4
Feedback: 65% (高め)
Filter Cutoff: 2000 Hz (暗い)

Return Track:

EQ Eight:
High Pass: 400 Hz
High Cut: 4000 Hz (暗く)

Saturator:
Drive: 5 dB (温かみ)

Send量:
Snare: 20%
Perc: 30%

結果:
Dub的長い残響
温かい
```

---

## Echo (Tape Delay)

**ビンテージ・テープディレイ:**

### 特徴

```
Echo vs Simple Delay:

Simple Delay:
デジタル
クリーン
正確

Echo:
テープエミュレート
ワウフラッター
温かい

用途:
ビンテージ
実験的
Dub
```

### パラメーター

```
Time:

1/4〜1/2
長め推奨

Feedback:

50-70%
多め

Modulation:

Rate: 0.5-1.0 Hz
ワウフラッター
テープ的

Gate:

Threshold:
小さい音カット

Ducking:

Input信号大きい時
Delay下げる
自動ダッキング

推奨設定:

Time: 1/4
Feedback: 60%
Modulation Rate: 0.7 Hz
Amount: 15%

結果:
温かいビンテージDelay
```

---

## Return Track戦略

**効率的な空間処理:**

### 標準的な4つのReturn

```
Return A: Main Reverb (Hall)

Reverb:
Type: Hall
Decay: 2.5 s
Size: 70%
Pre-Delay: 20 ms

EQ Eight:
High Pass: 300 Hz
High Cut: 10 kHz

Compressor:
Ratio: 3:1
Threshold: -15 dB

用途:
メインの空間
全トラック共有

Return B: Room Reverb

Reverb:
Type: Room
Decay: 1.0 s
Size: 40%

EQ Eight:
High Pass: 400 Hz

用途:
近い空間
Drums

Return C: Rhythmic Delay (1/8)

Simple Delay:
Time: 1/8
Feedback: 40%
Ping Pong: On

Filter:
High Pass: 500 Hz
Low Pass: 6000 Hz

用途:
リズミカル装飾

Return D: Creative Delay (1/4 Dotted)

Filter Delay:
Time: 1/4 Dotted
Feedback: 50%
Cutoff: 3000 Hz

用途:
実験的
装飾的
```

### Send量ガイドライン

```
Kick:

Reverb: 0% (タイト維持)
Delay: 0%

Bass:

Reverb: 0-5% (わずか)
Delay: 0%

理由: 低域濁る

Snare/Clap:

Reverb A: 25-35% (広い)
Reverb B: 10-15% (近い)
Delay C: 15-20%

Vocal:

Reverb A: 20-30%
Delay C: 15-20%
Delay D: 10% (装飾)

Lead:

Reverb A: 15-25%
Delay C: 20-30%

Pad:

Reverb A: 40-60% (多め)
Delay: 10-20%

Hi-Hat:

Reverb B: 8-12% (Room)
Delay: 5-10%

Percussion:

Reverb A: 25-35%
Delay: 20-30%
```

---

## よくある失敗

### 1. 低域にReverb

```
問題:
低域濁る
ミックス崩壊

原因:
Kick・BassにReverb

解決:

Return Track:
EQ Eight追加
High Pass: 300-500 Hz

Send量:
Kick: 0%
Bass: 0-5%

効果:
低域クリア
高域のみ空間
```

### 2. Decay Time長すぎ

```
問題:
音が遠い
埋もれる

原因:
Decay: 4.0 s+

解決:

Decay: 1.5-2.5 s
標準

テンポ連動:
BPM高い → Decay短く

ルール:
「少し足りない」
```

### 3. Pre-Delayなし

```
問題:
Dry音埋もれる
分離悪い

原因:
Pre-Delay: 0 ms

解決:

Pre-Delay: 20-30 ms
分離良い

特にVocal・Lead:
必須
```

### 4. Delayかかりすぎ

```
問題:
うるさい
濁る

原因:
Feedback: 70%+
Send: 40%+

解決:

Feedback: 30-50%
Send: 15-25%

ルール:
「わずかに聴こえる」
```

---

## ジャンル別設定

**Techno:**

```
特徴:
暗い
タイト
Reverb控えめ

推奨:

Reverb:
Type: Plate
Decay: 1.8-2.2 s
Damping: 4000-5000 Hz (暗い)

Delay:
1/8 または 1/16
Feedback: 35-45%

Send量:
全体的に控えめ
Snare: 25%
他: 10-15%
```

**House:**

```
特徴:
明るい
広がり
Vocal中心

推奨:

Reverb:
Type: Hall・Plate
Decay: 2.0-2.8 s
Damping: 6000-7000 Hz

Delay:
1/4 Dotted
Ping Pong
Feedback: 40-50%

Send量:
Vocal: 25-30%
Snare: 20-25%
広がり重視
```

---

## 実践ワークフロー

**30分練習:**

### Week 1: Reverb

```
Day 1 (15分):

1. Return A作成
   Reverb (Hall)

2. 設定:
   Decay: 2.5 s
   Size: 70%
   Pre-Delay: 20 ms

3. EQ追加:
   High Pass: 300 Hz

4. Send:
   Snare 25%

Day 2-3:

全トラックSend調整

Day 4-7:

Return B (Room)
各トラック最適化
```

### Week 2: Delay

```
Day 1 (15分):

1. Return C作成
   Simple Delay (1/8)

2. Feedback: 40%
   Ping Pong: On

3. Filter:
   High Pass: 500 Hz

Day 2-4:

各トラックSend
バランス調整

Day 5-7:

Return D (1/4 Dotted)
創造的使用
```

---

## まとめ

### Reverb

```
□ Type: Hall・Room・Plate使い分け
□ Decay: 1.5-2.5 s (標準)
□ Pre-Delay: 20-30 ms (分離)
□ Return TrackにEQ必須 (High Pass)
□ 低域にはかけない
```

### Delay

```
□ Time: 1/8または1/4 Dotted
□ Feedback: 30-50%
□ Ping Pong: 広がり
□ Filter: 暗く (Low Pass)
□ Return Track活用
```

### Return Track

```
□ 最低4つ推奨
  A: Hall, B: Room, C: Delay 1/8, D: Delay 1/4D
□ 全てにEQ (High Pass)
□ Compressor追加で制御
□ Dry/Wet: 100%
```

### 重要原則

```
□ Less is More
□ 低域濁り防止
□ Pre-Delay必須
□ A/B比較
□ テンポ連動
```

---

**次は:** [Distortion・Saturation](./distortion-saturation.md) - 歪み系で存在感と温かみを付加
