# Depth & Space

Reverb・Delayで奥行きを作ります。前後配置・Pre-Delay・Return Track戦略を完全マスターします。

## この章で学ぶこと

- 奥行きの作り方
- Return Track活用
- Pre-Delay重要性
- Send量調整
- 前中後配置戦略
- 低域Reverb問題
- EQとReverbの組み合わせ

---

## なぜDepth & Spaceが重要なのか

**立体感の実現:**

```
奥行きなし:

特徴:
平坦
2次元
単調

奥行きあり:

特徴:
立体的
3次元
没入感

プロとアマの差:

アマ:
全て同じ距離
平坦

プロ:
前中後明確
立体的

真実:

「プロの音」=
奥行きが完璧

方法:
Reverb・Delay
Pre-Delay設定
```

---

## 奥行きの原理

**音響心理学:**

### 距離感の要因

```
要因1: 音量

大きい:
近い

小さい:
遠い

要因2: 明るさ

明るい (高域多い):
近い

暗い (高域少ない):
遠い

理由:
空気吸収
高域減衰

要因3: Reverb量

Dry (Reverb少ない):
近い

Wet (Reverb多い):
遠い

要因4: Pre-Delay

短い (0-10 ms):
近い
密着

長い (20-40 ms):
適度な距離
分離

結論:

近く:
大きい、明るい、Dry、Pre-Delay短い

遠く:
小さい、暗い、Wet、Pre-Delay長い
```

---

## 前中後配置

**3層構造:**

### 最前列

```
楽器:

Kick:
最も前

Vocal:
Kick並び

Bass:
やや後ろ

Snare:
前

設定:

Reverb Send: 0-10%
Pre-Delay: 15-20 ms
EQ: 明るい (+2 dB @ 3 kHz)
Volume: 大きい (-6 dB)

理由:
パワー
存在感
```

### 中間

```
楽器:

Lead:
前寄り

Keys:
中間

Percussion:
中間

設定:

Reverb Send: 20-35%
Pre-Delay: 20-30 ms
EQ: 適度
Volume: 中程度 (-12 dB)

理由:
メロディー
主要要素
```

### 後列

```
楽器:

Pad:
後ろ

FX:
最も後ろ

Ambience:
背景

設定:

Reverb Send: 40-60%
Pre-Delay: 10-20 ms (短め)
EQ: 暗い (-3 dB @ 6 kHz)
Volume: 小さい (-18 dB)

理由:
背景
空間演出
```

---

## Return Track戦略

**効率的な空間処理:**

### 標準4 Returns

```
Return A: Main Reverb (Hall)

用途:
メイン空間
全般

Reverb:
Type: Hall
Decay: 2.5 s
Size: 70%
Pre-Delay: 25 ms

EQ (Post):
High Pass: 300 Hz
High Cut: 10 kHz

推奨Send:
Snare: 30%
Lead: 25%
Vocal: 25%

Return B: Room Reverb

用途:
近い空間
Drums

Reverb:
Type: Room
Decay: 1.0 s
Size: 40%
Pre-Delay: 15 ms

EQ (Post):
High Pass: 400 Hz

推奨Send:
Drums: 15%

Return C: Delay (1/8)

用途:
リズミカル装飾

Simple Delay:
Time: 1/8
Feedback: 40%
Ping Pong: On

Filter:
High Pass: 500 Hz
Low Pass: 6 kHz

推奨Send:
Vocal: 20%
Lead: 25%

Return D: Delay (1/4 Dotted)

用途:
創造的

Filter Delay:
Time: 1/4 Dotted
Feedback: 50%

推奨Send:
Lead: 15%
FX: 30%
```

---

## Pre-Delay完全ガイド

**分離の鍵:**

### なぜ重要？

```
Pre-Delay なし (0 ms):

Dry音:
すぐReverb

結果:
密着
分離悪い
Dry埋もれる

Pre-Delay あり (20-30 ms):

Dry音:
明確

その後:
Reverb

結果:
分離良い
Dry音明確

推奨値:

Kick: 20 ms
Bass: 25 ms (少ないSend)
Vocal: 25-30 ms (最重要)
Lead: 25 ms
Snare: 20 ms
Pad: 10-15 ms

ルール:
前に出したい = 長い Pre-Delay
後ろに = 短い Pre-Delay
```

---

## Send量調整

**トラック別推奨:**

### 詳細設定

```
Kick:

Reverb: 0%
Delay: 0%

理由:
タイト維持
パワー

Bass:

Reverb: 0-5%
Delay: 0%

理由:
低域濁る
わずかに許容

Snare/Clap:

Reverb A: 25-35%
Reverb B: 10-15%
Delay C: 15-20%

理由:
広がり
Techno/House必須

Vocal:

Reverb A: 20-30%
Delay C: 15-20%
Delay D: 10% (装飾)

理由:
深み
自然

Lead:

Reverb A: 20-30%
Delay C: 25-35%

理由:
空間
動き

Pad:

Reverb A: 40-60%
Delay: 10-20%

理由:
壮大
後ろに

Hi-Hat:

Reverb B: 8-12% (Room)
Delay: 5-10%

理由:
わずかな空間

Percussion:

Reverb A: 25-35%
Delay: 20-30%

理由:
装飾
```

---

## 低域Reverb問題

**濁り防止:**

### 原因

```
問題:

Kick・Bass:
Reverbあり

結果:
低域濁る
パワーダウン

理由:

Reverb:
全周波数に残響

低域残響:
濁りの原因
```

### 解決法

```
方法1: Sendなし

Kick・Bass:
Send: 0%

最も確実

方法2: Return Track EQ

Return A-D全て:

EQ Eight挿入
High Pass: 300-500 Hz

効果:
低域Reverbなし
高域のみ残響

推奨:
High Pass 400 Hz

方法3: Pre-EQ

Kick・Bassトラック:

Send前:
High Pass 200 Hz (Send専用Chain)

理由:
低域送らない

推奨:
方法2 (Return EQ)
最も実用的
```

---

## Decay Time設定

**テンポ連動:**

### BPM別推奨

```
BPM 128 (Techno/House):

Hall:
Decay: 2.0-2.5 s

Room:
Decay: 1.0-1.2 s

理由:
ちょうど良い

BPM 140 (Dubstep):

Hall:
Decay: 1.5-2.0 s

理由:
速いテンポ
短めが良い

BPM 90 (Hip Hop):

Hall:
Decay: 2.5-3.5 s

理由:
遅いテンポ
長めOK

計算式:

Decay (s) = 60 / BPM × 4
目安

例:
BPM 128
60 / 128 × 4 = 1.875 s
→ 約 2.0 s
```

---

## Delay Timeテンポ連動

**リズミカル設定:**

### 音符単位

```
1/4 (4分音符):

BPM 128:
468.75 ms

用途:
ゆっくり
Dub

1/8 (8分音符):

BPM 128:
234.375 ms

用途:
標準
Techno/House

1/16 (16分音符):

BPM 128:
117.1875 ms

用途:
速い
Hi-Hat

1/4 Dotted (付点4分):

BPM 128:
703.125 ms

用途:
The Edge風
創造的

推奨:

Techno/House:
1/8 または 1/16

Ambient:
1/4 Dotted

Sync: On
必須
```

---

## Parallel Reverb

**より自然:**

### 概念

```
通常Reverb:

Individual Track:
Dry/Wet 30%

問題:
Dry減る

Parallel (Return):

Dry: 100% (そのまま)
Wet: Send量調整

メリット:
Dry維持
Reverb追加

推奨:
必ずReturn Track使用
Individual Reverb不使用
```

---

## よくある失敗

### 1. 低域にReverb

```
問題:
濁る

原因:
Kick・BassにSend

解決:

Kick・Bass:
Send: 0%

Return Track:
High Pass 400 Hz

効果:
劇的にクリア
```

### 2. Pre-Delayなし

```
問題:
Dry埋もれる

原因:
Pre-Delay: 0 ms

解決:

Pre-Delay:
20-30 ms設定

特にVocal:
必須

効果:
分離良い
```

### 3. Decay長すぎ

```
問題:
濁る
音が遠い

原因:
Decay: 4.0 s+

解決:

Decay:
1.5-2.5 s (標準)

テンポ連動:
BPM高い → 短く

効果:
クリア
```

### 4. Send過剰

```
問題:
全体が遠い

原因:
全トラック Send 50%+

解決:

推奨Send:
Vocal・Lead: 20-30%
Pad: 40-60%
Kick・Bass: 0%

ルール:
「少し足りない」
```

---

## Reverb種類使い分け

**Type選択:**

### 状況別

```
Hall:

特徴:
大きい空間
長い残響

用途:
Pad・Lead・Snare
壮大

Decay: 2.0-3.0 s

Room:

特徴:
小さい空間
自然

用途:
Drums全般
リアル

Decay: 0.8-1.5 s

Plate:

特徴:
金属板
明るい
密度高い

用途:
Snare・Vocal
クラシック

Decay: 1.5-2.5 s

Chamber:

特徴:
反射多い
ビンテージ

用途:
Vocal (60年代風)

Decay: 1.0-2.0 s
```

---

## Convolution Reverb

**リアルな空間:**

### IR (Impulse Response)

```
機能:

実際の空間:
録音

再現:
その空間のReverb

メリット:

リアル:
本物の空間

高品質:
自然

用途:

Orchestra Hall:
クラシック

Abbey Road:
ビンテージ

Club:
Techno/House

Ableton:

なし (標準)

推奨:
付属Reverbで十分

上級:
Altiverb, Space等
```

---

## 実践ワークフロー

**30分で完成:**

### Step-by-Step

```
0-10分: Return Track設定

1. Return A: Hall, Decay 2.5s
2. EQ High Pass 400 Hz
3. Return B: Room, Decay 1.0s
4. Return C: Delay 1/8
5. Return D: Delay 1/4D

10-20分: Send量調整

1. Kick・Bass: 0%
2. Snare: 30%
3. Vocal: 25%
4. Lead: 25%
5. Pad: 50%
6. Hi-Hat: 10%

20-25分: Pre-Delay

全Return:
Pre-Delay 20-30 ms設定

25-30分: 確認

1. 全体聴き直し
2. 奥行き確認
3. 低域濁りチェック
4. 微調整
```

---

## まとめ

### Depth & Space

```
□ Return Track活用 (最低2つ)
□ Pre-Delay 20-30 ms必須
□ 低域Reverb厳禁 (High Pass 400 Hz)
□ Decay Time テンポ連動
□ Send量控えめ
```

### 配置

```
最前列: Kick・Vocal (Send 0-10%)
中間: Lead・Keys (Send 20-35%)
後列: Pad・FX (Send 40-60%)
```

### 重要原則

```
□ Kick・Bass Send 0%
□ Return Track EQ必須
□ Pre-Delay で分離
□ Individual Reverb不使用
□ 「少し足りない」が正解
```

---

**次は:** [Automation](./automation.md) - 動的ミックスで時間軸の変化を作る
