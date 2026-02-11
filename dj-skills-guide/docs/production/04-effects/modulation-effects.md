# Modulation Effects

Chorus・Flanger・Phaserで音を動かします。モジュレーション系エフェクトを完全マスターし、厚みと広がりを実現します。

## この章で学ぶこと

- Chorus（厚み・広がり）
- Flanger（ジェット音・うねり）
- Phaser（スイープ・位相変化）
- Auto Pan（左右移動）
- Rate・Depth・Feedback調整
- LFO（Low Frequency Oscillator）基礎
- トラック別活用法

---

## なぜModulationが重要なのか

**動きと生命感:**

```
Modulationなし:

特徴:
静的
平坦
単調

Modulationあり:

特徴:
動的
立体的
生命感

使用頻度:

Chorus: 30-40%
Lead・Pad

Flanger: 15-20%
特殊効果

Phaser: 20-30%
実験的

Auto Pan: 10-15%
Hi-Hat・FX

プロの使用:

控えめ:
わずかに

効果:
気づかない動き

真実:

「生きてる音」=
わずかなModulation

静的: 機械的
動的: 人間的
```

---

## Chorus 完全ガイド

**厚みと広がり:**

### 基本原理

```
原理:

元音:
100% (Center)

コピー:
わずかにDetune
わずかに遅延

LFO:
周期的に変化

結果:
2人が演奏
厚み・広がり

視覚化:

元音: ─────────────
Copy: ～～～～～～～～～
      (揺れる)

合計: 厚い、広い
```

### インターフェイス

```
┌─────────────────────────────────┐
│  Chorus                         │
├─────────────────────────────────┤
│  Rate: 0.50 Hz                  │
│  Amount: 25%                    │
│  Delay: 20.0 ms                 │
│                                 │
│  Feedback: 0%                   │
│  Polarity: [+]                  │
│                                 │
│  Dry/Wet: 30%                   │
│                                 │
│  [High Quality]                 │
└─────────────────────────────────┘

重要パラメーター:
- Rate (変化速度)
- Amount (変化量)
- Delay (遅延時間)
```

### パラメーター詳細

```
1. Rate:

機能:
LFO速度

単位:
Hz

設定:

遅い (0.1-0.5 Hz):
ゆっくり
自然

中間 (0.5-2.0 Hz):
標準

速い (2.0-5.0 Hz):
揺れすぎ
不自然

推奨:

Pad: 0.3 Hz (ゆっくり)
Lead: 0.5 Hz
Keys: 0.7 Hz

理由:
遅いほど自然

2. Amount (Depth):

機能:
変化量

単位:
% (0-100%)

設定:

わずか (10-20%):
自然
控えめ

中間 (20-40%):
標準
明確

多い (50%+):
強烈
不自然

推奨:

Pad: 25%
Lead: 30%
Vocal: 15% (控えめ)

3. Delay (Time):

機能:
初期遅延

単位:
ms

設定:

短い (5-15 ms):
タイト
Chorusらしい

中間 (15-30 ms):
標準

長い (30+ ms):
Flangerっぽい

推奨:

標準: 20 ms
Pad: 25 ms

4. Feedback:

機能:
フィードバック量

単位:
% (-100 〜 +100%)

設定:

0%:
Chorusらしい
推奨

30%+:
Flangerっぽい
実験的

推奨:
0% (Chorus)
30-60% (Flanger的)

5. Polarity:

+ (正):
標準

- (負):
位相反転
やや違う質感

推奨:
+ (標準)

6. Dry/Wet:

機能:
ミックス比率

設定:

Individual: 20-40%
Return: 100%

推奨:
20-30%
控えめ
```

---

## Chorus実践例

**トラック別設定:**

### Lead Synth

```
目標:
厚み、広がり

設定:

Rate: 0.5 Hz
Amount: 30%
Delay: 20 ms
Feedback: 0%
Polarity: +
Dry/Wet: 25%

Chain:

1. EQ Eight:
   High Pass 200 Hz

2. Chorus:
   上記設定

3. Reverb (Return):
   Send 20%

結果:
厚く、広いLead
```

### Pad

```
目標:
広大、ゆっくり

設定:

Rate: 0.3 Hz (遅い)
Amount: 25%
Delay: 25 ms
Feedback: 0%
Dry/Wet: 35%

追加:

Chorus (2台):

1台目: Rate 0.3 Hz
2台目: Rate 0.47 Hz
(わずかに違う)

効果:
複雑な動き
自然

結果:
広大なPad
```

### Electric Piano (Rhodes)

```
目標:
クラシックな質感

設定:

Rate: 0.7 Hz
Amount: 20% (控えめ)
Delay: 15 ms (短め)
Feedback: 0%
Dry/Wet: 30%

Chain:

Chorus → Reverb (Plate)

結果:
80年代的質感
```

### Vocal (House)

```
目標:
わずかな厚み

設定:

Rate: 0.4 Hz
Amount: 15% (軽め)
Delay: 18 ms
Feedback: 0%
Dry/Wet: 20% (控えめ)

注意:
Vocalには控えめ
過剰: 不自然

結果:
わずかな広がり
```

---

## Flanger 完全ガイド

**ジェット音・うねり:**

### 基本原理

```
原理:

Chorusに似る

違い:

Delay: 非常に短い (0.5-10 ms)
Feedback: 多い (30-70%)

結果:
コムフィルター
特徴的うねり

効果:
ジェット音
「シュワー」

用途:
特殊効果
Dubstep・EDM
```

### インターフェイス

```
┌─────────────────────────────────┐
│  Flanger                        │
├─────────────────────────────────┤
│  Rate: 0.30 Hz                  │
│  Amount: 50%                    │
│  Delay: 3.0 ms                  │
│                                 │
│  Feedback: 50%                  │
│  Polarity: [+]                  │
│                                 │
│  Dry/Wet: 30%                   │
│                                 │
│  [High Quality] [Envelope]      │
└─────────────────────────────────┘

Chorusとの違い:
- Delay短い
- Feedback多い
```

### パラメーター

```
1. Rate:

Flanger特有:

遅い (0.1-0.5 Hz):
ゆっくりうねり
標準

中間 (0.5-2.0 Hz):
明確なうねり

速い (2.0+ Hz):
ビブラート的

推奨:

Pad: 0.2 Hz (ゆっくり)
Lead: 0.4 Hz
Drums: 0.5 Hz

2. Amount:

Flanger:

30-70%
多めOK

効果:
多い = 強烈なうねり

3. Delay:

Flanger特有:

0.5-5 ms:
Flangerらしい

5-10 ms:
強烈

推奨:
2-4 ms

4. Feedback:

Flanger必須:

30-60%:
標準

70%+:
強烈
実験的

0%:
Chorusっぽい

推奨:
40-60%

5. Envelope:

機能:
Input信号でModulation

用途:
ダイナミック
```

---

## Flanger実践例

**用途別設定:**

### Techno Pad (Dark)

```
目標:
暗いうねり

設定:

Rate: 0.25 Hz (遅い)
Amount: 60%
Delay: 3.5 ms
Feedback: 55%
Polarity: -
Dry/Wet: 40%

Post処理:

EQ Eight:
High Cut 8 kHz (暗く)

結果:
暗い、深いうねり
Techno的
```

### Dubstep Buildup

```
目標:
強烈なうねり

設定:

Rate: 1.0 Hz (速め)
Amount: 80% (強烈)
Delay: 5 ms
Feedback: 70%
Dry/Wet: 60%

Automation:

Rate: 0.2 Hz → 2.0 Hz
8小節で上昇

Feedback: 40% → 80%
強烈に

結果:
ビルドアップ効果
```

### Hi-Hat (実験的)

```
目標:
動き

設定:

Rate: 0.4 Hz
Amount: 40%
Delay: 2 ms (短い)
Feedback: 45%
Dry/Wet: 25% (控えめ)

結果:
わずかなうねり
質感追加
```

---

## Phaser 完全ガイド

**位相変化・スイープ:**

### 基本原理

```
原理:

All-Pass Filter:
位相のみ変化
音量変わらない

LFO:
Filterを動かす

結果:
特徴的スイープ
「シュワシュワ」

Flanger vs Phaser:

Flanger:
Delay使用
強烈

Phaser:
Filter使用
自然

用途:
Flanger: 実験的
Phaser: 自然
```

### インターフェイス

```
┌─────────────────────────────────┐
│  Phaser                         │
├─────────────────────────────────┤
│  Poles: 4                       │
│  Frequency: 1000 Hz             │
│                                 │
│  Rate: 0.40 Hz                  │
│  Amount: 50%                    │
│                                 │
│  Feedback: 40%                  │
│  Dry/Wet: 30%                   │
│                                 │
│  [Envelope] [Spin]              │
└─────────────────────────────────┘

Phaser特有:
- Poles (次数)
- Frequency (中心周波数)
```

### パラメーター

```
1. Poles:

機能:
フィルター段数

設定:

2 Poles:
軽い
自然

4 Poles:
標準

8 Poles:
強烈
実験的

推奨:

Pad: 4 Poles
Lead: 4-8 Poles
Drums: 2-4 Poles

2. Frequency:

機能:
中心周波数

単位:
Hz

設定:

低い (200-500 Hz):
太い
暗い

中間 (500-2000 Hz):
標準

高い (2000-5000 Hz):
明るい
軽い

推奨:

Bass: 400 Hz
Lead: 1000 Hz
Hi-Hat: 3000 Hz

3. Rate:

0.2-1.0 Hz
Chorusと同様

4. Amount:

30-70%
変化量

5. Feedback:

20-60%
深さ

6. Spin (L/R):

機能:
左右で位相差

効果:
ステレオ広がり

推奨:
50-100%
広がり
```

---

## Phaser実践例

**トラック別設定:**

### Electric Piano

```
目標:
70年代Funk

設定:

Poles: 4
Frequency: 800 Hz
Rate: 0.5 Hz
Amount: 60%
Feedback: 45%
Spin: 80%
Dry/Wet: 35%

結果:
クラシックなPhaser
Funk的
```

### Techno Lead

```
目標:
動き

設定:

Poles: 8 (強烈)
Frequency: 1200 Hz
Rate: 0.3 Hz
Amount: 55%
Feedback: 50%
Dry/Wet: 30%

結果:
深い動き
```

### Strings

```
目標:
自然な動き

設定:

Poles: 2 (軽い)
Frequency: 600 Hz
Rate: 0.2 Hz (遅い)
Amount: 35% (控えめ)
Feedback: 30%
Dry/Wet: 20%

結果:
わずかな動き
自然
```

---

## Auto Pan 完全ガイド

**左右移動:**

### 基本原理

```
機能:
音量を左右で変化

LFO:
Pan位置を動かす

結果:
左右移動
ステレオ動き

用途:
Hi-Hat・Perc・FX
```

### インターフェイス

```
┌─────────────────────────────────┐
│  Auto Pan                       │
├─────────────────────────────────┤
│  Shape: [Sine ▼]                │
│  Rate: [1/8 ▼] Sync: On         │
│  Amount: 50%                    │
│                                 │
│  Phase: 180°                    │
│  Offset: 0%                     │
│                                 │
│  Invert: Off                    │
└─────────────────────────────────┘

重要:
- Shape (波形)
- Rate (速度)
- Amount (深さ)
```

### パラメーター

```
1. Shape:

Sine:
滑らか
標準

Triangle:
リニア

Square:
急激
左右切り替え

Random:
ランダム

推奨:

Hi-Hat: Sine
Perc: Triangle
FX: Random

2. Rate:

Sync: On (推奨)

設定:

1/16: 速い
1/8: 標準
1/4: ゆっくり

用途:

Techno/House:
1/8 または 1/16

3. Amount:

機能:
Pan幅

設定:

30%: わずか
50%: 標準
100%: 完全に左右

推奨:

Hi-Hat: 40%
Perc: 60%
FX: 80%

4. Phase:

L/R位相差

180°: 逆位相
最も広い

5. Offset:

Pan中心位置

0%: Center
±50%: 左右オフセット
```

---

## Auto Pan実践例

**トラック別設定:**

### Hi-Hat

```
目標:
わずかな動き

設定:

Shape: Sine
Rate: 1/8
Sync: On
Amount: 35% (控えめ)
Phase: 180°
Offset: 0%

結果:
わずかな左右移動
広がり
```

### Shaker

```
目標:
リズミカル移動

設定:

Shape: Triangle
Rate: 1/16 (速い)
Amount: 60%
Phase: 180°

結果:
リズミカル
ステレオ広い
```

### FX (Riser)

```
目標:
激しい動き

設定:

Shape: Random
Rate: 1/8
Amount: 80% (多め)

Automation:

Amount: 20% → 100%
8小節で上昇

結果:
ビルドアップ
激しい動き
```

---

## よくある失敗

### 1. Rateが速すぎ

```
問題:
不自然
うるさい

原因:
Rate: 2.0 Hz+

解決:

Chorus/Flanger/Phaser:
Rate: 0.2-0.7 Hz
ゆっくり

理由:
遅いほど自然
```

### 2. Amountが多すぎ

```
問題:
揺れすぎ
不自然

原因:
Amount: 70%+

解決:

Chorus: 20-35%
Flanger: 40-60%
Phaser: 30-50%

ルール:
「気づかない程度」
```

### 3. 全てにChorus

```
問題:
全体が揺れる
定位不明

原因:
全トラックにChorus

解決:

Chorusは一部のみ:
Lead・Pad・Keys

Kick・Bass:
なし

理由:
低域: タイト維持
```

### 4. FeedbackでFlangerに

```
問題:
Chorusのつもりが
Flangerに

原因:
Feedback: 50%+

解決:

Chorus:
Feedback: 0%

Flanger:
Feedback: 40-60%

明確に区別
```

---

## ジャンル別活用

**Techno:**

```
Flanger:
Pad・Lead
暗いうねり

設定:
Rate: 0.2-0.3 Hz (遅い)
Amount: 50-60%
Feedback: 50%

Phaser:
控えめ

Chorus:
あまり使わない
```

**House:**

```
Chorus:
Vocal・Keys
厚み

設定:
Rate: 0.5 Hz
Amount: 25%

Auto Pan:
Hi-Hat・Perc
広がり

Flanger:
実験的装飾
```

---

## 実践ワークフロー

**30分練習:**

### Week 1: Chorus

```
Day 1 (10分):

1. Leadトラック
2. Chorus挿入
3. Rate: 0.5 Hz
4. Amount: 30%
5. Dry/Wet: 25%
6. A/B比較

Day 2-3:

Pad・Keys

Day 4-7:

全トラック最適化
```

### Week 2: Flanger・Phaser

```
Day 1-3:

Flanger:
Pad
Rate 0.3 Hz
Feedback 50%

Day 4-7:

Phaser:
Keys
Poles 4
```

---

## まとめ

### Chorus

```
□ Rate: 0.3-0.7 Hz (遅い)
□ Amount: 20-35%
□ Feedback: 0%
□ Dry/Wet: 20-30%
□ 控えめが自然
```

### Flanger

```
□ Delay: 2-4 ms (短い)
□ Feedback: 40-60% (多め)
□ Rate: 0.2-0.5 Hz
□ 実験的用途
```

### Phaser

```
□ Poles: 4 (標準)
□ Rate: 0.2-0.5 Hz
□ Feedback: 30-50%
□ 自然な動き
```

### Auto Pan

```
□ Rate: 1/8 または 1/16
□ Amount: 30-60%
□ Shape: Sine (標準)
□ Hi-Hat・Percに
```

### 重要原則

```
□ 遅いRate = 自然
□ 控えめAmount
□ 全てにかけない
□ A/B比較必須
□ わずかな動きが正解
```

---

**次は:** [Creative Effects](./creative-effects.md) - Filter・LFO・特殊効果で個性を出す
