# Modulation Effects

Chorus・Flanger・Phaserで音を動かします。モジュレーション系エフェクトを完全マスターし、厚みと広がりを実現します。

## この章で学ぶこと

- Chorus(厚み・広がり)
- Flanger(ジェット音・うねり)
- Phaser(スイープ・位相変化)
- Auto Pan(左右移動)
- Rate・Depth・Feedback調整
- LFO(Low Frequency Oscillator)基礎
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

## LFO(Low Frequency Oscillator)深掘り

**モジュレーションの心臓部:**

### LFOとは

```
定義:

Low Frequency Oscillator
低周波発振器

機能:
周期的な変化を生成

周波数範囲:
0.01 Hz - 20 Hz
(可聴域以下)

用途:

Modulation:
Chorus・Flanger・Phaser

Vibrato:
ピッチ変化

Tremolo:
音量変化

Filter Sweep:
フィルター移動

Pan:
左右移動

原理:

信号生成:
┌─────────┐
│   LFO   │ → 波形出力
└─────────┘
     ↓
パラメーターを制御
(Pitch/Volume/Pan等)
```

### LFO波形の種類

```
1. Sine (サイン波):

形状:
滑らか
自然

視覚:
    ╱╲
  ╱    ╲
╱        ╲╱

特徴:
最も自然
標準的

用途:
Chorus・Vibrato
自然な動き

推奨:
ほとんどの用途

2. Triangle (三角波):

形状:
直線的
リニア

視覚:
   ╱╲
  ╱  ╲
 ╱    ╲╱

特徴:
Sineより鋭い
リニア変化

用途:
Filter Sweep
Auto Pan

3. Square (矩形波):

形状:
急激
On/Off

視覺:
┌───┐   ┌───
│   │   │
    └───┘

特徴:
即座に切り替わる

用途:
Gated効果
Trance Gate
実験的

4. Sawtooth (ノコギリ波):

形状:
一方向のランプ

視覚:
    ╱│    ╱│
   ╱ │   ╱ │
  ╱  │  ╱  │
 ╱   │ ╱   │

特徴:
上昇→急降下

用途:
Filter Sweep (上昇)
特殊効果

5. Random (ランダム):

形状:
不規則

視覚:
  ╱╲  ╱─╲╱
 ╱  ╲╱

特徴:
予測不可能

用途:
Sample & Hold
実験的質感
```

### LFOパラメーター詳細

```
1. Rate (速度):

単位:
Hz または BPM同期

設定範囲:

非常に遅い (0.01-0.1 Hz):
10-100秒周期
Ambient・Pad

遅い (0.1-0.5 Hz):
2-10秒周期
標準Chorus

中間 (0.5-2.0 Hz):
0.5-2秒周期
明確な動き

速い (2.0-10 Hz):
0.1-0.5秒周期
Vibrato・Tremolo

非常に速い (10-20 Hz):
0.05-0.1秒周期
特殊効果

BPM同期:

1/32: 非常に速い
1/16: 速い
1/8: 標準
1/4: ゆっくり
1/2: 遅い
1 Bar: 非常に遅い

推奨設定:

Chorus: 0.3-0.7 Hz
Flanger: 0.2-0.5 Hz
Phaser: 0.2-0.6 Hz
Tremolo: 4-8 Hz
Vibrato: 5-7 Hz

2. Depth (Amount):

機能:
変化の大きさ

単位:
% (0-100%)

設定:

わずか (5-15%):
subtle
気づかない

控えめ (15-35%):
自然
標準

中間 (35-60%):
明確
はっきり

強烈 (60-100%):
極端
実験的

推奨:

Chorus: 20-30%
Flanger: 40-60%
Vibrato: 10-20%
Tremolo: 30-50%

3. Phase (位相):

機能:
波形の開始位置

設定:

0°: 波形の頭から
90°: 1/4周期ずらす
180°: 逆位相
270°: 3/4周期ずらす

用途:

Stereo効果:
L: 0°
R: 180°
→ 最大の広がり

複数LFO:
位相をずらして
複雑な動き

4. Offset:

機能:
中心位置の移動

例:

Pan LFO:
Offset: 0% → Center中心
Offset: +30% → 右寄り中心
Offset: -30% → 左寄り中心

Filter LFO:
Offset調整で
動く範囲を変更
```

---

## LFOクリエイティブテクニック

**応用的使い方:**

### 1. 複数LFO重ね

```
目標:
複雑な動き

手法:

LFO 1:
Rate: 0.3 Hz
Depth: 30%
Wave: Sine

LFO 2:
Rate: 0.47 Hz (わずかに違う)
Depth: 20%
Wave: Sine

結果:
自然で複雑
予測不能な動き

用途:
Pad・Ambient
高級感

実装 (Ableton):

1. Chorus
2. Phaser
2台直列

それぞれ異なるRate
```

### 2. LFO to Filter Cutoff

```
目標:
動くフィルター

設定:

Filter:
Cutoff: 1000 Hz

LFO:
Rate: 1/8 (BPM Sync)
Depth: 40%
Wave: Sine

結果:
フィルターが周期的に開閉

応用:

Techno Bass:
Rate: 1/4
Depth: 30%
Dark → Bright

House Lead:
Rate: 1/8
Depth: 25%
軽い動き
```

### 3. LFO to Resonance

```
目標:
動くResonance

設定:

Filter:
Cutoff: 固定
Resonance: 50%

LFO → Resonance:
Rate: 1/4
Depth: 40%

結果:
Resonanceが変化
質感の変化

用途:
Techno・Minimal
微妙な動き
```

### 4. Sample & Hold

```
機能:
ランダムなステップ

波形:
Random + Sample & Hold

結果:
階段状のランダム値

視覚:
┌───┐ ┌─┐
│   └─┘ └──

用途:

ランダムFilter:
予測不能な動き

ランダムPitch:
実験的

Glitch効果:
ランダムPan

設定例:

Filter Cutoff:
LFO Wave: Random S&H
Rate: 1/16
Depth: 50%

結果:
1/16音符ごとに
ランダムな音色
```

### 5. Envelope Follower連動

```
機能:
入力信号でLFO制御

原理:

Input音量大:
LFO Depth増加

Input音量小:
LFO Depth減少

結果:
ダイナミックなModulation

用途:

Flanger:
強い音: 強烈なうねり
弱い音: わずか

実装 (Ableton):

1. Flanger
2. Envelope: On
3. Amount調整

結果:
演奏に反応
```

---

## サイドチェインモジュレーション

**他トラックで制御:**

### 基本概念

```
原理:

トラックA (Kick):
リズム信号

トラックB (Pad):
Modulationを受ける

Sidechain:
KickのタイミングでPad変化

結果:
Kickに合わせて
Padが動く
```

### 実装方法

```
方法1: Auto Filter + Sidechain

1. Padトラック:

Auto Filter挿入
Cutoff: 2000 Hz

2. Sidechain設定:

Audio From: Kick
Filter: On

3. Envelope調整:

Attack: Fast
Release: Medium

結果:

Kickが鳴る:
Filterが閉じる

Kick消える:
Filterが開く

効果:
Pumping Filter

方法2: LFO Tool (Max for Live)

1. LFO Tool挿入

2. Sidechain:
Audio From: Kick

3. LFO設定:

Shape: Saw Down
Depth: 50%

結果:
Kickに合わせて
音量変化

方法3: Auto Pan + Sidechain

1. Auto Pan挿入

2. Sidechain:
Kick信号

3. 設定:

Amount: Kick制御

結果:
Kickに合わせて
Pan移動
```

### 実践例

```
Techno Pad (Sidechain Filter):

目標:
Kickに合わせた動き

設定:

1. Auto Filter:
   Type: Low Pass
   Cutoff: 1500 Hz
   Resonance: 20%

2. Sidechain:
   Audio From: Kick
   Filter: On

3. Envelope:
   Attack: 10 ms
   Release: 300 ms

結果:
Kickでフィルター閉じる
タイトなGroove

House Bass (Volume Duck):

目標:
Kickにスペース

設定:

1. Compressor:
   Sidechain: Kick
   Ratio: 4:1
   Attack: Fast
   Release: Auto

結果:
Kick鳴る時
Bass下がる

Trance Lead (Filter Pump):

目標:
Pumping効果

設定:

1. Auto Filter
2. Sidechain: Kick
3. Envelope:
   Attack: 5 ms
   Release: 500 ms (長め)

結果:
長いFilter開閉
Tranceっぽい
```

---

## モジュレーションとオートメーション

**時間軸での変化:**

### オートメーション基本

```
定義:

時間経過で
パラメーターを変化

LFO vs Automation:

LFO:
周期的
繰り返し

Automation:
自由
1回きりOK

併用:
最強
```

### 実践テクニック

```
1. Rate Automation (Buildup):

目標:
徐々に速く

設定:

Flanger Rate:
小節1-8: 0.2 Hz
小節9-16: 0.5 Hz
小節17-24: 1.0 Hz

結果:
うねりが加速
ビルドアップ

2. Depth Automation (Breakdown):

目標:
徐々に強く

設定:

Chorus Amount:
小節1-4: 10%
小節5-8: 20%
小節9-12: 35%

結果:
広がりが増す
盛り上がり

3. Dry/Wet Automation (Intro):

目標:
徐々にエフェクト

設定:

Phaser Dry/Wet:
小節1: 0%
小節8: 30%

結果:
自然な導入

4. Feedback Automation (Extreme):

目標:
極端な効果

設定:

Flanger Feedback:
通常: 50%
Climax: 85% (1小節のみ)

結果:
一時的に強烈
インパクト

5. Filter + LFO Automation:

目標:
複合的変化

設定:

Auto Filter:
LFO Rate: 1/8 (固定)
LFO Depth: Automation

Automation:
Depth: 0% → 60%
8小節で上昇

結果:
Filterの動きが
徐々に激しく
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

## Ableton内蔵エフェクト詳細

**標準エフェクトの全貌:**

### Chorus-Ensemble

```
Abletonの2種類:

1. Chorus (シンプル):

特徴:
軽量
CPU負荷低
基本的

用途:
シンプルなChorus

2. Chorus-Ensemble (高品質):

特徴:
複数Voice
リッチ
CPU負荷やや高

パラメーター:

Voices: 2-4
Voice数

Delay: 10-50 ms
遅延時間

Rate: 0.1-10 Hz
LFO速度

Amount: 0-100%
変化量

Feedback: -100 to +100%
フィードバック

Dry/Wet: 0-100%
ミックス

推奨設定:

Lush Pad:
Voices: 4
Delay: 25 ms
Rate: 0.4 Hz
Amount: 30%
Dry/Wet: 40%

Subtle Lead:
Voices: 2
Delay: 15 ms
Rate: 0.6 Hz
Amount: 20%
Dry/Wet: 25%
```

### Flanger詳細

```
Ableton Flanger:

特徴:
High Quality モード
Envelope Follower
Sync機能

パラメーター詳細:

Hi Pass: 10-1000 Hz
低域カット
タイトに

Delay: 0.13-13 ms
遅延時間

Feedback: -100 to +100%
強度

Polarity: +/-
位相

Dry/Wet: 0-100%

Envelope:
Amount: 0-100%
Attack: 0.1-500 ms
Release: 1-5000 ms

Sync: On/Off
BPM同期

高度な使い方:

Envelope Modulation:

設定:
Envelope Amount: 50%
Attack: 10 ms
Release: 200 ms

効果:
強い音: Flanger強
弱い音: Flanger弱

用途:
ダイナミック
Techno Bass

Negative Feedback:

設定:
Feedback: -60%
Polarity: -

効果:
特殊な質感
実験的

Extreme Settings:

設定:
Delay: 13 ms (最大)
Feedback: 80%
Rate: 0.1 Hz (遅い)

効果:
非常に深いうねり
Dark Techno
```

### Phaser詳細

```
Ableton Phaser:

特徴:
12 Poles対応
Spin機能
Envelope Follower

パラメーター:

Poles: 2/4/8/12
段数

Color: -100 to +100
位相関係

Frequency: 200-8000 Hz
中心周波数

Env. Modulation:
-100 to +100%

Env. Attack: 0.1-500 ms
Env. Release: 1-5000 ms

Feedback: 0-95%

Output: -12 to +12 dB
ゲイン補正

Dry/Wet: 0-100%

Spin: 0-360°
L/R位相差

高度なテクニック:

12 Poles + Feedback:

設定:
Poles: 12
Feedback: 70%
Frequency: 1500 Hz
Rate: 0.3 Hz

効果:
非常に深いPhase
実験的

Envelope Modulation:

設定:
Env. Mod: 60%
Attack: 5 ms
Release: 150 ms

効果:
強い音: 高域Sweep
弱い音: 低域

Spin for Width:

設定:
Spin: 180°
Poles: 4
Rate: 0.5 Hz

効果:
広いステレオ
L/R逆位相
```

---

## ジャンル別戦略拡大

**詳細なジャンル別活用:**

### Techno (詳細)

```
基本方針:
暗い・深い・最小限

主要エフェクト:

1. Flanger (Pad・Lead):

設定:
Rate: 0.15-0.3 Hz (非常に遅い)
Amount: 55%
Delay: 4 ms
Feedback: 60%
Polarity: - (マイナス)
Dry/Wet: 35%

Post-EQ:
High Cut 6-8 kHz
暗くする

2. Phaser (Hi-Hat):

設定:
Poles: 2 (軽い)
Frequency: 3000 Hz
Rate: 0.4 Hz
Feedback: 25% (控えめ)
Dry/Wet: 15%

3. Auto Pan (Perc):

設定:
Shape: Triangle
Rate: 1/16 (Sync)
Amount: 40%
Phase: 180°

避けるべき:

Chorus:
明るすぎる
Technoに合わない

強いModulation:
最小限が美学
```

### House (詳細)

```
基本方針:
温かい・広い・グルーヴ

主要エフェクト:

1. Chorus (Vocal・Keys):

設定:
Voices: 2
Rate: 0.5 Hz
Amount: 25%
Delay: 18 ms
Dry/Wet: 30%

2. Phaser (Rhodes):

設定:
Poles: 4
Frequency: 800 Hz
Rate: 0.6 Hz
Feedback: 40%
Spin: 70%
Dry/Wet: 35%

3. Auto Pan (Hi-Hat):

設定:
Shape: Sine
Rate: 1/8
Amount: 35%

4. Flanger (Breakdown):

設定:
Rate: 0.3 Hz
Amount: 50%
Feedback: 45%
Dry/Wet: 25%

オートメーション:
Dry/Wet: 0% → 40%
4小節で上昇
```

### Trance (詳細)

```
基本方針:
広大・エモーショナル・Pump

主要エフェクト:

1. Chorus (Pad):

設定:
Voices: 4 (最大)
Rate: 0.4 Hz
Amount: 35%
Dry/Wet: 45% (多め)

2. Phaser (Lead):

設定:
Poles: 8
Frequency: 1200 Hz
Rate: 0.5 Hz
Feedback: 55%
Spin: 100%
Dry/Wet: 30%

3. Flanger (Buildup):

オートメーション:
Rate: 0.2 → 1.5 Hz
Feedback: 40% → 85%
8小節

4. Sidechain Filter (Pad):

設定:
Auto Filter
Sidechain: Kick
Attack: 5 ms
Release: 500 ms (長い)

効果:
長いPump
Tranceっぽい
```

### Dubstep (詳細)

```
基本方針:
極端・実験的・ダイナミック

主要エフェクト:

1. Flanger (Wobble Bass):

設定:
Rate: 1.0 Hz (速い)
Amount: 75%
Delay: 6 ms
Feedback: 80% (極端)
Envelope: On
Env. Amount: 70%

2. Phaser (Build):

オートメーション:
Frequency: 500 → 5000 Hz
Feedback: 30% → 90%
16小節

3. Auto Pan (Snare):

設定:
Shape: Random
Rate: 1/32 (非常に速い)
Amount: 70%

4. Extreme Modulation:

Flanger:
Feedback: 95% (ほぼ最大)
Rate: 2.0 Hz
1-2小節のみ

効果:
強烈なインパクト
```

### Ambient (詳細)

```
基本方針:
微細・長周期・自然

主要エフェクト:

1. Chorus (Pad):

複数台使用:

Chorus 1:
Rate: 0.08 Hz (非常に遅い)
Amount: 20%

Chorus 2:
Rate: 0.13 Hz
Amount: 15%

合計:
複雑で自然な動き

2. Phaser (Texture):

設定:
Poles: 2 (軽い)
Frequency: 400 Hz
Rate: 0.05 Hz (極遅)
Feedback: 20%
Dry/Wet: 15% (subtle)

3. LFO + Automation:

LFO:
Rate: 0.1 Hz (固定)

Automation:
Depth: 10% → 30% → 10%
32小節周期

効果:
超長期的変化
```

---

## トラブルシューティング詳細

**問題解決マニュアル:**

### 問題1: 音が薄くなる

```
症状:
Modulationかけると
音が痩せる

原因:

位相キャンセル:
Chorus・Flangerの
位相問題

Dry/Wet過剰:
Wet多すぎ

解決策:

1. Dry/Wet調整:

推奨:
Chorus: 20-30%
Flanger: 25-35%
過剰を避ける

2. Polarity確認:

設定:
Polarity: + (正)
- (負)は実験的

3. EQ補正:

Post-EQ:
High Shelfで補正
+2dB @ 8kHz

4. Parallel処理:

方法:
Return Track使用
Dry 100%維持
```

### 問題2: ステレオ広すぎ

```
症状:
モノラル再生で消える
位相問題

原因:

Chorus Amount過剰
Phaser Spin 180°

解決策:

1. Amount削減:

Chorus:
Amount: 25% → 15%

2. Spin調整:

Phaser:
Spin: 180° → 90°

3. Mono互換性確認:

手順:
Utility挿入
Width: 0% (Mono)
確認

基準:
Mono時も聞こえる
= OK

4. Mid/Side処理:

方法:
Chorusを
Sideのみにかける

実装:
1. Utility (Width 200%)
2. Chorus
3. Utility (Width 50%)
```

### 問題3: CPU負荷高い

```
症状:
Modulationで
CPUスパイク

原因:

High Qualityモード
複数Modulation
Oversampling

解決策:

1. Quality設定:

Chorus:
High Quality: Off
音質変化minimal

2. Freeze Track:

手順:
右クリック
Freeze Track

効果:
CPU大幅削減

3. Commit処理:

方法:
気に入った設定
オーディオ化

4. エフェクト削減:

優先順位:
Chorus: 重要
Flanger: 削除可
Phaser: 削除可
```

### 問題4: 他トラックと競合

```
症状:
複数トラックの
Modulationが干渉

原因:

同じRate使用
全体が揺れる

解決策:

1. Rate分散:

Track A: 0.3 Hz
Track B: 0.47 Hz
Track C: 0.61 Hz
わずかに違う

2. 適用トラック限定:

推奨:
Lead: Chorus
Pad: Flanger
Keys: Phaser
分ける

3. Depth調整:

全体:
各25% → 合計で自然

1つ強調:
Lead 40%
他 15%

4. 周波数帯域分け:

Low (Bass):
Modulationなし

Mid (Lead/Pad):
Chorus/Flanger

High (Keys):
Phaser
```

---

## 実践プロジェクト例

**完全なトラック設定:**

### Techno Track設定

```
BPM: 130

Track 1: Kick
Modulation: なし
理由: タイトに

Track 2: Bass
Modulation: なし
理由: 低域明瞭

Track 3: Lead Synth
Modulation:
Flanger
Rate: 0.25 Hz
Amount: 55%
Feedback: 60%
Dry/Wet: 30%

Track 4: Pad
Modulation:
Flanger (Dark)
Rate: 0.2 Hz
Amount: 50%
Feedback: 55%
Polarity: -
Dry/Wet: 35%

Post: EQ High Cut 7kHz

Track 5: Hi-Hat
Modulation:
Auto Pan
Shape: Sine
Rate: 1/16
Amount: 35%

Track 6: Perc
Modulation:
Phaser (Light)
Poles: 2
Frequency: 3500 Hz
Rate: 0.4 Hz
Dry/Wet: 15%

Return A: Reverb
Send: Lead 15%, Pad 25%

Return B: Delay
Send: Lead 20%

ミックス方針:
Modulationは控えめ
暗い質感維持
```

### House Track設定

```
BPM: 124

Track 1: Kick
Modulation: なし

Track 2: Bass
Modulation: なし

Track 3: Vocal
Modulation:
Chorus
Rate: 0.5 Hz
Amount: 18%
Delay: 18 ms
Dry/Wet: 25%

Track 4: Rhodes
Modulation:
Phaser
Poles: 4
Frequency: 800 Hz
Rate: 0.6 Hz
Feedback: 40%
Spin: 70%
Dry/Wet: 35%

Post: Reverb (Plate) 30%

Track 5: String Pad
Modulation:
Chorus (Dual)
Chorus 1: Rate 0.3 Hz
Chorus 2: Rate 0.47 Hz
各 Dry/Wet: 20%

Track 6: Hi-Hat
Modulation:
Auto Pan
Shape: Sine
Rate: 1/8
Amount: 40%

Return A: Reverb (Hall)
Send: Vocal 35%, Strings 40%

Return B: Delay (1/8)
Send: Vocal 25%

ミックス方針:
温かい質感
広がり重視
```

---

## サードパーティプラグイン

**プロ御用達:**

### Soundtoys EchoBoy Jr.

```
特徴:
Chorus/Flangerモード
ビンテージ質感
無料版あり

Chorusモード:

設定:
Time: 20 ms
Feedback: 0%
Wow: 30% (Rate相当)
Flutter: 20% (Depth相当)
Mix: 30%

用途:
ビンテージChorus
アナログ質感

推奨トラック:
Keys・Rhodes
Vocal
```

### FabFilter Timeless 3

```
特徴:
高品質Flanger/Phaser
モジュレーション自由度高
視覚的

Flangerモード:

設定:
Delay: 3 ms
Feedback: 50%
LFO Rate: 0.3 Hz
LFO Depth: 60%
Stereo: 100%

特殊機能:
複数LFO
Filter連動
Mid/Side処理

用途:
実験的Modulation
高品質処理
```

### Valhalla Freq Echo

```
特徴:
Frequency Shifter
実験的Modulation
独特

設定:

Frequency Shift: 5 Hz
Feedback: 40%
LFO Rate: 0.2 Hz
LFO Depth: 30%
Mix: 25%

効果:
金属的
非ハーモニック
実験的

用途:
Ambient
実験音楽
```

---

## よくある質問 (FAQ)

**Q&A集:**

### Q1: ChorusとFlangerの違いは?

```
A:

遅延時間:

Chorus: 15-30 ms
Flanger: 1-10 ms

Feedback:

Chorus: 0-20%
Flanger: 40-70%

用途:

Chorus: 厚み・広がり
Flanger: うねり・実験

判断:

自然な厚み → Chorus
特殊効果 → Flanger
```

### Q2: 全トラックにかけていい?

```
A:

ダメ:

理由:
全体が揺れる
定位不明
ミックス崩壊

推奨配分:

30-40%のトラック:
Lead・Pad・Keys

Kick・Bass:
絶対にかけない

判断基準:
少ないほど効果的
```

### Q3: Rateの最適値は?

```
A:

一般的:

Chorus: 0.3-0.7 Hz
Flanger: 0.2-0.5 Hz
Phaser: 0.2-0.6 Hz

理由:
遅いほど自然
速い = 不自然

例外:

Tremolo: 4-8 Hz
Vibrato: 5-7 Hz
特殊効果: 1.0+ Hz

判断:
聴いて判断
遅めから始める
```

### Q4: CPUを節約するには?

```
A:

方法1: Quality下げる

Chorus:
High Quality: Off
ほぼ同じ音質

方法2: Freeze Track

効果:
CPU 90%削減

方法3: オーディオ化

タイミング:
設定確定後
完全にCPUフリー

方法4: プラグイン選択

軽い:
Ableton内蔵

重い:
Arturia等エミュ

判断:
必要性で決定
```

### Q5: モノラル互換性は?

```
A:

確認方法:

1. Utility挿入
2. Width: 0% (Mono)
3. 聴く
4. 消えたらダメ

原因:

位相キャンセル
Stereo Width過剰

解決:

Amount削減
Spin調整
Mid/Side処理

基準:
Mono時80%残る = OK
```

---

## 練習課題

**スキルアップ:**

### 初級課題 (Week 1)

```
目標:
Chorus基礎マスター

Day 1:

1. Leadトラック作成
2. Chorus挿入
3. Rate: 0.5 Hz
4. Amount: 30%
5. Dry/Wet: 25%
6. A/B比較
7. 違いメモ

Day 2:

1. Padトラック
2. Chorus挿入
3. Rate: 0.3 Hz (遅め)
4. Amount: 25%
5. Dry/Wet: 35%
6. A/B比較

Day 3:

1. Vocal
2. Chorus軽め
3. Rate: 0.4 Hz
4. Amount: 15%
5. Dry/Wet: 20%
6. 過剰テスト
7. 50% → 不自然確認

Day 4-5:

Rateテスト:
0.2 / 0.5 / 1.0 / 2.0 Hz
違い体感

Day 6-7:

トラック完成:
3トラック
各Chorus最適化
ミックス
```

### 中級課題 (Week 2)

```
目標:
Flanger・Phaserマスター

Day 1-2:

Flanger:

1. Padトラック
2. Flanger挿入
3. Delay: 3 ms
4. Feedback: 50%
5. Rate: 0.3 Hz
6. Dry/Wet: 30%
7. Feedbackテスト:
   0% / 30% / 60% / 90%

Day 3-4:

Phaser:

1. Keysトラック
2. Phaser挿入
3. Poles: 4
4. Frequency: 1000 Hz
5. Rate: 0.5 Hz
6. Polesテスト:
   2 / 4 / 8 / 12

Day 5-6:

Auto Pan:

1. Hi-Hat
2. Auto Pan
3. Shape: Sine
4. Rate: 1/8
5. Amount: 35%
6. Shapeテスト:
   Sine / Triangle / Square

Day 7:

総合:
Full Track
各エフェクト配置
ミックス
```

### 上級課題 (Week 3-4)

```
目標:
複合テクニックマスター

Week 3:

LFO実験:

Day 1-2:
Filter + LFO
Cutoff Modulation

Day 3-4:
複数LFO重ね
Chorus + Phaser

Day 5-6:
Envelope Follower
ダイナミック処理

Day 7:
実験トラック完成

Week 4:

Automation:

Day 1-2:
Rate Automation
Buildup作成

Day 3-4:
Depth Automation
Breakdown作成

Day 5-6:
Sidechain Modulation
Techno Groove

Day 7:

Final Project:
Full Track (8小節)
全テクニック統合
ミックス・Export
```

---

## リファレンス設定集

**コピペ用設定:**

### Classic Chorus (Lead)

```
Ableton Chorus:

Rate: 0.5 Hz
Amount: 28%
Delay: 20 ms
Feedback: 0%
Polarity: +
Dry/Wet: 25%

用途:
一般的なLead
Synth全般
```

### Lush Pad Chorus

```
Ableton Chorus x2:

Chorus 1:
Rate: 0.3 Hz
Amount: 25%
Delay: 25 ms
Dry/Wet: 30%

Chorus 2:
Rate: 0.47 Hz
Amount: 18%
Delay: 22 ms
Dry/Wet: 25%

用途:
広大なPad
Ambient
```

### Dark Techno Flanger

```
Ableton Flanger:

Hi Pass: 100 Hz
Delay: 4 ms
Rate: 0.25 Hz
Amount: 58%
Feedback: 60%
Polarity: -
Dry/Wet: 35%

Post:
EQ Eight
High Cut 7 kHz

用途:
Techno Pad
暗い質感
```

### Funk Phaser (Rhodes)

```
Ableton Phaser:

Poles: 4
Frequency: 800 Hz
Rate: 0.6 Hz
Amount: 55%
Feedback: 42%
Spin: 75%
Dry/Wet: 35%

用途:
Electric Piano
Funk・Disco
```

### Subtle Vocal Chorus

```
Ableton Chorus:

Rate: 0.4 Hz
Amount: 15%
Delay: 18 ms
Feedback: 0%
Dry/Wet: 20%

用途:
House Vocal
わずかな厚み
```

### Buildupエフェクト

```
Ableton Flanger:

Automation (8小節):

Rate:
0.2 Hz → 1.5 Hz

Amount:
40% → 80%

Feedback:
40% → 85%

Dry/Wet:
20% → 60%

用途:
Buildup
Drop前
```

---

## 詳細テクニック集

**高度なテクニック:**

### パラレルModulation

```
概念:
Dry信号を保持
Wetを追加

メリット:
音痩せ防止
パンチ維持
厚み追加

実装:

方法1: Return Track

1. Return Track作成
2. Chorusを100% Wet
3. Sendで送る量調整
4. Dry信号はそのまま

設定例:

Lead Track:
Chorus Send: 30%

Return A:
Chorus
Dry/Wet: 100%

結果:
パンチあり
厚みあり

方法2: Audio Effect Rack

1. Audio Effect Rack作成
2. Chain 1: Dry (何もなし)
3. Chain 2: Chorus 100% Wet
4. Chain Mix調整

Chain 1 Volume: 0dB
Chain 2 Volume: -6dB

結果:
完全コントロール
柔軟性高い
```

### Mid/Side Modulation

```
概念:
Mid: モノラル成分
Side: ステレオ成分

用途:
Sideのみにエフェクト
Mono互換性維持

実装:

1. Utility挿入
   Width: 200% (M/S化)

2. Audio Effect Rack
   Chain 1: Mid (何もなし)
   Chain 2: Side (Chorus)

3. Utility挿入
   Width: 50% (通常に戻す)

設定例:

Side Chain:
Chorus
Rate: 0.5 Hz
Amount: 40%
Dry/Wet: 100%

結果:
Mono: クリーン
Stereo: 広い
互換性: 完璧
```

### マルチバンドModulation

```
概念:
周波数帯域ごとに
異なるModulation

用途:
低域: クリーン
高域: Modulation

実装:

1. Multiband Dynamics (3 Band)

2. 各帯域にChorus

Low (20-200 Hz):
Modulationなし

Mid (200-2000 Hz):
軽いChorus
Rate: 0.4 Hz
Amount: 20%

High (2000+ Hz):
強めChorus
Rate: 0.6 Hz
Amount: 35%

結果:
低域: タイト
高域: 広い
バランス: 最高
```

### 動的Modulation

```
概念:
音量に応じて
Modulation変化

用途:
強い音: 強いエフェクト
弱い音: 弱いエフェクト

実装:

Ableton Flanger:

1. Envelope: On
2. Env. Amount: 60%
3. Env. Attack: 10 ms
4. Env. Release: 200 ms

基本設定:
Rate: 0.3 Hz
Amount: 30% (Base)
Feedback: 50%

結果:
静かな部分: 30% Amount
大きな部分: 60% Amount
ダイナミック
```

### ステレオ幅コントロール

```
概念:
Modulationで
ステレオ幅を制御

実装:

方法1: Dual Chorus

Chorus 1 (L):
Phase: 0°

Chorus 2 (R):
Phase: 180°

Utility (Panning):
L/R分離

結果:
最大のステレオ幅

方法2: Phaser Spin

Phaser:
Spin: 150°
(完全180°は避ける)

結果:
広いが安全
Mono互換性あり

方法3: Auto Pan + Chorus

1. Auto Pan
   Amount: 30%
   Rate: 1/8

2. Chorus
   Rate: 0.5 Hz
   Amount: 25%

結果:
動的な広がり
複雑
```

---

## プロダクションチップス

**実践的アドバイス:**

### ミックスでの配置

```
原則:

Lead・Vocal:
Chorus控えめ
15-25%

Pad:
Chorus/Flanger多め
30-40%

Keys:
Phaser
25-35%

Hi-Hat・Perc:
Auto Pan
30-50%

Bass・Kick:
なし
絶対に

理由:

低域:
タイト必須
Modulationで濁る

中高域:
広がりOK
質感向上
```

### エフェクト順序

```
推奨順序:

1. EQ (High Pass)
   不要な低域カット

2. Compression
   ダイナミクス整理

3. Modulation (Chorus等)
   質感追加

4. EQ (調整)
   最終調整

5. Reverb/Delay (Return)
   空間系

理由:

Modulation前にクリーン化
Modulation後に調整

間違った順序:

Chorus → Compressor:
圧縮で効果減
不自然

正しい順序:

Compressor → Chorus:
均一な効果
自然
```

### CPU最適化

```
テクニック:

1. Freeze不要トラック

対象:
確定したトラック

効果:
CPU 80-90%削減

2. Quality設定見直し

Chorus High Quality:
ほぼ違いなし
Off推奨

3. エフェクト統合

悪い例:
Track 1: Chorus
Track 2: Chorus
Track 3: Chorus

良い例:
Return A: Chorus
各Track: Send調整

CPU削減:
66%削減

4. 必要性再検討

質問:
本当に必要?
A/Bで差ある?

答えがNo:
削除

効果:
不要なCPU使用なし
```

### バウンス前チェック

```
チェックリスト:

□ Mono互換性確認
  Utility Width: 0%
  聴く

□ 位相問題確認
  Correlation Meter
  -1に近い = 問題

□ CPU負荷確認
  50%以下推奨

□ エフェクト過剰確認
  A/B比較
  効果明確?

□ 低域クリーン確認
  Bass・Kick
  Modulationなし?

□ オートメーション確認
  ジャンプなし
  滑らか

すべてOK:
バウンス実行
```

---

## 最終チェックリスト

**プロジェクト完成前:**

### サウンドチェック

```
□ 各トラックA/B比較済み
  効果明確

□ Modulation過剰なし
  Rate遅め
  Amount控えめ

□ 低域クリーン
  Bass・Kick
  Modulationなし

□ Mono互換性OK
  主要要素残る

□ 位相問題なし
  Correlation正常

□ CPU負荷OK
  50%以下
```

### 技術チェック

```
□ エフェクト順序正しい
  EQ → Comp → Mod

□ Return活用
  CPU効率的

□ Automation滑らか
  ジャンプなし

□ パラメーター記録
  設定メモ済み

□ プリセット保存
  再利用可能
```

### ミックスチェック

```
□ 全体バランス良好
  各要素聴こえる

□ ステレオ幅適切
  広すぎない

□ ダイナミクス良好
  圧縮過剰なし

□ 周波数バランス
  低中高均等

□ クリッピングなし
  Master -6dB以下
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
