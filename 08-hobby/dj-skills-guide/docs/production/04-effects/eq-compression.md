# EQ・コンプレッサー

ミックスの基礎となる最重要エフェクト。EQ Eightとコンプレッサーを完全マスターし、プロの音質を実現します。

## この章で学ぶこと

- EQ Eightの8バンド
- 各バンドタイプ（Peak、Shelf、High/Low Pass）
- Compressorの全パラメーター
- Attack・Releaseの調整
- サイドチェイン圧縮
- トラック別EQ・コンプ設定
- よくある失敗と対処法

---

## なぜEQ・コンプが最重要なのか

**ミックスの90%:**

```
プロの音質:

1位: EQ Eight (50%)
周波数バランス

2位: Compressor (40%)
ダイナミクス制御

3位: その他 (10%)
空間系等

使用頻度:

EQ Eight:
全トラック100%
必須

Compressor:
Kick・Bass・Vocal: 100%
その他: 70%

真実:

「良いミックス」=
EQ・コンプが完璧

音色・楽曲:
二の次

プロとアマの差:
この2つのみ
```

---

## EQ Eight 完全ガイド

**8バンドパラメトリックEQ:**

### インターフェイス

```
┌─────────────────────────────────┐
│  EQ Eight                       │
├─────────────────────────────────┤
│                                 │
│      周波数カーブ表示            │
│         (視覚的)                │
│                                 │
│  Band 1  [●]───────────────     │
│    Freq: 100 Hz                │
│    Gain: +3.0 dB               │
│    Q: 1.0                      │
│                                 │
│  [1][2][3][4][5][6][7][8]      │
│   バンド選択                    │
│                                 │
│  [Adaptive Q] [Scale: 0 dB]    │
└─────────────────────────────────┘

8つのバンド:
それぞれ独立して調整可能

バンドタイプ:
- Low Cut (High Pass)
- Low Shelf
- Peak (Bell)
- High Shelf
- High Cut (Low Pass)
```

### バンドタイプ詳細

```
1. Low Cut (High Pass):

機能:
低域カット

用途:
不要な低域除去
必須処理

設定:

Frequency:
カット開始点

Slope:
6/12/18/24/48 dB/oct

推奨:

Kick: 30 Hz (24 dB/oct)
Bass: 40 Hz (24 dB/oct)
Lead: 200 Hz (12 dB/oct)
Vocal: 80 Hz (18 dB/oct)
Pad: 300 Hz (12 dB/oct)

理由:
低域濁り防止

2. Low Shelf:

機能:
低域全体を増減

用途:
温かみ追加・削減

設定:

Frequency:
変化開始点

Gain:
増減量 (-12 〜 +12 dB)

推奨:

Bass太く:
80 Hz, +2 dB

Vocal軽く:
120 Hz, -2 dB

3. Peak (Bell):

機能:
特定周波数を増減

用途:
問題周波数除去
良い周波数強調

設定:

Frequency:
中心周波数

Gain:
増減量

Q:
幅 (0.5 = 広い, 5.0 = 狭い)

推奨:

問題除去:
Q: 3-5 (狭い)
Gain: -3 〜 -6 dB

強調:
Q: 1-2 (広い)
Gain: +2 〜 +4 dB

4. High Shelf:

機能:
高域全体を増減

用途:
明るさ調整

設定:

Frequency:
変化開始点

Gain:
増減量

推奨:

明るく:
8 kHz, +2 dB

暗く:
6 kHz, -3 dB

5. High Cut (Low Pass):

機能:
高域カット

用途:
暗くする
Kickから高域除去

設定:

Frequency:
カット開始点

Slope:
6/12/18/24/48 dB/oct

推奨:

Kick: 8 kHz (12 dB/oct)
Bass: 5 kHz (18 dB/oct)
Pad (暗い): 10 kHz (12 dB/oct)
```

### 周波数帯域の特徴

```
20-60 Hz (Sub Bass):

特徴:
体で感じる
聴こえにくい

用途:
Kick・Bass

処理:
High Pass 30-40 Hz
不要カット

60-250 Hz (Low):

特徴:
パワー
温かみ

用途:
Kick・Bass・Tom

処理:
濁りやすい
注意

250-500 Hz (Low Mid):

特徴:
濁りやすい
問題多い

用途:
全楽器が集中

処理:
-2 〜 -4 dB
スペース作る

500 Hz-2 kHz (Mid):

特徴:
存在感
前に出る

用途:
Vocal・Snare・Guitar

処理:
+2 〜 +4 dB
存在感

2-5 kHz (Upper Mid):

特徴:
明瞭度
アタック

用途:
Vocal明瞭度
スネアスナップ

処理:
+2 〜 +3 dB (Vocal)
過剰: 刺さる

5-10 kHz (Presence):

特徴:
空気感
明るさ

用途:
Hi-Hat・Cymbals

処理:
+1 〜 +2 dB
わずかに

10-20 kHz (Air):

特徴:
空気感
高級感

用途:
Master全体

処理:
+0.5 〜 +1.5 dB
High Shelf
```

---

## EQ実践例

**トラック別設定:**

### Kick

```
目標:
パワフル、明確

Band 1 (High Pass):
Freq: 30 Hz
Slope: 24 dB/oct
理由: 不要な超低域カット

Band 2 (Low Shelf):
Freq: 60 Hz
Gain: +3 dB
理由: パワー

Band 3 (Peak):
Freq: 250 Hz
Gain: -3 dB
Q: 2.0
理由: 濁り除去

Band 4 (Peak):
Freq: 3-5 kHz
Gain: +2 dB
Q: 1.5
理由: アタック明確

Band 5 (High Cut):
Freq: 10 kHz
Slope: 12 dB/oct
理由: 不要な高域カット

結果:
太く、明確なKick
```

### Bass

```
目標:
太い、クリア

Band 1 (High Pass):
Freq: 40 Hz
Slope: 24 dB/oct

Band 2 (Low Shelf):
Freq: 80 Hz
Gain: +2 dB

Band 3 (Peak):
Freq: 200-300 Hz
Gain: -2 dB
Q: 2.0
理由: Kickとの分離

Band 4 (Peak):
Freq: 800-1200 Hz
Gain: +2 dB
Q: 1.5
理由: 存在感

Band 5 (High Cut):
Freq: 5 kHz
Slope: 18 dB/oct
理由: 高域不要

結果:
Kickと分離
明確なBass
```

### Vocal

```
目標:
明瞭、前に出る

Band 1 (High Pass):
Freq: 80 Hz
Slope: 18 dB/oct
理由: 不要低域カット

Band 2 (Peak):
Freq: 200-400 Hz
Gain: -2 〜 -4 dB
Q: 2.0
理由: こもり除去

Band 3 (Peak):
Freq: 2-3 kHz
Gain: +3 dB
Q: 1.5
理由: 明瞭度

Band 4 (Peak):
Freq: 6-8 kHz
Gain: -3 dB
Q: 3.0
理由: De-ess (歯擦音除去)

Band 5 (High Shelf):
Freq: 10 kHz
Gain: +1.5 dB
理由: 空気感

結果:
明瞭、自然なVocal
```

### Pad

```
目標:
広い、暗い

Band 1 (High Pass):
Freq: 300 Hz
Slope: 12 dB/oct
理由: 低域はKick・Bassに

Band 2 (Peak):
Freq: 500-800 Hz
Gain: -2 dB
Q: 1.5
理由: 濁り防止

Band 3 (High Shelf):
Freq: 6 kHz
Gain: -2 dB
理由: 暗く、後ろに

結果:
広く、邪魔しないPad
```

---

## Compressor 完全ガイド

**ダイナミクス制御:**

### インターフェイス

```
┌─────────────────────────────────┐
│  Compressor                     │
├─────────────────────────────────┤
│                                 │
│  Threshold: -18.0 dB            │
│  Ratio: 4:1                     │
│  Attack: 10.0 ms                │
│  Release: 100 ms                │
│  Knee: 0.0 dB                   │
│  Gain: +4.0 dB                  │
│                                 │
│  GR Meter: -6 dB ■■■□□         │
│  (Gain Reduction)               │
│                                 │
│  [Peak] [RMS]                   │
│  [Sidechain] [EQ]               │
└─────────────────────────────────┘

メーター:
Gain Reduction (GR)
どれだけ圧縮されているか
```

### パラメーター詳細

```
1. Threshold:

機能:
圧縮開始点

単位:
dB

設定:

高い (-5 dB):
少しだけ圧縮

低い (-30 dB):
多く圧縮

推奨:

Kick: -12 dB
Bass: -15 dB
Vocal: -18 dB
Master: -6 dB

目標GR:
-3 〜 -6 dB

2. Ratio:

機能:
圧縮比率

例:

4:1 =
Threshold超えた4 dB
→ 1 dBに圧縮

設定:

1:1 = 圧縮なし
2:1 = 軽い
4:1 = 標準
8:1 = 強い
∞:1 = Limiter

推奨:

Kick: 4:1
Bass: 6:1 (強め)
Vocal: 3:1 (軽め)
Drums: 3-4:1
Master: 2:1 (軽め)

3. Attack:

機能:
圧縮開始速度

単位:
ms (ミリ秒)

設定:

速い (0.1-5 ms):
アタック潰す
タイト

遅い (20-50 ms):
アタック残す
パンチ

推奨:

Kick: 10 ms
アタック残す

Bass: 30 ms
グルーヴ維持

Vocal: 5 ms (速い)
安定

Drums: 10-15 ms
パンチ維持

4. Release:

機能:
圧縮終了速度

単位:
ms

設定:

速い (40-80 ms):
リズミカル
追従

遅い (200-500 ms):
滑らか
自然

推奨:

Kick: 80 ms
タイト

Bass: 100 ms
グルーヴ

Vocal: 40-60 ms
自然

Master: Auto
テンポ追従

5. Knee:

機能:
圧縮カーブ

設定:

0 dB (Hard):
急激
明確

6-12 dB (Soft):
緩やか
自然

推奨:

Drums: 0 dB (Hard)
Vocal: 6 dB (Soft)
Master: 3-6 dB

6. Gain (Make-up):

機能:
音量補正

理由:
圧縮で音量↓
→ Gainで補正

推奨:

Auto:
自動調整

Manual:
GR分を補正
GR -6 dB → Gain +6 dB
```

### Peak vs RMS

```
Peak Mode:

検出:
ピーク値

特徴:
速い反応
正確

用途:
Drums
アタック重要

RMS Mode:

検出:
平均値

特徴:
滑らかな反応
自然

用途:
Vocal・Bass
全体的圧縮

推奨:

Kick・Snare: Peak
Bass・Vocal: RMS
Master: RMS
```

---

## Compressor実践例

**トラック別設定:**

### Kick

```
目標:
パンチ維持、安定

設定:

Threshold: -12 dB
Ratio: 4:1
Attack: 10 ms (アタック残す)
Release: 80 ms (タイト)
Knee: 0 dB (Hard)
Mode: Peak
Gain: +4 dB

GR目標:
-4 〜 -6 dB

結果:
パンチあり、安定したKick

NG例:

Attack: 0.1 ms
→ アタック潰れる
→ パンチなし
```

### Bass

```
目標:
安定、グルーヴ維持

設定:

Threshold: -15 dB
Ratio: 6:1 (強め)
Attack: 30 ms (グルーヴ)
Release: 100 ms
Knee: 3 dB
Mode: RMS
Gain: +6 dB

GR目標:
-6 〜 -9 dB
しっかり圧縮

結果:
音量安定、Kickと調和

NG例:

Attack: 5 ms
→ グルーヴ消える
→ 平坦
```

### Vocal

```
目標:
明瞭、音量安定

設定:

Threshold: -18 dB
Ratio: 3:1 (軽め)
Attack: 5 ms (速い)
Release: 40 ms (速い)
Knee: 6 dB (Soft)
Mode: RMS
Gain: +5 dB

GR目標:
-3 〜 -5 dB

結果:
安定、自然なVocal

追加:

2段コンプ:

1. 1st: 軽く (3:1)
2. 2nd: 中程度 (4:1)

プロ手法
```

### Drum Bus

```
目標:
全ドラムをまとめる

設定:

Threshold: -15 dB
Ratio: 3:1
Attack: 10 ms
Release: Auto (テンポ追従)
Knee: 3 dB
Mode: Peak
Gain: +4 dB

GR目標:
-2 〜 -4 dB (軽め)

結果:
一体感、グルー

手法:
Glue Compressor推奨
(専用デバイス)
```

---

## サイドチェイン圧縮

**Kickで他をダッキング:**

### 基本概念

```
通常Compressor:

Input信号:
自身のトラック

圧縮:
自身を圧縮

Sidechain:

Input信号:
別トラック (Kick)

圧縮:
自身を圧縮

効果:
Kick鳴る時
→ Bassが下がる
→ Kick明確

用途:
Techno/House必須
```

### 設定方法

```
1. Bass Trackにコンプ挿入

2. Sidechain:
   On

3. Audio From:
   Kick Track

4. パラメーター:

Threshold: -24 dB (低め)
Ratio: 8:1 (強め)
Attack: 0.1 ms (最速)
Release: 80-150 ms (タイト)
Gain: +8 dB

5. GR確認:

Kick鳴る時:
GR -6 〜 -10 dB

Kick無い時:
GR 0 dB

結果:
Kickのスペース確保
```

### サイドチェインEQ

```
機能:
特定周波数のみ検出

用途:
Kick低域のみでダッキング

設定:

1. Sidechain: On

2. EQ: On

3. Sidechain EQ:

High Pass: 30 Hz
Low Pass: 150 Hz
→ Kick低域のみ

効果:
低域だけダッキング
高域は自然

推奨:
Bass・Padに使用
```

---

## よくある失敗

### 1. High Pass不足

```
問題:
ミックス全体が濁る

原因:
Low-Mid (200-400 Hz)混雑

解決:

全トラックにHigh Pass:

Kick: 30 Hz
Bass: 40 Hz
Lead: 200 Hz
Pad: 300 Hz
Vocal: 80 Hz

効果:
劇的にクリア
```

### 2. コンプAttack速すぎ

```
問題:
Kickのパンチなし
平坦

原因:
Attack: 0.1 ms
アタック潰れる

解決:

Attack: 10 ms
アタック残す

A/B比較:
必須
```

### 3. EQブースト過剰

```
問題:
不自然
刺さる

原因:
+6 dB以上ブースト

解決:

ブースト:
最大 +3 〜 +4 dB

カット優先:
他をカット
→ 相対的にブースト

推奨:
「引き算EQ」
```

### 4. GR多すぎ

```
問題:
音が潰れる
ダイナミクスゼロ

原因:
GR -10 dB以上

解決:

GR目標:
-3 〜 -6 dB

Master:
-2 〜 -4 dB (軽め)

ルール:
「少し足りない」くらい
```

---

## EQ・コンプの順序

**どちらが先？**

```
標準: EQ → Compressor

理由:

1. 不要カット (EQ)
2. 圧縮 (Comp)

メリット:
コンプが効率的

例外: Compressor → EQ

用途:
問題周波数が
コンプで強調される場合

手順:

1. Comp先
2. 問題発生
3. EQ後で修正

推奨:

初心者: EQ → Comp
中級者: 両方試す
```

---

## 実践ワークフロー

**30分練習:**

### Week 1: EQ Eight

```
Day 1 (10分):

1. Kickトラック選択
2. EQ Eight挿入
3. High Pass 30 Hz
4. Low Shelf +3 dB @ 60 Hz
5. A/B比較

Day 2 (15分):

1. Bassトラック
2. High Pass 40 Hz
3. Peak -2 dB @ 250 Hz (濁り)
4. Peak +2 dB @ 1 kHz (存在感)
5. High Cut 5 kHz

Day 3 (20分):

1. Vocalトラック
2. High Pass 80 Hz
3. Peak -3 dB @ 300 Hz
4. Peak +3 dB @ 3 kHz (明瞭度)
5. Peak -3 dB @ 7 kHz (De-ess)

Day 4-7:

全トラックにEQ
バランス確認
```

### Week 2: Compressor

```
Day 1 (15分):

1. Kick
2. Compressor挿入
3. Threshold -12 dB
4. Ratio 4:1
5. Attack 10 ms
6. Release 80 ms
7. GR確認 (-4 〜 -6 dB)

Day 2 (15分):

1. Bass
2. Threshold -15 dB
3. Ratio 6:1
4. Attack 30 ms (重要)
5. A/B比較
   グルーヴ維持確認

Day 3-4 (20分):

Vocal・Drums
各トラック設定

Day 5-7 (30分):

サイドチェイン:
Kick → Bass
```

---

## まとめ

### EQ Eight

```
□ 全トラックHigh Pass必須
□ Low-Mid (250-500 Hz)カット優先
□ ブースト最大 +3 〜 +4 dB
□ Q値: カット広く、ブースト狭く
□ Spectrum確認しながら
```

### Compressor

```
□ GR: -3 〜 -6 dB
□ Attack: アタック残す (10 ms+)
□ Release: グルーヴ維持
□ Ratio: 3-4:1が標準
□ A/B比較必須
```

### サイドチェイン

```
□ Kick → Bass (必須)
□ Ratio: 8:1, Attack: 0.1 ms
□ GR: -6 〜 -10 dB
□ Sidechain EQ活用
```

### 重要原則

```
□ Less is More
□ 引き算EQ優先
□ コンプは控えめ
□ 全てA/B比較
□ Spectrumで確認
```

---

**次は:** [Reverb・Delay](./reverb-delay.md) - 空間系エフェクトで深みを作る
