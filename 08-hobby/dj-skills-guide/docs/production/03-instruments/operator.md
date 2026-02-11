# Operator

FM合成の力。ベル、ブラス、金属的サウンドなど、Wavetableでは作れない複雑な音色をマスターします。

## この章で学ぶこと

- FM合成の基礎
- Operatorの構造（4オシレーター）
- アルゴリズム理解
- オシレーター設定
- エンベロープ活用
- FMベル音色の作り方
- ブラス音色の作り方
- 金属的ベースの作り方

---

## なぜOperatorが重要なのか

**Wavetableにない音色:**

```
音色の特徴:

Wavetable:
温かい
アナログ的
ベース、パッド

Operator:
明るい
金属的
ベル、ブラス

使い分け:

ベース → Wavetable
パッド → Wavetable
ドラム → Drum Rack

ベル → Operator (唯一の選択肢)
ブラス → Operator
金属音 → Operator

プロの使用率:

Wavetable: 70%
Operator: 15%
その他: 15%

用途:

メロディ要素:
ベル
チャイム

アクセント:
金属的ヒット
FXサウンド

特殊ベース:
FM Bass
歪んだベース

歴史:

FM合成:
1980年代 Yamaha DX7
伝説的
```

---

## FM合成とは

**周波数変調:**

### 基本原理

```
通常の合成 (Wavetable):

Oscillator → Filter → 音

FM合成:

Oscillator A (Carrier)
    ↑
    変調
    ↑
Oscillator B (Modulator)

= 複雑な倍音

用語:

Carrier (キャリア):
聴こえる音

Modulator (モジュレーター):
Carrierを変調
聴こえない

変調量:
大きい = 明るい、金属的
小さい = シンプル

効果:

豊富な倍音:
ベル音
金属音
ブラス
```

### Operator vs Wavetable

```
Wavetable:
波形選択 → Filter
シンプル
直感的

Operator:
4つのオシレーター
相互変調
複雑
パワフル

向き不向き:

Wavetable ○:
ベース
パッド
ストリングス

Operator ○:
ベル
ブラス
エレピ
金属音
```

---

## Operatorの構造

**4オシレーターシステム:**

### 全体像

```
┌─────────────────────────────┐
│ Algorithm (アルゴリズム)     │
│ オシレーター接続方法          │
└─────────────────────────────┘
           │
    ┌──────┴──────┐
    │             │
Oscillator     Oscillator
A (Carrier)    B (Modulator)
    │             │
Oscillator     Oscillator
C (Modulator)  D (Modulator)
    │
    ▼
┌─────────────────────────────┐
│ Filter (オプション)          │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│ Output                      │
└─────────────────────────────┘

4つのオシレーター:

A, B, C, D:
それぞれ独立

役割:
Carrier または Modulator

接続:
Algorithmで決定
```

### インターフェイス

```
上部:
┌─────────────────────────────┐
│ Algorithm Selector          │
│ 11種類から選択              │
└─────────────────────────────┘

中央:
┌──────┬──────┬──────┬──────┐
│ OSC A│ OSC B│ OSC C│ OSC D│
│      │      │      │      │
│ Freq │ Freq │ Freq │ Freq │
│ Level│ Level│ Level│ Level│
└──────┴──────┴──────┴──────┘

下部:
┌─────────────────────────────┐
│ Envelope × 4 (各OSC用)      │
│ ADSR表示                    │
└─────────────────────────────┘

右側:
┌─────────────────────────────┐
│ Global Controls             │
│ Volume, Pan, Transpose      │
└─────────────────────────────┘
```

---

## Algorithm（アルゴリズム）

**接続パターン:**

### 主要アルゴリズム

```
Algorithm 1 (シンプル):

A (Carrier)
↑
B (Modulator)

用途:
基本的なFM
ベル音

Algorithm 2:

A (Carrier)
↑
B
↑
C

用途:
より複雑な音色

Algorithm 4 (並列):

A + B (両方Carrier)

用途:
厚みのある音

Algorithm 8 (ブラス):

A (Carrier)
↑ ↑
B  C (両方がAを変調)

用途:
ブラス
複雑な倍音

推奨:

初心者:
Algorithm 1
シンプル

ベル音:
Algorithm 1, 2

ブラス:
Algorithm 8

実験:
全11種類試す
```

---

## Oscillator設定

**4つのオシレーター:**

### Frequency（周波数）

```
Coarse:
倍音比率
1x, 2x, 3x...

Fine:
微調整
-100 〜 +100 cent

Fixed:
固定周波数
Hz指定

ベル音設定:

OSC A (Carrier):
Coarse: 1x
基音

OSC B (Modulator):
Coarse: 4x
4倍音
明るい音

OSC C:
Coarse: 2.5x
非整数倍
金属的

ブラス設定:

OSC A:
Coarse: 1x

OSC B:
Coarse: 1x
Fine: +5 cent
わずかにデチューン

OSC C:
Coarse: 2x
```

### Level（レベル）

```
機能:
オシレーターの音量

Carrier:
聴こえる音量

Modulator:
変調量
= 倍音の豊かさ

ベル音:

OSC A (Carrier):
Level: 100%

OSC B (Modulator):
Level: 50-80%
適度な変調

ブラス:

OSC A:
Level: 100%

OSC B, C:
Level: 30-50%
控えめ
```

---

## Envelope（エンベロープ）

**各OSCに個別:**

### 4つのEnvelope

```
特徴:
各オシレーターが独立したADSR

ベル音:

OSC A Envelope:
Attack: 0 ms
Decay: 2000 ms (長い)
Sustain: 0%
Release: 1000 ms

→ 余韻が長い

OSC B Envelope:
Attack: 0 ms
Decay: 500 ms (短い)
Sustain: 20%
Release: 300 ms

→ アタックが明るい

効果:
時間とともに音色変化
リアルなベル

ブラス:

全OSC共通:
Attack: 100-300 ms
徐々に明るく

Decay: 500 ms
Sustain: 70%
Release: 300 ms

効果:
ブレスの感じ
```

---

## Filter（フィルター）

**オプション:**

### Filter設定

```
Operator のFilter:
必須ではない

理由:
FM自体が音色制御

使用する場合:

Type:
Lowpass

Cutoff:
2000-5000 Hz
高め

Resonance:
0-20%
控えめ

用途:
さらに暗くしたい
高域カット

ベル音:
Filterなし (推奨)

ブラス:
Cutoff 3000 Hz
わずかに暗く
```

---

## 実践: FM Bellの作り方

**10分で完成:**

### Step 1: 初期化 (1分)

```
1. 新規MIDIトラック

2. Operator挿入:
   Browser > Instruments > Operator

3. Init:
   右クリック > Initialize

4. 確認:
   C4ノート入力
```

### Step 2: Algorithm選択 (1分)

```
Algorithm:
Algorithm 1

理由:
シンプルなFM
A ← B

表示:
A: Carrier
B: Modulator
C, D: Off
```

### Step 3: Oscillator設定 (4分)

```
OSC A (Carrier):

Frequency:
Coarse: 1x (基音)

Level:
100%

OSC B (Modulator):

Frequency:
Coarse: 4x (4倍音)

Level:
70%

効果:
明るい金属的音

OSC C:

Frequency:
Coarse: 2.5x

Level:
30%

On: ☑

効果:
非整数倍 = より金属的

OSC D:

Off (使わない)
```

### Step 4: Envelope設定 (3分)

```
OSC A Envelope:

Attack: 0 ms
Decay: 3000 ms (長い余韻)
Sustain: 0%
Release: 2000 ms

OSC B Envelope:

Attack: 0 ms
Decay: 800 ms (短め)
Sustain: 10%
Release: 500 ms

効果:
アタック明るい
徐々に暗くなる

OSC C Envelope:

Attack: 0 ms
Decay: 1500 ms
Sustain: 0%
Release: 1000 ms
```

### Step 5: 仕上げ (1分)

```
Global:

Transpose:
+12 st (1オクターブ上)
ベルは高音域

Time:
100%
Envelope速度

Volume:
-3 dB

確認:
C5, E5, G5 コード
美しいベル音

保存:
"My FM Bell"
```

---

## 実践: Brassの作り方

**12分で完成:**

### Step 1: Algorithm (2分)

```
Algorithm:
Algorithm 8

構造:
A (Carrier)
↑ ↑
B  C (両方がAを変調)

効果:
複雑な倍音
ブラスらしい
```

### Step 2: Oscillator (5分)

```
OSC A:

Coarse: 1x
Level: 100%

OSC B:

Coarse: 1x
Fine: +5 cent
Level: 40%

効果:
わずかなデチューン
厚み

OSC C:

Coarse: 2x (2倍音)
Level: 35%

効果:
明るさ

OSC D:

Off
```

### Step 3: Envelope (4分)

```
全OSC共通設定:

Attack: 200 ms
徐々に明るく
ブレスの感じ

Decay: 400 ms

Sustain: 75%

Release: 250 ms

OSC B のみ:
Attack: 150 ms
わずかに速く
```

### Step 4: Filter (1分)

```
Filter:

Type: Lowpass

Cutoff: 3500 Hz

Resonance: 15%

効果:
わずかに暗く
アナログ的
```

### Step 5: 仕上げ

```
Global:

Volume: -6 dB
Transpose: 0

確認:
C3 - C4 範囲
ブラスセクション

保存:
"My Brass"
```

---

## よくある質問

### Q1: Algorithmの選び方は？

**A:** 音色による

```
シンプルな音:
Algorithm 1, 2
Carrier 1つ

複雑な音:
Algorithm 7, 8
複数Modulator

実験:

1. Algorithm 1から開始

2. 順に試す

3. 耳で判断

4. 気に入ったもの選択

推奨:

ベル: 1, 2, 3
ブラス: 7, 8
エレピ: 4, 6
```

### Q2: Modulatorのレベルは？

**A:** 控えめから

```
変調量:

少ない (20-30%):
シンプル
暗い

中程度 (50-60%):
適度な倍音
推奨

多い (80-100%):
非常に明るい
金属的
ノイズ的

調整方法:

1. 0% から開始

2. 徐々に上げる

3. 明るすぎる手前で停止

ベル: 60-80%
ブラス: 30-50%
```

### Q3: Wavetableと使い分けは？

**A:** 音色次第

```
Wavetable使用:

ベース
パッド
ストリングス
温かい音

Operator使用:

ベル
ブラス
エレピ
金属音
FX

迷ったら:

1. Wavetableから試す

2. 物足りない

3. Operator試す

4. どちらか選択

両方使う:

Wavetable: メイン
Operator: アクセント
```

---

## まとめ

### Operator基礎

```
□ FM合成 = 周波数変調
□ 4オシレーター
□ Algorithm = 接続方法
□ Carrier = 聴こえる音
□ Modulator = 倍音追加
```

### 音色別設定

```
ベル:
Algorithm 1, OSC B Level 70%

ブラス:
Algorithm 8, Attack 200ms

金属音:
非整数倍Coarse (2.5x, 3.7x等)
```

### 重要ポイント

```
□ Algorithm 1から始める
□ Modulatorは控えめ
□ Envelope各OSC個別
□ プリセット活用推奨
□ Wavetableと使い分け
```

---

**次は:** [Analog](./analog.md) - アナログシンセの温かみ
