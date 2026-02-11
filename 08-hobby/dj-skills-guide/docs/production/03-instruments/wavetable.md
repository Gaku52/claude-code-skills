# Wavetable

Ableton Live最強のシンセサイザー。ベース、リード、パッドまで、Techno/Houseに必要な全ての音色を作り出します。

## この章で学ぶこと

- Wavetableの基本構造
- 2つのオシレーター活用
- フィルター完全理解
- エンベロープ（ADSR）
- LFOとモジュレーション
- Sub Bassの作り方
- Techno Bassの作り方
- リード・パッドの作り方

---

## なぜWavetableが最重要なのか

**Ableton制作の70%:**

```
プロの音源使用率:

Wavetable: 70%
圧倒的No.1

Drum Rack: 20%
ドラム専用

その他: 10%
特殊音色

理由:

汎用性:
ベース、リード、パッド
全て作れる

音質:
非常に高品質
プロレベル

軽さ:
CPU負荷低い

直感的:
使いやすい

Techno/House向き:
モダンな音色

他のシンセとの比較:

Serum (外部):
$189
Wavetableとほぼ同等

Massive X (外部):
$199
やや複雑

結論:
Wavetable で十分
追加購入不要
```

---

## Wavetableの基本構造

**シグナルフロー:**

### 全体像

```
┌──────────────────────────────┐
│ Oscillator 1 (オシレーター1)  │
│ Wavetable選択、Position      │
└──────────┬───────────────────┘
           │
┌──────────▼───────────────────┐
│ Oscillator 2 (オシレーター2)  │
│ Wavetable選択、Position      │
└──────────┬───────────────────┘
           │
           │ Mix (ミックス)
           ▼
┌──────────────────────────────┐
│ Filter (フィルター)            │
│ Cutoff, Resonance            │
└──────────┬───────────────────┘
           │
┌──────────▼───────────────────┐
│ Amplifier (アンプ)            │
│ Volume                       │
└──────────┬───────────────────┘
           │
┌──────────▼───────────────────┐
│ Effects (エフェクト)          │
│ Chorus, Reverb等             │
└──────────────────────────────┘

制御:

Envelope (エンベロープ):
時間変化 (ADSR)

LFO:
周期的変化

Modulation Matrix:
パラメーター接続
```

### インターフェイス

```
上部:
┌────────────────────────────┐
│ OSC 1   OSC 2   SUB        │
│ [波形表示とPosition]        │
└────────────────────────────┘

中央左:
┌────────────────────────────┐
│ FILTER                     │
│ Cutoff  Resonance  Type    │
└────────────────────────────┘

中央右:
┌────────────────────────────┐
│ AMP                        │
│ Level  Pan                 │
└────────────────────────────┘

下部:
┌────────────────────────────┐
│ Envelopes (Amp, Filter)    │
│ ADSR表示                   │
└────────────────────────────┘

最下部:
┌────────────────────────────┐
│ LFO  Modulation Matrix     │
└────────────────────────────┘
```

---

## Oscillator（オシレーター）

**音の源:**

### Oscillator 1 & 2

```
Wavetable選択:

クリック:
Wavetableリスト表示

カテゴリ:
Basic: 基本波形
Vocal: ボーカル的
Digital: デジタル
Analog: アナログ的

推奨 (ベース):
Basic > Saw
Basic > Square
Analog > Fat Saw

推奨 (リード):
Digital > Sync Saw
Digital > FM
Vocal > Formant

Position (重要):

つまみ:
Wavetable内を移動

効果:
音色が変化
モーフィング

ベース:
Position 30-50%
太い音

リード:
Position 70-100%
明るい音

自動化:
Automation で動かす
= 音色変化

Transpose:

半音単位:
音程調整

ベース:
-12 (1オクターブ下)

リード:
0 または +12

Fine:

セント単位:
微調整

Detune:
OSC2を +7 cent
= 厚み

Level:

音量:
OSC1とOSC2のバランス

ベース:
OSC1: 100%
OSC2: 50%

リード:
両方 80-90%
```

### Sub Oscillator

```
位置:
OSC 1の右側

機能:
1オクターブ下のサイン波

用途:
低域補強
ベースの土台

設定:

Level:
0-30%

ベース:
20-30% (推奨)

リード:
0% (不要)

注意:
上げすぎると低域飽和
```

---

## Filter（フィルター）

**音色調整の核心:**

### Filter Type

```
種類:

Lowpass (LP):
最も使用
高域をカット
暗い音

Highpass (HP):
低域をカット
明るい音
ハイハット等

Bandpass (BP):
中域のみ残す
電話っぽい

Notch:
中域をカット
特殊効果

推奨:

ベース: Lowpass
リード: Lowpass
パッド: Lowpass

99%はLowpass
```

### Cutoff（カットオフ）

```
最重要パラメーター:

範囲:
20 Hz - 20,000 Hz

効果:
低い: 暗い音
高い: 明るい音

ベース推奨値:

Sub Bass: 100-300 Hz
暗い、重い

Techno Bass: 500-1000 Hz
やや明るい

Mid Bass: 1000-3000 Hz
明るい

リード:
2000-5000 Hz
明るく目立つ

パッド:
800-2000 Hz
柔らかい

Automation:

フィルタースイープ:
Cutoff 200 Hz → 8000 Hz
ビルドアップ効果
```

### Resonance（レゾナンス）

```
機能:
Cutoff周波数を強調

範囲:
0% - 100%

効果:

0%:
滑らか
自然

30%:
わずかに強調
推奨 (ベース)

60%:
はっきり強調
Acid Bass

100%:
自己発振
特殊効果

推奨値:

Sub Bass: 0-10%
自然

Techno Bass: 20-40%
存在感

Acid Bass: 60-80%
TB-303的

リード: 30-50%
明るさ
```

### Drive

```
機能:
倍音付加
歪み

範囲:
0% - 100%

効果:
温かみ
アナログ感

推奨値:

ベース: 10-30%
わずかに

リード: 20-40%

パッド: 0-10%
控えめ
```

---

## Envelope（エンベロープ）

**時間変化:**

### ADSR構造

```
A - Attack (アタック):
音の立ち上がり時間

D - Decay (ディケイ):
ピークから減衰

S - Sustain (サステイン):
持続レベル

R - Release (リリース):
鍵盤を離した後

波形:
    ┌─ピーク
    │╲
    │ ╲_______S (Sustain)
    │        ╲
    │         ╲___
    │             R
A   D
```

### Filter Envelope

```
機能:
Filter Cutoffを時間制御

Amount:
エンベロープの効果量
-100% 〜 +100%

ベース設定:

Attack: 0 ms
即座に

Decay: 300-800 ms
ゆっくり閉じる

Sustain: 20%
低めに

Release: 100 ms
短め

Amount: +50%
適度に

効果:
「ボーン」という音
```

### Amp Envelope

```
機能:
音量を時間制御

ベース設定:

Attack: 5 ms
わずかに
クリック音防止

Decay: 100 ms
短め

Sustain: 100%
フル

Release: 50 ms
短め

パッド設定:

Attack: 500-1000 ms
ゆっくり

Decay: 500 ms

Sustain: 80%

Release: 1000 ms
長め

効果:
ふわっと包み込む
```

---

## LFO（Low Frequency Oscillator）

**周期的変化:**

### LFO基本

```
機能:
パラメーターを周期的に変化

Rate:
速度
1/16 (速い)
1 Bar (遅い)

Shape:
波形
Sine (滑らか)
Square (カクカク)
Saw (ノコギリ)

Amount:
変化量
0-100%

用途:

ビブラート:
LFO → Pitch

トレモロ:
LFO → Volume

ワブワブ:
LFO → Filter Cutoff
```

### 実用例

```
フィルターLFO (推奨):

Target:
Filter Cutoff

Rate:
1/8 (8分音符)

Shape:
Sine

Amount:
30%

効果:
ワブワブ
Techno的

Pitchビブラート:

Target:
Pitch

Rate:
6 Hz (自由)

Shape:
Sine

Amount:
10%

効果:
わずかなビブラート
リード向き
```

---

## Modulation Matrix

**パラメーター接続:**

### 使い方

```
下部のModulation Matrix:

1. Source選択:
   LFO 1
   Envelope
   等

2. Amount調整:
   変化量

3. Target選択:
   Filter Cutoff
   Pitch
   等

例:

Source: LFO 1
Amount: +30%
Target: Filter Cutoff

→ LFOがフィルターを動かす

複数設定:

Source 1: LFO → Filter
Source 2: Envelope → Pitch
Source 3: LFO → Pan

→ 複雑な音色
```

---

## 実践: Sub Bassの作り方

**10分で完成:**

### Step 1: 初期化 (1分)

```
1. 新規MIDIトラック:
   Cmd+T

2. Wavetable挿入:
   Browser > Instruments > Wavetable

3. Init Preset:
   右クリック > Initialize

4. 確認:
   C2ノート入力
   音が鳴る
```

### Step 2: Oscillator設定 (3分)

```
OSC 1:

Wavetable:
Basic > Sine

Position:
0% (純粋なサイン波)

Transpose:
0

OSC 2:

Level: 0%
使わない

Sub:

Level: 0%
使わない

理由:
Sub Bassはシンプルに
```

### Step 3: Filter設定 (2分)

```
Filter Type:
Lowpass

Cutoff:
150 Hz
非常に暗い

Resonance:
0%

Drive:
10%
わずかに温かみ
```

### Step 4: Envelope (2分)

```
Filter Envelope:
Amount: 0%
使わない

Amp Envelope:

Attack: 5 ms
Decay: 50 ms
Sustain: 100%
Release: 50 ms

理由:
クリーン
シンプル
```

### Step 5: 確認とエクスポート (2分)

```
1. MIDIノート:
   C1, C2, F1, G1
   4小節パターン

2. 再生:
   Space

3. 調整:
   Cutoff 微調整
   100-200 Hz

4. プリセット保存:
   右クリック > Save Preset
   "My Sub Bass"

完成:
深い、クリーンなSub Bass
```

---

## 実践: Techno Bassの作り方

**15分で完成:**

### Step 1: Oscillator (5分)

```
OSC 1:

Wavetable:
Analog > Fat Saw

Position:
40%

Transpose:
0

OSC 2:

Wavetable:
Analog > Fat Saw

Position:
50%

Transpose:
0

Fine:
+7 cent (Detune)

Level:
70%

Sub:

Level:
25%
低域補強

効果:
厚み、パワー
```

### Step 2: Filter (4分)

```
Filter Type:
Lowpass

Cutoff:
800 Hz
明るめ

Resonance:
35%
Acid的

Drive:
25%
歪み
```

### Step 3: Filter Envelope (3分)

```
Amount:
+60%
しっかりと

Attack:
0 ms

Decay:
400 ms
ゆっくり閉じる

Sustain:
30%

Release:
100 ms

効果:
「ボーン」
Techno的なアタック
```

### Step 4: Amp Envelope (2分)

```
Attack:
5 ms

Decay:
200 ms

Sustain:
80%

Release:
100 ms
```

### Step 5: 仕上げ (1分)

```
Global:

Volume:
-6 dB
ヘッドルーム確保

Unison:
Voices: 2
Detune: 10%
さらに厚み

プリセット保存:
"My Techno Bass"

完成:
力強いTechno Bass
```

---

## よくある質問

### Q1: OSC2は必要？

**A:** ベースなら必須

```
OSC1のみ:
シンプル
細い

OSC1 + OSC2:
厚み
パワー

推奨:

Sub Bass:
OSC1のみ

Techno Bass:
OSC1 + OSC2

Detune:
OSC2を +7 cent
= 厚みが出る
```

### Q2: Cutoffの適正値は？

**A:** 音域による

```
Sub Bass (C1-C2):
100-300 Hz
非常に暗い

Mid Bass (C2-C3):
500-1500 Hz
やや明るい

Lead (C4-C5):
2000-5000 Hz
明るい

調整方法:

1. Cutoff 0%から開始

2. 徐々に上げる

3. ちょうど良いところで停止

4. 耳で判断
```

### Q3: プリセットから始めていい？

**A:** 絶対推奨

```
プリセット活用:

Browser > Wavetable:
数百のプリセット

カテゴリ:
Bass
Lead
Pad
等

方法:

1. "Bass"で検索

2. 近い音色選択

3. Cutoff, Resonance調整

4. 完成

時間:
ゼロから: 30分
プリセット: 5分

プロ:
70%はプリセットベース
```

---

## まとめ

### Wavetable基本構造

```
□ 2つのOscillator
□ Filter (Lowpass推奨)
□ Cutoff = 最重要
□ Filter Envelope = ベース必須
□ LFO = 動き追加
```

### ベース音作り

```
Sub Bass:
OSC1 Sine, Cutoff 150 Hz

Techno Bass:
OSC1+2 Saw, Cutoff 800 Hz, Resonance 35%

Acid Bass:
Resonance 60-80%, Filter Envelope強め
```

### 重要ポイント

```
□ プリセット活用推奨
□ Cutoffで明るさ調整
□ Resonanceで存在感
□ Filter Envelopeで動き
□ Detuneで厚み
```

---

**次は:** [Operator](./operator.md) - FMシンセで金属的サウンド
