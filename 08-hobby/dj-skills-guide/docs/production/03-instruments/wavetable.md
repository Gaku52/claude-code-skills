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

## Wavetableシンセサイザーの原理と仕組み

**デジタルシンセの革命:**

### Wavetableとは何か

```
従来のシンセ (アナログ):

オシレーター:
単一の波形生成
Sine, Saw, Square

制約:
波形が固定
変化させにくい

Wavetableシンセ:

原理:
複数の波形をテーブル化
波形間を補間

利点:
滑らかな音色変化
多彩な倍音構造

構造:

Wavetable = 256個の波形:
┌─────┬─────┬─────┬─────┐
│Wave1│Wave2│Wave3│...  │Wave256
└─────┴─────┴─────┴─────┘
 ↑Position で移動↑

補間:
Position 50% = Wave128
Position 50.5% = Wave128とWave129の中間

結果:
滑らかな音色変化
無限の倍音バリエーション
```

### デジタル波形生成の仕組み

```
サンプリング:

サンプルレート:
44.1 kHz (標準)
= 1秒に44,100回サンプル

波形生成:

1. テーブル読み出し:
   Positionに応じた波形選択

2. 周波数計算:
   MIDIノート → Hz変換
   A4 (440 Hz)

3. 位相増加:
   Phase += Frequency / Sample Rate

4. サンプル出力:
   Wavetable[Phase] → 音声信号

5. 次サンプルへ:
   繰り返し

精度:
32-bit float
非常に高精度

CPU負荷:
テーブル参照のみ
軽量
```

### Wavetableの構造詳細

```
1つのWavetable:

サイズ:
2048サンプル (典型)

内容:
1周期分の波形データ

例 (Saw波):
[0.0, 0.001, 0.002, ... 0.999, 1.0]

Wavetableセット:

数:
256個の波形

配置:
連続配列

Position:
0-255のインデックス

補間方法:

Linear補間:
Wave[n] * (1-t) + Wave[n+1] * t

Cubic補間 (高品質):
4点補間
より滑らか

Ableton Wavetable:
Cubic補間使用
高音質
```

---

## Ableton Wavetable全パラメータ詳細解説

**完全リファレンス:**

### Oscillatorセクション詳細

```
Wavetable選択エリア:

表示:
波形ビジュアル
リアルタイム更新

クリック:
Wavetableブラウザ表示

カテゴリ:
- Basic (16種)
- Analog (24種)
- Digital (32種)
- Vocal (18種)
- Mallets (12種)
- Misc (多数)

Position:

範囲: 0-100%
分解能: 0.01%
Modulation対応

Positionモジュレーション:
Envelope
LFO
Macro
Velocity

効果:
音色の時間変化
ダイナミックな倍音

Transpose:

範囲: -48 〜 +48半音
用途: オクターブ調整

推奨:
Bass: -12 (1オクターブ下)
Lead: 0
Pad: 0 または -12

Fine Tune:

範囲: -50 〜 +50 cent
分解能: 0.1 cent

用途:
Detune (厚み)
微調整

推奨Detune:
OSC2: +7 cent
または -7 cent

Level:

範囲: 0-100%
用途: オシレーター音量

推奨:
OSC1: 100%
OSC2: 50-80%

Pan:

範囲: L50 〜 R50
用途: ステレオ配置

Stereo技:
OSC1: L20
OSC2: R20
= ワイド
```

### Unisonセクション

```
Voices:

範囲: 1-8 voices
効果: 重ね合わせ

CPU負荷:
Voices数に比例

推奨:
Bass: 2-3
Lead: 4-6
Pad: 2-4

Detune:

範囲: 0-100%
効果: ピッチ分散

推奨値:
Subtle: 5-10%
Medium: 15-25%
Wide: 30-50%

Blend:

範囲: 0-100%
0%: Dry (原音)
100%: Wet (Unison)

推奨: 100%

```

### Filter詳細パラメータ

```
Filter Type (12種類):

Lowpass:
- LP 12dB: 緩やか
- LP 24dB: 標準 (推奨)
- LP 36dB: 急峻
- LP Legacy: ビンテージ

Highpass:
- HP 12dB
- HP 24dB

Bandpass:
- BP 12dB
- BP 24dB

その他:
- Notch
- Formant (ボーカル的)
- Phaser
- Comb (特殊)

推奨:
99%の場合: LP 24dB

Cutoff詳細:

範囲: 20 Hz - 20,000 Hz
表示: Hz表示
Modulation: 複数ソース可

Keyboard Tracking:
鍵盤に応じてCutoff追従
推奨: 50-100%

Resonance詳細:

範囲: 0-100%
効果: Q値上昇

Self-Oscillation:
90-100%で自己発振
ピッチ音が鳴る

用途:
特殊効果
サイレン音

Drive詳細:

範囲: 0-100%
種類: Soft clipping

倍音:
奇数倍音付加
温かみ

注意:
上げすぎると歪む
```

### Envelopeセクション完全版

```
Filter Envelope:

Amount:
-100% 〜 +100%

マイナス値:
Cutoffを下げる
逆モーション

プラス値:
Cutoffを上げる
通常使用

Attack:
範囲: 0 ms - 20,000 ms
推奨: 0-50 ms

Decay:
範囲: 0 ms - 20,000 ms
推奨: 200-800 ms

Sustain:
範囲: 0-100%
推奨: 20-40%

Release:
範囲: 0 ms - 20,000 ms
推奨: 50-200 ms

Curve:
-100 (Exponential)
0 (Linear)
+100 (Logarithmic)

Amp Envelope:

同様のADSR構造

推奨設定 (Bass):
A: 5 ms
D: 100 ms
S: 100%
R: 50 ms

推奨設定 (Pad):
A: 500-1000 ms
D: 500 ms
S: 70%
R: 1000-2000 ms

Envelope 3 (Modern):

Target:
任意のパラメータ

用途:
Position変化
Pan変化
エフェクト量変化
```

---

## ウェーブテーブルの作成とインポート

**カスタムWavetable:**

### Wavetableファイル形式

```
対応フォーマット:

.wav:
最も推奨
標準的

.aif / .aiff:
Apple形式
対応

要件:

サンプルレート:
44.1 kHz以上

ビット深度:
16-bit以上
24-bit推奨

チャンネル:
Mono (1ch)
必須

長さ:
2048サンプル × 256フレーム
= 約12秒 @ 44.1kHz
```

### Wavetableの作成方法

```
方法1: 既存音源から:

1. オーディオファイル準備:
   好きな音源 (シンセ、声、等)

2. Wavetable変換ツール:
   - Serum (Wavetable Editorモード)
   - WaveEdit (フリー)
   - Ableton自動認識

3. エクスポート:
   .wav形式

4. インポート:
   後述

方法2: Serumで作成:

1. Serum起動:
   無料デモ版可

2. Wavetable Editor:
   独立モード

3. 波形描画:
   マウスで描く
   または
   数式入力

4. Morph設定:
   256フレーム配置

5. Export:
   .wav出力

6. Ableton読み込み:
   後述

方法3: プログラミング:

Python + NumPy:

import numpy as np
import scipy.io.wavfile as wav

frames = 256
samples_per_frame = 2048
sr = 44100

wavetable = []

for i in range(frames):
    # 波形生成
    t = np.linspace(0, 2*np.pi, samples_per_frame)

    # 例: SawからSquareへモーフ
    morph = i / frames
    saw = scipy.signal.sawtooth(t)
    square = scipy.signal.square(t)
    wave = saw * (1-morph) + square * morph

    wavetable.append(wave)

# 連結
final = np.concatenate(wavetable)

# 正規化
final = final / np.max(np.abs(final))

# 保存
wav.write('custom.wav', sr,
          (final * 32767).astype(np.int16))

結果:
custom.wav = Wavetable
```

### Ableton Wavetableへのインポート

```
手順:

1. User Libraryフォルダ:

   Mac:
   ~/Music/Ableton/User Library/
   Presets/Instruments/Wavetable/Wavetables/

   Win:
   Documents/Ableton/User Library/
   Presets/Instruments/Wavetable/Wavetables/

2. .wavファイル配置:
   上記フォルダに custom.wav コピー

3. Ableton再起動:
   または
   Rescan (Browser右クリック)

4. Wavetable読み込み:

   Wavetable起動 >
   OSC選択 >
   Wavetableブラウザ >
   User > custom

5. 確認:
   波形表示
   Position動かす
   音色変化確認

自動変換:

Ableton機能:
.wavを自動でWavetable化

条件:
2048サンプルの倍数
モノラル

処理:
FFT解析
フレーム分割
自動配置
```

---

## ジャンル別プリセット設計

**実践的音色作り:**

### Techno Bass設計

```
基本コンセプト:
力強い
存在感
グルーヴ

Oscillator:

OSC1:
Wavetable: Analog > Fat Saw
Position: 35%
Transpose: 0
Level: 100%

OSC2:
Wavetable: Analog > Fat Saw
Position: 42%
Transpose: 0
Fine: +7 cent
Level: 65%

Sub:
Level: 28%

Filter:

Type: LP 24dB
Cutoff: 750 Hz
Resonance: 32%
Drive: 22%
Keyboard Tracking: 60%

Filter Envelope:

Amount: +58%
Attack: 0 ms
Decay: 380 ms
Sustain: 25%
Release: 95 ms

Amp Envelope:

Attack: 3 ms
Decay: 150 ms
Sustain: 85%
Release: 80 ms

LFO (オプション):

Target: Filter Cutoff
Rate: 1/16
Shape: Sine
Amount: 12%
効果: 微妙な動き

結果:
定番Techno Bass
即戦力
```

### Progressive House Lead

```
コンセプト:
明るい
エモーショナル
メロディック

Oscillator:

OSC1:
Wavetable: Digital > Sync Saw
Position: 68%
Transpose: 0
Level: 90%

OSC2:
Wavetable: Digital > FM Bell
Position: 55%
Transpose: +12 (1オクターブ上)
Fine: -5 cent
Level: 45%

Sub: 0%

Unison:
Voices: 5
Detune: 18%
Blend: 100%

Filter:

Type: LP 24dB
Cutoff: 3500 Hz
Resonance: 38%
Drive: 15%

Filter Envelope:

Amount: +35%
Attack: 20 ms
Decay: 600 ms
Sustain: 50%
Release: 400 ms

Amp Envelope:

Attack: 15 ms
Decay: 300 ms
Sustain: 75%
Release: 350 ms

LFO:

Target: Pitch
Rate: 5.8 Hz
Shape: Sine
Amount: 8%
効果: ビブラート

Effects (内蔵):

Chorus: On
Depth: 30%
Rate: 0.8 Hz

Reverb: On
Size: Medium
Mix: 25%

結果:
感動的なリード
メロディ向き
```

### Deep House Pad

```
コンセプト:
温かい
包み込む
ムーディー

Oscillator:

OSC1:
Wavetable: Analog > Detune Saw
Position: 25%
Transpose: -12
Level: 80%

OSC2:
Wavetable: Vocal > Breath
Position: 40%
Transpose: -12
Fine: +11 cent
Level: 55%

Sub: 18%

Unison:
Voices: 3
Detune: 22%

Filter:

Type: LP 24dB
Cutoff: 1200 Hz
Resonance: 12%
Drive: 8%

Filter Envelope:

Amount: +18%
Attack: 150 ms
Decay: 800 ms
Sustain: 60%
Release: 1200 ms

Amp Envelope:

Attack: 650 ms
Decay: 400 ms
Sustain: 70%
Release: 1500 ms

LFO 1:

Target: Filter Cutoff
Rate: 1/2 (2小節)
Shape: Triangle
Amount: 15%

LFO 2:

Target: Pan
Rate: 1/4
Shape: Sine
Amount: 25%
効果: ステレオ動き

Effects:

Chorus: On
Depth: 45%
Rate: 0.4 Hz

Reverb: On
Size: Large
Mix: 40%

EQ:
High Cut: 8 kHz
Low Cut: 150 Hz

結果:
深みのあるパッド
背景に最適
```

### Acid Bass (TB-303スタイル)

```
コンセプト:
キレ
Squelchy
グリッシー

Oscillator:

OSC1:
Wavetable: Basic > Saw
Position: 0%
Transpose: -12
Level: 100%

OSC2: Off

Sub: 0%

Filter (最重要):

Type: LP 24dB
Cutoff: 450 Hz
Resonance: 72% (高め!)
Drive: 35%
Keyboard Tracking: 30%

Filter Envelope (最重要):

Amount: +78% (高め!)
Attack: 0 ms
Decay: 180 ms
Sustain: 5% (低め!)
Release: 80 ms
Curve: -40 (Exponential)

Amp Envelope:

Attack: 1 ms
Decay: 80 ms
Sustain: 100%
Release: 30 ms

Accent (Velocity対応):

Filter Env Amount:
Via Velocity → +50%

効果:
強く弾く = より開く

使い方:

MIDIノート:
C1, C2, D#1, F1
16分音符
ランダムVelocity

Slide/Glide:
Glide Time: 60 ms
一部ノートをレガート

結果:
本格的303サウンド
Acid House即戦力
```

---

## モジュレーションマトリクスの活用

**複雑な音色変化:**

### Modulation Matrixの構造

```
位置:
Wavetable下部
Matrix表示

構成:

8スロット:
最大8つのモジュレーション設定

各スロット要素:

Source (ソース):
- LFO 1, 2
- Envelope 1, 2, 3
- MIDI (Velocity, Modwheel, Aftertouch)
- Macro 1-8
- Random

Amount (量):
-100% 〜 +100%

Target (ターゲット):
- OSC1/2 Position
- Filter Cutoff
- Filter Resonance
- Pan
- Level
- Unison Detune
- 等 (30種類以上)

仕組み:

Source値 × Amount = Target変化量

例:
LFO1 (0-1) × 50% → Cutoff +50%変化
```

### 実践的なModulation設定

```
設定例1: ダイナミックベース

Slot 1:
Source: Envelope 2
Amount: +45%
Target: OSC1 Position
効果: アタック時に音色変化

Slot 2:
Source: LFO 1 (Rate: 1/16)
Amount: +25%
Target: Filter Cutoff
効果: リズミカルな動き

Slot 3:
Source: Velocity
Amount: +60%
Target: Filter Envelope Amount
効果: 強く弾くとより明るく

結果:
表現力豊かなベース

設定例2: エボリューションパッド

Slot 1:
Source: LFO 1 (Rate: 4 Bars)
Amount: +70%
Target: OSC1 Position
効果: ゆっくり音色変化

Slot 2:
Source: LFO 2 (Rate: 1 Bar)
Amount: +30%
Target: Filter Cutoff
効果: 中速の明るさ変化

Slot 3:
Source: Envelope 2
Amount: -20%
Target: OSC2 Level
効果: アタック時にOSC2減少

Slot 4:
Source: Random
Amount: +15%
Target: Pan
効果: ランダムなステレオ移動

結果:
有機的に変化するパッド

設定例3: グリッチリード

Slot 1:
Source: LFO 1 (Rate: 1/32, Shape: Sample&Hold)
Amount: +80%
Target: OSC1 Position
効果: 高速ランダム音色変化

Slot 2:
Source: LFO 2 (Rate: 1/16, Shape: Square)
Amount: +50%
Target: Filter Cutoff
効果: カクカクしたフィルター

Slot 3:
Source: Modwheel
Amount: +100%
Target: LFO 1 Amount
効果: Modwheelで効果量制御

結果:
制御可能なグリッチサウンド
```

### 高度なModulation技法

```
技法1: Macro連携

Macro設定:
Macro 1 = "Brightness"
→ Filter Cutoff
→ Resonance
→ Drive

Modulation追加:
Source: Macro 1
Amount: +40%
Target: OSC1 Position

効果:
1つのMacroで複数パラメータ制御
ライブ演奏向き

技法2: LFOのLFO

Slot 1:
Source: LFO 2 (Rate: 8 Bars)
Amount: +100%
Target: LFO 1 Rate

効果:
LFO速度が変化
複雑な周期

技法3: Envelope連鎖

Slot 1:
Source: Envelope 2
Amount: +50%
Target: Filter Envelope Amount

効果:
エンベロープの深さが変化
ダブルエンベロープ

技法4: Velocity Layering

Slot 1:
Source: Velocity
Amount: -100%
Target: OSC1 Level

Slot 2:
Source: Velocity
Amount: +100%
Target: OSC2 Level

効果:
弱く弾く = OSC1のみ
強く弾く = OSC2のみ
Velocity切り替え
```

---

## Serum/Vitalとの比較

**他シンセとの違い:**

### 機能比較表

```
           Wavetable  Serum    Vital
────────────────────────────────────
価格:      無料*      $189     無料
*Ableton付属

Wavetable数: 150+     450+     400+

Oscillator:  2 + Sub  2 + Sub  3 + Sample

Filter:      12種類   14種類   10種類

LFO:         2        4        4

Envelope:    3        4        6

Modulation:  8スロット 無制限   無制限

CPU負荷:     軽い     中程度   軽い

音質:        最高     最高     最高

UI:          シンプル リッチ   モダン

結論:
Wavetable = 十分な機能
追加購入不要
```

### 音質・音色の違い

```
Wavetable (Ableton):

特徴:
クリーン
正確
安定

向いている:
Techno
Minimal
House

フィルター:
正確
音楽的

Serum (Xfer):

特徴:
太い
温かい
アナログ的

向いている:
Bass Music
Dubstep
Future Bass

フィルター:
キャラクター強め
Resonance良好

Vital (Matt Tytel):

特徴:
モダン
エッジ
デジタル

向いている:
Future House
Psytrance
Experimental

フィルター:
多機能
実験的

実際の使い分け:

Techno/House制作:
Wavetable 100%で完結

Bass強化:
Serum追加検討

実験的:
Vital試用

プロの選択:
Wavetable + Serum
両方使用が多い
```

### Wavetableの利点

```
利点1: 統合性

Ableton完全統合:
保存、読み込み高速
ブラウザ統一
CPU共有最適化

他社シンセ:
独立プラグイン
オーバーヘッドあり

利点2: 安定性

クラッシュ:
ほぼ無し

他社:
稀にクラッシュ
プラグイン依存

利点3: 軽さ

CPU使用率:
Wavetable: 2-3%
Serum: 4-6%
Vital: 2-4%

結論:
Wavetableが最軽量

利点4: ワークフロー

保存先:
Abletonプロジェクト内
自動バックアップ

他社:
別フォルダ
手動管理必要

利点5: アップデート

Ableton連動:
Live更新時に自動改善

他社:
別途確認・更新必要
```

### 欠点と対処法

```
欠点1: Wavetable数

Wavetable: 150+
Serum: 450+

対処法:
カスタムWavetable作成
前述のインポート方法活用

欠点2: Modulation数

Wavetable: 8スロット
Serum/Vital: 無制限

対処法:
本当に必要な8つに絞る
複雑すぎは逆効果

欠点3: エフェクト

Wavetable: 基本のみ
Serum: リッチ

対処法:
Abletonエフェクト使用
より柔軟

欠点4: ビジュアル

Wavetable: シンプル
Serum: 派手

影響:
音質には無関係
好み次第

結論:

Techno/House制作:
Wavetable単独で十分

他ジャンル:
必要に応じて追加検討

初心者:
まずWavetable完全習得
他は後で
```

---

## 実践的Wavetableワークフロー

**プロの制作手順:**

### 新規トラック作成から完成まで

```
Step 1: プリセット選択 (2分)

1. Browser開く:
   Cmd+Option+B

2. Instruments > Wavetable:
   カテゴリ選択

3. 試聴:
   プリセットクリック
   C2ノート試奏

4. 近い音色選択:
   完璧でなくてOK
   70%合っていれば十分

Step 2: 基本調整 (3分)

1. Filter Cutoff:
   上下に動かす
   理想の明るさ探す

2. Resonance:
   0%から徐々に上げる
   存在感調整

3. Filter Envelope Amount:
   動きの深さ調整

4. 試奏:
   MIDIノート入力
   確認

Step 3: Detune/Unison (2分)

1. OSC2 Fine:
   +7 cent設定
   厚み確認

2. Unison Voices:
   2-4に設定

3. Unison Detune:
   10-20%に設定

4. 比較:
   Unison On/Off切り替え
   効果確認

Step 4: Modulation追加 (3分)

1. LFO 1設定:
   Target: Filter Cutoff
   Rate: 1/8
   Amount: 15%

2. 再生:
   動きを確認

3. 調整:
   Amount微調整

4. オプション:
   必要に応じて追加Modulation

Step 5: 保存 (1分)

1. プリセット保存:
   右クリック > Save Preset

2. 命名:
   "My Techno Bass 001"
   番号付けで管理

3. カテゴリ:
   User > Bass

完成:
10分で即戦力プリセット
```

### プリセット管理のベストプラクティス

```
フォルダ構造:

User Library/Wavetable/
├── Bass/
│   ├── Sub/
│   ├── Techno/
│   └── Acid/
├── Lead/
│   ├── Main/
│   └── Pluck/
├── Pad/
│   ├── Dark/
│   └── Bright/
└── FX/
    └── Sweep/

命名規則:

[ジャンル]_[音色]_[番号]

例:
Techno_Bass_001
House_Lead_Warm_003
Minimal_Pad_Dark_002

利点:
検索しやすい
即座に見つかる

タグ付け:

Abletonタグ機能:
右クリック > Edit Info

Tags:
Heavy, Bright, Dark, Soft等

Color:
視覚的分類
Bass = 赤
Lead = 青
Pad = 緑

バックアップ:

定期的:
月1回

方法:
User Libraryフォルダ丸ごとコピー

保存先:
外付けHDD
クラウド
```

### CPU負荷管理

```
Wavetable最適化:

Voices削減:
Unison 8 → 4
CPU 50%削減

不要なOSC:
OSC2使わない場合
Level 0%にする
若干軽量化

Filter Type:
LP 24dB: 標準
LP 36dB: やや重い
推奨: LP 24dB

Modulation最小化:
本当に必要なもののみ

フリーズ機能:

使い方:
トラック右クリック > Freeze Track

効果:
CPU使用率 → ほぼ0%

注意:
編集不可になる
完成後に実施

推奨タイミング:
10トラック超えたら
重いトラックからFreeze

Flatten:

さらに軽量化:
Freeze後
右クリック > Flatten

効果:
オーディオ化
完全に軽量

注意:
元に戻せない
必ずプロジェクト保存後
```

---

## トラブルシューティング

**よくある問題と解決:**

### 音が出ない

```
原因1: トラックミュート

確認:
トラック名左のアイコン
グレー = ミュート

解決:
クリックで解除

原因2: Wavetable音量ゼロ

確認:
Wavetable右上 Output Level

解決:
-6 dB程度に設定

原因3: Filter Cutoff低すぎ

確認:
Cutoff値確認

解決:
500 Hz以上に上げる

原因4: Envelope設定ミス

確認:
Amp Envelope Sustain = 0%

解決:
Sustain 100%に設定
```

### 音が細い・弱い

```
原因1: OSC1のみ使用

解決:
OSC2追加
Fine +7 cent
Level 60%

原因2: Unison未使用

解決:
Unison Voices: 3
Detune: 15%

原因3: Sub不足

解決:
Sub Level: 25%

原因4: Filter開きすぎ

解決:
Cutoff下げる
Resonance上げる

原因5: Drive不足

解決:
Drive 20%に設定
```

### 音が歪む・割れる

```
原因1: Output Level高すぎ

確認:
Wavetable Out メーター赤

解決:
Output Level -10 dB

原因2: Drive上げすぎ

確認:
Drive 80%以上

解決:
Drive 30%以下に

原因3: Resonance自己発振

確認:
Resonance 90%以上

解決:
Resonance 60%以下に

原因4: Unison Voices多すぎ

確認:
Voices 8

解決:
Voices 3-4に削減

原因5: 複数トラック合算

解決:
Master音量確認
各トラック -6 dB
```

### CPUが重い

```
原因1: Unison高設定

確認:
Voices 6-8

解決:
Voices 2-3に削減
Detune調整で補う

原因2: 複数Wavetableインスタンス

確認:
10トラック以上

解決:
Freeze Track実行
重い順から

原因3: エフェクト過多

確認:
各トラックエフェクト数

解決:
不要なエフェクト削除
Send/Returnに集約

原因4: 高サンプルレート

確認:
Preferences > Sample Rate

解決:
96kHz → 44.1kHz
体感差なし

解決策まとめ:

□ Freeze Track
□ Unison Voices削減
□ 不要エフェクト削除
□ サンプルレート44.1k
```

---

## 応用テクニック集

**上級者向け:**

### レイヤリング技法

```
技法1: オクターブレイヤー

トラック1:
Wavetable Bass (C1)
Cutoff 200 Hz
Sub 30%

トラック2:
Wavetable Bass (C2)
Cutoff 1500 Hz
Sub 0%

Mix:
トラック1: -6 dB
トラック2: -12 dB

効果:
奥行き
パワフル

技法2: 音色レイヤー

トラック1:
Saw波ベース
Cutoff 800 Hz

トラック2:
FM波ベース
Cutoff 2000 Hz

トラック3:
Sub Sine
Cutoff 150 Hz

Mix:
1: -6 dB
2: -15 dB
3: -10 dB

効果:
複雑な倍音
プロの音

技法3: ステレオレイヤー

トラック1 (L):
Wavetable Lead
Pan: L40
Unison Detune: 15%

トラック2 (R):
同プリセット
Pan: R40
Unison Detune: 18% (微妙に違う)

効果:
超ワイドステレオ
```

### Automation活用

```
技法1: Filter Sweep (ビルドアップ)

16小節:
Cutoff 300 Hz → 8000 Hz
Linear上昇

Resonance:
20% → 45%
連動上昇

タイミング:
ドロップ前16小節

効果:
緊張感
期待感

技法2: Position Morphing

8小節:
Position 0% → 100%
音色変化

Automation Shape:
Exponential

タイミング:
ブレイク中

効果:
進化する音色

技法3: Resonance Pumping

Rate: 1/4
Pattern:
0% → 60% → 0%
ノコギリ波

Sync: BPM連動

効果:
リズミカル
Acid的

技法4: Pan Movement

LFO使わず手動:

8小節:
L → Center → R → Center

Shape:
Sine曲線

効果:
空間的動き
```

### サイドチェイン連携

```
設定1: Kick連動Filter

Wavetable:
通常設定

Audio Effect Rack:
Compressor追加

Sidechain:
Input: Kick

Compressor設定:
Threshold: -20 dB
Ratio: 4:1
Attack: 1 ms
Release: 150 ms

Target:
Compressor → Filter Cutoff (Modulation)

効果:
Kickに合わせてFilter閉じる
グルーヴ感

設定2: Kick連動Volume

同様設定:
Target: Volume

効果:
Kickと住み分け
ミックス明瞭

設定3: Hat連動Resonance

Sidechain Input:
Hi-Hat

Target:
Resonance

Modulation Amount:
+30%

効果:
Hatに反応
リズム感
```

### Macro設定例

```
Macro 1: "Brightness"

Control:
Filter Cutoff (0-100%)
Resonance (0-50%)
Drive (0-40%)
OSC1 Position (0-50%)

効果:
1ノブで明るさ完全制御

Macro 2: "Movement"

Control:
LFO 1 Amount (0-100%)
LFO 1 Rate (1/16 - 2 Bars)
Filter Envelope Amount (0-80%)

効果:
動きの深さ制御

Macro 3: "Width"

Control:
Unison Voices (1-6)
Unison Detune (0-40%)
OSC2 Fine (-12 to +12 cent)

効果:
ステレオ幅制御

Macro 4: "Attack"

Control:
Amp Attack (0-500 ms)
Filter Envelope Attack (0-200 ms)
Filter Cutoff (-20% to +20%)

効果:
立ち上がり制御

使い方:

ライブ演奏:
MIDIコントローラー割当

Automation:
楽曲展開で動かす

プリセット:
Macroごと保存
```

---

## プロ直伝: 即戦力プリセットライブラリ

**コピペで使える設定集:**

### ジャンル特化プリセット30選

```
=== TECHNO (10種) ===

T01: Rolling Bass
OSC1: Analog > Fat Saw, Pos 38%
OSC2: Analog > Fat Saw, Pos 45%, Fine +7
Sub: 26%
Filter: LP 24dB, Cut 720Hz, Res 28%, Drive 24%
F.Env: Amt +55%, A 0, D 420, S 22%, R 90
A.Env: A 2, D 180, S 90%, R 75
LFO1 → Cutoff: 1/16, 18%

T02: Punchy Stab
OSC1: Basic > Saw, Pos 0%
OSC2: Off
Sub: 0%
Filter: LP 24dB, Cut 1800Hz, Res 42%
F.Env: Amt +72%, A 0, D 180, S 0%, R 50
A.Env: A 1, D 120, S 0%, R 30
Unison: 2 voices, 8%

T03: Deep Sub
OSC1: Basic > Sine, Pos 0%
OSC2: Off
Sub: 0%
Filter: LP 24dB, Cut 180Hz, Res 0%, Drive 12%
F.Env: Amt 0%
A.Env: A 8, D 60, S 100%, R 55
純粋サイン波、Technoの土台

T04: Industrial Lead
OSC1: Digital > Sync Saw, Pos 75%
OSC2: Digital > FM, Pos 62%, Transpose +12
Sub: 0%
Filter: LP 24dB, Cut 4200Hz, Res 48%
F.Env: Amt +38%, A 15, D 550, S 55%, R 380
Unison: 5, Detune 22%
LFO1 → Pitch: 6.2Hz, 6% (ビブラート)

T05: Acid Squelch
OSC1: Basic > Saw, Pos 0%
OSC2: Off
Sub: 0%
Filter: LP 24dB, Cut 520Hz, Res 78%, Drive 38%
F.Env: Amt +85%, A 0, D 160, S 3%, R 75
A.Env: A 0, D 65, S 100%, R 25
Velocity → F.Env Amt: +55%
303スタイル完全再現

T06: Dark Rumble
OSC1: Analog > Detune Saw, Pos 22%
OSC2: Analog > Detune Saw, Pos 28%, Fine -5
Sub: 32%
Filter: LP 36dB, Cut 280Hz, Res 8%, Drive 15%
F.Env: Amt +22%, A 0, D 850, S 40%, R 180
LFO1 → Cutoff: 2 Bars, 12%
ダークで重いサブ

T07: Plucky Bass
OSC1: Analog > Fat Saw, Pos 48%
OSC2: Off
Sub: 18%
Filter: LP 24dB, Cut 1200Hz, Res 35%
F.Env: Amt +68%, A 0, D 85, S 0%, R 40
A.Env: A 0, D 95, S 0%, R 35
短く切れるベース

T08: Minimal Groove
OSC1: Basic > Square, Pos 15%
OSC2: Basic > Saw, Pos 18%, Fine +11
Sub: 22%
Filter: LP 24dB, Cut 650Hz, Res 24%, Drive 18%
F.Env: Amt +42%, A 0, D 380, S 28%, R 85
LFO1 → F.Cutoff: 1/8, 22%
ミニマルな動き

T09: Heavy Kick Bass
OSC1: Basic > Sine, Pos 0%
OSC2: Off
Sub: 35%
Filter: LP 24dB, Cut 95Hz, Res 0%, Drive 25%
F.Env: Amt +48%, A 0, D 320, S 0%, R 180
A.Env: A 0, D 250, S 0%, R 200
Kickと一体化するベース

T10: Modular Style
OSC1: Digital > Harm Sync, Pos 55%
OSC2: Digital > Wasp, Pos 68%, Transpose +7
Sub: 0%
Filter: BP 12dB, Cut 880Hz, Res 55%
F.Env: Amt +62%, A 5, D 280, S 18%, R 120
LFO1 → Pos1: 1/4, 45%
LFO2 → Res: 1/3, 28%
複雑なモジュレーション

=== HOUSE (10種) ===

H01: Classic House Bass
OSC1: Analog > Fat Saw, Pos 32%
OSC2: Analog > Fat Saw, Pos 38%, Fine +7
Sub: 28%
Filter: LP 24dB, Cut 850Hz, Res 32%, Drive 22%
F.Env: Amt +52%, A 0, D 350, S 25%, R 95
A.Env: A 5, D 160, S 88%, R 80
定番ハウスベース

H02: Deep House Pad
OSC1: Analog > Detune Saw, Pos 25%
OSC2: Vocal > Breath, Pos 38%, Transpose -12, Fine +9
Sub: 18%
Filter: LP 24dB, Cut 1350Hz, Res 15%, Drive 8%
F.Env: Amt +22%, A 180, D 720, S 65%, R 1300
A.Env: A 720, D 420, S 72%, R 1600
Unison: 3, Detune 20%
温かいパッド

H03: Disco Stab
OSC1: Basic > Saw, Pos 0%
OSC2: Basic > Square, Pos 0%, Transpose +12
Sub: 0%
Filter: HP 12dB, Cut 320Hz, Res 18%
F.Env: Amt -28%, A 0, D 150, S 40%, R 180
A.Env: A 0, D 220, S 0%, R 120
Unison: 4, Detune 12%
ディスコスタブ

H04: Piano House Chord
OSC1: Mallets > Piano, Pos 45%
OSC2: Off
Sub: 0%
Filter: LP 12dB, Cut 5500Hz, Res 8%
F.Env: Amt +18%, A 25, D 450, S 60%, R 550
A.Env: A 12, D 380, S 68%, R 650
ピアノハウス的

H05: Vocal Chop
OSC1: Vocal > Formant, Pos 62%
OSC2: Vocal > Breath, Pos 48%, Fine -8
Sub: 0%
Filter: BP 12dB, Cut 1800Hz, Res 42%
F.Env: Amt +35%, A 8, D 220, S 35%, R 280
LFO1 → Pos1: 1/16, 55%
LFO2 → Filter: 1/8, 22%
ボーカルチョップ的

H06: Progressive Lead
OSC1: Digital > Sync Saw, Pos 68%
OSC2: Digital > FM Bell, Pos 52%, Transpose +12, Fine -4
Sub: 0%
Filter: LP 24dB, Cut 3800Hz, Res 38%, Drive 16%
F.Env: Amt +32%, A 22, D 580, S 52%, R 420
A.Env: A 18, D 320, S 78%, R 360
Unison: 5, Detune 19%
LFO1 → Pitch: 5.5Hz, 7%
プログレッシブリード

H07: Funky Pluck
OSC1: Analog > Fat Saw, Pos 55%
OSC2: Off
Sub: 8%
Filter: LP 24dB, Cut 2200Hz, Res 45%
F.Env: Amt +75%, A 0, D 120, S 8%, R 65
A.Env: A 0, D 150, S 12%, R 80
ファンキープラック

H08: Soulful Organ
OSC1: Basic > Sine, Pos 0%
OSC2: Basic > Sine, Pos 0%, Transpose +12
Sub: 0%
Filter: LP 12dB, Cut 4200Hz, Res 5%
F.Env: Amt +8%, A 45, D 180, S 88%, R 320
A.Env: A 55, D 220, S 85%, R 450
Unison: 2, Detune 5%
オルガン的

H09: Future Bass
OSC1: Digital > FM, Pos 72%
OSC2: Digital > Wasp, Pos 58%, Transpose +7
Sub: 15%
Filter: LP 24dB, Cut 2800Hz, Res 35%, Drive 28%
F.Env: Amt +48%, A 12, D 420, S 42%, R 350
Unison: 6, Detune 24%
LFO1 → Cutoff: 1/16, 28%
フューチャーベース

H10: Tropical Synth
OSC1: Mallets > Marimba, Pos 38%
OSC2: Digital > Bell, Pos 55%, Transpose +12
Sub: 0%
Filter: LP 12dB, Cut 6800Hz, Res 12%
F.Env: Amt +22%, A 18, D 320, S 55%, R 480
A.Env: A 15, D 280, S 62%, R 550
トロピカル

=== AMBIENT / EXPERIMENTAL (10種) ===

A01: Evolving Pad
OSC1: Analog > Detune Saw, Pos 28%
OSC2: Vocal > Breath, Pos 42%, Transpose -12
Sub: 12%
Filter: LP 24dB, Cut 1450Hz, Res 18%, Drive 5%
F.Env: Amt +15%, A 220, D 850, S 68%, R 1800
A.Env: A 850, D 520, S 75%, R 2200
Unison: 3, Detune 28%
LFO1 → Pos1: 8 Bars, 72%
LFO2 → Cutoff: 2 Bars, 32%
進化し続けるパッド

A02: Texture Drone
OSC1: Digital > Noise, Pos 85%
OSC2: Vocal > Formant, Pos 62%, Fine +15
Sub: 0%
Filter: BP 24dB, Cut 1200Hz, Res 62%
F.Env: Amt +25%, A 380, D 1200, S 58%, R 2500
LFO1 → Res: 4 Bars, 45%
LFO2 → Pan: 1 Bar, 38%
テクスチャドローン

A03: Granular Cloud
OSC1: Digital > Granular, Pos 68%
OSC2: Digital > Granular, Pos 75%, Fine -18
Sub: 0%
Filter: LP 24dB, Cut 3200Hz, Res 28%
F.Env: Amt +18%, A 550, D 980, S 62%, R 1500
Unison: 6, Detune 42%
LFO1 → Pos1: 1/32 (S&H), 85%
グラニュラークラウド

A04: Metallic Shimmer
OSC1: Digital > FM Bell, Pos 88%
OSC2: Digital > FM, Pos 72%, Transpose +24
Sub: 0%
Filter: HP 12dB, Cut 850Hz, Res 22%
F.Env: Amt +28%, A 120, D 650, S 48%, R 1200
A.Env: A 180, D 520, S 55%, R 1800
Unison: 7, Detune 32%
LFO1 → Pitch: 0.2Hz, 4%
メタリックシマー

A05: Dark Atmosphere
OSC1: Analog > Detune Saw, Pos 12%
OSC2: Vocal > Dark, Pos 22%, Transpose -24
Sub: 25%
Filter: LP 36dB, Cut 420Hz, Res 5%, Drive 8%
F.Env: Amt +8%, A 850, D 1500, S 72%, R 3200
A.Env: A 1200, D 980, S 78%, R 4500
LFO1 → Cutoff: 16 Bars, 15%
ダークアトモスフィア

A06: Cinematic Swell
OSC1: Analog > Fat Saw, Pos 35%
OSC2: Vocal > Breath, Pos 55%, Transpose -12, Fine +12
Sub: 20%
Filter: LP 24dB, Cut 2200Hz, Res 22%, Drive 12%
F.Env: Amt +35%, A 980, D 1200, S 68%, R 2500
A.Env: A 1500, D 850, S 75%, R 3200
Unison: 4, Detune 25%
Env3 → Pos1: +65%, A 2500, D 1800, S 72%
シネマティックスウェル

A07: Glitch Texture
OSC1: Digital > Harm Sync, Pos 78%
OSC2: Digital > Wasp, Pos 65%, Fine +22
Sub: 0%
Filter: Notch, Cut 1800Hz, Res 48%
F.Env: Amt +45%, A 0, D 85, S 22%, R 120
LFO1 → Pos1: 1/32 (S&H), 92%
LFO2 → Cutoff: 1/16 (Square), 58%
グリッチテクスチャ

A08: Ethereal Voice
OSC1: Vocal > Formant, Pos 48%
OSC2: Vocal > Breath, Pos 62%, Transpose +7
Sub: 0%
Filter: BP 12dB, Cut 2800Hz, Res 35%
F.Env: Amt +22%, A 420, D 850, S 62%, R 1500
A.Env: A 650, D 520, S 72%, R 1800
Unison: 4, Detune 18%
LFO1 → Pos1: 2 Bars, 42%
LFO2 → Pitch: 4.5Hz, 8%
エーテルボイス

A09: Sci-Fi FX
OSC1: Digital > FM, Pos 85%
OSC2: Digital > Sync Saw, Pos 92%, Transpose +19
Sub: 0%
Filter: Phaser, Cut 3500Hz, Res 55%
F.Env: Amt +62%, A 180, D 420, S 35%, R 680
LFO1 → Res: 1/4, 68%
LFO2 → Pos2: 1/3, 75%
SciFi効果音

A10: Reversed Pad
OSC1: Analog > Detune Saw, Pos 32%
OSC2: Vocal > Dark, Pos 45%, Transpose -5
Sub: 15%
Filter: LP 24dB, Cut 1800Hz, Res 18%
F.Env: Amt -42%, A 0, D 0, S 0%, R 2500 (逆)
A.Env: A 0, D 0, S 0%, R 3500 (逆)
Reverseエンベロープ模倣
```

### 使い方ガイド

```
コピー手順:

1. 上記設定テキストをコピー

2. Wavetable新規起動

3. パラメータを手動設定:
   OSC1から順に
   数値完全一致

4. 試奏:
   C2ノート

5. 保存:
   右クリック > Save Preset
   プリセット名: T01等

6. 完了:
   即使用可能

略記説明:

Pos = Position
Cut = Cutoff
Res = Resonance
F.Env = Filter Envelope
A.Env = Amp Envelope
Amt = Amount
A = Attack
D = Decay
S = Sustain
R = Release

数値単位:

Pos, Amt, Res, Drive: %
Cutoff: Hz
ADSR: ms (Sustain除く)
Transpose: 半音
Fine: cent

推奨調整:

全プリセット:
Output Level -6 dB設定
ヘッドルーム確保

Cutoff:
± 200 Hz調整OK
楽曲に合わせる

Resonance:
± 10%調整OK
好みで

ADSR:
BPMに応じて微調整
```

---

## よくある質問

### Q1: OSC2は必要?

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

### Q2: Cutoffの適正値は?

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

### Q3: プリセットから始めていい?

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
