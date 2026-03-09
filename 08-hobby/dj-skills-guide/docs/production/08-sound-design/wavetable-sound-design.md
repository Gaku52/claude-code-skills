# Wavetable Sound Design（Wavetableサウンドデザイン）

Ableton Live 12のWavetableを使った実践的なサウンドデザインを完全マスター。Sub Bass、Techno Bass、Lead、Padの4種類を段階的に作成し、プロレベルの音作りスキルを習得します。

## この章で学ぶこと

- Wavetableインターフェース完全理解
- Sub Bass作成（Step-by-Step）
- Techno Bass（TB-303スタイル）作成
- Lead Synth（Unison使用）作成
- Ambient Pad（Long Attack）作成
- 各音色のエフェクトチェイン
- Macroノブ設定テクニック
- 実践的な練習方法


## 前提知識

このガイドを読む前に、以下の知識があると理解が深まります:

- 基本的なプログラミングの知識
- 関連する基礎概念の理解
- [Synthesis Basics（シンセシス基礎）](./synthesis-basics.md) の内容を理解していること

---

## なぜWavetable Sound Designが重要なのか

**音作りの効率:**

```
Wavetableなし:

状況:
複雑なシンセを使う
パラメータ多すぎ
理解できない

結果:
時間浪費（3時間）
完成しない
挫折

Wavetableあり:

状況:
シンプルなインターフェース
必要なパラメータのみ
直感的

結果:
30分で完成
クオリティ高い
楽しい

プロの選択:

調査結果:
Wavetable使用率 70%
他のシンセ 30%

理由:
効率的
高品質
汎用性
```

**Wavetableの優位性:**

```
比較:

Wavetable vs Operator:
Wavetable: 簡単、ベース・リード・Pad
Operator: 複雑、特殊音のみ

Wavetable vs Analog:
Wavetable: モダン、クリア
Analog: ビンテージ、温かい

Wavetable vs Serum:
Wavetable: Abletonネイティブ、軽い
Serum: 強力、重い

結論:
70%の音はWavetableで十分
```

---

## Wavetableインターフェース完全理解

### セクション構成

```
上部:
┌─────────────────────────────────────┐
│ OSC 1    OSC 2    SUB    UNISON   │
│ [波形選択] [デチューン] [音量]      │
└─────────────────────────────────────┘

中央:
┌─────────────────────────────────────┐
│        FILTER                       │
│   Cutoff  Resonance  Envelope      │
└─────────────────────────────────────┘

下部:
┌─────────────────────────────────────┐
│  AMP ENVELOPE    LFO    MODULATION │
│  A D S R                           │
└─────────────────────────────────────┘

右部:
┌──────────┐
│ MACROS   │
│ 1-8      │
└──────────┘
```

### Oscillatorセクション

**OSC 1（メインオシレーター）:**

```
位置: 左上

パラメータ:

1. Wavetable選択:
   - カテゴリー: Basic Shapes、FM、Additive等
   - 波形: Sine、Saw、Square、Triangle
   - Position: 波形モーフィング（0-100%）

2. Octave:
   - 範囲: -3 〜 +3
   - 用途: Sub (-1)、Standard (0)、High (+1)

3. Semitone:
   - 範囲: -12 〜 +12
   - 用途: ハーモニー、5度上等

4. Detune:
   - 範囲: -50 〜 +50 cent
   - 用途: コーラス効果

使用頻度: 100%（必ず使用）
```

**OSC 2（セカンドオシレーター）:**

```
位置: OSC 1の右

用途:
- レイヤリング（音を厚くする）
- デチューン（コーラス効果）
- オクターブ上下（ハーモニー）

パラメータ:
- OSC 1と同様
- Volume: 0-100%（OSC 1とのバランス）

推奨設定:
Main Lead:
OSC 1: Saw、0 cent
OSC 2: Square、+7 cent、Volume 60%

Thick Bass:
OSC 1: Saw、0 cent
OSC 2: Saw、+5 cent、Volume 50%

使用頻度: 70%
```

**SUB（サブオシレーター）:**

```
位置: OSC 2の右

特徴:
- Sine波のみ
- OSC 1の1オクターブ下
- 低域補強

パラメータ:
- Volume: 0-100%
- Octave: -1（固定）

推奨設定:
ベース: Volume 30-40%
リード: Volume 10-20%
パッド: Volume 0-10%

使用頻度: 60%
```

**UNISON（ユニゾン）:**

```
位置: SUBの右

定義:
複数の音を同時に鳴らして厚みを出す

パラメータ:

1. Amount（音の数）:
   - 範囲: 1-16 voices
   - 推奨: 4-6 voices

2. Detune:
   - 範囲: 0-100%
   - 効果: 広がり、コーラス感

推奨設定:
リード:
Amount: 4 voices
Detune: 15%

パッド:
Amount: 6 voices
Detune: 25%

使用頻度: 50%（リード・パッド）
```

### Filterセクション

```
位置: 中央

パラメータ:

1. Filter Type:
   - Low Pass (LP): 最も一般的（80%）
   - High Pass (HP): 低域カット
   - Band Pass (BP): 帯域のみ通す
   - Notch: 特定周波数カット

2. Cutoff:
   - 範囲: 20 Hz - 20000 Hz
   - 最重要パラメータ

3. Resonance:
   - 範囲: 0-100%
   - Cutoff周波数を強調

4. Envelope:
   - Amount: -64 〜 +64
   - Filter Cutoffの時間的変化

5. Drive:
   - 範囲: 0-100%
   - 倍音追加（歪み）

推奨設定:
Sub Bass:
Type: LP 24dB
Cutoff: 150 Hz
Resonance: 0%

Techno Bass:
Type: LP 24dB
Cutoff: 500 Hz
Resonance: 60%

リード:
Type: LP 12dB
Cutoff: 2500 Hz
Resonance: 20%
```

### Envelopeセクション

```
位置: 下部左

種類:

1. Amp Envelope（音量）:
   - 全ての音に必須
   - A D S R

2. Filter Envelope（音色）:
   - Filter Cutoffに適用
   - Amount で深さ調整

パラメータ:
A (Attack): 0-5000 ms
D (Decay): 0-5000 ms
S (Sustain): 0-100%
R (Release): 0-10000 ms

用途別設定:
パーカッシブ（Pluck）:
A: 0, D: 300, S: 0%, R: 100

持続音（Pad）:
A: 1000, D: 1500, S: 70%, R: 3000

ベース:
A: 0, D: 100, S: 100%, R: 50
```

### LFOセクション

```
位置: 下部中央

用途:
周期的な変化（ビブラート、ワウワウ等）

パラメータ:

1. Destination:
   - Pitch、Filter、Volume、Pan等

2. Waveform:
   - Sine、Triangle、Square、Saw、Random

3. Rate:
   - Hz: 0.01-40 Hz
   - Sync: 1/4、1/8、1/16等

4. Depth:
   - 0-100%

推奨設定:
ビブラート:
Destination: Pitch
Waveform: Sine
Rate: 5 Hz
Depth: 5%

ワウワウ:
Destination: Filter Cutoff
Waveform: Triangle
Rate: 1/4
Depth: 40%
```

---

## 実践1: Sub Bass作成

**目標:** クラブで鳴る深いSub Bassを作る

### Step 1: プロジェクト準備（2分）

```
手順:

1. Ableton Live 12を起動
2. 新規プロジェクト作成
3. BPM: 128（Techno/House）
4. MIDIトラック追加
5. Wavetableを追加
```

### Step 2: Oscillator設定（5分）

```
OSC 1:
1. Wavetableカテゴリー: Basic Shapes
2. 波形: Sine
3. Position: 0%
4. Octave: 0
5. Semitone: 0
6. Detune: 0 cent

理由:
Sine波は倍音なし → 純粋な低音
Detuneなし → 安定した音

OSC 2:
- Off（使用しない）

SUB:
- Volume: 0%（Sine波なのでSUB不要）

UNISON:
- Amount: 1 voice（Unison不要）
```

### Step 3: Filter設定（5分）

```
Filter:
1. Type: Low Pass 24 dB
2. Cutoff: 200 Hz
3. Resonance: 0%
4. Envelope Amount: 0
5. Drive: 0%

理由:
LP 200 Hz → 200 Hz以上を完全カット
Resonance 0% → クリーンな低音
24 dB → 急峻なカット

確認:
スペクトラムで200 Hz以上がほぼない
```

### Step 4: Amp Envelope設定（5分）

```
Amp Envelope:
A: 0 ms
D: 50 ms
S: 100%
R: 10 ms

理由:
A 0 ms → 即座に鳴る（重要）
D 50 ms → わずかに減衰
S 100% → 鍵盤を押している間は最大音量
R 10 ms → 短い余韻（リズム明瞭）

テスト:
C1、F1、G1を演奏
タイトなベース音を確認
```

### Step 5: エフェクト追加（10分）

```
エフェクトチェイン:

1. EQ Eight:
   - High Pass: 30 Hz（不要な超低域カット）
   - Gain: 0 dB

2. Saturator:
   - Drive: 3 dB
   - Curve: Warm
   - Dry/Wet: 100%

理由:
30 Hz以下 → スピーカーで再生不可、カット
Saturator → 倍音追加、太い音

確認:
クラブ再生を想定
50-100 Hzが力強い
```

### Step 6: Macro設定（5分）

```
Audio Effect Rackで囲む:
1. 全てのデバイスを選択
2. 右クリック → Group
3. Audio Effect Rackに変換

Macro設定:

Macro 1 - Cutoff:
- Map To: Filter Cutoff
- Min: 100 Hz
- Max: 300 Hz
- Name: "Cutoff"

Macro 2 - Drive:
- Map To: Saturator Drive
- Min: 0 dB
- Max: 8 dB
- Name: "Drive"

用途:
ライブ演奏時にCutoffを動かす
Driveで音圧調整
```

### Step 7: テストと調整（8分）

```
テストMIDI:

パターン1（8小節）:
Bar 1: C1 ──── ──── ──── ────
Bar 2: F1 ──── ──── ──── ────
Bar 3: C1 ──── G1 ── ──── ────
Bar 4: F1 ──── ──── ──── ────

ベロシティ: 100-127（強く）
Length: 1拍（タイト）

確認ポイント:
□ 50-100 Hzが強い
□ 200 Hz以上がほぼない
□ タイトなアタック
□ クラブで腹に響く感じ

調整:
Cutoffが低すぎる → 上げる（150-250 Hz）
音が弱い → Saturator Drive増加
アタックが遅い → A 0 ms確認
```

### Step 8: 保存（2分）

```
手順:

1. プリセット保存:
   - Wavetable → Save Preset
   - Name: "Sub Bass Clean"
   - Category: Bass

2. Audio Effect Rack保存:
   - Rack → Save Preset
   - Name: "Sub Bass FX"

3. プロジェクト保存:
   - Cmd+S (Mac) / Ctrl+S (Win)
   - Name: "Sound Design - Sub Bass"

バックアップ:
File → Collect All and Save
→ 外部サンプル等を全て収集
```

### 完成基準

```
□ Sine波のみ
□ Filter LP 200 Hz
□ Attack 0 ms
□ 50-100 Hzが最強
□ 200 Hz以上ほぼなし
□ タイトなグルーヴ
□ クラブで鳴る音
□ Macro設定完了

所要時間: 40分
```

---

## 実践2: Techno Bass（TB-303スタイル）作成

**目標:** TB-303風のアシッドベースを作る

### Step 1: 初期設定（3分）

```
新規MIDIトラック:
1. Wavetable追加
2. BPM: 130-140（Techno）
3. Key: Am（Aマイナー）

準備:
Reference曲を聴く
→ TB-303の音を確認
```

### Step 2: Oscillator設定（5分）

```
OSC 1:
Wavetable: Basic Shapes > Saw
Position: 0%
Octave: 0
Semitone: 0
Detune: 0 cent

理由:
Saw波 → 倍音豊富、明るい音
TB-303はSaw波がメイン

OSC 2:
Off

SUB:
Volume: 0%

UNISON:
Amount: 1 voice
```

### Step 3: Filter設定（10分）

```
Filter:
Type: Low Pass 24 dB
Cutoff: 500 Hz（開始点、後で動かす）
Resonance: 70%
Envelope Amount: +45
Drive: 8%

Filter Envelope:
A: 0 ms
D: 180 ms
S: 15%
R: 40 ms

動作:
鍵盤を押す → Cutoff開く（明るい）
180 ms後 → Cutoff閉じる（暗い）
→ TB-303の特徴的な音色変化

Resonance 70%:
→ 「ピコピコ」音
→ TB-303の象徴的サウンド
```

### Step 4: Amp Envelope設定（5分）

```
Amp Envelope:
A: 0 ms
D: 80 ms
S: 85%
R: 15 ms

理由:
A 0 ms → パーカッシブなアタック
D 80 ms → わずかに減衰
S 85% → 持続音（Accentで変化）
R 15 ms → 短い余韻
```

### Step 5: エフェクトチェイン（15分）

```
エフェクト順序:

1. EQ Eight:
   High Pass: 35 Hz
   Low Shelf: 80 Hz、+2 dB
   理由: 低域補強

2. Saturator:
   Drive: 6 dB
   Curve: A Bit Warmer
   Dry/Wet: 100%
   理由: TB-303の温かみ

3. Auto Filter（オプション）:
   Filter Type: LP
   Frequency: Cutoffとリンク
   Resonance: +10%
   理由: さらなるFilter効果

4. Delay:
   Time: 1/8 Dotted
   Feedback: 25%
   Dry/Wet: 12%
   理由: Technoの空間感

5. Utility:
   Width: 0%（Mono）
   理由: ベースはMono推奨
```

### Step 6: Macro設定（10分）

```
重要なMacro:

Macro 1 - Cutoff:
Map: Filter Cutoff
Min: 200 Hz
Max: 2500 Hz
Name: "Cutoff"
→ 最重要、ライブで動かす

Macro 2 - Resonance:
Map: Filter Resonance
Min: 40%
Max: 85%
Name: "Reso"
→ ピコピコ度調整

Macro 3 - Env Amount:
Map: Filter Envelope Amount
Min: 0
Max: 60
Name: "Env"
→ 音色変化の深さ

Macro 4 - Delay Mix:
Map: Delay Dry/Wet
Min: 0%
Max: 30%
Name: "Delay"
→ 空間感調整

Macro 5 - Drive:
Map: Saturator Drive
Min: 0 dB
Max: 10 dB
Name: "Drive"
```

### Step 7: MIDIパターン作成（15分）

```
TB-303風パターン（16小節）:

Bar 1-2:
A2 ──── ──── C3 ── E3 ──
Velocity: 100-110

Bar 3-4:
A2 ── C3 ── E3 ── G3 ──
Velocity: 80-100

Bar 5-6:
A2 ──── D3 ── F3 ── A3 ──
Velocity: 90-110

Bar 7-8:
A2 ── C3 ── E3 ── C3 ──
Velocity: 100-127（Accent）

ポイント:
- 16分音符パターン
- ベロシティ変化（Accent）
- 高ベロシティ = Filter開く
- オクターブ跳躍

Filter Cutoff オートメーション:
Bar 1: 400 Hz
Bar 3: 800 Hz
Bar 5: 1200 Hz
Bar 7: 1800 Hz（ピーク）
→ 徐々にFilterを開く
```

### Step 8: 調整と完成（12分）

```
調整ポイント:

1. Resonanceバランス:
   - 70%: 標準的なTB-303
   - 60%: 控えめ
   - 80%: 強烈（自己発振注意）

2. Envelope Amount:
   - +30: 控えめな変化
   - +45: 標準的
   - +60: 強い変化

3. Saturator Drive:
   - 4 dB: クリーン
   - 6 dB: 標準的
   - 8 dB: 歪み強め

4. Delay調整:
   - 1/8: シンプル
   - 1/8 Dotted: 複雑
   - 1/16: 細かい

最終確認:
□ Filter動きがある
□ Resonanceで「ピコピコ」
□ Accent（高Velocity）でFilter開く
□ グルーヴが気持ち良い
```

### 完成基準

```
□ Saw波使用
□ Filter LP、Resonance 60-80%
□ Filter Envelope設定
□ Cutoffオートメーション
□ Macro 5個設定
□ 16分音符パターン
□ TB-303風サウンド
□ Technoで使える

所要時間: 60分
```

---

## 実践3: Lead Synth（Unison使用）作成

**目標:** メインメロディを引き立てる太いリードを作る

### Step 1: 初期設定（3分）

```
新規MIDIトラック:
Wavetable追加
BPM: 128
Key: Cm（Cマイナー）

コンセプト:
太い音
広がりがある
メロディが明瞭
```

### Step 2: Oscillator設定（10分）

```
OSC 1:
Wavetable: Modern Shapes > Formant Square
Position: 30%
Octave: 0
Detune: 0 cent
Volume: 100%

理由:
Formant Square → 独特の倍音、目立つ

OSC 2:
Wavetable: Basic Shapes > Saw
Position: 0%
Octave: 0
Detune: +8 cent
Volume: 65%

理由:
+8 cent → コーラス効果
Saw → 明るさ追加
Volume 65% → OSC 1を邪魔しない

SUB:
Volume: 15%
→ 低域補強

UNISON:
Amount: 5 voices
Detune: 18%

理由:
5 voices → 厚み
18% → 広がり、コーラス感
```

### Step 3: Filter設定（8分）

```
Filter:
Type: Low Pass 12 dB（24 dBより柔らかい）
Cutoff: 2200 Hz
Resonance: 22%
Envelope Amount: +18
Drive: 5%

Filter Envelope:
A: 40 ms
D: 450 ms
S: 55%
R: 180 ms

動作:
鍵盤を押す → Cutoff開く
450 ms後 → 中間レベルに
→ メロディの抑揚
```

### Step 4: Amp Envelope設定（5分）

```
Amp Envelope:
A: 12 ms
D: 220 ms
S: 75%
R: 320 ms

理由:
A 12 ms → わずかに柔らかいアタック
D 220 ms → 自然な減衰
S 75% → 持続音
R 320 ms → 程よい余韻
```

### Step 5: LFO設定（8分）

```
LFO 1（ビブラート）:
Destination: OSC 1 Pitch + OSC 2 Pitch
Waveform: Sine
Rate: 5.5 Hz
Depth: 4%
Retrigger: Off

理由:
5.5 Hz → 自然なビブラート
4% → 控えめ、耳障りでない

LFO 2（Filter動き）:
Destination: Filter Cutoff
Waveform: Triangle
Rate: 1/4（テンポ同期）
Depth: 12%
Retrigger: On

理由:
1/4 → テンポに合った周期的変化
12% → 微妙な音色変化
```

### Step 6: エフェクトチェイン（15分）

```
エフェクト順序:

1. EQ Eight:
   High Pass: 180 Hz
   理由: ベースと住み分け

2. Chorus:
   Rate: 0.48 Hz
   Amount: 32%
   Dry/Wet: 35%
   理由: さらなる広がり

3. Reverb:
   Type: Plate
   Decay: 1.8s
   Dry/Wet: 22%
   理由: 空間感、高級感

4. Delay:
   Time: 1/4
   Feedback: 18%
   Dry/Wet: 15%
   Filter: LP 3000 Hz
   理由: メロディの反復

5. EQ Eight（2個目）:
   Peak: 2500 Hz、+2 dB、Q 1.5
   理由: 存在感強調

6. Utility:
   Width: 120%
   Gain: -2 dB
   理由: ステレオ拡張、音量調整
```

### Step 7: Macro設定（10分）

```
Macro 1 - Cutoff:
Map: Filter Cutoff
Min: 800 Hz
Max: 4500 Hz
Name: "Bright"

Macro 2 - Reverb:
Map: Reverb Dry/Wet
Min: 5%
Max: 45%
Name: "Space"

Macro 3 - Unison:
Map: Unison Detune
Min: 0%
Max: 35%
Name: "Width"

Macro 4 - Vibrato:
Map: LFO 1 Depth
Min: 0%
Max: 10%
Name: "Vibrato"

Macro 5 - Delay:
Map: Delay Dry/Wet
Min: 0%
Max: 35%
Name: "Echo"

Macro 6 - Drive:
Map: Filter Drive
Min: 0%
Max: 15%
Name: "Grit"
```

### Step 8: テストメロディ（12分）

```
メロディパターン（8小節）:

Bar 1-2:
C4 ──── Eb4 ─ G4 ── Bb4 ─
Length: 1/2、1/4、1/4、1/4

Bar 3-4:
C5 ──── G4 ── Eb4 ─ C4 ───
Length: 1/2、1/4、1/4、1拍

Bar 5-6:
F4 ──── Ab4 ─ C5 ── Eb5 ─
Length: 1/2、1/4、1/4、1/4

Bar 7-8:
C5 ──── Bb4 ─ G4 ── C4 ───
Length: 1/2、1/4、1/4、1拍

ベロシティ: 90-115（強弱をつける）

確認:
□ メロディが明瞭
□ 広がりがある
□ ビブラートが自然
□ 存在感がある
```

### 完成基準

```
□ 2 OSC使用、Detune設定
□ Unison 4-6 voices
□ Filter Envelope設定
□ LFO 2個（Pitch、Filter）
□ Chorus + Reverb + Delay
□ Macro 6個設定
□ 広がり80%以上
□ メロディが引き立つ

所要時間: 70分
```

---

## 実践4: Ambient Pad（Long Attack）作成

**目標:** 空間を埋める柔らかいパッドを作る

### Step 1: 初期設定（3分）

```
新規MIDIトラック:
Wavetable追加
BPM: 120
Key: Dm（Dマイナー）

コンセプト:
ゆっくり立ち上がる
柔らかい
広がりがある
空間を埋める
```

### Step 2: Oscillator設定（12分）

```
OSC 1:
Wavetable: FM > Harmonic Flight
Position: 45%
Octave: 0
Detune: 0 cent
Volume: 100%

理由:
FM系 → 複雑な倍音、豊かな音色
Position 45% → 甘い音

OSC 2:
Wavetable: Additive Resonant > Formant Vowels
Position: 28%
Octave: +1（1オクターブ上）
Detune: +15 cent
Volume: 58%

理由:
+1 octave → ハーモニー
+15 cent → 広がり
Formant Vowels → 人間的な温かみ

SUB:
Volume: 0%
→ パッドにSUBは不要

UNISON:
Amount: 7 voices
Detune: 28%

理由:
7 voices → 非常に厚い
28% → 大きな広がり
```

### Step 3: Filter設定（8分）

```
Filter:
Type: Low Pass 12 dB
Cutoff: 1400 Hz
Resonance: 8%
Envelope Amount: 0
Drive: 0%

理由:
12 dB → 柔らかいカット
1400 Hz → 高域を削って柔らかく
Resonance 8% → わずかな強調
Envelope 0 → Filterは静的（LFOで動かす）
```

### Step 4: Amp Envelope設定（8分）

```
Amp Envelope:
A: 1200 ms（1.2秒）
D: 1800 ms
S: 75%
R: 3500 ms（3.5秒）

理由:
A 1200 ms → ゆっくり立ち上がる（重要）
D 1800 ms → 長い減衰
S 75% → ゆっくり減衰
R 3500 ms → 非常に長い余韻

効果:
鍵盤を押す → 1.2秒かけて音量最大
→ Ambient特有の柔らかさ
```

### Step 5: LFO設定（10分）

```
LFO 1（Filter変化）:
Destination: Filter Cutoff
Waveform: Sine
Rate: 0.25 Hz（ゆっくり）
Depth: 18%
Retrigger: Off

理由:
0.25 Hz → 4秒/周期、ゆっくり
18% → 微妙な音色変化

LFO 2（Pan変化）:
Destination: Pan
Waveform: Triangle
Rate: 0.15 Hz
Depth: 45%
Retrigger: Off

理由:
0.15 Hz → 6.6秒/周期
45% → 音が左右にゆっくり動く

LFO 3（Volume変化）:
Destination: Amp Volume
Waveform: Sine
Rate: 0.12 Hz
Depth: 8%
Retrigger: Off

理由:
0.12 Hz → 8.3秒/周期
8% → 微妙な音量変化（呼吸感）
```

### Step 6: エフェクトチェイン（20分）

```
エフェクト順序:

1. EQ Eight:
   High Pass: 280 Hz
   Low Pass: 9000 Hz
   理由: 中域のみ、ベース・高域と住み分け

2. Chorus:
   Rate: 0.28 Hz
   Amount: 48%
   Dry/Wet: 45%
   理由: 大きな広がり

3. Phaser（オプション）:
   Rate: 0.08 Hz
   Amount: 25%
   Dry/Wet: 18%
   理由: ゆっくりした位相変化

4. Reverb:
   Type: Large Hall
   Decay: 5.5s（長い）
   Dry/Wet: 48%
   Damping: 3500 Hz
   理由: 大きな空間感

5. Delay:
   Time: 1/4 Dotted
   Feedback: 38%
   Dry/Wet: 22%
   Filter: LP 4500 Hz
   理由: 複雑な反復

6. Auto Pan:
   Rate: 1/2
   Amount: 25%
   Phase: 180°
   理由: ステレオ効果

7. EQ Eight（2個目）:
   Peak: 1200 Hz、+1.5 dB、Q 0.8
   Peak: 4500 Hz、+1 dB、Q 1.2
   理由: 存在感

8. Utility:
   Width: 150%
   Gain: -3 dB
   理由: 大きなステレオ拡張
```

### Step 7: Macro設定（10分）

```
Macro 1 - Brightness:
Map: Filter Cutoff
Min: 600 Hz
Max: 3500 Hz
Name: "Bright"

Macro 2 - Space:
Map: Reverb Dry/Wet
Min: 15%
Max: 65%
Name: "Space"

Macro 3 - Width:
Map: Utility Width
Min: 80%
Max: 180%
Name: "Width"

Macro 4 - Movement:
Map: LFO 1 Rate
Min: 0.05 Hz
Max: 1 Hz
Name: "Move"

Macro 5 - Attack:
Map: Amp Envelope Attack
Min: 100 ms
Max: 3000 ms
Name: "Attack"

Macro 6 - Release:
Map: Amp Envelope Release
Min: 500 ms
Max: 8000 ms
Name: "Release"

Macro 7 - Delay:
Map: Delay Dry/Wet
Min: 0%
Max: 40%
Name: "Echo"

Macro 8 - Chorus:
Map: Chorus Dry/Wet
Min: 10%
Max: 60%
Name: "Chorus"
```

### Step 8: コード演奏（12分）

```
コードパターン（16小節）:

Bar 1-4:
Dm（D3、F3、A3）
Length: 4拍 × 4小節
Velocity: 80-90

Bar 5-8:
Bb（Bb2、D3、F3）
Length: 4拍 × 4小節
Velocity: 85-95

Bar 9-12:
F（F3、A3、C4）
Length: 4拍 × 4小節
Velocity: 75-85

Bar 13-16:
C（C3、E3、G3）
Length: 4拍 × 4小節
Velocity: 80-90

演奏方法:
同時に3音を押す
鍵盤を押したまま4拍キープ
次のコードに移る前に0.5秒重ねる

確認:
□ 1.2秒かけて立ち上がる
□ 柔らかい音色
□ 広がり100%以上
□ 空間を埋める
□ 音が左右に動く
```

### 完成基準

```
□ 2 OSC使用、異なるWavetable
□ Unison 6-8 voices、Detune 25%以上
□ Attack 1000 ms以上
□ Release 3000 ms以上
□ LFO 3個（Filter、Pan、Volume）
□ Reverb Decay 4s以上
□ Stereo Width 130%以上
□ Macro 8個設定
□ ゆっくり立ち上がる
□ 空間を埋める

所要時間: 80分
```

---

## サウンド比較表

```
比較項目:

                Sub Bass  Techno Bass  Lead       Pad
────────────────────────────────────────────────────
OSC 1波形:      Sine      Saw          Square+Saw FM
OSC 2:          Off       Off          Saw        Vowels
SUB:            0%        0%           15%        0%
Unison:         1         1            5          7
Filter Cutoff:  200 Hz    500 Hz       2200 Hz    1400 Hz
Resonance:      0%        70%          22%        8%
Attack:         0 ms      0 ms         12 ms      1200 ms
Release:        10 ms     15 ms        320 ms     3500 ms
LFO:            0         0            2          3
Reverb:         0%        0%           22%        48%
Width:          0%        0%           120%       150%
────────────────────────────────────────────────────
用途:           低域      リズム       メロディ   空間
使用頻度:       90%       70%          80%        60%
難易度:         ★         ★★           ★★★       ★★★★
```

---

## Macroノブ設定テクニック

### 基本的なMapping

```
推奨Macro配置:

Macro 1: Filter Cutoff
→ 最も重要、必ず1番に

Macro 2: Resonance
→ 音色変化、2番目に重要

Macro 3: Reverb/Delay Mix
→ 空間感調整

Macro 4: Drive/Saturation
→ 音圧調整

Macro 5-8: ジャンル・音色による

理由:
DJ MixerのEQと同じ配置
→ 筋肉記憶
```

### 範囲設定のコツ

```
良い範囲設定:

Filter Cutoff:
Min: 使用する最低値
Max: 使用する最高値
例: Sub Bass → 100-300 Hz

悪い範囲設定:
Min: 20 Hz
Max: 20000 Hz
→ 範囲が広すぎて調整困難

推奨:
実際に使う範囲の±20%
```

### 複数パラメータMapping

```
1つのMacroに複数Mapping:

例: "Brightness" Macro
- Filter Cutoff: 500-3000 Hz
- Resonance: 10-30%
- Drive: 0-8%

効果:
1つのノブで複合的な変化
→ 演奏性向上
```

---

## よくある質問（FAQ）

**Q1: WavetableとSerumの違いは？**

```
A: Serumの方が強力だが、Wavetableで十分

Wavetable:
メリット:
- Abletonネイティブ
- CPU軽い
- シンプル
- 初心者に優しい

デメリット:
- カスタムWavetableインポート制限
- エフェクト少ない

Serum:
メリット:
- カスタムWavetable自由
- エフェクト豊富
- 高度な機能

デメリット:
- CPU重い
- 有料（$189）
- 複雑

推奨:
最初の1年はWavetable
1年後、必要ならSerum購入
```

**Q2: Unisonをどのくらい使うべきですか？**

```
A: 音色と用途で変える

ベース:
Unison 1 voice
理由: Monoが基本、タイト

リード:
Unison 4-6 voices
Detune 10-20%
理由: 広がり、存在感

パッド:
Unison 6-8 voices
Detune 20-35%
理由: 大きな広がり

注意:
Unison増加 = CPU負荷増加
6 voices以上は必要な時のみ
```

**Q3: Filter Cutoffをどう動かすべきですか？**

```
A: 3つの方法

1. Filter Envelope:
自動的に時間変化
→ Pluck音、Acid Bass

2. LFO:
周期的変化
→ Wah-Wah効果

3. オートメーション:
手動で描く
→ 曲の展開に合わせる

推奨:
Acid Bass: Envelope + オートメーション
リード: 固定 or LFO
パッド: LFO
```

**Q4: ReverbとDelayのバランスは？**

```
A: 音色で変える

ベース:
Reverb: 0%
Delay: 0-15%
理由: 低域は空間少なく

リード:
Reverb: 15-30%
Delay: 10-20%
理由: 存在感と空間

パッド:
Reverb: 40-60%
Delay: 15-25%
理由: 大きな空間

合計:
Reverb + Delay < 70%
→ 濁らない
```

**Q5: プリセットを改変する場合のコツは？**

```
A: 50%以上変更する

手順:
1. 近いプリセット選択
2. OSC 1波形変更
3. Filter Cutoff調整
4. Envelope調整
5. エフェクト追加/削除
6. Macro再設定
7. 自分のプリセットとして保存

変更箇所:
最低5箇所以上
→ オリジナリティ

時間:
15-30分/プリセット
→ ゼロから作るより効率的
```

**Q6: CPU負荷を減らすには？**

```
A: 3つの方法

1. Unison削減:
8 voices → 4 voices
CPU 50%削減

2. エフェクト削減:
必須のみ残す
Reverb、Chorusは重い

3. Freeze:
トラックを右クリック → Freeze Track
→ オーディオ化、CPU削減

4. Resample:
新規オーディオトラック
録音
→ 完全にオーディオ化

推奨:
制作中: Freeze
完成後: Resample
```

---

## 練習方法

### Week 1: Sub Bass完全マスター

```
Day 1-2: 基本Sub Bass作成
1. 実践1を完全再現
2. C1、F1、G1で演奏
3. スペクトラム確認

Day 3-4: バリエーション
1. Cutoff変更（150 Hz、180 Hz、220 Hz）
2. 3種類作成
3. 違いを聴き比べ

Day 5-6: ジャンル別
1. Techno用: Cutoff 180 Hz
2. House用: Cutoff 200 Hz
3. Dubstep用: Cutoff 150 Hz

Day 7: 実戦投入
1. 自分の楽曲に使用
2. Kickとの相性確認
3. 保存してライブラリ化

目標:
Sub Bassを10分で作れる
```

### Week 2: Techno Bass完全マスター

```
Day 1-2: 基本Techno Bass作成
1. 実践2を完全再現
2. Resonance調整（60%、70%、80%）
3. Filter Envelope Amount調整

Day 3-4: MIDIパターン練習
1. 16分音符パターン5種類作成
2. Accentパターン
3. オクターブ跳躍

Day 5-6: Cutoffオートメーション
1. 8小節でCutoff 400 → 1800 Hz
2. 16小節で複雑な動き
3. Macroノブで演奏

Day 7: 実戦投入
1. Technoトラック作成
2. ドラムと合わせる
3. 完成させる

目標:
TB-303風サウンドを30分で作れる
```

### Week 3: Lead Synth完全マスター

```
Day 1-2: 基本Lead作成
1. 実践3を完全再現
2. Unison調整（3、5、7 voices）
3. Detune調整

Day 3-4: LFO練習
1. ビブラート調整（3%、5%、8%）
2. Filter LFO追加
3. Pan LFO追加

Day 5-6: エフェクトチェイン
1. Chorus、Reverb、Delay調整
2. 3種類のエフェクトチェイン作成
3. A/B比較

Day 7: メロディ作成
1. Lead用メロディ5個作成
2. 異なるジャンル
3. 保存

目標:
太いLeadを45分で作れる
```

### Week 4: Ambient Pad完全マスター

```
Day 1-2: 基本Pad作成
1. 実践4を完全再現
2. Attack調整（800 ms、1200 ms、2000 ms）
3. Release調整

Day 3-4: LFO練習
1. Filter LFO（ゆっくり）
2. Pan LFO
3. Volume LFO（呼吸感）

Day 5-6: Reverb深掘り
1. Decay調整（3s、5s、8s）
2. Damping調整
3. Dry/Wet調整

Day 7: コード進行
1. 3種類のコード進行でPad作成
2. Ambient楽曲制作
3. 完成

目標:
Ambient Padを60分で作れる
```


---

## 設計判断ガイド

### 選択基準マトリクス

技術選択を行う際の判断基準を以下にまとめます。

| 判断基準 | 重視する場合 | 妥協できる場合 |
|---------|------------|-------------|
| パフォーマンス | リアルタイム処理、大規模データ | 管理画面、バッチ処理 |
| 保守性 | 長期運用、チーム開発 | プロトタイプ、短期プロジェクト |
| スケーラビリティ | 成長が見込まれるサービス | 社内ツール、固定ユーザー |
| セキュリティ | 個人情報、金融データ | 公開データ、社内利用 |
| 開発速度 | MVP、市場投入スピード | 品質重視、ミッションクリティカル |

### アーキテクチャパターンの選択

```
┌─────────────────────────────────────────────────┐
│              アーキテクチャ選択フロー              │
├─────────────────────────────────────────────────┤
│                                                 │
│  ① チーム規模は？                                │
│    ├─ 小規模（1-5人）→ モノリス                   │
│    └─ 大規模（10人+）→ ②へ                       │
│                                                 │
│  ② デプロイ頻度は？                               │
│    ├─ 週1回以下 → モノリス + モジュール分割         │
│    └─ 毎日/複数回 → ③へ                          │
│                                                 │
│  ③ チーム間の独立性は？                            │
│    ├─ 高い → マイクロサービス                      │
│    └─ 中程度 → モジュラーモノリス                   │
│                                                 │
└─────────────────────────────────────────────────┘
```

### トレードオフの分析

技術的な判断には必ずトレードオフが伴います。以下の観点で分析を行いましょう:

**1. 短期 vs 長期のコスト**
- 短期的に速い方法が長期的には技術的負債になることがある
- 逆に、過剰な設計は短期的なコストが高く、プロジェクトの遅延を招く

**2. 一貫性 vs 柔軟性**
- 統一された技術スタックは学習コストが低い
- 多様な技術の採用は適材適所が可能だが、運用コストが増加

**3. 抽象化のレベル**
- 高い抽象化は再利用性が高いが、デバッグが困難になる場合がある
- 低い抽象化は直感的だが、コードの重複が発生しやすい

```python
# 設計判断の記録テンプレート
class ArchitectureDecisionRecord:
    """ADR (Architecture Decision Record) の作成"""

    def __init__(self, title: str):
        self.title = title
        self.context = ""
        self.decision = ""
        self.consequences = []
        self.alternatives = []

    def set_context(self, context: str):
        """背景と課題の記述"""
        self.context = context
        return self

    def set_decision(self, decision: str):
        """決定内容の記述"""
        self.decision = decision
        return self

    def add_consequence(self, consequence: str, positive: bool = True):
        """結果の追加"""
        self.consequences.append({
            'description': consequence,
            'type': 'positive' if positive else 'negative'
        })
        return self

    def add_alternative(self, name: str, reason_rejected: str):
        """却下した代替案の追加"""
        self.alternatives.append({
            'name': name,
            'reason_rejected': reason_rejected
        })
        return self

    def to_markdown(self) -> str:
        """Markdown形式で出力"""
        md = f"# ADR: {self.title}\n\n"
        md += f"## 背景\n{self.context}\n\n"
        md += f"## 決定\n{self.decision}\n\n"
        md += "## 結果\n"
        for c in self.consequences:
            icon = "✅" if c['type'] == 'positive' else "⚠️"
            md += f"- {icon} {c['description']}\n"
        md += "\n## 却下した代替案\n"
        for a in self.alternatives:
            md += f"- **{a['name']}**: {a['reason_rejected']}\n"
        return md
```

---

## 実務での適用シナリオ

### シナリオ1: スタートアップでのMVP開発

**状況:** 限られたリソースで素早くプロダクトをリリースする必要がある

**アプローチ:**
- シンプルなアーキテクチャを選択
- 必要最小限の機能に集中
- 自動テストはクリティカルパスのみ
- モニタリングは早期から導入

**学んだ教訓:**
- 完璧を求めすぎない（YAGNI原則）
- ユーザーフィードバックを早期に取得
- 技術的負債は意識的に管理する

### シナリオ2: レガシーシステムのモダナイゼーション

**状況:** 10年以上運用されているシステムを段階的に刷新する

**アプローチ:**
- Strangler Fig パターンで段階的に移行
- 既存のテストがない場合はCharacterization Testを先に作成
- APIゲートウェイで新旧システムを共存
- データ移行は段階的に実施

| フェーズ | 作業内容 | 期間目安 | リスク |
|---------|---------|---------|--------|
| 1. 調査 | 現状分析、依存関係の把握 | 2-4週間 | 低 |
| 2. 基盤 | CI/CD構築、テスト環境 | 4-6週間 | 低 |
| 3. 移行開始 | 周辺機能から順次移行 | 3-6ヶ月 | 中 |
| 4. コア移行 | 中核機能の移行 | 6-12ヶ月 | 高 |
| 5. 完了 | 旧システム廃止 | 2-4週間 | 中 |

### シナリオ3: 大規模チームでの開発

**状況:** 50人以上のエンジニアが同一プロダクトを開発する

**アプローチ:**
- ドメイン駆動設計で境界を明確化
- チームごとにオーナーシップを設定
- 共通ライブラリはInner Source方式で管理
- APIファーストで設計し、チーム間の依存を最小化

```python
# チーム間のAPI契約定義
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

class Priority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class APIContract:
    """チーム間のAPI契約"""
    endpoint: str
    method: str
    owner_team: str
    consumers: List[str]
    sla_ms: int  # レスポンスタイムSLA
    priority: Priority

    def validate_sla(self, actual_ms: int) -> bool:
        """SLA準拠の確認"""
        return actual_ms <= self.sla_ms

    def to_openapi(self) -> dict:
        """OpenAPI形式で出力"""
        return {
            'path': self.endpoint,
            'method': self.method,
            'x-owner': self.owner_team,
            'x-consumers': self.consumers,
            'x-sla-ms': self.sla_ms
        }

# 使用例
contracts = [
    APIContract(
        endpoint="/api/v1/users",
        method="GET",
        owner_team="user-team",
        consumers=["order-team", "notification-team"],
        sla_ms=200,
        priority=Priority.HIGH
    ),
    APIContract(
        endpoint="/api/v1/orders",
        method="POST",
        owner_team="order-team",
        consumers=["payment-team", "inventory-team"],
        sla_ms=500,
        priority=Priority.CRITICAL
    )
]
```

### シナリオ4: パフォーマンスクリティカルなシステム

**状況:** ミリ秒単位のレスポンスが求められるシステム

**最適化ポイント:**
1. キャッシュ戦略（L1: インメモリ、L2: Redis、L3: CDN）
2. 非同期処理の活用
3. コネクションプーリング
4. クエリ最適化とインデックス設計

| 最適化手法 | 効果 | 実装コスト | 適用場面 |
|-----------|------|-----------|---------|
| インメモリキャッシュ | 高 | 低 | 頻繁にアクセスされるデータ |
| CDN | 高 | 低 | 静的コンテンツ |
| 非同期処理 | 中 | 中 | I/O待ちが多い処理 |
| DB最適化 | 高 | 高 | クエリが遅い場合 |
| コード最適化 | 低-中 | 高 | CPU律速の場合 |
---

## まとめ

Wavetableは、Ableton Live 12で最も重要なシンセサイザーです。

**重要ポイント:**

1. **Sub Bass**: Sine波、LP 200 Hz、Attack 0 ms
2. **Techno Bass**: Saw波、Resonance 70%、Filter Envelope
3. **Lead**: 2 OSC Detune、Unison 5 voices、LFO
4. **Pad**: Long Attack、Unison 7 voices、大きなReverb

**学習順序:**
1. Sub Bass（最も簡単、40分）
2. Techno Bass（Acidサウンド、60分）
3. Lead（太い音、70分）
4. Pad（空間系、80分）

**効率化:**
- プリセット改変50%以上
- Macro設定必須
- ライブラリ構築
- 30日チャレンジ

**次のステップ:** [FM Sound Design（FM合成）](./fm-sound-design.md) へ進む

---

## 関連ファイル

- **[Synthesis Basics](./synthesis-basics.md)** - シンセシス基礎理論
- **[FM Sound Design](./fm-sound-design.md)** - Operator活用
- **[Modulation Techniques](./modulation-techniques.md)** - LFO・Envelope深掘り
- **[03-instruments/wavetable.md](../03-instruments/wavetable.md)** - Wavetable詳細リファレンス

---

**Wavetableで、あなただけのサウンドを作りましょう！**

---

## 高度テクニック: カスタムウェーブテーブル作成

### ウェーブテーブルの仕組みを深く理解する

```
ウェーブテーブルの基本構造:

┌─────────────────────────────────────────────┐
│  Frame 1   Frame 2   Frame 3  ...  Frame N  │
│  ┌─────┐  ┌─────┐  ┌─────┐      ┌─────┐   │
│  │ ∿∿∿ │  │ ∿∿∿ │  │ ∿∿∿ │      │ ∿∿∿ │   │
│  └─────┘  └─────┘  └─────┘      └─────┘   │
│     ↕         ↕         ↕            ↕      │
│  Position 0%      Position 50%   Position 100% │
└─────────────────────────────────────────────┘

各フレームは1つの波形サイクル（2048サンプル）
Positionノブでフレーム間を補間（モーフィング）

1ウェーブテーブル = 複数フレームの集合体
Ableton Wavetable: 最大256フレーム
各フレーム: 2048サンプル（44.1kHz時）
```

### Ableton Live内でのカスタムウェーブテーブル

```
方法1: オーディオからウェーブテーブルを作成

手順:
1. オーディオサンプルを用意
   - ボーカル一音
   - 楽器の持続音
   - 環境音
   - 自分で録音した音

2. Simpler/Samplerで読み込み
   - Simpler → Classic Mode
   - ループポイント設定
   - 1サイクル分を選択

3. Wavetableに変換
   - オーディオをWavetableのOSCにドラッグ
   - 自動的にウェーブテーブル化
   - Position で異なるポイントを選択

4. 微調整
   - Frame数の確認
   - Positionの最適値を探す
   - LFOでPositionを動かす

注意点:
- 短いサンプル（1-5秒）が最適
- 持続音が最も良い結果
- ノイズが多いサンプルは避ける
- サンプルレート44.1kHz推奨
```

```
方法2: 倍音加算合成によるカスタム波形

Additive合成の原理:
基本波（1倍音）+ 上の倍音 = 複雑な波形

倍音構成の例:

Organ風波形:
1倍音: 100%（基本）
2倍音: 80%
3倍音: 0%
4倍音: 60%
5倍音: 0%
6倍音: 40%
8倍音: 20%
→ 偶数倍音のみ = 温かい、丸い音

Brass風波形:
1倍音: 100%
2倍音: 70%
3倍音: 50%
4倍音: 35%
5倍音: 25%
6倍音: 18%
7倍音: 12%
8倍音: 8%
→ 全倍音が徐々に減衰 = 明るい、金属的

String風波形:
1倍音: 100%
2倍音: 45%
3倍音: 60%
4倍音: 20%
5倍音: 40%
6倍音: 10%
7倍音: 25%
→ 奇数倍音がやや強い = ザラついた質感

実践:
Wavetableの「User」カテゴリーで作成可能
各フレームの倍音バランスを変える
→ Position変化で音色が大きく変わる
```

```
方法3: 外部ツールでウェーブテーブル作成

推奨ツール:

1. WaveEdit（無料）:
   - VCV Rack開発のオープンソースツール
   - 256フレーム対応
   - グラフィカルなエディタ
   - .wav書き出し

2. Serum内蔵エディタ（Serum所有時）:
   - 最も高機能
   - 数式入力対応
   - FFT編集
   - .wav書き出し → Wavetableに読み込み

3. Audacity（無料）:
   - 波形を手動で描画
   - 2048サンプル単位でカット
   - 複数フレームを連結

ワークフロー:
外部ツール → .wav書き出し → Wavetable OSCにドラッグ
フレーム数は自動検出（2048サンプル/フレーム）
```

---

## ウェーブテーブル・モーフィング技法

### Positionモーフィングの基礎

```
Positionパラメータの活用:

静的設定:
- Position固定 → 一定の音色
- 用途: ベース、安定したリード

動的設定（推奨）:
- Envelope → Position: 時間変化
- LFO → Position: 周期的変化
- オートメーション → Position: 曲展開に合わせる

Envelope → Position の設定例:

パーカッシブモーフ:
Envelope: A 0, D 200ms, S 0%, R 50ms
Amount: 0% → 80%
→ アタック時に明るく、すぐ暗くなる
→ Pluck系サウンドに最適

スウェルモーフ:
Envelope: A 2000ms, D 500ms, S 60%, R 1000ms
Amount: 0% → 100%
→ ゆっくりと音色が変化
→ Pad、Ambient系に最適
```

### LFOによるモーフィング

```
LFO → Position の設定パターン:

パターン1: ゆっくりモーフ
LFO Waveform: Sine
Rate: 0.1 Hz（10秒/周期）
Depth: 40%
→ 10秒かけてゆっくり音色が変わる
→ Ambient Pad に最適

パターン2: リズミックモーフ
LFO Waveform: Square
Rate: 1/8（テンポ同期）
Depth: 50%
→ 1/8音符ごとに音色が切り替わる
→ Techno、EDM のリードに最適

パターン3: ランダムモーフ
LFO Waveform: Sample & Hold（ランダム）
Rate: 1/16
Depth: 30%
→ ランダムに音色変化
→ Glitch、Experimental に最適

パターン4: 非対称モーフ
LFO Waveform: Saw（上昇）
Rate: 1/4
Depth: 60%
→ 徐々に明るくなり、急に暗くなる
→ Trance、Progressive に最適

複数LFO組み合わせ:
LFO 1 → OSC 1 Position（Rate: 0.2 Hz、Depth: 35%）
LFO 2 → OSC 2 Position（Rate: 0.13 Hz、Depth: 45%）
→ 2つのOSCが異なる速度でモーフ
→ 非常に複雑で有機的な音色変化
```

### クロスフェード・モーフィング

```
OSC 1 と OSC 2 のクロスフェード:

設定:
OSC 1: Wavetable A（例: Basic Saw）
OSC 2: Wavetable B（例: FM Complex）

Macro でクロスフェード:
Macro "Morph":
  OSC 1 Volume: 100% → 0%
  OSC 2 Volume: 0% → 100%

効果:
Macro 0% = 純粋なSaw波
Macro 50% = 両方ミックス
Macro 100% = 純粋なFM
→ ライブ演奏で音色を劇的に変化

応用:
LFO → Macro割り当て
→ 自動的にOSC間をクロスフェード

オートメーション:
曲のビルドアップで 0% → 100%
→ ドロップで音色が完全に変わる
```

---

## ジャンル別サウンドレシピ集

### Deep House: Warm Chord Stab

```
コンセプト:
温かい、柔らかいコードスタブ
Deep Houseの定番サウンド

OSC 1:
Wavetable: Basic Shapes > Triangle
Position: 15%
Octave: 0
Detune: 0

OSC 2:
Wavetable: Additive > Organ
Position: 20%
Octave: 0
Detune: +6 cent
Volume: 55%

SUB: Volume 20%
UNISON: 3 voices, Detune 8%

Filter:
Type: LP 12dB
Cutoff: 1800 Hz
Resonance: 15%
Envelope Amount: +25

Filter Envelope:
A: 5ms, D: 350ms, S: 40%, R: 200ms

Amp Envelope:
A: 2ms, D: 400ms, S: 65%, R: 250ms

エフェクト:
1. EQ: HP 120Hz
2. Saturator: Drive 4dB, Warm
3. Chorus: Rate 0.5Hz, Amount 25%, Wet 30%
4. Reverb: Plate, Decay 2.2s, Wet 28%
5. Utility: Width 110%

Macro設定:
1. Cutoff (800-3000Hz)
2. Resonance (5-35%)
3. Space (Reverb 10-45%)
4. Warmth (Saturator Drive 0-8dB)

コード例: Cm7 → Fm7 → Gm7 → AbMaj7
ベロシティ: 85-105（控えめに）
```

### Trance: Supersaw Lead

```
コンセプト:
太く広がるスーパーソウ
Tranceの象徴的サウンド

OSC 1:
Wavetable: Basic Shapes > Saw
Position: 0%
Octave: 0
Detune: 0

OSC 2:
Wavetable: Basic Shapes > Saw
Position: 0%
Octave: +1
Detune: +12 cent
Volume: 70%

SUB: Volume 25%
UNISON: 8 voices, Detune 22%

Filter:
Type: LP 24dB
Cutoff: 3500 Hz
Resonance: 10%
Envelope Amount: +30

Filter Envelope:
A: 10ms, D: 600ms, S: 50%, R: 400ms

Amp Envelope:
A: 5ms, D: 300ms, S: 80%, R: 500ms

エフェクト:
1. EQ: HP 200Hz, Peak 3kHz +2dB
2. Chorus: Rate 0.6Hz, Amount 35%, Wet 25%
3. Reverb: Hall, Decay 3.0s, Wet 30%
4. Delay: 1/4 Dotted, FB 20%, Wet 18%
5. Utility: Width 140%, Gain -2dB

Macro設定:
1. Cutoff (1500-6000Hz)
2. Width (Unison Detune 5-35%)
3. Space (Reverb 10-50%)
4. Echo (Delay 0-30%)
5. Brightness (EQ Peak 0-4dB)

ポイント:
- Unison 8 voicesが鍵
- OctaveレイヤーでFullnessを確保
- Sidechain Compressionでキックとの共存
- ビルドアップでCutoffオートメーション
```

### Dubstep: Growl Bass

```
コンセプト:
うなるような攻撃的ベース
Dubstepのメインサウンド

OSC 1:
Wavetable: FM > Metallic
Position: 60%（重要: ここが音色の核）
Octave: -1
Detune: 0

OSC 2:
Wavetable: Distortion > Harsh Digital
Position: 40%
Octave: 0
Detune: +3 cent
Volume: 80%

SUB: Volume 35%
UNISON: 2 voices, Detune 5%

Filter:
Type: LP 24dB
Cutoff: 800 Hz
Resonance: 55%
Envelope Amount: +50
Drive: 15%

Filter Envelope:
A: 0ms, D: 250ms, S: 25%, R: 80ms

Amp Envelope:
A: 0ms, D: 150ms, S: 90%, R: 30ms

LFO設定（Growl効果の核心）:
LFO 1 → OSC 1 Position:
Waveform: Saw
Rate: 1/8（テンポ同期）
Depth: 70%

LFO 2 → Filter Cutoff:
Waveform: Square
Rate: 1/16
Depth: 45%

エフェクト:
1. EQ: HP 30Hz
2. Overdrive: Drive 60%, Tone 45%
3. OTT (Multiband Compression): Amount 40%
4. EQ: Peak 200Hz +3dB, Peak 800Hz +2dB
5. Utility: Width 0%（Mono、低域のため）

Macro設定:
1. Growl (LFO 1 Depth 20-90%)
2. Cutoff (300-2000Hz)
3. Aggression (Overdrive 20-80%)
4. LFO Rate (1/16 - 1/4)
5. Resonance (30-75%)

ポイント:
- LFO → Positionが「うなり」の正体
- Overdriveで攻撃性を出す
- OTTで音圧を稼ぐ
- 低域はMonoで安定させる
```

### Lo-Fi Hip Hop: Mellow Keys

```
コンセプト:
暖かくてノスタルジックな鍵盤サウンド
Lo-Fi Hip Hopの定番

OSC 1:
Wavetable: Additive > Electric Piano
Position: 35%
Octave: 0
Detune: +3 cent

OSC 2:
Wavetable: Basic Shapes > Triangle
Position: 0%
Octave: +1
Detune: -5 cent
Volume: 30%

SUB: Volume 10%
UNISON: 2 voices, Detune 12%

Filter:
Type: LP 12dB
Cutoff: 2000 Hz
Resonance: 12%
Envelope Amount: +15

Filter Envelope:
A: 3ms, D: 500ms, S: 30%, R: 300ms

Amp Envelope:
A: 5ms, D: 600ms, S: 50%, R: 400ms

エフェクト:
1. EQ: HP 150Hz, LP 6000Hz（高域カットで「Lo-Fi」感）
2. Saturator: Drive 5dB, Sinoid Fold
3. Chorus: Rate 0.3Hz, Amount 20%, Wet 22%
4. Vinyl Distortion: Tracing 40%, Pinch 15%
5. Reverb: Room, Decay 1.5s, Wet 25%
6. Auto Filter: LP, Cutoff 4kHz, LFO Rate 0.08Hz, Depth 15%

Macro設定:
1. Warmth (Saturator 0-10dB)
2. Lo-Fi (LP Cutoff 3000-8000Hz)
3. Vinyl (Tracing 0-60%)
4. Space (Reverb 10-40%)

ポイント:
- LP EQでハイカットが「Lo-Fi」の鍵
- Vinyl Distortionでレコード質感
- コード進行: Dm7 → G7 → CMaj7 → Am7
- ベロシティを不揃いにしてヒューマン感を出す
- Swing 55-60%を適用
```

### Minimal Techno: Percussive Synth Hit

```
コンセプト:
短くてパーカッシブなシンセヒット
Minimal Technoのアクセント

OSC 1:
Wavetable: Modern Shapes > Resonant Comb
Position: 55%
Octave: 0
Detune: 0

OSC 2: Off

SUB: Volume 0%
UNISON: 1 voice

Filter:
Type: BP（Band Pass）12dB
Cutoff: 1200 Hz
Resonance: 45%
Envelope Amount: +55
Drive: 10%

Filter Envelope:
A: 0ms, D: 120ms, S: 0%, R: 30ms

Amp Envelope:
A: 0ms, D: 80ms, S: 0%, R: 20ms

エフェクト:
1. EQ: HP 80Hz
2. Saturator: Drive 3dB
3. Delay: 1/16, FB 35%, Wet 20%, Ping Pong
4. Reverb: Small Room, Decay 0.8s, Wet 15%

Macro設定:
1. Tone (Filter Cutoff 400-3000Hz)
2. Snap (Envelope D 40-300ms)
3. Echo (Delay Wet 0-35%)

ポイント:
- Amp Envelope のDecayが非常に短い
- Band Passフィルターで独特の鳴り
- Ping Pong Delayでステレオ空間
- ベロシティでアクセント表現
```

---

## Wavetable vs Serum: 詳細比較

### 機能比較

```
項目                    Wavetable           Serum
──────────────────────────────────────────────────
価格                    Abletonに付属       $189（単体購入）
オシレーター数          2 + Sub             2 + Sub + Noise
ウェーブテーブル数      内蔵約200種         内蔵約140種 + 無限カスタム
カスタムWT作成          限定的              非常に強力
フィルタータイプ        約10種              約50種以上
内蔵エフェクト          なし（Rackで対応）  10種（Distortion、Reverb等）
モジュレーション        3 Envelope + 3 LFO  3 Envelope + 4 LFO + Macro
ビジュアル              2D波形表示          3Dウェーブテーブル表示
CPU負荷                 軽い（★★☆☆☆）     やや重い（★★★★☆）
学習曲線                緩やか              やや急
Abletonとの統合         完全ネイティブ      VST/AU
Macro                   8（Rack経由）       4（ネイティブ）
ドラッグ&ドロップ       対応                対応
リサイズ                不可                可能
プリセット数            内蔵豊富            膨大（サードパーティ含む）
コミュニティ            中規模              非常に大規模
```

### どちらを選ぶべきか

```
Wavetableを選ぶべき場面:
✓ Ableton Live をメインDAWとして使用
✓ CPU負荷を最小限にしたい
✓ シンプルな操作で素早く音作りしたい
✓ ベース、リード、パッドの基本音色で十分
✓ プラグイン予算を節約したい
✓ シンセサイザー初心者

Serumを選ぶべき場面:
✓ 複雑なサウンドデザインが必要
✓ カスタムウェーブテーブルを頻繁に作成
✓ Dubstep、Neurofunk等の攻撃的サウンド
✓ 内蔵エフェクトで完結させたい
✓ 3Dビジュアルで波形を確認したい
✓ サードパーティプリセットを活用したい

両方使い分ける（推奨）:
- メインの作業: Wavetable（軽量、高速）
- 特殊な音色: Serum（高機能）
- ライブ演奏: Wavetable（CPU安定）
- サウンドデザイン研究: Serum（教育的）

移行パス:
1年目: Wavetableで基礎固め
2年目: Serum導入、両方併用
3年目: 用途に応じて最適なツール選択
```

---

## 実践演習: 総合課題

### 演習1: 1曲分のサウンドセットを作る

```
課題:
Techno 1トラック分のサウンドを全てWavetableで作成

必要な音色:
1. Kick Layer（キックの上レイヤー）
2. Sub Bass
3. Acid Bass（TB-303風）
4. Percussive Hit
5. Pad（ブレイクダウン用）
6. Lead（ビルドアップ用）
7. FX Rise（ライザー）
8. FX Impact（インパクト）

手順:
Step 1: テンプレート作成（15分）
- BPM 130
- Key: Am
- 8トラック用意
- 各トラックにWavetable配置

Step 2: 各音色作成（各15-30分）
- このガイドのレシピを参考に
- 各音色をプリセット保存

Step 3: アレンジ（30分）
- イントロ → ビルドアップ → ドロップ → ブレイク → ドロップ2 → アウトロ
- 各セクションで使用する音色を配置

Step 4: ミキシング（20分）
- EQで住み分け
- コンプレッサーで音圧
- サイドチェイン設定

完成目標: 4-6分のTechnoトラック
所要時間: 3-4時間
```

### 演習2: モーフィングパッド作成チャレンジ

```
課題:
16小節かけて音色が完全に変化するパッドを作成

要件:
- OSC 1 と OSC 2 で異なるウェーブテーブル使用
- Position をオートメーションで動かす
- Filter Cutoff も連動して変化
- 最初は暗くシンプル → 最後は明るく複雑

Step 1: OSC設定
OSC 1: Basic > Sine（Position 0%スタート）
OSC 2: FM > Complex（Position 0%スタート）

Step 2: オートメーション作成
Bar 1-4: OSC 1 Position 0→25%, OSC 2 Position 0→15%
Bar 5-8: OSC 1 Position 25→50%, OSC 2 Position 15→40%
Bar 9-12: OSC 1 Position 50→75%, OSC 2 Position 40→70%
Bar 13-16: OSC 1 Position 75→100%, OSC 2 Position 70→100%

Step 3: Filter オートメーション
Bar 1: Cutoff 600 Hz
Bar 8: Cutoff 1500 Hz
Bar 16: Cutoff 4000 Hz

Step 4: エフェクトオートメーション
Reverb Wet: 20% → 55%（16小節かけて）
Chorus Amount: 15% → 45%
Width: 100% → 160%

評価基準:
□ 音色変化が滑らか
□ 16小節で明確に音色が変わった
□ フィルターと連動している
□ 空間感も変化している
□ ミックスに使えるクオリティ
```

### 演習3: リファレンス曲の音色再現

```
課題:
好きな曲のシンセ音色をWavetableで再現する

手順:

Step 1: リファレンス選択（5分）
- Shazam/SoundHound等で曲を特定
- 再現したい音色を1つ選ぶ
- その音色が鳴る箇所を繰り返し聴く

Step 2: 音色分析（15分）
チェックリスト:
□ 波形の種類は?（Saw/Square/Sine/複雑）
□ 明るさは?（Filter Cutoff推定）
□ 太さは?（Unisonの有無）
□ 時間変化は?（Envelope/LFO）
□ 空間は?（Reverb/Delay量）
□ 歪みは?（Saturation/Distortion）

Step 3: 再現（30-60分）
1. 波形選択から始める
2. Filter Cutoff大まかに合わせる
3. Envelopeで時間変化を再現
4. Unison/Detuneで太さを調整
5. エフェクトで空間を合わせる

Step 4: A/B比較（10分）
- リファレンスと交互に聴く
- 差異をメモ
- 微調整

合格基準:
80%の再現度で十分
完全な再現は不要（学習が目的）

推奨リファレンス曲:
Techno: Amelie Lens - "Exhale"（Lead）
House: Disclosure - "Latch"（Bass）
Trance: Above & Beyond - "Sun & Moon"（Pad）
Lo-Fi: Nujabes - "Aruarian Dance"（Keys）
```

---

## トラブルシューティング

### 音が出ない場合

```
チェックリスト:

1. MIDIノート確認:
   □ MIDIトラックにノートがある
   □ ベロシティが0ではない
   □ オクターブが適切（C-2等は聴こえない）
   □ MIDIチャンネルが正しい

2. Wavetable設定確認:
   □ OSC 1 のVolumeが0でない
   □ Filter Cutoffが極端に低くない（20Hz等）
   □ Amp Envelope のSustainが0でない
   □ Amp Envelope のAttackが極端に長くない

3. トラック設定確認:
   □ トラックがミュートされていない
   □ トラックのVolumeが0でない
   □ 出力がMasterに繋がっている
   □ モニター設定が「Auto」

4. オーディオ設定確認:
   □ オーディオインターフェースが認識されている
   □ サンプルレートが正しい
   □ バッファサイズが適切（256-1024）
   □ 出力チャンネルが正しい

解決しない場合:
- Wavetableを削除して再追加
- 新規MIDIトラックに変更
- Abletonを再起動
- オーディオインターフェースを再接続
```

### 音が歪む/割れる場合

```
原因と対策:

原因1: 音量オーバー
対策: Utility で -3dB 〜 -6dB 下げる
確認: Masterトラックのメーターが0dBを超えていないか

原因2: Resonance過大
対策: Resonance を50%以下に下げる
注意: 70%以上で自己発振する場合がある

原因3: Unison過多
対策: Voices数を減らす（8→4）
理由: 多すぎるVoicesは音量と位相で歪む

原因4: Drive/Saturation過大
対策: Drive値を下げる
目安: Saturator Drive 6dB以下、Filter Drive 10%以下

原因5: エフェクト過多
対策: エフェクトを1つずつバイパスして原因特定
方法: デバイスのON/OFFボタンを使用

原因6: サンプルレート不一致
対策: プロジェクトとインターフェースのSRを統一
推奨: 44.1kHz または 48kHz
```

### CPU負荷が高い場合

```
対策（効果順）:

1. Unison削減（効果: 大）
   8 voices → 4 voices
   CPU削減: 約40-50%

2. バッファサイズ増加（効果: 大）
   128 → 512 samples
   レイテンシ増加するが安定

3. Freeze Track（効果: 大）
   右クリック → Freeze Track
   完成したトラックから順に

4. 不要エフェクト削除（効果: 中）
   Chorusは特にCPU負荷が高い
   Reverb: Convolutionは重い → Algorithmicに変更

5. OSC 2をOff（効果: 中）
   シンプルな音色ならOSC 1のみ

6. LFO削減（効果: 小-中）
   使用していないLFOをOff

7. Resample（効果: 完全）
   MIDI → オーディオに録音
   CPU負荷: 0%

CPU負荷の目安:
制作時: 70%以下を維持
ライブ: 50%以下を推奨
録音時: 60%以下を推奨
```

### 音が薄い/迫力がない場合

```
改善策:

1. レイヤリング:
   OSC 2を追加
   異なるWavetable + Detuneで厚み

2. Unison追加:
   Amount: 4-6 voices
   Detune: 10-25%

3. サチュレーション:
   Saturator Drive: 3-6 dB
   → 倍音が追加され存在感アップ

4. EQブースト:
   ベース: 80-120 Hz +2dB
   リード: 2-4 kHz +2dB
   パッド: 1-3 kHz +1.5dB

5. コンプレッション:
   Ratio: 3:1 - 4:1
   Attack: 10-30ms
   Release: Auto
   → ダイナミクスを整えて音圧アップ

6. ステレオ拡張:
   Utility Width: 110-140%
   Chorus追加
   → 空間的な広がりで迫力

7. Sub Oscillator:
   Volume: 15-30%
   → 低域の土台を補強

順番:
まずOSC/Unisonで基本を太くする
→ 次にSaturation/EQで味付け
→ 最後にCompression/Stereoで仕上げ
```

### Filter Envelopeが効かない場合

```
確認項目:

1. Envelope Amount が 0 になっていないか
   → Amount を +20 〜 +50 に設定

2. Filter Cutoff が高すぎないか
   → Cutoff を下げる（Envelopeで開く余地が必要）
   例: Cutoff 500Hz + Amount +40 → 500-2000Hzの範囲で動く

3. Decay が短すぎないか
   → D: 200ms以上でまず試す
   短すぎると変化が聴こえない

4. ノートの長さが短すぎないか
   → 長いノートで確認（2拍以上）

5. Filter Type が正しいか
   → LP（Low Pass）で確認
   HP/BPでは効果が異なる

デバッグ方法:
1. Amount を最大 (+64) に設定
2. Cutoff を低め (200Hz) に設定
3. D を長め (1000ms) に設定
4. 長いノートを演奏
5. 効果を確認
6. 各パラメータを徐々に調整
```

---

## Wavetableサウンドデザイン用語集

```
用語                    説明
──────────────────────────────────────────────────
Wavetable               複数の波形フレームを格納したテーブル
Frame                   ウェーブテーブル内の1つの波形サイクル
Position                ウェーブテーブル内のフレーム位置（0-100%）
Morphing                Position変化による波形間の滑らかな遷移
Oscillator              音の波形を生成する部分
Sub Oscillator          メインの1オクターブ下を生成する補助発振器
Unison                  複数のVoiceを重ねて厚みを出す機能
Detune                  微妙なピッチずれでコーラス効果を生む
Voice                   Unisonの個々の音
Filter                  特定の周波数帯域を通過/遮断する回路
Cutoff                  フィルターが効き始める周波数
Resonance               Cutoff付近を強調するパラメータ
Drive                   信号に歪みを加えて倍音を増やす
Envelope (ADSR)         Attack/Decay/Sustain/Releaseの時間変化
Attack                  音が最大音量に達するまでの時間
Decay                   最大音量からSustainレベルまでの時間
Sustain                 鍵盤を押し続けている間の音量レベル
Release                 鍵盤を離してから音が消えるまでの時間
LFO                     低周波発振器、周期的な変調に使用
Modulation              あるパラメータで別のパラメータを変化させること
Macro                   複数パラメータを1つのノブで制御する機能
Sidechain               外部信号で内部パラメータを制御する手法
Saturation              軽い歪みを加えて倍音を豊かにする処理
Additive Synthesis      倍音を足し合わせて波形を作る合成方式
FM Synthesis            周波数変調による合成方式
Wavetable Synthesis     ウェーブテーブルを使った合成方式
Resampling              MIDIをオーディオに変換すること
Freeze                  トラックを一時的にオーディオ化してCPU削減
```

---

## 30日間マスタープラン

```
Week 1（基礎固め）:
Day 1: Wavetableインターフェース全体の理解
Day 2: Sub Bass作成（実践1完了）
Day 3: Sub Bassバリエーション3種
Day 4: Techno Bass作成開始（実践2前半）
Day 5: Techno Bass完成（実践2後半）
Day 6: Techno Bassバリエーション3種
Day 7: 復習、プリセット整理

Week 2（応用展開）:
Day 8: Lead Synth作成開始（実践3前半）
Day 9: Lead Synth完成（実践3後半）
Day 10: Leadバリエーション（Trance Supersaw）
Day 11: Ambient Pad作成開始（実践4前半）
Day 12: Ambient Pad完成（実践4後半）
Day 13: Padバリエーション（Lo-Fi Keys）
Day 14: Week 1-2の全音色レビュー

Week 3（ジャンル特化）:
Day 15: Deep House Chord Stab
Day 16: Dubstep Growl Bass
Day 17: Minimal Techno Percussive Hit
Day 18: カスタムウェーブテーブル実験
Day 19: モーフィングPad作成
Day 20: FXサウンド（Rise、Impact）
Day 21: ジャンルMix用サウンドセット完成

Week 4（実戦投入）:
Day 22: 演習1（1曲分のサウンドセット）開始
Day 23: 演習1続き、アレンジ開始
Day 24: 演習1完成、ミキシング
Day 25: 演習3（リファレンス再現）
Day 26: 自分のオリジナル音色5種作成
Day 27: ライブ演奏用Macro最適化
Day 28: 全プリセットの最終整理
Day 29: 完成曲のブラッシュアップ
Day 30: ポートフォリオ完成、次ステップ計画

1日の練習時間: 1-2時間
合計: 30-60時間
目標: 任意の音色を30分以内で作成可能
```

---

## 次に読むべきガイド

- 同カテゴリの他のガイドを参照してください

---

## 参考文献

- [MDN Web Docs](https://developer.mozilla.org/) - Web技術のリファレンス
- [Wikipedia](https://ja.wikipedia.org/) - 技術概念の概要
