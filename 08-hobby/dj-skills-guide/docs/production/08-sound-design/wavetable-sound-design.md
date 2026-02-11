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
