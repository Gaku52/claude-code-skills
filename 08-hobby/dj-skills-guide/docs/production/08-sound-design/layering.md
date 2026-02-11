# Layering（レイヤリング）

複数の音を重ねて厚みのあるサウンドを作るレイヤリング技術を完全マスター。周波数帯域の住み分け、EQ戦略、ベース・リード・ドラムのレイヤリングで、プロレベルの音圧と存在感を実現します。

## この章で学ぶこと

- レイヤリングの基本原理
- 周波数帯域の住み分け（Low/Mid/High）
- ベースレイヤリング（Sub + Mid）
- リードレイヤリング（Main + Octave）
- ドラムレイヤリング（Kick: Sub + Punch + Click）
- EQ戦略とバランス調整
- 実践的な練習方法

---

## なぜLayeringが重要なのか

**音の厚みと存在感:**

```
1音のみ:

状況:
Wavetable 1つ
シンプル
薄い音

結果:
存在感なし
迫力なし
プロと差がある

レイヤリングあり:

状況:
3-4音を重ねる
各周波数帯域
厚い音

結果:
存在感大
迫力大
プロレベル

プロの使用率:

ベースレイヤリング: 80%
リードレイヤリング: 70%
ドラムレイヤリング: 90%

理由:
1音では音圧不足
レイヤリング必須
```

**周波数帯域の重要性:**

```
全帯域1音:

問題:
Low（60-200 Hz）: 弱い
Mid（200 Hz-2 kHz）: 普通
High（2 kHz-10 kHz）: 弱い

結果:
薄い音
迫力なし

帯域別レイヤリング:

解決:
Low: Sub Bass（Sine波）
Mid: Main Bass（Saw波）
High: Click音（Noise）

結果:
全帯域カバー
厚い音
迫力大
```

---

## レイヤリングの基本原理

### 周波数帯域の住み分け

```
周波数スペクトラム（20 Hz - 20 kHz）:

Low（低域）: 60-200 Hz
用途: Sub Bass、Kick
感覚: 腹に響く、重低音
楽器: Sub Bass、808 Bass、Kick

Low-Mid（低中域）: 200 Hz-500 Hz
用途: ベース本体、男性ボーカル
感覚: 温かみ、厚み
楽器: Bass、Male Vocal、Snare Body

Mid（中域）: 500 Hz-2 kHz
用途: メロディ、コード
感覚: 存在感、明瞭度
楽器: Lead、Pad、Female Vocal

High-Mid（高中域）: 2 kHz-5 kHz
用途: アタック、明瞭度
感覚: 明るさ、エッジ
楽器: Lead High、Hi-Hat、Vocal Consonant

High（高域）: 5 kHz-10 kHz
用途: 空気感、煌びやか
感覚: キラキラ、エアー
楽器: Cymbal、Synth High、Vocal Breath

Very High（超高域）: 10 kHz-20 kHz
用途: 空気感、質感
感覚: 空間、透明感
楽器: Cymbal Shimmer、Reverb Tail
```

### レイヤリングの3原則

**原則1: 周波数帯域の住み分け**

```
NG例（住み分けなし）:

Layer 1: Sub Bass（60-200 Hz）
Layer 2: Sub Bass（60-200 Hz）
Layer 3: Sub Bass（60-200 Hz）

問題:
全て同じ帯域
濁る
音圧上がらない

OK例（住み分けあり）:

Layer 1: Sub Bass（60-150 Hz）
Layer 2: Mid Bass（150-800 Hz）
Layer 3: High Click（2 kHz-8 kHz）

結果:
各帯域カバー
明瞭
音圧大
```

**原則2: 音量バランス**

```
音量配分:

Low層: 50%（最重要）
Mid層: 30%
High層: 20%

理由:
低域 = エネルギーの源
中域 = メロディ認識
高域 = 煌びやか、控えめ

NG例:
Low: 30%
Mid: 30%
High: 40%（強すぎ）

結果:
シャカシャカ
迫力なし
```

**原則3: 位相の一致**

```
位相ズレ:

問題:
Layer 1: アタック 0 ms
Layer 2: アタック 20 ms

結果:
位相キャンセル
音量減少
薄い音

解決:
全Layer: アタック 0 ms
または
Sample Offsetで調整

確認:
位相メーター
波形の頭揃え
```

---

## 実践1: ベースレイヤリング（Sub + Mid）

**目標:** Sub Bass（低域）+ Mid Bass（中域）で太いベースを作る

### Step 1: Sub Bass作成（15分）

```
トラック1: Sub Bass

楽器: Wavetable

OSC 1:
Wavetable: Basic Shapes > Sine
Octave: 0
Volume: 100%

OSC 2: Off
SUB: 0%
UNISON: 1 voice

Filter:
Type: Low Pass 24 dB
Cutoff: 150 Hz
Resonance: 0%

Amp Envelope:
A: 0 ms
D: 50 ms
S: 100%
R: 10 ms

エフェクト:

1. EQ Eight:
   High Pass: 30 Hz
   Low Pass: 180 Hz（重要）
   理由: 60-150 Hzのみ残す

2. Saturator:
   Drive: 4 dB
   Curve: Warm

3. Utility:
   Width: 0%（Mono必須）
   Gain: -3 dB

役割: 低域のみ担当
確認: スペクトラムで150 Hz以上ほぼなし
```

### Step 2: Mid Bass作成（20分）

```
トラック2: Mid Bass

楽器: Wavetable

OSC 1:
Wavetable: Basic Shapes > Saw
Octave: 0
Volume: 100%

OSC 2:
Wavetable: Basic Shapes > Square
Octave: 0
Detune: +7 cent
Volume: 60%

SUB: 0%（Sub Bassに任せる）
UNISON: 3 voices、Detune 12%

Filter:
Type: Low Pass 24 dB
Cutoff: 1200 Hz
Resonance: 25%

Filter Envelope:
A: 0 ms
D: 150 ms
S: 30%
R: 40 ms
Amount: +25

Amp Envelope:
A: 0 ms
D: 80 ms
S: 85%
R: 15 ms

エフェクト:

1. EQ Eight:
   High Pass: 200 Hz（重要）
   Low Pass: 2500 Hz
   Peak: 800 Hz、+2 dB、Q 1.2
   理由: 200 Hz-2 kHz担当、Sub と住み分け

2. Saturator:
   Drive: 7 dB
   Curve: A Bit Warmer

3. Auto Filter（オプション）:
   Cutoff: 動かす

4. Utility:
   Width: 0%（Mono）
   Gain: -5 dB

役割: 中域担当、音色・メロディ認識
確認: スペクトラムで200 Hz-2 kHz が強い
```

### Step 3: レイヤリング調整（15分）

```
音量バランス:

Sub Bass: -3 dB（基準）
Mid Bass: -5 dB（少し小さめ）

比率:
Sub: 60%
Mid: 40%

EQ確認:

1. Sub Bass solo:
   60-150 Hz のみ

2. Mid Bass solo:
   200 Hz-2 kHz のみ

3. 両方On:
   60-2 kHz 全てカバー
   濁りなし

位相確認:

両方のアタック: 0 ms
→ 位相一致

確認方法:
波形表示で頭が揃っている
```

### Step 4: MIDI設定（10分）

```
MIDI共有:

方法:
1. Sub Bassトラックに MIDIクリップ作成
2. Mid Bassトラックの MIDI From: Sub Bass

理由:
1つのMIDIで両方制御
編集効率

MIDIパターン（8小節）:

Bar 1: C1 ──── ──── ──── ────
Bar 2: F1 ──── ──── ──── ────
Bar 3: C1 ──── G1 ── ──── ────
Bar 4: F1 ──── ──── ──── ────
Bar 5-8: 繰り返し

ベロシティ: 100-120
Length: 1拍
```

### Step 5: グループ化（5分）

```
Audio Effect Rackで囲む:

1. Sub Bass、Mid Bass 両トラック選択
2. Cmd+G（グループ化）
3. グループ名: "Layered Bass"

グループエフェクト:

1. Compressor（グルー）:
   Ratio: 2:1
   Threshold: -8 dB
   Attack: 10 ms
   Release: 80 ms
   Gain: +2 dB
   理由: 2層を接着

2. EQ Eight（最終調整）:
   High Pass: 35 Hz
   Peak: 100 Hz、+1 dB、Q 0.8

3. Limiter:
   Ceiling: -0.3 dB
   理由: 音量制御

4. Utility:
   Gain: 調整
   Width: 0%（Mono維持）
```

### 完成基準

```
□ Sub Bass（60-150 Hz）
□ Mid Bass（200 Hz-2 kHz）
□ 周波数住み分け完璧
□ 音量バランス Sub 60%、Mid 40%
□ 位相一致
□ Mono維持
□ 太い、迫力あるベース

所要時間: 65分
```

---

## 実践2: リードレイヤリング（Main + Octave + Detune）

**目標:** Main Lead + Octave Lead + Detune Leadで厚いリードを作る

### Step 1: Main Lead作成（20分）

```
トラック1: Main Lead

楽器: Wavetable

OSC 1:
Wavetable: Modern Shapes > Formant Square
Position: 25%
Octave: 0

OSC 2:
Wavetable: Basic Shapes > Saw
Octave: 0
Detune: +8 cent
Volume: 70%

SUB: 15%
UNISON: 5 voices、Detune 15%

Filter:
Type: Low Pass 12 dB
Cutoff: 2800 Hz
Resonance: 18%

Amp Envelope:
A: 15 ms
D: 250 ms
S: 70%
R: 350 ms

エフェクト:

1. EQ Eight:
   High Pass: 250 Hz
   Peak: 1500 Hz、+2 dB、Q 1.0

2. Chorus:
   Rate: 0.5 Hz
   Amount: 28%
   Dry/Wet: 35%

3. Reverb:
   Decay: 1.8s
   Dry/Wet: 20%

4. Utility:
   Width: 100%
   Gain: -2 dB（基準）

役割: メインメロディ
```

### Step 2: Octave Lead作成（15分）

```
トラック2: Octave Lead

楽器: Wavetable

設定:
Main Leadと同じ設定
ただし:

Transpose: +12 semitones（1オクターブ上）

OSC 2:
Detune: +12 cent（Main と少し違う）

UNISON:
Detune: 18%（Main より広い）

エフェクト調整:

1. EQ Eight:
   High Pass: 800 Hz（Main より高い）
   Peak: 3500 Hz、+2 dB、Q 1.2

2. Chorus:
   Dry/Wet: 40%（Main より強い）

3. Reverb:
   Dry/Wet: 25%

4. Utility:
   Width: 120%（Main より広い）
   Gain: -8 dB（小さめ）

役割: 1オクターブ上でハーモニー
音量: Main の 30%程度
```

### Step 3: Detune Lead作成（15分）

```
トラック3: Detune Lead

楽器: Wavetable

設定:
Main Leadと同じ基本設定
ただし:

OSC 1:
Detune: -12 cent（Main より低い）

OSC 2:
Detune: +18 cent（Main より高い）

UNISON:
Amount: 6 voices（Main より多い）
Detune: 22%（Main より広い）

Filter:
Cutoff: 3200 Hz（Main より開く）

エフェクト:

1. EQ Eight:
   High Pass: 300 Hz
   Peak: 2000 Hz、+1.5 dB、Q 0.8

2. Chorus:
   Dry/Wet: 45%（最も強い）

3. Reverb:
   Dry/Wet: 22%

4. Delay:
   Time: 1/8
   Feedback: 15%
   Dry/Wet: 12%

5. Utility:
   Width: 140%（最も広い）
   Gain: -6 dB

役割: 広がり、厚み
音量: Main の 40%程度
```

### Step 4: レイヤリング調整（20分）

```
音量バランス:

Main Lead: -2 dB（100%、基準）
Octave Lead: -8 dB（30%）
Detune Lead: -6 dB（40%）

合計: 170%

周波数帯域確認:

Main Lead: 250 Hz-5 kHz（メイン）
Octave Lead: 800 Hz-8 kHz（高域）
Detune Lead: 300 Hz-6 kHz（広がり）

ステレオ配置:

Main Lead: Width 100%（中央+両側）
Octave Lead: Width 120%（やや広い）
Detune Lead: Width 140%（最も広い）

結果:
中央に Main
両側に Octave、Detune
→ 立体的
```

### Step 5: グループエフェクト（15分）

```
グループ化:
3トラック選択 → Cmd+G
グループ名: "Layered Lead"

グループエフェクト:

1. Compressor:
   Ratio: 3:1
   Threshold: -10 dB
   Attack: 8 ms
   Release: 100 ms
   Gain: +3 dB
   理由: 3層を接着、ダイナミクス制御

2. EQ Eight:
   High Pass: 280 Hz
   Peak: 1800 Hz、+1.5 dB、Q 1.0
   Peak: 4000 Hz、+1 dB、Q 0.8
   High Shelf: 8 kHz、+1 dB
   理由: 全体の存在感向上

3. Reverb:
   Decay: 2.2s
   Dry/Wet: 15%
   理由: 統一感

4. Limiter:
   Ceiling: -0.5 dB

5. Utility:
   Gain: 調整
   Width: 130%（最終ステレオ調整）
```

### Step 6: MIDI演奏（10分）

```
MIDI共有:
Main Leadに MIDIクリップ
他2つは MIDI From: Main Lead

メロディ（8小節）:

Bar 1-2:
C4 ──── Eb4 ─ G4 ── Bb4 ─

Bar 3-4:
C5 ──── G4 ── Eb4 ─ C4 ───

Bar 5-6:
F4 ──── Ab4 ─ C5 ── Eb5 ─

Bar 7-8:
C5 ──── Bb4 ─ G4 ── C4 ───

ベロシティ: 95-115

確認:
□ メロディ明瞭
□ 厚みがある
□ 広がりがある
□ 3層が融合
□ 存在感大
```

### 完成基準

```
□ Main Lead（メイン）
□ Octave Lead（+12 semitones）
□ Detune Lead（広がり）
□ 音量バランス 100%:30%:40%
□ ステレオ配置 100%:120%:140%
□ グループCompressor
□ 厚い、存在感あるLead

所要時間: 95分
```

---

## 実践3: ドラムレイヤリング（Kick: Sub + Punch + Click）

**目標:** 3層Kickで完璧なバランスを作る

### Step 1: Sub Kick作成（12分）

```
トラック1: Sub Kick

方法1: Simplerでサンプル加工

サンプル: Sine Kick（低域のみ）
Transpose: -12 semitones
Filter: LP 120 Hz
Amp Envelope:
  A: 0 ms
  D: 80 ms
  S: 0%
  R: 0 ms

方法2: Wavetableで作成

OSC 1: Sine波
Filter: LP 100 Hz
Pitch Envelope:
  Amount: +12 semitones
  A: 0 ms
  D: 40 ms
  S: 0%
  R: 0 ms
→ ピッチ下降、キック感

エフェクト:

1. EQ Eight:
   High Pass: 35 Hz
   Low Pass: 150 Hz（重要）
   Peak: 60 Hz、+3 dB、Q 1.5
   理由: 60 Hzのみ強調

2. Saturator:
   Drive: 5 dB

3. Utility:
   Width: 0%（Mono必須）
   Gain: -1 dB

役割: 腹に響く低域
確認: 50-100 Hz が最強
```

### Step 2: Punch Kick作成（12分）

```
トラック2: Punch Kick

サンプル: Acoustic Kick、909 Kick

Simpler設定:
Transpose: 0 semitones
Filter: BP 200-800 Hz（Band Pass）
→ 中域のみ残す

Amp Envelope:
A: 0 ms
D: 50 ms
S: 0%
R: 0 ms

エフェクト:

1. EQ Eight:
   High Pass: 180 Hz（重要）
   Low Pass: 1200 Hz
   Peak: 400 Hz、+4 dB、Q 1.2
   理由: 200-800 Hz担当、パンチ感

2. Compressor:
   Ratio: 6:1
   Threshold: -12 dB
   Attack: 0 ms
   Release: 30 ms
   Gain: +6 dB
   理由: アタック強化

3. Saturator:
   Drive: 8 dB
   Curve: A Bit Warmer

4. Utility:
   Width: 0%（Mono）
   Gain: -4 dB

役割: パンチ、アタック
確認: 200-800 Hz が強い
```

### Step 3: Click Kick作成（12分）

```
トラック3: Click Kick

方法1: Noisesample + Filter

サンプル: White Noise
Length: 10 ms（超短い）

Simpler:
Filter: HP 2000 Hz
→ 高域のみ

Amp Envelope:
A: 0 ms
D: 8 ms（超短い）
S: 0%
R: 0 ms

方法2: 高域Kickサンプル

サンプル: 909 Rim、Electronic Kick
Filter: HP 1500 Hz

エフェクト:

1. EQ Eight:
   High Pass: 2000 Hz（重要）
   Peak: 4000 Hz、+3 dB、Q 1.5
   理由: 2-8 kHz担当、クリック音

2. Transient Shaper:
   Attack: +6 dB
   Sustain: -6 dB
   理由: アタック極大化

3. Saturator:
   Drive: 10 dB
   Curve: Hard Curve

4. Utility:
   Width: 0%（Mono）
   Gain: -10 dB（小さめ）

役割: クリック、アタック明瞭化
確認: 2-8 kHz のみ
```

### Step 4: レイヤリング調整（15分）

```
音量バランス:

Sub Kick: -1 dB（60%）
Punch Kick: -4 dB（30%）
Click Kick: -10 dB（10%）

比率:
Sub: 60%（最重要）
Punch: 30%
Click: 10%

周波数確認:

Sub: 50-150 Hz
Punch: 200-800 Hz
Click: 2-8 kHz

合計:
50 Hz-8 kHz 全てカバー
濁りなし

位相確認:

全てのアタック: 0 ms
→ 完璧な位相一致

Sample Offset調整:
必要なら微調整（±1 ms）
```

### Step 5: グループ化とエフェクト（15分）

```
グループ化:
3トラック選択 → Cmd+G
グループ名: "Layered Kick"

グループエフェクト:

1. Glue Compressor:
   Ratio: 2:1
   Threshold: -8 dB
   Attack: 0.1 ms
   Release: 60 ms
   Makeup: +3 dB
   理由: 3層を接着

2. EQ Eight:
   High Pass: 30 Hz
   Peak: 60 Hz、+2 dB、Q 1.2
   Peak: 400 Hz、+1 dB、Q 0.8
   理由: 全体バランス

3. Transient Shaper:
   Attack: +3 dB
   Sustain: -3 dB
   理由: アタック強化

4. Limiter:
   Ceiling: -0.3 dB
   理由: 音量制御

5. Utility:
   Width: 0%（Mono維持）
   Gain: 調整
```

### Step 6: 他のドラムと組み合わせ（10分）

```
ドラムパターン（4小節）:

Layered Kick:
|X---|X---|X---|X---|
4つ打ち

Snare:
|----X---|----X---|
2、4拍目

Hi-Hat（Closed）:
|X-X-X-X-|X-X-X-X-|
8分音符

Clap:
|----X---|----X---|
Snareと同時

確認:
□ Kick が明瞭
□ 低域（Sub）が強い
□ アタック（Click）が明瞭
□ 他のドラムと相性良い
```

### 完成基準

```
□ Sub Kick（50-150 Hz、60%）
□ Punch Kick（200-800 Hz、30%）
□ Click Kick（2-8 kHz、10%）
□ 周波数住み分け完璧
□ 位相一致
□ Mono維持
□ 完璧なKick

所要時間: 75分
```

---

## EQ戦略とバランス調整

### 周波数住み分けEQ設定

```
原則:

各Layer:
High Pass: その層の最低周波数
Low Pass: その層の最高周波数

例（ベースレイヤリング）:

Sub Bass:
HP: 30 Hz
LP: 180 Hz
→ 30-180 Hz のみ

Mid Bass:
HP: 200 Hz
LP: 2500 Hz
→ 200-2500 Hz のみ

結果:
各層が独立
濁りなし
明瞭
```

### Groupエフェクトの重要性

```
Glue Compressor（接着剤）:

設定:
Ratio: 2:1 - 3:1
Threshold: -8 dB
Attack: 遅め（5-10 ms）
Release: 速め（50-100 ms）

効果:
複数の音を1つの音に
一体感
グルーヴ

使用頻度: 90%
```

### 音量バランスの黄金比

```
ベースレイヤリング:

Sub: 60%（最重要）
Mid: 40%

リードレイヤリング:

Main: 100%（基準）
Octave: 30%
Detune: 40%
合計: 170%

ドラムレイヤリング:

Sub: 60%
Punch: 30%
Click: 10%
合計: 100%

理由:
低域 = エネルギー源
中域 = メロディ認識
高域 = 煌びやか（控えめ）
```

---

## よくある質問（FAQ）

**Q1: 何層まで重ねるべきですか？**

```
A: 2-4層が適切

推奨:

ベース:
2層（Sub + Mid）: 80%
3層（Sub + Mid + High）: 20%

リード:
3層（Main + Octave + Detune）: 70%
2層（Main + Octave）: 30%

ドラム:
Kick 3層（Sub + Punch + Click）: 90%
Snare 2層（Body + Snap）: 80%

理由:
2-4層 = バランス良い
5層以上 = 濁る、管理困難

NG:
10層 = 濁りまくり
```

**Q2: レイヤーが濁ります**

```
A: 周波数住み分け不足

チェックリスト:

□ 各層にHP/LP設定しているか
□ 周波数帯域が重複していないか
□ スペクトラムで視覚的確認したか

修正手順:

1. 各層をSolo
2. スペクトラム確認
3. 重複帯域発見
4. EQで削る

例:
Sub Bass: LP 180 Hz
Mid Bass: HP 200 Hz
→ 180-200 Hz ギャップOK

Sub Bass: LP 250 Hz
Mid Bass: HP 200 Hz
→ 200-250 Hz 重複 = 濁り
```

**Q3: 位相がズレているか確認する方法は？**

```
A: 波形とPhaseメーター

方法1: 波形確認

1. 2つの層を表示
2. 波形のアタック（頭）を見る
3. 揃っている = OK
4. ズレている = NG

修正:
Sample Offsetで調整

方法2: Phaseメーター

1. Utility追加
2. Phase表示: On
3. 0° = 完璧
4. ±30° = 許容範囲
5. ±90° = 問題あり

修正:
位相反転ボタン（Utility）
```

**Q4: ステレオ配置のコツは？**

```
A: 低域Mono、高域Stereo

原則:

Low（60-200 Hz）:
Width: 0%（完全Mono）
理由: 低域はMono必須、音圧

Mid（200 Hz-2 kHz）:
Width: 0-50%
理由: ややMono、中央定位

High（2 kHz以上）:
Width: 100-150%
理由: Stereo、広がり

例（リードレイヤリング）:

Main Lead: Width 100%
Octave Lead: Width 120%
Detune Lead: Width 140%

結果:
中央にMain
両側に高域
立体的
```

**Q5: CPU負荷が高すぎます**

```
A: Freeze、Resample

手順:

1. Freeze:
   各トラック右クリック → Freeze Track
   CPU大幅削減

2. Resample:
   新規Audioトラック
   レイヤーGroup を録音
   元のTrackを削除
   → 完全にAudio化

3. 必要な層のみ:
   3層で十分
   4層以上は本当に必要か再検討

4. エフェクト削減:
   各層のReverb、Delayを減らす
   Groupエフェクトに集約
```

**Q6: どの音をレイヤリングすべきですか？**

```
A: 重要な音のみ

優先順位:

1位: Kick（90%）
理由: 最重要、必須

2位: ベース（80%）
理由: 低域+中域必要

3位: Main Lead（70%）
理由: 存在感

4位: Snare（60%）
理由: アタック+Body

5位: Pad（50%）
理由: 広がり

不要:
Hi-Hat: 1層で十分
FX: 1層で十分
```

---

## 練習方法

### Week 1: ベースレイヤリングマスター

```
Day 1-2: 基本2層ベース
1. Sub + Mid Bass作成
2. 周波数住み分け確認
3. 5種類のベース作成

Day 3-4: 3層ベース
1. Sub + Mid + High Click
2. 周波数帯域 60-8 kHz
3. Techno、Dubstep用

Day 5-6: ジャンル別
1. House用（Sub + Mid）
2. Techno用（Sub + Mid + High）
3. Dubstep用（Sub + Mid + Wobble）

Day 7: 実戦投入
1. 自分の楽曲でレイヤーベース使用
2. Kickとの相性確認
3. ミックスバランス調整

目標:
ベースレイヤリングを30分で完了
```

### Week 2: リードレイヤリングマスター

```
Day 1-2: Main + Octave
1. 基本2層リード
2. 音量バランス 100%:30%
3. 5種類作成

Day 3-4: Main + Octave + Detune
1. 3層リード
2. ステレオ配置 100%:120%:140%
3. 厚みと広がり

Day 5-6: ジャンル別
1. Trance Pluck（2層）
2. House Lead（3層）
3. Dubstep Lead（3層）

Day 7: メロディ作成
1. レイヤーリードでメロディ5個
2. 異なるジャンル
3. 完成

目標:
リードレイヤリングを45分で完了
```

### Week 3: ドラムレイヤリングマスター

```
Day 1-2: Kick 3層
1. Sub + Punch + Click
2. 周波数 50-8 kHz
3. 5種類のKick作成

Day 3-4: Snare 2層
1. Body + Snap
2. 周波数 200 Hz-6 kHz
3. 5種類のSnare作成

Day 5-6: ドラムキット
1. レイヤーKick、Snare
2. Hi-Hat、Clap追加
3. 完全なドラムキット作成

Day 7: ドラムパターン
1. 5種類のジャンルパターン
2. Techno、House、Hip Hop、D&B、Dubstep
3. 完成

目標:
ドラムレイヤリングを60分で完了
```

### Week 4: 総合練習

```
Day 1-2: 楽曲制作（レイヤリングのみ）
1. レイヤーベース
2. レイヤーリード
3. レイヤードラム
4. 他の音は1層

Day 3-4: A/B比較
1. レイヤリングあり版
2. レイヤリングなし版（各1層）
3. 違いを確認
4. レイヤリングの効果実感

Day 5-6: プロ曲分析
1. 好きな曲をAbletonに読み込み
2. スペクトラム分析
3. 何層レイヤーされているか推測
4. 再現を試みる

Day 7: 完成曲制作
1. 全ての音をレイヤリング
2. ミックス
3. 完成、公開

目標:
レイヤリングを自在に使える
```

---

## まとめ

レイヤリングは、音に厚みと存在感を与える必須技術です。

**重要ポイント:**

1. **周波数住み分け**: 各層が独立した帯域を担当
2. **音量バランス**: Low 60%、Mid 30%、High 10%
3. **位相一致**: 全層のアタック 0 ms
4. **EQ戦略**: HP/LP で明確に帯域分離
5. **Glue Compressor**: 複数層を1つの音に
6. **Mono/Stereo**: 低域Mono、高域Stereo

**学習順序:**
1. ベースレイヤリング（Week 1）
2. リードレイヤリング（Week 2）
3. ドラムレイヤリング（Week 3）
4. 総合練習（Week 4）

**推奨層数:**
- ベース: 2層（Sub + Mid）
- リード: 3層（Main + Octave + Detune）
- Kick: 3層（Sub + Punch + Click）
- Snare: 2層（Body + Snap）

**次のステップ:** [Modulation Techniques（モジュレーション技術）](./modulation-techniques.md) へ進む

---

## 関連ファイル

- **[Wavetable Sound Design](./wavetable-sound-design.md)** - シンセサウンド作成
- **[Sampling Techniques](./sampling-techniques.md)** - サンプリング技術
- **[Modulation Techniques](./modulation-techniques.md)** - LFO・Envelope活用
- **[05-mixing/balance.md](../05-mixing/)** - ミキシングバランス

---

**レイヤリングで、太く存在感のある音を作りましょう！**
