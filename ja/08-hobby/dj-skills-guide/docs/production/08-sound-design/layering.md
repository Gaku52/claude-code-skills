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


## 前提知識

このガイドを読む前に、以下の知識があると理解が深まります:

- 基本的なプログラミングの知識
- 関連する基礎概念の理解
- [Genre Sounds（ジャンル別サウンド）](./genre-sounds.md) の内容を理解していること

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

---

## 高度なレイヤリングテクニック

基本的なレイヤリングをマスターしたら、次はより高度なテクニックを学んでいきます。プロのプロデューサーが実践する応用技術を詳しく解説します。

### パラレルプロセッシングとレイヤリングの融合

パラレルプロセッシング（ドライ信号とウェット信号を混ぜる技法）をレイヤリングに組み合わせると、さらに洗練された結果が得られます。

```
パラレルプロセッシングの基本概念:

通常のレイヤリング:
Layer 1 → エフェクト → 出力
Layer 2 → エフェクト → 出力
Layer 3 → エフェクト → 出力

パラレルプロセッシング併用:
Layer 1（Dry）────────────────────→ ミックス
Layer 1（Wet）→ Heavy Compression → ミックス
Layer 2（Dry）────────────────────→ ミックス
Layer 2（Wet）→ Saturator ────────→ ミックス

利点:
- 原音の質感を保持しつつ迫力を追加
- Dry/Wetバランスで微調整可能
- CPU効率が良い（レイヤー追加不要で厚みを出せる）
```

**パラレルコンプレッション（New York Compression）:**

```
設定手順（Ableton Live）:

1. Audio Effect Rack を作成
2. Chain 1: Dry（何もしない）
3. Chain 2: Wet（Compressor追加）

Chain 2 Compressor設定:
  Ratio: 10:1（ハードコンプレッション）
  Threshold: -30 dB（深いコンプ）
  Attack: 0 ms
  Release: Auto
  Gain: +12 dB

Chain 1 Volume: 0 dB
Chain 2 Volume: -12 dB（調整）

効果:
- 元の音のダイナミクスを保持
- Wet側で持続感と密度を追加
- ドラム全体、ベースに特に有効

使用率:
  Kick レイヤリング: 70%のプロが使用
  Snare レイヤリング: 80%のプロが使用
  ベース レイヤリング: 60%のプロが使用
```

**パラレルサチュレーション:**

```
設定手順:

1. Audio Effect Rackで2チェイン
2. Chain 1: Dry
3. Chain 2: Saturator + EQ

Chain 2設定:
  Saturator:
    Drive: 15-20 dB
    Curve: Hard Curve
    Dry/Wet: 100%

  EQ Eight:
    High Pass: 200 Hz
    Peak: 2 kHz, +3 dB, Q 1.0
    理由: 歪みの美味しい帯域だけ残す

  Volume: -15 dB

効果:
- ハーモニクス追加
- 倍音で音が太くなる
- 元音に混ぜて自然な厚み

適用先:
  ベースレイヤー: Mid Bass層に適用
  リードレイヤー: Main Lead層に適用
  Kick: Punch Kick層に適用
```

### マルチバンドレイヤリング

1つのサウンドを周波数帯域ごとに分割し、各帯域に異なる処理を適用する高度テクニックです。

```
マルチバンドレイヤリングの仕組み:

原音（フルレンジ）
  │
  ├── Low Band（30-200 Hz）→ 個別処理 → ミックス
  ├── Mid Band（200-2000 Hz）→ 個別処理 → ミックス
  └── High Band（2000-20000 Hz）→ 個別処理 → ミックス

Ableton Liveでの実装:

方法1: Audio Effect Rack + EQ

1. Audio Effect Rack 作成
2. 3つのChainを作成
3. 各ChainにEQ Eightを配置

Chain 1（Low）:
  EQ Eight:
    HP: 30 Hz
    LP: 200 Hz（24 dB/oct）
  処理: Saturator（Drive 5 dB）、Compressor
  Utility: Width 0%（Mono）

Chain 2（Mid）:
  EQ Eight:
    HP: 200 Hz（24 dB/oct）
    LP: 2000 Hz（24 dB/oct）
  処理: Chorus、Saturator（Drive 8 dB）
  Utility: Width 60%

Chain 3（High）:
  EQ Eight:
    HP: 2000 Hz（24 dB/oct）
    LP: 20000 Hz
  処理: Reverb（Dry/Wet 20%）、Delay
  Utility: Width 140%

方法2: Multiband Dynamics を活用

Multiband Dynamics:
  Low Band: 0-200 Hz
    Above: Ratio 2:1, Threshold -8 dB
  Mid Band: 200-2500 Hz
    Above: Ratio 3:1, Threshold -10 dB
  High Band: 2500-20000 Hz
    Above: Ratio 1.5:1, Threshold -6 dB

利点:
- 1つの音源から帯域別に独立処理
- レイヤー数を増やさず音の厚みを出せる
- CPU負荷が比較的軽い
```

### Sidechain レイヤリング

レイヤリングされた音にサイドチェインを適用する際のテクニックです。

```
サイドチェインの適用方法:

NG例（各層に個別サイドチェイン）:
  Sub Bass → Sidechain → 出力
  Mid Bass → Sidechain → 出力
  問題: タイミングがズレる可能性

OK例（グループにサイドチェイン）:
  Sub Bass ─┐
  Mid Bass ─┼→ Group Bus → Sidechain → 出力

  理由: 統一されたポンピング

サイドチェイン設定（ベースレイヤー）:

Compressor on Group Bus:
  Sidechain Source: Kick
  Ratio: 4:1
  Threshold: -20 dB
  Attack: 0.5 ms
  Release: 80 ms（BPM依存）

  BPM別 Release目安:
    120 BPM: 100 ms
    125 BPM: 90 ms
    128 BPM: 85 ms
    130 BPM: 80 ms
    140 BPM: 70 ms
    150 BPM: 60 ms
    170 BPM: 50 ms

Volume Duck量:
  -3 dB: 軽め（House）
  -6 dB: 標準（Progressive House）
  -12 dB: 強め（Trance、EDM）
  -18 dB: 極端（Sidechain Bass）
```

---

## 周波数帯域別レイヤー設計の詳細

各周波数帯域でどのような音を配置すべきか、さらに詳細に解説します。

### Sub Bass帯域（20-80 Hz）

```
Sub Bass帯域の特性:

物理的特性:
  波長: 4.3 m（80 Hz）〜 17.2 m（20 Hz）
  知覚: 振動として感じる帯域
  再生: サブウーファーが必要
  クラブ: 最も重要な帯域

設計原則:
  1. 純粋なSine波が最適
  2. 完全Mono必須
  3. 1音のみ（和音NG）
  4. ピッチエンベロープで動きを出す

サウンドデザイン:

Wavetable設定:
  OSC 1: Sine波
  Octave: -1 または 0
  UNISON: 1 voice（必須）

  Pitch Envelope:
    Amount: +5 to +12 semitones
    Attack: 0 ms
    Decay: 20-60 ms
    Sustain: 0 semitones
    Release: 0 ms

  効果: 「ドゥン」というキック的なアタック

EQ設定:
  HP: 20 Hz（DC Offset除去）
  LP: 80 Hz（24 dB/oct、急峻に）
  Peak: 50-60 Hz, +2 dB, Q 2.0

注意点:
  位相は絶対にズレてはいけない
  モニター環境が重要（ヘッドフォンだけでは不十分）
  スペクトラムアナライザーで常に確認
  A/B比較で音量チェック
```

### Low Bass帯域（80-200 Hz）

```
Low Bass帯域の特性:

物理的特性:
  波長: 1.7 m（200 Hz）〜 4.3 m（80 Hz）
  知覚: 低音の「重み」を感じる帯域
  再生: 通常のスピーカーでも再生可能
  ミックス: ベースの本体が存在する帯域

設計原則:
  1. Sub Bassとの住み分けが最重要
  2. Mono推奨（ステレオ幅は0-20%まで）
  3. ハーモニクスで存在感を出す
  4. Saturatorで倍音追加が有効

サウンドデザイン:

Wavetable設定:
  OSC 1: Triangle波 または Saw波
  Filter: LP 200 Hz
  Saturator: Drive 5-8 dB

  理由:
  Triangle波 → 奇数倍音のみ（クリーン）
  Saw波 → 全倍音（リッチ）

EQ設定:
  HP: 80 Hz（Sub Bassとの境界）
  LP: 200 Hz（Mid Bassとの境界）
  Peak: 120-150 Hz, +1.5 dB, Q 1.0

モニタリング:
  周波数アナライザーで80-200 Hzのエネルギー確認
  RMS値: -18 dBFS 〜 -12 dBFS が適正範囲
```

### Mid帯域（200 Hz-2 kHz）

```
Mid帯域の特性:

物理的特性:
  波長: 17 cm（2 kHz）〜 1.7 m（200 Hz）
  知覚: メロディ・コードの認識に最重要
  再生: 全てのスピーカーで再生可能
  ミックス: 最も「音楽的」な帯域

設計原則:
  1. 音色のキャラクターを決める帯域
  2. ステレオ幅 50-100%
  3. EQのカット/ブーストで音色変化が大きい
  4. マスキング（帯域の奪い合い）が最も起きやすい

問題帯域:
  200-400 Hz: 「モコモコ」（Muddy）
    対策: 不要ならカット -2 to -4 dB

  500-800 Hz: 「鼻声」（Nasal）
    対策: 狭いQ（3.0以上）でカット

  1-2 kHz: 「エッジ」（Presence）
    対策: ブーストで存在感、カットで後退

レイヤリングでの注意:
  複数のMid層は避ける
  1-2層に留める
  マスキングが発生したら片方をカット
  → 「周波数の譲り合い」が重要
```

### Presence帯域（2-5 kHz）

```
Presence帯域の特性:

物理的特性:
  波長: 7-17 cm
  知覚: 人間の耳が最も敏感な帯域
  再生: 全スピーカーで明瞭に再生
  ミックス: 音の「前に出る感」を決める

設計原則:
  1. 音のアタックが存在する帯域
  2. ステレオ幅 80-140%
  3. 過剰なブーストは耳に痛い（Harsh）
  4. De-Esserで処理する場合もある

レイヤリングでの活用:
  Click層: この帯域でアタック明瞭化
  Lead層: ここでメロディの明瞭度を確保

  ブースト例:
    +1 to +2 dB: 自然な存在感
    +3 to +4 dB: 前に出る
    +5 dB以上: 耳に痛い可能性

  カット例:
    -2 dB: やや後退
    -4 dB: 背景に下がる

EQ Eight設定:
  Peak: 3 kHz, +2 dB, Q 0.8
  Peak: 4.5 kHz, +1 dB, Q 1.2
```

### Air帯域（5-20 kHz）

```
Air帯域の特性:

物理的特性:
  波長: 1.7 cm（20 kHz）〜 7 cm（5 kHz）
  知覚: 「煌びやかさ」「空気感」
  再生: 高域の再生能力に依存
  ミックス: 「プロっぽさ」を決める帯域

設計原則:
  1. 控えめなレイヤリングで十分
  2. ステレオ幅 120-200%
  3. Reverb、Delay のテイルが多い帯域
  4. Noise層で空気感を追加

テクニック: Air Noise Layer

設定:
  サンプル: White Noise（Pink Noiseも可）
  Simpler:
    Filter: HP 8 kHz
    LP: 18 kHz

  Amp Envelope:
    A: 50 ms（ゆっくり立ち上がり）
    D: 200 ms
    S: 60%
    R: 500 ms

  Utility:
    Width: 200%（最大ステレオ）
    Gain: -20 dB（非常に小さく）

  EQ Eight:
    HP: 10 kHz
    Peak: 14 kHz, +2 dB, Q 0.5

効果:
  リードやパッドに空気感を追加
  「プロっぽい」音になる
  控えめに使うのがコツ
```

---

## キックのレイヤリング実践（応用編）

基本の3層キックをマスターしたら、さらに高度なキック制作に挑戦します。

### ジャンル別キック設計

```
Techno Kick（硬質・ミニマル）:

Layer 1: Sub（40-80 Hz）
  波形: Sine
  Pitch Env: +24 semitones → 0 semitones（30 ms）
  Duration: 100 ms
  EQ: HP 30 Hz, LP 100 Hz
  Volume: -2 dB

Layer 2: Body（100-500 Hz）
  サンプル: 909 Kick（加工）
  EQ: HP 100 Hz, LP 500 Hz
  Compressor: Ratio 4:1, Attack 1 ms
  Volume: -5 dB

Layer 3: Click（3-8 kHz）
  サンプル: Rim Shot + White Noise
  Duration: 5 ms
  EQ: HP 3 kHz, LP 10 kHz
  Volume: -12 dB

Layer 4: Tail（オプション）
  Reverb: Room, Decay 0.3s
  EQ: HP 200 Hz, LP 800 Hz
  Volume: -15 dB

グループ処理:
  Compressor: Ratio 3:1, Attack 0.5 ms, Release 40 ms
  EQ: Peak 60 Hz +2 dB, Cut 300 Hz -2 dB
  Saturator: Drive 3 dB, Warm
  Limiter: Ceiling -0.3 dB

特徴:
  タイトで短い
  クリック感が強い
  反復に耐える音質
```

```
House Kick（暖かみ・バウンス感）:

Layer 1: Sub（40-100 Hz）
  波形: Sine
  Pitch Env: +12 semitones → 0（40 ms）
  Duration: 150 ms（Technoより長い）
  EQ: HP 30 Hz, LP 120 Hz
  Volume: -1 dB

Layer 2: Body（100-600 Hz）
  サンプル: Acoustic Kick またはVintage Drum Machine
  EQ: HP 100 Hz, LP 600 Hz
  Compressor: Ratio 2:1, Attack 5 ms
  Saturator: Drive 4 dB, Warm Tube
  Volume: -4 dB

Layer 3: Click（2-6 kHz）
  サンプル: 木片、カスタネット系の短いトランジェント
  Duration: 8 ms
  EQ: HP 2 kHz, LP 8 kHz
  Volume: -14 dB

グループ処理:
  Glue Compressor: Ratio 2:1, Attack 3 ms, Release 80 ms
  EQ: Peak 80 Hz +1.5 dB, High Shelf 5 kHz -1 dB
  Limiter: Ceiling -0.5 dB

特徴:
  やや長めのテイル
  暖かみがある
  バウンス感重視
```

```
Drum & Bass Kick（タイトで重い）:

Layer 1: Sub（30-80 Hz）
  波形: Sine
  Pitch Env: +36 semitones → 0（20 ms）
  Duration: 60 ms（非常に短い）
  EQ: HP 25 Hz, LP 80 Hz
  Volume: -3 dB

Layer 2: Punch（150-800 Hz）
  サンプル: Acoustic Kick（タイトなもの）
  EQ: HP 150 Hz, LP 800 Hz
  Transient Shaper: Attack +8 dB
  Compressor: Ratio 6:1, Attack 0 ms
  Volume: -5 dB

Layer 3: Click（4-12 kHz）
  サンプル: Stick Hit、Wood Block
  Duration: 3 ms（超短い）
  EQ: HP 4 kHz
  Volume: -10 dB

グループ処理:
  Compressor: Ratio 4:1, Attack 0.1 ms, Release 30 ms
  EQ: Peak 50 Hz +3 dB, Cut 250 Hz -3 dB
  Limiter: Ceiling -0.1 dB

特徴:
  極めてタイト（短い）
  170 BPMでも明瞭
  Bassとの住み分けが重要
```

### キックのチューニング

```
キックのピッチとキーの関係:

重要性:
  キックのピッチが曲のキーと合っていないと
  ベースと「うなり」が発生する

確認方法:
  1. キックをソロで再生
  2. チューナーで基音を確認
  3. 曲のキーと合わせる

キー別推奨ピッチ:

Key: C → Kick: C1（32.7 Hz）
Key: D → Kick: D1（36.7 Hz）
Key: E → Kick: E1（41.2 Hz）
Key: F → Kick: F1（43.7 Hz）
Key: G → Kick: G1（49.0 Hz）
Key: A → Kick: A1（55.0 Hz）

Abletonでの調整:
  Simpler: Transpose パラメータで調整
  Wavetable: OSC Transpose で調整
  サンプル: Warp Modeで Transpose

注意:
  大きくTransposeすると音質劣化
  ±5 semitones以内が安全範囲
  それ以上はサンプルを変更
```

---

## スネアのレイヤリング実践

### 2層スネア（Body + Snap）

```
Layer 1: Body（スネア本体）

サンプル選び:
  推奨: 808 Snare、Acoustic Snare Body
  特性: 低中域が豊か、200-800 Hz

Simpler設定:
  Filter: BP 200-1000 Hz
  Amp Envelope:
    A: 0 ms
    D: 100 ms
    S: 20%
    R: 50 ms

EQ Eight:
  HP: 150 Hz
  LP: 1200 Hz
  Peak: 400 Hz, +2 dB, Q 1.0
  Peak: 800 Hz, +1 dB, Q 0.8

Compressor:
  Ratio: 4:1
  Threshold: -10 dB
  Attack: 2 ms
  Release: 50 ms

Utility:
  Width: 0%（Mono）
  Volume: -2 dB

役割: スネアの「胴鳴り」「重み」
```

```
Layer 2: Snap（スネアのアタック）

サンプル選び:
  推奨: Clap、Finger Snap、Noise Burst
  特性: 高域が明瞭、2-10 kHz

Simpler設定:
  Filter: HP 2000 Hz
  Amp Envelope:
    A: 0 ms
    D: 30 ms
    S: 0%
    R: 5 ms

EQ Eight:
  HP: 2000 Hz
  Peak: 4000 Hz, +3 dB, Q 1.5
  Peak: 7000 Hz, +2 dB, Q 1.0

Transient Shaper:
  Attack: +4 dB
  Sustain: -8 dB

Utility:
  Width: 60%（やや広げる）
  Volume: -8 dB

役割: スネアの「パチッ」というアタック感
```

```
グループ処理:

2層をグループ化: "Layered Snare"

Compressor:
  Ratio: 3:1
  Threshold: -8 dB
  Attack: 1 ms
  Release: 40 ms
  Gain: +2 dB

EQ Eight:
  HP: 120 Hz
  Peak: 250 Hz, +1 dB, Q 0.8
  Peak: 5 kHz, +1.5 dB, Q 1.0

Reverb:
  Type: Room
  Decay: 0.5 s
  Dry/Wet: 10%

Limiter:
  Ceiling: -0.5 dB

音量バランス:
  Body: 70%
  Snap: 30%
```

### 3層スネア（Body + Snap + Noise Tail）

```
Layer 3: Noise Tail（スネアの余韻）

追加するとさらにリッチに:

サンプル: White Noise
Length: 100-200 ms

Simpler設定:
  Filter: BP 1000-6000 Hz
  Amp Envelope:
    A: 5 ms
    D: 150 ms
    S: 0%
    R: 10 ms

EQ Eight:
  HP: 800 Hz
  LP: 8000 Hz
  Peak: 3000 Hz, +2 dB, Q 0.8

Utility:
  Width: 120%
  Volume: -15 dB

役割: スネアの「シャー」という余韻
    クラブスネアの質感を追加

3層の音量バランス:
  Body: 60%
  Snap: 25%
  Noise Tail: 15%
```

---

## ボーカルレイヤリング

ボーカルのレイヤリングは楽器のレイヤリングとは異なるアプローチが必要です。

### ボーカルダブリング

```
テクニック1: 実際のダブリング

方法:
  同じパートを2回以上録音
  微妙なタイミング・ピッチの差が自然な厚みに

録音時の注意:
  1. 同じマイクポジション
  2. 同じ距離
  3. 別テイクで録音（コピペNG）

パンニング:
  Take 1: Pan 0%（Center）
  Take 2: Pan -30%（やや左）
  Take 3: Pan +30%（やや右）

音量:
  Take 1: 0 dB（基準）
  Take 2: -4 dB
  Take 3: -4 dB

EQ処理:
  Take 1: フルレンジ
  Take 2: HP 300 Hz（低域カット）
  Take 3: HP 300 Hz（低域カット）

  理由: 低域は中央の1テイクのみ
```

```
テクニック2: 人工的なダブリング（ADT: Automatic Double Tracking）

Ableton Liveでの実装:

方法1: Simple Delay

設定:
  Mode: Time
  Left: 15-30 ms
  Right: 20-35 ms
  Feedback: 0%
  Dry/Wet: 30-40%

効果:
  ショートディレイでダブリング感
  手軽で効果的

方法2: Audio Effect Rack

Chain 1（Original）:
  そのまま

Chain 2（Double）:
  Simple Delay:
    Time: 20 ms
    Feedback: 0%
    Dry/Wet: 100%

  Pitch Shifter:
    Pitch: -5 to -10 cent

  EQ Eight:
    HP: 400 Hz
    LP: 8 kHz

  Utility:
    Width: 150%
    Volume: -6 dB

Chain 3（Triple、オプション）:
  Simple Delay:
    Time: 35 ms
    Feedback: 0%
    Dry/Wet: 100%

  Pitch Shifter:
    Pitch: +5 to +10 cent

  EQ Eight:
    HP: 500 Hz
    LP: 6 kHz

  Utility:
    Width: 180%
    Volume: -9 dB

効果:
  自然なダブリング/トリプリング
  ステレオの広がり
  ボーカルの存在感向上
```

### ボーカルハーモニーレイヤリング

```
ハーモニーレイヤリングの構成:

Lead Vocal: メインメロディ
  Pan: Center
  Volume: 0 dB（基準）
  EQ: フルレンジ
  Width: 100%

Harmony 1（3度上）:
  Pan: -40%（左）
  Volume: -6 dB
  EQ: HP 300 Hz, LP 10 kHz
  Width: 120%

Harmony 2（5度上）:
  Pan: +40%（右）
  Volume: -8 dB
  EQ: HP 400 Hz, LP 8 kHz
  Width: 130%

Harmony 3（オクターブ上、オプション）:
  Pan: Center
  Volume: -10 dB
  EQ: HP 500 Hz, LP 12 kHz
  Width: 140%

グループ処理:
  Bus Compressor:
    Ratio: 2:1
    Threshold: -6 dB
    Attack: 10 ms
    Release: 100 ms

  Reverb:
    Type: Plate
    Decay: 1.5 s
    Dry/Wet: 15%

  Delay:
    Time: 1/4
    Feedback: 20%
    Dry/Wet: 10%

EDMでのボーカルチョップとレイヤリング:

1. ボーカルサンプルをスライス
2. 各スライスをSimpler/Samplerにアサイン
3. MIDIでリズミカルに配置
4. レイヤー:
   - Original: Dry
   - Layer 2: Vocoder処理
   - Layer 3: Granular処理
5. グループで統一感を出す
```

---

## グループ処理とバスコンプレッションの詳細

### バスコンプレッションの種類と使い分け

```
1. VCA Compressor（SSL Bus Comp系）

特徴:
  クリーン、正確
  ミックスの統一感
  パンチ感

設定:
  Ratio: 2:1 - 4:1
  Threshold: -4 to -8 dB
  Attack: 10-30 ms（遅め）
  Release: Auto または 100-300 ms
  Gain Reduction: 2-4 dB

適用先:
  ドラムバス: 最適
  ミックスバス: 定番
  ベースグループ: 効果的

Ableton: Glue Compressor が該当
  Glue Compressorは SSL 4000シリーズをモデリング
```

```
2. FET Compressor（1176系）

特徴:
  カラフル、アグレッシブ
  アタックが速い
  倍音が豊か

設定:
  Ratio: 4:1 - 20:1
  Input（Threshold代わり）: 調整
  Attack: 最速（20 μs）
  Release: 速め（50 ms）

適用先:
  ボーカルバス: 存在感
  ドラムバス: アグレッシブ
  パラレルコンプ: 最適

Ableton: Compressor（Peak モード）で近似
  Attack: 0.01 ms
  Ratio: 高め
```

```
3. Optical Compressor（LA-2A系）

特徴:
  スムーズ、ナチュラル
  アタック/リリースが自動的
  温かみがある

設定:
  Peak Reduction: 調整
  Gain: 補正

適用先:
  ボーカルバス: 最適
  パッドグループ: 自然
  ベースグループ: スムーズ

Ableton: Compressor（RMS モード）で近似
  Attack: 遅め（10-30 ms）
  Release: Auto
```

```
4. バスコンプの適用順序

推奨チェイン（グループバス）:

1. EQ Eight（プリコンプ）
   HP: 不要な低域カット
   問題帯域のカット

2. Compressor / Glue Compressor
   Ratio: 2:1 - 3:1
   GR: 2-4 dB

3. EQ Eight（ポストコンプ）
   トーナル調整
   ブースト

4. Saturator（オプション）
   Drive: 2-4 dB
   倍音追加

5. Limiter
   Ceiling: -0.3 dB
   安全装置

6. Utility
   Gain: 最終調整
   Width: ステレオ調整
```

---

## 位相問題の対処法（詳細編）

### 位相とは何か

```
位相（Phase）の基礎知識:

定義:
  音波の時間的な位置関係
  0°（完全一致）〜 360°（1周期）

位相が一致（0°）:
  効果: 音量増大、音圧向上
  2つの同じ波形が重なる → +6 dB

位相が反転（180°）:
  効果: 音量消失、キャンセル
  2つの波形が打ち消し合う → 無音

位相のズレ（0°〜180°）:
  効果: コムフィルター（櫛型フィルター）
  特定の周波数が打ち消される → 薄い音

レイヤリングでの影響:
  2つの層のアタックがズレている
  → 低域の位相キャンセルが発生
  → 音圧が下がる
  → 薄く聞こえる
```

### 位相問題の検出方法

```
方法1: 聴覚による確認

手順:
  1. Layer AをSolo → 音を記憶
  2. Layer BをSolo → 音を記憶
  3. 両方On → 音量が増えるはず

  結果判定:
  □ 音量が増えた → 位相OK
  □ 音量が変わらない → 軽い位相ズレ
  □ 音量が減った → 位相キャンセル発生
  □ 音が薄くなった → コムフィルター発生
```

```
方法2: 位相反転テスト

手順:
  1. Layer Aを再生
  2. Layer BのUtility → Phase反転On（Øボタン）
  3. 両方On

  結果判定:
  □ 音量が大幅に減った → 位相が一致していた（良い状態）
  □ あまり変わらない → 位相がズレている（問題あり）

  理由:
  位相が一致 → 反転すると打ち消し → 音量減少
  位相がズレ → 反転しても変化少ない
```

```
方法3: スペクトラムアナライザーによる確認

手順:
  1. Layer A単体のスペクトラムを確認
  2. Layer B単体のスペクトラムを確認
  3. 両方Onのスペクトラムを確認

  確認ポイント:
  □ 全帯域で音量が増えている → OK
  □ 特定帯域で凹みがある → コムフィルター
  □ 低域が減っている → 低域位相キャンセル

  Abletonでの確認:
  Spectrum（Audioエフェクト）をグループに挿入
  Block Size: 4096（高精度）
  Average: On
```

```
方法4: 波形の目視確認

手順:
  1. 2つのレイヤーを並べて表示
  2. 波形のアタック（先頭）を拡大
  3. 波形の山と谷の位置を確認

  判定:
  □ 山と山が揃っている → 位相一致
  □ 山と谷が揃っている → 位相反転（180°）
  □ 微妙にズレている → 部分的位相キャンセル

  修正:
  Sample Offset で1サンプル単位で調整
  Abletonの場合: Clip → Start位置を微調整
```

### 位相問題の修正方法

```
修正法1: サンプルオフセット調整

手順:
  1. 基準レイヤーを決める（通常はSub層）
  2. 他のレイヤーのStart位置を微調整
  3. 1サンプルずつ移動
  4. スペクトラムで確認

  調整量の目安:
  44.1 kHz: 1 sample = 0.023 ms
  48 kHz: 1 sample = 0.021 ms

  実用的な調整範囲:
  ±10 samples（±0.23 ms以内）
```

```
修正法2: 位相反転（Phase Invert）

Utility:
  Phsボタン（Ø） → On

  効果:
  波形が上下反転（180° 回転）

  使用ケース:
  マイク録音の位相不一致
  サンプルの位相がたまたま逆の場合

  注意:
  完全な位相反転は0°か180°しかできない
  中間のズレには対応不可 → サンプルオフセットで
```

```
修正法3: リニアフェーズEQ

通常のEQ:
  位相回転が発生する（ミニマムフェーズ）
  各帯域で位相が変わる

リニアフェーズEQ:
  位相回転が発生しない
  全帯域で位相が一定

  欠点:
  CPU負荷が高い
  プリリンギング（音の前にゴースト音）
  レイテンシーが大きい

  使用推奨ケース:
  ドラムバスのEQ
  マスターバスのEQ
  位相精度が特に重要なレイヤリング

Abletonでの使用:
  EQ Eight → Mode: Linear Phase
  ※ CPU負荷に注意
```

---

## Ableton Rackでのレイヤー管理

### Instrument Rackを使ったレイヤリング

```
Instrument Rackの利点:

1. 1トラックで複数レイヤー管理
2. MIDIが自動で全Chainに配信
3. Chain Selectorで切り替え可能
4. Macroで一括コントロール
5. プリセットとして保存可能

設定手順:

1. MIDIトラックに Instrument Rack を挿入
2. 「Chain List」を表示
3. 各Chainにシンセ/サンプラーを配置

ベースレイヤリングの例:

Chain 1: "Sub Bass"
  │
  ├── Wavetable（Sine, LP 150 Hz）
  ├── Saturator（Drive 4 dB）
  ├── EQ Eight（HP 30 Hz, LP 180 Hz）
  └── Utility（Width 0%, Gain -3 dB）

Chain 2: "Mid Bass"
  │
  ├── Wavetable（Saw, Unison 3）
  ├── EQ Eight（HP 200 Hz, LP 2500 Hz）
  ├── Saturator（Drive 7 dB）
  └── Utility（Width 0%, Gain -5 dB）

Chain 3: "High Click"（オプション）
  │
  ├── Wavetable（Noise, HP 2 kHz）
  ├── EQ Eight（HP 2000 Hz）
  └── Utility（Width 0%, Gain -12 dB）

Rack After Chain:
  │
  ├── Compressor（Ratio 2:1）
  ├── EQ Eight（最終調整）
  └── Limiter（Ceiling -0.3 dB）
```

### Macroコントロールの設定

```
Macro 1: "Sub Volume"
  マッピング: Chain 1 Utility Gain
  範囲: -20 dB 〜 +6 dB
  デフォルト: -3 dB

Macro 2: "Mid Volume"
  マッピング: Chain 2 Utility Gain
  範囲: -20 dB 〜 +6 dB
  デフォルト: -5 dB

Macro 3: "Sub/Mid Balance"
  マッピング:
    Chain 1 Utility Gain: +6 dB 〜 -6 dB
    Chain 2 Utility Gain: -6 dB 〜 +6 dB
  効果: 1つのノブでバランスを調整

Macro 4: "Filter Cutoff"
  マッピング: Chain 2 Filter Cutoff
  範囲: 200 Hz 〜 3000 Hz

Macro 5: "Drive Amount"
  マッピング:
    Chain 1 Saturator Drive: 0 〜 8 dB
    Chain 2 Saturator Drive: 0 〜 15 dB

Macro 6: "Width"
  マッピング:
    Chain 2 Utility Width: 0% 〜 80%
    Chain 3 Utility Width: 0% 〜 150%

Macro 7: "Attack"
  マッピング:
    Chain 1 Amp Envelope Attack: 0 〜 20 ms
    Chain 2 Amp Envelope Attack: 0 〜 20 ms

Macro 8: "Group Compression"
  マッピング: Rack後 Compressor Threshold
  範囲: 0 dB 〜 -20 dB

保存:
  Rack全体を右クリック → Save Preset
  名前: "Layered Bass v1"
  場所: User Library > Presets > Instruments > Instrument Rack
```

### Drum Rackでのキックレイヤリング

```
Drum Rackを使ったキックレイヤリング:

設定手順:
  1. Drum Rack を挿入
  2. C1パッドに Chain List を展開
  3. 3つのChainを作成

Pad C1 内の構成:

Chain 1: "Kick Sub"
  Simpler:
    サンプル: Sine Kick
    Filter: LP 150 Hz
  EQ Eight: HP 30 Hz, LP 150 Hz
  Saturator: Drive 5 dB
  Utility: Width 0%, Gain -1 dB

Chain 2: "Kick Punch"
  Simpler:
    サンプル: 909 Kick
    Filter: BP 200-800 Hz
  EQ Eight: HP 180 Hz, LP 1200 Hz
  Compressor: Ratio 4:1
  Utility: Width 0%, Gain -4 dB

Chain 3: "Kick Click"
  Simpler:
    サンプル: Noise Burst (10 ms)
    Filter: HP 2 kHz
  EQ Eight: HP 2000 Hz
  Transient Shaper: Attack +6 dB
  Utility: Width 0%, Gain -10 dB

Pad After Chain:
  Compressor: Glue, Ratio 2:1
  EQ Eight: Peak 60 Hz +2 dB
  Limiter: -0.3 dB

利点:
  1パッドで3層キック
  他のパッド（Snare、HiHat等）と同居
  パターン作成が容易
  Velocity でレイヤーバランスも制御可能
```

### Chain Selectorの活用

```
Chain Selectorとは:

概念:
  Velocity、Key、または任意の値でChainのON/OFFを制御

Velocity Chain Selector:

設定:
  Chain 1（Sub）: Velocity 1-127（常にOn）
  Chain 2（Punch）: Velocity 40-127（中〜強でOn）
  Chain 3（Click）: Velocity 80-127（強い時のみOn）

効果:
  弱いベロシティ: Sub のみ（柔らかい）
  中程度: Sub + Punch（標準）
  強い: Sub + Punch + Click（フル）

  → ベロシティに応じてレイヤーが変化
  → 演奏に表情が出る
  → よりリアルなドラムサウンド

Key Zone Chain Selector:

設定（ベースの場合）:
  Chain 1（Sub Bass）: C0-B1（低音域）
  Chain 2（Mid Bass）: C0-B3（全域）
  Chain 3（High）: C2-B3（高音域のみ）

効果:
  低い音: Sub + Mid
  中間の音: Sub + Mid + High
  高い音: Mid + High

  → 音域に応じたレイヤー自動切り替え
```

---

## ジャンル別レイヤリング戦略

### Techno のレイヤリング戦略

```
Technoの特徴:
  BPM: 125-140
  音数: 少ない（ミニマル）
  重視: Kick、ベースライン、雰囲気

Kick レイヤリング:
  3層（Sub + Body + Click）
  タイトで短い（100-150 ms）
  Click は控えめ
  Pitch Envelope が重要

ベース レイヤリング:
  2層（Sub + Mid）
  Sub: Sine, 30-100 Hz
  Mid: Saw/Square, 100-800 Hz
  Mono 必須
  サイドチェインで Kick と住み分け

パッド レイヤリング:
  2層（Dark Pad + Texture）
  Dark Pad: フィルターで暗く
  Texture: Noise、Field Recording
  Width: 150-200%
  Volume: 控えめ（-15 dB）

注意点:
  音数が少ないので各音の質が重要
  過剰なレイヤリングは避ける
  空間と余白を大切にする
  Reverb / Delay でレイヤー代わりにする
```

### House のレイヤリング戦略

```
Houseの特徴:
  BPM: 118-130
  音数: 中程度
  重視: グルーヴ、ボーカル、ベースライン

Kick レイヤリング:
  2-3層（Sub + Body ± Click）
  暖かみのある音
  やや長めのテイル（150-200 ms）
  Vintage 感を意識

ベース レイヤリング:
  2層（Sub + Mid）
  Sub: Sine, 40-120 Hz
  Mid: Saw/Triangle, 120-1000 Hz
  ファンキーなベースライン
  サイドチェインは軽め（-3 to -6 dB）

ボーカル レイヤリング:
  2-3層（Lead + Double + Harmony）
  Lead: Center, フルレンジ
  Double: ADTで作成, Width 130%
  Harmony: 3度/5度上, Pan L/R
  Reverb: Plate, 1.5-2.0 s

コード（Stab/Chord）レイヤリング:
  2層（Main Chord + High Stab）
  Main: Saw Pad, 200-3000 Hz
  High Stab: Pluck音, 1-6 kHz
  Main Width: 100%
  High Width: 140%

グルーヴ処理:
  全体にGroove テンプレート適用
  MPC Swing 56-62%
  バウンス感重視
```

### Trance のレイヤリング戦略

```
Tranceの特徴:
  BPM: 130-145
  音数: 多い（壮大）
  重視: メロディ、ブレイクダウン、エネルギー

Kick レイヤリング:
  3層（Sub + Punch + Click）
  パワフルで明瞭
  ドライ（Reverb なし）
  Duration: 120 ms

ベース レイヤリング:
  3層（Sub + Mid + Acid）
  Sub: Sine, 30-100 Hz
  Mid: Saw, 100-1000 Hz
  Acid: TB-303系, 300-3000 Hz（オプション）
  サイドチェイン: 強め（-8 to -12 dB）

リード レイヤリング:
  4層（Main + Octave + Detune + Supersaw）
  Main: Saw/Square, 200-4000 Hz
  Octave: +12 semitones, 800-8000 Hz
  Detune: ±10 cent, Width 140%
  Supersaw: Unison 7-9 voices, Width 160%

  グループ処理:
  Compressor: Ratio 4:1
  EQ: Peak 2-4 kHz +2 dB
  Reverb: Hall, 2.5 s, Dry/Wet 15%
  Delay: 1/4, Feedback 25%, Dry/Wet 12%

パッド レイヤリング:
  3層（Warm + Bright + Air）
  Warm: Saw Pad, LP 1000 Hz, Width 120%
  Bright: Wavetable, 1-5 kHz, Width 150%
  Air: Noise, HP 8 kHz, Width 200%
  Reverb: Hall, 4.0 s

  ブレイクダウン時:
  フィルターを開いていく
  レイヤーを1つずつ追加
  ビルドアップでフル展開
```

### Dubstep / Bass Music のレイヤリング戦略

```
Dubstepの特徴:
  BPM: 140-150（ハーフタイム感）
  音数: 変動（ドロップで爆発）
  重視: ベースデザイン、Kick&Snare、ドロップ

Kick レイヤリング:
  2-3層
  Sub: 30-60 Hz、強め
  Punch: 200-600 Hz
  Click: 控えめ（ベースが高域担当）
  Duration: 80 ms（タイト）

Snare レイヤリング:
  3層（Body + Snap + Reverb）
  Body: 200-800 Hz, ヘビー
  Snap: 2-6 kHz, クリスプ
  Reverb Hit: リバーブをサンプルとして使用
  音量: 大きめ（Kickと同等）

ベース レイヤリング（最重要）:
  3-5層（非常に複雑）

  Sub Layer: 30-80 Hz
    Pure Sine
    Mono 必須
    サイドチェイン: Kick 時のみダック

  Mid Growl Layer: 80-500 Hz
    Serum / Vital で Wavetable モーフィング
    FM Synthesis
    LFO で動きをつける
    Mono

  High Growl Layer: 500-3000 Hz
    Resampling で作成
    Vocoder 処理
    Width: 60-100%

  Texture Layer: 3000-10000 Hz
    Noise ベース
    Granular 処理
    Width: 120%

  Formant Layer（オプション）: 800-5000 Hz
    Vocoder / Formant Filter
    人声的な質感
    Width: 80%

  グループ処理:
  OTT (Over The Top Compressor): 30-50%
  EQ: 各帯域の微調整
  Multiband Compressor: 帯域ごとのダイナミクス制御
  Limiter: -0.1 dB（音圧最大化）

ドロップ構築:
  Intro: Sub のみ
  Build: Sub + Mid
  Drop: 全レイヤーON
  → 劇的なインパクト
```

### Hip Hop / Trap のレイヤリング戦略

```
Hip Hop / Trapの特徴:
  BPM: 60-90（ハーフタイム）/ 130-170（ダブルタイム）
  重視: 808 Bass、Hi-Hat パターン、ボーカル

808 Bass レイヤリング:
  2層（Sub 808 + Distorted 808）

  Sub 808:
    サンプル: TR-808 Kick（ロングディケイ）
    EQ: HP 25 Hz, LP 100 Hz
    Pitch: 曲のキーに合わせる
    Volume: 0 dB
    Width: 0%
    Duration: 2-4 beats（長い）

  Distorted 808:
    同じサンプルをコピー
    Saturator: Drive 15-20 dB
    EQ: HP 80 Hz, LP 3000 Hz
    Volume: -6 dB
    Width: 0-30%

  結果:
  Sub がクラブで響く
  Distortion が小さいスピーカーでも聞こえる

Kick レイヤリング:
  2層（Acoustic Kick + 808 Kick Head）
  808のアタック部分のみ使用
  Acoustic: Body 感
  808 Head: パンチ感

Hi-Hat レイヤリング:
  通常は1層で十分
  ただし Trap の場合:
  Layer 1: Closed HH（メイン）
  Layer 2: Open HH（アクセント）
  Layer 3: Ride / Shaker（テクスチャ）
  各層をMIDIパターンで使い分け

Snare / Clap レイヤリング:
  3層（Acoustic Snare + Clap + Snap）
  Snare: 200-1000 Hz（Body）
  Clap: 500-4000 Hz（Presence）
  Snap: 2-8 kHz（Top）
  Reverb: 長め（0.8-1.5 s）
```

---

## 実践演習（Advanced）

### 演習1: A/B比較レイヤリングテスト

```
目的:
  レイヤリングの効果を客観的に確認する
  レイヤー数の最適解を見つける

手順:

Step 1: 単一音源でベースを作成（Version A）
  Wavetable 1つ
  Saw波、Unison 5、Filter LP 2 kHz
  エフェクト: EQ, Compressor, Saturator
  → できるだけ良い音に仕上げる

Step 2: レイヤリングでベースを作成（Version B）
  Sub Bass: Sine, 30-150 Hz
  Mid Bass: Saw, 200-2 kHz
  グループ: Compressor, EQ
  → 同じ音量に揃える

Step 3: A/B比較
  1. Volume マッチング（LUFS メーターで確認）
  2. Version A 再生 → 印象をメモ
  3. Version B 再生 → 印象をメモ
  4. ミックス全体の中で比較

評価項目:
  □ 音圧感: A vs B
  □ 低域の存在感: A vs B
  □ 中域の明瞭度: A vs B
  □ 全体のバランス: A vs B
  □ ミックスでの座り: A vs B

期待される結果:
  Version B（レイヤリング）が
  音圧、明瞭度、バランスで優れる

所要時間: 60分
```

### 演習2: プロ楽曲のレイヤー分析

```
目的:
  プロの楽曲がどのようにレイヤリングされているか分析する

準備:
  1. 好きな楽曲をAbletonに読み込み
  2. Spectrum アナライザーを挿入
  3. EQ Eightをバイパス可能な状態で配置

分析手順:

Step 1: 全体のスペクトラムを観察
  再生しながらSpectrum確認
  各帯域のエネルギーバランスをメモ

  基準値:
  Sub（30-80 Hz）: -18 to -12 dBFS
  Low（80-200 Hz）: -15 to -8 dBFS
  Mid（200-2000 Hz）: -12 to -6 dBFS
  High（2-8 kHz）: -18 to -10 dBFS
  Air（8-20 kHz）: -24 to -15 dBFS

Step 2: 帯域別アイソレーション
  EQ Eightで特定帯域のみ通す
  各帯域で何が聞こえるか記録

  30-80 Hz: Sub Bass? Kick Sub?
  80-200 Hz: Bass Body? Kick Body?
  200-800 Hz: Bass Mid? Snare Body?
  800-2000 Hz: Lead? Vocal?
  2-5 kHz: Lead High? Vocal Presence?
  5-10 kHz: Hi-Hat? Air?
  10-20 kHz: Cymbal? Reverb Tail?

Step 3: レイヤー数の推測
  各楽器要素について:
  - 何層でレイヤリングされているか推測
  - 各層の周波数帯域を推測
  - 音量バランスを推測

Step 4: 再現
  分析結果を元に同様のレイヤリングを試す
  オリジナルと比較

所要時間: 90分
```

### 演習3: ジャンル横断レイヤリング

```
目的:
  1つのレイヤリング構成を異なるジャンルに適用し
  ジャンルごとの調整力を身につける

課題:
  同じ3層ベース構成（Sub + Mid + High）を
  4つのジャンルに対応させる

Step 1: 基本3層ベース作成
  Sub: Sine, 30-150 Hz, Mono
  Mid: Saw, 200-2000 Hz, Mono
  High: Noise, 2-8 kHz, Mono
  → 「素材」として保存

Step 2: House版（BPM 124）
  Sub: Volume -2 dB, Duration 長め
  Mid: Saturator 軽め, Filter 動かす
  High: Volume -15 dB（控えめ）
  グループ: Sidechain 軽め（-4 dB）

Step 3: Techno版（BPM 132）
  Sub: Volume -3 dB, Duration 短め
  Mid: Saturator 強め, Filter 自動化
  High: Volume -12 dB
  グループ: Sidechain 中程度（-6 dB）

Step 4: Dubstep版（BPM 140）
  Sub: Volume 0 dB, Duration 長め
  Mid: Heavy Saturator + LFO modulation
  High: Volume -8 dB, Width 100%
  グループ: OTT 40%, Sidechain 強め（-10 dB）

Step 5: Trap版（BPM 70 ハーフタイム）
  Sub: 808スタイル, Duration 非常に長い
  Mid: Distortion 強め
  High: Volume -18 dB（ほぼ不要）
  グループ: Sidechain なし、Glide あり

比較検証:
  4バージョンを順番に再生
  各ジャンルの特徴がレイヤーバランスで表現できているか確認

所要時間: 120分
```

### 演習4: レイヤリングのトラブルシューティング

```
目的:
  意図的に問題のあるレイヤリングを作り
  問題発見→修正のプロセスを体験する

課題1: 位相キャンセル問題

準備:
  Sub Bass + Mid Bassを作成
  Mid BassのSample Startを10 ms ズラす（意図的に位相ズレ作成）

修正手順:
  1. 問題を聴覚で確認（音が薄い）
  2. Spectrum で確認（低域が減少）
  3. 位相反転テストで確認
  4. Sample Offset で修正
  5. 修正後のA/B比較

課題2: マスキング問題

準備:
  Sub Bass: HP 30 Hz, LP なし（全帯域出力）
  Mid Bass: HP なし, LP 3000 Hz（低域も出力）
  → 200-500 Hz帯域が重複

修正手順:
  1. 問題を聴覚で確認（モコモコ、濁り）
  2. 各層をSoloでSpectrum確認
  3. 重複帯域を特定
  4. Sub Bass: LP 180 Hz追加
  5. Mid Bass: HP 200 Hz追加
  6. 修正後のA/B比較

課題3: 音量バランス問題

準備:
  Sub: -8 dB（弱すぎ）
  Mid: -2 dB（強すぎ）
  Click: -3 dB（強すぎ）
  → 高域偏重でシャカシャカ

修正手順:
  1. 問題を確認（低域不足、シャカシャカ）
  2. RMS メーターで各層の音量確認
  3. 黄金比に修正: Sub 60%, Mid 30%, Click 10%
  4. 修正後のA/B比較

所要時間: 各課題30分、合計90分
```

### 演習5: 完全な楽曲でのレイヤリング実践

```
目的:
  楽曲全体を通してレイヤリングを適用し
  総合的なスキルを確認する

楽曲構成（8分間、128 BPM、Key: Am）:

Intro（0:00-1:00）:
  パッド: 2層（Warm + Air）
  Hi-Hat: 1層
  FX: Riser（1層）

Build（1:00-1:30）:
  + Kick: 3層（Sub + Punch + Click）
  + Snare: 2層（Body + Snap）
  + Bass: Sub層のみ

Drop 1（1:30-3:00）:
  Kick: 3層
  Snare: 2層
  Bass: 2層（Sub + Mid）
  Lead: 3層（Main + Octave + Detune）
  Pad: 2層
  Hi-Hat: 1層

Break（3:00-4:00）:
  パッド: 3層（Warm + Bright + Air）
  Vocal: 2層（Lead + Double）
  FX: Downlifter（1層）

Build 2（4:00-4:30）:
  全要素を徐々に追加
  フィルター開放
  Riser

Drop 2（4:30-6:00）:
  全要素フル
  Lead: 4層（+ Supersaw追加）
  Bass: 3層（+ High Click追加）
  Vocal: 3層（+ Harmony追加）

Breakdown（6:00-7:00）:
  要素を徐々に減らす
  パッド + ボーカルのみ

Outro（7:00-8:00）:
  最小構成に戻る
  パッド + FX

レイヤー総数:
  最小（Intro）: 4層
  最大（Drop 2）: 20層以上

チェックリスト:
  □ 全てのレイヤーに周波数住み分け設定済み
  □ 各グループにバスコンプ適用済み
  □ 位相チェック完了
  □ ステレオ配置設定済み
  □ サイドチェイン設定済み
  □ CPU負荷 50%以内
  □ マスターピーク: -1 dBFS以下

所要時間: 4-6時間
```

---

## レイヤリングのワークフロー最適化

### テンプレートの作成と活用

```
推奨テンプレート構成:

1. Bass Template:
   Track 1: Sub Bass（Wavetable + EQ + Utility）
   Track 2: Mid Bass（Wavetable + EQ + Saturator + Utility）
   Group: Bass Bus（Compressor + EQ + Limiter）

2. Lead Template:
   Track 1: Main Lead（Wavetable + EQ + Chorus + Reverb）
   Track 2: Octave Lead（Wavetable + EQ + Chorus）
   Track 3: Detune Lead（Wavetable + EQ + Delay）
   Group: Lead Bus（Compressor + EQ + Reverb + Limiter）

3. Kick Template:
   Track 1: Sub Kick（Simpler + EQ + Saturator）
   Track 2: Punch Kick（Simpler + EQ + Compressor）
   Track 3: Click Kick（Simpler + EQ + Transient Shaper）
   Group: Kick Bus（Glue Comp + EQ + Limiter）

4. Snare Template:
   Track 1: Snare Body（Simpler + EQ + Compressor）
   Track 2: Snare Snap（Simpler + EQ + Transient Shaper）
   Group: Snare Bus（Compressor + EQ + Reverb）

保存場所:
  User Library > Templates > Layering

使い方:
  新規楽曲作成時にテンプレートをドラッグ&ドロップ
  シンセ/サンプルのみ差し替え
  EQ値を微調整
  → 大幅な時短
```

### CPU負荷管理

```
CPU負荷の目安:

レイヤリングなし: 20-30%
基本レイヤリング（2-3層 × 4要素）: 40-50%
フルレイヤリング（3-4層 × 8要素）: 60-80%
限界: 80%超え → 対策必要

対策1: Freeze Track
  完成したレイヤーグループをFreeze
  CPU削減: 80-90%
  制限: エフェクト変更不可（Unfreeze必要）

対策2: Resample（Flatten）
  レイヤーグループをAudioにレンダリング
  右クリック → Flatten
  CPU削減: 100%（実質0%に）
  制限: 完全に編集不可

対策3: バッファサイズ調整
  作業中: 256-512 samples（低レイテンシー）
  ミックス中: 1024-2048 samples（安定性重視）

対策4: エフェクトの最適化
  各レイヤーの個別Reverb → Group Reverbに統合
  各レイヤーの個別Delay → Send/Returnに統合
  → エフェクトインスタンス削減

対策5: 不要レイヤーの削除
  ソロで聞いて効果がわからないレイヤーは削除
  「聞こえないレイヤー」は不要
  3層で十分な場合、4層目は不要
```

### レイヤリングのチェックリスト（最終版）

```
制作時チェックリスト:

□ Phase 1: 設計
  □ 何層にするか決定（2-4層）
  □ 各層の周波数帯域を決定
  □ 各層の役割を明確化
  □ 音量バランスの目標値設定

□ Phase 2: 制作
  □ 各層のサウンドデザイン完了
  □ EQで帯域分離完了
  □ 音量バランス調整完了
  □ ステレオ配置設定完了

□ Phase 3: グループ処理
  □ グループ化（Cmd+G）完了
  □ バスコンプレッション設定完了
  □ グループEQ調整完了
  □ リミッター設定完了

□ Phase 4: 品質チェック
  □ 位相チェック完了（位相反転テスト）
  □ Mono互換性チェック完了
  □ スペクトラム確認完了
  □ A/B比較完了（レイヤリングあり/なし）

□ Phase 5: ミックス統合
  □ 他の楽器とのマスキングチェック
  □ サイドチェイン設定完了
  □ 全体バランスでの確認完了
  □ リファレンス楽曲との比較完了

□ Phase 6: 最適化
  □ CPU負荷確認（80%以下）
  □ 不要レイヤーの削除
  □ 必要に応じてFreeze/Resample
  □ テンプレート保存
```

---

## レイヤリングの哲学と心構え

```
プロプロデューサーの考え方:

1. 「Less is More」の原則
   - レイヤーは多ければ良いわけではない
   - 2層で十分な場合、3層にしない
   - 各層に明確な役割があること
   - 「このレイヤーを消して変化がわからなければ、不要」

2. 「周波数の不動産」理論
   - 周波数帯域は有限のスペース
   - 各楽器が「場所」を持つ
   - 同じ場所に2つの音は存在できない
   - レイヤリングとは「1つの楽器が複数の場所を占める」こと

3. 「目的ファースト」の原則
   - まず「どんな音が欲しいか」を明確にする
   - 次に「1音で実現できるか」試す
   - 1音で不十分な場合のみレイヤリング
   - テクニックに振り回されない

4. 「耳で判断」の重要性
   - スペクトラムは参考情報
   - 最終判断は耳で行う
   - A/B比較を必ず行う
   - 「良い音」かどうかは数値ではなく感覚

5. リファレンスの重要性
   - プロの楽曲をリファレンスとして常に参照
   - 自分の音とプロの音を比較
   - 差を分析し、改善する
   - リファレンスは3-5曲用意
```

---

## 上級者向け: リサンプリングによるレイヤー統合

```
リサンプリング（Resampling）とは:

概念:
  レイヤリングされた音を一度録音（Resample）し
  その録音を新たな素材として再加工する

手順:

1. レイヤリングされたグループを作成
   Sub Bass + Mid Bass + High Click
   → グループ処理完了

2. Resample
   新規Audioトラック作成
   Input: "Layered Bass" Group
   Record: 数小節分録音

3. 録音されたAudioを新たなSimpler/Samplerに
   → 1つの「サンプル」として扱う

4. さらにレイヤリング
   Resampled Bass（元の3層が1つに）
   + 新しい Layer（追加の質感）
   → 「レイヤーのレイヤー」

利点:
  CPU大幅削減
  新しい加工の自由度
  Granular処理が可能に
  さらなるレイヤリングの余地

Dubstepでの活用:
  1. ベースを3層でレイヤリング
  2. Resample
  3. Resampled音をWavetableに読み込み
  4. 新たなWavetableとしてモーフィング
  5. さらに別の層とレイヤリング
  → 複雑で個性的なベースサウンド

注意:
  元のレイヤーは削除せず保存しておく
  Resample後に微調整したくなる場合がある
  プロジェクト管理を整理して行う
```

---

## レイヤリング関連プラグインとツール

```
推奨プラグイン:

1. 分析ツール
   SPAN（Voxengo）: 無料スペクトラムアナライザー
   Ozone Insight（iZotope）: 総合メータリング
   Correlometer: 位相相関メーター
   MSED（Voxengo）: Mid/Side確認

2. EQプラグイン
   FabFilter Pro-Q 3: 最高品質のEQ
     - Dynamic EQ 機能
     - Linear Phase モード
     - スペクトラム表示付き
   TDR Nova: 無料のダイナミックEQ

3. コンプレッサー
   FabFilter Pro-C 2: 多機能コンプ
   SSL Bus Compressor（Waves）: グルーコンプの定番
   OTT（Xfer）: 無料マルチバンドコンプ（Dubstep必須）

4. サチュレーター
   Decapitator（Soundtoys）: アナログ歪み
   Saturn 2（FabFilter）: マルチバンドサチュレーション
   Trash 2（iZotope）: エクストリーム歪み

5. ユーティリティ
   Utility（Ableton内蔵）: Width、Phase、Gain
   InPhase（Waves）: 位相調整専用
   bx_solo（Brainworx）: Mid/Side ソロ

6. レイヤリング専用
   Layers（Orchestral Tools）: オーケストラレイヤリング
   Serum（Xfer）: Wavetable + Noise OSC でレイヤリングに最適
   Vital（Matt Tytel）: 無料のSerum代替

選定基準:
  予算に合わせて選択
  無料プラグインでも十分なクオリティ
  まずはAbleton内蔵ツールをマスターしてから検討
```

---

## 最終まとめ: レイヤリングマスターへの道

レイヤリングは音楽制作における最も重要なテクニックの1つです。この章で学んだ内容を体系的に振り返ります。

```
習得ロードマップ:

Level 1: 初級（1-2週間）
  □ 2層ベースレイヤリング
  □ 周波数帯域の住み分け理解
  □ EQ設定の基本
  □ Mono/Stereoの基本

Level 2: 中級（3-4週間）
  □ 3層キックレイヤリング
  □ 3層リードレイヤリング
  □ グループ処理（バスコンプ）
  □ 位相確認と修正
  □ サイドチェイン連携

Level 3: 上級（1-2ヶ月）
  □ ジャンル別レイヤリング戦略
  □ パラレルプロセッシング併用
  □ マルチバンドレイヤリング
  □ ボーカルレイヤリング
  □ リサンプリング技法

Level 4: マスター（3ヶ月以上）
  □ 楽曲全体のレイヤリング設計
  □ CPU最適化
  □ テンプレート構築
  □ 独自のレイヤリングスタイル確立
  □ プロレベルの音質達成

最終目標:
  「レイヤリングを意識しなくても自然にできる」状態
  テクニックが体に染み込んでいること
  各ジャンルで即座に最適なレイヤリングを構築できること
```

**次のステップ:** [Modulation Techniques（モジュレーション技術）](./modulation-techniques.md) へ進む

---

## 次に読むべきガイド

- [Modulation Techniques（モジュレーション技術）](./modulation-techniques.md) - 次のトピックへ進む

---

## 参考文献

- [MDN Web Docs](https://developer.mozilla.org/) - Web技術のリファレンス
- [Wikipedia](https://ja.wikipedia.org/) - 技術概念の概要
