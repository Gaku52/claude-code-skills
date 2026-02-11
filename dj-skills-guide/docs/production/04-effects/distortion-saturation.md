# Distortion・Saturation

歪み系エフェクトで温かみと存在感を付加します。Saturatorを中心に、倍音生成と音色強化を完全マスター。

## この章で学ぶこと

- Saturator完全ガイド
- Curve Type使い分け（Analog・Digital・Warm等）
- Drive・Dry/Wet調整
- Overdrive・Distortion
- Parallel Saturation
- トラック別設定
- 倍音理論基礎

---

## なぜSaturationが重要なのか

**温かみと存在感:**

```
Saturationなし:

特徴:
クリーン
冷たい
薄い

Saturationあり:

特徴:
温かい
太い
存在感

使用頻度:

Saturator: 50%
Bass・Kick・Vocal

Overdrive: 15%
特定用途

プロの真実:

「アナログ的音」=
わずかなSaturation

デジタルっぽい:
完全クリーン

理由:

倍音:
奇数倍音・偶数倍音追加

音圧:
ピーク制御

一体感:
全体まとまる

結果:
プロの温かく太い音
```

---

## Saturator 完全ガイド

**最重要歪み系:**

### インターフェイス

```
┌─────────────────────────────────┐
│  Saturator                      │
├─────────────────────────────────┤
│  Curve Type: [Analog Clip ▼]    │
│                                 │
│  Drive: 5.0 dB                  │
│  Base: 0 Hz                     │
│                                 │
│  Dry/Wet: 100%                  │
│                                 │
│  Output: -5.0 dB                │
│                                 │
│  [Depth: 100%] [Color: Off]     │
│                                 │
│  波形表示 (視覚的)               │
└─────────────────────────────────┘

重要パラメーター:
- Curve Type (歪みタイプ)
- Drive (歪み量)
- Dry/Wet (ミックス)
```

### Curve Type詳細

```
Analog Clip (推奨):

特徴:
アナログ機器的
温かい
偶数倍音

用途:
Bass・Kick・Vocal
全般に使用

倍音:
2倍・4倍・6倍
(偶数倍音)

効果:
温かみ
厚み

推奨Drive:
3-8 dB

Digital Clip:

特徴:
デジタル的
ハード
奇数倍音

用途:
実験的
aggressive

倍音:
3倍・5倍・7倍
(奇数倍音)

効果:
明るい
鋭い

推奨Drive:
2-5 dB
強烈

Warm (Tube):

特徴:
真空管的
最も温かい
滑らか

用途:
Vocal・Pad
ジャズ・ソウル

倍音:
偶数倍音多め
低次倍音

効果:
ビンテージ
温かい

推奨Drive:
5-12 dB
多めOK

Sine Fold:

特徴:
折り返し
実験的

用途:
特殊効果
リード

効果:
金属的
複雑

推奨Drive:
8-15 dB

Soft Sine:

特徴:
柔らかい
自然

用途:
Master
全体的処理

効果:
わずかな厚み

推奨Drive:
2-5 dB
控えめ

A Bit Warmer:

特徴:
わずかに温かく
自然

用途:
Master
Bus処理

推奨Drive:
3-6 dB
```

### パラメーター詳細

```
1. Drive:

機能:
歪み量

単位:
dB

設定:

軽い (2-5 dB):
わずかな厚み
自然

中間 (5-10 dB):
明確な変化
温かみ

強い (10-20 dB):
強烈
実験的

推奨:

Bass: 5-8 dB
Kick: 3-5 dB
Vocal: 3-6 dB
Master: 2-4 dB

ルール:
「気づかない程度」
過剰注意

2. Base:

機能:
周波数依存処理

単位:
Hz

設定:

0 Hz (デフォルト):
全帯域

200 Hz:
200 Hz以下のみ処理

400 Hz:
400 Hz以下のみ

用途:

低域のみ太く:
Base: 200-300 Hz

全体:
Base: 0 Hz

推奨:

Bass: 0 Hz (全体)
Kick: 0 Hz
Drum Bus: 150 Hz (低域)

3. Dry/Wet:

機能:
ミックス比率

設定:

100%:
完全に歪み

50%:
半々 (Parallel)

20%:
わずかにブレンド

推奨:

Individual Track:
50-100%

Parallel処理:
20-50%

4. Output:

機能:
出力レベル

理由:
Drive上げる
→ 音量大きく
→ Output下げて補正

推奨:
Driveと同量逆方向
Drive +5 dB → Output -5 dB

5. Depth:

機能:
歪み深さ

設定:
通常100%

6. Color:

機能:
DC Offset追加

用途:
実験的
非対称歪み

推奨:
通常Off
```

---

## Saturator実践例

**トラック別設定:**

### Bass (Techno/House)

```
目標:
太い、温かい、存在感

設定:

Curve: Analog Clip
Drive: 6.0 dB
Base: 0 Hz (全帯域)
Dry/Wet: 100%
Output: -6.0 dB

Chain:

1. EQ Eight:
   High Pass 40 Hz

2. Saturator:
   上記設定

3. EQ Eight (Post):
   Peak -2 dB @ 300 Hz
   濁り除去

結果:
太く、温かいBass

A/B比較:
必須
低域確認
```

### Kick

```
目標:
パンチ、温かみ

設定:

Curve: Analog Clip
Drive: 4.0 dB (控えめ)
Base: 0 Hz
Dry/Wet: 100%
Output: -4.0 dB

Chain:

EQ → Compressor → Saturator

理由:
コンプ後にSaturator
安定した歪み

結果:
わずかな温かみ
自然

注意:
過剰: アタック潰れる
```

### Vocal

```
目標:
温かみ、存在感

設定:

Curve: Warm (Tube)
Drive: 5.0 dB
Base: 0 Hz
Dry/Wet: 70% (Parallel的)
Output: -5.0 dB

Chain:

1. EQ Eight:
   High Pass 80 Hz
   Peak +3 dB @ 3 kHz

2. Compressor:
   Ratio 3:1

3. Saturator:
   上記設定

4. De-esser (EQ):
   -3 dB @ 7 kHz

結果:
温かく、前に出るVocal
```

### Drum Bus

```
目標:
全ドラムまとめる

設定:

Curve: A Bit Warmer
Drive: 3.0 dB (軽め)
Base: 150 Hz (低域のみ)
Dry/Wet: 100%
Output: -3.0 dB

Chain:

Glue Compressor → Saturator

理由:
わずかな一体感

結果:
自然な厚み
```

### Master

```
目標:
全体的温かみ

設定:

Curve: Soft Sine
Drive: 2.5 dB (最も軽い)
Base: 0 Hz
Dry/Wet: 50% (Parallel)
Output: -2.5 dB

Chain:

EQ → Saturator → Limiter

注意:
Master: 最も控えめ
過剰: 崩壊

結果:
わずかな厚み
```

---

## Parallel Saturation

**より自然な歪み:**

### 概念

```
通常Saturation:

100%歪み
強烈

Parallel:

Dry: 50%
Wet: 50%
ブレンド

メリット:

自然:
元音残る

太い:
歪み追加

柔軟:
調整簡単
```

### 設定方法

**方法1: Dry/Wet使用:**

```
1. Saturator挿入

2. Dry/Wet: 30-50%

3. Drive: 高め (8-12 dB)

4. Output調整

メリット:
簡単
1つのデバイス

推奨:
初心者向け
```

**方法2: Return Track:**

```
1. Return E作成

2. Saturator挿入:
   Curve: Analog Clip
   Drive: 10 dB (強め)
   Dry/Wet: 100%
   Output: -10 dB

3. Send:
   Bass: 20-30%
   Drums: 15-20%

メリット:
複数トラック共有
CPU効率

推奨:
上級者
```

**方法3: Audio Effect Rack:**

```
1. Audio Effect Rack作成

2. Chain 1 (Dry):
   何もなし

3. Chain 2 (Wet):
   Saturator
   Drive: 12 dB

4. Chain Mix:
   Chain 1: 70%
   Chain 2: 30%

メリット:
最も柔軟
Macro作成可

推奨:
プロ手法
```

---

## Overdrive・Distortion

**より強烈な歪み:**

### Overdrive

```
┌─────────────────────────────────┐
│  Overdrive                      │
├─────────────────────────────────┤
│  Drive: 30%                     │
│  Tone: 60%                      │
│  Dynamics: 40%                  │
│                                 │
│  Dry/Wet: 50%                   │
└─────────────────────────────────┘

特徴:
ギター的
中域強調

用途:
Lead・Synth
実験的

パラメーター:

Drive:
歪み量
30-60%

Tone:
明るさ
40-70%

Dynamics:
元の音の保持
30-50%

推奨設定:

Lead:
Drive: 40%
Tone: 60%
Dry/Wet: 60%

結果:
明るく、前に出る
```

### Distortion

```
┌─────────────────────────────────┐
│  Distortion (Pedal)             │
├─────────────────────────────────┤
│  Drive: 50%                     │
│  Tone: 50%                      │
│  Output: -6 dB                  │
│                                 │
│  Dry/Wet: 40%                   │
└─────────────────────────────────┘

特徴:
ハード
aggressive

用途:
Dubstep・Drum & Bass
実験的リード

推奨:

Dubstep Bass:
Drive: 60-80%
Tone: 40% (暗め)
Dry/Wet: 70%

Post処理:
EQ Eight:
High Pass 200 Hz
低域クリーンに
```

---

## Erosion

**デジタル破壊:**

### 特徴

```
┌─────────────────────────────────┐
│  Erosion                        │
├─────────────────────────────────┤
│  Frequency: 2000 Hz             │
│  Mode: [Sine Modulation ▼]      │
│  Width: 50%                     │
│  Amount: 30%                    │
└─────────────────────────────────┘

機能:
ビット深度削減
サンプルレート削減

効果:
Lo-Fi
デジタルグリッチ

用途:
実験的
特殊効果
Hi-Hat・Perc

推奨設定:

Hi-Hat Lo-Fi:
Frequency: 4000 Hz
Mode: Noise
Width: 40%
Amount: 25%

結果:
Lo-Fi質感
```

---

## Vinyl Distortion

**レコード質感:**

### 特徴

```
┌─────────────────────────────────┐
│  Vinyl Distortion               │
├─────────────────────────────────┤
│  Tracing Model: [Pinch ▼]       │
│  Drive: 5.0 dB                  │
│                                 │
│  Crackle Volume: 20%            │
│  Crackle Density: 30%           │
└─────────────────────────────────┘

機能:
レコード的歪み
パチパチノイズ

用途:
Lo-Fi House
ビンテージ
Intro・Outro

推奨設定:

Lo-Fi Intro:
Tracing: Pinch
Drive: 8 dB
Crackle: 40%

結果:
レコード質感
ノスタルジック
```

---

## 倍音理論基礎

**なぜ温かくなる？**

### 倍音の種類

```
基音 (Fundamental):

周波数: 100 Hz
最も強い

2倍音 (偶数):

周波数: 200 Hz
温かい
調和的

3倍音 (奇数):

周波数: 300 Hz
明るい
鋭い

4倍音 (偶数):

周波数: 400 Hz
温かい

5倍音 (奇数):

周波数: 500 Hz
明るい

真実:

偶数倍音:
温かい
アナログ的

奇数倍音:
明るい
デジタル的

Saturation:
倍音追加
→ 太く、温かく
```

### Curve Typeと倍音

```
Analog Clip:

倍音: 偶数多い
効果: 温かい

Digital Clip:

倍音: 奇数多い
効果: 明るい、鋭い

Warm (Tube):

倍音: 低次偶数多い
効果: 最も温かい

Soft Sine:

倍音: わずか
効果: 自然

選択:

温かく: Analog・Warm
明るく: Digital
自然: Soft Sine
```

---

## よくある失敗

### 1. Driveかけすぎ

```
問題:
音が潰れる
濁る

原因:
Drive: 15 dB+

解決:

Drive: 3-8 dB
控えめ

A/B比較:
必須

ルール:
「わずかに聴こえる」
```

### 2. Outputバランス無視

```
問題:
音量大きくなる
ミックス崩壊

原因:
Output調整なし

解決:

Drive +5 dB
→ Output -5 dB

ゲインマッチング:
必須
```

### 3. 全てにSaturation

```
問題:
過剰
濁る

原因:
全トラック歪み

解決:

必要なトラックのみ:
Bass・Kick・Vocal
Master

他:
クリーン

推奨:
50%のトラック
```

### 4. Masterで強烈

```
問題:
全体崩壊

原因:
Master Drive高すぎ

解決:

Master:
Drive: 2-4 dB
最も控えめ

Curve: Soft Sine
自然

Dry/Wet: 50%
Parallel
```

---

## 実践ワークフロー

**30分練習:**

### Week 1: Saturator基礎

```
Day 1 (10分):

1. Bassトラック
2. Saturator挿入
3. Curve: Analog Clip
4. Drive: 5 dB
5. Output: -5 dB
6. A/B比較

Day 2 (15分):

1. Kick
2. Drive: 3 dB (軽め)
3. A/B比較
   アタック確認

Day 3-4:

Vocal
Curve: Warm
Drive: 5 dB

Day 5-7:

全トラック最適化
バランス調整
```

### Week 2: Parallel Saturation

```
Day 1-2:

Return Track方式

Day 3-4:

Audio Effect Rack
Chain構築

Day 5-7:

比較
最適な方法選択
```

---

## まとめ

### Saturator

```
□ Curve: Analog Clip (標準)
□ Drive: 3-8 dB (控えめ)
□ Output: Drive分マイナス
□ A/B比較必須
□ わずかな変化が正解
```

### Curve Type

```
温かい: Analog Clip・Warm
明るい: Digital Clip
自然: Soft Sine
実験的: Sine Fold
```

### 使用箇所

```
必須:
Bass・Kick・Vocal

推奨:
Drums・Lead

控えめ:
Pad

最軽量:
Master
```

### 重要原則

```
□ Less is More
□ 偶数倍音 = 温かい
□ 奇数倍音 = 明るい
□ Parallel処理活用
□ Output調整必須
```

---

**次は:** [Modulation Effects](./modulation-effects.md) - Chorus・Flanger・Phaserで動きを付ける
