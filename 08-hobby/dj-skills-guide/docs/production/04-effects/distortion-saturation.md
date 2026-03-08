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

## ディストーション/サチュレーションの種類と原理

**アナログ回路の違いを理解する:**

### チューブ（真空管）サチュレーション

```
原理:

真空管内の電子流:
非線形増幅
グリッド電圧超過
→ ソフトクリッピング

倍音特性:

偶数倍音中心:
2次、4次、6次
温かい、滑らか

圧縮特性:
自然なコンプレッション
ダイナミクス制御

周波数特性:

高域:
わずかにロールオフ
耳に優しい

低域:
トランス結合
トランジェント保持

音色:

特徴:
最も温かい
ビンテージ
滑らか

用途:

Vocal: 最適
ジャズ・ソウル
アナログ再現

Ableton再現:

Saturator:
Curve: Warm
Drive: 5-12 dB

外部プラグイン:
UAD Pultec
Waves Abbey Road
iZotope Exciter

設定例:

Vocal Tube:
Curve: Warm
Drive: 8 dB
Base: 0 Hz
Dry/Wet: 80%
Output: -8 dB

結果:
ビンテージ質感
温かいVocal

真空管の種類:

12AX7 (ECC83):
最も一般的
高ゲイン
鋭い

12AU7:
低ゲイン
滑らか
温かい

6L6:
パワー管
厚み
低域豊か

音響心理:

理由:
偶数倍音 =
オクターブ関係
音楽的
```

### テープサチュレーション

```
原理:

磁気テープ飽和:
磁性体の限界
記録レベル超過
→ 自然な圧縮

倍音特性:

偶数倍音:
2次中心
温かい

高域:
わずかに減衰
Tape Head特性

低域:
トランス飽和
厚み

コンプレッション:

特徴:
最も自然
ピーク制御
透明

時定数:
速い
トランジェント保持

周波数特性:

高域ロールオフ:
10 kHz以上
自然
耳に優しい

低域厚み:
60-200 Hz
トランス特性

ヒス/ノイズ:

特徴:
高域ノイズ
ビンテージ質感

制御:
Noise Gate
De-esser

音色:

特徴:
温かい
ビンテージ
滑らか

用途:

全般:
Master処理
Drum Bus
Bass

Hip-Hop:
SP-1200再現
Lo-Fi

Ableton再現:

方法1 Saturator:
Curve: Warm
Drive: 3-6 dB
Base: 0 Hz

方法2 外部:
UAD Studer A800
Waves J37
Slate Virtual Tape

設定例:

Drum Bus Tape:
Curve: Warm
Drive: 5 dB
Base: 200 Hz (低域のみ)
Dry/Wet: 100%
Output: -5 dB

+ EQ Eight:
High Shelf -1 dB @ 10 kHz
(テープロールオフ再現)

結果:
ビンテージドラム質感

テープ速度:

15 ips (高速):
高域伸びる
明るい

7.5 ips (標準):
バランス
ビンテージ

3.75 ips (低速):
高域減衰
Lo-Fi
```

### トランジスタ/ソリッドステート

```
原理:

トランジスタ飽和:
pn接合非線形
急激なクリッピング

倍音特性:

奇数倍音:
3次、5次、7次
明るい
エッジ

クリッピング:
ハードクリッピング
aggressive

周波数特性:

全帯域:
フラット
透明

高域:
クリア
鋭い

低域:
タイト
パンチ

音色:

特徴:
明るい
パンチ
aggressive

用途:

Kick: パンチ
Bass: エッジ
Rock: ギター

Ableton再現:

Saturator:
Curve: Digital Clip
Drive: 2-5 dB

Overdrive:
ソリッドステート的

設定例:

Kick Punch:
Curve: Digital Clip
Drive: 3 dB
Base: 0 Hz
Dry/Wet: 60% (Parallel)
Output: -3 dB

結果:
パンチ、エッジ

トランジスタ種類:

BJT (バイポーラ):
一般的
音楽的

FET:
真空管的
滑らか

ダイオード:
最もハード
aggressive
```

### コンソール/プリアンプ

```
原理:

入力段飽和:
トランス
アンプ段
出力段

複合的歪み:
複数段階
複雑な倍音

倍音特性:

偶数・奇数混合:
バランス
音楽的

トランス倍音:
低次偶数
厚み

周波数特性:

トランス:
低域厚み
高域滑らか

アンプ段:
全帯域
透明

音色:

特徴:
プロフェッショナル
クリア
厚み

用途:

全般:
万能
プロ品質

Master:
最適

有名コンソール:

Neve 1073:
温かい
厚み
偶数倍音

API 2500:
パンチ
明るい

SSL:
クリア
モダン

Ableton再現:

Saturator:
Curve: Analog Clip
Drive: 3-5 dB
Dry/Wet: 100%

外部:
Slate Virtual Console
UAD Neve/API

設定例:

Master Console:
Curve: Analog Clip
Drive: 3 dB
Base: 0 Hz
Dry/Wet: 100%
Output: -3 dB

結果:
プロフェッショナル質感
```

---

## Ableton内蔵エフェクト詳細解説

**全歪み系エフェクトマスター:**

### Saturator完全マスター

```
高度なパラメーター:

WS (Waveshaper):

機能:
波形整形
カスタマイズ可能

エンベロープ表示:
視覚的フィードバック
入出力関係

Curve編集:

方法:
グラフィックエディタ
ポイント追加

カスタム歪み:
独自Curve作成

高度なテクニック:

1. Curve調整:
   Default Curveから
   微調整

2. Base周波数活用:
   低域のみ処理
   150-300 Hz

3. Dry/Wet自動化:
   セクション別
   Intro: 30%
   Drop: 100%

4. Color活用:
   DC Offset
   非対称歪み
   実験的

M/S処理:

方法:
Utility (Width 0%) → Saturator → Utility (Width 200%)

Mid処理:
厚み

Side処理:
ステレオ幅

高度なChain:

EQ → Compressor → Saturator → EQ

理由:
1. EQ: 不要周波数除去
2. Comp: ダイナミクス制御
3. Sat: 倍音付加
4. EQ: 歪み調整

自動化ポイント:

Drive:
ビルドアップ増加

Dry/Wet:
セクション切替

Curve Type:
展開変化
```

### Pedal（ギターペダル）

```
┌─────────────────────────────────┐
│  Pedal                          │
├─────────────────────────────────┤
│  Model: [Overdrive ▼]           │
│                                 │
│  Gain: 50%                      │
│  Bass: 50%                      │
│  Mid: 60%                       │
│  Treble: 55%                    │
│                                 │
│  Output: 70%                    │
│  Dry: 0%                        │
└─────────────────────────────────┘

モデル一覧:

Overdrive:
特徴: 温かい
用途: Lead, Vocal

Distortion:
特徴: aggressive
用途: Rock Bass

Fuzz:
特徴: 極端
用途: 実験的

パラメーター:

Gain:
歪み量
30-70%

Bass:
低域
40-60%

Mid:
中域強調
50-70%

Treble:
高域
40-60%

Sub:
超低域
0-30%

Output:
出力レベル

Dry:
元音ミックス
0-30%

実践設定:

Synth Lead Overdrive:
Model: Overdrive
Gain: 55%
Bass: 45%
Mid: 65%
Treble: 50%
Output: 65%
Dry: 0%

結果:
温かく前に出る

Bass Fuzz (実験):
Model: Fuzz
Gain: 70%
Bass: 60%
Mid: 50%
Treble: 30%
Sub: 40%
Output: 50%
Dry: 20% (元音ブレンド)

Chain:
EQ High Pass 100 Hz → Pedal → EQ

理由:
低域濁り防止
```

### Amp（ギターアンプ）

```
┌─────────────────────────────────┐
│  Amp                            │
├─────────────────────────────────┤
│  Amp Type: [Clean ▼]            │
│                                 │
│  Gain: 40%                      │
│  Bass: 50%                      │
│  Middle: 60%                    │
│  Treble: 55%                    │
│  Presence: 45%                  │
│                                 │
│  Output: 70%                    │
│  Dry/Wet: 80%                   │
│                                 │
│  Cabinet: On                    │
│  Dual Mono: Off                 │
└─────────────────────────────────┘

Amp Type:

Clean:
特徴: 透明
わずかな厚み

Boost:
特徴: 中域強調
存在感

Blues:
特徴: 温かい
ビンテージ

Rock:
特徴: パンチ
aggressive

Lead:
特徴: ソロ向け
前に出る

Heavy:
特徴: 最も歪み
メタル

Bass:
特徴: Bass専用
低域保持

パラメーター詳細:

Gain:
歪み量
Clean: 20-40%
Rock: 50-70%

Bass:
低域EQ
40-60%

Middle:
中域EQ
50-70%
存在感

Treble:
高域EQ
40-60%

Presence:
超高域
40-60%
明るさ

Cabinet:

機能:
スピーカーシミュレーション

On: リアル
Off: ダイレクト

Dual Mono:

機能:
Left/Right独立処理

用途:
ステレオ幅

実践設定:

Synth Bass Rock:
Type: Rock
Gain: 60%
Bass: 55%
Middle: 65%
Treble: 45%
Presence: 40%
Output: 65%
Dry/Wet: 70%
Cabinet: On

Chain:
EQ → Compressor → Amp → EQ (Post)

結果:
パンチある歪みBass

Lead Vocal Warmth:
Type: Blues
Gain: 35%
Bass: 45%
Middle: 70%
Treble: 50%
Presence: 55%
Dry/Wet: 60%
Cabinet: Off

結果:
温かく前に出るVocal
```

---

## パラレルサチュレーションテクニック

**プロの秘密:**

### 概念深掘り

```
なぜParallel?

問題:
100%歪み
→ 元の質感失う

解決:
Dry + Wet
→ 両方の良さ

メリット:

1. 自然:
   元音保持

2. 太さ:
   倍音追加

3. 柔軟性:
   調整範囲広い

4. 透明度:
   クリアさ保持

プロの使用率:
80%以上がParallel
```

### New York Compression応用

```
概念:

NY Compression:
Parallel圧縮
ドラム定番

応用:
Parallel Saturation
同じ原理

設定:

1. Return Track E:

   Saturator:
   Curve: Analog Clip
   Drive: 12 dB (強烈)
   Output: -12 dB
   Dry/Wet: 100%

2. Send量:

   Kick: 15%
   Snare: 20%
   Bass: 25%
   Vocal: 15%

3. Return処理:

   EQ Eight:
   High Pass 100 Hz
   Low Pass 8 kHz

   Compressor:
   Ratio: 4:1
   Attack: 10 ms
   Release: 100 ms

結果:

元音:
クリーン保持

歪み:
厚み・温かみ

バランス:
Send量で調整

メリット:

CPU効率:
1つのSaturator
複数トラック処理

一貫性:
同じ質感

柔軟性:
個別Send調整
```

### マルチバンドParallel

```
高度なテクニック:

目的:
帯域別処理
低域・中域・高域

設定:

Audio Effect Rack:

Chain 1 (Low):
- EQ Eight:
  Band Pass 20-200 Hz
- Saturator:
  Curve: Warm
  Drive: 10 dB

Chain 2 (Mid):
- EQ Eight:
  Band Pass 200-2000 Hz
- Saturator:
  Curve: Analog Clip
  Drive: 8 dB

Chain 3 (High):
- EQ Eight:
  Band Pass 2000-20000 Hz
- Saturator:
  Curve: Soft Sine
  Drive: 4 dB

Chain 4 (Dry):
- 何もなし

Chain Mix:
Chain 1: 30%
Chain 2: 40%
Chain 3: 20%
Chain 4: 60% (Dry)

用途:

Master処理:
繊細な制御

Bass:
低域のみ歪み

Drum Bus:
帯域別調整

メリット:

精密:
帯域別最適化

透明:
過剰歪み回避

柔軟:
Mix調整自由

Macro作成:

Low Sat: Chain 1 Mix
Mid Sat: Chain 2 Mix
High Sat: Chain 3 Mix
Dry: Chain 4 Mix

結果:
ライブ調整可能
```

---

## ジャンル別サチュレーション活用法

**各ジャンルの定石:**

### Techno

```
特徴:

ダーク:
低域重視

アナログ感:
ビンテージ機材再現

ミニマル:
わずかな歪み

Kick設定:

Saturator:
Curve: Analog Clip
Drive: 4 dB
Base: 0 Hz
Dry/Wet: 100%
Output: -4 dB

Chain:
EQ → Transient Shaper → Saturator

理由:
パンチ + 温かみ

Bass設定:

Saturator:
Curve: Warm
Drive: 8 dB
Base: 0 Hz
Dry/Wet: 80%
Output: -8 dB

Chain:
Filter → Saturator → EQ

+ Parallel Return:
   Drive: 15 dB
   Send: 20%

理由:
太く、ダーク

Hi-Hat設定:

Erosion:
Frequency: 3000 Hz
Mode: Noise
Width: 30%
Amount: 20%

結果:
Lo-Fi質感
アナログ的

Master設定:

Saturator:
Curve: Soft Sine
Drive: 2 dB
Dry/Wet: 40% (Parallel)
Output: -2 dB

+ Vinyl Distortion:
   Drive: 3 dB
   Crackle: 10%

結果:
わずかなビンテージ感
```

### House

```
特徴:

温かい:
偶数倍音

グルーヴ:
ドラム一体感

ディスコ影響:
ビンテージ

Kick設定:

Saturator:
Curve: Analog Clip
Drive: 3 dB (軽め)
Base: 0 Hz
Dry/Wet: 100%
Output: -3 dB

理由:
温かく、パンチ

Bass設定:

Saturator:
Curve: Warm
Drive: 6 dB
Base: 0 Hz
Dry/Wet: 70%
Output: -6 dB

+ EQ Eight:
   Peak +2 dB @ 80 Hz
   High Pass 40 Hz

結果:
温かく、丸い

Vocal設定:

Saturator:
Curve: Warm (Tube)
Drive: 5 dB
Dry/Wet: 60%
Output: -5 dB

Chain:
EQ → Compressor → Saturator → De-esser

結果:
ディスコ的温かさ

Drum Bus設定:

Saturator:
Curve: A Bit Warmer
Drive: 4 dB
Base: 200 Hz
Dry/Wet: 100%
Output: -4 dB

+ Glue Compressor:
   Ratio: 2:1
   Attack: 10 ms
   Release: Auto

結果:
ドラム一体感

Master設定:

Saturator:
Curve: Warm
Drive: 3 dB
Dry/Wet: 50%
Output: -3 dB

+ Tape Emulation:
   EQ High Shelf -1 dB @ 10 kHz

結果:
ビンテージHouse質感
```

### Drum & Bass

```
特徴:

クリア:
高域明瞭

パンチ:
トランジスタ的

aggressive:
奇数倍音

Kick設定:

Saturator:
Curve: Digital Clip
Drive: 5 dB
Base: 0 Hz
Dry/Wet: 60% (Parallel)
Output: -5 dB

理由:
パンチ、エッジ

Bass設定:

Parallel Setup:

Chain 1 (Low):
- EQ: Band Pass 40-150 Hz
- Saturator:
  Curve: Analog Clip
  Drive: 6 dB

Chain 2 (Mid):
- EQ: Band Pass 150-2000 Hz
- Saturator:
  Curve: Digital Clip
  Drive: 10 dB

Chain 3 (High):
- EQ: Band Pass 2000-8000 Hz
- Pedal:
  Model: Distortion
  Gain: 60%

Chain Mix:
Low: 80%
Mid: 50%
High: 30%

結果:
aggressive、複雑

Snare設定:

Saturator:
Curve: Digital Clip
Drive: 4 dB
Base: 0 Hz
Dry/Wet: 70%
Output: -4 dB

+ Transient Shaper:
   Attack: +6 dB

結果:
鋭い、パンチ

Reese Bass:

Saturator 1:
Curve: Analog Clip
Drive: 8 dB
Output: -8 dB

Saturator 2 (Serial):
Curve: Digital Clip
Drive: 5 dB
Output: -5 dB

+ EQ Eight:
   Peak -4 dB @ 400 Hz
   Peak +3 dB @ 100 Hz

結果:
太く、aggressive

Master設定:

Saturator:
Curve: Analog Clip
Drive: 2 dB (控えめ)
Dry/Wet: 30%
Output: -2 dB

理由:
クリア保持
わずかな厚み
```

### Ambient / Downtempo

```
特徴:

自然:
わずかな歪み

温かい:
偶数倍音

Lo-Fi:
ビンテージ

Pad設定:

Saturator:
Curve: Warm
Drive: 6 dB
Base: 0 Hz
Dry/Wet: 50% (Parallel)
Output: -6 dB

Chain:
Reverb → Saturator

理由:
温かく、空間的

Vocal設定:

Saturator:
Curve: Warm
Drive: 8 dB
Dry/Wet: 40%
Output: -8 dB

+ Vinyl Distortion:
   Drive: 5 dB
   Crackle: 30%

結果:
Lo-Fi、ノスタルジック

Drum設定:

Erosion:
Frequency: 2000 Hz
Mode: Noise
Width: 40%
Amount: 30%

+ Saturator:
   Curve: Warm
   Drive: 5 dB
   Dry/Wet: 60%

結果:
Lo-Fi質感

Bass設定:

Saturator:
Curve: Warm
Drive: 7 dB
Base: 200 Hz (低域のみ)
Dry/Wet: 70%
Output: -7 dB

理由:
温かく、控えめ

Master設定:

Vinyl Distortion:
Tracing: Pinch
Drive: 6 dB
Crackle: 20%

+ Saturator:
   Curve: Soft Sine
   Drive: 3 dB
   Dry/Wet: 40%

結果:
ビンテージ、温かい
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

## マルチバンドサチュレーション

**帯域別の精密制御:**

### 概念と必要性

```
通常のSaturation:

全帯域一律:
20 Hz - 20 kHz
同じ処理

問題:

低域歪み:
濁る

高域歪み:
耳障り

解決:

マルチバンド:
帯域別最適化

メリット:

1. 精密制御:
   低域: 温かく
   中域: 存在感
   高域: 明るく

2. 透明度:
   過剰歪み回避

3. 柔軟性:
   独立調整

プロの使用:

Mastering:
80%以上

Mix Bus:
50%

Individual:
30%
```

### 3バンド構成

```
基本構成:

Low Band:
20-200 Hz
低域

Mid Band:
200-2000 Hz
中域

High Band:
2000-20000 Hz
高域

クロスオーバー:

Low/Mid: 200 Hz
Mid/High: 2 kHz

理由:
音楽的分割
```

### Audio Effect Rack設定

```
完全な構築手順:

1. Audio Effect Rack作成:

   Cmd + G
   グループ化

2. Chain 1 (Low):

   a. EQ Eight:
      High Pass: Off
      Low Pass: 200 Hz
      Slope: 48 dB/oct

   b. Saturator:
      Curve: Warm
      Drive: 10 dB
      Output: -10 dB
      Dry/Wet: 100%

   理由:
   低域は温かく
   偶数倍音

3. Chain 2 (Mid):

   a. EQ Eight:
      High Pass: 200 Hz
      Low Pass: 2000 Hz
      Slope: 24 dB/oct

   b. Saturator:
      Curve: Analog Clip
      Drive: 8 dB
      Output: -8 dB
      Dry/Wet: 100%

   理由:
   中域は存在感
   バランス

4. Chain 3 (High):

   a. EQ Eight:
      High Pass: 2000 Hz
      Low Pass: Off
      Slope: 24 dB/oct

   b. Saturator:
      Curve: Soft Sine
      Drive: 4 dB
      Output: -4 dB
      Dry/Wet: 100%

   理由:
   高域は控えめ
   耳に優しい

5. Chain 4 (Dry):

   何もなし

   理由:
   Parallel処理

6. Chain Mix設定:

   Chain 1: 40%
   Chain 2: 50%
   Chain 3: 30%
   Chain 4: 60% (Dry)

   調整:
   ジャンル別最適化
```

### Macroマッピング

```
8つのMacro:

Macro 1: Low Sat
→ Chain 1 Mix
範囲: 0-100%

Macro 2: Mid Sat
→ Chain 2 Mix
範囲: 0-100%

Macro 3: High Sat
→ Chain 3 Mix
範囲: 0-100%

Macro 4: Dry Mix
→ Chain 4 Mix
範囲: 0-100%

Macro 5: Low Drive
→ Chain 1 Saturator Drive
範囲: 0-15 dB

Macro 6: Mid Drive
→ Chain 2 Saturator Drive
範囲: 0-12 dB

Macro 7: High Drive
→ Chain 3 Saturator Drive
範囲: 0-8 dB

Macro 8: Master Output
→ 全Chain Output
範囲: -12 - 0 dB

メリット:

ライブ調整:
8つのノブ
直感的

保存:
プリセット化
再利用
```

### ジャンル別設定

```
Techno:

Low Sat: 50%
Mid Sat: 40%
High Sat: 20%
Dry: 50%

理由:
低域重視
ダーク

House:

Low Sat: 45%
Mid Sat: 55%
High Sat: 35%
Dry: 55%

理由:
バランス
温かい

Drum & Bass:

Low Sat: 35%
Mid Sat: 60%
High Sat: 40%
Dry: 50%

理由:
中域存在感
クリア

Ambient:

Low Sat: 30%
Mid Sat: 35%
High Sat: 25%
Dry: 70%

理由:
自然
透明
```

---

## サチュレーションとコンプレッションの併用

**相乗効果:**

### 順序の重要性

```
パターン1: Comp → Sat

設定:

Compressor:
Ratio: 4:1
Attack: 10 ms
Release: 100 ms

Saturator:
Curve: Analog Clip
Drive: 6 dB
Output: -6 dB

効果:

安定した歪み:
Comp後の均一レベル

予測可能:
ダイナミクス制御済み

用途:

Vocal: 最適
Bass: 推奨

パターン2: Sat → Comp

設定:

Saturator:
Curve: Warm
Drive: 8 dB
Output: -8 dB

Compressor:
Ratio: 2:1
Attack: 30 ms
Release: Auto

効果:

自然な圧縮:
倍音豊かな信号

複雑:
ダイナミック

用途:

Drums: 面白い
実験的

パターン3: Parallel両方

設定:

Return A (Comp):
Compressor
Ratio: 8:1
Attack: 1 ms
Release: 50 ms

Return B (Sat):
Saturator
Drive: 12 dB

元トラック:
クリーン

Send量:
Comp: 25%
Sat: 20%

効果:

最も柔軟:
独立制御

透明:
元音残る

プロ手法:
NY Compression応用
```

### 実践例

```
Vocal完全Chain:

1. EQ Eight:
   High Pass 80 Hz

2. Compressor:
   Ratio: 3:1
   Attack: 10 ms
   Release: 100 ms
   Knee: 6 dB

3. Saturator:
   Curve: Warm
   Drive: 5 dB
   Output: -5 dB
   Dry/Wet: 70%

4. EQ Eight (Post):
   Peak -2 dB @ 7 kHz (De-esser)

5. Compressor (Final):
   Ratio: 2:1
   Attack: 30 ms
   Release: Auto
   Knee: 3 dB

6. Limiter:
   Ceiling: -0.3 dB
   Release: 50 ms

結果:
プロ品質Vocal

Bass完全Chain:

1. EQ Eight:
   High Pass 40 Hz
   Peak -3 dB @ 300 Hz

2. Multiband Comp:
   Low: 20-150 Hz
   Ratio: 3:1

3. Saturator:
   Curve: Analog Clip
   Drive: 6 dB
   Output: -6 dB

4. Compressor:
   Ratio: 2:1
   Attack: 20 ms
   Sidechain: Kick

5. EQ Eight (Post):
   High Shelf -1 dB @ 8 kHz

結果:
太く、コントロール
```

---

## 高度なテクニック集

**プロの秘密:**

### Serial Saturation

```
概念:

複数Saturator:
直列接続

メリット:

複雑な倍音:
段階的構築

自然:
各段階は控えめ

設定例:

Bass Serial Chain:

1. Saturator 1:
   Curve: Warm
   Drive: 4 dB
   Output: -4 dB

2. EQ Eight:
   Peak -2 dB @ 500 Hz

3. Saturator 2:
   Curve: Analog Clip
   Drive: 3 dB
   Output: -3 dB

4. EQ Eight (Post):
   High Pass 40 Hz

結果:

複雑な倍音:
2段階生成

温かく太い:
自然な厚み

ルール:

各Saturator:
Drive: 3-5 dB (控えめ)

合計:
6-10 dB

理由:
過剰回避
```

### Frequency-Dependent Dry/Wet

```
高度なテクニック:

目的:
周波数別Dry/Wet比率

設定:

Audio Effect Rack:

Chain 1 (Low Wet):
- EQ: Band Pass 20-200 Hz
- Saturator:
  Drive: 10 dB
  Dry/Wet: 80%

Chain 2 (Mid Wet):
- EQ: Band Pass 200-2000 Hz
- Saturator:
  Drive: 8 dB
  Dry/Wet: 60%

Chain 3 (High Wet):
- EQ: Band Pass 2000-20000 Hz
- Saturator:
  Drive: 5 dB
  Dry/Wet: 40%

Chain 4 (Dry):
- 何もなし

Chain Mix:
All: 100%

効果:

低域:
最も歪み

高域:
控えめ

自然:
周波数特性維持
```

### Automation Techniques

```
ダイナミックSaturation:

1. Intro (0-32 bars):

   Drive: 0 dB
   Dry/Wet: 0%

   理由:
   クリーン
   導入

2. Build-up (33-48 bars):

   Drive: 0 → 8 dB (自動化)
   Dry/Wet: 0 → 80%

   理由:
   緊張感増加

3. Drop (49-64 bars):

   Drive: 8 dB
   Dry/Wet: 100%

   理由:
   最大インパクト

4. Breakdown (65-80 bars):

   Drive: 8 → 3 dB
   Dry/Wet: 100 → 40%

   理由:
   緩和

5. Final Drop (81-112 bars):

   Drive: 10 dB
   Dry/Wet: 100%

   理由:
   クライマックス

効果:

ダイナミック展開:
エネルギー制御

プロ品質:
計算された変化
```

### Sidechain Saturation

```
高度なテクニック:

概念:

Sidechain制御:
Saturator Dry/Wet
Kickに反応

設定:

1. Bass Track:

   Saturator:
   Curve: Analog Clip
   Drive: 8 dB
   Dry/Wet: Mapped to Compressor

2. Compressor (Sidechain):

   Input: Kick
   Ratio: 10:1
   Attack: 1 ms
   Release: 100 ms
   Map to: Saturator Dry/Wet

3. Mapping:

   Comp 0%: Sat 100%
   Comp 100%: Sat 20%

効果:

Kick時:
Bass歪み減少
クリア

Kick外:
Bass歪み増加
太い

結果:

明瞭:
Kick/Bass分離

太さ:
Bass存在感
```

---

## トラブルシューティング

**よくある問題と解決:**

### 問題1: 濁る

```
症状:

Mix全体:
濁る
不明瞭

原因:

低域歪み過剰:
200-500 Hz

解決策:

1. Base周波数活用:

   Saturator:
   Base: 0 Hz → 200 Hz

   効果:
   低域のみ処理

2. Post EQ:

   EQ Eight:
   Peak -2 to -4 dB @ 300-500 Hz

   効果:
   濁り除去

3. Dry/Wet調整:

   Dry/Wet: 100% → 60%

   効果:
   元音ブレンド

4. Curve変更:

   Warm → Soft Sine

   効果:
   控えめな歪み
```

### 問題2: 耳障り

```
症状:

高域:
耳障り
疲れる

原因:

奇数倍音過剰:
3 kHz - 8 kHz

解決策:

1. Curve変更:

   Digital Clip → Analog Clip

   効果:
   偶数倍音
   温かい

2. Post EQ:

   EQ Eight:
   Peak -2 dB @ 5 kHz
   Low Pass 12 kHz

   効果:
   高域制御

3. Dry/Wet:

   100% → 50%

   効果:
   自然

4. マルチバンド:

   High Band:
   Drive: 8 dB → 3 dB

   効果:
   高域控えめ
```

### 問題3: 薄い

```
症状:

歪み後:
薄い
存在感ない

原因:

設定不足:
Drive低すぎ
Dry/Wet低すぎ

解決策:

1. Drive増加:

   3 dB → 6-8 dB

   効果:
   倍音増加

2. Curve変更:

   Soft Sine → Analog Clip

   効果:
   明確な歪み

3. Serial Saturation:

   Saturator 2台:
   各4 dB

   効果:
   複雑な倍音

4. EQブースト:

   Pre EQ:
   Peak +2 dB @ 100 Hz
   Peak +3 dB @ 3 kHz

   効果:
   歪む帯域強調
```

### 問題4: 音量バラバラ

```
症状:

トラック間:
音量差
バランス崩壊

原因:

Output調整忘れ:
ゲインマッチングなし

解決策:

1. 全Saturator確認:

   Drive +5 dB
   → Output -5 dB

   ルール:
   必ず同量逆

2. Utility使用:

   Post Saturator:
   Utility
   Gain: -3 to -6 dB

   効果:
   全体レベル調整

3. メーター確認:

   各トラック:
   Peak: -6 dB
   RMS: -18 dB

   効果:
   一貫性

4. Parallel調整:

   Send量:
   15-25%

   効果:
   音量変化最小
```

---

## プロのワークフロー完全版

**実際の制作手順:**

### Day 1: 基礎設定

```
時間: 2時間

手順:

1. Bassトラック (30分):

   a. Saturator挿入
   b. Curve: Analog Clip
   c. Drive: 5 dB
   d. Output: -5 dB
   e. A/B比較
   f. 微調整

2. Kickトラック (30分):

   a. Saturator挿入
   b. Curve: Analog Clip
   c. Drive: 3 dB
   d. Output: -3 dB
   e. アタック確認
   f. 調整

3. Vocalトラック (30分):

   a. Saturator挿入
   b. Curve: Warm
   c. Drive: 5 dB
   d. Output: -5 dB
   e. 温かみ確認

4. 全体確認 (30分):

   バランス
   濁りチェック
   調整

結果:

基本3トラック:
歪み設定完了
```

### Day 2: 応用

```
時間: 2時間

手順:

1. Parallel Return作成 (45分):

   a. Return E作成
   b. Saturator:
      Drive: 12 dB
   c. EQ Eight:
      Band Pass 100-8000 Hz
   d. Send設定:
      各トラック15-25%

2. Drum Bus処理 (45分):

   a. Drum Bus作成
   b. Glue Compressor
   c. Saturator:
      Curve: A Bit Warmer
      Drive: 3 dB
   d. 一体感確認

3. Master処理 (30分):

   a. Saturator:
      Curve: Soft Sine
      Drive: 2 dB
      Dry/Wet: 50%
   b. 全体確認
   c. 最終調整

結果:

全トラック:
歪み最適化完了
```

### Day 3: 高度な最適化

```
時間: 3時間

手順:

1. マルチバンド構築 (90分):

   Bass用:
   3バンド構成

   Master用:
   3バンド構成

   Macro設定

2. Automation (60分):

   Intro → Drop:
   Drive自動化

   Breakdown:
   Dry/Wet自動化

3. 最終Mix (30分):

   全体バランス
   濁りチェック
   ゲインマッチング

結果:

完成:
プロ品質
```

---

## リファレンスチャート

**クイックガイド:**

### Curve Type選択

```
用途別:

Vocal:
Warm (Tube)

Bass:
Analog Clip

Kick:
Analog Clip

Snare:
Digital Clip (軽め)

Master:
Soft Sine

Pad:
Warm

Lead:
Analog Clip / Digital Clip

Hi-Hat:
Erosion
```

### Drive量ガイド

```
トラック別:

Bass: 5-8 dB
Kick: 3-5 dB
Vocal: 3-6 dB
Snare: 2-4 dB
Drum Bus: 3-4 dB
Master: 2-3 dB

Parallel Return:
10-15 dB (強め)
```

### Dry/Wet設定

```
用途別:

Individual Track:
60-100%

Parallel処理:
20-50%

Master:
40-60%

実験的:
30-70%
```

---

## さらに学ぶために

**次のステップ:**

### 推奨リソース

```
Ableton公式:

マニュアル:
Saturator詳細

チュートリアル:
YouTube公式チャンネル

外部リソース:

Sound on Sound:
サチュレーション理論

MixBus.tv:
実践テクニック

プラグイン:

UAD:
Neve, API, Studer

Waves:
J37, Abbey Road

Slate Digital:
Virtual Console

練習方法:

1. A/B比較:
   必ず実施

2. 倍音分析:
   Spectrum確認

3. リファレンス:
   プロトラック比較

4. 実験:
   様々な設定試す
```

---

## 最終チェックリスト

```
Saturation設定:

□ Curve Type選択済み
□ Drive量適切 (3-8 dB)
□ Output調整済み (Drive分マイナス)
□ Dry/Wet最適化
□ A/B比較実施

全体バランス:

□ 濁りなし
□ 高域耳障りなし
□ 低域クリア
□ 音量一定
□ ゲインマッチング

高度な設定:

□ Parallel Return (必要時)
□ マルチバンド (Master)
□ Automation (展開時)
□ Chain順序最適

最終確認:

□ Spectrum確認
□ リファレンス比較
□ 複数環境試聴
□ 保存・命名
```

---

**おめでとうございます！** Distortion・Saturationの完全マスターを達成しました。次は[Modulation Effects](./modulation-effects.md)でChorus、Flanger、Phaserによる動きの付け方を学びましょう。

**重要なポイント:**
- Less is More（控えめが正解）
- A/B比較は必須
- Output調整を忘れずに
- Parallelで自然な太さ
- ジャンル別最適化

継続的な練習で、プロレベルの温かく太いサウンドを実現できます。
