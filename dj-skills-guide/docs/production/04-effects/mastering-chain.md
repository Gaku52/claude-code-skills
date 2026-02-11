# Mastering Chain

最終仕上げのマスタリング実践。Master Trackエフェクトチェーンを完全マスターし、-14 LUFS・プロの音質を実現します。

## この章で学ぶこと

- マスタリングとは何か
- LUFS・True Peak基礎
- EQ Eight（マスタリング用）
- Multiband Dynamics
- Glue Compressor
- Limiter（最終音圧）
- 完全なMastering Chain
- リファレンストラック活用

---

## なぜMasteringが重要なのか

**最終10%の仕上げ:**

```
制作プロセス:

楽曲構成: 20%
音色選択: 15%
ドラム: 15%
ミックス: 40%
マスタリング: 10%

たった10%:

でも:
最も重要な10%

理由:

ミックス完璧:
でも音量小さい

マスタリング:
商業レベル音圧
-14 LUFS達成

プロとアマの差:

アマ:
ミックスまで

プロ:
マスタリングも完璧

結果:
プロ: 太い、大きい、クリア
アマ: 薄い、小さい
```

---

## マスタリング基礎

**目的と原則:**

### マスタリングの目的

```
1. 音量最適化:

目標:
-14 LUFS (Spotify等)
-16 LUFS (Apple Music)

True Peak:
-1.0 dBTP以下

2. 周波数バランス:

全帯域:
バランス良く

低域:
パワフル、クリア

高域:
明るい、刺さらない

3. ダイナミクス制御:

圧縮:
適度に

呼吸:
維持

4. ステレオイメージ:

広い:
でも低域Mono

5. 最終チェック:

全デバイス:
良い音

Spotify・Apple Music:
問題なし
```

### マスタリング vs ミックス

```
ミックス:

Individual Track:
各トラック処理

目的:
バランス

使用:
EQ・Comp・Reverb

マスタリング:

Master Track:
全体処理

目的:
最終仕上げ

使用:
EQ・Multiband・Limiter

重要:

ミックス完璧:
マスタリング簡単

ミックス甘い:
マスタリングで救えない

原則:
「ミックス80%、マスタリング20%」
```

---

## LUFS・True Peak基礎

**ラウドネス標準:**

### LUFS (Loudness Units Full Scale)

```
定義:

人間の聴覚に基づく
音量測定単位

dB vs LUFS:

dB:
物理的音量

LUFS:
知覚的音量

プラットフォーム標準:

Spotify: -14 LUFS
Apple Music: -16 LUFS
YouTube: -14 LUFS
SoundCloud: -14 LUFS (推奨)
Beatport: -9 〜 -6 LUFS (クラブ用)

推奨:

配信: -14 LUFS
クラブ: -8 LUFS

理由:
-14 LUFS:
ダイナミクス維持
高音質
```

### True Peak

```
定義:

デジタル変換後の
実際のピーク値

問題:

0 dBFS到達:
クリッピング

解決:

True Peak:
-1.0 dBTP以下

推奨:
-1.0 〜 -0.3 dBTP

測定:

Utility (Ableton):
True Peakメーター

Youlean Loudness Meter:
LUFS + True Peak
無料プラグイン
```

---

## Mastering Chain構成

**標準的な順序:**

### 推奨Chain

```
Master Track:

1. Utility (Gain In)
   ↓
2. EQ Eight (補正)
   ↓
3. Multiband Dynamics
   ↓
4. Glue Compressor
   ↓
5. EQ Eight (最終調整)
   ↓
6. Saturator (温かみ)
   ↓
7. Utility (Stereo Width調整)
   ↓
8. Limiter (最終音圧)
   ↓
9. Utility (Gain Out, True Peak)

理由:

順序重要:
各段階で微調整

Less is More:
最小限のエフェクト

ヘッドルーム:
Limiter前に-6 dB以上
```

---

## Step 1: EQ Eight (補正)

**周波数バランス:**

### 目的

```
問題周波数:
除去

全体バランス:
最適化

やってはいけない:
大きなブースト
```

### 設定例

```
Band 1 (High Pass):

Type: High Pass
Freq: 30 Hz
Slope: 24 dB/oct

理由:
不要な超低域カット
スピーカー保護

Band 2 (Low Shelf):

Freq: 100 Hz
Gain: +0.5 〜 +1.5 dB
Q: 1.0

用途:
わずかなパワー追加
控えめ

Band 3 (Peak):

Freq: 250-400 Hz
Gain: -0.5 〜 -2.0 dB
Q: 2.0

理由:
濁り除去
最も重要

Band 4 (Peak):

Freq: 3000 Hz
Gain: +0.5 〜 +1.5 dB
Q: 1.5

理由:
明瞭度
存在感

Band 5 (High Shelf):

Freq: 10000 Hz
Gain: +0.5 〜 +1.0 dB
Q: 1.0

理由:
空気感
高級感

重要:

全て控えめ:
±2 dB以内

A/B比較:
必須

Spectrumで確認:
視覚的
```

---

## Step 2: Multiband Dynamics

**帯域別圧縮:**

### 目的

```
各帯域:
独立制御

低域:
タイト、パワフル

中域:
明瞭

高域:
滑らか
```

### インターフェイス

```
┌─────────────────────────────────┐
│  Multiband Dynamics             │
├─────────────────────────────────┤
│  Band 1 (Low): 20-120 Hz        │
│    Threshold: -15 dB            │
│    Ratio: 3:1                   │
│    Attack: 30 ms                │
│    Release: 100 ms              │
│                                 │
│  Band 2 (Low-Mid): 120-1000 Hz  │
│    Threshold: -12 dB            │
│    Ratio: 2:1                   │
│                                 │
│  Band 3 (Mid): 1k-6k Hz         │
│    Threshold: -10 dB            │
│    Ratio: 2:1                   │
│                                 │
│  Band 4 (High): 6k-20k Hz       │
│    Threshold: -8 dB             │
│    Ratio: 2:1                   │
└─────────────────────────────────┘

4つの帯域:
それぞれ独立
```

### 推奨設定

```
Band 1 (20-120 Hz):

Threshold: -15 dB
Ratio: 3:1 (やや強め)
Attack: 30 ms
Release: 100 ms

理由:
低域タイト
パワー維持

GR目標:
-2 〜 -4 dB

Band 2 (120-1000 Hz):

Threshold: -12 dB
Ratio: 2:1 (軽め)
Attack: 10 ms
Release: Auto

理由:
濁り防止

GR目標:
-1 〜 -3 dB

Band 3 (1k-6k Hz):

Threshold: -10 dB
Ratio: 2:1
Attack: 5 ms (速い)
Release: 50 ms

理由:
明瞭度維持

GR目標:
-1 〜 -2 dB (軽め)

Band 4 (6k-20k Hz):

Threshold: -8 dB
Ratio: 2:1
Attack: 3 ms (最速)
Release: 40 ms

理由:
刺さり防止

GR目標:
-1 〜 -2 dB (軽め)
```

---

## Step 3: Glue Compressor

**全体まとめる:**

### 目的

```
機能:

全体を「接着」
一体感

特徴:
Analog的
SSL風

用途:
マスタリング必須
```

### 設定例

```
┌─────────────────────────────────┐
│  Glue Compressor                │
├─────────────────────────────────┤
│  Threshold: -15 dB              │
│  Ratio: 2:1                     │
│  Attack: 10 ms                  │
│  Release: Auto (0.3 s)          │
│                                 │
│  Make-Up: 2.0 dB                │
│  Dry/Wet: 100%                  │
│                                 │
│  Range: 0 dB (Full)             │
│  Soft Clip: On                  │
└─────────────────────────────────┘

推奨設定:

Threshold: -15 〜 -10 dB
Ratio: 2:1 (軽め)
Attack: 10 ms
Release: Auto

GR目標:
-2 〜 -4 dB
わずかに

Soft Clip:
On (温かみ)

Make-Up:
GR分補正
```

---

## Step 4: Saturator (温かみ)

**わずかな厚み:**

### 設定例

```
Curve: Soft Sine または A Bit Warmer

Drive: 2.0 〜 4.0 dB (最も控えめ)
Base: 0 Hz
Dry/Wet: 50% (Parallel)
Output: -2.0 〜 -4.0 dB

理由:

温かみ:
わずかに

Parallel:
自然

Master:
最も控えめ

注意:
過剰: 崩壊
```

---

## Step 5: Utility (Width調整)

**ステレオイメージ:**

### 設定

```
┌─────────────────────────────────┐
│  Utility                        │
├─────────────────────────────────┤
│  Width: 105-110%                │
│  Bass Mono: On                  │
│  Freq: 120 Hz                   │
└─────────────────────────────────┘

Width:

100%: 標準
105-110%: わずかに広く
推奨

理由:
広すぎ: 低域位相問題

Bass Mono:

必須:
120 Hz以下Mono

理由:
低域: Mono維持
高域: Stereo
```

---

## Step 6: Limiter (最終音圧)

**最重要デバイス:**

### 目的

```
機能:

音圧最大化
-14 LUFS達成

True Peak:
-1.0 dBTP以下

クリッピング:
防止
```

### インターフェイス

```
┌─────────────────────────────────┐
│  Limiter                        │
├─────────────────────────────────┤
│  Ceiling: -0.3 dB               │
│  Gain: 6.0 dB                   │
│                                 │
│  Release: Auto (400 ms)         │
│  Lookahead: 1.0 ms              │
│                                 │
│  Stereo Link: 100%              │
│                                 │
│  GR Meter: -4 dB ■■■□□          │
│                                 │
│  [LUFS Meter]                   │
└─────────────────────────────────┘

重要パラメーター:
- Ceiling (最大音量)
- Gain (入力ゲイン)
```

### パラメーター詳細

```
1. Ceiling:

機能:
絶対最大音量

設定:

-0.3 dB: 安全 (推奨)
-0.5 dB: より安全
-1.0 dB: 最も安全

理由:
True Peak保護

推奨:
-0.3 dB

2. Gain:

機能:
入力ゲイン

目標:

GR: -4 〜 -6 dB
→ -14 LUFS達成

調整方法:

1. Gain: 0 dB (初期)
2. LUFS確認
   例: -20 LUFS
3. 差分計算
   20 - 14 = 6 dB
4. Gain: +6 dB
5. 再確認

推奨:
+4 〜 +8 dB

3. Release:

Auto (推奨):
テンポ追従

Manual:
100-500 ms

推奨:
Auto

4. Lookahead:

機能:
先読み

設定:

1.0 ms: 標準
0.1 ms: 速い

推奨:
1.0 ms

5. Stereo Link:

100%: 完全リンク (推奨)
ステレオイメージ維持

0%: 独立
音圧最大
```

---

## 完全なMastering Chain例

**Techno/House標準:**

### Chain全体

```
Master Track:

1. Utility:
   Gain: 0 dB (基準)

2. EQ Eight:
   High Pass: 30 Hz
   Low Shelf: +1 dB @ 100 Hz
   Peak: -1.5 dB @ 300 Hz
   Peak: +1 dB @ 3 kHz
   High Shelf: +0.8 dB @ 10 kHz

3. Multiband Dynamics:
   Low: Ratio 3:1, GR -3 dB
   Low-Mid: Ratio 2:1, GR -2 dB
   Mid: Ratio 2:1, GR -1 dB
   High: Ratio 2:1, GR -1 dB

4. Glue Compressor:
   Threshold: -12 dB
   Ratio: 2:1
   Attack: 10 ms
   Release: Auto
   GR: -3 dB
   Make-Up: +3 dB

5. EQ Eight (2nd):
   わずかな最終調整
   ±0.5 dB

6. Saturator:
   Curve: Soft Sine
   Drive: 3 dB
   Dry/Wet: 50%
   Output: -3 dB

7. Utility:
   Width: 108%
   Bass Mono: On (120 Hz)

8. Limiter:
   Ceiling: -0.3 dB
   Gain: +6 dB
   Release: Auto
   GR: -5 dB

9. Utility (Output):
   True Peakメーター確認

結果:

LUFS: -14.0 LUFS
True Peak: -0.5 dBTP
ダイナミクス: 維持
```

---

## リファレンストラック活用

**比較の重要性:**

### 選び方

```
条件:

1. 同じジャンル:
   Techno → Techno
   House → House

2. プロ制作:
   商業リリース

3. 音質良い:
   WAV・FLAC
   320kbps MP3

4. 好きな曲:
   目標とする音

推奨:

Techno:
Adam Beyer, Amelie Lens

House:
CamelPhat, Fisher

入手:
Beatport (WAV)
Spotify (320kbps)
```

### 使用方法

```
Setup:

1. Audio Track作成
   名前: "Reference"

2. リファレンス配置

3. Solo聴き比べ

4. Spectrumで比較

チェックポイント:

LUFS:
同じくらいか？
-14 LUFS目標

低域:
太さ比較

高域:
明るさ比較

ステレオ:
広さ比較

調整:

EQ:
周波数バランス

Compressor:
ダイナミクス

Limiter:
音圧

推奨頻度:
10分ごと
```

---

## よくある失敗

### 1. Limiterかけすぎ

```
問題:
音が潰れる
ダイナミクスゼロ

原因:
Gain高すぎ
GR -10 dB+

解決:

GR目標:
-4 〜 -6 dB

LUFS:
-14 LUFS達成で十分

理由:
-9 LUFS: 潰れる
-14 LUFS: ダイナミクス維持
```

### 2. ミックスで問題

```
問題:
マスタリングで救えない

原因:
ミックス不完全

解決:

ミックス見直し:
- 低域濁り
- バランス
- ヘッドルーム

マスタリング:
最終10%のみ

原則:
「ミックス80%」
```

### 3. EQブーストしすぎ

```
問題:
不自然

原因:
±3 dB以上

解決:

Master EQ:
±2 dB以内

カット優先:
ブーストより

理由:
マスター: 最も控えめ
```

### 4. True Peak無視

```
問題:
配信でクリッピング

原因:
True Peak 0 dB超え

解決:

Limiter Ceiling:
-0.3 dB

確認:
True Peakメーター

推奨:
-1.0 〜 -0.3 dBTP
```

---

## マスタリング前チェックリスト

**ミックス完成確認:**

```
□ ヘッドルーム -6 dB以上 (Master)
□ 全トラックEQ済み
□ Kick・Bassバランス良い
□ 低域クリア (300 Hz以下)
□ Vocalが前に出る (必要なら)
□ Return Track設定済み
□ CPU使用率50%以下
□ 不要トラック削除
□ グループ化・整理済み
□ 保存済み
```

---

## マスタリング後チェックリスト

**書き出し前確認:**

```
□ LUFS: -14.0 LUFS (±0.5)
□ True Peak: -1.0 〜 -0.3 dBTP
□ クリッピングなし
□ リファレンスと比較済み
□ 全デバイスで確認 (ヘッドホン、スピーカー、車)
□ 低域: 太い、クリア
□ 高域: 明るい、刺さらない
□ ダイナミクス: 維持
□ ステレオ: 広い
□ 満足
```

---

## 書き出し設定

**最終Export:**

```
File Format:

WAV:
- Sample Rate: 44.1 kHz
- Bit Depth: 16-bit (配信)
             24-bit (マスター保存)

MP3:
- 320 kbps (CBR)

FLAC:
- ロスレス圧縮

推奨:

配信用:
WAV 16-bit 44.1 kHz

保存用:
WAV 24-bit 48 kHz

Normalize:
Off (Limiterで処理済み)

Dither:
On (24-bit → 16-bit時)
Triangular推奨
```

---

## 実践ワークフロー

**30分マスタリング:**

### Step-by-Step

```
0-5分: セットアップ

1. リファレンストラック配置
2. LUFS Meter挿入
3. 初期LUFS確認

5-10分: EQ

1. EQ Eight挿入
2. High Pass 30 Hz
3. 問題周波数カット
4. A/B比較

10-15分: ダイナミクス

1. Multiband Dynamics
2. 各帯域調整
3. Glue Compressor
   GR -3 dB

15-20分: 仕上げ

1. Saturator (わずか)
2. Utility (Width)
3. EQ微調整

20-25分: Limiter

1. Limiter挿入
2. Gain調整
3. -14 LUFS達成
4. True Peak確認

25-30分: 最終確認

1. リファレンス比較
2. 全体聴き直し
3. 書き出し
```

---

## まとめ

### Mastering Chain

```
標準構成:
EQ → Multiband → Glue → Saturator
→ Utility → Limiter

順序重要:
変更しない
```

### 目標値

```
□ LUFS: -14.0 LUFS
□ True Peak: -0.5 dBTP
□ ヘッドルーム: -6 dB (Limiter前)
□ GR (Limiter): -4〜-6 dB
```

### 重要原則

```
□ ミックス80%、マスタリング20%
□ Less is More (控えめ)
□ リファレンス比較必須
□ A/B比較
□ ダイナミクス維持
□ 複数デバイス確認
```

---

**次は:** [Audio Effect Rack](./effect-racks.md) - 複雑なエフェクトChain構築とMacro活用
