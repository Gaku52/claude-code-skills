# Stereo Imaging

ステレオ空間で広がりを作ります。Width・Panning・120 Hz以下Mono化を完全マスターします。

## この章で学ぶこと

- Stereo vs Mono基礎
- Width調整（Utility）
- Panning戦略
- 120 Hz以下Mono必須
- Mid/Side処理
- ステレオ幅測定
- Mono互換性確認

---

## なぜStereo Imagingが重要なのか

**広がりの秘訣:**

```
Stereo Imaging悪い:

特徴:
狭い
平坦
単調

Stereo Imaging良い:

特徴:
広い
立体的
没入感

プロとアマの差:

アマ:
全てCenter
狭い

プロ:
適切な配置
Wide

結果:
プロ: 広大、立体的
アマ: 狭い、平坦

重要ルール:

120 Hz以下:
必ずMono

理由:
低域Stereo
→ 位相問題
→ Mono再生で消える
```

---

## Stereo vs Mono基礎

**基本理解:**

### 定義

```
Mono:

チャンネル: 1つ
L = R
同じ音

特徴:
中央定位
タイト
パワフル

用途:
Kick・Bass・Vocal

Stereo:

チャンネル: 2つ
L ≠ R
違う音

特徴:
広がり
空間感

用途:
Pad・FX・Hi-Hat

Mono互換性:

問題:
Stereo信号
→ Mono再生
→ 位相キャンセル
→ 音量down/消える

解決:
120 Hz以下Mono化
必須
```

---

## Width調整

**Utility活用:**

### Widthパラメーター

```
Utility > Width:

0%:
完全Mono
L = R

100%:
元のStereo
デフォルト

105-120%:
広げる
Side成分強調

150%+:
過剰
位相問題

-100%:
位相反転
特殊効果

推奨設定:

Kick: 0% (Mono)
Bass: 0% (Mono)
Snare: 0-10%
Lead: 20-30%
Vocal: 0-10%
Pad: 80-100%
FX: 100-120%
Hi-Hat: 40-60%

ルール:
低域ほどMono
高域ほどWide
```

### Bass Mono機能

```
Utility > Bass Mono:

機能:
特定周波数以下をMono化

設定:

On/Off: On
Freq: 120 Hz

効果:
120 Hz以下 → Mono
120 Hz以上 → Stereo維持

メリット:

位相安全:
低域問題なし

高域Stereo:
広がり維持

必須:

全トラック:
Bass Mono: On

Master:
Bass Mono: On (120 Hz)

理由:
業界標準
Mono互換性
```

---

## Panning戦略

**左右配置:**

### 基本原則

```
Center (0%):

配置:
Kick・Bass・Snare・Vocal・Lead

理由:
最重要要素
パワー・存在感

L/R (-100% / +100%):

配置:
Hi-Hat・Percussion・FX

理由:
装飾
広がり

推奨配置:

Kick: Center
Bass: Center
Snare: Center
Vocal: Center
Lead: Center

Hi-Hat: L -30% / R +30%
Shaker: L -50% / R +50%
Percussion 1: L -40%
Percussion 2: R +40%

FX 1: L -80%
FX 2: R +80%

Pad: Center (Width 80-100%)

注意:

過剰Pan:
バランス悪い

LRペア:
バランス取る
```

---

## 120 Hz以下Mono化

**必須処理:**

### なぜ必須？

```
理由1: 位相問題

低域Stereo:
L/R位相差

Mono再生:
位相キャンセル
音量down

例:
Bass Stereo
→ Mono再生
→ 消える

理由2: 音響物理

低域:
波長長い
方向感ない

結果:
Stereo効果薄い

理由3: 再生環境

クラブ:
Mono再生多い

車:
Mono的

スマホ:
Mono

必須:
Mono互換性

業界標準:

全プロ:
120 Hz以下Mono

例外:
ほぼなし
```

### 実装方法

```
方法1: Utility (推奨)

全トラック:

Utility挿入
Bass Mono: On
Freq: 120 Hz

方法2: Master Track

Master:

Utility挿入
Bass Mono: On
Freq: 120 Hz

一括処理

方法3: Mid/Side EQ

EQ Eight:
Mode: Mid/Side

Side:
Low Cut 120 Hz

効果:
120 Hz以下 → Mid (Mono)
120 Hz以上 → Side (Stereo)

推奨:
方法1 (各トラック)
最も確実
```

---

## Mid/Side処理

**高度な技術:**

### 概念

```
Mid:

定義:
L + R
中央成分
Mono

Side:

定義:
L - R
側面成分
Stereo情報

Mid/Side EQ:

機能:
Mid・Side別にEQ

用途:

Mid:
存在感調整
Vocal・Lead

Side:
広がり調整
Pad・FX

Ableton:

EQ Eight:
Mode: Mid/Side選択可
```

### 実践例

```
Pad (広げる):

EQ Eight:
Mode: Mid/Side

Mid:
通常通り

Side:
High Shelf +2 dB @ 5 kHz
広がり強調

Vocal (明瞭に):

Mid:
Peak +3 dB @ 3 kHz
明瞭度

Side:
通常通り

Master (低域Mono):

Side:
Low Cut 120 Hz
低域Mono化
```

---

## ステレオ幅測定

**視覚的確認:**

### 測定方法

```
Goniometer:

表示:
L/R相関

形:

縦線:
Mono

丸:
Wide Stereo

横線:
位相問題

推奨:
やや縦長の楕円

Correlation Meter:

値:

+1.0: 完全Mono
+0.5-0.8: 良い
0.0: Wide
-1.0: 位相反転 (問題)

目標:
+0.4-0.7

Ableton:

なし (標準)

無料:
Youlean Loudness Meter
Correlation表示あり

有料:
iZotope Ozone Imager
```

---

## Mono互換性確認

**必須チェック:**

### 確認方法

```
方法1: Utility

Master:

Utility挿入
Width: 0%
Mono化

確認:

再生:
各トラック聴こえる？

問題:
消える・音量down

解決:
Bass Mono: On (120 Hz)

方法2: Headphone片耳

簡易:
片耳で聴く

確認:
バランス良い？

方法3: 実機

スマホ:
1スピーカー = Mono

車:
Mono的環境

確認:
実際の再生環境
```

---

## トラック別設定

**推奨値:**

### 詳細設定

```
Kick:

Width: 0% (完全Mono)
Pan: Center
Bass Mono: On (全帯域)

理由:
最もパワフル
中央定位

Bass:

Width: 0%
Pan: Center
Bass Mono: On

Snare/Clap:

Width: 0-10% (ほぼMono)
Pan: Center

Vocal:

Width: 0-10%
Pan: Center
存在感

Lead:

Width: 20-30%
Pan: Center
やや広がり

Pad:

Width: 80-100%
Pan: Center
広い

Hi-Hat:

Width: 40-60%
Pan: L/R Auto Pan
または
固定 L -30% / R +30%

Percussion:

Width: 60-80%
Pan: L/R振り分け

FX:

Width: 100-120%
Pan: L/R極端
```

---

## Haas Effect

**擬似ステレオ:**

### 原理

```
定義:

Mono信号:
L/R微妙に遅延差

結果:
Stereo感

設定:

Delay: 10-30 ms
片側のみ

効果:
広がり

注意:

Mono再生:
Comb Filter
音質劣化

推奨:
装飾のみ使用
Kick・Bass・Vocal不可
```

---

## よくある失敗

### 1. 低域Stereo

```
問題:
Mono再生で消える

原因:
120 Hz以下Stereo

解決:

全トラック:
Bass Mono: On (120 Hz)

確認:
Mono再生テスト

必須:
100%のプロジェクト
```

### 2. Width過剰

```
問題:
位相問題
不自然

原因:
Width 150%+

解決:

最大:
Width 120%

推奨:
80-100% (Pad・FX)

理由:
自然な広がり
```

### 3. Mono確認なし

```
問題:
Mono再生でバランス崩壊

原因:
確認忘れ

解決:

必須:
Mono確認

頻度:
ミックス完了前

方法:
Utility Width 0%
```

### 4. 全てWide

```
問題:
中央スカスカ
パワーなし

原因:
全トラックWide

解決:

Center:
Kick・Bass・Vocal・Lead

Wide:
Pad・FX・装飾のみ

バランス:
Center 60%
Wide 40%
```

---

## ステレオエンハンサー

**プラグイン活用:**

### 種類

```
1. Stereo Width:

機能:
Mid/Side調整

推奨:
iZotope Ozone Imager (無料)

2. Microshift:

機能:
わずかなDetune
Stereo感

推奨:
Soundtoys MicroShift

3. Doubler:

機能:
音声複製
Stereo配置

推奨:
Waves Doubler

注意:

使いすぎ:
不自然

推奨:
Ableton標準で十分
Utility + Chorus
```

---

## 実践ワークフロー

**30分で完成:**

### Step-by-Step

```
0-10分: Bass Mono設定

1. 全トラック Utility挿入
2. Bass Mono: On
3. Freq: 120 Hz
4. 確認

10-15分: Width設定

1. Kick・Bass: 0%
2. Snare: 0-10%
3. Lead: 20-30%
4. Pad: 80-100%
5. FX: 100-120%

15-20分: Panning

1. Hi-Hat: L/R
2. Percussion: L/R
3. FX: L/R極端
4. バランス確認

20-25分: 確認

1. Stereo再生
2. Mono再生 (Width 0%)
3. 比較

25-30分: 微調整

各設定微調整
完成
```

---

## まとめ

### Stereo Imaging

```
□ 120 Hz以下 必ずMono
□ Kick・Bass Width 0%
□ Pad・FX Width 80-120%
□ Mono互換性確認必須
□ Bass Mono: On (全トラック)
```

### Panning

```
Center: Kick・Bass・Vocal・Lead
L/R: Hi-Hat・Percussion・FX
バランス: Center 60%, Wide 40%
```

### 重要原則

```
□ 低域ほどMono
□ 高域ほどWide
□ Width過剰注意 (最大120%)
□ Mono確認必須
□ Correlation +0.4-0.7
```

---

**次は:** [Depth & Space](./depth-space.md) - Reverb・Delayで奥行きを作る
