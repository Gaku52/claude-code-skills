# Gain Staging

ヘッドルーム確保の基礎。Master -6 dB目標、適切な音量バランスを完全マスターします。

## この章で学ぶこと

- Gain Stagingとは何か
- Master -6 dB目標設定
- Utility活用法
- トラック別音量バランス
- Fader 0 dB vs -∞ dB
- Pink Noise法
- よくある失敗と対処法

---

## なぜGain Stagingが重要なのか

**ミキシングの土台:**

```
Gain Staging なし:

Master: 0 dB (クリッピング)
ダイナミクス: ゼロ
マスタリング: 不可能

結果:
音が潰れる
歪む

Gain Staging あり:

Master: -6 dB (ヘッドルーム)
ダイナミクス: 維持
マスタリング: 可能

結果:
クリア、太い

プロの真実:

Gain Staging:
ミキシング開始前
必須

理由:
全ての基礎
これなしは始まらない

使用頻度:
100%
全プロジェクト
```

---

## Gain Stagingとは

**適切な音量設定:**

### 基本概念

```
定義:

各段階で:
適切な音量レベル維持

目的:

1. ヘッドルーム確保:
   Master -6 dB以上

2. ノイズフロア回避:
   小さすぎない

3. ダイナミクス維持:
   圧縮避ける

デジタルオーディオ:

0 dBFS:
絶対最大

0 dBFS超え:
クリッピング
歪み

目標:
-6 dBFS (Master)

理由:

マスタリング:
+6 dB Gain可能

Limiter:
GR -6 dB余裕

ダイナミクス:
圧縮なし
```

---

## Master -6 dB目標

**なぜ-6 dBなのか:**

### 理由

```
マスタリング余裕:

現状:
Master -6 dB

Mastering:
Limiter Gain +6 dB

結果:
-14 LUFS達成

ダイナミクス保持:

-6 dB余裕:
Transient保持

0 dB:
圧縮される

エフェクト余裕:

EQ Boost:
+3 dB可能

Compressor Make-Up:
+4 dB可能

合計 +7 dB:
まだ -6 dB + 7 = +1 dB
→ クリッピング危険

だから:
-6 dB必要

業界標準:

ポップ: -6 dB
クラシック: -10 dB
EDM: -3 dB (aggressive)

推奨:
-6 dB (安全)
```

---

## Gain Staging手順

**10ステップ:**

### Step 1: 全トラックFader確認

```
開始前:

1. 全トラック選択

2. Fader位置確認:
   バラバラ?

3. 全て 0 dB にリセット:
   Option + クリック

理由:
統一した状態から開始
```

### Step 2: Master確認

```
再生:

全トラック:
Fader 0 dB

Master:
何dB?

例:
Master: +3 dB (赤)
→ クリッピング

Master: -2 dB
→ ヘッドルーム少ない

Master: -15 dB
→ 小さすぎ

目標:
Master: -6 dB
```

### Step 3: Utility挿入（全トラック）

```
方法:

1. Master Track選択

2. Utility挿入

3. Gain: 計算

例:

現在 Master: +3 dB
目標: -6 dB
差: -9 dB

全トラック Utility:
Gain: -9 dB

結果:
Master: -6 dB

効率化:

テンプレート作成:
Utility Gain -6 dB
全トラックにドロップ
```

### Step 4: Kick基準設定

```
Kick = 基準:

1. 全トラック Mute

2. Kick のみ Solo

3. 再生

4. Master確認:
   目標 -6 dB

5. Kick Fader調整:
   Master -6 dBになるまで

例:

Kick Fader: 0 dB
→ Master: -4 dB (大きすぎ)

Kick Fader: -2 dB
→ Master: -6 dB (完璧)

固定:
Kick Fader -2 dB
```

### Step 5: Bass追加

```
1. Bass Un-Mute

2. Kick + Bass再生

3. Master確認:
   目標 -6 dB維持

4. Bass Fader調整:
   Kickと同じくらい

例:

Bass Fader: 0 dB
→ Master: -3 dB (大きすぎ)

Bass Fader: -3 dB
→ Master: -6 dB (完璧)

バランス:
Kick -2 dB
Bass -3 dB
```

### Step 6: Drumsグループ追加

```
1. Snare・Hi-Hat Un-Mute

2. 再生

3. Master確認:
   目標 -6 dB

4. Snare Fader:
   Kickより小さく
   でも明確

5. Hi-Hat Fader:
   さらに小さく

例:

Snare: -6 dB
Hi-Hat: -12 dB
Master: -6 dB維持
```

### Step 7: Melody追加

```
1. Lead・Pad Un-Mute

2. 再生

3. バランス:

Lead: 前に出る
→ Fader -6 dB

Pad: 後ろに
→ Fader -12 dB

Master: -6 dB維持

調整:
各Fader微調整
```

### Step 8: Vocal追加

```
Vocal あり:

1. Vocal Un-Mute

2. 最前列:
   Kickと同じくらい

3. Fader調整:
   -6 dB程度

Master: -6 dB維持
```

### Step 9: 全体バランス

```
1. 全トラック再生

2. Master確認:
   -6 dB?

3. 微調整:
   各Fader ±1〜2 dB

4. 確認:
   全セクション (Intro・Verse・Drop)

5. 最も大きい部分:
   Master -6 dB
```

### Step 10: 保存

```
完成:

全トラック Fader設定完了
Master -6 dB

保存:
Cmd+S

次:
EQ・Compression開始
```

---

## Utility活用法

**Gain調整の最良ツール:**

### Utilityとは

```
機能:

Gain:
±35 dB

Width:
Stereo幅

Bass Mono:
低域Mono化

Phase:
位相反転

Pan:
左右

推奨:
各トラックに必ず挿入
最初のデバイス
```

### Utility配置

```
Chain順序:

1. Utility (Gain In)
   ↓
2. EQ Eight
   ↓
3. Compressor
   ↓
4. その他エフェクト
   ↓
5. Utility (Gain Out)

理由:

Gain In:
入力レベル調整

Gain Out:
出力レベル調整

メリット:
Fader触らない
Automation簡単
```

---

## Pink Noise法

**プロの技:**

### 原理

```
Pink Noise:

全周波数:
均等なエネルギー

特徴:
「シーーー」

用途:
音量バランス参照

方法:

1. Audio Track作成

2. Pink Noise生成:
   Generator > Pink Noise

3. Fader: -12 dB

4. 各トラック調整:
   Pink Noiseと同じくらい

5. Pink Noise削除

結果:
完璧なバランス
```

### 実践手順

```
Step 1: Pink Noise設定

1. 新規Audio Track
2. 名前: "Pink Noise Reference"
3. Looper:
   Pink Noise録音
   Loop: On
4. Fader: -12 dB
5. 再生

Step 2: トラック調整

1. Kick Solo + Pink Noise
2. Kick Fader調整:
   Pink Noiseと同じ音量感

3. Bass Solo + Pink Noise
4. Bass Fader調整

5. 全トラック繰り返し

Step 3: 削除

Pink Noise Track削除

確認:
Master -6 dB

メリット:

客観的:
感覚に頼らない

正確:
周波数バランス良い

プロ使用:
業界標準
```

---

## トラック別目標レベル

**参考値:**

### 標準設定

```
Kick:

Fader: -2 dB
Peak: -6 dB (Master)
最も大きい

Bass:

Fader: -3 dB
Peak: -9 dB
Kickの次

Snare/Clap:

Fader: -6 dB
Peak: -12 dB
明確だが控えめ

Hi-Hat:

Fader: -12 dB
Peak: -18 dB
聴こえる程度

Lead:

Fader: -6 dB
Peak: -12 dB
前に出る

Vocal:

Fader: -6 dB
Peak: -12 dB
Kick・Bassの次

Pad:

Fader: -12 dB
Peak: -18 dB
後ろに

FX:

Fader: -15 dB
Peak: -21 dB
装飾

ルール:

これは参考値
楽曲により変わる
大切なのは相対関係
```

---

## Fader 0 dB vs Utility

**どちらを使う？:**

### 比較

```
Fader使用:

メリット:
視覚的わかりやすい
直感的

デメリット:
Automation複雑
後で変更困難

Utility使用:

メリット:
Fader 0 dB維持
Automation簡単
後で変更簡単

デメリット:
1ステップ多い

推奨:

Utility:
Gain Staging

Fader:
最終バランス調整
Automation
```

---

## よくある失敗

**Gain Stagingの罠:**

### 1. Master 0 dB

```
問題:
クリッピング
歪み
マスタリング不可

原因:
ヘッドルームなし

解決:

全トラック Utility:
Gain -6 dB

Master:
-6 dB確保

理由:
必須
```

### 2. トラック小さすぎ

```
問題:
ノイズフロア近い
音質劣化

原因:
過剰にGain下げ

解決:

目標:

Individual Track:
-12 〜 -18 dB (Peak)

Master:
-6 dB (Peak)

ルール:
小さすぎもダメ
```

### 3. Faderバラバラ

```
問題:
管理困難
視覚的わかりにくい

原因:
計画なし

解決:

Fader:
なるべく 0 dB近く

Gain調整:
Utility使用

理由:
視覚的整理
```

### 4. セクションごとに差

```
問題:

Verse: -8 dB
Drop: -2 dB (大きすぎ)

原因:
Drop要素多い

解決:

Drop Gain下げ:
Master -6 dB維持

または:

Verse Gain上げ

目標:
全セクション -6 dB
```

---

## ダイナミクスレンジ

**適切な変化:**

### 概念

```
ダイナミクスレンジ:

最大音量:
Kickのピーク

最小音量:
Padの平均

差:

適切: 12-18 dB
狭すぎ: 6 dB (圧縮過剰)
広すぎ: 24 dB (バランス悪い)

Techno/House:

ダイナミクス:
やや狭い (8-12 dB)

理由:
クラブ再生
大音量
```

---

## Metering

**測定ツール:**

### 必須メーター

```
Peak Meter:

表示:
瞬間最大値

単位:
dBFS

目標:
-6 dBFS

RMS Meter:

表示:
平均音量

目標:
-18 〜 -12 dBFS

LUFS Meter:

表示:
知覚ラウドネス

目標:
-20 〜 -16 LUFS (Mix)

推奨ツール:

Ableton:
Master Track Meter

無料:
Youlean Loudness Meter

有料:
iZotope Insight
```

---

## 実践ワークフロー

**30分で完成:**

### Step-by-Step

```
0-5分: 準備

1. 全トラック Fader 0 dB
2. 再生・Master確認
3. 全トラック Utility挿入

5-10分: Kick・Bass

1. Kick Solo
2. Master -6 dB調整
3. Bass追加
4. サイドチェイン確認

10-15分: Drums

1. Snare追加
2. Hi-Hat追加
3. Percussion追加
4. バランス確認

15-20分: Melody

1. Lead追加
2. Pad追加
3. 前後関係確認

20-25分: Vocal・FX

1. Vocal追加 (あれば)
2. FX追加
3. 全体バランス

25-30分: 確認

1. 全セクション再生
2. Master -6 dB確認
3. 保存
```

---

## まとめ

### Gain Staging

```
□ Master -6 dB目標
□ Kick基準で調整
□ Utility活用
□ Fader 0 dB維持 (Utility使用時)
□ 全セクション確認
```

### 目標値

```
Master Peak: -6 dBFS
Individual Track: -12 〜 -18 dBFS
ダイナミクスレンジ: 12-18 dB
```

### 重要原則

```
□ ミキシング最初のステップ
□ これなしは始まらない
□ ヘッドルーム必須
□ クリッピング厳禁
□ 全トラックバランス
```

---

**次は:** [Frequency Balance](./frequency-balance.md) - 周波数分離でクリアなミックスを実現
