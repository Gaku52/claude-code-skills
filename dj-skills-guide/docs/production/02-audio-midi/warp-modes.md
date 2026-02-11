# Warp機能

Ableton Live最強の機能。異なるBPMのサンプルを完璧に同期させる魔法をマスターします。

## この章で学ぶこと

- Warpとは何か
- Warp Mode 6種類の使い分け
- Warp Marker配置テクニック
- タイミング修正方法
- ボーカルWarp完全ガイド
- ドラムループWarp
- トラブルシューティング

---

## なぜWarpが重要なのか

**Ableton Liveの代名詞:**

```
Warpなしでは:

サンプルのBPM:
バラバラ

プロジェクト: 128 BPM
サンプルA: 140 BPM
サンプルB: 120 BPM

→ 使えない

Warpあり:

全て 128 BPM に統一:
自動同期
完璧にグリッドに合う

結果:
どんなサンプルも使える

プロの制作:

Warp = 必須スキル
99%のサンプルをWarp
使わない理由がない

他DAW:

Logic: Flex Time
FL Studio: Time Stretching
Cubase: AudioWarp

Abletonが最強:
最も高音質
最も簡単
リアルタイム対応
```

---

## Warpとは

**BPM同期の魔法:**

### 基本概念

```
Warpの仕組み:

元のサンプル:
140 BPM

プロジェクト:
128 BPM

Warp処理:
140 BPM → 128 BPM
タイムストレッチ

結果:
テンポは変わる
ピッチは変わらない (デフォルト)

従来の方法:

ピッチ変更:
BPM変わる = ピッチも変わる
140 BPM → 128 BPM = 音が低くなる

Warp:
BPM だけ変わる
ピッチは保持
```

### Warp On/Off

```
Clip View > Warp:

☑ Warp (On)
プロジェクトBPMに同期

☐ Warp (Off)
元のテンポで再生

Auto-Warp:

Preferences > Record Warp Launch:
Auto-Warp Long Samples: ☑

効果:
長いサンプル (30秒以上) を
自動でWarp

短いサンプル (Kick等):
自動Warpなし
手動でOnにする
```

---

## Warp Mode 6種類

**音源別の使い分け:**

### 1. Beats Mode

```
用途:
ドラムループ
パーカッション
リズム重視

特徴:
トランジェント (アタック) を保持
グリッドに吸着

設定:

Preserve:
Transients (推奨)
または
Transients + Envelope

Granulation Resolution:
1/8 (デフォルト)

音質:
ドラムに最適
クリアなアタック

使用例:

Kick Loop: Beats
Hi-Hat Loop: Beats
Percussion: Beats
Drum Break: Beats

NG:

ボーカル: × (不自然)
ベース: △ (ケースバイケース)
```

### 2. Tones Mode

```
用途:
メロディ
ハーモニー
音程重視

特徴:
ピッチを保持
音楽的

設定:

Grain Size:
30-60 (デフォルト 44)

音質:
メロディに最適
自然な音程

使用例:

ボーカルメロディ: Tones
ストリングス: Tones
ブラス: Tones
ピアノ: Tones

NG:

ドラム: × (アタック鈍る)
```

### 3. Texture Mode

```
用途:
アンビエント
パッド
テクスチャ

特徴:
滑らかな変化
グラニュラー

設定:

Grain Size:
20-100

Flux:
0-100 (ランダム性)

音質:
柔らかい
エフェクト的

使用例:

Pad: Texture
アンビエント: Texture
SFX: Texture
ドローン: Texture

NG:

ドラム: × (ぼやける)
ボーカル: △ (特殊効果なら○)
```

### 4. Re-Pitch Mode

```
用途:
ターンテーブル的

特徴:
BPM変更 = ピッチ変更
Warpなし状態

設定:
なし (シンプル)

音質:
アナログ的
自然

使用例:

ヴィンテージ効果: Re-Pitch
DJ的なピッチ変更: Re-Pitch

NG:

通常の制作: × (BPM同期しない)
```

### 5. Complex Mode

```
用途:
複雑な音源
マルチトラック

特徴:
高音質
CPU負荷高い

設定:

なし (自動)

音質:
最高品質
万能

使用例:

フルミックス: Complex
ステムファイル: Complex
複雑なループ: Complex

推奨:

迷ったら Complex:
高音質保証

CPU余裕:
Complexが最強
```

### 6. Complex Pro Mode

```
用途:
Complexの上位版
ピッチ変更に対応

特徴:
最高音質
Formant保持可能

設定:

Formants:
100 (デフォルト)
ボーカルの音色保持

Envelope:
100 (デフォルト)

音質:
最高峰
CPU負荷最大

使用例:

ボーカル + Transpose: Complex Pro
重要なサンプル: Complex Pro

注意:
CPU重い
必要な場合のみ
```

---

## Warp Mode選択チャート

**一発で選ぶ:**

```
┌─────────────────────┐
│ 何をWarpする？      │
└─────────────────────┘
          │
    ┌─────┴─────┐
    │           │
 ドラム?      その他?
    │           │
   Beats    ┌───┴───┐
            │       │
         メロディ? パッド?
            │       │
          Tones  Texture
            │
        ┌───┴───┐
        │       │
    ボーカル? 楽器?
        │       │
    Complex   Tones
     Pro

実用的には:

ドラム: Beats
ボーカル: Complex Pro (Transpose時)
         Complex (そのまま)
メロディ楽器: Tones
パッド: Texture
フルミックス: Complex
迷ったら: Complex
```

---

## Warp Marker

**細かいタイミング調整:**

### Warp Markerとは

```
定義:
波形上の基準点

用途:
タイミング修正
手動グリッド合わせ

表示:

Clip View > Waveform:

     ↓ ← Warp Marker
~~~~~|~~~~~/\~~~~~~
     1.1.1

黄色い縦線:
グリッド位置

操作:

ダブルクリック:
Warp Marker 作成

ドラッグ:
位置調整

Cmd+クリック:
削除
```

### Warp Marker配置

```
基本フロー:

1. Warp: On

2. 最初のビートを探す

3. その位置で ダブルクリック
   → Warp Marker 作成

4. その Marker を 1.1.1 にドラッグ

5. 次のビート (1.2.1) を探す

6. ダブルクリック → ドラッグ

7. 繰り返し

自動配置:

多くの場合:
Abletonが自動で配置

手動調整:
ズレている箇所のみ修正

複数Marker:

ドラムループ:
4-8箇所

ボーカル:
フレーズごと
```

---

## 実践: ドラムループをWarp

**Step by Step:**

### Step 1: サンプル読み込み (2分)

```
1. Audio Track 作成

2. ドラムループ配置:
   Browser > Drums > Loops
   または
   ファイルをドラッグ

3. Clip View 開く:
   ダブルクリック

4. Warp 確認:
   ☑ Warp
```

### Step 2: BPM検出 (5分)

```
Clip View > Seg. BPM:

表示: 140.00
(サンプルの元BPM)

プロジェクトBPM:
128.00

確認:

再生:
Space

聴く:
グリッドに合っているか

ズレている場合:

Seg. BPM 調整:
139.50 に変更
または
Warp Marker 手動配置
```

### Step 3: Warp Mode選択 (1分)

```
Warp Mode:
Beats (ドラムなので)

Preserve:
Transients

再生:
確認

音質チェック:
アタックが保持されている?
```

### Step 4: 細かい調整 (10分)

```
ズレている箇所:

1. 波形を見る

2. ビートの位置を確認

3. その位置で ダブルクリック
   → Warp Marker

4. グリッド線にドラッグ

5. 再生して確認

6. 全体が合うまで繰り返し

完璧になったら:
Cmd+S で保存
```

---

## ボーカルWarp

**特殊なテクニック:**

### 準備

```
ボーカルサンプル:

元BPM: 不明
フリーテンポ

プロジェクト: 128 BPM

目標:
128 BPM に合わせる
音程は保持
```

### Warp手順

```
1. Warp: On

2. Warp Mode: Complex Pro
   Formants: 100

3. 最初の歌い出しを見つける

4. その位置を 1.1.1 に合わせる:
   Warp Marker ドラッグ

5. 次のフレーズ (例: 3.1.1)

6. そこにもWarp Marker

7. ドラッグで位置合わせ

8. フレーズごとに繰り返し

9. 全体を再生して確認

Transpose併用:

元のキー: Em
プロジェクト: Am

Transpose: +7 st

Complex Pro使用:
Formants 保持
自然な音
```

---

## よくある質問

### Q1: Warpすると音質劣化する？

**A:** Mode選択次第

```
高音質Mode:

Complex Pro: 最高
Complex: 非常に良い
Tones: 良い

低音質Mode:

Texture: 特殊効果向き
(音質劣化が特徴)

推奨:

重要なサンプル: Complex Pro
通常: Complex または Tones
ドラム: Beats

CPU余裕あれば:
全てComplex Pro
```

### Q2: 自動Warpがズレる

**A:** 手動で修正

```
原因:

BPM検出失敗:
複雑なリズム
フリーテンポ

解決:

1. Seg. BPM 手動入力:
   元のBPMが分かる場合

2. Warp Marker 手動配置:
   4-8箇所に配置

3. 1つずつ確認:
   グリッドに合わせる

時間:
最初は15分
慣れたら5分
```

### Q3: Warpした後Transposeすると変？

**A:** Complex Pro使用

```
問題:

Warp + Transpose:
音質劣化
不自然

解決:

Warp Mode: Complex Pro

Formants: 100

効果:
音色保持
自然なTranspose

特にボーカル:
必ずComplex Pro
```

---

## まとめ

### Warp Mode選択

```
ドラム: Beats
ボーカル: Complex Pro
メロディ: Tones
パッド: Texture
複雑: Complex
迷ったら: Complex
```

### Warp手順

```
1. Warp: On
2. Mode 選択
3. BPM 確認
4. Warp Marker 配置
5. 再生確認
6. 微調整
7. 保存
```

### 重要ポイント

```
□ Auto-Warp設定
□ Mode使い分け
□ Warp Marker配置
□ Complex Pro for ボーカル
□ 音質確認
```

---

**次は:** [Quantize](./quantization.md) - タイミングを完璧に
