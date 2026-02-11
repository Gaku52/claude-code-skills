# エディット・リミックス

**既存楽曲を改変・リミックスする技術を完全マスター**

エディットとリミックスは、既存の楽曲を自分なりに再構成・改変するクリエイティブな手法です。DJセット用のブートレグ、オフィシャルリミックス、マッシュアップまで、Ableton Liveを使った実践的なテクニックを学びます。

---

## この章で学ぶこと

- ✅ エディット vs リミックスの違い
- ✅ Abletonでのステム分離
- ✅ BPM変更とピッチ調整
- ✅ セクションの再構成
- ✅ エフェクトによる変化
- ✅ オリジナル要素の追加
- ✅ 法的な注意点（著作権）

**学習時間**: 4-6時間
**難易度**: ★★★★☆ 中上級

---

## エディット vs リミックス

### エディット（Edit）

**定義**:
```
= 原曲の構成やBPMを変更するが、音源は原曲のまま

変更内容:
  - BPM調整
  - セクション並び替え
  - Intro/Outro延長
  - 不要部分カット

追加要素:
  - 最小限（FX、フィルター程度）
```

**例**:
```
Original: 110 BPM、3分30秒
Edit: 128 BPM、5分（32小節Intro追加）
```

### リミックス（Remix）

**定義**:
```
= 原曲の一部要素を使い、新しい楽曲を制作

使用要素:
  - ボーカル（アカペラ）
  - メロディの一部
  - サンプル

追加要素:
  - 新しいドラム
  - 新しいベースライン
  - 新しいコード進行
```

**例**:
```
Original: Pop 110 BPM
Remix: House 125 BPM、完全に新しいトラック（ボーカルのみ原曲）
```

---

## Abletonでのエディット

### Step 1: 原曲インポート

```
1. File → Import Audio
2. 原曲WAVファイルをドラッグ
3. Audio Trackに配置
```

### Step 2: Warp設定

**BPM検出**:
```
1. Clip View → Warp: On
2. Segment BPM: Abletonが自動検出
3. 確認: タップテンポで検証
4. Warp Markerを小節頭に配置
```

**Warp Mode選択**:
```
ボーカル曲: Complex Pro（最高品質）
ドラム: Beats
シンセ/パッド: Texture

→ 音質劣化を最小限に
```

### Step 3: BPM変更

**例: 110 BPM → 128 BPM**:
```
1. Clip Warp: On
2. Segment BPM: 110（原曲）
3. Project Tempo: 128（新BPM）

→ Abletonが自動でストレッチ
```

**ピッチ維持**:
```
Warp Mode: Complex Pro
→ BPMは変わるが、ピッチ（キー）は変わらない
```

### Step 4: セクション再構成

**Arrangement Viewで編集**:
```
Original構成:
  Intro (8小節) → Verse → Chorus → Verse → Chorus → Outro

Edit構成（DJ用）:
  32小節 Intro → Chorus → Breakdown → Chorus → 32小節 Outro

手順:
  1. 不要なVerse削除
  2. Chorusを複製
  3. Introを32小節に延長（ループ）
```

**Intro延長**:
```
1. Intro部分（8小節）を選択
2. Cmd+D で複製×4回
3. 合計32小節のIntro完成
```

---

## リミックス制作

### Step 1: アカペラ入手

**公式ステム**:
```
- Beatport "Stems" セクション
- アーティスト公式サイト
- レーベル提供（リミックスコンテスト）
```

**自分で分離（非推奨、音質劣化）**:
```
- Spleeter（AI分離）
- iZotope RX（有料、高品質）
```

### Step 2: 新しいトラック制作

**Acapellaに合わせる**:
```
1. Acapella分析:
   - BPM: 110
   - Key: C Major

2. 新トラック設定:
   - BPM: 125（House）
   - Key: C Major（同じキー）
```

**ドラム制作**:
```
Houseドラム:
  - Kick: 4つ打ち
  - Snare: 2拍目、4拍目
  - Hi-Hat: 16分音符

→ Acapellaを引き立てる
```

**ベースライン**:
```
Original Key: C Major
Bassline: C2（ルート音）、4つ打ち

→ シンプルに
```

**コード進行**:
```
House定番: C - Am - F - G
Pad音色: Warm Analog Pad

→ Acapellaを支える
```

### Step 3: Acapellaの配置

**タイミング調整**:
```
1. Acapellaを Arrangement Viewに配置
2. Warp: On、BPM 110 → 125
3. 小節頭に整列
4. ボーカルのフレーズとドラムが合うように調整
```

**EQ処理**:
```
EQ Eight:
  1. High Pass 100 Hz（ランブル除去）
  2. Boost 3-5 kHz +2 dB（明瞭さ）

Compressor:
  - Ratio: 3:1
  - Threshold: -15 dB
```

---

## マッシュアップ

### 2曲を組み合わせる

**選曲**:
```
Track A: Acapella（ボーカル）
Track B: Instrumental（インスト）

条件:
  - 同じキー または 隣接キー
  - 近いBPM（±5 BPM）
```

**例**:
```
Track A: "Song X" Acapella
  - BPM: 120
  - Key: A Minor

Track B: "Song Y" Instrumental
  - BPM: 122
  - Key: C Major（A Minorの相対キー）

→ 相性◎
```

### Ableton での設定

```
Track 1: Song X Acapella
  - Warp: On、BPM 120 → 122

Track 2: Song Y Instrumental
  - Warp: On、BPM 122（そのまま）

Project Tempo: 122 BPM

→ 両方が自動で同期
```

---

## キーとピッチの調整

### キー変更

**Transpose**:
```
例: A Minor → G Minor（-2 semitones）

Clip View → Transpose: -2

→ キーが変わる、音質劣化なし（Complex Pro使用時）
```

### ピッチ補正

**Vocal Tuner**（Ableton Suite）:
```
Acapellaに適用:
  - Reference: C Major
  - Correction: 50-80%

→ 微妙な音程のずれを修正
```

---

## エフェクトによる変化

### Filter Sweep

```
Auto Filter:
  - Cutoff: 200 Hz → 8000 Hz（16小節で変化）
  - Resonance: 30%

用途:
  - Intro/Outroで徐々に明るく
  - ブレイクダウン→ドロップ
```

### Reverb Throw

```
Return Track: Reverb
  - Decay: 8.0s（非常に長い）

Acapellaの最後の一言だけSend 100%:
  - 劇的な残響効果
  - トランジションに最適
```

### Stutter Effect

```
Beat Repeat:
  - Repeat: 1/16
  - Gate: 50%

特定の1拍だけ適用:
  - スタッタリング効果
  - Hip Hop、EDM
```

---

## 法的注意点

### 著作権

**許可なしの使用**:
```
✅ OK:
  - DJセット内でのみ使用（非販売）
  - SoundCloud "Free Download"（個人使用）
  - 練習目的

❌ NG:
  - 販売（iTunes、Beatport）
  - ストリーミング配信（Spotify）
  - YouTube収益化
```

### クレジット表記

```
必須:
  - 原曲アーティスト名
  - 原曲タイトル

例:
  "Original Artist - Song Title (Your Name Remix)"
  "Song A vs Song B (Your Name Mashup)"
```

### 公式リミックス

**レーベル許可あり**:
```
✅ 全てOK:
  - 販売
  - ストリーミング配信
  - リミックスコンテスト優勝

契約:
  - レーベルと契約書
  - ロイヤリティ配分
```

---

## 練習プロジェクト

### プロジェクト1: BPMエディット

```
難易度: ★☆☆☆☆

1. お気に入りの曲（例: 100 BPM）
2. BPMを128に変更
3. 32小節Intro追加
4. DJセットで使用

完成目標: DJ用エディット版
```

### プロジェクト2: マッシュアップ

```
難易度: ★★★☆☆

1. Acapella入手（同じキー）
2. Instrumental入手
3. Abletonで組み合わせ
4. EQ/エフェクトで仕上げ

完成目標: 2分マッシュアップ
```

### プロジェクト3: フルリミックス

```
難易度: ★★★★☆

1. 公式ステム入手（リミックスコンテスト）
2. 新しいドラム制作
3. 新しいベースライン
4. オリジナルメロディ追加
5. Acapellaを配置
6. ミキシング・マスタリング

完成目標: 5分フルリミックス、コンテスト応募
```

---

## まとめ

### エディット・リミックスの核心

1. **エディット**: BPM/構成変更、原音源そのまま
2. **リミックス**: Acapella+新トラック
3. **Warp**: Complex Proで高音質
4. **著作権**: 個人使用のみ、販売NG（許可なし）
5. **クレジット**: 原曲アーティスト必須

### 次のステップ

1. [ライブプロダクション](./live-production.md) - Session Viewでのパフォーマンス
2. [Ableton for DJing](./ableton-for-djing.md) - DJとしてAbletonを使う

---

**🎵 既存楽曲を自分なりに再構成して、セットを差別化しましょう！**
