# DJ用トラック制作

**DJセット用のオリジナルトラックとツールを作成する**

DJ用トラックの制作は、通常の楽曲制作とは異なる特別な配慮が必要です。32小節のIntro/Outro、ミックスポイントの設計、DJフレンドリーな構成など、実際のDJセットで使用することを前提とした制作テクニックを学びます。

---

## この章で学ぶこと

- ✅ DJ対応 Intro/Outroの作り方（32小節必須）
- ✅ ミックスポイントの設計
- ✅ アカペラ・インストゥルメンタルの分離
- ✅ DJ用ループ・フィルインの制作
- ✅ トランジションツールの作成
- ✅ ブートレグ・エディットの作り方

**学習時間**: 3-5時間
**難易度**: ★★★☆☆ 中級

---

## DJ用トラックの特徴

### 通常のトラックとの違い

**通常のトラック**:
```
Intro: 8-16小節（短い）
構成: Intro → Verse → Chorus → Outro
Outro: 8小節
総尺: 3-4分

→ リスナー向け、単体で完結
```

**DJ用トラック**:
```
Intro: 32小節（長い、ミックス専用）
構成: Intro → Buildup → Drop → Breakdown → Drop → Outro
Outro: 32小節
総尺: 5-8分

→ DJ向け、ミックス前提
```

---

## Intro/Outro の設計

### 32小節ルール

**なぜ32小節か**:
```
DJのミックス時間:
  - 最短: 16小節（2フレーズ）
  - 標準: 32小節（4フレーズ）
  - 長め: 64小節（8フレーズ）

32小節 = 4フレーズ:
  - フレーズ1: 前の曲をフェードアウト開始
  - フレーズ2: 新曲のキックのみ
  - フレーズ3: ベースライン追加
  - フレーズ4: フル展開へ

→ 十分な時間でスムーズなミックス
```

### Intro の要素追加順序

**Bar 1-8 (フレーズ1): キックのみ**
```
Kick: 4つ打ち
BPM: 一定（重要！）
音量: -6 dB (Peak)

→ 前の曲と重ねやすい
```

**Bar 9-16 (フレーズ2): キック + ハイハット**
```
+ Hi-Hat: 16分音符
→ リズムが明確に
```

**Bar 17-24 (フレーズ3): + ベースライン**
```
+ Sub Bass: ルート音のみ
→ 低域が埋まる
```

**Bar 25-32 (フレーズ4): + パッド/シンセ**
```
+ Pad: コード進行
+ Lead Teaser: メロディの一部

→ フル展開の準備
```

**Bar 33〜: メインセクション**
```
全要素: ドロップ
```

### Outro の要素削除順序

**逆順で削除**:
```
Bar 1-8: 全要素 → パッド/リード削除
Bar 9-16: → ベースライン削除
Bar 17-24: → ハイハット削除
Bar 25-32: キックのみ

→ Introの逆
```

---

## ミックスポイントの設計

### ループポイント

**Introのループポイント**:
```
Bar 9-16 (キック+ハイハット) をループ可能に:
  - Bar 9と17が完全に繋がる
  - DJが好きなタイミングでミックス開始

設定方法（Ableton）:
  - Clip Loop: On
  - Loop Start: Bar 9
  - Loop End: Bar 17
```

### クオンタイズ（グリッド整列）

**DJ用トラックは完璧なグリッド必須**:
```
Warp設定:
  - Warp: On
  - BPM: 正確に設定（例: 128.00）
  - Warp Marker: 小節頭に正確に配置

理由:
  - Rekordboxのビートグリッドが正確に
  - Sync機能が正しく動作
  - ベ��トマッチングが容易
```

---

## アカペラ・インストの分離

### なぜ必要か

**DJ用途**:
```
Acapella（ボーカルのみ）:
  - マッシュアップ
  - 他のトラックのインストに重ねる

Instrumental（ボーカル抜き）:
  - ボーカルトラックの下に敷く
  - リミックスのベース
```

### 制作段階で分離

**トラック構成**:
```
Track 1-8: インストゥルメンタル
  - Kick, Bass, Synth, Pad など

Track 9-12: ボーカル
  - Lead Vocal
  - Backing Vocal
  - Harmony
```

**エクスポート**:
```
1. インストゥルメンタル版:
   - Track 9-12をミュート
   - Export: "Track Name (Instrumental).wav"

2. アカペラ版:
   - Track 1-8をミュート
   - Export: "Track Name (Acapella).wav"

3. フル版:
   - 全トラック有効
   - Export: "Track Name (Original Mix).wav"
```

---

## DJ用ループ・フィルイン

### ドラムループ（8-16小節）

**シンプルなドラムループ**:
```
BPM: 128
構成:
  - Kick: 4つ打ち
  - Snare: 2拍目、4拍目
  - Hi-Hat: 16分音符

用途:
  - トランジション
  - ブレイクダウン補強
  - フィルイン
```

**エクスポート**:
```
Loop: 8小節または16小節
ファイル名: "Drum Loop 128 BPM - Your Name.wav"
配布: SoundCloud、Bandcamp
```

### フィルイン（2-4小節）

**スネアロール**:
```
Bar 1-2: 通常のドラム
Bar 3: スネア連打（1/8音符）
Bar 4: スネアロール（1/16音符）

→ ビルドアップ用
```

**ライザー（Riser）**:
```
White Noise + Filter Sweep:
  - Bar 1: Filter Cutoff 200 Hz
  - Bar 4: Filter Cutoff 8000 Hz
  - Pitch: +12 semitones

→ 緊張感、ビルドアップ
```

---

## トランジションツール

### Filter Sweep ループ

**作り方**:
```
1. シンプルなコード進行（Am-F-C-G）
2. Wavetable Pad
3. Auto Filter:
   - LFO Rate: 1/4
   - Amount: 80%
4. 16小節ループ

用途:
  - 2曲の間に挟む
  - エネルギーレベル調整
```

### Ambient Pad

**作り方**:
```
1. Analog Pad
2. Long Release: 4000 ms
3. Reverb: Hall 4.0s
4. コード1つを32小節維持

用途:
  - ブレイクダウン
  - アンビエントトランジション
```

---

## ブートレグ・エディット

### 合法的な範囲

**ブートレグ（Bootleg）**:
```
= 非公式リミックス/エディット

合法的な使用:
  - DJセット内でのみ使用（非販売）
  - SoundCloudに「Free Download」
  - 原曲クレジット必須

違法:
  - 販売（iTunes、Beatportなど）
  - ストリーミング配信（Spotify）
```

### エディット例

**BPM変更**:
```
Original: 110 BPM (Hip Hop)
Edit: 128 BPM (House)

Ableton Warp:
  - Complex Pro（ボーカル）
  - Beats（ドラム）
  - Segment BPM: 128
```

**構成変更**:
```
Original:
  Intro → Verse → Chorus → Verse → Chorus → Outro

Edit（DJ用）:
  32小節 Intro → Chorus → Breakdown → Chorus → 32小節 Outro

→ セット内で使いやすい
```

---

## 実践ワークフロー

### Step 1: テンプレート作成

```
New Project: "DJ Tool Template 128 BPM"

Track構成:
  1. Kick
  2. Snare/Clap
  3. Hi-Hat
  4. Sub Bass
  5. Mid Bass
  6. Pad
  7. Lead
  8. FX

Return Tracks:
  A. Reverb (Hall 2.5s)
  B. Delay (1/8)

Tempo: 128 BPM
```

### Step 2: 32小節 Intro作成

```
Bar 1-8: Kick
Bar 9-16: + Hi-Hat
Bar 17-24: + Bass
Bar 25-32: + Pad/Lead Teaser
Bar 33〜: Full Drop
```

### Step 3: メインセクション

```
Bar 33-96: メロディ、ドロップ、ブレイクダウン
```

### Step 4: 32小節 Outro作成

```
逆順で要素削除
最後はキックのみ
```

### Step 5: エクスポート

```
Master:
  - "Track Name (Original Mix).wav"
  - 44.1 kHz, 16bit

Instrumental:
  - "Track Name (Instrumental).wav"

Acapella（あれば）:
  - "Track Name (Acapella).wav"
```

---

## キーとBPMの選択

### ジャンル別BPM

```
Techno: 125-135 BPM
House: 120-128 BPM
Progressive House: 125-130 BPM
Trance: 136-142 BPM
Dubstep: 140 BPM (Half-time 70)
Drum & Bass: 170-180 BPM
```

### ハーモニックミキシング対応

**キー設定**:
```
Mixed In Keyで分析される前提:
  - キー: Cm, Am, F# Minorなど明確に
  - スケール内で完結
  - 調号変更は避ける

理由:
  - DJがハーモニックミキシングしやすい
  - Camelot Wheelで隣接キー
```

---

## 配布とプロモーション

### SoundCloud無料配布

```
1. SoundCloudにアップロード
2. 「Enable Download」にチェック
3. タグ:
   - #FreeDownload
   - #DJTool
   - #Techno (ジャンル)

説明文:
  "Free Download for DJs!
  BPM: 128
  Key: A Minor
  Use in your sets, tag me @YourName"
```

### Bandcamp

```
価格: $0 (Free) または Name Your Price
ファイル: WAV + MP3
説明: DJ用、BPM/キー記載
```

---

## よくある失敗と対処法

### 失敗1: Introが短すぎる

**対処法**:
```
最低32小節
理想は64小節（余裕のあるミックス）
```

---

### 失敗2: BPMがずれている

**対処法**:
```
Warp設定を正確に
グリッドが小節頭に一致
Rekordboxで確認
```

---

### 失敗3: キーが不明瞭

**対処法**:
```
1つのスケール内で完結
Mixed In Keyで分析
メタデータにキー記載
```

---

## まとめ

### DJ用トラック制作の核心

1. **32小節 Intro/Outro**: 必須
2. **正確なBPM**: Warp設定
3. **明確なキー**: ハーモニックミキシング対応
4. **ループポイント**: ミックスしやすく
5. **アカペラ/インスト**: 分離エクスポート

### DJとして

- セットで使うトラックの構造を理解
- ミックスポイントがどこにあるか知っている

### プロデューサーとして

- DJフレンドリーなトラック制作
- 自分のセットで使える武器
- 他のDJにもプレイされる

### 次のステップ

1. [エディット・リミックス](./edits-remixes.md) - 既存曲を改変
2. [Ableton for DJing](./ableton-for-djing.md) - DJセット構築
3. [制作者のためのDJ知識](./production-for-djs.md) - DJ視点の制作

---

**🎵 DJセット用のオリジナルトラックを作成して、セットを差別化しましょう！**
