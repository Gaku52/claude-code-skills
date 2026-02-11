# Sampler

サンプルを楽器に。ワンショット、ループ、ボーカルチョップまで、サンプリングの全てをマスターします。

## この章で学ぶこと

- Samplerの基礎
- サンプル読み込み
- Zone設定とキーマップ
- ループポイント設定
- フィルター・エンベロープ
- ワンショット加工
- ボーカルチョップ作成
- Simpler との違い

---

## なぜSamplerが重要なのか

**無限の音源:**

```
サンプリングの力:

既存の音:
ドラムサンプル
ボーカル
楽器
効果音

→ 楽器化:
鍵盤で演奏可能
ピッチ変更
エフェクト

用途:

ドラムキット:
Kick, Snare等配置

ボーカルチョップ:
ボーカルを細かく切る
再構成

メロディ:
1音から和音作成

FX:
効果音を楽器に

プロの使用:

Hip Hop: 80%
サンプリング中心

Techno: 40%
ドラム、FX

House: 50%
ボーカル処理

理由:

オリジナル音色:
誰も持っていない

リアル:
実際の音源

効率:
既存音の活用
```

---

## Sampler vs Simpler

**2つのサンプラー:**

### 違い

```
Simpler:

シンプル:
基本機能のみ

1サンプル:
1つの音だけ

用途:
ドラム
ワンショット

Live Standard: ✅

Sampler:

高機能:
複雑な設定

複数サンプル:
鍵盤ごとに異なる音

Zone設定:
ベロシティ、キー範囲

Modulation:
複雑なルーティング

Live Standard: ✅
Live Suite: より高機能版

推奨:

初心者:
Simpler

ドラム:
Simpler (Drum Rack内)

複雑なマップ:
Sampler

ボーカルチョップ:
Sampler
```

---

## Samplerの構造

**多機能サンプラー:**

### 全体像

```
┌────────────────────────────┐
│ Sample (サンプル)          │
│ WAV/AIFF読み込み           │
└──────────┬─────────────────┘
           │
┌──────────▼─────────────────┐
│ Zone Mapping               │
│ キー範囲、ベロシティ範囲    │
└──────────┬─────────────────┘
           │
┌──────────▼─────────────────┐
│ Pitch & Loop               │
│ Transpose, Loop Point      │
└──────────┬─────────────────┘
           │
┌──────────▼─────────────────┐
│ Filter                     │
│ Cutoff, Resonance          │
└──────────┬─────────────────┘
           │
┌──────────▼─────────────────┐
│ Amplifier & Effects        │
└────────────────────────────┘

制御:

Envelopes:
Filter, Amp

LFO:
Modulation

Velocity:
強弱で音色変化
```

### インターフェイス

```
上部 (Zone View):
┌────────────────────────────┐
│ 鍵盤マップ表示              │
│ Zone配置                   │
└────────────────────────────┘

中央左 (Sample):
┌────────────────────────────┐
│ 波形表示                   │
│ Start, End, Loop設定       │
└────────────────────────────┘

中央右 (Filter):
┌────────────────────────────┐
│ Filter Type, Cutoff        │
│ Resonance                  │
└────────────────────────────┘

下部 (Modulation):
┌────────────────────────────┐
│ Envelopes, LFO             │
│ Modulation Matrix          │
└────────────────────────────┘
```

---

## サンプル読み込み

**基本操作:**

### 読み込み方法

```
方法1: ドラッグ&ドロップ

1. Finderから:
   WAV/AIFFファイル

2. Samplerにドロップ

3. 自動的に読み込み

方法2: Browser

1. Sampler上部:
   Sample タブ

2. Browser表示

3. サンプル選択

4. ダブルクリック

方法3: Hot-Swap

1. Sampler上部:
   フォルダアイコン

2. Browser表示

3. サンプル入れ替え

推奨:

初心者:
ドラッグ&ドロップ

大量:
Browser
```

### サンプルの種類

```
One-Shot (ワンショット):

特徴:
短い
ループなし

例:
Kick
Snare
FX

設定:
Loop: Off

Loop Sample:

特徴:
繰り返し

例:
ベースライン
パッド
アンビエント

設定:
Loop: On
Loop Point調整

Multi-Sample:

特徴:
複数サンプル
鍵盤ごと

例:
ピアノ
ストリングス

設定:
Zone配置
```

---

## Zone設定

**キーマップ:**

### Zone View

```
表示:
Zone タブ

鍵盤表示:
┌─────┬─────┬─────┬─────┐
│Zone1│Zone2│Zone3│Zone4│
│C1-D1│D#1  │E1-F1│F#1  │
└─────┴─────┴─────┴─────┘

Zone:
鍵盤範囲ごとに異なるサンプル

用途:

ドラムキット:
各鍵盤に楽器

Multi-Sample楽器:
音域ごと

ボーカルチョップ:
フレーズごと

操作:

Zone作成:
サンプルをドロップ

範囲調整:
Zone端をドラッグ

削除:
Zone選択 → Delete
```

### Key Range

```
Low Key:
範囲の開始

High Key:
範囲の終了

Root Key:
元の音程

例:

Zone 1:
Low: C1
High: C1
Root: C3 (Kick)

Zone 2:
Low: D1
High: D1
Root: D3 (Snare)

効果:
各鍵盤に割り当て
```

### Velocity Range

```
Vel Low:
ベロシティ範囲下限

Vel High:
ベロシティ範囲上限

用途:

ダイナミクス:
弱く弾く → 柔らかい音
強く弾く → 硬い音

例:

Zone A:
Vel: 1-63
Sample: Soft Snare

Zone B:
Vel: 64-127
Sample: Hard Snare

効果:
強弱で音色変化
```

---

## Loop設定

**ループポイント:**

### Loop Mode

```
種類:

Off:
ループなし
ワンショット

Forward:
前方ループ
最も一般的

Ping-Pong:
往復
特殊効果

Backward:
逆ループ
実験的

推奨:

ワンショット:
Off

持続音:
Forward
```

### Loop Point調整

```
Sample View:

波形表示:
┌────────────────────────┐
│ ~~~~/\~~~~~/\~~~~~/\~~ │
│    ↑Start   ↑End      │
│       [=Loop=]         │
└────────────────────────┘

Start Point:
ループ開始位置

End Point:
ループ終了位置

調整:

1. 波形を Zoom (+/-)

2. Start/End マーカードラッグ

3. 再生して確認

4. シームレス？

コツ:

ゼロクロス:
波形が 0 を通る点
クリック音防止

Match Phase:
開始と終了の位相合わせ

推奨:

パッド:
長いループ
数秒

ベース:
短いループ
1-2拍
```

---

## Filter & Envelope

**音色制御:**

### Filter

```
Type:

Lowpass:
最も使用
高域カット

Highpass:
低域カット

Bandpass:
中域のみ

Cutoff:

ワンショット:
5000-10000 Hz
明るめ

パッド:
1000-2000 Hz
暗め

Resonance:

0-30%:
自然

30-60%:
強調

推奨:

Kick: Lowpass, 8000 Hz
Snare: Lowpass, 12000 Hz
```

### Filter Envelope

```
Amount:
-100% 〜 +100%

ワンショット設定:

Attack: 0 ms
Decay: 100-300 ms
Sustain: 0%
Release: 50 ms
Amount: +40%

効果:
アタックが明るい

パッド設定:

Attack: 500 ms
Decay: 1000 ms
Sustain: 60%
Release: 1500 ms
Amount: +50%

効果:
ゆっくり明るくなる
```

### Amp Envelope

```
ワンショット:

Attack: 0 ms
Decay: 100 ms
Sustain: 0%
Release: 50 ms

パッド:

Attack: 500 ms
Decay: 800 ms
Sustain: 80%
Release: 2000 ms
```

---

## 実践: ワンショット加工

**Kickを楽器に:**

### Step 1: サンプル読み込み (2分)

```
1. 新規MIDI Track

2. Sampler挿入

3. Kickサンプルドロップ:
   Browser > Drums > Kicks

4. 確認:
   C3で再生
```

### Step 2: Pitch設定 (2分)

```
Transpose:
0 st
(元の音程)

Fine:
0 cent

Stretch:
Off
(ピッチ変化なし)

確認:
C2 - C4 範囲
音程変化
```

### Step 3: Filter (2分)

```
Type:
Lowpass

Cutoff:
8000 Hz

Resonance:
10%

Drive:
15%
わずかに温かく
```

### Step 4: Envelope (2分)

```
Filter Envelope:

Attack: 0 ms
Decay: 150 ms
Sustain: 0%
Release: 50 ms
Amount: +35%

Amp Envelope:

Attack: 0 ms
Decay: 150 ms
Sustain: 0%
Release: 50 ms
```

### Step 5: 仕上げ (2分)

```
Loop:
Off (ワンショット)

Global:
Volume: 0 dB

確認:
C2, C3, C4
音程変化したKick

用途:
ベースライン代わり
808 Bass的

プリセット保存:
"Kick Instrument"
```

---

## 実践: ボーカルチョップ

**ボーカルを楽器に:**

### Step 1: 準備 (5分)

```
ボーカルサンプル:

必要:
アカペラ
1-2小節フレーズ

入手先:
Splice
Loopcloud
自分で録音

読み込み:
Samplerにドロップ
```

### Step 2: スライス (10分)

```
手動スライス:

1. 波形表示

2. フレーズごとに:
   視覚的に確認

3. 各フレーズ:
   別々にエクスポート

4. 各フレーズ:
   別Zoneに配置

自動スライス (Simpler):

1. Simpler使用

2. Slice Mode: Beat

3. 自動分割

4. 各スライス:
   パッドに割当

推奨:

4-8フレーズ:
手動

16+フレーズ:
自動 (Simpler)
```

### Step 3: Zone配置 (5分)

```
Zone 1:

Key: C3
Sample: フレーズ1 "Hey"

Zone 2:

Key: D3
Sample: フレーズ2 "Yeah"

Zone 3:

Key: E3
Sample: フレーズ3 "Oh"

...8つまで

効果:
鍵盤で演奏
新しいメロディ
```

### Step 4: エフェクト (5分)

```
各Zone:

Filter Cutoff: 5000 Hz
Reverb: わずかに
Delay: Ping-Pong

Global:

Pitch Bend Range: ±2 st
ピッチベンド可能

確認:
フレーズ組み合わせ
オリジナルメロディ
```

---

## よくある質問

### Q1: Sampler vs Simpler どっち？

**A:** 用途次第

```
Simpler推奨:

ドラム:
1サンプル = 1パッド

ワンショット:
シンプル

初心者:
使いやすい

Sampler推奨:

複数Zone:
鍵盤マップ

ボーカルチョップ:
複数フレーズ

複雑な設定:
Velocity範囲等

推奨:

最初:
Simpler

慣れたら:
Sampler
```

### Q2: ループポイントが合わない

**A:** ゼロクロスで調整

```
問題:
ループ時にプツッ

原因:
位相不一致

解決:

1. 波形 Zoom

2. ゼロクロス探す:
   波形が 0 通過

3. Start/End調整:
   両方ゼロクロスに

4. 再生確認

ツール:

Snap to Zero-Crossing:
自動調整

推奨:
常にOn
```

### Q3: サンプルが音割れする

**A:** Normalize使用

```
問題:
音量小さい/大きい

解決:

1. Sample View

2. Normalize:
   右クリック

3. 0 dBに正規化

または:

Gain調整:
Sample タブ内

推奨値:
-3 〜 0 dB
```

---

## まとめ

### Sampler基礎

```
□ サンプル読み込み
□ Zone設定 (キー範囲)
□ Loop Point調整
□ Filter & Envelope
□ Simpler との使い分け
```

### 実践テクニック

```
ワンショット:
Loop Off, Envelope短め

パッド:
Loop On, Envelope長め

ボーカルチョップ:
複数Zone, 各鍵盤配置
```

### 重要ポイント

```
□ ゼロクロスでループ
□ Normalize で音量調整
□ Zone で鍵盤マップ
□ Filter で音色調整
□ Simpler から始める
```

---

**次は:** [Drum Rack](./drum-rack.md) - ドラムキット構築の全て
