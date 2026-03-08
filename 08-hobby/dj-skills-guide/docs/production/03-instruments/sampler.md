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

## Sampler全パラメータ詳細解説

### Zone Editorの全機能

**ゾーンマッピングの極意:**

```
Zone Editor表示:
Zone タブをクリック

基本操作:

新規Zone作成:
1. サンプルを鍵盤領域にドラッグ
2. 自動的にZone生成
3. Key/Vel範囲が表示される

Zone選択:
クリックで選択
複数選択: Shift+クリック

Zone分割:
1. Zone右クリック
2. "Split at Note"
3. 指定音で分割

Zone結合:
1. 複数Zone選択
2. 右クリック "Merge Zones"
3. 1つのZoneに統合

Zone複製:
Cmd/Ctrl+D
同じ設定のZone作成
```

### Key Zoneパラメータ

```
Low Key:
範囲の最低音
C-2 〜 G8

High Key:
範囲の最高音
Low Key以上

Root Key:
サンプルの原音
この音程で再生すると元の音

Detune:
-50 〜 +50 cent
微調整

Transpose:
-48 〜 +48 半音
粗調整

Key Track:
0 〜 100%
音程追従度
100%: 完全追従
0%: 音程固定

例: ドラムキット

Kick Zone:
Low: C1, High: C1
Root: C1
Key Track: 0%
→ どの音でも同じ音程

Piano Zone:
Low: C3, High: C4
Root: C3
Key Track: 100%
→ 音程変化
```

### Velocity Zoneパラメータ

```
Vel Low:
0 〜 127
範囲下限

Vel High:
0 〜 127
範囲上限

Vel→Volume:
-100 〜 +100%
ベロシティで音量変化

Vel→Filter:
-100 〜 +100%
ベロシティでフィルター変化

実践例: リアルピアノ

Zone 1: ppp
Vel: 1-20
Sample: 非常に柔らかい音
Vel→Volume: +100%

Zone 2: p
Vel: 21-40
Sample: 柔らかい音
Vel→Volume: +80%

Zone 3: mp
Vel: 41-60
Sample: 中くらい
Vel→Volume: +60%

Zone 4: mf
Vel: 61-80
Sample: やや強い
Vel→Volume: +40%

Zone 5: f
Vel: 81-100
Sample: 強い音
Vel→Volume: +20%

Zone 6: ff
Vel: 101-120
Sample: 非常に強い
Vel→Volume: 0%

Zone 7: fff
Vel: 121-127
Sample: 最強音
Vel→Volume: 0%
Vel→Filter: +50%
(さらに明るく)

効果:
弾き方で音色変化
超リアル
```

---

## Sample Tabパラメータ完全ガイド

### Sample Display

**波形表示と編集:**

```
Zoom Controls:
+ / - ボタン
拡大縮小

Waveform:
上: ステレオL
下: ステレオR
モノラル: 1つ

マーカー:

Start (青):
再生開始点
ドラッグで調整

End (赤):
再生終了点

Loop Start (緑):
ループ開始

Loop End (緑):
ループ終了

Sample Select (黄):
選択範囲
ドラッグで選択

操作:

Snap to Zero Crossing:
右下アイコン
クリック音防止
推奨: 常にOn

Normalize:
右クリック
0dBに正規化

Reverse:
右クリック
逆再生

Fade In/Out:
選択範囲右クリック
フェード適用
```

### Sample Playbackパラメータ

```
Transpose:
-48 〜 +48 半音
全体のピッチ調整

Fine Tune:
-50 〜 +50 cent
微調整
1 cent = 1/100半音

Detune:
0 〜 100 cent
ランダムデチューン
複数Voiceで太く

Spread:
0 〜 100%
ステレオワイド化
Detuneと連動

Volume:
-inf 〜 +12 dB
サンプル音量

Pan:
L50 〜 R50
定位

実践例: アナログシンセ風

Transpose: 0 st
Fine: +10 cent
Detune: 30 cent
Spread: 40%
→ 太い音
```

### Sample Start/End

```
Sample Start:
0 〜 サンプル長
再生開始点

Sample End:
Start 〜 サンプル長
再生終了点

Offset:
0 〜 100%
Start位置のランダム化

Offset Range:
0 〜 5000 ms
ランダム幅

用途:

ドラム:
Offset: 0%
常に頭から

パッド:
Offset: 10%
Range: 500ms
→ 毎回わずかに異なる開始点
自然な響き

ループ:
Offset: 0%
正確な開始点
```

### Loop Parameters

```
Loop Mode:
Off / Forward / Ping-Pong / Backward

Loop Start:
0 〜 サンプル長
ループ開始点

Loop Length:
0 〜 残り長
ループ区間の長さ

Loop End:
自動計算
Start + Length

Crossfade:
0 〜 500 ms
ループつなぎ目のクロスフェード

Interpolation:
Off / Linear / Cubic
補間方法
Cubic: 最高音質

実践設定: パッド

Mode: Forward
Start: 5000 samples
Length: 20000 samples
Crossfade: 100 ms
Interpolation: Cubic

効果:
完全にシームレスなループ
```

---

## Filter Tabの完全マスター

### Filter Type詳細

```
Lowpass:

12dB / oct:
緩やかな減衰
自然

24dB / oct:
急峻な減衰
シンセ的

State Variable:
アナログモデル
温かい

Ladder:
Moog型
太い

用途:
最も使用される
高域カット

Highpass:

12dB / oct:
緩やか

24dB / oct:
急峻

用途:
低域カット
濁り除去

Bandpass:

6dB / oct:
広い

12dB / oct:
標準

用途:
中域抽出
電話音声風

Notch (Band Reject):

特定周波数カット

用途:
ハウリング除去
共鳴除去

Formant:

2つのピーク
母音的

用途:
ボーカル風
Talkbox風

All Pass:

位相変化のみ
音量変化なし

用途:
フェイザー
実験的
```

### Filter Parameters

```
Frequency (Cutoff):
20 Hz 〜 20 kHz
カットオフ周波数

推奨値:

Kick: 80-150 Hz
低域残す

Bass: 200-500 Hz
中低域

Pad: 1000-3000 Hz
柔らかく

Lead: 5000-10000 Hz
明るく

Resonance (Q):
0 〜 100%
共鳴の強さ

0-20%:
自然
フィルター感弱い

20-40%:
Standard
音楽的

40-60%:
強調
キャラクター

60-80%:
実験的
発振寸前

80-100%:
自己発振
音程が出る

Drive:
0 〜 100%
歪み追加

0-20%:
わずかに温かく

20-40%:
サチュレーション

40-60%:
オーバードライブ

60-100%:
ディストーション

Morph:
0 〜 100%
フィルター形状変化
Type依存
```

### Filter Envelope

```
Amount:
-100 〜 +100%
エンベロープの影響度

+値: Cutoff上昇
-値: Cutoff下降

Attack:
0 〜 20000 ms
立ち上がり

Decay:
0 〜 20000 ms
減衰

Sustain:
0 〜 100%
持続レベル

Release:
0 〜 20000 ms
リリース

実践例1: プラック系

Amount: +60%
Attack: 0 ms
Decay: 300 ms
Sustain: 20%
Release: 100 ms

効果:
アタックが明るい
すぐ暗くなる

実践例2: パッド

Amount: +40%
Attack: 2000 ms
Decay: 1500 ms
Sustain: 60%
Release: 3000 ms

効果:
ゆっくり明るくなる
長い余韻

実践例3: ベース

Amount: +80%
Attack: 0 ms
Decay: 200 ms
Sustain: 0%
Release: 50 ms

効果:
パンチの効いたアタック
すぐ締まる
```

---

## Modulation Tab完全ガイド

### LFO 1 & 2

**周期的な変調:**

```
Waveform:

Sine:
滑らか
最も自然

Triangle:
リニア
音楽的

Square:
パルス
トレモロ

Saw Up:
上昇鋸歯状

Saw Down:
下降鋸歯状

Random:
ランダム
S&H風

Rate:
0.01 Hz 〜 40 Hz
変調速度

Sync: On
テンポ同期
1/32 〜 16 bars

Amount:
0 〜 100%
変調量

Destination:

Pitch:
ビブラート

Filter Cutoff:
ワウワウ

Pan:
オートパン

Volume:
トレモロ

実践例: ビブラート

LFO 1:
Wave: Sine
Rate: 5 Hz
Amount: 20%
Dest: Pitch

効果:
自然なビブラート

実践例: フィルタースイープ

LFO 2:
Wave: Triangle
Rate: 1/4 (Sync)
Amount: 60%
Dest: Filter

効果:
4分音符でフィルター開閉
```

### Modulation Matrix

**高度なルーティング:**

```
構造:

Source → Destination
Amount: -100 〜 +100%

利用可能Source:

LFO 1 / 2:
周期変調

Envelope 1-3:
エンベロープ

Velocity:
鍵盤の強さ

Key:
音程位置

Mod Wheel:
MIDIコントローラー

Aftertouch:
鍵盤圧力

Random:
ランダム

利用可能Destination:

Pitch:
音程

Filter Cutoff:
明るさ

Resonance:
共鳴

Volume:
音量

Pan:
定位

LFO Rate:
変調速度

実践例1: Vel→Filter

Source: Velocity
Dest: Filter Cutoff
Amount: +70%

効果:
強く弾く → 明るく
弱く弾く → 暗く

実践例2: Key→Pan

Source: Key
Dest: Pan
Amount: +50%

効果:
低音 → 左
高音 → 右
ピアノ的

実践例3: Random→Pitch

Source: Random
Dest: Pitch
Amount: +5%

効果:
音程わずかにランダム
アナログ的
```

---

## マルチサンプルインストゥルメント作成

### 実践: リアルピアノ

**プロレベルの楽器作成:**

```
Step 1: サンプル準備 (30分)

必要サンプル数:
最低: 12個 (1オクターブ)
推奨: 24個 (2オクターブ)
プロ: 88個 (全鍵盤)

ベロシティレイヤー:
最低: 2層 (pp, ff)
推奨: 4層 (pp, mp, f, ff)
プロ: 8層

収録:

音域:
C1, C2, C3, C4, C5, C6, C7

各音:
pp (Vel 1-31)
mp (Vel 32-63)
f (Vel 64-95)
ff (Vel 96-127)

命名規則:
Piano_C1_pp.wav
Piano_C1_mp.wav
Piano_C1_f.wav
Piano_C1_ff.wav
...

Step 2: Zone配置 (20分)

C1 pp Zone:
Low: C1
High: B1
Root: C1
Vel: 1-31
Sample: Piano_C1_pp.wav

C1 mp Zone:
Low: C1
High: B1
Root: C1
Vel: 32-63
Sample: Piano_C1_mp.wav

...全音域、全レイヤー配置

コツ:

Zone範囲:
隣り合う音でオーバーラップ
Low: C1, High: C#2
→ 中間音は自動補間

Root Key:
サンプルの実音に正確に設定

Step 3: 調整 (30分)

音量バランス:
1. 全Zone選択
2. Normalize適用
3. 各Zoneの音量確認
4. 不均衡があれば個別調整

音色統一:
Filter: Lowpass 8000 Hz
わずかに丸める

Envelope:
Attack: 10 ms
  (鍵盤音の自然なアタック)
Decay: 300 ms
Sustain: 70%
Release: 500 ms

ベロシティカーブ:
Vel→Volume: +80%
強弱の差を自然に

Step 4: 仕上げ (20分)

Reverb:
Room Size: 30%
わずかな響き

EQ:
Low: +2dB @ 80 Hz
  (豊かな低域)
High: -1dB @ 12 kHz
  (耳障りな高域カット)

Compressor:
Ratio: 2:1
Threshold: -12 dB
Attack: 10 ms
Release: 100 ms
  (音量均一化)

保存:
"Realistic Piano.adv"
```

---

## サンプリングワークフロー実践

### ワークフロー1: フィールドレコーディングから楽器作成

**環境音を楽器に:**

```
Step 1: 録音 (30分)

機材:
ハンディレコーダー (Zoom H4n等)
iPhoneボイスメモ

録音設定:
48kHz / 24bit
WAV形式

対象:

自然音:
雨、風、鳥の声

都市音:
電車、工事、群衆

楽器:
グラス、金属、木

コツ:
様々な強度で録音
複数テイク

Step 2: 編集 (20分)

1. DAWにインポート

2. ベストテイク選択

3. トリミング:
   不要部分カット

4. Normalize:
   -3dB

5. Fade In/Out:
   50-100ms

6. Export:
   個別WAVファイル

Step 3: Samplerに読み込み (15分)

1. Sampler挿入

2. サンプルドロップ

3. Zone設定:
   C3をRoot Keyに

4. Loop設定:
   Mode: Forward
   Pointを調整

5. 確認:
   鍵盤で演奏

Step 4: 音作り (30分)

Filter:
Type: Lowpass
Cutoff: 3000 Hz
Res: 30%

Filter Env:
Amount: +50%
Attack: 500 ms
Decay: 1000 ms
Sustain: 40%

Amp Env:
Attack: 200 ms
Release: 1500 ms

LFO 1:
Wave: Sine
Rate: 0.5 Hz (Sync: 2 bars)
Amount: 30%
Dest: Filter Cutoff

効果:
アンビエント楽器

保存:
"Ambient Rain Pad"
```

### ワークフロー2: ドラムキット構築

**カスタムキット作成:**

```
Step 1: サンプル収集 (20分)

必要:
Kick x 3-5
Snare x 3-5
Hi-Hat x 3-5
Clap x 2
Percussion x 5-10

入手先:
Splice
自作サンプル
サンプルパック

整理:
フォルダ分け
命名規則統一

Step 2: Zone配置 (30分)

Kick Zone:
C1: Main Kick
D1: Sub Kick
E1: Punchy Kick

Snare Zone:
F1: Main Snare
G1: Rim Shot
A1: Clap

Hi-Hat Zone:
A#1: Closed HH
B1: Open HH
C2: Pedal HH

Percussion Zone:
D2-G2: Shaker, Tambourine等

各Zone設定:
Loop: Off
Key Track: 0%
(音程変化なし)

Step 3: 音量調整 (20分)

全サンプル:
Normalize

相対音量:
Kick: 0 dB (基準)
Snare: -3 dB
Hi-Hat: -6 dB
Percussion: -8 dB

ベロシティ:
Vel→Volume: +60%
強弱つける

Step 4: エフェクト (30分)

Kick:
EQ: +3dB @ 60 Hz
Compressor: 4:1

Snare:
EQ: +2dB @ 200 Hz
Reverb: Room 15%

Hi-Hat:
HPF: 500 Hz
EQ: +3dB @ 8 kHz

Global:
Master Compressor
Limiter: -0.3 dB

保存:
"My Custom Kit.adv"
```

---

## ジャンル別サンプラー活用法

### Hip Hop: サンプリング中心

**ソウルサンプル加工:**

```
素材:
70年代ソウル
ファンク
ジャズ

Step 1: サンプル選定

ポイント:
ブレイク部分
ボーカルフレーズ
楽器ソロ

長さ:
2-4小節

Step 2: チョップ

手法:
4-8分割
不均等も可

配置:
C3-G3
8パッド

Step 3: エフェクト

Lo-Fi化:
Bit Crusher: 12bit
Sample Rate: 22kHz

Filter:
Lowpass 6000 Hz
わずかに籠もらせる

Vinyl:
Crackle追加
Wow/Flutter: わずかに

Compression:
Ratio: 6:1
Threshold: -18 dB
Pumping感

Step 4: アレンジ

パターン:
1-2小節ループ
微妙にずらす

ピッチ:
-3 〜 +3 半音
LoFi感

Swing:
60-70%
グルーヴ

保存:
"Soul Chop Kit"
```

### Techno: ドラム&FX

**インダストリアルサウンド:**

```
素材:
金属音
機械音
ノイズ

Step 1: サンプル作成

録音:
金属を叩く
機械の動作音
静電気ノイズ

加工:
Reverb: Large Hall
Distortion: Heavy
Pitch: -12 st

Step 2: ループ設定

Mode: Forward
Length: 短め (100-500ms)
Crossfade: 20ms

効果:
持続するインダストリアル音

Step 3: モジュレーション

LFO 1:
Wave: Random
Rate: 1/16 (Sync)
Amount: 80%
Dest: Filter Cutoff

LFO 2:
Wave: Square
Rate: 1/8 (Sync)
Amount: 50%
Dest: Pan

効果:
動的なテクスチャ

Step 4: レイヤー

Base:
Kick (808)

Layer 1:
金属音 Kick
High-pass 200Hz

Layer 2:
ノイズバースト
Gate: 1/16

効果:
複雑な Kick
```

### House: ボーカル処理

**ディスコボーカル加工:**

```
素材:
ディスコ
ファンク
R&B ボーカル

Step 1: スライス

フレーズ選定:
"Yeah", "Come On", "Ooh"等

長さ:
0.5-2秒

配置:
C3-C4 (12フレーズ)

Step 2: ピッチ調整

全体:
曲のKeyに合わせる

個別:
Root Key設定
各フレーズの原音程

Step 3: エフェクト

Filter:
Lowpass 8000 Hz
Warmth

Chorus:
Rate: 0.5 Hz
Depth: 30%
80's感

Reverb:
Plate
Size: 2.5s
Mix: 40%

Delay:
1/8 Dotted
Feedback: 30%
Mix: 20%

Step 4: 演奏

パターン:
4小節ループ
8小節で変化

ピッチベンド:
±2 半音
表現力

Velocity:
強弱つける
人間らしく

保存:
"Disco Vocal Chops"
```

---

## 高度なサンプラーテクニック

### テクニック1: グラニュラー合成

**サンプルを粒子化:**

```
設定:

Sample Start Offset:
Random: 100%
Range: 1000ms

Loop Mode:
Ping-Pong

Loop Length:
10-50ms (極小)

Crossfade:
5ms

Amp Envelope:
Attack: 0ms
Decay: 50ms
Sustain: 0%
Release: 20ms

効果:
細かい粒子の雲
アンビエント

用途:
パッド
テクスチャ
ドローン

実践例:

1. ボーカルサンプル読み込み

2. 上記設定適用

3. 鍵盤演奏:
   ボーカルが粒子化
   認識不可能な雰囲気

4. LFO追加:
   Filter Cutoff変調
   さらに動的
```

### テクニック2: ステッパー効果

**リズミカルな変化:**

```
設定:

LFO 1:
Wave: Random
Rate: 1/16 (Sync)
Amount: 100%
Dest: Volume

LFO 2:
Wave: Square
Rate: 1/8 (Sync)
Amount: 80%
Dest: Pan

Filter:
Type: Lowpass
Cutoff: 2000 Hz

Filter Envelope:
Amount: +50%
Attack: 0ms
Decay: 100ms
Sustain: 0%

効果:
音量とPanがランダム変化
リズミカルなカット

用途:
Techno
Glitch
実験音楽

実践例:

1. パッドサンプル

2. 上記設定

3. 1音伸ばす:
   自動的にリズムパターン

4. Automation:
   LFO Rateを変化
   複雑なパターン
```

### テクニック3: レイヤーサンプリング

**複数サンプルの重ね技:**

```
構造:

Zone 1 (Base):
Sample: Kick
Low: C1, High: C1
Volume: 0dB

Zone 2 (Attack):
Sample: Click
Low: C1, High: C1
Volume: -6dB
Vel→Volume: +100%

Zone 3 (Body):
Sample: Sub Bass
Low: C1, High: C1
Volume: -3dB
Key Track: 100%

Zone 4 (Tail):
Sample: Room
Low: C1, High: C1
Volume: -12dB
Sample Start: 50%

効果:
4レイヤーが合わさる
超複雑なKick

調整:

タイミング:
各ZoneのSample Start微調整
位相合わせ

音量バランス:
Baseが主
Attackで明瞭さ
Bodyで太さ
Tailで広がり

応用:

Snare:
Base + Click + Tail

Hi-Hat:
Closed + Noise + Click

Clap:
Sample1 + Sample2 + Reverb
```

### テクニック4: モーフィング

**サンプル間の移行:**

```
設定:

Zone 1:
Sample A (Soft Piano)
Vel: 1-63

Zone 2:
Sample B (Hard Piano)
Vel: 64-127

Crossfade設定:
Vel→Volume Zone1: +100%
Vel→Volume Zone2: -100%

効果:
Velocityで2音色が変化
滑らかな移行

高度な設定:

Zone 1:
Vel: 1-127
Vel→Filter: +60%
Vel→Volume: +80%

Zone 2 (同音程):
Vel: 64-127
Vel→Filter: +40%
Vel→Volume: +60%

Zone 1 Volume Curve:
127で0%に

Zone 2 Volume Curve:
1で0%に

効果:
Vel 1-63: Zone 1のみ
Vel 64-95: 両方ミックス
Vel 96-127: Zone 2のみ

応用:

ストリングス:
Soft → Intense

ブラス:
Mellow → Bright

ボーカル:
Whisper → Shout
```

---

## トラブルシューティング

### 問題1: CPUが高い

**解決法:**

```
原因:
複数サンプル同時再生
高品質Interpolation

対策:

1. Voices制限:
   Global Tab
   Max Voices: 8-16

2. Interpolation下げ:
   Linear使用

3. サンプルレート:
   48kHz → 44.1kHz

4. Freeze:
   トラックFreeze
   CPU削減

5. Bounce:
   Audioトラックに変換
```

### 問題2: 音が途切れる

**解決法:**

```
原因:
Release短すぎ
Voice数不足

対策:

1. Amp Release延長:
   100ms以上

2. Max Voices増加:
   16 → 32

3. Voice Mode:
   Polyphony確認
```

### 問題3: ピッチがずれる

**解決法:**

```
原因:
Root Key設定ミス
Detune設定

対策:

1. Root Key確認:
   サンプルの実音に設定

2. Detune Reset:
   0 cent

3. Transpose確認:
   0 st

4. チューナー使用:
   正確な音程確認
```

### 問題4: ノイズが入る

**解決法:**

```
原因:
ループポイント不適切
サンプルノイズ

対策:

1. Zero Crossing:
   Snap to Zero Crossing On

2. Crossfade延長:
   50-100ms

3. Fade In/Out:
   サンプル編集

4. Gate使用:
   低レベルノイズカット
```

---

## プリセット管理

### プリセット保存

**整理術:**

```
フォルダ構成:

Sampler/
├── Drums/
│   ├── Kicks/
│   ├── Snares/
│   └── Hi-Hats/
├── Instruments/
│   ├── Piano/
│   ├── Strings/
│   └── Brass/
├── Vocals/
│   ├── Chops/
│   └── Pads/
└── FX/
    ├── Ambient/
    └── Glitch/

命名規則:

ジャンル_楽器_特徴.adv

例:
Techno_Kick_Industrial.adv
House_Vocal_Disco.adv
Hip_Hop_Soul_Chop.adv

メタ情報:

Author: 自分の名前
Description: 用途、設定
Tags: Kick, 808, Bass
```

### ライブラリ構築

**効率的な管理:**

```
収集:

定期的:
週1でサンプル探索
新しいパック購入

整理:
すぐにフォルダ分け
命名規則統一

バックアップ:
外付けHDD
クラウド (Dropbox等)

サンプル管理ツール:

Splice Desktop:
自動整理
タグ付け

Loopcloud:
AI検索
キー検出

Native Access:
Kontakt等管理

推奨ワークフロー:

1. 新サンプル入手

2. 試聴

3. ベスト選択

4. Samplerで楽器化

5. プリセット保存

6. プロジェクトで使用

7. 気に入ったら保存
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

グラニュラー:
極小Loop, Offset Random

レイヤー:
複数Zoneで重ね技
```

### 重要ポイント

```
□ ゼロクロスでループ
□ Normalize で音量調整
□ Zone で鍵盤マップ
□ Filter で音色調整
□ Simpler から始める
□ CPUに注意
□ プリセット整理
```

### 次のステップ

```
1週目:
基本操作マスター
ワンショット加工

2週目:
ボーカルチョップ
ドラムキット作成

3週目:
マルチサンプル楽器
ジャンル別テクニック

4週目:
高度なテクニック
プリセットライブラリ構築

継続:
毎日1プリセット作成
ライブラリ拡充
実践で使用
```

---

## 実践プロジェクト集

### プロジェクト1: 完全オリジナルドラムキット

**ゼロから作る90分:**

```
目標:
16パッドドラムキット
全て自作サンプル

Step 1: サンプル作成 (40分)

Kick:
- テーブル叩く
- 口でバスドラム音
- ゴミ箱叩く

Snare:
- 新聞紙振る
- 机叩く
- 手を叩く

Hi-Hat:
- 鍵束振る
- ペットボトル振る
- シャカシャカ音

Clap:
- 実際の拍手
- 複数重ねる

Percussion:
- グラス叩く
- 鉛筆タップ
- 本を閉じる音
- ドアノック

Step 2: 編集 (20分)

各サンプル:
1. トリミング
2. Normalize -3dB
3. Fade In/Out 20ms
4. Export WAV

Step 3: Sampler配置 (15分)

Zone配置:
C1-D#2 (16パッド)

各Zone:
Loop: Off
Key Track: 0%
Volume調整

Step 4: 音作り (15分)

Kick:
EQ: +5dB @ 80Hz
Saturation: 20%

Snare:
HPF: 200Hz
Reverb: Small Room

Hi-Hat:
HPF: 1000Hz
EQ: +3dB @ 10kHz

保存:
"My Original Kit"

結果:
完全オリジナルキット
世界に1つだけ
```

### プロジェクト2: アンビエント楽器5種

**環境音から楽器へ (120分):**

```
楽器1: Rain Pad

録音:
雨音 30秒

加工:
Loop: 2秒区間
Filter: LP 3000Hz
Reverb: Hall 50%

楽器2: Wind Synth

録音:
風音 20秒

加工:
Loop: Ping-Pong
LFO: Filter Sweep
Detune: 40 cent

楽器3: Traffic Bass

録音:
車通過音

加工:
Pitch: -12 st
Filter: LP 400Hz
Compression: Heavy

楽器4: Bird Lead

録音:
鳥の鳴き声

加工:
Slice: 8フレーズ
Zone: C3-G3
Delay: 1/8 Dotted

楽器5: Metallic Perc

録音:
金属音各種

加工:
ワンショット化
Reverb: Plate
Pan: Random

コレクション保存:
"Ambient City Kit"
```

### プロジェクト3: ジャンル別サンプルパック

**各ジャンル3楽器 (180分):**

```
House Pack:

1. Disco Vocal Chop
   - ソウルボーカル
   - 8フレーズ
   - Chorus + Delay

2. Funky Bass
   - ファンクベース
   - Velocity 3層
   - Filter Envelope

3. String Stab
   - ストリングス hit
   - Reverb大
   - Short Release

Techno Pack:

1. Industrial Kick
   - 金属音レイヤー
   - Distortion
   - Tight Envelope

2. Acid Stab
   - TB-303風
   - Filter Sweep
   - Resonance高

3. Noise Sweep
   - ホワイトノイズ
   - LFO Filter
   - Gate 1/16

Hip Hop Pack:

1. Vinyl Drums
   - ブレイクビーツ
   - Lo-Fi化
   - Swing適用

2. Soul Sample
   - 70sサンプル
   - 4小節Loop
   - EQ Warm

3. Vocal Shout
   - アドリブ
   - Pitch変化
   - Reverb少

各パック保存:
別フォルダで整理
```

---

## パフォーマンステクニック

### ライブでのSampler活用

**リアルタイム演奏:**

```
セットアップ1: Pad演奏

MIDI Controller:
Akai MPD218
16パッド

マッピング:
C1-D#2: ドラム
E2-G#2: ボーカルチョップ
A2-C3: FX

テクニック:

Velocity活用:
強弱で音色変化
表現力UP

Pad Combination:
複数同時押し
コード的

Roll機能:
高速連打
ドラムロール

セットアップ2: 鍵盤演奏

MIDI Keyboard:
49鍵

マッピング:
Low: ドラム (C1-C2)
Mid: メロディ (C3-C5)
High: FX (C6-C7)

テクニック:

両手活用:
左: ベースライン
右: メロディ

Pitch Bend:
ボーカルスライド
表現的

Mod Wheel:
Filter Cutoff
リアルタイム変化

Sustain Pedal:
パッドサステイン
ドラムは無視

セットアップ3: Push統合

Ableton Push:
完全統合

表示:
Zone配置表示
パラメータ編集

操作:
パッドで演奏
ノブで調整
リアルタイム録音
```

### MIDIエフェクト活用

**さらなる可能性:**

```
MIDI Effect 1: Arpeggiator

設定:
Rate: 1/16
Style: Up-Down

用途:
1音押す → アルペジオ
ボーカルチョップに最適

実践:
1. Samplerにボーカル配置
2. Arpeggiator挿入
3. 1鍵押すだけで旋律

MIDI Effect 2: Chord

設定:
Shift1: +4 (Major 3rd)
Shift2: +7 (Perfect 5th)
Shift3: +12 (Octave)

用途:
1音 → 4音コード
簡単ハーモニー

実践:
1. Samplerにパッド配置
2. Chord挿入
3. メロディが自動でコード化

MIDI Effect 3: Random

設定:
Chance: 30%
Choices: 3-5音
Scale: Minor

用途:
ランダムフレーズ生成
実験的

実践:
1. Samplerにドラム配置
2. Random挿入
3. 予測不可能なパターン

MIDI Effect 4: Scale

設定:
Base: C
Scale: Minor Pentatonic

用途:
音痴防止
常にスケール内

実践:
1. Sampler楽器
2. Scale挿入
3. どの鍵盤でも音楽的
```

---

## Macro Control活用

### Audio Effect Rack化

**1ノブで複数制御:**

```
セットアップ:

1. Samplerを選択

2. Cmd/Ctrl+G
   Audio Effect Rack化

3. Macro設定

Macro 1: Filter Cutoff

マップ:
Sampler Filter Cutoff
Range: 200-8000 Hz

用途:
明るさ調整

Macro 2: Reverb Mix

マップ:
Reverb (挿入)
Mix: 0-60%

用途:
空間調整

Macro 3: Distortion

マップ:
Saturator Drive
Range: 0-40%

用途:
歪み量

Macro 4: Attack Time

マップ:
Sampler Amp Attack
Range: 0-500ms

用途:
アタック調整

Macro 5: Release Time

マップ:
Sampler Amp Release
Range: 50-2000ms

用途:
余韻調整

Macro 6: Pitch

マップ:
Sampler Transpose
Range: -12 ~ +12 st

用途:
ピッチシフト

Macro 7: LFO Rate

マップ:
Sampler LFO1 Rate
Range: 0.1-10 Hz

用途:
変調速度

Macro 8: Dry/Wet

マップ:
Rack Chain Mix
Range: 0-100%

用途:
原音ミックス

保存:
"Sampler Mega Rack"

利点:
- 8ノブで全制御
- Push/Controllerで操作
- ライブ向き
```

---

## サンプル加工の裏技

### 裏技1: Time Stretch活用

**BPM変更なしで長さ調整:**

```
通常の問題:
サンプル長さ変更 = ピッチ変化

Time Stretch:
長さ変更 / ピッチ維持

手順:

1. オーディオトラックに配置

2. Warpモード選択:
   Complex: 汎用
   Complex Pro: 高音質
   Texture: パッド向き

3. 長さ調整:
   クリップエンドドラッグ

4. Bounce:
   新WAVエクスポート

5. Samplerに読み込み

用途:
- ボーカルテンポ合わせ
- ドラムループ延長
- パッド長さ調整

コツ:
- 元BPM設定重要
- Complex Proが高音質
- 極端な変更は避ける
```

### 裏技2: Reverse活用

**逆再生の魔法:**

```
基本:

1. サンプル読み込み

2. Sample View

3. 右クリック → Reverse

効果:
逆再生サンプル

応用1: Reverse Reverb

手順:
1. サンプルにReverb大量
2. Bounce
3. Reverse
4. 神秘的な立ち上がり

応用2: Reverse Cymbal

手順:
1. Cymbalサンプル
2. Reverse
3. Crash前の溜め音

応用3: Reverse Vocal

手順:
1. ボーカル
2. Reverse
3. 不気味な効果

保存:
"Reverse Kit"
```

### 裏技3: Convolution活用

**インパルス応答:**

```
準備:

サンプル:
短い音 (Kick, Clap等)

インパルス応答:
空間の響き

手順:

1. Convolution Reverb挿入

2. Impulse Response:
   Kickサンプル読み込み

3. 他の音を鳴らす:
   Kickのキャラクター付与

4. Bounce

5. Samplerへ

効果:
全く新しい音色
予測不可能

実験例:

Kick → Snare:
Snareで演奏 = Kick質感

Vocal → Pad:
Padがボーカル的に

Glass → Everything:
全てがガラス質に
```

---

## よくある失敗と対策

### 失敗1: 音が薄い

**原因と対策:**

```
原因:
サンプル自体が薄い
フィルター掛けすぎ

対策:

1. サンプル選び直し

2. EQ:
   +3dB @ 100Hz (Body)
   +2dB @ 5kHz (Presence)

3. Saturation:
   Subtle 10-20%
   倍音付加

4. Doubling:
   同サンプル2層
   Detune 10 cent

5. Compression:
   Ratio 3:1
   音圧UP

6. Layering:
   別サンプル追加
   補完
```

### 失敗2: 位相ずれ

**原因と対策:**

```
原因:
複数サンプル重ね
ズレ発生

対策:

1. Sample Start揃える:
   全Zone 0に

2. Phase Invert試す:
   Utility挿入
   Phase反転

3. 波形確認:
   視覚的にチェック

4. EQ調整:
   被る周波数カット

5. Timing微調整:
   数ms単位で
```

### 失敗3: 音量不均一

**原因と対策:**

```
原因:
Zone間の音量差
Velocity設定ミス

対策:

1. 全Zone Normalize

2. Velocity Curve調整:
   Linear → Exponential

3. Compressor:
   各Zone個別に

4. Limiter:
   最終段で均一化

5. Automation:
   演奏で調整
```

---

## サンプラー哲学

### サンプリングの倫理

**著作権と創造性:**

```
基本ルール:

OK:
- 自分で録音
- ロイヤリティフリー
- パブリックドメイン
- サンプルパック (商用利用可)

NG:
- 商業音源そのまま
- 許可なし使用

グレーゾーン:
- 大幅加工
- 認識不可能レベル
- フェアユース
  (国により異なる)

推奨:

1. 自作サンプル中心

2. 購入サンプルパック活用

3. コラボ:
   許可取る

4. クレジット:
   サンプル元表記

5. 学習:
   私的利用のみ
```

### 創造的サンプリング

**アートとしての再構成:**

```
考え方:

従来:
既存曲の一部を使用

現代:
素材として再構成
完全に新しい作品

テクニック:

1. 極端な加工:
   原形を留めない

2. 文脈変更:
   元と異なる用途

3. レイヤリング:
   複数サンプル融合

4. 概念的使用:
   アイデアのみ借用

実例:

元: クラシックピアノ
→ Granular化
→ ドローンパッド
→ 完全に別物

元: 会話サンプル
→ Pitch極端変更
→ ベースライン
→ 言葉認識不可

哲学:
サンプリング = 引用ではなく
新しい文脈での創造
```

---

## 最終チェックリスト

### 制作時の確認事項

**完璧な仕上げ:**

```
□ サンプル品質
  - 48kHz/24bit以上
  - ノイズなし
  - クリーントリミング

□ Zone設定
  - 適切な範囲
  - Root Key正確
  - Velocity範囲

□ ループポイント
  - ゼロクロス
  - Crossfade適切
  - シームレス

□ 音量
  - Normalize済み
  - -3dB headroom
  - 均一化

□ フィルター/エンベロープ
  - 音楽的設定
  - 極端でない
  - 用途に適切

□ CPU効率
  - Voice数適切
  - Interpolation最適
  - 不要LFO Off

□ メタデータ
  - 命名規則
  - フォルダ整理
  - タグ付け

□ バックアップ
  - プリセット保存
  - サンプル保管
  - プロジェクト保存
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

グラニュラー:
極小Loop, Offset Random

レイヤー:
複数Zoneで重ね技
```

### 重要ポイント

```
□ ゼロクロスでループ
□ Normalize で音量調整
□ Zone で鍵盤マップ
□ Filter で音色調整
□ Simpler から始める
□ CPUに注意
□ プリセット整理
□ 著作権意識
```

### 次のステップ

```
1週目:
基本操作マスター
ワンショット加工

2週目:
ボーカルチョップ
ドラムキット作成

3週目:
マルチサンプル楽器
ジャンル別テクニック

4週目:
高度なテクニック
プリセットライブラリ構築

継続:
毎日1プリセット作成
ライブラリ拡充
実践で使用
創造的実験
```

---

**次は:** [Drum Rack](./drum-rack.md) - ドラムキット構築の全て
