# Automation（オートメーション）

パラメーターを時間制御。フィルタースイープ、ボリュームフェード、パンニング変化など、動的な変化を完全マスター。

## この章で学ぶこと

- Automationとは何か
- Envelope描画テクニック
- Automation Mode（Read, Write, Touch, Latch）
- MIDI Learn活用法
- Clip Automation vs Track Automation
- Breakpoint編集
- 実用的なAutomationパターン
- モジュレーションとの使い分け

---

## なぜAutomationが重要なのか

**音楽に生命を吹き込む:**

```
Automationなし:

全て固定:
音量同じ
フィルター同じ
パン同じ

結果:
平坦
退屈
プロの音にならない

Automationあり:

時間変化:
サビで音量UP
ドロップでフィルター開く
展開でパン動く

結果:
ダイナミック
興奮
プロの音

プロの使用率:

100%のトラック:
何らかのAutomation

平均:
1曲に50-100箇所のAutomation

特にTechno/House:
フィルタースイープ
Reverbセンド
パラメーター変化
= 必須テクニック
```

---

## Automationとは

**時間軸でのパラメーター制御:**

### 基本概念

```
定義:

パラメーターの値を
時間軸で変化させる

例:

フェーダー:
1小節目: -∞ dB (無音)
4小節目: 0 dB (通常)
→ フェードイン

フィルター:
ドロップ前: Cutoff 200 Hz
ドロップ: Cutoff 20,000 Hz
→ 開放感

パン:
左 → 中央 → 右
→ 動き

Abletonでの表示:

赤い線 = Automation
Breakpoint = ポイント

編集:

マウスでドラッグ
ブレイクポイント配置
曲線作成
```

### AutomationできるもののAutomation

```
ミキサー:

Volume (Fader)
Pan
Send A/B/C/D
Mute

エフェクト:

Filter Cutoff
Reverb Dry/Wet
Delay Time
Distortion Amount

音源:

Oscillator Level
Filter Envelope
LFO Rate
Macro Knob

プラグイン:

ほぼ全てのパラメーター
(Automate可能なら)

制限:

Tempo: 可能 (但し注意)
Time Signature: 不可
Audio Effect Bypass: 可能
```

---

## Clip Automation vs Track Automation

**2種類の使い分け:**

### Clip Automation

```
場所:

Clip View > Envelopes

特徴:

Clip単位:
各Clipごとに設定

ループ対応:
Loopと一緒に繰り返す

相対的:
Clipを移動しても維持

用途:

繰り返しパターン:
4小節のフィルター変化

クリップ固有:
このClipだけ特殊

Session View:
主にこちら使う

操作:

1. Clip選択

2. Clip View > Envelopes

3. Device選択:
   例: Mixer, Filter

4. Parameter選択:
   例: Volume, Cutoff

5. 描画:
   ブレイクポイント配置

表示:

赤い線:
Automation Envelope
```

### Track Automation

```
場所:

Arrangement View
トラック上

特徴:

トラック全体:
全Clipに影響

絶対的:
時間軸に固定

複雑な変化:
長い展開

用途:

曲構成:
イントロ → ドロップ

長いフェード:
16小節かけてビルドアップ

Arrangement View:
主にこちら使う

操作:

1. トラック選択

2. Automation Mode: A (Show)

3. Parameter選択:
   トラック上部のプルダウン

4. 描画:
   マウスでドラッグ

5. ブレイクポイント:
   クリックして追加

表示:

トラック上:
赤い線 = Automation
```

### 優先順位

```
Clip Automation:
高い優先度

Track Automation:
低い優先度

両方ある場合:

Clip Automation:
適用される

Track Automation:
無視される

推奨:

Session View制作:
Clip Automation

Arrangement View制作:
Track Automation

混在:
注意が必要
```

---

## Automation描画

**Envelopeを描く:**

### 手動描画

```
基本操作:

ブレイクポイント追加:
ダブルクリック
または
クリック

移動:
ドラッグ

削除:
Delete
または
Backspace

範囲選択:
ドラッグで囲む
→ まとめて編集

曲線タイプ:

直線:
デフォルト

曲線:
Cmd+クリック
→ カーブ表示
→ ドラッグで調整

階段:
Grid Snap: On
→ カクカク変化

パターン例:

フェードイン:
低 → 高
直線

フェードアウト:
高 → 低
直線

フィルタースイープ:
低 → 高
曲線 (Exponential)

パンニング:
左 → 右 → 左
波形

ツール:

Grid Snap:
Cmd押しながら = Off
自由配置

Zoom:
+ / - キー
細かい編集
```

### 録音モード

```
リアルタイム録音:

Automation Mode:

Touch Mode:
つまみ触っている間だけ録音

Latch Mode:
一度触ったら最後まで録音

Write Mode:
全て上書き (危険)

操作:

Session View:

1. Clip選択

2. Re-Enable Automation (右クリック)

3. 再生中:
   つまみを動かす

4. 自動的に記録

Arrangement View:

1. Automation: A (Show)

2. Automation Mode: Touch

3. 再生:
   Space

4. つまみ動かす:
   記録される

推奨:

初心者:
手動描画 (簡単)

慣れたら:
Touch Mode (直感的)
```

---

## Automation Mode

**4つのモード:**

### Read Mode（デフォルト）

```
機能:

既存のAutomation:
再生される

つまみ:
動かしても記録されない

表示:

Automation Mode: (なし)
または
Read

用途:

通常再生:
録音しない

誤操作防止:
Automationを守る

推奨:

通常時:
常にReadモード
```

### Touch Mode（推奨）

```
機能:

つまみ触っている間:
録音

離すと:
元のAutomationに戻る

用途:

部分修正:
特定箇所だけ

リアルタイム録音:
演奏しながら

操作:

1. Automation Mode: Touch

2. 再生:
   Space

3. つまみ触る:
   → 記録開始

4. 離す:
   → 元に戻る

メリット:

安全:
触っている間だけ

直感的:
演奏的

推奨:
最も使いやすい
```

### Latch Mode

```
機能:

つまみ触る:
→ 記録開始

離しても:
最後の値で継続

用途:

一定値:
ある値でキープ

操作:

1. Automation Mode: Latch

2. 再生

3. つまみ動かす

4. 離す:
   → その値で固定

注意:

上書き:
既存Automation消える

慎重に使う
```

### Write Mode（危険）

```
機能:

再生中:
全てのパラメーターを記録

触らなくても:
現在値を書き込む

用途:

全リセット:
既存Automation削除して新規

操作:

1. Automation Mode: Write

2. 再生:
   → 即座に記録開始

3. 停止:
   → 全て上書きされる

警告:

非常に危険:
既存Automation全消去

通常使わない:
Touch/Latch で十分
```

---

## MIDI Learn

**物理コントローラーで制御:**

### MIDI Learnとは

```
定義:

MIDIコントローラーのつまみ
↓
Abletonのパラメーター
= リンク

用途:

MIDIコントローラー:
物理的なつまみ・フェーダー

リアルタイム制御:
演奏的な操作

DDJ-FLX4でも:
つまみをMIDI Learn可能

操作:

1. パラメーター選択:
   例: Filter Cutoff

2. Cmd+M (MIDI Learn Mode)

3. 画面が青くなる

4. MIDIコントローラー:
   つまみを動かす

5. リンク完了

6. Cmd+M (終了)

7. つまみ動かす:
   パラメーターが動く

解除:

Cmd+M → つまみ → Delete
```

### MIDI Learn + Automation

```
組み合わせ:

MIDI Learn:
リアルタイム制御

Automation録音:
Touch Mode

結果:
演奏を記録

手順:

1. パラメーターをMIDI Learn

2. Automation Mode: Touch

3. 再生開始

4. MIDIコントローラー:
   つまみ動かす

5. 停止

6. Automation完成

メリット:

直感的:
マウスより演奏的

滑らか:
自然な変化

効率的:
速い

DDJ-FLX4活用:

FXつまみ:
MIDI Learn

Filter:
Cutoffに割当

Beatノブ:
Reverbに割当
```

---

## 実用的なAutomationパターン

**定番テクニック:**

### 1. フィルタースイープ

```
目的:
ビルドアップ効果

手順:

1. フィルターデバイス挿入:
   Auto Filter

2. Automation:
   Cutoff パラメーター

3. パターン:

   小節 1-16: 200 Hz (暗い)
   小節 17-32: 徐々に上昇
   小節 32: 20,000 Hz (全開)

4. 曲線:
   Exponential (急激)

用途:

ビルドアップ:
ドロップ前

トランジション:
セクション間

ジャンル:

Techno: 必須
House: 頻繁
EDM: 定番
```

### 2. ボリュームフェード

```
フェードイン:

小節 1: -∞ dB (無音)
小節 4: 0 dB (通常)

曲線: 直線

用途: イントロ

フェードアウト:

小節 28: 0 dB
小節 32: -∞ dB

曲線: 直線

用途: アウトロ

クロスフェード:

トラック1: 0 dB → -∞ dB
トラック2: -∞ dB → 0 dB

同時に実行
→ スムーズな切り替え
```

### 3. Sendオートメーション

```
Reverb Send:

通常: 0%
ドロップ前: 100%
ドロップ: 0%

効果:
空間が広がる → 引き締まる

Delay Send:

フレーズ終わり: 50%
次のフレーズ: 0%

効果:
リピート → クリア

パターン:

小節 8: Send 0%
小節 15: Send 0%
小節 16: Send 100% (瞬間)
小節 17: Send 0%

= フィルイン効果
```

### 4. パンニング

```
静的パン:

固定位置:
Bass: 中央
Hi-Hat: 少し右
Perc: 左

動的パン:

左右移動:
0%: 中央
25%: 左
50%: 中央
75%: 右
100%: 中央

= 回転効果

LFO的パン:

正弦波:
規則的に左右

ランダム:
不規則に移動

用途:
Pad, SFX
```

### 5. Macroオートメーション

```
Macro Knob活用:

1つのAutomation:
複数パラメーター制御

例:

Macro 1:
- Filter Cutoff (0-100%)
- Reverb Wet (0-50%)
- Volume (+0 〜 +6 dB)

→ 1つのAutomationで
   3つ同時制御

設定:

1. Instrument/Audio Rack作成

2. Macro Knobに:
   複数パラメーターMap

3. Macro 1をAutomate

効率:
1本の線で複雑な変化
```

---

## Breakpoint編集テクニック

**細かい制御:**

### Breakpoint操作

```
追加:

ダブルクリック:
新規ポイント

クリック:
既存ライン上

移動:

ドラッグ:
位置・値変更

矢印キー:
微調整

Shift+矢印:
細かい移動

削除:

選択 → Delete

Cmd+クリック:
クリックで削除

範囲選択:

ドラッグ:
複数選択

Cmd+A:
全選択

一括編集:

選択後:
まとめて移動

Scale:
Cmd+ドラッグ
→ 比率維持で拡大縮小

コピー:

Cmd+C / Cmd+V:
Breakpointコピー

別トラックへ:
ペースト可能
```

### 曲線調整

```
カーブハンドル:

Alt+ドラッグ:
曲線作成

左右ハンドル:
独立調整

種類:

Linear (直線):
デフォルト

Exponential:
急激な変化
フィルタースイープ

Logarithmic:
緩やかな変化
ボリュームフェード

S-Curve:
滑らかな加減速

使い分け:

Volume: Linear
Filter: Exponential
Pan: Linear
Tempo: Linear (注意)
```

---

## エンベロープカーブの詳細解説

**カーブタイプの選び方:**

### Linear（直線）カーブ

```
特徴:

一定速度:
同じペースで変化

視覚的:
直線

数学的:
y = ax + b

用途:

ボリュームフェード:
自然な減衰

パンニング:
均等な移動

Tempo変更:
安定した変化

実例:

フェードイン 4小節:

小節 1: -∞ dB
小節 2: -18 dB
小節 3: -9 dB
小節 4: 0 dB

→ 均等な上昇

パン移動 8小節:

小節 1: 100% Left
小節 5: Center
小節 8: 100% Right

→ 一定速度で移動

メリット:

予測しやすい:
計算不要

自然:
ボリュームに最適

デメリット:

単調:
変化に驚きがない
```

### Exponential（指数）カーブ

```
特徴:

加速変化:
徐々に速くなる

視覚的:
曲線（上に凸）

数学的:
y = a^x

用途:

フィルタースイープ:
ドロップへのビルド

周波数変化:
人間の聴覚特性

エネルギー増加:
盛り上がり

実例:

フィルタービルドアップ:

小節 1-8: 100 Hz → 500 Hz (緩やか)
小節 9-14: 500 Hz → 2000 Hz (中速)
小節 15-16: 2000 Hz → 20000 Hz (急激)

→ 最後に爆発

Resonance強調:

小節 1-12: 0% (変化なし)
小節 13-15: 0% → 30% (加速)
小節 16: 30% → 80% (爆発)

理由:

人間の耳:
対数的に聞こえる

周波数:
倍音関係が指数的

ビルドアップ:
加速が興奮を生む

テクニック:

Breakpoint 3つ:
開始、中間、終了

中間を低めに:
→ 最後が急激

Alt+ドラッグ:
カーブ調整
```

### Logarithmic（対数）カーブ

```
特徴:

減速変化:
徐々に遅くなる

視覚的:
曲線（下に凸）

数学的:
y = log(x)

用途:

ボリュームフェード:
自然な減衰

エネルギー減少:
ブレイクダウン

リバーブ減衰:
Tail処理

実例:

フェードアウト:

小節 1-2: 0 dB → -6 dB (急)
小節 3-4: -6 dB → -12 dB (中)
小節 5-8: -12 dB → -∞ dB (緩)

→ 自然な消失

Reverb Tail:

0.0s: 100% Wet
0.5s: 50% Wet
2.0s: 10% Wet
4.0s: 0% Wet

理由:

人間の知覚:
音量は対数的

自然な減衰:
物理的共鳴

心地よさ:
急激に消えない

用途別:

エンディング:
曲の終わり

トランジション:
セクション間

ブレイクダウン:
静かなパート
```

### S-Curve（S字）カーブ

```
特徴:

加減速:
緩やか → 急 → 緩やか

視覚的:
S字型

数学的:
Sigmoid関数

用途:

滑らかな移行:
自然な変化

トランジション:
セクション間

モーション:
動きのある変化

実例:

セクション移行:

小節 1-2: 緩やか開始
小節 3-4: 急激変化
小節 5-6: 緩やか終了

パン移動:

Left → Center:
S-Curve使用

→ 加速して減速
→ 自然な動き

Volume Swell:

-∞ dB → 0 dB:
S-Curve

→ 滑らかに増加
→ 滑らかに安定

作成方法:

2つのカーブ組合せ:

前半: Exponential
後半: Logarithmic

または:

Alt+ドラッグ:
両端を調整

中間を急に:
両端を緩やかに

用途:

映画的:
ドラマチック

DJ Mix:
トラック間移行

Pad/String:
音量変化
```

---

## ブレイクダウン/ビルドアップでのAutomation活用

**楽曲構成の核心:**

### ビルドアップ（Build-up）

```
定義:

エネルギーを徐々に上げる:
ドロップへの準備

期間:
4小節 / 8小節 / 16小節

目的:
興奮・期待感

Automationポイント:

1. フィルタースイープ:

   小節 1: Cutoff 200 Hz
   小節 8: Cutoff 20,000 Hz

   カーブ: Exponential
   効果: 音が開く

2. Reverb Send増加:

   小節 1: 0%
   小節 7: 80%
   小節 8: 100%

   効果: 空間拡大

3. Volume上昇:

   小節 1: -3 dB
   小節 8: +3 dB

   効果: エネルギー増

4. Hi-Pass Filter:

   全トラックに:
   Low Cut 徐々に上昇

   小節 1: 20 Hz
   小節 8: 200 Hz

   効果: 低音を抜く

5. Noise/Riser追加:

   White Noise:
   Volume 0% → 100%

   Pitch:
   -12 semitones → 0

   効果: 上昇感

6. Delay Feedback:

   小節 7-8:
   Feedback 0% → 90%

   効果: エコー増幅

実例パターン:

Techno Build:

小節 1-4:
- Filter 200 Hz → 1000 Hz
- Reverb 0% → 30%

小節 5-7:
- Filter 1000 Hz → 10000 Hz
- Reverb 30% → 80%
- Volume -1 dB → +2 dB

小節 8:
- Filter全開 20000 Hz
- Reverb 100%
- Noise Riser爆発

House Build:

小節 1-8:
- Vocal Reverb増加
- Clap Volume UP
- Hi-Pass Filter上昇

小節 8.4:
- 全てMute
- ドロップへ
```

### ブレイクダウン（Breakdown）

```
定義:

エネルギーを下げる:
静かなセクション

期間:
8小節 / 16小節 / 32小節

目的:
休息・展開

Automationポイント:

1. フィルター閉じる:

   ドロップ終わり:
   Cutoff 20000 Hz

   ブレイクダウン開始:
   Cutoff 500 Hz

   カーブ: Logarithmic
   効果: 急速に暗く

2. Reverb Send減少:

   100% → 20%

   効果: 空間縮小

3. Volume下降:

   +3 dB → -6 dB

   効果: 静かに

4. Low-Pass Filter:

   Bass以外:
   Cutoffを下げる

   効果: こもった音

5. Drums削減:

   Kick: Mute
   Snare: Volume Down
   Hi-Hat: 残す

6. Pad/Vocal追加:

   Volume 0% → 100%

   効果: メロディ強調

実例パターン:

Techno Breakdown:

小節 1-2:
- Kick Mute
- Filter 急速に閉じる
- Reverb減少

小節 3-8:
- Padのみ
- Ambient音
- 静寂

小節 9-16:
- 徐々にビルド開始

House Breakdown:

小節 1-4:
- Drums全停止
- Vocal + Pad
- Reverb大量

小節 5-8:
- Percussion徐々に追加

Progressive Breakdown:

小節 1-8:
- フィルター閉じる
- Arpeggio追加
- 展開セクション

小節 9-16:
- ビルドアップ開始
```

### ドロップ（Drop）Automation

```
瞬間的変化:

ドロップ直前:

小節 16.4.4: 全パラメーター設定

ドロップ開始:

小節 17.1.1: 全パラメーター全開

Automationポイント:

1. Filter全開:

   16.4.4: 200 Hz
   17.1.1: 20000 Hz

   瞬間変化

2. Reverb削除:

   16.4.4: 100%
   17.1.1: 0%

   効果: クリア

3. Volume最大:

   16.4.4: -6 dB
   17.1.1: +3 dB

4. Side-chain再開:

   16.4.4: Off
   17.1.1: On

5. Bass戻す:

   Hi-Pass Filter:
   200 Hz → 20 Hz

テクニック:

ブレイクポイント2つ:

16.4.4: 準備
17.1.1: 爆発

→ 瞬間的変化

Impact強調:

Drop直前:
一瞬の無音

または:
Reverse Cymbal

実例:

Big Room Drop:

16.4.3-4: 全Mute (一瞬)
17.1.1: 全開

効果: 最大インパクト
```

---

## MIDIマッピングとAutomation連携

**MIDI CC と Automation:**

### MIDI CC（Control Change）とは

```
定義:

MIDI信号:
コントロール番号で制御

範囲:
0-127

Abletonでの扱い:

MIDI CC:
パラメーター制御可能

Automation:
MIDI CCも録音可能

例:

CC 1: Mod Wheel
CC 7: Volume
CC 10: Pan
CC 11: Expression
CC 64: Sustain Pedal
CC 71: Resonance
CC 74: Cutoff

DDJ-FLX4:

FXつまみ:
MIDI CCを送信

Abletonで:
MIDI Learnで受信
```

### MIDI Learn実践

```
基本手順:

1. パラメーター選択:

   例: Auto Filter > Cutoff

2. MIDI Learn Mode:

   Cmd+M (Mac)
   Ctrl+M (Win)

   画面が青に

3. MIDIコントローラー:

   つまみ/フェーダー動かす

4. リンク完成:

   Cmd+M で終了

5. 動作確認:

   つまみ → パラメーター連動

実用例:

DDJ-FLX4 FXつまみ:

左チャンネル FX:
→ Deck A Track Filter

右チャンネル FX:
→ Deck B Track Filter

Beatノブ:
→ Reverb Send

CFX:
→ Delay Feedback

複数マッピング:

1つのつまみ:
複数パラメーター制御可能

例:

Macro Knob:
- Filter Cutoff
- Resonance
- Reverb

→ Macroに MIDI Learn
→ 1つのつまみで全制御
```

### MIDI Learn + Automation録音

```
ワークフロー:

1. セットアップ:

   パラメーター → MIDI Learn
   Automation Mode: Touch

2. 録音:

   再生開始
   MIDIコントローラーで演奏
   自動録音

3. 結果:

   Automation Envelope作成
   後から編集可能

メリット:

演奏的:
マウスより直感的

滑らか:
人間らしい変化

効率:
速い制作

実例:

フィルタースイープ録音:

1. Filter Cutoff → MIDI Learn

2. Automation Mode: Touch

3. 再生:
   8小節ループ

4. FXつまみ:
   徐々に開く

5. 停止:
   Automation完成

6. 編集:
   Breakpoint微調整

応用:

ライブ録音:

DJプレイ中:
FX操作を全録音

後で:
Arrangement Viewで確認
```

### MIDI Mapping vs Automation

```
MIDI Mapping:

リアルタイム:
即座に反応

柔軟性:
いつでも変更可能

ライブ向き:
パフォーマンス

制限:
MIDIコントローラー必要

Automation:

固定:
再現性100%

編集可能:
細かい調整

制作向き:
スタジオワーク

利点:
コントローラー不要

組み合わせ:

制作時:
MIDI Mapping で演奏録音

録音後:
Automation で微調整

ライブ時:
両方使用

推奨:

DDJ-FLX4所有:
MIDI Learnを活用

コントローラーなし:
手動Automation描画
```

---

## クリップ vs トラックAutomation 詳細

**使い分けの実践:**

### Clip Automation詳細

```
メリット:

ループ対応:
4小節パターン繰り返し

再利用:
Clipコピーで Automation も複製

相対的:
Clip移動しても維持

用途:

繰り返しパターン:

Hi-Hat:
4小節でパン左右

Bass:
8小節でFilter変化

Pad:
16小節でVolume変化

Session View:

Scene単位:
各SceneにClip固有Automation

即興的:
ライブパフォーマンス

手順:

1. Clip選択:
   ダブルクリック

2. Clip View下部:
   Envelopes タブ

3. Deviceプルダウン:
   例: Mixer

4. Parameterプルダウン:
   例: Volume

5. 描画:
   マウスでEnvelope

6. ループ確認:
   Loop再生で繰り返し

実例:

Hi-Hat Clip:

4小節パターン:

小節 1: Pan Center
小節 2: Pan Left
小節 3: Pan Center
小節 4: Pan Right

→ ループで繰り返し

Bass Clip:

8小節パターン:

小節 1-4: Filter Closed
小節 5-8: Filter Open

→ 繰り返し使用

注意点:

優先度高い:
Track Automationを上書き

変更困難:
Clip毎に編集必要

Arrangement移行:
そのまま残る
```

### Track Automation詳細

```
メリット:

全体制御:
トラック全体に適用

絶対的:
時間軸に固定

複雑な変化:
長い展開可能

用途:

楽曲構成:

イントロ: フェードイン
ビルド: Filter上昇
ドロップ: 全開
アウトロ: フェードアウト

Arrangement View:

時間軸ベース:
曲全体の流れ

固定的:
毎回同じ再生

手順:

1. Arrangement View:
   Tab で切替

2. トラック選択

3. Automation Show:
   A キー

4. Parameterプルダウン:
   トラック上部

5. 描画:
   タイムライン上

6. Breakpoint:
   クリックで追加

実例:

Master Trackビルドアップ:

小節 1-16:
- Hi-Pass Filter上昇
- Compressor Threshold下降

小節 17:
- Filter全開
- Compressor通常

Vocal Track展開:

小節 1-8: Reverb 10%
小節 9-16: Reverb 40% (サビ)
小節 17-24: Reverb 10% (Verse)
小節 25-32: Reverb 80% (アウトロ)

注意点:

優先度低い:
Clip Automationに負ける

Session View:
あまり使わない

移動不可:
時間軸固定
```

### 優先順位の実践

```
両方ある場合:

Clip Automation:
優先される

Track Automation:
無視される

確認方法:

Re-Enable Automation:

手動でつまみ動かす:
→ Automation無効化

右クリック:
→ Re-Enable Automation

→ どちらが効いているか確認

実例:

Track Automation:
Volume 0 dB (全曲)

Clip Automation:
Volume -6 dB (特定Clip)

結果:
Clip再生時は -6 dB

推奨ワークフロー:

Session View制作:

1. Clip Automation:
   パターン作成

2. Arrangement移行:
   Clipを配置

3. Track Automation:
   全体調整追加

Arrangement View制作:

1. Clipは素材:
   Automationなし

2. Track Automation:
   全て描画

混在回避:

どちらか統一:
混乱防止

Session → Arrangement:
計画的に移行
```

---

## よくある質問

### Q1: Clip AutomationとTrack Automationどっち?

**A:** View次第

```
Session View:

Clip Automation:
主にこちら

理由:
Clipベースの制作

Track Automation:
あまり使わない

Arrangement View:

Track Automation:
主にこちら

理由:
時間軸ベースの制作

Clip Automation:
Loopパターンのみ

推奨:

Session → Arrangement移行時:

Clip Automationは:
そのまま残る

Track Automationは:
後から追加
```

### Q2: Automationが効かない

**A:** Read Modeか確認

```
確認:

1. Automation: 描いてある?
   赤い線確認

2. Automation Mode:
   Read になっている?
   (デフォルト)

3. Re-Enable Automation:
   右クリック → Re-Enable

4. パラメーター:
   正しいもの選択?

5. Clip vs Track:
   Clipが優先

よくあるミス:

手動でつまみ動かした:
→ Automationが無効化
→ Re-Enable必要

間違ったパラメーター:
→ プルダウン確認

Session/Arrangement混在:
→ どちらか統一
```

### Q3: Automationが急に変わる

**A:** ブレイクポイント追加

```
問題:

小節 1: 0%
小節 16: 100%

→ 徐々に変化してしまう

解決:

小節 1: 0%
小節 15.4.4: 0% ← 追加
小節 16.1.1: 100%

→ 15.4.4まで0%維持
→ 16.1.1で瞬間変化

コツ:

階段状変化:
直前にブレイクポイント

急激な変化:
2つのポイント接近

滑らかな変化:
ポイント離す
```

---

## まとめ

### Automation基本

```
□ Clip vs Track理解
□ 手動描画できる
□ ブレイクポイント編集
□ 曲線調整
□ Re-Enable理解
```

### 実用パターン

```
□ フィルタースイープ
□ ボリュームフェード
□ Send Automation
□ パンニング
□ Macro活用
```

### Automation Mode

```
Read: 通常時（デフォルト）
Touch: 部分録音（推奨）
Latch: 値キープ
Write: 全上書き（危険）
```

### 重要ポイント

```
□ Automationは必須
□ フィルタースイープ = 定番
□ MIDI Learn活用
□ Macro で効率化
□ ビルドアップに使う
```

---

## Automation応用テクニック集

**プロの現場で使われる高度な手法:**

### 1. レイヤー連動Automation

```
複数トラック同時制御:

目的:
統一感のある変化

手法:

グループトラック作成:

1. 複数トラック選択:
   Cmd+G でグループ化

2. グループトラック:
   Automation描画

3. 全子トラックに:
   同時適用

実例:

Drum Group:

Kick + Snare + Hi-Hat:
グループ化

Group Volume:
Automationで一括制御

ビルドアップ:
全Drums一緒に上昇

Synth Stack:

Lead + Pad + Bass:
グループ化

Group Filter:
一括でスイープ

効果:
統一感

応用:

Master Track:

全体を一括制御

Hi-Pass Filter:
ビルドアップで全体に

Low-Pass Filter:
ブレイクダウンで全体に

Return Track:

Reverb/Delay:
Send量を個別Automate

複雑な空間変化
```

### 2. モジュレーションとAutomationの組み合わせ

```
LFO + Automation:

基本:

LFO:
規則的な変化

Automation:
不規則な変化

組み合わせ:
複雑な動き

実例:

Filter Cutoff:

LFO:
4分音符で上下動

Automation:
8小節で全体上昇

結果:
波打ちながら上昇

LFO Rate Automation:

LFO Rate:
Automationで変化

初期: 1/4 (遅い)
中盤: 1/16 (速い)
終盤: 1/32 (超速)

効果:
加速感

LFO Amount Automation:

LFO Amount:
0% → 100%

効果:
徐々に揺れ増加

用途:
ビルドアップ

実用例:

Techno Pad:

Filter:
LFO で揺れ

LFO Amount:
Automation で増加

8小節かけて:
静止 → 激しく揺れ

Bass Wobble:

LFO:
Cutoff制御

LFO Rate:
Automationで加速

ドロップで:
超高速Wobble
```

### 3. サイドチェインとAutomation

```
Side-chain Compression:

基本:

Kick:
トリガー信号

Bass/Synth:
音量下がる

Automation併用:

Side-chain Amount:
Automationで制御

実例:

ビルドアップ中:

Side-chain: Off
または Amount 0%

理由:
エネルギー維持

ドロップ:

Side-chain: On
Amount 100%

効果:
パンチング復活

Breakdown:

Side-chain: Off

理由:
落ち着き

手順:

1. Compressor設定:
   Side-chain On

2. Threshold/Ratio:
   固定

3. Dry/Wet:
   Automationで制御

4. パターン:

   ビルド: 0%
   ドロップ: 100%
   ブレイク: 0%

応用:

複数トラック:

Bass:
Side-chain 100%

Pad:
Side-chain 50%

Lead:
Side-chain 0%

→ 階層的制御
```

### 4. Utility デバイスAutomation

```
Utilityとは:

Ableton標準:
シンプルツール

機能:

Width (Stereo幅)
Gain (音量)
Phase (位相)
Bass Mono (低域モノラル化)

Width Automation:

ビルドアップ:

Width 0% (Mono)
→ Width 100% (Stereo)

効果:
広がる感覚

ドロップ:

Width 150% (超Wide)

効果:
迫力

実例:

Synth Pad:

イントロ: 50%
サビ: 120%
ブレイク: 30%

Vocal:

Verse: 70%
Chorus: 100%

Gain Automation:

ボリューム微調整:

Faderではなく:
Utilityで制御

理由:

Faderは:
ミックス用

Utilityは:
楽曲構成用

実例:

Master Track:

ドロップ直前:
-3 dB → 0 dB

効果:
インパクト増

Bass Mono Automation:

低域制御:

通常: 200 Hz以下モノラル
ドロップ: 100 Hz以下モノラル

効果:
低域の広がり調整
```

### 5. Tempo Automation（上級）

```
Tempo変化:

Abletonでは:
Tempo Automationが可能

用途:

イントロ:
徐々に加速

アウトロ:
徐々に減速

注意:

Audio Warp:
Warp On必要

MIDI:
自動追従

Audio (Warp Off):
追従しない

実例:

イントロ加速:

小節 1: 100 BPM
小節 8: 125 BPM

曲線: Linear
効果: 徐々に加速

アウトロ減速:

小節 25: 125 BPM
小節 32: 80 BPM

効果: フェードアウト的

注意点:

DJには不向き:
BPM一定が前提

制作専用:
アルバム曲向け

Live演奏:
ドラマチック効果

手順:

1. Master Track:
   Automation Show

2. Parameter:
   Song Tempo

3. 描画:
   Breakpoint配置

4. 確認:
   全トラックWarp On
```

### 6. Return Track Automation

```
Return Trackとは:

Send/Return:
エフェクトトラック

用途:

Reverb
Delay
その他エフェクト

Automation活用:

Send量:
各トラックから

Return設定:
エフェクトパラメーター

両方制御:
複雑な効果

実例:

Reverb Return:

パラメーター:

Decay Time:
ビルドで増加

Dry/Wet:
ドロップで減少

Size:
セクションで変化

パターン:

イントロ:
Decay 2.0s (短)

ビルド:
Decay 8.0s (長)

ドロップ:
Decay 1.5s (短)

Breakdown:
Decay 10.0s (超長)

Delay Return:

Delay Time:

通常: 1/4
フィルイン: 1/16

Automation:
瞬間切替

Feedback:

通常: 30%
ビルド終盤: 90%

効果:
エコー爆発

Freeze:

ドロップ直前:
Freeze On

効果:
持続音
```

---

## Automation整理とメンテナンス

**プロジェクト管理:**

### Automationの可視化

```
確認方法:

Arrangement View:

A キー:
Automation Show/Hide

全トラック:
Automation一覧表示

色分け:

赤い線:
Automation Envelope

濃さ:
複数Automation

管理:

Parameter切替:

プルダウン:
Automation選択

複数ある場合:
切り替えて確認

Automation Lane:

Unfold:
複数Automation表示

トラック展開:
全パラメーター一覧

整理テクニック:

命名:

トラック名:
明確に

例:
"Lead - Filter Sweep"
"Bass - Volume Build"

色分け:

重要トラック:
目立つ色

Automation多い:
特定色

グループ化:

関連トラック:
グループ化

Group Automation:
一括制御
```

### Automation編集のベストプラクティス

```
非破壊編集:

コピー保存:

重要Automation:
複製してバックアップ

手順:

1. Breakpoint全選択:
   Cmd+A

2. コピー:
   Cmd+C

3. 別トラック:
   Cmd+V (保管)

バージョン管理:

Save As:
こまめに保存

"Track v1"
"Track v2"

段階的編集:

少しずつ変更:
一度に大きく変えない

確認:
再生して聴く

調整:
細かく修正

Undo活用:

Cmd+Z:
いつでも戻せる

履歴:
Edit > Undo History

効率化:

テンプレート作成:

定番パターン:
保存しておく

例:

"Filter Sweep 8bar"
"Volume Fade 4bar"

再利用:

Clipコピー:
Automation込み

プロジェクト間:
Import可能

Macroで統合:

複数パラメーター:
1つに集約

効率的管理
```

### トラブルシューティング

```
問題1: Automationが消えた

原因:

手動操作:
つまみ動かした

解決:

右クリック:
Re-Enable Automation

確認:
赤い線復活

問題2: Automationが効きすぎ

原因:

Range広すぎ:
0-100%で設定

解決:

Breakpoint調整:
範囲を狭める

例:
20-80%に変更

Scale機能:
Cmd+ドラッグで縮小

問題3: カクカク変化

原因:

Breakpoint不足:
ポイント2つだけ

解決:

中間点追加:
滑らかに

曲線調整:
Alt+ドラッグ

問題4: 複数Automation競合

原因:

Clip + Track:
両方ある

解決:

優先順位確認:
Clipが優先

統一:
どちらか削除

問題5: コピーできない

原因:

選択ミス:
範囲指定間違い

解決:

Breakpoint選択:
ドラッグで囲む

Cmd+A:
全選択

コピー:
Cmd+C
```

---

## ジャンル別Automation実例

**スタイル別定番パターン:**

### Techno

```
必須Automation:

1. Hi-Pass Filter:

全トラック:
ビルドアップで上昇

20 Hz → 200 Hz

効果:
低音抜けて軽く

2. Reverb Send:

パーカッション:
セクションで変化

ドロップ: 0%
ビルド: 80%

3. Filter Cutoff:

Lead/Bass:
スイープ多用

Exponentialカーブ

4. Delay Feedback:

フィルイン:
瞬間的に増加

90%まで上昇

定番パターン:

16小節ビルド:

小節 1-8:
Filter 緩やか上昇

小節 9-14:
Filter 中速上昇
Reverb 増加

小節 15-16:
Filter 急上昇
Delay Feedback 爆発

小節 16.4:
一瞬Mute

小節 17:
ドロップ全開
```

### House

```
特徴的Automation:

1. Vocal Reverb:

Verse: 20%
Chorus: 60%

滑らかに変化

2. Clap/Snare Volume:

ビルドで強調:
-6 dB → +3 dB

3. Piano/Keys Filter:

セクション移動:
明るさ変化

4. Bass Side-chain:

ドロップ: 100%
Breakdown: 0%

定番パターン:

8小節ビルド:

小節 1-4:
Vocal Reverb増加

小節 5-8:
Clap Volume UP
Hi-Pass Filter

小節 8.4:
Snare Roll (別トラック)

小節 9:
ドロップ
```

### Progressive / Melodic

```
重視するAutomation:

1. Pad Volume Swell:

S-Curve:
滑らかな増減

-∞ dB → 0 dB → -∞ dB

16小節周期

2. Arpeggio Filter:

繰り返し変化:
4小節パターン

Open → Close → Open

3. String Width:

Stereo幅:
50% → 120%

展開で広がる

4. Lead Reverb:

フレーズ終わり:
Decay増加

空間的余韻

定番パターン:

32小節展開:

小節 1-8:
Pad Swell開始

小節 9-16:
Arp Filter変化
String Width拡大

小節 17-24:
Lead登場
Reverb増加

小節 25-32:
全体ビルドアップ
```

### Dubstep / Bass Music

```
特殊Automation:

1. LFO Rate:

Wobble Bass:
加速変化

1/4 → 1/64

2. Distortion Amount:

ドロップで増加:
0% → 80%

3. Resampler Pitch:

Riser効果:
-12 semitones → 0

4. Glitch Effect:

瞬間的On/Off:
リズミカル

定番パターン:

ドロップビルド:

小節 1-7:
LFO Rate徐々に加速

小節 8:
Distortion全開
Filter全開

小節 8.4:
全Mute (瞬間)

小節 9:
Bass Drop爆発
```

---

## Automation実践ワークフロー

**実際の制作手順:**

### ステップ1: 基本トラック作成

```
下準備:

1. ドラム配置:

   Kick, Snare, Hi-Hat:
   基本パターン作成

2. Bass配置:

   グルーヴ確立

3. メロディ配置:

   Lead, Pad:
   構成決定

4. 構成確認:

   イントロ:
   小節 1-8

   ビルド:
   小節 9-16

   ドロップ:
   小節 17-32

   ブレイク:
   小節 33-48

この時点:

Automationなし:
全て固定パラメーター

ミックス:
大まかなバランスのみ
```

### ステップ2: メインAutomation追加

```
優先順位:

1. フィルタースイープ:

   ビルドアップ:
   小節 9-16

   全トラック:
   Hi-Pass Filter

   20 Hz → 200 Hz

   効果:
   最も重要

2. ボリュームフェード:

   イントロ:
   小節 1-4

   アウトロ:
   小節 45-48

   基本的な出入り

3. Reverb Send:

   ビルド:
   徐々に増加

   ドロップ:
   瞬間的に削減

手順:

1. Arrangement View:
   Tab で切替

2. Automation Show:
   A キー

3. 1トラックずつ:
   丁寧に描画

4. 再生確認:
   効果検証

5. 微調整:
   Breakpoint位置
```

### ステップ3: 詳細Automation追加

```
細かい演出:

1. パンニング:

   Hi-Hat:
   左右動き

   Percussion:
   揺れ追加

2. Send変化:

   Delay Send:
   フィルイン時

   フレーズ終わりで増加

3. Macro制御:

   複数パラメーター:
   一括変化

   効率化

4. LFO Amount:

   Pad/Lead:
   揺れの強さ変化

   ビルドで増加

実施タイミング:

メインAutomation完成後:
細部追加

やりすぎ注意:
必要な分だけ

確認:
再生しながら判断
```

### ステップ4: トランジション強化

```
セクション間:

ビルド → ドロップ:

1. Filter全開:

   16.4.4: 準備
   17.1.1: 爆発

2. Reverb削除:

   瞬間的にDry

3. Volume調整:

   インパクト増

ドロップ → ブレイク:

1. Filter閉じる:

   Logarithmicカーブ
   急速に暗く

2. Drums減少:

   Kick Mute
   Hi-Hat のみ

3. Pad/Vocal追加:

   Volume Swell

ブレイク → ビルド:

1. 徐々に復活:

   Drums戻す
   Filter開く

2. エネルギー蓄積:

   次のドロップへ

チェックポイント:

各トランジション:
滑らかか？

驚きはあるか:
予測可能すぎない？

自然か:
不自然な変化ないか
```

### ステップ5: 最終調整とポリッシュ

```
全体確認:

1. 通し再生:

   全曲再生:
   流れ確認

   違和感:
   修正箇所メモ

2. Automation見直し:

   過剰な箇所:
   減らす

   不足箇所:
   追加

3. カーブ調整:

   Linear → Exponential:
   より効果的に

   滑らか過ぎ:
   メリハリ追加

4. Breakpoint整理:

   不要なポイント:
   削除

   重要なポイント:
   位置微調整

ポリッシュ:

微調整:

Volume:
±1-2 dB調整

Filter:
Cutoff位置微調整

Timing:
Breakpoint位置ずらす

バランス:

全体のダイナミクス:
適切か

静と動:
コントラストあるか

展開:
飽きさせないか

Export前最終確認:

全Automation:
Re-Enable済み

不要なAutomation:
削除済み

命名:
トラック名整理
```

---

## Automationと他機能の連携

**統合的なアプローチ:**

### AutomationとWarping

```
Warpとは:

Audio Clip:
テンポ同期機能

Warp On:
BPMに追従

Automation連携:

Tempo Automation:

Warp On必須:
Audioが追従

Warp Off:
追従しない (注意)

Transpose Automation:

Warp On:
ピッチ変化

Detune:
微調整

Start/End Automation:

Clip範囲:
Automationで変更

Loop変化:
動的に

実例:

Vocal Chop:

Start Marker:
Automationで移動

効果:
異なる部分再生

Drum Loop:

End Marker:
徐々に短縮

効果:
リズム変化
```

### AutomationとGroove

```
Grooveとは:

タイミング/Velocity:
微妙なズレ

人間的グルーヴ:
機械的でない

Automation併用:

Groove Amount:

通常: 30%
ドロップ: 0%

理由:
タイトなグルーヴ

Clip Automation:

Groove設定:
Clipごと

セクション別:
異なるGroove

実例:

イントロ:

Groove 50%:
ゆったり

ドロップ:

Groove 0%:
正確

ブレイク:

Groove 70%:
人間的
```

### AutomationとFollow Action

```
Follow Actionとは:

Clip自動切替:
Session View機能

ランダム/順番:
自動進行

Automation連携:

Clip Automation:

各Clip:
固有Automation

Follow Action:
Clip切替で変化

ライブ向き:

即興的:
予測不能

展開:
自動的

実例:

Hi-Hat Clips:

Clip A:
Pan Left Automation

Clip B:
Pan Right Automation

Follow Action:
ランダム切替

効果:
予測不能なパン

Bass Clips:

Clip 1:
Filter Closed

Clip 2:
Filter Open

Follow Action:
順番切替

効果:
自動展開
```

---

## Automation CPU管理

**負荷対策:**

### CPU負荷の理解

```
Automationの負荷:

基本:

Automation自体:
軽い

エフェクト:
重い可能性

注意ポイント:

多数のAutomation:

100以上:
問題なし

1000以上:
やや重い

複雑なエフェクト:

Reverb:
CPU消費大

複数Automation:
負荷増加

最適化:

Freeze Track:

重いトラック:
Audio化

Automation:
維持される

Flatten:

完全Audio化:
Automation消える

軽量化:
最大
```

### 効率的なAutomation配置

```
ベストプラクティス:

1. グループ化:

   複数トラック:
   グループAutomation

   1つの操作:
   複数制御

2. Macro活用:

   複数パラメーター:
   1つのAutomation

   CPU効率:
   向上

3. Return Track:

   共通エフェクト:
   Return配置

   個別Send:
   Automation

   効率的:
   1つのReverbを共有

4. Automation削減:

   不要な箇所:
   削除

   静止パラメーター:
   Automation不要

実例:

非効率:

10トラック:
各々Reverb挿入

各Reverb:
Decay Automation

結果:
CPU消費大

効率的:

1つのReturn:
Reverb配置

Return Automation:
Decay制御

各トラックSend:
Send量Automation

結果:
CPU軽い
```

---

## Automation Tips & Tricks

**プロのテクニック集:**

### Tip 1: ダブルAutomation

```
テクニック:

同じパラメーター:
2段階制御

手法:

Clip Automation:
短期的変化

Track Automation:
長期的変化

実例:

Hi-Hat Volume:

Clip Automation:
4小節パターン
-3 dB → 0 dB → -3 dB

Track Automation:
16小節全体
0 dB → +3 dB

結果:
波打ちながら全体上昇

注意:

優先順位:
Clipが上書き

意図的に:
計算して使用
```

### Tip 2: オフセットAutomation

```
テクニック:

複数トラック:
わずかにズラす

効果:

自然:
同時過ぎない

深み:
時間差

実例:

フィルタースイープ:

Bass:
小節 9.1.1 開始

Lead:
小節 9.2.1 開始 (0.5小節遅れ)

Pad:
小節 9.3.1 開始 (1小節遅れ)

結果:
段階的に開く

パン移動:

Hi-Hat:
左 → 右

Perc 1:
右 → 左 (逆方向)

Perc 2:
左 → 右 (Hi-Hat遅れ)

結果:
複雑な動き
```

### Tip 3: ミラーAutomation

```
テクニック:

対称的変化:
左右/上下

用途:

Stereo感:
強調

バランス:
維持

実例:

ステレオパン:

Synth L:
50% Left → Center

Synth R:
50% Right → Center

結果:
収束

Filter対比:

Bass:
Cutoff上昇

Pad:
Cutoff下降

結果:
対照的
```

### Tip 4: ステップAutomation

```
テクニック:

階段状変化:
カクカク

用途:

リズミカル:
グリッチ的

明確:
瞬間変化

作成方法:

1. Breakpoint密集:

   16分音符ごと:
   ポイント配置

2. 値を交互:

   High → Low → High

3. Grid Snap:

   On にして配置

実例:

Filter Cutoff:

16分音符:

1: 500 Hz
2: 5000 Hz
3: 500 Hz
4: 5000 Hz

結果:
リズミカル変化

Pan:

8分音符:

1: Left
2: Right
3: Left
4: Right

結果:
ステレオ揺れ
```

### Tip 5: ランダムAutomation

```
テクニック:

不規則変化:
予測不能

作成方法:

手動:

Breakpoint:
ランダム配置

値:
不規則に設定

LFO使用:

Random LFO:
不規則変化

Amount:
Automation制御

実例:

Pad Volume:

不規則Swell:
予測不能

自然:
人間的

Percussion Pan:

ランダム移動:
空間的広がり

効果:
動的
```

---

## まとめ

### Automation基本

```
□ Clip vs Track理解
□ 手動描画できる
□ ブレイクポイント編集
□ 曲線調整
□ Re-Enable理解
```

### 実用パターン

```
□ フィルタースイープ
□ ボリュームフェード
□ Send Automation
□ パンニング
□ Macro活用
```

### Automation Mode

```
Read: 通常時（デフォルト）
Touch: 部分録音（推奨）
Latch: 値キープ
Write: 全上書き（危険）
```

### 高度なテクニック

```
□ カーブタイプ使い分け
□ ビルド/ブレイクダウン構築
□ MIDI Learn + Automation
□ レイヤー連動制御
□ ジャンル別定番パターン
```

### プロのワークフロー

```
□ 段階的にAutomation追加
□ メイン → 詳細 → ポリッシュ
□ トランジション重視
□ 全体バランス確認
□ CPU効率化
```

### 重要ポイント

```
□ Automationは必須技術
□ フィルタースイープ = 定番中の定番
□ カーブ選択が仕上がりを左右
□ ビルドアップで期待感を作る
□ ブレイクダウンで展開を生む
□ MIDI Learnで効率化
□ Macroで複数パラメーター一括制御
□ ジャンルごとに定番パターンがある
□ 段階的制作で完成度UP
□ トランジションが曲の印象を決める
```

### 次のステップ

```
練習課題:

1. 8小節ビルドアップ:

   Filter Sweep:
   作成してみる

   Reverb増加:
   追加

   Volume上昇:
   仕上げ

2. トランジション作成:

   ビルド → ドロップ:
   瞬間変化

   Breakpoint 2つ:
   タイミング練習

3. ジャンル模倣:

   好きな曲:
   Automation分析

   真似て作成:
   学習

4. MIDI Learn練習:

   DDJ-FLX4:
   つまみ割当

   リアルタイム録音:
   演奏的制作

マスターへの道:

初級:
基本Automation描画

中級:
カーブタイプ使い分け

上級:
複雑な組み合わせ

プロ:
直感的・効率的制作
```

---

**次のセクション:** [03-instruments](../03-instruments/) - シンセ・ドラム音源完全理解
