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

## よくある質問

### Q1: Clip AutomationとTrack Automationどっち？

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

**次のセクション:** [03-instruments](../03-instruments/) - シンセ・ドラム音源完全理解
