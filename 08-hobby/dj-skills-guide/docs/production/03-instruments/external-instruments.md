# External Instrument

外部ハードウェアシンセを統合。Ableton LiveとMIDI機器をシームレスに接続します。

## この章で学ぶこと

- External Instrumentとは
- MIDIルーティング
- オーディオルーティング
- レイテンシー補正
- Freezeとバウンス
- ハードウェアシンセ接続例
- ドラムマシン統合

---

## なぜExternal Instrumentが重要なのか

**ハードウェアの統合:**

```
必要性:

Ableton付属音源:
十分すぎる

では、なぜ？

理由1: 既存機材活用

持っている:
ハードウェアシンセ
ドラムマシン

活用:
Abletonと統合
DAW内で制御

理由2: アナログ音質

ハードウェア:
真のアナログ
温かみ

VST:
エミュレート
近いが違う

理由3: ワークフロー

物理的:
つまみ触る
直感的

マウス:
クリック
時間かかる

使用率:

初心者: 0%
不要

中級者: 10-20%
一部活用

上級者: 30-40%
ハイブリッド

プロスタジオ:
50%+
アナログ中心
```

---

## External Instrumentとは

**ハードウェア統合デバイス:**

### 基本概念

```
役割:

MIDI送信:
Ableton → 外部機器

Audio受信:
外部機器 → Ableton

同期:
1つのトラック内

メリット:

シームレス:
内部音源と同じ操作

録音:
自動的に可能

Freeze:
オーディオ化

必要な機材:

ハードウェアシンセ:
Moog, Roland, Korg等

MIDIケーブル:
5pin MIDI

オーディオIF:
入力2ch以上

または:

オールインワン:
DDJ-FLX4 (MIDI Only)
+ 別途オーディオIF
```

---

## 接続方法

**MIDIとオーディオ:**

### MIDI接続

```
方式1: 5pin MIDI

ハードウェアシンセ:
MIDI IN

Mac/PC:
MIDIインターフェイス

ケーブル:
5pin MIDI

方式2: USB MIDI

モダンシンセ:
USB端子

Mac/PC:
直接接続

例:
Korg Minilogue XD
Arturia MicroFreak

推奨:
USB MIDI
シンプル

DDJ-FLX4では:

MIDI送信のみ:
可能

Audio入力:
不可
→ 別途IF必要
```

### オーディオ接続

```
シンセ出力:

Left / Right:
1/4" TRS

または:

Mono:
1/4" TS

オーディオIF入力:

Input 1/2:
ステレオ

または:

Input 1:
モノラル

推奨:

ステレオシンセ:
L/R → IF Input 1/2

モノラルシンセ:
Output → IF Input 1

DDJ-FLX4:

MIC入力:
使用可能
XLR/TRS

レベル調整:
Gain つまみ
```

---

## External Instrument設定

**Ableton内:**

### デバイス挿入

```
1. 新規MIDIトラック:
   Cmd+T

2. External Instrument挿入:
   Browser > Instruments
   → External Instrument

3. ドラッグ&ドロップ

4. 設定画面表示
```

### MIDI To設定

```
MIDI To:

出力先:
ハードウェアシンセ

選択:

例:
"USB MIDI Device"
"Korg Minilogue XD"

Channel:
1-16

推奨:
Channel 1

確認:

MIDIノート入力:
シンセから音

動作:
MIDI送信成功
```

### Audio From設定

```
Audio From:

入力元:
オーディオIF

選択:

例:
"Scarlett 2i2"
Input: Ext. In 1/2

Gain:

調整:
シンセ音量に合わせ

目標:
-12 〜 -6 dB ピーク

Monitoring:

In:
常に聴こえる

Auto:
録音時のみ
```

### Hardware Latency

```
機能:
遅延補正

原因:

ADC:
アナログ → デジタル変換

DAC:
デジタル → アナログ変換

遅延:
合計 5-15 ms

設定:

Latency:
0 ms (デフォルト)

調整:

1. 同じMIDIノート:
   内部音源 + External

2. 再生:
   ズレ確認

3. Latency調整:
   ズレなくなるまで

推奨:

Buffer 128:
Latency 3-5 ms

Buffer 512:
Latency 10-12 ms

自動測定:
Ableton 11.1+
```

---

## 録音とFreeze

**オーディオ化:**

### リアルタイム録音

```
方法:

1. External Instrument設定済み

2. MIDIノート入力:
   Clip View

3. Session Record (●)

4. 演奏:
   ハードウェアシンセ
   自動録音

5. 停止

結果:
Audio Clip作成

メリット:
リアルタイム演奏
表現豊か

デメリット:
録り直し手間
```

### Freeze

```
機能:
MIDI → Audio変換

手順:

1. MIDIクリップ作成

2. トラック右クリック:
   → Freeze Track

3. 処理:
   自動的にレンダリング

4. Audio再生:
   CPU負荷軽減

5. Flatten:
   Audio化確定

メリット:

編集可能:
Freeze解除で戻る

CPU軽い:
複数トラックでも

推奨:
制作完了後

Unfreeze:
右クリック → Unfreeze Track
→ MIDI編集再開
```

### Bounceに統合

```
方法:

1. MIDIクリップ選択

2. 右クリック:
   → Consolidate (Cmd+J)

3. Audio書き出し:
   内部処理

4. Audio Clip化

メリット:
即座
簡単

推奨:
最終段階
```

---

## 実践例: ハードウェアシンセ統合

**Moog Grandmother:**

### 接続 (10分)

```
機材:

Moog Grandmother:
セミモジュラーシンセ

オーディオIF:
Focusrite Scarlett 2i2

接続:

MIDI:
Mac → Grandmother (USB)

Audio:
Grandmother Out → Scarlett In 1/2
```

### Ableton設定 (5分)

```
1. 新規MIDI Track

2. External Instrument挿入

3. MIDI To:
   Grandmother (USB)
   Channel: 1

4. Audio From:
   Scarlett 2i2
   Ext. In 1/2

5. Monitoring:
   In

6. Gain:
   調整 (-6 dB目標)
```

### 演奏とレコーディング (15分)

```
1. MIDIノート入力:
   C3, E3, G3 (コード)

2. Grandmother:
   Filter つまみ調整
   リアルタイム

3. Record (●)

4. 演奏:
   フィルタースイープ

5. 停止

6. Audio Clip確認:
   フィルター変化記録

7. プリセット保存:
   Grandmother設定メモ
```

---

## 実践例: ドラムマシン統合

**Roland TR-8S:**

### 接続

```
TR-8S:

MIDI:
USB → Mac

Audio:
Main Out L/R → IF Input 1/2

個別Out:
使用しない (シンプルに)
```

### External Instrument

```
1. MIDI Track

2. External Instrument

3. MIDI To:
   TR-8S (USB)

4. Audio From:
   IF Ext. In 1/2

5. MIDI Clip:
   C1 = BD (Kick)
   C#1 = SD (Snare)
   D1 = CH
   等
```

### ドラムパターン

```
1. MIDIクリップ:
   4小節パターン

2. TR-8S:
   音色調整
   つまみ触る

3. 録音:
   Freezeまたはリアルタイム

4. Audio化:
   完成
```

---

## よくある質問

### Q1: レイテンシーがひどい

**A:** Buffer Size下げる

```
問題:
遅延大きい
演奏できない

原因:
Buffer Size大きい

解決:

Preferences > Audio:

Buffer Size:
512 → 128

または:

64 (最小)

効果:
レイテンシー 3-5 ms

注意:
CPU負荷増加

推奨:
録音時のみ128
ミックス時512
```

### Q2: 外部シンセ必要？

**A:** 不要

```
Ableton付属:
十分すぎる

外部シンセ:

必要な人:
すでに持っている
アナログ音質こだわり

不要な人:
初心者
予算限られる

推奨:

最初の1年:
Ableton付属のみ

1年後:
興味あれば検討

予算:

Minilogue XD: ¥70,000
Moog Grandmother: ¥130,000

必要性:
低い
```

### Q3: MIDIだけ送りたい

**A:** External Instrument不要

```
MIDI送信のみ:

1. MIDI Track

2. Output設定:
   Track I/O View表示
   MIDI To: ハードウェア

3. Audio:
   別トラックで録音

または:

ハードウェア:
スタンドアロン使用

Ableton:
Audio録音のみ

推奨:
External Instrument
統合管理
```

---

## まとめ

### External Instrument基礎

```
□ MIDI送信 + Audio受信
□ 1トラックで統合
□ レイテンシー補正
□ Freeze でAudio化
□ USB MIDIが簡単
```

### 接続手順

```
1. MIDI接続 (USB推奨)
2. Audio接続 (IF経由)
3. External Instrument設定
4. MIDI To / Audio From
5. Latency調整
```

### 重要ポイント

```
□ 外部シンセは必須ではない
□ USB MIDI が簡単
□ Buffer Size 128 (録音時)
□ Freeze で軽量化
□ 最初は不要、後から検討
```

---

**次は:** [Presets & Sound Design](./presets-sound-design.md) - プリセット活用とサウンドデザイン基礎
