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

## 外部機器接続の基本原則

**DAWとハードウェアの統合において、信号の流れを正しく理解することは最も重要な基礎知識です。**

### 信号フローの全体像

```
信号フロー概念図:

[DAW (Ableton Live)]
    ↓ MIDI出力
    ↓ (USB MIDI / 5pin DIN / Bluetooth MIDI)
[MIDIインターフェイス / USB接続]
    ↓ MIDI信号
[外部ハードウェア (シンセ / ドラムマシン / サンプラー)]
    ↓ オーディオ出力 (アナログ信号)
[オーディオインターフェイス (ADC)]
    ↓ デジタルオーディオ
[DAW (Ableton Live)]

重要な認識:

MIDI = 制御信号:
  ノートオン/オフ
  ベロシティ (0-127)
  CC (コントロールチェンジ)
  ピッチベンド
  プログラムチェンジ
  → 音そのものではない

Audio = 音声信号:
  アナログ波形
  電気信号
  ADC でデジタル化
  → 実際の音

この2つの信号を正しくルーティングすることが
External Instrument活用の核心
```

### 接続規格の種類と特徴

```
MIDI接続規格:

1. 5pin DIN (従来型):
   速度: 31.25 kbaud
   方向: 単方向 (IN/OUT/THRU)
   ケーブル長: 最大15m推奨
   遅延: 約1ms/3バイト
   メリット: 安定、干渉少ない
   デメリット: 専用IF必要、やや遅い

2. USB MIDI:
   速度: USB 2.0 / 3.0準拠
   方向: 双方向
   ケーブル長: 最大5m (USB 2.0)
   遅延: <1ms
   メリット: 高速、ドライバー不要多い
   デメリット: ケーブル長制限、ノイズ混入リスク

3. Bluetooth MIDI:
   速度: BLE準拠
   方向: 双方向
   距離: 約10m
   遅延: 3-10ms
   メリット: ワイヤレス
   デメリット: レイテンシー大、不安定

4. MIDI 2.0 (最新規格):
   速度: USB/ネットワーク準拠
   解像度: 32bit (従来7bit)
   方向: 双方向
   メリット: 高解像度CC、プロパティ交換
   デメリット: 対応機器少ない (2025年時点)

推奨優先順位:
  USB MIDI > 5pin DIN > Bluetooth MIDI
  (レイテンシーと安定性を重視)

オーディオ接続規格:

1. 1/4" TRS (バランス):
   インピーダンス: ロー
   ノイズ耐性: 高い
   用途: シンセ出力 → IF入力
   推奨ケーブル長: 最大10m

2. 1/4" TS (アンバランス):
   インピーダンス: ハイ
   ノイズ耐性: 低い
   用途: ギターペダル、一部シンセ
   推奨ケーブル長: 最大3m

3. XLR (バランス):
   インピーダンス: ロー
   ノイズ耐性: 最高
   用途: マイク、プロオーディオ
   推奨ケーブル長: 最大50m+

4. S/PDIF (デジタル):
   形式: コアキシャル / オプティカル
   サンプルレート: 最大96kHz
   用途: デジタル接続
   メリット: AD/DA変換不要

5. ADAT (デジタル):
   チャンネル数: 8ch (48kHz)
   用途: チャンネル拡張
   メリット: マルチチャンネル
```

### オーディオインターフェイスの選び方

```
外部機器統合に必要なIF要件:

入力数による分類:

2入力 (ステレオ1系統):
  シンセ1台のみ
  例: Focusrite Scarlett 2i2
  価格: ¥15,000-20,000
  推奨: 初心者、機器1台

4入力 (ステレオ2系統):
  シンセ + ドラムマシン
  例: Focusrite Scarlett 4i4
  価格: ¥25,000-35,000
  推奨: 中級者、機器2台

8入力 (ステレオ4系統):
  複数機器同時使用
  例: Focusrite Scarlett 18i8
  価格: ¥45,000-60,000
  推奨: 上級者、スタジオ運用

16入力以上:
  フルスタジオ
  例: RME Fireface UCX II
  価格: ¥150,000+
  推奨: プロスタジオ

MIDI端子付きIF:

メリット:
  別途MIDIインターフェイス不要
  5pin DIN対応シンセに直接接続

対応機種:
  MOTU M4 (MIDI IN/OUT付)
  RME Babyface Pro FS
  Audient iD44
  Universal Audio Apollo

ADC品質:
  ダイナミックレンジ: 110dB以上推奨
  サンプルレート: 44.1/48kHz (制作標準)
  ビット深度: 24bit以上

バッファサイズとレイテンシー:
  64サンプル: 約1.5ms (CPU負荷高)
  128サンプル: 約3ms (推奨)
  256サンプル: 約6ms (バランス)
  512サンプル: 約12ms (安定)
  1024サンプル: 約23ms (ミックス向け)
```

---

## External Instrumentデバイスの詳細設定

**Ableton LiveのExternal Instrumentは、見た目はシンプルですが多くの設定パラメータがあります。**

### パラメータ完全解説

```
External Instrumentのパラメータ:

1. MIDI To (MIDI出力先):

   ドロップダウンメニュー:
   - 接続されたMIDIデバイス一覧
   - 仮想MIDIポート (IAC Driver等)
   - ネットワークMIDI

   設定のコツ:
   - デバイス名で識別
   - 複数同名デバイスは番号で区別
   - ポート番号も確認

2. Channel (MIDIチャンネル):

   範囲: 1-16

   使い分け:
   - モノティンバー: Ch.1固定
   - マルチティンバー: 音色ごとに分ける
   - ドラムマシン: Ch.10 (GM規格)

   注意:
   - シンセ側の受信チャンネルと一致させる
   - Omni設定のシンセは全チャンネル受信

3. Audio From (オーディオ入力):

   選択肢:
   - オーディオIFの各入力
   - Ext. In (ステレオペア)
   - 個別入力 (モノ)

   設定のコツ:
   - ステレオシンセ → ステレオペア選択
   - モノシンセ → 個別入力選択
   - レベルメーター確認

4. Gain (ゲイン調整):

   範囲: -inf 〜 +6 dB

   目安:
   - ピーク: -6 dB前後
   - 平均: -18 〜 -12 dB
   - クリッピング絶対回避

5. Hardware Latency (遅延補正):

   範囲: -50 〜 +50 ms

   調整方法 (詳細は後述):
   - 正の値: 再生を早める
   - 負の値: 再生を遅らせる
   - 自動検出機能あり (Live 11.1+)
```

### マルチアウト活用法

```
ドラムマシンのマルチアウト:

概要:
  ドラムマシンの個別出力を
  それぞれ別トラックに録音

利点:
  個別EQ/コンプ
  個別ミックス
  高いコントロール

接続例 (TR-8S):

TR-8S出力:
  Main L/R → IF Input 1/2
  Assignable 1 → IF Input 3
  Assignable 2 → IF Input 4
  Assignable 3 → IF Input 5
  Assignable 4 → IF Input 6

Ableton設定:

Track 1: Kick
  External Instrument
  Audio From: Input 3

Track 2: Snare
  External Instrument
  Audio From: Input 4

Track 3: Hi-Hat
  External Instrument
  Audio From: Input 5

Track 4: Mix (残り)
  External Instrument
  Audio From: Input 1/2

MIDIルーティング:
  全トラック → TR-8S (同じMIDI To)
  チャンネルで分けるか
  ノートで分ける

注意:
  MIDI送信は1トラックのみで行い
  他トラックはAudio受信のみにする方法も有効
  → MIDIループ回避
```

---

## レイテンシー補正の詳細

**外部機器を使う上で最も厄介な問題がレイテンシーです。正確な補正方法を理解しましょう。**

### レイテンシーの発生源

```
レイテンシーチェーン:

1. DAWバッファ (出力):
   Buffer Size依存
   128サンプル @ 44.1kHz = 2.9ms
   256サンプル @ 44.1kHz = 5.8ms

2. DAC変換:
   D/A変換処理
   約0.5-1.0ms

3. アナログ伝送:
   ケーブル遅延
   ほぼ無視可能 (光速)

4. 機器内部処理:
   シンセ内部のバッファ
   機種依存: 0.5-5ms

5. ADC変換:
   A/D変換処理
   約0.5-1.0ms

6. DAWバッファ (入力):
   Buffer Size依存
   出力と同じ

合計RTL (Round-Trip Latency):
  最小: 約6ms (Buffer 64)
  一般: 約12ms (Buffer 128)
  安全: 約25ms (Buffer 512)

人間の知覚閾値:
  演奏: 10ms以下が理想
  聴取: 30ms以下で違和感なし
  同期: 3ms以下が理想
```

### 精密なレイテンシー測定方法

```
方法1: ループバックテスト

手順:
  1. オーディオIFの出力を入力に接続
     (Output 1 → Input 1)

  2. Abletonで新規Audioトラック
     Input: Ext. In 1
     Monitor: In

  3. 別トラックにクリック音配置
     Output: Ext. Out 1

  4. 両トラックを同時再生＆録音

  5. 波形のズレをサンプル単位で測定

  6. サンプル数 ÷ サンプルレート = 秒数
     例: 256 samples ÷ 44100 = 5.8ms

方法2: 内部vs外部比較

手順:
  1. 同じMIDIノートを用意

  2. Track A: Operator (内部音源)
     Track B: External Instrument (外部)

  3. 両方を同時に録音

  4. 波形のズレを測定

  5. Hardware Latencyパラメータに入力

方法3: 自動補正 (Ableton 11.1+)

手順:
  1. External Instrumentデバイス選択
  2. "Latency" の右の矢印アイコン
  3. "Detect Latency" クリック
  4. テスト信号が自動送信
  5. 測定結果が自動入力

注意:
  - 自動測定は完璧ではない
  - 微調整は手動で行うことを推奨
  - 温度変化でアナログ機器は特性が変わる
```

### Driver Error Compensationとの関係

```
Ableton Live の遅延補正体系:

1. Overall Latency (全体遅延):
   Options > Preferences > Audio
   "Overall Latency" 表示値
   → DAWが認識しているRTL

2. Driver Error Compensation:
   Preferences > Audio
   手動入力: サンプル単位
   → オーディオドライバの誤差補正
   → 全トラックに影響

3. Track Delay (トラック遅延):
   各トラックのミキサーセクション
   D ボタン → ms単位で調整
   → 個別トラックの微調整

4. External Instrument Hardware Latency:
   デバイス内パラメータ
   → 外部機器専用の補正
   → 他のトラックには影響しない

推奨設定フロー:

Step 1: Buffer Sizeを決定
  → 録音時128、ミックス時512

Step 2: Driver Error Compensationを測定
  → ループバックテストで全体補正

Step 3: External Instrumentごとに
  Hardware Latencyを個別設定
  → 各機器の固有遅延を補正

Step 4: 必要に応じてTrack Delayで微調整
  → 最終的なタイミング合わせ
```

---

## MIDI/オーディオルーティングの応用

**External Instrumentを超えた高度なルーティング技術を習得しましょう。**

### 複数External Instrumentの同時使用

```
シナリオ:
  シンセ3台 + ドラムマシン1台

必要機材:
  オーディオIF: 8入力以上
  MIDIインターフェイス or USB MIDI ×4

Ableton設定:

Track 1: Bass Synth (Moog Sub 37)
  External Instrument
  MIDI To: Sub 37 (USB), Ch.1
  Audio From: IF Input 1/2

Track 2: Pad Synth (Roland JUNO-X)
  External Instrument
  MIDI To: JUNO-X (USB), Ch.1
  Audio From: IF Input 3/4

Track 3: Lead Synth (Sequential Prophet-6)
  External Instrument
  MIDI To: Prophet-6 (USB), Ch.1
  Audio From: IF Input 5/6

Track 4: Drums (Elektron Digitakt)
  External Instrument
  MIDI To: Digitakt (USB), Ch.10
  Audio From: IF Input 7/8

ルーティング図:

Mac/PC
  ├─ USB MIDI → Sub 37
  ├─ USB MIDI → JUNO-X
  ├─ USB MIDI → Prophet-6
  └─ USB MIDI → Digitakt

Audio IF (8in)
  ├─ Input 1/2 ← Sub 37 L/R
  ├─ Input 3/4 ← JUNO-X L/R
  ├─ Input 5/6 ← Prophet-6 L/R
  └─ Input 7/8 ← Digitakt L/R

注意事項:
  - 各シンセのMIDI受信チャンネル確認
  - USB MIDI帯域に注意 (USBハブ品質)
  - 電源ノイズ対策 (パワーサプライ統一)
  - グラウンドループに注意
```

### MIDIチェイン接続 (Daisy Chain)

```
5pin MIDI使用時のチェイン:

接続:
  Mac → MIDI IF OUT → シンセA (IN)
  シンセA (THRU) → シンセB (IN)
  シンセB (THRU) → シンセC (IN)

チャンネル割り当て:
  シンセA: Ch.1受信
  シンセB: Ch.2受信
  シンセC: Ch.3受信

Ableton設定:
  Track 1: MIDI To → MIDI IF, Ch.1
  Track 2: MIDI To → MIDI IF, Ch.2
  Track 3: MIDI To → MIDI IF, Ch.3

注意:
  THRU接続は最大3台まで推奨
  4台以上はMIDIスプリッター使用

  THRU接続の問題:
  - 信号劣化 (各接続で微小劣化)
  - 遅延累積 (各段約1ms)
  - ジッター増加

推奨:
  3台以上 → MIDIパッチベイ or USBハブ
  例: iConnectivity mioXL
```

### 仮想MIDIポートの活用

```
macOS: IAC Driver

設定:
  1. Audio MIDI設定を開く
     (アプリケーション > ユーティリティ)

  2. IAC Driverをダブルクリック

  3. "デバイスはオンライン" にチェック

  4. ポートを追加:
     "Bus 1", "Bus 2" 等

活用例:

DAW間接続:
  Ableton → IAC Bus 1 → Logic Pro
  → 2つのDAWでMIDI同期

アプリ間:
  Ableton → IAC Bus 1 → Max/MSP
  → カスタムMIDI処理

External Instrumentと組み合わせ:
  Ableton External Instrument
  MIDI To: IAC Driver Bus 1
  → 別アプリでMIDI処理
  → 処理後Audio返送

Windows: loopMIDI

設定:
  1. loopMIDI インストール
  2. 仮想ポート作成
  3. 同様の使い方が可能
```

---

## ハードウェアシンセ連携の深掘り

**各種ハードウェアシンセの特性を理解し、Ableton Liveとの連携を最大限に活かす方法を解説します。**

### アナログシンセとデジタルシンセの違い

```
アナログシンセの特徴:

音の特性:
  - 温かみのある音色
  - 微妙なピッチのゆらぎ (アナログドリフト)
  - 倍音が豊か
  - フィルターレゾナンスが自然
  - 各個体で音が微妙に異なる

代表機種:
  Moog Subsequent 37: ¥200,000
    - モノフォニック/パラフォニック
    - 2オシレーター
    - ラダーフィルター
    - USB MIDI対応

  Sequential Prophet-6: ¥400,000
    - 6ボイスポリフォニック
    - VCO (本物のアナログオシレーター)
    - Curtis フィルター
    - USB MIDI対応

  Arturia MiniBrute 2S: ¥80,000
    - セミモジュラー
    - シーケンサー内蔵
    - パッチベイ
    - USB MIDI対応

デジタルシンセの特徴:

音の特性:
  - クリアで正確な音色
  - 安定したピッチ
  - 多彩な波形テーブル
  - プリセット豊富
  - 複雑なモジュレーション可能

代表機種:
  Roland JUNO-X: ¥180,000
    - モデリングエンジン
    - ZEN-Core テクノロジー
    - 過去のJUNOシリーズをモデリング
    - USB MIDI/Audio対応

  Korg Prologue: ¥200,000
    - アナログ+デジタルハイブリッド
    - Multi Engine (カスタムオシレーター)
    - 16ボイス
    - USB MIDI対応

  Waldorf Blofeld: ¥60,000
    - ウェーブテーブルシンセ
    - 25ボイス
    - マルチティンバー
    - USB MIDI対応

External Instrumentでの使い分け:

アナログシンセの場合:
  - つまみの動きをCC録音
  - Freeze推奨 (音が毎回微妙に変わるため)
  - モニタリング: 常にIn
  - Gain: 慎重に設定 (出力レベルが不安定な場合あり)

デジタルシンセの場合:
  - プリセット切替をProgram Change送信
  - USB Audio対応ならデジタル転送も可能
  - Freeze不要 (毎回同じ音が出る)
  - Gain: 安定しやすい
```

### MIDIコントロールチェンジ (CC) マッピング

```
External Instrumentを使ったCC送信:

基本:
  MIDIクリップ内のエンベロープ
  → CC値を時間軸で記録

主要CC番号:

CC1: モジュレーションホイール
  用途: ビブラート、LFO深度
  範囲: 0-127

CC7: ボリューム
  用途: 音量制御
  範囲: 0-127

CC10: パン
  用途: 定位
  範囲: 0 (L) - 64 (C) - 127 (R)

CC11: エクスプレッション
  用途: 音量変化 (CC7との乗算)
  範囲: 0-127

CC64: サスティンペダル
  用途: 音の持続
  範囲: 0 (OFF) / 127 (ON)

CC74: フィルターカットオフ (一般的)
  用途: ブライトネス
  範囲: 0-127

CC71: レゾナンス (一般的)
  用途: フィルター共振
  範囲: 0-127

Ableton Live での CC 記録方法:

方法1: リアルタイム録音
  1. MIDIコントローラーのつまみをマッピング
  2. Record中につまみを動かす
  3. CCオートメーションが記録される

方法2: エンベロープ描画
  1. MIDIクリップを開く
  2. Envelope表示 (E ボタン)
  3. ドロップダウンから "MIDI Ctrl" 選択
  4. CC番号を選択
  5. マウスで描画

方法3: Max for Live デバイス
  1. "LFO" MIDIエフェクト
  2. Map → CC番号
  3. LFOが自動的にCC値を変化

シンセ固有のCC設定:

注意:
  - シンセによってCC番号の割り当てが異なる
  - マニュアルで必ず確認
  - MIDI Learnで自動マッピング可能な機種もある
  - NRPNを使う機種もある (CC98/99/6/38)
```

### プログラムチェンジの活用

```
外部シンセの音色をAbleton内から切り替え:

設定方法:
  1. MIDIクリップを開く
  2. Envelope表示
  3. "MIDI Ctrl" → "Program Change"
  4. クリップ先頭にプログラム番号を記入

活用シーン:

セクションごとの音色切替:
  Intro: Program 1 (Pad)
  Verse: Program 5 (Bass)
  Chorus: Program 12 (Lead)
  → 1つのExternal Instrumentトラックで
    複数音色を切り替え

バンクセレクト:
  CC0 (MSB) + CC32 (LSB) + Program Change
  → 128以上の音色にアクセス

注意:
  - 切替時に一瞬音が途切れる場合あり
  - クリップの先頭で切替推奨
  - シンセの応答速度に依存
```

---

## ドラムマシン統合の応用テクニック

**ドラムマシンをAbleton Liveと連携する際の高度なテクニックを解説します。**

### 主要ドラムマシンの接続ガイド

```
Elektron Digitakt:

接続:
  USB: MIDI + Audio (Overbridge対応)

Overbridge使用時:
  メリット:
    - 個別トラック出力 (8トラック)
    - USB Audio転送 (ADC不要)
    - 低レイテンシー
    - パラメータ自動マッピング

  設定:
    1. Overbridge Engineインストール
    2. AbletonのAudio Preferencesで
       Overbridge選択
    3. 専用VST/AUプラグイン使用
    → External Instrument不要

Overbridge不使用時:
  MIDI: USB経由
  Audio: Main Out → IF Input 1/2
  → External Instrument使用

Roland TR-8S:

接続:
  USB: MIDI + Audio (USB Audio Class Compliant)

USB Audio使用:
  メリット:
    - 14ch USB Audio出力
    - 各パート個別出力
    - ADC不要
    - macOS標準ドライバで動作

  注意:
    - Aggregate Device作成が必要な場合あり
    - Abletonの入力でTR-8S選択

Arturia DrumBrute Impact:

接続:
  USB: MIDI
  Audio: Main Out → IF Input 1/2
  個別Out: 各パート → IF入力

特徴:
  - アナログドラムシンセ
  - 12パート
  - 個別出力あり
  - USB MIDIのみ (USB Audio非対応)

Teenage Engineering OP-1 Field:

接続:
  USB: MIDI + Audio (USB Audio)

特徴:
  - ポータブル
  - USB Audio 対応
  - テープレコーダー機能
  - External Instrumentで統合可能
```

### ドラムマシンのパラレル処理

```
DAW内でのドラムパラレルプロセッシング:

概要:
  ドラムマシンのオーディオを
  複数のトラックに分岐して並列処理

設定:

Track 1: Drum Machine (External Instrument)
  MIDI To: TR-8S
  Audio From: IF Input 1/2
  Output: "Drum Bus" に送信

Track 2: Drum Parallel (Audio Track)
  Input: Track 1 (Post FX)
  → Compressor (高圧縮)
  → Saturator
  → Drum Bus と並列ミックス

Track 3: Drum Bus (Group Track)
  ← Track 1 + Track 2
  → Master

パラレルコンプの設定:
  Ratio: 10:1 以上
  Attack: 1-5ms
  Release: 50-100ms
  Threshold: -30dB
  Mix: 30-50%

効果:
  - パンチ感の向上
  - ダイナミクス制御
  - 音の厚み追加
  - 原音のトランジェントを維持
```

---

## エフェクトペダルの活用

**ギターエフェクトペダルを音楽制作に活用する方法を解説します。External Audio Effectデバイスを使います。**

### External Audio Effectとは

```
External Audio Effect:
  External Instrumentの「エフェクト版」

役割:
  Audio送信: Ableton → 外部エフェクト
  Audio受信: 外部エフェクト → Ableton

使い方:
  任意のオーディオトラックに挿入
  → 内部音源の音を外部エフェクトに通す

設定パラメータ:
  Audio To: IF出力 (エフェクトへ)
  Audio From: IF入力 (エフェクトから)
  Dry/Wet: ミックス量
  Hardware Latency: 遅延補正
  Gain: 入出力レベル
```

### エフェクトペダルの接続

```
シグナルチェーン:

[Ableton Audio Track]
  ↓ External Audio Effect
  ↓ Audio To: IF Output 3/4
[オーディオIF Output 3/4]
  ↓ ケーブル (1/4" TS)
[エフェクトペダル Input]
  ↓ エフェクト処理
[エフェクトペダル Output]
  ↓ ケーブル (1/4" TS)
[オーディオIF Input 3/4]
  ↓ Audio From: IF Input 3/4
[Ableton Audio Track]

レベル問題:

オーディオIF出力: ラインレベル (+4dBu)
ペダル入力: インストゥルメントレベル (-20dBu)
→ レベル差あり

対策:
  1. IF出力のボリュームを下げる
  2. リアンプボックス使用
     (Radial ProRMP等)
  3. Abletonでゲイン調整
     (Utility プラグイン)

推奨ペダル (音楽制作向け):

ディレイ:
  Strymon Timeline: ¥60,000
  Boss DD-500: ¥35,000

リバーブ:
  Strymon BigSky: ¥60,000
  Eventide Space: ¥55,000

ディストーション:
  Pro Co RAT2: ¥12,000
  Electro-Harmonix Big Muff: ¥10,000

コーラス/モジュレーション:
  Boss CE-2W: ¥20,000
  Strymon Mobius: ¥60,000
```

### ペダルチェーンの構築

```
複数ペダルの直列接続:

IF Out 3 → [Compressor] → [Distortion]
  → [Chorus] → [Delay] → [Reverb] → IF In 3

External Audio Effect設定:
  Audio To: Output 3
  Audio From: Input 3

注意:
  - ペダルの順序が音に大きく影響
  - 一般的な順序: コンプ → 歪み → モジュレーション → ディレイ → リバーブ
  - 実験して最適な順序を探す

ステレオペダルの場合:
  IF Out 3/4 → [Stereo Pedal] → IF In 3/4
  External Audio Effect:
    Audio To: Output 3/4
    Audio From: Input 3/4

Wet/Dry ミックス:
  方法1: ペダル内蔵のDry/Wetつまみ
  方法2: External Audio EffectのDry/Wetパラメータ
  方法3: パラレル処理 (Rack使用)
```

---

## ミキサー経由接続

**アナログミキサーをDAWと組み合わせた、ハイブリッドワークフローを解説します。**

### アナログミキサーの役割

```
なぜアナログミキサーを使うのか:

理由1: 複数機器のモニタリング
  - DAWなしで全機器を聴ける
  - 素早い音出し確認
  - 電源ONですぐモニタリング

理由2: サミングミックス
  - アナログ回路でのミックス
  - 独特の「接着感」
  - DAW内蔵ミックスとは異なる質感

理由3: ルーティングの柔軟性
  - AUXセンド/リターン
  - インサート
  - グループバス
  - マスターセクション

推奨ミキサー:

小規模 (シンセ2-3台):
  Mackie Mix8: ¥10,000
  Allen & Heath ZEDi-10FX: ¥30,000

中規模 (4-8台):
  Soundcraft Signature 12 MTK: ¥60,000
  Allen & Heath SQ-5: ¥350,000

MTK (マルチトラック) ミキサー:
  特徴: 各チャンネルをUSBで個別にDAWへ送信
  メリット: ミキサー経由でマルチトラック録音
  例: Soundcraft Signature 12 MTK
      → 12ch USB Audio出力
```

### ミキサー + DAWのルーティング

```
基本パターン: ミキサー → DAW

ミキサーの役割:
  全ハードウェアの集約
  モニタリング
  ラフミックス

接続:
  シンセA → ミキサー Ch.1/2
  シンセB → ミキサー Ch.3/4
  ドラムマシン → ミキサー Ch.5/6
  ミキサー Main Out → IF Input 1/2

Ableton:
  Audio Track
  Input: IF 1/2
  → ミキサーのステレオミックスを録音

欠点:
  - 個別トラック録音不可
  - ミックスをやり直せない
  - 一発録り

改善パターン: MTKミキサー使用

接続:
  シンセA → ミキサー Ch.1/2
  シンセB → ミキサー Ch.3/4
  ドラムマシン → ミキサー Ch.5/6
  ミキサー USB → Mac

Ableton:
  Track 1: Input MTK Ch.1/2 (シンセA)
  Track 2: Input MTK Ch.3/4 (シンセB)
  Track 3: Input MTK Ch.5/6 (ドラム)
  → 個別マルチトラック録音可能
```

---

## マルチティンバー設定

**1台のシンセで複数音色を同時に鳴らすマルチティンバー設定を解説します。**

### マルチティンバーとは

```
概要:
  1台のハードウェアシンセで
  複数のMIDIチャンネルを受信し
  各チャンネルに異なる音色を割り当てる

例:
  Ch.1: ピアノ音色
  Ch.2: ベース音色
  Ch.3: ストリングス音色
  → 1台のシンセで3パート同時演奏

対応機種例:
  Korg Prologue: 2ティンバー (Dual)
  Waldorf Blofeld: 16ティンバー
  Roland FANTOM: 16ティンバー
  Yamaha MODX: 16ティンバー
  Access Virus TI: 16ティンバー

Ableton設定:

Track 1: Piano
  External Instrument
  MIDI To: Blofeld (USB), Ch.1
  Audio From: IF 1/2 (共通)

Track 2: Bass
  External Instrument
  MIDI To: Blofeld (USB), Ch.2
  Audio From: IF 1/2 (共通)

Track 3: Strings
  External Instrument
  MIDI To: Blofeld (USB), Ch.3
  Audio From: IF 1/2 (共通)

注意:
  Audio Fromが全トラック同じ場合
  → 個別のミックスバランスはシンセ側で調整
  → DAWでは全音色が1つのステレオ信号

解決策:
  マルチアウト対応シンセを使う
  → 各パートを個別出力に割り当て
```

---

## CV/Gate接続

**ユーロラック/モジュラーシンセとの接続方法であるCV/Gateについて解説します。**

### CV/Gateとは

```
概要:
  MIDIとは異なるアナログ制御規格

CV (Control Voltage):
  制御電圧
  - ピッチ: 1V/Oct (1ボルトで1オクターブ)
  - モジュレーション: 0-5V または 0-10V
  - 連続的なアナログ信号

Gate:
  ゲート信号
  - ノートオン/オフの代わり
  - High (+5V) = ノートオン
  - Low (0V) = ノートオフ
  - デジタル的なON/OFF

Trigger:
  トリガー信号
  - 短いパルス
  - エンベロープの起動等
  - ドラムモジュール向き

MIDIとの違い:

MIDI:
  デジタル
  離散値 (0-127)
  チャンネル16個
  多機能

CV/Gate:
  アナログ
  連続値 (無段階)
  1信号=1パラメータ
  シンプル・直感的
```

### AbletonからCV/Gate出力

```
方法1: DC-coupled オーディオIF

対応IF:
  Expert Sleepers ES-9: ¥80,000
  MOTU UltraLite mk5: ¥60,000
  RME Fireface UCX II: ¥150,000

仕組み:
  オーディオ出力からDC信号を出力
  → CV/Gate信号として使用

Ableton設定:
  1. CV Toolsパック (無料) インストール
  2. MIDIトラックに "CV Instrument" 配置
  3. Pitch: Output 3
  4. Gate: Output 4
  5. 対象モジュールのCV/Gate入力に接続

方法2: MIDI-CV変換器

変換器:
  Expert Sleepers FH-2: ¥50,000
  Kenton Pro Solo MkIII: ¥30,000
  Doepfer A-190-5: ¥20,000

接続:
  Ableton → USB MIDI → 変換器
  変換器 CV Out → モジュラーシンセ CV In
  変換器 Gate Out → モジュラーシンセ Gate In

Ableton設定:
  External Instrument
  MIDI To: 変換器 (USB)
  Audio From: IF入力 (モジュラーの出力)

方法3: CV対応シンセ経由

一部シンセにMIDI-CV変換機能あり:
  Moog Grandmother: MIDI → CV/Gate出力
  Arturia MiniBrute 2S: MIDI → CV/Gate出力

接続:
  Ableton → USB MIDI → Grandmother
  Grandmother CV Out → モジュラーシンセ
```

---

## ライブパフォーマンス設定

**External Instrumentを使ったライブパフォーマンスのための設定とノウハウを解説します。**

### ライブ用セッティングの基本

```
ライブパフォーマンスの要件:

1. 安定性最優先:
   - 音が途切れない
   - MIDI接続が落ちない
   - レイテンシーが許容範囲内

2. 素早い切替:
   - シーン切替でパラメータ一括変更
   - プログラムチェンジで音色切替
   - フットスイッチ活用

3. フェイルセーフ:
   - 機器が落ちた時の代替策
   - バックアップ音源の準備
   - 最悪の場合の復旧手順

Buffer Size設定:

ライブ中:
  推奨: 256サンプル
  理由: 安定性と低レイテンシーのバランス

  128サンプルは危険:
    CPU負荷でドロップアウトの可能性
    ライブ中のクラッシュは致命的

  512サンプルでは:
    レイテンシーが大きすぎる
    リアルタイム演奏に支障

ライブ用テンプレート:

Track 1-4: External Instrument (各シンセ)
Track 5-8: バックアップ内部音源
  → 同じMIDIクリップを共有
  → External落ちたら即座に内部に切替

Track 9: Click Track
  → モニター出力のみ (PAに送らない)

Track 10: Master FX
  → リバーブ/ディレイのリターン
```

### Session Viewの活用

```
ライブでのSession View構成:

横軸 (トラック):
  [Synth A] [Synth B] [Drums] [FX] [Master]

縦軸 (シーン):
  Scene 1: Intro
  Scene 2: Build Up
  Scene 3: Drop
  Scene 4: Breakdown
  Scene 5: Second Drop
  Scene 6: Outro

各シーンに設定するもの:
  - MIDIクリップ (各トラック)
  - テンポ変更 (必要に応じ)
  - プログラムチェンジ (音色切替)
  - CCオートメーション (フィルター等)

シーン切替方法:
  方法1: マウスクリック
  方法2: MIDIコントローラー (LaunchPad等)
  方法3: フットスイッチ (ギタリスト向き)
  方法4: Follow Actions (自動進行)

Follow Actions活用:
  各クリップに "Next" アクション
  → 自動的に次のシーンへ
  → 手動介入なしでセット進行

Tempo設定:
  シーンごとにBPMを設定可能
  Master Tempo欄に入力
  → シーン切替でテンポ自動変更
```

---

## ハイブリッドセットアップ

**DAW内部音源と外部ハードウェアを組み合わせた、実践的なセットアップを解説します。**

### 構成パターン別ガイド

```
パターン1: ミニマルセットアップ (初心者向け)

機材:
  - MacBook Pro
  - Ableton Live
  - オーディオIF (Scarlett 2i2)
  - ハードウェアシンセ 1台

構成:
  Track 1-6: Ableton内部音源
  Track 7: External Instrument (シンセ)
  Track 8: Audio (シンセ録音済み素材)

メリット:
  - シンプル
  - 持ち運びやすい
  - トラブル少ない

パターン2: スタンダードセットアップ (中級者)

機材:
  - MacBook Pro / Mac Studio
  - Ableton Live
  - オーディオIF (Scarlett 4i4)
  - シンセ 2台
  - ドラムマシン 1台

構成:
  Track 1-4: Ableton内部音源
  Track 5: External Instrument (シンセA - Bass)
  Track 6: External Instrument (シンセB - Lead)
  Track 7: External Instrument (ドラムマシン)
  Track 8: Resampling Track

メリット:
  - アナログとデジタルの融合
  - 十分な表現力
  - 管理可能な複雑さ

パターン3: フルハイブリッドセットアップ (上級者)

機材:
  - Mac Studio
  - Ableton Live
  - オーディオIF (RME Fireface UCX II)
  - シンセ 3-5台
  - ドラムマシン 1-2台
  - エフェクトペダル
  - アナログミキサー

構成:
  Track 1-4: Ableton内部音源
  Track 5-9: External Instrument (各ハードウェア)
  Track 10: External Audio Effect (ペダル)
  Track 11-12: Resampling / Recording
  Bus: アナログサミング

メリット:
  - 最大限の音質
  - 豊かな表現
  - プロフェッショナルな音像
```

---

## トラブルシューティング

**External Instrument使用時に発生しやすい問題と解決策をまとめます。**

### 音が出ない場合

```
チェックリスト:

1. MIDI接続確認:
   □ ケーブル接続 (USB or 5pin)
   □ シンセの電源ON
   □ MIDI To設定が正しいデバイスを指している
   □ MIDIチャンネルがシンセ側と一致

2. Audio接続確認:
   □ オーディオケーブル接続
   □ IF入力にシグナルあり (IFのメーターで確認)
   □ Audio Fromが正しい入力を指している
   □ Gainが0でない

3. Ableton設定確認:
   □ トラックがArmed (録音待機)
   □ Monitor: In または Auto
   □ トラックのOutput: Master
   □ Master出力がミュートされていない

4. シンセ側確認:
   □ 音量がゼロでない
   □ プログラムが無音でない
   □ MIDIチャンネル設定が正しい
   □ Local Control: ON (スタンドアロン時)
      または OFF (DAW連携時)
```

### MIDIフィードバックループ

```
症状:
  - MIDIノートが無限にループ
  - シンセが暴走
  - 音が止まらない

原因:
  シンセのMIDI OUT → Ableton → シンセのMIDI IN
  → 無限ループ

解決:

方法1: シンセのLocal Control OFF
  シンセの設定メニュー
  Local Control: OFF
  → シンセの鍵盤がMIDI OUTに送信しない

方法2: AbletonのMIDI Input無効化
  Preferences > Link Tempo MIDI
  Input: シンセの "Track" をOFF
  → シンセからのMIDIを受信しない

方法3: MIDI Filterを使用
  Track設定で特定チャンネルのみ受信
  → ループを断ち切る

予防:
  External Instrument使用時は
  必ずシンセのLocal Control OFFにする
```

### ノイズ・ハム問題

```
症状:
  - ブーンという低域ノイズ (50/60Hz)
  - ジーというデジタルノイズ
  - パチパチというクリックノイズ

原因と対策:

グラウンドループ:
  原因: 機器間の電位差
  対策:
    - 全機器を同じ電源タップに接続
    - グラウンドリフトスイッチ使用
    - DIボックスのGround Lift

USBノイズ:
  原因: PCのUSBからノイズ混入
  対策:
    - USB Isolator使用
    - 別電源のUSBハブ使用
    - USB AudioよりアナログIF推奨

電源ノイズ:
  原因: スイッチングアダプター
  対策:
    - リニア電源アダプター使用
    - パワーコンディショナー導入
    - デジタル機器とアナログ機器の電源分離

ケーブルノイズ:
  原因: シールド不良、長すぎ
  対策:
    - バランスケーブル (TRS) 使用
    - ケーブル長を最短に
    - 高品質ケーブル使用
```

---

## レコーディングワークフロー

**External Instrumentを使った効率的なレコーディングの手順を解説します。**

### ステップバイステップの録音手順

```
Phase 1: プリプロダクション

1. 曲の構成を決める
   - セクション (Intro/Verse/Chorus等)
   - コード進行
   - テンポ/キー

2. MIDIクリップを事前作成
   - 各パートのノートを打ち込み
   - CCオートメーションはまだ不要
   - 構成が固まってから本番録音

Phase 2: サウンドメイキング

3. 各External Instrumentの音色を決定
   - シンセのパッチ選択/作成
   - フィルター、エンベロープ調整
   - エフェクト設定

4. ラフミックスのバランス確認
   - 各トラックのレベル調整
   - 帯域の被り確認
   - パン配置

Phase 3: レコーディング

5. Buffer Sizeを128に設定
   - 低レイテンシーで録音

6. CCオートメーション録音
   - フィルタースイープ
   - モジュレーション
   - 表現力のある演奏

7. Freeze Track
   - 各External Instrumentをフリーズ
   - CPUリソース確保
   - 音色の確定

Phase 4: ポストプロダクション

8. Flatten (必要に応じ)
   - Audio化確定
   - ハードウェア不要に

9. ミックス作業
   - Buffer Size 512に戻す
   - EQ/コンプ等の処理
   - 空間系エフェクト追加

10. マスタリング
    - 最終音量調整
    - リミッター
    - 書き出し
```

---

## 同期設定の詳細

**Ableton Liveと外部機器のクロック同期について詳しく解説します。**

### MIDI Clock同期

```
Ableton Live をマスタークロックとして使用:

設定:
  Preferences > Link Tempo MIDI

  MIDI Clock送信:
    Output: 対象デバイス
    Sync: ON

  送信される情報:
    - MIDI Clock (24ppqn)
    - Start/Stop/Continue
    - Song Position Pointer

対応機器:
  ほぼ全てのハードウェアシンセ/ドラムマシン
  シーケンサー付き機器は内部SEQが同期

同期精度:
  MIDI Clock: 約0.4ms/tick (120BPM時)
  十分な精度だが完璧ではない

外部機器がマスターの場合:

設定:
  Preferences > Link Tempo MIDI
  Input: 対象デバイス
  Sync: ON

  Ableton Live がスレーブ動作:
    テンポが外部機器に追従
    Start/Stopも追従

推奨:
  Ableton Live をマスターにする方が安定
  → DAWがテンポ管理の中心
```

### Ableton Link

```
概要:
  ネットワーク経由のテンポ同期技術
  Wi-Fi/有線LANで複数デバイス同期

対応:
  Ableton Live
  iOS/Androidアプリ多数
  Max/MSP
  一部ハードウェア (Elektron等)

設定:
  Preferences > Link Tempo MIDI
  Link: ON (Enable)

メリット:
  - 設定が非常に簡単
  - 複数デバイスの同期
  - ビート同期 (タイミング合致)
  - テンポ変更が全体に反映

デメリット:
  - Start/Stopの同期は完璧でない
  - ネットワーク依存
  - レイテンシーがMIDI Clockより大きい場合あり

活用シーン:
  - 複数PCのAbleton Live同期
  - iPad上の音楽アプリとの同期
  - ジャムセッション
```

---

## 実践セットアップ例

**具体的な機材構成と設定手順をジャンル別に紹介します。**

### テクノ制作セットアップ

```
コンセプト:
  ハードウェアドラムマシン + アナログシンセベース
  + DAW内シンセでハイブリッド制作

機材:
  Mac + Ableton Live 12 Suite
  Focusrite Scarlett 4i4
  Roland TR-8S (ドラムマシン)
  Moog Minitaur (アナログベースシンセ)

接続:
  TR-8S: USB (MIDI + Audio)
  Minitaur: USB MIDI + 1/4" Out → IF Input 3

Ableton構成:
  Track 1: TR-8S (External Instrument)
    MIDI To: TR-8S, Ch.10
    Audio From: TR-8S (USB Audio)
    → キック、スネア、ハイハット

  Track 2: Minitaur Bass (External Instrument)
    MIDI To: Minitaur, Ch.1
    Audio From: IF Input 3 (モノ)
    → ディープベースライン

  Track 3: Wavetable (内部)
    → パッド、アトモスフィア

  Track 4: Operator (内部)
    → リードシンセ

  Track 5: Audio (リサンプリング)
    → ループ作成用

ワークフロー:
  1. TR-8Sでリズム構築
  2. Minitaurでベースライン
  3. DAW内音源でメロディ/パッド
  4. Freeze → Flatten → ミックス
```

### ハウスミュージック制作セットアップ

```
コンセプト:
  ウォームなアナログサウンド
  + ボーカルサンプル
  + クラシックなドラムサウンド

機材:
  Mac + Ableton Live 12 Suite
  MOTU M4 (MIDI IN/OUT付)
  Korg Minilogue XD (ポリシンセ)
  Arturia DrumBrute Impact (アナログドラム)

接続:
  Minilogue XD: USB MIDI + Out L/R → IF Input 1/2
  DrumBrute: MIDI OUT (IF) + Main Out → IF Input 3/4

Ableton構成:
  Track 1: DrumBrute (External Instrument)
    MIDI To: MOTU M4 MIDI OUT, Ch.1
    Audio From: IF Input 3/4
    → 4つ打ちキック + パーカッション

  Track 2: Minilogue Chords (External Instrument)
    MIDI To: Minilogue XD (USB), Ch.1
    Audio From: IF Input 1/2
    → ディスコ風コード

  Track 3: Simpler (内部)
    → ボーカルチョップ

  Track 4: Analog (内部)
    → サブベース

  Track 5: Return A
    → Reverb (内部)

  Track 6: Return B
    → Delay (内部)
```

### アンビエント制作セットアップ

```
コンセプト:
  長いリバーブテイル
  + テクスチャー
  + ペダルエフェクト

機材:
  Mac + Ableton Live 12 Suite
  RME Babyface Pro FS
  Moog Grandmother (セミモジュラー)
  Strymon BigSky (リバーブペダル)
  Strymon Timeline (ディレイペダル)

接続:
  Grandmother: USB MIDI + Out L/R → IF Input 1/2
  BigSky: IF Output 3/4 → BigSky → IF Input 3/4
  Timeline: BigSkyの前段に直列接続

Ableton構成:
  Track 1: Grandmother (External Instrument)
    MIDI To: Grandmother (USB), Ch.1
    Audio From: IF Input 1/2
    → ドローンパッド、ゆっくりした進行

  Track 2: Audio (Grandermother録音済み素材)
    → テクスチャー、アンビエンス

  Track 3: Wavetable (内部)
    → 深いパッド

  Track 4: FX Send (External Audio Effect)
    Audio To: IF Output 3/4
    Audio From: IF Input 3/4
    → Timeline → BigSky経由の空間処理

  Track 5: Corpus (内部エフェクト)
    → レゾネーター

ワークフロー:
  1. Grandmotherで基本フレーズ作成
  2. Freeze + Flatten で素材化
  3. 素材をペダル通しでリバーブ/ディレイ付加
  4. 内部音源でレイヤー追加
  5. ロングフォームで構成
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

### ハイブリッド制作のポイント

```
□ アナログとデジタルの長所を組み合わせる
□ バックアップ音源を常に用意
□ レイテンシー補正を正確に行う
□ Freezeで随時オーディオ化
□ 電源/グラウンドのノイズ対策を忘れずに
□ セッション前にすべての接続をテスト
□ MIDIフィードバックループに注意
□ ケーブル管理を整理整頓する
```

---

**次は:** [Presets & Sound Design](./presets-sound-design.md) - プリセット活用とサウンドデザイン基礎
