# MIDI基礎

音階情報を自在に操る。MIDIをマスターして、思い通りのメロディ・ベース・コードを作成します。

## この章で学ぶこと

- MIDIとは何か
- ノート（音階）の仕組み
- ベロシティ（強弱）
- MIDI CC（コントロールチェンジ）
- ピアノロール完全理解
- MIDI Clip構造
- MIDI編集テクニック


## 前提知識

このガイドを読む前に、以下の知識があると理解が深まります:

- 基本的なプログラミングの知識
- 関連する基礎概念の理解
- [Clip編集](./clip-editing.md) の内容を理解していること

---

## なぜMIDIが重要なのか

**制作の自由度:**

```
Audio:
録音した音
編集に制限

MIDI:
音の指示書
編集が自由

MIDIでできること:

音程変更:
C → D に一瞬で

タイミング修正:
ズレを簡単に直す

ベロシティ調整:
強弱を自在に

コピー:
パターンを複製

トランスポーズ:
キー変更が簡単

プロの制作:

ベースライン: MIDI
コード: MIDI
メロディ: MIDI
シンセパッド: MIDI

= 制作の60-70%がMIDI

マスター必須:
MIDI使えない = 制作できない
```

---

## MIDIとは

**Musical Instrument Digital Interface:**

### 概念

```
MIDIの本質:

音ではない:
音階情報のみ

指示書:
「C3を弾け」
「ベロシティ100で」
「0.5秒間」

音源が必要:
Wavetable
Operator
外部シンセ

例え:

楽譜 = MIDI
演奏者 = 音源
音 = Audio

MIDIファイル:
めちゃくちゃ軽い
数KB〜数十KB

Audioファイル:
重い
数MB〜数十MB
```

### MIDIメッセージの種類

```
Note On:
「音を出し始めろ」
ノート番号: C3 = 60
ベロシティ: 0-127

Note Off:
「音を止めろ」
ノート番号: C3 = 60

Control Change (CC):
「パラメーターを変更しろ」
CC番号: 1 = Modulation
値: 0-127

Program Change:
「音色を変更しろ」
プログラム番号: 0-127

Pitch Bend:
「ピッチを曲げろ」
±2半音が標準

Aftertouch:
「鍵盤を押した後の圧力」
表現力追加

Ableton で主に使うのは:
Note On/Off
Control Change
```

---

## ノート（音階）

**ピアノ鍵盤と対応:**

### ノート番号

```
MIDI Note番号:

C-2 = 0 (最低)
C-1 = 12
C0 = 24
C1 = 36
C2 = 48
C3 = 60 (中央C)
C4 = 72
C5 = 84
C6 = 96
C7 = 108
G8 = 127 (最高)

Ableton のデフォルト:
C3 = 中央C (MIDI 60)

他のDAW:
C4 = 中央C の場合も
→ 混乱しやすい

半音:

C → C# → D → D# → E → F → F# → G → G# → A → A# → B → C
0   1    2   3    4   5   6    7   8    9   10   11   12

1オクターブ = 12半音

計算:

C3 = 60
C4 = 60 + 12 = 72
C5 = 72 + 12 = 84
```

### 音域

```
楽器別の音域:

Sub Bass:
C1 - C2 (36-48)

Bass:
C2 - C4 (48-72)

Chords:
C3 - C5 (60-84)

Lead:
C4 - C6 (72-96)

Hi-Hat / Perc:
C5 - C7 (84-108)

Kick (サンプル):
通常 C3 (60) に配置
→ Drum Rackの標準

ボーカル:

男性: E2 - E4
女性: A3 - A5
```

---

## ベロシティ

**音の強弱:**

### ベロシティの仕組み

```
範囲:
0 - 127

0:
無音（Note Offと同じ）

1:
最小音量

64:
中間

127:
最大音量

用途:

音量:
ベロシティ高い = 大きい音

音色:
ベロシティ高い = 明るい音
（音源による）

表現:
強弱をつける
= 人間らしさ

機械的な打ち込み:
全て 100
= ベタ打ち

人間的な打ち込み:
90, 105, 95, 110, 88...
= バラつき
```

### ベロシティの使い分け

```
Kick:
127 (常にフル)
安定した土台

Bass:
100-120
メインなのでしっかり

Hi-Hat:
60-100
バラつき大
グルーヴ感

Chord:
80-100
中間的

Lead:
90-127
表現豊かに

推奨パターン:

オンビート: 高め (110-127)
オフビート: 低め (70-90)

アクセント:
4拍目だけ 127
他は 100
```

---

## ピアノロール

**MIDI編集の核心:**

### ピアノロールとは

```
表示:

┌────┬──■──■──────■──■──┐
│ C5 │                    │
├────┼──■──■────────────┤
│ A4 │                    │
├────┼────────■──■──■──■┤
│ F4 │                    │
├────┼──────────────────┤
│ C4 │  ■■■■■■■■■■■  │
└────┴────────────────────┘
     1.1   1.3   2.1   2.3
     (小節.拍)

縦軸:
音階（C3, D3, E3...）

横軸:
時間（小節.拍）

■:
MIDIノート

長さ:
音の長さ

色:
ベロシティ（濃い=強い）
```

### ピアノロール操作

```
ノート入力:

ダブルクリック:
ノート作成

ドラッグ:
長さ調整

Cmd+D:
複製

Delete:
削除

選択:

クリック:
1ノート選択

ドラッグ:
範囲選択

Cmd+A:
全選択

編集:

↑↓:
音程変更（半音単位）

Cmd+↑↓:
1オクターブ移動

←→:
位置移動

Cmd+E:
分割

Cmd+J:
連結
```

---

## MIDI CC（Control Change）

**パラメーター制御:**

### 主要なCC番号

```
CC 1: Modulation (モジュレーション)
用途: ビブラート、フィルター

CC 7: Volume (音量)
用途: トラック音量

CC 10: Pan (パンニング)
用途: 左右の定位

CC 11: Expression (表現)
用途: 音量の細かい制御

CC 64: Sustain Pedal (サステインペダル)
用途: 音を伸ばす
値: 0-63 = Off, 64-127 = On

CC 71: Filter Resonance (レゾナンス)
用途: フィルターの鋭さ

CC 74: Filter Cutoff (カットオフ)
用途: フィルター周波数

カスタムCC:
CC 20-31: 任意に割り当て
MIDI Learnで設定
```

### CC入力方法

```
Clip View > Envelopes:

1. Show/Hide Envelopes ボタン

2. プルダウン:
   MIDI Ctrl → CC番号選択

3. エンベロープ描画:
   ブレイクポイント配置
   曲線作成

4. 再生:
   パラメーターが変化

活用例:

Filter Sweep:
CC 74 (Cutoff)
0 → 127 に上昇
= ワブワブ

Volume Fade:
CC 7 (Volume)
127 → 0 に下降
= フェードアウト

Vibrato:
CC 1 (Modulation)
周期的に上下
= ビブラート
```

---

## MIDI Clipの構造

**Clip View詳細:**

### 基本設定

```
MIDI Clip:

┌──────────────────────────────┐
│ Clip Name: Bass_Am.mid       │
├──────────────────────────────┤
│ Piano Roll:                  │
│ [ピアノロール表示]            │
│                              │
│ Loop: [====================] │
│ Start: 1.1.1    End: 5.1.1   │
│ Length: 4 Bars               │
│                              │
│ Notes:                       │
│ Fold: ☐ (使用ノートのみ表示) │
│                              │
│ Velocity:                    │
│ Amount: 100%                 │
│ Random: 0%                   │
└──────────────────────────────┘

Fold:
使用している音階のみ表示
画面がすっきり

Velocity Amount:
全体の音量調整

Velocity Random:
ランダム化
人間らしさ
```

### Noteセクション

```
Transpose:
半音単位で移調
±48半音

Stretch:
MIDIノートの長さ変更
50% = 半分
200% = 2倍

Duplicate Loop:
ループを複製
2倍の長さに

Legato:
ノート間を埋める
隙間なし

Time Signature:
拍子変更
4/4, 3/4, 6/8等
```

---

## 実践: MIDIノート入力

**基本操作:**

### Step 1: MIDI Clip作成 (5分)

```
1. 新規MIDIトラック:
   Cmd+T

2. 音源挿入:
   Browser > Instruments > Wavetable
   トラックにドラッグ

3. 空のClip Slotをダブルクリック:
   → 空のMIDI Clip作成

4. Clip View確認:
   ピアノロール表示
```

### Step 2: ノート入力 (10分)

```
ベースライン作成:

1. Zoom:
   + / - キーで調整

2. ノート入力:

   小節 1.1.1: A2 (MIDI 45)
   ダブルクリック → ノート作成
   長さ: 1拍

   小節 1.2.1: A2
   小節 1.3.1: C3 (MIDI 48)
   小節 1.4.1: F2 (MIDI 41)

3. ベロシティ調整:
   各ノート選択
   下部のベロシティバーをドラッグ
   100前後に

4. 再生:
   Space
   → ベースライン確認

5. ループ設定:
   Loop Brace: 4 Bars
```

### Step 3: パターン複製 (5分)

```
1. 全ノート選択:
   Cmd+A

2. 複製:
   Cmd+D

3. 移動:
   選択したまま →キー
   → 1小節右に移動

4. オクターブ上げ:
   Cmd+↑
   → 12半音上がる

5. 再生:
   バリエーション確認
```

---

## MIDIプロトコルの技術的基礎

**通信規格の深層理解:**

### MIDIプロトコルの歴史と規格

```
MIDI 1.0の歴史:

1981年:
Dave Smith（Sequential Circuits）が提唱
IKEJAの加藤氏も共同提案

1983年:
MIDI 1.0規格策定
Roland / Sequential Circuits が初対応

MIDI規格のポイント:

通信速度: 31.25 kbps
非同期シリアル通信
8ビットデータ
1スタートビット + 1ストップビット
= 10ビット/バイト

物理接続（レガシー）:
5ピンDINコネクタ
MIDI IN / MIDI OUT / MIDI THRU
ケーブル最大長: 15m

現代の接続:
USB-MIDI（最も一般的）
Bluetooth MIDI
RTP-MIDI（ネットワーク経由）
TRS MIDI（3.5mm / 6.35mm）

伝送の仕組み:

送信側:
鍵盤を押す
→ Note On メッセージ生成
→ シリアルデータとして送信

受信側:
データを受け取る
→ メッセージを解析
→ 音源が発音

遅延（レイテンシー）:
MIDI 1.0: 最大 1ms/メッセージ
3バイトのNote On = 約0.96ms
人間の知覚限界: 約10ms
→ 通常は問題なし

ただし:
多数のノート同時送信時
= メッセージが順番に送られる
→ わずかなズレが発生
これを「MIDIチョーク」と呼ぶ
```

### MIDIメッセージのバイナリ構造

```
MIDIメッセージの構成:

ステータスバイト:
最上位ビット = 1（0x80以上）
メッセージの種類とチャンネルを示す

データバイト:
最上位ビット = 0（0x00-0x7F）
パラメーター値（0-127）

メッセージの種類（16進数）:

0x80-0x8F: Note Off
  例: 0x90 0x3C 0x64
  = Ch.1, Note 60(C3), Velocity 100

0x90-0x9F: Note On
  例: 0x91 0x3C 0x00
  = Ch.2, Note 60(C3), Velocity 0 (= Note Off)

0xA0-0xAF: Polyphonic Aftertouch
  個別のノートに対する圧力

0xB0-0xBF: Control Change
  例: 0xB0 0x01 0x7F
  = Ch.1, CC1 (Mod Wheel), Value 127

0xC0-0xCF: Program Change
  例: 0xC0 0x00
  = Ch.1, Program 0
  ※2バイトメッセージ

0xD0-0xDF: Channel Aftertouch
  チャンネル全体への圧力
  ※2バイトメッセージ

0xE0-0xEF: Pitch Bend
  例: 0xE0 0x00 0x40
  = Ch.1, Center (ベンドなし)
  14ビット分解能 = 16384段階

ランニングステータス:
同じメッセージの連続時
ステータスバイトを省略可能
→ データ量の節約

システムメッセージ:

0xF0: System Exclusive (SysEx)
  メーカー固有のデータ送信
  0xF0 [メーカーID] [データ...] 0xF7

0xF1: MTC Quarter Frame
  MIDI Time Code

0xF2: Song Position Pointer
  再生位置

0xF3: Song Select
  曲番号選択

0xF6: Tune Request
  チューニング要求

0xF8: Timing Clock
  24 PPQN（Pulses Per Quarter Note）
  BPM 120 = 1秒に48クロック

0xFA: Start
  再生開始

0xFB: Continue
  再生続行

0xFC: Stop
  再生停止

0xFE: Active Sensing
  接続確認（定期送信）

0xFF: System Reset
  全デバイスリセット
```

### MIDIチャンネルの詳細

```
MIDIチャンネルの概念:

16チャンネル (1-16):
1本のMIDIケーブルで
16の独立した楽器を制御可能

チャンネルの使い分け:

Ch.1: メロディ（リード）
Ch.2: ベース
Ch.3: コード（パッド）
Ch.4: ドラム
Ch.5-8: その他の楽器
Ch.9: 予備
Ch.10: ドラム（GM規格の標準）
Ch.11-16: 追加楽器

General MIDI (GM):

GM規格のドラムマップ（Ch.10）:
Note 35: Acoustic Bass Drum
Note 36: Bass Drum 1
Note 37: Side Stick
Note 38: Acoustic Snare
Note 39: Hand Clap
Note 40: Electric Snare
Note 42: Closed Hi-Hat
Note 44: Pedal Hi-Hat
Note 46: Open Hi-Hat
Note 49: Crash Cymbal 1
Note 51: Ride Cymbal 1
Note 57: Crash Cymbal 2

Abletonでのチャンネル:

デフォルト:
各MIDIトラック = Ch.1

変更方法:
MIDI To ドロップダウン
チャンネル選択

外部MIDI機器:
チャンネルで区別
マルチティンバー音源に便利

Drum Rack:
内部的にはNote番号で区別
チャンネルは無関係
パッド = ノート番号

マルチティンバー音源:
1つの音源で複数の音色
チャンネルで使い分け
例: Kontakt, Roland Cloud

設定例:
Ch.1 → ピアノ音色
Ch.2 → ストリングス音色
Ch.3 → ベース音色
Ch.10 → ドラムキット
→ 1つの音源プラグインで完結
```

### MIDI over USB の詳細

```
USB-MIDIの仕組み:

クラスコンプライアント:
USB Audio/MIDI Class
ドライバー不要で動作
macOS / Windows / Linux対応

USB-MIDIのメリット:
従来のDIN MIDIより高速
双方向通信（IN/OUTが1本で）
バスパワー給電可能
複数ポート対応

USB-MIDIのポート:
1つのUSB接続で
複数のMIDIポートを持てる

例:
AKAI MPK Mini → 1ポート（IN/OUT）
Native Instruments Komplete Kontrol → 2ポート
MOTU MIDI Express → 8ポート

Abletonでの認識:

Preferences > Link/Tempo/MIDI:

Input:
対応デバイスが自動表示
Track: On（ノート受信）
Sync: Off or On（クロック受信）
Remote: Off or On（リモート制御）

Output:
Track: On（ノート送信）
Sync: Off or On（クロック送信）
Remote: Off or On（フィードバック）

トラブルシューティング:
認識されない場合:
1. USBケーブルを交換（充電専用でないか確認）
2. 別のUSBポートを試す
3. ハブを外して直接接続
4. デバイスのファームウェア更新
5. macOS: Audio MIDI Setup で確認

USB-MIDIの注意点:
USBハブ経由:
動作するが遅延増加の可能性
セルフパワーハブ推奨

ケーブル長:
USB 2.0 = 最大5m
それ以上はアクティブリピーター

給電:
バスパワーで駆動するコントローラー多数
複数接続時は電力不足に注意
```

### Bluetooth MIDIの活用

```
Bluetooth MIDIの特徴:

規格:
BLE MIDI（Bluetooth Low Energy）
iOS / macOS / Windows 10以降対応

メリット:
ワイヤレス
ケーブル不要

デメリット:
レイテンシー: 3-10ms追加
接続の不安定さ
バッテリー必要

対応コントローラー:
ROLI Lightpad Block
KORG nanoKONTROL Studio
CME WIDI Master（後付けアダプタ）
Yamaha MD-BT01

macOSでの接続:
Audio MIDI Setup を開く
「Bluetooth Configuration」
デバイスを検出・ペアリング
→ Abletonで自動認識

活用シーン:
ライブパフォーマンス:
ステージ上の自由な移動

制作:
デスクから離れた操作
iPad連携

注意:
レイテンシーが気になる場合
→ USB接続に切り替え
リアルタイム演奏にはUSB推奨
```

### RTP-MIDI（ネットワークMIDI）

```
RTP-MIDIの概要:

正式名称:
RTP-MIDI (Real-time Transport Protocol MIDI)

特徴:
LANネットワーク経由のMIDI通信
Wi-Fi / Ethernet対応
低レイテンシー（LAN内）

macOSでの設定:
Audio MIDI Setup
→ 「ネットワーク」構成
→ セッション作成
→ 参加者を追加

メリット:
距離の制限なし（ネットワーク内）
複数デバイスの同時接続
DAW間の同期

使用例:
異なる部屋のMIDI機器を接続
複数のMac間でMIDI同期
ライブでのネットワーク接続

Abletonでの利用:
Preferencesに自動表示
通常のMIDI入力と同様に使用
```

---

## Program Changeの詳細

**音色切り替えの自動化:**

### Program Changeの仕組み

```
Program Change メッセージ:

用途:
音色（プリセット）の切り替え

範囲:
0-127（128種類）

Bank Select との組み合わせ:
128種類では足りない場合

CC 0: Bank Select MSB
CC 32: Bank Select LSB
Program Change: 音色番号

組み合わせ:
128 × 128 × 128 = 最大2,097,152音色

General MIDI音色マップ:

Piano:
0: Acoustic Grand Piano
1: Bright Acoustic Piano
2: Electric Grand Piano
3: Honky-tonk Piano
4: Electric Piano 1
5: Electric Piano 2
6: Harpsichord
7: Clavi

Chromatic Percussion:
8: Celesta
9: Glockenspiel
10: Music Box
11: Vibraphone
12: Marimba
13: Xylophone

Organ:
16: Drawbar Organ
17: Percussive Organ
18: Rock Organ
19: Church Organ

Guitar:
24: Acoustic Guitar (nylon)
25: Acoustic Guitar (steel)
26: Electric Guitar (jazz)
27: Electric Guitar (clean)
28: Electric Guitar (muted)
29: Overdriven Guitar
30: Distortion Guitar

Bass:
32: Acoustic Bass
33: Electric Bass (finger)
34: Electric Bass (pick)
35: Fretless Bass
36: Slap Bass 1
37: Slap Bass 2
38: Synth Bass 1
39: Synth Bass 2

Strings:
40: Violin
41: Viola
42: Cello
43: Contrabass
44: Tremolo Strings
48: String Ensemble 1

Synth Lead:
80: Square Lead
81: Sawtooth Lead
82: Calliope Lead

Synth Pad:
88: New Age Pad
89: Warm Pad
90: Polysynth Pad
91: Choir Pad

Abletonでの利用:
ソフトウェア音源:
Program Change受信で
プリセット自動切り替え

ハードウェア音源:
External Instrument デバイス
Program Change送信で
音色を遠隔操作

ライブでの活用:
曲ごとに音色を切り替え
MIDIクリップにProgram Changeを記録
→ 自動で音色チェンジ
```

---

## System Exclusive（SysEx）メッセージ

**メーカー固有の高度な制御:**

### SysExの概要

```
SysExメッセージの構造:

0xF0 [メーカーID] [データ...] 0xF7

メーカーID例:
0x41: Roland
0x42: KORG
0x43: Yamaha
0x7E: Non-Realtime Universal
0x7F: Realtime Universal

用途:

パッチデータの送受信:
シンセサイザーの音色データ
バックアップ・リストア

ファームウェアアップデート:
MIDI経由でデバイスを更新

ディスプレイ制御:
一部のコントローラーの
LCD/LED表示を変更

パラメーター制御:
CC では対応できない
詳細なパラメーター

AbletonとSysEx:

送信:
Max for Live デバイスで可能
SysEx送信用M4Lデバイス

受信:
基本的には無視
M4Lで解析可能

実用例:
シンセのプリセット切り替え
カスタムコントローラーの設定
ハードウェアとの深い連携
```

### MIDI Time Code（MTC）

```
MTC（MIDI Time Code）:

用途:
映像と音楽の同期
マルチトラック録音の同期

フォーマット:
SMPTE タイムコードをMIDI化
時:分:秒:フレーム

フレームレート:
24 fps: 映画
25 fps: PAL（欧州TV）
29.97 fps: NTSC ドロップフレーム
30 fps: NTSC ノンドロップ

Abletonでの設定:
Preferences > Link/Tempo/MIDI
MIDI Sync: MTC

外部同期:
映像編集ソフトと同期
Pro Toolsとの同期
テープマシンとの同期

注意:
MTC同期時は
Ableton側のテンポが
外部に追従する
→ Ableton がスレーブ

MIDI Clock vs MTC:
MIDI Clock: BPMベース（音楽用）
MTC: 絶対時間ベース（映像同期用）
```

### MIDI Clockの詳細

```
MIDI Clock（タイミングクロック）:

仕組み:
24 PPQN
= 1拍に24パルス

BPM 120の場合:
1拍 = 0.5秒
24パルス / 0.5秒 = 48パルス/秒

関連メッセージ:
0xF8: Clock（24 PPQN）
0xFA: Start
0xFB: Continue
0xFC: Stop
0xF2: Song Position Pointer

マスター/スレーブ:
マスター: クロック送信側
スレーブ: クロック受信側

Abletonの設定:

マスターとして:
MIDI Sync Output: On
→ 他のデバイスがAbletonに追従

スレーブとして:
MIDI Sync Input: On
→ Abletonが外部に追従

同期の精度:
MIDI Clock: ±1ms程度
Ableton Link: サブms
→ Linkの方が高精度

活用例:

ドラムマシンと同期:
Ableton(Master) → ドラムマシン(Slave)
→ テンポが一致

複数DAW同期:
DAW1(Master) → DAW2(Slave)
→ 同じBPMで再生

ライブ:
DJ Software → Ableton
→ テンポフォロー

注意点:
MIDI Clockにはテンポ情報なし
パルス間隔からBPMを計算
→ 不安定なクロックはテンポ揺れに
安定したマスターが重要
```

---

## Ableton LiveでのMIDI編集テクニック

**DAW上でのMIDI操作を極める:**

### Quantize（クオンタイズ）の詳細

```
クオンタイズとは:
MIDIノートのタイミングを
グリッドに合わせる機能

基本操作:
1. ノートを選択（Cmd+A で全選択）
2. Cmd+U でクオンタイズ実行
3. グリッドサイズに吸着

グリッドサイズの選択:

1/4（4分音符）:
最も荒い
大まかなタイミング修正

1/8（8分音符）:
スタンダード
多くの場面で使用

1/16（16分音符）:
細かい
Hi-Hatパターンに最適

1/32（32分音符）:
非常に細かい
装飾音符に使用

1/8T（8分音符3連）:
3連符グリッド
シャッフルビート

1/16T（16分音符3連）:
細かい3連符
トラップのハイハット

クオンタイズの強さ:

Quantize Amount:
100% = 完全にグリッドに吸着
50% = 半分だけグリッドに近づく
0% = 変化なし

推奨設定:
ドラム: 100%（タイト）
ベース: 80-100%
コード: 70-90%
メロディ: 50-80%（人間らしさ保持）

Cmd+Shift+U:
Quantize Settings を開く
Amount / Note Length を調整

クオンタイズの落とし穴:
全部100%にすると機械的
グルーヴが失われる
→ 適度にバラつかせる

手動でのタイミング微調整:
Alt+ドラッグ
→ グリッドを無視して移動
→ 意図的なズレを作れる
```

### Groove（グルーヴ）の適用

```
グルーヴプールとは:
タイミングのテンプレート
スウィング感やシャッフル感を追加

適用方法:

1. Browser > Grooves フォルダ
2. グルーヴファイルをドラッグ
3. MIDIクリップにドロップ

主要なグルーヴ:

MPC 16 Swing:
Akai MPC のスウィング
ヒップホップの定番
55-70%が人気

MPC 8 Swing:
8分音符のスウィング
ジャジーな感じ

Logic Swing:
Logic Pro 由来
クリーンなスウィング

Ableton Swing:
Ableton標準
ニュートラルな味

グルーヴパラメーター:

Timing: タイミングのズレ量
Random: ランダムなズレ
Velocity: ベロシティの変化
Quantize: ベースのクオンタイズ量
Base: 基準音符（8th/16th）

活用例:

House Music:
MPC 16 Swing 55%
軽いスウィング感

Techno:
スウィングなし（ストレート）
100% Quantize

Hip-Hop:
MPC 16 Swing 60-67%
重めのスウィング

Drum & Bass:
16th Swing 軽め
高速だが微妙なグルーヴ

Extract Groove:
Audio/MIDIクリップから
グルーヴを抽出
→ 他のクリップに適用
= 「あのグルーヴ」を再現

手順:
1. リファレンスのクリップを右クリック
2. 「Extract Groove」
3. グルーヴプールに追加
4. 対象クリップに適用
```

### Scale Mode（スケールモード）

```
Ableton Live 11+ のスケール機能:

有効化:
ピアノロール左上の「Scale」ボタン

設定:
Root Note: ルート音（C, C#, D...）
Scale Type: スケールの種類

主なスケール:

Major（メジャー）:
明るい、ポジティブ
C D E F G A B

Minor（マイナー）:
暗い、メランコリック
C D Eb F G Ab Bb

Dorian（ドリアン）:
ジャジー、ファンキー
C D Eb F G A Bb

Mixolydian（ミクソリディアン）:
ブルージー
C D E F G A Bb

Pentatonic Major（ペンタトニック・メジャー）:
シンプル、外れにくい
C D E G A

Pentatonic Minor（ペンタトニック・マイナー）:
ブルース、ロック
C Eb F G Bb

Chromatic（クロマティック）:
全半音
制限なし

Harmonic Minor（ハーモニックマイナー）:
エキゾチック、中東的
C D Eb F G Ab B

スケールモードの効果:

ハイライト表示:
スケール内の音が明るく表示
スケール外の音がグレー表示

Fold + Scale:
スケール内の音のみ表示
→ 「外れない」打ち込み

入力制限:
スケール内の音にスナップ
→ 間違ったノートを防止

活用シーン:
初心者: スケールに従って安全に打ち込み
作曲: キーを維持した実験
ライブ: アドリブ演奏の安全ネット
```

### Velocity Editor（ベロシティエディター）

```
ベロシティエディターの操作:

表示:
ピアノロール下部
縦棒グラフ

操作方法:

個別編集:
棒をドラッグ → 値変更

一括変更:
ノート選択 → 棒をドラッグ
→ 選択ノート全体が変化

描画モード:
Draw Mode (B) で直線描画
→ フェードイン/アウト

ランプ（傾斜）:
選択 → 最初の棒と最後の棒を設定
→ 自動で中間値を補間

ランダマイズ:
選択 → 右クリック
→ Randomize Velocity
→ ±20程度が自然

ベロシティカーブの設定:

MIDIキーボードの設定:
Preferences > Link/Tempo/MIDI
MIDI入力デバイス選択

カーブの種類:
リニア: 1:1（デフォルト）
ログ: 弱いタッチで音量が出やすい
エクスポネンシャル: 強く弾かないと音量が出ない

推奨:
演奏者に合わせて調整
弱く弾く人: ログカーブ
強く弾く人: エクスポネンシャル

ベロシティレイヤー活用:

音源側の設定:
ベロシティに応じて異なるサンプル再生
→ リアルな楽器サウンド

例（ピアノ音源）:
Vel 1-40: ソフトサンプル
Vel 41-80: ミディアムサンプル
Vel 81-127: ハードサンプル
→ ベロシティで音色が変わる

シンセでの活用:
ベロシティ → フィルターカットオフ
ベロシティ → エンベロープ量
ベロシティ → ディケイ長
→ 表現力が格段に上がる
```

---

## AbletonのMIDIエフェクト

**MIDIを加工するデバイス群:**

### Arpeggiator（アルペジエーター）

```
アルペジエーターとは:
押した和音を
1音ずつ順番に再生

設定パラメーター:

Style:
Up: 低い音から高い音へ
Down: 高い音から低い音へ
Up/Down: 上下往復
Down/Up: 下上往復
Converge: 外から内へ
Diverge: 内から外へ
Random: ランダム

Rate:
1/4: 4分音符ごと
1/8: 8分音符ごと（定番）
1/16: 16分音符ごと（速い）
1/32: 32分音符ごと（非常に速い）
1/8T: 8分音符3連（シャッフル感）
1/16T: 16分音符3連

Gate:
ノートの長さ（%）
100% = 音が途切れない
50% = スタッカート
200% = 次のノートと重なる

Steps:
何オクターブ分を繰り返すか
1: 入力した音域のみ
2: 1オクターブ上まで追加
3: 2オクターブ上まで追加

Offset:
アルペジオの開始位置をずらす

Velocity:
Target: 固定ベロシティ
Decay: 徐々に減衰

Retrigger:
On: 新しいノートで最初から
Off: 継続

活用例:

トランスパッド:
Style: Up
Rate: 1/16
Gate: 80%
和音を押すだけで
キラキラしたアルペジオ

Synthwave Lead:
Style: Up/Down
Rate: 1/8
Gate: 50%
Steps: 2
80年代風

Deep House Bass:
Style: Up
Rate: 1/8T
Gate: 90%
シャッフル感のあるベース

Ambient:
Style: Random
Rate: 1/4
Gate: 200%
ランダムで漂う感じ
```

### Chord（コードデバイス）

```
Chordデバイスとは:
1音の入力から
自動的に和音を生成

パラメーター:

Shift 1-6:
各追加ノートの
半音オフセット

例: メジャーコード
Shift 1: +4（長3度）
Shift 2: +7（完全5度）
他: 0

例: マイナーコード
Shift 1: +3（短3度）
Shift 2: +7（完全5度）

例: 7thコード（メジャー7）
Shift 1: +4
Shift 2: +7
Shift 3: +11

例: マイナー7thコード
Shift 1: +3
Shift 2: +7
Shift 3: +10

Velocity:
各追加ノートのベロシティ
0 = 元と同じ
負の値 = 元より弱い

活用:

1鍵で和音:
初心者でもコードが弾ける
キーボードが苦手でもOK

ライブでの活用:
1本指でコード進行
→ 残りの手でつまみ操作

パッドサウンド:
1ノート入力 → リッチな和音
→ シンセパッドが簡単に

注意:
Scale Mode と併用で
スケール外の音を防止
```

### Scale（スケールデバイス）

```
Scaleデバイスとは:
入力されたMIDIノートを
指定スケールに強制変換

パラメーター:

Base: ルート音
Scale: スケールの種類

マッピング:
各入力ノートに対して
出力ノートを指定

Fold:
On: 近い音にマッピング
Off: 特定の音にマッピング

活用:

スケール外の排除:
どの鍵盤を押しても
スケール内の音のみ出力
→ 間違いがなくなる

ライブパフォーマンス:
アドリブ演奏でも
絶対に外れない

MIDIキーボードとの組み合わせ:
白鍵のみで演奏
→ Scale デバイスが
指定キーに自動変換

プリセット例:
C Major → Am: 白鍵がAマイナーに
Pentatonic: 5音スケールに制限
Blues: ブルーススケールに
```

### Note Length（ノート長デバイス）

```
Note Lengthデバイスとは:
MIDIノートの長さを強制変更

パラメーター:

Length: ノートの長さ
Time: ms または Sync（音符）

Trigger:
Note On: ノートオンで発音
Note Off: ノートオフで発音

Gate: ゲート時間

活用:

ドラムのリトリガー:
短いGate時間
→ ドラムヒットが確実

パーカッション:
一定のノート長
→ タイトなリズム

ワンショットサンプル:
長いノートでも
短いトリガーに変換

逆再生トリガー:
Note Off モード
→ キーを離した時に発音
→ 面白いエフェクト
```

### Pitch（ピッチデバイス）

```
Pitchデバイスとは:
MIDIノートの音程を変更

パラメーター:

Pitch: 半音単位で移調（±128）
Range: ランダム範囲

活用:

トランスポーズ:
+12 = 1オクターブ上
-12 = 1オクターブ下
+7 = 完全5度上

ダブリング:
同じMIDIを2トラックに
片方に Pitch +12
→ オクターブユニゾン

レイヤー:
Pitch +7 で5度上
→ パワーコード風

ランダムピッチ:
Range: ±2
→ 微妙な音程変化
→ テクスチャー作成
```

### Random（ランダムデバイス）

```
Randomデバイスとは:
MIDIノートにランダム変化を追加

パラメーター:

Chance: ランダム適用確率
0% = 変化なし
100% = 常にランダム

Choices: ランダム範囲
Scale: スケール制限

Sign:
Add: 上方向のみ
Bi: 上下両方向

Mode:
Classic: 従来型
Alternating: 交互に

活用:

メロディの変化:
Chance: 30%
Choices: ±5
→ 時々違う音が混じる
→ ライブ感

パーカッション:
Chance: 50%
→ 不規則なパターン

アンビエント:
Chance: 80%
大きなRange
→ 不確定な音列
```

### Velocity（ベロシティデバイス）

```
Velocityデバイスとは:
MIDIベロシティを加工

パラメーター:

Drive: コンプレッション
高い値 = ダイナミクス圧縮

Compand: 圧縮/伸張
正 = ダイナミクス拡張
負 = ダイナミクス圧縮

Out Hi / Out Low:
出力範囲の制限
Out Hi: 127, Out Low: 80
→ 常に80-127の範囲内

Random:
ランダム変動量
10-20 = 自然な揺らぎ

Operation:
Clip: 範囲外をクリップ
Gate: 範囲外をミュート
Fixed: 固定値

活用:

ベロシティの均一化:
Out Hi: 100, Out Low: 100
→ 全て同じ強さ

自然な揺らぎ追加:
Random: 15
→ 機械的な打ち込みに生命感

ダイナミクス制御:
Drive で圧縮
→ 音量差を減らす

ゴーストノート作成:
Gate + 低い閾値
→ 弱いノートをカット
→ 強いアクセントのみ通過
```

---

## MIDIマッピング

**物理コントローラーとDAWの連携:**

### MIDI Learn（MIDIラーン）

```
MIDI Learnとは:
MIDIコントローラーのノブ/フェーダーを
AbletonのパラメーターにアサインBする機能

基本手順:

1. Cmd+M で MIDI Map Mode に入る
   画面が青く変わる

2. アサインしたいパラメーターをクリック
   例: フィルターのカットオフ

3. コントローラーのノブを動かす
   自動的にアサインされる

4. Cmd+M で MIDI Map Mode を抜ける

5. ノブを動かす → パラメーターが連動

アサインの確認:
Cmd+M でマッピング一覧表示
削除: 選択して Delete

注意点:
1つのCC に 1つのパラメーター
複数アサインは不可（標準では）

範囲設定:
Min / Max を設定可能
例: カットオフを 20Hz-5kHz の範囲に
→ 大きく動かしても範囲内

反転:
Min > Max に設定
→ ノブの動きが逆になる

Takeover Mode:
Pickup: 現在値に達するまで無効
Value Scaling: 段階的に追従
None: 即座にジャンプ
→ Pickup 推奨（値のジャンプ防止）
```

### MIDIマッピングの応用

```
マクロコントロール:

Rack内のマクロ:
1つのノブで複数パラメーター制御

設定:
1. Audio Effect Rack を作成
2. マクロを有効化
3. Rack内のパラメーターをマクロに割り当て
4. マクロにMIDI CC をアサイン

例:
マクロ1 = 「Intensity」
→ フィルターカットオフ: 0-100%
→ リバーブ Dry/Wet: 0-50%
→ ディストーション Drive: 0-60%
→ 1つのノブで3つが連動

DJ風コントロール:
マクロ1 = フィルター（Lo→Hi）
マクロ2 = リバーブ量
マクロ3 = ディレイ量
マクロ4 = ドライブ量
→ 4つのノブで即興パフォーマンス

ボタンマッピング:

トグル:
ボタン → On/Off
例: エフェクト有効/無効

モメンタリー:
押している間だけ On
例: リバーブスプラッシュ

シーン切り替え:
Session View のシーン
→ パッドにアサイン
→ ワンタップで切り替え

クリップ起動:
Clip Slot → パッドアサイン
→ ライブでのクリップ操作
```

### 外部MIDI機器の接続と設定

```
外部MIDI音源の接続:

USB接続:
1. USB ケーブルで接続
2. Ableton が自動認識
3. Preferences で Track: On

DIN MIDI接続:
1. MIDIインターフェース使用
   例: iConnectivity, MOTU
2. MIDI OUT → 音源の MIDI IN
3. Audio OUT → オーディオインターフェースの IN

External Instrument デバイス:

設定:
1. MIDIトラックに追加
2. MIDI To: 外部デバイス選択
3. Audio From: オーディオ入力選択
4. Hardware Latency: レイテンシー補正

メリット:
MIDIとAudioのルーティングが1つで完了
レイテンシー自動補正

ハードウェアシンセとの連携:

Ableton → MIDI → シンセ → Audio → Ableton

具体例:
1. MIDIクリップでフレーズ作成
2. External Instrument で MIDI送信
3. シンセのAudio出力を録音
4. Audioクリップとして保存

CV/Gate変換:
ユーロラック等のアナログシンセ
→ CV/Gateインターフェース使用
Expert Sleepers ES-8
MOTU Volta
Ableton CV Tools Pack

MIDI Thru チェイン:
複数のMIDI機器を数珠つなぎ
デバイス1 THRU → デバイス2 IN
デバイス2 THRU → デバイス3 IN
→ チャンネルで区別

注意:
THRU が長いとタイミングがズレる
→ MIDI パッチベイ推奨
→ またはUSBで個別接続
```

### MIDI Routingの高度なテクニック

```
トラック間MIDIルーティング:

送信:
MIDIトラック A の MIDI To
→ 別のトラック B を選択

受信:
トラック B は
トラック A の MIDI を受信

用途:

レイヤー:
1つのMIDI → 複数の音源
→ 分厚いサウンド

スプリット:
高音域 → 音源A
低音域 → 音源B

チャンネルフィルター:
特定チャンネルのみ送信

IAC Driver（macOS）:
アプリ間のMIDIルーティング

設定:
Audio MIDI Setup
→ IAC Driver を有効化

用途:
Ableton → 別のDAW
Ableton → VJ ソフト
Max/MSP → Ableton

仮想MIDIポート:
複数作成可能
各ポートに名前を付ける
→ 整理しやすい

MIDI Monitor:
MIDIメッセージの確認ツール
macOS: MIDI Monitor（無料アプリ）
→ 何が送受信されているか確認
→ トラブルシューティングに必須
```

---

## よくある質問

### Q1: ピアノ弾けないけど大丈夫？

**A:** 全く問題なし

```
マウスで入力:
ピアノロールにダブルクリック
→ 誰でもできる

グリッド:
自動で整列
リズム感不要

Quantize:
タイミング自動修正

コード理論:
基本だけでOK
Cメジャー、Aマイナー

学習:

Week 1: ピアノロール操作
Week 2: 基本コード
Week 3: スケール理解
Week 4: メロディ作成

結論:
ピアノ不要
マウスで十分
```

### Q2: ベロシティは全部同じでいい？

**A:** バラつかせるべき

```
全部同じ (ベタ打ち):
機械的
表現力ゼロ

バラつき:
人間らしい
グルーヴ感

推奨:

Kick: 127固定 (OK)
Bass: 100-120
Hi-Hat: 60-100 (バラつき大)
Melody: 80-127

ランダム化:

選択 → 右クリック
→ Randomize Velocity
→ 20-30%
```

### Q3: MIDIキーボード必要？

**A:** あった方が速い

```
マウス入力:
可能
遅い

MIDIキーボード:
速い
直感的

推奨機種:

AKAI MPK Mini MK3: ¥13,000
25鍵
パッド付き

M-Audio Keystation Mini: ¥8,000
32鍵
シンプル

Novation Launchkey Mini: ¥13,000
25鍵
Ableton統合

結論:
¥10,000前後の投資
制作スピード3倍
```

---

## MPE（MIDI Polyphonic Expression）

**次世代の表現力:**

### MPEの概要

```
MPEとは:
MIDI Polyphonic Expression
各ノートに独立した表現パラメーターを付与

従来のMIDI:
Pitch Bend → チャンネル全体
CC → チャンネル全体
= 和音の一部だけベンドできない

MPE:
各ノートが独立したチャンネル
→ 1音ずつベンド可能
→ 1音ずつ圧力変更可能

MPEの仕組み:
Master Channel: Ch.1（グローバル制御）
Member Channels: Ch.2-16（各ノートに割り当て）
→ 最大15ノートの独立制御

対応コントローラー:
ROLI Seaboard: 連続的なサーフェス
Sensel Morph: 圧力感知パッド
Linnstrument: グリッドレイアウト
Artiphon Instrument 1: 多目的
KMI K-Board Pro 4: コンパクト

Abletonでの設定:
Preferences > Link/Tempo/MIDI
MPE対応デバイス: 自動認識
MPEモード: On

対応音源:
Wavetable: MPE対応
Drift: MPE対応
Simpler: MPE対応（Live 11+）
サードパーティ: Serum, Pigments等

MPEの表現パラメーター:
Slide: Y軸（指の上下スライド）
Press: Z軸（圧力、アフタータッチ）
Glide: X軸（ピッチベンド）
→ 3次元の表現が可能

活用:
パッドサウンド:
各指の圧力で個別にフィルター制御
→ 有機的なテクスチャー

リード:
ギターのベンドのように
1音だけピッチを曲げる

アンビエント:
スライドで各音の音色を変化
→ 進化するサウンドスケープ
```

---

## MIDI 2.0

**MIDIの未来:**

### MIDI 2.0の新機能

```
MIDI 2.0（2020年策定）:

主な改善:

分解能の向上:
ベロシティ: 7ビット(128段階) → 16ビット(65536段階)
CC値: 7ビット → 32ビット(約43億段階)
Pitch Bend: 14ビット → 32ビット
→ 超滑らかなパラメーター変化

双方向通信:
MIDI-CI（Capability Inquiry）
デバイス同士が自動交渉
→ 設定の自動化

Property Exchange:
デバイス情報の取得
プリセット名の表示
パッチの送受信

Profile Configuration:
デバイスの役割を自動設定
ドラムモード自動認識
コントローラー自動アサイン

後方互換性:
MIDI 1.0デバイスとも通信可能
トランスレーター不要

Universal MIDI Packet (UMP):
新しいメッセージフォーマット
32ビット/64ビット/128ビットパケット
タイムスタンプ内蔵

Ableton対応状況:
段階的に対応中
将来的に完全対応予定

注意:
MIDI 2.0対応ハードウェアは
まだ少数
→ 普及には時間が必要
→ 現時点ではMIDI 1.0で十分

MIDI 2.0の恩恵を受ける場面:
ソフトウェア音源の高分解能制御
MPEとの組み合わせ
自動設定によるセットアップ簡略化
```

---

## MIDIトラブルシューティング

**よくある問題と解決策:**

### 音が出ない場合

```
チェックリスト:

1. MIDIトラックの確認:
   □ 音源（Instrument）がロードされているか
   □ トラックがアーム（録音待機）状態か
   □ モニター設定: In or Auto

2. MIDI入力の確認:
   □ Preferences > MIDI
   □ Input デバイスが有効か
   □ Track: On になっているか

3. チャンネルの確認:
   □ MIDI From: All Channels
   □ または正しいチャンネル

4. ルーティングの確認:
   □ MIDI To: 正しいデバイス
   □ Audio To: Master (or 正しい出力)

5. コントローラーの確認:
   □ USB接続を確認
   □ 電源が入っているか
   □ 正しいドライバーか

6. MIDI Indicator:
   トラック上部の小さなドット
   緑に光る = MIDI受信中
   光らない = MIDI届いていない

7. Audio設定:
   □ Audio出力が正しいか
   □ マスターボリュームが0でないか
   □ トラックボリュームが0でないか
```

### レイテンシーの問題

```
MIDIレイテンシー対策:

原因と対策:

バッファサイズ:
大きい = レイテンシー大
小さい = CPU負荷大
推奨: 128-256 samples

USB接続:
直接接続推奨
ハブ経由は遅延の原因

Bluetooth MIDI:
3-10ms追加
リアルタイム演奏には不向き

プラグイン遅延:
重いプラグインは遅延を生む
Reduced Latency When Monitoring: On
→ Preferences > Audio

Driver Error Compensation:
Preferences > Audio
→ 値を調整して補正

Track Delay:
各トラックのタイミング微調整
ms単位で前後に移動

External Instrument:
Hardware Latency 設定
→ 外部機器の遅延を補正

測定方法:
1. MIDIキーボードを押す
2. Audio出力を録音
3. 波形でタイミング確認
4. 遅延量を測定
```

### MIDIフィードバックループ

```
問題:
MIDIの入出力が循環
→ 無限ループ
→ ノートが止まらない
→ CPU暴走

原因:
MIDI IN → Ableton → MIDI OUT → 同じMIDI IN
→ 無限ループ

対策:
1. MIDI Thru を無効化:
   Options > Preferences
   MIDI Thru: 必要な場合のみOn

2. ルーティングの見直し:
   フィードバックパスがないか確認
   MIDI Monitorで送受信をチェック

3. パニック:
   Cmd+. (Cmd+ピリオド)
   → 全MIDIノートオフ
   → スタックノート解消

4. IAC Driver:
   不要な仮想ポートを無効化
   → 意図しないループ防止

スタックノート対策:
音が止まらない場合:
Cmd+. で All Notes Off
それでも止まらない場合:
トラックの MIDI Out を一時的にオフ
```

---

## 実践ワークフロー: MIDIを使った楽曲制作

**ステップバイステップガイド:**

### コード進行の打ち込み

```
Am → F → C → G（ポップ定番進行）:

Am (Aマイナー):
A2 (MIDI 45) + C3 (48) + E3 (52)
長さ: 1小節

F (Fメジャー):
F2 (MIDI 41) + A2 (45) + C3 (48)
長さ: 1小節

C (Cメジャー):
C3 (MIDI 48) + E3 (52) + G3 (55)
長さ: 1小節

G (Gメジャー):
G2 (MIDI 43) + B2 (47) + D3 (50)
長さ: 1小節

手順:
1. 新規MIDIトラック作成
2. Wavetable (Pad プリセット) をロード
3. 4小節のMIDIクリップ作成
4. 各小節にコードを入力
5. ベロシティ: 85-100
6. ループ再生で確認

バリエーション:
ボイシング変更:
転回形を使う
C3+E3+G3 → E3+G3+C4

オープンボイシング:
C2+G2+E3
→ 広がりのあるサウンド

アルペジオ化:
Arpeggiator追加
Rate: 1/8
→ コードが分散和音に
```

### ベースラインの作成

```
コード進行に合わせたベース:

基本ルール:
ルート音を基本に
オクターブ下で演奏

Am: A1 (MIDI 33)
F: F1 (MIDI 29)
C: C2 (MIDI 36)
G: G1 (MIDI 31)

パターン例:

シンプル（4つ打ち）:
各拍にルート音
安定感重視

オクターブ:
1拍目: ルート
3拍目: 1オクターブ上
→ ハウスの定番

ウォーキング:
ルート→3度→5度→オクターブ
→ ジャジーな動き

シンコペーション:
16分音符のズレ
→ ファンキーなグルーヴ

Acid:
16分音符連打
アクセント変化
スライド（Glide）多用
→ TB-303スタイル
```

### Max for Live MIDIデバイス

```
Max for Live (M4L) MIDI活用:

M4Lとは:
Max/MSPのAbleton統合環境
カスタムMIDIデバイスを作成可能

人気のM4L MIDIデバイス:

Probability Pack:
確率ベースのノート生成
→ ジェネラティブミュージック

Euclidean Sequencer:
ユークリッドリズムパターン
→ ポリリズム生成

CC Map:
複雑なCCマッピング
1つのCCで複数パラメーター制御

MIDI Feedback:
MIDIデータのリアルタイム変換
→ フィードバックループ的な進化

LFO (M4L版):
MIDIパラメーターのLFO変調
テンポ同期で周期的に変化

インストール:
ableton.com > Packs > Max for Live Essentials
→ 無料で基本セット入手

カスタム作成:
Max for Live Editor で
パッチを自作可能
→ 無限の可能性
```


---

## 実践演習

### 演習1: 基本的な実装

以下の要件を満たすコードを実装してください。

**要件:**
- 入力データの検証を行うこと
- エラーハンドリングを適切に実装すること
- テストコードも作成すること

```python
# 演習1: 基本実装のテンプレート
class Exercise1:
    """基本的な実装パターンの演習"""

    def __init__(self):
        self.data = []

    def validate_input(self, value):
        """入力値の検証"""
        if value is None:
            raise ValueError("入力値がNoneです")
        return True

    def process(self, value):
        """データ処理のメインロジック"""
        self.validate_input(value)
        self.data.append(value)
        return self.data

    def get_results(self):
        """処理結果の取得"""
        return {
            'count': len(self.data),
            'data': self.data
        }

# テスト
def test_exercise1():
    ex = Exercise1()
    assert ex.process(1) == [1]
    assert ex.process(2) == [1, 2]
    assert ex.get_results()['count'] == 2

    try:
        ex.process(None)
        assert False, "例外が発生するべき"
    except ValueError:
        pass

    print("全テスト合格!")

test_exercise1()
```

### 演習2: 応用パターン

基本実装を拡張して、以下の機能を追加してください。

```python
# 演習2: 応用パターン
from typing import List, Dict, Optional
from datetime import datetime

class AdvancedExercise:
    """応用パターンの演習"""

    def __init__(self, max_size: int = 100):
        self._items: List[Dict] = []
        self._max_size = max_size
        self._created_at = datetime.now()

    def add(self, key: str, value: any) -> bool:
        """アイテムの追加（サイズ制限付き）"""
        if len(self._items) >= self._max_size:
            return False
        self._items.append({
            'key': key,
            'value': value,
            'timestamp': datetime.now().isoformat()
        })
        return True

    def find(self, key: str) -> Optional[Dict]:
        """キーによる検索"""
        for item in reversed(self._items):
            if item['key'] == key:
                return item
        return None

    def remove(self, key: str) -> bool:
        """キーによる削除"""
        for i, item in enumerate(self._items):
            if item['key'] == key:
                self._items.pop(i)
                return True
        return False

    def stats(self) -> Dict:
        """統計情報"""
        return {
            'total_items': len(self._items),
            'max_size': self._max_size,
            'usage_percent': len(self._items) / self._max_size * 100,
            'uptime': str(datetime.now() - self._created_at)
        }

# テスト
def test_advanced():
    ex = AdvancedExercise(max_size=3)
    assert ex.add("a", 1) == True
    assert ex.add("b", 2) == True
    assert ex.add("c", 3) == True
    assert ex.add("d", 4) == False  # サイズ制限
    assert ex.find("b")['value'] == 2
    assert ex.remove("b") == True
    assert ex.find("b") is None
    stats = ex.stats()
    assert stats['total_items'] == 2
    print("応用テスト全合格!")

test_advanced()
```

### 演習3: パフォーマンス最適化

以下のコードのパフォーマンスを改善してください。

```python
# 演習3: パフォーマンス最適化
import time
from functools import lru_cache

# 最適化前（O(n^2)）
def slow_search(data: list, target: int) -> int:
    """非効率な検索"""
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            if data[i] + data[j] == target:
                return (i, j)
    return (-1, -1)

# 最適化後（O(n)）
def fast_search(data: list, target: int) -> tuple:
    """ハッシュマップを使った効率的な検索"""
    seen = {}
    for i, num in enumerate(data):
        complement = target - num
        if complement in seen:
            return (seen[complement], i)
        seen[num] = i
    return (-1, -1)

# ベンチマーク
def benchmark():
    import random
    data = list(range(5000))
    random.shuffle(data)
    target = data[100] + data[4000]

    start = time.time()
    result1 = slow_search(data, target)
    slow_time = time.time() - start

    start = time.time()
    result2 = fast_search(data, target)
    fast_time = time.time() - start

    print(f"非効率版: {slow_time:.4f}秒")
    print(f"効率版:   {fast_time:.6f}秒")
    print(f"高速化率: {slow_time/fast_time:.0f}倍")

benchmark()
```

**ポイント:**
- アルゴリズムの計算量を意識する
- 適切なデータ構造を選択する
- ベンチマークで効果を測定する

---

## トラブルシューティング

### よくあるエラーと解決策

| エラー | 原因 | 解決策 |
|--------|------|--------|
| 初期化エラー | 設定ファイルの不備 | 設定ファイルのパスと形式を確認 |
| タイムアウト | ネットワーク遅延/リソース不足 | タイムアウト値の調整、リトライ処理の追加 |
| メモリ不足 | データ量の増大 | バッチ処理の導入、ページネーションの実装 |
| 権限エラー | アクセス権限の不足 | 実行ユーザーの権限確認、設定の見直し |
| データ不整合 | 並行処理の競合 | ロック機構の導入、トランザクション管理 |

### デバッグの手順

1. **エラーメッセージの確認**: スタックトレースを読み、発生箇所を特定する
2. **再現手順の確立**: 最小限のコードでエラーを再現する
3. **仮説の立案**: 考えられる原因をリストアップする
4. **段階的な検証**: ログ出力やデバッガを使って仮説を検証する
5. **修正と回帰テスト**: 修正後、関連する箇所のテストも実行する

```python
# デバッグ用ユーティリティ
import logging
import traceback
from functools import wraps

# ロガーの設定
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

def debug_decorator(func):
    """関数の入出力をログ出力するデコレータ"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"呼び出し: {func.__name__}(args={args}, kwargs={kwargs})")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"戻り値: {func.__name__} -> {result}")
            return result
        except Exception as e:
            logger.error(f"例外発生: {func.__name__}: {e}")
            logger.error(traceback.format_exc())
            raise
    return wrapper

@debug_decorator
def process_data(items):
    """データ処理（デバッグ対象）"""
    if not items:
        raise ValueError("空のデータ")
    return [item * 2 for item in items]
```

### パフォーマンス問題の診断

パフォーマンス問題が発生した場合の診断手順:

1. **ボトルネックの特定**: プロファイリングツールで計測
2. **メモリ使用量の確認**: メモリリークの有無をチェック
3. **I/O待ちの確認**: ディスクやネットワークI/Oの状況を確認
4. **同時接続数の確認**: コネクションプールの状態を確認

| 問題の種類 | 診断ツール | 対策 |
|-----------|-----------|------|
| CPU負荷 | cProfile, py-spy | アルゴリズム改善、並列化 |
| メモリリーク | tracemalloc, objgraph | 参照の適切な解放 |
| I/Oボトルネック | strace, iostat | 非同期I/O、キャッシュ |
| DB遅延 | EXPLAIN, slow query log | インデックス、クエリ最適化 |

---

## 設計判断ガイド

### 選択基準マトリクス

技術選択を行う際の判断基準を以下にまとめます。

| 判断基準 | 重視する場合 | 妥協できる場合 |
|---------|------------|-------------|
| パフォーマンス | リアルタイム処理、大規模データ | 管理画面、バッチ処理 |
| 保守性 | 長期運用、チーム開発 | プロトタイプ、短期プロジェクト |
| スケーラビリティ | 成長が見込まれるサービス | 社内ツール、固定ユーザー |
| セキュリティ | 個人情報、金融データ | 公開データ、社内利用 |
| 開発速度 | MVP、市場投入スピード | 品質重視、ミッションクリティカル |

### アーキテクチャパターンの選択

```
┌─────────────────────────────────────────────────┐
│              アーキテクチャ選択フロー              │
├─────────────────────────────────────────────────┤
│                                                 │
│  ① チーム規模は？                                │
│    ├─ 小規模（1-5人）→ モノリス                   │
│    └─ 大規模（10人+）→ ②へ                       │
│                                                 │
│  ② デプロイ頻度は？                               │
│    ├─ 週1回以下 → モノリス + モジュール分割         │
│    └─ 毎日/複数回 → ③へ                          │
│                                                 │
│  ③ チーム間の独立性は？                            │
│    ├─ 高い → マイクロサービス                      │
│    └─ 中程度 → モジュラーモノリス                   │
│                                                 │
└─────────────────────────────────────────────────┘
```

### トレードオフの分析

技術的な判断には必ずトレードオフが伴います。以下の観点で分析を行いましょう:

**1. 短期 vs 長期のコスト**
- 短期的に速い方法が長期的には技術的負債になることがある
- 逆に、過剰な設計は短期的なコストが高く、プロジェクトの遅延を招く

**2. 一貫性 vs 柔軟性**
- 統一された技術スタックは学習コストが低い
- 多様な技術の採用は適材適所が可能だが、運用コストが増加

**3. 抽象化のレベル**
- 高い抽象化は再利用性が高いが、デバッグが困難になる場合がある
- 低い抽象化は直感的だが、コードの重複が発生しやすい

```python
# 設計判断の記録テンプレート
class ArchitectureDecisionRecord:
    """ADR (Architecture Decision Record) の作成"""

    def __init__(self, title: str):
        self.title = title
        self.context = ""
        self.decision = ""
        self.consequences = []
        self.alternatives = []

    def set_context(self, context: str):
        """背景と課題の記述"""
        self.context = context
        return self

    def set_decision(self, decision: str):
        """決定内容の記述"""
        self.decision = decision
        return self

    def add_consequence(self, consequence: str, positive: bool = True):
        """結果の追加"""
        self.consequences.append({
            'description': consequence,
            'type': 'positive' if positive else 'negative'
        })
        return self

    def add_alternative(self, name: str, reason_rejected: str):
        """却下した代替案の追加"""
        self.alternatives.append({
            'name': name,
            'reason_rejected': reason_rejected
        })
        return self

    def to_markdown(self) -> str:
        """Markdown形式で出力"""
        md = f"# ADR: {self.title}\n\n"
        md += f"## 背景\n{self.context}\n\n"
        md += f"## 決定\n{self.decision}\n\n"
        md += "## 結果\n"
        for c in self.consequences:
            icon = "✅" if c['type'] == 'positive' else "⚠️"
            md += f"- {icon} {c['description']}\n"
        md += "\n## 却下した代替案\n"
        for a in self.alternatives:
            md += f"- **{a['name']}**: {a['reason_rejected']}\n"
        return md
```

---

## 実務での適用シナリオ

### シナリオ1: スタートアップでのMVP開発

**状況:** 限られたリソースで素早くプロダクトをリリースする必要がある

**アプローチ:**
- シンプルなアーキテクチャを選択
- 必要最小限の機能に集中
- 自動テストはクリティカルパスのみ
- モニタリングは早期から導入

**学んだ教訓:**
- 完璧を求めすぎない（YAGNI原則）
- ユーザーフィードバックを早期に取得
- 技術的負債は意識的に管理する

### シナリオ2: レガシーシステムのモダナイゼーション

**状況:** 10年以上運用されているシステムを段階的に刷新する

**アプローチ:**
- Strangler Fig パターンで段階的に移行
- 既存のテストがない場合はCharacterization Testを先に作成
- APIゲートウェイで新旧システムを共存
- データ移行は段階的に実施

| フェーズ | 作業内容 | 期間目安 | リスク |
|---------|---------|---------|--------|
| 1. 調査 | 現状分析、依存関係の把握 | 2-4週間 | 低 |
| 2. 基盤 | CI/CD構築、テスト環境 | 4-6週間 | 低 |
| 3. 移行開始 | 周辺機能から順次移行 | 3-6ヶ月 | 中 |
| 4. コア移行 | 中核機能の移行 | 6-12ヶ月 | 高 |
| 5. 完了 | 旧システム廃止 | 2-4週間 | 中 |

### シナリオ3: 大規模チームでの開発

**状況:** 50人以上のエンジニアが同一プロダクトを開発する

**アプローチ:**
- ドメイン駆動設計で境界を明確化
- チームごとにオーナーシップを設定
- 共通ライブラリはInner Source方式で管理
- APIファーストで設計し、チーム間の依存を最小化

```python
# チーム間のAPI契約定義
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

class Priority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class APIContract:
    """チーム間のAPI契約"""
    endpoint: str
    method: str
    owner_team: str
    consumers: List[str]
    sla_ms: int  # レスポンスタイムSLA
    priority: Priority

    def validate_sla(self, actual_ms: int) -> bool:
        """SLA準拠の確認"""
        return actual_ms <= self.sla_ms

    def to_openapi(self) -> dict:
        """OpenAPI形式で出力"""
        return {
            'path': self.endpoint,
            'method': self.method,
            'x-owner': self.owner_team,
            'x-consumers': self.consumers,
            'x-sla-ms': self.sla_ms
        }

# 使用例
contracts = [
    APIContract(
        endpoint="/api/v1/users",
        method="GET",
        owner_team="user-team",
        consumers=["order-team", "notification-team"],
        sla_ms=200,
        priority=Priority.HIGH
    ),
    APIContract(
        endpoint="/api/v1/orders",
        method="POST",
        owner_team="order-team",
        consumers=["payment-team", "inventory-team"],
        sla_ms=500,
        priority=Priority.CRITICAL
    )
]
```

### シナリオ4: パフォーマンスクリティカルなシステム

**状況:** ミリ秒単位のレスポンスが求められるシステム

**最適化ポイント:**
1. キャッシュ戦略（L1: インメモリ、L2: Redis、L3: CDN）
2. 非同期処理の活用
3. コネクションプーリング
4. クエリ最適化とインデックス設計

| 最適化手法 | 効果 | 実装コスト | 適用場面 |
|-----------|------|-----------|---------|
| インメモリキャッシュ | 高 | 低 | 頻繁にアクセスされるデータ |
| CDN | 高 | 低 | 静的コンテンツ |
| 非同期処理 | 中 | 中 | I/O待ちが多い処理 |
| DB最適化 | 高 | 高 | クエリが遅い場合 |
| コード最適化 | 低-中 | 高 | CPU律速の場合 |

---

## チーム開発での活用

### コードレビューのチェックリスト

このトピックに関連するコードレビューで確認すべきポイント:

- [ ] 命名規則が一貫しているか
- [ ] エラーハンドリングが適切か
- [ ] テストカバレッジは十分か
- [ ] パフォーマンスへの影響はないか
- [ ] セキュリティ上の問題はないか
- [ ] ドキュメントは更新されているか

### ナレッジ共有のベストプラクティス

| 方法 | 頻度 | 対象 | 効果 |
|------|------|------|------|
| ペアプログラミング | 随時 | 複雑なタスク | 即時のフィードバック |
| テックトーク | 週1回 | チーム全体 | 知識の水平展開 |
| ADR (設計記録) | 都度 | 将来のメンバー | 意思決定の透明性 |
| 振り返り | 2週間ごと | チーム全体 | 継続的改善 |
| モブプログラミング | 月1回 | 重要な設計 | 合意形成 |

### 技術的負債の管理

```
優先度マトリクス:

        影響度 高
          │
    ┌─────┼─────┐
    │ 計画 │ 即座 │
    │ 的に │ に   │
    │ 対応 │ 対応 │
    ├─────┼─────┤
    │ 記録 │ 次の │
    │ のみ │ Sprint│
    │     │ で   │
    └─────┼─────┘
          │
        影響度 低
    発生頻度 低  発生頻度 高
```

---

## セキュリティの考慮事項

### 一般的な脆弱性と対策

| 脆弱性 | リスクレベル | 対策 | 検出方法 |
|--------|------------|------|---------|
| インジェクション攻撃 | 高 | 入力値のバリデーション・パラメータ化クエリ | SAST/DAST |
| 認証の不備 | 高 | 多要素認証・セッション管理の強化 | ペネトレーションテスト |
| 機密データの露出 | 高 | 暗号化・アクセス制御 | セキュリティ監査 |
| 設定の不備 | 中 | セキュリティヘッダー・最小権限の原則 | 構成スキャン |
| ログの不足 | 中 | 構造化ログ・監査証跡 | ログ分析 |

### セキュアコーディングのベストプラクティス

```python
# セキュアコーディング例
import hashlib
import secrets
import hmac
from typing import Optional

class SecurityUtils:
    """セキュリティユーティリティ"""

    @staticmethod
    def generate_token(length: int = 32) -> str:
        """暗号学的に安全なトークン生成"""
        return secrets.token_urlsafe(length)

    @staticmethod
    def hash_password(password: str, salt: Optional[str] = None) -> tuple:
        """パスワードのハッシュ化"""
        if salt is None:
            salt = secrets.token_hex(16)
        hashed = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            iterations=100000
        )
        return hashed.hex(), salt

    @staticmethod
    def verify_password(password: str, hashed: str, salt: str) -> bool:
        """パスワードの検証"""
        new_hash, _ = SecurityUtils.hash_password(password, salt)
        return hmac.compare_digest(new_hash, hashed)

    @staticmethod
    def sanitize_input(value: str) -> str:
        """入力値のサニタイズ"""
        dangerous_chars = ['<', '>', '"', "'", '&', '\\']
        result = value
        for char in dangerous_chars:
            result = result.replace(char, '')
        return result.strip()

# 使用例
token = SecurityUtils.generate_token()
hashed, salt = SecurityUtils.hash_password("my_password")
is_valid = SecurityUtils.verify_password("my_password", hashed, salt)
```

### セキュリティチェックリスト

- [ ] 全ての入力値がバリデーションされている
- [ ] 機密情報がログに出力されていない
- [ ] HTTPS が強制されている
- [ ] CORS ポリシーが適切に設定されている
- [ ] 依存パッケージの脆弱性スキャンが実施されている
- [ ] エラーメッセージに内部情報が含まれていない
---

## まとめ

### MIDI基礎

```
□ MIDIは音階情報
□ ノート番号 0-127
□ ベロシティ 0-127
□ ピアノロールで編集
□ CC でパラメーター制御
```

### 重要ポイント

```
□ C3 = 60 (中央C)
□ ベロシティでバラつき
□ Fold で画面整理
□ Quantize でタイミング修正
□ Cmd+D で複製
```

### 音域目安

```
Sub Bass: C1-C2
Bass: C2-C4
Chords: C3-C5
Lead: C4-C6
```

### 学習ロードマップ

```
Week 1-2: MIDI基礎
□ ピアノロール操作に慣れる
□ ノート入力、ベロシティ調整
□ クオンタイズの使い方

Week 3-4: コード & メロディ
□ 基本コード進行の打ち込み
□ スケールモードの活用
□ ベースラインの作成

Week 5-6: MIDI エフェクト
□ アルペジエーターの活用
□ Chord / Scale デバイス
□ MIDIマッピング設定

Week 7-8: 応用テクニック
□ グルーヴの適用
□ 外部MIDI機器の接続
□ MPE / 高度な表現

継続:
□ Max for Live MIDI デバイス
□ MIDI 2.0 の動向フォロー
□ 独自のワークフロー確立
```

---

**次は:** [Clip編集](./clip-editing.md) - カット、コピー、ペースト完全マスター

---

## 次に読むべきガイド

- [Quantize（クオンタイズ）](./quantization.md) - 次のトピックへ進む

---

## 参考文献

- [MDN Web Docs](https://developer.mozilla.org/) - Web技術のリファレンス
- [Wikipedia](https://ja.wikipedia.org/) - 技術概念の概要
