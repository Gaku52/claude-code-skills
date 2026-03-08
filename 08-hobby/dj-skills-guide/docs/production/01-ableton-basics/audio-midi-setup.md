# Audio/MIDI設定

オーディオとMIDI設定を完璧にする。レイテンシーなしの快適な制作環境を実現します。

## この章で学ぶこと

- オーディオインターフェイス設定
- サンプルレート、バッファサイズ最適化
- レイテンシー完全理解と対策
- MIDI機器接続（DDJ-FLX4、キーボード）
- 入出力ルーティング
- モニタリング設定

---

## なぜAudio/MIDI設定が重要なのか

**音の入出力の要:**

```
正しい設定:

低レイテンシー:
弾いてすぐ音が出る

安定:
プツプツノイズなし

高音質:
44.1kHz/24bit

間違った設定:

遅延:
鍵盤押して0.5秒後に音
→ 演奏不可能

ノイズ:
プツプツ、ブツブツ

クラッシュ:
頻繁にフリーズ

プロとアマの差:

プロ:
5分で正しく設定
快適に制作

アマ:
設定を知らない
イライラしながら制作
```

---

## オーディオインターフェイス

**内蔵 vs 外付け:**

### 内蔵オーディオ

```
Mac: Core Audio
Win: ASIO4ALL または Realtek

メリット:
無料
すぐ使える

デメリット:
レイテンシー高い (10-20ms)
音質普通
入力チャンネル少ない

使える:
最初の1ヶ月
簡単なMIDI制作

限界:
ボーカル録音
ギター録音
低レイテンシー必要な作業
```

### 外付けオーディオインターフェイス

```
推奨機種:

入門:
Focusrite Scarlett Solo: ¥14,000
Behringer U-PHORIA UM2: ¥7,000

中級:
Focusrite Scarlett 2i2: ¥20,000
Universal Audio Volt 2: ¥23,000

上級:
Universal Audio Apollo Twin: ¥100,000+
RME Babyface Pro: ¥110,000

メリット:

レイテンシー:
2-5ms (体感ゼロ)

音質:
プロレベル

入力:
マイク、ギター接続可能

安定:
専用ドライバ

推奨:

最初:
内蔵で試す

本気なら:
Scarlett Solo購入
(¥14,000の投資)
```

---

## Audio設定

**Preferences > Audio:**

### Audio Device (オーディオデバイス)

```
Mac:

Driver Type: Core Audio (固定)

Audio Input Device:
└─ Built-in Microphone
└─ Scarlett Solo (接続時)

Audio Output Device:
└─ Built-in Output
└─ Scarlett Solo (接続時)

選択:
Scarlett Solo接続時
→ Input/Output 両方を Scarlett Solo

Windows:

Driver Type:
└─ MME (非推奨)
└─ DirectX (非推奨)
└─ ASIO (推奨)

選択: ASIO

Audio Device:
└─ ASIO4ALL (内蔵の場合)
└─ Scarlett Solo ASIO (接続時)

重要:
必ず ASIO ドライバ使用
→ レイテンシー最小化
```

### Sample Rate (サンプルレート)

```
選択肢:

44100 Hz (CD品質) ← 推奨
48000 Hz (ビデオ)
88200 Hz (ハイレゾ)
96000 Hz (プロ)
192000 Hz (オーバースペック)

推奨: 44100 Hz

理由:

標準:
ほぼ全ての音楽配信
Spotify, Apple Music = 44.1kHz

軽い:
CPU負荷低い

十分:
20Hz-20kHz (人間の可聴域)
44.1kHzで完全カバー

48kHz使う場合:
ビデオ制作
映像の標準が48kHz

高い設定 (96kHz+):
CPU負荷2-4倍
メリット少ない
初心者には不要

設定:
Preferences > Audio > Sample Rate
→ 44100
```

### Buffer Size (バッファサイズ)

**最重要設定:**

```
Buffer Sizeとは:

音の処理単位:
小さい → リアルタイム性高い
大きい → CPU負荷低い

選択肢:

32 samples (超低レイテンシー)
64 samples (低レイテンシー)
128 samples (バランス)
256 samples (標準) ← 推奨(制作時)
512 samples (ミックス時)
1024 samples (マスタリング時)
2048 samples (重いプロジェクト)

レイテンシー計算:

128 samples @ 44.1kHz
= 128 / 44100 × 1000
= 2.9ms (体感ゼロ)

256 samples @ 44.1kHz
= 5.8ms (全く問題なし)

512 samples @ 44.1kHz
= 11.6ms (少し感じる)

1024 samples @ 44.1kHz
= 23.2ms (遅延感あり)

推奨設定:

録音時/MIDI演奏時:
128 samples
→ リアルタイム性重視

制作時(通常):
256 samples
→ バランス良い

ミックス時:
512 samples
→ CPU負荷軽減

重いプロジェクト:
1024 samples
→ 安定性優先

切り替え:
作業内容に応じて変更
→ Preferences > Audio > Buffer Size

症状別:

症状: プツプツノイズ
→ Buffer Size を大きく (512, 1024)

症状: レイテンシー(遅延)
→ Buffer Size を小さく (128, 64)

症状: CPU過負荷
→ Buffer Size を大きく
または Freeze Tracks
```

---

## レイテンシー対策

**遅延をゼロに:**

### レイテンシーとは

```
定義:

Input Latency:
鍵盤押す → Ableton が認識
= 入力遅延

Output Latency:
Ableton が音生成 → スピーカー
= 出力遅延

Total Latency:
Input + Output
= 体感する遅延

表示:

Preferences > Audio > Overall Latency:
「Input: 2.9ms / Output: 5.8ms」

許容範囲:

<5ms: 完璧 (体感ゼロ)
5-10ms: 全く問題なし
10-20ms: 少し感じる
>20ms: 不快

目標:
Total Latency < 10ms
```

### レイテンシー低減テクニック

```
1. Buffer Size を小さく:
   256 → 128 → 64

2. 外付けオーディオIF:
   Scarlett Solo 等

3. Direct Monitoring:
   オーディオIFのダイレクトモニター機能
   → Abletonを経由しない
   → レイテンシーゼロ

4. Reduce Latency When Monitoring:
   Preferences > Audio
   → ☑ Reduce Latency

5. Freeze Tracks:
   重いトラックをFreeze
   → CPU負荷減
   → Buffer Size を小さくできる

6. プラグイン削減:
   リバーブ、ディレイ多用
   → レイテンシー増加

7. ASIO Driver (Win):
   必ずASIO使用
   MME/DirectX は遅い
```

---

## MIDI設定

**Preferences > MIDI:**

### MIDI Ports

```
Input:

From [デバイス名]:
接続されたMIDI機器

例:
└─ DDJ-FLX4
└─ AKAI MPK Mini
└─ Launchpad

設定:

Track: On
→ MIDI信号をトラックに送る

Remote: Off (通常)
→ Ableton コントロール用
  (Push 2等)

MPE: Off
→ 特殊なMIDIコントローラー用

Output:

To [デバイス名]:
MIDI信号を送る先

通常:
使わない

用途:
外部シンセをコントロール
```

### DDJ-FLX4をMIDIコントローラー化

```
準備:

DDJ-FLX4をUSB接続:
Mac/PCに接続

Rekordbox閉じる:
Rekordboxと同時使用不可

Ableton設定:

1. Preferences > MIDI

2. Input:
   From DDJ-FLX4: Track On

3. テスト:
   新規MIDIトラック作成
   → DDJ-FLX4のパッド押す
   → MIDI信号入力される

活用:

Performance Pads:
Hot Cue A-H
→ Ableton のクリップ起動

Jog Wheel:
MIDI CC として使用
→ フィルター等をコントロール

制約:
Rekordboxと同時使用不可
どちらか選ぶ
```

### MIDI Keyboard接続

```
推奨機種:

AKAI MPK Mini MK3: ¥13,000
M-Audio Keystation Mini: ¥8,000
Novation Launchkey Mini: ¥13,000

接続:

USB接続:
自動認識

設定:

Preferences > MIDI
→ Input: Track On

使用:

MIDIトラック作成:
音源挿入 (Wavetable等)

鍵盤弾く:
音が出る

録音:
●Record → 弾く
→ MIDIノート記録

利点:
マウスでポチポチより
圧倒的に速い
```

---

## Input/Output Routing

**音の流れを理解:**

### Audio Input

```
Preferences > Audio > Input Config:

┌──────────────────┐
│ ☑ Mono           │
│ ☑ Stereo         │
│ ☐ 3/4            │
│ ☐ 5/6            │
└──────────────────┘

有効化:
使う入力にチェック

Scarlett Solo:
Input 1 (Mono): マイク/ギター
Input 1+2 (Stereo): ステレオ入力

トラックで選択:

Audio From:
└─ Ext. In
    └─ 1 (Mono)
    └─ 1/2 (Stereo)

録音:
このトラックで●Record
→ 外部音を録音
```

### Audio Output

```
Preferences > Audio > Output Config:

┌──────────────────┐
│ ☑ Master         │
│ ☐ 1/2 (別出力)   │
│ ☐ 3/4            │
└──────────────────┘

Master:
通常の出力
→ ヘッドフォン、スピーカー

別出力 (DJ用):
1/2: メイン
3/4: キュー(Cue)
→ DJミキサーと同じ

通常:
Master のみ有効
```

---

## モニタリング設定

**聴き方を選ぶ:**

### Monitor Modes

```
各トラック:

Monitor:
┌──────┐
│ In   │ ← 常時モニター
│ Auto │ ← 自動 (推奨)
│ Off  │ ← モニターなし
└──────┘

In:
常に入力を聴く
→ ボーカル録音時

Auto:
録音時・Arm時のみ聴く
→ 通常はこれ

Off:
モニターしない
→ フィードバック防止

推奨:
Auto
```

### Direct Monitoring

```
オーディオIF側の機能:

Direct Monitor スイッチ:
On → PCを経由せず直接モニター
     レイテンシーゼロ

Off → Ableton経由でモニター
      エフェクトかかる

使い分け:

ボーカル録音:
Direct Monitor On
→ レイテンシーなし

エフェクトかけながら:
Direct Monitor Off
→ リバーブ等聴きながら録音

Focusrite Scarlettの場合:
Direct Monitor ツマミ
→ 中央: 両方ミックス
```

---

## 実践: Audio/MIDI完全セットアップ

**30分の演習:**

### Step 1: Audio設定 (10分)

```
1. Preferences > Audio

2. Audio Device:
   Mac: Core Audio
   Win: ASIO

3. Input/Output:
   内蔵 or Scarlett Solo

4. Sample Rate: 44100 Hz

5. Buffer Size: 256 samples

6. Input Config:
   Mono, Stereo にチェック

7. Output Config:
   Master にチェック

8. Overall Latency 確認:
   <10ms なら完璧

9. 閉じる
```

### Step 2: MIDI設定 (10分)

```
1. MIDI Keyboard接続 (あれば)

2. Preferences > MIDI

3. Input:
   Track On

4. テスト:
   新規MIDIトラック
   音源挿入 (Wavetable)
   鍵盤弾く

5. 音が出たら成功
```

### Step 3: レイテンシーテスト (10分)

```
1. 新規MIDIトラック

2. Wavetable 挿入

3. MIDI Keyboard で弾く
   (またはマウスでMIDI入力)

4. 遅延チェック:
   感じる → Buffer Size 小さく
   感じない → OK

5. CPU負荷チェック:
   右上のCPUメーター
   >80% → Buffer Size 大きく

6. 最適値を見つける:
   128-256 samples が目標
```

---

## トラブルシューティング

### 音が出ない

```
チェック:

1. Output Device:
   正しいデバイス選択？

2. Master Volume:
   上がっている？

3. トラックVolume:
   ミュートになっていない？

4. ヘッドフォン:
   正しく接続？

5. Test Tone:
   Preferences > Audio > Test
   → トーンが聴こえるか
```

### プツプツノイズ

```
原因: Buffer Size 小さすぎ

解決:

1. Buffer Size を大きく:
   256 → 512 → 1024

2. Freeze Tracks:
   重いトラックをFreeze

3. CPU負荷軽減:
   不要なプラグイン削除

4. 他のアプリ閉じる:
   Chrome等のブラウザ
```

### MIDI信号入らない

```
チェック:

1. MIDI Device接続:
   USB接続確認

2. Preferences > MIDI:
   Track On?

3. トラック Arm:
   ●Arm ボタン押してる？

4. MIDI From:
   All Ins または 特定デバイス

5. Test:
   Transport > MIDI Indicator
   → 点灯するか
```

---

## よくある質問

### Q1: Scarlett Soloは必須？

**A:** 最初は不要、本気なら購入

```
内蔵で十分:
MIDI制作のみ
最初の1-2ヶ月

必要になる:

ボーカル録音:
マイク入力必要

ギター録音:
Hi-Z入力必要

低レイテンシー:
リアルタイム演奏

タイミング:
制作に慣れてから
= 2-3ヶ月後でOK

投資:
¥14,000で大幅改善
コスパ良い
```

### Q2: 96kHzの方が音質良い？

**A:** 44.1kHzで十分

```
理論:

人間の可聴域:
20Hz - 20,000Hz

44.1kHz:
最大 22,050Hz まで記録
→ 可聴域を完全カバー

96kHz:
最大 48,000Hz まで
→ 人間には聴こえない

デメリット:

CPU負荷:
2倍

ファイルサイズ:
2倍

プラグイン:
一部対応していない

プロの見解:
99%のプロが44.1kHz
Spotifyも44.1kHz

結論:
44.1kHz で十分
96kHzは不要
```

### Q3: DDJ-FLX4とMIDIキーボード、どっちが先？

**A:** MIDIキーボード

```
優先順位:

1位: MIDI Keyboard
理由: 音階入力が楽
¥8,000-13,000

2位: Audio Interface
理由: 音質とレイテンシー
¥14,000

3位: DDJ-FLX4 MIDI化
理由: あれば便利
でも優先度低い

初心者:
まずMIDI Keyboard
= 制作効率10倍

DJ兼任:
DDJ-FLX4既に持ってる
→ とりあえずMIDI化試す
→ 不便ならMIDI Keyboard購入
```

---

## オーディオインターフェイスの選び方詳細

**機材選定ガイド:**

### 選定基準

```
1. 入力数:

ソロアーティスト:
1-2入力で十分
例: Scarlett Solo, 2i2

バンド録音:
4-8入力
例: Scarlett 4i4, 8i6

スタジオ:
16+入力
例: RME Fireface UCX II

2. 接続方式:

USB 2.0:
最も一般的
レイテンシー: 4-6ms
例: Scarlett シリーズ

USB 3.0/USB-C:
高速転送
レイテンシー: 2-3ms
例: UAD Volt, RME Babyface

Thunderbolt:
最速
レイテンシー: 1-2ms
例: UAD Apollo, RME Fireface

3. ビット深度/サンプルレート:

基本:
24bit/44.1kHz
→ 全機種対応

ハイエンド:
32bit/192kHz
→ プロ機のみ

必要性:
24bit/44.1kHzで十分
高スペックは不要

4. プリアンプ品質:

入門機:
普通の音質
ノイズフロア: -100dB

中級機:
良い音質
ノイズフロア: -110dB

ハイエンド:
最高の音質
ノイズフロア: -120dB以下

実用上:
中級機で十分
```

### 価格帯別おすすめ

```
¥5,000-10,000:

Behringer U-PHORIA UM2:
最安値
1入力、USB 2.0
初めての1台に

Behringer UMC22:
UM2の改良版
ノイズ少ない

¥10,000-20,000:

Focusrite Scarlett Solo (3rd Gen):
定番中の定番
1入力、USB 2.0
Air モード搭載

M-Audio M-Track Solo:
Scarlett競合
コスパ良い

¥20,000-30,000:

Focusrite Scarlett 2i2:
2入力
MIDI In/Out
ベストセラー

Universal Audio Volt 2:
ビンテージコンプ内蔵
独特の音色

PreSonus AudioBox USB 96:
DAWバンドル豊富

¥30,000-50,000:

Audient iD4 MkII:
高品質プリアンプ
コンソールグレード

SSL 2:
SSLサウンド
Legacy 4K モード

Apogee Duet 3:
Mac専用
最高級プリアンプ

¥50,000-100,000:

Focusrite Clarett+ 2Pre:
Thunderbolt接続
超低レイテンシー

Universal Audio Arrow:
UADプラグイン使用可
リアルタイムDSP

¥100,000+:

Universal Audio Apollo Twin X:
業界標準
最高のプラグイン

RME Babyface Pro FS:
最強の安定性
ドライバ完璧

Antelope Audio Zen Go:
モデリング機能
多機能
```

---

## オーディオインターフェイス詳細設定

**各社専用ソフトウェア:**

### Focusrite Control (Scarlett用)

```
起動:

Scarlett接続
→ 自動起動

または:
Applications > Focusrite Control

主要設定:

1. Sample Rate:
   44.1kHz / 48kHz / 88.2kHz / 96kHz

   推奨: 44.1kHz

2. Buffer Size:
   32 / 64 / 128 / 256 / 512 / 1024

   推奨: 256 (制作時)
         128 (録音時)

3. Input:

   Gain:
   入力レベル調整
   目標: -12dB ~ -6dB (ピーク時)

   Air Mode:
   高域を明るくする
   ボーカル/アコギに効果的

   Pad:
   -10dB減衰
   大音量ソース用

4. Output:

   Volume:
   出力音量

   Dim:
   一時的に音量下げる

   Mute:
   出力ミュート

5. Monitor Mix:

   Direct Monitor:
   入力を直接モニター
   レイテンシーゼロ

   DAW:
   Ableton経由でモニター
   エフェクトかかる

   Balance:
   両方のミックス比率

便利機能:

Loopback:
PC内部音を録音
例: YouTube音声を録音

Talkback:
内蔵マイクで通話
リモートセッション時
```

### Universal Audio Console (Apollo/Volt用)

```
特徴:

リアルタイムDSP:
UADプラグイン
録音時にかけられる
レイテンシーゼロ

設定:

1. Input Channel:

   Preamp:
   ゲイン調整

   High Pass Filter:
   ローカット
   80Hz / 100Hz

   Phase:
   位相反転

   Vintage Mode (Voltのみ):
   ビンテージコンプ
   太い音に

2. Insert FX:

   UADプラグイン挿入:
   - 1176 Compressor
   - Neve 1073 EQ
   - LA-2A Compressor
   等

   録音時:
   リアルタイムで効果

   CPU:
   DSPチップで処理
   PC CPUは使わない

3. Monitor Mix:

   Cue Mix:
   各入力のバランス

   Talkback:
   内蔵マイク

注意:

UADプラグイン:
Apollo のみ
Volt は使用不可

購入:
別売り
無料プラグインあり
```

### RME TotalMix FX (Babyface用)

```
特徴:

最強のルーティング:
全入出力を自由にミックス

ゼロレイテンシー:
DSPミキサー内蔵

設定:

1. Input Channels:

   各入力:
   独立したフェーダー

   FX:
   EQ, Dynamics, Reverb
   録音前に適用可能

2. Playback Channels:

   DAWからの出力:
   各トラック個別表示

3. Output Channels:

   ルーティング:
   任意の入力/再生を
   任意の出力へ

   例:
   Input 1 → Output 1
   Input 2 → Output 3/4
   DAW Track 1 → Output 1
   等

4. Snapshots:

   保存:
   複数のミックス設定

   切替:
   ワンクリックで変更

   用途:
   録音 / ミックス / DJ
   各設定を保存

利点:

完璧なドライバ:
絶対に落ちない

低レイテンシー:
業界最速クラス

高音質:
プロスタジオ標準
```

---

## マルチチャンネルルーティング実践

**複雑なルーティング:**

### DJ用セットアップ

```
目的:

Output 1/2: メインスピーカー
Output 3/4: キュー(ヘッドフォン)

必要機材:

4出力以上のオーディオIF:
例: Scarlett 4i4
    RME Babyface Pro

Ableton設定:

1. Preferences > Audio > Output Config:

   ☑ Master (1/2)
   ☑ Ext. Out 3/4

2. トラックごとに設定:

   通常トラック:
   Audio To: Master
   → Output 1/2 へ

   キュートラック:
   Audio To: Ext. Out 3/4
   → Output 3/4 へ

3. Cue設定:

   ヘッドフォンアイコン:
   クリック
   → そのトラックをキュー出力

使用例:

次の曲を準備:
Track 2 をキュー出力
→ ヘッドフォンで確認
→ メインは Track 1

タイミング合わせ:
キューで聴きながら
BPM調整
```

### マルチトラックレコーディング

```
目的:

複数マイクを同時録音
例: ドラム4マイク

必要機材:

4入力以上:
例: Scarlett 4i4, 8i6
    Behringer UMC404HD

Ableton設定:

1. Preferences > Audio > Input Config:

   ☑ 1 (Mono)
   ☑ 2 (Mono)
   ☑ 3 (Mono)
   ☑ 4 (Mono)
   ☑ 1/2 (Stereo)
   ☑ 3/4 (Stereo)

2. 各トラック設定:

   Track 1 (キック):
   Audio From: Ext. In > 1

   Track 2 (スネア):
   Audio From: Ext. In > 2

   Track 3 (ハイハット):
   Audio From: Ext. In > 3

   Track 4 (オーバーヘッド):
   Audio From: Ext. In > 4

3. 同時録音:

   全トラック Arm:
   ●ボタンを全て押す

   録音:
   Transport > ●Record
   → 4トラック同時録音

利点:

後から調整:
各マイクの音量
EQ, コンプ
個別に調整可能

タイミング:
完璧に同期
```

### リアンプ(Re-Amping)設定

```
目的:

ギターをクリーン録音
後からアンプ通す

必要機材:

リアンプボックス:
例: Radial JCR
    Palmer DACCAPO

または:

Di-Boxを逆向きに

接続:

1. 録音時:

   ギター → Hi-Z Input → Ableton
   クリーントーンで録音

2. リアンプ時:

   Ableton → Output 2 → リアンプボックス → アンプ → マイク → Input 1 → Ableton

Ableton設定:

録音時:

Track 1 (ギター):
Audio From: Ext. In > 1
Monitor: In
→ クリーン録音

リアンプ時:

Track 1:
Audio To: Ext. Out 2
→ リアンプボックスへ

Track 2 (リアンプ):
Audio From: Ext. In > 1
Monitor: In
→ アンプ音を録音

利点:

後から調整:
アンプ、マイク位置
何度でも試せる

複数テイク:
1回の演奏で
複数のアンプサウンド
```

---

## 外部ハードウェアシンセ接続

**アナログシンセ統合:**

### 基本接続

```
必要機材:

ハードウェアシンセ:
例: Korg Minilogue
    Moog Mother-32
    Behringer Model D

接続:

MIDI Out (PC):
→ MIDI In (シンセ)

Audio Out (シンセ):
→ Audio In (IF)

Ableton設定:

1. MIDIトラック作成:

   MIDI To: 外部シンセ名

   MIDI From: All Ins

   Monitor: In

2. オーディオトラック作成:

   Audio From: Ext. In > シンセ接続チャンネル

   Monitor: In

   録音待機: ●Arm

使い方:

MIDIノート入力:
MIDIトラックにノート描画
→ シンセから音が出る

録音:
オーディオトラックで●Record
→ シンセの音を録音

リアルタイム:
MIDIトラック再生
= シンセ演奏
同時にオーディオ録音
```

### External Instrument活用

```
External Instrumentとは:

MIDI送信 + Audio受信:
1つのトラックで完結

設定:

1. MIDIトラック作成

2. External Instrument 挿入:

   Instruments > External Instrument

3. パラメータ:

   MIDI To:
   → 外部シンセ選択

   Audio From:
   → 接続チャンネル

   Gain:
   入力レベル調整

   Hardware Latency:
   自動補正
   (重要!)

4. 使用:

   MIDIノート入力:
   このトラックに描画

   音:
   シンセから出る
   同時にAbleton内で聴ける

   エフェクト:
   External Instrument後に
   Abletonエフェクト追加可能

Hardware Latency補正:

目的:
A/D変換の遅延補正

設定:

自動:
Preferences > Audio
Overall Latency の値

手動:
Click音とシンセ音を録音
→ ズレを測定
→ Latency値に入力

重要性:
他トラックと同期
正確なタイミング
```

### 複数シンセ同時使用

```
セットアップ:

シンセ1: Minilogue
MIDI Ch: 1
Audio: Input 1/2

シンセ2: Volca Bass
MIDI Ch: 2
Audio: Input 3

MIDI接続:

MIDI Out (PC):
→ MIDI Splitter
  ├─ Minilogue
  └─ Volca Bass

または:

MIDI Thru 活用:
PC → Minilogue MIDI In
Minilogue MIDI Thru → Volca MIDI In

チャンネル設定:

各シンセ:
異なるMIDI Ch設定

Ableton:

Track 1 (Minilogue):
External Instrument
MIDI To: Ch 1
Audio From: 1/2

Track 2 (Volca):
External Instrument
MIDI To: Ch 2
Audio From: 3

使用:

同時演奏:
両トラックにノート
→ 2台同時に鳴る

個別制御:
各トラック独立
```

---

## レイテンシー最適化の実践

**究極の低レイテンシー:**

### システム最適化 (Mac)

```
1. 不要なアプリ終了:

CPU喰い:
Chrome (全タブ)
Photoshop
Final Cut Pro

バックグラウンド:
Dropbox
Google Drive
Spotify

チェック:
Activity Monitor
→ CPU使用率確認

2. 省エネ設定無効化:

System Preferences > Energy Saver:
☐ Put hard disks to sleep
☐ Automatic graphics switching

目的:
CPUパワーを常に最大

3. Spotlight無効化 (任意):

System Preferences > Spotlight:
プロジェクトフォルダを除外

理由:
インデックス作成でCPU使用

4. Time Machine一時停止:

バックアップ中:
CPU/Disk負荷高い

作業中:
一時停止

5. Bluetooth無効化 (有線時):

System Preferences > Bluetooth:
オフ

理由:
わずかにCPU削減

6. Wi-Fi無効化 (有線LAN時):

ネットワーク:
有線LANのみ

理由:
CPU優先度向上
```

### システム最適化 (Windows)

```
1. 電源プラン:

Control Panel > Power Options:
→ High Performance

詳細設定:
Processor power management:
Minimum: 100%
Maximum: 100%

2. 不要サービス停止:

Services.msc:
以下を無効化:

Windows Search:
インデックス作成

Superfetch:
プリフェッチ

Windows Update:
自動更新
(手動で更新)

3. ASIO4ALL設定 (内蔵Audio時):

ASIO4ALL Control Panel:

Buffer Size: 128-256

Advanced Options:
☑ Force WDM
☑ Allow pull mode

デバイス選択:
使用デバイスのみ有効

4. CPUコア割り当て:

Task Manager:
Ableton Live 右クリック
→ Set Affinity
→ 特定コアに固定

効果:
CPU効率向上

5. リアルタイム優先度 (上級者):

Task Manager:
Ableton Live 右クリック
→ Priority > High

注意:
システム不安定になる可能性
通常は不要
```

### Ableton内部最適化

```
1. Multicore Support:

Preferences > CPU:
☑ Multicore/Multiprocessor Support

コア数:
自動検出
全コア使用

2. CPU Usage Simulator:

同設定画面:
使用率表示

目標:
<70% (安定)

3. Audio設定:

Sample Rate: 44.1kHz
(48kHz以上は避ける)

Buffer Size:
作業内容で調整

Driver Error Compensation:
Auto (通常)

4. Freeze Tracks:

CPU重いトラック:
右クリック > Freeze Track

効果:
CPU負荷激減
オーディオ化

解除:
Unfreeze で編集可能

5. Flatten:

Freeze より強力:
右クリック > Flatten

効果:
完全にオーディオ化
MIDI情報消失

使い時:
最終調整時
```

---

## 高度なトラブルシューティング

**難解な問題の解決:**

### ドライバー問題

```
症状:

デバイスが認識されない:
Preferences > Audio
デバイスリストに表示されない

音が出ない:
デバイス選択できるが音なし

クラッシュ:
Ableton起動時にフリーズ

解決手順:

1. ドライバー再インストール:

   Mac:
   メーカーサイトから最新版
   → .dmg インストール
   → 再起動

   Windows:
   デバイスマネージャー
   → オーディオデバイス右クリック
   → ドライバー更新
   または再インストール

2. USBポート変更:

   USB 3.0 → USB 2.0 (またはその逆)

   理由:
   互換性問題

   別のポート試す:
   直接マザーボード接続
   (ハブ経由避ける)

3. ケーブル交換:

   不良ケーブル:
   意外に多い

   高品質ケーブル:
   認証品推奨

4. OS再起動:

   シンプルだが効果的:
   ドライバー再読み込み

5. 安全モード起動 (Mac):

   Shift押しながら起動
   → サードパーティ拡張無効
   → 問題切り分け

6. クリーンインストール:

   最終手段:
   ドライバー完全削除
   → OS再起動
   → 最新版インストール
```

### レイテンシー補正問題

```
症状:

タイミングずれ:
MIDIと録音音声がずれる

外部シンセ遅延:
他トラックより遅れて聞こえる

解決:

1. Driver Error Compensation:

   Preferences > Audio:
   Driver Error Compensation
   → 自動調整

   通常: 自動で十分

2. Track Delay:

   各トラックに:
   Utility デバイス挿入
   → Delay (ms) 調整

   正の値: 遅らせる
   負の値: 早める

3. 手動測定:

   準備:
   クリックトラック作成
   外部機器で録音

   測定:
   Click音と録音音のズレ
   → サンプル数計算

   補正:
   External Instrument
   → Hardware Latency に入力

4. 全体オフセット:

   Preferences > Record/Warp/Launch:
   Overall Latency Compensation
   → 有効化

   効果:
   全トラック自動調整
```

### CPU過負荷の深刻なケース

```
症状:

常に90%以上:
何もしてなくても高負荷

プツプツが止まらない:
Buffer Size上げても改善せず

フリーズ頻発:
保存できないレベル

解決:

1. 問題トラック特定:

   半分ずつミュート:
   どのトラックが重いか特定

   プラグイン無効化:
   1つずつオフにして確認

   最重量犯人:
   通常はリバーブ、ビジュアライザー

2. 代替手段:

   重いプラグイン:
   軽量版に置き換え

   例:
   Valhalla VintageVerb
   → Ableton Reverb

   CPU使用率:
   1/3に削減

3. Render in Place:

   重いトラック:
   右クリック > Freeze
   さらに Flatten

   効果:
   オーディオ化で完全解決

4. プロジェクト分割:

   巨大プロジェクト:
   セクションごとに分割

   例:
   Intro / Verse / Chorus
   各セクション別ファイル

   後で統合:
   最終段階で結合

5. PC アップグレード:

   根本的解決:
   CPUアップグレード
   RAMを16GB→32GB

   投資:
   長期的には最良の選択
```

### MIDIタイミング問題

```
症状:

MIDIノートずれる:
Quantize してもずれ感じる

外部シンセタイミング悪い:
もたつく

解決:

1. MIDI Clock設定:

   Preferences > MIDI:
   Sync タブ

   External Sync:
   オフ推奨

   理由:
   外部同期で遅延

2. Buffer Size 最適化:

   MIDI演奏時:
   64-128 samples

   効果:
   入力遅延最小化

3. MIDI Thru 遅延:

   問題:
   多段接続で蓄積

   解決:
   MIDI Splitter 使用
   並列接続

4. USB-MIDI 遅延:

   問題:
   USB MIDI は若干遅い

   解決:
   5-pin MIDI ケーブル
   専用MIDIインターフェース

   例:
   iConnectivity mio
   MOTU MIDI Express

5. Reduce Latency When Monitoring:

   Preferences > Audio:
   ☑ 有効化

   効果:
   モニター時のみ最適化
   レイテンシー削減
```

---

## プロフェッショナル設定テンプレート

**用途別最適設定:**

### ボーカル録音セットアップ

```
機材:

マイク: Shure SM58 / Audio-Technica AT2020
オーディオIF: Scarlett Solo
ポップガード必須

Audio設定:

Sample Rate: 44.1kHz
Buffer Size: 128 samples
Input: Mono
Gain: -12dB〜-6dB (ピーク時)

Ableton設定:

1. Audio Track作成:

   Audio From: Ext. In > 1
   Monitor: In (録音中聴く)
   または Auto

2. エフェクト挿入 (録音前):

   なし推奨
   → クリーン録音
   → 後から処理

   どうしてもなら:
   軽いコンプ
   EQ (ローカットのみ)

3. Direct Monitor:

   Scarlett:
   Direct Monitor On

   利点:
   レイテンシーゼロ
   自然な聴こえ方

録音手順:

1. ゲイン調整:
   大声で歌う
   → Scarlett のLED緑
   → 赤にならないギリギリ

2. ヘッドフォン音量:
   快適なレベル
   大きすぎない

3. テストテイク:
   1フレーズ録音
   → 聴き返す
   → 音質確認

4. 本番:
   ●Record
   → パンチイン/アウト活用
```

### ライブパフォーマンスセットアップ

```
目的:

ステージ使用:
最大安定性
最低レイテンシー

Audio設定:

Sample Rate: 44.1kHz
Buffer Size: 128 samples
または 64 (マシンパワーあれば)

Output:
Master: メインスピーカー
Ext. Out 3/4: モニター (自分用)

Ableton設定:

1. CPU最適化:

   全トラックFreeze:
   ライブ前に全部Freeze

   不要デバイス削除:
   使わないエフェクト全削除

   ビジュアル削減:
   波形表示オフ
   CPU優先

2. バックアップ設定:

   自動保存:
   Preferences > File Folder
   Auto Save: 1 minute

   バックアップファイル:
   USB メモリに複製

3. MIDIコントローラー:

   必須機材:
   Launchpad / Push 2
   DDJ-FLX4

   マッピング:
   MIDI Map Mode
   → 全クリップマッピング

4. フェイルセーフ:

   プラン B:
   別トラックに同じクリップ

   緊急停止:
   Master Fader 活用

本番チェックリスト:

□ オーディオIF接続確認
□ Buffer Size 128
□ 全トラックFreeze済み
□ MIDIコントローラー動作OK
□ バックアップファイル持参
□ ヘッドフォン + 予備
□ 電源アダプタ
□ USBケーブル予備
```

### ミックス/マスタリングセットアップ

```
目的:

最高音質:
CPU使い放題
レイテンシー不要

Audio設定:

Sample Rate: 44.1kHz
(or 48kHz if ビデオ用)

Buffer Size: 1024 samples
または 2048 (安定性最優先)

Dither: POW-r 2 (16bit export時)

Ableton設定:

1. リファレンストラック:

   別トラック:
   商用リリース曲
   → 音質比較用

   A/B比較:
   自分のミックス
   ⇔ プロの音

2. メータリング:

   Youlean Loudness Meter:
   LUFS測定

   目標:
   Spotify: -14 LUFS
   Apple Music: -16 LUFS
   YouTube: -13 LUFS

3. モニター環境:

   スピーカー:
   ニアフィールドモニター
   例: Yamaha HS5

   ヘッドフォン:
   フラット特性
   例: AKG K240, Sony MDR-7506

   音量:
   会話レベル (70-75dB)

4. Export設定:

   File Type: WAV
   Bit Depth: 24bit (配布用)
              16bit (CD/配信用 + Dither)
   Sample Rate: プロジェクトと同じ
   Normalize: Off
   Dither: POW-r 2 (16bit時のみ)

ミックス手順:

1. Gain Staging:
   全トラック -6dB余裕
   Master Fader 0dB

2. バランス:
   ボリュームのみで調整
   EQ/Compは後

3. EQ:
   引き算EQ優先
   (boost より cut)

4. Compression:
   控えめに
   2-3dB reduction

5. Reverb/Delay:
   Sendトラック活用
   直接挿さない

6. Master処理:
   EQ → Compressor → Limiter
   各2-3dB程度
```

---

## 予防的メンテナンス

**トラブルを未然に防ぐ:**

### 定期チェックリスト (月1回)

```
オーディオIF確認:

□ ドライバー最新版チェック:
  メーカーサイト確認
  更新あれば適用

□ ファームウェア更新:
  本体ファームウェア確認
  最新版に更新

□ 接続確認:
  USBケーブル劣化チェック
  接触不良なし

□ クリーニング:
  端子を接点復活剤で清掃
  ホコリ除去

MIDI機器確認:

□ 電池交換:
  ワイヤレスMIDI
  電池残量確認

□ ファームウェア:
  MIDIキーボード更新

□ 設定リセット:
  初期値に戻す
  不要な設定クリア

システム確認:

□ Ableton更新:
  最新バージョンチェック
  安定版に更新

□ OS更新:
  セキュリティパッチ適用
  メジャーアップデートは慎重に

□ ストレージ空き容量:
  最低50GB確保
  不要ファイル削除

□ バックアップ:
  全プロジェクト外部HDD保存
  クラウドバックアップ
```

### トラブル予防テクニック

```
設定の記録:

1. スクリーンショット撮影:
   完璧な設定画面
   → フォルダ保存

   撮影対象:
   - Preferences > Audio
   - Preferences > MIDI
   - オーディオIF設定画面

2. 設定ファイルバックアップ:

   Mac:
   ~/Library/Preferences/Ableton/
   → Time Machine バックアップ

   Windows:
   C:\Users\[User]\AppData\Roaming\Ableton\
   → コピー保存

3. ドキュメント化:
   テキストファイルに記録

   内容例:
   Buffer Size: 256
   Sample Rate: 44100
   Input Gain: 3時方向
   等

物理的保護:

1. ケーブル管理:
   結束バンドで整理
   踏まない位置に配線

2. 電源安定化:
   UPS (無停電電源装置)
   突然の停電対策

3. 温度管理:
   PC/オーディオIF換気
   熱暴走防止

4. 静電気対策:
   冬場は加湿器
   機材触る前に金属触る

ソフトウェア対策:

1. プロジェクトテンプレート:
   完璧な設定を保存
   New Live Set from Template

   含む内容:
   - Audio/MIDI設定
   - トラック構成
   - Send/Return
   - MIDIマッピング

2. 自動保存設定:
   Preferences > File Folder
   Auto Save: 1 minute
   Keep: 100 backups

3. プラグイン管理:
   定期的にスキャン
   不要プラグイン削除
   安定性向上
```

---

## 環境別セットアップガイド

**シーン別最適設定:**

### ホームスタジオ

```
特徴:

時間制限なし:
じっくり制作可能

音質優先:
レイテンシー多少OK

推奨設定:

Audio:
Sample Rate: 44.1kHz
Buffer Size: 512 samples
Monitoring: Off (スピーカー)

機材:

オーディオIF:
Scarlett 2i2 以上

モニター:
スタジオモニタースピーカー
ヘッドフォン併用

MIDI:
フルサイズキーボード
61鍵以上推奨

環境:

防音:
簡易吸音材
夜間作業用ヘッドフォン

電源:
タップ整理
ノイズ対策

配置:
エルゴノミクス重視
長時間作業快適に
```

### モバイルセットアップ (ノートPC)

```
特徴:

持ち運び:
カフェ、移動中制作

省電力:
バッテリー駆動

推奨設定:

Audio:
Sample Rate: 44.1kHz
Buffer Size: 512 samples
内蔵オーディオ使用

または:

Portable IF:
Apogee Duet 3 (Mac)
Audient iD4 (Win/Mac)

MIDI:
コンパクトキーボード:
AKAI MPK Mini
Arturia MiniLab

電源管理:

Mac:
Battery Health
省エネモード無効

Windows:
High Performance プラン
CPU throttling 無効

ストレージ:

外付けSSD:
プロジェクトファイル
サンプルライブラリ

クラウド同期:
Dropbox / Google Drive
自動バックアップ
```

### コラボレーション環境

```
特徴:

複数人作業:
プロデューサー + エンジニア

リアルタイム共有:
画面・音声共有

推奨設定:

Audio:
Sample Rate: 44.1kHz
Buffer Size: 256 samples
複数出力設定

ルーティング:

Output 1/2:
メインモニター (共有)

Output 3/4:
個別ヘッドフォン
(プロデューサー)

Output 5/6:
個別ヘッドフォン
(エンジニア)

必要機材:

オーディオIF:
最低6出力
例: Scarlett 6i6
    MOTU M4

ヘッドフォンアンプ:
複数人分配

通信:

オンラインコラボ:
Audiomovers Listento
→ 超低レイテンシー配信

Zoom/Discord:
画面共有
音質は妥協
```

---

## ケーススタディ: 実際のトラブル解決例

**現場で起こった問題と解決法:**

### Case 1: ライブ中に突然音が出なくなった

```
状況:

DJ本番中:
突然全ての音が消失
観客は待機中
緊急対応必要

原因特定:

Buffer Size:
誤って32 samplesに設定
CPU過負荷でクラッシュ

解決手順 (3分以内):

1. Ableton 強制終了:
   Cmd+Q (Mac) / Alt+F4 (Win)

2. 再起動:
   最後の自動保存ファイル読込

3. Buffer Size変更:
   Preferences > 256 samples

4. 全トラック確認:
   ミュート解除

5. 再開:
   MCで謝罪しつつスムーズに復帰

教訓:

本番前チェック:
必ずBuffer Size確認
推奨: 128-256

自動保存:
1分間隔設定必須

バックアッププロジェクト:
USBメモリに別保存
```

### Case 2: MIDIキーボードが半音ずれる

```
状況:

MIDI録音:
全ノートが半音高い
または低い

原因:

Transpose設定:
MIDIキーボード本体が
+1または-1に設定

解決:

1. キーボード本体確認:
   Transpose ボタン
   → 0 (ゼロ) に戻す

2. Ableton側確認:
   MIDI Track
   → Pitch -12/+12 確認
   → 0 に戻す

3. MIDIエフェクト確認:
   Pitch デバイス削除

4. テスト:
   C音を弾く
   → C音が鳴るか確認

予防:

セットアップ時:
必ず音階確認
C-D-E-F-G-A-B-C

キーボード設定リセット:
本番前に初期化
```

### Case 3: 録音したボーカルが極端に小さい

```
状況:

ボーカル録音:
波形がほとんど見えない
ゲインを上げるとノイズまみれ

原因:

入力ゲイン不足:
オーディオIF側のゲインが低すぎ
録音レベル: -40dB (適正: -12dB)

解決:

1. 録音前に再確認:

   Scarlett Solo Gain:
   ノブを回す
   LED確認

   目標:
   大声で歌う → 緑点灯
   たまに黄色
   赤は絶対NG

2. テスト録音:

   1フレーズ録音
   波形確認
   -12dB〜-6dB (ピーク時)

3. 既存録音の救済:

   Utility デバイス:
   Gain +12dB〜+24dB

   EQ Eight:
   High Pass 80Hz
   → ノイズ軽減

   Compressor:
   Threshold -30dB
   Ratio 4:1
   → ダイナミクス補正

   注意:
   ノイズは消えない
   理想: 再録音

教訓:

録音は一発勝負:
必ずテストテイク
音量確認必須

適正レベル:
-12dB〜-6dB (ピーク時)
-18dB〜-12dB (平均時)
```

---

## まとめ

### 初心者向けクイックスタート

```
最小限これだけ:

1. Preferences > Audio:
   □ Device: Core Audio / ASIO
   □ Sample Rate: 44100 Hz
   □ Buffer Size: 256 samples

2. Preferences > MIDI:
   □ Input: Track On

3. テスト:
   □ 音が出るか確認
   □ MIDIキーボード反応するか

これでOK:
すぐ制作開始可能
```

### 中級者向けチェックリスト

```
最適化:

Audio:
□ Buffer Size を作業内容で切替
  (録音: 128 / 制作: 256 / ミックス: 512)
□ Overall Latency < 10ms 確認
□ CPU負荷 < 70% 維持

MIDI:
□ 外部シンセ External Instrument 設定
□ MIDIコントローラーマッピング
□ Hardware Latency 補正

システム:
□ 不要アプリ終了
□ 定期的に Freeze Tracks
```

### 上級者向け完璧セットアップ

```
プロ環境:

機材:
□ 高品質オーディオIF (RME / UAD)
□ MIDIキーボード
□ モニタースピーカー
□ 処理用ヘッドフォン

設定:
□ Thunderbolt 接続
□ Dedicated ASIO Driver
□ マルチチャンネルルーティング
□ External Instruments 完璧補正

ワークフロー:
□ テンプレートプロジェクト
□ カスタムMIDIマッピング
□ Freeze/Flatten 戦略
□ バックアップ自動化

投資総額:
¥50,000 - 200,000
でプロレベル環境構築可能
```

---

**次は:** [ファイル管理](./file-management.md) - プロジェクトとサンプルの整理術
