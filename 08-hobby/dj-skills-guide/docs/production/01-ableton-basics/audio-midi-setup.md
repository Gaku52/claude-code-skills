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

## まとめ

### Audio設定

```
□ Device: Core Audio / ASIO
□ Sample Rate: 44100 Hz
□ Buffer Size: 256 samples (通常)
□ Latency: <10ms
□ Input/Output Config 設定
```

### MIDI設定

```
□ MIDI Device接続
□ Input: Track On
□ MIDIキーボードテスト
□ DDJ-FLX4 MIDI化 (任意)
```

### Buffer Size使い分け

```
録音/演奏: 128 samples
制作: 256 samples
ミックス: 512 samples
重い: 1024 samples
```

---

**次は:** [ファイル管理](./file-management.md) - プロジェクトとサンプルの整理術
