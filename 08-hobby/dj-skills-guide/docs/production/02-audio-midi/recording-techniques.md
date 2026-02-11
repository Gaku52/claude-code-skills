# 録音テクニック

完璧な録音を。Audio・MIDI両方の録音方法、オーバーダビング、パンチイン/アウト、コンピングまで完全マスター。

## この章で学ぶこと

- Audio録音（マイク、ライン入力）
- MIDI録音（キーボード、打ち込み）
- Recording Mode（Session vs Arrangement）
- オーバーダビング（重ね録り）
- パンチイン/アウト
- コンピング（テイク選択）
- レイテンシー対策
- モニタリング設定

---

## なぜ録音テクニックが重要なのか

**制作の質を決める:**

```
プロとアマの差:

アマチュア:
録音方法知らない
何度も失敗
音質悪い

結果:
時間の無駄
やり直し多い

プロ:
録音設定完璧
1発で決める
高音質

結果:
効率的
クオリティ高い

統計:

録音時の設定:
後から直せない
= 最重要

良い録音:
ミックス楽
マスタリング楽

悪い録音:
どんなに処理しても
救えない
```

---

## Recording Mode

**Session vs Arrangement:**

### Session View録音

```
特徴:

Clip単位:
各Slotに録音

ループ録音:
自動的にループ

リアルタイム:
ライブ演奏向き

用途:

ライブパフォーマンス:
即興

アイデア収集:
素早くCapture

ループ作成:
4-8小節パターン

操作:

1. トラック選択

2. 空のClip Slot選択

3. Session Record ボタン (●)
   または
   トラックのArm (●)

4. 演奏開始

5. 自動的にループ

6. 停止
```

### Arrangement View録音

```
特徴:

時間軸:
Linear録音

精密:
タイミング正確

編集向き:
後処理しやすい

用途:

曲構成:
イントロ → ドロップ

重ね録り:
複数テイク

本番録音:
最終形

操作:

1. トラック選択

2. トラックArm (●)

3. 再生ヘッド配置

4. Arrangement Record (●)

5. 演奏

6. 停止
```

---

## Audio録音

**マイク・楽器を録る:**

### 録音準備

```
機材接続:

オーディオインターフェース:
DDJ-FLX4:
MICポート × 2

外部IF:
Focusrite Scarlett等

マイク:
XLR接続
+48V ファンタム電源 (コンデンサー)

ギター/ベース:
1/4" TRS ケーブル

Ableton設定:

Preferences > Audio:

Audio Device:
DDJ-FLX4 (または外部IF)

Sample Rate:
44.1 kHz (推奨)

Buffer Size:
128 samples (録音時)
512-1024 samples (ミックス時)

Input Config:
入力チャンネル確認
Mono / Stereo設定

トラック設定:

Audio Track作成:
Cmd+T

Input:
Ext. In → 1 (Mic 1)
または
Ext. In → 1/2 (Stereo)

Monitor:
In (常に聴こえる)
Auto (録音中のみ)

Arm:
● On (録音可能)
```

### 録音実行

```
Step 1: レベル調整 (5分)

1. トラックArm: ●

2. Monitor: In

3. 演奏/歌う:
レベルメーター確認

4. ゲイン調整:
IF のGainつまみ

目標:
ピーク -12 〜 -6 dB
赤色(Clip)絶対ダメ

5. 確認:
ヘッドフォンで音質

Step 2: カウントイン設定 (2分)

Preferences > Record Warp Launch:

Count In: 1 Bar (推奨)
または
2 Bars, 4 Bars

Metronome:
☑ On

効果:
録音前に1小節カウント
準備できる

Step 3: 録音 (本番)

Session View:

1. Clip Slot選択

2. Cmd+F9 (Start Record)

3. カウントイン待つ

4. 演奏

5. Space (Stop)

Arrangement View:

1. 再生ヘッド配置

2. F9 (Arrangement Record)

3. カウントイン

4. 演奏

5. Space

Step 4: 確認

1. Arm: Off (誤録音防止)

2. 再生:
Space

3. 波形確認:
クリップなし
ノイズなし

4. やり直し:
Delete → 再録音
```

---

## MIDI録音

**キーボード・打ち込み:**

### MIDIキーボード接続

```
接続方法:

USB MIDI:
MIDIキーボード → Mac/PC
USB接続のみ

5pin MIDI:
キーボード → IF → Mac/PC
古い機材

DDJ-FLX4:
MIDI入力なし
→ 別途MIDIキーボード必要

Ableton認識:

Preferences > Link Tempo MIDI:

MIDI Ports:

Input: Your MIDI Keyboard
Track: ☑ On
Remote: ☐ Off

トラック設定:

MIDI Track作成:
Cmd+T

音源読み込み:
Wavetable, Operator等

Input:
All Ins → All Channels

Monitor:
In (常に音が鳴る)

Arm:
● On
```

### MIDI録音実行

```
リアルタイム録音:

1. トラックArm

2. Monitor: In

3. 音色確認:
鍵盤弾いて確認

4. F9 (Record)

5. カウントイン

6. 演奏:
鍵盤を弾く

7. Space (Stop)

結果:
MIDIノートが記録される

ステップ録音 (推奨):

1. 空のMIDI Clip作成

2. Clip View表示

3. ピアノロールで:
ダブルクリック入力

メリット:
タイミング完璧
Quantize不要

Recording Quantization:

Preferences > Record Warp Launch:

MIDI Record Quantization:
1/16 (推奨)
または
None

効果:
録音中にリアルタイムQuantize
ズレ自動修正

注意:
機械的になる可能性
```

---

## オーバーダビング

**重ね録り:**

### オーバーダビングとは

```
定義:
既存の録音に追加録音

用途:

複数テイク:
ドラム → ベース → ギター

ハモリ:
メインボーカル + ハモリ

パート追加:
ストリングス追加

Session View:

同じClip Slot:
上書きされる (デフォルト)

別のClip Slot:
別トラックに録音

Arrangement View:

新規Clip:
自動的に作成

既存Clip上:
上書き (注意)
```

### オーバーダビング実行

```
方法1: 別トラック (推奨)

1. 既存トラック再生

2. 新規トラック作成

3. 新規トラックArm

4. 録音開始:
既存音を聴きながら

5. 完了

メリット:
元のテイク保存
安全

方法2: 同じトラック

Session View:

別のClip Slot使用:
縦に並ぶ

Arrangement View:

Preferences > Record Warp Launch:

Create Fades: ☑
Exclusive Arm: ☐

録音:
既存Clipの隣に

方法3: MIDI重ね録り

1. MIDI Clip選択

2. Arm + Record

3. 演奏:
既存ノートに追加

4. 結果:
ノートが追加される

注意:
上書きではなく追加
```

---

## パンチイン/アウト

**部分的な録音:**

### パンチイン/アウトとは

```
定義:
指定範囲のみ録音

用途:

ミス修正:
1箇所だけやり直し

部分差し替え:
サビだけ録り直し

効率化:
全体録音不要

Ableton での方法:

手動:
Recordボタン手動On/Off

Loop範囲:
Loop内のみ録音
```

### 手動パンチイン

```
Arrangement View:

Step 1: 準備

1. 再生ヘッドを配置:
録音開始位置の少し前

2. トラックArm

3. 再生開始:
Space

Step 2: パンチイン

1. 録音開始位置で:
F9 (Record)

2. 演奏

3. 録音終了位置で:
F9 (Record停止)

4. 完了

コツ:

準備時間:
4-8拍前から再生

タイミング:
F9を正確に押す

練習:
何度かリハーサル
```

### Loop録音

```
設定:

1. Loop Range設定:
録音したい範囲

2. Loop: ☑ On

3. トラックArm

録音:

1. F9 (Record)

2. Loopが回る:
何度も録音可能

3. 納得いくまで:
ループ繰り返し

4. Stop

Session View では:

Clip Loop Lengthが範囲
自動的にループ録音

用途:

ドラムループ:
4小節を完璧に

ベースライン:
8小節録音

ボーカルフレーズ:
サビ部分
```

---

## コンピング

**ベストテイク選択:**

### コンピングとは

```
定義:
複数テイクから良い部分を選択

例:

テイク1: イントロ◎, サビ△
テイク2: イントロ△, サビ◎
テイク3: イントロ○, サビ○

→ コンピング:
イントロ:テイク1
サビ:テイク2

結果:
完璧な1トラック

Abletonでの方法:

Take Lanes (Live 11+):
複数テイクを管理

手動編集:
Cmd+E で分割・選択
```

### コンピング実行

```
方法1: Take Lanes (Live 11+)

1. トラックArm

2. Loop録音:
複数回演奏

3. Take Lanes表示:
トラック左の [▼]

4. 各テイク表示:
縦に並ぶ

5. 良い部分選択:
クリックして有効化

6. Flatten:
1つのClipに統合

方法2: 手動編集 (Live 10以前)

1. 複数テイク録音:
別々のClipに

2. 良い部分を探す:
各Clip再生

3. 必要部分を切り出し:
Cmd+E で分割

4. 1つのトラックに配置:
コピー&ペースト

5. Consolidate:
Cmd+J で統合

方法3: 別トラック比較

1. テイクごとに別トラック

2. Solo/Mute で比較:
S / M キー

3. ベスト選択

4. 不要トラック削除
```

---

## モニタリング設定

**録音中の音を聴く:**

### Monitor設定

```
3つのモード:

In (常時モニタリング):

機能:
常に入力音が聴こえる

用途:
Audio録音
MIDI演奏

注意:
レイテンシー感じる

Auto (自動):

機能:
Arm時のみ聴こえる
再生中は聴こえない

用途:
ほとんどの場合

推奨:
これが標準

Off (モニタリングなし):

機能:
入力音聴こえない

用途:
再生専用トラック

設定場所:

トラック: In/Out セクション
Monitor: In / Auto / Off

推奨設定:

Audio録音: Auto
MIDI録音: In
再生のみ: Off
```

### Direct Monitoring

```
オーディオIF の機能:

Direct Monitoring:
IFで直接モニタリング
DAWを通さない

メリット:

レイテンシーゼロ:
遅延なし

CPU負荷なし:
IFで処理

デメリット:

エフェクト聴けない:
DAWのプラグイン無効

設定:

オーディオIF:
Direct Mon: On

Ableton:
Monitor: Off (重複防止)

または:

IF: Direct Mon Off
Ableton: Monitor In

推奨:

レイテンシー問題あり:
Direct Mon: On

問題なし:
Ableton Monitor: In
→ エフェクト使える
```

---

## レイテンシー対策

**遅延を最小化:**

### レイテンシーとは

```
定義:
入力から出力までの遅延

原因:

ADコンバーター:
アナログ → デジタル変換

バッファーサイズ:
音声処理のバッファー

DAコンバーター:
デジタル → アナログ変換

体感:

10 ms以下: 感じない
10-20 ms: わずかに感じる
20-50 ms: 明確に感じる
50 ms以上: 演奏不可能

測定:

Preferences > Audio:

Overall Latency:
Input Latency + Output Latency
例: 5.8 ms + 5.8 ms = 11.6 ms
```

### バッファーサイズ調整

```
設定:

Preferences > Audio:

Buffer Size:
32 samples (最小)
64 samples (小)
128 samples (録音推奨)
256 samples (通常)
512 samples (ミックス)
1024 samples (負荷大時)
2048 samples (マスタリング)

計算:

レイテンシー (ms) = バッファーサイズ / サンプルレート × 1000

44.1kHz, 128 samples:
128 / 44100 × 1000 = 2.9 ms

44.1kHz, 512 samples:
512 / 44100 × 1000 = 11.6 ms

推奨設定:

録音時:
128 samples
= レイテンシー小

ミックス時:
512-1024 samples
= CPU余裕

注意:

小さすぎ:
CPU負荷高い
ノイズ・途切れ

大きすぎ:
レイテンシー大
演奏しにくい
```

### Driver Errorハンドリング

```
問題:

バッファーサイズ小さい
+ CPU負荷高い
= Driver Error (CPU Overload)

対策:

1. バッファーサイズ増やす:
128 → 256

2. Freeze Track:
重いトラックをFreeze

3. プラグイン整理:
不要なものOff

4. Sample Rate確認:
96kHz → 44.1kHz

5. バックグラウンドアプリ:
Chrome等を終了

緊急時:

Preferences > CPU:

Multi-Core: ☑
Plug-In Load: 50%
```

---

## 実践: ボーカル録音

**完璧な録音:**

### Step 1: 準備 (10分)

```
機材:

マイク:
コンデンサーマイク推奨
(RODE NT1-A, Audio-Technica AT2020等)

オーディオIF:
ファンタム電源 +48V

ポップガード:
パ行・バ行対策

ヘッドフォン:
密閉型

接続:

1. マイク → IF (XLR)

2. +48V: On

3. IF → Mac/PC (USB)

4. ヘッドフォン → IF

Ableton設定:

1. Audio Track作成

2. Input: Ext. In 1

3. Monitor: In

4. Arm: ●

5. レベル確認:
ゲイン調整
ピーク -12 〜 -6 dB
```

### Step 2: 録音 (15分)

```
1. カウントイン設定:
   2 Bars

2. Metronome: On
   Click音量調整

3. リハーサル:
   1-2回通し練習

4. 本番録音:
   F9 → 歌う → Space

5. 確認:
   再生して聴く

6. 必要なら:
   テイク2, 3
```

### Step 3: コンピング (10分)

```
1. ベストテイク選択:
   各テイク再生

2. 良い部分切り出し:
   Cmd+E で分割

3. 配置:
   1つのトラックに

4. クロスフェード:
   つなぎ目を滑らかに

5. Consolidate:
   Cmd+J

6. 完成
```

---

## よくある質問

### Q1: レイテンシーが気になる

**A:** バッファーサイズを小さく

```
現在:
512 samples
= 11.6 ms レイテンシー

解決:

1. Buffer Size:
   512 → 128
   = 2.9 ms

2. それでもダメ:
   Direct Monitoring: On
   (IFの機能)

3. Mac の場合:
   CoreAudio は優秀
   128 samplesで安定

注意:

CPU負荷:
小さいバッファー = 高負荷

トラック数多い:
Freeze使用
```

### Q2: 録音レベルが小さい

**A:** ゲインを上げる

```
確認:

1. IF のゲインつまみ:
   右に回す

2. 目標レベル:
   ピーク -12 〜 -6 dB
   黄色表示

3. Clip (赤) は絶対ダメ:
   歪む

4. Ableton のフェーダー:
   0 dB (真ん中)

ゲイン vs フェーダー:

ゲイン (IF):
録音レベル
こっちで調整

Fader (Ableton):
再生音量
後で調整

推奨:

録音時:
ゲインで適正レベル

ミックス時:
フェーダーで音量バランス
```

### Q3: MIDIが録音されない

**A:** Armとモニター確認

```
チェックリスト:

1. トラックArm:
   ● が赤く点灯

2. Monitor:
   In (常に音が鳴る)

3. Input:
   All Ins / All Channels

4. 音源:
   Instrument挿入済み

5. MIDIキーボード:
   認識されている
   Preferences > MIDI

6. 鍵盤弾く:
   音が鳴る?

鳴らない場合:

MIDIキーボード:
USB接続確認
電源On

Ableton:
MIDI Ports設定
Track: ☑

音源:
Wavetable等を挿入
```

---

## まとめ

### 録音の基本

```
□ Buffer Size: 128 (録音時)
□ レベル: ピーク -12 〜 -6 dB
□ Monitor: Auto (推奨)
□ Count In: 1-2 Bars
□ Metronome: On
```

### Audio録音

```
□ IF接続確認
□ +48V (コンデンサーマイク)
□ ゲイン調整
□ クリップさせない
□ 24bit録音
```

### MIDI録音

```
□ MIDIキーボード認識
□ 音源挿入
□ Monitor: In
□ Recording Quantize: 1/16
□ テイク保存
```

### 重要ポイント

```
□ レイテンシー対策
□ オーバーダビング活用
□ コンピングでベストテイク
□ Direct Monitoring検討
□ 複数テイク録音
```

---

**次は:** [Automation](./automation.md) - パラメーターを自動制御
