# 環境設定（Preferences）

Ableton Liveを最適化する。Preferences設定で制作効率とパフォーマンスを最大化します。

## この章で学ぶこと

- Preferences完全ガイド
- Look/Feel設定（見た目と操作感）
- File Folder設定（ファイル管理）
- Record Warp Launch設定（録音とWarp）
- CPU/RAM最適化
- キーボードショートカット
- ライセンスとLibrary

---

## なぜPreferences設定が重要なのか

**快適な制作環境:**

```
デフォルト設定:

万人向け:
平均的な設定

最適ではない:
あなたのPCに合っていない
あなたのワークフローに合っていない

最適化後:

高速:
CPU負荷軽減
サクサク動く

効率的:
ショートカット設定
作業時間短縮

安定:
クラッシュ防止

差:

設定前: 重い、不安定
設定後: 軽い、快適

時間:
5分の設定で
数百時間の効率化
```

---

## Preferencesを開く

**アクセス方法:**

```
Mac:
Live > Preferences
または
Cmd+, (カンマ)

Windows:
Options > Preferences
または
Ctrl+, (カンマ)

画面:

┌──────────────────────────┐
│ Preferences              │
├────────┬─────────────────┤
│ Look/  │ [設定内容]       │
│ Feel   │                 │
│ Audio  │                 │
│ Link/  │                 │
│ Tempo  │                 │
│ MIDI   │                 │
│ File/  │                 │
│ Folder │                 │
│ Librry │                 │
│ Record │                 │
│ ...    │                 │
└────────┴─────────────────┘

左: カテゴリ
右: 設定項目
```

---

## Look/Feel（見た目と操作感）

**最初に設定:**

### テーマ

```
Preferences > Look/Feel > Theme:

┌──────────────────┐
│ Dark (デフォルト) │ ← 推奨
│ Light            │
│ Mid-Dark         │
│ Mid-Light        │
└──────────────────┘

Dark:
目に優しい
長時間作業向き
スタジオの標準

Light:
明るい環境向き
昼間の作業

推奨:
Dark
→ プロの標準
→ 夜間作業に最適
```

### Brightness（明るさ）

```
スライダー:

Dark ←―●―――――――→ Bright

推奨:
中央やや暗め
→ 目が疲れない
```

### Language（言語）

```
言語選択:

English (推奨)
日本語

なぜEnglish:

チュートリアル:
99%が英語

ヘルプ:
英語版が充実

慣れ:
1ヶ月で慣れる

日本語:
初心者向き
でも後で英語に慣れる必要

妥協案:
最初: 日本語
慣れたら: English
```

### HiDPI Display Support (Mac)

```
Mac Retina ディスプレイ:

☑ HiDPI

効果:
画面が鮮明
文字が読みやすい

必須:
MacBook Pro/Air
iMac 4K/5K

Windowsの場合:
自動対応
設定不要
```

---

## File/Folder（ファイル管理）

**重要な設定:**

### Temporary Folder

```
Preferences > File/Folder > Temporary Folder:

デフォルト:
~/Library/Application Support/Ableton/Live/Temp

変更推奨:
外付けSSD
(システムドライブ以外)

理由:

録音ファイル:
一時的にここに保存
→ 大容量必要

SSD推奨:
高速アクセス

設定:

Browse...
→ 外付けSSD選択
例: /Volumes/Music SSD/Ableton Temp/
```

### Library

```
Preferences > File/Folder > Library:

User Library:
~/Music/Ableton/User Library/

ここに保存:
ダウンロードしたパック
自作プリセット
サンプル

変更:
外付けSSDに移動可能
(容量節約)

Install Packs:
パックのインストール先
```

### Plug-In Sources

```
VST Plugins:

Use VST Plug-In System Folders:
☑ On (推奨)

Custom Folder:
追加のVSTフォルダ指定可能

Mac:
~/Library/Audio/Plug-Ins/VST/
/Library/Audio/Plug-Ins/VST/

Windows:
C:\Program Files\VSTPlugins\

Re-scan:
新しいVST追加後
→ Re-scan ボタン
```

---

## Record Warp Launch

**録音とWarp設定:**

### Count-In

```
Preferences > Record Warp Launch > Count-In:

Count-In:
録音開始前のカウント

設定:

None: カウントなし
1 Bar: 1小節
2 Bars: 2小節 (推奨)
4 Bars: 4小節

推奨: 2 Bars
理由:
録音準備の時間
「1, 2, 3, 4, 1, 2...」

Metronome During Count-In:
☑ On
→ カウント中にクリック音
```

### Auto-Warp Long Samples

```
Auto-Warp Long Samples:
☑ On (推奨)

効果:

長いオーディオ:
自動でWarp
→ BPMに追従

短いサンプル:
Warpなし
(キック、スネア等)

境界:
デフォルト: 30秒以上

利点:
ボーカル、ループ
自動でテンポ合う
```

### Default Launch Mode

```
Launch Mode:
クリップの再生方法

Gate:
押している間だけ再生

Trigger:
クリックで再生開始 (推奨)

Toggle:
on/off切り替え

Repeat:
繰り返し再生

推奨: Trigger
理由:
一般的な使い方
```

---

## CPU/メモリ最適化

**パフォーマンス向上:**

### Multicore/Multiprocessor Support

```
Preferences > CPU > Multicore Support:

┌─────┐
│ On  │ ← 必須
└─────┘

効果:
複数CPUコア使用
→ 重いプロジェクトも快適

必須:
現代のPCはマルチコア
必ずOn

Offにする理由:
なし
```

### Buffer Size

```
Preferences > Audio > Buffer Size:

128 samples ← デフォルト
256 samples ← 推奨(制作時)
512 samples
1024 samples

設定:
次のセクション (audio-midi-setup.md) で詳細

簡単に:

小さい (64-128):
レイテンシー低い
→ リアルタイム演奏向き
→ CPU負荷高い

大きい (512-1024):
レイテンシー高い
→ ミックス向き
→ CPU負荷低い

切り替え:
録音時: 128
ミックス時: 512
```

### Freeze Tracks

```
トラックのFreeze:

右クリック > Freeze Track

効果:
重い音源・エフェクトを
一時的にオーディオ化
→ CPU負荷激減

いつ使う:
プロジェクトが重くなったら

解除:
Unfreeze
→ 再編集可能

自動:
Preferences で設定不要
手動でトラックごとに
```

### Reduce Latency When Monitoring

```
Preferences > Audio:

Reduce Latency When Monitoring:
☑ On

効果:
録音中のレイテンシー低減

必須:
ボーカル録音
ギター録音

不要:
MIDIキーボードのみ
```

---

## キーボードショートカット

**カスタマイズ:**

### ショートカット表示

```
Help > Show Keyboard Shortcuts

または:
Opt+Cmd+K (Mac)
Alt+Ctrl+K (Win)

画面:
キーボード全体の割り当て表示

便利:
よく使う機能を確認
```

### カスタマイズ方法

```
Preferences > MIDI > Key Map Mode:

Key ボタンクリック
→ オレンジ色に

割り当て:
1. 機能をクリック (例: Play)
2. キーを押す (例: F5)
→ 割り当て完了

削除:
機能クリック → Delete

保存:
Key ボタン再クリック
→ オレンジ解除

おすすめカスタマイズ:

F5: Play/Stop (デフォルトはSpace)
F9: Record (そのまま)
F12: Export (自分で設定)
```

---

## ライセンスとアカウント

**認証管理:**

### ライセンス認証

```
Preferences > Licenses:

Authorize with ableton.com:
Abletonアカウントでログイン

効果:
他のPCでも使用可能
(最大2台同時)

Deauthorize:
このPCの認証解除
→ 別PCで使う

オフライン認証:
インターネットなしで認証
(レアケース)
```

### User Account

```
ログイン:
Preferences > Account

Ableton Account:
メールアドレス
パスワード

できること:

パック追加:
購入したパックをDL

クラウド同期:
プリセット、設定

サポート:
フォーラム、ヘルプ
```

---

## その他の重要設定

### Auto Save

```
残念ながら:
Ableton Live に自動保存機能なし

対策:

手動保存:
Cmd+S 連打

Time Machine (Mac):
1時間ごとに自動バックアップ

File History (Win):
同様の機能

習慣:
5分ごとにCmd+S
```

### Undo Steps

```
Preferences > Edit:

Maximum Number of Undo Steps:

デフォルト: 64 (十分)
最大: 1000

推奨: 64
理由:
多すぎるとメモリ消費

Cmd+Z:
1つ戻る

Cmd+Shift+Z:
1つ進む
```

### Snap to Grid

```
Preferences > Record Warp Launch:

Global Quantization:

None
8 Bars
4 Bars
1 Bar (推奨)

効果:
クリップが小節頭にスナップ

便利:
きれいに整列

無効化:
Cmd (Mac) / Ctrl (Win)
押しながらドラッグ
→ 自由配置
```

---

## Audio設定の詳細

### Audio Device設定

**重要な項目:**

```
Preferences > Audio > Audio Device:

Driver Type:
Mac: CoreAudio (標準)
Windows: ASIO (必須)

Audio Device:
Built-in Output (内蔵)
オーディオインターフェイス名

選択基準:

内蔵:
練習、確認用
レイテンシー高い

オーディオI/F:
制作用 (推奨)
低レイテンシー
高音質

人気モデル:
Focusrite Scarlett
Universal Audio Apollo
RME Babyface
```

### Sample Rate

```
Sample Rate設定:

44.1 kHz ← CD品質
48 kHz ← 映像用
88.2 kHz
96 kHz ← ハイレゾ
192 kHz

推奨: 44.1 kHz

理由:

互換性:
すべてのシステムで再生可能

ファイルサイズ:
適度

CPU:
負荷が少ない

高いレートは:

利点:
わずかに音質向上 (体感困難)

欠点:
ファイル2倍サイズ
CPU負荷2倍

実用的:
44.1 kHz で十分
プロも標準で使用
```

### Input/Output Configuration

```
Preferences > Audio > Input/Output:

Input:
┌──────────────────┐
│ 1/2 (Stereo) ☑   │ ← マイク入力
│ 3/4 (Stereo) ☐   │
└──────────────────┘

Output:
┌──────────────────┐
│ 1/2 (Stereo) ☑   │ ← メイン出力
│ 3/4 (Stereo) ☐   │
└──────────────────┘

有効化:
使用するチャンネルのみ☑

理由:
不要なチャンネル = CPU無駄

例:
2in/2out I/F
→ 1/2のみ有効

8in/8out I/F
→ 必要なペアのみ
```

### Output Latency

```
Latency表示:

Overall Latency:
Input: 5.8 ms
Output: 11.6 ms

低い (良い):
0-10 ms
リアルタイム演奏可能

中程度:
10-20 ms
わずかに遅延感

高い (悪い):
20 ms以上
演奏困難

改善:
Buffer Size 下げる
→ Latency 下がる
→ CPU負荷 上がる

トレードオフ:
レイテンシー vs CPU
```

### Test Tone

```
Test機能:

Preferences > Audio > Test:

Test Tone:
クリックで音が鳴る

確認:
スピーカー接続
出力設定

トラブル時:
音が出ない場合
設定を再確認
```

---

## MIDI設定の詳細

### MIDI Ports

**入出力設定:**

```
Preferences > MIDI:

Input:
┌──────────────────┬──────┬──────┬──────┐
│ Device           │ Track│ Sync │Remote│
├──────────────────┼──────┼──────┼──────┤
│ MIDI Keyboard    │  ☑   │  ☐   │  ☐   │
│ Controller       │  ☑   │  ☐   │  ☑   │
└──────────────────┴──────┴──────┴──────┘

Track:
MIDI演奏入力
☑ 必須

Sync:
外部機器と同期
☑ シンセと同期時

Remote:
コントローラー操作
☑ MIDI learn使用時

Output:
同様の設定
```

### MIDI Clock Sync

```
MIDI Clock設定:

Sync Type:
None (デフォルト)
Song Position
Pattern

使用例:

外部ドラムマシン:
Song Position
→ Liveと同期

モジュラーシンセ:
Pattern
→ テンポ同期

不要:
MIDIキーボードのみ
→ None
```

### Takeover Mode

```
Preferences > MIDI > Takeover Mode:

コントローラーノブと
Live内パラメータの
値が違う時の動作

None:
すぐに反映
→ 値が飛ぶ

Pickup:
物理ノブが追いつくまで待つ (推奨)
→ スムーズ

Value Scaling:
相対的に変化

推奨: Pickup
理由:
急な音量変化を防ぐ
```

### MIDI Remote Scripts

```
Control Surface:

Preferences > MIDI > Control Surface:

プリセット:
Ableton Push 2
Akai APC40
Native Instruments Maschine
Novation Launchpad
...

選択:
持っているコントローラー選択
→ 自動マッピング

None:
汎用MIDIコントローラー
→ 手動MIDI Learn
```

---

## Link/Tempo/MIDI設定

### Ableton Link

**デバイス同期:**

```
Preferences > Link Tempo MIDI:

Ableton Link:
☑ Enable Link

機能:
WiFi経由で
複数デバイス同期

使用例:

Live + Live:
2台のPC同期

Live + iOS:
iPhone/iPad アプリと同期

アプリ例:
Reason
Beatmaker
Launchpad

利点:
ケーブル不要
即座に同期

条件:
同一WiFiネットワーク
```

### Tempo設定

```
Default Tempo:

New Set Default Tempo:
120 BPM (デフォルト)

変更例:
House: 125 BPM
Techno: 130 BPM
Dubstep: 140 BPM
Hip-Hop: 90 BPM

効果:
新規プロジェクト作成時
この値が初期BPM

便利:
よく作るジャンルの
標準BPM設定
```

### Follow Song Tempo

```
External Sync:

Sync:
☐ Off (通常)
☑ On (同期時)

使用例:
外部機器がマスター
→ Liveがスレーブ

通常:
Off
→ Live内部クロック使用
```

---

## Library（ライブラリ管理）

### Pack設定

**パック管理:**

```
Preferences > Library:

Install Packs:
Abletonパックのインストール

場所:
~/Music/Ableton/Factory Packs/
~/Music/Ableton/User Library/

Installed Packs:
一覧表示

アンインストール:
不要なパック削除
→ 容量節約

推奨:
よく使うパックのみ保持
```

### User Library整理

```
User Library構造:

User Library/
├── Presets/
│   ├── Instruments/
│   └── Audio Effects/
├── Samples/
├── Clips/
├── Grooves/
└── Templates/

管理:

定期的に整理
不要ファイル削除
カテゴリ分け

Tips:
フォルダ名は英語
検索しやすい名前
```

### サンプルスキャン

```
Sample Scan設定:

Preferences > Library > Rescan:

Use Cases:
新しいサンプル追加後
外部ドライブ接続後

処理:
全フォルダスキャン
→ 時間かかる

頻度:
必要時のみ
自動スキャンなし
```

---

## 実践: Preferences最適化

**30分の設定:**

### Step 1: Look/Feel (5分)

```
1. Preferences開く: Cmd+,

2. Look/Feel:
   Theme: Dark
   Brightness: 中央やや暗め
   Language: English (または日本語)

3. Mac: HiDPI ☑

4. 閉じる
```

### Step 2: File/Folder (10分)

```
1. File/Folder:

2. Temporary Folder:
   Browse...
   → 外付けSSD選択(あれば)
   なければデフォルトのまま

3. Library確認:
   User Library の場所メモ

4. Plug-In Sources:
   Use VST: ☑
   Re-scan (VSTインストール済みの場合)
```

### Step 3: Record Warp Launch (5分)

```
1. Record Warp Launch:

2. Count-In: 2 Bars
   Metronome During Count-In: ☑

3. Auto-Warp Long Samples: ☑

4. Default Launch Mode: Trigger
```

### Step 4: CPU (5分)

```
1. CPU:

2. Multicore Support: On

3. Audio:
   Buffer Size: 256 samples
   Reduce Latency: ☑
```

### Step 5: ショートカット確認 (5分)

```
1. Help > Show Keyboard Shortcuts
   または Opt+Cmd+K

2. よく使う機能確認:
   Space: Play/Stop
   F9: Record
   Cmd+S: Save
   Cmd+Z: Undo

3. 覚える
```

---

## よくある質問

### Q1: Preferencesが反映されない

**A:** Ableton再起動

```
設定変更後:

一部設定:
すぐ反映

Audio/CPU設定:
再起動必要

手順:
1. Preferences設定
2. File > Quit (Cmd+Q)
3. 再起動
4. 確認
```

### Q2: デフォルトに戻したい

**A:** Preferences削除

```
Mac:
~/Library/Preferences/Ableton/Live x.x/
→ フォルダ削除
→ 再起動

Windows:
C:\Users\[ユーザー名]\AppData\Roaming\Ableton\Live x.x\Preferences\
→ フォルダ削除
→ 再起動

注意:
全設定がリセット
ライセンス再認証必要

やる前に:
フォルダをバックアップ
```

### Q3: 他のPCに設定を移行したい

**A:** Preferencesフォルダコピー

```
Export:

元PC:
~/Library/Preferences/Ableton/Live x.x/
→ フォルダをUSBにコピー

Import:

新PC:
同じ場所にフォルダ貼り付け
→ 再起動

注意:
Liveバージョン一致必要
(11.x → 11.x)

ライセンス:
別途認証必要
```

---

## まとめ

### 必須設定

```
Look/Feel:
□ Theme: Dark
□ Language: English (または日本語)

File/Folder:
□ Temporary Folder設定
□ Plug-In Sources確認

Record Warp Launch:
□ Count-In: 2 Bars
□ Auto-Warp: On
□ Launch Mode: Trigger

CPU:
□ Multicore: On
□ Buffer Size: 256
□ Reduce Latency: On
```

### 設定完了後

```
1. Ableton再起動
2. 動作確認
3. プロジェクト作成して試す
4. 快適になったか確認
```

### ショートカット

```
Preferences: Cmd+,
Keyboard Map: Opt+Cmd+K
```

---

## パフォーマンス最適化の実践テクニック

### CPU使用率モニタリング

**リソース監視:**

```
CPU Meter:
Ableton Live右上
常時表示

表示内容:
┌──────────────┐
│ CPU: 45%     │ ← 現在の使用率
│ DISK: 0.2%   │ ← ディスク負荷
└──────────────┘

目安:

緑 (0-60%):
余裕あり
快適

黄 (60-80%):
注意
重い処理で不安定

赤 (80-100%):
危険
音切れの可能性

対策:

Buffer Size上げる:
256 → 512
→ CPU負荷軽減

Freeze Track:
重いトラックを一時オーディオ化

プラグイン削減:
不要なエフェクト削除
```

### ディスク使用率の最適化

```
DISK Meter:

表示:
録音・再生時のディスク負荷

高い場合 (>50%):

原因:
遅いHDD
同時に多数のオーディオ再生

対策:

SSDに移行:
HDD → SSD
→ 劇的改善

トラック数削減:
使用していないトラック削除

サンプルレート下げる:
96kHz → 44.1kHz
→ ファイルサイズ半分
```

### メモリ管理

```
RAM使用:

Live 12推奨:
最小: 8GB
推奨: 16GB
理想: 32GB以上

確認:
Activity Monitor (Mac)
Task Manager (Win)

節約:

不要なアプリ終了:
ブラウザ、Slack等

サンプルの読み込み:
必要な分だけ
```

---

## 環境別最適設定

### ノートPC設定（MacBook等）

**モバイル制作環境:**

```
バッテリー駆動時:

Buffer Size: 512
→ CPU負荷軽減
→ 電池長持ち

Sample Rate: 44.1kHz
→ 処理軽量化

CPU Throttling:
省電力モードOFF
→ パフォーマンス優先

電源接続時:

Buffer Size: 256
→ 低レイテンシー

Multicore: On
→ フルパワー

冷却:
ノートPC スタンド使用
→ 熱対策
```

### デスクトップPC設定

**本格制作環境:**

```
ハイスペックPC:

Buffer Size: 128-256
→ 最低レイテンシー

Sample Rate: 44.1kHz
(または48kHz)

CPU:
Multicore: On
全コア活用

複数モニター:
メインモニター: Arrangement View
サブモニター: Mixer、Browser

推奨スペック:
CPU: i7/i9、Ryzen 7/9
RAM: 32GB以上
SSD: 512GB以上
```

### スタジオ固定設置

**プロ環境:**

```
専用オーディオI/F:

高品質モデル:
Universal Audio Apollo
RME Fireface
Apogee Symphony

設定:
Sample Rate: 44.1kHz
Buffer Size: 256
→ 安定性重視

外部機器連携:

MIDI Sync: On
Ableton Link: On
→ 全機器同期

モニター環境:
スタジオモニター
ヘッドホンアンプ
```

---

## トラブルシューティング

### 音が出ない場合

**チェックリスト:**

```
1. Audio Device確認:
   Preferences > Audio
   → 正しいデバイス選択

2. Output有効化:
   Preferences > Audio > Output
   → チャンネル ☑

3. Master Volume:
   Mixerのマスタートラック
   → 0dB付近

4. Test Tone:
   Preferences > Audio > Test
   → クリックして音確認

5. OS設定:
   システム環境設定 > サウンド
   → 出力デバイス確認

6. ケーブル:
   物理接続確認
```

### 音切れ・ノイズ

**原因と対策:**

```
症状:
プチプチ音
音飛び

原因1: Buffer Size小さい
対策:
256 → 512
または 1024

原因2: CPU過負荷
対策:
Freeze Track
プラグイン削減

原因3: ディスク遅い
対策:
SSDに移行
外付けHDD → 内蔵SSD

原因4: WiFi干渉
対策:
有線接続
WiFi OFF

原因5: バックグラウンドアプリ
対策:
不要アプリ終了
```

### レイテンシーが大きい

**低レイテンシー化:**

```
現在: 20ms以上
目標: 10ms以下

手順:

1. Buffer Size下げる:
   512 → 256 → 128

2. Sample Rate確認:
   44.1kHz推奨
   (高いとレイテンシー増)

3. Reduce Latency: On
   Preferences > Audio

4. Direct Monitoring:
   オーディオI/F側で
   入力を直接モニター
   → ゼロレイテンシー

注意:
Buffer下げすぎ → CPU過負荷
バランス調整必要
```

### MIDI機器が認識されない

**接続トラブル:**

```
確認手順:

1. 物理接続:
   USBケーブル再接続
   MIDIケーブル確認

2. デバイス電源:
   MIDI機器の電源ON

3. Live再起動:
   Ableton終了 → 再起動

4. MIDI設定:
   Preferences > MIDI
   → Track ☑

5. OSレベル:
   Audio MIDI Setup (Mac)
   → デバイス表示確認

6. ドライバ:
   メーカーサイト
   → 最新ドライバDL

7. USB hub:
   直接PC接続
   (hub経由 → 不安定)
```

---

## 高度な設定テクニック

### プロジェクトテンプレート作成

**効率化の極意:**

```
目的:
毎回同じ設定
→ 時間節約

作成手順:

1. 新規プロジェクト:
   File > New Live Set

2. 基本構成:
   よく使うトラック配置
   - Kick (Audio)
   - Bass (MIDI)
   - Lead (MIDI)
   - Return Tracks (Reverb, Delay)

3. デフォルトエフェクト:
   各トラックにEQ、Compressor等
   プリセット設定

4. BPM設定:
   よく作るジャンルの標準BPM

5. 保存:
   File > Save Live Set as Template
   → Templates/MyTemplate.als

使用:
次回から
File > New Live Set > MyTemplate
```

### カスタムキーマップ

**ワークフロー加速:**

```
効率的なショートカット:

再生/停止:
Space (デフォルト)
またはF5 (カスタム)

録音:
F9 (推奨そのまま)

新規MIDIトラック:
Cmd+Shift+T (Mac)
Ctrl+Shift+T (Win)

新規Audioトラック:
Cmd+T
Ctrl+T

トラック削除:
Cmd+Delete
Ctrl+Delete

カスタム例:

F1: Metronome On/Off
F2: Loop On/Off
F3: Session View
F4: Arrangement View
F12: Export Audio

設定:
Preferences > MIDI
→ Key Map Mode
→ 割り当て
```

### マルチアウト設定

**複数出力活用:**

```
用途:
各トラックを別々の出力
→ 外部ミキサーで処理

設定手順:

1. Audio I/F:
   8out以上推奨

2. Preferences > Audio:
   Output 3/4, 5/6, 7/8 ☑

3. トラック設定:
   Audio To: External Out
   → 3/4, 5/6等選択

使用例:

キック → Out 1/2
ベース → Out 3/4
リード → Out 5/6
ドラム → Out 7/8

メリット:
外部ハードウェアEQ
個別コンプレッサー
アナログミキサー
```

---

## バックアップとデータ管理

### プロジェクトバックアップ

**データ保護:**

```
重要度: 最高
失うと: 作業すべて消失

バックアップ方法:

1. Collect All and Save:
   File > Collect All and Save
   → 全サンプル・プラグイン設定保存

2. 自動バックアップ:
   Time Machine (Mac)
   File History (Win)
   → 1時間ごと

3. クラウド:
   Dropbox、Google Drive
   → プロジェクトフォルダ同期

4. 外付けドライブ:
   週1回手動バックアップ
   → 完全コピー

5. バージョン管理:
   ProjectName_v1.als
   ProjectName_v2.als
   → 重要な段階で保存
```

### サンプルライブラリ整理

**効率的な管理:**

```
フォルダ構造:

~/Music/Samples/
├── Kicks/
│   ├── 808/
│   ├── Acoustic/
│   └── Electronic/
├── Snares/
├── Percussion/
├── Vocals/
│   ├── Phrases/
│   └── Oneshots/
├── FX/
└── Loops/
    ├── Drum Loops/
    ├── Bass Loops/
    └── Melodic Loops/

命名規則:
Genre_Type_Note_BPM.wav

例:
Techno_Kick_C_128.wav
House_Bass_Am_124.wav

メタデータ:
BPM、Key情報
→ Live内検索で便利
```

### プリセット管理

**即座にアクセス:**

```
User Library活用:

保存場所:
~/Music/Ableton/User Library/Presets/

分類:

Instruments/
├── My Bass Sounds/
├── Lead Synths/
├── Pads/
└── Drums/

Audio Effects/
├── Vocal Chain/
├── Mastering/
└── Creative/

使い方:

1. 作成:
   音作り完成
   → Save Preset

2. 名前:
   わかりやすく
   "Fat Techno Bass"
   "Vocal Compression Chain"

3. タグ:
   検索用
   "bass, techno, 303"

4. 呼び出し:
   Browser > User Library
   → ドラッグ&ドロップ
```

---

## セキュリティとライセンス管理

### ライセンス保護

**認証情報の管理:**

```
重要:
シリアルナンバー保存

保管場所:

1. Abletonアカウント:
   オンラインで確認可能

2. メール:
   購入時の確認メール保存

3. 紙メモ:
   物理的に保管
   金庫等

4. パスワードマネージャー:
   1Password、LastPass

認証台数:
最大2台同時
→ 3台目使用時は
   1台をDeauthorize

再インストール時:
シリアル入力
→ すぐ使用可能
```

### プラグインライセンス

**サードパーティ管理:**

```
VST/AU プラグイン:

iLok:
多くのプロプラグイン
物理キーまたはクラウド

Waves Central:
Wavesプラグイン専用

Native Access:
Native Instruments

Plugin Alliance:
Installation Manager

重要:
各アカウント情報保存
シリアル番号記録

ライセンス移行:
古いPC → 新PC
各ツールでDeactivate/Activate
```

---

## まとめ

### 必須設定チェックリスト

```
基本設定:
□ Look/Feel: Dark
□ Language: English (または日本語)
□ HiDPI: On (Mac Retina)

オーディオ:
□ Audio Device: オーディオI/F選択
□ Sample Rate: 44.1kHz
□ Buffer Size: 256 (録音時128)
□ Reduce Latency: On
□ Input/Output: 使用チャンネルのみ☑

MIDI:
□ MIDI Ports: Track ☑
□ Control Surface: コントローラー選択
□ Takeover Mode: Pickup

ファイル:
□ Temporary Folder: 外付けSSD (推奨)
□ User Library: 場所確認
□ VST Sources: Use System Folders ☑

制作環境:
□ Count-In: 2 Bars
□ Auto-Warp: On
□ Launch Mode: Trigger
□ Multicore: On
□ Undo Steps: 64

バックアップ:
□ Time Machine/File History 有効
□ Collect All and Save 定期実行
□ シリアルナンバー保存
```

### 設定後のアクション

```
1. Ableton再起動:
   設定を確実に反映

2. テストプロジェクト:
   音が出るか確認
   レイテンシー確認
   MIDI動作確認

3. テンプレート作成:
   基本トラック配置
   Save as Template

4. ショートカット習得:
   Opt+Cmd+K で表示
   よく使う機能を暗記

5. バックアップ設定:
   自動バックアップ確認
   外付けドライブ準備
```

### パフォーマンス指標

```
目標値:

CPU使用率: 60%以下
レイテンシー: 10ms以下
ディスク負荷: 30%以下

達成できたら:
快適な制作環境
音切れなし
リアルタイム演奏可能
```

---

## ジャンル別推奨設定

### Techno/House制作

**ダンスミュージック最適化:**

```
BPM設定:
Default Tempo: 125 BPM (House)
または 130 BPM (Techno)

Warp設定:
Auto-Warp: On
→ ループサンプル自動同期

Launch Mode:
Trigger
→ ライブパフォーマンス向き

Quantization:
1 Bar
→ きれいにループ

Buffer Size:
録音時: 128
→ ドラムマシン入力
ミックス時: 512
→ 多数トラック処理

推奨プラグイン:
Wavetable (シンセ)
Echo (ディレイ)
Reverb (空間系)
EQ Eight (イコライザー)
Compressor (ダイナミクス)

トラックテンプレート:
- Kick (Audio)
- Bass (MIDI - Wavetable)
- Lead (MIDI - Wavetable)
- Perc 1, 2, 3 (Audio)
- FX (Audio)
- Return A: Reverb
- Return B: Delay
```

### Hip-Hop/Trap制作

**ビートメイキング設定:**

```
BPM設定:
Default Tempo: 90 BPM (Hip-Hop)
または 140 BPM (Trap)

Warp設定:
Auto-Warp: On
→ サンプルチョップ用

Count-In:
1 Bar
→ 素早くビート録音

Snap to Grid:
1/16 Note
→ 細かいハイハット配置

Buffer Size:
256 samples
→ MPC風の打ち込み

推奨プラグイン:
Simpler (サンプラー)
Drum Rack
Saturator (歪み)
EQ Eight
Glue Compressor

トラックテンプレート:
- Kick
- Snare
- Hi-Hat
- 808 Bass (MIDI)
- Sample 1, 2, 3
- Vocal
- Return A: Reverb Short
- Return B: Delay 1/4
```

### EDM/Pop制作

**商業音楽設定:**

```
BPM設定:
Default Tempo: 128 BPM (EDM)
または 120 BPM (Pop)

Sample Rate:
48 kHz
→ 映像同期用

Warp設定:
Auto-Warp: On
→ ボーカル処理

Buffer Size:
256 samples
→ バランス型

推奨プラグイン:
Wavetable
Operator (FM)
Auto Filter
Vocoder
Multiband Dynamics

トラックテンプレート:
- Kick
- Snare/Clap
- Bass (Sub + Mid)
- Lead Synth
- Pad
- Vocal (複数)
- FX/Riser
- Return A: Reverb Large
- Return B: Delay Sync
- Return C: Chorus
```

### Ambient/Experimental

**実験音楽設定:**

```
BPM設定:
Default Tempo: 80 BPM
または Free Tempo

Warp設定:
Auto-Warp: Off
→ 自然な時間軸

Quantization:
None
→ 自由配置

Buffer Size:
512-1024
→ 重いエフェクト多用

推奨プラグイン:
Granulator
Erosion
Resonators
Corpus
Reverb (100% Wet)

トラックテンプレート:
- Texture 1, 2, 3
- Field Recording
- Drone
- Processed Audio
- Return A: Reverb Infinite
- Return B: Grain Delay
- Return C: Frequency Shifter
```

---

## コラボレーション設定

### 複数人での制作

**プロジェクト共有:**

```
ファイル管理:

Collect All and Save:
必須
→ 全サンプル同梱

クラウド共有:
Dropbox, Google Drive
→ 自動同期

バージョン管理:
Track_v1_John.als
Track_v2_Mike.als
→ 誰が編集か明記

プラグイン統一:
使用プラグインリスト作成
→ 全員が同じもの所有

Sample Rate統一:
44.1 kHz (推奨)
→ 互換性最優先

テンプレート共有:
共通テンプレート使用
→ トラック配置統一
```

### リモートコラボレーション

**遠隔制作:**

```
Ableton Link:
Enable Link ☑
→ 同時演奏セッション

オンライン通話:
Zoom, Discord
→ 音声共有

画面共有:
編集作業を共有
→ リアルタイムフィードバック

ファイル転送:
WeTransfer (大容量)
→ プロジェクトファイル送信

Stems Export:
個別トラック書き出し
→ 別々にミックス可能

コミュニケーション:
定期的に進捗共有
コメント機能活用
```

---

## アップグレードとバージョン管理

### Live 12の新機能活用

**最新バージョン設定:**

```
Meld:
新シンセサイザー
Preferences > Plug-In Sources
→ 自動認識

Drift:
アナログシンセ
プリセットブラウザで検索

MIDI Polyphonic Expression (MPE):
Preferences > MIDI
→ MPE対応コントローラー設定

Spectral Resonator:
CPU負荷高い
→ Buffer Size 512推奨

Hybrid Reverb:
高品質リバーブ
→ Returnトラックに配置

Roar:
ディストーション
→ ドラムに最適
```

### 旧バージョンからの移行

**アップグレード手順:**

```
Live 11 → Live 12:

1. 現行プロジェクトバックアップ:
   全プロジェクトコピー
   → 外付けドライブ

2. Live 12インストール:
   Live 11と共存可能

3. Preferences移行:
   Live 11のPreferencesフォルダコピー
   → Live 12フォルダに貼り付け

4. ライセンス認証:
   Ableton Account でログイン

5. プラグイン再スキャン:
   Preferences > Plug-In Sources
   → Re-scan

6. 互換性確認:
   古いプロジェクト開く
   → 動作チェック

注意点:

Max for Live:
再インストール必要な場合あり

サードパーティVST:
最新版に更新推奨

プリセット:
User Libraryは自動移行
```

---

## モバイル連携設定

### iPad/iPhone連携

**iOS デバイス活用:**

```
Ableton Note (iOS):
アイデアスケッチアプリ

連携:
Ableton Cloudで同期
→ iPad作成メロディを
   Live で開く

設定:
Preferences > Account
→ ログイン
→ Cloud Sync ☑

使い方:
1. iPad で Note 起動
2. メロディ/ビート作成
3. Save to Cloud
4. Live で Open from Cloud

Ableton Link:
同一WiFi
→ Live + Note 同期演奏

推奨:
外出先でアイデア録り
→ 帰宅後Live で本格制作
```

### タブレットコントロール

**タッチスクリーン活用:**

```
TouchOSC (iOS/Android):
カスタムMIDIコントローラー

設定:

1. TouchOSC アプリDL:
   iOS App Store
   または Google Play

2. WiFi接続:
   PC と同一ネットワーク

3. Live 設定:
   Preferences > MIDI
   → TouchOSC (Network)
   Track ☑, Remote ☑

4. レイアウト作成:
   TouchOSC Editor
   → フェーダー、ボタン配置

5. MIDI Learn:
   Live で MIDI Map Mode
   → TouchOSC 操作で割り当て

使用例:
ミキサーコントロール
エフェクトパラメータ
クリップ起動
```

---

## プロフェッショナルワークフロー

### スタジオ標準設定

**プロ環境構築:**

```
オーディオI/F:
Universal Audio Apollo
RME Fireface UCX II
Apogee Symphony

設定:
Sample Rate: 44.1kHz
Buffer Size: 256 (録音128)
Bit Depth: 24-bit

モニタリング:
メインモニター: Genelec, Focal
サブウーファー: 低域確認
ヘッドホン: 複数種類
→ 異なる環境でチェック

MIDI機器:
MIDIキーボード: 88鍵
パッドコントローラー: Push 2
制御: MIDI Fighter Twister

ルーム処理:
吸音材設置
バスストラップ
モニター位置最適化
```

### マスタリングエンジニア納品設定

**プロ納品準備:**

```
Export設定:

Sample Rate:
44.1kHz (CD)
48kHz (映像)
96kHz (ハイレゾ)

Bit Depth:
24-bit (マスタリング用)
16-bit (CD最終)

Format:
WAV (推奨)
AIFF (Mac互換)

Dither:
POW-r 3
→ 24bit→16bit変換時

Normalization:
Off
→ マスタリングで調整

ファイル命名:
ArtistName_TrackTitle_Master_24bit_44k.wav

Stems Export:
必要に応じて
各トラックグループ別
```

---

## 緊急トラブル対処法

### クラッシュ復旧

**データ救済:**

```
Live がクラッシュした:

1. 再起動:
   Live を再起動

2. 自動回復:
   File > Open Recent
   → [Recovered] 付きファイル

3. Tempフォルダ確認:
   ~/Library/Application Support/Ableton/Live/Temp
   → .als.tmp ファイル探す

4. Time Machine:
   1時間前の状態に復元

5. 手動バックアップ:
   外付けドライブから復元

予防:
5分ごとCmd+S
定期的にバージョン保存
```

### 音が歪む・割れる

**音質トラブル:**

```
症状:
音が歪む、割れる

原因と対策:

1. クリッピング:
   Master Meter 確認
   → 赤点灯 = Over
   対策: 全トラック音量下げ

2. CPU過負荷:
   Buffer Size上げる
   Freeze Track

3. プラグイン不具合:
   1つずつバイパス
   → 原因特定

4. サンプルレート不一致:
   全サンプル44.1kHzに統一

5. オーディオI/F設定:
   ドライバ最新版に更新
   再起動
```

### ライセンス認証エラー

**認証問題:**

```
エラーメッセージ:
"Authorization failed"

対処:

1. インターネット接続確認:
   WiFi/有線確認

2. Abletonアカウント確認:
   https://www.ableton.com
   → ログイン
   → ライセンス表示確認

3. 認証台数確認:
   2台以上で使用中?
   → 1台Deauthorize

4. オフライン認証:
   Preferences > Licenses
   → Authorize Offline
   → 画面指示に従う

5. サポート連絡:
   https://www.ableton.com/support
   → シリアル番号準備
```

---

## 学習リソースと次のステップ

### 公式リソース

**Ableton提供:**

```
Ableton Manual:
https://www.ableton.com/manual
→ 全機能詳細解説

Learning Music:
https://learningmusic.ableton.com
→ 音楽理論基礎

Learning Synths:
https://learningsynths.ableton.com
→ シンセ基礎

One Thing:
YouTube Series
→ 週1回のTips動画

Certified Training:
有料オンラインコース
→ 体系的学習
```

### コミュニティ

**ユーザー交流:**

```
Reddit:
r/ableton
→ 質問、Tips共有

Facebook Groups:
Ableton Live Users
→ グローバルコミュニティ

Discord:
Ableton非公式サーバー
→ リアルタイム交流

フォーラム:
https://forum.ableton.com
→ 技術的質問

YouTube:
You Suck at Producing
Seed to Stage
→ 実践的チュートリアル
```

### スキルアップパス

**段階的成長:**

```
初級 (0-3ヶ月):
□ Preferencesマスター
□ 基本操作習得
□ 簡単なビート作成
□ オーディオ録音

中級 (3-12ヶ月):
□ 複雑なアレンジ
□ ミキシング基礎
□ プラグイン活用
□ サンプリング技術

上級 (1-2年):
□ マスタリング
□ ライブパフォーマンス
□ Max for Live
□ プロレベル作品完成

プロ (2年以上):
□ ジャンル横断制作
□ コラボレーション
□ リリース経験
□ 継続的スキル向上
```

---

## 最終チェックリスト

### 初回セットアップ完全版

**全項目確認:**

```
□ 1. ソフトウェア:
  □ Live 12 インストール
  □ ライセンス認証完了
  □ バージョン確認

□ 2. Look/Feel:
  □ Theme: Dark
  □ Brightness: 調整
  □ Language: 選択
  □ HiDPI: On (Mac)

□ 3. Audio:
  □ Audio Device: 選択
  □ Sample Rate: 44.1kHz
  □ Buffer Size: 256
  □ Reduce Latency: On
  □ Input/Output: 有効化
  □ Test Tone: 確認

□ 4. MIDI:
  □ Ports: Track ☑
  □ Control Surface: 設定
  □ Takeover Mode: Pickup
  □ MIDI機器動作確認

□ 5. File/Folder:
  □ Temporary Folder: 設定
  □ User Library: 確認
  □ VST Sources: ☑
  □ Re-scan実行

□ 6. Record Warp Launch:
  □ Count-In: 2 Bars
  □ Metronome: On
  □ Auto-Warp: On
  □ Launch Mode: Trigger
  □ Quantization: 1 Bar

□ 7. CPU最適化:
  □ Multicore: On
  □ CPU Meter: 表示確認

□ 8. Library:
  □ Packs インストール
  □ サンプル整理
  □ プリセット確認

□ 9. ショートカット:
  □ Keyboard Map確認
  □ カスタム設定
  □ 主要操作暗記

□ 10. バックアップ:
  □ Time Machine/File History
  □ クラウド同期
  □ 外付けドライブ

□ 11. テスト:
  □ 新規プロジェクト作成
  □ オーディオ録音
  □ MIDI入力
  □ 再生確認
  □ Export確認

□ 12. ドキュメント:
  □ 設定メモ保存
  □ シリアル番号記録
  □ プラグインリスト作成
```

### 定期メンテナンス

**月次チェック:**

```
□ ソフトウェア更新:
  □ Live アップデート確認
  □ プラグイン更新
  □ OSアップデート (慎重に)

□ バックアップ:
  □ プロジェクトバックアップ
  □ User Library バックアップ
  □ Preferences バックアップ

□ ライブラリ整理:
  □ 不要ファイル削除
  □ サンプル整理
  □ プリセット整理

□ パフォーマンス:
  □ CPU使用率確認
  □ ディスク空き容量確認
  □ 動作速度確認

□ ライセンス:
  □ 認証状態確認
  □ プラグインライセンス確認
```

---

## 結論: 最適化された制作環境

**成功への道:**

```
Preferences設定 = 基盤:

正しい設定:
効率的なワークフロー
ストレスフリー制作
プロ品質の作品

時間投資:
初回: 30分-1時間
効果: 数百時間の節約

継続的改善:
使いながら調整
自分に合った設定発見

次のステップ:
Audio/MIDI詳細設定
プラグイン習得
実際の制作開始

重要:
設定は手段
目的は音楽創造
```

**これで準備完了。**
**さあ、音楽を作りましょう。**

---

**次は:** [Audio/MIDI設定](./audio-midi-setup.md) - オーディオインターフェイス設定
